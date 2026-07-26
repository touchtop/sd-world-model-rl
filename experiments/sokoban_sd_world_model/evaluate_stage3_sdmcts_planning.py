#!/usr/bin/env python3
"""Closed-loop planning smoke for Sokoban SD-MCTS providers.

This is the first planning evaluation after the stage-2 provider interface
audit. It keeps the experiment deliberately small and auditable: each episode
uses receding-horizon planning from the current observed raw state, while the
MCTS internal abstract transitions are supplied by the selected provider.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Protocol, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import (
    LearnedStockFlowTransitionProvider,
    ProviderPrediction,
    action_to_dict,
    puct_score,
    select_levels,
    stock_reward,
    stock_value,
)
from sokoban_learned_action_prior import (
    load_action_prior_checkpoint,
    predict_action_prior_logit,
)
from sokoban_tva import load_tva_checkpoint, predict_tva_value, relative_path
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState
from src.sokoban.env import ACTION_NAMES, DIRECTIONS
from src.sokoban.search import best_first_search
from src.sokoban.sd_model import SokobanPushAction
from validate_sokoban_sd_design import FLOW_NAMES, STOCK_NAMES, diagnostic_flows, stock_proxy


class TransitionProvider(Protocol):
    provider_id: str

    def predict(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        action: SokobanPushAction,
    ) -> ProviderPrediction:
        ...


class TerminalValueEvaluator(Protocol):
    evaluator_id: str

    def should_stop(self, model: SokobanGoalDirectedSDModel, raw_state: SokobanState) -> bool:
        ...

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
    ) -> float:
        ...


class PlannerFeedback(Protocol):
    feedback_id: str

    def reward(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock_before: np.ndarray,
        next_raw_state: SokobanState,
        stock_after: np.ndarray,
    ) -> float:
        ...

    def action_priors(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        legal_actions: Sequence[SokobanPushAction],
    ) -> Dict[SokobanPushAction, float]:
        ...


@dataclass(frozen=True)
class PlannerAction:
    """A primitive push or a short high-level option made of primitive pushes."""

    primitives: tuple[SokobanPushAction, ...]
    source: str = field(default="primitive", compare=False, hash=False)
    structural_score: float = field(default=0.0, compare=False, hash=False)

    @property
    def duration(self) -> int:
        return len(self.primitives)

    @property
    def first(self) -> SokobanPushAction:
        return self.primitives[0]

    @property
    def flat_index_tuple(self) -> tuple[int, ...]:
        return tuple(action.flat_index for action in self.primitives)


class OracleAbstractTransitionProvider:
    """Exact abstract stock transition for upper-bound diagnostics."""

    provider_id = "oracle_abstract_stock_transition_for_diagnostic_upper_bound"

    def predict(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        action: SokobanPushAction,
    ) -> ProviderPrediction:
        next_state = model.step_state(raw_state, action)
        next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
        stock_before = np.asarray(stock, dtype=np.float64)
        delta = next_stock - stock_before
        flows = diagnostic_flows(delta).astype(np.float64)
        return ProviderPrediction(
            stock_before=stock_before,
            flows=flows,
            raw_next_stock=next_stock,
            next_stock=next_stock,
            clip_correction=np.zeros_like(next_stock),
            provider_id=self.provider_id,
        )


class ZeroFlowTransitionProvider:
    """No-learned-transition ablation with a fixed zero-flow world model."""

    provider_id = "zero_flow_no_learned_transition_ablation"

    def predict(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        action: SokobanPushAction,
    ) -> ProviderPrediction:
        stock_before = np.asarray(stock, dtype=np.float64)
        flows = np.zeros(len(FLOW_NAMES), dtype=np.float64)
        return ProviderPrediction(
            stock_before=stock_before,
            flows=flows,
            raw_next_stock=stock_before.copy(),
            next_stock=stock_before.copy(),
            clip_correction=np.zeros_like(stock_before),
            provider_id=self.provider_id,
        )


class StockTerminalValueEvaluator:
    """Original fixed stock readout used before TVA."""

    evaluator_id = "fixed_stock_value_leaf_readout"

    def should_stop(self, model: SokobanGoalDirectedSDModel, raw_state: SokobanState) -> bool:
        return False

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
    ) -> float:
        return stock_value(stock)


class ZeroTerminalValueEvaluator:
    """MCTS-zero style terminal value ablation."""

    evaluator_id = "zero_terminal_value_ablation"

    def should_stop(self, model: SokobanGoalDirectedSDModel, raw_state: SokobanState) -> bool:
        return False

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
    ) -> float:
        return 0.0


class SokobanTvaTerminalValueEvaluator:
    """Checkpoint-backed TVA leaf evaluator."""

    evaluator_id = "sokoban_tva_value_v1"

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.payload = load_tva_checkpoint(checkpoint_path)
        self.evaluator_id = str(
            self.payload.get("checkpoint_contract", {}).get("value_provider_id", self.evaluator_id)
        )

    def should_stop(self, model: SokobanGoalDirectedSDModel, raw_state: SokobanState) -> bool:
        return model.is_solved(raw_state) or not model.legal_pushes(raw_state)

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
    ) -> float:
        if model.is_solved(raw_state):
            return 10.0
        if model.is_deadlocked(raw_state) or not model.legal_pushes(raw_state):
            return -10.0
        return predict_tva_value(self.payload, model, raw_state, stock)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _floor_neighbors(model: SokobanGoalDirectedSDModel, cell: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for direction in DIRECTIONS:
        nxt = (cell[0] + direction[0], cell[1] + direction[1])
        if model.level.is_floor(nxt):
            out.append(nxt)
    return out


def _articulation_cells(model: SokobanGoalDirectedSDModel) -> set[tuple[int, int]]:
    """Static floor graph articulation points used as bottleneck feedback."""

    cells = set(model.floor_cells)
    if not cells:
        return set()
    adjacency = {cell: [n for n in _floor_neighbors(model, cell) if n in cells] for cell in cells}
    visited: set[tuple[int, int]] = set()
    disc: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    arts: set[tuple[int, int]] = set()
    time = 0

    def dfs(cell: tuple[int, int]) -> None:
        nonlocal time
        visited.add(cell)
        disc[cell] = time
        low[cell] = time
        time += 1
        child_count = 0
        for nxt in adjacency[cell]:
            if nxt not in visited:
                parent[nxt] = cell
                child_count += 1
                dfs(nxt)
                low[cell] = min(low[cell], low[nxt])
                if parent.get(cell) is None and child_count > 1:
                    arts.add(cell)
                if parent.get(cell) is not None and low[nxt] >= disc[cell]:
                    arts.add(cell)
            elif nxt != parent.get(cell):
                low[cell] = min(low[cell], disc[nxt])

    for cell in sorted(cells):
        if cell not in visited:
            parent[cell] = None
            dfs(cell)
    return arts


def _structural_pressures(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    *,
    static_articulations: set[tuple[int, int]] | None = None,
) -> Dict[str, float]:
    boxes = set(raw_state.boxes)
    goals = set(model.level.goals)
    n_boxes = max(model.level.n_boxes, 1)
    n_goals = max(len(goals), 1)

    blocked_goals = 0
    goal_neighbors: set[tuple[int, int]] = set()
    for goal in goals:
        neigh = set(_floor_neighbors(model, goal)) - goals
        goal_neighbors.update(neigh)
        if any(cell in boxes for cell in neigh):
            blocked_goals += 1

    floor_degree = {cell: len(_floor_neighbors(model, cell)) for cell in model.floor_cells}
    corridor_cells = {cell for cell, degree in floor_degree.items() if degree <= 2}
    articulations = static_articulations if static_articulations is not None else _articulation_cells(model)
    dead_square_adjacent = 0
    for box in raw_state.boxes:
        if box in goals:
            continue
        if any(neigh in model.dead_squares for neigh in _floor_neighbors(model, box)):
            dead_square_adjacent += 1

    return {
        "goal_entry_blocked_fraction": _safe_div(blocked_goals, n_goals),
        "goal_adjacent_box_pressure": _safe_div(len((boxes & goal_neighbors) - goals), n_boxes),
        "corridor_box_pressure": _safe_div(len((boxes & corridor_cells) - goals), n_boxes),
        "articulation_box_pressure": _safe_div(len((boxes & articulations) - goals), n_boxes),
        "dead_square_adjacent_box_pressure": _safe_div(dead_square_adjacent, n_boxes),
    }


def _goal_neighbor_degree(model: SokobanGoalDirectedSDModel, cell: tuple[int, int]) -> int:
    goals = set(model.level.goals)
    return sum(1 for neigh in _floor_neighbors(model, cell) if neigh in goals)


def _goal_room_order_bonus_from_cells(
    model: SokobanGoalDirectedSDModel,
    before_cell: tuple[int, int],
    after_cell: tuple[int, int],
) -> float:
    goals = set(model.level.goals)
    if before_cell not in goals or after_cell not in goals:
        return 0.0
    before_degree = _goal_neighbor_degree(model, before_cell)
    after_degree = _goal_neighbor_degree(model, after_cell)
    if after_degree >= before_degree:
        return 0.0
    return float(1.0 + 0.5 * max(0, 1 - after_degree))


def _goal_room_order_bonus_for_action(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    action: SokobanPushAction,
) -> float:
    if action.box_index < 0 or action.box_index >= len(raw_state.boxes):
        return 0.0
    before_cell = raw_state.boxes[action.box_index]
    direction = DIRECTIONS[action.direction]
    after_cell = (before_cell[0] + direction[0], before_cell[1] + direction[1])
    return _goal_room_order_bonus_from_cells(model, before_cell, after_cell)


def _goal_entry_delay_penalty_for_action(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    action: SokobanPushAction,
) -> float:
    if action.box_index < 0 or action.box_index >= len(raw_state.boxes):
        return 0.0
    goals = set(model.level.goals)
    before_cell = raw_state.boxes[action.box_index]
    direction = DIRECTIONS[action.direction]
    after_cell = (before_cell[0] + direction[0], before_cell[1] + direction[1])
    if before_cell in goals or after_cell not in goals:
        return 0.0
    for candidate in model.legal_pushes(raw_state):
        if _goal_room_order_bonus_for_action(model, raw_state, candidate) > 0.0:
            return 1.0
    return 0.0


def _goal_room_conflict_score_for_action(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    action: SokobanPushAction,
) -> float:
    """Narrow root-prior signal for goal-room order versus entry conflicts."""

    order_bonus = _goal_room_order_bonus_for_action(model, raw_state, action)
    entry_delay = _goal_entry_delay_penalty_for_action(model, raw_state, action)
    if order_bonus <= 0.0 and entry_delay <= 0.0:
        return 0.0
    return float(order_bonus - entry_delay)


def _goal_room_conflict_prior_active(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
) -> bool:
    return any(
        abs(_goal_room_conflict_score_for_action(model, raw_state, action)) > 1e-12
        for action in model.legal_pushes(raw_state)
    )


def _goal_room_order_bonus_for_transition(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    next_raw_state: SokobanState,
) -> float:
    removed = set(raw_state.boxes) - set(next_raw_state.boxes)
    added = set(next_raw_state.boxes) - set(raw_state.boxes)
    if len(removed) != 1 or len(added) != 1:
        return 0.0
    return _goal_room_order_bonus_from_cells(model, next(iter(removed)), next(iter(added)))


def _goal_entry_delay_penalty_for_transition(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    next_raw_state: SokobanState,
) -> float:
    removed = set(raw_state.boxes) - set(next_raw_state.boxes)
    added = set(next_raw_state.boxes) - set(raw_state.boxes)
    if len(removed) != 1 or len(added) != 1:
        return 0.0
    before_cell = next(iter(removed))
    after_cell = next(iter(added))
    goals = set(model.level.goals)
    if before_cell in goals or after_cell not in goals:
        return 0.0
    for action in model.legal_pushes(raw_state):
        if _goal_room_order_bonus_for_action(model, raw_state, action) > 0.0:
            return 1.0
    return 0.0


def expert_abstract_value(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    *,
    static_articulations: set[tuple[int, int]] | None = None,
    revisit_count: int = 0,
) -> float:
    """Bounded value over the SD abstraction plus structural feedback.

    The score encodes the same high-level mechanisms a Sokoban expert usually
    preserves while planning: goal progress, assignment feasibility, mobility,
    access, congestion, irreversibility, bottlenecks and loop pressure. It is
    used only by the planner's consumption layer; stock-flow transitions remain
    supplied by the selected transition provider.
    """

    stock_arr = np.asarray(stock, dtype=np.float64)
    features = model.abstract_features(raw_state)
    if features.solved:
        return 10.0
    if features.deadlocked or features.legal_push_count == 0:
        return -10.0

    n_boxes = max(model.level.n_boxes, 1)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    board_scale = max(model.level.height + model.level.width, 1)
    lb_pressure = (
        _clamp01(float(features.matching_lower_bound) / (n_boxes * board_scale))
        if np.isfinite(features.matching_lower_bound)
        else 1.0
    )
    goal_frac = _safe_div(features.boxes_on_goals, n_boxes)
    legal_ratio = _safe_div(features.legal_push_count, max_pushes)
    viable_ratio = _safe_div(features.viable_push_count, max_pushes)
    pressures = _structural_pressures(
        model,
        raw_state,
        static_articulations=static_articulations,
    )
    raw = (
        6.0 * goal_frac
        + 2.0 * (1.0 - lb_pressure)
        + 1.2 * viable_ratio
        + 0.6 * legal_ratio
        + 1.1 * float(stock_arr[2])
        + 0.9 * float(stock_arr[3])
        - 1.0 * float(stock_arr[1])
        - 1.2 * float(stock_arr[4])
        - 2.1 * float(stock_arr[5])
        - 0.9 * pressures["goal_entry_blocked_fraction"]
        - 0.8 * pressures["goal_adjacent_box_pressure"]
        - 0.7 * pressures["corridor_box_pressure"]
        - 0.9 * pressures["articulation_box_pressure"]
        - 1.6 * pressures["dead_square_adjacent_box_pressure"]
        - 0.35 * max(0, revisit_count)
        - 1.8
    )
    return float(np.clip(raw, -10.0, 10.0))


class StockPlannerFeedback:
    """Original planning consumption: fixed stock reward and uniform action prior."""

    feedback_id = "fixed_stock_reward_uniform_prior"

    def reward(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock_before: np.ndarray,
        next_raw_state: SokobanState,
        stock_after: np.ndarray,
    ) -> float:
        return stock_reward(stock_before, stock_after)

    def action_priors(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        legal_actions: Sequence[SokobanPushAction],
    ) -> Dict[SokobanPushAction, float]:
        prob = 1.0 / max(len(legal_actions), 1)
        return {action: prob for action in legal_actions}


class ExpertAbstractPlannerFeedback:
    """Human-expert-inspired SD feedback for reward shaping and action priors."""

    feedback_id = "expert_abstract_dynamics_feedback_v1"

    def __init__(
        self,
        *,
        delta_weight: float = 0.25,
        prior_temperature: float = 1.0,
        prior_uniform_mix: float = 0.10,
        viability_penalty: float = 2.0,
        goal_room_order_weight: float = 0.0,
    ):
        self.delta_weight = float(delta_weight)
        self.prior_temperature = max(float(prior_temperature), 1e-6)
        self.prior_uniform_mix = float(np.clip(prior_uniform_mix, 0.0, 1.0))
        self.viability_penalty = float(max(0.0, viability_penalty))
        self.goal_room_order_weight = float(max(0.0, goal_room_order_weight))
        self._articulation_cache: dict[int, set[tuple[int, int]]] = {}

    def _articulations(self, model: SokobanGoalDirectedSDModel) -> set[tuple[int, int]]:
        key = id(model)
        if key not in self._articulation_cache:
            self._articulation_cache[key] = _articulation_cells(model)
        return self._articulation_cache[key]

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        *,
        revisit_count: int = 0,
    ) -> float:
        return expert_abstract_value(
            model,
            raw_state,
            stock,
            static_articulations=self._articulations(model),
            revisit_count=revisit_count,
        )

    def reward(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock_before: np.ndarray,
        next_raw_state: SokobanState,
        stock_after: np.ndarray,
    ) -> float:
        before = self.value(model, raw_state, stock_before)
        after = self.value(model, next_raw_state, stock_after)
        shaped = stock_reward(stock_before, stock_after) + self.delta_weight * (after - before)
        if model.is_solved(next_raw_state):
            shaped += 4.0
        elif model.is_deadlocked(next_raw_state) or not model.legal_pushes(next_raw_state):
            shaped -= 4.0
        else:
            next_features = model.abstract_features(next_raw_state)
            if next_features.viable_push_count == 0:
                shaped -= self.viability_penalty
            elif next_features.viable_push_count == 1:
                shaped -= 0.5 * self.viability_penalty
        if self.goal_room_order_weight > 0.0:
            shaped += self.goal_room_order_weight * _goal_room_order_bonus_for_transition(
                model,
                raw_state,
                next_raw_state,
            )
            shaped -= 1.25 * self.goal_room_order_weight * _goal_entry_delay_penalty_for_transition(
                model,
                raw_state,
                next_raw_state,
            )
        return float(np.clip(shaped, -6.0, 6.0))

    def action_priors(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        legal_actions: Sequence[SokobanPushAction],
    ) -> Dict[SokobanPushAction, float]:
        if not legal_actions:
            return {}
        before = self.value(model, raw_state, stock)
        scores = []
        static_articulations = self._articulations(model)
        for action in legal_actions:
            next_state = model.step_state(raw_state, action)
            next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
            score = expert_abstract_value(
                model,
                next_state,
                next_stock,
                static_articulations=static_articulations,
            ) - before
            if model.is_solved(next_state):
                score += 5.0
            elif model.is_deadlocked(next_state) or not model.legal_pushes(next_state):
                score -= 5.0
            else:
                next_features = model.abstract_features(next_state)
                if next_features.viable_push_count == 0:
                    score -= self.viability_penalty
                elif next_features.viable_push_count == 1:
                    score -= 0.5 * self.viability_penalty
            if self.goal_room_order_weight > 0.0:
                score += self.goal_room_order_weight * _goal_room_order_bonus_for_action(
                    model,
                    raw_state,
                    action,
                )
                score -= 1.25 * self.goal_room_order_weight * _goal_entry_delay_penalty_for_action(
                    model,
                    raw_state,
                    action,
                )
            scores.append(float(score))
        logits = np.asarray(scores, dtype=np.float64) / self.prior_temperature
        logits -= float(np.max(logits))
        exp = np.exp(logits)
        learned = exp / max(float(np.sum(exp)), 1e-12)
        uniform = np.ones_like(learned) / len(learned)
        probs = (1.0 - self.prior_uniform_mix) * learned + self.prior_uniform_mix * uniform
        return {action: float(prob) for action, prob in zip(legal_actions, probs)}


class ExpertAbstractTerminalValueEvaluator:
    """Terminal value readout over the expert abstract SD feedback state."""

    evaluator_id = "expert_abstract_dynamics_value_v1"

    def __init__(self, feedback: ExpertAbstractPlannerFeedback | None = None):
        self.feedback = feedback or ExpertAbstractPlannerFeedback()

    def should_stop(self, model: SokobanGoalDirectedSDModel, raw_state: SokobanState) -> bool:
        return model.is_solved(raw_state) or not model.legal_pushes(raw_state)

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
    ) -> float:
        return self.feedback.value(model, raw_state, stock)


def planner_action_to_dict(action: PlannerAction | None) -> Dict[str, Any] | None:
    if action is None:
        return None
    return {
        "source": action.source,
        "duration": int(action.duration),
        "structural_score": float(action.structural_score),
        "first_action": action_to_dict(action.first) if action.primitives else None,
        "primitive_actions": [action_to_dict(primitive) for primitive in action.primitives],
        "flat_index_tuple": [int(idx) for idx in action.flat_index_tuple],
    }


def _bounded_matching_improvement(
    model: SokobanGoalDirectedSDModel,
    before: float,
    after: float,
) -> float:
    scale = max(model.level.n_boxes * (model.level.height + model.level.width), 1)
    if np.isfinite(before) and np.isfinite(after):
        return float(np.clip((before - after) / scale, -1.0, 1.0))
    if not np.isfinite(before) and np.isfinite(after):
        return 1.0
    if np.isfinite(before) and not np.isfinite(after):
        return -1.0
    return 0.0


def _score_option_sequence(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    primitives: Sequence[SokobanPushAction],
    *,
    static_articulations: set[tuple[int, int]] | None = None,
    before_features: Any | None = None,
) -> tuple[float, SokobanState]:
    """Local, non-search option score from task-oriented abstract consequences."""

    before = before_features or model.abstract_features(raw_state)
    state = raw_state
    for primitive in primitives:
        if primitive not in set(model.legal_pushes(state)):
            return -100.0, state
        state = model.step_state(state, primitive)
        if model.is_solved(state):
            break
    after = model.abstract_features(state)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    score = (
        4.0 * (after.boxes_on_goals - before.boxes_on_goals)
        + 2.5 * _bounded_matching_improvement(model, before.matching_lower_bound, after.matching_lower_bound)
        + 0.6 * _safe_div(after.viable_push_count - before.viable_push_count, max_pushes)
        + 0.3 * _safe_div(after.legal_push_count - before.legal_push_count, max_pushes)
        - 0.04 * max(len(primitives) - 1, 0)
    )
    if after.solved:
        score += 10.0
    elif after.deadlocked:
        score -= 10.0
    elif after.legal_push_count == 0:
        score -= 4.0
    elif after.viable_push_count <= 1:
        score -= 1.0
    pressures_before = _structural_pressures(
        model,
        raw_state,
        static_articulations=static_articulations,
    )
    pressures_after = _structural_pressures(
        model,
        state,
        static_articulations=static_articulations,
    )
    score += 0.5 * (
        pressures_before["goal_entry_blocked_fraction"] - pressures_after["goal_entry_blocked_fraction"]
    )
    score += 0.4 * (
        pressures_before["dead_square_adjacent_box_pressure"] - pressures_after["dead_square_adjacent_box_pressure"]
    )
    return float(score), state


def _target2_options(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    *,
    max_options_per_node: int,
) -> list[PlannerAction]:
    if max_options_per_node <= 0:
        return []
    options: list[PlannerAction] = []
    first_actions = tuple(model.legal_pushes(raw_state))
    before = model.abstract_features(raw_state)
    static_articulations = _articulation_cells(model)
    for first in first_actions:
        pushed_box = raw_state.boxes[first.box_index]
        direction = DIRECTIONS[first.direction]
        moved_box = (pushed_box[0] + direction[0], pushed_box[1] + direction[1])
        state1 = model.step_state(raw_state, first)
        if model.is_solved(state1):
            continue
        try:
            moved_box_index = state1.boxes.index(moved_box)
        except ValueError:
            continue
        for second in model.legal_pushes(state1):
            if second.box_index != moved_box_index:
                continue
            primitives = (first, second)
            score, state2 = _score_option_sequence(
                model,
                raw_state,
                primitives,
                static_articulations=static_articulations,
                before_features=before,
            )
            after = model.abstract_features(state2)
            improved = (
                after.solved
                or after.boxes_on_goals > before.boxes_on_goals
                or _bounded_matching_improvement(
                    model,
                    before.matching_lower_bound,
                    after.matching_lower_bound,
                )
                > 0.0
                or score > 0.05
            )
            if improved and not after.deadlocked:
                options.append(
                    PlannerAction(
                        primitives=primitives,
                        source="target2_local_structural_option",
                        structural_score=score,
                    )
                )
    options.sort(
        key=lambda action: (
            action.structural_score,
            -len(action.primitives),
            tuple(-idx for idx in action.flat_index_tuple),
        ),
        reverse=True,
    )
    return options[:max_options_per_node]


def _option_stock_return(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    planner_action: PlannerAction,
    *,
    gamma: float,
) -> float:
    """Evaluate a candidate option with SD stock reward plus fixed leaf value.

    This is a local option-level rollout value used only to shape planner
    priors. It uses exact raw state stepping for action support and stock proxy
    readout, but does not call a teacher and does not replace the selected
    transition provider used inside MCTS expansion.
    """

    state = raw_state
    current_stock = np.asarray(stock, dtype=np.float64)
    total = 0.0
    discount = 1.0
    for primitive in planner_action.primitives:
        if primitive not in set(model.legal_pushes(state)):
            return -12.0
        next_state = model.step_state(state, primitive)
        next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
        total += discount * stock_reward(current_stock, next_stock)
        discount *= gamma
        state = next_state
        current_stock = next_stock
        if model.is_solved(state) or not model.legal_pushes(state):
            break
    total += discount * stock_value(current_stock)
    if model.is_solved(state):
        total += 10.0
    elif model.is_deadlocked(state):
        total -= 10.0
    elif not model.legal_pushes(state):
        total -= 4.0
    return float(np.clip(total, -12.0, 12.0))


def make_planner_action_priors(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    planner_feedback: PlannerFeedback,
    *,
    action_abstraction_mode: str,
    max_option_length: int,
    max_options_per_node: int,
    option_prior_max_multiplier: float,
    gamma: float,
    previous_push: SokobanPushAction | None = None,
    previous_pushed_box_cell: tuple[int, int] | None = None,
    route_consistency_weight: float = 0.0,
    audit_proxy_prior_weight: float = 0.0,
    goal_room_conflict_prior_weight: float = 0.0,
    option_value_prior_weight: float = 0.0,
    learned_action_prior_payload: Dict[str, Any] | None = None,
    learned_action_prior_weight: float = 0.0,
    learned_action_prior_max_multiplier: float = 3.0,
) -> tuple[Dict[PlannerAction, float], Dict[str, Any]]:
    primitive_actions = tuple(model.legal_pushes(raw_state))
    if not primitive_actions:
        return {}, {
            "option_value_prior_rows": 0,
            "option_value_prior_positive_rows": 0,
            "option_value_prior_max": None,
            "option_value_prior_mean": None,
        }
    primitive_priors = planner_feedback.action_priors(
        model,
        raw_state,
        stock,
        primitive_actions,
    )
    planner_actions = [
        PlannerAction(primitives=(primitive,), source="primitive_push", structural_score=0.0)
        for primitive in primitive_actions
    ]
    target2_prior_by_first: dict[SokobanPushAction, float] = {}
    if action_abstraction_mode in {"target2", "target2_prior"} and max_option_length >= 2:
        target2_options = _target2_options(
            model,
            raw_state,
            max_options_per_node=max_options_per_node,
        )
        for option in target2_options:
            target2_prior_by_first[option.first] = max(
                target2_prior_by_first.get(option.first, float("-inf")),
                option.structural_score,
            )
        if action_abstraction_mode == "target2":
            planner_actions.extend(target2_options)
        elif action_abstraction_mode == "target2_prior":
            planner_actions = [
                PlannerAction(
                    primitives=action.primitives,
                    source="primitive_push_with_target2_prior",
                    structural_score=target2_prior_by_first.get(action.first, 0.0),
                )
                for action in planner_actions
            ]
    option_value_scores: dict[PlannerAction, float] = {}
    if option_value_prior_weight > 0.0:
        for planner_action in planner_actions:
            option_value_scores[planner_action] = _option_stock_return(
                model,
                raw_state,
                stock,
                planner_action,
                gamma=gamma,
            )
    weights: list[float] = []
    for planner_action in planner_actions:
        first_prior = primitive_priors.get(planner_action.first, 1.0 / max(len(primitive_actions), 1))
        if planner_action.duration == 1:
            weight = max(first_prior, 1e-9)
            if planner_action.source == "primitive_push_with_target2_prior":
                option_score = max(0.0, planner_action.structural_score)
                option_boost = float(np.exp(np.clip(option_score, 0.0, 4.0)))
                option_boost = min(option_boost, max(float(option_prior_max_multiplier), 1.0))
                weight *= option_boost
        else:
            option_boost = float(np.exp(np.clip(planner_action.structural_score, -4.0, 4.0)))
            option_boost = min(option_boost, max(float(option_prior_max_multiplier), 1.0))
            weight = max(first_prior, 1e-9) * option_boost
        if route_consistency_weight > 0.0 and (previous_push is not None or previous_pushed_box_cell is not None):
            same_physical_box = (
                previous_pushed_box_cell is not None
                and planner_action.first.box_index < len(raw_state.boxes)
                and raw_state.boxes[planner_action.first.box_index] == previous_pushed_box_cell
            )
            same_box = same_physical_box or (
                previous_push is not None and planner_action.first.box_index == previous_push.box_index
            )
            same_direction = previous_push is not None and planner_action.first.direction == previous_push.direction
            if same_physical_box and same_direction:
                weight *= 1.0 + 1.5 * float(route_consistency_weight)
            elif same_physical_box:
                weight *= 1.0 + float(route_consistency_weight)
            elif same_box:
                weight *= 1.0 + 0.35 * float(route_consistency_weight)
        if audit_proxy_prior_weight > 0.0:
            next_state = model.step_state(raw_state, planner_action.first)
            before_features = model.abstract_features(raw_state)
            after_features = model.abstract_features(next_state)
            goal_delta = after_features.boxes_on_goals - before_features.boxes_on_goals
            viable_delta = after_features.viable_push_count - before_features.viable_push_count
            legal_delta = after_features.legal_push_count - before_features.legal_push_count
            matching_improvement = _bounded_matching_improvement(
                model,
                before_features.matching_lower_bound,
                after_features.matching_lower_bound,
            )
            proxy_score = (
                1.5 * _goal_room_order_bonus_for_action(model, raw_state, planner_action.first)
                - 1.0 * _goal_entry_delay_penalty_for_action(model, raw_state, planner_action.first)
                + 0.6 * matching_improvement
                - 0.45 * max(goal_delta, 0)
                - 0.08 * max(-viable_delta, 0)
                - 0.04 * max(-legal_delta, 0)
            )
            proxy_boost = float(np.exp(np.clip(audit_proxy_prior_weight * proxy_score, -2.0, 2.0)))
            weight *= proxy_boost
        if goal_room_conflict_prior_weight > 0.0:
            conflict_score = _goal_room_conflict_score_for_action(
                model,
                raw_state,
                planner_action.first,
            )
            if abs(conflict_score) > 1e-12:
                conflict_boost = float(
                    np.exp(np.clip(goal_room_conflict_prior_weight * conflict_score, -2.5, 2.5))
                )
                weight *= conflict_boost
        if option_value_prior_weight > 0.0 and planner_action in option_value_scores:
            value_score = option_value_scores[planner_action]
            value_boost = float(
                np.exp(np.clip(option_value_prior_weight * value_score, -2.0, 2.0))
            )
            max_multiplier = max(float(option_prior_max_multiplier), 1.0)
            value_boost = float(np.clip(value_boost, 1.0 / max_multiplier, max_multiplier))
            weight *= value_boost
        if learned_action_prior_payload is not None and learned_action_prior_weight > 0.0:
            logit = predict_action_prior_logit(
                learned_action_prior_payload,
                model,
                raw_state,
                stock,
                planner_action.first,
                target2_structural_score=planner_action.structural_score,
            )
            prior_boost = float(np.exp(np.clip(learned_action_prior_weight * logit, -2.0, 2.0)))
            max_multiplier = max(float(learned_action_prior_max_multiplier), 1.0)
            prior_boost = float(np.clip(prior_boost, 1.0 / max_multiplier, max_multiplier))
            weight *= prior_boost
        weights.append(float(weight))
    option_value_meta = {
        "option_value_prior_rows": int(len(option_value_scores)),
        "option_value_prior_positive_rows": int(
            sum(1 for value in option_value_scores.values() if value > 0.0)
        ),
        "option_value_prior_max": (
            float(max(option_value_scores.values())) if option_value_scores else None
        ),
        "option_value_prior_mean": (
            float(np.mean(list(option_value_scores.values()))) if option_value_scores else None
        ),
    }
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        prob = 1.0 / max(len(planner_actions), 1)
        return {action: prob for action in planner_actions}, option_value_meta
    return {action: float(weight / weight_sum) for action, weight in zip(planner_actions, weights)}, option_value_meta


@dataclass
class PlanningNode:
    raw_state: SokobanState
    stock: np.ndarray
    depth: int
    parent: "PlanningNode | None" = None
    action: PlannerAction | None = None
    prior: float = 0.0
    reward: float = 0.0
    discount: float = 1.0
    children: Dict[PlannerAction, "PlanningNode"] = field(default_factory=dict)
    action_priors: Dict[PlannerAction, float] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class PlannerActionRollout:
    next_raw_state: SokobanState
    next_stock: np.ndarray
    reward: float
    discount: float
    duration: int
    provider_counts: Counter[str]
    planner_feedback_counts: Counter[str]
    revisit_feedback_hits: int


def rollout_planner_action(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    planner_action: PlannerAction,
    provider: TransitionProvider,
    planner_feedback: PlannerFeedback,
    *,
    gamma: float,
    visited_states: Counter[SokobanState],
    revisit_feedback_penalty: float,
) -> PlannerActionRollout:
    state = raw_state
    current_stock = np.asarray(stock, dtype=np.float64)
    cumulative_reward = 0.0
    cumulative_discount = 1.0
    provider_counts: Counter[str] = Counter()
    planner_feedback_counts: Counter[str] = Counter()
    revisit_feedback_hits = 0
    duration = 0
    for primitive in planner_action.primitives:
        if primitive not in set(model.legal_pushes(state)):
            break
        prediction = provider.predict(model, state, current_stock, primitive)
        provider_counts[prediction.provider_id] += 1
        next_state = model.step_state(state, primitive)
        revisit_count = int(visited_states.get(next_state, 0))
        reward = planner_feedback.reward(
            model,
            state,
            current_stock,
            next_state,
            prediction.next_stock,
        )
        if revisit_feedback_penalty > 0.0 and revisit_count > 0:
            reward -= revisit_feedback_penalty * revisit_count
            revisit_feedback_hits += 1
        planner_feedback_counts[planner_feedback.feedback_id] += 1
        cumulative_reward += cumulative_discount * reward
        cumulative_discount *= gamma
        duration += 1
        state = next_state
        current_stock = prediction.next_stock
        if model.is_solved(state) or not model.legal_pushes(state):
            break
    return PlannerActionRollout(
        next_raw_state=state,
        next_stock=current_stock,
        reward=float(np.clip(cumulative_reward, -12.0, 12.0)),
        discount=float(cumulative_discount),
        duration=duration,
        provider_counts=provider_counts,
        planner_feedback_counts=planner_feedback_counts,
        revisit_feedback_hits=int(revisit_feedback_hits),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
        help="Learned flow checkpoint for learned_sdmcts.",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used for planning evaluation.",
    )
    parser.add_argument("--use-checkpoint-valid-levels", action="store_true", default=False)
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--level-offset", type=int, default=0)
    parser.add_argument("--max-eval-levels", type=int, default=6)
    parser.add_argument(
        "--methods",
        default="learned_sdmcts,zero_flow_sdmcts,oracle_abstract_sdmcts,heuristic_diagnostic",
        help=(
            "Comma-separated methods to evaluate. Supported values: learned_sdmcts, "
            "zero_flow_sdmcts, oracle_abstract_sdmcts, heuristic_diagnostic, "
            "heuristic_best_first_diagnostic."
        ),
    )
    parser.add_argument("--budgets", default="16,48", help="Comma-separated MCTS simulation budgets.")
    parser.add_argument("--seeds", default="0,1", help="Comma-separated random seeds.")
    parser.add_argument("--mcts-depth", type=int, default=3)
    parser.add_argument("--puct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-pushes", type=int, default=30)
    parser.add_argument("--max-state-revisits", type=int, default=3)
    parser.add_argument(
        "--root-action-selection",
        choices=("visits", "value"),
        default="visits",
        help="Choose the executed root action by visit count or by root child value.",
    )
    parser.add_argument(
        "--terminal-value-mode",
        choices=("stock", "zero", "tva", "expert"),
        default="stock",
        help="Leaf value evaluator used by SD-MCTS. 'stock' preserves the pre-TVA baseline.",
    )
    parser.add_argument(
        "--planner-feedback-mode",
        choices=("stock", "expert"),
        default="stock",
        help="Reward and action-prior feedback consumed by SD-MCTS; transition provider is unchanged.",
    )
    parser.add_argument(
        "--expert-delta-weight",
        type=float,
        default=0.25,
        help="Weight on expert abstract value change when --planner-feedback-mode=expert.",
    )
    parser.add_argument(
        "--expert-prior-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for expert structural action priors.",
    )
    parser.add_argument(
        "--expert-prior-uniform-mix",
        type=float,
        default=0.10,
        help="Uniform prior mixture fraction for expert structural action priors.",
    )
    parser.add_argument(
        "--revisit-feedback-penalty",
        type=float,
        default=2.0,
        help="Penalty applied inside expert feedback when a planned successor repeats a closed-loop state.",
    )
    parser.add_argument(
        "--expert-viability-penalty",
        type=float,
        default=2.0,
        help="Penalty for expert feedback when a successor has no or one non-deadlocked push option.",
    )
    parser.add_argument(
        "--goal-room-order-weight",
        type=float,
        default=0.0,
        help="Expert feedback weight for moving a box within the goal region toward lower-degree endpoint goals.",
    )
    parser.add_argument(
        "--action-abstraction-mode",
        choices=("push", "target2", "target2_prior"),
        default="push",
        help=(
            "Planner action set: primitive pushes only, primitive pushes plus target2 options, "
            "or primitive pushes with target2 potential folded into priors."
        ),
    )
    parser.add_argument(
        "--max-option-length",
        type=int,
        default=2,
        help="Maximum primitive push length for option candidates when action abstraction is enabled.",
    )
    parser.add_argument(
        "--max-options-per-node",
        type=int,
        default=6,
        help="Maximum target2 option candidates added at each MCTS node.",
    )
    parser.add_argument(
        "--option-prior-max-multiplier",
        type=float,
        default=3.0,
        help="Maximum prior multiplier assigned to a structural option over its first primitive prior.",
    )
    parser.add_argument(
        "--option-value-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off planner prior boost from local option-level SD rollout return. "
            "Uses no teacher and does not change transition expansion."
        ),
    )
    parser.add_argument(
        "--option-execution-mode",
        choices=("first_push", "full_option"),
        default="first_push",
        help="Closed-loop execution of a selected option: execute only its first push or the full primitive sequence.",
    )
    parser.add_argument(
        "--route-consistency-weight",
        type=float,
        default=0.0,
        help="Prior multiplier weight for continuing the previous box/direction route intent.",
    )
    parser.add_argument(
        "--audit-proxy-prior-weight",
        type=float,
        default=0.0,
        help="Weight for a local prior derived from root-Q bias audit features.",
    )
    parser.add_argument(
        "--goal-room-conflict-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Root-only prior weight for conflicts between packing an existing goal-room box "
            "and entering a new box into the goal region."
        ),
    )
    parser.add_argument(
        "--learned-action-prior-checkpoint",
        default="",
        help="Checkpoint from train_sokoban_action_prior.py. Empty disables the learned prior.",
    )
    parser.add_argument(
        "--learned-action-prior-weight",
        type=float,
        default=0.0,
        help="Multiplier weight for the checkpoint-backed learned action prior.",
    )
    parser.add_argument(
        "--learned-action-prior-max-multiplier",
        type=float,
        default=3.0,
        help="Clamp for the learned prior multiplicative boost per candidate action.",
    )
    parser.add_argument(
        "--tva-checkpoint",
        default="",
        help="Checkpoint written by train_sokoban_tva.py when --terminal-value-mode=tva.",
    )
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/stage3_sdmcts_planning_smoke.json",
        help="JSON output path.",
    )
    return parser


def parse_csv_ints(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one integer.")
    return values


def parse_csv_strings(raw: str, *, name: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one method.")
    return values


def make_provider(method: str, learned_provider: LearnedStockFlowTransitionProvider) -> TransitionProvider | None:
    if method == "learned_sdmcts":
        return learned_provider
    if method == "oracle_abstract_sdmcts":
        return OracleAbstractTransitionProvider()
    if method == "zero_flow_sdmcts":
        return ZeroFlowTransitionProvider()
    if method in {"heuristic_diagnostic", "heuristic_best_first_diagnostic"}:
        return None
    raise SystemExit(f"Unknown method: {method}")


def make_terminal_value_evaluator(args: argparse.Namespace) -> TerminalValueEvaluator:
    if args.terminal_value_mode == "stock":
        return StockTerminalValueEvaluator()
    if args.terminal_value_mode == "zero":
        return ZeroTerminalValueEvaluator()
    if args.terminal_value_mode == "expert":
        return ExpertAbstractTerminalValueEvaluator(make_planner_feedback(args, force_expert=True))
    if args.terminal_value_mode == "tva":
        if not args.tva_checkpoint:
            raise SystemExit("--tva-checkpoint is required when --terminal-value-mode=tva.")
        return SokobanTvaTerminalValueEvaluator(Path(args.tva_checkpoint))
    raise SystemExit(f"Unknown terminal value mode: {args.terminal_value_mode}")


def make_planner_feedback(
    args: argparse.Namespace,
    *,
    force_expert: bool = False,
) -> PlannerFeedback:
    mode = "expert" if force_expert else args.planner_feedback_mode
    if mode == "stock":
        return StockPlannerFeedback()
    if mode == "expert":
        return ExpertAbstractPlannerFeedback(
            delta_weight=args.expert_delta_weight,
            prior_temperature=args.expert_prior_temperature,
            prior_uniform_mix=args.expert_prior_uniform_mix,
            viability_penalty=args.expert_viability_penalty,
            goal_room_order_weight=args.goal_room_order_weight,
        )
    raise SystemExit(f"Unknown planner feedback mode: {mode}")


def choose_mcts_action(
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    provider: TransitionProvider,
    terminal_value: TerminalValueEvaluator,
    planner_feedback: PlannerFeedback,
    *,
    num_simulations: int,
    max_depth: int,
    puct_c: float,
    gamma: float,
    root_action_selection: str,
    visited_states: Counter[SokobanState] | None,
    revisit_feedback_penalty: float,
    action_abstraction_mode: str,
    max_option_length: int,
    max_options_per_node: int,
    option_prior_max_multiplier: float,
    option_value_prior_weight: float,
    previous_push: SokobanPushAction | None,
    previous_pushed_box_cell: tuple[int, int] | None,
    route_consistency_weight: float,
    audit_proxy_prior_weight: float,
    goal_room_conflict_prior_weight: float = 0.0,
    learned_action_prior_payload: Dict[str, Any] | None = None,
    learned_action_prior_weight: float = 0.0,
    learned_action_prior_max_multiplier: float = 3.0,
    rng: np.random.RandomState | None = None,
) -> tuple[PlannerAction | None, Dict[str, Any]]:
    rng = rng or np.random.RandomState(0)
    root_stock = stock_proxy(model, root_state).as_array().astype(np.float64)
    root = PlanningNode(raw_state=root_state, stock=root_stock, depth=0)
    provider_counts: Counter[str] = Counter()
    terminal_value_counts: Counter[str] = Counter()
    terminal_leaf_counts: Counter[str] = Counter()
    planner_feedback_counts: Counter[str] = Counter()
    revisit_feedback_hits = 0
    expansions = 0
    option_candidates = 0
    option_expansions = 0
    option_primitives = 0
    option_value_prior_rows = 0
    option_value_prior_positive_rows = 0
    option_value_prior_max_values: list[float] = []
    option_value_prior_mean_values: list[float] = []
    goal_room_conflict_prior_roots = 0
    visited_states = visited_states or Counter()
    effective_revisit_penalty = (
        float(revisit_feedback_penalty)
        if planner_feedback.feedback_id == ExpertAbstractPlannerFeedback.feedback_id
        else 0.0
    )

    for _sim in range(num_simulations):
        node = root
        path = [node]
        while node.depth < max_depth and not terminal_value.should_stop(model, node.raw_state):
            legal_actions = tuple(model.legal_pushes(node.raw_state))
            if not legal_actions:
                break
            if not node.action_priors:
                node.action_priors, option_value_meta = make_planner_action_priors(
                    model,
                    node.raw_state,
                    node.stock,
                    planner_feedback,
                    action_abstraction_mode=action_abstraction_mode,
                    max_option_length=max_option_length,
                    max_options_per_node=max_options_per_node,
                    option_prior_max_multiplier=option_prior_max_multiplier,
                    gamma=gamma,
                    previous_push=previous_push if node is root else None,
                    previous_pushed_box_cell=previous_pushed_box_cell if node is root else None,
                    route_consistency_weight=route_consistency_weight if node is root else 0.0,
                    audit_proxy_prior_weight=audit_proxy_prior_weight if node is root else 0.0,
                    goal_room_conflict_prior_weight=(
                        goal_room_conflict_prior_weight if node is root else 0.0
                    ),
                    option_value_prior_weight=option_value_prior_weight,
                    learned_action_prior_payload=learned_action_prior_payload,
                    learned_action_prior_weight=learned_action_prior_weight,
                    learned_action_prior_max_multiplier=learned_action_prior_max_multiplier,
                )
                option_value_prior_rows += int(option_value_meta.get("option_value_prior_rows", 0))
                option_value_prior_positive_rows += int(
                    option_value_meta.get("option_value_prior_positive_rows", 0)
                )
                if option_value_meta.get("option_value_prior_max") is not None:
                    option_value_prior_max_values.append(float(option_value_meta["option_value_prior_max"]))
                if option_value_meta.get("option_value_prior_mean") is not None:
                    option_value_prior_mean_values.append(float(option_value_meta["option_value_prior_mean"]))
                if (
                    node is root
                    and goal_room_conflict_prior_weight > 0.0
                    and _goal_room_conflict_prior_active(model, node.raw_state)
                ):
                    goal_room_conflict_prior_roots += 1
                if action_abstraction_mode == "target2_prior":
                    option_candidates += sum(
                        1
                        for action in node.action_priors
                        if action.source == "primitive_push_with_target2_prior"
                        and action.structural_score > 0.0
                    )
                else:
                    option_candidates += sum(1 for action in node.action_priors if action.duration > 1)
            action_priors = node.action_priors
            unexpanded = [action for action in action_priors if action not in node.children]
            if unexpanded:
                if planner_feedback.feedback_id == StockPlannerFeedback.feedback_id:
                    planner_action = unexpanded[int(rng.randint(len(unexpanded)))]
                else:
                    def prior_key(candidate: PlannerAction) -> tuple[float, float, tuple[int, ...]]:
                        next_state_for_prior = node.raw_state
                        for primitive in candidate.primitives:
                            if primitive not in set(model.legal_pushes(next_state_for_prior)):
                                break
                            next_state_for_prior = model.step_state(next_state_for_prior, primitive)
                        revisit_count = int(visited_states.get(next_state_for_prior, 0))
                        adjusted_prior = action_priors.get(candidate, 0.0) * np.exp(
                            -effective_revisit_penalty * revisit_count
                        )
                        return (
                            float(adjusted_prior),
                            float(candidate.structural_score),
                            tuple(-idx for idx in candidate.flat_index_tuple),
                        )

                    planner_action = max(
                        unexpanded,
                        key=prior_key,
                    )
                rollout = rollout_planner_action(
                    model,
                    node.raw_state,
                    node.stock,
                    planner_action,
                    provider,
                    planner_feedback,
                    gamma=gamma,
                    visited_states=visited_states,
                    revisit_feedback_penalty=effective_revisit_penalty,
                )
                if rollout.duration <= 0:
                    node.children[planner_action] = PlanningNode(
                        raw_state=node.raw_state,
                        stock=node.stock.copy(),
                        depth=node.depth + 1,
                        parent=node,
                        action=planner_action,
                        prior=action_priors.get(planner_action, 1.0 / max(len(action_priors), 1)),
                        reward=-6.0,
                        discount=gamma,
                    )
                    break
                provider_counts.update(rollout.provider_counts)
                planner_feedback_counts.update(rollout.planner_feedback_counts)
                revisit_feedback_hits += rollout.revisit_feedback_hits
                if planner_action.duration > 1:
                    option_expansions += 1
                    option_primitives += rollout.duration
                child = PlanningNode(
                    raw_state=rollout.next_raw_state,
                    stock=rollout.next_stock,
                    depth=node.depth + 1,
                    parent=node,
                    action=planner_action,
                    prior=action_priors.get(planner_action, 1.0 / max(len(action_priors), 1)),
                    reward=rollout.reward,
                    discount=rollout.discount,
                )
                node.children[planner_action] = child
                expansions += 1
                node = child
                path.append(node)
                break
            node = max(node.children.values(), key=lambda child: puct_score(node, child, puct_c))
            path.append(node)

        total = terminal_value.value(model, node.raw_state, node.stock)
        terminal_value_counts[terminal_value.evaluator_id] += 1
        leaf_revisit_count = int(visited_states.get(node.raw_state, 0))
        if effective_revisit_penalty > 0.0 and leaf_revisit_count > 0:
            total -= effective_revisit_penalty * leaf_revisit_count
        if model.is_solved(node.raw_state):
            terminal_leaf_counts["solved"] += 1
        elif leaf_revisit_count > 0:
            terminal_leaf_counts["revisited"] += 1
        elif model.is_deadlocked(node.raw_state):
            terminal_leaf_counts["deadlocked"] += 1
        elif not model.legal_pushes(node.raw_state):
            terminal_leaf_counts["no_legal_actions"] += 1
        elif node.depth >= max_depth:
            terminal_leaf_counts["depth_cutoff"] += 1
        else:
            terminal_leaf_counts["frontier_cutoff"] += 1
        for path_node in reversed(path):
            if path_node.parent is not None:
                total = path_node.reward + path_node.discount * total
            path_node.visits += 1
            path_node.value_sum += total

    if not root.children:
        return None, {
            "provider_counts": dict(provider_counts),
            "terminal_value_counts": dict(terminal_value_counts),
            "terminal_leaf_counts": dict(terminal_leaf_counts),
            "planner_feedback_counts": dict(planner_feedback_counts),
            "revisit_feedback_hits": int(revisit_feedback_hits),
            "tree_expansions": expansions,
            "option_candidates": int(option_candidates),
            "option_expansions": int(option_expansions),
            "option_primitives": int(option_primitives),
            "option_value_prior_rows": int(option_value_prior_rows),
            "option_value_prior_positive_rows": int(option_value_prior_positive_rows),
            "option_value_prior_max": (
                float(max(option_value_prior_max_values)) if option_value_prior_max_values else None
            ),
            "option_value_prior_mean": (
                float(np.mean(option_value_prior_mean_values)) if option_value_prior_mean_values else None
            ),
            "goal_room_conflict_prior_roots": int(goal_room_conflict_prior_roots),
            "root_children": 0,
            "root_value": float(root.value),
            "root_policy": [],
        }

    children = list(root.children.items())
    visit_sum = sum(child.visits for _action, child in children)
    if root_action_selection == "visits":
        best_planner_action, best_child = max(
            children,
            key=lambda item: (item[1].visits, item[1].value, tuple(-idx for idx in item[0].flat_index_tuple)),
        )
    elif root_action_selection == "value":
        best_planner_action, best_child = max(
            children,
            key=lambda item: (item[1].value, item[1].visits, tuple(-idx for idx in item[0].flat_index_tuple)),
        )
    else:
        raise ValueError(f"Unknown root_action_selection: {root_action_selection}")
    policy = [
        {
            "action": planner_action_to_dict(action),
            "visits": int(child.visits),
            "q_value": float(child.value),
            "prior": float(child.prior),
            "probability": float(child.visits / visit_sum) if visit_sum else 0.0,
        }
        for action, child in sorted(children, key=lambda item: item[0].flat_index_tuple)
    ]
    return best_planner_action, {
        "provider_counts": dict(provider_counts),
        "terminal_value_counts": dict(terminal_value_counts),
        "terminal_leaf_counts": dict(terminal_leaf_counts),
        "planner_feedback_counts": dict(planner_feedback_counts),
        "revisit_feedback_hits": int(revisit_feedback_hits),
        "tree_expansions": expansions,
        "option_candidates": int(option_candidates),
        "option_expansions": int(option_expansions),
        "option_primitives": int(option_primitives),
        "option_value_prior_rows": int(option_value_prior_rows),
        "option_value_prior_positive_rows": int(option_value_prior_positive_rows),
        "option_value_prior_max": (
            float(max(option_value_prior_max_values)) if option_value_prior_max_values else None
        ),
        "option_value_prior_mean": (
            float(np.mean(option_value_prior_mean_values)) if option_value_prior_mean_values else None
        ),
        "goal_room_conflict_prior_roots": int(goal_room_conflict_prior_roots),
        "root_children": len(root.children),
        "root_value": float(root.value),
        "best_child_value": float(best_child.value),
        "best_planner_action": planner_action_to_dict(best_planner_action),
        "root_action_selection": root_action_selection,
        "root_policy": policy,
    }


def choose_heuristic_action(model: SokobanGoalDirectedSDModel, state: SokobanState) -> SokobanPushAction | None:
    ranked = model.rank_actions(state)
    return ranked[0][1] if ranked else None


def run_episode(
    model: SokobanGoalDirectedSDModel,
    *,
    method: str,
    provider: TransitionProvider | None,
    terminal_value: TerminalValueEvaluator,
    planner_feedback: PlannerFeedback,
    num_simulations: int,
    max_depth: int,
    puct_c: float,
    gamma: float,
    root_action_selection: str,
    max_pushes: int,
    max_state_revisits: int,
    revisit_feedback_penalty: float,
    action_abstraction_mode: str,
    max_option_length: int,
    max_options_per_node: int,
    option_prior_max_multiplier: float,
    option_value_prior_weight: float,
    option_execution_mode: str,
    route_consistency_weight: float,
    audit_proxy_prior_weight: float,
    goal_room_conflict_prior_weight: float = 0.0,
    learned_action_prior_payload: Dict[str, Any] | None = None,
    learned_action_prior_weight: float = 0.0,
    learned_action_prior_max_multiplier: float = 3.0,
    rng: np.random.RandomState | None = None,
) -> Dict[str, Any]:
    rng = rng or np.random.RandomState(0)
    state = model.initial_state
    visited: Counter[SokobanState] = Counter([state])
    trace: list[dict[str, Any]] = []
    provider_counts: Counter[str] = Counter()
    terminal_value_counts: Counter[str] = Counter()
    terminal_leaf_counts: Counter[str] = Counter()
    planner_feedback_counts: Counter[str] = Counter()
    revisit_feedback_hits = 0
    option_candidates = 0
    option_expansions = 0
    option_primitives = 0
    option_value_prior_rows = 0
    option_value_prior_positive_rows = 0
    goal_room_conflict_prior_roots = 0
    total_expansions = 0
    planning_time = 0.0
    pushes = 0
    previous_push: SokobanPushAction | None = None
    previous_pushed_box_cell: tuple[int, int] | None = None

    while pushes < max_pushes:
        if model.is_solved(state):
            return episode_result(
                solved=True,
                failure_type="solved",
                pushes=pushes,
                trace=trace,
                provider_counts=provider_counts,
                total_expansions=total_expansions,
                planning_time=planning_time,
                terminal_value_counts=terminal_value_counts,
                terminal_leaf_counts=terminal_leaf_counts,
                planner_feedback_counts=planner_feedback_counts,
                revisit_feedback_hits=revisit_feedback_hits,
                option_candidates=option_candidates,
                option_expansions=option_expansions,
                option_primitives=option_primitives,
                option_value_prior_rows=option_value_prior_rows,
                option_value_prior_positive_rows=option_value_prior_positive_rows,
                goal_room_conflict_prior_roots=goal_room_conflict_prior_roots,
            )
        legal = tuple(model.legal_pushes(state))
        if not legal:
            return episode_result(
                solved=False,
                failure_type="no_legal_actions",
                pushes=pushes,
                trace=trace,
                provider_counts=provider_counts,
                total_expansions=total_expansions,
                planning_time=planning_time,
                terminal_value_counts=terminal_value_counts,
                terminal_leaf_counts=terminal_leaf_counts,
                planner_feedback_counts=planner_feedback_counts,
                revisit_feedback_hits=revisit_feedback_hits,
                option_candidates=option_candidates,
                option_expansions=option_expansions,
                option_primitives=option_primitives,
                option_value_prior_rows=option_value_prior_rows,
                option_value_prior_positive_rows=option_value_prior_positive_rows,
                goal_room_conflict_prior_roots=goal_room_conflict_prior_roots,
            )

        t0 = perf_counter()
        root_previous_push = previous_push
        root_previous_pushed_box_cell = previous_pushed_box_cell
        if method == "heuristic_diagnostic":
            selected_planner_action = (
                PlannerAction(primitives=(action,), source="heuristic_primitive_push")
                if (action := choose_heuristic_action(model, state)) is not None
                else None
            )
            info = {
                "provider_counts": {"heuristic_transition_score_diagnostic": 1},
                "tree_expansions": 0,
                "root_children": len(legal),
                "root_value": None,
                "root_policy": [],
                "planner_feedback_counts": {},
                "option_candidates": 0,
                "option_expansions": 0,
                "option_primitives": 0,
            }
        else:
            if provider is None:
                raise RuntimeError(f"Method {method} requires a provider.")
            selected_planner_action, info = choose_mcts_action(
                model,
                state,
                provider,
                terminal_value,
                planner_feedback,
                num_simulations=num_simulations,
                max_depth=max_depth,
                puct_c=puct_c,
                gamma=gamma,
                root_action_selection=root_action_selection,
                visited_states=visited,
                revisit_feedback_penalty=revisit_feedback_penalty,
                action_abstraction_mode=action_abstraction_mode,
                max_option_length=max_option_length,
                max_options_per_node=max_options_per_node,
                option_prior_max_multiplier=option_prior_max_multiplier,
                option_value_prior_weight=option_value_prior_weight,
                previous_push=root_previous_push,
                previous_pushed_box_cell=root_previous_pushed_box_cell,
                route_consistency_weight=route_consistency_weight,
                audit_proxy_prior_weight=audit_proxy_prior_weight,
                goal_room_conflict_prior_weight=goal_room_conflict_prior_weight,
                learned_action_prior_payload=learned_action_prior_payload,
                learned_action_prior_weight=learned_action_prior_weight,
                learned_action_prior_max_multiplier=learned_action_prior_max_multiplier,
                rng=rng,
            )
        planning_time += perf_counter() - t0

        provider_counts.update(info["provider_counts"])
        terminal_value_counts.update(info.get("terminal_value_counts", {}))
        terminal_leaf_counts.update(info.get("terminal_leaf_counts", {}))
        planner_feedback_counts.update(info.get("planner_feedback_counts", {}))
        revisit_feedback_hits += int(info.get("revisit_feedback_hits", 0))
        option_candidates += int(info.get("option_candidates", 0))
        option_expansions += int(info.get("option_expansions", 0))
        option_primitives += int(info.get("option_primitives", 0))
        option_value_prior_rows += int(info.get("option_value_prior_rows", 0))
        option_value_prior_positive_rows += int(info.get("option_value_prior_positive_rows", 0))
        goal_room_conflict_prior_roots += int(info.get("goal_room_conflict_prior_roots", 0))
        total_expansions += int(info["tree_expansions"])
        if selected_planner_action is None:
            return episode_result(
                solved=False,
                failure_type="planner_returned_no_action",
                pushes=pushes,
                trace=trace,
                provider_counts=provider_counts,
                total_expansions=total_expansions,
                planning_time=planning_time,
                terminal_value_counts=terminal_value_counts,
                terminal_leaf_counts=terminal_leaf_counts,
                planner_feedback_counts=planner_feedback_counts,
                revisit_feedback_hits=revisit_feedback_hits,
                option_candidates=option_candidates,
                option_expansions=option_expansions,
                option_primitives=option_primitives,
                option_value_prior_rows=option_value_prior_rows,
                option_value_prior_positive_rows=option_value_prior_positive_rows,
                goal_room_conflict_prior_roots=goal_room_conflict_prior_roots,
            )

        executed_actions: list[dict[str, Any]] = []
        next_state = state
        primitives_to_execute = (
            selected_planner_action.primitives
            if option_execution_mode == "full_option"
            else selected_planner_action.primitives[:1]
        )
        for primitive in primitives_to_execute:
            if pushes >= max_pushes:
                break
            if primitive not in set(model.legal_pushes(next_state)):
                break
            pushed_box_before = next_state.boxes[primitive.box_index]
            direction = DIRECTIONS[primitive.direction]
            pushed_box_after = (pushed_box_before[0] + direction[0], pushed_box_before[1] + direction[1])
            next_state = model.step_state(next_state, primitive)
            pushes += 1
            executed_actions.append(action_to_dict(primitive))
            previous_push = primitive
            previous_pushed_box_cell = pushed_box_after
            if model.is_solved(next_state) or not model.legal_pushes(next_state):
                break
        if not executed_actions:
            return episode_result(
                solved=False,
                failure_type="planner_returned_no_executable_action",
                pushes=pushes,
                trace=trace,
                provider_counts=provider_counts,
                total_expansions=total_expansions,
                planning_time=planning_time,
                terminal_value_counts=terminal_value_counts,
                terminal_leaf_counts=terminal_leaf_counts,
                planner_feedback_counts=planner_feedback_counts,
                revisit_feedback_hits=revisit_feedback_hits,
                option_candidates=option_candidates,
                option_expansions=option_expansions,
                option_primitives=option_primitives,
                option_value_prior_rows=option_value_prior_rows,
                option_value_prior_positive_rows=option_value_prior_positive_rows,
                goal_room_conflict_prior_roots=goal_room_conflict_prior_roots,
            )
        trace.append(
            {
                "step": len(trace),
                "pushes_after_execution": pushes,
                "legal_actions": len(legal),
                "action": executed_actions[0],
                "executed_actions": executed_actions,
                "executed_action_count": len(executed_actions),
                "previous_push_before_planning": action_to_dict(root_previous_push),
                "previous_pushed_box_cell_before_planning": (
                    list(root_previous_pushed_box_cell)
                    if root_previous_pushed_box_cell is not None
                    else None
                ),
                "selected_planner_action": planner_action_to_dict(selected_planner_action),
                "tree_expansions": int(info["tree_expansions"]),
                "root_children": int(info["root_children"]),
                "root_value": info["root_value"],
                "provider_counts": info["provider_counts"],
                "terminal_value_counts": info.get("terminal_value_counts", {}),
                "terminal_leaf_counts": info.get("terminal_leaf_counts", {}),
                "planner_feedback_counts": info.get("planner_feedback_counts", {}),
                "revisit_feedback_hits": int(info.get("revisit_feedback_hits", 0)),
                "option_candidates": int(info.get("option_candidates", 0)),
                "option_expansions": int(info.get("option_expansions", 0)),
                "option_primitives": int(info.get("option_primitives", 0)),
                "option_value_prior_rows": int(info.get("option_value_prior_rows", 0)),
                "option_value_prior_positive_rows": int(
                    info.get("option_value_prior_positive_rows", 0)
                ),
                "option_value_prior_max": info.get("option_value_prior_max"),
                "option_value_prior_mean": info.get("option_value_prior_mean"),
                "goal_room_conflict_prior_roots": int(info.get("goal_room_conflict_prior_roots", 0)),
                "best_planner_action": info.get("best_planner_action"),
                "root_action_selection": info.get("root_action_selection"),
                "root_policy": info["root_policy"][:8],
            }
        )
        state = next_state
        visited[state] += 1
        if visited[state] > max_state_revisits:
            return episode_result(
                solved=False,
                failure_type="state_revisit_limit",
                pushes=pushes,
                trace=trace,
                provider_counts=provider_counts,
                total_expansions=total_expansions,
                planning_time=planning_time,
                terminal_value_counts=terminal_value_counts,
                terminal_leaf_counts=terminal_leaf_counts,
                planner_feedback_counts=planner_feedback_counts,
                revisit_feedback_hits=revisit_feedback_hits,
                option_candidates=option_candidates,
                option_expansions=option_expansions,
                option_primitives=option_primitives,
                option_value_prior_rows=option_value_prior_rows,
                option_value_prior_positive_rows=option_value_prior_positive_rows,
                goal_room_conflict_prior_roots=goal_room_conflict_prior_roots,
            )

    return episode_result(
        solved=model.is_solved(state),
        failure_type="solved" if model.is_solved(state) else "max_pushes",
        pushes=max_pushes,
        trace=trace,
        provider_counts=provider_counts,
        total_expansions=total_expansions,
        planning_time=planning_time,
        terminal_value_counts=terminal_value_counts,
        terminal_leaf_counts=terminal_leaf_counts,
        planner_feedback_counts=planner_feedback_counts,
        revisit_feedback_hits=revisit_feedback_hits,
        option_candidates=option_candidates,
        option_expansions=option_expansions,
        option_primitives=option_primitives,
        option_value_prior_rows=option_value_prior_rows,
        option_value_prior_positive_rows=option_value_prior_positive_rows,
        goal_room_conflict_prior_roots=goal_room_conflict_prior_roots,
    )


def run_best_first_diagnostic_episode(
    model: SokobanGoalDirectedSDModel,
    *,
    max_expansions: int,
) -> Dict[str, Any]:
    t0 = perf_counter()
    result = best_first_search(
        model,
        max_expansions=max_expansions,
        heuristic_weight=1.0,
        prune_deadlocks=True,
    )
    elapsed = perf_counter() - t0
    failure_type = "solved" if result.solved else "max_expansions"
    return {
        "solved": bool(result.solved),
        "failure_type": failure_type,
        "pushes": int(result.solution_pushes) if result.solution_pushes is not None else None,
        "trace_len": 0,
        "provider_counts": {"heuristic_best_first_diagnostic": int(result.expanded)},
        "terminal_value_counts": {},
        "terminal_leaf_counts": {},
        "planner_feedback_counts": {},
        "revisit_feedback_hits": 0,
        "total_tree_expansions": int(result.expanded),
        "planning_time_sec": float(elapsed),
        "best_first": result.to_dict(),
        "trace_sample": [],
    }


def episode_result(
    *,
    solved: bool,
    failure_type: str,
    pushes: int,
    trace: list[dict[str, Any]],
    provider_counts: Counter[str],
    total_expansions: int,
    planning_time: float,
    terminal_value_counts: Counter[str] | None = None,
    terminal_leaf_counts: Counter[str] | None = None,
    planner_feedback_counts: Counter[str] | None = None,
    revisit_feedback_hits: int = 0,
    option_candidates: int = 0,
    option_expansions: int = 0,
    option_primitives: int = 0,
    option_value_prior_rows: int = 0,
    option_value_prior_positive_rows: int = 0,
    goal_room_conflict_prior_roots: int = 0,
) -> Dict[str, Any]:
    terminal_value_counts = terminal_value_counts or Counter()
    terminal_leaf_counts = terminal_leaf_counts or Counter()
    planner_feedback_counts = planner_feedback_counts or Counter()
    if len(trace) <= 10:
        trace_sample = trace
    else:
        trace_sample = trace[:5] + trace[-5:]
    return {
        "solved": bool(solved),
        "failure_type": failure_type,
        "pushes": int(pushes),
        "trace_len": len(trace),
        "provider_counts": dict(provider_counts),
        "terminal_value_counts": dict(terminal_value_counts),
        "terminal_leaf_counts": dict(terminal_leaf_counts),
        "planner_feedback_counts": dict(planner_feedback_counts),
        "revisit_feedback_hits": int(revisit_feedback_hits),
        "option_candidates": int(option_candidates),
        "option_expansions": int(option_expansions),
        "option_primitives": int(option_primitives),
        "option_value_prior_rows": int(option_value_prior_rows),
        "option_value_prior_positive_rows": int(option_value_prior_positive_rows),
        "goal_room_conflict_prior_roots": int(goal_room_conflict_prior_roots),
        "option_expansion_fraction": _safe_div(option_expansions, total_expansions),
        "total_tree_expansions": int(total_expansions),
        "planning_time_sec": float(planning_time),
        "trace_sample": trace_sample,
    }


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["budget"], row["seed"])].append(row)

    by_run: list[dict[str, Any]] = []
    for (method, budget, seed), vals in sorted(groups.items()):
        solved = [row["episode"]["solved"] for row in vals]
        pushes = [row["episode"]["pushes"] for row in vals]
        solved_pushes = [row["episode"]["pushes"] for row in vals if row["episode"]["solved"]]
        failures = Counter(row["episode"]["failure_type"] for row in vals)
        provider_counts: Counter[str] = Counter()
        terminal_value_counts: Counter[str] = Counter()
        terminal_leaf_counts: Counter[str] = Counter()
        planner_feedback_counts: Counter[str] = Counter()
        revisit_feedback_hits = 0
        option_candidates = 0
        option_expansions = 0
        option_primitives = 0
        option_value_prior_rows = 0
        option_value_prior_positive_rows = 0
        goal_room_conflict_prior_roots = 0
        for row in vals:
            provider_counts.update(row["episode"]["provider_counts"])
            terminal_value_counts.update(row["episode"].get("terminal_value_counts", {}))
            terminal_leaf_counts.update(row["episode"].get("terminal_leaf_counts", {}))
            planner_feedback_counts.update(row["episode"].get("planner_feedback_counts", {}))
            revisit_feedback_hits += int(row["episode"].get("revisit_feedback_hits", 0))
            option_candidates += int(row["episode"].get("option_candidates", 0))
            option_expansions += int(row["episode"].get("option_expansions", 0))
            option_primitives += int(row["episode"].get("option_primitives", 0))
            option_value_prior_rows += int(row["episode"].get("option_value_prior_rows", 0))
            option_value_prior_positive_rows += int(
                row["episode"].get("option_value_prior_positive_rows", 0)
            )
            goal_room_conflict_prior_roots += int(
                row["episode"].get("goal_room_conflict_prior_roots", 0)
            )
        by_run.append(
            {
                "method": method,
                "budget": int(budget),
                "seed": int(seed),
                "episodes": len(vals),
                "success_rate": float(np.mean(solved)) if solved else 0.0,
                "mean_pushes": float(np.mean(pushes)) if pushes else None,
                "mean_solution_pushes": float(np.mean(solved_pushes)) if solved_pushes else None,
                "failure_counts": dict(failures),
                "provider_counts": dict(provider_counts),
                "terminal_value_counts": dict(terminal_value_counts),
                "terminal_leaf_counts": dict(terminal_leaf_counts),
                "planner_feedback_counts": dict(planner_feedback_counts),
                "revisit_feedback_hits": int(revisit_feedback_hits),
                "option_candidates": int(option_candidates),
                "option_expansions": int(option_expansions),
                "option_primitives": int(option_primitives),
                "option_value_prior_rows": int(option_value_prior_rows),
                "option_value_prior_positive_rows": int(option_value_prior_positive_rows),
                "goal_room_conflict_prior_roots": int(goal_room_conflict_prior_roots),
                "option_expansion_fraction": _safe_div(
                    option_expansions,
                    sum(row["episode"]["total_tree_expansions"] for row in vals),
                ),
                "mean_tree_expansions": float(
                    np.mean([row["episode"]["total_tree_expansions"] for row in vals])
                )
                if vals
                else None,
                "mean_planning_time_sec": float(
                    np.mean([row["episode"]["planning_time_sec"] for row in vals])
                )
                if vals
                else None,
            }
        )

    method_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in by_run:
        method_groups[(row["method"], row["budget"])].append(row)
    by_method_budget = []
    for (method, budget), vals in sorted(method_groups.items()):
        by_method_budget.append(
            {
                "method": method,
                "budget": int(budget),
                "seeds": [row["seed"] for row in vals],
                "success_rate_mean": float(np.mean([row["success_rate"] for row in vals])),
                "success_rate_std": float(np.std([row["success_rate"] for row in vals])),
                "mean_pushes_mean": float(np.mean([row["mean_pushes"] for row in vals])),
                "mean_solution_pushes_mean": (
                    float(np.mean([row["mean_solution_pushes"] for row in vals if row["mean_solution_pushes"] is not None]))
                    if any(row["mean_solution_pushes"] is not None for row in vals)
                    else None
                ),
                "mean_planning_time_sec": float(np.mean([row["mean_planning_time_sec"] for row in vals])),
            }
        )
    return {
        "by_run": by_run,
        "by_method_budget": by_method_budget,
    }


def validate_provider_usage(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    failed: list[str] = []
    learned_rows = [row for row in rows if row["method"] == "learned_sdmcts"]
    if learned_rows:
        total = 0
        learned = 0
        for row in learned_rows:
            counts = row["episode"]["provider_counts"]
            total += sum(counts.values())
            learned += counts.get("learned_stock_flow_checkpoint_v1", 0)
        learned_fraction = float(learned / total) if total else 0.0
        if learned_fraction < 1.0:
            failed.append("learned_provider_fraction")
    else:
        learned_fraction = None
        return {
            "verdict": "NOT_APPLICABLE",
            "failed_checks": [],
            "observed": {
                "learned_provider_fraction": None,
                "learned_rows": 0,
            },
            "thresholds": {
                "learned_provider_fraction_min": 1.0,
                "learned_rows_min": 1,
            },
        }

    return {
        "verdict": "PASS" if not failed else "PARTIAL",
        "failed_checks": failed,
        "observed": {
            "learned_provider_fraction": learned_fraction,
            "learned_rows": len(learned_rows),
        },
        "thresholds": {
            "learned_provider_fraction_min": 1.0,
            "learned_rows_min": 1,
        },
    }


def validate_terminal_value_usage(
    rows: Sequence[Dict[str, Any]],
    terminal_value: TerminalValueEvaluator,
    mode: str,
) -> Dict[str, Any]:
    if mode != "tva":
        return {
            "verdict": "NOT_APPLICABLE",
            "failed_checks": [],
            "observed": {
                "terminal_value_mode": mode,
                "tva_evaluations": None,
            },
            "thresholds": {
                "tva_evaluations_min": 1,
            },
        }

    total = 0
    for row in rows:
        if row["method"] in {"heuristic_diagnostic", "heuristic_best_first_diagnostic"}:
            continue
        total += row["episode"].get("terminal_value_counts", {}).get(terminal_value.evaluator_id, 0)
    failed = [] if total >= 1 else ["tva_evaluations"]
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "observed": {
            "terminal_value_mode": mode,
            "tva_provider_id": terminal_value.evaluator_id,
            "tva_evaluations": int(total),
        },
        "thresholds": {
            "tva_evaluations_min": 1,
        },
    }


def validate_planner_feedback_usage(
    rows: Sequence[Dict[str, Any]],
    planner_feedback: PlannerFeedback,
    mode: str,
) -> Dict[str, Any]:
    total = 0
    for row in rows:
        if row["method"] in {"heuristic_diagnostic", "heuristic_best_first_diagnostic"}:
            continue
        total += row["episode"].get("planner_feedback_counts", {}).get(planner_feedback.feedback_id, 0)
    failed = [] if total >= 1 else ["planner_feedback_evaluations"]
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "observed": {
            "planner_feedback_mode": mode,
            "planner_feedback_id": planner_feedback.feedback_id,
            "planner_feedback_evaluations": int(total),
        },
        "thresholds": {
            "planner_feedback_evaluations_min": 1,
        },
    }


def validate_action_abstraction_usage(
    rows: Sequence[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    if mode == "push":
        return {
            "verdict": "NOT_APPLICABLE",
            "failed_checks": [],
            "observed": {
                "action_abstraction_mode": mode,
                "option_candidates": 0,
                "option_expansions": 0,
                "option_primitives": 0,
            },
            "thresholds": {
                "option_candidates_min": None,
                "option_expansions_min": None,
            },
        }
    option_candidates = 0
    option_expansions = 0
    option_primitives = 0
    prior_scored_roots = 0
    for row in rows:
        if row["method"] in {"heuristic_diagnostic", "heuristic_best_first_diagnostic"}:
            continue
        episode = row["episode"]
        option_candidates += int(episode.get("option_candidates", 0))
        option_expansions += int(episode.get("option_expansions", 0))
        option_primitives += int(episode.get("option_primitives", 0))
        for trace_row in episode.get("trace_sample", []):
            for policy_row in trace_row.get("root_policy", []):
                action = policy_row.get("action", {})
                if action.get("source") == "primitive_push_with_target2_prior" and float(
                    action.get("structural_score", 0.0)
                ) > 0.0:
                    prior_scored_roots += 1
                    break
    failed = []
    if mode == "target2_prior":
        if prior_scored_roots < 1:
            failed.append("target2_prior_scored_roots")
    else:
        if option_candidates < 1:
            failed.append("option_candidates")
        if option_expansions < 1:
            failed.append("option_expansions")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "observed": {
            "action_abstraction_mode": mode,
            "option_candidates": int(option_candidates),
            "option_expansions": int(option_expansions),
            "option_primitives": int(option_primitives),
            "target2_prior_scored_roots": int(prior_scored_roots),
        },
        "thresholds": {
            "option_candidates_min": None if mode == "target2_prior" else 1,
            "option_expansions_min": None if mode == "target2_prior" else 1,
            "target2_prior_scored_roots_min": 1 if mode == "target2_prior" else None,
        },
    }


def validate_option_value_prior_usage(
    rows: Sequence[Dict[str, Any]],
    *,
    weight: float,
) -> Dict[str, Any]:
    if weight <= 0.0:
        return {
            "verdict": "NOT_APPLICABLE",
            "failed_checks": [],
            "observed": {
                "option_value_prior_enabled": False,
                "option_value_prior_rows": 0,
                "option_value_prior_positive_rows": 0,
            },
            "thresholds": {
                "option_value_prior_rows_min": None,
            },
        }
    rows_total = 0
    positive_total = 0
    for row in rows:
        if row["method"] in {"heuristic_diagnostic", "heuristic_best_first_diagnostic"}:
            continue
        episode = row["episode"]
        rows_total += int(episode.get("option_value_prior_rows", 0))
        positive_total += int(episode.get("option_value_prior_positive_rows", 0))
    failed = [] if rows_total >= 1 else ["option_value_prior_rows"]
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "observed": {
            "option_value_prior_enabled": True,
            "option_value_prior_rows": int(rows_total),
            "option_value_prior_positive_rows": int(positive_total),
        },
        "thresholds": {
            "option_value_prior_rows_min": 1,
        },
    }


def validate_learned_action_prior_usage(
    rows: Sequence[Dict[str, Any]],
    *,
    checkpoint: str,
    weight: float,
) -> Dict[str, Any]:
    if not checkpoint or weight <= 0.0:
        return {
            "verdict": "NOT_APPLICABLE",
            "failed_checks": [],
            "observed": {
                "learned_action_prior_enabled": False,
                "learned_action_prior_root_policy_rows": 0,
            },
            "thresholds": {
                "learned_action_prior_root_policy_rows_min": None,
            },
        }
    total_rows = 0
    for row in rows:
        if row["method"] in {"heuristic_diagnostic", "heuristic_best_first_diagnostic"}:
            continue
        for trace_row in row["episode"].get("trace_sample", []):
            total_rows += len(trace_row.get("root_policy", []))
    failed = [] if total_rows >= 1 else ["learned_action_prior_root_policy_rows"]
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "observed": {
            "learned_action_prior_enabled": True,
            "learned_action_prior_root_policy_rows": int(total_rows),
        },
        "thresholds": {
            "learned_action_prior_root_policy_rows_min": 1,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_csv_strings(args.methods, name="--methods")
    budgets = parse_csv_ints(args.budgets, name="--budgets")
    seeds = parse_csv_ints(args.seeds, name="--seeds")
    learned_provider = LearnedStockFlowTransitionProvider(Path(args.checkpoint))
    terminal_value = make_terminal_value_evaluator(args)
    planner_feedback = make_planner_feedback(args)
    learned_action_prior_payload = (
        load_action_prior_checkpoint(Path(args.learned_action_prior_checkpoint))
        if args.learned_action_prior_checkpoint and args.learned_action_prior_weight > 0.0
        else None
    )
    levels, selection = select_levels(
        learned_provider,
        levels_path=args.levels,
        max_levels=args.max_levels,
        level_offset=args.level_offset,
        use_checkpoint_valid_levels=args.use_checkpoint_valid_levels,
    )
    levels = levels[: args.max_eval_levels]
    if not levels:
        raise SystemExit("No levels selected for planning evaluation.")

    rows: list[dict[str, Any]] = []
    t0 = perf_counter()
    for method in methods:
        provider = make_provider(method, learned_provider)
        for budget in budgets:
            for seed in seeds:
                rng = np.random.RandomState(seed)
                torch.manual_seed(seed)
                for level in levels:
                    model = SokobanGoalDirectedSDModel(level)
                    if method == "heuristic_best_first_diagnostic":
                        episode = run_best_first_diagnostic_episode(
                            model,
                            max_expansions=budget,
                        )
                    else:
                        episode = run_episode(
                            model,
                            method=method,
                            provider=provider,
                            terminal_value=terminal_value,
                            planner_feedback=planner_feedback,
                            num_simulations=budget,
                            max_depth=args.mcts_depth,
                            puct_c=args.puct_c,
                            gamma=args.gamma,
                            root_action_selection=args.root_action_selection,
                            max_pushes=args.max_pushes,
                            max_state_revisits=args.max_state_revisits,
                            revisit_feedback_penalty=args.revisit_feedback_penalty,
                            action_abstraction_mode=args.action_abstraction_mode,
                            max_option_length=args.max_option_length,
                            max_options_per_node=args.max_options_per_node,
                            option_prior_max_multiplier=args.option_prior_max_multiplier,
                            option_value_prior_weight=args.option_value_prior_weight,
                            option_execution_mode=args.option_execution_mode,
                            route_consistency_weight=args.route_consistency_weight,
                            audit_proxy_prior_weight=args.audit_proxy_prior_weight,
                            goal_room_conflict_prior_weight=args.goal_room_conflict_prior_weight,
                            learned_action_prior_payload=learned_action_prior_payload,
                            learned_action_prior_weight=args.learned_action_prior_weight,
                            learned_action_prior_max_multiplier=args.learned_action_prior_max_multiplier,
                            rng=rng,
                        )
                    rows.append(
                        {
                            "method": method,
                            "budget": int(budget),
                            "seed": int(seed),
                            "level_id": level.level_id,
                            "n_boxes": level.n_boxes,
                            "episode": episode,
                        }
                    )

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_stage3_sdmcts_planning_smoke_v1",
        },
        "config": {
            "checkpoint": os.path.relpath(args.checkpoint, ROOT)
            if Path(args.checkpoint).is_absolute()
            else args.checkpoint,
            "levels": os.path.relpath(args.levels, ROOT)
            if Path(args.levels).is_absolute()
            else args.levels,
            "level_selection": selection,
            "max_eval_levels": args.max_eval_levels,
            "methods": methods,
            "budgets": budgets,
            "seeds": seeds,
            "mcts_depth": args.mcts_depth,
            "puct_c": args.puct_c,
            "gamma": args.gamma,
            "max_pushes": args.max_pushes,
            "max_state_revisits": args.max_state_revisits,
            "root_action_selection": args.root_action_selection,
            "terminal_value_mode": args.terminal_value_mode,
            "terminal_value_provider": terminal_value.evaluator_id,
            "planner_feedback_mode": args.planner_feedback_mode,
            "planner_feedback_provider": planner_feedback.feedback_id,
            "expert_delta_weight": args.expert_delta_weight,
            "expert_prior_temperature": args.expert_prior_temperature,
            "expert_prior_uniform_mix": args.expert_prior_uniform_mix,
            "revisit_feedback_penalty": args.revisit_feedback_penalty,
            "expert_viability_penalty": args.expert_viability_penalty,
            "goal_room_order_weight": args.goal_room_order_weight,
            "action_abstraction_mode": args.action_abstraction_mode,
            "max_option_length": args.max_option_length,
            "max_options_per_node": args.max_options_per_node,
            "option_prior_max_multiplier": args.option_prior_max_multiplier,
            "option_value_prior_weight": args.option_value_prior_weight,
            "option_execution_mode": args.option_execution_mode,
            "route_consistency_weight": args.route_consistency_weight,
            "audit_proxy_prior_weight": args.audit_proxy_prior_weight,
            "goal_room_conflict_prior_weight": args.goal_room_conflict_prior_weight,
            "learned_action_prior_checkpoint": relative_path(args.learned_action_prior_checkpoint, ROOT)
            if args.learned_action_prior_checkpoint
            else "",
            "learned_action_prior_provider": (
                str(
                    learned_action_prior_payload.get("checkpoint_contract", {}).get(
                        "prior_id",
                        learned_action_prior_payload.get("checkpoint_contract", {}).get("value_provider_id", ""),
                    )
                )
                if learned_action_prior_payload is not None
                else ""
            ),
            "learned_action_prior_weight": args.learned_action_prior_weight,
            "learned_action_prior_max_multiplier": args.learned_action_prior_max_multiplier,
            "tva_checkpoint": relative_path(args.tva_checkpoint, ROOT) if args.tva_checkpoint else "",
        },
        "planning_contract": {
            "closed_loop_receding_horizon": True,
            "raw_state_used_for_action_support": True,
            "learned_sdmcts_abstract_transition_provider": "learned_stock_flow_checkpoint_v1",
            "zero_flow_ablation_provider": "zero_flow_no_learned_transition_ablation",
            "oracle_abstract_provider": "oracle_abstract_stock_transition_for_diagnostic_upper_bound",
            "heuristic_diagnostic_is_not_sdmcts": True,
            "heuristic_best_first_diagnostic_is_not_sdmcts": True,
            "terminal_value_leaf_evaluator": terminal_value.evaluator_id,
            "planner_feedback": planner_feedback.feedback_id,
            "planner_feedback_changes_transition_provider": False,
            "expert_feedback_teacher_used_online": False,
            "expert_feedback_best_first_used_online": False,
            "expert_feedback_exact_successor_used_for_structural_prior_only": args.planner_feedback_mode == "expert",
            "expert_feedback_revisit_penalty_uses_closed_loop_state_counter": args.planner_feedback_mode == "expert",
            "expert_feedback_viability_penalty_uses_abstract_features": args.planner_feedback_mode == "expert",
            "expert_feedback_goal_room_order_prior_used": args.goal_room_order_weight > 0.0,
            "expert_feedback_goal_room_order_changes_transition_provider": False,
            "action_abstraction_mode": args.action_abstraction_mode,
            "option_provider_id": (
                "target2_local_structural_option_generator_v1"
                if args.action_abstraction_mode == "target2"
                else (
                    "target2_local_structural_prior_v1"
                    if args.action_abstraction_mode == "target2_prior"
                    else "primitive_push_only"
                )
            ),
            "option_primitive_transition_provider": (
                "learned_stock_flow_checkpoint_v1"
                if args.action_abstraction_mode == "target2"
                else (
                    "not_applicable_prior_only"
                    if args.action_abstraction_mode == "target2_prior"
                    else "primitive_push_only"
                )
            ),
            "option_search_budget": 0,
            "option_best_first_used_online": False,
            "option_exact_abstract_transition_used_online": False,
            "option_raw_state_exact_step_used_for_support_only": args.action_abstraction_mode == "target2",
            "option_value_prior_used": args.option_value_prior_weight > 0.0,
            "option_value_prior_source": "local_sd_stock_rollout_return_no_teacher",
            "option_value_prior_changes_transition_provider": False,
            "option_value_prior_best_first_used_online": False,
            "option_value_prior_teacher_used_online": False,
            "option_execution_mode": args.option_execution_mode,
            "route_consistency_prior_used": args.route_consistency_weight > 0.0,
            "route_consistency_prior_changes_transition_provider": False,
            "audit_proxy_prior_used": args.audit_proxy_prior_weight > 0.0,
            "audit_proxy_prior_changes_transition_provider": False,
            "audit_proxy_prior_best_first_used_online": False,
            "goal_room_conflict_prior_used": args.goal_room_conflict_prior_weight > 0.0,
            "goal_room_conflict_prior_root_only": True,
            "goal_room_conflict_prior_changes_transition_provider": False,
            "goal_room_conflict_prior_best_first_used_online": False,
            "goal_room_conflict_prior_changes_reward": False,
            "learned_action_prior_used": learned_action_prior_payload is not None,
            "learned_action_prior_changes_transition_provider": False,
            "learned_action_prior_best_first_used_online": False,
            "learned_action_prior_teacher_used_online": False,
            "learned_action_prior_planner_consumption": "action_prior_multiplier_only",
            "tva_teacher_used_online": False,
            "stock_names": STOCK_NAMES,
            "flow_names": FLOW_NAMES,
        },
        "aggregate": aggregate(rows),
        "provider_usage_gate": validate_provider_usage(rows),
        "terminal_value_usage_gate": validate_terminal_value_usage(
            rows,
            terminal_value,
            args.terminal_value_mode,
        ),
        "planner_feedback_usage_gate": validate_planner_feedback_usage(
            rows,
            planner_feedback,
            args.planner_feedback_mode,
        ),
        "action_abstraction_usage_gate": validate_action_abstraction_usage(
            rows,
            args.action_abstraction_mode,
        ),
        "option_value_prior_usage_gate": validate_option_value_prior_usage(
            rows,
            weight=args.option_value_prior_weight,
        ),
        "learned_action_prior_usage_gate": validate_learned_action_prior_usage(
            rows,
            checkpoint=args.learned_action_prior_checkpoint,
            weight=args.learned_action_prior_weight,
        ),
        "episodes": rows,
        "elapsed_sec": float(perf_counter() - t0),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "aggregate": summary["aggregate"]["by_method_budget"],
        "provider_usage_gate": summary["provider_usage_gate"],
        "terminal_value_usage_gate": summary["terminal_value_usage_gate"],
        "planner_feedback_usage_gate": summary["planner_feedback_usage_gate"],
        "action_abstraction_usage_gate": summary["action_abstraction_usage_gate"],
        "option_value_prior_usage_gate": summary["option_value_prior_usage_gate"],
        "learned_action_prior_usage_gate": summary["learned_action_prior_usage_gate"],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
