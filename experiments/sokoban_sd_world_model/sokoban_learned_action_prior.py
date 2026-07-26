"""Learned action-prior utilities for Sokoban SD-MCTS.

The prior is a planner-consumption signal: it scores candidate primitive
pushes from the current abstract SD state and immediate action consequences.
It does not generate actions and does not modify stock-flow transitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from sokoban_tva import TVA_FEATURE_NAMES, tva_feature_vector
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState
from src.sokoban.env import ACTION_NAMES, DIRECTIONS
from src.sokoban.sd_model import SokobanPushAction
from validate_sokoban_sd_design import STOCK_NAMES, stock_proxy


ACTION_ONEHOT_FEATURE_NAMES = tuple(f"action:direction:{name}" for name in ACTION_NAMES)
ACTION_SCALAR_FEATURE_NAMES = (
    "action:box_index_norm",
    "action:flat_index_norm",
    "action:box_on_goal_before",
    "action:target_on_goal_after",
    "action:target_dead_square_after",
    "action:target_floor_degree_norm",
    "action:target_goal_neighbor_degree_norm",
)
ACTION_AFTER_STOCK_FEATURE_NAMES = tuple(f"after_stock:{name}" for name in STOCK_NAMES)
ACTION_STOCK_DELTA_FEATURE_NAMES = tuple(f"delta_stock:{name}" for name in STOCK_NAMES)
ACTION_CONSEQUENCE_FEATURE_NAMES = (
    "consequence:boxes_on_goals_delta_norm",
    "consequence:matching_improvement",
    "consequence:reachable_delta_norm",
    "consequence:legal_push_delta_norm",
    "consequence:viable_push_delta_norm",
    "consequence:dead_square_box_delta_norm",
    "consequence:frozen_box_delta_norm",
    "consequence:solved_after",
    "consequence:deadlocked_after",
    "consequence:no_legal_pushes_after",
    "consequence:goal_room_order_bonus",
    "consequence:goal_entry_delay_penalty",
    "consequence:target2_structural_score",
)
ACTION_PRIOR_FEATURE_NAMES = (
    TVA_FEATURE_NAMES
    + ACTION_ONEHOT_FEATURE_NAMES
    + ACTION_SCALAR_FEATURE_NAMES
    + ACTION_AFTER_STOCK_FEATURE_NAMES
    + ACTION_STOCK_DELTA_FEATURE_NAMES
    + ACTION_CONSEQUENCE_FEATURE_NAMES
)


class SokobanActionPriorNet(nn.Module):
    """Small binary scorer used only to multiply SD-MCTS action priors."""

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _floor_neighbors(model: SokobanGoalDirectedSDModel, cell: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for direction in DIRECTIONS:
        nxt = (cell[0] + direction[0], cell[1] + direction[1])
        if model.level.is_floor(nxt):
            out.append(nxt)
    return out


def _goal_neighbor_degree(model: SokobanGoalDirectedSDModel, cell: tuple[int, int]) -> int:
    goals = set(model.level.goals)
    return sum(1 for neigh in _floor_neighbors(model, cell) if neigh in goals)


def _bounded_matching_improvement(
    model: SokobanGoalDirectedSDModel,
    before: float,
    after: float,
) -> float:
    scale = max(model.level.n_boxes * (model.level.height + model.level.width), 1)
    if np.isfinite(before) and np.isfinite(after):
        return _clamp((before - after) / scale, -1.0, 1.0)
    if not np.isfinite(before) and np.isfinite(after):
        return 1.0
    if np.isfinite(before) and not np.isfinite(after):
        return -1.0
    return 0.0


def _goal_room_order_bonus(
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


def _goal_entry_delay_penalty(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    action: SokobanPushAction,
) -> float:
    if action.box_index < 0 or action.box_index >= len(raw_state.boxes):
        return 0.0
    goals = set(model.level.goals)
    before_cell = raw_state.boxes[action.box_index]
    after_cell = (
        before_cell[0] + DIRECTIONS[action.direction][0],
        before_cell[1] + DIRECTIONS[action.direction][1],
    )
    if before_cell in goals or after_cell not in goals:
        return 0.0
    for candidate in model.legal_pushes(raw_state):
        candidate_box = raw_state.boxes[candidate.box_index]
        candidate_after = (
            candidate_box[0] + DIRECTIONS[candidate.direction][0],
            candidate_box[1] + DIRECTIONS[candidate.direction][1],
        )
        if _goal_room_order_bonus(model, candidate_box, candidate_after) > 0.0:
            return 1.0
    return 0.0


def action_prior_feature_vector(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    action: SokobanPushAction,
    *,
    target2_structural_score: float = 0.0,
) -> np.ndarray:
    """Feature vector for a root/action prior sample.

    The vector joins the current SD abstract state with immediate action
    consequences. Exact raw stepping is used only to form action-support
    features for the candidate action; transition rollout remains supplied by
    the selected SD transition provider in SD-MCTS.
    """

    stock_arr = np.asarray(stock, dtype=np.float32)
    state_features = tva_feature_vector(model, raw_state, stock_arr)
    before = model.abstract_features(raw_state)
    next_state = model.step_state(raw_state, action)
    after = model.abstract_features(next_state)
    next_stock = stock_proxy(model, next_state).as_array().astype(np.float32)
    stock_delta = next_stock - stock_arr

    direction_onehot = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    if 0 <= action.direction < len(ACTION_NAMES):
        direction_onehot[action.direction] = 1.0

    n_boxes = max(model.level.n_boxes, 1)
    max_actions = max(model.level.n_boxes * len(ACTION_NAMES) - 1, 1)
    floor_degree_scale = max(len(DIRECTIONS), 1)
    goal_degree_scale = max(len(DIRECTIONS), 1)
    floor_count = max(len(model.floor_cells), 1)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    goals = set(model.level.goals)

    if action.box_index < len(raw_state.boxes):
        before_cell = raw_state.boxes[action.box_index]
        after_cell = (
            before_cell[0] + DIRECTIONS[action.direction][0],
            before_cell[1] + DIRECTIONS[action.direction][1],
        )
        target_floor_degree = len(_floor_neighbors(model, after_cell)) if model.level.is_floor(after_cell) else 0
        target_goal_degree = _goal_neighbor_degree(model, after_cell) if model.level.is_floor(after_cell) else 0
        box_on_goal_before = float(before_cell in goals)
        target_on_goal_after = float(after_cell in goals)
        target_dead_square_after = float(after_cell in model.dead_squares)
        goal_room_order = _goal_room_order_bonus(model, before_cell, after_cell)
    else:
        target_floor_degree = 0
        target_goal_degree = 0
        box_on_goal_before = 0.0
        target_on_goal_after = 0.0
        target_dead_square_after = 0.0
        goal_room_order = 0.0

    action_scalars = np.array(
        [
            _safe_div(action.box_index, max(n_boxes - 1, 1)),
            _safe_div(action.flat_index, max_actions),
            box_on_goal_before,
            target_on_goal_after,
            target_dead_square_after,
            _safe_div(target_floor_degree, floor_degree_scale),
            _safe_div(target_goal_degree, goal_degree_scale),
        ],
        dtype=np.float32,
    )
    consequences = np.array(
        [
            _safe_div(after.boxes_on_goals - before.boxes_on_goals, n_boxes),
            _bounded_matching_improvement(model, before.matching_lower_bound, after.matching_lower_bound),
            _safe_div(after.reachable_count - before.reachable_count, floor_count),
            _safe_div(after.legal_push_count - before.legal_push_count, max_pushes),
            _safe_div(after.viable_push_count - before.viable_push_count, max_pushes),
            _safe_div(after.dead_square_boxes - before.dead_square_boxes, n_boxes),
            _safe_div(after.frozen_boxes - before.frozen_boxes, n_boxes),
            float(after.solved),
            float(after.deadlocked),
            float(after.legal_push_count == 0),
            goal_room_order,
            _goal_entry_delay_penalty(model, raw_state, action),
            float(target2_structural_score),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            state_features,
            direction_onehot,
            action_scalars,
            next_stock,
            stock_delta,
            consequences,
        ],
        axis=0,
    ).astype(np.float32)


def load_action_prior_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    expected = tuple(payload["feature_names"])
    if expected != ACTION_PRIOR_FEATURE_NAMES:
        raise ValueError(
            "Action-prior feature contract mismatch: "
            f"checkpoint has {len(expected)} features, code expects {len(ACTION_PRIOR_FEATURE_NAMES)}."
        )
    net = SokobanActionPriorNet(
        input_dim=int(payload["architecture"]["input_dim"]),
        hidden_dim=int(payload["architecture"]["hidden_dim"]),
    ).to(device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    payload["model"] = net
    return payload


def predict_action_prior_logit(
    payload: Dict[str, Any],
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    action: SokobanPushAction,
    *,
    target2_structural_score: float = 0.0,
    device: str | torch.device = "cpu",
) -> float:
    features = action_prior_feature_vector(
        model,
        raw_state,
        stock,
        action,
        target2_structural_score=target2_structural_score,
    )
    mean = np.asarray(payload["feature_stats"]["mean"], dtype=np.float32)
    std = np.asarray(payload["feature_stats"]["std"], dtype=np.float32)
    normalized = (features - mean) / np.maximum(std, 1e-6)
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        logit = payload["model"](tensor)
    return float(logit.detach().cpu().view(-1)[0].item())


def feature_summary(feature_matrix: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if feature_matrix.size == 0:
        return {}
    return {
        name: {
            "mean": float(np.mean(feature_matrix[:, idx])),
            "std": float(np.std(feature_matrix[:, idx])),
            "min": float(np.min(feature_matrix[:, idx])),
            "max": float(np.max(feature_matrix[:, idx])),
        }
        for idx, name in enumerate(feature_names)
    }
