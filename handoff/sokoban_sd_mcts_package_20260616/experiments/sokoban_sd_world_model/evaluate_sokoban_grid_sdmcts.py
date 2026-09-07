#!/usr/bin/env python3
"""Closed-loop exact-grid SD-MCTS evaluation for Sokoban.

This is the revised Sokoban planner entry point. It expands MCTS on exact raw
Sokoban states using the grid stock-flow SD provider, and optionally uses a TVA
checkpoint that was produced by the independent TVA audit. The teacher is never
called online in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import SokobanGridStockFlowModel, SokobanPushAction, SokobanState, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES, state_to_observation

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_sokoban_grid_tva import grid_tva_features


@dataclass
class TvaCheckpoint:
    path: str
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    feature_dim: int
    observation_shape: tuple[int, int, int] | None
    value_scale: dict[str, float]
    target_mode: str = "value"
    model_type: str = "ridge"
    layers: list | None = None


@dataclass
class CandidateAction:
    action: SokobanPushAction
    child_state: SokobanState
    prior: float = 0.0
    prior_score: float = 0.0


@dataclass
class EdgeStats:
    action: SokobanPushAction
    child_state: SokobanState
    child_node: "MCTSNode"
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    prior_score: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class MCTSNode:
    state: SokobanState
    depth_from_root: int
    untried: list[CandidateAction] | None = None
    children: dict[int, EdgeStats] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    pruned_on_init: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory.",
    )
    parser.add_argument(
        "--level-offset",
        type=int,
        default=0,
        help="Skip this many loaded levels before evaluation; used for held-out slices.",
    )
    parser.add_argument("--max-levels", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--leaf-value", choices=("zero", "tva", "sd_heuristic"), default="zero")
    parser.add_argument(
        "--tva-checkpoint",
        default="results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json",
        help="Checkpoint from audit_sokoban_grid_tva.py, required when --leaf-value=tva.",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=("none", "tier_a", "exact_deadlock"),
        default="tier_a",
        help="Constraint feedback used to prune impossible child states.",
    )
    parser.add_argument("--select-by", choices=("visits", "q"), default="visits")
    parser.add_argument(
        "--revisit-penalty",
        type=float,
        default=0.0,
        help="Reward penalty for transitions into states already visited during execution.",
    )
    parser.add_argument(
        "--action-prior",
        choices=("none", "sd_delta"),
        default="none",
        help="SD-derived action prior for expansion order and PUCT selection.",
    )
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.1)
    parser.add_argument("--prior-progress-weight", type=float, default=1.0)
    parser.add_argument("--prior-goal-weight", type=float, default=1.0)
    parser.add_argument(
        "--prior-mobility-weight",
        type=float,
        default=0.0,
        help="Weight for the cheap legal-push-count mobility delta; default off for throughput.",
    )
    parser.add_argument("--prior-revisit-weight", type=float, default=1.0)
    parser.add_argument(
        "--transposition",
        action="store_true",
        help="Share MCTS nodes across identical push-level canonical states.",
    )
    parser.add_argument(
        "--reuse-tree",
        action="store_true",
        help="Persist the search tree across execution steps; implies --transposition.",
    )
    parser.add_argument("--trace-limit", type=int, default=20)
    parser.add_argument("--sd-validation-path", default="")
    parser.add_argument("--tva-audit-path", default="")
    parser.add_argument("--require-independent-gates", action="store_true")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/grid_sdmcts_eval_seed0.json",
    )
    return parser


def _rel(path: str | Path) -> str:
    path = Path(path)
    return os.path.relpath(path, ROOT) if path.is_absolute() else str(path)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tva_checkpoint(path: str | Path) -> TvaCheckpoint:
    payload = load_json(path)
    feature_contract = payload["feature_contract"]
    obs_shape_raw = feature_contract.get("observation_shape")
    obs_shape = tuple(int(v) for v in obs_shape_raw) if obs_shape_raw else None
    model_type = str(payload.get("model_type", "ridge"))
    common = dict(
        path=_rel(path),
        feature_dim=int(feature_contract["feature_dim"]),
        observation_shape=obs_shape,
        value_scale={key: float(value) for key, value in payload.get("value_scale", {}).items()},
        target_mode=str(payload.get("target_mode", "value")),
        model_type=model_type,
    )
    if model_type == "mlp":
        mlp = payload["mlp"]
        return TvaCheckpoint(
            mean=np.asarray(mlp["mean"], dtype=np.float64),
            std=np.asarray(mlp["std"], dtype=np.float64),
            weights=np.zeros(0, dtype=np.float64),
            layers=mlp["layers"],
            **common,
        )
    ridge = payload["ridge"]
    return TvaCheckpoint(
        mean=np.asarray(ridge["mean"], dtype=np.float64),
        std=np.asarray(ridge["std"], dtype=np.float64),
        weights=np.asarray(ridge["weights"], dtype=np.float64),
        **common,
    )


def predict_tva(checkpoint: TvaCheckpoint, model: SokobanGridStockFlowModel, state: SokobanState) -> float:
    features = grid_tva_features(model, state).astype(np.float64)
    if features.shape[0] != checkpoint.feature_dim:
        raise ValueError(
            f"TVA feature dimension mismatch: checkpoint={checkpoint.feature_dim}, current={features.shape[0]}"
        )
    z = (features - checkpoint.mean) / checkpoint.std
    if checkpoint.model_type == "mlp":
        for i, layer in enumerate(checkpoint.layers):
            z = z @ np.asarray(layer["w"], dtype=np.float64).T + np.asarray(layer["b"], dtype=np.float64)
            if i < len(checkpoint.layers) - 1:
                z = np.maximum(z, 0.0)
        return float(z[0])
    design = np.concatenate([z, np.ones(1, dtype=np.float64)], axis=0)
    return float(design @ checkpoint.weights)


def action_to_dict(action: SokobanPushAction) -> dict[str, Any]:
    return {
        "box_index": int(action.box_index),
        "direction": int(action.direction),
        "direction_name": ACTION_NAMES[action.direction],
        "flat_index": int(action.flat_index),
    }


def check_independent_gates(args: argparse.Namespace) -> dict[str, Any]:
    status: dict[str, Any] = {
        "require_independent_gates": bool(args.require_independent_gates),
        "sd_validation_path": _rel(args.sd_validation_path) if args.sd_validation_path else None,
        "tva_audit_path": _rel(args.tva_audit_path) if args.tva_audit_path else None,
    }
    failures: list[str] = []
    if args.sd_validation_path:
        sd_payload = load_json(args.sd_validation_path)
        sd_gate = sd_payload.get("aggregate", {}).get("grid_sd_gate")
        status["grid_sd_gate"] = sd_gate
        if sd_gate != "PASS":
            failures.append("grid_sd_gate")
    elif args.require_independent_gates:
        failures.append("missing_sd_validation_path")

    if args.leaf_value == "tva":
        if args.tva_audit_path:
            tva_payload = load_json(args.tva_audit_path)
            tva_gate = tva_payload.get("gate", {}).get("tva_independent_gate")
            tva_gate_source = "grid_tva_independent_gate"
            if tva_gate is None:
                gft_gate = tva_payload.get("gft_tva_independent_accuracy_gate", {})
                tva_gate = gft_gate.get("verdict")
                tva_gate_source = "gft_tva_independent_accuracy_gate"
            status["tva_independent_gate"] = tva_gate
            status["tva_independent_gate_source"] = tva_gate_source
            if tva_gate != "PASS":
                failures.append("tva_independent_gate")
        elif args.require_independent_gates:
            failures.append("missing_tva_audit_path")

    status["failed_checks"] = failures
    if args.require_independent_gates and failures:
        raise SystemExit(f"Independent gate requirement failed: {failures}")
    return status


class GridSDMCTSPlanner:
    def __init__(
        self,
        model: SokobanGridStockFlowModel,
        *,
        horizon: int,
        budget: int,
        rng: random.Random,
        uct_c: float,
        gamma: float,
        goal_value: float,
        push_cost: float,
        deadend_value: float,
        leaf_value: str,
        tva_checkpoint: TvaCheckpoint | None,
        constraint_mode: str,
        select_by: str,
        revisit_penalty: float = 0.0,
        visited_keys: set | None = None,
        transposition: bool = False,
        action_prior: str = "none",
        prior_temperature: float = 1.0,
        prior_uniform_mix: float = 0.1,
        prior_progress_weight: float = 1.0,
        prior_goal_weight: float = 1.0,
        prior_mobility_weight: float = 0.0,
        prior_revisit_weight: float = 1.0,
    ):
        self.model = model
        self.horizon = int(horizon)
        self.budget = int(budget)
        self.rng = rng
        self.uct_c = float(uct_c)
        self.gamma = float(gamma)
        self.goal_value = float(goal_value)
        self.push_cost = float(push_cost)
        self.deadend_value = float(deadend_value)
        self.leaf_value_mode = leaf_value
        self.tva_checkpoint = tva_checkpoint
        self.constraint_mode = constraint_mode
        self.select_by = select_by
        self.revisit_penalty = float(revisit_penalty)
        self.visited_keys = visited_keys if visited_keys is not None else set()
        self.transposition = bool(transposition)
        self.action_prior = action_prior
        self.prior_temperature = max(float(prior_temperature), 1e-9)
        self.prior_uniform_mix = min(max(float(prior_uniform_mix), 0.0), 1.0)
        self.prior_progress_weight = float(prior_progress_weight)
        self.prior_goal_weight = float(prior_goal_weight)
        self.prior_mobility_weight = float(prior_mobility_weight)
        self.prior_revisit_weight = float(prior_revisit_weight)
        self._node_registry: dict[tuple, MCTSNode] = {}
        self._sd_leaf_cache: dict[tuple, float] = {}
        self._lb_cache: dict[tuple, float] = {}
        self._key_cache: dict[SokobanState, tuple] = {}
        self.deadlock_prunes = 0

    def state_key(self, state: SokobanState) -> tuple:
        key = self._key_cache.get(state)
        if key is None:
            key = self.model.canonical_key(state)
            self._key_cache[state] = key
        return key

    def node_for(self, state: SokobanState, depth: int) -> MCTSNode:
        if not self.transposition:
            return MCTSNode(state, depth)
        key = self.state_key(state)
        node = self._node_registry.get(key)
        if node is None:
            node = MCTSNode(state, depth)
            self._node_registry[key] = node
        return node

    def transition_cost(self, child_state: SokobanState) -> float:
        cost = self.push_cost
        if self.revisit_penalty > 0.0 and self.visited_keys:
            if self.state_key(child_state) in self.visited_keys:
                cost += self.revisit_penalty
        return cost

    def is_blocked(self, state: SokobanState) -> bool:
        if self.constraint_mode == "none" or self.model.is_solved(state):
            return False
        if self.constraint_mode == "tier_a":
            return self.model.is_tier_a_deadlocked(state)
        return self.model.is_deadlocked(state)

    def matching_lower_bound_cached(self, boxes: tuple) -> float:
        cached = self._lb_cache.get(boxes)
        if cached is None:
            cached = self.model.matching_lower_bound(boxes)
            self._lb_cache[boxes] = cached
        return cached

    def candidate_actions(self, state: SokobanState) -> tuple[list[CandidateAction], int]:
        candidates = []
        pruned = 0
        for action in self.model.legal_flows(state):
            nxt = self.model.step_grid_flow(state, action)
            if self.is_blocked(nxt):
                pruned += 1
                continue
            candidates.append(CandidateAction(action=action, child_state=nxt))
        return candidates, pruned

    def boxes_on_goals(self, state: SokobanState) -> int:
        return sum(1 for box in state.boxes if box in self.model.level.goals)

    def assign_sd_delta_priors(self, state: SokobanState, candidates: list[CandidateAction]) -> None:
        if not candidates:
            return
        uniform = 1.0 / len(candidates)
        parent_lb = self.matching_lower_bound_cached(state.boxes)
        parent_goals = self.boxes_on_goals(state)
        use_mobility = self.prior_mobility_weight != 0.0
        # Cheap mobility proxy: legal push count without per-push deadlock checks.
        parent_mobility = len(self.model.legal_flows(state)) if use_mobility else 0
        max_pushes = max(self.model.level.n_boxes * len(ACTION_NAMES), 1)
        scores = []
        for candidate in candidates:
            child = candidate.child_state
            child_lb = self.matching_lower_bound_cached(child.boxes)
            if math.isfinite(parent_lb) and math.isfinite(child_lb):
                progress = max(-4.0, min(4.0, parent_lb - child_lb))
            elif math.isfinite(parent_lb):
                progress = -4.0
            elif math.isfinite(child_lb):
                progress = 4.0
            else:
                progress = 0.0
            mobility_delta = 0.0
            if use_mobility:
                mobility_delta = (len(self.model.legal_flows(child)) - parent_mobility) / max_pushes
            revisit = 1.0 if self.state_key(child) in self.visited_keys else 0.0
            candidate.prior_score = (
                self.prior_progress_weight * progress
                + self.prior_goal_weight * (self.boxes_on_goals(child) - parent_goals)
                + self.prior_mobility_weight * mobility_delta
                - self.prior_revisit_weight * revisit
            )
            scores.append(candidate.prior_score)
        logits = np.asarray(scores, dtype=np.float64) / self.prior_temperature
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        denom = float(np.sum(probs))
        probs = probs / denom if denom > 0.0 else np.full(len(candidates), uniform)
        mix = self.prior_uniform_mix
        for candidate, prob in zip(candidates, probs):
            candidate.prior = float((1.0 - mix) * prob + mix * uniform)

    def leaf_value(self, state: SokobanState) -> float:
        if self.model.is_solved(state):
            return self.goal_value
        if self.is_blocked(state):
            return self.deadend_value
        if self.leaf_value_mode == "zero":
            return 0.0
        if self.leaf_value_mode == "sd_heuristic":
            return self.sd_heuristic_value(state)
        if self.tva_checkpoint is None:
            raise ValueError("TVA leaf value requested without a checkpoint.")
        prediction = predict_tva(self.tva_checkpoint, self.model, state)
        if self.tva_checkpoint.target_mode == "residual":
            return self.sd_heuristic_value(state) + prediction
        return prediction

    def sd_heuristic_value(self, state: SokobanState) -> float:
        """Truncation value from the SD readout: matching_lower_bound pushes to go.

        Same functional form as the teacher value label, with the teacher's
        solution length replaced by the SD matching lower bound.
        """
        cached = self._sd_leaf_cache.get(state.boxes)
        if cached is not None:
            return cached
        lb = self.matching_lower_bound_cached(state.boxes)
        if not math.isfinite(lb):
            value = self.deadend_value
        elif self.gamma == 1.0:
            value = self.goal_value - self.push_cost * lb
        else:
            discount = self.gamma**lb
            value = discount * self.goal_value - self.push_cost * (1.0 - discount) / (1.0 - self.gamma)
        self._sd_leaf_cache[state.boxes] = value
        return value

    def initialise_node(self, node: MCTSNode) -> None:
        if node.untried is not None:
            return
        candidates, pruned = self.candidate_actions(node.state)
        if self.action_prior == "none":
            self.rng.shuffle(candidates)
        else:
            self.assign_sd_delta_priors(node.state, candidates)
            # untried is consumed via pop() from the tail, so sort ascending to
            # expand the highest-prior candidate first.
            candidates.sort(key=lambda candidate: (candidate.prior, candidate.prior_score))
        node.untried = candidates
        node.pruned_on_init = pruned
        self.deadlock_prunes += pruned

    def select_child(self, node: MCTSNode) -> EdgeStats:
        parent_visits = max(node.visits, 1)

        def score(edge: EdgeStats) -> float:
            if self.action_prior == "none":
                if edge.visits == 0:
                    return float("inf")
                bonus = self.uct_c * math.sqrt(math.log(parent_visits + 1.0) / edge.visits)
            else:
                bonus = self.uct_c * edge.prior * math.sqrt(parent_visits) / (1.0 + edge.visits)
            return edge.mean_value + bonus

        return max(node.children.values(), key=score)

    def simulate(self, node: MCTSNode, depth_remaining: int) -> float:
        if self.model.is_solved(node.state):
            value = self.goal_value
        elif self.is_blocked(node.state):
            value = self.deadend_value
        elif depth_remaining <= 0:
            value = self.leaf_value(node.state)
        else:
            self.initialise_node(node)
            if not node.untried and not node.children:
                value = self.deadend_value
            elif node.untried:
                candidate = node.untried.pop()
                action = candidate.action
                child_state = candidate.child_state
                child_node = self.node_for(child_state, node.depth_from_root + 1)
                child_value = self.leaf_value(child_state)
                value = -self.transition_cost(child_state) + self.gamma * child_value
                edge = EdgeStats(
                    action=action,
                    child_state=child_state,
                    child_node=child_node,
                    prior=candidate.prior,
                    prior_score=candidate.prior_score,
                )
                edge.visits = 1
                edge.value_sum = value
                node.children[action.flat_index] = edge
            else:
                edge = self.select_child(node)
                child_value = self.simulate(edge.child_node, depth_remaining - 1)
                value = -self.transition_cost(edge.child_state) + self.gamma * child_value
                edge.visits += 1
                edge.value_sum += value

        node.visits += 1
        node.value_sum += value
        return value

    def plan(self, state: SokobanState) -> dict[str, Any]:
        root = self.node_for(state, 0)
        prunes_before = self.deadlock_prunes
        for _ in range(max(self.budget, 1)):
            self.simulate(root, self.horizon)
        plan_prunes = self.deadlock_prunes - prunes_before
        if not root.children:
            return {
                "action": None,
                "root_visits": root.visits,
                "root_value": root.value_sum / root.visits if root.visits else 0.0,
                "child_count": 0,
                "deadlock_prunes": plan_prunes,
                "children": [],
            }
        if self.select_by == "q":
            best = max(root.children.values(), key=lambda edge: (edge.mean_value, edge.visits))
        else:
            best = max(root.children.values(), key=lambda edge: (edge.visits, edge.mean_value))
        children = sorted(
            (
                {
                    "action": action_to_dict(edge.action),
                    "visits": int(edge.visits),
                    "mean_value": float(edge.mean_value),
                    "prior": float(edge.prior),
                    "prior_score": float(edge.prior_score),
                    "solved_child": bool(self.model.is_solved(edge.child_state)),
                    "blocked_child": bool(self.is_blocked(edge.child_state)),
                }
                for edge in root.children.values()
            ),
            key=lambda row: (row["visits"], row["mean_value"]),
            reverse=True,
        )
        return {
            "action": best.action,
            "root_visits": root.visits,
            "root_value": root.value_sum / root.visits if root.visits else 0.0,
            "child_count": len(root.children),
            "deadlock_prunes": plan_prunes,
            "children": children,
        }


def run_level(
    level_idx: int,
    model: SokobanGridStockFlowModel,
    args: argparse.Namespace,
    tva_checkpoint: TvaCheckpoint | None,
) -> dict[str, Any]:
    rng = random.Random(args.seed + level_idx * 1009)
    state = model.initial_state
    visited_keys = {model.canonical_key(state)}
    trace: list[dict[str, Any]] = []
    total_deadlock_prunes = 0
    pushes_executed = 0
    t0 = perf_counter()
    status = "max_steps"
    persistent_planner: GridSDMCTSPlanner | None = None
    for step in range(args.max_steps + 1):
        if model.is_solved(state):
            status = "solved"
            break
        if step >= args.max_steps:
            break
        if persistent_planner is not None:
            planner = persistent_planner
        else:
            planner = GridSDMCTSPlanner(
                model,
                horizon=args.horizon,
                budget=args.budget,
                rng=rng,
                uct_c=args.uct_c,
                gamma=args.gamma,
                goal_value=args.goal_value,
                push_cost=args.push_cost,
                deadend_value=args.deadend_value,
                leaf_value=args.leaf_value,
                tva_checkpoint=tva_checkpoint,
                constraint_mode=args.constraint_mode,
                select_by=args.select_by,
                revisit_penalty=args.revisit_penalty,
                visited_keys=visited_keys,
                transposition=args.transposition or args.reuse_tree,
                action_prior=args.action_prior,
                prior_temperature=args.prior_temperature,
                prior_uniform_mix=args.prior_uniform_mix,
                prior_progress_weight=args.prior_progress_weight,
                prior_goal_weight=args.prior_goal_weight,
                prior_mobility_weight=args.prior_mobility_weight,
                prior_revisit_weight=args.prior_revisit_weight,
            )
            if args.reuse_tree:
                persistent_planner = planner
        plan = planner.plan(state)
        total_deadlock_prunes += int(plan["deadlock_prunes"])
        action = plan["action"]
        if action is None:
            status = "no_action"
            if len(trace) < args.trace_limit:
                trace.append(
                    {
                        "step": step,
                        "status": "no_action",
                        "root_visits": int(plan["root_visits"]),
                        "root_value": float(plan["root_value"]),
                    }
                )
            break
        next_state = model.step_grid_flow(state, action)
        pushes_executed += 1
        if len(trace) < args.trace_limit:
            trace.append(
                {
                    "step": step,
                    "action": action_to_dict(action),
                    "root_visits": int(plan["root_visits"]),
                    "root_value": float(plan["root_value"]),
                    "child_count": int(plan["child_count"]),
                    "top_children": plan["children"][:5],
                    "boxes_on_goals_after": int(sum(1 for box in next_state.boxes if box in model.level.goals)),
                    "tier_a_after": model.tier_a_status(next_state),
                }
            )
        state = next_state
        visited_keys.add(model.canonical_key(state))
        if model.is_deadlocked(state) and not model.is_solved(state):
            status = "deadlocked"
            break

    elapsed = perf_counter() - t0
    solved = model.is_solved(state)
    return {
        "level_index": int(level_idx),
        "level_id": model.level.level_id,
        "height": model.level.height,
        "width": model.level.width,
        "n_boxes": model.level.n_boxes,
        "solved": bool(solved),
        "status": "solved" if solved else status,
        "pushes": pushes_executed if solved else None,
        "steps_executed": pushes_executed,
        "deadlock_prunes": int(total_deadlock_prunes),
        "elapsed_sec": elapsed,
        "trace": trace,
        "final_features": model.abstract_features(state).to_dict(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [row for row in rows if row["solved"]]
    return {
        "levels_evaluated": len(rows),
        "solved_count": len(solved),
        "success_rate": float(len(solved) / len(rows)) if rows else 0.0,
        "mean_pushes_solved": float(np.mean([row["pushes"] for row in solved])) if solved else None,
        "total_deadlock_prunes": int(sum(row["deadlock_prunes"] for row in rows)),
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    gate_status = check_independent_gates(args)
    if args.leaf_value == "tva":
        tva_checkpoint = load_tva_checkpoint(args.tva_checkpoint)
        mismatched = {
            key: (tva_checkpoint.value_scale.get(key), getattr(args, key))
            for key in ("gamma", "goal_value", "push_cost", "deadend_value")
            if key in tva_checkpoint.value_scale
            and not math.isclose(tva_checkpoint.value_scale[key], getattr(args, key), abs_tol=1e-9)
        }
        if mismatched:
            raise SystemExit(f"TVA checkpoint value_scale mismatches runtime args: {mismatched}")
    else:
        tva_checkpoint = None

    if args.level_offset < 0:
        raise SystemExit("--level-offset must be non-negative.")
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")
    rows = [
        run_level(level_idx, SokobanGridStockFlowModel(level), args, tva_checkpoint)
        for level_idx, level in enumerate(levels)
    ]
    summary = {
        "material_passport": {
            "origin": "sokoban_grid_sdmcts_refactor",
            "verification_status": "ANALYZED",
            "version_label": "exact_grid_sdmcts_closed_loop_v1",
        },
        "config": {
            "levels": _rel(args.levels),
            "level_offset": args.level_offset,
            "max_levels": args.max_levels,
            "horizon": args.horizon,
            "budget": args.budget,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "uct_c": args.uct_c,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "leaf_value": args.leaf_value,
            "constraint_mode": args.constraint_mode,
            "select_by": args.select_by,
            "revisit_penalty": args.revisit_penalty,
            "transposition": bool(args.transposition or args.reuse_tree),
            "reuse_tree": bool(args.reuse_tree),
            "action_prior": args.action_prior,
            "prior_temperature": args.prior_temperature,
            "prior_uniform_mix": args.prior_uniform_mix,
            "prior_progress_weight": args.prior_progress_weight,
            "prior_goal_weight": args.prior_goal_weight,
            "prior_mobility_weight": args.prior_mobility_weight,
            "prior_revisit_weight": args.prior_revisit_weight,
        },
        "framework_contract": {
            "transition_provider": SokobanGridStockFlowModel.provider_id,
            "planner_state": "exact_sokoban_grid_state",
            "planner_action": "push_level_grid_flow",
            "legacy_six_stock_rollout_used": False,
            "teacher_used_online": False,
            "tva_checkpoint": tva_checkpoint.path if tva_checkpoint else None,
        },
        "independent_gate_status": gate_status,
        "aggregate": aggregate(rows),
        "levels": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
