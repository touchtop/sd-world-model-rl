#!/usr/bin/env python3
"""Audit goal-room packing as a candidate Sokoban SD stock/feedback.

This diagnostic asks whether goal-room organization is a stable task-relevant
state abstraction, before adding it to the learned SD world model. It does not
train a transition provider and does not modify SD-MCTS.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_sokoban_sd_goal_alignment import (  # noqa: E402
    fit_ridge,
    provider_level_split,
    ridge_predict,
    standardize,
    teacher_value,
)
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from src.sokoban.env import DIRECTIONS  # noqa: E402
from validate_sokoban_sd_design import STOCK_NAMES, iter_states, stock_proxy  # noqa: E402


GOAL_ROOM_FEATURE_NAMES = (
    "goal_room:packing_deficit",
    "goal_room:endpoint_fill_fraction",
    "goal_room:center_goal_occupancy",
    "goal_room:goal_entry_blocked_fraction",
    "goal_room:goal_adjacent_box_pressure",
    "goal_room:movable_goal_box_fraction",
    "goal_room:orderable_goal_box_fraction",
    "goal_room:premature_goal_entry_available",
    "goal_room:order_action_available",
    "goal_room:order_vs_entry_conflict",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--state-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=12)
    parser.add_argument("--teacher-max-expansions", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--alias-round-decimals", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/goal_room_stock_probe_quick_seed0.json",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    state: SokobanState,
    action,
) -> float:
    if action.box_index < 0 or action.box_index >= len(state.boxes):
        return 0.0
    before_cell = state.boxes[action.box_index]
    direction = DIRECTIONS[action.direction]
    after_cell = (before_cell[0] + direction[0], before_cell[1] + direction[1])
    return _goal_room_order_bonus_from_cells(model, before_cell, after_cell)


def _goal_entry_action(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    action,
) -> bool:
    if action.box_index < 0 or action.box_index >= len(state.boxes):
        return False
    goals = set(model.level.goals)
    before_cell = state.boxes[action.box_index]
    direction = DIRECTIONS[action.direction]
    after_cell = (before_cell[0] + direction[0], before_cell[1] + direction[1])
    return before_cell not in goals and after_cell in goals


def goal_room_features(model: SokobanGoalDirectedSDModel, state: SokobanState) -> np.ndarray:
    goals = set(model.level.goals)
    boxes = set(state.boxes)
    n_boxes = max(model.level.n_boxes, 1)
    n_goals = max(len(goals), 1)
    goal_degrees = {goal: _goal_neighbor_degree(model, goal) for goal in goals}
    endpoint_goals = {goal for goal, degree in goal_degrees.items() if degree <= 1}
    center_goals = goals - endpoint_goals
    endpoint_fill = _safe_div(len(boxes & endpoint_goals), max(len(endpoint_goals), 1))
    center_occupancy = _safe_div(len(boxes & center_goals), max(len(center_goals), 1))

    goal_neighbors: set[tuple[int, int]] = set()
    blocked_goals = 0
    for goal in goals:
        neigh = set(_floor_neighbors(model, goal)) - goals
        goal_neighbors.update(neigh)
        if any(cell in boxes for cell in neigh):
            blocked_goals += 1

    legal = tuple(model.legal_pushes(state))
    movable_goal_boxes = {
        action.box_index
        for action in legal
        if action.box_index < len(state.boxes) and state.boxes[action.box_index] in goals
    }
    order_actions = [action for action in legal if _goal_room_order_bonus_for_action(model, state, action) > 0.0]
    entry_actions = [action for action in legal if _goal_entry_action(model, state, action)]
    orderable_goal_boxes = {
        action.box_index
        for action in order_actions
        if action.box_index < len(state.boxes) and state.boxes[action.box_index] in goals
    }
    boxes_on_goals = len(boxes & goals)
    packing_deficit = _clamp01(
        _safe_div(max(len(endpoint_goals) - len(boxes & endpoint_goals), 0), max(len(endpoint_goals), 1))
        * _safe_div(boxes_on_goals, n_boxes)
    )
    return np.array(
        [
            packing_deficit,
            endpoint_fill,
            center_occupancy,
            _safe_div(blocked_goals, n_goals),
            _safe_div(len((boxes & goal_neighbors) - goals), n_boxes),
            _safe_div(len(movable_goal_boxes), n_boxes),
            _safe_div(len(orderable_goal_boxes), n_boxes),
            float(bool(entry_actions)),
            float(bool(order_actions)),
            float(bool(entry_actions) and bool(order_actions)),
        ],
        dtype=np.float32,
    )


def ridge_metrics(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray) -> Dict[str, Any]:
    if len(train_idx) == 0 or len(valid_idx) == 0:
        return {}
    train_x_std, mean, std = standardize(features[train_idx].astype(np.float64), features[train_idx].astype(np.float64))
    valid_x_std = (features[valid_idx].astype(np.float64) - mean) / std
    weights = fit_ridge(train_x_std, targets[train_idx].astype(np.float64), alpha=1.0)
    pred = ridge_predict(valid_x_std, weights)
    target = targets[valid_idx]
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 1e-12 and np.std(target) > 1e-12 else 0.0
    return {
        "r2": r2,
        "corr": corr,
        "mae": float(np.mean(np.abs(pred - target))),
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
    }


def alias_variance_ratio(features: np.ndarray, targets: np.ndarray, decimals: int) -> Dict[str, Any]:
    buckets: dict[tuple[float, ...], list[float]] = defaultdict(list)
    for row, target in zip(features, targets):
        buckets[tuple(np.round(row, decimals).tolist())].append(float(target))
    within = []
    for vals in buckets.values():
        if len(vals) >= 2:
            within.extend((np.asarray(vals) - float(np.mean(vals))).tolist())
    global_var = float(np.var(targets)) if len(targets) else 0.0
    within_var = float(np.var(within)) if within else 0.0
    return {
        "bucket_count": len(buckets),
        "multi_item_bucket_count": sum(1 for vals in buckets.values() if len(vals) >= 2),
        "within_var": within_var,
        "global_var": global_var,
        "within_global_ratio": float(within_var / global_var) if global_var > 1e-12 else 0.0,
    }


def delta_summary(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    delta = after - before
    return {
        "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "nonzero_fraction_gt_1e_6": float(np.mean(np.abs(delta) > 1e-6)) if delta.size else 0.0,
        "per_feature_mean_abs_delta": {
            name: float(np.mean(np.abs(delta[:, idx]))) if delta.size else 0.0
            for idx, name in enumerate(GOAL_ROOM_FEATURE_NAMES)
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit("No levels loaded.")
    split = provider_level_split(args.provider_checkpoint, [level.level_id for level in levels], args.seed)
    rows: list[dict[str, Any]] = []
    transition_before: list[np.ndarray] = []
    transition_after: list[np.ndarray] = []
    label_counts: Counter[str] = Counter()
    t0 = perf_counter()

    for level in levels:
        if level.level_id in split["train_levels"]:
            split_name = "train"
        elif level.level_id in split["valid_levels"]:
            split_name = "valid"
        else:
            continue
        model = SokobanGoalDirectedSDModel(level)
        level_rows = 0
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            teacher, target_value = teacher_value(
                args,
                model,
                state,
            )
            stock = stock_proxy(model, state).as_array().astype(np.float32)
            goal_room = goal_room_features(model, state)
            rows.append(
                {
                    "split": split_name,
                    "level_id": level.level_id,
                    "state_depth": int(state_depth),
                    "stock": stock,
                    "goal_room": goal_room,
                    "target_value": float(target_value),
                    "teacher_label": str(teacher.failure_type),
                }
            )
            label_counts[str(teacher.failure_type)] += 1
            level_rows += 1
            for action in model.legal_pushes(state):
                nxt = model.step_state(state, action)
                transition_before.append(goal_room)
                transition_after.append(goal_room_features(model, nxt))
        print(json.dumps({"level_id": level.level_id, "split": split_name, "states": level_rows}))

    if not rows:
        raise SystemExit("No rows collected.")
    stock_features = np.stack([row["stock"] for row in rows]).astype(np.float32)
    goal_room_matrix = np.stack([row["goal_room"] for row in rows]).astype(np.float32)
    combined = np.concatenate([stock_features, goal_room_matrix], axis=1)
    targets = np.asarray([row["target_value"] for row in rows], dtype=np.float32)
    train_idx = np.asarray([idx for idx, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    valid_idx = np.asarray([idx for idx, row in enumerate(rows) if row["split"] == "valid"], dtype=np.int64)
    stock_metrics = ridge_metrics(stock_features, targets, train_idx, valid_idx)
    combined_metrics = ridge_metrics(combined, targets, train_idx, valid_idx)
    goal_only_metrics = ridge_metrics(goal_room_matrix, targets, train_idx, valid_idx)
    before = np.stack(transition_before).astype(np.float32) if transition_before else np.zeros((0, len(GOAL_ROOM_FEATURE_NAMES)))
    after = np.stack(transition_after).astype(np.float32) if transition_after else np.zeros((0, len(GOAL_ROOM_FEATURE_NAMES)))
    r2_gain = float(combined_metrics.get("r2", 0.0) - stock_metrics.get("r2", 0.0))
    gate_failed = []
    if r2_gain < 0.02:
        gate_failed.append("valid_r2_gain_min_0.02")
    transition_delta = delta_summary(before, after)
    if transition_delta["nonzero_fraction_gt_1e_6"] < 0.05:
        gate_failed.append("goal_room_transition_signal_min_0.05")

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_goal_room_stock_probe_v1",
        },
        "config": vars(args),
        "audit_contract": {
            "diagnostic_only": True,
            "transition_provider_modified": False,
            "sd_mcts_run": False,
            "best_first_used_online": False,
            "teacher_used_for_offline_labels_only": True,
        },
        "feature_names": {
            "stock": list(STOCK_NAMES),
            "goal_room": list(GOAL_ROOM_FEATURE_NAMES),
        },
        "dataset": {
            "samples": len(rows),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "teacher_label_counts": dict(label_counts),
            "level_split": split,
        },
        "metrics": {
            "stock_only": stock_metrics,
            "goal_room_only": goal_only_metrics,
            "stock_plus_goal_room": combined_metrics,
            "valid_r2_gain_over_stock": r2_gain,
            "stock_alias": alias_variance_ratio(stock_features[valid_idx], targets[valid_idx], args.alias_round_decimals),
            "stock_plus_goal_room_alias": alias_variance_ratio(
                combined[valid_idx],
                targets[valid_idx],
                args.alias_round_decimals,
            ),
            "transition_delta": transition_delta,
        },
        "candidate_stock_gate": {
            "verdict": "PASS" if not gate_failed else "FAIL",
            "failed_checks": gate_failed,
            "thresholds": {
                "valid_r2_gain_min": 0.02,
                "goal_room_transition_nonzero_fraction_min": 0.05,
            },
        },
        "elapsed_sec": float(perf_counter() - t0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": summary["candidate_stock_gate"],
                "metrics": {
                    "stock_only": stock_metrics,
                    "stock_plus_goal_room": combined_metrics,
                    "valid_r2_gain_over_stock": r2_gain,
                    "transition_delta": transition_delta,
                },
                "out": str(out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
