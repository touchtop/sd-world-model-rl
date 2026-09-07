#!/usr/bin/env python3
"""Audit whether Sokoban SD stocks encode task-goal relevant abstractions.

This script is diagnostic. It does not train the SD transition model, does not
run closed-loop SD-MCTS, and does not use the diagnostic teacher online. The
teacher is used only to label sampled states and one-step action consequences
so that the current SD abstraction can be tested against task-goal targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import stock_reward, stock_value  # noqa: E402
from sokoban_tva import TVA_FEATURE_NAMES, relative_path, tva_feature_vector  # noqa: E402
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from src.sokoban.env import DIRECTIONS  # noqa: E402
from train_sokoban_tva import best_first_from_state, target_from_teacher  # noqa: E402
from validate_sokoban_sd_design import STOCK_NAMES, iter_states, stock_proxy  # noqa: E402


GOAL_PROBE_FEATURE_NAMES = (
    "probe:goal_adjacent_box_pressure",
    "probe:goal_entry_blocked_fraction",
    "probe:occupied_goal_neighborhood_fraction",
    "probe:finite_box_goal_edge_density",
    "probe:min_box_goal_options_norm",
    "probe:mean_box_goal_options_norm",
    "probe:assignment_slack_norm",
    "probe:unreachable_box_fraction",
    "probe:matching_pressure_norm",
    "probe:corridor_box_pressure",
    "probe:articulation_box_pressure",
    "probe:dead_square_adjacent_box_pressure",
    "probe:player_box_separation_norm",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
        help="Stage-2 checkpoint used only to reuse the level split.",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory to audit.",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--state-depth", type=int, default=3)
    parser.add_argument("--max-states-per-level", type=int, default=90)
    parser.add_argument("--teacher-max-expansions", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--alias-round-decimals", type=int, default=1)
    parser.add_argument("--alias-min-bucket-size", type=int, default=3)
    parser.add_argument(
        "--max-action-states-per-level",
        type=int,
        default=25,
        help="Maximum sampled states per level for one-step action consequence audit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_seed0.json",
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


def _torch_load(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fallback_level_split(level_ids: Sequence[str], valid_fraction: float, seed: int) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    shuffled = np.array(sorted(level_ids), dtype=object)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_fraction))) if len(shuffled) > 1 else 0
    valid = set(shuffled[:valid_count].tolist())
    train = [level_id for level_id in shuffled.tolist() if level_id not in valid]
    return {"split_mode": "fallback_level", "train_levels": sorted(train), "valid_levels": sorted(valid)}


def provider_level_split(provider_checkpoint: str | Path, all_level_ids: Sequence[str], seed: int) -> Dict[str, Any]:
    checkpoint = Path(provider_checkpoint)
    if checkpoint.exists():
        payload = _torch_load(checkpoint)
        split = payload.get("summary", {}).get("dataset", {}).get("split", {})
        train_levels = sorted(set(split.get("train_levels", [])))
        valid_levels = sorted(set(split.get("valid_levels", [])))
        loaded = set(all_level_ids)
        if loaded.intersection(train_levels) and loaded.intersection(valid_levels):
            return {
                "split_mode": split.get("split_mode", "level"),
                "train_levels": train_levels,
                "valid_levels": valid_levels,
                "source": str(checkpoint),
            }
    split = fallback_level_split(all_level_ids, valid_fraction=0.2, seed=seed)
    split["source"] = "fallback"
    return split


def floor_neighbors(model: SokobanGoalDirectedSDModel, cell: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for direction in DIRECTIONS:
        nxt = (cell[0] + direction[0], cell[1] + direction[1])
        if model.level.is_floor(nxt):
            out.append(nxt)
    return out


def articulation_cells(model: SokobanGoalDirectedSDModel) -> set[tuple[int, int]]:
    """Static floor-graph articulation points used as a bottleneck proxy."""

    cells = set(model.floor_cells)
    if not cells:
        return set()
    adjacency = {cell: [n for n in floor_neighbors(model, cell) if n in cells] for cell in cells}
    visited: set[tuple[int, int]] = set()
    disc: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    arts: set[tuple[int, int]] = set()
    time = 0

    def dfs(u: tuple[int, int]) -> None:
        nonlocal time
        visited.add(u)
        disc[u] = time
        low[u] = time
        time += 1
        children = 0
        for v in adjacency[u]:
            if v not in visited:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    arts.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    arts.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for cell in sorted(cells):
        if cell not in visited:
            parent[cell] = None
            dfs(cell)
    return arts


def goal_probe_features(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    *,
    static_articulation_cells: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    features = model.abstract_features(state)
    boxes = set(state.boxes)
    goals = set(model.level.goals)
    n_boxes = max(model.level.n_boxes, 1)
    n_goals = max(len(goals), 1)
    board_scale = max(model.level.height + model.level.width, 1)

    goal_neighbors: set[tuple[int, int]] = set()
    blocked_goals = 0
    for goal in goals:
        neigh = set(floor_neighbors(model, goal)) - goals
        goal_neighbors.update(neigh)
        if any(cell in boxes for cell in neigh):
            blocked_goals += 1

    occupied_goal_neighborhood = len((boxes & goal_neighbors) - goals)
    goal_adjacent_pressure = _safe_div(occupied_goal_neighborhood, n_boxes)
    goal_entry_blocked = _safe_div(blocked_goals, n_goals)

    option_counts = []
    finite_edges = 0
    unreachable_boxes = 0
    for box in state.boxes:
        count = 0
        for dist in model.pull_distances:
            if dist.get(box, 10**9) < 10**9:
                count += 1
        option_counts.append(count)
        finite_edges += count
        if count == 0:
            unreachable_boxes += 1

    edge_density = _safe_div(finite_edges, n_boxes * n_goals)
    min_options = _safe_div(min(option_counts) if option_counts else 0, n_goals)
    mean_options = _safe_div(float(np.mean(option_counts)) if option_counts else 0.0, n_goals)
    assignment_slack = _safe_div(max(finite_edges - n_boxes, 0), max(n_boxes * (n_goals - 1), 1))
    unreachable_fraction = _safe_div(unreachable_boxes, n_boxes)
    matching_pressure = (
        _clamp01(float(features.matching_lower_bound) / (n_boxes * board_scale))
        if np.isfinite(features.matching_lower_bound)
        else 1.0
    )

    floor_degree = {cell: len(floor_neighbors(model, cell)) for cell in model.floor_cells}
    corridor_cells = {cell for cell, degree in floor_degree.items() if degree <= 2}
    corridor_pressure = _safe_div(len((boxes & corridor_cells) - goals), n_boxes)
    arts = static_articulation_cells if static_articulation_cells is not None else articulation_cells(model)
    articulation_pressure = _safe_div(len((boxes & arts) - goals), n_boxes)
    dead_square_adjacent = 0
    for box in state.boxes:
        if box in goals:
            continue
        if any(neigh in model.dead_squares for neigh in floor_neighbors(model, box)):
            dead_square_adjacent += 1
    dead_square_adjacent_pressure = _safe_div(dead_square_adjacent, n_boxes)
    if state.boxes:
        sep = min(abs(state.player[0] - box[0]) + abs(state.player[1] - box[1]) for box in state.boxes)
    else:
        sep = 0.0
    player_box_separation = _clamp01(sep / board_scale)

    return np.array(
        [
            goal_adjacent_pressure,
            goal_entry_blocked,
            _safe_div(occupied_goal_neighborhood, max(len(goal_neighbors), 1)),
            edge_density,
            min_options,
            mean_options,
            assignment_slack,
            unreachable_fraction,
            matching_pressure,
            corridor_pressure,
            articulation_pressure,
            dead_square_adjacent_pressure,
            player_box_separation,
        ],
        dtype=np.float32,
    )


def teacher_value(args: argparse.Namespace, model: SokobanGoalDirectedSDModel, state: SokobanState) -> tuple[Any, float]:
    teacher = best_first_from_state(model, state, max_expansions=args.teacher_max_expansions)
    value = target_from_teacher(
        teacher,
        gamma=args.gamma,
        goal_value=args.goal_value,
        push_cost=args.push_cost,
        deadend_value=args.deadend_value,
        unsolved_value=args.unsolved_value,
    )
    return teacher, float(value)


def collect_state_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit("No levels loaded for SD goal-alignment audit.")

    split = provider_level_split(args.provider_checkpoint, [level.level_id for level in levels], args.seed)
    rows: list[dict[str, Any]] = []
    teacher_cache: dict[tuple[str, SokobanState], tuple[Any, float]] = {}
    split_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    t0 = perf_counter()

    for level_index, level in enumerate(levels):
        if level.level_id in split["train_levels"]:
            split_name = "train"
        elif level.level_id in split["valid_levels"]:
            split_name = "valid"
        else:
            continue
        model = SokobanGoalDirectedSDModel(level)
        static_articulations = articulation_cells(model)
        level_rows = 0
        seen: set[SokobanState] = set()
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            if state in seen:
                continue
            seen.add(state)
            stock = stock_proxy(model, state).as_array().astype(np.float32)
            key = (level.level_id, state)
            if key not in teacher_cache:
                teacher_cache[key] = teacher_value(args, model, state)
            teacher, value = teacher_cache[key]
            tva = tva_feature_vector(model, state, stock).astype(np.float32)
            probes = goal_probe_features(
                model,
                state,
                static_articulation_cells=static_articulations,
            ).astype(np.float32)
            rows.append(
                {
                    "level_index": int(level_index),
                    "level_id": level.level_id,
                    "split": split_name,
                    "state_depth": int(state_depth),
                    "stock": stock,
                    "tva": tva,
                    "goal_probe": probes,
                    "target_value": float(value),
                    "teacher_solved": bool(teacher.solved),
                    "teacher_solution_pushes": (
                        int(teacher.solution_pushes) if teacher.solution_pushes is not None else None
                    ),
                    "teacher_failure_type": str(teacher.failure_type),
                    "root_state_type": (
                        "solved"
                        if model.is_solved(state)
                        else "deadlocked"
                        if model.is_deadlocked(state)
                        else "no_legal_actions"
                        if not model.legal_pushes(state)
                        else "active"
                    ),
                    "fixed_stock_value": float(stock_value(stock.astype(np.float64))),
                }
            )
            teacher_counts[str(teacher.failure_type)] += 1
            split_counts[split_name] += 1
            level_rows += 1
        print(
            json.dumps(
                {
                    "level_index": level_index,
                    "level_id": level.level_id,
                    "split": split_name,
                    "goal_alignment_state_rows": level_rows,
                },
                ensure_ascii=False,
            )
        )

    return rows, {
        "level_split": split,
        "split_counts": dict(split_counts),
        "teacher_counts": dict(teacher_counts),
        "collection_elapsed_sec": float(perf_counter() - t0),
    }


def feature_sets(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    stock = np.stack([row["stock"] for row in rows]).astype(np.float32)
    tva = np.stack([row["tva"] for row in rows]).astype(np.float32)
    probes = np.stack([row["goal_probe"] for row in rows]).astype(np.float32)
    return {
        "six_stocks": (stock, tuple(f"stock:{name}" for name in STOCK_NAMES)),
        "six_stocks_plus_feedback": (
            tva[:, : len(STOCK_NAMES) + 9],
            tuple(TVA_FEATURE_NAMES[: len(STOCK_NAMES) + 9]),
        ),
        "six_stocks_plus_existing_structural": (tva, tuple(TVA_FEATURE_NAMES)),
        "six_stocks_plus_goal_probes": (
            np.concatenate([stock, probes], axis=1),
            tuple(f"stock:{name}" for name in STOCK_NAMES) + GOAL_PROBE_FEATURE_NAMES,
        ),
        "all_existing_plus_goal_probes": (
            np.concatenate([tva, probes], axis=1),
            tuple(TVA_FEATURE_NAMES) + GOAL_PROBE_FEATURE_NAMES,
        ),
    }


def split_indices(rows: Sequence[dict[str, Any]], split_name: str) -> np.ndarray:
    return np.array([idx for idx, row in enumerate(rows) if row["split"] == split_name], dtype=np.int64)


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ train_y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return xb @ weights


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size == 0:
        return {}
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "samples": float(pred.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "corr": float(np.corrcoef(pred, target)[0, 1]) if pred.size > 1 and np.std(pred) > 1e-12 else 0.0,
    }


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return float(wins / (len(pos) * len(neg)))


def evaluate_feature_sets(
    rows: Sequence[dict[str, Any]],
    sets: dict[str, tuple[np.ndarray, tuple[str, ...]]],
    *,
    alpha: float,
) -> dict[str, Any]:
    y = np.array([row["target_value"] for row in rows], dtype=np.float64)
    solved = np.array([row["teacher_solved"] for row in rows], dtype=bool)
    train_idx = split_indices(rows, "train")
    valid_idx = split_indices(rows, "valid")
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise SystemExit("Goal-alignment audit requires non-empty train and valid splits.")

    out: dict[str, Any] = {}
    for name, (x_raw, names) in sets.items():
        x_raw = x_raw.astype(np.float64)
        train_x_std, mean, std = standardize(x_raw[train_idx], x_raw[train_idx])
        all_x_std = (x_raw - mean) / std
        weights = fit_ridge(train_x_std, y[train_idx], alpha)
        pred = ridge_predict(all_x_std, weights)
        out[name] = {
            "feature_count": int(x_raw.shape[1]),
            "features": list(names),
            "train": regression_metrics(pred[train_idx], y[train_idx]),
            "valid": regression_metrics(pred[valid_idx], y[valid_idx]),
            "valid_auc_teacher_solved": auc_score(pred[valid_idx], solved[valid_idx]),
        }

    fixed = np.array([row["fixed_stock_value"] for row in rows], dtype=np.float64)
    out["fixed_stock_value_readout"] = {
        "feature_count": 0,
        "features": ["fixed_stage3_stock_value_formula"],
        "train": regression_metrics(fixed[train_idx], y[train_idx]),
        "valid": regression_metrics(fixed[valid_idx], y[valid_idx]),
        "valid_auc_teacher_solved": auc_score(fixed[valid_idx], solved[valid_idx]),
    }
    return out


def alias_audit_for_matrix(
    x: np.ndarray,
    target: np.ndarray,
    solved: np.ndarray,
    *,
    round_decimals: int,
    min_bucket_size: int,
) -> dict[str, Any]:
    buckets: dict[tuple[float, ...], list[int]] = defaultdict(list)
    rounded = np.round(x.astype(np.float64), int(round_decimals))
    for idx, row in enumerate(rounded):
        buckets[tuple(row.tolist())].append(idx)
    global_var = float(np.var(target)) if len(target) else 0.0
    multi = {key: vals for key, vals in buckets.items() if len(vals) >= min_bucket_size}
    weighted_var_num = 0.0
    weighted_count = 0
    mixed = 0
    max_var = 0.0
    max_bucket_size = 0
    for vals in multi.values():
        arr = target[vals]
        var = float(np.var(arr))
        weighted_var_num += var * len(vals)
        weighted_count += len(vals)
        max_var = max(max_var, var)
        max_bucket_size = max(max_bucket_size, len(vals))
        labels = solved[vals]
        if bool(np.any(labels)) and bool(np.any(~labels)):
            mixed += 1
    weighted_var = float(weighted_var_num / weighted_count) if weighted_count else 0.0
    return {
        "round_decimals": int(round_decimals),
        "min_bucket_size": int(min_bucket_size),
        "bucket_count": int(len(buckets)),
        "multi_sample_bucket_count": int(len(multi)),
        "multi_sample_row_fraction": float(weighted_count / max(len(target), 1)),
        "global_target_variance": global_var,
        "weighted_within_bucket_target_variance": weighted_var,
        "within_to_global_variance_ratio": float(weighted_var / global_var) if global_var > 1e-12 else 0.0,
        "max_bucket_target_variance": max_var,
        "max_bucket_size": int(max_bucket_size),
        "mixed_solved_unsolved_bucket_count": int(mixed),
        "mixed_solved_unsolved_bucket_fraction": float(mixed / max(len(multi), 1)) if multi else 0.0,
    }


def alias_audit(rows: Sequence[dict[str, Any]], sets: dict[str, tuple[np.ndarray, tuple[str, ...]]], args: argparse.Namespace) -> dict[str, Any]:
    target = np.array([row["target_value"] for row in rows], dtype=np.float64)
    solved = np.array([row["teacher_solved"] for row in rows], dtype=bool)
    valid_idx = split_indices(rows, "valid")
    out = {}
    for name in ("six_stocks", "six_stocks_plus_goal_probes"):
        x = sets[name][0]
        out[f"{name}_valid"] = alias_audit_for_matrix(
            x[valid_idx],
            target[valid_idx],
            solved[valid_idx],
            round_decimals=args.alias_round_decimals,
            min_bucket_size=args.alias_min_bucket_size,
        )
    return out


def collect_action_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    split = provider_level_split(args.provider_checkpoint, [level.level_id for level in levels], args.seed)
    rows: list[dict[str, Any]] = []
    teacher_cache: dict[tuple[str, SokobanState], tuple[Any, float]] = {}
    split_counts: Counter[str] = Counter()
    t0 = perf_counter()

    for level_index, level in enumerate(levels):
        if level.level_id in split["train_levels"]:
            split_name = "train"
        elif level.level_id in split["valid_levels"]:
            split_name = "valid"
        else:
            continue
        model = SokobanGoalDirectedSDModel(level)
        states_used = 0
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            legal = tuple(model.legal_pushes(state))
            if len(legal) <= 1:
                continue
            if states_used >= args.max_action_states_per_level:
                break
            states_used += 1
            stock = stock_proxy(model, state).as_array().astype(np.float64)
            action_rows = []
            for action in legal:
                next_state = model.step_state(state, action)
                key = (level.level_id, next_state)
                if key not in teacher_cache:
                    teacher_cache[key] = teacher_value(args, model, next_state)
                teacher, next_value = teacher_cache[key]
                next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
                immediate = stock_reward(stock, next_stock)
                teacher_q = float(immediate + args.gamma * next_value)
                fixed_stock_q = float(immediate + args.gamma * stock_value(next_stock))
                action_rows.append(
                    {
                        "action_flat_index": int(action.flat_index),
                        "teacher_q": teacher_q,
                        "fixed_stock_q": fixed_stock_q,
                        "reward_only_q": float(immediate),
                        "next_teacher_failure_type": str(teacher.failure_type),
                        "next_teacher_solved": bool(teacher.solved),
                    }
                )
            teacher_best = max(action_rows, key=lambda row: (row["teacher_q"], -row["action_flat_index"]))
            fixed_best = max(action_rows, key=lambda row: (row["fixed_stock_q"], -row["action_flat_index"]))
            reward_best = max(action_rows, key=lambda row: (row["reward_only_q"], -row["action_flat_index"]))
            teacher_values = np.array([row["teacher_q"] for row in action_rows], dtype=np.float64)
            fixed_values = np.array([row["fixed_stock_q"] for row in action_rows], dtype=np.float64)
            reward_values = np.array([row["reward_only_q"] for row in action_rows], dtype=np.float64)
            fixed_corr = (
                float(np.corrcoef(teacher_values, fixed_values)[0, 1])
                if len(action_rows) > 1 and np.std(fixed_values) > 1e-12 and np.std(teacher_values) > 1e-12
                else 0.0
            )
            reward_corr = (
                float(np.corrcoef(teacher_values, reward_values)[0, 1])
                if len(action_rows) > 1 and np.std(reward_values) > 1e-12 and np.std(teacher_values) > 1e-12
                else 0.0
            )
            rows.append(
                {
                    "level_index": int(level_index),
                    "level_id": level.level_id,
                    "split": split_name,
                    "state_depth": int(state_depth),
                    "legal_actions": int(len(action_rows)),
                    "teacher_best_action": int(teacher_best["action_flat_index"]),
                    "fixed_stock_best_action": int(fixed_best["action_flat_index"]),
                    "reward_only_best_action": int(reward_best["action_flat_index"]),
                    "teacher_best_q": float(teacher_best["teacher_q"]),
                    "fixed_selected_teacher_q": float(
                        next(row["teacher_q"] for row in action_rows if row["action_flat_index"] == fixed_best["action_flat_index"])
                    ),
                    "reward_selected_teacher_q": float(
                        next(row["teacher_q"] for row in action_rows if row["action_flat_index"] == reward_best["action_flat_index"])
                    ),
                    "fixed_action_q_corr": fixed_corr,
                    "reward_only_action_q_corr": reward_corr,
                    "action_rows_sample": action_rows[:12],
                }
            )
            split_counts[split_name] += 1
        print(
            json.dumps(
                {
                    "level_index": level_index,
                    "level_id": level.level_id,
                    "split": split_name,
                    "goal_alignment_action_states": states_used,
                },
                ensure_ascii=False,
            )
        )

    return rows, {"split_counts": dict(split_counts), "collection_elapsed_sec": float(perf_counter() - t0)}


def summarize_action_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name in ("train", "valid"):
        split_rows = [row for row in rows if row["split"] == split_name]
        if not split_rows:
            out[split_name] = {}
            continue
        fixed_hit = [row["fixed_stock_best_action"] == row["teacher_best_action"] for row in split_rows]
        reward_hit = [row["reward_only_best_action"] == row["teacher_best_action"] for row in split_rows]
        fixed_regret = [row["teacher_best_q"] - row["fixed_selected_teacher_q"] for row in split_rows]
        reward_regret = [row["teacher_best_q"] - row["reward_selected_teacher_q"] for row in split_rows]
        out[split_name] = {
            "states": int(len(split_rows)),
            "mean_legal_actions": float(np.mean([row["legal_actions"] for row in split_rows])),
            "fixed_stock_q_top1_agreement": float(np.mean(fixed_hit)),
            "reward_only_top1_agreement": float(np.mean(reward_hit)),
            "fixed_stock_q_mean_teacher_regret": float(np.mean(fixed_regret)),
            "reward_only_mean_teacher_regret": float(np.mean(reward_regret)),
            "fixed_stock_q_mean_action_corr": float(np.mean([row["fixed_action_q_corr"] for row in split_rows])),
            "reward_only_mean_action_corr": float(np.mean([row["reward_only_action_q_corr"] for row in split_rows])),
        }
    return out


def build_verdict(
    predictive: dict[str, Any],
    aliasing: dict[str, Any],
    action_summary: dict[str, Any],
) -> dict[str, Any]:
    six_valid = predictive["six_stocks"]["valid"]
    all_valid = predictive["all_existing_plus_goal_probes"]["valid"]
    alias = aliasing["six_stocks_valid"]
    action_valid = action_summary.get("valid", {})
    failed = []
    if six_valid.get("r2", 0.0) < 0.50:
        failed.append("six_stock_teacher_value_r2_below_0.50")
    if alias.get("within_to_global_variance_ratio", 1.0) > 0.60:
        failed.append("six_stock_alias_target_variance_ratio_above_0.60")
    if action_valid and action_valid.get("fixed_stock_q_top1_agreement", 0.0) < 0.50:
        failed.append("fixed_stock_action_top1_agreement_below_0.50")

    probe_gain = all_valid.get("r2", 0.0) - six_valid.get("r2", 0.0)
    status = "PASS" if not failed else "PARTIAL"
    interpretation = (
        "current_six_stock_abstraction_is_goal_aligned_under_this_audit"
        if status == "PASS"
        else "current_six_stock_abstraction_is_directionally_goal_aligned_but_insufficient_for_task_planning"
    )
    return {
        "status": status,
        "failed_checks": failed,
        "thresholds": {
            "six_stock_teacher_value_valid_r2_min": 0.50,
            "six_stock_alias_target_variance_ratio_max": 0.60,
            "fixed_stock_action_top1_agreement_min": 0.50,
        },
        "observed": {
            "six_stock_teacher_value_valid_r2": six_valid.get("r2", 0.0),
            "all_existing_plus_goal_probe_valid_r2": all_valid.get("r2", 0.0),
            "probe_valid_r2_gain_over_six_stock": probe_gain,
            "six_stock_alias_target_variance_ratio": alias.get("within_to_global_variance_ratio", 0.0),
            "fixed_stock_action_top1_agreement": action_valid.get("fixed_stock_q_top1_agreement"),
        },
        "interpretation": interpretation,
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    state_rows, state_collection = collect_state_rows(args)
    sets = feature_sets(state_rows)
    predictive = evaluate_feature_sets(state_rows, sets, alpha=args.ridge_alpha)
    aliasing = alias_audit(state_rows, sets, args)
    action_rows, action_collection = collect_action_rows(args)
    action_summary = summarize_action_rows(action_rows)
    verdict = build_verdict(predictive, aliasing, action_summary)

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_sd_goal_alignment_audit_v1",
        },
        "config": {
            "provider_checkpoint": relative_path(args.provider_checkpoint, ROOT),
            "levels": relative_path(args.levels, ROOT),
            "max_levels": args.max_levels,
            "state_depth": args.state_depth,
            "max_states_per_level": args.max_states_per_level,
            "teacher_max_expansions": args.teacher_max_expansions,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "unsolved_value": args.unsolved_value,
            "ridge_alpha": args.ridge_alpha,
            "max_action_states_per_level": args.max_action_states_per_level,
            "seed": args.seed,
        },
        "audit_contract": {
            "purpose": "test whether the current SD abstraction distinguishes task-goal relevant state and action consequences",
            "teacher_usage": "offline_label_only_not_online_planning",
            "sd_transition_training_changed": False,
            "sd_mcts_evaluation_changed": False,
            "primary_current_abstraction": list(STOCK_NAMES),
            "candidate_probe_features": list(GOAL_PROBE_FEATURE_NAMES),
        },
        "state_collection": state_collection,
        "predictive_alignment": predictive,
        "aliasing": aliasing,
        "action_collection": action_collection,
        "action_alignment": action_summary,
        "verdict": verdict,
        "action_rows_sample": action_rows[:20],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "verdict": verdict}, ensure_ascii=False))


if __name__ == "__main__":
    main()
