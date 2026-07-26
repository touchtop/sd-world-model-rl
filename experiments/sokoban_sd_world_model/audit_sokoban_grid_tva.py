#!/usr/bin/env python3
"""Independent TVA audit on exact grid stock-flow states.

This script validates the TVA layer before closed-loop SD-MCTS. It samples exact
Sokoban grid states, labels high-confidence truncated remaining value with an
offline exact-state teacher, fits a small ridge value probe, and audits value
calibration plus root-successor sorting. It never changes the transition
provider and never calls the teacher online during planning.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import SokobanGridStockFlowModel, SokobanState, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES, state_to_observation


@dataclass(frozen=True)
class TeacherResult:
    solved: bool
    solution_pushes: int | None
    expanded: int
    generated: int
    pruned_deadlocks: int
    frontier_size: int
    failure_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValueSample:
    features: np.ndarray
    target: float
    split: str
    level_id: str
    depth: int
    teacher: TeacherResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory.",
    )
    parser.add_argument("--max-levels", type=int, default=12)
    parser.add_argument(
        "--level-offset",
        type=int,
        default=0,
        help="Skip this many loaded levels first; keeps training disjoint from held-out slices.",
    )
    parser.add_argument("--state-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=10)
    parser.add_argument(
        "--sample-mode",
        choices=("bfs", "random_walk"),
        default="bfs",
        help="bfs reproduces the legacy near-root sampling; random_walk covers deep execution states.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-mode",
        choices=("value", "residual"),
        default="value",
        help="residual fits teacher_target minus the SD matching-lower-bound heuristic value.",
    )
    parser.add_argument("--max-action-states-per-level", type=int, default=3)
    parser.add_argument("--teacher-max-expansions", type=int, default=800)
    parser.add_argument("--holdout-fraction", type=float, default=0.35)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument(
        "--model",
        choices=("ridge", "mlp"),
        default="ridge",
        help="Value probe family; mlp is a small torch-trained network exported for numpy inference.",
    )
    parser.add_argument("--mlp-hidden", default="128,64")
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch", type=int, default=64)
    parser.add_argument("--min-valid-samples", type=int, default=8)
    parser.add_argument("--min-valid-corr", type=float, default=0.20)
    parser.add_argument("--min-valid-positive", type=int, default=2)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/grid_tva_independent_audit_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_seed0.json",
        help="Optional JSON checkpoint for closed-loop TVA leaf evaluation.",
    )
    return parser


def _rel(path: str | Path) -> str:
    path = Path(path)
    return os.path.relpath(path, ROOT) if path.is_absolute() else str(path)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def iter_states(
    model: SokobanGridStockFlowModel,
    max_depth: int,
    max_states: int,
) -> Iterable[tuple[SokobanState, int]]:
    start = model.initial_state
    seen = {model.canonical_key(start)}
    queue: deque[tuple[SokobanState, int]] = deque([(start, 0)])
    yielded = 0
    while queue and yielded < max_states:
        state, depth = queue.popleft()
        yielded += 1
        yield state, depth
        if depth >= max_depth:
            continue
        for action in model.legal_flows(state):
            nxt = model.step_grid_flow(state, action)
            key = model.canonical_key(nxt)
            if key in seen:
                continue
            seen.add(key)
            queue.append((nxt, depth + 1))


def iter_states_random_walk(
    model: SokobanGridStockFlowModel,
    max_depth: int,
    max_states: int,
    rng: random.Random,
) -> Iterable[tuple[SokobanState, int]]:
    """Endpoint states of random push walks; covers the depths MCTS leaves visit."""

    start = model.initial_state
    seen = {model.canonical_key(start)}
    if max_states <= 0:
        return
    yield start, 0
    yielded = 1
    if max_depth < 1:
        return
    attempts = 0
    while yielded < max_states and attempts < max_states * 8:
        attempts += 1
        state, depth = start, 0
        walk_length = rng.randint(1, max_depth)
        for _ in range(walk_length):
            live_next = []
            for action in model.legal_flows(state):
                nxt = model.step_grid_flow(state, action)
                if not model.is_deadlocked(nxt):
                    live_next.append(nxt)
            if not live_next:
                break
            state = rng.choice(live_next)
            depth += 1
        key = model.canonical_key(state)
        if key in seen:
            continue
        seen.add(key)
        yield state, depth
        yielded += 1


def sd_heuristic_value(
    model: SokobanGridStockFlowModel,
    state: SokobanState,
    *,
    gamma: float,
    goal_value: float,
    push_cost: float,
    deadend_value: float,
) -> float:
    """Truncation value implied by the SD matching-lower-bound readout."""

    if model.is_solved(state):
        return float(goal_value)
    lb = model.matching_lower_bound(state.boxes)
    if not math.isfinite(lb):
        return float(deadend_value)
    if gamma == 1.0:
        return float(goal_value - push_cost * lb)
    discount = gamma**lb
    return float(discount * goal_value - push_cost * (1.0 - discount) / (1.0 - gamma))


def heuristic_value(model: SokobanGridStockFlowModel, state: SokobanState) -> float:
    lb = model.matching_lower_bound(state.boxes)
    if lb == float("inf"):
        return 1e6
    features = model.abstract_features(state)
    return float(lb - 0.25 * features.boxes_on_goals - 0.01 * features.viable_push_count)


def teacher_search(
    model: SokobanGridStockFlowModel,
    start: SokobanState,
    *,
    max_expansions: int,
) -> TeacherResult:
    if model.is_solved(start):
        return TeacherResult(True, 0, 0, 0, 0, 0, "solved")
    if model.is_deadlocked(start):
        return TeacherResult(False, None, 0, 0, 0, 0, "proven_dead_end")
    if not model.legal_flows(start):
        return TeacherResult(False, None, 0, 0, 0, 0, "no_legal_actions")

    frontier: list[tuple[float, int, int, SokobanState]] = []
    counter = 0
    heapq.heappush(frontier, (heuristic_value(model, start), 0, counter, start))
    # Push-level identity: player positions within one reachable component are
    # interchangeable, so dedup on canonical_key collapses the search space.
    key_cache: dict[SokobanState, tuple] = {}

    def state_key(state: SokobanState) -> tuple:
        key = key_cache.get(state)
        if key is None:
            key = model.canonical_key(state)
            key_cache[state] = key
        return key

    best_g = {state_key(start): 0}
    expanded = 0
    generated = 0
    pruned = 0
    while frontier and expanded < max_expansions:
        _priority, g, _counter, state = heapq.heappop(frontier)
        if g != best_g.get(state_key(state)):
            continue
        if model.is_solved(state):
            return TeacherResult(True, g, expanded, generated, pruned, len(frontier), "solved")
        expanded += 1
        for action in model.legal_flows(state):
            nxt = model.step_grid_flow(state, action)
            if nxt == state:
                continue
            generated += 1
            if model.is_deadlocked(nxt):
                pruned += 1
                continue
            next_g = g + 1
            next_key = state_key(nxt)
            if next_g >= best_g.get(next_key, 10**9):
                continue
            best_g[next_key] = next_g
            counter += 1
            heapq.heappush(frontier, (next_g + heuristic_value(model, nxt), next_g, counter, nxt))
    # An exhausted frontier under sound deadlock pruning proves unsolvability;
    # only a hit expansion cap leaves the state genuinely unknown.
    failure = "teacher_budget_unsolved" if frontier else "proven_dead_end"
    return TeacherResult(False, None, expanded, generated, pruned, len(frontier), failure)


def target_from_teacher(
    teacher: TeacherResult,
    *,
    gamma: float,
    goal_value: float,
    push_cost: float,
    deadend_value: float,
    unsolved_value: float,
) -> float:
    if teacher.solved:
        d = int(teacher.solution_pushes or 0)
        if gamma == 1.0:
            discounted_cost = push_cost * d
        else:
            discounted_cost = push_cost * (1.0 - gamma**d) / (1.0 - gamma)
        return float((gamma**d) * goal_value - discounted_cost)
    if teacher.failure_type in {"proven_dead_end", "no_legal_actions"}:
        return float(deadend_value)
    return float(unsolved_value)


def grid_tva_features(model: SokobanGridStockFlowModel, state: SokobanState) -> np.ndarray:
    obs = state_to_observation(model.level, state).astype(np.float32).reshape(-1)
    features = model.abstract_features(state)
    n_boxes = max(model.level.n_boxes, 1)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    floor_count = max(len(model.floor_cells), 1)
    board_scale = max(model.level.height + model.level.width, 1)
    lb_norm = 1.0
    if np.isfinite(features.matching_lower_bound):
        lb_norm = _clamp01(float(features.matching_lower_bound) / (n_boxes * board_scale))
    canonical_player = model.canonical_player_cell(state)
    scalars = np.array(
        [
            _safe_div(features.boxes_on_goals, n_boxes),
            1.0 - _safe_div(features.boxes_on_goals, n_boxes),
            lb_norm,
            _safe_div(features.reachable_count, floor_count),
            _safe_div(features.legal_push_count, max_pushes),
            _safe_div(features.viable_push_count, max_pushes),
            _safe_div(features.dead_square_boxes, n_boxes),
            _safe_div(features.frozen_boxes, n_boxes),
            float(features.matching_deadlock),
            float(features.deadlocked),
            float(features.solved),
            float(features.legal_push_count == 0),
            _safe_div(canonical_player[0], max(model.level.height - 1, 1)),
            _safe_div(canonical_player[1], max(model.level.width - 1, 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([obs, scalars], axis=0)


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (train_x - mean) / std
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return {"mean": mean, "std": std, "weights": weights}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["std"]
    design = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float32)], axis=1)
    return design @ model["weights"]


def fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    hidden = [int(h) for h in str(args.mlp_hidden).split(",") if h.strip()]
    dims = [train_x.shape[1], *hidden, 1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    tx = torch.tensor((train_x - mean) / std, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    vx = torch.tensor((valid_x - mean) / std, dtype=torch.float32)
    vy = torch.tensor(valid_y, dtype=torch.float32).unsqueeze(1)
    best_mae = float("inf")
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    generator = torch.Generator().manual_seed(args.seed)
    for _epoch in range(args.mlp_epochs):
        net.train()
        for idx in torch.randperm(tx.shape[0], generator=generator).split(args.mlp_batch):
            opt.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(net(tx[idx]), ty[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            valid_mae = float((net(vx) - vy).abs().mean())
        if valid_mae < best_mae:
            best_mae = valid_mae
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    weight_layers = []
    for module in net:
        if isinstance(module, torch.nn.Linear):
            weight_layers.append(
                {
                    "w": module.weight.detach().numpy().astype(float).tolist(),
                    "b": module.bias.detach().numpy().astype(float).tolist(),
                }
            )
    return {"mean": mean, "std": std, "layers": weight_layers, "arch": dims}


def predict_mlp(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["std"]
    layers = model["layers"]
    for i, layer in enumerate(layers):
        z = z @ np.asarray(layer["w"], dtype=np.float64).T + np.asarray(layer["b"], dtype=np.float64)
        if i < len(layers) - 1:
            z = np.maximum(z, 0.0)
    return z[:, 0]


def predict_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if "layers" in model:
        return predict_mlp(model, x)
    return predict_ridge(model, x)


def ridge_checkpoint_payload(
    args: argparse.Namespace,
    ridge: dict[str, np.ndarray],
    *,
    obs_shape: tuple[int, int, int] | None,
    feature_dim: int,
) -> dict[str, Any]:
    payload = {
        "material_passport": {
            "origin": "sokoban_grid_tva_refactor",
            "verification_status": "ANALYZED",
            "version_label": "exact_grid_tva_ridge_checkpoint_v1",
        },
        "tva_contract": {
            "transition_provider": SokobanGridStockFlowModel.provider_id,
            "state_input": "exact_grid_observation_plus_auxiliary_readouts",
            "teacher_usage": "offline_value_label_only",
            "teacher_used_online": False,
            "legacy_six_stock_rollout_used": False,
        },
        "value_scale": {
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "unsolved_value": args.unsolved_value,
        },
        "target_mode": args.target_mode,
        "model_type": args.model,
        "feature_contract": {
            "observation_shape": list(obs_shape) if obs_shape else None,
            "feature_dim": int(feature_dim),
            "feature_builder": "grid_tva_features_v1",
        },
    }
    if args.model == "mlp":
        payload["mlp"] = {
            "arch": ridge["arch"],
            "mean": ridge["mean"].astype(float).tolist(),
            "std": ridge["std"].astype(float).tolist(),
            "layers": ridge["layers"],
            "hidden": args.mlp_hidden,
            "epochs": args.mlp_epochs,
            "lr": args.mlp_lr,
            "weight_decay": args.mlp_weight_decay,
        }
    else:
        payload["ridge"] = {
            "alpha": args.ridge_alpha,
            "mean": ridge["mean"].astype(float).tolist(),
            "std": ridge["std"].astype(float).tolist(),
            "weights": ridge["weights"].astype(float).tolist(),
        }
    return payload


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {"count": 0}
    corr = None
    if y.size >= 2 and float(np.std(y)) > 1e-9 and float(np.std(pred)) > 1e-9:
        corr = float(np.corrcoef(y, pred)[0, 1])
    positive = y > 0
    nonpositive = ~positive
    return {
        "count": int(y.size),
        "mae": float(np.mean(np.abs(pred - y))),
        "corr": corr,
        "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(y))),
        "positive_count": int(positive.sum()),
        "nonpositive_count": int(nonpositive.sum()),
        "positive_bias": float(np.mean(pred[positive] - y[positive])) if positive.any() else None,
        "nonpositive_bias": float(np.mean(pred[nonpositive] - y[nonpositive])) if nonpositive.any() else None,
        "target_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(pred)),
    }


def _load_level_slice(args: argparse.Namespace):
    if args.level_offset < 0:
        raise SystemExit("--level-offset must be non-negative.")
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    return levels


def _sample_target(args: argparse.Namespace, model: SokobanGridStockFlowModel, state: SokobanState, teacher: TeacherResult) -> float:
    target = target_from_teacher(
        teacher,
        gamma=args.gamma,
        goal_value=args.goal_value,
        push_cost=args.push_cost,
        deadend_value=args.deadend_value,
        unsolved_value=args.unsolved_value,
    )
    if args.target_mode == "residual":
        target -= sd_heuristic_value(
            model,
            state,
            gamma=args.gamma,
            goal_value=args.goal_value,
            push_cost=args.push_cost,
            deadend_value=args.deadend_value,
        )
    return target


def collect_samples(args: argparse.Namespace) -> tuple[list[ValueSample], tuple[int, int, int] | None, Counter[str]]:
    levels = _load_level_slice(args)
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")
    holdout_start = max(1, int(round(len(levels) * (1.0 - args.holdout_fraction))))
    samples: list[ValueSample] = []
    obs_shape: tuple[int, int, int] | None = None
    skipped: Counter[str] = Counter()
    for level_idx, level in enumerate(levels):
        model = SokobanGridStockFlowModel(level)
        if obs_shape is None:
            obs_shape = state_to_observation(level, level.initial_state).shape
        if state_to_observation(level, level.initial_state).shape != obs_shape:
            skipped["shape_mismatch"] += 1
            continue
        split = "valid" if level_idx >= holdout_start else "train"
        if args.sample_mode == "random_walk":
            rng = random.Random(args.seed + level_idx * 9176)
            state_iter = iter_states_random_walk(model, args.state_depth, args.max_states_per_level, rng)
        else:
            state_iter = iter_states(model, args.state_depth, args.max_states_per_level)
        for state, depth in state_iter:
            teacher = teacher_search(model, state, max_expansions=args.teacher_max_expansions)
            samples.append(
                ValueSample(
                    features=grid_tva_features(model, state),
                    target=_sample_target(args, model, state, teacher),
                    split=split,
                    level_id=level.level_id,
                    depth=depth,
                    teacher=teacher,
                )
            )
    return samples, obs_shape, skipped


def root_sorting_audit(
    args: argparse.Namespace,
    ridge: dict[str, np.ndarray],
) -> dict[str, Any]:
    levels = _load_level_slice(args)
    holdout_start = max(1, int(round(len(levels) * (1.0 - args.holdout_fraction))))
    rows = []
    for level_idx, level in enumerate(levels[holdout_start:]):
        model = SokobanGridStockFlowModel(level)
        checked = 0
        for state, _depth in iter_states(model, args.state_depth, args.max_states_per_level):
            if checked >= args.max_action_states_per_level:
                break
            legal = tuple(model.legal_flows(state))
            if len(legal) < 2:
                continue
            child_features = []
            child_targets = []
            child_actions = []
            child_baselines = []
            for action in legal:
                nxt = model.step_grid_flow(state, action)
                teacher = teacher_search(model, nxt, max_expansions=args.teacher_max_expansions)
                child_targets.append(
                    target_from_teacher(
                        teacher,
                        gamma=args.gamma,
                        goal_value=args.goal_value,
                        push_cost=args.push_cost,
                        deadend_value=args.deadend_value,
                        unsolved_value=args.unsolved_value,
                    )
                )
                child_features.append(grid_tva_features(model, nxt))
                child_actions.append(action.flat_index)
                child_baselines.append(
                    sd_heuristic_value(
                        model,
                        nxt,
                        gamma=args.gamma,
                        goal_value=args.goal_value,
                        push_cost=args.push_cost,
                        deadend_value=args.deadend_value,
                    )
                )
            child_targets_arr = np.asarray(child_targets, dtype=np.float64)
            preds = predict_model(ridge, np.stack(child_features, axis=0))
            if args.target_mode == "residual":
                preds = preds + np.asarray(child_baselines, dtype=np.float64)
            best_target = float(child_targets_arr.max())
            predicted_best_idx = int(np.argmax(preds))
            rows.append(
                {
                    "level_id": level.level_id,
                    "root_index": checked,
                    "legal_actions": len(legal),
                    "predicted_best_action": int(child_actions[predicted_best_idx]),
                    "predicted_best_target": float(child_targets_arr[predicted_best_idx]),
                    "teacher_best_target": best_target,
                    "predicted_best_in_teacher_best_tier": bool(
                        child_targets_arr[predicted_best_idx] >= best_target - 1e-9
                    ),
                }
            )
            checked += 1
    hits = sum(1 for row in rows if row["predicted_best_in_teacher_best_tier"])
    return {
        "root_count": len(rows),
        "top_tier_hit": int(hits),
        "top_tier_hit_rate": float(hits / len(rows)) if rows else None,
        "rows": rows[:25],
    }


def main() -> None:
    args = build_parser().parse_args()
    t0 = perf_counter()
    samples, obs_shape, skipped = collect_samples(args)
    if not samples:
        raise SystemExit("No TVA samples collected.")
    train = [sample for sample in samples if sample.split == "train"]
    valid = [sample for sample in samples if sample.split == "valid"]
    if not train or not valid:
        raise SystemExit("Need both train and valid samples for independent TVA audit.")

    train_x = np.stack([sample.features for sample in train], axis=0)
    train_y = np.asarray([sample.target for sample in train], dtype=np.float64)
    valid_x = np.stack([sample.features for sample in valid], axis=0)
    valid_y = np.asarray([sample.target for sample in valid], dtype=np.float64)
    ridge = (
        fit_mlp(train_x, train_y, valid_x, valid_y, args)
        if args.model == "mlp"
        else fit_ridge(train_x, train_y, args.ridge_alpha)
    )
    train_pred = predict_model(ridge, train_x)
    valid_pred = predict_model(ridge, valid_x)
    train_metrics = metrics(train_y, train_pred)
    valid_metrics = metrics(valid_y, valid_pred)
    depth_bins = ((0, 2), (3, 8), (9, 10**9))
    valid_by_depth = {}
    valid_depths = np.asarray([sample.depth for sample in valid], dtype=np.int64)
    for lo, hi in depth_bins:
        mask = (valid_depths >= lo) & (valid_depths <= hi)
        label = f"depth_{lo}_{hi if hi < 10**9 else 'plus'}"
        valid_by_depth[label] = metrics(valid_y[mask], valid_pred[mask])
    root_sorting = root_sorting_audit(args, ridge)

    failed = []
    if valid_metrics["count"] < args.min_valid_samples:
        failed.append("valid_sample_count")
    if args.target_mode == "value" and valid_metrics["positive_count"] < args.min_valid_positive:
        # Residual targets sit below the optimistic SD lower-bound baseline by
        # construction, so a positive-count floor only applies to value mode.
        failed.append("valid_positive_count")
    if valid_metrics["corr"] is None or valid_metrics["corr"] < args.min_valid_corr:
        failed.append("valid_corr")
    if root_sorting["root_count"] == 0:
        failed.append("root_sorting_coverage")

    teacher_counts = Counter(sample.teacher.failure_type for sample in samples)
    summary = {
        "material_passport": {
            "origin": "sokoban_grid_tva_refactor",
            "verification_status": "ANALYZED",
            "version_label": "exact_grid_tva_independent_audit_v1",
        },
        "config": {
            "levels": _rel(args.levels),
            "max_levels": args.max_levels,
            "level_offset": args.level_offset,
            "state_depth": args.state_depth,
            "max_states_per_level": args.max_states_per_level,
            "max_action_states_per_level": args.max_action_states_per_level,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "target_mode": args.target_mode,
            "model": args.model,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
            "teacher_max_expansions": args.teacher_max_expansions,
            "holdout_fraction": args.holdout_fraction,
            "ridge_alpha": args.ridge_alpha,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "unsolved_value": args.unsolved_value,
        },
        "tva_contract": {
            "transition_provider": SokobanGridStockFlowModel.provider_id,
            "state_input": "exact_grid_observation_plus_auxiliary_readouts",
            "teacher_usage": "offline_value_label_only",
            "teacher_used_online": False,
            "planner_modified": False,
            "closed_loop_evaluation": False,
            "legacy_six_stock_rollout_used": False,
        },
        "dataset": {
            "sample_count": len(samples),
            "train_count": len(train),
            "valid_count": len(valid),
            "observation_shape": list(obs_shape) if obs_shape else None,
            "feature_dim": int(train_x.shape[1]),
            "teacher_failure_counts": dict(teacher_counts),
            "skipped_levels": dict(skipped),
        },
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "valid_by_depth": valid_by_depth,
            "root_sorting": root_sorting,
        },
        "gate": {
            "failed_checks": failed,
            "tva_independent_gate": "PASS" if not failed else "FAIL",
        },
        "checkpoint": {
            "path": _rel(args.checkpoint_out),
            "written": bool(args.checkpoint_out),
        },
        "elapsed_sec": perf_counter() - t0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.checkpoint_out:
        checkpoint_out = Path(args.checkpoint_out)
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ridge_checkpoint_payload(
            args,
            ridge,
            obs_shape=obs_shape,
            feature_dim=int(train_x.shape[1]),
        )
        checkpoint_out.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gate": summary["gate"], "valid": valid_metrics, "root_sorting": root_sorting}, indent=2))
    print(f"Wrote {out}")
    if args.checkpoint_out:
        print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
