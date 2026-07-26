#!/usr/bin/env python3
"""Train a terminal value alignment network for Sokoban SD-MCTS.

This script follows the Beer Game TVA pattern used in the NeurIPS materials:
the learned value network is a terminal evaluator for truncated MCTS leaves.
The stock-flow transition provider remains unchanged. Targets are generated
offline by a diagnostic exact-state teacher and distilled into a small network
over the abstract SD state.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sokoban_tva import (
    TVA_FEATURE_NAMES,
    SokobanTvaValueNet,
    feature_summary,
    load_tva_checkpoint,
    relative_path,
    tva_feature_vector,
)
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels
from validate_sokoban_sd_design import STOCK_NAMES, iter_states, stock_proxy
from audit_stage2_learned_provider_sdmcts import LearnedStockFlowTransitionProvider
from evaluate_stage3_sdmcts_planning import SokobanTvaTerminalValueEvaluator, choose_mcts_action


@dataclass(frozen=True)
class TeacherResult:
    solved: bool
    solution_pushes: Optional[int]
    expanded: int
    generated: int
    pruned_deadlocks: int
    frontier_size: int
    failure_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TvaSample:
    features: np.ndarray
    target_value: float
    split: str
    level_id: str
    state_depth: int
    teacher: TeacherResult
    source: str = "teacher_state_sample"
    sample_weight: float = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
        help="Stage-2 provider checkpoint whose level split is reused for TVA.",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used to collect TVA states.",
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
    parser.add_argument("--solved-sample-weight", type=float, default=1.0)
    parser.add_argument("--deadend-sample-weight", type=float, default=1.0)
    parser.add_argument("--unsolved-sample-weight", type=float, default=1.0)
    parser.add_argument(
        "--augment-failed-rollouts",
        action="store_true",
        default=False,
        help="Add train-level failed closed-loop states as on-policy negative TVA samples.",
    )
    parser.add_argument(
        "--failed-rollout-tva-checkpoint",
        default="",
        help="TVA checkpoint used to generate failed closed-loop training trajectories.",
    )
    parser.add_argument("--failed-rollout-budget", type=int, default=48)
    parser.add_argument("--failed-rollout-depth", type=int, default=3)
    parser.add_argument("--failed-rollout-max-pushes", type=int, default=30)
    parser.add_argument("--failed-rollout-max-state-revisits", type=int, default=3)
    parser.add_argument("--failed-rollout-seeds", default="0")
    parser.add_argument("--failed-rollout-max-levels", type=int, default=24)
    parser.add_argument("--failed-rollout-max-samples", type=int, default=600)
    parser.add_argument("--failed-rollout-sample-weight", type=float, default=2.0)
    parser.add_argument("--closed-loop-failure-value", type=float, default=-8.0)
    parser.add_argument(
        "--closed-loop-discount-horizon",
        type=int,
        default=10,
        help="Maximum remaining failed trajectory length used to discount closed-loop failure labels.",
    )
    parser.add_argument(
        "--root-action-selection",
        choices=("visits", "value"),
        default="visits",
        help="Root action selection used when collecting failed rollout augmentation states.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-provider-id", default="sokoban_tva_value_v1")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_tva_train_seed0.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt",
        help="Torch checkpoint for the TVA value network.",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _torch_load(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def provider_level_split(provider_checkpoint: str | Path) -> Dict[str, Any]:
    payload = _torch_load(provider_checkpoint)
    summary = payload.get("summary", {})
    split = summary.get("dataset", {}).get("split", {})
    train_levels = set(split.get("train_levels", []))
    valid_levels = set(split.get("valid_levels", []))
    if not train_levels or not valid_levels:
        raise SystemExit(
            "Provider checkpoint does not contain a level split. "
            "Train the stage-2 provider with --split-mode level first."
        )
    return {
        "split_mode": split.get("split_mode", "level"),
        "train_levels": sorted(train_levels),
        "valid_levels": sorted(valid_levels),
        "provider_summary_config": summary.get("config", {}),
    }


def heuristic_value(model: SokobanGoalDirectedSDModel, state: SokobanState) -> float:
    lb = model.matching_lower_bound(state.boxes)
    if lb == float("inf"):
        return 1e6
    features = model.abstract_features(state)
    return float(lb - 0.25 * features.boxes_on_goals - 0.01 * features.viable_push_count)


def best_first_from_state(
    model: SokobanGoalDirectedSDModel,
    start: SokobanState,
    *,
    max_expansions: int,
) -> TeacherResult:
    """Solve from an arbitrary state using the diagnostic exact-state teacher."""

    if model.is_solved(start):
        return TeacherResult(True, 0, 0, 0, 0, 0, "solved")
    if model.is_deadlocked(start):
        return TeacherResult(False, None, 0, 0, 0, 0, "proven_dead_end")
    if not model.legal_pushes(start):
        return TeacherResult(False, None, 0, 0, 0, 0, "no_legal_actions")

    frontier: list[tuple[float, int, int, SokobanState]] = []
    counter = 0
    heapq.heappush(frontier, (heuristic_value(model, start), 0, counter, start))
    best_g: Dict[SokobanState, int] = {start: 0}
    expanded = 0
    generated = 0
    pruned = 0

    while frontier and expanded < max_expansions:
        _priority, g, _counter, state = heapq.heappop(frontier)
        if g != best_g.get(state):
            continue
        if model.is_solved(state):
            return TeacherResult(True, g, expanded, generated, pruned, len(frontier), "solved")
        expanded += 1
        for action in model.legal_pushes(state):
            next_state = model.step_state(state, action)
            if next_state == state:
                continue
            generated += 1
            if model.is_deadlocked(next_state):
                pruned += 1
                continue
            next_g = g + 1
            if next_g >= best_g.get(next_state, 10**9):
                continue
            best_g[next_state] = next_g
            counter += 1
            heapq.heappush(
                frontier,
                (next_g + heuristic_value(model, next_state), next_g, counter, next_state),
            )

    return TeacherResult(False, None, expanded, generated, pruned, len(frontier), "teacher_budget_unsolved")


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
            discounted_step_cost = push_cost * d
        else:
            discounted_step_cost = push_cost * (1.0 - gamma**d) / (1.0 - gamma)
        return float((gamma**d) * goal_value - discounted_step_cost)
    if teacher.failure_type in {"proven_dead_end", "no_legal_actions"}:
        return float(deadend_value)
    return float(unsolved_value)


def sample_weight_from_teacher(teacher: TeacherResult, args: argparse.Namespace) -> float:
    if teacher.solved:
        return float(args.solved_sample_weight)
    if teacher.failure_type in {"proven_dead_end", "no_legal_actions"}:
        return float(args.deadend_sample_weight)
    return float(args.unsolved_sample_weight)


def parse_csv_ints(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one integer.")
    return values


def fallback_level_split(level_ids: Sequence[str], valid_fraction: float, seed: int) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    shuffled = np.array(sorted(level_ids), dtype=object)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_fraction))) if len(shuffled) > 1 else 0
    valid = set(shuffled[:valid_count].tolist())
    train = [level_id for level_id in shuffled.tolist() if level_id not in valid]
    return {"split_mode": "fallback_level", "train_levels": sorted(train), "valid_levels": sorted(valid)}


def collect_samples(args: argparse.Namespace) -> tuple[list[TvaSample], Dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit("No levels loaded for TVA training.")

    split = provider_level_split(args.provider_checkpoint)
    all_loaded_ids = {level.level_id for level in levels}
    if not all_loaded_ids.intersection(split["train_levels"]):
        split = fallback_level_split(sorted(all_loaded_ids), valid_fraction=0.2, seed=args.seed)

    samples: list[TvaSample] = []
    teacher_counts: Counter[str] = Counter()
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
        level_samples = 0
        seen_states: set[SokobanState] = set()
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            if state in seen_states:
                continue
            seen_states.add(state)
            stock = stock_proxy(model, state).as_array().astype(np.float32)
            teacher = best_first_from_state(
                model,
                state,
                max_expansions=args.teacher_max_expansions,
            )
            target = target_from_teacher(
                teacher,
                gamma=args.gamma,
                goal_value=args.goal_value,
                push_cost=args.push_cost,
                deadend_value=args.deadend_value,
                unsolved_value=args.unsolved_value,
            )
            samples.append(
                TvaSample(
                    features=tva_feature_vector(model, state, stock),
                    target_value=target,
                    split=split_name,
                    level_id=level.level_id,
                    state_depth=int(state_depth),
                    teacher=teacher,
                    sample_weight=sample_weight_from_teacher(teacher, args),
                )
            )
            teacher_counts[teacher.failure_type] += 1
            split_counts[split_name] += 1
            level_samples += 1

        print(
            json.dumps(
                {
                    "level_index": level_index,
                    "level_id": level.level_id,
                    "split": split_name,
                    "samples": level_samples,
                },
                ensure_ascii=False,
            )
        )

    if not any(sample.split == "train" for sample in samples):
        raise SystemExit("TVA collection produced no train samples.")
    return samples, {
        "level_split": split,
        "teacher_counts": dict(teacher_counts),
        "split_counts": dict(split_counts),
        "collection_elapsed_sec": float(perf_counter() - t0),
    }


def closed_loop_failure_target(
    *,
    failure_value: float,
    gamma: float,
    step: int,
    trajectory_len: int,
    discount_horizon: int,
) -> float:
    remaining = max(trajectory_len - 1 - step, 0)
    discounted_steps = min(remaining, max(discount_horizon, 0))
    return float((gamma**discounted_steps) * failure_value)


def collect_failed_rollout_samples(
    args: argparse.Namespace,
    split: Dict[str, Any],
) -> tuple[list[TvaSample], Dict[str, Any]]:
    """Collect train-level closed-loop failure states as additional TVA labels."""

    if not args.augment_failed_rollouts:
        return [], {
            "enabled": False,
            "samples": 0,
        }
    if not args.failed_rollout_tva_checkpoint:
        raise SystemExit("--failed-rollout-tva-checkpoint is required with --augment-failed-rollouts.")

    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    train_levels = set(split["train_levels"])
    selected_levels = [level for level in levels if level.level_id in train_levels]
    selected_levels = selected_levels[: max(args.failed_rollout_max_levels, 0)]
    if not selected_levels:
        raise SystemExit("No train levels available for failed rollout augmentation.")

    provider = LearnedStockFlowTransitionProvider(Path(args.provider_checkpoint))
    terminal_value = SokobanTvaTerminalValueEvaluator(Path(args.failed_rollout_tva_checkpoint))
    seeds = parse_csv_ints(args.failed_rollout_seeds, name="--failed-rollout-seeds")
    rng_by_seed = {seed: np.random.RandomState(seed) for seed in seeds}

    samples: list[TvaSample] = []
    failure_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    t0 = perf_counter()
    for seed in seeds:
        rng = rng_by_seed[seed]
        torch.manual_seed(seed)
        for level in selected_levels:
            if len(samples) >= args.failed_rollout_max_samples:
                break
            model = SokobanGoalDirectedSDModel(level)
            state = model.initial_state
            visited: Counter[SokobanState] = Counter([state])
            trajectory: list[SokobanState] = []
            failure_type = "max_pushes"

            for _step in range(args.failed_rollout_max_pushes):
                trajectory.append(state)
                if model.is_solved(state):
                    failure_type = "solved"
                    break
                if not model.legal_pushes(state):
                    failure_type = "no_legal_actions"
                    break
                action, _info = choose_mcts_action(
                    model,
                    state,
                    provider,
                    terminal_value,
                    num_simulations=args.failed_rollout_budget,
                    max_depth=args.failed_rollout_depth,
                    puct_c=1.25,
                    gamma=args.gamma,
                    root_action_selection=args.root_action_selection,
                    rng=rng,
                )
                if action is None:
                    failure_type = "planner_returned_no_action"
                    break
                state = model.step_state(state, action)
                visited[state] += 1
                if visited[state] > args.failed_rollout_max_state_revisits:
                    trajectory.append(state)
                    failure_type = "state_revisit_limit"
                    break
            else:
                failure_type = "max_pushes"

            if failure_type == "solved":
                continue
            failure_counts[failure_type] += 1
            level_counts[level.level_id] += 1
            for step, failed_state in enumerate(trajectory):
                if len(samples) >= args.failed_rollout_max_samples:
                    break
                stock = stock_proxy(model, failed_state).as_array().astype(np.float32)
                target = closed_loop_failure_target(
                    failure_value=args.closed_loop_failure_value,
                    gamma=args.gamma,
                    step=step,
                    trajectory_len=len(trajectory),
                    discount_horizon=args.closed_loop_discount_horizon,
                )
                source = f"closed_loop_{failure_type}"
                teacher = TeacherResult(
                    solved=False,
                    solution_pushes=None,
                    expanded=0,
                    generated=0,
                    pruned_deadlocks=0,
                    frontier_size=0,
                    failure_type=source,
                )
                samples.append(
                    TvaSample(
                        features=tva_feature_vector(model, failed_state, stock),
                        target_value=target,
                        split="train",
                        level_id=level.level_id,
                        state_depth=int(step),
                        teacher=teacher,
                        source=source,
                        sample_weight=float(args.failed_rollout_sample_weight),
                    )
                )
            if len(samples) >= args.failed_rollout_max_samples:
                break
        if len(samples) >= args.failed_rollout_max_samples:
            break

    return samples, {
        "enabled": True,
        "samples": len(samples),
        "failure_counts": dict(failure_counts),
        "level_counts": dict(level_counts),
        "seeds": seeds,
        "checkpoint": relative_path(args.failed_rollout_tva_checkpoint, ROOT),
        "budget": args.failed_rollout_budget,
        "depth": args.failed_rollout_depth,
        "max_pushes": args.failed_rollout_max_pushes,
        "max_state_revisits": args.failed_rollout_max_state_revisits,
        "sample_weight": args.failed_rollout_sample_weight,
        "closed_loop_failure_value": args.closed_loop_failure_value,
        "elapsed_sec": float(perf_counter() - t0),
    }


def indices_for_split(samples: Sequence[TvaSample], split: str) -> np.ndarray:
    return np.array([idx for idx, sample in enumerate(samples) if sample.split == split], dtype=np.int64)


def batch_iter(indices: np.ndarray, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed + epoch * 1009)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if target.size == 0:
        return {}
    mae = float(np.mean(np.abs(pred - target)))
    mse = float(np.mean((pred - target) ** 2))
    if target.size > 1 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = 0.0
    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "corr": corr,
        "target_mean": float(np.mean(target)),
        "pred_mean": float(np.mean(pred)),
    }


def evaluate_model(
    model: SokobanTvaValueNet,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
) -> Dict[str, float]:
    if len(indices) == 0:
        return {}
    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        pred_norm = model(features[idx])
        pred = pred_norm.detach().cpu().numpy() * target_std + target_mean
    target = targets[idx].detach().cpu().numpy()
    return regression_metrics(pred, target)


def grouped_metrics(
    samples: Sequence[TvaSample],
    pred_values: np.ndarray,
    target_values: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices.tolist():
        groups[samples[idx].teacher.failure_type].append(idx)
    out: dict[str, dict[str, float]] = {}
    for name, group in sorted(groups.items()):
        arr = np.array(group, dtype=np.int64)
        out[name] = regression_metrics(pred_values[arr], target_values[arr])
        out[name]["samples"] = float(len(arr))
    return out


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    samples, collection = collect_samples(args)
    failed_samples, failed_collection = collect_failed_rollout_samples(
        args,
        collection["level_split"],
    )
    samples.extend(failed_samples)
    collection["failed_rollout_augmentation"] = failed_collection

    features_np = np.stack([sample.features for sample in samples]).astype(np.float32)
    targets_np = np.array([sample.target_value for sample in samples], dtype=np.float32)
    weights_np = np.array([sample.sample_weight for sample in samples], dtype=np.float32)
    train_idx = indices_for_split(samples, "train")
    valid_idx = indices_for_split(samples, "valid")

    target_mean = float(np.mean(targets_np[train_idx]))
    target_std = float(max(np.std(targets_np[train_idx]), 1.0))
    targets_norm_np = (targets_np - target_mean) / target_std

    features = torch.tensor(features_np, dtype=torch.float32)
    targets_norm = torch.tensor(targets_norm_np, dtype=torch.float32)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    sample_weights = torch.tensor(weights_np, dtype=torch.float32)
    model = SokobanTvaValueNet(input_dim=features.shape[1], hidden_dim=args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    t0 = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batch_iter(train_idx, args.batch_size, args.seed, epoch):
            idx = torch.tensor(batch, dtype=torch.long)
            pred = model(features[idx])
            raw_loss = (pred - targets_norm[idx]) ** 2
            weights = sample_weights[idx] / torch.clamp(sample_weights[idx].mean(), min=1e-12)
            loss = torch.mean(raw_loss * weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))

        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            model.eval()
            train_metrics = evaluate_model(
                model,
                features,
                targets,
                train_idx,
                target_mean=target_mean,
                target_std=target_std,
            )
            valid_metrics = evaluate_model(
                model,
                features,
                targets,
                valid_idx,
                target_mean=target_mean,
                target_std=target_std,
            )
            row = {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "train_mae": train_metrics.get("mae", 0.0),
                "valid_mae": valid_metrics.get("mae", 0.0),
                "valid_corr": valid_metrics.get("corr", 0.0),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))

    model.eval()
    with torch.no_grad():
        all_pred_norm = model(features)
        all_pred = all_pred_norm.detach().cpu().numpy() * target_std + target_mean

    metrics = {
        "train": regression_metrics(all_pred[train_idx], targets_np[train_idx]),
        "valid": regression_metrics(all_pred[valid_idx], targets_np[valid_idx])
        if len(valid_idx)
        else {},
        "valid_by_teacher_failure_type": grouped_metrics(samples, all_pred, targets_np, valid_idx)
        if len(valid_idx)
        else {},
    }

    by_split_teacher: dict[str, dict[str, int]] = {}
    for split_name in ("train", "valid"):
        counts: Counter[str] = Counter(
            sample.teacher.failure_type for sample in samples if sample.split == split_name
        )
        by_split_teacher[split_name] = dict(counts)
    by_source: dict[str, int] = dict(Counter(sample.source for sample in samples))
    weight_summary = {
        "mean": float(np.mean(weights_np)),
        "min": float(np.min(weights_np)),
        "max": float(np.max(weights_np)),
        "by_source": {
            source: {
                "samples": int(sum(1 for sample in samples if sample.source == source)),
                "mean_weight": float(np.mean([sample.sample_weight for sample in samples if sample.source == source])),
            }
            for source in sorted(by_source)
        },
    }

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": args.value_provider_id,
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
            "solved_sample_weight": args.solved_sample_weight,
            "deadend_sample_weight": args.deadend_sample_weight,
            "unsolved_sample_weight": args.unsolved_sample_weight,
            "augment_failed_rollouts": args.augment_failed_rollouts,
            "failed_rollout_tva_checkpoint": relative_path(args.failed_rollout_tva_checkpoint, ROOT)
            if args.failed_rollout_tva_checkpoint
            else "",
            "failed_rollout_budget": args.failed_rollout_budget,
            "failed_rollout_depth": args.failed_rollout_depth,
            "failed_rollout_max_pushes": args.failed_rollout_max_pushes,
            "failed_rollout_max_state_revisits": args.failed_rollout_max_state_revisits,
            "failed_rollout_seeds": args.failed_rollout_seeds,
            "failed_rollout_max_levels": args.failed_rollout_max_levels,
            "failed_rollout_max_samples": args.failed_rollout_max_samples,
            "failed_rollout_sample_weight": args.failed_rollout_sample_weight,
            "closed_loop_failure_value": args.closed_loop_failure_value,
            "closed_loop_discount_horizon": args.closed_loop_discount_horizon,
            "root_action_selection": args.root_action_selection,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "value_provider_id": args.value_provider_id,
        },
        "tva_contract": {
            "leaf_evaluation_only": True,
            "transition_provider_modified": False,
            "teacher_used_online": False,
            "target_formula": (
                "solved: gamma^d*goal_value - discounted_push_cost; "
                "dead/no-legal: deadend_value; budget-unsolved: unsolved_value; "
                "optional closed-loop failed states: gamma^remaining*closed_loop_failure_value"
            ),
            "feature_names": list(TVA_FEATURE_NAMES),
            "stock_names": list(STOCK_NAMES),
        },
        "dataset": {
            "samples": len(samples),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "feature_dim": int(features_np.shape[1]),
            "target_stats_train": {
                "mean": target_mean,
                "std": target_std,
                "min": float(np.min(targets_np[train_idx])),
                "max": float(np.max(targets_np[train_idx])),
            },
            "teacher_counts_by_split": by_split_teacher,
            "source_counts": by_source,
            "sample_weight_summary": weight_summary,
            "collection": collection,
            "feature_summary": feature_summary(features_np, TVA_FEATURE_NAMES),
        },
        "metrics": metrics,
        "history": history,
        "elapsed_sec": float(perf_counter() - t0 + collection["collection_elapsed_sec"]),
    }

    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "checkpoint_contract": {
            "value_provider_id": args.value_provider_id,
            "leaf_evaluation_only": True,
            "teacher_used_online": False,
        },
        "architecture": {
            "model_class": "SokobanTvaValueNet",
            "input_dim": int(features_np.shape[1]),
            "hidden_dim": int(args.hidden_dim),
        },
        "feature_names": list(TVA_FEATURE_NAMES),
        "target_stats": {
            "mean": target_mean,
            "std": target_std,
        },
        "model_state_dict": model.state_dict(),
        "summary": summary,
    }
    torch.save(checkpoint_payload, checkpoint_out)
    summary["checkpoint"] = {
        "saved": True,
        "path": relative_path(checkpoint_out, ROOT),
        "value_provider_id": args.value_provider_id,
    }

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
