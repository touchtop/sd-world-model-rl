#!/usr/bin/env python3
"""Train leaf-supervised GFT-SD TVA with censored-risk handling.

The previous GFT-TVA direct-value run regressed a single linear function onto
root-level MCTS search values.  This script instead samples actual truncated
leaf states from GFT-SD-MCTS, labels them offline, and trains a terminal value
estimator for those leaf states.  States that exhaust the teacher budget are
handled as censored hard-state risk examples, not as deterministic negative
terminal-value labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_sokoban_gft_tva_truncation_value import (  # noqa: E402
    LeafSample,
    collect_leaf_samples,
    compute_gft_sd_leaf_value,
    compute_sd_heuristic_value,
)
from evaluate_sokoban_gft_sdmcts import (  # noqa: E402
    gft_model_from_level,
    gft_sd_leaf_weights_from_args,
    gft_soft_deadlock_weights_from_args,
    gft_transport_weights_from_args,
    gft_weights_from_args,
)
from evaluate_sokoban_grid_sdmcts import _rel  # noqa: E402
from src.sokoban import SokobanGFTSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import best_first_from_state, target_from_teacher  # noqa: E402


@dataclass
class StateSample:
    level_index: int
    level_id: str
    state: SokobanState
    state_type: str
    source: str
    root_step: int = -1
    planner_blocked: bool = False


@dataclass
class LeafTvaTrainingSample:
    features: np.ndarray
    target_value: float | None
    value_label: bool
    value_weight: float
    risk_label: float
    risk_weight: float
    split: str
    level_index: int
    level_id: str
    source: str
    root_step: int
    state_type: str
    teacher_failure_type: str
    teacher_solution_pushes: int | None
    teacher_expanded: int
    gft_sd_leaf_value: float
    sd_heuristic_value: float

    def summary_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("features")
        return data


class LeafTvaNet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Linear] = []
        in_dim = int(feature_dim)
        for _ in range(int(hidden_layers)):
            layer = nn.Linear(in_dim, int(hidden_dim))
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0))
            layers.append(layer)
            in_dim = int(hidden_dim)
        self.hidden_layers = nn.ModuleList(layers)
        self.value_head = nn.Linear(in_dim, 1)
        self.risk_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        return self.value_head(h).squeeze(-1), self.risk_head(h).squeeze(-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used to collect GFT-TVA leaf states.",
    )
    parser.add_argument("--level-offset", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=20)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--budget", type=int, default=192)
    parser.add_argument("--closed-loop-steps", type=int, default=2)
    parser.add_argument("--max-leaf-states-per-level", type=int, default=12)
    parser.add_argument("--teacher-max-expansions", type=int, default=4000)
    parser.add_argument("--targeted-state-depth", type=int, default=3)
    parser.add_argument("--targeted-max-states-per-level", type=int, default=48)
    parser.add_argument("--targeted-max-blocked-per-level", type=int, default=6)
    parser.add_argument("--uct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--constraint-mode", choices=("none", "tier_a", "exact_deadlock"), default="tier_a")
    parser.add_argument("--select-by", choices=("visits", "q"), default="visits")
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--transposition", action="store_true", default=True)
    parser.add_argument("--no-transposition", action="store_false", dest="transposition")
    parser.add_argument("--leaf-value-for-sampling", choices=("gft_sd", "sd_heuristic", "zero"), default="gft_sd")
    parser.add_argument("--gft-prior", choices=("none", "gft_delta"), default="gft_delta")
    parser.add_argument("--gft-prior-temperature", type=float, default=2.0)
    parser.add_argument("--gft-prior-uniform-mix", type=float, default=0.1)
    parser.add_argument("--gft-progress-weight", type=float, default=1.0)
    parser.add_argument("--gft-goal-weight", type=float, default=1.0)
    parser.add_argument("--gft-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-capacity-weight", type=float, default=0.0)
    parser.add_argument("--gft-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-ordering-weight", type=float, default=0.0)
    parser.add_argument("--gft-commitment-weight", type=float, default=0.0)
    parser.add_argument("--gft-bottleneck-weight", type=float, default=0.0)
    parser.add_argument("--gft-revisit-weight", type=float, default=1.0)
    parser.add_argument("--gft-transport-turn-weight", type=float, default=0.10)
    parser.add_argument("--gft-transport-access-weight", type=float, default=0.25)
    parser.add_argument("--gft-transport-zone-weight", type=float, default=0.25)
    parser.add_argument("--gft-soft-deadlock-mobility-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-matching-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-capacity-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-access-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-order-weight", type=float, default=1.0)
    parser.add_argument("--gft-commitment-beta", type=float, default=1.0)
    parser.add_argument("--gft-sinkhorn-iterations", type=int, default=25)
    parser.add_argument(
        "--gft-sd-leaf-mode",
        choices=("linear_projected", "discounted_progress"),
        default="discounted_progress",
    )
    parser.add_argument("--gft-sd-leaf-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-capacity-weight", type=float, default=0.05)
    parser.add_argument("--gft-sd-leaf-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-ordering-weight", type=float, default=0.05)
    parser.add_argument("--gft-sd-leaf-commitment-weight", type=float, default=0.05)
    parser.add_argument("--solved-value-weight", type=float, default=1.0)
    parser.add_argument("--deadend-value-weight", type=float, default=1.5)
    parser.add_argument("--solved-risk-weight", type=float, default=1.0)
    parser.add_argument("--censored-risk-weight", type=float, default=1.0)
    parser.add_argument("--deadend-risk-weight", type=float, default=1.5)
    parser.add_argument("--risk-loss-weight", type=float, default=0.25)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=0.50)
    parser.add_argument("--anchor-bias-mode", choices=("none", "train_mean", "train_median"), default="train_median")
    parser.add_argument("--anchor-bias-labels", choices=("solved", "all_value"), default="solved")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-value-train-samples", type=int, default=20)
    parser.add_argument("--min-risk-classes", type=int, default=2)
    parser.add_argument("--value-provider-id", default="sokoban_gft_leaf_tva_mlp_guarded_v4")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_levels(level_ids: Sequence[str], valid_fraction: float, seed: int) -> dict[str, Any]:
    ids = list(level_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    valid_count = max(1, int(round(len(ids) * valid_fraction))) if len(ids) > 1 else 0
    valid = set(ids[:valid_count])
    train = [level_id for level_id in ids if level_id not in valid]
    return {
        "split_mode": "level_random",
        "train_levels": sorted(train),
        "valid_levels": sorted(valid),
        "valid_fraction": float(valid_fraction),
        "seed": int(seed),
    }


def state_type(model: SokobanGFTSDModel, state: SokobanState) -> str:
    if model.is_solved(state):
        return "solved"
    if model.is_deadlocked(state):
        return "deadlocked"
    if not model.legal_flows(state):
        return "no_legal_actions"
    return "active"


def normalize_leaf_samples(samples: Sequence[LeafSample]) -> list[StateSample]:
    return [
        StateSample(
            level_index=int(sample.level_index),
            level_id=sample.level_id,
            state=sample.state,
            state_type=sample.state_type,
            source="gft_sdmcts_leaf",
            root_step=int(sample.root_step),
            planner_blocked=bool(sample.planner_blocked),
        )
        for sample in samples
    ]


def collect_targeted_blocked_samples(
    levels: Sequence[Any],
    args: argparse.Namespace,
    seen_keys: set[tuple[str, tuple]],
) -> tuple[list[StateSample], dict[str, Any]]:
    out: list[StateSample] = []
    counts: Counter[str] = Counter()
    t0 = perf_counter()
    for level_index, level in enumerate(levels):
        model = gft_model_from_level(level, args)
        queue: deque[tuple[SokobanState, int]] = deque([(model.initial_state, 0)])
        visited = {model.gft_canonical_key_cached(model.initial_state)}
        level_blocked = 0
        while queue and len(visited) <= int(args.targeted_max_states_per_level):
            state, depth = queue.popleft()
            if depth >= int(args.targeted_state_depth):
                continue
            actions = sorted(model.legal_flows(state), key=lambda action: action.flat_index)
            for action in actions:
                child = model.step_grid_flow(state, action)
                child_key = model.gft_canonical_key_cached(child)
                if child_key not in visited:
                    visited.add(child_key)
                    queue.append((child, depth + 1))
                stype = state_type(model, child)
                sample_key = (level.level_id, child_key)
                if stype not in {"deadlocked", "no_legal_actions"} or sample_key in seen_keys:
                    continue
                seen_keys.add(sample_key)
                out.append(
                    StateSample(
                        level_index=int(level_index),
                        level_id=level.level_id,
                        state=child,
                        state_type=stype,
                        source="targeted_blocked_bfs",
                        root_step=-1,
                        planner_blocked=True,
                    )
                )
                counts[level.level_id] += 1
                level_blocked += 1
                if level_blocked >= int(args.targeted_max_blocked_per_level):
                    break
            if level_blocked >= int(args.targeted_max_blocked_per_level):
                break
    return out, {
        "samples": int(len(out)),
        "per_level": dict(counts),
        "elapsed_sec": float(perf_counter() - t0),
    }


def load_selected_levels(args: argparse.Namespace):
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")
    return levels


def collect_state_samples(args: argparse.Namespace) -> tuple[list[StateSample], dict[str, Any], list[Any]]:
    levels = load_selected_levels(args)
    leaf_samples, leaf_collection = collect_leaf_samples(args)
    samples = normalize_leaf_samples(leaf_samples)
    seen_keys: set[tuple[str, tuple]] = set()
    for sample in samples:
        model = gft_model_from_level(levels[sample.level_index], args)
        seen_keys.add((sample.level_id, model.gft_canonical_key_cached(sample.state)))
    targeted, targeted_collection = collect_targeted_blocked_samples(levels, args, seen_keys)
    samples.extend(targeted)
    return samples, {
        "leaf_collection": leaf_collection,
        "targeted_blocked_collection": targeted_collection,
        "total_state_samples": int(len(samples)),
        "source_counts": dict(Counter(sample.source for sample in samples)),
    }, levels


def feature_row(model: SokobanGFTSDModel, state: SokobanState, feature_names: Sequence[str]) -> np.ndarray:
    features = model.gft_tva_feature_dict(state)["features"]
    return np.asarray([float(features[name]) for name in feature_names], dtype=np.float64)


def label_samples(
    samples: Sequence[StateSample],
    levels: Sequence[Any],
    split: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[LeafTvaTrainingSample], dict[str, Any], list[str]]:
    model_by_level = {level.level_id: gft_model_from_level(level, args) for level in levels}
    feature_names: list[str] | None = None
    rows: list[LeafTvaTrainingSample] = []
    teacher_cache: dict[tuple[str, SokobanState], Any] = {}
    teacher_counts: Counter[str] = Counter()
    value_label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    t0 = perf_counter()
    train_levels = set(split["train_levels"])
    valid_levels = set(split["valid_levels"])

    for sample in samples:
        model = model_by_level[sample.level_id]
        if feature_names is None:
            feature_names = list(model.gft_tva_feature_dict(sample.state)["feature_names"])
        cache_key = (sample.level_id, sample.state)
        if cache_key not in teacher_cache:
            teacher_cache[cache_key] = best_first_from_state(
                model,
                sample.state,
                max_expansions=args.teacher_max_expansions,
            )
        teacher = teacher_cache[cache_key]
        failure_type = str(teacher.failure_type)
        teacher_counts[failure_type] += 1
        split_name = "train" if sample.level_id in train_levels else "valid" if sample.level_id in valid_levels else "skip"
        if split_name == "skip":
            continue
        target_value: float | None = None
        value_label = failure_type in {"solved", "proven_dead_end", "no_legal_actions"}
        if value_label:
            target_value = target_from_teacher(
                teacher,
                gamma=args.gamma,
                goal_value=args.goal_value,
                push_cost=args.push_cost,
                deadend_value=args.deadend_value,
                unsolved_value=args.unsolved_value,
            )
        if failure_type == "solved":
            risk_label = 0.0
            risk_weight = float(args.solved_risk_weight)
            value_weight = float(args.solved_value_weight)
        elif failure_type in {"proven_dead_end", "no_legal_actions"}:
            risk_label = 1.0
            risk_weight = float(args.deadend_risk_weight)
            value_weight = float(args.deadend_value_weight)
        elif failure_type == "teacher_budget_unsolved":
            risk_label = 1.0
            risk_weight = float(args.censored_risk_weight)
            value_weight = 0.0
        else:
            risk_label = 1.0
            risk_weight = float(args.censored_risk_weight)
            value_weight = 0.0
        rows.append(
            LeafTvaTrainingSample(
                features=feature_row(model, sample.state, feature_names),
                target_value=target_value,
                value_label=bool(value_label),
                value_weight=float(value_weight),
                risk_label=float(risk_label),
                risk_weight=float(risk_weight),
                split=split_name,
                level_index=int(sample.level_index),
                level_id=sample.level_id,
                source=sample.source,
                root_step=int(sample.root_step),
                state_type=sample.state_type,
                teacher_failure_type=failure_type,
                teacher_solution_pushes=(
                    int(teacher.solution_pushes) if teacher.solution_pushes is not None else None
                ),
                teacher_expanded=int(teacher.expanded),
                gft_sd_leaf_value=compute_gft_sd_leaf_value(model, sample.state, args),
                sd_heuristic_value=compute_sd_heuristic_value(model, sample.state, args),
            )
        )
        split_counts[split_name] += 1
        value_label_counts["value_label" if value_label else "censored_only"] += 1
    if feature_names is None:
        raise SystemExit("Could not infer GFT-TVA feature names.")
    return rows, {
        "teacher_counts": dict(teacher_counts),
        "value_label_counts": dict(value_label_counts),
        "split_counts": dict(split_counts),
        "label_elapsed_sec": float(perf_counter() - t0),
    }, feature_names


def indices_for_split(samples: Sequence[LeafTvaTrainingSample], split: str) -> np.ndarray:
    return np.asarray([idx for idx, sample in enumerate(samples) if sample.split == split], dtype=np.int64)


def safe_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size <= 1 or float(np.std(pred)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels > 0.5
    n_pos = int(np.sum(pos))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[pos]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def value_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {"samples": 0.0}
    err = pred - target
    return {
        "samples": float(target.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": safe_corr(pred, target),
        "bias": float(np.mean(err)),
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
    }


def risk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if labels.size == 0:
        return {"samples": 0.0}
    pred = scores >= 0.5
    truth = labels > 0.5
    tp = int(np.sum(pred & truth))
    tn = int(np.sum((~pred) & (~truth)))
    fp = int(np.sum(pred & (~truth)))
    fn = int(np.sum((~pred) & truth))
    pos = max(tp + fn, 1)
    neg = max(tn + fp, 1)
    return {
        "samples": float(labels.size),
        "auc": auc_score(labels, scores),
        "balanced_accuracy": float(0.5 * (tp / pos + tn / neg)),
        "risk_recall": float(tp / pos),
        "solved_recall": float(tn / neg),
        "score_mean": float(np.mean(scores)),
        "score_mean_risk": float(np.mean(scores[truth])) if np.any(truth) else 0.0,
        "score_mean_solved": float(np.mean(scores[~truth])) if np.any(~truth) else 0.0,
    }


def feature_summary(features: np.ndarray, feature_names: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean(features[:, idx])),
            "std": float(np.std(features[:, idx])),
            "min": float(np.min(features[:, idx])),
            "max": float(np.max(features[:, idx])),
        }
        for idx, name in enumerate(feature_names)
    }


def final_prediction(
    value_head: torch.Tensor,
    risk_logit: torch.Tensor,
    anchor: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    risk_prob = torch.sigmoid(risk_logit)
    guarded = value_head - float(args.risk_penalty) * risk_prob
    pred = (1.0 - float(args.anchor_weight)) * guarded + float(args.anchor_weight) * anchor
    return torch.clamp(
        pred,
        min=min(float(args.deadend_value), float(args.goal_value)),
        max=max(float(args.deadend_value), float(args.goal_value)),
    )


def train_network(
    samples: Sequence[LeafTvaTrainingSample],
    feature_names: Sequence[str],
    args: argparse.Namespace,
) -> tuple[LeafTvaNet, dict[str, Any], dict[str, np.ndarray]]:
    features = np.stack([sample.features for sample in samples]).astype(np.float64)
    train_idx = indices_for_split(samples, "train")
    valid_idx = indices_for_split(samples, "valid")
    mean = np.mean(features[train_idx], axis=0)
    std = np.maximum(np.std(features[train_idx], axis=0), 1e-6)
    z = ((features - mean) / std).astype(np.float32)
    target = np.asarray(
        [0.0 if sample.target_value is None else float(sample.target_value) for sample in samples],
        dtype=np.float32,
    )
    value_mask = np.asarray([sample.value_label for sample in samples], dtype=bool)
    value_weight = np.asarray([sample.value_weight for sample in samples], dtype=np.float32)
    risk_label = np.asarray([sample.risk_label for sample in samples], dtype=np.float32)
    risk_weight = np.asarray([sample.risk_weight for sample in samples], dtype=np.float32)
    anchor_raw = np.asarray([sample.gft_sd_leaf_value for sample in samples], dtype=np.float32)
    bias_mask = value_mask[train_idx].copy()
    if args.anchor_bias_labels == "solved":
        train_failures = np.asarray([samples[idx].teacher_failure_type for idx in train_idx], dtype=object)
        bias_mask &= train_failures == "solved"
    if np.any(bias_mask) and args.anchor_bias_mode != "none":
        residuals = target[train_idx][bias_mask] - anchor_raw[train_idx][bias_mask]
        if args.anchor_bias_mode == "train_mean":
            anchor_bias = float(np.mean(residuals))
        else:
            anchor_bias = float(np.median(residuals))
    else:
        anchor_bias = 0.0
    anchor = np.clip(
        anchor_raw + anchor_bias,
        min(float(args.deadend_value), float(args.goal_value)),
        max(float(args.deadend_value), float(args.goal_value)),
    ).astype(np.float32)

    x_t = torch.as_tensor(z)
    target_t = torch.as_tensor(target)
    value_mask_t = torch.as_tensor(value_mask)
    value_weight_t = torch.as_tensor(value_weight)
    risk_label_t = torch.as_tensor(risk_label)
    risk_weight_t = torch.as_tensor(risk_weight)
    anchor_t = torch.as_tensor(anchor)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)

    net = LeafTvaNet(features.shape[1], args.hidden_dim, args.hidden_layers)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    batch_size = max(1, int(args.batch_size))
    for epoch in range(int(args.epochs)):
        order = rng.permutation(train_idx)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            value_head, risk_logit = net(x_t[batch])
            pred = final_prediction(value_head, risk_logit, anchor_t[batch], args)
            mask = value_mask_t[batch]
            if torch.any(mask):
                weights = value_weight_t[batch][mask]
                squared = (pred[mask] - target_t[batch][mask]) ** 2
                value_loss = torch.sum(squared * weights) / torch.clamp(torch.sum(weights), min=1e-6)
            else:
                value_loss = torch.zeros((), dtype=torch.float32)
            risk_raw = F.binary_cross_entropy_with_logits(
                risk_logit,
                risk_label_t[batch],
                reduction="none",
            )
            risk_loss = torch.sum(risk_raw * risk_weight_t[batch]) / torch.clamp(
                torch.sum(risk_weight_t[batch]), min=1e-6
            )
            loss = value_loss + float(args.risk_loss_weight) * risk_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == int(args.epochs):
            with torch.no_grad():
                value_head_all, risk_logit_all = net(x_t[train_idx_t])
                pred_all = final_prediction(value_head_all, risk_logit_all, anchor_t[train_idx_t], args)
                train_value_mask = value_mask[train_idx]
                if np.any(train_value_mask):
                    train_pred_np = pred_all.detach().cpu().numpy()[train_value_mask]
                    train_target_np = target[train_idx][train_value_mask]
                    train_mae = float(np.mean(np.abs(train_pred_np - train_target_np)))
                else:
                    train_mae = 0.0
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    "train_value_mae": train_mae,
                }
            )

    with torch.no_grad():
        value_head_all, risk_logit_all = net(x_t)
        pred = final_prediction(value_head_all, risk_logit_all, anchor_t, args).detach().cpu().numpy()
        risk_prob = torch.sigmoid(risk_logit_all).detach().cpu().numpy()
        value_head_np = value_head_all.detach().cpu().numpy()

    def split_value_metrics(idx: np.ndarray) -> dict[str, float]:
        mask = value_mask[idx]
        return value_metrics(pred[idx][mask], target[idx][mask])

    def split_risk_metrics(idx: np.ndarray) -> dict[str, float]:
        return risk_metrics(risk_label[idx], risk_prob[idx])

    metrics = {
        "value": {
            "train": split_value_metrics(train_idx),
            "valid": split_value_metrics(valid_idx) if len(valid_idx) else {"samples": 0.0},
        },
        "risk": {
            "train": split_risk_metrics(train_idx),
            "valid": split_risk_metrics(valid_idx) if len(valid_idx) else {"samples": 0.0},
        },
        "prediction_summary": {
            "prediction_mean": float(np.mean(pred)),
            "prediction_min": float(np.min(pred)),
            "prediction_max": float(np.max(pred)),
            "risk_probability_mean": float(np.mean(risk_prob)),
            "value_head_mean": float(np.mean(value_head_np)),
            "anchor_mean": float(np.mean(anchor)),
            "anchor_raw_mean": float(np.mean(anchor_raw)),
            "anchor_bias": float(anchor_bias),
        },
        "history": history,
    }
    arrays = {
        "features": features,
        "mean": mean,
        "std": std,
        "prediction": pred,
        "risk_probability": risk_prob,
        "value_head": value_head_np,
        "target": target,
        "value_mask": value_mask,
        "risk_label": risk_label,
        "anchor": anchor,
        "anchor_raw": anchor_raw,
        "anchor_bias": np.asarray([anchor_bias], dtype=np.float64),
    }
    return net, metrics, arrays


def export_mlp(net: LeafTvaNet) -> dict[str, Any]:
    return {
        "hidden_layers": [
            {
                "weight": layer.weight.detach().cpu().numpy().astype(float).tolist(),
                "bias": layer.bias.detach().cpu().numpy().astype(float).tolist(),
            }
            for layer in net.hidden_layers
        ],
        "value_head": {
            "weight": net.value_head.weight.detach().cpu().numpy().astype(float).tolist(),
            "bias": net.value_head.bias.detach().cpu().numpy().astype(float).tolist(),
        },
        "risk_head": {
            "weight": net.risk_head.weight.detach().cpu().numpy().astype(float).tolist(),
            "bias": net.risk_head.bias.detach().cpu().numpy().astype(float).tolist(),
        },
        "activation": "relu",
    }


def build_summary(
    args: argparse.Namespace,
    samples: Sequence[LeafTvaTrainingSample],
    feature_names: Sequence[str],
    split: dict[str, Any],
    collection: dict[str, Any],
    labeling: dict[str, Any],
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    train_idx = indices_for_split(samples, "train")
    risk_classes = sorted(set(float(sample.risk_label) for sample in samples if sample.split == "train"))
    value_train_samples = int(np.sum(arrays["value_mask"][train_idx]))
    failed = []
    if value_train_samples < int(args.min_value_train_samples):
        failed.append("min_value_train_samples")
    if len(risk_classes) < int(args.min_risk_classes):
        failed.append("min_risk_classes")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["prediction"]).all():
        failed.append("nonfinite_features_or_predictions")
    source_counts = Counter(sample.source for sample in samples)
    teacher_counts = Counter(sample.teacher_failure_type for sample in samples)
    split_counts = Counter(sample.split for sample in samples)
    return {
        "material_passport": {
            "origin": "sokoban_gft_leaf_tva",
            "verification_status": "ANALYZED",
            "version_label": args.value_provider_id,
        },
        "config": {
            "levels": _rel(args.levels),
            "level_offset": args.level_offset,
            "max_levels": args.max_levels,
            "valid_fraction": args.valid_fraction,
            "horizon": args.horizon,
            "budget": args.budget,
            "closed_loop_steps": args.closed_loop_steps,
            "max_leaf_states_per_level": args.max_leaf_states_per_level,
            "teacher_max_expansions": args.teacher_max_expansions,
            "targeted_state_depth": args.targeted_state_depth,
            "targeted_max_states_per_level": args.targeted_max_states_per_level,
            "targeted_max_blocked_per_level": args.targeted_max_blocked_per_level,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "unsolved_value": args.unsolved_value,
            "constraint_mode": args.constraint_mode,
            "select_by": args.select_by,
            "revisit_penalty": args.revisit_penalty,
            "transposition": bool(args.transposition),
            "leaf_value_for_sampling": args.leaf_value_for_sampling,
            "gft_prior": args.gft_prior,
            "gft_prior_temperature": args.gft_prior_temperature,
            "gft_prior_uniform_mix": args.gft_prior_uniform_mix,
            "gft_prior_weights": gft_weights_from_args(args),
            "gft_transport_cost_weights": gft_transport_weights_from_args(args),
            "gft_soft_deadlock_weights": gft_soft_deadlock_weights_from_args(args),
            "gft_commitment_beta": args.gft_commitment_beta,
            "gft_sinkhorn_iterations": args.gft_sinkhorn_iterations,
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
            "risk_loss_weight": args.risk_loss_weight,
            "risk_penalty": args.risk_penalty,
            "anchor_weight": args.anchor_weight,
            "anchor_bias_mode": args.anchor_bias_mode,
            "anchor_bias_labels": args.anchor_bias_labels,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "value_provider_id": args.value_provider_id,
        },
        "tva_contract": {
            "value_provider_id": args.value_provider_id,
            "leaf_evaluation_only": True,
            "target_source": "offline_teacher_labels_on_actual_gft_sdmcts_leaf_states",
            "value_label_source": ["teacher_solved", "proven_dead_end", "no_legal_actions"],
            "censored_risk_source": ["teacher_budget_unsolved"],
            "teacher_budget_unsolved_as_value_target": False,
            "risk_head_used_for_guarded_prediction": True,
            "anchor_bias_mode": args.anchor_bias_mode,
            "anchor_bias_labels": args.anchor_bias_labels,
            "leaf_formula": (
                "V_leaf=clip((1-anchor_weight)*(value_head-risk_penalty*risk_probability)"
                "+anchor_weight*(V_SD+anchor_bias))"
            ),
            "target_mode": "direct_value",
            "paper_alignment": "leaf_terminal_value_distillation_with_censored_risk_guard",
            "mcts_leaf_states_used": True,
            "teacher_used_online": False,
            "teacher_used_for_training_target": True,
            "transition_provider_modified": False,
            "sd_provider": SokobanGFTSDModel.provider_id,
            "transition_provider": SokobanGFTSDModel.transition_provider_id,
            "feature_contract": "gft_tva_feature_vector_v1",
            "feature_names": list(feature_names),
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
        },
        "gate": {
            "tva_training_gate": "PASS" if not failed else "FAIL",
            "failed_checks": failed,
            "thresholds": {
                "min_value_train_samples": args.min_value_train_samples,
                "min_risk_classes": args.min_risk_classes,
            },
        },
        "dataset": {
            "samples": int(len(samples)),
            "train_samples": int(split_counts.get("train", 0)),
            "valid_samples": int(split_counts.get("valid", 0)),
            "value_train_samples": value_train_samples,
            "feature_dim": int(arrays["features"].shape[1]),
            "level_split": split,
            "source_counts": dict(source_counts),
            "teacher_counts": dict(teacher_counts),
            "collection": collection,
            "labeling": labeling,
            "feature_summary": feature_summary(arrays["features"], feature_names),
        },
        "metrics": metrics,
        "sample_preview": [sample.summary_dict() for sample in samples[:8]],
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= float(args.anchor_weight) <= 1.0:
        raise SystemExit("--anchor-weight must be between 0 and 1.")
    set_seed(args.seed)
    t0 = perf_counter()
    state_samples, collection, levels = collect_state_samples(args)
    split = split_levels([level.level_id for level in levels], args.valid_fraction, args.seed)
    samples, labeling, feature_names = label_samples(state_samples, levels, split, args)
    if not samples:
        raise SystemExit("Leaf TVA collection produced no labelled samples.")
    net, metrics, arrays = train_network(samples, feature_names, args)
    summary = build_summary(args, samples, feature_names, split, collection, labeling, metrics, arrays)
    summary["elapsed_sec"] = float(perf_counter() - t0)

    checkpoint = {
        "value_provider_id": args.value_provider_id,
        "model_type": "mlp_guarded_v1",
        "target_mode": "direct_value",
        "feature_contract": {
            "contract": "gft_tva_feature_vector_v1",
            "feature_names": list(feature_names),
            "feature_dim": int(len(feature_names)),
        },
        "value_scale": {
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
        },
        "checkpoint_contract": summary["tva_contract"],
        "normalization": {
            "mean": arrays["mean"].astype(float).tolist(),
            "std": arrays["std"].astype(float).tolist(),
        },
        "mlp": export_mlp(net),
        "risk_guard": {
            "risk_penalty": float(args.risk_penalty),
            "anchor_weight": float(args.anchor_weight),
            "anchor_bias": float(arrays["anchor_bias"][0]),
            "anchor_bias_mode": args.anchor_bias_mode,
            "anchor_bias_labels": args.anchor_bias_labels,
            "clip_to_value_scale": True,
        },
        "summary": summary,
    }
    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_out.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["checkpoint"] = {
        "saved": True,
        "path": _rel(checkpoint_out),
        "value_provider_id": args.value_provider_id,
    }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "dataset": {
                    "samples": summary["dataset"]["samples"],
                    "value_train_samples": summary["dataset"]["value_train_samples"],
                    "teacher_counts": summary["dataset"]["teacher_counts"],
                },
                "metrics": summary["metrics"],
                "checkpoint": summary["checkpoint"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {out}")
    print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
