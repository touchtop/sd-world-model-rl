#!/usr/bin/env python3
"""Train a state value network from train-split offline successor teacher-Q.

This is not a strict search-statistics TVA target. It tests the next auditable
route after primitive action priors failed: compress the same offline action-Q
signal into a state/option-level terminal value consumed only at SD-MCTS leaves.
The transition provider is not modified and the exact-state teacher is never
called online.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_sokoban_train_action_target_signal import (  # noqa: E402
    build_gate,
    collect_rows,
    evaluate_signal,
)
from sokoban_learned_action_prior import ACTION_PRIOR_FEATURE_NAMES  # noqa: E402
from sokoban_tva import (  # noqa: E402
    TVA_FEATURE_NAMES,
    SokobanTvaValueNet,
    feature_summary,
    relative_path,
)
from train_sokoban_tva import batch_iter  # noqa: E402


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
    parser.add_argument("--max-train-levels", type=int, default=12)
    parser.add_argument("--internal-holdout-fraction", type=float, default=0.25)
    parser.add_argument("--state-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=12)
    parser.add_argument("--max-action-states-per-level", type=int, default=5)
    parser.add_argument("--teacher-max-expansions", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--target-aggregation",
        choices=("max", "softmax"),
        default="max",
        help="Aggregate action-Q rows into one state value target.",
    )
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--positive-target-sample-weight", type=float, default=1.0)
    parser.add_argument("--nonpositive-target-sample-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-provider-id", default="sokoban_teacher_q_state_value_v1")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_teacher_q_state_value_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_teacher_q_state_value_seed0.pt",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def indices_for_split(samples: Sequence[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray([idx for idx, row in enumerate(samples) if row["split"] == split], dtype=np.int64)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.size == 0:
        return {}
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    return {
        "samples": float(pred.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "corr": (
            float(np.corrcoef(pred, target)[0, 1])
            if pred.size > 1 and np.std(pred) > 1e-12 and np.std(target) > 1e-12
            else 0.0
        ),
        "target_mean": float(np.mean(target)),
        "pred_mean": float(np.mean(pred)),
    }


def _softmax_value(values: np.ndarray, temperature: float) -> float:
    temp = max(float(temperature), 1e-6)
    logits = values / temp
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-12)
    return float(np.dot(weights, values))


def state_samples_from_action_rows(
    rows: Sequence[dict[str, Any]],
    *,
    target_aggregation: str,
    softmax_temperature: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["state_key"]].append(row)

    state_samples: list[dict[str, Any]] = []
    state_feature_dim = len(TVA_FEATURE_NAMES)
    if ACTION_PRIOR_FEATURE_NAMES[:state_feature_dim] != TVA_FEATURE_NAMES:
        raise ValueError("Action-prior feature prefix no longer matches TVA_FEATURE_NAMES.")

    for state_key, state_rows in sorted(grouped.items()):
        values = np.asarray([row["teacher_q"] for row in state_rows], dtype=np.float64)
        if target_aggregation == "max":
            target_value = float(np.max(values))
        elif target_aggregation == "softmax":
            target_value = _softmax_value(values, softmax_temperature)
        else:
            raise ValueError(f"Unknown target aggregation: {target_aggregation}")
        best = max(state_rows, key=lambda row: (row["teacher_q"], -row["flat_index"]))
        state_features = np.asarray(best["features"][:state_feature_dim], dtype=np.float32)
        state_samples.append(
            {
                "state_key": state_key,
                "level_id": best["level_id"],
                "split": best["split"],
                "state_depth": int(best["state_depth"]),
                "features": state_features,
                "target_value": target_value,
                "action_count": int(len(state_rows)),
                "best_flat_index": int(best["flat_index"]),
                "best_teacher_q": float(best["teacher_q"]),
                "best_teacher_solved": bool(best["teacher_solved"]),
                "best_teacher_failure_type": str(best["teacher_failure_type"]),
                "teacher_q_std": float(np.std(values)),
                "teacher_q_min": float(np.min(values)),
                "teacher_q_max": float(np.max(values)),
            }
        )
    return state_samples


def evaluate_model(
    model: SokobanTvaValueNet,
    features: torch.Tensor,
    targets_np: np.ndarray,
    indices: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    if len(indices) == 0:
        return {}
    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        pred_norm = model(features[idx]).detach().cpu().numpy()
    pred = pred_norm * float(target_std) + float(target_mean)
    return regression_metrics(pred, targets_np[indices])


def grouped_metrics(
    samples: Sequence[dict[str, Any]],
    pred_values: np.ndarray,
    target_values: np.ndarray,
    indices: np.ndarray,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices.tolist():
        groups["positive_target" if target_values[idx] > 0.0 else "nonpositive_target"].append(idx)
        groups[f"teacher:{samples[idx]['best_teacher_failure_type']}"].append(idx)
    out: dict[str, dict[str, float]] = {}
    for name, group in sorted(groups.items()):
        arr = np.asarray(group, dtype=np.int64)
        out[name] = regression_metrics(pred_values[arr], target_values[arr])
        out[name]["samples"] = float(len(arr))
    return out


def build_state_value_gate(metrics: dict[str, Any], samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = metrics.get("valid", {})
    train_positive = sum(1 for row in samples if row["split"] == "train" and row["target_value"] > 0.0)
    valid_positive = sum(1 for row in samples if row["split"] == "valid" and row["target_value"] > 0.0)
    failed = []
    if len([row for row in samples if row["split"] == "train"]) < 20:
        failed.append("train_state_samples")
    if len([row for row in samples if row["split"] == "valid"]) < 5:
        failed.append("valid_state_samples")
    if train_positive < 3:
        failed.append("train_positive_state_targets")
    if valid_positive < 1:
        failed.append("valid_positive_state_targets")
    if float(valid.get("corr", 0.0)) < 0.3:
        failed.append("valid_corr")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "observed": {
            "train_state_samples": int(sum(1 for row in samples if row["split"] == "train")),
            "valid_state_samples": int(sum(1 for row in samples if row["split"] == "valid")),
            "train_positive_state_targets": int(train_positive),
            "valid_positive_state_targets": int(valid_positive),
            "valid_corr": float(valid.get("corr", 0.0)),
            "valid_mae": float(valid.get("mae", 0.0)),
        },
        "thresholds": {
            "train_state_samples_min": 20,
            "valid_state_samples_min": 5,
            "train_positive_state_targets_min": 3,
            "valid_positive_state_targets_min": 1,
            "valid_corr_min": 0.3,
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    t0 = perf_counter()
    rows, collection = collect_rows(args)
    signal_metrics = evaluate_signal(rows, args)
    signal_gate = build_gate(signal_metrics)
    if signal_gate["verdict"] != "PASS":
        return {
            "material_passport": {
                "origin_skill": "academic-research-suite/experiment-agent",
                "origin_mode": "run",
                "verification_status": "FAILED_GATE",
                "version_label": args.value_provider_id,
            },
            "failure_reason": "action_q_target_signal_gate_failed",
            "target_signal_gate": signal_gate,
            "collection": collection,
        }

    samples = state_samples_from_action_rows(
        rows,
        target_aggregation=args.target_aggregation,
        softmax_temperature=args.softmax_temperature,
    )
    if not samples:
        raise SystemExit("No state samples produced from action-Q rows.")

    features_np = np.stack([row["features"] for row in samples]).astype(np.float32)
    targets_np = np.asarray([row["target_value"] for row in samples], dtype=np.float32)
    train_idx = indices_for_split(samples, "train")
    valid_idx = indices_for_split(samples, "valid")
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise SystemExit("Teacher-Q state value training requires train and internal valid states.")

    target_mean = float(np.mean(targets_np[train_idx]))
    target_std = float(max(np.std(targets_np[train_idx]), 1.0))
    targets_norm_np = (targets_np - target_mean) / target_std

    features = torch.tensor(features_np, dtype=torch.float32)
    targets = torch.tensor(targets_norm_np, dtype=torch.float32)
    sample_weights_np = np.where(
        targets_np > 0.0,
        float(args.positive_target_sample_weight),
        float(args.nonpositive_target_sample_weight),
    ).astype(np.float32)
    sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32)

    model = SokobanTvaValueNet(input_dim=features.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_valid = evaluate_model(
        model,
        features,
        targets_np,
        valid_idx,
        target_mean=target_mean,
        target_std=target_std,
    )

    def valid_key(metric: dict[str, float]) -> tuple[float, float]:
        return (float(metric.get("corr", 0.0)), -float(metric.get("mae", 1e9)))

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batch_iter(train_idx, args.batch_size, args.seed, epoch):
            idx = torch.tensor(batch, dtype=torch.long)
            pred = model(features[idx])
            raw_loss = (pred - targets[idx]) ** 2
            weights = sample_weights[idx] / torch.clamp(sample_weights[idx].mean(), min=1e-12)
            loss = torch.mean(raw_loss * weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        current_valid = evaluate_model(
            model,
            features,
            targets_np,
            valid_idx,
            target_mean=target_mean,
            target_std=target_std,
        )
        if valid_key(current_valid) > valid_key(best_valid):
            best_valid = current_valid
            best_epoch = int(epoch + 1)
            best_state = deepcopy(model.state_dict())
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 50 == 0:
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(np.mean(losses)) if losses else 0.0,
                    "train": evaluate_model(
                        model,
                        features,
                        targets_np,
                        train_idx,
                        target_mean=target_mean,
                        target_std=target_std,
                    ),
                    "valid": current_valid,
                }
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_pred_norm = model(features).detach().cpu().numpy()
    all_pred = all_pred_norm * target_std + target_mean
    metrics = {
        "best_epoch": int(best_epoch),
        "train": regression_metrics(all_pred[train_idx], targets_np[train_idx]),
        "valid": regression_metrics(all_pred[valid_idx], targets_np[valid_idx]),
        "valid_by_group": grouped_metrics(samples, all_pred, targets_np, valid_idx),
        "history": history,
    }
    state_value_gate = build_state_value_gate(metrics, samples)

    split_counts = Counter(row["split"] for row in samples)
    teacher_counts = Counter(row["best_teacher_failure_type"] for row in samples)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": args.value_provider_id,
        },
        "config": {
            "provider_checkpoint": relative_path(args.provider_checkpoint, ROOT),
            "levels": relative_path(args.levels, ROOT),
            "max_levels": args.max_levels,
            "max_train_levels": args.max_train_levels,
            "internal_holdout_fraction": args.internal_holdout_fraction,
            "state_depth": args.state_depth,
            "max_states_per_level": args.max_states_per_level,
            "max_action_states_per_level": args.max_action_states_per_level,
            "teacher_max_expansions": args.teacher_max_expansions,
            "gamma": args.gamma,
            "target_aggregation": args.target_aggregation,
            "softmax_temperature": args.softmax_temperature,
            "positive_target_sample_weight": args.positive_target_sample_weight,
            "nonpositive_target_sample_weight": args.nonpositive_target_sample_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "value_contract": {
            "value_provider_id": args.value_provider_id,
            "target_source": "stage2_train_split_offline_successor_teacher_q_state_aggregation",
            "target_formula": "max_a Q_teacher(s,a)"
            if args.target_aggregation == "max"
            else "softmax_temperature_weighted_mean_a Q_teacher(s,a)",
            "uses_stage2_train_split_only": True,
            "uses_heldout_valid_split_for_training": False,
            "teacher_used_online": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "planner_consumption": "terminal_leaf_value_only",
            "strict_search_statistics_tva": False,
            "reason_not_strict_tva": "target comes from offline successor teacher-Q rather than MCTS root statistics",
            "feature_names": list(TVA_FEATURE_NAMES),
        },
        "collection": collection,
        "target_signal_gate": signal_gate,
        "state_value_gate": state_value_gate,
        "dataset": {
            "action_rows": int(len(rows)),
            "state_samples": int(len(samples)),
            "split_counts": dict(split_counts),
            "teacher_counts": dict(teacher_counts),
            "positive_state_targets": int(np.sum(targets_np > 0.0)),
            "target_stats_train": {
                "mean": target_mean,
                "std": target_std,
                "min": float(np.min(targets_np[train_idx])),
                "max": float(np.max(targets_np[train_idx])),
            },
            "feature_dim": int(features_np.shape[1]),
            "feature_summary": feature_summary(features_np, TVA_FEATURE_NAMES),
            "sample_weight_summary": {
                "mean": float(np.mean(sample_weights_np)),
                "min": float(np.min(sample_weights_np)),
                "max": float(np.max(sample_weights_np)),
            },
        },
        "metrics": metrics,
        "elapsed_sec": float(perf_counter() - t0),
    }

    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_contract": {
            "value_provider_id": args.value_provider_id,
            "target_source": "stage2_train_split_offline_successor_teacher_q_state_aggregation",
            "target_formula": summary["value_contract"]["target_formula"],
            "uses_stage2_train_split_only": True,
            "uses_heldout_valid_split_for_training": False,
            "teacher_used_online": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "planner_consumption": "terminal_leaf_value_only",
            "strict_search_statistics_tva": False,
            "model_class": "SokobanTvaValueNet",
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
    torch.save(checkpoint, checkpoint_out)
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
    print(
        json.dumps(
            {
                "target_signal_gate": summary.get("target_signal_gate"),
                "state_value_gate": summary.get("state_value_gate"),
                "metrics": summary.get("metrics", {}),
                "checkpoint": summary.get("checkpoint"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {out}")
    if summary.get("checkpoint", {}).get("saved"):
        print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
