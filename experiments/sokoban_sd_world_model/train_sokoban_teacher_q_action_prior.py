#!/usr/bin/env python3
"""Train a teacher-Q action prior for Sokoban SD-MCTS.

The target is an offline successor action-Q label collected only on the
stage-2 train split. The checkpoint is consumed as an action-prior multiplier;
it does not provide transitions and does not call the teacher online.
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
from sokoban_learned_action_prior import (  # noqa: E402
    ACTION_PRIOR_FEATURE_NAMES,
    SokobanActionPriorNet,
    feature_summary,
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
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-provider-id", default="sokoban_teacher_q_action_prior_v1")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_seed0.pt",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def indices_for_split(rows: Sequence[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray([idx for idx, row in enumerate(rows) if row["split"] == split], dtype=np.int64)


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
    }


def rank_metrics(rows: Sequence[dict[str, Any]], predictions: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    by_state: dict[str, list[int]] = defaultdict(list)
    selected = set(int(idx) for idx in indices.tolist())
    for idx, row in enumerate(rows):
        if idx in selected:
            by_state[row["state_key"]].append(idx)
    hits = []
    regrets = []
    random_hits = []
    for idxs in by_state.values():
        if len(idxs) <= 1:
            continue
        teacher_best = max(idxs, key=lambda idx: (rows[idx]["teacher_q"], -rows[idx]["flat_index"]))
        pred_best = max(idxs, key=lambda idx: (predictions[idx], -rows[idx]["flat_index"]))
        hits.append(rows[pred_best]["flat_index"] == rows[teacher_best]["flat_index"])
        regrets.append(float(rows[teacher_best]["teacher_q"] - rows[pred_best]["teacher_q"]))
        random_hits.append(1.0 / len(idxs))
    return {
        "states": int(len(hits)),
        "top1_agreement": float(np.mean(hits)) if hits else 0.0,
        "mean_teacher_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_random_top1": float(np.mean(random_hits)) if random_hits else 0.0,
    }


def evaluate_model(
    model: SokobanActionPriorNet,
    features: torch.Tensor,
    target_mean: float,
    target_std: float,
    targets: np.ndarray,
    rows: Sequence[dict[str, Any]],
    indices: np.ndarray,
) -> dict[str, Any]:
    if len(indices) == 0:
        return {}
    with torch.no_grad():
        pred_norm = model(features).detach().cpu().numpy()
    pred = pred_norm * float(target_std) + float(target_mean)
    return {
        "regression": regression_metrics(pred[indices], targets[indices]),
        "rank": rank_metrics(rows, pred, indices),
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    t0 = perf_counter()
    rows, collection = collect_rows(args)
    signal_metrics = evaluate_signal(rows, args)
    signal_gate = build_gate(signal_metrics)
    if signal_gate["verdict"] != "PASS":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "material_passport": {
                "origin_skill": "academic-research-suite/experiment-agent",
                "origin_mode": "run",
                "verification_status": "FAILED_GATE",
                "version_label": "sokoban_teacher_q_action_prior_train_v1",
            },
            "failure_reason": "target_signal_gate_failed",
            "target_signal_gate": signal_gate,
            "collection": collection,
        }
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit("Teacher-Q action-prior training skipped because target signal gate failed.")

    features_np = np.stack([row["features"] for row in rows]).astype(np.float32)
    targets_np = np.asarray([row["teacher_q"] for row in rows], dtype=np.float32)
    train_idx = indices_for_split(rows, "train")
    valid_idx = indices_for_split(rows, "valid")
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise SystemExit("Teacher-Q action-prior training requires train and internal valid splits.")

    feature_mean = features_np[train_idx].mean(axis=0).astype(np.float32)
    feature_std = np.maximum(features_np[train_idx].std(axis=0).astype(np.float32), 1e-6)
    norm_features_np = (features_np - feature_mean) / feature_std
    target_mean = float(np.mean(targets_np[train_idx]))
    target_std = float(max(np.std(targets_np[train_idx]), 1e-6))
    norm_targets_np = (targets_np - target_mean) / target_std

    features = torch.tensor(norm_features_np, dtype=torch.float32)
    targets = torch.tensor(norm_targets_np, dtype=torch.float32)
    model = SokobanActionPriorNet(
        input_dim=len(ACTION_PRIOR_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_valid: dict[str, Any] = evaluate_model(
        model,
        features,
        target_mean,
        target_std,
        targets_np,
        rows,
        valid_idx,
    )

    def valid_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
        rank = metrics.get("rank", {})
        regression = metrics.get("regression", {})
        return (
            float(rank.get("top1_agreement", 0.0)),
            -float(rank.get("mean_teacher_regret", 1e9)),
            float(regression.get("corr", 0.0)),
        )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batch_iter(train_idx, args.batch_size, args.seed, epoch):
            idx = torch.tensor(batch, dtype=torch.long)
            pred = model(features[idx])
            loss = F.mse_loss(pred, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        current_valid = evaluate_model(
            model,
            features,
            target_mean,
            target_std,
            targets_np,
            rows,
            valid_idx,
        )
        if valid_key(current_valid) > valid_key(best_valid):
            best_valid = current_valid
            best_epoch = int(epoch + 1)
            best_state = deepcopy(model.state_dict())
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 50 == 0:
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(np.mean(losses)) if losses else 0.0,
                    "train": evaluate_model(model, features, target_mean, target_std, targets_np, rows, train_idx),
                    "valid": current_valid,
                }
            )

    model.load_state_dict(best_state)
    model.eval()
    final_train = evaluate_model(model, features, target_mean, target_std, targets_np, rows, train_idx)
    final_valid = evaluate_model(model, features, target_mean, target_std, targets_np, rows, valid_idx)
    label_counts = Counter(
        f"{row['split']}:{'best' if row['teacher_best_label'] > 0.5 else 'nonbest'}"
        for row in rows
    )
    teacher_counts = Counter(row["teacher_failure_type"] for row in rows)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_teacher_q_action_prior_train_v1",
        },
        "config": {
            "provider_checkpoint": os.path.relpath(args.provider_checkpoint, ROOT)
            if Path(args.provider_checkpoint).is_absolute()
            else args.provider_checkpoint,
            "levels": os.path.relpath(args.levels, ROOT) if Path(args.levels).is_absolute() else args.levels,
            "max_levels": args.max_levels,
            "max_train_levels": args.max_train_levels,
            "internal_holdout_fraction": args.internal_holdout_fraction,
            "state_depth": args.state_depth,
            "max_states_per_level": args.max_states_per_level,
            "max_action_states_per_level": args.max_action_states_per_level,
            "teacher_max_expansions": args.teacher_max_expansions,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "unsolved_value": args.unsolved_value,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "prior_contract": {
            "prior_id": args.value_provider_id,
            "target_source": "stage2_train_split_offline_successor_teacher_q",
            "uses_stage2_train_split_only": True,
            "uses_heldout_valid_split_for_training": False,
            "teacher_used_online": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "planner_consumption": "action_prior_multiplier_only",
        },
        "collection": collection,
        "target_signal_gate": signal_gate,
        "dataset": {
            "samples": int(len(rows)),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "label_counts": dict(label_counts),
            "teacher_counts": dict(teacher_counts),
            "target_mean": target_mean,
            "target_std": target_std,
            "feature_dim": int(features_np.shape[1]),
            "feature_names": list(ACTION_PRIOR_FEATURE_NAMES),
            "feature_summary": feature_summary(features_np, ACTION_PRIOR_FEATURE_NAMES),
        },
        "metrics": {
            "signal_audit": signal_metrics,
            "best_epoch": int(best_epoch),
            "best_valid_selection": best_valid,
            "train": final_train,
            "valid": final_valid,
            "history": history,
        },
        "elapsed_sec": float(perf_counter() - t0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    checkpoint = {
        "checkpoint_contract": {
            "value_provider_id": args.value_provider_id,
            "prior_id": args.value_provider_id,
            "target_source": "stage2_train_split_offline_successor_teacher_q",
            "uses_stage2_train_split_only": True,
            "uses_heldout_valid_split_for_training": False,
            "teacher_used_online": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "planner_consumption": "action_prior_multiplier_only",
            "target_mean": target_mean,
            "target_std": target_std,
        },
        "architecture": {
            "model_kind": "sokoban_teacher_q_action_prior_regression_mlp",
            "input_dim": len(ACTION_PRIOR_FEATURE_NAMES),
            "hidden_dim": args.hidden_dim,
        },
        "feature_names": list(ACTION_PRIOR_FEATURE_NAMES),
        "feature_stats": {
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
        "target_stats": {
            "mean": target_mean,
            "std": target_std,
        },
        "model_state_dict": model.state_dict(),
        "summary": summary,
    }
    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_out)
    print(
        json.dumps(
            {
                "samples": len(rows),
                "target_signal_gate": signal_gate,
                "train": final_train,
                "valid": final_valid,
                "out": str(out),
                "checkpoint_out": str(checkpoint_out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
