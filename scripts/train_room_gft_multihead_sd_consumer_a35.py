#!/usr/bin/env python3
"""Train an A35 multi-head neural SD-consumer checkpoint.

A35 keeps the A32-A34 offline distillation contract, but decomposes the target
into value, release-progress, loop-risk, and trap-risk heads.  The exported
checkpoint is plain JSON and is consumed online only as a bounded soft prior;
teacher/reference actions are not queried during solving.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A34_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a34_train",
    ROOT / "scripts/train_room_gft_milestone_sd_consumer_a34.py",
)
if _A34_TRAIN_SPEC is None or _A34_TRAIN_SPEC.loader is None:
    raise RuntimeError("Unable to load A34 milestone SD-consumer training helpers.")
_A34_TRAIN = importlib.util.module_from_spec(_A34_TRAIN_SPEC)
_A34_TRAIN_SPEC.loader.exec_module(_A34_TRAIN)

_A33_TRAIN = _A34_TRAIN._A33_TRAIN

from src.sokoban.room_gft_mcts import LEARNED_SD_CONSUMER_FEATURE_NAMES  # noqa: E402


DEFAULT_INPUTS = [
    *_A34_TRAIN.DEFAULT_INPUTS,
    "results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json",
]

HEAD_NAMES = ("value", "release_progress", "loop_risk", "trap_risk")
COMBINATION_WEIGHTS = {
    "value": 0.45,
    "release_progress": 1.15,
    "loop_risk": -1.05,
    "trap_risk": -1.25,
}
HEAD_OUTPUT_CLIPS = {
    "value": [-8.0, 10.0],
    "release_progress": [0.0, 4.0],
    "loop_risk": [0.0, 1.5],
    "trap_risk": [0.0, 1.5],
}


def _feature_map(row: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(LEARNED_SD_CONSUMER_FEATURE_NAMES, row["features"])
    }


def _positive(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _selected_future(row: dict[str, Any], key: str) -> float:
    if not bool(row.get("selected")):
        return 0.0
    return _positive((row.get("future_summary") or {}).get(key))


def _head_targets(row: dict[str, Any]) -> dict[str, float]:
    features = _feature_map(row)
    value = float(row["target"])
    release_progress = (
        1.00 * _positive(features["release_action_count_delta"])
        + 0.48 * _positive(features["dependency_total_delta"])
        + 0.44 * _positive(features["target_blocker_count_delta"])
        + 0.08 * min(_positive(features["slot_schedule_cost_delta"]), 8.0)
        + 0.35 * _positive(features["goal_delta"])
        + 0.70 * _selected_future(row, "future_release_gain")
        + 0.24 * _selected_future(row, "future_dependency_gain")
        + 0.24 * _selected_future(row, "future_target_blocker_gain")
        + 0.03 * min(_selected_future(row, "future_slot_gain"), 12.0)
    )
    loop_risk = (
        0.46 * features["pseudo_milestone_loop"]
        + 0.32 * features["repeated_edge"]
        + 0.30 * features["repeated_schedule_signature"]
        + 0.20 * features["reverse_recent_edge"]
        + 0.18 * features["stagnant_context"]
        + 0.18 * features["no_progress"]
        + 0.42 * _selected_future(row, "release_open_stagnation")
        + 0.25 * _selected_future(row, "no_future_milestone")
    )
    trap_risk = (
        0.36 * features["trap_risk"]
        + 0.50 * features["severe_trap_risk"]
        + 0.36 * features["generalized_failure_risk"]
        + 0.32 * features["trap_predecessor"]
        + 0.28 * features["stand_lock_risk"]
        + 0.38 * features["unrecoverable_no_progress"]
        + 0.24 * features["low_margin"]
        + 0.32 * features["hard_negative"]
    )
    return {
        "value": float(value),
        "release_progress": float(min(4.0, release_progress)),
        "loop_risk": float(min(1.5, loop_risk)),
        "trap_risk": float(min(1.5, trap_risk)),
    }


def _combined_target(heads: dict[str, float]) -> float:
    return float(
        sum(float(COMBINATION_WEIGHTS[name]) * float(heads[name]) for name in HEAD_NAMES)
    )


def collect_multihead_rows(paths: list[Path], *, horizon: int) -> list[dict[str, Any]]:
    rows = _A34_TRAIN.collect_milestone_rows(paths, horizon=horizon)
    for row in rows:
        targets = _head_targets(row)
        row["targets"] = targets
        row["combined_target"] = _combined_target(targets)
    return rows


def _top1_metrics_for(
    rows: list[dict[str, Any]],
    pred: np.ndarray,
    indices: np.ndarray,
    *,
    target_key: str,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for local_pos, row_index in enumerate(indices.tolist()):
        group_id = _A34_TRAIN._stable_group_id(rows[row_index])
        by_group.setdefault(group_id, []).append(local_pos)
    target_hits = 0
    selected_hits = 0
    selected_groups = 0
    group_count = 0
    for positions in by_group.values():
        if len(positions) < 2:
            continue
        group_count += 1
        if target_key == "combined_target":
            targets = np.asarray(
                [
                    float(rows[int(indices[pos])]["combined_target"])
                    for pos in positions
                ],
                dtype=np.float64,
            )
        else:
            targets = np.asarray(
                [
                    float(rows[int(indices[pos])]["targets"][target_key])
                    for pos in positions
                ],
                dtype=np.float64,
            )
        scores = pred[np.asarray(positions, dtype=np.int64)]
        best_score_pos = int(np.argmax(scores))
        best_target_pos = int(np.argmax(targets))
        if best_score_pos == best_target_pos:
            target_hits += 1
        selected_positions = [
            pos for pos in positions if bool(rows[int(indices[pos])].get("selected"))
        ]
        if selected_positions:
            selected_groups += 1
            selected_local = positions.index(selected_positions[0])
            if best_score_pos == selected_local:
                selected_hits += 1
    return {
        "group_count": int(group_count),
        "target_top1_accuracy": None
        if group_count == 0
        else float(target_hits / group_count),
        "selected_group_count": int(selected_groups),
        "selected_top1_accuracy": None
        if selected_groups == 0
        else float(selected_hits / selected_groups),
    }


def _predict(
    x: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    *,
    clip: list[float],
) -> np.ndarray:
    pred, _, _ = _A33_TRAIN._forward(x, weights, biases)
    low, high = float(clip[0]), float(clip[1])
    return np.clip(pred, low, high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--future-horizon", type=int, default=8)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3501)
    parser.add_argument(
        "--out-dataset",
        default="results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_dataset.json",
    )
    parser.add_argument(
        "--out-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json",
    )
    args = parser.parse_args()

    hidden_sizes = [
        int(part)
        for part in str(args.hidden_sizes).split(",")
        if part.strip()
    ]
    paths = [ROOT / path for path in args.inputs]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input traces: {missing}")
    rows = collect_multihead_rows(paths, horizon=int(args.future_horizon))
    if not rows:
        raise RuntimeError("No rows collected for A35 multi-head SD consumer.")

    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    group_ids = [_A34_TRAIN._stable_group_id(row) for row in rows]
    validation_mask = np.asarray(
        [
            _A34_TRAIN._group_is_validation(
                group_id,
                float(args.validation_fraction),
            )
            for group_id in group_ids
        ],
        dtype=bool,
    )
    if not np.any(validation_mask) or np.all(validation_mask):
        raise RuntimeError("Validation split is empty or covers all rows.")
    train_indices = np.nonzero(~validation_mask)[0]
    val_indices = np.nonzero(validation_mask)[0]
    x_train = x[train_indices]
    x_val = x[val_indices]
    mean = np.mean(x_train, axis=0)
    std = np.maximum(np.std(x_train, axis=0), 1e-6)
    x_train_std = (x_train - mean) / std
    x_val_std = (x_val - mean) / std

    head_payloads: dict[str, Any] = {}
    head_metrics: dict[str, Any] = {}
    train_preds: dict[str, np.ndarray] = {}
    val_preds: dict[str, np.ndarray] = {}
    for offset, head_name in enumerate(HEAD_NAMES):
        y = np.asarray(
            [float(row["targets"][head_name]) for row in rows],
            dtype=np.float64,
        )
        y_train = y[train_indices]
        y_val = y[val_indices]
        weights, biases, fit_metrics = _A33_TRAIN._train_mlp(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
            hidden_sizes=hidden_sizes,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed) + offset,
        )
        train_pred = _predict(
            x_train_std,
            weights,
            biases,
            clip=HEAD_OUTPUT_CLIPS[head_name],
        )
        val_pred = _predict(
            x_val_std,
            weights,
            biases,
            clip=HEAD_OUTPUT_CLIPS[head_name],
        )
        train_preds[head_name] = train_pred
        val_preds[head_name] = val_pred
        fit_metrics["train"].update(
            _top1_metrics_for(rows, train_pred, train_indices, target_key=head_name)
        )
        fit_metrics["validation"].update(
            _top1_metrics_for(rows, val_pred, val_indices, target_key=head_name)
        )
        head_metrics[head_name] = {
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
            **fit_metrics,
        }
        head_payloads[head_name] = {
            "layers": [
                {
                    "w": weight.astype(float).tolist(),
                    "b": bias.astype(float).tolist(),
                }
                for weight, bias in zip(weights, biases)
            ]
        }

    combined_train_pred = np.zeros((train_indices.size,), dtype=np.float64)
    combined_val_pred = np.zeros((val_indices.size,), dtype=np.float64)
    for head_name in HEAD_NAMES:
        combined_train_pred += (
            float(COMBINATION_WEIGHTS[head_name]) * train_preds[head_name]
        )
        combined_val_pred += (
            float(COMBINATION_WEIGHTS[head_name]) * val_preds[head_name]
        )
    combined_target = np.asarray(
        [float(row["combined_target"]) for row in rows],
        dtype=np.float64,
    )
    combined_metrics = {
        "train": {
            **_A33_TRAIN._metrics(combined_target[train_indices], combined_train_pred),
            **_top1_metrics_for(
                rows,
                combined_train_pred,
                train_indices,
                target_key="combined_target",
            ),
        },
        "validation": {
            **_A33_TRAIN._metrics(combined_target[val_indices], combined_val_pred),
            **_top1_metrics_for(
                rows,
                combined_val_pred,
                val_indices,
                target_key="combined_target",
            ),
        },
    }
    metrics = {
        "sample_count": int(len(rows)),
        "train_count": int(train_indices.size),
        "validation_count": int(val_indices.size),
        "group_count": int(len(set(group_ids))),
        "validation_group_count": int(
            len({group_ids[index] for index in val_indices.tolist()})
        ),
        "heads": head_metrics,
        "combined": combined_metrics,
    }
    contract = {
        "teacher_actions_used_online": False,
        "source_usage": "offline_trace_multihead_distillation_only",
        "feature_contract": "room_sd_candidate_features_v1",
        "training_objective": "value_release_loop_trap_multihead_regression_v1",
    }
    feature_scaling = {
        "action_box_index_norm_denominator": 16.0,
        "action_flat_index_norm_denominator": 64.0,
    }
    dataset = {
        "contract": {
            **contract,
            "dataset": "room_gft_a35_multihead_sd_consumer_dataset_v1",
        },
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "feature_scaling": feature_scaling,
        "head_names": list(HEAD_NAMES),
        "combination_weights": COMBINATION_WEIGHTS,
        "head_output_clips": HEAD_OUTPUT_CLIPS,
        "source_files": [str(path.relative_to(ROOT)) for path in paths],
        "future_horizon": int(args.future_horizon),
        "rows": rows,
        "split": {
            "seed": int(args.seed),
            "validation_fraction": float(args.validation_fraction),
            "train_count": int(train_indices.size),
            "validation_count": int(val_indices.size),
        },
        "metrics": metrics,
    }
    checkpoint = {
        "contract": {
            **contract,
            "checkpoint": "room_gft_a35_multihead_sd_consumer_mlp_v1",
        },
        "model_type": "multihead_mlp",
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "feature_scaling": feature_scaling,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "head_names": list(HEAD_NAMES),
        "heads": head_payloads,
        "combination_weights": COMBINATION_WEIGHTS,
        "head_output_clips": HEAD_OUTPUT_CLIPS,
        "training_config": {
            "hidden_sizes": hidden_sizes,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "future_horizon": int(args.future_horizon),
        },
        "metrics": metrics,
    }
    out_dataset = ROOT / args.out_dataset
    out_checkpoint = ROOT / args.out_checkpoint
    out_dataset.parent.mkdir(parents=True, exist_ok=True)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out_dataset.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    out_checkpoint.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dataset": str(out_dataset.relative_to(ROOT)),
                "out_checkpoint": str(out_checkpoint.relative_to(ROOT)),
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
