#!/usr/bin/env python3
"""Train an A33 neural SD-consumer checkpoint from Room-GFT traces.

The checkpoint is an offline trace-distilled MLP. It is exported as plain JSON
so online planning can evaluate it without importing a training framework.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A32_BUILD_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a32_build",
    ROOT / "scripts/build_room_gft_learned_sd_consumer_a32.py",
)
if _A32_BUILD_SPEC is None or _A32_BUILD_SPEC.loader is None:
    raise RuntimeError("Unable to load A32 learned SD-consumer dataset builder.")
_A32_BUILD = importlib.util.module_from_spec(_A32_BUILD_SPEC)
_A32_BUILD_SPEC.loader.exec_module(_A32_BUILD)

from src.sokoban.room_gft_mcts import LEARNED_SD_CONSUMER_FEATURE_NAMES  # noqa: E402


DEFAULT_INPUTS = [
    "results/sokoban_sd_world_model/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a26_sd_value_ranker_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a28_sd_value_loop_escape_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json",
]


def _stable_group_id(row: dict[str, Any]) -> str:
    return f"{row['source_file']}::{row['version']}::{row['step']}"


def _group_is_validation(group_id: str, validation_fraction: float) -> bool:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < validation_fraction


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    if float(np.std(y_true)) <= 1e-9 or float(np.std(y_pred)) <= 1e-9:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(y_true.size),
        "target_mean": float(np.mean(y_true)) if y_true.size else None,
        "target_std": float(np.std(y_true)) if y_true.size else None,
        "pred_mean": float(np.mean(y_pred)) if y_pred.size else None,
        "pred_std": float(np.std(y_pred)) if y_pred.size else None,
        "mae": float(np.mean(np.abs(y_pred - y_true))) if y_true.size else None,
        "mse": float(np.mean((y_pred - y_true) ** 2)) if y_true.size else None,
        "corr": _corr(y_true, y_pred),
    }


def _top1_metrics(
    rows: list[dict[str, Any]],
    pred: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for local_pos, row_index in enumerate(indices.tolist()):
        group_id = _stable_group_id(rows[row_index])
        by_group.setdefault(group_id, []).append(local_pos)
    target_hits = 0
    selected_hits = 0
    selected_groups = 0
    group_count = 0
    for positions in by_group.values():
        if len(positions) < 2:
            continue
        group_count += 1
        targets = np.asarray(
            [float(rows[int(indices[pos])]["target"]) for pos in positions],
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


def _forward(
    x: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    activations = [x]
    preacts: list[np.ndarray] = []
    current = x
    for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
        z = current @ weight.T + bias
        preacts.append(z)
        if layer_index < len(weights) - 1:
            current = np.maximum(z, 0.0)
        else:
            current = z
        activations.append(current)
    return current.reshape(-1), activations, preacts


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_sizes: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    rs = np.random.RandomState(seed)
    layer_sizes = [int(x_train.shape[1]), *hidden_sizes, 1]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = np.sqrt(2.0 / max(fan_in, 1))
        weights.append(rs.normal(0.0, scale, size=(fan_out, fan_in)).astype(np.float64))
        biases.append(np.zeros((fan_out,), dtype=np.float64))

    m_w = [np.zeros_like(weight) for weight in weights]
    v_w = [np.zeros_like(weight) for weight in weights]
    m_b = [np.zeros_like(bias) for bias in biases]
    v_b = [np.zeros_like(bias) for bias in biases]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    n = max(int(x_train.shape[0]), 1)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        pred, activations, preacts = _forward(x_train, weights, biases)
        error = pred - y_train
        delta = (2.0 / n) * error.reshape(-1, 1)
        grad_w: list[np.ndarray] = [np.zeros_like(weight) for weight in weights]
        grad_b: list[np.ndarray] = [np.zeros_like(bias) for bias in biases]
        for layer_index in reversed(range(len(weights))):
            grad_w[layer_index] = delta.T @ activations[layer_index]
            grad_b[layer_index] = np.sum(delta, axis=0)
            if weight_decay > 0.0:
                grad_w[layer_index] += float(weight_decay) * weights[layer_index]
            if layer_index > 0:
                delta = delta @ weights[layer_index]
                delta = delta * (preacts[layer_index - 1] > 0.0)
        for layer_index in range(len(weights)):
            m_w[layer_index] = beta1 * m_w[layer_index] + (1.0 - beta1) * grad_w[layer_index]
            v_w[layer_index] = beta2 * v_w[layer_index] + (1.0 - beta2) * (grad_w[layer_index] ** 2)
            m_b[layer_index] = beta1 * m_b[layer_index] + (1.0 - beta1) * grad_b[layer_index]
            v_b[layer_index] = beta2 * v_b[layer_index] + (1.0 - beta2) * (grad_b[layer_index] ** 2)
            m_w_hat = m_w[layer_index] / (1.0 - beta1**epoch)
            v_w_hat = v_w[layer_index] / (1.0 - beta2**epoch)
            m_b_hat = m_b[layer_index] / (1.0 - beta1**epoch)
            v_b_hat = v_b[layer_index] / (1.0 - beta2**epoch)
            weights[layer_index] -= float(lr) * m_w_hat / (np.sqrt(v_w_hat) + eps)
            biases[layer_index] -= float(lr) * m_b_hat / (np.sqrt(v_b_hat) + eps)

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            train_pred, _, _ = _forward(x_train, weights, biases)
            val_pred, _, _ = _forward(x_val, weights, biases)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_mae": float(np.mean(np.abs(train_pred - y_train))),
                    "val_mae": float(np.mean(np.abs(val_pred - y_val))),
                }
            )

    train_pred, _, _ = _forward(x_train, weights, biases)
    val_pred, _, _ = _forward(x_val, weights, biases)
    fit_metrics = {
        "train": _metrics(y_train, train_pred),
        "validation": _metrics(y_val, val_pred),
        "history": history,
    }
    return weights, biases, fit_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument(
        "--out-dataset",
        default="results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_dataset.json",
    )
    parser.add_argument(
        "--out-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_mlp_checkpoint.json",
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
    rows = _A32_BUILD.collect_rows(paths)
    if not rows:
        raise RuntimeError("No rows collected for A33 neural SD consumer.")

    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["target"] for row in rows], dtype=np.float64)
    group_ids = [_stable_group_id(row) for row in rows]
    validation_mask = np.asarray(
        [
            _group_is_validation(group_id, float(args.validation_fraction))
            for group_id in group_ids
        ],
        dtype=bool,
    )
    if not np.any(validation_mask) or np.all(validation_mask):
        raise RuntimeError("Validation split is empty or covers all rows.")
    train_indices = np.nonzero(~validation_mask)[0]
    val_indices = np.nonzero(validation_mask)[0]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.maximum(std, 1e-6)
    x_train_std = (x_train - mean) / std
    x_val_std = (x_val - mean) / std

    weights, biases, fit_metrics = _train_mlp(
        x_train_std,
        y_train,
        x_val_std,
        y_val,
        hidden_sizes=hidden_sizes,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    train_pred, _, _ = _forward(x_train_std, weights, biases)
    val_pred, _, _ = _forward(x_val_std, weights, biases)
    fit_metrics["train"].update(_top1_metrics(rows, train_pred, train_indices))
    fit_metrics["validation"].update(_top1_metrics(rows, val_pred, val_indices))
    metrics = {
        "sample_count": int(len(rows)),
        "train_count": int(train_indices.size),
        "validation_count": int(val_indices.size),
        "group_count": int(len(set(group_ids))),
        "validation_group_count": int(
            len({group_ids[index] for index in val_indices.tolist()})
        ),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        **fit_metrics,
    }
    layers = [
        {
            "w": weight.astype(float).tolist(),
            "b": bias.astype(float).tolist(),
        }
        for weight, bias in zip(weights, biases)
    ]
    contract = {
        "teacher_actions_used_online": False,
        "source_usage": "offline_trace_distillation_only",
        "feature_contract": "room_sd_candidate_features_v1",
        "training_objective": "group_split_mlp_regression_on_sd_progress_target",
    }
    feature_scaling = {
        "action_box_index_norm_denominator": 16.0,
        "action_flat_index_norm_denominator": 64.0,
    }
    dataset = {
        "contract": {
            **contract,
            "dataset": "room_gft_a33_neural_sd_consumer_dataset_v1",
        },
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "feature_scaling": feature_scaling,
        "source_files": [str(path.relative_to(ROOT)) for path in paths],
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
            "checkpoint": "room_gft_a33_neural_sd_consumer_mlp_v1",
        },
        "model_type": "mlp",
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "feature_scaling": feature_scaling,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "layers": layers,
        "training_config": {
            "hidden_sizes": hidden_sizes,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
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
