#!/usr/bin/env python3
"""Train an A34 milestone-target neural SD-consumer checkpoint.

A34 keeps the A33 neural consumer format but changes the target from immediate
SD progress to a future-window milestone signal estimated from offline traces.
Online planning with the exported checkpoint does not query teacher/reference
actions.
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

_A33_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a33_train",
    ROOT / "scripts/train_room_gft_neural_sd_consumer_a33.py",
)
if _A33_TRAIN_SPEC is None or _A33_TRAIN_SPEC.loader is None:
    raise RuntimeError("Unable to load A33 neural SD-consumer training helpers.")
_A33_TRAIN = importlib.util.module_from_spec(_A33_TRAIN_SPEC)
_A33_TRAIN_SPEC.loader.exec_module(_A33_TRAIN)

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
    "results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json",
    "results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json",
]


def _stable_group_id(row: dict[str, Any]) -> str:
    return f"{row['source_file']}::{row['version']}::{row['step']}"


def _group_is_validation(group_id: str, validation_fraction: float) -> bool:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < validation_fraction


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _row_metrics(trace_row: dict[str, Any]) -> dict[str, float]:
    schedule = trace_row.get("schedule_plan_after", {}).get("totals", {}) or {}
    slot = trace_row.get("slot_progress_after", {}) or {}
    return {
        "boxes": _as_float(trace_row.get("boxes_on_goals_after")),
        "slot_cost": _as_float(slot.get("slot_schedule_cost")),
        "dependency_total": _as_float(schedule.get("dependency_count"))
        + _as_float(schedule.get("future_dependency_count")),
        "target_blocker": _as_float(schedule.get("target_blocker_count")),
        "release_action": _as_float(schedule.get("release_action_count")),
    }


def _future_target(
    trace: list[dict[str, Any]],
    index: int,
    *,
    horizon: int,
) -> tuple[float, dict[str, float]]:
    current = _row_metrics(trace[index])
    future = [
        _row_metrics(trace[pos])
        for pos in range(index + 1, min(len(trace), index + horizon + 1))
        if "physical_action" in trace[pos]
    ]
    if not future:
        future = [current]
    max_boxes = max(row["boxes"] for row in future)
    min_slot = min(row["slot_cost"] for row in future)
    min_dep = min(row["dependency_total"] for row in future)
    min_target = min(row["target_blocker"] for row in future)
    max_release = max(row["release_action"] for row in future)
    final = future[-1]
    goal_gain = max(0.0, max_boxes - current["boxes"])
    slot_gain = max(0.0, current["slot_cost"] - min_slot)
    dep_gain = max(0.0, current["dependency_total"] - min_dep)
    target_gain = max(0.0, current["target_blocker"] - min_target)
    release_gain = max(0.0, max_release - current["release_action"])
    final_goal_loss = max(0.0, current["boxes"] - final["boxes"])
    no_future_milestone = float(
        goal_gain <= 0.0
        and release_gain <= 0.0
        and slot_gain < 1.0
        and dep_gain <= 0.0
        and target_gain <= 0.0
    )
    release_open_stagnation = float(
        current["release_action"] > 0.0
        and goal_gain <= 0.0
        and dep_gain <= 0.0
        and target_gain <= 0.0
        and slot_gain < 1.0
    )
    value = (
        5.0 * goal_gain
        + 2.2 * release_gain
        + 1.4 * dep_gain
        + 1.2 * target_gain
        + 0.18 * min(slot_gain, 12.0)
        - 2.4 * release_open_stagnation
        - 1.2 * no_future_milestone
        - 3.0 * final_goal_loss
    )
    summary = {
        "future_goal_gain": float(goal_gain),
        "future_release_gain": float(release_gain),
        "future_dependency_gain": float(dep_gain),
        "future_target_blocker_gain": float(target_gain),
        "future_slot_gain": float(slot_gain),
        "future_goal_loss": float(final_goal_loss),
        "release_open_stagnation": float(release_open_stagnation),
        "no_future_milestone": float(no_future_milestone),
    }
    return float(value), summary


def collect_milestone_rows(paths: list[Path], *, horizon: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, run in payload.items():
            if not key.endswith("_closed_loop_smoke") or not isinstance(run, dict):
                continue
            version = key.replace("_closed_loop_smoke", "")
            trace = [
                row
                for row in run.get("trace", [])
                if isinstance(row, dict) and "physical_action" in row
            ]
            for index, trace_row in enumerate(trace):
                children = trace_row.get("top_children", [])
                if not children:
                    continue
                chosen_key = trace_row.get("semantic_action", {}).get("semantic_key")
                chosen_future, future_summary = _future_target(
                    trace,
                    index,
                    horizon=horizon,
                )
                for rank, child in enumerate(children):
                    feature_dict = _A32_BUILD._feature_dict(child)
                    immediate = _A32_BUILD._target_from_child(child)
                    selected = bool(
                        child.get("semantic_action", {}).get("semantic_key")
                        == chosen_key
                    )
                    sdv = child.get("room_sd_value", {}) or {}
                    phase = child.get("room_phase_progress", {}) or {}
                    trap = child.get("room_pre_release_trap", {}) or {}
                    stand = child.get("room_stand_lock", {}) or {}
                    child_release_delta = _as_float(
                        phase.get("release_action_count_delta")
                    )
                    child_goal_delta = _as_float(phase.get("goal_delta"))
                    child_slot_delta = _as_float(phase.get("slot_schedule_cost_delta"))
                    child_dep_delta = _as_float(phase.get("dependency_total_delta"))
                    child_target_delta = _as_float(
                        phase.get("target_blocker_count_delta")
                    )
                    selected_bonus = chosen_future if selected else 0.0
                    target = (
                        0.35 * immediate
                        + selected_bonus
                        + 1.5 * max(0.0, child_goal_delta)
                        + 0.9 * max(0.0, child_release_delta)
                        + 0.45 * max(0.0, child_dep_delta)
                        + 0.40 * max(0.0, child_target_delta)
                        + 0.05 * max(0.0, min(child_slot_delta, 8.0))
                    )
                    if bool(sdv.get("pseudo_milestone_loop")):
                        target -= 2.0
                    if bool(sdv.get("hard_negative")):
                        target -= 1.5
                    if bool(phase.get("repeated_edge")) or bool(
                        phase.get("child_repeated_schedule_signature")
                    ):
                        target -= 0.7
                    if bool(trap.get("severe_trap_risk")):
                        target -= 1.4
                    if bool(stand.get("stand_lock_risk")):
                        target -= 1.0
                    rows.append(
                        {
                            "source_file": str(path),
                            "version": version,
                            "step": int(trace_row.get("step", -1)),
                            "rank": int(rank),
                            "selected": selected,
                            "features": [
                                feature_dict[name]
                                for name in LEARNED_SD_CONSUMER_FEATURE_NAMES
                            ],
                            "target": float(target),
                            "immediate_target": float(immediate),
                            "selected_future_target": float(chosen_future),
                            "future_summary": future_summary,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--future-horizon", type=int, default=8)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument(
        "--out-dataset",
        default="results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_dataset.json",
    )
    parser.add_argument(
        "--out-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json",
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
    rows = collect_milestone_rows(paths, horizon=int(args.future_horizon))
    if not rows:
        raise RuntimeError("No rows collected for A34 milestone SD consumer.")

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
    std = np.maximum(np.std(x_train, axis=0), 1e-6)
    x_train_std = (x_train - mean) / std
    x_val_std = (x_val - mean) / std

    weights, biases, fit_metrics = _A33_TRAIN._train_mlp(
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
    train_pred, _, _ = _A33_TRAIN._forward(x_train_std, weights, biases)
    val_pred, _, _ = _A33_TRAIN._forward(x_val_std, weights, biases)
    fit_metrics["train"].update(
        _A33_TRAIN._top1_metrics(rows, train_pred, train_indices)
    )
    fit_metrics["validation"].update(
        _A33_TRAIN._top1_metrics(rows, val_pred, val_indices)
    )
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
        "source_usage": "offline_trace_future_window_distillation_only",
        "feature_contract": "room_sd_candidate_features_v1",
        "training_objective": "future_window_milestone_regression_v1",
    }
    feature_scaling = {
        "action_box_index_norm_denominator": 16.0,
        "action_flat_index_norm_denominator": 64.0,
    }
    dataset = {
        "contract": {
            **contract,
            "dataset": "room_gft_a34_milestone_sd_consumer_dataset_v1",
        },
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "feature_scaling": feature_scaling,
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
            "checkpoint": "room_gft_a34_milestone_sd_consumer_mlp_v1",
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
