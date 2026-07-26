#!/usr/bin/env python3
"""Build a compact learned SD-consumer checkpoint from Room-GFT traces.

The script distills A24-A31 diagnostic traces into a small linear checkpoint
for A32. It uses only offline trace summaries; online solving with the produced
checkpoint does not query teacher/reference actions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban.room_gft_mcts import LEARNED_SD_CONSUMER_FEATURE_NAMES  # noqa: E402


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any) -> float:
    return float(bool(value))


def _feature_dict(child: dict[str, Any]) -> dict[str, float]:
    phase = child.get("room_phase_progress", {}) or {}
    trap = child.get("room_pre_release_trap", {}) or {}
    sdg = child.get("room_sd_generalization", {}) or {}
    recovery = child.get("room_recoverability", {}) or {}
    stand = child.get("room_stand_lock", {}) or {}
    stage = child.get("room_stage", {}) or {}
    sdv = child.get("room_sd_value", {}) or {}
    after = trap.get("after", {}) or {}
    stand_after = stand.get("after", {}) or {}
    before_stage = str((stage.get("before") or {}).get("stage", ""))
    after_stage = str((stage.get("after") or {}).get("stage", ""))
    action = child.get("semantic_action", {}).get("physical_action", {}) or {}
    direction = int(action.get("direction", -1))
    semantic_key = child.get("semantic_action", {}).get("semantic_key", [0])
    flat_index = int(action.get("flat_index", semantic_key[0] if semantic_key else 0))
    box_index = int(action.get("box_index", 0))
    low_margin = bool(
        int(after.get("dependency_total", 0)) > 0
        and int(after.get("nonblocked_legal_push_count", 0)) <= 2
    )
    features = {
        "bias": 1.0,
        "goal_delta": _as_float(phase.get("goal_delta")),
        "release_action_count_delta": _as_float(
            phase.get("release_action_count_delta")
        ),
        "dependency_total_delta": _as_float(phase.get("dependency_total_delta")),
        "target_blocker_count_delta": _as_float(
            phase.get("target_blocker_count_delta")
        ),
        "slot_schedule_cost_delta": _as_float(
            phase.get("slot_schedule_cost_delta")
        ),
        "schedule_barrier_cost_delta": _as_float(
            phase.get("schedule_barrier_cost_delta")
        ),
        "legal_push_count_delta": _as_float(trap.get("legal_push_count_delta")),
        "after_legal_push_count": _as_float(after.get("nonblocked_legal_push_count")),
        "after_dependency_total": _as_float(after.get("dependency_total")),
        "after_target_blocker_count": _as_float(after.get("target_blocker_count")),
        "after_release_action_count": _as_float(after.get("release_action_count")),
        "after_goal_locked_stand_blocker_count": _as_float(
            stand_after.get("goal_locked_stand_blocker_count")
        ),
        "after_locked_stand_blocker_count": _as_float(
            stand_after.get("locked_stand_blocker_count")
        ),
        "after_stand_release_option_count": _as_float(
            stand_after.get("release_option_count")
        ),
        "stage_before_fill": float(before_stage == "fill"),
        "stage_before_release": float(before_stage == "release"),
        "stage_before_recovery": float(before_stage == "recovery"),
        "stage_after_release": float(after_stage == "release"),
        "stage_after_recovery": float(after_stage == "recovery"),
        "phase_progress": _bool(phase.get("phase_progress")),
        "no_progress": _bool(phase.get("no_progress")),
        "stagnant_context": _bool(phase.get("stagnant_context")),
        "repeated_schedule_signature": _bool(
            phase.get("child_repeated_schedule_signature")
        ),
        "repeated_edge": _bool(phase.get("repeated_edge")),
        "reverse_recent_edge": _bool(phase.get("reverse_recent_edge")),
        "no_release_context": _bool(phase.get("no_release_context")),
        "pseudo_milestone_loop": _bool(sdv.get("pseudo_milestone_loop")),
        "hard_negative": _bool(sdv.get("hard_negative")),
        "soft_negative": _bool(sdv.get("soft_negative")),
        "trap_risk": _bool(trap.get("trap_risk")),
        "severe_trap_risk": _bool(trap.get("severe_trap_risk")),
        "generalized_failure_risk": _bool(sdg.get("generalized_failure_risk")),
        "trap_predecessor": _bool(
            (sdg.get("successor_probe") or {}).get("trap_predecessor")
        ),
        "stand_lock_risk": _bool(stand.get("stand_lock_risk")),
        "recoverable_no_progress": _bool(recovery.get("recoverable_no_progress")),
        "unrecoverable_no_progress": _bool(
            recovery.get("unrecoverable_no_progress")
        ),
        "low_margin": float(low_margin),
        "action_direction_up": float(direction == 0),
        "action_direction_down": float(direction == 1),
        "action_direction_left": float(direction == 2),
        "action_direction_right": float(direction == 3),
        "action_box_index_norm": float(box_index) / 16.0,
        "action_flat_index_norm": float(flat_index) / 64.0,
    }
    return {
        name: float(features.get(name, 0.0))
        for name in LEARNED_SD_CONSUMER_FEATURE_NAMES
    }


def _target_from_child(child: dict[str, Any]) -> float:
    features = _feature_dict(child)
    progress = (
        4.0 * max(0.0, features["goal_delta"])
        + 2.0 * max(0.0, features["release_action_count_delta"])
        + 1.2 * max(0.0, features["dependency_total_delta"])
        + 1.1 * max(0.0, features["target_blocker_count_delta"])
        + 0.2 * max(0.0, min(features["slot_schedule_cost_delta"], 5.0))
        + 0.1 * max(0.0, features["legal_push_count_delta"])
    )
    penalty = (
        4.0 * features["pseudo_milestone_loop"]
        + 3.0 * features["hard_negative"]
        + 2.0 * features["severe_trap_risk"]
        + 1.5 * features["trap_predecessor"]
        + 1.2 * features["stand_lock_risk"]
        + 1.0 * features["unrecoverable_no_progress"]
        + 0.8 * features["low_margin"]
        + 0.6 * features["stagnant_context"]
        + 0.6 * features["repeated_schedule_signature"]
        + 0.6 * features["repeated_edge"]
        + 0.8 * features["reverse_recent_edge"]
        + 0.4 * features["no_progress"]
        + 0.3 * max(0.0, -features["slot_schedule_cost_delta"])
        + 0.5 * max(0.0, -features["dependency_total_delta"])
        + 0.5 * max(0.0, -features["target_blocker_count_delta"])
    )
    return float(progress - penalty)


def collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, run in payload.items():
            if not key.endswith("_closed_loop_smoke") or not isinstance(run, dict):
                continue
            version = key.replace("_closed_loop_smoke", "")
            for trace_row in run.get("trace", []):
                children = trace_row.get("top_children", [])
                if not children:
                    continue
                chosen_key = trace_row.get("semantic_action", {}).get(
                    "semantic_key"
                )
                for rank, child in enumerate(children):
                    feature_dict = _feature_dict(child)
                    rows.append(
                        {
                            "source_file": str(path),
                            "version": version,
                            "step": int(trace_row.get("step", -1)),
                            "rank": int(rank),
                            "selected": bool(
                                child.get("semantic_action", {}).get("semantic_key")
                                == chosen_key
                            ),
                            "features": [
                                feature_dict[name]
                                for name in LEARNED_SD_CONSUMER_FEATURE_NAMES
                            ],
                            "target": _target_from_child(child),
                        }
                    )
    return rows


def fit_linear(rows: list[dict[str, Any]], ridge: float) -> tuple[list[float], float, dict[str, Any]]:
    import numpy as np

    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["target"] for row in rows], dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    pred = x_aug @ beta
    corr = None
    if len(y) >= 2 and float(np.std(y)) > 1e-9 and float(np.std(pred)) > 1e-9:
        corr = float(np.corrcoef(y, pred)[0, 1])
    metrics = {
        "sample_count": int(len(rows)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "mae": float(np.mean(np.abs(pred - y))),
        "corr": corr,
    }
    return beta[:-1].astype(float).tolist(), float(beta[-1]), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/sokoban_sd_world_model/room_gft_xsokoban_a28_sd_value_loop_escape_l002_s80_b128_n8.json",
            "results/sokoban_sd_world_model/room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.json",
            "results/sokoban_sd_world_model/room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json",
            "results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json",
        ],
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--out-dataset",
        default="results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_dataset.json",
    )
    parser.add_argument(
        "--out-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_linear_checkpoint.json",
    )
    args = parser.parse_args()

    paths = [ROOT / path for path in args.inputs]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input traces: {missing}")
    rows = collect_rows(paths)
    if not rows:
        raise RuntimeError("No learned SD consumer rows collected.")
    weights, bias, metrics = fit_linear(rows, ridge=float(args.ridge))
    dataset = {
        "contract": {
            "dataset": "room_gft_a32_learned_sd_consumer_dataset_v1",
            "teacher_actions_used_online": False,
            "source_usage": "offline_trace_distillation_only",
            "feature_contract": "room_sd_candidate_features_v1",
        },
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "source_files": [str(path.relative_to(ROOT)) for path in paths],
        "rows": rows,
        "metrics": metrics,
    }
    checkpoint = {
        "contract": {
            "checkpoint": "room_gft_a32_learned_sd_consumer_linear_v1",
            "teacher_actions_used_online": False,
            "source_usage": "offline_trace_distillation_only",
            "feature_contract": "room_sd_candidate_features_v1",
        },
        "model_type": "linear",
        "feature_names": list(LEARNED_SD_CONSUMER_FEATURE_NAMES),
        "weights": weights,
        "bias": bias,
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
