#!/usr/bin/env python3
"""Audit whether simple state predicates can safely route Sokoban TVA alpha.

The audit replays saved closed-loop traces only.  It does not run MCTS and does
not let TVA choose new actions.  The goal is to compare positive fallback
levels, where fixed alpha=0.60 repaired a state-gated failure, against negative
counterexamples where the same high-alpha behavior is harmful or ineffective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sokoban_gft_sdmcts import (  # noqa: E402
    GFTTvaCheckpoint,
    discounted_progress_gft_sd_value,
    load_tva_checkpoint,
    predict_gft_tva_details,
)
from evaluate_sokoban_grid_sdmcts import _rel  # noqa: E402
from src.sokoban import SokobanGFTSDModel, SokobanPushAction, load_boxoban_levels  # noqa: E402


PRIMARY_STATE_GATED = (
    "results/sokoban_sd_world_model/"
    "gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json"
)
COMBO_FALLBACK = (
    "results/sokoban_sd_world_model/"
    "gft_sdmcts_holdout50_state_gated_then_fixed_alpha060_fallback_seed0_exploratory.json"
)
FIXED_ALPHA_GATE10 = (
    "results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_fixed_alpha060_h20_b4096_seed0.json"
)
PHASE_LEVEL_PATHS = {
    5: "results/sokoban_sd_world_model/gft_sdmcts_l5_state_gated_phase060_p20_p10_c025_seed0.json",
    6: "results/sokoban_sd_world_model/gft_sdmcts_l6_state_gated_phase060_p20_p10_c025_seed0.json",
    14: "results/sokoban_sd_world_model/gft_sdmcts_l14_state_gated_phase060_p20_p10_c025_seed0.json",
    45: "results/sokoban_sd_world_model/gft_sdmcts_l45_state_gated_phase060_p20_p10_c025_seed0.json",
    46: "results/sokoban_sd_world_model/gft_sdmcts_l46_state_gated_phase060_p20_p10_c025_seed0.json",
}
FIXED_FALLBACK_PATHS = {
    0: "results/sokoban_sd_world_model/gft_sdmcts_l00_fixed_alpha060_seed0.json",
    14: "results/sokoban_sd_world_model/gft_sdmcts_l14_fixed_alpha060_seed0.json",
    45: "results/sokoban_sd_world_model/gft_sdmcts_l45_fixed_alpha060_seed0.json",
    46: "results/sokoban_sd_world_model/gft_sdmcts_l46_fixed_alpha060_seed0.json",
}
HELDOUT_BASE_LEVEL_OFFSET = 48


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default=PRIMARY_STATE_GATED)
    parser.add_argument("--combo", default=COMBO_FALLBACK)
    parser.add_argument("--fixed-alpha-gate10", default=FIXED_ALPHA_GATE10)
    parser.add_argument("--max-prefix-steps", type=int, default=20)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/dynamic_tva_alpha_feature_audit_seed0.json",
    )
    return parser


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_levels(payload: dict[str, Any]):
    config = payload["config"]
    levels = load_boxoban_levels(
        config["levels"],
        max_levels=int(config.get("level_offset", 0)) + int(config["max_levels"]),
    )
    offset = int(config.get("level_offset", 0))
    return levels[offset:] if offset else levels


def heldout_index(payload: dict[str, Any], row: dict[str, Any]) -> int:
    level_no = int(str(row["level_id"]).rsplit(":", 1)[1])
    return int(level_no - HELDOUT_BASE_LEVEL_OFFSET)


def action_from_row(action_row: dict[str, Any]) -> SokobanPushAction:
    return SokobanPushAction(
        box_index=int(action_row["box_index"]),
        direction=int(action_row["direction"]),
    )


def leaf_weights_from_config(config: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in dict(config.get("gft_sd_leaf_weights", {})).items()}


def gft_model_from_config(level, config: dict[str, Any]) -> SokobanGFTSDModel:
    return SokobanGFTSDModel(
        level,
        transport_cost_weights=dict(config.get("gft_transport_cost_weights", {})),
        soft_deadlock_weights=dict(config.get("gft_soft_deadlock_weights", {})),
        commitment_beta=float(config.get("gft_commitment_beta", 1.0)),
        sinkhorn_iterations=int(config.get("gft_sinkhorn_iterations", 25)),
    )


def tva_details_for_state(
    model: SokobanGFTSDModel,
    state,
    payload: dict[str, Any],
    checkpoint: GFTTvaCheckpoint | None,
) -> dict[str, float]:
    config = payload["config"]
    vsd = discounted_progress_gft_sd_value(
        model,
        state,
        gamma=float(config["gamma"]),
        goal_value=float(config["goal_value"]),
        push_cost=float(config["push_cost"]),
        deadend_value=float(config["deadend_value"]),
        weights=leaf_weights_from_config(config),
    )
    if checkpoint is None:
        return {
            "vsd": float(vsd),
            "prediction": 0.0,
            "delta": 0.0,
            "risk_probability": 0.0,
            "feature_max_abs_z": 0.0,
            "feature_l2_z": 0.0,
        }
    details = predict_gft_tva_details(checkpoint, model, state)
    prediction = float(details["prediction"])
    return {
        "vsd": float(vsd),
        "prediction": prediction,
        "delta": float(prediction - vsd),
        "risk_probability": float(details.get("risk_probability", 0.0)),
        "feature_max_abs_z": float(details.get("feature_max_abs_z", 0.0)),
        "feature_l2_z": float(details.get("feature_l2_z", 0.0)),
    }


def posonly_alpha(config: dict[str, Any], tva: dict[str, float]) -> float:
    if tva["risk_probability"] <= float(config.get("tva_state_gated_positive_risk_max", 0.30)):
        if tva["delta"] >= float(config.get("tva_state_gated_positive_delta_min", 0.25)):
            blend = float(config.get("tva_state_gated_positive_blend", -1.0))
            return float(config.get("tva_value_blend", 1.0)) if blend < 0.0 else blend
    if tva["risk_probability"] >= float(config.get("tva_state_gated_negative_risk_min", 0.95)):
        if tva["delta"] <= -float(config.get("tva_state_gated_negative_delta_min", 1.0)):
            return float(config.get("tva_state_gated_negative_blend", 0.0))
    return float(config.get("tva_state_gated_default_blend", 0.0))


def phase060_alpha(config: dict[str, Any], features: dict[str, float], tva: dict[str, float]) -> tuple[float, str]:
    p = float(features["P"])
    c = float(features["C"])
    if math.isfinite(p) and p >= 20.0:
        return 0.60, "P>=20"
    if math.isfinite(p) and p <= 10.0 and c >= 0.25:
        return 0.60, "P<=10_and_C>=0.25"
    return posonly_alpha(config, tva), "posonly_or_default"


def root_summary(trace_row: dict[str, Any]) -> dict[str, Any]:
    rows = list(trace_row.get("top_children", []))
    if not rows:
        return {
            "selected_flat": None,
            "visit_best_flat": None,
            "q_best_flat": None,
            "visit_best_mean": 0.0,
            "q_best_mean": 0.0,
            "q_minus_visit_best": 0.0,
            "selected_is_visit_best": False,
            "selected_is_q_best": False,
        }
    selected = (trace_row.get("action") or {}).get("flat_index")
    visit_best = max(rows, key=lambda row: (int(row.get("visits", 0)), float(row.get("mean_value", 0.0))))
    q_best = max(rows, key=lambda row: (float(row.get("mean_value", 0.0)), int(row.get("visits", 0))))
    visit_flat = visit_best["action"]["flat_index"]
    q_flat = q_best["action"]["flat_index"]
    return {
        "selected_flat": selected,
        "visit_best_flat": int(visit_flat),
        "q_best_flat": int(q_flat),
        "visit_best_mean": float(visit_best.get("mean_value", 0.0)),
        "q_best_mean": float(q_best.get("mean_value", 0.0)),
        "q_minus_visit_best": float(q_best.get("mean_value", 0.0)) - float(visit_best.get("mean_value", 0.0)),
        "selected_is_visit_best": selected == visit_flat,
        "selected_is_q_best": selected == q_flat,
    }


def state_metrics(model: SokobanGFTSDModel, state) -> dict[str, Any]:
    features = model.gft_tva_feature_dict(state)["features"]
    abstract = model.abstract_features(state).to_dict()
    capacity = model.gft_capacity_summary_cached(state)
    ordering = model.gft_ordering_summary_cached(state)
    deadlock = model.gft_deadlock_summary_cached(state)
    commitment = model.gft_commitment_summary_cached(state)
    return {
        "features": features,
        "legal_push_count": int(abstract["legal_push_count"]),
        "viable_push_count": int(abstract["viable_push_count"]),
        "reachable_count": int(abstract["reachable_count"]),
        "boxes_on_goals": int(abstract["boxes_on_goals"]),
        "matching_lower_bound": float(abstract["matching_lower_bound"]),
        "deadlocked": bool(abstract["deadlocked"]),
        "capacity_debt": int(capacity["capacity_debt"]),
        "blocked_entry_count": int(sum(int(row["blocked_entry_count"]) for row in capacity["regions"])),
        "ordering_debt": int(ordering["ordering_debt"]),
        "entry_violations": int(len(ordering["entry_violations"])),
        "hard_deadlock_debt": int(deadlock["hard_deadlock_debt"]),
        "soft_deadlock_pressure": float(deadlock["soft_deadlock_pressure"]),
        "assignment_entropy": float(commitment["assignment_entropy"]),
        "assignment_normalized_entropy": float(commitment["normalized_entropy"]),
    }


def replay_result(path: str | Path, *, max_prefix_steps: int) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    config = payload["config"]
    levels = selected_levels(payload)
    checkpoint = None
    checkpoint_path = (
        config.get("tva_checkpoint")
        or payload.get("framework_contract", {}).get("tva_checkpoint")
        or payload.get("config", {}).get("primary_config", {}).get("tva_checkpoint")
    )
    if config.get("leaf_value") == "tva" and checkpoint_path:
        loaded = load_tva_checkpoint(checkpoint_path)
        checkpoint = loaded if isinstance(loaded, GFTTvaCheckpoint) else None

    out: dict[int, dict[str, Any]] = {}
    for row, level in zip(payload["levels"], levels):
        model = gft_model_from_config(level, config)
        state = model.initial_state
        per_step: list[dict[str, Any]] = []
        for trace_row in list(row.get("trace", []))[:max_prefix_steps]:
            metrics = state_metrics(model, state)
            tva = tva_details_for_state(model, state, payload, checkpoint)
            phase_alpha, phase_reason = phase060_alpha(config, metrics["features"], tva)
            per_step.append(
                {
                    "step": int(trace_row.get("step", len(per_step))),
                    "status": trace_row.get("status", "action"),
                    "action": trace_row.get("action"),
                    "P": float(metrics["features"]["P"]),
                    "M": float(metrics["features"]["M"]),
                    "A": float(metrics["features"]["A"]),
                    "K": float(metrics["features"]["K"]),
                    "D": float(metrics["features"]["D"]),
                    "Q": float(metrics["features"]["Q"]),
                    "C": float(metrics["features"]["C"]),
                    "legal_push_count": metrics["legal_push_count"],
                    "viable_push_count": metrics["viable_push_count"],
                    "blocked_entry_count": metrics["blocked_entry_count"],
                    "capacity_debt": metrics["capacity_debt"],
                    "ordering_debt": metrics["ordering_debt"],
                    "soft_deadlock_pressure": metrics["soft_deadlock_pressure"],
                    "tva_vsd": tva["vsd"],
                    "tva_prediction": tva["prediction"],
                    "tva_delta": tva["delta"],
                    "tva_risk_probability": tva["risk_probability"],
                    "posonly_alpha": posonly_alpha(config, tva),
                    "phase060_alpha": phase_alpha,
                    "phase060_reason": phase_reason,
                    **root_summary(trace_row),
                }
            )
            action = trace_row.get("action")
            if not action:
                break
            state = model.step_grid_flow(state, action_from_row(action))

        hidx = heldout_index(payload, row)
        out[hidx] = {
            "result_path": _rel(path),
            "level_index": int(row["level_index"]),
            "heldout_index": hidx,
            "level_id": row["level_id"],
            "solved": bool(row["solved"]),
            "status": str(row["status"]),
            "steps_executed": int(row["steps_executed"]),
            "trace_steps_available": len(row.get("trace", [])),
            "prefix_steps_analyzed": len(per_step),
            "initial": per_step[0] if per_step else None,
            "per_step": per_step,
        }
    return out


def prefix_summary(record: dict[str, Any]) -> dict[str, Any]:
    rows = list(record.get("per_step", []))
    if not rows:
        return {}
    phase_steps = [row["step"] for row in rows if float(row["phase060_alpha"]) >= 0.60]
    positive_steps = [row["step"] for row in rows if float(row["posonly_alpha"]) > 0.0]
    q_gap_steps = [row for row in rows if not row["selected_is_q_best"] and row["q_minus_visit_best"] > 1e-9]
    return {
        "initial_P": rows[0]["P"],
        "initial_M": rows[0]["M"],
        "initial_A": rows[0]["A"],
        "initial_K": rows[0]["K"],
        "initial_D": rows[0]["D"],
        "initial_Q": rows[0]["Q"],
        "initial_C": rows[0]["C"],
        "initial_legal": rows[0]["legal_push_count"],
        "initial_viable": rows[0]["viable_push_count"],
        "initial_tva_delta": rows[0]["tva_delta"],
        "initial_tva_risk": rows[0]["tva_risk_probability"],
        "initial_posonly_alpha": rows[0]["posonly_alpha"],
        "initial_phase060_alpha": rows[0]["phase060_alpha"],
        "initial_phase060_reason": rows[0]["phase060_reason"],
        "phase060_trigger_steps": phase_steps,
        "posonly_trigger_steps": positive_steps,
        "phase060_trigger_count": len(phase_steps),
        "posonly_trigger_count": len(positive_steps),
        "min_legal_prefix": min(row["legal_push_count"] for row in rows),
        "min_viable_prefix": min(row["viable_push_count"] for row in rows),
        "max_D_prefix": max(row["D"] for row in rows),
        "max_Q_prefix": max(row["Q"] for row in rows),
        "max_C_prefix": max(row["C"] for row in rows),
        "root_q_visit_split_count": len(q_gap_steps),
        "max_q_minus_visit_best": max((row["q_minus_visit_best"] for row in rows), default=0.0),
    }


def main() -> None:
    args = build_parser().parse_args()
    primary = replay_result(args.primary, max_prefix_steps=args.max_prefix_steps)
    fixed_gate10 = replay_result(args.fixed_alpha_gate10, max_prefix_steps=args.max_prefix_steps)
    fixed_fallback = {
        level_idx: replay_result(path, max_prefix_steps=args.max_prefix_steps)[level_idx]
        for level_idx, path in FIXED_FALLBACK_PATHS.items()
    }
    phase_runs = {
        level_idx: replay_result(path, max_prefix_steps=args.max_prefix_steps)[level_idx]
        for level_idx, path in PHASE_LEVEL_PATHS.items()
    }

    labels = {
        0: "negative_high_alpha_ineffective",
        5: "negative_high_alpha_harm",
        6: "negative_high_alpha_harm",
        14: "positive_fixed_alpha_repair",
        45: "positive_fixed_alpha_repair",
        46: "positive_fixed_alpha_repair",
    }
    tracked = sorted(labels)
    comparison: dict[str, Any] = {}
    for level_idx in tracked:
        comparison[str(level_idx)] = {
            "label": labels[level_idx],
            "primary_state_gated": {
                "outcome": {
                    "solved": primary[level_idx]["solved"],
                    "status": primary[level_idx]["status"],
                    "steps_executed": primary[level_idx]["steps_executed"],
                },
                "prefix_summary": prefix_summary(primary[level_idx]),
                "prefix_rows": primary[level_idx]["per_step"],
            },
            "fixed_alpha060": {
                "outcome": {
                    "solved": (fixed_fallback.get(level_idx) or fixed_gate10.get(level_idx))["solved"],
                    "status": (fixed_fallback.get(level_idx) or fixed_gate10.get(level_idx))["status"],
                    "steps_executed": (fixed_fallback.get(level_idx) or fixed_gate10.get(level_idx))[
                        "steps_executed"
                    ],
                },
                "prefix_summary": prefix_summary(fixed_fallback.get(level_idx) or fixed_gate10[level_idx]),
                "prefix_rows": (fixed_fallback.get(level_idx) or fixed_gate10[level_idx])["per_step"],
            },
        }
        if level_idx in phase_runs:
            comparison[str(level_idx)]["phase060_dynamic"] = {
                "outcome": {
                    "solved": phase_runs[level_idx]["solved"],
                    "status": phase_runs[level_idx]["status"],
                    "steps_executed": phase_runs[level_idx]["steps_executed"],
                },
                "prefix_summary": prefix_summary(phase_runs[level_idx]),
                "prefix_rows": phase_runs[level_idx]["per_step"],
            }

    rule_counts = defaultdict(int)
    for level_idx in tracked:
        summary = comparison[str(level_idx)]["primary_state_gated"]["prefix_summary"]
        if summary.get("initial_phase060_alpha", 0.0) >= 0.60:
            rule_counts[f"{labels[level_idx]}:initial_phase060"] += 1
        if summary.get("phase060_trigger_count", 0) > 0:
            rule_counts[f"{labels[level_idx]}:prefix_phase060"] += 1

    payload = {
        "material_passport": {
            "origin": "sokoban_dynamic_tva_alpha_feature_audit",
            "verification_status": "ANALYZED",
            "version_label": "dynamic_tva_alpha_feature_audit_v1",
        },
        "config": {
            "primary": _rel(args.primary),
            "combo": _rel(args.combo),
            "fixed_alpha_gate10": _rel(args.fixed_alpha_gate10),
            "phase_level_paths": {str(k): _rel(v) for k, v in PHASE_LEVEL_PATHS.items()},
            "fixed_fallback_paths": {str(k): _rel(v) for k, v in FIXED_FALLBACK_PATHS.items()},
            "max_prefix_steps": args.max_prefix_steps,
            "phase060_rule": "alpha=0.60 when P>=20 or (P<=10 and C>=0.25); otherwise use posonly state gate",
            "posonly_rule": "alpha=0.25 when risk<=0.30 and V_TVA - V_SD >= 0.25; otherwise alpha=0",
            "labels": labels,
        },
        "rule_trigger_counts": dict(sorted(rule_counts.items())),
        "comparison": comparison,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": _rel(out), "rule_trigger_counts": payload["rule_trigger_counts"]}, indent=2))


if __name__ == "__main__":
    main()
