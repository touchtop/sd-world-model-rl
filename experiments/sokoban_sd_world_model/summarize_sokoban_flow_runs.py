#!/usr/bin/env python3
"""Summarize Sokoban SD learned-flow smoke runs and compare reruns.

The script is intentionally small: it only reads JSON artifacts produced by
``train_sokoban_flow_smoke.py`` and writes an auditable aggregate table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRIMARY_METRICS = {
    "valid_stock_mae": ("metrics", "valid", "stock_mae"),
    "valid_flow_mae": ("metrics", "valid", "flow_mae"),
    "valid_zero_flow_stock_mae": ("metrics", "valid", "zero_flow_stock_mae"),
    "valid_rollout_stock_mae_mean": ("metrics", "valid_rollout", "stock_mae_mean"),
    "valid_rollout_zero_flow_stock_mae_mean": (
        "metrics",
        "valid_rollout",
        "zero_flow_stock_mae_mean",
    ),
    "valid_rollout_improvement_over_zero_flow_mean": (
        "metrics",
        "valid_rollout",
        "stock_mae_improvement_over_zero_flow_mean",
    ),
    "valid_rollout_step1_stock_mae": (
        "metrics",
        "valid_rollout",
        "stock_mae_by_step",
        "step_1",
    ),
    "valid_rollout_step2_stock_mae": (
        "metrics",
        "valid_rollout",
        "stock_mae_by_step",
        "step_2",
    ),
    "valid_rollout_step1_flow_mae": (
        "metrics",
        "valid_rollout",
        "flow_mae_by_step",
        "step_1",
    ),
    "valid_rollout_step2_flow_mae": (
        "metrics",
        "valid_rollout",
        "flow_mae_by_step",
        "step_2",
    ),
}

AUDIT_METRICS = {
    "valid_clip_any_stock_fraction": (
        "audit",
        "valid_dimensional",
        "clip_audit",
        "any_stock_clipped_fraction",
    ),
    "valid_clip_mean_abs_correction": (
        "audit",
        "valid_dimensional",
        "clip_audit",
        "mean_abs_clip_correction",
    ),
    "valid_alias_stock_max_next_delta_variance": (
        "audit",
        "valid_aliasing",
        "stock_bucket_aliasing",
        "max_next_delta_variance",
    ),
    "valid_alias_stock_action_max_next_delta_variance": (
        "audit",
        "valid_aliasing",
        "stock_action_bucket_aliasing",
        "max_next_delta_variance",
    ),
}


def collect_pair_coactivation(row: dict[str, Any]) -> dict[str, float]:
    pairs = get_path(row, ("audit", "valid_dimensional", "flow_pair_coactivation"))
    if not isinstance(pairs, dict):
        return {}
    return {
        pair_name: float(pair_row["pred_coactive_fraction_gt_1e-6"])
        for pair_name, pair_row in pairs.items()
        if isinstance(pair_row, dict) and pair_row.get("pred_coactive_fraction_gt_1e-6") is not None
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--compare-summary", default="")
    parser.add_argument("--compare-group", default="")
    parser.add_argument(
        "--repro-threshold",
        type=float,
        default=0.05,
        help="Relative threshold for stochastic reproducibility verdicts.",
    )
    parser.add_argument("--out", required=True)
    return parser


def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_path(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def relpath(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return os.path.relpath(path_obj, ROOT)
    return path


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
    }


def summarize_runs(files: list[str]) -> dict[str, Any]:
    rows = [read_json(path) for path in files]
    metrics: dict[str, Any] = {}
    for name, path in {**PRIMARY_METRICS, **AUDIT_METRICS}.items():
        values = [get_path(row, path) for row in rows]
        numeric_values = [float(value) for value in values if value is not None]
        if numeric_values:
            metrics[name] = summarize_values(numeric_values)

    per_stock_step2: dict[str, Any] = {}
    for row in rows:
        step2 = get_path(row, ("metrics", "valid_rollout", "per_stock_mae_by_step", "step_2"))
        if isinstance(step2, dict):
            for stock_name, value in step2.items():
                per_stock_step2.setdefault(f"valid_rollout_step2_{stock_name}_mae", []).append(float(value))
    per_stock_step2 = {
        name: summarize_values(values)
        for name, values in per_stock_step2.items()
    }

    per_stock_onestep: dict[str, Any] = {}
    for row in rows:
        per_stock = get_path(row, ("audit", "valid_dimensional", "per_stock_mae"))
        if isinstance(per_stock, dict):
            for stock_name, value in per_stock.items():
                per_stock_onestep.setdefault(f"valid_{stock_name}_mae", []).append(float(value))
    per_stock_onestep = {
        name: summarize_values(values)
        for name, values in per_stock_onestep.items()
    }

    pair_coactivation_values: dict[str, list[float]] = {}
    for row in rows:
        for pair_name, value in collect_pair_coactivation(row).items():
            pair_coactivation_values.setdefault(pair_name, []).append(value)
    pair_coactivation = {
        name: summarize_values(values)
        for name, values in pair_coactivation_values.items()
    }
    all_pair_values = [
        value
        for values in pair_coactivation_values.values()
        for value in values
    ]
    if all_pair_values:
        metrics["valid_flow_pair_pred_coactivation_mean"] = summarize_values(all_pair_values)
        metrics["valid_flow_pair_pred_coactivation_max"] = {
            "values": all_pair_values,
            "mean": float(np.max(all_pair_values)),
            "std": 0.0,
        }

    return {
        "files": [relpath(path) for path in files],
        "seeds": [get_path(row, ("config", "seed")) for row in rows],
        "config_digest": {
            "model_kind": get_path(rows[0], ("config", "model_kind")),
            "flow_parameterization": get_path(rows[0], ("config", "flow_parameterization")),
            "feature_mode": get_path(rows[0], ("config", "feature_mode")),
            "stock_loss_preset": get_path(rows[0], ("config", "stock_loss_preset")),
            "rollout_loss_weight": get_path(rows[0], ("config", "rollout_loss_weight")),
            "rollout_final_step_weight": get_path(rows[0], ("config", "rollout_final_step_weight")),
            "rollout_final_stock_weights": get_path(rows[0], ("config", "rollout_final_stock_weights")),
            "rollout_depth": get_path(rows[0], ("config", "rollout_depth")),
            "split_mode": get_path(rows[0], ("config", "split_mode")),
            "baseline_id": get_path(rows[0], ("baseline_contract", "baseline_id")),
            "is_frozen_minimal_baseline": get_path(
                rows[0],
                ("baseline_contract", "is_frozen_minimal_baseline"),
            ),
        },
        "metrics": metrics,
        "per_stock_step2": per_stock_step2,
        "per_stock_one_step": per_stock_onestep,
        "flow_pair_pred_coactivation": pair_coactivation,
    }


def load_compare_group(path: str, group_name: str) -> dict[str, Any]:
    if not path:
        return {}
    row = read_json(path)
    if group_name:
        return row.get("groups", {}).get(group_name, {})
    return row


def metric_mean(group: dict[str, Any], metric_name: str) -> float | None:
    metric = group.get("metrics", {}).get(metric_name)
    if isinstance(metric, dict) and metric.get("mean") is not None:
        return float(metric["mean"])
    return None


def compare_groups(current: dict[str, Any], reference: dict[str, Any], threshold: float) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    mismatches: list[str] = []
    for name in PRIMARY_METRICS:
        cur = metric_mean(current, name)
        ref = metric_mean(reference, name)
        if cur is None or ref is None:
            continue
        denom = max(abs(cur), abs(ref), 1e-12)
        rel_delta = (cur - ref) / denom
        exact_match = math.isclose(cur, ref, rel_tol=0.0, abs_tol=0.0)
        within_threshold = abs(rel_delta) <= threshold
        if not within_threshold:
            mismatches.append(name)
        comparisons[name] = {
            "current_mean": cur,
            "reference_mean": ref,
            "delta": cur - ref,
            "relative_delta": rel_delta,
            "exact_match": exact_match,
            "within_threshold": within_threshold,
        }
    return {
        "threshold": threshold,
        "metrics": comparisons,
        "verdict": "REPRODUCIBLE" if not mismatches else "PARTIALLY_REPRODUCIBLE",
        "mismatched_metrics": mismatches,
    }


def main() -> None:
    args = build_parser().parse_args()
    current_group = summarize_runs(args.files)
    compare_group = load_compare_group(args.compare_summary, args.compare_group)
    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "VERIFIED" if compare_group else "ANALYZED",
            "version_label": "sokoban_flow_baseline_rerun_summary_v1",
        },
        "group_name": args.group_name,
        "scope": {
            "experiment_type": "frozen_minimal_baseline_retest",
            "sd_mcts_enabled": False,
            "transition_labels_from": "exact_single_push_diagnostic_provider",
            "learned_component": "nonnegative_flow_model",
        },
        "groups": {
            args.group_name: current_group,
        },
        "reproducibility_comparison": (
            compare_groups(current_group, compare_group, args.repro_threshold)
            if compare_group
            else None
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["groups"][args.group_name]["metrics"], indent=2, ensure_ascii=False))
    if summary["reproducibility_comparison"]:
        print(json.dumps(summary["reproducibility_comparison"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
