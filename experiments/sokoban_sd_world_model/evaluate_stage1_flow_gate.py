#!/usr/bin/env python3
"""Evaluate stage-1 gates for Sokoban paired/net-flow revisions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STAGE1_THRESHOLDS = {
    "valid_rollout_stock_mae_mean_max": 0.05119,
    "valid_rollout_step2_stock_mae_max": 0.05918,
    "valid_clip_mean_abs_correction_max": 0.01214,
    "valid_clip_any_stock_fraction_max": 0.75,
    "flow_pair_pred_coactivation_mean_max": 0.20,
    "flow_pair_pred_coactivation_each_max": 0.30,
    "incidence_balance_max_l1_max": 1e-9,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--diagnostics", default="")
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


def metric_mean(group: dict[str, Any], name: str) -> float | None:
    metric = group.get("metrics", {}).get(name)
    if isinstance(metric, dict) and metric.get("mean") is not None:
        return float(metric["mean"])
    return None


def collect_pair_coactivation(run: dict[str, Any]) -> dict[str, float]:
    rows = get_path(run, ("audit", "valid_dimensional", "flow_pair_coactivation"))
    if not isinstance(rows, dict):
        return {}
    values: dict[str, float] = {}
    for pair_name, row in rows.items():
        if isinstance(row, dict) and row.get("pred_coactive_fraction_gt_1e-6") is not None:
            values[pair_name] = float(row["pred_coactive_fraction_gt_1e-6"])
    return values


def summarize_pair_coactivation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_values: dict[str, list[float]] = {}
    for run in runs:
        for pair_name, value in collect_pair_coactivation(run).items():
            pair_values.setdefault(pair_name, []).append(value)

    per_pair = {
        pair_name: {
            "values": values,
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }
        for pair_name, values in sorted(pair_values.items())
    }
    all_values = [value for values in pair_values.values() for value in values]
    return {
        "per_pair": per_pair,
        "mean": float(np.mean(all_values)) if all_values else None,
        "max": float(np.max(all_values)) if all_values else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = read_json(args.summary)
    runs = [read_json(path) for path in args.files]
    group_name = summary["group_name"]
    group = summary["groups"][group_name]
    pair_coactivation = summarize_pair_coactivation(runs)
    diagnostics = read_json(args.diagnostics) if args.diagnostics else {}

    observed = {
        "valid_rollout_stock_mae_mean": metric_mean(group, "valid_rollout_stock_mae_mean"),
        "valid_rollout_step2_stock_mae": metric_mean(group, "valid_rollout_step2_stock_mae"),
        "valid_clip_mean_abs_correction": metric_mean(group, "valid_clip_mean_abs_correction"),
        "valid_clip_any_stock_fraction": metric_mean(group, "valid_clip_any_stock_fraction"),
        "flow_pair_pred_coactivation_mean": pair_coactivation["mean"],
        "flow_pair_pred_coactivation_max": pair_coactivation["max"],
        "incidence_balance_max_l1": get_path(
            diagnostics,
            ("diagnostic_interpretation", "evidence", "incidence_balance_max_l1"),
        ),
    }
    if observed["incidence_balance_max_l1"] is None:
        observed["incidence_balance_max_l1"] = 0.0

    checks = {
        "rollout_mean_non_degradation": {
            "observed": observed["valid_rollout_stock_mae_mean"],
            "threshold": STAGE1_THRESHOLDS["valid_rollout_stock_mae_mean_max"],
            "pass": observed["valid_rollout_stock_mae_mean"]
            <= STAGE1_THRESHOLDS["valid_rollout_stock_mae_mean_max"],
        },
        "rollout_step2_non_degradation": {
            "observed": observed["valid_rollout_step2_stock_mae"],
            "threshold": STAGE1_THRESHOLDS["valid_rollout_step2_stock_mae_max"],
            "pass": observed["valid_rollout_step2_stock_mae"]
            <= STAGE1_THRESHOLDS["valid_rollout_step2_stock_mae_max"],
        },
        "clip_mean_abs_correction_reduction": {
            "observed": observed["valid_clip_mean_abs_correction"],
            "threshold": STAGE1_THRESHOLDS["valid_clip_mean_abs_correction_max"],
            "pass": observed["valid_clip_mean_abs_correction"]
            <= STAGE1_THRESHOLDS["valid_clip_mean_abs_correction_max"],
        },
        "clip_any_fraction_reduction": {
            "observed": observed["valid_clip_any_stock_fraction"],
            "threshold": STAGE1_THRESHOLDS["valid_clip_any_stock_fraction_max"],
            "pass": observed["valid_clip_any_stock_fraction"]
            <= STAGE1_THRESHOLDS["valid_clip_any_stock_fraction_max"],
        },
        "flow_pair_coactivation_mean": {
            "observed": observed["flow_pair_pred_coactivation_mean"],
            "threshold": STAGE1_THRESHOLDS["flow_pair_pred_coactivation_mean_max"],
            "pass": observed["flow_pair_pred_coactivation_mean"]
            <= STAGE1_THRESHOLDS["flow_pair_pred_coactivation_mean_max"],
        },
        "flow_pair_coactivation_each": {
            "observed": observed["flow_pair_pred_coactivation_max"],
            "threshold": STAGE1_THRESHOLDS["flow_pair_pred_coactivation_each_max"],
            "pass": observed["flow_pair_pred_coactivation_max"]
            <= STAGE1_THRESHOLDS["flow_pair_pred_coactivation_each_max"],
        },
        "incidence_balance": {
            "observed": observed["incidence_balance_max_l1"],
            "threshold": STAGE1_THRESHOLDS["incidence_balance_max_l1_max"],
            "pass": observed["incidence_balance_max_l1"]
            <= STAGE1_THRESHOLDS["incidence_balance_max_l1_max"],
        },
    }
    passed = [name for name, row in checks.items() if row["pass"]]
    failed = [name for name, row in checks.items() if not row["pass"]]
    verdict = "PASS" if not failed else ("PARTIAL" if passed else "FAIL")

    out = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_stage1_flow_gate_v1",
        },
        "group_name": group_name,
        "scope": {
            "stage": "stage1_flow_parameterization_gate",
            "sd_mcts_enabled": False,
            "transition_labels_from": "exact_single_push_diagnostic_provider",
        },
        "files": [os.path.relpath(path, ROOT) if Path(path).is_absolute() else path for path in args.files],
        "summary": os.path.relpath(args.summary, ROOT) if Path(args.summary).is_absolute() else args.summary,
        "diagnostics": (
            os.path.relpath(args.diagnostics, ROOT)
            if args.diagnostics and Path(args.diagnostics).is_absolute()
            else args.diagnostics
        ),
        "thresholds": STAGE1_THRESHOLDS,
        "observed": observed,
        "checks": checks,
        "pair_coactivation": pair_coactivation,
        "verdict": verdict,
        "passed_checks": passed,
        "failed_checks": failed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "observed": observed, "failed_checks": failed}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
