#!/usr/bin/env python3
"""Scan reward/TVA leaf scales on existing Sokoban root-sorting audit rows.

This script is deliberately offline. It does not run MCTS, does not call the
teacher, and does not modify the transition provider. It reads the per-root
decomposition produced by ``audit_sokoban_tva_root_sorting.py`` and asks whether
a simple one-step backup scale can select teacher-supported root children.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-sorting-audit",
        default="results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json",
    )
    parser.add_argument(
        "--reward-scales",
        default="0.0,0.1,0.25,0.5,1.0,1.5,2.0",
        help="Comma-separated multipliers for expert feedback reward.",
    )
    parser.add_argument(
        "--leaf-scales",
        default="0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated multipliers for TVA leaf value.",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/tva_backup_scale_audit_teacherq_accuracygate_seed0.json",
    )
    return parser


def _parse_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("Scale list cannot be empty.")
    return values


def _rows_for_mode(root_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = root_row.get("modes", {}).get("teacherq_tva", {}).get("rows", [])
    return [row for row in rows if row.get("teacher_value") is not None]


def _score_row(
    row: dict[str, Any],
    *,
    reward_key: str,
    leaf_key: str,
    reward_scale: float,
    leaf_scale: float,
    gamma: float,
) -> float:
    reward = float(row[reward_key])
    leaf = float(row[leaf_key])
    return float(reward_scale * reward + gamma * leaf_scale * leaf)


def _evaluate_grid(
    root_rows: list[dict[str, Any]],
    *,
    reward_key: str,
    leaf_key: str,
    reward_scales: list[float],
    leaf_scales: list[float],
    gamma: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for reward_scale in reward_scales:
        for leaf_scale in leaf_scales:
            selected_tier = 0
            critical_miss = 0
            selected_teacher_values = []
            teacher_best_values = []
            selected_gap_values = []
            root_count = 0
            rows_out = []
            for root_row in root_rows:
                rows = [
                    row
                    for row in _rows_for_mode(root_row)
                    if row.get(reward_key) is not None and row.get(leaf_key) is not None
                ]
                if not rows:
                    continue
                root_count += 1
                scored = [
                    (
                        _score_row(
                            row,
                            reward_key=reward_key,
                            leaf_key=leaf_key,
                            reward_scale=reward_scale,
                            leaf_scale=leaf_scale,
                            gamma=gamma,
                        ),
                        row,
                    )
                    for row in rows
                ]
                score, selected = max(scored, key=lambda item: (item[0], -int(item[1]["flat"])))
                best_teacher = max(float(row["teacher_value"]) for row in rows)
                selected_teacher = float(selected["teacher_value"])
                gap = float(best_teacher - selected_teacher)
                is_tier = bool(selected.get("in_teacher_best_tier"))
                selected_tier += int(is_tier)
                critical_miss += int(gap > 0.25)
                selected_teacher_values.append(selected_teacher)
                teacher_best_values.append(best_teacher)
                selected_gap_values.append(gap)
                rows_out.append(
                    {
                        "episode_index": int(root_row["episode_index"]),
                        "step": int(root_row["step"]),
                        "selected_flat": int(selected["flat"]),
                        "selected_score": float(score),
                        "selected_teacher_value": selected_teacher,
                        "teacher_best_value": best_teacher,
                        "selected_teacher_gap": gap,
                        "selected_in_teacher_best_tier": is_tier,
                    }
                )
            out.append(
                {
                    "reward_scale": float(reward_scale),
                    "leaf_scale": float(leaf_scale),
                    "roots": int(root_count),
                    "q_best_in_teacher_best_tier": int(selected_tier),
                    "critical_miss": int(critical_miss),
                    "mean_selected_teacher_value": (
                        float(np.mean(selected_teacher_values)) if selected_teacher_values else None
                    ),
                    "mean_teacher_best_value": (
                        float(np.mean(teacher_best_values)) if teacher_best_values else None
                    ),
                    "mean_selected_teacher_gap": (
                        float(np.mean(selected_gap_values)) if selected_gap_values else None
                    ),
                    "rows": rows_out,
                }
            )
    out.sort(
        key=lambda row: (
            row["q_best_in_teacher_best_tier"],
            -row["critical_miss"],
            -(row["mean_selected_teacher_gap"] or 0.0),
            row["mean_selected_teacher_value"] or -1e9,
        ),
        reverse=True,
    )
    return out


def main() -> None:
    args = build_parser().parse_args()
    data = json.loads(Path(args.root_sorting_audit).read_text(encoding="utf-8"))
    root_rows = data.get("root_rows", [])
    reward_scales = _parse_floats(args.reward_scales)
    leaf_scales = _parse_floats(args.leaf_scales)
    provider_grid = _evaluate_grid(
        root_rows,
        reward_key="expert_reward_provider_stock",
        leaf_key="tva_provider_leaf_value",
        reward_scales=reward_scales,
        leaf_scales=leaf_scales,
        gamma=args.gamma,
    )
    exact_grid = _evaluate_grid(
        root_rows,
        reward_key="expert_reward_exact_stock",
        leaf_key="tva_exact_leaf_value",
        reward_scales=reward_scales,
        leaf_scales=leaf_scales,
        gamma=args.gamma,
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_tva_backup_scale_audit_v1",
        },
        "config": {
            "root_sorting_audit": args.root_sorting_audit,
            "reward_scales": reward_scales,
            "leaf_scales": leaf_scales,
            "gamma": float(args.gamma),
        },
        "audit_contract": {
            "offline_scale_scan_only": True,
            "teacher_called": False,
            "mcts_rerun": False,
            "transition_provider_modified": False,
            "planner_modified": False,
        },
        "observed": {
            "root_count": int(len(root_rows)),
            "provider_stock_best": provider_grid[0] if provider_grid else None,
            "exact_stock_best": exact_grid[0] if exact_grid else None,
        },
        "provider_stock_grid": provider_grid,
        "exact_stock_grid": exact_grid,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "observed": summary["observed"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
