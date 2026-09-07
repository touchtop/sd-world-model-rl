#!/usr/bin/env python3
"""Attribute root-backup value errors to learned provider stock predictions.

This is an offline diagnostic. It reconstructs the root states used by
``audit_sokoban_tva_root_sorting.py``, compares learned-provider next stocks
with exact stock proxies, and estimates each stock's contribution by replacing
one predicted stock at a time with the exact value.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import LearnedStockFlowTransitionProvider  # noqa: E402
from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    ExpertAbstractPlannerFeedback,
    SokobanTvaTerminalValueEvaluator,
)
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from validate_sokoban_sd_design import STOCK_NAMES, stock_proxy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-sorting-audit",
        default="results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json",
    )
    parser.add_argument(
        "--source-result",
        default="",
        help="Override source result. Defaults to the source_result in the root sorting audit config.",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--out",
        default=(
            "results/sokoban_sd_world_model/"
            "provider_stock_value_attribution_teacherq_accuracygate_seed0.json"
        ),
    )
    return parser


def _score(
    model: SokobanGoalDirectedSDModel,
    feedback: ExpertAbstractPlannerFeedback,
    tva: SokobanTvaTerminalValueEvaluator,
    state: SokobanState,
    root_stock: np.ndarray,
    next_state: SokobanState,
    next_stock: np.ndarray,
    *,
    gamma: float,
) -> float:
    reward = feedback.reward(model, state, root_stock, next_state, next_stock)
    leaf = tva.value(model, next_state, next_stock)
    return float(reward + gamma * leaf)


def _root_rows_by_episode(root_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in root_rows:
        out[int(row["episode_index"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda row: int(row["step"]))
    return out


def main() -> None:
    args = build_parser().parse_args()
    root_audit = json.loads(Path(args.root_sorting_audit).read_text(encoding="utf-8"))
    config = root_audit.get("config", {})
    source_result_path = Path(args.source_result or config["source_result"])
    provider_checkpoint = Path(config["provider_checkpoint"])
    tva_checkpoint = Path(config["tva_checkpoint"])
    levels_path = config["levels"]

    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    levels = {
        level.level_id: level
        for level in load_boxoban_levels(levels_path, max_levels=source_result.get("config", {}).get("max_levels", 30))
    }
    provider = LearnedStockFlowTransitionProvider(provider_checkpoint)
    feedback = ExpertAbstractPlannerFeedback()
    tva = SokobanTvaTerminalValueEvaluator(tva_checkpoint)

    rows_by_episode = _root_rows_by_episode(root_audit.get("root_rows", []))
    attribution_rows: list[dict[str, Any]] = []
    stock_stats = {
        name: {
            "count": 0,
            "mean_error_sum": 0.0,
            "abs_error_sum": 0.0,
            "signed_contribution_sum": 0.0,
            "abs_contribution_sum": 0.0,
            "teacher_tier_abs_contribution_sum": 0.0,
            "non_tier_abs_contribution_sum": 0.0,
        }
        for name in STOCK_NAMES
    }

    for episode_index, episode in enumerate(source_result.get("episodes", [])):
        if episode_index not in rows_by_episode:
            continue
        level_id = episode["level_id"]
        model = SokobanGoalDirectedSDModel(levels[level_id])
        state = model.initial_state
        trace_rows = episode["episode"].get("trace_sample", [])
        root_rows = rows_by_episode[episode_index]
        root_row_by_step = {int(row["step"]): row for row in root_rows}

        for trace_row in trace_rows:
            step = int(trace_row.get("step", -1))
            root_row = root_row_by_step.get(step)
            if root_row is None:
                continue
            root_stock = stock_proxy(model, state).as_array().astype(np.float64)
            legal = {action.flat_index: action for action in model.legal_pushes(state)}
            mode_rows = {
                int(row["flat"]): row
                for row in root_row.get("modes", {}).get("teacherq_tva", {}).get("rows", [])
            }

            for flat, policy_row in mode_rows.items():
                action = legal.get(flat)
                if action is None:
                    continue
                next_state = model.step_state(state, action)
                exact_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
                prediction = provider.predict(model, state, root_stock, action)
                provider_stock = np.asarray(prediction.next_stock, dtype=np.float64)
                provider_score = _score(
                    model,
                    feedback,
                    tva,
                    state,
                    root_stock,
                    next_state,
                    provider_stock,
                    gamma=args.gamma,
                )
                exact_score = _score(
                    model,
                    feedback,
                    tva,
                    state,
                    root_stock,
                    next_state,
                    exact_stock,
                    gamma=args.gamma,
                )
                contributions = {}
                for idx, stock_name in enumerate(STOCK_NAMES):
                    hybrid = provider_stock.copy()
                    hybrid[idx] = exact_stock[idx]
                    hybrid_score = _score(
                        model,
                        feedback,
                        tva,
                        state,
                        root_stock,
                        next_state,
                        hybrid,
                        gamma=args.gamma,
                    )
                    contribution = float(hybrid_score - provider_score)
                    error = float(provider_stock[idx] - exact_stock[idx])
                    stats = stock_stats[stock_name]
                    stats["count"] += 1
                    stats["mean_error_sum"] += error
                    stats["abs_error_sum"] += abs(error)
                    stats["signed_contribution_sum"] += contribution
                    stats["abs_contribution_sum"] += abs(contribution)
                    if bool(policy_row.get("in_teacher_best_tier")):
                        stats["teacher_tier_abs_contribution_sum"] += abs(contribution)
                    else:
                        stats["non_tier_abs_contribution_sum"] += abs(contribution)
                    contributions[stock_name] = {
                        "provider_minus_exact": error,
                        "single_stock_exact_replacement_delta_score": contribution,
                    }
                attribution_rows.append(
                    {
                        "episode_index": int(episode_index),
                        "level_id": level_id,
                        "step": int(step),
                        "flat": int(flat),
                        "in_teacher_best_tier": bool(policy_row.get("in_teacher_best_tier")),
                        "teacher_value": float(policy_row["teacher_value"]),
                        "provider_score": float(provider_score),
                        "exact_score": float(exact_score),
                        "exact_minus_provider_score": float(exact_score - provider_score),
                        "provider_stock_l1": float(np.mean(np.abs(provider_stock - exact_stock))),
                        "stock_contributions": contributions,
                    }
                )

            observed_flat = trace_row.get("action", {}).get("flat_index")
            if observed_flat not in legal:
                break
            state = model.step_state(state, legal[int(observed_flat)])

    stock_summary = []
    for name, stats in stock_stats.items():
        count = max(int(stats["count"]), 1)
        stock_summary.append(
            {
                "stock": name,
                "count": int(stats["count"]),
                "mean_provider_minus_exact": float(stats["mean_error_sum"] / count),
                "mean_abs_provider_minus_exact": float(stats["abs_error_sum"] / count),
                "mean_signed_score_contribution": float(stats["signed_contribution_sum"] / count),
                "mean_abs_score_contribution": float(stats["abs_contribution_sum"] / count),
                "teacher_tier_abs_contribution_sum": float(stats["teacher_tier_abs_contribution_sum"]),
                "non_tier_abs_contribution_sum": float(stats["non_tier_abs_contribution_sum"]),
            }
        )
    stock_summary.sort(key=lambda row: row["mean_abs_score_contribution"], reverse=True)

    critical_rows = sorted(
        attribution_rows,
        key=lambda row: abs(float(row["exact_minus_provider_score"])),
        reverse=True,
    )[:20]
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_provider_stock_value_attribution_v1",
        },
        "config": {
            "root_sorting_audit": args.root_sorting_audit,
            "source_result": str(source_result_path),
            "provider_checkpoint": str(provider_checkpoint),
            "tva_checkpoint": str(tva_checkpoint),
            "levels": levels_path,
            "gamma": float(args.gamma),
        },
        "audit_contract": {
            "offline_attribution_only": True,
            "teacher_called": False,
            "mcts_rerun": False,
            "transition_provider_modified": False,
            "planner_modified": False,
            "single_stock_exact_replacement_is_diagnostic_not_online": True,
        },
        "observed": {
            "candidate_rows": int(len(attribution_rows)),
            "stock_summary": stock_summary,
            "largest_score_error_rows": critical_rows,
        },
        "rows": attribution_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "candidate_rows": len(attribution_rows),
                "top_stocks": stock_summary[:6],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
