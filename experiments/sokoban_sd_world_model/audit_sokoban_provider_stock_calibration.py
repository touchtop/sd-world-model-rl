#!/usr/bin/env python3
"""Audit a train-split affine calibration for provider next-stock predictions.

The fitted calibrator is diagnostic only. It is trained on the stage-2 provider
train-level split and evaluated on the fixed root states from the TVA root
sorting audit. It does not modify online SD-MCTS or the saved provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from train_sokoban_tva import provider_level_split  # noqa: E402
from validate_sokoban_sd_design import STOCK_NAMES, iter_states, stock_proxy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-sorting-audit",
        default="results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json",
    )
    parser.add_argument("--max-train-levels", type=int, default=16)
    parser.add_argument("--train-state-depth", type=int, default=2)
    parser.add_argument("--max-train-states-per-level", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/provider_stock_calibration_audit_seed0.json",
    )
    return parser


def _fit_affine(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    params = []
    calibrated = np.zeros_like(pred)
    for idx, stock_name in enumerate(STOCK_NAMES):
        x = pred[:, idx]
        y = target[:, idx]
        if float(np.std(x)) < 1e-9:
            a, b = 1.0, 0.0
        else:
            design = np.column_stack([x, np.ones_like(x)])
            a, b = np.linalg.lstsq(design, y, rcond=None)[0].tolist()
        z = np.clip(a * x + b, 0.0, 1.0)
        calibrated[:, idx] = z
        params.append(
            {
                "stock": stock_name,
                "scale": float(a),
                "bias": float(b),
                "train_mae_before": float(np.mean(np.abs(x - y))),
                "train_mae_after": float(np.mean(np.abs(z - y))),
                "train_bias_before": float(np.mean(x - y)),
                "train_bias_after": float(np.mean(z - y)),
            }
        )
    return {"params": params, "calibrated": calibrated}


def _apply_affine(stock: np.ndarray, params: list[dict[str, Any]]) -> np.ndarray:
    out = np.asarray(stock, dtype=np.float64).copy()
    for idx, param in enumerate(params):
        out[idx] = float(param["scale"]) * out[idx] + float(param["bias"])
    return np.clip(out, 0.0, 1.0)


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


def _collect_train_pairs(
    levels: list[Any],
    provider: LearnedStockFlowTransitionProvider,
    *,
    max_train_levels: int,
    state_depth: int,
    max_states_per_level: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred_rows = []
    target_rows = []
    transition_counts = Counter()
    for level in levels[:max_train_levels]:
        model = SokobanGoalDirectedSDModel(level)
        for state, _depth in iter_states(model, max_depth=state_depth, max_states=max_states_per_level):
            stock = stock_proxy(model, state).as_array().astype(np.float64)
            for action in model.legal_pushes(state):
                next_state = model.step_state(state, action)
                exact = stock_proxy(model, next_state).as_array().astype(np.float64)
                pred = provider.predict(model, state, stock, action).next_stock
                pred_rows.append(np.asarray(pred, dtype=np.float64))
                target_rows.append(exact)
                transition_counts[level.level_id] += 1
    if not pred_rows:
        raise SystemExit("No train transitions collected for provider calibration audit.")
    return (
        np.vstack(pred_rows),
        np.vstack(target_rows),
        {
            "train_levels_used": int(min(len(levels), max_train_levels)),
            "train_transition_rows": int(len(pred_rows)),
            "transition_counts_by_level": dict(transition_counts),
        },
    )


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
    config = root_audit["config"]
    source_result = json.loads(Path(config["source_result"]).read_text(encoding="utf-8"))
    provider_checkpoint = Path(config["provider_checkpoint"])
    levels_path = config["levels"]
    split = provider_level_split(provider_checkpoint)
    all_levels = load_boxoban_levels(levels_path, max_levels=source_result.get("config", {}).get("max_levels", 30))
    train_level_ids = set(split["train_levels"])
    train_levels = [level for level in all_levels if level.level_id in train_level_ids]
    provider = LearnedStockFlowTransitionProvider(provider_checkpoint)
    pred_train, exact_train, collection = _collect_train_pairs(
        train_levels,
        provider,
        max_train_levels=args.max_train_levels,
        state_depth=args.train_state_depth,
        max_states_per_level=args.max_train_states_per_level,
    )
    fit = _fit_affine(pred_train, exact_train)
    params = fit["params"]

    levels = {level.level_id: level for level in all_levels}
    feedback = ExpertAbstractPlannerFeedback()
    tva = SokobanTvaTerminalValueEvaluator(Path(config["tva_checkpoint"]))
    rows_by_episode = _root_rows_by_episode(root_audit["root_rows"])

    eval_rows = []
    top_counts = {
        "raw_provider": 0,
        "calibrated_provider": 0,
        "exact_stock": 0,
    }
    critical_counts = {
        "raw_provider": 0,
        "calibrated_provider": 0,
        "exact_stock": 0,
    }
    root_count = 0
    for episode_index, episode in enumerate(source_result.get("episodes", [])):
        if episode_index not in rows_by_episode:
            continue
        model = SokobanGoalDirectedSDModel(levels[episode["level_id"]])
        state = model.initial_state
        root_row_by_step = {int(row["step"]): row for row in rows_by_episode[episode_index]}
        for trace_row in episode["episode"].get("trace_sample", []):
            step = int(trace_row.get("step", -1))
            root_row = root_row_by_step.get(step)
            if root_row is None:
                continue
            root_count += 1
            root_stock = stock_proxy(model, state).as_array().astype(np.float64)
            legal = {action.flat_index: action for action in model.legal_pushes(state)}
            mode_rows = root_row.get("modes", {}).get("teacherq_tva", {}).get("rows", [])
            candidate_rows = []
            for row in mode_rows:
                flat = int(row["flat"])
                action = legal.get(flat)
                if action is None:
                    continue
                next_state = model.step_state(state, action)
                exact_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
                raw_stock = provider.predict(model, state, root_stock, action).next_stock
                calibrated_stock = _apply_affine(raw_stock, params)
                scores = {
                    "raw_provider": _score(
                        model,
                        feedback,
                        tva,
                        state,
                        root_stock,
                        next_state,
                        raw_stock,
                        gamma=args.gamma,
                    ),
                    "calibrated_provider": _score(
                        model,
                        feedback,
                        tva,
                        state,
                        root_stock,
                        next_state,
                        calibrated_stock,
                        gamma=args.gamma,
                    ),
                    "exact_stock": _score(
                        model,
                        feedback,
                        tva,
                        state,
                        root_stock,
                        next_state,
                        exact_stock,
                        gamma=args.gamma,
                    ),
                }
                candidate_rows.append(
                    {
                        "flat": flat,
                        "teacher_value": float(row["teacher_value"]),
                        "in_teacher_best_tier": bool(row["in_teacher_best_tier"]),
                        "scores": scores,
                        "raw_stock_l1": float(np.mean(np.abs(raw_stock - exact_stock))),
                        "calibrated_stock_l1": float(
                            np.mean(np.abs(calibrated_stock - exact_stock))
                        ),
                    }
                )
            best_teacher_value = max(row["teacher_value"] for row in candidate_rows)
            root_out = {
                "episode_index": int(episode_index),
                "level_id": episode["level_id"],
                "step": int(step),
                "teacher_best_value": float(best_teacher_value),
                "candidates": candidate_rows,
                "selection": {},
            }
            for mode in top_counts:
                selected = max(
                    candidate_rows,
                    key=lambda row: (row["scores"][mode], -int(row["flat"])),
                )
                gap = float(best_teacher_value - selected["teacher_value"])
                top_counts[mode] += int(bool(selected["in_teacher_best_tier"]))
                critical_counts[mode] += int(gap > 0.25)
                root_out["selection"][mode] = {
                    "selected_flat": int(selected["flat"]),
                    "selected_score": float(selected["scores"][mode]),
                    "selected_teacher_value": float(selected["teacher_value"]),
                    "selected_in_teacher_best_tier": bool(selected["in_teacher_best_tier"]),
                    "selected_teacher_gap": gap,
                }
            eval_rows.append(root_out)

            observed_flat = trace_row.get("action", {}).get("flat_index")
            if observed_flat not in legal:
                break
            state = model.step_state(state, legal[int(observed_flat)])

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_provider_stock_calibration_audit_v1",
        },
        "config": {
            "root_sorting_audit": args.root_sorting_audit,
            "provider_checkpoint": str(provider_checkpoint),
            "levels": levels_path,
            "max_train_levels": int(args.max_train_levels),
            "train_state_depth": int(args.train_state_depth),
            "max_train_states_per_level": int(args.max_train_states_per_level),
            "gamma": float(args.gamma),
        },
        "audit_contract": {
            "train_split_only_calibrator_fit": True,
            "teacher_called": False,
            "mcts_rerun": False,
            "transition_provider_modified": False,
            "planner_modified": False,
            "calibrator_online_enabled": False,
        },
        "calibration_collection": collection,
        "calibration_params": params,
        "observed": {
            "root_count": int(root_count),
            "teacher_best_tier_hits": top_counts,
            "critical_misses": critical_counts,
        },
        "rows": eval_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "observed": summary["observed"]}, indent=2))


if __name__ == "__main__":
    main()
