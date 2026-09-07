#!/usr/bin/env python3
"""Audit Sokoban TVA accuracy against offline truncation-value labels.

The audit treats TVA as a leaf-value estimator for truncated SD-MCTS. It uses
the exact-state diagnostic teacher only offline to label sampled states with a
remaining-solution value. The teacher is not called by online SD-MCTS and this
script does not modify the transition provider, planner feedback, or action
prior.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import stock_value  # noqa: E402
from evaluate_stage3_sdmcts_planning import ExpertAbstractPlannerFeedback  # noqa: E402
from sokoban_tva import load_tva_checkpoint, predict_tva_value, relative_path  # noqa: E402
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import best_first_from_state, provider_level_split, target_from_teacher  # noqa: E402
from validate_sokoban_sd_design import iter_states, stock_proxy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--max-eval-levels", type=int, default=12)
    parser.add_argument("--state-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=8)
    parser.add_argument("--teacher-max-expansions", type=int, default=1200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument(
        "--tva-checkpoints",
        default=(
            "results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_seed0.pt,"
            "results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt"
        ),
        help="Comma-separated TVA checkpoints to audit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/tva_truncation_value_audit_quick_seed0.json",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_csv_paths(raw: str) -> list[Path]:
    paths = [Path(item.strip()) for item in raw.split(",") if item.strip()]
    if not paths:
        raise SystemExit("At least one TVA checkpoint must be supplied.")
    return paths


def _safe_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size <= 1 or np.std(pred) <= 1e-12 or np.std(target) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.size == 0:
        return {
            "samples": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "corr": 0.0,
            "bias": 0.0,
            "sign_accuracy": 0.0,
        }
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    return {
        "samples": float(pred.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "corr": _safe_corr(pred, target),
        "bias": float(np.mean(err)),
        "sign_accuracy": float(np.mean(np.signbit(pred) == np.signbit(target))),
    }


def _grouped_metrics(rows: Sequence[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {
        "all_high_confidence": [
            row for row in rows if row["teacher_confidence"] == "high_confidence"
        ],
        "solved": [row for row in rows if row["teacher_failure_type"] == "solved"],
        "dead_or_no_legal": [
            row
            for row in rows
            if row["teacher_failure_type"] in {"proven_dead_end", "no_legal_actions"}
        ],
        "budget_unsolved": [
            row for row in rows if row["teacher_failure_type"] == "teacher_budget_unsolved"
        ],
        "positive_value": [
            row
            for row in rows
            if row["teacher_confidence"] == "high_confidence" and row["target_value"] > 0.0
        ],
        "nonpositive_value": [
            row
            for row in rows
            if row["teacher_confidence"] == "high_confidence" and row["target_value"] <= 0.0
        ],
    }
    for name, group in groups.items():
        pred = np.asarray([row[pred_key] for row in group], dtype=np.float64)
        target = np.asarray([row["target_value"] for row in group], dtype=np.float64)
        out[name] = _regression_metrics(pred, target)
    return out


def teacher_value(args: argparse.Namespace, model: SokobanGoalDirectedSDModel, state: SokobanState) -> tuple[Any, float]:
    teacher = best_first_from_state(model, state, max_expansions=args.teacher_max_expansions)
    value = target_from_teacher(
        teacher,
        gamma=args.gamma,
        goal_value=args.goal_value,
        push_cost=args.push_cost,
        deadend_value=args.deadend_value,
        unsolved_value=args.unsolved_value,
    )
    return teacher, float(value)


def collect_rows(
    args: argparse.Namespace,
    tva_payloads: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    split = provider_level_split(args.provider_checkpoint)
    allowed_ids = set(split["train_levels"]) | set(split["valid_levels"])
    selected = [level for level in levels if level.level_id in allowed_ids]
    if args.max_eval_levels > 0:
        selected = selected[: args.max_eval_levels]
    if not selected:
        raise SystemExit("No stage-2 split levels selected for TVA truncation audit.")

    rows: list[dict[str, Any]] = []
    teacher_cache: dict[tuple[str, SokobanState], tuple[Any, float]] = {}
    teacher_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    root_state_counts: Counter[str] = Counter()
    expert_feedback = ExpertAbstractPlannerFeedback()
    t0 = perf_counter()

    for level_index, level in enumerate(selected):
        if level.level_id in split["train_levels"]:
            split_name = "train"
        elif level.level_id in split["valid_levels"]:
            split_name = "valid"
        else:
            continue
        model = SokobanGoalDirectedSDModel(level)
        seen: set[SokobanState] = set()
        level_rows = 0
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            if state in seen:
                continue
            seen.add(state)
            stock = stock_proxy(model, state).as_array().astype(np.float32)
            key = (level.level_id, state)
            if key not in teacher_cache:
                teacher_cache[key] = teacher_value(args, model, state)
            teacher, value = teacher_cache[key]
            if model.is_solved(state):
                root_state_type = "solved"
            elif model.is_deadlocked(state):
                root_state_type = "deadlocked"
            elif not model.legal_pushes(state):
                root_state_type = "no_legal_actions"
            else:
                root_state_type = "active"
            confidence = (
                "high_confidence"
                if teacher.failure_type in {"solved", "proven_dead_end", "no_legal_actions"}
                else "budget_unsolved_low_confidence"
            )
            row = {
                "level_index": int(level_index),
                "level_id": level.level_id,
                "split": split_name,
                "state_depth": int(state_depth),
                "root_state_type": root_state_type,
                "target_value": float(value),
                "teacher_confidence": confidence,
                "teacher_solved": bool(teacher.solved),
                "teacher_solution_pushes": (
                    int(teacher.solution_pushes) if teacher.solution_pushes is not None else None
                ),
                "teacher_failure_type": str(teacher.failure_type),
                "stock_value": float(stock_value(stock.astype(np.float64))),
                "expert_value": float(expert_feedback.value(model, state, stock.astype(np.float64))),
            }
            for provider_id, payload in tva_payloads.items():
                row[f"tva:{provider_id}"] = float(predict_tva_value(payload, model, state, stock))
            rows.append(row)
            teacher_counts[str(teacher.failure_type)] += 1
            split_counts[split_name] += 1
            root_state_counts[root_state_type] += 1
            level_rows += 1
        print(
            json.dumps(
                {
                    "level_index": int(level_index),
                    "level_id": level.level_id,
                    "split": split_name,
                    "tva_audit_state_rows": int(level_rows),
                },
                ensure_ascii=False,
            )
        )
    return rows, {
        "level_split": split,
        "selected_levels": [level.level_id for level in selected],
        "split_counts": dict(split_counts),
        "root_state_counts": dict(root_state_counts),
        "teacher_counts": dict(teacher_counts),
        "collection_elapsed_sec": float(perf_counter() - t0),
    }


def evaluate(rows: Sequence[dict[str, Any]], provider_ids: Sequence[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "stock_value": _grouped_metrics(rows, "stock_value"),
        "expert_value": _grouped_metrics(rows, "expert_value"),
    }
    for provider_id in provider_ids:
        metrics[f"tva:{provider_id}"] = _grouped_metrics(rows, f"tva:{provider_id}")
    return metrics


def build_gate(metrics: dict[str, Any], provider_ids: Sequence[str]) -> dict[str, Any]:
    failed: list[str] = []
    stock_mae = metrics["stock_value"]["all_high_confidence"]["mae"]
    expert_mae = metrics["expert_value"]["all_high_confidence"]["mae"]
    best_baseline = min(stock_mae, expert_mae)
    observed: dict[str, Any] = {
        "stock_value_high_conf_mae": float(stock_mae),
        "expert_value_high_conf_mae": float(expert_mae),
        "best_baseline_high_conf_mae": float(best_baseline),
    }
    provider_results = {}
    for provider_id in provider_ids:
        key = f"tva:{provider_id}"
        high = metrics[key]["all_high_confidence"]
        positive = metrics[key]["positive_value"]
        nonpositive = metrics[key]["nonpositive_value"]
        provider_failed = []
        if high["samples"] < 20:
            provider_failed.append("high_confidence_samples_min_20")
        if high["mae"] > best_baseline:
            provider_failed.append("mae_not_better_than_best_baseline")
        if high["corr"] < 0.30:
            provider_failed.append("corr_min_0.30")
        if positive["samples"] >= 3 and positive["bias"] < -1.0:
            provider_failed.append("positive_value_underprediction_bias")
        if nonpositive["samples"] >= 3 and nonpositive["bias"] > 1.0:
            provider_failed.append("nonpositive_value_overprediction_bias")
        provider_results[provider_id] = {
            "failed_checks": provider_failed,
            "high_confidence_samples": float(high["samples"]),
            "high_confidence_mae": float(high["mae"]),
            "high_confidence_corr": float(high["corr"]),
            "high_confidence_sign_accuracy": float(high["sign_accuracy"]),
            "positive_samples": float(positive["samples"]),
            "positive_bias": float(positive["bias"]),
            "nonpositive_samples": float(nonpositive["samples"]),
            "nonpositive_bias": float(nonpositive["bias"]),
        }
        if provider_failed:
            failed.append(f"{provider_id}:{','.join(provider_failed)}")
    observed["providers"] = provider_results
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "thresholds": {
            "high_confidence_samples_min": 20,
            "tva_mae_must_not_exceed_best_stock_or_expert_baseline": True,
            "high_confidence_corr_min": 0.30,
            "positive_underprediction_bias_min": -1.0,
            "nonpositive_overprediction_bias_max": 1.0,
        },
        "observed": observed,
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    checkpoint_paths = parse_csv_paths(args.tva_checkpoints)
    tva_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_contracts: dict[str, Any] = {}
    for checkpoint in checkpoint_paths:
        payload = load_tva_checkpoint(checkpoint)
        provider_id = str(
            payload.get("checkpoint_contract", {}).get(
                "value_provider_id",
                payload.get("summary", {}).get("config", {}).get("value_provider_id", checkpoint.stem),
            )
        )
        if provider_id in tva_payloads:
            provider_id = f"{provider_id}#{len(tva_payloads)}"
        tva_payloads[provider_id] = payload
        checkpoint_contracts[provider_id] = {
            "path": relative_path(checkpoint, ROOT),
            "checkpoint_contract": payload.get("checkpoint_contract", {}),
            "target_stats": payload.get("target_stats", {}),
        }

    rows, collection = collect_rows(args, tva_payloads)
    provider_ids = list(tva_payloads.keys())
    metrics = evaluate(rows, provider_ids)
    gate = build_gate(metrics, provider_ids)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_tva_truncation_value_audit_v1",
        },
        "config": {
            **vars(args),
            "tva_checkpoints": [relative_path(path, ROOT) for path in checkpoint_paths],
        },
        "audit_contract": {
            "purpose": "evaluate TVA as truncated SD-MCTS leaf-value compensation against offline remaining-value labels",
            "teacher_usage": "offline_state_label_only",
            "teacher_used_online": False,
            "best_first_used_online": False,
            "planner_modified": False,
            "transition_provider_modified": False,
            "sd_independent_test_requires_tva": False,
            "sdmcts_closed_loop_without_tva_is_truncation_bias_diagnostic_only": True,
            "high_confidence_labels": ["solved", "proven_dead_end", "no_legal_actions"],
            "budget_unsolved_labels": "reported_but_not_used_as_high_confidence_gate",
        },
        "checkpoint_contracts": checkpoint_contracts,
        "collection": collection,
        "metrics": metrics,
        "tva_truncation_value_gate": gate,
        "sample_rows": rows[:40],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "tva_truncation_value_gate": gate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
