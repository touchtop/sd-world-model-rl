#!/usr/bin/env python3
"""Audit train-split action target signal for Sokoban SD-MCTS priors.

This diagnostic checks whether stage-2 train levels contain enough offline
action-Q signal to train a learned planner prior. It does not modify SD-MCTS,
does not train a transition provider, and does not use held-out valid levels as
training labels. The exact-state teacher is used only offline on train-split
successor states.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import stock_reward, stock_value  # noqa: E402
from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    ExpertAbstractPlannerFeedback,
    _goal_room_conflict_score_for_action,
    make_planner_action_priors,
)
from sokoban_learned_action_prior import (  # noqa: E402
    ACTION_PRIOR_FEATURE_NAMES,
    action_prior_feature_vector,
)
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
    parser.add_argument("--max-train-levels", type=int, default=12)
    parser.add_argument("--internal-holdout-fraction", type=float, default=0.25)
    parser.add_argument("--state-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=12)
    parser.add_argument("--max-action-states-per-level", type=int, default=5)
    parser.add_argument("--teacher-max-expansions", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/train_action_target_signal_quick_seed0.json",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    penalty = np.eye(xb.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + penalty, xb.T @ train_y)


def _predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return xb @ weights


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.size == 0:
        return {}
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    return {
        "samples": float(pred.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "corr": (
            float(np.corrcoef(pred, target)[0, 1])
            if pred.size > 1 and np.std(pred) > 1e-12 and np.std(target) > 1e-12
            else 0.0
        ),
    }


def _rank_metrics(rows: Sequence[dict[str, Any]], score_key: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["state_key"]].append(row)
    hits = []
    regrets = []
    random_hits = []
    state_count = 0
    for state_rows in groups.values():
        if len(state_rows) <= 1:
            continue
        state_count += 1
        teacher_best = max(state_rows, key=lambda row: (row["teacher_q"], -row["flat_index"]))
        pred_best = max(state_rows, key=lambda row: (row[score_key], -row["flat_index"]))
        hits.append(pred_best["flat_index"] == teacher_best["flat_index"])
        regrets.append(float(teacher_best["teacher_q"] - pred_best["teacher_q"]))
        random_hits.append(1.0 / len(state_rows))
    return {
        "states": int(state_count),
        "top1_agreement": float(np.mean(hits)) if hits else 0.0,
        "mean_teacher_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_random_top1": float(np.mean(random_hits)) if random_hits else 0.0,
    }


def _target_from_teacher(args: argparse.Namespace, teacher) -> float:
    return float(
        target_from_teacher(
            teacher,
            gamma=args.gamma,
            goal_value=args.goal_value,
            push_cost=args.push_cost,
            deadend_value=args.deadend_value,
            unsolved_value=args.unsolved_value,
        )
    )


def _target2_scores(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    stock: np.ndarray,
) -> dict[int, float]:
    feedback = ExpertAbstractPlannerFeedback()
    priors = make_planner_action_priors(
        model,
        state,
        stock,
        feedback,
        action_abstraction_mode="target2_prior",
        max_option_length=2,
        max_options_per_node=4,
        option_prior_max_multiplier=2.0,
    )
    return {
        int(action.first.flat_index): float(action.structural_score)
        for action in priors
        if action.source == "primitive_push_with_target2_prior"
    }


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    split = provider_level_split(args.provider_checkpoint)
    train_ids = set(split["train_levels"])
    selected = [level for level in levels if level.level_id in train_ids]
    if args.max_train_levels > 0:
        selected = selected[: args.max_train_levels]
    if not selected:
        raise SystemExit("No stage-2 train levels selected for action-target audit.")

    holdout_count = (
        max(1, int(round(len(selected) * args.internal_holdout_fraction)))
        if len(selected) > 3
        else 1
    )
    holdout_ids = {level.level_id for level in selected[-holdout_count:]}
    rows: list[dict[str, Any]] = []
    teacher_cache: dict[tuple[str, SokobanState], Any] = {}
    teacher_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    t0 = perf_counter()

    for level in selected:
        split_name = "valid" if level.level_id in holdout_ids else "train"
        model = SokobanGoalDirectedSDModel(level)
        action_states = 0
        state_index = 0
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            legal = tuple(model.legal_pushes(state))
            if len(legal) <= 1:
                continue
            if action_states >= args.max_action_states_per_level:
                break
            if model.is_solved(state) or model.is_deadlocked(state):
                continue
            stock = stock_proxy(model, state).as_array().astype(np.float32)
            target2_by_flat = _target2_scores(model, state, stock)
            action_rows: list[dict[str, Any]] = []
            for action in legal:
                next_state = model.step_state(state, action)
                key = (level.level_id, next_state)
                if key not in teacher_cache:
                    teacher_cache[key] = best_first_from_state(
                        model,
                        next_state,
                        max_expansions=args.teacher_max_expansions,
                    )
                teacher = teacher_cache[key]
                next_value = _target_from_teacher(args, teacher)
                next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
                reward = float(stock_reward(stock.astype(np.float64), next_stock))
                teacher_q = float(reward + args.gamma * next_value)
                fixed_stock_q = float(reward + args.gamma * stock_value(next_stock))
                feature_vec = action_prior_feature_vector(
                    model,
                    state,
                    stock,
                    action,
                    target2_structural_score=target2_by_flat.get(int(action.flat_index), 0.0),
                )
                row = {
                    "level_id": level.level_id,
                    "split": split_name,
                    "state_key": f"{level.level_id}#{state_index}",
                    "state_depth": int(state_depth),
                    "flat_index": int(action.flat_index),
                    "teacher_q": teacher_q,
                    "fixed_stock_q": fixed_stock_q,
                    "stock_reward": reward,
                    "target2_structural_score": float(target2_by_flat.get(int(action.flat_index), 0.0)),
                    "goal_room_conflict_score": float(
                        _goal_room_conflict_score_for_action(model, state, action)
                    ),
                    "teacher_failure_type": str(teacher.failure_type),
                    "teacher_solved": bool(teacher.solved),
                    "features": feature_vec.astype(np.float32),
                }
                teacher_counts[str(teacher.failure_type)] += 1
                action_rows.append(row)
            if len(action_rows) <= 1:
                continue
            best_q = max(row["teacher_q"] for row in action_rows)
            for row in action_rows:
                row["teacher_best_label"] = float(abs(row["teacher_q"] - best_q) < 1e-9)
                rows.append(row)
            action_states += 1
            state_index += 1
            split_counts[split_name] += 1
        print(
            json.dumps(
                {
                    "level_id": level.level_id,
                    "split": split_name,
                    "action_states": action_states,
                    "rows_so_far": len(rows),
                },
                ensure_ascii=False,
            )
        )
    return rows, {
        "level_split": split,
        "selected_train_levels": [level.level_id for level in selected],
        "internal_holdout_levels": sorted(holdout_ids),
        "split_state_counts": dict(split_counts),
        "teacher_counts": dict(teacher_counts),
        "collection_elapsed_sec": float(perf_counter() - t0),
    }


def evaluate_signal(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not rows:
        return {"status": "NO_ROWS"}
    features = np.stack([row["features"] for row in rows]).astype(np.float64)
    target = np.asarray([row["teacher_q"] for row in rows], dtype=np.float64)
    train_idx = np.asarray([idx for idx, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    valid_idx = np.asarray([idx for idx, row in enumerate(rows) if row["split"] == "valid"], dtype=np.int64)
    if len(train_idx) == 0 or len(valid_idx) == 0:
        return {"status": "MISSING_SPLIT", "train_rows": int(len(train_idx)), "valid_rows": int(len(valid_idx))}

    train_x_std, mean, std = _standardize(features[train_idx], features[train_idx])
    all_x_std = (features - mean) / std
    weights = _fit_ridge(train_x_std, target[train_idx], args.ridge_alpha)
    pred = _predict_ridge(all_x_std, weights)
    enriched = []
    for idx, row in enumerate(rows):
        copy = {key: value for key, value in row.items() if key != "features"}
        copy["ridge_pred_q"] = float(pred[idx])
        enriched.append(copy)

    metrics: dict[str, Any] = {
        "ridge_all_features": {
            "feature_count": int(features.shape[1]),
            "train": _regression_metrics(pred[train_idx], target[train_idx]),
            "valid": _regression_metrics(pred[valid_idx], target[valid_idx]),
            "train_rank": _rank_metrics([enriched[idx] for idx in train_idx], "ridge_pred_q"),
            "valid_rank": _rank_metrics([enriched[idx] for idx in valid_idx], "ridge_pred_q"),
        },
        "baselines": {},
    }
    for score_key in (
        "stock_reward",
        "fixed_stock_q",
        "target2_structural_score",
        "goal_room_conflict_score",
    ):
        metrics["baselines"][score_key] = {
            "train_rank": _rank_metrics([enriched[idx] for idx in train_idx], score_key),
            "valid_rank": _rank_metrics([enriched[idx] for idx in valid_idx], score_key),
        }

    label_counts = Counter(
        f"{row['split']}:{'best' if row['teacher_best_label'] > 0.5 else 'nonbest'}"
        for row in enriched
    )
    q_stats = {}
    for split_name, idxs in (("train", train_idx), ("valid", valid_idx)):
        values = target[idxs]
        q_stats[split_name] = {
            "rows": int(len(idxs)),
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "std": float(np.std(values)) if len(values) else 0.0,
            "min": float(np.min(values)) if len(values) else 0.0,
            "max": float(np.max(values)) if len(values) else 0.0,
            "positive_fraction": float(np.mean(values > 0.0)) if len(values) else 0.0,
        }
    metrics["label_counts"] = dict(label_counts)
    metrics["teacher_q_stats"] = q_stats
    metrics["sample_rows"] = enriched[:40]
    return metrics


def build_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    failed = []
    if metrics.get("status") in {"NO_ROWS", "MISSING_SPLIT"}:
        return {
            "verdict": "FAIL",
            "failed_checks": [str(metrics.get("status"))],
            "thresholds": {},
            "observed": metrics,
        }
    ridge = metrics["ridge_all_features"]
    valid_rank = ridge["valid_rank"]
    train_counts = metrics["label_counts"]
    valid_q = metrics["teacher_q_stats"]["valid"]
    if train_counts.get("train:best", 0) < 10:
        failed.append("train_best_action_labels_min_10")
    if train_counts.get("valid:best", 0) < 3:
        failed.append("valid_best_action_labels_min_3")
    if valid_rank.get("states", 0) < 5:
        failed.append("valid_rank_states_min_5")
    if valid_rank.get("top1_agreement", 0.0) < 0.40:
        failed.append("valid_top1_agreement_min_0.40")
    if ridge["valid"].get("corr", 0.0) < 0.20:
        failed.append("valid_q_corr_min_0.20")
    if valid_q.get("std", 0.0) < 0.25:
        failed.append("valid_teacher_q_std_min_0.25")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "thresholds": {
            "train_best_action_labels_min": 10,
            "valid_best_action_labels_min": 3,
            "valid_rank_states_min": 5,
            "valid_top1_agreement_min": 0.40,
            "valid_q_corr_min": 0.20,
            "valid_teacher_q_std_min": 0.25,
        },
        "observed": {
            "train_best_action_labels": int(train_counts.get("train:best", 0)),
            "valid_best_action_labels": int(train_counts.get("valid:best", 0)),
            "valid_rank_states": int(valid_rank.get("states", 0)),
            "valid_top1_agreement": float(valid_rank.get("top1_agreement", 0.0)),
            "valid_q_corr": float(ridge["valid"].get("corr", 0.0)),
            "valid_teacher_q_std": float(valid_q.get("std", 0.0)),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    rows, collection = collect_rows(args)
    metrics = evaluate_signal(rows, args)
    gate = build_gate(metrics)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_train_action_target_signal_audit_v1",
        },
        "config": vars(args),
        "audit_contract": {
            "purpose": "check train-split offline action-Q signal before training a learned planner prior",
            "uses_stage2_train_split_only": True,
            "heldout_valid_levels_used_for_training_labels": False,
            "teacher_usage": "offline_successor_label_only",
            "teacher_used_online": False,
            "best_first_used_online": False,
            "planner_modified": False,
            "transition_provider_modified": False,
        },
        "collection": collection,
        "feature_names": list(ACTION_PRIOR_FEATURE_NAMES),
        "metrics": metrics,
        "target_signal_gate": gate,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "target_signal_gate": gate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
