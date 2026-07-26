#!/usr/bin/env python3
"""Audit offline option-level target signal for Sokoban SD-MCTS.

This diagnostic checks whether short push options carry learnable value signal
that is missed by the current primitive first-push prior. It is deliberately
offline: the exact-state teacher labels only train-split successor states and
is never called by SD-MCTS during planning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
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

from audit_stage2_learned_provider_sdmcts import action_to_dict, stock_reward, stock_value  # noqa: E402
from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    ExpertAbstractPlannerFeedback,
    PlannerAction,
    _score_option_sequence,
    _target2_options,
)
from sokoban_learned_action_prior import (  # noqa: E402
    ACTION_PRIOR_FEATURE_NAMES,
    action_prior_feature_vector,
)
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import best_first_from_state, provider_level_split, target_from_teacher  # noqa: E402
from validate_sokoban_sd_design import iter_states, stock_proxy  # noqa: E402


OPTION_EXTRA_FEATURE_NAMES = (
    "option:duration_norm",
    "option:target2_structural_score",
    "option:teacher_step_reward_sum",
    "option:fixed_stock_step_return",
    "option:first_push_target2_rank_norm",
)
OPTION_FEATURE_NAMES = ACTION_PRIOR_FEATURE_NAMES + OPTION_EXTRA_FEATURE_NAMES


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
    parser.add_argument("--max-options-per-node", type=int, default=6)
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
        default="results/sokoban_sd_world_model/option_target_signal_quick_seed0.json",
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
    option_hit_states = 0
    state_count = 0
    for state_rows in groups.values():
        if len(state_rows) <= 1:
            continue
        state_count += 1
        teacher_best = max(state_rows, key=lambda row: (row["teacher_q"], -row["row_index"]))
        pred_best = max(state_rows, key=lambda row: (row[score_key], -row["row_index"]))
        hits.append(pred_best["row_index"] == teacher_best["row_index"])
        regrets.append(float(teacher_best["teacher_q"] - pred_best["teacher_q"]))
        random_hits.append(1.0 / len(state_rows))
        option_hit_states += int(teacher_best["duration"] > 1)
    return {
        "states": int(state_count),
        "top1_agreement": float(np.mean(hits)) if hits else 0.0,
        "mean_teacher_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_random_top1": float(np.mean(random_hits)) if random_hits else 0.0,
        "teacher_best_option_state_fraction": _safe_div(option_hit_states, state_count),
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


def _option_label(
    args: argparse.Namespace,
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    option: PlannerAction,
    stock: np.ndarray,
    teacher_cache: dict[tuple[str, SokobanState], Any],
    level_id: str,
) -> tuple[float, dict[str, Any]] | None:
    current = state
    current_stock = np.asarray(stock, dtype=np.float64)
    discounted_reward = 0.0
    fixed_stock_return = 0.0
    discount = 1.0
    executed = 0
    for primitive in option.primitives:
        legal = set(model.legal_pushes(current))
        if primitive not in legal:
            return None
        next_state = model.step_state(current, primitive)
        next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
        step_reward = stock_reward(current_stock, next_stock)
        discounted_reward += discount * step_reward
        fixed_stock_return += discount * step_reward
        current = next_state
        current_stock = next_stock
        discount *= args.gamma
        executed += 1
        if model.is_solved(current):
            break
    key = (level_id, current)
    if key not in teacher_cache:
        teacher_cache[key] = best_first_from_state(
            model,
            current,
            max_expansions=args.teacher_max_expansions,
        )
    teacher = teacher_cache[key]
    next_value = _target_from_teacher(args, teacher)
    teacher_q = float(discounted_reward + discount * next_value)
    fixed_stock_return += discount * stock_value(current_stock)
    return teacher_q, {
        "next_state": current,
        "discounted_reward": float(discounted_reward),
        "fixed_stock_return": float(fixed_stock_return),
        "discount": float(discount),
        "executed": int(executed),
        "teacher": teacher,
    }


def _candidate_options(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    args: argparse.Namespace,
) -> list[PlannerAction]:
    primitive_options = [
        PlannerAction(primitives=(action,), source="primitive_push", structural_score=0.0)
        for action in model.legal_pushes(state)
    ]
    target2_options = _target2_options(
        model,
        state,
        max_options_per_node=args.max_options_per_node,
    )
    seen: set[tuple[int, ...]] = set()
    options: list[PlannerAction] = []
    for option in primitive_options + target2_options:
        key = option.flat_index_tuple
        if key in seen:
            continue
        seen.add(key)
        options.append(option)
    return options


def _first_push_target2_ranks(options: Sequence[PlannerAction]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for option in options:
        if option.duration <= 1:
            scores.setdefault(option.first.flat_index, 0.0)
            continue
        scores[option.first.flat_index] = max(
            scores.get(option.first.flat_index, float("-inf")),
            float(option.structural_score),
        )
    ordered = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    denom = max(len(ordered) - 1, 1)
    return {flat: 1.0 - _safe_div(rank, denom) for rank, (flat, _score) in enumerate(ordered)}


def _option_feature_vector(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    stock: np.ndarray,
    option: PlannerAction,
    label_info: dict[str, Any],
    first_push_ranks: dict[int, float],
) -> np.ndarray:
    base = action_prior_feature_vector(
        model,
        state,
        stock,
        option.first,
        target2_structural_score=float(option.structural_score),
    )
    extras = np.asarray(
        [
            _safe_div(option.duration, 2.0),
            float(option.structural_score),
            float(label_info["discounted_reward"]),
            float(label_info["fixed_stock_return"]),
            float(first_push_ranks.get(option.first.flat_index, 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extras], axis=0).astype(np.float32)


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    split = provider_level_split(args.provider_checkpoint)
    train_ids = set(split["train_levels"])
    selected = [level for level in levels if level.level_id in train_ids]
    if args.max_train_levels > 0:
        selected = selected[: args.max_train_levels]
    if not selected:
        raise SystemExit("No stage-2 train levels selected for option-target audit.")

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
    option_source_counts: Counter[str] = Counter()
    feedback = ExpertAbstractPlannerFeedback()
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
            options = _candidate_options(model, state, args)
            if len(options) <= 1:
                continue
            first_push_ranks = _first_push_target2_ranks(options)
            state_rows: list[dict[str, Any]] = []
            for option in options:
                label = _option_label(args, model, state, option, stock, teacher_cache, level.level_id)
                if label is None:
                    continue
                teacher_q, label_info = label
                teacher = label_info["teacher"]
                feature_vec = _option_feature_vector(
                    model,
                    state,
                    stock,
                    option,
                    label_info,
                    first_push_ranks,
                )
                exact_score, _exact_next_state = _score_option_sequence(model, state, option.primitives)
                feedback_prior = feedback.action_priors(model, state, stock, legal).get(option.first, 0.0)
                row = {
                    "level_id": level.level_id,
                    "split": split_name,
                    "state_key": f"{level.level_id}#{state_index}",
                    "state_depth": int(state_depth),
                    "row_index": len(rows) + len(state_rows),
                    "duration": int(option.duration),
                    "source": option.source,
                    "flat_index_tuple": [int(idx) for idx in option.flat_index_tuple],
                    "first_action": action_to_dict(option.first),
                    "teacher_q": float(teacher_q),
                    "fixed_stock_return": float(label_info["fixed_stock_return"]),
                    "target2_structural_score": float(option.structural_score),
                    "exact_option_score": float(exact_score),
                    "expert_first_prior": float(feedback_prior),
                    "first_push_target2_rank": float(
                        first_push_ranks.get(option.first.flat_index, 0.0)
                    ),
                    "teacher_failure_type": str(teacher.failure_type),
                    "teacher_solved": bool(teacher.solved),
                    "executed": int(label_info["executed"]),
                    "features": feature_vec.astype(np.float32),
                }
                teacher_counts[str(teacher.failure_type)] += 1
                option_source_counts[option.source] += 1
                state_rows.append(row)
            if len(state_rows) <= 1:
                continue
            best_q = max(row["teacher_q"] for row in state_rows)
            best_first_push_q: dict[int, float] = {}
            for row in state_rows:
                first_flat = int(row["flat_index_tuple"][0])
                best_first_push_q[first_flat] = max(
                    best_first_push_q.get(first_flat, float("-inf")),
                    float(row["teacher_q"]),
                )
            best_first_value = max(best_first_push_q.values()) if best_first_push_q else best_q
            for row in state_rows:
                first_flat = int(row["flat_index_tuple"][0])
                row["teacher_best_label"] = float(abs(row["teacher_q"] - best_q) < 1e-9)
                row["teacher_best_first_push_label"] = float(
                    abs(best_first_push_q[first_flat] - best_first_value) < 1e-9
                )
                rows.append(row)
            action_states += 1
            state_index += 1
            split_counts[split_name] += 1
        print(
            json.dumps(
                {
                    "level_id": level.level_id,
                    "split": split_name,
                    "option_states": action_states,
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
        "option_source_counts": dict(option_source_counts),
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
        "ridge_all_option_features": {
            "feature_count": int(features.shape[1]),
            "train": _regression_metrics(pred[train_idx], target[train_idx]),
            "valid": _regression_metrics(pred[valid_idx], target[valid_idx]),
            "train_rank": _rank_metrics([enriched[idx] for idx in train_idx], "ridge_pred_q"),
            "valid_rank": _rank_metrics([enriched[idx] for idx in valid_idx], "ridge_pred_q"),
        },
        "baselines": {},
    }
    for score_key in (
        "fixed_stock_return",
        "target2_structural_score",
        "exact_option_score",
        "expert_first_prior",
        "first_push_target2_rank",
    ):
        metrics["baselines"][score_key] = {
            "train_rank": _rank_metrics([enriched[idx] for idx in train_idx], score_key),
            "valid_rank": _rank_metrics([enriched[idx] for idx in valid_idx], score_key),
        }

    label_counts = Counter(
        f"{row['split']}:{'best' if row['teacher_best_label'] > 0.5 else 'nonbest'}"
        for row in enriched
    )
    best_source_counts = Counter(
        f"{row['split']}:{row['source']}"
        for row in enriched
        if row["teacher_best_label"] > 0.5
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
            "option_fraction": float(np.mean([enriched[idx]["duration"] > 1 for idx in idxs]))
            if len(idxs)
            else 0.0,
        }
    metrics["label_counts"] = dict(label_counts)
    metrics["teacher_best_source_counts"] = dict(best_source_counts)
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
    ridge = metrics["ridge_all_option_features"]
    valid_rank = ridge["valid_rank"]
    train_counts = metrics["label_counts"]
    valid_q = metrics["teacher_q_stats"]["valid"]
    best_source_counts = metrics["teacher_best_source_counts"]
    if train_counts.get("train:best", 0) < 10:
        failed.append("train_best_option_labels_min_10")
    if train_counts.get("valid:best", 0) < 3:
        failed.append("valid_best_option_labels_min_3")
    if valid_rank.get("states", 0) < 5:
        failed.append("valid_rank_states_min_5")
    if valid_rank.get("top1_agreement", 0.0) < 0.40:
        failed.append("valid_top1_agreement_min_0.40")
    if ridge["valid"].get("corr", 0.0) < 0.20:
        failed.append("valid_q_corr_min_0.20")
    if valid_q.get("std", 0.0) < 0.25:
        failed.append("valid_teacher_q_std_min_0.25")
    if best_source_counts.get("train:target2_local_structural_option", 0) < 3:
        failed.append("train_best_target2_option_labels_min_3")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "thresholds": {
            "train_best_option_labels_min": 10,
            "valid_best_option_labels_min": 3,
            "valid_rank_states_min": 5,
            "valid_top1_agreement_min": 0.40,
            "valid_q_corr_min": 0.20,
            "valid_teacher_q_std_min": 0.25,
            "train_best_target2_option_labels_min": 3,
        },
        "observed": {
            "train_best_option_labels": int(train_counts.get("train:best", 0)),
            "valid_best_option_labels": int(train_counts.get("valid:best", 0)),
            "valid_rank_states": int(valid_rank.get("states", 0)),
            "valid_top1_agreement": float(valid_rank.get("top1_agreement", 0.0)),
            "valid_q_corr": float(ridge["valid"].get("corr", 0.0)),
            "valid_teacher_q_std": float(valid_q.get("std", 0.0)),
            "train_best_target2_option_labels": int(
                best_source_counts.get("train:target2_local_structural_option", 0)
            ),
            "valid_teacher_best_option_state_fraction": float(
                valid_rank.get("teacher_best_option_state_fraction", 0.0)
            ),
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
            "version_label": "sokoban_option_target_signal_audit_v1",
        },
        "config": vars(args),
        "audit_contract": {
            "purpose": "check train-split offline option-Q signal before changing TVA or planner prior targets",
            "uses_stage2_train_split_only": True,
            "heldout_valid_levels_used_for_training_labels": False,
            "teacher_usage": "offline_option_successor_label_only",
            "teacher_used_online": False,
            "best_first_used_online": False,
            "planner_modified": False,
            "transition_provider_modified": False,
            "candidate_options": "primitive_pushes_plus_target2_local_structural_options",
        },
        "collection": collection,
        "feature_names": list(OPTION_FEATURE_NAMES),
        "metrics": metrics,
        "option_target_signal_gate": gate,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "option_target_signal_gate": gate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
