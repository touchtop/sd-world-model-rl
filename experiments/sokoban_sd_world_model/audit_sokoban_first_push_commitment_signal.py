#!/usr/bin/env python3
"""Audit first-push commitment targets induced by route/goal options.

The target aggregates option-level offline teacher-Q labels back to the first
primitive push:

    Q_commit(s, a_first) = max_o Q_teacher(s, o), where first(o) = a_first.

This checks whether route/goal option structure can improve the selection of
the next committed box/direction without directly adding those options to
online SD-MCTS.
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

from audit_sokoban_route_goal_commitment_options import (  # noqa: E402
    _candidate_options,
    _label_option,
)
from audit_sokoban_train_action_target_signal import (  # noqa: E402
    _fit_ridge,
    _predict_ridge,
    _regression_metrics,
    _standardize,
)
from audit_stage2_learned_provider_sdmcts import action_to_dict  # noqa: E402
from sokoban_learned_action_prior import (  # noqa: E402
    ACTION_PRIOR_FEATURE_NAMES,
    action_prior_feature_vector,
)
from src.sokoban import SokobanGoalDirectedSDModel, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import provider_level_split  # noqa: E402
from validate_sokoban_sd_design import iter_states, stock_proxy  # noqa: E402


COMMITMENT_FEATURE_NAMES = ACTION_PRIOR_FEATURE_NAMES + (
    "commitment:route_goal_available",
    "commitment:route_goal_count_norm",
    "commitment:route_goal_best_structural_score",
    "commitment:route_goal_best_duration_norm",
    "commitment:route_goal_best_distance_gain",
    "commitment:route_goal_best_end_distance_norm",
    "commitment:route_goal_reaches_goal",
    "commitment:target2_available",
    "commitment:target2_best_structural_score",
)


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
    parser.add_argument("--max-route-length", type=int, default=4)
    parser.add_argument("--max-route-options-per-root", type=int, default=16)
    parser.add_argument("--max-target2-options-per-root", type=int, default=6)
    parser.add_argument("--teacher-max-expansions", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--teacher-tie-tolerance", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/first_push_commitment_signal_quick_seed0.json",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _rank_metrics(
    rows: Sequence[dict[str, Any]],
    score_key: str,
    *,
    tie_tolerance: float,
) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["state_key"]].append(row)
    exact_hits = []
    tier_hits = []
    regrets = []
    random_hits = []
    state_count = 0
    pred_route_rows = 0
    teacher_route_states = 0
    for state_rows in groups.values():
        if len(state_rows) <= 1:
            continue
        state_count += 1
        best_value = max(float(row["teacher_q"]) for row in state_rows)
        teacher_best = max(state_rows, key=lambda row: (row["teacher_q"], -row["flat_index"]))
        pred_best = max(state_rows, key=lambda row: (row[score_key], -row["flat_index"]))
        best_tier = {
            int(row["flat_index"])
            for row in state_rows
            if float(row["teacher_q"]) >= best_value - tie_tolerance
        }
        exact_hits.append(int(pred_best["flat_index"]) == int(teacher_best["flat_index"]))
        tier_hits.append(int(pred_best["flat_index"]) in best_tier)
        regrets.append(float(best_value - float(pred_best["teacher_q"])))
        random_hits.append(len(best_tier) / max(len(state_rows), 1))
        pred_route_rows += int(bool(pred_best.get("best_option_is_route_goal", False)))
        teacher_route_states += int(
            any(
                bool(row.get("best_option_is_route_goal", False))
                and int(row["flat_index"]) in best_tier
                for row in state_rows
            )
        )
    return {
        "states": int(state_count),
        "top1_exact_agreement": float(np.mean(exact_hits)) if exact_hits else 0.0,
        "top_tier_hit": float(np.mean(tier_hits)) if tier_hits else 0.0,
        "mean_teacher_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_random_tier_hit": float(np.mean(random_hits)) if random_hits else 0.0,
        "pred_route_goal_fraction": _safe_div(pred_route_rows, state_count),
        "teacher_best_route_goal_state_fraction": _safe_div(teacher_route_states, state_count),
    }


def _option_label_rows(
    args: argparse.Namespace,
    model: SokobanGoalDirectedSDModel,
    state,
) -> list[dict[str, Any]]:
    teacher_cache = {}
    rows = []
    for option in _candidate_options(model, state, args):
        label = _label_option(args, model, state, option, teacher_cache)
        if label is None:
            continue
        rows.append(
            {
                "first_flat": int(option.first.flat_index),
                "source": option.source,
                "duration": int(option.duration),
                "structural_score": float(option.structural_score),
                "route_distance_gain": float(getattr(option, "route_distance_gain", 0.0)),
                "end_distance": getattr(option, "end_distance", None),
                "teacher_q": float(label["teacher_q"]),
                "fixed_stock_return": float(label["fixed_stock_return"]),
                "exact_option_score": float(label["exact_option_score"]),
                "teacher_failure_type": str(label["teacher_failure_type"]),
                "teacher_solved": bool(label["teacher_solved"]),
            }
        )
    return rows


def _feature_extras(
    *,
    action_flat: int,
    option_rows: Sequence[dict[str, Any]],
    max_route_length: int,
    board_scale: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    first_rows = [row for row in option_rows if int(row["first_flat"]) == action_flat]
    route_rows = [row for row in first_rows if row["source"] == "route_goal_monotone_option"]
    target2_rows = [row for row in first_rows if row["source"] == "target2_local_structural_option"]
    route_best = max(route_rows, key=lambda row: row["structural_score"], default=None)
    target2_best = max(target2_rows, key=lambda row: row["structural_score"], default=None)
    route_best_end = route_best.get("end_distance") if route_best else None
    route_best_end_norm = (
        _safe_div(float(route_best_end), board_scale)
        if route_best_end is not None
        else 0.0
    )
    route_reaches_goal = bool(route_best is not None and float(route_best_end or 1.0) <= 0.0)
    extras = np.asarray(
        [
            float(bool(route_rows)),
            _safe_div(len(route_rows), max(1, max_route_length * 4)),
            float(route_best["structural_score"]) if route_best else 0.0,
            _safe_div(float(route_best["duration"]), max_route_length) if route_best else 0.0,
            float(route_best["route_distance_gain"]) if route_best else 0.0,
            route_best_end_norm,
            float(route_reaches_goal),
            float(bool(target2_rows)),
            float(target2_best["structural_score"]) if target2_best else 0.0,
        ],
        dtype=np.float32,
    )
    meta = {
        "route_goal_available": bool(route_rows),
        "route_goal_count": int(len(route_rows)),
        "route_goal_best_structural_score": float(route_best["structural_score"]) if route_best else 0.0,
        "route_goal_best_duration": int(route_best["duration"]) if route_best else 0,
        "route_goal_best_distance_gain": float(route_best["route_distance_gain"]) if route_best else 0.0,
        "route_goal_best_end_distance": route_best_end,
        "route_goal_reaches_goal": route_reaches_goal,
        "target2_available": bool(target2_rows),
        "target2_best_structural_score": float(target2_best["structural_score"]) if target2_best else 0.0,
    }
    return extras, meta


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    split = provider_level_split(args.provider_checkpoint)
    train_ids = set(split["train_levels"])
    selected = [level for level in levels if level.level_id in train_ids]
    if args.max_train_levels > 0:
        selected = selected[: args.max_train_levels]
    if not selected:
        raise SystemExit("No stage-2 train levels selected for first-push commitment audit.")

    holdout_count = (
        max(1, int(round(len(selected) * args.internal_holdout_fraction)))
        if len(selected) > 3
        else 1
    )
    holdout_ids = {level.level_id for level in selected[-holdout_count:]}
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    option_source_counts: Counter[str] = Counter()
    best_option_source_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    t0 = perf_counter()

    for level in selected:
        split_name = "valid" if level.level_id in holdout_ids else "train"
        model = SokobanGoalDirectedSDModel(level)
        action_states = 0
        state_index = 0
        board_scale = max(model.level.height + model.level.width, 1)
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
            option_rows = _option_label_rows(args, model, state)
            if not option_rows:
                continue
            option_source_counts.update(row["source"] for row in option_rows)
            teacher_counts.update(row["teacher_failure_type"] for row in option_rows)
            rows_by_first: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for option_row in option_rows:
                rows_by_first[int(option_row["first_flat"])].append(option_row)

            state_rows: list[dict[str, Any]] = []
            for action in legal:
                first_rows = rows_by_first.get(int(action.flat_index), [])
                if not first_rows:
                    continue
                best_option = max(
                    first_rows,
                    key=lambda row: (float(row["teacher_q"]), float(row["structural_score"])),
                )
                primitive_row = next(
                    (row for row in first_rows if row["source"] == "primitive_push"),
                    best_option,
                )
                route_rows = [row for row in first_rows if row["source"] == "route_goal_monotone_option"]
                target2_rows = [row for row in first_rows if row["source"] == "target2_local_structural_option"]
                route_best_teacher = max(route_rows, key=lambda row: row["teacher_q"], default=None)
                target2_best_teacher = max(target2_rows, key=lambda row: row["teacher_q"], default=None)
                extras, meta = _feature_extras(
                    action_flat=int(action.flat_index),
                    option_rows=option_rows,
                    max_route_length=args.max_route_length,
                    board_scale=board_scale,
                )
                target2_structural = float(meta["target2_best_structural_score"])
                base = action_prior_feature_vector(
                    model,
                    state,
                    stock,
                    action,
                    target2_structural_score=target2_structural,
                )
                feature_vec = np.concatenate([base, extras], axis=0).astype(np.float32)
                row = {
                    "level_id": level.level_id,
                    "split": split_name,
                    "state_key": f"{level.level_id}#{state_index}",
                    "state_depth": int(state_depth),
                    "flat_index": int(action.flat_index),
                    "action": action_to_dict(action),
                    "teacher_q": float(best_option["teacher_q"]),
                    "teacher_best_option_source": best_option["source"],
                    "teacher_best_option_duration": int(best_option["duration"]),
                    "best_option_is_route_goal": best_option["source"] == "route_goal_monotone_option",
                    "best_option_is_target2": best_option["source"] == "target2_local_structural_option",
                    "best_option_is_primitive": best_option["source"] == "primitive_push",
                    "primitive_teacher_q": float(primitive_row["teacher_q"]),
                    "primitive_fixed_stock_return": float(primitive_row["fixed_stock_return"]),
                    "primitive_exact_option_score": float(primitive_row["exact_option_score"]),
                    "route_goal_best_teacher_q": (
                        float(route_best_teacher["teacher_q"]) if route_best_teacher else None
                    ),
                    "target2_best_teacher_q": (
                        float(target2_best_teacher["teacher_q"]) if target2_best_teacher else None
                    ),
                    "route_goal_structural_score": float(meta["route_goal_best_structural_score"]),
                    "route_goal_distance_gain": float(meta["route_goal_best_distance_gain"]),
                    "route_goal_count": int(meta["route_goal_count"]),
                    "route_goal_exact_option_score": (
                        float(max(route_rows, key=lambda row: row["exact_option_score"])["exact_option_score"])
                        if route_rows
                        else 0.0
                    ),
                    "target2_structural_score": target2_structural,
                    "route_goal_available": bool(meta["route_goal_available"]),
                    "target2_available": bool(meta["target2_available"]),
                    "features": feature_vec,
                }
                best_option_source_counts[f"{split_name}:{best_option['source']}"] += 1
                state_rows.append(row)
            if len(state_rows) <= 1:
                continue
            best_q = max(row["teacher_q"] for row in state_rows)
            for row in state_rows:
                row["teacher_best_label"] = float(row["teacher_q"] >= best_q - args.teacher_tie_tolerance)
                rows.append(row)
            action_states += 1
            state_index += 1
            split_counts[split_name] += 1
        print(
            json.dumps(
                {
                    "level_id": level.level_id,
                    "split": split_name,
                    "commitment_states": action_states,
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
        "option_source_counts": dict(option_source_counts),
        "best_option_source_counts": dict(best_option_source_counts),
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
        copy["ridge_pred_commit_q"] = float(pred[idx])
        enriched.append(copy)

    metrics: dict[str, Any] = {
        "ridge_all_commitment_features": {
            "feature_count": int(features.shape[1]),
            "train": _regression_metrics(pred[train_idx], target[train_idx]),
            "valid": _regression_metrics(pred[valid_idx], target[valid_idx]),
            "train_rank": _rank_metrics(
                [enriched[idx] for idx in train_idx],
                "ridge_pred_commit_q",
                tie_tolerance=args.teacher_tie_tolerance,
            ),
            "valid_rank": _rank_metrics(
                [enriched[idx] for idx in valid_idx],
                "ridge_pred_commit_q",
                tie_tolerance=args.teacher_tie_tolerance,
            ),
        },
        "baselines": {},
    }
    for score_key in (
        "primitive_teacher_q",
        "primitive_fixed_stock_return",
        "primitive_exact_option_score",
        "target2_structural_score",
        "route_goal_structural_score",
        "route_goal_distance_gain",
        "route_goal_exact_option_score",
        "route_goal_count",
    ):
        metrics["baselines"][score_key] = {
            "train_rank": _rank_metrics(
                [enriched[idx] for idx in train_idx],
                score_key,
                tie_tolerance=args.teacher_tie_tolerance,
            ),
            "valid_rank": _rank_metrics(
                [enriched[idx] for idx in valid_idx],
                score_key,
                tie_tolerance=args.teacher_tie_tolerance,
            ),
        }

    label_counts = Counter(
        f"{row['split']}:{'best' if row['teacher_best_label'] > 0.5 else 'nonbest'}"
        for row in enriched
    )
    best_source_counts = Counter(
        f"{row['split']}:{row['teacher_best_option_source']}"
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
            "route_goal_available_fraction": float(
                np.mean([enriched[idx]["route_goal_available"] for idx in idxs])
            )
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
    ridge = metrics["ridge_all_commitment_features"]
    valid_rank = ridge["valid_rank"]
    train_counts = metrics["label_counts"]
    valid_q = metrics["teacher_q_stats"]["valid"]
    source_counts = metrics["teacher_best_source_counts"]
    route_baseline = metrics["baselines"]["route_goal_structural_score"]["valid_rank"]
    if train_counts.get("train:best", 0) < 10:
        failed.append("train_best_first_push_labels_min_10")
    if train_counts.get("valid:best", 0) < 3:
        failed.append("valid_best_first_push_labels_min_3")
    if valid_rank.get("states", 0) < 5:
        failed.append("valid_rank_states_min_5")
    if valid_rank.get("top_tier_hit", 0.0) < 0.40:
        failed.append("valid_top_tier_hit_min_0.40")
    if ridge["valid"].get("corr", 0.0) < 0.20:
        failed.append("valid_q_corr_min_0.20")
    if valid_q.get("std", 0.0) < 0.25:
        failed.append("valid_teacher_q_std_min_0.25")
    if valid_q.get("route_goal_available_fraction", 0.0) < 0.25:
        failed.append("valid_route_goal_available_fraction_min_0.25")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "thresholds": {
            "train_best_first_push_labels_min": 10,
            "valid_best_first_push_labels_min": 3,
            "valid_rank_states_min": 5,
            "valid_top_tier_hit_min": 0.40,
            "valid_q_corr_min": 0.20,
            "valid_teacher_q_std_min": 0.25,
            "valid_route_goal_available_fraction_min": 0.25,
        },
        "observed": {
            "train_best_first_push_labels": int(train_counts.get("train:best", 0)),
            "valid_best_first_push_labels": int(train_counts.get("valid:best", 0)),
            "valid_rank_states": int(valid_rank.get("states", 0)),
            "valid_top_tier_hit": float(valid_rank.get("top_tier_hit", 0.0)),
            "valid_top1_exact_agreement": float(valid_rank.get("top1_exact_agreement", 0.0)),
            "valid_q_corr": float(ridge["valid"].get("corr", 0.0)),
            "valid_teacher_q_std": float(valid_q.get("std", 0.0)),
            "valid_route_goal_available_fraction": float(
                valid_q.get("route_goal_available_fraction", 0.0)
            ),
            "valid_route_goal_structural_top_tier_hit": float(
                route_baseline.get("top_tier_hit", 0.0)
            ),
            "valid_teacher_best_route_goal_source_labels": int(
                source_counts.get("valid:route_goal_monotone_option", 0)
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
            "version_label": "sokoban_first_push_commitment_signal_audit_v1",
        },
        "config": vars(args),
        "audit_contract": {
            "purpose": "check option-induced first-push commitment target before planner integration",
            "uses_stage2_train_split_only": True,
            "heldout_valid_levels_used_for_training_labels": False,
            "teacher_usage": "offline_option_successor_label_only",
            "teacher_used_online": False,
            "best_first_used_online": False,
            "planner_modified": False,
            "transition_provider_modified": False,
            "candidate_options": "primitive_pushes_plus_target2_plus_route_goal_monotone_options",
            "target_formula": "Q_commit(s,a_first)=max_o Q_teacher(s,o), first(o)=a_first",
        },
        "collection": collection,
        "feature_names": list(COMMITMENT_FEATURE_NAMES),
        "metrics": metrics,
        "first_push_commitment_signal_gate": gate,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "first_push_commitment_signal_gate": gate},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
