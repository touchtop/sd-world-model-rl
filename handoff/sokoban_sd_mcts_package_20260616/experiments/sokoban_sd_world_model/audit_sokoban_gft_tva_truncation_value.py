#!/usr/bin/env python3
"""Audit GFT-TVA accuracy on independently labelled truncation states.

This script does not let TVA choose actions.  It samples leaf states reached by
the current GFT-SD-MCTS search using an explicit SD leaf, labels those states
offline with an exact-state best-first teacher, and then checks whether a
GFT-TVA checkpoint predicts the resulting remaining-value labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sokoban_gft_sdmcts import (  # noqa: E402
    GFTSDMCTSPlanner,
    GFTTvaCheckpoint,
    discounted_progress_gft_sd_value,
    gft_model_from_level,
    gft_sd_leaf_weights_from_args,
    gft_weights_from_args,
    load_tva_checkpoint,
    normalize_gft_tva_target_mode,
    predict_gft_tva,
    projected_gft_sd_value,
)
from evaluate_sokoban_grid_sdmcts import _rel  # noqa: E402
from src.sokoban import SokobanGFTSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import best_first_from_state, target_from_teacher  # noqa: E402


@dataclass
class LeafSample:
    level_index: int
    level_id: str
    root_step: int
    state: SokobanState
    state_type: str
    planner_blocked: bool


class LeafRecorder:
    def __init__(self, *, max_per_level: int):
        self.max_per_level = int(max_per_level)
        self.samples: list[LeafSample] = []
        self._seen: set[tuple[str, tuple]] = set()
        self._counts: Counter[int] = Counter()

    def add(
        self,
        model: SokobanGFTSDModel,
        state: SokobanState,
        *,
        level_index: int,
        level_id: str,
        root_step: int,
        planner_blocked: bool,
    ) -> None:
        if self._counts[int(level_index)] >= self.max_per_level:
            return
        key = (level_id, model.gft_canonical_key_cached(state))
        if key in self._seen:
            return
        self._seen.add(key)
        if model.is_solved(state):
            state_type = "solved"
        elif model.is_deadlocked(state):
            state_type = "deadlocked"
        elif not model.legal_flows(state):
            state_type = "no_legal_actions"
        else:
            state_type = "active"
        self.samples.append(
            LeafSample(
                level_index=int(level_index),
                level_id=level_id,
                root_step=int(root_step),
                state=state,
                state_type=state_type,
                planner_blocked=bool(planner_blocked),
            )
        )
        self._counts[int(level_index)] += 1


class AuditedGFTSDMCTSPlanner(GFTSDMCTSPlanner):
    def __init__(
        self,
        *args,
        leaf_recorder: LeafRecorder,
        audit_level_index: int,
        audit_level_id: str,
        audit_root_step: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.leaf_recorder = leaf_recorder
        self.audit_level_index = int(audit_level_index)
        self.audit_level_id = str(audit_level_id)
        self.audit_root_step = int(audit_root_step)

    def leaf_value(self, state) -> float:
        self.leaf_recorder.add(
            self.model,
            state,
            level_index=self.audit_level_index,
            level_id=self.audit_level_id,
            root_step=self.audit_root_step,
            planner_blocked=self.is_blocked(state),
        )
        return super().leaf_value(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument("--level-offset", type=int, default=48)
    parser.add_argument("--max-levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--budget", type=int, default=192)
    parser.add_argument("--closed-loop-steps", type=int, default=2)
    parser.add_argument("--max-leaf-states-per-level", type=int, default=12)
    parser.add_argument("--teacher-max-expansions", type=int, default=4000)
    parser.add_argument("--uct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--unsolved-value", type=float, default=-2.5)
    parser.add_argument("--constraint-mode", choices=("none", "tier_a", "exact_deadlock"), default="tier_a")
    parser.add_argument("--select-by", choices=("visits", "q"), default="visits")
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--transposition", action="store_true", default=True)
    parser.add_argument("--no-transposition", action="store_false", dest="transposition")
    parser.add_argument("--leaf-value-for-sampling", choices=("gft_sd", "sd_heuristic", "zero"), default="gft_sd")
    parser.add_argument(
        "--tva-checkpoint",
        default="results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json",
    )
    parser.add_argument("--tva-value-blend", type=float, default=1.0)
    parser.add_argument(
        "--gft-prior",
        choices=("none", "gft_delta"),
        default="gft_delta",
    )
    parser.add_argument("--gft-prior-temperature", type=float, default=2.0)
    parser.add_argument("--gft-prior-uniform-mix", type=float, default=0.1)
    parser.add_argument("--gft-progress-weight", type=float, default=1.0)
    parser.add_argument("--gft-goal-weight", type=float, default=1.0)
    parser.add_argument("--gft-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-capacity-weight", type=float, default=0.0)
    parser.add_argument("--gft-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-ordering-weight", type=float, default=0.0)
    parser.add_argument("--gft-commitment-weight", type=float, default=0.0)
    parser.add_argument("--gft-bottleneck-weight", type=float, default=0.0)
    parser.add_argument("--gft-revisit-weight", type=float, default=1.0)
    parser.add_argument("--gft-transport-turn-weight", type=float, default=0.10)
    parser.add_argument("--gft-transport-access-weight", type=float, default=0.25)
    parser.add_argument("--gft-transport-zone-weight", type=float, default=0.25)
    parser.add_argument("--gft-soft-deadlock-mobility-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-matching-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-capacity-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-access-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-order-weight", type=float, default=1.0)
    parser.add_argument("--gft-commitment-beta", type=float, default=1.0)
    parser.add_argument("--gft-sinkhorn-iterations", type=int, default=25)
    parser.add_argument(
        "--gft-sd-leaf-mode",
        choices=("linear_projected", "discounted_progress"),
        default="discounted_progress",
    )
    parser.add_argument("--gft-sd-leaf-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-capacity-weight", type=float, default=0.05)
    parser.add_argument("--gft-sd-leaf-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-ordering-weight", type=float, default=0.05)
    parser.add_argument("--gft-sd-leaf-commitment-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_seed0.json",
    )
    return parser


def _safe_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size <= 1 or float(np.std(pred)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def regression_metrics(rows: Sequence[dict[str, Any]], pred_key: str) -> dict[str, float]:
    if not rows:
        return {
            "samples": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "corr": 0.0,
            "bias": 0.0,
            "sign_accuracy": 0.0,
        }
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    target = np.asarray([float(row["target_value"]) for row in rows], dtype=np.float64)
    err = pred - target
    return {
        "samples": float(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": _safe_corr(pred, target),
        "bias": float(np.mean(err)),
        "sign_accuracy": float(np.mean(np.signbit(pred) == np.signbit(target))),
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
    }


def pairwise_order_accuracy(rows: Sequence[dict[str, Any]], pred_key: str, *, tolerance: float) -> dict[str, float]:
    total = 0
    correct = 0
    tied = 0
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["level_index"]), int(row["root_step"]))].append(row)
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                target_delta = float(group[i]["target_value"]) - float(group[j]["target_value"])
                if abs(target_delta) <= tolerance:
                    tied += 1
                    continue
                pred_delta = float(group[i][pred_key]) - float(group[j][pred_key])
                if abs(pred_delta) <= tolerance:
                    total += 1
                    continue
                total += 1
                if math.copysign(1.0, pred_delta) == math.copysign(1.0, target_delta):
                    correct += 1
    return {
        "pairs": float(total),
        "correct": float(correct),
        "target_tied_pairs": float(tied),
        "accuracy": float(correct / total) if total else 0.0,
    }


def compute_sd_heuristic_value(model: SokobanGFTSDModel, state: SokobanState, args: argparse.Namespace) -> float:
    p = model.gft_transport_progress_cached(state)
    if not math.isfinite(p):
        return float(args.deadend_value)
    if args.gamma == 1.0:
        return float(args.goal_value) - float(args.push_cost) * p
    discount = float(args.gamma) ** p
    return float(discount * float(args.goal_value) - float(args.push_cost) * (1.0 - discount) / (1.0 - float(args.gamma)))


def compute_gft_sd_leaf_value(model: SokobanGFTSDModel, state: SokobanState, args: argparse.Namespace) -> float:
    if args.gft_sd_leaf_mode == "discounted_progress":
        return discounted_progress_gft_sd_value(
            model,
            state,
            gamma=args.gamma,
            goal_value=args.goal_value,
            push_cost=args.push_cost,
            deadend_value=args.deadend_value,
            weights=gft_sd_leaf_weights_from_args(args),
        )
    return projected_gft_sd_value(
        model,
        state,
        deadend_value=args.deadend_value,
        goal_value=args.goal_value,
        weights=gft_sd_leaf_weights_from_args(args),
    )


def compute_tva_values(
    checkpoint: GFTTvaCheckpoint,
    model: SokobanGFTSDModel,
    state: SokobanState,
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    prediction = predict_gft_tva(checkpoint, model, state)
    target_mode = normalize_gft_tva_target_mode(checkpoint.target_mode)
    if target_mode == "gft_sd_residual":
        return prediction, compute_gft_sd_leaf_value(model, state, args) + prediction, "tva_leaf_value"
    if target_mode == "sd_heuristic_residual":
        return prediction, compute_sd_heuristic_value(model, state, args) + prediction, "tva_leaf_value"
    if target_mode == "direct_value":
        if float(args.tva_value_blend) >= 1.0:
            return prediction, prediction, "tva_prediction"
        leaf = (1.0 - float(args.tva_value_blend)) * compute_gft_sd_leaf_value(
            model, state, args
        ) + float(args.tva_value_blend) * prediction
        return prediction, leaf, "tva_leaf_value"
    raise ValueError(f"Unsupported GFT-TVA target_mode: {checkpoint.target_mode}")


def collect_leaf_samples(args: argparse.Namespace) -> tuple[list[LeafSample], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    recorder = LeafRecorder(max_per_level=args.max_leaf_states_per_level)
    t0 = perf_counter()
    planner_action_prior = "sd_delta" if args.gft_prior == "gft_delta" else "none"
    level_counts: Counter[str] = Counter()
    for level_index, level in enumerate(levels):
        model = gft_model_from_level(level, args)
        rng = random.Random(args.seed + level_index * 1009)
        state = model.initial_state
        visited_keys = {model.gft_canonical_key_cached(state)}
        for root_step in range(args.closed_loop_steps):
            if model.is_solved(state):
                break
            planner = AuditedGFTSDMCTSPlanner(
                model,
                horizon=args.horizon,
                budget=args.budget,
                rng=rng,
                uct_c=args.uct_c,
                gamma=args.gamma,
                goal_value=args.goal_value,
                push_cost=args.push_cost,
                deadend_value=args.deadend_value,
                leaf_value=args.leaf_value_for_sampling,
                tva_checkpoint=None,
                constraint_mode=args.constraint_mode,
                select_by=args.select_by,
                revisit_penalty=args.revisit_penalty,
                visited_keys=visited_keys,
                transposition=args.transposition,
                action_prior=planner_action_prior,
                prior_temperature=1.0,
                prior_uniform_mix=0.1,
                prior_progress_weight=1.0,
                prior_goal_weight=1.0,
                prior_mobility_weight=0.0,
                prior_revisit_weight=1.0,
                gft_prior=args.gft_prior,
                gft_prior_temperature=args.gft_prior_temperature,
                gft_prior_uniform_mix=args.gft_prior_uniform_mix,
                gft_prior_weights=gft_weights_from_args(args),
                gft_sd_leaf_mode=args.gft_sd_leaf_mode,
                gft_sd_leaf_weights=gft_sd_leaf_weights_from_args(args),
                leaf_recorder=recorder,
                audit_level_index=level_index,
                audit_level_id=level.level_id,
                audit_root_step=root_step,
            )
            plan = planner.plan(state)
            action = plan["action"]
            if action is None:
                break
            state = model.step_grid_flow(state, action)
            visited_keys.add(model.gft_canonical_key_cached(state))
            if model.is_deadlocked(state) and not model.is_solved(state):
                break
        level_counts[level.level_id] = sum(1 for sample in recorder.samples if sample.level_id == level.level_id)
        print(
            json.dumps(
                {
                    "level_index": int(level_index),
                    "level_id": level.level_id,
                    "leaf_samples": int(level_counts[level.level_id]),
                },
                ensure_ascii=False,
            )
        )
    return recorder.samples, {
        "levels": [level.level_id for level in levels],
        "leaf_samples_per_level": dict(level_counts),
        "leaf_collection_elapsed_sec": float(perf_counter() - t0),
    }


def label_and_score(samples: Sequence[LeafSample], checkpoint: GFTTvaCheckpoint, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher_cache: dict[tuple[str, SokobanState], Any] = {}
    rows: list[dict[str, Any]] = []
    teacher_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    t0 = perf_counter()
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    model_by_level = {level.level_id: gft_model_from_level(level, args) for level in levels}

    for idx, sample in enumerate(samples):
        model = model_by_level[sample.level_id]
        key = (sample.level_id, sample.state)
        if key not in teacher_cache:
            teacher_cache[key] = best_first_from_state(
                model,
                sample.state,
                max_expansions=args.teacher_max_expansions,
            )
        teacher = teacher_cache[key]
        target = target_from_teacher(
            teacher,
            gamma=args.gamma,
            goal_value=args.goal_value,
            push_cost=args.push_cost,
            deadend_value=args.deadend_value,
            unsolved_value=args.unsolved_value,
        )
        confidence = (
            "high_confidence"
            if teacher.failure_type in {"solved", "proven_dead_end", "no_legal_actions"}
            else "budget_unsolved_low_confidence"
        )
        tva_prediction, tva_leaf_value, audit_prediction_key = compute_tva_values(checkpoint, model, sample.state, args)
        gft_sd_leaf_value = compute_gft_sd_leaf_value(model, sample.state, args)
        sd_heuristic_value = compute_sd_heuristic_value(model, sample.state, args)
        row = {
            "sample_index": int(idx),
            "level_index": int(sample.level_index),
            "level_id": sample.level_id,
            "root_step": int(sample.root_step),
            "state_type": sample.state_type,
            "planner_blocked": bool(sample.planner_blocked),
            "player": list(sample.state.player),
            "boxes": [list(box) for box in sample.state.boxes],
            "target_value": float(target),
            "teacher_confidence": confidence,
            "teacher_solved": bool(teacher.solved),
            "teacher_solution_pushes": int(teacher.solution_pushes) if teacher.solution_pushes is not None else None,
            "teacher_failure_type": str(teacher.failure_type),
            "teacher_expanded": int(teacher.expanded),
            "teacher_generated": int(teacher.generated),
            "tva_prediction": float(tva_prediction),
            "tva_leaf_value": float(tva_leaf_value),
            "gft_sd_leaf_value": float(gft_sd_leaf_value),
            "sd_heuristic_value": float(sd_heuristic_value),
            "audit_prediction_key": audit_prediction_key,
        }
        rows.append(row)
        teacher_counts[str(teacher.failure_type)] += 1
        confidence_counts[confidence] += 1
    return rows, {
        "teacher_counts": dict(teacher_counts),
        "confidence_counts": dict(confidence_counts),
        "label_elapsed_sec": float(perf_counter() - t0),
    }


def grouped_metrics(rows: Sequence[dict[str, Any]], pred_keys: Sequence[str]) -> dict[str, Any]:
    groups = {
        "all": list(rows),
        "high_confidence": [row for row in rows if row["teacher_confidence"] == "high_confidence"],
        "solved": [row for row in rows if row["teacher_failure_type"] == "solved"],
        "dead_or_no_legal": [
            row for row in rows if row["teacher_failure_type"] in {"proven_dead_end", "no_legal_actions"}
        ],
        "budget_unsolved": [row for row in rows if row["teacher_failure_type"] == "teacher_budget_unsolved"],
        "positive_high_confidence": [
            row for row in rows if row["teacher_confidence"] == "high_confidence" and row["target_value"] > 0.0
        ],
        "nonpositive_high_confidence": [
            row for row in rows if row["teacher_confidence"] == "high_confidence" and row["target_value"] <= 0.0
        ],
    }
    out: dict[str, Any] = {}
    for pred_key in pred_keys:
        out[pred_key] = {
            name: regression_metrics(group, pred_key)
            for name, group in groups.items()
        }
        out[pred_key]["pairwise_high_confidence_by_root"] = pairwise_order_accuracy(
            groups["high_confidence"],
            pred_key,
            tolerance=1e-9,
        )
    return out


def build_gate(metrics: dict[str, Any], audit_key: str) -> dict[str, Any]:
    high = metrics[audit_key]["high_confidence"]
    gft_high = metrics["gft_sd_leaf_value"]["high_confidence"]
    pair = metrics[audit_key]["pairwise_high_confidence_by_root"]
    gft_pair = metrics["gft_sd_leaf_value"]["pairwise_high_confidence_by_root"]
    positive = metrics[audit_key]["positive_high_confidence"]
    nonpositive = metrics[audit_key]["nonpositive_high_confidence"]
    failed: list[str] = []
    if high["samples"] < 20:
        failed.append("high_confidence_samples_min_20")
    if high["mae"] > gft_high["mae"]:
        failed.append("mae_not_better_than_gft_sd_leaf")
    if high["corr"] < 0.30:
        failed.append("corr_min_0.30")
    if pair["pairs"] >= 10 and pair["accuracy"] < max(0.50, gft_pair["accuracy"]):
        failed.append("pairwise_order_not_better_than_gft_sd_leaf")
    if positive["samples"] >= 3 and positive["bias"] < -1.0:
        failed.append("positive_value_underprediction_bias")
    if nonpositive["samples"] >= 3 and nonpositive["bias"] > 1.0:
        failed.append("nonpositive_value_overprediction_bias")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "thresholds": {
            "high_confidence_samples_min": 20,
            "mae_must_not_exceed_gft_sd_leaf": True,
            "corr_min": 0.30,
            "pairwise_order_accuracy_min": "max(0.50, gft_sd_leaf_pairwise_accuracy) when at least 10 pairs exist",
            "positive_underprediction_bias_min": -1.0,
            "nonpositive_overprediction_bias_max": 1.0,
        },
        "observed": {
            "audit_prediction_key": audit_key,
            "tva_high_confidence": high,
            "gft_sd_leaf_high_confidence": gft_high,
            "tva_pairwise_high_confidence": pair,
            "gft_sd_leaf_pairwise_high_confidence": gft_pair,
            "positive_high_confidence": positive,
            "nonpositive_high_confidence": nonpositive,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.level_offset < 0:
        raise SystemExit("--level-offset must be non-negative.")
    if args.max_leaf_states_per_level <= 0:
        raise SystemExit("--max-leaf-states-per-level must be positive.")
    if not 0.0 <= float(args.tva_value_blend) <= 1.0:
        raise SystemExit("--tva-value-blend must be between 0 and 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = load_tva_checkpoint(args.tva_checkpoint)
    if not isinstance(checkpoint, GFTTvaCheckpoint):
        raise SystemExit("This audit expects a GFT-TVA checkpoint with gft_tva_feature_vector_v1.")

    samples, collection = collect_leaf_samples(args)
    rows, labeling = label_and_score(samples, checkpoint, args)
    audit_key = rows[0]["audit_prediction_key"] if rows else "tva_prediction"
    pred_keys = ["tva_prediction", "tva_leaf_value", "gft_sd_leaf_value", "sd_heuristic_value"]
    metrics = grouped_metrics(rows, pred_keys)
    gate = build_gate(metrics, audit_key)
    summary = {
        "material_passport": {
            "origin": "sokoban_gft_tva_truncation_value_audit",
            "verification_status": "ANALYZED",
            "version_label": "gft_tva_independent_accuracy_gate_v1",
        },
        "config": {
            **vars(args),
            "levels": _rel(args.levels),
            "tva_checkpoint": _rel(args.tva_checkpoint),
        },
        "audit_contract": {
            "purpose": "independently evaluate GFT-TVA as a truncated leaf-value estimator before closed-loop use",
            "leaf_state_source": "GFT-SD-MCTS sampled with explicit SD leaf value; TVA is not used for sampling decisions",
            "label_source": "offline exact-state best-first teacher from sampled leaf state",
            "teacher_used_online": False,
            "tva_used_for_action_choice": False,
            "transition_provider_modified": False,
            "high_confidence_labels": ["solved", "proven_dead_end", "no_legal_actions"],
            "budget_unsolved_labels": "reported separately and excluded from high-confidence gate",
        },
        "checkpoint_contract": {
            "value_provider_id": checkpoint.value_provider_id,
            "target_mode": checkpoint.target_mode,
            "path": checkpoint.path,
            "checkpoint_contract": checkpoint.checkpoint_contract,
        },
        "collection": collection,
        "labeling": labeling,
        "metrics": metrics,
        "gft_tva_independent_accuracy_gate": gate,
        "rows": rows,
        "sample_rows": rows[:80],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "samples": len(rows),
                "gate": gate,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
