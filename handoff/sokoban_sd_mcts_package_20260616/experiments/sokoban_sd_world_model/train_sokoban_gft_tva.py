#!/usr/bin/env python3
"""Train GFT-SD TVA values from GFT-MCTS root statistics.

This script implements the strict TVA training semantics for the current
two-layer Sokoban GFT-SD stack.  Targets are constructed from MCTS root visit
counts and root child Q estimates.  Diagnostic teachers are not used online or
for the training target.  The default learned term is the direct terminal value
``V_psi(s_leaf)`` consumed at MCTS leaves:

    V_leaf(s_leaf) = V_psi(s_leaf),  V_psi target = V_search

The older ``gft_sd_residual`` target remains available as an explicit negative
engineering variant:

    V_leaf(z) = V_SD(z) + V_TVA(z),  V_TVA target = V_search - V_SD
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
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
    action_to_dict,
    gft_soft_deadlock_weights_from_args,
    gft_transport_weights_from_args,
    gft_weights_from_args,
    gft_sd_leaf_weights_from_args,
    discounted_progress_gft_sd_value,
    projected_gft_sd_value,
)
from src.sokoban import (  # noqa: E402
    DEFAULT_GFT_SOFT_DEADLOCK_WEIGHTS,
    DEFAULT_GFT_TRANSPORT_COST_WEIGHTS,
    SokobanGFTSDModel,
    SokobanState,
    load_boxoban_levels,
)


def normalize_target_mode(mode: str) -> str:
    """Canonicalize historical target-mode names used by old Sokoban TVA runs."""

    if mode == "value":
        return "direct_value"
    return str(mode)


@dataclass
class GFTSearchValueSample:
    features: np.ndarray
    target_value: float
    v_search: float
    v_sd: float
    split: str
    level_id: str
    state_depth: int
    root_policy: list[dict[str, Any]]
    root_children: int
    root_visits: int
    root_value: float
    root_state_type: str

    def summary_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("features")
        return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used to collect GFT-TVA states.",
    )
    parser.add_argument("--level-offset", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=40)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--state-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=24)
    parser.add_argument("--collect-horizon", type=int, default=6)
    parser.add_argument("--collect-budget", type=int, default=96)
    parser.add_argument("--uct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--constraint-mode", choices=("none", "tier_a", "exact_deadlock"), default="tier_a")
    parser.add_argument("--select-by", choices=("visits", "q"), default="visits")
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--transposition", action="store_true", default=True)
    parser.add_argument("--no-transposition", action="store_false", dest="transposition")
    parser.add_argument("--bootstrap-leaf-value", choices=("gft_sd", "sd_heuristic", "zero"), default="gft_sd")
    parser.add_argument(
        "--gft-sd-leaf-mode",
        choices=("linear_projected", "discounted_progress"),
        default="linear_projected",
        help="How explicit GFT V_SD is mapped to the MCTS leaf-value scale.",
    )
    parser.add_argument("--gft-sd-leaf-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-capacity-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-ordering-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-commitment-weight", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--target-mode",
        choices=("gft_sd_residual", "direct_value", "value"),
        default="direct_value",
        help="'direct_value' trains V_psi directly to V_search; 'value' is a backward-compatible alias.",
    )
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-train-samples", type=int, default=8)
    parser.add_argument("--min-valid-samples", type=int, default=2)
    parser.add_argument("--valid-corr-min", type=float, default=-1.0)
    parser.add_argument("--value-provider-id", default="sokoban_gft_search_value_tva_direct_v1")
    parser.add_argument("--gft-prior", choices=("none", "gft_delta"), default="gft_delta")
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
    parser.add_argument("--gft-transport-turn-weight", type=float, default=0.0)
    parser.add_argument("--gft-transport-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-transport-zone-weight", type=float, default=0.0)
    parser.add_argument("--gft-soft-deadlock-mobility-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-matching-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-capacity-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-access-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-order-weight", type=float, default=1.0)
    parser.add_argument("--gft-commitment-beta", type=float, default=1.0)
    parser.add_argument("--gft-sinkhorn-iterations", type=int, default=25)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_gft_tva_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_gft_tva_seed0.json",
    )
    return parser


def _rel(path: str | Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def root_state_type(model: SokobanGFTSDModel, state: SokobanState) -> str:
    if model.is_solved(state):
        return "solved"
    if model.is_deadlocked(state):
        return "deadlocked"
    if not model.legal_flows(state):
        return "no_legal_actions"
    return "active"


def iter_gft_states(
    model: SokobanGFTSDModel,
    *,
    max_depth: int,
    max_states: int,
) -> Sequence[tuple[SokobanState, int]]:
    queue: deque[tuple[SokobanState, int]] = deque([(model.initial_state, 0)])
    seen = {model.gft_canonical_key_cached(model.initial_state)}
    out: list[tuple[SokobanState, int]] = []
    while queue and len(out) < max_states:
        state, depth = queue.popleft()
        out.append((state, depth))
        if depth >= max_depth:
            continue
        actions = sorted(model.legal_flows(state), key=lambda action: action.flat_index)
        for action in actions:
            child = model.step_grid_flow(state, action)
            key = model.gft_canonical_key_cached(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, depth + 1))
            if len(seen) >= max_states * 4:
                break
    return out


def compute_search_value(root_policy: Sequence[dict[str, Any]], alpha: float) -> tuple[float, list[float]]:
    visits = np.asarray([max(float(row["visits"]), 0.0) for row in root_policy], dtype=np.float64)
    q_values = np.asarray([float(row["q_value"]) for row in root_policy], dtype=np.float64)
    if not len(visits):
        raise ValueError("Cannot compute V_search without root children.")
    if not np.isfinite(q_values).all():
        raise ValueError("Root policy contains non-finite q_value.")
    weights = np.ones_like(visits) if alpha == 0.0 else np.power(visits, float(alpha))
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(visits)
    probs = weights / weights.sum()
    return float(np.dot(probs, q_values)), [float(value) for value in probs.tolist()]


def feature_row(model: SokobanGFTSDModel, state: SokobanState, feature_names: Sequence[str]) -> np.ndarray:
    features = model.gft_tva_feature_dict(state)["features"]
    return np.asarray([float(features[name]) for name in feature_names], dtype=np.float64)


def make_model(level, args: argparse.Namespace) -> SokobanGFTSDModel:
    return SokobanGFTSDModel(
        level,
        transport_cost_weights=gft_transport_weights_from_args(args),
        soft_deadlock_weights=gft_soft_deadlock_weights_from_args(args),
        commitment_beta=args.gft_commitment_beta,
        sinkhorn_iterations=args.gft_sinkhorn_iterations,
    )


def collect_root_policy(
    model: SokobanGFTSDModel,
    state: SokobanState,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    planner_action_prior = "sd_delta" if args.gft_prior == "gft_delta" else "none"
    planner = GFTSDMCTSPlanner(
        model,
        horizon=args.collect_horizon,
        budget=args.collect_budget,
        rng=rng,
        uct_c=args.uct_c,
        gamma=args.gamma,
        goal_value=args.goal_value,
        push_cost=args.push_cost,
        deadend_value=args.deadend_value,
        leaf_value=args.bootstrap_leaf_value,
        tva_checkpoint=None,
        constraint_mode=args.constraint_mode,
        select_by=args.select_by,
        revisit_penalty=args.revisit_penalty,
        visited_keys={model.gft_canonical_key_cached(state)},
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
    )
    plan = planner.plan(state)
    root_policy = []
    for row in plan["children"]:
        copied = dict(row)
        copied["q_value"] = float(copied["mean_value"])
        root_policy.append(copied)
    plan["root_policy"] = root_policy
    return plan


def split_levels(level_ids: Sequence[str], valid_fraction: float, seed: int) -> dict[str, Any]:
    ids = list(level_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    valid_count = max(1, int(round(len(ids) * valid_fraction))) if len(ids) > 1 else 0
    valid = set(ids[:valid_count])
    train = [level_id for level_id in ids if level_id not in valid]
    return {
        "split_mode": "level_random",
        "train_levels": sorted(train),
        "valid_levels": sorted(valid),
        "valid_fraction": float(valid_fraction),
        "seed": int(seed),
    }


def collect_samples(args: argparse.Namespace) -> tuple[list[GFTSearchValueSample], dict[str, Any], list[str]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    split = split_levels([level.level_id for level in levels], args.valid_fraction, args.seed)
    train_ids = set(split["train_levels"])
    valid_ids = set(split["valid_levels"])
    samples: list[GFTSearchValueSample] = []
    skipped: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    root_child_counts: list[int] = []
    root_state_counts: Counter[str] = Counter()
    feature_names: list[str] | None = None
    t0 = perf_counter()

    for level_index, level in enumerate(levels):
        split_name = "train" if level.level_id in train_ids else "valid" if level.level_id in valid_ids else "skip"
        if split_name == "skip":
            continue
        model = make_model(level, args)
        if feature_names is None:
            feature_names = list(model.gft_tva_feature_dict(model.initial_state)["feature_names"])
        rng = random.Random(args.seed + level_index * 1009)
        level_samples = 0
        for state, state_depth in iter_gft_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            state_type = root_state_type(model, state)
            if state_type != "active":
                skipped[state_type] += 1
                continue
            plan = collect_root_policy(model, state, args, rng)
            if not plan["root_policy"]:
                skipped["empty_root_policy"] += 1
                continue
            v_search, probs = compute_search_value(plan["root_policy"], args.alpha)
            if args.gft_sd_leaf_mode == "discounted_progress":
                v_sd = discounted_progress_gft_sd_value(
                    model,
                    state,
                    gamma=args.gamma,
                    goal_value=args.goal_value,
                    push_cost=args.push_cost,
                    deadend_value=args.deadend_value,
                    weights=gft_sd_leaf_weights_from_args(args),
                )
            else:
                v_sd = projected_gft_sd_value(
                    model,
                    state,
                    deadend_value=args.deadend_value,
                    goal_value=args.goal_value,
                    weights=gft_sd_leaf_weights_from_args(args),
                )
            if not math.isfinite(v_search) or not math.isfinite(v_sd):
                skipped["nonfinite_target_or_vsd"] += 1
                continue
            if args.target_mode == "gft_sd_residual":
                target = float(v_search - v_sd)
            elif args.target_mode == "direct_value":
                target = float(v_search)
            else:
                raise ValueError(f"Unsupported target_mode: {args.target_mode}")
            root_policy = []
            for row, prob in zip(plan["root_policy"], probs):
                copied = dict(row)
                copied["search_probability"] = float(prob)
                root_policy.append(copied)
            samples.append(
                GFTSearchValueSample(
                    features=feature_row(model, state, feature_names),
                    target_value=target,
                    v_search=float(v_search),
                    v_sd=float(v_sd),
                    split=split_name,
                    level_id=level.level_id,
                    state_depth=int(state_depth),
                    root_policy=root_policy,
                    root_children=int(plan["child_count"]),
                    root_visits=int(plan["root_visits"]),
                    root_value=float(plan["root_value"]),
                    root_state_type=state_type,
                )
            )
            split_counts[split_name] += 1
            root_child_counts.append(int(plan["child_count"]))
            root_state_counts[state_type] += 1
            level_samples += 1
        print(
            json.dumps(
                {
                    "level_index": level_index,
                    "level_id": level.level_id,
                    "split": split_name,
                    "samples": level_samples,
                },
                ensure_ascii=False,
            )
        )

    if feature_names is None:
        raise SystemExit("Could not infer GFT-TVA feature names.")
    return samples, {
        "level_split": split,
        "split_counts": dict(split_counts),
        "skipped_counts": dict(skipped),
        "root_state_counts": dict(root_state_counts),
        "root_child_summary": {
            "mean": float(np.mean(root_child_counts)) if root_child_counts else 0.0,
            "min": int(min(root_child_counts)) if root_child_counts else 0,
            "max": int(max(root_child_counts)) if root_child_counts else 0,
        },
        "collection_elapsed_sec": float(perf_counter() - t0),
    }, feature_names


def indices_for_split(samples: Sequence[GFTSearchValueSample], split: str) -> np.ndarray:
    return np.asarray([idx for idx, sample in enumerate(samples) if sample.split == split], dtype=np.int64)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {}
    err = pred - target
    corr = 0.0
    if len(target) >= 2 and float(np.std(pred)) > 1e-12 and float(np.std(target)) > 1e-12:
        corr = float(np.corrcoef(pred, target)[0, 1])
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": corr,
        "target_mean": float(np.mean(target)),
        "pred_mean": float(np.mean(pred)),
        "samples": float(len(target)),
    }


def feature_summary(features: np.ndarray, feature_names: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean(features[:, idx])),
            "std": float(np.std(features[:, idx])),
            "min": float(np.min(features[:, idx])),
            "max": float(np.max(features[:, idx])),
        }
        for idx, name in enumerate(feature_names)
    }


def fit_ridge(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, l2: float) -> dict[str, Any]:
    mean = np.mean(features[train_idx], axis=0)
    std = np.maximum(np.std(features[train_idx], axis=0), 1e-6)
    z = (features - mean) / std
    design = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    train_design = design[train_idx]
    lhs = train_design.T @ train_design
    penalty = float(l2) * np.eye(lhs.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = train_design.T @ targets[train_idx]
    weights = np.linalg.solve(lhs + penalty, rhs)
    pred = design @ weights
    return {
        "mean": mean,
        "std": std,
        "weights": weights,
        "pred": pred,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.target_mode = normalize_target_mode(args.target_mode)
    set_seed(args.seed)
    samples, collection, feature_names = collect_samples(args)
    if not samples:
        raise SystemExit("GFT-TVA collection produced no samples.")

    features = np.stack([sample.features for sample in samples]).astype(np.float64)
    targets = np.asarray([sample.target_value for sample in samples], dtype=np.float64)
    train_idx = indices_for_split(samples, "train")
    valid_idx = indices_for_split(samples, "valid")
    ridge = fit_ridge(features, targets, train_idx, args.ridge_l2)
    pred = np.asarray(ridge["pred"], dtype=np.float64)
    metrics = {
        "train": regression_metrics(pred[train_idx], targets[train_idx]),
        "valid": regression_metrics(pred[valid_idx], targets[valid_idx]) if len(valid_idx) else {},
    }
    failed = []
    if len(train_idx) < args.min_train_samples:
        failed.append("min_train_samples")
    if len(valid_idx) < args.min_valid_samples:
        failed.append("min_valid_samples")
    valid_corr = metrics["valid"].get("corr", 0.0) if metrics["valid"] else 0.0
    if len(valid_idx) and valid_corr < args.valid_corr_min:
        failed.append("valid_corr_min")
    if not np.isfinite(features).all() or not np.isfinite(targets).all() or not np.isfinite(pred).all():
        failed.append("nonfinite_features_targets_or_predictions")

    summary = {
        "material_passport": {
            "origin": "sokoban_gft_tva",
            "verification_status": "ANALYZED",
            "version_label": args.value_provider_id,
        },
        "config": {
            "levels": _rel(args.levels),
            "level_offset": args.level_offset,
            "max_levels": args.max_levels,
            "valid_fraction": args.valid_fraction,
            "state_depth": args.state_depth,
            "max_states_per_level": args.max_states_per_level,
            "collect_horizon": args.collect_horizon,
            "collect_budget": args.collect_budget,
            "uct_c": args.uct_c,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "constraint_mode": args.constraint_mode,
            "select_by": args.select_by,
            "revisit_penalty": args.revisit_penalty,
            "transposition": bool(args.transposition),
            "bootstrap_leaf_value": args.bootstrap_leaf_value,
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
            "alpha": args.alpha,
            "target_mode": args.target_mode,
            "ridge_l2": args.ridge_l2,
            "seed": args.seed,
            "gft_prior": args.gft_prior,
            "gft_prior_temperature": args.gft_prior_temperature,
            "gft_prior_uniform_mix": args.gft_prior_uniform_mix,
            "gft_prior_weights": gft_weights_from_args(args),
            "gft_transport_cost_weights": gft_transport_weights_from_args(args),
            "gft_soft_deadlock_weights": gft_soft_deadlock_weights_from_args(args),
            "gft_commitment_beta": args.gft_commitment_beta,
            "gft_sinkhorn_iterations": args.gft_sinkhorn_iterations,
            "value_provider_id": args.value_provider_id,
        },
        "tva_contract": {
            "value_provider_id": args.value_provider_id,
            "leaf_evaluation_only": True,
            "target_source": "gft_sdmcts_root_statistics",
            "target_formula": "V_search=sum_a visit_alpha_probability(a)*Q_H(s,a)",
            "direct_value_formula": "V_psi=V_search" if args.target_mode == "direct_value" else "disabled",
            "residual_formula": "V_TVA=V_search-V_SD" if args.target_mode == "gft_sd_residual" else "disabled",
            "leaf_formula": (
                "V_leaf=V_SD+V_TVA"
                if args.target_mode == "gft_sd_residual"
                else "V_leaf=V_psi"
            ),
            "paper_alignment": (
                "direct_terminal_value_tva"
                if args.target_mode == "direct_value"
                else "residual_engineering_variant"
            ),
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
            "mcts_root_statistics_used": True,
            "teacher_used_online": False,
            "teacher_used_for_training_target": False,
            "transition_provider_modified": False,
            "sd_provider": SokobanGFTSDModel.provider_id,
            "transition_provider": SokobanGFTSDModel.transition_provider_id,
            "feature_contract": "gft_tva_feature_vector_v1",
            "feature_names": list(feature_names),
        },
        "gate": {
            "tva_independent_gate": "PASS" if not failed else "FAIL",
            "failed_checks": failed,
            "thresholds": {
                "min_train_samples": args.min_train_samples,
                "min_valid_samples": args.min_valid_samples,
                "valid_corr_min": args.valid_corr_min,
            },
        },
        "dataset": {
            "samples": int(len(samples)),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "feature_dim": int(features.shape[1]),
            "target_mode": args.target_mode,
            "target_stats_train": {
                "mean": float(np.mean(targets[train_idx])),
                "std": float(np.std(targets[train_idx])),
                "min": float(np.min(targets[train_idx])),
                "max": float(np.max(targets[train_idx])),
            },
            "collection": collection,
            "feature_summary": feature_summary(features, feature_names),
        },
        "metrics": metrics,
        "sample_preview": [sample.summary_dict() for sample in samples[:5]],
        "elapsed_sec": float(collection["collection_elapsed_sec"]),
    }

    checkpoint = {
        "value_provider_id": args.value_provider_id,
        "model_type": "ridge",
        "target_mode": args.target_mode,
        "feature_contract": {
            "contract": "gft_tva_feature_vector_v1",
            "feature_names": list(feature_names),
            "feature_dim": int(features.shape[1]),
        },
        "value_scale": {
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
        },
        "checkpoint_contract": summary["tva_contract"],
        "ridge": {
            "mean": ridge["mean"].tolist(),
            "std": ridge["std"].tolist(),
            "weights": ridge["weights"].tolist(),
            "l2": float(args.ridge_l2),
        },
        "summary": summary,
    }
    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_out.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["checkpoint"] = {
        "saved": True,
        "path": _rel(checkpoint_out),
        "value_provider_id": args.value_provider_id,
    }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.target_mode = normalize_target_mode(args.target_mode)
    summary = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gate": summary["gate"], "metrics": summary["metrics"], "checkpoint": summary["checkpoint"]}, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
