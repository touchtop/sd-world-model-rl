#!/usr/bin/env python3
"""Train GFT-TVA from closed-loop online SD-MCTS trajectories.

This script samples the states actually visited by the current GFT-SD-MCTS
controller, then trains the same guarded MLP checkpoint format used by the
leaf-supervised TVA line.  Successful trajectories provide high-confidence
value labels, deadlocked/no-action trajectories provide high-confidence
negative labels, and max-step trajectories are treated as censored risk by
default rather than as deterministic negative value labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
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

from audit_sokoban_gft_tva_truncation_value import (  # noqa: E402
    compute_gft_sd_leaf_value,
    compute_sd_heuristic_value,
)
from evaluate_sokoban_gft_sdmcts import (  # noqa: E402
    GFTSDMCTSPlanner,
    GFTTvaCheckpoint,
    gft_model_from_level,
    gft_sd_leaf_weights_from_args,
    gft_soft_deadlock_weights_from_args,
    gft_transport_weights_from_args,
    gft_weights_from_args,
    load_tva_checkpoint,
)
from evaluate_sokoban_grid_sdmcts import TvaCheckpoint, _rel  # noqa: E402
from src.sokoban import SokobanGFTSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from train_sokoban_gft_leaf_tva import (  # noqa: E402
    LeafTvaTrainingSample,
    export_mlp,
    feature_summary,
    indices_for_split,
    set_seed,
    split_levels,
    train_network,
)


@dataclass
class OnlineStateSample:
    level_index: int
    level_id: str
    state: SokobanState
    source: str
    root_step: int
    remaining_pushes: int
    trajectory_status: str
    state_type: str
    terminal_value_source: str

    def key_tuple(self) -> tuple[str, tuple, str]:
        # Status is part of the key so the same grid state may be retained when
        # it appears once as an active root and later as an explicit terminal.
        return self.level_id, self.state, self.terminal_value_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used to collect online trajectories.",
    )
    parser.add_argument("--level-offset", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=20)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--uct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--goal-value", type=float, default=10.0)
    parser.add_argument("--push-cost", type=float, default=0.05)
    parser.add_argument("--deadend-value", type=float, default=-10.0)
    parser.add_argument("--max-steps-value", type=float, default=-2.5)
    parser.add_argument("--no-action-value", type=float, default=-10.0)
    parser.add_argument("--constraint-mode", choices=("none", "tier_a", "exact_deadlock"), default="tier_a")
    parser.add_argument("--select-by", choices=("visits", "q"), default="visits")
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--transposition", action="store_true", default=True)
    parser.add_argument("--no-transposition", action="store_false", dest="transposition")
    parser.add_argument("--reuse-tree", action="store_true", default=False)
    parser.add_argument("--sampling-leaf-value", choices=("gft_sd", "sd_heuristic", "zero", "tva"), default="gft_sd")
    parser.add_argument("--tva-checkpoint", default="")
    parser.add_argument("--tva-value-blend", type=float, default=0.25)
    parser.add_argument("--action-prior", choices=("none", "sd_delta"), default="none")
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.1)
    parser.add_argument("--prior-progress-weight", type=float, default=1.0)
    parser.add_argument("--prior-goal-weight", type=float, default=1.0)
    parser.add_argument("--prior-mobility-weight", type=float, default=0.0)
    parser.add_argument("--prior-revisit-weight", type=float, default=1.0)
    parser.add_argument("--gft-prior", choices=("none", "gft_delta"), default="gft_delta")
    parser.add_argument("--gft-prior-temperature", type=float, default=2.0)
    parser.add_argument("--gft-prior-uniform-mix", type=float, default=0.1)
    parser.add_argument("--gft-progress-weight", type=float, default=1.0)
    parser.add_argument("--gft-goal-weight", type=float, default=1.0)
    parser.add_argument("--gft-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-capacity-weight", type=float, default=0.0)
    parser.add_argument("--gft-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-ordering-weight", type=float, default=0.05)
    parser.add_argument("--gft-commitment-weight", type=float, default=0.05)
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
    parser.add_argument("--solved-value-weight", type=float, default=1.0)
    parser.add_argument("--deadend-value-weight", type=float, default=1.5)
    parser.add_argument("--max-steps-value-weight", type=float, default=0.0)
    parser.add_argument(
        "--failure-value-label-scope",
        choices=("final", "all"),
        default="final",
        help=(
            "Which states from deadlocked/no-action trajectories receive negative value labels. "
            "Intermediate states still receive risk labels."
        ),
    )
    parser.add_argument("--solved-risk-weight", type=float, default=1.0)
    parser.add_argument("--censored-risk-weight", type=float, default=1.0)
    parser.add_argument("--deadend-risk-weight", type=float, default=1.5)
    parser.add_argument("--risk-loss-weight", type=float, default=0.25)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=0.75)
    parser.add_argument("--anchor-bias-mode", choices=("none", "train_mean", "train_median"), default="train_median")
    parser.add_argument("--anchor-bias-labels", choices=("solved", "all_value"), default="solved")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-value-train-samples", type=int, default=20)
    parser.add_argument("--min-risk-classes", type=int, default=2)
    parser.add_argument("--deduplicate-states", action="store_true", default=True)
    parser.add_argument("--no-deduplicate-states", action="store_false", dest="deduplicate_states")
    parser.add_argument("--collect-final-states", action="store_true", default=True)
    parser.add_argument("--no-collect-final-states", action="store_false", dest="collect_final_states")
    parser.add_argument("--value-provider-id", default="sokoban_gft_online_tva_guarded_v1")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_seed0.json",
    )
    return parser


def load_selected_levels(args: argparse.Namespace):
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")
    return levels


def state_type(model: SokobanGFTSDModel, state: SokobanState) -> str:
    if model.is_solved(state):
        return "solved"
    if model.is_deadlocked(state):
        return "deadlocked"
    if not model.legal_flows(state):
        return "no_legal_actions"
    return "active"


def discounted_terminal_value(remaining_pushes: int, terminal_value: float, args: argparse.Namespace) -> float:
    remaining = max(int(remaining_pushes), 0)
    gamma = float(args.gamma)
    if gamma == 1.0:
        return float(terminal_value) - float(args.push_cost) * remaining
    discount = gamma**remaining
    push_cost_sum = float(args.push_cost) * (1.0 - discount) / (1.0 - gamma)
    return float(discount * float(terminal_value) - push_cost_sum)


def terminal_value_for_status(status: str, args: argparse.Namespace) -> float:
    if status == "solved":
        return float(args.goal_value)
    if status == "max_steps":
        return float(args.max_steps_value)
    if status == "no_action":
        return float(args.no_action_value)
    return float(args.deadend_value)


def make_planner(
    model: SokobanGFTSDModel,
    args: argparse.Namespace,
    rng: random.Random,
    visited_keys: set,
    tva_checkpoint: TvaCheckpoint | GFTTvaCheckpoint | None,
) -> GFTSDMCTSPlanner:
    planner_action_prior = "sd_delta" if args.gft_prior == "gft_delta" else args.action_prior
    return GFTSDMCTSPlanner(
        model,
        horizon=args.horizon,
        budget=args.budget,
        rng=rng,
        uct_c=args.uct_c,
        gamma=args.gamma,
        goal_value=args.goal_value,
        push_cost=args.push_cost,
        deadend_value=args.deadend_value,
        leaf_value=args.sampling_leaf_value,
        tva_checkpoint=tva_checkpoint,
        constraint_mode=args.constraint_mode,
        select_by=args.select_by,
        revisit_penalty=args.revisit_penalty,
        visited_keys=visited_keys,
        transposition=args.transposition or args.reuse_tree,
        action_prior=planner_action_prior,
        prior_temperature=args.prior_temperature,
        prior_uniform_mix=args.prior_uniform_mix,
        prior_progress_weight=args.prior_progress_weight,
        prior_goal_weight=args.prior_goal_weight,
        prior_mobility_weight=args.prior_mobility_weight,
        prior_revisit_weight=args.prior_revisit_weight,
        gft_prior=args.gft_prior,
        gft_prior_temperature=args.gft_prior_temperature,
        gft_prior_uniform_mix=args.gft_prior_uniform_mix,
        gft_prior_weights=gft_weights_from_args(args),
        gft_sd_leaf_mode=args.gft_sd_leaf_mode,
        gft_sd_leaf_weights=gft_sd_leaf_weights_from_args(args),
        tva_value_blend=args.tva_value_blend,
    )


def collect_level_trajectory(
    level_index: int,
    level,
    args: argparse.Namespace,
    tva_checkpoint: TvaCheckpoint | GFTTvaCheckpoint | None,
) -> tuple[list[OnlineStateSample], dict[str, Any]]:
    model = gft_model_from_level(level, args)
    rng = random.Random(int(args.seed) + int(level_index) * 1009)
    state = model.initial_state
    visited_keys = {model.canonical_key(state)}
    root_states: list[tuple[int, SokobanState, str]] = []
    total_deadlock_prunes = 0
    status = "max_steps"
    persistent_planner: GFTSDMCTSPlanner | None = None
    pushes_executed = 0
    t0 = perf_counter()

    for step in range(int(args.max_steps) + 1):
        if model.is_solved(state):
            status = "solved"
            break
        if step >= int(args.max_steps):
            status = "max_steps"
            break
        root_states.append((step, state, state_type(model, state)))
        if persistent_planner is None:
            planner = make_planner(model, args, rng, visited_keys, tva_checkpoint)
            if args.reuse_tree:
                persistent_planner = planner
        else:
            planner = persistent_planner
        plan = planner.plan(state)
        total_deadlock_prunes += int(plan["deadlock_prunes"])
        action = plan["action"]
        if action is None:
            status = "no_action"
            break
        state = model.step_grid_flow(state, action)
        pushes_executed += 1
        visited_keys.add(model.canonical_key(state))
        if model.is_deadlocked(state) and not model.is_solved(state):
            status = "deadlocked"
            break

    samples: list[OnlineStateSample] = []
    for step, sample_state, sample_state_type in root_states:
        samples.append(
            OnlineStateSample(
                level_index=int(level_index),
                level_id=level.level_id,
                state=sample_state,
                source="online_root_state",
                root_step=int(step),
                remaining_pushes=max(int(pushes_executed) - int(step), 0),
                trajectory_status=status,
                state_type=sample_state_type,
                terminal_value_source=status,
            )
        )
    if bool(args.collect_final_states):
        samples.append(
            OnlineStateSample(
                level_index=int(level_index),
                level_id=level.level_id,
                state=state,
                source="online_final_state",
                root_step=int(pushes_executed),
                remaining_pushes=0,
                trajectory_status=status,
                state_type=state_type(model, state),
                terminal_value_source=status,
            )
        )
    return samples, {
        "level_index": int(level_index),
        "level_id": level.level_id,
        "status": status,
        "steps_executed": int(pushes_executed),
        "raw_samples": int(len(samples)),
        "deadlock_prunes": int(total_deadlock_prunes),
        "elapsed_sec": float(perf_counter() - t0),
    }


def deduplicate_samples(samples: Sequence[OnlineStateSample]) -> list[OnlineStateSample]:
    by_key: dict[tuple[str, SokobanState, str], OnlineStateSample] = {}
    for sample in samples:
        key = sample.key_tuple()
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = sample
            continue
        if previous.source != "online_final_state" and sample.source == "online_final_state":
            by_key[key] = sample
    return list(by_key.values())


def collect_online_samples(
    levels: Sequence[Any],
    args: argparse.Namespace,
) -> tuple[list[OnlineStateSample], dict[str, Any]]:
    if args.sampling_leaf_value == "tva":
        if not args.tva_checkpoint:
            raise SystemExit("--sampling-leaf-value=tva requires --tva-checkpoint.")
        tva_checkpoint = load_tva_checkpoint(args.tva_checkpoint)
    else:
        tva_checkpoint = None
    all_samples: list[OnlineStateSample] = []
    per_level: list[dict[str, Any]] = []
    t0 = perf_counter()
    for level_index, level in enumerate(levels):
        samples, row = collect_level_trajectory(level_index, level, args, tva_checkpoint)
        all_samples.extend(samples)
        per_level.append(row)
    raw_count = len(all_samples)
    if bool(args.deduplicate_states):
        all_samples = deduplicate_samples(all_samples)
    return all_samples, {
        "raw_samples": int(raw_count),
        "samples_after_dedup": int(len(all_samples)),
        "deduplicate_states": bool(args.deduplicate_states),
        "status_counts": dict(Counter(row["status"] for row in per_level)),
        "per_level": per_level,
        "elapsed_sec": float(perf_counter() - t0),
    }


def feature_row(model: SokobanGFTSDModel, state: SokobanState, feature_names: Sequence[str]) -> np.ndarray:
    features = model.gft_tva_feature_dict(state)["features"]
    return np.asarray([float(features[name]) for name in feature_names], dtype=np.float64)


def label_online_samples(
    online_samples: Sequence[OnlineStateSample],
    levels: Sequence[Any],
    split: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[LeafTvaTrainingSample], dict[str, Any], list[str]]:
    model_by_level = {level.level_id: gft_model_from_level(level, args) for level in levels}
    train_levels = set(split["train_levels"])
    valid_levels = set(split["valid_levels"])
    feature_names: list[str] | None = None
    rows: list[LeafTvaTrainingSample] = []
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    t0 = perf_counter()

    for sample in online_samples:
        split_name = "train" if sample.level_id in train_levels else "valid" if sample.level_id in valid_levels else "skip"
        if split_name == "skip":
            continue
        model = model_by_level[sample.level_id]
        if feature_names is None:
            feature_names = list(model.gft_tva_feature_dict(sample.state)["feature_names"])
        terminal_value = terminal_value_for_status(sample.trajectory_status, args)
        target_value: float | None = None
        if sample.trajectory_status == "solved":
            value_label = True
        elif sample.trajectory_status in {"deadlocked", "no_action"}:
            value_label = (
                str(args.failure_value_label_scope) == "all"
                or sample.source == "online_final_state"
            )
        else:
            value_label = False
        if sample.trajectory_status == "max_steps" and float(args.max_steps_value_weight) > 0.0:
            value_label = True
        if value_label:
            target_value = discounted_terminal_value(sample.remaining_pushes, terminal_value, args)
        if sample.trajectory_status == "solved":
            value_weight = float(args.solved_value_weight)
            risk_label = 0.0
            risk_weight = float(args.solved_risk_weight)
        elif sample.trajectory_status in {"deadlocked", "no_action"}:
            value_weight = float(args.deadend_value_weight)
            risk_label = 1.0
            risk_weight = float(args.deadend_risk_weight)
        elif sample.trajectory_status == "max_steps":
            value_weight = float(args.max_steps_value_weight) if value_label else 0.0
            risk_label = 1.0
            risk_weight = float(args.censored_risk_weight)
        else:
            value_weight = 0.0
            risk_label = 1.0
            risk_weight = float(args.censored_risk_weight)
        rows.append(
            LeafTvaTrainingSample(
                features=feature_row(model, sample.state, feature_names),
                target_value=target_value,
                value_label=bool(value_label),
                value_weight=float(value_weight),
                risk_label=float(risk_label),
                risk_weight=float(risk_weight),
                split=split_name,
                level_index=int(sample.level_index),
                level_id=sample.level_id,
                source=sample.source,
                root_step=int(sample.root_step),
                state_type=sample.state_type,
                teacher_failure_type=sample.trajectory_status,
                teacher_solution_pushes=(
                    int(sample.remaining_pushes) if sample.trajectory_status == "solved" else None
                ),
                teacher_expanded=0,
                gft_sd_leaf_value=compute_gft_sd_leaf_value(model, sample.state, args),
                sd_heuristic_value=compute_sd_heuristic_value(model, sample.state, args),
            )
        )
        label_counts[sample.trajectory_status] += 1
        split_counts[split_name] += 1
        source_counts[sample.source] += 1
    if feature_names is None:
        raise SystemExit("Could not infer GFT-TVA feature names from online samples.")
    return rows, {
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "label_elapsed_sec": float(perf_counter() - t0),
        "max_steps_as_value_target": float(args.max_steps_value_weight) > 0.0,
    }, feature_names


def build_summary(
    args: argparse.Namespace,
    samples: Sequence[LeafTvaTrainingSample],
    feature_names: Sequence[str],
    split: dict[str, Any],
    collection: dict[str, Any],
    labeling: dict[str, Any],
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    train_idx = indices_for_split(samples, "train")
    risk_classes = sorted(set(float(sample.risk_label) for sample in samples if sample.split == "train"))
    value_train_samples = int(np.sum(arrays["value_mask"][train_idx]))
    failed = []
    if value_train_samples < int(args.min_value_train_samples):
        failed.append("min_value_train_samples")
    if len(risk_classes) < int(args.min_risk_classes):
        failed.append("min_risk_classes")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["prediction"]).all():
        failed.append("nonfinite_features_or_predictions")
    source_counts = Counter(sample.source for sample in samples)
    status_counts = Counter(sample.teacher_failure_type for sample in samples)
    split_counts = Counter(sample.split for sample in samples)
    return {
        "material_passport": {
            "origin": "sokoban_gft_online_tva",
            "verification_status": "ANALYZED",
            "version_label": args.value_provider_id,
        },
        "config": {
            "levels": _rel(args.levels),
            "level_offset": args.level_offset,
            "max_levels": args.max_levels,
            "valid_fraction": args.valid_fraction,
            "horizon": args.horizon,
            "budget": args.budget,
            "max_steps": args.max_steps,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "max_steps_value": args.max_steps_value,
            "no_action_value": args.no_action_value,
            "constraint_mode": args.constraint_mode,
            "select_by": args.select_by,
            "revisit_penalty": args.revisit_penalty,
            "transposition": bool(args.transposition),
            "reuse_tree": bool(args.reuse_tree),
            "sampling_leaf_value": args.sampling_leaf_value,
            "tva_checkpoint": _rel(args.tva_checkpoint) if args.tva_checkpoint else None,
            "tva_value_blend": args.tva_value_blend,
            "gft_prior": args.gft_prior,
            "gft_prior_temperature": args.gft_prior_temperature,
            "gft_prior_uniform_mix": args.gft_prior_uniform_mix,
            "gft_prior_weights": gft_weights_from_args(args),
            "gft_transport_cost_weights": gft_transport_weights_from_args(args),
            "gft_soft_deadlock_weights": gft_soft_deadlock_weights_from_args(args),
            "gft_commitment_beta": args.gft_commitment_beta,
            "gft_sinkhorn_iterations": args.gft_sinkhorn_iterations,
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
            "max_steps_value_weight": args.max_steps_value_weight,
            "failure_value_label_scope": args.failure_value_label_scope,
            "risk_loss_weight": args.risk_loss_weight,
            "risk_penalty": args.risk_penalty,
            "anchor_weight": args.anchor_weight,
            "anchor_bias_mode": args.anchor_bias_mode,
            "anchor_bias_labels": args.anchor_bias_labels,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "value_provider_id": args.value_provider_id,
        },
        "tva_contract": {
            "value_provider_id": args.value_provider_id,
            "target_source": "closed_loop_online_gft_sdmcts_trajectories",
            "value_label_source": ["solved_trajectory", "deadlocked_trajectory", "no_action_trajectory"],
            "failure_value_label_scope": args.failure_value_label_scope,
            "censored_risk_source": ["max_steps_trajectory"],
            "max_steps_as_value_target": float(args.max_steps_value_weight) > 0.0,
            "teacher_used_online": False,
            "teacher_used_for_training_target": False,
            "risk_head_used_for_guarded_prediction": True,
            "leaf_formula": (
                "V_leaf=clip((1-anchor_weight)*(value_head-risk_penalty*risk_probability)"
                "+anchor_weight*(V_SD+anchor_bias))"
            ),
            "target_mode": "direct_value",
            "paper_alignment": "online_trajectory_terminal_value_with_censored_risk_guard",
            "transition_provider_modified": False,
            "sd_provider": SokobanGFTSDModel.provider_id,
            "transition_provider": SokobanGFTSDModel.transition_provider_id,
            "feature_contract": "gft_tva_feature_vector_v1",
            "feature_names": list(feature_names),
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
        },
        "gate": {
            "tva_training_gate": "PASS" if not failed else "FAIL",
            "failed_checks": failed,
            "thresholds": {
                "min_value_train_samples": args.min_value_train_samples,
                "min_risk_classes": args.min_risk_classes,
            },
        },
        "dataset": {
            "samples": int(len(samples)),
            "train_samples": int(split_counts.get("train", 0)),
            "valid_samples": int(split_counts.get("valid", 0)),
            "value_train_samples": value_train_samples,
            "feature_dim": int(arrays["features"].shape[1]),
            "level_split": split,
            "source_counts": dict(source_counts),
            "trajectory_status_counts": dict(status_counts),
            "collection": collection,
            "labeling": labeling,
            "feature_summary": feature_summary(arrays["features"], feature_names),
        },
        "metrics": metrics,
        "sample_preview": [asdict(sample) | {"features": "<array>"} for sample in samples[:8]],
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= float(args.anchor_weight) <= 1.0:
        raise SystemExit("--anchor-weight must be between 0 and 1.")
    if args.sampling_leaf_value != "tva" and args.tva_checkpoint:
        raise SystemExit("--tva-checkpoint is only used with --sampling-leaf-value=tva.")
    set_seed(args.seed)
    t0 = perf_counter()
    levels = load_selected_levels(args)
    online_samples, collection = collect_online_samples(levels, args)
    split = split_levels([level.level_id for level in levels], args.valid_fraction, args.seed)
    samples, labeling, feature_names = label_online_samples(online_samples, levels, split, args)
    if not samples:
        raise SystemExit("Online TVA collection produced no labelled samples.")
    net, metrics, arrays = train_network(samples, feature_names, args)
    summary = build_summary(args, samples, feature_names, split, collection, labeling, metrics, arrays)
    summary["elapsed_sec"] = float(perf_counter() - t0)

    checkpoint = {
        "value_provider_id": args.value_provider_id,
        "model_type": "mlp_guarded_v1",
        "target_mode": "direct_value",
        "feature_contract": {
            "contract": "gft_tva_feature_vector_v1",
            "feature_names": list(feature_names),
            "feature_dim": int(len(feature_names)),
        },
        "value_scale": {
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
        },
        "checkpoint_contract": summary["tva_contract"],
        "normalization": {
            "mean": arrays["mean"].astype(float).tolist(),
            "std": arrays["std"].astype(float).tolist(),
        },
        "mlp": export_mlp(net),
        "risk_guard": {
            "risk_penalty": float(args.risk_penalty),
            "anchor_weight": float(args.anchor_weight),
            "anchor_bias": float(arrays["anchor_bias"][0]),
            "anchor_bias_mode": args.anchor_bias_mode,
            "anchor_bias_labels": args.anchor_bias_labels,
            "clip_to_value_scale": True,
            "max_steps_as_value_target": float(args.max_steps_value_weight) > 0.0,
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
    summary = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": summary["gate"],
                "dataset": {
                    "samples": summary["dataset"]["samples"],
                    "value_train_samples": summary["dataset"]["value_train_samples"],
                    "trajectory_status_counts": summary["dataset"]["trajectory_status_counts"],
                    "collection_status_counts": summary["dataset"]["collection"]["status_counts"],
                },
                "metrics": summary["metrics"],
                "checkpoint": summary["checkpoint"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {out}")
    print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
