#!/usr/bin/env python3
"""Audit whether a terminal value compensator changes SD-MCTS root ordering.

The teacher is used only after planning, to label each root child's exact
successor state. Online SD-MCTS still consumes only the selected transition
provider, planner feedback, action abstraction, and terminal value evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import LearnedStockFlowTransitionProvider  # noqa: E402
from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    ExpertAbstractPlannerFeedback,
    SokobanTvaTerminalValueEvaluator,
    StockTerminalValueEvaluator,
    ZeroTerminalValueEvaluator,
    choose_mcts_action,
)
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import best_first_from_state, target_from_teacher  # noqa: E402
from validate_sokoban_sd_design import stock_proxy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-result",
        default=(
            "results/sokoban_sd_world_model/"
            "stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json"
        ),
        help="Existing SD-MCTS result whose trace_sample roots define the audit states.",
    )
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument(
        "--tva-checkpoint",
        default="results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt",
    )
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--mcts-depth", type=int, default=4)
    parser.add_argument("--puct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--teacher-max-expansions", type=int, default=800)
    parser.add_argument("--roots-per-episode", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--teacher-tie-tolerance", type=float, default=1e-9)
    parser.add_argument("--critical-gap", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json",
    )
    return parser


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranks_desc(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], -idx), reverse=True)
    ranks = [0 for _ in values]
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _policy_by_flat(policy: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in policy:
        action = row.get("action") or {}
        first = action.get("first_action") if isinstance(action, dict) else None
        if first and "flat_index" in first:
            rows[int(first["flat_index"])] = row
    return rows


def _teacher_value(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    cache: dict[SokobanState, tuple[Any, float]],
    *,
    teacher_max_expansions: int,
    gamma: float,
) -> tuple[Any, float]:
    if state not in cache:
        teacher = best_first_from_state(model, state, max_expansions=teacher_max_expansions)
        value = target_from_teacher(
            teacher,
            gamma=gamma,
            goal_value=10.0,
            push_cost=0.05,
            deadend_value=-10.0,
            unsolved_value=-2.5,
        )
        cache[state] = (teacher, float(value))
    return cache[state]


def _summarize_policy(
    policy: list[dict[str, Any]],
    teacher_by_flat: dict[int, dict[str, Any]],
    *,
    tie_tolerance: float,
    critical_gap: float,
) -> dict[str, Any]:
    rows = []
    for policy_row in policy:
        action = policy_row.get("action") or {}
        first = action.get("first_action") if isinstance(action, dict) else None
        if not first or "flat_index" not in first:
            continue
        flat = int(first["flat_index"])
        teacher = teacher_by_flat.get(flat)
        if teacher is None:
            continue
        rows.append(
            {
                "flat": flat,
                "q": float(policy_row["q_value"]),
                "visits": int(policy_row["visits"]),
                "prior": float(policy_row["prior"]),
                "teacher_value": float(teacher["value"]),
                "teacher_failure_type": teacher["failure_type"],
                "teacher_solution_pushes": teacher["solution_pushes"],
                "tva_provider_leaf_value": teacher.get("tva_provider_leaf_value"),
                "tva_exact_leaf_value": teacher.get("tva_exact_leaf_value"),
                "provider_stock_l1": teacher.get("provider_stock_l1"),
                "expert_reward_provider_stock": teacher.get("expert_reward_provider_stock"),
                "expert_reward_exact_stock": teacher.get("expert_reward_exact_stock"),
                "one_step_provider_score": teacher.get("one_step_provider_score"),
                "one_step_exact_score": teacher.get("one_step_exact_score"),
            }
        )
    if not rows:
        return {}

    flats = [int(row["flat"]) for row in rows]
    q_values = [float(row["q"]) for row in rows]
    visits = [float(row["visits"]) for row in rows]
    priors = [float(row["prior"]) for row in rows]
    teacher_values = [float(row["teacher_value"]) for row in rows]
    one_step_provider_values = [
        float(row["one_step_provider_score"])
        for row in rows
        if row.get("one_step_provider_score") is not None
    ]
    one_step_exact_values = [
        float(row["one_step_exact_score"])
        for row in rows
        if row.get("one_step_exact_score") is not None
    ]
    q_ranks = _ranks_desc(q_values)
    visit_ranks = _ranks_desc(visits)
    prior_ranks = _ranks_desc(priors)
    teacher_ranks = _ranks_desc(teacher_values)

    max_teacher = float(max(teacher_values))
    teacher_best_tier = {
        flats[idx]
        for idx, value in enumerate(teacher_values)
        if value >= max_teacher - tie_tolerance
    }
    q_best_idx = int(np.argmax(q_values))
    visit_best_idx = int(np.argmax(visits))
    prior_best_idx = int(np.argmax(priors))
    teacher_best_idx = int(np.argmax(teacher_values))
    selected_flat = flats[q_best_idx]
    selected_teacher_value = float(teacher_values[q_best_idx])
    selected_gap = float(max_teacher - selected_teacher_value)
    one_step_provider_best_flat = None
    one_step_provider_best_in_teacher_best_tier = None
    one_step_provider_teacher_corr = None
    one_step_exact_teacher_corr = None
    if len(one_step_provider_values) == len(rows):
        one_step_provider_best_idx = int(np.argmax(one_step_provider_values))
        one_step_provider_best_flat = int(flats[one_step_provider_best_idx])
        one_step_provider_best_in_teacher_best_tier = bool(
            one_step_provider_best_flat in teacher_best_tier
        )
        one_step_provider_teacher_corr = _corr(one_step_provider_values, teacher_values)
    if len(one_step_exact_values) == len(rows):
        one_step_exact_teacher_corr = _corr(one_step_exact_values, teacher_values)

    for row, q_rank, visit_rank, prior_rank, teacher_rank in zip(
        rows,
        q_ranks,
        visit_ranks,
        prior_ranks,
        teacher_ranks,
    ):
        row["q_rank"] = int(q_rank)
        row["visit_rank"] = int(visit_rank)
        row["prior_rank"] = int(prior_rank)
        row["teacher_rank"] = int(teacher_rank)
        row["in_teacher_best_tier"] = bool(row["flat"] in teacher_best_tier)

    return {
        "candidate_count": int(len(rows)),
        "teacher_best_flat": int(flats[teacher_best_idx]),
        "teacher_best_value": max_teacher,
        "teacher_best_tier": sorted(int(flat) for flat in teacher_best_tier),
        "teacher_best_tier_size": int(len(teacher_best_tier)),
        "q_best_flat": int(selected_flat),
        "visit_best_flat": int(flats[visit_best_idx]),
        "prior_best_flat": int(flats[prior_best_idx]),
        "q_best_in_teacher_best_tier": bool(selected_flat in teacher_best_tier),
        "visit_best_in_teacher_best_tier": bool(flats[visit_best_idx] in teacher_best_tier),
        "prior_best_in_teacher_best_tier": bool(flats[prior_best_idx] in teacher_best_tier),
        "one_step_provider_best_flat": one_step_provider_best_flat,
        "one_step_provider_best_in_teacher_best_tier": one_step_provider_best_in_teacher_best_tier,
        "teacher_best_q_rank": int(q_ranks[teacher_best_idx]),
        "teacher_best_visit_rank": int(visit_ranks[teacher_best_idx]),
        "teacher_best_prior_rank": int(prior_ranks[teacher_best_idx]),
        "selected_teacher_rank": int(teacher_ranks[q_best_idx]),
        "selected_teacher_value": selected_teacher_value,
        "selected_teacher_gap": selected_gap,
        "critical_miss": bool(selected_gap > critical_gap),
        "q_teacher_corr": _corr(q_values, teacher_values),
        "visit_teacher_corr": _corr(visits, teacher_values),
        "prior_teacher_corr": _corr(priors, teacher_values),
        "one_step_provider_teacher_corr": one_step_provider_teacher_corr,
        "one_step_exact_teacher_corr": one_step_exact_teacher_corr,
        "rows": sorted(rows, key=lambda row: (row["q_rank"], row["flat"])),
    }


def _aggregate_pairwise(root_rows: list[dict[str, Any]], base_mode: str) -> dict[str, Any]:
    rank_diffs = []
    corr_tva = []
    corr_base = []
    selected_better = 0
    selected_worse = 0
    selected_same = 0
    top_tier_tva = 0
    top_tier_base = 0
    critical_miss_tva = 0
    critical_miss_base = 0
    one_step_top_tier_tva = 0
    one_step_corr_tva = []
    usable = 0

    for row in root_rows:
        tva = row["modes"].get("teacherq_tva") or {}
        base = row["modes"].get(base_mode) or {}
        if not tva or not base:
            continue
        usable += 1
        rank_diffs.append(
            int(base.get("teacher_best_q_rank", 999))
            - int(tva.get("teacher_best_q_rank", 999))
        )
        if tva.get("q_teacher_corr") is not None:
            corr_tva.append(float(tva["q_teacher_corr"]))
        if base.get("q_teacher_corr") is not None:
            corr_base.append(float(base["q_teacher_corr"]))

        tva_selected = float(tva.get("selected_teacher_value", 0.0))
        base_selected = float(base.get("selected_teacher_value", 0.0))
        if tva_selected > base_selected + 1e-9:
            selected_better += 1
        elif tva_selected < base_selected - 1e-9:
            selected_worse += 1
        else:
            selected_same += 1
        top_tier_tva += int(bool(tva.get("q_best_in_teacher_best_tier")))
        top_tier_base += int(bool(base.get("q_best_in_teacher_best_tier")))
        critical_miss_tva += int(bool(tva.get("critical_miss")))
        critical_miss_base += int(bool(base.get("critical_miss")))
        if tva.get("one_step_provider_best_in_teacher_best_tier") is not None:
            one_step_top_tier_tva += int(bool(tva["one_step_provider_best_in_teacher_best_tier"]))
        if tva.get("one_step_provider_teacher_corr") is not None:
            one_step_corr_tva.append(float(tva["one_step_provider_teacher_corr"]))

    return {
        "roots": int(usable),
        "mean_teacher_best_q_rank_improvement_positive_means_tva_better": (
            float(np.mean(rank_diffs)) if rank_diffs else None
        ),
        "teacher_best_q_rank_improved_roots": int(sum(diff > 0 for diff in rank_diffs)),
        "teacher_best_q_rank_worse_roots": int(sum(diff < 0 for diff in rank_diffs)),
        "teacher_best_q_rank_same_roots": int(sum(diff == 0 for diff in rank_diffs)),
        "selected_teacher_value_better_roots": int(selected_better),
        "selected_teacher_value_worse_roots": int(selected_worse),
        "selected_teacher_value_same_roots": int(selected_same),
        "q_best_in_teacher_best_tier_tva": int(top_tier_tva),
        "q_best_in_teacher_best_tier_base": int(top_tier_base),
        "critical_miss_tva": int(critical_miss_tva),
        "critical_miss_base": int(critical_miss_base),
        "one_step_provider_best_in_teacher_best_tier_tva": int(one_step_top_tier_tva),
        "mean_q_teacher_corr_tva": float(np.mean(corr_tva)) if corr_tva else None,
        "mean_q_teacher_corr_base": float(np.mean(corr_base)) if corr_base else None,
        "mean_one_step_provider_teacher_corr_tva": (
            float(np.mean(one_step_corr_tva)) if one_step_corr_tva else None
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    source = json.loads(Path(args.source_result).read_text(encoding="utf-8"))
    levels = {
        level.level_id: level
        for level in load_boxoban_levels(args.levels, max_levels=source.get("config", {}).get("max_levels", 30))
    }
    provider = LearnedStockFlowTransitionProvider(Path(args.provider_checkpoint))
    feedback = ExpertAbstractPlannerFeedback()
    terminal_modes = {
        "zero_leaf": ZeroTerminalValueEvaluator(),
        "stock_leaf": StockTerminalValueEvaluator(),
        "teacherq_tva": SokobanTvaTerminalValueEvaluator(Path(args.tva_checkpoint)),
    }

    root_rows: list[dict[str, Any]] = []
    for episode_index, episode_row in enumerate(source.get("episodes", [])[: args.max_episodes]):
        level_id = episode_row["level_id"]
        if level_id not in levels:
            raise SystemExit(f"Level {level_id!r} from source result not found in {args.levels}.")
        model = SokobanGoalDirectedSDModel(levels[level_id])
        state = model.initial_state
        visited_states: Counter[SokobanState] = Counter([state])
        teacher_cache: dict[SokobanState, tuple[Any, float]] = {}

        for trace_row in episode_row["episode"].get("trace_sample", [])[: args.roots_per_episode]:
            raw_policies: dict[str, list[dict[str, Any]]] = {}
            union_flats: set[int] = set()
            for mode, terminal in terminal_modes.items():
                rng = np.random.RandomState(args.seed)
                torch.manual_seed(args.seed)
                _selected, info = choose_mcts_action(
                    model,
                    state,
                    provider,
                    terminal,
                    feedback,
                    num_simulations=args.budget,
                    max_depth=args.mcts_depth,
                    puct_c=args.puct_c,
                    gamma=args.gamma,
                    root_action_selection="value",
                    visited_states=visited_states,
                    revisit_feedback_penalty=2.0,
                    action_abstraction_mode="target2_prior",
                    max_option_length=2,
                    max_options_per_node=6,
                    option_prior_max_multiplier=3.0,
                    option_value_prior_weight=0.0,
                    previous_push=None,
                    previous_pushed_box_cell=None,
                    route_consistency_weight=0.0,
                    audit_proxy_prior_weight=0.0,
                    goal_room_conflict_prior_weight=0.0,
                    learned_action_prior_payload=None,
                    learned_action_prior_weight=0.0,
                    learned_action_prior_max_multiplier=3.0,
                    rng=rng,
                )
                raw_policies[mode] = info.get("root_policy", [])
                union_flats.update(_policy_by_flat(raw_policies[mode]).keys())

            legal = {action.flat_index: action for action in model.legal_pushes(state)}
            teacher_by_flat: dict[int, dict[str, Any]] = {}
            root_stock = stock_proxy(model, state).as_array().astype(np.float64)
            for flat in sorted(union_flats):
                action = legal.get(flat)
                if action is None:
                    continue
                next_state = model.step_state(state, action)
                teacher, value = _teacher_value(
                    model,
                    next_state,
                    teacher_cache,
                    teacher_max_expansions=args.teacher_max_expansions,
                    gamma=args.gamma,
                )
                prediction = provider.predict(model, state, root_stock, action)
                exact_next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
                tva = terminal_modes["teacherq_tva"]
                tva_provider_leaf_value = tva.value(model, next_state, prediction.next_stock)
                tva_exact_leaf_value = tva.value(model, next_state, exact_next_stock)
                expert_reward_provider = feedback.reward(
                    model,
                    state,
                    root_stock,
                    next_state,
                    prediction.next_stock,
                )
                expert_reward_exact = feedback.reward(
                    model,
                    state,
                    root_stock,
                    next_state,
                    exact_next_stock,
                )
                teacher_by_flat[int(flat)] = {
                    "value": float(value),
                    "failure_type": teacher.failure_type,
                    "solved": bool(teacher.solved),
                    "solution_pushes": (
                        int(teacher.solution_pushes)
                        if teacher.solution_pushes is not None
                        else None
                    ),
                    "tva_provider_leaf_value": float(tva_provider_leaf_value),
                    "tva_exact_leaf_value": float(tva_exact_leaf_value),
                    "provider_stock_l1": float(
                        np.mean(np.abs(prediction.next_stock - exact_next_stock))
                    ),
                    "expert_reward_provider_stock": float(expert_reward_provider),
                    "expert_reward_exact_stock": float(expert_reward_exact),
                    "one_step_provider_score": float(
                        expert_reward_provider + args.gamma * tva_provider_leaf_value
                    ),
                    "one_step_exact_score": float(
                        expert_reward_exact + args.gamma * tva_exact_leaf_value
                    ),
                }

            root_rows.append(
                {
                    "episode_index": int(episode_index),
                    "level_id": level_id,
                    "step": int(trace_row.get("step", len(root_rows))),
                    "source_observed_selected_flat": trace_row.get("action", {}).get("flat_index"),
                    "teacher_by_flat": teacher_by_flat,
                    "modes": {
                        mode: _summarize_policy(
                            policy,
                            teacher_by_flat,
                            tie_tolerance=args.teacher_tie_tolerance,
                            critical_gap=args.critical_gap,
                        )
                        for mode, policy in raw_policies.items()
                    },
                }
            )

            observed_flat = trace_row.get("action", {}).get("flat_index")
            if observed_flat not in legal:
                break
            state = model.step_state(state, legal[int(observed_flat)])
            visited_states[state] += 1

    aggregate = {
        "zero_leaf_vs_teacherq_tva": _aggregate_pairwise(root_rows, "zero_leaf"),
        "stock_leaf_vs_teacherq_tva": _aggregate_pairwise(root_rows, "stock_leaf"),
    }
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_tva_root_sorting_audit_v1",
        },
        "config": {
            "source_result": os.path.relpath(args.source_result, ROOT),
            "provider_checkpoint": os.path.relpath(args.provider_checkpoint, ROOT),
            "levels": args.levels,
            "tva_checkpoint": os.path.relpath(args.tva_checkpoint, ROOT),
            "budget": int(args.budget),
            "mcts_depth": int(args.mcts_depth),
            "puct_c": float(args.puct_c),
            "gamma": float(args.gamma),
            "teacher_max_expansions": int(args.teacher_max_expansions),
            "roots_per_episode": int(args.roots_per_episode),
            "max_episodes": int(args.max_episodes),
            "teacher_tie_tolerance": float(args.teacher_tie_tolerance),
            "critical_gap": float(args.critical_gap),
            "seed": int(args.seed),
        },
        "audit_contract": {
            "teacher_usage": "offline_successor_label_only",
            "teacher_used_online": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "planner_modified": False,
            "compared_terminal_modes": list(terminal_modes.keys()),
            "root_action_selection": "value",
        },
        "aggregate": aggregate,
        "root_rows": root_rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "aggregate": aggregate, "roots": len(root_rows)}, indent=2))


if __name__ == "__main__":
    main()
