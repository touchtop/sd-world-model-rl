#!/usr/bin/env python3
"""Audit route/goal commitment options on fixed SD-MCTS root states.

This diagnostic is intentionally offline. It generates candidate options that
bind one pushed box to one target goal and require each successive push of that
box to reduce the static reverse-pull distance to the bound goal. The teacher is
used only after candidate generation to label option successor states.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import action_to_dict, stock_reward, stock_value  # noqa: E402
from evaluate_stage3_sdmcts_planning import PlannerAction, _score_option_sequence, _target2_options  # noqa: E402
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from src.sokoban.env import Cell, DIRECTIONS, add_cells  # noqa: E402
from src.sokoban.sd_model import INF_COST, SokobanPushAction  # noqa: E402
from train_sokoban_tva import best_first_from_state, target_from_teacher  # noqa: E402
from validate_sokoban_sd_design import stock_proxy  # noqa: E402


@dataclass(frozen=True)
class CandidateOption:
    primitives: tuple[SokobanPushAction, ...]
    source: str
    structural_score: float
    target_goal_index: int | None = None
    target_goal: Cell | None = None
    initial_box: Cell | None = None
    final_box: Cell | None = None
    start_distance: float | None = None
    end_distance: float | None = None
    route_distance_gain: float = 0.0

    @property
    def duration(self) -> int:
        return len(self.primitives)

    @property
    def first(self) -> SokobanPushAction:
        return self.primitives[0]

    @property
    def flat_index_tuple(self) -> tuple[int, ...]:
        return tuple(action.flat_index for action in self.primitives)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-result",
        default=(
            "results/sokoban_sd_world_model/"
            "stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json"
        ),
        help="Existing failed SD-MCTS result whose trace_sample roots define the audit states.",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--roots-per-episode", type=int, default=5)
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
    parser.add_argument("--critical-gap", type=float, default=0.25)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/route_goal_commitment_option_audit_seed0.json",
    )
    return parser


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _planner_to_candidate(option: PlannerAction) -> CandidateOption:
    return CandidateOption(
        primitives=option.primitives,
        source=option.source,
        structural_score=float(option.structural_score),
    )


def _route_score(
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    primitives: Sequence[SokobanPushAction],
    *,
    goal_idx: int,
    initial_box: Cell,
    final_box: Cell,
    start_distance: float,
    end_distance: float,
) -> float:
    exact_score, final_state = _score_option_sequence(model, root_state, primitives)
    board_scale = max(model.level.height + model.level.width, 1)
    route_gain = _safe_div(start_distance - end_distance, board_scale)
    target_reached = float(final_box == model.goals_sorted[goal_idx])
    after = model.abstract_features(final_state)
    terminal_bonus = 10.0 if after.solved else 0.0
    return float(exact_score + 2.0 * route_gain + 3.0 * target_reached + terminal_bonus)


def _append_route_option(
    out: list[CandidateOption],
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    primitives: tuple[SokobanPushAction, ...],
    *,
    goal_idx: int,
    initial_box: Cell,
    final_box: Cell,
    start_distance: float,
    end_distance: float,
) -> None:
    if len(primitives) < 2:
        return
    board_scale = max(model.level.height + model.level.width, 1)
    route_gain = _safe_div(start_distance - end_distance, board_scale)
    score = _route_score(
        model,
        root_state,
        primitives,
        goal_idx=goal_idx,
        initial_box=initial_box,
        final_box=final_box,
        start_distance=start_distance,
        end_distance=end_distance,
    )
    out.append(
        CandidateOption(
            primitives=primitives,
            source="route_goal_monotone_option",
            structural_score=score,
            target_goal_index=int(goal_idx),
            target_goal=model.goals_sorted[goal_idx],
            initial_box=initial_box,
            final_box=final_box,
            start_distance=float(start_distance),
            end_distance=float(end_distance),
            route_distance_gain=float(route_gain),
        )
    )


def _extend_route_options(
    out: list[CandidateOption],
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    current_state: SokobanState,
    current_box: Cell,
    primitives: tuple[SokobanPushAction, ...],
    *,
    goal_idx: int,
    initial_box: Cell,
    start_distance: float,
    current_distance: float,
    max_route_length: int,
) -> None:
    _append_route_option(
        out,
        model,
        root_state,
        primitives,
        goal_idx=goal_idx,
        initial_box=initial_box,
        final_box=current_box,
        start_distance=start_distance,
        end_distance=current_distance,
    )
    if len(primitives) >= max_route_length:
        return
    try:
        box_index = current_state.boxes.index(current_box)
    except ValueError:
        return
    dist = model.pull_distances[goal_idx]
    for action in model.legal_pushes(current_state):
        if action.box_index != box_index:
            continue
        direction = DIRECTIONS[action.direction]
        next_box = add_cells(current_box, direction)
        next_distance = float(dist.get(next_box, INF_COST))
        if next_distance >= current_distance or next_distance >= INF_COST:
            continue
        next_state = model.step_state(current_state, action)
        if model.is_deadlocked(next_state) and not model.is_solved(next_state):
            continue
        _extend_route_options(
            out,
            model,
            root_state,
            next_state,
            next_box,
            primitives + (action,),
            goal_idx=goal_idx,
            initial_box=initial_box,
            start_distance=start_distance,
            current_distance=next_distance,
            max_route_length=max_route_length,
        )


def _route_goal_options(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    *,
    max_route_length: int,
    max_options_per_root: int,
) -> list[CandidateOption]:
    if max_route_length < 2 or max_options_per_root <= 0:
        return []
    options: list[CandidateOption] = []
    for first in model.legal_pushes(state):
        initial_box = state.boxes[first.box_index]
        direction = DIRECTIONS[first.direction]
        first_box = add_cells(initial_box, direction)
        state1 = model.step_state(state, first)
        if model.is_deadlocked(state1) and not model.is_solved(state1):
            continue
        for goal_idx, dist in enumerate(model.pull_distances):
            start_distance = float(dist.get(initial_box, INF_COST))
            after_distance = float(dist.get(first_box, INF_COST))
            if (
                start_distance >= INF_COST
                or after_distance >= INF_COST
                or after_distance >= start_distance
            ):
                continue
            _extend_route_options(
                options,
                model,
                state,
                state1,
                first_box,
                (first,),
                goal_idx=goal_idx,
                initial_box=initial_box,
                start_distance=start_distance,
                current_distance=after_distance,
                max_route_length=max_route_length,
            )
    best_by_sequence: dict[tuple[int, ...], CandidateOption] = {}
    for option in options:
        key = option.flat_index_tuple
        old = best_by_sequence.get(key)
        if old is None or option.structural_score > old.structural_score:
            best_by_sequence[key] = option
    deduped = list(best_by_sequence.values())
    deduped.sort(
        key=lambda option: (
            option.structural_score,
            option.route_distance_gain,
            option.duration,
            tuple(-idx for idx in option.flat_index_tuple),
        ),
        reverse=True,
    )
    return deduped[:max_options_per_root]


def _candidate_options(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    args: argparse.Namespace,
) -> list[CandidateOption]:
    primitives = [
        CandidateOption(
            primitives=(action,),
            source="primitive_push",
            structural_score=0.0,
        )
        for action in model.legal_pushes(state)
    ]
    target2 = [
        _planner_to_candidate(option)
        for option in _target2_options(
            model,
            state,
            max_options_per_node=args.max_target2_options_per_root,
        )
    ]
    route_goal = _route_goal_options(
        model,
        state,
        max_route_length=args.max_route_length,
        max_options_per_root=args.max_route_options_per_root,
    )
    best_by_key: dict[tuple[str, tuple[int, ...]], CandidateOption] = {}
    for option in primitives + target2 + route_goal:
        key = (option.source, option.flat_index_tuple)
        old = best_by_key.get(key)
        if old is None or option.structural_score > old.structural_score:
            best_by_key[key] = option
    return list(best_by_key.values())


def _teacher_value(args: argparse.Namespace, model: SokobanGoalDirectedSDModel, state: SokobanState):
    teacher = best_first_from_state(
        model,
        state,
        max_expansions=args.teacher_max_expansions,
    )
    value = target_from_teacher(
        teacher,
        gamma=args.gamma,
        goal_value=args.goal_value,
        push_cost=args.push_cost,
        deadend_value=args.deadend_value,
        unsolved_value=args.unsolved_value,
    )
    return teacher, float(value)


def _label_option(
    args: argparse.Namespace,
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    option: CandidateOption,
    teacher_cache: dict[SokobanState, tuple[Any, float]],
) -> dict[str, Any] | None:
    current = state
    current_stock = stock_proxy(model, current).as_array().astype(np.float64)
    discounted_reward = 0.0
    fixed_return = 0.0
    discount = 1.0
    executed = 0
    for primitive in option.primitives:
        if primitive not in set(model.legal_pushes(current)):
            return None
        next_state = model.step_state(current, primitive)
        next_stock = stock_proxy(model, next_state).as_array().astype(np.float64)
        reward = stock_reward(current_stock, next_stock)
        discounted_reward += discount * reward
        fixed_return += discount * reward
        discount *= args.gamma
        current = next_state
        current_stock = next_stock
        executed += 1
        if model.is_solved(current) or not model.legal_pushes(current):
            break
    if current not in teacher_cache:
        teacher_cache[current] = _teacher_value(args, model, current)
    teacher, next_value = teacher_cache[current]
    teacher_q = float(discounted_reward + discount * next_value)
    fixed_return += discount * stock_value(current_stock)
    exact_score, _exact_state = _score_option_sequence(model, state, option.primitives)
    return {
        "teacher_q": teacher_q,
        "successor_teacher_value": float(next_value),
        "discounted_reward": float(discounted_reward),
        "fixed_stock_return": float(fixed_return),
        "exact_option_score": float(exact_score),
        "discount": float(discount),
        "executed": int(executed),
        "successor_solved": bool(model.is_solved(current)),
        "successor_deadlocked": bool(model.is_deadlocked(current)),
        "teacher_failure_type": str(teacher.failure_type),
        "teacher_solved": bool(teacher.solved),
        "teacher_solution_pushes": (
            int(len(teacher.actions)) if getattr(teacher, "actions", None) is not None else None
        ),
    }


def _load_root_states(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = json.loads(Path(args.source_result).read_text(encoding="utf-8"))
    levels = {
        level.level_id: level
        for level in load_boxoban_levels(args.levels, max_levels=args.max_levels)
    }
    roots: list[dict[str, Any]] = []
    skipped = Counter()
    for episode_index, episode in enumerate(source.get("episodes", [])[: args.max_episodes]):
        level_id = episode.get("level_id")
        if level_id not in levels:
            skipped["missing_level"] += 1
            continue
        model = SokobanGoalDirectedSDModel(levels[level_id])
        state = model.initial_state
        kept = 0
        for trace_row in episode.get("episode", {}).get("trace_sample", []):
            if kept >= args.roots_per_episode:
                break
            roots.append(
                {
                    "episode_index": int(episode_index),
                    "level_id": level_id,
                    "step": int(trace_row.get("step", kept)),
                    "state": state,
                }
            )
            kept += 1
            observed = trace_row.get("action", {}).get("flat_index")
            legal = {action.flat_index: action for action in model.legal_pushes(state)}
            if observed not in legal:
                skipped["observed_action_not_legal"] += 1
                break
            state = model.step_state(state, legal[int(observed)])
    return roots, {
        "source_result": args.source_result,
        "root_count": int(len(roots)),
        "skipped": dict(skipped),
    }


def _action_dicts(option: CandidateOption) -> list[dict[str, Any]]:
    return [action_to_dict(action) for action in option.primitives]


def _row_from_option(
    option: CandidateOption,
    label: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": option.source,
        "duration": int(option.duration),
        "flat_index_tuple": [int(idx) for idx in option.flat_index_tuple],
        "first_flat": int(option.first.flat_index),
        "first_action": action_to_dict(option.first),
        "primitive_actions": _action_dicts(option),
        "structural_score": float(option.structural_score),
        "teacher_q": float(label["teacher_q"]),
        "successor_teacher_value": float(label["successor_teacher_value"]),
        "discounted_reward": float(label["discounted_reward"]),
        "fixed_stock_return": float(label["fixed_stock_return"]),
        "exact_option_score": float(label["exact_option_score"]),
        "executed": int(label["executed"]),
        "successor_solved": bool(label["successor_solved"]),
        "successor_deadlocked": bool(label["successor_deadlocked"]),
        "teacher_failure_type": label["teacher_failure_type"],
        "teacher_solved": bool(label["teacher_solved"]),
        "teacher_solution_pushes": label["teacher_solution_pushes"],
        "target_goal_index": option.target_goal_index,
        "target_goal": list(option.target_goal) if option.target_goal is not None else None,
        "initial_box": list(option.initial_box) if option.initial_box is not None else None,
        "final_box": list(option.final_box) if option.final_box is not None else None,
        "start_distance": option.start_distance,
        "end_distance": option.end_distance,
        "route_distance_gain": float(option.route_distance_gain),
    }


def _best(rows: Sequence[dict[str, Any]], score_key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row[score_key]), -int(row["first_flat"])))


def _summarize_root(
    rows: list[dict[str, Any]],
    *,
    tie_tolerance: float,
    critical_gap: float,
) -> dict[str, Any]:
    if not rows:
        return {}
    max_q = max(float(row["teacher_q"]) for row in rows)
    teacher_best_tier = [
        row for row in rows if float(row["teacher_q"]) >= max_q - tie_tolerance
    ]
    teacher_best_sources = Counter(row["source"] for row in teacher_best_tier)
    best_first_q: dict[int, float] = defaultdict(lambda: float("-inf"))
    for row in rows:
        first = int(row["first_flat"])
        best_first_q[first] = max(best_first_q[first], float(row["teacher_q"]))
    best_first_value = max(best_first_q.values()) if best_first_q else max_q
    teacher_best_first_tier = {
        first for first, value in best_first_q.items() if value >= best_first_value - tie_tolerance
    }
    route_rows = [row for row in rows if row["source"] == "route_goal_monotone_option"]
    nonroute_rows = [row for row in rows if row["source"] != "route_goal_monotone_option"]
    route_best = _best(route_rows, "teacher_q")
    nonroute_best = _best(nonroute_rows, "teacher_q")
    score_summaries = {}
    for score_key in ("structural_score", "fixed_stock_return", "exact_option_score"):
        selected = _best(rows, score_key)
        route_selected = _best(route_rows, score_key)
        if selected is not None:
            gap = float(max_q - float(selected["teacher_q"]))
            score_summaries[score_key] = {
                "selected_source": selected["source"],
                "selected_flat_index_tuple": selected["flat_index_tuple"],
                "selected_teacher_q": float(selected["teacher_q"]),
                "selected_in_teacher_best_tier": bool(selected in teacher_best_tier),
                "selected_first_in_teacher_best_first_tier": bool(
                    int(selected["first_flat"]) in teacher_best_first_tier
                ),
                "selected_teacher_gap": gap,
                "selected_critical_miss": bool(gap > critical_gap),
            }
        if route_selected is not None:
            route_gap = float(max_q - float(route_selected["teacher_q"]))
            score_summaries[f"route_only_{score_key}"] = {
                "selected_source": route_selected["source"],
                "selected_flat_index_tuple": route_selected["flat_index_tuple"],
                "selected_teacher_q": float(route_selected["teacher_q"]),
                "selected_in_teacher_best_tier": bool(route_selected in teacher_best_tier),
                "selected_first_in_teacher_best_first_tier": bool(
                    int(route_selected["first_flat"]) in teacher_best_first_tier
                ),
                "selected_teacher_gap": route_gap,
                "selected_critical_miss": bool(route_gap > critical_gap),
            }
    return {
        "candidate_count": int(len(rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "teacher_best_value": float(max_q),
        "teacher_best_tier_size": int(len(teacher_best_tier)),
        "teacher_best_sources": dict(teacher_best_sources),
        "teacher_best_first_tier": sorted(int(first) for first in teacher_best_first_tier),
        "route_goal_candidate_count": int(len(route_rows)),
        "route_goal_best_teacher_q": (
            float(route_best["teacher_q"]) if route_best is not None else None
        ),
        "nonroute_best_teacher_q": (
            float(nonroute_best["teacher_q"]) if nonroute_best is not None else None
        ),
        "route_goal_beats_nonroute": bool(
            route_best is not None
            and nonroute_best is not None
            and float(route_best["teacher_q"]) > float(nonroute_best["teacher_q"]) + tie_tolerance
        ),
        "score_summaries": score_summaries,
    }


def _aggregate(root_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter()
    teacher_best_source_counts = Counter()
    score_counts: dict[str, Counter] = defaultdict(Counter)
    root_count = 0
    route_roots = 0
    route_beats_nonroute = 0
    for root in root_rows:
        summary = root["summary"]
        if not summary:
            continue
        root_count += 1
        source_counts.update(summary.get("source_counts", {}))
        teacher_best_source_counts.update(summary.get("teacher_best_sources", {}))
        if summary.get("route_goal_candidate_count", 0) > 0:
            route_roots += 1
        if summary.get("route_goal_beats_nonroute", False):
            route_beats_nonroute += 1
        for key, score_summary in summary.get("score_summaries", {}).items():
            score_counts[key]["evaluated_roots"] += 1
            score_counts[key]["teacher_best_tier_hits"] += int(
                bool(score_summary.get("selected_in_teacher_best_tier", False))
            )
            score_counts[key]["teacher_best_first_tier_hits"] += int(
                bool(score_summary.get("selected_first_in_teacher_best_first_tier", False))
            )
            score_counts[key]["critical_misses"] += int(
                bool(score_summary.get("selected_critical_miss", False))
            )
    score_metrics = {}
    for key, counts in score_counts.items():
        evaluated = int(counts["evaluated_roots"])
        score_metrics[key] = {
            "evaluated_roots": evaluated,
            "teacher_best_tier_hits": int(counts["teacher_best_tier_hits"]),
            "teacher_best_tier_hit_rate": _safe_div(
                counts["teacher_best_tier_hits"], evaluated
            ),
            "teacher_best_first_tier_hits": int(counts["teacher_best_first_tier_hits"]),
            "teacher_best_first_tier_hit_rate": _safe_div(
                counts["teacher_best_first_tier_hits"], evaluated
            ),
            "critical_misses": int(counts["critical_misses"]),
            "critical_miss_rate": _safe_div(counts["critical_misses"], evaluated),
        }
    return {
        "root_count": int(root_count),
        "candidate_source_counts": dict(source_counts),
        "teacher_best_source_counts": dict(teacher_best_source_counts),
        "route_goal_candidate_roots": int(route_roots),
        "route_goal_candidate_root_fraction": _safe_div(route_roots, root_count),
        "route_goal_beats_nonroute_roots": int(route_beats_nonroute),
        "route_goal_beats_nonroute_fraction": _safe_div(route_beats_nonroute, root_count),
        "score_metrics": score_metrics,
    }


def _build_gate(observed: dict[str, Any]) -> dict[str, Any]:
    failed = []
    root_count = int(observed.get("root_count", 0))
    route_fraction = float(observed.get("route_goal_candidate_root_fraction", 0.0))
    route_beats = int(observed.get("route_goal_beats_nonroute_roots", 0))
    route_struct = observed.get("score_metrics", {}).get("route_only_structural_score", {})
    route_struct_hits = float(route_struct.get("teacher_best_tier_hit_rate", 0.0))
    route_first_hits = float(route_struct.get("teacher_best_first_tier_hit_rate", 0.0))
    if root_count < 5:
        failed.append("root_count_min_5")
    if route_fraction < 0.50:
        failed.append("route_goal_candidate_root_fraction_min_0.50")
    if route_beats < 2:
        failed.append("route_goal_beats_nonroute_roots_min_2")
    if route_struct_hits < 0.30:
        failed.append("route_only_structural_top1_tier_hit_rate_min_0.30")
    if route_first_hits < 0.40:
        failed.append("route_only_structural_first_tier_hit_rate_min_0.40")
    return {
        "verdict": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "thresholds": {
            "root_count_min": 5,
            "route_goal_candidate_root_fraction_min": 0.50,
            "route_goal_beats_nonroute_roots_min": 2,
            "route_only_structural_top1_tier_hit_rate_min": 0.30,
            "route_only_structural_first_tier_hit_rate_min": 0.40,
        },
        "observed": {
            "root_count": root_count,
            "route_goal_candidate_root_fraction": route_fraction,
            "route_goal_beats_nonroute_roots": route_beats,
            "route_only_structural_top1_tier_hit_rate": route_struct_hits,
            "route_only_structural_first_tier_hit_rate": route_first_hits,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    roots, root_collection = _load_root_states(args)
    levels = {
        level.level_id: level
        for level in load_boxoban_levels(args.levels, max_levels=args.max_levels)
    }
    rows = []
    teacher_counts = Counter()
    for root in roots:
        model = SokobanGoalDirectedSDModel(levels[root["level_id"]])
        state = root["state"]
        teacher_cache: dict[SokobanState, tuple[Any, float]] = {}
        candidate_rows = []
        for option in _candidate_options(model, state, args):
            label = _label_option(args, model, state, option, teacher_cache)
            if label is None:
                continue
            teacher_counts[label["teacher_failure_type"]] += 1
            candidate_rows.append(_row_from_option(option, label))
        summary = _summarize_root(
            candidate_rows,
            tie_tolerance=args.teacher_tie_tolerance,
            critical_gap=args.critical_gap,
        )
        rows.append(
            {
                "episode_index": int(root["episode_index"]),
                "level_id": root["level_id"],
                "step": int(root["step"]),
                "summary": summary,
                "candidates": candidate_rows,
            }
        )
        print(
            json.dumps(
                {
                    "episode_index": int(root["episode_index"]),
                    "step": int(root["step"]),
                    "candidate_count": len(candidate_rows),
                    "route_goal_candidates": summary.get("route_goal_candidate_count", 0),
                    "route_goal_beats_nonroute": summary.get("route_goal_beats_nonroute", False),
                },
                ensure_ascii=False,
            )
        )
    observed = _aggregate(rows)
    gate = _build_gate(observed)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_route_goal_commitment_option_audit_v1",
        },
        "config": vars(args),
        "audit_contract": {
            "purpose": "test whether box-goal bound monotone route options merit planner integration",
            "teacher_usage": "offline_option_successor_label_only",
            "teacher_used_online": False,
            "best_first_used_online": False,
            "mcts_rerun": False,
            "planner_modified": False,
            "transition_provider_modified": False,
            "candidate_generation_uses_teacher": False,
            "candidate_generation_uses_best_first": False,
            "candidate_options": "primitive_pushes_plus_target2_plus_route_goal_monotone_options",
        },
        "root_collection": root_collection,
        "teacher_counts": dict(teacher_counts),
        "observed": observed,
        "route_goal_commitment_option_gate": gate,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "route_goal_commitment_option_gate": gate},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
