#!/usr/bin/env python3
"""Teacher-trace gap analysis for the Room-GFT XSokoban hard line.

This diagnostic uses exact-state teacher trajectories as evidence, not as an
online solver.  For each teacher state, it asks how the current Room-GFT/A6
scoring layer ranks the teacher action and which room/slot signals explain the
gap.  The output is intended to identify missing SD state variables, feedback
terms, and action abstractions.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A7_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a7_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a7_trace_anchor.py",
)
if _A7_SPEC is None or _A7_SPEC.loader is None:
    raise RuntimeError("Unable to load A7 diagnostic helpers.")
_A7 = importlib.util.module_from_spec(_A7_SPEC)
_A7_SPEC.loader.exec_module(_A7)

from src.sokoban import (  # noqa: E402
    RoomCommitmentState,
    RoomCorridorMacroGFTSDMCTSPlanner,
    RoomSlotOption,
    SokobanPushAction,
    SokobanRoomGFTSDModel,
    SokobanState,
    load_boxoban_levels,
)
from src.sokoban.env import ACTION_NAMES, DIRECTIONS, add_cells  # noqa: E402


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def action_label(action: SokobanPushAction | dict[str, Any] | None) -> str:
    if action is None:
        return "NA"
    if isinstance(action, dict):
        return f"b{action['box_index']}:{action['direction_name']}"
    return f"b{action.box_index}:{ACTION_NAMES[action.direction]}"


def option_label(option: RoomSlotOption | None) -> str:
    if option is None:
        return "NA"
    if option.kind == "slot":
        return f"slot{option.slot_id}@{list(option.slot_cell)}"
    return str(option.kind)


def teacher_trace_search_bounded(
    model: SokobanRoomGFTSDModel,
    start: SokobanState,
    *,
    max_expansions: int,
    max_seconds: float,
    heuristic_weight: float = 1.0,
    use_viable_push_term: bool = True,
    deadlock_mode: str = "exact",
    matching_backend: str = "dp",
) -> dict[str, Any]:
    """Run the exact-state teacher with expansion and wall-clock bounds."""

    legal_cache: dict[SokobanState, tuple[SokobanPushAction, ...]] = {}
    deadlock_cache: dict[SokobanState, bool] = {}
    heuristic_cache: dict[SokobanState, float] = {}
    frozen_cache: dict[SokobanState, int] = {}
    goals = set(model.level.goals)
    goals_sorted = tuple(model.goals_sorted)
    matching_backend = str(matching_backend)
    deadlock_mode = str(deadlock_mode)
    heuristic_weight = float(heuristic_weight)
    use_viable_push_term = bool(use_viable_push_term)
    hungarian_solver = None
    np_module = None
    if matching_backend == "hungarian":
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import numpy as _np
                from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

            np_module = _np
            hungarian_solver = _linear_sum_assignment
        except Exception:
            matching_backend = "dp"
            hungarian_solver = None
            np_module = None
    assignment_cache: dict[tuple[tuple[int, int], ...], float] = {}

    def over_time(started_at: float) -> bool:
        return max_seconds > 0.0 and perf_counter() - started_at >= max_seconds

    def legal_pushes(state: SokobanState) -> tuple[SokobanPushAction, ...]:
        cached = legal_cache.get(state)
        if cached is None:
            cached = model.legal_pushes(state)
            legal_cache[state] = cached
        return cached

    def matching_lower_bound(boxes) -> float:
        key = tuple(boxes)
        if matching_backend != "hungarian" or hungarian_solver is None or np_module is None:
            return float(model.gft_matching_lower_bound_cached(key))
        cached = assignment_cache.get(key)
        if cached is not None:
            return cached
        inf_cost = 10**6
        matrix = np_module.empty((len(key), len(goals_sorted)), dtype=np_module.float64)
        for row, box in enumerate(key):
            for col, _goal in enumerate(goals_sorted):
                distance = model.pull_distances[col].get(box)
                matrix[row, col] = inf_cost if distance is None or distance >= 10**8 else float(distance)
        rows, cols = hungarian_solver(matrix)
        total = float(matrix[rows, cols].sum())
        value = math.inf if total >= inf_cost else total
        assignment_cache[key] = value
        return value

    def is_deadlocked(state: SokobanState) -> bool:
        cached = deadlock_cache.get(state)
        if cached is None:
            if model.is_solved(state):
                cached = False
            elif deadlock_mode == "none":
                cached = False
            elif deadlock_mode == "exact" and not math.isfinite(matching_lower_bound(state.boxes)):
                cached = True
            elif any(box not in goals and box in model.dead_squares for box in state.boxes):
                cached = True
            else:
                frozen = frozen_cache.get(state)
                if frozen is None:
                    frozen = model.frozen_boxes(state)
                    frozen_cache[state] = frozen
                cached = frozen > 0
            deadlock_cache[state] = cached
        return cached

    def heuristic(state: SokobanState) -> float:
        cached = heuristic_cache.get(state)
        if cached is not None:
            return cached
        lb = matching_lower_bound(state.boxes)
        if lb == float("inf"):
            value = 1e6
        else:
            viable = 0
            if use_viable_push_term:
                for action in legal_pushes(state):
                    child = model.step_state(state, action)
                    if not is_deadlocked(child):
                        viable += 1
            value = float(
                heuristic_weight * lb
                - 0.25 * model.boxes_on_goals(state)
                - (0.01 * viable if use_viable_push_term else 0.0)
            )
        heuristic_cache[state] = value
        return value

    started_at = perf_counter()
    if model.is_solved(start):
        return {
            "solved": True,
            "solution_pushes": 0,
            "expanded": 0,
            "generated": 0,
            "pruned_deadlocks": 0,
            "frontier_size": 0,
            "failure_type": "solved",
            "seconds": 0.0,
            "actions": [],
            "states": [start],
            "best_seen": 1,
            "cache_sizes": {"heuristic": 0, "legal": 0, "deadlock": 0, "assignment": 0},
            "teacher_mode": {
                "heuristic_weight": float(heuristic_weight),
                "use_viable_push_term": bool(use_viable_push_term),
                "deadlock_mode": str(deadlock_mode),
                "matching_backend": str(matching_backend),
            },
        }
    if model.is_deadlocked(start):
        return {
            "solved": False,
            "solution_pushes": None,
            "expanded": 0,
            "generated": 0,
            "pruned_deadlocks": 0,
            "frontier_size": 0,
            "failure_type": "proven_dead_end",
            "seconds": 0.0,
            "actions": [],
            "states": [start],
            "best_seen": 1,
            "cache_sizes": {"heuristic": 0, "legal": 0, "deadlock": 0, "assignment": 0},
            "teacher_mode": {
                "heuristic_weight": float(heuristic_weight),
                "use_viable_push_term": bool(use_viable_push_term),
                "deadlock_mode": str(deadlock_mode),
                "matching_backend": str(matching_backend),
            },
        }

    frontier: list[tuple[float, int, int, SokobanState]] = []
    counter = 0
    heapq.heappush(frontier, (heuristic(start), 0, counter, start))
    best_g: dict[SokobanState, int] = {start: 0}
    parent: dict[SokobanState, tuple[SokobanState, SokobanPushAction]] = {}
    expanded = 0
    generated = 0
    pruned = 0
    solved_state: SokobanState | None = None
    solved_depth: int | None = None
    failure_type = "teacher_budget_unsolved"

    while frontier and expanded < max_expansions:
        if over_time(started_at):
            failure_type = "teacher_time_budget_unsolved"
            break
        _priority, g, _counter, state = heapq.heappop(frontier)
        if g != best_g.get(state):
            continue
        if model.is_solved(state):
            solved_state = state
            solved_depth = g
            failure_type = "solved"
            break
        expanded += 1
        for action in legal_pushes(state):
            if over_time(started_at):
                failure_type = "teacher_time_budget_unsolved"
                break
            child = model.step_state(state, action)
            if child == state:
                continue
            generated += 1
            if is_deadlocked(child):
                pruned += 1
                continue
            child_g = g + 1
            if child_g >= best_g.get(child, 10**9):
                continue
            best_g[child] = child_g
            parent[child] = (state, action)
            counter += 1
            heapq.heappush(frontier, (child_g + heuristic(child), child_g, counter, child))
        if failure_type == "teacher_time_budget_unsolved":
            break

    actions: list[SokobanPushAction] = []
    states: list[SokobanState] = [start]
    if solved_state is not None:
        cursor = solved_state
        while cursor in parent:
            prev, action = parent[cursor]
            actions.append(action)
            cursor = prev
        actions.reverse()
        states = [start]
        cursor = start
        for action in actions:
            cursor = model.step_state(cursor, action)
            states.append(cursor)

    return {
        "solved": solved_state is not None,
        "solution_pushes": solved_depth,
        "expanded": int(expanded),
        "generated": int(generated),
        "pruned_deadlocks": int(pruned),
        "frontier_size": int(len(frontier)),
        "failure_type": failure_type,
        "seconds": float(perf_counter() - started_at),
        "actions": actions,
        "states": states,
        "best_seen": int(len(best_g)),
        "cache_sizes": {
            "heuristic": int(len(heuristic_cache)),
            "legal": int(len(legal_cache)),
            "deadlock": int(len(deadlock_cache)),
            "assignment": int(len(assignment_cache)),
        },
        "teacher_mode": {
            "heuristic_weight": float(heuristic_weight),
            "use_viable_push_term": bool(use_viable_push_term),
            "deadlock_mode": str(deadlock_mode),
            "matching_backend": str(matching_backend),
        },
    }


def teacher_summary(teacher: dict[str, Any]) -> dict[str, Any]:
    return {
        "solved": bool(teacher["solved"]),
        "solution_pushes": teacher["solution_pushes"],
        "expanded": int(teacher["expanded"]),
        "generated": int(teacher["generated"]),
        "pruned_deadlocks": int(teacher["pruned_deadlocks"]),
        "frontier_size": int(teacher["frontier_size"]),
        "failure_type": str(teacher["failure_type"]),
        "seconds": float(teacher["seconds"]),
        "best_seen": int(teacher.get("best_seen", 0)),
        "cache_sizes": teacher.get("cache_sizes", {}),
        "teacher_mode": teacher.get("teacher_mode", {}),
    }


def build_a6_planner(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    rng: random.Random,
    visited_keys: set,
    boundary_distance: int,
    previous_physical_action: SokobanPushAction | None,
    previous_moved_source,
    previous_moved_target,
    recent_commitment_keys: set,
):
    return RoomCorridorMacroGFTSDMCTSPlanner(
        model,
        horizon=1,
        budget=1,
        rng=rng,
        uct_c=float(config["uct_c"]),
        gamma=float(config["gamma"]),
        goal_value=float(config["goal_value"]),
        push_cost=float(config["push_cost"]),
        deadend_value=float(config["deadend_value"]),
        constraint_mode=str(config["constraint_mode"]),
        select_by=str(config["select_by"]),
        revisit_penalty=float(config["revisit_penalty"]),
        visited_keys=visited_keys,
        transposition=True,
        gft_prior_weights={key: float(value) for key, value in config["gft_prior_weights"].items()},
        gft_prior_temperature=float(config["gft_prior_temperature"]),
        gft_prior_uniform_mix=float(config["gft_prior_uniform_mix"]),
        gft_sd_leaf_weights={key: float(value) for key, value in config["gft_sd_leaf_weights"].items()},
        semantic_commitment_boundary_distance=int(boundary_distance),
        previous_physical_action=previous_physical_action,
        previous_moved_source=previous_moved_source,
        previous_moved_target=previous_moved_target,
        recent_commitment_keys=recent_commitment_keys,
    )


def candidate_to_row(candidate) -> dict[str, Any]:
    action = candidate.action.physical_action
    progress = candidate.room_slot_progress_summary or {}
    delta = progress.get("delta", {})
    return {
        "box_index": int(action.box_index),
        "direction": int(action.direction),
        "direction_name": ACTION_NAMES[action.direction],
        "flat_index": int(action.flat_index),
        "option": candidate.action.option.to_dict(),
        "prior": float(candidate.prior),
        "prior_score": float(candidate.prior_score),
        "physical_prior_score": float(candidate.physical_prior_score),
        "room_prior_score": float(candidate.room_prior_score),
        "room_slot_progress_delta": float(candidate.room_slot_progress_delta),
        "room_slot_transport_delta": float(candidate.room_slot_transport_delta),
        "room_slot_committed_transport_delta": float(candidate.room_slot_committed_transport_delta),
        "room_slot_frontier_transport_delta": float(candidate.room_slot_frontier_transport_delta),
        "room_slot_barrier_delta": float(candidate.room_slot_barrier_delta),
        "room_slot_delta": {
            "slot_schedule_cost": float(delta.get("slot_schedule_cost", 0.0)),
            "total_transport_cost": float(delta.get("total_transport_cost", 0.0)),
            "committed_transport_cost": float(delta.get("committed_transport_cost", 0.0)),
            "uncommitted_frontier_transport_cost": float(delta.get("uncommitted_frontier_transport_cost", 0.0)),
            "barrier_cost": float(delta.get("barrier_cost", 0.0)),
            "committed_infeasible_increase": int(delta.get("committed_infeasible_increase", 0)),
            "committed_pass_through_increase": int(delta.get("committed_pass_through_increase", 0)),
        },
        "impact_codes": list(candidate.room_action_summary.get("impact_codes", [])),
        "loop_score": float(candidate.room_loop_penalty_score),
        "loop": candidate.room_loop_summary,
    }


def physical_group_best(candidates) -> list[dict[str, Any]]:
    by_flat: dict[int, Any] = {}
    for candidate in candidates:
        flat = int(candidate.action.physical_action.flat_index)
        current = by_flat.get(flat)
        if current is None or candidate.prior_score > current.prior_score:
            by_flat[flat] = candidate
    rows = [candidate_to_row(candidate) for candidate in by_flat.values()]
    rows.sort(key=lambda row: (row["prior_score"], row["prior"]), reverse=True)
    return rows


def matching_slot_candidate(candidates, flat_index: int, future_goal: tuple[int, int] | None):
    matching = [
        candidate
        for candidate in candidates
        if int(candidate.action.physical_action.flat_index) == int(flat_index)
    ]
    if not matching:
        return None
    if future_goal is not None:
        slot_matches = [
            candidate
            for candidate in matching
            if candidate.action.option.kind == "slot" and candidate.action.option.slot_cell == future_goal
        ]
        if slot_matches:
            return max(slot_matches, key=lambda candidate: (candidate.prior_score, candidate.prior))
    return max(matching, key=lambda candidate: (candidate.prior_score, candidate.prior))


def teacher_option_for_step(
    model: SokobanRoomGFTSDModel,
    state: SokobanState,
    action: SokobanPushAction,
    child: SokobanState,
    commitment_state: RoomCommitmentState,
    future_goal: tuple[int, int] | None,
    *,
    boundary_distance: int,
) -> RoomSlotOption:
    source = state.boxes[action.box_index]
    existing = commitment_state.commitment_for_box(source)
    options = model.room_semantic_slot_options(
        state,
        action,
        child,
        commitment_state=commitment_state,
        include_uncommitted_option=False,
        commitment_boundary_distance=int(boundary_distance),
    )
    if future_goal is not None:
        for option in options:
            if option.kind == "slot" and option.slot_cell == future_goal:
                return option
    if existing is not None:
        for option in options:
            if option.kind == "slot" and option.slot_cell == existing.slot_cell:
                return option
    for option in options:
        if option.kind in {"none", "uncommitted"}:
            return option
    return options[0] if options else RoomSlotOption(kind="none")


def gap_tags_for_row(
    teacher_row: dict[str, Any],
    top_row: dict[str, Any] | None,
    *,
    future_goal_option_available: bool,
    teacher_candidate_missing: bool,
) -> list[str]:
    tags: list[str] = []
    rank = teacher_row.get("teacher_physical_prior_rank")
    if teacher_candidate_missing:
        tags.append("teacher_action_pruned_or_missing")
        return tags
    if isinstance(rank, int) and rank > 3:
        tags.append("teacher_low_prior_rank")
    elif isinstance(rank, int) and rank > 1:
        tags.append("teacher_not_top_prior")
    if teacher_row["future_goal_cell"] is not None and not future_goal_option_available:
        tags.append("future_goal_slot_option_unavailable")
    if teacher_row["future_goal_cell"] is not None and teacher_row.get("teacher_option_slot_match") is False:
        tags.append("slot_commitment_mismatch")
    if teacher_row["teacher_room_slot_progress_delta"] <= 0.0 and isinstance(rank, int) and rank > 1:
        tags.append("setup_or_release_underrewarded")
    if "CROSS_ROOM_BOUNDARY" in teacher_row["teacher_impact_codes"] and isinstance(rank, int) and rank > 1:
        tags.append("entry_commitment_underweighted")
    if top_row is not None and top_row["flat_index"] != teacher_row["teacher_action"]["flat_index"]:
        if top_row["room_slot_transport_delta"] > teacher_row["teacher_room_slot_transport_delta"] + 0.5:
            tags.append("short_transport_competes")
        risky = {"FILL_NONFRONTIER_SLOT", "INCREASE_ROOM_ORDER_DEBT", "BLOCK_UNFILLED_SLOT"}
        if risky.intersection(set(top_row["impact_codes"])):
            tags.append("top_action_room_order_risk")
    return tags


def corridor_run_summary(states: list[SokobanState], actions: list[SokobanPushAction]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for step, action in enumerate(actions):
        source = states[step].boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        if (
            current is not None
            and current["direction"] == int(action.direction)
            and tuple(current["last_target"]) == source
        ):
            current["length"] += 1
            current["end_step"] = int(step)
            current["last_target"] = list(target)
        else:
            if current is not None:
                runs.append(current)
            current = {
                "start_step": int(step),
                "end_step": int(step),
                "length": 1,
                "direction": int(action.direction),
                "direction_name": ACTION_NAMES[action.direction],
                "first_source": list(source),
                "last_target": list(target),
            }
    if current is not None:
        runs.append(current)
    long_runs = [run for run in runs if int(run["length"]) >= 3]
    return {
        "run_count": int(len(runs)),
        "long_run_count_ge3": int(len(long_runs)),
        "long_run_lengths": [int(run["length"]) for run in long_runs],
        "long_runs_prefix": long_runs[:12],
        "max_run_length": max([int(run["length"]) for run in runs], default=0),
    }


def goal_fill_events(
    model: SokobanRoomGFTSDModel,
    states: list[SokobanState],
    actions: list[SokobanPushAction],
    future_goals: list[tuple[int, int] | None],
) -> list[dict[str, Any]]:
    events = []
    for step, action in enumerate(actions):
        before = model.boxes_on_goals(states[step])
        after = model.boxes_on_goals(states[step + 1])
        if after <= before:
            continue
        source = states[step].boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        events.append(
            {
                "step": int(step),
                "action": _A7.action_dict(states[step], action),
                "filled_target": list(target),
                "future_goal_cell": None if future_goals[step] is None else list(future_goals[step]),
                "boxes_on_goals_after": int(after),
            }
        )
    return events


def analyze_solved_teacher_trace(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    teacher: dict[str, Any],
    *,
    level_index: int,
    boundary_distance: int,
    max_rank_states: int,
    mcts_sample_stride: int,
    mcts_budget: int,
) -> dict[str, Any]:
    states: list[SokobanState] = teacher["states"]
    actions: list[SokobanPushAction] = teacher["actions"]
    future_goals = _A7.infer_future_goal_cells(model, states, actions)
    commitment_state: RoomCommitmentState = model.empty_room_commitment_state()
    visited_keys = {model.canonical_key(states[0])}
    recent_commitment_keys = {model.room_commitment_key(states[0], commitment_state)}
    previous_action: SokobanPushAction | None = None
    previous_source = None
    previous_target = None
    rng = random.Random(int(config["seed"]) + 31013 + level_index * 101)
    rows: list[dict[str, Any]] = []
    mcts_rows: list[dict[str, Any]] = []

    limit = min(len(actions), int(max_rank_states))
    for step in range(limit):
        state = states[step]
        action = actions[step]
        child = states[step + 1]
        future_goal = future_goals[step]
        planner = build_a6_planner(
            model,
            config,
            rng=rng,
            visited_keys=visited_keys,
            boundary_distance=boundary_distance,
            previous_physical_action=previous_action,
            previous_moved_source=previous_source,
            previous_moved_target=previous_target,
            recent_commitment_keys=recent_commitment_keys,
        )
        candidates, pruned = planner.candidate_actions(state, commitment_state)
        physical_rows = physical_group_best(candidates)
        rank_by_flat = {int(row["flat_index"]): rank + 1 for rank, row in enumerate(physical_rows)}
        top_row = physical_rows[0] if physical_rows else None
        teacher_candidate = matching_slot_candidate(candidates, action.flat_index, future_goal)
        matching_candidates = [
            candidate
            for candidate in candidates
            if int(candidate.action.physical_action.flat_index) == int(action.flat_index)
        ]
        future_goal_option_available = bool(
            future_goal is not None
            and any(
                candidate.action.option.kind == "slot" and candidate.action.option.slot_cell == future_goal
                for candidate in matching_candidates
            )
        )
        teacher_option_slot_match = None
        if teacher_candidate is not None and future_goal is not None:
            teacher_option_slot_match = bool(
                teacher_candidate.action.option.kind == "slot"
                and teacher_candidate.action.option.slot_cell == future_goal
            )
        teacher_candidate_row = None if teacher_candidate is None else candidate_to_row(teacher_candidate)
        row = {
            "step": int(step),
            "teacher_action": _A7.action_dict(state, action),
            "future_goal_cell": None if future_goal is None else list(future_goal),
            "boxes_on_goals_before": int(model.boxes_on_goals(state)),
            "boxes_on_goals_after": int(model.boxes_on_goals(child)),
            "slot_schedule_cost_before": float(
                model.room_slot_progress_summary(state, commitment_state)["slot_schedule_cost"]
            ),
            "candidate_count": int(len(candidates)),
            "physical_candidate_count": int(len(physical_rows)),
            "deadlock_pruned_children": int(pruned),
            "teacher_physical_prior_rank": rank_by_flat.get(int(action.flat_index)),
            "teacher_candidate_missing": teacher_candidate is None,
            "teacher_candidate": teacher_candidate_row,
            "teacher_option_slot_match": teacher_option_slot_match,
            "future_goal_option_available": bool(future_goal_option_available),
            "top_candidate": top_row,
            "top_action_is_teacher": bool(top_row and int(top_row["flat_index"]) == int(action.flat_index)),
            "teacher_room_slot_progress_delta": float(
                0.0 if teacher_candidate is None else teacher_candidate.room_slot_progress_delta
            ),
            "teacher_room_slot_transport_delta": float(
                0.0 if teacher_candidate is None else teacher_candidate.room_slot_transport_delta
            ),
            "teacher_room_slot_barrier_delta": float(
                0.0 if teacher_candidate is None else teacher_candidate.room_slot_barrier_delta
            ),
            "teacher_impact_codes": [] if teacher_candidate is None else list(
                teacher_candidate.room_action_summary.get("impact_codes", [])
            ),
        }
        row["gap_tags"] = gap_tags_for_row(
            row,
            top_row,
            future_goal_option_available=future_goal_option_available,
            teacher_candidate_missing=teacher_candidate is None,
        )
        rows.append(row)

        if mcts_sample_stride > 0 and step % int(mcts_sample_stride) == 0:
            mcts_planner = build_a6_planner(
                model,
                config,
                rng=rng,
                visited_keys=visited_keys,
                boundary_distance=boundary_distance,
                previous_physical_action=previous_action,
                previous_moved_source=previous_source,
                previous_moved_target=previous_target,
                recent_commitment_keys=recent_commitment_keys,
            )
            mcts_planner.budget = int(mcts_budget)
            mcts_planner.horizon = 8
            plan = mcts_planner.plan(state, commitment_state)
            selected = plan["physical_action"]
            mcts_rows.append(
                {
                    "step": int(step),
                    "teacher_action": _A7.action_dict(state, action),
                    "selected_action": None if selected is None else _A7.action_dict(state, selected),
                    "selected_matches_teacher": bool(
                        selected is not None and int(selected.flat_index) == int(action.flat_index)
                    ),
                    "root_visits": int(plan["root_visits"]),
                    "root_value": float(plan["root_value"]),
                    "semantic_child_count": int(plan["semantic_child_count"]),
                    "physical_child_count": int(plan["physical_child_count"]),
                }
            )

        teacher_option = teacher_option_for_step(
            model,
            state,
            action,
            child,
            commitment_state,
            future_goal,
            boundary_distance=boundary_distance,
        )
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        commitment_state = model.advance_room_commitment_state(
            state,
            commitment_state,
            action,
            teacher_option,
            child,
        )
        visited_keys.add(model.canonical_key(child))
        recent_commitment_keys.add(model.room_commitment_key(child, commitment_state))
        previous_action = action
        previous_source = source
        previous_target = target

    ranks = [int(row["teacher_physical_prior_rank"]) for row in rows if row["teacher_physical_prior_rank"] is not None]
    rank1 = [rank for rank in ranks if rank == 1]
    rank_le3 = [rank for rank in ranks if rank <= 3]
    tag_counts = Counter(tag for row in rows for tag in row["gap_tags"])
    setup_rows = [
        row
        for row in rows
        if "setup_or_release_underrewarded" in row["gap_tags"] or "teacher_low_prior_rank" in row["gap_tags"]
    ]
    slot_available_rows = [row for row in rows if row["future_goal_cell"] is not None]
    mcts_matches = [row for row in mcts_rows if row["selected_matches_teacher"]]
    return {
        "teacher_summary": teacher_summary(teacher),
        "rank_scope_steps": int(len(rows)),
        "rank_summary": {
            "ranked_steps": int(len(ranks)),
            "teacher_rank_mean": None if not ranks else float(mean(ranks)),
            "teacher_rank_median": None if not ranks else float(median(ranks)),
            "teacher_rank_max": None if not ranks else int(max(ranks)),
            "teacher_rank1_count": int(len(rank1)),
            "teacher_rank_le3_count": int(len(rank_le3)),
            "teacher_rank1_fraction": None if not ranks else float(len(rank1) / len(ranks)),
            "teacher_rank_le3_fraction": None if not ranks else float(len(rank_le3) / len(ranks)),
            "teacher_missing_count": int(sum(1 for row in rows if row["teacher_candidate_missing"])),
            "future_goal_steps": int(len(slot_available_rows)),
            "future_goal_option_available_count": int(
                sum(1 for row in slot_available_rows if row["future_goal_option_available"])
            ),
            "future_goal_slot_match_count": int(
                sum(1 for row in slot_available_rows if row["teacher_option_slot_match"] is True)
            ),
        },
        "gap_tag_counts": dict(sorted(tag_counts.items())),
        "corridor_runs": corridor_run_summary(states, actions),
        "goal_fill_events": goal_fill_events(model, states, actions, future_goals),
        "mcts_sample_summary": {
            "sample_stride": int(mcts_sample_stride),
            "sample_count": int(len(mcts_rows)),
            "selected_teacher_match_count": int(len(mcts_matches)),
            "selected_teacher_match_fraction": None if not mcts_rows else float(len(mcts_matches) / len(mcts_rows)),
        },
        "mcts_sample_rows": mcts_rows[:40],
        "gap_examples": {
            "teacher_low_rank_or_setup_prefix": setup_rows[:25],
            "top_action_not_teacher_prefix": [
                row for row in rows if not row["top_action_is_teacher"]
            ][:25],
        },
        "rank_rows_prefix": rows[:120],
    }


def infer_missing_sd_mechanisms(level_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    solved = [result for result in level_results if result["teacher"]["solved"]]
    aggregate_tags = Counter()
    long_runs = 0
    rank1_total = 0
    ranked_total = 0
    mcts_total = 0
    mcts_match = 0
    for result in solved:
        analysis = result["gap_analysis"]
        aggregate_tags.update(analysis["gap_tag_counts"])
        long_runs += int(analysis["corridor_runs"]["long_run_count_ge3"])
        rank1_total += int(analysis["rank_summary"]["teacher_rank1_count"])
        ranked_total += int(analysis["rank_summary"]["ranked_steps"])
        mcts_total += int(analysis["mcts_sample_summary"]["sample_count"])
        mcts_match += int(analysis["mcts_sample_summary"]["selected_teacher_match_count"])
    rank1_fraction = None if ranked_total == 0 else rank1_total / ranked_total
    mcts_fraction = None if mcts_total == 0 else mcts_match / mcts_total
    rank1_text = "NA" if rank1_fraction is None else f"{rank1_fraction:.3f}"
    mcts_text = "NA" if mcts_fraction is None else f"{mcts_fraction:.3f}"

    missing = [
        {
            "mechanism": "room_fill_plan_state",
            "diagnostic_basis": (
                "A7 解依赖 future-goal/slot trace，而 A6 在相同结构下不能自主生成后续槽位顺序。"
            ),
            "candidate_fix": "在 RoomCommitmentState 之外增加 room fill phase、frontier slot queue 或可更新的 fill-plan stock。",
        },
        {
            "mechanism": "box_release_and_setup_value",
            "diagnostic_basis": (
                f"gap tag 中 setup_or_release_underrewarded={aggregate_tags.get('setup_or_release_underrewarded', 0)}，"
                f"teacher_low_prior_rank={aggregate_tags.get('teacher_low_prior_rank', 0)}。"
            ),
            "candidate_fix": "增加释放后续箱子、保持通道和准备入房动作的正反馈，而不是只奖励即时 transport/slot cost 下降。",
        },
        {
            "mechanism": "autonomous_corridor_option_with_schedule_guard",
            "diagnostic_basis": f"teacher 轨迹中长度大于等于 3 的同向走廊段共有 {long_runs} 段。",
            "candidate_fix": "保留 A6 corridor macro，但由自主 fill plan 决定何时触发、触发哪个箱子和哪个 slot。",
        },
        {
            "mechanism": "planner_choice_calibration",
            "diagnostic_basis": (
                f"rank-1 fraction={rank1_text}；"
                f"MCTS sampled teacher-match fraction={mcts_text}。"
            ),
            "candidate_fix": "对 teacher 轨迹上 teacher action rank 低的位置生成 value/prior 审计样本，用于校准 A6 leaf 和 room prior。",
        },
    ]
    if aggregate_tags.get("short_transport_competes", 0) > 0:
        missing.append(
            {
                "mechanism": "long_horizon_transport_vs_schedule_tradeoff",
                "diagnostic_basis": f"short_transport_competes={aggregate_tags['short_transport_competes']}。",
                "candidate_fix": "将短期运输下降与后续 room-order 可行性分离计入价值，避免局部 P 下降覆盖全局调度风险。",
            }
        )
    return missing


def analyze_levels(args: argparse.Namespace) -> dict[str, Any]:
    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    levels = load_boxoban_levels(ROOT / config["levels"], max_levels=int(args.max_levels))
    level_results: list[dict[str, Any]] = []
    for level_index, level in enumerate(levels):
        model = _A7._A6._A5.build_model(level, config)
        teacher = teacher_trace_search_bounded(
            model,
            model.initial_state,
            max_expansions=int(args.teacher_max_expansions),
            max_seconds=float(args.teacher_max_seconds_per_level),
            heuristic_weight=float(args.teacher_heuristic_weight),
            use_viable_push_term=bool(args.teacher_use_viable_push_term),
            deadlock_mode=str(args.teacher_deadlock_mode),
            matching_backend=str(args.teacher_matching_backend),
        )
        result: dict[str, Any] = {
            "level_index": int(level_index),
            "level_id": level.level_id,
            "level_shape": {
                "height": int(level.height),
                "width": int(level.width),
                "n_boxes": int(level.n_boxes),
                "n_goals": int(len(level.goals)),
            },
            "teacher": teacher_summary(teacher),
        }
        if teacher["solved"]:
            result["gap_analysis"] = analyze_solved_teacher_trace(
                model,
                config,
                teacher,
                level_index=level_index,
                boundary_distance=int(args.boundary_distance),
                max_rank_states=int(args.max_rank_states_per_level),
                mcts_sample_stride=int(args.mcts_sample_stride),
                mcts_budget=int(args.mcts_budget),
            )
        else:
            result["gap_analysis"] = None
        level_results.append(result)
    solved_count = sum(1 for result in level_results if result["teacher"]["solved"])
    return {
        "contract": {
            "analysis": "room_gft_xsokoban_teacher_trace_gap_v1",
            "failure_log": _rel(failure_path),
            "base_model": "SokobanRoomGFTSDModel",
            "diagnosed_planner": RoomCorridorMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_used_for_online_solving": False,
            "teacher_used_for_gap_diagnosis": True,
        },
        "config_extract": {
            "levels": _rel(ROOT / config["levels"]),
            "max_levels": int(args.max_levels),
            "teacher_max_expansions": int(args.teacher_max_expansions),
            "teacher_max_seconds_per_level": float(args.teacher_max_seconds_per_level),
            "teacher_heuristic_weight": float(args.teacher_heuristic_weight),
            "teacher_use_viable_push_term": bool(args.teacher_use_viable_push_term),
            "teacher_deadlock_mode": str(args.teacher_deadlock_mode),
            "teacher_matching_backend": str(args.teacher_matching_backend),
            "boundary_distance": int(args.boundary_distance),
            "max_rank_states_per_level": int(args.max_rank_states_per_level),
            "mcts_sample_stride": int(args.mcts_sample_stride),
            "mcts_budget": int(args.mcts_budget),
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "summary": {
            "level_count": int(len(level_results)),
            "teacher_solved_count": int(solved_count),
            "teacher_unsolved_count": int(len(level_results) - solved_count),
            "solved_level_indices": [
                int(result["level_index"]) for result in level_results if result["teacher"]["solved"]
            ],
            "unsolved_level_indices": [
                int(result["level_index"]) for result in level_results if not result["teacher"]["solved"]
            ],
        },
        "level_results": level_results,
        "missing_sd_mechanisms": infer_missing_sd_mechanisms(level_results),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    config = payload["config_extract"]
    lines = [
        "# XSokoban teacher 轨迹驱动的 Room-GFT 缺失机制分析",
        "",
        "## 核心结论",
        "",
        (
            f"本次尝试分析 XSokoban first 前 `{summary['level_count']}` 关。"
            f"在本次 teacher 配置下解出 `{summary['teacher_solved_count']}` 关，"
            f"未解出 `{summary['teacher_unsolved_count']}` 关。"
        ),
        (
            f"teacher 配置为 matching backend `{config.get('teacher_matching_backend', 'dp')}`、"
            f"deadlock mode `{config.get('teacher_deadlock_mode', 'exact')}`、"
            f"heuristic weight `{_fmt(config.get('teacher_heuristic_weight', 1.0))}`、"
            f"viable term `{config.get('teacher_use_viable_push_term', True)}`、"
            f"每关最多 `{_fmt(config['teacher_max_seconds_per_level'])}` 秒或 "
            f"`{config['teacher_max_expansions']}` 个展开。"
        ),
        (
            "teacher 轨迹只用于诊断当前 Room-GFT/A6 评分层的缺口，不注入 planner 出手，"
            "也不作为无 teacher 求解率结果。"
        ),
        (
            "当前最稳定的结论是：A6 的主要缺失不是 exact transition 或 corridor macro 的执行能力，"
            "而是缺少可自主生成的 room fill plan、箱子释放/准备动作价值和多箱槽位调度状态。"
        ),
        "",
        "## Teacher 可用性",
        "",
        "| level | boxes | teacher status | pushes | expanded | generated | seconds |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for result in payload["level_results"]:
        teacher = result["teacher"]
        lines.append(
            "| "
            f"{result['level_index']} | {result['level_shape']['n_boxes']} | {teacher['failure_type']} | "
            f"{_fmt(teacher['solution_pushes'])} | {teacher['expanded']} | {teacher['generated']} | "
            f"{_fmt(teacher['seconds'])} |"
        )
    lines.extend(["", "## 已解 teacher 轨迹上的 A6 评分差距", ""])
    lines.extend(
        [
            "| level | rank scope | rank1 frac | rank<=3 frac | max rank | setup/release underrewarded | low rank | long corridor runs | MCTS match frac |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["level_results"]:
        analysis = result.get("gap_analysis")
        if analysis is None:
            continue
        rank = analysis["rank_summary"]
        tags = analysis["gap_tag_counts"]
        mcts = analysis["mcts_sample_summary"]
        lines.append(
            "| "
            f"{result['level_index']} | {analysis['rank_scope_steps']} | "
            f"{_fmt(rank['teacher_rank1_fraction'])} | {_fmt(rank['teacher_rank_le3_fraction'])} | "
            f"{_fmt(rank['teacher_rank_max'])} | "
            f"{tags.get('setup_or_release_underrewarded', 0)} | {tags.get('teacher_low_prior_rank', 0)} | "
            f"{analysis['corridor_runs']['long_run_count_ge3']} | "
            f"{_fmt(mcts['selected_teacher_match_fraction'])} |"
        )
    lines.extend(["", "## 缺失机制清单", ""])
    for item in payload["missing_sd_mechanisms"]:
        lines.extend(
            [
                f"### {item['mechanism']}",
                "",
                f"诊断依据：{item['diagnostic_basis']}",
                "",
                f"修订方向：{item['candidate_fix']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            (
                "若第 2、3 关在本次 teacher 时间预算内未解出，则多关综合结论目前主要来自已解关卡。"
                "这不是 A6 缺失不存在的证据，而是说明 teacher 侧也需要更强启发式、缓存或更高预算才能形成完整多关轨迹库。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure-log",
        default="results/sokoban_sd_world_model/xsokoban_first40_bestconfig_exact_l001_maxsteps1000_seed0.json",
    )
    parser.add_argument("--max-levels", type=int, default=3)
    parser.add_argument("--teacher-max-expansions", type=int, default=100000)
    parser.add_argument("--teacher-max-seconds-per-level", type=float, default=90.0)
    parser.add_argument("--teacher-heuristic-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-no-viable-push-term",
        action="store_false",
        dest="teacher_use_viable_push_term",
        default=True,
    )
    parser.add_argument("--teacher-deadlock-mode", choices=("exact", "tier_a", "none"), default="exact")
    parser.add_argument("--teacher-matching-backend", choices=("dp", "hungarian"), default="dp")
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--max-rank-states-per-level", type=int, default=120)
    parser.add_argument("--mcts-sample-stride", type=int, default=10)
    parser.add_argument("--mcts-budget", type=int, default=64)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_teacher_trace_gap.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_teacher_trace_gap.md")
    args = parser.parse_args()

    payload = analyze_levels(args)
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"out_json": _rel(out_json), "out_md": _rel(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
