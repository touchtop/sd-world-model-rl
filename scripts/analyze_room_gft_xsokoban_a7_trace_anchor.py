#!/usr/bin/env python3
"""A7 teacher-trace anchoring diagnostics for XSokoban first level.

The diagnostic first builds a local exact-state teacher path with parent
pointers, then supplies the recovered trace to a hard-instance subclass of the
A6 planner.  The experiment checks whether explicit reference scheduling is
enough to move beyond the A5/A6 no-action state.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A6_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a6_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a6_corridor_macro.py",
)
if _A6_SPEC is None or _A6_SPEC.loader is None:
    raise RuntimeError("Unable to load A6 diagnostic helpers.")
_A6 = importlib.util.module_from_spec(_A6_SPEC)
_A6_SPEC.loader.exec_module(_A6)

from src.sokoban import (  # noqa: E402
    RoomCommitmentState,
    RoomCorridorMacroGFTSDMCTSPlanner,
    RoomTraceAnchoredGFTSDMCTSPlanner,
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


def action_label(action: SokobanPushAction) -> str:
    return f"b{action.box_index}:{ACTION_NAMES[action.direction]}"


def action_dict(state: SokobanState, action: SokobanPushAction) -> dict[str, Any]:
    source = state.boxes[action.box_index]
    target = add_cells(source, DIRECTIONS[action.direction])
    return {
        "box_index": int(action.box_index),
        "direction": int(action.direction),
        "direction_name": ACTION_NAMES[action.direction],
        "flat_index": int(action.flat_index),
        "source": list(source),
        "target": list(target),
    }


def teacher_trace_search(
    model: SokobanRoomGFTSDModel,
    start: SokobanState,
    *,
    max_expansions: int,
) -> dict[str, Any]:
    """Run the local exact-state teacher and reconstruct its push path."""

    legal_cache: dict[SokobanState, tuple[SokobanPushAction, ...]] = {}
    deadlock_cache: dict[SokobanState, bool] = {}
    heuristic_cache: dict[SokobanState, float] = {}

    def legal_pushes(state: SokobanState) -> tuple[SokobanPushAction, ...]:
        cached = legal_cache.get(state)
        if cached is None:
            cached = model.legal_pushes(state)
            legal_cache[state] = cached
        return cached

    def is_deadlocked(state: SokobanState) -> bool:
        cached = deadlock_cache.get(state)
        if cached is None:
            cached = model.is_deadlocked(state)
            deadlock_cache[state] = cached
        return cached

    def heuristic(state: SokobanState) -> float:
        cached = heuristic_cache.get(state)
        if cached is not None:
            return cached
        lb = model.matching_lower_bound(state.boxes)
        if lb == float("inf"):
            value = 1e6
        else:
            viable = 0
            for action in legal_pushes(state):
                child = model.step_state(state, action)
                if not is_deadlocked(child):
                    viable += 1
            value = float(lb - 0.25 * model.boxes_on_goals(state) - 0.01 * viable)
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
            "cache_sizes": {"heuristic": 0, "legal": 0, "deadlock": 0},
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
            "cache_sizes": {"heuristic": 0, "legal": 0, "deadlock": 0},
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

    while frontier and expanded < max_expansions:
        _priority, g, _counter, state = heapq.heappop(frontier)
        if g != best_g.get(state):
            continue
        if model.is_solved(state):
            solved_state = state
            solved_depth = g
            break
        expanded += 1
        for action in legal_pushes(state):
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
        "failure_type": "solved" if solved_state is not None else "teacher_budget_unsolved",
        "seconds": float(perf_counter() - started_at),
        "actions": actions,
        "states": states,
        "best_seen": int(len(best_g)),
        "cache_sizes": {
            "heuristic": int(len(heuristic_cache)),
            "legal": int(len(legal_cache)),
            "deadlock": int(len(deadlock_cache)),
        },
    }


def infer_future_goal_cells(
    model: SokobanRoomGFTSDModel,
    states: list[SokobanState],
    actions: list[SokobanPushAction],
) -> list[tuple[int, int] | None]:
    goals = set(model.level.goals)
    future_cells: list[tuple[int, int] | None] = []
    for index, action in enumerate(actions):
        source = states[index].boxes[action.box_index]
        current = add_cells(source, DIRECTIONS[action.direction])
        for later in range(index + 1, len(actions)):
            later_action = actions[later]
            later_source = states[later].boxes[later_action.box_index]
            if later_source == current:
                current = add_cells(later_source, DIRECTIONS[later_action.direction])
        future_cells.append(current if current in goals else None)
    return future_cells


def build_reference_rows(
    model: SokobanRoomGFTSDModel,
    states: list[SokobanState],
    actions: list[SokobanPushAction],
) -> tuple[dict[tuple, dict[str, Any]], list[dict[str, Any]]]:
    future_goals = infer_future_goal_cells(model, states, actions)
    by_key: dict[tuple, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for step, (state, action) in enumerate(zip(states, actions)):
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        future_goal = future_goals[step]
        row = {
            "step": int(step),
            "box_index": int(action.box_index),
            "direction": int(action.direction),
            "direction_name": ACTION_NAMES[action.direction],
            "flat_index": int(action.flat_index),
            "source": list(source),
            "target": list(target),
            "future_goal_cell": None if future_goal is None else list(future_goal),
            "boxes_on_goals_before": int(model.boxes_on_goals(state)),
            "boxes_on_goals_after": int(model.boxes_on_goals(states[step + 1])),
        }
        rows.append(row)
        by_key.setdefault(model.gft_canonical_key_cached(state), row)
    return by_key, rows


def build_a7_planner(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    boundary_distance: int,
    reference_action_by_key: dict[tuple, dict[str, Any]],
    previous_physical_action: SokobanPushAction | None = None,
    previous_moved_source=None,
    previous_moved_target=None,
    recent_commitment_keys: set | None = None,
    corridor_macro_max_extra_pushes: int = 8,
    corridor_macro_min_total_pushes: int = 2,
):
    return RoomTraceAnchoredGFTSDMCTSPlanner(
        model,
        horizon=horizon,
        budget=budget,
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
        corridor_macro_max_extra_pushes=int(corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(corridor_macro_min_total_pushes),
        reference_action_by_state_key=reference_action_by_key,
    )


def _chosen_child(plan: dict[str, Any]) -> dict[str, Any]:
    semantic_action = plan.get("semantic_action")
    if semantic_action is None:
        return {}
    semantic_key = semantic_action.key
    for child in plan.get("children", []):
        action = child["semantic_action"]
        if tuple(action["semantic_key"][1]) == tuple(semantic_key[1]) and int(action["semantic_key"][0]) == int(
            semantic_key[0]
        ):
            return child
    return plan["children"][0] if plan.get("children") else {}


def run_a7_smoke(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    reference_action_by_key: dict[tuple, dict[str, Any]],
    reference_terminal_key: tuple,
    budget: int,
    horizon: int,
    max_pushes: int,
    boundary_distance: int,
    seed_offset: int,
    corridor_macro_max_extra_pushes: int,
    corridor_macro_min_total_pushes: int,
) -> dict[str, Any]:
    rng = random.Random(int(config["seed"]) + seed_offset)
    state = model.initial_state
    commitment_state: RoomCommitmentState = model.empty_room_commitment_state()
    visited_keys = {model.canonical_key(state)}
    recent_commitment_keys = {model.room_commitment_key(state, commitment_state)}
    previous_action: SokobanPushAction | None = None
    previous_source = None
    previous_target = None
    trace: list[dict[str, Any]] = []
    status = "max_steps"
    executed_pushes = 0

    for decision_step in range(max_pushes):
        if executed_pushes >= max_pushes:
            status = "max_steps"
            break
        if model.is_solved(state):
            status = "solved"
            break
        reference_before = reference_action_by_key.get(model.gft_canonical_key_cached(state))
        planner = build_a7_planner(
            model,
            config,
            rng=rng,
            budget=budget,
            horizon=horizon,
            visited_keys=visited_keys,
            boundary_distance=boundary_distance,
            reference_action_by_key=reference_action_by_key,
            previous_physical_action=previous_action,
            previous_moved_source=previous_source,
            previous_moved_target=previous_target,
            recent_commitment_keys=recent_commitment_keys,
            corridor_macro_max_extra_pushes=corridor_macro_max_extra_pushes,
            corridor_macro_min_total_pushes=corridor_macro_min_total_pushes,
        )
        plan = planner.plan_with_corridor_macro(state, commitment_state)
        physical = plan["physical_action"]
        if physical is None:
            status = "no_action"
            trace.append(
                {
                    "step": int(decision_step),
                    "status": "no_action",
                    "root_visits": int(plan["root_visits"]),
                    "root_value": float(plan["root_value"]),
                    "executed_pushes_so_far": int(executed_pushes),
                    "reference_before": reference_before,
                }
            )
            break

        macro = plan["corridor_macro"]
        macro_actions = list(macro["physical_actions"])
        macro_commitments = list(macro["after_commitment_states"])
        if executed_pushes + len(macro_actions) > max_pushes:
            allowed = max(1, max_pushes - executed_pushes)
            macro_actions = macro_actions[:allowed]
            macro_commitments = macro_commitments[:allowed]
            macro = {
                **macro,
                "triggered": len(macro_actions) >= int(corridor_macro_min_total_pushes),
                "reason": "truncated_by_push_budget",
                "physical_actions": macro_actions,
                "after_commitment_states": macro_commitments,
                "macro_push_count": len(macro_actions),
                "extra_push_count": max(0, len(macro_actions) - 1),
            }

        before_commitment_state = commitment_state
        first_source = None
        first_target = None
        executed_rows = []
        reference_matches = 0
        for macro_index, action in enumerate(macro_actions):
            reference_row = reference_action_by_key.get(model.gft_canonical_key_cached(state))
            source, target = _A6._A5.moved_cells(state, action)
            if macro_index == 0:
                first_source, first_target = source, target
            next_state = model.step_grid_flow(state, action)
            if macro_index < len(macro_commitments):
                commitment_state = macro_commitments[macro_index]
            else:
                commitment_state = model.normalize_room_commitment_state(next_state, commitment_state)
            reference_match = bool(reference_row and int(reference_row["flat_index"]) == int(action.flat_index))
            reference_matches += int(reference_match)
            executed_rows.append(
                {
                    "macro_index": int(macro_index),
                    "action": action_dict(state, action),
                    "reference_step": None if reference_row is None else int(reference_row["step"]),
                    "reference_action_match": bool(reference_match),
                    "boxes_on_goals_after": int(model.boxes_on_goals(next_state)),
                    "slot_schedule_cost_after": float(
                        model.room_slot_progress_summary(next_state, commitment_state)["slot_schedule_cost"]
                    ),
                }
            )
            state = next_state
            executed_pushes += 1
            visited_keys.add(model.canonical_key(state))
            recent_commitment_keys.add(model.room_commitment_key(state, commitment_state))
            previous_action = action
            previous_source = source
            previous_target = target
            if model.is_solved(state):
                status = "solved"
                break
            if model.is_deadlocked(state) and not model.is_solved(state):
                status = "deadlocked"
                break

        ex_post_loop = _A6._A5.transition_loop_summary(
            model,
            before_commitment_state,
            commitment_state,
            first_source,
            first_target,
            None,
            None,
            recent_commitment_keys,
            state,
        ) if first_source is not None and first_target is not None else {}
        chosen_child = _chosen_child(plan)
        chosen_loop = chosen_child.get("room_loop") or ex_post_loop
        reference_after = reference_action_by_key.get(model.gft_canonical_key_cached(state))
        trace.append(
            {
                "step": int(decision_step),
                "physical_action": {
                    "box_index": int(physical.box_index),
                    "direction": int(physical.direction),
                    "direction_name": ACTION_NAMES[physical.direction],
                    "flat_index": int(physical.flat_index),
                },
                "semantic_action": plan["semantic_action"].to_dict(),
                "root_visits": int(plan["root_visits"]),
                "root_value": float(plan["root_value"]),
                "physical_child_count": int(plan["physical_child_count"]),
                "semantic_child_count": int(plan["semantic_child_count"]),
                "top_children": plan["children"][:5],
                "chosen_loop": chosen_loop,
                "chosen_loop_penalty_score": float(chosen_loop.get("score", 0.0)) if chosen_loop else 0.0,
                "chosen_trace_anchor": chosen_child.get("room_trace_anchor", {}),
                "corridor_macro": {
                    "triggered": bool(macro.get("triggered")),
                    "reason": macro.get("reason"),
                    "macro_push_count": int(macro.get("macro_push_count", len(macro_actions))),
                    "extra_push_count": int(macro.get("extra_push_count", max(0, len(macro_actions) - 1))),
                    "reference_macro_checked": bool(macro.get("reference_macro_checked", False)),
                    "reference_macro_match_count": int(macro.get("reference_macro_match_count", 0)),
                    "reference_macro_trimmed": bool(macro.get("reference_macro_trimmed", False)),
                    "rows": macro.get("rows", []),
                },
                "reference": {
                    "before": reference_before,
                    "after": reference_after,
                    "first_action_match": bool(
                        reference_before and int(reference_before["flat_index"]) == int(physical.flat_index)
                    ),
                    "macro_action_match_count": int(reference_matches),
                    "macro_action_count": int(len(macro_actions)),
                },
                "executed_push_count": int(len(macro_actions)),
                "executed_rows": executed_rows,
                "executed_pushes_so_far": int(executed_pushes),
                "boxes_on_goals_after": int(model.boxes_on_goals(state)),
                "commitment_after": commitment_state.to_dict(),
                "slot_progress_after": model.room_slot_progress_summary(state, commitment_state),
                "room_summary_after": model.room_state_summary(state),
            }
        )
        if status in {"solved", "deadlocked"}:
            break

    if model.is_solved(state):
        status = "solved"
    summary = _A6.summarise_macro_trace(trace)
    reference_rows = [row for row in trace if "physical_action" in row]
    reference_hits = [row for row in reference_rows if row.get("reference", {}).get("before") is not None]
    reference_first_matches = [
        row for row in reference_rows if row.get("reference", {}).get("first_action_match")
    ]
    reference_macro_trims = [
        row for row in reference_rows if row.get("corridor_macro", {}).get("reference_macro_trimmed")
    ]
    summary.update(
        {
            "status": status,
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "final_commitment_count": len(commitment_state.commitments),
            "final_slot_schedule_cost": float(
                model.room_slot_progress_summary(state, commitment_state)["slot_schedule_cost"]
            ),
            "reference_state_hit_decisions": int(len(reference_hits)),
            "reference_first_action_match_decisions": int(len(reference_first_matches)),
            "reference_macro_trimmed_decisions": int(len(reference_macro_trims)),
            "final_on_reference_terminal": bool(model.gft_canonical_key_cached(state) == reference_terminal_key),
        }
    )
    return {
        "summary": summary,
        "trace": trace,
        "final_room_summary": model.room_state_summary(state),
        "final_commitment_summary": model.room_commitment_summary(state, commitment_state),
        "final_slot_progress": model.room_slot_progress_summary(state, commitment_state),
        "final_tier_a_status": model.tier_a_status(state),
    }


def compare_a6_a7(a6: dict[str, Any], a7: dict[str, Any]) -> dict[str, Any]:
    a6_summary = a6["summary"]
    a7_summary = a7["summary"]
    return {
        "a6_status": a6_summary["status"],
        "a7_status": a7_summary["status"],
        "a6_executed_pushes": int(a6_summary["executed_pushes"]),
        "a7_executed_pushes": int(a7_summary["executed_pushes"]),
        "a6_boxes_on_goals_final": int(a6_summary["boxes_on_goals_final"]),
        "a7_boxes_on_goals_final": int(a7_summary["boxes_on_goals_final"]),
        "a6_final_slot_schedule_cost": float(a6_summary["final_slot_schedule_cost"]),
        "a7_final_slot_schedule_cost": float(a7_summary["final_slot_schedule_cost"]),
        "a7_reference_state_hit_decisions": int(a7_summary["reference_state_hit_decisions"]),
        "a7_reference_first_action_match_decisions": int(a7_summary["reference_first_action_match_decisions"]),
        "a7_final_on_reference_terminal": bool(a7_summary["final_on_reference_terminal"]),
    }


def teacher_summary_for_json(teacher: dict[str, Any]) -> dict[str, Any]:
    return {
        "solved": bool(teacher["solved"]),
        "solution_pushes": teacher["solution_pushes"],
        "expanded": int(teacher["expanded"]),
        "generated": int(teacher["generated"]),
        "pruned_deadlocks": int(teacher["pruned_deadlocks"]),
        "frontier_size": int(teacher["frontier_size"]),
        "failure_type": teacher["failure_type"],
        "seconds": float(teacher["seconds"]),
        "best_seen": int(teacher.get("best_seen", 0)),
        "cache_sizes": teacher.get("cache_sizes", {}),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    teacher = payload["teacher_trace"]["summary"]
    a6 = payload["a6_closed_loop_smoke"]["summary"]
    a7 = payload["a7_trace_anchored_smoke"]["summary"]
    lines = [
        "# XSokoban first 第 1 关 Room-GFT A7 trace-anchor 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A7 在 A6 forced-corridor macro 基础上加入 teacher/reference trace anchor。"
            "该 anchor 只在当前状态命中参考轨迹时生效，对参考物理动作和该箱子未来目标槽位加 prior，"
            "并在根节点优先选择参考物理动作。"
        ),
        (
            f"本地 exact-state teacher 在 `{teacher['expanded']}` 个展开、"
            f"`{_fmt(teacher['seconds'])}` 秒内解出该关，参考解长度为 "
            f"`{teacher['solution_pushes']}` 个 push。"
        ),
        (
            f"A6 smoke 为 `{a6['status']}`，执行 `{a6['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a6['boxes_on_goals_final']}`；"
            f"A7 trace-anchor smoke 为 `{a7['status']}`，执行 `{a7['executed_pushes']}` 个 push、"
            f"`{a7['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a7['boxes_on_goals_final']}`。"
        ),
        (
            "该结果用于机制定位：若参考调度被显式锚定，当前 exact transition、room-slot commitment "
            "和 corridor macro 可以沿可行轨迹推进；A5/A6 的失败主要来自多箱调度/槽位顺序选择，"
            "而不是单步转移或步数上限本身。A7 不是无 teacher 的通用求解率结果。"
        ),
        "",
        "## A6/A7 摘要",
        "",
        "| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | macro triggers | reference hits | reference matches |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A6 | {a6['status']} | {a6['closed_loop_decision_steps']} | {a6['executed_pushes']} | "
            f"{a6['boxes_on_goals_final']} | {_fmt(a6['final_slot_schedule_cost'])} | "
            f"{a6['macro_trigger_steps']} | 0 | 0 |"
        ),
        (
            f"| A7 | {a7['status']} | {a7['closed_loop_decision_steps']} | {a7['executed_pushes']} | "
            f"{a7['boxes_on_goals_final']} | {_fmt(a7['final_slot_schedule_cost'])} | "
            f"{a7['macro_trigger_steps']} | {a7['reference_state_hit_decisions']} | "
            f"{a7['reference_first_action_match_decisions']} |"
        ),
        "",
        "## A7 轨迹前缀",
        "",
        "| step | action | executed pushes | macro | reference step | reference match | goals | slot cost |",
        "|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for row in payload["a7_trace_anchored_smoke"]["trace"][:40]:
        if "physical_action" not in row:
            lines.append(f"| {row['step']} | no_action | 0 | False | NA | False | NA | NA |")
            continue
        ref_before = row.get("reference", {}).get("before")
        ref_step = None if ref_before is None else ref_before.get("step")
        macro = row.get("corridor_macro", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {macro.get('triggered', False)} | "
            f"{_fmt(ref_step)} | {row.get('reference', {}).get('first_action_match', False)} | "
            f"{row['boxes_on_goals_after']} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A7 把 teacher 调用限定为一次离线参考轨迹生成，teacher 不参与 Boxoban 主线、A6 基线或在线泛化评测。"
                "因此，A7 的意义是定位失败机制和验证修复方向：后续若要形成无参考解版本，"
                "需要把 trace anchor 中暴露出的多箱 slot schedule 转化为可自主生成的 room fill plan、"
                "释放动作约束或价值审计目标。"
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
    parser.add_argument("--teacher-max-expansions", type=int, default=100000)
    parser.add_argument("--a6-budget", type=int, default=256)
    parser.add_argument("--a7-budget", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--a6-smoke-steps", type=int, default=40)
    parser.add_argument("--a7-max-pushes", type=int, default=120)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a7_trace_anchor.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a7_trace_anchor.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = _A6._A5.build_model(level, config)
    teacher = teacher_trace_search(
        model,
        model.initial_state,
        max_expansions=int(args.teacher_max_expansions),
    )
    if not teacher["solved"]:
        raise SystemExit(
            "Teacher trace was not found within budget; A7 trace-anchor smoke requires a reference path."
        )
    reference_action_by_key, reference_rows = build_reference_rows(model, teacher["states"], teacher["actions"])
    reference_terminal_key = model.gft_canonical_key_cached(teacher["states"][-1])

    a6 = _A6.run_a6_smoke(
        model,
        config,
        budget=int(args.a6_budget),
        horizon=int(args.horizon),
        max_steps=int(args.a6_smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=27059,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
    )
    a7 = run_a7_smoke(
        model,
        config,
        reference_action_by_key=reference_action_by_key,
        reference_terminal_key=reference_terminal_key,
        budget=int(args.a7_budget),
        horizon=int(args.horizon),
        max_pushes=int(args.a7_max_pushes),
        boundary_distance=int(args.boundary_distance),
        seed_offset=28057,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a7_trace_anchor_v1",
            "failure_log": _rel(failure_path),
            "base_model": "SokobanRoomGFTSDModel",
            "a6_planner": RoomCorridorMacroGFTSDMCTSPlanner.provider_id,
            "a7_planner": RoomTraceAnchoredGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_trace_used": True,
            "teacher_calls": 1,
            "trace_scope": "a6_baseline_vs_a7_teacher_trace_anchor_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "a6_budget": int(args.a6_budget),
            "a7_budget": int(args.a7_budget),
            "a6_smoke_steps": int(args.a6_smoke_steps),
            "a7_max_pushes": int(args.a7_max_pushes),
            "teacher_max_expansions": int(args.teacher_max_expansions),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "corridor_macro_max_extra_pushes": int(args.corridor_macro_max_extra_pushes),
            "corridor_macro_min_total_pushes": int(args.corridor_macro_min_total_pushes),
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": model.room_static_summary(),
        "teacher_trace": {
            "summary": teacher_summary_for_json(teacher),
            "reference_rows_prefix": reference_rows[:120],
            "reference_row_count": int(len(reference_rows)),
        },
        "a6_closed_loop_smoke": a6,
        "a7_trace_anchored_smoke": a7,
        "comparison": compare_a6_a7(a6, a7),
    }
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"out_json": _rel(out_json), "out_md": _rel(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
