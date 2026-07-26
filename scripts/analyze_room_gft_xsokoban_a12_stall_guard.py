#!/usr/bin/env python3
"""A12 dependency-stall loop guard diagnostics for XSokoban first levels.

A12 keeps the A11 schedule-plan SD representation and adds a consumer-side
guard for repeated non-improving dependency states.  No teacher actions or
reference traces are used online.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A11_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a11_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a11_schedule_plan.py",
)
if _A11_SPEC is None or _A11_SPEC.loader is None:
    raise RuntimeError("Unable to load A11 diagnostic helpers.")
_A11 = importlib.util.module_from_spec(_A11_SPEC)
_A11_SPEC.loader.exec_module(_A11)

from src.sokoban import (  # noqa: E402
    RoomCommitmentState,
    RoomSchedulePlanGFTSDMCTSPlanner,
    RoomScheduleStallGuardGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)
from src.sokoban.env import ACTION_NAMES  # noqa: E402


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


def schedule_signature(
    model: SokobanRoomSchedulePlanGFTSDModel,
    state,
    commitment_state: RoomCommitmentState,
    *,
    lookahead_slots: int,
) -> tuple:
    schedule = model.room_schedule_plan_summary(
        state,
        commitment_state,
        lookahead_slots=lookahead_slots,
    )
    slot = model.room_slot_progress_summary(state, commitment_state)
    totals = schedule["totals"]
    return (
        int(model.boxes_on_goals(state)),
        int(round(float(slot["slot_schedule_cost"]) * 1000.0)),
        int(totals["dependency_count"]),
        int(totals["future_dependency_count"]),
        int(totals["critical_stand_box_count"]),
        int(totals["release_action_count"]),
    )


def build_model(level, config: dict[str, Any]) -> SokobanRoomSchedulePlanGFTSDModel:
    return SokobanRoomSchedulePlanGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def build_a12_planner(
    model: SokobanRoomSchedulePlanGFTSDModel,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    boundary_distance: int,
    recent_moved_edges: list[tuple],
    recent_schedule_keys: set[tuple],
    previous_physical_action: SokobanPushAction | None = None,
    previous_moved_source=None,
    previous_moved_target=None,
    recent_commitment_keys: set | None = None,
    corridor_macro_max_extra_pushes: int = 8,
    corridor_macro_min_total_pushes: int = 2,
    schedule_lookahead_slots: int = 2,
) -> RoomScheduleStallGuardGFTSDMCTSPlanner:
    return RoomScheduleStallGuardGFTSDMCTSPlanner(
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
        room_schedule_lookahead_slots=int(schedule_lookahead_slots),
        recent_moved_edges=recent_moved_edges,
        recent_schedule_keys=recent_schedule_keys,
    )


def summarise_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [row for row in trace if "physical_action" in row]
    macro_rows = [row for row in decision_rows if row.get("corridor_macro", {}).get("triggered")]
    macro_push_counts = [int(row.get("corridor_macro", {}).get("macro_push_count", 1)) for row in macro_rows]
    stall_scores = [float(row.get("chosen_schedule_stall_score", 0.0)) for row in decision_rows]
    stall_rows = [
        row.get("chosen_schedule_stall", {})
        for row in decision_rows
    ]
    return {
        "closed_loop_decision_steps": int(len(decision_rows)),
        "executed_pushes": int(sum(int(row.get("executed_push_count", 1)) for row in decision_rows)),
        "macro_trigger_steps": int(len(macro_rows)),
        "macro_total_extra_pushes": int(sum(max(0, count - 1) for count in macro_push_counts)),
        "macro_push_count_median": None if not macro_push_counts else float(median(macro_push_counts)),
        "macro_push_count_max": None if not macro_push_counts else int(max(macro_push_counts)),
        "schedule_stall_score_median": None if not stall_scores else float(median(stall_scores)),
        "schedule_stall_score_min": None if not stall_scores else float(min(stall_scores)),
        "schedule_stall_score_max": None if not stall_scores else float(max(stall_scores)),
        "selected_stalled_count": int(sum(bool(row.get("stalled")) for row in stall_rows)),
        "selected_reverse_recent_edge_count": int(sum(bool(row.get("reverse_recent_edge")) for row in stall_rows)),
        "selected_repeated_signature_count": int(
            sum(bool(row.get("repeated_schedule_signature")) for row in stall_rows)
        ),
    }


def run_a12_smoke(
    model: SokobanRoomSchedulePlanGFTSDModel,
    config: dict[str, Any],
    *,
    budget: int,
    horizon: int,
    max_steps: int,
    boundary_distance: int,
    seed_offset: int,
    corridor_macro_max_extra_pushes: int,
    corridor_macro_min_total_pushes: int,
    schedule_lookahead_slots: int,
    recent_edge_window: int,
) -> dict[str, Any]:
    rng = random.Random(int(config["seed"]) + seed_offset)
    state = model.initial_state
    commitment_state: RoomCommitmentState = model.empty_room_commitment_state()
    visited_keys = {model.canonical_key(state)}
    recent_commitment_keys = {model.room_commitment_key(state, commitment_state)}
    recent_moved_edges: deque[tuple] = deque(maxlen=max(int(recent_edge_window), 1))
    recent_schedule_keys = {
        schedule_signature(
            model,
            state,
            commitment_state,
            lookahead_slots=schedule_lookahead_slots,
        )
    }
    previous_action: SokobanPushAction | None = None
    previous_source = None
    previous_target = None
    trace: list[dict[str, Any]] = []
    status = "max_steps"
    executed_push_budget = int(max_steps)
    executed_pushes = 0

    for decision_step in range(max_steps):
        if executed_pushes >= executed_push_budget:
            status = "max_steps"
            break
        if model.is_solved(state):
            status = "solved"
            break
        planner = build_a12_planner(
            model,
            config,
            rng=rng,
            budget=budget,
            horizon=horizon,
            visited_keys=visited_keys,
            boundary_distance=boundary_distance,
            previous_physical_action=previous_action,
            previous_moved_source=previous_source,
            previous_moved_target=previous_target,
            recent_commitment_keys=recent_commitment_keys,
            recent_moved_edges=list(recent_moved_edges),
            recent_schedule_keys=recent_schedule_keys,
            corridor_macro_max_extra_pushes=corridor_macro_max_extra_pushes,
            corridor_macro_min_total_pushes=corridor_macro_min_total_pushes,
            schedule_lookahead_slots=schedule_lookahead_slots,
        )
        plan = planner.plan_with_corridor_macro(state, commitment_state)
        physical = plan["physical_action"]
        if physical is None:
            status = "no_action"
            trace.append(
                {
                    "step": decision_step,
                    "status": "no_action",
                    "root_visits": int(plan["root_visits"]),
                    "root_value": float(plan["root_value"]),
                    "executed_pushes_so_far": int(executed_pushes),
                    "schedule_plan": model.room_schedule_plan_summary(
                        state,
                        commitment_state,
                        lookahead_slots=schedule_lookahead_slots,
                    ),
                }
            )
            break

        macro = plan["corridor_macro"]
        macro_actions = list(macro["physical_actions"])
        macro_commitments = list(macro["after_commitment_states"])
        if executed_pushes + len(macro_actions) > executed_push_budget:
            allowed = max(1, executed_push_budget - executed_pushes)
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

        first_source = None
        first_target = None
        executed_rows = []
        for macro_index, action in enumerate(macro_actions):
            source, target = _A11._A10._A9._A8._A6._A5.moved_cells(state, action)
            if macro_index == 0:
                first_source, first_target = source, target
            next_state = model.step_grid_flow(state, action)
            if macro_index < len(macro_commitments):
                commitment_state = macro_commitments[macro_index]
            else:
                commitment_state = model.normalize_room_commitment_state(next_state, commitment_state)
            schedule_after_step = model.room_schedule_plan_summary(
                next_state,
                commitment_state,
                lookahead_slots=schedule_lookahead_slots,
            )
            executed_rows.append(
                {
                    "macro_index": int(macro_index),
                    "action": {
                        "box_index": int(action.box_index),
                        "direction": int(action.direction),
                        "direction_name": ACTION_NAMES[action.direction],
                        "flat_index": int(action.flat_index),
                        "source": list(source),
                        "target": list(target),
                    },
                    "boxes_on_goals_after": int(model.boxes_on_goals(next_state)),
                    "slot_schedule_cost_after": float(
                        model.room_slot_progress_summary(next_state, commitment_state)["slot_schedule_cost"]
                    ),
                    "schedule_barrier_after": float(schedule_after_step["totals"]["schedule_barrier_cost"]),
                    "dependency_count_after": int(schedule_after_step["totals"]["dependency_count"]),
                }
            )
            state = next_state
            executed_pushes += 1
            visited_keys.add(model.canonical_key(state))
            recent_commitment_keys.add(model.room_commitment_key(state, commitment_state))
            recent_moved_edges.append((tuple(source), tuple(target)))
            recent_schedule_keys.add(
                schedule_signature(
                    model,
                    state,
                    commitment_state,
                    lookahead_slots=schedule_lookahead_slots,
                )
            )
            previous_action = action
            previous_source = source
            previous_target = target
            if model.is_solved(state):
                status = "solved"
                break
            if model.is_deadlocked(state) and not model.is_solved(state):
                status = "deadlocked"
                break

        chosen_child = None
        semantic_key = plan["semantic_action"].key
        for child in plan["children"]:
            action = child["semantic_action"]
            if tuple(action["semantic_key"][1]) == tuple(semantic_key[1]) and int(action["semantic_key"][0]) == int(
                semantic_key[0]
            ):
                chosen_child = child
                break
        if chosen_child is None:
            chosen_child = plan["children"][0] if plan["children"] else {}
        schedule_after = model.room_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=schedule_lookahead_slots,
        )
        trace.append(
            {
                "step": decision_step,
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
                "chosen_schedule_plan": chosen_child.get("room_schedule_plan", {}),
                "chosen_schedule_plan_score": float(chosen_child.get("room_schedule_plan_score", 0.0)),
                "chosen_schedule_stall": chosen_child.get("room_schedule_stall", {}),
                "chosen_schedule_stall_score": float(chosen_child.get("room_schedule_stall_score", 0.0)),
                "corridor_macro": {
                    "triggered": bool(macro.get("triggered")),
                    "reason": macro.get("reason"),
                    "macro_push_count": int(macro.get("macro_push_count", len(macro_actions))),
                    "extra_push_count": int(macro.get("extra_push_count", max(0, len(macro_actions) - 1))),
                    "rows": macro.get("rows", []),
                },
                "executed_push_count": int(len(macro_actions)),
                "executed_rows": executed_rows,
                "executed_pushes_so_far": int(executed_pushes),
                "boxes_on_goals_after": int(model.boxes_on_goals(state)),
                "commitment_after": commitment_state.to_dict(),
                "slot_progress_after": model.room_slot_progress_summary(state, commitment_state),
                "schedule_plan_after": schedule_after,
                "first_moved_source": None if first_source is None else list(first_source),
                "first_moved_target": None if first_target is None else list(first_target),
            }
        )
        if status in {"solved", "deadlocked"}:
            break

    if model.is_solved(state):
        status = "solved"
    final_schedule = model.room_schedule_plan_summary(
        state,
        commitment_state,
        lookahead_slots=schedule_lookahead_slots,
    )
    final_slot = model.room_slot_progress_summary(state, commitment_state)
    summary = summarise_trace(trace)
    summary.update(
        {
            "status": status,
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "final_commitment_count": len(commitment_state.commitments),
            "final_slot_schedule_cost": float(final_slot["slot_schedule_cost"]),
            "final_schedule_barrier_cost": float(final_schedule["totals"]["schedule_barrier_cost"]),
            "final_dependency_count": int(final_schedule["totals"]["dependency_count"]),
            "final_future_dependency_count": int(final_schedule["totals"]["future_dependency_count"]),
            "final_release_action_count": int(final_schedule["totals"]["release_action_count"]),
            "final_destructive_action_count": int(final_schedule["totals"]["destructive_action_count"]),
            "recent_schedule_key_count": int(len(recent_schedule_keys)),
        }
    )
    return {
        "summary": summary,
        "trace": trace,
        "final_slot_progress": final_slot,
        "final_schedule_plan": final_schedule,
        "final_tier_a_status": model.tier_a_status(state),
    }


def compare_a11_a12(a11: dict[str, Any], a12: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a11_actions = actions(a11)
    a12_actions = actions(a12)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a11_actions, a12_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a11_actions) != len(a12_actions):
        first_diff = min(len(a11_actions), len(a12_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a11_status": a11["summary"]["status"],
        "a12_status": a12["summary"]["status"],
        "a11_boxes_on_goals_final": a11["summary"]["boxes_on_goals_final"],
        "a12_boxes_on_goals_final": a12["summary"]["boxes_on_goals_final"],
        "a11_executed_pushes": a11["summary"]["executed_pushes"],
        "a12_executed_pushes": a12["summary"]["executed_pushes"],
        "a11_final_slot_schedule_cost": a11["summary"]["final_slot_schedule_cost"],
        "a12_final_slot_schedule_cost": a12["summary"]["final_slot_schedule_cost"],
        "a11_prefix_actions": a11_actions[:40],
        "a12_prefix_actions": a12_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a11 = payload["a11_closed_loop_smoke"]["summary"]
    a12 = payload["a12_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A12 dependency-stall guard 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A12 保留 A11 的 schedule-plan SD 表示，只在 MCTS 消费端加入 dependency-stall loop guard。"
            "该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定。"
        ),
        (
            f"A11 guarded smoke 为 `{a11['status']}`，执行 `{a11['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a11['boxes_on_goals_final']}`；"
            f"A12 smoke 为 `{a12['status']}`，执行 `{a12['executed_pushes']}` 个 push、"
            f"`{a12['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a12['boxes_on_goals_final']}`。"
        ),
        (
            f"A11/A12 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A12 最终 slot cost 为 `{_fmt(a12['final_slot_schedule_cost'])}`，"
            f"dependency count 为 `{a12['final_dependency_count']}`，"
            f"selected stalled count 为 `{a12['selected_stalled_count']}`。"
        ),
        "",
        "## A11/A12 摘要",
        "",
        "| planner | status | decision steps | pushes | goals | final slot cost | dependencies | stalled selections | reverse recent | repeated signatures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A11 | {a11['status']} | {a11['closed_loop_decision_steps']} | {a11['executed_pushes']} | "
            f"{a11['boxes_on_goals_final']} | {_fmt(a11['final_slot_schedule_cost'])} | "
            f"{a11['final_dependency_count']} | NA | NA | NA |"
        ),
        (
            f"| A12 | {a12['status']} | {a12['closed_loop_decision_steps']} | {a12['executed_pushes']} | "
            f"{a12['boxes_on_goals_final']} | {_fmt(a12['final_slot_schedule_cost'])} | "
            f"{a12['final_dependency_count']} | {a12['selected_stalled_count']} | "
            f"{a12['selected_reverse_recent_edge_count']} | {a12['selected_repeated_signature_count']} |"
        ),
        "",
        "## A12 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | stall score | stalled | reverse | repeated signature |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["a12_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | NA | NA | NA |"
            )
            continue
        stall = row.get("chosen_schedule_stall", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{_fmt(row.get('chosen_schedule_stall_score'))} | "
            f"{stall.get('stalled', False)} | {stall.get('reverse_recent_edge', False)} | "
            f"{stall.get('repeated_schedule_signature', False)} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A12 检验的是：在 A11 已有调度图的情况下，单纯加入 dependency-stall loop guard "
                "是否足以打破第二关的二箱循环。若仍未解出，应继续转向 release macro 或 TVA 前缀校准。"
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
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument("--schedule-lookahead-slots", type=int, default=2)
    parser.add_argument("--recent-edge-window", type=int, default=12)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a12_stall_guard_l002_s200.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a12_stall_guard_l002_s200.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a11_model = build_model(level, config)
    a12_model = build_model(level, config)
    a11 = _A11.run_a11_smoke(
        a11_model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=31199,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
        schedule_lookahead_slots=int(args.schedule_lookahead_slots),
    )
    a12 = run_a12_smoke(
        a12_model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=32309,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
        schedule_lookahead_slots=int(args.schedule_lookahead_slots),
        recent_edge_window=int(args.recent_edge_window),
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a12_stall_guard_v1",
            "failure_log": _rel(failure_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a11_planner": RoomSchedulePlanGFTSDMCTSPlanner.provider_id,
            "a12_planner": RoomScheduleStallGuardGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a11_schedule_plan_vs_a12_dependency_stall_guard_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "corridor_macro_max_extra_pushes": int(args.corridor_macro_max_extra_pushes),
            "corridor_macro_min_total_pushes": int(args.corridor_macro_min_total_pushes),
            "schedule_lookahead_slots": int(args.schedule_lookahead_slots),
            "recent_edge_window": int(args.recent_edge_window),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a12_model.room_static_summary(),
        "initial_schedule_plan": a12_model.room_schedule_plan_summary(
            a12_model.initial_state,
            a12_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a11_closed_loop_smoke": a11,
        "a12_closed_loop_smoke": a12,
        "comparison": compare_a11_a12(a11, a12),
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
