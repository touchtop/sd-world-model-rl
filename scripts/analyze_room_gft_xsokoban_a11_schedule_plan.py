#!/usr/bin/env python3
"""A11 schedule-plan SD diagnostics for XSokoban first levels.

A11 keeps the A10 access-plan mechanism and adds an explicit multi-box release
dependency graph.  No teacher actions or reference traces are used online.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A10_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a10_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a10_access_plan.py",
)
if _A10_SPEC is None or _A10_SPEC.loader is None:
    raise RuntimeError("Unable to load A10 diagnostic helpers.")
_A10 = importlib.util.module_from_spec(_A10_SPEC)
_A10_SPEC.loader.exec_module(_A10)

from src.sokoban import (  # noqa: E402
    RoomAccessPlanGFTSDMCTSPlanner,
    RoomCommitmentState,
    RoomSchedulePlanGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomAccessPlanGFTSDModel,
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


def build_model(level, config: dict[str, Any]) -> SokobanRoomSchedulePlanGFTSDModel:
    return SokobanRoomSchedulePlanGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def build_a11_planner(
    model: SokobanRoomSchedulePlanGFTSDModel,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    boundary_distance: int,
    previous_physical_action: SokobanPushAction | None = None,
    previous_moved_source=None,
    previous_moved_target=None,
    recent_commitment_keys: set | None = None,
    corridor_macro_max_extra_pushes: int = 8,
    corridor_macro_min_total_pushes: int = 2,
    schedule_lookahead_slots: int = 3,
) -> RoomSchedulePlanGFTSDMCTSPlanner:
    return RoomSchedulePlanGFTSDMCTSPlanner(
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
    )


def summarise_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [row for row in trace if "physical_action" in row]
    macro_rows = [row for row in decision_rows if row.get("corridor_macro", {}).get("triggered")]
    macro_push_counts = [
        int(row.get("corridor_macro", {}).get("macro_push_count", 1))
        for row in macro_rows
    ]
    scores = [float(row.get("chosen_schedule_plan_score", 0.0)) for row in decision_rows]
    schedule_totals = [
        row["schedule_plan_after"]["totals"]
        for row in decision_rows
        if "schedule_plan_after" in row
    ]
    return {
        "closed_loop_decision_steps": int(len(decision_rows)),
        "executed_pushes": int(sum(int(row.get("executed_push_count", 1)) for row in decision_rows)),
        "macro_trigger_steps": int(len(macro_rows)),
        "macro_total_extra_pushes": int(sum(max(0, count - 1) for count in macro_push_counts)),
        "macro_push_count_median": None if not macro_push_counts else float(median(macro_push_counts)),
        "macro_push_count_max": None if not macro_push_counts else int(max(macro_push_counts)),
        "schedule_plan_score_median": None if not scores else float(median(scores)),
        "schedule_plan_score_min": None if not scores else float(min(scores)),
        "schedule_plan_score_max": None if not scores else float(max(scores)),
        "max_dependency_count": 0 if not schedule_totals else max(int(row["dependency_count"]) for row in schedule_totals),
        "max_release_action_count": 0 if not schedule_totals else max(int(row["release_action_count"]) for row in schedule_totals),
    }


def run_a11_smoke(
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
    executed_push_budget = int(max_steps)
    executed_pushes = 0

    for decision_step in range(max_steps):
        if executed_pushes >= executed_push_budget:
            status = "max_steps"
            break
        if model.is_solved(state):
            status = "solved"
            break
        planner = build_a11_planner(
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
                    "access_plan": model.room_access_plan_summary(
                        state,
                        commitment_state,
                        lookahead_slots=schedule_lookahead_slots,
                    ),
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
            source, target = _A10._A9._A8._A6._A5.moved_cells(state, action)
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
                    "schedule_barrier_after": float(
                        schedule_after_step["totals"]["schedule_barrier_cost"]
                    ),
                    "dependency_count_after": int(schedule_after_step["totals"]["dependency_count"]),
                    "release_action_count_after": int(schedule_after_step["totals"]["release_action_count"]),
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
        access_after = model.room_access_plan_summary(
            state,
            commitment_state,
            lookahead_slots=schedule_lookahead_slots,
        )
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
                "chosen_access_plan": chosen_child.get("room_access_plan", {}),
                "chosen_access_plan_score": float(chosen_child.get("room_access_plan_score", 0.0)),
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
                "access_plan_after": access_after,
                "schedule_plan_after": schedule_after,
                "room_summary_after": model.room_state_summary(state),
                "first_moved_source": None if first_source is None else list(first_source),
                "first_moved_target": None if first_target is None else list(first_target),
            }
        )
        if status in {"solved", "deadlocked"}:
            break

    if model.is_solved(state):
        status = "solved"
    final_access = model.room_access_plan_summary(
        state,
        commitment_state,
        lookahead_slots=schedule_lookahead_slots,
    )
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
            "final_access_barrier_cost": float(final_access["totals"]["access_barrier_cost"]),
            "final_future_access_barrier_cost": float(
                final_access["totals"]["access_future_barrier_cost"]
            ),
            "final_schedule_barrier_cost": float(final_schedule["totals"]["schedule_barrier_cost"]),
            "final_dependency_count": int(final_schedule["totals"]["dependency_count"]),
            "final_future_dependency_count": int(final_schedule["totals"]["future_dependency_count"]),
            "final_blocker_box_count": int(final_schedule["totals"]["blocker_box_count"]),
            "final_critical_stand_box_count": int(final_schedule["totals"]["critical_stand_box_count"]),
            "final_release_action_count": int(final_schedule["totals"]["release_action_count"]),
            "final_destructive_action_count": int(final_schedule["totals"]["destructive_action_count"]),
        }
    )
    return {
        "summary": summary,
        "trace": trace,
        "final_room_summary": model.room_state_summary(state),
        "final_commitment_summary": model.room_commitment_summary(state, commitment_state),
        "final_slot_progress": final_slot,
        "final_access_plan": final_access,
        "final_schedule_plan": final_schedule,
        "final_tier_a_status": model.tier_a_status(state),
    }


def compare_a10_a11(a10: dict[str, Any], a11: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a10_actions = actions(a10)
    a11_actions = actions(a11)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a10_actions, a11_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a10_actions) != len(a11_actions):
        first_diff = min(len(a10_actions), len(a11_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a10_status": a10["summary"]["status"],
        "a11_status": a11["summary"]["status"],
        "a10_boxes_on_goals_final": a10["summary"]["boxes_on_goals_final"],
        "a11_boxes_on_goals_final": a11["summary"]["boxes_on_goals_final"],
        "a10_executed_pushes": a10["summary"]["executed_pushes"],
        "a11_executed_pushes": a11["summary"]["executed_pushes"],
        "a10_final_slot_schedule_cost": a10["summary"]["final_slot_schedule_cost"],
        "a11_final_slot_schedule_cost": a11["summary"]["final_slot_schedule_cost"],
        "a10_prefix_actions": a10_actions[:40],
        "a11_prefix_actions": a11_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a10 = payload["a10_closed_loop_smoke"]["summary"]
    a11 = payload["a11_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A11 schedule-plan 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A11 在 A10 access-plan 基础上加入 state-derived multi-box release dependency graph。"
            "该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定。"
        ),
        (
            f"A10 smoke 为 `{a10['status']}`，执行 `{a10['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a10['boxes_on_goals_final']}`；"
            f"A11 smoke 为 `{a11['status']}`，执行 `{a11['executed_pushes']}` 个 push、"
            f"`{a11['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a11['boxes_on_goals_final']}`。"
        ),
        (
            f"A10/A11 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A11 最终 slot cost 为 `{_fmt(a11['final_slot_schedule_cost'])}`，"
            f"schedule barrier 为 `{_fmt(a11['final_schedule_barrier_cost'])}`，"
            f"dependency count 为 `{a11['final_dependency_count']}`。"
        ),
        "",
        "## A10/A11 摘要",
        "",
        "| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | access barrier | schedule barrier | dependencies | release actions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A10 | {a10['status']} | {a10['closed_loop_decision_steps']} | {a10['executed_pushes']} | "
            f"{a10['boxes_on_goals_final']} | {_fmt(a10['final_slot_schedule_cost'])} | "
            f"{_fmt(a10['final_access_barrier_cost'])} | NA | NA | NA |"
        ),
        (
            f"| A11 | {a11['status']} | {a11['closed_loop_decision_steps']} | {a11['executed_pushes']} | "
            f"{a11['boxes_on_goals_final']} | {_fmt(a11['final_slot_schedule_cost'])} | "
            f"{_fmt(a11['final_access_barrier_cost'])} | {_fmt(a11['final_schedule_barrier_cost'])} | "
            f"{a11['final_dependency_count']} | {a11['final_release_action_count']} |"
        ),
        "",
        "## A11 轨迹前缀",
        "",
        "| step | action | pushes | macro | goals | slot cost | schedule barrier | dependencies | release actions | schedule score |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["a11_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | False | NA | NA | "
                f"{_fmt(schedule.get('schedule_barrier_cost'))} | "
                f"{_fmt(schedule.get('dependency_count'))} | "
                f"{_fmt(schedule.get('release_action_count'))} | NA |"
            )
            continue
        macro = row.get("corridor_macro", {})
        schedule = row["schedule_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {macro.get('triggered', False)} | "
            f"{row['boxes_on_goals_after']} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{_fmt(schedule['schedule_barrier_cost'])} | "
            f"{schedule['dependency_count']} | "
            f"{schedule['release_action_count']} | "
            f"{_fmt(row.get('chosen_schedule_plan_score'))} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A11 检验的是：将多箱释放依赖显式放入 SD 状态后，MCTS 是否能够消费这些调度信号。"
                "若仍未解出，则剩余缺口应继续定位在 release macro、跨房间调度或 TVA 前缀校准，"
                "而不是 teacher 在线动作锚定。"
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
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a11_schedule_plan_guarded_l002_s200.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a11_schedule_plan_guarded_l002_s200.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a10_model = SokobanRoomAccessPlanGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )
    a11_model = build_model(level, config)
    a10 = _A10.run_a10_smoke(
        a10_model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=30089,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
        access_lookahead_slots=2,
    )
    a11 = run_a11_smoke(
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
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a11_schedule_plan_v1",
            "failure_log": _rel(failure_path),
            "a10_model": SokobanRoomAccessPlanGFTSDModel.provider_id,
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a10_planner": RoomAccessPlanGFTSDMCTSPlanner.provider_id,
            "a11_planner": RoomSchedulePlanGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a10_access_plan_vs_a11_schedule_plan_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "corridor_macro_max_extra_pushes": int(args.corridor_macro_max_extra_pushes),
            "corridor_macro_min_total_pushes": int(args.corridor_macro_min_total_pushes),
            "a10_access_lookahead_slots": 2,
            "a11_schedule_lookahead_slots": int(args.schedule_lookahead_slots),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a11_model.room_static_summary(),
        "initial_access_plan": a11_model.room_access_plan_summary(
            a11_model.initial_state,
            a11_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "initial_schedule_plan": a11_model.room_schedule_plan_summary(
            a11_model.initial_state,
            a11_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a10_closed_loop_smoke": a10,
        "a11_closed_loop_smoke": a11,
        "comparison": compare_a10_a11(a10, a11),
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
