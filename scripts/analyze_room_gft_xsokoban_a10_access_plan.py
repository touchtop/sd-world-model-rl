#!/usr/bin/env python3
"""A10 access-plan SD diagnostics for XSokoban first level.

This diagnostic extends A9 by protecting stand cells required by upcoming
room slots, not only the current active slot.  It does not pass teacher actions
to the planner.
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

_A9_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a9_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a9_release_plan.py",
)
if _A9_SPEC is None or _A9_SPEC.loader is None:
    raise RuntimeError("Unable to load A9 diagnostic helpers.")
_A9 = importlib.util.module_from_spec(_A9_SPEC)
_A9_SPEC.loader.exec_module(_A9)

from src.sokoban import (  # noqa: E402
    RoomAccessPlanGFTSDMCTSPlanner,
    RoomCommitmentState,
    RoomReleasePlanGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomAccessPlanGFTSDModel,
    SokobanRoomReleasePlanGFTSDModel,
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


def build_model(level, config: dict[str, Any]) -> SokobanRoomAccessPlanGFTSDModel:
    return SokobanRoomAccessPlanGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def build_a10_planner(
    model: SokobanRoomAccessPlanGFTSDModel,
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
    access_lookahead_slots: int = 3,
) -> RoomAccessPlanGFTSDMCTSPlanner:
    return RoomAccessPlanGFTSDMCTSPlanner(
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
        room_access_lookahead_slots=int(access_lookahead_slots),
    )


def summarise_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [row for row in trace if "physical_action" in row]
    macro_rows = [row for row in decision_rows if row.get("corridor_macro", {}).get("triggered")]
    macro_push_counts = [
        int(row.get("corridor_macro", {}).get("macro_push_count", 1))
        for row in macro_rows
    ]
    access_scores = [float(row.get("chosen_access_plan_score", 0.0)) for row in decision_rows]
    access_totals = [
        row["access_plan_after"]["totals"]
        for row in decision_rows
        if "access_plan_after" in row
    ]
    future_blocker_steps = [
        int(row["step"])
        for row in decision_rows
        if int(row.get("access_plan_after", {}).get("totals", {}).get("access_future_stand_blocker_count", 0)) > 0
    ]
    return {
        "closed_loop_decision_steps": int(len(decision_rows)),
        "executed_pushes": int(sum(int(row.get("executed_push_count", 1)) for row in decision_rows)),
        "macro_trigger_steps": int(len(macro_rows)),
        "macro_total_extra_pushes": int(sum(max(0, count - 1) for count in macro_push_counts)),
        "macro_push_count_median": None if not macro_push_counts else float(median(macro_push_counts)),
        "macro_push_count_max": None if not macro_push_counts else int(max(macro_push_counts)),
        "access_plan_score_median": None if not access_scores else float(median(access_scores)),
        "access_plan_score_min": None if not access_scores else float(min(access_scores)),
        "access_plan_score_max": None if not access_scores else float(max(access_scores)),
        "first_future_stand_blocker_step": None if not future_blocker_steps else min(future_blocker_steps),
        "max_future_stand_blocker_count": 0 if not access_totals else max(
            int(total["access_future_stand_blocker_count"]) for total in access_totals
        ),
    }


def run_a10_smoke(
    model: SokobanRoomAccessPlanGFTSDModel,
    config: dict[str, Any],
    *,
    budget: int,
    horizon: int,
    max_steps: int,
    boundary_distance: int,
    seed_offset: int,
    corridor_macro_max_extra_pushes: int,
    corridor_macro_min_total_pushes: int,
    access_lookahead_slots: int,
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
        planner = build_a10_planner(
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
            access_lookahead_slots=access_lookahead_slots,
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
                    "release_plan": model.room_release_plan_summary(state, commitment_state),
                    "access_plan": model.room_access_plan_summary(
                        state,
                        commitment_state,
                        lookahead_slots=access_lookahead_slots,
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
            source, target = _A9._A8._A6._A5.moved_cells(state, action)
            if macro_index == 0:
                first_source, first_target = source, target
            next_state = model.step_grid_flow(state, action)
            if macro_index < len(macro_commitments):
                commitment_state = macro_commitments[macro_index]
            else:
                commitment_state = model.normalize_room_commitment_state(next_state, commitment_state)
            access_after_step = model.room_access_plan_summary(
                next_state,
                commitment_state,
                lookahead_slots=access_lookahead_slots,
            )
            release_after_step = model.room_release_plan_summary(next_state, commitment_state)
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
                    "release_barrier_after": float(release_after_step["totals"]["release_barrier_cost"]),
                    "access_barrier_after": float(access_after_step["totals"]["access_barrier_cost"]),
                    "future_access_barrier_after": float(
                        access_after_step["totals"]["access_future_barrier_cost"]
                    ),
                    "future_access_stand_blocker_after": int(
                        access_after_step["totals"]["access_future_stand_blocker_count"]
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
        release_after = model.room_release_plan_summary(state, commitment_state)
        access_after = model.room_access_plan_summary(
            state,
            commitment_state,
            lookahead_slots=access_lookahead_slots,
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
                "chosen_access_plan": chosen_child.get("room_access_plan", {}),
                "chosen_access_plan_score": float(chosen_child.get("room_access_plan_score", 0.0)),
                "chosen_release_plan": chosen_child.get("room_release_plan", {}),
                "chosen_release_plan_score": float(chosen_child.get("room_release_plan_score", 0.0)),
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
                "release_plan_after": release_after,
                "access_plan_after": access_after,
                "room_summary_after": model.room_state_summary(state),
                "first_moved_source": None if first_source is None else list(first_source),
                "first_moved_target": None if first_target is None else list(first_target),
            }
        )
        if status in {"solved", "deadlocked"}:
            break

    if model.is_solved(state):
        status = "solved"
    final_release = model.room_release_plan_summary(state, commitment_state)
    final_access = model.room_access_plan_summary(
        state,
        commitment_state,
        lookahead_slots=access_lookahead_slots,
    )
    summary = summarise_trace(trace)
    summary.update(
        {
            "status": status,
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "final_commitment_count": len(commitment_state.commitments),
            "final_slot_schedule_cost": float(
                model.room_slot_progress_summary(state, commitment_state)["slot_schedule_cost"]
            ),
            "final_release_barrier_cost": float(final_release["totals"]["release_barrier_cost"]),
            "final_release_need_count": int(final_release["totals"]["active_release_need_count"]),
            "final_release_stand_blocker_count": int(
                final_release["totals"]["active_release_stand_blocker_count"]
            ),
            "final_access_barrier_cost": float(final_access["totals"]["access_barrier_cost"]),
            "final_future_access_barrier_cost": float(
                final_access["totals"]["access_future_barrier_cost"]
            ),
            "final_access_need_count": int(final_access["totals"]["access_need_count"]),
            "final_future_access_need_count": int(final_access["totals"]["access_future_need_count"]),
            "final_access_stand_blocker_count": int(final_access["totals"]["access_stand_blocker_count"]),
            "final_future_access_stand_blocker_count": int(
                final_access["totals"]["access_future_stand_blocker_count"]
            ),
        }
    )
    return {
        "summary": summary,
        "trace": trace,
        "final_room_summary": model.room_state_summary(state),
        "final_commitment_summary": model.room_commitment_summary(state, commitment_state),
        "final_slot_progress": model.room_slot_progress_summary(state, commitment_state),
        "final_release_plan": final_release,
        "final_access_plan": final_access,
        "final_tier_a_status": model.tier_a_status(state),
    }


def compare_a9_a10(a9: dict[str, Any], a10: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a9_actions = actions(a9)
    a10_actions = actions(a10)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a9_actions, a10_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a9_actions) != len(a10_actions):
        first_diff = min(len(a9_actions), len(a10_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a9_prefix_actions": a9_actions[:40],
        "a10_prefix_actions": a10_actions[:40],
        "a9_status": a9["summary"]["status"],
        "a10_status": a10["summary"]["status"],
        "a9_boxes_on_goals_final": a9["summary"]["boxes_on_goals_final"],
        "a10_boxes_on_goals_final": a10["summary"]["boxes_on_goals_final"],
        "a9_executed_pushes": a9["summary"]["executed_pushes"],
        "a10_executed_pushes": a10["summary"]["executed_pushes"],
        "a9_final_slot_schedule_cost": a9["summary"]["final_slot_schedule_cost"],
        "a10_final_slot_schedule_cost": a10["summary"]["final_slot_schedule_cost"],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a9 = payload["a9_closed_loop_smoke"]["summary"]
    a10 = payload["a10_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_offset"]) + 1
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A10 access-plan 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A10 在 A9 release-plan SD 基础上加入 lookahead access-plan 状态。该版本不读取 "
            "teacher 动作表，也不继承 A7 的逐状态参考锚定；它只消费当前状态可计算的后续 deep slot "
            "站位格保护信号。"
        ),
        (
            f"A9 smoke 为 `{a9['status']}`，执行 `{a9['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a9['boxes_on_goals_final']}`；"
            f"A10 smoke 为 `{a10['status']}`，执行 `{a10['executed_pushes']}` 个 push、"
            f"`{a10['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a10['boxes_on_goals_final']}`。"
        ),
        (
            f"A9/A10 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A10 最终 slot cost 为 `{_fmt(a10['final_slot_schedule_cost'])}`，"
            f"access barrier 为 `{_fmt(a10['final_access_barrier_cost'])}`，"
            f"future access barrier 为 `{_fmt(a10['final_future_access_barrier_cost'])}`。"
        ),
        "",
        "## A9/A10 摘要",
        "",
        "| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | release barrier | access barrier | first future blocker step | max future blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A9 | {a9['status']} | {a9['closed_loop_decision_steps']} | {a9['executed_pushes']} | "
            f"{a9['boxes_on_goals_final']} | {_fmt(a9['final_slot_schedule_cost'])} | "
            f"{_fmt(a9['final_release_barrier_cost'])} | NA | NA | NA |"
        ),
        (
            f"| A10 | {a10['status']} | {a10['closed_loop_decision_steps']} | {a10['executed_pushes']} | "
            f"{a10['boxes_on_goals_final']} | {_fmt(a10['final_slot_schedule_cost'])} | "
            f"{_fmt(a10['final_release_barrier_cost'])} | {_fmt(a10['final_access_barrier_cost'])} | "
            f"{_fmt(a10['first_future_stand_blocker_step'])} | {a10['max_future_stand_blocker_count']} |"
        ),
        "",
        "## A10 轨迹前缀",
        "",
        "| step | action | executed pushes | macro | goals | slot cost | access barrier | future access barrier | future blockers | access score |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["a10_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            access = row.get("access_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | False | NA | NA | "
                f"{_fmt(access.get('access_barrier_cost'))} | "
                f"{_fmt(access.get('access_future_barrier_cost'))} | "
                f"{_fmt(access.get('access_future_stand_blocker_count'))} | NA |"
            )
            continue
        macro = row.get("corridor_macro", {})
        access = row["access_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {macro.get('triggered', False)} | "
            f"{row['boxes_on_goals_after']} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{_fmt(access['access_barrier_cost'])} | "
            f"{_fmt(access['access_future_barrier_cost'])} | "
            f"{access['access_future_stand_blocker_count']} | "
            f"{_fmt(row.get('chosen_access_plan_score'))} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A10 检验的是：把未来 deep slot 的站位格提前纳入 SD 状态后，是否足以改变 A9 "
                "在第三个 deep slot 前的失败。若仍未解出，则说明当前缺口不只是“看见 future blocker”，"
                "还包括释放动作是否仍可达、是否需要更长程宏动作或 TVA 对前缀状态进行校准。"
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
    parser.add_argument("--level-offset", type=int, default=None)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=120)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument("--access-lookahead-slots", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a10_access_plan.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(config["level_offset"]) if args.level_offset is None else int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a9_model: SokobanRoomReleasePlanGFTSDModel = _A9.build_model(level, config)
    a10_model = build_model(level, config)
    a9 = _A9.run_a9_smoke(
        a9_model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=29079,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
    )
    a10 = run_a10_smoke(
        a10_model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=30089,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
        access_lookahead_slots=int(args.access_lookahead_slots),
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a10_access_plan_v1",
            "failure_log": _rel(failure_path),
            "base_model": SokobanRoomAccessPlanGFTSDModel.provider_id,
            "a9_model": SokobanRoomReleasePlanGFTSDModel.provider_id,
            "a9_planner": RoomReleasePlanGFTSDMCTSPlanner.provider_id,
            "a10_planner": RoomAccessPlanGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a9_baseline_vs_a10_lookahead_access_plan_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "corridor_macro_max_extra_pushes": int(args.corridor_macro_max_extra_pushes),
            "corridor_macro_min_total_pushes": int(args.corridor_macro_min_total_pushes),
            "access_lookahead_slots": int(args.access_lookahead_slots),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a10_model.room_static_summary(),
        "initial_release_plan": a10_model.room_release_plan_summary(
            a10_model.initial_state,
            a10_model.empty_room_commitment_state(),
        ),
        "initial_access_plan": a10_model.room_access_plan_summary(
            a10_model.initial_state,
            a10_model.empty_room_commitment_state(),
            lookahead_slots=int(args.access_lookahead_slots),
        ),
        "a9_closed_loop_smoke": a9,
        "a10_closed_loop_smoke": a10,
        "comparison": compare_a9_a10(a9, a10),
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
