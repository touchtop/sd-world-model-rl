#!/usr/bin/env python3
"""A9 release-plan SD diagnostics for XSokoban first level.

This diagnostic extends A8 with state-derived active-box release signals.  It
does not pass teacher actions to the planner; the teacher trace is used only as
the prior diagnostic source for deciding which missing SD variables to model.
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

_A8_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a8_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a8_fill_plan.py",
)
if _A8_SPEC is None or _A8_SPEC.loader is None:
    raise RuntimeError("Unable to load A8 diagnostic helpers.")
_A8 = importlib.util.module_from_spec(_A8_SPEC)
_A8_SPEC.loader.exec_module(_A8)

from src.sokoban import (  # noqa: E402
    RoomCommitmentState,
    RoomFillPlanGFTSDMCTSPlanner,
    RoomReleasePlanGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomFillPlanGFTSDModel,
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


def build_model(level, config: dict[str, Any]) -> SokobanRoomReleasePlanGFTSDModel:
    return SokobanRoomReleasePlanGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def build_a9_planner(
    model: SokobanRoomReleasePlanGFTSDModel,
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
) -> RoomReleasePlanGFTSDMCTSPlanner:
    return RoomReleasePlanGFTSDMCTSPlanner(
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
    )


def summarise_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [row for row in trace if "physical_action" in row]
    macro_rows = [row for row in decision_rows if row.get("corridor_macro", {}).get("triggered")]
    macro_push_counts = [
        int(row.get("corridor_macro", {}).get("macro_push_count", 1))
        for row in macro_rows
    ]
    release_scores = [float(row.get("chosen_release_plan_score", 0.0)) for row in decision_rows]
    return {
        "closed_loop_decision_steps": int(len(decision_rows)),
        "executed_pushes": int(sum(int(row.get("executed_push_count", 1)) for row in decision_rows)),
        "macro_trigger_steps": int(len(macro_rows)),
        "macro_total_extra_pushes": int(sum(max(0, count - 1) for count in macro_push_counts)),
        "macro_push_count_median": None if not macro_push_counts else float(median(macro_push_counts)),
        "macro_push_count_max": None if not macro_push_counts else int(max(macro_push_counts)),
        "release_plan_score_median": None if not release_scores else float(median(release_scores)),
        "release_plan_score_min": None if not release_scores else float(min(release_scores)),
        "release_plan_score_max": None if not release_scores else float(max(release_scores)),
    }


def run_a9_smoke(
    model: SokobanRoomReleasePlanGFTSDModel,
    config: dict[str, Any],
    *,
    budget: int,
    horizon: int,
    max_steps: int,
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
    executed_push_budget = int(max_steps)
    executed_pushes = 0

    for decision_step in range(max_steps):
        if executed_pushes >= executed_push_budget:
            status = "max_steps"
            break
        if model.is_solved(state):
            status = "solved"
            break
        planner = build_a9_planner(
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
                    "fill_plan": model.room_fill_plan_summary(state, commitment_state),
                    "release_plan": model.room_release_plan_summary(state, commitment_state),
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

        before_commitment_state = commitment_state
        first_source = None
        first_target = None
        executed_rows = []
        for macro_index, action in enumerate(macro_actions):
            source, target = _A8._A6._A5.moved_cells(state, action)
            if macro_index == 0:
                first_source, first_target = source, target
            next_state = model.step_grid_flow(state, action)
            if macro_index < len(macro_commitments):
                commitment_state = macro_commitments[macro_index]
            else:
                commitment_state = model.normalize_room_commitment_state(next_state, commitment_state)
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
                    "fill_plan_barrier_after": float(
                        model.room_fill_plan_summary(next_state, commitment_state)["totals"][
                            "plan_barrier_cost"
                        ]
                    ),
                    "release_barrier_after": float(
                        release_after_step["totals"]["release_barrier_cost"]
                    ),
                    "release_need_after": int(
                        release_after_step["totals"]["active_release_need_count"]
                    ),
                    "release_stand_blocker_after": int(
                        release_after_step["totals"]["active_release_stand_blocker_count"]
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
        ex_post_loop = (
            _A8._A6._A5.transition_loop_summary(
                model,
                before_commitment_state,
                commitment_state,
                first_source,
                first_target,
                None,
                None,
                recent_commitment_keys,
                state,
            )
            if first_source is not None and first_target is not None
            else {}
        )
        chosen_loop = chosen_child.get("room_loop") or ex_post_loop
        chosen_fill = chosen_child.get("room_fill_plan", {})
        chosen_release = chosen_child.get("room_release_plan", {})
        fill_after = model.room_fill_plan_summary(state, commitment_state)
        release_after = model.room_release_plan_summary(state, commitment_state)
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
                "chosen_loop": chosen_loop,
                "chosen_fill_plan": chosen_fill,
                "chosen_fill_plan_score": float(chosen_child.get("room_fill_plan_score", 0.0)),
                "chosen_release_plan": chosen_release,
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
                "fill_plan_after": fill_after,
                "release_plan_after": release_after,
                "room_summary_after": model.room_state_summary(state),
            }
        )
        if status in {"solved", "deadlocked"}:
            break

    if model.is_solved(state):
        status = "solved"
    final_release = model.room_release_plan_summary(state, commitment_state)
    summary = summarise_trace(trace)
    summary.update(
        {
            "status": status,
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "final_commitment_count": len(commitment_state.commitments),
            "final_slot_schedule_cost": float(
                model.room_slot_progress_summary(state, commitment_state)["slot_schedule_cost"]
            ),
            "final_fill_plan_barrier_cost": float(
                model.room_fill_plan_summary(state, commitment_state)["totals"]["plan_barrier_cost"]
            ),
            "final_release_barrier_cost": float(final_release["totals"]["release_barrier_cost"]),
            "final_release_need_count": int(final_release["totals"]["active_release_need_count"]),
            "final_release_stand_blocker_count": int(
                final_release["totals"]["active_release_stand_blocker_count"]
            ),
            "final_release_monotone_available_count": int(
                final_release["totals"]["active_release_monotone_available_count"]
            ),
        }
    )
    return {
        "summary": summary,
        "trace": trace,
        "final_room_summary": model.room_state_summary(state),
        "final_commitment_summary": model.room_commitment_summary(state, commitment_state),
        "final_slot_progress": model.room_slot_progress_summary(state, commitment_state),
        "final_fill_plan": model.room_fill_plan_summary(state, commitment_state),
        "final_release_plan": final_release,
        "final_tier_a_status": model.tier_a_status(state),
    }


def compare_a8_a9(a8: dict[str, Any], a9: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a8_actions = actions(a8)
    a9_actions = actions(a9)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a8_actions, a9_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a8_actions) != len(a9_actions):
        first_diff = min(len(a8_actions), len(a9_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a8_prefix_actions": a8_actions[:40],
        "a9_prefix_actions": a9_actions[:40],
        "a8_status": a8["summary"]["status"],
        "a9_status": a9["summary"]["status"],
        "a8_boxes_on_goals_final": a8["summary"]["boxes_on_goals_final"],
        "a9_boxes_on_goals_final": a9["summary"]["boxes_on_goals_final"],
        "a8_executed_pushes": a8["summary"]["executed_pushes"],
        "a9_executed_pushes": a9["summary"]["executed_pushes"],
        "a8_final_slot_schedule_cost": a8["summary"]["final_slot_schedule_cost"],
        "a9_final_slot_schedule_cost": a9["summary"]["final_slot_schedule_cost"],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a8 = payload["a8_closed_loop_smoke"]["summary"]
    a9 = payload["a9_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    lines = [
        "# XSokoban first 第 1 关 Room-GFT A9 release-plan 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A9 在 A8 fill-plan SD 基础上加入 active-box release-plan 状态。该版本不读取 "
            "teacher 动作表，也不继承 A7 的逐状态参考锚定；它只消费当前状态可计算的候选单调推进、"
            "所需站位格、站位格阻塞和玩家可达性信号。"
        ),
        (
            f"A8 smoke 为 `{a8['status']}`，执行 `{a8['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a8['boxes_on_goals_final']}`；"
            f"A9 smoke 为 `{a9['status']}`，执行 `{a9['executed_pushes']}` 个 push、"
            f"`{a9['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a9['boxes_on_goals_final']}`。"
        ),
        (
            f"A8/A9 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A9 最终 slot cost 为 `{_fmt(a9['final_slot_schedule_cost'])}`，"
            f"fill barrier 为 `{_fmt(a9['final_fill_plan_barrier_cost'])}`，"
            f"release barrier 为 `{_fmt(a9['final_release_barrier_cost'])}`。"
        ),
        "",
        "## A8/A9 摘要",
        "",
        "| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | fill barrier | release barrier | release need | stand blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A8 | {a8['status']} | {a8['closed_loop_decision_steps']} | {a8['executed_pushes']} | "
            f"{a8['boxes_on_goals_final']} | {_fmt(a8['final_slot_schedule_cost'])} | "
            f"{_fmt(a8['final_fill_plan_barrier_cost'])} | NA | NA | NA |"
        ),
        (
            f"| A9 | {a9['status']} | {a9['closed_loop_decision_steps']} | {a9['executed_pushes']} | "
            f"{a9['boxes_on_goals_final']} | {_fmt(a9['final_slot_schedule_cost'])} | "
            f"{_fmt(a9['final_fill_plan_barrier_cost'])} | {_fmt(a9['final_release_barrier_cost'])} | "
            f"{a9['final_release_need_count']} | {a9['final_release_stand_blocker_count']} |"
        ),
        "",
        "## A9 轨迹前缀",
        "",
        "| step | action | executed pushes | macro | goals | slot cost | fill barrier | release barrier | release need | stand blockers | release score |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["a9_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            release = row.get("release_plan", {}).get("totals", {})
            fill = row.get("fill_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | False | NA | NA | "
                f"{_fmt(fill.get('plan_barrier_cost'))} | {_fmt(release.get('release_barrier_cost'))} | "
                f"{_fmt(release.get('active_release_need_count'))} | "
                f"{_fmt(release.get('active_release_stand_blocker_count'))} | NA |"
            )
            continue
        macro = row.get("corridor_macro", {})
        release = row["release_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {macro.get('triggered', False)} | "
            f"{row['boxes_on_goals_after']} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{_fmt(row['fill_plan_after']['totals']['plan_barrier_cost'])} | "
            f"{_fmt(release['release_barrier_cost'])} | "
            f"{release['active_release_need_count']} | "
            f"{release['active_release_stand_blocker_count']} | "
            f"{_fmt(row.get('chosen_release_plan_score'))} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A9 检验的是：A8 末端暴露的 active box blocked 状态，是否可以通过显式建模站位格释放"
                "来转化为无 teacher 在线动作锚定的改进。若 A9 仍未解出，则应记录其对 release barrier "
                "和 stand blocker 的实际影响，并据此判断后续缺口是否已经转向更长程的多箱调度或 TVA "
                "价值校准。"
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
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=120)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a9_release_plan.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a9_release_plan.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    a8_model: SokobanRoomFillPlanGFTSDModel = _A8.build_model(level, config)
    a9_model = build_model(level, config)
    a8 = _A8.run_a8_smoke(
        a8_model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=28069,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
    )
    a9 = run_a9_smoke(
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
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a9_release_plan_v1",
            "failure_log": _rel(failure_path),
            "base_model": SokobanRoomReleasePlanGFTSDModel.provider_id,
            "a8_model": SokobanRoomFillPlanGFTSDModel.provider_id,
            "a8_planner": RoomFillPlanGFTSDMCTSPlanner.provider_id,
            "a9_planner": RoomReleasePlanGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a8_baseline_vs_a9_state_derived_release_plan_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "corridor_macro_max_extra_pushes": int(args.corridor_macro_max_extra_pushes),
            "corridor_macro_min_total_pushes": int(args.corridor_macro_min_total_pushes),
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a9_model.room_static_summary(),
        "initial_fill_plan": a9_model.room_fill_plan_summary(
            a9_model.initial_state,
            a9_model.empty_room_commitment_state(),
        ),
        "initial_release_plan": a9_model.room_release_plan_summary(
            a9_model.initial_state,
            a9_model.empty_room_commitment_state(),
        ),
        "a8_closed_loop_smoke": a8,
        "a9_closed_loop_smoke": a9,
        "comparison": compare_a8_a9(a8, a9),
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
