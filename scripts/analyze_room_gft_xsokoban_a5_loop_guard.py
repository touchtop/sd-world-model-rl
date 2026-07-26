#!/usr/bin/env python3
"""A5 loop-guard diagnostics for XSokoban first level.

The A5 planner inherits A4 slot-schedule progress and adds soft penalties for
committed-box distance regression, direct push reversal, and repeated
commitment states.  This script compares A4 and A5 on the same short smoke
scope; it is a mechanism diagnostic, not a solved-rate benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import (  # noqa: E402
    RoomCommitmentState,
    RoomLoopGuardGFTSDMCTSPlanner,
    RoomSlotProgressGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomGFTSDModel,
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


def option_label(option: dict[str, Any]) -> str:
    if option["kind"] == "slot":
        return f"slot{option['slot_id']}@{option['slot_cell']}"
    return str(option["kind"])


def semantic_label(row: dict[str, Any]) -> str:
    action = row["semantic_action"]["physical_action"]
    option = row["semantic_action"]["commitment_option"]
    return f"b{action['box_index']}:{action['direction_name']}|{option_label(option)}"


def build_model(level, config: dict[str, Any]) -> SokobanRoomGFTSDModel:
    return SokobanRoomGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def build_planner(
    planner_cls,
    model: SokobanRoomGFTSDModel,
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
):
    kwargs = {}
    if issubclass(planner_cls, RoomLoopGuardGFTSDMCTSPlanner):
        kwargs.update(
            {
                "previous_physical_action": previous_physical_action,
                "previous_moved_source": previous_moved_source,
                "previous_moved_target": previous_moved_target,
                "recent_commitment_keys": recent_commitment_keys,
            }
        )
    return planner_cls(
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
        **kwargs,
    )


def moved_cells(state, action: SokobanPushAction):
    source = state.boxes[action.box_index]
    target = add_cells(source, DIRECTIONS[action.direction])
    return source, target


def summarise_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_counts = [row.get("semantic_child_count", 0) for row in trace if "semantic_child_count" in row]
    loop_rows = [row for row in trace if row.get("chosen_loop", {}).get("score", 0.0) < 0.0]
    direct_reverse = [
        row for row in trace if row.get("chosen_loop", {}).get("direct_reverse_previous_push")
    ]
    distance_regress = [
        row for row in trace if row.get("chosen_loop", {}).get("distance_regress", 0.0) > 0.0
    ]
    repeated = [
        row for row in trace if row.get("chosen_loop", {}).get("repeated_commitment_state")
    ]
    return {
        "steps_executed": len([row for row in trace if "physical_action" in row]),
        "semantic_child_count_median": median(semantic_counts) if semantic_counts else None,
        "semantic_child_count_max": max(semantic_counts) if semantic_counts else None,
        "chosen_loop_penalty_steps": int(len(loop_rows)),
        "chosen_direct_reverse_steps": int(len(direct_reverse)),
        "chosen_distance_regress_steps": int(len(distance_regress)),
        "chosen_repeated_commitment_steps": int(len(repeated)),
        "first_chosen_loop_penalty_step": loop_rows[0]["step"] if loop_rows else None,
    }


def committed_distance_for_box(
    model: SokobanRoomGFTSDModel,
    commitment_state: RoomCommitmentState,
    box_cell,
) -> float | None:
    commitment = commitment_state.commitment_for_box(box_cell)
    if commitment is None:
        return None
    distance = model._slot_pull_distance(box_cell, commitment.slot_cell)
    if distance == float("inf"):
        return float("inf")
    return float(distance)


def transition_loop_summary(
    model: SokobanRoomGFTSDModel,
    commitment_state: RoomCommitmentState,
    child_commitment_state: RoomCommitmentState,
    source,
    target,
    previous_source,
    previous_target,
    recent_commitment_keys: set,
    child_state,
) -> dict[str, Any]:
    before_distance = committed_distance_for_box(model, commitment_state, source)
    after_distance = committed_distance_for_box(model, child_commitment_state, target)
    distance_regress = 0.0
    if before_distance is not None and after_distance is not None:
        if before_distance != float("inf") and after_distance != float("inf"):
            distance_regress = max(0.0, after_distance - before_distance)
        elif before_distance != float("inf") and after_distance == float("inf"):
            distance_regress = 1.0
    direct_reverse = bool(
        previous_source is not None
        and previous_target is not None
        and source == previous_target
        and target == previous_source
    )
    repeated = model.room_commitment_key(child_state, child_commitment_state) in recent_commitment_keys
    score = -(
        8.0 * float(distance_regress)
        + 6.0 * float(direct_reverse)
        + 5.0 * float(repeated)
    )
    return {
        "moved_source": list(source),
        "moved_target": list(target),
        "before_committed_distance": before_distance,
        "after_committed_distance": after_distance,
        "distance_regress": float(distance_regress),
        "direct_reverse_previous_push": bool(direct_reverse),
        "repeated_commitment_state": bool(repeated),
        "recent_commitment_key_count": int(len(recent_commitment_keys)),
        "score": float(score),
    }


def run_smoke(
    planner_cls,
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    budget: int,
    horizon: int,
    max_steps: int,
    boundary_distance: int,
    seed_offset: int,
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

    for step in range(max_steps):
        if model.is_solved(state):
            status = "solved"
            break
        planner = build_planner(
            planner_cls,
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
        )
        plan = planner.plan(state, commitment_state)
        physical = plan["physical_action"]
        if physical is None:
            status = "no_action"
            trace.append(
                {
                    "step": step,
                    "status": "no_action",
                    "root_visits": int(plan["root_visits"]),
                    "root_value": float(plan["root_value"]),
                }
            )
            break
        source, target = moved_cells(state, physical)
        next_state = model.step_grid_flow(state, physical)
        before_commitment_state = commitment_state
        commitment_state = plan["next_commitment_state"]
        ex_post_loop = transition_loop_summary(
            model,
            before_commitment_state,
            commitment_state,
            source,
            target,
            previous_source,
            previous_target,
            recent_commitment_keys,
            next_state,
        )
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
        chosen_loop = chosen_child.get("room_loop") or ex_post_loop
        trace.append(
            {
                "step": step,
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
                "chosen_loop_penalty_score": float(chosen_loop.get("score", 0.0)),
                "boxes_on_goals_after": int(model.boxes_on_goals(next_state)),
                "commitment_after": commitment_state.to_dict(),
                "slot_progress_after": model.room_slot_progress_summary(next_state, commitment_state),
                "room_summary_after": model.room_state_summary(next_state),
            }
        )
        state = next_state
        visited_keys.add(model.canonical_key(state))
        recent_commitment_keys.add(model.room_commitment_key(state, commitment_state))
        previous_action = physical
        previous_source = source
        previous_target = target
        if model.is_deadlocked(state) and not model.is_solved(state):
            status = "deadlocked"
            break
    if model.is_solved(state):
        status = "solved"
    summary = summarise_trace(trace)
    summary.update(
        {
            "status": status,
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "final_commitment_count": len(commitment_state.commitments),
            "final_slot_schedule_cost": float(
                model.room_slot_progress_summary(state, commitment_state)["slot_schedule_cost"]
            ),
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


def compare_smokes(a4: dict[str, Any], a5: dict[str, Any]) -> dict[str, Any]:
    a4_actions = [
        f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
        for row in a4["trace"]
        if "physical_action" in row
    ]
    a5_actions = [
        f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
        for row in a5["trace"]
        if "physical_action" in row
    ]
    first_diff = None
    for idx, (left, right) in enumerate(zip(a4_actions, a5_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a4_actions) != len(a5_actions):
        first_diff = min(len(a4_actions), len(a5_actions))
    return {
        "first_action_difference_step": first_diff,
        "a4_prefix_actions": a4_actions[:40],
        "a5_prefix_actions": a5_actions[:40],
        "a4_final_slot_schedule_cost": a4["summary"]["final_slot_schedule_cost"],
        "a5_final_slot_schedule_cost": a5["summary"]["final_slot_schedule_cost"],
        "a4_boxes_on_goals_final": a4["summary"]["boxes_on_goals_final"],
        "a5_boxes_on_goals_final": a5["summary"]["boxes_on_goals_final"],
        "a4_chosen_direct_reverse_steps": a4["summary"].get("chosen_direct_reverse_steps", 0),
        "a5_chosen_direct_reverse_steps": a5["summary"].get("chosen_direct_reverse_steps", 0),
        "a4_chosen_distance_regress_steps": a4["summary"].get("chosen_distance_regress_steps", 0),
        "a5_chosen_distance_regress_steps": a5["summary"].get("chosen_distance_regress_steps", 0),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a4 = payload["a4_closed_loop_smoke"]["summary"]
    a5 = payload["a5_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    lines = [
        "# XSokoban first 第 1 关 Room-GFT A5 loop guard 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A5 在 A4 槽位顺序进度基础上加入 committed-box loop guard："
            "对承诺槽位距离回退、直接反向推动和重复承诺状态施加软惩罚。"
            "该 planner 为 `RoomLoopGuardGFTSDMCTSPlanner`，仍只属于 hard-instance 派生线。"
        ),
        (
            f"A4 smoke 状态为 `{a4['status']}`，执行 `{a4['steps_executed']}` 步后 "
            f"`boxes_on_goals={a4['boxes_on_goals_final']}`；"
            f"A5 smoke 状态为 `{a5['status']}`，执行 `{a5['steps_executed']}` 步后 "
            f"`boxes_on_goals={a5['boxes_on_goals_final']}`。"
            f"二者首个动作差异发生在 step `{_fmt(cmp['first_action_difference_step'])}`。"
        ),
        (
            f"A4 选择轨迹中 distance-regress 步数为 `{a4['chosen_distance_regress_steps']}`，"
            f"direct-reverse 步数为 `{a4['chosen_direct_reverse_steps']}`；"
            f"A5 对应为 `{a5['chosen_distance_regress_steps']}` 和 `{a5['chosen_direct_reverse_steps']}`。"
            "该结果用于检查 loop guard 是否实际进入动作选择，而非只改变未选 child 的 prior。"
        ),
        "",
        "## A4/A5 摘要",
        "",
        "| planner | status | steps | boxes_on_goals | final slot cost | loop penalty steps | distance regress | direct reverse | repeated commitment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A4 | {a4['status']} | {a4['steps_executed']} | {a4['boxes_on_goals_final']} | "
            f"{_fmt(a4['final_slot_schedule_cost'])} | {a4['chosen_loop_penalty_steps']} | "
            f"{a4['chosen_distance_regress_steps']} | {a4['chosen_direct_reverse_steps']} | "
            f"{a4['chosen_repeated_commitment_steps']} |"
        ),
        (
            f"| A5 | {a5['status']} | {a5['steps_executed']} | {a5['boxes_on_goals_final']} | "
            f"{_fmt(a5['final_slot_schedule_cost'])} | {a5['chosen_loop_penalty_steps']} | "
            f"{a5['chosen_distance_regress_steps']} | {a5['chosen_direct_reverse_steps']} | "
            f"{a5['chosen_repeated_commitment_steps']} |"
        ),
        "",
        "## A5 轨迹前缀",
        "",
        "| step | action | commitment | goals | slot cost | loop score | regress | reverse | repeated |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["a5_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            lines.append(f"| {row['step']} | no_action | NA | NA | NA | NA | NA | NA | NA |")
            continue
        option = row["semantic_action"]["commitment_option"]
        loop = row.get("chosen_loop", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{option_label(option)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{_fmt(row.get('chosen_loop_penalty_score', 0.0))} | "
            f"{_fmt(loop.get('distance_regress', 0.0))} | "
            f"{loop.get('direct_reverse_previous_push', False)} | "
            f"{loop.get('repeated_commitment_state', False)} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A5 的目标是处理 A4 smoke 暴露出的走廊内承诺箱子来回推动，而不是改变 room-slot 静态结构。"
                "若 A5 仍不能推进到 goal，则说明仅靠软 loop guard 不足，需要进一步引入 corridor contraction、"
                "forced-corridor macro 或参考解锚定。"
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
    parser.add_argument("--smoke-steps", type=int, default=40)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a5_loop_guard.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a5_loop_guard.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = build_model(level, config)
    a4 = run_smoke(
        RoomSlotProgressGFTSDMCTSPlanner,
        model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=25043,
    )
    a5 = run_smoke(
        RoomLoopGuardGFTSDMCTSPlanner,
        model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=26047,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a5_loop_guard_v1",
            "failure_log": _rel(failure_path),
            "base_model": "SokobanRoomGFTSDModel",
            "a4_planner": RoomSlotProgressGFTSDMCTSPlanner.provider_id,
            "a5_planner": RoomLoopGuardGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "trace_scope": "short_a4_vs_a5_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": model.room_static_summary(),
        "a4_closed_loop_smoke": a4,
        "a5_closed_loop_smoke": a5,
        "comparison": compare_smokes(a4, a5),
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
