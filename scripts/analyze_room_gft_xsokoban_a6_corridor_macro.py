#!/usr/bin/env python3
"""A6 forced-corridor macro diagnostics for XSokoban first level.

The A6 planner inherits the A5 loop guard and adds a conservative root-level
macro: when the selected committed box has exactly one legal monotone push
toward its committed slot, repeated pushes are contracted into one closed-loop
macro execution.  This is a hard-instance diagnostic, not a solved-rate
benchmark.
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

_A5_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a5_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a5_loop_guard.py",
)
if _A5_SPEC is None or _A5_SPEC.loader is None:
    raise RuntimeError("Unable to load A5 diagnostic helpers.")
_A5 = importlib.util.module_from_spec(_A5_SPEC)
_A5_SPEC.loader.exec_module(_A5)

from src.sokoban import (  # noqa: E402
    RoomCommitmentState,
    RoomCorridorMacroGFTSDMCTSPlanner,
    RoomLoopGuardGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomGFTSDModel,
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


def action_label(action: SokobanPushAction) -> str:
    return f"b{action.box_index}:{ACTION_NAMES[action.direction]}"


def build_a6_planner(
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
    corridor_macro_max_extra_pushes: int = 8,
    corridor_macro_min_total_pushes: int = 2,
):
    return RoomCorridorMacroGFTSDMCTSPlanner(
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


def summarise_macro_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    base = _A5.summarise_trace(trace)
    macro_rows = [row for row in trace if row.get("corridor_macro", {}).get("triggered")]
    macro_push_counts = [
        int(row.get("corridor_macro", {}).get("macro_push_count", 1))
        for row in macro_rows
    ]
    executed_pushes = sum(int(row.get("executed_push_count", 1)) for row in trace if "physical_action" in row)
    base.update(
        {
            "closed_loop_decision_steps": len([row for row in trace if "physical_action" in row]),
            "executed_pushes": int(executed_pushes),
            "macro_trigger_steps": int(len(macro_rows)),
            "macro_total_extra_pushes": int(sum(max(0, count - 1) for count in macro_push_counts)),
            "macro_push_count_median": median(macro_push_counts) if macro_push_counts else None,
            "macro_push_count_max": max(macro_push_counts) if macro_push_counts else None,
        }
    )
    return base


def run_a6_smoke(
    model: SokobanRoomGFTSDModel,
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
        planner = build_a6_planner(
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
            source, target = _A5.moved_cells(state, action)
            if macro_index == 0:
                first_source, first_target = source, target
            next_state = model.step_grid_flow(state, action)
            if macro_index < len(macro_commitments):
                commitment_state = macro_commitments[macro_index]
            else:
                commitment_state = model.normalize_room_commitment_state(next_state, commitment_state)
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

        ex_post_loop = _A5.transition_loop_summary(
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
                "chosen_loop_penalty_score": float(chosen_loop.get("score", 0.0)) if chosen_loop else 0.0,
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
                "room_summary_after": model.room_state_summary(state),
            }
        )
        if status in {"solved", "deadlocked"}:
            break
    if model.is_solved(state):
        status = "solved"
    summary = summarise_macro_trace(trace)
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


def compare_a5_a6(a5: dict[str, Any], a6: dict[str, Any]) -> dict[str, Any]:
    a5_actions = [
        f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
        for row in a5["trace"]
        if "physical_action" in row
    ]
    a6_actions = [
        f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
        for row in a6["trace"]
        if "physical_action" in row
    ]
    first_diff = None
    for idx, (left, right) in enumerate(zip(a5_actions, a6_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a5_actions) != len(a6_actions):
        first_diff = min(len(a5_actions), len(a6_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a5_prefix_actions": a5_actions[:40],
        "a6_prefix_actions": a6_actions[:40],
        "a5_status": a5["summary"]["status"],
        "a6_status": a6["summary"]["status"],
        "a5_boxes_on_goals_final": a5["summary"]["boxes_on_goals_final"],
        "a6_boxes_on_goals_final": a6["summary"]["boxes_on_goals_final"],
        "a5_decision_steps": a5["summary"]["steps_executed"],
        "a6_decision_steps": a6["summary"]["closed_loop_decision_steps"],
        "a6_executed_pushes": a6["summary"]["executed_pushes"],
        "a6_macro_trigger_steps": a6["summary"]["macro_trigger_steps"],
        "a6_macro_total_extra_pushes": a6["summary"]["macro_total_extra_pushes"],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a5 = payload["a5_closed_loop_smoke"]["summary"]
    a6 = payload["a6_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    lines = [
        "# XSokoban first 第 1 关 Room-GFT A6 forced-corridor macro 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A6 在 A5 loop guard 基础上加入 root-level forced-corridor macro。"
            "宏只在已承诺箱子存在唯一单调推进方向时触发，树内仍保持 single-push MCTS。"
        ),
        (
            f"A5 smoke 为 `{a5['status']}`，执行 `{a5['steps_executed']}` 个 push 后 "
            f"`boxes_on_goals={a5['boxes_on_goals_final']}`；"
            f"A6 smoke 为 `{a6['status']}`，执行 `{a6['executed_pushes']}` 个 push、"
            f"`{a6['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a6['boxes_on_goals_final']}`。"
        ),
        (
            f"A6 触发 corridor macro `{a6['macro_trigger_steps']}` 次，"
            f"额外压缩 push 数 `{a6['macro_total_extra_pushes']}`。"
            "该结果用于判断走廊压缩是否真正改变闭环执行，而不是只改变评分。"
        ),
        "",
        "## A5/A6 摘要",
        "",
        "| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | macro triggers | macro extra pushes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| A5 | {a5['status']} | {a5['steps_executed']} | {a5['steps_executed']} | "
            f"{a5['boxes_on_goals_final']} | {_fmt(a5['final_slot_schedule_cost'])} | 0 | 0 |"
        ),
        (
            f"| A6 | {a6['status']} | {a6['closed_loop_decision_steps']} | {a6['executed_pushes']} | "
            f"{a6['boxes_on_goals_final']} | {_fmt(a6['final_slot_schedule_cost'])} | "
            f"{a6['macro_trigger_steps']} | {a6['macro_total_extra_pushes']} |"
        ),
        "",
        "## A6 轨迹前缀",
        "",
        "| step | action | executed pushes | macro | goals | slot cost | reason |",
        "|---:|---|---:|---|---:|---:|---|",
    ]
    for row in payload["a6_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            lines.append(f"| {row['step']} | no_action | 0 | False | NA | NA | NA |")
            continue
        macro = row.get("corridor_macro", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {macro.get('triggered', False)} | "
            f"{row['boxes_on_goals_after']} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{macro.get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A6 检验的是 corridor contraction 是否能减少 A5 的逐步走廊推进开销。"
                "若仍进入 no_action 或不能增加目标完成数，则说明当前主要瓶颈已转向宏后多箱 room-slot 调度，"
                "需要参考解/teacher trace 锚定或更强的 room option，而不是只压缩单箱走廊。"
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
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a6_corridor_macro.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a6_corridor_macro.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = _A5.build_model(level, config)
    a5 = _A5.run_smoke(
        RoomLoopGuardGFTSDMCTSPlanner,
        model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=26047,
    )
    a6 = run_a6_smoke(
        model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        seed_offset=27059,
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a6_corridor_macro_v1",
            "failure_log": _rel(failure_path),
            "base_model": "SokobanRoomGFTSDModel",
            "a5_planner": RoomLoopGuardGFTSDMCTSPlanner.provider_id,
            "a6_planner": RoomCorridorMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "trace_scope": "short_a5_vs_a6_smoke",
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
        "room_static_summary": model.room_static_summary(),
        "a5_closed_loop_smoke": a5,
        "a6_closed_loop_smoke": a6,
        "comparison": compare_a5_a6(a5, a6),
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
