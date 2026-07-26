#!/usr/bin/env python3
"""A3 semantic-commitment child diagnostics for XSokoban first level.

This script isolates the A3 mechanism: a physical push can generate multiple
semantic children, each carrying a different room-slot commitment state.
It keeps the existing Boxoban GFT-SD-MCTS evaluator untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import (
    RoomCommitmentState,
    RoomSemanticGFTSDMCTSPlanner,
    SokobanPushAction,
    SokobanRoomGFTSDModel,
    load_boxoban_levels,
    semantic_child_count_by_physical,
)
from src.sokoban.env import ACTION_NAMES


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


def build_model(level, config: dict[str, Any]) -> SokobanRoomGFTSDModel:
    return SokobanRoomGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def build_planner(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    include_uncommitted_option: bool,
    boundary_distance: int,
) -> RoomSemanticGFTSDMCTSPlanner:
    return RoomSemanticGFTSDMCTSPlanner(
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
        include_uncommitted_semantic_option=include_uncommitted_option,
        semantic_commitment_boundary_distance=int(boundary_distance),
    )


def replay_saved_trace(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    saved_trace: list[dict[str, Any]],
    *,
    include_uncommitted_option: bool,
    boundary_distance: int,
) -> dict[str, Any]:
    rng = random.Random(int(config["seed"]))
    state = model.initial_state
    commitment_state = model.empty_room_commitment_state()
    visited_keys = {model.canonical_key(state)}
    rows: list[dict[str, Any]] = []

    for trace in saved_trace:
        step = int(trace["step"])
        selected = SokobanPushAction(
            box_index=int(trace["action"]["box_index"]),
            direction=int(trace["action"]["direction"]),
        )
        planner = build_planner(
            model,
            config,
            rng=rng,
            budget=1,
            horizon=1,
            visited_keys=visited_keys,
            include_uncommitted_option=include_uncommitted_option,
            boundary_distance=boundary_distance,
        )
        candidates, pruned = planner.candidate_actions(state, commitment_state)
        physical_labels = {
            action_label(candidate.action.physical_action)
            for candidate in candidates
        }
        by_physical = semantic_child_count_by_physical(candidates)
        selected_candidates = [
            candidate
            for candidate in candidates
            if candidate.action.physical_action.box_index == selected.box_index
            and candidate.action.physical_action.direction == selected.direction
        ]
        if not selected_candidates:
            raise RuntimeError(f"Saved selected action is not legal at trace step {step}.")
        selected_candidates.sort(key=lambda item: (item.prior_score, item.prior), reverse=True)
        adopted = selected_candidates[0]
        next_state = model.step_grid_flow(state, selected)
        commitment_state = adopted.child_commitment_state
        rows.append(
            {
                "step": step,
                "saved_selected_action": action_label(selected),
                "physical_child_count_log": int(trace["child_count"]),
                "physical_child_count_recomputed": len(physical_labels),
                "semantic_child_count": len(candidates),
                "semantic_child_count_by_physical": by_physical,
                "semantic_gain_over_log_child_count": len(candidates) - int(trace["child_count"]),
                "pruned_physical_children": int(pruned),
                "selected_physical_semantic_options": len(selected_candidates),
                "selected_adopted_semantic_action": adopted.action.to_dict(),
                "selected_adopted_prior_score": float(adopted.prior_score),
                "selected_adopted_room_prior_score": float(adopted.room_prior_score),
                "selected_room_action_impact_codes": adopted.room_action_summary.get("impact_codes", []),
                "commitment_after": commitment_state.to_dict(),
                "commitment_summary_after": model.room_commitment_summary(next_state, commitment_state),
                "room_summary_after": model.room_state_summary(next_state),
            }
        )
        state = next_state
        visited_keys.add(model.canonical_key(state))

    semantic_counts = [row["semantic_child_count"] for row in rows]
    selected_option_counts = [row["selected_physical_semantic_options"] for row in rows]
    return {
        "summary": {
            "trace_steps": len(rows),
            "semantic_child_count_median": median(semantic_counts) if semantic_counts else None,
            "semantic_child_count_max": max(semantic_counts) if semantic_counts else None,
            "semantic_child_count_mean": mean(semantic_counts) if semantic_counts else None,
            "selected_physical_multi_option_steps": sum(1 for value in selected_option_counts if value > 1),
            "selected_physical_semantic_options_max": max(selected_option_counts) if selected_option_counts else None,
            "single_physical_child_steps_with_semantic_split": sum(
                1
                for row in rows
                if row["physical_child_count_log"] == 1 and row["semantic_child_count"] > 1
            ),
        },
        "rows": rows,
    }


def run_semantic_smoke(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    budget: int,
    horizon: int,
    max_steps: int,
    include_uncommitted_option: bool,
    boundary_distance: int,
) -> dict[str, Any]:
    rng = random.Random(int(config["seed"]) + 17017)
    state = model.initial_state
    commitment_state: RoomCommitmentState = model.empty_room_commitment_state()
    visited_keys = {model.canonical_key(state)}
    trace: list[dict[str, Any]] = []
    status = "max_steps"

    for step in range(max_steps):
        if model.is_solved(state):
            status = "solved"
            break
        planner = build_planner(
            model,
            config,
            rng=rng,
            budget=budget,
            horizon=horizon,
            visited_keys=visited_keys,
            include_uncommitted_option=include_uncommitted_option,
            boundary_distance=boundary_distance,
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
        next_state = model.step_grid_flow(state, physical)
        commitment_state = plan["next_commitment_state"]
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
                "boxes_on_goals_after": int(model.boxes_on_goals(next_state)),
                "commitment_after": commitment_state.to_dict(),
                "commitment_summary_after": model.room_commitment_summary(next_state, commitment_state),
                "room_summary_after": model.room_state_summary(next_state),
            }
        )
        state = next_state
        visited_keys.add(model.canonical_key(state))
        if model.is_deadlocked(state) and not model.is_solved(state):
            status = "deadlocked"
            break
    if model.is_solved(state):
        status = "solved"
    semantic_counts = [row.get("semantic_child_count", 0) for row in trace if "semantic_child_count" in row]
    physical_counts = [row.get("physical_child_count", 0) for row in trace if "physical_child_count" in row]
    return {
        "summary": {
            "status": status,
            "steps_executed": len([row for row in trace if "physical_action" in row]),
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "semantic_child_count_median": median(semantic_counts) if semantic_counts else None,
            "semantic_child_count_max": max(semantic_counts) if semantic_counts else None,
            "physical_child_count_median": median(physical_counts) if physical_counts else None,
            "final_commitment_count": len(commitment_state.commitments),
        },
        "trace": trace,
        "final_room_summary": model.room_state_summary(state),
        "final_commitment_summary": model.room_commitment_summary(state, commitment_state),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    replay = payload["saved_trace_replay"]["summary"]
    smoke = payload["semantic_closed_loop_smoke"]["summary"]
    lines = [
        "# XSokoban first 第 1 关 Room-GFT A3 语义 child 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A3 已将物理 push 扩展为 `(box,direction,slot)` 语义 child，并把 room-slot commitment "
            "纳入 MCTS node identity。该实现位于派生 hard-instance planner 中，不修改现有 Boxoban 主线 planner。"
        ),
        (
            f"在旧失败 trace 的 `{replay['trace_steps']}` 个 root 上重放枚举时，semantic child 中位数为 "
            f"`{_fmt(replay['semantic_child_count_median'])}`，最大为 `{_fmt(replay['semantic_child_count_max'])}`。"
            f"旧 trace 中物理 child 为 1 但 A3 产生语义分裂的步数为 "
            f"`{replay['single_physical_child_steps_with_semantic_split']}`；"
            f"被选物理动作拥有多个 slot 选项的步数为 `{replay['selected_physical_multi_option_steps']}`。"
            "这表明 A3 已能在 commitment boundary 处提前展开 slot 承诺，但长走廊内部一旦承诺确定，"
            "仍会表现为单一后继；这部分需要 A4 的 slot-schedule progress 或 forced-corridor option 继续处理。"
        ),
        (
            f"短闭环 semantic smoke 的状态为 `{smoke['status']}`，执行 `{smoke['steps_executed']}` 步，"
            f"最终 `boxes_on_goals={smoke['boxes_on_goals_final']}`。该 smoke 只验证 A3 分支机制和承诺状态推进，"
            "不作为 XSokoban 解题率结果。"
        ),
        "",
        "## 保存 trace 上的语义分裂",
        "",
        "| step | saved action | physical log | semantic children | selected options | adopted commitment | impact codes |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in payload["saved_trace_replay"]["rows"]:
        option = row["selected_adopted_semantic_action"]["commitment_option"]
        if option["kind"] == "slot":
            commitment = f"room {option['room_id']} slot {option['slot_id']} {option['slot_cell']}"
        else:
            commitment = option["kind"]
        lines.append(
            "| "
            f"{row['step']} | {row['saved_selected_action']} | {row['physical_child_count_log']} | "
            f"{row['semantic_child_count']} | {row['selected_physical_semantic_options']} | "
            f"{commitment} | {','.join(row['selected_room_action_impact_codes']) or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## 短闭环 smoke",
            "",
            "| step | physical action | semantic children | top commitment | boxes_on_goals | committed boxes |",
            "|---:|---|---:|---|---:|---:|",
        ]
    )
    for row in payload["semantic_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            lines.append(f"| {row['step']} | no_action | 0 | NA | NA | NA |")
            continue
        option = row["semantic_action"]["commitment_option"]
        if option["kind"] == "slot":
            commitment = f"slot {option['slot_id']} {option['slot_cell']}"
        else:
            commitment = option["kind"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row['semantic_child_count']} | {commitment} | {row['boxes_on_goals_after']} | "
            f"{row['commitment_summary_after']['committed_box_count']} |"
        )

    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A3 开始处理 `child_count=1` 掩盖战略分支的问题：同一物理后继可以因为不同 room-slot 承诺而进入不同节点。"
                "当前版本仍是最小机制实现，commitment 以箱子当前位置和 slot 绑定表示，并只在 room boundary 或 entry "
                "邻域触发语义分裂；后续 A4 应把运输进展从任意 goal "
                "距离改为 slot-schedule constrained progress，并进一步强化非 frontier 填充和无承诺 entry 的 barrier。"
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
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--include-uncommitted-option", action="store_true")
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a3_semantic_children.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a3_semantic_children.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = build_model(level, config)
    replay = replay_saved_trace(
        model,
        config,
        failure["levels"][0]["trace"],
        include_uncommitted_option=bool(args.include_uncommitted_option),
        boundary_distance=int(args.boundary_distance),
    )
    smoke = run_semantic_smoke(
        model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        include_uncommitted_option=bool(args.include_uncommitted_option),
        boundary_distance=int(args.boundary_distance),
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a3_semantic_children_v1",
            "failure_log": _rel(failure_path),
            "base_model": "SokobanRoomGFTSDModel",
            "planner": RoomSemanticGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "trace_scope": "saved_trace_prefix_plus_short_semantic_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "include_uncommitted_option": bool(args.include_uncommitted_option),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": model.room_static_summary(),
        "saved_trace_replay": replay,
        "semantic_closed_loop_smoke": smoke,
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
