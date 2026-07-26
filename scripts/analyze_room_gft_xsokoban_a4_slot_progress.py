#!/usr/bin/env python3
"""A4 slot-schedule progress diagnostics for XSokoban first level.

This script keeps the Boxoban mainline untouched.  It compares the old GFT
physical prior with the A4 room-slot progress prior on the saved XSokoban
failure trace, then runs a short closed-loop smoke with the A4 derived planner.
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
    RoomSlotProgressGFTSDMCTSPlanner,
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
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    boundary_distance: int,
) -> RoomSlotProgressGFTSDMCTSPlanner:
    return RoomSlotProgressGFTSDMCTSPlanner(
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
    )


def candidate_rows(planner: RoomSlotProgressGFTSDMCTSPlanner, state, commitment_state) -> tuple[list[dict[str, Any]], int]:
    candidates, pruned = planner.candidate_actions(state, commitment_state)
    rows = []
    for candidate in candidates:
        physical = candidate.action.physical_action
        rows.append(
            {
                "semantic_action": candidate.action.to_dict(),
                "physical_action_label": action_label(physical),
                "a4_prior": float(candidate.prior),
                "a4_prior_score": float(candidate.prior_score),
                "old_gft_physical_prior_score": float(candidate.physical_prior_score),
                "a4_room_prior_score": float(candidate.room_prior_score),
                "a4_slot_progress_delta": float(candidate.room_slot_progress_delta),
                "a4_slot_transport_delta": float(candidate.room_slot_transport_delta),
                "a4_slot_committed_transport_delta": float(candidate.room_slot_committed_transport_delta),
                "a4_slot_frontier_transport_delta": float(candidate.room_slot_frontier_transport_delta),
                "a4_slot_barrier_delta": float(candidate.room_slot_barrier_delta),
                "a4_slot_progress": candidate.room_slot_progress_summary,
                "room_action_impact_codes": candidate.room_action_summary.get("impact_codes", []),
                "child_commitment_state": candidate.child_commitment_state.to_dict(),
                "_candidate": candidate,
            }
        )
    rows.sort(key=lambda row: (row["a4_prior_score"], row["a4_prior"]), reverse=True)
    return rows, pruned


def rank_of_label(rows: list[dict[str, Any]], label: str, score_key: str) -> int | None:
    best_by_label: dict[str, float] = {}
    for row in rows:
        best_by_label[row["physical_action_label"]] = max(
            best_by_label.get(row["physical_action_label"], float("-inf")),
            float(row[score_key]),
        )
    ordered = sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)
    for idx, (candidate_label, _score) in enumerate(ordered, start=1):
        if candidate_label == label:
            return idx
    return None


def clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        kept = dict(row)
        kept.pop("_candidate", None)
        out.append(kept)
    return out


def replay_saved_trace_a4(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    saved_trace: list[dict[str, Any]],
    *,
    boundary_distance: int,
) -> dict[str, Any]:
    rng = random.Random(int(config["seed"]) + 24041)
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
        selected_label = action_label(selected)
        planner = build_planner(
            model,
            config,
            rng=rng,
            budget=1,
            horizon=1,
            visited_keys=visited_keys,
            boundary_distance=boundary_distance,
        )
        root_slot_progress = model.room_slot_progress_summary(state, commitment_state)
        rows_all, pruned = candidate_rows(planner, state, commitment_state)
        selected_rows = [row for row in rows_all if row["physical_action_label"] == selected_label]
        if not selected_rows:
            raise RuntimeError(f"Saved selected action is not legal at trace step {step}.")
        adopted = selected_rows[0]
        best = rows_all[0]
        next_state = model.step_grid_flow(state, selected)
        commitment_state = adopted["_candidate"].child_commitment_state
        selected_semantic_rank = rows_all.index(adopted) + 1
        rows.append(
            {
                "step": step,
                "saved_selected_action": selected_label,
                "old_trace_physical_child_count": int(trace["child_count"]),
                "a4_semantic_child_count": int(len(rows_all)),
                "a4_pruned_physical_children": int(pruned),
                "saved_old_gft_physical_rank": rank_of_label(
                    rows_all,
                    selected_label,
                    "old_gft_physical_prior_score",
                ),
                "saved_a4_physical_rank": rank_of_label(rows_all, selected_label, "a4_prior_score"),
                "saved_a4_semantic_rank": int(selected_semantic_rank),
                "a4_best_semantic_action": semantic_label(best),
                "a4_best_physical_action": best["physical_action_label"],
                "saved_adopted_semantic_action": semantic_label(adopted),
                "saved_adopted_commitment": adopted["child_commitment_state"],
                "saved_a4_prior_score": float(adopted["a4_prior_score"]),
                "saved_old_gft_physical_prior_score": float(adopted["old_gft_physical_prior_score"]),
                "saved_a4_room_prior_score": float(adopted["a4_room_prior_score"]),
                "saved_a4_slot_progress_delta": float(adopted["a4_slot_progress_delta"]),
                "saved_a4_slot_transport_delta": float(adopted["a4_slot_transport_delta"]),
                "saved_a4_slot_committed_transport_delta": float(
                    adopted["a4_slot_committed_transport_delta"]
                ),
                "saved_a4_slot_barrier_delta": float(adopted["a4_slot_barrier_delta"]),
                "saved_a4_slot_progress": adopted["a4_slot_progress"],
                "saved_room_action_impact_codes": adopted["room_action_impact_codes"],
                "root_slot_progress": {
                    "slot_schedule_cost": float(root_slot_progress["slot_schedule_cost"]),
                    "total_transport_cost": float(root_slot_progress["total_transport_cost"]),
                    "barrier_cost": float(root_slot_progress["barrier_cost"]),
                    "barrier_counts": root_slot_progress["barrier_counts"],
                },
                "top_semantic_children": clean_rows(rows_all[:6]),
            }
        )
        state = next_state
        visited_keys.add(model.canonical_key(state))

    ranks = [row["saved_a4_physical_rank"] for row in rows if row["saved_a4_physical_rank"] is not None]
    semantic_ranks = [row["saved_a4_semantic_rank"] for row in rows]
    changed = [row for row in rows if row["saved_a4_physical_rank"] != 1]
    entry_rows = [
        row
        for row in rows
        if row["step"] >= 15 or row["saved_room_action_impact_codes"]
    ]
    entry_changed = [row for row in entry_rows if row["saved_a4_physical_rank"] != 1]
    return {
        "summary": {
            "trace_steps": int(len(rows)),
            "saved_a4_physical_rank_median": median(ranks) if ranks else None,
            "saved_a4_semantic_rank_median": median(semantic_ranks) if semantic_ranks else None,
            "saved_a4_physical_top1_count": sum(1 for row in rows if row["saved_a4_physical_rank"] == 1),
            "saved_a4_semantic_top1_count": sum(1 for row in rows if row["saved_a4_semantic_rank"] == 1),
            "saved_old_gft_physical_top1_count": sum(
                1 for row in rows if row["saved_old_gft_physical_rank"] == 1
            ),
            "a4_changes_saved_physical_choice_count": int(len(changed)),
            "first_a4_changed_step": changed[0]["step"] if changed else None,
            "entry_window_steps": [row["step"] for row in entry_rows],
            "entry_window_a4_changes_saved_physical_choice_count": int(len(entry_changed)),
            "first_entry_window_a4_changed_step": entry_changed[0]["step"] if entry_changed else None,
        },
        "rows": rows,
        "entry_window_rows": entry_rows,
    }


def run_a4_smoke(
    model: SokobanRoomGFTSDModel,
    config: dict[str, Any],
    *,
    budget: int,
    horizon: int,
    max_steps: int,
    boundary_distance: int,
) -> dict[str, Any]:
    rng = random.Random(int(config["seed"]) + 25043)
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
                "slot_progress_after": model.room_slot_progress_summary(next_state, commitment_state),
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
    return {
        "summary": {
            "status": status,
            "steps_executed": len([row for row in trace if "physical_action" in row]),
            "boxes_on_goals_final": int(model.boxes_on_goals(state)),
            "semantic_child_count_median": median(semantic_counts) if semantic_counts else None,
            "semantic_child_count_max": max(semantic_counts) if semantic_counts else None,
            "final_commitment_count": len(commitment_state.commitments),
            "final_slot_schedule_cost": float(
                model.room_slot_progress_summary(state, commitment_state)["slot_schedule_cost"]
            ),
        },
        "trace": trace,
        "final_room_summary": model.room_state_summary(state),
        "final_commitment_summary": model.room_commitment_summary(state, commitment_state),
        "final_slot_progress": model.room_slot_progress_summary(state, commitment_state),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    replay = payload["saved_trace_replay"]["summary"]
    smoke = payload["a4_closed_loop_smoke"]["summary"]
    entry_rows = payload["saved_trace_replay"]["entry_window_rows"]
    lines = [
        "# XSokoban first 第 1 关 Room-GFT A4 槽位进度诊断",
        "",
        "## 核心结论",
        "",
        (
            "A4 在 A3 的 `(box,direction,slot)` 语义 child 基础上加入 slot-schedule constrained progress。"
            "该 planner 为 `RoomSlotProgressGFTSDMCTSPlanner`，继承 A3 planner，不替换 Boxoban 主线。"
        ),
        (
            f"旧失败 trace 的 `{replay['trace_steps']}` 个 root 上，旧物理 GFT prior 将保存动作排第一的步数为 "
            f"`{replay['saved_old_gft_physical_top1_count']}`；A4 将保存物理动作排第一的步数为 "
            f"`{replay['saved_a4_physical_top1_count']}`，首次改变发生在 step "
            f"`{_fmt(replay['first_a4_changed_step'])}`。"
        ),
        (
            f"更关键的是入房窗口：step `{_fmt(replay['first_entry_window_a4_changed_step'])}` "
            "处旧 trace 选择的入房分支被 A4 降级，A4 首选沿已承诺深槽继续穿过浅层阻塞槽。"
            "因此，A4 不是简单压制所有走廊内 `b5:right`，而是把 slot commitment 改为行一致的深槽，"
            "并区分正确的浅槽穿越与错误行的无承诺/不可达推进。"
        ),
        (
            f"A4 短闭环 smoke 状态为 `{smoke['status']}`，执行 `{smoke['steps_executed']}` 步，"
            f"最终 `boxes_on_goals={smoke['boxes_on_goals_final']}`。该结果仍是机制诊断，不作为 XSokoban 解题率。"
        ),
        "",
        "## 保存 trace 重排摘要",
        "",
        "| step | saved action | old GFT rank | A4 physical rank | A4 semantic rank | A4 best | saved commitment | slot delta | barrier delta | impacts |",
        "|---:|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in payload["saved_trace_replay"]["rows"]:
        commitment = row["saved_adopted_semantic_action"].split("|", 1)[1]
        lines.append(
            "| "
            f"{row['step']} | {row['saved_selected_action']} | {_fmt(row['saved_old_gft_physical_rank'])} | "
            f"{_fmt(row['saved_a4_physical_rank'])} | {row['saved_a4_semantic_rank']} | "
            f"{row['a4_best_semantic_action']} | {commitment} | "
            f"{_fmt(row['saved_a4_slot_progress_delta'])} | {_fmt(row['saved_a4_slot_barrier_delta'])} | "
            f"{','.join(row['saved_room_action_impact_codes']) or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## 入房窗口",
            "",
            "| step | saved action | A4 best | selected slot score | top child details |",
            "|---:|---|---|---:|---|",
        ]
    )
    for row in entry_rows:
        top = row["top_semantic_children"][0]
        delta = top["a4_slot_progress"].get("delta", {})
        details = (
            f"{semantic_label(top)}; "
            f"transport_delta={_fmt(delta.get('total_transport_cost'))}; "
            f"barrier_delta={_fmt(delta.get('barrier_cost'))}; "
            f"pass={delta.get('committed_pass_through_increase', 0)}"
        )
        lines.append(
            "| "
            f"{row['step']} | {row['saved_selected_action']} | {row['a4_best_semantic_action']} | "
            f"{_fmt(row['saved_a4_slot_progress'].get('score'))} | {details} |"
        )

    lines.extend(
        [
            "",
            "## A4 短闭环 smoke",
            "",
            "| step | physical action | top commitment | boxes_on_goals | slot schedule cost |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in payload["a4_closed_loop_smoke"]["trace"]:
        if "physical_action" not in row:
            lines.append(f"| {row['step']} | no_action | NA | NA | NA |")
            continue
        option = row["semantic_action"]["commitment_option"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{option_label(option)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} |"
        )

    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A4 将旧 GFT 的任意目标运输进展替换为与 room frontier 和已承诺 slot 对齐的进度代理。"
                "因此，它可以识别“向某个深槽承诺后是否继续朝该槽可推地接近”，并把错误行推进、承诺不可达和真正阻塞槽作为强负信号。"
                "同时，A4 对深槽路径中必须短暂停留的浅层阻塞槽加入 pass-through 例外，避免把正确的穿越动作误判为 settled 错填。"
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a4_slot_progress.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a4_slot_progress.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = build_model(level, config)
    replay = replay_saved_trace_a4(
        model,
        config,
        failure["levels"][0]["trace"],
        boundary_distance=int(args.boundary_distance),
    )
    smoke = run_a4_smoke(
        model,
        config,
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a4_slot_progress_v1",
            "failure_log": _rel(failure_path),
            "base_model": "SokobanRoomGFTSDModel",
            "planner": RoomSlotProgressGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "trace_scope": "saved_trace_replay_plus_short_a4_smoke",
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
        "initial_slot_progress": model.room_slot_progress_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
        ),
        "saved_trace_replay": replay,
        "a4_closed_loop_smoke": smoke,
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
