#!/usr/bin/env python3
"""A14 recovery-macro diagnostics for XSokoban first levels.

A14 keeps A11 schedule-plan SD, A12 dependency-stall guard, and A13 access
recovery margin.  It adds a bounded release/access recovery macro after the
selected root action.  No teacher actions or reference traces are used online.
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

_A13_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a13_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a13_access_recovery.py",
)
if _A13_SPEC is None or _A13_SPEC.loader is None:
    raise RuntimeError("Unable to load A13 diagnostic helpers.")
_A13 = importlib.util.module_from_spec(_A13_SPEC)
_A13_SPEC.loader.exec_module(_A13)

from src.sokoban import (  # noqa: E402
    RoomScheduleAccessRecoveryGFTSDMCTSPlanner,
    RoomScheduleRecoveryMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


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


def build_a14_planner(
    model,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    boundary_distance: int,
    recent_moved_edges: list[tuple],
    recent_schedule_keys: set[tuple],
    previous_physical_action=None,
    previous_moved_source=None,
    previous_moved_target=None,
    recent_commitment_keys: set | None = None,
    corridor_macro_max_extra_pushes: int = 8,
    corridor_macro_min_total_pushes: int = 2,
    schedule_lookahead_slots: int = 2,
    recovery_min_legal_pushes: int = 2,
    recovery_macro_max_extra_pushes: int = 4,
    recovery_macro_max_nodes: int = 400,
    recovery_macro_score_threshold: float = 2.0,
) -> RoomScheduleRecoveryMacroGFTSDMCTSPlanner:
    return RoomScheduleRecoveryMacroGFTSDMCTSPlanner(
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
        room_access_recovery_min_legal_pushes=int(recovery_min_legal_pushes),
        recovery_macro_max_extra_pushes=int(recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(recovery_macro_score_threshold),
    )


def augment_macro_summary(result: dict[str, Any]) -> dict[str, Any]:
    trace = result["trace"]
    macro_payloads = [
        row.get("corridor_macro", {})
        for row in trace
        if "physical_action" in row
    ]
    triggered = [
        macro
        for macro in macro_payloads
        if any(row.get("reason") == "access_recovery_macro_push" for row in macro.get("rows", []))
    ]
    scores = [
        max(float(row.get("recovery_score", 0.0)) for row in macro.get("rows", []))
        for macro in triggered
    ]
    extra = [
        sum(1 for row in macro.get("rows", []) if row.get("reason") == "access_recovery_macro_push")
        for macro in triggered
    ]
    result["summary"].update(
        {
            "recovery_macro_candidate_steps": int(len(macro_payloads)),
            "recovery_macro_trigger_steps": int(len(triggered)),
            "recovery_macro_total_extra_pushes": int(sum(extra)),
            "recovery_macro_score_median": None if not scores else float(median(scores)),
            "recovery_macro_score_max": None if not scores else float(max(scores)),
        }
    )
    return result


def run_a14_smoke(
    model,
    config: dict[str, Any],
    *,
    recovery_macro_max_extra_pushes: int,
    recovery_macro_max_nodes: int,
    recovery_macro_score_threshold: float,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a14_planner(
            *args,
            recovery_macro_max_extra_pushes=recovery_macro_max_extra_pushes,
            recovery_macro_max_nodes=recovery_macro_max_nodes,
            recovery_macro_score_threshold=recovery_macro_score_threshold,
            **builder_kwargs,
        )

    original = _A13.build_a13_planner
    _A13.build_a13_planner = builder
    try:
        result = _A13.run_a13_smoke(model, config, **kwargs)
    finally:
        _A13.build_a13_planner = original
    return augment_macro_summary(result)


def compare_a13_a14(a13: dict[str, Any], a14: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a13_actions = actions(a13)
    a14_actions = actions(a14)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a13_actions, a14_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a13_actions) != len(a14_actions):
        first_diff = min(len(a13_actions), len(a14_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a13_status": a13["summary"]["status"],
        "a14_status": a14["summary"]["status"],
        "a13_boxes_on_goals_final": a13["summary"]["boxes_on_goals_final"],
        "a14_boxes_on_goals_final": a14["summary"]["boxes_on_goals_final"],
        "a13_executed_pushes": a13["summary"]["executed_pushes"],
        "a14_executed_pushes": a14["summary"]["executed_pushes"],
        "a13_final_slot_schedule_cost": a13["summary"]["final_slot_schedule_cost"],
        "a14_final_slot_schedule_cost": a14["summary"]["final_slot_schedule_cost"],
        "a13_prefix_actions": a13_actions[:40],
        "a14_prefix_actions": a14_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a13 = payload["a13_closed_loop_smoke"]["summary"]
    a14 = payload["a14_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A14 recovery macro 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A14 保留 A11-A13 的 SD 与消费端信号，并在根动作后追加 bounded release/access "
            "recovery macro。该版本不读取 teacher 动作表。"
        ),
        (
            f"A13 smoke 为 `{a13['status']}`，执行 `{a13['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a13['boxes_on_goals_final']}`；"
            f"A14 smoke 为 `{a14['status']}`，执行 `{a14['executed_pushes']}` 个 push、"
            f"`{a14['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a14['boxes_on_goals_final']}`。"
        ),
        (
            f"A13/A14 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A14 最终 slot cost 为 `{_fmt(a14['final_slot_schedule_cost'])}`，"
            f"recovery macro 触发 `{a14['recovery_macro_trigger_steps']}` 次，"
            f"追加 `{a14['recovery_macro_total_extra_pushes']}` 个 push。"
        ),
        "",
        "## A13/A14 摘要",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | recovery macro triggers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A13 | {a13['status']} | {a13['closed_loop_decision_steps']} | {a13['executed_pushes']} | "
            f"{a13['boxes_on_goals_final']} | {_fmt(a13['final_slot_schedule_cost'])} | "
            f"{a13['final_dependency_count']} | {a13['final_target_blocker_count']} | NA |"
        ),
        (
            f"| A14 | {a14['status']} | {a14['closed_loop_decision_steps']} | {a14['executed_pushes']} | "
            f"{a14['boxes_on_goals_final']} | {_fmt(a14['final_slot_schedule_cost'])} | "
            f"{a14['final_dependency_count']} | {a14['final_target_blocker_count']} | "
            f"{a14['recovery_macro_trigger_steps']} |"
        ),
        "",
        "## A14 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | recovery macro | macro extra | macro score |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["a14_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | 0 | NA |"
            )
            continue
        macro = row.get("corridor_macro", {})
        recovery_rows = [
            macro_row
            for macro_row in macro.get("rows", [])
            if macro_row.get("reason") == "access_recovery_macro_push"
        ]
        recovery_triggered = bool(recovery_rows)
        recovery_extra = len(recovery_rows)
        recovery_score = None if not recovery_rows else max(
            float(macro_row.get("recovery_score", 0.0))
            for macro_row in recovery_rows
        )
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{recovery_triggered} | "
            f"{recovery_extra} | "
            f"{_fmt(recovery_score)} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A14 检验的是：从 A13 的低合法余量诊断进入可执行短宏动作后，是否能恢复站位、"
                "移开 target blocker 或打开 release action。"
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
    parser.add_argument("--recovery-min-legal-pushes", type=int, default=2)
    parser.add_argument("--recovery-macro-max-extra-pushes", type=int, default=4)
    parser.add_argument("--recovery-macro-max-nodes", type=int, default=400)
    parser.add_argument("--recovery-macro-score-threshold", type=float, default=2.0)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a14_recovery_macro_l002_s200.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a14_recovery_macro_l002_s200.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a13_model = _A13._A12.build_model(level, config)
    a14_model = _A13._A12.build_model(level, config)
    common = dict(
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
        schedule_lookahead_slots=int(args.schedule_lookahead_slots),
        recent_edge_window=int(args.recent_edge_window),
        recovery_min_legal_pushes=int(args.recovery_min_legal_pushes),
    )
    a13 = _A13.run_a13_smoke(
        a13_model,
        config,
        seed_offset=33521,
        **common,
    )
    a14 = run_a14_smoke(
        a14_model,
        config,
        seed_offset=34721,
        recovery_macro_max_extra_pushes=int(args.recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(args.recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(args.recovery_macro_score_threshold),
        **common,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a14_recovery_macro_v1",
            "failure_log": _rel(failure_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a13_planner": RoomScheduleAccessRecoveryGFTSDMCTSPlanner.provider_id,
            "a14_planner": RoomScheduleRecoveryMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a13_access_recovery_guard_vs_a14_recovery_macro_smoke",
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
            "recovery_min_legal_pushes": int(args.recovery_min_legal_pushes),
            "recovery_macro_max_extra_pushes": int(args.recovery_macro_max_extra_pushes),
            "recovery_macro_max_nodes": int(args.recovery_macro_max_nodes),
            "recovery_macro_score_threshold": float(args.recovery_macro_score_threshold),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a14_model.room_static_summary(),
        "initial_schedule_plan": a14_model.room_schedule_plan_summary(
            a14_model.initial_state,
            a14_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a13_closed_loop_smoke": a13,
        "a14_closed_loop_smoke": a14,
        "comparison": compare_a13_a14(a13, a14),
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
