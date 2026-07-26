#!/usr/bin/env python3
"""A17 phase-progress root-guard diagnostics for XSokoban first levels.

A17 inherits A16 feasibility-guarded dependency macros and adds a root-level
schedule-phase no-progress guard.  The online solver does not receive teacher
actions or reference traces.
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

_A16_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a16_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a16_feasibility_guarded_macro.py",
)
if _A16_SPEC is None or _A16_SPEC.loader is None:
    raise RuntimeError("Unable to load A16 diagnostic helpers.")
_A16 = importlib.util.module_from_spec(_A16_SPEC)
_A16_SPEC.loader.exec_module(_A16)

from src.sokoban import (  # noqa: E402
    RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner,
    RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner,
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


def build_a17_planner(
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
    recovery_macro_max_nodes: int = 20,
    recovery_macro_score_threshold: float = 2.0,
    dependency_bound_macro_score_threshold: float = 20.0,
    feasibility_guard_max_slot_cost_regression: float = 0.0,
    feasibility_guard_max_schedule_barrier_regression: float = 8.0,
) -> RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner:
    return RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner(
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
        dependency_bound_macro_score_threshold=float(dependency_bound_macro_score_threshold),
        feasibility_guard_max_slot_cost_regression=float(
            feasibility_guard_max_slot_cost_regression
        ),
        feasibility_guard_max_schedule_barrier_regression=float(
            feasibility_guard_max_schedule_barrier_regression
        ),
    )


def augment_phase_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A16.augment_feasibility_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    phase_rows = [row.get("chosen_phase_progress", {}) for row in decision_rows]
    scores = [float(row.get("score", 0.0)) for row in phase_rows]
    result["summary"].update(
        {
            "phase_progress_selected_count": int(
                sum(bool(row.get("phase_progress")) for row in phase_rows)
            ),
            "phase_no_progress_selected_count": int(
                sum(bool(row.get("no_progress")) for row in phase_rows)
            ),
            "phase_stagnant_context_selected_count": int(
                sum(bool(row.get("stagnant_context")) for row in phase_rows)
            ),
            "phase_repeated_edge_selected_count": int(
                sum(bool(row.get("repeated_edge")) for row in phase_rows)
            ),
            "phase_reverse_edge_selected_count": int(
                sum(bool(row.get("reverse_recent_edge")) for row in phase_rows)
            ),
            "phase_repeated_signature_selected_count": int(
                sum(bool(row.get("child_repeated_schedule_signature")) for row in phase_rows)
            ),
            "phase_progress_score_median": None if not scores else float(median(scores)),
            "phase_progress_score_min": None if not scores else float(min(scores)),
            "phase_progress_score_max": None if not scores else float(max(scores)),
        }
    )
    return result


def run_a17_smoke(
    model,
    config: dict[str, Any],
    *,
    recovery_macro_max_extra_pushes: int,
    recovery_macro_max_nodes: int,
    recovery_macro_score_threshold: float,
    dependency_bound_macro_score_threshold: float,
    feasibility_guard_max_slot_cost_regression: float,
    feasibility_guard_max_schedule_barrier_regression: float,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a17_planner(
            *args,
            recovery_macro_max_extra_pushes=recovery_macro_max_extra_pushes,
            recovery_macro_max_nodes=recovery_macro_max_nodes,
            recovery_macro_score_threshold=recovery_macro_score_threshold,
            dependency_bound_macro_score_threshold=dependency_bound_macro_score_threshold,
            feasibility_guard_max_slot_cost_regression=feasibility_guard_max_slot_cost_regression,
            feasibility_guard_max_schedule_barrier_regression=(
                feasibility_guard_max_schedule_barrier_regression
            ),
            **builder_kwargs,
        )

    original = _A16._A15._A14._A13.build_a13_planner
    _A16._A15._A14._A13.build_a13_planner = builder
    try:
        result = _A16._A15._A14._A13.run_a13_smoke(model, config, **kwargs)
    finally:
        _A16._A15._A14._A13.build_a13_planner = original
    return augment_phase_summary(result)


def compare_a16_a17(a16: dict[str, Any], a17: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a16_actions = actions(a16)
    a17_actions = actions(a17)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a16_actions, a17_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a16_actions) != len(a17_actions):
        first_diff = min(len(a16_actions), len(a17_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a16_status": a16["summary"]["status"],
        "a17_status": a17["summary"]["status"],
        "a16_boxes_on_goals_final": a16["summary"]["boxes_on_goals_final"],
        "a17_boxes_on_goals_final": a17["summary"]["boxes_on_goals_final"],
        "a16_executed_pushes": a16["summary"]["executed_pushes"],
        "a17_executed_pushes": a17["summary"]["executed_pushes"],
        "a16_final_slot_schedule_cost": a16["summary"]["final_slot_schedule_cost"],
        "a17_final_slot_schedule_cost": a17["summary"]["final_slot_schedule_cost"],
        "a16_prefix_actions": a16_actions[:40],
        "a17_prefix_actions": a17_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a16 = payload["a16_closed_loop_smoke"]["summary"]
    a17 = payload["a17_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A17 phase-progress guard 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A17 继承 A16 的 feasibility-guarded macro，并在根选择层加入 schedule-phase "
            "no-progress guard，用于惩罚 dependency/slot phase 不变时的重复动作和反向动作。"
        ),
        (
            f"A16 对照为 `{a16['status']}`，执行 `{a16['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a16['boxes_on_goals_final']}`；"
            f"A17 smoke 为 `{a17['status']}`，执行 `{a17['executed_pushes']}` 个 push、"
            f"`{a17['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a17['boxes_on_goals_final']}`。"
        ),
        (
            f"A16/A17 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A17 phase no-progress selections 为 `{a17['phase_no_progress_selected_count']}`，"
            f"stagnant selections 为 `{a17['phase_stagnant_context_selected_count']}`。"
        ),
        "",
        "## A16/A17 摘要",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | no-progress selections |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A16 | {a16['status']} | {a16['closed_loop_decision_steps']} | {a16['executed_pushes']} | "
            f"{a16['boxes_on_goals_final']} | {_fmt(a16['final_slot_schedule_cost'])} | "
            f"{a16['final_dependency_count']} | {a16['final_target_blocker_count']} | NA |"
        ),
        (
            f"| A17 | {a17['status']} | {a17['closed_loop_decision_steps']} | {a17['executed_pushes']} | "
            f"{a17['boxes_on_goals_final']} | {_fmt(a17['final_slot_schedule_cost'])} | "
            f"{a17['final_dependency_count']} | {a17['final_target_blocker_count']} | "
            f"{a17['phase_no_progress_selected_count']} |"
        ),
        "",
        "## A17 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | phase progress | no progress | stagnant | phase score |",
        "|---:|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for row in payload["a17_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | NA | NA | NA |"
            )
            continue
        phase = row.get("chosen_phase_progress", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{phase.get('phase_progress', False)} | "
            f"{phase.get('no_progress', False)} | "
            f"{phase.get('stagnant_context', False)} | "
            f"{_fmt(row.get('chosen_phase_progress_score'))} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A17 检验的是：仅过滤宏执行结果不足以避免根选择层长循环，"
                "还需要对 schedule phase 无进展的重复动作施加直接约束。"
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
    parser.add_argument(
        "--a16-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.json",
    )
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=80)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument("--schedule-lookahead-slots", type=int, default=2)
    parser.add_argument("--recent-edge-window", type=int, default=12)
    parser.add_argument("--recovery-min-legal-pushes", type=int, default=2)
    parser.add_argument("--recovery-macro-max-extra-pushes", type=int, default=4)
    parser.add_argument("--recovery-macro-max-nodes", type=int, default=20)
    parser.add_argument("--recovery-macro-score-threshold", type=float, default=2.0)
    parser.add_argument("--dependency-bound-macro-score-threshold", type=float, default=20.0)
    parser.add_argument("--feasibility-guard-max-slot-cost-regression", type=float, default=0.0)
    parser.add_argument("--feasibility-guard-max-schedule-barrier-regression", type=float, default=8.0)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a16_path = ROOT / args.a16_json
    if not a16_path.exists():
        raise FileNotFoundError(f"A16 baseline JSON not found: {a16_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a16_payload = json.loads(a16_path.read_text(encoding="utf-8"))
    a16 = a16_payload["a16_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a17_model = _A16._A15._A14._A13._A12.build_model(level, config)
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
    a17 = run_a17_smoke(
        a17_model,
        config,
        seed_offset=38327,
        recovery_macro_max_extra_pushes=int(args.recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(args.recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(args.recovery_macro_score_threshold),
        dependency_bound_macro_score_threshold=float(args.dependency_bound_macro_score_threshold),
        feasibility_guard_max_slot_cost_regression=float(
            args.feasibility_guard_max_slot_cost_regression
        ),
        feasibility_guard_max_schedule_barrier_regression=float(
            args.feasibility_guard_max_schedule_barrier_regression
        ),
        **common,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a17_phase_progress_guarded_macro_v1",
            "failure_log": _rel(failure_path),
            "a16_json": _rel(a16_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a16_planner": RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a17_planner": RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a16_feasibility_guarded_macro_vs_a17_phase_progress_guarded_macro_smoke",
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
            "dependency_bound_macro_score_threshold": float(
                args.dependency_bound_macro_score_threshold
            ),
            "feasibility_guard_max_slot_cost_regression": float(
                args.feasibility_guard_max_slot_cost_regression
            ),
            "feasibility_guard_max_schedule_barrier_regression": float(
                args.feasibility_guard_max_schedule_barrier_regression
            ),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a17_model.room_static_summary(),
        "initial_schedule_plan": a17_model.room_schedule_plan_summary(
            a17_model.initial_state,
            a17_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a16_closed_loop_smoke": a16,
        "a17_closed_loop_smoke": a17,
        "comparison": compare_a16_a17(a16, a17),
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
