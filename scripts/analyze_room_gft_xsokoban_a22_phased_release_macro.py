#!/usr/bin/env python3
"""A22 phased release macro diagnostics for XSokoban first levels.

A22 inherits A21 but turns stand-lock from a global guard into a milestone-gated
release/action macro objective.  The online solver does not receive teacher
actions or reference traces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A21_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a21_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a21_stand_lock_guarded_release.py",
)
if _A21_SPEC is None or _A21_SPEC.loader is None:
    raise RuntimeError("Unable to load A21 diagnostic helpers.")
_A21 = importlib.util.module_from_spec(_A21_SPEC)
_A21_SPEC.loader.exec_module(_A21)

from src.sokoban import (  # noqa: E402
    RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner,
    RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner,
    RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    return _A21._fmt(value)


def build_a22_planner(
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
    recoverability_probe_depth: int = 2,
    recoverability_probe_max_nodes: int = 8,
    survivable_recoverability_min_depth: int = 2,
    survivable_recoverability_min_legal_pushes: int = 3,
    stage_release_min_goals: int = 2,
    stage_stand_lock_score_scale: float = 0.35,
    stage_release_stand_occupied_resolution_weight: float = 85.0,
    stage_release_goal_nonregression: bool = True,
) -> RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner:
    return RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner(
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
        recoverability_probe_depth=int(recoverability_probe_depth),
        recoverability_probe_max_nodes=int(recoverability_probe_max_nodes),
        survivable_recoverability_min_depth=int(survivable_recoverability_min_depth),
        survivable_recoverability_min_legal_pushes=int(
            survivable_recoverability_min_legal_pushes
        ),
        root_recoverability_only=True,
        stage_release_min_goals=int(stage_release_min_goals),
        stage_stand_lock_score_scale=float(stage_stand_lock_score_scale),
        stage_release_stand_occupied_resolution_weight=float(
            stage_release_stand_occupied_resolution_weight
        ),
        stage_release_goal_nonregression=bool(stage_release_goal_nonregression),
    )


def augment_phased_release_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A21.augment_stand_lock_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    stage_rows = [row.get("chosen_stage", {}) for row in decision_rows]
    stage_names = [
        str(row.get("before", {}).get("stage"))
        for row in stage_rows
        if row.get("before", {}).get("stage") is not None
    ]
    release_macro_rows = []
    pre_release_macro_rows = []
    release_macro_steps = 0
    pre_release_macro_steps = 0
    goal_regression_rejections = 0
    for row in decision_rows:
        rows = row.get("corridor_macro", {}).get("rows", [])
        matched = [
            item
            for item in rows
            if item.get("reason") == "phased_release_macro_push"
            and item.get("bound_dependency", {}).get("phase_local_dependency_mode")
            != "pre_release_a20_compatible"
        ]
        pre_release_matched = [
            item
            for item in rows
            if item.get("reason") == "phased_release_macro_push"
            and item.get("bound_dependency", {}).get("phase_local_dependency_mode")
            == "pre_release_a20_compatible"
        ]
        if matched:
            release_macro_steps += 1
            release_macro_rows.extend(matched)
        if pre_release_matched:
            pre_release_macro_steps += 1
            pre_release_macro_rows.extend(pre_release_matched)
        for item in rows:
            progress = item.get("dependency_progress", {})
            violations = progress.get("global_feasibility_violations", [])
            if "goal_regression" in violations:
                goal_regression_rejections += 1
    result["summary"].update(
        {
            "stage_selected_count": int(len(stage_rows)),
            "stage_before_counts": dict(Counter(stage_names)),
            "release_stage_selected_count": int(stage_names.count("release")),
            "recovery_stage_selected_count": int(stage_names.count("recovery")),
            "phased_release_macro_trigger_steps": int(release_macro_steps),
            "phased_release_macro_push_count": int(len(release_macro_rows)),
            "pre_release_compatible_macro_trigger_steps": int(pre_release_macro_steps),
            "pre_release_compatible_macro_push_count": int(len(pre_release_macro_rows)),
            "phased_release_goal_regression_rejection_count": int(
                goal_regression_rejections
            ),
            "phased_release_bound_dependency_count": int(
                sum(bool(row.get("bound_dependency")) for row in release_macro_rows)
            ),
        }
    )
    return result


def run_a22_smoke(
    model,
    config: dict[str, Any],
    *,
    recovery_macro_max_extra_pushes: int,
    recovery_macro_max_nodes: int,
    recovery_macro_score_threshold: float,
    dependency_bound_macro_score_threshold: float,
    feasibility_guard_max_slot_cost_regression: float,
    feasibility_guard_max_schedule_barrier_regression: float,
    recoverability_probe_depth: int,
    recoverability_probe_max_nodes: int,
    survivable_recoverability_min_depth: int,
    survivable_recoverability_min_legal_pushes: int,
    stage_release_min_goals: int,
    stage_stand_lock_score_scale: float,
    stage_release_stand_occupied_resolution_weight: float,
    stage_release_goal_nonregression: bool,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a22_planner(
            *args,
            recovery_macro_max_extra_pushes=recovery_macro_max_extra_pushes,
            recovery_macro_max_nodes=recovery_macro_max_nodes,
            recovery_macro_score_threshold=recovery_macro_score_threshold,
            dependency_bound_macro_score_threshold=dependency_bound_macro_score_threshold,
            feasibility_guard_max_slot_cost_regression=feasibility_guard_max_slot_cost_regression,
            feasibility_guard_max_schedule_barrier_regression=(
                feasibility_guard_max_schedule_barrier_regression
            ),
            recoverability_probe_depth=recoverability_probe_depth,
            recoverability_probe_max_nodes=recoverability_probe_max_nodes,
            survivable_recoverability_min_depth=survivable_recoverability_min_depth,
            survivable_recoverability_min_legal_pushes=(
                survivable_recoverability_min_legal_pushes
            ),
            stage_release_min_goals=stage_release_min_goals,
            stage_stand_lock_score_scale=stage_stand_lock_score_scale,
            stage_release_stand_occupied_resolution_weight=(
                stage_release_stand_occupied_resolution_weight
            ),
            stage_release_goal_nonregression=stage_release_goal_nonregression,
            **builder_kwargs,
        )

    original = _A21._A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner
    _A21._A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner = builder
    try:
        result = _A21._A20._A19._A18._A17._A16._A15._A14._A13.run_a13_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A21._A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner = original
    return augment_phased_release_summary(result)


def compare_planners(a20: dict[str, Any], a21: dict[str, Any], a22: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    def first_diff(left: list[str], right: list[str]) -> int | None:
        for idx, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return idx
        if len(left) != len(right):
            return min(len(left), len(right))
        return None

    a20_actions = actions(a20)
    a21_actions = actions(a21)
    a22_actions = actions(a22)
    return {
        "a20_a22_first_decision_difference_step": first_diff(a20_actions, a22_actions),
        "a21_a22_first_decision_difference_step": first_diff(a21_actions, a22_actions),
        "a20_status": a20["summary"]["status"],
        "a21_status": a21["summary"]["status"],
        "a22_status": a22["summary"]["status"],
        "a20_boxes_on_goals_final": a20["summary"]["boxes_on_goals_final"],
        "a21_boxes_on_goals_final": a21["summary"]["boxes_on_goals_final"],
        "a22_boxes_on_goals_final": a22["summary"]["boxes_on_goals_final"],
        "a20_executed_pushes": a20["summary"]["executed_pushes"],
        "a21_executed_pushes": a21["summary"]["executed_pushes"],
        "a22_executed_pushes": a22["summary"]["executed_pushes"],
        "a20_final_slot_schedule_cost": a20["summary"]["final_slot_schedule_cost"],
        "a21_final_slot_schedule_cost": a21["summary"]["final_slot_schedule_cost"],
        "a22_final_slot_schedule_cost": a22["summary"]["final_slot_schedule_cost"],
        "a20_prefix_actions": a20_actions[:60],
        "a21_prefix_actions": a21_actions[:60],
        "a22_prefix_actions": a22_actions[:60],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a20 = payload["a20_closed_loop_smoke"]["summary"]
    a21 = payload["a21_closed_loop_smoke"]["summary"]
    a22 = payload["a22_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    if int(a22["boxes_on_goals_final"]) > int(a20["boxes_on_goals_final"]):
        outcome = "A22 是相对 A20 的正向结果。"
    elif int(a22["boxes_on_goals_final"]) == int(a20["boxes_on_goals_final"]):
        outcome = "A22 与 A20 目标数持平，主要检验阶段化消费是否避免 A21 的退化。"
    else:
        outcome = "A22 仍低于 A20，说明阶段化 release macro 的入口或验收条件仍需修订。"
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A22 phased release macro 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A22 继承 A21，但将 stand-lock 从全局强 guard 改为阶段化 release/action "
            "生成入口；默认在完成至少 2 个目标后才进入 release 阶段。"
        ),
        (
            f"A20 为 `{a20['status']}`、goals={a20['boxes_on_goals_final']}；"
            f"A21 为 `{a21['status']}`、goals={a21['boxes_on_goals_final']}；"
            f"A22 为 `{a22['status']}`、pushes={a22['executed_pushes']}、"
            f"decisions={a22['closed_loop_decision_steps']}、goals={a22['boxes_on_goals_final']}。"
        ),
        (
            f"A20/A22 首个决策差异为 `{cmp['a20_a22_first_decision_difference_step']}`；"
            f"A21/A22 首个决策差异为 `{cmp['a21_a22_first_decision_difference_step']}`。"
        ),
        outcome,
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | release-stage decisions | release macro steps | pre-release macro steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A20 | {a20['status']} | {a20['closed_loop_decision_steps']} | {a20['executed_pushes']} | "
            f"{a20['boxes_on_goals_final']} | {_fmt(a20['final_slot_schedule_cost'])} | "
            f"{a20['final_dependency_count']} | {a20['final_target_blocker_count']} | "
            f"{a20['final_release_action_count']} | NA | NA | NA |"
        ),
        (
            f"| A21 | {a21['status']} | {a21['closed_loop_decision_steps']} | {a21['executed_pushes']} | "
            f"{a21['boxes_on_goals_final']} | {_fmt(a21['final_slot_schedule_cost'])} | "
            f"{a21['final_dependency_count']} | {a21['final_target_blocker_count']} | "
            f"{a21['final_release_action_count']} | NA | NA | NA |"
        ),
        (
            f"| A22 | {a22['status']} | {a22['closed_loop_decision_steps']} | {a22['executed_pushes']} | "
            f"{a22['boxes_on_goals_final']} | {_fmt(a22['final_slot_schedule_cost'])} | "
            f"{a22['final_dependency_count']} | {a22['final_target_blocker_count']} | "
            f"{a22['final_release_action_count']} | {a22['release_stage_selected_count']} | "
            f"{a22['phased_release_macro_trigger_steps']} | "
            f"{a22['pre_release_compatible_macro_trigger_steps']} |"
        ),
        "",
        "## A22 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dependencies | locked after | phased macro |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["a22_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        stage = row.get("chosen_stage", {}).get("before", {}).get("stage", "NA")
        lock_after = row.get("chosen_stand_lock", {}).get("after", {})
        macro_rows = [
            item
            for item in row.get("corridor_macro", {}).get("rows", [])
            if item.get("reason") == "phased_release_macro_push"
            and item.get("bound_dependency", {}).get("phase_local_dependency_mode")
            != "pre_release_a20_compatible"
        ]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{stage} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{_fmt(lock_after.get('locked_stand_blocker_count'))} | "
            f"{len(macro_rows)} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A22 检验的是：复杂 Sokoban 决策机制是否应按里程碑阶段消费不同的 SD 摘要。"
                "若 A22 保留 A20 前缀而避免 A21 的目标数退化，则说明阶段化消费方向成立；"
                "若仍无法触发 release macro，则说明下一步需要修订阶段入口或依赖绑定条件。"
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
        "--a20-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--a21-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.json",
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
    parser.add_argument("--recoverability-probe-depth", type=int, default=2)
    parser.add_argument("--recoverability-probe-max-nodes", type=int, default=8)
    parser.add_argument("--survivable-recoverability-min-depth", type=int, default=2)
    parser.add_argument("--survivable-recoverability-min-legal-pushes", type=int, default=3)
    parser.add_argument("--stage-release-min-goals", type=int, default=2)
    parser.add_argument("--stage-stand-lock-score-scale", type=float, default=0.35)
    parser.add_argument("--stage-release-stand-occupied-resolution-weight", type=float, default=85.0)
    parser.add_argument(
        "--stage-release-goal-regression-allowed",
        action="store_true",
        help="Allow release macros to reduce the current number of boxes on goals.",
    )
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a20_path = ROOT / args.a20_json
    a21_path = ROOT / args.a21_json
    if not a20_path.exists():
        raise FileNotFoundError(f"A20 baseline JSON not found: {a20_path}")
    if not a21_path.exists():
        raise FileNotFoundError(f"A21 baseline JSON not found: {a21_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a20_payload = json.loads(a20_path.read_text(encoding="utf-8"))
    a21_payload = json.loads(a21_path.read_text(encoding="utf-8"))
    a20 = a20_payload["a20_closed_loop_smoke"]
    a21 = a21_payload["a21_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a22_model = _A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(level, config)
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
    a22 = run_a22_smoke(
        a22_model,
        config,
        seed_offset=46121,
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
        recoverability_probe_depth=int(args.recoverability_probe_depth),
        recoverability_probe_max_nodes=int(args.recoverability_probe_max_nodes),
        survivable_recoverability_min_depth=int(args.survivable_recoverability_min_depth),
        survivable_recoverability_min_legal_pushes=int(
            args.survivable_recoverability_min_legal_pushes
        ),
        stage_release_min_goals=int(args.stage_release_min_goals),
        stage_stand_lock_score_scale=float(args.stage_stand_lock_score_scale),
        stage_release_stand_occupied_resolution_weight=float(
            args.stage_release_stand_occupied_resolution_weight
        ),
        stage_release_goal_nonregression=not bool(args.stage_release_goal_regression_allowed),
        **common,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a22_phased_release_macro_v1",
            "failure_log": _rel(failure_path),
            "a20_json": _rel(a20_path),
            "a21_json": _rel(a21_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a20_planner": RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a21_planner": RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a22_planner": RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a20_a21_vs_a22_phased_release_macro_smoke",
        },
        "a20_config_extract": a20_payload.get("config_extract", {}),
        "a21_config_extract": a21_payload.get("config_extract", {}),
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
            "recoverability_probe_depth": int(args.recoverability_probe_depth),
            "recoverability_probe_max_nodes": int(args.recoverability_probe_max_nodes),
            "survivable_recoverability_min_depth": int(
                args.survivable_recoverability_min_depth
            ),
            "survivable_recoverability_min_legal_pushes": int(
                args.survivable_recoverability_min_legal_pushes
            ),
            "stage_release_min_goals": int(args.stage_release_min_goals),
            "stage_stand_lock_score_scale": float(args.stage_stand_lock_score_scale),
            "stage_release_stand_occupied_resolution_weight": float(
                args.stage_release_stand_occupied_resolution_weight
            ),
            "stage_release_goal_nonregression": not bool(args.stage_release_goal_regression_allowed),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a22_model.room_static_summary(),
        "initial_schedule_plan": a22_model.room_schedule_plan_summary(
            a22_model.initial_state,
            a22_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a20_closed_loop_smoke": a20,
        "a21_closed_loop_smoke": a21,
        "a22_closed_loop_smoke": a22,
        "comparison": compare_planners(a20, a21, a22),
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
