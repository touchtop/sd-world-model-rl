#!/usr/bin/env python3
"""A31 goal-lock predecessor release diagnostics for XSokoban first levels.

A31 inherits A30 and moves the goal-lock macro failure signal forward: when
A30 would truncate a release macro because it creates an unreleasable
goal-locked stand blocker, A31 tries to replace the root action with a
same-root candidate that avoids the lock. It does not use teacher actions or
reference traces online.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A30_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a30_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a30_controlled_release_regression.py",
)
if _A30_SPEC is None or _A30_SPEC.loader is None:
    raise RuntimeError("Unable to load A30 diagnostic helpers.")
_A30 = importlib.util.module_from_spec(_A30_SPEC)
_A30_SPEC.loader.exec_module(_A30)

from src.sokoban import (  # noqa: E402
    RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner,
    RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner,
    RoomScheduleReleaseActionMacroGFTSDMCTSPlanner,
    RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner,
    RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner,
    RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner,
    RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    return _A30._fmt(value)


def augment_a31_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A30.augment_a30_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    guard_rows = [
        row
        for row in decision_rows
        if bool(
            row.get("corridor_macro", {}).get(
                "goal_lock_predecessor_guard_triggered",
                False,
            )
        )
    ]
    candidate_counts = [
        int(row.get("corridor_macro", {}).get("goal_lock_predecessor_candidate_count", 0))
        for row in decision_rows
        if row.get("corridor_macro", {}).get("goal_lock_predecessor_candidate_count")
        is not None
    ]
    reason_counts: dict[str, int] = {}
    structural_progress_count = 0
    release_open_count = 0
    goal_lock_after_count = 0
    rejected_counts: dict[str, int] = {}
    for row in decision_rows:
        macro = row.get("corridor_macro", {})
        reason = macro.get("goal_lock_predecessor_guard_reason")
        if reason is not None:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        for key, value in (macro.get("goal_lock_predecessor_rejected_counts") or {}).items():
            rejected_counts[str(key)] = rejected_counts.get(str(key), 0) + int(value)
        if not bool(macro.get("goal_lock_predecessor_guard_triggered", False)):
            continue
        if bool(macro.get("goal_lock_predecessor_replacement_structural_progress", False)):
            structural_progress_count += 1
        progress = macro.get("goal_lock_predecessor_replacement_progress") or {}
        if (
            int(progress.get("release_action_count_delta", 0)) > 0
            or int(progress.get("selected_release_action_delta", 0)) > 0
        ):
            release_open_count += 1
        metrics = macro.get("goal_lock_predecessor_replacement_metrics") or {}
        if int(metrics.get("goal_locked_stand_blocker_count", 0)) > 0:
            goal_lock_after_count += 1
    result["summary"].update(
        {
            "goal_lock_predecessor_override_count": int(len(guard_rows)),
            "goal_lock_predecessor_structural_progress_count": int(
                structural_progress_count
            ),
            "goal_lock_predecessor_opened_release_count": int(release_open_count),
            "goal_lock_predecessor_replacement_goal_lock_after_count": int(
                goal_lock_after_count
            ),
            "goal_lock_predecessor_guard_reason_counts": reason_counts,
            "goal_lock_predecessor_rejected_counts": rejected_counts,
            "goal_lock_predecessor_candidate_count_min": (
                None if not candidate_counts else int(min(candidate_counts))
            ),
            "goal_lock_predecessor_candidate_count_max": (
                None if not candidate_counts else int(max(candidate_counts))
            ),
        }
    )
    return result


def run_a31_smoke(model, config: dict[str, Any], **kwargs) -> dict[str, Any]:
    original = _A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner
    _A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = (
        RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner
    )
    try:
        result = _A30._A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a31_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a28 = payload["a28_closed_loop_smoke"]["summary"]
    a29 = payload["a29_closed_loop_smoke"]["summary"]
    a30 = payload["a30_closed_loop_smoke"]["summary"]
    a31 = payload["a31_closed_loop_smoke"]["summary"]
    cmp_a30 = payload["comparison_a30_a31"]
    level_number = int(payload["config_extract"]["level_number"])
    override_count = int(a31.get("goal_lock_predecessor_override_count", 0))
    if override_count > 0:
        implication = (
            "A31 已把 A30 的 goal-lock 截断信号前移到 root 替代动作选择。"
            f"本轮触发替代 {override_count} 次；若目标数仍未恢复，说明当前同根候选"
            "只能避开坏宏，尚不能生成足够的 release 调度进展。"
        )
    else:
        implication = (
            "A31 未触发 root 替代动作。若 A30 仍存在 goal-lock 截断，则说明当前"
            "根候选集合中没有满足非目标回退与非 goal-lock 条件的替代动作；若 A30 "
            "截断也未再出现，则说明前缀轨迹已经偏离原 failure 入口。"
        )

    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A31 goal-lock predecessor 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A31 继承 A30，并将 A30 的 goal-lock 宏截断信号前移到 release "
            "阶段 root 候选替换。该版本仍不接入 teacher/reference 在线动作。"
        ),
        (
            f"A28 为 `{a28['status']}`、pushes={a28['executed_pushes']}、"
            f"decisions={a28['closed_loop_decision_steps']}、goals={a28['boxes_on_goals_final']}；"
            f"A29 为 `{a29['status']}`、pushes={a29['executed_pushes']}、"
            f"decisions={a29['closed_loop_decision_steps']}、goals={a29['boxes_on_goals_final']}；"
            f"A30 为 `{a30['status']}`、pushes={a30['executed_pushes']}、"
            f"decisions={a30['closed_loop_decision_steps']}、goals={a30['boxes_on_goals_final']}；"
            f"A31 为 `{a31['status']}`、pushes={a31['executed_pushes']}、"
            f"decisions={a31['closed_loop_decision_steps']}、goals={a31['boxes_on_goals_final']}。"
        ),
        (
            f"A30/A31 首个决策差异为 `{cmp_a30['a30_a31_first_decision_difference_step']}`；"
            f"A31 root 替代次数为 `{a31.get('goal_lock_predecessor_override_count', 0)}`，"
            f"替代后打开 release action 次数为 "
            f"`{a31.get('goal_lock_predecessor_opened_release_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | truncations | root replacements | replacement progress | opened release |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A28 | {a28['status']} | {a28['closed_loop_decision_steps']} | {a28['executed_pushes']} | "
            f"{a28['boxes_on_goals_final']} | {_fmt(a28['final_slot_schedule_cost'])} | "
            f"{a28['final_dependency_count']} | {a28['final_target_blocker_count']} | "
            f"{a28['final_release_action_count']} | NA | NA | NA | NA |"
        ),
        (
            f"| A29 | {a29['status']} | {a29['closed_loop_decision_steps']} | {a29['executed_pushes']} | "
            f"{a29['boxes_on_goals_final']} | {_fmt(a29['final_slot_schedule_cost'])} | "
            f"{a29['final_dependency_count']} | {a29['final_target_blocker_count']} | "
            f"{a29['final_release_action_count']} | NA | NA | NA | "
            f"{a29.get('release_action_macro_opened_release_count', 'NA')} |"
        ),
        (
            f"| A30 | {a30['status']} | {a30['closed_loop_decision_steps']} | {a30['executed_pushes']} | "
            f"{a30['boxes_on_goals_final']} | {_fmt(a30['final_slot_schedule_cost'])} | "
            f"{a30['final_dependency_count']} | {a30['final_target_blocker_count']} | "
            f"{a30['final_release_action_count']} | "
            f"{a30.get('controlled_release_goal_lock_truncation_count', 'NA')} | "
            f"NA | NA | {a30.get('release_action_macro_opened_release_count', 'NA')} |"
        ),
        (
            f"| A31 | {a31['status']} | {a31['closed_loop_decision_steps']} | {a31['executed_pushes']} | "
            f"{a31['boxes_on_goals_final']} | {_fmt(a31['final_slot_schedule_cost'])} | "
            f"{a31['final_dependency_count']} | {a31['final_target_blocker_count']} | "
            f"{a31['final_release_action_count']} | "
            f"{a31.get('controlled_release_goal_lock_truncation_count', 'NA')} | "
            f"{a31.get('goal_lock_predecessor_override_count', 'NA')} | "
            f"{a31.get('goal_lock_predecessor_structural_progress_count', 'NA')} | "
            f"{a31.get('goal_lock_predecessor_opened_release_count', 'NA')} |"
        ),
        "",
        "## A31 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dependencies | macro reason | root replacement | candidates | opened release |",
        "|---:|---|---:|---:|---|---:|---:|---|---|---:|---|",
    ]
    for row in payload["a31_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        macro = row.get("corridor_macro", {})
        progress = macro.get("goal_lock_predecessor_replacement_progress") or {}
        opened = bool(
            int(progress.get("release_action_count_delta", 0)) > 0
            or int(progress.get("selected_release_action_delta", 0)) > 0
        )
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{macro.get('reason')} | "
            f"{macro.get('goal_lock_predecessor_guard_triggered', False)} | "
            f"{macro.get('goal_lock_predecessor_candidate_count')} | "
            f"{opened} |"
        )
    lines.extend(["", "## 方法含义", "", implication, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure-log",
        default="results/sokoban_sd_world_model/xsokoban_first40_bestconfig_exact_l001_maxsteps1000_seed0.json",
    )
    parser.add_argument(
        "--a30-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json",
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
    parser.add_argument("--stage-release-goal-regression-allowed", action="store_true")
    parser.add_argument("--pre-release-corridor-guard-min-pushes", type=int, default=2)
    parser.add_argument("--pre-release-corridor-guard-legal-margin", type=int, default=2)
    parser.add_argument("--pre-release-root-trap-legal-margin", type=int, default=2)
    parser.add_argument("--pre-release-root-trap-locked-increase-penalty", type=float, default=110.0)
    parser.add_argument("--pre-release-root-trap-locked-absolute-penalty", type=float, default=30.0)
    parser.add_argument("--pre-release-root-trap-conversion-penalty", type=float, default=70.0)
    parser.add_argument("--pre-release-root-trap-occupied-increase-penalty", type=float, default=18.0)
    parser.add_argument("--pre-release-root-trap-low-margin-penalty", type=float, default=24.0)
    parser.add_argument("--pre-release-root-trap-no-release-penalty", type=float, default=10.0)
    parser.add_argument("--pre-release-root-trap-dependency-reduction-bonus", type=float, default=18.0)
    parser.add_argument("--pre-release-root-trap-target-reduction-bonus", type=float, default=14.0)
    parser.add_argument("--pre-release-root-trap-release-open-bonus", type=float, default=24.0)
    parser.add_argument("--pre-release-root-trap-legal-margin-bonus", type=float, default=4.0)
    parser.add_argument("--sd-generalization-legal-margin", type=int, default=2)
    parser.add_argument("--sd-generalization-predecessor-penalty", type=float, default=95.0)
    parser.add_argument("--sd-generalization-all-bad-successor-penalty", type=float, default=125.0)
    parser.add_argument("--sd-generalization-failure-similarity-penalty", type=float, default=70.0)
    parser.add_argument("--sd-generalization-success-similarity-bonus", type=float, default=20.0)
    parser.add_argument("--sd-generalization-dependency-reduction-bonus", type=float, default=18.0)
    parser.add_argument("--sd-generalization-release-open-bonus", type=float, default=24.0)
    parser.add_argument("--sd-generalization-legal-margin-bonus", type=float, default=4.0)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a30_path = ROOT / args.a30_json
    if not a30_path.exists():
        raise FileNotFoundError(f"A30 baseline JSON not found: {a30_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a30_payload = json.loads(a30_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[
        level_offset
    ]
    model = _A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
        level,
        config,
    )
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
    a31 = run_a31_smoke(
        model,
        config,
        seed_offset=51204,
        recovery_macro_max_extra_pushes=int(args.recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(args.recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(args.recovery_macro_score_threshold),
        dependency_bound_macro_score_threshold=float(
            args.dependency_bound_macro_score_threshold
        ),
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
        pre_release_corridor_guard_min_pushes=int(
            args.pre_release_corridor_guard_min_pushes
        ),
        pre_release_corridor_guard_legal_margin=int(
            args.pre_release_corridor_guard_legal_margin
        ),
        pre_release_root_trap_legal_margin=int(args.pre_release_root_trap_legal_margin),
        pre_release_root_trap_locked_increase_penalty=float(
            args.pre_release_root_trap_locked_increase_penalty
        ),
        pre_release_root_trap_locked_absolute_penalty=float(
            args.pre_release_root_trap_locked_absolute_penalty
        ),
        pre_release_root_trap_conversion_penalty=float(
            args.pre_release_root_trap_conversion_penalty
        ),
        pre_release_root_trap_occupied_increase_penalty=float(
            args.pre_release_root_trap_occupied_increase_penalty
        ),
        pre_release_root_trap_low_margin_penalty=float(
            args.pre_release_root_trap_low_margin_penalty
        ),
        pre_release_root_trap_no_release_penalty=float(
            args.pre_release_root_trap_no_release_penalty
        ),
        pre_release_root_trap_dependency_reduction_bonus=float(
            args.pre_release_root_trap_dependency_reduction_bonus
        ),
        pre_release_root_trap_target_reduction_bonus=float(
            args.pre_release_root_trap_target_reduction_bonus
        ),
        pre_release_root_trap_release_open_bonus=float(
            args.pre_release_root_trap_release_open_bonus
        ),
        pre_release_root_trap_legal_margin_bonus=float(
            args.pre_release_root_trap_legal_margin_bonus
        ),
        sd_generalization_legal_margin=int(args.sd_generalization_legal_margin),
        sd_generalization_predecessor_penalty=float(
            args.sd_generalization_predecessor_penalty
        ),
        sd_generalization_all_bad_successor_penalty=float(
            args.sd_generalization_all_bad_successor_penalty
        ),
        sd_generalization_failure_similarity_penalty=float(
            args.sd_generalization_failure_similarity_penalty
        ),
        sd_generalization_success_similarity_bonus=float(
            args.sd_generalization_success_similarity_bonus
        ),
        sd_generalization_dependency_reduction_bonus=float(
            args.sd_generalization_dependency_reduction_bonus
        ),
        sd_generalization_release_open_bonus=float(
            args.sd_generalization_release_open_bonus
        ),
        sd_generalization_legal_margin_bonus=float(
            args.sd_generalization_legal_margin_bonus
        ),
        **common,
    )
    a28 = a30_payload["a28_closed_loop_smoke"]
    a29 = a30_payload["a29_closed_loop_smoke"]
    a30 = a30_payload["a30_closed_loop_smoke"]
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a31_goal_lock_predecessor_release_v1",
            "failure_log": _rel(failure_path),
            "a30_json": _rel(a30_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a25_planner": RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a26_planner": RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner.provider_id,
            "a27_planner": RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner.provider_id,
            "a28_planner": RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner.provider_id,
            "a29_planner": RoomScheduleReleaseActionMacroGFTSDMCTSPlanner.provider_id,
            "a30_planner": RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner.provider_id,
            "a31_planner": RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a30_a31_goal_lock_predecessor_release_smoke",
        },
        "config_extract": {
            **a30_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "goal_lock_predecessor_release_root_replacer_v1",
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a28_closed_loop_smoke": a28,
        "a29_closed_loop_smoke": a29,
        "a30_closed_loop_smoke": a30,
        "a31_closed_loop_smoke": a31,
        "comparison_a30_a31": _A30.compare_runs("a30", a30, "a31", a31),
        "comparison_a29_a31": _A30.compare_runs("a29", a29, "a31", a31),
        "comparison_a28_a31": _A30.compare_runs("a28", a28, "a31", a31),
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
