#!/usr/bin/env python3
"""A32 learned SD-consumer diagnostics for XSokoban first levels.

A32 inherits A31 and consumes an offline-trained compact SD candidate scorer.
The checkpoint is used only as a root ranking/value consumer; Sokoban dynamics
and action generation remain exact and no teacher/reference action is queried
online.
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

_A31_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a31_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a31_goal_lock_predecessor_release.py",
)
if _A31_SPEC is None or _A31_SPEC.loader is None:
    raise RuntimeError("Unable to load A31 diagnostic helpers.")
_A31 = importlib.util.module_from_spec(_A31_SPEC)
_A31_SPEC.loader.exec_module(_A31)

from src.sokoban import (  # noqa: E402
    RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner,
    RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner,
    RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner,
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
    return _A31._fmt(value)


def augment_a32_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A31.augment_a31_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_sd_value", {}).get("learned_sd_consumer_checkpoint_loaded"))
    ]
    scores = [
        float(row.get("chosen_sd_value", {}).get("learned_sd_consumer_score"))
        for row in loaded_rows
        if row.get("chosen_sd_value", {}).get("learned_sd_consumer_score") is not None
    ]
    selected_positive = [score for score in scores if score > 0.0]
    result["summary"].update(
        {
            "learned_sd_consumer_loaded_count": int(len(loaded_rows)),
            "learned_sd_consumer_selected_positive_count": int(
                len(selected_positive)
            ),
            "learned_sd_consumer_score_min": None if not scores else float(min(scores)),
            "learned_sd_consumer_score_max": None if not scores else float(max(scores)),
        }
    )
    return result


def run_a32_smoke(
    model,
    config: dict[str, Any],
    *,
    learned_sd_consumer_checkpoint: str,
    learned_sd_consumer_root_override: bool = False,
    **kwargs,
) -> dict[str, Any]:
    original = _A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A32Planner(RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner):
        def __init__(self, *args, **planner_kwargs):
            super().__init__(
                *args,
                learned_sd_consumer_checkpoint=learned_sd_consumer_checkpoint,
                learned_sd_consumer_root_override=learned_sd_consumer_root_override,
                **planner_kwargs,
            )

    _A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A32Planner
    try:
        result = _A31._A30._A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a32_summary(result)


def compare_runs(left_name: str, left: dict[str, Any], right_name: str, right: dict[str, Any]) -> dict[str, Any]:
    return _A31._A30.compare_runs(left_name, left, right_name, right)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a31 = payload["a31_closed_loop_smoke"]["summary"]
    a32 = payload["a32_closed_loop_smoke"]["summary"]
    cmp_a31 = payload["comparison_a31_a32"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A32 learned SD-consumer 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A32 继承 A31，并加载离线 trace 蒸馏得到的 compact SD candidate scorer。"
            "默认配置仅将 learned score 作为 root prior/value 的软信号消费，"
            "不启用 learned scorer 对父类根选择的硬覆盖；该版本不接入 teacher/reference "
            "在线动作，也不替代 exact Sokoban dynamics。"
        ),
        (
            f"A31 为 `{a31['status']}`、pushes={a31['executed_pushes']}、"
            f"decisions={a31['closed_loop_decision_steps']}、goals={a31['boxes_on_goals_final']}、"
            f"release actions={a31['final_release_action_count']}；"
            f"A32 为 `{a32['status']}`、pushes={a32['executed_pushes']}、"
            f"decisions={a32['closed_loop_decision_steps']}、goals={a32['boxes_on_goals_final']}、"
            f"release actions={a32['final_release_action_count']}。"
        ),
        (
            f"A31/A32 首个决策差异为 `{cmp_a31['a31_a32_first_decision_difference_step']}`；"
            f"A32 learned scorer 加载步数为 `{a32.get('learned_sd_consumer_loaded_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | learned loaded | learned positive |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A31 | {a31['status']} | {a31['closed_loop_decision_steps']} | {a31['executed_pushes']} | "
            f"{a31['boxes_on_goals_final']} | {_fmt(a31['final_slot_schedule_cost'])} | "
            f"{a31['final_dependency_count']} | {a31['final_target_blocker_count']} | "
            f"{a31['final_release_action_count']} | NA | NA |"
        ),
        (
            f"| A32 | {a32['status']} | {a32['closed_loop_decision_steps']} | {a32['executed_pushes']} | "
            f"{a32['boxes_on_goals_final']} | {_fmt(a32['final_slot_schedule_cost'])} | "
            f"{a32['final_dependency_count']} | {a32['final_target_blocker_count']} | "
            f"{a32['final_release_action_count']} | "
            f"{a32.get('learned_sd_consumer_loaded_count', 'NA')} | "
            f"{a32.get('learned_sd_consumer_selected_positive_count', 'NA')} |"
        ),
        "",
        "## A32 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dependencies | release actions | learned score | macro reason |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["a32_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        sdv = row.get("chosen_sd_value", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{row['schedule_plan_after']['totals']['release_action_count']} | "
            f"{_fmt(sdv.get('learned_sd_consumer_score'))} | "
            f"{row.get('corridor_macro', {}).get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A32 的主要作用是验证 learned-consumer 接口链路：统一 SD 特征、离线 checkpoint、"
                "root ranking 消费和 trace 记录已经连通。该版本的 checkpoint 来自 A28-A31 "
                "trace 的启发式蒸馏，不能视为充分训练的神经 TVA。"
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
        "--a31-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--learned-sd-consumer-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_linear_checkpoint.json",
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
        "--learned-sd-consumer-root-override",
        action="store_true",
        help=(
            "Allow the learned SD scorer to override the inherited root choice. "
            "The default keeps it as a soft prior/value consumer only."
        ),
    )
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a32_learned_sd_consumer_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a31_path = ROOT / args.a31_json
    checkpoint_path = ROOT / args.learned_sd_consumer_checkpoint
    if not a31_path.exists():
        raise FileNotFoundError(f"A31 baseline JSON not found: {a31_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"A32 checkpoint not found: {checkpoint_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a31_payload = json.loads(a31_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    model = _A31._A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a32 = run_a32_smoke(
        model,
        config,
        learned_sd_consumer_checkpoint=str(checkpoint_path),
        learned_sd_consumer_root_override=bool(args.learned_sd_consumer_root_override),
        seed_offset=51205,
        recovery_macro_max_extra_pushes=int(args.recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(args.recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(args.recovery_macro_score_threshold),
        dependency_bound_macro_score_threshold=float(args.dependency_bound_macro_score_threshold),
        feasibility_guard_max_slot_cost_regression=float(args.feasibility_guard_max_slot_cost_regression),
        feasibility_guard_max_schedule_barrier_regression=float(args.feasibility_guard_max_schedule_barrier_regression),
        recoverability_probe_depth=int(args.recoverability_probe_depth),
        recoverability_probe_max_nodes=int(args.recoverability_probe_max_nodes),
        survivable_recoverability_min_depth=int(args.survivable_recoverability_min_depth),
        survivable_recoverability_min_legal_pushes=int(args.survivable_recoverability_min_legal_pushes),
        stage_release_min_goals=int(args.stage_release_min_goals),
        stage_stand_lock_score_scale=float(args.stage_stand_lock_score_scale),
        stage_release_stand_occupied_resolution_weight=float(args.stage_release_stand_occupied_resolution_weight),
        stage_release_goal_nonregression=not bool(args.stage_release_goal_regression_allowed),
        pre_release_corridor_guard_min_pushes=int(args.pre_release_corridor_guard_min_pushes),
        pre_release_corridor_guard_legal_margin=int(args.pre_release_corridor_guard_legal_margin),
        pre_release_root_trap_legal_margin=int(args.pre_release_root_trap_legal_margin),
        pre_release_root_trap_locked_increase_penalty=float(args.pre_release_root_trap_locked_increase_penalty),
        pre_release_root_trap_locked_absolute_penalty=float(args.pre_release_root_trap_locked_absolute_penalty),
        pre_release_root_trap_conversion_penalty=float(args.pre_release_root_trap_conversion_penalty),
        pre_release_root_trap_occupied_increase_penalty=float(args.pre_release_root_trap_occupied_increase_penalty),
        pre_release_root_trap_low_margin_penalty=float(args.pre_release_root_trap_low_margin_penalty),
        pre_release_root_trap_no_release_penalty=float(args.pre_release_root_trap_no_release_penalty),
        pre_release_root_trap_dependency_reduction_bonus=float(args.pre_release_root_trap_dependency_reduction_bonus),
        pre_release_root_trap_target_reduction_bonus=float(args.pre_release_root_trap_target_reduction_bonus),
        pre_release_root_trap_release_open_bonus=float(args.pre_release_root_trap_release_open_bonus),
        pre_release_root_trap_legal_margin_bonus=float(args.pre_release_root_trap_legal_margin_bonus),
        sd_generalization_legal_margin=int(args.sd_generalization_legal_margin),
        sd_generalization_predecessor_penalty=float(args.sd_generalization_predecessor_penalty),
        sd_generalization_all_bad_successor_penalty=float(args.sd_generalization_all_bad_successor_penalty),
        sd_generalization_failure_similarity_penalty=float(args.sd_generalization_failure_similarity_penalty),
        sd_generalization_success_similarity_bonus=float(args.sd_generalization_success_similarity_bonus),
        sd_generalization_dependency_reduction_bonus=float(args.sd_generalization_dependency_reduction_bonus),
        sd_generalization_release_open_bonus=float(args.sd_generalization_release_open_bonus),
        sd_generalization_legal_margin_bonus=float(args.sd_generalization_legal_margin_bonus),
        **common,
    )
    a31 = a31_payload["a31_closed_loop_smoke"]
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a32_learned_sd_consumer_v1",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "learned_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a25_planner": RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a26_planner": RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner.provider_id,
            "a27_planner": RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner.provider_id,
            "a28_planner": RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner.provider_id,
            "a29_planner": RoomScheduleReleaseActionMacroGFTSDMCTSPlanner.provider_id,
            "a30_planner": RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner.provider_id,
            "a31_planner": RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner.provider_id,
            "a32_planner": RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a31_a32_learned_sd_consumer_smoke",
        },
        "config_extract": {
            **a31_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "learned_sd_consumer_root_ranker_v1",
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a31_closed_loop_smoke": a31,
        "a32_closed_loop_smoke": a32,
        "comparison_a31_a32": compare_runs("a31", a31, "a32", a32),
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
