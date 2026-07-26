#!/usr/bin/env python3
"""A27 SD-value loop-calibrated diagnostics for XSokoban first levels.

A27 inherits A26 and calibrates repeated pseudo-milestones.  It keeps the
SD value/ranking interface but does not use teacher actions or reference traces
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

_A25_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a25_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a25_sd_generalization_guard.py",
)
if _A25_SPEC is None or _A25_SPEC.loader is None:
    raise RuntimeError("Unable to load A25 diagnostic helpers.")
_A25 = importlib.util.module_from_spec(_A25_SPEC)
_A25_SPEC.loader.exec_module(_A25)

from src.sokoban import (  # noqa: E402
    RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner,
    RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner,
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
    return _A25._fmt(value)


def _actions(payload: dict[str, Any]) -> list[str]:
    return [
        f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
        for row in payload["trace"]
        if "physical_action" in row
    ]


def compare_runs(
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
) -> dict[str, Any]:
    left_actions = _actions(left)
    right_actions = _actions(right)
    first_diff = None
    for idx, (a, b) in enumerate(zip(left_actions, right_actions)):
        if a != b:
            first_diff = idx
            break
    if first_diff is None and len(left_actions) != len(right_actions):
        first_diff = min(len(left_actions), len(right_actions))
    return {
        f"{left_name}_{right_name}_first_decision_difference_step": first_diff,
        f"{left_name}_status": left["summary"]["status"],
        f"{right_name}_status": right["summary"]["status"],
        f"{left_name}_boxes_on_goals_final": left["summary"]["boxes_on_goals_final"],
        f"{right_name}_boxes_on_goals_final": right["summary"]["boxes_on_goals_final"],
        f"{left_name}_executed_pushes": left["summary"]["executed_pushes"],
        f"{right_name}_executed_pushes": right["summary"]["executed_pushes"],
        f"{left_name}_final_slot_schedule_cost": left["summary"]["final_slot_schedule_cost"],
        f"{right_name}_final_slot_schedule_cost": right["summary"]["final_slot_schedule_cost"],
        f"{left_name}_prefix_actions": left_actions[:60],
        f"{right_name}_prefix_actions": right_actions[:60],
    }


def augment_a27_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A25.augment_a25_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    milestone_rows = [
        row for row in decision_rows if bool(row.get("chosen_sd_value", {}).get("milestone_progress"))
    ]
    loop_risk_rows = [
        row for row in decision_rows if bool(row.get("chosen_sd_value", {}).get("no_progress_loop_risk"))
    ]
    hard_negative_rows = [
        row for row in decision_rows if bool(row.get("chosen_sd_value", {}).get("hard_negative"))
    ]
    soft_negative_rows = [
        row for row in decision_rows if bool(row.get("chosen_sd_value", {}).get("soft_negative"))
    ]
    pseudo_milestone_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_sd_value", {}).get("pseudo_milestone_loop"))
    ]
    scores: list[float] = []
    top_loop_count = 0
    top_hard_negative_count = 0
    top_pseudo_milestone_count = 0
    inferred_override_count = 0
    for row in decision_rows:
        chosen = row.get("chosen_sd_value", {})
        if "score" in chosen:
            scores.append(float(chosen["score"]))
        children = row.get("top_children", [])
        if not children:
            continue
        top = children[0].get("room_sd_value", {})
        if bool(top.get("no_progress_loop_risk")):
            top_loop_count += 1
        if bool(top.get("hard_negative")):
            top_hard_negative_count += 1
        if bool(top.get("pseudo_milestone_loop")):
            top_pseudo_milestone_count += 1
        chosen_key = tuple(row.get("semantic_action", {}).get("semantic_key", []))
        top_key = tuple(children[0].get("semantic_action", {}).get("semantic_key", []))
        if (
            (bool(top.get("no_progress_loop_risk")) or bool(top.get("hard_negative")))
            and chosen_key != top_key
            and not bool(chosen.get("no_progress_loop_risk"))
            and not bool(chosen.get("hard_negative"))
        ):
            inferred_override_count += 1
    result["summary"].update(
        {
            "sd_value_milestone_selected_count": int(len(milestone_rows)),
            "sd_value_no_progress_loop_selected_count": int(len(loop_risk_rows)),
            "sd_value_hard_negative_selected_count": int(len(hard_negative_rows)),
            "sd_value_soft_negative_selected_count": int(len(soft_negative_rows)),
            "sd_value_pseudo_milestone_loop_selected_count": int(len(pseudo_milestone_rows)),
            "sd_value_top_loop_risk_count": int(top_loop_count),
            "sd_value_top_hard_negative_count": int(top_hard_negative_count),
            "sd_value_top_pseudo_milestone_count": int(top_pseudo_milestone_count),
            "sd_value_inferred_guard_override_count": int(inferred_override_count),
            "sd_value_score_min": None if not scores else float(min(scores)),
            "sd_value_score_max": None if not scores else float(max(scores)),
        }
    )
    return result


def run_a27_smoke(model, config: dict[str, Any], **kwargs) -> dict[str, Any]:
    original = _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner
    _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = (
        RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner
    )
    try:
        result = _A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a27_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a24 = payload["a24_closed_loop_smoke"]["summary"]
    a25 = payload["a25_closed_loop_smoke"]["summary"]
    a26 = payload["a26_closed_loop_smoke"]["summary"]
    a27 = payload["a27_closed_loop_smoke"]["summary"]
    cmp_a26 = payload["comparison_a26_a27"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A27 SD value loop-calibrated 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A27 继承 A26，并校准重复伪里程碑。该版本保留 root-only "
            "SD value/ranking consumer 接口，但仍不接入 teacher/reference 在线动作。"
        ),
        (
            f"A24 为 `{a24['status']}`、pushes={a24['executed_pushes']}、"
            f"decisions={a24['closed_loop_decision_steps']}、goals={a24['boxes_on_goals_final']}；"
            f"A25 为 `{a25['status']}`、pushes={a25['executed_pushes']}、"
            f"decisions={a25['closed_loop_decision_steps']}、goals={a25['boxes_on_goals_final']}；"
            f"A26 为 `{a26['status']}`、pushes={a26['executed_pushes']}、"
            f"decisions={a26['closed_loop_decision_steps']}、goals={a26['boxes_on_goals_final']}；"
            f"A27 为 `{a27['status']}`、pushes={a27['executed_pushes']}、"
            f"decisions={a27['closed_loop_decision_steps']}、goals={a27['boxes_on_goals_final']}。"
        ),
        (
            f"A26/A27 首个决策差异为 `{cmp_a26['a26_a27_first_decision_difference_step']}`；"
            f"A27 milestone 选择步数为 `{a27['sd_value_milestone_selected_count']}`，"
            f"pseudo milestone loop 选择步数为 `{a27.get('sd_value_pseudo_milestone_loop_selected_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | severe trap | generalized failure | value milestone | value loop | pseudo milestone |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A24 | {a24['status']} | {a24['closed_loop_decision_steps']} | {a24['executed_pushes']} | "
            f"{a24['boxes_on_goals_final']} | {_fmt(a24['final_slot_schedule_cost'])} | "
            f"{a24['final_dependency_count']} | {a24['final_target_blocker_count']} | "
            f"{a24['final_release_action_count']} | "
            f"{a24.get('pre_release_root_trap_selected_severe_count', 'NA')} | NA | NA | NA | NA |"
        ),
        (
            f"| A25 | {a25['status']} | {a25['closed_loop_decision_steps']} | {a25['executed_pushes']} | "
            f"{a25['boxes_on_goals_final']} | {_fmt(a25['final_slot_schedule_cost'])} | "
            f"{a25['final_dependency_count']} | {a25['final_target_blocker_count']} | "
            f"{a25['final_release_action_count']} | "
            f"{a25.get('sd_generalization_selected_severe_count', 'NA')} | "
            f"{a25.get('sd_generalization_generalized_failure_selected_count', 'NA')} | NA | NA | NA |"
        ),
        (
            f"| A26 | {a26['status']} | {a26['closed_loop_decision_steps']} | {a26['executed_pushes']} | "
            f"{a26['boxes_on_goals_final']} | {_fmt(a26['final_slot_schedule_cost'])} | "
            f"{a26['final_dependency_count']} | {a26['final_target_blocker_count']} | "
            f"{a26['final_release_action_count']} | "
            f"{a26.get('sd_generalization_selected_severe_count', 'NA')} | "
            f"{a26.get('sd_generalization_generalized_failure_selected_count', 'NA')} | "
            f"{a26['sd_value_milestone_selected_count']} | "
            f"{a26['sd_value_no_progress_loop_selected_count']} | "
            f"{a26.get('sd_value_pseudo_milestone_loop_selected_count', 'NA')} |"
        ),
        (
            f"| A27 | {a27['status']} | {a27['closed_loop_decision_steps']} | {a27['executed_pushes']} | "
            f"{a27['boxes_on_goals_final']} | {_fmt(a27['final_slot_schedule_cost'])} | "
            f"{a27['final_dependency_count']} | {a27['final_target_blocker_count']} | "
            f"{a27['final_release_action_count']} | "
            f"{a27.get('sd_generalization_selected_severe_count', 'NA')} | "
            f"{a27.get('sd_generalization_generalized_failure_selected_count', 'NA')} | "
            f"{a27['sd_value_milestone_selected_count']} | "
            f"{a27['sd_value_no_progress_loop_selected_count']} | "
            f"{a27.get('sd_value_pseudo_milestone_loop_selected_count', 'NA')} |"
        ),
        "",
        "## A27 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dependencies | sd value | milestone | loop risk | pseudo | hard neg | macro reason |",
        "|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in payload["a27_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        stage = row.get("chosen_stage", {}).get("before", {}).get("stage", "NA")
        sdv = row.get("chosen_sd_value", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{stage} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{_fmt(sdv.get('score'))} | {sdv.get('milestone_progress', False)} | "
            f"{sdv.get('no_progress_loop_risk', False)} | "
            f"{sdv.get('pseudo_milestone_loop', False)} | "
            f"{sdv.get('hard_negative', False)} | "
            f"{row.get('corridor_macro', {}).get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A27 检验的是：A26 的 value/ranking consumer 是否需要显式区分真实"
                "里程碑与重复伪里程碑。当前校准仍是线性规则，但其接口对应后续"
                " learned consumer 的负例标签。"
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
        "--a26-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a26_sd_value_ranker_l002_s80_b128_n8.json",
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a26_path = ROOT / args.a26_json
    if not a26_path.exists():
        raise FileNotFoundError(f"A26 baseline JSON not found: {a26_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a26_payload = json.loads(a26_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    model = _A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a27 = run_a27_smoke(
        model,
        config,
        seed_offset=49937,
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
    a22 = a26_payload["a22_closed_loop_smoke"]
    a23 = a26_payload["a23_closed_loop_smoke"]
    a24 = a26_payload["a24_closed_loop_smoke"]
    a25 = a26_payload["a25_closed_loop_smoke"]
    a26 = a26_payload["a26_closed_loop_smoke"]
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a27_sd_value_loop_calibrated_v1",
            "failure_log": _rel(failure_path),
            "a26_json": _rel(a26_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a25_planner": RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a26_planner": RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner.provider_id,
            "a27_planner": RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a26_a27_sd_value_loop_calibrated_smoke",
        },
        "a22_config_extract": a26_payload.get("a22_config_extract", {}),
        "a23_config_extract": a26_payload.get("a23_config_extract", {}),
        "a24_config_extract": a26_payload.get("a24_config_extract", {}),
        "a25_config_extract": a26_payload.get("config_extract", {}),
        "config_extract": {
            **a26_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "loop_calibrated_linear_root_ranker_v1",
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a22_closed_loop_smoke": a22,
        "a23_closed_loop_smoke": a23,
        "a24_closed_loop_smoke": a24,
        "a25_closed_loop_smoke": a25,
        "a26_closed_loop_smoke": a26,
        "a27_closed_loop_smoke": a27,
        "comparison_a26_a27": compare_runs("a26", a26, "a27", a27),
        "comparison_a25_a27": compare_runs("a25", a25, "a27", a27),
        "comparison_a24_a27": compare_runs("a24", a24, "a27", a27),
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
