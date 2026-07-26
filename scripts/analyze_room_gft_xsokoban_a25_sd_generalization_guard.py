#!/usr/bin/env python3
"""A25 SD-generalization guard diagnostics for XSokoban first levels.

A25 inherits A24, keeps its pre-release trap score, and adds a prototype-style
SD-consumer generalization score.  It does not use teacher actions or reference
traces online.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A22_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a22_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a22_phased_release_macro.py",
)
if _A22_SPEC is None or _A22_SPEC.loader is None:
    raise RuntimeError("Unable to load A22 diagnostic helpers.")
_A22 = importlib.util.module_from_spec(_A22_SPEC)
_A22_SPEC.loader.exec_module(_A22)

from src.sokoban import (  # noqa: E402
    RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner,
    RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner,
    RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner,
    RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    return _A22._fmt(value)


def build_a25_planner(
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
    pre_release_corridor_guard_min_pushes: int = 2,
    pre_release_corridor_guard_legal_margin: int = 2,
    pre_release_root_trap_legal_margin: int = 2,
    pre_release_root_trap_locked_increase_penalty: float = 110.0,
    pre_release_root_trap_locked_absolute_penalty: float = 30.0,
    pre_release_root_trap_conversion_penalty: float = 70.0,
    pre_release_root_trap_occupied_increase_penalty: float = 18.0,
    pre_release_root_trap_low_margin_penalty: float = 24.0,
    pre_release_root_trap_no_release_penalty: float = 10.0,
    pre_release_root_trap_dependency_reduction_bonus: float = 18.0,
    pre_release_root_trap_target_reduction_bonus: float = 14.0,
    pre_release_root_trap_release_open_bonus: float = 24.0,
    pre_release_root_trap_legal_margin_bonus: float = 4.0,
    sd_generalization_legal_margin: int = 2,
    sd_generalization_predecessor_penalty: float = 95.0,
    sd_generalization_all_bad_successor_penalty: float = 125.0,
    sd_generalization_failure_similarity_penalty: float = 70.0,
    sd_generalization_success_similarity_bonus: float = 20.0,
    sd_generalization_dependency_reduction_bonus: float = 18.0,
    sd_generalization_release_open_bonus: float = 24.0,
    sd_generalization_legal_margin_bonus: float = 4.0,
) -> RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner:
    return RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner(
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
        pre_release_corridor_guard_min_pushes=int(pre_release_corridor_guard_min_pushes),
        pre_release_corridor_guard_legal_margin=int(
            pre_release_corridor_guard_legal_margin
        ),
        pre_release_root_trap_legal_margin=int(pre_release_root_trap_legal_margin),
        pre_release_root_trap_locked_increase_penalty=float(
            pre_release_root_trap_locked_increase_penalty
        ),
        pre_release_root_trap_locked_absolute_penalty=float(
            pre_release_root_trap_locked_absolute_penalty
        ),
        pre_release_root_trap_conversion_penalty=float(
            pre_release_root_trap_conversion_penalty
        ),
        pre_release_root_trap_occupied_increase_penalty=float(
            pre_release_root_trap_occupied_increase_penalty
        ),
        pre_release_root_trap_low_margin_penalty=float(
            pre_release_root_trap_low_margin_penalty
        ),
        pre_release_root_trap_no_release_penalty=float(
            pre_release_root_trap_no_release_penalty
        ),
        pre_release_root_trap_dependency_reduction_bonus=float(
            pre_release_root_trap_dependency_reduction_bonus
        ),
        pre_release_root_trap_target_reduction_bonus=float(
            pre_release_root_trap_target_reduction_bonus
        ),
        pre_release_root_trap_release_open_bonus=float(
            pre_release_root_trap_release_open_bonus
        ),
        pre_release_root_trap_legal_margin_bonus=float(
            pre_release_root_trap_legal_margin_bonus
        ),
        sd_generalization_legal_margin=int(sd_generalization_legal_margin),
        sd_generalization_predecessor_penalty=float(
            sd_generalization_predecessor_penalty
        ),
        sd_generalization_all_bad_successor_penalty=float(
            sd_generalization_all_bad_successor_penalty
        ),
        sd_generalization_failure_similarity_penalty=float(
            sd_generalization_failure_similarity_penalty
        ),
        sd_generalization_success_similarity_bonus=float(
            sd_generalization_success_similarity_bonus
        ),
        sd_generalization_dependency_reduction_bonus=float(
            sd_generalization_dependency_reduction_bonus
        ),
        sd_generalization_release_open_bonus=float(
            sd_generalization_release_open_bonus
        ),
        sd_generalization_legal_margin_bonus=float(
            sd_generalization_legal_margin_bonus
        ),
    )


def augment_a25_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A22.augment_phased_release_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    truncated = [
        row
        for row in decision_rows
        if row.get("corridor_macro", {}).get("reason")
        == "pre_release_bottleneck_truncated_corridor"
    ]
    truncated_rows = [
        item
        for row in truncated
        for item in row.get("corridor_macro", {}).get("rows", [])
        if item.get("pre_release_bottleneck_guard_triggered")
    ]
    chosen_trap_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_pre_release_trap", {}).get("trap_risk"))
    ]
    chosen_severe_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_pre_release_trap", {}).get("severe_trap_risk"))
    ]
    chosen_conversion_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_pre_release_trap", {}).get("precondition_to_lock_conversion"))
    ]
    chosen_generalized_failure_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_sd_generalization", {}).get("generalized_failure_risk"))
    ]
    chosen_trap_predecessor_rows = [
        row
        for row in decision_rows
        if bool(
            row.get("chosen_sd_generalization", {})
            .get("successor_probe", {})
            .get("trap_predecessor")
        )
    ]
    top_severe_count = 0
    inferred_guard_override_count = 0
    root_trap_scores: list[float] = []
    sd_generalization_scores: list[float] = []
    for row in decision_rows:
        chosen = row.get("chosen_pre_release_trap", {})
        if "score" in chosen:
            root_trap_scores.append(float(chosen["score"]))
        chosen_sdg = row.get("chosen_sd_generalization", {})
        if "score" in chosen_sdg:
            sd_generalization_scores.append(float(chosen_sdg["score"]))
        children = row.get("top_children", [])
        if not children:
            continue
        top = children[0]
        top_trap = top.get("room_pre_release_trap", {})
        if bool(top_trap.get("severe_trap_risk")):
            top_severe_count += 1
        chosen_key = tuple(row.get("semantic_action", {}).get("semantic_key", []))
        top_key = tuple(top.get("semantic_action", {}).get("semantic_key", []))
        if (
            bool(top_trap.get("severe_trap_risk"))
            and chosen_key != top_key
            and not bool(chosen.get("severe_trap_risk"))
        ):
            inferred_guard_override_count += 1
    result["summary"].update(
        {
            "pre_release_bottleneck_truncation_steps": int(len(truncated)),
            "pre_release_bottleneck_truncation_rows": int(len(truncated_rows)),
            "pre_release_bottleneck_truncated_pushes_saved": int(
                sum(
                    max(
                        0,
                        int(item.get("pre_release_bottleneck_original_macro_push_count", 0))
                        - int(row.get("corridor_macro", {}).get("macro_push_count", 0)),
                    )
                    for row in truncated
                    for item in row.get("corridor_macro", {}).get("rows", [])[-1:]
                )
            ),
            "sd_generalization_selected_count": int(len(chosen_trap_rows)),
            "sd_generalization_selected_severe_count": int(len(chosen_severe_rows)),
            "sd_generalization_selected_conversion_count": int(len(chosen_conversion_rows)),
            "sd_generalization_generalized_failure_selected_count": int(
                len(chosen_generalized_failure_rows)
            ),
            "sd_generalization_trap_predecessor_selected_count": int(
                len(chosen_trap_predecessor_rows)
            ),
            "sd_generalization_top_severe_count": int(top_severe_count),
            "sd_generalization_inferred_guard_override_count": int(
                inferred_guard_override_count
            ),
            "pre_release_root_trap_score_min": None
            if not root_trap_scores
            else float(min(root_trap_scores)),
            "pre_release_root_trap_score_max": None
            if not root_trap_scores
            else float(max(root_trap_scores)),
            "sd_generalization_score_min": None
            if not sd_generalization_scores
            else float(min(sd_generalization_scores)),
            "sd_generalization_score_max": None
            if not sd_generalization_scores
            else float(max(sd_generalization_scores)),
        }
    )
    return result


def run_a25_smoke(
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
    pre_release_corridor_guard_min_pushes: int,
    pre_release_corridor_guard_legal_margin: int,
    pre_release_root_trap_legal_margin: int,
    pre_release_root_trap_locked_increase_penalty: float,
    pre_release_root_trap_locked_absolute_penalty: float,
    pre_release_root_trap_conversion_penalty: float,
    pre_release_root_trap_occupied_increase_penalty: float,
    pre_release_root_trap_low_margin_penalty: float,
    pre_release_root_trap_no_release_penalty: float,
    pre_release_root_trap_dependency_reduction_bonus: float,
    pre_release_root_trap_target_reduction_bonus: float,
    pre_release_root_trap_release_open_bonus: float,
    pre_release_root_trap_legal_margin_bonus: float,
    sd_generalization_legal_margin: int,
    sd_generalization_predecessor_penalty: float,
    sd_generalization_all_bad_successor_penalty: float,
    sd_generalization_failure_similarity_penalty: float,
    sd_generalization_success_similarity_bonus: float,
    sd_generalization_dependency_reduction_bonus: float,
    sd_generalization_release_open_bonus: float,
    sd_generalization_legal_margin_bonus: float,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a25_planner(
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
            pre_release_corridor_guard_min_pushes=pre_release_corridor_guard_min_pushes,
            pre_release_corridor_guard_legal_margin=(
                pre_release_corridor_guard_legal_margin
            ),
            pre_release_root_trap_legal_margin=pre_release_root_trap_legal_margin,
            pre_release_root_trap_locked_increase_penalty=(
                pre_release_root_trap_locked_increase_penalty
            ),
            pre_release_root_trap_locked_absolute_penalty=(
                pre_release_root_trap_locked_absolute_penalty
            ),
            pre_release_root_trap_conversion_penalty=(
                pre_release_root_trap_conversion_penalty
            ),
            pre_release_root_trap_occupied_increase_penalty=(
                pre_release_root_trap_occupied_increase_penalty
            ),
            pre_release_root_trap_low_margin_penalty=(
                pre_release_root_trap_low_margin_penalty
            ),
            pre_release_root_trap_no_release_penalty=(
                pre_release_root_trap_no_release_penalty
            ),
            pre_release_root_trap_dependency_reduction_bonus=(
                pre_release_root_trap_dependency_reduction_bonus
            ),
            pre_release_root_trap_target_reduction_bonus=(
                pre_release_root_trap_target_reduction_bonus
            ),
            pre_release_root_trap_release_open_bonus=(
                pre_release_root_trap_release_open_bonus
            ),
            pre_release_root_trap_legal_margin_bonus=(
                pre_release_root_trap_legal_margin_bonus
            ),
            sd_generalization_legal_margin=sd_generalization_legal_margin,
            sd_generalization_predecessor_penalty=(
                sd_generalization_predecessor_penalty
            ),
            sd_generalization_all_bad_successor_penalty=(
                sd_generalization_all_bad_successor_penalty
            ),
            sd_generalization_failure_similarity_penalty=(
                sd_generalization_failure_similarity_penalty
            ),
            sd_generalization_success_similarity_bonus=(
                sd_generalization_success_similarity_bonus
            ),
            sd_generalization_dependency_reduction_bonus=(
                sd_generalization_dependency_reduction_bonus
            ),
            sd_generalization_release_open_bonus=(
                sd_generalization_release_open_bonus
            ),
            sd_generalization_legal_margin_bonus=(
                sd_generalization_legal_margin_bonus
            ),
            **builder_kwargs,
        )

    original = _A22._A21._A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner
    _A22._A21._A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner = builder
    try:
        result = _A22._A21._A20._A19._A18._A17._A16._A15._A14._A13.run_a13_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A22._A21._A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner = original
    return augment_a25_summary(result)


def compare_a22_a25(a22: dict[str, Any], a25: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    left = actions(a22)
    right = actions(a25)
    first_diff = None
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            first_diff = idx
            break
    if first_diff is None and len(left) != len(right):
        first_diff = min(len(left), len(right))
    return {
        "a22_a25_first_decision_difference_step": first_diff,
        "a22_status": a22["summary"]["status"],
        "a25_status": a25["summary"]["status"],
        "a22_boxes_on_goals_final": a22["summary"]["boxes_on_goals_final"],
        "a25_boxes_on_goals_final": a25["summary"]["boxes_on_goals_final"],
        "a22_executed_pushes": a22["summary"]["executed_pushes"],
        "a25_executed_pushes": a25["summary"]["executed_pushes"],
        "a22_final_slot_schedule_cost": a22["summary"]["final_slot_schedule_cost"],
        "a25_final_slot_schedule_cost": a25["summary"]["final_slot_schedule_cost"],
        "a22_prefix_actions": left[:60],
        "a25_prefix_actions": right[:60],
    }


def compare_a23_a25(a23: dict[str, Any], a25: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    left = actions(a23)
    right = actions(a25)
    first_diff = None
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            first_diff = idx
            break
    if first_diff is None and len(left) != len(right):
        first_diff = min(len(left), len(right))
    return {
        "a23_a25_first_decision_difference_step": first_diff,
        "a23_status": a23["summary"]["status"],
        "a25_status": a25["summary"]["status"],
        "a23_boxes_on_goals_final": a23["summary"]["boxes_on_goals_final"],
        "a25_boxes_on_goals_final": a25["summary"]["boxes_on_goals_final"],
        "a23_executed_pushes": a23["summary"]["executed_pushes"],
        "a25_executed_pushes": a25["summary"]["executed_pushes"],
        "a23_final_slot_schedule_cost": a23["summary"]["final_slot_schedule_cost"],
        "a25_final_slot_schedule_cost": a25["summary"]["final_slot_schedule_cost"],
        "a23_prefix_actions": left[:60],
        "a25_prefix_actions": right[:60],
    }


def compare_a24_a25(a24: dict[str, Any], a25: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    left = actions(a24)
    right = actions(a25)
    first_diff = None
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            first_diff = idx
            break
    if first_diff is None and len(left) != len(right):
        first_diff = min(len(left), len(right))
    return {
        "a24_a25_first_decision_difference_step": first_diff,
        "a24_status": a24["summary"]["status"],
        "a25_status": a25["summary"]["status"],
        "a24_boxes_on_goals_final": a24["summary"]["boxes_on_goals_final"],
        "a25_boxes_on_goals_final": a25["summary"]["boxes_on_goals_final"],
        "a24_executed_pushes": a24["summary"]["executed_pushes"],
        "a25_executed_pushes": a25["summary"]["executed_pushes"],
        "a24_final_slot_schedule_cost": a24["summary"]["final_slot_schedule_cost"],
        "a25_final_slot_schedule_cost": a25["summary"]["final_slot_schedule_cost"],
        "a24_prefix_actions": left[:60],
        "a25_prefix_actions": right[:60],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a22 = payload["a22_closed_loop_smoke"]["summary"]
    a23 = payload["a23_closed_loop_smoke"]["summary"]
    a24 = payload["a24_closed_loop_smoke"]["summary"]
    a25 = payload["a25_closed_loop_smoke"]["summary"]
    cmp_a24 = payload["comparison_a24_a25"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A25 SD generalization guard 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A25 继承 A24，保留 pre-release trap 风险评分，并新增原型式"
            " SD-consumer 泛化评分；该版本不接入 teacher/reference 在线动作。"
        ),
        (
            f"A22 为 `{a22['status']}`、pushes={a22['executed_pushes']}、"
            f"decisions={a22['closed_loop_decision_steps']}、goals={a22['boxes_on_goals_final']}；"
            f"A23 为 `{a23['status']}`、pushes={a23['executed_pushes']}、"
            f"decisions={a23['closed_loop_decision_steps']}、goals={a23['boxes_on_goals_final']}；"
            f"A24 为 `{a24['status']}`、pushes={a24['executed_pushes']}、"
            f"decisions={a24['closed_loop_decision_steps']}、goals={a24['boxes_on_goals_final']}；"
            f"A25 为 `{a25['status']}`、pushes={a25['executed_pushes']}、"
            f"decisions={a25['closed_loop_decision_steps']}、goals={a25['boxes_on_goals_final']}。"
        ),
        (
            f"A24/A25 首个决策差异为 `{cmp_a24['a24_a25_first_decision_difference_step']}`；"
            f"A25 generalized failure 选择步数为 `{a25['sd_generalization_generalized_failure_selected_count']}`，"
            f"trap-predecessor 选择步数为 `{a25['sd_generalization_trap_predecessor_selected_count']}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | bottleneck truncations | severe trap | generalized failure | trap predecessor |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A22 | {a22['status']} | {a22['closed_loop_decision_steps']} | {a22['executed_pushes']} | "
            f"{a22['boxes_on_goals_final']} | {_fmt(a22['final_slot_schedule_cost'])} | "
            f"{a22['final_dependency_count']} | {a22['final_target_blocker_count']} | "
            f"{a22['final_release_action_count']} | NA | NA | NA | NA |"
        ),
        (
            f"| A23 | {a23['status']} | {a23['closed_loop_decision_steps']} | {a23['executed_pushes']} | "
            f"{a23['boxes_on_goals_final']} | {_fmt(a23['final_slot_schedule_cost'])} | "
            f"{a23['final_dependency_count']} | {a23['final_target_blocker_count']} | "
            f"{a23['final_release_action_count']} | "
            f"{a23.get('pre_release_bottleneck_truncation_steps', 'NA')} | NA | NA | NA |"
        ),
        (
            f"| A24 | {a24['status']} | {a24['closed_loop_decision_steps']} | {a24['executed_pushes']} | "
            f"{a24['boxes_on_goals_final']} | {_fmt(a24['final_slot_schedule_cost'])} | "
            f"{a24['final_dependency_count']} | {a24['final_target_blocker_count']} | "
            f"{a24['final_release_action_count']} | "
            f"{a24.get('pre_release_bottleneck_truncation_steps', 'NA')} | "
            f"{a24.get('pre_release_root_trap_selected_severe_count', 'NA')} | NA | NA |"
        ),
        (
            f"| A25 | {a25['status']} | {a25['closed_loop_decision_steps']} | {a25['executed_pushes']} | "
            f"{a25['boxes_on_goals_final']} | {_fmt(a25['final_slot_schedule_cost'])} | "
            f"{a25['final_dependency_count']} | {a25['final_target_blocker_count']} | "
            f"{a25['final_release_action_count']} | "
            f"{a25['pre_release_bottleneck_truncation_steps']} | "
            f"{a25['sd_generalization_selected_severe_count']} | "
            f"{a25['sd_generalization_generalized_failure_selected_count']} | "
            f"{a25['sd_generalization_trap_predecessor_selected_count']} |"
        ),
        "",
        "## A25 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dependencies | trap score | gen score | gen fail | predecessor | macro reason |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["a25_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        stage = row.get("chosen_stage", {}).get("before", {}).get("stage", "NA")
        trap = row.get("chosen_pre_release_trap", {})
        sdg = row.get("chosen_sd_generalization", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{stage} | {_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{_fmt(trap.get('score'))} | {_fmt(sdg.get('score'))} | "
            f"{sdg.get('generalized_failure_risk', False)} | "
            f"{sdg.get('successor_probe', {}).get('trap_predecessor', False)} | "
            f"{row.get('corridor_macro', {}).get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A25 检验的是：能否把 A24 已知失败形态抽象为 SD 结构原型，并将其泛化到"
                "未见根候选上。该版本是后续可学习 TVA/value consumer 的接口雏形，而不是"
                "最终神经网络实现。"
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
        "--a22-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--a23-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--a24-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.json",
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a22_path = ROOT / args.a22_json
    a23_path = ROOT / args.a23_json
    a24_path = ROOT / args.a24_json
    if not a22_path.exists():
        raise FileNotFoundError(f"A22 baseline JSON not found: {a22_path}")
    if not a23_path.exists():
        raise FileNotFoundError(f"A23 baseline JSON not found: {a23_path}")
    if not a24_path.exists():
        raise FileNotFoundError(f"A24 baseline JSON not found: {a24_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a22_payload = json.loads(a22_path.read_text(encoding="utf-8"))
    a23_payload = json.loads(a23_path.read_text(encoding="utf-8"))
    a24_payload = json.loads(a24_path.read_text(encoding="utf-8"))
    a22 = a22_payload["a22_closed_loop_smoke"]
    a23 = a23_payload["a23_closed_loop_smoke"]
    a24 = a24_payload["a24_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    model = _A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a25 = run_a25_smoke(
        model,
        config,
        seed_offset=47293,
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
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a25_sd_generalization_guard_v1",
            "failure_log": _rel(failure_path),
            "a22_json": _rel(a22_path),
            "a23_json": _rel(a23_path),
            "a24_json": _rel(a24_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a22_planner": RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner.provider_id,
            "a23_planner": RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a24_planner": RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a25_planner": RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a22_a23_a25_sd_generalization_guard_smoke",
        },
        "a22_config_extract": a22_payload.get("config_extract", {}),
        "a23_config_extract": a23_payload.get("config_extract", {}),
        "a24_config_extract": a24_payload.get("config_extract", {}),
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
            "pre_release_corridor_guard_min_pushes": int(
                args.pre_release_corridor_guard_min_pushes
            ),
            "pre_release_corridor_guard_legal_margin": int(
                args.pre_release_corridor_guard_legal_margin
            ),
            "pre_release_root_trap_legal_margin": int(args.pre_release_root_trap_legal_margin),
            "pre_release_root_trap_locked_increase_penalty": float(
                args.pre_release_root_trap_locked_increase_penalty
            ),
            "pre_release_root_trap_locked_absolute_penalty": float(
                args.pre_release_root_trap_locked_absolute_penalty
            ),
            "pre_release_root_trap_conversion_penalty": float(
                args.pre_release_root_trap_conversion_penalty
            ),
            "pre_release_root_trap_occupied_increase_penalty": float(
                args.pre_release_root_trap_occupied_increase_penalty
            ),
            "pre_release_root_trap_low_margin_penalty": float(
                args.pre_release_root_trap_low_margin_penalty
            ),
            "pre_release_root_trap_no_release_penalty": float(
                args.pre_release_root_trap_no_release_penalty
            ),
            "pre_release_root_trap_dependency_reduction_bonus": float(
                args.pre_release_root_trap_dependency_reduction_bonus
            ),
            "pre_release_root_trap_target_reduction_bonus": float(
                args.pre_release_root_trap_target_reduction_bonus
            ),
            "pre_release_root_trap_release_open_bonus": float(
                args.pre_release_root_trap_release_open_bonus
            ),
            "pre_release_root_trap_legal_margin_bonus": float(
                args.pre_release_root_trap_legal_margin_bonus
            ),
            "sd_generalization_legal_margin": int(args.sd_generalization_legal_margin),
            "sd_generalization_predecessor_penalty": float(
                args.sd_generalization_predecessor_penalty
            ),
            "sd_generalization_all_bad_successor_penalty": float(
                args.sd_generalization_all_bad_successor_penalty
            ),
            "sd_generalization_failure_similarity_penalty": float(
                args.sd_generalization_failure_similarity_penalty
            ),
            "sd_generalization_success_similarity_bonus": float(
                args.sd_generalization_success_similarity_bonus
            ),
            "sd_generalization_dependency_reduction_bonus": float(
                args.sd_generalization_dependency_reduction_bonus
            ),
            "sd_generalization_release_open_bonus": float(
                args.sd_generalization_release_open_bonus
            ),
            "sd_generalization_legal_margin_bonus": float(
                args.sd_generalization_legal_margin_bonus
            ),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
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
        "comparison_a22_a25": compare_a22_a25(a22, a25),
        "comparison_a23_a25": compare_a23_a25(a23, a25),
        "comparison_a24_a25": compare_a24_a25(a24, a25),
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
