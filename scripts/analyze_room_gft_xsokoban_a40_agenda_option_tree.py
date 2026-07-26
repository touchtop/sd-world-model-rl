#!/usr/bin/env python3
"""A40 bounded agenda-option tree diagnostics for XSokoban first levels."""

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

_A39_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a39_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a39_agenda_tier_choice.py",
)
if _A39_SPEC is None or _A39_SPEC.loader is None:
    raise RuntimeError("Unable to load A39 diagnostic helpers.")
_A39 = importlib.util.module_from_spec(_A39_SPEC)
_A39_SPEC.loader.exec_module(_A39)

from src.sokoban import (  # noqa: E402
    RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A39._rel(path)


def _fmt(value: Any) -> str:
    return _A39._fmt(value)


def _option_tree(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("agenda_option_tree", {}) or {}


def _option_progress(row: dict[str, Any], key: str) -> dict[str, Any]:
    option = row.get(key, {}) if isinstance(row.get(key), dict) else {}
    progress = option.get("best_progress", {}) if isinstance(option, dict) else {}
    return progress if isinstance(progress, dict) else {}


def augment_a40_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A39.augment_a39_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_option_tree(row) for row in decision_rows if _option_tree(row)]
    reason_counts: dict[str, int] = {}
    parent_option_tier_counts: dict[str, int] = {}
    selected_option_tier_counts: dict[str, int] = {}
    selected_agenda_type_counts: dict[str, int] = {}
    total_evaluated = 0
    max_depth = 0
    max_expanded = 0
    for row in loaded:
        reason = str(row.get("agenda_option_tree_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        parent_tier = str(row.get("parent_option_tier", "NA"))
        selected_tier = str(row.get("selected_option_tier", "NA"))
        parent_option_tier_counts[parent_tier] = (
            parent_option_tier_counts.get(parent_tier, 0) + 1
        )
        selected_option_tier_counts[selected_tier] = (
            selected_option_tier_counts.get(selected_tier, 0) + 1
        )
        selected_progress = _option_progress(row, "selected_option")
        agenda_type = str(selected_progress.get("agenda_type", "NA"))
        selected_agenda_type_counts[agenda_type] = (
            selected_agenda_type_counts.get(agenda_type, 0) + 1
        )
        total_evaluated += int(row.get("evaluated_option_count", 0))
        selected_option = row.get("selected_option", {}) or {}
        max_depth = max(max_depth, int(selected_option.get("best_depth", 0) or 0))
        max_expanded = max(
            max_expanded,
            int(selected_option.get("expanded_nodes", 0) or 0),
        )
    result["summary"].update(
        {
            "agenda_option_tree_loaded_count": int(len(loaded)),
            "agenda_option_tree_override_count": int(
                sum(bool(row.get("agenda_option_tree_override")) for row in loaded)
            ),
            "agenda_option_tree_higher_option_tier_selected_count": int(
                sum(
                    int(row.get("selected_option_tier", 0) or 0)
                    > int(row.get("parent_option_tier", 0) or 0)
                    for row in loaded
                )
            ),
            "agenda_option_tree_selected_done_count": int(
                sum(
                    bool(
                        _option_progress(row, "selected_option").get(
                            "agenda_done",
                            False,
                        )
                    )
                    for row in loaded
                )
            ),
            "agenda_option_tree_selected_progress_count": int(
                sum(
                    bool(
                        _option_progress(row, "selected_option").get(
                            "agenda_progress",
                            False,
                        )
                    )
                    for row in loaded
                )
            ),
            "agenda_option_tree_selected_loop_count": int(
                sum(
                    bool(
                        _option_progress(row, "selected_option").get(
                            "agenda_loop",
                            False,
                        )
                    )
                    for row in loaded
                )
            ),
            "agenda_option_tree_reason_counts": reason_counts,
            "agenda_option_tree_parent_option_tier_counts": parent_option_tier_counts,
            "agenda_option_tree_selected_option_tier_counts": selected_option_tier_counts,
            "agenda_option_tree_selected_agenda_type_counts": selected_agenda_type_counts,
            "agenda_option_tree_total_evaluated_option_count": int(total_evaluated),
            "agenda_option_tree_max_selected_depth": int(max_depth),
            "agenda_option_tree_max_selected_expanded_nodes": int(max_expanded),
        }
    )
    return result


def run_a40_smoke(
    model,
    config: dict[str, Any],
    *,
    multihead_sd_consumer_checkpoint: str,
    learned_sd_consumer_weight: float = 0.35,
    learned_sd_consumer_root_override: bool = False,
    neural_sd_consumer_score_clip: float = 3.0,
    multihead_sd_consumer_value_weight: float = 0.45,
    multihead_sd_consumer_release_weight: float = 1.15,
    multihead_sd_consumer_loop_weight: float = 1.05,
    multihead_sd_consumer_trap_weight: float = 1.25,
    agenda_option_tree_max_pushes: int = 5,
    agenda_option_tree_max_nodes: int = 96,
    agenda_option_tree_max_root_candidates: int = 10,
    agenda_option_tree_min_tier: int = 5,
    agenda_option_tree_min_tier_improvement: int = 1,
    **kwargs,
) -> dict[str, Any]:
    original = _A39._A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A40Planner(RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner):
        def __init__(self, *args, **planner_kwargs):
            super().__init__(
                *args,
                learned_sd_consumer_checkpoint=multihead_sd_consumer_checkpoint,
                learned_sd_consumer_weight=learned_sd_consumer_weight,
                learned_sd_consumer_root_override=learned_sd_consumer_root_override,
                neural_sd_consumer_score_clip=neural_sd_consumer_score_clip,
                multihead_sd_consumer_value_weight=multihead_sd_consumer_value_weight,
                multihead_sd_consumer_release_weight=multihead_sd_consumer_release_weight,
                multihead_sd_consumer_loop_weight=multihead_sd_consumer_loop_weight,
                multihead_sd_consumer_trap_weight=multihead_sd_consumer_trap_weight,
                agenda_option_tree_max_pushes=agenda_option_tree_max_pushes,
                agenda_option_tree_max_nodes=agenda_option_tree_max_nodes,
                agenda_option_tree_max_root_candidates=(
                    agenda_option_tree_max_root_candidates
                ),
                agenda_option_tree_min_tier=agenda_option_tree_min_tier,
                agenda_option_tree_min_tier_improvement=(
                    agenda_option_tree_min_tier_improvement
                ),
                **planner_kwargs,
            )

    _A39._A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A40Planner
    try:
        result = _A39._A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.run_a25_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A39._A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a40_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a39 = payload["a39_closed_loop_smoke"]["summary"]
    a40 = payload["a40_closed_loop_smoke"]["summary"]
    cmp_a39 = payload["comparison_a39_a40"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A40 agenda-option tree 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A40 继承 A39，对应专家 27 点中的第 14 点：在 root 处将单步候选"
            "提升为 bounded agenda-option 评价。后续 suffix 只用于评价，不作为宏动作执行。"
        ),
        (
            f"A39 为 `{a39['status']}`、pushes={a39['executed_pushes']}、"
            f"decisions={a39['closed_loop_decision_steps']}、goals={a39['boxes_on_goals_final']}；"
            f"A40 为 `{a40['status']}`、pushes={a40['executed_pushes']}、"
            f"decisions={a40['closed_loop_decision_steps']}、goals={a40['boxes_on_goals_final']}。"
        ),
        (
            f"A39/A40 首个决策差异为 `{cmp_a39['a39_a40_first_decision_difference_step']}`；"
            f"A40 option override 次数为 `{a40.get('agenda_option_tree_override_count', 0)}`，"
            f"selected progress 次数为 `{a40.get('agenda_option_tree_selected_progress_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | release actions | tier override | option loaded | option override | option progress | option done | option loop |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A39 | {a39['status']} | {a39['closed_loop_decision_steps']} | {a39['executed_pushes']} | "
            f"{a39['boxes_on_goals_final']} | {_fmt(a39['final_slot_schedule_cost'])} | "
            f"{a39['final_release_action_count']} | "
            f"{a39.get('agenda_tier_choice_override_count', 'NA')} | NA | NA | NA | NA | NA |"
        ),
        (
            f"| A40 | {a40['status']} | {a40['closed_loop_decision_steps']} | {a40['executed_pushes']} | "
            f"{a40['boxes_on_goals_final']} | {_fmt(a40['final_slot_schedule_cost'])} | "
            f"{a40['final_release_action_count']} | "
            f"{a40.get('agenda_tier_choice_override_count', 'NA')} | "
            f"{a40.get('agenda_option_tree_loaded_count', 'NA')} | "
            f"{a40.get('agenda_option_tree_override_count', 'NA')} | "
            f"{a40.get('agenda_option_tree_selected_progress_count', 'NA')} | "
            f"{a40.get('agenda_option_tree_selected_done_count', 'NA')} | "
            f"{a40.get('agenda_option_tree_selected_loop_count', 'NA')} |"
        ),
        "",
        "## A40 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | root tier | option tier | override | option reason | option depth | option progress | option done | loop |",
        "|---:|---|---:|---:|---|---:|---:|---:|---|---|---:|---|---|---|",
    ]
    for row in payload["a40_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        cert = row.get("chosen_sd_value", {}).get("agenda_certificate", {}) or {}
        option = _option_tree(row)
        selected = option.get("selected_option", {}) or {}
        progress = selected.get("best_progress", {}) or {}
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{cert.get('agenda_tier')} | "
            f"{option.get('selected_option_tier')} | "
            f"{option.get('agenda_option_tree_override')} | "
            f"{option.get('agenda_option_tree_reason')} | "
            f"{selected.get('best_depth')} | "
            f"{progress.get('agenda_progress')} | "
            f"{progress.get('agenda_done')} | "
            f"{progress.get('agenda_loop')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A40 检验的是：A39 已能按 tier 改写 root 选择后，短程 option 评价是否能"
                "进一步稳定识别可完成当前 consume-release 议程的候选。若仍不能求解，"
                "缺口更可能在显式 agenda state 或 learned agenda distance，而非单步分数。"
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
        "--a39-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--multihead-sd-consumer-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json",
    )
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=80)
    parser.add_argument("--learned-sd-consumer-weight", type=float, default=0.35)
    parser.add_argument("--neural-sd-consumer-score-clip", type=float, default=3.0)
    parser.add_argument("--multihead-sd-consumer-value-weight", type=float, default=0.45)
    parser.add_argument("--multihead-sd-consumer-release-weight", type=float, default=1.15)
    parser.add_argument("--multihead-sd-consumer-loop-weight", type=float, default=1.05)
    parser.add_argument("--multihead-sd-consumer-trap-weight", type=float, default=1.25)
    parser.add_argument("--learned-sd-consumer-root-override", action="store_true")
    parser.add_argument("--agenda-option-tree-max-pushes", type=int, default=5)
    parser.add_argument("--agenda-option-tree-max-nodes", type=int, default=96)
    parser.add_argument("--agenda-option-tree-max-root-candidates", type=int, default=10)
    parser.add_argument("--agenda-option-tree-min-tier", type=int, default=5)
    parser.add_argument("--agenda-option-tree-min-tier-improvement", type=int, default=1)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a31_path = ROOT / args.a31_json
    a39_path = ROOT / args.a39_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    if not a31_path.exists():
        raise FileNotFoundError(f"A31 baseline JSON not found: {a31_path}")
    if not a39_path.exists():
        raise FileNotFoundError(f"A39 baseline JSON not found: {a39_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"A35 checkpoint not found: {checkpoint_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a31_payload = json.loads(a31_path.read_text(encoding="utf-8"))
    a39_payload = json.loads(a39_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = _A39._A38._A37._A36.load_boxoban_levels(
        ROOT / config["levels"],
        max_levels=level_offset + 1,
    )[level_offset]
    model = _A39._A38._A37._A36._A35._A34._A33._A32._A31._A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
        level,
        config,
    )
    common = dict(
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=6,
        corridor_macro_max_extra_pushes=8,
        corridor_macro_min_total_pushes=2,
        schedule_lookahead_slots=2,
        recent_edge_window=12,
        recovery_min_legal_pushes=2,
        recovery_macro_max_extra_pushes=4,
        recovery_macro_max_nodes=20,
        recovery_macro_score_threshold=2.0,
        dependency_bound_macro_score_threshold=20.0,
        feasibility_guard_max_slot_cost_regression=0.0,
        feasibility_guard_max_schedule_barrier_regression=8.0,
        recoverability_probe_depth=2,
        recoverability_probe_max_nodes=8,
        survivable_recoverability_min_depth=2,
        survivable_recoverability_min_legal_pushes=3,
        stage_release_min_goals=2,
        stage_stand_lock_score_scale=0.35,
        stage_release_stand_occupied_resolution_weight=85.0,
        stage_release_goal_nonregression=True,
        pre_release_corridor_guard_min_pushes=2,
        pre_release_corridor_guard_legal_margin=2,
        pre_release_root_trap_legal_margin=2,
        pre_release_root_trap_locked_increase_penalty=110.0,
        pre_release_root_trap_locked_absolute_penalty=30.0,
        pre_release_root_trap_conversion_penalty=70.0,
        pre_release_root_trap_occupied_increase_penalty=18.0,
        pre_release_root_trap_low_margin_penalty=24.0,
        pre_release_root_trap_no_release_penalty=10.0,
        pre_release_root_trap_dependency_reduction_bonus=18.0,
        pre_release_root_trap_target_reduction_bonus=14.0,
        pre_release_root_trap_release_open_bonus=24.0,
        pre_release_root_trap_legal_margin_bonus=4.0,
        sd_generalization_legal_margin=2,
        sd_generalization_predecessor_penalty=95.0,
        sd_generalization_all_bad_successor_penalty=125.0,
        sd_generalization_failure_similarity_penalty=70.0,
        sd_generalization_success_similarity_bonus=20.0,
        sd_generalization_dependency_reduction_bonus=18.0,
        sd_generalization_release_open_bonus=24.0,
        sd_generalization_legal_margin_bonus=4.0,
    )
    a40 = run_a40_smoke(
        model,
        config,
        multihead_sd_consumer_checkpoint=str(checkpoint_path),
        learned_sd_consumer_weight=float(args.learned_sd_consumer_weight),
        learned_sd_consumer_root_override=bool(args.learned_sd_consumer_root_override),
        neural_sd_consumer_score_clip=float(args.neural_sd_consumer_score_clip),
        multihead_sd_consumer_value_weight=float(args.multihead_sd_consumer_value_weight),
        multihead_sd_consumer_release_weight=float(args.multihead_sd_consumer_release_weight),
        multihead_sd_consumer_loop_weight=float(args.multihead_sd_consumer_loop_weight),
        multihead_sd_consumer_trap_weight=float(args.multihead_sd_consumer_trap_weight),
        agenda_option_tree_max_pushes=int(args.agenda_option_tree_max_pushes),
        agenda_option_tree_max_nodes=int(args.agenda_option_tree_max_nodes),
        agenda_option_tree_max_root_candidates=int(
            args.agenda_option_tree_max_root_candidates
        ),
        agenda_option_tree_min_tier=int(args.agenda_option_tree_min_tier),
        agenda_option_tree_min_tier_improvement=int(
            args.agenda_option_tree_min_tier_improvement
        ),
        seed_offset=52759,
        **common,
    )
    a31 = a31_payload["a31_closed_loop_smoke"]
    a39 = a39_payload["a39_closed_loop_smoke"]
    comparison_a31 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a31",
        a31,
        "a40",
        a40,
    )
    comparison_a39 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a39",
        a39,
        "a40",
        a40,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a40_agenda_option_tree_v1",
            "agenda_gft_alignment": "E14_agenda_option_tree",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "a39_json": _rel(a39_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a39_planner": _A39.RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner.provider_id,
            "a40_planner": RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
            "trace_scope": "a31_a39_a40_agenda_option_tree_smoke",
        },
        "config_extract": {
            **a39_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "agenda_option_tree_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E14_agenda_option_tree",
            "agenda_option_tree_mode": "consume_release_triggered",
            "agenda_option_tree_max_pushes": int(args.agenda_option_tree_max_pushes),
            "agenda_option_tree_max_nodes": int(args.agenda_option_tree_max_nodes),
            "agenda_option_tree_max_root_candidates": int(
                args.agenda_option_tree_max_root_candidates
            ),
            "agenda_option_tree_min_tier": int(args.agenda_option_tree_min_tier),
            "agenda_option_tree_min_tier_improvement": int(
                args.agenda_option_tree_min_tier_improvement
            ),
            "learned_sd_consumer_weight": float(args.learned_sd_consumer_weight),
            "neural_sd_consumer_score_clip": float(args.neural_sd_consumer_score_clip),
            "multihead_sd_consumer_value_weight": float(
                args.multihead_sd_consumer_value_weight
            ),
            "multihead_sd_consumer_release_weight": float(
                args.multihead_sd_consumer_release_weight
            ),
            "multihead_sd_consumer_loop_weight": float(
                args.multihead_sd_consumer_loop_weight
            ),
            "multihead_sd_consumer_trap_weight": float(
                args.multihead_sd_consumer_trap_weight
            ),
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a31_closed_loop_smoke": a31,
        "a39_closed_loop_smoke": a39,
        "a40_closed_loop_smoke": a40,
        "comparison_a31_a40": comparison_a31,
        "comparison_a39_a40": comparison_a39,
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
