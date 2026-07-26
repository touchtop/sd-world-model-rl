#!/usr/bin/env python3
"""A39 lexicographic agenda-tier diagnostics for XSokoban first levels."""

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

_A38_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a38_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a38_agenda_leaf_potential.py",
)
if _A38_SPEC is None or _A38_SPEC.loader is None:
    raise RuntimeError("Unable to load A38 diagnostic helpers.")
_A38 = importlib.util.module_from_spec(_A38_SPEC)
_A38_SPEC.loader.exec_module(_A38)

from src.sokoban import (  # noqa: E402
    RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A38._rel(path)


def _fmt(value: Any) -> str:
    return _A38._fmt(value)


def _tier_choice(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("agenda_tier_choice", {}) or {}


def augment_a39_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A38.augment_a38_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_tier_choice(row) for row in decision_rows if _tier_choice(row)]
    reason_counts: dict[str, int] = {}
    parent_tier_counts: dict[str, int] = {}
    selected_tier_counts: dict[str, int] = {}
    max_tier_counts: dict[str, int] = {}
    for row in loaded:
        reason = str(row.get("agenda_tier_choice_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        parent_tier = str(row.get("parent_tier", "NA"))
        selected_tier = str(row.get("selected_tier", "NA"))
        max_tier = str(row.get("max_available_tier", "NA"))
        parent_tier_counts[parent_tier] = parent_tier_counts.get(parent_tier, 0) + 1
        selected_tier_counts[selected_tier] = selected_tier_counts.get(selected_tier, 0) + 1
        max_tier_counts[max_tier] = max_tier_counts.get(max_tier, 0) + 1
    result["summary"].update(
        {
            "agenda_tier_choice_loaded_count": int(len(loaded)),
            "agenda_tier_choice_override_count": int(
                sum(bool(row.get("agenda_tier_choice_override")) for row in loaded)
            ),
            "agenda_tier_choice_higher_tier_selected_count": int(
                sum(
                    int(row.get("selected_tier", 0)) > int(row.get("parent_tier", 0))
                    for row in loaded
                )
            ),
            "agenda_tier_choice_reason_counts": reason_counts,
            "agenda_tier_choice_parent_tier_counts": parent_tier_counts,
            "agenda_tier_choice_selected_tier_counts": selected_tier_counts,
            "agenda_tier_choice_max_tier_counts": max_tier_counts,
        }
    )
    return result


def run_a39_smoke(
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
    **kwargs,
) -> dict[str, Any]:
    original = _A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A39Planner(RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner):
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
                **planner_kwargs,
            )

    _A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A39Planner
    try:
        result = _A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.run_a25_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A38._A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a39_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a31 = payload["a31_closed_loop_smoke"]["summary"]
    a39 = payload["a39_closed_loop_smoke"]["summary"]
    cmp_a31 = payload["comparison_a31_a39"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A39 agenda tier choice 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A39 继承 A38，复用 A35 multi-head checkpoint。新增部分对应专家 27 点中的"
            "第 16 点：把已有 agenda_tier/off-agenda/release-loop 证书从软分数转为"
            "root choice 的 lexicographic tier 比较。"
        ),
        (
            f"A31 为 `{a31['status']}`、pushes={a31['executed_pushes']}、"
            f"decisions={a31['closed_loop_decision_steps']}、goals={a31['boxes_on_goals_final']}；"
            f"A39 为 `{a39['status']}`、pushes={a39['executed_pushes']}、"
            f"decisions={a39['closed_loop_decision_steps']}、goals={a39['boxes_on_goals_final']}。"
        ),
        (
            f"A31/A39 首个决策差异为 `{cmp_a31['a31_a39_first_decision_difference_step']}`；"
            f"A39 tier override 次数为 `{a39.get('agenda_tier_choice_override_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | release actions | tier loaded | tier override | higher-tier selected | off agenda | release loop |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A31 | {a31['status']} | {a31['closed_loop_decision_steps']} | {a31['executed_pushes']} | "
            f"{a31['boxes_on_goals_final']} | {_fmt(a31['final_slot_schedule_cost'])} | "
            f"{a31['final_release_action_count']} | NA | NA | NA | NA | NA |"
        ),
        (
            f"| A39 | {a39['status']} | {a39['closed_loop_decision_steps']} | {a39['executed_pushes']} | "
            f"{a39['boxes_on_goals_final']} | {_fmt(a39['final_slot_schedule_cost'])} | "
            f"{a39['final_release_action_count']} | "
            f"{a39.get('agenda_tier_choice_loaded_count', 'NA')} | "
            f"{a39.get('agenda_tier_choice_override_count', 'NA')} | "
            f"{a39.get('agenda_tier_choice_higher_tier_selected_count', 'NA')} | "
            f"{a39.get('active_agenda_off_agenda_action_count', 'NA')} | "
            f"{a39.get('active_agenda_release_loop_count', 'NA')} |"
        ),
        "",
        "## A39 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | release actions | parent tier | selected tier | max tier | override | agenda | progress | off | loop |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in payload["a39_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        sdv = row.get("chosen_sd_value", {})
        cert = sdv.get("agenda_certificate", {}) or {}
        tier = _tier_choice(row)
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['release_action_count']} | "
            f"{tier.get('parent_tier')} | {tier.get('selected_tier')} | "
            f"{tier.get('max_available_tier')} | "
            f"{tier.get('agenda_tier_choice_override')} | "
            f"{cert.get('active_agenda_type')} | "
            f"{cert.get('active_agenda_progress')} | "
            f"{cert.get('active_agenda_off_agenda_action')} | "
            f"{cert.get('active_agenda_release_loop')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A39 只检验 lexicographic tier / hard downgrade，不新增动作生成或 teacher 信号。"
                "如果它改变轨迹但仍未求解，说明 active agenda 的分层消费方向有效，后续"
                "仍应在专家 27 点内围绕 option tree 或更精细的 tier 定义做单独决策。"
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
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a31_path = ROOT / args.a31_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    if not a31_path.exists():
        raise FileNotFoundError(f"A31 baseline JSON not found: {a31_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"A35 checkpoint not found: {checkpoint_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a31_payload = json.loads(a31_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = _A38._A37._A36.load_boxoban_levels(
        ROOT / config["levels"],
        max_levels=level_offset + 1,
    )[level_offset]
    model = _A38._A37._A36._A35._A34._A33._A32._A31._A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a39 = run_a39_smoke(
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
        seed_offset=51425,
        **common,
    )
    a31 = a31_payload["a31_closed_loop_smoke"]
    comparison = _A38._A37._A36._A35._A34._A33._A32.compare_runs("a31", a31, "a39", a39)
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a39_agenda_tier_choice_v1",
            "agenda_gft_alignment": "E16_lexicographic_agenda_tier_root_choice",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a36_planner": _A38._A37._A36.RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner.provider_id,
            "a37_planner": _A38._A37.RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner.provider_id,
            "a38_planner": _A38.RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner.provider_id,
            "a39_planner": RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a31_a39_agenda_tier_choice_smoke",
        },
        "config_extract": {
            **a31_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "agenda_tier_choice_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E16_lexicographic_agenda_tier_root_choice",
            "agenda_tier_choice_mode": "release_loop_only",
            "agenda_tier_choice_parent_max_tier": 1,
            "agenda_tier_choice_min_improvement": 1,
            "learned_sd_consumer_weight": float(args.learned_sd_consumer_weight),
            "neural_sd_consumer_score_clip": float(args.neural_sd_consumer_score_clip),
            "multihead_sd_consumer_value_weight": float(args.multihead_sd_consumer_value_weight),
            "multihead_sd_consumer_release_weight": float(args.multihead_sd_consumer_release_weight),
            "multihead_sd_consumer_loop_weight": float(args.multihead_sd_consumer_loop_weight),
            "multihead_sd_consumer_trap_weight": float(args.multihead_sd_consumer_trap_weight),
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
        "comparison_a31_a39": comparison,
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
