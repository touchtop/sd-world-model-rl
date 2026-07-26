#!/usr/bin/env python3
"""A38 agenda leaf-potential diagnostics for XSokoban first levels."""

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

_A37_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a37_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a37_active_agenda_certificate.py",
)
if _A37_SPEC is None or _A37_SPEC.loader is None:
    raise RuntimeError("Unable to load A37 diagnostic helpers.")
_A37 = importlib.util.module_from_spec(_A37_SPEC)
_A37_SPEC.loader.exec_module(_A37)

from src.sokoban import (  # noqa: E402
    RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A37._rel(path)


def _fmt(value: Any) -> str:
    return _A37._fmt(value)


def _leaf(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("agenda_leaf_potential", {}) or {}


def augment_a38_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A37.augment_a37_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_leaf(row) for row in decision_rows if _leaf(row)]
    adjustments = [
        float(row["agenda_leaf_adjustment"])
        for row in loaded
        if row.get("agenda_leaf_adjustment") is not None
    ]
    eval_counts = [
        int(row["agenda_leaf_eval_count"])
        for row in loaded
        if row.get("agenda_leaf_eval_count") is not None
    ]
    loop_eval_counts = [
        int(row.get("agenda_leaf_loop_eval_count", 0)) for row in loaded
    ]
    stale_eval_counts = [
        int(row.get("agenda_leaf_stale_release_eval_count", 0)) for row in loaded
    ]
    type_counts: dict[str, int] = {}
    for row in loaded:
        agenda_type = str(row.get("agenda_leaf_agenda_type", "NA"))
        type_counts[agenda_type] = type_counts.get(agenda_type, 0) + 1
    result["summary"].update(
        {
            "agenda_leaf_loaded_count": int(len(loaded)),
            "agenda_leaf_adjustment_min": None
            if not adjustments
            else float(min(adjustments)),
            "agenda_leaf_adjustment_max": None
            if not adjustments
            else float(max(adjustments)),
            "agenda_leaf_adjustment_mean": None
            if not adjustments
            else float(sum(adjustments) / len(adjustments)),
            "agenda_leaf_eval_count_total": int(sum(eval_counts)),
            "agenda_leaf_loop_eval_count_total": int(sum(loop_eval_counts)),
            "agenda_leaf_stale_release_eval_count_total": int(sum(stale_eval_counts)),
            "agenda_leaf_root_loop_count": int(
                sum(bool(row.get("agenda_leaf_loop")) for row in loaded)
            ),
            "agenda_leaf_root_stale_open_release_count": int(
                sum(bool(row.get("agenda_leaf_stale_open_release")) for row in loaded)
            ),
            "agenda_leaf_type_counts": type_counts,
        }
    )
    return result


def run_a38_smoke(
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
    original = _A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A38Planner(RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner):
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

    _A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A38Planner
    try:
        result = _A37._A36._A35._A34._A33._A32._A31._A30._A25.run_a25_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A37._A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a38_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a31 = payload["a31_closed_loop_smoke"]["summary"]
    a38 = payload["a38_closed_loop_smoke"]["summary"]
    cmp_a31 = payload["comparison_a31_a38"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A38 agenda leaf potential 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A38 继承 A37，复用 A35 multi-head checkpoint。新增部分对应专家 27 点中的 E3："
            "在 MCTS leaf value 中加入显式 agenda potential，检验 active-agenda 信号从 root "
            "soft prior 移到 leaf 消费端后是否改变闭环。"
        ),
        (
            f"A31 为 `{a31['status']}`、pushes={a31['executed_pushes']}、"
            f"decisions={a31['closed_loop_decision_steps']}、goals={a31['boxes_on_goals_final']}；"
            f"A38 为 `{a38['status']}`、pushes={a38['executed_pushes']}、"
            f"decisions={a38['closed_loop_decision_steps']}、goals={a38['boxes_on_goals_final']}。"
        ),
        (
            f"A31/A38 首个决策差异为 `{cmp_a31['a31_a38_first_decision_difference_step']}`；"
            f"A38 leaf 加载步数为 `{a38.get('agenda_leaf_loaded_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | release actions | active progress | off agenda | leaf adj mean | leaf loop eval | leaf stale eval |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A31 | {a31['status']} | {a31['closed_loop_decision_steps']} | {a31['executed_pushes']} | "
            f"{a31['boxes_on_goals_final']} | {_fmt(a31['final_slot_schedule_cost'])} | "
            f"{a31['final_release_action_count']} | NA | NA | NA | NA | NA |"
        ),
        (
            f"| A38 | {a38['status']} | {a38['closed_loop_decision_steps']} | {a38['executed_pushes']} | "
            f"{a38['boxes_on_goals_final']} | {_fmt(a38['final_slot_schedule_cost'])} | "
            f"{a38['final_release_action_count']} | "
            f"{a38.get('active_agenda_progress_count', 'NA')} | "
            f"{a38.get('active_agenda_off_agenda_action_count', 'NA')} | "
            f"{_fmt(a38.get('agenda_leaf_adjustment_mean'))} | "
            f"{a38.get('agenda_leaf_loop_eval_count_total', 'NA')} | "
            f"{a38.get('agenda_leaf_stale_release_eval_count_total', 'NA')} |"
        ),
        "",
        "## A38 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | release actions | score | agenda | progress | off | loop | leaf type | leaf adj | leaf eval | leaf loop eval |",
        "|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["a38_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        sdv = row.get("chosen_sd_value", {})
        cert = sdv.get("agenda_certificate", {}) or {}
        leaf = _leaf(row)
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['release_action_count']} | "
            f"{_fmt(sdv.get('learned_sd_consumer_score'))} | "
            f"{cert.get('active_agenda_type')} | "
            f"{cert.get('active_agenda_progress')} | "
            f"{cert.get('active_agenda_off_agenda_action')} | "
            f"{cert.get('active_agenda_release_loop')} | "
            f"{leaf.get('agenda_leaf_agenda_type')} | "
            f"{_fmt(leaf.get('agenda_leaf_adjustment'))} | "
            f"{leaf.get('agenda_leaf_eval_count')} | "
            f"{leaf.get('agenda_leaf_loop_eval_count')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A38 只检验 E3 leaf agenda potential，不新增图网络、teacher imitation、"
                "hard gate 或 agenda-option tree。若该版本仍与 A37 同轨迹，则说明仅将"
                "显式 agenda potential 接入 leaf 仍不足以改变当前 root/top-child 优势。"
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.md",
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
    level = _A37._A36.load_boxoban_levels(
        ROOT / config["levels"],
        max_levels=level_offset + 1,
    )[level_offset]
    model = _A37._A36._A35._A34._A33._A32._A31._A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a38 = run_a38_smoke(
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
    comparison = _A37._A36._A35._A34._A33._A32.compare_runs("a31", a31, "a38", a38)
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a38_agenda_leaf_potential_v1",
            "agenda_gft_alignment": "E3_leaf_value_uses_agenda_potential",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a36_planner": _A37._A36.RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner.provider_id,
            "a37_planner": _A37.RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner.provider_id,
            "a38_planner": RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a31_a38_agenda_leaf_potential_smoke",
        },
        "config_extract": {
            **a31_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "agenda_leaf_potential_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E3_leaf_value_uses_agenda_potential",
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
        "a38_closed_loop_smoke": a38,
        "comparison_a31_a38": comparison,
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
