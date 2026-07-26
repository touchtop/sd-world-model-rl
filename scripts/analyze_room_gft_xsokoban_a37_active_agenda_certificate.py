#!/usr/bin/env python3
"""A37 active-agenda certificate diagnostics for XSokoban first levels."""

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

_A36_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a36_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a36_agenda_certificate.py",
)
if _A36_SPEC is None or _A36_SPEC.loader is None:
    raise RuntimeError("Unable to load A36 diagnostic helpers.")
_A36 = importlib.util.module_from_spec(_A36_SPEC)
_A36_SPEC.loader.exec_module(_A36)

from src.sokoban import (  # noqa: E402
    RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A36._rel(path)


def _fmt(value: Any) -> str:
    return _A36._fmt(value)


def _active_cert(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("chosen_sd_value", {}).get("agenda_certificate", {}) or {}


def _summary_bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(bool(_active_cert(row).get(key)) for row in rows))


def augment_a37_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A36.augment_a36_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded_rows = [
        row
        for row in decision_rows
        if bool(
            row.get("chosen_sd_value", {}).get(
                "agenda_certificate_sd_consumer_enabled"
            )
        )
    ]
    type_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    blocker_deltas: list[int] = []
    for row in loaded_rows:
        cert = _active_cert(row)
        agenda_type = str(cert.get("active_agenda_type", "NA"))
        type_counts[agenda_type] = type_counts.get(agenda_type, 0) + 1
        reason = str(cert.get("active_agenda_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if cert.get("active_agenda_blocker_delta") is not None:
            blocker_deltas.append(int(cert["active_agenda_blocker_delta"]))

    result["summary"].update(
        {
            "active_agenda_loaded_count": int(len(loaded_rows)),
            "active_agenda_progress_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_progress",
            ),
            "active_agenda_off_agenda_action_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_off_agenda_action",
            ),
            "active_agenda_pseudo_progress_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_pseudo_progress",
            ),
            "active_agenda_stale_open_release_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_stale_open_release",
            ),
            "active_agenda_loop_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_loop",
            ),
            "active_agenda_release_loop_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_release_loop",
            ),
            "active_agenda_release_consumed_count": _summary_bool_count(
                loaded_rows,
                "active_agenda_release_consumed",
            ),
            "active_agenda_type_counts": type_counts,
            "active_agenda_reason_counts": reason_counts,
            "active_agenda_blocker_delta_min": None
            if not blocker_deltas
            else int(min(blocker_deltas)),
            "active_agenda_blocker_delta_max": None
            if not blocker_deltas
            else int(max(blocker_deltas)),
        }
    )
    return result


def run_a37_smoke(
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
    original = _A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A37Planner(RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner):
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

    _A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A37Planner
    try:
        result = _A36._A35._A34._A33._A32._A31._A30._A25.run_a25_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A36._A35._A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a37_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a31 = payload["a31_closed_loop_smoke"]["summary"]
    a37 = payload["a37_closed_loop_smoke"]["summary"]
    cmp_a31 = payload["comparison_a31_a37"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A37 active agenda 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A37 继承 A36，复用 A35 multi-head checkpoint。新增部分是最小 active-agenda "
            "层：先判定当前义务为 fill_frontier_slot、release_blocker 或 consume_release，"
            "再判断候选动作是否推进该义务。"
        ),
        (
            f"A31 为 `{a31['status']}`、pushes={a31['executed_pushes']}、"
            f"decisions={a31['closed_loop_decision_steps']}、goals={a31['boxes_on_goals_final']}；"
            f"A37 为 `{a37['status']}`、pushes={a37['executed_pushes']}、"
            f"decisions={a37['closed_loop_decision_steps']}、goals={a37['boxes_on_goals_final']}。"
        ),
        (
            f"A31/A37 首个决策差异为 `{cmp_a31['a31_a37_first_decision_difference_step']}`；"
            f"A37 active-agenda 加载步数为 `{a37.get('active_agenda_loaded_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | release actions | active progress | off agenda | pseudo | release loop | consumed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A31 | {a31['status']} | {a31['closed_loop_decision_steps']} | {a31['executed_pushes']} | "
            f"{a31['boxes_on_goals_final']} | {_fmt(a31['final_slot_schedule_cost'])} | "
            f"{a31['final_release_action_count']} | NA | NA | NA | NA | NA |"
        ),
        (
            f"| A37 | {a37['status']} | {a37['closed_loop_decision_steps']} | {a37['executed_pushes']} | "
            f"{a37['boxes_on_goals_final']} | {_fmt(a37['final_slot_schedule_cost'])} | "
            f"{a37['final_release_action_count']} | "
            f"{a37.get('active_agenda_progress_count', 'NA')} | "
            f"{a37.get('active_agenda_off_agenda_action_count', 'NA')} | "
            f"{a37.get('active_agenda_pseudo_progress_count', 'NA')} | "
            f"{a37.get('active_agenda_release_loop_count', 'NA')} | "
            f"{a37.get('active_agenda_release_consumed_count', 'NA')} |"
        ),
        "",
        "## A37 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | release actions | score | agenda | blk delta | progress | off | loop | rel raw | rel eff | tier |",
        "|---:|---|---:|---:|---|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|",
    ]
    for row in payload["a37_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        sdv = row.get("chosen_sd_value", {})
        cert = sdv.get("agenda_certificate", {}) or {}
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['release_action_count']} | "
            f"{_fmt(sdv.get('learned_sd_consumer_score'))} | "
            f"{cert.get('active_agenda_type')} | "
            f"{cert.get('active_agenda_blocker_delta')} | "
            f"{cert.get('active_agenda_progress')} | "
            f"{cert.get('active_agenda_off_agenda_action')} | "
            f"{cert.get('active_agenda_release_loop')} | "
            f"{_fmt(cert.get('release_head_raw'))} | {_fmt(cert.get('release_head_effective'))} | "
            f"{cert.get('agenda_tier')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A37 检验专家建议中的 active obligation 命题：progress certificate 必须绑定当前义务。"
                "局部 release closure 或 blocker delta 只有在服务当前义务时才保留为 certified progress；"
                "重复上下文中的 release closure 被记录为 off-agenda/release-loop。"
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.md",
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
    level = _A36.load_boxoban_levels(
        ROOT / config["levels"],
        max_levels=level_offset + 1,
    )[level_offset]
    model = _A36._A35._A34._A33._A32._A31._A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a37 = run_a37_smoke(
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
    comparison = _A36._A35._A34._A33._A32.compare_runs("a31", a31, "a37", a37)
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a37_active_agenda_certificate_v1",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a25_planner": _A36.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a31_planner": _A36.RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner.provider_id,
            "a35_planner": _A36.RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner.provider_id,
            "a36_planner": _A36.RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner.provider_id,
            "a37_planner": RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a31_a37_active_agenda_certificate_smoke",
        },
        "config_extract": {
            **a31_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "active_agenda_certificate_multihead_soft_prior_v1",
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
        "a37_closed_loop_smoke": a37,
        "comparison_a31_a37": comparison,
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
