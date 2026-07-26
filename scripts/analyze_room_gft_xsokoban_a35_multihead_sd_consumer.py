#!/usr/bin/env python3
"""A35 multi-head SD-consumer diagnostics for XSokoban first levels."""

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

_A34_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a34_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a34_milestone_sd_consumer.py",
)
if _A34_SPEC is None or _A34_SPEC.loader is None:
    raise RuntimeError("Unable to load A34 diagnostic helpers.")
_A34 = importlib.util.module_from_spec(_A34_SPEC)
_A34_SPEC.loader.exec_module(_A34)

from src.sokoban import (  # noqa: E402
    RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner,
    RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner,
    RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner,
    RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner,
    RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner,
    RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner,
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
    return _A34._fmt(value)


def augment_a35_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A34._A33._A32.augment_a32_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded_rows = [
        row
        for row in decision_rows
        if bool(row.get("chosen_sd_value", {}).get("multihead_sd_consumer_enabled"))
        and bool(row.get("chosen_sd_value", {}).get("learned_sd_consumer_checkpoint_loaded"))
    ]
    scores = [
        float(row.get("chosen_sd_value", {}).get("learned_sd_consumer_score"))
        for row in loaded_rows
        if row.get("chosen_sd_value", {}).get("learned_sd_consumer_score") is not None
    ]
    head_names = ("value", "release_progress", "loop_risk", "trap_risk")
    head_stats: dict[str, dict[str, float | None]] = {}
    for head_name in head_names:
        values = [
            float(
                row.get("chosen_sd_value", {})
                .get("multihead_sd_consumer_heads", {})
                .get(head_name)
            )
            for row in loaded_rows
            if row.get("chosen_sd_value", {})
            .get("multihead_sd_consumer_heads", {})
            .get(head_name)
            is not None
        ]
        head_stats[head_name] = {
            "min": None if not values else float(min(values)),
            "max": None if not values else float(max(values)),
            "mean": None if not values else float(sum(values) / len(values)),
        }
    result["summary"].update(
        {
            "multihead_sd_consumer_loaded_count": int(len(loaded_rows)),
            "multihead_sd_consumer_selected_positive_count": int(
                len([score for score in scores if score > 0.0])
            ),
            "multihead_sd_consumer_score_min": None
            if not scores
            else float(min(scores)),
            "multihead_sd_consumer_score_max": None
            if not scores
            else float(max(scores)),
            "multihead_sd_consumer_heads": head_stats,
        }
    )
    return result


def run_a35_smoke(
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
    original = _A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A35Planner(RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner):
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

    _A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A35Planner
    try:
        result = _A34._A33._A32._A31._A30._A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A34._A33._A32._A31._A30._A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a35_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a31 = payload["a31_closed_loop_smoke"]["summary"]
    a35 = payload["a35_closed_loop_smoke"]["summary"]
    cmp_a31 = payload["comparison_a31_a35"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A35 multi-head SD-consumer 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A35 继承 A34/A33/A32/A31，并加载 value、release-progress、loop-risk、"
            "trap-risk 四头 MLP checkpoint。默认配置只将组合分数作为 root prior/value "
            "的软信号消费，不启用 root hard override；该版本不接入 teacher/reference "
            "在线动作，也不替代 exact Sokoban dynamics。"
        ),
        (
            f"A31 为 `{a31['status']}`、pushes={a31['executed_pushes']}、"
            f"decisions={a31['closed_loop_decision_steps']}、goals={a31['boxes_on_goals_final']}、"
            f"release actions={a31['final_release_action_count']}；"
            f"A35 为 `{a35['status']}`、pushes={a35['executed_pushes']}、"
            f"decisions={a35['closed_loop_decision_steps']}、goals={a35['boxes_on_goals_final']}、"
            f"release actions={a35['final_release_action_count']}。"
        ),
        (
            f"A31/A35 首个决策差异为 `{cmp_a31['a31_a35_first_decision_difference_step']}`；"
            f"A35 multi-head scorer 加载步数为 `{a35.get('multihead_sd_consumer_loaded_count', 0)}`。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | multi-head loaded | positive |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A31 | {a31['status']} | {a31['closed_loop_decision_steps']} | {a31['executed_pushes']} | "
            f"{a31['boxes_on_goals_final']} | {_fmt(a31['final_slot_schedule_cost'])} | "
            f"{a31['final_dependency_count']} | {a31['final_target_blocker_count']} | "
            f"{a31['final_release_action_count']} | NA | NA |"
        ),
        (
            f"| A35 | {a35['status']} | {a35['closed_loop_decision_steps']} | {a35['executed_pushes']} | "
            f"{a35['boxes_on_goals_final']} | {_fmt(a35['final_slot_schedule_cost'])} | "
            f"{a35['final_dependency_count']} | {a35['final_target_blocker_count']} | "
            f"{a35['final_release_action_count']} | "
            f"{a35.get('multihead_sd_consumer_loaded_count', 'NA')} | "
            f"{a35.get('multihead_sd_consumer_selected_positive_count', 'NA')} |"
        ),
        "",
        "## A35 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dependencies | release actions | score | value | release | loop | trap | macro reason |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["a35_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        sdv = row.get("chosen_sd_value", {})
        heads = sdv.get("multihead_sd_consumer_heads", {}) or {}
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{row['schedule_plan_after']['totals']['release_action_count']} | "
            f"{_fmt(sdv.get('learned_sd_consumer_score'))} | "
            f"{_fmt(heads.get('value'))} | {_fmt(heads.get('release_progress'))} | "
            f"{_fmt(heads.get('loop_risk'))} | {_fmt(heads.get('trap_risk'))} | "
            f"{row.get('corridor_macro', {}).get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A35 的主要作用是检验 A34 单一 milestone scalar 是否造成机制混叠。"
                "四个头共享同一 SD 特征空间和同一离线 trace 来源，仍属于结构化 SD "
                "表示的神经消费者，而不是新增在线 teacher 或手写动作规则。"
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.md",
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
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    model = _A34._A33._A32._A31._A30._A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
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
    a35 = run_a35_smoke(
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
        seed_offset=51415,
        **common,
    )
    a31 = a31_payload["a31_closed_loop_smoke"]
    comparison = _A34._A33._A32.compare_runs("a31", a31, "a35", a35)
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a35_multihead_sd_consumer_v1",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a25_planner": RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a26_planner": RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner.provider_id,
            "a27_planner": RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner.provider_id,
            "a28_planner": RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner.provider_id,
            "a29_planner": RoomScheduleReleaseActionMacroGFTSDMCTSPlanner.provider_id,
            "a30_planner": RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner.provider_id,
            "a31_planner": RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner.provider_id,
            "a32_planner": RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner.provider_id,
            "a33_planner": RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner.provider_id,
            "a34_planner": RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner.provider_id,
            "a35_planner": RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a31_a35_multihead_sd_consumer_smoke",
        },
        "config_extract": {
            **a31_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "multihead_sd_consumer_soft_prior_v1",
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
        "a35_closed_loop_smoke": a35,
        "comparison_a31_a35": comparison,
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
