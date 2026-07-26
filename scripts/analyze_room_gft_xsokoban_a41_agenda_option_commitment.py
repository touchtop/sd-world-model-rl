#!/usr/bin/env python3
"""A41 bounded agenda-option commitment diagnostics for XSokoban first levels."""

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

_A40_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a40_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a40_agenda_option_tree.py",
)
if _A40_SPEC is None or _A40_SPEC.loader is None:
    raise RuntimeError("Unable to load A40 diagnostic helpers.")
_A40 = importlib.util.module_from_spec(_A40_SPEC)
_A40_SPEC.loader.exec_module(_A40)

_A39 = _A40._A39
_A25 = _A39._A38._A37._A36._A35._A34._A33._A32._A31._A30._A25

from src.sokoban import (  # noqa: E402
    RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A40._rel(path)


def _fmt(value: Any) -> str:
    return _A40._fmt(value)


def _commitment(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("agenda_option_commitment", {})
    return value if isinstance(value, dict) else {}


def augment_a41_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A40.augment_a40_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_commitment(row) for row in decision_rows if _commitment(row)]
    triggered = [
        row for row in loaded if bool(row.get("agenda_option_commitment_triggered"))
    ]
    reason_counts: dict[str, int] = {}
    for row in loaded:
        reason = str(row.get("agenda_option_commitment_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    first_trigger_step = None
    for trace_row in decision_rows:
        row = _commitment(trace_row)
        if bool(row.get("agenda_option_commitment_triggered")):
            first_trigger_step = int(trace_row["step"])
            break
    result["summary"].update(
        {
            "agenda_option_commitment_loaded_count": int(len(loaded)),
            "agenda_option_commitment_triggered_count": int(len(triggered)),
            "agenda_option_commitment_reason_counts": reason_counts,
            "agenda_option_commitment_first_trigger_step": first_trigger_step,
            "agenda_option_commitment_extra_push_count": int(
                sum(
                    int(row.get("committed_extra_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "agenda_option_commitment_validated_push_count": int(
                sum(
                    int(row.get("validated_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "agenda_option_commitment_done_count": int(
                sum(
                    bool(
                        row.get("final_progress", {}).get("agenda_done", False)
                        if isinstance(row.get("final_progress", {}), dict)
                        else False
                    )
                    for row in triggered
                )
            ),
        }
    )
    return result


def run_a41_smoke(
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
    agenda_option_commitment_min_tier: int = 7,
    agenda_option_commitment_require_done: bool = True,
    agenda_option_commitment_max_extra_pushes: int = 4,
    agenda_option_commitment_min_total_pushes: int = 2,
    **kwargs,
) -> dict[str, Any]:
    original = _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A41Planner(RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner):
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
                agenda_option_commitment_min_tier=agenda_option_commitment_min_tier,
                agenda_option_commitment_require_done=(
                    agenda_option_commitment_require_done
                ),
                agenda_option_commitment_max_extra_pushes=(
                    agenda_option_commitment_max_extra_pushes
                ),
                agenda_option_commitment_min_total_pushes=(
                    agenda_option_commitment_min_total_pushes
                ),
                **planner_kwargs,
            )

    _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A41Planner
    try:
        result = _A25.run_a25_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a41_summary(result)


def _common_args(args: argparse.Namespace) -> dict[str, Any]:
    return dict(
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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a40 = payload["a40_closed_loop_smoke"]["summary"]
    a41 = payload["a41_closed_loop_smoke"]["summary"]
    cmp_a40 = payload["comparison_a40_a41"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A41 agenda-option commitment 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A41 继承 A40，对应专家 27 点中的第 14 点和第 6 点：A40 负责在 root "
            "处发现短程 agenda-option，A41 只在该 option 已达到 tier-7 且 "
            "`agenda_done=true` 时，执行一个受限承诺后缀。"
        ),
        (
            f"A40 为 `{a40['status']}`、pushes={a40['executed_pushes']}、"
            f"decisions={a40['closed_loop_decision_steps']}、goals={a40['boxes_on_goals_final']}；"
            f"A41 为 `{a41['status']}`、pushes={a41['executed_pushes']}、"
            f"decisions={a41['closed_loop_decision_steps']}、goals={a41['boxes_on_goals_final']}。"
        ),
        (
            f"A40/A41 首个决策差异为 `{cmp_a40['a40_a41_first_decision_difference_step']}`；"
            f"A41 commitment 触发 `{a41.get('agenda_option_commitment_triggered_count', 0)}` 次，"
            f"额外执行 `{a41.get('agenda_option_commitment_extra_push_count', 0)}` 个承诺 push。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | release actions | option done | commitment triggers | commitment extra pushes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A40 | {a40['status']} | {a40['closed_loop_decision_steps']} | "
            f"{a40['executed_pushes']} | {a40['boxes_on_goals_final']} | "
            f"{_fmt(a40['final_slot_schedule_cost'])} | {a40['final_release_action_count']} | "
            f"{a40.get('agenda_option_tree_selected_done_count', 'NA')} | NA | NA |"
        ),
        (
            f"| A41 | {a41['status']} | {a41['closed_loop_decision_steps']} | "
            f"{a41['executed_pushes']} | {a41['boxes_on_goals_final']} | "
            f"{_fmt(a41['final_slot_schedule_cost'])} | {a41['final_release_action_count']} | "
            f"{a41.get('agenda_option_tree_selected_done_count', 'NA')} | "
            f"{a41.get('agenda_option_commitment_triggered_count', 'NA')} | "
            f"{a41.get('agenda_option_commitment_extra_push_count', 'NA')} |"
        ),
        "",
        "## A41 轨迹前缀",
        "",
        "| step | action | executed pushes | goals | stage | slot cost | option tier | commitment | reason | committed extra |",
        "|---:|---|---:|---:|---|---:|---:|---|---|---:|",
    ]
    for row in payload["a41_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            continue
        option = row.get("agenda_option_tree", {}) or {}
        commitment = _commitment(row)
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{option.get('selected_option_tier')} | "
            f"{commitment.get('agenda_option_commitment_triggered')} | "
            f"{commitment.get('agenda_option_commitment_reason')} | "
            f"{commitment.get('committed_extra_push_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A41 不新增 teacher，也不把参考轨迹作为在线锚定；它检验的是模型自身在 A40 "
                "中已经发现的 option 是否必须跨闭环决策保持。本轮结果显示，承诺机制可以"
                "完成当前 consume-release 义务，但仍未求解第二关；下一缺口应定位到"
                "release 完成后的阶段化目标、slot-distance/value 表达和可复用函数近似，"
                "而不是继续单点规则叠加。"
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
        "--a40-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json",
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
    parser.add_argument("--agenda-option-commitment-min-tier", type=int, default=7)
    parser.add_argument(
        "--agenda-option-commitment-require-done",
        dest="agenda_option_commitment_require_done",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-agenda-option-commitment-require-done",
        dest="agenda_option_commitment_require_done",
        action="store_false",
    )
    parser.add_argument("--agenda-option-commitment-max-extra-pushes", type=int, default=4)
    parser.add_argument("--agenda-option-commitment-min-total-pushes", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a31_path = ROOT / args.a31_json
    a39_path = ROOT / args.a39_json
    a40_path = ROOT / args.a40_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a31_path, a39_path, a40_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a31_payload = json.loads(a31_path.read_text(encoding="utf-8"))
    a39_payload = json.loads(a39_path.read_text(encoding="utf-8"))
    a40_payload = json.loads(a40_path.read_text(encoding="utf-8"))
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = _A39._A38._A37._A36.load_boxoban_levels(
        ROOT / config["levels"],
        max_levels=level_offset + 1,
    )[level_offset]
    model = _A25._A22._A21._A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(
        level,
        config,
    )
    a41 = run_a41_smoke(
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
        agenda_option_commitment_min_tier=int(args.agenda_option_commitment_min_tier),
        agenda_option_commitment_require_done=bool(
            args.agenda_option_commitment_require_done
        ),
        agenda_option_commitment_max_extra_pushes=int(
            args.agenda_option_commitment_max_extra_pushes
        ),
        agenda_option_commitment_min_total_pushes=int(
            args.agenda_option_commitment_min_total_pushes
        ),
        seed_offset=54179,
        **_common_args(args),
    )
    a31 = a31_payload["a31_closed_loop_smoke"]
    a39 = a39_payload["a39_closed_loop_smoke"]
    a40 = a40_payload["a40_closed_loop_smoke"]
    comparison_a31 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a31",
        a31,
        "a41",
        a41,
    )
    comparison_a39 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a39",
        a39,
        "a41",
        a41,
    )
    comparison_a40 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a40",
        a40,
        "a41",
        a41,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a41_agenda_option_commitment_v1",
            "agenda_gft_alignment": "E14_E6_option_commitment",
            "failure_log": _rel(failure_path),
            "a31_json": _rel(a31_path),
            "a39_json": _rel(a39_path),
            "a40_json": _rel(a40_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a40_planner": _A40.RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner.provider_id,
            "a41_planner": RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
            "trace_scope": "a31_a39_a40_a41_agenda_option_commitment_smoke",
        },
        "config_extract": {
            **a40_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "agenda_option_commitment_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E14_E6_option_commitment",
            "agenda_option_commitment_min_tier": int(
                args.agenda_option_commitment_min_tier
            ),
            "agenda_option_commitment_require_done": bool(
                args.agenda_option_commitment_require_done
            ),
            "agenda_option_commitment_max_extra_pushes": int(
                args.agenda_option_commitment_max_extra_pushes
            ),
            "agenda_option_commitment_min_total_pushes": int(
                args.agenda_option_commitment_min_total_pushes
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
        "a41_closed_loop_smoke": a41,
        "comparison_a31_a41": comparison_a31,
        "comparison_a39_a41": comparison_a39,
        "comparison_a40_a41": comparison_a40,
    }
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(out_md, payload)
    print(json.dumps({"out_json": _rel(out_json), "out_md": _rel(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
