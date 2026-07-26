#!/usr/bin/env python3
"""A44 active-blocker obligation diagnostics for XSokoban first levels."""

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

_A43_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a43_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a43_release_blocker_option.py",
)
if _A43_SPEC is None or _A43_SPEC.loader is None:
    raise RuntimeError("Unable to load A43 diagnostic helpers.")
_A43 = importlib.util.module_from_spec(_A43_SPEC)
_A43_SPEC.loader.exec_module(_A43)

_A42 = _A43._A42
_A41 = _A43._A41
_A39 = _A43._A39
_A25 = _A43._A25

from src.sokoban import (  # noqa: E402
    RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A43._rel(path)


def _fmt(value: Any) -> str:
    return _A43._fmt(value)


def _active_blocker_option(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("active_blocker_obligation_option", {})
    return value if isinstance(value, dict) else {}


def augment_a44_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A43.augment_a43_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [
        _active_blocker_option(row)
        for row in decision_rows
        if _active_blocker_option(row)
    ]
    reason_counts: dict[str, int] = {}
    obligation_count_total = 0
    top_true_progress = 0
    top_selected_release = 0
    selected_true_progress = 0
    selected_release_opened = 0
    first_override_step = None
    for trace_row in decision_rows:
        row = _active_blocker_option(trace_row)
        if not row:
            continue
        reason = str(row.get("active_blocker_obligation_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        obligation_count_total += int(row.get("obligation_count", 0))
        selected = row.get("selected_option", {}) or {}
        selected_progress = selected.get("best_progress", {}) or {}
        if bool(selected_progress.get("active_blocker_true_progress", False)):
            selected_true_progress += 1
        if bool(selected_progress.get("selected_release_opened", False)):
            selected_release_opened += 1
        top_options = row.get("top_active_blocker_options", []) or []
        if top_options:
            top_progress = top_options[0].get("best_progress", {}) or {}
            if bool(top_progress.get("active_blocker_true_progress", False)):
                top_true_progress += 1
            if bool(top_progress.get("selected_release_opened", False)):
                top_selected_release += 1
        if (
            first_override_step is None
            and bool(row.get("active_blocker_obligation_override", False))
        ):
            first_override_step = int(trace_row["step"])
    result["summary"].update(
        {
            "active_blocker_obligation_loaded_count": int(len(loaded)),
            "active_blocker_obligation_override_count": int(
                sum(
                    bool(row.get("active_blocker_obligation_override", False))
                    for row in loaded
                )
            ),
            "active_blocker_obligation_first_override_step": first_override_step,
            "active_blocker_obligation_reason_counts": reason_counts,
            "active_blocker_obligation_total_obligation_count": int(
                obligation_count_total
            ),
            "active_blocker_obligation_top_true_progress_count": int(
                top_true_progress
            ),
            "active_blocker_obligation_top_selected_release_count": int(
                top_selected_release
            ),
            "active_blocker_obligation_selected_true_progress_count": int(
                selected_true_progress
            ),
            "active_blocker_obligation_selected_release_opened_count": int(
                selected_release_opened
            ),
        }
    )
    return result


def run_a44_smoke(
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
    fill_frontier_option_min_slot_delta: float = 1.0,
    fill_frontier_option_require_goal_nonregression: bool = True,
    fill_frontier_option_parent_max_tier: int = 4,
    release_blocker_option_max_pushes: int = 5,
    release_blocker_option_max_nodes: int = 96,
    release_blocker_option_max_root_candidates: int = 12,
    release_blocker_option_min_tier: int = 5,
    release_blocker_option_min_tier_improvement: int = 1,
    release_blocker_option_parent_max_tier: int = 2,
    release_blocker_option_min_valid_filled_slots: int = 2,
    active_blocker_obligation_max_pushes: int = 6,
    active_blocker_obligation_max_nodes: int = 128,
    active_blocker_obligation_max_root_candidates: int = 12,
    active_blocker_obligation_max_obligations: int = 4,
    active_blocker_obligation_min_tier: int = 5,
    active_blocker_obligation_min_tier_improvement: int = 1,
    active_blocker_obligation_parent_max_tier: int = 2,
    active_blocker_obligation_min_valid_filled_slots: int = 2,
    **kwargs,
) -> dict[str, Any]:
    original = _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A44Planner(RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner):
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
                fill_frontier_option_min_slot_delta=fill_frontier_option_min_slot_delta,
                fill_frontier_option_require_goal_nonregression=(
                    fill_frontier_option_require_goal_nonregression
                ),
                fill_frontier_option_parent_max_tier=(
                    fill_frontier_option_parent_max_tier
                ),
                release_blocker_option_max_pushes=release_blocker_option_max_pushes,
                release_blocker_option_max_nodes=release_blocker_option_max_nodes,
                release_blocker_option_max_root_candidates=(
                    release_blocker_option_max_root_candidates
                ),
                release_blocker_option_min_tier=release_blocker_option_min_tier,
                release_blocker_option_min_tier_improvement=(
                    release_blocker_option_min_tier_improvement
                ),
                release_blocker_option_parent_max_tier=(
                    release_blocker_option_parent_max_tier
                ),
                release_blocker_option_min_valid_filled_slots=(
                    release_blocker_option_min_valid_filled_slots
                ),
                active_blocker_obligation_max_pushes=(
                    active_blocker_obligation_max_pushes
                ),
                active_blocker_obligation_max_nodes=(
                    active_blocker_obligation_max_nodes
                ),
                active_blocker_obligation_max_root_candidates=(
                    active_blocker_obligation_max_root_candidates
                ),
                active_blocker_obligation_max_obligations=(
                    active_blocker_obligation_max_obligations
                ),
                active_blocker_obligation_min_tier=(
                    active_blocker_obligation_min_tier
                ),
                active_blocker_obligation_min_tier_improvement=(
                    active_blocker_obligation_min_tier_improvement
                ),
                active_blocker_obligation_parent_max_tier=(
                    active_blocker_obligation_parent_max_tier
                ),
                active_blocker_obligation_min_valid_filled_slots=(
                    active_blocker_obligation_min_valid_filled_slots
                ),
                **planner_kwargs,
            )

    _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A44Planner
    try:
        result = _A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a44_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a43 = payload["a43_closed_loop_smoke"]["summary"]
    a44 = payload["a44_closed_loop_smoke"]["summary"]
    cmp_a43 = payload["comparison_a43_a44"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A44 active-blocker obligation 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A44 继承 A43，对应专家 27 点中的 active obligation、"
            "blocker-specific relation、agenda option tree 和 agenda state "
            "persistence。它不再只用 aggregate blocker count，而是把 A11 "
            "schedule dependency row 暴露为具体 active blocker obligation。"
        ),
        (
            f"A43 为 `{a43['status']}`、pushes={a43['executed_pushes']}、"
            f"decisions={a43['closed_loop_decision_steps']}、goals={a43['boxes_on_goals_final']}；"
            f"A44 为 `{a44['status']}`、pushes={a44['executed_pushes']}、"
            f"decisions={a44['closed_loop_decision_steps']}、goals={a44['boxes_on_goals_final']}。"
        ),
        (
            f"A43/A44 首个决策差异为 `{cmp_a43['a43_a44_first_decision_difference_step']}`；"
            f"A44 active-blocker override `{a44.get('active_blocker_obligation_override_count', 0)}` 次，"
            f"top true-progress `{a44.get('active_blocker_obligation_top_true_progress_count', 0)}` 次。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | abo overrides | abo true progress |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A43 | {a43['status']} | {a43['closed_loop_decision_steps']} | "
            f"{a43['executed_pushes']} | {a43['boxes_on_goals_final']} | "
            f"{_fmt(a43['final_slot_schedule_cost'])} | {a43['final_dependency_count']} | "
            f"{a43['final_target_blocker_count']} | {a43['final_release_action_count']} | NA | NA |"
        ),
        (
            f"| A44 | {a44['status']} | {a44['closed_loop_decision_steps']} | "
            f"{a44['executed_pushes']} | {a44['boxes_on_goals_final']} | "
            f"{_fmt(a44['final_slot_schedule_cost'])} | {a44['final_dependency_count']} | "
            f"{a44['final_target_blocker_count']} | {a44['final_release_action_count']} | "
            f"{a44.get('active_blocker_obligation_override_count', 'NA')} | "
            f"{a44.get('active_blocker_obligation_selected_true_progress_count', 'NA')} |"
        ),
        "",
        "## A44 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dep | target | release | abo override | reason | option tier | true progress | progress kinds |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for row in payload["a44_closed_loop_smoke"]["trace"][:90]:
        if "physical_action" not in row:
            continue
        option = _active_blocker_option(row)
        selected = option.get("selected_option", {}) or {}
        progress = selected.get("best_progress", {}) or {}
        totals = row["schedule_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{totals['dependency_count']} | {totals['target_blocker_count']} | "
            f"{totals['release_action_count']} | "
            f"{option.get('active_blocker_obligation_override')} | "
            f"{option.get('active_blocker_obligation_reason')} | "
            f"{progress.get('active_blocker_obligation_tier')} | "
            f"{progress.get('active_blocker_true_progress')} | "
            f"{','.join(progress.get('active_blocker_progress_kinds', []) or [])} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A44 检验的是 A43 负诊断后的下一层原因：当前失败是否来自 "
                "aggregate blocker count 隐去了具体 active blocker identity。"
                "该版本仍不在线调用 teacher，不改 exact transition，也不改 Boxoban 主线。"
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
        "--a43-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json",
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
        "--no-agenda-option-commitment-require-done",
        dest="agenda_option_commitment_require_done",
        action="store_false",
        default=True,
    )
    parser.add_argument("--agenda-option-commitment-max-extra-pushes", type=int, default=4)
    parser.add_argument("--agenda-option-commitment-min-total-pushes", type=int, default=2)
    parser.add_argument("--fill-frontier-option-min-slot-delta", type=float, default=1.0)
    parser.add_argument(
        "--no-fill-frontier-option-require-goal-nonregression",
        dest="fill_frontier_option_require_goal_nonregression",
        action="store_false",
        default=True,
    )
    parser.add_argument("--fill-frontier-option-parent-max-tier", type=int, default=4)
    parser.add_argument("--release-blocker-option-max-pushes", type=int, default=5)
    parser.add_argument("--release-blocker-option-max-nodes", type=int, default=96)
    parser.add_argument("--release-blocker-option-max-root-candidates", type=int, default=12)
    parser.add_argument("--release-blocker-option-min-tier", type=int, default=5)
    parser.add_argument("--release-blocker-option-min-tier-improvement", type=int, default=1)
    parser.add_argument("--release-blocker-option-parent-max-tier", type=int, default=2)
    parser.add_argument("--release-blocker-option-min-valid-filled-slots", type=int, default=2)
    parser.add_argument("--active-blocker-obligation-max-pushes", type=int, default=6)
    parser.add_argument("--active-blocker-obligation-max-nodes", type=int, default=128)
    parser.add_argument("--active-blocker-obligation-max-root-candidates", type=int, default=12)
    parser.add_argument("--active-blocker-obligation-max-obligations", type=int, default=4)
    parser.add_argument("--active-blocker-obligation-min-tier", type=int, default=5)
    parser.add_argument("--active-blocker-obligation-min-tier-improvement", type=int, default=1)
    parser.add_argument("--active-blocker-obligation-parent-max-tier", type=int, default=2)
    parser.add_argument("--active-blocker-obligation-min-valid-filled-slots", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a43_path = ROOT / args.a43_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a43_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a43_payload = json.loads(a43_path.read_text(encoding="utf-8"))
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
    a44 = run_a44_smoke(
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
        fill_frontier_option_min_slot_delta=float(
            args.fill_frontier_option_min_slot_delta
        ),
        fill_frontier_option_require_goal_nonregression=bool(
            args.fill_frontier_option_require_goal_nonregression
        ),
        fill_frontier_option_parent_max_tier=int(
            args.fill_frontier_option_parent_max_tier
        ),
        release_blocker_option_max_pushes=int(args.release_blocker_option_max_pushes),
        release_blocker_option_max_nodes=int(args.release_blocker_option_max_nodes),
        release_blocker_option_max_root_candidates=int(
            args.release_blocker_option_max_root_candidates
        ),
        release_blocker_option_min_tier=int(args.release_blocker_option_min_tier),
        release_blocker_option_min_tier_improvement=int(
            args.release_blocker_option_min_tier_improvement
        ),
        release_blocker_option_parent_max_tier=int(
            args.release_blocker_option_parent_max_tier
        ),
        release_blocker_option_min_valid_filled_slots=int(
            args.release_blocker_option_min_valid_filled_slots
        ),
        active_blocker_obligation_max_pushes=int(
            args.active_blocker_obligation_max_pushes
        ),
        active_blocker_obligation_max_nodes=int(
            args.active_blocker_obligation_max_nodes
        ),
        active_blocker_obligation_max_root_candidates=int(
            args.active_blocker_obligation_max_root_candidates
        ),
        active_blocker_obligation_max_obligations=int(
            args.active_blocker_obligation_max_obligations
        ),
        active_blocker_obligation_min_tier=int(
            args.active_blocker_obligation_min_tier
        ),
        active_blocker_obligation_min_tier_improvement=int(
            args.active_blocker_obligation_min_tier_improvement
        ),
        active_blocker_obligation_parent_max_tier=int(
            args.active_blocker_obligation_parent_max_tier
        ),
        active_blocker_obligation_min_valid_filled_slots=int(
            args.active_blocker_obligation_min_valid_filled_slots
        ),
        seed_offset=57201,
        **_A41._common_args(args),
    )
    a43 = a43_payload["a43_closed_loop_smoke"]
    comparison_a43 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a43",
        a43,
        "a44",
        a44,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a44_active_blocker_obligation_v1",
            "agenda_gft_alignment": "E6_E9_E14_active_blocker_obligation",
            "failure_log": _rel(failure_path),
            "a43_json": _rel(a43_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a43_planner": _A43.RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner.provider_id,
            "a44_planner": RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
            "trace_scope": "a43_a44_active_blocker_obligation_smoke",
        },
        "config_extract": {
            **a43_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "active_blocker_obligation_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E6_E9_E14_active_blocker_obligation",
            "active_blocker_obligation_max_pushes": int(
                args.active_blocker_obligation_max_pushes
            ),
            "active_blocker_obligation_max_nodes": int(
                args.active_blocker_obligation_max_nodes
            ),
            "active_blocker_obligation_max_root_candidates": int(
                args.active_blocker_obligation_max_root_candidates
            ),
            "active_blocker_obligation_max_obligations": int(
                args.active_blocker_obligation_max_obligations
            ),
            "active_blocker_obligation_min_tier": int(
                args.active_blocker_obligation_min_tier
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a43_closed_loop_smoke": a43,
        "a44_closed_loop_smoke": a44,
        "comparison_a43_a44": comparison_a43,
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
    print(
        json.dumps(
            {"out_json": _rel(out_json), "out_md": _rel(out_md)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
