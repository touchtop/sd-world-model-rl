#!/usr/bin/env python3
"""A43 release-blocker option diagnostics for XSokoban first levels."""

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

_A42_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a42_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a42_fill_frontier_option.py",
)
if _A42_SPEC is None or _A42_SPEC.loader is None:
    raise RuntimeError("Unable to load A42 diagnostic helpers.")
_A42 = importlib.util.module_from_spec(_A42_SPEC)
_A42_SPEC.loader.exec_module(_A42)

_A41 = _A42._A41
_A39 = _A42._A39
_A25 = _A42._A25

from src.sokoban import (  # noqa: E402
    RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A42._rel(path)


def _fmt(value: Any) -> str:
    return _A42._fmt(value)


def _release_blocker_option(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("release_blocker_option", {})
    return value if isinstance(value, dict) else {}


def augment_a43_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A42.augment_a42_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [
        _release_blocker_option(row)
        for row in decision_rows
        if _release_blocker_option(row)
    ]
    reason_counts: dict[str, int] = {}
    for row in loaded:
        reason = str(row.get("release_blocker_option_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    first_override_step = None
    for trace_row in decision_rows:
        row = _release_blocker_option(trace_row)
        if bool(row.get("release_blocker_option_override")):
            first_override_step = int(trace_row["step"])
            break
    result["summary"].update(
        {
            "release_blocker_option_loaded_count": int(len(loaded)),
            "release_blocker_option_override_count": int(
                sum(bool(row.get("release_blocker_option_override")) for row in loaded)
            ),
            "release_blocker_option_first_override_step": first_override_step,
            "release_blocker_option_reason_counts": reason_counts,
            "release_blocker_option_selected_progress_count": int(
                sum(
                    bool(
                        row.get("selected_option", {})
                        .get("best_progress", {})
                        .get("agenda_progress", False)
                    )
                    if isinstance(row.get("selected_option", {}), dict)
                    else False
                    for row in loaded
                )
            ),
            "release_blocker_option_selected_done_count": int(
                sum(
                    bool(
                        row.get("selected_option", {})
                        .get("best_progress", {})
                        .get("agenda_done", False)
                    )
                    if isinstance(row.get("selected_option", {}), dict)
                    else False
                    for row in loaded
                )
            ),
            "release_blocker_option_selected_release_open_count": int(
                sum(
                    bool(
                        row.get("selected_option", {})
                        .get("best_progress", {})
                        .get("release_opened", False)
                    )
                    if isinstance(row.get("selected_option", {}), dict)
                    else False
                    for row in loaded
                )
            ),
        }
    )
    return result


def run_a43_smoke(
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
    **kwargs,
) -> dict[str, Any]:
    original = _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A43Planner(RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner):
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
                **planner_kwargs,
            )

    _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A43Planner
    try:
        result = _A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a43_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a42 = payload["a42_closed_loop_smoke"]["summary"]
    a43 = payload["a43_closed_loop_smoke"]["summary"]
    cmp_a42 = payload["comparison_a42_a43"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A43 release-blocker option 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A43 继承 A42，对应专家 27 点中的 agenda-option tree、"
            "frontier obligation 后续 release-blocker option 和 agenda state "
            "persistence。它只在 post-frontier release-blocker 低 tier/off-agenda "
            "状态中评价短程 option，并只选择第一步。"
        ),
        (
            f"A42 为 `{a42['status']}`、pushes={a42['executed_pushes']}、"
            f"decisions={a42['closed_loop_decision_steps']}、goals={a42['boxes_on_goals_final']}；"
            f"A43 为 `{a43['status']}`、pushes={a43['executed_pushes']}、"
            f"decisions={a43['closed_loop_decision_steps']}、goals={a43['boxes_on_goals_final']}。"
        ),
        (
            f"A42/A43 首个决策差异为 `{cmp_a42['a42_a43_first_decision_difference_step']}`；"
            f"A43 release-blocker override `{a43.get('release_blocker_option_override_count', 0)}` 次。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | rb overrides | rb progress |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A42 | {a42['status']} | {a42['closed_loop_decision_steps']} | "
            f"{a42['executed_pushes']} | {a42['boxes_on_goals_final']} | "
            f"{_fmt(a42['final_slot_schedule_cost'])} | {a42['final_dependency_count']} | "
            f"{a42['final_target_blocker_count']} | {a42['final_release_action_count']} | NA | NA |"
        ),
        (
            f"| A43 | {a43['status']} | {a43['closed_loop_decision_steps']} | "
            f"{a43['executed_pushes']} | {a43['boxes_on_goals_final']} | "
            f"{_fmt(a43['final_slot_schedule_cost'])} | {a43['final_dependency_count']} | "
            f"{a43['final_target_blocker_count']} | {a43['final_release_action_count']} | "
            f"{a43.get('release_blocker_option_override_count', 'NA')} | "
            f"{a43.get('release_blocker_option_selected_progress_count', 'NA')} |"
        ),
        "",
        "## A43 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dep | target | release | rb override | reason | option tier | option progress |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in payload["a43_closed_loop_smoke"]["trace"][:90]:
        if "physical_action" not in row:
            continue
        option = _release_blocker_option(row)
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
            f"{option.get('release_blocker_option_override')} | "
            f"{option.get('release_blocker_option_reason')} | "
            f"{progress.get('agenda_tier')} | "
            f"{progress.get('agenda_progress')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A43 检验的是 A42 完成 fill frontier 后，release-blocker "
                "义务是否需要短程 option 评价。该版本不执行 suffix，"
                "因此若仍失败，缺口应继续定位为 option commitment、"
                "agenda state persistence 或更高层里程碑生成，而不是 teacher 在线锚定。"
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
        "--a42-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json",
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
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a42_path = ROOT / args.a42_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a42_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a42_payload = json.loads(a42_path.read_text(encoding="utf-8"))
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
    a43 = run_a43_smoke(
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
        seed_offset=56201,
        **_A41._common_args(args),
    )
    a42 = a42_payload["a42_closed_loop_smoke"]
    comparison_a42 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a42",
        a42,
        "a43",
        a43,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a43_release_blocker_option_v1",
            "agenda_gft_alignment": "E14_E6_release_blocker_option",
            "failure_log": _rel(failure_path),
            "a42_json": _rel(a42_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a42_planner": _A42.RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner.provider_id,
            "a43_planner": RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
            "trace_scope": "a42_a43_release_blocker_option_smoke",
        },
        "config_extract": {
            **a42_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "release_blocker_option_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E14_E6_release_blocker_option",
            "release_blocker_option_max_pushes": int(
                args.release_blocker_option_max_pushes
            ),
            "release_blocker_option_max_nodes": int(
                args.release_blocker_option_max_nodes
            ),
            "release_blocker_option_max_root_candidates": int(
                args.release_blocker_option_max_root_candidates
            ),
            "release_blocker_option_min_tier": int(
                args.release_blocker_option_min_tier
            ),
            "release_blocker_option_min_valid_filled_slots": int(
                args.release_blocker_option_min_valid_filled_slots
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a42_closed_loop_smoke": a42,
        "a43_closed_loop_smoke": a43,
        "comparison_a42_a43": comparison_a42,
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
