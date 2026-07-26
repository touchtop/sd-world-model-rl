#!/usr/bin/env python3
"""A45 active-blocker commitment diagnostics for XSokoban first levels."""

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

_A44_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a44_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a44_active_blocker_obligation.py",
)
if _A44_SPEC is None or _A44_SPEC.loader is None:
    raise RuntimeError("Unable to load A44 diagnostic helpers.")
_A44 = importlib.util.module_from_spec(_A44_SPEC)
_A44_SPEC.loader.exec_module(_A44)

_A43 = _A44._A43
_A41 = _A44._A41
_A39 = _A44._A39
_A25 = _A44._A25

from src.sokoban import (  # noqa: E402
    RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A44._rel(path)


def _fmt(value: Any) -> str:
    return _A44._fmt(value)


def _commitment(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("active_blocker_commitment", {})
    return value if isinstance(value, dict) else {}


def augment_a45_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A44.augment_a44_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_commitment(row) for row in decision_rows if _commitment(row)]
    triggered = [
        row for row in loaded if bool(row.get("active_blocker_commitment_triggered"))
    ]
    reason_counts: dict[str, int] = {}
    for row in loaded:
        reason = str(row.get("active_blocker_commitment_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    first_trigger_step = None
    for trace_row in decision_rows:
        row = _commitment(trace_row)
        if bool(row.get("active_blocker_commitment_triggered")):
            first_trigger_step = int(trace_row["step"])
            break
    result["summary"].update(
        {
            "active_blocker_commitment_loaded_count": int(len(loaded)),
            "active_blocker_commitment_triggered_count": int(len(triggered)),
            "active_blocker_commitment_reason_counts": reason_counts,
            "active_blocker_commitment_first_trigger_step": first_trigger_step,
            "active_blocker_commitment_extra_push_count": int(
                sum(
                    int(row.get("committed_extra_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "active_blocker_commitment_validated_push_count": int(
                sum(
                    int(row.get("validated_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "active_blocker_commitment_true_progress_count": int(
                sum(
                    bool(
                        row.get("final_progress", {}).get(
                            "active_blocker_true_progress",
                            False,
                        )
                    )
                    if isinstance(row.get("final_progress", {}), dict)
                    else False
                    for row in triggered
                )
            ),
            "active_blocker_commitment_min_final_tier": None
            if not triggered
            else int(
                min(
                    int(
                        row.get("final_progress", {}).get(
                            "active_blocker_obligation_tier",
                            0,
                        )
                    )
                    for row in triggered
                    if isinstance(row.get("final_progress", {}), dict)
                )
            ),
        }
    )
    return result


def run_a45_smoke(
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
    active_blocker_commitment_min_tier: int = 8,
    active_blocker_commitment_require_true_progress: bool = True,
    active_blocker_commitment_max_extra_pushes: int = 5,
    active_blocker_commitment_min_total_pushes: int = 2,
    **kwargs,
) -> dict[str, Any]:
    original = _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner

    class A45Planner(RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner):
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
                active_blocker_commitment_min_tier=(
                    active_blocker_commitment_min_tier
                ),
                active_blocker_commitment_require_true_progress=(
                    active_blocker_commitment_require_true_progress
                ),
                active_blocker_commitment_max_extra_pushes=(
                    active_blocker_commitment_max_extra_pushes
                ),
                active_blocker_commitment_min_total_pushes=(
                    active_blocker_commitment_min_total_pushes
                ),
                **planner_kwargs,
            )

    _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = A45Planner
    try:
        result = _A25.run_a25_smoke(model, config, **kwargs)
    finally:
        _A25.RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner = original
    return augment_a45_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a44 = payload["a44_closed_loop_smoke"]["summary"]
    a45 = payload["a45_closed_loop_smoke"]["summary"]
    cmp_a44 = payload["comparison_a44_a45"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A45 active-blocker commitment 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A45 继承 A44，对应专家 27 点中的 active obligation、agenda option "
            "tree 与 agenda state persistence。A44 负责识别具体 blocker "
            "obligation 的高 tier 后缀，A45 只在后缀经 exact transition "
            "重放验证后，把该后缀作为 bounded commitment 执行。"
        ),
        (
            f"A44 为 `{a44['status']}`、pushes={a44['executed_pushes']}、"
            f"decisions={a44['closed_loop_decision_steps']}、goals={a44['boxes_on_goals_final']}；"
            f"A45 为 `{a45['status']}`、pushes={a45['executed_pushes']}、"
            f"decisions={a45['closed_loop_decision_steps']}、goals={a45['boxes_on_goals_final']}。"
        ),
        (
            f"A44/A45 首个决策差异为 `{cmp_a44['a44_a45_first_decision_difference_step']}`；"
            f"A45 commitment 触发 `{a45.get('active_blocker_commitment_triggered_count', 0)}` 次，"
            f"额外执行 `{a45.get('active_blocker_commitment_extra_push_count', 0)}` 个承诺 push。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | abo true progress | abc triggers | abc extra pushes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A44 | {a44['status']} | {a44['closed_loop_decision_steps']} | "
            f"{a44['executed_pushes']} | {a44['boxes_on_goals_final']} | "
            f"{_fmt(a44['final_slot_schedule_cost'])} | {a44['final_dependency_count']} | "
            f"{a44['final_target_blocker_count']} | {a44['final_release_action_count']} | "
            f"{a44.get('active_blocker_obligation_selected_true_progress_count', 'NA')} | NA | NA |"
        ),
        (
            f"| A45 | {a45['status']} | {a45['closed_loop_decision_steps']} | "
            f"{a45['executed_pushes']} | {a45['boxes_on_goals_final']} | "
            f"{_fmt(a45['final_slot_schedule_cost'])} | {a45['final_dependency_count']} | "
            f"{a45['final_target_blocker_count']} | {a45['final_release_action_count']} | "
            f"{a45.get('active_blocker_obligation_selected_true_progress_count', 'NA')} | "
            f"{a45.get('active_blocker_commitment_triggered_count', 'NA')} | "
            f"{a45.get('active_blocker_commitment_extra_push_count', 'NA')} |"
        ),
        "",
        "## A45 轨迹前缀",
        "",
        "| step | action | pushes | goals | stage | slot cost | dep | target | release | abo reason | abo tier | abc | abc reason | abc extra | final tier | progress kinds |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|",
    ]
    for row in payload["a45_closed_loop_smoke"]["trace"][:90]:
        if "physical_action" not in row:
            continue
        option = _A44._active_blocker_option(row)
        selected = option.get("selected_option", {}) or {}
        option_progress = selected.get("best_progress", {}) or {}
        commitment = _commitment(row)
        final_progress = commitment.get("final_progress", {}) or {}
        totals = row["schedule_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{row.get('chosen_stage', {}).get('before', {}).get('stage', 'NA')} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{totals['dependency_count']} | {totals['target_blocker_count']} | "
            f"{totals['release_action_count']} | "
            f"{option.get('active_blocker_obligation_reason')} | "
            f"{option_progress.get('active_blocker_obligation_tier')} | "
            f"{commitment.get('active_blocker_commitment_triggered')} | "
            f"{commitment.get('active_blocker_commitment_reason')} | "
            f"{commitment.get('committed_extra_push_count', 0)} | "
            f"{final_progress.get('active_blocker_obligation_tier')} | "
            f"{','.join(final_progress.get('active_blocker_progress_kinds', []) or [])} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A45 检验的是 A44 暴露出的消费端缺口：具体 blocker-specific "
                "后缀已经可见，但闭环若每步只执行第一个 root edge，可能无法保持同一"
                "义务的连续执行。该版本仍不在线调用 teacher，不引入参考轨迹锚定，"
                "不改 exact transition，也不改 Boxoban 主线。"
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
        "--a44-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json",
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
    parser.add_argument("--active-blocker-commitment-min-tier", type=int, default=8)
    parser.add_argument(
        "--no-active-blocker-commitment-require-true-progress",
        dest="active_blocker_commitment_require_true_progress",
        action="store_false",
        default=True,
    )
    parser.add_argument("--active-blocker-commitment-max-extra-pushes", type=int, default=5)
    parser.add_argument("--active-blocker-commitment-min-total-pushes", type=int, default=2)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a44_path = ROOT / args.a44_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a44_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a44_payload = json.loads(a44_path.read_text(encoding="utf-8"))
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
    a45 = run_a45_smoke(
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
        active_blocker_commitment_min_tier=int(
            args.active_blocker_commitment_min_tier
        ),
        active_blocker_commitment_require_true_progress=bool(
            args.active_blocker_commitment_require_true_progress
        ),
        active_blocker_commitment_max_extra_pushes=int(
            args.active_blocker_commitment_max_extra_pushes
        ),
        active_blocker_commitment_min_total_pushes=int(
            args.active_blocker_commitment_min_total_pushes
        ),
        seed_offset=58463,
        **_A41._common_args(args),
    )
    a44 = a44_payload["a44_closed_loop_smoke"]
    comparison_a44 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a44",
        a44,
        "a45",
        a45,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a45_active_blocker_commitment_v1",
            "agenda_gft_alignment": "E6_E9_E14_active_blocker_commitment",
            "failure_log": _rel(failure_path),
            "a44_json": _rel(a44_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a44_planner": _A44.RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner.provider_id,
            "a45_planner": RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": bool(
                args.learned_sd_consumer_root_override
            ),
            "trace_scope": "a44_a45_active_blocker_commitment_smoke",
        },
        "config_extract": {
            **a44_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "active_blocker_commitment_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E6_E9_E14_active_blocker_commitment",
            "active_blocker_commitment_min_tier": int(
                args.active_blocker_commitment_min_tier
            ),
            "active_blocker_commitment_require_true_progress": bool(
                args.active_blocker_commitment_require_true_progress
            ),
            "active_blocker_commitment_max_extra_pushes": int(
                args.active_blocker_commitment_max_extra_pushes
            ),
            "active_blocker_commitment_min_total_pushes": int(
                args.active_blocker_commitment_min_total_pushes
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a44_closed_loop_smoke": a44,
        "a45_closed_loop_smoke": a45,
        "comparison_a44_a45": comparison_a44,
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
