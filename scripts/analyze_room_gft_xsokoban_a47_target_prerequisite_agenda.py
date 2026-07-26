#!/usr/bin/env python3
"""A47 target-prerequisite agenda diagnostics for XSokoban first levels."""

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

_A46_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a46_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a46_residual_active_agenda.py",
)
if _A46_SPEC is None or _A46_SPEC.loader is None:
    raise RuntimeError("Unable to load A46 diagnostic helpers.")
_A46 = importlib.util.module_from_spec(_A46_SPEC)
_A46_SPEC.loader.exec_module(_A46)

_A45 = _A46._A45
_A41 = _A46._A41
_A39 = _A46._A39
_A25 = _A46._A25

from src.sokoban import (  # noqa: E402
    RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner,
    RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner,
    RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A46._rel(path)


def _fmt(value: Any) -> str:
    return _A46._fmt(value)


def _prerequisite(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("target_prerequisite_agenda", {})
    return value if isinstance(value, dict) else {}


def augment_a47_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A46.augment_a46_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_prerequisite(row) for row in decision_rows if _prerequisite(row)]
    triggered = [
        row
        for row in loaded
        if bool(row.get("target_prerequisite_agenda_triggered"))
    ]
    reason_counts: dict[str, int] = {}
    obligation_count = 0
    parent_assigned_count = 0
    legal_push_count_total = 0
    for row in loaded:
        reason = str(row.get("target_prerequisite_agenda_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        obligation = row.get("target_prerequisite_obligation", {})
        if isinstance(obligation, dict) and obligation:
            obligation_count += 1
            parent_assigned_count += int(
                bool(
                    obligation.get(
                        "required_stand_blocker_is_parent_assigned_box",
                        False,
                    )
                )
            )
            legal_push_count_total += int(
                obligation.get("target_blocker_legal_push_count", 0) or 0
            )
    first_obligation_step = None
    first_trigger_step = None
    for trace_row in decision_rows:
        row = _prerequisite(trace_row)
        obligation = row.get("target_prerequisite_obligation", {})
        if first_obligation_step is None and isinstance(obligation, dict) and obligation:
            first_obligation_step = int(trace_row["step"])
        if bool(row.get("target_prerequisite_agenda_triggered")):
            first_trigger_step = int(trace_row["step"])
            break
    result["summary"].update(
        {
            "target_prerequisite_agenda_loaded_count": int(len(loaded)),
            "target_prerequisite_agenda_triggered_count": int(len(triggered)),
            "target_prerequisite_agenda_reason_counts": reason_counts,
            "target_prerequisite_agenda_first_obligation_step": first_obligation_step,
            "target_prerequisite_agenda_first_trigger_step": first_trigger_step,
            "target_prerequisite_agenda_obligation_count": int(obligation_count),
            "target_prerequisite_parent_assigned_blocker_count": int(
                parent_assigned_count
            ),
            "target_prerequisite_target_legal_push_count_total": int(
                legal_push_count_total
            ),
            "target_prerequisite_agenda_committed_push_count": int(
                sum(
                    int(row.get("committed_prerequisite_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "target_prerequisite_agenda_evaluated_option_count": int(
                sum(int(row.get("evaluated_option_count", 0) or 0) for row in loaded)
            ),
            "target_prerequisite_agenda_true_progress_count": int(
                sum(
                    bool(
                        row.get("final_progress", {}).get(
                            "target_prerequisite_true_progress",
                            False,
                        )
                    )
                    if isinstance(row.get("final_progress", {}), dict)
                    else False
                    for row in triggered
                )
            ),
            "target_prerequisite_agenda_max_expanded_nodes": None
            if not loaded
            else int(max(int(row.get("expanded_nodes", 0) or 0) for row in loaded)),
        }
    )
    return result


def run_a47_smoke(
    model,
    config: dict[str, Any],
    *,
    residual_active_agenda_max_pushes: int = 8,
    residual_active_agenda_max_nodes: int = 192,
    residual_active_agenda_min_total_pushes: int = 1,
    residual_active_agenda_require_target_progress: bool = True,
    target_prerequisite_agenda_max_pushes: int = 8,
    target_prerequisite_agenda_max_nodes: int = 256,
    target_prerequisite_agenda_min_total_pushes: int = 1,
    target_prerequisite_agenda_require_progress: bool = True,
    **kwargs,
) -> dict[str, Any]:
    original = _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner

    class ConfiguredA47Planner(
        RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner
    ):
        def __init__(self, *args, **planner_kwargs):
            super().__init__(
                *args,
                residual_active_agenda_max_pushes=residual_active_agenda_max_pushes,
                residual_active_agenda_max_nodes=residual_active_agenda_max_nodes,
                residual_active_agenda_min_total_pushes=(
                    residual_active_agenda_min_total_pushes
                ),
                residual_active_agenda_require_target_progress=(
                    residual_active_agenda_require_target_progress
                ),
                target_prerequisite_agenda_max_pushes=(
                    target_prerequisite_agenda_max_pushes
                ),
                target_prerequisite_agenda_max_nodes=(
                    target_prerequisite_agenda_max_nodes
                ),
                target_prerequisite_agenda_min_total_pushes=(
                    target_prerequisite_agenda_min_total_pushes
                ),
                target_prerequisite_agenda_require_progress=(
                    target_prerequisite_agenda_require_progress
                ),
                **planner_kwargs,
            )

    _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner = (
        ConfiguredA47Planner
    )
    try:
        result = _A45.run_a45_smoke(model, config, **kwargs)
    finally:
        _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner = original
    return augment_a47_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a46 = payload["a46_closed_loop_smoke"]["summary"]
    a47 = payload["a47_closed_loop_smoke"]["summary"]
    cmp_a46 = payload["comparison_a46_a47"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A47 target-prerequisite 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A47 继承 A46，对应专家 27 点中的 obligation frontier、blocker set、"
            "active agenda state 与 agenda-option tree。A46 若在 target-only residual "
            "agenda 中找不到 target progress，A47 会进一步检查 target blocker "
            "自身是否因 push stand cell 被箱子占用而不可执行，并形成 "
            "`ClearTargetPushStand` 前提义务。"
        ),
        (
            f"A46 为 `{a46['status']}`、pushes={a46['executed_pushes']}、"
            f"decisions={a46['closed_loop_decision_steps']}、goals={a46['boxes_on_goals_final']}；"
            f"A47 为 `{a47['status']}`、pushes={a47['executed_pushes']}、"
            f"decisions={a47['closed_loop_decision_steps']}、goals={a47['boxes_on_goals_final']}。"
        ),
        (
            f"A46/A47 首个决策差异为 `{cmp_a46['a46_a47_first_decision_difference_step']}`；"
            f"A47 构造 prerequisite obligation `{a47.get('target_prerequisite_agenda_obligation_count', 0)}` 次，"
            f"触发 `{a47.get('target_prerequisite_agenda_triggered_count', 0)}` 次，"
            f"追加 `{a47.get('target_prerequisite_agenda_committed_push_count', 0)}` 个 prerequisite push。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | residual triggers | prereq obligations | prereq triggers | prereq pushes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A46 | {a46['status']} | {a46['closed_loop_decision_steps']} | "
            f"{a46['executed_pushes']} | {a46['boxes_on_goals_final']} | "
            f"{_fmt(a46['final_slot_schedule_cost'])} | {a46['final_dependency_count']} | "
            f"{a46['final_target_blocker_count']} | "
            f"{a46.get('residual_active_agenda_triggered_count', 'NA')} | NA | NA | NA |"
        ),
        (
            f"| A47 | {a47['status']} | {a47['closed_loop_decision_steps']} | "
            f"{a47['executed_pushes']} | {a47['boxes_on_goals_final']} | "
            f"{_fmt(a47['final_slot_schedule_cost'])} | {a47['final_dependency_count']} | "
            f"{a47['final_target_blocker_count']} | "
            f"{a47.get('residual_active_agenda_triggered_count', 'NA')} | "
            f"{a47.get('target_prerequisite_agenda_obligation_count', 'NA')} | "
            f"{a47.get('target_prerequisite_agenda_triggered_count', 'NA')} | "
            f"{a47.get('target_prerequisite_agenda_committed_push_count', 'NA')} |"
        ),
        "",
        "## A47 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dep | target | residual reason | prereq reason | prereq pushes | prereq best tier |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in payload["a47_closed_loop_smoke"]["trace"][:90]:
        if "physical_action" not in row:
            continue
        residual = row.get("residual_active_agenda", {}) or {}
        prereq = _prerequisite(row)
        best_progress = prereq.get("best_option_progress", {}) or {}
        totals = row["schedule_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{totals['dependency_count']} | {totals['target_blocker_count']} | "
            f"{residual.get('residual_active_agenda_reason')} | "
            f"{prereq.get('target_prerequisite_agenda_reason')} | "
            f"{prereq.get('committed_prerequisite_push_count', 0)} | "
            f"{best_progress.get('target_prerequisite_agenda_tier')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A47 检验的是 residual target blocker 的下一层可执行前提。"
                "该版本不在线调用 teacher，不引入参考轨迹锚定，不改变 exact "
                "transition，也不改 Boxoban 主线。"
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
        "--a46-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--multihead-sd-consumer-checkpoint",
        default="results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json",
    )
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=80)
    parser.add_argument("--residual-active-agenda-max-pushes", type=int, default=8)
    parser.add_argument("--residual-active-agenda-max-nodes", type=int, default=192)
    parser.add_argument("--residual-active-agenda-min-total-pushes", type=int, default=1)
    parser.add_argument("--target-prerequisite-agenda-max-pushes", type=int, default=8)
    parser.add_argument("--target-prerequisite-agenda-max-nodes", type=int, default=256)
    parser.add_argument(
        "--target-prerequisite-agenda-min-total-pushes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no-target-prerequisite-agenda-require-progress",
        dest="target_prerequisite_agenda_require_progress",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a46_path = ROOT / args.a46_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a46_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a46_payload = json.loads(a46_path.read_text(encoding="utf-8"))
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
    a47 = run_a47_smoke(
        model,
        config,
        multihead_sd_consumer_checkpoint=str(checkpoint_path),
        residual_active_agenda_max_pushes=int(
            args.residual_active_agenda_max_pushes
        ),
        residual_active_agenda_max_nodes=int(args.residual_active_agenda_max_nodes),
        residual_active_agenda_min_total_pushes=int(
            args.residual_active_agenda_min_total_pushes
        ),
        target_prerequisite_agenda_max_pushes=int(
            args.target_prerequisite_agenda_max_pushes
        ),
        target_prerequisite_agenda_max_nodes=int(
            args.target_prerequisite_agenda_max_nodes
        ),
        target_prerequisite_agenda_min_total_pushes=int(
            args.target_prerequisite_agenda_min_total_pushes
        ),
        target_prerequisite_agenda_require_progress=bool(
            args.target_prerequisite_agenda_require_progress
        ),
        seed_offset=58465,
        **_A41._common_args(args),
    )
    a46 = a46_payload["a46_closed_loop_smoke"]
    comparison_a46 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a46",
        a46,
        "a47",
        a47,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a47_target_prerequisite_agenda_v1",
            "agenda_gft_alignment": "E3_E6_E11_E14_target_prerequisite_agenda",
            "failure_log": _rel(failure_path),
            "a46_json": _rel(a46_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a45_planner": RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner.provider_id,
            "a46_planner": RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner.provider_id,
            "a47_planner": RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": False,
            "trace_scope": "a46_a47_target_prerequisite_agenda_smoke",
        },
        "config_extract": {
            **a46_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "target_prerequisite_agenda_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E3_E6_E11_E14_target_prerequisite_agenda",
            "residual_active_agenda_max_pushes": int(
                args.residual_active_agenda_max_pushes
            ),
            "residual_active_agenda_max_nodes": int(
                args.residual_active_agenda_max_nodes
            ),
            "target_prerequisite_agenda_max_pushes": int(
                args.target_prerequisite_agenda_max_pushes
            ),
            "target_prerequisite_agenda_max_nodes": int(
                args.target_prerequisite_agenda_max_nodes
            ),
            "target_prerequisite_agenda_min_total_pushes": int(
                args.target_prerequisite_agenda_min_total_pushes
            ),
            "target_prerequisite_agenda_require_progress": bool(
                args.target_prerequisite_agenda_require_progress
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a46_closed_loop_smoke": a46,
        "a47_closed_loop_smoke": a47,
        "comparison_a46_a47": comparison_a46,
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
