#!/usr/bin/env python3
"""A48 assignment-conflict agenda diagnostics for XSokoban first levels."""

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

_A47_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a47_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a47_target_prerequisite_agenda.py",
)
if _A47_SPEC is None or _A47_SPEC.loader is None:
    raise RuntimeError("Unable to load A47 diagnostic helpers.")
_A47 = importlib.util.module_from_spec(_A47_SPEC)
_A47_SPEC.loader.exec_module(_A47)

_A46 = _A47._A46
_A45 = _A47._A45
_A41 = _A47._A41
_A39 = _A47._A39
_A25 = _A47._A25

from src.sokoban import (  # noqa: E402
    RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner,
    RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A47._rel(path)


def _fmt(value: Any) -> str:
    return _A47._fmt(value)


def _assignment(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("assignment_conflict_agenda", {})
    return value if isinstance(value, dict) else {}


def augment_a48_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A47.augment_a47_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_assignment(row) for row in decision_rows if _assignment(row)]
    triggered = [
        row
        for row in loaded
        if bool(row.get("assignment_conflict_agenda_triggered"))
    ]
    reason_counts: dict[str, int] = {}
    obligation_count = 0
    assigned_push_total = 0
    assigned_toward_slot_total = 0
    for row in loaded:
        reason = str(row.get("assignment_conflict_agenda_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        obligation = row.get("assignment_conflict_obligation", {})
        if isinstance(obligation, dict) and obligation:
            obligation_count += 1
            assigned_push_total += int(
                obligation.get("assigned_box_legal_push_count", 0) or 0
            )
            assigned_toward_slot_total += int(
                obligation.get("assigned_box_toward_slot_push_count", 0) or 0
            )
    first_obligation_step = None
    first_trigger_step = None
    for trace_row in decision_rows:
        row = _assignment(trace_row)
        obligation = row.get("assignment_conflict_obligation", {})
        if first_obligation_step is None and isinstance(obligation, dict) and obligation:
            first_obligation_step = int(trace_row["step"])
        if bool(row.get("assignment_conflict_agenda_triggered")):
            first_trigger_step = int(trace_row["step"])
            break
    result["summary"].update(
        {
            "assignment_conflict_agenda_loaded_count": int(len(loaded)),
            "assignment_conflict_agenda_triggered_count": int(len(triggered)),
            "assignment_conflict_agenda_reason_counts": reason_counts,
            "assignment_conflict_agenda_first_obligation_step": first_obligation_step,
            "assignment_conflict_agenda_first_trigger_step": first_trigger_step,
            "assignment_conflict_agenda_obligation_count": int(obligation_count),
            "assignment_conflict_assigned_legal_push_count_total": int(
                assigned_push_total
            ),
            "assignment_conflict_assigned_toward_slot_push_count_total": int(
                assigned_toward_slot_total
            ),
            "assignment_conflict_agenda_committed_push_count": int(
                sum(
                    int(row.get("committed_conflict_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "assignment_conflict_agenda_evaluated_option_count": int(
                sum(int(row.get("evaluated_option_count", 0) or 0) for row in loaded)
            ),
            "assignment_conflict_agenda_done_count": int(
                sum(
                    bool(
                        row.get("final_progress", {}).get(
                            "assignment_conflict_done",
                            False,
                        )
                    )
                    if isinstance(row.get("final_progress", {}), dict)
                    else False
                    for row in triggered
                )
            ),
            "assignment_conflict_agenda_max_expanded_nodes": None
            if not loaded
            else int(max(int(row.get("expanded_nodes", 0) or 0) for row in loaded)),
        }
    )
    return result


def run_a48_smoke(
    model,
    config: dict[str, Any],
    *,
    residual_active_agenda_max_pushes: int = 8,
    residual_active_agenda_max_nodes: int = 192,
    target_prerequisite_agenda_max_pushes: int = 8,
    target_prerequisite_agenda_max_nodes: int = 256,
    assignment_conflict_agenda_max_pushes: int = 8,
    assignment_conflict_agenda_max_nodes: int = 320,
    assignment_conflict_agenda_min_total_pushes: int = 1,
    assignment_conflict_agenda_require_done: bool = True,
    **kwargs,
) -> dict[str, Any]:
    original = _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner

    class ConfiguredA48Planner(
        RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner
    ):
        def __init__(self, *args, **planner_kwargs):
            super().__init__(
                *args,
                residual_active_agenda_max_pushes=residual_active_agenda_max_pushes,
                residual_active_agenda_max_nodes=residual_active_agenda_max_nodes,
                target_prerequisite_agenda_max_pushes=(
                    target_prerequisite_agenda_max_pushes
                ),
                target_prerequisite_agenda_max_nodes=(
                    target_prerequisite_agenda_max_nodes
                ),
                assignment_conflict_agenda_max_pushes=(
                    assignment_conflict_agenda_max_pushes
                ),
                assignment_conflict_agenda_max_nodes=(
                    assignment_conflict_agenda_max_nodes
                ),
                assignment_conflict_agenda_min_total_pushes=(
                    assignment_conflict_agenda_min_total_pushes
                ),
                assignment_conflict_agenda_require_done=(
                    assignment_conflict_agenda_require_done
                ),
                **planner_kwargs,
            )

    _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner = (
        ConfiguredA48Planner
    )
    try:
        result = _A45.run_a45_smoke(model, config, **kwargs)
    finally:
        _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner = original
    return augment_a48_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a47 = payload["a47_closed_loop_smoke"]["summary"]
    a48 = payload["a48_closed_loop_smoke"]["summary"]
    cmp_a47 = payload["comparison_a47_a48"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A48 assignment-conflict 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A48 继承 A47，对应专家 27 点中的 obligation frontier、blocker set、"
            "active agenda state 与 agenda-option tree。A47 若发现 target blocker "
            "所需站位被 parent assigned box 占用，A48 将其记录为 assignment/prerequisite "
            "冲突义务，而不是坐标级规则。"
        ),
        (
            f"A47 为 `{a47['status']}`、pushes={a47['executed_pushes']}、"
            f"decisions={a47['closed_loop_decision_steps']}、goals={a47['boxes_on_goals_final']}；"
            f"A48 为 `{a48['status']}`、pushes={a48['executed_pushes']}、"
            f"decisions={a48['closed_loop_decision_steps']}、goals={a48['boxes_on_goals_final']}。"
        ),
        (
            f"A47/A48 首个决策差异为 `{cmp_a47['a47_a48_first_decision_difference_step']}`；"
            f"A48 构造 assignment-conflict obligation `{a48.get('assignment_conflict_agenda_obligation_count', 0)}` 次，"
            f"触发 `{a48.get('assignment_conflict_agenda_triggered_count', 0)}` 次，"
            f"追加 `{a48.get('assignment_conflict_agenda_committed_push_count', 0)}` 个 conflict push。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | prereq obligations | conflict obligations | conflict triggers | conflict pushes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A47 | {a47['status']} | {a47['closed_loop_decision_steps']} | "
            f"{a47['executed_pushes']} | {a47['boxes_on_goals_final']} | "
            f"{_fmt(a47['final_slot_schedule_cost'])} | {a47['final_dependency_count']} | "
            f"{a47['final_target_blocker_count']} | "
            f"{a47.get('target_prerequisite_agenda_obligation_count', 'NA')} | NA | NA | NA |"
        ),
        (
            f"| A48 | {a48['status']} | {a48['closed_loop_decision_steps']} | "
            f"{a48['executed_pushes']} | {a48['boxes_on_goals_final']} | "
            f"{_fmt(a48['final_slot_schedule_cost'])} | {a48['final_dependency_count']} | "
            f"{a48['final_target_blocker_count']} | "
            f"{a48.get('target_prerequisite_agenda_obligation_count', 'NA')} | "
            f"{a48.get('assignment_conflict_agenda_obligation_count', 'NA')} | "
            f"{a48.get('assignment_conflict_agenda_triggered_count', 'NA')} | "
            f"{a48.get('assignment_conflict_agenda_committed_push_count', 'NA')} |"
        ),
        "",
        "## A48 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dep | target | prereq reason | conflict reason | conflict pushes | conflict best tier |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in payload["a48_closed_loop_smoke"]["trace"][:90]:
        if "physical_action" not in row:
            continue
        prereq = row.get("target_prerequisite_agenda", {}) or {}
        conflict = _assignment(row)
        best_progress = conflict.get("best_option_progress", {}) or {}
        totals = row["schedule_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{totals['dependency_count']} | {totals['target_blocker_count']} | "
            f"{prereq.get('target_prerequisite_agenda_reason')} | "
            f"{conflict.get('assignment_conflict_agenda_reason')} | "
            f"{conflict.get('committed_conflict_push_count', 0)} | "
            f"{best_progress.get('assignment_conflict_agenda_tier')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A48 检验的是 target prerequisite 与当前 slot assignment 之间的冲突。"
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
        "--a47-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json",
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
    parser.add_argument("--target-prerequisite-agenda-max-pushes", type=int, default=8)
    parser.add_argument("--target-prerequisite-agenda-max-nodes", type=int, default=256)
    parser.add_argument("--assignment-conflict-agenda-max-pushes", type=int, default=8)
    parser.add_argument("--assignment-conflict-agenda-max-nodes", type=int, default=320)
    parser.add_argument(
        "--assignment-conflict-agenda-min-total-pushes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no-assignment-conflict-agenda-require-done",
        dest="assignment_conflict_agenda_require_done",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a47_path = ROOT / args.a47_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a47_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a47_payload = json.loads(a47_path.read_text(encoding="utf-8"))
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
    a48 = run_a48_smoke(
        model,
        config,
        multihead_sd_consumer_checkpoint=str(checkpoint_path),
        residual_active_agenda_max_pushes=int(
            args.residual_active_agenda_max_pushes
        ),
        residual_active_agenda_max_nodes=int(args.residual_active_agenda_max_nodes),
        target_prerequisite_agenda_max_pushes=int(
            args.target_prerequisite_agenda_max_pushes
        ),
        target_prerequisite_agenda_max_nodes=int(
            args.target_prerequisite_agenda_max_nodes
        ),
        assignment_conflict_agenda_max_pushes=int(
            args.assignment_conflict_agenda_max_pushes
        ),
        assignment_conflict_agenda_max_nodes=int(
            args.assignment_conflict_agenda_max_nodes
        ),
        assignment_conflict_agenda_min_total_pushes=int(
            args.assignment_conflict_agenda_min_total_pushes
        ),
        assignment_conflict_agenda_require_done=bool(
            args.assignment_conflict_agenda_require_done
        ),
        seed_offset=58466,
        **_A41._common_args(args),
    )
    a47 = a47_payload["a47_closed_loop_smoke"]
    comparison_a47 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a47",
        a47,
        "a48",
        a48,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a48_assignment_conflict_agenda_v1",
            "agenda_gft_alignment": "E3_E6_E11_E14_assignment_conflict_agenda",
            "failure_log": _rel(failure_path),
            "a47_json": _rel(a47_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a47_planner": RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner.provider_id,
            "a48_planner": RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": False,
            "trace_scope": "a47_a48_assignment_conflict_agenda_smoke",
        },
        "config_extract": {
            **a47_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "assignment_conflict_agenda_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E3_E6_E11_E14_assignment_conflict_agenda",
            "assignment_conflict_agenda_max_pushes": int(
                args.assignment_conflict_agenda_max_pushes
            ),
            "assignment_conflict_agenda_max_nodes": int(
                args.assignment_conflict_agenda_max_nodes
            ),
            "assignment_conflict_agenda_min_total_pushes": int(
                args.assignment_conflict_agenda_min_total_pushes
            ),
            "assignment_conflict_agenda_require_done": bool(
                args.assignment_conflict_agenda_require_done
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a47_closed_loop_smoke": a47,
        "a48_closed_loop_smoke": a48,
        "comparison_a47_a48": comparison_a47,
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
