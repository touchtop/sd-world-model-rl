#!/usr/bin/env python3
"""A46 residual active-agenda diagnostics for XSokoban first levels."""

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

_A45_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a45_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a45_active_blocker_commitment.py",
)
if _A45_SPEC is None or _A45_SPEC.loader is None:
    raise RuntimeError("Unable to load A45 diagnostic helpers.")
_A45 = importlib.util.module_from_spec(_A45_SPEC)
_A45_SPEC.loader.exec_module(_A45)

_A44 = _A45._A44
_A41 = _A45._A41
_A39 = _A45._A39
_A25 = _A45._A25

from src.sokoban import (  # noqa: E402
    RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner,
    RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner,
)


def _rel(path: Path | str) -> str:
    return _A45._rel(path)


def _fmt(value: Any) -> str:
    return _A45._fmt(value)


def _residual(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("residual_active_agenda", {})
    return value if isinstance(value, dict) else {}


def augment_a46_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A45.augment_a45_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    loaded = [_residual(row) for row in decision_rows if _residual(row)]
    triggered = [
        row for row in loaded if bool(row.get("residual_active_agenda_triggered"))
    ]
    reason_counts: dict[str, int] = {}
    for row in loaded:
        reason = str(row.get("residual_active_agenda_reason", "NA"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    first_trigger_step = None
    for trace_row in decision_rows:
        row = _residual(trace_row)
        if bool(row.get("residual_active_agenda_triggered")):
            first_trigger_step = int(trace_row["step"])
            break
    result["summary"].update(
        {
            "residual_active_agenda_loaded_count": int(len(loaded)),
            "residual_active_agenda_triggered_count": int(len(triggered)),
            "residual_active_agenda_reason_counts": reason_counts,
            "residual_active_agenda_first_trigger_step": first_trigger_step,
            "residual_active_agenda_committed_push_count": int(
                sum(
                    int(row.get("committed_residual_push_count", 0) or 0)
                    for row in triggered
                )
            ),
            "residual_active_agenda_evaluated_option_count": int(
                sum(int(row.get("evaluated_option_count", 0) or 0) for row in loaded)
            ),
            "residual_active_agenda_target_progress_count": int(
                sum(
                    bool(
                        row.get("final_progress", {}).get(
                            "residual_active_agenda_target_progress",
                            False,
                        )
                    )
                    if isinstance(row.get("final_progress", {}), dict)
                    else False
                    for row in triggered
                )
            ),
            "residual_active_agenda_max_expanded_nodes": None
            if not loaded
            else int(max(int(row.get("expanded_nodes", 0) or 0) for row in loaded)),
        }
    )
    return result


def run_a46_smoke(
    model,
    config: dict[str, Any],
    *,
    residual_active_agenda_max_pushes: int = 8,
    residual_active_agenda_max_nodes: int = 192,
    residual_active_agenda_min_total_pushes: int = 1,
    residual_active_agenda_require_target_progress: bool = True,
    **kwargs,
) -> dict[str, Any]:
    original = _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner

    class ConfiguredA46Planner(RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner):
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
                **planner_kwargs,
            )

    _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner = (
        ConfiguredA46Planner
    )
    try:
        result = _A45.run_a45_smoke(model, config, **kwargs)
    finally:
        _A45.RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner = original
    return augment_a46_summary(result)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a45 = payload["a45_closed_loop_smoke"]["summary"]
    a46 = payload["a46_closed_loop_smoke"]["summary"]
    cmp_a45 = payload["comparison_a45_a46"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A46 residual active-agenda 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A46 继承 A45，对应专家 27 点中的 active agenda state、distance to "
            "next agenda milestone 与 agenda-option tree。A45 承诺若只完成 "
            "`stand_resolved` 而留下同一 relation 的 `target_occupied_by_box`，"
            "A46 才在该宏边界后追加一次有界 residual option 搜索；若未找到 target "
            "进展，则不改变 A45 动作，只记录负证据。"
        ),
        (
            f"A45 为 `{a45['status']}`、pushes={a45['executed_pushes']}、"
            f"decisions={a45['closed_loop_decision_steps']}、goals={a45['boxes_on_goals_final']}；"
            f"A46 为 `{a46['status']}`、pushes={a46['executed_pushes']}、"
            f"decisions={a46['closed_loop_decision_steps']}、goals={a46['boxes_on_goals_final']}。"
        ),
        (
            f"A45/A46 首个决策差异为 `{cmp_a45['a45_a46_first_decision_difference_step']}`；"
            f"A46 residual agenda 触发 `{a46.get('residual_active_agenda_triggered_count', 0)}` 次，"
            f"追加 `{a46.get('residual_active_agenda_committed_push_count', 0)}` 个 residual push。"
        ),
        "",
        "## 摘要对照",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | abc triggers | residual triggers | residual pushes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A45 | {a45['status']} | {a45['closed_loop_decision_steps']} | "
            f"{a45['executed_pushes']} | {a45['boxes_on_goals_final']} | "
            f"{_fmt(a45['final_slot_schedule_cost'])} | {a45['final_dependency_count']} | "
            f"{a45['final_target_blocker_count']} | "
            f"{a45.get('active_blocker_commitment_triggered_count', 'NA')} | NA | NA |"
        ),
        (
            f"| A46 | {a46['status']} | {a46['closed_loop_decision_steps']} | "
            f"{a46['executed_pushes']} | {a46['boxes_on_goals_final']} | "
            f"{_fmt(a46['final_slot_schedule_cost'])} | {a46['final_dependency_count']} | "
            f"{a46['final_target_blocker_count']} | "
            f"{a46.get('active_blocker_commitment_triggered_count', 'NA')} | "
            f"{a46.get('residual_active_agenda_triggered_count', 'NA')} | "
            f"{a46.get('residual_active_agenda_committed_push_count', 'NA')} |"
        ),
        "",
        "## A46 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dep | target | abc | residual reason | residual pushes | residual target |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in payload["a46_closed_loop_smoke"]["trace"][:90]:
        if "physical_action" not in row:
            continue
        residual = _residual(row)
        final_progress = residual.get("final_progress", {}) or {}
        totals = row["schedule_plan_after"]["totals"]
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{totals['dependency_count']} | {totals['target_blocker_count']} | "
            f"{row.get('active_blocker_commitment', {}).get('active_blocker_commitment_triggered')} | "
            f"{residual.get('residual_active_agenda_reason')} | "
            f"{residual.get('committed_residual_push_count', 0)} | "
            f"{final_progress.get('residual_active_agenda_target_progress')} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A46 检验的是 A45 后的残余义务是否能在同一宏边界内继续推进。"
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
        "--a45-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json",
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
    parser.add_argument(
        "--no-residual-active-agenda-require-target-progress",
        dest="residual_active_agenda_require_target_progress",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a45_path = ROOT / args.a45_json
    checkpoint_path = ROOT / args.multihead_sd_consumer_checkpoint
    for required in (failure_path, a45_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a45_payload = json.loads(a45_path.read_text(encoding="utf-8"))
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
    a46 = run_a46_smoke(
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
        residual_active_agenda_require_target_progress=bool(
            args.residual_active_agenda_require_target_progress
        ),
        seed_offset=58464,
        **_A41._common_args(args),
    )
    a45 = a45_payload["a45_closed_loop_smoke"]
    comparison_a45 = _A39._A38._A37._A36._A35._A34._A33._A32.compare_runs(
        "a45",
        a45,
        "a46",
        a46,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a46_residual_active_agenda_v1",
            "agenda_gft_alignment": "E6_E11_E14_residual_active_agenda",
            "failure_log": _rel(failure_path),
            "a45_json": _rel(a45_path),
            "multihead_sd_consumer_checkpoint": _rel(checkpoint_path),
            "a11_model": _A39._A38._A37._A36.SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a45_planner": RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner.provider_id,
            "a46_planner": RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "learned_sd_consumer_root_override": False,
            "trace_scope": "a45_a46_residual_active_agenda_smoke",
        },
        "config_extract": {
            **a45_payload.get("config_extract", {}),
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "sd_value_consumer": "residual_active_agenda_multihead_soft_prior_v1",
            "agenda_gft_alignment": "E6_E11_E14_residual_active_agenda",
            "residual_active_agenda_max_pushes": int(
                args.residual_active_agenda_max_pushes
            ),
            "residual_active_agenda_max_nodes": int(
                args.residual_active_agenda_max_nodes
            ),
            "residual_active_agenda_min_total_pushes": int(
                args.residual_active_agenda_min_total_pushes
            ),
            "residual_active_agenda_require_target_progress": bool(
                args.residual_active_agenda_require_target_progress
            ),
        },
        "room_static_summary": model.room_static_summary(),
        "initial_schedule_plan": model.room_schedule_plan_summary(
            model.initial_state,
            model.empty_room_commitment_state(),
            lookahead_slots=2,
        ),
        "a45_closed_loop_smoke": a45,
        "a46_closed_loop_smoke": a46,
        "comparison_a45_a46": comparison_a45,
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
