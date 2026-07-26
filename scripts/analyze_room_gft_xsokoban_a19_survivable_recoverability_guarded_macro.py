#!/usr/bin/env python3
"""A19 survivable recoverability root-guard diagnostics for XSokoban first levels.

A19 inherits A18 and tightens recoverability: slot-cost-only recovery is kept
as a diagnostic but no longer counts as sufficient.  The online solver does
not receive teacher actions or reference traces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A18_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a18_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a18_recoverability_guarded_macro.py",
)
if _A18_SPEC is None or _A18_SPEC.loader is None:
    raise RuntimeError("Unable to load A18 diagnostic helpers.")
_A18 = importlib.util.module_from_spec(_A18_SPEC)
_A18_SPEC.loader.exec_module(_A18)

from src.sokoban import (  # noqa: E402
    RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner,
    RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    return _A18._fmt(value)


def build_a19_planner(
    model,
    config: dict[str, Any],
    *,
    rng: random.Random,
    budget: int,
    horizon: int,
    visited_keys: set,
    boundary_distance: int,
    recent_moved_edges: list[tuple],
    recent_schedule_keys: set[tuple],
    previous_physical_action=None,
    previous_moved_source=None,
    previous_moved_target=None,
    recent_commitment_keys: set | None = None,
    corridor_macro_max_extra_pushes: int = 8,
    corridor_macro_min_total_pushes: int = 2,
    schedule_lookahead_slots: int = 2,
    recovery_min_legal_pushes: int = 2,
    recovery_macro_max_extra_pushes: int = 4,
    recovery_macro_max_nodes: int = 20,
    recovery_macro_score_threshold: float = 2.0,
    dependency_bound_macro_score_threshold: float = 20.0,
    feasibility_guard_max_slot_cost_regression: float = 0.0,
    feasibility_guard_max_schedule_barrier_regression: float = 8.0,
    recoverability_probe_depth: int = 2,
    recoverability_probe_max_nodes: int = 16,
    survivable_recoverability_min_depth: int = 2,
    survivable_recoverability_min_legal_pushes: int = 3,
) -> RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner:
    return RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner(
        model,
        horizon=horizon,
        budget=budget,
        rng=rng,
        uct_c=float(config["uct_c"]),
        gamma=float(config["gamma"]),
        goal_value=float(config["goal_value"]),
        push_cost=float(config["push_cost"]),
        deadend_value=float(config["deadend_value"]),
        constraint_mode=str(config["constraint_mode"]),
        select_by=str(config["select_by"]),
        revisit_penalty=float(config["revisit_penalty"]),
        visited_keys=visited_keys,
        transposition=True,
        gft_prior_weights={key: float(value) for key, value in config["gft_prior_weights"].items()},
        gft_prior_temperature=float(config["gft_prior_temperature"]),
        gft_prior_uniform_mix=float(config["gft_prior_uniform_mix"]),
        gft_sd_leaf_weights={key: float(value) for key, value in config["gft_sd_leaf_weights"].items()},
        semantic_commitment_boundary_distance=int(boundary_distance),
        previous_physical_action=previous_physical_action,
        previous_moved_source=previous_moved_source,
        previous_moved_target=previous_moved_target,
        recent_commitment_keys=recent_commitment_keys,
        corridor_macro_max_extra_pushes=int(corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(corridor_macro_min_total_pushes),
        room_schedule_lookahead_slots=int(schedule_lookahead_slots),
        recent_moved_edges=recent_moved_edges,
        recent_schedule_keys=recent_schedule_keys,
        room_access_recovery_min_legal_pushes=int(recovery_min_legal_pushes),
        recovery_macro_max_extra_pushes=int(recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(recovery_macro_score_threshold),
        dependency_bound_macro_score_threshold=float(dependency_bound_macro_score_threshold),
        feasibility_guard_max_slot_cost_regression=float(
            feasibility_guard_max_slot_cost_regression
        ),
        feasibility_guard_max_schedule_barrier_regression=float(
            feasibility_guard_max_schedule_barrier_regression
        ),
        recoverability_probe_depth=int(recoverability_probe_depth),
        recoverability_probe_max_nodes=int(recoverability_probe_max_nodes),
        survivable_recoverability_min_depth=int(survivable_recoverability_min_depth),
        survivable_recoverability_min_legal_pushes=int(
            survivable_recoverability_min_legal_pushes
        ),
    )


def augment_survivable_recoverability_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A18.augment_recoverability_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    recovery_rows = [row.get("chosen_recoverability", {}) for row in decision_rows]
    progress_rows = [
        row.get("probe", {}).get("best_progress", {})
        for row in recovery_rows
        if not row.get("probe_skipped", True)
    ]
    result["summary"].update(
        {
            "survivable_structural_progress_selected_count": int(
                sum(bool(row.get("structural_progress")) for row in progress_rows)
            ),
            "survivable_slot_only_progress_selected_count": int(
                sum(bool(row.get("slot_only_progress")) for row in progress_rows)
            ),
            "survivable_slot_only_rejected_selected_count": int(
                sum(bool(row.get("survivable_slot_only_rejected")) for row in recovery_rows)
            ),
            "survivable_slot_only_survived_selected_count": int(
                sum(bool(row.get("slot_only_survivable")) for row in progress_rows)
            ),
        }
    )
    return result


def run_a19_smoke(
    model,
    config: dict[str, Any],
    *,
    recovery_macro_max_extra_pushes: int,
    recovery_macro_max_nodes: int,
    recovery_macro_score_threshold: float,
    dependency_bound_macro_score_threshold: float,
    feasibility_guard_max_slot_cost_regression: float,
    feasibility_guard_max_schedule_barrier_regression: float,
    recoverability_probe_depth: int,
    recoverability_probe_max_nodes: int,
    survivable_recoverability_min_depth: int,
    survivable_recoverability_min_legal_pushes: int,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a19_planner(
            *args,
            recovery_macro_max_extra_pushes=recovery_macro_max_extra_pushes,
            recovery_macro_max_nodes=recovery_macro_max_nodes,
            recovery_macro_score_threshold=recovery_macro_score_threshold,
            dependency_bound_macro_score_threshold=dependency_bound_macro_score_threshold,
            feasibility_guard_max_slot_cost_regression=feasibility_guard_max_slot_cost_regression,
            feasibility_guard_max_schedule_barrier_regression=(
                feasibility_guard_max_schedule_barrier_regression
            ),
            recoverability_probe_depth=recoverability_probe_depth,
            recoverability_probe_max_nodes=recoverability_probe_max_nodes,
            survivable_recoverability_min_depth=survivable_recoverability_min_depth,
            survivable_recoverability_min_legal_pushes=(
                survivable_recoverability_min_legal_pushes
            ),
            **builder_kwargs,
        )

    original = _A18._A17._A16._A15._A14._A13.build_a13_planner
    _A18._A17._A16._A15._A14._A13.build_a13_planner = builder
    try:
        result = _A18._A17._A16._A15._A14._A13.run_a13_smoke(model, config, **kwargs)
    finally:
        _A18._A17._A16._A15._A14._A13.build_a13_planner = original
    return augment_survivable_recoverability_summary(result)


def compare_a18_a19(a18: dict[str, Any], a19: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a18_actions = actions(a18)
    a19_actions = actions(a19)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a18_actions, a19_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a18_actions) != len(a19_actions):
        first_diff = min(len(a18_actions), len(a19_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a18_status": a18["summary"]["status"],
        "a19_status": a19["summary"]["status"],
        "a18_boxes_on_goals_final": a18["summary"]["boxes_on_goals_final"],
        "a19_boxes_on_goals_final": a19["summary"]["boxes_on_goals_final"],
        "a18_executed_pushes": a18["summary"]["executed_pushes"],
        "a19_executed_pushes": a19["summary"]["executed_pushes"],
        "a18_final_slot_schedule_cost": a18["summary"]["final_slot_schedule_cost"],
        "a19_final_slot_schedule_cost": a19["summary"]["final_slot_schedule_cost"],
        "a18_prefix_actions": a18_actions[:40],
        "a19_prefix_actions": a19_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a18 = payload["a18_closed_loop_smoke"]["summary"]
    a19 = payload["a19_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A19 survivable recoverability 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A19 继承 A18，但将可恢复性收紧为结构进展，"
            "避免把纯 slot-cost 局部改善直接当作 recoverable no-progress。"
        ),
        (
            f"A18 对照为 `{a18['status']}`，执行 `{a18['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a18['boxes_on_goals_final']}`；"
            f"A19 smoke 为 `{a19['status']}`，执行 `{a19['executed_pushes']}` 个 push、"
            f"`{a19['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a19['boxes_on_goals_final']}`。"
        ),
        (
            f"A18/A19 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A19 recoverable no-progress selections 为 `{a19['recoverable_no_progress_selected_count']}`，"
            f"unrecoverable selections 为 `{a19['unrecoverable_no_progress_selected_count']}`，"
            f"slot-only rejected selections 为 `{a19['survivable_slot_only_rejected_selected_count']}`。"
        ),
        "",
        "## A18/A19 摘要",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | recoverable | unrecoverable | slot-only rejected |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A18 | {a18['status']} | {a18['closed_loop_decision_steps']} | {a18['executed_pushes']} | "
            f"{a18['boxes_on_goals_final']} | {_fmt(a18['final_slot_schedule_cost'])} | "
            f"{a18['final_dependency_count']} | {a18['final_target_blocker_count']} | "
            f"{a18['recoverable_no_progress_selected_count']} | "
            f"{a18['unrecoverable_no_progress_selected_count']} | NA |"
        ),
        (
            f"| A19 | {a19['status']} | {a19['closed_loop_decision_steps']} | {a19['executed_pushes']} | "
            f"{a19['boxes_on_goals_final']} | {_fmt(a19['final_slot_schedule_cost'])} | "
            f"{a19['final_dependency_count']} | {a19['final_target_blocker_count']} | "
            f"{a19['recoverable_no_progress_selected_count']} | "
            f"{a19['unrecoverable_no_progress_selected_count']} | "
            f"{a19['survivable_slot_only_rejected_selected_count']} |"
        ),
        "",
        "## A19 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | no progress | recoverable | unrecoverable | slot-only rejected | score |",
        "|---:|---|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    for row in payload["a19_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | NA | NA | NA | NA |"
            )
            continue
        phase = row.get("chosen_phase_progress", {})
        recovery = row.get("chosen_recoverability", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{phase.get('no_progress', False)} | "
            f"{recovery.get('recoverable_no_progress', False)} | "
            f"{recovery.get('unrecoverable_no_progress', False)} | "
            f"{recovery.get('survivable_slot_only_rejected', False)} | "
            f"{_fmt(row.get('chosen_recoverability_score'))} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A19 检验的是：A18 的 false-positive recoverability 是否来自纯 slot-cost 局部恢复。"
                "若 A19 仍失败，则应进一步分析根选择替代动作是否缺少可执行 release 生成机制。"
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
        "--a18-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s40_b64_n8.json",
    )
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=40)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument("--schedule-lookahead-slots", type=int, default=2)
    parser.add_argument("--recent-edge-window", type=int, default=12)
    parser.add_argument("--recovery-min-legal-pushes", type=int, default=2)
    parser.add_argument("--recovery-macro-max-extra-pushes", type=int, default=4)
    parser.add_argument("--recovery-macro-max-nodes", type=int, default=20)
    parser.add_argument("--recovery-macro-score-threshold", type=float, default=2.0)
    parser.add_argument("--dependency-bound-macro-score-threshold", type=float, default=20.0)
    parser.add_argument("--feasibility-guard-max-slot-cost-regression", type=float, default=0.0)
    parser.add_argument("--feasibility-guard-max-schedule-barrier-regression", type=float, default=8.0)
    parser.add_argument("--recoverability-probe-depth", type=int, default=2)
    parser.add_argument("--recoverability-probe-max-nodes", type=int, default=8)
    parser.add_argument("--survivable-recoverability-min-depth", type=int, default=2)
    parser.add_argument("--survivable-recoverability-min-legal-pushes", type=int, default=3)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a18_path = ROOT / args.a18_json
    if not a18_path.exists():
        raise FileNotFoundError(f"A18 baseline JSON not found: {a18_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a18_payload = json.loads(a18_path.read_text(encoding="utf-8"))
    a18 = a18_payload["a18_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a19_model = _A18._A17._A16._A15._A14._A13._A12.build_model(level, config)
    common = dict(
        budget=int(args.budget),
        horizon=int(args.horizon),
        max_steps=int(args.smoke_steps),
        boundary_distance=int(args.boundary_distance),
        corridor_macro_max_extra_pushes=int(args.corridor_macro_max_extra_pushes),
        corridor_macro_min_total_pushes=int(args.corridor_macro_min_total_pushes),
        schedule_lookahead_slots=int(args.schedule_lookahead_slots),
        recent_edge_window=int(args.recent_edge_window),
        recovery_min_legal_pushes=int(args.recovery_min_legal_pushes),
    )
    a19 = run_a19_smoke(
        a19_model,
        config,
        seed_offset=41557,
        recovery_macro_max_extra_pushes=int(args.recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(args.recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(args.recovery_macro_score_threshold),
        dependency_bound_macro_score_threshold=float(args.dependency_bound_macro_score_threshold),
        feasibility_guard_max_slot_cost_regression=float(
            args.feasibility_guard_max_slot_cost_regression
        ),
        feasibility_guard_max_schedule_barrier_regression=float(
            args.feasibility_guard_max_schedule_barrier_regression
        ),
        recoverability_probe_depth=int(args.recoverability_probe_depth),
        recoverability_probe_max_nodes=int(args.recoverability_probe_max_nodes),
        survivable_recoverability_min_depth=int(args.survivable_recoverability_min_depth),
        survivable_recoverability_min_legal_pushes=int(
            args.survivable_recoverability_min_legal_pushes
        ),
        **common,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_v1",
            "failure_log": _rel(failure_path),
            "a18_json": _rel(a18_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a18_planner": RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a19_planner": (
                RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner.provider_id
            ),
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a18_recoverability_guarded_macro_vs_a19_survivable_recoverability_guarded_macro_smoke",
        },
        "config_extract": {
            "horizon": int(args.horizon),
            "budget": int(args.budget),
            "smoke_steps": int(args.smoke_steps),
            "semantic_commitment_boundary_distance": int(args.boundary_distance),
            "corridor_macro_max_extra_pushes": int(args.corridor_macro_max_extra_pushes),
            "corridor_macro_min_total_pushes": int(args.corridor_macro_min_total_pushes),
            "schedule_lookahead_slots": int(args.schedule_lookahead_slots),
            "recent_edge_window": int(args.recent_edge_window),
            "recovery_min_legal_pushes": int(args.recovery_min_legal_pushes),
            "recovery_macro_max_extra_pushes": int(args.recovery_macro_max_extra_pushes),
            "recovery_macro_max_nodes": int(args.recovery_macro_max_nodes),
            "recovery_macro_score_threshold": float(args.recovery_macro_score_threshold),
            "dependency_bound_macro_score_threshold": float(
                args.dependency_bound_macro_score_threshold
            ),
            "feasibility_guard_max_slot_cost_regression": float(
                args.feasibility_guard_max_slot_cost_regression
            ),
            "feasibility_guard_max_schedule_barrier_regression": float(
                args.feasibility_guard_max_schedule_barrier_regression
            ),
            "recoverability_probe_depth": int(args.recoverability_probe_depth),
            "recoverability_probe_max_nodes": int(args.recoverability_probe_max_nodes),
            "survivable_recoverability_min_depth": int(
                args.survivable_recoverability_min_depth
            ),
            "survivable_recoverability_min_legal_pushes": int(
                args.survivable_recoverability_min_legal_pushes
            ),
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a19_model.room_static_summary(),
        "initial_schedule_plan": a19_model.room_schedule_plan_summary(
            a19_model.initial_state,
            a19_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a18_closed_loop_smoke": a18,
        "a19_closed_loop_smoke": a19,
        "comparison": compare_a18_a19(a18, a19),
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
