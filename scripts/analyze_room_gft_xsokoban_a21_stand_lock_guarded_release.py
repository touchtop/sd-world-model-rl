#!/usr/bin/env python3
"""A21 stand-lock guarded release diagnostics for XSokoban first levels.

A21 inherits A20 and adds a root/leaf guard against unreleasable critical
stand blockers, especially boxes already sitting on goal cells.  The online
solver does not receive teacher actions or reference traces.
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

_A20_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a20_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a20_root_cached_survivable_recoverability.py",
)
if _A20_SPEC is None or _A20_SPEC.loader is None:
    raise RuntimeError("Unable to load A20 diagnostic helpers.")
_A20 = importlib.util.module_from_spec(_A20_SPEC)
_A20_SPEC.loader.exec_module(_A20)

from src.sokoban import (  # noqa: E402
    RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner,
    RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    return _A20._fmt(value)


def build_a21_planner(
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
    recoverability_probe_max_nodes: int = 8,
    survivable_recoverability_min_depth: int = 2,
    survivable_recoverability_min_legal_pushes: int = 3,
) -> RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner:
    return RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner(
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
        root_recoverability_only=True,
    )


def augment_stand_lock_summary(result: dict[str, Any]) -> dict[str, Any]:
    result = _A20.augment_root_cached_summary(result)
    decision_rows = [row for row in result["trace"] if "physical_action" in row]
    lock_rows = [row.get("chosen_stand_lock", {}) for row in decision_rows]
    after_rows = [row.get("after", {}) for row in lock_rows]
    scores = [
        float(row.get("score", 0.0))
        for row in lock_rows
        if row
    ]
    result["summary"].update(
        {
            "stand_lock_selected_count": int(sum(bool(row) for row in lock_rows)),
            "stand_lock_risk_selected_count": int(
                sum(bool(row.get("stand_lock_risk")) for row in lock_rows)
            ),
            "locked_stand_blocker_selected_count": int(
                sum(int(row.get("locked_stand_blocker_count", 0)) > 0 for row in after_rows)
            ),
            "assigned_goal_reassignment_selected_count": int(
                sum(int(row.get("assigned_goal_reassignment_count", 0)) > 0 for row in after_rows)
            ),
            "stand_lock_score_min": None if not scores else float(min(scores)),
            "stand_lock_score_max": None if not scores else float(max(scores)),
        }
    )
    return result


def run_a21_smoke(
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
        return build_a21_planner(
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

    original = _A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner
    _A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner = builder
    try:
        result = _A20._A19._A18._A17._A16._A15._A14._A13.run_a13_smoke(
            model,
            config,
            **kwargs,
        )
    finally:
        _A20._A19._A18._A17._A16._A15._A14._A13.build_a13_planner = original
    return augment_stand_lock_summary(result)


def compare_a20_a21(a20: dict[str, Any], a21: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a20_actions = actions(a20)
    a21_actions = actions(a21)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a20_actions, a21_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a20_actions) != len(a21_actions):
        first_diff = min(len(a20_actions), len(a21_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a20_status": a20["summary"]["status"],
        "a21_status": a21["summary"]["status"],
        "a20_boxes_on_goals_final": a20["summary"]["boxes_on_goals_final"],
        "a21_boxes_on_goals_final": a21["summary"]["boxes_on_goals_final"],
        "a20_executed_pushes": a20["summary"]["executed_pushes"],
        "a21_executed_pushes": a21["summary"]["executed_pushes"],
        "a20_final_slot_schedule_cost": a20["summary"]["final_slot_schedule_cost"],
        "a21_final_slot_schedule_cost": a21["summary"]["final_slot_schedule_cost"],
        "a20_prefix_actions": a20_actions[:50],
        "a21_prefix_actions": a21_actions[:50],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a20 = payload["a20_closed_loop_smoke"]["summary"]
    a21 = payload["a21_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    goal_delta = int(a21["boxes_on_goals_final"]) - int(a20["boxes_on_goals_final"])
    if goal_delta < 0:
        outcome = (
            "本次 A21 是负结果：stand-lock 风险信号是真实的，但只在根选择和 leaf "
            "中惩罚不可释放站位阻塞，会过早偏离 A20 的有效装填前缀；该版本没有生成新的 "
            "release action，因此不能修复两个目标后的释放缺口。"
        )
    elif goal_delta > 0:
        outcome = (
            "本次 A21 是正向结果：stand-lock guard 在不引入 teacher/reference "
            "在线动作的前提下，提高了目标完成数。"
        )
    else:
        outcome = (
            "本次 A21 是持平诊断：stand-lock guard 改变了调度轨迹，但没有提高目标完成数。"
        )
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A21 stand-lock guard 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A21 继承 A20，增加不可释放 critical stand blocker 诊断与根保护，"
            "用于避免调度依赖已在目标/角落中的箱子作为后续站位格。"
        ),
        (
            f"A20 对照为 `{a20['status']}`，执行 `{a20['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a20['boxes_on_goals_final']}`；"
            f"A21 smoke 为 `{a21['status']}`，执行 `{a21['executed_pushes']}` 个 push、"
            f"`{a21['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a21['boxes_on_goals_final']}`。"
        ),
        (
            f"A20/A21 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A21 locked stand selected 为 `{a21['locked_stand_blocker_selected_count']}`，"
            f"assigned-goal selected 为 `{a21['assigned_goal_reassignment_selected_count']}`。"
        ),
        outcome,
        "",
        "## A20/A21 摘要",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | locked selected | assigned-goal selected |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A20 | {a20['status']} | {a20['closed_loop_decision_steps']} | {a20['executed_pushes']} | "
            f"{a20['boxes_on_goals_final']} | {_fmt(a20['final_slot_schedule_cost'])} | "
            f"{a20['final_dependency_count']} | {a20['final_target_blocker_count']} | "
            f"{a20['final_release_action_count']} | NA | NA |"
        ),
        (
            f"| A21 | {a21['status']} | {a21['closed_loop_decision_steps']} | {a21['executed_pushes']} | "
            f"{a21['boxes_on_goals_final']} | {_fmt(a21['final_slot_schedule_cost'])} | "
            f"{a21['final_dependency_count']} | {a21['final_target_blocker_count']} | "
            f"{a21['final_release_action_count']} | "
            f"{a21['locked_stand_blocker_selected_count']} | "
            f"{a21['assigned_goal_reassignment_selected_count']} |"
        ),
        "",
        "## A21 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | locked after | assigned-goal after | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["a21_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | NA | NA |"
            )
            continue
        lock = row.get("chosen_stand_lock", {})
        after = lock.get("after", {})
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{_fmt(after.get('locked_stand_blocker_count'))} | "
            f"{_fmt(after.get('assigned_goal_reassignment_count'))} | "
            f"{_fmt(row.get('chosen_stand_lock_score'))} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A21 检验的是：A20 后半段的失败是否来自不可释放 stand-lock。"
                "若该 guard 降低目标数或提前失败，则说明仅禁止 stand-lock 还需要配套可执行 release 生成。"
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
        "--a20-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.json",
    )
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=80)
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
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.md",
    )
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a20_path = ROOT / args.a20_json
    if not a20_path.exists():
        raise FileNotFoundError(f"A20 baseline JSON not found: {a20_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a20_payload = json.loads(a20_path.read_text(encoding="utf-8"))
    a20 = a20_payload["a20_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a21_model = _A20._A19._A18._A17._A16._A15._A14._A13._A12.build_model(level, config)
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
    a21 = run_a21_smoke(
        a21_model,
        config,
        seed_offset=45119,
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
            "analysis": "room_gft_xsokoban_a21_stand_lock_guarded_release_v1",
            "failure_log": _rel(failure_path),
            "a20_json": _rel(a20_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a20_planner": RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner.provider_id,
            "a21_planner": RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a20_root_cached_recoverability_vs_a21_stand_lock_guarded_release_smoke",
        },
        "a20_config_extract": a20_payload.get("config_extract", {}),
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
        "room_static_summary": a21_model.room_static_summary(),
        "initial_schedule_plan": a21_model.room_schedule_plan_summary(
            a21_model.initial_state,
            a21_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a20_closed_loop_smoke": a20,
        "a21_closed_loop_smoke": a21,
        "comparison": compare_a20_a21(a20, a21),
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
