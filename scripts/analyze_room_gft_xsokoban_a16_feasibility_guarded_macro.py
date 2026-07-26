#!/usr/bin/env python3
"""A16 feasibility-guarded dependency macro diagnostics for XSokoban first levels.

A16 inherits the A15 dependency-bound macro and admits a short macro only when
global slot-fill and legal-action feasibility guards pass.  The online solver
does not receive teacher actions or reference traces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A15_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a15_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a15_dependency_bound_macro.py",
)
if _A15_SPEC is None or _A15_SPEC.loader is None:
    raise RuntimeError("Unable to load A15 diagnostic helpers.")
_A15 = importlib.util.module_from_spec(_A15_SPEC)
_A15_SPEC.loader.exec_module(_A15)

from src.sokoban import (  # noqa: E402
    RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner,
    RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


GUARDED_REASON = RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner.dependency_bound_macro_reason


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_a16_planner(
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
    recovery_macro_max_nodes: int = 400,
    recovery_macro_score_threshold: float = 2.0,
    dependency_bound_macro_score_threshold: float = 20.0,
    feasibility_guard_max_slot_cost_regression: float = 0.0,
    feasibility_guard_max_schedule_barrier_regression: float = 8.0,
) -> RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner:
    return RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner(
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
    )


def guarded_rows(macro: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in macro.get("rows", []) if row.get("reason") == GUARDED_REASON]


def augment_feasibility_summary(result: dict[str, Any]) -> dict[str, Any]:
    trace = result["trace"]
    macro_payloads = [
        row.get("corridor_macro", {})
        for row in trace
        if "physical_action" in row
    ]
    triggered = [macro for macro in macro_payloads if guarded_rows(macro)]
    all_rows = [row for macro in triggered for row in guarded_rows(macro)]
    final_rows = [guarded_rows(macro)[-1] for macro in triggered]
    scores = [float(row.get("dependency_bound_score", row.get("recovery_score", 0.0))) for row in final_rows]
    progress = [row.get("dependency_progress", {}) for row in final_rows]
    violation_counter: Counter[str] = Counter()
    for row in progress:
        violation_counter.update(row.get("global_feasibility_violations", []))
    result["summary"].update(
        {
            "feasibility_guarded_macro_candidate_steps": int(len(macro_payloads)),
            "feasibility_guarded_macro_trigger_steps": int(len(triggered)),
            "feasibility_guarded_macro_total_extra_pushes": int(len(all_rows)),
            "feasibility_guarded_macro_score_median": None if not scores else float(median(scores)),
            "feasibility_guarded_macro_score_max": None if not scores else float(max(scores)),
            "feasibility_guarded_macro_progress_rows": int(
                sum(bool(row.get("dependency_progress")) for row in progress)
            ),
            "feasibility_guarded_macro_row_disappeared_count": int(
                sum(bool(row.get("row_disappeared")) for row in progress)
            ),
            "feasibility_guarded_macro_target_resolved_count": int(
                sum(bool(row.get("target_resolved")) for row in progress)
            ),
            "feasibility_guarded_macro_stand_resolved_count": int(
                sum(bool(row.get("stand_resolved")) for row in progress)
            ),
            "feasibility_guarded_macro_violation_counts": dict(sorted(violation_counter.items())),
        }
    )
    return result


def run_a16_smoke(
    model,
    config: dict[str, Any],
    *,
    recovery_macro_max_extra_pushes: int,
    recovery_macro_max_nodes: int,
    recovery_macro_score_threshold: float,
    dependency_bound_macro_score_threshold: float,
    feasibility_guard_max_slot_cost_regression: float,
    feasibility_guard_max_schedule_barrier_regression: float,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a16_planner(
            *args,
            recovery_macro_max_extra_pushes=recovery_macro_max_extra_pushes,
            recovery_macro_max_nodes=recovery_macro_max_nodes,
            recovery_macro_score_threshold=recovery_macro_score_threshold,
            dependency_bound_macro_score_threshold=dependency_bound_macro_score_threshold,
            feasibility_guard_max_slot_cost_regression=feasibility_guard_max_slot_cost_regression,
            feasibility_guard_max_schedule_barrier_regression=(
                feasibility_guard_max_schedule_barrier_regression
            ),
            **builder_kwargs,
        )

    original = _A15._A14._A13.build_a13_planner
    _A15._A14._A13.build_a13_planner = builder
    try:
        result = _A15._A14._A13.run_a13_smoke(model, config, **kwargs)
    finally:
        _A15._A14._A13.build_a13_planner = original
    return augment_feasibility_summary(result)


def compare_a15_a16(a15: dict[str, Any], a16: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a15_actions = actions(a15)
    a16_actions = actions(a16)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a15_actions, a16_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a15_actions) != len(a16_actions):
        first_diff = min(len(a15_actions), len(a16_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a15_status": a15["summary"]["status"],
        "a16_status": a16["summary"]["status"],
        "a15_boxes_on_goals_final": a15["summary"]["boxes_on_goals_final"],
        "a16_boxes_on_goals_final": a16["summary"]["boxes_on_goals_final"],
        "a15_executed_pushes": a15["summary"]["executed_pushes"],
        "a16_executed_pushes": a16["summary"]["executed_pushes"],
        "a15_final_slot_schedule_cost": a15["summary"]["final_slot_schedule_cost"],
        "a16_final_slot_schedule_cost": a16["summary"]["final_slot_schedule_cost"],
        "a15_prefix_actions": a15_actions[:40],
        "a16_prefix_actions": a16_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a15 = payload["a15_closed_loop_smoke"]["summary"]
    a16 = payload["a16_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A16 feasibility-guarded macro 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A16 继承 A15 的 dependency-bound macro，但只有当 slot cost、target blocker、"
            "critical stand、destructive action 和合法动作余量等全局后置条件通过时才执行短宏。"
        ),
        (
            f"A15 对照为 `{a15['status']}`，执行 `{a15['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a15['boxes_on_goals_final']}`；"
            f"A16 smoke 为 `{a16['status']}`，执行 `{a16['executed_pushes']}` 个 push、"
            f"`{a16['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a16['boxes_on_goals_final']}`。"
        ),
        (
            f"A15/A16 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A16 可行性保护宏触发 `{a16['feasibility_guarded_macro_trigger_steps']}` 次，"
            f"追加 `{a16['feasibility_guarded_macro_total_extra_pushes']}` 个 push，"
            f"最终 dependency count 为 `{a16['final_dependency_count']}`。"
        ),
        "",
        "## A15/A16 摘要",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | guarded macro triggers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A15 | {a15['status']} | {a15['closed_loop_decision_steps']} | {a15['executed_pushes']} | "
            f"{a15['boxes_on_goals_final']} | {_fmt(a15['final_slot_schedule_cost'])} | "
            f"{a15['final_dependency_count']} | {a15['final_target_blocker_count']} | NA |"
        ),
        (
            f"| A16 | {a16['status']} | {a16['closed_loop_decision_steps']} | {a16['executed_pushes']} | "
            f"{a16['boxes_on_goals_final']} | {_fmt(a16['final_slot_schedule_cost'])} | "
            f"{a16['final_dependency_count']} | {a16['final_target_blocker_count']} | "
            f"{a16['feasibility_guarded_macro_trigger_steps']} |"
        ),
        "",
        "## A16 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | guarded macro | macro extra | best score | feasibility | progress |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in payload["a16_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | 0 | NA | NA | NA |"
            )
            continue
        macro = row.get("corridor_macro", {})
        rows = guarded_rows(macro)
        final_row = None if not rows else rows[-1]
        best_score = None if final_row is None else float(final_row.get("dependency_bound_score", 0.0))
        progress_bits = []
        feasibility_bits = []
        for field, label in [
            ("row_disappeared", "row"),
            ("target_resolved", "target"),
            ("stand_resolved", "stand"),
        ]:
            if final_row is not None and bool(final_row.get("dependency_progress", {}).get(field)):
                progress_bits.append(label)
        if final_row is not None:
            feasibility_bits = final_row.get("dependency_progress", {}).get(
                "global_feasibility_violations",
                [],
            )
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{bool(rows)} | "
            f"{len(rows)} | "
            f"{_fmt(best_score)} | "
            f"{','.join(feasibility_bits) if feasibility_bits else ('pass' if rows else 'NA')} | "
            f"{','.join(progress_bits) if progress_bits else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A16 检验的是：A15 的局部 dependency row 解析必须与全局 slot-fill 可行性同时成立，"
                "否则短宏不应执行。"
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
        "--a15-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.json",
    )
    parser.add_argument("--level-offset", type=int, default=1)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--boundary-distance", type=int, default=6)
    parser.add_argument("--corridor-macro-max-extra-pushes", type=int, default=8)
    parser.add_argument("--corridor-macro-min-total-pushes", type=int, default=2)
    parser.add_argument("--schedule-lookahead-slots", type=int, default=2)
    parser.add_argument("--recent-edge-window", type=int, default=12)
    parser.add_argument("--recovery-min-legal-pushes", type=int, default=2)
    parser.add_argument("--recovery-macro-max-extra-pushes", type=int, default=4)
    parser.add_argument("--recovery-macro-max-nodes", type=int, default=80)
    parser.add_argument("--recovery-macro-score-threshold", type=float, default=2.0)
    parser.add_argument("--dependency-bound-macro-score-threshold", type=float, default=20.0)
    parser.add_argument("--feasibility-guard-max-slot-cost-regression", type=float, default=0.0)
    parser.add_argument("--feasibility-guard-max-schedule-barrier-regression", type=float, default=8.0)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s200.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s200.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a15_path = ROOT / args.a15_json
    if not a15_path.exists():
        raise FileNotFoundError(f"A15 baseline JSON not found: {a15_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a15_payload = json.loads(a15_path.read_text(encoding="utf-8"))
    a15 = a15_payload["a15_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a16_model = _A15._A14._A13._A12.build_model(level, config)
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
    a16 = run_a16_smoke(
        a16_model,
        config,
        seed_offset=37123,
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
        **common,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a16_feasibility_guarded_macro_v1",
            "failure_log": _rel(failure_path),
            "a15_json": _rel(a15_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a15_planner": RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner.provider_id,
            "a16_planner": RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a15_dependency_bound_macro_vs_a16_feasibility_guarded_macro_smoke",
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
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a16_model.room_static_summary(),
        "initial_schedule_plan": a16_model.room_schedule_plan_summary(
            a16_model.initial_state,
            a16_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a15_closed_loop_smoke": a15,
        "a16_closed_loop_smoke": a16,
        "comparison": compare_a15_a16(a15, a16),
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
