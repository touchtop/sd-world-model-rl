#!/usr/bin/env python3
"""A15 dependency-bound recovery-macro diagnostics for XSokoban first levels.

A15 keeps A11-A14 unchanged and derives a hard-instance planner whose recovery
macro is bound to one concrete schedule dependency row.  The online solver does
not receive teacher actions or reference traces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_A14_SPEC = importlib.util.spec_from_file_location(
    "room_gft_a14_diag",
    ROOT / "scripts/analyze_room_gft_xsokoban_a14_recovery_macro.py",
)
if _A14_SPEC is None or _A14_SPEC.loader is None:
    raise RuntimeError("Unable to load A14 diagnostic helpers.")
_A14 = importlib.util.module_from_spec(_A14_SPEC)
_A14_SPEC.loader.exec_module(_A14)

from src.sokoban import (  # noqa: E402
    RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner,
    RoomScheduleRecoveryMacroGFTSDMCTSPlanner,
    SokobanRoomSchedulePlanGFTSDModel,
    load_boxoban_levels,
)


DEPENDENCY_REASON = RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner.dependency_bound_macro_reason


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


def build_a15_planner(
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
) -> RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner:
    return RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner(
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
    )


def dependency_rows(macro: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in macro.get("rows", [])
        if row.get("reason") == DEPENDENCY_REASON
    ]


def augment_dependency_macro_summary(result: dict[str, Any]) -> dict[str, Any]:
    trace = result["trace"]
    macro_payloads = [
        row.get("corridor_macro", {})
        for row in trace
        if "physical_action" in row
    ]
    triggered = [macro for macro in macro_payloads if dependency_rows(macro)]
    all_rows = [row for macro in triggered for row in dependency_rows(macro)]
    scores = [float(row.get("dependency_bound_score", row.get("recovery_score", 0.0))) for row in all_rows]
    progress = [row.get("dependency_progress", {}) for row in all_rows]
    kinds = sorted(
        {
            kind
            for row in all_rows
            for kind in row.get("bound_dependency", {}).get("selected_dependency_kinds", [])
        }
    )
    result["summary"].update(
        {
            "dependency_bound_macro_candidate_steps": int(len(macro_payloads)),
            "dependency_bound_macro_trigger_steps": int(len(triggered)),
            "dependency_bound_macro_total_extra_pushes": int(len(all_rows)),
            "dependency_bound_macro_score_median": None if not scores else float(median(scores)),
            "dependency_bound_macro_score_max": None if not scores else float(max(scores)),
            "dependency_bound_macro_progress_rows": int(
                sum(bool(row.get("dependency_progress")) for row in progress)
            ),
            "dependency_bound_macro_row_disappeared_count": int(
                sum(bool(row.get("row_disappeared")) for row in progress)
            ),
            "dependency_bound_macro_target_resolved_count": int(
                sum(bool(row.get("target_resolved")) for row in progress)
            ),
            "dependency_bound_macro_stand_resolved_count": int(
                sum(bool(row.get("stand_resolved")) for row in progress)
            ),
            "dependency_bound_macro_bound_kinds": kinds,
        }
    )
    return result


def run_a15_smoke(
    model,
    config: dict[str, Any],
    *,
    recovery_macro_max_extra_pushes: int,
    recovery_macro_max_nodes: int,
    recovery_macro_score_threshold: float,
    dependency_bound_macro_score_threshold: float,
    **kwargs,
) -> dict[str, Any]:
    def builder(*args, **builder_kwargs):
        return build_a15_planner(
            *args,
            recovery_macro_max_extra_pushes=recovery_macro_max_extra_pushes,
            recovery_macro_max_nodes=recovery_macro_max_nodes,
            recovery_macro_score_threshold=recovery_macro_score_threshold,
            dependency_bound_macro_score_threshold=dependency_bound_macro_score_threshold,
            **builder_kwargs,
        )

    original = _A14._A13.build_a13_planner
    _A14._A13.build_a13_planner = builder
    try:
        result = _A14._A13.run_a13_smoke(model, config, **kwargs)
    finally:
        _A14._A13.build_a13_planner = original
    return augment_dependency_macro_summary(result)


def compare_a14_a15(a14: dict[str, Any], a15: dict[str, Any]) -> dict[str, Any]:
    def actions(payload: dict[str, Any]) -> list[str]:
        return [
            f"b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']}"
            for row in payload["trace"]
            if "physical_action" in row
        ]

    a14_actions = actions(a14)
    a15_actions = actions(a15)
    first_diff = None
    for idx, (left, right) in enumerate(zip(a14_actions, a15_actions)):
        if left != right:
            first_diff = idx
            break
    if first_diff is None and len(a14_actions) != len(a15_actions):
        first_diff = min(len(a14_actions), len(a15_actions))
    return {
        "first_decision_difference_step": first_diff,
        "a14_status": a14["summary"]["status"],
        "a15_status": a15["summary"]["status"],
        "a14_boxes_on_goals_final": a14["summary"]["boxes_on_goals_final"],
        "a15_boxes_on_goals_final": a15["summary"]["boxes_on_goals_final"],
        "a14_executed_pushes": a14["summary"]["executed_pushes"],
        "a15_executed_pushes": a15["summary"]["executed_pushes"],
        "a14_final_slot_schedule_cost": a14["summary"]["final_slot_schedule_cost"],
        "a15_final_slot_schedule_cost": a15["summary"]["final_slot_schedule_cost"],
        "a14_prefix_actions": a14_actions[:40],
        "a15_prefix_actions": a15_actions[:40],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a14 = payload["a14_closed_loop_smoke"]["summary"]
    a15 = payload["a15_closed_loop_smoke"]["summary"]
    cmp = payload["comparison"]
    level_number = int(payload["config_extract"]["level_number"])
    lines = [
        f"# XSokoban first 第 {level_number} 关 Room-GFT A15 dependency-bound macro 诊断",
        "",
        "## 核心结论",
        "",
        (
            "A15 继承 A14 的 bounded recovery macro，但短宏搜索必须绑定到一个具体的 "
            "`stand_cell_unreachable_by_player` 或 `target_occupied_by_box` 依赖行。"
        ),
        (
            f"A14 对照为 `{a14['status']}`，执行 `{a14['executed_pushes']}` 个 push 后 "
            f"`boxes_on_goals={a14['boxes_on_goals_final']}`；"
            f"A15 smoke 为 `{a15['status']}`，执行 `{a15['executed_pushes']}` 个 push、"
            f"`{a15['closed_loop_decision_steps']}` 个闭环决策后 "
            f"`boxes_on_goals={a15['boxes_on_goals_final']}`。"
        ),
        (
            f"A14/A15 首个决策差异为 `{cmp['first_decision_difference_step']}`；"
            f"A15 绑定依赖宏触发 `{a15['dependency_bound_macro_trigger_steps']}` 次，"
            f"追加 `{a15['dependency_bound_macro_total_extra_pushes']}` 个 push，"
            f"最终 dependency count 为 `{a15['final_dependency_count']}`。"
        ),
        "",
        "## A14/A15 摘要",
        "",
        "| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | bound macro triggers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| A14 | {a14['status']} | {a14['closed_loop_decision_steps']} | {a14['executed_pushes']} | "
            f"{a14['boxes_on_goals_final']} | {_fmt(a14['final_slot_schedule_cost'])} | "
            f"{a14['final_dependency_count']} | {a14['final_target_blocker_count']} | NA |"
        ),
        (
            f"| A15 | {a15['status']} | {a15['closed_loop_decision_steps']} | {a15['executed_pushes']} | "
            f"{a15['boxes_on_goals_final']} | {_fmt(a15['final_slot_schedule_cost'])} | "
            f"{a15['final_dependency_count']} | {a15['final_target_blocker_count']} | "
            f"{a15['dependency_bound_macro_trigger_steps']} |"
        ),
        "",
        "## A15 轨迹前缀",
        "",
        "| step | action | pushes | goals | slot cost | dependencies | bound kind | bound macro | macro extra | best score | progress |",
        "|---:|---|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in payload["a15_closed_loop_smoke"]["trace"][:80]:
        if "physical_action" not in row:
            schedule = row.get("schedule_plan", {}).get("totals", {})
            lines.append(
                f"| {row['step']} | no_action | 0 | NA | NA | "
                f"{_fmt(schedule.get('dependency_count'))} | NA | NA | 0 | NA | NA |"
            )
            continue
        macro = row.get("corridor_macro", {})
        rows = dependency_rows(macro)
        best_score = None if not rows else max(
            float(macro_row.get("dependency_bound_score", 0.0))
            for macro_row in rows
        )
        kinds = []
        if rows:
            kinds = rows[0].get("bound_dependency", {}).get("selected_dependency_kinds", [])
        progress_bits = []
        for field, label in [
            ("row_disappeared", "row"),
            ("target_resolved", "target"),
            ("stand_resolved", "stand"),
        ]:
            if any(bool(item.get("dependency_progress", {}).get(field)) for item in rows):
                progress_bits.append(label)
        lines.append(
            "| "
            f"{row['step']} | b{row['physical_action']['box_index']}:{row['physical_action']['direction_name']} | "
            f"{row.get('executed_push_count', 1)} | {row['boxes_on_goals_after']} | "
            f"{_fmt(row['slot_progress_after']['slot_schedule_cost'])} | "
            f"{row['schedule_plan_after']['totals']['dependency_count']} | "
            f"{','.join(kinds) if kinds else 'NA'} | "
            f"{bool(rows)} | "
            f"{len(rows)} | "
            f"{_fmt(best_score)} | "
            f"{','.join(progress_bits) if progress_bits else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A15 检验的是：A14 已可执行短宏后，若宏目标从一般恢复指标改为具体依赖行，"
                "是否能更稳定地修复不可达站位或移开目标格阻塞箱。"
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
        "--a14-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a14_recovery_macro_l002_s200.json",
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
    parser.add_argument("--recovery-macro-max-nodes", type=int, default=400)
    parser.add_argument("--recovery-macro-score-threshold", type=float, default=2.0)
    parser.add_argument("--dependency-bound-macro-score-threshold", type=float, default=20.0)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    a14_path = ROOT / args.a14_json
    if not a14_path.exists():
        raise FileNotFoundError(f"A14 baseline JSON not found: {a14_path}")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    a14_payload = json.loads(a14_path.read_text(encoding="utf-8"))
    a14 = a14_payload["a14_closed_loop_smoke"]
    config = failure["config"]
    level_offset = int(args.level_offset)
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=level_offset + 1)[level_offset]
    a15_model = _A14._A13._A12.build_model(level, config)
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
    a15 = run_a15_smoke(
        a15_model,
        config,
        seed_offset=35917,
        recovery_macro_max_extra_pushes=int(args.recovery_macro_max_extra_pushes),
        recovery_macro_max_nodes=int(args.recovery_macro_max_nodes),
        recovery_macro_score_threshold=float(args.recovery_macro_score_threshold),
        dependency_bound_macro_score_threshold=float(args.dependency_bound_macro_score_threshold),
        **common,
    )
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a15_dependency_bound_macro_v1",
            "failure_log": _rel(failure_path),
            "a14_json": _rel(a14_path),
            "a11_model": SokobanRoomSchedulePlanGFTSDModel.provider_id,
            "a14_planner": RoomScheduleRecoveryMacroGFTSDMCTSPlanner.provider_id,
            "a15_planner": RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner.provider_id,
            "mainline_mutation": False,
            "teacher_actions_used_online": False,
            "trace_scope": "a14_recovery_macro_vs_a15_dependency_bound_macro_smoke",
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
            "level_offset": int(level_offset),
            "level_number": int(level_offset) + 1,
            "source_failure_horizon": int(config["horizon"]),
            "source_failure_budget": int(config["budget"]),
            "source_failure_max_steps": int(config["max_steps"]),
        },
        "room_static_summary": a15_model.room_static_summary(),
        "initial_schedule_plan": a15_model.room_schedule_plan_summary(
            a15_model.initial_state,
            a15_model.empty_room_commitment_state(),
            lookahead_slots=int(args.schedule_lookahead_slots),
        ),
        "a14_closed_loop_smoke": a14,
        "a15_closed_loop_smoke": a15,
        "comparison": compare_a14_a15(a14, a15),
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
