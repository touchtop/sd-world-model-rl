#!/usr/bin/env python3
"""Audit root-Q bias on known divergent Sokoban SD-MCTS traces.

This diagnostic does not run best-first search and does not modify the SD-MCTS
planner. It compares root policy statistics from existing stage-3 result JSONs
against manually specified action prefixes that were already observed in
closed-loop SD-MCTS traces. The goal is to quantify when root Q prefers a local
gain over a task-relevant route/packing action.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    _goal_entry_delay_penalty_for_action,
    _goal_room_order_bonus_for_action,
    _safe_div,
)
from src.sokoban import SokobanGoalDirectedSDModel, load_boxoban_levels  # noqa: E402
from src.sokoban.env import ACTION_NAMES  # noqa: E402
from validate_sokoban_sd_design import STOCK_NAMES, stock_proxy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument("--level-suffix", default=":6")
    parser.add_argument(
        "--success-result",
        default="results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json",
    )
    parser.add_argument(
        "--failure-result",
        default="results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json",
    )
    parser.add_argument(
        "--success-prefix",
        default="1,3,12,15,4",
        help="Comma-separated flat action ids for an observed successful SD-MCTS prefix.",
    )
    parser.add_argument(
        "--failure-prefix",
        default="1,7,8,4,4",
        help="Comma-separated flat action ids for an observed failed SD-MCTS prefix.",
    )
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/root_q_bias_audit_level6_target2_prior_seed0.json",
    )
    return parser


def parse_prefix(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("Prefix must contain at least one flat action id.")
    return values


def load_episode(result_path: str | Path, level_suffix: str) -> dict[str, Any]:
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    for row in data["episodes"]:
        if row["level_id"].endswith(level_suffix):
            return row
    raise SystemExit(f"No episode ending with {level_suffix!r} in {result_path}")


def action_from_policy(trace_row: dict[str, Any], flat_index: int) -> dict[str, Any] | None:
    for policy_row in trace_row.get("root_policy", []):
        action = policy_row.get("action") or {}
        first = action.get("first_action") if isinstance(action, dict) else None
        if first and int(first.get("flat_index", -1)) == int(flat_index):
            return policy_row
    return None


def state_after_prefix(
    model: SokobanGoalDirectedSDModel,
    prefix: list[int],
    upto: int,
):
    state = model.initial_state
    for flat in prefix[:upto]:
        legal = {action.flat_index: action for action in model.legal_pushes(state)}
        if flat not in legal:
            raise RuntimeError(f"Action {flat} not legal at prefix position {upto}.")
        state = model.step_state(state, legal[flat])
    return state


def action_consequence(model: SokobanGoalDirectedSDModel, state, flat_index: int) -> dict[str, Any] | None:
    legal = {action.flat_index: action for action in model.legal_pushes(state)}
    action = legal.get(flat_index)
    if action is None:
        return None
    next_state = model.step_state(state, action)
    before_features = model.abstract_features(state)
    after_features = model.abstract_features(next_state)
    before_stock = stock_proxy(model, state).as_array()
    after_stock = stock_proxy(model, next_state).as_array()
    return {
        "action": {
            "box_index": int(action.box_index),
            "direction": int(action.direction),
            "direction_name": ACTION_NAMES[action.direction],
            "flat_index": int(action.flat_index),
            "box_cell_before": list(state.boxes[action.box_index]),
        },
        "boxes_on_goals_delta": int(after_features.boxes_on_goals - before_features.boxes_on_goals),
        "matching_lower_bound_delta": (
            float(after_features.matching_lower_bound - before_features.matching_lower_bound)
            if np.isfinite(before_features.matching_lower_bound)
            and np.isfinite(after_features.matching_lower_bound)
            else None
        ),
        "legal_push_delta": int(after_features.legal_push_count - before_features.legal_push_count),
        "viable_push_delta": int(after_features.viable_push_count - before_features.viable_push_count),
        "goal_room_order_bonus": float(_goal_room_order_bonus_for_action(model, state, action)),
        "goal_entry_delay_penalty": float(_goal_entry_delay_penalty_for_action(model, state, action)),
        "stock_delta": {
            name: float(value)
            for name, value in zip(STOCK_NAMES, (after_stock - before_stock).tolist())
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    success_prefix = parse_prefix(args.success_prefix)
    failure_prefix = parse_prefix(args.failure_prefix)
    levels = load_boxoban_levels(args.levels, max_levels=30)
    level = next((level for level in levels if level.level_id.endswith(args.level_suffix)), None)
    if level is None:
        raise SystemExit(f"No level ending with {args.level_suffix!r}.")
    model = SokobanGoalDirectedSDModel(level)
    success_episode = load_episode(args.success_result, args.level_suffix)
    failure_episode = load_episode(args.failure_result, args.level_suffix)
    success_trace = {row["step"]: row for row in success_episode["episode"]["trace_sample"]}
    failure_trace = {row["step"]: row for row in failure_episode["episode"]["trace_sample"]}

    rows: list[dict[str, Any]] = []
    max_steps = min(len(success_prefix), len(failure_prefix))
    for step in range(max_steps):
        success_action = success_prefix[step]
        failure_action = failure_prefix[step]
        state = state_after_prefix(model, success_prefix, step)
        success_row = success_trace.get(step, {})
        failure_row = failure_trace.get(step, {})
        success_policy_on_failure = action_from_policy(failure_row, success_action)
        failure_policy_on_failure = action_from_policy(failure_row, failure_action)
        row = {
            "step": int(step),
            "success_prefix_action": int(success_action),
            "failure_prefix_action": int(failure_action),
            "actions_diverge": bool(success_action != failure_action),
            "failure_trace_chosen_action": failure_row.get("action"),
            "success_policy_in_failure_root": success_policy_on_failure,
            "failure_policy_in_failure_root": failure_policy_on_failure,
            "q_gap_failure_minus_success": (
                float(failure_policy_on_failure["q_value"] - success_policy_on_failure["q_value"])
                if failure_policy_on_failure and success_policy_on_failure
                else None
            ),
            "prior_gap_failure_minus_success": (
                float(failure_policy_on_failure["prior"] - success_policy_on_failure["prior"])
                if failure_policy_on_failure and success_policy_on_failure
                else None
            ),
            "visit_ratio_failure_to_success": (
                _safe_div(float(failure_policy_on_failure["visits"]), float(success_policy_on_failure["visits"]))
                if failure_policy_on_failure and success_policy_on_failure
                else None
            ),
            "success_action_consequence": action_consequence(model, state, success_action),
            "failure_action_consequence": action_consequence(model, state, failure_action),
        }
        rows.append(row)

    divergence_rows = [row for row in rows if row["actions_diverge"]]
    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_root_q_bias_audit_v1",
        },
        "config": {
            "levels": args.levels,
            "level_id": level.level_id,
            "success_result": os.path.relpath(args.success_result, ROOT),
            "failure_result": os.path.relpath(args.failure_result, ROOT),
            "success_prefix": success_prefix,
            "failure_prefix": failure_prefix,
        },
        "audit_contract": {
            "best_first_used_online": False,
            "planner_modified": False,
            "transition_provider_modified": False,
            "uses_existing_sdmcts_root_policy": True,
        },
        "observed": {
            "steps_compared": len(rows),
            "divergence_steps": [row["step"] for row in divergence_rows],
            "first_divergence_step": divergence_rows[0]["step"] if divergence_rows else None,
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "observed": summary["observed"],
        "out": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
