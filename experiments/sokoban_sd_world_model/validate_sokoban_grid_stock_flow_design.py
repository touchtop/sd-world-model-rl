#!/usr/bin/env python3
"""Validate the exact grid stock-flow SD core for Sokoban.

This audit is independent from the legacy six-stock learned-flow line. It checks
that push-level Sokoban dynamics can be represented as a conservative grid flow:
``b_next = b + B f`` with exact legal push support and a push-level canonical
player-region key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import SokobanGridStockFlowModel, SokobanState, load_boxoban_levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory to validate.",
    )
    parser.add_argument("--max-levels", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=80)
    parser.add_argument("--sample-limit", type=int, default=40)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/grid_stock_flow_design_smoke.json",
        help="JSON output path.",
    )
    return parser


def iter_states(
    model: SokobanGridStockFlowModel,
    max_depth: int,
    max_states: int,
) -> Iterable[tuple[SokobanState, int]]:
    start = model.initial_state
    seen = {model.canonical_key(start)}
    queue: deque[tuple[SokobanState, int]] = deque([(start, 0)])
    yielded = 0
    while queue and yielded < max_states:
        state, depth = queue.popleft()
        yielded += 1
        yield state, depth
        if depth >= max_depth:
            continue
        for action in model.legal_flows(state):
            nxt = model.step_grid_flow(state, action)
            key = model.canonical_key(nxt)
            if key in seen:
                continue
            seen.add(key)
            queue.append((nxt, depth + 1))


def _rel(path: str | Path) -> str:
    path = Path(path)
    return os.path.relpath(path, ROOT) if path.is_absolute() else str(path)


def check_canonical_region(model: SokobanGridStockFlowModel, state: SokobanState) -> dict[str, Any]:
    base_key = model.canonical_key(state)
    base_actions = sorted(action.flat_index for action in model.legal_flows(state))
    mismatches = []
    checked = 0
    for player in sorted(model.reachable_cells(state)):
        alt = SokobanState(player=player, boxes=state.boxes)
        checked += 1
        if model.canonical_key(alt) != base_key:
            mismatches.append({"player": list(player), "failure": "canonical_key"})
            continue
        alt_actions = sorted(action.flat_index for action in model.legal_flows(alt))
        if alt_actions != base_actions:
            mismatches.append(
                {
                    "player": list(player),
                    "failure": "legal_flow_support",
                    "actions": alt_actions,
                }
            )
    return {
        "reachable_players_checked": checked,
        "canonical_mismatch_count": len(mismatches),
        "mismatch_samples": mismatches[:5],
    }


def validate_level(
    model: SokobanGridStockFlowModel,
    *,
    max_depth: int,
    max_states: int,
    sample_limit: int,
) -> dict[str, Any]:
    states_checked = 0
    transitions_checked = 0
    box_mass_violations = 0
    player_mass_violations = 0
    flow_onehot_violations = 0
    box_balance_violations = 0
    step_mismatch_violations = 0
    canonical_mismatches = 0
    reachable_players_checked = 0
    tier_a_counts: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []

    for state, depth in iter_states(model, max_depth=max_depth, max_states=max_states):
        states_checked += 1
        if int(model.box_stock(state).sum()) != model.level.n_boxes:
            box_mass_violations += 1
        if int(model.player_stock(state).sum()) != 1:
            player_mass_violations += 1

        canonical = check_canonical_region(model, state)
        canonical_mismatches += int(canonical["canonical_mismatch_count"])
        reachable_players_checked += int(canonical["reachable_players_checked"])

        for action in model.legal_flows(state):
            audit = model.audit_grid_transition(state, action)
            transitions_checked += 1
            if audit["box_mass_before"] != model.level.n_boxes or audit["box_mass_after"] != model.level.n_boxes:
                box_mass_violations += 1
            if audit["player_mass_before"] != 1 or audit["player_mass_after"] != 1:
                player_mass_violations += 1
            if not audit["flow_onehot"]:
                flow_onehot_violations += 1
            if audit["box_balance_l1"] != 0:
                box_balance_violations += 1
            if not audit["step_matches_grid_flow"]:
                step_mismatch_violations += 1

            tier_a_after = audit["tier_a_after"]
            if tier_a_after["solved"]:
                tier_a_counts["solved"] += 1
            if tier_a_after["tier_a_deadlocked"]:
                tier_a_counts["tier_a_deadlocked"] += 1
            if tier_a_after["no_legal_pushes"]:
                tier_a_counts["no_legal_pushes"] += 1

            if len(sample_rows) < sample_limit:
                sample_rows.append(
                    {
                        "depth": depth,
                        "canonical_key_before": audit["canonical_key_before"],
                        "action": audit["action"],
                        "flow": audit["flow"],
                        "box_balance_l1": audit["box_balance_l1"],
                        "tier_a_after": tier_a_after,
                    }
                )

    passed = (
        states_checked > 0
        and transitions_checked > 0
        and box_mass_violations == 0
        and player_mass_violations == 0
        and flow_onehot_violations == 0
        and box_balance_violations == 0
        and step_mismatch_violations == 0
        and canonical_mismatches == 0
    )
    return {
        "level_id": model.level.level_id,
        "height": model.level.height,
        "width": model.level.width,
        "n_boxes": model.level.n_boxes,
        "floor_cells": len(model.floor_cells),
        "directed_edges": model.flow_dim,
        "states_checked": states_checked,
        "transitions_checked": transitions_checked,
        "reachable_players_checked": reachable_players_checked,
        "box_mass_violations": box_mass_violations,
        "player_mass_violations": player_mass_violations,
        "flow_onehot_violations": flow_onehot_violations,
        "box_balance_violations": box_balance_violations,
        "step_mismatch_violations": step_mismatch_violations,
        "canonical_mismatches": canonical_mismatches,
        "tier_a_counts": dict(tier_a_counts),
        "incidence_shape": list(model.grid_incidence_matrix.shape),
        "level_pass": passed,
        "samples": sample_rows,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    tier_a = Counter()
    for row in rows:
        for key in (
            "states_checked",
            "transitions_checked",
            "reachable_players_checked",
            "box_mass_violations",
            "player_mass_violations",
            "flow_onehot_violations",
            "box_balance_violations",
            "step_mismatch_violations",
            "canonical_mismatches",
        ):
            totals[key] += int(row[key])
        tier_a.update(row["tier_a_counts"])
    failed_checks = []
    for key in (
        "box_mass_violations",
        "player_mass_violations",
        "flow_onehot_violations",
        "box_balance_violations",
        "step_mismatch_violations",
        "canonical_mismatches",
    ):
        if totals[key] != 0:
            failed_checks.append(key)
    if not rows or totals["transitions_checked"] == 0:
        failed_checks.append("no_transitions_checked")
    return {
        "levels_checked": len(rows),
        "levels_passed": sum(1 for row in rows if row["level_pass"]),
        **dict(totals),
        "tier_a_counts": dict(tier_a),
        "failed_checks": failed_checks,
        "grid_sd_gate": "PASS" if not failed_checks else "FAIL",
    }


def main() -> None:
    args = build_parser().parse_args()
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    rows = [
        validate_level(
            SokobanGridStockFlowModel(level),
            max_depth=args.max_depth,
            max_states=args.max_states_per_level,
            sample_limit=args.sample_limit,
        )
        for level in levels
    ]
    summary = {
        "material_passport": {
            "origin": "sokoban_grid_stock_flow_refactor",
            "verification_status": "ANALYZED",
            "version_label": "exact_grid_stock_flow_sd_validation_v1",
        },
        "config": {
            "levels": _rel(args.levels),
            "max_levels": args.max_levels,
            "max_depth": args.max_depth,
            "max_states_per_level": args.max_states_per_level,
            "sample_limit": args.sample_limit,
        },
        "design_contract": {
            "provider_id": SokobanGridStockFlowModel.provider_id,
            "state_stocks": ["box_grid_occupancy_b", "player_grid_occupancy_p"],
            "flow": "one_hot_directed_floor_edge_for_each_legal_push",
            "transition": "b_next = b + B f; p_next = one_hot(box_source)",
            "legacy_six_stock_line": "frozen_diagnostic_only",
            "learned_transition_parameters": [],
        },
        "aggregate": aggregate(rows),
        "levels": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
