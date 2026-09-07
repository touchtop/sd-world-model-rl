#!/usr/bin/env python3
"""Validation smoke test for the revised Sokoban stock-flow SD design.

This script does not train a neural world model. It builds diagnostic labels
from exact single-push transitions and checks whether the revised six-stock
contract, incidence matrix, risk-label taxonomy, and provider audit can be
populated reproducibly before learned SD-MCTS experiments are attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES
from src.sokoban.sd_model import SokobanPushAction


STOCK_NAMES = (
    "goal_completion",
    "unresolved_box_pressure",
    "mobility",
    "reachable_capacity",
    "congestion",
    "irreversibility_risk",
)

FLOW_NAMES = (
    "goal_fill_flow",
    "goal_unfill_flow",
    "assignment_relief_flow",
    "assignment_worsening_flow",
    "mobility_loss_flow",
    "mobility_recovery_flow",
    "reachability_loss_flow",
    "reachability_recovery_flow",
    "congestion_generation_flow",
    "congestion_release_flow",
    "risk_accumulation_flow",
    "risk_relief_flow",
)

# Rows follow STOCK_NAMES, columns follow FLOW_NAMES. This is the revised
# design's first diagnostic incidence matrix. Magnitudes are diagnostic labels,
# not learned parameters.
INCIDENCE_MATRIX = np.array(
    [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # G
        [-1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # U
        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],  # M
        [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],  # R
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],  # C
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # I
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class StockProxy:
    goal_completion: float
    unresolved_box_pressure: float
    mobility: float
    reachable_capacity: float
    congestion: float
    irreversibility_risk: float

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in STOCK_NAMES], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory to validate.",
    )
    parser.add_argument("--max-levels", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-states-per-level", type=int, default=40)
    parser.add_argument("--alias-round-decimals", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=40)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/design_validation_smoke.json",
        help="JSON output path.",
    )
    return parser


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def stock_proxy(model: SokobanGoalDirectedSDModel, state: SokobanState) -> StockProxy:
    features = model.abstract_features(state)
    n_boxes = max(model.level.n_boxes, 1)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    floor_count = max(len(model.floor_cells), 1)
    board_scale = max(model.level.height + model.level.width, 1)

    if np.isfinite(features.matching_lower_bound):
        matching_pressure = _clamp01(features.matching_lower_bound / (n_boxes * board_scale))
    else:
        matching_pressure = 1.0

    unresolved_count = _safe_div(n_boxes - features.boxes_on_goals, n_boxes)
    structural_deadlock_pressure = _safe_div(
        features.dead_square_boxes + features.frozen_boxes,
        n_boxes,
    )
    if features.matching_deadlock:
        structural_deadlock_pressure = 1.0

    return StockProxy(
        goal_completion=_clamp01(_safe_div(features.boxes_on_goals, n_boxes)),
        unresolved_box_pressure=_clamp01(0.7 * unresolved_count + 0.3 * matching_pressure),
        mobility=_clamp01(_safe_div(features.viable_push_count, max_pushes)),
        reachable_capacity=_clamp01(_safe_div(features.reachable_count, floor_count)),
        congestion=_clamp01(1.0 - _safe_div(features.legal_push_count, max_pushes)),
        irreversibility_risk=_clamp01(structural_deadlock_pressure),
    )


def diagnostic_flows(delta: np.ndarray) -> np.ndarray:
    flows = np.zeros(len(FLOW_NAMES), dtype=np.float64)

    goal_delta = float(delta[0])
    flows[0] = max(goal_delta, 0.0)
    flows[1] = max(-goal_delta, 0.0)

    unresolved_residual = float(delta[1] - (-flows[0] + flows[1]))
    flows[2] = max(-unresolved_residual, 0.0)
    flows[3] = max(unresolved_residual, 0.0)

    mobility_delta = float(delta[2])
    flows[4] = max(-mobility_delta, 0.0)
    flows[5] = max(mobility_delta, 0.0)

    reach_delta = float(delta[3])
    flows[6] = max(-reach_delta, 0.0)
    flows[7] = max(reach_delta, 0.0)

    congestion_delta = float(delta[4])
    flows[8] = max(congestion_delta, 0.0)
    flows[9] = max(-congestion_delta, 0.0)

    risk_delta = float(delta[5])
    flows[10] = max(risk_delta, 0.0)
    flows[11] = max(-risk_delta, 0.0)
    return flows


def risk_label(model: SokobanGoalDirectedSDModel, state: SokobanState) -> str:
    features = model.abstract_features(state)
    if features.matching_deadlock or features.dead_square_boxes > 0 or features.frozen_boxes > 0:
        return "proven_dead_end"
    if not features.solved and features.legal_push_count == 0:
        return "short_rollout_failure"
    if not features.solved and features.viable_push_count == 0:
        return "heuristic_warning"
    return "none"


def iter_states(
    model: SokobanGoalDirectedSDModel,
    max_depth: int,
    max_states: int,
) -> Iterable[Tuple[SokobanState, int]]:
    start = model.initial_state
    seen = {start}
    queue: deque[Tuple[SokobanState, int]] = deque([(start, 0)])
    yielded = 0
    while queue and yielded < max_states:
        state, depth = queue.popleft()
        yielded += 1
        yield state, depth
        if depth >= max_depth:
            continue
        for action in model.legal_pushes(state):
            nxt = model.step_state(state, action)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, depth + 1))


def action_to_dict(action: SokobanPushAction) -> Dict[str, Any]:
    return {
        "box_index": action.box_index,
        "direction": int(action.direction),
        "direction_name": ACTION_NAMES[action.direction],
        "flat_index": action.flat_index,
    }


def validate_level(
    model: SokobanGoalDirectedSDModel,
    *,
    max_depth: int,
    max_states: int,
    alias_round_decimals: int,
    sample_limit: int,
) -> Dict[str, Any]:
    transition_count = 0
    range_violations = 0
    nonnegative_flow_violations = 0
    max_balance_l1 = 0.0
    max_abs_delta = 0.0
    risk_counts: Counter[str] = Counter()
    state_provider_counts: Counter[str] = Counter()
    transition_provider_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    alias_buckets: dict[Tuple[float, ...], list[np.ndarray]] = defaultdict(list)

    for state, depth in iter_states(model, max_depth=max_depth, max_states=max_states):
        x = stock_proxy(model, state).as_array()
        state_provider_counts["diagnostic_stock_proxy_encoder"] += 1
        if np.any(x < -1e-9) or np.any(x > 1.0 + 1e-9):
            range_violations += 1

        bucket = tuple(np.round(x, alias_round_decimals).tolist())
        for action in model.legal_pushes(state):
            next_state = model.step_state(state, action)
            next_x = stock_proxy(model, next_state).as_array()
            if np.any(next_x < -1e-9) or np.any(next_x > 1.0 + 1e-9):
                range_violations += 1

            delta = next_x - x
            flows = diagnostic_flows(delta)
            if np.any(flows < -1e-12):
                nonnegative_flow_violations += 1
            reconstructed = INCIDENCE_MATRIX @ flows
            balance_l1 = float(np.abs(reconstructed - delta).sum())
            max_balance_l1 = max(max_balance_l1, balance_l1)
            max_abs_delta = max(max_abs_delta, float(np.abs(delta).max()))

            label = risk_label(model, next_state)
            risk_counts[label] += 1
            state_provider_counts["diagnostic_stock_proxy_encoder"] += 1
            transition_provider_counts["exact_single_push_diagnostic_provider"] += 1
            alias_buckets[bucket].append(delta)
            transition_count += 1

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "depth": depth,
                        "action": action_to_dict(action),
                        "stock_before": dict(zip(STOCK_NAMES, x.tolist())),
                        "stock_after": dict(zip(STOCK_NAMES, next_x.tolist())),
                        "stock_delta": dict(zip(STOCK_NAMES, delta.tolist())),
                        "diagnostic_flows": dict(zip(FLOW_NAMES, flows.tolist())),
                        "balance_l1": balance_l1,
                        "risk_label_after": label,
                    }
                )

    alias_variances = []
    aliased_buckets = 0
    for deltas in alias_buckets.values():
        if len(deltas) < 2:
            continue
        aliased_buckets += 1
        arr = np.stack(deltas, axis=0)
        alias_variances.append(float(np.var(arr, axis=0).sum()))

    invariant_pass = (
        transition_count > 0
        and range_violations == 0
        and nonnegative_flow_violations == 0
        and max_balance_l1 <= 1e-9
    )
    return {
        "level_id": model.level.level_id,
        "n_boxes": model.level.n_boxes,
        "states_checked": int(sum(state_provider_counts.values()) - transition_count),
        "transitions_checked": transition_count,
        "range_violations": range_violations,
        "nonnegative_flow_violations": nonnegative_flow_violations,
        "max_balance_l1": max_balance_l1,
        "max_abs_stock_delta": max_abs_delta,
        "risk_label_counts": dict(risk_counts),
        "state_provider_counts": dict(state_provider_counts),
        "transition_provider_counts": dict(transition_provider_counts),
        "aliasing": {
            "stock_bucket_count": len(alias_buckets),
            "aliased_bucket_count": aliased_buckets,
            "max_next_delta_variance": max(alias_variances) if alias_variances else 0.0,
            "mean_next_delta_variance": float(np.mean(alias_variances)) if alias_variances else 0.0,
        },
        "invariant_pass": invariant_pass,
        "samples": samples,
    }


def aggregate(level_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    risk_counts: Counter[str] = Counter()
    transition_provider_counts: Counter[str] = Counter()
    state_provider_counts: Counter[str] = Counter()
    for row in level_rows:
        risk_counts.update(row["risk_label_counts"])
        transition_provider_counts.update(row["transition_provider_counts"])
        state_provider_counts.update(row["state_provider_counts"])

    return {
        "levels_checked": len(level_rows),
        "levels_passed": sum(1 for row in level_rows if row["invariant_pass"]),
        "states_checked": sum(row["states_checked"] for row in level_rows),
        "transitions_checked": sum(row["transitions_checked"] for row in level_rows),
        "range_violations": sum(row["range_violations"] for row in level_rows),
        "nonnegative_flow_violations": sum(row["nonnegative_flow_violations"] for row in level_rows),
        "max_balance_l1": max((row["max_balance_l1"] for row in level_rows), default=0.0),
        "max_abs_stock_delta": max((row["max_abs_stock_delta"] for row in level_rows), default=0.0),
        "risk_label_counts": dict(risk_counts),
        "state_provider_counts": dict(state_provider_counts),
        "transition_provider_counts": dict(transition_provider_counts),
        "max_alias_next_delta_variance": max(
            (row["aliasing"]["max_next_delta_variance"] for row in level_rows),
            default=0.0,
        ),
        "design_smoke_pass": bool(level_rows)
        and all(row["invariant_pass"] for row in level_rows),
    }


def main() -> None:
    args = build_parser().parse_args()
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    rows = [
        validate_level(
            SokobanGoalDirectedSDModel(level),
            max_depth=args.max_depth,
            max_states=args.max_states_per_level,
            alias_round_decimals=args.alias_round_decimals,
            sample_limit=args.sample_limit,
        )
        for level in levels
    ]
    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_sd_design_validation_smoke_v1",
        },
        "config": {
            "levels": os.path.relpath(args.levels, ROOT) if os.path.isabs(args.levels) else args.levels,
            "max_levels": args.max_levels,
            "max_depth": args.max_depth,
            "max_states_per_level": args.max_states_per_level,
            "alias_round_decimals": args.alias_round_decimals,
            "sample_limit": args.sample_limit,
        },
        "design_contract": {
            "stocks": STOCK_NAMES,
            "flows": FLOW_NAMES,
            "incidence_matrix": INCIDENCE_MATRIX.astype(int).tolist(),
            "action_scope": "legal_single_push_option",
            "learned_transition_provider_required_for_formal_sdmcts": True,
            "current_provider_is_diagnostic_only": True,
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
