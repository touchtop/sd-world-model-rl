#!/usr/bin/env python3
"""Audit stock-flow diagnostics for the frozen Sokoban SD baseline.

This script does not train a model. It reconstructs exact single-push
diagnostic transitions, audits the six-stock proxy and diagnostic flow labels,
and optionally aggregates frozen-baseline prediction audits emitted by
``train_sokoban_flow_smoke.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES
from src.sokoban.sd_model import SokobanPushAction
from train_sokoban_flow_smoke import FLOW_PAIRS, action_features
from validate_sokoban_sd_design import (
    FLOW_NAMES,
    INCIDENCE_MATRIX,
    STOCK_NAMES,
    diagnostic_flows,
    iter_states,
    stock_proxy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used by the frozen baseline.",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=30)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=("level", "transition"), default="level")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alias-round-decimals", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-limit-per-bucket", type=int, default=4)
    parser.add_argument("--baseline-files", nargs="*", default=[])
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json",
    )
    return parser


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _summary(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _finite_summary(values: Iterable[float]) -> dict[str, Any]:
    return _summary(v for v in values if math.isfinite(float(v)))


def _nonzero_fraction(values: Iterable[float], eps: float = 1e-12) -> float | None:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return None
    return float((np.abs(arr) > eps).mean())


def stock_proxy_components(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
) -> dict[str, Any]:
    """Expose the raw feature terms used to build the current six stocks."""

    features = model.abstract_features(state)
    n_boxes = max(model.level.n_boxes, 1)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    floor_count = max(len(model.floor_cells), 1)
    board_scale = max(model.level.height + model.level.width, 1)

    if np.isfinite(features.matching_lower_bound):
        matching_pressure = _clamp01(features.matching_lower_bound / (n_boxes * board_scale))
    else:
        matching_pressure = 1.0

    unresolved_count_ratio = _safe_div(n_boxes - features.boxes_on_goals, n_boxes)
    structural_deadlock_pressure = _safe_div(
        features.dead_square_boxes + features.frozen_boxes,
        n_boxes,
    )
    if features.matching_deadlock:
        structural_deadlock_pressure = 1.0

    stocks = stock_proxy(model, state).to_dict()
    return {
        "stocks": {name: float(stocks[name]) for name in STOCK_NAMES},
        "raw_features": {
            "reachable_count": int(features.reachable_count),
            "legal_push_count": int(features.legal_push_count),
            "viable_push_count": int(features.viable_push_count),
            "boxes_on_goals": int(features.boxes_on_goals),
            "matching_lower_bound": _json_number(features.matching_lower_bound),
            "matching_deadlock": bool(features.matching_deadlock),
            "dead_square_boxes": int(features.dead_square_boxes),
            "frozen_boxes": int(features.frozen_boxes),
            "solved": bool(features.solved),
        },
        "normalizers": {
            "n_boxes": int(n_boxes),
            "max_pushes": int(max_pushes),
            "floor_count": int(floor_count),
            "board_scale": int(board_scale),
        },
        "component_terms": {
            "goal_completion_boxes_on_goals_ratio": _safe_div(features.boxes_on_goals, n_boxes),
            "unresolved_count_ratio": unresolved_count_ratio,
            "matching_pressure": matching_pressure,
            "unresolved_box_pressure": stocks["unresolved_box_pressure"],
            "mobility_viable_push_ratio": _safe_div(features.viable_push_count, max_pushes),
            "reachable_capacity_reachable_floor_ratio": _safe_div(
                features.reachable_count,
                floor_count,
            ),
            "congestion_legal_push_deficit_ratio": 1.0
            - _safe_div(features.legal_push_count, max_pushes),
            "irreversibility_structural_deadlock_pressure": structural_deadlock_pressure,
        },
    }


def action_to_dict(action: SokobanPushAction) -> dict[str, Any]:
    return {
        "box_index": int(action.box_index),
        "direction": int(action.direction),
        "direction_name": ACTION_NAMES[action.direction],
        "flat_index": int(action.flat_index),
    }


def state_signature(state: SokobanState) -> dict[str, Any]:
    return {
        "player": [int(state.player[0]), int(state.player[1])],
        "boxes": [[int(cell[0]), int(cell[1])] for cell in state.boxes],
    }


def collect_diagnostic_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for level in levels:
        model = SokobanGoalDirectedSDModel(level)
        for state, depth in iter_states(
            model,
            max_depth=args.max_depth,
            max_states=args.max_states_per_level,
        ):
            before = stock_proxy_components(model, state)
            x = np.array([before["stocks"][name] for name in STOCK_NAMES], dtype=np.float64)
            legal_pushes = model.legal_pushes(state)
            state_rows.append(
                {
                    "level_id": level.level_id,
                    "depth": int(depth),
                    "state": state_signature(state),
                    "stock": x,
                    "components": before,
                    "legal_push_count_from_iterator": int(len(legal_pushes)),
                }
            )

            for action in legal_pushes:
                next_state = model.step_state(state, action)
                after = stock_proxy_components(model, next_state)
                next_x = np.array([after["stocks"][name] for name in STOCK_NAMES], dtype=np.float64)
                delta = next_x - x
                flows = diagnostic_flows(delta)
                reconstructed = INCIDENCE_MATRIX @ flows
                transition_rows.append(
                    {
                        "level_id": level.level_id,
                        "depth": int(depth),
                        "state": state_signature(state),
                        "action": action_to_dict(action),
                        "action_features": action_features(model, state, action).astype(np.float64),
                        "stock_before": x,
                        "stock_after": next_x,
                        "stock_delta": delta,
                        "diagnostic_flows": flows,
                        "balance_l1": float(np.abs(reconstructed - delta).sum()),
                        "components_before": before,
                        "components_after": after,
                    }
                )
    return state_rows, transition_rows


def split_indices(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    valid_fraction: float,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.arange(len(rows))
    rng = np.random.RandomState(seed)
    if split_mode == "transition":
        rng.shuffle(indices)
        valid_size = max(1, int(round(len(rows) * valid_fraction))) if len(rows) > 1 else 0
        valid_idx = indices[:valid_size]
        train_idx = indices[valid_size:]
        return train_idx, valid_idx, {
            "split_mode": split_mode,
            "train_levels": sorted({rows[int(i)]["level_id"] for i in train_idx}),
            "valid_levels": sorted({rows[int(i)]["level_id"] for i in valid_idx}),
        }

    levels = np.array(sorted({row["level_id"] for row in rows}), dtype=object)
    rng.shuffle(levels)
    valid_level_count = max(1, int(round(len(levels) * valid_fraction))) if len(levels) > 1 else 0
    valid_levels = set(levels[:valid_level_count].tolist())
    train_idx = np.array(
        [idx for idx, row in enumerate(rows) if row["level_id"] not in valid_levels],
        dtype=np.int64,
    )
    valid_idx = np.array(
        [idx for idx, row in enumerate(rows) if row["level_id"] in valid_levels],
        dtype=np.int64,
    )
    return train_idx, valid_idx, {
        "split_mode": split_mode,
        "train_levels": sorted({rows[int(i)]["level_id"] for i in train_idx}),
        "valid_levels": sorted(valid_levels),
    }


def named_array(values: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(names)}


def compact_transition(row: dict[str, Any]) -> dict[str, Any]:
    before = row["components_before"]
    after = row["components_after"]
    return {
        "level_id": row["level_id"],
        "depth": row["depth"],
        "state": row["state"],
        "action": row["action"],
        "stock_before": named_array(row["stock_before"], STOCK_NAMES),
        "stock_delta": named_array(row["stock_delta"], STOCK_NAMES),
        "stock_after": named_array(row["stock_after"], STOCK_NAMES),
        "diagnostic_flows": named_array(row["diagnostic_flows"], FLOW_NAMES),
        "raw_before": before["raw_features"],
        "raw_after": after["raw_features"],
        "component_terms_before": before["component_terms"],
        "component_terms_after": after["component_terms"],
    }


def stock_component_audit(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stock_values = {name: [] for name in STOCK_NAMES}
    raw_values: dict[str, list[float]] = defaultdict(list)
    component_values: dict[str, list[float]] = defaultdict(list)
    normalizer_values: dict[str, list[float]] = defaultdict(list)
    risk_counts: Counter[str] = Counter()

    for row in state_rows:
        components = row["components"]
        for idx, name in enumerate(STOCK_NAMES):
            stock_values[name].append(float(row["stock"][idx]))
        for key, value in components["raw_features"].items():
            if isinstance(value, bool):
                raw_values[key].append(float(value))
            elif value is not None:
                raw_values[key].append(float(value))
        for key, value in components["component_terms"].items():
            component_values[key].append(float(value))
        for key, value in components["normalizers"].items():
            normalizer_values[key].append(float(value))

        raw = components["raw_features"]
        if raw["matching_deadlock"] or raw["dead_square_boxes"] > 0 or raw["frozen_boxes"] > 0:
            risk_counts["structural_deadlock_evidence"] += 1
        elif raw["legal_push_count"] == 0 and not raw["solved"]:
            risk_counts["no_legal_push_unsolved"] += 1
        elif raw["viable_push_count"] == 0 and not raw["solved"]:
            risk_counts["no_viable_push_unsolved"] += 1
        else:
            risk_counts["none"] += 1

    return {
        "states": len(state_rows),
        "stock_value_summary": {name: _summary(values) for name, values in stock_values.items()},
        "raw_feature_summary": {
            key: (_finite_summary(values) if key == "matching_lower_bound" else _summary(values))
            for key, values in raw_values.items()
        },
        "component_term_summary": {
            key: _summary(values) for key, values in component_values.items()
        },
        "normalizer_summary": {
            key: _summary(values) for key, values in normalizer_values.items()
        },
        "risk_proxy_state_counts": dict(risk_counts),
        "proxy_notes": {
            "reachable_capacity": (
                "reachable_count / floor_count; the denominator is static board floor area, "
                "so this stock can be insensitive to local push consequences when player "
                "reachability changes in small connected regions."
            ),
            "congestion": (
                "1 - legal_push_count / (n_boxes * 4); this is bounded by the number of "
                "currently legal pushes and can remain high even when local congestion "
                "mechanisms differ structurally."
            ),
        },
    }


def transition_distribution_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stock_delta = {name: [] for name in STOCK_NAMES}
    flow_values = {name: [] for name in FLOW_NAMES}
    flow_index = {name: idx for idx, name in enumerate(FLOW_NAMES)}
    for row in rows:
        for idx, name in enumerate(STOCK_NAMES):
            stock_delta[name].append(float(row["stock_delta"][idx]))
        for idx, name in enumerate(FLOW_NAMES):
            flow_values[name].append(float(row["diagnostic_flows"][idx]))

    pair_rows: dict[str, Any] = {}
    for left, right in FLOW_PAIRS:
        left_values = np.asarray(
            [row["diagnostic_flows"][flow_index[left]] for row in rows],
            dtype=np.float64,
        )
        right_values = np.asarray(
            [row["diagnostic_flows"][flow_index[right]] for row in rows],
            dtype=np.float64,
        )
        if left_values.size:
            coactive = np.logical_and(left_values > 1e-12, right_values > 1e-12)
            min_pair = np.minimum(left_values, right_values)
            pair_rows[f"{left}__{right}"] = {
                "target_coactive_fraction_gt_1e-12": float(coactive.mean()),
                "target_mean_min_pair_flow": float(min_pair.mean()),
            }

    balance_values = [row["balance_l1"] for row in rows]
    return {
        "transitions": len(rows),
        "stock_delta_summary": {
            name: {
                **_summary(values),
                "nonzero_fraction": _nonzero_fraction(values),
                "positive_fraction": float((np.asarray(values) > 1e-12).mean()) if values else None,
                "negative_fraction": float((np.asarray(values) < -1e-12).mean()) if values else None,
            }
            for name, values in stock_delta.items()
        },
        "flow_target_summary": {
            name: {
                **_summary(values),
                "nonzero_fraction": _nonzero_fraction(values),
            }
            for name, values in flow_values.items()
        },
        "flow_pair_target_coactivation": pair_rows,
        "incidence_balance": {
            "max_l1": max(balance_values) if balance_values else 0.0,
            "mean_l1": float(np.mean(balance_values)) if balance_values else 0.0,
            "exact_balance_fraction": float((np.asarray(balance_values) <= 1e-12).mean())
            if balance_values
            else None,
        },
    }


def alias_bucket_audit(
    rows: list[dict[str, Any]],
    *,
    round_decimals: int,
    top_k: int,
    sample_limit_per_bucket: int,
    include_action: bool,
) -> dict[str, Any]:
    buckets: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(np.round(row["stock_before"], round_decimals).tolist())
        if include_action:
            key = key + tuple(np.round(row["action_features"], round_decimals).tolist())
        buckets[key].append(row)

    bucket_rows: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        if len(bucket) < 2:
            continue
        deltas = np.stack([row["stock_delta"] for row in bucket], axis=0)
        per_stock_variance = np.var(deltas, axis=0)
        per_stock_range = np.max(deltas, axis=0) - np.min(deltas, axis=0)
        bucket_rows.append(
            {
                "bucket_key": [float(value) for value in key],
                "count": int(len(bucket)),
                "level_count": int(len({row["level_id"] for row in bucket})),
                "next_delta_variance_sum": float(per_stock_variance.sum()),
                "next_delta_abs_range_sum": float(np.abs(per_stock_range).sum()),
                "per_stock_next_delta_variance": {
                    name: float(per_stock_variance[idx])
                    for idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_next_delta_range": {
                    name: float(per_stock_range[idx])
                    for idx, name in enumerate(STOCK_NAMES)
                },
                "mean_next_delta": {
                    name: float(deltas[:, idx].mean())
                    for idx, name in enumerate(STOCK_NAMES)
                },
                "examples": [
                    compact_transition(sample)
                    for sample in bucket[:sample_limit_per_bucket]
                ],
            }
        )
    bucket_rows.sort(key=lambda row: row["next_delta_variance_sum"], reverse=True)
    variances = [row["next_delta_variance_sum"] for row in bucket_rows]
    return {
        "round_decimals": round_decimals,
        "include_action": include_action,
        "bucket_count": len(buckets),
        "aliased_bucket_count": len(bucket_rows),
        "max_next_delta_variance": max(variances) if variances else 0.0,
        "mean_next_delta_variance": float(np.mean(variances)) if variances else 0.0,
        "top_buckets": bucket_rows[:top_k],
    }


def _get_path(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _collect_numeric_leaves(value: Any, prefix: tuple[str, ...], out: dict[str, list[float]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_numeric_leaves(child, prefix + (str(key),), out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            out[".".join(prefix)].append(float(value))


def summarize_numeric_tree(rows: Iterable[Any]) -> dict[str, Any]:
    leaves: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        _collect_numeric_leaves(row, tuple(), leaves)
    return {key: _summary(values) for key, values in sorted(leaves.items())}


def aggregate_baseline_prediction_audit(paths: list[str]) -> dict[str, Any]:
    if not paths:
        return {"files": [], "available": False}

    runs = []
    for path in paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        runs.append((path, run))

    valid_dimensional = [
        _get_path(run, ("audit", "valid_dimensional"))
        for _path, run in runs
        if _get_path(run, ("audit", "valid_dimensional")) is not None
    ]
    valid_rollout_step2 = [
        _get_path(run, ("metrics", "valid_rollout", "per_stock_mae_by_step", "step_2"))
        for _path, run in runs
        if _get_path(run, ("metrics", "valid_rollout", "per_stock_mae_by_step", "step_2"))
        is not None
    ]

    baseline_contracts = [_get_path(run, ("baseline_contract",)) for _path, run in runs]
    config_rows = [_get_path(run, ("config",)) for _path, run in runs]

    return {
        "available": True,
        "files": [
            os.path.relpath(path, ROOT) if Path(path).is_absolute() else path
            for path, _run in runs
        ],
        "seeds": [_get_path(run, ("config", "seed")) for _path, run in runs],
        "all_frozen_minimal_baseline": all(
            bool(contract and contract.get("is_frozen_minimal_baseline"))
            for contract in baseline_contracts
        ),
        "config_digest": {
            "baseline_id": _get_path(runs[0][1], ("baseline_contract", "baseline_id")),
            "model_kind": _get_path(runs[0][1], ("config", "model_kind")),
            "feature_mode": _get_path(runs[0][1], ("config", "feature_mode")),
            "stock_loss_preset": _get_path(runs[0][1], ("config", "stock_loss_preset")),
            "rollout_depth": _get_path(runs[0][1], ("config", "rollout_depth")),
            "rollout_loss_weight": _get_path(runs[0][1], ("config", "rollout_loss_weight")),
            "split_mode": _get_path(runs[0][1], ("config", "split_mode")),
            "max_levels": _get_path(runs[0][1], ("config", "max_levels")),
            "max_depth": _get_path(runs[0][1], ("config", "max_depth")),
            "max_states_per_level": _get_path(runs[0][1], ("config", "max_states_per_level")),
        },
        "config_rows": config_rows,
        "valid_rollout_step2_per_stock_mae_summary": summarize_numeric_tree(valid_rollout_step2),
        "valid_dimensional_numeric_summary": summarize_numeric_tree(valid_dimensional),
    }


def per_seed_exact_audits(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in args.seeds:
        train_idx, valid_idx, split = split_indices(
            rows,
            seed=seed,
            valid_fraction=args.valid_fraction,
            split_mode=args.split_mode,
        )
        valid_rows = [rows[int(idx)] for idx in valid_idx]
        out[f"seed_{seed}"] = {
            "split": split,
            "train_transitions": int(len(train_idx)),
            "valid_transitions": int(len(valid_idx)),
            "valid_transition_distribution": transition_distribution_audit(valid_rows),
            "valid_stock_aliasing": alias_bucket_audit(
                valid_rows,
                round_decimals=args.alias_round_decimals,
                top_k=args.top_k,
                sample_limit_per_bucket=args.sample_limit_per_bucket,
                include_action=False,
            ),
            "valid_stock_action_aliasing": alias_bucket_audit(
                valid_rows,
                round_decimals=args.alias_round_decimals,
                top_k=args.top_k,
                sample_limit_per_bucket=args.sample_limit_per_bucket,
                include_action=True,
            ),
        }
    return out


def interpret_diagnostics(
    exact_transition_audit: dict[str, Any],
    seed_audits: dict[str, Any],
    baseline_audit: dict[str, Any],
) -> dict[str, Any]:
    stock_delta = exact_transition_audit["stock_delta_summary"]
    rc_nonzero = stock_delta["reachable_capacity"]["nonzero_fraction"]
    c_nonzero = stock_delta["congestion"]["nonzero_fraction"]
    max_stock_action_alias = max(
        (
            row["valid_stock_action_aliasing"]["max_next_delta_variance"]
            for row in seed_audits.values()
        ),
        default=0.0,
    )
    max_stock_alias = max(
        (
            row["valid_stock_aliasing"]["max_next_delta_variance"]
            for row in seed_audits.values()
        ),
        default=0.0,
    )
    balance_max_l1 = exact_transition_audit["incidence_balance"]["max_l1"]

    baseline_summary = baseline_audit.get("valid_dimensional_numeric_summary", {})
    clip_any = baseline_summary.get("clip_audit.any_stock_clipped_fraction", {}).get("mean")
    clip_correction = baseline_summary.get("clip_audit.mean_abs_clip_correction", {}).get("mean")

    flow_pair_summary = baseline_summary
    pred_pair_coactivation = {
        key: value.get("mean")
        for key, value in flow_pair_summary.items()
        if key.endswith("pred_coactive_fraction_gt_1e-6")
    }
    high_pred_pair_coactivation = {
        key: value
        for key, value in pred_pair_coactivation.items()
        if value is not None and value > 0.5
    }

    findings: list[str] = []
    if balance_max_l1 <= 1e-12:
        findings.append(
            "diagnostic flow labels exactly reconstruct observed stock deltas under the fixed incidence matrix."
        )
    if clip_any is not None and clip_any > 0.5:
        findings.append(
            "frozen-baseline prediction relies heavily on stock clipping; pre-clip dynamics must be audited before SD-MCTS."
        )
    if max_stock_action_alias > 0.05:
        findings.append(
            "stock-action alias buckets still contain high next-delta variance, so the six-stock abstraction is not yet Markov-sufficient."
        )
    if high_pred_pair_coactivation:
        findings.append(
            "learned softplus flows co-activate opposite flow pairs much more often than diagnostic labels, suggesting compensating flows and clipping are interacting."
        )
    if rc_nonzero is not None and c_nonzero is not None:
        findings.append(
            "reachable_capacity and congestion change on a substantial subset of exact transitions, but both are derived from coarse global normalizers."
        )

    return {
        "primary_findings": findings,
        "evidence": {
            "incidence_balance_max_l1": balance_max_l1,
            "baseline_clip_any_stock_fraction_mean": clip_any,
            "baseline_mean_abs_clip_correction_mean": clip_correction,
            "max_valid_stock_alias_next_delta_variance": max_stock_alias,
            "max_valid_stock_action_alias_next_delta_variance": max_stock_action_alias,
            "reachable_capacity_delta_nonzero_fraction": rc_nonzero,
            "congestion_delta_nonzero_fraction": c_nonzero,
            "high_predicted_opposite_flow_pair_coactivation": high_pred_pair_coactivation,
        },
        "recommended_next_tests": [
            "keep the frozen baseline unchanged as the comparison point",
            "run one proxy revision at a time for reachable_capacity and congestion",
            "test bounded or paired-net-flow parameterizations to reduce opposite-flow compensation",
            "add alias-aware auxiliary features only after the exact alias buckets are documented",
            "defer learned-transition-provider SD-MCTS until clipping and aliasing audits improve",
        ],
    }


def main() -> None:
    args = build_parser().parse_args()
    state_rows, transition_rows = collect_diagnostic_rows(args)

    exact_state_audit = stock_component_audit(state_rows)
    exact_transition_audit = transition_distribution_audit(transition_rows)
    seed_audits = per_seed_exact_audits(transition_rows, args)
    baseline_audit = aggregate_baseline_prediction_audit(args.baseline_files)
    interpretation = interpret_diagnostics(exact_transition_audit, seed_audits, baseline_audit)

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_stock_flow_diagnostics_v1",
        },
        "config": {
            "levels": os.path.relpath(args.levels, ROOT) if os.path.isabs(args.levels) else args.levels,
            "max_levels": args.max_levels,
            "max_depth": args.max_depth,
            "max_states_per_level": args.max_states_per_level,
            "valid_fraction": args.valid_fraction,
            "split_mode": args.split_mode,
            "seeds": args.seeds,
            "alias_round_decimals": args.alias_round_decimals,
            "top_k": args.top_k,
            "sample_limit_per_bucket": args.sample_limit_per_bucket,
        },
        "design_scope": {
            "sd_mcts_enabled": False,
            "transition_labels_from": "exact_single_push_diagnostic_provider",
            "learned_transition_provider_used_for_sdmcts": False,
            "audit_only": True,
        },
        "design_contract": {
            "stocks": list(STOCK_NAMES),
            "flows": list(FLOW_NAMES),
            "flow_pairs": [list(pair) for pair in FLOW_PAIRS],
            "incidence_matrix": INCIDENCE_MATRIX.astype(int).tolist(),
        },
        "dataset": {
            "states": len(state_rows),
            "transitions": len(transition_rows),
            "levels": len({row["level_id"] for row in transition_rows}),
        },
        "exact_provider_state_proxy_audit": exact_state_audit,
        "exact_provider_transition_audit": exact_transition_audit,
        "per_seed_exact_provider_audits": seed_audits,
        "frozen_baseline_prediction_audit": baseline_audit,
        "diagnostic_interpretation": interpretation,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(interpretation, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
