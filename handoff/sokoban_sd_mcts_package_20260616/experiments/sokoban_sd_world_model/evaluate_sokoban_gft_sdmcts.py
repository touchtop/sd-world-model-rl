#!/usr/bin/env python3
"""Closed-loop two-layer GFT-SD-MCTS evaluation for Sokoban.

This entry point preserves the exact-grid SD-MCTS evaluator as the baseline and
adds an independent GFT-SD resource layer on top of the audited grid ledger
state.  The GFT layer is opt-in through this script and does not change
``evaluate_sokoban_grid_sdmcts.py`` or its current best results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.sokoban import (
    DEFAULT_GFT_PRIOR_WEIGHTS,
    DEFAULT_GFT_SOFT_DEADLOCK_WEIGHTS,
    DEFAULT_GFT_TRANSPORT_COST_WEIGHTS,
    SokobanGFTSDModel,
    load_boxoban_levels,
)

from evaluate_sokoban_grid_sdmcts import (
    CandidateAction,
    GridSDMCTSPlanner,
    TvaCheckpoint,
    _rel,
    action_to_dict,
    aggregate,
    build_parser as build_grid_parser,
    check_independent_gates,
    load_tva_checkpoint as load_grid_tva_checkpoint,
)


@dataclass
class GFTTvaCheckpoint:
    path: str
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]
    feature_dim: int
    value_scale: dict[str, float]
    target_mode: str
    model_type: str
    value_provider_id: str
    feature_contract: str
    checkpoint_contract: dict[str, Any]
    weights: np.ndarray | None = None
    mlp: dict[str, Any] | None = None
    risk_guard: dict[str, Any] | None = None


def normalize_gft_tva_target_mode(mode: str) -> str:
    """Canonicalize historical GFT-TVA target-mode names."""

    if mode == "value":
        return "direct_value"
    return str(mode)


def load_tva_checkpoint(path: str | Path) -> TvaCheckpoint | GFTTvaCheckpoint:
    """Load either the legacy grid TVA checkpoint or the GFT-TVA checkpoint."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    feature_contract = payload.get("feature_contract", {})
    if feature_contract.get("contract") != "gft_tva_feature_vector_v1":
        return load_grid_tva_checkpoint(path)

    model_type = str(payload.get("model_type", "ridge"))
    feature_names = [str(name) for name in feature_contract["feature_names"]]
    base = {
        "path": _rel(path),
        "feature_names": feature_names,
        "feature_dim": int(feature_contract["feature_dim"]),
        "value_scale": {key: float(value) for key, value in payload.get("value_scale", {}).items()},
        "target_mode": normalize_gft_tva_target_mode(str(payload.get("target_mode", "gft_sd_residual"))),
        "model_type": model_type,
        "value_provider_id": str(payload.get("value_provider_id", "gft_tva_v1")),
        "feature_contract": str(feature_contract.get("contract")),
        "checkpoint_contract": dict(payload.get("checkpoint_contract", {})),
    }
    if model_type == "ridge":
        ridge = payload["ridge"]
        return GFTTvaCheckpoint(
            mean=np.asarray(ridge["mean"], dtype=np.float64),
            std=np.asarray(ridge["std"], dtype=np.float64),
            weights=np.asarray(ridge["weights"], dtype=np.float64),
            **base,
        )
    if model_type == "mlp_guarded_v1":
        normalization = payload["normalization"]
        return GFTTvaCheckpoint(
            mean=np.asarray(normalization["mean"], dtype=np.float64),
            std=np.asarray(normalization["std"], dtype=np.float64),
            mlp=dict(payload["mlp"]),
            risk_guard=dict(payload.get("risk_guard", {})),
            **base,
        )
    raise ValueError(f"Unsupported GFT-TVA model_type: {model_type}")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return float(1.0 / (1.0 + z))
    z = math.exp(value)
    return float(z / (1.0 + z))


def _gft_tva_anchor_value(checkpoint: GFTTvaCheckpoint, model: SokobanGFTSDModel, state) -> float:
    contract = dict(checkpoint.checkpoint_contract or {})
    leaf_mode = str(contract.get("gft_sd_leaf_mode", "discounted_progress"))
    weights = contract.get("gft_sd_leaf_weights", {})
    gamma = float(checkpoint.value_scale.get("gamma", 0.95))
    goal_value = float(checkpoint.value_scale.get("goal_value", 10.0))
    push_cost = float(checkpoint.value_scale.get("push_cost", 0.05))
    deadend_value = float(checkpoint.value_scale.get("deadend_value", -10.0))
    if leaf_mode == "discounted_progress":
        value = discounted_progress_gft_sd_value(
            model,
            state,
            gamma=gamma,
            goal_value=goal_value,
            push_cost=push_cost,
            deadend_value=deadend_value,
            weights={key: float(value) for key, value in dict(weights).items()},
        )
    else:
        value = projected_gft_sd_value(
            model,
            state,
            deadend_value=deadend_value,
            goal_value=goal_value,
            weights={key: float(value) for key, value in dict(weights).items()},
        )
    guard = dict(checkpoint.risk_guard or {})
    value += float(guard.get("anchor_bias", 0.0))
    low = min(deadend_value, goal_value)
    high = max(deadend_value, goal_value)
    return float(max(low, min(high, value)))


def predict_gft_tva_details(checkpoint: GFTTvaCheckpoint, model: SokobanGFTSDModel, state) -> dict[str, float]:
    feature_dict = model.gft_tva_feature_dict(state)["features"]
    features = np.asarray([float(feature_dict[name]) for name in checkpoint.feature_names], dtype=np.float64)
    if features.shape[0] != checkpoint.feature_dim:
        raise ValueError(
            f"GFT-TVA feature dimension mismatch: checkpoint={checkpoint.feature_dim}, current={features.shape[0]}"
        )
    z = (features - checkpoint.mean) / checkpoint.std
    feature_max_abs_z = float(np.max(np.abs(z))) if z.size else 0.0
    feature_l2_z = float(np.linalg.norm(z)) if z.size else 0.0
    if checkpoint.model_type == "ridge":
        if checkpoint.weights is None:
            raise ValueError("Ridge GFT-TVA checkpoint has no weights.")
        design = np.concatenate([z, np.ones(1, dtype=np.float64)], axis=0)
        value = float(design @ checkpoint.weights)
        return {
            "prediction": value,
            "value_head": value,
            "risk_probability": 0.0,
            "risk_logit": 0.0,
            "anchor_value": 0.0,
            "anchor_weight": 0.0,
            "guarded_value": value,
            "feature_max_abs_z": feature_max_abs_z,
            "feature_l2_z": feature_l2_z,
        }
    if checkpoint.model_type != "mlp_guarded_v1":
        raise ValueError(f"Unsupported GFT-TVA model_type: {checkpoint.model_type}")
    if checkpoint.mlp is None:
        raise ValueError("MLP GFT-TVA checkpoint has no network weights.")

    h = z
    for layer in checkpoint.mlp["hidden_layers"]:
        weight = np.asarray(layer["weight"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        h = np.maximum(h @ weight.T + bias, 0.0)
    value_weight = np.asarray(checkpoint.mlp["value_head"]["weight"], dtype=np.float64)
    value_bias = np.asarray(checkpoint.mlp["value_head"]["bias"], dtype=np.float64)
    risk_weight = np.asarray(checkpoint.mlp["risk_head"]["weight"], dtype=np.float64)
    risk_bias = np.asarray(checkpoint.mlp["risk_head"]["bias"], dtype=np.float64)
    value_head = float((h @ value_weight.T + value_bias).reshape(-1)[0])
    risk_logit = float((h @ risk_weight.T + risk_bias).reshape(-1)[0])
    risk_probability = _sigmoid(risk_logit)

    guard = dict(checkpoint.risk_guard or {})
    risk_penalty = float(guard.get("risk_penalty", 0.0))
    guarded_value = value_head - risk_penalty * risk_probability
    anchor_weight = float(guard.get("anchor_weight", 0.0))
    anchor_value = _gft_tva_anchor_value(checkpoint, model, state) if anchor_weight > 0.0 else 0.0
    prediction = (1.0 - anchor_weight) * guarded_value + anchor_weight * anchor_value
    if bool(guard.get("clip_to_value_scale", True)):
        low = min(
            float(checkpoint.value_scale.get("deadend_value", -10.0)),
            float(checkpoint.value_scale.get("goal_value", 10.0)),
        )
        high = max(
            float(checkpoint.value_scale.get("deadend_value", -10.0)),
            float(checkpoint.value_scale.get("goal_value", 10.0)),
        )
        prediction = float(max(low, min(high, prediction)))
    return {
        "prediction": float(prediction),
        "value_head": float(value_head),
        "risk_probability": float(risk_probability),
        "risk_logit": float(risk_logit),
        "anchor_value": float(anchor_value),
        "anchor_weight": float(anchor_weight),
        "guarded_value": float(guarded_value),
        "feature_max_abs_z": feature_max_abs_z,
        "feature_l2_z": feature_l2_z,
    }


def predict_gft_tva(checkpoint: GFTTvaCheckpoint, model: SokobanGFTSDModel, state) -> float:
    return float(predict_gft_tva_details(checkpoint, model, state)["prediction"])


def projected_gft_sd_value(
    model: SokobanGFTSDModel,
    state,
    *,
    deadend_value: float,
    goal_value: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Project explicit ``V_SD(z)`` onto the planner's return scale."""

    penalty = max(abs(float(deadend_value)), abs(float(goal_value)), 1.0)
    raw = float(model.gft_sd_value(state, weights=weights, nonfinite_progress_penalty=penalty)["value"])
    if not math.isfinite(raw):
        return float(deadend_value)
    low = min(float(deadend_value), float(goal_value))
    high = max(float(deadend_value), float(goal_value))
    return float(max(low, min(high, raw)))


def has_nonzero_gft_leaf_weights(weights: dict[str, float] | None) -> bool:
    return any(abs(float(value)) > 1e-12 for value in dict(weights or {}).values())


def discounted_progress_gft_sd_value(
    model: SokobanGFTSDModel,
    state,
    *,
    gamma: float,
    goal_value: float,
    push_cost: float,
    deadend_value: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Planner-scale ``V_SD``: discounted transport progress plus resource terms."""

    p = model.gft_transport_progress_cached(state)
    if not math.isfinite(p):
        base = float(deadend_value)
    elif gamma == 1.0:
        base = float(goal_value) - float(push_cost) * p
    else:
        discount = float(gamma) ** p
        base = discount * float(goal_value) - float(push_cost) * (1.0 - discount) / (1.0 - float(gamma))

    weights = dict(weights or {})
    if not has_nonzero_gft_leaf_weights(weights):
        return float(base)

    features = model.gft_tva_feature_dict(state)["features"]
    max_pushes = max(model.level.n_boxes * 4, 1)
    n_boxes = max(model.level.n_boxes, 1)
    adjustment = (
        float(weights.get("mobility", 0.0)) * float(features["M"]) / max_pushes
        + float(weights.get("access", 0.0)) * float(features["A"]) / max_pushes
        + float(weights.get("capacity", 0.0)) * float(features["K"]) / n_boxes
        - float(weights.get("deadlock", 0.0)) * float(features["D"]) / n_boxes
        - float(weights.get("ordering", 0.0)) * float(features["Q"]) / n_boxes
        + float(weights.get("commitment", 0.0)) * float(features["C"])
    )
    raw = float(base + adjustment)
    low = min(float(deadend_value), float(goal_value))
    high = max(float(deadend_value), float(goal_value))
    return float(max(low, min(high, raw)))


class GFTSDMCTSPlanner(GridSDMCTSPlanner):
    """Grid MCTS planner with a GFT-SD action relevance prior."""

    def __init__(
        self,
        *args,
        gft_prior: str,
        gft_prior_temperature: float,
        gft_prior_uniform_mix: float,
        gft_prior_weights: dict[str, float],
        gft_sd_leaf_mode: str,
        gft_sd_leaf_weights: dict[str, float],
        tva_value_blend: float = 1.0,
        tva_dynamic_blend: bool = False,
        tva_dynamic_high_risk_threshold: float = 0.9,
        tva_dynamic_negative_delta_threshold: float = 1.0,
        tva_dynamic_high_risk_blend: float = 0.75,
        tva_dynamic_low_risk_threshold: float = 0.3,
        tva_dynamic_positive_delta_threshold: float = 0.25,
        tva_dynamic_positive_blend: float = -1.0,
        tva_dynamic_revisit_penalty: float = 0.0,
        tva_dynamic_prior_penalty_weight: float = 0.0,
        tva_dynamic_prior_penalty_cap: float = 4.0,
        tva_dynamic_leaf_penalty_weight: float = 0.0,
        tva_dynamic_leaf_penalty_cap: float = 4.0,
        tva_dynamic_root_select: bool = False,
        tva_dynamic_root_select_margin: float = 0.0,
        tva_state_gated_blend: bool = False,
        tva_state_gated_default_blend: float = 0.0,
        tva_state_gated_positive_blend: float = -1.0,
        tva_state_gated_positive_risk_max: float = 0.3,
        tva_state_gated_positive_delta_min: float = 0.25,
        tva_state_gated_negative_blend: float = 0.0,
        tva_state_gated_negative_risk_min: float = 0.95,
        tva_state_gated_negative_delta_min: float = 1.0,
        tva_state_gated_ood_z_threshold: float = -1.0,
        tva_state_gated_ood_blend: float = 0.0,
        tva_transport_phase_blend: float = -1.0,
        tva_transport_phase_high_progress_min: float = -1.0,
        tva_transport_phase_low_progress_max: float = -1.0,
        tva_transport_phase_commitment_min: float = 0.25,
        tva_transport_phase_max_closed_loop_step: int = -1,
        tva_transport_phase_require_root_eligible: bool = False,
        closed_loop_step: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gft_prior = gft_prior
        self.gft_prior_temperature = max(float(gft_prior_temperature), 1e-9)
        self.gft_prior_uniform_mix = min(max(float(gft_prior_uniform_mix), 0.0), 1.0)
        self.gft_prior_weights = dict(gft_prior_weights)
        self.gft_sd_leaf_mode = str(gft_sd_leaf_mode)
        self.gft_sd_leaf_weights = dict(gft_sd_leaf_weights)
        self.tva_value_blend = min(max(float(tva_value_blend), 0.0), 1.0)
        self.tva_dynamic_blend = bool(tva_dynamic_blend)
        self.tva_dynamic_high_risk_threshold = min(max(float(tva_dynamic_high_risk_threshold), 0.0), 1.0)
        self.tva_dynamic_negative_delta_threshold = max(float(tva_dynamic_negative_delta_threshold), 0.0)
        self.tva_dynamic_high_risk_blend = min(max(float(tva_dynamic_high_risk_blend), 0.0), 1.0)
        self.tva_dynamic_low_risk_threshold = min(max(float(tva_dynamic_low_risk_threshold), 0.0), 1.0)
        self.tva_dynamic_positive_delta_threshold = max(float(tva_dynamic_positive_delta_threshold), 0.0)
        self.tva_dynamic_positive_blend = (
            None
            if float(tva_dynamic_positive_blend) < 0.0
            else min(max(float(tva_dynamic_positive_blend), 0.0), 1.0)
        )
        self.tva_dynamic_revisit_penalty = max(float(tva_dynamic_revisit_penalty), 0.0)
        self.tva_dynamic_prior_penalty_weight = max(float(tva_dynamic_prior_penalty_weight), 0.0)
        self.tva_dynamic_prior_penalty_cap = max(float(tva_dynamic_prior_penalty_cap), 0.0)
        self.tva_dynamic_leaf_penalty_weight = max(float(tva_dynamic_leaf_penalty_weight), 0.0)
        self.tva_dynamic_leaf_penalty_cap = max(float(tva_dynamic_leaf_penalty_cap), 0.0)
        self.tva_dynamic_root_select = bool(tva_dynamic_root_select)
        self.tva_dynamic_root_select_margin = float(tva_dynamic_root_select_margin)
        self.tva_state_gated_blend = bool(tva_state_gated_blend)
        self.tva_state_gated_default_blend = min(max(float(tva_state_gated_default_blend), 0.0), 1.0)
        self.tva_state_gated_positive_blend = (
            self.tva_value_blend
            if float(tva_state_gated_positive_blend) < 0.0
            else min(max(float(tva_state_gated_positive_blend), 0.0), 1.0)
        )
        self.tva_state_gated_positive_risk_max = min(max(float(tva_state_gated_positive_risk_max), 0.0), 1.0)
        self.tva_state_gated_positive_delta_min = max(float(tva_state_gated_positive_delta_min), 0.0)
        self.tva_state_gated_negative_blend = min(max(float(tva_state_gated_negative_blend), 0.0), 1.0)
        self.tva_state_gated_negative_risk_min = min(max(float(tva_state_gated_negative_risk_min), 0.0), 1.0)
        self.tva_state_gated_negative_delta_min = max(float(tva_state_gated_negative_delta_min), 0.0)
        self.tva_state_gated_ood_z_threshold = float(tva_state_gated_ood_z_threshold)
        self.tva_state_gated_ood_blend = min(max(float(tva_state_gated_ood_blend), 0.0), 1.0)
        self.tva_transport_phase_blend = (
            None
            if float(tva_transport_phase_blend) < 0.0
            else min(max(float(tva_transport_phase_blend), 0.0), 1.0)
        )
        self.tva_transport_phase_high_progress_min = float(tva_transport_phase_high_progress_min)
        self.tva_transport_phase_low_progress_max = float(tva_transport_phase_low_progress_max)
        self.tva_transport_phase_commitment_min = max(float(tva_transport_phase_commitment_min), 0.0)
        self.tva_transport_phase_max_closed_loop_step = int(tva_transport_phase_max_closed_loop_step)
        self.tva_transport_phase_require_root_eligible = bool(tva_transport_phase_require_root_eligible)
        self._tva_transport_phase_root_eligible = not self.tva_transport_phase_require_root_eligible
        self.closed_loop_step = int(closed_loop_step)
        self._gft_sd_leaf_cache: dict[tuple, float] = {}
        self._gft_tva_prediction_cache: dict[tuple, float] = {}
        self._gft_tva_details_cache: dict[tuple, dict[str, float]] = {}

    def transport_phase_eligible(self, state) -> bool:
        if self.tva_transport_phase_blend is None:
            return False
        features = self.model.gft_tva_feature_dict(state)["features"]
        progress = float(features.get("P", math.inf))
        commitment = float(features.get("C", 0.0))
        if not math.isfinite(progress):
            return False
        high_progress = (
            self.tva_transport_phase_high_progress_min >= 0.0
            and progress >= self.tva_transport_phase_high_progress_min
        )
        low_committed_progress = (
            self.tva_transport_phase_low_progress_max >= 0.0
            and progress <= self.tva_transport_phase_low_progress_max
            and commitment >= self.tva_transport_phase_commitment_min
        )
        return bool(high_progress or low_committed_progress)

    def assign_sd_delta_priors(self, state, candidates: list[CandidateAction]) -> None:
        if self.gft_prior != "gft_delta":
            return super().assign_sd_delta_priors(state, candidates)
        if not candidates:
            return

        rows = self.model.gft_action_prior_rows(
            state,
            [(candidate.action, candidate.child_state) for candidate in candidates],
            visited_keys=self.visited_keys,
            weights=self.gft_prior_weights,
            temperature=self.gft_prior_temperature,
            uniform_mix=self.gft_prior_uniform_mix,
        )
        if self.tva_dynamic_prior_penalty_weight <= 0.0 or not isinstance(self.tva_checkpoint, GFTTvaCheckpoint):
            for candidate, row in zip(candidates, rows):
                candidate.prior_score = float(row["score"])
                candidate.prior = float(row["prior"])
            return

        scores = []
        for candidate, row in zip(candidates, rows):
            score = float(row["score"])
            penalty_signal = self.dynamic_negative_tva_signal(candidate.child_state)
            if penalty_signal > 0.0:
                score -= self.tva_dynamic_prior_penalty_weight * penalty_signal
            candidate.prior_score = score
            scores.append(score)
        uniform = 1.0 / len(candidates)
        logits = np.asarray(scores, dtype=np.float64) / self.gft_prior_temperature
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        denom = float(np.sum(probs))
        probs = probs / denom if denom > 0.0 else np.full(len(candidates), uniform)
        mix = self.gft_prior_uniform_mix
        for candidate, prob in zip(candidates, probs):
            candidate.prior = float((1.0 - mix) * prob + mix * uniform)

    def sd_heuristic_value(self, state) -> float:
        """Use the GFT transport-progress stock for the SD heuristic leaf value."""

        cached = self._sd_leaf_cache.get(self.model.gft_canonical_key_cached(state))
        if cached is not None:
            return cached
        lb = self.model.gft_transport_progress_cached(state)
        if not math.isfinite(lb):
            value = self.deadend_value
        elif self.gamma == 1.0:
            value = self.goal_value - self.push_cost * lb
        else:
            discount = self.gamma**lb
            value = discount * self.goal_value - self.push_cost * (1.0 - discount) / (1.0 - self.gamma)
        self._sd_leaf_cache[self.model.gft_canonical_key_cached(state)] = value
        return value

    def gft_sd_leaf_value(self, state) -> float:
        key = self.model.gft_canonical_key_cached(state)
        cached = self._gft_sd_leaf_cache.get(key)
        if cached is not None:
            return cached
        if self.gft_sd_leaf_mode == "discounted_progress":
            value = discounted_progress_gft_sd_value(
                self.model,
                state,
                gamma=self.gamma,
                goal_value=self.goal_value,
                push_cost=self.push_cost,
                deadend_value=self.deadend_value,
                weights=self.gft_sd_leaf_weights,
            )
        else:
            value = projected_gft_sd_value(
                self.model,
                state,
                deadend_value=self.deadend_value,
                goal_value=self.goal_value,
                weights=self.gft_sd_leaf_weights if has_nonzero_gft_leaf_weights(self.gft_sd_leaf_weights) else None,
            )
        self._gft_sd_leaf_cache[key] = value
        return value

    def gft_tva_prediction_value(self, state) -> float:
        return float(self.gft_tva_prediction_details(state)["prediction"])

    def gft_tva_prediction_details(self, state) -> dict[str, float]:
        key = self.model.gft_canonical_key_cached(state)
        cached = self._gft_tva_details_cache.get(key)
        if cached is not None:
            return cached
        details = predict_gft_tva_details(self.tva_checkpoint, self.model, state)
        self._gft_tva_details_cache[key] = details
        self._gft_tva_prediction_cache[key] = float(details["prediction"])
        return details

    def dynamic_negative_tva_signal(self, state) -> float:
        if (
            not self.tva_dynamic_blend
            and self.tva_dynamic_prior_penalty_weight <= 0.0
            and not self.tva_dynamic_root_select
            and not self.tva_state_gated_blend
        ):
            return 0.0
        if not isinstance(self.tva_checkpoint, GFTTvaCheckpoint):
            return 0.0
        if normalize_gft_tva_target_mode(self.tva_checkpoint.target_mode) != "direct_value":
            return 0.0
        vsd = self.gft_sd_leaf_value(state)
        details = self.gft_tva_prediction_details(state)
        delta = float(details["prediction"]) - vsd
        risk = float(details.get("risk_probability", 0.0))
        if risk < self.tva_dynamic_high_risk_threshold:
            return 0.0
        if delta > -self.tva_dynamic_negative_delta_threshold:
            return 0.0
        return float(min(max(-delta, 0.0), self.tva_dynamic_prior_penalty_cap))

    def state_gated_tva_alpha(self, state, vsd: float, prediction: float, details: dict[str, float]) -> float:
        alpha = self.tva_state_gated_default_blend
        max_abs_z = float(details.get("feature_max_abs_z", 0.0))
        if self.tva_state_gated_ood_z_threshold > 0.0 and max_abs_z > self.tva_state_gated_ood_z_threshold:
            return self.tva_state_gated_ood_blend

        if self.tva_transport_phase_blend is not None:
            features = self.model.gft_tva_feature_dict(state)["features"]
            progress = float(features.get("P", math.inf))
            commitment = float(features.get("C", 0.0))
            within_prefix = (
                self.tva_transport_phase_max_closed_loop_step < 0
                or self.closed_loop_step <= self.tva_transport_phase_max_closed_loop_step
            )
            root_allowed = (
                not self.tva_transport_phase_require_root_eligible
                or self._tva_transport_phase_root_eligible
            )
            if within_prefix and root_allowed and math.isfinite(progress):
                high_progress = (
                    self.tva_transport_phase_high_progress_min >= 0.0
                    and progress >= self.tva_transport_phase_high_progress_min
                )
                low_committed_progress = (
                    self.tva_transport_phase_low_progress_max >= 0.0
                    and progress <= self.tva_transport_phase_low_progress_max
                    and commitment >= self.tva_transport_phase_commitment_min
                )
                if high_progress or low_committed_progress:
                    return self.tva_transport_phase_blend

        delta = float(prediction) - float(vsd)
        risk = float(details.get("risk_probability", 0.0))
        if risk <= self.tva_state_gated_positive_risk_max and delta >= self.tva_state_gated_positive_delta_min:
            return self.tva_state_gated_positive_blend
        if risk >= self.tva_state_gated_negative_risk_min and delta <= -self.tva_state_gated_negative_delta_min:
            return self.tva_state_gated_negative_blend
        return alpha

    def direct_gft_tva_leaf_value(self, state) -> float:
        vsd = self.gft_sd_leaf_value(state)
        details = self.gft_tva_prediction_details(state)
        prediction = float(details["prediction"])
        if self.tva_value_blend >= 1.0 and not self.tva_dynamic_blend and not self.tva_state_gated_blend:
            return prediction

        if self.tva_state_gated_blend:
            alpha = self.state_gated_tva_alpha(state, vsd, prediction, details)
        else:
            alpha = self.tva_value_blend
        delta = prediction - vsd
        if self.tva_dynamic_blend and not self.tva_state_gated_blend:
            negative_signal = self.dynamic_negative_tva_signal(state)
            risk = float(details.get("risk_probability", 0.0))
            if negative_signal > 0.0:
                alpha = max(alpha, self.tva_dynamic_high_risk_blend)
            elif (
                risk <= self.tva_dynamic_low_risk_threshold
                and delta >= self.tva_dynamic_positive_delta_threshold
                and self.tva_dynamic_positive_blend is not None
            ):
                alpha = self.tva_dynamic_positive_blend

        value = (1.0 - alpha) * vsd + alpha * prediction
        if self.tva_dynamic_leaf_penalty_weight > 0.0:
            penalty_signal = self.dynamic_negative_tva_signal(state)
            if penalty_signal > 0.0:
                value -= self.tva_dynamic_leaf_penalty_weight * min(
                    penalty_signal, self.tva_dynamic_leaf_penalty_cap
                )
        if self.tva_dynamic_blend and self.tva_dynamic_revisit_penalty > 0.0:
            if self.model.gft_canonical_key_cached(state) in self.visited_keys:
                value -= self.tva_dynamic_revisit_penalty
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        return float(max(low, min(high, value)))

    def leaf_value(self, state) -> float:
        if self.model.is_solved(state):
            return self.goal_value
        if self.is_blocked(state):
            return self.deadend_value
        if self.leaf_value_mode == "zero":
            return 0.0
        if self.leaf_value_mode == "sd_heuristic":
            return self.sd_heuristic_value(state)
        if self.leaf_value_mode == "gft_sd":
            return self.gft_sd_leaf_value(state)
        if self.tva_checkpoint is None:
            raise ValueError("TVA leaf value requested without a checkpoint.")
        if isinstance(self.tva_checkpoint, GFTTvaCheckpoint):
            prediction = self.gft_tva_prediction_value(state)
            target_mode = normalize_gft_tva_target_mode(self.tva_checkpoint.target_mode)
            if target_mode == "gft_sd_residual":
                return self.gft_sd_leaf_value(state) + prediction
            if target_mode == "sd_heuristic_residual":
                return self.sd_heuristic_value(state) + prediction
            if target_mode == "direct_value":
                return self.direct_gft_tva_leaf_value(state)
            raise ValueError(f"Unsupported GFT-TVA target_mode: {self.tva_checkpoint.target_mode}")
        return super().leaf_value(state)

    def choose_root_edge(self, root) -> Any:
        visit_best = max(root.children.values(), key=lambda edge: (edge.visits, edge.mean_value))
        if self.select_by == "q":
            return max(root.children.values(), key=lambda edge: (edge.mean_value, edge.visits))
        if not self.tva_dynamic_root_select:
            return visit_best
        q_best = max(root.children.values(), key=lambda edge: (edge.mean_value, edge.visits))
        if q_best is visit_best:
            return visit_best
        if self.dynamic_negative_tva_signal(visit_best.child_state) <= 0.0:
            return visit_best
        if q_best.mean_value < visit_best.mean_value + self.tva_dynamic_root_select_margin:
            return visit_best
        return q_best

    def plan(self, state) -> dict[str, Any]:
        if self.tva_transport_phase_require_root_eligible:
            self._tva_transport_phase_root_eligible = self.transport_phase_eligible(state)
        root = self.node_for(state, 0)
        prunes_before = self.deadlock_prunes
        for _ in range(max(self.budget, 1)):
            self.simulate(root, self.horizon)
        plan_prunes = self.deadlock_prunes - prunes_before
        if not root.children:
            return {
                "action": None,
                "root_visits": root.visits,
                "root_value": root.value_sum / root.visits if root.visits else 0.0,
                "child_count": 0,
                "deadlock_prunes": plan_prunes,
                "children": [],
            }
        best = self.choose_root_edge(root)
        children = sorted(
            (
                {
                    "action": action_to_dict(edge.action),
                    "visits": int(edge.visits),
                    "mean_value": float(edge.mean_value),
                    "prior": float(edge.prior),
                    "prior_score": float(edge.prior_score),
                    "solved_child": bool(self.model.is_solved(edge.child_state)),
                    "blocked_child": bool(self.is_blocked(edge.child_state)),
                }
                for edge in root.children.values()
            ),
            key=lambda row: (row["visits"], row["mean_value"]),
            reverse=True,
        )
        return {
            "action": best.action,
            "root_visits": root.visits,
            "root_value": root.value_sum / root.visits if root.visits else 0.0,
            "child_count": len(root.children),
            "deadlock_prunes": plan_prunes,
            "children": children,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = build_grid_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "leaf_value":
            action.choices = ("zero", "tva", "sd_heuristic", "gft_sd")
        if action.dest == "tva_checkpoint":
            action.help = "Legacy grid TVA or GFT-TVA checkpoint, required when --leaf-value=tva."
    parser.set_defaults(
        leaf_value="sd_heuristic",
        revisit_penalty=1.0,
        out="results/sokoban_sd_world_model/gft_sdmcts_eval_seed0.json",
    )
    parser.add_argument(
        "--gft-prior",
        choices=("none", "gft_delta"),
        default="gft_delta",
        help="GFT-SD action relevance prior. 'none' falls back to --action-prior.",
    )
    parser.add_argument("--gft-prior-temperature", type=float, default=2.0)
    parser.add_argument("--gft-prior-uniform-mix", type=float, default=0.1)
    parser.add_argument("--gft-progress-weight", type=float, default=1.0)
    parser.add_argument("--gft-goal-weight", type=float, default=1.0)
    parser.add_argument("--gft-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-capacity-weight", type=float, default=0.0)
    parser.add_argument("--gft-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-ordering-weight", type=float, default=0.0)
    parser.add_argument("--gft-commitment-weight", type=float, default=0.0)
    parser.add_argument("--gft-bottleneck-weight", type=float, default=0.0)
    parser.add_argument("--gft-revisit-weight", type=float, default=1.0)
    parser.add_argument("--gft-transport-turn-weight", type=float, default=0.0)
    parser.add_argument("--gft-transport-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-transport-zone-weight", type=float, default=0.0)
    parser.add_argument("--gft-soft-deadlock-mobility-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-matching-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-capacity-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-access-weight", type=float, default=1.0)
    parser.add_argument("--gft-soft-deadlock-order-weight", type=float, default=1.0)
    parser.add_argument("--gft-commitment-beta", type=float, default=1.0)
    parser.add_argument("--gft-sinkhorn-iterations", type=int, default=25)
    parser.add_argument(
        "--gft-sd-leaf-mode",
        choices=("linear_projected", "discounted_progress"),
        default="linear_projected",
        help="How explicit GFT V_SD is mapped to the MCTS leaf-value scale.",
    )
    parser.add_argument("--gft-sd-leaf-mobility-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-access-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-capacity-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-deadlock-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-ordering-weight", type=float, default=0.0)
    parser.add_argument("--gft-sd-leaf-commitment-weight", type=float, default=0.0)
    parser.add_argument(
        "--tva-value-blend",
        type=float,
        default=1.0,
        help=(
            "For direct-value GFT-TVA checkpoints, alpha in "
            "V_leaf=(1-alpha)*V_SD+alpha*V_psi. alpha=1 uses pure TVA."
        ),
    )
    parser.add_argument(
        "--tva-dynamic-blend",
        action="store_true",
        help=(
            "For direct-value GFT-TVA checkpoints, dynamically adjust alpha using "
            "the candidate child state's TVA risk probability and V_TVA-V_SD correction."
        ),
    )
    parser.add_argument("--tva-dynamic-high-risk-threshold", type=float, default=0.9)
    parser.add_argument("--tva-dynamic-negative-delta-threshold", type=float, default=1.0)
    parser.add_argument("--tva-dynamic-high-risk-blend", type=float, default=0.75)
    parser.add_argument("--tva-dynamic-low-risk-threshold", type=float, default=0.3)
    parser.add_argument("--tva-dynamic-positive-delta-threshold", type=float, default=0.25)
    parser.add_argument(
        "--tva-dynamic-positive-blend",
        type=float,
        default=-1.0,
        help="Alpha used for low-risk positive TVA corrections; negative means keep --tva-value-blend.",
    )
    parser.add_argument(
        "--tva-dynamic-revisit-penalty",
        type=float,
        default=0.0,
        help="Extra leaf-value penalty when the candidate state has already appeared in the closed-loop history.",
    )
    parser.add_argument(
        "--tva-dynamic-prior-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Softly subtract weight*min(-(V_TVA-V_SD), cap) from the GFT action prior score "
            "when the same high-risk negative-correction condition is met."
        ),
    )
    parser.add_argument("--tva-dynamic-prior-penalty-cap", type=float, default=4.0)
    parser.add_argument(
        "--tva-dynamic-leaf-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Softly subtract weight*min(-(V_TVA-V_SD), cap) from direct-value leaf estimates "
            "when the same high-risk negative-correction condition is met."
        ),
    )
    parser.add_argument("--tva-dynamic-leaf-penalty-cap", type=float, default=4.0)
    parser.add_argument(
        "--tva-dynamic-root-select",
        action="store_true",
        help=(
            "Keep visit-count root selection by default, but switch to the higher-value root child "
            "when the visit-best child has the high-risk negative TVA signal."
        ),
    )
    parser.add_argument("--tva-dynamic-root-select-margin", type=float, default=0.0)
    parser.add_argument(
        "--tva-state-gated-blend",
        action="store_true",
        help=(
            "Use state-dependent alpha(s) for direct-value GFT-TVA checkpoints. "
            "The default policy can set alpha=0 and only trust TVA in configured state regions."
        ),
    )
    parser.add_argument("--tva-state-gated-default-blend", type=float, default=0.0)
    parser.add_argument(
        "--tva-state-gated-positive-blend",
        type=float,
        default=-1.0,
        help="Alpha for low-risk positive TVA corrections; negative means use --tva-value-blend.",
    )
    parser.add_argument("--tva-state-gated-positive-risk-max", type=float, default=0.30)
    parser.add_argument("--tva-state-gated-positive-delta-min", type=float, default=0.25)
    parser.add_argument("--tva-state-gated-negative-blend", type=float, default=0.0)
    parser.add_argument("--tva-state-gated-negative-risk-min", type=float, default=0.95)
    parser.add_argument("--tva-state-gated-negative-delta-min", type=float, default=1.0)
    parser.add_argument(
        "--tva-state-gated-ood-z-threshold",
        type=float,
        default=-1.0,
        help="If positive, states with max absolute normalized TVA feature above this threshold use OOD blend.",
    )
    parser.add_argument("--tva-state-gated-ood-blend", type=float, default=0.0)
    parser.add_argument(
        "--tva-transport-phase-blend",
        type=float,
        default=-1.0,
        help=(
            "Optional alpha for transport-phase states in state-gated direct-value TVA. "
            "Negative disables this phase rule."
        ),
    )
    parser.add_argument("--tva-transport-phase-high-progress-min", type=float, default=-1.0)
    parser.add_argument("--tva-transport-phase-low-progress-max", type=float, default=-1.0)
    parser.add_argument("--tva-transport-phase-commitment-min", type=float, default=0.25)
    parser.add_argument(
        "--tva-transport-phase-max-closed-loop-step",
        type=int,
        default=-1,
        help=(
            "If non-negative, the transport-phase TVA blend can trigger only through this "
            "closed-loop push step. This is a history-aware guard for early commitment windows."
        ),
    )
    parser.add_argument(
        "--tva-transport-phase-require-root-eligible",
        action="store_true",
        help=(
            "Only allow transport-phase TVA blending when the current closed-loop root state "
            "itself satisfies the transport-phase predicate."
        ),
    )
    parser.add_argument(
        "--trace-gft-readouts",
        action="store_true",
        help="Include active GFT resource readouts in trace rows.",
    )
    return parser


def gft_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    weights = dict(DEFAULT_GFT_PRIOR_WEIGHTS)
    weights.update(
        {
            "progress": args.gft_progress_weight,
            "goal": args.gft_goal_weight,
            "mobility": args.gft_mobility_weight,
            "access": args.gft_access_weight,
            "capacity": args.gft_capacity_weight,
            "deadlock": args.gft_deadlock_weight,
            "ordering": args.gft_ordering_weight,
            "commitment": args.gft_commitment_weight,
            "bottleneck": args.gft_bottleneck_weight,
            "revisit": args.gft_revisit_weight,
        }
    )
    return weights


def gft_transport_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    weights = dict(DEFAULT_GFT_TRANSPORT_COST_WEIGHTS)
    weights.update(
        {
            "turn": args.gft_transport_turn_weight,
            "access": args.gft_transport_access_weight,
            "zone": args.gft_transport_zone_weight,
        }
    )
    return weights


def gft_soft_deadlock_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    weights = dict(DEFAULT_GFT_SOFT_DEADLOCK_WEIGHTS)
    weights.update(
        {
            "mobility_loss": args.gft_soft_deadlock_mobility_weight,
            "matching_fragility": args.gft_soft_deadlock_matching_weight,
            "capacity_stress": args.gft_soft_deadlock_capacity_weight,
            "access_loss": args.gft_soft_deadlock_access_weight,
            "order_violation": args.gft_soft_deadlock_order_weight,
        }
    )
    return weights


def gft_sd_leaf_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "mobility": float(args.gft_sd_leaf_mobility_weight),
        "access": float(args.gft_sd_leaf_access_weight),
        "capacity": float(args.gft_sd_leaf_capacity_weight),
        "deadlock": float(args.gft_sd_leaf_deadlock_weight),
        "ordering": float(args.gft_sd_leaf_ordering_weight),
        "commitment": float(args.gft_sd_leaf_commitment_weight),
    }


def gft_model_from_level(level, args: argparse.Namespace) -> SokobanGFTSDModel:
    return SokobanGFTSDModel(
        level,
        transport_cost_weights=gft_transport_weights_from_args(args),
        soft_deadlock_weights=gft_soft_deadlock_weights_from_args(args),
        commitment_beta=args.gft_commitment_beta,
        sinkhorn_iterations=args.gft_sinkhorn_iterations,
    )


def run_level(
    level_idx: int,
    model: SokobanGFTSDModel,
    args: argparse.Namespace,
    tva_checkpoint: TvaCheckpoint | None,
) -> dict[str, Any]:
    rng = random.Random(args.seed + level_idx * 1009)
    state = model.initial_state
    visited_keys = {model.canonical_key(state)}
    trace: list[dict[str, Any]] = []
    total_deadlock_prunes = 0
    pushes_executed = 0
    t0 = perf_counter()
    status = "max_steps"
    persistent_planner: GFTSDMCTSPlanner | None = None
    gft_weights = gft_weights_from_args(args)
    planner_action_prior = "sd_delta" if args.gft_prior == "gft_delta" else args.action_prior

    for step in range(args.max_steps + 1):
        if model.is_solved(state):
            status = "solved"
            break
        if step >= args.max_steps:
            break
        if persistent_planner is not None:
            planner = persistent_planner
        else:
            planner = GFTSDMCTSPlanner(
                model,
                horizon=args.horizon,
                budget=args.budget,
                rng=rng,
                uct_c=args.uct_c,
                gamma=args.gamma,
                goal_value=args.goal_value,
                push_cost=args.push_cost,
                deadend_value=args.deadend_value,
                leaf_value=args.leaf_value,
                tva_checkpoint=tva_checkpoint,
                constraint_mode=args.constraint_mode,
                select_by=args.select_by,
                revisit_penalty=args.revisit_penalty,
                visited_keys=visited_keys,
                transposition=args.transposition or args.reuse_tree,
                action_prior=planner_action_prior,
                prior_temperature=args.prior_temperature,
                prior_uniform_mix=args.prior_uniform_mix,
                prior_progress_weight=args.prior_progress_weight,
                prior_goal_weight=args.prior_goal_weight,
                prior_mobility_weight=args.prior_mobility_weight,
                prior_revisit_weight=args.prior_revisit_weight,
                gft_prior=args.gft_prior,
                gft_prior_temperature=args.gft_prior_temperature,
                gft_prior_uniform_mix=args.gft_prior_uniform_mix,
                gft_prior_weights=gft_weights,
                gft_sd_leaf_mode=args.gft_sd_leaf_mode,
                gft_sd_leaf_weights=gft_sd_leaf_weights_from_args(args),
                tva_value_blend=args.tva_value_blend,
                tva_dynamic_blend=args.tva_dynamic_blend,
                tva_dynamic_high_risk_threshold=args.tva_dynamic_high_risk_threshold,
                tva_dynamic_negative_delta_threshold=args.tva_dynamic_negative_delta_threshold,
                tva_dynamic_high_risk_blend=args.tva_dynamic_high_risk_blend,
                tva_dynamic_low_risk_threshold=args.tva_dynamic_low_risk_threshold,
                tva_dynamic_positive_delta_threshold=args.tva_dynamic_positive_delta_threshold,
                tva_dynamic_positive_blend=args.tva_dynamic_positive_blend,
                tva_dynamic_revisit_penalty=args.tva_dynamic_revisit_penalty,
                tva_dynamic_prior_penalty_weight=args.tva_dynamic_prior_penalty_weight,
                tva_dynamic_prior_penalty_cap=args.tva_dynamic_prior_penalty_cap,
                tva_dynamic_leaf_penalty_weight=args.tva_dynamic_leaf_penalty_weight,
                tva_dynamic_leaf_penalty_cap=args.tva_dynamic_leaf_penalty_cap,
                tva_dynamic_root_select=args.tva_dynamic_root_select,
                tva_dynamic_root_select_margin=args.tva_dynamic_root_select_margin,
                tva_state_gated_blend=args.tva_state_gated_blend,
                tva_state_gated_default_blend=args.tva_state_gated_default_blend,
                tva_state_gated_positive_blend=args.tva_state_gated_positive_blend,
                tva_state_gated_positive_risk_max=args.tva_state_gated_positive_risk_max,
                tva_state_gated_positive_delta_min=args.tva_state_gated_positive_delta_min,
                tva_state_gated_negative_blend=args.tva_state_gated_negative_blend,
                tva_state_gated_negative_risk_min=args.tva_state_gated_negative_risk_min,
                tva_state_gated_negative_delta_min=args.tva_state_gated_negative_delta_min,
                tva_state_gated_ood_z_threshold=args.tva_state_gated_ood_z_threshold,
                tva_state_gated_ood_blend=args.tva_state_gated_ood_blend,
                tva_transport_phase_blend=args.tva_transport_phase_blend,
                tva_transport_phase_high_progress_min=args.tva_transport_phase_high_progress_min,
                tva_transport_phase_low_progress_max=args.tva_transport_phase_low_progress_max,
                tva_transport_phase_commitment_min=args.tva_transport_phase_commitment_min,
                tva_transport_phase_max_closed_loop_step=args.tva_transport_phase_max_closed_loop_step,
                tva_transport_phase_require_root_eligible=args.tva_transport_phase_require_root_eligible,
                closed_loop_step=step,
            )
            if args.reuse_tree:
                persistent_planner = planner
        plan = planner.plan(state)
        total_deadlock_prunes += int(plan["deadlock_prunes"])
        action = plan["action"]
        if action is None:
            status = "no_action"
            if len(trace) < args.trace_limit:
                row = {
                    "step": step,
                    "status": "no_action",
                    "root_visits": int(plan["root_visits"]),
                    "root_value": float(plan["root_value"]),
                }
                if args.trace_gft_readouts:
                    row["gft_state"] = model.gft_state(state, active_only=True).to_dict()
                    row["gft_sd_value"] = model.gft_sd_value(state)
                    row["gft_tva_features"] = model.gft_tva_feature_dict(state)
                trace.append(row)
            break

        next_state = model.step_grid_flow(state, action)
        pushes_executed += 1
        if len(trace) < args.trace_limit:
            row = {
                "step": step,
                "action": action_to_dict(action),
                "root_visits": int(plan["root_visits"]),
                "root_value": float(plan["root_value"]),
                "child_count": int(plan["child_count"]),
                "top_children": plan["children"][:5],
                "boxes_on_goals_after": int(model.boxes_on_goals(next_state)),
                "tier_a_after": model.tier_a_status(next_state),
            }
            if args.trace_gft_readouts:
                row["gft_state"] = model.gft_state(state, active_only=True).to_dict()
                row["gft_delta"] = model.gft_delta(state, next_state)
                row["gft_flow"] = model.gft_resource_flow(state, next_state)
                row["gft_sd_reward"] = model.gft_sd_reward(
                    state,
                    next_state,
                    push_cost=args.push_cost,
                    goal_value=args.goal_value,
                    dead_value=args.deadend_value,
                )
                row["gft_sd_value_before"] = model.gft_sd_value(state)
                row["gft_sd_value_after"] = model.gft_sd_value(next_state)
                row["gft_tva_features_after"] = model.gft_tva_feature_dict(next_state)
            trace.append(row)
        state = next_state
        visited_keys.add(model.canonical_key(state))
        if model.is_deadlocked(state) and not model.is_solved(state):
            status = "deadlocked"
            break

    elapsed = perf_counter() - t0
    solved = model.is_solved(state)
    return {
        "level_index": int(level_idx),
        "level_id": model.level.level_id,
        "height": model.level.height,
        "width": model.level.width,
        "n_boxes": model.level.n_boxes,
        "solved": bool(solved),
        "status": "solved" if solved else status,
        "pushes": pushes_executed if solved else None,
        "steps_executed": pushes_executed,
        "deadlock_prunes": int(total_deadlock_prunes),
        "elapsed_sec": elapsed,
        "trace": trace,
        "final_features": model.abstract_features(state).to_dict(),
        "final_gft_state": model.gft_state(state).to_dict(active_only=False),
    }


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= float(args.tva_value_blend) <= 1.0:
        raise SystemExit("--tva-value-blend must be between 0 and 1.")
    for name in (
        "tva_dynamic_high_risk_threshold",
        "tva_dynamic_low_risk_threshold",
        "tva_dynamic_high_risk_blend",
        "tva_state_gated_default_blend",
        "tva_state_gated_positive_risk_max",
        "tva_state_gated_negative_blend",
        "tva_state_gated_negative_risk_min",
        "tva_state_gated_ood_blend",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if float(args.tva_dynamic_positive_blend) > 1.0:
        raise SystemExit("--tva-dynamic-positive-blend must be <= 1, or negative to keep --tva-value-blend.")
    if float(args.tva_state_gated_positive_blend) > 1.0:
        raise SystemExit("--tva-state-gated-positive-blend must be <= 1, or negative to keep --tva-value-blend.")
    if float(args.tva_transport_phase_blend) > 1.0:
        raise SystemExit("--tva-transport-phase-blend must be <= 1, or negative to disable the phase rule.")
    if float(args.tva_dynamic_negative_delta_threshold) < 0.0:
        raise SystemExit("--tva-dynamic-negative-delta-threshold must be non-negative.")
    if float(args.tva_dynamic_positive_delta_threshold) < 0.0:
        raise SystemExit("--tva-dynamic-positive-delta-threshold must be non-negative.")
    if float(args.tva_state_gated_positive_delta_min) < 0.0:
        raise SystemExit("--tva-state-gated-positive-delta-min must be non-negative.")
    if float(args.tva_state_gated_negative_delta_min) < 0.0:
        raise SystemExit("--tva-state-gated-negative-delta-min must be non-negative.")
    if float(args.tva_dynamic_revisit_penalty) < 0.0:
        raise SystemExit("--tva-dynamic-revisit-penalty must be non-negative.")
    if float(args.tva_dynamic_prior_penalty_weight) < 0.0:
        raise SystemExit("--tva-dynamic-prior-penalty-weight must be non-negative.")
    if float(args.tva_dynamic_prior_penalty_cap) < 0.0:
        raise SystemExit("--tva-dynamic-prior-penalty-cap must be non-negative.")
    if float(args.tva_dynamic_leaf_penalty_weight) < 0.0:
        raise SystemExit("--tva-dynamic-leaf-penalty-weight must be non-negative.")
    if float(args.tva_dynamic_leaf_penalty_cap) < 0.0:
        raise SystemExit("--tva-dynamic-leaf-penalty-cap must be non-negative.")
    if float(args.tva_transport_phase_commitment_min) < 0.0:
        raise SystemExit("--tva-transport-phase-commitment-min must be non-negative.")
    if int(args.tva_transport_phase_max_closed_loop_step) < -1:
        raise SystemExit("--tva-transport-phase-max-closed-loop-step must be >= -1.")
    if (
        args.tva_dynamic_blend
        or float(args.tva_dynamic_prior_penalty_weight) > 0.0
        or float(args.tva_dynamic_leaf_penalty_weight) > 0.0
        or args.tva_dynamic_root_select
        or args.tva_state_gated_blend
    ) and args.leaf_value != "tva":
        raise SystemExit("TVA dynamic leaf/prior controls require --leaf-value=tva.")
    gate_status = check_independent_gates(args)
    if args.leaf_value == "tva":
        tva_checkpoint = load_tva_checkpoint(args.tva_checkpoint)
        mismatched = {
            key: (tva_checkpoint.value_scale.get(key), getattr(args, key))
            for key in ("gamma", "goal_value", "push_cost", "deadend_value")
            if key in tva_checkpoint.value_scale
            and not math.isclose(tva_checkpoint.value_scale[key], getattr(args, key), abs_tol=1e-9)
        }
        if mismatched:
            raise SystemExit(f"TVA checkpoint value_scale mismatches runtime args: {mismatched}")
        if isinstance(tva_checkpoint, GFTTvaCheckpoint):
            target_mode = normalize_gft_tva_target_mode(tva_checkpoint.target_mode)
            if target_mode not in ("gft_sd_residual", "sd_heuristic_residual", "direct_value"):
                raise SystemExit(f"Unsupported GFT-TVA checkpoint target_mode: {tva_checkpoint.target_mode}")
            if (
                args.tva_dynamic_blend or float(args.tva_dynamic_prior_penalty_weight) > 0.0
                or args.tva_dynamic_root_select
                or args.tva_state_gated_blend
            ) and target_mode != "direct_value":
                raise SystemExit(
                    "TVA dynamic leaf/prior controls are only supported for direct-value GFT-TVA checkpoints."
                )
        if isinstance(tva_checkpoint, GFTTvaCheckpoint) and (
            tva_checkpoint.target_mode == "gft_sd_residual"
            or (tva_checkpoint.target_mode == "direct_value" and float(args.tva_value_blend) < 1.0)
        ):
            contract = tva_checkpoint.checkpoint_contract
            expected_leaf_mode = contract.get("gft_sd_leaf_mode")
            if expected_leaf_mode and str(expected_leaf_mode) != str(args.gft_sd_leaf_mode):
                raise SystemExit(
                    "GFT-TVA checkpoint gft_sd_leaf_mode mismatches runtime args: "
                    f"checkpoint={expected_leaf_mode!r}, runtime={args.gft_sd_leaf_mode!r}"
                )
            expected_leaf_weights = contract.get("gft_sd_leaf_weights")
            if isinstance(expected_leaf_weights, dict):
                runtime_leaf_weights = gft_sd_leaf_weights_from_args(args)
                mismatched_leaf_weights = {
                    key: (float(expected), float(runtime_leaf_weights.get(key, 0.0)))
                    for key, expected in expected_leaf_weights.items()
                    if not math.isclose(
                        float(expected),
                        float(runtime_leaf_weights.get(key, 0.0)),
                        abs_tol=1e-12,
                    )
                }
                if mismatched_leaf_weights:
                    raise SystemExit(
                        "GFT-TVA checkpoint gft_sd_leaf_weights mismatch runtime args: "
                        f"{mismatched_leaf_weights}"
                    )
    else:
        tva_checkpoint = None

    if args.level_offset < 0:
        raise SystemExit("--level-offset must be non-negative.")
    levels = load_boxoban_levels(args.levels, max_levels=args.level_offset + args.max_levels)
    if args.level_offset:
        levels = levels[args.level_offset :]
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    rows = [
        run_level(level_idx, gft_model_from_level(level, args), args, tva_checkpoint)
        for level_idx, level in enumerate(levels)
    ]
    summary = {
        "material_passport": {
            "origin": "sokoban_gft_sdmcts_rebuild",
            "verification_status": (
                "VERIFIED"
                if bool(gate_status.get("require_independent_gates")) and not gate_status.get("failed_checks")
                else "ANALYZED"
            ),
            "version_label": "gft_sd_two_layer_closed_loop_v1",
        },
        "config": {
            "levels": _rel(args.levels),
            "level_offset": args.level_offset,
            "max_levels": args.max_levels,
            "horizon": args.horizon,
            "budget": args.budget,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "uct_c": args.uct_c,
            "gamma": args.gamma,
            "goal_value": args.goal_value,
            "push_cost": args.push_cost,
            "deadend_value": args.deadend_value,
            "leaf_value": args.leaf_value,
            "constraint_mode": args.constraint_mode,
            "select_by": args.select_by,
            "revisit_penalty": args.revisit_penalty,
            "transposition": bool(args.transposition or args.reuse_tree),
            "reuse_tree": bool(args.reuse_tree),
            "action_prior": args.action_prior,
            "planner_action_prior": "sd_delta" if args.gft_prior == "gft_delta" else args.action_prior,
            "gft_prior": args.gft_prior,
            "gft_prior_temperature": args.gft_prior_temperature,
            "gft_prior_uniform_mix": args.gft_prior_uniform_mix,
            "gft_prior_weights": gft_weights_from_args(args),
            "gft_transport_cost_weights": gft_transport_weights_from_args(args),
            "gft_soft_deadlock_weights": gft_soft_deadlock_weights_from_args(args),
            "gft_commitment_beta": args.gft_commitment_beta,
            "gft_sinkhorn_iterations": args.gft_sinkhorn_iterations,
            "gft_sd_leaf_mode": args.gft_sd_leaf_mode,
            "gft_sd_leaf_weights": gft_sd_leaf_weights_from_args(args),
            "tva_value_blend": args.tva_value_blend,
            "tva_dynamic_blend": bool(args.tva_dynamic_blend),
            "tva_dynamic_high_risk_threshold": args.tva_dynamic_high_risk_threshold,
            "tva_dynamic_negative_delta_threshold": args.tva_dynamic_negative_delta_threshold,
            "tva_dynamic_high_risk_blend": args.tva_dynamic_high_risk_blend,
            "tva_dynamic_low_risk_threshold": args.tva_dynamic_low_risk_threshold,
            "tva_dynamic_positive_delta_threshold": args.tva_dynamic_positive_delta_threshold,
            "tva_dynamic_positive_blend": args.tva_dynamic_positive_blend,
            "tva_dynamic_revisit_penalty": args.tva_dynamic_revisit_penalty,
            "tva_dynamic_prior_penalty_weight": args.tva_dynamic_prior_penalty_weight,
            "tva_dynamic_prior_penalty_cap": args.tva_dynamic_prior_penalty_cap,
            "tva_dynamic_leaf_penalty_weight": args.tva_dynamic_leaf_penalty_weight,
            "tva_dynamic_leaf_penalty_cap": args.tva_dynamic_leaf_penalty_cap,
            "tva_dynamic_root_select": bool(args.tva_dynamic_root_select),
            "tva_dynamic_root_select_margin": args.tva_dynamic_root_select_margin,
            "tva_state_gated_blend": bool(args.tva_state_gated_blend),
            "tva_state_gated_default_blend": args.tva_state_gated_default_blend,
            "tva_state_gated_positive_blend": args.tva_state_gated_positive_blend,
            "tva_state_gated_positive_risk_max": args.tva_state_gated_positive_risk_max,
            "tva_state_gated_positive_delta_min": args.tva_state_gated_positive_delta_min,
            "tva_state_gated_negative_blend": args.tva_state_gated_negative_blend,
            "tva_state_gated_negative_risk_min": args.tva_state_gated_negative_risk_min,
            "tva_state_gated_negative_delta_min": args.tva_state_gated_negative_delta_min,
            "tva_state_gated_ood_z_threshold": args.tva_state_gated_ood_z_threshold,
            "tva_state_gated_ood_blend": args.tva_state_gated_ood_blend,
            "tva_transport_phase_blend": args.tva_transport_phase_blend,
            "tva_transport_phase_high_progress_min": args.tva_transport_phase_high_progress_min,
            "tva_transport_phase_low_progress_max": args.tva_transport_phase_low_progress_max,
            "tva_transport_phase_commitment_min": args.tva_transport_phase_commitment_min,
            "tva_transport_phase_max_closed_loop_step": args.tva_transport_phase_max_closed_loop_step,
            "tva_transport_phase_require_root_eligible": bool(args.tva_transport_phase_require_root_eligible),
        },
        "framework_contract": {
            "sd_provider": SokobanGFTSDModel.provider_id,
            "transition_provider": SokobanGFTSDModel.transition_provider_id,
            "ledger_state": "exact_sokoban_grid_state",
            "resource_state": "gft_phi_y_readout",
            "gft_spec_alignment": SokobanGFTSDModel.gft_spec_alignment(),
            "planner_action": "push_level_grid_flow",
            "legacy_six_stock_rollout_used": False,
            "teacher_used_online": False,
            "gft_resource_layer_replaces_transition_provider": False,
            "inactive_resources": [],
            "tva_checkpoint": tva_checkpoint.path if tva_checkpoint else None,
            "tva_dynamic_blend_uses_risk_as_gate": False,
            "tva_dynamic_blend_changes_transition_provider": False,
            "tva_dynamic_prior_penalty_uses_risk_as_gate": False,
            "tva_dynamic_prior_penalty_changes_transition_provider": False,
            "tva_dynamic_leaf_penalty_uses_risk_as_gate": False,
            "tva_dynamic_leaf_penalty_changes_transition_provider": False,
            "tva_dynamic_root_select_changes_transition_provider": False,
            "tva_state_gated_blend_changes_transition_provider": False,
            "tva_state_gated_blend_can_set_alpha_zero": bool(args.tva_state_gated_blend),
            "tva_transport_phase_blend_changes_transition_provider": False,
            "tva_leaf_formula": (
                None
                if not isinstance(tva_checkpoint, GFTTvaCheckpoint)
                else (
                    "V_leaf=V_SD+V_TVA"
                    if tva_checkpoint.target_mode == "gft_sd_residual"
                    else "V_leaf=(1-alpha(s))*V_SD+alpha(s)*V_psi"
                    if tva_checkpoint.target_mode == "direct_value" and bool(args.tva_state_gated_blend)
                    else "V_leaf=(1-alpha(s))*V_SD+alpha(s)*V_psi"
                    if tva_checkpoint.target_mode == "direct_value" and bool(args.tva_dynamic_blend)
                    else "V_leaf=(1-alpha)*V_SD+alpha*V_psi"
                    if tva_checkpoint.target_mode == "direct_value" and float(args.tva_value_blend) < 1.0
                    else "V_leaf=V_psi"
                )
            ),
        },
        "independent_gate_status": gate_status,
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
