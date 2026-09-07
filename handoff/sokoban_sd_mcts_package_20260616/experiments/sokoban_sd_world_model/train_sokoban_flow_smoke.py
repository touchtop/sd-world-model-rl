#!/usr/bin/env python3
"""Train a minimal learned flow model for the revised Sokoban SD design.

This is a validation smoke, not a full SD-MCTS experiment. It learns
``x_t, a_t -> f_t`` for diagnostic single-push transition labels, applies the
fixed incidence matrix ``B``, and compares the predicted abstract next stock
against a zero-flow baseline. Optional short rollout supervision checks whether
the learned stock-flow model remains consistent under repeated application.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES, DIRECTIONS, add_cells
from src.sokoban.sd_model import SokobanPushAction
from validate_sokoban_sd_design import (
    FLOW_NAMES,
    INCIDENCE_MATRIX,
    STOCK_NAMES,
    diagnostic_flows,
    iter_states,
    stock_proxy,
)

FEEDBACK_NAMES = (
    "target_pressure",
    "goal_completion",
    "mobility_deficit",
    "reachability_deficit",
    "congestion_pressure",
    "congestion_mobility_coupling",
    "congestion_reachability_coupling",
    "risk_pressure",
    "risk_congestion_coupling",
)

FROZEN_BASELINE_ID = "sokoban_six_stock_nonnegative_flow_rollout_sup_base_v1"

FLOW_PAIRS = (
    ("goal_fill_flow", "goal_unfill_flow"),
    ("assignment_relief_flow", "assignment_worsening_flow"),
    ("mobility_loss_flow", "mobility_recovery_flow"),
    ("reachability_loss_flow", "reachability_recovery_flow"),
    ("congestion_generation_flow", "congestion_release_flow"),
    ("risk_accumulation_flow", "risk_relief_flow"),
)
FLOW_PAIR_INDEXES = tuple((FLOW_NAMES.index(left), FLOW_NAMES.index(right)) for left, right in FLOW_PAIRS)
FLOW_PAIR_NAMES = tuple(f"{left}__{right}" for left, right in FLOW_PAIRS)


@dataclass
class FlowSample:
    stock_features: np.ndarray
    action_features: np.ndarray
    feedback_features: np.ndarray
    stock_before: np.ndarray
    stock_after: np.ndarray
    target_flows: np.ndarray
    level_id: str


@dataclass
class RolloutSample:
    stock_sequence: np.ndarray
    action_features_sequence: np.ndarray
    target_flows_sequence: np.ndarray
    level_id: str


class FlowNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        flow_dim: int,
        flow_parameterization: str,
        flow_pair_scales: torch.Tensor,
        capacity_projection_stock_mask: torch.Tensor,
        capacity_projection_passes: int,
    ):
        super().__init__()
        self.flow_parameterization = flow_parameterization
        if flow_parameterization not in {"softplus", "paired_net", "paired_bounded"}:
            raise ValueError(f"Unknown flow parameterization: {flow_parameterization}")
        output_dim = len(FLOW_PAIRS) if flow_parameterization in {"paired_net", "paired_bounded"} else flow_dim
        self.capacity_projection_passes = capacity_projection_passes
        self.register_buffer("flow_pair_scales", flow_pair_scales.detach().clone().float())
        self.register_buffer(
            "capacity_projection_stock_mask",
            capacity_projection_stock_mask.detach().clone().float(),
        )
        self.register_buffer("incidence", torch.tensor(INCIDENCE_MATRIX, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.flow_parameterization == "softplus":
            flows = F.softplus(raw)
        else:
            pair_logits = raw * self.flow_pair_scales.to(device=raw.device, dtype=raw.dtype)
            if self.flow_parameterization == "paired_net":
                flows = paired_net_flows(pair_logits, device=x.device, dtype=x.dtype)
            else:
                stock = x[..., : len(STOCK_NAMES)]
                flows = paired_bounded_flows(pair_logits, stock=stock, device=x.device, dtype=x.dtype)

        if torch.any(self.capacity_projection_stock_mask > 0):
            stock = x[..., : len(STOCK_NAMES)]
            flows = capacity_project_flows(
                flows,
                stock=stock,
                incidence=self.incidence,
                stock_mask=self.capacity_projection_stock_mask,
                passes=self.capacity_projection_passes,
            )
        return flows


def paired_net_flows(
    pair_logits: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map signed pair logits to twelve nonnegative mutually exclusive flows."""

    flows = torch.zeros(
        (*pair_logits.shape[:-1], len(FLOW_NAMES)),
        dtype=dtype,
        device=device,
    )
    positive = F.relu(pair_logits)
    negative = F.relu(-pair_logits)
    for pair_idx, (left_idx, right_idx) in enumerate(FLOW_PAIR_INDEXES):
        flows[..., left_idx] = positive[..., pair_idx]
        flows[..., right_idx] = negative[..., pair_idx]
    return flows


def paired_bounded_flows(
    pair_logits: torch.Tensor,
    *,
    stock: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map signed pair logits to mutually exclusive flows bounded by stock slack."""

    flows = torch.zeros(
        (*pair_logits.shape[:-1], len(FLOW_NAMES)),
        dtype=dtype,
        device=device,
    )
    magnitude = torch.tanh(torch.abs(pair_logits))
    positive_side = pair_logits >= 0.0

    g = stock[..., 0]
    u = stock[..., 1]
    m = stock[..., 2]
    r = stock[..., 3]
    c = stock[..., 4]
    i = stock[..., 5]
    pair_caps = (
        (torch.minimum(1.0 - g, u), torch.minimum(g, 1.0 - u)),  # goal fill / unfill
        (u, 1.0 - u),  # assignment relief / worsening
        (m, 1.0 - m),  # mobility loss / recovery
        (r, 1.0 - r),  # reachability loss / recovery
        (1.0 - c, c),  # congestion generation / release
        (1.0 - i, i),  # risk accumulation / relief
    )
    for pair_idx, (left_idx, right_idx) in enumerate(FLOW_PAIR_INDEXES):
        left_cap, right_cap = pair_caps[pair_idx]
        left_value = magnitude[..., pair_idx] * left_cap
        right_value = magnitude[..., pair_idx] * right_cap
        flows[..., left_idx] = torch.where(positive_side[..., pair_idx], left_value, torch.zeros_like(left_value))
        flows[..., right_idx] = torch.where(positive_side[..., pair_idx], torch.zeros_like(right_value), right_value)
    return flows


def _sum_selected_flows(flows: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    if indexes.numel() == 0:
        return torch.zeros(flows.shape[:-1], device=flows.device, dtype=flows.dtype)
    return flows.index_select(-1, indexes).sum(dim=-1)


def capacity_project_flows(
    flows: torch.Tensor,
    *,
    stock: torch.Tensor,
    incidence: torch.Tensor,
    stock_mask: torch.Tensor,
    passes: int,
) -> torch.Tensor:
    """Scale only active in/out flow groups that would move selected stocks outside [0, 1]."""

    projected = flows
    incidence = incidence.to(device=flows.device, dtype=flows.dtype)
    stock_mask = stock_mask.to(device=flows.device, dtype=flows.dtype)
    for _pass_idx in range(max(passes, 1)):
        for stock_idx in range(len(STOCK_NAMES)):
            if float(stock_mask[stock_idx].detach().cpu().item()) <= 0.0:
                continue
            row = incidence[stock_idx]
            pos_idx = torch.nonzero(row > 0, as_tuple=False).flatten()
            neg_idx = torch.nonzero(row < 0, as_tuple=False).flatten()
            pos_total = _sum_selected_flows(projected, pos_idx)
            neg_total = _sum_selected_flows(projected, neg_idx)
            delta = pos_total - neg_total
            upper_room = torch.clamp(1.0 - stock[..., stock_idx], min=0.0)
            lower_room = torch.clamp(stock[..., stock_idx], min=0.0)

            excess_high = torch.clamp(delta - upper_room, min=0.0)
            pos_scale = torch.clamp((pos_total - excess_high) / torch.clamp(pos_total, min=1e-12), 0.0, 1.0)
            if pos_idx.numel() > 0:
                scaled = projected.index_select(-1, pos_idx) * pos_scale.unsqueeze(-1)
                projected = projected.scatter(-1, pos_idx.expand(projected.shape[:-1] + pos_idx.shape), scaled)

            pos_total = _sum_selected_flows(projected, pos_idx)
            neg_total = _sum_selected_flows(projected, neg_idx)
            delta = pos_total - neg_total
            excess_low = torch.clamp(-lower_room - delta, min=0.0)
            neg_scale = torch.clamp((neg_total - excess_low) / torch.clamp(neg_total, min=1e-12), 0.0, 1.0)
            if neg_idx.numel() > 0:
                scaled = projected.index_select(-1, neg_idx) * neg_scale.unsqueeze(-1)
                projected = projected.scatter(-1, neg_idx.expand(projected.shape[:-1] + neg_idx.shape), scaled)
    return projected


class FeedbackFlowNet(nn.Module):
    """Flow network whose nonnegative flows are modulated by feedback gates."""

    def __init__(self, input_dim: int, feedback_dim: int, hidden_dim: int, flow_dim: int):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flow_dim),
        )
        self.feedback_context = nn.Sequential(
            nn.Linear(feedback_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feedback_gate_out = nn.Linear(hidden_dim, flow_dim)
        self.feedback_head = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feedback_dim),
        )

    def forward(self, x: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        flows, _pred_feedback = self.forward_details(x, feedback, predict_feedback=False)
        return flows

    def forward_details(
        self,
        x: torch.Tensor,
        feedback: torch.Tensor,
        *,
        predict_feedback: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_flows = F.softplus(self.base(x))
        feedback_context = self.feedback_context(feedback)
        gate = 0.5 + torch.sigmoid(self.feedback_gate_out(feedback_context))
        flows = base_flows * gate
        if not predict_feedback:
            return flows, None
        pred_feedback = torch.sigmoid(self.feedback_head(torch.cat([x, feedback_context], dim=-1)))
        return flows, pred_feedback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory to sample transitions from.",
    )
    parser.add_argument("--max-levels", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-states-per-level", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--flow-l1-weight",
        type=float,
        default=0.0,
        help="Optional L1 penalty on predicted flow magnitude. Default preserves existing baselines.",
    )
    parser.add_argument(
        "--raw-boundary-loss-weight",
        type=float,
        default=0.0,
        help="Optional penalty for raw x+Bf values outside [0, 1]. Default preserves existing baselines.",
    )
    parser.add_argument(
        "--raw-boundary-stock-weights",
        default="",
        help=(
            "Optional comma-separated six-stock weights for --raw-boundary-loss-weight. "
            "Default uses equal weights and preserves existing baselines when the loss weight is zero."
        ),
    )
    parser.add_argument(
        "--flow-pair-scales",
        default="",
        help=(
            "Optional comma-separated six pair-logit scales following FLOW_PAIRS order. "
            "Default all ones preserves existing paired_net/paired_bounded behavior."
        ),
    )
    parser.add_argument(
        "--capacity-projection-stock-mask",
        default="",
        help=(
            "Optional comma-separated six-stock 0/1 mask for a post-network capacity projection. "
            "Default all zeros disables projection and preserves existing baselines."
        ),
    )
    parser.add_argument(
        "--capacity-projection-passes",
        type=int,
        default=2,
        help="Number of sequential projection passes when --capacity-projection-stock-mask is enabled.",
    )
    parser.add_argument(
        "--rollout-depth",
        type=int,
        default=1,
        help="Number of consecutive single-push transitions used for rollout evaluation/training.",
    )
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.0,
        help="Weight for recurrent multi-step stock prediction loss. Default preserves one-step smoke.",
    )
    parser.add_argument(
        "--rollout-final-step-weight",
        type=float,
        default=1.0,
        help="Training weight multiplier for the final rollout step. Default preserves existing behavior.",
    )
    parser.add_argument(
        "--rollout-final-stock-weights",
        "--rollout-step2-stock-weights",
        dest="rollout_final_stock_weights",
        default="",
        help=(
            "Optional comma-separated six-stock weights used only on the final rollout training step. "
            "With --rollout-depth 2 this is the step-2 stock weighting. Default all ones preserves behavior."
        ),
    )
    parser.add_argument(
        "--max-rollout-samples-per-level",
        type=int,
        default=2000,
        help="Deterministic cap for complete rollout trajectories collected per level.",
    )
    parser.add_argument(
        "--model-kind",
        choices=("base", "feedback"),
        default="base",
        help="Use no-feedback base flow net or feedback-gated flow net.",
    )
    parser.add_argument(
        "--flow-parameterization",
        choices=("softplus", "paired_net", "paired_bounded"),
        default="softplus",
        help=(
            "softplus keeps the original independent nonnegative flows; paired_net "
            "uses signed net flows so each opposite-flow pair has at most one active side; "
            "paired_bounded additionally scales the active side by current stock slack."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("stock_action", "stock_only", "action_only"),
        default="stock_action",
        help="Choose whether the flow net input includes stocks, action features, or both.",
    )
    parser.add_argument(
        "--stock-loss-preset",
        choices=("uniform", "reach_congestion"),
        default="uniform",
        help="Preset for training stock-loss weights. Metrics remain unweighted.",
    )
    parser.add_argument(
        "--stock-loss-weights",
        default="",
        help="Optional comma-separated six-stock training weights overriding --stock-loss-preset.",
    )
    parser.add_argument(
        "--feedback-target-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for predicting next feedback features in feedback models.",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=("transition", "level"),
        default="transition",
        help="Use random transition split or hold out complete levels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alias-round-decimals",
        type=int,
        default=2,
        help="Decimal precision used for diagnostic abstract-state alias buckets.",
    )
    parser.add_argument(
        "--audit-top-k",
        type=int,
        default=8,
        help="Number of highest-variance alias buckets retained in the result audit.",
    )
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/flow_smoke.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="",
        help=(
            "Optional torch checkpoint path for the learned flow model. "
            "Default writes no checkpoint and preserves earlier smoke outputs."
        ),
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stock_loss_weights_from_args(args: argparse.Namespace) -> torch.Tensor:
    if args.stock_loss_weights:
        values = [float(value.strip()) for value in args.stock_loss_weights.split(",") if value.strip()]
        if len(values) != len(STOCK_NAMES):
            raise SystemExit(
                f"--stock-loss-weights must contain {len(STOCK_NAMES)} comma-separated values."
            )
        weights = np.array(values, dtype=np.float32)
    elif args.stock_loss_preset == "reach_congestion":
        weights = np.array([1.0, 1.0, 1.0, 2.5, 2.5, 1.0], dtype=np.float32)
    else:
        weights = np.ones(len(STOCK_NAMES), dtype=np.float32)

    if np.any(weights <= 0):
        raise SystemExit("Stock loss weights must be positive.")
    weights = weights / float(weights.mean())
    return torch.tensor(weights, dtype=torch.float32)


def parse_six_values(
    raw: str,
    *,
    default: list[float],
    name: str,
    normalize_positive_mean: bool = False,
) -> torch.Tensor:
    if raw:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
        if len(values) != len(STOCK_NAMES):
            raise SystemExit(f"{name} must contain {len(STOCK_NAMES)} comma-separated values.")
    else:
        values = default
    arr = np.array(values, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"{name} must contain finite values.")
    if normalize_positive_mean:
        if np.any(arr < 0):
            raise SystemExit(f"{name} weights must be nonnegative.")
        positive = arr[arr > 0]
        if positive.size:
            arr = arr / float(positive.mean())
    return torch.tensor(arr, dtype=torch.float32)


def parse_flow_pair_scales(raw: str) -> torch.Tensor:
    scales = parse_six_values(
        raw,
        default=[1.0] * len(FLOW_PAIRS),
        name="--flow-pair-scales",
        normalize_positive_mean=False,
    )
    if torch.any(scales < 0):
        raise SystemExit("--flow-pair-scales values must be nonnegative.")
    return scales


def parse_capacity_projection_stock_mask(raw: str) -> torch.Tensor:
    mask = parse_six_values(
        raw,
        default=[0.0] * len(STOCK_NAMES),
        name="--capacity-projection-stock-mask",
        normalize_positive_mean=False,
    )
    if torch.any((mask < 0) | (mask > 1)):
        raise SystemExit("--capacity-projection-stock-mask values must be in [0, 1].")
    return (mask > 0).float()


def parse_raw_boundary_stock_weights(raw: str) -> torch.Tensor:
    return parse_six_values(
        raw,
        default=[1.0] * len(STOCK_NAMES),
        name="--raw-boundary-stock-weights",
        normalize_positive_mean=True,
    )


def parse_rollout_final_stock_weights(raw: str) -> torch.Tensor:
    weights = parse_six_values(
        raw,
        default=[1.0] * len(STOCK_NAMES),
        name="--rollout-final-stock-weights",
        normalize_positive_mean=False,
    )
    if torch.any(weights <= 0):
        raise SystemExit("--rollout-final-stock-weights values must be positive.")
    return weights


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def frozen_baseline_contract(args: argparse.Namespace) -> Dict[str, Any]:
    required_config = {
        "model_kind": "base",
        "flow_parameterization": "softplus",
        "feature_mode": "stock_action",
        "stock_loss_preset": "uniform",
        "flow_l1_weight": 0.0,
        "raw_boundary_loss_weight": 0.0,
        "raw_boundary_stock_weights": [1.0] * len(STOCK_NAMES),
        "flow_pair_scales": [1.0] * len(FLOW_PAIRS),
        "capacity_projection_stock_mask": [0.0] * len(STOCK_NAMES),
        "capacity_projection_passes": 2,
        "feedback_target_loss_weight": 0.0,
        "rollout_depth": 2,
        "rollout_loss_weight": 1.0,
        "rollout_final_step_weight": 1.0,
        "rollout_final_stock_weights": [1.0] * len(STOCK_NAMES),
        "split_mode": "level",
        "action_scope": "legal_single_push_option",
        "transition_labels_from": "exact_single_push_diagnostic_provider",
        "sd_mcts_enabled": False,
    }
    observed_config = {
        "model_kind": args.model_kind,
        "flow_parameterization": args.flow_parameterization,
        "feature_mode": args.feature_mode,
        "stock_loss_preset": args.stock_loss_preset,
        "flow_l1_weight": args.flow_l1_weight,
        "raw_boundary_loss_weight": args.raw_boundary_loss_weight,
        "raw_boundary_stock_weights": [float(value) for value in parse_raw_boundary_stock_weights(args.raw_boundary_stock_weights).tolist()],
        "flow_pair_scales": [float(value) for value in parse_flow_pair_scales(args.flow_pair_scales).tolist()],
        "capacity_projection_stock_mask": [
            float(value) for value in parse_capacity_projection_stock_mask(args.capacity_projection_stock_mask).tolist()
        ],
        "capacity_projection_passes": args.capacity_projection_passes,
        "feedback_target_loss_weight": args.feedback_target_loss_weight,
        "rollout_depth": args.rollout_depth,
        "rollout_loss_weight": args.rollout_loss_weight,
        "rollout_final_step_weight": args.rollout_final_step_weight,
        "rollout_final_stock_weights": [
            float(value) for value in parse_rollout_final_stock_weights(args.rollout_final_stock_weights).tolist()
        ],
        "split_mode": args.split_mode,
        "action_scope": "legal_single_push_option",
        "transition_labels_from": "exact_single_push_diagnostic_provider",
        "sd_mcts_enabled": False,
    }
    return {
        "baseline_id": FROZEN_BASELINE_ID,
        "status": "diagnostic_learned_stock_flow_baseline",
        "is_frozen_minimal_baseline": observed_config == required_config,
        "required_config": required_config,
        "observed_config": observed_config,
    }


def _static_corner(model: SokobanGoalDirectedSDModel, cell: tuple[int, int]) -> float:
    vertical = model.level.is_wall((cell[0] - 1, cell[1])) or model.level.is_wall((cell[0] + 1, cell[1]))
    horizontal = model.level.is_wall((cell[0], cell[1] - 1)) or model.level.is_wall((cell[0], cell[1] + 1))
    return float(vertical and horizontal and cell not in model.level.goals)


def action_features(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    action: SokobanPushAction,
) -> np.ndarray:
    direction = DIRECTIONS[action.direction]
    box = state.boxes[action.box_index]
    target = add_cells(box, direction)
    direction_one_hot = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    direction_one_hot[action.direction] = 1.0
    n_boxes = max(model.level.n_boxes, 1)
    return np.concatenate(
        [
            direction_one_hot,
            np.array(
                [
                    float(box in model.level.goals),
                    float(target in model.level.goals),
                    float(target in model.dead_squares),
                    _static_corner(model, target),
                    _safe_div(action.box_index, n_boxes),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )


def feedback_features(stock: np.ndarray) -> np.ndarray:
    g, u, m, r, c, i = [float(v) for v in stock]
    return np.array(
        [
            u,
            g,
            1.0 - m,
            1.0 - r,
            c,
            c * (1.0 - m),
            c * (1.0 - r),
            i,
            i * c,
        ],
        dtype=np.float32,
    )


def build_samples(
    *,
    levels_path: str,
    max_levels: int,
    max_depth: int,
    max_states_per_level: int,
) -> List[FlowSample]:
    levels = load_boxoban_levels(levels_path, max_levels=max_levels)
    samples: list[FlowSample] = []
    for level in levels:
        model = SokobanGoalDirectedSDModel(level)
        for state, _depth in iter_states(model, max_depth=max_depth, max_states=max_states_per_level):
            x = stock_proxy(model, state).as_array()
            for action in model.legal_pushes(state):
                next_state = model.step_state(state, action)
                next_x = stock_proxy(model, next_state).as_array()
                delta = next_x - x
                samples.append(
                    FlowSample(
                        stock_features=x.astype(np.float32),
                        action_features=action_features(model, state, action),
                        feedback_features=feedback_features(x),
                        stock_before=x.astype(np.float32),
                        stock_after=next_x.astype(np.float32),
                        target_flows=diagnostic_flows(delta).astype(np.float32),
                        level_id=level.level_id,
                    )
                )
    return samples


def build_rollout_samples(
    *,
    levels_path: str,
    max_levels: int,
    max_depth: int,
    max_states_per_level: int,
    rollout_depth: int,
    max_rollout_samples_per_level: int,
) -> List[RolloutSample]:
    if rollout_depth < 1:
        return []

    levels = load_boxoban_levels(levels_path, max_levels=max_levels)
    rollouts: list[RolloutSample] = []
    for level in levels:
        model = SokobanGoalDirectedSDModel(level)
        level_count = 0
        for state, _depth in iter_states(model, max_depth=max_depth, max_states=max_states_per_level):
            if level_count >= max_rollout_samples_per_level:
                break

            start_stock = stock_proxy(model, state).as_array().astype(np.float32)

            def extend(
                current_state: SokobanState,
                stocks: list[np.ndarray],
                actions: list[np.ndarray],
                flows: list[np.ndarray],
            ) -> None:
                nonlocal level_count
                if level_count >= max_rollout_samples_per_level:
                    return
                if len(actions) == rollout_depth:
                    rollouts.append(
                        RolloutSample(
                            stock_sequence=np.stack(stocks).astype(np.float32),
                            action_features_sequence=np.stack(actions).astype(np.float32),
                            target_flows_sequence=np.stack(flows).astype(np.float32),
                            level_id=level.level_id,
                        )
                    )
                    level_count += 1
                    return

                current_stock = stocks[-1]
                for action in model.legal_pushes(current_state):
                    next_state = model.step_state(current_state, action)
                    next_stock = stock_proxy(model, next_state).as_array().astype(np.float32)
                    delta = next_stock - current_stock
                    extend(
                        next_state,
                        stocks + [next_stock],
                        actions + [action_features(model, current_state, action)],
                        flows + [diagnostic_flows(delta).astype(np.float32)],
                    )
                    if level_count >= max_rollout_samples_per_level:
                        break

            extend(state, [start_stock], [], [])
    return rollouts


def sample_features(sample: FlowSample, feature_mode: str) -> np.ndarray:
    if feature_mode == "stock_action":
        return np.concatenate([sample.stock_features, sample.action_features], axis=0)
    if feature_mode == "stock_only":
        return sample.stock_features
    if feature_mode == "action_only":
        return sample.action_features
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def tensor_features_from_stock_action(
    stock: torch.Tensor,
    action: torch.Tensor,
    feature_mode: str,
) -> torch.Tensor:
    if feature_mode == "stock_action":
        return torch.cat([stock, action], dim=-1)
    if feature_mode == "stock_only":
        return stock
    if feature_mode == "action_only":
        return action
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def split_indices(
    samples: List[FlowSample],
    valid_fraction: float,
    seed: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(samples))
    if split_mode == "transition":
        rng.shuffle(indices)
        valid_size = max(1, int(round(len(samples) * valid_fraction))) if len(samples) > 1 else 0
        valid_idx = indices[:valid_size]
        train_idx = indices[valid_size:]
        split_info = {
            "split_mode": split_mode,
            "train_levels": sorted({samples[int(i)].level_id for i in train_idx}),
            "valid_levels": sorted({samples[int(i)].level_id for i in valid_idx}),
        }
    elif split_mode == "level":
        levels = np.array(sorted({sample.level_id for sample in samples}), dtype=object)
        rng.shuffle(levels)
        valid_level_count = max(1, int(round(len(levels) * valid_fraction))) if len(levels) > 1 else 0
        valid_levels = set(levels[:valid_level_count].tolist())
        train_idx = np.array(
            [idx for idx, sample in enumerate(samples) if sample.level_id not in valid_levels],
            dtype=np.int64,
        )
        valid_idx = np.array(
            [idx for idx, sample in enumerate(samples) if sample.level_id in valid_levels],
            dtype=np.int64,
        )
        split_info = {
            "split_mode": split_mode,
            "train_levels": sorted({samples[int(i)].level_id for i in train_idx}),
            "valid_levels": sorted(valid_levels),
        }
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    if len(train_idx) == 0 and len(valid_idx) > 0:
        train_idx = valid_idx
    return train_idx, valid_idx, split_info


def split_rollout_indices(
    rollouts: List[RolloutSample],
    valid_fraction: float,
    seed: int,
    split_mode: str,
    split_info: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not rollouts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.RandomState(seed + 7919)
    indices = np.arange(len(rollouts))
    if split_mode == "level":
        valid_levels = set(split_info["valid_levels"])
        train_idx = np.array(
            [idx for idx, sample in enumerate(rollouts) if sample.level_id not in valid_levels],
            dtype=np.int64,
        )
        valid_idx = np.array(
            [idx for idx, sample in enumerate(rollouts) if sample.level_id in valid_levels],
            dtype=np.int64,
        )
    elif split_mode == "transition":
        rng.shuffle(indices)
        valid_size = max(1, int(round(len(rollouts) * valid_fraction))) if len(rollouts) > 1 else 0
        valid_idx = indices[:valid_size]
        train_idx = indices[valid_size:]
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    if len(train_idx) == 0 and len(valid_idx) > 0:
        train_idx = valid_idx
    return train_idx, valid_idx


def batch_iter(indices: np.ndarray, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed + epoch)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def tensors_from_samples(samples: List[FlowSample], feature_mode: str) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.tensor(
            np.stack([sample_features(s, feature_mode) for s in samples]),
            dtype=torch.float32,
        ),
        "feedback_features": torch.tensor(
            np.stack([s.feedback_features for s in samples]),
            dtype=torch.float32,
        ),
        "stock_before": torch.tensor(np.stack([s.stock_before for s in samples]), dtype=torch.float32),
        "stock_after": torch.tensor(np.stack([s.stock_after for s in samples]), dtype=torch.float32),
        "target_flows": torch.tensor(np.stack([s.target_flows for s in samples]), dtype=torch.float32),
    }


def tensors_from_rollouts(rollouts: List[RolloutSample]) -> Dict[str, torch.Tensor]:
    if not rollouts:
        return {}
    return {
        "stock_sequence": torch.tensor(
            np.stack([s.stock_sequence for s in rollouts]),
            dtype=torch.float32,
        ),
        "action_features_sequence": torch.tensor(
            np.stack([s.action_features_sequence for s in rollouts]),
            dtype=torch.float32,
        ),
        "target_flows_sequence": torch.tensor(
            np.stack([s.target_flows_sequence for s in rollouts]),
            dtype=torch.float32,
        ),
    }


def predict_next_stock(
    stock_before: torch.Tensor,
    flows: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    delta = flows @ incidence.T
    return torch.clamp(stock_before + delta, 0.0, 1.0)


def raw_next_stock(
    stock_before: torch.Tensor,
    flows: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    return stock_before + flows @ incidence.T


def boundary_violation_loss(raw_next: torch.Tensor, stock_weights: torch.Tensor | None = None) -> torch.Tensor:
    low = F.relu(-raw_next)
    high = F.relu(raw_next - 1.0)
    loss = low.pow(2) + high.pow(2)
    if stock_weights is not None:
        weights = stock_weights.to(device=raw_next.device, dtype=raw_next.dtype).view(1, -1)
        loss = loss * weights
    return torch.mean(loss)


def weighted_stock_mse(pred: torch.Tensor, target: torch.Tensor, stock_weights: torch.Tensor) -> torch.Tensor:
    weights = stock_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1)
    return torch.mean((pred - target) ** 2 * weights)


def feedback_features_tensor(stock: torch.Tensor) -> torch.Tensor:
    g = stock[..., 0:1]
    u = stock[..., 1:2]
    m = stock[..., 2:3]
    r = stock[..., 3:4]
    c = stock[..., 4:5]
    i = stock[..., 5:6]
    return torch.cat(
        [
            u,
            g,
            1.0 - m,
            1.0 - r,
            c,
            c * (1.0 - m),
            c * (1.0 - r),
            i,
            i * c,
        ],
        dim=-1,
    )


def forward_flows_from_tensors(
    model: nn.Module,
    features: torch.Tensor,
    feedback: torch.Tensor,
    model_kind: str,
) -> torch.Tensor:
    if model_kind == "feedback":
        return model(features, feedback)
    return model(features)


def forward_outputs_from_tensors(
    model: nn.Module,
    features: torch.Tensor,
    feedback: torch.Tensor,
    model_kind: str,
    *,
    predict_feedback: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if model_kind == "feedback" and predict_feedback:
        return model.forward_details(features, feedback, predict_feedback=True)  # type: ignore[attr-defined]
    return forward_flows_from_tensors(model, features, feedback, model_kind), None


def metrics(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    indices: np.ndarray,
    incidence: torch.Tensor,
    model_kind: str,
) -> Dict[str, float]:
    if len(indices) == 0:
        return {}
    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        pred_flows = forward_flows(model, data, idx, model_kind)
        pred_next = predict_next_stock(data["stock_before"][idx], pred_flows, incidence)
        target_next = data["stock_after"][idx]
        target_flows = data["target_flows"][idx]
        zero_next = data["stock_before"][idx]
        return {
            "stock_mae": float(torch.mean(torch.abs(pred_next - target_next)).item()),
            "stock_mse": float(torch.mean((pred_next - target_next) ** 2).item()),
            "flow_mae": float(torch.mean(torch.abs(pred_flows - target_flows)).item()),
            "flow_mse": float(torch.mean((pred_flows - target_flows) ** 2).item()),
            "zero_flow_stock_mae": float(torch.mean(torch.abs(zero_next - target_next)).item()),
            "zero_flow_stock_mse": float(torch.mean((zero_next - target_next) ** 2).item()),
        }


def dimensional_audit(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    indices: np.ndarray,
    incidence: torch.Tensor,
    model_kind: str,
) -> Dict[str, Any]:
    if len(indices) == 0:
        return {}
    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        pred_flows = forward_flows(model, data, idx, model_kind)
        raw_next = data["stock_before"][idx] + pred_flows @ incidence.T
        pred_next = torch.clamp(raw_next, 0.0, 1.0)
        target_next = data["stock_after"][idx]
        target_flows = data["target_flows"][idx]
        stock_abs = torch.abs(pred_next - target_next)
        flow_abs = torch.abs(pred_flows - target_flows)
        clipped = torch.logical_or(raw_next < 0.0, raw_next > 1.0)
        clipped_low = raw_next < 0.0
        clipped_high = raw_next > 1.0
        clip_correction = torch.abs(pred_next - raw_next)
        target_delta = target_next - data["stock_before"][idx]
        pred_delta = pred_next - data["stock_before"][idx]
        flow_index = {name: flow_idx for flow_idx, name in enumerate(FLOW_NAMES)}

        return {
            "per_stock_mae": {
                name: float(stock_abs[:, stock_idx].mean().item())
                for stock_idx, name in enumerate(STOCK_NAMES)
            },
            "per_stock_delta_mae": {
                name: float(torch.abs(pred_delta[:, stock_idx] - target_delta[:, stock_idx]).mean().item())
                for stock_idx, name in enumerate(STOCK_NAMES)
            },
            "per_flow_mae": {
                name: float(flow_abs[:, flow_idx].mean().item())
                for flow_idx, name in enumerate(FLOW_NAMES)
            },
            "clip_audit": {
                "any_stock_clipped_fraction": float(clipped.any(dim=1).float().mean().item()),
                "stock_value_clipped_fraction": float(clipped.float().mean().item()),
                "mean_abs_clip_correction": float(clip_correction.mean().item()),
                "max_abs_clip_correction": float(clip_correction.max().item()),
                "per_stock_clipped_fraction": {
                    name: float(clipped[:, stock_idx].float().mean().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_clipped_low_fraction": {
                    name: float(clipped_low[:, stock_idx].float().mean().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_clipped_high_fraction": {
                    name: float(clipped_high[:, stock_idx].float().mean().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_mean_abs_clip_correction": {
                    name: float(clip_correction[:, stock_idx].mean().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_max_abs_clip_correction": {
                    name: float(clip_correction[:, stock_idx].max().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_raw_next_min": {
                    name: float(raw_next[:, stock_idx].min().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_raw_next_max": {
                    name: float(raw_next[:, stock_idx].max().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
                "per_stock_raw_next_mean": {
                    name: float(raw_next[:, stock_idx].mean().item())
                    for stock_idx, name in enumerate(STOCK_NAMES)
                },
            },
            "per_stock_target_delta_summary": {
                name: {
                    "mean": float(target_delta[:, stock_idx].mean().item()),
                    "mean_abs": float(torch.abs(target_delta[:, stock_idx]).mean().item()),
                    "std": float(target_delta[:, stock_idx].std(unbiased=False).item()),
                    "min": float(target_delta[:, stock_idx].min().item()),
                    "max": float(target_delta[:, stock_idx].max().item()),
                    "nonzero_fraction": float(
                        (torch.abs(target_delta[:, stock_idx]) > 1e-12).float().mean().item()
                    ),
                }
                for stock_idx, name in enumerate(STOCK_NAMES)
            },
            "per_stock_pred_delta_summary": {
                name: {
                    "mean": float(pred_delta[:, stock_idx].mean().item()),
                    "mean_abs": float(torch.abs(pred_delta[:, stock_idx]).mean().item()),
                    "std": float(pred_delta[:, stock_idx].std(unbiased=False).item()),
                    "min": float(pred_delta[:, stock_idx].min().item()),
                    "max": float(pred_delta[:, stock_idx].max().item()),
                    "nonzero_fraction": float(
                        (torch.abs(pred_delta[:, stock_idx]) > 1e-12).float().mean().item()
                    ),
                }
                for stock_idx, name in enumerate(STOCK_NAMES)
            },
            "per_flow_target_summary": {
                name: {
                    "mean": float(target_flows[:, flow_idx].mean().item()),
                    "mean_abs": float(torch.abs(target_flows[:, flow_idx]).mean().item()),
                    "std": float(target_flows[:, flow_idx].std(unbiased=False).item()),
                    "min": float(target_flows[:, flow_idx].min().item()),
                    "max": float(target_flows[:, flow_idx].max().item()),
                    "nonzero_fraction": float(
                        (torch.abs(target_flows[:, flow_idx]) > 1e-12).float().mean().item()
                    ),
                }
                for flow_idx, name in enumerate(FLOW_NAMES)
            },
            "per_flow_pred_summary": {
                name: {
                    "mean": float(pred_flows[:, flow_idx].mean().item()),
                    "mean_abs": float(torch.abs(pred_flows[:, flow_idx]).mean().item()),
                    "std": float(pred_flows[:, flow_idx].std(unbiased=False).item()),
                    "min": float(pred_flows[:, flow_idx].min().item()),
                    "max": float(pred_flows[:, flow_idx].max().item()),
                    "nonzero_fraction": float(
                        (torch.abs(pred_flows[:, flow_idx]) > 1e-12).float().mean().item()
                    ),
                }
                for flow_idx, name in enumerate(FLOW_NAMES)
            },
            "flow_pair_coactivation": {
                f"{left}__{right}": {
                    "target_coactive_fraction_gt_1e-6": float(
                        torch.logical_and(
                            target_flows[:, flow_index[left]] > 1e-6,
                            target_flows[:, flow_index[right]] > 1e-6,
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "pred_coactive_fraction_gt_1e-6": float(
                        torch.logical_and(
                            pred_flows[:, flow_index[left]] > 1e-6,
                            pred_flows[:, flow_index[right]] > 1e-6,
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                    "pred_mean_min_pair_flow": float(
                        torch.minimum(
                            pred_flows[:, flow_index[left]],
                            pred_flows[:, flow_index[right]],
                        )
                        .mean()
                        .item()
                    ),
                    "target_mean_min_pair_flow": float(
                        torch.minimum(
                            target_flows[:, flow_index[left]],
                            target_flows[:, flow_index[right]],
                        )
                        .mean()
                        .item()
                    ),
                }
                for left, right in FLOW_PAIRS
            },
        }


def _alias_bucket_summary(
    buckets: dict[tuple[float, ...], list[np.ndarray]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key, deltas in buckets.items():
        if len(deltas) < 2:
            continue
        arr = np.stack(deltas, axis=0)
        per_stock_variance = np.var(arr, axis=0)
        rows.append(
            {
                "bucket_key": [float(value) for value in key],
                "count": int(len(deltas)),
                "next_delta_variance_sum": float(per_stock_variance.sum()),
                "per_stock_next_delta_variance": {
                    name: float(per_stock_variance[idx])
                    for idx, name in enumerate(STOCK_NAMES)
                },
                "mean_next_delta": {
                    name: float(arr[:, idx].mean())
                    for idx, name in enumerate(STOCK_NAMES)
                },
            }
        )
    rows.sort(key=lambda row: row["next_delta_variance_sum"], reverse=True)
    variances = [float(row["next_delta_variance_sum"]) for row in rows]
    return {
        "bucket_count": len(buckets),
        "aliased_bucket_count": len(rows),
        "max_next_delta_variance": max(variances) if variances else 0.0,
        "mean_next_delta_variance": float(np.mean(variances)) if variances else 0.0,
        "top_buckets": rows[:top_k],
    }


def aliasing_audit(
    samples: List[FlowSample],
    indices: np.ndarray,
    *,
    round_decimals: int,
    top_k: int,
) -> Dict[str, Any]:
    stock_buckets: dict[tuple[float, ...], list[np.ndarray]] = defaultdict(list)
    stock_action_buckets: dict[tuple[float, ...], list[np.ndarray]] = defaultdict(list)
    for idx in indices:
        sample = samples[int(idx)]
        delta = sample.stock_after - sample.stock_before
        stock_key = tuple(np.round(sample.stock_before, round_decimals).tolist())
        action_key = tuple(np.round(sample.action_features, round_decimals).tolist())
        stock_buckets[stock_key].append(delta)
        stock_action_buckets[stock_key + action_key].append(delta)

    return {
        "round_decimals": round_decimals,
        "top_k": top_k,
        "stock_bucket_aliasing": _alias_bucket_summary(stock_buckets, top_k=top_k),
        "stock_action_bucket_aliasing": _alias_bucket_summary(stock_action_buckets, top_k=top_k),
    }


def rollout_training_loss(
    model: nn.Module,
    rollout_data: Dict[str, torch.Tensor],
    indices: np.ndarray,
    incidence: torch.Tensor,
    model_kind: str,
    feature_mode: str,
    stock_weights: torch.Tensor,
    rollout_final_stock_weights: torch.Tensor,
    raw_boundary_stock_weights: torch.Tensor,
    feedback_target_loss_weight: float,
    raw_boundary_loss_weight: float,
    rollout_final_step_weight: float,
) -> torch.Tensor:
    idx = torch.tensor(indices, dtype=torch.long)
    stock_sequence = rollout_data["stock_sequence"][idx]
    actions = rollout_data["action_features_sequence"][idx]
    pred_stock = stock_sequence[:, 0, :]
    step_losses: list[torch.Tensor] = []
    feedback_losses: list[torch.Tensor] = []
    predict_feedback = feedback_target_loss_weight > 0 and model_kind == "feedback"
    for step in range(actions.shape[1]):
        features = tensor_features_from_stock_action(pred_stock, actions[:, step, :], feature_mode)
        feedback = feedback_features_tensor(pred_stock)
        pred_flows, pred_feedback = forward_outputs_from_tensors(
            model,
            features,
            feedback,
            model_kind,
            predict_feedback=predict_feedback,
        )
        raw_next = raw_next_stock(pred_stock, pred_flows, incidence)
        pred_stock = torch.clamp(raw_next, 0.0, 1.0)
        target_stock = stock_sequence[:, step + 1, :]
        step_weight = rollout_final_step_weight if step == actions.shape[1] - 1 else 1.0
        step_stock_weights = (
            stock_weights * rollout_final_stock_weights
            if step == actions.shape[1] - 1
            else stock_weights
        )
        step_losses.append(step_weight * weighted_stock_mse(pred_stock, target_stock, step_stock_weights))
        if raw_boundary_loss_weight > 0:
            step_losses.append(
                raw_boundary_loss_weight * boundary_violation_loss(raw_next, raw_boundary_stock_weights)
            )
        if pred_feedback is not None:
            feedback_losses.append(F.mse_loss(pred_feedback, feedback_features_tensor(target_stock)))

    loss = torch.stack(step_losses).mean()
    if feedback_losses:
        loss = loss + feedback_target_loss_weight * torch.stack(feedback_losses).mean()
    return loss


def rollout_metrics(
    model: nn.Module,
    rollout_data: Dict[str, torch.Tensor],
    indices: np.ndarray,
    incidence: torch.Tensor,
    model_kind: str,
    feature_mode: str,
) -> Dict[str, Any]:
    if not rollout_data or len(indices) == 0:
        return {}

    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        stock_sequence = rollout_data["stock_sequence"][idx]
        actions = rollout_data["action_features_sequence"][idx]
        target_flows = rollout_data["target_flows_sequence"][idx]
        pred_stock = stock_sequence[:, 0, :]
        zero_stock = stock_sequence[:, 0, :]
        stock_mae_by_step: dict[str, float] = {}
        stock_mse_by_step: dict[str, float] = {}
        zero_mae_by_step: dict[str, float] = {}
        flow_mae_by_step: dict[str, float] = {}
        per_stock_mae_by_step: dict[str, dict[str, float]] = {}

        for step in range(actions.shape[1]):
            features = tensor_features_from_stock_action(pred_stock, actions[:, step, :], feature_mode)
            feedback = feedback_features_tensor(pred_stock)
            pred_flows = forward_flows_from_tensors(model, features, feedback, model_kind)
            pred_stock = predict_next_stock(pred_stock, pred_flows, incidence)
            target_stock = stock_sequence[:, step + 1, :]
            stock_abs = torch.abs(pred_stock - target_stock)
            step_name = f"step_{step + 1}"
            stock_mae_by_step[step_name] = float(stock_abs.mean().item())
            stock_mse_by_step[step_name] = float(torch.mean((pred_stock - target_stock) ** 2).item())
            zero_mae_by_step[step_name] = float(torch.mean(torch.abs(zero_stock - target_stock)).item())
            flow_mae_by_step[step_name] = float(
                torch.mean(torch.abs(pred_flows - target_flows[:, step, :])).item()
            )
            per_stock_mae_by_step[step_name] = {
                stock_name: float(stock_abs[:, stock_idx].mean().item())
                for stock_idx, stock_name in enumerate(STOCK_NAMES)
            }

        mean_stock_mae = float(np.mean(list(stock_mae_by_step.values())))
        mean_zero_mae = float(np.mean(list(zero_mae_by_step.values())))
        return {
            "rollout_depth": int(actions.shape[1]),
            "stock_mae_by_step": stock_mae_by_step,
            "stock_mse_by_step": stock_mse_by_step,
            "zero_flow_stock_mae_by_step": zero_mae_by_step,
            "flow_mae_by_step": flow_mae_by_step,
            "per_stock_mae_by_step": per_stock_mae_by_step,
            "stock_mae_mean": mean_stock_mae,
            "zero_flow_stock_mae_mean": mean_zero_mae,
            "stock_mae_improvement_over_zero_flow_mean": (
                float((mean_zero_mae - mean_stock_mae) / mean_zero_mae) if mean_zero_mae > 0 else None
            ),
        }


def forward_flows(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    idx: torch.Tensor,
    model_kind: str,
) -> torch.Tensor:
    return forward_flows_from_tensors(
        model,
        data["features"][idx],
        data["feedback_features"][idx],
        model_kind,
    )


def train(args: argparse.Namespace) -> Dict[str, Any]:
    if args.rollout_depth < 1:
        raise SystemExit("--rollout-depth must be >= 1.")
    if args.rollout_loss_weight < 0:
        raise SystemExit("--rollout-loss-weight must be >= 0.")
    if args.rollout_final_step_weight <= 0:
        raise SystemExit("--rollout-final-step-weight must be > 0.")
    if args.flow_l1_weight < 0:
        raise SystemExit("--flow-l1-weight must be >= 0.")
    if args.raw_boundary_loss_weight < 0:
        raise SystemExit("--raw-boundary-loss-weight must be >= 0.")
    if args.capacity_projection_passes < 1:
        raise SystemExit("--capacity-projection-passes must be >= 1.")
    if args.feedback_target_loss_weight < 0:
        raise SystemExit("--feedback-target-loss-weight must be >= 0.")
    if args.feedback_target_loss_weight > 0 and args.model_kind != "feedback":
        raise SystemExit("--feedback-target-loss-weight requires --model-kind feedback.")
    if args.flow_parameterization in {"paired_net", "paired_bounded"} and args.model_kind != "base":
        raise SystemExit(
            "--flow-parameterization paired_net/paired_bounded is currently restricted to --model-kind base."
        )
    set_seed(args.seed)
    stock_weights = stock_loss_weights_from_args(args)
    rollout_final_stock_weights = parse_rollout_final_stock_weights(args.rollout_final_stock_weights)
    raw_boundary_stock_weights = parse_raw_boundary_stock_weights(args.raw_boundary_stock_weights)
    flow_pair_scales = parse_flow_pair_scales(args.flow_pair_scales)
    capacity_projection_stock_mask = parse_capacity_projection_stock_mask(args.capacity_projection_stock_mask)
    samples = build_samples(
        levels_path=args.levels,
        max_levels=args.max_levels,
        max_depth=args.max_depth,
        max_states_per_level=args.max_states_per_level,
    )
    if not samples:
        raise SystemExit("No transition samples generated.")

    rollouts = build_rollout_samples(
        levels_path=args.levels,
        max_levels=args.max_levels,
        max_depth=args.max_depth,
        max_states_per_level=args.max_states_per_level,
        rollout_depth=args.rollout_depth,
        max_rollout_samples_per_level=args.max_rollout_samples_per_level,
    )
    data = tensors_from_samples(samples, args.feature_mode)
    train_idx, valid_idx, split_info = split_indices(
        samples,
        args.valid_fraction,
        args.seed,
        args.split_mode,
    )
    rollout_data = tensors_from_rollouts(rollouts)
    train_rollout_idx, valid_rollout_idx = split_rollout_indices(
        rollouts,
        args.valid_fraction,
        args.seed,
        args.split_mode,
        split_info,
    )
    if args.rollout_loss_weight > 0 and len(train_rollout_idx) == 0:
        raise SystemExit("No rollout samples generated for rollout supervision.")

    if args.model_kind == "feedback":
        model: nn.Module = FeedbackFlowNet(
            data["features"].shape[1],
            data["feedback_features"].shape[1],
            args.hidden_dim,
            len(FLOW_NAMES),
        )
    else:
        model = FlowNet(
            data["features"].shape[1],
            args.hidden_dim,
            len(FLOW_NAMES),
            args.flow_parameterization,
            flow_pair_scales,
            capacity_projection_stock_mask,
            args.capacity_projection_passes,
        )
    incidence = torch.tensor(INCIDENCE_MATRIX, dtype=torch.float32)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        rollout_batches = (
            batch_iter(train_rollout_idx, args.batch_size, args.seed + 104729, epoch)
            if args.rollout_loss_weight > 0 and len(train_rollout_idx) > 0
            else []
        )
        for batch_num, batch in enumerate(batch_iter(train_idx, args.batch_size, args.seed, epoch)):
            idx = torch.tensor(batch, dtype=torch.long)
            predict_feedback = args.feedback_target_loss_weight > 0 and args.model_kind == "feedback"
            pred_flows, pred_feedback = forward_outputs_from_tensors(
                model,
                data["features"][idx],
                data["feedback_features"][idx],
                args.model_kind,
                predict_feedback=predict_feedback,
            )
            raw_next = raw_next_stock(data["stock_before"][idx], pred_flows, incidence)
            pred_next = torch.clamp(raw_next, 0.0, 1.0)
            stock_loss = weighted_stock_mse(pred_next, data["stock_after"][idx], stock_weights)
            flow_loss = F.mse_loss(pred_flows, data["target_flows"][idx])
            loss = stock_loss + args.flow_loss_weight * flow_loss
            if args.flow_l1_weight > 0:
                loss = loss + args.flow_l1_weight * torch.mean(torch.abs(pred_flows))
            if args.raw_boundary_loss_weight > 0:
                loss = loss + args.raw_boundary_loss_weight * boundary_violation_loss(
                    raw_next,
                    raw_boundary_stock_weights,
                )
            if pred_feedback is not None:
                target_feedback = feedback_features_tensor(data["stock_after"][idx])
                loss = loss + args.feedback_target_loss_weight * F.mse_loss(pred_feedback, target_feedback)
            if rollout_batches:
                rollout_batch = rollout_batches[batch_num % len(rollout_batches)]
                loss = loss + args.rollout_loss_weight * rollout_training_loss(
                    model,
                    rollout_data,
                    rollout_batch,
                    incidence,
                    args.model_kind,
                    args.feature_mode,
                    stock_weights,
                    rollout_final_stock_weights,
                    raw_boundary_stock_weights,
                    args.feedback_target_loss_weight,
                    args.raw_boundary_loss_weight,
                    args.rollout_final_step_weight,
                )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

    model.eval()
    train_metrics = metrics(model, data, train_idx, incidence, args.model_kind)
    valid_metrics = metrics(model, data, valid_idx, incidence, args.model_kind)
    train_dimensional_audit = dimensional_audit(model, data, train_idx, incidence, args.model_kind)
    valid_dimensional_audit = dimensional_audit(model, data, valid_idx, incidence, args.model_kind)
    train_rollout_metrics = rollout_metrics(
        model,
        rollout_data,
        train_rollout_idx,
        incidence,
        args.model_kind,
        args.feature_mode,
    )
    valid_rollout_metrics = rollout_metrics(
        model,
        rollout_data,
        valid_rollout_idx,
        incidence,
        args.model_kind,
        args.feature_mode,
    )
    valid_improvement = None
    if valid_metrics:
        baseline = valid_metrics["zero_flow_stock_mae"]
        learned = valid_metrics["stock_mae"]
        valid_improvement = float((baseline - learned) / baseline) if baseline > 0 else None

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": (
                "sokoban_flow_rollout_smoke_v1" if args.rollout_depth > 1 else "sokoban_flow_smoke_v1"
            ),
        },
        "baseline_contract": frozen_baseline_contract(args),
        "config": {
            "levels": os.path.relpath(args.levels, ROOT) if os.path.isabs(args.levels) else args.levels,
            "max_levels": args.max_levels,
            "max_depth": args.max_depth,
            "max_states_per_level": args.max_states_per_level,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "flow_loss_weight": args.flow_loss_weight,
            "flow_l1_weight": args.flow_l1_weight,
            "raw_boundary_loss_weight": args.raw_boundary_loss_weight,
            "raw_boundary_stock_weights": [float(value) for value in raw_boundary_stock_weights.tolist()],
            "flow_pair_names": FLOW_PAIR_NAMES,
            "flow_pair_scales": [float(value) for value in flow_pair_scales.tolist()],
            "capacity_projection_stock_mask": [
                float(value) for value in capacity_projection_stock_mask.tolist()
            ],
            "capacity_projection_passes": args.capacity_projection_passes,
            "rollout_depth": args.rollout_depth,
            "rollout_loss_weight": args.rollout_loss_weight,
            "rollout_final_step_weight": args.rollout_final_step_weight,
            "rollout_final_stock_weights": [float(value) for value in rollout_final_stock_weights.tolist()],
            "max_rollout_samples_per_level": args.max_rollout_samples_per_level,
            "model_kind": args.model_kind,
            "flow_parameterization": args.flow_parameterization,
            "feature_mode": args.feature_mode,
            "stock_loss_preset": args.stock_loss_preset,
            "stock_loss_weights": [float(value) for value in stock_weights.tolist()],
            "feedback_target_loss_weight": args.feedback_target_loss_weight,
            "valid_fraction": args.valid_fraction,
            "split_mode": args.split_mode,
            "seed": args.seed,
            "alias_round_decimals": args.alias_round_decimals,
            "audit_top_k": args.audit_top_k,
        },
        "design_scope": {
            "learned_component": "nonnegative_flow_model",
            "flow_parameterization": args.flow_parameterization,
            "opposite_flow_pair_exclusivity": args.flow_parameterization == "paired_net",
            "flow_l1_penalty_enabled": args.flow_l1_weight > 0,
            "raw_boundary_penalty_enabled": args.raw_boundary_loss_weight > 0,
            "raw_boundary_stock_weighting_enabled": bool(args.raw_boundary_stock_weights),
            "flow_pair_scaling_enabled": bool(args.flow_pair_scales),
            "capacity_projection_enabled": bool(torch.any(capacity_projection_stock_mask > 0).item()),
            "feedback_enabled": args.model_kind == "feedback",
            "rollout_supervision_enabled": args.rollout_loss_weight > 0,
            "feedback_target_enabled": args.feedback_target_loss_weight > 0,
            "sd_mcts_enabled": False,
            "action_scope": "legal_single_push_option",
            "transition_labels_from": "exact_single_push_diagnostic_provider",
        },
        "dataset": {
            "samples": len(samples),
            "rollout_samples": len(rollouts),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "train_rollout_samples": int(len(train_rollout_idx)),
            "valid_rollout_samples": int(len(valid_rollout_idx)),
            "input_dim": int(data["features"].shape[1]),
            "feedback_dim": int(data["feedback_features"].shape[1]),
            "stock_names": STOCK_NAMES,
            "flow_names": FLOW_NAMES,
            "feedback_names": FEEDBACK_NAMES,
            "levels_with_samples": len({sample.level_id for sample in samples}),
            "split": split_info,
        },
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "train_rollout": train_rollout_metrics,
            "valid_rollout": valid_rollout_metrics,
            "valid_stock_mae_improvement_over_zero_flow": valid_improvement,
        },
        "audit": {
            "train_dimensional": train_dimensional_audit,
            "valid_dimensional": valid_dimensional_audit,
            "valid_aliasing": aliasing_audit(
                samples,
                valid_idx,
                round_decimals=args.alias_round_decimals,
                top_k=args.audit_top_k,
            ),
            "provider_audit": {
                "state_encoder": "diagnostic_stock_proxy_encoder",
                "transition_label_provider": "exact_single_push_diagnostic_provider",
                "rollout_transition_provider": "exact_single_push_diagnostic_provider_for_labels",
                "learned_transition_provider_used_for_sdmcts": False,
            },
        },
    }
    if args.checkpoint_out:
        checkpoint_out = Path(args.checkpoint_out)
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "checkpoint_contract": {
                "provider_id": "learned_stock_flow_checkpoint_v1",
                "learned_component": "nonnegative_flow_model",
                "action_scope": "legal_single_push_option",
                "sd_mcts_ready": True,
            },
            "architecture": {
                "model_kind": args.model_kind,
                "feature_mode": args.feature_mode,
                "input_dim": int(data["features"].shape[1]),
                "feedback_dim": int(data["feedback_features"].shape[1]),
                "hidden_dim": args.hidden_dim,
                "flow_dim": len(FLOW_NAMES),
                "flow_parameterization": args.flow_parameterization,
                "flow_pair_scales": [float(value) for value in flow_pair_scales.tolist()],
                "capacity_projection_stock_mask": [
                    float(value) for value in capacity_projection_stock_mask.tolist()
                ],
                "capacity_projection_passes": args.capacity_projection_passes,
                "stock_names": STOCK_NAMES,
                "flow_names": FLOW_NAMES,
                "feedback_names": FEEDBACK_NAMES,
                "flow_pair_names": FLOW_PAIR_NAMES,
                "incidence_matrix": INCIDENCE_MATRIX.astype(float).tolist(),
            },
            "model_state_dict": model.state_dict(),
            "summary": summary,
        }
        torch.save(checkpoint_payload, checkpoint_out)
        summary["checkpoint"] = {
            "saved": True,
            "path": os.path.relpath(checkpoint_out, ROOT)
            if checkpoint_out.is_absolute()
            else str(checkpoint_out),
            "provider_id": "learned_stock_flow_checkpoint_v1",
        }
    else:
        summary["checkpoint"] = {
            "saved": False,
            "path": None,
            "provider_id": None,
        }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
