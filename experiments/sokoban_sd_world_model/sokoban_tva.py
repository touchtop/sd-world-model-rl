"""Shared terminal value alignment utilities for Sokoban SD-MCTS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.sokoban import SokobanGoalDirectedSDModel, SokobanState
from src.sokoban.env import ACTION_NAMES
from validate_sokoban_sd_design import STOCK_NAMES


TVA_STOCK_FEATURE_NAMES = tuple(f"stock:{name}" for name in STOCK_NAMES)
TVA_FEEDBACK_FEATURE_NAMES = (
    "feedback:target_pressure",
    "feedback:goal_completion",
    "feedback:mobility_deficit",
    "feedback:reachability_deficit",
    "feedback:congestion_pressure",
    "feedback:congestion_mobility_coupling",
    "feedback:congestion_reachability_coupling",
    "feedback:risk_pressure",
    "feedback:risk_congestion_coupling",
)
TVA_STRUCTURAL_FEATURE_NAMES = (
    "struct:boxes_on_goals_frac",
    "struct:matching_lower_bound_norm",
    "struct:reachable_count_norm",
    "struct:legal_push_count_norm",
    "struct:viable_push_count_norm",
    "struct:dead_square_box_frac",
    "struct:frozen_box_frac",
    "struct:matching_deadlock",
    "struct:deadlocked",
    "struct:solved",
    "struct:no_legal_pushes",
    "struct:goal_deficit",
    "struct:viable_push_deficit",
    "struct:risk_mobility_coupling",
    "struct:pressure_congestion_coupling",
)
TVA_FEATURE_NAMES = TVA_STOCK_FEATURE_NAMES + TVA_FEEDBACK_FEATURE_NAMES + TVA_STRUCTURAL_FEATURE_NAMES


class SokobanTvaValueNet(nn.Module):
    """Small normalized value network used only for MCTS leaf evaluation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def tva_feedback_features(stock: np.ndarray) -> np.ndarray:
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


def tva_feature_vector(
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
) -> np.ndarray:
    """Build the high-level abstract state vector for TVA.

    The first six entries are the SD stock state supplied by the planner. The
    remaining entries are normalized structural support scalars used to align
    terminal value, not to replace stock-flow transition dynamics.
    """

    stock_arr = np.asarray(stock, dtype=np.float32)
    if stock_arr.shape != (len(STOCK_NAMES),):
        raise ValueError(f"Expected {len(STOCK_NAMES)} stock values, got shape {stock_arr.shape}.")

    features = model.abstract_features(raw_state)
    n_boxes = max(model.level.n_boxes, 1)
    max_pushes = max(model.level.n_boxes * len(ACTION_NAMES), 1)
    floor_count = max(len(model.floor_cells), 1)
    board_scale = max(model.level.height + model.level.width, 1)
    if np.isfinite(features.matching_lower_bound):
        lb_norm = _clamp01(float(features.matching_lower_bound) / (n_boxes * board_scale))
    else:
        lb_norm = 1.0

    structural = np.array(
        [
            _safe_div(features.boxes_on_goals, n_boxes),
            lb_norm,
            _safe_div(features.reachable_count, floor_count),
            _safe_div(features.legal_push_count, max_pushes),
            _safe_div(features.viable_push_count, max_pushes),
            _safe_div(features.dead_square_boxes, n_boxes),
            _safe_div(features.frozen_boxes, n_boxes),
            float(features.matching_deadlock),
            float(features.deadlocked),
            float(features.solved),
            float(features.legal_push_count == 0),
            1.0 - _safe_div(features.boxes_on_goals, n_boxes),
            1.0 - _safe_div(features.viable_push_count, max_pushes),
            float(stock_arr[5]) * (1.0 - float(stock_arr[2])),
            float(stock_arr[1]) * float(stock_arr[4]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stock_arr, tva_feedback_features(stock_arr), structural], axis=0).astype(np.float32)


def relative_path(path: str | Path, root: Path) -> str:
    path = Path(path)
    return os.path.relpath(path, root) if path.is_absolute() else str(path)


def load_tva_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    expected = tuple(payload["feature_names"])
    if expected != TVA_FEATURE_NAMES:
        raise ValueError(
            "TVA feature contract mismatch: "
            f"checkpoint has {len(expected)} features, code expects {len(TVA_FEATURE_NAMES)}."
        )
    model = SokobanTvaValueNet(
        input_dim=int(payload["architecture"]["input_dim"]),
        hidden_dim=int(payload["architecture"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    payload["model"] = model
    return payload


def denormalize_value(normalized: torch.Tensor, target_mean: float, target_std: float) -> torch.Tensor:
    return normalized * float(target_std) + float(target_mean)


def predict_tva_value(
    payload: Dict[str, Any],
    model: SokobanGoalDirectedSDModel,
    raw_state: SokobanState,
    stock: np.ndarray,
    device: str | torch.device = "cpu",
) -> float:
    features = torch.tensor(
        tva_feature_vector(model, raw_state, stock),
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    with torch.no_grad():
        normalized = payload["model"](features)
        value = denormalize_value(
            normalized,
            float(payload["target_stats"]["mean"]),
            float(payload["target_stats"]["std"]),
        )
    return float(value.detach().cpu().view(-1)[0].item())


def feature_summary(feature_matrix: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if feature_matrix.size == 0:
        return {}
    return {
        name: {
            "mean": float(np.mean(feature_matrix[:, idx])),
            "std": float(np.std(feature_matrix[:, idx])),
            "min": float(np.min(feature_matrix[:, idx])),
            "max": float(np.max(feature_matrix[:, idx])),
        }
        for idx, name in enumerate(feature_names)
    }
