#!/usr/bin/env python3
"""Common helpers for Hopper SD diagnostics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


ANGLE_STATE_CHANNELS = {
    "theta": DMControlHopperSDModel.PITCH_IDX,
}


def angle_diff_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))


def angle_diff_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(pred - target), np.cos(pred - target))


def mask_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if not np.any(mask):
        return {"count": 0}
    x = values[mask]
    return {
        "count": int(x.shape[0]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def build_phase_masks(
    model: DMControlHopperSDModel,
    feat_now: Dict[str, torch.Tensor],
    feat_next_true: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    grounded = (feat_now["any_contact_z"] <= model.STATE_GROUND_EPS).cpu().numpy().astype(bool)
    airborne_true = (feat_now["airborne_true"] >= 0.5).cpu().numpy().astype(bool)
    next_grounded = (feat_next_true["any_contact_z"] <= model.STATE_GROUND_EPS).cpu().numpy().astype(bool)
    next_airborne_true = (feat_next_true["airborne_true"] >= 0.5).cpu().numpy().astype(bool)
    body_collision = (feat_now["body_collision"] >= 0.5).cpu().numpy().astype(bool)
    support_touch = model.STATE_SUPPORT_TOUCH
    toe_only = (
        grounded
        & (feat_now["touch_toe"].cpu().numpy() >= support_touch)
        & (feat_now["touch_heel"].cpu().numpy() < support_touch)
    )
    heel_only = (
        grounded
        & (feat_now["touch_heel"].cpu().numpy() >= support_touch)
        & (feat_now["touch_toe"].cpu().numpy() < support_touch)
    )
    full_foot = grounded & (feat_now["full_foot_support"].cpu().numpy() >= support_touch)
    landing = airborne_true & next_grounded
    takeoff = grounded & next_airborne_true
    high_omega = (torch.abs(feat_now["pitch_rate"]) >= model.STATE_UPRIGHT_OMEGA_MAX).cpu().numpy().astype(bool)
    high_vz_impact = landing & (feat_now["vz"].cpu().numpy() <= model.DENSE_LANDING_VZ_TARGET)
    near_contact = (
        (feat_now["any_contact_z"] > model.STATE_GROUND_EPS)
        & (feat_now["any_contact_z"] <= 0.10)
    ).cpu().numpy().astype(bool)
    return {
        "all": np.ones_like(grounded, dtype=bool),
        "grounded": grounded,
        "airborne_true": airborne_true,
        "landing": landing,
        "takeoff": takeoff,
        "toe_only": toe_only,
        "heel_only": heel_only,
        "full_foot": full_foot,
        "body_collision": body_collision,
        "high_omega": high_omega,
        "high_vz_impact": high_vz_impact,
        "near_contact": near_contact,
    }


def state_abs_error_channels(
    pred_next: torch.Tensor,
    true_next: torch.Tensor,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    named_channels: Tuple[Tuple[str, int], ...] = (
        ("H", DMControlHopperSDModel.H_IDX),
        ("vx", DMControlHopperSDModel.VX_IDX),
        ("vz", DMControlHopperSDModel.VZ_IDX),
        ("theta", DMControlHopperSDModel.PITCH_IDX),
        ("omega", DMControlHopperSDModel.PITCH_RATE_IDX),
        ("touch_toe", DMControlHopperSDModel.TOE_TOUCH_IDX),
        ("touch_heel", DMControlHopperSDModel.HEEL_TOUCH_IDX),
        ("z_min", DMControlHopperSDModel.ZMIN_IDX),
    )
    pred_np = pred_next.detach().cpu().numpy()
    true_np = true_next.detach().cpu().numpy()
    for name, idx in named_channels:
        if name in ANGLE_STATE_CHANNELS:
            err = np.abs(angle_diff_np(pred_np[:, idx], true_np[:, idx]))
        else:
            err = np.abs(pred_np[:, idx] - true_np[:, idx])
        out[name] = err
    return out

