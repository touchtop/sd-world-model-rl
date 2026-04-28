#!/usr/bin/env python3
"""Analyze how much the Hopper transition residual is doing.

This script compares the explicit SD base delta against the learned
transition-residual delta on the validation split of the same supervised
dataset protocol used by ``fit_hopper_sd_supervised.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_hopper_sd_supervised import collect_random_dataset, set_seed, split_indices
from scripts.hopper_sd_diagnostics_common import build_phase_masks, mask_stats
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


CHANNELS = (
    ("vx", DMControlHopperSDModel.VX_IDX, 0),
    ("vz", DMControlHopperSDModel.VZ_IDX, 1),
    ("omega", DMControlHopperSDModel.PITCH_RATE_IDX, 2),
)


def _compatible_state_dict(model_state: Dict[str, torch.Tensor], target: DMControlHopperSDModel) -> Dict[str, torch.Tensor]:
    target_state = target.state_dict()
    return {k: v for k, v in model_state.items() if k in target_state}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hopper transition residual usage on supervised validation data.")
    parser.add_argument("--checkpoint", required=True, help="Path to hopper_sd_supervised.pt checkpoint.")
    parser.add_argument("--out", required=True, help="Path to output json.")
    parser.add_argument("--ratio-eps", type=float, default=1e-4)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["args"]
    if not cfg.get("use_transition_residual", False):
        raise ValueError("Checkpoint does not enable transition residual.")

    set_seed(int(cfg["seed"]))
    dataset = collect_random_dataset(int(cfg["seed"]), int(cfg["episodes"]), int(cfg["max_steps"]))
    _, val_idx = split_indices(dataset.episode_ids, float(cfg["val_fraction"]), int(cfg["seed"]))

    residual_model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
        use_transition_residual=True,
        transition_residual_scale=float(cfg.get("transition_residual_scale", 1.0)),
    )
    residual_model.load_state_dict(ckpt["model_state"], strict=True)
    residual_model.eval()

    base_model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
        use_transition_residual=False,
    )
    base_model.load_state_dict(_compatible_state_dict(ckpt["model_state"], base_model), strict=False)
    base_model.eval()

    raw = torch.tensor(dataset.raw_obs[val_idx], dtype=torch.float32)
    actions = torch.tensor(dataset.actions[val_idx], dtype=torch.float32)

    with torch.no_grad():
        states = base_model.encode_observation(raw)
        true_next = base_model.encode_observation(torch.tensor(dataset.next_raw_obs[val_idx], dtype=torch.float32))
        next_base = base_model(states, actions)
        next_resid = residual_model(states, actions)
        feat = base_model._state_reward_features(states)
        feat_next_true = base_model._state_reward_features(true_next)

    phase_masks = build_phase_masks(base_model, feat, feat_next_true)

    result: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "dataset_args": cfg,
        "num_val_samples": int(len(val_idx)),
        "channels": {},
    }

    state_np = states.cpu().numpy()
    next_base_np = next_base.cpu().numpy()
    next_resid_np = next_resid.cpu().numpy()

    for name, state_idx, resid_idx in CHANNELS:
        base_delta = next_base_np[:, state_idx] - state_np[:, state_idx]
        resid_delta = next_resid_np[:, state_idx] - next_base_np[:, state_idx]
        total_delta = next_resid_np[:, state_idx] - state_np[:, state_idx]
        abs_base = np.abs(base_delta)
        abs_resid = np.abs(resid_delta)
        ratio = abs_resid / np.maximum(abs_base, args.ratio_eps)
        sign_match = np.sign(base_delta) == np.sign(resid_delta)
        tanh_output = resid_delta / (
            float(cfg.get("transition_residual_scale", 1.0))
            * float(residual_model.transition_residual_base_scale[resid_idx].item())
        )
        tanh_abs = np.abs(np.clip(tanh_output, -1.0, 1.0))
        saturation = (tanh_abs > 0.95).astype(np.float32)

        channel_stats = {
            "global": {
                "mean_abs_base_delta": float(np.mean(abs_base)),
                "mean_abs_resid_delta": float(np.mean(abs_resid)),
                "mean_abs_total_delta": float(np.mean(np.abs(total_delta))),
                "mean_ratio_resid_over_base": float(np.mean(ratio)),
                "median_ratio_resid_over_base": float(np.median(ratio)),
                "p90_ratio_resid_over_base": float(np.percentile(ratio, 90)),
                "p95_ratio_resid_over_base": float(np.percentile(ratio, 95)),
                "mean_signed_resid_delta": float(np.mean(resid_delta)),
                "mean_sign_match_with_base": float(np.mean(sign_match.astype(np.float32))),
                "mean_abs_tanh_output": float(np.mean(tanh_abs)),
                "saturation_rate": float(np.mean(saturation)),
            },
            "by_phase": {},
        }
        for phase_name, mask in phase_masks.items():
            channel_stats["by_phase"][phase_name] = {
                "abs_base_delta": mask_stats(abs_base, mask),
                "abs_resid_delta": mask_stats(abs_resid, mask),
                "ratio_resid_over_base": mask_stats(ratio, mask),
                "signed_resid_delta": mask_stats(resid_delta, mask),
                "abs_tanh_output": mask_stats(tanh_abs, mask),
                "saturation": mask_stats(saturation, mask),
                "mean_sign_match_with_base": float(np.mean(sign_match[mask].astype(np.float32))) if np.any(mask) else float("nan"),
            }
        result["channels"][name] = channel_stats

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
