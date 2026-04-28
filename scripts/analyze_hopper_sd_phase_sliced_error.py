#!/usr/bin/env python3
"""Phase-sliced one-step error analysis for the Hopper SD dynamics core."""

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
from scripts.hopper_sd_diagnostics_common import build_phase_masks, mask_stats, state_abs_error_channels
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


def _feature_abs_errors(
    pred_feat: Dict[str, torch.Tensor],
    true_feat: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = (
        "support_contact",
        "full_foot_support",
        "body_contact_z",
        "foot_contact_z",
    )
    for key in keys:
        out[key] = torch.abs(pred_feat[key] - true_feat[key]).detach().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hopper SD one-step error by contact/flight phase.")
    parser.add_argument("--checkpoint", required=True, help="Path to hopper_sd_supervised.pt checkpoint.")
    parser.add_argument("--out", required=True, help="Path to output json.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["args"]
    set_seed(int(cfg["seed"]))
    dataset = collect_random_dataset(int(cfg["seed"]), int(cfg["episodes"]), int(cfg["max_steps"]))
    _, val_idx = split_indices(dataset.episode_ids, float(cfg["val_fraction"]), int(cfg["seed"]))

    model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
        use_transition_residual=bool(cfg.get("use_transition_residual", False)),
        transition_residual_scale=float(cfg.get("transition_residual_scale", 1.0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    raw = torch.tensor(dataset.raw_obs[val_idx], dtype=torch.float32)
    actions = torch.tensor(dataset.actions[val_idx], dtype=torch.float32)
    next_raw = torch.tensor(dataset.next_raw_obs[val_idx], dtype=torch.float32)

    with torch.no_grad():
        states = model.encode_observation(raw)
        true_next = model.encode_observation(next_raw)
        pred_next = model(states, actions)
        feat_now = model._state_reward_features(states)
        feat_true_next = model._state_reward_features(true_next)
        feat_pred_next = model._state_reward_features(pred_next)
        dominant_true = model._dominant_state_ids(feat_true_next)
        dominant_pred = model._dominant_state_ids(feat_pred_next)

    phase_masks = build_phase_masks(model, feat_now, feat_true_next)
    channel_errors = state_abs_error_channels(pred_next, true_next)
    channel_errors.update(_feature_abs_errors(feat_pred_next, feat_true_next))
    dominant_match = (dominant_true == dominant_pred).detach().cpu().numpy().astype(np.float32)

    result: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "dataset_args": cfg,
        "num_val_samples": int(len(val_idx)),
        "phases": {},
    }

    for phase_name, mask in phase_masks.items():
        phase_result: Dict[str, object] = {
            "count": int(mask.sum()),
            "dominant_state_match": mask_stats(dominant_match, mask),
            "errors": {},
        }
        for key, values in channel_errors.items():
            phase_result["errors"][key] = mask_stats(values, mask)
        result["phases"][phase_name] = phase_result

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
