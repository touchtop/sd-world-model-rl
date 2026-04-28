#!/usr/bin/env python3
"""Evaluate Hopper SD dynamics on replay data from TD-MPC2 policy distribution."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_hopper_sd_supervised import key_metrics, rmse_tensor, valid_rollout_starts
from scripts.hopper_sd_diagnostics_common import build_phase_masks, mask_stats, state_abs_error_channels
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


@dataclass
class ReplayTransitionDataset:
    raw_obs: np.ndarray
    actions: np.ndarray
    next_raw_obs: np.ndarray
    dones: np.ndarray
    episode_ids: np.ndarray
    source_tags: np.ndarray


def _finite_row_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=1)


def _load_replay_dataset(paths: Sequence[Path]) -> ReplayTransitionDataset:
    raw_obs_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    next_raw_obs_list: List[np.ndarray] = []
    dones_list: List[np.ndarray] = []
    episode_ids_list: List[np.ndarray] = []
    source_tags_list: List[np.ndarray] = []
    ep_offset = 0

    for source_idx, path in enumerate(paths):
        state = torch.load(path, map_location="cpu", weights_only=False)
        episodes = list(state["buffer_episodes"])
        for ep in episodes:
            obs = ep.get("obs").detach().cpu().numpy().astype(np.float32)
            action = ep.get("action").detach().cpu().numpy().astype(np.float32)
            terminated = ep.get("terminated").detach().cpu().numpy().astype(np.float32)
            if obs.shape[0] < 2:
                ep_offset += 1
                continue
            cur_obs = obs[:-1]
            next_obs = obs[1:]
            act = action[1:]
            done = terminated[1:]
            valid = _finite_row_mask(cur_obs) & _finite_row_mask(next_obs) & _finite_row_mask(act)
            if not np.any(valid):
                ep_offset += 1
                continue
            raw_obs_list.append(cur_obs[valid])
            actions_list.append(act[valid])
            next_raw_obs_list.append(next_obs[valid])
            dones_list.append(done[valid])
            n = int(valid.sum())
            episode_ids_list.append(np.full((n,), ep_offset, dtype=np.int32))
            source_tags_list.append(np.full((n,), source_idx, dtype=np.int32))
            ep_offset += 1

    if not raw_obs_list:
        raise ValueError("No valid replay transitions found in the provided full-state checkpoints.")

    return ReplayTransitionDataset(
        raw_obs=np.concatenate(raw_obs_list, axis=0),
        actions=np.concatenate(actions_list, axis=0),
        next_raw_obs=np.concatenate(next_raw_obs_list, axis=0),
        dones=np.concatenate(dones_list, axis=0),
        episode_ids=np.concatenate(episode_ids_list, axis=0),
        source_tags=np.concatenate(source_tags_list, axis=0),
    )


def _infer_model_payload(payload: Dict[str, object]) -> tuple[Dict[str, torch.Tensor], Dict[str, object], bool]:
    if "model_state" in payload:
        model_state = payload["model_state"]
        cfg = dict(payload.get("args", {}))
        use_transition_residual = bool(cfg.get("use_transition_residual", False))
        return model_state, cfg, use_transition_residual
    if "model" in payload:
        src = payload["model"]
        model_state = {}
        for key, value in src.items():
            if key.startswith("_dynamics."):
                model_state[key[len("_dynamics."):]] = value
        if not model_state:
            raise ValueError("Checkpoint contains a TD-MPC2 model state, but no `_dynamics.*` weights were found.")
        use_transition_residual = any(k.startswith("transition_residual_head.") for k in model_state.keys())
        return model_state, {}, use_transition_residual
    raise ValueError("Unsupported checkpoint format. Expected `model_state` or `model`.")


def _load_dynamics_model(checkpoint: Path, transition_residual_scale: float | None) -> tuple[DMControlHopperSDModel, Dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state, cfg, use_transition_residual = _infer_model_payload(payload)
    residual_scale = transition_residual_scale
    if residual_scale is None:
        residual_scale = cfg.get("transition_residual_scale", None)
    if residual_scale is None:
        raise ValueError(
            "transition_residual_scale could not be inferred from the checkpoint. "
            "Please pass --transition-residual-scale explicitly."
        )
    model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
        use_transition_residual=use_transition_residual,
        transition_residual_scale=float(residual_scale),
    )
    target_state = model.state_dict()
    compatible = {
        k: v for k, v in model_state.items()
        if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
    }
    missing = sorted(set(target_state.keys()) - set(compatible.keys()))
    unexpected = sorted(k for k in model_state.keys() if k not in compatible)
    model.load_state_dict(compatible, strict=False)
    model.eval()
    meta = {
        "checkpoint": str(checkpoint),
        "use_transition_residual": use_transition_residual,
        "transition_residual_scale": float(residual_scale),
        "loaded_keys": len(compatible),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_args": cfg,
    }
    return model, meta


def _subsample_dataset(
    dataset: ReplayTransitionDataset,
    max_transitions: int | None,
    seed: int,
) -> ReplayTransitionDataset:
    n = dataset.raw_obs.shape[0]
    if max_transitions is None or max_transitions <= 0 or n <= max_transitions:
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_transitions, replace=False))
    return ReplayTransitionDataset(
        raw_obs=dataset.raw_obs[idx],
        actions=dataset.actions[idx],
        next_raw_obs=dataset.next_raw_obs[idx],
        dones=dataset.dones[idx],
        episode_ids=dataset.episode_ids[idx],
        source_tags=dataset.source_tags[idx],
    )


def _rollout_rmse_policy(
    model: DMControlHopperSDModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    dones: np.ndarray,
    horizon: int,
    max_starts: int,
    seed: int,
) -> Dict[str, float]:
    starts = valid_rollout_starts(dones, horizon)
    if starts.size == 0:
        return {"count": 0, "rmse": float("nan")}
    if starts.size > max_starts:
        rng = np.random.default_rng(seed + horizon)
        starts = np.sort(rng.choice(starts, size=max_starts, replace=False))
    errs: List[float] = []
    with torch.no_grad():
        for start in starts.tolist():
            pred = states[start:start + 1]
            for t in range(horizon):
                pred = model(pred, actions[start + t:start + t + 1])
            target = states[start + horizon:start + horizon + 1]
            errs.append(rmse_tensor(pred, target))
    return {
        "count": int(len(errs)),
        "rmse": float(np.mean(errs)) if errs else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hopper SD dynamics on TD-MPC2 replay distribution.")
    parser.add_argument("--model-checkpoint", required=True, help="Supervised checkpoint or TD-MPC2 .pt/.full.pt checkpoint.")
    parser.add_argument("--replay-full-state", nargs="+", required=True, help="One or more TD-MPC2 .full.pt checkpoints containing replay episodes.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--transition-residual-scale", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-transitions", type=int, default=0, help="Optional subsample size; 0 means use all transitions.")
    parser.add_argument("--rollout-horizons", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-rollout-starts", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    replay_paths = [Path(p) for p in args.replay_full_state]
    dataset = _load_replay_dataset(replay_paths)
    dataset = _subsample_dataset(dataset, args.max_transitions, args.seed)
    model, model_meta = _load_dynamics_model(Path(args.model_checkpoint), args.transition_residual_scale)
    model.to(args.device)

    raw = torch.tensor(dataset.raw_obs, dtype=torch.float32, device=args.device)
    actions = torch.tensor(dataset.actions, dtype=torch.float32, device=args.device)
    next_raw = torch.tensor(dataset.next_raw_obs, dtype=torch.float32, device=args.device)

    with torch.no_grad():
        states = model.encode_observation(raw)
        true_next = model.encode_observation(next_raw)
        pred_next = model(states, actions)
        feat_now = model._state_reward_features(states)
        feat_next_true = model._state_reward_features(true_next)
        pred_feat = model._state_reward_features(pred_next)
        dominant_true = model._dominant_state_ids(feat_next_true)
        dominant_pred = model._dominant_state_ids(pred_feat)

    phase_masks = build_phase_masks(model, feat_now, feat_next_true)
    state_errors = state_abs_error_channels(pred_next, true_next)
    dominant_match = (dominant_pred == dominant_true).to(dtype=torch.float32).cpu().numpy()

    phase_summary: Dict[str, object] = {}
    for phase_name, mask in phase_masks.items():
        phase_summary[phase_name] = {
            "count": int(mask.sum()),
            "dominant_state_match": mask_stats(dominant_match, mask),
            "errors": {name: mask_stats(values, mask) for name, values in state_errors.items()},
        }

    rollout_metrics = {}
    for horizon in sorted(set(int(h) for h in args.rollout_horizons if int(h) >= 1)):
        rollout_metrics[f"rollout_{horizon}"] = _rollout_rmse_policy(
            model=model,
            states=states,
            actions=actions,
            dones=dataset.dones,
            horizon=horizon,
            max_starts=args.max_rollout_starts,
            seed=args.seed,
        )

    source_counts = {}
    for idx, path in enumerate(replay_paths):
        mask = dataset.source_tags == idx
        source_counts[str(path)] = {
            "transitions": int(mask.sum()),
            "episodes": int(np.unique(dataset.episode_ids[mask]).shape[0]) if np.any(mask) else 0,
        }

    result = {
        "model": model_meta,
        "replay_sources": source_counts,
        "num_transitions": int(dataset.raw_obs.shape[0]),
        "num_episodes": int(np.unique(dataset.episode_ids).shape[0]),
        "one_step_struct_rmse": rmse_tensor(pred_next, true_next),
        "rollout_metrics": rollout_metrics,
        "key_metrics": key_metrics(model, pred_next, true_next),
        "phases": phase_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
