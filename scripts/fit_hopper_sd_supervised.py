#!/usr/bin/env python3
"""Supervised fitting and rollout evaluation for the Hopper SD dynamics core.

This script deliberately avoids TD-MPC2 training. The goal is to answer a
single question first: can the structured Hopper dynamics fit simulator
transitions well enough to serve as a world-model core?
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The current environment can import and step dm_control under ``glfw`` even
# without a visible DISPLAY, while ``egl``/``osmesa`` are unavailable here.
os.environ.setdefault("MUJOCO_GL", "glfw")

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel

ANGLE_CHANNELS = (3, 6, 7, 8, 9)
SHORT_HORIZON_CHANNELS = (
    DMControlHopperSDModel.VX_IDX,
    DMControlHopperSDModel.VZ_IDX,
    DMControlHopperSDModel.PITCH_RATE_IDX,
)
TOUCH_CHANNELS = (
    DMControlHopperSDModel.TOE_TOUCH_IDX,
    DMControlHopperSDModel.HEEL_TOUCH_IDX,
)
SHORT_HORIZON_CHANNEL_WEIGHTS = torch.tensor(
    [
        1.0,  # vx
        1.0,  # vz
        1.5,  # omega is still the most stubborn short-horizon channel
    ],
    dtype=torch.float32,
)
STATE_LOSS_WEIGHTS = torch.tensor(
    [
        1.0,   # H
        3.0,   # vx
        3.0,   # vz
        1.5,   # theta
        3.0,   # omega
        0.1,   # activation proxy
        1.0,   # waist
        1.0,   # hip
        1.0,   # knee
        1.0,   # ankle
        1.0,   # waist_dot
        1.0,   # hip_dot
        1.0,   # knee_dot
        1.0,   # ankle_dot
        0.2,   # touch_toe
        0.2,   # touch_heel
        0.2,   # z_min
    ],
    dtype=torch.float32,
)


@dataclass
class TransitionDataset:
    raw_obs: np.ndarray
    actions: np.ndarray
    next_raw_obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_ids: np.ndarray


def _inv_contact(x: float) -> float:
    x = min(max(x, 0.0), 1.0 - 1e-6)
    return -np.log(max(1.0 - x, 1e-6))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_timestep_obs(timestep) -> np.ndarray:
    return DMControlHopperSDModel.flatten_obs(timestep.observation)


def collect_random_dataset(
    seed: int,
    episodes: int,
    max_steps: int,
) -> TransitionDataset:
    from dm_control import suite

    env = suite.load("hopper", "hop", task_kwargs={"random": seed})
    rng = np.random.default_rng(seed)
    raw_obs: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_raw_obs: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[float] = []
    episode_ids: List[int] = []

    for episode_id in range(episodes):
        ts = env.reset()
        obs = flatten_timestep_obs(ts)
        for _step in range(max_steps):
            action = rng.uniform(low=-1.0, high=1.0, size=(DMControlHopperSDModel.ACTION_DIM,)).astype(np.float32)
            next_ts = env.step(action)
            next_obs = flatten_timestep_obs(next_ts)
            raw_obs.append(obs.copy())
            actions.append(action.copy())
            next_raw_obs.append(next_obs.copy())
            rewards.append(float(next_ts.reward or 0.0))
            dones.append(float(next_ts.last()))
            episode_ids.append(episode_id)
            obs = next_obs
            if next_ts.last():
                break

    return TransitionDataset(
        raw_obs=np.asarray(raw_obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_raw_obs=np.asarray(next_raw_obs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )


def collect_trace_dataset(trace_dir: str) -> TransitionDataset:
    trace_paths = sorted(Path(trace_dir).glob("episode_*.csv"))
    if not trace_paths:
        raise ValueError(f"No episode_*.csv files found under {trace_dir}")
    raw_obs: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_raw_obs: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[float] = []
    episode_ids: List[int] = []

    for episode_id, path in enumerate(trace_paths):
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            continue
        for i in range(len(rows) - 1):
            row = rows[i]
            nxt = rows[i + 1]
            raw_obs.append(
                np.array(
                    [
                        float(row["qpos_rootz"]),
                        float(row["qpos_rooty"]),
                        float(row["qpos_waist"]),
                        float(row["qpos_hip"]),
                        float(row["qpos_knee"]),
                        float(row["qpos_ankle"]),
                        float(row["qvel_rootx"]),
                        float(row["qvel_rootz"]),
                        float(row["qvel_rooty"]),
                        float(row["qvel_waist"]),
                        float(row["qvel_hip"]),
                        float(row["qvel_knee"]),
                        float(row["qvel_ankle"]),
                        _inv_contact(float(row["touch_toe"])),
                        _inv_contact(float(row["touch_heel"])),
                    ],
                    dtype=np.float32,
                )
            )
            next_raw_obs.append(
                np.array(
                    [
                        float(nxt["qpos_rootz"]),
                        float(nxt["qpos_rooty"]),
                        float(nxt["qpos_waist"]),
                        float(nxt["qpos_hip"]),
                        float(nxt["qpos_knee"]),
                        float(nxt["qpos_ankle"]),
                        float(nxt["qvel_rootx"]),
                        float(nxt["qvel_rootz"]),
                        float(nxt["qvel_rooty"]),
                        float(nxt["qvel_waist"]),
                        float(nxt["qvel_hip"]),
                        float(nxt["qvel_knee"]),
                        float(nxt["qvel_ankle"]),
                        _inv_contact(float(nxt["touch_toe"])),
                        _inv_contact(float(nxt["touch_heel"])),
                    ],
                    dtype=np.float32,
                )
            )
            actions.append(
                np.array(
                    [
                        float(row["action_0"]),
                        float(row["action_1"]),
                        float(row["action_2"]),
                        float(row["action_3"]),
                    ],
                    dtype=np.float32,
                )
            )
            rewards.append(float(nxt.get("reward", 0.0) or 0.0))
            dones.append(float(nxt.get("done", 0.0) or 0.0))
            episode_ids.append(episode_id)

    return TransitionDataset(
        raw_obs=np.asarray(raw_obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_raw_obs=np.asarray(next_raw_obs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )


def structured_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Only true angles should use wrapped residuals. Root velocities and omega
    # need explicit supervision; otherwise the current calibration target is too weak.
    err = pred - target
    angle_idx = torch.as_tensor(ANGLE_CHANNELS, dtype=torch.long, device=pred.device)
    angle_err = torch.atan2(
        torch.sin(pred.index_select(dim=1, index=angle_idx) - target.index_select(dim=1, index=angle_idx)),
        torch.cos(pred.index_select(dim=1, index=angle_idx) - target.index_select(dim=1, index=angle_idx)),
    )
    err = err.clone()
    err[:, angle_idx] = angle_err
    per_dim = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none")
    weights = STATE_LOSS_WEIGHTS.to(device=pred.device, dtype=pred.dtype).unsqueeze(0)
    return (per_dim * weights).mean()


def short_horizon_channel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    idx = torch.as_tensor(SHORT_HORIZON_CHANNELS, dtype=torch.long, device=pred.device)
    pred_sel = pred.index_select(dim=1, index=idx)
    target_sel = target.index_select(dim=1, index=idx)
    per_dim = F.smooth_l1_loss(pred_sel, target_sel, reduction="none")
    weights = SHORT_HORIZON_CHANNEL_WEIGHTS.to(device=pred.device, dtype=pred.dtype).unsqueeze(0)
    return (per_dim * weights).mean()


def touch_channel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    idx = torch.as_tensor(TOUCH_CHANNELS, dtype=torch.long, device=pred.device)
    pred_sel = pred.index_select(dim=1, index=idx)
    target_sel = target.index_select(dim=1, index=idx)
    return F.smooth_l1_loss(pred_sel, target_sel)


def rmse_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def angle_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    err = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
    return float(torch.mean(torch.abs(err)).item())


def split_indices(episode_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_episodes = np.unique(episode_ids)
    if unique_episodes.size < 2:
        raise ValueError("At least 2 episodes are required for an episode-level train/validation split.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_episodes)
    n_val = int(round(unique_episodes.size * val_fraction))
    n_val = min(max(1, n_val), unique_episodes.size - 1)
    val_episodes = np.sort(perm[:n_val])
    train_episodes = np.sort(perm[n_val:])
    val_idx = np.flatnonzero(np.isin(episode_ids, val_episodes))
    train_idx = np.flatnonzero(np.isin(episode_ids, train_episodes))
    return train_idx, val_idx


def valid_rollout_starts(dones: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("rollout horizon must be >= 1")
    starts: List[int] = []
    for start in range(len(dones) - horizon):
        if dones[start:start + horizon].any():
            continue
        starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def rollout_rmse(
    model: DMControlHopperSDModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    dones: np.ndarray,
    horizon: int,
    max_starts: int,
) -> float:
    starts = valid_rollout_starts(dones, horizon)
    if starts.size == 0:
        return float("nan")
    starts = starts[: min(max_starts, starts.size)]
    errs = []
    with torch.no_grad():
        for start in starts.tolist():
            pred = states[start:start + 1]
            for t in range(horizon):
                pred = model(pred, actions[start + t:start + t + 1])
            target = states[start + horizon:start + horizon + 1]
            errs.append(rmse_tensor(pred, target))
    return float(np.mean(errs))


def key_metrics(model: DMControlHopperSDModel, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_feat = model._state_reward_features(pred)
    true_feat = model._state_reward_features(target)
    return {
        "height_mae": float(torch.mean(torch.abs(pred_feat["height"] - true_feat["height"])).item()),
        "vx_mae": float(torch.mean(torch.abs(pred_feat["vx"] - true_feat["vx"])).item()),
        "vz_mae": float(torch.mean(torch.abs(pred_feat["vz"] - true_feat["vz"])).item()),
        "pitch_mae": angle_mae(pred_feat["pitch"], true_feat["pitch"]),
        "pitch_rate_mae": float(torch.mean(torch.abs(pred_feat["pitch_rate"] - true_feat["pitch_rate"])).item()),
        "support_contact_mae": float(
            torch.mean(torch.abs(pred_feat["support_contact"] - true_feat["support_contact"])).item()
        ),
        "full_foot_support_mae": float(
            torch.mean(torch.abs(pred_feat["full_foot_support"] - true_feat["full_foot_support"])).item()
        ),
        "body_contact_z_mae": float(
            torch.mean(torch.abs(pred_feat["body_contact_z"] - true_feat["body_contact_z"])).item()
        ),
    }


def fit_model(
    model: DMControlHopperSDModel,
    dataset: TransitionDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    rollout_horizon: int,
    rollout_weight: float,
    rollout_batch_size: int,
    short_rollout_horizons: List[int],
    short_rollout_weight: float,
    touch_loss_weight: float,
    device: str,
) -> Dict[str, object]:
    model.to(device)
    train_raw = torch.tensor(dataset.raw_obs[train_idx], dtype=torch.float32, device=device)
    train_actions = torch.tensor(dataset.actions[train_idx], dtype=torch.float32, device=device)
    train_next_raw = torch.tensor(dataset.next_raw_obs[train_idx], dtype=torch.float32, device=device)
    val_raw = torch.tensor(dataset.raw_obs[val_idx], dtype=torch.float32, device=device)
    val_actions = torch.tensor(dataset.actions[val_idx], dtype=torch.float32, device=device)
    val_next_raw = torch.tensor(dataset.next_raw_obs[val_idx], dtype=torch.float32, device=device)

    train_states = model.encode_observation(train_raw)
    train_next_states = model.encode_observation(train_next_raw)
    val_states = model.encode_observation(val_raw)
    val_next_states = model.encode_observation(val_next_raw)
    train_rollout_starts = valid_rollout_starts(dataset.dones[train_idx], rollout_horizon)
    short_rollout_start_map = {
        int(h): valid_rollout_starts(dataset.dones[train_idx], int(h))
        for h in short_rollout_horizons
        if int(h) > 1
    }

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        perm = np.random.permutation(len(train_idx))
        train_losses: List[float] = []
        model.train()
        for start in range(0, len(perm), batch_size):
            batch = perm[start:start + batch_size]
            s = train_states[batch]
            a = train_actions[batch]
            ns = train_next_states[batch]
            pred = model(s, a)
            loss = structured_state_loss(pred, ns)
            if touch_loss_weight > 0.0:
                loss = loss + touch_loss_weight * touch_channel_loss(pred, ns)
            if rollout_horizon > 1 and rollout_weight > 0.0 and train_rollout_starts.size > 0:
                rollout_sample_size = min(rollout_batch_size, len(batch), train_rollout_starts.size)
                rollout_choice = np.random.randint(0, train_rollout_starts.size, size=rollout_sample_size)
                rollout_starts = train_rollout_starts[rollout_choice]
                rollout_starts_t = torch.as_tensor(rollout_starts, dtype=torch.long, device=device)
                rollout_pred = train_states[rollout_starts_t]
                for t in range(rollout_horizon):
                    rollout_pred = model(rollout_pred, train_actions[rollout_starts_t + t])
                rollout_target = train_states[rollout_starts_t + rollout_horizon]
                rollout_loss = structured_state_loss(rollout_pred, rollout_target)
                loss = loss + rollout_weight * rollout_loss
            if short_rollout_weight > 0.0 and short_rollout_start_map:
                short_losses: List[torch.Tensor] = []
                for horizon, rollout_starts in short_rollout_start_map.items():
                    if rollout_starts.size == 0:
                        continue
                    rollout_sample_size = min(rollout_batch_size, len(batch), rollout_starts.size)
                    rollout_choice = np.random.randint(0, rollout_starts.size, size=rollout_sample_size)
                    rollout_start_batch = rollout_starts[rollout_choice]
                    rollout_starts_t = torch.as_tensor(rollout_start_batch, dtype=torch.long, device=device)
                    rollout_pred = train_states[rollout_starts_t]
                    for t in range(horizon):
                        rollout_pred = model(rollout_pred, train_actions[rollout_starts_t + t])
                    rollout_target = train_states[rollout_starts_t + horizon]
                    short_losses.append(short_horizon_channel_loss(rollout_pred, rollout_target))
                if short_losses:
                    loss = loss + short_rollout_weight * torch.stack(short_losses).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_states, val_actions)
            val_loss = structured_state_loss(val_pred, val_next_states)
        val_loss_f = float(val_loss.item())
        if val_loss_f < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": val_loss_f,
                "best_val_loss": best_val,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    with torch.no_grad():
        val_pred = model(val_states, val_actions)
    # TD-MPC2 currently plans and trains with horizon=3 on Hopper.  Therefore
    # 1/3/5-step rollout error is the primary acceptance band; 10/20-step
    # rollout remains useful as a drift diagnostic, but not as the main gate.
    metrics = {
        "one_step_struct_rmse": rmse_tensor(val_pred, val_next_states),
        "rollout_1_rmse": rollout_rmse(model, val_states, val_actions, dataset.dones[val_idx], 1, 128),
        "rollout_3_rmse": rollout_rmse(model, val_states, val_actions, dataset.dones[val_idx], 3, 128),
        "rollout_5_rmse": rollout_rmse(model, val_states, val_actions, dataset.dones[val_idx], 5, 128),
        "rollout_10_rmse": rollout_rmse(model, val_states, val_actions, dataset.dones[val_idx], 10, 128),
        "rollout_20_rmse": rollout_rmse(model, val_states, val_actions, dataset.dones[val_idx], 20, 128),
    }
    metrics.update(key_metrics(model, val_pred, val_next_states))
    return {
        "history": history,
        "metrics": metrics,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fit for Hopper SD dynamics.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--trace_dir", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--rollout_horizon", type=int, default=5)
    parser.add_argument("--rollout_weight", type=float, default=0.25)
    parser.add_argument("--rollout_batch_size", type=int, default=64)
    parser.add_argument("--short_rollout_horizons", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--short_rollout_weight", type=float, default=0.0)
    parser.add_argument("--touch_loss_weight", type=float, default=0.0)
    parser.add_argument("--use_transition_residual", action="store_true")
    parser.add_argument("--transition_residual_scale", type=float, default=1.0)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default="results/hopper_sd_supervised/default_run")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.trace_dir:
        dataset = collect_trace_dataset(args.trace_dir)
    else:
        dataset = collect_random_dataset(args.seed, args.episodes, args.max_steps)
    train_idx, val_idx = split_indices(dataset.episode_ids, args.val_fraction, args.seed)

    model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
        use_transition_residual=args.use_transition_residual,
        transition_residual_scale=args.transition_residual_scale,
    )
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        src_state = payload.get("model_state", payload)
        target_state = model.state_dict()
        compatible = {
            k: v for k, v in src_state.items()
            if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
        }
        model.load_state_dict(compatible, strict=False)
    result = fit_model(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rollout_horizon=args.rollout_horizon,
        rollout_weight=args.rollout_weight,
        rollout_batch_size=args.rollout_batch_size,
        short_rollout_horizons=args.short_rollout_horizons,
        short_rollout_weight=args.short_rollout_weight,
        touch_loss_weight=args.touch_loss_weight,
        device=args.device,
    )

    torch.save(
        {
            "model_state": result["model_state"],
            "args": vars(args),
            "metrics": result["metrics"],
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
        },
        out_dir / "hopper_sd_supervised.pt",
    )
    write_json(
        out_dir / "summary.json",
        {
            "args": vars(args),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "metrics": result["metrics"],
            "learned_params": model.get_learned_params(),
        },
    )
    write_json(out_dir / "history.json", {"history": result["history"]})
    print(json.dumps({"metrics": result["metrics"], "out_dir": str(out_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
