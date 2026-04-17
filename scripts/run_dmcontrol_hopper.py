#!/usr/bin/env python3
"""Prototype trainer/evaluator for DMControl Hopper-Hop structured model."""

import argparse
import atexit
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from dm_control import suite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel

PHASE_COMPRESSION = 0
PHASE_TAKEOFF = 1
PHASE_FLIGHT = 2
PHASE_LANDING = 3
HOPPER_PHASE_NAMES = ("compression", "takeoff", "flight", "landing")
LOG_FILE = None


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


def setup_logging(log_path: str) -> None:
    global LOG_FILE
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE = path.open("a", encoding="utf-8")


def close_logging() -> None:
    global LOG_FILE
    if LOG_FILE is not None:
        LOG_FILE.close()
        LOG_FILE = None


def default_log_path(checkpoint_path: str, resume_checkpoint: str) -> str:
    base = checkpoint_path or resume_checkpoint
    if not base:
        return ""
    base_path = Path(base)
    if base_path.suffix:
        return str(base_path.with_suffix(".log"))
    return str(base_path) + ".log"


atexit.register(close_logging)


def infer_hopper_phase_np(state: np.ndarray) -> int:
    """Infer a coarse hop phase from structured state."""
    touch = float(np.mean(state[13:15]))
    z_dot = float(state[7])
    if touch < 0.08:
        return PHASE_FLIGHT
    if z_dot > 0.20:
        return PHASE_TAKEOFF
    if touch >= 0.08 and z_dot < -0.05:
        return PHASE_LANDING
    return PHASE_COMPRESSION


def infer_hopper_phase_torch(state: torch.Tensor) -> torch.Tensor:
    if state.dim() == 1:
        state = state.unsqueeze(0)
    touch = state[:, 13:15].mean(dim=-1)
    z_dot = state[:, 7]
    phase = torch.full((state.shape[0],), PHASE_COMPRESSION, dtype=torch.long, device=state.device)
    flight = touch < 0.08
    takeoff = (~flight) & (z_dot > 0.20)
    landing = (~flight) & (~takeoff) & (z_dot < -0.05)
    phase = torch.where(takeoff, torch.full_like(phase, PHASE_TAKEOFF), phase)
    phase = torch.where(flight, torch.full_like(phase, PHASE_FLIGHT), phase)
    phase = torch.where(landing, torch.full_like(phase, PHASE_LANDING), phase)
    return phase


def hopper_phase_onehot_torch(state: torch.Tensor) -> torch.Tensor:
    phase = infer_hopper_phase_torch(state)
    return F.one_hot(phase, num_classes=4).to(dtype=state.dtype)


def takeoff_release_score_torch(action: torch.Tensor) -> torch.Tensor:
    if action.dim() == 1:
        action = action.unsqueeze(0)
    hip_release = torch.clamp(-action[:, 1], min=0.0, max=1.0)
    knee_release = torch.clamp(action[:, 2], min=0.0, max=1.0)
    ankle_release = torch.clamp(action[:, 3], min=0.0, max=1.0)
    return (hip_release + knee_release + ankle_release) / 3.0


def compression_drive_score_torch(action: torch.Tensor) -> torch.Tensor:
    if action.dim() == 1:
        action = action.unsqueeze(0)
    hip_compress = torch.clamp(action[:, 1], min=0.0, max=1.0)
    knee_compress = torch.clamp(-action[:, 2], min=0.0, max=1.0)
    ankle_compress = torch.clamp(-action[:, 3], min=0.0, max=1.0)
    return (hip_compress + knee_compress + ankle_compress) / 3.0


def init_hopper_event_state_torch(batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    zeros = torch.zeros(batch_size, dtype=dtype, device=device)
    return {
        "compression_credit": zeros.clone(),
        "compression_dwell": zeros.clone(),
        "takeoff_armed": zeros.clone(),
        "valid_flight": zeros.clone(),
        "takeoff_window": zeros.clone(),
    }


def _hopper_event_terms(
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    hop_speed: float,
    event_state: Dict[str, torch.Tensor] | None = None,
):
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    if next_state.dim() == 1:
        next_state = next_state.unsqueeze(0)
    if event_state is None:
        event_state = init_hopper_event_state_torch(state.shape[0], state.device, state.dtype)

    prev_phase = infer_hopper_phase_torch(state)
    next_phase = infer_hopper_phase_torch(next_state)

    touch = next_state[:, 13:15].mean(dim=-1)
    z_dot = next_state[:, 7]
    x_dot = next_state[:, 6]
    rooty = next_state[:, 1]

    upright = (1.0 + torch.cos(rooty)) / 2.0
    contact = torch.clamp(touch, min=0.0, max=1.0)
    light_contact = 1.0 - contact
    height_progress = torch.clamp((next_state[:, 0] + 0.10) / 0.20, min=0.0, max=1.0)
    lift = torch.clamp(z_dot / 2.0, min=0.0, max=1.0)
    speed_progress = torch.clamp(x_dot / max(hop_speed, 1e-6), min=0.0, max=1.0)
    soft_landing = torch.clamp(1.0 - torch.abs(z_dot) / 2.0, min=0.0, max=1.0)
    release_score = takeoff_release_score_torch(action)
    compression_drive = compression_drive_score_torch(action)

    prev_credit = event_state["compression_credit"]
    prev_dwell = event_state["compression_dwell"]
    prev_armed = event_state["takeoff_armed"]
    prev_valid_flight = event_state["valid_flight"]
    prev_takeoff_window = event_state.get("takeoff_window", torch.zeros_like(prev_valid_flight))

    in_compression = (next_phase == PHASE_COMPRESSION).to(state.dtype)
    in_takeoff = (next_phase == PHASE_TAKEOFF).to(state.dtype)
    in_flight = (next_phase == PHASE_FLIGHT).to(state.dtype)
    in_landing = (next_phase == PHASE_LANDING).to(state.dtype)

    credit_gain = in_compression * contact * upright * (0.55 * compression_drive + 0.25 * (1.0 - lift) + 0.20 * (1.0 - speed_progress))
    next_credit = torch.clamp(0.88 * prev_credit + credit_gain - 0.30 * release_score * in_takeoff, min=0.0, max=1.5)
    next_dwell = torch.where(in_compression > 0.5, prev_dwell + 1.0, 0.50 * prev_dwell)

    armed_condition = ((next_credit > 0.26) & (contact > 0.10) & (upright > 0.45)).to(state.dtype)
    next_armed = torch.where(
        in_compression > 0.5,
        torch.maximum(0.85 * prev_armed, armed_condition),
        0.35 * prev_armed,
    )

    valid_takeoff = (
        ((prev_phase == PHASE_COMPRESSION) | (prev_phase == PHASE_TAKEOFF)).to(state.dtype)
        * in_takeoff
        * (prev_armed > 0.08).to(state.dtype)
        * (prev_credit > 0.22).to(state.dtype)
        * (release_score > 0.22).to(state.dtype)
        * (lift > 0.08).to(state.dtype)
        * (contact > 0.06).to(state.dtype)
    )
    takeoff_window = torch.where(
        valid_takeoff > 0.5,
        torch.ones_like(prev_takeoff_window),
        0.65 * prev_takeoff_window,
    )
    valid_t2f = (
        ((prev_phase == PHASE_TAKEOFF) | (prev_takeoff_window > 0.18)).to(state.dtype)
        * in_flight
        * (prev_takeoff_window > 0.18).to(state.dtype)
        * (light_contact > 0.35).to(state.dtype)
        * ((speed_progress > 0.15) | (lift > 0.05)).to(state.dtype)
    )
    next_valid_flight = torch.where(
        in_flight > 0.5,
        torch.maximum(0.85 * prev_valid_flight, valid_takeoff + valid_t2f),
        torch.where(in_landing > 0.5, 0.30 * prev_valid_flight, 0.70 * prev_valid_flight),
    )
    valid_landing = (
        (prev_valid_flight > 0.2).to(state.dtype)
        * in_landing
        * (contact > 0.18).to(state.dtype)
        * (soft_landing > 0.35).to(state.dtype)
    )
    reset_cycle = (valid_landing > 0.5) & (in_compression > 0.5)

    next_event_state = {
        "compression_credit": torch.where(reset_cycle, 0.25 * next_credit, next_credit),
        "compression_dwell": torch.where(reset_cycle, torch.zeros_like(next_dwell), next_dwell),
        "takeoff_armed": torch.where(valid_takeoff > 0.5, 0.15 * next_armed, next_armed),
        "valid_flight": torch.where(valid_landing > 0.5, 0.20 * next_valid_flight, next_valid_flight),
        "takeoff_window": torch.where(valid_landing > 0.5, 0.10 * takeoff_window, takeoff_window),
    }
    info = {
        "prev_phase": prev_phase,
        "next_phase": next_phase,
        "upright": upright,
        "contact": contact,
        "light_contact": light_contact,
        "height_progress": height_progress,
        "lift": lift,
        "speed_progress": speed_progress,
        "soft_landing": soft_landing,
        "release_score": release_score,
        "compression_drive": compression_drive,
        "prev_credit": prev_credit,
        "next_credit": next_credit,
        "prev_dwell": prev_dwell,
        "next_dwell": next_dwell,
        "prev_armed": prev_armed,
        "next_armed": next_armed,
        "prev_valid_flight": prev_valid_flight,
        "next_valid_flight": next_valid_flight,
        "prev_takeoff_window": prev_takeoff_window,
        "next_takeoff_window": takeoff_window,
        "valid_takeoff": valid_takeoff,
        "valid_t2f": valid_t2f,
        "valid_landing": valid_landing,
        "reset_cycle": reset_cycle.to(state.dtype),
    }
    return next_event_state, info


def format_phase_counts(counts: np.ndarray) -> str:
    total = max(1, int(np.sum(counts)))
    parts = []
    for idx, name in enumerate(HOPPER_PHASE_NAMES):
        frac = float(counts[idx]) / total
        parts.append(f"{name[0]}:{frac:.2f}")
    return " ".join(parts)


def phase_counts_np(states: np.ndarray) -> np.ndarray:
    counts = np.zeros(4, dtype=np.int64)
    for s in states:
        counts[infer_hopper_phase_np(s)] += 1
    return counts


def phase_transition_counts_np(states: np.ndarray, next_states: np.ndarray) -> Dict[str, int]:
    counts = {"c2t": 0, "t2f": 0, "f2l": 0, "l2c": 0}
    for s, ns in zip(states, next_states):
        p = infer_hopper_phase_np(s)
        q = infer_hopper_phase_np(ns)
        if p == PHASE_COMPRESSION and q == PHASE_TAKEOFF:
            counts["c2t"] += 1
        elif p == PHASE_TAKEOFF and q == PHASE_FLIGHT:
            counts["t2f"] += 1
        elif p == PHASE_FLIGHT and q == PHASE_LANDING:
            counts["f2l"] += 1
        elif p == PHASE_LANDING and q == PHASE_COMPRESSION:
            counts["l2c"] += 1
    return counts


def summarize_hopper_rollout(
    model,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    device: str = "cpu",
    actor_net=None,
) -> Dict[str, object]:
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    if len(states) == 0:
        return {
            "true_return": 0.0,
            "true_nonzero_rate": 0.0,
            "phase_counts": np.zeros(4, dtype=np.int64),
            "transitions": {"c2t": 0, "t2f": 0, "f2l": 0, "l2c": 0},
            "max_height": 0.0,
            "max_xdot": 0.0,
            "max_zdot": 0.0,
            "mean_touch": 0.0,
            "mean_plan_reward": 0.0,
            "mean_phase_reward": 0.0,
            "mean_release_score": 0.0,
            "mean_compression_drive": 0.0,
            "mean_credit": 0.0,
            "armed_rate": 0.0,
            "valid_flight_rate": 0.0,
            "compression_dwell": 0.0,
            "valid_takeoff_count": 0,
            "invalid_takeoff_count": 0,
            "valid_t2f_count": 0,
            "valid_landing_count": 0,
            "guide_gap": 0.0,
            "actor_gap": None,
        }

    phase_counts = phase_counts_np(states)
    transitions = phase_transition_counts_np(states, next_states)
    guide_actions = np.stack([hopper_guidance_action(s) for s in states], axis=0)
    guide_gap = float(np.mean(np.linalg.norm(actions - guide_actions, axis=-1)))

    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    action_t = torch.tensor(actions, dtype=torch.float32, device=device)
    next_state_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        plan_reward = model.predict_plan_reward(state_t, action_t, next_state_t)
        event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
        phase_rewards = []
        credits = []
        armed_flags = []
        valid_flights = []
        dwells = []
        valid_takeoff_count = 0
        invalid_takeoff_count = 0
        valid_t2f_count = 0
        valid_landing_count = 0
        for i in range(len(state_t)):
            phase_reward_i, event_state, info = predict_phase_progress_reward(
                state_t[i:i + 1], action_t[i:i + 1], next_state_t[i:i + 1], model.hop_speed,
                event_state=event_state, return_info=True,
            )
            phase_rewards.append(phase_reward_i.squeeze(0))
            credits.append(info["next_credit"].squeeze(0))
            armed_flags.append((info["next_armed"] > 0.25).to(state_t.dtype).squeeze(0))
            valid_flights.append((info["next_valid_flight"] > 0.20).to(state_t.dtype).squeeze(0))
            dwells.append(info["next_dwell"].squeeze(0))
            valid_takeoff_count += int((info["valid_takeoff"] > 0.5).sum().item())
            invalid_takeoff_count += int((((info["next_phase"] == PHASE_TAKEOFF).to(state_t.dtype) > 0.5) & (info["valid_takeoff"] < 0.5)).sum().item())
            valid_t2f_count += int((info["valid_t2f"] > 0.5).sum().item())
            valid_landing_count += int((info["valid_landing"] > 0.5).sum().item())
        phase_reward = torch.stack(phase_rewards, dim=0)
        pred_height = model.predict_height(next_state_t)
        release_score = takeoff_release_score_torch(action_t)
        compression_drive = compression_drive_score_torch(action_t)
    actor_gap = None
    if actor_net is not None:
        with torch.no_grad():
            actor_pred = actor_net(state_t, hopper_phase_onehot_torch(state_t)).detach().cpu().numpy()
        actor_gap = float(np.mean(np.linalg.norm(actions - actor_pred, axis=-1)))

    return {
        "true_return": float(rewards.sum()),
        "true_nonzero_rate": float(np.mean(rewards > 0)),
        "phase_counts": phase_counts,
        "transitions": transitions,
        "max_height": float(pred_height.max().item()),
        "max_xdot": float(np.max(next_states[:, 6])),
        "max_zdot": float(np.max(next_states[:, 7])),
        "mean_touch": float(np.mean(next_states[:, 13:15])),
        "mean_plan_reward": float(plan_reward.mean().item()),
        "mean_phase_reward": float(phase_reward.mean().item()),
        "mean_release_score": float(release_score.mean().item()),
        "mean_compression_drive": float(compression_drive.mean().item()),
        "mean_credit": float(torch.stack(credits).mean().item()),
        "armed_rate": float(torch.stack(armed_flags).mean().item()),
        "valid_flight_rate": float(torch.stack(valid_flights).mean().item()),
        "compression_dwell": float(torch.stack(dwells).mean().item()),
        "valid_takeoff_count": int(valid_takeoff_count),
        "invalid_takeoff_count": int(invalid_takeoff_count),
        "valid_t2f_count": int(valid_t2f_count),
        "valid_landing_count": int(valid_landing_count),
        "guide_gap": guide_gap,
        "actor_gap": actor_gap,
    }


def format_hopper_summary(tag: str, summary: Dict[str, object]) -> str:
    tr = summary["transitions"]
    actor_gap = "n/a" if summary["actor_gap"] is None else f"{summary['actor_gap']:.3f}"
    return (
        f"{tag} | true_ret={summary['true_return']:.2f} | true_nz={summary['true_nonzero_rate']:.3f} | "
        f"plan={summary['mean_plan_reward']:.3f} | phase={summary['mean_phase_reward']:.3f} | rel={summary['mean_release_score']:.3f} | "
        f"cmp={summary['mean_compression_drive']:.3f} | credit={summary['mean_credit']:.3f} | armed={summary['armed_rate']:.3f} | "
        f"vflight={summary['valid_flight_rate']:.3f} | dwell={summary['compression_dwell']:.1f} | "
        f"max_h={summary['max_height']:.3f} | max_x={summary['max_xdot']:.3f} | max_z={summary['max_zdot']:.3f} | "
        f"touch={summary['mean_touch']:.3f} | phases={format_phase_counts(summary['phase_counts'])} | "
        f"trans=c2t:{tr['c2t']} t2f:{tr['t2f']} f2l:{tr['f2l']} l2c:{tr['l2c']} | "
        f"evt=vt:{summary['valid_takeoff_count']} it:{summary['invalid_takeoff_count']} vf:{summary['valid_t2f_count']} vl:{summary['valid_landing_count']} | "
        f"guide_gap={summary['guide_gap']:.3f} | actor_gap={actor_gap}"
    )


def exploratory_action(state, rng: np.random.RandomState, step: int) -> np.ndarray:
    """Mixture exploration for sparse hopping transitions.

    Modes:
    - random torque
    - extend leg to push off
    - compress leg to load spring-like posture
    - alternating pulse to induce vertical oscillation
    """
    rootz, rooty, waist, hip, knee, ankle = state[:6]
    mode = rng.randint(4)
    if mode == 0:
        return rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
    if mode == 1:
        # Extension: open knee/ankle while keeping torso roughly upright.
        return np.array([0.2, -0.2, 1.0, 0.8], dtype=np.float32)
    if mode == 2:
        # Compression: bend hip/knee and dorsiflex ankle.
        return np.array([-0.2, 0.8, -1.0, -0.6], dtype=np.float32)
    pulse = 1.0 if (step // 8) % 2 == 0 else -1.0
    # Small torso stabilization + alternating leg pulse.
    return np.array([0.1 * np.sign(rooty), 0.5 * pulse, -0.8 * pulse, -0.4 * pulse], dtype=np.float32)


def hopper_guidance_action(state: np.ndarray, event_state: Dict[str, torch.Tensor] | None = None) -> np.ndarray:
    """Phase-conditioned behavior guide for Hopper-Hop."""
    rooty, waist = state[1], state[2]
    x_dot, z_dot = state[6], state[7]
    phase = infer_hopper_phase_np(state)
    armed = 0.0
    valid_flight = 0.0
    credit = 0.0
    if event_state is not None:
        armed = float(event_state["takeoff_armed"].mean().item())
        valid_flight = float(event_state["valid_flight"].mean().item())
        credit = float(event_state["compression_credit"].mean().item())

    torso_corr = np.clip(-0.6 * rooty - 0.15 * waist, -0.4, 0.4)
    speed_corr = np.clip(0.2 * (1.5 - x_dot), -0.2, 0.3)

    if phase == PHASE_COMPRESSION:
        release_bias = np.clip(0.40 * max(0.0, credit - 0.18), 0.0, 0.25)
        base = np.array([torso_corr - 0.1, 0.95 - release_bias, -1.0 + release_bias, -0.75 + 0.5 * release_bias], dtype=np.float32)
    elif phase == PHASE_TAKEOFF:
        if armed < 0.10:
            base = np.array([torso_corr - 0.05, 0.75, -0.90, -0.65], dtype=np.float32)
        else:
            base = np.array([torso_corr + 0.2, -0.35 + speed_corr, 1.0, 0.95], dtype=np.float32)
    elif phase == PHASE_FLIGHT:
        if valid_flight < 0.10:
            base = np.array([torso_corr - 0.05, 0.45, -0.55, -0.25], dtype=np.float32)
        else:
            base = np.array([torso_corr, -0.20 + speed_corr, 0.45, -0.10], dtype=np.float32)
    else:
        # Landing / reset: absorb impact while keeping torso recoverable.
        landing_brake = np.clip(-0.25 * z_dot, -0.2, 0.4)
        base = np.array([torso_corr - 0.05, 0.30 + landing_brake, -0.45, -0.20], dtype=np.float32)

    return np.clip(base, -1.0, 1.0).astype(np.float32)


def hopper_flight_template_action(step_since_takeoff: int) -> np.ndarray:
    """Short open-loop air template immediately after a valid takeoff."""
    if step_since_takeoff <= 0:
        return np.array([0.05, -0.30, 0.65, -0.25], dtype=np.float32)
    if step_since_takeoff == 1:
        return np.array([0.02, -0.20, 0.45, -0.10], dtype=np.float32)
    return np.array([0.00, -0.10, 0.20, -0.05], dtype=np.float32)


def phase_conditioned_guide_mix(state: np.ndarray, base_mix: float, event_state: Dict[str, torch.Tensor] | None = None) -> float:
    """Use stronger guide blending in the critical pre-takeoff phases."""
    phase = infer_hopper_phase_np(state)
    armed = 0.0
    credit = 0.0
    if event_state is not None:
        armed = float(event_state["takeoff_armed"].mean().item())
        credit = float(event_state["compression_credit"].mean().item())
    if phase in (PHASE_COMPRESSION, PHASE_TAKEOFF) and (armed > 0.20 or credit > 0.65):
        return min(base_mix, 0.18)
    if phase == PHASE_COMPRESSION:
        return max(base_mix, 0.55)
    if phase == PHASE_TAKEOFF:
        return max(base_mix, 0.45)
    if phase == PHASE_FLIGHT:
        return min(base_mix, 0.18)
    return min(base_mix, 0.30)


def guided_exploration_action(state, rng: np.random.RandomState, step: int) -> np.ndarray:
    """Exploration centered around a plausible hopping pattern."""
    mode = rng.randint(4)
    guide = hopper_guidance_action(state)
    if mode == 0:
        return np.clip(guide + rng.normal(0.0, 0.18, size=guide.shape), -1.0, 1.0).astype(np.float32)
    if mode == 1:
        return np.clip(0.75 * guide + 0.25 * exploratory_action(state, rng, step), -1.0, 1.0).astype(np.float32)
    if mode == 2:
        return exploratory_action(state, rng, step)
    pulse = 1.0 if (step // 6) % 2 == 0 else -1.0
    guided_pulse = guide.copy()
    guided_pulse[1] = np.clip(guided_pulse[1] + 0.35 * pulse, -1.0, 1.0)
    guided_pulse[2] = np.clip(guided_pulse[2] - 0.50 * pulse, -1.0, 1.0)
    return guided_pulse.astype(np.float32)


def collect_trajectories(env, n_steps: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    data = {
        "states": [], "actions": [], "rewards": [], "next_states": [], "dones": [],
        "heights": [], "next_heights": [],
    }
    ts = env.reset()
    obs = ts.observation
    for step in range(n_steps):
        state = DMControlHopperSDModel.obs_to_state(obs)
        height = float(env.physics.height())
        action = guided_exploration_action(state, rng, step)
        ts = env.step(action)
        next_obs = ts.observation
        next_state = DMControlHopperSDModel.obs_to_state(next_obs)
        next_height = float(env.physics.height())
        data["states"].append(state)
        data["actions"].append(action)
        data["rewards"].append(float(ts.reward))
        data["next_states"].append(next_state)
        data["dones"].append(bool(ts.last()))
        data["heights"].append(height)
        data["next_heights"].append(next_height)
        obs = next_obs
        if ts.last():
            ts = env.reset()
            obs = ts.observation
    return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}


class TransitionReplay:
    """Simple replay buffer for online Hopper training.

    The purpose is to mimic the most important TD-MPC2 behavior:
    experience is not collected once and frozen, but is accumulated online
    and repeatedly reused for model updates.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._data: Dict[str, List[np.ndarray]] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "heights": [],
            "next_heights": [],
        }

    def __len__(self) -> int:
        return len(self._data["states"])

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        height: float,
        next_height: float,
    ) -> None:
        for key, value in {
            "states": state.astype(np.float32),
            "actions": action.astype(np.float32),
            "rewards": np.float32(reward),
            "next_states": next_state.astype(np.float32),
            "dones": np.float32(done),
            "heights": np.float32(height),
            "next_heights": np.float32(next_height),
        }.items():
            self._data[key].append(value)
            if len(self._data[key]) > self.capacity:
                self._data[key].pop(0)

    def sample_batch(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self), size=batch_size)
        batch = {}
        for key, values in self._data.items():
            arr = np.asarray([values[i] for i in idx], dtype=np.float32)
            batch[key] = torch.tensor(arr, dtype=torch.float32, device=device)
        return batch

    def as_dataset(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in self._data.items()}

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "data": {k: list(v) for k, v in self._data.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.capacity = int(state["capacity"])
        self._data = {k: list(v) for k, v in state["data"].items()}


def _state_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos_loss = F.smooth_l1_loss(pred[:, :1], target[:, :1])
    ang_err = torch.atan2(torch.sin(pred[:, 1:6] - target[:, 1:6]), torch.cos(pred[:, 1:6] - target[:, 1:6]))
    ang_loss = F.smooth_l1_loss(ang_err, torch.zeros_like(ang_err))
    vel_loss = F.smooth_l1_loss(pred[:, 6:13], target[:, 6:13])
    touch_loss = F.smooth_l1_loss(pred[:, 13:], target[:, 13:])
    return pos_loss + ang_loss + vel_loss + 0.2 * touch_loss


def _state_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pos_err = pred[:, :1] - target[:, :1]
    ang_err = torch.atan2(torch.sin(pred[:, 1:6] - target[:, 1:6]), torch.cos(pred[:, 1:6] - target[:, 1:6]))
    vel_err = pred[:, 6:13] - target[:, 6:13]
    touch_err = pred[:, 13:] - target[:, 13:]
    err = torch.cat([pos_err, ang_err, vel_err, touch_err], dim=-1)
    return float(torch.sqrt((err ** 2).mean()).item())


class HopperValueNet(torch.nn.Module):
    """Minimal TVA value network on structured Hopper state."""

    def __init__(self, state_dim: int = 15, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HopperActorNet(torch.nn.Module):
    """Small actor prior on structured Hopper state.

    The actor is not used to replace planning. Instead, it provides
    candidate trajectories to the planner, closer to TD-MPC2's policy-prior
    role, while the SD world model remains the dynamics backbone.
    """

    def __init__(self, state_dim: int = 15, phase_dim: int = 4, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + phase_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, phase_onehot: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if phase_onehot.dim() == 1:
            phase_onehot = phase_onehot.unsqueeze(0)
        return self.net(torch.cat([state, phase_onehot], dim=-1))


def compute_discounted_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if dones[i]:
            running = 0.0
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns


def build_tva_targets(model, data, gamma: float = 0.99, true_coef: float = 1.0, shaped_coef: float = 1.0, device: str = "cpu"):
    states = torch.tensor(data["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=device)
    done_arr = np.asarray(data["dones"], dtype=bool)
    with torch.no_grad():
        event_state = init_hopper_event_state_torch(1, states.device, states.dtype)
        shaped = []
        target_adjust = []
        for i in range(len(states)):
            reward_i, event_state, info = predict_structured_plan_reward(
                model, states[i:i + 1], actions[i:i + 1], next_states[i:i + 1],
                event_state=event_state, return_info=True,
            )
            shaped.append(float(reward_i.item()))
            frontier_bonus = (
                0.45 * float((info["valid_takeoff"] > 0.5).item())
                + 0.20 * float(info["valid_t2f"].item())
                + 1.20 * float((info["valid_landing"] > 0.5).item())
                + 0.50 * float(info["reset_cycle"].item())
            )
            pseudo_penalty = (
                0.35 * float(((info["next_phase"] == PHASE_FLIGHT) & (info["next_valid_flight"] < 0.20)).item())
                + 0.25 * float(((info["next_phase"] == PHASE_FLIGHT) & (info["prev_takeoff_window"] <= 0.18)).item())
            )
            target_adjust.append(frontier_bonus - pseudo_penalty)
            if done_arr[i]:
                event_state = init_hopper_event_state_torch(1, states.device, states.dtype)
        shaped = np.asarray(shaped, dtype=np.float32)
        target_adjust = np.asarray(target_adjust, dtype=np.float32)
    true_returns = compute_discounted_returns(np.asarray(data["rewards"], dtype=np.float32), done_arr, gamma=gamma)
    shaped_returns = compute_discounted_returns(shaped, done_arr, gamma=gamma)
    adjust_returns = compute_discounted_returns(target_adjust, done_arr, gamma=gamma)
    targets = true_coef * true_returns + shaped_coef * shaped_returns + 0.65 * adjust_returns
    return states.detach().cpu(), torch.tensor(targets, dtype=torch.float32)


def train_value_net(value_net, states, targets, epochs=300, lr=1e-3, batch_size=256, min_loss=0.0, device="cpu"):
    value_net.to(device)
    states = states.to(device)
    targets = targets.to(device)
    optim = torch.optim.Adam(value_net.parameters(), lr=lr)
    n = len(states)
    if n == 0:
        return 0.0
    last_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        loss_sum = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = value_net(states[idx])
            loss = F.smooth_l1_loss(pred, targets[idx])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            optim.step()
            loss_sum += float(loss.item())
            n_batches += 1
        last_loss = loss_sum / max(1, n_batches)
        if min_loss > 0 and last_loss < min_loss:
            break
    value_net.eval()
    return last_loss


def train_actor_prior(actor_net, states, target_actions, optim, updates=25, batch_size=256, device="cpu", sample_mask=None):
    """Behavior-clone executed actions to build a learned action prior."""
    if actor_net is None or len(states) == 0:
        return None
    state_arr = np.asarray(states, dtype=np.float32)
    action_arr = np.asarray(target_actions, dtype=np.float32)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.any():
            state_arr = state_arr[mask]
            action_arr = action_arr[mask]
    if len(state_arr) == 0:
        return None
    actor_net.train()
    states_t = torch.tensor(state_arr, dtype=torch.float32, device=device)
    targets_t = torch.tensor(action_arr, dtype=torch.float32, device=device)
    phase_t = hopper_phase_onehot_torch(states_t)
    last_loss = None
    for _ in range(updates):
        if len(states_t) <= batch_size:
            batch_idx = torch.arange(len(states_t), device=device)
        else:
            batch_idx = torch.randperm(len(states_t), device=device)[:batch_size]
        pred = actor_net(states_t[batch_idx], phase_t[batch_idx])
        loss = F.smooth_l1_loss(pred, targets_t[batch_idx])
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0)
        optim.step()
        last_loss = float(loss.item())
    actor_net.eval()
    return last_loss


def _episode_value_targets(model, states, actions, rewards, next_states, dones, gamma, true_coef, shaped_coef, device):
    state_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
    action_t = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
    next_state_t = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    done_arr = np.asarray(dones, dtype=bool)
    with torch.no_grad():
        event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
        shaped = []
        target_adjust = []
        for i in range(len(state_t)):
            reward_i, event_state, info = predict_structured_plan_reward(
                model, state_t[i:i + 1], action_t[i:i + 1], next_state_t[i:i + 1],
                event_state=event_state, return_info=True,
            )
            shaped.append(float(reward_i.item()))
            frontier_bonus = (
                0.45 * float((info["valid_takeoff"] > 0.5).item())
                + 0.20 * float(info["valid_t2f"].item())
                + 1.20 * float((info["valid_landing"] > 0.5).item())
                + 0.50 * float(info["reset_cycle"].item())
            )
            pseudo_penalty = (
                0.35 * float(((info["next_phase"] == PHASE_FLIGHT) & (info["next_valid_flight"] < 0.20)).item())
                + 0.25 * float(((info["next_phase"] == PHASE_FLIGHT) & (info["prev_takeoff_window"] <= 0.18)).item())
            )
            target_adjust.append(frontier_bonus - pseudo_penalty)
            if done_arr[i]:
                event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
        shaped = np.asarray(shaped, dtype=np.float32)
        target_adjust = np.asarray(target_adjust, dtype=np.float32)
    true_returns = compute_discounted_returns(reward_arr, done_arr, gamma=gamma)
    shaped_returns = compute_discounted_returns(shaped, done_arr, gamma=gamma)
    adjust_returns = compute_discounted_returns(target_adjust, done_arr, gamma=gamma)
    targets = true_coef * true_returns + shaped_coef * shaped_returns + 0.65 * adjust_returns
    return state_t.detach(), torch.tensor(targets, dtype=torch.float32, device=device)


def _episode_actor_mask(model, states, actions, next_states, dones, device):
    state_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
    action_t = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
    next_state_t = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)
    done_arr = np.asarray(dones, dtype=bool)
    if len(state_t) == 0:
        return np.zeros(0, dtype=bool)
    with torch.no_grad():
        event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
        event_scores = []
        valid_takeoffs = []
        valid_landings = []
        for i in range(len(state_t)):
            reward_i, event_state, info = predict_phase_progress_reward(
                state_t[i:i + 1], action_t[i:i + 1], next_state_t[i:i + 1], model.hop_speed,
                event_state=event_state, return_info=True,
            )
            event_scores.append(float(reward_i.item()))
            valid_takeoffs.append(bool((info["valid_takeoff"] > 0.5).item()))
            valid_landings.append(bool((info["valid_landing"] > 0.5).item()))
            if done_arr[i]:
                event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
    event_scores = np.asarray(event_scores, dtype=np.float32)
    valid_takeoffs = np.asarray(valid_takeoffs, dtype=bool)
    valid_landings = np.asarray(valid_landings, dtype=bool)
    if len(event_scores) == 0:
        return np.zeros(0, dtype=bool)
    score_cut = float(np.quantile(event_scores, 0.75))
    return valid_takeoffs | valid_landings | (event_scores >= score_cut)


def _episode_event_stats(model, states, actions, next_states, dones, device):
    state_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
    action_t = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
    next_state_t = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)
    done_arr = np.asarray(dones, dtype=bool)
    if len(state_t) == 0:
        return {
            "event_scores": np.zeros(0, dtype=np.float32),
            "valid_takeoffs": np.zeros(0, dtype=bool),
            "valid_flights": np.zeros(0, dtype=bool),
            "valid_landings": np.zeros(0, dtype=bool),
            "pseudo_flight_mask": np.zeros(0, dtype=bool),
        }
    with torch.no_grad():
        event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
        event_scores = []
        valid_takeoffs = []
        valid_flights = []
        valid_landings = []
        pseudo_flights = []
        for i in range(len(state_t)):
            reward_i, event_state, info = predict_phase_progress_reward(
                state_t[i:i + 1], action_t[i:i + 1], next_state_t[i:i + 1], model.hop_speed,
                event_state=event_state, return_info=True,
            )
            event_scores.append(float(reward_i.item()))
            valid_takeoffs.append(bool((info["valid_takeoff"] > 0.5).item()))
            valid_flights.append(bool((info["next_valid_flight"] > 0.20).item()))
            valid_landings.append(bool((info["valid_landing"] > 0.5).item()))
            pseudo_flights.append(bool(
                ((info["next_phase"] == PHASE_FLIGHT) & (info["next_valid_flight"] < 0.20)).item()
            ))
            if done_arr[i]:
                event_state = init_hopper_event_state_torch(1, state_t.device, state_t.dtype)
    return {
        "event_scores": np.asarray(event_scores, dtype=np.float32),
        "valid_takeoffs": np.asarray(valid_takeoffs, dtype=bool),
        "valid_flights": np.asarray(valid_flights, dtype=bool),
        "valid_landings": np.asarray(valid_landings, dtype=bool),
        "pseudo_flight_mask": np.asarray(pseudo_flights, dtype=bool),
    }


def _expand_mask(mask: np.ndarray, left: int = 0, right: int = 0) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return mask
    expanded = mask.copy()
    idx = np.flatnonzero(mask)
    for i in idx:
        start = max(0, i - left)
        end = min(len(mask), i + right + 1)
        expanded[start:end] = True
    return expanded


def _frontier_masks(ep_event_stats: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid_takeoffs = np.asarray(ep_event_stats["valid_takeoffs"], dtype=bool)
    valid_landings = np.asarray(ep_event_stats["valid_landings"], dtype=bool)
    frontier_mask = valid_takeoffs | valid_landings
    if len(frontier_mask) == 0:
        return frontier_mask, frontier_mask

    # 将有效落地及其前置离地片段视为“强前沿”，避免再次被伪 flight 片段冲掉。
    strong_frontier = _expand_mask(valid_landings, left=6, right=3)
    takeoff_support = valid_takeoffs & _expand_mask(valid_landings, left=7, right=0)
    strong_frontier = strong_frontier | takeoff_support
    return frontier_mask, strong_frontier


def predict_phase_progress_reward(
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    hop_speed: float,
    event_state: Dict[str, torch.Tensor] | None = None,
    return_info: bool = False,
):
    """Structured reward based on event-valid hop progression rather than phase labels alone."""
    next_event_state, info = _hopper_event_terms(state, action, next_state, hop_speed, event_state=event_state)
    dtype = next_state.dtype if next_state.dim() > 1 else state.dtype

    credit_gain = torch.clamp(info["next_credit"] - info["prev_credit"], min=0.0, max=1.0)
    armed_gain = torch.clamp(info["next_armed"] - info["prev_armed"], min=0.0, max=1.0)
    event_reward = (
        0.06 * credit_gain
        + 0.12 * armed_gain
        + info["valid_takeoff"] * (0.85 + 0.30 * info["release_score"] + 0.15 * info["lift"])
        + info["valid_t2f"] * (0.18 + 0.08 * info["light_contact"])
        + info["valid_landing"] * (0.75 + 0.25 * info["soft_landing"])
        + 0.12 * info["reset_cycle"]
    )

    invalid_takeoff = (info["next_phase"] == PHASE_TAKEOFF).to(dtype) * (1.0 - info["valid_takeoff"])
    invalid_flight = (info["next_phase"] == PHASE_FLIGHT).to(dtype) * (1.0 - (info["next_valid_flight"] > 0.20).to(dtype))
    orphan_flight = (info["next_valid_flight"] > 0.20).to(dtype) * (info["prev_takeoff_window"] <= 0.18).to(dtype)
    compression_stall = (info["next_phase"] == PHASE_COMPRESSION).to(dtype) * torch.clamp(info["next_dwell"] - 6.0, min=0.0, max=10.0) / 10.0
    weak_release = (
        ((info["next_phase"] == PHASE_TAKEOFF) | (info["next_phase"] == PHASE_FLIGHT)).to(dtype)
        * torch.clamp(0.28 - info["release_score"], min=0.0, max=0.28) / 0.28
    )
    premature_contact = (
        (info["prev_phase"] == PHASE_TAKEOFF).to(dtype)
        * (info["next_phase"] != PHASE_FLIGHT).to(dtype)
        * torch.clamp(info["contact"] - 0.35, min=0.0, max=1.0)
    )

    penalty = (
        0.35 * invalid_takeoff
        + 0.42 * invalid_flight
        + 0.55 * orphan_flight
        + 0.12 * compression_stall
        + 0.15 * weak_release
        + 0.30 * premature_contact
    )
    reward = event_reward - penalty
    if return_info:
        return reward, next_event_state, info
    return reward


def predict_structured_plan_reward(
    model,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    event_state: Dict[str, torch.Tensor] | None = None,
    return_info: bool = False,
):
    phase_reward, next_event_state, info = predict_phase_progress_reward(
        state, action, next_state, model.hop_speed, event_state=event_state, return_info=True
    )
    total_reward = model.predict_plan_reward(state, action, next_state) + phase_reward
    if return_info:
        return total_reward, next_event_state, info
    return total_reward


def save_training_checkpoint(path: str, payload: Dict[str, object]) -> None:
    if not path:
        return
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)


def load_training_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, object]:
    return torch.load(path, map_location=map_location)


def train_sd_model(model, data, epochs=100, batch_size=256, lr=1e-3, device="cpu"):
    n = len(data["states"])
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(0.1 * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    model.to(device)
    states = torch.tensor(data["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=device)
    next_heights = torch.tensor(data["next_heights"], dtype=torch.float32, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        perm = np.random.permutation(train_idx)
        train_sum = 0.0
        n_batches = 0
        model.train()
        for start in range(0, len(perm), batch_size):
            batch = perm[start:start + batch_size]
            s = states[batch]
            a = actions[batch]
            ns = next_states[batch]
            r = rewards[batch]
            pred_ns = model(s, a)
            pred_r = model.predict_reward(s, a, pred_ns)
            pred_h = model.predict_height(pred_ns)
            loss = (
                _state_prediction_loss(pred_ns, ns) +
                0.2 * F.smooth_l1_loss(pred_r, r) +
                0.5 * F.smooth_l1_loss(pred_h, next_heights[batch])
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            train_sum += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vs = states[val_idx]
            va = actions[val_idx]
            vns = next_states[val_idx]
            vr = rewards[val_idx]
            pred_vns = model(vs, va)
            pred_vr = model.predict_reward(vs, va, pred_vns)
            pred_vh = model.predict_height(pred_vns)
            val_loss = (
                _state_prediction_loss(pred_vns, vns) +
                0.2 * F.smooth_l1_loss(pred_vr, vr) +
                0.5 * F.smooth_l1_loss(pred_vh, next_heights[val_idx])
            )
        current_val = float(val_loss.item())
        if current_val < best_val:
            best_val = current_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            log(f"epoch {epoch + 1:4d} | train={train_sum / max(n_batches,1):.4f} | val={current_val:.4f} | best={best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return best_val


def _batch_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    states = batch["states"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_states = batch["next_states"]
    next_heights = batch["next_heights"]
    pred_ns = model(states, actions)
    pred_r = model.predict_reward(states, actions, pred_ns)
    pred_h = model.predict_height(pred_ns)
    return (
        _state_prediction_loss(pred_ns, next_states) +
        0.2 * F.smooth_l1_loss(pred_r, rewards) +
        0.5 * F.smooth_l1_loss(pred_h, next_heights)
    )


def online_train_sd_model(
    model,
    env,
    total_steps: int,
    seed: int = 0,
    seed_steps: int = 2500,
    batch_size: int = 256,
    lr: float = 1e-3,
    updates_per_step: int = 1,
    seed_pretrain_updates: int = 512,
    device: str = "cpu",
    planner_horizon: int = 8,
    planner_population: int = 128,
    planner_elites: int = 16,
    planner_iterations: int = 3,
    planner_temperature: float = 0.5,
    planner_num_prior_trajs: int = 16,
    planner_terminal_value_coef: float = 1.0,
    use_plan_reward: bool = True,
    behavior_guide_mix_start: float = 0.45,
    behavior_guide_mix_end: float = 0.15,
    value_net=None,
    use_tva: bool = False,
    tva_gamma: float = 0.99,
    tva_true_coef: float = 1.0,
    tva_shaped_coef: float = 1.0,
    tva_lr: float = 1e-3,
    tva_updates_per_episode: int = 25,
    tva_batch_size: int = 256,
    actor_net=None,
    use_actor_prior: bool = False,
    actor_lr: float = 1e-3,
    actor_updates_per_episode: int = 25,
    actor_batch_size: int = 256,
    planner_num_actor_trajs: int = 8,
    checkpoint_path: str = "",
    checkpoint_every: int = 500,
    resume_state: Dict[str, object] | None = None,
):
    """Online training loop inspired by TD-MPC2.

    The migrated parts are:
    1. seed-data collection followed by a concentrated pretrain burst,
    2. online planner-driven data generation after seed steps,
    3. replay-based repeated reuse of past transitions.
    """

    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    replay = TransitionReplay(capacity=total_steps)
    rng = np.random.RandomState(seed)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=tva_lr) if (use_tva and value_net is not None) else None
    actor_optim = torch.optim.Adam(actor_net.parameters(), lr=actor_lr) if (use_actor_prior and actor_net is not None) else None

    step_start = 0
    last_loss = None
    last_tva_loss = None
    last_actor_loss = None
    actor_ready = False
    prev_mean = None
    solver_event_state = init_hopper_event_state_torch(1, torch.device(device), torch.float32)
    post_takeoff_steps = -1
    frontier_states, frontier_actions = [], []
    strong_frontier_states, strong_frontier_actions = [], []
    locked_frontier_states, locked_frontier_targets = [], []
    locked_frontier_actions = []
    frontier_lock_remaining = 0
    last_lock_diag = None
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optim.load_state_dict(resume_state["model_optim_state"])
        replay.load_state_dict(resume_state["replay_state"])
        replay.capacity = total_steps
        rng.set_state(resume_state["numpy_rng_state"])
        step_start = int(resume_state["step"])
        last_loss = resume_state.get("last_loss")
        last_tva_loss = resume_state.get("last_tva_loss")
        if use_tva and value_net is not None and resume_state.get("value_net_state") is not None:
            value_net.load_state_dict(resume_state["value_net_state"])
            value_optim.load_state_dict(resume_state["value_optim_state"])
        if use_actor_prior and actor_net is not None and resume_state.get("actor_net_state") is not None:
            actor_net.load_state_dict(resume_state["actor_net_state"])
            actor_optim.load_state_dict(resume_state["actor_optim_state"])
            actor_ready = bool(resume_state.get("actor_ready", True))
        last_actor_loss = resume_state.get("last_actor_loss")
        log(f"Resuming online Hopper training from step {step_start}")

    ts = env.reset()
    obs = ts.observation
    episode_states, episode_actions = [], []
    episode_rewards, episode_next_states, episode_dones = [], [], []
    recent_window = 512

    for step in range(step_start, total_steps):
        state = DMControlHopperSDModel.obs_to_state(obs)
        height = float(env.physics.height())

        if step < seed_steps:
            action = guided_exploration_action(state, rng, step)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            planner_out = plan_action_cem(
                model,
                state_t,
                horizon=planner_horizon,
                population=planner_population,
                elites=planner_elites,
                iterations=planner_iterations,
                use_plan_reward=use_plan_reward,
                prev_mean=prev_mean,
                temperature=planner_temperature,
                num_prior_trajs=planner_num_prior_trajs,
                num_actor_trajs=planner_num_actor_trajs,
                terminal_value_coef=planner_terminal_value_coef,
                eval_mode=False,
                value_net=value_net if use_tva else None,
                actor_net=actor_net if (use_actor_prior and actor_ready) else None,
                event_state=solver_event_state,
                lock_filter=frontier_lock_remaining > 0,
                return_diag=frontier_lock_remaining > 0,
            )
            if frontier_lock_remaining > 0:
                action_t, prev_mean, last_lock_diag = planner_out
            else:
                action_t, prev_mean = planner_out
                last_lock_diag = None
            action = action_t.detach().cpu().numpy().astype(np.float32)
            progress = (step - seed_steps) / max(1, total_steps - seed_steps)
            guide_mix = behavior_guide_mix_start + progress * (behavior_guide_mix_end - behavior_guide_mix_start)
            guide_mix = phase_conditioned_guide_mix(state, guide_mix, event_state=solver_event_state)
            guide = hopper_guidance_action(state, event_state=solver_event_state)
            action = np.clip((1.0 - guide_mix) * action + guide_mix * guide, -1.0, 1.0).astype(np.float32)
            if post_takeoff_steps >= 0:
                flight_mix = 0.55 if post_takeoff_steps == 0 else (0.35 if post_takeoff_steps == 1 else 0.20)
                flight_template = hopper_flight_template_action(post_takeoff_steps)
                action = np.clip((1.0 - flight_mix) * action + flight_mix * flight_template, -1.0, 1.0).astype(np.float32)

        ts = env.step(action)
        next_obs = ts.observation
        next_state = DMControlHopperSDModel.obs_to_state(next_obs)
        next_height = float(env.physics.height())
        replay.add(state, action, float(ts.reward or 0.0), next_state, bool(ts.last()), height, next_height)
        with torch.no_grad():
            _, solver_event_state, event_info = predict_phase_progress_reward(
                torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(action, dtype=torch.float32, device=device),
                torch.tensor(next_state, dtype=torch.float32, device=device),
                model.hop_speed,
                event_state=solver_event_state,
                return_info=True,
            )
        if float(event_info["valid_takeoff"].item()) > 0.5:
            post_takeoff_steps = 0
        elif post_takeoff_steps >= 0:
            post_takeoff_steps += 1
            if post_takeoff_steps >= 3 or float(event_info["next_valid_flight"].item()) > 0.20:
                post_takeoff_steps = -1
        episode_states.append(state.copy())
        episode_actions.append(action.copy())
        episode_rewards.append(float(ts.reward or 0.0))
        episode_next_states.append(next_state.copy())
        episode_dones.append(bool(ts.last()))

        if frontier_lock_remaining > 0:
            frontier_lock_remaining -= 1

        if len(replay) >= batch_size and step + 1 >= seed_steps and frontier_lock_remaining <= 0:
            num_updates = seed_pretrain_updates if step + 1 == seed_steps else updates_per_step
            if step + 1 == seed_steps:
                log(f"reached seed_steps={seed_steps}; start replay pretraining for {num_updates} updates")
            for _ in range(num_updates):
                batch = replay.sample_batch(batch_size=batch_size, device=device)
                loss = _batch_loss(model, batch)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
                last_loss = float(loss.item())

        obs = next_obs
        if ts.last():
            episode_summary = summarize_hopper_rollout(
                model,
                np.asarray(episode_states, dtype=np.float32),
                np.asarray(episode_actions, dtype=np.float32),
                np.asarray(episode_rewards, dtype=np.float32),
                np.asarray(episode_next_states, dtype=np.float32),
                device=device,
                actor_net=actor_net if (use_actor_prior and actor_ready) else None,
            )
            ep_event_stats = _episode_event_stats(
                model,
                episode_states,
                episode_actions,
                episode_next_states,
                episode_dones,
                device=device,
            )
            frontier_mask, strong_frontier_mask = _frontier_masks(ep_event_stats)
            if frontier_mask.any():
                frontier_states.extend(np.asarray(episode_states, dtype=np.float32)[frontier_mask].tolist())
                frontier_actions.extend(np.asarray(episode_actions, dtype=np.float32)[frontier_mask].tolist())
                if len(frontier_states) > 4096:
                    frontier_states = frontier_states[-4096:]
                    frontier_actions = frontier_actions[-4096:]
            if strong_frontier_mask.any():
                strong_frontier_states.extend(np.asarray(episode_states, dtype=np.float32)[strong_frontier_mask].tolist())
                strong_frontier_actions.extend(np.asarray(episode_actions, dtype=np.float32)[strong_frontier_mask].tolist())
                if len(strong_frontier_states) > 2048:
                    strong_frontier_states = strong_frontier_states[-2048:]
                    strong_frontier_actions = strong_frontier_actions[-2048:]
            frontier_hits = int(ep_event_stats["valid_takeoffs"].sum() + ep_event_stats["valid_flights"].sum())
            strong_frontier_hits = int(strong_frontier_mask.sum())
            completed_cycle = bool(
                ep_event_stats["valid_takeoffs"].any()
                and ep_event_stats["valid_flights"].any()
                and ep_event_stats["valid_landings"].any()
            )
            pseudo_ratio = float(ep_event_stats["pseudo_flight_mask"].mean()) if len(ep_event_stats["pseudo_flight_mask"]) > 0 else 0.0
            allow_regular_update = (frontier_hits > 0 and pseudo_ratio < 0.20) or (pseudo_ratio < 0.12)
            allow_value_update = (strong_frontier_hits > 0) or (frontier_hits > 0 and pseudo_ratio < 0.25) or (pseudo_ratio < 0.12)
            if use_tva and value_net is not None and len(episode_states) > 0:
                ep_states_t, ep_targets_t = _episode_value_targets(
                    model,
                    episode_states,
                    episode_actions,
                    episode_rewards,
                    episode_next_states,
                    episode_dones,
                    gamma=tva_gamma,
                    true_coef=tva_true_coef,
                    shaped_coef=tva_shaped_coef,
                    device=device,
                )
                if completed_cycle:
                    cycle_mask = strong_frontier_mask if strong_frontier_mask.any() else frontier_mask
                    if cycle_mask.any():
                        cycle_states = np.asarray(episode_states, dtype=np.float32)[cycle_mask]
                        cycle_actions = np.asarray(episode_actions, dtype=np.float32)[cycle_mask]
                        cycle_targets = ep_targets_t[cycle_mask].detach().cpu().numpy()
                        locked_frontier_states.extend(cycle_states.tolist())
                        locked_frontier_actions.extend(cycle_actions.tolist())
                        locked_frontier_targets.extend(cycle_targets.tolist())
                        if len(locked_frontier_states) > 2048:
                            locked_frontier_states = locked_frontier_states[-2048:]
                            locked_frontier_actions = locked_frontier_actions[-2048:]
                            locked_frontier_targets = locked_frontier_targets[-2048:]
                        frontier_lock_remaining = max(frontier_lock_remaining, 900)
                if frontier_lock_remaining > 0 and locked_frontier_states:
                    value_net.train()
                    lock_states_t = torch.tensor(np.asarray(locked_frontier_states, dtype=np.float32), dtype=torch.float32, device=device)
                    lock_targets_t = torch.tensor(np.asarray(locked_frontier_targets, dtype=np.float32), dtype=torch.float32, device=device)
                    for _ in range(max(12, 2 * tva_updates_per_episode)):
                        if len(lock_states_t) <= tva_batch_size:
                            batch_idx = torch.arange(len(lock_states_t), device=device)
                        else:
                            batch_idx = torch.randperm(len(lock_states_t), device=device)[:tva_batch_size]
                        pred = value_net(lock_states_t[batch_idx])
                        vloss = F.smooth_l1_loss(pred, lock_targets_t[batch_idx])
                        value_optim.zero_grad()
                        vloss.backward()
                        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                        value_optim.step()
                        last_tva_loss = float(vloss.item())
                    value_net.eval()
                elif allow_value_update:
                    value_net.train()
                    for _ in range(tva_updates_per_episode):
                        if len(ep_states_t) <= tva_batch_size:
                            batch_idx = torch.arange(len(ep_states_t), device=device)
                        else:
                            batch_idx = torch.randperm(len(ep_states_t), device=device)[:tva_batch_size]
                        pred = value_net(ep_states_t[batch_idx])
                        vloss = F.smooth_l1_loss(pred, ep_targets_t[batch_idx])
                        value_optim.zero_grad()
                        vloss.backward()
                        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                        value_optim.step()
                        last_tva_loss = float(vloss.item())
                    value_net.eval()
            if use_actor_prior and actor_net is not None and len(episode_states) > 0:
                actor_mask = _episode_actor_mask(
                    model,
                    episode_states,
                    episode_actions,
                    episode_next_states,
                    episode_dones,
                    device=device,
                )
                if strong_frontier_states:
                    strong_frontier_arr_states = np.asarray(strong_frontier_states, dtype=np.float32)
                    strong_frontier_arr_actions = np.asarray(strong_frontier_actions, dtype=np.float32)
                    actor_loss_frontier = train_actor_prior(
                        actor_net,
                        strong_frontier_arr_states,
                        strong_frontier_arr_actions,
                        actor_optim,
                        updates=max(16, 2 * actor_updates_per_episode),
                        batch_size=actor_batch_size,
                        device=device,
                    )
                    if actor_loss_frontier is not None:
                        last_actor_loss = actor_loss_frontier
                        actor_ready = True
                elif frontier_states:
                    frontier_arr_states = np.asarray(frontier_states, dtype=np.float32)
                    frontier_arr_actions = np.asarray(frontier_actions, dtype=np.float32)
                    actor_loss_frontier = train_actor_prior(
                        actor_net,
                        frontier_arr_states,
                        frontier_arr_actions,
                        actor_optim,
                        updates=max(8, actor_updates_per_episode),
                        batch_size=actor_batch_size,
                        device=device,
                    )
                    if actor_loss_frontier is not None:
                        last_actor_loss = actor_loss_frontier
                        actor_ready = True
                if (not strong_frontier_states) and (not frontier_states) and allow_regular_update:
                    actor_loss = train_actor_prior(
                        actor_net,
                        episode_states,
                        episode_actions,
                        actor_optim,
                        updates=actor_updates_per_episode,
                        batch_size=actor_batch_size,
                        device=device,
                        sample_mask=actor_mask,
                    )
                    if actor_loss is not None:
                        last_actor_loss = actor_loss
                        actor_ready = True

            log(format_hopper_summary(f"episode end @{step + 1}", episode_summary))

            ts = env.reset()
            obs = ts.observation
            prev_mean = None
            solver_event_state = init_hopper_event_state_torch(1, torch.device(device), torch.float32)
            post_takeoff_steps = -1
            episode_states, episode_actions = [], []
            episode_rewards, episode_next_states, episode_dones = [], [], []

        if (step + 1) % 500 == 0 or step == total_steps - 1:
            loss_msg = "n/a" if last_loss is None else f"{last_loss:.4f}"
            tva_msg = "n/a" if last_tva_loss is None else f"{last_tva_loss:.4f}"
            actor_msg = "n/a" if last_actor_loss is None else f"{last_actor_loss:.4f}"
            log(f"online step {step + 1:5d} | replay={len(replay):5d} | last_loss={loss_msg} | tva={tva_msg} | actor={actor_msg} | lock={frontier_lock_remaining}")
            if last_lock_diag is not None:
                log(
                    "lockdiag | "
                    f"elite_vt={last_lock_diag['elite_vt']:.2f} | "
                    f"elite_vf={last_lock_diag['elite_vf']:.2f} | "
                    f"elite_vl={last_lock_diag['elite_vl']:.2f} | "
                    f"elite_full={last_lock_diag['elite_full']:.2f} | "
                    f"elite_pseudo={last_lock_diag['elite_pseudo']:.2f} | "
                    f"elite_f={last_lock_diag['elite_flight']:.2f} | "
                    f"a0={last_lock_diag['elite_a0']:.2f}"
                )
            tail = replay.as_dataset()
            if len(tail["states"]) > 0:
                start = max(0, len(tail["states"]) - recent_window)
                recent_summary = summarize_hopper_rollout(
                    model,
                    tail["states"][start:],
                    tail["actions"][start:],
                    tail["rewards"][start:],
                    tail["next_states"][start:],
                    device=device,
                    actor_net=actor_net if (use_actor_prior and actor_ready) else None,
                )
                log(format_hopper_summary(f"recent{len(tail['states'][start:])}", recent_summary))
        if checkpoint_path and ((step + 1) % checkpoint_every == 0 or step == total_steps - 1):
            save_training_checkpoint(
                checkpoint_path,
                {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "model_optim_state": optim.state_dict(),
                    "replay_state": replay.state_dict(),
                    "numpy_rng_state": rng.get_state(),
                    "last_loss": last_loss,
                    "last_tva_loss": last_tva_loss,
                    "last_actor_loss": last_actor_loss,
                    "value_net_state": value_net.state_dict() if (use_tva and value_net is not None) else None,
                    "value_optim_state": value_optim.state_dict() if (use_tva and value_optim is not None) else None,
                    "actor_net_state": actor_net.state_dict() if (use_actor_prior and actor_net is not None) else None,
                    "actor_optim_state": actor_optim.state_dict() if (use_actor_prior and actor_optim is not None) else None,
                    "actor_ready": actor_ready,
                },
            )

    if use_tva and value_net is not None and len(episode_states) > 0:
        ep_event_stats = _episode_event_stats(
            model,
            episode_states,
            episode_actions,
            episode_next_states,
            episode_dones,
            device=device,
        )
        frontier_mask, strong_frontier_mask = _frontier_masks(ep_event_stats)
        if frontier_mask.any():
            frontier_states.extend(np.asarray(episode_states, dtype=np.float32)[frontier_mask].tolist())
            frontier_actions.extend(np.asarray(episode_actions, dtype=np.float32)[frontier_mask].tolist())
            if len(frontier_states) > 4096:
                frontier_states = frontier_states[-4096:]
                frontier_actions = frontier_actions[-4096:]
        if strong_frontier_mask.any():
            strong_frontier_states.extend(np.asarray(episode_states, dtype=np.float32)[strong_frontier_mask].tolist())
            strong_frontier_actions.extend(np.asarray(episode_actions, dtype=np.float32)[strong_frontier_mask].tolist())
            if len(strong_frontier_states) > 2048:
                strong_frontier_states = strong_frontier_states[-2048:]
                strong_frontier_actions = strong_frontier_actions[-2048:]
        frontier_hits = int(ep_event_stats["valid_takeoffs"].sum() + ep_event_stats["valid_flights"].sum())
        strong_frontier_hits = int(strong_frontier_mask.sum())
        pseudo_ratio = float(ep_event_stats["pseudo_flight_mask"].mean()) if len(ep_event_stats["pseudo_flight_mask"]) > 0 else 0.0
        allow_regular_update = (frontier_hits > 0 and pseudo_ratio < 0.20) or (pseudo_ratio < 0.12)
        allow_value_update = (strong_frontier_hits > 0) or (frontier_hits > 0 and pseudo_ratio < 0.25) or (pseudo_ratio < 0.12)
        if allow_value_update:
            value_net.train()
            ep_states_t, ep_targets_t = _episode_value_targets(
                model,
                episode_states,
                episode_actions,
                episode_rewards,
                episode_next_states,
                episode_dones,
                gamma=tva_gamma,
                true_coef=tva_true_coef,
                shaped_coef=tva_shaped_coef,
                device=device,
            )
            for _ in range(tva_updates_per_episode):
                if len(ep_states_t) <= tva_batch_size:
                    batch_idx = torch.arange(len(ep_states_t), device=device)
                else:
                    batch_idx = torch.randperm(len(ep_states_t), device=device)[:tva_batch_size]
                pred = value_net(ep_states_t[batch_idx])
                vloss = F.smooth_l1_loss(pred, ep_targets_t[batch_idx])
                value_optim.zero_grad()
                vloss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                value_optim.step()
                last_tva_loss = float(vloss.item())
            value_net.eval()
    if use_actor_prior and actor_net is not None and len(episode_states) > 0:
        if strong_frontier_states:
            strong_frontier_arr_states = np.asarray(strong_frontier_states, dtype=np.float32)
            strong_frontier_arr_actions = np.asarray(strong_frontier_actions, dtype=np.float32)
            actor_loss_frontier = train_actor_prior(
                actor_net,
                strong_frontier_arr_states,
                strong_frontier_arr_actions,
                actor_optim,
                updates=max(16, 2 * actor_updates_per_episode),
                batch_size=actor_batch_size,
                device=device,
            )
            if actor_loss_frontier is not None:
                last_actor_loss = actor_loss_frontier
                actor_ready = True
        elif frontier_states:
            frontier_arr_states = np.asarray(frontier_states, dtype=np.float32)
            frontier_arr_actions = np.asarray(frontier_actions, dtype=np.float32)
            actor_loss_frontier = train_actor_prior(
                actor_net,
                frontier_arr_states,
                frontier_arr_actions,
                actor_optim,
                updates=max(8, actor_updates_per_episode),
                batch_size=actor_batch_size,
                device=device,
            )
            if actor_loss_frontier is not None:
                last_actor_loss = actor_loss_frontier
                actor_ready = True
        actor_mask = _episode_actor_mask(
            model,
            episode_states,
            episode_actions,
            episode_next_states,
            episode_dones,
            device=device,
        )
        if (not strong_frontier_states) and (not frontier_states) and allow_regular_update:
            actor_loss = train_actor_prior(
                actor_net,
                episode_states,
                episode_actions,
                actor_optim,
                updates=actor_updates_per_episode,
                batch_size=actor_batch_size,
                device=device,
                sample_mask=actor_mask,
            )
            if actor_loss is not None:
                last_actor_loss = actor_loss
                actor_ready = True

    if checkpoint_path:
        save_training_checkpoint(
            checkpoint_path,
            {
                "step": total_steps,
                "model_state": model.state_dict(),
                "model_optim_state": optim.state_dict(),
                "replay_state": replay.state_dict(),
                "numpy_rng_state": rng.get_state(),
                "last_loss": last_loss,
                "last_tva_loss": last_tva_loss,
                "last_actor_loss": last_actor_loss,
                "value_net_state": value_net.state_dict() if (use_tva and value_net is not None) else None,
                "value_optim_state": value_optim.state_dict() if (use_tva and value_optim is not None) else None,
                "actor_net_state": actor_net.state_dict() if (use_actor_prior and actor_net is not None) else None,
                "actor_optim_state": actor_optim.state_dict() if (use_actor_prior and actor_optim is not None) else None,
                "actor_ready": actor_ready,
            },
        )

    model.eval()
    return replay.as_dataset(), {"final_loss": last_loss, "final_tva_loss": last_tva_loss, "final_actor_loss": last_actor_loss}


def evaluate_model(model, data, rollout_horizons=(1, 5, 10), device="cpu"):
    model.to(device)
    states = torch.tensor(data["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=device)
    dones = np.asarray(data["dones"], dtype=bool)
    with torch.no_grad():
        pred_next = model(states, actions)
    metrics = {"one_step_rmse": _state_rmse(pred_next, next_states)}
    valid_starts = []
    max_h = max(rollout_horizons)
    for start in range(len(states) - max_h):
        if dones[start:start + max_h].any():
            continue
        valid_starts.append(start)
    starts = valid_starts[: min(128, len(valid_starts))]
    if not starts:
        return metrics
    for horizon in rollout_horizons:
        errs = []
        for start in starts:
            pred = states[start:start + 1]
            for t in range(horizon):
                pred = model(pred, actions[start + t:start + t + 1])
            target = next_states[start + horizon - 1:start + horizon]
            errs.append(_state_rmse(pred, target))
        metrics[f"rollout_{horizon}_rmse"] = float(np.mean(errs))
    return metrics

def terminal_plan_value(model, state, value_net=None):
    """Terminal value for planning.

    If a learned TVA value net is provided, use it. Otherwise fall back
    to the smooth heuristic bootstrap used in earlier Hopper prototypes.
    """
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if value_net is not None:
        with torch.no_grad():
            return value_net(state).squeeze(-1)
    height = model.predict_height(state)
    rooty = state[:, 1]
    x_dot = state[:, 6]
    z_dot = state[:, 7]
    touch = state[:, 13:].mean(dim=-1)
    stand_soft = torch.clamp((height - 0.45) / 0.35, min=0.0, max=1.0)
    upright = (1.0 + torch.cos(rooty)) / 2.0
    speed_progress = torch.clamp(x_dot / max(model.hop_speed, 1e-6), min=0.0, max=1.0)
    lift = torch.clamp(z_dot / 2.0, min=0.0, max=1.0)
    light_contact = 1.0 - 0.3 * torch.clamp(touch, min=0.0, max=1.0)
    return stand_soft * upright * (0.40 + 0.40 * speed_progress + 0.20 * lift) * light_contact


def terminal_event_bonus(event_state: Dict[str, torch.Tensor] | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if event_state is None:
        return torch.zeros(1, dtype=dtype, device=device)
    credit = event_state["compression_credit"].to(device=device, dtype=dtype)
    armed = event_state["takeoff_armed"].to(device=device, dtype=dtype)
    valid_flight = event_state["valid_flight"].to(device=device, dtype=dtype)
    return 0.25 * torch.clamp(credit, 0.0, 1.2) + 0.70 * torch.clamp(armed, 0.0, 1.0) + 0.15 * torch.clamp(valid_flight, 0.0, 1.0)


def rollout_return(model, init_state, action_seqs, gamma=0.99, use_plan_reward=True,
                   terminal_value_fn=None, terminal_value_coef=1.0, value_net=None,
                   event_state=None, return_stats: bool = False):
    if init_state.dim() == 1:
        init_state = init_state.unsqueeze(0)
    pop, horizon, _ = action_seqs.shape
    state = init_state.expand(pop, -1)
    if event_state is None:
        event_state = init_hopper_event_state_torch(pop, state.device, state.dtype)
    else:
        event_state = {
            key: value.to(device=state.device, dtype=state.dtype).expand(pop).clone()
            for key, value in event_state.items()
        }
    returns = torch.zeros(pop, dtype=state.dtype, device=state.device)
    discount = torch.ones(pop, dtype=state.dtype, device=state.device)
    active = torch.ones(pop, dtype=state.dtype, device=state.device)
    pseudo_flight_streak = torch.zeros(pop, dtype=state.dtype, device=state.device)
    unrecovered_flight_streak = torch.zeros(pop, dtype=state.dtype, device=state.device)
    orphan_flight_streak = torch.zeros(pop, dtype=state.dtype, device=state.device)
    vt_hits = torch.zeros(pop, dtype=state.dtype, device=state.device)
    vf_hits = torch.zeros(pop, dtype=state.dtype, device=state.device)
    vl_hits = torch.zeros(pop, dtype=state.dtype, device=state.device)
    pseudo_hits = torch.zeros(pop, dtype=state.dtype, device=state.device)
    flight_hits = torch.zeros(pop, dtype=state.dtype, device=state.device)
    for t in range(horizon):
        action = action_seqs[:, t]
        next_state = model(state, action)
        if use_plan_reward:
            reward, event_state, info = predict_structured_plan_reward(
                model, state, action, next_state, event_state=event_state, return_info=True
            )
        else:
            reward = model.predict_reward(state, action, next_state)
            info = None
        if info is not None:
            dtype = state.dtype
            pseudo_flight = (
                (info["next_phase"] == PHASE_FLIGHT).to(dtype)
                * (info["next_valid_flight"] <= 0.20).to(dtype)
            )
            orphan_flight = (
                (info["next_phase"] == PHASE_FLIGHT).to(dtype)
                * (info["prev_takeoff_window"] <= 0.18).to(dtype)
            )
            recovery_event = (
                (info["valid_landing"] > 0.5).to(dtype)
                + (info["reset_cycle"] > 0.5).to(dtype)
            ).clamp(max=1.0)
            vt_hits = vt_hits + (info["valid_takeoff"] > 0.5).to(dtype)
            vf_hits = vf_hits + (info["next_valid_flight"] > 0.20).to(dtype)
            vl_hits = vl_hits + (info["valid_landing"] > 0.5).to(dtype)
            pseudo_hits = pseudo_hits + pseudo_flight
            flight_hits = flight_hits + (info["next_phase"] == PHASE_FLIGHT).to(dtype)
            pseudo_flight_streak = torch.where(
                pseudo_flight > 0.5,
                pseudo_flight_streak + 1.0,
                torch.where((info["next_phase"] == PHASE_FLIGHT).to(dtype) > 0.5, 0.5 * pseudo_flight_streak, torch.zeros_like(pseudo_flight_streak)),
            )
            unrecovered_flight_streak = torch.where(
                ((info["next_phase"] == PHASE_FLIGHT).to(dtype) > 0.5) & (recovery_event < 0.5),
                unrecovered_flight_streak + 1.0,
                torch.zeros_like(unrecovered_flight_streak),
            )
            orphan_flight_streak = torch.where(
                (orphan_flight > 0.5) & (recovery_event < 0.5),
                orphan_flight_streak + 1.0,
                torch.zeros_like(orphan_flight_streak),
            )
            hard_fail = (
                (orphan_flight_streak >= 4.0)
                & (pseudo_flight_streak >= 3.0)
                & (unrecovered_flight_streak >= 4.0)
            ).to(dtype)
            reward = reward - (
                0.35 * pseudo_flight
                + 0.25 * orphan_flight
                + 0.08 * torch.clamp(pseudo_flight_streak - 1.0, min=0.0, max=4.0)
                + 1.75 * hard_fail
            )
            active = active * (1.0 - hard_fail)
        returns = returns + active * discount * reward
        discount = discount * gamma
        state = next_state
    if terminal_value_fn is not None:
        returns = returns + active * discount * terminal_value_coef * terminal_value_fn(model, state, value_net=value_net)
    returns = returns + active * discount * terminal_event_bonus(event_state, state.device, state.dtype)
    if return_stats:
        stats = {
            "vt_hits": vt_hits,
            "vf_hits": vf_hits,
            "vl_hits": vl_hits,
            "pseudo_frac": pseudo_hits / max(1, horizon),
            "flight_frac": flight_hits / max(1, horizon),
        }
        return returns, stats
    return returns


def hopper_prior_trajectories(model, state, horizon, num_trajs, device, noise_scale=0.12, event_state=None):
    """Guided prior trajectories rolled out under the current model."""
    if num_trajs <= 0:
        return None
    seqs = []
    for _ in range(num_trajs):
        sim_state = state.clone()
        if sim_state.dim() == 1:
            sim_state = sim_state.unsqueeze(0)
        sim_event_state = init_hopper_event_state_torch(1, sim_state.device, sim_state.dtype) if event_state is None else {
            key: value.to(device=sim_state.device, dtype=sim_state.dtype).clone() for key, value in event_state.items()
        }
        traj = []
        for _ in range(horizon):
            guide = hopper_guidance_action(sim_state[0].detach().cpu().numpy(), event_state=sim_event_state)
            a = torch.tensor(guide, dtype=torch.float32, device=device)
            a = torch.clamp(a + noise_scale * torch.randn_like(a), -1.0, 1.0)
            traj.append(a)
            with torch.no_grad():
                next_state = model(sim_state, a.unsqueeze(0))
                _, sim_event_state, _ = predict_phase_progress_reward(
                    sim_state, a.unsqueeze(0), next_state, model.hop_speed,
                    event_state=sim_event_state, return_info=True,
                )
                sim_state = next_state
        seqs.append(torch.stack(traj, dim=0))
    return torch.stack(seqs, dim=0)


def hopper_release_prior_trajectories(model, state, horizon, num_trajs, device, noise_scale=0.08, event_state=None):
    """Prior trajectories specialized for the armed-to-release transition."""
    if num_trajs <= 0:
        return None
    seqs = []
    for _ in range(num_trajs):
        sim_state = state.clone()
        if sim_state.dim() == 1:
            sim_state = sim_state.unsqueeze(0)
        sim_event_state = init_hopper_event_state_torch(1, sim_state.device, sim_state.dtype) if event_state is None else {
            key: value.to(device=sim_state.device, dtype=sim_state.dtype).clone() for key, value in event_state.items()
        }
        traj = []
        for t in range(horizon):
            if t < 2:
                base = np.array([0.15, -0.55, 1.0, 0.95], dtype=np.float32)
            elif t < 4:
                base = np.array([0.05, -0.30, 0.55, -0.25], dtype=np.float32)
            else:
                base = hopper_guidance_action(sim_state[0].detach().cpu().numpy(), event_state=sim_event_state)
            a = torch.tensor(base, dtype=torch.float32, device=device)
            a = torch.clamp(a + noise_scale * torch.randn_like(a), -1.0, 1.0)
            traj.append(a)
            with torch.no_grad():
                next_state = model(sim_state, a.unsqueeze(0))
                _, sim_event_state, _ = predict_phase_progress_reward(
                    sim_state, a.unsqueeze(0), next_state, model.hop_speed,
                    event_state=sim_event_state, return_info=True,
                )
                sim_state = next_state
        seqs.append(torch.stack(traj, dim=0))
    return torch.stack(seqs, dim=0)


def hopper_windowed_flight_trajectories(model, state, horizon, num_trajs, device, noise_scale=0.06, event_state=None):
    """Candidate trajectories constrained around the short post-takeoff flight template."""
    if num_trajs <= 0:
        return None
    seqs = []
    for _ in range(num_trajs):
        sim_state = state.clone()
        if sim_state.dim() == 1:
            sim_state = sim_state.unsqueeze(0)
        sim_event_state = init_hopper_event_state_torch(1, sim_state.device, sim_state.dtype) if event_state is None else {
            key: value.to(device=sim_state.device, dtype=sim_state.dtype).clone() for key, value in event_state.items()
        }
        traj = []
        for t in range(horizon):
            if t < 3:
                base = hopper_flight_template_action(t)
            else:
                base = hopper_guidance_action(sim_state[0].detach().cpu().numpy(), event_state=sim_event_state)
            a = torch.tensor(base, dtype=torch.float32, device=device)
            a = torch.clamp(a + noise_scale * torch.randn_like(a), -1.0, 1.0)
            traj.append(a)
            with torch.no_grad():
                next_state = model(sim_state, a.unsqueeze(0))
                _, sim_event_state, _ = predict_phase_progress_reward(
                    sim_state, a.unsqueeze(0), next_state, model.hop_speed,
                    event_state=sim_event_state, return_info=True,
                )
                sim_state = next_state
        seqs.append(torch.stack(traj, dim=0))
    return torch.stack(seqs, dim=0)


def actor_prior_trajectories(model, actor_net, state, horizon, num_trajs, device, noise_scale=0.10, event_state=None):
    """Roll out actor-generated candidate trajectories under the SD model."""
    if num_trajs <= 0 or actor_net is None:
        return None
    actor_net.eval()
    seqs = []
    for _ in range(num_trajs):
        sim_state = state.clone()
        if sim_state.dim() == 1:
            sim_state = sim_state.unsqueeze(0)
        sim_event_state = init_hopper_event_state_torch(1, sim_state.device, sim_state.dtype) if event_state is None else {
            key: value.to(device=sim_state.device, dtype=sim_state.dtype).clone() for key, value in event_state.items()
        }
        traj = []
        for _ in range(horizon):
            phase_onehot = hopper_phase_onehot_torch(sim_state)
            with torch.no_grad():
                a = actor_net(sim_state, phase_onehot).squeeze(0)
            a = torch.clamp(a + noise_scale * torch.randn_like(a), -1.0, 1.0)
            traj.append(a)
            with torch.no_grad():
                next_state = model(sim_state, a.unsqueeze(0))
                _, sim_event_state, _ = predict_phase_progress_reward(
                    sim_state, a.unsqueeze(0), next_state, model.hop_speed,
                    event_state=sim_event_state, return_info=True,
                )
                sim_state = next_state
        seqs.append(torch.stack(traj, dim=0))
    return torch.stack(seqs, dim=0)


def plan_action_cem(model, state, horizon=8, population=128, elites=16, iterations=3,
                    gamma=0.99, use_plan_reward=True, prev_mean=None,
                    temperature=0.5, min_std=0.05, max_std=2.0,
                    num_prior_trajs=16, num_actor_trajs=8, terminal_value_coef=1.0,
                    eval_mode=True, value_net=None, actor_net=None, event_state=None,
                    lock_filter=False, return_diag=False):
    device = state.device
    action_dim = 4
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.full((horizon, action_dim), max_std, device=device)
    if prev_mean is not None:
        mean[:-1] = prev_mean[1:]
    armed_level = 0.0
    credit_level = 0.0
    takeoff_window_level = 0.0
    valid_flight_level = 0.0
    if event_state is not None:
        armed_level = float(event_state["takeoff_armed"].mean().item())
        credit_level = float(event_state["compression_credit"].mean().item())
        takeoff_window_level = float(event_state["takeoff_window"].mean().item())
        valid_flight_level = float(event_state["valid_flight"].mean().item())
    diag = None
    with torch.no_grad():
        for _ in range(iterations):
            noise = torch.randn(population, horizon, action_dim, device=device)
            seqs = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, -1.0, 1.0)
            if num_prior_trajs > 0:
                base_prior_n = min(num_prior_trajs, population)
                offset = 0
                if takeoff_window_level > 0.18 or valid_flight_level > 0.10:
                    flight_n = max(1, base_prior_n // 2)
                    flight_prior = hopper_windowed_flight_trajectories(
                        model, state, horizon, min(flight_n, population), device, event_state=event_state
                    )
                    seqs[:flight_prior.shape[0]] = flight_prior
                    offset = flight_prior.shape[0]
                    base_prior_n = max(0, base_prior_n - offset)
                if armed_level > 0.18 or credit_level > 0.70:
                    release_n = max(1, base_prior_n // 2)
                    release_prior = hopper_release_prior_trajectories(model, state, horizon, min(release_n, population), device, event_state=event_state)
                    seqs[offset:offset + release_prior.shape[0]] = release_prior
                    offset = offset + release_prior.shape[0]
                    base_prior_n = max(0, base_prior_n - release_prior.shape[0])
                if base_prior_n > 0:
                    prior = hopper_prior_trajectories(model, state, horizon, base_prior_n, device, event_state=event_state)
                    seqs[offset:offset + prior.shape[0]] = prior
            if num_actor_trajs > 0 and actor_net is not None:
                prior_offset = min(num_prior_trajs, population)
                remain = max(0, population - prior_offset)
                if remain > 0:
                    actor_prior = actor_prior_trajectories(model, actor_net, state, horizon, min(num_actor_trajs, remain), device, event_state=event_state)
                    seqs[prior_offset:prior_offset + actor_prior.shape[0]] = actor_prior
            if lock_filter:
                returns, stats = rollout_return(
                    model, state, seqs, gamma=gamma, use_plan_reward=use_plan_reward,
                    terminal_value_fn=terminal_plan_value,
                    terminal_value_coef=terminal_value_coef,
                    value_net=value_net,
                    event_state=event_state,
                    return_stats=True,
                )
                vt_any = (stats["vt_hits"] > 0).to(seqs.dtype)
                vf_any = (stats["vf_hits"] > 0).to(seqs.dtype)
                vl_any = (stats["vl_hits"] > 0).to(seqs.dtype)
                full_cycle = vt_any * vf_any * vl_any
                pseudo_frac = stats["pseudo_frac"]
                flight_frac = stats["flight_frac"]
                no_event_flight = ((flight_frac > 0.65) & (vf_any < 0.5) & (vl_any < 0.5)).to(seqs.dtype)
                event_bonus = 0.20 * vt_any + 0.30 * vf_any + 0.35 * vl_any + 0.40 * full_cycle
                filter_scores = returns + event_bonus - 0.40 * pseudo_frac - 0.55 * no_event_flight
                elite_idx = torch.topk(filter_scores, k=elites, dim=0).indices
                elite_values = filter_scores[elite_idx]
                if return_diag:
                    elite_stats = {k: v[elite_idx] for k, v in stats.items()}
                    diag = {
                        "elite_vt": float((elite_stats["vt_hits"] > 0).float().mean().item()),
                        "elite_vf": float((elite_stats["vf_hits"] > 0).float().mean().item()),
                        "elite_vl": float((elite_stats["vl_hits"] > 0).float().mean().item()),
                        "elite_full": float((((elite_stats["vt_hits"] > 0) & (elite_stats["vf_hits"] > 0) & (elite_stats["vl_hits"] > 0)).float().mean().item())),
                        "elite_pseudo": float(elite_stats["pseudo_frac"].mean().item()),
                        "elite_flight": float(elite_stats["flight_frac"].mean().item()),
                        "elite_a0": float(elite[:, 0].norm(dim=-1).mean().item()) if 'elite' in locals() else 0.0,
                    }
            else:
                returns = rollout_return(
                    model, state, seqs, gamma=gamma, use_plan_reward=use_plan_reward,
                    terminal_value_fn=terminal_plan_value,
                    terminal_value_coef=terminal_value_coef,
                    value_net=value_net,
                    event_state=event_state,
                )
                elite_idx = torch.topk(returns, k=elites, dim=0).indices
                elite_values = returns[elite_idx]
            elite = seqs[elite_idx]
            if lock_filter and return_diag and diag is not None:
                diag["elite_a0"] = float(elite[:, 0].norm(dim=-1).mean().item())
            max_value = elite_values.max()
            score = torch.exp(temperature * (elite_values - max_value))
            score = score / (score.sum() + 1e-9)
            mean = (score.view(-1, 1, 1) * elite).sum(dim=0)
            std = ((score.view(-1, 1, 1) * (elite - mean.unsqueeze(0)) ** 2).sum(dim=0)).sqrt()
            std = std.clamp(min_std, max_std)
    action = mean[0]
    if not eval_mode:
        action = torch.clamp(action + std[0] * torch.randn(action_dim, device=device), -1.0, 1.0)
    if return_diag:
        return action.detach(), mean.detach(), diag
    return action.detach(), mean.detach()


def evaluate_agent(env, model, n_episodes=3, max_steps=1000, planner_horizon=8,
                   planner_population=128, planner_elites=16, planner_iterations=3,
                   device="cpu", use_plan_reward=True, planner_temperature=0.5,
                   planner_num_prior_trajs=16, planner_num_actor_trajs=8, planner_terminal_value_coef=1.0,
                   behavior_guide_eval_mix=0.25, value_net=None, actor_net=None):
    returns = []
    model.eval()
    model.to(device)
    for ep in range(n_episodes):
        ts = env.reset()
        ep_ret = 0.0
        prev_mean = None
        solver_event_state = init_hopper_event_state_torch(1, torch.device(device), torch.float32)
        post_takeoff_steps = -1
        ep_states, ep_actions, ep_rewards, ep_next_states = [], [], [], []
        for _ in range(max_steps):
            state = torch.tensor(DMControlHopperSDModel.obs_to_state(ts.observation), dtype=torch.float32, device=device)
            action, prev_mean = plan_action_cem(
                model, state,
                horizon=planner_horizon,
                population=planner_population,
                elites=planner_elites,
                iterations=planner_iterations,
                use_plan_reward=use_plan_reward,
                prev_mean=prev_mean,
                temperature=planner_temperature,
                num_prior_trajs=planner_num_prior_trajs,
                num_actor_trajs=planner_num_actor_trajs,
                terminal_value_coef=planner_terminal_value_coef,
                value_net=value_net,
                actor_net=actor_net,
                event_state=solver_event_state,
            )
            if behavior_guide_eval_mix > 0:
                state_np = state.detach().cpu().numpy()
                guide_mix = phase_conditioned_guide_mix(state_np, behavior_guide_eval_mix, event_state=solver_event_state)
                guide = torch.tensor(hopper_guidance_action(state_np, event_state=solver_event_state), dtype=torch.float32, device=device)
                action = torch.clamp((1.0 - guide_mix) * action + guide_mix * guide, -1.0, 1.0)
            if post_takeoff_steps >= 0:
                flight_mix = 0.55 if post_takeoff_steps == 0 else (0.35 if post_takeoff_steps == 1 else 0.20)
                flight_template = torch.tensor(hopper_flight_template_action(post_takeoff_steps), dtype=torch.float32, device=device)
                action = torch.clamp((1.0 - flight_mix) * action + flight_mix * flight_template, -1.0, 1.0)
            else:
                state_np = state.detach().cpu().numpy()
            action_np = action.cpu().numpy()
            ts = env.step(action_np)
            ep_ret += float(ts.reward or 0.0)
            ep_states.append(state_np)
            ep_actions.append(action_np)
            ep_rewards.append(float(ts.reward or 0.0))
            next_state_np = DMControlHopperSDModel.obs_to_state(ts.observation)
            ep_next_states.append(next_state_np)
            with torch.no_grad():
                _, solver_event_state, event_info = predict_phase_progress_reward(
                    state,
                    action,
                    torch.tensor(next_state_np, dtype=torch.float32, device=device),
                    model.hop_speed,
                    event_state=solver_event_state,
                    return_info=True,
                )
            if float(event_info["valid_takeoff"].item()) > 0.5:
                post_takeoff_steps = 0
            elif post_takeoff_steps >= 0:
                post_takeoff_steps += 1
                if post_takeoff_steps >= 3 or float(event_info["next_valid_flight"].item()) > 0.20:
                    post_takeoff_steps = -1
            if ts.last():
                break
        returns.append(ep_ret)
        log(f"eval episode {ep}: return={ep_ret:.1f}")
        summary = summarize_hopper_rollout(
            model,
            np.asarray(ep_states, dtype=np.float32),
            np.asarray(ep_actions, dtype=np.float32),
            np.asarray(ep_rewards, dtype=np.float32),
            np.asarray(ep_next_states, dtype=np.float32),
            device=device,
            actor_net=actor_net,
        )
        log(format_hopper_summary(f"eval episode {ep} diag", summary))
    return np.asarray(returns, dtype=np.float32)


def evaluate_random(env, n_episodes=3, max_steps=1000, seed=0):
    rng = np.random.RandomState(seed)
    returns = []
    for _ in range(n_episodes):
        ts = env.reset()
        ep_ret = 0.0
        for _ in range(max_steps):
            action = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            ts = env.step(action)
            ep_ret += float(ts.reward or 0.0)
            if ts.last():
                break
        returns.append(ep_ret)
    return np.asarray(returns, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Prototype Hopper-Hop SD trainer")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, choices=("offline", "online"), default="online")
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--planner_horizon", type=int, default=8)
    parser.add_argument("--planner_population", type=int, default=128)
    parser.add_argument("--planner_elites", type=int, default=16)
    parser.add_argument("--planner_iterations", type=int, default=3)
    parser.add_argument("--planner_temperature", type=float, default=0.5)
    parser.add_argument("--planner_num_prior_trajs", type=int, default=16)
    parser.add_argument("--planner_num_actor_trajs", type=int, default=8)
    parser.add_argument("--planner_terminal_value_coef", type=float, default=1.0)
    parser.add_argument("--behavior_guide_mix_start", type=float, default=0.45)
    parser.add_argument("--behavior_guide_mix_end", type=float, default=0.15)
    parser.add_argument("--behavior_guide_eval_mix", type=float, default=0.25)
    parser.add_argument("--use_tva", action="store_true")
    parser.add_argument("--tva_epochs", type=int, default=300)
    parser.add_argument("--tva_hidden", type=int, default=128)
    parser.add_argument("--tva_gamma", type=float, default=0.99)
    parser.add_argument("--tva_true_coef", type=float, default=1.0)
    parser.add_argument("--tva_shaped_coef", type=float, default=1.0)
    parser.add_argument("--tva_lr", type=float, default=1e-3)
    parser.add_argument("--tva_updates_per_episode", type=int, default=25)
    parser.add_argument("--tva_batch_size", type=int, default=256)
    parser.add_argument("--use_actor_prior", action="store_true")
    parser.add_argument("--actor_hidden", type=int, default=128)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--actor_updates_per_episode", type=int, default=25)
    parser.add_argument("--actor_batch_size", type=int, default=256)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--seed_steps", type=int, default=2500)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--seed_pretrain_updates", type=int, default=512)
    parser.add_argument("--no_plan_reward", action="store_true")
    args = parser.parse_args()

    if not args.log_path:
        args.log_path = default_log_path(args.checkpoint_path, args.resume_checkpoint)
    setup_logging(args.log_path)
    if args.log_path:
        log(f"logging to {args.log_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = suite.load("hopper", "hop", task_kwargs={"random": args.seed})
    model = DMControlHopperSDModel()
    resume_state = load_training_checkpoint(args.resume_checkpoint, map_location="cpu") if args.resume_checkpoint else None

    value_net = HopperValueNet(hidden=args.tva_hidden) if args.use_tva else None
    if value_net is not None:
        value_net.to(args.device)
        value_net.eval()
    actor_net = HopperActorNet(hidden=args.actor_hidden) if args.use_actor_prior else None
    if actor_net is not None:
        actor_net.to(args.device)
        actor_net.eval()

    if args.mode == "offline":
        log(f"Collecting {args.budget} steps of exploratory Hopper-Hop data")
        data = collect_trajectories(env, args.budget, seed=args.seed)
        log("Training structured Hopper model in offline mode")
        best_val = train_sd_model(model, data, epochs=args.epochs, device=args.device)
    else:
        log(f"Online Hopper-Hop training with replay, budget={args.budget}, seed_steps={args.seed_steps}")
        data, online_stats = online_train_sd_model(
            model,
            env,
            total_steps=args.budget,
            seed=args.seed,
            seed_steps=min(args.seed_steps, args.budget),
            updates_per_step=args.updates_per_step,
            seed_pretrain_updates=args.seed_pretrain_updates,
            device=args.device,
            planner_horizon=args.planner_horizon,
            planner_population=args.planner_population,
            planner_elites=args.planner_elites,
            planner_iterations=args.planner_iterations,
            planner_temperature=args.planner_temperature,
            planner_num_prior_trajs=args.planner_num_prior_trajs,
            planner_num_actor_trajs=args.planner_num_actor_trajs,
            planner_terminal_value_coef=args.planner_terminal_value_coef,
            use_plan_reward=not args.no_plan_reward,
            behavior_guide_mix_start=args.behavior_guide_mix_start,
            behavior_guide_mix_end=args.behavior_guide_mix_end,
            value_net=value_net,
            use_tva=args.use_tva,
            tva_gamma=args.tva_gamma,
            tva_true_coef=args.tva_true_coef,
            tva_shaped_coef=args.tva_shaped_coef,
            tva_lr=args.tva_lr,
            tva_updates_per_episode=args.tva_updates_per_episode,
            tva_batch_size=args.tva_batch_size,
            actor_net=actor_net,
            use_actor_prior=args.use_actor_prior,
            actor_lr=args.actor_lr,
            actor_updates_per_episode=args.actor_updates_per_episode,
            actor_batch_size=args.actor_batch_size,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_state=resume_state,
        )
        best_val = float("nan") if online_stats["final_loss"] is None else online_stats["final_loss"]

    metrics = evaluate_model(model, data, device=args.device)

    tva_loss = None
    if args.use_tva and args.mode == "offline":
        log("Training Hopper TVA value network")
        value_states, value_targets = build_tva_targets(
            model, data, gamma=args.tva_gamma,
            true_coef=args.tva_true_coef,
            shaped_coef=args.tva_shaped_coef,
            device=args.device,
        )
        tva_loss = train_value_net(
            value_net, value_states, value_targets,
            epochs=args.tva_epochs, device=args.device,
        )
    elif args.use_tva and args.mode == "online":
        tva_loss = online_stats.get("final_tva_loss")

    eval_env = suite.load("hopper", "hop", task_kwargs={"random": args.seed + 1000})
    random_returns = evaluate_random(eval_env, n_episodes=args.eval_episodes, seed=args.seed + 2000)
    learned_returns = evaluate_agent(
        eval_env,
        model,
        n_episodes=args.eval_episodes,
        planner_horizon=args.planner_horizon,
        planner_population=args.planner_population,
        planner_elites=args.planner_elites,
        planner_iterations=args.planner_iterations,
        planner_temperature=args.planner_temperature,
        planner_num_prior_trajs=args.planner_num_prior_trajs,
        planner_num_actor_trajs=args.planner_num_actor_trajs,
        planner_terminal_value_coef=args.planner_terminal_value_coef,
        behavior_guide_eval_mix=args.behavior_guide_eval_mix,
        device=args.device,
        use_plan_reward=not args.no_plan_reward,
        value_net=value_net,
        actor_net=actor_net,
    )

    log(f"Best validation loss: {best_val:.4f}")
    for key, value in metrics.items():
        log(f"{key}: {value:.4f}")
    if tva_loss is not None:
        log(f"tva_loss: {tva_loss:.4f}")
    if args.use_actor_prior and args.mode == "online":
        actor_loss = online_stats.get("final_actor_loss")
        if actor_loss is not None:
            log(f"actor_loss: {actor_loss:.4f}")
    log(f"random return: mean={float(random_returns.mean()):.1f} std={float(random_returns.std()):.1f}")
    log(f"learned-MPC return: mean={float(learned_returns.mean()):.1f} std={float(learned_returns.std()):.1f}")
    log(f"Learned params: {model.get_learned_params()}")


if __name__ == "__main__":
    main()
