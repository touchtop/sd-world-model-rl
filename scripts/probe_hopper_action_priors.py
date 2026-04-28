#!/usr/bin/env python
"""Probe state-conditional Hopper action priors from saved trace CSV files.

This script is intended for the current recovery-prior design work:
1. pick representative rear-lying / prone-lying states from failed eval traces;
2. roll the current SD dynamics forward under single-joint constant actions;
3. inspect which action signs reduce |pitch| or improve the current six-state recovery signals.

It does not change training behavior. It is only a repeatable analysis tool.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


ACTION_NAMES = ["waist", "hip", "knee", "ankle"]


@dataclass
class StateSample:
    label: str
    path: str
    step: int
    row: dict[str, str]


def _inv_contact(x: float) -> float:
    x = min(max(x, 0.0), 1.0 - 1e-6)
    return -math.log(max(1.0 - x, 1e-6))


def row_to_raw_obs(row: dict[str, str]) -> torch.Tensor:
    obs = [
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
    ]
    return torch.tensor(obs, dtype=torch.float32)


def row_to_qpos_qvel(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.array(
        [
            float(row["qpos_rootx"]),
            float(row["qpos_rootz"]),
            float(row["qpos_rooty"]),
            float(row["qpos_waist"]),
            float(row["qpos_hip"]),
            float(row["qpos_knee"]),
            float(row["qpos_ankle"]),
        ],
        dtype=np.float64,
    )
    qvel = np.array(
        [
            float(row["qvel_rootx"]),
            float(row["qvel_rootz"]),
            float(row["qvel_rooty"]),
            float(row["qvel_waist"]),
            float(row["qvel_hip"]),
            float(row["qvel_knee"]),
            float(row["qvel_ankle"]),
        ],
        dtype=np.float64,
    )
    return qpos, qvel


def _select_samples(trace_dir: str) -> list[StateSample]:
    rear_best = None
    prone_best = None
    rear_score = -1.0
    prone_score = -1.0

    for path in sorted(glob.glob(os.path.join(trace_dir, "episode_*.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                z_min = float(row.get("z_min", 1.0) or 1.0)
                foot = float(row.get("full_foot_support", 0.0) or 0.0)
                pitch = float(row.get("pitch", 0.0) or 0.0)
                step = int(float(row.get("step", 0.0) or 0.0))

                if z_min <= DMControlHopperSDModel.STATE_GROUND_EPS and foot > 0.8 and pitch < -1.5:
                    score = foot - 0.05 * abs(float(row.get("vz", 0.0) or 0.0))
                    if score > rear_score:
                        rear_score = score
                        rear_best = StateSample("rear_lying", path, step, row)

                if z_min <= DMControlHopperSDModel.STATE_GROUND_EPS and pitch > 1.5 and foot < 0.2:
                    score = 1.0 - 0.05 * abs(float(row.get("vz", 0.0) or 0.0))
                    if score > prone_score:
                        prone_score = score
                        prone_best = StateSample("prone_lying", path, step, row)

    samples = [s for s in [rear_best, prone_best] if s is not None]
    if not samples:
        raise RuntimeError(f"No representative rear/prone states found in {trace_dir}")
    return samples


def _fmt_feat(feat: dict[str, torch.Tensor]) -> str:
    return (
        f"pitch={float(feat['pitch']):+.3f} "
        f"|pitch|={abs(float(feat['pitch'])):.3f} "
        f"rear={float(feat['grounded_rear']):.3f} "
        f"prone={float(feat['grounded_prone']):.3f} "
        f"upright={float(feat['grounded_upright']):.3f} "
        f"foot={float(feat['full_foot_support']):.3f} "
        f"vz={float(feat['vz']):+.3f} "
        f"z_min={float(feat['z_min']):.3f}"
    )


def _score_transition(prev: dict[str, torch.Tensor], nxt: dict[str, torch.Tensor]) -> float:
    prev_abs_pitch = abs(float(prev["pitch"]))
    next_abs_pitch = abs(float(nxt["pitch"]))
    pitch_gain = prev_abs_pitch - next_abs_pitch
    upright_gain = float(nxt["grounded_upright"]) - float(prev["grounded_upright"])
    foot_gain = float(nxt["full_foot_support"]) - float(prev["full_foot_support"])
    zmin_gain = float(prev["z_min"]) - float(nxt["z_min"])
    return 1.5 * pitch_gain + 0.8 * foot_gain + 0.3 * upright_gain + 0.1 * zmin_gain


def _roll_constant_action(
    model: DMControlHopperSDModel,
    state: torch.Tensor,
    action: torch.Tensor,
    steps: int,
) -> list[dict[str, torch.Tensor]]:
    feats = []
    cur = state.clone()
    for _ in range(steps):
        cur = model(cur, action)
        feats.append(model._state_reward_features(cur))  # analysis helper only
    return feats


def _iter_probe_actions(action_dim: int, amplitudes: Iterable[float]) -> Iterable[tuple[str, torch.Tensor]]:
    for amp in amplitudes:
        for dim in range(action_dim):
            for sign in (-1.0, 1.0):
                action = torch.zeros(1, action_dim, dtype=torch.float32)
                action[0, dim] = sign * amp
                name = f"{ACTION_NAMES[dim]}={'+' if sign > 0 else '-'}{amp:.2f}"
                yield name, action


def _make_probe_env():
    sys.path.append(os.path.join(REPO_ROOT, "external/tdmpc2/tdmpc2"))
    from envs.tasks import hopper  # noqa: F401
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    env = suite.load("hopper", "hop", task_kwargs={"random": 0}, visualize_reward=False)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def _roll_env_constant_action(
    env,
    model: DMControlHopperSDModel,
    sample: StateSample,
    action: torch.Tensor,
    steps: int,
) -> list[dict[str, torch.Tensor]]:
    qpos, qvel = row_to_qpos_qvel(sample.row)
    physics = env._env.physics
    env.reset()
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
    physics.forward()

    feats = []
    action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float64)
    for _ in range(steps):
        for _ in range(DMControlHopperSDModel.ENV_ACTION_REPEAT):
            ts = env.step(action_np)
        raw_obs = torch.from_numpy(DMControlHopperSDModel.flatten_obs(ts.observation)).float().unsqueeze(0)
        state = model.encode_observation(raw_obs)
        feats.append(model._state_reward_features(state))
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, help="Directory containing episode_XX.csv trace files.")
    parser.add_argument("--backend", choices=["sd", "env"], default="sd")
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70])
    parser.add_argument("--topk", type=int, default=6)
    args = parser.parse_args()

    model = DMControlHopperSDModel().eval()
    samples = _select_samples(args.trace_dir)
    probe_env = _make_probe_env() if args.backend == "env" else None

    for sample in samples:
        raw_obs = row_to_raw_obs(sample.row).unsqueeze(0)
        state = model.encode_observation(raw_obs)
        prev_feat = model._state_reward_features(state)

        print(f"== {sample.label} ==")
        print(f"backend={args.backend}")
        print(f"source={sample.path}")
        print(f"step={sample.step}")
        print(_fmt_feat(prev_feat))

        scored = []
        for name, action in _iter_probe_actions(model.ACTION_DIM, args.amplitudes):
            if args.backend == "env":
                rollout = _roll_env_constant_action(probe_env, model, sample, action, args.rollout_steps)
            else:
                rollout = _roll_constant_action(model, state, action, args.rollout_steps)
            next_feat = rollout[-1]
            score = _score_transition(prev_feat, next_feat)
            scored.append((score, name, next_feat))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, name, feat in scored[: args.topk]:
            print(f"  {name:<14} score={score:+.3f} {_fmt_feat(feat)}")
        print()


if __name__ == "__main__":
    main()
