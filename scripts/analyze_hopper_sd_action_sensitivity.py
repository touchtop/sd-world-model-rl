#!/usr/bin/env python3
"""Compare one-step action responses between DMControl Hopper and the SD model.

The goal is not to score policy performance. The goal is to test whether,
for the same state and action, the SD world-model produces a qualitatively
and quantitatively similar one-step response in the quantities that matter
for TD-MPC2 planning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The current machine can step dm_control under glfw despite the missing
# DISPLAY warning. This keeps the script consistent with the training setup.
os.environ.setdefault("MUJOCO_GL", "glfw")

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


ACTION_NAMES = ["waist", "hip", "knee", "ankle"]
SUMMARY_KEYS = (
    "height",
    "vx",
    "vz",
    "pitch",
    "pitch_rate",
    "support_contact",
    "full_foot_support",
    "body_contact_z",
    "z_min",
)


@dataclass
class TraceState:
    source_csv: str
    episode: int
    step: int
    raw_obs: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray


def _inv_contact(x: float) -> float:
    x = min(max(x, 0.0), 1.0 - 1e-6)
    return -math.log(max(1.0 - x, 1e-6))


def _row_to_trace_state(path: str, row: dict[str, str]) -> TraceState:
    raw_obs = np.array(
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
    return TraceState(
        source_csv=path,
        episode=int(float(row.get("episode", 0) or 0)),
        step=int(float(row["step"])),
        raw_obs=raw_obs,
        qpos=qpos,
        qvel=qvel,
    )


def load_trace_states(trace_paths: Iterable[str], samples_per_trace: int) -> List[TraceState]:
    states: List[TraceState] = []
    for trace_path in trace_paths:
        with open(trace_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        keep = min(samples_per_trace, len(rows))
        if keep <= 0:
            continue
        idxs = np.linspace(0, len(rows) - 1, num=keep, dtype=int)
        for idx in idxs.tolist():
            states.append(_row_to_trace_state(trace_path, rows[idx]))
    if not states:
        raise RuntimeError("No trace states loaded from the provided CSV files.")
    return states


def iter_trace_paths(trace_csvs: List[str], trace_dir: str | None) -> List[str]:
    out = list(trace_csvs)
    if trace_dir:
        out.extend(str(p) for p in sorted(Path(trace_dir).glob("episode_*.csv")))
    unique = []
    seen = set()
    for path in out:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise RuntimeError("At least one --trace-csv or a non-empty --trace-dir is required.")
    return unique


def build_action_bank(amplitudes: Iterable[float]) -> List[tuple[str, np.ndarray]]:
    bank: List[tuple[str, np.ndarray]] = []
    for amp in amplitudes:
        for dim, name in enumerate(ACTION_NAMES):
            for sign, sign_name in ((-1.0, "neg"), (1.0, "pos")):
                action = np.zeros((DMControlHopperSDModel.ACTION_DIM,), dtype=np.float32)
                action[dim] = sign * float(amp)
                bank.append((f"{name}_{sign_name}_{amp:.2f}", action))
    return bank


def make_env():
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    env = suite.load("hopper", "hop", task_kwargs={"random": 0}, visualize_reward=False)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def state_features(model: DMControlHopperSDModel, state: torch.Tensor) -> Dict[str, float]:
    feat = model._state_reward_features(state)
    dominant = int(model._dominant_state_ids(feat)[0].item())
    out = {
        key: float(feat[key][0].item())
        for key in SUMMARY_KEYS
    }
    out["dominant_state_id"] = dominant
    return out


def step_env_once(env, model: DMControlHopperSDModel, sample: TraceState, action: np.ndarray) -> torch.Tensor:
    _ = env.reset()
    physics = env.physics
    physics.data.qpos[:] = sample.qpos
    physics.data.qvel[:] = sample.qvel
    physics.forward()
    timestep = None
    for _ in range(DMControlHopperSDModel.ENV_ACTION_REPEAT):
        timestep = env.step(action.astype(np.float64))
    assert timestep is not None
    next_raw = torch.from_numpy(DMControlHopperSDModel.flatten_obs(timestep.observation)).float().unsqueeze(0)
    return model.encode_observation(next_raw)


def step_sd_once(model: DMControlHopperSDModel, sample: TraceState, action: np.ndarray) -> torch.Tensor:
    raw = torch.from_numpy(sample.raw_obs).float().unsqueeze(0)
    state = model.encode_observation(raw)
    action_t = torch.from_numpy(action).float().unsqueeze(0)
    return model(state, action_t)


def angle_diff(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def signed_delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in SUMMARY_KEYS:
        if key == "pitch":
            out[key] = angle_diff(after[key], before[key])
        else:
            out[key] = after[key] - before[key]
    return out


def dominant_match(a: Dict[str, float], b: Dict[str, float]) -> float:
    return 1.0 if int(a["dominant_state_id"]) == int(b["dominant_state_id"]) else 0.0


def direction_match(a: float, b: float, tol: float = 1e-4) -> float:
    if abs(a) <= tol and abs(b) <= tol:
        return 1.0
    if abs(a) <= tol or abs(b) <= tol:
        return 0.0
    return 1.0 if math.copysign(1.0, a) == math.copysign(1.0, b) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Hopper env vs SD one-step action response.")
    parser.add_argument("--trace-csv", action="append", default=[], help="Path to an episode_XX.csv trace file.")
    parser.add_argument("--trace-dir", default=None, help="Directory containing episode_XX.csv files.")
    parser.add_argument("--samples-per-trace", type=int, default=4)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--out-dir", default="results/hopper_sd_action_sensitivity/default_run")
    args = parser.parse_args()

    model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
    ).eval()
    env = make_env()
    trace_paths = iter_trace_paths(args.trace_csv, args.trace_dir)
    states = load_trace_states(trace_paths, args.samples_per_trace)
    actions = build_action_bank(args.amplitudes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_rows: List[Dict[str, object]] = []

    for sample_idx, sample in enumerate(states):
        base_state = model.encode_observation(torch.from_numpy(sample.raw_obs).float().unsqueeze(0))
        base_feat = state_features(model, base_state)
        for action_name, action in actions:
            env_next = step_env_once(env, model, sample, action)
            sd_next = step_sd_once(model, sample, action)
            env_feat = state_features(model, env_next)
            sd_feat = state_features(model, sd_next)
            env_delta = signed_delta(env_feat, base_feat)
            sd_delta = signed_delta(sd_feat, base_feat)
            row: Dict[str, object] = {
                "sample_idx": sample_idx,
                "source_csv": sample.source_csv,
                "episode": sample.episode,
                "step": sample.step,
                "action_name": action_name,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_3": float(action[3]),
                "base_state_id": int(base_feat["dominant_state_id"]),
                "env_state_id": int(env_feat["dominant_state_id"]),
                "sd_state_id": int(sd_feat["dominant_state_id"]),
                "dominant_state_match": dominant_match(env_feat, sd_feat),
            }
            for key in SUMMARY_KEYS:
                row[f"base_{key}"] = base_feat[key]
                row[f"env_{key}"] = env_feat[key]
                row[f"sd_{key}"] = sd_feat[key]
                row[f"env_delta_{key}"] = env_delta[key]
                row[f"sd_delta_{key}"] = sd_delta[key]
                row[f"abs_err_{key}"] = abs(env_delta[key] - sd_delta[key])
                row[f"dir_match_{key}"] = direction_match(env_delta[key], sd_delta[key])
            detail_rows.append(row)

    detail_csv = out_dir / "action_sensitivity_details.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary: Dict[str, object] = {
        "num_states": len(states),
        "num_actions": len(actions),
        "detail_csv": str(detail_csv),
        "aggregate": {},
    }
    for key in SUMMARY_KEYS:
        abs_err = np.array([float(r[f"abs_err_{key}"]) for r in detail_rows], dtype=np.float64)
        dir_match_arr = np.array([float(r[f"dir_match_{key}"]) for r in detail_rows], dtype=np.float64)
        summary["aggregate"][key] = {
            "abs_err_mean": float(abs_err.mean()),
            "abs_err_std": float(abs_err.std()),
            "dir_match_mean": float(dir_match_arr.mean()),
        }
    dominant_match_arr = np.array([float(r["dominant_state_match"]) for r in detail_rows], dtype=np.float64)
    summary["dominant_state_match_mean"] = float(dominant_match_arr.mean())
    summary["trace_csvs"] = trace_paths
    summary["amplitudes"] = [float(x) for x in args.amplitudes]
    summary["samples_per_trace"] = int(args.samples_per_trace)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
