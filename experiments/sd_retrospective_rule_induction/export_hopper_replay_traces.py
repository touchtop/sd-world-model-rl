#!/usr/bin/env python3
"""Export Hopper replay episodes from a full TD-MPC2 checkpoint.

The retrospective rule loop needs 5k-cycle trajectories even when formal eval
is only run every 25k. This script converts replay-buffer episodes from
`final.full.pt` into the same CSV shape consumed by the retrospective
attribution tools. The resulting files are explicitly stored under
`retrospective_train_traces/` to distinguish them from official eval traces.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch


FIELDNAMES = [
    "episode",
    "step",
    "reward",
    "done",
    "qpos_rootx",
    "qpos_rootz",
    "qpos_rooty",
    "qpos_waist",
    "qpos_hip",
    "qpos_knee",
    "qpos_ankle",
    "qvel_rootx",
    "qvel_rootz",
    "qvel_rooty",
    "qvel_waist",
    "qvel_hip",
    "qvel_knee",
    "qvel_ankle",
    "rootx",
    "rootz",
    "vx",
    "vz",
    "pitch",
    "pitch_rate",
    "activation",
    "touch_toe",
    "touch_heel",
    "full_foot_support",
    "foot_grounded",
    "grounded_rear",
    "grounded_prone",
    "grounded_hop",
    "grounded_upright",
    "grounded_prepared",
    "falling_air",
    "rising_air",
    "backward_speed_gt_05",
    "rotation_penalty",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
]

SUMMARY_FIELDS = [
    "episode_idx",
    "episode_reward",
    "episode_success",
    "length",
    "trace_csv",
    "final_rootx",
    "final_pitch",
    "max_rootx",
    "max_torso_z",
    "max_vx",
    "max_vz",
    "max_full_foot_support",
    "falling_air_frac",
    "grounded_rear_frac",
    "grounded_prone_frac",
    "grounded_hop_frac",
    "grounded_upright_frac",
    "grounded_prepared_frac",
    "rising_air_frac",
    "full_foot_support_frac",
    "backward_speed_frac",
    "rotation_penalty_frac",
    "source",
]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def contact_from_obs(value: float) -> float:
    value = max(0.0, value)
    return 1.0 - math.exp(-value)


def phase_flags(rootz: float, pitch: float, vx: float, vz: float, full_foot: float) -> dict[str, float]:
    prone = 1.0 if abs(pitch) > 1.7 and rootz < 0.35 else 0.0
    rear = 1.0 if pitch < -1.2 and rootz < 0.55 else 0.0
    hop = 1.0 if rootz >= 0.55 and abs(pitch) <= 1.10 and vx >= 0.75 else 0.0
    upright = 1.0 if abs(pitch) < 0.55 and rootz > 0.65 else 0.0
    prepared = 1.0 if upright and full_foot > 0.35 else 0.0
    falling = 1.0 if rootz < 0.45 and full_foot < 0.10 and vz < -0.05 else 0.0
    rising = 1.0 if vz > 0.2 and rootz > 0.45 and vx >= 0.0 else 0.0
    return {
        "grounded_rear": rear,
        "grounded_prone": prone,
        "grounded_hop": hop,
        "grounded_upright": upright,
        "grounded_prepared": prepared,
        "falling_air": falling,
        "rising_air": rising,
    }


def row_from_episode(ep: Any, episode_idx: int, step_idx: int, rootx: float, dt: float) -> dict[str, float]:
    obs = ep["obs"][step_idx].detach().cpu().flatten().tolist()
    action_idx = min(step_idx + 1, len(ep["action"]) - 1)
    action = ep["action"][action_idx].detach().cpu().flatten().tolist()
    reward_idx = min(step_idx + 1, len(ep["reward"]) - 1)
    reward = finite_float(ep["reward"][reward_idx])
    done = finite_float(ep["terminated"][reward_idx]) if "terminated" in ep.keys() else 0.0

    rootz = finite_float(obs[0])
    pitch = finite_float(obs[1])
    vx = finite_float(obs[6])
    vz = finite_float(obs[7])
    pitch_rate = finite_float(obs[8])
    touch_toe = contact_from_obs(finite_float(obs[13])) if len(obs) > 13 else 0.0
    touch_heel = contact_from_obs(finite_float(obs[14])) if len(obs) > 14 else 0.0
    full_foot = min(touch_toe, touch_heel)
    foot_grounded = max(touch_toe, touch_heel)
    phases = phase_flags(rootz, pitch, vx, vz, full_foot)
    rotation_penalty = 1.0 if abs(pitch_rate) >= 3.0 else 0.0

    out = {
        "episode": float(episode_idx),
        "step": float(step_idx),
        "reward": reward,
        "done": done,
        "qpos_rootx": rootx,
        "qpos_rootz": rootz,
        "qpos_rooty": pitch,
        "qpos_waist": finite_float(obs[2]),
        "qpos_hip": finite_float(obs[3]),
        "qpos_knee": finite_float(obs[4]),
        "qpos_ankle": finite_float(obs[5]),
        "qvel_rootx": vx,
        "qvel_rootz": vz,
        "qvel_rooty": pitch_rate,
        "qvel_waist": finite_float(obs[9]),
        "qvel_hip": finite_float(obs[10]),
        "qvel_knee": finite_float(obs[11]),
        "qvel_ankle": finite_float(obs[12]),
        "rootx": rootx,
        "rootz": rootz,
        "vx": vx,
        "vz": vz,
        "pitch": pitch,
        "pitch_rate": pitch_rate,
        "activation": sum(abs(finite_float(x)) for x in action) / max(len(action), 1),
        "touch_toe": touch_toe,
        "touch_heel": touch_heel,
        "full_foot_support": full_foot,
        "foot_grounded": foot_grounded,
        "backward_speed_gt_05": 1.0 if vx < -0.5 else 0.0,
        "rotation_penalty": rotation_penalty,
    }
    out.update(phases)
    for i in range(4):
        out[f"action_{i}"] = finite_float(action[i]) if i < len(action) else 0.0
    return out


def summarize_rows(rows: list[dict[str, float]], trace_csv: Path, episode_idx: int) -> dict[str, Any]:
    n = max(len(rows), 1)
    reward = sum(finite_float(row["reward"]) for row in rows)
    return {
        "episode_idx": episode_idx,
        "episode_reward": reward,
        "episode_success": 0.0,
        "length": len(rows),
        "trace_csv": str(trace_csv),
        "final_rootx": rows[-1]["rootx"] if rows else 0.0,
        "final_pitch": rows[-1]["pitch"] if rows else 0.0,
        "max_rootx": max((row["rootx"] for row in rows), default=0.0),
        "max_torso_z": max((row["rootz"] for row in rows), default=0.0),
        "max_vx": max((row["vx"] for row in rows), default=0.0),
        "max_vz": max((row["vz"] for row in rows), default=0.0),
        "max_full_foot_support": max((row["full_foot_support"] for row in rows), default=0.0),
        "falling_air_frac": sum(row["falling_air"] for row in rows) / n,
        "grounded_rear_frac": sum(row["grounded_rear"] for row in rows) / n,
        "grounded_prone_frac": sum(row["grounded_prone"] for row in rows) / n,
        "grounded_hop_frac": sum(row["grounded_hop"] for row in rows) / n,
        "grounded_upright_frac": sum(row["grounded_upright"] for row in rows) / n,
        "grounded_prepared_frac": sum(row["grounded_prepared"] for row in rows) / n,
        "rising_air_frac": sum(row["rising_air"] for row in rows) / n,
        "full_foot_support_frac": sum(1.0 for row in rows if row["full_foot_support"] > 0.35) / n,
        "backward_speed_frac": sum(row["backward_speed_gt_05"] for row in rows) / n,
        "rotation_penalty_frac": sum(row["rotation_penalty"] for row in rows) / n,
        "source": "train_replay",
    }


def export_checkpoint(checkpoint: Path, output_dir: Path, dt: float) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    episodes = state.get("buffer_episodes", [])
    step = int(state.get("step", 0))
    step_dir = output_dir / f"step_{step:07d}"
    episode_dir = step_dir / "episode_csv"
    episode_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for episode_idx, ep in enumerate(episodes):
        length = len(ep["obs"])
        if length < 2:
            continue
        rows: list[dict[str, float]] = []
        rootx = 0.0
        for step_idx in range(length - 1):
            if rows:
                rootx = rows[-1]["rootx"] + rows[-1]["vx"] * dt
            row = row_from_episode(ep, episode_idx, step_idx, rootx, dt)
            rows.append(row)
        trace_csv = episode_dir / f"episode_{episode_idx:02d}.csv"
        with trace_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        summaries.append(summarize_rows(rows, trace_csv, episode_idx))

    summary_csv = step_dir / "eval_episode_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    return {"step": step, "num_episodes": len(summaries), "summary_csv": str(summary_csv)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_checkpoint(args.checkpoint, args.output_dir, args.dt)
    payload["checkpoint"] = str(args.checkpoint)
    payload["output_dir"] = str(args.output_dir)
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        import json

        with args.metadata.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    print(f"wrote replay traces: {payload['summary_csv']}")
    print(f"episodes: {payload['num_episodes']}")


if __name__ == "__main__":
    main()
