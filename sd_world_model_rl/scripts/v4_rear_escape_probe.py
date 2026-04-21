"""Probe whether explicit SD rear-escape dynamics can rotate a rear-trap state forward.

This script is intentionally narrow: it replays a structured state extracted from
closed-loop diagnosis CSV, applies a fixed action for several model steps, and
reports whether pitch / rootz move toward the recovery corridor. The goal is to
separate "reward not exploited" from "explicit dynamics cannot express escape".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_csv", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--action", type=float, nargs=4, default=[1.0, -1.0, 1.0, 1.0])
    parser.add_argument("--rollout_steps", type=int, default=12)
    return parser.parse_args()


def load_model(checkpoint: Path | None) -> DMControlHopperSDModel:
    model = DMControlHopperSDModel()
    if checkpoint is None:
        model.eval()
        return model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    dyn_state = {k.removeprefix("_dynamics."): v for k, v in state.items() if k.startswith("_dynamics.")}
    missing, unexpected = model.load_state_dict(dyn_state, strict=False)
    if missing:
        raise RuntimeError(f"missing dynamics params in checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected dynamics params in checkpoint: {unexpected}")
    model.eval()
    return model


def load_state_row(episode_csv: Path, step: int) -> torch.Tensor:
    with episode_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if step < 0 or step >= len(rows):
        raise IndexError(f"step {step} out of range for {episode_csv} with {len(rows)} rows")
    row = rows[step]
    height = float(row["height"])
    compression = max(0.0, DMControlHopperSDModel.TARGET_STAND_HEIGHT - height)
    values = [
        float(row["vx"]),
        float(row["vz"]),
        float(row["pitch"]),
        float(row["qvel_rooty"]),
        compression,
        -float(row["vz"]),
        float(row["support"]),
        float(row["rootz"]),
        float(row["capture_memory"]),
    ]
    return torch.tensor([values], dtype=torch.float32)


def tensor_to_dict(state: torch.Tensor) -> dict[str, float]:
    s = state[0]
    return {
        "vx": float(s[DMControlHopperSDModel.VX_IDX].item()),
        "vz": float(s[DMControlHopperSDModel.VZ_IDX].item()),
        "pitch": float(s[DMControlHopperSDModel.PITCH_IDX].item()),
        "pitch_rate": float(s[DMControlHopperSDModel.PITCH_RATE_IDX].item()),
        "compression": float(s[DMControlHopperSDModel.COMP_IDX].item()),
        "compression_rate": float(s[DMControlHopperSDModel.COMP_RATE_IDX].item()),
        "support": float(s[DMControlHopperSDModel.SUPPORT_IDX].item()),
        "rootz": float(s[DMControlHopperSDModel.ROOTZ_IDX].item()),
        "capture_memory": float(s[DMControlHopperSDModel.CAPTURE_IDX].item()),
    }


def summarize_rollout(model: DMControlHopperSDModel, state: torch.Tensor, action: torch.Tensor, rollout_steps: int) -> dict[str, object]:
    history: list[dict[str, float]] = []
    current = state
    for i in range(rollout_steps + 1):
        row = tensor_to_dict(current)
        row["step"] = i
        row["rear_phase"] = float(model._rear_recovery_phase_from_state(current[:, model.PITCH_IDX], current[:, model.ROOTZ_IDX])[0].item())
        row["precapture_phase"] = float(
            model._precapture_phase_from_state(
                current[:, model.PITCH_IDX],
                current[:, model.SUPPORT_IDX],
                current[:, model.COMP_IDX],
                current[:, model.ROOTZ_IDX],
                current[:, model.CAPTURE_IDX],
            )[0].item()
        )
        history.append(row)
        if i < rollout_steps:
            current = model.forward(current, action)

    pitch_series = [row["pitch"] for row in history]
    rootz_series = [row["rootz"] for row in history]
    return {
        "initial": history[0],
        "final": history[-1],
        "max_pitch": max(pitch_series),
        "max_rootz": max(rootz_series),
        "pitch_crossed_minus_0_3": any(p > -0.3 for p in pitch_series),
        "pitch_crossed_0": any(p > 0.0 for p in pitch_series),
        "rootz_crossed_minus_0_7": any(z > -0.7 for z in rootz_series),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    model = load_model(args.checkpoint)
    state = load_state_row(args.episode_csv, args.step)
    action = torch.tensor([args.action], dtype=torch.float32)
    report = summarize_rollout(model, state, action, args.rollout_steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
