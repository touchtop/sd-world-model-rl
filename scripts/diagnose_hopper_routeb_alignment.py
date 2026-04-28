#!/usr/bin/env python3
"""On-policy one-step alignment diagnostic for Hopper Route B checkpoints.

The diagnostic follows the current Route B policy in the real environment and,
for selected (intent, subphase) events, compares:

1. the real one-step state delta after executing the chosen action, and
2. the world model's one-step prediction for the same state-action pair.

This is intended to test whether training changes reduce model ranking bias on
the key structured quantities that currently drive planner preference:
`vx`, `vz`, `support`, and `clearance`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.run_hopper_mcts import (
    INTENT_NAMES,
    DMControlHopperSDModel,
    flatten_obs,
    infer_phase_context,
    intent_to_action,
    make_env,
    mcts_search,
    obs_to_struct,
)


TRACKED_KEYS: List[Tuple[str, int]] = [
    ("vx", DMControlHopperSDModel.VX_IDX),
    ("vz", DMControlHopperSDModel.VZ_IDX),
    ("support", DMControlHopperSDModel.SUPPORT_IDX),
    ("clearance", DMControlHopperSDModel.CLEARANCE_IDX),
]

TARGET_EVENTS = [
    ("launch", "stance_release"),
    ("drive", "stance_release"),
    ("absorb", "landing_impact"),
]


def load_model(checkpoint: str, device: str) -> Tuple[DMControlHopperSDModel, Dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    args = payload.get("args", {})
    model = DMControlHopperSDModel(dt=0.02, hidden=256, hop_speed=2.0).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, args


def format_metrics(delta: np.ndarray) -> str:
    parts = []
    for name, idx in TRACKED_KEYS:
        parts.append(f"d{name}={delta[idx]:+.4f}")
    return ", ".join(parts)


def summarize(records: List[Dict]) -> Dict[str, np.ndarray]:
    actual = np.stack([item["actual_delta"] for item in records], axis=0)
    pred = np.stack([item["pred_delta"] for item in records], axis=0)
    return {
        "count": np.array([len(records)], dtype=np.int32),
        "actual_mean": actual.mean(axis=0),
        "pred_mean": pred.mean(axis=0),
        "gap_mean": np.abs(pred - actual).mean(axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Hopper Route B on-policy one-step alignment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--rollout_depth", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--limit_per_event", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, ckpt_args = load_model(args.checkpoint, args.device)
    env = make_env(args.seed)

    event_store: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

    print(f"checkpoint={args.checkpoint}")
    print(f"device={args.device}")
    print(f"episodes={args.episodes}, max_steps={args.max_steps}")
    print(f"num_simulations={args.num_simulations}, rollout_depth={args.rollout_depth}, gamma={args.gamma}")
    if ckpt_args:
        print(f"checkpoint_args={ckpt_args}")

    for ep in range(args.episodes):
        ts = env.reset()
        obs_flat = flatten_obs(ts.observation)
        struct_state = obs_to_struct(model, obs_flat)
        prev_intent = -1

        for step in range(args.max_steps):
            best_intent, info = mcts_search(
                struct_state,
                model,
                num_simulations=args.num_simulations,
                rollout_depth=args.rollout_depth,
                gamma=args.gamma,
                prev_intent=prev_intent,
            )
            intent_name = INTENT_NAMES[best_intent]
            _, subphase = infer_phase_context(struct_state)
            action = intent_to_action(best_intent).to(args.device)

            with torch.no_grad():
                pred_next = model(struct_state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

            ts = env.step(action.detach().cpu().numpy())
            next_obs_flat = flatten_obs(ts.observation)
            next_struct = obs_to_struct(model, next_obs_flat)

            event_key = (intent_name, subphase)
            if event_key in TARGET_EVENTS and len(event_store[event_key]) < args.limit_per_event:
                event_store[event_key].append(
                    {
                        "episode": ep,
                        "step": step,
                        "root_phase": info.get("root_phase"),
                        "root_subphase": info.get("root_subphase"),
                        "actual_delta": (next_struct - struct_state).detach().cpu().numpy(),
                        "pred_delta": (pred_next - struct_state).detach().cpu().numpy(),
                    }
                )

            struct_state = next_struct
            prev_intent = best_intent

    print("")
    print("=== Alignment Summary ===")
    for intent_name, subphase in TARGET_EVENTS:
        records = event_store.get((intent_name, subphase), [])
        print(f"\n[{intent_name} @ {subphase}] count={len(records)}")
        if not records:
            print("  no samples collected")
            continue
        summary = summarize(records)
        actual_mean = summary["actual_mean"]
        pred_mean = summary["pred_mean"]
        gap_mean = summary["gap_mean"]
        print(f"  actual_mean: {format_metrics(actual_mean)}")
        print(f"  pred_mean:   {format_metrics(pred_mean)}")
        print(f"  mae_mean:    {format_metrics(gap_mean)}")
        first = records[0]
        print(
            "  first_sample:"
            f" ep={first['episode']} step={first['step']}"
            f" root_phase={first['root_phase']}"
            f" root_subphase={first['root_subphase']}"
        )


if __name__ == "__main__":
    main()
