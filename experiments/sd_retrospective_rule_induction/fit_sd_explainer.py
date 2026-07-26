#!/usr/bin/env python3
"""Fit an offline SD explainer from selected Hopper trace transitions."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    DEFAULT_EXPLAINER,
    DEFAULT_EXPLAINER_SUMMARY,
    DEFAULT_SELECTION,
    load_json,
    read_csv,
    row_to_action,
    row_to_raw_obs,
    write_json,
)
from scripts.fit_hopper_sd_supervised import (  # noqa: E402
    TransitionDataset,
    fit_model,
    split_indices,
)
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


@dataclass
class TraceTransitionPack:
    raw_obs: list[list[float]]
    actions: list[list[float]]
    next_raw_obs: list[list[float]]
    rewards: list[float]
    dones: list[float]
    episode_ids: list[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(selection: dict[str, Any], max_transitions: int, seed: int) -> TransitionDataset:
    pack = TraceTransitionPack([], [], [], [], [], [])
    selected = selection["selected_episodes"]
    rng = random.Random(seed)
    if max_transitions > 0:
        selected = list(selected)
        rng.shuffle(selected)

    episode_id = 0
    for ep in selected:
        rows = read_csv(Path(ep["trace_csv"]))
        if len(rows) < 2:
            continue
        for i in range(len(rows) - 1):
            row = rows[i]
            nxt = rows[i + 1]
            pack.raw_obs.append(row_to_raw_obs(row))
            pack.actions.append(row_to_action(row))
            pack.next_raw_obs.append(row_to_raw_obs(nxt))
            pack.rewards.append(float(nxt.get("reward", 0.0) or 0.0))
            pack.dones.append(float(nxt.get("done", 0.0) or 0.0))
            pack.episode_ids.append(episode_id)
            if max_transitions > 0 and len(pack.raw_obs) >= max_transitions:
                break
        episode_id += 1
        if max_transitions > 0 and len(pack.raw_obs) >= max_transitions:
            break

    if len(pack.raw_obs) < 100:
        raise RuntimeError(f"too few transitions for explainer fitting: {len(pack.raw_obs)}")

    return TransitionDataset(
        raw_obs=np.asarray(pack.raw_obs, dtype=np.float32),
        actions=np.asarray(pack.actions, dtype=np.float32),
        next_raw_obs=np.asarray(pack.next_raw_obs, dtype=np.float32),
        rewards=np.asarray(pack.rewards, dtype=np.float32),
        dones=np.asarray(pack.dones, dtype=np.float32),
        episode_ids=np.asarray(pack.episode_ids, dtype=np.int32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPLAINER)
    parser.add_argument("--summary", type=Path, default=DEFAULT_EXPLAINER_SUMMARY)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--max-transitions", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-transition-residual", action="store_true")
    parser.add_argument("--transition-residual-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    selection = load_json(args.selection)
    dataset = build_dataset(selection, args.max_transitions, args.seed)
    train_idx, val_idx = split_indices(dataset.episode_ids, val_fraction=0.1, seed=args.seed)

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
    init_loaded = 0
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"init checkpoint not found: {args.init_checkpoint}")
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        src_state = payload.get("model_state", payload)
        target_state = model.state_dict()
        compatible = {
            k: v for k, v in src_state.items()
            if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
        }
        model.load_state_dict(compatible, strict=False)
        init_loaded = len(compatible)

    result = fit_model(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rollout_horizon=3,
        rollout_weight=0.10,
        rollout_batch_size=64,
        short_rollout_horizons=[3, 5],
        short_rollout_weight=0.05,
        touch_loss_weight=0.10,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": result["model_state"],
            "args": vars(args),
            "metrics": result["metrics"],
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "num_transitions": int(len(dataset.raw_obs)),
            "model_config": {
                "use_transition_residual": bool(args.use_transition_residual),
                "transition_residual_scale": float(args.transition_residual_scale),
            },
        },
        args.output,
    )
    write_json(
        args.summary,
        {
            "selection": str(args.selection),
            "checkpoint": str(args.output),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "init_loaded_tensors": init_loaded,
            "seed": args.seed,
            "num_transitions": int(len(dataset.raw_obs)),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "epochs": args.epochs,
            "metrics": result["metrics"],
            "history_tail": result["history"][-5:],
            "model_config": {
                "use_transition_residual": bool(args.use_transition_residual),
                "transition_residual_scale": float(args.transition_residual_scale),
            },
        },
    )
    print(f"wrote explainer: {args.output}")
    print(f"wrote summary: {args.summary}")
    if args.init_checkpoint is not None:
        print(f"initialized from: {args.init_checkpoint} tensors={init_loaded}")
    print(f"transitions: {len(dataset.raw_obs)} train={len(train_idx)} val={len(val_idx)}")


if __name__ == "__main__":
    main()
