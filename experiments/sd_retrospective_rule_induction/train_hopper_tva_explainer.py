#!/usr/bin/env python3
"""Train the audit-only Hopper TVA value net for one retrospective cycle.

This script intentionally reuses the TVA implementation from
``scripts/run_dmcontrol_hopper.py``.  The retrospective rule pipeline only
wraps that code so the value net can be trained beside each fitted SD
explainer without changing the online planner.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import DEFAULT_SELECTION, load_json, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


def load_hopper_tva_module():
    module_path = REPO_ROOT / "scripts" / "run_dmcontrol_hopper.py"
    spec = importlib.util.spec_from_file_location("sdmcts_hopper_tva_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load TVA source module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class TvaTransitionPack:
    raw_obs: np.ndarray
    actions: np.ndarray
    next_raw_obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def build_transition_pack(selection: dict[str, Any], max_transitions: int, seed: int) -> TvaTransitionPack:
    raw_obs: list[list[float]] = []
    actions: list[list[float]] = []
    next_raw_obs: list[list[float]] = []
    rewards: list[float] = []
    dones: list[float] = []
    selected = list(selection["selected_episodes"])
    rng = random.Random(seed)
    if max_transitions > 0:
        rng.shuffle(selected)
    for ep in selected:
        rows = read_csv(Path(ep["trace_csv"]))
        if len(rows) < 2:
            continue
        for i in range(len(rows) - 1):
            row = rows[i]
            nxt = rows[i + 1]
            raw_obs.append(row_to_raw_obs(row))
            actions.append(row_to_action(row))
            next_raw_obs.append(row_to_raw_obs(nxt))
            rewards.append(float(nxt.get("reward", 0.0) or 0.0))
            dones.append(float(nxt.get("done", 0.0) or 0.0))
            if max_transitions > 0 and len(raw_obs) >= max_transitions:
                break
        if max_transitions > 0 and len(raw_obs) >= max_transitions:
            break
    if len(raw_obs) < 100:
        raise RuntimeError(f"too few transitions for TVA fitting: {len(raw_obs)}")
    return TvaTransitionPack(
        raw_obs=np.asarray(raw_obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_raw_obs=np.asarray(next_raw_obs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sd_model(checkpoint: Path) -> DMControlHopperSDModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_kwargs: dict[str, Any] = {
        "action_prior_enabled": False,
        "shaping_scale": 0.0,
        "dense_pose_delta_weight": 0.0,
        "dense_pose_abs_weight": 0.0,
        "dense_pose_milestone_scale": 0.0,
        "recover_penalty_scale": 0.0,
        "stage_mode": "full",
    }
    model_kwargs.update(payload.get("model_config", {}))
    model = DMControlHopperSDModel(**model_kwargs).eval()
    src_state = payload.get("model_state", payload)
    target_state = model.state_dict()
    compatible = {
        k: v for k, v in src_state.items()
        if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return model


def encode_in_batches(model: DMControlHopperSDModel, raw_obs: np.ndarray, batch_size: int) -> np.ndarray:
    encoded: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(raw_obs), batch_size):
            obs_t = torch.tensor(raw_obs[start:start + batch_size], dtype=torch.float32)
            encoded.append(model.encode_observation(obs_t).detach().cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float32)


def load_value_checkpoint(value_net: torch.nn.Module, init_checkpoint: Path | None) -> int:
    if init_checkpoint is None:
        return 0
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"TVA init checkpoint not found: {init_checkpoint}")
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    src_state = payload.get("value_net_state", payload)
    target_state = value_net.state_dict()
    compatible = {
        k: v for k, v in src_state.items()
        if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
    }
    value_net.load_state_dict(compatible, strict=False)
    return len(compatible)


def write_markdown(summary_path: Path, payload: dict[str, Any]) -> None:
    md_path = summary_path.with_suffix(".md")
    lines = [
        "# Hopper TVA Explainer",
        "",
        f"Selection: `{payload['selection']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['checkpoint']}`",
        f"Init checkpoint: `{payload['init_checkpoint']}`",
        f"Transitions: `{payload['num_transitions']}`",
        f"Train loss: `{payload['train_loss']:.6f}`",
        f"Eval smooth-l1: `{payload['eval_smooth_l1']:.6f}`",
        f"Eval MSE: `{payload['eval_mse']:.6f}`",
        "",
        "## Target Statistics",
        "",
    ]
    for key, value in payload["target_stats"].items():
        lines.append(f"- `{key}`: `{value:.6f}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--max-transitions", type=int, default=50000)
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--true-coef", type=float, default=1.0)
    parser.add_argument("--shaped-coef", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tva_source = load_hopper_tva_module()
    selection = load_json(args.selection)
    dataset = build_transition_pack(selection, args.max_transitions, args.seed)
    sd_model = load_sd_model(args.sd_checkpoint)

    states = encode_in_batches(sd_model, dataset.raw_obs, args.encode_batch_size)
    next_states = encode_in_batches(sd_model, dataset.next_raw_obs, args.encode_batch_size)
    data = {
        "states": states,
        "actions": dataset.actions.astype(np.float32),
        "next_states": next_states,
        "rewards": dataset.rewards.astype(np.float32),
        "dones": dataset.dones.astype(bool),
    }
    value_states, value_targets = tva_source.build_tva_targets(
        sd_model,
        data,
        gamma=args.gamma,
        true_coef=args.true_coef,
        shaped_coef=args.shaped_coef,
        device="cpu",
    )

    value_net = tva_source.HopperValueNet(state_dim=int(value_states.shape[-1]), hidden=args.hidden)
    init_loaded = load_value_checkpoint(value_net, args.init_checkpoint)
    train_loss = tva_source.train_value_net(
        value_net,
        value_states,
        value_targets,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    with torch.no_grad():
        pred = value_net.to("cpu")(value_states)
        eval_smooth_l1 = float(F.smooth_l1_loss(pred, value_targets).item())
        eval_mse = float(F.mse_loss(pred, value_targets).item())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "value_net_state": value_net.to("cpu").state_dict(),
            "args": vars(args),
            "state_dim": int(value_states.shape[-1]),
            "hidden": int(args.hidden),
            "num_transitions": int(len(value_states)),
            "train_loss": float(train_loss),
            "eval_smooth_l1": eval_smooth_l1,
            "eval_mse": eval_mse,
            "target_mean": float(value_targets.mean().item()),
            "target_std": float(value_targets.std(unbiased=False).item()),
            "sd_checkpoint": str(args.sd_checkpoint),
        },
        args.output,
    )

    payload = {
        "selection": str(args.selection),
        "sd_checkpoint": str(args.sd_checkpoint),
        "checkpoint": str(args.output),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "init_loaded_tensors": init_loaded,
        "seed": args.seed,
        "num_transitions": int(len(value_states)),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "gamma": float(args.gamma),
        "true_coef": float(args.true_coef),
        "shaped_coef": float(args.shaped_coef),
        "train_loss": float(train_loss),
        "eval_smooth_l1": eval_smooth_l1,
        "eval_mse": eval_mse,
        "target_stats": {
            "mean": float(value_targets.mean().item()),
            "std": float(value_targets.std(unbiased=False).item()),
            "min": float(value_targets.min().item()),
            "max": float(value_targets.max().item()),
        },
    }
    write_json(args.summary, payload)
    write_markdown(args.summary, payload)
    print(f"wrote TVA checkpoint: {args.output}")
    print(f"wrote TVA summary: {args.summary}")
    print(f"transitions: {len(value_states)} train_loss={train_loss:.6f}")


if __name__ == "__main__":
    main()
