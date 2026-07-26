#!/usr/bin/env python3
"""Audit TVA accuracy as terminal compensation for SD rollouts.

The diagnostic compares learned TVA values at h-step terminal states with
realized remaining TVA targets from the same trajectory.  It reports two
settings:

1. oracle_terminal: TVA evaluated on the true trajectory state s_{t+h}.
2. sd_terminal: TVA evaluated on the SD rollout state after replaying the
   actual actions from s_t for h steps.

The first setting isolates value-network fitting.  The second setting is the
actual terminal-compensation use case for SD planning.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, read_csv, row_to_action, row_to_raw_obs, f, write_json  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


def load_hopper_tva_module():
    module_path = REPO_ROOT / "scripts" / "run_dmcontrol_hopper.py"
    spec = importlib.util.spec_from_file_location("sdmcts_hopper_tva_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load TVA source module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sd_model(checkpoint: Path, device: str) -> DMControlHopperSDModel:
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
    model = DMControlHopperSDModel(**model_kwargs).to(device).eval()
    src_state = payload.get("model_state", payload)
    target_state = model.state_dict()
    compatible = {
        k: v for k, v in src_state.items()
        if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return model


def load_value_net(checkpoint: Path, device: str) -> torch.nn.Module:
    tva_source = load_hopper_tva_module()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value_net = tva_source.HopperValueNet(
        state_dim=int(payload.get("state_dim", 17)),
        hidden=int(payload.get("hidden", 128)),
    )
    value_net.load_state_dict(payload.get("value_net_state", payload), strict=False)
    return value_net.to(device).eval()


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return pearson(rankdata(xs), rankdata(ys))


def choose_indices(n: int, max_count: int, rng: random.Random) -> list[int]:
    if n <= 0:
        return []
    if max_count <= 0 or n <= max_count:
        return list(range(n))
    return sorted(rng.sample(range(n), max_count))


def state_rmse(model: DMControlHopperSDModel, pred: torch.Tensor, target: torch.Tensor) -> float:
    pos_err = pred[:, :1] - target[:, :1]
    ang_err = torch.atan2(torch.sin(pred[:, 1:6] - target[:, 1:6]), torch.cos(pred[:, 1:6] - target[:, 1:6]))
    vel_err = pred[:, 6:13] - target[:, 6:13]
    touch_err = pred[:, 13:] - target[:, 13:]
    err = torch.cat([pos_err, ang_err, vel_err, touch_err], dim=-1)
    del model
    return float(torch.sqrt((err ** 2).mean()).item())


def summarize(pairs: list[dict[str, float]], pred_key: str) -> dict[str, float]:
    targets = [float(r["target"]) for r in pairs]
    preds = [float(r[pred_key]) for r in pairs]
    if not targets:
        return {
            "n": 0,
            "target_mean": 0.0,
            "pred_mean": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "smooth_l1": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "r2": 0.0,
            "baseline_mae": 0.0,
            "baseline_rmse": 0.0,
        }
    target_arr = np.asarray(targets, dtype=np.float64)
    pred_arr = np.asarray(preds, dtype=np.float64)
    err = pred_arr - target_arr
    baseline = np.full_like(target_arr, float(target_arr.mean()))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target_arr - target_arr.mean()) ** 2))
    return {
        "n": int(len(targets)),
        "target_mean": float(target_arr.mean()),
        "target_std": float(target_arr.std()),
        "pred_mean": float(pred_arr.mean()),
        "pred_std": float(pred_arr.std()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "smooth_l1": float(F.smooth_l1_loss(torch.tensor(pred_arr), torch.tensor(target_arr)).item()),
        "pearson": float(pearson(preds, targets)),
        "spearman": float(spearman(preds, targets)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0,
        "baseline_mae": float(np.mean(np.abs(baseline - target_arr))),
        "baseline_rmse": float(np.sqrt(np.mean((baseline - target_arr) ** 2))),
    }


def parse_horizons(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive horizons")
    return sorted(set(out))


def evaluate_episode(
    model: DMControlHopperSDModel,
    value_net: torch.nn.Module,
    tva_source: Any,
    episode: dict[str, Any],
    horizons: list[int],
    max_starts_per_episode: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[int, list[dict[str, float]]]:
    rows = read_csv(Path(episode["trace_csv"]))
    if len(rows) < max(horizons) + 2:
        return {h: [] for h in horizons}

    raw_obs = np.asarray([row_to_raw_obs(row) for row in rows], dtype=np.float32)
    actions = np.asarray([row_to_action(row) for row in rows[:-1]], dtype=np.float32)
    next_raw_obs = raw_obs[1:]
    rewards = np.asarray([f(row, "reward", 0.0) for row in rows[1:]], dtype=np.float32)
    dones = np.asarray([bool(round(f(row, "done", 0.0))) for row in rows[1:]], dtype=bool)
    if len(dones):
        dones[-1] = True

    with torch.no_grad():
        raw_t = torch.tensor(raw_obs, dtype=torch.float32, device=args.device)
        states = model.encode_observation(raw_t)
        data = {
            "states": states[:-1].detach().cpu().numpy().astype(np.float32),
            "actions": actions.astype(np.float32),
            "next_states": states[1:].detach().cpu().numpy().astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "dones": dones,
        }
        target_states, targets = tva_source.build_tva_targets(
            model,
            data,
            gamma=args.gamma,
            true_coef=args.true_coef,
            shaped_coef=args.shaped_coef,
            device=args.device,
        )
        target_states = target_states.to(args.device)
        targets = targets.to(args.device)

    out = {h: [] for h in horizons}
    for h in horizons:
        valid_max = len(targets) - h
        starts = choose_indices(valid_max, max_starts_per_episode, rng)
        if not starts:
            continue
        for i in starts:
            with torch.no_grad():
                pred_state = target_states[i:i + 1].clone()
                for k in range(h):
                    action_t = torch.tensor(actions[i + k:i + k + 1], dtype=torch.float32, device=args.device)
                    pred_state = model(pred_state, action_t)
                terminal_idx = i + h
                true_terminal = target_states[terminal_idx:terminal_idx + 1]
                target = float(targets[terminal_idx].item())
                oracle_v = float(value_net(true_terminal).item())
                sd_v = float(value_net(pred_state).item())
                out[h].append(
                    {
                        "target": target,
                        "oracle_value": oracle_v,
                        "sd_value": sd_v,
                        "sd_state_rmse": state_rmse(model, pred_state.detach().cpu(), true_terminal.detach().cpu()),
                        "episode_reward": float(episode.get("episode_reward", 0.0)),
                    }
                )
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper TVA Terminal Compensation Audit",
        "",
        f"Selection: `{payload['selection']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['tva_checkpoint']}`",
        f"Episodes used: `{payload['episodes_used']}`",
        f"Samples: `{payload['total_samples']}`",
        "",
        "## Horizon Summary",
        "",
        "| h | mode | n | target std | pred std | MAE | RMSE | baseline RMSE | Pearson | Spearman | R2 | SD state RMSE |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for h in payload["horizons"]:
        hkey = str(h)
        state_rmse_mean = payload["horizon_summary"][hkey].get("sd_state_rmse_mean", 0.0)
        for mode in ["oracle_terminal", "sd_terminal"]:
            stats = payload["horizon_summary"][hkey][mode]
            lines.append(
                "| "
                f"{h} | `{mode}` | {stats['n']} | "
                f"{stats['target_std']:.3f} | {stats['pred_std']:.3f} | "
                f"{stats['mae']:.3f} | {stats['rmse']:.3f} | {stats['baseline_rmse']:.3f} | "
                f"{stats['pearson']:.3f} | {stats['spearman']:.3f} | {stats['r2']:.3f} | "
                f"{state_rmse_mean:.3f} |"
            )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--tva-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=parse_horizons, default=[1, 3, 5])
    parser.add_argument("--max-episodes", type=int, default=300)
    parser.add_argument("--max-starts-per-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--true-coef", type=float, default=1.0)
    parser.add_argument("--shaped-coef", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tva_source = load_hopper_tva_module()
    selection = load_json(args.selection)
    episodes = list(selection["selected_episodes"])
    rng.shuffle(episodes)
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    model = load_sd_model(args.sd_checkpoint, args.device)
    value_net = load_value_net(args.tva_checkpoint, args.device)

    samples_by_horizon: dict[int, list[dict[str, float]]] = {h: [] for h in args.horizons}
    for ep in episodes:
        ep_samples = evaluate_episode(model, value_net, tva_source, ep, args.horizons, args.max_starts_per_episode, rng, args)
        for h, rows in ep_samples.items():
            samples_by_horizon[h].extend(rows)

    horizon_summary: dict[str, Any] = {}
    for h, rows in samples_by_horizon.items():
        horizon_summary[str(h)] = {
            "oracle_terminal": summarize(rows, "oracle_value"),
            "sd_terminal": summarize(rows, "sd_value"),
            "sd_state_rmse_mean": float(np.mean([r["sd_state_rmse"] for r in rows])) if rows else 0.0,
            "sd_state_rmse_std": float(np.std([r["sd_state_rmse"] for r in rows])) if rows else 0.0,
        }

    payload = {
        "audit_family": "hopper_tva_terminal_compensation_v1",
        "selection": str(args.selection),
        "sd_checkpoint": str(args.sd_checkpoint),
        "tva_checkpoint": str(args.tva_checkpoint),
        "horizons": args.horizons,
        "max_episodes": int(args.max_episodes),
        "max_starts_per_episode": int(args.max_starts_per_episode),
        "episodes_used": len(episodes),
        "total_samples": int(sum(len(v) for v in samples_by_horizon.values())),
        "gamma": float(args.gamma),
        "true_coef": float(args.true_coef),
        "shaped_coef": float(args.shaped_coef),
        "horizon_summary": horizon_summary,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA terminal compensation audit: {args.output}")
    print(f"samples: {payload['total_samples']}")


if __name__ == "__main__":
    main()
