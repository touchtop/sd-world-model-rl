#!/usr/bin/env python3
"""Audit whether a Hopper TVA checkpoint carries useful value signal.

This is a side-effect-free diagnostic.  It evaluates TVA values on selected
trajectory states and compares episode-level TVA statistics with realized
episode returns and selection labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import DEFAULT_SELECTION, load_json, read_csv, row_to_raw_obs, write_json  # noqa: E402
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
        state_dim=int(payload.get("state_dim", 15)),
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


def choose_row_indices(n: int, max_rows: int) -> list[int]:
    if n <= 0:
        return []
    if max_rows <= 0 or n <= max_rows:
        return list(range(n))
    if max_rows == 1:
        return [0]
    return sorted({round(i * (n - 1) / (max_rows - 1)) for i in range(max_rows)})


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_episode(
    model: DMControlHopperSDModel,
    value_net: torch.nn.Module,
    episode: dict[str, Any],
    max_rows_per_episode: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    rows = read_csv(Path(episode["trace_csv"]))
    indices = choose_row_indices(len(rows), max_rows_per_episode)
    values: list[float] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        raw = [row_to_raw_obs(rows[i]) for i in batch_indices]
        obs_t = torch.tensor(raw, dtype=torch.float32, device=device)
        with torch.no_grad():
            states = model.encode_observation(obs_t)
            vals = value_net(states).detach().cpu().tolist()
        values.extend(float(v) for v in vals)
    first_n = max(1, min(5, len(values)))
    last_n = max(1, min(5, len(values)))
    return {
        "episode_id": f"seed{episode['seed']}:{episode['run_name']}:episode{episode['episode_idx']:02d}",
        "seed": episode["seed"],
        "cycle_step": episode["cycle_step"],
        "episode_idx": episode["episode_idx"],
        "label": episode["label"],
        "episode_reward": float(episode["episode_reward"]),
        "num_rows": len(rows),
        "num_eval_rows": len(values),
        "tva_mean": mean(values),
        "tva_max": max(values) if values else 0.0,
        "tva_min": min(values) if values else 0.0,
        "tva_first_mean": mean(values[:first_n]),
        "tva_last_mean": mean(values[-last_n:]),
        "trace_csv": episode["trace_csv"],
    }


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        groups.setdefault(f"label:{row['label']}", []).append(row)
        groups.setdefault(f"seed:{row['seed']}", []).append(row)
    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items()):
        rewards = [float(v["episode_reward"]) for v in vals]
        tva_means = [float(v["tva_mean"]) for v in vals]
        out[key] = {
            "count": len(vals),
            "reward_mean": mean(rewards),
            "tva_mean": mean(tva_means),
            "tva_max_mean": mean([float(v["tva_max"]) for v in vals]),
            "tva_first_mean": mean([float(v["tva_first_mean"]) for v in vals]),
            "tva_last_mean": mean([float(v["tva_last_mean"]) for v in vals]),
            "pearson_reward_tva_mean": pearson(rewards, tva_means),
            "spearman_reward_tva_mean": spearman(rewards, tva_means),
        }
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper TVA Value Signal Audit",
        "",
        f"Selection: `{payload['selection']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['tva_checkpoint']}`",
        f"Episodes: `{payload['num_episodes']}`",
        f"Max rows per episode: `{payload['max_rows_per_episode']}`",
        "",
        "## Aggregate",
        "",
        "| Group | Count | Reward mean | TVA mean | TVA max mean | Pearson | Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in payload["aggregate"].items():
        lines.append(
            "| "
            f"`{key}` | "
            f"{stats['count']} | "
            f"{stats['reward_mean']:.3f} | "
            f"{stats['tva_mean']:.3f} | "
            f"{stats['tva_max_mean']:.3f} | "
            f"{stats['pearson_reward_tva_mean']:.3f} | "
            f"{stats['spearman_reward_tva_mean']:.3f} |"
        )
    lines.extend(["", "## Top Episodes by TVA Mean", ""])
    for row in sorted(payload["episodes"], key=lambda x: x["tva_mean"], reverse=True)[:20]:
        lines.append(
            f"- `{row['episode_id']}` reward=`{row['episode_reward']:.3f}` "
            f"label=`{row['label']}` tva_mean=`{row['tva_mean']:.3f}`"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--tva-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-rows-per-episode", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = load_json(args.selection)
    episodes = selection["selected_episodes"]
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
    model = load_sd_model(args.sd_checkpoint, args.device)
    value_net = load_value_net(args.tva_checkpoint, args.device)
    rows = [
        evaluate_episode(model, value_net, ep, args.max_rows_per_episode, args.batch_size, args.device)
        for ep in episodes
    ]
    payload = {
        "audit_family": "hopper_tva_value_signal_v1",
        "selection": str(args.selection),
        "sd_checkpoint": str(args.sd_checkpoint),
        "tva_checkpoint": str(args.tva_checkpoint),
        "max_episodes": int(args.max_episodes),
        "max_rows_per_episode": int(args.max_rows_per_episode),
        "num_episodes": len(rows),
        "aggregate": group_stats(rows),
        "episodes": rows,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA value audit: {args.output}")
    print(f"episodes: {len(rows)}")


if __name__ == "__main__":
    main()
