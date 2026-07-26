#!/usr/bin/env python3
"""Audit the interface between Hopper SD rollouts and TVA terminal values.

This diagnostic does not test whether TVA or SD is useful in isolation.
Instead, it measures whether a TVA value learned on true encoded states remains
stable when evaluated on SD-predicted terminal states.  The key quantity is
V(SD(s_t, a_t:t+h)) - V(s_{t+h}), together with the state components that
explain this value shift.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit_hopper_tva_terminal_compensation import (  # noqa: E402
    choose_indices,
    load_hopper_tva_module,
    load_sd_model,
    load_value_net,
    parse_horizons,
    pearson,
    spearman,
    summarize,
)
from common import f, load_json, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402


STATE_GROUPS = {
    "height": (0, 1),
    "angles": (1, 6),
    "velocities": (6, 13),
    "touch": (13, None),
}


def rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values ** 2)))


def state_errors(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    err = pred_np - target_np
    angle_err = np.arctan2(np.sin(err[:, 1:6]), np.cos(err[:, 1:6]))
    wrapped = np.concatenate([err[:, :1], angle_err, err[:, 6:]], axis=1)
    out = {"state_rmse": rmse(wrapped)}
    for name, (start, end) in STATE_GROUPS.items():
        out[f"{name}_rmse"] = rmse(wrapped[:, start:end])
    return out


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_interface(rows: list[dict[str, float]]) -> dict[str, Any]:
    value_delta = [float(r["value_delta"]) for r in rows]
    abs_value_delta = [abs(v) for v in value_delta]
    out: dict[str, Any] = {
        "value_delta": numeric_summary(value_delta),
        "abs_value_delta": numeric_summary(abs_value_delta),
        "state_error": {},
        "abs_value_delta_correlations": {},
        "sd_minus_oracle": {},
    }
    for key in ["state_rmse", "height_rmse", "angles_rmse", "velocities_rmse", "touch_rmse"]:
        vals = [float(r[key]) for r in rows]
        out["state_error"][key] = numeric_summary(vals)
        out["abs_value_delta_correlations"][key] = {
            "pearson": float(pearson(vals, abs_value_delta)),
            "spearman": float(spearman(vals, abs_value_delta)),
        }
    oracle = summarize(rows, "oracle_value")
    sd = summarize(rows, "sd_value")
    out["oracle_terminal"] = oracle
    out["sd_terminal"] = sd
    out["sd_minus_oracle"] = {
        "pearson": float(sd["pearson"] - oracle["pearson"]),
        "spearman": float(sd["spearman"] - oracle["spearman"]),
        "r2": float(sd["r2"] - oracle["r2"]),
        "rmse": float(sd["rmse"] - oracle["rmse"]),
    }
    return out


def evaluate_episode(
    model: torch.nn.Module,
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
        for i in starts:
            with torch.no_grad():
                pred_state = target_states[i:i + 1].clone()
                for k in range(h):
                    action_t = torch.tensor(actions[i + k:i + k + 1], dtype=torch.float32, device=args.device)
                    pred_state = model(pred_state, action_t)
                terminal_idx = i + h
                true_terminal = target_states[terminal_idx:terminal_idx + 1]
                oracle_value = float(value_net(true_terminal).item())
                sd_value = float(value_net(pred_state).item())
                record = {
                    "target": float(targets[terminal_idx].item()),
                    "oracle_value": oracle_value,
                    "sd_value": sd_value,
                    "value_delta": float(sd_value - oracle_value),
                    "episode_reward": float(episode.get("episode_reward", 0.0)),
                    "seed": float(episode.get("seed", 0)),
                    "episode_idx": float(episode.get("episode_idx", 0)),
                    "start_index": float(i),
                    "terminal_index": float(terminal_idx),
                }
                record.update(state_errors(pred_state, true_terminal))
                out[h].append(record)
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper TVA-SD Interface Audit",
        "",
        f"Selection: `{payload['selection']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['tva_checkpoint']}`",
        f"Episodes used: `{payload['episodes_used']}`",
        f"Samples: `{payload['total_samples']}`",
        "",
        "## Horizon Summary",
        "",
        "| h | oracle r | sd r | oracle R2 | sd R2 | |delta V| mean | state RMSE | angle RMSE | velocity RMSE | corr(|delta V|, state) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for h in payload["horizons"]:
        s = payload["horizon_summary"][str(h)]
        lines.append(
            "| "
            f"{h} | "
            f"{s['oracle_terminal']['pearson']:.3f} | "
            f"{s['sd_terminal']['pearson']:.3f} | "
            f"{s['oracle_terminal']['r2']:.3f} | "
            f"{s['sd_terminal']['r2']:.3f} | "
            f"{s['abs_value_delta']['mean']:.3f} | "
            f"{s['state_error']['state_rmse']['mean']:.3f} | "
            f"{s['state_error']['angles_rmse']['mean']:.3f} | "
            f"{s['state_error']['velocities_rmse']['mean']:.3f} | "
            f"{s['abs_value_delta_correlations']['state_rmse']['pearson']:.3f} |"
        )
    lines.extend(["", "## Largest Value Gaps", ""])
    for row in payload["largest_value_gaps"][:20]:
        lines.append(
            "- "
            f"h=`{row['h']}` seed=`{int(row['seed'])}` episode=`{int(row['episode_idx'])}` "
            f"step=`{int(row['start_index'])}` reward=`{row['episode_reward']:.3f}` "
            f"delta_v=`{row['value_delta']:.3f}` state_rmse=`{row['state_rmse']:.3f}` "
            f"angle_rmse=`{row['angles_rmse']:.3f}` velocity_rmse=`{row['velocities_rmse']:.3f}`"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--tva-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=parse_horizons, default=[1, 3, 5])
    parser.add_argument("--max-episodes", type=int, default=60)
    parser.add_argument("--max-starts-per-episode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260518)
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

    rows_by_horizon: dict[int, list[dict[str, float]]] = {h: [] for h in args.horizons}
    for episode in episodes:
        ep_rows = evaluate_episode(model, value_net, tva_source, episode, args.horizons, args.max_starts_per_episode, rng, args)
        for h, rows in ep_rows.items():
            rows_by_horizon[h].extend(rows)

    all_rows: list[dict[str, float]] = []
    horizon_summary: dict[str, Any] = {}
    for h, rows in rows_by_horizon.items():
        for row in rows:
            row["h"] = float(h)
        all_rows.extend(rows)
        horizon_summary[str(h)] = summarize_interface(rows)

    largest = sorted(all_rows, key=lambda r: abs(float(r["value_delta"])), reverse=True)[:50]
    payload = {
        "audit_family": "hopper_tva_sd_interface_v1",
        "selection": str(args.selection),
        "sd_checkpoint": str(args.sd_checkpoint),
        "tva_checkpoint": str(args.tva_checkpoint),
        "horizons": args.horizons,
        "max_episodes": int(args.max_episodes),
        "max_starts_per_episode": int(args.max_starts_per_episode),
        "episodes_used": len(episodes),
        "total_samples": int(sum(len(rows) for rows in rows_by_horizon.values())),
        "gamma": float(args.gamma),
        "true_coef": float(args.true_coef),
        "shaped_coef": float(args.shaped_coef),
        "horizon_summary": horizon_summary,
        "largest_value_gaps": largest,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA-SD interface audit: {args.output}")
    print(f"samples: {payload['total_samples']}")


if __name__ == "__main__":
    main()
