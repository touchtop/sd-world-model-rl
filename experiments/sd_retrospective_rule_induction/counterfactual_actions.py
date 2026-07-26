#!/usr/bin/env python3
"""Run short-horizon SD counterfactual action comparisons on selected traces."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    DEFAULT_ATTRIBUTIONS,
    DEFAULT_COUNTERFACTUALS,
    DEFAULT_EXPLAINER,
    f,
    load_json,
    mean_or_zero,
    read_csv,
    resolve_workers,
    row_to_action,
    row_to_raw_obs,
    write_json,
)
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


ACTION_NAMES = ("waist", "hip", "knee", "ankle")
_WORKER_MODEL: DMControlHopperSDModel | None = None
_WORKER_ACTION_BANK: list[tuple[str, torch.Tensor]] = []
_WORKER_HORIZON = 5
_WORKER_MAX_STATES_PER_EPISODE = 3


def load_model(model_checkpoint: Path | None) -> tuple[DMControlHopperSDModel, bool]:
    model_kwargs: dict[str, Any] = {
        "action_prior_enabled": False,
        "shaping_scale": 0.0,
        "dense_pose_delta_weight": 0.0,
        "dense_pose_abs_weight": 0.0,
        "dense_pose_milestone_scale": 0.0,
        "recover_penalty_scale": 0.0,
        "stage_mode": "full",
    }
    payload = None
    if model_checkpoint is not None and model_checkpoint.exists():
        payload = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
        model_kwargs.update(payload.get("model_config", {}))
    model = DMControlHopperSDModel(**model_kwargs).eval()
    if payload is not None:
        src_state = payload.get("model_state", payload)
        target_state = model.state_dict()
        compatible = {
            k: v for k, v in src_state.items()
            if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
        }
        model.load_state_dict(compatible, strict=False)
    return model, payload is not None


def init_counterfactual_worker(
    model_checkpoint: str | None,
    amplitudes: list[float],
    horizon: int,
    max_states_per_episode: int,
) -> None:
    global _WORKER_MODEL, _WORKER_ACTION_BANK, _WORKER_HORIZON, _WORKER_MAX_STATES_PER_EPISODE
    torch.set_num_threads(1)
    checkpoint_path = Path(model_checkpoint) if model_checkpoint else None
    _WORKER_MODEL, _ = load_model(checkpoint_path)
    _WORKER_ACTION_BANK = build_action_bank(amplitudes)
    _WORKER_HORIZON = horizon
    _WORKER_MAX_STATES_PER_EPISODE = max_states_per_episode


def process_counterfactual_episode(ep: dict[str, Any]) -> list[dict[str, Any]]:
    if _WORKER_MODEL is None:
        raise RuntimeError("counterfactual worker model was not initialized")
    records: list[dict[str, Any]] = []
    rows = read_csv(Path(ep["trace_csv"]))
    for row in select_rows(rows, _WORKER_MAX_STATES_PER_EPISODE):
        state_record = evaluate_state(_WORKER_MODEL, row, _WORKER_ACTION_BANK, _WORKER_HORIZON)
        state_record.update(
            {
                "episode_id": ep["episode_id"],
                "seed": ep["seed"],
                "cycle_step": ep["cycle_step"],
                "label": ep["label"],
                "episode_reward": ep["episode_reward"],
                "trace_csv": ep["trace_csv"],
            }
        )
        records.append(state_record)
    return records


def build_action_bank(amplitudes: list[float]) -> list[tuple[str, torch.Tensor]]:
    bank: list[tuple[str, torch.Tensor]] = []
    zero = torch.zeros(1, DMControlHopperSDModel.ACTION_DIM, dtype=torch.float32)
    bank.append(("zero", zero))
    for amp in amplitudes:
        for dim, name in enumerate(ACTION_NAMES):
            for sign in (-1.0, 1.0):
                action = torch.zeros(1, DMControlHopperSDModel.ACTION_DIM, dtype=torch.float32)
                action[0, dim] = sign * float(amp)
                bank.append((f"{name}_{'pos' if sign > 0 else 'neg'}_{amp:.2f}", action))
    return bank


def desirability(feat: dict[str, torch.Tensor]) -> float:
    """Score short-horizon structural outcome for recovery/useful locomotion.

    The score intentionally stays simple and interpretable. It rewards forward
    velocity, prepared support and upright grounded states, while penalizing
    prone/rear traps, rotation risk and unsupported falling.
    """

    full_foot = float(feat["full_foot_support"][0].item())
    upright = float(feat["grounded_upright"][0].item())
    prepared = float(feat["grounded_prepared"][0].item())
    rear = float(feat["grounded_rear"][0].item())
    prone = float(feat["grounded_prone"][0].item())
    falling = float(feat["falling_air"][0].item())
    vx = float(feat["vx"][0].item())
    vz = float(feat["vz"][0].item())
    omega = abs(float(feat["pitch_rate"][0].item()))
    support = max(full_foot, prepared)
    return (
        1.10 * support
        + 0.80 * upright
        + 0.35 * max(vx, -1.0)
        + 0.15 * max(vz, -1.0)
        - 1.20 * max(rear, prone)
        - 0.60 * falling
        - 0.08 * min(omega, 8.0)
    )


def desirability_batch(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    full_foot = feat["full_foot_support"]
    upright = feat["grounded_upright"]
    prepared = feat["grounded_prepared"]
    rear = feat["grounded_rear"]
    prone = feat["grounded_prone"]
    falling = feat["falling_air"]
    vx = feat["vx"]
    vz = feat["vz"]
    omega = torch.abs(feat["pitch_rate"])
    support = torch.maximum(full_foot, prepared)
    return (
        1.10 * support
        + 0.80 * upright
        + 0.35 * torch.clamp(vx, min=-1.0)
        + 0.15 * torch.clamp(vz, min=-1.0)
        - 1.20 * torch.maximum(rear, prone)
        - 0.60 * falling
        - 0.08 * torch.clamp(omega, max=8.0)
    )


def rollout_scores(
    model: DMControlHopperSDModel,
    state: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
) -> tuple[list[float], list[dict[str, float]]]:
    cur = state.repeat(actions.shape[0], 1)
    with torch.no_grad():
        for _ in range(horizon):
            cur = model(cur, actions)
        feat = model._state_reward_features(cur)
        score_t = desirability_batch(feat)
    scores = [float(x) for x in score_t.detach().cpu().tolist()]
    summaries: list[dict[str, float]] = []
    for i, score in enumerate(scores):
        summaries.append(
            {
                "score": score,
                "full_foot_support": float(feat["full_foot_support"][i].item()),
                "grounded_upright": float(feat["grounded_upright"][i].item()),
                "grounded_prepared": float(feat["grounded_prepared"][i].item()),
                "grounded_rear": float(feat["grounded_rear"][i].item()),
                "grounded_prone": float(feat["grounded_prone"][i].item()),
                "falling_air": float(feat["falling_air"][i].item()),
                "vx": float(feat["vx"][i].item()),
                "vz": float(feat["vz"][i].item()),
                "pitch_rate": float(feat["pitch_rate"][i].item()),
            }
        )
    return scores, summaries


def classify_event(row: dict[str, str]) -> tuple[str, float]:
    unsupported = (
        max(abs(x) for x in row_to_action(row)) >= 0.65
        and f(row, "full_foot_support") < 0.20
        and f(row, "foot_grounded") < 0.50
    )
    landing_unprepared = f(row, "falling_air") >= 0.50 and f(row, "vz") < -0.20
    rotation = f(row, "rotation_penalty") >= 0.50 or abs(f(row, "pitch_rate")) >= 3.0
    trap = f(row, "grounded_prone") >= 0.50 or f(row, "grounded_rear") >= 0.50
    prepared = (
        f(row, "full_foot_support") >= 0.50
        and (f(row, "grounded_prepared") >= 0.50 or f(row, "grounded_upright") >= 0.50)
    )
    effective = f(row, "vx") > 0.75 and f(row, "full_foot_support") >= 0.35

    candidates = [
        ("unsupported_high_torque", 1.0 if unsupported else 0.0),
        ("landing_unprepared", 1.0 if landing_unprepared else 0.0),
        ("rotation_risk", 1.0 if rotation else 0.0),
        ("prone_or_rear_trap", 1.0 if trap else 0.0),
        ("prepared_support", 1.0 if prepared else 0.0),
        ("effective_drive", 1.0 if effective else 0.0),
    ]
    return max(candidates, key=lambda item: item[1])


def select_rows(rows: list[dict[str, str]], max_states: int) -> list[dict[str, str]]:
    scored: list[tuple[float, int, dict[str, str]]] = []
    for i, row in enumerate(rows):
        event_name, event_hit = classify_event(row)
        score = (
            2.0 * event_hit
            + f(row, "grounded_prone")
            + f(row, "grounded_rear")
            + f(row, "grounded_upright")
            + f(row, "grounded_prepared")
            + f(row, "full_foot_support")
            + 0.2 * abs(f(row, "pitch_rate"))
        )
        row_out = dict(row)
        row_out["_event_name"] = event_name
        row_out["_event_score"] = str(event_hit)
        scored.append((score, i, row_out))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = sorted(scored[:max_states], key=lambda item: item[1])
    return [row for _, _, row in chosen]


def evaluate_state(
    model: DMControlHopperSDModel,
    row: dict[str, str],
    action_bank: list[tuple[str, torch.Tensor]],
    horizon: int,
) -> dict[str, Any]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32).unsqueeze(0))
    actual_action = torch.tensor(row_to_action(row), dtype=torch.float32).unsqueeze(0).clamp(-1.0, 1.0)
    probe_names = ["actual"] + [name for name, _ in action_bank]
    probe_actions = torch.cat([actual_action, *[action for _, action in action_bank]], dim=0)
    scores, summaries = rollout_scores(model, state, probe_actions, horizon)
    actual_score = scores[0]
    actual_summary = summaries[0]

    alternatives: list[dict[str, Any]] = []
    for name, score, summary in zip(probe_names[1:], scores[1:], summaries[1:]):
        alternatives.append({"name": name, "score": score, "summary": summary})
    alternatives.sort(key=lambda item: item["score"], reverse=True)
    best = alternatives[0]
    rank = 1 + sum(1 for item in alternatives if item["score"] > actual_score)
    event_name, event_hit = classify_event(row)
    return {
        "step": int(f(row, "step")),
        "event": event_name,
        "event_hit": event_hit,
        "actual_action": row_to_action(row),
        "actual_score": actual_score,
        "actual_summary": actual_summary,
        "best_action": best["name"],
        "best_score": best["score"],
        "best_summary": best["summary"],
        "advantage_over_actual": best["score"] - actual_score,
        "actual_rank_among_probe_actions": rank,
        "num_probe_actions": len(alternatives),
        "top_actions": alternatives[:5],
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups["all"].append(rec)
        groups[f"label:{rec['label']}"].append(rec)
        groups[f"event:{rec['event']}"].append(rec)

    out: dict[str, Any] = {}
    for key, vals in groups.items():
        out[key] = {
            "count": len(vals),
            "mean_advantage_over_actual": mean_or_zero(v["advantage_over_actual"] for v in vals),
            "mean_actual_rank": mean_or_zero(v["actual_rank_among_probe_actions"] for v in vals),
            "actual_top3_rate": mean_or_zero(
                1.0 if v["actual_rank_among_probe_actions"] <= 3 else 0.0 for v in vals
            ),
        }
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    md_path = path.with_suffix(".md")
    lines = [
        "# Hopper Offline Counterfactuals v1",
        "",
        f"State records: `{payload['num_state_records']}`",
        f"Horizon: `{payload['horizon']}`",
        f"Amplitudes: `{payload['amplitudes']}`",
        "",
        "## Aggregate",
        "",
        "| Group | Count | Mean advantage | Mean actual rank | Actual top-3 rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in sorted(payload["aggregate"].items()):
        lines.append(
            "| "
            f"`{key}` | "
            f"{stats['count']} | "
            f"{stats['mean_advantage_over_actual']:.4f} | "
            f"{stats['mean_actual_rank']:.2f} | "
            f"{stats['actual_top3_rate']:.3f} |"
        )
    lines.extend(["", "## Top Counterfactual Gaps", ""])
    top_records = sorted(
        payload["records"],
        key=lambda rec: rec["advantage_over_actual"],
        reverse=True,
    )[:20]
    for rec in top_records:
        lines.extend(
            [
                f"### {rec['episode_id']} step {rec['step']}",
                "",
                f"- Label: `{rec['label']}`",
                f"- Event: `{rec['event']}`",
                f"- Actual score: `{rec['actual_score']:.4f}`",
                f"- Best action: `{rec['best_action']}`",
                f"- Best score: `{rec['best_score']:.4f}`",
                f"- Advantage: `{rec['advantage_over_actual']:.4f}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines))


def run_counterfactuals(
    attributions: dict[str, Any],
    max_episodes: int,
    max_states_per_episode: int,
    horizon: int,
    amplitudes: list[float],
    model_checkpoint: Path | None,
    workers: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    episodes = attributions["episodes"][:max_episodes] if max_episodes > 0 else attributions["episodes"]
    worker_count = resolve_workers(workers)
    checkpoint_text = str(model_checkpoint) if model_checkpoint is not None and model_checkpoint.exists() else None
    if worker_count <= 1 or len(episodes) <= 1:
        init_counterfactual_worker(checkpoint_text, amplitudes, horizon, max_states_per_episode)
        for ep in episodes:
            records.extend(process_counterfactual_episode(ep))
    else:
        chunksize = max(1, len(episodes) // (worker_count * 4))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_counterfactual_worker,
            initargs=(checkpoint_text, amplitudes, horizon, max_states_per_episode),
        ) as pool:
            for episode_records in pool.map(process_counterfactual_episode, episodes, chunksize=chunksize):
                records.extend(episode_records)

    return {
        "source_attributions": attributions.get("selection"),
        "model": "DMControlHopperSDModel(offline-explainer)" if checkpoint_text else "DMControlHopperSDModel(default-structural-offline)",
        "model_checkpoint": checkpoint_text,
        "horizon": horizon,
        "amplitudes": amplitudes,
        "max_states_per_episode": max_states_per_episode,
        "num_episodes": len(episodes),
        "num_state_records": len(records),
        "workers": worker_count,
        "aggregate": aggregate(records),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions", type=Path, default=DEFAULT_ATTRIBUTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_COUNTERFACTUALS)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-states-per-episode", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0, help="CPU worker processes; 0 uses SD_RETRO_WORKERS or a bounded auto default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attributions = load_json(args.attributions)
    payload = run_counterfactuals(
        attributions,
        args.max_episodes,
        args.max_states_per_episode,
        args.horizon,
        args.amplitudes,
        args.model_checkpoint,
        args.workers,
    )
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote counterfactuals: {args.output}")
    print(f"state records: {payload['num_state_records']}")


if __name__ == "__main__":
    main()
