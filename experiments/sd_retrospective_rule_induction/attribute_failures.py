#!/usr/bin/env python3
"""Attribute Hopper successes and failures from SD-derived trace features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ATTRIBUTIONS,
    DEFAULT_SELECTION,
    action_abs_mean,
    f,
    load_json,
    mean_or_zero,
    read_csv,
    write_json,
)


def window_max(rows: list[dict[str, str]], key: str, start: int, horizon: int) -> float:
    end = min(len(rows), start + horizon)
    if start >= end:
        return 0.0
    return max(f(row, key) for row in rows[start:end])


def summarize_trace(trace_path: Path, horizon: int) -> dict[str, Any]:
    rows = read_csv(trace_path)
    n = len(rows)
    if n == 0:
        return {"num_frames": 0, "events": {}, "features": {}}

    unsupported_high_torque = 0
    landing_unprepared = 0
    rotation_risk = 0
    prone_or_rear_trap = 0
    prepared_support = 0
    effective_drive = 0
    counterfactual_candidates: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        full_foot = f(row, "full_foot_support")
        foot_grounded = f(row, "foot_grounded")
        grounded_prepared = f(row, "grounded_prepared")
        grounded_upright = f(row, "grounded_upright")
        grounded_prone = f(row, "grounded_prone")
        grounded_rear = f(row, "grounded_rear")
        falling_air = f(row, "falling_air")
        rotation_penalty = f(row, "rotation_penalty")
        pitch_rate = abs(f(row, "pitch_rate"))
        vx = f(row, "vx")
        vz = f(row, "vz")
        action_mag = action_abs_mean(row)

        if action_mag >= 0.65 and full_foot < 0.20 and foot_grounded < 0.50:
            unsupported_high_torque += 1
            if len(counterfactual_candidates) < 8:
                counterfactual_candidates.append(
                    {
                        "step": int(f(row, "step", i)),
                        "type": "unsupported_high_torque",
                        "action_abs_mean": action_mag,
                        "full_foot_support": full_foot,
                        "foot_grounded": foot_grounded,
                        "pitch_rate": f(row, "pitch_rate"),
                    }
                )

        next_support = window_max(rows, "full_foot_support", i + 1, horizon)
        if falling_air >= 0.50 and vz < -0.20 and next_support < 0.35:
            landing_unprepared += 1

        if rotation_penalty >= 0.50 or pitch_rate >= 3.0:
            rotation_risk += 1

        if grounded_prone >= 0.50 or grounded_rear >= 0.50:
            prone_or_rear_trap += 1

        if full_foot >= 0.50 and (grounded_prepared >= 0.50 or grounded_upright >= 0.50):
            prepared_support += 1

        if vx > 0.75 and action_mag >= 0.40 and full_foot >= 0.35:
            effective_drive += 1

    rootx_values = [f(row, "rootx", f(row, "qpos_rootx")) for row in rows]
    features = {
        "unsupported_high_torque_frac": unsupported_high_torque / n,
        "landing_unprepared_frac": landing_unprepared / n,
        "rotation_risk_frac": rotation_risk / n,
        "prone_or_rear_trap_frac": prone_or_rear_trap / n,
        "prepared_support_frac": prepared_support / n,
        "effective_drive_frac": effective_drive / n,
        "mean_full_foot_support": mean_or_zero(f(row, "full_foot_support") for row in rows),
        "mean_grounded_upright": mean_or_zero(f(row, "grounded_upright") for row in rows),
        "mean_grounded_prepared": mean_or_zero(f(row, "grounded_prepared") for row in rows),
        "mean_vx": mean_or_zero(f(row, "vx") for row in rows),
        "max_rootx": max(rootx_values) if rootx_values else 0.0,
        "final_rootx": rootx_values[-1] if rootx_values else 0.0,
        "mean_action_abs": mean_or_zero(action_abs_mean(row) for row in rows),
    }
    events = {
        "unsupported_high_torque_frames": unsupported_high_torque,
        "landing_unprepared_frames": landing_unprepared,
        "rotation_risk_frames": rotation_risk,
        "prone_or_rear_trap_frames": prone_or_rear_trap,
        "prepared_support_frames": prepared_support,
        "effective_drive_frames": effective_drive,
    }
    return {
        "num_frames": n,
        "features": features,
        "events": events,
        "counterfactual_candidates": counterfactual_candidates,
    }


def attribute(selection: dict[str, Any], horizon: int) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    missing_traces = 0
    for ep in selection["selected_episodes"]:
        trace_csv = ep.get("trace_csv")
        if not trace_csv or not Path(trace_csv).exists():
            missing_traces += 1
            continue
        trace_summary = summarize_trace(Path(trace_csv), horizon)
        features = dict(ep.get("summary_metrics", {}))
        features.update(trace_summary["features"])
        episodes.append(
            {
                "episode_id": (
                    f"seed{ep['seed']}:{ep['run_name']}:"
                    f"{ep['cycle_step']}:episode{ep['episode_idx']:02d}"
                ),
                "seed": ep["seed"],
                "run_name": ep["run_name"],
                "cycle_step": ep["cycle_step"],
                "episode_idx": ep["episode_idx"],
                "label": ep["label"],
                "random_selected": ep["random_selected"],
                "episode_reward": ep["episode_reward"],
                "trace_csv": trace_csv,
                "features": features,
                "events": trace_summary["events"],
                "counterfactual_candidates": trace_summary["counterfactual_candidates"],
            }
        )

    return {
        "source_selection": selection.get("manifest"),
        "horizon": horizon,
        "num_episodes": len(episodes),
        "missing_traces": missing_traces,
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_ATTRIBUTIONS)
    parser.add_argument("--horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = load_json(args.selection)
    payload = attribute(selection, args.horizon)
    payload["selection"] = str(args.selection)
    write_json(args.output, payload)
    print(f"wrote attributions: {args.output}")
    print(f"attributed episodes: {payload['num_episodes']} missing={payload['missing_traces']}")


if __name__ == "__main__":
    main()

