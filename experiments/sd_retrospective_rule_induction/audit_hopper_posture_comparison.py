#!/usr/bin/env python3
"""Compare Hopper official-eval posture metrics across experiment groups.

This audit is read-only with respect to training logs. It is intended for
post-hoc research checks where return alone is insufficient: the report keeps
reward, structural posture, support mode, collision, and toe-launch indicators
in the same table so that rule effects can be interpreted behaviorally.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def phase(row: dict[str, str]) -> str:
    falling = as_float(row, "falling_air") >= 0.5 or as_float(row, "falling_air_gt_05") >= 0.5
    rear = as_float(row, "grounded_rear") >= 0.5 or as_float(row, "grounded_rear_gt_05") >= 0.5
    prone = as_float(row, "grounded_prone") >= 0.5 or as_float(row, "grounded_prone_gt_05") >= 0.5
    hop = as_float(row, "grounded_hop") >= 0.5 or as_float(row, "grounded_hop_gt_05") >= 0.5
    upright = (
        as_float(row, "grounded_upright") >= 0.5
        or as_float(row, "grounded_upright_gt_05") >= 0.5
        or as_float(row, "grounded_prepared") >= 0.5
        or as_float(row, "grounded_prepared_gt_05") >= 0.5
    )
    rising = as_float(row, "rising_air") >= 0.5 or as_float(row, "rising_air_gt_05") >= 0.5
    if falling:
        return "falling"
    if rear:
        return "rear"
    if hop:
        return "hop"
    if prone:
        return "prone"
    if upright:
        return "upright"
    if rising:
        return "rising"
    return "other"


def structural_success(row: dict[str, str]) -> bool:
    return (
        as_float(row, "torso_z") >= 0.75
        and abs(as_float(row, "pitch")) <= 0.45
        and as_float(row, "vx") >= 0.80
        and abs(as_float(row, "pitch_rate")) <= 3.50
        and as_float(row, "body_collision") < 0.50
    )


def continuation_score(row: dict[str, str]) -> float:
    pitch = abs(as_float(row, "pitch"))
    torso_z = as_float(row, "torso_z")
    vx = as_float(row, "vx")
    omega = abs(as_float(row, "pitch_rate"))
    uprightness = max(0.0, 1.0 - pitch / 0.45)
    height_quality = min(max((torso_z - 0.45) / 0.45, 0.0), 1.0)
    forward_speed = min(max(vx / 2.0, 0.0), 1.0)
    spin_control = max(0.0, 1.0 - omega / 3.50)
    support_readiness = max(as_float(row, "support_contact"), as_float(row, "full_foot_support"))
    non_collision = 1.0 - min(max(as_float(row, "body_collision"), 0.0), 1.0)
    return (
        0.25 * uprightness
        + 0.20 * height_quality
        + 0.20 * forward_speed
        + 0.15 * spin_control
        + 0.10 * support_readiness
        + 0.10 * non_collision
    )


def toe_only_low_support(row: dict[str, str]) -> bool:
    return (
        as_float(row, "touch_toe") >= 0.80
        and as_float(row, "touch_heel") < 0.50
        and as_float(row, "full_foot_support") < 0.35
    )


def toe_only_launch(row: dict[str, str]) -> bool:
    return toe_only_low_support(row) and as_float(row, "vx") >= 0.80


def full_support_drive(row: dict[str, str]) -> bool:
    return as_float(row, "full_foot_support") >= 0.35 and as_float(row, "vx") >= 0.80


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    phases = Counter(phase(row) for row in rows)
    return {
        "rows": float(len(rows)),
        "structural_success_frac": mean(1.0 if structural_success(row) else 0.0 for row in rows),
        "dense_pose_high_frac": mean(1.0 if as_float(row, "dense_pose_potential") >= 0.65 else 0.0 for row in rows),
        "continue_score_ge_060_frac": mean(1.0 if continuation_score(row) >= 0.60 else 0.0 for row in rows),
        "near_good_frac": mean(
            1.0
            if structural_success(row) or as_float(row, "dense_pose_potential") >= 0.65 or as_float(row, "reward") > 0.0
            else 0.0
            for row in rows
        ),
        "positive_reward_frac": mean(1.0 if as_float(row, "reward") > 0.0 else 0.0 for row in rows),
        "mean_continue_score": mean(continuation_score(row) for row in rows),
        "mean_vx": mean(as_float(row, "vx") for row in rows),
        "mean_abs_pitch": mean(abs(as_float(row, "pitch")) for row in rows),
        "mean_abs_omega": mean(abs(as_float(row, "pitch_rate")) for row in rows),
        "mean_support_contact": mean(as_float(row, "support_contact") for row in rows),
        "mean_full_foot_support": mean(as_float(row, "full_foot_support") for row in rows),
        "full_foot_support_gt_035_frac": mean(
            1.0 if as_float(row, "full_foot_support") >= 0.35 else 0.0 for row in rows
        ),
        "body_collision_mean": mean(as_float(row, "body_collision") for row in rows),
        "rotation_penalty_frac": mean(1.0 if as_float(row, "rotation_penalty") >= 0.5 else 0.0 for row in rows),
        "toe_only_low_support_frac": mean(1.0 if toe_only_low_support(row) else 0.0 for row in rows),
        "toe_only_launch_frac": mean(1.0 if toe_only_launch(row) else 0.0 for row in rows),
        "full_support_drive_frac": mean(1.0 if full_support_drive(row) else 0.0 for row in rows),
        "phase_falling_frac": phases["falling"] / len(rows),
        "phase_rear_frac": phases["rear"] / len(rows),
        "phase_prone_frac": phases["prone"] / len(rows),
        "phase_hop_frac": phases["hop"] / len(rows),
        "phase_upright_frac": phases["upright"] / len(rows),
        "phase_rising_frac": phases["rising"] / len(rows),
        "phase_other_frac": phases["other"] / len(rows),
    }


def episode_id(path: Path) -> int:
    stem = path.stem
    if "_" in stem:
        return int(stem.rsplit("_", 1)[1])
    return 0


def seed_id(path: Path) -> int | None:
    parts = path.parts
    try:
        idx = parts.index("hopper-hop")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return None


def audit_group(label: str, trace_dirs: list[Path]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    all_rows: list[dict[str, str]] = []
    for trace_dir in trace_dirs:
        for path in sorted(trace_dir.glob("episode_*.csv")):
            rows = read_csv(path)
            all_rows.extend(rows)
            metrics = summarize_rows(rows)
            metrics.update(
                {
                    "seed": seed_id(path),
                    "episode": episode_id(path),
                    "episode_reward": sum(as_float(row, "reward") for row in rows),
                    "trace_csv": str(path),
                }
            )
            episodes.append(metrics)
    rewards = [float(ep["episode_reward"]) for ep in episodes]
    overall = summarize_rows(all_rows)
    overall.update(
        {
            "num_episodes": len(episodes),
            "episode_reward_mean": mean(rewards) if rewards else 0.0,
            "episode_reward_max": max(rewards) if rewards else 0.0,
            "episode_reward_min": min(rewards) if rewards else 0.0,
            "episode_reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        }
    )
    return {
        "label": label,
        "trace_dirs": [str(path) for path in trace_dirs],
        "overall": overall,
        "episodes": sorted(episodes, key=lambda item: float(item["episode_reward"]), reverse=True),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    groups = payload["groups"]
    metrics = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_std",
        "structural_success_frac",
        "dense_pose_high_frac",
        "continue_score_ge_060_frac",
        "near_good_frac",
        "positive_reward_frac",
        "mean_continue_score",
        "mean_vx",
        "mean_abs_pitch",
        "mean_abs_omega",
        "body_collision_mean",
        "mean_full_foot_support",
        "toe_only_low_support_frac",
        "toe_only_launch_frac",
        "full_support_drive_frac",
        "phase_prone_frac",
        "phase_hop_frac",
        "phase_rear_frac",
        "phase_rising_frac",
        "phase_upright_frac",
    ]
    lines = ["# Hopper Posture Comparison", ""]
    lines += ["## Group Summary", "", "| metric | " + " | ".join(g["label"] for g in groups) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in groups) + "|")
    for metric in metrics:
        lines.append(
            "| {metric} | {values} |".format(
                metric=metric,
                values=" | ".join(f"{g['overall'].get(metric, 0.0):.6f}" for g in groups),
            )
        )
    lines += ["", "## Top Episodes", ""]
    for group in groups:
        lines += [
            f"### {group['label']}",
            "",
            "| seed | episode | reward | structural | near_good | prone | full_foot | toe_launch | trace |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for ep in group["episodes"][:10]:
            lines.append(
                f"| {ep.get('seed')} | {ep.get('episode')} | {ep.get('episode_reward', 0.0):.6f} | "
                f"{ep.get('structural_success_frac', 0.0):.3f} | {ep.get('near_good_frac', 0.0):.3f} | "
                f"{ep.get('phase_prone_frac', 0.0):.3f} | {ep.get('mean_full_foot_support', 0.0):.4f} | "
                f"{ep.get('toe_only_launch_frac', 0.0):.3f} | `{ep.get('trace_csv')}` |"
            )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        nargs=2,
        metavar=("LABEL", "TRACE_DIR"),
        required=True,
        help="Add one trace directory to a named group. Repeat for multiple seeds.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[Path]] = {}
    for label, trace_dir in args.group:
        grouped.setdefault(label, []).append(Path(trace_dir))
    payload = {
        "audit_family": "hopper-posture-comparison-v0",
        "groups": [audit_group(label, paths) for label, paths in grouped.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output, payload)
    print(f"wrote posture comparison: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    for group in payload["groups"]:
        overall = group["overall"]
        print(
            f"{group['label']}: episodes={overall['num_episodes']} "
            f"mean_reward={overall['episode_reward_mean']:.3f} "
            f"near_good={overall['near_good_frac']:.4f} "
            f"toe_launch={overall['toe_only_launch_frac']:.4f}"
        )


if __name__ == "__main__":
    main()
