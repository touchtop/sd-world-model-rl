#!/usr/bin/env python3
"""Audit final Hopper v8 formal 100k trajectories against online rules.

The script is intentionally read-only with respect to experiment logs. It
summarizes posture fragments and checks whether the final online rule library
matches the final official-eval trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RUN = "tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019"
DEFAULT_RESULT_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_100k_empty_seed123_20260508"
)
DEFAULT_RULE_LIBRARY = DEFAULT_RESULT_ROOT / "cycle_020/rules/library_closed_loop_cycle_020.json"
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "hopper_v8formal_100k_empty_seed123_final_audit_20260508.json"
)

JOINT_TO_ACTION = {"waist": "action_0", "hip": "action_1", "knee": "action_2", "ankle": "action_3"}
ONLINE_STATUSES = {"v8_replay_safe_candidate"}


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def omega_bin(row: dict[str, str]) -> str:
    omega = as_float(row, "pitch_rate")
    if omega > 1.0:
        return "omega_pos"
    if omega < -1.0:
        return "omega_neg"
    return "omega_mid"


def vx_bin(row: dict[str, str]) -> str:
    vx = as_float(row, "vx")
    if vx >= 0.8:
        return "vx_high"
    if vx < 0.2:
        return "vx_low"
    return "vx_mid"


def support_bin(row: dict[str, str]) -> str:
    support = as_float(row, "full_foot_support")
    if support < 0.1:
        return "support_none"
    if support < 0.35:
        return "support_low"
    return "support_some"


def phase(row: dict[str, str]) -> str:
    falling = as_float(row, "falling_air") >= 0.5 or as_float(row, "falling_air_gt_05") >= 0.5
    rear = as_float(row, "grounded_rear") >= 0.5 or as_float(row, "grounded_rear_gt_05") >= 0.5
    prone = as_float(row, "grounded_prone") >= 0.5 or as_float(row, "grounded_prone_gt_05") >= 0.5
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
    if prone:
        return "prone"
    if upright:
        return "upright"
    if rising:
        return "rising"
    return "other"


def substate_key(row: dict[str, str], event: str) -> str:
    return "|".join([event, omega_bin(row), vx_bin(row), support_bin(row), phase(row)])


def rotation_risk(row: dict[str, str]) -> bool:
    return as_float(row, "rotation_penalty") >= 0.5 or abs(as_float(row, "pitch_rate")) >= 3.0


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


def rule_id(rule: dict[str, Any]) -> str:
    return f"{rule.get('substate_key')}->prefer:{rule.get('action_family')}"


def load_online_rules(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        rule
        for rule in payload.get("rules", [])
        if rule.get("status") in ONLINE_STATUSES
        and rule.get("source") == "v4_substate"
        and rule.get("intervention_type") == "soft_prefer_action_family"
    ]


def rule_matches(rule: dict[str, Any], row: dict[str, str]) -> bool:
    key = str(rule.get("substate_key", ""))
    event = key.split("|", 1)[0] if "|" in key else str(rule.get("state_event", ""))
    if event == "rotation_risk" and not rotation_risk(row):
        return False
    return substate_key(row, event) == key


def action_alignment(row: dict[str, str], family: str, threshold: float) -> int:
    if "_" not in family:
        return 0
    joint, sign = family.split("_", 1)
    col = JOINT_TO_ACTION.get(joint)
    if col is None:
        return 0
    signed = as_float(row, col) * (1.0 if sign == "pos" else -1.0)
    if signed >= threshold:
        return 1
    if signed <= -threshold:
        return -1
    return 0


def high_torque(row: dict[str, str]) -> bool:
    return max(abs(as_float(row, f"action_{idx}")) for idx in range(4)) >= 0.65


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "steps": float(len(rows)),
        "reward_sum": sum(as_float(row, "reward") for row in rows),
        "mean_continue_score": mean(continuation_score(row) for row in rows),
        "structural_success_frac": mean(1.0 if structural_success(row) else 0.0 for row in rows),
        "dense_pose_high_frac": mean(1.0 if as_float(row, "dense_pose_potential") >= 0.65 else 0.0 for row in rows),
        "positive_reward_frac": mean(1.0 if as_float(row, "reward") > 0.0 else 0.0 for row in rows),
        "mean_torso_z": mean(as_float(row, "torso_z") for row in rows),
        "mean_abs_pitch": mean(abs(as_float(row, "pitch")) for row in rows),
        "mean_vx": mean(as_float(row, "vx") for row in rows),
        "mean_abs_omega": mean(abs(as_float(row, "pitch_rate")) for row in rows),
        "mean_support_contact": mean(as_float(row, "support_contact") for row in rows),
        "mean_full_foot_support": mean(as_float(row, "full_foot_support") for row in rows),
        "rotation_penalty_frac": mean(1.0 if as_float(row, "rotation_penalty") >= 0.5 else 0.0 for row in rows),
        "rear_frac": mean(1.0 if phase(row) == "rear" else 0.0 for row in rows),
        "prone_frac": mean(1.0 if phase(row) == "prone" else 0.0 for row in rows),
        "upright_frac": mean(1.0 if phase(row) == "upright" else 0.0 for row in rows),
    }


def contiguous_fragments(rows: list[dict[str, str]], min_len: int) -> list[dict[str, Any]]:
    flags = [structural_success(row) or as_float(row, "dense_pose_potential") >= 0.65 or as_float(row, "reward") > 0.0 for row in rows]
    fragments: list[dict[str, Any]] = []
    start: int | None = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        end_reached = idx == len(flags) - 1
        if start is not None and ((not flag) or end_reached):
            end = idx if flag and end_reached else idx - 1
            if end - start + 1 >= min_len:
                window = rows[start : end + 1]
                fragments.append({"start": start, "end": end, "length": end - start + 1, "metrics": summarize_rows(window)})
            start = None
    return fragments


def trace_files(run_name: str, seeds: list[int]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for seed in seeds:
        d = Path(f"logs/hopper-hop/{seed}/{run_name}/official_eval/step_0100019/episode_csv")
        out.extend((seed, path) for path in sorted(d.glob("episode_*.csv")))
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rules = load_online_rules(args.rule_library)
    rule_stats: dict[str, Counter[str]] = {rule_id(rule): Counter() for rule in rules}
    rule_by_episode: dict[str, Counter[str]] = defaultdict(Counter)
    episode_rows: list[dict[str, Any]] = []
    top_fragments: list[dict[str, Any]] = []
    all_rows: list[dict[str, str]] = []
    good_episode_trace_rows: list[dict[str, str]] = []

    for seed, path in trace_files(args.run_name, args.seeds):
        rows = read_csv(path)
        all_rows.extend(rows)
        reward = sum(as_float(row, "reward") for row in rows)
        if reward >= args.good_episode_reward:
            good_episode_trace_rows.extend(rows)
        episode = int(path.stem.split("_", 1)[1])
        matched_steps = 0
        aligned_steps = 0
        opposed_steps = 0
        for row in rows:
            row_matched = False
            for rule in rules:
                if not rule_matches(rule, row):
                    continue
                rid = rule_id(rule)
                row_matched = True
                rule_stats[rid]["state_hits"] += 1
                if str(rule.get("substate_key", "")).startswith("unsupported_high_torque") and high_torque(row):
                    rule_stats[rid]["high_torque_hits"] += 1
                align = action_alignment(row, str(rule.get("action_family", "")), args.action_threshold)
                if align > 0:
                    rule_stats[rid]["aligned"] += 1
                    aligned_steps += 1
                elif align < 0:
                    rule_stats[rid]["opposed"] += 1
                    opposed_steps += 1
                else:
                    rule_stats[rid]["neutral"] += 1
                rule_by_episode[f"seed{seed}_ep{episode:02d}"][rid] += 1
            if row_matched:
                matched_steps += 1

        fragments = contiguous_fragments(rows, args.min_fragment_len)
        for fragment in fragments:
            item = dict(fragment)
            item.update({"seed": seed, "episode": episode, "episode_reward": reward, "trace_csv": str(path)})
            top_fragments.append(item)
        row_summary = summarize_rows(rows)
        row_summary.update(
            {
                "seed": seed,
                "episode": episode,
                "episode_reward": reward,
                "matched_rule_step_frac": matched_steps / len(rows) if rows else 0.0,
                "aligned_rule_steps": aligned_steps,
                "opposed_rule_steps": opposed_steps,
                "trace_csv": str(path),
            }
        )
        episode_rows.append(row_summary)

    for rid, stats in rule_stats.items():
        hits = stats["state_hits"]
        stats["alignment_rate_pct"] = round(100.0 * stats["aligned"] / hits, 3) if hits else 0.0
        stats["opposed_rate_pct"] = round(100.0 * stats["opposed"] / hits, 3) if hits else 0.0
        stats["neutral_rate_pct"] = round(100.0 * stats["neutral"] / hits, 3) if hits else 0.0

    top_fragments.sort(
        key=lambda item: (
            item["metrics"].get("reward_sum", 0.0)
            + 5.0 * item["metrics"].get("structural_success_frac", 0.0)
            + item["length"] * 0.01
        ),
        reverse=True,
    )
    good_episodes = [row for row in episode_rows if row["episode_reward"] >= args.good_episode_reward]
    weak_episodes = [row for row in episode_rows if row["episode_reward"] < args.good_episode_reward]
    return {
        "audit_family": "hopper-v8formal-100k-final-trajectory-rule-audit",
        "run_name": args.run_name,
        "rule_library": str(args.rule_library),
        "seeds": args.seeds,
        "num_rules": len(rules),
        "online_rules": [rule_id(rule) for rule in rules],
        "num_eval_episodes": len(episode_rows),
        "overall": summarize_rows(all_rows),
        "good_episode_threshold": args.good_episode_reward,
        "good_episode_summary": summarize_rows(good_episode_trace_rows),
        "episode_summaries": sorted(episode_rows, key=lambda row: row["episode_reward"], reverse=True),
        "good_episodes": sorted(good_episodes, key=lambda row: row["episode_reward"], reverse=True),
        "weak_episode_count": len(weak_episodes),
        "rule_stats": {rid: dict(stats) for rid, stats in rule_stats.items()},
        "rule_hits_by_episode": {key: dict(value) for key, value in sorted(rule_by_episode.items())},
        "top_fragments": top_fragments[: args.max_fragments],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper v8 Formal 100k Final Audit",
        "",
        f"Run: `{payload['run_name']}`",
        f"Rules: `{payload['num_rules']}`",
        f"Eval episodes: `{payload['num_eval_episodes']}`",
        "",
        "## Online Rules",
        "",
    ]
    for rule in payload["online_rules"]:
        lines.append(f"- `{rule}`")
    overall = payload["overall"]
    lines += [
        "",
        "## Overall Trajectory Summary",
        "",
        f"- Reward sum: `{overall.get('reward_sum', 0.0):.6f}`",
        f"- Structural success frac: `{overall.get('structural_success_frac', 0.0):.4f}`",
        f"- Dense pose high frac: `{overall.get('dense_pose_high_frac', 0.0):.4f}`",
        f"- Mean continue score: `{overall.get('mean_continue_score', 0.0):.4f}`",
        f"- Rear / prone / upright frac: `{overall.get('rear_frac', 0.0):.4f}` / `{overall.get('prone_frac', 0.0):.4f}` / `{overall.get('upright_frac', 0.0):.4f}`",
        "",
        "## Top Episodes",
        "",
        "| seed | episode | reward | structural | dense | continue | matched_rule_frac | trace |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["episode_summaries"][:10]:
        lines.append(
            f"| {row['seed']} | {row['episode']} | {row['episode_reward']:.6f} | "
            f"{row.get('structural_success_frac', 0.0):.3f} | {row.get('dense_pose_high_frac', 0.0):.3f} | "
            f"{row.get('mean_continue_score', 0.0):.3f} | {row.get('matched_rule_step_frac', 0.0):.3f} | "
            f"`{row['trace_csv']}` |"
        )
    lines += ["", "## Rule Hit Statistics", "", "| rule | hits | aligned | opposed | neutral | align% | oppose% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for rid, stats in payload["rule_stats"].items():
        lines.append(
            f"| `{rid}` | {stats.get('state_hits', 0)} | {stats.get('aligned', 0)} | "
            f"{stats.get('opposed', 0)} | {stats.get('neutral', 0)} | "
            f"{stats.get('alignment_rate_pct', 0.0):.3f} | {stats.get('opposed_rate_pct', 0.0):.3f} |"
        )
    lines += ["", "## Top Posture Fragments", ""]
    for idx, frag in enumerate(payload["top_fragments"][:12], start=1):
        m = frag["metrics"]
        lines += [
            f"### {idx}. seed={frag['seed']} episode={frag['episode']} frames={frag['start']}-{frag['end']}",
            "",
            f"- Episode reward: `{frag['episode_reward']:.6f}`",
            f"- Length: `{frag['length']}`",
            f"- Reward sum: `{m.get('reward_sum', 0.0):.6f}`",
            f"- Structural / dense / positive reward frac: `{m.get('structural_success_frac', 0.0):.3f}` / `{m.get('dense_pose_high_frac', 0.0):.3f}` / `{m.get('positive_reward_frac', 0.0):.3f}`",
            f"- Mean torso_z / |pitch| / vx / |omega|: `{m.get('mean_torso_z', 0.0):.3f}` / `{m.get('mean_abs_pitch', 0.0):.3f}` / `{m.get('mean_vx', 0.0):.3f}` / `{m.get('mean_abs_omega', 0.0):.3f}`",
            f"- Trace: `{frag['trace_csv']}`",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rule-library", type=Path, default=DEFAULT_RULE_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-threshold", type=float, default=0.2)
    parser.add_argument("--min-fragment-len", type=int, default=3)
    parser.add_argument("--max-fragments", type=int, default=40)
    parser.add_argument("--good-episode-reward", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote audit: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(f"rules: {payload['num_rules']} episodes: {payload['num_eval_episodes']}")


if __name__ == "__main__":
    main()
