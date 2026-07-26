#!/usr/bin/env python3
"""Compare rule-on/off eval traces for the Hopper stage-constraint rule.

This read-only audit focuses on post-trigger structure rather than immediate
reward. It uses the same toe-recovery trigger predicate as the online prior and
checks whether the following short horizon moves toward heel/full-foot support
without losing support, entering falling-air states, or increasing rotation
risk.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import as_float, row_float, row_parts, stage_metric


DEFAULT_ROOT = Path("logs/hopper-hop")
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "hopper_185k_stageconstraint_evalonly_20260514/post_trigger_structure_compare.json"
)
RUN_TEMPLATE = "tdmpc2_stageconstraint_evalonly_185k_seed{seed}_rule_{mode}_20260514_r4"
DEFAULT_STEP = "step_0185037"
TRIGGER_KEY = "toe_recovery_recoverable_low_support"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def trace_paths(seed: int, mode: str, run_template: str, step: str) -> list[Path]:
    root = DEFAULT_ROOT / str(seed) / run_template.format(seed=seed, mode=mode) / "official_eval" / step / "episode_csv"
    return sorted(root.glob("episode_*.csv"))


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def event_summary(rows: list[dict[str, str]], idx: int, horizon: int) -> dict[str, float | int | bool]:
    current = rows[idx]
    end = min(len(rows) - 1, idx + horizon)
    future = rows[end]
    window = rows[idx + 1 : end + 1] or [future]

    current_support = stage_metric(current, "support_contact")
    terminal_support = stage_metric(future, "support_contact")
    min_support = min(stage_metric(row, "support_contact") for row in window)
    current_heel = stage_metric(current, "touch_heel")
    terminal_heel = stage_metric(future, "touch_heel")
    current_full = stage_metric(current, "full_foot_support")
    terminal_full = stage_metric(future, "full_foot_support")
    max_falling = max(stage_metric(row, "falling_air") for row in window)
    max_rotation = max(stage_metric(row, "rotation_penalty") for row in window)
    max_pitch_rate_abs = max(abs(as_float(row, "pitch_rate")) for row in window)
    terminal_torso_z = row_float(future, "torso_z", "rootz", "qpos_rootz")
    terminal_vx = as_float(future, "vx")

    heel_progress = terminal_heel - current_heel
    full_progress = terminal_full - current_full
    support_progress = terminal_support - current_support
    support_not_lost = min_support >= max(0.0, current_support - 0.05)
    min_support_ok = min_support >= 0.55
    terminal_support_ok = terminal_support >= 0.65
    heel_progress_ok = heel_progress >= 0.05
    full_foot_progress_ok = full_progress >= 0.03
    falling_ok = max_falling <= 0.25
    rotation_ok = max_rotation <= 0.35
    rule_constraint_satisfied = (
        min_support_ok
        and terminal_support_ok
        and heel_progress_ok
        and full_foot_progress_ok
        and falling_ok
        and rotation_ok
    )
    expected_good = (
        support_not_lost
        and heel_progress >= 0.05
        and full_progress >= 0.03
        and max_falling <= max(0.25, stage_metric(current, "falling_air") + 0.05)
        and max_rotation <= max(0.35, stage_metric(current, "rotation_penalty") + 0.05)
    )

    return {
        "idx": idx,
        "horizon_end": end,
        "current_support": current_support,
        "terminal_support": terminal_support,
        "min_support": min_support,
        "support_progress": support_progress,
        "heel_progress": heel_progress,
        "full_foot_progress": full_progress,
        "max_falling_air": max_falling,
        "max_rotation_penalty": max_rotation,
        "max_abs_pitch_rate": max_pitch_rate_abs,
        "terminal_torso_z": terminal_torso_z,
        "terminal_vx": terminal_vx,
        "support_not_lost": support_not_lost,
        "min_support_ok": min_support_ok,
        "terminal_support_ok": terminal_support_ok,
        "heel_progress_ok": heel_progress_ok,
        "full_foot_progress_ok": full_foot_progress_ok,
        "falling_ok": falling_ok,
        "rotation_ok": rotation_ok,
        "rule_constraint_satisfied": rule_constraint_satisfied,
        "expected_good": expected_good,
    }


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"trigger_count": len(events)}
    numeric_keys = [
        "current_support",
        "terminal_support",
        "min_support",
        "support_progress",
        "heel_progress",
        "full_foot_progress",
        "max_falling_air",
        "max_rotation_penalty",
        "max_abs_pitch_rate",
        "terminal_torso_z",
        "terminal_vx",
    ]
    bool_keys = [
        "support_not_lost",
        "min_support_ok",
        "terminal_support_ok",
        "heel_progress_ok",
        "full_foot_progress_ok",
        "falling_ok",
        "rotation_ok",
        "rule_constraint_satisfied",
        "expected_good",
    ]
    for key in numeric_keys:
        vals = [float(ev[key]) for ev in events]
        out[f"mean_{key}"] = safe_mean(vals)
    for key in bool_keys:
        out[f"{key}_rate"] = safe_mean([1.0 if ev[key] else 0.0 for ev in events])
    return out


def audit_trace(path: Path, horizon: int, continue_score_min: float) -> dict[str, Any]:
    rows = read_rows(path)
    episode = int(path.stem.split("_", 1)[1])
    reward = sum(as_float(row, "reward") for row in rows)
    events = []
    for idx, row in enumerate(rows):
        parts = row_parts(row, continue_score_min)
        if parts.get(TRIGGER_KEY) != "1":
            continue
        events.append(event_summary(rows, idx, horizon))
    summary = summarize_events(events)
    summary.update(
        {
            "episode": episode,
            "episode_reward": reward,
            "steps": len(rows),
            "trace_csv": str(path),
        }
    )
    return summary


def aggregate_episode_summaries(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total_triggers = sum(int(ep["trigger_count"]) for ep in episodes)
    out: dict[str, Any] = {
        "episodes": len(episodes),
        "total_triggers": total_triggers,
        "mean_episode_reward": safe_mean([float(ep["episode_reward"]) for ep in episodes]),
        "max_episode_reward": max([float(ep["episode_reward"]) for ep in episodes], default=None),
        "triggered_episodes": sum(1 for ep in episodes if int(ep["trigger_count"]) > 0),
    }
    metric_keys = [
        "mean_current_support",
        "mean_terminal_support",
        "mean_min_support",
        "mean_support_progress",
        "mean_heel_progress",
        "mean_full_foot_progress",
        "mean_max_falling_air",
        "mean_max_rotation_penalty",
        "mean_max_abs_pitch_rate",
        "mean_terminal_torso_z",
        "mean_terminal_vx",
        "support_not_lost_rate",
        "min_support_ok_rate",
        "terminal_support_ok_rate",
        "heel_progress_ok_rate",
        "full_foot_progress_ok_rate",
        "falling_ok_rate",
        "rotation_ok_rate",
        "rule_constraint_satisfied_rate",
        "expected_good_rate",
    ]
    for key in metric_keys:
        weighted = [
            (float(ep[key]), int(ep["trigger_count"]))
            for ep in episodes
            if ep.get(key) is not None and int(ep["trigger_count"]) > 0
        ]
        denom = sum(weight for _, weight in weighted)
        out[key] = sum(value * weight for value, weight in weighted) / denom if denom else None
    return out


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    by_seed_mode: dict[str, dict[str, Any]] = {}
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode_templates = parse_mode_templates(args.mode_run_template)
    for seed in args.seeds:
        for mode in args.modes:
            run_template = mode_templates.get(mode, args.run_template)
            episodes = [
                audit_trace(path, args.horizon, args.continue_score_min)
                for path in trace_paths(seed, mode, run_template, args.step)
            ]
            key = f"seed{seed}_rule_{mode}"
            aggregate = aggregate_episode_summaries(episodes)
            by_seed_mode[key] = {
                "seed": seed,
                "mode": mode,
                "aggregate": aggregate,
                "episodes": sorted(episodes, key=lambda row: row["episode_reward"], reverse=True),
            }
            by_mode[mode].extend(episodes)

    mode_aggregate = {mode: aggregate_episode_summaries(episodes) for mode, episodes in by_mode.items()}
    delta: dict[str, float | None] = {}
    if args.compare_mode in mode_aggregate and args.baseline_mode in mode_aggregate:
        for key, on_value in mode_aggregate[args.compare_mode].items():
            off_value = mode_aggregate[args.baseline_mode].get(key)
            if isinstance(on_value, (int, float)) and isinstance(off_value, (int, float)):
                delta[f"{key}_{args.compare_mode}_minus_{args.baseline_mode}"] = float(on_value) - float(off_value)
    return {
        "audit_family": "hopper-stage-constraint-evalonly-post-trigger-structure-v0",
        "trigger_key": TRIGGER_KEY,
        "horizon": args.horizon,
        "continue_score_min": args.continue_score_min,
        "seeds": args.seeds,
        "modes": args.modes,
        "baseline_mode": args.baseline_mode,
        "compare_mode": args.compare_mode,
        "step": args.step,
        "run_template": args.run_template,
        "mode_run_templates": mode_templates,
        "mode_aggregate": mode_aggregate,
        "mode_delta_compare_minus_baseline": delta,
        "by_seed_mode": by_seed_mode,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Stage-Constraint Eval-Only Post-Trigger Audit",
        "",
        f"Trigger: `{payload['trigger_key']}`",
        f"Horizon: `{payload['horizon']}`",
        "",
        "## Mode Aggregate",
        "",
        "| mode | episodes | triggers | reward_mean | expected_good | constraint_satisfied | heel_progress | full_progress | min_support | max_falling | max_rotation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in payload["modes"]:
        agg = payload["mode_aggregate"].get(mode, {})
        lines.append(
            "| {mode} | {episodes} | {triggers} | {reward} | {good} | {constraint} | {heel} | {full} | {support} | {falling} | {rotation} |".format(
                mode=mode,
                episodes=agg.get("episodes", 0),
                triggers=agg.get("total_triggers", 0),
                reward=format_float(agg.get("mean_episode_reward")),
                good=format_float(agg.get("expected_good_rate")),
                constraint=format_float(agg.get("rule_constraint_satisfied_rate")),
                heel=format_float(agg.get("mean_heel_progress")),
                full=format_float(agg.get("mean_full_foot_progress")),
                support=format_float(agg.get("mean_min_support")),
                falling=format_float(agg.get("mean_max_falling_air")),
                rotation=format_float(agg.get("mean_max_rotation_penalty")),
            )
        )
    lines += ["", "## Seed Breakdown", ""]
    lines += [
        "| seed/mode | triggers | reward_mean | expected_good | constraint_satisfied | heel_progress | full_progress | min_support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(payload["by_seed_mode"]):
        agg = payload["by_seed_mode"][key]["aggregate"]
        lines.append(
            "| {key} | {triggers} | {reward} | {good} | {constraint} | {heel} | {full} | {support} |".format(
                key=key,
                triggers=agg.get("total_triggers", 0),
                reward=format_float(agg.get("mean_episode_reward")),
                good=format_float(agg.get("expected_good_rate")),
                constraint=format_float(agg.get("rule_constraint_satisfied_rate")),
                heel=format_float(agg.get("mean_heel_progress")),
                full=format_float(agg.get("mean_full_foot_progress")),
                support=format_float(agg.get("mean_min_support")),
            )
        )
    lines += ["", "## Constraint Components", ""]
    lines += [
        "| mode | min_support_ok | terminal_support_ok | heel_progress_ok | full_foot_progress_ok | falling_ok | rotation_ok | support_not_lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in payload["modes"]:
        agg = payload["mode_aggregate"].get(mode, {})
        lines.append(
            "| {mode} | {min_support} | {terminal_support} | {heel} | {full} | {falling} | {rotation} | {support_not_lost} |".format(
                mode=mode,
                min_support=format_float(agg.get("min_support_ok_rate")),
                terminal_support=format_float(agg.get("terminal_support_ok_rate")),
                heel=format_float(agg.get("heel_progress_ok_rate")),
                full=format_float(agg.get("full_foot_progress_ok_rate")),
                falling=format_float(agg.get("falling_ok_rate")),
                rotation=format_float(agg.get("rotation_ok_rate")),
                support_not_lost=format_float(agg.get("support_not_lost_rate")),
            )
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def parse_mode_templates(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"mode-run-template must use mode=template form: {item}")
        mode, template = item.split("=", 1)
        out[mode] = template
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--modes", nargs="+", default=["off", "on"])
    parser.add_argument("--baseline-mode", default="off")
    parser.add_argument("--compare-mode", default="on")
    parser.add_argument("--run-template", default=RUN_TEMPLATE)
    parser.add_argument("--mode-run-template", action="append", default=[])
    parser.add_argument("--step", default=DEFAULT_STEP)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output, payload)
    print(f"wrote audit: {args.output}")
    for mode, agg in payload["mode_aggregate"].items():
        print(
            f"{mode}: episodes={agg['episodes']} triggers={agg['total_triggers']} "
            f"reward_mean={format_float(agg['mean_episode_reward'])} "
            f"expected_good={format_float(agg['expected_good_rate'])} "
            f"constraint={format_float(agg['rule_constraint_satisfied_rate'])}"
        )


if __name__ == "__main__":
    main()
