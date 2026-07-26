#!/usr/bin/env python3
"""Audit Hopper v8 transition rules on train replay traces for false-intervention risk."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import f, load_json, mean_or_zero, read_csv, row_to_action, write_json
from mine_hopper_v8_success_segments import HopperV8Adapter
from mine_hopper_v8_transition_actions import PHASE_ANY, phase_any_key, refined_key


DEFAULT_VALIDATION = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_rule_validation_200k_v72.json")
DEFAULT_EXPORTS = [
    Path(
        "results/sd_retrospective_rule_induction/closed_loop/"
        "hopper_v72_multiseed_shared_200k_seed123_resume100k_20260507_r2/"
        f"cycle_040/train_replay_exports/seed{seed}.json"
    )
    for seed in (1, 2, 3)
]
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_replay_safety_200k_v72.json")
ACTION_NAMES = ("waist", "hip", "knee", "ankle")


def action_family_from_row(row: dict[str, str], threshold: float) -> str:
    actions = row_to_action(row)
    idx = max(range(len(actions)), key=lambda i: abs(actions[i]))
    if abs(actions[idx]) < threshold:
        return "weak"
    return f"{ACTION_NAMES[idx]}_{'pos' if actions[idx] > 0 else 'neg'}"


def opposite_family(action_family: str) -> str | None:
    if action_family.endswith("_pos"):
        return action_family[:-4] + "_neg"
    if action_family.endswith("_neg"):
        return action_family[:-4] + "_pos"
    return None


def quality_bin_from_rule(rule: dict[str, Any]) -> str:
    parts = str(rule["rule_key"]).split("|")
    return parts[1] if len(parts) > 1 else str(rule.get("quality_bin", "transition_gap"))


def rule_phase_token(rule: dict[str, Any]) -> str:
    parts = str(rule["rule_key"]).split("|")
    return parts[5] if len(parts) == 7 else ""


def success_flags(adapter: HopperV8Adapter, row: dict[str, str], continue_score_min: float) -> dict[str, bool]:
    structural = adapter.structural_success(row)
    dense = f(row, "dense_pose_potential") >= adapter.dense_pose_min
    reward = f(row, "reward") > adapter.reward_min
    continue_good = adapter.continuation_score(row) >= continue_score_min
    return {
        "structural_success": structural,
        "dense_pose_high": dense,
        "positive_reward": reward,
        "continue_score_high": continue_good,
        "any_good": structural or dense or reward or continue_good,
    }


def load_trace_entries(export_jsons: list[Path], max_episodes_per_seed: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for export_path in export_jsons:
        export = json.loads(export_path.read_text())
        summary_csv = Path(export["summary_csv"])
        seed = None
        stem = export_path.stem
        if stem.startswith("seed"):
            try:
                seed = int(stem.replace("seed", ""))
            except ValueError:
                seed = None
        with summary_csv.open("r", newline="") as fobj:
            rows = list(csv.DictReader(fobj))
        rows = sorted(rows, key=lambda row: int(row["episode_idx"]))
        if max_episodes_per_seed > 0:
            rows = rows[:max_episodes_per_seed]
        for row in rows:
            entries.append(
                {
                    "seed": seed,
                    "episode": int(row["episode_idx"]),
                    "episode_reward": float(row["episode_reward"]),
                    "trace_csv": row["trace_csv"],
                    "source": row.get("source", "train_replay"),
                }
            )
    return entries


def select_rules(validation: dict[str, Any], allowed_statuses: set[str]) -> list[dict[str, Any]]:
    return [
        rule
        for rule in validation.get("rules", [])
        if rule.get("validation_status") in allowed_statuses
        and rule.get("rule_role") == "transition"
        and rule.get("intervention_type") in {"avoid_action_family", "soft_prefer_action_family"}
    ]


def init_stats(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule": rule,
        "hits": 0,
        "good_hits": 0,
        "structural_success_hits": 0,
        "dense_pose_high_hits": 0,
        "positive_reward_hits": 0,
        "continue_score_high_hits": 0,
        "risky_hits": 0,
        "bad_hits": 0,
        "actual_same_family": 0,
        "actual_opposite_family": 0,
        "good_actual_same_family": 0,
        "good_actual_opposite_family": 0,
        "episode_hits": Counter(),
        "seed_hits": Counter(),
        "phase_hits": Counter(),
        "good_episode_rewards": [],
        "hit_continue_scores": [],
        "good_hit_continue_scores": [],
    }


def audit_rules(
    rules: list[dict[str, Any]],
    trace_entries: list[dict[str, Any]],
    adapter: HopperV8Adapter,
    action_threshold: float,
    continue_score_min: float,
) -> dict[str, Any]:
    rules_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        rules_by_key[rule["rule_key"]].append(rule)
    key_specs = sorted({(rule["rule_role"], quality_bin_from_rule(rule), rule_phase_token(rule)) for rule in rules})
    stats = {rule["id"]: init_stats(rule) for rule in rules}

    total_steps = 0
    total_good_steps = 0
    trace_summaries: list[dict[str, Any]] = []
    for entry in trace_entries:
        rows = read_csv(Path(entry["trace_csv"]))
        trace_hits = 0
        trace_good_hits = 0
        for row in rows:
            total_steps += 1
            flags = success_flags(adapter, row, continue_score_min)
            if flags["any_good"]:
                total_good_steps += 1
            candidate_keys = set()
            for role, quality, phase_token in key_specs:
                candidate_keys.add(refined_key(adapter, role, quality, row))
                if phase_token == PHASE_ANY:
                    candidate_keys.add(phase_any_key(adapter, role, quality, row))
            for key in candidate_keys:
                for rule in rules_by_key.get(key, []):
                    rid = rule["id"]
                    st = stats[rid]
                    st["hits"] += 1
                    trace_hits += 1
                    if flags["any_good"]:
                        st["good_hits"] += 1
                        trace_good_hits += 1
                        st["good_episode_rewards"].append(float(entry["episode_reward"]))
                        st["good_hit_continue_scores"].append(adapter.continuation_score(row))
                    for flag_name in ("structural_success", "dense_pose_high", "positive_reward", "continue_score_high"):
                        if flags[flag_name]:
                            st[f"{flag_name}_hits"] += 1
                    if adapter.risk_bin(row) != "stable":
                        st["risky_hits"] += 1
                    if not flags["any_good"]:
                        st["bad_hits"] += 1
                    actual = action_family_from_row(row, action_threshold)
                    target = rule["action_family"]
                    opposite = opposite_family(target)
                    if actual == target:
                        st["actual_same_family"] += 1
                        if flags["any_good"]:
                            st["good_actual_same_family"] += 1
                    if opposite is not None and actual == opposite:
                        st["actual_opposite_family"] += 1
                        if flags["any_good"]:
                            st["good_actual_opposite_family"] += 1
                    st["episode_hits"][f"{entry['seed']}:{entry['episode']}"] += 1
                    st["seed_hits"][str(entry["seed"])] += 1
                    st["phase_hits"][key.split("|")[5]] += 1
                    st["hit_continue_scores"].append(adapter.continuation_score(row))
        trace_summaries.append(
            {
                "seed": entry["seed"],
                "episode": entry["episode"],
                "episode_reward": entry["episode_reward"],
                "trace_csv": entry["trace_csv"],
                "hits": trace_hits,
                "good_hits": trace_good_hits,
            }
        )

    rule_rows: list[dict[str, Any]] = []
    for rid, st in stats.items():
        hits = st["hits"]
        good_hits = st["good_hits"]
        rule = st["rule"]
        false_intervention_count = (
            st["good_actual_same_family"]
            if rule["intervention_type"] == "avoid_action_family"
            else st["good_actual_opposite_family"]
        )
        false_intervention_rate = false_intervention_count / good_hits if good_hits else 0.0
        good_hit_rate = good_hits / hits if hits else 0.0
        bad_hit_rate = st["bad_hits"] / hits if hits else 0.0
        safe_for_next_stage = hits >= 20 and good_hits > 0 and false_intervention_rate <= 0.10
        if hits == 0:
            replay_status = "replay_no_support"
        elif safe_for_next_stage:
            replay_status = "replay_safety_supported"
        elif false_intervention_rate > 0.25:
            replay_status = "replay_false_intervention_risk"
        else:
            replay_status = "replay_observe"
        rule_rows.append(
            {
                "id": rid,
                "rule_key": rule["rule_key"],
                "action_family": rule["action_family"],
                "intervention_type": rule["intervention_type"],
                "validation_status": rule.get("validation_status"),
                "heldout_strength": rule.get("heldout_strength"),
                "replay_status": replay_status,
                "hits": hits,
                "good_hits": good_hits,
                "good_hit_rate": good_hit_rate,
                "bad_hit_rate": bad_hit_rate,
                "false_intervention_count": false_intervention_count,
                "false_intervention_rate": false_intervention_rate,
                "actual_same_family_rate": st["actual_same_family"] / hits if hits else 0.0,
                "actual_opposite_family_rate": st["actual_opposite_family"] / hits if hits else 0.0,
                "good_actual_same_family_rate": st["good_actual_same_family"] / good_hits if good_hits else 0.0,
                "good_actual_opposite_family_rate": st["good_actual_opposite_family"] / good_hits if good_hits else 0.0,
                "structural_success_hit_rate": st["structural_success_hits"] / hits if hits else 0.0,
                "dense_pose_high_hit_rate": st["dense_pose_high_hits"] / hits if hits else 0.0,
                "positive_reward_hit_rate": st["positive_reward_hits"] / hits if hits else 0.0,
                "continue_score_high_hit_rate": st["continue_score_high_hits"] / hits if hits else 0.0,
                "risky_hit_rate": st["risky_hits"] / hits if hits else 0.0,
                "mean_continue_score_on_hits": mean_or_zero(st["hit_continue_scores"]),
                "mean_continue_score_on_good_hits": mean_or_zero(st["good_hit_continue_scores"]),
                "mean_episode_reward_on_good_hits": mean_or_zero(st["good_episode_rewards"]),
                "episodes_hit": len(st["episode_hits"]),
                "seed_counts": dict(st["seed_hits"]),
                "phase_counts": dict(st["phase_hits"]),
            }
        )
    rule_rows.sort(
        key=lambda item: (
            item["replay_status"] == "replay_safety_supported",
            -item["false_intervention_rate"],
            item["hits"],
        ),
        reverse=True,
    )
    return {
        "audit_family": "hopper-v8-transition-replay-safety-v0",
        "num_rules": len(rules),
        "num_traces": len(trace_entries),
        "total_steps": total_steps,
        "total_good_steps": total_good_steps,
        "total_good_step_rate": total_good_steps / total_steps if total_steps else 0.0,
        "status_counts": dict(Counter(row["replay_status"] for row in rule_rows)),
        "intervention_counts": dict(Counter(row["intervention_type"] for row in rule_rows)),
        "rules": rule_rows,
        "trace_summaries": trace_summaries,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_rules: int) -> None:
    lines = [
        "# Hopper v8 Transition Replay Safety",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Traces: `{payload['num_traces']}`",
        f"Total steps: `{payload['total_steps']}`",
        f"Good step rate: `{payload['total_good_step_rate']:.4f}`",
        f"Status counts: `{payload['status_counts']}`",
        "",
    ]
    for rule in payload["rules"][:max_rules]:
        lines.extend(
            [
                f"## `{rule['id']}`",
                "",
                f"- Replay status: `{rule['replay_status']}`",
                f"- Intervention: `{rule['intervention_type']}`",
                f"- Hits / good hits / episodes: `{rule['hits']}` / `{rule['good_hits']}` / `{rule['episodes_hit']}`",
                f"- Good hit / bad hit rate: `{rule['good_hit_rate']:.3f}` / `{rule['bad_hit_rate']:.3f}`",
                f"- False intervention rate: `{rule['false_intervention_rate']:.3f}`",
                f"- Actual same / opposite family rate: `{rule['actual_same_family_rate']:.3f}` / `{rule['actual_opposite_family_rate']:.3f}`",
                f"- Good actual same / opposite family rate: `{rule['good_actual_same_family_rate']:.3f}` / `{rule['good_actual_opposite_family_rate']:.3f}`",
                f"- Mean continue score on hits: `{rule['mean_continue_score_on_hits']:.4f}`",
                f"- Seed counts: `{rule['seed_counts']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--train-replay-export", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allowed-validation-status", action="append", default=["heldout_supported"])
    parser.add_argument("--max-episodes-per-seed", type=int, default=0)
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--torso-z-min", type=float, default=0.75)
    parser.add_argument("--abs-pitch-max", type=float, default=0.45)
    parser.add_argument("--vx-min", type=float, default=0.80)
    parser.add_argument("--abs-omega-max", type=float, default=3.50)
    parser.add_argument("--dense-pose-min", type=float, default=0.65)
    parser.add_argument("--reward-min", type=float, default=0.0)
    parser.add_argument("--max-markdown-rules", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_paths = args.train_replay_export or DEFAULT_EXPORTS
    adapter = HopperV8Adapter(
        torso_z_min=args.torso_z_min,
        abs_pitch_max=args.abs_pitch_max,
        vx_min=args.vx_min,
        abs_omega_max=args.abs_omega_max,
        dense_pose_min=args.dense_pose_min,
        reward_min=args.reward_min,
    )
    validation = load_json(args.validation)
    rules = select_rules(validation, set(args.allowed_validation_status))
    trace_entries = load_trace_entries(export_paths, args.max_episodes_per_seed)
    payload = audit_rules(
        rules=rules,
        trace_entries=trace_entries,
        adapter=adapter,
        action_threshold=args.action_threshold,
        continue_score_min=args.continue_score_min,
    )
    payload["source_validation"] = str(args.validation)
    payload["train_replay_exports"] = [str(p) for p in export_paths]
    payload["allowed_validation_statuses"] = args.allowed_validation_status
    payload["config"] = {
        "max_episodes_per_seed": args.max_episodes_per_seed,
        "action_threshold": args.action_threshold,
        "continue_score_min": args.continue_score_min,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_rules)
    print(f"wrote v8 replay safety audit: {args.output}")
    print(f"rules: {payload['num_rules']} statuses: {payload['status_counts']}")


if __name__ == "__main__":
    main()
