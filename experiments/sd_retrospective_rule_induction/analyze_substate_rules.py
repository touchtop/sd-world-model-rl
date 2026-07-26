#!/usr/bin/env python3
"""Split coarse state events into substates and mine action-family preferences."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_COUNTERFACTUALS_V2, DEFAULT_SUBSTATE_RULES_V4, f, load_json, mean_or_zero, read_csv, write_json


def action_family(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def row_context_for_record(record: dict[str, Any]) -> tuple[dict[str, str], dict[str, str] | None]:
    rows = read_csv(Path(record["trace_csv"]))
    target_step = int(record["step"])
    for idx, row in enumerate(rows):
        if int(f(row, "step")) == target_step:
            return row, rows[idx - 1] if idx > 0 else None
    idx = min(max(target_step, 0), len(rows) - 1)
    return rows[idx], rows[idx - 1] if idx > 0 else None


def simple_phase(row: dict[str, str] | None) -> str:
    if row is None:
        return "unknown"
    falling = f(row, "falling_air") >= 0.5
    rising = f(row, "rising_air") >= 0.5
    rear = f(row, "grounded_rear") >= 0.5
    prone = f(row, "grounded_prone") >= 0.5
    upright = f(row, "grounded_upright") >= 0.5 or f(row, "grounded_prepared") >= 0.5
    return (
        "air_falling"
        if falling
        else "air_rising"
        if rising
        else "ground_rear"
        if rear
        else "ground_front"
        if prone
        else "ground_upright"
        if upright
        else "other"
    )


def substate_key(record: dict[str, Any], row: dict[str, str], prev_row: dict[str, str] | None = None) -> str:
    pitch_rate = f(row, "pitch_rate")
    vx = f(row, "vx")
    full_foot = f(row, "full_foot_support")
    toe = f(row, "touch_toe")
    heel = f(row, "touch_heel")

    omega_bin = "omega_pos" if pitch_rate > 1.0 else "omega_neg" if pitch_rate < -1.0 else "omega_mid"
    coarse_omega_bin = "omega_neg" if omega_bin == "omega_neg" else "omega_nonneg"
    vx_bin = "vx_high" if vx >= 0.8 else "vx_low" if vx < 0.2 else "vx_mid"
    if full_foot >= 0.35:
        support_bin = "support_full"
    elif max(toe, heel) >= 0.2:
        support_bin = "support_partial"
    else:
        support_bin = "support_none"
    return (
        f"prev={simple_phase(prev_row)}|cur={simple_phase(row)}|"
        f"omega={coarse_omega_bin}|vx={vx_bin}|support={support_bin}"
    )


def mine_substate_rules(
    counterfactuals: dict[str, Any],
    events: set[str],
    min_support: int,
    min_advantage: float,
    max_good_share: float,
    candidate_min_support: int,
    candidate_min_advantage: float,
    candidate_max_good_share: float,
    max_rules: int,
    max_rules_per_substate: int,
    max_rules_per_event_action: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_cache: dict[tuple[str, int], tuple[dict[str, str], dict[str, str] | None]] = {}
    for rec in counterfactuals["records"]:
        if rec["event"] not in events:
            continue
        cache_key = (rec["trace_csv"], int(rec["step"]))
        context = row_cache.get(cache_key)
        if context is None:
            context = row_context_for_record(rec)
            row_cache[cache_key] = context
        row, prev_row = context
        rec_out = dict(rec)
        rec_out["substate_key"] = substate_key(rec, row, prev_row)
        rec_out["best_action_family"] = action_family(rec["best_action"])
        groups[rec_out["substate_key"]].append(rec_out)

    rules: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for key, records in sorted(groups.items()):
        action_counts = Counter(r["best_action_family"] for r in records)
        label_counts = Counter(r["label"] for r in records)
        total = len(records)
        diagnostics[key] = {
            "count": total,
            "action_distribution": dict(action_counts),
            "label_distribution": dict(label_counts),
            "mean_advantage": mean_or_zero(float(r["advantage_over_actual"]) for r in records),
        }
        for family, support in action_counts.items():
            matched = [r for r in records if r["best_action_family"] == family]
            mean_adv = mean_or_zero(float(r["advantage_over_actual"]) for r in matched)
            good_share = sum(1 for r in matched if r["label"] == "good") / support if support else 0.0
            passes_supported = support >= min_support and mean_adv >= min_advantage and good_share <= max_good_share
            passes_candidate = (
                support >= candidate_min_support
                and mean_adv >= candidate_min_advantage
                and good_share <= candidate_max_good_share
            )
            if not passes_supported and not passes_candidate:
                continue
            failed_checks: list[str] = []
            if support < min_support:
                failed_checks.append("support_below_online_threshold")
            if mean_adv < min_advantage:
                failed_checks.append("advantage_below_online_threshold")
            if good_share > max_good_share:
                failed_checks.append("good_share_above_online_threshold")
            rules.append(
                {
                    "id": f"{key}->prefer:{family}",
                    "substate_key": key,
                    "state_event": "abstract_transition_context",
                    "base_event": records[0]["event"],
                    "substate_predicate": key,
                    "rule_role": "abstract_transition_context",
                    "action_family": family,
                    "intervention_type": "soft_prefer_action_family",
                    "support": support,
                    "substate_support": total,
                    "substate_coverage": support / total if total else 0.0,
                    "mean_advantage": mean_adv,
                    "good_share": good_share,
                    "status": "cycle_supported" if passes_supported else "cycle_candidate",
                    "failed_online_checks": failed_checks,
                    "candidate_thresholds": {
                        "min_support": candidate_min_support,
                        "min_advantage": candidate_min_advantage,
                        "max_good_share": candidate_max_good_share,
                    },
                    "online_thresholds": {
                        "min_support": min_support,
                        "min_advantage": min_advantage,
                        "max_good_share": max_good_share,
                    },
                    "label_counts": dict(Counter(r["label"] for r in matched)),
                    "example_records": [
                        {
                            "episode_id": r["episode_id"],
                            "step": r["step"],
                            "label": r["label"],
                            "best_action": r["best_action"],
                            "advantage_over_actual": r["advantage_over_actual"],
                        }
                        for r in sorted(matched, key=lambda x: x["advantage_over_actual"], reverse=True)[:5]
                    ],
                }
            )

    rules.sort(key=lambda r: (r["mean_advantage"], r["support"]), reverse=True)
    pre_compression_rules = len(rules)
    if max_rules > 0 or max_rules_per_substate > 0 or max_rules_per_event_action > 0:
        kept: list[dict[str, Any]] = []
        per_substate: Counter[str] = Counter()
        per_event_action: Counter[tuple[str, str]] = Counter()
        for rule in rules:
            substate = str(rule.get("substate_key"))
            event_action = (str(rule.get("state_event")), str(rule.get("action_family")))
            if max_rules_per_substate > 0 and per_substate[substate] >= max_rules_per_substate:
                continue
            if max_rules_per_event_action > 0 and per_event_action[event_action] >= max_rules_per_event_action:
                continue
            kept.append(rule)
            per_substate[substate] += 1
            per_event_action[event_action] += 1
            if max_rules > 0 and len(kept) >= max_rules:
                break
        rules = kept
    return {
        "rule_family": "substate-action-rules-v4",
        "source_counterfactuals": counterfactuals.get("model_checkpoint"),
        "events": sorted(events),
        "min_support": min_support,
        "min_advantage": min_advantage,
        "max_good_share": max_good_share,
        "candidate_min_support": candidate_min_support,
        "candidate_min_advantage": candidate_min_advantage,
        "candidate_max_good_share": candidate_max_good_share,
        "num_substates": len(groups),
        "num_rules": len(rules),
        "num_rules_pre_compression": pre_compression_rules,
        "compression": {
            "max_rules": max_rules,
            "max_rules_per_substate": max_rules_per_substate,
            "max_rules_per_event_action": max_rules_per_event_action,
            "removed_rules": pre_compression_rules - len(rules),
        },
        "num_cycle_candidates": sum(1 for r in rules if r.get("status") == "cycle_candidate"),
        "num_cycle_supported": sum(1 for r in rules if r.get("status") == "cycle_supported"),
        "rules": rules,
        "diagnostics": diagnostics,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Substate Rules v4",
        "",
        f"Substates: `{payload['num_substates']}`",
        f"Rules: `{payload['num_rules']}`",
        f"Rules before compression: `{payload.get('num_rules_pre_compression', payload['num_rules'])}`",
        f"Cycle candidates: `{payload.get('num_cycle_candidates', 0)}`",
        f"Cycle supported: `{payload.get('num_cycle_supported', 0)}`",
        f"Compression: `{payload.get('compression', {})}`",
        "",
    ]
    for rule in payload["rules"]:
        lines += [
            f"## {rule['id']}",
            "",
            f"- Status: `{rule.get('status')}`",
            f"- Substate: `{rule['substate_key']}`",
            f"- Action family: `{rule['action_family']}`",
            f"- Support: `{rule['support']}` / `{rule['substate_support']}`",
            f"- Mean advantage: `{rule['mean_advantage']:.4f}`",
            f"- Good share: `{rule['good_share']:.3f}`",
            f"- Label counts: `{rule['label_counts']}`",
            f"- Failed online checks: `{rule.get('failed_online_checks', [])}`",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactuals", type=Path, default=DEFAULT_COUNTERFACTUALS_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBSTATE_RULES_V4)
    parser.add_argument("--event", action="append", default=["unsupported_high_torque", "rotation_risk"])
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-advantage", type=float, default=0.12)
    parser.add_argument("--max-good-share", type=float, default=0.25)
    parser.add_argument("--candidate-min-support", type=int, default=2)
    parser.add_argument("--candidate-min-advantage", type=float, default=0.02)
    parser.add_argument("--candidate-max-good-share", type=float, default=0.50)
    parser.add_argument("--max-rules", type=int, default=48)
    parser.add_argument("--max-rules-per-substate", type=int, default=2)
    parser.add_argument("--max-rules-per-event-action", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = mine_substate_rules(
        load_json(args.counterfactuals),
        set(args.event),
        args.min_support,
        args.min_advantage,
        args.max_good_share,
        args.candidate_min_support,
        args.candidate_min_advantage,
        args.candidate_max_good_share,
        args.max_rules,
        args.max_rules_per_substate,
        args.max_rules_per_event_action,
    )
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote substate rules: {args.output}")
    print(f"rules: {payload['num_rules']} substates: {payload['num_substates']}")


if __name__ == "__main__":
    main()
