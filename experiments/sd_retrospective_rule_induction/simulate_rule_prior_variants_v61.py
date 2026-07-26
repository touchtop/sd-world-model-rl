#!/usr/bin/env python3
"""Audit v6.1 rule-prior variants on saved Hopper eval traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_LIBRARY = Path("results/sd_retrospective_rule_induction/rules/library_v5.json")
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit.json")
DEFAULT_TRACES = [
    "off:logs/hopper-hop/1/sd_hopper_v6_3k_rule_prior_off_seed1_finalonly_20260505/official_eval/step_0003000/episode_csv/episode_00.csv",
    "on:logs/hopper-hop/1/sd_hopper_v6_3k_rule_prior_on_seed1_finalonly_20260505/official_eval/step_0003000/episode_csv/episode_00.csv",
]
JOINT_TO_DIM = {"waist": 0, "hip": 1, "knee": 2, "ankle": 3}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def substate_key(row: dict[str, str], event: str = "unsupported_high_torque") -> str:
    pitch_rate = f(row, "pitch_rate")
    vx = f(row, "vx")
    full_foot = f(row, "full_foot_support")
    falling = f(row, "falling_air") >= 0.5
    rear = f(row, "grounded_rear") >= 0.5
    prone = f(row, "grounded_prone") >= 0.5
    upright = f(row, "grounded_upright") >= 0.5 or f(row, "grounded_prepared") >= 0.5
    omega_bin = "omega_pos" if pitch_rate > 1.0 else "omega_neg" if pitch_rate < -1.0 else "omega_mid"
    vx_bin = "vx_high" if vx >= 0.8 else "vx_low" if vx < 0.2 else "vx_mid"
    support_bin = "support_none" if full_foot < 0.1 else "support_low" if full_foot < 0.35 else "support_some"
    phase = "falling" if falling else "rear" if rear else "prone" if prone else "upright" if upright else "other"
    return f"{event}|{omega_bin}|{vx_bin}|{support_bin}|{phase}"


def action_family(row: dict[str, str]) -> str:
    names = ["waist", "hip", "knee", "ankle"]
    vals = [f(row, f"action_{i}") for i in range(4)]
    idx = max(range(4), key=lambda i: abs(vals[i]))
    if abs(vals[idx]) < 0.2:
        return "weak"
    sign = "pos" if vals[idx] >= 0 else "neg"
    return f"{names[idx]}_{sign}"


def rule_dim_sign(rule: dict[str, Any]) -> tuple[int, int]:
    family = str(rule["action_family"])
    joint, sign = family.split("_", 1)
    return JOINT_TO_DIM[joint], 1 if sign == "pos" else -1


def high_torque(row: dict[str, str]) -> bool:
    return max(abs(f(row, f"action_{i}")) for i in range(4)) >= 0.65


def variant_score(
    row: dict[str, str],
    rule: dict[str, Any],
    mode: str,
    weight: float,
    opposite_penalty: float,
    preferred_cap: float,
    cap_penalty: float,
    max_bonus: float,
) -> float:
    if not high_torque(row):
        return 0.0
    dim, sign = rule_dim_sign(rule)
    signed = f(row, f"action_{dim}") * sign
    prefer = max(signed, 0.0)
    oppose = max(-signed, 0.0)
    if mode == "prefer":
        score = weight * min(prefer, 1.0) - opposite_penalty * min(oppose, 1.0)
        score -= cap_penalty * max(min(prefer, 1.0) - preferred_cap, 0.0)
    elif mode == "penalty_only":
        score = -opposite_penalty * min(oppose, 1.0)
        score -= cap_penalty * max(min(prefer, 1.0) - preferred_cap, 0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return max(-max_bonus, min(max_bonus, score))


def audit_trace(
    name: str,
    path: Path,
    rules: list[dict[str, Any]],
    weight: float,
    opposite_penalty: float,
    preferred_cap: float,
    cap_penalty: float,
    max_bonus: float,
) -> dict[str, Any]:
    rows = read_csv(path)
    by_rule: dict[str, Any] = {}
    for rule in rules:
        dim, sign = rule_dim_sign(rule)
        hits = [row for row in rows if substate_key(row, rule["state_event"]) == rule["substate_key"]]
        signed_vals = [f(row, f"action_{dim}") * sign for row in hits]
        family_counts = Counter(action_family(row) for row in hits)
        prefer_scores = [
            variant_score(row, rule, "prefer", weight, opposite_penalty, preferred_cap, cap_penalty, max_bonus)
            for row in hits
        ]
        penalty_scores = [
            variant_score(row, rule, "penalty_only", weight, opposite_penalty, preferred_cap, cap_penalty, max_bonus)
            for row in hits
        ]
        by_rule[rule["id"]] = {
            "hits": len(hits),
            "action_family_counts": dict(family_counts),
            "preferred_positive_count": sum(1 for x in signed_vals if x >= 0.2),
            "opposed_count": sum(1 for x in signed_vals if x <= -0.2),
            "preferred_saturated_count": sum(1 for x in signed_vals if x > preferred_cap),
            "mean_signed_action": mean(signed_vals) if signed_vals else 0.0,
            "prefer_variant": {
                "mean_score_on_actual": mean(prefer_scores) if prefer_scores else 0.0,
                "positive_score_count": sum(1 for x in prefer_scores if x > 1e-9),
                "negative_score_count": sum(1 for x in prefer_scores if x < -1e-9),
            },
            "penalty_only_variant": {
                "mean_score_on_actual": mean(penalty_scores) if penalty_scores else 0.0,
                "penalized_actual_count": sum(1 for x in penalty_scores if x < -1e-9),
                "unchanged_actual_count": sum(1 for x in penalty_scores if abs(x) <= 1e-9),
            },
            "context": {
                "mean_backward_speed": mean([f(row, "backward_speed") for row in hits]) if hits else 0.0,
                "mean_rotation_penalty": mean([f(row, "rotation_penalty") for row in hits]) if hits else 0.0,
                "mean_full_foot_support": mean([f(row, "full_foot_support") for row in hits]) if hits else 0.0,
            },
        }
    return {"name": name, "trace_csv": str(path), "num_rows": len(rows), "rules": by_rule}


def parse_trace_arg(item: str) -> tuple[str, Path]:
    if ":" not in item:
        path = Path(item)
        return path.stem, path
    name, path = item.split(":", 1)
    return name, Path(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper v6.1 Rule Prior Variant Audit",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Weight: `{payload['weight']}`",
        f"Opposite penalty: `{payload['opposite_penalty']}`",
        f"Preferred cap: `{payload['preferred_cap']}`",
        f"Cap penalty: `{payload['cap_penalty']}`",
        "",
    ]
    for trace in payload["traces"]:
        lines += [f"## Trace: {trace['name']}", "", f"Rows: `{trace['num_rows']}`", ""]
        for rule_id, stats in trace["rules"].items():
            lines += [
                f"### {rule_id}",
                "",
                f"- Hits: `{stats['hits']}`",
                f"- Action families: `{stats['action_family_counts']}`",
                f"- Preferred positive count: `{stats['preferred_positive_count']}`",
                f"- Opposed count: `{stats['opposed_count']}`",
                f"- Preferred saturated count: `{stats['preferred_saturated_count']}`",
                f"- Mean signed action: `{stats['mean_signed_action']:.4f}`",
                f"- Prefer positive / negative scores: `{stats['prefer_variant']['positive_score_count']}` / `{stats['prefer_variant']['negative_score_count']}`",
                f"- Penalty-only penalized / unchanged actual actions: `{stats['penalty_only_variant']['penalized_actual_count']}` / `{stats['penalty_only_variant']['unchanged_actual_count']}`",
                f"- Mean backward speed / rotation / support: `{stats['context']['mean_backward_speed']:.4f}` / `{stats['context']['mean_rotation_penalty']:.4f}` / `{stats['context']['mean_full_foot_support']:.4f}`",
                "",
            ]
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--trace", action="append", default=None, help="name:path trace CSV; may be repeated")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument("--opposite-penalty", type=float, default=0.02)
    parser.add_argument("--preferred-cap", type=float, default=0.70)
    parser.add_argument("--cap-penalty", type=float, default=0.02)
    parser.add_argument("--max-bonus", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_json(args.library)
    rules = [
        rule
        for rule in library.get("rules", [])
        if rule.get("status") == "replay_validated"
        and rule.get("source") == "v4_substate"
        and rule.get("intervention_type") == "soft_prefer_action_family"
    ]
    traces = [parse_trace_arg(item) for item in (args.trace or DEFAULT_TRACES)]
    payload = {
        "rule_family": "online-rule-prior-variant-audit-v61",
        "library": str(args.library),
        "num_rules": len(rules),
        "weight": args.weight,
        "opposite_penalty": args.opposite_penalty,
        "preferred_cap": args.preferred_cap,
        "cap_penalty": args.cap_penalty,
        "max_bonus": args.max_bonus,
        "traces": [
            audit_trace(name, path, rules, args.weight, args.opposite_penalty, args.preferred_cap, args.cap_penalty, args.max_bonus)
            for name, path in traces
        ],
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote v6.1 variant audit: {args.output}")


if __name__ == "__main__":
    main()
