#!/usr/bin/env python3
"""Export replay-safe Hopper v8 transition rules as a narrow candidate library."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_json, write_json


DEFAULT_SAFETY = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_replay_safety_200k_v72.json")
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.json")


def export_rules(safety: dict[str, Any], max_false_intervention_rate: float) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    for row in safety.get("rules", []):
        if row.get("replay_status") != "replay_safety_supported":
            continue
        if float(row.get("false_intervention_rate", 1.0)) > max_false_intervention_rate:
            continue
        rule = {
            "id": row["id"],
            "rule_key": row["rule_key"],
            "rule_role": "transition",
            "action_family": row["action_family"],
            "intervention_type": row["intervention_type"],
            "status": "v8_replay_safe_candidate",
            "source": "v8_transition_replay_safety",
            "heldout_strength": row.get("heldout_strength"),
            "replay_hits": row["hits"],
            "replay_good_hits": row["good_hits"],
            "replay_false_intervention_rate": row["false_intervention_rate"],
            "replay_good_hit_rate": row["good_hit_rate"],
            "replay_actual_same_family_rate": row["actual_same_family_rate"],
            "replay_actual_opposite_family_rate": row["actual_opposite_family_rate"],
            "seed_counts": row["seed_counts"],
        }
        rules.append(rule)

    rules.sort(
        key=lambda rule: (
            rule["replay_false_intervention_rate"],
            -int(rule["replay_hits"]),
        )
    )
    return {
        "rule_family": "hopper-v8-transition-replay-safe-candidates",
        "source_safety_report": safety.get("source_validation"),
        "num_rules": len(rules),
        "status_counts": dict(Counter(rule["status"] for rule in rules)),
        "intervention_counts": dict(Counter(rule["intervention_type"] for rule in rules)),
        "max_false_intervention_rate": max_false_intervention_rate,
        "rules": rules,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper v8 Replay-safe Transition Rules",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Intervention counts: `{payload['intervention_counts']}`",
        f"Max false intervention rate: `{payload['max_false_intervention_rate']}`",
        "",
    ]
    for rule in payload["rules"]:
        lines.extend(
            [
                f"## `{rule['id']}`",
                "",
                f"- Intervention: `{rule['intervention_type']}`",
                f"- Replay hits / good hits: `{rule['replay_hits']}` / `{rule['replay_good_hits']}`",
                f"- False intervention rate: `{rule['replay_false_intervention_rate']:.3f}`",
                f"- Seed counts: `{rule['seed_counts']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-false-intervention-rate", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_rules(load_json(args.safety), args.max_false_intervention_rate)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote v8 replay-safe rules: {args.output}")
    print(f"rules: {payload['num_rules']} interventions: {payload['intervention_counts']}")


if __name__ == "__main__":
    main()
