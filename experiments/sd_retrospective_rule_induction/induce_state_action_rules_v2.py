#!/usr/bin/env python3
"""Induce state-action rules from short-horizon SD counterfactual records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_COUNTERFACTUALS_V2, DEFAULT_RULES_V2, load_json, mean_or_zero, write_json


def action_family(name: str) -> str:
    if name in {"actual", "zero"}:
        return name
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return name


def event_condition(event: str) -> dict[str, Any]:
    mapping = {
        "unsupported_high_torque": {
            "description": "动作幅度较高，但足底支撑与接地不足。",
            "state_predicate": "action_abs_mean >= 0.65 and full_foot_support < 0.20 and foot_grounded < 0.50",
        },
        "landing_unprepared": {
            "description": "下落阶段后续短视界支撑不足。",
            "state_predicate": "falling_air >= 0.50 and vz < -0.20",
        },
        "rotation_risk": {
            "description": "高角速度或旋转惩罚较高。",
            "state_predicate": "rotation_penalty >= 0.50 or abs(pitch_rate) >= 3.0",
        },
        "prone_or_rear_trap": {
            "description": "处于后仰或俯伏接地陷阱。",
            "state_predicate": "grounded_prone >= 0.50 or grounded_rear >= 0.50",
        },
        "prepared_support": {
            "description": "足底支撑与近直立/准备态同时存在。",
            "state_predicate": "full_foot_support >= 0.50 and (grounded_prepared >= 0.50 or grounded_upright >= 0.50)",
        },
        "effective_drive": {
            "description": "正向速度、支撑与动作幅度共同出现。",
            "state_predicate": "vx > 0.75 and full_foot_support >= 0.35",
        },
    }
    return mapping.get(event, {"description": event, "state_predicate": event})


def induce(counterfactuals: dict[str, Any], min_support: int, min_advantage: float) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    event_counts: dict[str, int] = defaultdict(int)
    for rec in counterfactuals["records"]:
        event = rec["event"]
        family = action_family(rec["best_action"])
        grouped[(event, family)].append(rec)
        event_counts[event] += 1

    rules: list[dict[str, Any]] = []
    for (event, family), records in grouped.items():
        support = len(records)
        if support < min_support:
            continue
        mean_adv = mean_or_zero(float(r["advantage_over_actual"]) for r in records)
        if mean_adv < min_advantage:
            continue
        total_event = event_counts[event]
        rules.append(
            {
                "id": f"{event}:prefer:{family}",
                "state_event": event,
                "state_condition": event_condition(event),
                "action_family": family,
                "intervention_type": "prefer_action_family",
                "support": support,
                "event_coverage": support / total_event if total_event else 0.0,
                "mean_advantage_over_actual": mean_adv,
                "mean_actual_rank": mean_or_zero(float(r["actual_rank_among_probe_actions"]) for r in records),
                "confidence": min(1.0, (support / max(min_support, 1)) * max(mean_adv, 0.0)),
                "example_records": [
                    {
                        "episode_id": r["episode_id"],
                        "step": r["step"],
                        "label": r["label"],
                        "best_action": r["best_action"],
                        "advantage_over_actual": r["advantage_over_actual"],
                    }
                    for r in sorted(records, key=lambda x: x["advantage_over_actual"], reverse=True)[:5]
                ],
            }
        )
    rules.sort(key=lambda r: (r["confidence"], r["mean_advantage_over_actual"], r["support"]), reverse=True)
    return {
        "source_counterfactuals": counterfactuals.get("model_checkpoint"),
        "rule_family": "state-event-to-action-family-v2",
        "min_support": min_support,
        "min_advantage": min_advantage,
        "num_records": counterfactuals["num_state_records"],
        "rules": rules,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    md_path = path.with_suffix(".md")
    lines = [
        "# Hopper State-Action Rules v2",
        "",
        f"Records: `{payload['num_records']}`",
        f"Rules: `{len(payload['rules'])}`",
        "",
    ]
    for idx, rule in enumerate(payload["rules"], start=1):
        lines.extend(
            [
                f"## Rule {idx}: {rule['id']}",
                "",
                f"- State predicate: `{rule['state_condition']['state_predicate']}`",
                f"- Action family: `{rule['action_family']}`",
                f"- Intervention: `{rule['intervention_type']}`",
                f"- Support: `{rule['support']}`",
                f"- Event coverage: `{rule['event_coverage']:.3f}`",
                f"- Mean advantage: `{rule['mean_advantage_over_actual']:.4f}`",
                f"- Mean actual rank: `{rule['mean_actual_rank']:.2f}`",
                f"- Confidence: `{rule['confidence']:.3f}`",
                f"- Interpretation: {rule['state_condition']['description']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactuals", type=Path, default=DEFAULT_COUNTERFACTUALS_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_RULES_V2)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-advantage", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = induce(load_json(args.counterfactuals), args.min_support, args.min_advantage)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote state-action rules: {args.output}")
    print(f"rules: {len(payload['rules'])}")


if __name__ == "__main__":
    main()

