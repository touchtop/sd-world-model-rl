#!/usr/bin/env python3
"""Audit v2 state-action rules for cross-seed consistency and good-trace risk."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_COUNTERFACTUALS_V2,
    DEFAULT_RULE_CONSISTENCY_V3,
    DEFAULT_RULES_V2,
    DEFAULT_RULES_V3,
    load_json,
    mean_or_zero,
    safe_rate,
    write_json,
)


def action_family(name: str) -> str:
    if name in {"actual", "zero"}:
        return name
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def entropy_norm(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    h = 0.0
    for count in counts.values():
        p = count / total
        h -= p * math.log(p + 1e-12)
    return h / math.log(len(counts))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "mean_advantage": mean_or_zero(float(r["advantage_over_actual"]) for r in records),
        "mean_actual_rank": mean_or_zero(float(r["actual_rank_among_probe_actions"]) for r in records),
        "labels": dict(Counter(str(r["label"]) for r in records)),
        "seeds": dict(Counter(str(r["seed"]) for r in records)),
        "cycles": dict(Counter(str(r["cycle_step"]) for r in records)),
    }


def audit_rule(
    rule: dict[str, Any],
    records: list[dict[str, Any]],
    event_records: list[dict[str, Any]],
    min_seeds: int,
    max_good_false_prefer_rate: float,
) -> dict[str, Any]:
    matched = [
        rec for rec in records
        if rec["event"] == rule["state_event"] and action_family(rec["best_action"]) == rule["action_family"]
    ]
    label_counts = Counter(str(r["label"]) for r in matched)
    seed_counts = Counter(str(r["seed"]) for r in matched)
    cycle_counts = Counter(str(r["cycle_step"]) for r in matched)
    event_action_counts = Counter(action_family(r["best_action"]) for r in event_records)
    top_action, top_count = event_action_counts.most_common(1)[0] if event_action_counts else ("", 0)

    support = len(matched)
    event_support = len(event_records)
    mean_advantage = mean_or_zero(float(r["advantage_over_actual"]) for r in matched)
    good_false_prefer_rate = safe_rate(label_counts.get("good", 0), support)
    conflict_entropy = entropy_norm(dict(event_action_counts))
    seeds_covered = len(seed_counts)
    cycles_covered = len(cycle_counts)
    reliable = (
        support >= 20
        and seeds_covered >= min_seeds
        and mean_advantage >= 0.10
        and good_false_prefer_rate <= max_good_false_prefer_rate
    )
    grade = "A" if reliable and seeds_covered >= 3 and safe_rate(support, event_support) >= 0.10 else "B" if reliable else "C"
    notes: list[str] = []
    if seeds_covered < min_seeds:
        notes.append("支持样本集中于少数 seed，跨 seed 稳定性不足。")
    if good_false_prefer_rate > max_good_false_prefer_rate:
        notes.append("好轨迹中也频繁出现该偏好，接入训练前需检查误伤风险。")
    if conflict_entropy > 0.75:
        notes.append("同一状态事件下最优动作族分散，需进一步细分状态条件。")
    if support < 20:
        notes.append("支持数偏低，暂不适合作为在线先验。")
    if not notes:
        notes.append("离线审查未发现明显稳定性问题，可作为软先验候选。")

    return {
        "id": rule["id"],
        "state_event": rule["state_event"],
        "action_family": rule["action_family"],
        "support": support,
        "event_support": event_support,
        "event_coverage": safe_rate(support, event_support),
        "mean_advantage": mean_advantage,
        "mean_actual_rank": mean_or_zero(float(r["actual_rank_among_probe_actions"]) for r in matched),
        "seeds_covered": seeds_covered,
        "cycles_covered": cycles_covered,
        "label_counts": dict(label_counts),
        "seed_counts": dict(seed_counts),
        "cycle_counts": dict(cycle_counts),
        "good_false_prefer_rate": good_false_prefer_rate,
        "same_event_action_distribution": dict(event_action_counts),
        "same_event_top_action_family": top_action,
        "same_event_top_action_share": safe_rate(top_count, event_support),
        "same_event_conflict_entropy": conflict_entropy,
        "reliable": reliable,
        "grade": grade,
        "notes": notes,
    }


def build_reliable_rules(audits: list[dict[str, Any]], rules_payload: dict[str, Any]) -> dict[str, Any]:
    src_by_id = {r["id"]: r for r in rules_payload["rules"]}
    out_rules = []
    for audit in audits:
        if not audit["reliable"]:
            continue
        src = src_by_id[audit["id"]]
        out_rules.append(
            {
                "id": src["id"],
                "grade": audit["grade"],
                "state_event": src["state_event"],
                "state_condition": src["state_condition"],
                "action_family": src["action_family"],
                "intervention_type": "soft_prefer_action_family",
                "support": audit["support"],
                "seeds_covered": audit["seeds_covered"],
                "cycles_covered": audit["cycles_covered"],
                "event_coverage": audit["event_coverage"],
                "mean_advantage": audit["mean_advantage"],
                "good_false_prefer_rate": audit["good_false_prefer_rate"],
                "same_event_conflict_entropy": audit["same_event_conflict_entropy"],
                "notes": audit["notes"],
            }
        )
    out_rules.sort(key=lambda r: (r["grade"], r["mean_advantage"], r["support"]), reverse=True)
    return {
        "rule_family": "reliable-state-action-rules-v3",
        "source_rule_family": rules_payload.get("rule_family"),
        "num_source_rules": len(rules_payload["rules"]),
        "num_reliable_rules": len(out_rules),
        "rules": out_rules,
    }


def write_markdown(path: Path, consistency: dict[str, Any], reliable: dict[str, Any]) -> None:
    lines = [
        "# Hopper Rule Consistency Audit v3",
        "",
        f"Records: `{consistency['num_records']}`",
        f"Source rules: `{consistency['num_rules']}`",
        f"Reliable rules: `{reliable['num_reliable_rules']}`",
        "",
        "## Reliable Rules",
        "",
    ]
    for rule in reliable["rules"]:
        lines += [
            f"### {rule['id']}",
            "",
            f"- Grade: `{rule['grade']}`",
            f"- Support: `{rule['support']}`",
            f"- Seeds covered: `{rule['seeds_covered']}`",
            f"- Cycles covered: `{rule['cycles_covered']}`",
            f"- Event coverage: `{rule['event_coverage']:.3f}`",
            f"- Mean advantage: `{rule['mean_advantage']:.4f}`",
            f"- Good false-prefer rate: `{rule['good_false_prefer_rate']:.3f}`",
            f"- Conflict entropy: `{rule['same_event_conflict_entropy']:.3f}`",
            f"- Notes: {' '.join(rule['notes'])}",
            "",
        ]
    lines += ["## All Audits", ""]
    for audit in consistency["audits"]:
        lines += [
            f"### {audit['id']}",
            "",
            f"- Grade: `{audit['grade']}`",
            f"- Reliable: `{audit['reliable']}`",
            f"- Support: `{audit['support']}`",
            f"- Seeds covered: `{audit['seeds_covered']}`",
            f"- Good false-prefer rate: `{audit['good_false_prefer_rate']:.3f}`",
            f"- Conflict entropy: `{audit['same_event_conflict_entropy']:.3f}`",
            f"- Notes: {' '.join(audit['notes'])}",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines))


def analyze(
    counterfactuals: dict[str, Any],
    rules_payload: dict[str, Any],
    min_seeds: int,
    max_good_false_prefer_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = counterfactuals["records"]
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_event[rec["event"]].append(rec)
    audits = [
        audit_rule(rule, records, by_event.get(rule["state_event"], []), min_seeds, max_good_false_prefer_rate)
        for rule in rules_payload["rules"]
    ]
    audits.sort(key=lambda a: (a["reliable"], a["grade"], a["mean_advantage"], a["support"]), reverse=True)
    consistency = {
        "source_counterfactuals": counterfactuals.get("model_checkpoint"),
        "num_records": len(records),
        "num_rules": len(rules_payload["rules"]),
        "min_seeds": min_seeds,
        "max_good_false_prefer_rate": max_good_false_prefer_rate,
        "event_summaries": {event: summarize_records(vals) for event, vals in sorted(by_event.items())},
        "audits": audits,
    }
    return consistency, build_reliable_rules(audits, rules_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactuals", type=Path, default=DEFAULT_COUNTERFACTUALS_V2)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_RULE_CONSISTENCY_V3)
    parser.add_argument("--reliable-output", type=Path, default=DEFAULT_RULES_V3)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--max-good-false-prefer-rate", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    consistency, reliable = analyze(
        load_json(args.counterfactuals),
        load_json(args.rules),
        args.min_seeds,
        args.max_good_false_prefer_rate,
    )
    write_json(args.output, consistency)
    write_json(args.reliable_output, reliable)
    write_markdown(args.output, consistency, reliable)
    write_markdown(args.reliable_output, consistency, reliable)
    print(f"wrote consistency: {args.output}")
    print(f"wrote reliable rules: {args.reliable_output}")
    print(f"reliable rules: {reliable['num_reliable_rules']}")


if __name__ == "__main__":
    main()

