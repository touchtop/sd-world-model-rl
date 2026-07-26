#!/usr/bin/env python3
"""Audit rule-context action competition with TVA counterfactual values.

This read-only audit treats a rule context as a state description, then asks a
generic question: among available action families in that same context, which
families receive higher terminal value after SD rollout and TVA evaluation?

The report is intended to separate "a rule often fires" from "the rule's
action family is value-supported inside its own context". It does not encode
task-specific posture targets; task-specific labels remain diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import row_parts, state_matches  # noqa: E402
from apply_rule_effect_confidence import rule_id  # noqa: E402
from apply_tva_shadow_update import opposite_action_family, record_family_values  # noqa: E402
from common import load_json, write_json  # noqa: E402


DEFAULT_STATUSES = ("confidence_active", "confidence_observing")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def family_joint(family: str | None) -> str:
    if not family or "_" not in family:
        return str(family or "")
    return family.rsplit("_", 1)[0]


def context_key(rule: dict[str, Any]) -> str:
    return str(rule.get("rule_key") or rule.get("substate_key") or rule_id(rule).split("->", 1)[0])


def load_rules(path: Path, statuses: set[str]) -> list[dict[str, Any]]:
    payload = load_json(path)
    return [rule for rule in payload.get("rules", []) if str(rule.get("status")) in statuses]


def init_context() -> dict[str, Any]:
    return {
        "num_records": 0,
        "labels": Counter(),
        "events": Counter(),
        "episodes": Counter(),
        "best_family": Counter(),
        "families": defaultdict(
            lambda: {
                "seen": 0,
                "values": [],
                "adv_over_actual": [],
                "margin_to_best": [],
                "best_hits": 0,
                "top2_hits": 0,
                "opposite_margins": [],
            }
        ),
    }


def update_context(ctx: dict[str, Any], record: dict[str, Any]) -> None:
    family_values, ranked = record_family_values(record)
    if not family_values:
        return
    actual_value = float(record.get("actual_tva_value", 0.0) or 0.0)
    best_family, best_value = ranked[0]
    top2 = {family for family, _value in ranked[:2]}
    ctx["num_records"] += 1
    ctx["labels"][str(record.get("label", "unknown"))] += 1
    ctx["events"][str(record.get("event", "unknown"))] += 1
    ctx["episodes"][f"{record.get('seed', '?')}:{record.get('episode_id', '?')}"] += 1
    ctx["best_family"][best_family] += 1
    for family, value in family_values.items():
        stats = ctx["families"][family]
        stats["seen"] += 1
        stats["values"].append(float(value))
        stats["adv_over_actual"].append(float(value - actual_value))
        stats["margin_to_best"].append(float(value - best_value))
        if family == best_family:
            stats["best_hits"] += 1
        if family in top2:
            stats["top2_hits"] += 1
        opposite = opposite_action_family(family)
        if opposite is not None and opposite in family_values:
            stats["opposite_margins"].append(float(value - family_values[opposite]))


def summarize_family(stats: dict[str, Any], num_records: int) -> dict[str, Any]:
    seen = int(stats["seen"])
    return {
        "seen": seen,
        "coverage": seen / num_records if num_records else 0.0,
        "mean_tva_value": safe_mean(stats["values"]),
        "mean_advantage_over_actual": safe_mean(stats["adv_over_actual"]),
        "mean_margin_to_best": safe_mean(stats["margin_to_best"]),
        "best_rate": stats["best_hits"] / seen if seen else 0.0,
        "top2_rate": stats["top2_hits"] / seen if seen else 0.0,
        "mean_opposite_margin": safe_mean(stats["opposite_margins"]),
    }


def summarize_rule(rule: dict[str, Any], family_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    intervention = str(rule.get("intervention_type", ""))
    family = str(rule.get("action_family", ""))
    fam = family_summary.get(family, {})
    opposite = opposite_action_family(family)
    opp = family_summary.get(opposite or "", {})
    mean_margin_to_best = float(fam.get("mean_margin_to_best", 0.0) or 0.0)
    best_rate = float(fam.get("best_rate", 0.0) or 0.0)
    top2_rate = float(fam.get("top2_rate", 0.0) or 0.0)
    opposite_margin = float(fam.get("mean_opposite_margin", 0.0) or 0.0)
    if intervention == "avoid_action_family":
        supported = (
            best_rate <= 0.20
            and top2_rate <= 0.50
            and (not opp or float(opp.get("mean_tva_value", 0.0)) >= float(fam.get("mean_tva_value", 0.0)))
        )
        suspect = best_rate >= 0.35 or mean_margin_to_best >= -0.05
    else:
        supported = best_rate >= 0.35 or top2_rate >= 0.70 or opposite_margin >= 0.05
        suspect = best_rate <= 0.10 and top2_rate <= 0.40 and opposite_margin <= -0.05
    return {
        "id": rule_id(rule),
        "status": rule.get("status"),
        "confidence": rule.get("confidence"),
        "source": rule.get("source"),
        "intervention_type": intervention,
        "action_family": family,
        "joint": family_joint(family),
        "mean_tva_value": fam.get("mean_tva_value"),
        "mean_advantage_over_actual": fam.get("mean_advantage_over_actual"),
        "mean_margin_to_best": fam.get("mean_margin_to_best"),
        "best_rate": fam.get("best_rate"),
        "top2_rate": fam.get("top2_rate"),
        "mean_opposite_margin": fam.get("mean_opposite_margin"),
        "tva_supported": supported,
        "tva_suspect": suspect,
    }


def context_conflicts(rule_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rule_rows:
        by_joint[str(row.get("joint", ""))].append(row)
    out: list[dict[str, Any]] = []
    for joint, rows in by_joint.items():
        if not joint or joint == "zero":
            continue
        signs = {str(row.get("action_family", "")).rsplit("_", 1)[-1] for row in rows}
        if {"pos", "neg"} <= signs:
            out.append(
                {
                    "joint": joint,
                    "families": sorted({str(row.get("action_family")) for row in rows}),
                    "num_rules": len(rows),
                    "supported": [row["id"] for row in rows if row.get("tva_supported")],
                    "suspect": [row["id"] for row in rows if row.get("tva_suspect")],
                }
            )
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rules = load_rules(args.library, set(args.status))
    rules_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        rules_by_context[context_key(rule)].append(rule)

    contexts: dict[str, dict[str, Any]] = defaultdict(init_context)
    trace_cache: dict[str, list[dict[str, str]]] = {}
    matched_records = 0
    for record in load_json(args.tva_counterfactual_audit).get("records", []):
        trace_csv = str(record.get("trace_csv", ""))
        if not trace_csv:
            continue
        rows = trace_cache.get(trace_csv)
        if rows is None:
            rows = read_csv(Path(trace_csv))
            trace_cache[trace_csv] = rows
        step = int(record.get("step", -1) or -1)
        if step < 0 or step >= len(rows):
            continue
        parts = row_parts(rows[step], args.continue_score_min)
        matched_contexts: set[str] = set()
        for key, key_rules in rules_by_context.items():
            if any(state_matches(rule, parts) for rule in key_rules):
                matched_contexts.add(key)
        for key in matched_contexts:
            matched_records += 1
            update_context(contexts[key], record)

    context_rows: list[dict[str, Any]] = []
    for key, ctx in contexts.items():
        n = int(ctx["num_records"])
        if n < args.min_records:
            continue
        family_summary = {
            family: summarize_family(stats, n)
            for family, stats in ctx["families"].items()
        }
        ranked_families = sorted(
            family_summary.items(),
            key=lambda item: (
                item[1]["best_rate"],
                item[1]["top2_rate"],
                item[1]["mean_tva_value"],
            ),
            reverse=True,
        )
        rule_rows = [summarize_rule(rule, family_summary) for rule in rules_by_context[key]]
        context_rows.append(
            {
                "context_key": key,
                "num_records": n,
                "labels": dict(ctx["labels"].most_common()),
                "events": dict(ctx["events"].most_common()),
                "best_family_counts": dict(ctx["best_family"].most_common()),
                "ranked_families": [
                    {"family": family, **summary} for family, summary in ranked_families
                ],
                "rules": sorted(
                    rule_rows,
                    key=lambda row: (
                        bool(row.get("tva_supported")),
                        float(row.get("best_rate") or 0.0),
                        float(row.get("top2_rate") or 0.0),
                    ),
                    reverse=True,
                ),
                "conflicts": context_conflicts(rule_rows),
                "num_rules": len(rule_rows),
                "num_tva_supported_rules": sum(1 for row in rule_rows if row.get("tva_supported")),
                "num_tva_suspect_rules": sum(1 for row in rule_rows if row.get("tva_suspect")),
            }
        )
    context_rows.sort(
        key=lambda row: (
            row["num_records"],
            row["num_tva_suspect_rules"],
            len(row["conflicts"]),
        ),
        reverse=True,
    )
    return {
        "audit_family": "tva_context_action_competition_v0",
        "source_library": str(args.library),
        "source_tva_counterfactual_audit": str(args.tva_counterfactual_audit),
        "statuses": args.status,
        "continue_score_min": args.continue_score_min,
        "min_records": args.min_records,
        "num_rules": len(rules),
        "num_contexts_in_library": len(rules_by_context),
        "matched_context_records": matched_records,
        "num_contexts_reported": len(context_rows),
        "summary": {
            "contexts_with_conflicts": sum(1 for row in context_rows if row["conflicts"]),
            "tva_supported_rules": sum(row["num_tva_supported_rules"] for row in context_rows),
            "tva_suspect_rules": sum(row["num_tva_suspect_rules"] for row in context_rows),
        },
        "contexts": context_rows,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_contexts: int) -> None:
    lines = [
        "# TVA Context Action Competition Audit",
        "",
        "This report is read-only. Contexts come from the current rule language; action-family ranking comes from SD rollout plus TVA terminal values.",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Contexts reported: `{payload['num_contexts_reported']}`",
        f"Matched context records: `{payload['matched_context_records']}`",
        f"Contexts with conflicts: `{payload['summary']['contexts_with_conflicts']}`",
        f"TVA-supported rules: `{payload['summary']['tva_supported_rules']}`",
        f"TVA-suspect rules: `{payload['summary']['tva_suspect_rules']}`",
        "",
    ]
    for ctx in payload["contexts"][:max_contexts]:
        lines.extend(
            [
                f"## `{ctx['context_key']}`",
                "",
                f"- Records: `{ctx['num_records']}`",
                f"- Labels: `{ctx['labels']}`",
                f"- Events: `{ctx['events']}`",
                f"- Best-family counts: `{ctx['best_family_counts']}`",
                f"- Conflicts: `{ctx['conflicts']}`",
                "",
                "| family | best% | top2% | mean value | adv actual | margin best | opposite margin |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fam in ctx["ranked_families"][:10]:
            lines.append(
                f"| `{fam['family']}` | {100*fam['best_rate']:.1f} | {100*fam['top2_rate']:.1f} | "
                f"{fam['mean_tva_value']:.4f} | {fam['mean_advantage_over_actual']:.4f} | "
                f"{fam['mean_margin_to_best']:.4f} | {fam['mean_opposite_margin']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| rule | intervention | confidence | best% | top2% | opposite margin | supported | suspect |",
                "|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for rule in ctx["rules"]:
            lines.append(
                f"| `{rule['id']}` | `{rule['intervention_type']}:{rule['action_family']}` | "
                f"{float(rule.get('confidence') or 0.0):.3f} | "
                f"{100*float(rule.get('best_rate') or 0.0):.1f} | "
                f"{100*float(rule.get('top2_rate') or 0.0):.1f} | "
                f"{float(rule.get('mean_opposite_margin') or 0.0):.4f} | "
                f"`{rule['tva_supported']}` | `{rule['tva_suspect']}` |"
            )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tva-counterfactual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", action="append", default=list(DEFAULT_STATUSES))
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--min-records", type=int, default=20)
    parser.add_argument("--max-contexts-md", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_contexts_md)
    print(f"wrote TVA context action competition audit: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(
        "contexts={contexts} conflicts={conflicts} supported={supported} suspect={suspect}".format(
            contexts=payload["num_contexts_reported"],
            conflicts=payload["summary"]["contexts_with_conflicts"],
            supported=payload["summary"]["tva_supported_rules"],
            suspect=payload["summary"]["tva_suspect_rules"],
        )
    )


if __name__ == "__main__":
    main()
