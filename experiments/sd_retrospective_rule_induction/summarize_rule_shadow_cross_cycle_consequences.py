#!/usr/bin/env python3
"""Summarize cross-cycle support for rule-shadow consequence audits.

Each input audit is produced by `audit_rule_shadow_short_window_consequences.py`.
This script aggregates the same rule across audits and reports whether the
short-window consequence direction is stable enough to be considered for a
future live update. It is read-only and does not write a rule library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def relation_compared(row: dict[str, Any], min_relation_count: int) -> bool:
    return (
        int(row.get("expected_count", 0) or 0) >= min_relation_count
        and int(row.get("contrary_count", 0) or 0) >= min_relation_count
        and row.get("outcome_margin") is not None
    )


def summarize_rule(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    compared = [row for row in rows if relation_compared(row, args.min_relation_count)]
    supported = [
        row
        for row in compared
        if bool(row.get("consequence_supported", False))
        and float(row.get("outcome_margin", 0.0) or 0.0) >= args.min_outcome_margin
    ]
    margins = [float(row.get("outcome_margin", 0.0) or 0.0) for row in compared]
    deltas = [float(row.get("shadow_confidence_delta", 0.0) or 0.0) for row in rows]
    support_rate = len(supported) / len(compared) if compared else 0.0
    stable = (
        len(compared) >= args.min_compared_audits
        and len(supported) >= args.min_supported_audits
        and support_rate >= args.min_support_rate
        and (mean(margins) if margins else 0.0) >= args.min_mean_margin
    )
    first = rows[0]
    return {
        "id": first.get("id"),
        "source": first.get("source"),
        "rule_key": first.get("rule_key"),
        "action_family": first.get("action_family"),
        "num_audits_seen": len(rows),
        "num_compared_audits": len(compared),
        "num_supported_audits": len(supported),
        "support_rate": support_rate,
        "mean_outcome_margin": mean(margins) if margins else None,
        "min_outcome_margin": min(margins) if margins else None,
        "max_outcome_margin": max(margins) if margins else None,
        "mean_shadow_confidence_delta": mean(deltas) if deltas else 0.0,
        "stable_consequence_supported": bool(stable),
        "audit_paths": [row["_audit_path"] for row in rows],
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    by_rule: dict[str, list[dict[str, Any]]] = {}
    audits: list[dict[str, Any]] = []
    for path in args.audit:
        payload = load_json(path)
        audits.append(
            {
                "path": str(path),
                "num_changed_rules": payload.get("num_changed_rules"),
                "num_rules_with_relation_comparison": payload.get("num_rules_with_relation_comparison"),
                "num_consequence_supported": payload.get("num_consequence_supported"),
                "support_rate_among_compared": payload.get("support_rate_among_compared"),
            }
        )
        for row in payload.get("rules", []):
            rid = str(row.get("id", ""))
            if not rid:
                continue
            item = dict(row)
            item["_audit_path"] = str(path)
            by_rule.setdefault(rid, []).append(item)

    rules = [summarize_rule(rows, args) for rows in by_rule.values()]
    rules.sort(
        key=lambda row: (
            bool(row["stable_consequence_supported"]),
            row["num_supported_audits"],
            row["support_rate"],
            row["mean_outcome_margin"] if row["mean_outcome_margin"] is not None else -999.0,
        ),
        reverse=True,
    )
    stable = [row for row in rules if row["stable_consequence_supported"]]
    return {
        "audit_family": "rule_shadow_cross_cycle_consequence_summary_v0",
        "audits": audits,
        "parameters": {
            "min_relation_count": args.min_relation_count,
            "min_outcome_margin": args.min_outcome_margin,
            "min_compared_audits": args.min_compared_audits,
            "min_supported_audits": args.min_supported_audits,
            "min_support_rate": args.min_support_rate,
            "min_mean_margin": args.min_mean_margin,
        },
        "num_input_audits": len(args.audit),
        "num_rules_seen": len(rules),
        "num_stable_supported_rules": len(stable),
        "rules": rules,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Rule Shadow Cross-Cycle Consequence Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| input audits | {payload['num_input_audits']} |",
        f"| rules seen | {payload['num_rules_seen']} |",
        f"| stable supported rules | {payload['num_stable_supported_rules']} |",
        "",
        "## Audits",
        "",
        "| audit | changed | compared | supported | support rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for audit in payload["audits"]:
        lines.append(
            "| "
            f"`{audit['path']}` | "
            f"{audit.get('num_changed_rules')} | "
            f"{audit.get('num_rules_with_relation_comparison')} | "
            f"{audit.get('num_consequence_supported')} | "
            f"{float(audit.get('support_rate_among_compared') or 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "| stable | supported/compared | support rate | mean margin | rule |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["rules"]:
        margin = row["mean_outcome_margin"]
        lines.append(
            "| "
            f"`{row['stable_consequence_supported']}` | "
            f"{row['num_supported_audits']}/{row['num_compared_audits']} | "
            f"{row['support_rate']:.3f} | "
            f"{'n/a' if margin is None else f'{margin:+.4f}'} | "
            f"`{row['id']}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-relation-count", type=int, default=10)
    parser.add_argument("--min-outcome-margin", type=float, default=0.0)
    parser.add_argument("--min-compared-audits", type=int, default=2)
    parser.add_argument("--min-supported-audits", type=int, default=2)
    parser.add_argument("--min-support-rate", type=float, default=0.67)
    parser.add_argument("--min-mean-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote cross-cycle consequence summary: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(
        "audits={audits} rules={rules} stable={stable}".format(
            audits=payload["num_input_audits"],
            rules=payload["num_rules_seen"],
            stable=payload["num_stable_supported_rules"],
        )
    )


if __name__ == "__main__":
    main()
