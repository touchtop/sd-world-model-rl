#!/usr/bin/env python3
"""Apply a consequence gate to a generic shadow rule library.

Given a source library, a shadow library, and a short-window consequence audit,
this script keeps only confidence changes that passed the consequence check and
reverts unsupported changes to the source confidence. The output is still an
eval/shadow artifact and does not modify the source library.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apply_rule_effect_confidence import confidence_status, rule_id


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def confidence(rule: dict[str, Any]) -> float:
    return float(rule.get("confidence", 0.0) or 0.0)


def consequence_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row.get("id")): row for row in payload.get("rules", []) if row.get("id")}


def gate_passes(evidence: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if evidence is None:
        return False
    if not bool(evidence.get("consequence_supported", False)):
        return False
    expected_count = int(evidence.get("expected_count", 0) or 0)
    contrary_count = int(evidence.get("contrary_count", 0) or 0)
    margin = evidence.get("outcome_margin")
    if margin is None:
        return False
    return (
        expected_count >= args.min_relation_count
        and contrary_count >= args.min_relation_count
        and float(margin) >= args.min_outcome_margin
    )


def apply_gate(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(args.source_library)
    shadow = load_json(args.shadow_library)
    evidence_by_id = consequence_index(args.consequence_audit)
    source_by_id = {rule_id(rule): rule for rule in source.get("rules", [])}

    out_rules: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    kept = 0
    rejected = 0
    changed = 0
    for rule in shadow.get("rules", []):
        rid = rule_id(rule)
        source_rule = source_by_id.get(rid)
        if source_rule is None:
            out_rules.append(rule)
            continue
        before = confidence(source_rule)
        after = confidence(rule)
        delta = after - before
        out = dict(rule)
        evidence = evidence_by_id.get(rid)
        passed = True
        reason = "unchanged"
        if abs(delta) > args.delta_epsilon:
            changed += 1
            passed = gate_passes(evidence, args)
            if passed:
                kept += 1
                reason = "consequence_supported_change_kept"
            else:
                rejected += 1
                reason = "consequence_unsupported_change_reverted"
                out["confidence"] = round(before, 6)
                out["confidence_status"] = confidence_status(before, args.active_threshold, args.retired_threshold)
                out["status"] = out["confidence_status"]
                out["effective_prior_scale"] = round(before, 6)
        out["consequence_gate"] = {
            "source_consequence_audit": str(args.consequence_audit),
            "passed": bool(passed),
            "reason": reason,
            "confidence_before_shadow": round(before, 6),
            "confidence_before_gate": round(after, 6),
            "confidence_after_gate": round(float(out.get("confidence", 0.0) or 0.0), 6),
            "outcome_margin": evidence.get("outcome_margin") if evidence else None,
            "expected_relation": evidence.get("expected_relation") if evidence else None,
            "contrary_relation": evidence.get("contrary_relation") if evidence else None,
            "expected_count": int(evidence.get("expected_count", 0) or 0) if evidence else 0,
            "contrary_count": int(evidence.get("contrary_count", 0) or 0) if evidence else 0,
            "min_relation_count": int(args.min_relation_count),
            "min_outcome_margin": float(args.min_outcome_margin),
        }
        rows.append({"id": rid, "delta_before_gate": delta, **out["consequence_gate"]})
        out_rules.append(out)

    status_counts = Counter(str(rule.get("status")) for rule in out_rules)
    payload = dict(shadow)
    payload["rules"] = out_rules
    payload["library_version"] = f"{shadow.get('library_version', 'rule_library')}_consequence_gated"
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["source_library"] = str(args.source_library)
    payload["source_shadow_library"] = str(args.shadow_library)
    payload["source_consequence_audit"] = str(args.consequence_audit)
    payload["rule_shadow_consequence_gate"] = {
        "num_changed_before_gate": changed,
        "num_kept": kept,
        "num_rejected": rejected,
        "status_counts": dict(status_counts),
        "min_relation_count": int(args.min_relation_count),
        "min_outcome_margin": float(args.min_outcome_margin),
        "updates": rows,
    }
    payload["online_enabled"] = any(rule.get("status") == "confidence_active" for rule in out_rules)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    gate = payload["rule_shadow_consequence_gate"]
    lines = [
        "# Rule Shadow Consequence-Gated Library",
        "",
        f"Source library: `{payload['source_library']}`",
        f"Source shadow library: `{payload['source_shadow_library']}`",
        f"Source consequence audit: `{payload['source_consequence_audit']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| changed before gate | {gate['num_changed_before_gate']} |",
        f"| kept | {gate['num_kept']} |",
        f"| rejected | {gate['num_rejected']} |",
        f"| status counts | `{gate['status_counts']}` |",
        "",
        "## Changed Rules",
        "",
        "| rule | delta before gate | gate | margin | counts |",
        "|---|---:|---|---:|---:|",
    ]
    for row in gate["updates"]:
        if abs(float(row.get("delta_before_gate", 0.0) or 0.0)) <= 1e-9:
            continue
        margin = row.get("outcome_margin")
        lines.append(
            "| "
            f"`{row['id']}` | "
            f"{float(row['delta_before_gate']):+.3f} | "
            f"`{row['reason']}` | "
            f"{'n/a' if margin is None else f'{float(margin):+.4f}'} | "
            f"{row['expected_count']}/{row['contrary_count']} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--shadow-library", type=Path, required=True)
    parser.add_argument("--consequence-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--min-relation-count", type=int, default=10)
    parser.add_argument("--min-outcome-margin", type=float, default=0.0)
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = apply_gate(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    gate = payload["rule_shadow_consequence_gate"]
    print(f"wrote consequence-gated rule shadow library: {args.output}")
    print(f"kept={gate['num_kept']} rejected={gate['num_rejected']} status_counts={gate['status_counts']}")


if __name__ == "__main__":
    main()
