#!/usr/bin/env python3
"""Apply a read-only consequence gate to TVA shadow update proposals.

The input shadow update remains the source of TVA value proposals. This script
only creates a second shadow report in which unsupported proposals are reverted
to their original confidence. It does not modify the live rule library.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from apply_rule_effect_confidence import confidence_status  # noqa: E402
from common import load_json, write_json  # noqa: E402


DEFAULT_REPORT_DIR = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "tva_validation_300k_oldscheme_20260515"
)


def consequence_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row["id"]): row for row in payload.get("rules", [])}


def previous_confidence(row: dict[str, Any]) -> float:
    return float(row.get("confidence", row.get("source_confidence", 0.0)) or 0.0)


def consequence_gate_passes(row: dict[str, Any], evidence: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    delta = float(row.get("shadow_confidence_delta", 0.0) or 0.0)
    if abs(delta) <= args.delta_epsilon:
        return True
    if evidence is None:
        return False
    if not bool(evidence.get("consequence_supported", False)):
        return False
    margin = evidence.get("outcome_margin")
    if margin is None or float(margin) < args.min_outcome_margin:
        return False
    expected_count = int(evidence.get("expected_count", 0) or 0)
    contrary_count = int(evidence.get("contrary_count", 0) or 0)
    return expected_count >= args.min_relation_count and contrary_count >= args.min_relation_count


def apply_gate(shadow: dict[str, Any], consequence: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    original_changed = 0
    kept_changed = 0
    rejected = 0
    deltas: list[float] = []
    for row in shadow.get("rules", []):
        out = dict(row)
        delta = float(row.get("shadow_confidence_delta", 0.0) or 0.0)
        evidence = consequence.get(str(row.get("id")))
        out["consequence_gate"] = {
            "source_consequence_audit": str(args.consequence_audit),
            "enabled": True,
            "consequence_supported": bool(evidence.get("consequence_supported", False)) if evidence else False,
            "support_reason": evidence.get("support_reason") if evidence else "missing_consequence_evidence",
            "outcome_margin": evidence.get("outcome_margin") if evidence else None,
            "expected_relation": evidence.get("expected_relation") if evidence else None,
            "contrary_relation": evidence.get("contrary_relation") if evidence else None,
            "expected_count": int(evidence.get("expected_count", 0) or 0) if evidence else 0,
            "contrary_count": int(evidence.get("contrary_count", 0) or 0) if evidence else 0,
            "min_outcome_margin": float(args.min_outcome_margin),
            "min_relation_count": int(args.min_relation_count),
        }
        if abs(delta) > args.delta_epsilon:
            original_changed += 1
        if consequence_gate_passes(row, evidence, args):
            if abs(delta) > args.delta_epsilon:
                kept_changed += 1
        else:
            if abs(delta) > args.delta_epsilon:
                rejected += 1
            previous = previous_confidence(row)
            out["shadow_confidence"] = round(previous, 6)
            out["shadow_confidence_status"] = confidence_status(
                previous,
                float(shadow.get("active_threshold", args.active_threshold)),
                float(shadow.get("retired_threshold", args.retired_threshold)),
            )
            out["shadow_effective_prior_scale"] = round(previous, 6)
            out["shadow_update_reason_before_consequence_gate"] = row.get("shadow_update_reason")
            out["shadow_update_reason"] = "tva_shadow_rejected_by_consequence_gate"
            out["shadow_confidence_delta"] = 0.0
            out["shadow_status_change"] = f"{row.get('confidence_status', row.get('status'))} -> {out['shadow_confidence_status']}"
            out["shadow_last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        deltas.append(float(out.get("shadow_confidence_delta", 0.0) or 0.0))
        rows.append(out)

    summary = {
        "num_rules": len(rows),
        "num_strengthened": sum(1 for row in rows if float(row.get("shadow_confidence_delta", 0.0) or 0.0) > args.delta_epsilon),
        "num_weakened": sum(1 for row in rows if float(row.get("shadow_confidence_delta", 0.0) or 0.0) < -args.delta_epsilon),
        "num_unchanged": sum(1 for row in rows if abs(float(row.get("shadow_confidence_delta", 0.0) or 0.0)) <= args.delta_epsilon),
        "num_retired": sum(1 for row in rows if row.get("shadow_confidence_status") == "confidence_retired"),
        "num_active": sum(1 for row in rows if row.get("shadow_confidence_status") == "confidence_active"),
        "num_observing": sum(1 for row in rows if row.get("shadow_confidence_status") == "confidence_observing"),
        "mean_shadow_delta": float(mean(deltas)) if deltas else 0.0,
        "median_shadow_delta": sorted(deltas)[len(deltas) // 2] if deltas else 0.0,
        "max_shadow_delta": max(deltas) if deltas else 0.0,
        "min_shadow_delta": min(deltas) if deltas else 0.0,
        "num_original_changed": original_changed,
        "num_consequence_kept_changed": kept_changed,
        "num_consequence_rejected_changed": rejected,
    }
    payload = dict(shadow)
    payload["audit_family"] = "hopper_tva_shadow_update_consequence_gated_v1"
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["source_shadow_update"] = str(args.shadow_update)
    payload["source_consequence_audit"] = str(args.consequence_audit)
    payload["consequence_gate_min_outcome_margin"] = float(args.min_outcome_margin)
    payload["consequence_gate_min_relation_count"] = int(args.min_relation_count)
    payload["summary"] = summary
    payload["rules"] = rows
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TVA Shadow Consequence-Gated Update",
        "",
        "This is an audit-only shadow report. It does not change the live rule library or planner prior.",
        "",
        f"Source shadow update: `{payload['source_shadow_update']}`",
        f"Source consequence audit: `{payload['source_consequence_audit']}`",
        f"Minimum outcome margin: `{payload['consequence_gate_min_outcome_margin']}`",
        f"Minimum relation count: `{payload['consequence_gate_min_relation_count']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "num_rules",
        "num_original_changed",
        "num_consequence_kept_changed",
        "num_consequence_rejected_changed",
        "num_strengthened",
        "num_weakened",
        "num_unchanged",
        "mean_shadow_delta",
        "max_shadow_delta",
        "min_shadow_delta",
    ):
        value = payload["summary"].get(key)
        if isinstance(value, float):
            lines.append(f"| `{key}` | `{value:.6f}` |")
        else:
            lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Changed Rules",
            "",
            "| Rule | Delta | Gate | Margin | Counts | Reason |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    changed = [row for row in payload["rules"] if abs(float(row.get("shadow_confidence_delta", 0.0) or 0.0)) > 1e-6]
    rejected = [
        row
        for row in payload["rules"]
        if row.get("shadow_update_reason") == "tva_shadow_rejected_by_consequence_gate"
    ]
    for row in changed + rejected:
        gate = row.get("consequence_gate", {})
        margin = gate.get("outcome_margin")
        lines.append(
            "| "
            f"`{row['id']}` | "
            f"`{float(row.get('shadow_confidence_delta', 0.0) or 0.0):+.3f}` | "
            f"`{'kept' if row in changed else 'rejected'}` | "
            f"{'`n/a`' if margin is None else f'`{float(margin):+.4f}`'} | "
            f"`{gate.get('expected_count', 0)}`/`{gate.get('contrary_count', 0)}` | "
            f"`{row.get('shadow_update_reason')}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-update", type=Path, required=True)
    parser.add_argument("--consequence-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-outcome-margin", type=float, default=0.0)
    parser.add_argument("--min-relation-count", type=int, default=10)
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shadow = load_json(args.shadow_update)
    consequence = consequence_index(args.consequence_audit)
    payload = apply_gate(shadow, consequence, args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote consequence-gated TVA shadow update: {args.output}")
    print(f"kept changed rules: {payload['summary']['num_consequence_kept_changed']}")
    print(f"rejected changed rules: {payload['summary']['num_consequence_rejected_changed']}")


if __name__ == "__main__":
    main()
