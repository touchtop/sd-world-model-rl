#!/usr/bin/env python3
"""Build a shadow rule library from TVA context-action competition.

This script is intentionally output-only. It reads an existing confidence rule
library and a read-only TVA context competition audit, then writes a proposed
shadow library in which rules that lose same-context action-family competition
are weakened, while TVA-supported rules may receive a small confidence lift.

The research intent is to test whether replay + SD rollout + TVA terminal value
can provide a generic rule-value signal, without encoding Hopper-specific
posture goals into the rule criterion.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apply_rule_effect_confidence import clipped, confidence_status, rule_id
from common import load_json, write_json


def flatten_audit(audit: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Return per-rule TVA competition rows and rule ids involved in conflicts."""

    rows_by_id: dict[str, dict[str, Any]] = {}
    conflict_ids: set[str] = set()
    for context in audit.get("contexts", []):
        context_key = str(context.get("context_key", ""))
        num_records = int(context.get("num_records", 0) or 0)
        labels = context.get("labels", {})
        events = context.get("events", {})
        best_family_counts = context.get("best_family_counts", {})
        for conflict in context.get("conflicts", []):
            for key in ("supported", "suspect"):
                conflict_ids.update(str(rid) for rid in conflict.get(key, []))
        for row in context.get("rules", []):
            rid = str(row.get("id", ""))
            if not rid:
                continue
            rows_by_id[rid] = {
                **row,
                "context_key": context_key,
                "context_records": num_records,
                "context_labels": labels,
                "context_events": events,
                "context_best_family_counts": best_family_counts,
                "context_has_conflict": bool(context.get("conflicts")),
            }
    return rows_by_id, conflict_ids


def evidence_strength(row: dict[str, Any], opposite_margin_scale: float) -> float:
    best = clipped(float(row.get("best_rate", 0.0) or 0.0))
    top2 = clipped(float(row.get("top2_rate", 0.0) or 0.0))
    margin = float(row.get("mean_opposite_margin", 0.0) or 0.0)
    margin_score = clipped(max(0.0, margin) / max(opposite_margin_scale, 1e-6))
    return max(best, 0.5 * top2, margin_score)


def update_rule(
    rule: dict[str, Any],
    row: dict[str, Any] | None,
    conflict_ids: set[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(rule)
    rid = rule_id(rule)
    previous = clipped(float(out.get("confidence", args.default_confidence) or 0.0))
    confidence = previous
    reason = "not_in_tva_context_competition_audit"
    components: dict[str, Any] = {"previous_confidence": previous}

    if row is not None:
        supported = bool(row.get("tva_supported", False))
        suspect = bool(row.get("tva_suspect", False))
        strength = evidence_strength(row, args.opposite_margin_scale)
        components.update(
            {
                "context_key": row.get("context_key"),
                "context_records": row.get("context_records"),
                "best_rate": row.get("best_rate"),
                "top2_rate": row.get("top2_rate"),
                "mean_tva_value": row.get("mean_tva_value"),
                "mean_advantage_over_actual": row.get("mean_advantage_over_actual"),
                "mean_opposite_margin": row.get("mean_opposite_margin"),
                "tva_supported": supported,
                "tva_suspect": suspect,
                "context_has_conflict": row.get("context_has_conflict"),
                "evidence_strength": strength,
            }
        )
        if suspect:
            confidence = clipped(previous * args.suspect_decay)
            if args.demote_suspect_below_active:
                confidence = min(confidence, args.active_threshold - args.conflict_demote_margin)
            reason = "tva_context_competition_suspect_weakened"
        elif supported:
            confidence = clipped(previous + args.support_lr * strength * (1.0 - previous))
            reason = "tva_context_competition_supported_strengthened"
        elif rid in conflict_ids:
            confidence = clipped(previous * args.conflict_ambiguous_decay)
            if args.demote_ambiguous_conflict_below_active:
                confidence = min(confidence, args.active_threshold - args.conflict_demote_margin)
            reason = "tva_context_competition_conflict_ambiguous_soft_decay"
        else:
            reason = "tva_context_competition_observed_no_change"

    confidence = clipped(confidence)
    status = confidence_status(confidence, args.active_threshold, args.retired_threshold)
    out["confidence_before_tva_context_competition_shadow"] = round(previous, 6)
    out["confidence"] = round(confidence, 6)
    out["confidence_status"] = status
    out["status"] = status
    out["effective_prior_scale"] = round(confidence, 6)
    out["last_tva_context_competition_shadow_update_utc"] = datetime.now(timezone.utc).isoformat()
    out["last_tva_context_competition_shadow_reason"] = reason
    notes = list(out.get("audit_notes", []))
    if reason != "not_in_tva_context_competition_audit":
        notes.append("TVA 上下文动作竞争影子更新：同一状态上下文内按 SD 推演与 TVA 终端价值比较动作族。")
    out["audit_notes"] = notes

    history = list(out.get("tva_context_competition_shadow_history", []))
    history.append(
        {
            "updated_at_utc": out["last_tva_context_competition_shadow_update_utc"],
            "reason": reason,
            "confidence_before": round(previous, 6),
            "confidence_after": round(confidence, 6),
            "source_audit": str(args.audit),
            **components,
        }
    )
    out["tva_context_competition_shadow_history"] = history

    return out, {
        "id": rid,
        "previous_confidence": previous,
        "confidence": confidence,
        "confidence_status": status,
        "reason": reason,
        **components,
    }


def build_shadow(args: argparse.Namespace) -> dict[str, Any]:
    library = load_json(args.library)
    audit = load_json(args.audit)
    rows_by_id, conflict_ids = flatten_audit(audit)
    target_statuses = set(args.target_status)

    updated_rules: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for rule in library.get("rules", []):
        if target_statuses and rule.get("status") not in target_statuses:
            updated_rules.append(rule)
            continue
        updated, row = update_rule(rule, rows_by_id.get(rule_id(rule)), conflict_ids, args)
        updated_rules.append(updated)
        update_rows.append(row)

    status_counts = Counter(str(rule.get("status")) for rule in updated_rules)
    reason_counts = Counter(str(row.get("reason")) for row in update_rows)
    delta_rows = [
        row
        for row in update_rows
        if abs(float(row.get("confidence", 0.0)) - float(row.get("previous_confidence", 0.0))) > 1e-9
    ]

    out = dict(library)
    out["rules"] = updated_rules
    out["library_version"] = f"{library.get('library_version', 'rule_library')}_tva_context_competition_shadow"
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["source_library"] = str(args.library)
    out["tva_context_competition_shadow_update"] = {
        "audit": str(args.audit),
        "target_statuses": sorted(target_statuses),
        "parameters": {
            "active_threshold": args.active_threshold,
            "retired_threshold": args.retired_threshold,
            "default_confidence": args.default_confidence,
            "support_lr": args.support_lr,
            "suspect_decay": args.suspect_decay,
            "conflict_ambiguous_decay": args.conflict_ambiguous_decay,
            "conflict_demote_margin": args.conflict_demote_margin,
            "opposite_margin_scale": args.opposite_margin_scale,
            "demote_suspect_below_active": args.demote_suspect_below_active,
            "demote_ambiguous_conflict_below_active": args.demote_ambiguous_conflict_below_active,
        },
        "source_audit_summary": audit.get("summary", {}),
        "num_rules_seen_in_audit": len(rows_by_id),
        "num_updated_rules": len(update_rows),
        "num_confidence_changed": len(delta_rows),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "updates": sorted(update_rows, key=lambda row: (row["reason"], row["confidence"], row["id"])),
    }
    out["online_enabled"] = any(rule.get("status") == "confidence_active" for rule in updated_rules)
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    update = payload["tva_context_competition_shadow_update"]
    lines = [
        "# TVA Context Competition Shadow Library",
        "",
        f"Source library: `{payload['source_library']}`",
        f"Audit: `{update['audit']}`",
        f"Updated rules: `{update['num_updated_rules']}`",
        f"Confidence changed: `{update['num_confidence_changed']}`",
        f"Status counts: `{update['status_counts']}`",
        f"Reason counts: `{update['reason_counts']}`",
        "",
        "This is an eval/shadow artifact. It does not modify the live rule library.",
        "",
        "## Changed Rules",
        "",
        "| status | confidence | delta | reason | rule | context | best% | top2% | opposite margin |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for row in update["updates"]:
        before = float(row.get("previous_confidence", 0.0) or 0.0)
        after = float(row.get("confidence", 0.0) or 0.0)
        if abs(after - before) <= 1e-9:
            continue
        lines.append(
            "| "
            f"`{row['confidence_status']}` | "
            f"{after:.3f} | "
            f"{after - before:+.3f} | "
            f"`{row['reason']}` | "
            f"`{row['id']}` | "
            f"`{row.get('context_key', '')}` | "
            f"{100*float(row.get('best_rate', 0.0) or 0.0):.1f} | "
            f"{100*float(row.get('top2_rate', 0.0) or 0.0):.1f} | "
            f"{float(row.get('mean_opposite_margin', 0.0) or 0.0):.4f} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-status", action="append", default=["confidence_active", "confidence_observing"])
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--default-confidence", type=float, default=0.56)
    parser.add_argument("--support-lr", type=float, default=0.10)
    parser.add_argument("--suspect-decay", type=float, default=0.50)
    parser.add_argument("--conflict-ambiguous-decay", type=float, default=0.90)
    parser.add_argument("--conflict-demote-margin", type=float, default=0.05)
    parser.add_argument("--opposite-margin-scale", type=float, default=2.0)
    parser.add_argument("--demote-suspect-below-active", action="store_true", default=True)
    parser.add_argument("--no-demote-suspect-below-active", dest="demote_suspect_below_active", action="store_false")
    parser.add_argument("--demote-ambiguous-conflict-below-active", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_shadow(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    update = payload["tva_context_competition_shadow_update"]
    print(f"wrote TVA context competition shadow library: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(f"changed={update['num_confidence_changed']} status_counts={update['status_counts']}")
    print(f"reason_counts={update['reason_counts']}")


if __name__ == "__main__":
    main()
