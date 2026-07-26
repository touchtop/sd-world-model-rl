#!/usr/bin/env python3
"""Build a role/effect-gated TVA context-competition shadow library.

This script is a conservative follow-up to
`build_tva_context_competition_shadow_library.py`. It keeps TVA context action
competition as a value signal, but does not let TVA alone demote a rule that is
behaviorally consistent in real trajectories. This protects low-return recovery
rules whose terminal TVA value may be weak even though they stabilize necessary
intermediate states.

The output is still shadow/eval-only and does not modify the source rule
library.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apply_rule_effect_confidence import clipped, confidence_status, rule_id
from build_tva_context_competition_shadow_library import evidence_strength, flatten_audit
from common import load_json, write_json


def load_effect_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row.get("id")): row for row in payload.get("rule_summaries", []) if row.get("id")}


def classify_role(effect: dict[str, Any] | None, args: argparse.Namespace) -> str:
    """Classify rule role from observed behavior rather than task-specific labels."""

    if effect is None:
        return "unknown"
    hits = int(effect.get("selected_gate_hits", 0) or 0)
    if hits < args.min_effect_hits:
        return "low_evidence"
    consistent = clipped(float(effect.get("direction_consistent_rate", 0.0) or 0.0))
    harmful = clipped(float(effect.get("direction_harmful_rate", 0.0) or 0.0))
    good_state = clipped(float(effect.get("good_state_hit_rate", 0.0) or 0.0))
    high_ep = clipped(float(effect.get("high_episode_hit_rate", 0.0) or 0.0))
    if consistent >= args.recovery_consistent_min and max(good_state, high_ep) <= args.recovery_context_max:
        return "recovery_consistent"
    if max(good_state, high_ep) >= args.progress_context_min:
        return "progress_context"
    if harmful >= args.effect_harmful_min:
        return "effect_harmful"
    return "neutral"


def should_demote(
    row: dict[str, Any],
    effect: dict[str, Any] | None,
    role: str,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """Decide whether TVA-suspect evidence is allowed to weaken confidence."""

    if not bool(row.get("tva_suspect", False)):
        return False, "not_tva_suspect"
    if effect is None:
        return False, "tva_suspect_but_no_effect_evidence"
    hits = int(effect.get("selected_gate_hits", 0) or 0)
    if hits < args.min_effect_hits:
        return False, "tva_suspect_but_low_effect_hits"
    consistent = clipped(float(effect.get("direction_consistent_rate", 0.0) or 0.0))
    harmful = clipped(float(effect.get("direction_harmful_rate", 0.0) or 0.0))
    if role == "recovery_consistent":
        return False, "tva_suspect_but_recovery_consistent_protected"
    if harmful >= args.effect_harmful_min:
        return True, "tva_suspect_and_effect_harmful"
    if consistent <= args.effect_consistent_max_for_demote:
        return True, "tva_suspect_and_effect_not_consistent"
    return False, "tva_suspect_but_effect_consistent"


def should_strengthen(
    row: dict[str, Any],
    effect: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if not bool(row.get("tva_supported", False)):
        return False, "not_tva_supported"
    if effect is None:
        return False, "tva_supported_but_no_effect_evidence"
    hits = int(effect.get("selected_gate_hits", 0) or 0)
    if hits < args.min_effect_hits:
        return False, "tva_supported_but_low_effect_hits"
    harmful = clipped(float(effect.get("direction_harmful_rate", 0.0) or 0.0))
    consistent = clipped(float(effect.get("direction_consistent_rate", 0.0) or 0.0))
    if harmful >= args.effect_harmful_block_strengthen:
        return False, "tva_supported_but_effect_harmful"
    if consistent < args.effect_consistent_min_for_strengthen:
        return False, "tva_supported_but_low_effect_consistency"
    return True, "tva_supported_and_effect_consistent"


def update_rule(
    rule: dict[str, Any],
    row: dict[str, Any] | None,
    effect: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(rule)
    rid = rule_id(rule)
    previous = clipped(float(out.get("confidence", args.default_confidence) or 0.0))
    confidence = previous
    role = classify_role(effect, args)
    reason = "not_in_tva_context_competition_audit"
    components: dict[str, Any] = {
        "previous_confidence": previous,
        "role": role,
    }

    if effect is not None:
        components.update(
            {
                "effect_selected_gate_hits": effect.get("selected_gate_hits"),
                "effect_direction_consistent_rate": effect.get("direction_consistent_rate"),
                "effect_direction_harmful_rate": effect.get("direction_harmful_rate"),
                "effect_good_state_hit_rate": effect.get("good_state_hit_rate"),
                "effect_high_episode_hit_rate": effect.get("high_episode_hit_rate"),
            }
        )

    if row is not None:
        strength = evidence_strength(row, args.opposite_margin_scale)
        components.update(
            {
                "context_key": row.get("context_key"),
                "context_records": row.get("context_records"),
                "best_rate": row.get("best_rate"),
                "top2_rate": row.get("top2_rate"),
                "mean_opposite_margin": row.get("mean_opposite_margin"),
                "tva_supported": bool(row.get("tva_supported", False)),
                "tva_suspect": bool(row.get("tva_suspect", False)),
                "evidence_strength": strength,
            }
        )
        demote, demote_reason = should_demote(row, effect, role, args)
        strengthen, strengthen_reason = should_strengthen(row, effect, args)
        if demote:
            confidence = clipped(previous * args.suspect_decay)
            if args.demote_suspect_below_active:
                confidence = min(confidence, args.active_threshold - args.conflict_demote_margin)
            reason = demote_reason
        elif strengthen:
            confidence = clipped(previous + args.support_lr * strength * (1.0 - previous))
            reason = strengthen_reason
        else:
            reason = demote_reason if bool(row.get("tva_suspect", False)) else strengthen_reason

    confidence = clipped(confidence)
    status = confidence_status(confidence, args.active_threshold, args.retired_threshold)
    out["confidence_before_tva_context_role_gated_shadow"] = round(previous, 6)
    out["confidence"] = round(confidence, 6)
    out["confidence_status"] = status
    out["status"] = status
    out["effective_prior_scale"] = round(confidence, 6)
    out["last_tva_context_role_gated_shadow_update_utc"] = datetime.now(timezone.utc).isoformat()
    out["last_tva_context_role_gated_shadow_reason"] = reason
    history = list(out.get("tva_context_role_gated_shadow_history", []))
    history.append(
        {
            "updated_at_utc": out["last_tva_context_role_gated_shadow_update_utc"],
            "reason": reason,
            "confidence_before": round(previous, 6),
            "confidence_after": round(confidence, 6),
            "source_tva_audit": str(args.tva_audit),
            "source_effect_audit": str(args.effect_audit),
            **components,
        }
    )
    out["tva_context_role_gated_shadow_history"] = history

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
    tva_audit = load_json(args.tva_audit)
    rows_by_id, _conflict_ids = flatten_audit(tva_audit)
    effect_by_id = load_effect_rows(args.effect_audit)
    target_statuses = set(args.target_status)

    updated_rules: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for rule in library.get("rules", []):
        if target_statuses and rule.get("status") not in target_statuses:
            updated_rules.append(rule)
            continue
        rid = rule_id(rule)
        updated, row = update_rule(rule, rows_by_id.get(rid), effect_by_id.get(rid), args)
        updated_rules.append(updated)
        update_rows.append(row)

    status_counts = Counter(str(rule.get("status")) for rule in updated_rules)
    reason_counts = Counter(str(row.get("reason")) for row in update_rows)
    role_counts = Counter(str(row.get("role")) for row in update_rows)
    delta_rows = [
        row
        for row in update_rows
        if abs(float(row.get("confidence", 0.0)) - float(row.get("previous_confidence", 0.0))) > 1e-9
    ]

    out = dict(library)
    out["rules"] = updated_rules
    out["library_version"] = f"{library.get('library_version', 'rule_library')}_tva_context_role_gated_shadow"
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["source_library"] = str(args.library)
    parameters = {
        **vars(args),
        "library": str(args.library),
        "tva_audit": str(args.tva_audit),
        "effect_audit": str(args.effect_audit),
        "output": str(args.output),
    }
    out["tva_context_role_gated_shadow_update"] = {
        "tva_audit": str(args.tva_audit),
        "effect_audit": str(args.effect_audit),
        "target_statuses": sorted(target_statuses),
        "parameters": parameters,
        "source_tva_audit_summary": tva_audit.get("summary", {}),
        "num_updated_rules": len(update_rows),
        "num_confidence_changed": len(delta_rows),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "role_counts": dict(role_counts),
        "updates": sorted(update_rows, key=lambda row: (row["reason"], row["confidence"], row["id"])),
    }
    out["online_enabled"] = any(rule.get("status") == "confidence_active" for rule in updated_rules)
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    update = payload["tva_context_role_gated_shadow_update"]
    lines = [
        "# TVA Context Role-Gated Shadow Library",
        "",
        f"Source library: `{payload['source_library']}`",
        f"TVA audit: `{update['tva_audit']}`",
        f"Effect audit: `{update['effect_audit']}`",
        f"Updated rules: `{update['num_updated_rules']}`",
        f"Confidence changed: `{update['num_confidence_changed']}`",
        f"Status counts: `{update['status_counts']}`",
        f"Role counts: `{update['role_counts']}`",
        f"Reason counts: `{update['reason_counts']}`",
        "",
        "This is a shadow artifact. TVA suspicion cannot demote behaviorally consistent recovery rules by itself.",
        "",
        "## Changed Rules",
        "",
        "| status | confidence | delta | role | reason | rule | hits | consistent | harmful |",
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
            f"`{row.get('role', '')}` | "
            f"`{row['reason']}` | "
            f"`{row['id']}` | "
            f"{int(row.get('effect_selected_gate_hits', 0) or 0)} | "
            f"{float(row.get('effect_direction_consistent_rate', 0.0) or 0.0):.3f} | "
            f"{float(row.get('effect_direction_harmful_rate', 0.0) or 0.0):.3f} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tva-audit", type=Path, required=True)
    parser.add_argument("--effect-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-status", action="append", default=["confidence_active", "confidence_observing"])
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--default-confidence", type=float, default=0.56)
    parser.add_argument("--support-lr", type=float, default=0.10)
    parser.add_argument("--suspect-decay", type=float, default=0.50)
    parser.add_argument("--conflict-demote-margin", type=float, default=0.05)
    parser.add_argument("--opposite-margin-scale", type=float, default=2.0)
    parser.add_argument("--min-effect-hits", type=int, default=20)
    parser.add_argument("--recovery-consistent-min", type=float, default=0.80)
    parser.add_argument("--recovery-context-max", type=float, default=0.05)
    parser.add_argument("--progress-context-min", type=float, default=0.20)
    parser.add_argument("--effect-harmful-min", type=float, default=0.45)
    parser.add_argument("--effect-consistent-max-for-demote", type=float, default=0.55)
    parser.add_argument("--effect-harmful-block-strengthen", type=float, default=0.40)
    parser.add_argument("--effect-consistent-min-for-strengthen", type=float, default=0.60)
    parser.add_argument("--demote-suspect-below-active", action="store_true", default=True)
    parser.add_argument("--no-demote-suspect-below-active", dest="demote_suspect_below_active", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_shadow(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    update = payload["tva_context_role_gated_shadow_update"]
    print(f"wrote TVA context role-gated shadow library: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(f"changed={update['num_confidence_changed']} status_counts={update['status_counts']}")
    print(f"reason_counts={update['reason_counts']}")


if __name__ == "__main__":
    main()
