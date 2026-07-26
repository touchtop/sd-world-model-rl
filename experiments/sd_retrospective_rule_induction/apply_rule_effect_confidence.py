#!/usr/bin/env python3
"""Update rule confidence from rule-effect audit statistics.

This script consumes a rule library and a behavioral rule-effect audit, then
writes a new confidence-weighted library. It is intentionally output-only: the
source library and experiment logs are not modified.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import load_json, write_json


DEFAULT_LIBRARY = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/"
    "cycle_019/rules/library_closed_loop_cycle_019.json"
)
DEFAULT_AUDIT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_cycle019_active_20260509.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/rules/"
    "hopper_v8phaseany_widefunnel_cycle019_effect_confidence_20260509.json"
)


def rule_id(rule: dict[str, Any]) -> str:
    return str(
        rule.get("id")
        or f"{rule.get('rule_key') or rule.get('substate_key')}->{rule.get('intervention_type')}:{rule.get('action_family')}"
    )


def confidence_status(confidence: float, active_threshold: float, retired_threshold: float) -> str:
    if confidence >= active_threshold:
        return "confidence_active"
    if confidence <= retired_threshold:
        return "confidence_retired"
    return "confidence_observing"


def clipped(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def log_hit_weight(hits: int, hit_ref: float) -> float:
    if hits <= 0:
        return 0.0
    return min(1.0, math.log1p(float(hits)) / math.log1p(max(hit_ref, 1.0)))


def rule_state_key(rule: dict[str, Any]) -> str:
    return str(rule.get("rule_key") or rule.get("substate_key") or "")


def effective_action_direction(rule: dict[str, Any]) -> tuple[str, str] | None:
    """Return the effective preferred direction used for conflict checks.

    `avoid:hip_neg` and `soft_prefer:hip_pos` both discourage negative hip
    actions under the current penalty-only online prior, so they are treated as
    the same effective direction rather than as a conflict.
    """

    family = str(rule.get("action_family", ""))
    if "_" not in family:
        return None
    joint, sign = family.split("_", 1)
    if sign not in {"pos", "neg"}:
        return None
    if rule.get("intervention_type") == "avoid_action_family":
        sign = "neg" if sign == "pos" else "pos"
    return joint, sign


def latest_effect_metric(rule: dict[str, Any], key: str, default: float = 0.0) -> float:
    history = rule.get("effect_confidence_history", [])
    if history:
        try:
            return float(history[-1].get(key, default) or default)
        except (TypeError, ValueError):
            return default
    return default


def protected_teacher_rule(rule: dict[str, Any], args: argparse.Namespace) -> bool:
    origins = {item.strip() for item in str(args.protected_teacher_origins).split(",") if item.strip()}
    return bool(args.protect_teacher_conflicts and str(rule.get("rule_origin", "")) in origins)


def protected_teacher_low_evidence(rule: dict[str, Any], args: argparse.Namespace) -> bool:
    origins = {item.strip() for item in str(args.protected_teacher_origins).split(",") if item.strip()}
    return bool(args.protect_teacher_low_evidence and str(rule.get("rule_origin", "")) in origins)


def rule_conflict_stats(rule: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, float]:
    row = rows_by_id.get(rule_id(rule), {})
    return {
        "confidence": float(rule.get("confidence", 0.0) or 0.0),
        "harmful": float(row.get("direction_harmful_rate", latest_effect_metric(rule, "direction_harmful_rate", 0.0)) or 0.0),
        "hits": float(row.get("hits", latest_effect_metric(rule, "hits", 0.0)) or 0.0),
    }


def update_confidence(
    rule: dict[str, Any],
    stats: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(rule)
    rid = rule_id(rule)
    previous = float(out.get("confidence", args.default_confidence) or 0.0)
    previous = clipped(previous)
    confidence = previous
    reason = "unchanged"
    components: dict[str, Any] = {
        "previous_confidence": previous,
        "rule_effect_audit_id": rid,
    }
    teacher_low_evidence_protected = protected_teacher_low_evidence(rule, args)

    if stats is None:
        if teacher_low_evidence_protected:
            reason = "teacher_missing_from_effect_audit_retained"
            components.update({"teacher_low_evidence_protected": True})
        else:
            confidence *= args.absence_decay
            reason = "missing_from_effect_audit_absence_decay"
            components.update({"absence_decay": args.absence_decay})
    else:
        hits = int(stats.get("selected_gate_hits", 0) or 0)
        harmful = clipped(float(stats.get("direction_harmful_rate", 0.0) or 0.0))
        consistent = clipped(float(stats.get("direction_consistent_rate", 0.0) or 0.0))
        high_ep = clipped(float(stats.get("high_episode_hit_rate", 0.0) or 0.0))
        good_state = clipped(float(stats.get("good_state_hit_rate", 0.0) or 0.0))
        hit_weight = log_hit_weight(hits, args.hit_ref)
        positive_context = max(high_ep, good_state)
        context_weight = max(args.context_floor, positive_context)
        if hits == 0:
            if teacher_low_evidence_protected:
                reason = "teacher_zero_effect_hits_retained"
                components.update({"hits": hits, "teacher_low_evidence_protected": True})
            else:
                confidence *= args.absence_decay
                reason = "zero_effect_hits_absence_decay"
                components.update({"hits": hits, "absence_decay": args.absence_decay})
        elif hits < args.min_effect_hits:
            if teacher_low_evidence_protected:
                reason = "teacher_low_effect_hits_retained"
                components.update({"hits": hits, "teacher_low_evidence_protected": True})
            else:
                confidence *= args.low_hit_decay
                reason = "low_effect_hits_soft_decay"
                components.update({"hits": hits, "low_hit_decay": args.low_hit_decay})
        else:
            positive_excess = max(0.0, consistent - args.positive_threshold) / max(
                1.0 - args.positive_threshold,
                1e-6,
            )
            harmful_excess = max(0.0, harmful - args.harmful_threshold) / max(
                1.0 - args.harmful_threshold,
                1e-6,
            )
            context_ok = positive_context >= args.positive_context_threshold
            positive_delta = (
                args.positive_lr * positive_excess * hit_weight * context_weight
                if context_ok
                else 0.0
            )
            harmful_delta = args.harmful_lr * harmful_excess * hit_weight * context_weight
            hard_harmful = hits >= args.hard_harmful_min_hits and harmful >= args.hard_harmful_threshold
            if teacher_low_evidence_protected and not hard_harmful:
                confidence = clipped(confidence + positive_delta)
                if positive_delta > 0:
                    reason = "teacher_effect_consistent_strengthened"
                else:
                    reason = "teacher_effect_retained_no_strong_counterexample"
            else:
                confidence = clipped(confidence + positive_delta - harmful_delta)
            if hard_harmful:
                confidence = min(confidence, max(args.retired_threshold, args.active_threshold - args.hard_harmful_margin))
                reason = (
                    "teacher_strong_counterexample_deactivated"
                    if teacher_low_evidence_protected
                    else "effect_harmful_deactivated"
                )
            elif (not teacher_low_evidence_protected) and harmful_delta > positive_delta and harmful >= args.harmful_threshold:
                reason = "effect_harmful_weakened"
            elif (not teacher_low_evidence_protected) and positive_delta > 0:
                reason = "effect_consistent_strengthened"
            elif (not teacher_low_evidence_protected) and positive_excess > 0 and not context_ok:
                reason = "effect_consistent_but_low_context"
            elif not teacher_low_evidence_protected:
                reason = "effect_observed_no_large_change"
            components.update(
                {
                    "hits": hits,
                    "direction_consistent_rate": consistent,
                    "direction_harmful_rate": harmful,
                    "high_episode_hit_rate": high_ep,
                    "good_state_hit_rate": good_state,
                    "positive_context": positive_context,
                    "positive_context_threshold": args.positive_context_threshold,
                    "positive_context_ok": context_ok,
                    "hit_weight": hit_weight,
                    "context_weight": context_weight,
                    "positive_delta": positive_delta,
                    "harmful_delta": harmful_delta,
                    "hard_harmful": hard_harmful,
                    "teacher_low_evidence_protected": teacher_low_evidence_protected,
                }
            )

    status = confidence_status(confidence, args.active_threshold, args.retired_threshold)
    out["pre_confidence_status"] = rule.get("status")
    out["confidence"] = round(confidence, 6)
    out["confidence_status"] = status
    out["status"] = status
    out["effective_prior_scale"] = round(confidence, 6)
    out["last_effect_confidence_update_utc"] = datetime.now(timezone.utc).isoformat()
    out["last_effect_confidence_reason"] = reason
    history = list(out.get("effect_confidence_history", []))
    history.append(
        {
            "updated_at_utc": out["last_effect_confidence_update_utc"],
            "reason": reason,
            "confidence_before": round(previous, 6),
            "confidence_after": round(confidence, 6),
            "source_audit": str(args.audit),
            **components,
        }
    )
    out["effect_confidence_history"] = history
    return out, {
        "id": rid,
        "previous_confidence": previous,
        "confidence": confidence,
        "confidence_status": status,
        "reason": reason,
        **components,
    }


def resolve_active_conflicts(
    rules: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, list[int]]] = {}
    for idx, rule in enumerate(rules):
        if rule.get("status") != "confidence_active":
            continue
        direction = effective_action_direction(rule)
        if direction is None:
            continue
        joint, sign = direction
        key = (rule_state_key(rule), joint)
        groups.setdefault(key, {"pos": [], "neg": []})[sign].append(idx)

    for (_state_key, _joint), signs in groups.items():
        if not signs["pos"] or not signs["neg"]:
            continue
        idxs = signs["pos"] + signs["neg"]

        def sort_key(idx: int) -> tuple[float, float, float]:
            stats = rule_conflict_stats(rules[idx], rows_by_id)
            return stats["confidence"], -stats["harmful"], stats["hits"]

        ranked = sorted(idxs, key=sort_key, reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        best_stats = rule_conflict_stats(rules[best], rows_by_id)
        second_stats = rule_conflict_stats(rules[second], rows_by_id) if second is not None else {
            "confidence": 0.0,
            "harmful": 1.0,
            "hits": 0.0,
        }
        keep_best = (
            best_stats["confidence"] - second_stats["confidence"] >= args.conflict_keep_confidence_margin
            and second_stats["harmful"] - best_stats["harmful"] >= args.conflict_keep_harmful_margin
        )
        demote = [idx for idx in idxs if idx != best] if keep_best else idxs

        protected_idxs = [idx for idx in idxs if protected_teacher_rule(rules[idx], args)]
        if protected_idxs:
            protected_best = sorted(protected_idxs, key=sort_key, reverse=True)[0]
            protected_stats = rule_conflict_stats(rules[protected_best], rows_by_id)
            challenger_idxs = [idx for idx in idxs if idx not in protected_idxs]
            challenger_best = sorted(challenger_idxs, key=sort_key, reverse=True)[0] if challenger_idxs else None
            challenger_stats = (
                rule_conflict_stats(rules[challenger_best], rows_by_id)
                if challenger_best is not None
                else {"confidence": 0.0, "harmful": 1.0, "hits": 0.0}
            )
            challenger_can_override = (
                challenger_best is not None
                and challenger_stats["hits"] >= args.teacher_conflict_min_challenger_hits
                and challenger_stats["confidence"] - protected_stats["confidence"] >= args.teacher_conflict_confidence_margin
                and protected_stats["harmful"] - challenger_stats["harmful"] >= args.teacher_conflict_harmful_margin
            )
            if challenger_can_override:
                demote = [idx for idx in idxs if idx != challenger_best]
            else:
                demote = [idx for idx in idxs if idx not in protected_idxs]
                for idx in protected_idxs:
                    rid = rule_id(rules[idx])
                    rules[idx]["last_effect_confidence_reason"] = "teacher_core_conflict_protected"
                    notes = list(rules[idx].get("audit_notes", []))
                    notes.append("教师核心初始规则受到冲突保护；反向候选证据不足时不降为观察状态。")
                    rules[idx]["audit_notes"] = notes
                    if rid in rows_by_id:
                        rows_by_id[rid]["reason"] = "teacher_core_conflict_protected"
                        rows_by_id[rid]["teacher_core_conflict_protected"] = True

        for idx in demote:
            rule = rules[idx]
            rid = rule_id(rule)
            rule["confidence"] = min(float(rule.get("confidence", 0.0) or 0.0), args.active_threshold - args.conflict_demote_margin)
            rule["confidence"] = round(max(args.retired_threshold, rule["confidence"]), 6)
            rule["confidence_status"] = confidence_status(rule["confidence"], args.active_threshold, args.retired_threshold)
            rule["status"] = rule["confidence_status"]
            rule["effective_prior_scale"] = rule["confidence"]
            rule["last_effect_confidence_reason"] = "active_conflict_demoted"
            notes = list(rule.get("audit_notes", []))
            notes.append("同一状态同一关节存在相反有效动作方向；按置信与有害率仲裁降为观察状态。")
            rule["audit_notes"] = notes
            if rid in rows_by_id:
                rows_by_id[rid]["confidence"] = rule["confidence"]
                rows_by_id[rid]["confidence_status"] = rule["status"]
                rows_by_id[rid]["reason"] = "active_conflict_demoted"
                rows_by_id[rid]["conflict_demoted"] = True
    return rules


def apply_update(args: argparse.Namespace) -> dict[str, Any]:
    library = load_json(args.library)
    audit = load_json(args.audit)
    stats_by_id = {str(row.get("id")): row for row in audit.get("rule_summaries", [])}
    target_statuses = set(args.target_status)

    updated_rules: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for rule in library.get("rules", []):
        rid = rule_id(rule)
        if target_statuses and rule.get("status") not in target_statuses:
            updated_rules.append(rule)
            continue
        updated, row = update_confidence(rule, stats_by_id.get(rid), args)
        updated_rules.append(updated)
        update_rows.append(row)

    rows_by_id = {str(row["id"]): row for row in update_rows}
    updated_rules = resolve_active_conflicts(updated_rules, rows_by_id, args)
    update_rows = list(rows_by_id.values())

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in update_rows:
        status_counts[row["confidence_status"]] = status_counts.get(row["confidence_status"], 0) + 1
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    out = dict(library)
    out["library_version"] = f"{library.get('library_version', 'rule_library')}_effect_confidence"
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["source_library"] = str(args.library)
    out["effect_confidence_update"] = {
        "audit": str(args.audit),
        "target_statuses": sorted(target_statuses),
        "parameters": {
            "default_confidence": args.default_confidence,
            "active_threshold": args.active_threshold,
            "retired_threshold": args.retired_threshold,
            "min_effect_hits": args.min_effect_hits,
            "hit_ref": args.hit_ref,
            "positive_threshold": args.positive_threshold,
            "harmful_threshold": args.harmful_threshold,
            "positive_lr": args.positive_lr,
            "harmful_lr": args.harmful_lr,
            "absence_decay": args.absence_decay,
            "low_hit_decay": args.low_hit_decay,
            "context_floor": args.context_floor,
            "positive_context_threshold": args.positive_context_threshold,
            "hard_harmful_threshold": args.hard_harmful_threshold,
            "hard_harmful_min_hits": args.hard_harmful_min_hits,
            "hard_harmful_margin": args.hard_harmful_margin,
            "conflict_keep_confidence_margin": args.conflict_keep_confidence_margin,
            "conflict_keep_harmful_margin": args.conflict_keep_harmful_margin,
            "conflict_demote_margin": args.conflict_demote_margin,
            "protect_teacher_low_evidence": args.protect_teacher_low_evidence,
        },
        "num_updated_rules": len(update_rows),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "updates": sorted(update_rows, key=lambda row: row["confidence"]),
    }
    status_meaning = dict(out.get("status_meaning", {}))
    status_meaning.update(
        {
            "confidence_active": "rule confidence is high enough to be enabled when rule_prior_allowed_statuses includes confidence_active",
            "confidence_observing": "rule is retained but should not be used online by default",
            "confidence_retired": "rule confidence is too low for online use",
        }
    )
    out["status_meaning"] = status_meaning
    out["online_enabled"] = any(rule.get("status") == "confidence_active" for rule in updated_rules)
    out["rules"] = updated_rules
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    update = payload["effect_confidence_update"]
    lines = [
        "# Rule Effect Confidence Update",
        "",
        f"Source library: `{payload['source_library']}`",
        f"Audit: `{update['audit']}`",
        f"Updated rules: `{update['num_updated_rules']}`",
        f"Status counts: `{update['status_counts']}`",
        f"Reason counts: `{update['reason_counts']}`",
        "",
        "## Updated Rules",
        "",
        "| status | confidence | reason | rule | hits | consistent | harmful |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in update["updates"]:
        lines.append(
            "| "
            f"`{row['confidence_status']}` | "
            f"{row['confidence']:.3f} | "
            f"`{row['reason']}` | "
            f"`{row['id']}` | "
            f"{int(row.get('hits', 0) or 0)} | "
            f"{float(row.get('direction_consistent_rate', 0.0) or 0.0):.3f} | "
            f"{float(row.get('direction_harmful_rate', 0.0) or 0.0):.3f} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-status", action="append", default=["v8_replay_safe_candidate"])
    parser.add_argument("--default-confidence", type=float, default=0.56)
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--min-effect-hits", type=int, default=20)
    parser.add_argument("--hit-ref", type=float, default=500.0)
    parser.add_argument("--positive-threshold", type=float, default=0.75)
    parser.add_argument("--harmful-threshold", type=float, default=0.45)
    parser.add_argument("--positive-lr", type=float, default=0.25)
    parser.add_argument("--harmful-lr", type=float, default=0.90)
    parser.add_argument("--absence-decay", type=float, default=0.98)
    parser.add_argument("--low-hit-decay", type=float, default=0.99)
    parser.add_argument("--context-floor", type=float, default=0.50)
    parser.add_argument("--positive-context-threshold", type=float, default=0.0)
    parser.add_argument("--hard-harmful-threshold", type=float, default=0.50)
    parser.add_argument("--hard-harmful-min-hits", type=int, default=100)
    parser.add_argument("--hard-harmful-margin", type=float, default=0.05)
    parser.add_argument("--conflict-keep-confidence-margin", type=float, default=0.15)
    parser.add_argument("--conflict-keep-harmful-margin", type=float, default=0.10)
    parser.add_argument("--conflict-demote-margin", type=float, default=0.05)
    parser.add_argument("--protect-teacher-conflicts", dest="protect_teacher_conflicts", action="store_true", default=True)
    parser.add_argument("--no-protect-teacher-conflicts", dest="protect_teacher_conflicts", action="store_false")
    parser.add_argument("--protect-teacher-low-evidence", dest="protect_teacher_low_evidence", action="store_true", default=True)
    parser.add_argument("--no-protect-teacher-low-evidence", dest="protect_teacher_low_evidence", action="store_false")
    parser.add_argument("--protected-teacher-origins", default="teacher_core_standard_hop")
    parser.add_argument("--teacher-conflict-min-challenger-hits", type=float, default=1000.0)
    parser.add_argument("--teacher-conflict-confidence-margin", type=float, default=0.25)
    parser.add_argument("--teacher-conflict-harmful-margin", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = apply_update(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    update = payload["effect_confidence_update"]
    print(f"wrote confidence library: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(f"updated={update['num_updated_rules']} status_counts={update['status_counts']}")


if __name__ == "__main__":
    main()
