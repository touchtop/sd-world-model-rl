#!/usr/bin/env python3
"""Build a conservative same-context action-family consolidated shadow library.

The script uses three generic signals to arbitrate rules that prescribe
opposing action families in the same rule context:

1. SD rollout + TVA terminal-value competition within the context.
2. Real eval trajectory rule-effect direction evidence.
3. Optional real short-window consequence evidence for shadow rule changes.

It writes a shadow/eval-only rule library. The source confidence library is not
modified. The research intent is to test whether rule value can be consolidated
from replay, SD and TVA evidence without adding Hopper-specific posture goals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import log1p
from pathlib import Path
from typing import Any

from apply_rule_effect_confidence import clipped, confidence_status, rule_id
from audit_tva_context_action_competition import context_key, family_joint
from build_tva_context_competition_shadow_library import flatten_audit
from build_tva_context_role_gated_shadow_library import classify_role, load_effect_rows
from common import load_json, write_json


def action_sign(family: str | None) -> str:
    if not family or "_" not in family:
        return ""
    return str(family).rsplit("_", 1)[-1]


def load_consequence_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    return {str(row.get("id")): row for row in payload.get("rules", []) if row.get("id")}


def effect_score(effect: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if effect is None:
        return 0.0
    hits = int(effect.get("selected_gate_hits", 0) or 0)
    if hits <= 0:
        return 0.0
    consistent = clipped(float(effect.get("direction_consistent_rate", 0.0) or 0.0))
    harmful = clipped(float(effect.get("direction_harmful_rate", 0.0) or 0.0))
    hit_weight = clipped(log1p(hits) / log1p(max(args.effect_hit_ref, 1)))
    return hit_weight * (consistent - harmful)


def tva_score(row: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if row is None:
        return 0.0
    best = clipped(float(row.get("best_rate", 0.0) or 0.0))
    top2 = clipped(float(row.get("top2_rate", 0.0) or 0.0))
    margin = float(row.get("mean_opposite_margin", 0.0) or 0.0)
    margin_score = max(-1.0, min(1.0, margin / max(args.opposite_margin_scale, 1e-6)))
    score = max(best, 0.5 * top2, margin_score)
    if bool(row.get("tva_supported", False)):
        score += args.tva_supported_bonus
    if bool(row.get("tva_suspect", False)):
        score -= args.tva_suspect_penalty
    return max(-1.0, min(1.0, score))


def consequence_score(row: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if row is None:
        return 0.0
    margin = row.get("outcome_margin")
    if margin is None:
        return 0.0
    score = max(-1.0, min(1.0, float(margin) / max(args.consequence_margin_scale, 1e-6)))
    if bool(row.get("consequence_supported", False)):
        score += args.consequence_supported_bonus
    return max(-1.0, min(1.0, score))


def combined_score(
    *,
    tva_row: dict[str, Any] | None,
    effect: dict[str, Any] | None,
    consequence: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    tv = tva_score(tva_row, args)
    ef = effect_score(effect, args)
    co = consequence_score(consequence, args)
    total = args.tva_weight * tv + args.effect_weight * ef + args.consequence_weight * co
    return {"total": total, "tva": tv, "effect": ef, "consequence": co}


def has_opposing_signs(rows: list[dict[str, Any]]) -> bool:
    signs = {action_sign(row.get("action_family")) for row in rows}
    return {"pos", "neg"} <= signs


def candidate_group_key(rule: dict[str, Any], tva_row: dict[str, Any] | None) -> tuple[str, str]:
    key = str((tva_row or {}).get("context_key") or context_key(rule))
    joint = family_joint(str(rule.get("action_family", "")))
    return key, joint


def should_protect_loser(role: str, effect: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if role != "recovery_consistent":
        return False
    if effect is None:
        return True
    harmful = clipped(float(effect.get("direction_harmful_rate", 0.0) or 0.0))
    return harmful < args.protected_recovery_harmful_min


def build_shadow(args: argparse.Namespace) -> dict[str, Any]:
    library = load_json(args.library)
    tva_audit = load_json(args.tva_audit)
    tva_by_id, _conflict_ids = flatten_audit(tva_audit)
    effect_by_id = load_effect_rows(args.effect_audit)
    consequence_by_id = load_consequence_rows(args.consequence_audit)
    target_statuses = set(args.target_status)

    rule_by_id = {rule_id(rule): rule for rule in library.get("rules", [])}
    scored: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for rule in library.get("rules", []):
        if target_statuses and rule.get("status") not in target_statuses:
            continue
        rid = rule_id(rule)
        family = str(rule.get("action_family", ""))
        joint = family_joint(family)
        if not joint or joint == "zero":
            continue
        tva_row = tva_by_id.get(rid)
        effect = effect_by_id.get(rid)
        consequence = consequence_by_id.get(rid)
        role = classify_role(effect, args)
        scores = combined_score(tva_row=tva_row, effect=effect, consequence=consequence, args=args)
        item = {
            "id": rid,
            "context_key": candidate_group_key(rule, tva_row)[0],
            "joint": joint,
            "action_family": family,
            "confidence": float(rule.get("confidence", 0.0) or 0.0),
            "status": rule.get("status"),
            "role": role,
            "scores": scores,
            "effect_selected_gate_hits": int((effect or {}).get("selected_gate_hits", 0) or 0),
            "effect_direction_consistent_rate": (effect or {}).get("direction_consistent_rate"),
            "effect_direction_harmful_rate": (effect or {}).get("direction_harmful_rate"),
            "tva_supported": bool((tva_row or {}).get("tva_supported", False)),
            "tva_suspect": bool((tva_row or {}).get("tva_suspect", False)),
            "tva_context_records": int((tva_row or {}).get("context_records", 0) or 0),
            "consequence_supported": bool((consequence or {}).get("consequence_supported", False)),
            "consequence_margin": (consequence or {}).get("outcome_margin"),
        }
        scored[rid] = item
        grouped[(item["context_key"], joint)].append(item)

    decisions: dict[str, dict[str, Any]] = {}
    group_rows: list[dict[str, Any]] = []
    for (ctx, joint), rows in sorted(grouped.items()):
        if len(rows) < 2 or not has_opposing_signs(rows):
            continue
        ordered = sorted(rows, key=lambda row: row["scores"]["total"], reverse=True)
        winner = ordered[0]
        second = ordered[1]
        gap = float(winner["scores"]["total"] - second["scores"]["total"])
        winner_hits = int(winner.get("effect_selected_gate_hits", 0) or 0)
        winner_records = int(winner.get("tva_context_records", 0) or 0)
        eligible = (
            gap >= args.min_score_gap
            and winner_hits >= args.min_winner_effect_hits
            and winner_records >= args.min_tva_context_records
        )
        group_decisions: list[dict[str, Any]] = []
        for row in ordered:
            rid = row["id"]
            decision = "winner_kept" if rid == winner["id"] else "loser_kept"
            reason = "context_not_consolidated"
            if eligible and rid != winner["id"]:
                protected = should_protect_loser(str(row.get("role")), effect_by_id.get(rid), args)
                if protected:
                    decision = "loser_protected_recovery"
                    reason = "recovery_consistent_protected"
                else:
                    decision = "loser_demoted"
                    reason = "same_context_lower_value_family"
            elif eligible and rid == winner["id"]:
                reason = "same_context_winner"
            decisions[rid] = {
                **row,
                "decision": decision,
                "decision_reason": reason,
                "winner_id": winner["id"],
                "winner_action_family": winner["action_family"],
                "score_gap_to_winner": float(winner["scores"]["total"] - row["scores"]["total"]),
                "group_score_gap": gap,
                "group_eligible": bool(eligible),
            }
            group_decisions.append(decisions[rid])
        group_rows.append(
            {
                "context_key": ctx,
                "joint": joint,
                "eligible": bool(eligible),
                "winner_id": winner["id"],
                "winner_action_family": winner["action_family"],
                "score_gap": gap,
                "winner_effect_hits": winner_hits,
                "winner_tva_context_records": winner_records,
                "rules": group_decisions,
            }
        )

    updated_rules: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for rule in library.get("rules", []):
        rid = rule_id(rule)
        decision = decisions.get(rid)
        out = dict(rule)
        before = clipped(float(out.get("confidence", args.default_confidence) or 0.0))
        after = before
        reason = "not_in_context_consolidation_group"
        if decision is not None:
            reason = str(decision["decision_reason"])
            if decision["decision"] == "loser_demoted":
                after = clipped(before * args.loser_decay)
                if args.demote_loser_below_active:
                    after = min(after, args.active_threshold - args.demote_margin)
            elif decision["decision"] == "winner_kept" and args.strengthen_winner:
                strength = clipped(float(decision["scores"]["total"]))
                if strength > 0.0:
                    after = clipped(before + args.winner_lr * strength * (1.0 - before))
        status = confidence_status(after, args.active_threshold, args.retired_threshold)
        if decision is not None:
            out["confidence_before_context_consolidation_shadow"] = round(before, 6)
            out["confidence"] = round(after, 6)
            out["confidence_status"] = status
            out["status"] = status
            out["effective_prior_scale"] = round(after, 6)
            out["last_context_consolidation_shadow_update_utc"] = now
            out["last_context_consolidation_shadow_reason"] = reason
            history = list(out.get("context_consolidation_shadow_history", []))
            history.append(
                {
                    "updated_at_utc": now,
                    "source_library": str(args.library),
                    "source_tva_audit": str(args.tva_audit),
                    "source_effect_audit": str(args.effect_audit),
                    "source_consequence_audit": str(args.consequence_audit) if args.consequence_audit else None,
                    "confidence_before": round(before, 6),
                    "confidence_after": round(after, 6),
                    **decision,
                }
            )
            out["context_consolidation_shadow_history"] = history
            update_rows.append(
                {
                    **decision,
                    "id": rid,
                    "previous_confidence": before,
                    "confidence": after,
                    "confidence_status": status,
                }
            )
        updated_rules.append(out)

    status_counts = Counter(str(rule.get("status")) for rule in updated_rules)
    decision_counts = Counter(str(row.get("decision")) for row in update_rows)
    changed_rows = [
        row for row in update_rows if abs(float(row["confidence"]) - float(row["previous_confidence"])) > 1e-9
    ]

    payload = dict(library)
    payload["rules"] = updated_rules
    payload["library_version"] = f"{library.get('library_version', 'rule_library')}_context_consolidated_shadow"
    payload["created_at_utc"] = now
    payload["source_library"] = str(args.library)
    parameters = {
        **vars(args),
        "library": str(args.library),
        "tva_audit": str(args.tva_audit),
        "effect_audit": str(args.effect_audit),
        "consequence_audit": str(args.consequence_audit) if args.consequence_audit else None,
        "output": str(args.output),
    }
    payload["context_consolidation_shadow_update"] = {
        "tva_audit": str(args.tva_audit),
        "effect_audit": str(args.effect_audit),
        "consequence_audit": str(args.consequence_audit) if args.consequence_audit else None,
        "parameters": parameters,
        "num_candidate_groups": len(group_rows),
        "num_eligible_groups": sum(1 for row in group_rows if row["eligible"]),
        "num_rules_in_groups": len(update_rows),
        "num_confidence_changed": len(changed_rows),
        "status_counts": dict(status_counts),
        "decision_counts": dict(decision_counts),
        "groups": group_rows,
        "updates": sorted(update_rows, key=lambda row: (row["decision"], -row["group_score_gap"], row["id"])),
    }
    payload["online_enabled"] = any(rule.get("status") == "confidence_active" for rule in updated_rules)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    update = payload["context_consolidation_shadow_update"]
    lines = [
        "# TVA Context Consolidated Shadow Library",
        "",
        f"Source library: `{payload['source_library']}`",
        f"TVA audit: `{update['tva_audit']}`",
        f"Effect audit: `{update['effect_audit']}`",
        f"Consequence audit: `{update['consequence_audit']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| candidate groups | {update['num_candidate_groups']} |",
        f"| eligible groups | {update['num_eligible_groups']} |",
        f"| rules in groups | {update['num_rules_in_groups']} |",
        f"| confidence changed | {update['num_confidence_changed']} |",
        f"| status counts | `{update['status_counts']}` |",
        f"| decision counts | `{update['decision_counts']}` |",
        "",
        "This is a shadow artifact. It does not modify the live confidence library.",
        "",
        "## Eligible Groups",
        "",
        "| context | joint | winner | gap | hits | records |",
        "|---|---|---|---:|---:|---:|",
    ]
    for group in update["groups"]:
        if not group["eligible"]:
            continue
        lines.append(
            "| "
            f"`{group['context_key']}` | "
            f"`{group['joint']}` | "
            f"`{group['winner_action_family']}` | "
            f"{group['score_gap']:.3f} | "
            f"{group['winner_effect_hits']} | "
            f"{group['winner_tva_context_records']} |"
        )
    lines.extend(
        [
            "",
            "## Rule Decisions",
            "",
            "| decision | confidence | delta | score | tva | effect | consequence | role | rule |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in update["updates"]:
        before = float(row.get("previous_confidence", 0.0) or 0.0)
        after = float(row.get("confidence", 0.0) or 0.0)
        if row["decision"] == "loser_kept" and abs(after - before) <= 1e-9:
            continue
        scores = row.get("scores", {})
        lines.append(
            "| "
            f"`{row['decision']}` | "
            f"{after:.3f} | "
            f"{after - before:+.3f} | "
            f"{float(scores.get('total', 0.0)):.3f} | "
            f"{float(scores.get('tva', 0.0)):.3f} | "
            f"{float(scores.get('effect', 0.0)):.3f} | "
            f"{float(scores.get('consequence', 0.0)):.3f} | "
            f"`{row.get('role')}` | "
            f"`{row['id']}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tva-audit", type=Path, required=True)
    parser.add_argument("--effect-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consequence-audit", type=Path)
    parser.add_argument("--target-status", action="append", default=["confidence_active", "confidence_observing"])
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--default-confidence", type=float, default=0.56)
    parser.add_argument("--tva-weight", type=float, default=0.45)
    parser.add_argument("--effect-weight", type=float, default=0.35)
    parser.add_argument("--consequence-weight", type=float, default=0.20)
    parser.add_argument("--opposite-margin-scale", type=float, default=2.0)
    parser.add_argument("--consequence-margin-scale", type=float, default=0.10)
    parser.add_argument("--tva-supported-bonus", type=float, default=0.10)
    parser.add_argument("--tva-suspect-penalty", type=float, default=0.20)
    parser.add_argument("--consequence-supported-bonus", type=float, default=0.10)
    parser.add_argument("--effect-hit-ref", type=int, default=500)
    parser.add_argument("--min-score-gap", type=float, default=0.12)
    parser.add_argument("--min-winner-effect-hits", type=int, default=50)
    parser.add_argument("--min-tva-context-records", type=int, default=20)
    parser.add_argument("--loser-decay", type=float, default=0.50)
    parser.add_argument("--demote-margin", type=float, default=0.05)
    parser.add_argument("--demote-loser-below-active", action="store_true", default=True)
    parser.add_argument("--no-demote-loser-below-active", dest="demote_loser_below_active", action="store_false")
    parser.add_argument("--strengthen-winner", action="store_true", default=False)
    parser.add_argument("--winner-lr", type=float, default=0.05)
    parser.add_argument("--min-effect-hits", type=int, default=20)
    parser.add_argument("--recovery-consistent-min", type=float, default=0.80)
    parser.add_argument("--recovery-context-max", type=float, default=0.05)
    parser.add_argument("--progress-context-min", type=float, default=0.20)
    parser.add_argument("--effect-harmful-min", type=float, default=0.45)
    parser.add_argument("--protected-recovery-harmful-min", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_shadow(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    update = payload["context_consolidation_shadow_update"]
    print(f"wrote context-consolidated shadow library: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(
        "groups={groups} eligible={eligible} changed={changed} decisions={decisions}".format(
            groups=update["num_candidate_groups"],
            eligible=update["num_eligible_groups"],
            changed=update["num_confidence_changed"],
            decisions=update["decision_counts"],
        )
    )


if __name__ == "__main__":
    main()
