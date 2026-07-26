#!/usr/bin/env python3
"""Merge newly validated cycle rules into a versioned rule library."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_RULE_LIBRARY_V5, load_json, write_json


CONFIDENCE_FIELDS = (
    "confidence",
    "confidence_status",
    "effective_prior_scale",
    "positive_evidence_cycles",
    "negative_evidence_cycles",
    "last_seen_cycle",
    "first_seen_cycle",
    "support_total",
    "weighted_advantage_sum",
    "mean_advantage_cumulative",
    "evidence_history",
    "conflict_group",
    "opposite_action_family",
)

PROVENANCE_FIELDS = (
    "rule_origin",
    "teacher_source",
    "teacher_examples",
    "teacher_episode_support",
    "mean_teacher_episode_reward",
)

VALID_ACTION_FAMILIES = {
    "waist_pos",
    "waist_neg",
    "hip_pos",
    "hip_neg",
    "knee_pos",
    "knee_neg",
    "ankle_pos",
    "ankle_neg",
}


def rule_score(rule: dict[str, Any]) -> tuple[float, int, float]:
    mean_advantage = float(
        rule.get(
            "mean_advantage",
            rule.get("mean_advantage_over_actual", rule.get("heldout_strength", 0.0)),
        )
        or 0.0
    )
    support = int(rule.get("support", rule.get("replay_hits", 0)) or 0)
    substate_coverage = float(
        rule.get(
            "substate_coverage",
            rule.get("event_coverage", rule.get("replay_good_hit_rate", 0.0)),
        )
        or 0.0
    )
    return mean_advantage, support, substate_coverage


def rule_intervention_supported(rule: dict[str, Any]) -> bool:
    intervention = rule.get("intervention_type")
    if intervention == "soft_prefer_action_family":
        return True
    if rule.get("source") == "v8_transition_replay_safety" and intervention == "avoid_action_family":
        return True
    return False


def valid_action_family(rule: dict[str, Any]) -> bool:
    intervention = rule.get("intervention_type")
    if intervention not in {"soft_prefer_action_family", "avoid_action_family"}:
        return True
    return str(rule.get("action_family", "")) in VALID_ACTION_FAMILIES


def eligible_rule(rule: dict[str, Any], required_status: str, allow_sources: set[str]) -> bool:
    if required_status and rule.get("status") != required_status:
        return False
    if allow_sources and rule.get("source") not in allow_sources:
        return False
    if not rule_intervention_supported(rule):
        return False
    if not valid_action_family(rule):
        return False
    return True


def evidence_candidate_rule(rule: dict[str, Any], allow_sources: set[str], statuses: set[str]) -> bool:
    if statuses and rule.get("status") not in statuses:
        return False
    if allow_sources and rule.get("source") not in allow_sources:
        return False
    if not rule_intervention_supported(rule):
        return False
    if not valid_action_family(rule):
        return False
    return True


def annotate_rule(rule: dict[str, Any], cycle: int, cycle_library: Path) -> dict[str, Any]:
    out = dict(rule)
    out["first_closed_loop_cycle"] = cycle
    out["latest_closed_loop_cycle"] = cycle
    out["closed_loop_source_library"] = str(cycle_library)
    notes = list(out.get("audit_notes", []))
    notes.append("闭环周期规则更新器选择为本周期最显著上线规则。")
    out["audit_notes"] = notes
    return out


def annotate_evidence_rule(rule: dict[str, Any], cycle: int, cycle_library: Path) -> dict[str, Any]:
    out = dict(rule)
    out.setdefault("first_closed_loop_cycle", cycle)
    out["latest_closed_loop_cycle"] = cycle
    out["closed_loop_source_library"] = str(cycle_library)
    notes = list(out.get("audit_notes", []))
    notes.append("闭环周期规则更新器保留为周期候选证据；该状态不参与在线动作先验。")
    out["audit_notes"] = notes
    return out


def merge_evidence(existing: dict[str, Any], incoming: dict[str, Any], cycle: int) -> dict[str, Any]:
    out = dict(incoming)
    prev_cycles = list(existing.get("evidence_cycles", []))
    if not prev_cycles and existing.get("first_closed_loop_cycle") is not None:
        prev_cycles = [existing.get("first_closed_loop_cycle")]
    evidence_cycles = sorted({int(c) for c in prev_cycles + [cycle] if c is not None})
    prev_support = int(existing.get("cumulative_support", existing.get("support", 0)) or 0)
    cur_support = int(incoming.get("support", 0) or 0)
    prev_weighted_adv = float(existing.get("cumulative_mean_advantage", existing.get("mean_advantage", 0.0)) or 0.0) * max(prev_support, 0)
    cur_adv = float(incoming.get("mean_advantage", 0.0) or 0.0)
    cumulative_support = prev_support + cur_support
    out["first_closed_loop_cycle"] = min(evidence_cycles) if evidence_cycles else cycle
    out["latest_closed_loop_cycle"] = cycle
    out["evidence_cycles"] = evidence_cycles
    out["cycles_covered"] = len(evidence_cycles)
    out["cumulative_support"] = cumulative_support
    out["cumulative_mean_advantage"] = (prev_weighted_adv + cur_adv * cur_support) / cumulative_support if cumulative_support else 0.0
    return out


def maybe_promote_rolling(rule: dict[str, Any], min_cycles: int, min_support: int, min_advantage: float, max_good_share: float) -> dict[str, Any]:
    out = dict(rule)
    cycles = int(out.get("cycles_covered", 1) or 1)
    support = int(out.get("cumulative_support", out.get("support", 0)) or 0)
    mean_adv = float(out.get("cumulative_mean_advantage", out.get("mean_advantage", 0.0)) or 0.0)
    good_share = float(out.get("good_share", 0.0) or 0.0)
    if cycles >= min_cycles and support >= min_support and mean_adv >= min_advantage and good_share <= max_good_share:
        out["status"] = "rolling_validated"
        notes = list(out.get("audit_notes", []))
        notes.append("跨周期累计证据达到 rolling_validated 阈值；仍需进一步回放或跨 seed 审查。")
        out["audit_notes"] = notes
    return out


def rolling_promoted_status(required_status: str) -> str:
    """Choose the online status produced by repeated closed-loop evidence.

    Legacy runs used `rolling_validated`. v8 transition pilots deliberately
    gate online use through `v8_replay_safe_candidate`, so repeated evidence
    should promote into that same online-readable status.
    """
    if required_status.startswith("v8_"):
        return required_status
    return "rolling_validated"


def maybe_promote_rolling_to_status(
    rule: dict[str, Any],
    min_cycles: int,
    min_support: int,
    min_advantage: float,
    max_good_share: float,
    weaken_max_good_share: float,
    promoted_status: str,
) -> dict[str, Any]:
    cycles = int(rule.get("cycles_covered", 1) or 1)
    support = int(rule.get("cumulative_support", rule.get("support", 0)) or 0)
    mean_adv = float(rule.get("cumulative_mean_advantage", rule.get("mean_advantage", 0.0)) or 0.0)
    good_share = float(rule.get("good_share", 0.0) or 0.0)
    if cycles >= min_cycles and support >= min_support and mean_adv >= min_advantage and good_share > weaken_max_good_share:
        out = dict(rule)
        out["status"] = "cycle_weakened"
        notes = list(out.get("audit_notes", []))
        notes.append("跨周期累计证据达到基本阈值，但好轨迹占比超过观察层削弱阈值；削弱为非在线候选并退出滚动晋升池。")
        out["audit_notes"] = notes
        return out
    out = maybe_promote_rolling(rule, min_cycles, min_support, min_advantage, max_good_share)
    if out.get("status") == "rolling_validated" and promoted_status != "rolling_validated":
        out["status"] = promoted_status
        notes = list(out.get("audit_notes", []))
        notes.append(f"跨周期累计证据达到阈值；按本轮在线门控晋升为 {promoted_status}。")
        out["audit_notes"] = notes
    return out


def preserve_confidence_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Keep confidence replay state when legacy cycle evidence updates the same rule.

    Closed-loop summaries still emit legacy v5 statuses such as `cycle_candidate`
    and `rolling_validated`. During confidence-prior continuation those records
    are evidence, but they should not silently disable an already active
    confidence rule before the next offline confidence replay is run.
    """
    if "confidence" not in existing and "confidence_status" not in existing:
        return incoming
    if "confidence" in incoming or "confidence_status" in incoming:
        return incoming

    out = dict(incoming)
    out["latest_cycle_rule_status"] = incoming.get("status")
    out["latest_cycle_rule_support"] = incoming.get("support")
    out["latest_cycle_rule_mean_advantage"] = incoming.get("mean_advantage")
    existing_status = str(existing.get("status", ""))
    if existing_status.startswith("confidence_"):
        out["status"] = existing_status
    for field in CONFIDENCE_FIELDS:
        if field in existing:
            out[field] = existing[field]
    for field in PROVENANCE_FIELDS:
        if field in existing and not out.get(field):
            out[field] = existing[field]
    notes = list(out.get("audit_notes", []))
    notes.append("保留既有 confidence replay 元数据；本周期规则仅作为后续信心重放证据。")
    out["audit_notes"] = notes
    return out


def upsert_rule(rules: list[dict[str, Any]], rule: dict[str, Any]) -> bool:
    rule_id = rule.get("id")
    for idx, existing in enumerate(rules):
        if existing.get("id") == rule_id:
            rules[idx] = preserve_confidence_metadata(existing, rule)
            return True
    rules.append(rule)
    return False


def merge_one_rule(
    base_library: dict[str, Any],
    cycle_library: dict[str, Any],
    cycle: int,
    cycle_library_path: Path,
    required_status: str,
    allow_sources: set[str],
    carry_candidate_statuses: set[str],
    max_online_rules_per_cycle: int,
    max_cycle_candidates: int,
    rolling_min_cycles: int,
    rolling_min_support: int,
    rolling_min_advantage: float,
    rolling_max_good_share: float,
    rolling_weaken_max_good_share: float,
) -> dict[str, Any]:
    promoted_status = rolling_promoted_status(required_status)
    candidates = [
        rule for rule in cycle_library.get("rules", [])
        if eligible_rule(rule, required_status, allow_sources)
    ]
    candidates.sort(key=rule_score, reverse=True)
    selected_rules = candidates[:max(0, max_online_rules_per_cycle)]

    evidence_candidates = [
        rule for rule in cycle_library.get("rules", [])
        if evidence_candidate_rule(rule, allow_sources, carry_candidate_statuses)
    ]
    rules = list(base_library.get("rules", []))
    active_evidence_statuses = carry_candidate_statuses | {"rolling_validated", required_status}
    existing_evidence_ids = {
        str(rule.get("id"))
        for rule in rules
        if rule.get("status") in active_evidence_statuses
    }
    evidence_candidates.sort(key=rule_score, reverse=True)
    repeated_candidates = [rule for rule in evidence_candidates if str(rule.get("id")) in existing_evidence_ids]
    new_candidates = [rule for rule in evidence_candidates if str(rule.get("id")) not in existing_evidence_ids]
    carried_candidates: list[dict[str, Any]] = []
    seen_carried: set[str] = set()
    for rule in repeated_candidates + new_candidates:
        rule_id = str(rule.get("id"))
        if rule_id in seen_carried:
            continue
        carried_candidates.append(rule)
        seen_carried.add(rule_id)
        if len(carried_candidates) >= max_cycle_candidates:
            break

    action = "no_rule_added"
    selected_ids: list[str] = []
    replaced_selected_count = 0
    added_selected_count = 0
    for selected in selected_rules:
        selected_out = annotate_rule(selected, cycle, cycle_library_path)
        selected_ids.append(str(selected_out["id"]))
        replaced_existing = upsert_rule(rules, selected_out)
        if replaced_existing:
            replaced_selected_count += 1
        else:
            added_selected_count += 1
    if selected_ids:
        action = "merged_online_rules"

    carried_ids: list[str] = []
    replaced_candidate_count = 0
    rolling_promoted_ids: list[str] = []
    for rule in carried_candidates:
        candidate_out = annotate_evidence_rule(rule, cycle, cycle_library_path)
        existing = next((r for r in rules if r.get("id") == candidate_out.get("id")), None)
        if existing is not None and existing.get("status") in carry_candidate_statuses | {"rolling_validated"}:
            candidate_out = merge_evidence(existing, candidate_out, cycle)
        else:
            candidate_out["evidence_cycles"] = [cycle]
            candidate_out["cycles_covered"] = 1
            candidate_out["cumulative_support"] = int(candidate_out.get("support", 0) or 0)
            candidate_out["cumulative_mean_advantage"] = float(candidate_out.get("mean_advantage", 0.0) or 0.0)
        candidate_out = maybe_promote_rolling_to_status(
            candidate_out,
            rolling_min_cycles,
            rolling_min_support,
            rolling_min_advantage,
            rolling_max_good_share,
            rolling_weaken_max_good_share,
            promoted_status,
        )
        if candidate_out.get("status") == promoted_status:
            rolling_promoted_ids.append(str(candidate_out.get("id")))
        carried_ids.append(str(candidate_out.get("id")))
        if upsert_rule(rules, candidate_out):
            replaced_candidate_count += 1

    has_online_rule = any(
        eligible_rule(rule, required_status, allow_sources)
        for rule in rules
    )

    status_meaning = dict(base_library.get("status_meaning", {}))
    status_meaning.setdefault("cycle_candidate", "single-cycle near-threshold evidence; not eligible for online use")
    status_meaning.setdefault("cycle_supported", "passed single-cycle substate mining thresholds; requires cumulative validation")
    status_meaning.setdefault("cycle_weakened", "repeated evidence failed safety or good-share checks; not eligible for online use")
    status_meaning.setdefault("rolling_validated", "validated by repeated within-run evidence; optional online use in closed-loop pilots")
    if required_status.startswith("v8_"):
        status_meaning.setdefault(required_status, "v8 replay-safe transition rule eligible for online low-strength prior")
    out = {
        "library_version": f"closed_loop_v7_cycle_{cycle:03d}",
        "online_enabled": has_online_rule,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_library": base_library.get("library_version"),
        "cycle": cycle,
        "status_meaning": status_meaning,
        "closed_loop_update": {
            "policy": "bounded_online_rules_per_cycle",
            "required_status": required_status,
            "allow_sources": sorted(allow_sources),
            "cycle_library": str(cycle_library_path),
            "num_eligible_candidates": len(candidates),
            "num_online_rules_after_update": sum(
                1 for rule in rules if eligible_rule(rule, required_status, allow_sources)
            ),
            "action": action,
            "max_online_rules_per_cycle": max_online_rules_per_cycle,
            "selected_rule_ids": selected_ids,
            "selected_rule_id": selected_ids[0] if selected_ids else None,
            "replaced_existing": replaced_selected_count > 0,
            "added_selected_rule_count": added_selected_count,
            "replaced_selected_rule_count": replaced_selected_count,
            "carry_candidate_statuses": sorted(carry_candidate_statuses),
            "num_cycle_evidence_candidates": len(evidence_candidates),
            "max_cycle_candidates_carried": max_cycle_candidates,
            "carried_cycle_candidate_ids": carried_ids,
            "replaced_cycle_candidate_count": replaced_candidate_count,
            "rolling_validation": {
                "min_cycles": rolling_min_cycles,
                "min_support": rolling_min_support,
                "min_advantage": rolling_min_advantage,
                "max_good_share": rolling_max_good_share,
                "weaken_max_good_share": rolling_weaken_max_good_share,
                "promoted_status": promoted_status,
                "promoted_rule_ids": rolling_promoted_ids,
            },
        },
        "rules": rules,
    }
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    update = payload["closed_loop_update"]
    lines = [
        f"# Closed-loop rule update cycle {payload['cycle']:03d}",
        "",
        f"- Policy: `{update['policy']}`",
        f"- Required status: `{update['required_status']}`",
        f"- Allowed sources: `{update['allow_sources']}`",
        f"- Eligible candidates: `{update['num_eligible_candidates']}`",
        f"- Online rules after update: `{update.get('num_online_rules_after_update', 0)}`",
        f"- Action: `{update['action']}`",
        f"- Max online rules per cycle: `{update.get('max_online_rules_per_cycle', 1)}`",
        f"- Selected rules: `{update.get('selected_rule_ids', [])}`",
        f"- Cycle evidence candidates: `{update.get('num_cycle_evidence_candidates', 0)}`",
        f"- Carried cycle candidates: `{len(update.get('carried_cycle_candidate_ids', []))}`",
        f"- Output rules: `{len(payload['rules'])}`",
        "",
    ]
    for rule_id in update.get("carried_cycle_candidate_ids", []):
        lines.append(f"- Carried evidence: `{rule_id}`")
    promoted = update.get("rolling_validation", {}).get("promoted_rule_ids", [])
    for rule_id in promoted:
        lines.append(f"- Rolling validated: `{rule_id}`")
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-library", type=Path, default=DEFAULT_RULE_LIBRARY_V5)
    parser.add_argument("--cycle-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--required-status", default="replay_validated")
    parser.add_argument("--allow-source", action="append", default=["v4_substate"])
    parser.add_argument("--carry-candidate-status", action="append", default=["cycle_candidate", "cycle_supported"])
    parser.add_argument("--max-online-rules-per-cycle", type=int, default=1)
    parser.add_argument("--max-cycle-candidates", type=int, default=3)
    parser.add_argument("--rolling-min-cycles", type=int, default=2)
    parser.add_argument("--rolling-min-support", type=int, default=6)
    parser.add_argument("--rolling-min-advantage", type=float, default=0.05)
    parser.add_argument("--rolling-max-good-share", type=float, default=0.25)
    parser.add_argument("--rolling-weaken-max-good-share", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = merge_one_rule(
        load_json(args.base_library),
        load_json(args.cycle_library),
        args.cycle,
        args.cycle_library,
        args.required_status,
        set(args.allow_source),
        set(args.carry_candidate_status),
        args.max_online_rules_per_cycle,
        args.max_cycle_candidates,
        args.rolling_min_cycles,
        args.rolling_min_support,
        args.rolling_min_advantage,
        args.rolling_max_good_share,
        args.rolling_weaken_max_good_share
        if args.rolling_weaken_max_good_share is not None
        else args.rolling_max_good_share,
    )
    write_json(args.output, payload)
    write_report(args.output, payload)
    update = payload["closed_loop_update"]
    print(f"wrote closed-loop library: {args.output}")
    print(f"action: {update['action']} selected={update['selected_rule_ids']}")


if __name__ == "__main__":
    main()
