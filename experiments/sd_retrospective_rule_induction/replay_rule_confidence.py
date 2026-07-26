#!/usr/bin/env python3
"""Replay closed-loop cycle rules into a confidence-weighted rule library."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import load_json, write_json


DEFAULT_ROOT = Path("results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506")
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json")


def action_parts(action_family: str) -> tuple[str, str]:
    if "_" not in action_family:
        return action_family, ""
    joint, sign = action_family.rsplit("_", 1)
    return joint, sign


def opposite_action(action_family: str) -> str | None:
    joint, sign = action_parts(action_family)
    if sign == "pos":
        return f"{joint}_neg"
    if sign == "neg":
        return f"{joint}_pos"
    return None


def conflict_group(rule: dict[str, Any]) -> str:
    joint, _ = action_parts(str(rule.get("action_family", "")))
    return f"{rule.get('substate_key', rule.get('state_event', 'unknown'))}|joint:{joint}"


def rule_id_from(rule: dict[str, Any]) -> str:
    return str(rule.get("id") or f"{rule.get('substate_key')}->prefer:{rule.get('action_family')}")


def eligible_evidence(rule: dict[str, Any], statuses: set[str], sources: set[str]) -> bool:
    if statuses and rule.get("status") not in statuses:
        return False
    if sources and rule.get("source") not in sources:
        return False
    if rule.get("intervention_type") != "soft_prefer_action_family":
        return False
    if not rule.get("action_family"):
        return False
    if not rule.get("substate_key"):
        return False
    return True


def evidence_strength(rule: dict[str, Any], support_ref: float, advantage_ref: float) -> float:
    support = max(float(rule.get("support", 0) or 0), 0.0)
    advantage = max(float(rule.get("mean_advantage", 0.0) or 0.0), 0.0)
    good_share = min(max(float(rule.get("good_share", 0.0) or 0.0), 0.0), 1.0)
    support_term = min(1.0, math.log1p(support) / math.log1p(max(support_ref, 1.0)))
    advantage_term = min(1.0, advantage / max(advantage_ref, 1e-6))
    return support_term * advantage_term * (1.0 - good_share)


def cycle_library_path(root: Path, cycle: int) -> Path:
    return root / f"cycle_{cycle:03d}" / "rules" / "library_v5.json"


def confidence_status(confidence: float, active_threshold: float, retired_threshold: float) -> str:
    if confidence >= active_threshold:
        return "confidence_active"
    if confidence <= retired_threshold:
        return "confidence_retired"
    return "confidence_observing"


def replay_confidence(
    root: Path,
    start_cycle: int,
    end_cycle: int,
    statuses: set[str],
    sources: set[str],
    positive_lr: float,
    conflict_lr: float,
    absence_decay: float,
    support_ref: float,
    advantage_ref: float,
    active_threshold: float,
    retired_threshold: float,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    group_to_rules: dict[str, set[str]] = defaultdict(set)
    cycle_summaries: list[dict[str, Any]] = []

    for cycle in range(start_cycle, end_cycle + 1):
        path = cycle_library_path(root, cycle)
        if not path.exists():
            cycle_summaries.append({"cycle": cycle, "path": str(path), "missing": True})
            continue
        payload = load_json(path)
        evidence_rules = [
            r for r in payload.get("rules", [])
            if eligible_evidence(r, statuses, sources)
        ]

        touched: set[str] = set()
        positive_ids: list[str] = []
        weakened_ids: list[str] = []
        strength_by_id: dict[str, float] = {}
        for rule in evidence_rules:
            rid = rule_id_from(rule)
            group = conflict_group(rule)
            action = str(rule.get("action_family", ""))
            strength = evidence_strength(rule, support_ref, advantage_ref)
            delta = positive_lr * strength
            strength_by_id[rid] = strength
            if rid not in records:
                records[rid] = {
                    "id": rid,
                    "substate_key": rule.get("substate_key"),
                    "state_event": rule.get("state_event"),
                    "action_family": action,
                    "intervention_type": rule.get("intervention_type"),
                    "source": rule.get("source"),
                    "confidence": 0.0,
                    "positive_evidence_cycles": [],
                    "negative_evidence_cycles": [],
                    "last_seen_cycle": None,
                    "first_seen_cycle": cycle,
                    "support_total": 0,
                    "weighted_advantage_sum": 0.0,
                    "evidence_history": [],
                    "conflict_group": group,
                    "opposite_action_family": opposite_action(action),
                }
            rec = records[rid]
            old_conf = float(rec.get("confidence", 0.0))
            new_conf = min(1.0, old_conf + delta)
            rec["confidence"] = new_conf
            rec["last_seen_cycle"] = cycle
            rec["positive_evidence_cycles"] = sorted(set(list(rec.get("positive_evidence_cycles", [])) + [cycle]))
            support = int(rule.get("support", 0) or 0)
            rec["support_total"] = int(rec.get("support_total", 0) or 0) + support
            rec["weighted_advantage_sum"] = float(rec.get("weighted_advantage_sum", 0.0) or 0.0) + support * float(rule.get("mean_advantage", 0.0) or 0.0)
            rec["mean_advantage_cumulative"] = rec["weighted_advantage_sum"] / rec["support_total"] if rec["support_total"] else 0.0
            rec["latest_rule_status"] = rule.get("status")
            rec["latest_good_share"] = float(rule.get("good_share", 0.0) or 0.0)
            rec["latest_support"] = support
            rec["latest_mean_advantage"] = float(rule.get("mean_advantage", 0.0) or 0.0)
            rec["evidence_history"].append(
                {
                    "cycle": cycle,
                    "type": "positive",
                    "strength": strength,
                    "delta": delta,
                    "support": support,
                    "mean_advantage": rec["latest_mean_advantage"],
                    "good_share": rec["latest_good_share"],
                    "status": rule.get("status"),
                }
            )
            group_to_rules[group].add(rid)
            touched.add(rid)
            positive_ids.append(rid)

            opposite = opposite_action(action)
            if opposite is None:
                continue
            for other_id in list(group_to_rules[group]):
                if other_id == rid:
                    continue
                other = records[other_id]
                if other.get("action_family") != opposite:
                    continue
                neg_delta = conflict_lr * delta
                old_other = float(other.get("confidence", 0.0))
                other["confidence"] = max(0.0, old_other - neg_delta)
                other["negative_evidence_cycles"] = sorted(set(list(other.get("negative_evidence_cycles", [])) + [cycle]))
                other["evidence_history"].append(
                    {
                        "cycle": cycle,
                        "type": "conflict",
                        "opposing_rule_id": rid,
                        "strength": strength,
                        "delta": -neg_delta,
                        "confidence_before": old_other,
                        "confidence_after": other["confidence"],
                    }
                )
                weakened_ids.append(other_id)

        for rid, rec in records.items():
            if rid in touched:
                continue
            old_conf = float(rec.get("confidence", 0.0))
            new_conf = old_conf * absence_decay
            if new_conf != old_conf:
                rec["confidence"] = new_conf
                rec["evidence_history"].append(
                    {
                        "cycle": cycle,
                        "type": "absence_decay",
                        "delta": new_conf - old_conf,
                        "confidence_before": old_conf,
                        "confidence_after": new_conf,
                    }
                )

        cycle_summaries.append(
            {
                "cycle": cycle,
                "path": str(path),
                "missing": False,
                "num_evidence_rules": len(evidence_rules),
                "positive_rule_ids": positive_ids,
                "weakened_rule_ids": sorted(set(weakened_ids)),
                "mean_strength": sum(strength_by_id.values()) / len(strength_by_id) if strength_by_id else 0.0,
            }
        )

    rules = []
    for rec in records.values():
        out = dict(rec)
        out["confidence"] = round(float(out.get("confidence", 0.0)), 6)
        out["confidence_status"] = confidence_status(out["confidence"], active_threshold, retired_threshold)
        out["status"] = out["confidence_status"]
        out["effective_prior_scale"] = out["confidence"]
        rules.append(out)
    rules.sort(key=lambda r: (r["confidence"], len(r.get("positive_evidence_cycles", [])), r.get("support_total", 0)), reverse=True)

    group_summary: list[dict[str, Any]] = []
    for group, ids in sorted(group_to_rules.items()):
        group_rules = [records[i] for i in ids]
        action_counts = Counter(r.get("action_family") for r in group_rules)
        group_summary.append(
            {
                "conflict_group": group,
                "num_rules": len(group_rules),
                "actions": dict(action_counts),
                "rules": [
                    {
                        "id": r["id"],
                        "action_family": r.get("action_family"),
                        "confidence": round(float(r.get("confidence", 0.0)), 6),
                        "positive_cycles": r.get("positive_evidence_cycles", []),
                        "negative_cycles": r.get("negative_evidence_cycles", []),
                    }
                    for r in sorted(group_rules, key=lambda x: x.get("confidence", 0.0), reverse=True)
                ],
            }
        )

    return {
        "library_version": "confidence_replay_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "cycle_range": [start_cycle, end_cycle],
        "parameters": {
            "statuses": sorted(statuses),
            "sources": sorted(sources),
            "positive_lr": positive_lr,
            "conflict_lr": conflict_lr,
            "absence_decay": absence_decay,
            "support_ref": support_ref,
            "advantage_ref": advantage_ref,
            "active_threshold": active_threshold,
            "retired_threshold": retired_threshold,
        },
        "status_meaning": {
            "confidence_active": "confidence is high enough to influence online prior",
            "confidence_observing": "rule is retained as evidence but should have weak or zero online influence",
            "confidence_retired": "rule confidence decayed below the retirement threshold",
        },
        "num_rules": len(rules),
        "num_active": sum(1 for r in rules if r["confidence_status"] == "confidence_active"),
        "num_observing": sum(1 for r in rules if r["confidence_status"] == "confidence_observing"),
        "num_retired": sum(1 for r in rules if r["confidence_status"] == "confidence_retired"),
        "rules": rules,
        "conflict_groups": group_summary,
        "cycle_summaries": cycle_summaries,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Rule Confidence Replay",
        "",
        f"Source root: `{payload['source_root']}`",
        f"Cycles: `{payload['cycle_range'][0]}`-`{payload['cycle_range'][1]}`",
        f"Rules: `{payload['num_rules']}`",
        f"Active / observing / retired: `{payload['num_active']}` / `{payload['num_observing']}` / `{payload['num_retired']}`",
        "",
        "## Top Rules",
        "",
        "| status | confidence | rule | positive cycles | negative cycles | support | mean advantage |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for rule in payload["rules"][:20]:
        lines.append(
            "| "
            f"`{rule['confidence_status']}` | "
            f"{rule['confidence']:.3f} | "
            f"`{rule['id']}` | "
            f"`{rule.get('positive_evidence_cycles', [])}` | "
            f"`{rule.get('negative_evidence_cycles', [])}` | "
            f"{rule.get('support_total', 0)} | "
            f"{float(rule.get('mean_advantage_cumulative', 0.0) or 0.0):.3f} |"
        )
    lines += ["", "## Conflict Groups", ""]
    for group in payload["conflict_groups"]:
        lines += [f"### `{group['conflict_group']}`", ""]
        lines.append("| confidence | action | rule | positive cycles | negative cycles |")
        lines.append("|---:|---|---|---|---|")
        for rule in group["rules"]:
            lines.append(
                "| "
                f"{rule['confidence']:.3f} | "
                f"`{rule['action_family']}` | "
                f"`{rule['id']}` | "
                f"`{rule['positive_cycles']}` | "
                f"`{rule['negative_cycles']}` |"
            )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-cycle", type=int, default=1)
    parser.add_argument("--end-cycle", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", action="append", default=["cycle_candidate", "cycle_supported", "replay_validated", "rolling_validated"])
    parser.add_argument("--source", action="append", default=["v4_substate"])
    parser.add_argument("--positive-lr", type=float, default=0.35)
    parser.add_argument("--conflict-lr", type=float, default=0.85)
    parser.add_argument("--absence-decay", type=float, default=0.98)
    parser.add_argument("--support-ref", type=float, default=8.0)
    parser.add_argument("--advantage-ref", type=float, default=0.20)
    parser.add_argument("--active-threshold", type=float, default=0.35)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = replay_confidence(
        args.root,
        args.start_cycle,
        args.end_cycle,
        set(args.status),
        set(args.source),
        args.positive_lr,
        args.conflict_lr,
        args.absence_decay,
        args.support_ref,
        args.advantage_ref,
        args.active_threshold,
        args.retired_threshold,
    )
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote confidence replay: {args.output}")
    print(f"rules: {payload['num_rules']} active={payload['num_active']} observing={payload['num_observing']} retired={payload['num_retired']}")


if __name__ == "__main__":
    main()
