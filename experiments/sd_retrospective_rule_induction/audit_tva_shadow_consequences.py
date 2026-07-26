#!/usr/bin/env python3
"""Audit short-window trajectory consequences for TVA shadow rule proposals.

This script is read-only. It checks whether rules strengthened or weakened by
TVA shadow update are aligned with observed short-window consequences in the
same trajectory data used for TVA counterfactual auditing. The result is not a
causal estimate; it is a consequence-level consistency check before any live
confidence update is considered.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import (  # noqa: E402
    as_float,
    continuation_score,
    row_parts,
    selected_action_relation,
    selected_gate_applies,
    stage_metric,
    state_matches,
    structural_success,
)
from apply_rule_effect_confidence import rule_id  # noqa: E402
from common import load_json, read_csv, write_json  # noqa: E402


DEFAULT_REPORT_DIR = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "tva_validation_300k_oldscheme_20260515"
)


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def relation_order() -> tuple[str, str, str]:
    return ("same", "opposite", "neutral")


def load_shadow_rules(path: Path, delta_epsilon: float) -> list[dict[str, Any]]:
    payload = load_json(path)
    rules = []
    for rule in payload.get("rules", []):
        delta = float(rule.get("shadow_confidence_delta", 0.0) or 0.0)
        if abs(delta) <= delta_epsilon:
            continue
        rules.append(rule)
    rules.sort(key=lambda item: abs(float(item.get("shadow_confidence_delta", 0.0) or 0.0)), reverse=True)
    return rules


def window_rows(rows: list[dict[str, str]], idx: int, horizon: int) -> tuple[dict[str, str], list[dict[str, str]]]:
    end = min(len(rows) - 1, idx + max(1, horizon))
    future = rows[end]
    window = rows[idx + 1 : end + 1] or [future]
    return future, window


def consequence_metrics(rows: list[dict[str, str]], idx: int, horizon: int) -> dict[str, float]:
    current = rows[idx]
    future, window = window_rows(rows, idx, horizon)
    current_continuation = continuation_score(current)
    future_continuation = continuation_score(future)
    min_support = min(stage_metric(row, "support_contact") for row in window)
    min_spin = min(stage_metric(row, "spin_control") for row in window)
    max_falling = max(stage_metric(row, "falling_air") for row in window)
    max_rotation = max(stage_metric(row, "rotation_penalty") for row in window)
    max_collision = max(stage_metric(row, "body_collision") for row in window)
    reward_sum = sum(as_float(row, "reward") for row in window)
    outcome_score = (
        0.45 * future_continuation
        + 0.25 * (future_continuation - current_continuation)
        + 0.15 * min_support
        + 0.15 * min_spin
        - 0.20 * max_collision
        - 0.10 * max_falling
    )
    return {
        "outcome_score": outcome_score,
        "current_continuation": current_continuation,
        "future_continuation": future_continuation,
        "delta_continuation": future_continuation - current_continuation,
        "terminal_structural_success": 1.0 if structural_success(future) else 0.0,
        "terminal_support_contact": stage_metric(future, "support_contact"),
        "terminal_full_foot_support": stage_metric(future, "full_foot_support"),
        "terminal_spin_control": stage_metric(future, "spin_control"),
        "terminal_forward_speed": stage_metric(future, "forward_speed"),
        "terminal_torso_z_quality": stage_metric(future, "torso_z_quality"),
        "min_support_contact": min_support,
        "min_spin_control": min_spin,
        "max_falling_air": max_falling,
        "max_rotation_penalty": max_rotation,
        "max_body_collision": max_collision,
        "reward_sum": reward_sum,
        "positive_reward": 1.0 if reward_sum > 0.0 else 0.0,
    }


def init_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "metrics": {},
        "labels": Counter(),
        "events": Counter(),
        "episodes": Counter(),
    }


def add_hit(bucket: dict[str, Any], record: dict[str, Any], metrics: dict[str, float]) -> None:
    bucket["count"] += 1
    for key, value in metrics.items():
        bucket["metrics"].setdefault(key, []).append(float(value))
    bucket["labels"][str(record.get("label", "unknown"))] += 1
    bucket["events"][str(record.get("event", "unknown"))] += 1
    bucket["episodes"][str(record.get("episode_id", "unknown"))] += 1


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(bucket["count"]),
        "metrics": {key: avg(values) for key, values in sorted(bucket["metrics"].items())},
        "label_counts": dict(bucket["labels"].most_common()),
        "event_counts": dict(bucket["events"].most_common()),
        "unique_episodes": len(bucket["episodes"]),
    }


def expected_relation(delta: float) -> tuple[str, str]:
    if delta > 0.0:
        return "same", "opposite"
    return "opposite", "same"


def compare_relations(groups: dict[str, dict[str, Any]], delta: float) -> dict[str, Any]:
    expected, contrary = expected_relation(delta)
    expected_metrics = groups.get(expected, {}).get("metrics", {})
    contrary_metrics = groups.get(contrary, {}).get("metrics", {})
    expected_count = int(groups.get(expected, {}).get("count", 0) or 0)
    contrary_count = int(groups.get(contrary, {}).get("count", 0) or 0)
    margin = None
    supported = False
    reason = "insufficient_relation_coverage"
    if expected_count > 0 and contrary_count > 0:
        margin = float(expected_metrics.get("outcome_score", 0.0) - contrary_metrics.get("outcome_score", 0.0))
        supported = margin > 0.0
        reason = "expected_relation_better" if supported else "contrary_relation_better"
    elif expected_count > 0:
        reason = "expected_relation_only"
    elif contrary_count > 0:
        reason = "contrary_relation_only"
    return {
        "expected_relation": expected,
        "contrary_relation": contrary,
        "expected_count": expected_count,
        "contrary_count": contrary_count,
        "outcome_margin": margin,
        "consequence_supported": supported,
        "support_reason": reason,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    shadow_rules = load_shadow_rules(args.shadow_update, args.delta_epsilon)
    audit_payload = load_json(args.tva_counterfactual_audit)
    trace_cache: dict[str, list[dict[str, str]]] = {}
    stats: dict[str, dict[str, Any]] = {}
    rule_by_id = {rule_id(rule): rule for rule in shadow_rules}
    for rid, rule in rule_by_id.items():
        stats[rid] = {
            "id": rid,
            "source": rule.get("source"),
            "rule_key": rule.get("rule_key") or rule.get("substate_key"),
            "action_family": rule.get("action_family"),
            "shadow_confidence_delta": float(rule.get("shadow_confidence_delta", 0.0) or 0.0),
            "shadow_update_reason": rule.get("shadow_update_reason"),
            "tva_shadow_selected_gate_hits": int(rule.get("tva_shadow_selected_gate_hits", 0) or 0),
            "tva_shadow_same_rate": float(rule.get("tva_shadow_same_rate", 0.0) or 0.0),
            "tva_shadow_opposite_rate": float(rule.get("tva_shadow_opposite_rate", 0.0) or 0.0),
            "state_hits": 0,
            "selected_gate_hits": 0,
            "relations": {relation: init_bucket() for relation in relation_order()},
        }

    for record in audit_payload.get("records", []):
        trace_csv = str(record.get("trace_csv", ""))
        if not trace_csv:
            continue
        rows = trace_cache.get(trace_csv)
        if rows is None:
            rows = read_csv(Path(trace_csv))
            trace_cache[trace_csv] = rows
        idx = int(record.get("step", -1) or -1)
        if idx < 0 or idx >= len(rows):
            continue
        row = rows[idx]
        parts = row_parts(row, args.continue_score_min)
        metrics = consequence_metrics(rows, idx, args.horizon)
        for rid, rule in rule_by_id.items():
            if not state_matches(rule, parts):
                continue
            stats[rid]["state_hits"] += 1
            if not selected_gate_applies(rule, row):
                continue
            stats[rid]["selected_gate_hits"] += 1
            relation = selected_action_relation(row, str(rule.get("action_family", "")), args.action_threshold)
            add_hit(stats[rid]["relations"][relation], record, metrics)

    rows_out = []
    for rid, row in stats.items():
        finalized_relations = {relation: finalize_bucket(bucket) for relation, bucket in row["relations"].items()}
        delta = float(row["shadow_confidence_delta"])
        comparison = compare_relations(finalized_relations, delta)
        rows_out.append(
            {
                **{key: value for key, value in row.items() if key != "relations"},
                "relations": finalized_relations,
                **comparison,
            }
        )
    rows_out.sort(key=lambda item: abs(float(item["shadow_confidence_delta"])), reverse=True)
    supported = [row for row in rows_out if row["consequence_supported"]]
    compared = [row for row in rows_out if row["outcome_margin"] is not None]
    return {
        "audit_family": "tva_shadow_consequence_audit_v1",
        "shadow_update": str(args.shadow_update),
        "tva_counterfactual_audit": str(args.tva_counterfactual_audit),
        "horizon": int(args.horizon),
        "action_threshold": float(args.action_threshold),
        "continue_score_min": float(args.continue_score_min),
        "num_shadow_changed_rules": len(rows_out),
        "num_rules_with_relation_comparison": len(compared),
        "num_consequence_supported": len(supported),
        "support_rate_among_compared": len(supported) / len(compared) if compared else 0.0,
        "num_trace_csv_read": len(trace_cache),
        "rules": rows_out,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TVA Shadow Consequence Audit",
        "",
        f"Shadow update: `{payload['shadow_update']}`",
        f"TVA counterfactual audit: `{payload['tva_counterfactual_audit']}`",
        f"Horizon: `{payload['horizon']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Changed rules | `{payload['num_shadow_changed_rules']}` |",
        f"| Rules with same/opposite comparison | `{payload['num_rules_with_relation_comparison']}` |",
        f"| Consequence-supported rules | `{payload['num_consequence_supported']}` |",
        f"| Support rate among compared | `{payload['support_rate_among_compared']:.3f}` |",
        f"| Trace CSV read | `{payload['num_trace_csv_read']}` |",
        "",
        "## Rules",
        "",
        "| Rule | Delta | Expected vs contrary | Counts | Outcome margin | Same score | Opposite score | Reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rules"]:
        same = row["relations"]["same"]
        opposite = row["relations"]["opposite"]
        same_score = same["metrics"].get("outcome_score", 0.0)
        opposite_score = opposite["metrics"].get("outcome_score", 0.0)
        margin = row["outcome_margin"]
        lines.append(
            "| "
            f"`{row['id']}` | "
            f"`{row['shadow_confidence_delta']:+.3f}` | "
            f"`{row['expected_relation']}` vs `{row['contrary_relation']}` | "
            f"`{row['expected_count']}`/`{row['contrary_count']}` | "
            f"{'`n/a`' if margin is None else f'`{margin:+.4f}`'} | "
            f"`{same_score:.4f}` | "
            f"`{opposite_score:.4f}` | "
            f"`{row['support_reason']}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-update", type=Path, required=True)
    parser.add_argument("--tva-counterfactual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA shadow consequence audit: {args.output}")
    print(f"changed rules: {payload['num_shadow_changed_rules']}")
    print(f"support rate among compared: {payload['support_rate_among_compared']:.3f}")


if __name__ == "__main__":
    main()
