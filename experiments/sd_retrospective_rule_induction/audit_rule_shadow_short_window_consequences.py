#!/usr/bin/env python3
"""Audit short-window consequences for changed rules in a shadow library.

The audit compares a source rule library with a shadow rule library, identifies
rules whose confidence changed, then checks real trajectory windows where those
rules match. For strengthened rules, same-direction actions are expected to have
better short-window outcomes than opposite-direction actions; for weakened
rules, the opposite relation is expected to look better.

This is a read-only consistency check, not a causal estimate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import (
    as_float,
    continuation_score,
    row_parts,
    selected_action_relation,
    selected_gate_applies,
    stage_metric,
    state_matches,
    structural_success,
)
from apply_rule_effect_confidence import rule_id
from common import read_csv


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


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


def changed_rules(source_library: Path, shadow_library: Path, delta_epsilon: float) -> list[dict[str, Any]]:
    source = load_json(source_library)
    shadow = load_json(shadow_library)
    source_by_id = {rule_id(rule): rule for rule in source.get("rules", [])}
    out: list[dict[str, Any]] = []
    for rule in shadow.get("rules", []):
        rid = rule_id(rule)
        before_rule = source_by_id.get(rid)
        if before_rule is None:
            continue
        before = confidence(before_rule)
        after = confidence(rule)
        delta = after - before
        if abs(delta) <= delta_epsilon:
            continue
        merged = dict(rule)
        merged["id"] = rid
        merged["source_confidence"] = before
        merged["shadow_confidence"] = after
        merged["shadow_confidence_delta"] = delta
        out.append(merged)
    out.sort(key=lambda item: abs(float(item["shadow_confidence_delta"])), reverse=True)
    return out


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
        "min_support_contact": min_support,
        "min_spin_control": min_spin,
        "max_falling_air": max_falling,
        "max_body_collision": max_collision,
        "reward_sum": reward_sum,
        "positive_reward": 1.0 if reward_sum > 0.0 else 0.0,
    }


def init_bucket() -> dict[str, Any]:
    return {"count": 0, "metrics": {}, "episodes": Counter()}


def add_hit(bucket: dict[str, Any], episode_key: str, metrics: dict[str, float]) -> None:
    bucket["count"] += 1
    bucket["episodes"][episode_key] += 1
    for key, value in metrics.items():
        bucket["metrics"].setdefault(key, []).append(float(value))


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(bucket["count"]),
        "unique_episodes": len(bucket["episodes"]),
        "metrics": {key: avg(values) for key, values in sorted(bucket["metrics"].items())},
    }


def expected_relation(delta: float) -> tuple[str, str]:
    if delta > 0.0:
        return "same", "opposite"
    return "opposite", "same"


def compare(groups: dict[str, dict[str, Any]], delta: float, min_relation_count: int, min_outcome_margin: float) -> dict[str, Any]:
    expected, contrary = expected_relation(delta)
    expected_group = groups.get(expected, {})
    contrary_group = groups.get(contrary, {})
    expected_count = int(expected_group.get("count", 0) or 0)
    contrary_count = int(contrary_group.get("count", 0) or 0)
    margin = None
    supported = False
    reason = "insufficient_relation_coverage"
    if expected_count >= min_relation_count and contrary_count >= min_relation_count:
        margin = float(
            expected_group.get("metrics", {}).get("outcome_score", 0.0)
            - contrary_group.get("metrics", {}).get("outcome_score", 0.0)
        )
        supported = margin >= min_outcome_margin
        reason = "expected_relation_better" if supported else "contrary_relation_better"
    elif expected_count > 0 or contrary_count > 0:
        reason = "low_relation_count"
    return {
        "expected_relation": expected,
        "contrary_relation": contrary,
        "expected_count": expected_count,
        "contrary_count": contrary_count,
        "outcome_margin": margin,
        "consequence_supported": supported,
        "support_reason": reason,
    }


def trace_files(trace_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for trace_dir in trace_dirs:
        out.extend(sorted(trace_dir.glob("episode_*.csv")))
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rules = changed_rules(args.source_library, args.shadow_library, args.delta_epsilon)
    stats = {
        rule["id"]: {
            "id": rule["id"],
            "source": rule.get("source"),
            "rule_key": rule.get("rule_key") or rule.get("substate_key"),
            "action_family": rule.get("action_family"),
            "source_confidence": rule["source_confidence"],
            "shadow_confidence": rule["shadow_confidence"],
            "shadow_confidence_delta": rule["shadow_confidence_delta"],
            "state_hits": 0,
            "selected_gate_hits": 0,
            "relations": {relation: init_bucket() for relation in ("same", "opposite", "neutral")},
        }
        for rule in rules
    }
    rule_by_id = {rule["id"]: rule for rule in rules}
    paths = trace_files(args.trace_dir)
    for path in paths:
        rows = read_csv(path)
        episode_key = str(path)
        for idx, row in enumerate(rows):
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
                add_hit(stats[rid]["relations"][relation], episode_key, metrics)

    rules_out: list[dict[str, Any]] = []
    for rid, row in stats.items():
        finalized = {relation: finalize_bucket(bucket) for relation, bucket in row["relations"].items()}
        comparison = compare(
            finalized,
            float(row["shadow_confidence_delta"]),
            args.min_relation_count,
            args.min_outcome_margin,
        )
        rules_out.append(
            {
                **{key: value for key, value in row.items() if key != "relations"},
                "relations": finalized,
                **comparison,
            }
        )
    rules_out.sort(key=lambda item: abs(float(item["shadow_confidence_delta"])), reverse=True)
    compared = [row for row in rules_out if row["outcome_margin"] is not None]
    supported = [row for row in compared if row["consequence_supported"]]
    return {
        "audit_family": "rule_shadow_short_window_consequence_v0",
        "source_library": str(args.source_library),
        "shadow_library": str(args.shadow_library),
        "trace_dirs": [str(path) for path in args.trace_dir],
        "horizon": int(args.horizon),
        "action_threshold": float(args.action_threshold),
        "min_relation_count": int(args.min_relation_count),
        "min_outcome_margin": float(args.min_outcome_margin),
        "num_changed_rules": len(rules_out),
        "num_rules_with_relation_comparison": len(compared),
        "num_consequence_supported": len(supported),
        "support_rate_among_compared": len(supported) / len(compared) if compared else 0.0,
        "num_trace_files": len(paths),
        "rules": rules_out,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Rule Shadow Short-Window Consequence Audit",
        "",
        f"Source library: `{payload['source_library']}`",
        f"Shadow library: `{payload['shadow_library']}`",
        f"Horizon: `{payload['horizon']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| changed rules | {payload['num_changed_rules']} |",
        f"| compared rules | {payload['num_rules_with_relation_comparison']} |",
        f"| consequence supported | {payload['num_consequence_supported']} |",
        f"| support rate among compared | {payload['support_rate_among_compared']:.3f} |",
        f"| trace files | {payload['num_trace_files']} |",
        "",
        "## Rules",
        "",
        "| rule | delta | expected/contrary | counts | margin | same score | opposite score | reason |",
        "|---|---:|---|---:|---:|---:|---:|---|",
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
            f"{row['shadow_confidence_delta']:+.3f} | "
            f"`{row['expected_relation']}`/`{row['contrary_relation']}` | "
            f"{row['expected_count']}/{row['contrary_count']} | "
            f"{'n/a' if margin is None else f'{margin:+.4f}'} | "
            f"{same_score:.4f} | "
            f"{opposite_score:.4f} | "
            f"`{row['support_reason']}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--shadow-library", type=Path, required=True)
    parser.add_argument("--trace-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--min-relation-count", type=int, default=10)
    parser.add_argument("--min-outcome-margin", type=float, default=0.0)
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote short-window consequence audit: {args.output}")
    print(f"changed rules: {payload['num_changed_rules']}")
    print(f"support rate among compared: {payload['support_rate_among_compared']:.3f}")


if __name__ == "__main__":
    main()
