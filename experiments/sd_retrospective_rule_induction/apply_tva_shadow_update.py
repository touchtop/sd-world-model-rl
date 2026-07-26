#!/usr/bin/env python3
"""Produce an audit-only TVA shadow update for Hopper rule libraries.

This script does not modify the source rule library or planner prior. It reads
the existing rule library together with a TVA counterfactual audit, then
computes a shadow confidence proposal for each rule. The proposal is written as
an independent report so that TVA can be evaluated as a rule-value signal before
any online use.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import row_parts, selected_gate_applies, state_matches  # noqa: E402
from apply_rule_effect_confidence import confidence_status, clipped, log_hit_weight, rule_id  # noqa: E402
from common import load_json, read_csv, write_json  # noqa: E402


DEFAULT_LIBRARY = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/"
    "cycle_060/rules/library_closed_loop_cycle_060_confidence.json"
)
DEFAULT_TVA_AUDIT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "tva_validation_300k_oldscheme_20260515/"
    "hopper_tva_counterfactual_audit_cycle_060_stratified.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "tva_validation_300k_oldscheme_20260515/"
    "hopper_tva_shadow_update_cycle_060.json"
)


def action_family_from_name(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(name)


def opposite_action_family(action_family: str) -> str | None:
    if action_family.endswith("_pos"):
        return action_family[:-4] + "_neg"
    if action_family.endswith("_neg"):
        return action_family[:-4] + "_pos"
    return None


def family_from_action(action: list[float], threshold: float) -> str:
    names = ("waist", "hip", "knee", "ankle")
    idx = max(range(len(action)), key=lambda i: abs(action[i]))
    if abs(action[idx]) < threshold:
        return "weak"
    return f"{names[idx]}_{'pos' if action[idx] > 0 else 'neg'}"


def family_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(max(len(counts), 2))


def record_family_values(record: dict[str, Any]) -> tuple[dict[str, float], list[tuple[str, float]]]:
    probe_items = record.get("probe_actions")
    if not probe_items:
        probe_items = record.get("top_actions", [])
    family_values: dict[str, float] = {}
    for item in probe_items:
        name = str(item.get("name", ""))
        family = str(item.get("family") or action_family_from_name(name))
        try:
            value = float(item.get("tva_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if family not in family_values or value > family_values[family]:
            family_values[family] = value
    ranked = list(family_values.items())
    ranked.sort(key=lambda item: item[1], reverse=True)
    return family_values, ranked


def load_effect_hit_index(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    payload = load_json(path)
    return {
        str(row["id"]): int(row.get("selected_gate_hits", 0) or 0)
        for row in payload.get("rule_summaries", [])
    }


def init_rule_stats(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rule_id(rule),
        "source": rule.get("source"),
        "state_event": rule.get("state_event"),
        "rule_key": rule.get("rule_key") or rule.get("substate_key"),
        "action_family": rule.get("action_family"),
        "source_confidence": rule.get("confidence"),
        "source_status": rule.get("confidence_status", rule.get("status")),
        "state_hits": 0,
        "selected_gate_hits": 0,
        "same_hits": 0,
        "opposite_hits": 0,
        "neutral_hits": 0,
        "same_advantages": [],
        "opposite_advantages": [],
        "target_family_advantages": [],
        "opposite_family_advantages": [],
        "target_minus_opposite_margins": [],
        "opposite_minus_target_margins": [],
        "same_ranks": [],
        "opposite_ranks": [],
        "tva_advantages": [],
        "best_family_hits": Counter(),
        "seed_hits": Counter(),
        "episode_hits": Counter(),
        "label_hits": Counter(),
        "event_hits": Counter(),
    }


def update_rule_stats(
    stats: dict[str, Any],
    rule: dict[str, Any],
    row: dict[str, str],
    record: dict[str, Any],
    parts: dict[str, str],
    family_top_k: int,
    family_advantage_threshold: float,
) -> None:
    stats["state_hits"] += 1
    stats["seed_hits"][str(record.get("seed", "unknown"))] += 1
    stats["episode_hits"][str(record.get("episode_id", "unknown"))] += 1
    stats["label_hits"][str(record.get("label", "unknown"))] += 1
    stats["event_hits"][str(record.get("event", "unknown"))] += 1

    if not selected_gate_applies(rule, row):
        return

    stats["selected_gate_hits"] += 1
    actual_value = float(record.get("actual_tva_value", 0.0) or 0.0)
    family_values, ranked = record_family_values(record)
    target_family = str(rule.get("action_family", ""))
    opposite_family = opposite_action_family(target_family)
    target_value = family_values.get(target_family)
    opposite_value = family_values.get(opposite_family) if opposite_family else None

    family_rank: dict[str, int] = {family: idx + 1 for idx, (family, _value) in enumerate(ranked)}
    target_rank = family_rank.get(target_family)
    opposite_rank = family_rank.get(opposite_family) if opposite_family else None
    if ranked:
        stats["best_family_hits"][ranked[0][0]] += 1
    target_margin = target_value - opposite_value if target_value is not None and opposite_value is not None else None
    opposite_margin = opposite_value - target_value if target_value is not None and opposite_value is not None else None
    if target_value is not None:
        stats["target_family_advantages"].append(float(target_value - actual_value))
    if opposite_value is not None:
        stats["opposite_family_advantages"].append(float(opposite_value - actual_value))
    if target_margin is not None:
        stats["target_minus_opposite_margins"].append(float(target_margin))
        stats["opposite_minus_target_margins"].append(float(opposite_margin))

    target_support = (
        target_value is not None
        and target_rank is not None
        and target_rank <= family_top_k
        and (target_value - actual_value) >= family_advantage_threshold
        and (target_margin is None or target_margin >= family_advantage_threshold)
    )
    opposite_support = (
        opposite_value is not None
        and opposite_rank is not None
        and opposite_rank <= family_top_k
        and (opposite_value - actual_value) >= family_advantage_threshold
        and (opposite_margin is None or opposite_margin >= family_advantage_threshold)
        and (target_rank is None or opposite_rank < target_rank or opposite_value > (target_value if target_value is not None else -math.inf))
    )

    if target_support:
        stats["same_hits"] += 1
        stats["same_advantages"].append(float(target_value - actual_value))
        stats["same_ranks"].append(float(target_rank))
    elif opposite_support:
        stats["opposite_hits"] += 1
        stats["opposite_advantages"].append(float(opposite_value - actual_value))
        stats["opposite_ranks"].append(float(opposite_rank))
    else:
        stats["neutral_hits"] += 1

    stats["tva_advantages"].append(float(record.get("tva_advantage_over_actual", 0.0) or 0.0))


def finalized_rule_rows(
    rules: list[dict[str, Any]],
    stats_by_id: dict[str, dict[str, Any]],
    effect_hits_by_id: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for rule in rules:
        rid = rule_id(rule)
        stats = stats_by_id[rid]
        hits = int(stats["selected_gate_hits"])
        state_hits = int(stats["state_hits"])
        effect_hits = int(effect_hits_by_id.get(rid, 0) or 0)
        tva_effect_ratio = hits / effect_hits if effect_hits else None
        previous = clipped(float(rule.get("confidence", args.default_confidence) or 0.0))
        confidence = previous
        reason = "unchanged"
        components: dict[str, Any] = {
            "previous_confidence": previous,
            "tva_shadow_audit_id": rid,
            "state_hits": state_hits,
        }

        same_rate = stats["same_hits"] / hits if hits else 0.0
        opposite_rate = stats["opposite_hits"] / hits if hits else 0.0
        neutral_rate = stats["neutral_hits"] / hits if hits else 0.0
        mean_tva_advantage = float(mean(stats["tva_advantages"])) if stats["tva_advantages"] else 0.0
        mean_same_advantage = float(mean(stats["same_advantages"])) if stats["same_advantages"] else 0.0
        mean_opposite_advantage = float(mean(stats["opposite_advantages"])) if stats["opposite_advantages"] else 0.0
        mean_target_advantage = float(mean(stats["target_family_advantages"])) if stats["target_family_advantages"] else 0.0
        mean_target_minus_opposite = float(mean(stats["target_minus_opposite_margins"])) if stats["target_minus_opposite_margins"] else 0.0
        mean_opposite_minus_target = float(mean(stats["opposite_minus_target_margins"])) if stats["opposite_minus_target_margins"] else 0.0
        same_rank_mean = float(mean(stats["same_ranks"])) if stats["same_ranks"] else 0.0
        opposite_rank_mean = float(mean(stats["opposite_ranks"])) if stats["opposite_ranks"] else 0.0
        best_entropy = family_entropy(stats["best_family_hits"])
        hit_weight = log_hit_weight(hits, args.hit_ref)
        advantage_strength = math.tanh(max(0.0, mean_tva_advantage) / max(args.advantage_scale, 1e-6))
        same_strength = math.tanh(max(0.0, mean_same_advantage) / max(args.advantage_scale, 1e-6))
        opposite_strength = math.tanh(max(0.0, mean_opposite_advantage) / max(args.advantage_scale, 1e-6))
        coverage_ok = hits >= args.min_hits and (
            tva_effect_ratio is None or tva_effect_ratio >= args.min_effect_hit_ratio
        )
        positive_direction_ok = same_rate >= opposite_rate + args.min_directional_rate_margin
        harmful_direction_ok = opposite_rate >= same_rate + args.min_directional_rate_margin
        positive_gate_ok = (
            coverage_ok
            and same_rate >= args.min_positive_rate
            and positive_direction_ok
            and best_entropy <= args.max_best_family_entropy
            and mean_target_minus_opposite >= args.target_opposite_margin
            and mean_target_advantage >= args.family_advantage_threshold
        )
        harmful_gate_ok = (
            coverage_ok
            and opposite_rate >= args.min_opposite_rate
            and harmful_direction_ok
            and best_entropy <= args.max_best_family_entropy
            and mean_opposite_minus_target >= args.target_opposite_margin
            and mean_opposite_advantage >= args.family_advantage_threshold
        )
        positive_signal = (
            same_rate * max(advantage_strength, same_strength) * hit_weight
            if positive_gate_ok
            else 0.0
        )
        harmful_signal = (
            opposite_rate * max(advantage_strength, opposite_strength) * hit_weight
            if harmful_gate_ok
            else 0.0
        )
        positive_delta = args.positive_lr * positive_signal
        harmful_delta = args.harmful_lr * harmful_signal

        if hits == 0:
            confidence *= args.absence_decay
            reason = "missing_from_tva_audit_absence_decay"
            components.update({"absence_decay": args.absence_decay, "selected_gate_hits": hits})
        elif hits < args.min_hits:
            confidence *= args.low_hit_decay
            reason = "low_tva_hits_soft_decay"
            components.update({"low_hit_decay": args.low_hit_decay, "selected_gate_hits": hits})
        elif tva_effect_ratio is not None and tva_effect_ratio < args.min_effect_hit_ratio:
            reason = "insufficient_tva_effect_coverage"
            components.update(
                {
                    "selected_gate_hits": hits,
                    "effect_selected_gate_hits": effect_hits,
                    "tva_effect_hit_ratio": tva_effect_ratio,
                    "min_effect_hit_ratio": args.min_effect_hit_ratio,
                }
            )
        else:
            confidence = clipped(confidence + positive_delta - harmful_delta)
            hard_harmful = hits >= args.hard_harmful_min_hits and opposite_rate >= args.hard_harmful_threshold and harmful_signal >= positive_signal
            if hard_harmful:
                confidence = min(
                    confidence,
                    max(args.retired_threshold, args.active_threshold - args.hard_harmful_margin),
                )
                reason = "tva_shadow_harmful_deactivated"
            elif harmful_delta > positive_delta:
                reason = (
                    "tva_shadow_harmful_weakened"
                    if opposite_rate >= args.harmful_threshold
                    else "tva_shadow_soft_weakened"
                )
            elif positive_delta > harmful_delta and positive_delta > 0:
                reason = "tva_shadow_consistent_strengthened"
            elif same_rate >= args.min_positive_rate and best_entropy > args.max_best_family_entropy:
                reason = "tva_shadow_positive_but_diffuse_best_family"
            elif same_rate >= args.min_positive_rate and not positive_direction_ok:
                reason = "tva_shadow_positive_but_low_directional_rate_margin"
            elif same_rate >= args.min_positive_rate and mean_target_minus_opposite < args.target_opposite_margin:
                reason = "tva_shadow_positive_but_low_target_margin"
            elif opposite_rate >= args.min_opposite_rate and best_entropy > args.max_best_family_entropy:
                reason = "tva_shadow_opposite_but_diffuse_best_family"
            elif opposite_rate >= args.min_opposite_rate and not harmful_direction_ok:
                reason = "tva_shadow_opposite_but_low_directional_rate_margin"
            elif opposite_rate >= args.min_opposite_rate and mean_opposite_minus_target < args.target_opposite_margin:
                reason = "tva_shadow_opposite_but_low_margin"
            else:
                reason = "tva_shadow_observed_no_large_change"
            components.update(
                {
                    "selected_gate_hits": hits,
                    "effect_selected_gate_hits": effect_hits,
                    "tva_effect_hit_ratio": tva_effect_ratio,
                    "same_hits": stats["same_hits"],
                    "opposite_hits": stats["opposite_hits"],
                    "neutral_hits": stats["neutral_hits"],
                    "same_rate": same_rate,
                    "opposite_rate": opposite_rate,
                    "neutral_rate": neutral_rate,
                    "mean_tva_advantage_over_actual": mean_tva_advantage,
                    "mean_same_family_advantage": mean_same_advantage,
                    "mean_opposite_family_advantage": mean_opposite_advantage,
                    "mean_target_family_advantage": mean_target_advantage,
                    "mean_target_minus_opposite_margin": mean_target_minus_opposite,
                    "mean_opposite_minus_target_margin": mean_opposite_minus_target,
                    "mean_same_family_rank": same_rank_mean,
                    "mean_opposite_family_rank": opposite_rank_mean,
                    "best_family_entropy": best_entropy,
                    "best_family_counts": dict(stats["best_family_hits"].most_common()),
                    "hit_weight": hit_weight,
                    "advantage_strength": advantage_strength,
                    "same_strength": same_strength,
                    "opposite_strength": opposite_strength,
                    "coverage_ok": coverage_ok,
                    "positive_direction_ok": positive_direction_ok,
                    "harmful_direction_ok": harmful_direction_ok,
                    "positive_gate_ok": positive_gate_ok,
                    "harmful_gate_ok": harmful_gate_ok,
                    "min_positive_rate": args.min_positive_rate,
                    "min_opposite_rate": args.min_opposite_rate,
                    "min_directional_rate_margin": args.min_directional_rate_margin,
                    "max_best_family_entropy": args.max_best_family_entropy,
                    "target_opposite_margin": args.target_opposite_margin,
                    "positive_signal": positive_signal,
                    "harmful_signal": harmful_signal,
                    "positive_delta": positive_delta,
                    "harmful_delta": harmful_delta,
                    "family_top_k": args.family_top_k,
                    "family_advantage_threshold": args.family_advantage_threshold,
                    "hard_harmful": hard_harmful,
                }
            )

        status = confidence_status(confidence, args.active_threshold, args.retired_threshold)
        delta = confidence - previous
        deltas.append(delta)
        rows.append(
            {
                **rule,
                "shadow_confidence": round(confidence, 6),
                "shadow_confidence_status": status,
                "shadow_effective_prior_scale": round(confidence, 6),
                "shadow_update_reason": reason,
                "shadow_update_components": components,
                "shadow_confidence_delta": round(delta, 6),
                "shadow_last_updated_utc": datetime.now(timezone.utc).isoformat(),
                "shadow_status_change": f"{rule.get('confidence_status', rule.get('status'))} -> {status}",
                "tva_shadow_selected_gate_hits": hits,
                "tva_shadow_effect_selected_gate_hits": effect_hits,
                "tva_shadow_effect_hit_ratio": tva_effect_ratio,
                "tva_shadow_state_hits": state_hits,
                "tva_shadow_same_rate": same_rate,
                "tva_shadow_opposite_rate": opposite_rate,
                "tva_shadow_best_family_entropy": best_entropy,
                "tva_shadow_target_opposite_margin": mean_target_minus_opposite,
                "tva_shadow_mean_advantage": mean_tva_advantage,
            }
        )
    summary = {
        "num_rules": len(rows),
        "num_strengthened": sum(1 for row in rows if float(row["shadow_confidence_delta"]) > 1e-6),
        "num_weakened": sum(1 for row in rows if float(row["shadow_confidence_delta"]) < -1e-6),
        "num_unchanged": sum(1 for row in rows if abs(float(row["shadow_confidence_delta"])) <= 1e-6),
        "num_retired": sum(1 for row in rows if row["shadow_confidence_status"] == "confidence_retired"),
        "num_active": sum(1 for row in rows if row["shadow_confidence_status"] == "confidence_active"),
        "num_observing": sum(1 for row in rows if row["shadow_confidence_status"] == "confidence_observing"),
        "mean_shadow_delta": float(mean(deltas)) if deltas else 0.0,
        "median_shadow_delta": sorted(deltas)[len(deltas) // 2] if deltas else 0.0,
        "max_shadow_delta": max(deltas) if deltas else 0.0,
        "min_shadow_delta": min(deltas) if deltas else 0.0,
    }
    return rows, summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper TVA Shadow Update",
        "",
        "This is an audit-only shadow proposal. It does not change the live rule library or planner prior.",
        "",
        f"Source library: `{payload['source_library']}`",
        f"TVA counterfactual audit: `{payload['source_tva_audit']}`",
        f"Rule-effect audit: `{payload['source_rule_effect_audit']}`",
        f"Family top-k: `{payload['family_top_k']}`",
        f"Family advantage threshold: `{payload['family_advantage_threshold']}`",
        f"Minimum TVA/effect hit ratio: `{payload['min_effect_hit_ratio']}`",
        f"Maximum best-family entropy for strengthening: `{payload['max_best_family_entropy']}`",
        f"Target-opposite margin: `{payload['target_opposite_margin']}`",
        f"Minimum directional rate margin: `{payload['min_directional_rate_margin']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "num_rules",
        "num_strengthened",
        "num_weakened",
        "num_unchanged",
        "num_retired",
        "num_active",
        "num_observing",
        "mean_shadow_delta",
        "median_shadow_delta",
        "max_shadow_delta",
        "min_shadow_delta",
    ):
        value = payload["summary"][key]
        if isinstance(value, float):
            lines.append(f"| `{key}` | `{value:.6f}` |")
        else:
            lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Largest Strengthenings", "", "| Rule | Source | Current | Shadow | Delta | TVA/effect hits | Same / Opposite | Entropy | Margin | Reason |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in sorted(payload["rules"], key=lambda item: float(item["shadow_confidence_delta"]), reverse=True)[:15]:
        lines.append(
            "| "
            f"`{row['id']}` | "
            f"`{row.get('source', '')}` | "
            f"`{float(row.get('confidence', 0.0) or 0.0):.3f}` | "
            f"`{float(row['shadow_confidence']):.3f}` | "
            f"`{float(row['shadow_confidence_delta']):+.3f}` | "
            f"`{row['tva_shadow_selected_gate_hits']}`/`{row.get('tva_shadow_effect_selected_gate_hits', 0)}` | "
            f"`{row['tva_shadow_same_rate']:.3f}` / `{row['tva_shadow_opposite_rate']:.3f}` | "
            f"`{row['tva_shadow_best_family_entropy']:.3f}` | "
            f"`{row['tva_shadow_target_opposite_margin']:.3f}` | "
            f"`{row['shadow_update_reason']}` |"
        )

    lines.extend(["", "## Largest Weakenings", "", "| Rule | Source | Current | Shadow | Delta | TVA/effect hits | Same / Opposite | Entropy | Margin | Reason |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in sorted(payload["rules"], key=lambda item: float(item["shadow_confidence_delta"]))[:15]:
        lines.append(
            "| "
            f"`{row['id']}` | "
            f"`{row.get('source', '')}` | "
            f"`{float(row.get('confidence', 0.0) or 0.0):.3f}` | "
            f"`{float(row['shadow_confidence']):.3f}` | "
            f"`{float(row['shadow_confidence_delta']):+.3f}` | "
            f"`{row['tva_shadow_selected_gate_hits']}`/`{row.get('tva_shadow_effect_selected_gate_hits', 0)}` | "
            f"`{row['tva_shadow_same_rate']:.3f}` / `{row['tva_shadow_opposite_rate']:.3f}` | "
            f"`{row['tva_shadow_best_family_entropy']:.3f}` | "
            f"`{row['tva_shadow_target_opposite_margin']:.3f}` | "
            f"`{row['shadow_update_reason']}` |"
        )

    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--tva-counterfactual-audit", type=Path, default=DEFAULT_TVA_AUDIT)
    parser.add_argument("--rule-effect-audit", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--family-top-k", type=int, default=2)
    parser.add_argument("--family-advantage-threshold", type=float, default=0.05)
    parser.add_argument("--default-confidence", type=float, default=0.56)
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--min-hits", type=int, default=20)
    parser.add_argument("--min-effect-hit-ratio", type=float, default=0.003)
    parser.add_argument("--min-positive-rate", type=float, default=0.10)
    parser.add_argument("--min-opposite-rate", type=float, default=0.10)
    parser.add_argument("--min-directional-rate-margin", type=float, default=0.05)
    parser.add_argument("--max-best-family-entropy", type=float, default=0.65)
    parser.add_argument("--target-opposite-margin", type=float, default=0.50)
    parser.add_argument("--hit-ref", type=float, default=300.0)
    parser.add_argument("--positive-lr", type=float, default=0.25)
    parser.add_argument("--harmful-lr", type=float, default=0.70)
    parser.add_argument("--absence-decay", type=float, default=1.0)
    parser.add_argument("--low-hit-decay", type=float, default=1.0)
    parser.add_argument("--advantage-scale", type=float, default=20.0)
    parser.add_argument("--harmful-threshold", type=float, default=0.20)
    parser.add_argument("--hard-harmful-threshold", type=float, default=0.35)
    parser.add_argument("--hard-harmful-min-hits", type=int, default=60)
    parser.add_argument("--hard-harmful-margin", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_json(args.library)
    audit = load_json(args.tva_counterfactual_audit)
    effect_hits_by_id = load_effect_hit_index(args.rule_effect_audit)
    rules = list(library.get("rules", []))
    stats_by_id = {rule_id(rule): init_rule_stats(rule) for rule in rules}

    trace_cache: dict[str, list[dict[str, str]]] = {}
    matched_records = 0
    for record in audit.get("records", []):
        trace_csv = str(record.get("trace_csv", ""))
        if not trace_csv:
            continue
        rows = trace_cache.get(trace_csv)
        if rows is None:
            rows = read_csv(Path(trace_csv))
            trace_cache[trace_csv] = rows
        step = int(record.get("step", -1) or -1)
        if step < 0 or step >= len(rows):
            continue
        row = rows[step]
        parts = row_parts(row, args.continue_score_min)
        for rule in rules:
            if not state_matches(rule, parts):
                continue
            matched_records += 1
            update_rule_stats(
                stats_by_id[rule_id(rule)],
                rule,
                row,
                record,
                parts,
                args.family_top_k,
                args.family_advantage_threshold,
            )

    rows, summary = finalized_rule_rows(rules, stats_by_id, effect_hits_by_id, args)
    payload = {
        "audit_family": "hopper_tva_shadow_update_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_library": str(args.library),
        "source_tva_audit": str(args.tva_counterfactual_audit),
        "source_rule_effect_audit": str(args.rule_effect_audit) if args.rule_effect_audit else None,
        "family_top_k": int(args.family_top_k),
        "family_advantage_threshold": float(args.family_advantage_threshold),
        "min_effect_hit_ratio": float(args.min_effect_hit_ratio),
        "min_positive_rate": float(args.min_positive_rate),
        "min_opposite_rate": float(args.min_opposite_rate),
        "min_directional_rate_margin": float(args.min_directional_rate_margin),
        "max_best_family_entropy": float(args.max_best_family_entropy),
        "target_opposite_margin": float(args.target_opposite_margin),
        "continue_score_min": float(args.continue_score_min),
        "matched_state_rule_pairs": int(matched_records),
        "summary": summary,
        "rules": rows,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA shadow update: {args.output}")
    print(f"matched state-rule pairs: {matched_records}")


if __name__ == "__main__":
    main()
