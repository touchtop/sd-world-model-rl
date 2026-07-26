#!/usr/bin/env python3
"""Inspect disagreements between TVA shadow updates and rule-effect audits.

The goal is diagnostic: identify whether disagreements are caused by metric
semantics, sparse TVA probe coverage, or local counterfactual value estimates.
The script is read-only and does not update rule libraries.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import row_parts, selected_gate_applies, state_matches
from common import load_json, read_csv, row_to_action, write_json


DEFAULT_RUN_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1"
)
DEFAULT_REPORT_DIR = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "tva_validation_300k_oldscheme_20260515"
)
DEFAULT_OUTPUT = DEFAULT_REPORT_DIR / "tva_shadow_mismatch_audit.json"
DEFAULT_CYCLES = [41, 50, 60]
ACTION_NAMES = ("waist", "hip", "knee", "ankle")


def action_family_from_name(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(name)


def family_from_action(action: list[float], threshold: float) -> str:
    idx = max(range(len(action)), key=lambda i: abs(action[i]))
    if abs(action[idx]) < threshold:
        return "weak"
    return f"{ACTION_NAMES[idx]}_{'pos' if action[idx] > 0 else 'neg'}"


def opposite_family(action_family: str) -> str | None:
    if action_family.endswith("_pos"):
        return action_family[:-4] + "_neg"
    if action_family.endswith("_neg"):
        return action_family[:-4] + "_pos"
    return None


def effect_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row["id"]): row for row in payload.get("rule_summaries", [])}


def shadow_path(args: argparse.Namespace, cycle: int) -> Path:
    return args.report_dir / f"hopper_tva_shadow_update_cycle_{cycle:03d}{args.shadow_suffix}.json"


def tva_audit_path(report_dir: Path, cycle: int) -> Path:
    return report_dir / f"hopper_tva_counterfactual_audit_cycle_{cycle:03d}_stratified.json"


def rule_effect_path(run_root: Path, cycle: int) -> Path:
    return run_root / f"cycle_{cycle:03d}" / "reports" / "rule_effect_audit.json"


def classify_mismatch(row: dict[str, Any], effect: dict[str, Any], args: argparse.Namespace) -> str | None:
    delta = float(row.get("shadow_confidence_delta", 0.0) or 0.0)
    consistent = float(effect.get("direction_consistent_rate", 0.0) or 0.0)
    harmful = float(effect.get("direction_harmful_rate", 0.0) or 0.0)
    if delta < -args.delta_epsilon and consistent >= args.high_consistent_threshold:
        return "tva_weakened_effect_consistent"
    if delta > args.delta_epsilon and harmful >= args.high_harmful_threshold:
        return "tva_strengthened_effect_harmful"
    return None


def family_values(record: dict[str, Any]) -> dict[str, float]:
    items = record.get("probe_actions") or record.get("top_actions") or []
    values: dict[str, float] = {}
    for item in items:
        family = str(item.get("family") or action_family_from_name(str(item.get("name", ""))))
        value = float(item.get("tva_value", 0.0) or 0.0)
        if family not in values or value > values[family]:
            values[family] = value
    return values


def best_family(record: dict[str, Any]) -> str:
    return str(record.get("best_action_family") or action_family_from_name(str(record.get("best_action", ""))))


def actual_family(record: dict[str, Any], threshold: float) -> str:
    if record.get("actual_action_family"):
        return str(record["actual_action_family"])
    return family_from_action([float(x) for x in record.get("actual_action", [])], threshold)


def family_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(max(len(counts), 2))


def init_probe_stats() -> dict[str, Any]:
    return {
        "tva_gate_hits_reconstructed": 0,
        "actual_same_rule": 0,
        "actual_opposite_rule": 0,
        "actual_other": 0,
        "best_same_rule": 0,
        "best_opposite_rule": 0,
        "best_other": 0,
        "actual_same_but_tva_best_opposite": 0,
        "actual_opposite_but_tva_best_same": 0,
        "target_family_value_minus_actual_values": [],
        "opposite_family_value_minus_actual_values": [],
        "best_family_counts": Counter(),
        "actual_family_counts": Counter(),
        "label_counts": Counter(),
        "event_counts": Counter(),
        "examples": [],
    }


def update_probe_stats(
    stats: dict[str, Any],
    rule: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    target = str(rule.get("action_family", ""))
    opposite = opposite_family(target)
    actual = actual_family(record, args.action_threshold)
    best = best_family(record)
    actual_value = float(record.get("actual_tva_value", 0.0) or 0.0)
    values = family_values(record)

    stats["tva_gate_hits_reconstructed"] += 1
    stats["best_family_counts"][best] += 1
    stats["actual_family_counts"][actual] += 1
    stats["label_counts"][str(record.get("label", "unknown"))] += 1
    stats["event_counts"][str(record.get("event", "unknown"))] += 1

    if actual == target:
        stats["actual_same_rule"] += 1
    elif opposite is not None and actual == opposite:
        stats["actual_opposite_rule"] += 1
    else:
        stats["actual_other"] += 1

    if best == target:
        stats["best_same_rule"] += 1
    elif opposite is not None and best == opposite:
        stats["best_opposite_rule"] += 1
    else:
        stats["best_other"] += 1

    if actual == target and opposite is not None and best == opposite:
        stats["actual_same_but_tva_best_opposite"] += 1
    if opposite is not None and actual == opposite and best == target:
        stats["actual_opposite_but_tva_best_same"] += 1

    if target in values:
        stats["target_family_value_minus_actual_values"].append(values[target] - actual_value)
    if opposite is not None and opposite in values:
        stats["opposite_family_value_minus_actual_values"].append(values[opposite] - actual_value)

    if len(stats["examples"]) < args.max_examples:
        stats["examples"].append(
            {
                "episode_id": record.get("episode_id"),
                "step": record.get("step"),
                "label": record.get("label"),
                "event": record.get("event"),
                "actual_family": actual,
                "best_family": best,
                "target_family": target,
                "opposite_family": opposite,
                "actual_tva_value": actual_value,
                "best_tva_value": record.get("best_tva_value"),
                "tva_advantage_over_actual": record.get("tva_advantage_over_actual"),
                "trace_csv": record.get("trace_csv"),
            }
        )


def finalize_probe_stats(stats: dict[str, Any]) -> dict[str, Any]:
    hits = int(stats["tva_gate_hits_reconstructed"])
    out = {
        "tva_gate_hits_reconstructed": hits,
        "actual_same_rule_rate": stats["actual_same_rule"] / hits if hits else 0.0,
        "actual_opposite_rule_rate": stats["actual_opposite_rule"] / hits if hits else 0.0,
        "actual_other_rate": stats["actual_other"] / hits if hits else 0.0,
        "best_same_rule_rate": stats["best_same_rule"] / hits if hits else 0.0,
        "best_opposite_rule_rate": stats["best_opposite_rule"] / hits if hits else 0.0,
        "best_other_rate": stats["best_other"] / hits if hits else 0.0,
        "actual_same_but_tva_best_opposite_rate": stats["actual_same_but_tva_best_opposite"] / hits if hits else 0.0,
        "actual_opposite_but_tva_best_same_rate": stats["actual_opposite_but_tva_best_same"] / hits if hits else 0.0,
        "target_family_value_minus_actual_mean": sum(stats["target_family_value_minus_actual_values"]) / len(stats["target_family_value_minus_actual_values"]) if stats["target_family_value_minus_actual_values"] else 0.0,
        "opposite_family_value_minus_actual_mean": sum(stats["opposite_family_value_minus_actual_values"]) / len(stats["opposite_family_value_minus_actual_values"]) if stats["opposite_family_value_minus_actual_values"] else 0.0,
        "best_family_counts": dict(stats["best_family_counts"].most_common()),
        "actual_family_counts": dict(stats["actual_family_counts"].most_common()),
        "best_family_entropy": family_entropy(stats["best_family_counts"]),
        "actual_family_entropy": family_entropy(stats["actual_family_counts"]),
        "label_counts": dict(stats["label_counts"].most_common()),
        "event_counts": dict(stats["event_counts"].most_common()),
        "examples": stats["examples"],
    }
    return out


def reconstruct_probe_stats(rule: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    stats = init_probe_stats()
    trace_cache: dict[str, list[dict[str, str]]] = {}
    for record in records:
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
        if not state_matches(rule, parts):
            continue
        if not selected_gate_applies(rule, row):
            continue
        update_probe_stats(stats, rule, record, args)
    return finalize_probe_stats(stats)


def audit_cycle(args: argparse.Namespace, cycle: int) -> dict[str, Any]:
    shadow = load_json(shadow_path(args, cycle))
    effects = effect_index(rule_effect_path(args.run_root, cycle))
    tva = load_json(tva_audit_path(args.report_dir, cycle))
    records = tva.get("records", [])
    mismatches: list[dict[str, Any]] = []
    for row in shadow.get("rules", []):
        rid = str(row["id"])
        effect = effects.get(rid, {})
        kind = classify_mismatch(row, effect, args)
        if kind is None:
            continue
        probe = reconstruct_probe_stats(row, records, args)
        effect_hits = int(effect.get("selected_gate_hits", 0) or 0)
        tva_hits = int(row.get("tva_shadow_selected_gate_hits", 0) or 0)
        mismatches.append(
            {
                "id": rid,
                "kind": kind,
                "source": row.get("source"),
                "action_family": row.get("action_family"),
                "confidence": row.get("confidence"),
                "shadow_confidence": row.get("shadow_confidence"),
                "shadow_confidence_delta": row.get("shadow_confidence_delta"),
                "shadow_update_reason": row.get("shadow_update_reason"),
                "tva_shadow_selected_gate_hits": tva_hits,
                "effect_selected_gate_hits": effect_hits,
                "tva_to_effect_hit_ratio": tva_hits / effect_hits if effect_hits else 0.0,
                "tva_shadow_same_rate": row.get("tva_shadow_same_rate"),
                "tva_shadow_opposite_rate": row.get("tva_shadow_opposite_rate"),
                "effect_direction_consistent_rate": effect.get("direction_consistent_rate", 0.0),
                "effect_direction_harmful_rate": effect.get("direction_harmful_rate", 0.0),
                "effect_high_episode_hit_rate": effect.get("high_episode_hit_rate", 0.0),
                "effect_good_state_hit_rate": effect.get("good_state_hit_rate", 0.0),
                "probe_diagnostics": probe,
            }
        )
    mismatches.sort(
        key=lambda item: (
            item["kind"],
            -abs(float(item.get("shadow_confidence_delta", 0.0) or 0.0)),
        )
    )
    return {
        "cycle": cycle,
        "shadow_update": str(shadow_path(args, cycle)),
        "tva_counterfactual_audit": str(tva_audit_path(args.report_dir, cycle)),
        "rule_effect_audit": str(rule_effect_path(args.run_root, cycle)),
        "num_mismatches": len(mismatches),
        "mismatch_counts": dict(Counter(item["kind"] for item in mismatches)),
        "mismatches": mismatches,
    }


def write_markdown(path: Path, payload: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# TVA Shadow Mismatch Audit",
        "",
        "This report inspects rules where TVA shadow and rule-effect audit disagree.",
        "",
        f"Run root: `{payload['run_root']}`",
        f"Report directory: `{payload['report_dir']}`",
        "",
    ]
    for cycle in payload["cycles"]:
        lines.extend(
            [
                f"## Cycle {cycle['cycle']}",
                "",
                f"Mismatches: `{cycle['num_mismatches']}` `{cycle['mismatch_counts']}`",
                "",
                "| Rule | Kind | Delta | TVA/effect hits | TVA same/opposite | Effect consistent/harmful | Actual same/opposite | Best same/opposite | Probe explanation |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in cycle["mismatches"][: args.top_k]:
            probe = item["probe_diagnostics"]
            explanation = "metric_conflict"
            if item["tva_to_effect_hit_ratio"] < args.low_tva_effect_ratio:
                explanation = "sparse_tva_probe_coverage"
            if probe["actual_same_but_tva_best_opposite_rate"] >= args.strong_probe_conflict_rate:
                explanation = "actual_alignment_but_tva_opposite"
            elif probe["best_family_entropy"] >= args.high_entropy_threshold:
                explanation = "diffuse_tva_best_families"
            lines.append(
                "| "
                f"`{item['id']}` | "
                f"`{item['kind']}` | "
                f"{float(item['shadow_confidence_delta']):+.3f} | "
                f"{item['tva_shadow_selected_gate_hits']}/{item['effect_selected_gate_hits']} "
                f"({item['tva_to_effect_hit_ratio']:.5f}) | "
                f"{float(item['tva_shadow_same_rate']):.3f}/{float(item['tva_shadow_opposite_rate']):.3f} | "
                f"{float(item['effect_direction_consistent_rate']):.3f}/{float(item['effect_direction_harmful_rate']):.3f} | "
                f"{probe['actual_same_rule_rate']:.3f}/{probe['actual_opposite_rule_rate']:.3f} | "
                f"{probe['best_same_rule_rate']:.3f}/{probe['best_opposite_rule_rate']:.3f} | "
                f"`{explanation}` |"
            )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--shadow-suffix", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cycles", type=int, nargs="+", default=DEFAULT_CYCLES)
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--high-consistent-threshold", type=float, default=0.65)
    parser.add_argument("--high-harmful-threshold", type=float, default=0.45)
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--low-tva-effect-ratio", type=float, default=0.001)
    parser.add_argument("--strong-probe-conflict-rate", type=float, default=0.10)
    parser.add_argument("--high-entropy-threshold", type=float, default=0.75)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "audit_family": "tva_shadow_mismatch_audit_v1",
        "run_root": str(args.run_root),
        "report_dir": str(args.report_dir),
        "cycles": [audit_cycle(args, cycle) for cycle in args.cycles],
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload, args)
    print(f"wrote TVA shadow mismatch audit: {args.output}")


if __name__ == "__main__":
    main()
