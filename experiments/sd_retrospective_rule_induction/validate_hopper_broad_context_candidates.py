#!/usr/bin/env python3
"""Validate broad Hopper transition-context candidates with contrast data.

This validator is deliberately offline and conservative.  It does not promote
rules into the online library.  It checks whether a broad candidate extracted
from high-score teacher traces is stable enough to justify a later SD/TVA
counterfactual audit.

The validation logic follows the current methodological split:

    broad candidate extraction -> contrast/diagnostic validation -> SD/TVA validation

Fine-grained labels are used only as diagnostic subregions; they do not define
the broad candidate itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, mean_or_zero, write_json  # noqa: E402


def load_extractor_module():
    module_path = Path(__file__).resolve().with_name("extract_hopper_transition_context_rules.py")
    spec = importlib.util.spec_from_file_location("hopper_transition_context_extractor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extractor module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def outcome_rate(rows: list[dict[str, Any]], outcome: str) -> float:
    return mean([1.0 if row.get("future_outcome") == outcome else 0.0 for row in rows])


def family_summary(rows: list[dict[str, Any]], target_family: str) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("action_family", ""))].append(row)
    family_rows = []
    for family, vals in by_family.items():
        scores = [float(row.get("transition_score", 0.0) or 0.0) for row in vals]
        family_rows.append(
            {
                "action_family": family,
                "support": len(vals),
                "mean_score": mean(scores),
                "cycle_continue_rate": outcome_rate(vals, "cycle_continue"),
                "unstable_rate": outcome_rate(vals, "unstable"),
            }
        )
    family_rows.sort(key=lambda item: (item["mean_score"], item["support"]), reverse=True)
    target_rows = by_family.get(target_family, [])
    target_scores = [float(row.get("transition_score", 0.0) or 0.0) for row in target_rows]
    baseline_scores = [
        float(row.get("transition_score", 0.0) or 0.0)
        for row in rows
        if row.get("action_family") != target_family
    ]
    return {
        "target_support": len(target_rows),
        "target_coverage": len(target_rows) / len(rows) if rows else 0.0,
        "target_mean_score": mean(target_scores),
        "baseline_mean_score": mean(baseline_scores),
        "target_advantage": mean(target_scores) - mean(baseline_scores) if target_scores and baseline_scores else 0.0,
        "top_families": family_rows[:8],
    }


def diagnostic_key(row: dict[str, Any]) -> str:
    parts = row.get("parts", {})
    return "|".join(
        [
            f"omega={parts.get('omega_bin', 'unknown')}",
            f"support={parts.get('support_bin', 'unknown')}",
            f"risk={parts.get('risk_bin', 'unknown')}",
        ]
    )


def diagnostic_subregions(
    rows: list[dict[str, Any]],
    target_family: str,
    min_support: int,
    min_advantage: float,
    max_unstable_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[diagnostic_key(row)].append(row)
    subregions: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for key, vals in sorted(by_key.items(), key=lambda item: len(item[1]), reverse=True):
        fam = family_summary(vals, target_family)
        row = {
            "diagnostic_key": key,
            "support": len(vals),
            "cycle_continue_rate": outcome_rate(vals, "cycle_continue"),
            "unstable_rate": outcome_rate(vals, "unstable"),
            **fam,
        }
        row["flagged"] = (
            len(vals) >= min_support
            and (
                row["target_advantage"] < min_advantage
                or row["unstable_rate"] > max_unstable_rate
            )
        )
        subregions.append(row)
        if row["flagged"]:
            flagged.append(row)
    return subregions, flagged


def summarize_context(rows: list[dict[str, Any]], target_family: str) -> dict[str, Any]:
    outcomes = Counter(str(row.get("future_outcome", "unknown")) for row in rows)
    return {
        "support": len(rows),
        "outcome_counts": dict(outcomes),
        "cycle_continue_rate": outcome_rate(rows, "cycle_continue"),
        "recoverable_progress_rate": outcome_rate(rows, "recoverable_progress"),
        "unstable_rate": outcome_rate(rows, "unstable"),
        **family_summary(rows, target_family),
    }


def decision_for_rule(
    high: dict[str, Any],
    low: dict[str, Any],
    flagged_subregions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if high["support"] < args.min_high_context_support:
        return "rejected_insufficient_teacher_support", ["high-score teacher support is too low"]
    if high["cycle_continue_rate"] < args.min_high_cycle_continue_rate:
        return "rejected_low_teacher_cycle_continuation", ["teacher context does not reliably continue the cycle"]
    if high["unstable_rate"] > args.max_high_unstable_rate:
        return "rejected_high_teacher_instability", ["teacher context has excessive unstable outcomes"]
    if high["target_support"] < args.min_target_family_support:
        return "rejected_low_target_family_support", ["preferred action family has insufficient observed support"]
    if high["target_advantage"] < args.min_target_advantage:
        return "rejected_weak_observational_advantage", ["preferred action family advantage is weak in teacher traces"]

    if low["support"] >= args.min_low_overlap_support:
        if low["cycle_continue_rate"] >= args.max_low_cycle_continue_rate:
            notes.append("low-score contrast also shows cycle continuation in this context")
        if low["unstable_rate"] <= args.min_low_unstable_rate:
            notes.append("low-score contrast is not clearly unstable in this context")

    if flagged_subregions:
        notes.append(f"{len(flagged_subregions)} diagnostic subregions require exclusion or splitting")
        return "split_required_before_counterfactual", notes
    if notes:
        return "counterfactual_with_contrast_caution", notes
    return "counterfactual_ready", ["observational contrast checks passed; SD/TVA validation still required"]


def extractor_args(base_args: argparse.Namespace, trace_globs: list[str], parameters: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        run_prefix="",
        logs_root=Path("logs/hopper-hop"),
        cycles="",
        seeds="",
        trace_glob=trace_globs,
        output=Path("/tmp/unused.json"),
        past_window=int(parameters.get("past_window", base_args.past_window)),
        future_window=int(parameters.get("future_window", base_args.future_window)),
        outcome_mode=str(parameters.get("outcome_mode", base_args.outcome_mode)),
        phase_schema=str(parameters.get("phase_schema", base_args.phase_schema)),
        context_mode=str(parameters.get("context_mode", base_args.context_mode)),
        action_threshold=float(parameters.get("action_threshold", base_args.action_threshold)),
        focus_grounded_hop=False,
        min_context_support=0,
        min_family_support=0,
        min_score_advantage=0.0,
        max_rules=0,
        max_contexts_report=0,
        max_families_per_context_report=8,
    )


def validate(args: argparse.Namespace) -> dict[str, Any]:
    extractor = load_extractor_module()
    library = load_json(args.candidate_library)
    parameters = dict(library.get("parameters", {}))
    high_records = extractor.extract_records(extractor_args(args, args.high_trace_glob, parameters))
    low_records = extractor.extract_records(extractor_args(args, args.low_trace_glob, parameters)) if args.low_trace_glob else []

    high_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    low_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in high_records:
        high_by_key[str(row["context_key"])].append(row)
    for row in low_records:
        low_by_key[str(row["context_key"])].append(row)

    rule_reports: list[dict[str, Any]] = []
    for rule in library.get("rules", []):
        key = str(rule.get("rule_key"))
        target_family = str(rule.get("action_family"))
        high_rows = high_by_key.get(key, [])
        low_rows = low_by_key.get(key, [])
        high_summary = summarize_context(high_rows, target_family)
        low_summary = summarize_context(low_rows, target_family)
        subregions, flagged = diagnostic_subregions(
            high_rows,
            target_family,
            args.min_subregion_support,
            args.min_subregion_advantage,
            args.max_subregion_unstable_rate,
        )
        decision, notes = decision_for_rule(high_summary, low_summary, flagged, args)
        rule_reports.append(
            {
                "id": rule.get("id"),
                "rule_key": key,
                "action_family": target_family,
                "source_rule": rule,
                "decision": decision,
                "decision_notes": notes,
                "high_score_context": high_summary,
                "low_score_context": low_summary,
                "diagnostic_subregions": subregions[: args.max_subregions_report],
                "flagged_subregions": flagged[: args.max_subregions_report],
            }
        )

    decisions = Counter(row["decision"] for row in rule_reports)
    return {
        "analysis_family": "hopper_broad_context_candidate_validator_v1",
        "candidate_library": str(args.candidate_library),
        "high_trace_glob": args.high_trace_glob,
        "low_trace_glob": args.low_trace_glob,
        "parameters": {
            "min_high_context_support": args.min_high_context_support,
            "min_target_family_support": args.min_target_family_support,
            "min_target_advantage": args.min_target_advantage,
            "min_high_cycle_continue_rate": args.min_high_cycle_continue_rate,
            "max_high_unstable_rate": args.max_high_unstable_rate,
            "min_low_overlap_support": args.min_low_overlap_support,
            "max_low_cycle_continue_rate": args.max_low_cycle_continue_rate,
            "min_low_unstable_rate": args.min_low_unstable_rate,
            "min_subregion_support": args.min_subregion_support,
            "min_subregion_advantage": args.min_subregion_advantage,
            "max_subregion_unstable_rate": args.max_subregion_unstable_rate,
        },
        "extractor_parameters": parameters,
        "num_high_records": len(high_records),
        "num_low_records": len(low_records),
        "num_rules": len(rule_reports),
        "decision_counts": dict(decisions),
        "rules": rule_reports,
        "status_meaning": {
            "counterfactual_ready": "passes observational contrast checks; requires SD/TVA counterfactual validation before online use",
            "counterfactual_with_contrast_caution": "passes teacher checks but contrast data suggests caution",
            "split_required_before_counterfactual": "broad context has unstable or weak diagnostic subregions; split or exclude before SD/TVA validation",
            "rejected_*": "not recommended for counterfactual validation in current form",
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Broad Context Candidate Validation",
        "",
        f"Candidate library: `{payload['candidate_library']}`",
        f"High records: `{payload['num_high_records']}`",
        f"Low records: `{payload['num_low_records']}`",
        f"Rules: `{payload['num_rules']}`",
        f"Decision counts: `{payload['decision_counts']}`",
        "",
        "This is an observational contrast validator. Passing rules still require SD/TVA counterfactual validation before online use.",
        "",
        "## Rules",
        "",
        "| decision | rule | action | high ctx | high adv | high cycle | high unstable | low ctx | low unstable | flagged subregions |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rules"]:
        high = row["high_score_context"]
        low = row["low_score_context"]
        lines.append(
            f"| `{row['decision']}` | `{row['rule_key']}` | `{row['action_family']}` | "
            f"{high['support']} | {high['target_advantage']:.4f} | "
            f"{high['cycle_continue_rate']:.3f} | {high['unstable_rate']:.3f} | "
            f"{low['support']} | {low['unstable_rate']:.3f} | {len(row['flagged_subregions'])} |"
        )
    lines.extend(["", "## Flagged Subregions", ""])
    for row in payload["rules"]:
        if not row["flagged_subregions"]:
            continue
        lines.extend([f"### `{row['rule_key']} -> {row['action_family']}`", ""])
        for sub in row["flagged_subregions"][:10]:
            lines.append(
                f"- `{sub['diagnostic_key']}`: support `{sub['support']}`, "
                f"adv `{sub['target_advantage']:.4f}`, unstable `{sub['unstable_rate']:.3f}`"
            )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--high-trace-glob", action="append", required=True)
    parser.add_argument("--low-trace-glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--past-window", type=int, default=8)
    parser.add_argument("--future-window", type=int, default=20)
    parser.add_argument("--outcome-mode", default="cycle_v1")
    parser.add_argument("--phase-schema", default="simple_v1")
    parser.add_argument("--context-mode", default="broad_context_v1")
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--min-high-context-support", type=int, default=50)
    parser.add_argument("--min-target-family-support", type=int, default=5)
    parser.add_argument("--min-target-advantage", type=float, default=0.04)
    parser.add_argument("--min-high-cycle-continue-rate", type=float, default=0.90)
    parser.add_argument("--max-high-unstable-rate", type=float, default=0.10)
    parser.add_argument("--min-low-overlap-support", type=int, default=20)
    parser.add_argument("--max-low-cycle-continue-rate", type=float, default=0.20)
    parser.add_argument("--min-low-unstable-rate", type=float, default=0.60)
    parser.add_argument("--min-subregion-support", type=int, default=20)
    parser.add_argument("--min-subregion-advantage", type=float, default=-0.01)
    parser.add_argument("--max-subregion-unstable-rate", type=float, default=0.15)
    parser.add_argument("--max-subregions-report", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote broad context candidate validation: {args.output}")
    print(f"rules={payload['num_rules']} decisions={payload['decision_counts']}")


if __name__ == "__main__":
    main()
