#!/usr/bin/env python3
"""Generate broad Hopper rules from SD/TVA joint counterfactual rankings.

This script keeps the rule form simple: a broad observable context maps to one
preferred action family.  Unlike the observational extractor, the action family
is not chosen from the most common or highest-return action observed in teacher
traces.  It is chosen from the action family that most consistently appears in
the top ranks under short SD rollout return plus TVA terminal value.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_json, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from extract_hopper_transition_context_rules import opposite_action_family  # noqa: E402
from validate_hopper_broad_context_counterfactuals import (  # noqa: E402
    evaluate_record,
    extractor_args,
    load_extractor_module,
    load_sd_model,
    load_value_net,
    select_records,
)


RULE_FAMILIES = {
    "waist_neg",
    "waist_pos",
    "hip_neg",
    "hip_pos",
    "knee_neg",
    "knee_pos",
    "ankle_neg",
    "ankle_pos",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    vals = [count / total for count in counter.values() if count > 0]
    ent = -sum(p * math.log(p) for p in vals)
    return ent / math.log(len(vals)) if len(vals) > 1 else 0.0


def outcome_rates(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "cycle_continue_rate": mean([1.0 if r.get("future_outcome") == "cycle_continue" else 0.0 for r in records]),
        "recoverable_progress_rate": mean([1.0 if r.get("future_outcome") == "recoverable_progress" else 0.0 for r in records]),
        "unstable_rate": mean([1.0 if r.get("future_outcome") == "unstable" else 0.0 for r in records]),
        "mean_transition_score": mean([float(r.get("transition_score", 0.0)) for r in records]),
    }


def family_rows(best_counts: Counter[str], top3_counts: Counter[str], n: int) -> list[dict[str, Any]]:
    rows = []
    for family in sorted(RULE_FAMILIES):
        rows.append(
            {
                "action_family": family,
                "joint_best_count": int(best_counts.get(family, 0)),
                "joint_top3_count": int(top3_counts.get(family, 0)),
                "joint_best_rate": best_counts.get(family, 0) / n if n else 0.0,
                "joint_top3_rate": top3_counts.get(family, 0) / n if n else 0.0,
            }
        )
    rows.sort(key=lambda row: (row["joint_top3_rate"], row["joint_best_rate"], row["joint_top3_count"]), reverse=True)
    return rows


def load_source(path: Path | None) -> dict[str, Any]:
    return load_json(path) if path is not None and path.exists() else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = load_source(args.source_library)
    params = dict(source.get("parameters", source.get("extractor_parameters", {})))
    trace_globs = args.trace_glob or list(params.get("trace_glob", []))
    if not trace_globs:
        raise ValueError("Provide --trace-glob or --source-library with parameters.trace_glob.")

    extractor = load_extractor_module()
    records = extractor.extract_records(extractor_args(args, trace_globs, params))
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_context[str(record["context_key"])].append(record)

    model = load_sd_model(args.sd_checkpoint, args.device)
    value_net = load_value_net(args.tva_checkpoint, args.device)
    action_bank = build_action_bank(args.amplitudes)
    trace_cache: dict[str, list[dict[str, str]]] = {}

    rules: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for key, context_records in sorted(by_context.items()):
        if len(context_records) < args.min_context_support:
            continue
        rates = outcome_rates(context_records)
        good_rate = rates["cycle_continue_rate"] + rates["recoverable_progress_rate"]
        selected = select_records(context_records, args.max_records_per_context, args.seed)
        best_counts: Counter[str] = Counter()
        top3_counts: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        for record in selected:
            trace_csv = str(record["trace_csv"])
            rows = trace_cache.get(trace_csv)
            if rows is None:
                rows = read_csv(Path(trace_csv))
                trace_cache[trace_csv] = rows
            idx = int(record["index"])
            if idx < 0 or idx >= len(rows):
                continue
            result = evaluate_record(model, value_net, rows[idx], action_bank, args.horizon, args.gamma, args.device)
            joint_top = result["joint_top_families"]
            if not joint_top:
                continue
            best_family = str(joint_top[0]["family"])
            if best_family in RULE_FAMILIES:
                best_counts[best_family] += 1
            for item in joint_top[:3]:
                family = str(item["family"])
                if family in RULE_FAMILIES:
                    top3_counts[family] += 1
            if len(examples) < args.max_examples_per_context:
                examples.append(
                    {
                        "trace_csv": trace_csv,
                        "episode": record.get("episode"),
                        "step": record.get("step"),
                        "future_outcome": record.get("future_outcome"),
                        "joint_top_families": joint_top[:5],
                    }
                )

        n = sum(1 for _ in selected)
        fam_rows = family_rows(best_counts, top3_counts, n)
        if not fam_rows:
            continue
        winner = fam_rows[0]
        entropy = entropy_from_counter(best_counts)
        decision = "rejected_no_stable_counterfactual_family"
        notes: list[str] = []
        if good_rate < args.min_good_outcome_rate:
            notes.append("teacher context does not have enough cycle/recoverable outcomes")
        if rates["unstable_rate"] > args.max_unstable_rate:
            notes.append("teacher context has too many unstable futures")
        if winner["joint_top3_rate"] < args.min_joint_top3_rate:
            notes.append("best counterfactual family is not frequently top-3")
        if winner["joint_best_rate"] < args.min_joint_best_rate:
            notes.append("best counterfactual family is rarely the top family")
        if entropy > args.max_best_family_entropy:
            notes.append("joint best-family distribution is too diffuse")
        if not notes:
            decision = "counterfactual_rule_candidate"
        elif winner["joint_top3_rate"] >= args.weak_joint_top3_rate and good_rate >= args.min_good_outcome_rate:
            decision = "weak_counterfactual_pattern"

        context_summary = {
            "context_key": key,
            "context_support": len(context_records),
            "num_counterfactual_records": n,
            "decision": decision,
            "decision_notes": notes,
            "good_outcome_rate": good_rate,
            **rates,
            "best_family_entropy": entropy,
            "top_families": fam_rows[: args.max_families_per_context_report],
            "examples": examples,
        }
        contexts.append(context_summary)
        if decision == "counterfactual_rule_candidate":
            rules.append(
                {
                    "id": f"broad_context_counterfactual|{key}->prefer:{winner['action_family']}",
                    "rule_key": key,
                    "substate_key": key,
                    "rule_role": "transition_context",
                    "source": "sd_tva_joint_counterfactual_v1",
                    "rule_origin": "broad_context_counterfactual_generator_v1",
                    "status": "counterfactual_rule_candidate",
                    "intervention_type": "soft_prefer_action_family",
                    "action_family": winner["action_family"],
                    "opposite_action_family": opposite_action_family(winner["action_family"]),
                    "context_support": len(context_records),
                    "support": winner["joint_top3_count"],
                    "num_counterfactual_records": n,
                    "joint_top3_rate": winner["joint_top3_rate"],
                    "joint_best_rate": winner["joint_best_rate"],
                    "best_family_entropy": entropy,
                    "good_outcome_rate": good_rate,
                    **rates,
                    "top_families": fam_rows[: args.max_families_per_context_report],
                    "audit_notes": [
                        "Action family is generated from SD rollout return plus TVA terminal value, not from observed action frequency.",
                        "Candidate remains offline-only until low-score contrast and later online rule-library conversion are explicitly performed.",
                    ],
                }
            )

    rules.sort(key=lambda row: (row["joint_top3_rate"], row["joint_best_rate"], row["context_support"]), reverse=True)
    contexts.sort(key=lambda row: (row["decision"] == "counterfactual_rule_candidate", row["context_support"]), reverse=True)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_family": "hopper_broad_context_sd_tva_counterfactual_rules_v1",
        "source_library": str(args.source_library) if args.source_library else None,
        "sd_checkpoint": str(args.sd_checkpoint),
        "tva_checkpoint": str(args.tva_checkpoint),
        "trace_glob": trace_globs,
        "parameters": {
            "phase_schema": str(params.get("phase_schema", args.phase_schema)),
            "context_mode": str(params.get("context_mode", args.context_mode)),
            "outcome_mode": str(params.get("outcome_mode", args.outcome_mode)),
            "past_window": int(params.get("past_window", args.past_window)),
            "future_window": int(params.get("future_window", args.future_window)),
            "horizon": args.horizon,
            "gamma": args.gamma,
            "amplitudes": args.amplitudes,
            "min_context_support": args.min_context_support,
            "max_records_per_context": args.max_records_per_context,
            "min_good_outcome_rate": args.min_good_outcome_rate,
            "max_unstable_rate": args.max_unstable_rate,
            "min_joint_top3_rate": args.min_joint_top3_rate,
            "weak_joint_top3_rate": args.weak_joint_top3_rate,
            "min_joint_best_rate": args.min_joint_best_rate,
            "max_best_family_entropy": args.max_best_family_entropy,
            "seed": args.seed,
            "device": args.device,
        },
        "num_records": len(records),
        "num_contexts": len(by_context),
        "num_tested_contexts": len(contexts),
        "num_rules": len(rules),
        "decision_counts": dict(Counter(row["decision"] for row in contexts).most_common()),
        "rules": rules[: args.max_rules],
        "context_summaries": contexts[: args.max_contexts_report],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Broad Context SD/TVA Counterfactual Rules",
        "",
        f"- Records: `{payload['num_records']}`",
        f"- Contexts: `{payload['num_contexts']}`",
        f"- Tested contexts: `{payload['num_tested_contexts']}`",
        f"- Candidate rules: `{payload['num_rules']}`",
        f"- Decision counts: `{payload['decision_counts']}`",
        "",
        "## Candidate Rules",
        "",
        "| rule | action | ctx support | cf records | top3 | best | entropy | good | unstable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule in payload["rules"][:30]:
        lines.append(
            f"| `{rule['rule_key']}` | `{rule['action_family']}` | {rule['context_support']} | "
            f"{rule['num_counterfactual_records']} | {rule['joint_top3_rate']:.3f} | "
            f"{rule['joint_best_rate']:.3f} | {rule['best_family_entropy']:.3f} | "
            f"{rule['good_outcome_rate']:.3f} | {rule['unstable_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Tested Contexts",
            "",
            "| decision | context | support | top families | notes |",
            "|---|---|---:|---|---|",
        ]
    )
    for ctx in payload["context_summaries"][:40]:
        fam = ", ".join(
            f"{row['action_family']}:{row['joint_top3_rate']:.2f}/{row['joint_best_rate']:.2f}"
            for row in ctx.get("top_families", [])[:4]
        )
        notes = "; ".join(ctx.get("decision_notes", []))
        lines.append(f"| `{ctx['decision']}` | `{ctx['context_key']}` | {ctx['context_support']} | {fam} | {notes} |")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path)
    parser.add_argument("--trace-glob", action="append", default=[])
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--tva-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--past-window", type=int, default=8)
    parser.add_argument("--future-window", type=int, default=20)
    parser.add_argument("--outcome-mode", default="cycle_v1")
    parser.add_argument("--phase-schema", default="simple_v1")
    parser.add_argument("--context-mode", default="broad_context_v1")
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--min-context-support", type=int, default=40)
    parser.add_argument("--max-records-per-context", type=int, default=96)
    parser.add_argument("--min-good-outcome-rate", type=float, default=0.75)
    parser.add_argument("--max-unstable-rate", type=float, default=0.25)
    parser.add_argument("--min-joint-top3-rate", type=float, default=0.60)
    parser.add_argument("--weak-joint-top3-rate", type=float, default=0.45)
    parser.add_argument("--min-joint-best-rate", type=float, default=0.15)
    parser.add_argument("--max-best-family-entropy", type=float, default=0.85)
    parser.add_argument("--max-rules", type=int, default=100)
    parser.add_argument("--max-contexts-report", type=int, default=100)
    parser.add_argument("--max-families-per-context-report", type=int, default=8)
    parser.add_argument("--max-examples-per-context", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote broad context counterfactual rules: {args.output}")
    print(
        f"contexts={payload['num_tested_contexts']} rules={payload['num_rules']} "
        f"decisions={payload['decision_counts']}"
    )


if __name__ == "__main__":
    main()
