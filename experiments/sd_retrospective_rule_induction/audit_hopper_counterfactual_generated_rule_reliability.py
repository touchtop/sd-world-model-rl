#!/usr/bin/env python3
"""Audit reliability of SD/TVA-generated broad Hopper rule candidates.

The generator creates candidate action families from high-quality traces.  This
script keeps those candidates fixed and re-evaluates them on another trace set
or another SD/TVA checkpoint.  It is intended for low-score contrast and
cross-cycle stability checks before any rule-library conversion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_json, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from generate_hopper_broad_context_counterfactual_rules import mean, outcome_rates  # noqa: E402
from validate_hopper_broad_context_counterfactuals import (  # noqa: E402
    evaluate_record,
    extractor_args,
    load_extractor_module,
    load_sd_model,
    load_value_net,
    score_target,
    select_records,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def run(args: argparse.Namespace) -> dict[str, Any]:
    rules_payload = load_json(args.rules)
    source_payload = load_json(args.source_library) if args.source_library else rules_payload
    params = dict(source_payload.get("parameters", source_payload.get("extractor_parameters", {})))
    trace_globs = args.trace_glob or list(params.get("trace_glob", []))
    if not trace_globs:
        raise ValueError("Provide --trace-glob or a source file containing trace_glob parameters.")

    extractor = load_extractor_module()
    records = extractor.extract_records(extractor_args(args, trace_globs, params))
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_context[str(record["context_key"])].append(record)

    model = load_sd_model(args.sd_checkpoint, args.device)
    value_net = load_value_net(args.tva_checkpoint, args.device)
    action_bank = build_action_bank(args.amplitudes)
    trace_cache: dict[str, list[dict[str, str]]] = {}

    reports: list[dict[str, Any]] = []
    for rule in rules_payload.get("rules", []):
        key = str(rule.get("rule_key"))
        target = str(rule.get("action_family"))
        context_records = by_context.get(key, [])
        selected = select_records(context_records, args.max_records_per_rule, args.seed)
        rec_reports: list[dict[str, Any]] = []
        best_counts: Counter[str] = Counter()
        top3_counts: Counter[str] = Counter()
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
            joint = score_target(result["joint_top_families"], target)
            best_counts[str(joint["best_family"])] += 1
            for item in result["joint_top_families"][:3]:
                top3_counts[str(item["family"])] += 1
            rec_reports.append(
                {
                    "trace_csv": trace_csv,
                    "episode": record.get("episode"),
                    "step": record.get("step"),
                    "future_outcome": record.get("future_outcome"),
                    "target_top3": joint["target_top3"],
                    "target_best": joint["target_best"],
                    "target_margin_to_best": joint["target_margin_to_best"],
                    "best_family": joint["best_family"],
                    "joint_top_families": result["joint_top_families"][:5],
                }
            )

        n = len(rec_reports)
        rates = outcome_rates(context_records)
        target_top3 = mean([1.0 if row["target_top3"] else 0.0 for row in rec_reports])
        target_best = mean([1.0 if row["target_best"] else 0.0 for row in rec_reports])
        margin = mean([float(row["target_margin_to_best"]) for row in rec_reports if row["target_margin_to_best"] is not None])
        if n <= 0:
            status = "no_overlap"
        elif target_top3 >= args.support_top3_threshold:
            status = "supported"
        elif target_top3 >= args.weak_top3_threshold:
            status = "weak"
        else:
            status = "not_supported"
        reports.append(
            {
                "rule_key": key,
                "action_family": target,
                "status": status,
                "context_support": len(context_records),
                "num_counterfactual_records": n,
                "context_support_ratio": len(context_records) / len(records) if records else 0.0,
                "target_joint_top3_rate": target_top3,
                "target_joint_best_rate": target_best,
                "mean_target_margin_to_best": margin,
                "source_joint_top3_rate": rule.get("joint_top3_rate"),
                "source_joint_best_rate": rule.get("joint_best_rate"),
                "best_family_counts": dict(best_counts.most_common()),
                "top3_family_counts": dict(top3_counts.most_common()),
                **rates,
                "examples": rec_reports[: args.max_records_report],
            }
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_family": "hopper_counterfactual_generated_rule_reliability_v1",
        "label": args.label,
        "rules": str(args.rules),
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
            "max_records_per_rule": args.max_records_per_rule,
            "support_top3_threshold": args.support_top3_threshold,
            "weak_top3_threshold": args.weak_top3_threshold,
            "seed": args.seed,
            "device": args.device,
        },
        "num_records": len(records),
        "num_contexts": len(by_context),
        "num_rules": len(reports),
        "status_counts": dict(Counter(row["status"] for row in reports).most_common()),
        "rules_report": reports,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Counterfactual Generated Rule Reliability Audit",
        "",
        f"- Label: `{payload['label']}`",
        f"- Records: `{payload['num_records']}`",
        f"- Contexts: `{payload['num_contexts']}`",
        f"- Rules: `{payload['num_rules']}`",
        f"- Status counts: `{payload['status_counts']}`",
        "",
        "| status | rule | action | ctx support | cf records | top3 | best | margin | good | unstable |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rules_report"]:
        good = row["cycle_continue_rate"] + row["recoverable_progress_rate"]
        lines.append(
            f"| `{row['status']}` | `{row['rule_key']}` | `{row['action_family']}` | "
            f"{row['context_support']} | {row['num_counterfactual_records']} | "
            f"{row['target_joint_top3_rate']:.3f} | {row['target_joint_best_rate']:.3f} | "
            f"{row['mean_target_margin_to_best']:.4f} | {good:.3f} | {row['unstable_rate']:.3f} |"
        )
    lines.extend(["", "## Best Families", ""])
    for row in payload["rules_report"]:
        lines.extend(
            [
                f"### `{row['rule_key']} -> {row['action_family']}`",
                "",
                f"- Status: `{row['status']}`",
                f"- Best family counts: `{row['best_family_counts']}`",
                f"- Top3 family counts: `{row['top3_family_counts']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rules", type=Path, required=True)
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
    parser.add_argument("--max-records-per-rule", type=int, default=96)
    parser.add_argument("--max-records-report", type=int, default=20)
    parser.add_argument("--support-top3-threshold", type=float, default=0.60)
    parser.add_argument("--weak-top3-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote generated-rule reliability audit: {args.output}")
    print(f"label={payload['label']} statuses={payload['status_counts']}")


if __name__ == "__main__":
    main()
