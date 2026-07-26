#!/usr/bin/env python3
"""Minimal stable-effect audit for Hopper ability-entry directions.

This is a deliberately conservative follow-up to the effect-space prototype.
It only adds two stabilizers:

1. cycle-wise robust normalization of delta_z;
2. a two-bin entry tier: low_entry (R<=10) and high_entry (R>=50).

The output is a stable/unstable audit.  It does not build online priors.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import mean_or_zero, quantile, write_json  # noqa: E402
from audit_hopper_ability_entry_effect_space import EFFECT_KEYS, dot, normalize  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import learn_direction, recompute_effect_records, score_group  # noqa: E402


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_minimal_stable_effect_audit.json"
)


def entry_tier(record: dict[str, Any]) -> str:
    threshold = float(record["threshold"])
    if threshold <= 10.0:
        return "low_entry"
    return "high_entry"


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = (
            record.get("trace_csv"),
            record.get("anchor_step"),
            record.get("role"),
            record.get("seed"),
            record.get("episode"),
            record.get("anchor_cycle"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def robust_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    med = quantile(values, 0.50)
    q25 = quantile(values, 0.25)
    q75 = quantile(values, 0.75)
    scale = q75 - q25
    if abs(scale) < 1e-6:
        mean_abs = mean_or_zero(abs(v - med) for v in values)
        scale = mean_abs if mean_abs > 1e-6 else 1.0
    return med, scale


def normalize_by_cycle(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cycle[int(record["anchor_cycle"])].append(record)

    stats: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for cycle, vals in sorted(by_cycle.items()):
        dim_stats = []
        for dim, name in enumerate(EFFECT_KEYS):
            med, scale = robust_stats([float(rec["actual_delta_z"][dim]) for rec in vals])
            dim_stats.append({"key": name, "median": med, "scale": scale})
        stats[str(cycle)] = dim_stats
        for rec in vals:
            normed = [
                (float(rec["actual_delta_z"][dim]) - dim_stats[dim]["median"]) / dim_stats[dim]["scale"]
                for dim in range(len(EFFECT_KEYS))
            ]
            out.append({**rec, "raw_actual_delta_z": rec["actual_delta_z"], "actual_delta_z": normed})
    return out, stats


def usable_tier_cycle_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        tier = entry_tier(record)
        cycle = int(record["anchor_cycle"])
        groups[f"{tier}|cycle:{cycle:03d}"].append({**record, "entry_tier": tier})
    return {
        key: vals
        for key, vals in groups.items()
        if any(rec["role"] == "entry_success" for rec in vals)
        and any(rec["role"] == "matched_failure" for rec in vals)
    }


def tier_of_group(key: str) -> str:
    return key.split("|", 1)[0]


def cycle_of_group(key: str) -> int:
    for part in key.split("|"):
        if part.startswith("cycle:"):
            return int(part.split(":", 1)[1])
    return -1


def cosine(xs: list[float], ys: list[float]) -> float:
    return dot(xs, ys)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "positive_margin_rate": mean_or_zero(1.0 if row["alignment_margin"] > 0.0 else 0.0 for row in rows),
        "auc_ge_060_rate": mean_or_zero(1.0 if row["auc_success_gt_failure"] >= 0.60 else 0.0 for row in rows),
        "mean_alignment_margin": mean_or_zero(row["alignment_margin"] for row in rows),
        "mean_auc": mean_or_zero(row["auc_success_gt_failure"] for row in rows),
    }


def tier_decision(rows: list[dict[str, Any]], cosines: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_rows(rows)
    cosine_mean = mean_or_zero(row["cosine"] for row in cosines)
    stable = (
        summary["count"] > 0
        and summary["positive_margin_rate"] >= 1.0
        and summary["auc_ge_060_rate"] >= 1.0
        and cosine_mean > 0.0
    )
    return {
        **summary,
        "direction_cosine_mean": cosine_mean,
        "decision": "stable" if stable else "unstable",
        "criterion": "all transfers positive margin, all transfers AUC>=0.60, mean direction cosine > 0",
    }


def build_minimal_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = usable_tier_cycle_groups(records)
    directions: dict[str, dict[str, Any]] = {}
    for key, vals in groups.items():
        direction = learn_direction(vals, key)
        if direction is not None:
            directions[key] = direction

    transfer_rows: list[dict[str, Any]] = []
    keys = sorted(groups)
    for source in keys:
        for target in keys:
            if source == target or tier_of_group(source) != tier_of_group(target):
                continue
            scored = score_group(groups[target], directions[source]["unit_direction"])
            if scored is None:
                continue
            transfer_rows.append(
                {
                    "source_key": source,
                    "target_key": target,
                    "tier": tier_of_group(source),
                    "source_cycle": cycle_of_group(source),
                    "target_cycle": cycle_of_group(target),
                    **scored,
                }
            )

    leave_one_rows: list[dict[str, Any]] = []
    for target in keys:
        target_tier = tier_of_group(target)
        train = [
            rec
            for key, vals in groups.items()
            if key != target and tier_of_group(key) == target_tier
            for rec in vals
        ]
        direction = learn_direction(train, f"leave_one:{target}")
        if direction is None:
            continue
        scored = score_group(groups[target], direction["unit_direction"])
        if scored is not None:
            leave_one_rows.append({"target_key": target, "tier": target_tier, **scored, "top_components": direction["top_components"]})

    cosine_rows: list[dict[str, Any]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if tier_of_group(a) != tier_of_group(b):
                continue
            cosine_rows.append(
                {
                    "key_a": a,
                    "key_b": b,
                    "tier": tier_of_group(a),
                    "cosine": cosine(directions[a]["unit_direction"], directions[b]["unit_direction"]),
                }
            )

    tiers = sorted({tier_of_group(key) for key in keys})
    tier_reports: dict[str, Any] = {}
    for tier in tiers:
        tier_transfers = [row for row in transfer_rows if row["tier"] == tier]
        tier_leave_one = [row for row in leave_one_rows if row["tier"] == tier]
        tier_cosines = [row for row in cosine_rows if row["tier"] == tier]
        tier_reports[tier] = {
            "transfer_decision": tier_decision(tier_transfers, tier_cosines),
            "leave_one_summary": summarize_rows(tier_leave_one),
            "num_groups": sum(1 for key in keys if tier_of_group(key) == tier),
            "groups": [key for key in keys if tier_of_group(key) == tier],
        }

    return {
        "usable_groups": keys,
        "directions": directions,
        "transfer_rows": sorted(transfer_rows, key=lambda row: (row["tier"], row["auc_success_gt_failure"]), reverse=True),
        "leave_one_rows": sorted(leave_one_rows, key=lambda row: (row["tier"], row["auc_success_gt_failure"]), reverse=True),
        "direction_cosines": cosine_rows,
        "tier_reports": tier_reports,
        "summary": {
            "all_transfer": summarize_rows(transfer_rows),
            "all_leave_one": summarize_rows(leave_one_rows),
            "stable_tiers": [tier for tier, item in tier_reports.items() if item["transfer_decision"]["decision"] == "stable"],
            "unstable_tiers": [tier for tier, item in tier_reports.items() if item["transfer_decision"]["decision"] != "stable"],
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Minimal Stable-Effect Audit",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        f"Usable tier-cycle groups: `{payload['num_usable_groups']}`",
        f"Normalization: `{payload['config']['normalization']}`",
        "",
        "## Tier Decisions",
        "",
        "| Tier | Groups | Decision | Transfer positive | Transfer AUC>=0.60 | Mean AUC | Mean cosine |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for tier, item in payload["minimal_stability"]["tier_reports"].items():
        decision = item["transfer_decision"]
        lines.append(
            f"| `{tier}` | {item['num_groups']} | `{decision['decision']}` | "
            f"{decision['positive_margin_rate']:.3f} | {decision['auc_ge_060_rate']:.3f} | "
            f"{decision['mean_auc']:.3f} | {decision['direction_cosine_mean']:.3f} |"
        )
    lines.extend(["", "## Transfers", "", "| Source | Target | Margin | AUC |", "| --- | --- | ---: | ---: |"])
    for row in payload["minimal_stability"]["transfer_rows"]:
        lines.append(
            f"| `{row['source_key']}` | `{row['target_key']}` | "
            f"{row['alignment_margin']:.4f} | {row['auc_success_gt_failure']:.3f} |"
        )
    lines.extend(["", "## Leave-One", "", "| Target | Margin | AUC | Top components |", "| --- | ---: | ---: | --- |"])
    for row in payload["minimal_stability"]["leave_one_rows"]:
        components = ", ".join(f"{c['key']}={c['value']:.3f}" for c in row["top_components"][:4])
        lines.append(
            f"| `{row['target_key']}` | {row['alignment_margin']:.4f} | "
            f"{row['auc_success_gt_failure']:.3f} | `{components}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "该报告采用严格稳定标准。若 tier 被标记为 unstable，则不生成效果空间 prior，相关问题交还给训练或后续更小范围的诊断。"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Reuse the stability script's argument names while keeping this script small.
    if args.segments is None:
        from audit_hopper_ability_entry_effect_space import DEFAULT_SEGMENTS
        args.segments = DEFAULT_SEGMENTS
    if args.results_root is None:
        from audit_hopper_ability_entry_effect_space import DEFAULT_RESULTS_ROOT
        args.results_root = DEFAULT_RESULTS_ROOT

    raw_records, skipped = recompute_effect_records(args)
    deduped = dedupe_records(raw_records)
    normalized, normalization_stats = normalize_by_cycle(deduped)
    minimal = build_minimal_report(normalized)
    payload = {
        "analysis_family": "hopper_ability_entry_minimal_stable_effect_v1",
        "num_raw_records": len(raw_records),
        "num_records": len(normalized),
        "num_skipped": len(skipped),
        "num_usable_groups": len(minimal["usable_groups"]),
        "effect_keys": EFFECT_KEYS,
        "config": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "max_windows": args.max_windows,
            "normalization": "cycle_wise_robust_median_iqr",
            "tiers": {"low_entry": "threshold <= 10", "high_entry": "threshold >= 50"},
            "stable_criterion": "all transfers positive margin, all transfers AUC>=0.60, mean direction cosine > 0",
        },
        "normalization_stats": normalization_stats,
        "minimal_stability": minimal,
        "skipped": skipped,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote minimal stable-effect audit: {args.output}")
    print(
        f"records: {payload['num_records']} skipped: {payload['num_skipped']} "
        f"usable_groups: {payload['num_usable_groups']}"
    )


if __name__ == "__main__":
    main()
