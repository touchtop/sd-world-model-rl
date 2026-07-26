#!/usr/bin/env python3
"""Discriminant effect-space audit for Hopper ability-entry rules.

This script tests whether a slightly stronger mathematical direction estimator
can find stable effect-space rules without changing the rule form.  It compares:

1. mean_diff: mean(success delta_z) - mean(failure delta_z);
2. ridge_lda: regularized Fisher/LDA direction;
3. sign_consensus_ridge_lda: ridge LDA after retaining dimensions whose
   success-failure sign is consistent across training entry groups.

The audit remains offline and only reports stable/unstable candidates.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import mean_or_zero, write_json  # noqa: E402
from audit_hopper_ability_entry_effect_space import EFFECT_KEYS, dot, normalize  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater, recompute_effect_records, score_group  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import (  # noqa: E402
    dedupe_records,
    normalize_by_cycle,
    tier_of_group,
    cycle_of_group,
    usable_tier_cycle_groups,
)


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_effect_space_discriminant_audit.json"
)


def group_diff(records: list[dict[str, Any]]) -> list[float] | None:
    success = [rec["actual_delta_z"] for rec in records if rec["role"] == "entry_success"]
    failure = [rec["actual_delta_z"] for rec in records if rec["role"] == "matched_failure"]
    if not success or not failure:
        return None
    return [
        mean_or_zero(vec[i] for vec in success) - mean_or_zero(vec[i] for vec in failure)
        for i in range(len(EFFECT_KEYS))
    ]


def component_rows(raw: list[float]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": value}
        for key, value in sorted(zip(EFFECT_KEYS, raw), key=lambda item: abs(item[1]), reverse=True)
    ][:8]


def mean_diff_direction(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    raw = group_diff(records)
    if raw is None:
        return None
    return {
        "key": key,
        "method": "mean_diff",
        "raw_direction": raw,
        "unit_direction": normalize(raw),
        "top_components": component_rows(raw),
        "selected_dims": EFFECT_KEYS,
    }


def sign_consensus_mask(group_records: dict[str, list[dict[str, Any]]], min_agreement: float) -> list[bool]:
    diffs = []
    for vals in group_records.values():
        diff = group_diff(vals)
        if diff is not None:
            diffs.append(diff)
    if not diffs:
        return [True for _ in EFFECT_KEYS]
    mask: list[bool] = []
    for dim in range(len(EFFECT_KEYS)):
        signs = [1 if diff[dim] > 1e-9 else -1 if diff[dim] < -1e-9 else 0 for diff in diffs]
        nonzero = [s for s in signs if s != 0]
        if not nonzero:
            mask.append(False)
            continue
        pos = sum(1 for s in nonzero if s > 0) / len(nonzero)
        neg = sum(1 for s in nonzero if s < 0) / len(nonzero)
        mask.append(max(pos, neg) >= min_agreement)
    return mask


def ridge_lda_direction(
    records: list[dict[str, Any]],
    key: str,
    method: str,
    ridge: float,
    dim_mask: list[bool] | None,
) -> dict[str, Any] | None:
    success = [rec["actual_delta_z"] for rec in records if rec["role"] == "entry_success"]
    failure = [rec["actual_delta_z"] for rec in records if rec["role"] == "matched_failure"]
    if not success or not failure:
        return None
    x_pos = np.asarray(success, dtype=np.float64)
    x_neg = np.asarray(failure, dtype=np.float64)
    mu_pos = x_pos.mean(axis=0)
    mu_neg = x_neg.mean(axis=0)
    diff = mu_pos - mu_neg
    x = np.vstack([x_pos - mu_pos, x_neg - mu_neg])
    if len(x) <= 1:
        cov = np.eye(len(EFFECT_KEYS), dtype=np.float64)
    else:
        cov = np.dot(x.T, x) / max(1, len(x) - 2)
    scale = float(np.trace(cov) / max(1, cov.shape[0]))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    reg_cov = cov + float(ridge) * scale * np.eye(cov.shape[0], dtype=np.float64)
    if dim_mask is not None:
        mask_arr = np.asarray(dim_mask, dtype=bool)
        if not mask_arr.any():
            return None
        inactive = ~mask_arr
        diff = diff.copy()
        diff[inactive] = 0.0
        reg_cov = reg_cov.copy()
        for idx, keep in enumerate(mask_arr):
            if not keep:
                reg_cov[idx, :] = 0.0
                reg_cov[:, idx] = 0.0
                reg_cov[idx, idx] = 1.0
    try:
        weights = np.linalg.solve(reg_cov, diff)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(reg_cov, diff, rcond=None)[0]
    raw = [float(v) for v in weights.tolist()]
    if dim_mask is not None:
        raw = [v if dim_mask[i] else 0.0 for i, v in enumerate(raw)]
    selected = [key_name for key_name, keep in zip(EFFECT_KEYS, dim_mask or [True for _ in EFFECT_KEYS]) if keep]
    return {
        "key": key,
        "method": method,
        "raw_direction": raw,
        "unit_direction": normalize(raw),
        "top_components": component_rows(raw),
        "selected_dims": selected,
        "ridge": ridge,
    }


def train_direction(
    records: list[dict[str, Any]],
    grouped_train: dict[str, list[dict[str, Any]]],
    key: str,
    method: str,
    ridge: float,
    min_sign_agreement: float,
) -> dict[str, Any] | None:
    if method == "mean_diff":
        return mean_diff_direction(records, key)
    if method == "ridge_lda":
        return ridge_lda_direction(records, key, method, ridge, None)
    if method == "sign_consensus_ridge_lda":
        mask = sign_consensus_mask(grouped_train, min_sign_agreement)
        return ridge_lda_direction(records, key, method, ridge, mask)
    raise ValueError("unknown method: " + method)


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


def build_report(records: list[dict[str, Any]], methods: list[str], ridge: float, min_sign_agreement: float) -> dict[str, Any]:
    groups = usable_tier_cycle_groups(records)
    keys = sorted(groups)
    method_reports: dict[str, Any] = {}
    for method in methods:
        transfer_rows: list[dict[str, Any]] = []
        leave_one_rows: list[dict[str, Any]] = []
        cosine_rows: list[dict[str, Any]] = []
        group_directions: dict[str, dict[str, Any]] = {}

        for key in keys:
            direction = train_direction(
                groups[key],
                {key: groups[key]},
                key,
                method,
                ridge,
                min_sign_agreement,
            )
            if direction is not None:
                group_directions[key] = direction

        for source in keys:
            source_direction = group_directions.get(source)
            if source_direction is None:
                continue
            for target in keys:
                if source == target or tier_of_group(source) != tier_of_group(target):
                    continue
                scored = score_group(groups[target], source_direction["unit_direction"])
                if scored is None:
                    continue
                transfer_rows.append(
                    {
                        "method": method,
                        "tier": tier_of_group(source),
                        "source_key": source,
                        "target_key": target,
                        "source_cycle": cycle_of_group(source),
                        "target_cycle": cycle_of_group(target),
                        **scored,
                    }
                )

        for target in keys:
            tier = tier_of_group(target)
            grouped_train = {
                key: vals
                for key, vals in groups.items()
                if key != target and tier_of_group(key) == tier
            }
            train = [rec for vals in grouped_train.values() for rec in vals]
            direction = train_direction(
                train,
                grouped_train,
                "leave_one:" + target,
                method,
                ridge,
                min_sign_agreement,
            )
            if direction is None:
                continue
            scored = score_group(groups[target], direction["unit_direction"])
            if scored is not None:
                leave_one_rows.append(
                    {
                        "method": method,
                        "tier": tier,
                        "target_key": target,
                        **scored,
                        "top_components": direction["top_components"],
                        "selected_dims": direction["selected_dims"],
                    }
                )

        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if tier_of_group(a) != tier_of_group(b):
                    continue
                if a not in group_directions or b not in group_directions:
                    continue
                cosine_rows.append(
                    {
                        "method": method,
                        "tier": tier_of_group(a),
                        "key_a": a,
                        "key_b": b,
                        "cosine": cosine(group_directions[a]["unit_direction"], group_directions[b]["unit_direction"]),
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

        method_reports[method] = {
            "tier_reports": tier_reports,
            "transfer_rows": sorted(transfer_rows, key=lambda row: (row["tier"], row["auc_success_gt_failure"]), reverse=True),
            "leave_one_rows": sorted(leave_one_rows, key=lambda row: (row["tier"], row["auc_success_gt_failure"]), reverse=True),
            "direction_cosines": cosine_rows,
            "summary": {
                "all_transfer": summarize_rows(transfer_rows),
                "all_leave_one": summarize_rows(leave_one_rows),
                "stable_tiers": [tier for tier, item in tier_reports.items() if item["transfer_decision"]["decision"] == "stable"],
                "unstable_tiers": [tier for tier, item in tier_reports.items() if item["transfer_decision"]["decision"] != "stable"],
            },
            "group_directions": group_directions,
        }

    return {
        "usable_groups": keys,
        "methods": method_reports,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Effect-Space Discriminant Audit",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        f"Methods: `{payload['config']['methods']}`",
        "",
        "## Tier Decisions",
        "",
        "| Method | Tier | Groups | Decision | Transfer positive | Transfer AUC>=0.60 | Mean AUC | Mean cosine |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, report in payload["discriminant_stability"]["methods"].items():
        for tier, item in report["tier_reports"].items():
            decision = item["transfer_decision"]
            lines.append(
                f"| `{method}` | `{tier}` | {item['num_groups']} | `{decision['decision']}` | "
                f"{decision['positive_margin_rate']:.3f} | {decision['auc_ge_060_rate']:.3f} | "
                f"{decision['mean_auc']:.3f} | {decision['direction_cosine_mean']:.3f} |"
            )
    lines.extend(["", "## Leave-One Summary", "", "| Method | Tier | Positive | AUC>=0.60 | Mean AUC |", "| --- | --- | ---: | ---: | ---: |"])
    for method, report in payload["discriminant_stability"]["methods"].items():
        for tier, item in report["tier_reports"].items():
            stats = item["leave_one_summary"]
            lines.append(
                f"| `{method}` | `{tier}` | {stats['positive_margin_rate']:.3f} | "
                f"{stats['auc_ge_060_rate']:.3f} | {stats['mean_auc']:.3f} |"
            )
    lines.extend(["", "## Top Leave-One Rows", "", "| Method | Target | Margin | AUC | Components |", "| --- | --- | ---: | ---: | --- |"])
    rows = []
    for method, report in payload["discriminant_stability"]["methods"].items():
        rows.extend(report["leave_one_rows"])
    rows = sorted(rows, key=lambda row: row["auc_success_gt_failure"], reverse=True)[:25]
    for row in rows:
        components = ", ".join(f"{c['key']}={c['value']:.3f}" for c in row["top_components"][:4])
        lines.append(
            f"| `{row['method']}` | `{row['target_key']}` | {row['alignment_margin']:.4f} | "
            f"{row['auc_success_gt_failure']:.3f} | `{components}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "若更强判别方向仍不能在 transfer 和 leave-one 中通过严格稳定标准，则效果空间规则暂不进入在线 prior。"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ridge", type=float, default=0.25)
    parser.add_argument("--min-sign-agreement", type=float, default=0.75)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mean_diff", "ridge_lda", "sign_consensus_ridge_lda"],
        choices=["mean_diff", "ridge_lda", "sign_consensus_ridge_lda"],
    )
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.segments is None:
        from audit_hopper_ability_entry_effect_space import DEFAULT_SEGMENTS
        args.segments = DEFAULT_SEGMENTS
    if args.results_root is None:
        from audit_hopper_ability_entry_effect_space import DEFAULT_RESULTS_ROOT
        args.results_root = DEFAULT_RESULTS_ROOT

    raw_records, skipped = recompute_effect_records(args)
    deduped = dedupe_records(raw_records)
    normalized, normalization_stats = normalize_by_cycle(deduped)
    report = build_report(normalized, args.methods, args.ridge, args.min_sign_agreement)
    payload = {
        "analysis_family": "hopper_ability_entry_effect_space_discriminant_v1",
        "num_raw_records": len(raw_records),
        "num_records": len(normalized),
        "num_skipped": len(skipped),
        "num_usable_groups": len(report["usable_groups"]),
        "effect_keys": EFFECT_KEYS,
        "config": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "max_windows": args.max_windows,
            "normalization": "cycle_wise_robust_median_iqr",
            "tiers": {"low_entry": "threshold <= 10", "high_entry": "threshold >= 50"},
            "methods": args.methods,
            "ridge": args.ridge,
            "min_sign_agreement": args.min_sign_agreement,
            "stable_criterion": "all transfers positive margin, all transfers AUC>=0.60, mean direction cosine > 0",
        },
        "normalization_stats": normalization_stats,
        "discriminant_stability": report,
        "skipped": skipped,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote effect-space discriminant audit: {args.output}")
    print(
        f"records: {payload['num_records']} skipped: {payload['num_skipped']} "
        f"usable_groups: {payload['num_usable_groups']}"
    )


if __name__ == "__main__":
    main()
