#!/usr/bin/env python3
"""Audit local action-effect advantage space for Hopper ability entries.

The effect-space delta_z used earlier mixes state baseline and action effect.
This script tests a more local mathematical object:

    A_z(s, a) = delta_z(s, a) - mean_{a' in probe_bank} delta_z(s, a')

It asks whether the actual action's local effect advantage is more stable
across cycles than raw delta_z.  The audit is offline and does not build priors.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import mean_or_zero, quantile, read_csv, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from audit_hopper_ability_entry_counterfactuals import checkpoint_paths, iter_window_records, load_sd_model, load_value_net  # noqa: E402
from audit_hopper_ability_entry_effect_space import DEFAULT_RESULTS_ROOT, DEFAULT_SEGMENTS, EFFECT_KEYS, dot, evaluate_record_effects, normalize  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater, score_group  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import dedupe_records, entry_tier  # noqa: E402
from audit_hopper_ability_entry_effect_space_discriminant import ridge_lda_direction, summarize_rows  # noqa: E402


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_local_advantage_space_audit.json"
)


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
        for dim, key in enumerate(EFFECT_KEYS):
            med, scale = robust_stats([float(rec["actual_delta_z"][dim]) for rec in vals])
            dim_stats.append({"key": key, "median": med, "scale": scale})
        stats[str(cycle)] = dim_stats
        for rec in vals:
            normed = [
                (float(rec["actual_delta_z"][dim]) - dim_stats[dim]["median"]) / dim_stats[dim]["scale"]
                for dim in range(len(EFFECT_KEYS))
            ]
            out.append({**rec, "raw_actual_delta_z": rec["actual_delta_z"], "actual_delta_z": normed})
    return out, stats


def select_records(records: list[dict[str, Any]], max_windows: int) -> list[dict[str, Any]]:
    if max_windows <= 0 or len(records) <= max_windows:
        return records
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[f"{entry_tier(rec)}|cycle:{int(rec['anchor_cycle']):03d}|role:{rec['role']}"].append(rec)
    selected: list[dict[str, Any]] = []
    per_group = max(1, max_windows // max(1, len(groups)))
    leftovers: list[dict[str, Any]] = []
    for key in sorted(groups):
        vals = sorted(groups[key], key=lambda item: (item["future_reward_sum"], item["future_rootx_delta"]), reverse=True)
        selected.extend(vals[:per_group])
        leftovers.extend(vals[per_group:])
    if len(selected) < max_windows:
        leftovers = sorted(leftovers, key=lambda item: item["future_reward_sum"], reverse=True)
        selected.extend(leftovers[: max_windows - len(selected)])
    return selected[:max_windows]


def recompute_advantage_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = __import__("common").load_json(args.segments)
    raw_records = select_records(iter_window_records(payload.get("segments", [])), args.max_windows)
    raw_records = dedupe_records(raw_records)
    action_bank = build_action_bank(args.amplitudes)
    model_cache: dict[int, torch.nn.Module] = {}
    tva_cache: dict[int, torch.nn.Module] = {}
    trace_cache: dict[str, list[dict[str, str]]] = {}
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in raw_records:
        cycle = int(item["anchor_cycle"])
        sd_path, tva_path = checkpoint_paths(args.results_root, cycle)
        if not sd_path.exists() or not tva_path.exists():
            skipped.append({**item, "reason": "missing_sd_or_tva_checkpoint"})
            continue
        if cycle not in model_cache:
            model_cache[cycle] = load_sd_model(sd_path, args.device)
        if cycle not in tva_cache:
            tva_cache[cycle] = load_value_net(tva_path, args.device)
        rows = trace_cache.get(item["trace_csv"])
        if rows is None:
            path = Path(item["trace_csv"])
            if not path.exists():
                skipped.append({**item, "reason": "missing_trace_csv"})
                continue
            rows = read_csv(path)
            trace_cache[item["trace_csv"]] = rows
        idx = int(item["anchor_index"])
        if idx < 0 or idx >= len(rows):
            skipped.append({**item, "reason": "anchor_index_out_of_range"})
            continue
        effects = evaluate_record_effects(
            model_cache[cycle],
            tva_cache[cycle],
            rows[idx],
            action_bank,
            args.horizon,
            args.device,
        )
        actual = next(action["delta_z"] for action in effects["actions"] if action["name"] == "actual")
        probes = [action["delta_z"] for action in effects["actions"] if action["name"] != "actual"]
        baseline = [mean_or_zero(vec[dim] for vec in probes) for dim in range(len(EFFECT_KEYS))]
        advantage = [a - b for a, b in zip(actual, baseline)]
        out.append({**item, "raw_delta_z": actual, "probe_mean_delta_z": baseline, "actual_delta_z": advantage})
    return out, skipped


def tier_cycle_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[f"{entry_tier(rec)}|cycle:{int(rec['anchor_cycle']):03d}"].append(rec)
    return {
        key: vals
        for key, vals in groups.items()
        if any(rec["role"] == "entry_success" for rec in vals)
        and any(rec["role"] == "matched_failure" for rec in vals)
    }


def tier_of(key: str) -> str:
    return key.split("|", 1)[0]


def cycle_of(key: str) -> int:
    for part in key.split("|"):
        if part.startswith("cycle:"):
            return int(part.split(":", 1)[1])
    return -1


def mean_diff_direction(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    success = [rec["actual_delta_z"] for rec in records if rec["role"] == "entry_success"]
    failure = [rec["actual_delta_z"] for rec in records if rec["role"] == "matched_failure"]
    if not success or not failure:
        return None
    raw = [
        mean_or_zero(vec[i] for vec in success) - mean_or_zero(vec[i] for vec in failure)
        for i in range(len(EFFECT_KEYS))
    ]
    return {"key": key, "raw_direction": raw, "unit_direction": normalize(raw), "top_components": top_components(raw)}


def top_components(raw: list[float]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": value}
        for key, value in sorted(zip(EFFECT_KEYS, raw), key=lambda item: abs(item[1]), reverse=True)[:8]
    ]


def train_direction(records: list[dict[str, Any]], key: str, method: str, ridge: float) -> dict[str, Any] | None:
    if method == "mean_diff":
        return mean_diff_direction(records, key)
    if method == "ridge_lda":
        return ridge_lda_direction(records, key, "ridge_lda", ridge, None)
    raise ValueError("unknown method: " + method)


def cosine(a: list[float], b: list[float]) -> float:
    return dot(a, b)


def build_report(records: list[dict[str, Any]], methods: list[str], ridge: float) -> dict[str, Any]:
    groups = tier_cycle_groups(records)
    keys = sorted(groups)
    method_reports: dict[str, Any] = {}
    for method in methods:
        directions: dict[str, dict[str, Any]] = {}
        for key, vals in groups.items():
            direction = train_direction(vals, key, method, ridge)
            if direction is not None:
                directions[key] = direction
        transfer_rows: list[dict[str, Any]] = []
        leave_one_rows: list[dict[str, Any]] = []
        cosine_rows: list[dict[str, Any]] = []
        for source in keys:
            if source not in directions:
                continue
            for target in keys:
                if source == target or tier_of(source) != tier_of(target):
                    continue
                scored = score_group(groups[target], directions[source]["unit_direction"])
                if scored is not None:
                    transfer_rows.append({"method": method, "tier": tier_of(source), "source_key": source, "target_key": target, **scored})
        for target in keys:
            train = [rec for key, vals in groups.items() if key != target and tier_of(key) == tier_of(target) for rec in vals]
            direction = train_direction(train, "leave_one:" + target, method, ridge)
            if direction is None:
                continue
            scored = score_group(groups[target], direction["unit_direction"])
            if scored is not None:
                leave_one_rows.append({"method": method, "tier": tier_of(target), "target_key": target, **scored, "top_components": direction["top_components"]})
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if tier_of(a) != tier_of(b) or a not in directions or b not in directions:
                    continue
                cosine_rows.append({"method": method, "tier": tier_of(a), "key_a": a, "key_b": b, "cosine": cosine(directions[a]["unit_direction"], directions[b]["unit_direction"])})
        tier_reports: dict[str, Any] = {}
        for tier in sorted({tier_of(key) for key in keys}):
            transfers = [row for row in transfer_rows if row["tier"] == tier]
            leaves = [row for row in leave_one_rows if row["tier"] == tier]
            cosines = [row for row in cosine_rows if row["tier"] == tier]
            stable = bool(transfers) and all(row["alignment_margin"] > 0 and row["auc_success_gt_failure"] >= 0.60 for row in transfers) and mean_or_zero(row["cosine"] for row in cosines) > 0
            tier_reports[tier] = {
                "decision": "stable" if stable else "unstable",
                "transfer": summarize_rows(transfers),
                "leave_one": summarize_rows(leaves),
                "mean_cosine": mean_or_zero(row["cosine"] for row in cosines),
                "groups": [key for key in keys if tier_of(key) == tier],
            }
        method_reports[method] = {
            "tier_reports": tier_reports,
            "transfer_rows": sorted(transfer_rows, key=lambda row: (row["tier"], row["auc_success_gt_failure"]), reverse=True),
            "leave_one_rows": sorted(leave_one_rows, key=lambda row: (row["tier"], row["auc_success_gt_failure"]), reverse=True),
            "direction_cosines": cosine_rows,
            "summary": {
                "all_transfer": summarize_rows(transfer_rows),
                "all_leave_one": summarize_rows(leave_one_rows),
                "stable_tiers": [tier for tier, item in tier_reports.items() if item["decision"] == "stable"],
                "unstable_tiers": [tier for tier, item in tier_reports.items() if item["decision"] != "stable"],
            },
        }
    return {"usable_groups": keys, "methods": method_reports}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "positive_margin_rate": mean_or_zero(1.0 if row["alignment_margin"] > 0.0 else 0.0 for row in rows),
        "auc_ge_060_rate": mean_or_zero(1.0 if row["auc_success_gt_failure"] >= 0.60 else 0.0 for row in rows),
        "mean_margin": mean_or_zero(row["alignment_margin"] for row in rows),
        "mean_auc": mean_or_zero(row["auc_success_gt_failure"] for row in rows),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Local Action-Effect Advantage Audit",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        "",
        "## Tier Decisions",
        "",
        "| Method | Tier | Decision | Transfer positive | Transfer AUC>=0.60 | Mean AUC | Mean cosine |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, report in payload["advantage_stability"]["methods"].items():
        for tier, item in report["tier_reports"].items():
            lines.append(
                f"| `{method}` | `{tier}` | `{item['decision']}` | "
                f"{item['transfer']['positive_margin_rate']:.3f} | {item['transfer']['auc_ge_060_rate']:.3f} | "
                f"{item['transfer']['mean_auc']:.3f} | {item['mean_cosine']:.3f} |"
            )
    lines.extend(["", "## Leave-One Rows", "", "| Method | Target | Margin | AUC | Components |", "| --- | --- | ---: | ---: | --- |"])
    rows = []
    for method, report in payload["advantage_stability"]["methods"].items():
        rows.extend(report["leave_one_rows"])
    for row in sorted(rows, key=lambda item: item["auc_success_gt_failure"], reverse=True)[:24]:
        comps = ", ".join(f"{c['key']}={c['value']:.3f}" for c in row["top_components"][:4])
        lines.append(f"| `{row['method']}` | `{row['target_key']}` | {row['alignment_margin']:.4f} | {row['auc_success_gt_failure']:.3f} | `{comps}` |")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "A stable result here would mean local action-effect advantage is a better rule space. "
        "If it is still unstable, effect-space rules should remain diagnostic rather than online priors."
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
    parser.add_argument("--methods", nargs="+", default=["mean_diff", "ridge_lda"], choices=["mean_diff", "ridge_lda"])
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--normalize", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, skipped = recompute_advantage_records(args)
    if args.normalize:
        records, normalization_stats = normalize_by_cycle(records)
    else:
        normalization_stats = {}
    report = build_report(records, args.methods, args.ridge)
    payload = {
        "analysis_family": "hopper_ability_entry_local_action_effect_advantage_v1",
        "num_records": len(records),
        "num_skipped": len(skipped),
        "effect_keys": EFFECT_KEYS,
        "config": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "methods": args.methods,
            "ridge": args.ridge,
            "normalize": args.normalize,
            "advantage_definition": "delta_z(s,a_actual)-mean_{probe actions} delta_z(s,a_probe)",
        },
        "normalization_stats": normalization_stats,
        "advantage_stability": report,
        "skipped": skipped,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote local advantage audit: {args.output}")
    print(f"records: {payload['num_records']} skipped: {payload['num_skipped']}")


if __name__ == "__main__":
    main()
