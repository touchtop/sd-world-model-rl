#!/usr/bin/env python3
"""Check cross-cycle stability of Hopper ability-entry effect directions.

This audit is a transfer test for the effect-space rule prototype.  It learns a
direction from one entry group, then evaluates whether that direction still
separates entry-success windows from matched failures in another group.  It
does not modify online rules or training checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, mean_or_zero, read_csv, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from audit_hopper_ability_entry_counterfactuals import checkpoint_paths, iter_window_records, load_sd_model, load_value_net  # noqa: E402
from audit_hopper_ability_entry_effect_space import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_EFFECT_AUDIT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SEGMENTS,
    EFFECT_KEYS,
    auc_positive_greater,
    dot,
    evaluate_record_effects,
    group_key,
    normalize,
    select_effect_records,
)


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_effect_space_stability.json"
)


def learn_direction(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    success = [rec["actual_delta_z"] for rec in records if rec["role"] == "entry_success"]
    failure = [rec["actual_delta_z"] for rec in records if rec["role"] == "matched_failure"]
    if not success or not failure:
        return None
    mean_success = [mean_or_zero(vec[i] for vec in success) for i in range(len(EFFECT_KEYS))]
    mean_failure = [mean_or_zero(vec[i] for vec in failure) for i in range(len(EFFECT_KEYS))]
    raw = [a - b for a, b in zip(mean_success, mean_failure)]
    components = [
        {"key": name, "value": value}
        for name, value in sorted(zip(EFFECT_KEYS, raw), key=lambda item: abs(item[1]), reverse=True)
    ]
    return {
        "key": key,
        "success_count": len(success),
        "failure_count": len(failure),
        "raw_direction": raw,
        "unit_direction": normalize(raw),
        "top_components": components[:8],
    }


def threshold_of_key(key: str) -> str:
    for part in key.split("|"):
        if part.startswith("threshold:"):
            return part.split(":", 1)[1]
    return "unknown"


def cycle_of_key(key: str) -> int:
    for part in key.split("|"):
        if part.startswith("cycle:"):
            return int(part.split(":", 1)[1])
    return -1


def score_group(records: list[dict[str, Any]], direction: list[float]) -> dict[str, Any] | None:
    success = [dot(rec["actual_delta_z"], direction) for rec in records if rec["role"] == "entry_success"]
    failure = [dot(rec["actual_delta_z"], direction) for rec in records if rec["role"] == "matched_failure"]
    if not success or not failure:
        return None
    return {
        "success_count": len(success),
        "failure_count": len(failure),
        "success_mean_alignment": mean_or_zero(success),
        "failure_mean_alignment": mean_or_zero(failure),
        "alignment_margin": mean_or_zero(success) - mean_or_zero(failure),
        "auc_success_gt_failure": auc_positive_greater(success, failure),
    }


def cosine(d1: list[float], d2: list[float]) -> float:
    return dot(d1, d2)


def recompute_effect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(args.segments)
    raw_records = select_effect_records(iter_window_records(payload.get("segments", [])), args.max_windows)
    action_bank = build_action_bank(args.amplitudes)
    model_cache: dict[int, torch.nn.Module] = {}
    tva_cache: dict[int, torch.nn.Module] = {}
    trace_cache: dict[str, list[dict[str, str]]] = {}
    effect_records: list[dict[str, Any]] = []
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
        anchor_index = int(item["anchor_index"])
        if anchor_index < 0 or anchor_index >= len(rows):
            skipped.append({**item, "reason": "anchor_index_out_of_range"})
            continue
        effects = evaluate_record_effects(
            model_cache[cycle],
            tva_cache[cycle],
            rows[anchor_index],
            action_bank,
            args.horizon,
            args.device,
        )
        actual_delta = next(action["delta_z"] for action in effects["actions"] if action["name"] == "actual")
        effect_records.append({**item, "actual_delta_z": actual_delta})
    return effect_records, skipped


def build_stability_report(effect_records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in effect_records:
        groups[group_key(rec)].append(rec)

    usable_groups = {
        key: vals
        for key, vals in groups.items()
        if any(rec["role"] == "entry_success" for rec in vals)
        and any(rec["role"] == "matched_failure" for rec in vals)
    }
    directions: dict[str, dict[str, Any]] = {}
    for key, vals in usable_groups.items():
        direction = learn_direction(vals, key)
        if direction is not None:
            directions[key] = direction
    global_direction = learn_direction(effect_records, "global")
    if global_direction is not None:
        directions["global"] = global_direction

    in_sample: dict[str, Any] = {}
    for key, direction in directions.items():
        if key == "global":
            target = effect_records
        else:
            target = usable_groups.get(key, [])
        scored = score_group(target, direction["unit_direction"])
        if scored is not None:
            in_sample[key] = scored

    transfer_rows: list[dict[str, Any]] = []
    group_keys = sorted(usable_groups)
    for source_key in group_keys:
        source_direction = directions[source_key]["unit_direction"]
        for target_key in group_keys:
            if source_key == target_key:
                continue
            scored = score_group(usable_groups[target_key], source_direction)
            if scored is None:
                continue
            relation = "same_threshold" if threshold_of_key(source_key) == threshold_of_key(target_key) else "cross_threshold"
            transfer_rows.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "relation": relation,
                    "source_cycle": cycle_of_key(source_key),
                    "target_cycle": cycle_of_key(target_key),
                    "source_threshold": threshold_of_key(source_key),
                    "target_threshold": threshold_of_key(target_key),
                    **scored,
                }
            )

    leave_one_rows: list[dict[str, Any]] = []
    for target_key in group_keys:
        train = [rec for key, vals in usable_groups.items() if key != target_key for rec in vals]
        direction = learn_direction(train, f"leave_one:{target_key}")
        if direction is None:
            continue
        scored = score_group(usable_groups[target_key], direction["unit_direction"])
        if scored is not None:
            leave_one_rows.append({"target_key": target_key, **scored, "top_components": direction["top_components"]})

    same_threshold_leave_one_rows: list[dict[str, Any]] = []
    for target_key in group_keys:
        target_threshold = threshold_of_key(target_key)
        train = [
            rec
            for key, vals in usable_groups.items()
            if key != target_key and threshold_of_key(key) == target_threshold
            for rec in vals
        ]
        direction = learn_direction(train, f"same_threshold_leave_one:{target_key}")
        if direction is None:
            continue
        scored = score_group(usable_groups[target_key], direction["unit_direction"])
        if scored is not None:
            same_threshold_leave_one_rows.append(
                {"target_key": target_key, **scored, "top_components": direction["top_components"]}
            )

    cosine_rows: list[dict[str, Any]] = []
    for i, key_a in enumerate(group_keys):
        for key_b in group_keys[i + 1:]:
            relation = "same_threshold" if threshold_of_key(key_a) == threshold_of_key(key_b) else "cross_threshold"
            cosine_rows.append(
                {
                    "key_a": key_a,
                    "key_b": key_b,
                    "relation": relation,
                    "cosine": cosine(directions[key_a]["unit_direction"], directions[key_b]["unit_direction"]),
                }
            )

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "positive_margin_rate": mean_or_zero(1.0 if row["alignment_margin"] > 0.0 else 0.0 for row in rows),
            "auc_gt_055_rate": mean_or_zero(1.0 if row["auc_success_gt_failure"] >= 0.55 else 0.0 for row in rows),
            "mean_alignment_margin": mean_or_zero(row["alignment_margin"] for row in rows),
            "mean_auc": mean_or_zero(row["auc_success_gt_failure"] for row in rows),
        }

    same_threshold_transfer = [row for row in transfer_rows if row["relation"] == "same_threshold"]
    cross_threshold_transfer = [row for row in transfer_rows if row["relation"] == "cross_threshold"]
    same_threshold_cosines = [row for row in cosine_rows if row["relation"] == "same_threshold"]
    cross_threshold_cosines = [row for row in cosine_rows if row["relation"] == "cross_threshold"]
    return {
        "usable_groups": sorted(group_keys),
        "directions": directions,
        "in_sample": in_sample,
        "transfer_rows": sorted(
            transfer_rows,
            key=lambda row: (row["relation"] == "same_threshold", row["auc_success_gt_failure"], row["alignment_margin"]),
            reverse=True,
        ),
        "leave_one_rows": sorted(leave_one_rows, key=lambda row: row["auc_success_gt_failure"], reverse=True),
        "same_threshold_leave_one_rows": sorted(
            same_threshold_leave_one_rows,
            key=lambda row: row["auc_success_gt_failure"],
            reverse=True,
        ),
        "direction_cosines": sorted(cosine_rows, key=lambda row: row["cosine"], reverse=True),
        "summary": {
            "in_sample": summarize_rows(list(in_sample.values())),
            "all_transfer": summarize_rows(transfer_rows),
            "same_threshold_transfer": summarize_rows(same_threshold_transfer),
            "cross_threshold_transfer": summarize_rows(cross_threshold_transfer),
            "leave_one_global": summarize_rows(leave_one_rows),
            "same_threshold_leave_one": summarize_rows(same_threshold_leave_one_rows),
            "same_threshold_direction_cosine_mean": mean_or_zero(row["cosine"] for row in same_threshold_cosines),
            "cross_threshold_direction_cosine_mean": mean_or_zero(row["cosine"] for row in cross_threshold_cosines),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Ability Entry Effect-Space Stability",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        f"Usable groups: `{payload['num_usable_groups']}`",
        "",
        "## Summary",
        "",
        "| Test | Count | Positive margin | AUC >= 0.55 | Mean margin | Mean AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in payload["stability"]["summary"].items():
        if not isinstance(stats, dict):
            continue
        lines.append(
            "| "
            f"`{key}` | {stats['count']} | {stats['positive_margin_rate']:.3f} | "
            f"{stats['auc_gt_055_rate']:.3f} | {stats['mean_alignment_margin']:.4f} | {stats['mean_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Same-threshold direction cosine mean: `{payload['stability']['summary']['same_threshold_direction_cosine_mean']:.4f}`",
            f"Cross-threshold direction cosine mean: `{payload['stability']['summary']['cross_threshold_direction_cosine_mean']:.4f}`",
            "",
            "## Same-Threshold Transfers",
            "",
            "| Source | Target | Margin | AUC |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in [r for r in payload["stability"]["transfer_rows"] if r["relation"] == "same_threshold"]:
        lines.append(
            f"| `{row['source_key']}` | `{row['target_key']}` | "
            f"{row['alignment_margin']:.4f} | {row['auc_success_gt_failure']:.3f} |"
        )
    lines.extend(["", "## Leave-One-Group", "", "| Target | Margin | AUC | Top components |", "| --- | ---: | ---: | --- |"])
    for row in payload["stability"]["leave_one_rows"]:
        components = ", ".join(f"{c['key']}={c['value']:.3f}" for c in row["top_components"][:4])
        lines.append(
            f"| `{row['target_key']}` | {row['alignment_margin']:.4f} | "
            f"{row['auc_success_gt_failure']:.3f} | `{components}` |"
        )
    lines.extend(["", "## Direction Cosines", "", "| A | B | Relation | Cosine |", "| --- | --- | --- | ---: |"])
    for row in payload["stability"]["direction_cosines"]:
        lines.append(f"| `{row['key_a']}` | `{row['key_b']}` | `{row['relation']}` | {row['cosine']:.4f} |")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "该报告用于判断效果方向是否跨周期稳定。若 same-threshold transfer 或 leave-one-group "
        "保持正 margin 和 AUC>0.55，可作为 shadow 版效果方向规则的候选；否则应继续改进效果空间或上下文划分。"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reference-effect-audit", type=Path, default=DEFAULT_EFFECT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effect_records, skipped = recompute_effect_records(args)
    stability = build_stability_report(effect_records)
    payload = {
        "analysis_family": "hopper_ability_entry_effect_space_stability_v1",
        "source_segments": str(args.segments),
        "reference_effect_audit": str(args.reference_effect_audit),
        "results_root": str(args.results_root),
        "effect_keys": EFFECT_KEYS,
        "num_records": len(effect_records),
        "num_skipped": len(skipped),
        "num_usable_groups": len(stability["usable_groups"]),
        "config": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "max_windows": args.max_windows,
            "device": args.device,
        },
        "stability": stability,
        "skipped": skipped,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote effect-space stability audit: {args.output}")
    print(
        f"records: {payload['num_records']} skipped: {payload['num_skipped']} "
        f"usable_groups: {payload['num_usable_groups']}"
    )


if __name__ == "__main__":
    main()
