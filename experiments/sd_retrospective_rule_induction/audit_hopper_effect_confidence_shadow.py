#!/usr/bin/env python3
"""Shadow confidence audit for Hopper effect-space rules.

This script tests whether effect-space rules can use the same maintenance idea
as action-family rules:

    context_key -> prefer_effect_direction
    confidence_t -> confidence_{t+1}

The audit is offline and read-only.  It recomputes SD/TVA effect vectors on
strict ability-entry windows, forms a cumulative success-minus-failure effect
direction per context, and replays cycle evidence into a shadow confidence
value.  Early SD/TVA evidence is down-weighted by a maturity schedule because
the models are not yet fully trained.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import mean_or_zero, quantile, read_csv, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from audit_hopper_ability_entry_counterfactuals import checkpoint_paths, load_sd_model, load_value_net  # noqa: E402
from audit_hopper_ability_entry_effect_space import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    EFFECT_KEYS,
    dot,
    evaluate_record_effects,
    normalize,
)
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import entry_tier  # noqa: E402
from audit_hopper_phi_regret_potential import entropy_norm  # noqa: E402
from audit_hopper_strict_context_variants import DEFAULT_BASE_DIR, context_variants, flatten_segments  # noqa: E402


DEFAULT_SEGMENTS = DEFAULT_BASE_DIR / "entry_candidate_segments_strict_no_pre_entry.json"
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "entry_effect_confidence_shadow_strict_no_pre_entry.json"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def context_key(record: dict[str, Any], variant: str, include_tier: bool) -> str:
    variants = context_variants()
    if variant not in variants:
        raise ValueError(f"unknown context variant: {variant}")
    key = variants[variant](record)
    if include_tier:
        key = f"{entry_tier(record)}::{key}"
    return key


def top_components(raw_direction: list[float], k: int = 8) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": value}
        for key, value in sorted(zip(EFFECT_KEYS, raw_direction), key=lambda item: abs(item[1]), reverse=True)[:k]
    ]


def robust_scale(values: list[float]) -> float:
    if not values:
        return 1.0
    q25 = quantile(values, 0.25)
    q75 = quantile(values, 0.75)
    scale = q75 - q25
    if abs(scale) < 1e-6:
        mean_abs = mean_or_zero(abs(x - mean_or_zero(values)) for x in values)
        scale = mean_abs
    return scale if abs(scale) >= 1e-6 else 1.0


def learn_direction(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    success = [rec["actual_delta_z"] for rec in records if rec["role"] == "entry_success"]
    failure = [rec["actual_delta_z"] for rec in records if rec["role"] == "matched_failure"]
    if not success or not failure:
        return None
    raw = [
        mean_or_zero(vec[i] for vec in success) - mean_or_zero(vec[i] for vec in failure)
        for i in range(len(EFFECT_KEYS))
    ]
    return {
        "raw_direction": raw,
        "unit_direction": normalize(raw),
        "top_components": top_components(raw),
        "success_count": len(success),
        "failure_count": len(failure),
    }


def rank_effect_families(actions: list[dict[str, Any]], direction: list[float]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for action in actions:
        family = str(action["family"])
        score = dot(action["delta_z"], direction)
        current = grouped.get(family)
        if current is None or score > current["alignment"]:
            grouped[family] = {
                "family": family,
                "name": action["name"],
                "alignment": score,
                "delta_z": action["delta_z"],
            }
    return sorted(grouped.values(), key=lambda item: item["alignment"], reverse=True)


def maturity_weight(cycle: int, min_cycle: int, full_cycle: int, floor: float) -> float:
    if full_cycle <= min_cycle:
        return 1.0
    ratio = (cycle - min_cycle) / float(full_cycle - min_cycle)
    return max(floor, min(1.0, ratio))


def evidence_from_cycle(records: list[dict[str, Any]], direction: list[float]) -> dict[str, Any]:
    rows = []
    for rec in records:
        alignment = dot(rec["actual_delta_z"], direction)
        ranked = rank_effect_families(rec["actions"], direction)
        rows.append(
            {
                **rec,
                "actual_alignment": alignment,
                "best_effect_family": ranked[0]["family"] if ranked else "none",
                "best_effect_alignment": ranked[0]["alignment"] if ranked else 0.0,
                "effect_top_families": ranked[:5],
            }
        )
    success = [row for row in rows if row["role"] == "entry_success"]
    failure = [row for row in rows if row["role"] == "matched_failure"]
    all_alignments = [row["actual_alignment"] for row in rows]
    scale = robust_scale(all_alignments)
    if success and failure:
        s_vals = [row["actual_alignment"] for row in success]
        f_vals = [row["actual_alignment"] for row in failure]
        margin = mean_or_zero(s_vals) - mean_or_zero(f_vals)
        auc = auc_positive_greater(s_vals, f_vals)
        margin_score = sigmoid(margin / scale)
        evidence_score = 0.5 * auc + 0.5 * margin_score
        evidence_strength = 1.0
        evidence_kind = "contrast"
    elif success:
        margin = mean_or_zero(row["actual_alignment"] for row in success)
        auc = 0.0
        evidence_score = sigmoid(margin / scale)
        evidence_strength = 0.50
        evidence_kind = "success_only"
    elif failure:
        margin = -mean_or_zero(row["actual_alignment"] for row in failure)
        auc = 0.0
        evidence_score = sigmoid(margin / scale)
        evidence_strength = 0.50
        evidence_kind = "failure_only"
    else:
        margin = 0.0
        auc = 0.0
        evidence_score = 0.5
        evidence_strength = 0.0
        evidence_kind = "empty"

    success_best = Counter(row["best_effect_family"] for row in success)
    all_best = Counter(row["best_effect_family"] for row in rows)
    return {
        "rows": rows,
        "count": len(rows),
        "success_count": len(success),
        "failure_count": len(failure),
        "evidence_kind": evidence_kind,
        "evidence_score": evidence_score,
        "evidence_strength": evidence_strength,
        "alignment_margin_or_single_role_score": margin,
        "auc_success_gt_failure": auc,
        "success_mean_alignment": mean_or_zero(row["actual_alignment"] for row in success),
        "failure_mean_alignment": mean_or_zero(row["actual_alignment"] for row in failure),
        "success_best_family_counts": dict(success_best.most_common()),
        "success_best_family_entropy": entropy_norm(success_best),
        "all_best_family_counts": dict(all_best.most_common()),
        "future_reward_mean": mean_or_zero(row["future_reward_sum"] for row in rows),
        "future_rootx_mean": mean_or_zero(row["future_rootx_delta"] for row in rows),
    }


def update_confidence(confidence: float, evidence_score: float, evidence_strength: float, maturity: float, lr: float, decay: float) -> tuple[float, float]:
    centered = (confidence - 0.5) * decay
    delta = lr * maturity * evidence_strength * (evidence_score - 0.5)
    new_conf = max(0.0, min(1.0, 0.5 + centered + delta))
    return new_conf, delta


def recompute_effect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = flatten_segments(args.segments)
    if args.max_windows > 0 and len(records) > args.max_windows:
        records = sorted(
            records,
            key=lambda item: (
                item["role"] == "entry_success",
                float(item["threshold"]),
                safe_float(item.get("future_reward_sum")),
            ),
            reverse=True,
        )[: args.max_windows]
    action_bank = build_action_bank(args.amplitudes)
    model_cache: dict[int, torch.nn.Module] = {}
    tva_cache: dict[int, torch.nn.Module] = {}
    trace_cache: dict[str, list[dict[str, str]]] = {}
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in records:
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
        out.append(
            {
                **item,
                "context_key": context_key(item, args.context_variant, args.include_tier),
                "actual_delta_z": actual_delta,
                "actions": effects["actions"],
            }
        )
    return out, skipped


def build_shadow_report(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    cycles = sorted({int(rec["anchor_cycle"]) for rec in records})
    min_cycle = min(cycles) if cycles else 0
    full_cycle = int(args.maturity_full_cycle) if args.maturity_full_cycle > 0 else (max(cycles) if cycles else min_cycle)
    by_context_cycle: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_context_cycle[str(rec["context_key"])][int(rec["anchor_cycle"])].append(rec)

    rules: list[dict[str, Any]] = []
    for ctx, by_cycle in sorted(by_context_cycle.items()):
        total = [rec for vals in by_cycle.values() for rec in vals]
        role_counts = Counter(rec["role"] for rec in total)
        if len(total) < args.min_context_records:
            continue
        if role_counts.get("entry_success", 0) < args.min_role_records or role_counts.get("matched_failure", 0) < args.min_role_records:
            continue
        confidence = float(args.initial_confidence)
        cumulative: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        final_direction: dict[str, Any] | None = None
        for cycle in sorted(by_cycle):
            cumulative.extend(by_cycle[cycle])
            direction = learn_direction(cumulative)
            if direction is None:
                updates.append(
                    {
                        "cycle": cycle,
                        "count": len(by_cycle[cycle]),
                        "maturity_weight": maturity_weight(cycle, min_cycle, full_cycle, args.maturity_floor),
                        "confidence_before": confidence,
                        "confidence_after": confidence,
                        "status": "waiting_for_both_roles",
                        "cumulative_role_counts": dict(Counter(rec["role"] for rec in cumulative).most_common()),
                    }
                )
                continue
            final_direction = direction
            evidence = evidence_from_cycle(by_cycle[cycle], direction["unit_direction"])
            maturity = maturity_weight(cycle, min_cycle, full_cycle, args.maturity_floor)
            before = confidence
            confidence, delta = update_confidence(
                confidence,
                evidence["evidence_score"],
                evidence["evidence_strength"],
                maturity,
                args.learning_rate,
                args.decay_to_neutral,
            )
            updates.append(
                {
                    "cycle": cycle,
                    "count": evidence["count"],
                    "success_count": evidence["success_count"],
                    "failure_count": evidence["failure_count"],
                    "maturity_weight": maturity,
                    "evidence_kind": evidence["evidence_kind"],
                    "evidence_score": evidence["evidence_score"],
                    "evidence_strength": evidence["evidence_strength"],
                    "confidence_before": before,
                    "confidence_delta": delta,
                    "confidence_after": confidence,
                    "alignment_margin_or_single_role_score": evidence["alignment_margin_or_single_role_score"],
                    "auc_success_gt_failure": evidence["auc_success_gt_failure"],
                    "success_mean_alignment": evidence["success_mean_alignment"],
                    "failure_mean_alignment": evidence["failure_mean_alignment"],
                    "success_best_family_counts": evidence["success_best_family_counts"],
                    "success_best_family_entropy": evidence["success_best_family_entropy"],
                    "future_reward_mean": evidence["future_reward_mean"],
                    "future_rootx_mean": evidence["future_rootx_mean"],
                    "cumulative_role_counts": dict(Counter(rec["role"] for rec in cumulative).most_common()),
                }
            )
        if final_direction is None:
            continue
        final_evidence = evidence_from_cycle(total, final_direction["unit_direction"])
        rules.append(
            {
                "rule_id": f"{ctx}->prefer_effect_direction",
                "rule_form": "context_key -> prefer_effect_direction",
                "context_key": ctx,
                "status": "shadow_effect_confidence_audited",
                "final_confidence": confidence,
                "num_updates": len([u for u in updates if u.get("status") != "waiting_for_both_roles"]),
                "total_count": len(total),
                "role_counts": dict(role_counts.most_common()),
                "cycles": sorted(by_cycle),
                "final_direction": final_direction,
                "aggregate_evidence": {
                    k: v for k, v in final_evidence.items()
                    if k != "rows"
                },
                "updates": updates,
            }
        )
    rules.sort(key=lambda item: (item["final_confidence"], item["aggregate_evidence"]["evidence_score"], item["total_count"]), reverse=True)
    return {
        "cycles": cycles,
        "maturity": {
            "min_cycle": min_cycle,
            "full_cycle": full_cycle,
            "floor": args.maturity_floor,
        },
        "rules": rules,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Effect-Space Shadow Confidence Audit",
        "",
        f"Segments: `{payload['config']['segments']}`",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        f"Context variant: `{payload['config']['context_variant']}`",
        f"Maturity full cycle: `{payload['shadow_report']['maturity']['full_cycle']}`",
        "",
        "## Rule Confidence",
        "",
        "| Rule | Count | Roles | Cycles | Confidence | Evidence | AUC | Margin | Top components |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rule in payload["shadow_report"]["rules"][:40]:
        comps = ", ".join(
            f"{c['key']}={c['value']:.3f}"
            for c in rule["final_direction"]["top_components"][:4]
        )
        ev = rule["aggregate_evidence"]
        lines.append(
            f"| `{rule['context_key']}` | {rule['total_count']} | `{rule['role_counts']}` | "
            f"`{','.join(str(c) for c in rule['cycles'])}` | {rule['final_confidence']:.3f} | "
            f"{ev['evidence_score']:.3f} | {ev['auc_success_gt_failure']:.3f} | "
            f"{ev['alignment_margin_or_single_role_score']:.3f} | `{comps}` |"
        )
    lines.extend([
        "",
        "## Top Rule Updates",
        "",
        "| Rule | Cycle | Maturity | Kind | Evidence | Delta | Confidence | Best families |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ])
    rows = []
    for rule in payload["shadow_report"]["rules"][:20]:
        for update in rule["updates"]:
            if update.get("status") == "waiting_for_both_roles":
                continue
            rows.append((rule, update))
    rows.sort(key=lambda item: abs(item[1]["confidence_delta"]), reverse=True)
    for rule, update in rows[:40]:
        lines.append(
            f"| `{rule['context_key']}` | {update['cycle']} | {update['maturity_weight']:.3f} | "
            f"`{update['evidence_kind']}` | {update['evidence_score']:.3f} | "
            f"{update['confidence_delta']:.3f} | {update['confidence_after']:.3f} | "
            f"`{update['success_best_family_counts']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "该报告只维护 shadow confidence，不写入在线规则库。早期 SD/TVA 证据通过 maturity weight 降权；因此早期反事实不支持不会强烈削弱规则，后期成熟模型的一致证据会产生更大信心更新。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-variant", default="anchor_refined_type")
    parser.add_argument("--include-tier", action="store_true", default=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--min-context-records", type=int, default=8)
    parser.add_argument("--min-role-records", type=int, default=2)
    parser.add_argument("--initial-confidence", type=float, default=0.50)
    parser.add_argument("--learning-rate", type=float, default=0.30)
    parser.add_argument("--decay-to-neutral", type=float, default=0.98)
    parser.add_argument("--maturity-floor", type=float, default=0.20)
    parser.add_argument("--maturity-full-cycle", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, skipped = recompute_effect_records(args)
    shadow_report = build_shadow_report(records, args)
    payload = {
        "analysis_family": "hopper_effect_space_shadow_confidence_v1",
        "effect_keys": EFFECT_KEYS,
        "num_records": len(records),
        "num_skipped": len(skipped),
        "config": {
            "segments": str(args.segments),
            "results_root": str(args.results_root),
            "context_variant": args.context_variant,
            "include_tier": args.include_tier,
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "max_windows": args.max_windows,
            "min_context_records": args.min_context_records,
            "min_role_records": args.min_role_records,
            "initial_confidence": args.initial_confidence,
            "learning_rate": args.learning_rate,
            "decay_to_neutral": args.decay_to_neutral,
            "maturity_floor": args.maturity_floor,
            "maturity_full_cycle": args.maturity_full_cycle,
            "device": args.device,
        },
        "shadow_report": shadow_report,
        "skipped": skipped,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote effect-space shadow confidence audit: {args.output}")
    print(f"records: {len(records)} skipped: {len(skipped)} rules: {len(shadow_report['rules'])}")


if __name__ == "__main__":
    main()
