#!/usr/bin/env python3
"""Audit context variants for strict Hopper ability-entry labels.

The strict entry labels improved outcome separability, but earlier mathematical
audits still failed to produce stable rules.  This script keeps the analysis
model-free and asks whether a better context key can expose simple candidates
of the form:

    context_key -> prefer_action_family

No rule library is written.  The output is an offline diagnostic for deciding
whether context definition, rather than classifier complexity, is the current
bottleneck.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, mean_or_zero, write_json  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import entry_tier  # noqa: E402
from audit_hopper_phi_regret_potential import entropy_norm  # noqa: E402


DEFAULT_BASE_DIR = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1"
)
DEFAULT_SEGMENTS = DEFAULT_BASE_DIR / "entry_candidate_segments_strict_no_pre_entry.json"
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "entry_strict_context_variant_audit_no_pre_entry.json"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def flatten_segments(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    records: list[dict[str, Any]] = []
    for segment in payload.get("segments", []):
        base = {
            "role": str(segment["role"]),
            "threshold": float(segment["threshold"]),
            "anchor_cycle": int(segment["anchor_cycle"]),
            "cycle": int(segment["cycle"]),
            "seed": int(segment["seed"]),
            "episode": int(segment["episode"]),
            "episode_reward": safe_float(segment.get("episode_reward")),
            "trace_csv": str(segment.get("trace_csv", "")),
            "cycle_offset": int(segment["cycle"]) - int(segment["anchor_cycle"]),
        }
        for window in segment.get("windows", []):
            item = {**base, **window}
            item["tier"] = entry_tier(item)
            item["window_uid"] = (
                item["trace_csv"],
                int(item.get("anchor_index", -1)),
                str(item.get("anchor_type", "")),
            )
            records.append(item)
    return records


def top_counter_key(counts: dict[str, int] | None, default: str) -> str:
    if not counts:
        return default
    counter = Counter({str(k): int(v) for k, v in counts.items()})
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def frac_bin(value: Any, name: str, low: float = 0.10, high: float = 0.50) -> str:
    x = safe_float(value)
    if x < low:
        return f"{name}_none"
    if x < high:
        return f"{name}_some"
    return f"{name}_high"


def support_bin(value: Any, name: str) -> str:
    x = safe_float(value)
    if x < 0.10:
        return f"{name}_none"
    if x < 0.35:
        return f"{name}_low"
    return f"{name}_high"


def rate_bin(value: Any) -> str:
    x = safe_float(value)
    if x < 0.80:
        return "pitch_rate_low"
    if x < 1.50:
        return "pitch_rate_mid"
    return "pitch_rate_high"


def pitch_bin(value: Any) -> str:
    x = safe_float(value)
    if x < 0.60:
        return "pitch_low"
    if x < 1.10:
        return "pitch_mid"
    return "pitch_high"


def sig(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("anchor_signature") or {}


def join(parts: list[Any]) -> str:
    return "|".join(str(part) for part in parts)


def context_variants() -> dict[str, Callable[[dict[str, Any]], str]]:
    return {
        "anchor_phase": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            sig(r).get("support_bin", "support_unknown"),
            sig(r).get("phase", "phase_unknown"),
            sig(r).get("risk_bin", "risk_unknown"),
        ]),
        "anchor_refined_phase": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            sig(r).get("support_bin", "support_unknown"),
            sig(r).get("refined_phase", "phase_unknown"),
            sig(r).get("risk_bin", "risk_unknown"),
        ]),
        "anchor_refined_type": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            sig(r).get("support_bin", "support_unknown"),
            sig(r).get("refined_phase", "phase_unknown"),
            sig(r).get("risk_bin", "risk_unknown"),
            r.get("anchor_type", "anchor_unknown"),
        ]),
        "dominant_refined_phase": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            sig(r).get("support_bin", "support_unknown"),
            top_counter_key(r.get("refined_phase_counts"), "phase_unknown"),
            sig(r).get("risk_bin", "risk_unknown"),
        ]),
        "compact_mechanism": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            support_bin(r.get("support_contact_mean"), "contact"),
            top_counter_key(r.get("refined_phase_counts"), "phase_unknown"),
            frac_bin(r.get("falling_frac"), "falling"),
        ]),
        "support_mechanism": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            support_bin(r.get("support_contact_mean"), "contact"),
            support_bin(r.get("full_foot_support_mean"), "fullfoot"),
            top_counter_key(r.get("refined_phase_counts"), "phase_unknown"),
            frac_bin(r.get("falling_frac"), "falling"),
        ]),
        "risk_mechanism": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            support_bin(r.get("support_contact_mean"), "contact"),
            top_counter_key(r.get("refined_phase_counts"), "phase_unknown"),
            frac_bin(r.get("falling_frac"), "falling"),
            frac_bin(r.get("rotation_penalty_frac"), "rotation"),
            frac_bin(r.get("prone_frac"), "prone"),
        ]),
        "posture_mechanism": lambda r: join([
            sig(r).get("omega_bin", "omega_unknown"),
            sig(r).get("vx_bin", "vx_unknown"),
            support_bin(r.get("support_contact_mean"), "contact"),
            pitch_bin(r.get("mean_abs_pitch")),
            rate_bin(r.get("mean_abs_pitch_rate")),
            top_counter_key(r.get("refined_phase_counts"), "phase_unknown"),
        ]),
    }


def family_rate(records: list[dict[str, Any]], family: str) -> float:
    return mean_or_zero(1.0 if str(r.get("dominant_action_family")) == family else 0.0 for r in records)


def binary_family_auc(success: list[dict[str, Any]], failure: list[dict[str, Any]], family: str) -> float:
    s_scores = [1.0 if str(r.get("dominant_action_family")) == family else 0.0 for r in success]
    f_scores = [1.0 if str(r.get("dominant_action_family")) == family else 0.0 for r in failure]
    return auc_positive_greater(s_scores, f_scores) if s_scores and f_scores else 0.0


def summarize_group(key: str, records: list[dict[str, Any]], min_cycles: int) -> dict[str, Any]:
    roles = Counter(str(r["role"]) for r in records)
    success = [r for r in records if r["role"] == "entry_success"]
    failure = [r for r in records if r["role"] == "matched_failure"]
    cycles = sorted({int(r["anchor_cycle"]) for r in records})
    success_cycles = sorted({int(r["anchor_cycle"]) for r in success})
    success_families = Counter(str(r.get("dominant_action_family", "family_unknown")) for r in success)
    all_families = Counter(str(r.get("dominant_action_family", "family_unknown")) for r in records)
    family, family_count = success_families.most_common(1)[0] if success_families else ("family_unknown", 0)
    success_rate = family_rate(success, family)
    failure_rate = family_rate(failure, family)
    cycle_success_rates = [
        family_rate([r for r in success if int(r["anchor_cycle"]) == cycle], family)
        for cycle in success_cycles
    ]
    future_reward_margin = (
        mean_or_zero(safe_float(r.get("future_reward_sum")) for r in success)
        - mean_or_zero(safe_float(r.get("future_reward_sum")) for r in failure)
    )
    future_rootx_margin = (
        mean_or_zero(safe_float(r.get("future_rootx_delta")) for r in success)
        - mean_or_zero(safe_float(r.get("future_rootx_delta")) for r in failure)
    )
    family_margin = success_rate - failure_rate
    family_auc = binary_family_auc(success, failure, family)
    stable_like = (
        len(success) >= 4
        and len(failure) >= 4
        and len(success_cycles) >= min_cycles
        and success_rate >= 0.45
        and family_margin >= 0.20
        and min(cycle_success_rates or [0.0]) >= 0.25
        and future_reward_margin > 0.0
        and future_rootx_margin > 0.0
    )
    ambiguous = (
        entropy_norm(roles) >= 0.80
        or entropy_norm(all_families) >= 0.80
        or (cycle_success_rates and min(cycle_success_rates) <= 0.0)
    )
    return {
        "context_key": key,
        "count": len(records),
        "cycles": cycles,
        "success_cycles": success_cycles,
        "role_counts": dict(roles.most_common()),
        "role_entropy": entropy_norm(roles),
        "success_count": len(success),
        "failure_count": len(failure),
        "success_fraction": roles.get("entry_success", 0) / float(len(records)) if records else 0.0,
        "success_dominant_family": family,
        "success_dominant_family_rate": success_rate,
        "failure_same_family_rate": failure_rate,
        "family_margin": family_margin,
        "family_auc_success_gt_failure": family_auc,
        "success_family_cycle_min_rate": min(cycle_success_rates) if cycle_success_rates else 0.0,
        "success_family_entropy": entropy_norm(success_families),
        "all_family_entropy": entropy_norm(all_families),
        "all_family_counts": dict(all_families.most_common()),
        "success_family_counts": dict(success_families.most_common()),
        "future_reward_margin": future_reward_margin,
        "future_rootx_margin": future_rootx_margin,
        "stable_like": stable_like,
        "ambiguous": ambiguous,
    }


def summarize_variant(
    records: list[dict[str, Any]],
    variant_name: str,
    key_fn: Callable[[dict[str, Any]], str],
    scope: str,
    min_count: int,
    min_cycles: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if scope == "tier_context":
            key = f"{record['tier']}::{key}"
        groups[key].append(record)
    analyzed = [
        summarize_group(key, vals, min_cycles)
        for key, vals in sorted(groups.items())
        if len(vals) >= min_count
    ]
    candidates = [item for item in analyzed if item["stable_like"]]
    ambiguous = [item for item in analyzed if item["ambiguous"]]
    return {
        "variant": variant_name,
        "scope": scope,
        "raw_group_count": len(groups),
        "analyzed_group_count": len(analyzed),
        "stable_like_count": len(candidates),
        "ambiguous_count": len(ambiguous),
        "mean_family_margin": mean_or_zero(item["family_margin"] for item in analyzed),
        "mean_family_auc": mean_or_zero(item["family_auc_success_gt_failure"] for item in analyzed),
        "mean_future_reward_margin": mean_or_zero(item["future_reward_margin"] for item in analyzed),
        "top_candidates": sorted(
            candidates,
            key=lambda item: (
                item["family_auc_success_gt_failure"],
                item["family_margin"],
                item["count"],
            ),
            reverse=True,
        )[:20],
        "top_nonstable_by_auc": sorted(
            analyzed,
            key=lambda item: (
                item["family_auc_success_gt_failure"],
                item["family_margin"],
                item["count"],
            ),
            reverse=True,
        )[:20],
        "top_ambiguous": sorted(
            ambiguous,
            key=lambda item: (item["role_entropy"], item["all_family_entropy"], item["count"]),
            reverse=True,
        )[:20],
    }


def build_report(records: list[dict[str, Any]], min_count: int, min_cycles: int) -> dict[str, Any]:
    variants = context_variants()
    variant_reports = []
    for name, fn in variants.items():
        for scope in ["context", "tier_context"]:
            variant_reports.append(summarize_variant(records, name, fn, scope, min_count, min_cycles))
    ranked = sorted(
        variant_reports,
        key=lambda item: (
            item["stable_like_count"],
            -item["ambiguous_count"],
            item["mean_family_auc"],
        ),
        reverse=True,
    )
    return {
        "num_records": len(records),
        "role_counts": dict(Counter(r["role"] for r in records).most_common()),
        "cycle_counts": dict(Counter(str(int(r["anchor_cycle"])) for r in records).most_common()),
        "variant_reports": variant_reports,
        "ranked_variants": ranked,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Strict Context Variant Audit",
        "",
        f"Segments: `{payload['config']['segments']}`",
        f"Records: `{payload['num_records']}`",
        f"Min count: `{payload['config']['min_count']}`",
        f"Min success cycles: `{payload['config']['min_cycles']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Scope | Groups | Stable-like | Ambiguous | Mean family AUC | Mean family margin |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["ranked_variants"]:
        lines.append(
            f"| `{item['variant']}` | `{item['scope']}` | {item['analyzed_group_count']} | "
            f"{item['stable_like_count']} | {item['ambiguous_count']} | "
            f"{item['mean_family_auc']:.3f} | {item['mean_family_margin']:.3f} |"
        )
    lines.extend([
        "",
        "## Stable-like Candidates",
        "",
        "| Variant | Scope | Context | Count | Success/Failure | Family | Success rate | Failure rate | Family AUC | Reward margin | Root-x margin |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    rows = []
    for item in payload["variant_reports"]:
        for candidate in item["top_candidates"]:
            rows.append((item["variant"], item["scope"], candidate))
    rows = sorted(
        rows,
        key=lambda row: (
            row[2]["family_auc_success_gt_failure"],
            row[2]["family_margin"],
            row[2]["count"],
        ),
        reverse=True,
    )
    if not rows:
        lines.append("| `_none_` |  |  | 0 |  |  |  |  |  |  |  |")
    for variant, scope, item in rows[:40]:
        role_counts = f"{item['success_count']}/{item['failure_count']}"
        lines.append(
            f"| `{variant}` | `{scope}` | `{item['context_key']}` | {item['count']} | "
            f"{role_counts} | `{item['success_dominant_family']}` | "
            f"{item['success_dominant_family_rate']:.3f} | {item['failure_same_family_rate']:.3f} | "
            f"{item['family_auc_success_gt_failure']:.3f} | {item['future_reward_margin']:.3f} | "
            f"{item['future_rootx_margin']:.3f} |"
        )
    lines.extend([
        "",
        "## Best Non-stable Rows",
        "",
        "| Variant | Scope | Context | Count | Family | Family AUC | Success rate | Failure rate | Cycle-min | Reward margin |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    rows = []
    for item in payload["variant_reports"]:
        for candidate in item["top_nonstable_by_auc"][:5]:
            if candidate["stable_like"]:
                continue
            rows.append((item["variant"], item["scope"], candidate))
    rows = sorted(
        rows,
        key=lambda row: (
            row[2]["family_auc_success_gt_failure"],
            row[2]["family_margin"],
            row[2]["count"],
        ),
        reverse=True,
    )
    for variant, scope, item in rows[:30]:
        lines.append(
            f"| `{variant}` | `{scope}` | `{item['context_key']}` | {item['count']} | "
            f"`{item['success_dominant_family']}` | {item['family_auc_success_gt_failure']:.3f} | "
            f"{item['success_dominant_family_rate']:.3f} | {item['failure_same_family_rate']:.3f} | "
            f"{item['success_family_cycle_min_rate']:.3f} | {item['future_reward_margin']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "该审计不训练分类器，也不生成规则。`stable-like` 仅表示严格入口标签下，某个上下文中的成功窗口在动作族、后续奖励、后续位移和跨周期出现上同时满足保守筛查条件；它仍需后续 SD/TVA 反事实和真实后果审计后，才可能转化为规则。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--min-cycles", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = flatten_segments(args.segments)
    report = build_report(records, args.min_count, args.min_cycles)
    payload = {
        "analysis_family": "hopper_strict_context_variant_audit_v1",
        "config": {
            "segments": str(args.segments),
            "min_count": int(args.min_count),
            "min_cycles": int(args.min_cycles),
        },
        **report,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    top = payload["ranked_variants"][0] if payload["ranked_variants"] else {}
    print(f"wrote strict context variant audit: {args.output}")
    print(
        "top variant:",
        top.get("variant"),
        top.get("scope"),
        "stable_like",
        top.get("stable_like_count"),
        "ambiguous",
        top.get("ambiguous_count"),
    )


if __name__ == "__main__":
    main()
