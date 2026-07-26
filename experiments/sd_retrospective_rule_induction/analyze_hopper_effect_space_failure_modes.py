#!/usr/bin/env python3
"""Analyze failure modes of Hopper ability-entry effect-space directions.

This report is diagnostic.  It recomputes normalized effect deltas, summarizes
group geometry, and points to mathematical causes of unstable transfer.  It
does not build rules or online priors.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import mean_or_zero, write_json  # noqa: E402
from audit_hopper_ability_entry_effect_space import EFFECT_KEYS, dot, normalize  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater, recompute_effect_records  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import (  # noqa: E402
    dedupe_records,
    entry_tier,
    normalize_by_cycle,
    usable_tier_cycle_groups,
)


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/effect_space_failure_modes.json"
)


def role_vectors(records: list[dict[str, Any]], role: str) -> list[list[float]]:
    return [rec["actual_delta_z"] for rec in records if rec["role"] == role]


def mean_vec(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return [0.0 for _ in EFFECT_KEYS]
    return [mean_or_zero(vec[i] for vec in vectors) for i in range(len(EFFECT_KEYS))]


def diff_vec(records: list[dict[str, Any]]) -> list[float] | None:
    success = role_vectors(records, "entry_success")
    failure = role_vectors(records, "matched_failure")
    if not success or not failure:
        return None
    s = mean_vec(success)
    f = mean_vec(failure)
    return [a - b for a, b in zip(s, f)]


def group_key(record: dict[str, Any]) -> str:
    return f"{entry_tier(record)}|cycle:{int(record['anchor_cycle']):03d}"


def cosine(a: list[float], b: list[float]) -> float:
    return dot(normalize(a), normalize(b))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def top_components(v: list[float], k: int = 8) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": value}
        for key, value in sorted(zip(EFFECT_KEYS, v), key=lambda item: abs(item[1]), reverse=True)[:k]
    ]


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    diff = diff_vec(records) or [0.0 for _ in EFFECT_KEYS]
    success = [dot(rec["actual_delta_z"], normalize(diff)) for rec in records if rec["role"] == "entry_success"]
    failure = [dot(rec["actual_delta_z"], normalize(diff)) for rec in records if rec["role"] == "matched_failure"]
    sig_counts = Counter()
    for rec in records:
        sig = rec.get("anchor_signature", {})
        sig_counts.update([f"{sig.get('phase','?')}|{sig.get('vx_bin','?')}|{sig.get('support_bin','?')}|{sig.get('risk_bin','?')}"])
    return {
        "count": len(records),
        "success_count": sum(1 for rec in records if rec["role"] == "entry_success"),
        "failure_count": sum(1 for rec in records if rec["role"] == "matched_failure"),
        "mean_success_future_reward": mean_or_zero(rec["future_reward_sum"] for rec in records if rec["role"] == "entry_success"),
        "mean_failure_future_reward": mean_or_zero(rec["future_reward_sum"] for rec in records if rec["role"] == "matched_failure"),
        "direction_norm": norm(diff),
        "in_group_margin": mean_or_zero(success) - mean_or_zero(failure),
        "in_group_auc": auc_positive_greater(success, failure),
        "top_components": top_components(diff),
        "anchor_signature_counts": dict(sig_counts.most_common(8)),
    }


def sign_consistency(group_diffs: dict[str, list[float]]) -> dict[str, Any]:
    rows = []
    for dim, key in enumerate(EFFECT_KEYS):
        signs = []
        vals = []
        for diff in group_diffs.values():
            val = diff[dim]
            vals.append(val)
            signs.append(1 if val > 1e-9 else -1 if val < -1e-9 else 0)
        nonzero = [s for s in signs if s != 0]
        pos_rate = sum(1 for s in nonzero if s > 0) / len(nonzero) if nonzero else 0.0
        neg_rate = sum(1 for s in nonzero if s < 0) / len(nonzero) if nonzero else 0.0
        rows.append(
            {
                "key": key,
                "mean_abs_diff": mean_or_zero(abs(v) for v in vals),
                "mean_diff": mean_or_zero(vals),
                "pos_rate": pos_rate,
                "neg_rate": neg_rate,
                "max_sign_agreement": max(pos_rate, neg_rate),
            }
        )
    rows.sort(key=lambda item: (item["max_sign_agreement"], item["mean_abs_diff"]), reverse=True)
    return {"dimensions": rows}


def variance_decomposition(records: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_vec = np.asarray([rec["actual_delta_z"] for rec in records], dtype=np.float64)
    if len(all_vec) == 0:
        return {}
    overall = all_vec.mean(axis=0)
    total_var = ((all_vec - overall) ** 2).mean(axis=0)
    group_means = []
    group_sizes = []
    within_vars = []
    for vals in groups.values():
        arr = np.asarray([rec["actual_delta_z"] for rec in vals], dtype=np.float64)
        if len(arr) == 0:
            continue
        group_means.append(arr.mean(axis=0))
        group_sizes.append(len(arr))
        within_vars.append(((arr - arr.mean(axis=0)) ** 2).mean(axis=0))
    weights = np.asarray(group_sizes, dtype=np.float64)
    weights = weights / max(weights.sum(), 1.0)
    between = np.zeros(len(EFFECT_KEYS), dtype=np.float64)
    within = np.zeros(len(EFFECT_KEYS), dtype=np.float64)
    for w, mu, var in zip(weights, group_means, within_vars):
        between += w * ((mu - overall) ** 2)
        within += w * var
    rows = []
    for i, key in enumerate(EFFECT_KEYS):
        rows.append(
            {
                "key": key,
                "total_var": float(total_var[i]),
                "between_group_var": float(between[i]),
                "within_group_var": float(within[i]),
                "between_share": float(between[i] / total_var[i]) if total_var[i] > 1e-12 else 0.0,
            }
        )
    rows.sort(key=lambda item: item["between_share"], reverse=True)
    return {"dimensions": rows}


def pca_summary(group_diffs: dict[str, list[float]]) -> dict[str, Any]:
    keys = sorted(group_diffs)
    if len(keys) < 2:
        return {}
    x = np.asarray([group_diffs[key] for key in keys], dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        _, s, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return {}
    var = s ** 2
    total = float(var.sum())
    components = []
    for idx in range(min(4, len(vt))):
        vec = vt[idx].tolist()
        components.append(
            {
                "explained_ratio": float(var[idx] / total) if total > 1e-12 else 0.0,
                "top_components": top_components(vec, 6),
            }
        )
    return {"group_keys": keys, "components": components}


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = usable_tier_cycle_groups(records)
    group_diffs: dict[str, list[float]] = {}
    for key, vals in groups.items():
        diff = diff_vec(vals)
        if diff is not None:
            group_diffs[key] = diff
    group_summaries = {key: summarize_group(vals) for key, vals in groups.items()}

    cosine_rows = []
    keys = sorted(group_diffs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            cosine_rows.append(
                {
                    "a": a,
                    "b": b,
                    "same_tier": a.split("|", 1)[0] == b.split("|", 1)[0],
                    "cosine": cosine(group_diffs[a], group_diffs[b]),
                }
            )
    cosine_rows.sort(key=lambda item: item["cosine"], reverse=True)

    tier_rows = {}
    for tier in sorted({key.split("|", 1)[0] for key in keys}):
        tier_cos = [row["cosine"] for row in cosine_rows if row["same_tier"] and row["a"].startswith(tier)]
        tier_rows[tier] = {
            "groups": [key for key in keys if key.startswith(tier)],
            "mean_same_tier_cosine": mean_or_zero(tier_cos),
            "min_same_tier_cosine": min(tier_cos) if tier_cos else 0.0,
            "max_same_tier_cosine": max(tier_cos) if tier_cos else 0.0,
        }

    return {
        "group_summaries": group_summaries,
        "direction_cosines": cosine_rows,
        "tier_summary": tier_rows,
        "sign_consistency": sign_consistency(group_diffs),
        "variance_decomposition": variance_decomposition(records, groups),
        "pca_group_diff": pca_summary(group_diffs),
        "first_principles_diagnosis": [
            "Current delta_z mixes action consequence with state/cycle baseline; unstable transfer indicates covariate shift rather than a single invariant control direction.",
            "Reward-threshold groups conflate different mechanisms: early escape from zero reward, later positive-reward continuation, and high-threshold posture maintenance.",
            "TVA and posture dimensions carry large between-group variation; linear directions learned on one group can reverse when the target group lies on another state manifold.",
            "A better next mathematical object is local action-effect advantage: delta_z(s,a) minus the candidate-action-set baseline at the same state, then learn invariant directions on that centered tangent space.",
        ],
        "recommended_next_method": {
            "name": "local_action_effect_advantage_space",
            "definition": "A_z(s,a)=delta_z(s,a)-mean_{a' in probe_bank} delta_z(s,a')",
            "reason": "It removes state/cycle baseline and asks which action moves better than other actions available at the same state.",
            "rule_form": "context_key -> prefer direction in local action-effect advantage space",
            "complexity_control": "offline audit only; require cross-cycle transfer before any shadow prior",
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Effect-Space Failure Mode Analysis",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        "",
        "## Tier Geometry",
        "",
        "| Tier | Groups | Mean cosine | Min cosine | Max cosine |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tier, row in payload["analysis"]["tier_summary"].items():
        lines.append(
            f"| `{tier}` | {len(row['groups'])} | {row['mean_same_tier_cosine']:.3f} | "
            f"{row['min_same_tier_cosine']:.3f} | {row['max_same_tier_cosine']:.3f} |"
        )
    lines.extend(["", "## Group Summaries", ""])
    for key, row in payload["analysis"]["group_summaries"].items():
        comps = ", ".join(f"{c['key']}={c['value']:.3f}" for c in row["top_components"][:5])
        lines.append(
            f"- `{key}`: success=`{row['success_count']}`, failure=`{row['failure_count']}`, "
            f"in_group_auc=`{row['in_group_auc']:.3f}`, future_reward="
            f"`{row['mean_success_future_reward']:.3f}/{row['mean_failure_future_reward']:.3f}`, "
            f"components=`{comps}`"
        )
    lines.extend(["", "## Sign Consistency", ""])
    for row in payload["analysis"]["sign_consistency"]["dimensions"][:10]:
        lines.append(
            f"- `{row['key']}`: agreement=`{row['max_sign_agreement']:.3f}`, "
            f"mean_abs_diff=`{row['mean_abs_diff']:.3f}`, mean_diff=`{row['mean_diff']:.3f}`"
        )
    lines.extend(["", "## Variance Decomposition", ""])
    for row in payload["analysis"]["variance_decomposition"]["dimensions"][:10]:
        lines.append(
            f"- `{row['key']}`: between_share=`{row['between_share']:.3f}`, "
            f"total_var=`{row['total_var']:.3f}`"
        )
    lines.extend(["", "## First-Principles Diagnosis", ""])
    for item in payload["analysis"]["first_principles_diagnosis"]:
        lines.append(f"- {item}")
    method = payload["analysis"]["recommended_next_method"]
    lines.extend(
        [
            "",
            "## Recommended Mathematical Method",
            "",
            f"Name: `{method['name']}`",
            "",
            f"Definition: `{method['definition']}`",
            "",
            f"Reason: {method['reason']}",
            "",
            f"Rule form: `{method['rule_form']}`",
            "",
            f"Complexity control: {method['complexity_control']}",
        ]
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
    if args.segments is None:
        from audit_hopper_ability_entry_effect_space import DEFAULT_SEGMENTS
        args.segments = DEFAULT_SEGMENTS
    if args.results_root is None:
        from audit_hopper_ability_entry_effect_space import DEFAULT_RESULTS_ROOT
        args.results_root = DEFAULT_RESULTS_ROOT
    raw_records, skipped = recompute_effect_records(args)
    deduped = dedupe_records(raw_records)
    normalized, normalization_stats = normalize_by_cycle(deduped)
    analysis = build_report(normalized)
    payload = {
        "analysis_family": "hopper_effect_space_failure_modes_v1",
        "num_raw_records": len(raw_records),
        "num_records": len(normalized),
        "num_skipped": len(skipped),
        "effect_keys": EFFECT_KEYS,
        "normalization_stats": normalization_stats,
        "analysis": analysis,
        "skipped": skipped,
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote effect-space failure analysis: {args.output}")
    print(f"records: {payload['num_records']} skipped: {payload['num_skipped']}")


if __name__ == "__main__":
    from audit_hopper_ability_entry_minimal_stable_effect import normalize_by_cycle
    from audit_hopper_ability_entry_effect_space_stability import recompute_effect_records

    main()
