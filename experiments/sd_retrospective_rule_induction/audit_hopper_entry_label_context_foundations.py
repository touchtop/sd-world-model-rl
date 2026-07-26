#!/usr/bin/env python3
"""Basic audit of Hopper ability-entry labels and context keys.

This script intentionally avoids new models.  It reviews whether the existing
ability-entry labels and context keys are clean enough to support rule
induction.  The audit is read-only and only uses previously exported entry
segments.
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

from common import load_json, mean_or_zero, quantile, write_json  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import entry_tier  # noqa: E402
from audit_hopper_phi_regret_potential import entropy_norm  # noqa: E402


DEFAULT_SEGMENTS = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_label_context_foundation_audit.json"
)


OUTCOME_KEYS = [
    "episode_reward",
    "reward_sum",
    "future_reward_sum",
    "rootx_delta",
    "future_rootx_delta",
    "mean_vx",
    "future_mean_vx",
    "support_contact_mean",
    "full_foot_support_mean",
    "rotation_penalty_frac",
    "falling_frac",
    "prone_frac",
    "hop_frac",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def context_key(record: dict[str, Any], include_refined_phase: bool = False) -> str:
    sig = record.get("anchor_signature") or {}
    phase = sig.get("refined_phase") if include_refined_phase else sig.get("phase")
    return "|".join(
        [
            str(sig.get("omega_bin", "omega_unknown")),
            str(sig.get("vx_bin", "vx_unknown")),
            str(sig.get("support_bin", "support_unknown")),
            str(phase or "phase_unknown"),
            str(sig.get("risk_bin", "risk_unknown")),
        ]
    )


def flatten_segments(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    out = []
    for segment in payload.get("segments", []):
        base = {
            "role": segment["role"],
            "threshold": float(segment["threshold"]),
            "anchor_cycle": int(segment["anchor_cycle"]),
            "cycle": int(segment["cycle"]),
            "seed": int(segment["seed"]),
            "episode": int(segment["episode"]),
            "episode_reward": float(segment["episode_reward"]),
            "trace_csv": segment["trace_csv"],
            "cycle_offset": int(segment["cycle"]) - int(segment["anchor_cycle"]),
        }
        for window in segment.get("windows", []):
            item = {**base, **window}
            item["tier"] = entry_tier(item)
            item["context_key"] = context_key(item, include_refined_phase=False)
            item["refined_context_key"] = context_key(item, include_refined_phase=True)
            item["window_uid"] = (
                str(item.get("trace_csv", "")),
                int(item.get("anchor_index", -1)),
                str(item.get("anchor_type", "")),
            )
            out.append(item)
    return out


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
    }


def role_outcome_contrast(records: list[dict[str, Any]]) -> dict[str, Any]:
    success = [r for r in records if r["role"] == "entry_success"]
    failure = [r for r in records if r["role"] == "matched_failure"]
    out: dict[str, Any] = {
        "count": len(records),
        "success_count": len(success),
        "failure_count": len(failure),
    }
    for key in OUTCOME_KEYS:
        s_vals = [safe_float(r.get(key)) for r in success]
        f_vals = [safe_float(r.get(key)) for r in failure]
        out[key] = {
            "success": summarize_values(s_vals),
            "failure": summarize_values(f_vals),
            "success_minus_failure_mean": mean_or_zero(s_vals) - mean_or_zero(f_vals),
            "auc_success_gt_failure": auc_positive_greater(s_vals, f_vals) if s_vals and f_vals else 0.0,
        }
    return out


def label_extension_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups["all"].append(r)
        groups[f"role:{r['role']}"].append(r)
        groups[f"tier:{r['tier']}|role:{r['role']}"].append(r)
    out = {}
    for key, vals in sorted(groups.items()):
        offsets = Counter(int(r["cycle_offset"]) for r in vals)
        thresholds = Counter(float(r["threshold"]) for r in vals)
        anchor_types = Counter(str(r.get("anchor_type", "unknown")) for r in vals)
        out[key] = {
            "count": len(vals),
            "cycle_offset_counts": dict(sorted(offsets.items())),
            "threshold_counts": {str(k): v for k, v in sorted(thresholds.items())},
            "anchor_type_counts": dict(anchor_types.most_common()),
            "pre_entry_fraction": mean_or_zero(1.0 if int(r["cycle_offset"]) < 0 else 0.0 for r in vals),
            "post_entry_fraction": mean_or_zero(1.0 if int(r["cycle_offset"]) > 0 else 0.0 for r in vals),
        }
    return out


def duplicate_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_uid: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_uid_threshold: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_uid[r["window_uid"]].append(r)
        by_uid_threshold[(*r["window_uid"], float(r["threshold"]))].append(r)
    duplicated = [vals for vals in by_uid.values() if len(vals) > 1]
    role_conflicts = [
        vals for vals in duplicated
        if len({v["role"] for v in vals}) > 1
    ]
    threshold_reuse = [
        vals for vals in duplicated
        if len({float(v["threshold"]) for v in vals}) > 1
    ]
    return {
        "raw_windows": len(records),
        "unique_window_uids": len(by_uid),
        "duplicate_uid_groups": len(duplicated),
        "duplicate_window_fraction": 1.0 - (len(by_uid) / float(len(records))) if records else 0.0,
        "role_conflict_uid_groups": len(role_conflicts),
        "threshold_reuse_uid_groups": len(threshold_reuse),
        "examples": [
            [
                {
                    "role": v["role"],
                    "threshold": v["threshold"],
                    "anchor_cycle": v["anchor_cycle"],
                    "cycle": v["cycle"],
                    "seed": v["seed"],
                    "episode": v["episode"],
                    "anchor_type": v["anchor_type"],
                    "future_reward_sum": v["future_reward_sum"],
                }
                for v in vals[:8]
            ]
            for vals in (role_conflicts[:5] or threshold_reuse[:5] or duplicated[:5])
        ],
    }


def ablation_contrasts(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_uid: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_uid[r["window_uid"]].append(r)
    role_consistent_uids = {
        uid for uid, vals in by_uid.items()
        if len({v["role"] for v in vals}) == 1
    }
    ablations = {
        "raw": records,
        "entry_cycle_only": [r for r in records if int(r["cycle_offset"]) == 0],
        "role_consistent_uids": [r for r in records if r["window_uid"] in role_consistent_uids],
        "entry_cycle_only_role_consistent_uids": [
            r for r in records
            if int(r["cycle_offset"]) == 0 and r["window_uid"] in role_consistent_uids
        ],
        "no_pre_entry": [r for r in records if int(r["cycle_offset"]) >= 0],
    }
    return {key: role_outcome_contrast(vals) for key, vals in ablations.items()}


def context_purity(records: list[dict[str, Any]], key_name: str, min_count: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r[key_name])].append(r)
    contexts = []
    for key, vals in sorted(groups.items()):
        if len(vals) < min_count:
            continue
        roles = Counter(r["role"] for r in vals)
        cycles = sorted({int(r["anchor_cycle"]) for r in vals})
        thresholds = sorted({float(r["threshold"]) for r in vals})
        families = Counter(str(r["dominant_action_family"]) for r in vals)
        family, family_count = families.most_common(1)[0]
        cycle_family_rates = []
        for cycle in cycles:
            cvals = [r for r in vals if int(r["anchor_cycle"]) == cycle]
            cycle_family_rates.append(mean_or_zero(1.0 if r["dominant_action_family"] == family else 0.0 for r in cvals))
        success = [r for r in vals if r["role"] == "entry_success"]
        failure = [r for r in vals if r["role"] == "matched_failure"]
        future_rewards = [safe_float(r["future_reward_sum"]) for r in vals]
        contexts.append(
            {
                "context_key": key,
                "count": len(vals),
                "cycles": cycles,
                "thresholds": thresholds,
                "role_counts": dict(roles.most_common()),
                "role_entropy": entropy_norm(roles),
                "success_fraction": roles.get("entry_success", 0) / float(len(vals)),
                "dominant_family": family,
                "dominant_family_rate": family_count / float(len(vals)),
                "dominant_family_cycle_min_rate": min(cycle_family_rates) if cycle_family_rates else 0.0,
                "action_family_entropy": entropy_norm(families),
                "action_family_counts": dict(families.most_common()),
                "future_reward_mean": mean_or_zero(future_rewards),
                "future_reward_std": float(np.std(future_rewards)) if future_rewards else 0.0,
                "success_future_reward_mean": mean_or_zero(safe_float(r["future_reward_sum"]) for r in success),
                "failure_future_reward_mean": mean_or_zero(safe_float(r["future_reward_sum"]) for r in failure),
                "success_minus_failure_future_reward": mean_or_zero(safe_float(r["future_reward_sum"]) for r in success) - mean_or_zero(safe_float(r["future_reward_sum"]) for r in failure),
                "success_minus_failure_future_rootx": mean_or_zero(safe_float(r["future_rootx_delta"]) for r in success) - mean_or_zero(safe_float(r["future_rootx_delta"]) for r in failure),
            }
        )
    ambiguous = [
        item for item in contexts
        if item["role_entropy"] >= 0.80
        or item["action_family_entropy"] >= 0.80
        or item["dominant_family_cycle_min_rate"] <= 0.0
    ]
    inverted = [
        item for item in contexts
        if item["success_minus_failure_future_reward"] < 0.0
    ]
    return {
        "num_contexts": len(contexts),
        "ambiguous_context_count": len(ambiguous),
        "inverted_future_reward_context_count": len(inverted),
        "top_ambiguous_contexts": sorted(
            ambiguous,
            key=lambda x: (x["role_entropy"], x["action_family_entropy"], x["count"]),
            reverse=True,
        )[:30],
        "top_inverted_contexts": sorted(
            inverted,
            key=lambda x: (x["success_minus_failure_future_reward"], -x["count"]),
        )[:30],
        "top_contexts_by_count": sorted(contexts, key=lambda x: x["count"], reverse=True)[:30],
    }


def event_group_contrasts(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups["all"].append(r)
        groups[f"tier:{r['tier']}"].append(r)
        groups[f"threshold:{float(r['threshold']):g}|cycle:{int(r['anchor_cycle']):03d}"].append(r)
        groups[f"tier:{r['tier']}|cycle:{int(r['anchor_cycle']):03d}"].append(r)
    out = {}
    for key, vals in sorted(groups.items()):
        summary = role_outcome_contrast(vals)
        out[key] = summary
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Entry Label/Context Foundation Audit",
        "",
        f"Raw windows: `{payload['duplicate_audit']['raw_windows']}`",
        f"Unique window uids: `{payload['duplicate_audit']['unique_window_uids']}`",
        f"Duplicate fraction: `{payload['duplicate_audit']['duplicate_window_fraction']:.3f}`",
        "",
        "## Label Extension",
        "",
        "| Group | Count | Pre-entry | Post-entry | Offsets |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, item in payload["label_extension_audit"].items():
        if key in ["all", "role:entry_success", "role:matched_failure", "tier:low_entry|role:entry_success", "tier:high_entry|role:entry_success"]:
            lines.append(
                f"| `{key}` | {item['count']} | {item['pre_entry_fraction']:.3f} | "
                f"{item['post_entry_fraction']:.3f} | `{item['cycle_offset_counts']}` |"
            )
    lines.extend([
        "",
        "## Role Outcome Contrasts",
        "",
        "| Group | Success | Failure | Future reward margin | Future reward AUC | Future rootx margin | Future rootx AUC | Episode reward margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in ["all", "tier:low_entry", "tier:high_entry"]:
        item = payload["event_group_contrasts"][key]
        fr = item["future_reward_sum"]
        fx = item["future_rootx_delta"]
        er = item["episode_reward"]
        lines.append(
            f"| `{key}` | {item['success_count']} | {item['failure_count']} | "
            f"{fr['success_minus_failure_mean']:.3f} | {fr['auc_success_gt_failure']:.3f} | "
            f"{fx['success_minus_failure_mean']:.3f} | {fx['auc_success_gt_failure']:.3f} | "
            f"{er['success_minus_failure_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Label Ablations",
        "",
        "| Ablation | Success | Failure | Future reward margin | Future reward AUC | Future rootx margin | Future rootx AUC | Episode reward margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in ["raw", "no_pre_entry", "entry_cycle_only", "role_consistent_uids", "entry_cycle_only_role_consistent_uids"]:
        item = payload["ablation_contrasts"][key]
        fr = item["future_reward_sum"]
        fx = item["future_rootx_delta"]
        er = item["episode_reward"]
        lines.append(
            f"| `{key}` | {item['success_count']} | {item['failure_count']} | "
            f"{fr['success_minus_failure_mean']:.3f} | {fr['auc_success_gt_failure']:.3f} | "
            f"{fx['success_minus_failure_mean']:.3f} | {fx['auc_success_gt_failure']:.3f} | "
            f"{er['success_minus_failure_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Context Purity",
        "",
        "| Key | Contexts | Ambiguous | Inverted future reward |",
        "| --- | ---: | ---: | ---: |",
    ])
    for key in ["context_key", "refined_context_key"]:
        item = payload["context_purity"][key]
        lines.append(
            f"| `{key}` | {item['num_contexts']} | {item['ambiguous_context_count']} | "
            f"{item['inverted_future_reward_context_count']} |"
        )
    lines.extend([
        "",
        "## Top Ambiguous Contexts",
        "",
        "| Context | Count | Cycles | Success frac | Role entropy | Family | Family rate | Family entropy | Cycle min rate | Future reward margin |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["context_purity"]["context_key"]["top_ambiguous_contexts"][:20]:
        lines.append(
            f"| `{item['context_key']}` | {item['count']} | `{','.join(str(c) for c in item['cycles'])}` | "
            f"{item['success_fraction']:.3f} | {item['role_entropy']:.3f} | `{item['dominant_family']}` | "
            f"{item['dominant_family_rate']:.3f} | {item['action_family_entropy']:.3f} | "
            f"{item['dominant_family_cycle_min_rate']:.3f} | {item['success_minus_failure_future_reward']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "该审查只评估现有标签与上下文本身，不使用 SD/TVA 反事实。若 success/failure 标签在未来后果上不稳定，或同一 context_key 内动作族高熵且跨周期最小命中为 0，则后续模型难以从该口径中稳定提取规则。",
    ])
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-context-count", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = flatten_segments(args.segments)
    payload = {
        "analysis_family": "hopper_entry_label_context_foundation_v1",
        "config": {
            "segments": str(args.segments),
            "min_context_count": int(args.min_context_count),
        },
        "num_records": len(records),
        "duplicate_audit": duplicate_audit(records),
        "label_extension_audit": label_extension_audit(records),
        "ablation_contrasts": ablation_contrasts(records),
        "event_group_contrasts": event_group_contrasts(records),
        "context_purity": {
            "context_key": context_purity(records, "context_key", args.min_context_count),
            "refined_context_key": context_purity(records, "refined_context_key", args.min_context_count),
        },
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote entry label/context foundation audit: {args.output}")
    print(f"records: {len(records)}")


if __name__ == "__main__":
    main()
