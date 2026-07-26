#!/usr/bin/env python3
"""Extract Hopper v8 transition action evidence from mined success fragments.

This script is deliberately observational: it reads saved trace CSV files and
the v8 success-fragment report, then converts key frames into state-action
evidence. It does not update training checkpoints, rule libraries, or replay
buffers. The research purpose is to prepare auditable candidates before adding
SD counterfactual support.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import f, load_json, mean_or_zero, read_csv, write_json
from mine_hopper_v8_success_segments import HopperV8Adapter


DEFAULT_FRAGMENTS = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.json")
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.json")
ACTION_NAMES = ("waist", "hip", "knee", "ankle")


def action_values(row: dict[str, str]) -> dict[str, float]:
    return {name: f(row, f"action_{idx}") for idx, name in enumerate(ACTION_NAMES)}


def signed_family(name: str, value: float, threshold: float) -> str:
    if value > threshold:
        return f"{name}_pos"
    if value < -threshold:
        return f"{name}_neg"
    return f"{name}_weak"


def primary_action_family(row: dict[str, str], threshold: float) -> str:
    vals = action_values(row)
    name, value = max(vals.items(), key=lambda item: abs(item[1]))
    return signed_family(name, value, threshold)


def action_families(row: dict[str, str], threshold: float) -> list[str]:
    families: list[str] = []
    for name, value in action_values(row).items():
        family = signed_family(name, value, threshold)
        if not family.endswith("_weak"):
            families.append(family)
    return families or ["all_weak"]


def refined_phase(adapter: HopperV8Adapter, row: dict[str, str]) -> str:
    """Separate near-good transition states from legacy prone/rear labels.

    Existing SD phase labels are retained in each record, but positive-reward
    Hopper fragments often carry `grounded_prone` while still having high torso
    height and forward speed. For v8 rule keys, such frames are better treated
    as successful transition states whose next support is unresolved.
    """

    if adapter.structural_success(row) or f(row, "reward") > adapter.reward_min:
        if f(row, "full_foot_support") >= 0.35:
            return "near_good_support"
        if f(row, "support_contact") >= 0.10 or f(row, "foot_grounded") >= 0.10:
            return "near_good_contact"
        return "near_good_air"
    return adapter.phase(row)


PHASE_ANY = "phase_any"


def refined_key(adapter: HopperV8Adapter, role: str, quality_bin: str, row: dict[str, str]) -> str:
    return "|".join(
        [
            role,
            quality_bin,
            adapter.omega_bin(row),
            adapter.vx_bin(row),
            adapter.support_bin(row),
            refined_phase(adapter, row),
            adapter.risk_bin(row),
        ]
    )


def phase_any_key(adapter: HopperV8Adapter, role: str, quality_bin: str, row: dict[str, str]) -> str:
    """Aggregate transition evidence across near-good/falling phase aliases.

    Hopper transition-gap records are sparse; using the full refined phase in
    the online rule key split otherwise coherent transition evidence across
    `falling`, `near_good_air`, and `near_good_contact`. The phase token is kept
    as an explicit wildcard so downstream matchers can still distinguish this
    intentional aggregation from an accidental short key.
    """

    return "|".join(
        [
            role,
            quality_bin,
            adapter.omega_bin(row),
            adapter.vx_bin(row),
            adapter.support_bin(row),
            PHASE_ANY,
            adapter.risk_bin(row),
        ]
    )


def transition_rule_key(adapter: HopperV8Adapter, role: str, quality_bin: str, row: dict[str, str], mode: str) -> str:
    if mode == "exact":
        return refined_key(adapter, role, quality_bin, row)
    if mode == "phase_any":
        return phase_any_key(adapter, role, quality_bin, row)
    raise ValueError(f"unknown v8 transition key mode: {mode}")


def score_snapshot(adapter: HopperV8Adapter, row: dict[str, str]) -> dict[str, float]:
    return {
        "continue_score": adapter.continuation_score(row),
        "transition_score": adapter.transition_score(row),
        "reward": f(row, "reward"),
        "pitch": f(row, "pitch"),
        "pitch_rate": f(row, "pitch_rate"),
        "torso_z": f(row, "torso_z"),
        "vx": f(row, "vx"),
        "vz": f(row, "vz"),
        "support_contact": f(row, "support_contact"),
        "full_foot_support": f(row, "full_foot_support"),
        "body_collision": f(row, "body_collision"),
    }


def local_degradation(adapter: HopperV8Adapter, row: dict[str, str]) -> list[str]:
    types: list[str] = []
    if f(row, "body_collision") >= 0.5:
        types.append("body_collision")
    if abs(f(row, "pitch_rate")) >= 3.5 or f(row, "rotation_penalty") >= 0.5:
        types.append("rotation_risk")
    if f(row, "torso_z") <= 0.55:
        types.append("height_low")
    if f(row, "vx") <= 0.20:
        types.append("forward_speed_low")
    phase = adapter.phase(row)
    if phase in {"rear", "prone"} and not (adapter.structural_success(row) or f(row, "reward") > adapter.reward_min):
        types.append(f"entered_{phase}")
    if phase in {"rear", "prone", "upright"} and f(row, "support_contact") < 0.1:
        types.append("support_failure")
    return types


def candidate_steps(fragment: dict[str, Any], rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    end = int(fragment["end"])
    out: list[tuple[str, int]] = [("last_success", end)]
    first_step = fragment.get("degradation", {}).get("first_step")
    if isinstance(first_step, int):
        if first_step - 1 > end:
            out.append(("pre_degradation", first_step - 1))
        if 0 <= first_step < len(rows):
            out.append(("degradation_onset", first_step))
    elif end + 1 < len(rows):
        out.append(("post_success", end + 1))

    seen: set[int] = set()
    deduped: list[tuple[str, int]] = []
    for anchor, idx in out:
        if 0 <= idx < len(rows) and idx not in seen:
            seen.add(idx)
            deduped.append((anchor, idx))
    return deduped


def build_records(
    fragments: dict[str, Any],
    adapter: HopperV8Adapter,
    action_threshold: float,
    key_mode: str,
) -> list[dict[str, Any]]:
    trace_cache: dict[str, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []
    for fragment in fragments.get("fragments", []):
        trace_csv = str(fragment["trace_csv"])
        rows = trace_cache.get(trace_csv)
        if rows is None:
            rows = read_csv(Path(trace_csv))
            trace_cache[trace_csv] = rows
        for anchor_type, idx in candidate_steps(fragment, rows):
            row = rows[idx]
            next_row = rows[idx + 1] if idx + 1 < len(rows) else None
            now = score_snapshot(adapter, row)
            next_snapshot = score_snapshot(adapter, next_row) if next_row is not None else None
            continue_delta = (
                next_snapshot["continue_score"] - now["continue_score"] if next_snapshot is not None else 0.0
            )
            transition_delta = (
                next_snapshot["transition_score"] - now["transition_score"] if next_snapshot is not None else 0.0
            )
            next_degradation = local_degradation(adapter, next_row) if next_row is not None else []
            current_degradation = local_degradation(adapter, row)
            families = action_families(row, action_threshold)
            record = {
                "trace_csv": trace_csv,
                "seed": fragment.get("seed"),
                "episode": fragment.get("episode"),
                "fragment_start": fragment["start"],
                "fragment_end": fragment["end"],
                "fragment_role": fragment["role"],
                "fragment_quality_bin": fragment["quality_bin"],
                "fragment_score": fragment["score"],
                "fragment_key_legacy": fragment["key"],
                "fragment_degradation": fragment.get("degradation", {}),
                "anchor_type": anchor_type,
                "step": int(f(row, "step", idx)),
                "row_index": idx,
                "rule_key": transition_rule_key(adapter, fragment["role"], fragment["quality_bin"], row, key_mode),
                "exact_rule_key": refined_key(adapter, fragment["role"], fragment["quality_bin"], row),
                "rule_key_mode": key_mode,
                "legacy_phase": adapter.phase(row),
                "refined_phase": refined_phase(adapter, row),
                "risk_bin": adapter.risk_bin(row),
                "omega_bin": adapter.omega_bin(row),
                "vx_bin": adapter.vx_bin(row),
                "support_bin": adapter.support_bin(row),
                "action_values": action_values(row),
                "action_families": families,
                "primary_action_family": primary_action_family(row, action_threshold),
                "score": now,
                "next_score": next_snapshot,
                "continue_delta_next": continue_delta,
                "transition_delta_next": transition_delta,
                "current_degradation": current_degradation,
                "next_degradation": next_degradation,
                "next_degradation_hit": bool(next_degradation),
            }
            records.append(record)
    return records


def summarize_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["rule_key"]].append(rec)

    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items()):
        family_counts = Counter(fam for rec in vals for fam in rec["action_families"])
        primary_counts = Counter(rec["primary_action_family"] for rec in vals)
        out[key] = {
            "count": len(vals),
            "seeds": sorted({rec["seed"] for rec in vals if rec.get("seed") is not None}),
            "episodes": len({(rec.get("seed"), rec.get("episode")) for rec in vals}),
            "anchor_counts": dict(Counter(rec["anchor_type"] for rec in vals)),
            "legacy_phase_counts": dict(Counter(rec["legacy_phase"] for rec in vals)),
            "refined_phase_counts": dict(Counter(rec["refined_phase"] for rec in vals)),
            "action_family_counts": dict(family_counts.most_common()),
            "primary_action_family_counts": dict(primary_counts.most_common()),
            "mean_continue_delta_next": mean_or_zero(rec["continue_delta_next"] for rec in vals),
            "mean_transition_delta_next": mean_or_zero(rec["transition_delta_next"] for rec in vals),
            "next_degradation_rate": mean_or_zero(1.0 if rec["next_degradation_hit"] else 0.0 for rec in vals),
            "mean_continue_score": mean_or_zero(rec["score"]["continue_score"] for rec in vals),
            "mean_transition_score": mean_or_zero(rec["score"]["transition_score"] for rec in vals),
        }
    return out


def induce_observational_candidates(
    records: list[dict[str, Any]],
    min_support: int,
    min_seeds: int,
    prefer_min_delta: float,
    avoid_max_delta: float,
    avoid_min_degradation_rate: float,
    prefer_max_degradation_rate: float,
) -> list[dict[str, Any]]:
    causal_anchor_types = {"last_success", "pre_degradation", "post_success"}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["anchor_type"] not in causal_anchor_types:
            continue
        family = rec["primary_action_family"]
        if family.endswith("_weak"):
            continue
        grouped[(rec["rule_key"], family)].append(rec)

    candidates: list[dict[str, Any]] = []
    for (key, family), vals in sorted(grouped.items()):
        support = len(vals)
        seed_count = len({rec["seed"] for rec in vals if rec.get("seed") is not None})
        if support < min_support or seed_count < min_seeds:
            continue
        mean_continue_delta = mean([rec["continue_delta_next"] for rec in vals])
        mean_transition_delta = mean([rec["transition_delta_next"] for rec in vals])
        next_degradation_rate = mean([1.0 if rec["next_degradation_hit"] else 0.0 for rec in vals])
        role = key.split("|", 1)[0]

        intervention: str | None = None
        evidence_reason: list[str] = []
        if mean_continue_delta <= avoid_max_delta or next_degradation_rate >= avoid_min_degradation_rate:
            intervention = "avoid_action_family"
            if mean_continue_delta <= avoid_max_delta:
                evidence_reason.append("continue_score_dropped")
            if next_degradation_rate >= avoid_min_degradation_rate:
                evidence_reason.append("next_degradation_frequent")
        elif role == "continue" and mean_continue_delta >= prefer_min_delta and next_degradation_rate <= prefer_max_degradation_rate:
            intervention = "soft_prefer_action_family"
            evidence_reason.append("continue_score_preserved")
        elif role == "transition" and mean_transition_delta >= prefer_min_delta and next_degradation_rate <= prefer_max_degradation_rate:
            intervention = "soft_prefer_action_family"
            evidence_reason.append("transition_score_improved")

        if intervention is None:
            continue
        candidates.append(
            {
                "id": f"{key}->{intervention.replace('_action_family', '')}:{family}",
                "rule_key": key,
                "rule_role": role,
                "quality_bin": key.split("|")[1] if "|" in key else "",
                "action_family": family,
                "intervention_type": intervention,
                "support": support,
                "seeds_covered": seed_count,
                "episodes_covered": len({(rec.get("seed"), rec.get("episode")) for rec in vals}),
                "mean_continue_delta_next": mean_continue_delta,
                "mean_transition_delta_next": mean_transition_delta,
                "next_degradation_rate": next_degradation_rate,
                "status": "offline_observational_candidate",
                "evidence_kind": "trace_next_step_score_delta",
                "action_family_source": "primary_action_family",
                "candidate_anchor_types": sorted(causal_anchor_types),
                "evidence_reason": evidence_reason,
                "example_records": [
                    {
                        "trace_csv": rec["trace_csv"],
                        "seed": rec["seed"],
                        "episode": rec["episode"],
                        "step": rec["step"],
                        "anchor_type": rec["anchor_type"],
                        "continue_delta_next": rec["continue_delta_next"],
                        "transition_delta_next": rec["transition_delta_next"],
                        "next_degradation": rec["next_degradation"],
                    }
                    for rec in sorted(
                        vals,
                        key=lambda item: (
                            bool(item["next_degradation_hit"]),
                            -item["continue_delta_next"],
                            -item["transition_delta_next"],
                        ),
                        reverse=True,
                    )[:5]
                ],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["seeds_covered"],
            item["support"],
            item["next_degradation_rate"],
            -item["mean_continue_delta_next"],
        ),
        reverse=True,
    )
    return candidates


def write_markdown(path: Path, payload: dict[str, Any], max_candidates: int, max_groups: int) -> None:
    lines = [
        "# Hopper v8 Transition Action Evidence",
        "",
        f"Records: `{payload['num_records']}`",
        f"Rule-key groups: `{payload['num_rule_key_groups']}`",
        f"Observational candidates: `{payload['num_candidates']}`",
        "",
        "## Phase Refinement",
        "",
        "The report keeps legacy SD phases in each record, but v8 rule keys use refined near-good phases.",
        "This prevents positive-reward contact/flight fragments from being treated as ordinary prone failures.",
        "",
        "## Top Refined Rule Keys",
        "",
    ]
    for key, stats in sorted(payload["groups"].items(), key=lambda item: item[1]["count"], reverse=True)[:max_groups]:
        lines.extend(
            [
                f"### `{key}`",
                "",
                f"- Count: `{stats['count']}`",
                f"- Seeds: `{stats['seeds']}`",
                f"- Anchors: `{stats['anchor_counts']}`",
                f"- Legacy phases: `{stats['legacy_phase_counts']}`",
                f"- Refined phases: `{stats['refined_phase_counts']}`",
                f"- Primary action families: `{stats['primary_action_family_counts']}`",
                f"- Mean continue / transition delta next: `{stats['mean_continue_delta_next']:.4f}` / `{stats['mean_transition_delta_next']:.4f}`",
                f"- Next degradation rate: `{stats['next_degradation_rate']:.3f}`",
                "",
            ]
        )
    lines.extend(["", "## Candidate Rules", ""])
    for item in payload["candidates"][:max_candidates]:
        lines.extend(
            [
                f"### `{item['id']}`",
                "",
                f"- Status: `{item['status']}`",
                f"- Intervention: `{item['intervention_type']}`",
                f"- Support / seeds / episodes: `{item['support']}` / `{item['seeds_covered']}` / `{item['episodes_covered']}`",
                f"- Mean continue / transition delta next: `{item['mean_continue_delta_next']:.4f}` / `{item['mean_transition_delta_next']:.4f}`",
                f"- Next degradation rate: `{item['next_degradation_rate']:.3f}`",
                f"- Evidence reason: `{item['evidence_reason']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--rule-key-mode", choices=["phase_any", "exact"], default="phase_any")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--prefer-min-delta", type=float, default=0.02)
    parser.add_argument("--avoid-max-delta", type=float, default=-0.03)
    parser.add_argument("--avoid-min-degradation-rate", type=float, default=0.50)
    parser.add_argument("--prefer-max-degradation-rate", type=float, default=0.25)
    parser.add_argument("--max-markdown-candidates", type=int, default=30)
    parser.add_argument("--max-markdown-groups", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fragments = load_json(args.fragments)
    cfg = fragments.get("config", {})
    adapter = HopperV8Adapter(
        torso_z_min=float(cfg.get("torso_z_min", 0.75)),
        abs_pitch_max=float(cfg.get("abs_pitch_max", 0.45)),
        vx_min=float(cfg.get("vx_min", 0.80)),
        abs_omega_max=float(cfg.get("abs_omega_max", 3.50)),
        dense_pose_min=float(cfg.get("dense_pose_min", 0.65)),
        reward_min=float(cfg.get("reward_min", 0.0)),
    )
    records = build_records(fragments, adapter, args.action_threshold, args.rule_key_mode)
    groups = summarize_groups(records)
    candidates = induce_observational_candidates(
        records=records,
        min_support=args.min_support,
        min_seeds=args.min_seeds,
        prefer_min_delta=args.prefer_min_delta,
        avoid_max_delta=args.avoid_max_delta,
        avoid_min_degradation_rate=args.avoid_min_degradation_rate,
        prefer_max_degradation_rate=args.prefer_max_degradation_rate,
    )
    payload = {
        "evidence_family": "hopper-v8-transition-action-observational-evidence",
        "source_fragments": str(args.fragments),
        "num_records": len(records),
        "num_rule_key_groups": len(groups),
        "num_candidates": len(candidates),
        "config": {
            "action_threshold": args.action_threshold,
            "rule_key_mode": args.rule_key_mode,
            "min_support": args.min_support,
            "min_seeds": args.min_seeds,
            "prefer_min_delta": args.prefer_min_delta,
            "avoid_max_delta": args.avoid_max_delta,
            "avoid_min_degradation_rate": args.avoid_min_degradation_rate,
            "prefer_max_degradation_rate": args.prefer_max_degradation_rate,
        },
        "groups": groups,
        "candidates": candidates,
        "records": records,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_candidates, args.max_markdown_groups)
    print(f"wrote transition action evidence: {args.output}")
    print(f"records: {payload['num_records']} groups: {payload['num_rule_key_groups']}")
    print(f"observational candidates: {payload['num_candidates']}")


if __name__ == "__main__":
    main()
