#!/usr/bin/env python3
"""Validate and weaken Hopper v8 transition rules on held-out trace records.

The validation split is deterministic at the episode level. This is not a
formal train/test proof, because the current v8 candidates were mined from the
same final-eval pool; it is a conservative diagnostic layer before any online
rule-library integration.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import load_json, mean_or_zero, write_json


DEFAULT_RULES = Path("results/sd_retrospective_rule_induction/rules/hopper_v8_transition_rules_200k_v72.json")
DEFAULT_OBSERVATIONAL = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.json")
DEFAULT_COUNTERFACTUALS = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.json")
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_rule_validation_200k_v72.json")


def joint_name(action_family: str) -> str:
    return action_family.split("_", 1)[0] if "_" in action_family else action_family


def heldout_record(rec: dict[str, Any], modulus: int, remainder: int) -> bool:
    episode = rec.get("episode")
    if episode is None:
        return False
    return int(episode) % modulus == remainder


def evidence_strength(rule: dict[str, Any], validation: dict[str, Any]) -> float:
    support_term = min(float(validation.get("support", 0)) / 20.0, 1.0)
    seed_term = min(float(validation.get("seeds_covered", 0)) / 3.0, 1.0)
    if rule["intervention_type"] == "soft_prefer_action_family":
        adv_term = min(max(float(validation.get("mean_advantage_over_actual", 0.0)), 0.0) / 0.08, 1.0)
        return 0.25 * support_term + 0.15 * seed_term + 0.60 * adv_term

    degradation_term = min(max(float(validation.get("next_degradation_rate", 0.0)), 0.0), 1.0)
    continue_drop = min(max(-float(validation.get("mean_continue_delta_next", 0.0)), 0.0) / 0.15, 1.0)
    transition_drop = min(max(-float(validation.get("mean_transition_delta_next", 0.0)), 0.0) / 0.15, 1.0)
    return (
        0.25 * support_term
        + 0.15 * seed_term
        + 0.35 * degradation_term
        + 0.15 * continue_drop
        + 0.10 * transition_drop
    )


def validate_observational_rule(
    rule: dict[str, Any],
    records: list[dict[str, Any]],
    episode_modulus: int,
    episode_remainder: int,
    min_support: int,
    min_seeds: int,
    avoid_max_delta: float,
    avoid_min_degradation_rate: float,
) -> dict[str, Any]:
    matched = [
        rec
        for rec in records
        if rec.get("rule_key") == rule["rule_key"]
        and rec.get("primary_action_family") == rule["action_family"]
        and rec.get("anchor_type") in {"last_success", "pre_degradation", "post_success"}
        and heldout_record(rec, episode_modulus, episode_remainder)
    ]
    support = len(matched)
    seeds = sorted({rec.get("seed") for rec in matched if rec.get("seed") is not None})
    mean_continue_delta = mean_or_zero(float(rec.get("continue_delta_next", 0.0)) for rec in matched)
    mean_transition_delta = mean_or_zero(float(rec.get("transition_delta_next", 0.0)) for rec in matched)
    next_degradation_rate = mean_or_zero(1.0 if rec.get("next_degradation_hit") else 0.0 for rec in matched)
    passes = (
        support >= min_support
        and len(seeds) >= min_seeds
        and (mean_continue_delta <= avoid_max_delta or next_degradation_rate >= avoid_min_degradation_rate)
    )
    return {
        "support": support,
        "seeds_covered": len(seeds),
        "seed_ids": seeds,
        "mean_continue_delta_next": mean_continue_delta,
        "mean_transition_delta_next": mean_transition_delta,
        "next_degradation_rate": next_degradation_rate,
        "passes_heldout": passes,
    }


def validate_counterfactual_rule(
    rule: dict[str, Any],
    records: list[dict[str, Any]],
    episode_modulus: int,
    episode_remainder: int,
    min_support: int,
    min_seeds: int,
    min_advantage: float,
) -> dict[str, Any]:
    matched = [
        rec
        for rec in records
        if rec.get("rule_key") == rule["rule_key"]
        and rec.get("best_action_family") == rule["action_family"]
        and heldout_record(rec, episode_modulus, episode_remainder)
    ]
    support = len(matched)
    seeds = sorted({rec.get("seed") for rec in matched if rec.get("seed") is not None})
    mean_advantage = mean_or_zero(float(rec.get("advantage_over_actual", 0.0)) for rec in matched)
    passes = support >= min_support and len(seeds) >= min_seeds and mean_advantage >= min_advantage
    return {
        "support": support,
        "seeds_covered": len(seeds),
        "seed_ids": seeds,
        "mean_advantage_over_actual": mean_advantage,
        "passes_heldout": passes,
    }


def base_status(rule: dict[str, Any], validation: dict[str, Any]) -> str:
    if validation["support"] <= 0:
        return "heldout_no_support"
    if validation["passes_heldout"]:
        return "heldout_supported"
    return "heldout_weak"


def resolve_conflicts(rules: list[dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        groups[(rule["rule_key"], joint_name(rule["action_family"]))].append(rule)

    out: list[dict[str, Any]] = []
    for group_rules in groups.values():
        supported = [r for r in group_rules if r["validation_status"] == "heldout_supported"]
        if len(supported) <= 1:
            out.extend(group_rules)
            continue
        supported = sorted(supported, key=lambda r: r["heldout_strength"], reverse=True)
        winner = supported[0]
        runner_up = supported[1]
        unresolved = winner["heldout_strength"] - runner_up["heldout_strength"] < margin
        for rule in group_rules:
            if rule not in supported:
                out.append(rule)
                continue
            if unresolved:
                rule = dict(rule)
                rule["validation_status"] = "heldout_conflict_unresolved"
                rule["conflict_winner_id"] = winner["id"]
                out.append(rule)
            elif rule["id"] == winner["id"]:
                out.append(rule)
            else:
                rule = dict(rule)
                rule["validation_status"] = "heldout_weakened_by_conflict"
                rule["conflict_winner_id"] = winner["id"]
                out.append(rule)
    return out


def validate_rules(
    rules_payload: dict[str, Any],
    observational: dict[str, Any],
    counterfactuals: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    validated: list[dict[str, Any]] = []
    obs_records = observational.get("records", [])
    cf_records = counterfactuals.get("records", [])

    for rule in rules_payload.get("rules", []):
        if rule["intervention_type"] == "avoid_action_family":
            validation = validate_observational_rule(
                rule=rule,
                records=obs_records,
                episode_modulus=args.heldout_episode_modulus,
                episode_remainder=args.heldout_episode_remainder,
                min_support=args.min_support,
                min_seeds=args.min_seeds,
                avoid_max_delta=args.avoid_max_delta,
                avoid_min_degradation_rate=args.avoid_min_degradation_rate,
            )
        else:
            validation = validate_counterfactual_rule(
                rule=rule,
                records=cf_records,
                episode_modulus=args.heldout_episode_modulus,
                episode_remainder=args.heldout_episode_remainder,
                min_support=args.min_support,
                min_seeds=args.min_seeds,
                min_advantage=args.min_advantage,
            )
        item = dict(rule)
        item["heldout_validation"] = validation
        item["heldout_strength"] = evidence_strength(rule, validation)
        item["validation_status"] = base_status(rule, validation)
        validated.append(item)

    validated = resolve_conflicts(validated, args.conflict_margin)
    validated.sort(
        key=lambda r: (
            r["validation_status"] == "heldout_supported",
            r["heldout_strength"],
            r["heldout_validation"]["support"],
        ),
        reverse=True,
    )
    return {
        "validation_family": "hopper-v8-transition-rules-heldout-v0",
        "source_rules": str(args.rules),
        "source_observational": str(args.observational),
        "source_counterfactuals": str(args.counterfactuals),
        "heldout_split": {
            "episode_modulus": args.heldout_episode_modulus,
            "episode_remainder": args.heldout_episode_remainder,
        },
        "thresholds": {
            "min_support": args.min_support,
            "min_seeds": args.min_seeds,
            "min_advantage": args.min_advantage,
            "avoid_max_delta": args.avoid_max_delta,
            "avoid_min_degradation_rate": args.avoid_min_degradation_rate,
            "conflict_margin": args.conflict_margin,
        },
        "num_rules": len(validated),
        "status_counts": dict(Counter(rule["validation_status"] for rule in validated)),
        "intervention_counts": dict(Counter(rule["intervention_type"] for rule in validated)),
        "rules": validated,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_rules: int) -> None:
    lines = [
        "# Hopper v8 Transition Rule Held-out Validation",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Status counts: `{payload['status_counts']}`",
        f"Intervention counts: `{payload['intervention_counts']}`",
        f"Held-out split: `{payload['heldout_split']}`",
        "",
    ]
    for rule in payload["rules"][:max_rules]:
        val = rule["heldout_validation"]
        lines.extend(
            [
                f"## `{rule['id']}`",
                "",
                f"- Validation status: `{rule['validation_status']}`",
                f"- Original status: `{rule.get('status')}`",
                f"- Intervention: `{rule['intervention_type']}`",
                f"- Held-out support / seeds: `{val['support']}` / `{val['seeds_covered']}`",
                f"- Held-out strength: `{rule['heldout_strength']:.4f}`",
                f"- Conflicts: `{rule.get('conflicts', [])}`",
                "",
            ]
        )
        if rule["intervention_type"] == "soft_prefer_action_family":
            lines.append(f"- Mean advantage: `{val.get('mean_advantage_over_actual', 0.0):.4f}`")
        else:
            lines.append(
                f"- Mean continue / transition delta: `{val.get('mean_continue_delta_next', 0.0):.4f}` / `{val.get('mean_transition_delta_next', 0.0):.4f}`"
            )
            lines.append(f"- Next degradation rate: `{val.get('next_degradation_rate', 0.0):.3f}`")
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--observational", type=Path, default=DEFAULT_OBSERVATIONAL)
    parser.add_argument("--counterfactuals", type=Path, default=DEFAULT_COUNTERFACTUALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heldout-episode-modulus", type=int, default=3)
    parser.add_argument("--heldout-episode-remainder", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--min-advantage", type=float, default=0.02)
    parser.add_argument("--avoid-max-delta", type=float, default=-0.03)
    parser.add_argument("--avoid-min-degradation-rate", type=float, default=0.50)
    parser.add_argument("--conflict-margin", type=float, default=0.10)
    parser.add_argument("--max-markdown-rules", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_rules(
        rules_payload=load_json(args.rules),
        observational=load_json(args.observational),
        counterfactuals=load_json(args.counterfactuals),
        args=args,
    )
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_rules)
    print(f"wrote v8 transition validation: {args.output}")
    print(f"rules: {payload['num_rules']} statuses: {payload['status_counts']}")


if __name__ == "__main__":
    main()
