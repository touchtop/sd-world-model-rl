#!/usr/bin/env python3
"""Merge Hopper v8 observational and SD-counterfactual transition candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import load_json, write_json


DEFAULT_OBSERVATIONAL = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.json")
DEFAULT_COUNTERFACTUALS = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.json")
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/rules/hopper_v8_transition_rules_200k_v72.json")


def opposite_family(action_family: str) -> str | None:
    if action_family.endswith("_pos"):
        return action_family[:-4] + "_neg"
    if action_family.endswith("_neg"):
        return action_family[:-4] + "_pos"
    return None


def index_by_key_and_family(items: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        out[(str(item["rule_key"]), str(item["action_family"]))] = item
    return out


def merge_candidates(
    observational: dict[str, Any],
    counterfactuals: dict[str, Any],
    min_seeds: int,
) -> dict[str, Any]:
    obs_candidates = observational.get("candidates", [])
    cf_candidates = counterfactuals.get("candidates", [])
    obs_idx = index_by_key_and_family(obs_candidates)
    cf_idx = index_by_key_and_family(cf_candidates)

    merged: list[dict[str, Any]] = []
    all_keys = sorted(set(obs_idx) | set(cf_idx))
    for key, family in all_keys:
        obs = obs_idx.get((key, family))
        cf = cf_idx.get((key, family))
        opposite = opposite_family(family)
        obs_opposite = obs_idx.get((key, opposite)) if opposite else None
        cf_opposite = cf_idx.get((key, opposite)) if opposite else None

        if cf is not None:
            conflicts: list[str] = []
            if obs is not None:
                conflicts.append("same_family_observed_avoid")
            if obs_opposite is not None:
                conflicts.append("opposite_family_observed_avoid")
            seeds = int(cf.get("seeds_covered", 0))
            status = "offline_v8_conflict" if conflicts else "offline_v8_supported"
            if seeds < min_seeds:
                status = "offline_v8_low_seed_support"
            merged.append(
                {
                    "id": f"{key}->soft_prefer:{family}",
                    "rule_key": key,
                    "rule_role": key.split("|", 1)[0],
                    "action_family": family,
                    "intervention_type": "soft_prefer_action_family",
                    "status": status,
                    "support": cf.get("support", 0),
                    "seeds_covered": seeds,
                    "mean_advantage_over_actual": cf.get("mean_advantage_over_actual", 0.0),
                    "observational_conflict": obs is not None,
                    "opposite_observational_conflict": obs_opposite is not None,
                    "conflicts": conflicts,
                    "source": "v8_transition_sd_counterfactual",
                }
            )

        if obs is not None:
            conflicts = []
            if cf is not None:
                conflicts.append("same_family_sd_prefer")
            if cf_opposite is not None:
                conflicts.append("opposite_family_sd_prefer")
            seeds = int(obs.get("seeds_covered", 0))
            status = "offline_v8_conflict" if conflicts else "offline_v8_supported"
            if seeds < min_seeds:
                status = "offline_v8_low_seed_support"
            merged.append(
                {
                    "id": f"{key}->avoid:{family}",
                    "rule_key": key,
                    "rule_role": key.split("|", 1)[0],
                    "action_family": family,
                    "intervention_type": "avoid_action_family",
                    "status": status,
                    "support": obs.get("support", 0),
                    "seeds_covered": seeds,
                    "mean_continue_delta_next": obs.get("mean_continue_delta_next", 0.0),
                    "mean_transition_delta_next": obs.get("mean_transition_delta_next", 0.0),
                    "next_degradation_rate": obs.get("next_degradation_rate", 0.0),
                    "sd_conflict": cf is not None,
                    "opposite_sd_conflict": cf_opposite is not None,
                    "conflicts": conflicts,
                    "source": "v8_transition_observational",
                }
            )

    by_status: dict[str, int] = defaultdict(int)
    by_intervention: dict[str, int] = defaultdict(int)
    for item in merged:
        by_status[item["status"]] += 1
        by_intervention[item["intervention_type"]] += 1

    merged.sort(
        key=lambda item: (
            item["status"] != "offline_v8_supported",
            item["seeds_covered"],
            item["support"],
        ),
        reverse=True,
    )
    return {
        "rule_family": "hopper-v8-transition-rules-offline",
        "source_observational": observational.get("source_fragments"),
        "source_counterfactuals": counterfactuals.get("source_evidence"),
        "num_rules": len(merged),
        "status_counts": dict(sorted(by_status.items())),
        "intervention_counts": dict(sorted(by_intervention.items())),
        "rules": merged,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_rules: int) -> None:
    lines = [
        "# Hopper v8 Transition Rules",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Status counts: `{payload['status_counts']}`",
        f"Intervention counts: `{payload['intervention_counts']}`",
        "",
    ]
    for rule in payload["rules"][:max_rules]:
        lines.extend(
            [
                f"## `{rule['id']}`",
                "",
                f"- Status: `{rule['status']}`",
                f"- Intervention: `{rule['intervention_type']}`",
                f"- Support / seeds: `{rule['support']}` / `{rule['seeds_covered']}`",
                f"- Conflicts: `{rule.get('conflicts', [])}`",
                f"- Source: `{rule['source']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observational", type=Path, default=DEFAULT_OBSERVATIONAL)
    parser.add_argument("--counterfactuals", type=Path, default=DEFAULT_COUNTERFACTUALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--max-markdown-rules", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = merge_candidates(
        observational=load_json(args.observational),
        counterfactuals=load_json(args.counterfactuals),
        min_seeds=args.min_seeds,
    )
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_rules)
    print(f"wrote v8 transition rules: {args.output}")
    print(f"rules: {payload['num_rules']} statuses: {payload['status_counts']}")


if __name__ == "__main__":
    main()
