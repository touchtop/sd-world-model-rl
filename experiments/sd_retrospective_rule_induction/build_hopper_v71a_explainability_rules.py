#!/usr/bin/env python3
"""Build an eval-only Hopper rule library from explainability-v0 scores."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPLAINABILITY = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def conflict_group(substate_key: str, action_family: str) -> str:
    joint = action_family.split("_", 1)[0] if "_" in action_family else action_family
    return f"{substate_key}|joint:{joint}"


def opposite_family(action_family: str) -> str:
    if action_family.endswith("_pos"):
        return action_family[:-4] + "_neg"
    if action_family.endswith("_neg"):
        return action_family[:-4] + "_pos"
    return ""


def build_library(
    explainability: dict[str, Any],
    *,
    min_score: float,
    min_hits: float,
    max_rules: int,
    scale_denominator: float,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for row in explainability.get("rules", []):
        score = float(row.get("explainability_v0", 0.0) or 0.0)
        hits = float(row.get("raw", {}).get("observed_hit_steps", 0.0) or 0.0)
        if score < min_score or hits < min_hits:
            continue
        substate_key = str(row.get("substate_key", ""))
        action_family = str(row.get("action_family", ""))
        if not substate_key or "_" not in action_family:
            continue
        group = conflict_group(substate_key, action_family)
        if group in seen_groups:
            continue
        seen_groups.add(group)
        scale = max(0.0, min(1.0, score / scale_denominator))
        selected.append(
            {
                "id": f"{substate_key}->prefer:{action_family}",
                "source": "v4_substate",
                "status": "explain_active",
                "confidence_status": "explain_active",
                "intervention_type": "soft_prefer_action_family",
                "state_event": substate_key.split("|", 1)[0],
                "substate_key": substate_key,
                "action_family": action_family,
                "opposite_action_family": opposite_family(action_family),
                "conflict_group": group,
                "confidence": scale,
                "effective_prior_scale": scale,
                "explainability_v0": score,
                "explainability_components": row.get("components", {}),
                "explainability_raw": row.get("raw", {}),
                "previous_status": row.get("status"),
                "previous_source": row.get("source"),
                "sources_seen": row.get("sources_seen", []),
            }
        )
        if len(selected) >= max_rules:
            break

    return {
        "library_version": "hopper_v71a_explainability_v0_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_explainability_report": str(DEFAULT_EXPLAINABILITY.relative_to(REPO_ROOT)),
        "online_enabled": False,
        "selection": {
            "min_score": min_score,
            "min_observed_hit_steps": min_hits,
            "max_rules": max_rules,
            "scale_denominator": scale_denominator,
            "policy": "ranked_by_explainability_v0_one_rule_per_conflict_group",
        },
        "status_meaning": {
            "explain_active": "selected by explainability-v0 audit for eval-only rule-prior comparison",
        },
        "rules": selected,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper v7.1a Explainability-v0 Eval Rule Library",
        "",
        f"Rules: `{len(payload['rules'])}`",
        "",
        "| rank | rule | scale | expl | previous status | hits | cf_n |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for idx, rule in enumerate(payload["rules"], start=1):
        raw = rule.get("explainability_raw", {})
        lines.append(
            f"| {idx} | `{rule['id']}` | {rule['effective_prior_scale']:.3f} | "
            f"{rule['explainability_v0']:.3f} | {rule.get('previous_status')} | "
            f"{float(raw.get('observed_hit_steps', 0.0) or 0.0):.0f} | "
            f"{float(raw.get('cf_count', 0.0) or 0.0):.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explainability", type=Path, default=DEFAULT_EXPLAINABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--min-hits", type=float, default=100.0)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--scale-denominator", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explainability = load_json(args.explainability)
    payload = build_library(
        explainability,
        min_score=args.min_score,
        min_hits=args.min_hits,
        max_rules=args.max_rules,
        scale_denominator=args.scale_denominator,
    )
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(args.output)
    print(f"rules={len(payload['rules'])}")


if __name__ == "__main__":
    main()
