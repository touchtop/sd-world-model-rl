#!/usr/bin/env python3
"""Validate induced Hopper rules on selected held-out seeds or all episodes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_ATTRIBUTIONS, DEFAULT_RULES, DEFAULT_VALIDATION, load_json, safe_rate, write_json


def rule_matches(rule: dict[str, Any], features: dict[str, float]) -> bool:
    cond = rule["condition"]
    value = float(features.get(cond["feature"], 0.0))
    if cond["operator"] == ">=":
        return value >= float(cond["threshold"])
    if cond["operator"] == "<=":
        return value <= float(cond["threshold"])
    raise ValueError(f"unsupported operator: {cond['operator']}")


def evaluate_subset(episodes: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "num_episodes": len(episodes),
        "label_counts": defaultdict(int),
        "risk_alerts": 0,
        "success_alerts": 0,
        "bad_with_risk_alert": 0,
        "good_with_risk_alert": 0,
        "good_with_success_alert": 0,
        "bad_with_success_alert": 0,
    }
    for ep in episodes:
        label = ep["label"]
        stats["label_counts"][label] += 1
        risk = any(rule_matches(rule, ep["features"]) for rule in rules if rule["type"] == "risk")
        success = any(rule_matches(rule, ep["features"]) for rule in rules if rule["type"] == "success")
        stats["risk_alerts"] += int(risk)
        stats["success_alerts"] += int(success)
        stats["bad_with_risk_alert"] += int(label == "bad" and risk)
        stats["good_with_risk_alert"] += int(label == "good" and risk)
        stats["good_with_success_alert"] += int(label == "good" and success)
        stats["bad_with_success_alert"] += int(label == "bad" and success)

    bad_count = stats["label_counts"]["bad"]
    good_count = stats["label_counts"]["good"]
    stats["risk_bad_recall"] = safe_rate(stats["bad_with_risk_alert"], bad_count)
    stats["risk_good_false_alert_rate"] = safe_rate(stats["good_with_risk_alert"], good_count)
    stats["success_good_recall"] = safe_rate(stats["good_with_success_alert"], good_count)
    stats["success_bad_false_accept_rate"] = safe_rate(stats["bad_with_success_alert"], bad_count)
    stats["label_counts"] = dict(stats["label_counts"])
    return stats


def validate(
    attributions: dict[str, Any],
    rules_payload: dict[str, Any],
    holdout_seeds: list[int],
) -> dict[str, Any]:
    episodes = attributions["episodes"]
    rules = rules_payload["rules"]
    holdout = [ep for ep in episodes if int(ep["seed"]) in holdout_seeds]
    train_like = [ep for ep in episodes if int(ep["seed"]) not in holdout_seeds]
    by_cycle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_cycle[str(ep["cycle_step"])].append(ep)

    return {
        "rules": rules_payload.get("rules", []),
        "holdout_seeds": holdout_seeds,
        "all": evaluate_subset(episodes, rules),
        "train_like": evaluate_subset(train_like, rules),
        "holdout": evaluate_subset(holdout, rules),
        "by_cycle": {
            cycle: evaluate_subset(eps, rules)
            for cycle, eps in sorted(by_cycle.items(), key=lambda item: int(item[0]))
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions", type=Path, default=DEFAULT_ATTRIBUTIONS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--holdout-seed", type=int, action="append", default=[3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attributions = load_json(args.attributions)
    rules_payload = load_json(args.rules)
    payload = validate(attributions, rules_payload, args.holdout_seed)
    write_json(args.output, payload)
    print(f"wrote validation: {args.output}")
    print(f"holdout episodes: {payload['holdout']['num_episodes']}")


if __name__ == "__main__":
    main()

