#!/usr/bin/env python3
"""Update rule library statuses using v4/v5 replay validation results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_REPLAY_SCORING_V4,
    DEFAULT_RULE_LIBRARY,
    DEFAULT_RULE_LIBRARY_V5,
    DEFAULT_SUBSTATE_VALIDATION_V5,
    load_json,
    write_json,
)


def classify_reliable_rule(rule: dict[str, Any], replay_stats: dict[str, Any]) -> tuple[str, list[str]]:
    stats = replay_stats.get(rule["id"], {})
    notes: list[str] = []
    if not stats:
        return "candidate", ["缺少回放统计，保持候选状态。"]
    good_hits = stats.get("label_counts", {}).get("good", 0)
    hits = stats.get("hits", 0)
    good_rate = good_hits / hits if hits else 0.0
    if good_rate > 0.20:
        notes.append("粗规则命中大量好轨迹片段，需进一步细分。")
        return "needs_split", notes
    notes.append("通过粗规则回放审查，可保留为离线可靠候选。")
    return "replay_validated", notes


def classify_substate_rule(rule: dict[str, Any], validation_stats: dict[str, Any]) -> tuple[str, list[str]]:
    stats = validation_stats.get(rule["id"], {})
    fallback_status = rule.get("status", "candidate")
    notes: list[str] = []
    if not stats:
        return fallback_status, ["缺少子状态回放统计，保持当前周期候选状态。"]
    hits = stats.get("hits", 0)
    good_rate = stats.get("good_hit_rate", 0.0)
    bad_rate = stats.get("bad_hit_rate", 0.0)
    seeds = stats.get("seeds_covered", 0)
    if hits < 20:
        notes.append("逐帧回放命中数偏低，继续保持当前周期候选状态。")
        return fallback_status, notes
    if good_rate <= 0.15 and bad_rate >= 0.60 and seeds >= 2:
        notes.append("子状态规则主要命中差轨迹，且跨 seed 覆盖较好，可升级为 replay_validated。")
        return "replay_validated", notes
    if good_rate > 0.30:
        support = int(rule.get("support", 0) or 0)
        mean_advantage = float(rule.get("mean_advantage", 0.0) or 0.0)
        if support >= 20 or mean_advantage >= 0.12:
            notes.append("子状态规则命中好轨迹比例偏高，但支持度或反事实优势较强；保留为宽规则候选，等待后续周期细分或削弱。")
            return "cycle_candidate", notes
        notes.append("子状态规则命中好轨迹比例偏高，且支持度与优势不足，暂不进入在线候选。")
        return "rejected", notes
    notes.append("子状态规则证据尚不充分，保持当前周期候选状态。")
    return fallback_status, notes


def update_library(
    library: dict[str, Any],
    replay: dict[str, Any],
    substate_validation: dict[str, Any],
) -> dict[str, Any]:
    replay_stats = replay.get("rules", {})
    validation_stats = substate_validation.get("rules", {})
    out = {
        "library_version": "offline_v5",
        "online_enabled": False,
        "status_meaning": {
            "candidate": "candidate rule; requires additional evidence",
            "cycle_candidate": "single-cycle near-threshold evidence; not eligible for online use",
            "cycle_supported": "passed single-cycle substate mining thresholds; requires cumulative validation",
            "replay_validated": "passed offline replay validation",
            "needs_split": "coarse rule has useful signal but too broad for online use",
            "rejected": "failed offline replay validation",
        },
        "rules": [],
    }
    for rule in library.get("rules", []):
        rule_out = dict(rule)
        if rule.get("source") == "v3":
            status, notes = classify_reliable_rule(rule, replay_stats)
        elif rule.get("source") == "v4_substate":
            status, notes = classify_substate_rule(rule, validation_stats)
        else:
            status, notes = rule.get("status", "candidate"), ["未知来源，保留原状态。"]
        rule_out["status"] = status
        rule_out["audit_notes"] = notes
        out["rules"].append(rule_out)
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Rule Library v5",
        "",
        f"Online enabled: `{payload['online_enabled']}`",
        f"Rules: `{len(payload['rules'])}`",
        "",
    ]
    for rule in payload["rules"]:
        lines += [
            f"## {rule['id']}",
            "",
            f"- Source: `{rule.get('source')}`",
            f"- Status: `{rule['status']}`",
            f"- Action family: `{rule.get('action_family')}`",
            f"- Notes: {' '.join(rule.get('audit_notes', []))}",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_RULE_LIBRARY)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY_SCORING_V4)
    parser.add_argument("--substate-validation", type=Path, default=DEFAULT_SUBSTATE_VALIDATION_V5)
    parser.add_argument("--output", type=Path, default=DEFAULT_RULE_LIBRARY_V5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = update_library(load_json(args.library), load_json(args.replay), load_json(args.substate_validation))
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote library v5: {args.output}")


if __name__ == "__main__":
    main()
