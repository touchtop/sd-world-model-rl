#!/usr/bin/env python3
"""Replay reliable rules on selected traces to estimate correction and false-intervention rates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ATTRIBUTIONS,
    DEFAULT_REPLAY_SCORING_V4,
    DEFAULT_RULES_V3,
    f,
    load_json,
    read_csv,
    resolve_workers,
    row_to_action,
    write_json,
)


_WORKER_REPLAY_RULES: list[dict[str, Any]] = []


def family_from_action(action: list[float]) -> str:
    names = ["waist", "hip", "knee", "ankle"]
    idx = max(range(len(action)), key=lambda i: abs(action[i]))
    sign = "pos" if action[idx] >= 0 else "neg"
    if abs(action[idx]) < 0.2:
        return "weak"
    return f"{names[idx]}_{sign}"


def event_hit(rule: dict[str, Any], row: dict[str, str]) -> bool:
    event = rule["state_event"]
    if event == "unsupported_high_torque":
        return max(abs(x) for x in row_to_action(row)) >= 0.65 and f(row, "full_foot_support") < 0.20 and f(row, "foot_grounded") < 0.50
    if event == "rotation_risk":
        return f(row, "rotation_penalty") >= 0.50 or abs(f(row, "pitch_rate")) >= 3.0
    if event == "prone_or_rear_trap":
        return f(row, "grounded_prone") >= 0.50 or f(row, "grounded_rear") >= 0.50
    return False


def init_replay_worker(rules: list[dict[str, Any]]) -> None:
    global _WORKER_REPLAY_RULES
    _WORKER_REPLAY_RULES = rules


def empty_stats() -> dict[str, Any]:
    return {
            "hits": 0,
            "aligned": 0,
            "opposed": 0,
            "unrelated": 0,
            "labels": Counter(),
            "aligned_labels": Counter(),
            "opposed_labels": Counter(),
            "seeds": Counter(),
            "cycles": Counter(),
    }


def merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ("hits", "aligned", "opposed", "unrelated"):
        dst[key] += src[key]
    for key in ("labels", "aligned_labels", "opposed_labels", "seeds", "cycles"):
        dst[key].update(src[key])


def score_episode_replay(ep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats_by_rule: dict[str, dict[str, Any]] = defaultdict(empty_stats)
    rows = read_csv(Path(ep["trace_csv"]))
    for row in rows:
        actual_family: str | None = None
        for rule in _WORKER_REPLAY_RULES:
            if not event_hit(rule, row):
                continue
            rid = rule["id"]
            stats = stats_by_rule[rid]
            stats["hits"] += 1
            stats["labels"][ep["label"]] += 1
            stats["seeds"][str(ep["seed"])] += 1
            stats["cycles"][str(ep["cycle_step"])] += 1
            if actual_family is None:
                actual_family = family_from_action(row_to_action(row))
            preferred = rule["action_family"]
            preferred_joint = preferred.split("_")[0]
            preferred_sign = preferred.split("_")[1] if "_" in preferred else ""
            if actual_family == preferred:
                stats["aligned"] += 1
                stats["aligned_labels"][ep["label"]] += 1
            elif actual_family.startswith(preferred_joint + "_") and not actual_family.endswith(preferred_sign):
                stats["opposed"] += 1
                stats["opposed_labels"][ep["label"]] += 1
            else:
                stats["unrelated"] += 1
    return dict(stats_by_rule)


def score_replay(attributions: dict[str, Any], rules_payload: dict[str, Any], workers: int = 0) -> dict[str, Any]:
    rules = list(rules_payload["rules"])
    worker_count = resolve_workers(workers)
    accum: dict[str, dict[str, Any]] = {rule["id"]: empty_stats() for rule in rules}
    episodes = list(attributions["episodes"])
    if worker_count <= 1 or len(episodes) <= 1:
        init_replay_worker(rules)
        for ep in episodes:
            for rid, stats in score_episode_replay(ep).items():
                merge_stats(accum[rid], stats)
    else:
        chunksize = max(1, len(episodes) // (worker_count * 4))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_replay_worker,
            initargs=(rules,),
        ) as pool:
            for episode_stats in pool.map(score_episode_replay, episodes, chunksize=chunksize):
                for rid, stats in episode_stats.items():
                    merge_stats(accum[rid], stats)

    rule_stats: dict[str, Any] = {}
    for rule in rules:
        stats = accum[rule["id"]]
        preferred = rule["action_family"]
        hits = stats["hits"]
        rule_stats[rule["id"]] = {
            "rule": rule,
            "hits": hits,
            "aligned": stats["aligned"],
            "opposed": stats["opposed"],
            "unrelated": stats["unrelated"],
            "alignment_rate": stats["aligned"] / hits if hits else 0.0,
            "opposed_rate": stats["opposed"] / hits if hits else 0.0,
            "label_counts": dict(stats["labels"]),
            "aligned_label_counts": dict(stats["aligned_labels"]),
            "opposed_label_counts": dict(stats["opposed_labels"]),
            "seed_counts": dict(stats["seeds"]),
            "cycle_counts": dict(stats["cycles"]),
        }
    return {
        "rule_family": "offline-replay-rule-scoring-v4",
        "num_rules": len(rules_payload["rules"]),
        "workers": worker_count,
        "rules": rule_stats,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Hopper Replay Rule Scoring v4", "", f"Rules: `{payload['num_rules']}`", ""]
    for rule_id, stats in payload["rules"].items():
        lines += [
            f"## {rule_id}",
            "",
            f"- Hits: `{stats['hits']}`",
            f"- Alignment rate: `{stats['alignment_rate']:.3f}`",
            f"- Opposed rate: `{stats['opposed_rate']:.3f}`",
            f"- Label counts: `{stats['label_counts']}`",
            f"- Opposed label counts: `{stats['opposed_label_counts']}`",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions", type=Path, default=DEFAULT_ATTRIBUTIONS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_V3)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPLAY_SCORING_V4)
    parser.add_argument("--workers", type=int, default=0, help="CPU worker processes; 0 uses SD_RETRO_WORKERS or a bounded auto default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = score_replay(load_json(args.attributions), load_json(args.rules), args.workers)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote replay scoring: {args.output}")


if __name__ == "__main__":
    main()
