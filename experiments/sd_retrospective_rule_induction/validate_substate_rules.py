#!/usr/bin/env python3
"""Replay v4 substate rules on selected traces and estimate false-intervention risk."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ATTRIBUTIONS,
    DEFAULT_SUBSTATE_RULES_V4,
    DEFAULT_SUBSTATE_VALIDATION_V5,
    f,
    load_json,
    read_csv,
    resolve_workers,
    row_to_action,
    write_json,
)
from analyze_substate_rules import substate_key
from replay_rule_scoring import family_from_action


_WORKER_RULES_BY_SUBSTATE: dict[str, list[dict[str, Any]]] = {}
_WORKER_STATE_EVENTS: list[str] = []


def rule_matches(rule: dict[str, Any], row: dict[str, str], prev_row: dict[str, str] | None = None) -> bool:
    fake_record = {"event": rule["state_event"]}
    return substate_key(fake_record, row, prev_row) == rule["substate_key"]


def init_substate_worker(rules: list[dict[str, Any]]) -> None:
    global _WORKER_RULES_BY_SUBSTATE, _WORKER_STATE_EVENTS
    rules_by_substate: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        rules_by_substate.setdefault(str(rule["substate_key"]), []).append(rule)
    _WORKER_RULES_BY_SUBSTATE = rules_by_substate
    _WORKER_STATE_EVENTS = sorted({str(rule["state_event"]) for rule in rules})


def init_stats(rule: dict[str, Any]) -> dict[str, Any]:
    preferred = rule["action_family"]
    return {
        "preferred": preferred,
        "preferred_joint": preferred.split("_")[0],
        "preferred_sign": preferred.split("_")[1] if "_" in preferred else "",
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


def validate_episode(ep: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    trace_csv = str(ep["trace_csv"])
    rows = read_csv(Path(trace_csv))
    stats_by_rule: dict[str, dict[str, Any]] = {}
    prev_row: dict[str, str] | None = None
    for row in rows:
        matched_rules: list[dict[str, Any]] = []
        for event in _WORKER_STATE_EVENTS:
            key = substate_key({"event": event}, row, prev_row)
            matched_rules.extend(_WORKER_RULES_BY_SUBSTATE.get(key, []))
        prev_row = row
        if not matched_rules:
            continue
        actual_family = family_from_action(row_to_action(row))
        for rule in matched_rules:
            stats = stats_by_rule.setdefault(rule["id"], init_stats(rule))
            stats["hits"] += 1
            stats["labels"][ep["label"]] += 1
            stats["seeds"][str(ep["seed"])] += 1
            stats["cycles"][str(ep["cycle_step"])] += 1
            if actual_family == stats["preferred"]:
                stats["aligned"] += 1
                stats["aligned_labels"][ep["label"]] += 1
            elif actual_family.startswith(stats["preferred_joint"] + "_") and not actual_family.endswith(stats["preferred_sign"]):
                stats["opposed"] += 1
                stats["opposed_labels"][ep["label"]] += 1
            else:
                stats["unrelated"] += 1
    return trace_csv, stats_by_rule


def validate(attributions: dict[str, Any], rules_payload: dict[str, Any], workers: int = 0) -> dict[str, Any]:
    rules = list(rules_payload["rules"])
    state_events = sorted({str(rule["state_event"]) for rule in rules})
    stats_by_rule: dict[str, dict[str, Any]] = {rule["id"]: init_stats(rule) for rule in rules}
    episodes = list(attributions["episodes"])
    worker_count = resolve_workers(workers)
    cached_traces: set[str] = set()
    if worker_count <= 1 or len(episodes) <= 1:
        init_substate_worker(rules)
        episode_results = [validate_episode(ep) for ep in episodes]
    else:
        chunksize = max(1, len(episodes) // (worker_count * 4))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_substate_worker,
            initargs=(rules,),
        ) as pool:
            episode_results = list(pool.map(validate_episode, episodes, chunksize=chunksize))

    for trace_csv, rule_updates in episode_results:
        cached_traces.add(trace_csv)
        for rid, stats in rule_updates.items():
            merge_stats(stats_by_rule[rid], stats)

    results: dict[str, Any] = {}
    for rule in rules:
        stats = stats_by_rule[rule["id"]]
        hits = stats["hits"]
        good_hits = stats["labels"].get("good", 0)
        bad_hits = stats["labels"].get("bad", 0)
        results[rule["id"]] = {
            "rule": rule,
            "hits": hits,
            "aligned": stats["aligned"],
            "opposed": stats["opposed"],
            "unrelated": stats["unrelated"],
            "alignment_rate": stats["aligned"] / hits if hits else 0.0,
            "opposed_rate": stats["opposed"] / hits if hits else 0.0,
            "good_hit_rate": good_hits / hits if hits else 0.0,
            "bad_hit_rate": bad_hits / hits if hits else 0.0,
            "label_counts": dict(stats["labels"]),
            "aligned_label_counts": dict(stats["aligned_labels"]),
            "opposed_label_counts": dict(stats["opposed_labels"]),
            "seed_counts": dict(stats["seeds"]),
            "cycle_counts": dict(stats["cycles"]),
            "seeds_covered": len(stats["seeds"]),
            "cycles_covered": len(stats["cycles"]),
        }
    return {
        "rule_family": "substate-rule-replay-validation-v5",
        "num_rules": len(rules),
        "num_cached_traces": len(cached_traces),
        "num_state_events": len(state_events),
        "workers": worker_count,
        "rules": results,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Substate Rule Validation v5",
        "",
        f"Rules: `{payload['num_rules']}`",
        f"Cached traces: `{payload.get('num_cached_traces', 0)}`",
        f"State events: `{payload.get('num_state_events', 0)}`",
        "",
    ]
    for rule_id, stats in payload["rules"].items():
        lines += [
            f"## {rule_id}",
            "",
            f"- Hits: `{stats['hits']}`",
            f"- Good hit rate: `{stats['good_hit_rate']:.3f}`",
            f"- Bad hit rate: `{stats['bad_hit_rate']:.3f}`",
            f"- Alignment rate: `{stats['alignment_rate']:.3f}`",
            f"- Opposed rate: `{stats['opposed_rate']:.3f}`",
            f"- Seeds covered: `{stats['seeds_covered']}`",
            f"- Cycles covered: `{stats['cycles_covered']}`",
            f"- Label counts: `{stats['label_counts']}`",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions", type=Path, default=DEFAULT_ATTRIBUTIONS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_SUBSTATE_RULES_V4)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBSTATE_VALIDATION_V5)
    parser.add_argument("--workers", type=int, default=0, help="CPU worker processes; 0 uses SD_RETRO_WORKERS or a bounded auto default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate(load_json(args.attributions), load_json(args.rules), args.workers)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote substate validation: {args.output}")


if __name__ == "__main__":
    main()
