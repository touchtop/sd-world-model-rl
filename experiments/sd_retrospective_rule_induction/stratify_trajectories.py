#!/usr/bin/env python3
"""Stratify Hopper evaluation trajectories into good, bad, middle, and random sets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_MANIFEST,
    DEFAULT_SELECTION,
    f,
    label_episode,
    load_json,
    parse_step_dir,
    quantile,
    read_csv,
    repo_path,
    write_json,
)


def collect_episodes(repo_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        summary_sources = [
            ("official_eval", item) for item in run.get("official_eval_summaries", [])
        ] + [
            ("train_replay", item) for item in run.get("train_replay_summaries", [])
        ]
        for source, summary_info in summary_sources:
            summary_path = repo_path(repo_root, summary_info["summary_csv"])
            if summary_path is None or not summary_path.exists():
                continue
            cycle_step = parse_step_dir(summary_info["step_dir"])
            for row in read_csv(summary_path):
                reward = f(row, "episode_reward")
                trace_csv = row.get("trace_csv")
                trace_path = Path(trace_csv) if trace_csv else None
                episodes.append(
                    {
                        "seed": run["seed"],
                        "run_name": run["run_name"],
                        "run_dir": run["run_dir"],
                        "cycle_step": cycle_step,
                        "episode_idx": int(f(row, "episode_idx", -1)),
                        "episode_reward": reward,
                        "episode_success": f(row, "episode_success"),
                        "length": int(f(row, "length")),
                        "trace_csv": str(trace_path) if trace_path else None,
                        "source": source,
                        "summary_metrics": {
                            key: f(row, key)
                            for key in (
                                "final_rootx",
                                "final_pitch",
                                "max_rootx",
                                "max_torso_z",
                                "max_vx",
                                "max_vz",
                                "max_full_foot_support",
                                "falling_air_frac",
                                "grounded_rear_frac",
                                "grounded_prone_frac",
                                "grounded_upright_frac",
                                "grounded_prepared_frac",
                                "rising_air_frac",
                                "full_foot_support_frac",
                                "backward_speed_frac",
                                "rotation_penalty_frac",
                            )
                        },
                    }
                )
    return episodes


def stage_score(ep: dict[str, Any]) -> float:
    """Stage-aware score used when sparse reward cannot separate trajectories."""
    m = ep.get("summary_metrics", {})
    forward = 0.60 * float(m.get("final_rootx", 0.0)) + 0.40 * float(m.get("max_rootx", 0.0))
    support = 0.50 * float(m.get("max_full_foot_support", 0.0)) + 0.75 * float(m.get("full_foot_support_frac", 0.0))
    posture = 0.25 * float(m.get("grounded_upright_frac", 0.0)) + 0.20 * float(m.get("grounded_prepared_frac", 0.0))
    penalties = 0.75 * float(m.get("backward_speed_frac", 0.0)) + 0.75 * float(m.get("rotation_penalty_frac", 0.0))
    prone_penalty = 0.20 * float(m.get("grounded_prone_frac", 0.0)) + 0.15 * float(m.get("grounded_rear_frac", 0.0))
    return forward + support + posture - penalties - prone_penalty


def evidence_score(ep: dict[str, Any], score_mode: str) -> float:
    reward = float(ep["episode_reward"])
    stage = stage_score(ep)
    if score_mode == "reward":
        return reward
    if score_mode == "stage":
        return stage
    if score_mode == "hybrid":
        return reward + stage
    raise ValueError(f"Unknown score mode: {score_mode}")


def maybe_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    result = int(value)
    if result <= 0:
        return None
    return result


def default_memory_policy_path(output: Path) -> Path | None:
    for parent in (output.parent, *output.parents):
        path = parent / "selection_memory_policy.json"
        if path.exists():
            return path
    return None


def load_memory_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    policy_path = args.memory_policy or default_memory_policy_path(args.output)
    if policy_path is not None and policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["_policy_path"] = str(policy_path)

    if args.max_selected_episodes is not None:
        policy["max_selected_episodes"] = args.max_selected_episodes
    if args.max_selected_per_cycle_step is not None:
        policy["max_selected_per_cycle_step"] = args.max_selected_per_cycle_step
    if args.preserve_source:
        policy["preserve_sources"] = args.preserve_source

    policy["max_selected_episodes"] = maybe_int(policy.get("max_selected_episodes"))
    policy["max_selected_per_cycle_step"] = maybe_int(policy.get("max_selected_per_cycle_step"))
    preserve_sources = policy.get("preserve_sources", [])
    if isinstance(preserve_sources, str):
        preserve_sources = [x.strip() for x in preserve_sources.split(",") if x.strip()]
    policy["preserve_sources"] = sorted(set(preserve_sources))
    policy["enabled"] = bool(policy["max_selected_episodes"] or policy["max_selected_per_cycle_step"])
    return policy


def label_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(ep.get("label", "unknown")) for ep in episodes))


def cycle_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(ep.get("cycle_step", "unknown")) for ep in episodes))


def rank_extremity(ep: dict[str, Any]) -> float:
    score = float(ep.get("evidence_score", 0.0))
    label = ep.get("label")
    if label == "good":
        return max(0.0, score - float(ep.get("good_threshold", score)))
    if label == "bad":
        return max(0.0, float(ep.get("bad_threshold", score)) - score)
    return 0.0


def sort_for_memory(episodes: list[dict[str, Any]], newest_first: bool = False) -> list[dict[str, Any]]:
    label_priority = {"bad": 0, "good": 0, "middle": 1}
    return sorted(
        episodes,
        key=lambda ep: (
            -int(ep.get("cycle_step", 0)) if newest_first else int(ep.get("cycle_step", 0)),
            label_priority.get(str(ep.get("label")), 2),
            -rank_extremity(ep),
            int(ep.get("seed", 0)),
            str(ep.get("source", "")),
            str(ep.get("run_name", "")),
            int(ep.get("episode_idx", -1)),
        ),
    )


def take_ranked(episodes: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    return sort_for_memory(episodes)[:n]


def cap_cycle_step_group(
    episodes: list[dict[str, Any]],
    cap: int | None,
    preserve_sources: set[str],
) -> list[dict[str, Any]]:
    if cap is None or len(episodes) <= cap:
        return episodes

    preserved = [ep for ep in episodes if str(ep.get("source")) in preserve_sources]
    remaining = [ep for ep in episodes if str(ep.get("source")) not in preserve_sources]
    budget = max(0, cap - len(preserved))
    if budget <= 0:
        return preserved

    random_middle = [ep for ep in remaining if ep.get("label") == "middle" and ep.get("random_selected")]
    random_quota = min(len(random_middle), max(0, min(10, budget // 10)))
    kept = preserved + take_ranked(random_middle, random_quota)
    budget -= random_quota

    good = [ep for ep in remaining if ep.get("label") == "good"]
    bad = [ep for ep in remaining if ep.get("label") == "bad"]
    good_quota = budget // 2
    bad_quota = budget - good_quota
    good_keep = take_ranked(good, good_quota)
    bad_keep = take_ranked(bad, bad_quota)
    kept.extend(good_keep)
    kept.extend(bad_keep)

    unused = budget - len(good_keep) - len(bad_keep)
    if unused > 0:
        used = {id(ep) for ep in kept}
        spillover = [ep for ep in remaining if id(ep) not in used and ep.get("label") != "middle"]
        kept.extend(take_ranked(spillover, unused))
    return kept


def apply_memory_policy(
    selected: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before = list(selected)
    if not policy.get("enabled"):
        return selected, {
            "enabled": False,
            "num_selected_before_memory": len(before),
            "num_selected_after_memory": len(selected),
            "cycle_step_counts_before": cycle_counts(before),
            "cycle_step_counts_after": cycle_counts(selected),
            "label_counts_before": label_counts(before),
            "label_counts_after": label_counts(selected),
        }

    max_per_cycle_step = policy.get("max_selected_per_cycle_step")
    max_total = policy.get("max_selected_episodes")
    preserve_sources = set(policy.get("preserve_sources", []))

    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in selected:
        by_cycle[int(ep["cycle_step"])].append(ep)

    capped: list[dict[str, Any]] = []
    for _, eps in sorted(by_cycle.items()):
        capped.extend(cap_cycle_step_group(eps, max_per_cycle_step, preserve_sources))

    if max_total is not None and len(capped) > max_total:
        preserved = [ep for ep in capped if str(ep.get("source")) in preserve_sources]
        remaining = [ep for ep in capped if str(ep.get("source")) not in preserve_sources]
        budget = max(0, max_total - len(preserved))
        capped = preserved + sort_for_memory(remaining, newest_first=True)[:budget]

    capped = sort_for_memory(capped)
    return capped, {
        "enabled": True,
        "policy_path": policy.get("_policy_path"),
        "max_selected_episodes": max_total,
        "max_selected_per_cycle_step": max_per_cycle_step,
        "preserve_sources": sorted(preserve_sources),
        "num_selected_before_memory": len(before),
        "num_selected_after_memory": len(capped),
        "cycle_step_counts_before": cycle_counts(before),
        "cycle_step_counts_after": cycle_counts(capped),
        "label_counts_before": label_counts(before),
        "label_counts_after": label_counts(capped),
    }


def stratify(
    episodes: list[dict[str, Any]],
    good_quantile: float,
    bad_quantile: float,
    random_per_cycle: int,
    seed: int,
    score_mode: str,
    memory_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_cycle[int(ep["cycle_step"])].append(ep)

    selected: list[dict[str, Any]] = []
    cycle_stats: dict[str, Any] = {}
    for cycle_step, eps in sorted(by_cycle.items()):
        scores = [evidence_score(ep, score_mode) for ep in eps]
        bad_thr = quantile(scores, bad_quantile)
        good_thr = quantile(scores, good_quantile)
        random_ids = set()
        if random_per_cycle > 0 and eps:
            sample_size = min(random_per_cycle, len(eps))
            random_ids = {id(ep) for ep in rng.sample(eps, sample_size)}

        counts = {"bad": 0, "middle": 0, "good": 0, "random": 0}
        for ep in eps:
            score = evidence_score(ep, score_mode)
            label = label_episode(score, bad_thr, good_thr)
            ep_out = dict(ep)
            ep_out["label"] = label
            ep_out["label_score_mode"] = score_mode
            ep_out["stage_score"] = stage_score(ep)
            ep_out["evidence_score"] = score
            ep_out["bad_threshold"] = bad_thr
            ep_out["good_threshold"] = good_thr
            ep_out["random_selected"] = id(ep) in random_ids
            counts[label] += 1
            if ep_out["random_selected"]:
                counts["random"] += 1
            if label != "middle" or ep_out["random_selected"]:
                selected.append(ep_out)

        cycle_stats[str(cycle_step)] = {
            "num_episodes": len(eps),
            "score_mode": score_mode,
            "bad_threshold": bad_thr,
            "good_threshold": good_thr,
            "counts": counts,
        }

    selected, memory_stats = apply_memory_policy(selected, memory_policy or {})
    return {
        "selection_policy": {
            "good_quantile": good_quantile,
            "bad_quantile": bad_quantile,
            "random_per_cycle": random_per_cycle,
            "random_seed": seed,
            "score_mode": score_mode,
            "memory_policy": memory_policy or {"enabled": False},
        },
        "memory_stats": memory_stats,
        "cycle_stats": cycle_stats,
        "num_input_episodes": len(episodes),
        "num_selected_episodes": len(selected),
        "selected_episodes": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--good-quantile", type=float, default=0.80)
    parser.add_argument("--bad-quantile", type=float, default=0.20)
    parser.add_argument("--random-per-cycle", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--score-mode", choices=["reward", "stage", "hybrid"], default="reward")
    parser.add_argument(
        "--memory-policy",
        type=Path,
        default=None,
        help="Optional JSON policy for bounded retrospective selection; defaults to selection_memory_policy.json in an output parent directory.",
    )
    parser.add_argument("--max-selected-episodes", type=int, default=None)
    parser.add_argument("--max-selected-per-cycle-step", type=int, default=None)
    parser.add_argument("--preserve-source", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    repo_root = Path(manifest["repo_root"])
    episodes = collect_episodes(repo_root, manifest)
    memory_policy = load_memory_policy(args)
    payload = stratify(
        episodes,
        args.good_quantile,
        args.bad_quantile,
        args.random_per_cycle,
        args.seed,
        args.score_mode,
        memory_policy,
    )
    payload["manifest"] = str(args.manifest)
    write_json(args.output, payload)
    print(f"wrote selection: {args.output}")
    print(f"selected episodes: {payload['num_selected_episodes']} / {payload['num_input_episodes']}")


if __name__ == "__main__":
    main()
