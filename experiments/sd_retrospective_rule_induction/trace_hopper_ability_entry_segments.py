#!/usr/bin/env python3
"""Trace Hopper ability-entry segments from retrospective multi-seed traces.

The script is intentionally offline.  It reads retrospective train traces from a
closed-loop run, detects when each seed first crosses return thresholds, and
then compares the crossing seed's local trajectory windows with same-cycle
non-crossing seeds.  The research intent is to study how a high-quality ability
first enters the replay distribution before turning it into online rules.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import f, load_json, mean_or_zero, read_csv, write_json
from mine_hopper_v8_success_segments import HopperV8Adapter
from mine_hopper_v8_transition_actions import action_families, primary_action_family, refined_phase


DEFAULT_RUN_PREFIX = "tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1"
DEFAULT_OUTPUT_DIR = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1"
)
DEFAULT_THRESHOLDS = (1.0, 10.0, 50.0, 100.0)


def parse_cycle(path: Path) -> int | None:
    match = re.search(r"_cycle(\d+)_target", str(path))
    return int(match.group(1)) if match else None


def discover_summary_paths(run_prefix: str, seeds: list[int], max_cycle: int | None) -> list[tuple[int, int, Path]]:
    paths: list[tuple[int, int, Path]] = []
    for seed in seeds:
        pattern = (
            f"logs/hopper-hop/{seed}/{run_prefix}_cycle*_target*/"
            "retrospective_train_traces/step_*/eval_episode_summary.csv"
        )
        for path in sorted(Path(".").glob(pattern)):
            cycle = parse_cycle(path)
            if cycle is None:
                continue
            if max_cycle is not None and cycle > max_cycle:
                continue
            paths.append((seed, cycle, path))
    return sorted(paths, key=lambda item: (item[1], item[0], str(item[2])))


def trace_path(row: dict[str, str]) -> Path | None:
    value = row.get("trace_csv")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.exists() else None


def episode_id(row: dict[str, str]) -> int:
    raw = row.get("episode_idx") or row.get("episode") or "-1"
    try:
        return int(float(raw))
    except ValueError:
        return -1


def load_episode_tables(
    run_prefix: str,
    seeds: list[int],
    max_cycle: int | None,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[tuple[int, int], dict[str, Any]]]:
    by_seed_cycle: dict[tuple[int, int], list[dict[str, Any]]] = {}
    cycle_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for seed, cycle, path in discover_summary_paths(run_prefix, seeds, max_cycle):
        rows: list[dict[str, Any]] = []
        for row in read_csv(path):
            reward = f(row, "episode_reward")
            rows.append(
                {
                    "seed": seed,
                    "cycle": cycle,
                    "episode": episode_id(row),
                    "episode_reward": reward,
                    "length": int(f(row, "length")),
                    "trace_csv": str(trace_path(row) or row.get("trace_csv", "")),
                    "max_rootx": f(row, "max_rootx"),
                    "max_vx": f(row, "max_vx"),
                    "max_vz": f(row, "max_vz"),
                    "max_full_foot_support": f(row, "max_full_foot_support"),
                    "falling_air_frac": f(row, "falling_air_frac"),
                    "grounded_rear_frac": f(row, "grounded_rear_frac"),
                    "grounded_prone_frac": f(row, "grounded_prone_frac"),
                    "grounded_hop_frac": f(row, "grounded_hop_frac"),
                    "full_foot_support_frac": f(row, "full_foot_support_frac"),
                    "backward_speed_frac": f(row, "backward_speed_frac"),
                    "rotation_penalty_frac": f(row, "rotation_penalty_frac"),
                }
            )
        by_seed_cycle[(seed, cycle)] = rows
        rewards = [item["episode_reward"] for item in rows]
        cycle_stats[(seed, cycle)] = {
            "seed": seed,
            "cycle": cycle,
            "num_episodes": len(rows),
            "mean_reward": mean_or_zero(rewards),
            "max_reward": max(rewards, default=0.0),
            "count_gt_1": sum(1 for value in rewards if value >= 1.0),
            "count_gt_10": sum(1 for value in rewards if value >= 10.0),
            "count_gt_50": sum(1 for value in rewards if value >= 50.0),
            "count_gt_100": sum(1 for value in rewards if value >= 100.0),
            "mean_max_rootx": mean_or_zero(item["max_rootx"] for item in rows),
            "mean_max_vx": mean_or_zero(item["max_vx"] for item in rows),
            "mean_falling_air_frac": mean_or_zero(item["falling_air_frac"] for item in rows),
            "mean_grounded_prone_frac": mean_or_zero(item["grounded_prone_frac"] for item in rows),
            "mean_rotation_penalty_frac": mean_or_zero(item["rotation_penalty_frac"] for item in rows),
        }
    return by_seed_cycle, cycle_stats


def first_crossings(
    by_seed_cycle: dict[tuple[int, int], list[dict[str, Any]]],
    seeds: list[int],
    thresholds: list[float],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cycles = sorted({cycle for _, cycle in by_seed_cycle})
    for seed in seeds:
        for threshold in thresholds:
            key = f"seed{seed}_R>{threshold:g}"
            crossing: dict[str, Any] | None = None
            for cycle in cycles:
                rows = by_seed_cycle.get((seed, cycle), [])
                hits = [row for row in rows if row["episode_reward"] >= threshold]
                if hits:
                    best = max(hits, key=lambda item: item["episode_reward"])
                    crossing = {
                        "seed": seed,
                        "threshold": threshold,
                        "cycle": cycle,
                        "best_episode": best["episode"],
                        "best_reward": best["episode_reward"],
                        "num_crossing_episodes": len(hits),
                        "trace_csv": best["trace_csv"],
                    }
                    break
            out[key] = crossing or {
                "seed": seed,
                "threshold": threshold,
                "cycle": None,
                "best_episode": None,
                "best_reward": 0.0,
                "num_crossing_episodes": 0,
                "trace_csv": None,
            }
    return out


def load_rule_audit_stats(results_root: Path, max_cycle: int | None) -> dict[tuple[int, int], dict[str, Any]]:
    stats: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(results_root.glob("cycle_*/reports/rule_effect_audit.json")):
        cycle = parse_cycle(path)
        if cycle is None:
            match = re.search(r"cycle_(\d+)", str(path))
            cycle = int(match.group(1)) if match else None
        if cycle is None or (max_cycle is not None and cycle > max_cycle):
            continue
        payload = load_json(path)
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("episode_summaries", []):
            try:
                seed = int(row.get("seed"))
            except (TypeError, ValueError):
                continue
            by_seed[seed].append(row)
        for seed, rows in by_seed.items():
            steps = sum(int(row.get("steps", 0) or 0) for row in rows)
            gate_hits = sum(int(row.get("gate_hits", 0) or 0) for row in rows)
            transition_hits = sum(int(row.get("transition_gate_hits", 0) or 0) for row in rows)
            v4_hits = sum(int(row.get("v4_gate_hits", 0) or 0) for row in rows)
            rewards = [float(row.get("episode_reward", 0.0) or 0.0) for row in rows]
            stats[(seed, cycle)] = {
                "seed": seed,
                "cycle": cycle,
                "audit_episodes": len(rows),
                "audit_reward_mean": mean_or_zero(rewards),
                "audit_reward_max": max(rewards, default=0.0),
                "rule_gate_density": gate_hits / steps if steps else 0.0,
                "transition_rule_gate_density": transition_hits / steps if steps else 0.0,
                "v4_rule_gate_density": v4_hits / steps if steps else 0.0,
                "effective_direction_frac": mean_or_zero(
                    float(row.get("effective_direction_frac", 0.0) or 0.0) for row in rows
                ),
            }
    return stats


def safe_window(rows: list[dict[str, str]], center: int, before: int, after: int) -> tuple[int, int]:
    if not rows:
        return 0, -1
    start = max(0, center - before)
    end = min(len(rows) - 1, center + after)
    return start, end


def row_signature(adapter: HopperV8Adapter, row: dict[str, str]) -> dict[str, str]:
    return {
        "omega_bin": adapter.omega_bin(row),
        "vx_bin": adapter.vx_bin(row),
        "support_bin": adapter.support_bin(row),
        "phase": adapter.phase(row),
        "refined_phase": refined_phase(adapter, row),
        "risk_bin": adapter.risk_bin(row),
    }


def window_summary(
    adapter: HopperV8Adapter,
    rows: list[dict[str, str]],
    start: int,
    end: int,
    horizon: int,
    action_threshold: float,
) -> dict[str, Any]:
    window = rows[start : end + 1] if end >= start else []
    future = rows[end + 1 : min(len(rows), end + 1 + horizon)] if window else []
    phase_counts = Counter(adapter.phase(row) for row in window)
    refined_phase_counts = Counter(refined_phase(adapter, row) for row in window)
    primary_counts = Counter(primary_action_family(row, action_threshold) for row in window)
    family_counts = Counter(family for row in window for family in action_families(row, action_threshold))
    reward_sum = sum(f(row, "reward") for row in window)
    future_reward_sum = sum(f(row, "reward") for row in future)
    rootx_start = f(window[0], "rootx") if window else 0.0
    rootx_end = f(window[-1], "rootx") if window else 0.0
    future_rootx_end = f(future[-1], "rootx") if future else rootx_end
    return {
        "start": start,
        "end": end,
        "length": len(window),
        "reward_sum": reward_sum,
        "future_reward_sum": future_reward_sum,
        "rootx_delta": rootx_end - rootx_start,
        "future_rootx_delta": future_rootx_end - rootx_end,
        "mean_vx": mean_or_zero(f(row, "vx") for row in window),
        "future_mean_vx": mean_or_zero(f(row, "vx") for row in future),
        "mean_vz": mean_or_zero(f(row, "vz") for row in window),
        "mean_abs_pitch": mean_or_zero(abs(f(row, "pitch")) for row in window),
        "mean_abs_pitch_rate": mean_or_zero(abs(f(row, "pitch_rate")) for row in window),
        "support_contact_mean": mean_or_zero(max(f(row, "support_contact"), f(row, "foot_grounded")) for row in window),
        "full_foot_support_mean": mean_or_zero(f(row, "full_foot_support") for row in window),
        "rotation_penalty_frac": mean_or_zero(1.0 if f(row, "rotation_penalty") >= 0.5 else 0.0 for row in window),
        "body_collision_frac": mean_or_zero(f(row, "body_collision") for row in window),
        "falling_frac": mean_or_zero(1.0 if adapter.phase(row) == "falling" else 0.0 for row in window),
        "prone_frac": mean_or_zero(1.0 if adapter.phase(row) == "prone" else 0.0 for row in window),
        "hop_frac": mean_or_zero(1.0 if adapter.phase(row) == "hop" else 0.0 for row in window),
        "phase_counts": dict(phase_counts.most_common()),
        "refined_phase_counts": dict(refined_phase_counts.most_common()),
        "primary_action_counts": dict(primary_counts.most_common()),
        "action_family_counts": dict(family_counts.most_common()),
        "dominant_primary_action": primary_counts.most_common(1)[0][0] if primary_counts else "none",
        "dominant_action_family": family_counts.most_common(1)[0][0] if family_counts else "none",
        "anchor_signature": row_signature(adapter, rows[(start + end) // 2]) if window else {},
    }


def anchor_indices(rows: list[dict[str, str]], window_size: int) -> list[tuple[str, int]]:
    if not rows:
        return []
    rewards = [f(row, "reward") for row in rows]
    positive = next((idx for idx, value in enumerate(rewards) if value > 0.0), None)
    best_reward = max(range(len(rows)), key=lambda idx: rewards[idx])

    best_window_start = 0
    best_window_value = float("-inf")
    for start in range(0, max(1, len(rows) - window_size + 1)):
        end = min(len(rows), start + window_size)
        value = sum(rewards[start:end]) + 0.25 * (f(rows[end - 1], "rootx") - f(rows[start], "rootx"))
        if value > best_window_value:
            best_window_value = value
            best_window_start = start
    best_window_center = min(len(rows) - 1, best_window_start + window_size // 2)

    forward = None
    for idx in range(0, max(1, len(rows) - window_size)):
        segment = rows[idx : min(len(rows), idx + window_size)]
        if mean_or_zero(f(row, "vx") for row in segment) >= 0.50 and (
            f(segment[-1], "rootx") - f(segment[0], "rootx")
        ) >= 0.08:
            forward = idx + len(segment) // 2
            break

    anchors = [
        ("first_positive_reward", positive),
        ("best_reward", best_reward),
        ("best_reward_window", best_window_center),
        ("first_sustained_forward", forward),
    ]
    seen: set[int] = set()
    out: list[tuple[str, int]] = []
    for name, idx in anchors:
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        out.append((name, idx))
    return out


def analyze_episode(
    adapter: HopperV8Adapter,
    episode: dict[str, Any],
    role: str,
    threshold: float,
    anchor_cycle: int,
    window_size: int,
    horizon: int,
    action_threshold: float,
) -> dict[str, Any] | None:
    path = Path(str(episode.get("trace_csv", "")))
    if not path.exists():
        return None
    rows = read_csv(path)
    if not rows:
        return None
    windows: list[dict[str, Any]] = []
    before = max(1, window_size // 2)
    after = max(1, window_size - before - 1)
    for anchor_type, idx in anchor_indices(rows, window_size):
        start, end = safe_window(rows, idx, before, after)
        item = window_summary(adapter, rows, start, end, horizon, action_threshold)
        item.update(
            {
                "anchor_type": anchor_type,
                "anchor_index": idx,
                "anchor_step": int(f(rows[idx], "step", idx)),
                "anchor_signature": row_signature(adapter, rows[idx]),
            }
        )
        windows.append(item)

    return {
        "role": role,
        "threshold": threshold,
        "anchor_cycle": anchor_cycle,
        "seed": episode["seed"],
        "cycle": episode["cycle"],
        "episode": episode["episode"],
        "episode_reward": episode["episode_reward"],
        "trace_csv": str(path),
        "cycle_offset": episode["cycle"] - anchor_cycle,
        "episode_summary": {
            key: episode[key]
            for key in (
                "max_rootx",
                "max_vx",
                "max_vz",
                "max_full_foot_support",
                "falling_air_frac",
                "grounded_rear_frac",
                "grounded_prone_frac",
                "grounded_hop_frac",
                "full_foot_support_frac",
                "backward_speed_frac",
                "rotation_penalty_frac",
            )
        },
        "windows": windows,
    }


def pick_episode_records(
    by_seed_cycle: dict[tuple[int, int], list[dict[str, Any]]],
    seed: int,
    cycle: int,
    threshold: float,
    top_k: int,
    prefer_crossing: bool,
) -> list[dict[str, Any]]:
    rows = list(by_seed_cycle.get((seed, cycle), []))
    if prefer_crossing:
        selected = [row for row in rows if row["episode_reward"] >= threshold]
        if selected:
            return sorted(selected, key=lambda item: item["episode_reward"], reverse=True)[:top_k]
    return sorted(rows, key=lambda item: item["episode_reward"], reverse=True)[:top_k]


def build_entry_events(
    by_seed_cycle: dict[tuple[int, int], list[dict[str, Any]]],
    crossings: dict[str, dict[str, Any]],
    seeds: list[int],
    thresholds: list[float],
    pre_cycles: int,
    post_cycles: int,
    top_k: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for threshold in thresholds:
        for seed in seeds:
            crossing = crossings[f"seed{seed}_R>{threshold:g}"]
            cycle = crossing.get("cycle")
            if cycle is None:
                continue
            cycle = int(cycle)
            success_records: list[dict[str, Any]] = []
            failure_records: list[dict[str, Any]] = []
            for target_cycle in range(max(1, cycle - pre_cycles), cycle + post_cycles + 1):
                success_records.extend(
                    pick_episode_records(by_seed_cycle, seed, target_cycle, threshold, top_k, target_cycle >= cycle)
                )
                for other_seed in seeds:
                    if other_seed == seed:
                        continue
                    other_crossing = crossings[f"seed{other_seed}_R>{threshold:g}"].get("cycle")
                    if other_crossing is not None and int(other_crossing) <= cycle:
                        continue
                    failure_records.extend(
                        pick_episode_records(
                            by_seed_cycle,
                            other_seed,
                            target_cycle,
                            threshold,
                            top_k,
                            prefer_crossing=False,
                        )
                    )
            events.append(
                {
                    "threshold": threshold,
                    "entry_seed": seed,
                    "entry_cycle": cycle,
                    "crossing": crossing,
                    "success_records": success_records,
                    "matched_failure_records": failure_records,
                }
            )
    return events


def aggregate_windows(episode_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in episode_analyses:
        for window in ep["windows"]:
            grouped[(ep["threshold"], ep["anchor_cycle"], ep["role"])].append(window)

    out: dict[str, Any] = {}
    for (threshold, cycle, role), windows in grouped.items():
        key = f"R>{threshold:g}_cycle{cycle:03d}_{role}"
        primary = Counter(window["dominant_primary_action"] for window in windows)
        family = Counter(window["dominant_action_family"] for window in windows)
        phase = Counter(
            phase_name
            for window in windows
            for phase_name, count in window.get("phase_counts", {}).items()
            for _ in range(int(count))
        )
        out[key] = {
            "threshold": threshold,
            "entry_cycle": cycle,
            "role": role,
            "num_windows": len(windows),
            "mean_reward_sum": mean_or_zero(window["reward_sum"] for window in windows),
            "mean_future_reward_sum": mean_or_zero(window["future_reward_sum"] for window in windows),
            "mean_rootx_delta": mean_or_zero(window["rootx_delta"] for window in windows),
            "mean_future_rootx_delta": mean_or_zero(window["future_rootx_delta"] for window in windows),
            "mean_vx": mean_or_zero(window["mean_vx"] for window in windows),
            "mean_future_vx": mean_or_zero(window["future_mean_vx"] for window in windows),
            "mean_support_contact": mean_or_zero(window["support_contact_mean"] for window in windows),
            "mean_full_foot_support": mean_or_zero(window["full_foot_support_mean"] for window in windows),
            "mean_rotation_penalty_frac": mean_or_zero(window["rotation_penalty_frac"] for window in windows),
            "dominant_primary_actions": dict(primary.most_common()),
            "dominant_action_families": dict(family.most_common()),
            "phase_counts": dict(phase.most_common()),
        }
    return out


def candidate_action_contrasts(episode_analyses: list[dict[str, Any]], min_success_windows: int) -> list[dict[str, Any]]:
    by_event_role: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in episode_analyses:
        for window in ep["windows"]:
            by_event_role[(ep["threshold"], ep["anchor_cycle"], ep["role"])].append(window)

    candidates: list[dict[str, Any]] = []
    event_keys = {(threshold, cycle) for threshold, cycle, _ in by_event_role}
    for threshold, cycle in sorted(event_keys):
        success = by_event_role.get((threshold, cycle, "entry_success"), [])
        failure = by_event_role.get((threshold, cycle, "matched_failure"), [])
        if len(success) < min_success_windows:
            continue
        success_counts = Counter(window["dominant_action_family"] for window in success)
        failure_counts = Counter(window["dominant_action_family"] for window in failure)
        total_success = sum(success_counts.values())
        total_failure = sum(failure_counts.values())
        for family, count in success_counts.most_common(8):
            success_rate = count / total_success if total_success else 0.0
            failure_rate = failure_counts[family] / total_failure if total_failure else 0.0
            future_margin = mean_or_zero(
                window["future_rootx_delta"] for window in success if window["dominant_action_family"] == family
            ) - mean_or_zero(
                window["future_rootx_delta"] for window in failure if window["dominant_action_family"] == family
            )
            reward_margin = mean_or_zero(
                window["future_reward_sum"] for window in success if window["dominant_action_family"] == family
            ) - mean_or_zero(
                window["future_reward_sum"] for window in failure if window["dominant_action_family"] == family
            )
            candidates.append(
                {
                    "threshold": threshold,
                    "entry_cycle": cycle,
                    "action_family": family,
                    "success_count": count,
                    "failure_count": failure_counts[family],
                    "success_rate": success_rate,
                    "failure_rate": failure_rate,
                    "rate_margin": success_rate - failure_rate,
                    "future_rootx_margin": future_margin,
                    "future_reward_margin": reward_margin,
                    "status": "offline_ability_entry_candidate",
                    "evidence_kind": "success_entry_vs_matched_failure_window_contrast",
                }
            )
    candidates.sort(
        key=lambda item: (
            item["rate_margin"],
            item["future_reward_margin"],
            item["future_rootx_margin"],
            item["success_count"],
        ),
        reverse=True,
    )
    return candidates


def write_report_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Ability Entry Trace Report",
        "",
        "本报告只读 retrospective train traces，用于追踪高质量能力首次进入 replay 分布的过程；输出不接入在线训练。",
        "",
        "## First Crossings",
        "",
    ]
    for key, item in payload["first_crossings"].items():
        lines.append(
            f"- `{key}`: cycle=`{item['cycle']}`, episode=`{item['best_episode']}`, "
            f"best_reward=`{item['best_reward']:.3f}`, crossing_episodes=`{item['num_crossing_episodes']}`"
        )
    lines.extend(["", "## Cycle Summary", ""])
    for item in payload["cycle_summaries"]:
        lines.append(
            f"- seed `{item['seed']}` cycle `{item['cycle']:03d}`: "
            f"mean=`{item['mean_reward']:.3f}`, max=`{item['max_reward']:.3f}`, "
            f">1/10/50/100=`{item['count_gt_1']}/{item['count_gt_10']}/{item['count_gt_50']}/{item['count_gt_100']}`, "
            f"rule_gate=`{item.get('rule_gate_density', 0.0):.3f}`, "
            f"transition_gate=`{item.get('transition_rule_gate_density', 0.0):.3f}`"
        )
    lines.extend(["", "## Entry Window Contrasts", ""])
    for key, item in payload["window_group_summaries"].items():
        lines.append(
            f"- `{key}`: windows=`{item['num_windows']}`, "
            f"reward/future_reward=`{item['mean_reward_sum']:.3f}/{item['mean_future_reward_sum']:.3f}`, "
            f"rootx/future_rootx=`{item['mean_rootx_delta']:.3f}/{item['mean_future_rootx_delta']:.3f}`, "
            f"vx/future_vx=`{item['mean_vx']:.3f}/{item['mean_future_vx']:.3f}`, "
            f"dominant_actions=`{item['dominant_action_families']}`"
        )
    lines.extend(["", "## Candidate Action Contrasts", ""])
    for item in payload["candidate_action_contrasts"][:30]:
        lines.append(
            f"- R>`{item['threshold']:g}` cycle `{item['entry_cycle']:03d}` `{item['action_family']}`: "
            f"success `{item['success_count']}` ({item['success_rate']:.3f}) vs "
            f"failure `{item['failure_count']}` ({item['failure_rate']:.3f}), "
            f"rate_margin=`{item['rate_margin']:.3f}`, "
            f"future_reward_margin=`{item['future_reward_margin']:.3f}`"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "这些候选只表示入口窗口中的观察性差异。后续若转为规则，需要再经过 SD/TVA 反事实推演、跨周期后果验证和在线 shadow 审计。"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--max-cycle", type=int, default=25)
    parser.add_argument("--pre-cycles", type=int, default=1)
    parser.add_argument("--post-cycles", type=int, default=1)
    parser.add_argument("--top-k-episodes", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--future-horizon", type=int, default=30)
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--min-success-windows", type=int, default=3)
    parser.add_argument("--torso-z-min", type=float, default=0.75)
    parser.add_argument("--abs-pitch-max", type=float, default=1.10)
    parser.add_argument("--vx-min", type=float, default=0.50)
    parser.add_argument("--abs-omega-max", type=float, default=4.00)
    parser.add_argument("--dense-pose-min", type=float, default=0.65)
    parser.add_argument("--reward-min", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = HopperV8Adapter(
        torso_z_min=args.torso_z_min,
        abs_pitch_max=args.abs_pitch_max,
        vx_min=args.vx_min,
        abs_omega_max=args.abs_omega_max,
        dense_pose_min=args.dense_pose_min,
        reward_min=args.reward_min,
    )
    by_seed_cycle, cycle_stats = load_episode_tables(args.run_prefix, args.seeds, args.max_cycle)
    crossings = first_crossings(by_seed_cycle, args.seeds, args.thresholds)
    audit_stats = load_rule_audit_stats(args.results_root, args.max_cycle)
    cycle_summaries: list[dict[str, Any]] = []
    for key in sorted(cycle_stats):
        item = dict(cycle_stats[key])
        item.update(audit_stats.get(key, {}))
        cycle_summaries.append(item)

    events = build_entry_events(
        by_seed_cycle=by_seed_cycle,
        crossings=crossings,
        seeds=args.seeds,
        thresholds=args.thresholds,
        pre_cycles=args.pre_cycles,
        post_cycles=args.post_cycles,
        top_k=args.top_k_episodes,
    )
    episode_analyses: list[dict[str, Any]] = []
    for event in events:
        for record in event["success_records"]:
            analysis = analyze_episode(
                adapter,
                record,
                role="entry_success",
                threshold=event["threshold"],
                anchor_cycle=event["entry_cycle"],
                window_size=args.window_size,
                horizon=args.future_horizon,
                action_threshold=args.action_threshold,
            )
            if analysis is not None:
                episode_analyses.append(analysis)
        for record in event["matched_failure_records"]:
            analysis = analyze_episode(
                adapter,
                record,
                role="matched_failure",
                threshold=event["threshold"],
                anchor_cycle=event["entry_cycle"],
                window_size=args.window_size,
                horizon=args.future_horizon,
                action_threshold=args.action_threshold,
            )
            if analysis is not None:
                episode_analyses.append(analysis)

    window_groups = aggregate_windows(episode_analyses)
    candidates = candidate_action_contrasts(episode_analyses, args.min_success_windows)
    payload = {
        "analysis_family": "hopper_ability_entry_v1",
        "run_prefix": args.run_prefix,
        "results_root": str(args.results_root),
        "config": {
            "seeds": args.seeds,
            "thresholds": args.thresholds,
            "max_cycle": args.max_cycle,
            "pre_cycles": args.pre_cycles,
            "post_cycles": args.post_cycles,
            "top_k_episodes": args.top_k_episodes,
            "window_size": args.window_size,
            "future_horizon": args.future_horizon,
            "action_threshold": args.action_threshold,
            "min_success_windows": args.min_success_windows,
        },
        "first_crossings": crossings,
        "cycle_summaries": cycle_summaries,
        "entry_events": [
            {
                "threshold": event["threshold"],
                "entry_seed": event["entry_seed"],
                "entry_cycle": event["entry_cycle"],
                "num_success_records": len(event["success_records"]),
                "num_matched_failure_records": len(event["matched_failure_records"]),
                "crossing": event["crossing"],
            }
            for event in events
        ],
        "window_group_summaries": window_groups,
        "candidate_action_contrasts": candidates,
        "episode_window_records": episode_analyses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "entry_trace_report.json"
    write_json(report_json, payload)
    write_json(args.output_dir / "entry_candidate_segments.json", {"segments": episode_analyses})
    write_json(args.output_dir / "entry_candidate_actions.json", {"candidates": candidates})
    write_report_markdown(args.output_dir / "entry_trace_report.md", payload)
    print(f"wrote ability entry report: {report_json}")
    print(f"events: {len(events)} episode_records: {len(episode_analyses)} candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
