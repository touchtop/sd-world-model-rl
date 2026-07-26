#!/usr/bin/env python3
"""Mine Hopper success-continuation and transition-gap fragments for v8 rules."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from common import f, read_csv, resolve_workers, write_json


DEFAULT_RUN = "tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle040_target200039"
DEFAULT_TRACE_DIRS = [
    Path(f"logs/hopper-hop/{seed}/{DEFAULT_RUN}/official_eval/step_0200039/episode_csv")
    for seed in (1, 2, 3)
]
DEFAULT_RULE_LIBRARY = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v72_multiseed_shared_200k_seed123_resume100k_20260507_r2/"
    "cycle_040/rules/library_closed_loop_cycle_040.json"
)
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.json")

JOINT_TO_DIM = {"waist": 0, "hip": 1, "knee": 2, "ankle": 3}
ACTIVE_RULE_STATUSES = {"replay_validated", "rolling_validated"}
_MINE_ADAPTER: "HopperV8Adapter | None" = None
_MINE_ACTIVE_RULES: list[dict[str, Any]] = []
_MINE_MIN_LENGTH = 1
_MINE_LOOKAHEAD = 25


@dataclass(frozen=True)
class Fragment:
    trace_csv: str
    seed: int | None
    episode: int | None
    start: int
    end: int
    length: int
    role: str
    quality_bin: str
    key: str
    score: float
    metrics: dict[str, float]
    degradation: dict[str, Any]
    existing_rule_matches: list[str]


class HopperV8Adapter:
    """Hopper-specific interpretation layer for the task-agnostic v8 pipeline."""

    def __init__(
        self,
        torso_z_min: float,
        abs_pitch_max: float,
        vx_min: float,
        abs_omega_max: float,
        dense_pose_min: float,
        reward_min: float,
    ) -> None:
        self.torso_z_min = torso_z_min
        self.abs_pitch_max = abs_pitch_max
        self.vx_min = vx_min
        self.abs_omega_max = abs_omega_max
        self.dense_pose_min = dense_pose_min
        self.reward_min = reward_min

    def omega_bin(self, row: dict[str, str]) -> str:
        omega = f(row, "pitch_rate")
        if omega > 1.0:
            return "omega_pos"
        if omega < -1.0:
            return "omega_neg"
        return "omega_mid"

    def vx_bin(self, row: dict[str, str]) -> str:
        vx = f(row, "vx")
        if vx >= 0.8:
            return "vx_high"
        if vx < 0.2:
            return "vx_low"
        return "vx_mid"

    def support_bin(self, row: dict[str, str]) -> str:
        support = f(row, "full_foot_support")
        if support < 0.1:
            return "support_none"
        if support < 0.35:
            return "support_low"
        return "support_some"

    def phase(self, row: dict[str, str]) -> str:
        falling = f(row, "falling_air") >= 0.5 or f(row, "falling_air_gt_05") >= 0.5
        rear = f(row, "grounded_rear") >= 0.5 or f(row, "grounded_rear_gt_05") >= 0.5
        prone = f(row, "grounded_prone") >= 0.5 or f(row, "grounded_prone_gt_05") >= 0.5
        hop = f(row, "grounded_hop") >= 0.5 or f(row, "grounded_hop_gt_05") >= 0.5
        upright = (
            f(row, "grounded_upright") >= 0.5
            or f(row, "grounded_upright_gt_05") >= 0.5
            or f(row, "grounded_prepared") >= 0.5
            or f(row, "grounded_prepared_gt_05") >= 0.5
        )
        rising = f(row, "rising_air") >= 0.5 or f(row, "rising_air_gt_05") >= 0.5
        if falling:
            return "falling"
        if rear:
            return "rear"
        if hop:
            return "hop"
        if prone:
            return "prone"
        if upright:
            return "upright"
        if rising:
            return "rising"
        return "other"

    def risk_bin(self, row: dict[str, str]) -> str:
        if f(row, "body_collision") >= 0.5:
            return "collision_risk"
        if f(row, "rotation_penalty") >= 0.5 or abs(f(row, "pitch_rate")) >= 3.5:
            return "rotation_risk"
        if self.phase(row) in {"falling", "rising"} and f(row, "full_foot_support") < 0.1:
            return "support_risk"
        return "stable"

    def structural_success(self, row: dict[str, str]) -> bool:
        return (
            f(row, "torso_z") >= self.torso_z_min
            and abs(f(row, "pitch")) <= self.abs_pitch_max
            and f(row, "vx") >= self.vx_min
            and abs(f(row, "pitch_rate")) <= self.abs_omega_max
            and f(row, "body_collision") < 0.5
        )

    def success_reason(self, row: dict[str, str]) -> list[str]:
        reasons: list[str] = []
        if self.structural_success(row):
            reasons.append("structural_success")
        if f(row, "dense_pose_potential") >= self.dense_pose_min:
            reasons.append("dense_pose_high")
        if f(row, "reward") > self.reward_min:
            reasons.append("positive_reward")
        return reasons

    def is_success_candidate(self, row: dict[str, str]) -> bool:
        return bool(self.success_reason(row))

    def continuation_score(self, row: dict[str, str]) -> float:
        uprightness = max(0.0, 1.0 - abs(f(row, "pitch")) / max(self.abs_pitch_max, 1e-6))
        height_quality = min(max((f(row, "torso_z") - 0.45) / 0.45, 0.0), 1.0)
        forward_speed = min(max(f(row, "vx") / 2.0, 0.0), 1.0)
        spin_control = max(0.0, 1.0 - abs(f(row, "pitch_rate")) / max(self.abs_omega_max, 1e-6))
        support_readiness = max(f(row, "support_contact"), f(row, "full_foot_support"))
        non_collision = 1.0 - min(max(f(row, "body_collision"), 0.0), 1.0)
        return (
            0.25 * uprightness
            + 0.20 * height_quality
            + 0.20 * forward_speed
            + 0.15 * spin_control
            + 0.10 * support_readiness
            + 0.10 * non_collision
        )

    def transition_score(self, row: dict[str, str]) -> float:
        positive_vx = min(max(f(row, "vx") / 2.0, 0.0), 1.0)
        low_rotation = 1.0 - min(max(f(row, "rotation_penalty"), 0.0), 1.0)
        return (
            0.25 * f(row, "landing_pose_score")
            + 0.20 * f(row, "ground_pose_score")
            + 0.20 * f(row, "rise_pose_score")
            + 0.15 * f(row, "support_contact")
            + 0.10 * positive_vx
            + 0.10 * low_rotation
        )

    def key(self, role: str, quality_bin: str, row: dict[str, str]) -> str:
        return "|".join(
            [
                role,
                quality_bin,
                self.omega_bin(row),
                self.vx_bin(row),
                self.support_bin(row),
                self.phase(row),
                self.risk_bin(row),
            ]
        )

    def legacy_substate_key(self, event: str, row: dict[str, str]) -> str:
        return "|".join([event, self.omega_bin(row), self.vx_bin(row), self.support_bin(row), self.phase(row)])


def trace_paths(trace_dirs: list[Path], trace_csvs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for d in trace_dirs:
        if d.exists():
            paths.extend(sorted(d.glob("episode_*.csv")))
    paths.extend(trace_csvs)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return out


def infer_seed_episode(path: Path) -> tuple[int | None, int | None]:
    seed = None
    episode = None
    parts = path.parts
    if "hopper-hop" in parts:
        idx = parts.index("hopper-hop")
        if idx + 1 < len(parts):
            try:
                seed = int(parts[idx + 1])
            except ValueError:
                seed = None
    stem = path.stem
    if stem.startswith("episode_"):
        try:
            episode = int(stem.split("_", 1)[1])
        except ValueError:
            episode = None
    return seed, episode


def contiguous_segments(rows: list[dict[str, str]], flags: list[bool], min_length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        if (not flag or idx == len(flags) - 1) and start is not None:
            end = idx if flag and idx == len(flags) - 1 else idx - 1
            if end - start + 1 >= min_length:
                segments.append((start, end))
            start = None
    return segments


def mean_metric(rows: list[dict[str, str]], key: str) -> float:
    return mean([f(row, key) for row in rows]) if rows else 0.0


def max_metric(rows: list[dict[str, str]], key: str) -> float:
    return max((f(row, key) for row in rows), default=0.0)


def summarize_window(adapter: HopperV8Adapter, rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "structural_success_frac": mean(1.0 if adapter.structural_success(row) else 0.0 for row in rows) if rows else 0.0,
        "dense_pose_high_frac": mean(1.0 if f(row, "dense_pose_potential") >= adapter.dense_pose_min else 0.0 for row in rows)
        if rows
        else 0.0,
        "positive_reward_frac": mean(1.0 if f(row, "reward") > adapter.reward_min else 0.0 for row in rows) if rows else 0.0,
        "mean_reward": mean_metric(rows, "reward"),
        "sum_reward": sum(f(row, "reward") for row in rows),
        "mean_dense_pose": mean_metric(rows, "dense_pose_potential"),
        "max_dense_pose": max_metric(rows, "dense_pose_potential"),
        "mean_continue_score": mean([adapter.continuation_score(row) for row in rows]) if rows else 0.0,
        "mean_transition_score": mean([adapter.transition_score(row) for row in rows]) if rows else 0.0,
        "mean_pitch": mean_metric(rows, "pitch"),
        "mean_abs_pitch": mean([abs(f(row, "pitch")) for row in rows]) if rows else 0.0,
        "mean_torso_z": mean_metric(rows, "torso_z"),
        "mean_vx": mean_metric(rows, "vx"),
        "mean_vz": mean_metric(rows, "vz"),
        "mean_abs_omega": mean([abs(f(row, "pitch_rate")) for row in rows]) if rows else 0.0,
        "mean_support_contact": mean_metric(rows, "support_contact"),
        "mean_full_foot_support": mean_metric(rows, "full_foot_support"),
        "body_collision_frac": mean_metric(rows, "body_collision"),
        "rotation_penalty_frac": mean(1.0 if f(row, "rotation_penalty") >= 0.5 else 0.0 for row in rows)
        if rows
        else 0.0,
    }


def degradation_after(
    adapter: HopperV8Adapter,
    rows: list[dict[str, str]],
    start: int,
    end: int,
    lookahead: int,
) -> dict[str, Any]:
    segment_rows = rows[start : end + 1]
    future_start = min(end + 1, len(rows))
    future_end = min(len(rows), end + 1 + lookahead)
    future = rows[future_start:future_end]
    if not future:
        return {"has_degradation": False, "types": [], "first_step": None, "window": []}

    base_abs_pitch = mean([abs(f(row, "pitch")) for row in segment_rows]) if segment_rows else 0.0
    base_torso_z = mean_metric(segment_rows, "torso_z")
    base_vx = mean_metric(segment_rows, "vx")
    types_by_step: list[dict[str, Any]] = []
    for offset, row in enumerate(future, start=future_start):
        types: list[str] = []
        if abs(f(row, "pitch")) >= max(0.80, base_abs_pitch + 0.35):
            types.append("pitch_deterioration")
        if abs(f(row, "pitch_rate")) >= 3.5 or f(row, "rotation_penalty") >= 0.5:
            types.append("rotation_risk")
        if f(row, "torso_z") <= min(0.55, base_torso_z - 0.20):
            types.append("height_drop")
        if f(row, "vx") <= min(0.20, base_vx - 0.80):
            types.append("forward_speed_loss")
        if adapter.phase(row) in {"rear", "prone"}:
            types.append(f"entered_{adapter.phase(row)}")
        if f(row, "body_collision") >= 0.5:
            types.append("body_collision")
        if adapter.phase(row) in {"rear", "prone", "upright"} and f(row, "support_contact") < 0.1:
            types.append("support_failure")
        if types:
            types_by_step.append({"step": offset, "types": types})
            break

    all_types = types_by_step[0]["types"] if types_by_step else []
    return {
        "has_degradation": bool(types_by_step),
        "types": all_types,
        "first_step": types_by_step[0]["step"] if types_by_step else None,
        "lookahead_start": future_start,
        "lookahead_end": future_end - 1,
    }


def load_active_rules(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    import json

    payload = json.loads(path.read_text())
    return [
        rule
        for rule in payload.get("rules", [])
        if rule.get("status") in ACTIVE_RULE_STATUSES
        and rule.get("source") == "v4_substate"
        and rule.get("intervention_type") == "soft_prefer_action_family"
    ]


def matching_legacy_rules(adapter: HopperV8Adapter, row: dict[str, str], active_rules: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for rule in active_rules:
        event = str(rule.get("state_event") or str(rule.get("substate_key", "")).split("|", 1)[0])
        if event == "rotation_risk" and adapter.risk_bin(row) != "rotation_risk":
            continue
        key = adapter.legacy_substate_key(event, row)
        if key == rule.get("substate_key"):
            out.append(str(rule.get("id")))
    return out


def init_mine_worker(
    adapter: HopperV8Adapter,
    active_rules: list[dict[str, Any]],
    min_length: int,
    lookahead: int,
) -> None:
    global _MINE_ADAPTER, _MINE_ACTIVE_RULES, _MINE_MIN_LENGTH, _MINE_LOOKAHEAD
    _MINE_ADAPTER = adapter
    _MINE_ACTIVE_RULES = active_rules
    _MINE_MIN_LENGTH = min_length
    _MINE_LOOKAHEAD = lookahead


def mine_trace(path: Path) -> tuple[list[Fragment], dict[str, Any] | None]:
    if _MINE_ADAPTER is None:
        raise RuntimeError("mine worker adapter was not initialized")
    adapter = _MINE_ADAPTER
    rows = read_csv(path)
    seed, episode = infer_seed_episode(path)
    if not rows:
        return [], None
    flags = [adapter.is_success_candidate(row) for row in rows]
    segments = contiguous_segments(rows, flags, _MINE_MIN_LENGTH)
    trace_summary = {
        "trace_csv": str(path),
        "seed": seed,
        "episode": episode,
        "length": len(rows),
        "episode_reward": sum(f(row, "reward") for row in rows),
        "num_success_segments": len(segments),
        "success_frame_frac": sum(1 for flag in flags if flag) / len(flags),
    }
    fragments: list[Fragment] = []
    for start, end in segments:
        window = rows[start : end + 1]
        best_local = max(range(start, end + 1), key=lambda idx: adapter.continuation_score(rows[idx]))
        best_row = rows[best_local]
        metrics = summarize_window(adapter, window)
        degradation = degradation_after(adapter, rows, start, end, _MINE_LOOKAHEAD)
        role = "transition" if degradation["has_degradation"] else "continue"
        quality_bin = "transition_gap" if degradation["has_degradation"] else "good_fragment"
        rule_key = adapter.key(role, quality_bin, best_row)
        existing = matching_legacy_rules(adapter, best_row, _MINE_ACTIVE_RULES)
        score = (
            metrics["mean_continue_score"]
            + 0.35 * metrics["mean_transition_score"]
            + 0.05 * min(metrics["sum_reward"], 10.0)
            + 0.03 * len(window)
        )
        fragments.append(
            Fragment(
                trace_csv=str(path),
                seed=seed,
                episode=episode,
                start=start,
                end=end,
                length=end - start + 1,
                role=role,
                quality_bin=quality_bin,
                key=rule_key,
                score=score,
                metrics=metrics,
                degradation=degradation,
                existing_rule_matches=existing,
            )
        )
    return fragments, trace_summary


def mine_fragments(
    paths: list[Path],
    adapter: HopperV8Adapter,
    active_rules: list[dict[str, Any]],
    min_length: int,
    lookahead: int,
    workers: int = 0,
) -> dict[str, Any]:
    fragments: list[Fragment] = []
    trace_summaries: list[dict[str, Any]] = []
    worker_count = resolve_workers(workers)
    if worker_count <= 1 or len(paths) <= 1:
        init_mine_worker(adapter, active_rules, min_length, lookahead)
        trace_results = [mine_trace(path) for path in paths]
    else:
        chunksize = max(1, len(paths) // (worker_count * 4))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_mine_worker,
            initargs=(adapter, active_rules, min_length, lookahead),
        ) as pool:
            trace_results = list(pool.map(mine_trace, paths, chunksize=chunksize))

    for trace_fragments, trace_summary in trace_results:
        fragments.extend(trace_fragments)
        if trace_summary is not None:
            trace_summaries.append(trace_summary)

    fragment_rows = [fragment_to_dict(item) for item in sorted(fragments, key=lambda x: x.score, reverse=True)]
    key_counts = Counter(item["key"] for item in fragment_rows)
    reason_counts = Counter(
        reason
        for item in fragment_rows
        for reason, metric_key in (
            ("structural_success", "structural_success_frac"),
            ("dense_pose_high", "dense_pose_high_frac"),
            ("positive_reward", "positive_reward_frac"),
        )
        if item["metrics"][metric_key] > 0.0
    )
    degradation_counts = Counter(
        dtype
        for item in fragment_rows
        for dtype in item.get("degradation", {}).get("types", [])
    )
    return {
        "fragment_family": "hopper-v8-success-continuation-and-transition-gaps",
        "num_traces": len(paths),
        "workers": worker_count,
        "num_fragments": len(fragment_rows),
        "num_continue_fragments": sum(1 for item in fragment_rows if item["role"] == "continue"),
        "num_transition_gap_fragments": sum(1 for item in fragment_rows if item["role"] == "transition"),
        "num_fragments_with_existing_rule_match": sum(1 for item in fragment_rows if item["existing_rule_matches"]),
        "top_keys": dict(key_counts.most_common(20)),
        "success_reason_counts": dict(reason_counts.most_common()),
        "degradation_counts": dict(degradation_counts.most_common()),
        "trace_summaries": trace_summaries,
        "fragments": fragment_rows,
    }


def fragment_to_dict(fragment: Fragment) -> dict[str, Any]:
    return {
        "trace_csv": fragment.trace_csv,
        "seed": fragment.seed,
        "episode": fragment.episode,
        "start": fragment.start,
        "end": fragment.end,
        "length": fragment.length,
        "role": fragment.role,
        "quality_bin": fragment.quality_bin,
        "key": fragment.key,
        "score": fragment.score,
        "metrics": fragment.metrics,
        "degradation": fragment.degradation,
        "existing_rule_matches": fragment.existing_rule_matches,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_rows: int) -> None:
    lines = [
        "# Hopper v8 Success Fragment Mining",
        "",
        f"Traces: `{payload['num_traces']}`",
        f"Fragments: `{payload['num_fragments']}`",
        f"Continue fragments: `{payload['num_continue_fragments']}`",
        f"Transition-gap fragments: `{payload['num_transition_gap_fragments']}`",
        f"Fragments with existing v7 rule match: `{payload['num_fragments_with_existing_rule_match']}`",
        "",
        "## Top Rule Keys",
        "",
    ]
    for key, count in payload["top_keys"].items():
        lines.append(f"- `{key}`: `{count}`")
    lines += ["", "## Success Candidate Reasons", ""]
    for key, count in payload["success_reason_counts"].items():
        lines.append(f"- `{key}`: `{count}`")
    lines += ["", "## Degradation Types", ""]
    for key, count in payload["degradation_counts"].items():
        lines.append(f"- `{key}`: `{count}`")
    lines += ["", "## Top Fragments", ""]
    for idx, item in enumerate(payload["fragments"][:max_rows], start=1):
        metrics = item["metrics"]
        degradation = item["degradation"]
        lines += [
            f"### {idx}. seed={item['seed']} episode={item['episode']} frames={item['start']}-{item['end']}",
            "",
            f"- Role: `{item['role']}`",
            f"- Key: `{item['key']}`",
            f"- Length: `{item['length']}`",
            f"- Score: `{item['score']:.4f}`",
            f"- Reward sum: `{metrics['sum_reward']:.4f}`",
            f"- Candidate fractions structural / dense / reward: `{metrics['structural_success_frac']:.3f}` / `{metrics['dense_pose_high_frac']:.3f}` / `{metrics['positive_reward_frac']:.3f}`",
            f"- Mean continue / transition score: `{metrics['mean_continue_score']:.4f}` / `{metrics['mean_transition_score']:.4f}`",
            f"- Mean pitch / torso_z / vx / |omega|: `{metrics['mean_pitch']:.4f}` / `{metrics['mean_torso_z']:.4f}` / `{metrics['mean_vx']:.4f}` / `{metrics['mean_abs_omega']:.4f}`",
            f"- Support contact / full foot: `{metrics['mean_support_contact']:.4f}` / `{metrics['mean_full_foot_support']:.4f}`",
            f"- Degradation: `{degradation}`",
            f"- Existing rule matches: `{item['existing_rule_matches']}`",
            f"- Trace: `{item['trace_csv']}`",
            "",
        ]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", type=Path, default=[])
    parser.add_argument("--trace-csv", action="append", type=Path, default=[])
    parser.add_argument("--use-default-v72-200k", action="store_true", default=False)
    parser.add_argument("--rule-library", type=Path, default=DEFAULT_RULE_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=25)
    parser.add_argument("--torso-z-min", type=float, default=0.75)
    parser.add_argument("--abs-pitch-max", type=float, default=0.45)
    parser.add_argument("--vx-min", type=float, default=0.80)
    parser.add_argument("--abs-omega-max", type=float, default=3.50)
    parser.add_argument("--dense-pose-min", type=float, default=0.65)
    parser.add_argument("--reward-min", type=float, default=0.0)
    parser.add_argument("--max-markdown-fragments", type=int, default=40)
    parser.add_argument("--workers", type=int, default=0, help="CPU worker processes; 0 uses SD_RETRO_WORKERS or a bounded auto default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dirs = list(args.trace_dir)
    if args.use_default_v72_200k or (not trace_dirs and not args.trace_csv):
        trace_dirs.extend(DEFAULT_TRACE_DIRS)
    paths = trace_paths(trace_dirs, args.trace_csv)
    adapter = HopperV8Adapter(
        torso_z_min=args.torso_z_min,
        abs_pitch_max=args.abs_pitch_max,
        vx_min=args.vx_min,
        abs_omega_max=args.abs_omega_max,
        dense_pose_min=args.dense_pose_min,
        reward_min=args.reward_min,
    )
    active_rules = load_active_rules(args.rule_library)
    payload = mine_fragments(
        paths=paths,
        adapter=adapter,
        active_rules=active_rules,
        min_length=args.min_length,
        lookahead=args.lookahead,
        workers=args.workers,
    )
    payload["config"] = {
        "trace_dirs": [str(p) for p in trace_dirs],
        "trace_csvs": [str(p) for p in args.trace_csv],
        "rule_library": str(args.rule_library) if args.rule_library else None,
        "active_rules": len(active_rules),
        "min_length": args.min_length,
        "lookahead": args.lookahead,
        "torso_z_min": args.torso_z_min,
        "abs_pitch_max": args.abs_pitch_max,
        "vx_min": args.vx_min,
        "abs_omega_max": args.abs_omega_max,
        "dense_pose_min": args.dense_pose_min,
        "reward_min": args.reward_min,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_fragments)
    print(f"wrote fragments: {args.output}")
    print(f"fragments: {payload['num_fragments']} traces: {payload['num_traces']}")
    print(f"transition gaps: {payload['num_transition_gap_fragments']}")
    print(f"existing-rule-covered fragments: {payload['num_fragments_with_existing_rule_match']}")


if __name__ == "__main__":
    main()
