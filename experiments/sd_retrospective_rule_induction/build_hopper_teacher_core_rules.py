#!/usr/bin/env python3
"""Build a small teacher-core Hopper rule library from high-quality traces.

The script mines conservative action-family preferences from completed
high-return Hopper trajectories.  It is intentionally separate from the
closed-loop self-induction pipeline: the output is an initialization library
for a teacher-core experiment, not evidence that the same rules were
autonomously discovered from failed replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from audit_hopper_v8_widefunnel_rule_effects import (
    as_float,
    continuation_score,
    phase,
    row_parts,
    selected_action_relation,
    structural_success,
    support_bin,
    vx_bin,
    omega_bin,
)


JOINT_TO_ACTION = {"waist": "action_0", "hip": "action_1", "knee": "action_2", "ankle": "action_3"}
ACTION_TO_JOINT = {v: k for k, v in JOINT_TO_ACTION.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def episode_reward(rows: list[dict[str, str]]) -> float:
    return sum(as_float(row, "reward") for row in rows)


def infer_seed(path: Path, fallback: int) -> int:
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part == "hopper-hop" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return fallback
    return fallback


def episode_idx(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def collect_trace_paths(trace_dirs: list[Path], trace_csvs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for trace_dir in trace_dirs:
        out.extend(sorted(trace_dir.glob("episode_*.csv")))
    out.extend(trace_csvs)
    return sorted(dict.fromkeys(path.resolve() for path in out))


def action_family(row: dict[str, str], action_col: str, threshold: float) -> str | None:
    value = as_float(row, action_col)
    if abs(value) < threshold:
        return None
    joint = ACTION_TO_JOINT[action_col]
    sign = "pos" if value > 0 else "neg"
    return f"{joint}_{sign}"


def opposite_action_family(family: str) -> str | None:
    if family.endswith("_pos"):
        return family[:-4] + "_neg"
    if family.endswith("_neg"):
        return family[:-4] + "_pos"
    return None


def state_key(row: dict[str, str], event: str, continue_score_min: float) -> str:
    parts = row_parts(row, continue_score_min)
    return "|".join([event, parts["omega_bin"], parts["vx_bin"], parts["support_bin"], parts["phase"]])


def row_passes_quality_filter(
    row: dict[str, str],
    min_row_reward: float,
    min_vx: float,
    max_body_collision: float,
    max_abs_pitch: float,
    continue_score_min: float,
    require_near_good: bool,
    allowed_phases: set[str],
) -> bool:
    if as_float(row, "reward") < min_row_reward:
        return False
    if as_float(row, "vx") < min_vx:
        return False
    if as_float(row, "body_collision") > max_body_collision:
        return False
    if abs(as_float(row, "pitch")) > max_abs_pitch:
        return False
    if require_near_good and not (structural_success(row) or continuation_score(row) >= continue_score_min):
        return False
    if allowed_phases and phase(row) not in allowed_phases:
        return False
    return True


def mine_teacher_rules(
    traces: list[Path],
    min_episode_reward: float,
    min_row_reward: float,
    min_vx: float,
    max_body_collision: float,
    max_abs_pitch: float,
    continue_score_min: float,
    require_near_good: bool,
    allowed_phases: set[str],
    action_threshold: float,
    min_support: int,
    min_episodes: int,
    min_directional_share: float,
    max_opposite_share: float,
    max_rules: int,
    max_rules_per_substate: int,
    include_rotation_risk_event: bool,
) -> dict[str, Any]:
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    trace_summaries: list[dict[str, Any]] = []

    for trace_idx, path in enumerate(traces):
        rows = read_csv(path)
        if not rows:
            continue
        total_reward = episode_reward(rows)
        seed = infer_seed(path, trace_idx)
        ep_idx = episode_idx(path)
        selected = total_reward >= min_episode_reward
        kept_rows = 0
        if selected:
            for step_idx, row in enumerate(rows):
                if not row_passes_quality_filter(
                    row,
                    min_row_reward=min_row_reward,
                    min_vx=min_vx,
                    max_body_collision=max_body_collision,
                    max_abs_pitch=max_abs_pitch,
                    continue_score_min=continue_score_min,
                    require_near_good=require_near_good,
                    allowed_phases=allowed_phases,
                ):
                    continue
                kept_rows += 1
                events = ["unsupported_high_torque"]
                if include_rotation_risk_event and (
                    as_float(row, "rotation_penalty") >= 0.5 or abs(as_float(row, "pitch_rate")) >= 3.0
                ):
                    events.append("rotation_risk")
                for event in events:
                    key = state_key(row, event, continue_score_min)
                    row_rec = {
                        "trace_csv": str(path),
                        "seed": seed,
                        "episode": ep_idx,
                        "episode_reward": total_reward,
                        "step": int(as_float(row, "step", step_idx)),
                        "row": row,
                    }
                    group_rows[key].append(row_rec)
                    for action_col in JOINT_TO_ACTION.values():
                        family = action_family(row, action_col, action_threshold)
                        if family is not None:
                            candidate_rows[(key, family)].append(row_rec)
        trace_summaries.append(
            {
                "trace_csv": str(path),
                "seed": seed,
                "episode": ep_idx,
                "episode_reward": total_reward,
                "selected": selected,
                "rows": len(rows),
                "quality_rows": kept_rows,
            }
        )

    rules: list[dict[str, Any]] = []
    competition_diagnostics: list[dict[str, Any]] = []
    for key, rows in sorted(group_rows.items()):
        substate_total = len(rows)
        by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (candidate_key, family), hits in candidate_rows.items():
            if candidate_key != key:
                continue
            joint = family.rsplit("_", 1)[0]
            support = len(hits)
            episodes = {(hit["seed"], hit["episode"]) for hit in hits}
            relation_counts = Counter(selected_action_relation(hit["row"], family, action_threshold) for hit in rows)
            same = relation_counts["same"]
            opposite = relation_counts["opposite"]
            neutral = relation_counts["neutral"]
            directional_total = same + opposite
            directional_share = same / directional_total if directional_total else 0.0
            opposite_share = opposite / directional_total if directional_total else 0.0
            support_share = support / substate_total if substate_total else 0.0
            if support < min_support:
                continue
            if len(episodes) < min_episodes:
                continue
            if directional_share < min_directional_share:
                continue
            if opposite_share > max_opposite_share:
                continue
            confidence = min(
                1.0,
                0.70
                + 0.20 * directional_share
                + 0.05 * min(1.0, len(episodes) / max(1, min_episodes + 2))
                + 0.05 * min(1.0, math.log1p(support) / math.log1p(max(min_support, support))),
            )
            event = key.split("|", 1)[0]
            family_rows = [
                {
                    "seed": hit["seed"],
                    "episode": hit["episode"],
                    "step": hit["step"],
                    "episode_reward": hit["episode_reward"],
                    "trace_csv": hit["trace_csv"],
                }
                for hit in hits[:5]
            ]
            by_joint[joint].append(
                {
                    "id": f"{key}->prefer:{family}",
                    "substate_key": key,
                    "rule_key": key,
                    "state_event": event,
                    "substate_predicate": key,
                    "action_family": family,
                    "opposite_action_family": opposite_action_family(family),
                    "intervention_type": "soft_prefer_action_family",
                    "source": "v4_substate",
                    "rule_origin": "teacher_core_standard_hop",
                    "teacher_source": "hopper_high_quality_official_eval",
                    "status": "confidence_active",
                    "confidence_status": "confidence_active",
                    "confidence": round(confidence, 6),
                    "effective_prior_scale": round(confidence, 6),
                    "support": support,
                    "teacher_episode_support": len(episodes),
                    "substate_support": substate_total,
                    "substate_coverage": support_share,
                    "directional_share": directional_share,
                    "opposite_share": opposite_share,
                    "same": same,
                    "opposite": opposite,
                    "neutral": neutral,
                    "mean_teacher_episode_reward": mean(hit["episode_reward"] for hit in hits),
                    "online_thresholds": {
                        "min_support": min_support,
                        "min_episodes": min_episodes,
                        "min_directional_share": min_directional_share,
                        "max_opposite_share": max_opposite_share,
                    },
                    "failed_online_checks": [],
                    "teacher_examples": family_rows,
                    "audit_notes": [
                        "由高质量教师轨迹中的动作方向稳定性生成；作为教师核心规则初始化，不等同于纯自学习发现。",
                        "source 字段保持 v4_substate 以兼容现有 online_rule_prior_v6。rule_origin 标记真实来源。",
                    ],
                }
            )
        for joint, joint_rules in by_joint.items():
            joint_rules.sort(key=lambda r: (r["directional_share"], r["support"], r["teacher_episode_support"]), reverse=True)
            competition_diagnostics.append(
                {
                    "substate_key": key,
                    "joint": joint,
                    "num_candidates": len(joint_rules),
                    "kept": joint_rules[0]["id"] if joint_rules else None,
                    "candidate_ids": [rule["id"] for rule in joint_rules],
                }
            )
            if joint_rules:
                rules.append(joint_rules[0])

    rules.sort(key=lambda r: (r["directional_share"], r["teacher_episode_support"], r["support"]), reverse=True)
    pre_limit = len(rules)
    if max_rules_per_substate > 0:
        kept: list[dict[str, Any]] = []
        per_substate: Counter[str] = Counter()
        for rule in rules:
            key = str(rule["substate_key"])
            if per_substate[key] >= max_rules_per_substate:
                continue
            kept.append(rule)
            per_substate[key] += 1
        rules = kept
    if max_rules > 0:
        rules = rules[:max_rules]

    return {
        "rule_family": "hopper-teacher-core-rules-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "online_enabled": bool(rules),
        "compatibility": {
            "online_prior": "online_rule_prior_v6",
            "required_allowed_status": "confidence_active",
            "source_compatibility": "source is set to v4_substate; rule_origin records teacher provenance",
        },
        "selection": {
            "min_episode_reward": min_episode_reward,
            "min_row_reward": min_row_reward,
            "min_vx": min_vx,
            "max_body_collision": max_body_collision,
            "max_abs_pitch": max_abs_pitch,
            "continue_score_min": continue_score_min,
            "require_near_good": require_near_good,
            "allowed_phases": sorted(allowed_phases),
            "action_threshold": action_threshold,
            "min_support": min_support,
            "min_episodes": min_episodes,
            "min_directional_share": min_directional_share,
            "max_opposite_share": max_opposite_share,
            "max_rules": max_rules,
            "max_rules_per_substate": max_rules_per_substate,
            "include_rotation_risk_event": include_rotation_risk_event,
        },
        "num_traces": len(trace_summaries),
        "num_selected_traces": sum(1 for item in trace_summaries if item["selected"]),
        "num_rules": len(rules),
        "num_rules_pre_limit": pre_limit,
        "trace_summaries": trace_summaries,
        "competition_diagnostics": competition_diagnostics,
        "rules": rules,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Hopper Teacher Core Rule Library",
        "",
        "该报告记录从高质量 Hopper official eval 轨迹中提取的教师核心规则。规则用于初始化独立实验线，不作为纯在线自学习证据。",
        "",
        "## Summary",
        "",
        f"- Traces: `{payload['num_traces']}`",
        f"- Selected traces: `{payload['num_selected_traces']}`",
        f"- Rules: `{payload['num_rules']}`",
        f"- Rules before final limit: `{payload['num_rules_pre_limit']}`",
        "",
        "## Selection",
        "",
    ]
    for key, value in payload["selection"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Selected Traces", "", "| seed | episode | reward | quality_rows | trace |", "|---:|---:|---:|---:|---|"])
    for item in payload["trace_summaries"]:
        if not item["selected"]:
            continue
        lines.append(
            f"| {item['seed']} | {item['episode']} | {item['episode_reward']:.3f} | "
            f"{item['quality_rows']} | `{item['trace_csv']}` |"
        )
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "| rule | action | support | episodes | dir% | opposite% | confidence |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rule in payload["rules"]:
        lines.append(
            f"| `{rule['substate_key']}` | `{rule['action_family']}` | {rule['support']} | "
            f"{rule['teacher_episode_support']} | {100.0 * rule['directional_share']:.1f} | "
            f"{100.0 * rule['opposite_share']:.1f} | {rule['confidence']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", action="append", type=Path, default=[])
    parser.add_argument("--trace-csv", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-episode-reward", type=float, default=450.0)
    parser.add_argument("--min-row-reward", type=float, default=0.0)
    parser.add_argument("--min-vx", type=float, default=0.50)
    parser.add_argument("--max-body-collision", type=float, default=0.20)
    parser.add_argument("--max-abs-pitch", type=float, default=1.10)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--require-near-good", action="store_true")
    parser.add_argument("--allowed-phase", action="append", default=[])
    parser.add_argument("--action-threshold", type=float, default=0.25)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--min-episodes", type=int, default=6)
    parser.add_argument("--min-directional-share", type=float, default=0.72)
    parser.add_argument("--max-opposite-share", type=float, default=0.28)
    parser.add_argument("--max-rules", type=int, default=24)
    parser.add_argument("--max-rules-per-substate", type=int, default=3)
    parser.add_argument("--include-rotation-risk-event", action="store_true")
    args = parser.parse_args()

    traces = collect_trace_paths(args.trace_dir, args.trace_csv)
    payload = mine_teacher_rules(
        traces=traces,
        min_episode_reward=args.min_episode_reward,
        min_row_reward=args.min_row_reward,
        min_vx=args.min_vx,
        max_body_collision=args.max_body_collision,
        max_abs_pitch=args.max_abs_pitch,
        continue_score_min=args.continue_score_min,
        require_near_good=args.require_near_good,
        allowed_phases=set(args.allowed_phase),
        action_threshold=args.action_threshold,
        min_support=args.min_support,
        min_episodes=args.min_episodes,
        min_directional_share=args.min_directional_share,
        max_opposite_share=args.max_opposite_share,
        max_rules=args.max_rules,
        max_rules_per_substate=args.max_rules_per_substate,
        include_rotation_risk_event=args.include_rotation_risk_event,
    )
    write_json(args.output, payload)
    report = args.report or args.output.with_suffix(".md")
    write_markdown(report, payload)
    print(f"wrote teacher core rules: {args.output}")
    print(f"markdown: {report}")
    print(f"selected_traces={payload['num_selected_traces']} rules={payload['num_rules']}")


if __name__ == "__main__":
    main()
