#!/usr/bin/env python3
"""Extract Hopper transition-context action-rule candidates from replay traces.

The extractor is deliberately offline-only.  It uses SD state features to
construct short transition contexts:

    past dominant phase -> current candidate phase -> future outcome

and then summarizes which action families are associated with better future
transition scores inside the same observable context.  The resulting rules are
marked as observational candidates; they are not replay-safe or online-ready
until a later SD/TVA counterfactual audit validates the action effect.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import write_json  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


OBS_COLS = [
    "qpos_rootz",
    "qpos_rooty",
    "qpos_waist",
    "qpos_hip",
    "qpos_knee",
    "qpos_ankle",
    "qvel_rootx",
    "qvel_rootz",
    "qvel_rooty",
    "qvel_waist",
    "qvel_hip",
    "qvel_knee",
    "qvel_ankle",
    "touch_toe",
    "touch_heel",
]

ACTION_FAMILIES = [
    ("waist", "action_0"),
    ("hip", "action_1"),
    ("knee", "action_2"),
    ("ankle", "action_3"),
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def dominant(items: list[str], default: str = "unknown") -> str:
    if not items:
        return default
    return Counter(items).most_common(1)[0][0]


def action_family(row: dict[str, str], threshold: float) -> str | None:
    best_joint = None
    best_value = 0.0
    for joint, col in ACTION_FAMILIES:
        value = safe_float(row.get(col))
        if abs(value) > abs(best_value):
            best_joint = joint
            best_value = value
    if best_joint is None or abs(best_value) < threshold:
        return None
    return f"{best_joint}_{'pos' if best_value > 0 else 'neg'}"


def opposite_action_family(family: str) -> str | None:
    if "_" not in family:
        return None
    joint, sign = family.rsplit("_", 1)
    if sign == "pos":
        return f"{joint}_neg"
    if sign == "neg":
        return f"{joint}_pos"
    return None


def feature_value(feat: dict[str, torch.Tensor], key: str, idx: int) -> float:
    value = feat.get(key)
    if value is None:
        return 0.0
    return float(value[idx].detach().cpu().item())


def phase_at(
    feat: dict[str, torch.Tensor],
    idx: int,
    phase_schema: str,
    grounded_hop_as_phase: bool = False,
) -> str:
    if phase_schema == "legacy":
        if feature_value(feat, "falling_air", idx) >= 0.5:
            return "falling"
        if feature_value(feat, "grounded_rear", idx) >= 0.5:
            return "rear"
        if grounded_hop_as_phase and feature_value(feat, "grounded_hop", idx) >= 0.5:
            return "grounded_hop_candidate"
        if feature_value(feat, "grounded_prone", idx) >= 0.5:
            return "prone"
        if feature_value(feat, "grounded_upright", idx) >= 0.5 or feature_value(feat, "grounded_prepared", idx) >= 0.5:
            return "upright"
        if feature_value(feat, "rising_air", idx) >= 0.5:
            return "rising"
        return "other"
    if phase_schema != "simple_v1":
        raise ValueError(f"Unknown phase_schema: {phase_schema}")
    if feature_value(feat, "falling_air", idx) >= 0.5:
        return "air_falling"
    if feature_value(feat, "rising_air", idx) >= 0.5:
        return "air_rising"
    if feature_value(feat, "grounded_rear", idx) >= 0.5:
        return "ground_rear"
    if feature_value(feat, "grounded_prone", idx) >= 0.5:
        return "ground_front"
    if feature_value(feat, "grounded_upright", idx) >= 0.5 or feature_value(feat, "grounded_prepared", idx) >= 0.5:
        return "ground_upright"
    return "other"


def bin_omega(pitch_rate: float) -> str:
    if pitch_rate > 1.0:
        return "omega_pos"
    if pitch_rate < -1.0:
        return "omega_neg"
    return "omega_mid"


def coarse_omega_bin(omega_bin: str) -> str:
    if omega_bin == "omega_neg":
        return "omega_neg"
    return "omega_nonneg"


def bin_vx(vx: float) -> str:
    if vx >= 0.8:
        return "vx_high"
    if vx < 0.2:
        return "vx_low"
    return "vx_mid"


def bin_support(feat: dict[str, torch.Tensor], idx: int) -> str:
    full = feature_value(feat, "full_foot_support", idx)
    toe = feature_value(feat, "touch_toe", idx)
    heel = feature_value(feat, "touch_heel", idx)
    if full >= 0.35:
        return "support_full"
    if toe >= 0.5 and heel < 0.2:
        return "support_toe"
    if heel >= 0.5 and toe < 0.2:
        return "support_heel"
    if max(toe, heel) >= 0.2:
        return "support_partial"
    return "support_none"


def coarse_support_bin(support_bin: str) -> str:
    if support_bin == "support_none":
        return "support_none"
    if support_bin == "support_full":
        return "support_full"
    return "support_partial"


def risk_bin(feat: dict[str, torch.Tensor], idx: int) -> str:
    if feature_value(feat, "body_collision", idx) >= 0.5:
        return "collision_risk"
    if feature_value(feat, "rotation_penalty", idx) >= 0.5 or abs(feature_value(feat, "pitch_rate", idx)) >= 3.5:
        return "rotation_risk"
    if feature_value(feat, "falling_air", idx) >= 0.5 and feature_value(feat, "full_foot_support", idx) < 0.1:
        return "support_risk"
    return "stable"


def future_summary_legacy(
    feat: dict[str, torch.Tensor],
    rows: list[dict[str, str]],
    idx: int,
    horizon: int,
) -> dict[str, float | str]:
    lo = idx + 1
    hi = min(len(rows), idx + horizon + 1)
    if lo >= hi:
        return {
            "future_outcome": "terminal",
            "transition_score": 0.0,
            "future_rootx_delta": 0.0,
            "future_mean_vx": 0.0,
            "future_rising_frac": 0.0,
            "future_falling_frac": 0.0,
            "future_rotation_frac": 0.0,
            "future_full_foot_frac": 0.0,
            "future_high_spin_frac": 0.0,
        }

    count = hi - lo
    rising = mean([1.0 if feature_value(feat, "rising_air", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    falling = mean([1.0 if feature_value(feat, "falling_air", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    rotation = mean([1.0 if feature_value(feat, "rotation_penalty", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    full_foot = mean([1.0 if feature_value(feat, "full_foot_support", j) >= 0.35 else 0.0 for j in range(lo, hi)])
    high_spin = mean([1.0 if abs(feature_value(feat, "pitch_rate", j)) >= 3.5 else 0.0 for j in range(lo, hi)])
    vx_vals = [feature_value(feat, "vx", j) for j in range(lo, hi)]
    mean_vx = mean(vx_vals)
    rootx_delta = safe_float(rows[hi - 1].get("rootx"), safe_float(rows[hi - 1].get("qpos_rootx"))) - safe_float(
        rows[idx].get("rootx"),
        safe_float(rows[idx].get("qpos_rootx")),
    )

    score = (
        0.35 * rising
        + 0.20 * clamp(rootx_delta / 0.50)
        + 0.15 * clamp(mean_vx / 2.0)
        + 0.15 * full_foot
        - 0.30 * rotation
        - 0.25 * falling
        - 0.20 * high_spin
    )
    if rising >= 0.10 and rotation < 0.35 and falling < 0.35:
        outcome = "controlled_rise"
    elif full_foot >= 0.35 and rotation < 0.35 and falling < 0.35:
        outcome = "support_recovery"
    elif rotation >= 0.50 or high_spin >= 0.50 or falling >= 0.50:
        outcome = "unstable"
    else:
        outcome = "neutral"
    return {
        "future_outcome": outcome,
        "transition_score": float(score),
        "future_rootx_delta": float(rootx_delta),
        "future_mean_vx": float(mean_vx),
        "future_rising_frac": float(rising),
        "future_falling_frac": float(falling),
        "future_rotation_frac": float(rotation),
        "future_full_foot_frac": float(full_foot),
        "future_high_spin_frac": float(high_spin),
        "future_window_size": float(count),
    }


def future_summary_cycle_v1(
    feat: dict[str, torch.Tensor],
    rows: list[dict[str, str]],
    idx: int,
    horizon: int,
) -> dict[str, float | str]:
    """Cycle-aware outcome summary for high-quality periodic Hopper Hop traces.

    The legacy outcome treats high angular velocity, airborne intervals, and low
    full-foot support as direct instability evidence.  Those signals are useful
    during recovery analysis, but they are normal within a successful Hop cycle.
    This calibrated summary therefore judges local action consequences by
    forward progress and non-collapse over the future window.  Rotation is only
    a penalty when it appears together with poor progress or collision.
    """

    lo = idx + 1
    hi = min(len(rows), idx + horizon + 1)
    if lo >= hi:
        return {
            "future_outcome": "terminal",
            "transition_score": 0.0,
            "future_rootx_delta": 0.0,
            "future_mean_vx": 0.0,
            "future_rising_frac": 0.0,
            "future_falling_frac": 0.0,
            "future_rotation_frac": 0.0,
            "future_full_foot_frac": 0.0,
            "future_high_spin_frac": 0.0,
            "future_collision_frac": 0.0,
            "future_low_height_frac": 0.0,
            "future_support_contact_frac": 0.0,
            "future_terminal_height": 0.0,
            "future_noncollapse_score": 0.0,
            "future_progress_score": 0.0,
        }

    count = hi - lo
    rising = mean([1.0 if feature_value(feat, "rising_air", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    falling = mean([1.0 if feature_value(feat, "falling_air", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    rotation = mean([1.0 if feature_value(feat, "rotation_penalty", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    full_foot = mean([1.0 if feature_value(feat, "full_foot_support", j) >= 0.35 else 0.0 for j in range(lo, hi)])
    high_spin = mean([1.0 if abs(feature_value(feat, "pitch_rate", j)) >= 3.5 else 0.0 for j in range(lo, hi)])
    collision = mean([1.0 if feature_value(feat, "body_collision", j) >= 0.5 else 0.0 for j in range(lo, hi)])
    low_height = mean([1.0 if feature_value(feat, "height", j) < 0.35 else 0.0 for j in range(lo, hi)])
    support_contact = mean([1.0 if feature_value(feat, "support_contact", j) >= 0.30 else 0.0 for j in range(lo, hi)])
    heights = [feature_value(feat, "height", j) for j in range(lo, hi)]
    vx_vals = [feature_value(feat, "vx", j) for j in range(lo, hi)]
    mean_vx = mean(vx_vals)
    terminal_height = heights[-1] if heights else 0.0
    rootx_delta = safe_float(rows[hi - 1].get("rootx"), safe_float(rows[hi - 1].get("qpos_rootx"))) - safe_float(
        rows[idx].get("rootx"),
        safe_float(rows[idx].get("qpos_rootx")),
    )

    progress_score = clamp(rootx_delta / max(0.04 * count, 1e-6))
    speed_score = clamp(mean_vx / 1.50)
    height_score = clamp((mean(heights) - 0.35) / 0.40)
    noncollapse_score = clamp(1.0 - 1.8 * collision - 1.2 * low_height)
    contact_cycle_score = clamp(support_contact + 0.5 * full_foot)
    rotation_penalty = 0.10 * rotation if progress_score >= 0.60 and collision < 0.05 else 0.25 * rotation
    spin_penalty = 0.05 * high_spin if progress_score >= 0.60 and collision < 0.05 else 0.15 * high_spin

    score = (
        0.35 * progress_score
        + 0.20 * speed_score
        + 0.20 * noncollapse_score
        + 0.10 * height_score
        + 0.10 * contact_cycle_score
        + 0.05 * clamp(rising + falling)
        - rotation_penalty
        - spin_penalty
    )

    collapsed = collision >= 0.15 or low_height >= 0.30 or terminal_height < 0.25 or mean_vx < 0.20 or rootx_delta < 0.05
    strong_cycle = progress_score >= 0.75 and speed_score >= 0.60 and noncollapse_score >= 0.80
    recoverable_cycle = progress_score >= 0.45 and speed_score >= 0.40 and noncollapse_score >= 0.60
    if collapsed:
        outcome = "unstable"
    elif strong_cycle:
        outcome = "cycle_continue"
    elif recoverable_cycle:
        outcome = "recoverable_progress"
    else:
        outcome = "neutral"

    return {
        "future_outcome": outcome,
        "transition_score": float(score),
        "future_rootx_delta": float(rootx_delta),
        "future_mean_vx": float(mean_vx),
        "future_rising_frac": float(rising),
        "future_falling_frac": float(falling),
        "future_rotation_frac": float(rotation),
        "future_full_foot_frac": float(full_foot),
        "future_high_spin_frac": float(high_spin),
        "future_collision_frac": float(collision),
        "future_low_height_frac": float(low_height),
        "future_support_contact_frac": float(support_contact),
        "future_terminal_height": float(terminal_height),
        "future_noncollapse_score": float(noncollapse_score),
        "future_progress_score": float(progress_score),
        "future_window_size": float(count),
    }


def future_summary(
    feat: dict[str, torch.Tensor],
    rows: list[dict[str, str]],
    idx: int,
    horizon: int,
    outcome_mode: str = "legacy",
) -> dict[str, float | str]:
    if outcome_mode == "cycle_v1":
        return future_summary_cycle_v1(feat, rows, idx, horizon)
    if outcome_mode != "legacy":
        raise ValueError(f"Unknown outcome_mode: {outcome_mode}")
    return future_summary_legacy(feat, rows, idx, horizon)


def context_key(parts: dict[str, str], context_mode: str = "full") -> str:
    if context_mode == "full":
        keys = [
            ("prev", "prev_phase"),
            ("cur", "current_phase"),
            ("omega", "omega_bin"),
            ("vx", "vx_bin"),
            ("support", "support_bin"),
            ("risk", "risk_bin"),
        ]
    elif context_mode == "broad_context_v1":
        keys = [
            ("prev", "prev_phase"),
            ("cur", "current_phase"),
            ("omega", "coarse_omega_bin"),
            ("vx", "vx_bin"),
            ("support", "coarse_support_bin"),
        ]
    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")
    return "|".join(f"{label}={parts[key]}" for label, key in keys)


def read_trace(path: Path) -> tuple[list[dict[str, str]], torch.Tensor]:
    rows: list[dict[str, str]] = []
    obs: list[list[float]] = []
    with path.open() as fp:
        for row in csv.DictReader(fp):
            if not all(col in row for col in OBS_COLS):
                continue
            rows.append(row)
            obs.append([safe_float(row[col]) for col in OBS_COLS])
    return rows, torch.tensor(obs, dtype=torch.float32)


def iter_trace_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.trace_glob:
        paths.extend(Path(item) for item in glob.glob(pattern))
    if args.run_prefix:
        cycles = [int(c.strip()) for c in args.cycles.split(",") if c.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        for seed in seeds:
            seed_root = args.logs_root / str(seed)
            for cycle in cycles:
                pattern = (
                    f"{args.run_prefix}_cycle{cycle:03d}_target*/"
                    "retrospective_train_traces/step_*/episode_csv/episode_*.csv"
                )
                paths.extend(seed_root.glob(pattern))
    return sorted({p.resolve() for p in paths if p.exists()})


def extract_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    model = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
    )
    model.eval()
    records: list[dict[str, Any]] = []
    trace_paths = iter_trace_paths(args)
    args._resolved_trace_count = len(trace_paths)
    for trace_path in trace_paths:
        rows, obs = read_trace(trace_path)
        if len(rows) <= args.past_window + args.future_window + 1:
            continue
        with torch.no_grad():
            feat = model._state_reward_features(model.encode_observation(obs))
            feat = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in feat.items()}
        phases = [phase_at(feat, i, args.phase_schema, grounded_hop_as_phase=False) for i in range(len(rows))]
        for idx in range(args.past_window, len(rows) - args.future_window):
            current_raw_hop = feature_value(feat, "grounded_hop", idx) >= 0.5
            current_phase = (
                "grounded_hop_candidate"
                if args.phase_schema == "legacy" and current_raw_hop
                else phases[idx]
            )
            if args.focus_grounded_hop and not current_raw_hop:
                continue
            family = action_family(rows[idx], args.action_threshold)
            if family is None:
                continue
            past_phases = phases[idx - args.past_window:idx]
            parts = {
                "prev_phase": dominant(past_phases, default="unknown"),
                "current_phase": current_phase,
                "omega_bin": bin_omega(feature_value(feat, "pitch_rate", idx)),
                "vx_bin": bin_vx(feature_value(feat, "vx", idx)),
                "support_bin": bin_support(feat, idx),
                "risk_bin": risk_bin(feat, idx),
            }
            parts["coarse_omega_bin"] = coarse_omega_bin(parts["omega_bin"])
            parts["coarse_support_bin"] = coarse_support_bin(parts["support_bin"])
            future = future_summary(feat, rows, idx, args.future_window, args.outcome_mode)
            records.append(
                {
                    "trace_csv": str(trace_path),
                    "episode": int(safe_float(rows[idx].get("episode"), -1)),
                    "step": int(safe_float(rows[idx].get("step"), idx)),
                    "index": idx,
                    "context_key": context_key(parts, args.context_mode),
                    "action_family": family,
                    "opposite_action_family": opposite_action_family(family),
                    "raw_grounded_hop": bool(current_raw_hop),
                    "parts": parts,
                    "height": feature_value(feat, "height", idx),
                    "vx": feature_value(feat, "vx", idx),
                    "vz": feature_value(feat, "vz", idx),
                    "pitch": feature_value(feat, "pitch", idx),
                    "pitch_rate": feature_value(feat, "pitch_rate", idx),
                    "full_foot_support": feature_value(feat, "full_foot_support", idx),
                    "touch_toe": feature_value(feat, "touch_toe", idx),
                    "touch_heel": feature_value(feat, "touch_heel", idx),
                    "rotation_penalty": feature_value(feat, "rotation_penalty", idx),
                    **future,
                }
            )
    return records


def build_rules(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_context[record["context_key"]].append(record)

    rules: list[dict[str, Any]] = []
    context_summaries: list[dict[str, Any]] = []
    for key, rows in sorted(by_context.items()):
        context_support = len(rows)
        outcomes = Counter(str(r["future_outcome"]) for r in rows)
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            families[row["action_family"]].append(row)
        family_rows = []
        for family, vals in families.items():
            scores = [float(v["transition_score"]) for v in vals]
            family_rows.append(
                {
                    "action_family": family,
                    "support": len(vals),
                    "mean_score": mean(scores),
                    "controlled_rise_rate": mean([1.0 if v["future_outcome"] == "controlled_rise" else 0.0 for v in vals]),
                    "cycle_continue_rate": mean([1.0 if v["future_outcome"] == "cycle_continue" else 0.0 for v in vals]),
                    "recoverable_progress_rate": mean([1.0 if v["future_outcome"] == "recoverable_progress" else 0.0 for v in vals]),
                    "unstable_rate": mean([1.0 if v["future_outcome"] == "unstable" else 0.0 for v in vals]),
                    "support_recovery_rate": mean([1.0 if v["future_outcome"] == "support_recovery" else 0.0 for v in vals]),
                }
            )
        family_rows.sort(key=lambda row: (row["mean_score"], row["support"]), reverse=True)
        context_summaries.append(
            {
                "context_key": key,
                "support": context_support,
                "outcome_counts": dict(outcomes),
                "family_summaries": family_rows[: args.max_families_per_context_report],
            }
        )
        if context_support < args.min_context_support or not family_rows:
            continue
        winner = family_rows[0]
        if winner["support"] < args.min_family_support:
            continue
        baseline_scores = [
            float(r["transition_score"])
            for r in rows
            if r["action_family"] != winner["action_family"]
        ]
        if not baseline_scores:
            continue
        baseline = mean(baseline_scores)
        advantage = float(winner["mean_score"] - baseline)
        if advantage < args.min_score_advantage:
            continue
        rule_id = f"transition_context|{key}->prefer:{winner['action_family']}"
        rules.append(
            {
                "id": rule_id,
                "rule_key": key,
                "substate_key": key,
                "rule_role": "transition_context",
                "source": "transition_context_observational_v1",
                "rule_origin": "transition_context_extractor_v1",
                "status": "transition_context_candidate",
                "intervention_type": "soft_prefer_action_family",
                "action_family": winner["action_family"],
                "opposite_action_family": opposite_action_family(winner["action_family"]),
                "support": winner["support"],
                "context_support": context_support,
                "context_coverage": winner["support"] / context_support if context_support else 0.0,
                "mean_transition_score": winner["mean_score"],
                "baseline_transition_score": baseline,
                "score_advantage": advantage,
                "controlled_rise_rate": winner["controlled_rise_rate"],
                "cycle_continue_rate": winner["cycle_continue_rate"],
                "recoverable_progress_rate": winner["recoverable_progress_rate"],
                "unstable_rate": winner["unstable_rate"],
                "support_recovery_rate": winner["support_recovery_rate"],
                "outcome_counts": dict(outcomes),
                "audit_notes": [
                    "Observational transition-context candidate only; requires SD/TVA counterfactual validation before online use.",
                    "Rule precondition uses past/current observable context; future outcome is evidence, not an online condition.",
                ],
            }
        )

    rules.sort(key=lambda rule: (rule["score_advantage"], rule["support"]), reverse=True)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_family": "hopper_transition_context_rules_v1",
        "parameters": {
            "past_window": args.past_window,
            "future_window": args.future_window,
            "outcome_mode": args.outcome_mode,
            "phase_schema": args.phase_schema,
            "context_mode": args.context_mode,
            "action_threshold": args.action_threshold,
            "focus_grounded_hop": args.focus_grounded_hop,
            "min_context_support": args.min_context_support,
            "min_family_support": args.min_family_support,
            "min_score_advantage": args.min_score_advantage,
            "run_prefix": args.run_prefix,
            "cycles": args.cycles,
            "seeds": args.seeds,
            "trace_glob": args.trace_glob,
            "resolved_trace_count": getattr(args, "_resolved_trace_count", 0),
        },
        "num_records": len(records),
        "num_contexts": len(by_context),
        "num_rules": len(rules),
        "status_meaning": {
            "transition_context_candidate": "observational candidate; not online-ready until SD/TVA counterfactual validation",
        },
        "rules": rules[: args.max_rules],
        "context_summaries": sorted(context_summaries, key=lambda row: row["support"], reverse=True)[: args.max_contexts_report],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Transition-Context Rule Candidates",
        "",
        f"- Records: `{payload['num_records']}`",
        f"- Contexts: `{payload['num_contexts']}`",
        f"- Candidate rules: `{payload['num_rules']}`",
        "",
        "These candidates are observational. They should not be merged into an online rule library before SD/TVA counterfactual validation.",
        "",
        "## Top Rules",
        "",
        "| rule | support | ctx support | score adv | cycle continue | unstable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rule in payload["rules"][:20]:
        lines.append(
            f"| `{rule['rule_key']} -> {rule['action_family']}` | "
            f"{rule['support']} | {rule['context_support']} | "
            f"{rule['score_advantage']:.4f} | "
            f"{rule.get('cycle_continue_rate', rule.get('controlled_rise_rate', 0.0)):.3f} | "
            f"{rule['unstable_rate']:.3f} |"
        )
    lines.extend(["", "## Largest Contexts", "", "| context | support | top families |", "|---|---:|---|"])
    for ctx in payload["context_summaries"][:20]:
        families = ", ".join(
            f"{row['action_family']}:{row['support']}/{row['mean_score']:.3f}"
            for row in ctx.get("family_summaries", [])[:4]
        )
        lines.append(f"| `{ctx['context_key']}` | {ctx['support']} | {families} |")
    path.with_suffix(".md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--logs-root", type=Path, default=Path("logs/hopper-hop"))
    parser.add_argument("--cycles", default="1,5,10,15,20")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--trace-glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--past-window", type=int, default=8)
    parser.add_argument("--future-window", type=int, default=20)
    parser.add_argument("--outcome-mode", choices=("legacy", "cycle_v1"), default="legacy")
    parser.add_argument("--phase-schema", choices=("legacy", "simple_v1"), default="simple_v1")
    parser.add_argument("--context-mode", choices=("full", "broad_context_v1"), default="full")
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--focus-grounded-hop", action="store_true", default=False)
    parser.add_argument("--min-context-support", type=int, default=30)
    parser.add_argument("--min-family-support", type=int, default=8)
    parser.add_argument("--min-score-advantage", type=float, default=0.05)
    parser.add_argument("--max-rules", type=int, default=100)
    parser.add_argument("--max-contexts-report", type=int, default=100)
    parser.add_argument("--max-families-per-context-report", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_prefix and not args.trace_glob:
        raise ValueError("Provide --run-prefix or at least one --trace-glob.")
    records = extract_records(args)
    payload = build_rules(records, args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote transition-context rules: {args.output}")
    print(f"records={payload['num_records']} contexts={payload['num_contexts']} rules={payload['num_rules']}")


if __name__ == "__main__":
    main()
