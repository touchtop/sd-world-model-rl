#!/usr/bin/env python3
"""Audit online rule effects on Hopper v8 widefunnel final eval traces.

This script is read-only. It mirrors the online rule-prior state matching used
by `online_rule_prior_v6.py`, then checks whether selected eval actions are
consistent with each online rule. The result is behavioral evidence, not a
counterfactual proof of causal improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

from common import resolve_workers


DEFAULT_RUN = "tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019"
DEFAULT_RULE_LIBRARY = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/"
    "cycle_020/rules/library_closed_loop_cycle_020.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_20260509.json"
)

JOINT_TO_ACTION = {"waist": "action_0", "hip": "action_1", "knee": "action_2", "ankle": "action_3"}
ONLINE_STATUSES = {"v8_replay_safe_candidate", "confidence_active"}
ACTION_FAMILY_INTERVENTIONS = {"soft_prefer_action_family", "avoid_action_family"}
STAGE_EFFECT_INTERVENTIONS = {"stage_effect_prior"}
_AUDIT_RULES: list[dict[str, Any]] = []
_AUDIT_CONTINUE_SCORE_MIN = 0.60
_AUDIT_HIGH_EPISODE_REWARD = 100.0
_AUDIT_ACTION_THRESHOLD = 0.20


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def row_float(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    """Read a semantic feature from eval or replay CSV variants."""

    for key in keys:
        if key in row:
            return as_float(row, key, default)
    return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def omega_bin(row: dict[str, str]) -> str:
    omega = as_float(row, "pitch_rate")
    if omega > 1.0:
        return "omega_pos"
    if omega < -1.0:
        return "omega_neg"
    return "omega_mid"


def vx_bin(row: dict[str, str]) -> str:
    vx = as_float(row, "vx")
    if vx >= 0.8:
        return "vx_high"
    if vx < 0.2:
        return "vx_low"
    return "vx_mid"


def support_bin(row: dict[str, str]) -> str:
    support = as_float(row, "full_foot_support")
    if support < 0.1:
        return "support_none"
    if support < 0.35:
        return "support_low"
    return "support_some"


def phase(row: dict[str, str]) -> str:
    falling = as_float(row, "falling_air") >= 0.5 or as_float(row, "falling_air_gt_05") >= 0.5
    rear = as_float(row, "grounded_rear") >= 0.5 or as_float(row, "grounded_rear_gt_05") >= 0.5
    prone = as_float(row, "grounded_prone") >= 0.5 or as_float(row, "grounded_prone_gt_05") >= 0.5
    hop = as_float(row, "grounded_hop") >= 0.5 or as_float(row, "grounded_hop_gt_05") >= 0.5
    upright = (
        as_float(row, "grounded_upright") >= 0.5
        or as_float(row, "grounded_upright_gt_05") >= 0.5
        or as_float(row, "grounded_prepared") >= 0.5
        or as_float(row, "grounded_prepared_gt_05") >= 0.5
    )
    rising = as_float(row, "rising_air") >= 0.5 or as_float(row, "rising_air_gt_05") >= 0.5
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


def continuation_score(row: dict[str, str]) -> float:
    pitch = abs(as_float(row, "pitch"))
    torso_z = row_float(row, "torso_z", "rootz", "qpos_rootz")
    vx = as_float(row, "vx")
    omega = abs(as_float(row, "pitch_rate"))
    uprightness = max(0.0, min(1.0, 1.0 - pitch / 0.45))
    height_quality = max(0.0, min(1.0, (torso_z - 0.45) / 0.45))
    forward_speed = max(0.0, min(1.0, vx / 2.0))
    spin_control = max(0.0, min(1.0, 1.0 - omega / 3.50))
    support_readiness = max(
        row_float(row, "support_contact", "foot_grounded"),
        as_float(row, "full_foot_support"),
    )
    non_collision = 1.0 - max(0.0, min(1.0, as_float(row, "body_collision")))
    return (
        0.25 * uprightness
        + 0.20 * height_quality
        + 0.20 * forward_speed
        + 0.15 * spin_control
        + 0.10 * support_readiness
        + 0.10 * non_collision
    )


def structural_success(row: dict[str, str]) -> bool:
    return (
        row_float(row, "torso_z", "rootz", "qpos_rootz") >= 0.75
        and abs(as_float(row, "pitch")) <= 0.45
        and as_float(row, "vx") >= 0.80
        and abs(as_float(row, "pitch_rate")) <= 3.50
        and as_float(row, "body_collision") < 0.50
    )


def refined_phase(row: dict[str, str], continue_score_min: float) -> str:
    if structural_success(row) or continuation_score(row) >= continue_score_min:
        if as_float(row, "full_foot_support") >= 0.35:
            return "near_good_support"
        if row_float(row, "support_contact", "foot_grounded") >= 0.10:
            return "near_good_contact"
        return "near_good_air"
    return phase(row)


def risk_bin(row: dict[str, str]) -> str:
    ph = phase(row)
    if as_float(row, "body_collision") >= 0.5:
        return "collision_risk"
    if as_float(row, "rotation_penalty") >= 0.5 or abs(as_float(row, "pitch_rate")) >= 3.5:
        return "rotation_risk"
    if ph in {"falling", "rising"} and as_float(row, "full_foot_support") < 0.1:
        return "support_risk"
    return "stable"


def row_parts(row: dict[str, str], continue_score_min: float) -> dict[str, str]:
    rot = as_float(row, "rotation_penalty") >= 0.5 or abs(as_float(row, "pitch_rate")) >= 3.0
    torso_z = row_float(row, "torso_z", "rootz", "qpos_rootz")
    toe_recovery_low_support = (
        as_float(row, "touch_toe") >= 0.80
        and as_float(row, "touch_heel") < 0.50
        and as_float(row, "full_foot_support") < 0.35
        and as_float(row, "grounded_upright") < 0.50
        and (
            as_float(row, "grounded_prone") >= 0.50
            or torso_z <= 0.72
            or as_float(row, "rotation_penalty") >= 0.50
            or abs(as_float(row, "pitch_rate")) >= 3.0
        )
    )
    toe_recovery_recoverable_low_support = toe_recovery_low_support and (
        as_float(row, "grounded_prone") >= 0.50
        or as_float(row, "grounded_rear") >= 0.50
    )
    toe_recovery_unstable_low_support = toe_recovery_low_support and not toe_recovery_recoverable_low_support
    toe_only_launch_trap = (
        as_float(row, "touch_toe") >= 0.80
        and as_float(row, "touch_heel") < 0.20
        and row_float(row, "support_contact", "foot_grounded") >= 0.80
        and as_float(row, "full_foot_support") < 0.35
        and as_float(row, "vx") >= 0.80
        and torso_z >= 0.55
        and as_float(row, "grounded_upright") < 0.50
    )
    current_phase = "air_falling" if as_float(row, "falling_air") >= 0.50 else (
        "air_rising" if as_float(row, "rising_air") >= 0.50 else (
            "ground_rear" if as_float(row, "grounded_rear") >= 0.50 else (
                "ground_front" if as_float(row, "grounded_prone") >= 0.50 else (
                    "ground_upright"
                    if as_float(row, "grounded_upright") >= 0.50 or as_float(row, "grounded_prepared") >= 0.50
                    else "other"
                )
            )
        )
    )
    coarse_omega = "omega_neg" if omega_bin(row) == "omega_neg" else "omega_nonneg"
    if as_float(row, "full_foot_support") >= 0.35:
        broad_support = "support_full"
    elif max(as_float(row, "touch_toe"), as_float(row, "touch_heel")) >= 0.2:
        broad_support = "support_partial"
    else:
        broad_support = "support_none"
    return {
        "omega_bin": omega_bin(row),
        "coarse_omega_bin": coarse_omega,
        "vx_bin": vx_bin(row),
        "support_bin": support_bin(row),
        "broad_support_bin": broad_support,
        "phase": phase(row),
        "current_simple_phase": current_phase,
        "refined_phase": refined_phase(row, continue_score_min),
        "risk_bin": risk_bin(row),
        "rotation_risk": "1" if rot else "0",
        "toe_recovery_low_support": "1" if toe_recovery_low_support else "0",
        "toe_recovery_recoverable_low_support": "1" if toe_recovery_recoverable_low_support else "0",
        "toe_recovery_unstable_low_support": "1" if toe_recovery_unstable_low_support else "0",
        "toe_only_launch_trap": "1" if toe_only_launch_trap else "0",
    }


def high_torque(row: dict[str, str]) -> bool:
    return max(abs(as_float(row, f"action_{idx}")) for idx in range(4)) >= 0.65


def selected_action_relation(row: dict[str, str], family: str, threshold: float) -> str:
    if "_" not in family:
        return "neutral"
    joint, sign = family.split("_", 1)
    col = JOINT_TO_ACTION.get(joint)
    if col is None:
        return "neutral"
    signed = as_float(row, col) * (1.0 if sign == "pos" else -1.0)
    if signed >= threshold:
        return "same"
    if signed <= -threshold:
        return "opposite"
    return "neutral"


def default_stage_objective(rule: dict[str, Any]) -> dict[str, dict[str, float]]:
    if str(rule.get("stage_goal", "")) == "recover_full_foot_support":
        return {
            "terminal_weights": {
                "full_foot_support": 0.40,
                "support_contact": 0.25,
                "torso_z_quality": 0.15,
                "rotation_penalty": -0.15,
                "body_collision": -0.15,
            },
            "progress_weights": {
                "full_foot_support": 0.45,
                "support_contact": 0.25,
                "torso_z_quality": 0.10,
                "rotation_penalty": -0.15,
                "body_collision": -0.10,
            },
            "mean_weights": {
                "full_foot_support": 0.10,
                "support_contact": 0.10,
                "rotation_penalty": -0.10,
                "body_collision": -0.10,
            },
        }
    if str(rule.get("stage_goal", "")) == "stabilize_toe_only_launch":
        return {
            "terminal_weights": {
                "support_contact": 0.35,
                "spin_control": 0.25,
                "torso_z_quality": 0.20,
                "pitch_control": 0.15,
                "touch_heel": 0.10,
                "full_foot_support": 0.10,
                "rotation_penalty": -0.25,
                "body_collision": -0.25,
                "falling_air": -0.20,
                "airborne_true": -0.10,
            },
            "progress_weights": {
                "support_contact": 0.35,
                "spin_control": 0.20,
                "torso_z_quality": 0.10,
                "pitch_control": 0.10,
                "touch_heel": 0.10,
                "full_foot_support": 0.05,
                "rotation_penalty": -0.25,
                "body_collision": -0.20,
                "falling_air": -0.20,
            },
            "mean_weights": {
                "support_contact": 0.15,
                "spin_control": 0.10,
                "rotation_penalty": -0.10,
                "body_collision": -0.10,
            },
        }
    return {"terminal_weights": {}, "progress_weights": {}, "mean_weights": {}}


def stage_metric(row: dict[str, str], name: str) -> float:
    if name == "torso_z_quality":
        return max(0.0, min(1.0, (row_float(row, "torso_z", "rootz", "qpos_rootz") - 0.45) / 0.40))
    if name == "pitch_control":
        return max(0.0, min(1.0, 1.0 - abs(as_float(row, "pitch")) / 0.75))
    if name == "spin_control":
        return max(0.0, min(1.0, 1.0 - abs(as_float(row, "pitch_rate")) / 3.50))
    if name == "forward_speed":
        return max(0.0, min(1.0, as_float(row, "vx") / 2.0))
    if name == "support_contact":
        return max(0.0, min(1.0, row_float(row, "support_contact", "foot_grounded", default=0.0)))
    return max(0.0, min(1.0, row_float(row, name, default=0.0)))


def stage_constraint_value(
    rows: list[dict[str, str]],
    current: dict[str, str],
    future: dict[str, str],
    metric: str,
    scope: str,
) -> float:
    if scope == "initial":
        return stage_metric(current, metric)
    if scope == "terminal":
        return stage_metric(future, metric)
    if scope == "progress":
        return stage_metric(future, metric) - stage_metric(current, metric)
    values = [stage_metric(row, metric) for row in rows]
    if not values:
        return stage_metric(future, metric)
    if scope == "mean":
        return mean(values)
    if scope == "min":
        return min(values)
    if scope == "max":
        return max(values)
    raise ValueError(f"Unknown stage constraint scope: {scope}")


def stage_constraint_penalty(
    objective: dict[str, Any],
    current: dict[str, str],
    future: dict[str, str],
    window: list[dict[str, str]],
) -> float:
    penalty = 0.0
    for constraint in objective.get("stage_constraints", []) or []:
        metric = str(constraint.get("metric", ""))
        if not metric:
            continue
        scope = str(constraint.get("scope", "terminal"))
        op = str(constraint.get("op", "min"))
        threshold = float(constraint.get("threshold", 0.0))
        weight = float(constraint.get("penalty", 0.0))
        if weight <= 0.0:
            continue
        value = stage_constraint_value(window, current, future, metric, scope)
        if op == "min":
            violation = max(0.0, threshold - value)
        elif op == "max":
            violation = max(0.0, value - threshold)
        else:
            raise ValueError(f"Unknown stage constraint op: {op}")
        penalty += weight * violation
    return penalty


def stage_effect_score(rule: dict[str, Any], rows: list[dict[str, str]], idx: int) -> float:
    horizon = max(1, int(rule.get("horizon", 5) or 5))
    end = min(len(rows) - 1, idx + horizon)
    objective = rule.get("stage_objective") or rule.get("objective") or default_stage_objective(rule)
    current = rows[idx]
    future = rows[end]
    window = rows[idx + 1 : end + 1] or [future]
    score = 0.0
    for name, weight in dict(objective.get("terminal_weights", {})).items():
        score += float(weight) * stage_metric(future, name)
    for name, weight in dict(objective.get("progress_weights", {})).items():
        score += float(weight) * (stage_metric(future, name) - stage_metric(current, name))
    for name, weight in dict(objective.get("mean_weights", {})).items():
        score += float(weight) * mean([stage_metric(row, name) for row in window])
    score -= stage_constraint_penalty(objective, current, future, window)
    return score


def stage_effect_relation(rule: dict[str, Any], rows: list[dict[str, str]], idx: int) -> str:
    score = stage_effect_score(rule, rows, idx)
    positive_threshold = float(rule.get("audit_positive_threshold", 0.02) or 0.02)
    harmful_threshold = float(rule.get("audit_harmful_threshold", -0.02) or -0.02)
    if score >= positive_threshold:
        return "same"
    if score <= harmful_threshold:
        return "opposite"
    return "neutral"


def rule_id(rule: dict[str, Any]) -> str:
    return str(
        rule.get("id")
        or f"{rule.get('rule_key') or rule.get('substate_key')}->{rule.get('intervention_type')}:{rule.get('action_family')}"
    )


def state_matches(rule: dict[str, Any], parts: dict[str, str]) -> bool:
    key = str(rule.get("substate_key") or rule.get("rule_key") or "")
    if key.startswith("prev="):
        items: dict[str, str] = {}
        for token in key.split("|"):
            if "=" not in token:
                return False
            name, value = token.split("=", 1)
            items[name] = value
        return (
            items.get("prev") == parts.get("prev_simple_phase")
            and items.get("cur") == parts.get("current_simple_phase")
            and items.get("omega") == parts.get("coarse_omega_bin")
            and items.get("vx") == parts.get("vx_bin")
            and items.get("support") == parts.get("broad_support_bin")
        )
    tokens = key.split("|")
    if len(tokens) == 5:
        event, omega, vx, support, ph = tokens
        event_supported = event == "unsupported_high_torque" or (
            event == "rotation_risk" and parts.get("rotation_risk") == "1"
        )
        return (
            event_supported
            and omega == parts["omega_bin"]
            and vx == parts["vx_bin"]
            and support == parts["support_bin"]
            and ph == parts["phase"]
        )
    if len(tokens) == 7:
        role, _quality, omega, vx, support, ph, risk = tokens
        phase_matches = ph in {"phase_any", "*", "any"} or ph == parts["refined_phase"]
        return (
            role == "transition"
            and omega == parts["omega_bin"]
            and vx == parts["vx_bin"]
            and support == parts["support_bin"]
            and phase_matches
            and risk == parts["risk_bin"]
        )
    if key == "toe_recovery_low_support":
        return parts.get("toe_recovery_low_support") == "1"
    if key == "toe_recovery_recoverable_low_support":
        return parts.get("toe_recovery_recoverable_low_support") == "1"
    if key == "toe_recovery_unstable_low_support":
        return parts.get("toe_recovery_unstable_low_support") == "1"
    if key == "toe_only_launch_trap":
        return parts.get("toe_only_launch_trap") == "1"
    return False


def selected_gate_applies(rule: dict[str, Any], row: dict[str, str]) -> bool:
    if rule.get("intervention_type") in STAGE_EFFECT_INTERVENTIONS:
        return True
    key = str(rule.get("substate_key") or rule.get("rule_key") or "")
    if key.startswith("toe_recovery_") or key == "toe_only_launch_trap":
        return True
    if key.startswith("prev=") or rule.get("source") == "sd_tva_broad_context_counterfactual":
        return True
    if rule.get("rule_role") == "transition" or key.startswith("transition|"):
        return True
    event = str(rule.get("state_event") or str(rule.get("substate_key", "")).split("|", 1)[0])
    if event == "rotation_risk":
        return True
    return high_torque(row)


def load_online_rules(path: Path, statuses: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for rule in payload.get("rules", []):
        if rule.get("status") not in statuses:
            continue
        intervention = rule.get("intervention_type")
        if rule.get("source") == "v4_substate" and intervention == "soft_prefer_action_family":
            out.append(rule)
        elif rule.get("source") == "v8_transition_replay_safety" and intervention in ACTION_FAMILY_INTERVENTIONS:
            out.append(rule)
        elif rule.get("source") == "sd_diagnostic_toe_recovery" and intervention in ACTION_FAMILY_INTERVENTIONS:
            out.append(rule)
        elif rule.get("source") == "sd_tva_broad_context_counterfactual" and intervention in ACTION_FAMILY_INTERVENTIONS:
            out.append(rule)
        elif rule.get("source") in {"sd_stage_effect", "sd_diagnostic_toe_recovery"} and intervention in STAGE_EFFECT_INTERVENTIONS:
            out.append(rule)
    return out


def trace_files(run_name: str, seeds: list[int], step: str) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for seed in seeds:
        root = Path(f"logs/hopper-hop/{seed}/{run_name}/official_eval/{step}/episode_csv")
        out.extend((seed, path) for path in sorted(root.glob("episode_*.csv")))
    return out


def infer_seed(path: Path, fallback: int) -> int:
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part == "hopper-hop" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return fallback
    return fallback


def trace_files_from_dirs(trace_dirs: list[Path]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for idx, trace_dir in enumerate(trace_dirs):
        for path in sorted(trace_dir.glob("episode_*.csv")):
            out.append((infer_seed(path, idx), path))
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sqrt(vx * vy)


def init_rule_stats(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rule_id(rule),
        "source": rule.get("source"),
        "intervention_type": rule.get("intervention_type"),
        "action_family": rule.get("action_family"),
        "rule_key": rule.get("rule_key") or rule.get("substate_key"),
        "state_hits": 0,
        "selected_gate_hits": 0,
        "same": 0,
        "opposite": 0,
        "neutral": 0,
        "stage_effective": 0,
        "stage_harmful": 0,
        "good_state_hits": 0,
        "structural_hits": 0,
        "positive_reward_hits": 0,
        "high_episode_hits": 0,
        "seed_hits": Counter(),
        "episode_hits": Counter(),
        "phase_hits": Counter(),
        "rewards_on_hit_episodes": [],
    }


def init_audit_worker(
    rules: list[dict[str, Any]],
    continue_score_min: float,
    high_episode_reward: float,
    action_threshold: float,
) -> None:
    global _AUDIT_RULES, _AUDIT_CONTINUE_SCORE_MIN, _AUDIT_HIGH_EPISODE_REWARD, _AUDIT_ACTION_THRESHOLD
    _AUDIT_RULES = rules
    _AUDIT_CONTINUE_SCORE_MIN = continue_score_min
    _AUDIT_HIGH_EPISODE_REWARD = high_episode_reward
    _AUDIT_ACTION_THRESHOLD = action_threshold


def merge_rule_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "state_hits",
        "selected_gate_hits",
        "same",
        "opposite",
        "neutral",
        "stage_effective",
        "stage_harmful",
        "good_state_hits",
        "structural_hits",
        "positive_reward_hits",
        "high_episode_hits",
    ):
        dst[key] += src[key]
    for key in ("seed_hits", "episode_hits", "phase_hits"):
        dst[key].update(src[key])
    dst["rewards_on_hit_episodes"].extend(src["rewards_on_hit_episodes"])


def audit_trace(trace: tuple[int, Path]) -> dict[str, Any]:
    seed, path = trace
    rows = read_csv(path)
    episode = int(path.stem.split("_", 1)[1])
    reward = sum(as_float(row, "reward") for row in rows)
    ep_key = f"seed{seed}_ep{episode:02d}"
    local_rule_stats: dict[str, dict[str, Any]] = {}
    all_rows = 0
    any_rule_state_hit_rows = 0
    any_rule_gate_hit_rows = 0
    state_hits = 0
    gate_hits = 0
    transition_gate_hits = 0
    v4_gate_hits = 0
    effective_good_hits = 0
    prev_simple_phase = "unknown"
    for row_idx, row in enumerate(rows):
        all_rows += 1
        parts = row_parts(row, _AUDIT_CONTINUE_SCORE_MIN)
        parts["prev_simple_phase"] = prev_simple_phase
        prev_simple_phase = parts["current_simple_phase"]
        row_state_hit = False
        row_gate_hit = False
        good_state = (
            structural_success(row)
            or as_float(row, "reward") > 0.0
            or as_float(row, "dense_pose_potential") >= 0.65
            or continuation_score(row) >= _AUDIT_CONTINUE_SCORE_MIN
        )
        for rule in _AUDIT_RULES:
            if not state_matches(rule, parts):
                continue
            rid = rule_id(rule)
            stats = local_rule_stats.setdefault(rid, init_rule_stats(rule))
            row_state_hit = True
            state_hits += 1
            stats["state_hits"] += 1
            stats["phase_hits"][parts["phase"]] += 1
            if good_state:
                stats["good_state_hits"] += 1
            if structural_success(row):
                stats["structural_hits"] += 1
            if as_float(row, "reward") > 0.0:
                stats["positive_reward_hits"] += 1
            if reward >= _AUDIT_HIGH_EPISODE_REWARD:
                stats["high_episode_hits"] += 1
            stats["seed_hits"][str(seed)] += 1
            stats["episode_hits"][ep_key] += 1
            if selected_gate_applies(rule, row):
                row_gate_hit = True
                gate_hits += 1
                stats["selected_gate_hits"] += 1
                if rule.get("source") == "v8_transition_replay_safety":
                    transition_gate_hits += 1
                else:
                    v4_gate_hits += 1
                if rule.get("intervention_type") in STAGE_EFFECT_INTERVENTIONS:
                    rel = stage_effect_relation(rule, rows, row_idx)
                else:
                    rel = selected_action_relation(row, str(rule.get("action_family", "")), _AUDIT_ACTION_THRESHOLD)
                stats[rel] += 1
                if rule.get("intervention_type") in STAGE_EFFECT_INTERVENTIONS:
                    if rel == "same":
                        effective_good_hits += 1
                        stats["stage_effective"] += 1
                    elif rel == "opposite":
                        stats["stage_harmful"] += 1
                elif rule.get("intervention_type") == "avoid_action_family":
                    if rel != "same":
                        effective_good_hits += 1
                elif rel != "opposite":
                    effective_good_hits += 1
                stats["rewards_on_hit_episodes"].append(reward)
        if row_state_hit:
            any_rule_state_hit_rows += 1
        if row_gate_hit:
            any_rule_gate_hit_rows += 1
    return {
        "all_rows": all_rows,
        "any_rule_state_hit_rows": any_rule_state_hit_rows,
        "any_rule_gate_hit_rows": any_rule_gate_hit_rows,
        "rule_stats": local_rule_stats,
        "episode_row": {
            "seed": seed,
            "episode": episode,
            "episode_reward": reward,
            "steps": len(rows),
            "state_hits": state_hits,
            "gate_hits": gate_hits,
            "transition_gate_hits": transition_gate_hits,
            "v4_gate_hits": v4_gate_hits,
            "rule_hit_density": gate_hits / len(rows) if rows else 0.0,
            "transition_rule_hit_density": transition_gate_hits / len(rows) if rows else 0.0,
            "effective_direction_frac": effective_good_hits / gate_hits if gate_hits else 0.0,
            "trace_csv": str(path),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rules = load_online_rules(args.rule_library, set(args.audit_status))
    rule_stats = {rule_id(rule): init_rule_stats(rule) for rule in rules}
    source_counts = Counter(rule.get("source") for rule in rules)
    intervention_counts = Counter(rule.get("intervention_type") for rule in rules)
    episode_rows: list[dict[str, Any]] = []
    all_rows = 0
    any_rule_state_hit_rows = 0
    any_rule_gate_hit_rows = 0

    traces = trace_files_from_dirs(args.trace_dir) if args.trace_dir else trace_files(args.run_name, args.seeds, args.step)
    worker_count = resolve_workers(args.workers)
    if worker_count <= 1 or len(traces) <= 1:
        init_audit_worker(rules, args.continue_score_min, args.high_episode_reward, args.action_threshold)
        trace_results = [audit_trace(trace) for trace in traces]
    else:
        chunksize = max(1, len(traces) // (worker_count * 4))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_audit_worker,
            initargs=(rules, args.continue_score_min, args.high_episode_reward, args.action_threshold),
        ) as pool:
            trace_results = list(pool.map(audit_trace, traces, chunksize=chunksize))

    for result in trace_results:
        all_rows += result["all_rows"]
        any_rule_state_hit_rows += result["any_rule_state_hit_rows"]
        any_rule_gate_hit_rows += result["any_rule_gate_hit_rows"]
        episode_rows.append(result["episode_row"])
        for rid, stats in result["rule_stats"].items():
            merge_rule_stats(rule_stats[rid], stats)

    rule_rows: list[dict[str, Any]] = []
    for rid, stats in rule_stats.items():
        gate_hits = int(stats["selected_gate_hits"])
        state_hits = int(stats["state_hits"])
        same = int(stats["same"])
        opposite = int(stats["opposite"])
        neutral = int(stats["neutral"])
        intervention = stats["intervention_type"]
        if intervention in STAGE_EFFECT_INTERVENTIONS:
            effective = int(stats["stage_effective"])
            harmful = int(stats["stage_harmful"])
        elif intervention == "avoid_action_family":
            effective = gate_hits - same
            harmful = same
        else:
            effective = gate_hits - opposite
            harmful = opposite
        rule_rows.append(
            {
                "id": rid,
                "source": stats["source"],
                "intervention_type": intervention,
                "action_family": stats["action_family"],
                "rule_key": stats["rule_key"],
                "state_hits": state_hits,
                "selected_gate_hits": gate_hits,
                "gate_hit_rate_among_state_hits": gate_hits / state_hits if state_hits else 0.0,
                "same": same,
                "opposite": opposite,
                "neutral": neutral,
                "stage_effective": int(stats["stage_effective"]),
                "stage_harmful": int(stats["stage_harmful"]),
                "same_rate": same / gate_hits if gate_hits else 0.0,
                "opposite_rate": opposite / gate_hits if gate_hits else 0.0,
                "neutral_rate": neutral / gate_hits if gate_hits else 0.0,
                "direction_consistent_rate": effective / gate_hits if gate_hits else 0.0,
                "direction_harmful_rate": harmful / gate_hits if gate_hits else 0.0,
                "good_state_hit_rate": stats["good_state_hits"] / state_hits if state_hits else 0.0,
                "structural_hit_rate": stats["structural_hits"] / state_hits if state_hits else 0.0,
                "positive_reward_hit_rate": stats["positive_reward_hits"] / state_hits if state_hits else 0.0,
                "high_episode_hit_rate": stats["high_episode_hits"] / state_hits if state_hits else 0.0,
                "mean_reward_of_hit_episodes": mean(stats["rewards_on_hit_episodes"])
                if stats["rewards_on_hit_episodes"]
                else None,
                "seed_hits": dict(stats["seed_hits"]),
                "top_episode_hits": dict(stats["episode_hits"].most_common(5)),
                "phase_hits": dict(stats["phase_hits"]),
            }
        )
    rule_rows.sort(
        key=lambda row: (
            row["selected_gate_hits"],
            row["direction_consistent_rate"],
            row["good_state_hit_rate"],
        ),
        reverse=True,
    )

    rewards = [row["episode_reward"] for row in episode_rows]
    gate_fracs = [row["rule_hit_density"] for row in episode_rows]
    transition_fracs = [row["transition_rule_hit_density"] for row in episode_rows]
    return {
        "audit_family": "hopper-v8-widefunnel-rule-effect-audit-v0",
        "run_name": args.run_name,
        "step": args.step,
        "rule_library": str(args.rule_library),
        "num_online_rules": len(rules),
        "audit_statuses": args.audit_status,
        "source_counts": dict(source_counts),
        "intervention_counts": dict(intervention_counts),
        "workers": worker_count,
        "num_eval_episodes": len(episode_rows),
        "total_rows": all_rows,
        "any_rule_state_hit_rows": any_rule_state_hit_rows,
        "any_rule_state_hit_frac": any_rule_state_hit_rows / all_rows if all_rows else 0.0,
        "any_rule_selected_gate_hit_rows": any_rule_gate_hit_rows,
        "any_rule_selected_gate_hit_frac": any_rule_gate_hit_rows / all_rows if all_rows else 0.0,
        "episode_reward_mean": mean(rewards) if rewards else 0.0,
        "episode_reward_max": max(rewards) if rewards else 0.0,
        "gate_hit_reward_corr": pearson(gate_fracs, rewards),
        "transition_gate_hit_reward_corr": pearson(transition_fracs, rewards),
        "episode_summaries": sorted(episode_rows, key=lambda row: row["episode_reward"], reverse=True),
        "rule_summaries": rule_rows,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_rules: int) -> None:
    lines = [
        "# Hopper v8 Widefunnel Rule Effect Audit",
        "",
        f"Run: `{payload['run_name']}`",
        f"Rules: `{payload['num_online_rules']}`",
        f"Eval episodes: `{payload['num_eval_episodes']}`",
        f"Any-rule selected gate hit frac: `{payload['any_rule_selected_gate_hit_frac']:.4f}`",
        f"Rule-hit-density / reward correlation: `{payload['gate_hit_reward_corr']}`",
        f"Transition-hit-density / reward correlation: `{payload['transition_gate_hit_reward_corr']}`",
        "",
        "## Rule Summary",
        "",
        "| rule | source | intervention | hits | same | opposite | neutral | consistent% | harmful% | good_state% | high_ep% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rule_summaries"][:max_rules]:
        lines.append(
            f"| `{row['id']}` | {row['source']} | {row['intervention_type']} | "
            f"{row['selected_gate_hits']} | {row['same']} | {row['opposite']} | {row['neutral']} | "
            f"{100.0 * row['direction_consistent_rate']:.1f} | {100.0 * row['direction_harmful_rate']:.1f} | "
            f"{100.0 * row['good_state_hit_rate']:.1f} | {100.0 * row['high_episode_hit_rate']:.1f} |"
        )
    lines += [
        "",
        "## Top Episodes",
        "",
        "| seed | episode | reward | rule_hit_density | transition_hit_density | effective_direction_frac | trace |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["episode_summaries"][:12]:
        lines.append(
            f"| {row['seed']} | {row['episode']} | {row['episode_reward']:.6f} | "
            f"{row['rule_hit_density']:.4f} | {row['transition_rule_hit_density']:.4f} | "
            f"{row['effective_direction_frac']:.4f} | `{row['trace_csv']}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--step", default="step_0100019")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--trace-dir", action="append", type=Path, default=[])
    parser.add_argument("--audit-status", action="append", default=sorted(ONLINE_STATUSES))
    parser.add_argument("--rule-library", type=Path, default=DEFAULT_RULE_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-threshold", type=float, default=0.2)
    parser.add_argument("--continue-score-min", type=float, default=0.60)
    parser.add_argument("--high-episode-reward", type=float, default=100.0)
    parser.add_argument("--max-rules-md", type=int, default=40)
    parser.add_argument("--workers", type=int, default=0, help="CPU worker processes; 0 uses SD_RETRO_WORKERS or a bounded auto default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output, payload, args.max_rules_md)
    print(f"wrote audit: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(
        "rules={rules} episodes={episodes} gate_hit_frac={hit:.4f}".format(
            rules=payload["num_online_rules"],
            episodes=payload["num_eval_episodes"],
            hit=payload["any_rule_selected_gate_hit_frac"],
        )
    )


if __name__ == "__main__":
    main()
