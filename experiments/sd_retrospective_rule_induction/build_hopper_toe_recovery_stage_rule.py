#!/usr/bin/env python3
"""Append a stage-effect toe-recovery rule to an existing Hopper rule library."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import load_json, write_json


DEFAULT_BASE = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/"
    "cycle_040/rules/library_closed_loop_cycle_040_confidence.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/rules/"
    "hopper_cycle040_confidence_plus_toe_recovery_stage_constraint.json"
)


def stabilize_objective() -> dict[str, Any]:
    return {
        "terminal_weights": {
            "support_contact": 0.25,
            "touch_heel": 0.30,
            "full_foot_support": 0.30,
            "spin_control": 0.15,
            "pitch_control": 0.10,
            "torso_z_quality": 0.10,
            "rotation_penalty": -0.35,
            "body_collision": -0.30,
            "falling_air": -0.45,
            "airborne_true": -0.20,
        },
        "progress_weights": {
            "support_contact": 0.25,
            "touch_heel": 0.45,
            "full_foot_support": 0.40,
            "spin_control": 0.10,
            "pitch_control": 0.10,
            "rotation_penalty": -0.35,
            "body_collision": -0.25,
            "falling_air": -0.45,
        },
        "mean_weights": {
            "support_contact": 0.20,
            "spin_control": 0.08,
            "rotation_penalty": -0.15,
            "falling_air": -0.25,
            "body_collision": -0.10,
        },
        "constraint_priority_weight": 0.18,
        "constraint_satisfied_bonus": 0.03,
        "constraint_absolute_weight": 1.0,
        "stage_constraints": [
            {
                "metric": "support_contact",
                "scope": "min",
                "op": "min",
                "threshold": 0.55,
                "penalty": 0.20,
                "reason": "候选序列不能通过丢失支撑来换取短期姿态分数。",
            },
            {
                "metric": "support_contact",
                "scope": "terminal",
                "op": "min",
                "threshold": 0.65,
                "penalty": 0.16,
                "reason": "阶段结束时仍应保留可继续恢复的支撑基础。",
            },
            {
                "metric": "touch_heel",
                "scope": "progress",
                "op": "min",
                "threshold": 0.05,
                "penalty": 0.08,
                "reason": "脚尖支撑恢复应显式推动脚跟接触，而不是停留在脚尖弹射。",
            },
            {
                "metric": "full_foot_support",
                "scope": "progress",
                "op": "min",
                "threshold": 0.03,
                "penalty": 0.08,
                "reason": "规则目标包含向全足支撑恢复的最小进展。",
            },
            {
                "metric": "falling_air",
                "scope": "max",
                "op": "max",
                "threshold": 0.25,
                "penalty": 0.16,
                "reason": "避免候选序列在恢复窗口内转入跌落状态。",
            },
            {
                "metric": "rotation_penalty",
                "scope": "max",
                "op": "max",
                "threshold": 0.35,
                "penalty": 0.12,
                "reason": "避免通过增加旋转风险换取支撑或高度信号。",
            },
        ],
    }


def unstable_objective() -> dict[str, Any]:
    return {
        "terminal_weights": {
            "support_contact": 0.30,
            "spin_control": 0.30,
            "pitch_control": 0.15,
            "touch_heel": 0.08,
            "full_foot_support": 0.05,
            "rotation_penalty": -0.35,
            "falling_air": -0.35,
            "body_collision": -0.25,
            "airborne_true": -0.15,
        },
        "progress_weights": {
            "support_contact": 0.25,
            "spin_control": 0.25,
            "rotation_penalty": -0.35,
            "falling_air": -0.35,
            "body_collision": -0.20,
            "touch_heel": 0.08,
        },
        "mean_weights": {
            "spin_control": 0.12,
            "rotation_penalty": -0.12,
            "falling_air": -0.12,
            "body_collision": -0.10,
        },
    }


def stage_rule(
    *,
    rule_key: str,
    stage_goal: str,
    confidence: float,
    status: str,
    horizon: int,
    stage_prior_weight: float,
    stage_max_bonus: float,
    objective: dict[str, Any],
    audit_notes: list[str],
) -> dict[str, Any]:
    return {
        "id": f"{rule_key}->stage_effect:{stage_goal}",
        "source": "sd_stage_effect",
        "rule_key": rule_key,
        "intervention_type": "stage_effect_prior",
        "stage_goal": stage_goal,
        "horizon": horizon,
        "confidence": confidence,
        "confidence_status": status,
        "status": status,
        "effective_prior_scale": confidence if status == "confidence_active" else 0.0,
        "stage_prior_weight": stage_prior_weight,
        "stage_max_bonus": stage_max_bonus,
        "center_stage_score": True,
        "normalize_stage_score": True,
        "audit_positive_threshold": 0.02,
        "audit_harmful_threshold": -0.02,
        "stage_objective": objective,
        "audit_notes": audit_notes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def split_stage_rules(
    confidence: float,
    horizon: int,
    stage_prior_weight: float,
    stage_max_bonus: float,
) -> list[dict[str, Any]]:
    return [
        stage_rule(
            rule_key="toe_recovery_recoverable_low_support",
            stage_goal="constrain_recoverable_toe_support",
            confidence=confidence,
            status="confidence_active",
            horizon=horizon,
            stage_prior_weight=stage_prior_weight,
            stage_max_bonus=stage_max_bonus,
            objective=stabilize_objective(),
            audit_notes=[
                "该规则仅在 toe_recovery_low_support 且 grounded_prone/grounded_rear 成立时触发，对应现有数据中方向一致率较高的可恢复子阶段。",
                "阶段目标不再只做软加权评分，而是对支撑丢失、跌落、旋转风险和缺少脚跟/全足恢复进展施加显式约束惩罚。",
                "约束违背量作为候选排序的优先项，软阶段分数只在约束水平相近的候选之间发挥细分作用。",
            ],
        ),
        stage_rule(
            rule_key="toe_recovery_unstable_low_support",
            stage_goal="observe_unstable_toe_support",
            confidence=0.35,
            status="confidence_observing",
            horizon=horizon,
            stage_prior_weight=stage_prior_weight * 0.5,
            stage_max_bonus=stage_max_bonus * 0.5,
            objective=unstable_objective(),
            audit_notes=[
                "该规则覆盖 falling_air/toe_support 等现有数据中有害率较高的低支撑子阶段，初始仅观察，不进入在线 prior。",
                "若后续闭环审计显示该子阶段目标稳定有效，再由 confidence 机制自然增强；否则保持观察或退役。",
            ],
        ),
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    library = load_json(args.base_library)
    out = dict(library)
    stage_ids = {
        "toe_recovery_low_support->soft_prefer:waist_pos",
        "toe_recovery_low_support->stage_effect:recover_full_foot_support",
        "toe_recovery_low_support->stage_effect:stabilize_toe_only_launch",
        "toe_only_launch_trap->stage_effect:stabilize_toe_only_launch",
        "toe_recovery_recoverable_low_support->stage_effect:stabilize_recoverable_toe_support",
        "toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support",
        "toe_recovery_unstable_low_support->stage_effect:observe_unstable_toe_support",
    }
    rules = [rule for rule in library.get("rules", []) if rule.get("id") not in stage_ids]
    rules.extend(split_stage_rules(args.confidence, args.horizon, args.stage_prior_weight, args.stage_max_bonus))
    out["rules"] = rules
    out["library_version"] = f"{library.get('library_version', 'rule_library')}_toe_recovery_stage_constraint"
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["source_library"] = str(args.base_library)
    intervention_type_meaning = dict(out.get("intervention_type_meaning", {}))
    intervention_type_meaning["stage_effect_prior"] = (
        "stage-level prior that scores candidate action sequences by short-horizon state-effect objectives"
    )
    out["intervention_type_meaning"] = intervention_type_meaning
    out["online_enabled"] = True
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-library", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--stage-prior-weight", type=float, default=0.04)
    parser.add_argument("--stage-max-bonus", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args)
    write_json(args.output, payload)
    print(f"wrote stage-effect rule library: {args.output}")
    print(f"rules: {len(payload.get('rules', []))}")


if __name__ == "__main__":
    main()
