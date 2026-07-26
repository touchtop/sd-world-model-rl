#!/usr/bin/env python3
"""Audit whether a Hopper stage-effect rule ranks desired candidate sequences.

This script is offline and read-only. It samples probe action sequences from
saved replay traces, rolls them forward with the fitted SD explainer, and checks
whether the stage-effect prior ranks sequences that satisfy the intended
short-horizon structural constraints above sequences that lose support or enter
falling/rotation-risk states.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import f, load_json, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from counterfactual_hopper_v8_transition_actions import load_model  # noqa: E402


DEFAULT_RULE_LIBRARY = Path(
    "results/sd_retrospective_rule_induction/rules/"
    "hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json"
)
DEFAULT_BASELINE_RULE_LIBRARY = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/"
    "cycle_037/rules/library_closed_loop_cycle_037_confidence.json"
)
DEFAULT_SD_CHECKPOINT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/"
    "cycle_037/sd_explainers/sd_explainer_cycle_037.pt"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "hopper_185k_cycle037_stage_constraint_candidate_ranking.json"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def support_contact(row: dict[str, str]) -> float:
    if "support_contact" in row and row.get("support_contact") not in {None, ""}:
        return f(row, "support_contact")
    return f(row, "foot_grounded")


def is_recoverable_toe_support(row: dict[str, str]) -> bool:
    toe_recovery_low_support = (
        f(row, "touch_toe") >= 0.80
        and f(row, "touch_heel") < 0.50
        and f(row, "full_foot_support") < 0.35
        and f(row, "grounded_upright") < 0.50
        and (
            f(row, "grounded_prone") >= 0.50
            or f(row, "rootz", f(row, "qpos_rootz")) <= 0.72
            or f(row, "rotation_penalty") >= 0.50
            or abs(f(row, "pitch_rate")) >= 3.0
        )
    )
    return toe_recovery_low_support and (f(row, "grounded_prone") >= 0.50 or f(row, "grounded_rear") >= 0.50)


def find_rule(library: dict[str, Any], preferred_id: str | None, rule_key: str) -> dict[str, Any]:
    rules = library.get("rules", [])
    if preferred_id:
        for rule in rules:
            if rule.get("id") == preferred_id:
                return rule
    candidates = [
        rule
        for rule in rules
        if rule.get("rule_key") == rule_key and rule.get("intervention_type") == "stage_effect_prior"
    ]
    if not candidates:
        raise ValueError(f"No stage-effect rule found for rule_key={rule_key}")
    active = [rule for rule in candidates if rule.get("confidence_status") == "confidence_active"]
    return active[0] if active else candidates[0]


def metric_value(feat: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name == "torso_z_quality":
        return torch.clamp((feat["torso_z"] - 0.45) / 0.40, min=0.0, max=1.0)
    if name == "pitch_control":
        return torch.clamp(1.0 - torch.abs(feat["pitch"]) / 0.75, min=0.0, max=1.0)
    if name == "spin_control":
        return torch.clamp(1.0 - torch.abs(feat["pitch_rate"]) / 3.50, min=0.0, max=1.0)
    if name == "forward_speed":
        return torch.clamp(feat["vx"] / 2.0, min=0.0, max=1.0)
    return torch.clamp(feat[name], min=0.0, max=1.0)


def constraint_value(
    init_feat: dict[str, torch.Tensor],
    rollout_feats: list[dict[str, torch.Tensor]],
    metric: str,
    scope: str,
) -> torch.Tensor:
    values = [metric_value(feat, metric) for feat in rollout_feats]
    if scope == "initial":
        return metric_value(init_feat, metric)
    if scope == "terminal":
        return values[-1]
    if scope == "progress":
        return values[-1] - metric_value(init_feat, metric)
    stacked = torch.stack(values, dim=0)
    if scope == "mean":
        return stacked.mean(dim=0)
    if scope == "min":
        return stacked.min(dim=0).values
    if scope == "max":
        return stacked.max(dim=0).values
    raise ValueError(f"Unknown scope: {scope}")


def stage_constraint_penalty(
    objective: dict[str, Any],
    init_feat: dict[str, torch.Tensor],
    rollout_feats: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    penalty = torch.zeros_like(metric_value(rollout_feats[-1], "support_contact"))
    for constraint in objective.get("stage_constraints", []) or []:
        metric = str(constraint.get("metric", ""))
        if not metric:
            continue
        value = constraint_value(init_feat, rollout_feats, metric, str(constraint.get("scope", "terminal")))
        threshold = float(constraint.get("threshold", 0.0))
        weight = float(constraint.get("penalty", 0.0))
        if str(constraint.get("op", "min")) == "min":
            violation = torch.relu(torch.as_tensor(threshold, dtype=value.dtype, device=value.device) - value)
        else:
            violation = torch.relu(value - torch.as_tensor(threshold, dtype=value.dtype, device=value.device))
        penalty = penalty + weight * violation
    return penalty


def stage_scores(
    rule: dict[str, Any],
    init_feat: dict[str, torch.Tensor],
    rollout_feats: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    objective = rule.get("stage_objective") or rule.get("objective") or {}
    terminal_feat = rollout_feats[-1]
    score = torch.zeros_like(metric_value(terminal_feat, "support_contact"))
    for name, weight in dict(objective.get("terminal_weights", {})).items():
        score = score + float(weight) * metric_value(terminal_feat, name)
    for name, weight in dict(objective.get("progress_weights", {})).items():
        score = score + float(weight) * (metric_value(terminal_feat, name) - metric_value(init_feat, name))
    for name, weight in dict(objective.get("mean_weights", {})).items():
        values = [metric_value(feat, name) for feat in rollout_feats]
        score = score + float(weight) * torch.stack(values, dim=0).mean(dim=0)

    normalized = score
    if bool(rule.get("center_stage_score", True)):
        normalized = normalized - normalized.mean()
    if bool(rule.get("normalize_stage_score", True)):
        normalized = normalized / normalized.abs().mean().clamp_min(1e-6)
    weighted = float(rule.get("stage_prior_weight", 0.05) or 0.05) * normalized
    penalty = stage_constraint_penalty(objective, init_feat, rollout_feats)
    final_unclamped = weighted
    priority_weight = float(objective.get("constraint_priority_weight", rule.get("constraint_priority_weight", 0.0)) or 0.0)
    if priority_weight > 0.0:
        penalty_span = penalty.max() - penalty.min()
        if torch.isfinite(penalty_span) and penalty_span.item() > 1e-6:
            relative_penalty = (penalty - penalty.min()) / penalty_span.clamp_min(1e-6)
            final_unclamped = final_unclamped - priority_weight * relative_penalty
        satisfied_bonus = float(objective.get("constraint_satisfied_bonus", rule.get("constraint_satisfied_bonus", 0.0)) or 0.0)
        if satisfied_bonus > 0.0:
            final_unclamped = final_unclamped + satisfied_bonus * (penalty <= 1e-6).to(final_unclamped.dtype)
    absolute_weight = float(objective.get("constraint_absolute_weight", rule.get("constraint_absolute_weight", 1.0)) or 0.0)
    final = (final_unclamped - absolute_weight * penalty).clamp(
        min=-float(rule.get("stage_max_bonus", 0.10) or 0.10),
        max=float(rule.get("stage_max_bonus", 0.10) or 0.10),
    )
    return final, weighted, penalty


def rollout_features(model: torch.nn.Module, row: dict[str, str], actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32).unsqueeze(0))
    state = state.repeat(actions.shape[1], 1)
    init_feat = model._state_reward_features(state)
    feats: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for t in range(actions.shape[0]):
            state = model(state, actions[t])
            feats.append(model._state_reward_features(state))
    return init_feat, feats


def candidate_sequences(
    rows: list[dict[str, str]],
    idx: int,
    horizon: int,
    amplitudes: list[float],
    random_sequences: int,
    rng: random.Random,
) -> tuple[list[str], torch.Tensor]:
    names: list[str] = []
    sequences: list[torch.Tensor] = []

    actual_steps = []
    for offset in range(horizon):
        row = rows[min(idx + offset, len(rows) - 1)]
        actual_steps.append(torch.tensor(row_to_action(row), dtype=torch.float32).clamp(-1.0, 1.0))
    names.append("actual_trace")
    sequences.append(torch.stack(actual_steps, dim=0))

    first_actual = torch.tensor(row_to_action(rows[idx]), dtype=torch.float32).clamp(-1.0, 1.0)
    names.append("actual_repeat")
    sequences.append(first_actual.repeat(horizon, 1))

    for name, action in build_action_bank(amplitudes):
        names.append(f"constant:{name}")
        sequences.append(action.squeeze(0).repeat(horizon, 1))

    for seq_idx in range(random_sequences):
        vals = [[rng.uniform(-1.0, 1.0) for _ in range(4)] for _ in range(horizon)]
        names.append(f"random:{seq_idx:03d}")
        sequences.append(torch.tensor(vals, dtype=torch.float32))

    return names, torch.stack(sequences, dim=1)


def feature_float(feat: dict[str, torch.Tensor], metric: str, idx: int) -> float:
    return float(metric_value(feat, metric)[idx].item())


def candidate_summary(
    init_feat: dict[str, torch.Tensor],
    rollout_feats: list[dict[str, torch.Tensor]],
    idx: int,
) -> dict[str, float]:
    terminal = rollout_feats[-1]
    support_values = [feature_float(feat, "support_contact", idx) for feat in rollout_feats]
    falling_values = [feature_float(feat, "falling_air", idx) for feat in rollout_feats]
    rotation_values = [feature_float(feat, "rotation_penalty", idx) for feat in rollout_feats]
    return {
        "terminal_support_contact": feature_float(terminal, "support_contact", idx),
        "min_support_contact": min(support_values),
        "terminal_touch_heel": feature_float(terminal, "touch_heel", idx),
        "touch_heel_progress": feature_float(terminal, "touch_heel", idx) - feature_float(init_feat, "touch_heel", idx),
        "terminal_full_foot_support": feature_float(terminal, "full_foot_support", idx),
        "full_foot_progress": feature_float(terminal, "full_foot_support", idx) - feature_float(init_feat, "full_foot_support", idx),
        "max_falling_air": max(falling_values),
        "max_rotation_penalty": max(rotation_values),
        "terminal_pitch_rate_abs": abs(float(terminal["pitch_rate"][idx].item())),
    }


def satisfies_constraints(summary: dict[str, float]) -> bool:
    return (
        summary["min_support_contact"] >= 0.55
        and summary["terminal_support_contact"] >= 0.65
        and summary["touch_heel_progress"] >= 0.05
        and summary["full_foot_progress"] >= 0.03
        and summary["max_falling_air"] <= 0.25
        and summary["max_rotation_penalty"] <= 0.35
    )


def select_states(trace_dirs: list[Path], max_states: int, max_states_per_trace: int, horizon: int) -> list[tuple[Path, int, list[dict[str, str]]]]:
    selected: list[tuple[Path, int, list[dict[str, str]], float]] = []
    for trace_dir in trace_dirs:
        for path in sorted(trace_dir.glob("episode_*.csv")):
            rows = read_csv(path)
            trace_hits: list[tuple[int, float]] = []
            for idx, row in enumerate(rows[:-horizon]):
                if not is_recoverable_toe_support(row):
                    continue
                future = rows[min(idx + horizon, len(rows) - 1)]
                priority = (
                    1.0 * max(0.0, support_contact(row) - support_contact(future))
                    + 0.6 * f(future, "falling_air")
                    + 0.4 * f(future, "rotation_penalty")
                    + 0.2 * f(row, "grounded_prone")
                )
                trace_hits.append((idx, priority))
            trace_hits.sort(key=lambda item: item[1], reverse=True)
            for idx, priority in trace_hits[:max_states_per_trace]:
                selected.append((path, idx, rows, priority))
    selected.sort(key=lambda item: item[3], reverse=True)
    return [(path, idx, rows) for path, idx, rows, _ in selected[:max_states]]


def evaluate_rule(
    model: torch.nn.Module,
    rule: dict[str, Any],
    states: list[tuple[Path, int, list[dict[str, str]]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    horizon = min(int(rule.get("horizon", args.horizon) or args.horizon), args.horizon)
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for path, idx, rows in states:
        names, actions = candidate_sequences(rows, idx, horizon, args.amplitudes, args.random_sequences, rng)
        init_feat, rollout_feats = rollout_features(model, rows[idx], actions)
        final_score, weighted_score, penalty = stage_scores(rule, init_feat, rollout_feats)
        order = torch.argsort(final_score, descending=True).cpu().tolist()
        top_idx = int(order[0])
        actual_idx = 0
        top_summary = candidate_summary(init_feat, rollout_feats, top_idx)
        actual_summary = candidate_summary(init_feat, rollout_feats, actual_idx)
        records.append(
            {
                "trace_csv": str(path),
                "row_index": idx,
                "step": int(f(rows[idx], "step", idx)),
                "top_candidate": names[top_idx],
                "top_score": float(final_score[top_idx].item()),
                "top_weighted_score": float(weighted_score[top_idx].item()),
                "top_constraint_penalty": float(penalty[top_idx].item()),
                "top_summary": top_summary,
                "top_satisfies_constraints": satisfies_constraints(top_summary),
                "actual_score": float(final_score[actual_idx].item()),
                "actual_weighted_score": float(weighted_score[actual_idx].item()),
                "actual_constraint_penalty": float(penalty[actual_idx].item()),
                "actual_summary": actual_summary,
                "actual_satisfies_constraints": satisfies_constraints(actual_summary),
                "actual_rank": 1 + sum(1 for value in final_score if value.item() > final_score[actual_idx].item()),
                "score_margin_top_minus_actual": float((final_score[top_idx] - final_score[actual_idx]).item()),
                "top5": [
                    {
                        "name": names[j],
                        "score": float(final_score[j].item()),
                        "penalty": float(penalty[j].item()),
                        "satisfies_constraints": satisfies_constraints(candidate_summary(init_feat, rollout_feats, j)),
                    }
                    for j in order[:5]
                ],
            }
        )

    def m(key: str, nested: str | None = None) -> float:
        vals = [
            (rec[key] if nested is None else rec[key][nested])
            for rec in records
        ]
        return float(mean(vals)) if vals else 0.0

    top_names = Counter(rec["top_candidate"].split(":", 1)[0] for rec in records)
    return {
        "rule_id": rule.get("id"),
        "horizon": horizon,
        "num_states": len(records),
        "top_constraint_satisfied_rate": m("top_satisfies_constraints"),
        "actual_constraint_satisfied_rate": m("actual_satisfies_constraints"),
        "mean_actual_rank": m("actual_rank"),
        "mean_score_margin_top_minus_actual": m("score_margin_top_minus_actual"),
        "mean_top_penalty": m("top_constraint_penalty"),
        "mean_actual_penalty": m("actual_constraint_penalty"),
        "mean_top_min_support": m("top_summary", "min_support_contact"),
        "mean_actual_min_support": m("actual_summary", "min_support_contact"),
        "mean_top_terminal_support": m("top_summary", "terminal_support_contact"),
        "mean_actual_terminal_support": m("actual_summary", "terminal_support_contact"),
        "mean_top_touch_heel_progress": m("top_summary", "touch_heel_progress"),
        "mean_actual_touch_heel_progress": m("actual_summary", "touch_heel_progress"),
        "mean_top_full_foot_progress": m("top_summary", "full_foot_progress"),
        "mean_actual_full_foot_progress": m("actual_summary", "full_foot_progress"),
        "mean_top_max_falling": m("top_summary", "max_falling_air"),
        "mean_actual_max_falling": m("actual_summary", "max_falling_air"),
        "mean_top_max_rotation": m("top_summary", "max_rotation_penalty"),
        "mean_actual_max_rotation": m("actual_summary", "max_rotation_penalty"),
        "top_candidate_type_counts": dict(top_names),
        "records_sample": records[: min(args.sample_records, len(records))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-library", type=Path, default=DEFAULT_RULE_LIBRARY)
    parser.add_argument("--baseline-rule-library", type=Path, default=DEFAULT_BASELINE_RULE_LIBRARY)
    parser.add_argument("--rule-id", default="toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support")
    parser.add_argument("--baseline-rule-id", default="toe_recovery_recoverable_low_support->stage_effect:stabilize_recoverable_toe_support")
    parser.add_argument("--sd-checkpoint", type=Path, default=DEFAULT_SD_CHECKPOINT)
    parser.add_argument("--trace-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=600)
    parser.add_argument("--max-states-per-trace", type=int, default=2)
    parser.add_argument("--random-sequences", type=int, default=64)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--sample-records", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, loaded_checkpoint = load_model(args.sd_checkpoint)
    states = select_states(args.trace_dir, args.max_states, args.max_states_per_trace, args.horizon)
    rule = find_rule(load_json(args.rule_library), args.rule_id, "toe_recovery_recoverable_low_support")
    payload: dict[str, Any] = {
        "audit_family": "hopper_stage_constraint_candidate_ranking",
        "sd_checkpoint": loaded_checkpoint,
        "rule_library": str(args.rule_library),
        "trace_dirs": [str(path) for path in args.trace_dir],
        "num_selected_states": len(states),
        "candidate_setup": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "random_sequences": args.random_sequences,
            "seed": args.seed,
        },
        "stage_constraint_rule": evaluate_rule(model, rule, states, args),
    }
    if args.baseline_rule_library.exists():
        baseline_rule = find_rule(
            load_json(args.baseline_rule_library),
            args.baseline_rule_id,
            "toe_recovery_recoverable_low_support",
        )
        payload["baseline_stage_rule"] = evaluate_rule(model, baseline_rule, states, args)
        payload["baseline_rule_library"] = str(args.baseline_rule_library)
    write_json(args.output, payload)
    print(f"wrote candidate-ranking audit: {args.output}")
    print(f"states: {len(states)}")
    print(
        "constraint top/actual rates:",
        payload["stage_constraint_rule"]["top_constraint_satisfied_rate"],
        payload["stage_constraint_rule"]["actual_constraint_satisfied_rate"],
    )


if __name__ == "__main__":
    main()
