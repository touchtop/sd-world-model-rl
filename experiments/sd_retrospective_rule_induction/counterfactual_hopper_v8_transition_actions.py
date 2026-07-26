#!/usr/bin/env python3
"""Run SD counterfactuals for Hopper v8 transition-action evidence."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import f, load_json, mean_or_zero, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


DEFAULT_EVIDENCE = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.json")
DEFAULT_MODEL_CHECKPOINT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v72_multiseed_shared_200k_seed123_resume100k_20260507_r2/"
    "cycle_040/sd_explainers/sd_explainer_cycle_040.pt"
)
DEFAULT_OUTPUT = Path("results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.json")


def action_family(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def continue_score(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    uprightness = torch.clamp(1.0 - torch.abs(feat["pitch"]) / 0.45, min=0.0, max=1.0)
    height_quality = torch.clamp((feat["torso_z"] - 0.45) / 0.45, min=0.0, max=1.0)
    forward_speed = torch.clamp(feat["vx"] / 2.0, min=0.0, max=1.0)
    spin_control = torch.clamp(1.0 - torch.abs(feat["pitch_rate"]) / 3.50, min=0.0, max=1.0)
    support_readiness = torch.maximum(feat["support_contact"], feat["full_foot_support"])
    non_collision = 1.0 - torch.clamp(feat["body_collision"], min=0.0, max=1.0)
    return (
        0.25 * uprightness
        + 0.20 * height_quality
        + 0.20 * forward_speed
        + 0.15 * spin_control
        + 0.10 * support_readiness
        + 0.10 * non_collision
    )


def transition_score(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    positive_vx = torch.clamp(feat["vx"] / 2.0, min=0.0, max=1.0)
    low_rotation = 1.0 - torch.clamp(feat["rotation_penalty"], min=0.0, max=1.0)
    return (
        0.25 * feat["landing_pose_score"]
        + 0.20 * feat["ground_pose_score"]
        + 0.20 * feat["rise_pose_score"]
        + 0.15 * feat["support_contact"]
        + 0.10 * positive_vx
        + 0.10 * low_rotation
    )


def objective_score(feat: dict[str, torch.Tensor], objective: str) -> torch.Tensor:
    cont = continue_score(feat)
    trans = transition_score(feat)
    if objective == "continue":
        return cont
    if objective == "transition":
        return trans
    if objective == "joint":
        return 0.50 * cont + 0.50 * trans
    raise ValueError(f"unknown objective: {objective}")


def feature_summary(feat: dict[str, torch.Tensor], idx: int) -> dict[str, float]:
    keys = [
        "torso_z",
        "vx",
        "vz",
        "pitch",
        "pitch_rate",
        "support_contact",
        "full_foot_support",
        "body_collision",
        "landing_pose_score",
        "ground_pose_score",
        "rise_pose_score",
        "rotation_penalty",
    ]
    return {key: float(feat[key][idx].item()) for key in keys}


def load_model(model_checkpoint: Path | None) -> tuple[DMControlHopperSDModel, str | None]:
    model_kwargs: dict[str, Any] = {
        "action_prior_enabled": False,
        "shaping_scale": 0.0,
        "dense_pose_delta_weight": 0.0,
        "dense_pose_abs_weight": 0.0,
        "dense_pose_milestone_scale": 0.0,
        "recover_penalty_scale": 0.0,
        "stage_mode": "full",
    }
    payload = None
    if model_checkpoint is not None and model_checkpoint.exists():
        payload = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
        model_kwargs.update(payload.get("model_config", {}))
    model = DMControlHopperSDModel(**model_kwargs).eval()
    if payload is not None:
        src_state = payload.get("model_state", payload)
        target_state = model.state_dict()
        compatible = {
            k: v for k, v in src_state.items()
            if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
        }
        model.load_state_dict(compatible, strict=False)
        return model, str(model_checkpoint)
    return model, None


def rollout_probe_scores(
    model: DMControlHopperSDModel,
    row: dict[str, str],
    probe_actions: torch.Tensor,
    horizon: int,
    objective: str,
) -> tuple[list[float], list[dict[str, float]]]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32).unsqueeze(0))
    cur = state.repeat(probe_actions.shape[0], 1)
    with torch.no_grad():
        for _ in range(horizon):
            cur = model(cur, probe_actions)
        feat = model._state_reward_features(cur)
        scores = objective_score(feat, objective)
    return [float(x) for x in scores.cpu().tolist()], [feature_summary(feat, i) for i in range(probe_actions.shape[0])]


def select_records(evidence: dict[str, Any], max_records_per_key: int, anchors: set[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in evidence.get("records", []):
        if rec.get("anchor_type") not in anchors:
            continue
        groups[rec["rule_key"]].append(rec)
    selected: list[dict[str, Any]] = []
    for vals in groups.values():
        vals = sorted(
            vals,
            key=lambda rec: (
                bool(rec.get("next_degradation_hit")),
                -float(rec.get("continue_delta_next", 0.0)),
                -float(rec.get("transition_delta_next", 0.0)),
            ),
            reverse=True,
        )
        selected.extend(vals[:max_records_per_key] if max_records_per_key > 0 else vals)
    return sorted(selected, key=lambda rec: (str(rec["trace_csv"]), int(rec["row_index"])))


def evaluate_records(
    records: list[dict[str, Any]],
    model: DMControlHopperSDModel,
    objective: str,
    horizon: int,
    amplitudes: list[float],
) -> list[dict[str, Any]]:
    action_bank = build_action_bank(amplitudes)
    trace_cache: dict[str, list[dict[str, str]]] = {}
    out: list[dict[str, Any]] = []
    for rec in records:
        trace_csv = str(rec["trace_csv"])
        rows = trace_cache.get(trace_csv)
        if rows is None:
            rows = read_csv(Path(trace_csv))
            trace_cache[trace_csv] = rows
        row = rows[int(rec["row_index"])]
        actual_action = torch.tensor(row_to_action(row), dtype=torch.float32).unsqueeze(0).clamp(-1.0, 1.0)
        names = ["actual"] + [name for name, _ in action_bank]
        actions = torch.cat([actual_action, *[action for _, action in action_bank]], dim=0)
        scores, summaries = rollout_probe_scores(model, row, actions, horizon, objective)
        alternatives = [
            {"name": name, "family": action_family(name), "score": score, "summary": summary}
            for name, score, summary in zip(names[1:], scores[1:], summaries[1:])
        ]
        alternatives.sort(key=lambda item: item["score"], reverse=True)
        best = alternatives[0]
        actual_score = scores[0]
        out.append(
            {
                "trace_csv": trace_csv,
                "seed": rec.get("seed"),
                "episode": rec.get("episode"),
                "step": rec.get("step"),
                "row_index": rec.get("row_index"),
                "anchor_type": rec.get("anchor_type"),
                "rule_key": rec["rule_key"],
                "actual_primary_action_family": rec.get("primary_action_family"),
                "objective": objective,
                "actual_score": actual_score,
                "best_action": best["name"],
                "best_action_family": best["family"],
                "best_score": best["score"],
                "advantage_over_actual": best["score"] - actual_score,
                "actual_rank_among_probe_actions": 1 + sum(1 for item in alternatives if item["score"] > actual_score),
                "top_actions": alternatives[:5],
                "observed_continue_delta_next": rec.get("continue_delta_next"),
                "observed_transition_delta_next": rec.get("transition_delta_next"),
                "observed_next_degradation_hit": rec.get("next_degradation_hit"),
            }
        )
    return out


def aggregate(records: list[dict[str, Any]], min_support: int, min_advantage: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["rule_key"]].append(rec)

    by_key: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        family_counts = Counter(rec["best_action_family"] for rec in vals)
        by_key[key] = {
            "count": len(vals),
            "seeds": sorted({rec["seed"] for rec in vals if rec.get("seed") is not None}),
            "mean_advantage_over_actual": mean_or_zero(rec["advantage_over_actual"] for rec in vals),
            "mean_actual_rank": mean_or_zero(rec["actual_rank_among_probe_actions"] for rec in vals),
            "best_action_family_counts": dict(family_counts.most_common()),
        }
        for family, support in family_counts.items():
            matched = [rec for rec in vals if rec["best_action_family"] == family]
            mean_adv = mean_or_zero(rec["advantage_over_actual"] for rec in matched)
            if support >= min_support and mean_adv >= min_advantage:
                candidates.append(
                    {
                        "id": f"{key}->soft_prefer:{family}",
                        "rule_key": key,
                        "rule_role": key.split("|", 1)[0],
                        "action_family": family,
                        "intervention_type": "soft_prefer_action_family",
                        "support": support,
                        "seeds_covered": len({rec["seed"] for rec in matched if rec.get("seed") is not None}),
                        "mean_advantage_over_actual": mean_adv,
                        "status": "offline_sd_counterfactual_candidate",
                        "evidence_kind": "sd_v8_objective_counterfactual",
                    }
                )
    candidates.sort(key=lambda item: (item["seeds_covered"], item["support"], item["mean_advantage_over_actual"]), reverse=True)
    return {"by_rule_key": by_key, "candidates": candidates}


def write_markdown(path: Path, payload: dict[str, Any], max_candidates: int) -> None:
    lines = [
        "# Hopper v8 Transition Counterfactuals",
        "",
        f"Objective: `{payload['objective']}`",
        f"Horizon: `{payload['horizon']}`",
        f"Records: `{payload['num_records']}`",
        f"Candidates: `{payload['num_candidates']}`",
        f"Model checkpoint: `{payload['model_checkpoint']}`",
        "",
        "## Candidate Rules",
        "",
    ]
    for item in payload["candidates"][:max_candidates]:
        lines.extend(
            [
                f"### `{item['id']}`",
                "",
                f"- Support / seeds: `{item['support']}` / `{item['seeds_covered']}`",
                f"- Mean advantage over actual: `{item['mean_advantage_over_actual']:.4f}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--objective", choices=["continue", "transition", "joint"], default="joint")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--anchor-type", action="append", default=["last_success", "pre_degradation"])
    parser.add_argument("--max-records-per-key", type=int, default=40)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-advantage", type=float, default=0.02)
    parser.add_argument("--max-markdown-candidates", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = load_json(args.evidence)
    model, loaded_checkpoint = load_model(args.model_checkpoint)
    selected = select_records(evidence, args.max_records_per_key, set(args.anchor_type))
    records = evaluate_records(selected, model, args.objective, args.horizon, args.amplitudes)
    agg = aggregate(records, args.min_support, args.min_advantage)
    payload = {
        "counterfactual_family": "hopper-v8-transition-objective-counterfactuals",
        "source_evidence": str(args.evidence),
        "model_checkpoint": loaded_checkpoint,
        "objective": args.objective,
        "horizon": args.horizon,
        "amplitudes": args.amplitudes,
        "anchor_types": args.anchor_type,
        "num_records": len(records),
        "num_rule_key_groups": len(agg["by_rule_key"]),
        "num_candidates": len(agg["candidates"]),
        "by_rule_key": agg["by_rule_key"],
        "candidates": agg["candidates"],
        "records": records,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_candidates)
    print(f"wrote v8 counterfactuals: {args.output}")
    print(f"records: {payload['num_records']} candidates: {payload['num_candidates']}")


if __name__ == "__main__":
    main()
