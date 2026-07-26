#!/usr/bin/env python3
"""Run SD/TVA counterfactual validation for broad Hopper candidates.

This script validates observational broad-context candidates by replacing the
current action with a small bank of probe actions, rolling the SD model forward,
and ranking action families by a joint short-horizon SD return plus TVA terminal
value.  Separate SD terminal and TVA terminal rankings are retained for audit.

The output is still offline evidence.  A candidate becomes suitable for an
initial teacher library only if this audit supports the proposed action family
and the later rule-library builder preserves the audit metadata.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from audit_hopper_ability_entry_counterfactuals import load_sd_model, load_value_net  # noqa: E402


def load_extractor_module():
    module_path = Path(__file__).resolve().with_name("extract_hopper_transition_context_rules.py")
    spec = importlib.util.spec_from_file_location("hopper_transition_context_extractor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extractor module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def clamp_tensor(value: torch.Tensor, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    return torch.clamp(value, min=low, max=high)


def sd_cycle_terminal_score(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cycle-oriented terminal score for short SD rollouts.

    This is intentionally process-light: it rewards non-collapse, height,
    forward speed, and contact availability, while treating rotation as a weak
    penalty unless it accompanies poor terminal structure.  TVA remains the
    main learned terminal signal.
    """

    vx_score = clamp_tensor(feat["vx"] / 1.5)
    height_score = clamp_tensor((feat["height"] - 0.35) / 0.40)
    collision = feat["body_collision"]
    low_height = (feat["height"] < 0.30).to(feat["height"].dtype)
    noncollapse = clamp_tensor(1.0 - 1.8 * collision - 1.2 * low_height)
    support = clamp_tensor(feat["support_contact"] + 0.5 * feat["full_foot_support"])
    rotation = clamp_tensor(feat["rotation_penalty"])
    spin = clamp_tensor(torch.abs(feat["pitch_rate"]) / 10.0)
    return (
        0.35 * vx_score
        + 0.25 * noncollapse
        + 0.20 * height_score
        + 0.10 * support
        - 0.06 * rotation
        - 0.04 * spin
    )


def action_family_from_probe(name: str) -> str:
    if name in {"actual", "zero"}:
        return name
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def family_best(names: list[str], values: list[float]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in zip(names, values):
        family = action_family_from_probe(name)
        current = grouped.get(family)
        if current is None or value > current["score"]:
            grouped[family] = {"family": family, "name": name, "score": float(value)}
    return grouped


def rank_families(grouped: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(grouped.values(), key=lambda item: item["score"], reverse=True)


def score_target(ranked: list[dict[str, Any]], target: str) -> dict[str, Any]:
    best_score = ranked[0]["score"] if ranked else 0.0
    target_row = next((item for item in ranked if item["family"] == target), None)
    target_score = float(target_row["score"]) if target_row else None
    target_rank = 1 + sum(1 for item in ranked if target_score is not None and item["score"] > target_score)
    return {
        "target_score": target_score,
        "best_family": ranked[0]["family"] if ranked else "none",
        "best_score": best_score,
        "target_rank": target_rank if target_score is not None else None,
        "target_best": bool(ranked and ranked[0]["family"] == target),
        "target_top3": bool(target_score is not None and target_rank <= 3),
        "target_margin_to_best": (target_score - best_score) if target_score is not None else None,
    }


def diagnostic_key(record: dict[str, Any]) -> str:
    parts = record.get("parts", {})
    return "|".join(
        [
            f"omega={parts.get('omega_bin', 'unknown')}",
            f"support={parts.get('support_bin', 'unknown')}",
            f"risk={parts.get('risk_bin', 'unknown')}",
        ]
    )


def select_records(records: list[dict[str, Any]], max_records: int, seed: int) -> list[dict[str, Any]]:
    if max_records <= 0 or len(records) <= max_records:
        return records
    rng = random.Random(seed)
    by_diag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_diag[diagnostic_key(record)].append(record)
    selected: list[dict[str, Any]] = []
    keys = sorted(by_diag, key=lambda key: len(by_diag[key]), reverse=True)
    per_group = max(1, max_records // max(1, len(keys)))
    leftovers: list[dict[str, Any]] = []
    for key in keys:
        vals = list(by_diag[key])
        rng.shuffle(vals)
        selected.extend(vals[:per_group])
        leftovers.extend(vals[per_group:])
    if len(selected) < max_records:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_records - len(selected)])
    return selected[:max_records]


def evaluate_record(
    model: torch.nn.Module,
    value_net: torch.nn.Module,
    row: dict[str, str],
    action_bank: list[tuple[str, torch.Tensor]],
    horizon: int,
    gamma: float,
    device: str,
) -> dict[str, Any]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32, device=device).unsqueeze(0))
    names = [name for name, _action in action_bank]
    actions = torch.cat([action.to(device) for _name, action in action_bank], dim=0)
    cur = state.repeat(actions.shape[0], 1)
    with torch.no_grad():
        rollout_return = torch.zeros(actions.shape[0], dtype=cur.dtype, device=cur.device)
        discount = 1.0
        for _ in range(horizon):
            nxt = model(cur, actions)
            rollout_return = rollout_return + discount * model.predict_plan_reward(cur, actions, nxt)
            cur = nxt
            discount *= float(gamma)
        feat = model._state_reward_features(cur)
        sd_scores = [float(x) for x in sd_cycle_terminal_score(feat).detach().cpu().view(-1).tolist()]
        tva_scores = [float(x) for x in value_net(cur).detach().cpu().view(-1).tolist()]
        joint_scores_t = rollout_return + discount * value_net(cur)
        joint_scores = [float(x) for x in joint_scores_t.detach().cpu().view(-1).tolist()]
    sd_ranked = rank_families(family_best(names, sd_scores))
    tva_ranked = rank_families(family_best(names, tva_scores))
    joint_ranked = rank_families(family_best(names, joint_scores))
    return {
        "joint_top_families": joint_ranked[:6],
        "sd_top_families": sd_ranked[:6],
        "tva_top_families": tva_ranked[:6],
    }


def aggregate_rule(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "joint_target_best_rate": mean_flag(records, "joint_target_best"),
        "joint_target_top3_rate": mean_flag(records, "joint_target_top3"),
        "sd_target_best_rate": mean_flag(records, "sd_target_best"),
        "sd_target_top3_rate": mean_flag(records, "sd_target_top3"),
        "tva_target_best_rate": mean_flag(records, "tva_target_best"),
        "tva_target_top3_rate": mean_flag(records, "tva_target_top3"),
        "all_top3_rate": mean(
            [1.0 if r["joint_target_top3"] and r["sd_target_top3"] and r["tva_target_top3"] else 0.0 for r in records]
        ),
        "mean_joint_margin_to_best": mean_optional(records, "joint_target_margin_to_best"),
        "mean_sd_margin_to_best": mean_optional(records, "sd_target_margin_to_best"),
        "mean_tva_margin_to_best": mean_optional(records, "tva_target_margin_to_best"),
        "joint_best_family_counts": dict(Counter(r["joint_best_family"] for r in records).most_common()),
        "sd_best_family_counts": dict(Counter(r["sd_best_family"] for r in records).most_common()),
        "tva_best_family_counts": dict(Counter(r["tva_best_family"] for r in records).most_common()),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mean_flag(records: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if record.get(key) else 0.0 for record in records])


def mean_optional(records: list[dict[str, Any]], key: str) -> float:
    vals = [float(record[key]) for record in records if record.get(key) is not None]
    return mean(vals)


def validation_status(agg: dict[str, Any]) -> str:
    if agg["count"] <= 0:
        return "no_counterfactual_records"
    if agg["joint_target_top3_rate"] >= 0.50:
        return "joint_top3_supported"
    if agg["joint_target_top3_rate"] >= 0.30:
        return "partially_supported"
    return "not_supported"


def extractor_args(base_args: argparse.Namespace, trace_globs: list[str], parameters: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        run_prefix="",
        logs_root=Path("logs/hopper-hop"),
        cycles="",
        seeds="",
        trace_glob=trace_globs,
        output=Path("/tmp/unused.json"),
        past_window=int(parameters.get("past_window", base_args.past_window)),
        future_window=int(parameters.get("future_window", base_args.future_window)),
        outcome_mode=str(parameters.get("outcome_mode", base_args.outcome_mode)),
        phase_schema=str(parameters.get("phase_schema", base_args.phase_schema)),
        context_mode=str(parameters.get("context_mode", base_args.context_mode)),
        action_threshold=float(parameters.get("action_threshold", base_args.action_threshold)),
        focus_grounded_hop=False,
        min_context_support=0,
        min_family_support=0,
        min_score_advantage=0.0,
        max_rules=0,
        max_contexts_report=0,
        max_families_per_context_report=8,
    )


def allowed_rules(payload: dict[str, Any], allowed_decisions: set[str]) -> list[dict[str, Any]]:
    out = []
    for row in payload.get("rules", []):
        if str(row.get("decision")) in allowed_decisions:
            source = dict(row.get("source_rule", {}))
            source["_validation_decision"] = row.get("decision")
            source["_observational_validation"] = row
            out.append(source)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    extractor = load_extractor_module()
    observational = load_json(args.observational_validation)
    rules = allowed_rules(observational, set(args.allowed_decision))
    parameters = dict(observational.get("extractor_parameters", {}))
    records = extractor.extract_records(extractor_args(args, args.trace_glob, parameters))
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_key[str(record["context_key"])].append(record)

    model = load_sd_model(args.sd_checkpoint, args.device)
    value_net = load_value_net(args.tva_checkpoint, args.device)
    action_bank = build_action_bank(args.amplitudes)

    trace_cache: dict[str, list[dict[str, str]]] = {}
    reports: list[dict[str, Any]] = []
    for rule in rules:
        key = str(rule.get("rule_key"))
        target = str(rule.get("action_family"))
        selected = select_records(by_key.get(key, []), args.max_records_per_rule, args.seed)
        record_reports: list[dict[str, Any]] = []
        for record in selected:
            trace_csv = str(record["trace_csv"])
            rows = trace_cache.get(trace_csv)
            if rows is None:
                rows = read_csv(Path(trace_csv))
                trace_cache[trace_csv] = rows
            idx = int(record["index"])
            if idx < 0 or idx >= len(rows):
                continue
            result = evaluate_record(model, value_net, rows[idx], action_bank, args.horizon, args.gamma, args.device)
            joint_score = score_target(result["joint_top_families"], target)
            sd_score = score_target(result["sd_top_families"], target)
            tva_score = score_target(result["tva_top_families"], target)
            record_reports.append(
                {
                    "trace_csv": trace_csv,
                    "episode": record.get("episode"),
                    "step": record.get("step"),
                    "index": idx,
                    "diagnostic_key": diagnostic_key(record),
                    "future_outcome": record.get("future_outcome"),
                    "transition_score": record.get("transition_score"),
                    "target_family": target,
                    "joint_best_family": joint_score["best_family"],
                    "joint_target_best": joint_score["target_best"],
                    "joint_target_top3": joint_score["target_top3"],
                    "joint_target_margin_to_best": joint_score["target_margin_to_best"],
                    "sd_best_family": sd_score["best_family"],
                    "sd_target_best": sd_score["target_best"],
                    "sd_target_top3": sd_score["target_top3"],
                    "sd_target_margin_to_best": sd_score["target_margin_to_best"],
                    "tva_best_family": tva_score["best_family"],
                    "tva_target_best": tva_score["target_best"],
                    "tva_target_top3": tva_score["target_top3"],
                    "tva_target_margin_to_best": tva_score["target_margin_to_best"],
                    "joint_top_families": result["joint_top_families"],
                    "sd_top_families": result["sd_top_families"],
                    "tva_top_families": result["tva_top_families"],
                }
            )
        agg = aggregate_rule(record_reports)
        reports.append(
            {
                "rule_key": key,
                "action_family": target,
                "observational_decision": rule.get("_validation_decision"),
                "num_context_records": len(by_key.get(key, [])),
                "num_counterfactual_records": len(record_reports),
                "status": validation_status(agg),
                "aggregate": agg,
                "source_rule": rule,
                "records": record_reports[: args.max_records_report],
            }
        )

    return {
        "analysis_family": "hopper_broad_context_sd_tva_counterfactual_v1",
        "observational_validation": str(args.observational_validation),
        "sd_checkpoint": str(args.sd_checkpoint),
        "tva_checkpoint": str(args.tva_checkpoint),
        "trace_glob": args.trace_glob,
        "parameters": {
            "horizon": args.horizon,
            "gamma": args.gamma,
            "amplitudes": args.amplitudes,
            "max_records_per_rule": args.max_records_per_rule,
            "seed": args.seed,
            "device": args.device,
            "allowed_decision": args.allowed_decision,
        },
        "num_rules": len(reports),
        "status_counts": dict(Counter(row["status"] for row in reports)),
        "rules": reports,
        "status_meaning": {
            "joint_top3_supported": "target action family is frequently top-3 under SD rollout return plus TVA terminal value",
            "partially_supported": "target action family sometimes appears near the top under the joint score; needs caution",
            "not_supported": "target action family is rarely near the top under the joint counterfactual score",
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Broad Context SD/TVA Counterfactual Validation",
        "",
        f"Observational validation: `{payload['observational_validation']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['tva_checkpoint']}`",
        f"Status counts: `{payload['status_counts']}`",
        "",
        "| status | rule | action | records | joint top3 | SD top3 | TVA top3 | joint margin | SD margin | TVA margin |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rules"]:
        agg = row["aggregate"]
        lines.append(
            f"| `{row['status']}` | `{row['rule_key']}` | `{row['action_family']}` | "
            f"{agg['count']} | {agg['joint_target_top3_rate']:.3f} | "
            f"{agg['sd_target_top3_rate']:.3f} | {agg['tva_target_top3_rate']:.3f} | "
            f"{agg['mean_joint_margin_to_best']:.4f} | "
            f"{agg['mean_sd_margin_to_best']:.4f} | {agg['mean_tva_margin_to_best']:.4f} |"
        )
    lines.extend(["", "## Best Family Counts", ""])
    for row in payload["rules"]:
        agg = row["aggregate"]
        lines.extend(
            [
                f"### `{row['rule_key']} -> {row['action_family']}`",
                "",
                f"- Status: `{row['status']}`",
                f"- Joint best families: `{agg['joint_best_family_counts']}`",
                f"- SD best families: `{agg['sd_best_family_counts']}`",
                f"- TVA best families: `{agg['tva_best_family_counts']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observational-validation", type=Path, required=True)
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--tva-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-decision", action="append", default=["counterfactual_ready"])
    parser.add_argument("--past-window", type=int, default=8)
    parser.add_argument("--future-window", type=int, default=20)
    parser.add_argument("--outcome-mode", default="cycle_v1")
    parser.add_argument("--phase-schema", default="simple_v1")
    parser.add_argument("--context-mode", default="broad_context_v1")
    parser.add_argument("--action-threshold", type=float, default=0.20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-records-per-rule", type=int, default=96)
    parser.add_argument("--max-records-report", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote broad context SD/TVA counterfactual validation: {args.output}")
    print(f"rules={payload['num_rules']} statuses={payload['status_counts']}")


if __name__ == "__main__":
    main()
