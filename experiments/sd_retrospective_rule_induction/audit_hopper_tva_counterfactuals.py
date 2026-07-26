#!/usr/bin/env python3
"""Audit SD counterfactual actions with a trained Hopper TVA value net.

The output is intentionally side-effect free: it scores the same probe action
set used by ``counterfactual_actions.py`` with TVA terminal values, but does
not update the rule library or planner prior.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, mean_or_zero, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


def action_family_from_name(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(name)


def family_from_action(action: list[float], threshold: float = 0.20) -> str:
    names = ("waist", "hip", "knee", "ankle")
    idx = max(range(len(action)), key=lambda i: abs(action[i]))
    if abs(action[idx]) < threshold:
        return "weak"
    return f"{names[idx]}_{'pos' if action[idx] > 0 else 'neg'}"


def load_hopper_tva_module():
    module_path = REPO_ROOT / "scripts" / "run_dmcontrol_hopper.py"
    spec = importlib.util.spec_from_file_location("sdmcts_hopper_tva_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load TVA source module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sd_model(checkpoint: Path, device: str) -> DMControlHopperSDModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_kwargs: dict[str, Any] = {
        "action_prior_enabled": False,
        "shaping_scale": 0.0,
        "dense_pose_delta_weight": 0.0,
        "dense_pose_abs_weight": 0.0,
        "dense_pose_milestone_scale": 0.0,
        "recover_penalty_scale": 0.0,
        "stage_mode": "full",
    }
    model_kwargs.update(payload.get("model_config", {}))
    model = DMControlHopperSDModel(**model_kwargs).to(device).eval()
    src_state = payload.get("model_state", payload)
    target_state = model.state_dict()
    compatible = {
        k: v for k, v in src_state.items()
        if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return model


def load_value_net(checkpoint: Path, device: str) -> torch.nn.Module:
    tva_source = load_hopper_tva_module()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value_net = tva_source.HopperValueNet(
        state_dim=int(payload.get("state_dim", 15)),
        hidden=int(payload.get("hidden", 128)),
    )
    value_net.load_state_dict(payload.get("value_net_state", payload), strict=False)
    return value_net.to(device).eval()


def action_scores(
    model: DMControlHopperSDModel,
    value_net: torch.nn.Module,
    row: dict[str, str],
    action_bank: list[tuple[str, torch.Tensor]],
    horizon: int,
    device: str,
) -> dict[str, Any]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32, device=device).unsqueeze(0))
    actual_action = torch.tensor(row_to_action(row), dtype=torch.float32, device=device).unsqueeze(0).clamp(-1.0, 1.0)
    probe_names = ["actual"] + [name for name, _ in action_bank]
    probe_actions = torch.cat(
        [actual_action, *[action.to(device) for _, action in action_bank]],
        dim=0,
    )
    cur = state.repeat(probe_actions.shape[0], 1)
    with torch.no_grad():
        for _ in range(horizon):
            cur = model(cur, probe_actions)
        values = value_net(cur).detach().cpu().tolist()
    actual_value = float(values[0])
    alternatives = [
        {
            "name": name,
            "family": action_family_from_name(name),
            "tva_value": float(value),
        }
        for name, value in zip(probe_names[1:], values[1:])
    ]
    alternatives.sort(key=lambda item: item["tva_value"], reverse=True)
    best = alternatives[0]
    return {
        "step": int(float(row.get("step", 0) or 0)),
        "actual_action": row_to_action(row),
        "actual_action_family": family_from_action(row_to_action(row)),
        "actual_tva_value": actual_value,
        "best_action": best["name"],
        "best_action_family": best["family"],
        "best_tva_value": best["tva_value"],
        "tva_advantage_over_actual": best["tva_value"] - actual_value,
        "actual_rank_among_probe_actions": 1 + sum(1 for item in alternatives if item["tva_value"] > actual_value),
        "num_probe_actions": len(alternatives),
        "probe_actions": alternatives,
        "top_actions": alternatives[:5],
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups["all"].append(rec)
        groups[f"label:{rec['label']}"].append(rec)
        groups[f"event:{rec['event']}"].append(rec)
    out: dict[str, Any] = {}
    for key, vals in groups.items():
        out[key] = {
            "count": len(vals),
            "mean_tva_advantage_over_actual": mean_or_zero(v["tva_advantage_over_actual"] for v in vals),
            "mean_actual_tva_value": mean_or_zero(v["actual_tva_value"] for v in vals),
            "mean_best_tva_value": mean_or_zero(v["best_tva_value"] for v in vals),
            "mean_actual_rank": mean_or_zero(v["actual_rank_among_probe_actions"] for v in vals),
            "actual_top3_rate": mean_or_zero(
                1.0 if v["actual_rank_among_probe_actions"] <= 3 else 0.0 for v in vals
            ),
        }
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper TVA Counterfactual Audit",
        "",
        f"Counterfactual source: `{payload['source_counterfactuals']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['tva_checkpoint']}`",
        f"State records: `{payload['num_state_records']}`",
        f"Horizon: `{payload['horizon']}`",
        "",
        "## Aggregate",
        "",
        "| Group | Count | Mean TVA advantage | Mean actual value | Mean best value | Mean actual rank | Actual top-3 rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in sorted(payload["aggregate"].items()):
        lines.append(
            "| "
            f"`{key}` | "
            f"{stats['count']} | "
            f"{stats['mean_tva_advantage_over_actual']:.4f} | "
            f"{stats['mean_actual_tva_value']:.4f} | "
            f"{stats['mean_best_tva_value']:.4f} | "
            f"{stats['mean_actual_rank']:.2f} | "
            f"{stats['actual_top3_rate']:.3f} |"
        )
    lines.extend(["", "## Top TVA Gaps", ""])
    for rec in sorted(payload["records"], key=lambda x: x["tva_advantage_over_actual"], reverse=True)[:20]:
        lines.extend(
            [
                f"### {rec['episode_id']} step {rec['step']}",
                "",
                f"- Label: `{rec['label']}`",
                f"- Event: `{rec['event']}`",
                f"- Actual TVA value: `{rec['actual_tva_value']:.4f}`",
                f"- Best action: `{rec['best_action']}`",
                f"- Best TVA value: `{rec['best_tva_value']:.4f}`",
                f"- TVA advantage: `{rec['tva_advantage_over_actual']:.4f}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--sd-checkpoint", type=Path, required=True)
    parser.add_argument("--tva-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["first", "stratified", "label_event"], default="stratified")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sample_group_key(record: dict[str, Any], mode: str) -> str:
    if mode == "label_event":
        return f"{record.get('label', 'unknown')}|{record.get('event', 'unknown')}"
    return str(record.get("label", "unknown"))


def select_records(records: list[dict[str, Any]], max_records: int, mode: str, seed: int) -> list[dict[str, Any]]:
    if max_records <= 0 or len(records) <= max_records:
        return records
    if mode == "first":
        return records[:max_records]
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(sample_group_key(rec, mode), []).append(rec)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    labels = sorted(groups)
    per_group = max(1, max_records // max(1, len(labels)))
    leftovers: list[dict[str, Any]] = []
    for label in labels:
        vals = list(groups[label])
        rng.shuffle(vals)
        selected.extend(vals[:per_group])
        leftovers.extend(vals[per_group:])
    if len(selected) < max_records:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:max_records - len(selected)])
    return selected[:max_records]


def selected_group_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.get('label', 'unknown')}|{record.get('event', 'unknown')}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    counterfactuals = load_json(args.counterfactuals)
    model = load_sd_model(args.sd_checkpoint, args.device)
    value_net = load_value_net(args.tva_checkpoint, args.device)
    action_bank = build_action_bank(args.amplitudes)

    records: list[dict[str, Any]] = []
    trace_cache: dict[str, list[dict[str, str]]] = {}
    source_records = select_records(
        counterfactuals.get("records", []),
        args.max_records,
        args.sample_mode,
        args.seed,
    )
    for rec in source_records:
        trace_csv = str(rec["trace_csv"])
        rows = trace_cache.get(trace_csv)
        if rows is None:
            rows = read_csv(Path(trace_csv))
            trace_cache[trace_csv] = rows
        step = int(rec["step"])
        if step < 0 or step >= len(rows):
            continue
        scored = action_scores(model, value_net, rows[step], action_bank, args.horizon, args.device)
        scored.update(
            {
                "episode_id": rec["episode_id"],
                "seed": rec["seed"],
                "cycle_step": rec["cycle_step"],
                "label": rec["label"],
                "event": rec["event"],
                "episode_reward": rec["episode_reward"],
                "trace_csv": rec["trace_csv"],
            }
        )
        records.append(scored)

    payload = {
        "audit_family": "hopper_tva_counterfactuals_v1",
        "source_counterfactuals": str(args.counterfactuals),
        "sd_checkpoint": str(args.sd_checkpoint),
        "tva_checkpoint": str(args.tva_checkpoint),
        "horizon": int(args.horizon),
        "amplitudes": args.amplitudes,
        "max_records": int(args.max_records),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "source_num_state_records": len(counterfactuals.get("records", [])),
        "selected_group_counts": selected_group_counts(source_records),
        "num_trace_csv_read": len(trace_cache),
        "num_state_records": len(records),
        "aggregate": aggregate(records),
        "records": records,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA counterfactual audit: {args.output}")
    print(f"state records: {len(records)}")


if __name__ == "__main__":
    main()
