#!/usr/bin/env python3
"""Audit ability-entry window actions with SD and TVA counterfactual rollouts.

This script consumes the offline ability-entry segments and asks whether the
dominant action family observed in an entry window is also preferred by a
same-cycle SD rollout objective and by the same-cycle TVA terminal value.  It is
read-only: it does not change the online rule library or training checkpoints.
"""

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

from common import mean_or_zero, read_csv, row_to_action, row_to_raw_obs, write_json, load_json  # noqa: E402
from counterfactual_actions import build_action_bank, desirability_batch  # noqa: E402
from mine_hopper_v8_transition_actions import primary_action_family  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


DEFAULT_SEGMENTS = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit.json"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1"
)


class HopperValueNet(torch.nn.Module):
    """Checkpoint-compatible TVA value network on structured Hopper state."""

    def __init__(self, state_dim: int = 17, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value_net = HopperValueNet(
        state_dim=int(payload.get("state_dim", 17)),
        hidden=int(payload.get("hidden", 128)),
    )
    value_net.load_state_dict(payload.get("value_net_state", payload), strict=False)
    return value_net.to(device).eval()


def family_from_probe_name(name: str) -> str:
    if name == "actual":
        return "actual"
    if name == "zero":
        return "zero"
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def checkpoint_paths(results_root: Path, cycle: int) -> tuple[Path, Path]:
    cycle_dir = results_root / f"cycle_{cycle:03d}"
    return (
        cycle_dir / "sd_explainers" / f"sd_explainer_cycle_{cycle:03d}.pt",
        cycle_dir / "tva" / f"hopper_tva_cycle_{cycle:03d}.pt",
    )


def family_best(
    names: list[str],
    values: list[float],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in zip(names, values):
        family = family_from_probe_name(name)
        current = grouped.get(family)
        if current is None or value > current["score"]:
            grouped[family] = {"family": family, "name": name, "score": value}
    return grouped


def top_families(grouped: dict[str, dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(grouped.values(), key=lambda item: item["score"], reverse=True)[:k]


def evaluate_anchor(
    model: DMControlHopperSDModel,
    value_net: torch.nn.Module,
    row: dict[str, str],
    action_bank: list[tuple[str, torch.Tensor]],
    horizon: int,
    device: str,
) -> dict[str, Any]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32, device=device).unsqueeze(0))
    actual_action = torch.tensor(row_to_action(row), dtype=torch.float32, device=device).unsqueeze(0).clamp(-1.0, 1.0)
    action_names = ["actual"] + [name for name, _ in action_bank]
    actions = torch.cat([actual_action, *[action.to(device) for _, action in action_bank]], dim=0)
    cur = state.repeat(actions.shape[0], 1)
    with torch.no_grad():
        for _ in range(horizon):
            cur = model(cur, actions)
        feat = model._state_reward_features(cur)
        sd_scores = [float(x) for x in desirability_batch(feat).detach().cpu().tolist()]
        tva_scores = [float(x) for x in value_net(cur).detach().cpu().view(-1).tolist()]
    sd_by_family = family_best(action_names, sd_scores)
    tva_by_family = family_best(action_names, tva_scores)
    actual_family = primary_action_family(row, threshold=0.20)
    return {
        "actual_anchor_action": row_to_action(row),
        "actual_anchor_family": actual_family,
        "sd_by_family": sd_by_family,
        "tva_by_family": tva_by_family,
        "sd_top_families": top_families(sd_by_family, 5),
        "tva_top_families": top_families(tva_by_family, 5),
    }


def score_target_family(result: dict[str, Any], target_family: str) -> dict[str, Any]:
    sd_group = result["sd_by_family"]
    tva_group = result["tva_by_family"]
    sd_top = result["sd_top_families"]
    tva_top = result["tva_top_families"]
    sd_target = sd_group.get(target_family)
    tva_target = tva_group.get(target_family)
    sd_best_score = sd_top[0]["score"] if sd_top else 0.0
    tva_best_score = tva_top[0]["score"] if tva_top else 0.0
    sd_target_score = float(sd_target["score"]) if sd_target else None
    tva_target_score = float(tva_target["score"]) if tva_target else None
    return {
        "target_family": target_family,
        "sd_target_available": sd_target is not None,
        "tva_target_available": tva_target is not None,
        "sd_target_score": sd_target_score,
        "tva_target_score": tva_target_score,
        "sd_best_family": sd_top[0]["family"] if sd_top else "none",
        "tva_best_family": tva_top[0]["family"] if tva_top else "none",
        "sd_best_score": sd_best_score,
        "tva_best_score": tva_best_score,
        "sd_target_margin_to_best": (sd_target_score - sd_best_score) if sd_target_score is not None else None,
        "tva_target_margin_to_best": (tva_target_score - tva_best_score) if tva_target_score is not None else None,
        "sd_target_top3": any(item["family"] == target_family for item in sd_top[:3]),
        "tva_target_top3": any(item["family"] == target_family for item in tva_top[:3]),
        "sd_target_best": bool(sd_top and sd_top[0]["family"] == target_family),
        "tva_target_best": bool(tva_top and tva_top[0]["family"] == target_family),
    }


def iter_window_records(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for segment in segments:
        for window in segment.get("windows", []):
            records.append(
                {
                    "role": segment["role"],
                    "threshold": float(segment["threshold"]),
                    "anchor_cycle": int(segment["anchor_cycle"]),
                    "cycle": int(segment["cycle"]),
                    "seed": int(segment["seed"]),
                    "episode": int(segment["episode"]),
                    "episode_reward": float(segment["episode_reward"]),
                    "trace_csv": segment["trace_csv"],
                    "anchor_type": window["anchor_type"],
                    "anchor_index": int(window["anchor_index"]),
                    "anchor_step": int(window["anchor_step"]),
                    "dominant_action_family": window["dominant_action_family"],
                    "dominant_primary_action": window["dominant_primary_action"],
                    "future_reward_sum": float(window["future_reward_sum"]),
                    "future_rootx_delta": float(window["future_rootx_delta"]),
                    "mean_vx": float(window["mean_vx"]),
                    "future_mean_vx": float(window["future_mean_vx"]),
                    "anchor_signature": window["anchor_signature"],
                }
            )
    return records


def select_records(records: list[dict[str, Any]], max_windows: int) -> list[dict[str, Any]]:
    if max_windows <= 0 or len(records) <= max_windows:
        return records
    # Keep entry-success windows first, then high-threshold and high-future-outcome examples.
    return sorted(
        records,
        key=lambda item: (
            item["role"] == "entry_success",
            item["threshold"],
            item["future_reward_sum"],
            item["future_rootx_delta"],
        ),
        reverse=True,
    )[:max_windows]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups["all"].append(rec)
        groups[f"role:{rec['role']}"].append(rec)
        groups[f"threshold:{rec['threshold']:g}|role:{rec['role']}"].append(rec)
        groups[f"threshold:{rec['threshold']:g}|cycle:{rec['anchor_cycle']:03d}|family:{rec['target_family']}"].append(rec)

    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items()):
        sd_best_counts = Counter(rec["sd_best_family"] for rec in vals)
        tva_best_counts = Counter(rec["tva_best_family"] for rec in vals)
        target_counts = Counter(rec["target_family"] for rec in vals)
        out[key] = {
            "count": len(vals),
            "target_family_counts": dict(target_counts.most_common()),
            "sd_best_family_counts": dict(sd_best_counts.most_common()),
            "tva_best_family_counts": dict(tva_best_counts.most_common()),
            "sd_target_best_rate": mean_or_zero(1.0 if rec["sd_target_best"] else 0.0 for rec in vals),
            "tva_target_best_rate": mean_or_zero(1.0 if rec["tva_target_best"] else 0.0 for rec in vals),
            "sd_target_top3_rate": mean_or_zero(1.0 if rec["sd_target_top3"] else 0.0 for rec in vals),
            "tva_target_top3_rate": mean_or_zero(1.0 if rec["tva_target_top3"] else 0.0 for rec in vals),
            "mean_sd_target_margin_to_best": mean_or_zero(
                rec["sd_target_margin_to_best"] for rec in vals if rec["sd_target_margin_to_best"] is not None
            ),
            "mean_tva_target_margin_to_best": mean_or_zero(
                rec["tva_target_margin_to_best"] for rec in vals if rec["tva_target_margin_to_best"] is not None
            ),
            "mean_future_reward_sum": mean_or_zero(rec["future_reward_sum"] for rec in vals),
            "mean_future_rootx_delta": mean_or_zero(rec["future_rootx_delta"] for rec in vals),
        }
    return out


def candidate_support(records: list[dict[str, Any]], min_count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec["threshold"], rec["anchor_cycle"], rec["role"], rec["target_family"])].append(rec)
    out: list[dict[str, Any]] = []
    for (threshold, cycle, role, family), vals in sorted(groups.items()):
        if len(vals) < min_count:
            continue
        out.append(
            {
                "threshold": threshold,
                "anchor_cycle": cycle,
                "role": role,
                "action_family": family,
                "count": len(vals),
                "sd_target_best_rate": mean_or_zero(1.0 if rec["sd_target_best"] else 0.0 for rec in vals),
                "tva_target_best_rate": mean_or_zero(1.0 if rec["tva_target_best"] else 0.0 for rec in vals),
                "sd_target_top3_rate": mean_or_zero(1.0 if rec["sd_target_top3"] else 0.0 for rec in vals),
                "tva_target_top3_rate": mean_or_zero(1.0 if rec["tva_target_top3"] else 0.0 for rec in vals),
                "mean_sd_target_margin_to_best": mean_or_zero(
                    rec["sd_target_margin_to_best"]
                    for rec in vals
                    if rec["sd_target_margin_to_best"] is not None
                ),
                "mean_tva_target_margin_to_best": mean_or_zero(
                    rec["tva_target_margin_to_best"]
                    for rec in vals
                    if rec["tva_target_margin_to_best"] is not None
                ),
                "mean_future_reward_sum": mean_or_zero(rec["future_reward_sum"] for rec in vals),
                "mean_future_rootx_delta": mean_or_zero(rec["future_rootx_delta"] for rec in vals),
                "status": "offline_entry_counterfactual_audited",
            }
        )
    out.sort(
        key=lambda item: (
            item["role"] == "entry_success",
            item["sd_target_top3_rate"] + item["tva_target_top3_rate"],
            item["mean_future_reward_sum"],
            item["count"],
        ),
        reverse=True,
    )
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Ability Entry Counterfactual Audit",
        "",
        f"Segments: `{payload['source_segments']}`",
        f"Windows audited: `{payload['num_records']}`",
        f"Horizon: `{payload['config']['horizon']}`",
        "",
        "## Aggregate",
        "",
        "| Group | Count | SD top-3 | TVA top-3 | SD best | TVA best | Mean future reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in payload["aggregate"].items():
        if not (key == "all" or key.startswith("role:") or key.startswith("threshold:")):
            continue
        lines.append(
            "| "
            f"`{key}` | {stats['count']} | "
            f"{stats['sd_target_top3_rate']:.3f} | {stats['tva_target_top3_rate']:.3f} | "
            f"{stats['sd_target_best_rate']:.3f} | {stats['tva_target_best_rate']:.3f} | "
            f"{stats['mean_future_reward_sum']:.3f} |"
        )
    lines.extend(["", "## Candidate Family Support", ""])
    for item in payload["candidate_support"][:40]:
        lines.append(
            f"- R>`{item['threshold']:g}` cycle `{item['anchor_cycle']:03d}` "
            f"`{item['role']}` `{item['action_family']}`: count=`{item['count']}`, "
            f"SD top3=`{item['sd_target_top3_rate']:.3f}`, TVA top3=`{item['tva_target_top3_rate']:.3f}`, "
            f"future_reward=`{item['mean_future_reward_sum']:.3f}`"
        )
    lines.extend(["", "## Largest Disagreements", ""])
    disagreements = sorted(
        payload["records"],
        key=lambda rec: (
            rec["role"] == "entry_success",
            not rec["sd_target_top3"],
            not rec["tva_target_top3"],
            rec["future_reward_sum"],
        ),
        reverse=True,
    )
    for rec in disagreements[:20]:
        lines.append(
            f"- R>`{rec['threshold']:g}` seed `{rec['seed']}` cycle `{rec['cycle']:03d}` ep `{rec['episode']}` "
            f"step `{rec['anchor_step']}` target=`{rec['target_family']}`, "
            f"SD best=`{rec['sd_best_family']}`, TVA best=`{rec['tva_best_family']}`, "
            f"future_reward=`{rec['future_reward_sum']:.3f}`, trace=`{rec['trace_csv']}`"
        )
    lines.append("")
    lines.append(
        "该审计只判断入口候选动作族是否被同周期 SD/TVA 支持；支持结果仍需跨周期后果验证后才可转为在线规则。"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--min-candidate-count", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.segments)
    raw_records = select_records(iter_window_records(payload.get("segments", [])), args.max_windows)
    action_bank = build_action_bank(args.amplitudes)

    model_cache: dict[int, DMControlHopperSDModel] = {}
    tva_cache: dict[int, torch.nn.Module] = {}
    trace_cache: dict[str, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for item in raw_records:
        cycle = int(item["anchor_cycle"])
        sd_path, tva_path = checkpoint_paths(args.results_root, cycle)
        if not sd_path.exists() or not tva_path.exists():
            skipped.append({**item, "reason": "missing_sd_or_tva_checkpoint", "sd": str(sd_path), "tva": str(tva_path)})
            continue
        if cycle not in model_cache:
            model_cache[cycle] = load_sd_model(sd_path, args.device)
        if cycle not in tva_cache:
            tva_cache[cycle] = load_value_net(tva_path, args.device)
        rows = trace_cache.get(item["trace_csv"])
        if rows is None:
            path = Path(item["trace_csv"])
            if not path.exists():
                skipped.append({**item, "reason": "missing_trace_csv"})
                continue
            rows = read_csv(path)
            trace_cache[item["trace_csv"]] = rows
        if item["anchor_index"] < 0 or item["anchor_index"] >= len(rows):
            skipped.append({**item, "reason": "anchor_index_out_of_range"})
            continue
        result = evaluate_anchor(
            model_cache[cycle],
            tva_cache[cycle],
            rows[item["anchor_index"]],
            action_bank,
            args.horizon,
            args.device,
        )
        target = item["dominant_action_family"]
        scored = score_target_family(result, target)
        records.append(
            {
                **item,
                **scored,
                "actual_anchor_family": result["actual_anchor_family"],
                "sd_top_families": result["sd_top_families"],
                "tva_top_families": result["tva_top_families"],
            }
        )

    report = {
        "analysis_family": "hopper_ability_entry_counterfactual_v1",
        "source_segments": str(args.segments),
        "results_root": str(args.results_root),
        "num_raw_windows": len(raw_records),
        "num_records": len(records),
        "num_skipped": len(skipped),
        "config": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "max_windows": args.max_windows,
            "min_candidate_count": args.min_candidate_count,
            "device": args.device,
        },
        "aggregate": aggregate(records),
        "candidate_support": candidate_support(records, args.min_candidate_count),
        "records": records,
        "skipped": skipped,
    }
    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"wrote ability entry counterfactual audit: {args.output}")
    print(f"windows: {len(records)} skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
