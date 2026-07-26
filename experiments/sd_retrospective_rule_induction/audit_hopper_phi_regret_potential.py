#!/usr/bin/env python3
"""Audit TVA as Phi for local counterfactual regret on Hopper entry windows.

This script is an offline diagnostic.  It tests whether a learned value
potential Phi, currently provided by TVA, can support the first-principles
object:

    U_h(s, a) = Phi(SD_h(s, a))
    Regret_h(s, a) = max_{a'} U_h(s, a') - U_h(s, a)
    Adv_h(s, a) = U_h(s, a) - mean_{a'} U_h(s, a')

The audit does not modify online rules or planner priors.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import load_json, mean_or_zero, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from mine_hopper_v8_transition_actions import primary_action_family  # noqa: E402
from audit_hopper_ability_entry_counterfactuals import checkpoint_paths, iter_window_records, load_sd_model, load_value_net  # noqa: E402
from audit_hopper_ability_entry_effect_space import DEFAULT_RESULTS_ROOT, DEFAULT_SEGMENTS, family_from_name  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater  # noqa: E402
from audit_hopper_ability_entry_minimal_stable_effect import dedupe_records, entry_tier  # noqa: E402
from audit_hopper_tva_terminal_compensation import pearson, spearman, parse_horizons  # noqa: E402


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_phi_regret_potential_audit.json"
)


def entropy_norm(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / float(total)
        if p > 0:
            entropy -= p * math.log(p)
    return float(entropy / math.log(len(counts)))


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def context_key(record: dict[str, Any]) -> str:
    sig = record.get("anchor_signature") or {}
    parts = [
        str(sig.get("omega_bin", "omega_unknown")),
        str(sig.get("vx_bin", "vx_unknown")),
        str(sig.get("support_bin", "support_unknown")),
        str(sig.get("phase", "phase_unknown")),
        str(sig.get("risk_bin", "risk_unknown")),
    ]
    return "|".join(parts)


def select_records(records: list[dict[str, Any]], max_windows: int) -> list[dict[str, Any]]:
    if max_windows <= 0 or len(records) <= max_windows:
        return records
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[f"{entry_tier(rec)}|cycle:{int(rec['anchor_cycle']):03d}|role:{rec['role']}"].append(rec)
    selected: list[dict[str, Any]] = []
    per_group = max(1, max_windows // max(1, len(groups)))
    leftovers: list[dict[str, Any]] = []
    for key in sorted(groups):
        vals = sorted(groups[key], key=lambda item: (item["future_reward_sum"], item["future_rootx_delta"]), reverse=True)
        selected.extend(vals[:per_group])
        leftovers.extend(vals[per_group:])
    if len(selected) < max_windows:
        leftovers = sorted(leftovers, key=lambda item: item["future_reward_sum"], reverse=True)
        selected.extend(leftovers[: max_windows - len(selected)])
    return selected[:max_windows]


def summarize_pairs(rows: list[dict[str, Any]], pred_key: str, target_key: str) -> dict[str, Any]:
    xs = [float(row[pred_key]) for row in rows if row.get(pred_key) is not None and row.get(target_key) is not None]
    ys = [float(row[target_key]) for row in rows if row.get(pred_key) is not None and row.get(target_key) is not None]
    if not xs:
        return {
            "n": 0,
            "pred_mean": 0.0,
            "target_mean": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "top_third_target_mean": 0.0,
            "bottom_third_target_mean": 0.0,
            "top_minus_bottom_target": 0.0,
        }
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    k = max(1, len(order) // 3)
    bottom = [ys[i] for i in order[:k]]
    top = [ys[i] for i in order[-k:]]
    return {
        "n": len(xs),
        "pred_mean": safe_mean(xs),
        "pred_std": safe_std(xs),
        "target_mean": safe_mean(ys),
        "target_std": safe_std(ys),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "top_third_target_mean": safe_mean(top),
        "bottom_third_target_mean": safe_mean(bottom),
        "top_minus_bottom_target": safe_mean(top) - safe_mean(bottom),
    }


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_phi_regret(
    model: torch.nn.Module,
    value_net: torch.nn.Module,
    row: dict[str, str],
    action_bank: list[tuple[str, torch.Tensor]],
    horizons: list[int],
    device: str,
) -> dict[str, Any]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32, device=device).unsqueeze(0))
    actual_action = torch.tensor(row_to_action(row), dtype=torch.float32, device=device).unsqueeze(0).clamp(-1.0, 1.0)
    names = ["actual"] + [name for name, _ in action_bank]
    actions = torch.cat([actual_action, *[action.to(device) for _, action in action_bank]], dim=0)
    actual_primary = primary_action_family(row, threshold=0.20)
    with torch.no_grad():
        current_phi = float(value_net(state).detach().view(-1)[0].item())
    horizon_out: dict[str, Any] = {}
    for h in horizons:
        cur = state.repeat(actions.shape[0], 1)
        with torch.no_grad():
            for _ in range(h):
                cur = model(cur, actions)
            values = [float(x) for x in value_net(cur).detach().view(-1).cpu().tolist()]
        actual_u = values[0]
        probe_values = values[1:]
        probe_names = names[1:]
        mean_probe = safe_mean(probe_values)
        best_probe = max(probe_values) if probe_values else actual_u
        by_family: dict[str, dict[str, Any]] = {}
        for name, value in zip(probe_names, probe_values):
            family = family_from_name(name)
            current = by_family.get(family)
            if current is None or value > current["u"]:
                by_family[family] = {
                    "family": family,
                    "name": name,
                    "u": value,
                    "adv": value - mean_probe,
                    "regret": best_probe - value,
                }
        sorted_families = sorted(by_family.values(), key=lambda item: item["u"], reverse=True)
        best_family = sorted_families[0]["family"] if sorted_families else "none"
        family_counts = Counter({item["family"]: 1 for item in sorted_families})
        del family_counts
        target = by_family.get(actual_primary)
        horizon_out[str(h)] = {
            "current_phi": current_phi,
            "actual_primary_family": actual_primary,
            "actual_u": actual_u,
            "actual_adv": actual_u - mean_probe,
            "actual_regret": best_probe - actual_u,
            "probe_mean_u": mean_probe,
            "probe_best_u": best_probe,
            "probe_best_family": best_family,
            "target_family_u": float(target["u"]) if target else None,
            "target_family_adv": float(target["adv"]) if target else None,
            "target_family_regret": float(target["regret"]) if target else None,
            "family_values": sorted_families,
        }
    return {"current_phi": current_phi, "horizons": horizon_out}


def collect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(args.segments)
    raw_records = select_records(iter_window_records(payload.get("segments", [])), args.max_windows)
    raw_records = dedupe_records(raw_records)
    action_bank = build_action_bank(args.amplitudes)
    model_cache: dict[int, torch.nn.Module] = {}
    tva_cache: dict[int, torch.nn.Module] = {}
    trace_cache: dict[str, list[dict[str, str]]] = {}
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in raw_records:
        cycle = int(item["anchor_cycle"])
        sd_path, tva_path = checkpoint_paths(args.results_root, cycle)
        if not sd_path.exists() or not tva_path.exists():
            skipped.append({**item, "reason": "missing_sd_or_tva_checkpoint"})
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
        idx = int(item["anchor_index"])
        if idx < 0 or idx >= len(rows):
            skipped.append({**item, "reason": "anchor_index_out_of_range"})
            continue
        audit = evaluate_phi_regret(
            model_cache[cycle],
            tva_cache[cycle],
            rows[idx],
            action_bank,
            args.horizons,
            args.device,
        )
        out.append(
            {
                **item,
                "tier": entry_tier(item),
                "context_key": context_key(item),
                "current_phi": audit["current_phi"],
                "phi_regret": audit["horizons"],
            }
        )
    return out, skipped


def flatten_horizon(records: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    key = str(horizon)
    rows: list[dict[str, Any]] = []
    for rec in records:
        item = rec["phi_regret"].get(key)
        if not item:
            continue
        rows.append(
            {
                "role": rec["role"],
                "tier": rec["tier"],
                "cycle": int(rec["anchor_cycle"]),
                "context_key": rec["context_key"],
                "threshold": float(rec["threshold"]),
                "current_phi": float(rec["current_phi"]),
                "actual_u": float(item["actual_u"]),
                "actual_adv": float(item["actual_adv"]),
                "actual_regret": float(item["actual_regret"]),
                "probe_best_family": item["probe_best_family"],
                "target_family": item["actual_primary_family"],
                "target_family_adv": item["target_family_adv"],
                "target_family_regret": item["target_family_regret"],
                "future_reward_sum": float(rec["future_reward_sum"]),
                "future_rootx_delta": float(rec["future_rootx_delta"]),
                "future_mean_vx": float(rec["future_mean_vx"]),
                "episode_reward": float(rec["episode_reward"]),
            }
        )
    return rows


def phi_state_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "role": rec["role"],
            "tier": rec["tier"],
            "current_phi": float(rec["current_phi"]),
            "future_reward_sum": float(rec["future_reward_sum"]),
            "future_rootx_delta": float(rec["future_rootx_delta"]),
            "future_mean_vx": float(rec["future_mean_vx"]),
            "episode_reward": float(rec["episode_reward"]),
        }
        for rec in records
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"role:{row['role']}"].append(row)
        groups[f"tier:{row['tier']}"].append(row)
    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items()):
        out[key] = {
            "future_reward_sum": summarize_pairs(vals, "current_phi", "future_reward_sum"),
            "future_rootx_delta": summarize_pairs(vals, "current_phi", "future_rootx_delta"),
            "future_mean_vx": summarize_pairs(vals, "current_phi", "future_mean_vx"),
            "episode_reward": summarize_pairs(vals, "current_phi", "episode_reward"),
        }
    return out


def actual_action_separation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"tier:{row['tier']}"].append(row)
        groups[f"tier:{row['tier']}|cycle:{row['cycle']:03d}"].append(row)
    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items()):
        success = [row for row in vals if row["role"] == "entry_success"]
        failure = [row for row in vals if row["role"] == "matched_failure"]
        out[key] = {
            "count": len(vals),
            "success_count": len(success),
            "failure_count": len(failure),
            "actual_adv_success_mean": safe_mean([row["actual_adv"] for row in success]),
            "actual_adv_failure_mean": safe_mean([row["actual_adv"] for row in failure]),
            "actual_adv_margin": safe_mean([row["actual_adv"] for row in success]) - safe_mean([row["actual_adv"] for row in failure]),
            "actual_adv_auc_success_gt_failure": auc_positive_greater(
                [row["actual_adv"] for row in success],
                [row["actual_adv"] for row in failure],
            ) if success and failure else 0.0,
            "actual_regret_success_mean": safe_mean([row["actual_regret"] for row in success]),
            "actual_regret_failure_mean": safe_mean([row["actual_regret"] for row in failure]),
            "actual_regret_margin_failure_minus_success": safe_mean([row["actual_regret"] for row in failure]) - safe_mean([row["actual_regret"] for row in success]),
            "actual_low_regret_auc_success_gt_failure": auc_positive_greater(
                [-row["actual_regret"] for row in success],
                [-row["actual_regret"] for row in failure],
            ) if success and failure else 0.0,
            "actual_u_future_reward": summarize_pairs(vals, "actual_u", "future_reward_sum"),
            "actual_adv_future_reward": summarize_pairs(vals, "actual_adv", "future_reward_sum"),
            "actual_regret_future_reward": summarize_pairs(
                [{**row, "neg_actual_regret": -row["actual_regret"]} for row in vals],
                "neg_actual_regret",
                "future_reward_sum",
            ),
        }
    return out


def family_regret_contexts(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['tier']}|{row['context_key']}"].append(row)
    candidates: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        if len(vals) < args.min_context_records:
            continue
        cycles = sorted({int(row["cycle"]) for row in vals})
        if len(cycles) < args.min_context_cycles:
            continue
        best_counts = Counter(str(row["probe_best_family"]) for row in vals)
        top_family, top_count = best_counts.most_common(1)[0]
        top_rows = [row for row in vals if row["probe_best_family"] == top_family]
        observed_rows = [row for row in vals if row["target_family"] == top_family]
        other_rows = [row for row in vals if row["target_family"] != top_family]
        cycle_best_rates = []
        for cycle in cycles:
            cvals = [row for row in vals if int(row["cycle"]) == cycle]
            cycle_best_rates.append(mean_or_zero(1.0 if row["probe_best_family"] == top_family else 0.0 for row in cvals))
        candidates.append(
            {
                "key": key,
                "tier": key.split("|", 1)[0],
                "context_key": key.split("|", 1)[1],
                "count": len(vals),
                "cycles": cycles,
                "top_family": top_family,
                "top_family_best_rate": float(top_count) / float(len(vals)),
                "best_family_entropy": entropy_norm(best_counts),
                "cycle_best_rate_min": min(cycle_best_rates) if cycle_best_rates else 0.0,
                "cycle_best_rate_mean": safe_mean(cycle_best_rates),
                "observed_top_family_count": len(observed_rows),
                "observed_top_family_future_reward_mean": safe_mean([row["future_reward_sum"] for row in observed_rows]),
                "observed_other_future_reward_mean": safe_mean([row["future_reward_sum"] for row in other_rows]),
                "observed_top_minus_other_future_reward": safe_mean([row["future_reward_sum"] for row in observed_rows]) - safe_mean([row["future_reward_sum"] for row in other_rows]),
                "target_adv_when_observed_mean": safe_mean([row["target_family_adv"] for row in observed_rows if row["target_family_adv"] is not None]),
                "target_regret_when_observed_mean": safe_mean([row["target_family_regret"] for row in observed_rows if row["target_family_regret"] is not None]),
                "best_family_counts": dict(best_counts.most_common()),
            }
        )
    stable = [
        item for item in candidates
        if item["best_family_entropy"] <= args.max_best_family_entropy
        and item["top_family_best_rate"] >= args.min_best_family_rate
        and item["cycle_best_rate_min"] >= args.min_cycle_best_rate
    ]
    return {
        "num_contexts": len(candidates),
        "num_stable_like_contexts": len(stable),
        "stable_like_contexts": sorted(stable, key=lambda item: (item["top_family_best_rate"], -item["best_family_entropy"], item["count"]), reverse=True)[:50],
        "top_contexts": sorted(candidates, key=lambda item: (item["top_family_best_rate"], -item["best_family_entropy"], item["count"]), reverse=True)[:50],
    }


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    horizon_reports: dict[str, Any] = {}
    for h in args.horizons:
        rows = flatten_horizon(records, h)
        horizon_reports[str(h)] = {
            "actual_action_separation": actual_action_separation(rows),
            "family_regret_contexts": family_regret_contexts(rows, args),
            "actual_adv_summary": summarize_numeric([row["actual_adv"] for row in rows]),
            "actual_regret_summary": summarize_numeric([row["actual_regret"] for row in rows]),
        }
    return {
        "phi_state_calibration": phi_state_calibration(records),
        "horizons": horizon_reports,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Phi/Regret Potential Audit",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        f"Phi source: `{payload['config']['phi_source']}`",
        "",
        "## Phi 状态校准",
        "",
        "| Group | Target | n | Pearson | Spearman | Top-bottom target |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group, item in payload["summary"]["phi_state_calibration"].items():
        for target in ["future_reward_sum", "future_rootx_delta", "future_mean_vx", "episode_reward"]:
            stats = item[target]
            lines.append(
                f"| `{group}` | `{target}` | {stats['n']} | "
                f"{stats['pearson']:.3f} | {stats['spearman']:.3f} | {stats['top_minus_bottom_target']:.3f} |"
            )
    lines.extend([
        "",
        "## Horizon 诊断",
        "",
        "| h | Group | adv margin | adv AUC | regret margin | low-regret AUC | U-reward Spearman | stable-like contexts |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for h in payload["config"]["horizons"]:
        hrep = payload["summary"]["horizons"][str(h)]
        contexts = hrep["family_regret_contexts"]["num_stable_like_contexts"]
        for group in ["all", "tier:low_entry", "tier:high_entry"]:
            stats = hrep["actual_action_separation"].get(group)
            if not stats:
                continue
            u_reward = stats["actual_u_future_reward"]
            lines.append(
                f"| {h} | `{group}` | "
                f"{stats['actual_adv_margin']:.3f} | {stats['actual_adv_auc_success_gt_failure']:.3f} | "
                f"{stats['actual_regret_margin_failure_minus_success']:.3f} | "
                f"{stats['actual_low_regret_auc_success_gt_failure']:.3f} | "
                f"{u_reward['spearman']:.3f} | {contexts} |"
            )
    lines.extend([
        "",
        "## 通过稳定门槛的上下文",
        "",
        "| h | Context | Family | Count | Cycles | Best rate | Min cycle rate | Entropy | Observed reward margin |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for h in payload["config"]["horizons"]:
        stable = payload["summary"]["horizons"][str(h)]["family_regret_contexts"]["stable_like_contexts"][:12]
        for item in stable:
            lines.append(
                f"| {h} | `{item['context_key']}` | `{item['top_family']}` | {item['count']} | "
                f"`{','.join(str(c) for c in item['cycles'])}` | "
                f"{item['top_family_best_rate']:.3f} | {item['cycle_best_rate_min']:.3f} | "
                f"{item['best_family_entropy']:.3f} | {item['observed_top_minus_other_future_reward']:.3f} |"
            )
    lines.extend([
        "",
        "## 最高候选上下文（未必稳定）",
        "",
        "| h | Context | Family | Count | Cycles | Best rate | Min cycle rate | Entropy | Observed reward margin |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for h in payload["config"]["horizons"]:
        top = payload["summary"]["horizons"][str(h)]["family_regret_contexts"]["top_contexts"][:12]
        for item in top:
            lines.append(
                f"| {h} | `{item['context_key']}` | `{item['top_family']}` | {item['count']} | "
                f"`{','.join(str(c) for c in item['cycles'])}` | "
                f"{item['top_family_best_rate']:.3f} | {item['cycle_best_rate_min']:.3f} | "
                f"{item['best_family_entropy']:.3f} | {item['observed_top_minus_other_future_reward']:.3f} |"
            )
    lines.extend([
        "",
        "## 解释",
        "",
        "可用的 Phi 至少应满足三层检查：较高的当前 Phi 应对应较好的未来后果；经 SD 推演得到的 U_h 应与真实后续窗口存在同向关系；局部遗憾应能产生低熵、跨周期的上下文-动作候选。通过该审计仍不等于可以在线更新，它只是后续生成候选规则的前置条件。",
    ])
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=parse_horizons, default=[1, 3, 5])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--min-context-records", type=int, default=12)
    parser.add_argument("--min-context-cycles", type=int, default=2)
    parser.add_argument("--max-best-family-entropy", type=float, default=0.65)
    parser.add_argument("--min-best-family-rate", type=float, default=0.40)
    parser.add_argument("--min-cycle-best-rate", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, skipped = collect_records(args)
    summary = build_summary(records, args)
    payload = {
        "analysis_family": "hopper_phi_regret_potential_v1",
        "config": {
            "phi_source": "TVA",
            "segments": str(args.segments),
            "results_root": str(args.results_root),
            "horizons": args.horizons,
            "amplitudes": args.amplitudes,
            "max_windows": int(args.max_windows),
            "min_context_records": int(args.min_context_records),
            "min_context_cycles": int(args.min_context_cycles),
            "max_best_family_entropy": float(args.max_best_family_entropy),
            "min_best_family_rate": float(args.min_best_family_rate),
            "min_cycle_best_rate": float(args.min_cycle_best_rate),
        },
        "num_records": len(records),
        "num_skipped": len(skipped),
        "skipped": skipped[:200],
        "summary": summary,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote Phi/regret potential audit: {args.output}")
    print(f"records: {len(records)} skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
