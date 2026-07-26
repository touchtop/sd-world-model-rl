#!/usr/bin/env python3
"""Prototype effect-space rules for Hopper ability-entry windows.

The script keeps the rule object simple by moving complexity into a projected
effect space.  For each ability-entry group, it estimates an entry direction

    v_entry = mean(delta_z | entry_success) - mean(delta_z | matched_failure)

where delta_z is the SD-predicted short-horizon effect vector of the actual
action.  It then checks whether ranking actions by alignment with v_entry
better separates entry-success windows from matched failures than raw action
family frequency.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import f, load_json, mean_or_zero, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank, desirability_batch  # noqa: E402
from audit_hopper_ability_entry_counterfactuals import (  # noqa: E402
    checkpoint_paths,
    iter_window_records,
    load_sd_model,
    load_value_net,
)


DEFAULT_SEGMENTS = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json"
)
DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit.json"
)
DEFAULT_RESULTS_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1"
)

EFFECT_KEYS = [
    "tva",
    "desirability",
    "vx",
    "vz",
    "torso_z",
    "support_contact",
    "full_foot_support",
    "upright_or_prepared",
    "hop",
    "not_prone_rear",
    "not_falling",
    "rotation_control",
    "pitch_control",
]


def safe_feature(feat: dict[str, torch.Tensor], key: str, idx: int) -> float:
    value = feat.get(key)
    if value is None:
        return 0.0
    return float(value[idx].item())


def effect_vector(
    feat: dict[str, torch.Tensor],
    values: torch.Tensor,
    idx: int,
) -> list[float]:
    prone = safe_feature(feat, "grounded_prone", idx)
    rear = safe_feature(feat, "grounded_rear", idx)
    falling = safe_feature(feat, "falling_air", idx)
    pitch = safe_feature(feat, "pitch", idx)
    pitch_rate = safe_feature(feat, "pitch_rate", idx)
    upright = safe_feature(feat, "grounded_upright", idx)
    prepared = safe_feature(feat, "grounded_prepared", idx)
    return [
        float(values[idx].item()),
        float(desirability_batch(feat)[idx].item()),
        safe_feature(feat, "vx", idx),
        safe_feature(feat, "vz", idx),
        safe_feature(feat, "torso_z", idx),
        safe_feature(feat, "support_contact", idx),
        safe_feature(feat, "full_foot_support", idx),
        max(upright, prepared),
        safe_feature(feat, "grounded_hop", idx),
        1.0 - max(prone, rear),
        1.0 - falling,
        1.0 - min(abs(pitch_rate) / 8.0, 1.0),
        1.0 - min(abs(pitch) / 2.0, 1.0),
    ]


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


def dot(xs: list[float], ys: list[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))


def family_from_name(name: str) -> str:
    if name == "actual":
        return "actual"
    if name == "zero":
        return "zero"
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def group_key(record: dict[str, Any]) -> str:
    return f"threshold:{record['threshold']:g}|cycle:{record['anchor_cycle']:03d}"


def select_effect_records(records: list[dict[str, Any]], max_windows: int) -> list[dict[str, Any]]:
    if max_windows <= 0 or len(records) <= max_windows:
        return records
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[f"{group_key(record)}|role:{record['role']}"].append(record)
    selected: list[dict[str, Any]] = []
    labels = sorted(groups)
    per_group = max(1, max_windows // max(1, len(labels)))
    leftovers: list[dict[str, Any]] = []
    for label in labels:
        vals = sorted(
            groups[label],
            key=lambda item: (item["future_reward_sum"], item["future_rootx_delta"]),
            reverse=True,
        )
        selected.extend(vals[:per_group])
        leftovers.extend(vals[per_group:])
    if len(selected) < max_windows:
        leftovers = sorted(
            leftovers,
            key=lambda item: (item["role"] == "entry_success", item["threshold"], item["future_reward_sum"]),
            reverse=True,
        )
        selected.extend(leftovers[: max_windows - len(selected)])
    return selected[:max_windows]


def evaluate_record_effects(
    model: torch.nn.Module,
    value_net: torch.nn.Module,
    row: dict[str, str],
    action_bank: list[tuple[str, torch.Tensor]],
    horizon: int,
    device: str,
) -> dict[str, Any]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32, device=device).unsqueeze(0))
    actual_action = torch.tensor(row_to_action(row), dtype=torch.float32, device=device).unsqueeze(0).clamp(-1.0, 1.0)
    names = ["actual"] + [name for name, _ in action_bank]
    actions = torch.cat([actual_action, *[action.to(device) for _, action in action_bank]], dim=0)
    cur = state.repeat(actions.shape[0], 1)
    with torch.no_grad():
        current_feat = model._state_reward_features(state)
        current_value = value_net(state).detach().view(-1)
        current_z = effect_vector(current_feat, current_value, 0)
        for _ in range(horizon):
            cur = model(cur, actions)
        terminal_feat = model._state_reward_features(cur)
        terminal_values = value_net(cur).detach().view(-1)
    actions_out: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        terminal_z = effect_vector(terminal_feat, terminal_values, idx)
        delta_z = [b - a for a, b in zip(current_z, terminal_z)]
        actions_out.append(
            {
                "name": name,
                "family": family_from_name(name),
                "delta_z": delta_z,
                "terminal_z": terminal_z,
            }
        )
    return {"current_z": current_z, "actions": actions_out}


def learn_directions(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["global"] = records
    for rec in records:
        grouped[group_key(rec)].append(rec)

    directions: dict[str, dict[str, Any]] = {}
    for key, vals in grouped.items():
        success = [rec["actual_delta_z"] for rec in vals if rec["role"] == "entry_success"]
        failure = [rec["actual_delta_z"] for rec in vals if rec["role"] == "matched_failure"]
        if not success or not failure:
            continue
        mean_success = [mean_or_zero(vec[i] for vec in success) for i in range(len(EFFECT_KEYS))]
        mean_failure = [mean_or_zero(vec[i] for vec in failure) for i in range(len(EFFECT_KEYS))]
        raw = [a - b for a, b in zip(mean_success, mean_failure)]
        unit = normalize(raw)
        components = [
            {"key": key_name, "value": value}
            for key_name, value in sorted(zip(EFFECT_KEYS, raw), key=lambda item: abs(item[1]), reverse=True)
        ]
        directions[key] = {
            "key": key,
            "success_count": len(success),
            "failure_count": len(failure),
            "raw_direction": raw,
            "unit_direction": unit,
            "top_components": components[:8],
        }
    return directions


def rank_by_direction(actions: list[dict[str, Any]], direction: list[float]) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for action in actions:
        score = dot(action["delta_z"], direction)
        family = action["family"]
        current = by_family.get(family)
        if current is None or score > current["alignment"]:
            by_family[family] = {
                "family": family,
                "name": action["name"],
                "alignment": score,
                "delta_z": action["delta_z"],
            }
    return sorted(by_family.values(), key=lambda item: item["alignment"], reverse=True)


def apply_directions(records: list[dict[str, Any]], directions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        key = group_key(rec)
        direction_info = directions.get(key) or directions.get("global")
        if direction_info is None:
            continue
        ranked = rank_by_direction(rec["actions"], direction_info["unit_direction"])
        actual_family = "actual"
        observed_family = rec["dominant_action_family"]
        actual_alignment = next((item["alignment"] for item in ranked if item["family"] == actual_family), 0.0)
        observed_rank = 1 + sum(
            1 for item in ranked
            if item["family"] != observed_family and item["alignment"] > next(
                (x["alignment"] for x in ranked if x["family"] == observed_family),
                float("-inf"),
            )
        )
        out.append(
            {
                **{k: v for k, v in rec.items() if k not in {"actions", "actual_delta_z"}},
                "effect_direction_key": direction_info["key"],
                "actual_alignment": actual_alignment,
                "best_effect_family": ranked[0]["family"] if ranked else "none",
                "best_effect_alignment": ranked[0]["alignment"] if ranked else 0.0,
                "observed_family": observed_family,
                "observed_family_rank": observed_rank,
                "observed_family_top3": observed_rank <= 3,
                "effect_top_families": ranked[:5],
            }
        )
    return out


def auc_positive_greater(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    total = 0.0
    for x in pos:
        for y in neg:
            if x > y:
                total += 1.0
            elif x == y:
                total += 0.5
    return total / (len(pos) * len(neg))


def aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in scored:
        groups["all"].append(rec)
        groups[f"role:{rec['role']}"].append(rec)
        groups[group_key(rec)].append(rec)
        groups[f"{group_key(rec)}|role:{rec['role']}"].append(rec)

    out: dict[str, Any] = {}
    for key, vals in sorted(groups.items()):
        out[key] = {
            "count": len(vals),
            "mean_actual_alignment": mean_or_zero(rec["actual_alignment"] for rec in vals),
            "mean_best_effect_alignment": mean_or_zero(rec["best_effect_alignment"] for rec in vals),
            "observed_family_top3_rate": mean_or_zero(1.0 if rec["observed_family_top3"] else 0.0 for rec in vals),
            "best_effect_family_counts": dict(Counter(rec["best_effect_family"] for rec in vals).most_common()),
            "mean_future_reward_sum": mean_or_zero(rec["future_reward_sum"] for rec in vals),
            "mean_future_rootx_delta": mean_or_zero(rec["future_rootx_delta"] for rec in vals),
        }

    for key in sorted({group_key(rec) for rec in scored} | {"global"}):
        vals = scored if key == "global" else [rec for rec in scored if group_key(rec) == key]
        pos = [rec["actual_alignment"] for rec in vals if rec["role"] == "entry_success"]
        neg = [rec["actual_alignment"] for rec in vals if rec["role"] == "matched_failure"]
        if pos and neg:
            out[f"separation:{key}"] = {
                "success_count": len(pos),
                "failure_count": len(neg),
                "success_mean_actual_alignment": mean_or_zero(pos),
                "failure_mean_actual_alignment": mean_or_zero(neg),
                "alignment_margin": mean_or_zero(pos) - mean_or_zero(neg),
                "auc_success_gt_failure": auc_positive_greater(pos, neg),
            }
    return out


def effect_rule_candidates(scored: list[dict[str, Any]], directions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key, direction in directions.items():
        if key == "global":
            continue
        vals = [rec for rec in scored if rec["effect_direction_key"] == key and rec["role"] == "entry_success"]
        failures = [rec for rec in scored if rec["effect_direction_key"] == key and rec["role"] == "matched_failure"]
        if not vals or not failures:
            continue
        family_counts = Counter(rec["best_effect_family"] for rec in vals)
        candidates.append(
            {
                "id": f"{key}->prefer_effect_alignment",
                "effect_direction_key": key,
                "rule_form": "context_key -> prefer_effect_direction",
                "entry_success_windows": len(vals),
                "matched_failure_windows": len(failures),
                "success_mean_actual_alignment": mean_or_zero(rec["actual_alignment"] for rec in vals),
                "failure_mean_actual_alignment": mean_or_zero(rec["actual_alignment"] for rec in failures),
                "alignment_margin": mean_or_zero(rec["actual_alignment"] for rec in vals)
                - mean_or_zero(rec["actual_alignment"] for rec in failures),
                "auc_success_gt_failure": auc_positive_greater(
                    [rec["actual_alignment"] for rec in vals],
                    [rec["actual_alignment"] for rec in failures],
                ),
                "best_effect_family_counts_on_success": dict(family_counts.most_common()),
                "top_direction_components": direction["top_components"],
                "status": "offline_effect_space_candidate",
            }
        )
    candidates.sort(key=lambda item: (item["auc_success_gt_failure"], item["alignment_margin"]), reverse=True)
    return candidates


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Ability Entry Effect-Space Audit",
        "",
        f"Segments: `{payload['source_segments']}`",
        f"Records: `{payload['num_records']}`",
        f"Horizon: `{payload['config']['horizon']}`",
        "",
        "## Separation",
        "",
        "| Group | Success | Failure | Margin | AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in payload["aggregate"].items():
        if not key.startswith("separation:"):
            continue
        lines.append(
            "| "
            f"`{key.replace('separation:', '')}` | "
            f"{stats['success_count']} | {stats['failure_count']} | "
            f"{stats['alignment_margin']:.4f} | {stats['auc_success_gt_failure']:.3f} |"
        )
    lines.extend(["", "## Effect Rule Candidates", ""])
    for item in payload["effect_rule_candidates"][:30]:
        components = ", ".join(
            f"{c['key']}={c['value']:.3f}" for c in item["top_direction_components"][:5]
        )
        lines.append(
            f"- `{item['id']}`: success=`{item['entry_success_windows']}`, "
            f"failure=`{item['matched_failure_windows']}`, "
            f"margin=`{item['alignment_margin']:.4f}`, "
            f"AUC=`{item['auc_success_gt_failure']:.3f}`, "
            f"best_families=`{item['best_effect_family_counts_on_success']}`, "
            f"components=`{components}`"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "该报告验证的是效果空间方向是否能作为简洁规则对象；若 AUC 和 margin 稳定为正，规则可保持为 "
        "`context_key -> prefer_effect_direction`，在线阶段仍只需对候选动作按 SD/TVA 推演后果排序。"
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
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.segments)
    raw_records = select_effect_records(iter_window_records(payload.get("segments", [])), args.max_windows)
    action_bank = build_action_bank(args.amplitudes)
    model_cache: dict[int, torch.nn.Module] = {}
    tva_cache: dict[int, torch.nn.Module] = {}
    trace_cache: dict[str, list[dict[str, str]]] = {}
    effect_records: list[dict[str, Any]] = []
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
        anchor_index = int(item["anchor_index"])
        if anchor_index < 0 or anchor_index >= len(rows):
            skipped.append({**item, "reason": "anchor_index_out_of_range"})
            continue
        effects = evaluate_record_effects(
            model_cache[cycle],
            tva_cache[cycle],
            rows[anchor_index],
            action_bank,
            args.horizon,
            args.device,
        )
        actual_delta = next(action["delta_z"] for action in effects["actions"] if action["name"] == "actual")
        effect_records.append({**item, "actual_delta_z": actual_delta, "actions": effects["actions"]})

    directions = learn_directions(effect_records)
    scored = apply_directions(effect_records, directions)
    report = {
        "analysis_family": "hopper_ability_entry_effect_space_v1",
        "source_segments": str(args.segments),
        "results_root": str(args.results_root),
        "effect_keys": EFFECT_KEYS,
        "num_raw_windows": len(raw_records),
        "num_records": len(scored),
        "num_skipped": len(skipped),
        "config": {
            "horizon": args.horizon,
            "amplitudes": args.amplitudes,
            "max_windows": args.max_windows,
            "device": args.device,
        },
        "directions": directions,
        "aggregate": aggregate(scored),
        "effect_rule_candidates": effect_rule_candidates(scored, directions),
        "records": scored,
        "skipped": skipped,
    }
    write_json(args.output, report)
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"wrote ability entry effect-space audit: {args.output}")
    print(f"records: {len(scored)} skipped: {len(skipped)} directions: {len(directions)}")


if __name__ == "__main__":
    main()
