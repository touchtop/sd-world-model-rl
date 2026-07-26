#!/usr/bin/env python3
"""Compute an initial explainability score for Hopper retrospective rules.

The score is an offline audit metric. It does not modify the rule library or
training loop. The goal is to separate rules that merely appear often from
rules that explain high/low trajectory differences and have SD counterfactual
support for useful decisions.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs/hopper-hop/1"
RESULT_ROOT = REPO_ROOT / "results/sd_retrospective_rule_induction"
RUN_ROOT = RESULT_ROOT / "closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506"
REPORT_JSON = RESULT_ROOT / "reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.json"
REPORT_MD = RESULT_ROOT / "reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.md"

JOINT_TO_DIM = {"waist": 0, "hip": 1, "knee": 2, "ankle": 3}
WEIGHTS = {
    "past_separation": 0.35,
    "past_label_power": 0.15,
    "cf_value_gain": 0.25,
    "transition_gain": 0.15,
    "basin_coverage": 0.10,
    "conflict_cost": -0.10,
    "complexity_cost": -0.05,
}


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value not in ("", None) else default
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def squash(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.tanh(x / scale)


def phase(row: dict[str, str]) -> str:
    if f(row, "falling_air") >= 0.5 or f(row, "falling_air_gt_05") >= 0.5:
        return "falling"
    if f(row, "grounded_rear") >= 0.5 or f(row, "grounded_rear_gt_05") >= 0.5:
        return "rear"
    if f(row, "grounded_prone") >= 0.5 or f(row, "grounded_prone_gt_05") >= 0.5:
        return "prone"
    if (
        f(row, "grounded_upright") >= 0.5
        or f(row, "grounded_upright_gt_05") >= 0.5
        or f(row, "grounded_prepared") >= 0.5
        or f(row, "grounded_prepared_gt_05") >= 0.5
    ):
        return "upright"
    return "other"


def substate_key_for_event(row: dict[str, str], event: str) -> str:
    pitch_rate = f(row, "pitch_rate")
    vx = f(row, "vx")
    full_foot = f(row, "full_foot_support")
    omega_bin = "omega_pos" if pitch_rate > 1.0 else "omega_neg" if pitch_rate < -1.0 else "omega_mid"
    vx_bin = "vx_high" if vx >= 0.8 else "vx_low" if vx < 0.2 else "vx_mid"
    support_bin = "support_none" if full_foot < 0.1 else "support_low" if full_foot < 0.35 else "support_some"
    return f"{event}|{omega_bin}|{vx_bin}|{support_bin}|{phase(row)}"


def event_keys(row: dict[str, str]) -> list[str]:
    keys = ["unsupported_high_torque"]
    if f(row, "rotation_penalty") >= 0.5 or abs(f(row, "pitch_rate")) >= 3.0:
        keys.append("rotation_risk")
    return [substate_key_for_event(row, event) for event in keys]


def action_family_from_name(name: str) -> str:
    parts = str(name).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(name)


def split_family(family: str) -> tuple[str | None, int]:
    if "_" not in family:
        return None, 0
    joint, sign = family.split("_", 1)
    if joint not in JOINT_TO_DIM:
        return None, 0
    return joint, 1 if sign == "pos" else -1


def action_alignment(row: dict[str, str], family: str, threshold: float = 0.2) -> int:
    joint, sign = split_family(family)
    if joint is None:
        return 0
    value = f(row, f"action_{JOINT_TO_DIM[joint]}") * sign
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def stage_score(row: dict[str, str]) -> float:
    """Dense trajectory score used only for explanation auditing.

    The scale is intentionally simple and bounded by tanh downstream. It rewards
    forward movement and recoverable prone/upright structure, while penalizing
    rear traps, backward speed, and rotation risk.
    """
    return (
        1.0 * f(row, "rootx")
        + 0.5 * f(row, "vx")
        + 0.4 * f(row, "grounded_prone")
        + 0.8 * max(f(row, "grounded_upright"), f(row, "grounded_prepared"))
        + 0.3 * f(row, "full_foot_support")
        - 0.7 * f(row, "grounded_rear")
        - 0.5 * f(row, "backward_speed")
        - 0.4 * f(row, "rotation_penalty")
    )


def transition_score(start: dict[str, str], end: dict[str, str]) -> float:
    start_phase = phase(start)
    end_phase = phase(end)
    score = 0.0
    if start_phase == "rear":
        score += 1.0 if end_phase in {"prone", "upright"} else 0.0
        score -= 1.0 if end_phase == "rear" else 0.0
    elif start_phase == "prone":
        score += 1.0 if f(end, "vx") > 0.2 else 0.0
        score += 0.5 if end_phase == "upright" else 0.0
        score -= 0.5 if end_phase == "rear" else 0.0
    elif start_phase == "falling":
        score += 1.0 if end_phase in {"prone", "upright"} else 0.0
        score -= 1.0 if end_phase == "rear" else 0.0
    else:
        score += 0.5 if f(end, "vx") > f(start, "vx") else 0.0
    score += clamp(f(end, "rootx") - f(start, "rootx"), -1.0, 1.0)
    score -= 0.5 * f(end, "backward_speed")
    return score


def run_dir(cycle: int, condition: str) -> Path:
    step = 145028 if cycle == 29 else 150029
    step_dir = 145029 if cycle == 29 else 150030
    return (
        LOG_ROOT
        / f"tdmpc2_v71a_10ep_sweep_cycle{cycle:03d}_step{step}_{condition}_seed1_20260506"
        / "official_eval"
        / f"step_{step_dir:07d}"
    )


def load_eval_traces(cycles: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cycle in cycles:
        for condition in ("rule_off", "rule_on"):
            for ep in read_csv(run_dir(cycle, condition) / "eval_episode_summary.csv"):
                trace_rows = read_csv(Path(ep["trace_csv"]))
                records.append(
                    {
                        "cycle": cycle,
                        "condition": condition,
                        "episode": int(float(ep["episode_idx"])),
                        "reward": f(ep, "episode_reward"),
                        "trace_rows": trace_rows,
                    }
                )
    return records


def load_rule_pool(cycles: list[int]) -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}

    for cycle in cycles + [30]:
        for rel in [
            f"cycle_{cycle:03d}/rules/library_closed_loop_cycle_{cycle:03d}.json",
            f"cycle_{cycle:03d}/rules/library_v5.json",
            f"cycle_{cycle:03d}/rules/substate_rules_v4.json",
        ]:
            path = RUN_ROOT / rel
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for rule in payload.get("rules", []):
                if rule.get("intervention_type") != "soft_prefer_action_family":
                    continue
                substate = str(rule.get("substate_key", ""))
                family = str(rule.get("action_family", ""))
                if not substate or "_" not in family:
                    continue
                rid = f"{substate}->prefer:{family}"
                existing = rules.get(rid, {})
                merged = dict(existing)
                merged.update(rule)
                merged["id"] = rid
                merged.setdefault("sources_seen", [])
                merged["sources_seen"] = sorted(set(existing.get("sources_seen", [])) | {rel})
                rules[rid] = merged

    for cycle in cycles + [30]:
        path = RUN_ROOT / f"cycle_{cycle:03d}/attributions/counterfactuals_explainer.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for rec in payload.get("records", []):
            family = action_family_from_name(rec.get("best_action", ""))
            trace_csv = Path(rec["trace_csv"])
            trace_rows = read_csv(trace_csv)
            target_step = int(rec["step"])
            row = next((r for r in trace_rows if int(f(r, "step")) == target_step), trace_rows[min(target_step, len(trace_rows) - 1)])
            substate = substate_key_for_event(row, str(rec["event"]))
            rid = f"{substate}->prefer:{family}"
            rules.setdefault(
                rid,
                {
                    "id": rid,
                    "substate_key": substate,
                    "state_event": str(rec["event"]),
                    "action_family": family,
                    "intervention_type": "soft_prefer_action_family",
                    "status": "diagnostic_group",
                    "source": "counterfactual_diagnostics",
                    "sources_seen": [],
                },
            )
            rules[rid].setdefault("counterfactual_records", []).append(rec)
    return rules


def counterfactual_stats(rule: dict[str, Any]) -> dict[str, float]:
    records = list(rule.get("counterfactual_records", []))
    if not records:
        return {
            "cf_count": 0.0,
            "cf_value_gain_raw": float(rule.get("mean_advantage", 0.0) or 0.0),
            "cf_value_gain": squash(float(rule.get("mean_advantage", 0.0) or 0.0), 0.20),
            "basin_coverage": 0.0,
        }
    advantages = [float(rec.get("advantage_over_actual", 0.0) or 0.0) for rec in records]
    return {
        "cf_count": float(len(records)),
        "cf_value_gain_raw": mean(advantages),
        "cf_value_gain": squash(mean(advantages), 0.20),
        "basin_coverage": sum(1 for x in advantages if x > 0.0) / len(advantages),
    }


def observed_stats(rule: dict[str, Any], traces: list[dict[str, Any]], horizon: int) -> dict[str, float]:
    substate = str(rule["substate_key"])
    family = str(rule["action_family"])
    aligned_outcomes: list[float] = []
    opposed_outcomes: list[float] = []
    aligned_transitions: list[float] = []
    opposed_transitions: list[float] = []
    episode_features: list[tuple[float, float]] = []
    high_align: list[float] = []
    low_align: list[float] = []
    total_hits = 0

    for ep in traces:
        rows = ep["trace_rows"]
        ep_aligned = 0
        ep_opposed = 0
        ep_hits = 0
        for idx, row in enumerate(rows):
            if substate not in event_keys(row):
                continue
            align = action_alignment(row, family)
            if align == 0:
                continue
            end = rows[min(idx + horizon, len(rows) - 1)]
            outcome = stage_score(end) - stage_score(row)
            trans = transition_score(row, end)
            ep_hits += 1
            total_hits += 1
            if align > 0:
                ep_aligned += 1
                aligned_outcomes.append(outcome)
                aligned_transitions.append(trans)
            else:
                ep_opposed += 1
                opposed_outcomes.append(outcome)
                opposed_transitions.append(trans)
        if ep_hits:
            feature = (ep_aligned - ep_opposed) / ep_hits
            episode_features.append((feature, float(ep["reward"])))
            if float(ep["reward"]) >= 100.0:
                high_align.append(feature)
            if float(ep["reward"]) <= 1.0:
                low_align.append(feature)

    raw_sep = mean(aligned_outcomes) - mean(opposed_outcomes)
    raw_trans = mean(aligned_transitions) - mean(opposed_transitions)
    label_power_raw = mean(high_align) - mean(low_align)

    rewards = [reward for _, reward in episode_features]
    features = [feature for feature, _ in episode_features]
    if len(episode_features) >= 3 and len(set(features)) > 1:
        f_mean = mean(features)
        r_mean = mean(rewards)
        cov = mean([(x - f_mean) * (y - r_mean) for x, y in episode_features])
        var_f = mean([(x - f_mean) ** 2 for x in features])
        var_r = mean([(y - r_mean) ** 2 for y in rewards])
        corr = cov / math.sqrt(var_f * var_r) if var_f > 0 and var_r > 0 else 0.0
        residual_gain = corr * corr
    else:
        residual_gain = 0.0

    return {
        "observed_hit_steps": float(total_hits),
        "aligned_count": float(len(aligned_outcomes)),
        "opposed_count": float(len(opposed_outcomes)),
        "past_separation_raw": raw_sep,
        "past_separation": squash(raw_sep, 1.0),
        "transition_gain_raw": raw_trans,
        "transition_gain": squash(raw_trans, 1.0),
        "past_label_power_raw": label_power_raw,
        "past_label_power": clamp(label_power_raw, -1.0, 1.0),
        "past_residual_gain": residual_gain,
    }


def conflict_group(rule: dict[str, Any]) -> str:
    family = str(rule.get("action_family", ""))
    joint = family.split("_", 1)[0] if "_" in family else family
    return f"{rule.get('substate_key')}|joint:{joint}"


def complexity_cost(rule: dict[str, Any]) -> float:
    tokens = str(rule.get("substate_key", "")).split("|")
    specified = sum(1 for token in tokens if token)
    return min(1.0, specified / 5.0)


def score_rules(rules: dict[str, dict[str, Any]], traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preliminary: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for rid, rule in rules.items():
        obs = observed_stats(rule, traces, horizon=50)
        cf = counterfactual_stats(rule)
        base = (
            WEIGHTS["past_separation"] * obs["past_separation"]
            + WEIGHTS["past_label_power"] * obs["past_label_power"]
            + WEIGHTS["cf_value_gain"] * cf["cf_value_gain"]
            + WEIGHTS["transition_gain"] * obs["transition_gain"]
            + WEIGHTS["basin_coverage"] * cf["basin_coverage"]
        )
        preliminary[rid] = base
        row = dict(rule)
        row.update(obs)
        row.update(cf)
        row["complexity_cost"] = complexity_cost(rule)
        row["_base_score"] = base
        rows.append(row)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[conflict_group(row)].append(row)

    scored = []
    for row in rows:
        family = str(row.get("action_family", ""))
        opposite = None
        if family.endswith("_pos"):
            opposite = family[:-4] + "_neg"
        elif family.endswith("_neg"):
            opposite = family[:-4] + "_pos"
        conflict = 0.0
        if opposite:
            for other in by_group[conflict_group(row)]:
                if other.get("action_family") == opposite:
                    conflict = max(conflict, max(0.0, float(other.get("_base_score", 0.0))))
        explainability = (
            row["_base_score"]
            + WEIGHTS["conflict_cost"] * conflict
            + WEIGHTS["complexity_cost"] * row["complexity_cost"]
        )
        out = {
            "id": row["id"],
            "substate_key": row.get("substate_key"),
            "action_family": row.get("action_family"),
            "status": row.get("status"),
            "source": row.get("source"),
            "sources_seen": row.get("sources_seen", []),
            "support": row.get("support", 0),
            "good_share": row.get("good_share", None),
            "confidence": row.get("confidence", None),
            "explainability_v0": explainability,
            "components": {
                "past_separation": row["past_separation"],
                "past_label_power": row["past_label_power"],
                "past_residual_gain": row["past_residual_gain"],
                "cf_value_gain": row["cf_value_gain"],
                "transition_gain": row["transition_gain"],
                "basin_coverage": row["basin_coverage"],
                "conflict_cost": conflict,
                "complexity_cost": row["complexity_cost"],
            },
            "raw": {
                "observed_hit_steps": row["observed_hit_steps"],
                "aligned_count": row["aligned_count"],
                "opposed_count": row["opposed_count"],
                "past_separation_raw": row["past_separation_raw"],
                "transition_gain_raw": row["transition_gain_raw"],
                "past_label_power_raw": row["past_label_power_raw"],
                "cf_count": row["cf_count"],
                "cf_value_gain_raw": row["cf_value_gain_raw"],
            },
        }
        scored.append(out)
    scored.sort(key=lambda x: x["explainability_v0"], reverse=True)
    return scored


def write_md(scored: list[dict[str, Any]], path: Path) -> None:
    active = [r for r in scored if r.get("status") == "confidence_active"]
    non_active = [r for r in scored if r.get("status") != "confidence_active"]
    lines = [
        "# Hopper v7.1a Explainability v0 Audit",
        "",
        "该报告是离线审计，不改变训练和在线 prior。分数用于比较规则是否能解释高低回报差异，并是否有 SD 反事实支持。",
        "",
        "## Formula",
        "",
        "`Expl_v0 = 0.35*past_separation + 0.15*past_label_power + 0.25*cf_value_gain + 0.15*transition_gain + 0.10*basin_coverage - 0.10*conflict_cost - 0.05*complexity_cost`",
        "",
        "## Top Rules",
        "",
        "| rank | status | rule | expl | past | label | cf | trans | basin | conflict | hits | cf_n |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(scored[:20], start=1):
        comp = row["components"]
        raw = row["raw"]
        lines.append(
            f"| {idx} | {row.get('status')} | `{row['id']}` | {row['explainability_v0']:.3f} | "
            f"{comp['past_separation']:.3f} | {comp['past_label_power']:.3f} | {comp['cf_value_gain']:.3f} | "
            f"{comp['transition_gain']:.3f} | {comp['basin_coverage']:.3f} | {comp['conflict_cost']:.3f} | "
            f"{raw['observed_hit_steps']:.0f} | {raw['cf_count']:.0f} |"
        )
    lines += ["", "## Active Rules", "", "| rule | expl | rank | past | cf | hits |", "|---|---:|---:|---:|---:|---:|"]
    rank = {row["id"]: idx for idx, row in enumerate(scored, start=1)}
    for row in sorted(active, key=lambda x: x["explainability_v0"], reverse=True):
        lines.append(
            f"| `{row['id']}` | {row['explainability_v0']:.3f} | {rank[row['id']]} | "
            f"{row['components']['past_separation']:.3f} | {row['components']['cf_value_gain']:.3f} | "
            f"{row['raw']['observed_hit_steps']:.0f} |"
        )
    lines += ["", "## High Explainability Non-Active Rules", "", "| rule | status | expl | past | label | cf | trans | hits | cf_n |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in non_active[:15]:
        comp = row["components"]
        raw = row["raw"]
        lines.append(
            f"| `{row['id']}` | {row.get('status')} | {row['explainability_v0']:.3f} | "
            f"{comp['past_separation']:.3f} | {comp['past_label_power']:.3f} | {comp['cf_value_gain']:.3f} | "
            f"{comp['transition_gain']:.3f} | {raw['observed_hit_steps']:.0f} | {raw['cf_count']:.0f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `past_separation` 使用已发生 trace 中规则对齐动作与反向动作后的 50-step 阶段结果差异。",
        "- `past_label_power` 衡量高回报 episode 与低回报 episode 中规则对齐程度的差异。",
        "- `cf_value_gain` 与 `basin_coverage` 来自 SD counterfactual 记录。",
        "- 该版本是解释力初稿，主要用于排序和诊断，不作为上线阈值。",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cycles = [29, 30]
    traces = load_eval_traces(cycles)
    rule_pool = load_rule_pool(cycles)
    scored = score_rules(rule_pool, traces)
    payload = {
        "formula": WEIGHTS,
        "cycles": cycles,
        "num_rules_scored": len(scored),
        "rules": scored,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(scored, REPORT_MD)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
