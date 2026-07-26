#!/usr/bin/env python3
"""Audit whether the v7.1a Hopper rules cover key behavioral states.

This script is intentionally eval-only. It reuses saved 10-episode rule-on/off
traces and asks whether the active confidence rules cover the state regions
that separate high-return and low-return episodes.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs/hopper-hop/1"
RESULT_ROOT = REPO_ROOT / "results/sd_retrospective_rule_induction"
REPORT_JSON = RESULT_ROOT / "reports/hopper_v71a_rule_keyness_audit_cycle029_030_seed1_20260506.json"
REPORT_MD = RESULT_ROOT / "reports/hopper_v71a_rule_keyness_audit_cycle029_030_seed1_20260506.md"


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


def phase(row: dict[str, str]) -> str:
    if f(row, "falling_air") >= 0.5 or f(row, "falling_air_gt_05") >= 0.5:
        return "falling"
    if f(row, "grounded_rear") >= 0.5 or f(row, "grounded_rear_gt_05") >= 0.5:
        return "rear"
    if f(row, "grounded_hop") >= 0.5 or f(row, "grounded_hop_gt_05") >= 0.5:
        return "hop"
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


def substate_parts(row: dict[str, str]) -> dict[str, str]:
    pitch_rate = f(row, "pitch_rate")
    vx = f(row, "vx")
    support = f(row, "full_foot_support")
    return {
        "omega_bin": "omega_pos" if pitch_rate > 1.0 else "omega_neg" if pitch_rate < -1.0 else "omega_mid",
        "vx_bin": "vx_high" if vx >= 0.8 else "vx_low" if vx < 0.2 else "vx_mid",
        "support_bin": "support_none" if support < 0.1 else "support_low" if support < 0.35 else "support_some",
        "phase": phase(row),
        "rotation_risk": "1" if (f(row, "rotation_penalty") >= 0.5 or abs(pitch_rate) >= 3.0) else "0",
    }


def event_keys(row: dict[str, str]) -> list[str]:
    parts = substate_parts(row)
    base = f"{parts['omega_bin']}|{parts['vx_bin']}|{parts['support_bin']}|{parts['phase']}"
    keys = [f"unsupported_high_torque|{base}"]
    if parts["rotation_risk"] == "1":
        keys.append(f"rotation_risk|{base}")
    return keys


def run_dir(cycle: int, condition: str) -> Path:
    step = 145028 if cycle == 29 else 150029
    step_dir = 145029 if cycle == 29 else 150030
    return (
        LOG_ROOT
        / f"tdmpc2_v71a_10ep_sweep_cycle{cycle:03d}_step{step}_{condition}_seed1_20260506"
        / "official_eval"
        / f"step_{step_dir:07d}"
    )


def load_active_rules(cycle: int) -> list[dict[str, Any]]:
    path = (
        RESULT_ROOT
        / f"closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_{cycle:03d}"
        / f"rules/library_closed_loop_cycle_{cycle:03d}.json"
    )
    payload = json.loads(path.read_text())
    return [
        rule
        for rule in payload.get("rules", [])
        if rule.get("status") == "confidence_active"
        and rule.get("source") == "v4_substate"
        and rule.get("intervention_type") == "soft_prefer_action_family"
    ]


def load_trace_records(cycles: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cycle in cycles:
        for condition in ("rule_off", "rule_on"):
            summary_path = run_dir(cycle, condition) / "eval_episode_summary.csv"
            for ep in read_csv(summary_path):
                episode = int(float(ep["episode_idx"]))
                reward = f(ep, "episode_reward")
                rows = read_csv(Path(ep["trace_csv"]))
                for row in rows:
                    records.append(
                        {
                            "cycle": cycle,
                            "condition": condition,
                            "episode": episode,
                            "episode_reward": reward,
                            "step": int(float(row["step"])),
                            "row": row,
                        }
                    )
    return records


def summarize_active_rules(rules: list[dict[str, Any]]) -> dict[str, Any]:
    substates = sorted({str(rule.get("substate_key")) for rule in rules})
    phases = sorted({key.split("|")[-1] for key in substates})
    events = sorted({key.split("|")[0] for key in substates})
    action_families = sorted({str(rule.get("action_family")) for rule in rules})
    joints = sorted({family.split("_", 1)[0] for family in action_families if "_" in family})
    return {
        "num_active_rules": len(rules),
        "num_unique_substates": len(substates),
        "substates": substates,
        "events": events,
        "phases": phases,
        "action_families": action_families,
        "joints": joints,
    }


def coverage(records: list[dict[str, Any]], active_substates: set[str]) -> dict[str, Any]:
    total = len(records)
    matched = 0
    phase_counts: Counter[str] = Counter()
    phase_matched: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    for rec in records:
        row = rec["row"]
        ph = phase(row)
        phase_counts[ph] += 1
        keys = event_keys(row)
        for key in keys:
            key_counts[key] += 1
        if any(key in active_substates for key in keys):
            matched += 1
            phase_matched[ph] += 1
    return {
        "total_steps": total,
        "matched_steps": matched,
        "matched_frac": matched / total if total else 0.0,
        "phase_frac": {key: value / total for key, value in sorted(phase_counts.items())},
        "phase_matched_frac_within_phase": {
            key: phase_matched[key] / value if value else 0.0 for key, value in sorted(phase_counts.items())
        },
        "top_uncovered_keys": [
            {"substate_key": key, "steps": count, "frac": count / total}
            for key, count in key_counts.most_common()
            if key not in active_substates
        ][:12],
    }


def episode_level(cycles: list[int], active_substates: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        for condition in ("rule_off", "rule_on"):
            for ep in read_csv(run_dir(cycle, condition) / "eval_episode_summary.csv"):
                trace_rows = read_csv(Path(ep["trace_csv"]))
                matched = sum(1 for row in trace_rows if any(key in active_substates for key in event_keys(row)))
                ph_counts = Counter(phase(row) for row in trace_rows)
                rows.append(
                    {
                        "cycle": cycle,
                        "condition": condition,
                        "episode": int(float(ep["episode_idx"])),
                        "reward": f(ep, "episode_reward"),
                        "rule_covered_steps": matched,
                        "rule_covered_frac": matched / len(trace_rows),
                        "rear_frac": ph_counts["rear"] / len(trace_rows),
                        "prone_frac": ph_counts["prone"] / len(trace_rows),
                        "falling_frac": ph_counts["falling"] / len(trace_rows),
                        "final_rootx": f(ep, "final_rootx"),
                        "max_rootx": f(ep, "max_rootx"),
                    }
                )
    return rows


def contrast_groups(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    high = [ep for ep in episodes if ep["reward"] >= 100.0]
    low = [ep for ep in episodes if ep["reward"] <= 1.0]
    mid = [ep for ep in episodes if 1.0 < ep["reward"] < 100.0]
    def pack(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "n": float(len(group)),
            "reward_mean": mean([ep["reward"] for ep in group]),
            "rule_covered_frac_mean": mean([ep["rule_covered_frac"] for ep in group]),
            "rear_frac_mean": mean([ep["rear_frac"] for ep in group]),
            "prone_frac_mean": mean([ep["prone_frac"] for ep in group]),
            "final_rootx_mean": mean([ep["final_rootx"] for ep in group]),
        }
    return {"high_reward_ge_100": pack(high), "low_reward_le_1": pack(low), "mid_reward": pack(mid)}


def large_deltas(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for ep in episodes:
        by_key[(ep["cycle"], ep["episode"])][ep["condition"]] = ep
    rows = []
    for (cycle, episode), vals in sorted(by_key.items()):
        if "rule_on" not in vals or "rule_off" not in vals:
            continue
        off, on = vals["rule_off"], vals["rule_on"]
        delta = on["reward"] - off["reward"]
        if abs(delta) >= 40.0:
            rows.append(
                {
                    "cycle": cycle,
                    "episode": episode,
                    "reward_off": off["reward"],
                    "reward_on": on["reward"],
                    "delta_on_minus_off": delta,
                    "off_rule_covered_steps": off["rule_covered_steps"],
                    "on_rule_covered_steps": on["rule_covered_steps"],
                    "off_rear_frac": off["rear_frac"],
                    "on_rear_frac": on["rear_frac"],
                    "off_prone_frac": off["prone_frac"],
                    "on_prone_frac": on["prone_frac"],
                    "off_final_rootx": off["final_rootx"],
                    "on_final_rootx": on["final_rootx"],
                }
            )
    return rows


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hopper v7.1a Rule Keyness Audit",
        "",
        "该审计复用 cycle029 与 cycle030 的 10-episode rule-on/off trace，检查 active confidence 规则是否覆盖区分高低回报的关键状态区域。",
        "",
        "## Active Rule Footprint",
        "",
    ]
    fp = payload["active_rule_footprint"]
    lines += [
        f"- Active rules: `{fp['num_active_rules']}`",
        f"- Unique substates: `{fp['num_unique_substates']}`",
        f"- Events: `{', '.join(fp['events'])}`",
        f"- Phases: `{', '.join(fp['phases'])}`",
        f"- Action families: `{', '.join(fp['action_families'])}`",
        f"- Joints: `{', '.join(fp['joints'])}`",
        "",
        "| active substate |",
        "|---|",
    ]
    for key in fp["substates"]:
        lines.append(f"| `{key}` |")
    lines += ["", "## Coverage", "", "| group | matched steps | matched frac | rear frac | prone frac | falling frac |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, cov in payload["coverage"].items():
        phase = cov["phase_frac"]
        lines.append(
            f"| {name} | {cov['matched_steps']} / {cov['total_steps']} | {cov['matched_frac']:.3f} | "
            f"{phase.get('rear', 0.0):.3f} | {phase.get('prone', 0.0):.3f} | {phase.get('falling', 0.0):.3f} |"
        )
    lines += ["", "## High vs Low Return Episodes", "", "| group | n | reward mean | rule covered | rear frac | prone frac | final rootx |"]
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, group in payload["contrast_groups"].items():
        lines.append(
            f"| {name} | {group['n']:.0f} | {group['reward_mean']:.3f} | {group['rule_covered_frac_mean']:.3f} | "
            f"{group['rear_frac_mean']:.3f} | {group['prone_frac_mean']:.3f} | {group['final_rootx_mean']:.3f} |"
        )
    lines += ["", "## Largest On/Off Divergences", "", "| cycle | ep | off | on | delta | off covered | on covered | off rear | on rear | off prone | on prone |"]
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["large_reward_deltas"]:
        lines.append(
            f"| {row['cycle']} | {row['episode']} | {row['reward_off']:.3f} | {row['reward_on']:.3f} | "
            f"{row['delta_on_minus_off']:+.3f} | {row['off_rule_covered_steps']} | {row['on_rule_covered_steps']} | "
            f"{row['off_rear_frac']:.3f} | {row['on_rear_frac']:.3f} | {row['off_prone_frac']:.3f} | {row['on_prone_frac']:.3f} |"
        )
    lines += ["", "## Top Uncovered State Keys", ""]
    for name, cov in payload["coverage"].items():
        lines.append(f"### {name}")
        lines += ["", "| substate | steps | frac |", "|---|---:|---:|"]
        for row in cov["top_uncovered_keys"][:8]:
            lines.append(f"| `{row['substate_key']}` | {row['steps']} | {row['frac']:.3f} |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "Active rules are few and concentrated: they cover only `omega_neg/vx_low/support_none` under `prone` or `falling`, and do not cover `rear`, `upright`, `vx_mid/high`, `omega_pos/mid`, hip, or ankle decisions.",
        "",
        "The largest negative rule-on divergences in cycle030 are dominated by trajectories that remain in `rear` and almost never enter active rule substates. Thus the rules can affect early action ranking, but once the trajectory branches into an uncovered rear-recovery regime, the current rule library provides little guidance.",
        "",
        "The largest positive cycle029 divergences are the opposite pattern: rule-on branches enter more active-rule-covered prone states and then obtain higher forward progress. This supports the hypothesis that the rule set is locally meaningful but not task-complete or sufficiently key-stage covering.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cycles = [29, 30]
    rules = load_active_rules(30)
    active_substates = {str(rule.get("substate_key")) for rule in rules}
    trace_records = load_trace_records(cycles)
    episodes = episode_level(cycles, active_substates)
    coverage_groups = {"all_steps": coverage(trace_records, active_substates)}
    for cycle in cycles:
        coverage_groups[f"cycle{cycle:03d}"] = coverage([r for r in trace_records if r["cycle"] == cycle], active_substates)
    payload = {
        "active_rule_footprint": summarize_active_rules(rules),
        "coverage": coverage_groups,
        "contrast_groups": contrast_groups(episodes),
        "large_reward_deltas": large_deltas(episodes),
        "episodes": episodes,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(payload, REPORT_MD)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
