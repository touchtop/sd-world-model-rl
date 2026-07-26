#!/usr/bin/env python3
"""Audit why closed-loop cycles did not add new rules."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ROOT = Path("results/sd_retrospective_rule_induction/closed_loop/hopper_v7_10k5_seed1_20260505")
DEFAULT_REPORT = Path("results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_rule_funnel_audit_20260506.md")
DEFAULT_JSON = Path("results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_rule_funnel_audit_20260506.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def action_family(name: str) -> str:
    if name in {"actual", "zero"}:
        return name
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def substate_key(record: dict[str, Any], row: dict[str, str]) -> str:
    event = record["event"]
    pitch_rate = f(row, "pitch_rate")
    vx = f(row, "vx")
    full_foot = f(row, "full_foot_support")
    falling = f(row, "falling_air") >= 0.5
    rear = f(row, "grounded_rear") >= 0.5
    prone = f(row, "grounded_prone") >= 0.5
    upright = f(row, "grounded_upright") >= 0.5 or f(row, "grounded_prepared") >= 0.5

    omega_bin = "omega_pos" if pitch_rate > 1.0 else "omega_neg" if pitch_rate < -1.0 else "omega_mid"
    vx_bin = "vx_high" if vx >= 0.8 else "vx_low" if vx < 0.2 else "vx_mid"
    support_bin = "support_none" if full_foot < 0.1 else "support_low" if full_foot < 0.35 else "support_some"
    phase = "falling" if falling else "rear" if rear else "prone" if prone else "upright" if upright else "other"
    return f"{event}|{omega_bin}|{vx_bin}|{support_bin}|{phase}"


def row_for_record(record: dict[str, Any], cache: dict[tuple[str, int], dict[str, str]]) -> dict[str, str]:
    trace_csv = str(record["trace_csv"])
    step = int(record["step"])
    key = (trace_csv, step)
    if key in cache:
        return cache[key]
    rows = read_csv(Path(trace_csv))
    for row in rows:
        if int(f(row, "step")) == step:
            cache[key] = row
            return row
    row = rows[min(max(step, 0), len(rows) - 1)]
    cache[key] = row
    return row


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summarize_v2_groups(records: list[dict[str, Any]], min_support: int, min_advantage: float) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec["event"], action_family(rec["best_action"]))].append(rec)
    rows = []
    passing = []
    for (event, family), vals in groups.items():
        support = len(vals)
        mean_adv = mean_or_zero([float(v["advantage_over_actual"]) for v in vals])
        row = {
            "event": event,
            "action_family": family,
            "support": support,
            "mean_advantage": mean_adv,
            "labels": dict(Counter(str(v["label"]) for v in vals)),
        }
        rows.append(row)
        if support >= min_support and mean_adv >= min_advantage:
            passing.append(row)
    rows.sort(key=lambda r: (r["support"], r["mean_advantage"]), reverse=True)
    return {
        "num_groups": len(rows),
        "num_passing_groups": len(passing),
        "max_support": rows[0]["support"] if rows else 0,
        "max_mean_advantage": max((r["mean_advantage"] for r in rows), default=0.0),
        "top_groups": rows[:5],
        "blocker": classify_v2_blocker(rows, min_support, min_advantage),
    }


def classify_v2_blocker(rows: list[dict[str, Any]], min_support: int, min_advantage: float) -> str:
    if not rows:
        return "no_counterfactual_groups"
    if max(r["support"] for r in rows) < min_support:
        return "support_below_min_support"
    if max(r["mean_advantage"] for r in rows) < min_advantage:
        return "advantage_below_min_advantage"
    return "mixed_or_no_single_group_passed"


def summarize_v4_groups(
    records: list[dict[str, Any]],
    min_support: int,
    min_advantage: float,
    max_good_share: float,
) -> dict[str, Any]:
    cache: dict[tuple[str, int], dict[str, str]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    substate_counts: Counter[str] = Counter()
    for rec in records:
        if rec["event"] not in {"unsupported_high_torque", "rotation_risk"}:
            continue
        key = substate_key(rec, row_for_record(rec, cache))
        family = action_family(rec["best_action"])
        groups[(key, family)].append(rec)
        substate_counts[key] += 1

    rows = []
    passing = []
    for (key, family), vals in groups.items():
        support = len(vals)
        labels = Counter(str(v["label"]) for v in vals)
        good_share = labels.get("good", 0) / support if support else 0.0
        mean_adv = mean_or_zero([float(v["advantage_over_actual"]) for v in vals])
        row = {
            "substate_key": key,
            "action_family": family,
            "support": support,
            "substate_support": substate_counts[key],
            "mean_advantage": mean_adv,
            "good_share": good_share,
            "labels": dict(labels),
        }
        rows.append(row)
        if support >= min_support and mean_adv >= min_advantage and good_share <= max_good_share:
            passing.append(row)
    rows.sort(key=lambda r: (r["support"], r["mean_advantage"]), reverse=True)
    return {
        "num_substates": len(substate_counts),
        "num_substate_action_groups": len(rows),
        "num_passing_groups": len(passing),
        "max_substate_support": max(substate_counts.values(), default=0),
        "max_action_support": rows[0]["support"] if rows else 0,
        "max_mean_advantage": max((r["mean_advantage"] for r in rows), default=0.0),
        "top_groups": rows[:8],
        "blocker": classify_v4_blocker(rows, min_support, min_advantage, max_good_share),
    }


def classify_v4_blocker(
    rows: list[dict[str, Any]],
    min_support: int,
    min_advantage: float,
    max_good_share: float,
) -> str:
    if not rows:
        return "no_substate_groups"
    support_ok = [r for r in rows if r["support"] >= min_support]
    if not support_ok:
        return "substate_action_support_below_min_support"
    advantage_ok = [r for r in support_ok if r["mean_advantage"] >= min_advantage]
    if not advantage_ok:
        return "advantage_below_min_advantage_after_support"
    good_ok = [r for r in advantage_ok if r["good_share"] <= max_good_share]
    if not good_ok:
        return "good_share_above_max_good_share"
    return "unexpected_no_rule_despite_passing_group"


def audit_cycle(root: Path, cycle: int) -> dict[str, Any]:
    croot = root / f"cycle_{cycle:03d}"
    selected = load_json(croot / "manifests" / "selected_trajectories.json")
    counterfactuals = load_json(croot / "attributions" / "counterfactuals_explainer.json")
    state_rules = load_json(croot / "rules" / "state_action_rules_v2.json")
    consistency = load_json(croot / "reports" / "rule_consistency_v3.json")
    substate_rules = load_json(croot / "rules" / "substate_rules_v4.json")
    validation = load_json(croot / "reports" / "substate_rule_validation_v5.json")
    library_v5 = load_json(croot / "rules" / "library_v5.json")
    closed_loop = load_json(croot / "rules" / f"library_closed_loop_cycle_{cycle:03d}.json")

    selected_eps = selected.get("selected_episodes", [])
    records = counterfactuals.get("records", [])
    seeds = {str(ep.get("seed")) for ep in selected_eps}
    labels = Counter(str(ep.get("label")) for ep in selected_eps)
    record_labels = Counter(str(rec.get("label")) for rec in records)
    event_counts = Counter(str(rec.get("event")) for rec in records)
    event_adv = {
        event: mean_or_zero([float(rec["advantage_over_actual"]) for rec in records if rec.get("event") == event])
        for event in sorted(event_counts)
    }

    return {
        "cycle": cycle,
        "selected": {
            "input_episodes": selected.get("num_input_episodes"),
            "selected_episodes": selected.get("num_selected_episodes"),
            "labels": dict(labels),
            "seeds_covered": len(seeds),
            "cycle_steps": sorted(selected.get("cycle_stats", {}).keys()),
        },
        "counterfactuals": {
            "num_records": counterfactuals.get("num_state_records"),
            "labels": dict(record_labels),
            "event_counts": dict(event_counts),
            "event_mean_advantage": event_adv,
            "aggregate": counterfactuals.get("aggregate", {}),
        },
        "v2_state_action": {
            "configured_min_support": state_rules.get("min_support"),
            "configured_min_advantage": state_rules.get("min_advantage"),
            "generated_rules": len(state_rules.get("rules", [])),
            "groups": summarize_v2_groups(
                records,
                int(state_rules.get("min_support", 8)),
                float(state_rules.get("min_advantage", 0.10)),
            ),
        },
        "v3_consistency": {
            "source_rules": consistency.get("num_rules"),
            "audits": len(consistency.get("audits", [])),
            "min_seeds": consistency.get("min_seeds"),
            "reliable_rules": load_json(croot / "rules" / "reliable_rules_v3.json").get("num_reliable_rules"),
        },
        "v4_substate": {
            "configured_min_support": substate_rules.get("min_support"),
            "configured_min_advantage": substate_rules.get("min_advantage"),
            "configured_max_good_share": substate_rules.get("max_good_share"),
            "generated_rules": substate_rules.get("num_rules"),
            "groups": summarize_v4_groups(
                records,
                int(substate_rules.get("min_support", 5)),
                float(substate_rules.get("min_advantage", 0.12)),
                float(substate_rules.get("max_good_share", 0.25)),
            ),
        },
        "v5_validation": {
            "validated_rule_entries": len(validation.get("rules", {})),
            "library_v5_rules": len(library_v5.get("rules", [])),
            "library_v5_statuses": dict(Counter(str(rule.get("status")) for rule in library_v5.get("rules", []))),
            "single_seed_replay_validated_blocker": len(seeds) < 2,
        },
        "closed_loop_update": closed_loop.get("closed_loop_update", {}),
    }


def build_payload(root: Path) -> dict[str, Any]:
    cycles = []
    for croot in sorted(root.glob("cycle_[0-9][0-9][0-9]")):
        cycles.append(audit_cycle(root, int(croot.name.split("_")[1])))
    return {
        "root": str(root),
        "num_cycles": len(cycles),
        "cycles": cycles,
        "summary": summarize_all(cycles),
    }


def summarize_all(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cycle_rules_generated": sum(int(c["v4_substate"]["generated_rules"] or 0) for c in cycles),
        "cycle_library_rules_generated": sum(int(c["v5_validation"]["library_v5_rules"] or 0) for c in cycles),
        "closed_loop_rules_added": sum(
            1 for c in cycles if c["closed_loop_update"].get("action") in {"added_new_rule", "updated_existing_rule"}
        ),
        "cycles_with_no_good_selected_episode": [
            c["cycle"] for c in cycles if int(c["selected"]["labels"].get("good", 0)) == 0
        ],
        "cycles_with_single_seed_only": [
            c["cycle"] for c in cycles if int(c["selected"]["seeds_covered"]) < 2
        ],
        "v2_blockers": dict(Counter(c["v2_state_action"]["groups"]["blocker"] for c in cycles)),
        "v4_blockers": dict(Counter(c["v4_substate"]["groups"]["blocker"] for c in cycles)),
    }


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.0000"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper v7 规则未新增漏斗审计",
        "",
        "整理日期：2026-05-06",
        "",
        "## 结论",
        "",
        "本轮 `10k x 5` 闭环没有新增可上线规则，主要瓶颈不在闭环更新器，而在周期规则证据不足：周期候选层已能保留近阈值规则，但这些规则仍未达到 `replay_validated` 上线条件。",
        "",
        f"当前周期候选层共保留 `{payload['summary']['cycle_library_rules_generated']}` 条周期候选规则，闭环累计规则库中仍新增 `0` 条 `replay_validated` 上线规则。直接原因有三项：第一，周期总结使用的官方评估轨迹太少，第 2 至第 5 周期每轮只有 3 条选中 episode、9 个反事实状态记录；第二，第 2 至第 5 周期没有选中 good 轨迹，缺少好坏对比；第三，当前 per-cycle 运行只有 seed 1，而 v3/v5 的可靠性准入仍保留跨 seed 要求，周期候选难以直接升级为 `replay_validated`。",
        "",
        "## 漏斗总览",
        "",
        "| 周期 | selected episodes | good/bad | records | v2 rules | v4 rules | cycle library v5 rules | closed-loop action |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in payload["cycles"]:
        labels = c["selected"]["labels"]
        lines.append(
            f"| {c['cycle']} | {c['selected']['selected_episodes']} | "
            f"{labels.get('good', 0)}/{labels.get('bad', 0)} | "
            f"{c['counterfactuals']['num_records']} | "
            f"{c['v2_state_action']['generated_rules']} | "
            f"{c['v4_substate']['generated_rules']} | "
            f"{c['v5_validation']['library_v5_rules']} | "
            f"`{c['closed_loop_update'].get('action')}` |"
        )

    lines += [
        "",
        "## 规则生成卡点",
        "",
        "| 周期 | v2 主要卡点 | v2 max support | v2 max advantage | v4 主要卡点 | v4 max action support | v4 max advantage |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for c in payload["cycles"]:
        v2 = c["v2_state_action"]["groups"]
        v4 = c["v4_substate"]["groups"]
        lines.append(
            f"| {c['cycle']} | `{v2['blocker']}` | {v2['max_support']} | {fmt_float(v2['max_mean_advantage'])} | "
            f"`{v4['blocker']}` | {v4['max_action_support']} | {fmt_float(v4['max_mean_advantage'])} |"
        )

    lines += [
        "",
        "解释：v2 粗事件规则要求同一 `event -> action_family` 至少 `8` 个支持样本且平均优势不少于 `0.10`；v4 子状态上线支持要求同一 `substate -> action_family` 至少 `5` 个支持样本、平均优势不少于 `0.12` 且 good share 不高于 `0.25`。本轮所有周期均未达到上线支持条件，但周期候选层会保留支持数不少于 `2`、平均优势不少于 `0.02` 且 good share 不高于 `0.50` 的近阈值规则。",
        "",
        "## 逐周期近邻候选",
        "",
    ]
    for c in payload["cycles"]:
        lines += [
            f"### Cycle {c['cycle']:03d}",
            "",
            f"- 选中轨迹：`{c['selected']['selected_episodes']}`；标签分布：`{c['selected']['labels']}`；seed 覆盖：`{c['selected']['seeds_covered']}`。",
            f"- 反事实记录：`{c['counterfactuals']['num_records']}`；事件分布：`{c['counterfactuals']['event_counts']}`。",
            f"- v2 卡点：`{c['v2_state_action']['groups']['blocker']}`。",
            f"- v4 卡点：`{c['v4_substate']['groups']['blocker']}`。",
            "",
            "v4 最接近候选：",
            "",
            "| substate | action | support | mean_adv | good_share | labels |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
        for row in c["v4_substate"]["groups"]["top_groups"][:4]:
            lines.append(
                f"| `{row['substate_key']}` | `{row['action_family']}` | {row['support']} | "
                f"{fmt_float(row['mean_advantage'])} | {fmt_float(row['good_share'])} | `{row['labels']}` |"
            )
        lines.append("")

    lines += [
        "## 方法含义",
        "",
        "当前闭环的失败点不是动作先验没有加载，也不是 `closed_loop_rule_update.py` 没有选择规则，而是周期总结只产生了弱周期候选，尚未产出可上线的 `replay_validated` 规则。第 2 至第 5 周期使用的评估轨迹全部为 bad episode，这可以定位失败形态，但不足以支持“哪些决策相对更好”的规则抽取。与此同时，单 seed 周期总结与跨 seed 可靠性阈值存在结构性不匹配。",
        "",
        "下一步应先修正审计发现的证据供给问题：周期总结应纳入训练 episode 中的非零回报片段或阶段诊断较好的片段，增加 good/near-good 对照；同时把单周期准入与跨周期/跨 seed 准入分离，允许周期内形成 `cycle_candidate`，再由累计证据升级为 `replay_validated`。在此之前，不宜通过降低最终上线阈值或增强既有动作先验来制造表面收益。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.root)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(f"wrote audit json: {args.output_json}")
    print(f"wrote audit report: {args.output_md}")


if __name__ == "__main__":
    main()
