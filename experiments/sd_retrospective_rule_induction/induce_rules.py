#!/usr/bin/env python3
"""Induce a first decision-list style rule tree from Hopper attributions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DEFAULT_ATTRIBUTIONS, DEFAULT_RULES, load_json, quantile, safe_rate, write_json


RISK_FEATURES = (
    "unsupported_high_torque_frac",
    "landing_unprepared_frac",
    "rotation_risk_frac",
    "prone_or_rear_trap_frac",
    "backward_speed_frac",
    "rotation_penalty_frac",
)

SUCCESS_FEATURES = (
    "prepared_support_frac",
    "effective_drive_frac",
    "mean_full_foot_support",
    "mean_grounded_upright",
    "mean_vx",
    "max_rootx",
)


def feature_values(episodes: list[dict[str, Any]], feature: str) -> list[float]:
    return [float(ep["features"].get(feature, 0.0)) for ep in episodes]


def rate_for_condition(
    episodes: list[dict[str, Any]],
    feature: str,
    threshold: float,
    direction: str,
    label: str,
) -> tuple[float, int]:
    matched = []
    for ep in episodes:
        value = float(ep["features"].get(feature, 0.0))
        ok = value >= threshold if direction == ">=" else value <= threshold
        if ok:
            matched.append(ep)
    if not matched:
        return 0.0, 0
    positives = sum(1 for ep in matched if ep["label"] == label)
    return safe_rate(positives, len(matched)), len(matched)


def make_rule(
    episodes: list[dict[str, Any]],
    feature: str,
    threshold: float,
    direction: str,
    target_label: str,
    rule_type: str,
) -> dict[str, Any]:
    precision, support = rate_for_condition(episodes, feature, threshold, direction, target_label)
    total_target = sum(1 for ep in episodes if ep["label"] == target_label)
    covered_target = 0
    for ep in episodes:
        value = float(ep["features"].get(feature, 0.0))
        ok = value >= threshold if direction == ">=" else value <= threshold
        if ok and ep["label"] == target_label:
            covered_target += 1
    recall = safe_rate(covered_target, total_target)
    return {
        "id": f"{rule_type}:{feature}:{direction}:{threshold:.4f}",
        "type": rule_type,
        "target_label": target_label,
        "condition": {
            "feature": feature,
            "operator": direction,
            "threshold": threshold,
        },
        "support": support,
        "precision": precision,
        "recall": recall,
        "interpretation": interpret_rule(rule_type, feature, direction, threshold),
    }


def interpret_rule(rule_type: str, feature: str, direction: str, threshold: float) -> str:
    if rule_type == "risk":
        mapping = {
            "unsupported_high_torque_frac": "无充分支撑时高幅动作占比偏高，提示推进或恢复动作可能过早。",
            "landing_unprepared_frac": "下落阶段后续未形成足底支撑，提示落地准备不足。",
            "rotation_risk_frac": "高角速度或旋转惩罚片段偏多，提示姿态控制失稳。",
            "prone_or_rear_trap_frac": "后仰或俯伏接地片段偏多，提示恢复链条未建立。",
            "backward_speed_frac": "后退速度片段偏多，提示推进方向与任务目标不一致。",
            "rotation_penalty_frac": "旋转惩罚占比偏高，提示动作使躯干角速度恶化。",
        }
    else:
        mapping = {
            "prepared_support_frac": "充分支撑与准备姿态同时出现，提示可进入推进或起跳窗口。",
            "effective_drive_frac": "正向速度、支撑和中高动作幅度同时出现，提示推进动作有效。",
            "mean_full_foot_support": "平均足底支撑较高，提示稳定接触基础较好。",
            "mean_grounded_upright": "接地近直立片段较多，提示恢复或支撑姿态较好。",
            "mean_vx": "平均正向速度较高，提示推进收益稳定。",
            "max_rootx": "最大前向位移较高，提示轨迹整体推进有效。",
        }
    text = mapping.get(feature, feature)
    return f"{text} 条件为 `{feature} {direction} {threshold:.4f}`。"


def induce_rules(attributions: dict[str, Any], max_rules: int, min_support: int) -> dict[str, Any]:
    episodes = attributions["episodes"]
    candidate_rules: list[dict[str, Any]] = []
    for feature in RISK_FEATURES:
        values = feature_values(episodes, feature)
        if not values:
            continue
        threshold = quantile(values, 0.75)
        candidate_rules.append(make_rule(episodes, feature, threshold, ">=", "bad", "risk"))

    for feature in SUCCESS_FEATURES:
        values = feature_values(episodes, feature)
        if not values:
            continue
        threshold = quantile(values, 0.75)
        candidate_rules.append(make_rule(episodes, feature, threshold, ">=", "good", "success"))

    kept = [
        rule
        for rule in candidate_rules
        if rule["support"] >= min_support and rule["precision"] >= 0.45
    ]
    kept.sort(key=lambda r: (r["precision"], r["recall"], r["support"]), reverse=True)
    kept = kept[:max_rules]
    return {
        "source_attributions": attributions.get("selection"),
        "rule_family": "single-condition-decision-list-v1",
        "num_input_episodes": len(episodes),
        "min_support": min_support,
        "max_rules": max_rules,
        "rules": kept,
    }


def write_markdown(path: Path, rules_payload: dict[str, Any]) -> None:
    md_path = path.with_suffix(".md")
    lines = [
        "# Hopper Offline Rule Tree v1",
        "",
        f"Input episodes: `{rules_payload['num_input_episodes']}`",
        f"Rule family: `{rules_payload['rule_family']}`",
        "",
    ]
    for idx, rule in enumerate(rules_payload["rules"], start=1):
        cond = rule["condition"]
        lines.extend(
            [
                f"## Rule {idx}: {rule['type']}",
                "",
                f"- Condition: `{cond['feature']} {cond['operator']} {cond['threshold']:.4f}`",
                f"- Precision: `{rule['precision']:.3f}`",
                f"- Recall: `{rule['recall']:.3f}`",
                f"- Support: `{rule['support']}`",
                f"- Interpretation: {rule['interpretation']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions", type=Path, default=DEFAULT_ATTRIBUTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--max-rules", type=int, default=25)
    parser.add_argument("--min-support", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attributions = load_json(args.attributions)
    payload = induce_rules(attributions, args.max_rules, args.min_support)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote rules: {args.output}")
    print(f"rules kept: {len(payload['rules'])}")


if __name__ == "__main__":
    main()

