"""Audit Hopper rule quality under corrected hop/prone labels.

This script is read-only. It summarizes whether old prone/rotation rules still
look like recovery rules after hop labels are corrected, or whether they mostly
cover normal hop-cycle states.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _phase_from_rule_id(rule_id: str) -> str:
    key = rule_id.split("->", 1)[0]
    parts = key.split("|")
    if "phase_any" in parts:
        return "phase_any"
    if parts:
        return parts[-1]
    return "unknown"


def _family_from_rule_id(rule_id: str) -> str:
    if "->avoid:" in rule_id:
        return "avoid:" + rule_id.split("->avoid:", 1)[1]
    if "->soft_prefer:" in rule_id:
        return "prefer:" + rule_id.split("->soft_prefer:", 1)[1]
    if "->prefer:" in rule_id:
        return "prefer:" + rule_id.split("->prefer:", 1)[1]
    return "unknown"


def _library_rules(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    rules = data.get("rules", data if isinstance(data, list) else [])
    return {str(r.get("id", "")): r for r in rules if r.get("id")}


def _annotate_rule(summary: dict[str, Any], library: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rule_id = summary["id"]
    phase_hits = summary.get("phase_hits") or {}
    state_hits = float(summary.get("state_hits") or 0.0)
    selected_hits = float(summary.get("selected_gate_hits") or 0.0)
    hop_cycle_hits = sum(float(phase_hits.get(k, 0.0)) for k in ("hop", "rising", "falling"))
    grounded_problem_hits = sum(float(phase_hits.get(k, 0.0)) for k in ("prone", "rear"))
    meta = library.get(rule_id, {})
    phase = _phase_from_rule_id(rule_id)
    good = float(summary.get("good_state_hit_rate") or 0.0)
    consistent = float(summary.get("direction_consistent_rate") or 0.0)
    harmful = float(summary.get("direction_harmful_rate") or 0.0)

    if selected_hits >= 100 and consistent >= 0.65 and harmful <= 0.35:
        health = "healthy_direction"
    elif selected_hits >= 100 and harmful >= 0.45:
        health = "direction_suspect"
    elif selected_hits == 0:
        health = "inactive"
    else:
        health = "mixed_or_low_support"

    if phase == "prone" and _safe_div(phase_hits.get("prone", 0.0), state_hits) >= 0.95:
        interpretation = "explicit_prone_recovery"
    elif phase == "hop" or _safe_div(phase_hits.get("hop", 0.0), state_hits) >= 0.50:
        interpretation = "hop_cycle"
    elif _safe_div(hop_cycle_hits, state_hits) >= 0.50:
        interpretation = "mixed_hop_air_cycle"
    else:
        interpretation = "mixed_or_recovery"

    return {
        "id": rule_id,
        "family": _family_from_rule_id(rule_id),
        "source": summary.get("source"),
        "rule_origin": meta.get("rule_origin", "closed_loop_or_legacy"),
        "status": meta.get("status"),
        "confidence_status": meta.get("confidence_status"),
        "confidence": meta.get("confidence"),
        "intervention_type": summary.get("intervention_type"),
        "phase_in_rule": phase,
        "state_hits": int(state_hits),
        "selected_gate_hits": int(selected_hits),
        "phase_hits": phase_hits,
        "hop_cycle_share": _safe_div(hop_cycle_hits, state_hits),
        "hop_share": _safe_div(float(phase_hits.get("hop", 0.0)), state_hits),
        "prone_share": _safe_div(float(phase_hits.get("prone", 0.0)), state_hits),
        "grounded_problem_share": _safe_div(grounded_problem_hits, state_hits),
        "good_state_hit_rate": good,
        "high_episode_hit_rate": float(summary.get("high_episode_hit_rate") or 0.0),
        "direction_consistent_rate": consistent,
        "direction_harmful_rate": harmful,
        "health": health,
        "interpretation": interpretation,
    }


def _mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Hopper Corrected-Label Rule Quality Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    s = report["summary"]
    for key in (
        "num_rules",
        "selected_gate_hit_frac",
        "gate_hit_reward_corr",
        "transition_gate_hit_reward_corr",
        "explicit_prone_rules",
        "phase_any_rules",
        "hop_cycle_interpreted_rules",
        "direction_suspect_rules",
        "inactive_rules",
    ):
        lines.append(f"- `{key}`: `{s[key]}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "修正后标签下，显式 `prone` 规则仍主要命中 `prone` 帧，而不是被大规模转移到 `hop`。"
        "因此，旧规则的问题不是简单的 hop/prone 标签互换；更准确的解释是，`prone` 在 Hopper 中常表示"
        "前倾接地恢复或支撑片段，不能直接等同于失败。"
    )
    lines.append("")
    lines.append(
        "`phase_any` transition 规则覆盖了大量 hop、falling、rising 周期，并且 transition 命中密度与回报的相关"
        f"为 `{s['transition_gate_hit_reward_corr']:.3f}`。这类规则更像是在约束高速跳跃周期中的危险动作，"
        "是当前规则库中最值得保留和进一步审查的部分。"
    )
    lines.append("")
    lines.append(
        "方向可疑规则主要表现为高命中但 opposite/harmful 比例偏高。它们不应在修正标签口径下直接作为下一轮"
        "初始强规则使用，而应降权、重新归因，或通过 TVA/SD 后果审计重新验证。"
    )
    lines.append("")

    lines.append("## Aggregates")
    lines.append("")
    lines.append("| group | rules | selected hits | mean good-state | mean consistent | mean harmful |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["aggregates"]:
        lines.append(
            f"| `{row['group']}` | {row['rules']} | {row['selected_gate_hits']} | "
            f"{row['mean_good_state']:.3f} | {row['mean_consistent']:.3f} | {row['mean_harmful']:.3f} |"
        )
    lines.append("")

    for title, key in (
        ("High-Impact Healthy Rules", "top_healthy"),
        ("Direction-Suspect Rules", "top_suspect"),
        ("Explicit Prone Recovery Rules", "top_prone"),
        ("Hop/Air-Cycle Rules", "top_hop_cycle"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| rule | origin | hits | phase | hop-cycle | prone | good | consistent | harmful |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for r in report[key]:
            lines.append(
                f"| `{r['id']}` | `{r['rule_origin']}` | {r['selected_gate_hits']} | "
                f"`{r['phase_in_rule']}` | {r['hop_cycle_share']:.3f} | {r['prone_share']:.3f} | "
                f"{r['good_state_hit_rate']:.3f} | {r['direction_consistent_rate']:.3f} | "
                f"{r['direction_harmful_rate']:.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-effect-audit", type=Path, required=True)
    parser.add_argument("--rule-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = _load_json(args.rule_effect_audit)
    library = _library_rules(args.rule_library)
    rules = [_annotate_rule(r, library) for r in audit["rule_summaries"]]

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rules:
        by_group[r["interpretation"]].append(r)
        by_group[f"source:{r['source']}"].append(r)
        by_group[f"origin:{r['rule_origin']}"].append(r)

    aggregates = []
    for group, items in sorted(by_group.items()):
        aggregates.append(
            {
                "group": group,
                "rules": len(items),
                "selected_gate_hits": sum(int(r["selected_gate_hits"]) for r in items),
                "mean_good_state": _mean([r["good_state_hit_rate"] for r in items]),
                "mean_consistent": _mean([r["direction_consistent_rate"] for r in items]),
                "mean_harmful": _mean([r["direction_harmful_rate"] for r in items]),
            }
        )

    top_healthy = sorted(
        [r for r in rules if r["health"] == "healthy_direction"],
        key=lambda r: (r["selected_gate_hits"], r["good_state_hit_rate"]),
        reverse=True,
    )[:15]
    top_suspect = sorted(
        [r for r in rules if r["health"] == "direction_suspect"],
        key=lambda r: (r["selected_gate_hits"], r["direction_harmful_rate"]),
        reverse=True,
    )[:20]
    top_prone = sorted(
        [r for r in rules if r["interpretation"] == "explicit_prone_recovery"],
        key=lambda r: r["selected_gate_hits"],
        reverse=True,
    )[:20]
    top_hop_cycle = sorted(
        [r for r in rules if r["interpretation"] in {"hop_cycle", "mixed_hop_air_cycle"}],
        key=lambda r: (r["selected_gate_hits"], r["good_state_hit_rate"]),
        reverse=True,
    )[:20]

    report = {
        "audit_family": "hopper-corrected-label-rule-quality-v0",
        "inputs": {
            "rule_effect_audit": str(args.rule_effect_audit),
            "rule_library": str(args.rule_library),
        },
        "summary": {
            "num_rules": len(rules),
            "selected_gate_hit_frac": audit.get("any_rule_selected_gate_hit_frac"),
            "gate_hit_reward_corr": audit.get("gate_hit_reward_corr"),
            "transition_gate_hit_reward_corr": audit.get("transition_gate_hit_reward_corr"),
            "explicit_prone_rules": sum(r["interpretation"] == "explicit_prone_recovery" for r in rules),
            "phase_any_rules": sum(r["phase_in_rule"] == "phase_any" for r in rules),
            "hop_cycle_interpreted_rules": sum(
                r["interpretation"] in {"hop_cycle", "mixed_hop_air_cycle"} for r in rules
            ),
            "direction_suspect_rules": sum(r["health"] == "direction_suspect" for r in rules),
            "inactive_rules": sum(r["health"] == "inactive" for r in rules),
            "source_counts": Counter(r["source"] for r in rules),
            "origin_counts": Counter(r["rule_origin"] for r in rules),
            "health_counts": Counter(r["health"] for r in rules),
            "interpretation_counts": Counter(r["interpretation"] for r in rules),
        },
        "aggregates": aggregates,
        "rules": rules,
        "top_healthy": top_healthy,
        "top_suspect": top_suspect,
        "top_prone": top_prone,
        "top_hop_cycle": top_hop_cycle,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), report)
    print(f"wrote {args.output}")
    print(f"markdown {args.output.with_suffix('.md')}")
    print(
        "rules={num_rules} suspect={direction_suspect_rules} hop_cycle={hop_cycle_interpreted_rules} "
        "explicit_prone={explicit_prone_rules}".format(**report["summary"])
    )


if __name__ == "__main__":
    main()
