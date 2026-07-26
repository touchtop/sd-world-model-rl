#!/usr/bin/env python3
"""Summarize TVA shadow updates against rule-effect audits across cycles."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import load_json, write_json


DEFAULT_RUN_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1"
)
DEFAULT_SHADOW_DIR = Path(
    "results/sd_retrospective_rule_induction/reports/"
    "tva_validation_300k_oldscheme_20260515"
)
DEFAULT_OUTPUT = DEFAULT_SHADOW_DIR / "tva_shadow_validation_summary.json"
DEFAULT_KEY_RULES = [
    "toe_recovery_low_support->soft_prefer:waist_pos",
    "unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos",
    "rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos",
    "rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos",
    "unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg",
]


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def effect_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row["id"]): row for row in payload.get("rule_summaries", [])}


def row_effect_metrics(row: dict[str, Any], effects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eff = effects.get(str(row["id"]), {})
    return {
        "effect_selected_gate_hits": int(eff.get("selected_gate_hits", 0) or 0),
        "effect_direction_consistent_rate": float(eff.get("direction_consistent_rate", 0.0) or 0.0),
        "effect_direction_harmful_rate": float(eff.get("direction_harmful_rate", 0.0) or 0.0),
        "effect_high_episode_hit_rate": float(eff.get("high_episode_hit_rate", 0.0) or 0.0),
        "effect_good_state_hit_rate": float(eff.get("good_state_hit_rate", 0.0) or 0.0),
    }


def cycle_paths(args: argparse.Namespace, cycle: int) -> tuple[Path, Path]:
    c = f"{cycle:03d}"
    shadow = args.shadow_dir / f"hopper_tva_shadow_update_cycle_{c}{args.shadow_suffix}.json"
    effect = args.run_root / f"cycle_{c}" / "reports" / "rule_effect_audit.json"
    return shadow, effect


def summarize_group(rows: list[dict[str, Any]], effects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = [{**row, **row_effect_metrics(row, effects)} for row in rows]
    return {
        "count": len(merged),
        "mean_delta": avg([float(row["shadow_confidence_delta"]) for row in merged]),
        "mean_tva_same_rate": avg([float(row.get("tva_shadow_same_rate", 0.0) or 0.0) for row in merged]),
        "mean_tva_opposite_rate": avg([float(row.get("tva_shadow_opposite_rate", 0.0) or 0.0) for row in merged]),
        "mean_effect_consistent_rate": avg([row["effect_direction_consistent_rate"] for row in merged]),
        "mean_effect_harmful_rate": avg([row["effect_direction_harmful_rate"] for row in merged]),
        "mean_effect_high_episode_hit_rate": avg([row["effect_high_episode_hit_rate"] for row in merged]),
        "mean_effect_good_state_hit_rate": avg([row["effect_good_state_hit_rate"] for row in merged]),
    }


def summarize_cycle(args: argparse.Namespace, cycle: int) -> dict[str, Any]:
    shadow_path, effect_path = cycle_paths(args, cycle)
    shadow = load_json(shadow_path)
    effects = effect_index(effect_path)
    rules = shadow.get("rules", [])
    strengthened = [row for row in rules if float(row.get("shadow_confidence_delta", 0.0) or 0.0) > args.delta_epsilon]
    weakened = [row for row in rules if float(row.get("shadow_confidence_delta", 0.0) or 0.0) < -args.delta_epsilon]
    unchanged = [row for row in rules if abs(float(row.get("shadow_confidence_delta", 0.0) or 0.0)) <= args.delta_epsilon]
    key_rules = []
    for rid in args.key_rule:
        row = next((item for item in rules if str(item.get("id")) == rid), None)
        if row is None:
            key_rules.append({"id": rid, "present": False})
            continue
        key_rules.append(
            {
                "id": rid,
                "present": True,
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "confidence_status": row.get("confidence_status", row.get("status")),
                "shadow_confidence": float(row.get("shadow_confidence", 0.0) or 0.0),
                "shadow_confidence_status": row.get("shadow_confidence_status"),
                "shadow_confidence_delta": float(row.get("shadow_confidence_delta", 0.0) or 0.0),
                "shadow_update_reason": row.get("shadow_update_reason"),
                "tva_shadow_selected_gate_hits": int(row.get("tva_shadow_selected_gate_hits", 0) or 0),
                "tva_shadow_same_rate": float(row.get("tva_shadow_same_rate", 0.0) or 0.0),
                "tva_shadow_opposite_rate": float(row.get("tva_shadow_opposite_rate", 0.0) or 0.0),
                **row_effect_metrics(row, effects),
            }
        )
    return {
        "cycle": cycle,
        "shadow_update": str(shadow_path),
        "rule_effect_audit": str(effect_path),
        "matched_state_rule_pairs": shadow.get("matched_state_rule_pairs"),
        "shadow_summary": shadow.get("summary", {}),
        "strengthened_effect_summary": summarize_group(strengthened, effects),
        "weakened_effect_summary": summarize_group(weakened, effects),
        "unchanged_effect_summary": summarize_group(unchanged, effects),
        "top_strengthened": [
            {**{key: row.get(key) for key in ("id", "source", "confidence", "shadow_confidence", "shadow_confidence_delta", "shadow_update_reason", "tva_shadow_selected_gate_hits", "tva_shadow_same_rate", "tva_shadow_opposite_rate")}, **row_effect_metrics(row, effects)}
            for row in sorted(strengthened, key=lambda item: float(item["shadow_confidence_delta"]), reverse=True)[: args.top_k]
        ],
        "top_weakened": [
            {**{key: row.get(key) for key in ("id", "source", "confidence", "shadow_confidence", "shadow_confidence_delta", "shadow_update_reason", "tva_shadow_selected_gate_hits", "tva_shadow_same_rate", "tva_shadow_opposite_rate")}, **row_effect_metrics(row, effects)}
            for row in sorted(weakened, key=lambda item: float(item["shadow_confidence_delta"]))[: args.top_k]
        ],
        "key_rules": key_rules,
    }


def persistent_changes(cycles: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cyc in cycles:
        shadow = load_json(Path(cyc["shadow_update"]))
        for row in shadow.get("rules", []):
            delta = float(row.get("shadow_confidence_delta", 0.0) or 0.0)
            if abs(delta) <= args.delta_epsilon:
                continue
            by_rule[str(row["id"])].append(
                {
                    "cycle": cyc["cycle"],
                    "delta": delta,
                    "reason": row.get("shadow_update_reason"),
                    "same_rate": row.get("tva_shadow_same_rate"),
                    "opposite_rate": row.get("tva_shadow_opposite_rate"),
                }
            )
    strengthened = []
    weakened = []
    for rid, vals in by_rule.items():
        pos = [v for v in vals if v["delta"] > 0]
        neg = [v for v in vals if v["delta"] < 0]
        if len(pos) >= args.min_persistent_cycles and not neg:
            strengthened.append({"id": rid, "cycles": [v["cycle"] for v in pos], "mean_delta": avg([v["delta"] for v in pos])})
        if len(neg) >= args.min_persistent_cycles and not pos:
            weakened.append({"id": rid, "cycles": [v["cycle"] for v in neg], "mean_delta": avg([v["delta"] for v in neg])})
    strengthened.sort(key=lambda item: item["mean_delta"], reverse=True)
    weakened.sort(key=lambda item: item["mean_delta"])
    return {
        "strengthened": strengthened,
        "weakened": weakened,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TVA Shadow Validation Summary",
        "",
        f"Run root: `{payload['run_root']}`",
        f"Shadow directory: `{payload['shadow_dir']}`",
        f"Cycles: `{payload['cycles_requested']}`",
        "",
        "## Cycle Overview",
        "",
        "| Cycle | Rules | Matched pairs | Strengthened | Weakened | Unchanged | Mean delta | Strengthened effect consistent/harmful | Weakened effect consistent/harmful |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cyc in payload["cycles"]:
        summary = cyc["shadow_summary"]
        s_eff = cyc["strengthened_effect_summary"]
        w_eff = cyc["weakened_effect_summary"]
        lines.append(
            "| "
            f"{cyc['cycle']} | "
            f"{summary.get('num_rules', 0)} | "
            f"{cyc.get('matched_state_rule_pairs', 0)} | "
            f"{summary.get('num_strengthened', 0)} | "
            f"{summary.get('num_weakened', 0)} | "
            f"{summary.get('num_unchanged', 0)} | "
            f"{float(summary.get('mean_shadow_delta', 0.0)):.4f} | "
            f"{s_eff['mean_effect_consistent_rate']:.3f}/{s_eff['mean_effect_harmful_rate']:.3f} | "
            f"{w_eff['mean_effect_consistent_rate']:.3f}/{w_eff['mean_effect_harmful_rate']:.3f} |"
        )
    lines.extend(["", "## Persistent TVA Shadow Changes", ""])
    for label, rows in (("Strengthened", payload["persistent_changes"]["strengthened"]), ("Weakened", payload["persistent_changes"]["weakened"])):
        lines.extend([f"### {label}", "", "| Rule | Cycles | Mean delta |", "| --- | --- | ---: |"])
        for row in rows[:20]:
            lines.append(f"| `{row['id']}` | `{row['cycles']}` | `{row['mean_delta']:+.4f}` |")
        if not rows:
            lines.append("| none |  |  |")
        lines.append("")
    lines.extend(["## Key Rules", ""])
    for cyc in payload["cycles"]:
        lines.extend([f"### Cycle {cyc['cycle']}", "", "| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
        for row in cyc["key_rules"]:
            if not row.get("present"):
                lines.append(f"| `{row['id']}` | missing |  |  |  |  |  |")
                continue
            lines.append(
                "| "
                f"`{row['id']}` | "
                f"{row['confidence']:.3f} -> {row['shadow_confidence']:.3f} | "
                f"{row['shadow_confidence_delta']:+.3f} | "
                f"{row['tva_shadow_selected_gate_hits']} | "
                f"{row['tva_shadow_same_rate']:.3f}/{row['tva_shadow_opposite_rate']:.3f} | "
                f"{row['effect_direction_consistent_rate']:.3f}/{row['effect_direction_harmful_rate']:.3f} | "
                f"`{row['shadow_update_reason']}` |"
            )
        lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--shadow-suffix", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cycles", type=int, nargs="+", default=[41, 50, 60])
    parser.add_argument("--key-rule", action="append", default=list(DEFAULT_KEY_RULES))
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--min-persistent-cycles", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycles = [summarize_cycle(args, cycle) for cycle in args.cycles]
    payload = {
        "run_root": str(args.run_root),
        "shadow_dir": str(args.shadow_dir),
        "cycles_requested": args.cycles,
        "cycles": cycles,
        "persistent_changes": persistent_changes(cycles, args),
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA shadow validation summary: {args.output}")


if __name__ == "__main__":
    main()
