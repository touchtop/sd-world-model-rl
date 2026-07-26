#!/usr/bin/env python3
"""Audit whether TVA shadow updates predict later rule-effect behavior.

The analysis is read-only. It compares an anchor cycle's TVA shadow proposal
with rule-effect audits in later cycles. This checks predictive alignment, not
causal improvement, because the TVA shadow proposal was not used online.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = DEFAULT_SHADOW_DIR / "tva_shadow_lag_effects.json"
DEFAULT_WINDOWS = ["041:042-050", "050:051-060"]


METRICS = (
    "direction_consistent_rate",
    "direction_harmful_rate",
    "high_episode_hit_rate",
    "good_state_hit_rate",
)


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def weighted_avg(values: list[tuple[float, float]]) -> float:
    denom = sum(weight for _value, weight in values)
    if denom <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in values) / denom


def parse_window(text: str) -> tuple[int, list[int]]:
    if ":" not in text or "-" not in text:
        raise argparse.ArgumentTypeError("expected ANCHOR:START-END")
    anchor_text, range_text = text.split(":", 1)
    start_text, end_text = range_text.split("-", 1)
    anchor = int(anchor_text)
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise argparse.ArgumentTypeError("window start must be <= end")
    return anchor, list(range(start, end + 1))


def effect_path(run_root: Path, cycle: int) -> Path:
    return run_root / f"cycle_{cycle:03d}" / "reports" / "rule_effect_audit.json"


def shadow_path(args: argparse.Namespace, cycle: int) -> Path:
    return args.shadow_dir / f"hopper_tva_shadow_update_cycle_{cycle:03d}{args.shadow_suffix}.json"


def effect_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {str(row["id"]): row for row in payload.get("rule_summaries", [])}


def effect_metrics(row: dict[str, Any] | None) -> dict[str, float | int]:
    if row is None:
        return {
            "selected_gate_hits": 0,
            "direction_consistent_rate": 0.0,
            "direction_harmful_rate": 0.0,
            "high_episode_hit_rate": 0.0,
            "good_state_hit_rate": 0.0,
        }
    return {
        "selected_gate_hits": int(row.get("selected_gate_hits", 0) or 0),
        "direction_consistent_rate": float(row.get("direction_consistent_rate", 0.0) or 0.0),
        "direction_harmful_rate": float(row.get("direction_harmful_rate", 0.0) or 0.0),
        "high_episode_hit_rate": float(row.get("high_episode_hit_rate", 0.0) or 0.0),
        "good_state_hit_rate": float(row.get("good_state_hit_rate", 0.0) or 0.0),
    }


def future_metrics(rule_id: str, future_effects: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for cycle, index in future_effects.items():
        row = index.get(rule_id)
        if row is None:
            continue
        metrics = effect_metrics(row)
        metrics["cycle"] = cycle
        rows.append(metrics)
    out: dict[str, Any] = {
        "future_cycles_seen": [int(row["cycle"]) for row in rows],
        "future_cycles_seen_count": len(rows),
        "future_selected_gate_hits_total": sum(int(row["selected_gate_hits"]) for row in rows),
    }
    for metric in METRICS:
        out[f"future_mean_{metric}"] = avg([float(row[metric]) for row in rows])
        out[f"future_weighted_{metric}"] = weighted_avg(
            [(float(row[metric]), float(row["selected_gate_hits"])) for row in rows]
        )
    return out


def classify_shadow(row: dict[str, Any], epsilon: float) -> str:
    delta = float(row.get("shadow_confidence_delta", 0.0) or 0.0)
    if delta > epsilon:
        return "strengthened"
    if delta < -epsilon:
        return "weakened"
    return "unchanged"


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen = [row for row in rows if int(row.get("future_cycles_seen_count", 0) or 0) > 0]
    out: dict[str, Any] = {
        "count": len(rows),
        "future_seen_count": len(seen),
        "mean_shadow_delta": avg([float(row.get("shadow_confidence_delta", 0.0) or 0.0) for row in rows]),
        "mean_tva_same_rate": avg([float(row.get("tva_shadow_same_rate", 0.0) or 0.0) for row in rows]),
        "mean_tva_opposite_rate": avg([float(row.get("tva_shadow_opposite_rate", 0.0) or 0.0) for row in rows]),
        "mean_anchor_selected_gate_hits": avg([float(row.get("anchor_selected_gate_hits", 0.0) or 0.0) for row in rows]),
        "mean_future_selected_gate_hits_total": avg([float(row.get("future_selected_gate_hits_total", 0.0) or 0.0) for row in seen]),
    }
    for metric in METRICS:
        out[f"mean_anchor_{metric}"] = avg([float(row.get(f"anchor_{metric}", 0.0) or 0.0) for row in rows])
        out[f"mean_future_{metric}"] = avg([float(row.get(f"future_weighted_{metric}", 0.0) or 0.0) for row in seen])
        out[f"mean_future_minus_anchor_{metric}"] = avg(
            [
                float(row.get(f"future_weighted_{metric}", 0.0) or 0.0)
                - float(row.get(f"anchor_{metric}", 0.0) or 0.0)
                for row in seen
            ]
        )
    return out


def audit_window(args: argparse.Namespace, anchor: int, future_cycles: list[int]) -> dict[str, Any]:
    shadow = load_json(shadow_path(args, anchor))
    anchor_effects = effect_index(effect_path(args.run_root, anchor))
    future_effects = {cycle: effect_index(effect_path(args.run_root, cycle)) for cycle in future_cycles}
    records = []
    for row in shadow.get("rules", []):
        rid = str(row["id"])
        group = classify_shadow(row, args.delta_epsilon)
        anchor_metrics = effect_metrics(anchor_effects.get(rid))
        future = future_metrics(rid, future_effects)
        rec = {
            "id": rid,
            "group": group,
            "source": row.get("source"),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "shadow_confidence": float(row.get("shadow_confidence", 0.0) or 0.0),
            "shadow_confidence_delta": float(row.get("shadow_confidence_delta", 0.0) or 0.0),
            "shadow_update_reason": row.get("shadow_update_reason"),
            "tva_shadow_selected_gate_hits": int(row.get("tva_shadow_selected_gate_hits", 0) or 0),
            "tva_shadow_same_rate": float(row.get("tva_shadow_same_rate", 0.0) or 0.0),
            "tva_shadow_opposite_rate": float(row.get("tva_shadow_opposite_rate", 0.0) or 0.0),
            "anchor_selected_gate_hits": anchor_metrics["selected_gate_hits"],
            **{f"anchor_{metric}": anchor_metrics[metric] for metric in METRICS},
            **future,
        }
        for metric in METRICS:
            rec[f"future_minus_anchor_{metric}"] = (
                float(rec.get(f"future_weighted_{metric}", 0.0) or 0.0)
                - float(rec.get(f"anchor_{metric}", 0.0) or 0.0)
            )
        records.append(rec)

    groups = {
        name: [row for row in records if row["group"] == name]
        for name in ("strengthened", "weakened", "unchanged")
    }
    return {
        "anchor_cycle": anchor,
        "future_cycles": future_cycles,
        "shadow_update": str(shadow_path(args, anchor)),
        "anchor_rule_effect_audit": str(effect_path(args.run_root, anchor)),
        "future_rule_effect_audits": [str(effect_path(args.run_root, cycle)) for cycle in future_cycles],
        "group_summaries": {name: summarize_group(rows) for name, rows in groups.items()},
        "top_strengthened": sorted(groups["strengthened"], key=lambda row: row["shadow_confidence_delta"], reverse=True)[: args.top_k],
        "top_weakened": sorted(groups["weakened"], key=lambda row: row["shadow_confidence_delta"])[: args.top_k],
        "records": records,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TVA Shadow Lag Effects",
        "",
        "This report checks whether an anchor cycle's TVA shadow recommendation predicts later rule-effect audits. It is predictive validation only; the shadow recommendation was not used online.",
        "",
        f"Run root: `{payload['run_root']}`",
        f"Shadow directory: `{payload['shadow_dir']}`",
        "",
    ]
    for window in payload["windows"]:
        lines.extend(
            [
                f"## Anchor {window['anchor_cycle']} -> Future {window['future_cycles'][0]}-{window['future_cycles'][-1]}",
                "",
                "| Group | Count | Seen later | Shadow delta | TVA same/opposite | Future consistent/harmful | Future - anchor consistent/harmful | Future high/good |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in ("strengthened", "weakened", "unchanged"):
            stats = window["group_summaries"][group]
            lines.append(
                "| "
                f"`{group}` | "
                f"{stats['count']} | "
                f"{stats['future_seen_count']} | "
                f"{stats['mean_shadow_delta']:+.4f} | "
                f"{stats['mean_tva_same_rate']:.3f}/{stats['mean_tva_opposite_rate']:.3f} | "
                f"{stats['mean_future_direction_consistent_rate']:.3f}/{stats['mean_future_direction_harmful_rate']:.3f} | "
                f"{stats['mean_future_minus_anchor_direction_consistent_rate']:+.3f}/{stats['mean_future_minus_anchor_direction_harmful_rate']:+.3f} | "
                f"{stats['mean_future_high_episode_hit_rate']:.3f}/{stats['mean_future_good_state_hit_rate']:.3f} |"
            )
        for title, key in (("Top Strengthened", "top_strengthened"), ("Top Weakened", "top_weakened")):
            lines.extend(
                [
                    "",
                    f"### {title}",
                    "",
                    "| Rule | Delta | TVA hits | TVA same/opposite | Anchor consistent/harmful | Future consistent/harmful | Future cycles |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in window[key]:
                lines.append(
                    "| "
                    f"`{row['id']}` | "
                    f"{row['shadow_confidence_delta']:+.3f} | "
                    f"{row['tva_shadow_selected_gate_hits']} | "
                    f"{row['tva_shadow_same_rate']:.3f}/{row['tva_shadow_opposite_rate']:.3f} | "
                    f"{row['anchor_direction_consistent_rate']:.3f}/{row['anchor_direction_harmful_rate']:.3f} | "
                    f"{row['future_weighted_direction_consistent_rate']:.3f}/{row['future_weighted_direction_harmful_rate']:.3f} | "
                    f"`{row['future_cycles_seen']}` |"
                )
            if not window[key]:
                lines.append("| none |  |  |  |  |  |  |")
            lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--shadow-suffix", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", action="append", type=parse_window, default=[])
    parser.add_argument("--delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = args.window or [parse_window(text) for text in DEFAULT_WINDOWS]
    payload = {
        "audit_family": "tva_shadow_lag_effects_v1",
        "run_root": str(args.run_root),
        "shadow_dir": str(args.shadow_dir),
        "windows_requested": [
            {"anchor_cycle": anchor, "future_cycles": future_cycles}
            for anchor, future_cycles in windows
        ],
        "windows": [audit_window(args, anchor, future_cycles) for anchor, future_cycles in windows],
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA shadow lag effects: {args.output}")


if __name__ == "__main__":
    main()
