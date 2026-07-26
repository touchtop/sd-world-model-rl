#!/usr/bin/env python3
"""Audit teacher-core rule coverage on exported Hopper trajectories.

The audit is read-only. It checks whether the current state-label and action
gate semantics give the selected teacher-core rules enough opportunities to
affect MPPI candidate ranking on replayed trajectories.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_PHASE_MODES = ("current", "no_hop", "hop_as_prone", "hop_alias")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_trace_set(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected LABEL=GLOB")
    label, pattern = text.split("=", 1)
    label = label.strip()
    pattern = pattern.strip()
    if not label or not pattern:
        raise argparse.ArgumentTypeError("trace set label and glob must be non-empty")
    return label, pattern


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def flag(row: dict[str, str], key: str) -> bool:
    if f"{key}_gt_05" in row:
        return row.get(f"{key}_gt_05") in {"1", "1.0", "true", "True"}
    return as_float(row, key) >= 0.5


def row_parts(row: dict[str, str], phase_mode: str) -> dict[str, str]:
    pitch_rate = as_float(row, "pitch_rate")
    vx = as_float(row, "vx")
    full_foot = as_float(row, "full_foot_support")

    omega_bin = "omega_pos" if pitch_rate > 1.0 else "omega_neg" if pitch_rate < -1.0 else "omega_mid"
    vx_bin = "vx_high" if vx >= 0.8 else "vx_low" if vx < 0.2 else "vx_mid"
    support_bin = "support_none" if full_foot < 0.1 else "support_low" if full_foot < 0.35 else "support_some"

    falling = flag(row, "falling_air")
    rear = flag(row, "grounded_rear")
    hop = flag(row, "grounded_hop")
    prone = flag(row, "grounded_prone")
    upright = flag(row, "grounded_upright") or flag(row, "grounded_prepared")

    if phase_mode == "no_hop":
        hop = False
    if falling:
        phase = "falling"
    elif rear:
        phase = "rear"
    elif hop and phase_mode != "hop_as_prone":
        phase = "hop"
    elif (hop and phase_mode == "hop_as_prone") or prone:
        phase = "prone"
    elif upright:
        phase = "upright"
    else:
        phase = "other"

    return {
        "omega_bin": omega_bin,
        "vx_bin": vx_bin,
        "support_bin": support_bin,
        "phase": phase,
    }


def state_matches(rule: dict[str, Any], parts: dict[str, str], phase_mode: str) -> bool:
    key = str(rule.get("substate_key") or rule.get("rule_key") or "")
    tokens = key.split("|")
    if len(tokens) != 5:
        return False
    event, omega_bin, vx_bin, support_bin, phase = tokens
    if event != "unsupported_high_torque":
        return False
    if omega_bin != parts["omega_bin"] or vx_bin != parts["vx_bin"] or support_bin != parts["support_bin"]:
        return False
    if phase_mode == "hop_alias" and phase == "hop":
        return parts["phase"] in {"hop", "prone", "other"}
    return phase == parts["phase"]


def action_gate(row: dict[str, str], threshold: float) -> bool:
    action_values = [
        abs(as_float(row, f"action_{idx}", 0.0))
        for idx in range(4)
        if f"action_{idx}" in row
    ]
    return bool(action_values) and max(action_values) >= threshold


def summarize_trace_set(
    label: str,
    pattern: str,
    rules: list[dict[str, Any]],
    phase_modes: list[str],
    action_threshold: float,
    max_files: int,
) -> dict[str, Any]:
    files = sorted(glob.glob(pattern, recursive=True))
    if max_files > 0:
        files = files[:max_files]
    summaries: dict[str, Any] = {}
    for mode in phase_modes:
        steps = 0
        state_match_steps = 0
        gated_match_steps = 0
        matched_rules = 0
        gated_rules = 0
        phase_counts: Counter[str] = Counter()
        key_counts: Counter[str] = Counter()
        gated_key_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    steps += 1
                    parts = row_parts(row, mode)
                    phase_counts[parts["phase"]] += 1
                    state_hits = [rule for rule in rules if state_matches(rule, parts, mode)]
                    if not state_hits:
                        continue
                    state_match_steps += 1
                    matched_rules += len(state_hits)
                    for rule in state_hits:
                        key_counts[str(rule.get("substate_key") or rule.get("rule_key"))] += 1
                        family_counts[str(rule.get("action_family", ""))] += 1
                    if action_gate(row, action_threshold):
                        gated_match_steps += 1
                        gated_rules += len(state_hits)
                        for rule in state_hits:
                            gated_key_counts[str(rule.get("substate_key") or rule.get("rule_key"))] += 1
        summaries[mode] = {
            "files": len(files),
            "steps": steps,
            "state_match_steps": state_match_steps,
            "state_match_rate": state_match_steps / steps if steps else 0.0,
            "gated_match_steps": gated_match_steps,
            "gated_match_rate": gated_match_steps / steps if steps else 0.0,
            "matched_rules": matched_rules,
            "gated_rules": gated_rules,
            "mean_state_matched_rules_per_step": matched_rules / steps if steps else 0.0,
            "mean_gated_rules_per_step": gated_rules / steps if steps else 0.0,
            "phase_counts": dict(phase_counts.most_common()),
            "top_state_match_keys": dict(key_counts.most_common(12)),
            "top_gated_match_keys": dict(gated_key_counts.most_common(12)),
            "top_action_families": dict(family_counts.most_common(12)),
        }
    return {"label": label, "pattern": pattern, "summaries": summaries}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Teacher-Core Prior Coverage Audit",
        "",
        f"Rule library: `{payload['rule_library']}`",
        f"Rules audited: `{payload['num_rules']}`",
        f"Action gate threshold: `{payload['action_threshold']}`",
        "",
        "| trace set | phase mode | files | steps | state match | gated match | matched rules/step | gated rules/step |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for trace_set in payload["trace_sets"]:
        label = trace_set["label"]
        for mode, summary in trace_set["summaries"].items():
            lines.append(
                "| "
                f"{label} | {mode} | {summary['files']} | {summary['steps']} | "
                f"{summary['state_match_rate']:.4f} | {summary['gated_match_rate']:.4f} | "
                f"{summary['mean_state_matched_rules_per_step']:.4f} | {summary['mean_gated_rules_per_step']:.4f} |"
            )
    lines.append("")
    lines.append("## Top Gated Keys")
    for trace_set in payload["trace_sets"]:
        lines.append("")
        lines.append(f"### {trace_set['label']}")
        for mode, summary in trace_set["summaries"].items():
            top = ", ".join(f"`{k}`={v}" for k, v in list(summary["top_gated_match_keys"].items())[:6])
            lines.append(f"- `{mode}`: {top or 'none'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-library", type=Path, required=True)
    parser.add_argument("--trace-set", action="append", type=parse_trace_set, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-mode", action="append", choices=VALID_PHASE_MODES)
    parser.add_argument("--action-threshold", type=float, default=0.65)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--teacher-origin", default="teacher_core_standard_hop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = load_json(args.rule_library)
    rules = [
        rule for rule in library.get("rules", [])
        if str(rule.get("rule_origin", "")) == args.teacher_origin
    ]
    phase_modes = args.phase_mode or list(VALID_PHASE_MODES)
    payload = {
        "rule_library": str(args.rule_library),
        "teacher_origin": args.teacher_origin,
        "num_rules": len(rules),
        "action_threshold": args.action_threshold,
        "phase_modes": phase_modes,
        "trace_sets": [
            summarize_trace_set(label, pattern, rules, phase_modes, args.action_threshold, args.max_files)
            for label, pattern in args.trace_set
        ],
    }
    write_json(args.output, payload)
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
