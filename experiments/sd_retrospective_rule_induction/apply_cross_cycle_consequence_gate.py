#!/usr/bin/env python3
"""Apply a cross-cycle consequence gate to a shadow rule library.

The script compares a source confidence library with a candidate shadow
library. Candidate confidence changes are kept only when the rule is marked as
stable consequence-supported in a cross-cycle consequence summary. Other
changes are reverted to the source library. The output remains a shadow/eval
artifact and does not modify the live confidence library.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apply_rule_effect_confidence import confidence_status, rule_id


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def confidence(rule: dict[str, Any]) -> float:
    return float(rule.get("confidence", 0.0) or 0.0)


def stable_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {
        str(row.get("id")): row
        for row in payload.get("rules", [])
        if row.get("id") and bool(row.get("stable_consequence_supported", False))
    }


def apply_gate(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(args.source_library)
    shadow = load_json(args.shadow_library)
    stable_by_id = stable_index(args.cross_cycle_summary)
    source_by_id = {rule_id(rule): rule for rule in source.get("rules", [])}

    out_rules: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    kept = 0
    rejected = 0
    changed = 0
    missing_from_source = 0
    for rule in shadow.get("rules", []):
        rid = rule_id(rule)
        source_rule = source_by_id.get(rid)
        if source_rule is None:
            missing_from_source += 1
            out_rules.append(rule)
            continue

        before = confidence(source_rule)
        after = confidence(rule)
        delta = after - before
        out = dict(rule)
        stable = stable_by_id.get(rid)
        passed = True
        reason = "unchanged"
        if abs(delta) > args.delta_epsilon or source_rule.get("status") != rule.get("status"):
            changed += 1
            passed = stable is not None
            if passed:
                kept += 1
                reason = "cross_cycle_stable_change_kept"
            else:
                rejected += 1
                reason = "cross_cycle_unstable_change_reverted"
                out["confidence"] = round(before, 6)
                out["confidence_status"] = confidence_status(before, args.active_threshold, args.retired_threshold)
                out["status"] = out["confidence_status"]
                out["effective_prior_scale"] = round(before, 6)

        out["cross_cycle_consequence_gate"] = {
            "source_cross_cycle_summary": str(args.cross_cycle_summary),
            "passed": bool(passed),
            "reason": reason,
            "confidence_before_shadow": round(before, 6),
            "confidence_before_gate": round(after, 6),
            "confidence_after_gate": round(float(out.get("confidence", 0.0) or 0.0), 6),
            "num_compared_audits": int(stable.get("num_compared_audits", 0) or 0) if stable else 0,
            "num_supported_audits": int(stable.get("num_supported_audits", 0) or 0) if stable else 0,
            "support_rate": stable.get("support_rate") if stable else None,
            "mean_outcome_margin": stable.get("mean_outcome_margin") if stable else None,
            "mean_shadow_confidence_delta": stable.get("mean_shadow_confidence_delta") if stable else None,
        }
        rows.append({"id": rid, "delta_before_gate": delta, **out["cross_cycle_consequence_gate"]})
        out_rules.append(out)

    status_counts = Counter(str(rule.get("status")) for rule in out_rules)
    now = datetime.now(timezone.utc).isoformat()
    payload = dict(shadow)
    payload["rules"] = out_rules
    payload["library_version"] = f"{shadow.get('library_version', 'rule_library')}_cross_cycle_consequence_gated"
    payload["created_at_utc"] = now
    payload["source_library"] = str(args.source_library)
    payload["cross_cycle_consequence_gate_update"] = {
        "source_library": str(args.source_library),
        "shadow_library": str(args.shadow_library),
        "cross_cycle_summary": str(args.cross_cycle_summary),
        "parameters": {
            "active_threshold": args.active_threshold,
            "retired_threshold": args.retired_threshold,
            "delta_epsilon": args.delta_epsilon,
        },
        "num_stable_supported_rules": len(stable_by_id),
        "num_candidate_rules": len(shadow.get("rules", [])),
        "num_candidate_changed": changed,
        "num_changed_kept": kept,
        "num_changed_rejected": rejected,
        "num_missing_from_source": missing_from_source,
        "status_counts": dict(status_counts),
        "rows": sorted(rows, key=lambda row: (not row["passed"], -abs(float(row["delta_before_gate"])), row["id"])),
    }
    payload["online_enabled"] = any(rule.get("status") == "confidence_active" for rule in out_rules)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    update = payload["cross_cycle_consequence_gate_update"]
    lines = [
        "# Cross-Cycle Consequence-Gated Shadow Library",
        "",
        f"Source library: `{update['source_library']}`",
        f"Candidate shadow library: `{update['shadow_library']}`",
        f"Cross-cycle summary: `{update['cross_cycle_summary']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| stable supported rules | {update['num_stable_supported_rules']} |",
        f"| candidate rules | {update['num_candidate_rules']} |",
        f"| candidate changed rules | {update['num_candidate_changed']} |",
        f"| changed kept | {update['num_changed_kept']} |",
        f"| changed rejected | {update['num_changed_rejected']} |",
        f"| status counts | `{update['status_counts']}` |",
        "",
        "This is a shadow artifact. It does not modify the live confidence library.",
        "",
        "## Changed Rules",
        "",
        "| kept | delta | supported/compared | support rate | mean margin | rule |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in update["rows"]:
        if abs(float(row.get("delta_before_gate", 0.0) or 0.0)) <= 1e-12:
            continue
        support_rate = row.get("support_rate")
        mean_margin = row.get("mean_outcome_margin")
        lines.append(
            "| "
            f"`{row['passed']}` | "
            f"{float(row['delta_before_gate']):+.3f} | "
            f"{row['num_supported_audits']}/{row['num_compared_audits']} | "
            f"{'n/a' if support_rate is None else f'{float(support_rate):.3f}'} | "
            f"{'n/a' if mean_margin is None else f'{float(mean_margin):+.4f}'} | "
            f"`{row['id']}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--shadow-library", type=Path, required=True)
    parser.add_argument("--cross-cycle-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-threshold", type=float, default=0.55)
    parser.add_argument("--retired-threshold", type=float, default=0.05)
    parser.add_argument("--delta-epsilon", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = apply_gate(args)
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    update = payload["cross_cycle_consequence_gate_update"]
    print(f"wrote cross-cycle consequence-gated shadow library: {args.output}")
    print(f"markdown: {args.output.with_suffix('.md')}")
    print(
        "stable={stable} changed={changed} kept={kept} rejected={rejected} status={status}".format(
            stable=update["num_stable_supported_rules"],
            changed=update["num_candidate_changed"],
            kept=update["num_changed_kept"],
            rejected=update["num_changed_rejected"],
            status=update["status_counts"],
        )
    )


if __name__ == "__main__":
    main()
