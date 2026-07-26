"""Build a compact Hopper initial rule library after corrected-label audit.

The output is intended for the next online smoke/retraining line. It keeps the
rule mechanism simple: transition safety rules and audited teacher-core action
rules are retained, while direction-suspect recovery rules are not promoted.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rule_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return list(payload.get("rules", []))
    if isinstance(payload, list):
        return list(payload)
    raise TypeError(f"Unsupported rule payload type: {type(payload)!r}")


def _keep_reason(audit_rule: dict[str, Any]) -> str | None:
    if audit_rule.get("source") == "v8_transition_replay_safety":
        if audit_rule.get("health") != "direction_suspect" and int(audit_rule.get("selected_gate_hits") or 0) > 0:
            return "corrected_label_transition_non_suspect"
        return None
    if (
        audit_rule.get("rule_origin") == "teacher_core_standard_hop"
        and audit_rule.get("health") == "healthy_direction"
        and int(audit_rule.get("selected_gate_hits") or 0) > 0
    ):
        return "corrected_label_teacher_core_healthy"
    return None


def _render_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    summary = payload["corrected_label_initial_library"]
    lines.append("# Corrected-Label Initial Rule Library")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "source_library",
        "quality_audit",
        "num_source_rules",
        "num_audited_active_rules",
        "num_kept_rules",
    ):
        lines.append(f"- `{key}`: `{summary[key]}`")
    lines.append(f"- `keep_reason_counts`: `{summary['keep_reason_counts']}`")
    lines.append(f"- `source_counts`: `{summary['source_counts']}`")
    lines.append(f"- `origin_counts`: `{summary['origin_counts']}`")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(
        "该库只保留修正标签审计下的非可疑 transition 安全规则和方向健康的教师核心规则。"
        "方向可疑的旧 prone/falling 恢复规则不作为下一轮初始强先验。"
    )
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append("| reason | source | origin | confidence | rule |")
    lines.append("|---|---|---|---:|---|")
    for rule in payload["rules"]:
        lines.append(
            f"| `{rule.get('corrected_label_keep_reason')}` | `{rule.get('source')}` | "
            f"`{rule.get('rule_origin', 'closed_loop_or_legacy')}` | "
            f"{float(rule.get('confidence', rule.get('effective_prior_scale', 0.0)) or 0.0):.3f} | "
            f"`{rule.get('id')}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build(source_library: Path, quality_audit: Path, output: Path) -> dict[str, Any]:
    source_payload = _load_json(source_library)
    audit_payload = _load_json(quality_audit)
    source_rules = {str(rule.get("id")): rule for rule in _rule_list(source_payload) if rule.get("id")}
    audit_rules = list(audit_payload.get("rules", []))

    kept: list[dict[str, Any]] = []
    omitted_reasons: Counter[str] = Counter()
    for audit_rule in audit_rules:
        rule_id = str(audit_rule.get("id", ""))
        source_rule = source_rules.get(rule_id)
        if source_rule is None:
            omitted_reasons["missing_from_source_library"] += 1
            continue
        reason = _keep_reason(audit_rule)
        if reason is None:
            omitted_reasons[f"omit:{audit_rule.get('health')}"] += 1
            continue
        rule = copy.deepcopy(source_rule)
        rule["status"] = "confidence_active"
        rule["confidence_status"] = "confidence_active"
        rule["corrected_label_keep_reason"] = reason
        rule["corrected_label_quality"] = {
            "health": audit_rule.get("health"),
            "interpretation": audit_rule.get("interpretation"),
            "selected_gate_hits": audit_rule.get("selected_gate_hits"),
            "good_state_hit_rate": audit_rule.get("good_state_hit_rate"),
            "direction_consistent_rate": audit_rule.get("direction_consistent_rate"),
            "direction_harmful_rate": audit_rule.get("direction_harmful_rate"),
            "hop_cycle_share": audit_rule.get("hop_cycle_share"),
            "prone_share": audit_rule.get("prone_share"),
        }
        notes = list(rule.get("audit_notes", []))
        notes.append(
            "保留于 corrected-label 初始规则库；依据修正 hop/prone 标签后的规则质量审计筛选。"
        )
        rule["audit_notes"] = notes
        kept.append(rule)

    keep_reason_counts = Counter(rule.get("corrected_label_keep_reason") for rule in kept)
    source_counts = Counter(rule.get("source") for rule in kept)
    origin_counts = Counter(rule.get("rule_origin", "closed_loop_or_legacy") for rule in kept)

    out = copy.deepcopy(source_payload) if isinstance(source_payload, dict) else {}
    out.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_version": "corrected_label_initial_v1",
            "online_enabled": any(rule.get("status") == "confidence_active" for rule in kept),
            "rules": kept,
            "corrected_label_initial_library": {
                "source_library": str(source_library),
                "quality_audit": str(quality_audit),
                "num_source_rules": len(source_rules),
                "num_audited_active_rules": len(audit_rules),
                "num_kept_rules": len(kept),
                "keep_reason_counts": dict(keep_reason_counts),
                "source_counts": dict(source_counts),
                "origin_counts": dict(origin_counts),
                "omitted_reason_counts": dict(omitted_reasons),
                "selection_policy": {
                    "keep_transition": "source == v8_transition_replay_safety and health != direction_suspect and selected_gate_hits > 0",
                    "keep_teacher_core": "rule_origin == teacher_core_standard_hop and health == healthy_direction and selected_gate_hits > 0",
                    "status_for_kept_rules": "confidence_active",
                },
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_markdown(output.with_suffix(".md"), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--quality-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    out = build(args.source_library, args.quality_audit, args.output)
    summary = out["corrected_label_initial_library"]
    print(f"wrote {args.output}")
    print(f"markdown {args.output.with_suffix('.md')}")
    print(
        "kept={num_kept_rules} sources={source_counts} origins={origin_counts}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
