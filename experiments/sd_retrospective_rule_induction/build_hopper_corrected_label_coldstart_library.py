"""Build a corrected-label Hopper cold-start initial rule library.

This script differs from ``build_hopper_corrected_label_initial_library.py``:
the earlier script was a compact quality filter for smoke testing.  Here the
research intent is cold-start coverage.  Teacher-core rules from high-quality
Hop trajectories are preserved, while corrected-label audit evidence adjusts
confidence instead of deleting most entry rules.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_ACTION_FAMILIES = {
    "hip_pos",
    "hip_neg",
    "knee_pos",
    "knee_neg",
    "ankle_pos",
    "ankle_neg",
    "waist_pos",
    "waist_neg",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rule_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return list(payload.get("rules", []))
    if isinstance(payload, list):
        return list(payload)
    raise TypeError(f"Unsupported rule payload type: {type(payload)!r}")


def _confidence(rule: dict[str, Any]) -> float:
    raw = rule.get("confidence", rule.get("effective_prior_scale", 0.56))
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.56


def _set_confidence(rule: dict[str, Any], confidence: float) -> None:
    confidence = round(max(0.0, min(1.0, float(confidence))), 6)
    rule["confidence"] = confidence
    rule["effective_prior_scale"] = confidence
    rule["status"] = "confidence_active"
    rule["confidence_status"] = "confidence_active"


def _valid_action_rule(rule: dict[str, Any]) -> bool:
    action = str(rule.get("action_family", ""))
    opposite = str(rule.get("opposite_action_family", ""))
    if action not in VALID_ACTION_FAMILIES:
        return False
    if opposite and opposite not in VALID_ACTION_FAMILIES:
        return False
    return True


def _teacher_confidence_policy(
    rule: dict[str, Any], audit_rule: dict[str, Any] | None
) -> tuple[float, str]:
    base = _confidence(rule)
    if audit_rule is None:
        # Missing corrected-label audit evidence usually means the rule did not
        # fire in the mature eval audit window.  It is still useful as a
        # cold-start entry rule because it came from high-quality teacher Hop
        # trajectories, but it should not remain a hard prior.
        return min(base, 0.90), "teacher_core_entry_preserved_audit_missing"

    health = str(audit_rule.get("health", ""))
    if health == "healthy_direction":
        return base, "teacher_core_corrected_label_healthy"
    if health == "mixed_or_low_support":
        return min(base, 0.65), "teacher_core_corrected_label_mixed_low_confidence"
    if health == "direction_suspect":
        return min(base, 0.35), "teacher_core_corrected_label_suspect_very_low_confidence"
    if health == "inactive":
        return min(base, 0.50), "teacher_core_corrected_label_inactive_low_confidence"
    return min(base, 0.56), f"teacher_core_corrected_label_{health or 'unknown'}"


def _transition_keep_policy(
    rule: dict[str, Any], audit_rule: dict[str, Any] | None
) -> tuple[float, str] | None:
    if audit_rule is None:
        return None
    if int(audit_rule.get("selected_gate_hits") or 0) <= 0:
        return None
    health = str(audit_rule.get("health", ""))
    if health == "direction_suspect":
        return None
    base = _confidence(rule)
    if health == "healthy_direction":
        return base, "transition_corrected_label_healthy"
    return min(base, 0.56), "transition_corrected_label_non_suspect"


def _attach_corrected_audit(rule: dict[str, Any], audit_rule: dict[str, Any] | None) -> None:
    if audit_rule is None:
        rule["corrected_label_quality"] = {"health": "audit_missing", "selected_gate_hits": 0}
        return
    fields = (
        "health",
        "interpretation",
        "selected_gate_hits",
        "good_state_hit_rate",
        "direction_consistent_rate",
        "direction_harmful_rate",
        "hop_cycle_share",
        "hop_share",
        "prone_share",
        "grounded_problem_share",
        "high_episode_hit_rate",
    )
    rule["corrected_label_quality"] = {key: audit_rule.get(key) for key in fields}


def _append_note(rule: dict[str, Any], note: str) -> None:
    notes = list(rule.get("audit_notes", []))
    notes.append(note)
    rule["audit_notes"] = notes


def _render_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["corrected_label_coldstart_library"]
    lines = [
        "# Corrected-Label Cold-Start Initial Rule Library",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "teacher_library",
        "closed_loop_library",
        "quality_audit",
        "num_rules",
    ):
        lines.append(f"- `{key}`: `{summary[key]}`")
    lines.append(f"- `source_counts`: `{summary['source_counts']}`")
    lines.append(f"- `origin_counts`: `{summary['origin_counts']}`")
    lines.append(f"- `keep_reason_counts`: `{summary['keep_reason_counts']}`")
    lines.append(f"- `health_counts`: `{summary['health_counts']}`")
    lines.append(f"- `confidence_summary`: `{summary['confidence_summary']}`")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(
        "该库在修正标签口径下重建，但以冷启动覆盖为目标。教师核心规则不因成熟评估窗口"
        "中未命中而被删除；修正标签审计用于调节信心和记录风险。transition 规则只保留"
        "非方向可疑项。"
    )
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append("| reason | source | origin | confidence | health | rule |")
    lines.append("|---|---|---|---:|---|---|")
    for rule in payload["rules"]:
        quality = rule.get("corrected_label_quality", {})
        lines.append(
            f"| `{rule.get('cold_start_keep_reason')}` | `{rule.get('source')}` | "
            f"`{rule.get('rule_origin', 'closed_loop_or_legacy')}` | "
            f"{float(rule.get('confidence', 0.0)):.3f} | `{quality.get('health')}` | "
            f"`{rule.get('id')}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build(
    teacher_library: Path,
    closed_loop_library: Path,
    quality_audit: Path,
    output: Path,
) -> dict[str, Any]:
    teacher_payload = _load_json(teacher_library)
    closed_loop_payload = _load_json(closed_loop_library)
    audit_payload = _load_json(quality_audit)

    audit_by_id = {str(rule.get("id")): rule for rule in audit_payload.get("rules", []) if rule.get("id")}
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    omitted: Counter[str] = Counter()

    for source_rule in _rule_list(teacher_payload):
        rule_id = str(source_rule.get("id", ""))
        if not rule_id or rule_id in seen:
            omitted["teacher_duplicate_or_missing_id"] += 1
            continue
        if not _valid_action_rule(source_rule):
            omitted["teacher_invalid_action_family"] += 1
            continue
        rule = copy.deepcopy(source_rule)
        audit_rule = audit_by_id.get(rule_id)
        confidence, reason = _teacher_confidence_policy(rule, audit_rule)
        _set_confidence(rule, confidence)
        rule["cold_start_keep_reason"] = reason
        rule["corrected_label_library_role"] = "teacher_core_cold_start_entry"
        _attach_corrected_audit(rule, audit_rule)
        _append_note(
            rule,
            "保留于 corrected-label cold-start 初始规则库；修正标签审计用于信心校准而非直接删除入口规则。",
        )
        rules.append(rule)
        seen.add(rule_id)

    for source_rule in _rule_list(closed_loop_payload):
        if source_rule.get("source") != "v8_transition_replay_safety":
            continue
        rule_id = str(source_rule.get("id", ""))
        if not rule_id or rule_id in seen:
            omitted["transition_duplicate_or_missing_id"] += 1
            continue
        if not _valid_action_rule(source_rule):
            omitted["transition_invalid_action_family"] += 1
            continue
        audit_rule = audit_by_id.get(rule_id)
        policy = _transition_keep_policy(source_rule, audit_rule)
        if policy is None:
            omitted["transition_not_non_suspect"] += 1
            continue
        confidence, reason = policy
        rule = copy.deepcopy(source_rule)
        _set_confidence(rule, confidence)
        rule["cold_start_keep_reason"] = reason
        rule["corrected_label_library_role"] = "transition_safety"
        _attach_corrected_audit(rule, audit_rule)
        _append_note(
            rule,
            "保留于 corrected-label cold-start 初始规则库；transition 规则需在修正标签审计中非方向可疑。",
        )
        rules.append(rule)
        seen.add(rule_id)

    source_counts = Counter(str(rule.get("source")) for rule in rules)
    origin_counts = Counter(str(rule.get("rule_origin", "closed_loop_or_legacy")) for rule in rules)
    keep_reason_counts = Counter(str(rule.get("cold_start_keep_reason")) for rule in rules)
    health_counts = Counter(
        str(rule.get("corrected_label_quality", {}).get("health", "unknown")) for rule in rules
    )
    confidences = [_confidence(rule) for rule in rules]
    confidence_summary = {
        "min": round(min(confidences), 6) if confidences else None,
        "mean": round(sum(confidences) / len(confidences), 6) if confidences else None,
        "max": round(max(confidences), 6) if confidences else None,
    }

    payload: dict[str, Any] = {
        "rule_family": "hopper_corrected_label_coldstart_initial_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "online_enabled": bool(rules),
        "library_version": "corrected_label_coldstart_initial_v2",
        "compatibility": {
            "online_prior": "online_rule_prior_v6",
            "allowed_statuses": ["confidence_active"],
            "requires_corrected_hop_label": True,
        },
        "status_meaning": {
            "confidence_active": "online-readable rule; confidence controls prior scale and stochastic-gate probability",
            "confidence_observing": "tracked but not used by confidence_active-only online prior",
            "confidence_retired": "not used online",
        },
        "rules": rules,
        "corrected_label_coldstart_library": {
            "teacher_library": str(teacher_library),
            "closed_loop_library": str(closed_loop_library),
            "quality_audit": str(quality_audit),
            "num_rules": len(rules),
            "source_counts": dict(source_counts),
            "origin_counts": dict(origin_counts),
            "keep_reason_counts": dict(keep_reason_counts),
            "health_counts": dict(health_counts),
            "confidence_summary": confidence_summary,
            "omitted_reason_counts": dict(omitted),
            "selection_policy": {
                "teacher_core": (
                    "preserve valid teacher_core_standard_hop action-family rules for cold-start coverage; "
                    "use corrected-label audit to calibrate confidence"
                ),
                "transition": (
                    "add v8_transition_replay_safety rules only when corrected-label audit is non-suspect "
                    "and selected_gate_hits > 0"
                ),
                "invalid_action_family": "drop action families that cannot be mapped by online_rule_prior_v6",
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_markdown(output.with_suffix(".md"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-library", type=Path, required=True)
    parser.add_argument("--closed-loop-library", type=Path, required=True)
    parser.add_argument("--quality-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build(
        teacher_library=args.teacher_library,
        closed_loop_library=args.closed_loop_library,
        quality_audit=args.quality_audit,
        output=args.output,
    )
    summary = payload["corrected_label_coldstart_library"]
    print(f"wrote {args.output}")
    print(f"markdown {args.output.with_suffix('.md')}")
    print(
        "rules={num_rules} sources={source_counts} health={health_counts} confidence={confidence_summary}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
