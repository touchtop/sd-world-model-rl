"""Gray causal certificates for Hopper rule generation.

This module keeps the current SD model intact. It treats existing SD,
replay, and TVA audit outputs as evidence for gray rule edges:

    context, do(action_family) -> short-horizon outcome/value

The output is an auditable certificate suggestion. It is not written back to
the online rule library.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


TRUSTED_MECHANISM_SOURCES = {
    "sd_tva_broad_context_counterfactual",
    "sd_tva_counterfactual_teacher_core_v1",
    "sd_tva_counterfactual_teacher_rich_init_v1",
    "teacher_core_standard_hop",
    "v8_transition_replay_safety",
}


@dataclass(frozen=True)
class CertificateThresholds:
    """Thresholds for the first read-only certificate pass."""

    min_support: int = 20
    min_counterfactual_records: int = 20
    max_entropy: float = 0.65
    min_direction_consistent_rate: float = 0.60
    max_direction_harmful_rate: float = 0.45
    min_positive_reward_hit_rate: float = 0.10
    min_high_episode_hit_rate: float = 0.10
    min_good_outcome_rate: float = 0.10
    min_tva_effect_hit_ratio: float = 0.003
    min_tva_direction_margin: float = 0.05
    max_tva_opposite_rate: float = 0.45
    strict_prior_cap: float = 0.70
    value_prior_cap: float = 0.55
    shadow_prior_cap: float = 0.25


def load_json(path: str | Path | None) -> Any:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(payload: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def extract_rules(payload: Any) -> list[dict[str, Any]]:
    """Extract a rule list from either a list payload or an object payload."""

    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        rules = payload.get("rules")
        if isinstance(rules, list):
            return [r for r in rules if isinstance(r, dict)]
    return []


def index_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        rid = rule_id(record)
        if rid:
            indexed[rid] = record
    return indexed


def extract_rule_effect_summaries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("rule_summaries")
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
    return []


def rule_id(rule: dict[str, Any]) -> str:
    return str(rule.get("id") or rule.get("rule_id") or "")


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def source_of(rule: dict[str, Any]) -> str:
    return str(
        first_present(
            rule.get("rule_origin"),
            rule.get("source"),
            rule.get("teacher_source"),
            rule.get("origin"),
            "",
        )
    )


def context_key(rule: dict[str, Any]) -> str:
    return str(
        first_present(
            rule.get("rule_key"),
            rule.get("substate_key"),
            rule.get("state_event"),
            rule.get("event"),
            "",
        )
    )


def action_family(rule: dict[str, Any]) -> str:
    return str(rule.get("action_family") or "")


def current_prior_scale(rule: dict[str, Any]) -> float:
    return as_float(
        first_present(rule.get("effective_prior_scale"), rule.get("confidence")),
        0.0,
    )


def confidence(rule: dict[str, Any]) -> float:
    return as_float(rule.get("confidence"), 0.0)


def consequence_rejected(consequence_rule: dict[str, Any] | None) -> bool:
    if not consequence_rule:
        return False
    gate = consequence_rule.get("consequence_gate")
    if not isinstance(gate, dict):
        return False
    decision = str(gate.get("decision") or gate.get("status") or "").lower()
    return "reject" in decision or gate.get("passed") is False


class GrayRuleCertificateBuilder:
    """Build gray causal rule certificates from existing audit evidence."""

    def __init__(self, thresholds: CertificateThresholds | None = None) -> None:
        self.thresholds = thresholds or CertificateThresholds()

    def build_report(
        self,
        *,
        rule_library: Any,
        rule_effect_audit: Any = None,
        tva_shadow_update: Any = None,
        tva_consequence_gate: Any = None,
        tva_terminal_gate: Any = None,
        inputs: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        rules = extract_rules(rule_library)
        effect_by_id = index_by_id(extract_rule_effect_summaries(rule_effect_audit))
        shadow_by_id = index_by_id(extract_rules(tva_shadow_update))
        consequence_by_id = index_by_id(extract_rules(tva_consequence_gate))

        certificates = [
            self.build_certificate(
                rule,
                effect=effect_by_id.get(rule_id(rule)),
                tva_shadow=shadow_by_id.get(rule_id(rule)),
                consequence=consequence_by_id.get(rule_id(rule)),
            )
            for rule in rules
        ]

        return {
            "audit_family": "gray_causal_rule_certificate_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs or {},
            "thresholds": asdict(self.thresholds),
            "global_tva_gate": self._global_tva_gate_summary(tva_terminal_gate),
            "summary": self._summary(certificates),
            "certificates": certificates,
        }

    def build_certificate(
        self,
        rule: dict[str, Any],
        *,
        effect: dict[str, Any] | None = None,
        tva_shadow: dict[str, Any] | None = None,
        consequence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mechanism = self._mechanism(rule, effect, tva_shadow)
        action = self._action_stability(rule, effect, tva_shadow)
        value = self._value_usable(rule, effect, tva_shadow)
        safety = self._structure_safety(effect, tva_shadow, consequence)
        recommendation = self._recommendation(mechanism, action, value, safety)
        prior_scale = self._recommended_prior_scale(rule, recommendation)

        return {
            "id": rule_id(rule),
            "source": source_of(rule),
            "status": rule.get("status"),
            "current_confidence": confidence(rule),
            "current_effective_prior_scale": current_prior_scale(rule),
            "gray_edge": {
                "context": context_key(rule),
                "action_family": action_family(rule),
                "intervention_type": rule.get("intervention_type"),
                "outcome_nodes": [
                    "support_contact",
                    "full_foot_support",
                    "grounded_hop",
                    "rotation_penalty",
                    "body_collision",
                    "vx",
                    "tva_value",
                ],
            },
            "mechanism_admissibility": mechanism,
            "action_stability": action,
            "value_usability": value,
            "structure_safety": safety,
            "certificate_type": recommendation,
            "recommended_prior_scale": prior_scale,
        }

    def _mechanism(
        self,
        rule: dict[str, Any],
        effect: dict[str, Any] | None,
        tva_shadow: dict[str, Any] | None,
    ) -> dict[str, Any]:
        t = self.thresholds
        src = source_of(rule)
        rid = rule_id(rule)
        has_context = bool(context_key(rule) or "->" in rid)
        has_action = bool(action_family(rule) and rule.get("intervention_type"))
        support = max(
            as_int(rule.get("support")),
            as_int(rule.get("context_support")),
            as_int(rule.get("num_counterfactual_records")),
        )
        sd_supported = support >= t.min_counterfactual_records or src in TRUSTED_MECHANISM_SOURCES

        replay_hits = as_int(effect.get("selected_gate_hits") if effect else None)
        replay_supported = replay_hits >= t.min_support

        tva_hits = max(
            as_int(tva_shadow.get("tva_shadow_selected_gate_hits") if tva_shadow else None),
            as_int(tva_shadow.get("tva_shadow_effect_selected_gate_hits") if tva_shadow else None),
        )
        tva_effect_ratio = as_float(
            tva_shadow.get("tva_shadow_effect_hit_ratio") if tva_shadow else None
        )
        tva_advantage = as_float(
            tva_shadow.get("tva_shadow_mean_advantage") if tva_shadow else None
        )
        tva_same = as_float(tva_shadow.get("tva_shadow_same_rate") if tva_shadow else None)
        tva_opp = as_float(tva_shadow.get("tva_shadow_opposite_rate") if tva_shadow else None)
        tva_margin = tva_same - tva_opp
        tva_supported = (
            tva_effect_ratio >= t.min_tva_effect_hit_ratio
            or abs(tva_advantage) > 1e-9
            or abs(tva_margin) >= t.min_tva_direction_margin
        )

        evidence_sources: list[str] = []
        if sd_supported:
            evidence_sources.append("sd_or_teacher_counterfactual")
        if replay_supported:
            evidence_sources.append("replay_rule_effect")
        if tva_supported:
            evidence_sources.append("tva_action_value")

        failures: list[str] = []
        if not has_context:
            failures.append("missing_explicit_context")
        if not has_action:
            failures.append("missing_action_family")
        if not evidence_sources:
            failures.append("missing_mechanism_evidence")

        passed = has_context and has_action and bool(evidence_sources)
        return {
            "passed": passed,
            "context_object_present": has_context,
            "action_object_present": has_action,
            "support": support,
            "replay_selected_gate_hits": replay_hits,
            "tva_selected_gate_hits": tva_hits,
            "evidence_sources": evidence_sources,
            "failures": failures,
        }

    def _action_stability(
        self,
        rule: dict[str, Any],
        effect: dict[str, Any] | None,
        tva_shadow: dict[str, Any] | None,
    ) -> dict[str, Any]:
        t = self.thresholds
        entropy = as_float(
            first_present(
                tva_shadow.get("tva_shadow_best_family_entropy") if tva_shadow else None,
                rule.get("best_family_entropy"),
            ),
            1.0,
        )
        direction_consistent = as_float(
            effect.get("direction_consistent_rate") if effect else None,
            0.0,
        )
        tva_same = as_float(tva_shadow.get("tva_shadow_same_rate") if tva_shadow else None)
        tva_opp = as_float(tva_shadow.get("tva_shadow_opposite_rate") if tva_shadow else None)
        tva_margin = tva_same - tva_opp

        entropy_passed = entropy <= t.max_entropy
        replay_passed = direction_consistent >= t.min_direction_consistent_rate
        tva_passed = tva_margin >= t.min_tva_direction_margin
        passed = entropy_passed or replay_passed or tva_passed

        stability_score = max(
            1.0 - min(max(entropy, 0.0), 1.0),
            direction_consistent,
            max(tva_margin, 0.0),
        )
        action_span_proxy = max(0.0, 1.0 - stability_score)

        return {
            "passed": passed,
            "best_family_entropy": entropy,
            "direction_consistent_rate": direction_consistent,
            "tva_same_rate": tva_same,
            "tva_opposite_rate": tva_opp,
            "tva_direction_margin": tva_margin,
            "action_span_proxy": action_span_proxy,
            "evidence": {
                "entropy_passed": entropy_passed,
                "replay_direction_passed": replay_passed,
                "tva_margin_passed": tva_passed,
            },
        }

    def _value_usable(
        self,
        rule: dict[str, Any],
        effect: dict[str, Any] | None,
        tva_shadow: dict[str, Any] | None,
    ) -> dict[str, Any]:
        t = self.thresholds
        positive_rate = as_float(effect.get("positive_reward_hit_rate") if effect else None)
        high_episode_rate = as_float(effect.get("high_episode_hit_rate") if effect else None)
        good_outcome_rate = as_float(rule.get("good_outcome_rate"))
        tva_advantage = as_float(
            tva_shadow.get("tva_shadow_mean_advantage") if tva_shadow else None
        )

        replay_value_passed = (
            positive_rate >= t.min_positive_reward_hit_rate
            or high_episode_rate >= t.min_high_episode_hit_rate
        )
        rule_value_passed = good_outcome_rate >= t.min_good_outcome_rate
        tva_value_passed = tva_advantage > 0.0
        passed = replay_value_passed or rule_value_passed or tva_value_passed

        return {
            "passed": passed,
            "positive_reward_hit_rate": positive_rate,
            "high_episode_hit_rate": high_episode_rate,
            "good_outcome_rate": good_outcome_rate,
            "tva_mean_advantage": tva_advantage,
            "evidence": {
                "replay_value_passed": replay_value_passed,
                "rule_value_passed": rule_value_passed,
                "tva_value_passed": tva_value_passed,
            },
        }

    def _structure_safety(
        self,
        effect: dict[str, Any] | None,
        tva_shadow: dict[str, Any] | None,
        consequence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        t = self.thresholds
        harmful_rate = as_float(effect.get("direction_harmful_rate") if effect else None)
        good_state_hit_rate = as_float(effect.get("good_state_hit_rate") if effect else None)
        tva_opp = as_float(tva_shadow.get("tva_shadow_opposite_rate") if tva_shadow else None)
        rejected = consequence_rejected(consequence)

        passed = (
            harmful_rate <= t.max_direction_harmful_rate
            and tva_opp <= t.max_tva_opposite_rate
            and not rejected
        )
        return {
            "passed": passed,
            "direction_harmful_rate": harmful_rate,
            "good_state_hit_rate": good_state_hit_rate,
            "tva_opposite_rate": tva_opp,
            "consequence_rejected": rejected,
        }

    def _recommendation(
        self,
        mechanism: dict[str, Any],
        action: dict[str, Any],
        value: dict[str, Any],
        safety: dict[str, Any],
    ) -> str:
        if not safety["passed"]:
            return "reject_or_retire"
        if not mechanism["passed"]:
            return "shadow_only"
        if action["passed"]:
            return "strict_action_certificate"
        if value["passed"]:
            return "value_certificate"
        return "shadow_only"

    def _recommended_prior_scale(self, rule: dict[str, Any], recommendation: str) -> float:
        t = self.thresholds
        base = max(current_prior_scale(rule), confidence(rule))
        if recommendation == "strict_action_certificate":
            return round(min(t.strict_prior_cap, max(base, 0.55)), 6)
        if recommendation == "value_certificate":
            return round(min(t.value_prior_cap, max(base * 0.8, 0.30)), 6)
        if recommendation == "shadow_only":
            return round(min(t.shadow_prior_cap, base * 0.5), 6)
        return 0.0

    def _global_tva_gate_summary(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        decision = payload.get("decision")
        return {
            "passed": payload.get("passed"),
            "strong_passed": payload.get("strong_passed"),
            "weak_passed": payload.get("weak_passed"),
            "decision": decision if isinstance(decision, dict) else {},
        }

    def _summary(self, certificates: list[dict[str, Any]]) -> dict[str, Any]:
        by_type = Counter(c["certificate_type"] for c in certificates)
        by_status = Counter(str(c.get("status")) for c in certificates)
        by_source = Counter(str(c.get("source")) for c in certificates)
        active = [c for c in certificates if c.get("status") == "confidence_active"]
        return {
            "num_rules": len(certificates),
            "certificate_counts": dict(sorted(by_type.items())),
            "status_counts": dict(sorted(by_status.items())),
            "source_counts": dict(by_source.most_common()),
            "active_certificate_counts": dict(
                sorted(Counter(c["certificate_type"] for c in active).items())
            ),
            "mean_recommended_prior_scale": (
                sum(c["recommended_prior_scale"] for c in certificates) / len(certificates)
                if certificates
                else 0.0
            ),
        }


def render_markdown(report: dict[str, Any], *, max_rules: int = 40) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Gray Rule Certificate Audit",
        "",
        f"Audit family: `{report.get('audit_family')}`",
        f"Created at UTC: `{report.get('created_at_utc')}`",
        "",
        "## Summary",
        "",
        f"- Rules: `{summary.get('num_rules', 0)}`",
        f"- Certificate counts: `{summary.get('certificate_counts', {})}`",
        f"- Active certificate counts: `{summary.get('active_certificate_counts', {})}`",
        f"- Mean recommended prior scale: `{summary.get('mean_recommended_prior_scale', 0.0):.4f}`",
        "",
        "## Top Certificate Rows",
        "",
        "| type | status | scale | source | rule |",
        "|---|---|---:|---|---|",
    ]

    certificates = report.get("certificates", [])
    rank = {
        "strict_action_certificate": 0,
        "value_certificate": 1,
        "shadow_only": 2,
        "reject_or_retire": 3,
    }
    ordered = sorted(
        certificates,
        key=lambda c: (
            rank.get(c.get("certificate_type"), 9),
            -as_float(c.get("recommended_prior_scale")),
            str(c.get("id")),
        ),
    )
    for cert in ordered[:max_rules]:
        rid = str(cert.get("id", "")).replace("|", "\\|")
        lines.append(
            "| {type} | {status} | {scale:.3f} | {source} | `{rule}` |".format(
                type=cert.get("certificate_type"),
                status=cert.get("status"),
                scale=as_float(cert.get("recommended_prior_scale")),
                source=cert.get("source"),
                rule=rid,
            )
        )
    lines.append("")
    return "\n".join(lines)
