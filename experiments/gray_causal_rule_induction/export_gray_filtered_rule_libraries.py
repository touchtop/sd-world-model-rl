#!/usr/bin/env python3
"""Export rule-library variants from gray causal rule certificates.

The script only writes derived libraries to a new output directory. It never
edits the source rule library.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Callable

try:
    from .gray_rule_graph import dump_json, load_json, rule_id, source_of
except ImportError:  # pragma: no cover - direct script execution
    from gray_rule_graph import dump_json, load_json, rule_id, source_of  # type: ignore


CERTIFIED_TYPES = {"strict_action_certificate", "value_certificate"}
TEACHER_OR_V8_SOURCES = {
    "sd_tva_counterfactual_teacher_rich_init_v1",
    "sd_tva_counterfactual_teacher_core_v1",
    "sd_tva_broad_context_counterfactual",
    "teacher_core_standard_hop",
    "v8_transition_replay_safety",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create gray-certificate-filtered Hopper rule library variants."
    )
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="gray_filtered")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_rules(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return [r for r in payload["rules"] if isinstance(r, dict)]
    raise ValueError("Unsupported rule library format; expected list or object with rules.")


def replace_rules(template: Any, rules: list[dict[str, Any]]) -> Any:
    """Preserve the source library container expected by the online prior loader."""

    if isinstance(template, dict) and isinstance(template.get("rules"), list):
        out = copy.deepcopy(template)
        out["rules"] = rules
        out["gray_filter_export"] = {
            "preserved_source_container": True,
            "note": "Only rules were transformed; top-level metadata was retained for loader compatibility.",
        }
        return out
    return rules


def certificate_map(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("certificates"), list):
        raise ValueError("Certificate report must contain a certificates list.")
    out: dict[str, dict[str, Any]] = {}
    for cert in payload["certificates"]:
        if isinstance(cert, dict) and cert.get("id"):
            out[str(cert["id"])] = cert
    return out


def guard_output_dir(path: Path) -> None:
    if "sd_retrospective_rule_induction" in path.parts:
        raise ValueError(
            "Refusing to write derived gray libraries under sd_retrospective_rule_induction."
        )
    path.mkdir(parents=True, exist_ok=True)


def is_active(rule: dict[str, Any]) -> bool:
    return rule.get("status") == "confidence_active"


def is_certified(cert: dict[str, Any] | None) -> bool:
    return bool(cert and cert.get("certificate_type") in CERTIFIED_TYPES)


def is_strict(cert: dict[str, Any] | None) -> bool:
    return bool(cert and cert.get("certificate_type") == "strict_action_certificate")


def is_reject(cert: dict[str, Any] | None) -> bool:
    return bool(cert and cert.get("certificate_type") == "reject_or_retire")


def is_teacher_or_v8(rule: dict[str, Any]) -> bool:
    return source_of(rule) in TEACHER_OR_V8_SOURCES


def demote(rule: dict[str, Any], reason: str) -> dict[str, Any]:
    out = copy.deepcopy(rule)
    out["gray_filter_original_status"] = rule.get("status")
    out["gray_filter_decision"] = "demoted"
    out["gray_filter_reason"] = reason
    out["status"] = "confidence_retired"
    out["effective_prior_scale"] = 0.0
    return out


def keep(rule: dict[str, Any], reason: str) -> dict[str, Any]:
    out = copy.deepcopy(rule)
    out["gray_filter_original_status"] = rule.get("status")
    out["gray_filter_decision"] = "kept"
    out["gray_filter_reason"] = reason
    return out


VariantPredicate = Callable[[dict[str, Any], dict[str, Any] | None], tuple[bool, str]]


def export_variant(
    *,
    name: str,
    rules: list[dict[str, Any]],
    source_payload: Any,
    certs: dict[str, dict[str, Any]],
    predicate: VariantPredicate,
    output_dir: Path,
    prefix: str,
    overwrite: bool,
) -> dict[str, Any]:
    derived: list[dict[str, Any]] = []
    demoted_active = 0
    kept_active = 0
    for rule in rules:
        cert = certs.get(rule_id(rule))
        should_keep, reason = predicate(rule, cert)
        if should_keep:
            out = keep(rule, reason)
            if is_active(rule):
                kept_active += 1
        else:
            out = demote(rule, reason)
            if is_active(rule):
                demoted_active += 1
        derived.append(out)

    status_counts = Counter(str(r.get("status")) for r in derived)
    cert_counts = Counter(
        str(certs.get(rule_id(r), {}).get("certificate_type", "missing")) for r in rules
    )
    path = output_dir / f"{prefix}_{name}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")
    dump_json(replace_rules(source_payload, derived), path)
    return {
        "variant": name,
        "path": str(path),
        "num_rules": len(derived),
        "kept_active": kept_active,
        "demoted_active": demoted_active,
        "status_counts": dict(sorted(status_counts.items())),
        "source_counts": dict(Counter(source_of(r) for r in derived).most_common()),
        "input_certificate_counts": dict(sorted(cert_counts.items())),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gray Filtered Rule Libraries",
        "",
        f"Source rule library: `{summary['source_rule_library']}`",
        f"Certificate report: `{summary['certificate_report']}`",
        "",
        "| variant | kept active | demoted active | output |",
        "|---|---:|---:|---|",
    ]
    for item in summary["variants"]:
        lines.append(
            "| {variant} | {kept} | {demoted} | `{path}` |".format(
                variant=item["variant"],
                kept=item["kept_active"],
                demoted=item["demoted_active"],
                path=item["path"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    guard_output_dir(output_dir)

    source_payload = load_json(args.rule_library)
    rules = extract_rules(source_payload)
    certs = certificate_map(load_json(args.certificates))

    variants: list[tuple[str, VariantPredicate]] = [
        (
            "drop_reject_active",
            lambda rule, cert: (
                (not is_active(rule)) or (not is_reject(cert)),
                "drop only active reject_or_retire rules",
            ),
        ),
        (
            "certified_active",
            lambda rule, cert: (
                (not is_active(rule)) or is_certified(cert),
                "keep active rules with strict/value certificate",
            ),
        ),
        (
            "strict_active",
            lambda rule, cert: (
                (not is_active(rule)) or is_strict(cert),
                "keep active rules with strict action certificate",
            ),
        ),
        (
            "teacher_v8_active",
            lambda rule, cert: (
                (not is_active(rule)) or is_teacher_or_v8(rule),
                "keep active teacher/v8-transition sources",
            ),
        ),
        (
            "teacher_v8_certified_active",
            lambda rule, cert: (
                (not is_active(rule)) or (is_teacher_or_v8(rule) and is_certified(cert)),
                "keep active teacher/v8-transition sources with strict/value certificate",
            ),
        ),
    ]

    summary = {
        "source_rule_library": args.rule_library,
        "certificate_report": args.certificates,
        "prefix": args.prefix,
        "variants": [
            export_variant(
                name=name,
                rules=rules,
                source_payload=source_payload,
                certs=certs,
                predicate=predicate,
                output_dir=output_dir,
                prefix=args.prefix,
                overwrite=args.overwrite,
            )
            for name, predicate in variants
        ],
    }
    summary_path = output_dir / f"{args.prefix}_summary.json"
    md_path = output_dir / f"{args.prefix}_summary.md"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {summary_path}")
    if md_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {md_path}")
    dump_json(summary, summary_path)
    write_markdown(summary, md_path)
    print(f"wrote variants: {summary_path}")
    for item in summary["variants"]:
        print(
            "{variant}: kept_active={kept_active} demoted_active={demoted_active} path={path}".format(
                **item
            )
        )


if __name__ == "__main__":
    main()
