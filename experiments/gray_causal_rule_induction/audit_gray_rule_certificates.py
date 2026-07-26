#!/usr/bin/env python3
"""CLI for read-only gray causal rule certificate audits."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .gray_rule_graph import (
        CertificateThresholds,
        GrayRuleCertificateBuilder,
        dump_json,
        load_json,
        render_markdown,
    )
except ImportError:  # pragma: no cover - direct script execution
    from gray_rule_graph import (  # type: ignore
        CertificateThresholds,
        GrayRuleCertificateBuilder,
        dump_json,
        load_json,
        render_markdown,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build read-only gray causal certificates for an existing Hopper "
            "rule library."
        )
    )
    parser.add_argument("--rule-library", required=True, help="Existing rule library JSON.")
    parser.add_argument("--rule-effect-audit", help="Existing rule_effect_audit.json.")
    parser.add_argument("--tva-shadow-update", help="Existing tva_shadow_update.json.")
    parser.add_argument(
        "--tva-consequence-gate",
        help="Existing tva_shadow_consequence_gated_update.json.",
    )
    parser.add_argument("--tva-terminal-gate", help="Existing tva_terminal_prior_gate.json.")
    parser.add_argument("--output-json", required=True, help="New gray certificate output JSON.")
    parser.add_argument("--output-md", help="Optional new markdown summary output.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing outputs.")
    parser.add_argument("--max-md-rules", type=int, default=40)

    parser.add_argument("--min-support", type=int, default=CertificateThresholds.min_support)
    parser.add_argument(
        "--min-counterfactual-records",
        type=int,
        default=CertificateThresholds.min_counterfactual_records,
    )
    parser.add_argument("--max-entropy", type=float, default=CertificateThresholds.max_entropy)
    parser.add_argument(
        "--max-direction-harmful-rate",
        type=float,
        default=CertificateThresholds.max_direction_harmful_rate,
    )
    return parser.parse_args()


def guard_output_path(path: Path, overwrite: bool) -> None:
    if "sd_retrospective_rule_induction" in path.parts:
        raise ValueError(
            "Refusing to write gray causal outputs under sd_retrospective_rule_induction."
        )
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md) if args.output_md else None
    guard_output_path(output_json, args.overwrite)
    if output_md is not None:
        guard_output_path(output_md, args.overwrite)

    thresholds = CertificateThresholds(
        min_support=args.min_support,
        min_counterfactual_records=args.min_counterfactual_records,
        max_entropy=args.max_entropy,
        max_direction_harmful_rate=args.max_direction_harmful_rate,
    )
    builder = GrayRuleCertificateBuilder(thresholds)
    inputs = {
        "rule_library": args.rule_library,
        "rule_effect_audit": args.rule_effect_audit or "",
        "tva_shadow_update": args.tva_shadow_update or "",
        "tva_consequence_gate": args.tva_consequence_gate or "",
        "tva_terminal_gate": args.tva_terminal_gate or "",
    }
    report = builder.build_report(
        rule_library=load_json(args.rule_library),
        rule_effect_audit=load_json(args.rule_effect_audit),
        tva_shadow_update=load_json(args.tva_shadow_update),
        tva_consequence_gate=load_json(args.tva_consequence_gate),
        tva_terminal_gate=load_json(args.tva_terminal_gate),
        inputs=inputs,
    )
    dump_json(report, output_json)
    if output_md is not None:
        output_md.write_text(
            render_markdown(report, max_rules=args.max_md_rules),
            encoding="utf-8",
        )
    summary = report["summary"]
    print(
        "wrote gray certificates: "
        f"{output_json} rules={summary['num_rules']} "
        f"counts={summary['certificate_counts']}"
    )
    if output_md is not None:
        print(f"markdown: {output_md}")


if __name__ == "__main__":
    main()

