#!/usr/bin/env python3
"""Run the multiseed Hopper closed loop with gray-certified online rules.

This wrapper leaves the existing TD-MPC2, SD, TVA, and retrospective rule
pipeline unchanged. It only intercepts the online rule library passed into each
cycle after the first one: the previous cycle's confidence library is audited by
the gray causal certificate layer, gray-filtered variants are exported, and the
`strict_active` variant is used as the next cycle's online rule prior.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SD_RETRO_DIR = REPO_ROOT / "experiments" / "sd_retrospective_rule_induction"
if str(SD_RETRO_DIR) not in sys.path:
    sys.path.insert(0, str(SD_RETRO_DIR))

import run_hopper_v7_multiseed_shared_rules as base  # noqa: E402


GRAY_VARIANT = os.environ.get("GRAY_RULE_VARIANT", "strict_active")
GRAY_AUDIT = REPO_ROOT / "experiments" / "gray_causal_rule_induction" / "audit_gray_rule_certificates.py"
GRAY_EXPORT = REPO_ROOT / "experiments" / "gray_causal_rule_induction" / "export_gray_filtered_rule_libraries.py"

_ORIGINAL_RUN_SEED_BATCH = base.run_seed_batch
_ORIGINAL_WRITE_JSON = base.write_json


def _path_if_exists(path: Path) -> list[str]:
    return [str(path)] if path.exists() else []


def _ensure_gray_output_root(output_root: Path) -> None:
    # The gray line is intentionally a separate experiment filesystem. Refusing
    # the old root prevents accidental overwrite of the ongoing SD mainline.
    if "sd_retrospective_rule_induction" in output_root.parts:
        raise ValueError(
            "gray causal closed-loop outputs must not be placed under "
            "results/sd_retrospective_rule_induction"
        )


def _update_cycle_config(args: Any, cycle: int, payload: dict[str, Any]) -> None:
    path = args.output_root / f"cycle_{cycle:03d}" / "cycle_config.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["gray_causal_rule_prior"] = payload
    _ORIGINAL_WRITE_JSON(path, data)


def build_gray_online_library(args: Any, *, source_cycle: int, source_rule_library: Path) -> Path:
    """Build and return the gray-filtered rule library used online."""

    _ensure_gray_output_root(args.output_root)
    paths = base.cycle_paths(args.output_root, source_cycle)
    gray_dir = paths.root / "gray_causal_rule_induction"
    variants_dir = gray_dir / "rule_library_variants"
    prefix = f"cycle{source_cycle:03d}"
    certificate_json = gray_dir / f"{prefix}_gray_rule_certificates.json"
    certificate_md = gray_dir / f"{prefix}_gray_rule_certificates.md"

    audit_cmd = [
        args.python_bin,
        str(GRAY_AUDIT),
        "--rule-library",
        str(source_rule_library),
        "--output-json",
        str(certificate_json),
        "--output-md",
        str(certificate_md),
        "--overwrite",
    ]
    optional_inputs = [
        ("--rule-effect-audit", paths.rule_effect_audit),
        ("--tva-shadow-update", paths.tva_shadow_update),
        ("--tva-consequence-gate", paths.tva_shadow_consequence_gated_update),
        ("--tva-terminal-gate", base.tva_gate_report_path(args, source_cycle)),
    ]
    for flag, path in optional_inputs:
        if path.exists():
            audit_cmd.extend([flag, str(path)])

    export_cmd = [
        args.python_bin,
        str(GRAY_EXPORT),
        "--rule-library",
        str(source_rule_library),
        "--certificates",
        str(certificate_json),
        "--output-dir",
        str(variants_dir),
        "--prefix",
        prefix,
        "--overwrite",
    ]

    print(" ".join(audit_cmd), flush=True)
    if not args.dry_run:
        subprocess.run(audit_cmd, cwd=REPO_ROOT, check=True)
    print(" ".join(export_cmd), flush=True)
    if not args.dry_run:
        subprocess.run(export_cmd, cwd=REPO_ROOT, check=True)

    selected = variants_dir / f"{prefix}_{GRAY_VARIANT}.json"
    if not args.dry_run and not selected.exists():
        raise FileNotFoundError(f"gray rule variant not found: {selected}")
    return selected


def run_seed_batch_with_gray(
    args: Any,
    cycle: int,
    seeds: list[int],
    rule_library: Path,
    resumes: dict[int, Path | None],
    sd_checkpoint: Path | None,
    tva_checkpoint: Path | None,
    tva_terminal_prior_policy: dict,
) -> dict[int, Path]:
    online_rule_library = rule_library
    gray_payload: dict[str, Any] | None = None
    if cycle > args.start_cycle:
        source_cycle = cycle - 1
        online_rule_library = build_gray_online_library(
            args,
            source_cycle=source_cycle,
            source_rule_library=rule_library,
        )
        gray_payload = {
            "enabled": True,
            "mechanism": "gray_causal_rule_certificate_v1",
            "selected_variant": GRAY_VARIANT,
            "source_cycle": source_cycle,
            "source_rule_library": str(rule_library),
            "online_rule_library": str(online_rule_library),
            "certificate_report": str(
                args.output_root
                / f"cycle_{source_cycle:03d}"
                / "gray_causal_rule_induction"
                / f"cycle{source_cycle:03d}_gray_rule_certificates.json"
            ),
        }
    else:
        gray_payload = {
            "enabled": False,
            "reason": "first cycle uses the initial teacher rule library before gray evidence exists",
            "online_rule_library": str(online_rule_library),
        }
    _update_cycle_config(args, cycle, gray_payload)
    return _ORIGINAL_RUN_SEED_BATCH(
        args,
        cycle,
        seeds,
        online_rule_library,
        resumes,
        sd_checkpoint,
        tva_checkpoint,
        tva_terminal_prior_policy,
    )


def write_json_with_gray_manifest(path: Path, payload: dict[str, Any]) -> None:
    out = payload
    if Path(path).name == "run_config.json":
        out = copy.deepcopy(payload)
        out["gray_causal_rule_mechanism"] = {
            "enabled": True,
            "online_variant": GRAY_VARIANT,
            "activation": "cycle_002_and_later",
            "certificate_audit": str(GRAY_AUDIT),
            "variant_export": str(GRAY_EXPORT),
            "output_policy": "all closed-loop outputs are expected under results/gray_causal_rule_induction",
        }
    _ORIGINAL_WRITE_JSON(path, out)


def main() -> None:
    base.run_seed_batch = run_seed_batch_with_gray
    base.write_json = write_json_with_gray_manifest
    base.main()


if __name__ == "__main__":
    main()
