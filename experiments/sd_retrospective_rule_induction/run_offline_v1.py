#!/usr/bin/env python3
"""Run the full offline v1 Hopper retrospective rule-induction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_dir: Path, script_name: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script_dir / script_name), *extra_args]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--holdout-seed", type=int, action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root.resolve()
    manifest = repo_root / "results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_manifest.json"
    selection = repo_root / "results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_selected_trajectories.json"
    attributions = repo_root / "results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_attributions.json"
    counterfactuals = repo_root / "results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.json"
    rules = repo_root / "results/sd_retrospective_rule_induction/rules/hopper_offline_v1_rules.json"
    validation = repo_root / "results/sd_retrospective_rule_induction/reports/hopper_offline_v1_rule_validation.json"

    run_step(script_dir, "build_hopper_manifest.py", ["--output", str(manifest)])
    run_step(
        script_dir,
        "stratify_trajectories.py",
        ["--manifest", str(manifest), "--output", str(selection), "--seed", str(args.seed)],
    )
    run_step(
        script_dir,
        "attribute_failures.py",
        ["--selection", str(selection), "--output", str(attributions)],
    )
    run_step(
        script_dir,
        "counterfactual_actions.py",
        ["--attributions", str(attributions), "--output", str(counterfactuals)],
    )
    run_step(
        script_dir,
        "induce_rules.py",
        ["--attributions", str(attributions), "--output", str(rules)],
    )
    holdout_args: list[str] = []
    holdout_seeds = args.holdout_seed or [3]
    for seed in holdout_seeds:
        holdout_args.extend(["--holdout-seed", str(seed)])
    run_step(
        script_dir,
        "validate_rules.py",
        ["--attributions", str(attributions), "--rules", str(rules), "--output", str(validation), *holdout_args],
    )


if __name__ == "__main__":
    main()
