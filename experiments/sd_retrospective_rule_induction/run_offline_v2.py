#!/usr/bin/env python3
"""Run offline v2: fit SD explainer, run explainer counterfactuals, induce state-action rules."""

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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-transitions", type=int, default=50000)
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = args.repo_root.resolve()
    selection = root / "results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_selected_trajectories.json"
    attributions = root / "results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_attributions.json"
    explainer = root / "results/sd_retrospective_rule_induction/sd_explainers/hopper_offline_v2_sd_explainer.pt"
    explainer_summary = root / "results/sd_retrospective_rule_induction/sd_explainers/hopper_offline_v2_sd_explainer_summary.json"
    counterfactuals = root / "results/sd_retrospective_rule_induction/attributions/hopper_offline_v2_counterfactuals_explainer.json"
    rules_v2 = root / "results/sd_retrospective_rule_induction/rules/hopper_offline_v2_state_action_rules.json"
    consistency_v3 = root / "results/sd_retrospective_rule_induction/reports/hopper_offline_v3_rule_consistency.json"
    reliable_v3 = root / "results/sd_retrospective_rule_induction/rules/hopper_offline_v3_reliable_rules.json"
    substate_v4 = root / "results/sd_retrospective_rule_induction/rules/hopper_offline_v4_substate_rules.json"
    replay_v4 = root / "results/sd_retrospective_rule_induction/reports/hopper_offline_v4_replay_rule_scoring.json"
    library = root / "results/sd_retrospective_rule_induction/rules/library.json"
    substate_validation_v5 = root / "results/sd_retrospective_rule_induction/reports/hopper_offline_v5_substate_rule_validation.json"
    library_v5 = root / "results/sd_retrospective_rule_induction/rules/library_v5.json"

    if not selection.exists() or not attributions.exists():
        run_step(script_dir, "run_offline_v1.py", ["--seed", str(args.seed)])

    run_step(
        script_dir,
        "fit_sd_explainer.py",
        [
            "--selection", str(selection),
            "--output", str(explainer),
            "--summary", str(explainer_summary),
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--max-transitions", str(args.max_transitions),
            "--use-transition-residual",
            "--transition-residual-scale", "1.0",
        ],
    )
    cf_args = [
        "--attributions", str(attributions),
        "--output", str(counterfactuals),
        "--model-checkpoint", str(explainer),
    ]
    if args.max_episodes > 0:
        cf_args.extend(["--max-episodes", str(args.max_episodes)])
    run_step(script_dir, "counterfactual_actions.py", cf_args)
    run_step(
        script_dir,
        "induce_state_action_rules_v2.py",
        ["--counterfactuals", str(counterfactuals), "--output", str(rules_v2)],
    )
    run_step(
        script_dir,
        "analyze_rule_consistency.py",
        [
            "--counterfactuals", str(counterfactuals),
            "--rules", str(rules_v2),
            "--output", str(consistency_v3),
            "--reliable-output", str(reliable_v3),
        ],
    )
    run_step(
        script_dir,
        "analyze_substate_rules.py",
        ["--counterfactuals", str(counterfactuals), "--output", str(substate_v4)],
    )
    run_step(
        script_dir,
        "replay_rule_scoring.py",
        ["--attributions", str(attributions), "--rules", str(reliable_v3), "--output", str(replay_v4)],
    )
    run_step(
        script_dir,
        "build_rule_library.py",
        ["--reliable-rules", str(reliable_v3), "--substate-rules", str(substate_v4), "--output", str(library)],
    )
    run_step(
        script_dir,
        "validate_substate_rules.py",
        ["--attributions", str(attributions), "--rules", str(substate_v4), "--output", str(substate_validation_v5)],
    )
    run_step(
        script_dir,
        "update_rule_library_status.py",
        [
            "--library", str(library),
            "--replay", str(replay_v4),
            "--substate-validation", str(substate_validation_v5),
            "--output", str(library_v5),
        ],
    )


if __name__ == "__main__":
    main()
