#!/usr/bin/env python3
"""Configure a short resumed run for explainability-v0 Hopper rules.

The script is a thin, reproducible wrapper around run_hopper_v7_closed_loop.py.
It exists to keep the explainability-rule smoke isolated from the earlier
confidence-prior runs. Defaults intentionally use a short 5k x 2 continuation
and a lower rule-prior penalty, because eval-only replacement showed
distribution mismatch with old-rule checkpoints.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/anaconda3/envs/sdmcts_p100/bin/python"
RUNNER = REPO_ROOT / "experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py"
BASE_100K_CKPT = (
    REPO_ROOT
    / "logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt"
)
BASE_100K_SD = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/sd_explainers/sd_explainer_cycle_020.pt"
)
EXPLAIN_RULES = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506"
)


def build_cmd(args: argparse.Namespace) -> list[str]:
    return [
        PYTHON,
        str(RUNNER),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--resume-from", str(args.resume_from),
        "--initial-sd-checkpoint", str(args.initial_sd_checkpoint),
        "--initial-rule-library", str(args.initial_rule_library),
        "--exp-prefix", args.exp_prefix,
        "--output-root", str(args.output_root),
        "--start-cycle", str(args.start_cycle),
        "--cycles", str(args.cycles),
        "--cycle-steps", str(args.cycle_steps),
        "--align-resumed-eval",
        "--summary-window-cycles", str(args.summary_window_cycles),
        "--trajectory-score-mode", args.trajectory_score_mode,
        "--rule-prior-allowed-statuses", "explain_active",
        "--rule-required-status", "explain_active",
        "--rule-prior-mode", "penalty_only",
        "--rule-prior-opposite-penalty", str(args.rule_prior_opposite_penalty),
        "--rule-prior-preferred-cap", str(args.rule_prior_preferred_cap),
        "--rule-prior-cap-penalty", str(args.rule_prior_cap_penalty),
        "--eval-episodes", str(args.eval_episodes),
        "--eval-freq", str(args.eval_freq),
        "--checkpoint-freq", str(args.checkpoint_freq),
        "--sd-epochs", str(args.sd_epochs),
        "--sd-max-transitions", str(args.sd_max_transitions),
        "--sd-batch-size", str(args.sd_batch_size),
        "--sd-device", args.sd_device,
        "--max-states-per-episode", str(args.max_states_per_episode),
        "--counterfactual-horizon", str(args.counterfactual_horizon),
        "--random-per-cycle", str(args.random_per_cycle),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume-from", type=Path, default=BASE_100K_CKPT)
    parser.add_argument("--initial-sd-checkpoint", type=Path, default=BASE_100K_SD)
    parser.add_argument("--initial-rule-library", type=Path, default=EXPLAIN_RULES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--exp-prefix", default="tdmpc2_v71a_explain_short_5k_seed1_20260506")
    parser.add_argument("--start-cycle", type=int, default=21)
    parser.add_argument("--cycles", type=int, default=22)
    parser.add_argument("--cycle-steps", type=int, default=5000)
    parser.add_argument("--summary-window-cycles", type=int, default=3)
    parser.add_argument("--trajectory-score-mode", choices=["reward", "stage", "hybrid"], default="stage")
    parser.add_argument("--rule-prior-opposite-penalty", type=float, default=0.01)
    parser.add_argument("--rule-prior-preferred-cap", type=float, default=0.70)
    parser.add_argument("--rule-prior-cap-penalty", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument("--sd-epochs", type=int, default=3)
    parser.add_argument("--sd-max-transitions", type=int, default=50000)
    parser.add_argument("--sd-batch-size", type=int, default=512)
    parser.add_argument("--sd-device", default="cpu")
    parser.add_argument("--max-states-per-episode", type=int, default=3)
    parser.add_argument("--counterfactual-horizon", type=int, default=5)
    parser.add_argument("--random-per-cycle", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.resume_from, args.initial_sd_checkpoint, args.initial_rule_library):
        if not path.exists():
            raise FileNotFoundError(path)
    cmd = build_cmd(args)
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
