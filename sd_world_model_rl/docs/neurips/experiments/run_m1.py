#!/usr/bin/env python
"""
M1 Experiment Runner - OR Journal Submission

Usage:
    python run_m1.py                          # Run full M1 experiments
    python run_m1.py --quick                  # Quick test (3 seeds, 2000 steps)
    python run_m1.py --heuristics-only        # Only run heuristic baselines
    python run_m1.py --seeds 20               # Use 20 seeds
"""

import argparse
import sys
import os

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.m1_framework import run_m1_experiments


def main():
    parser = argparse.ArgumentParser(description="M1 Experiment Framework")
    parser.add_argument("--output-dir", type=str, default="results/m1",
                        help="Output directory for results")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds (default: 10)")
    parser.add_argument("--budget", type=int, default=8000,
                        help="Training budget in env steps (default: 8000)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (3 seeds, 2000 steps)")
    parser.add_argument("--heuristics-only", action="store_true",
                        help="Only run heuristic baselines (no PPO/SD+MCTS)")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 3
        args.budget = 2000
        print("[Quick Mode] Using 3 seeds and 2000 step budget")

    results = run_m1_experiments(
        output_dir=args.output_dir,
        env_steps_budget=args.budget,
        num_seeds=args.seeds,
        include_learning_methods=not args.heuristics_only,
    )

    print("\n[Summary] Key Results:")
    for scenario, policies in results.items():
        print(f"\n  {scenario}:")
        for policy, metrics in policies.items():
            avg_q = metrics.get("avg_queue_length_mean", 0)
            avg_q_ci = metrics.get("avg_queue_length_ci95", 0)
            fatigue = metrics.get("fatigue_trigger_rate_mean", 0)
            print(f"    {policy:12s}: avg_queue={avg_q:.2f}±{avg_q_ci:.2f}, fatigue_rate={fatigue:.3f}")


if __name__ == "__main__":
    main()
