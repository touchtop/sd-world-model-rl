#!/usr/bin/env python3
"""
NeurIPS Experiment: Demand Pattern Transfer Comparison

Compare SD-MCTS (15K) vs BlackBox-MCTS (30K) on zero-shot demand transfer.

Design rationale:
- SD-MCTS with 15K steps achieves similar in-distribution performance as BlackBox with 30K
- This isolates the generalization advantage of structural priors from sample efficiency

Usage:
    python run_demand_transfer_comparison.py
    python run_demand_transfer_comparison.py --seeds 5
    python run_demand_transfer_comparison.py --sd-budget 15000 --bb-budget 30000
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from run_demand_transfer import (
    train_sd_mcts,
    train_blackbox_mcts,
    evaluate_sd_mcts,
    evaluate_blackbox_mcts,
    create_env_with_pattern,
    compute_statistics,
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Training budgets (SD needs less samples to match BlackBox performance)
    "sd_budget": 15000,
    "blackbox_budget": 30000,

    # Demand patterns
    "train_pattern": "step",
    "test_patterns": ["step", "sinusoidal", "random_walk", "bursty"],

    # Evaluation
    "n_seeds": 3,
    "eval_episodes": 20,
    "num_simulations": 200,
    "gamma": 0.99,
    "device": "cpu",
}


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(config):
    """Run full comparison experiment."""
    results = []

    methods = [
        ("sd_mcts", config["sd_budget"]),
        ("blackbox_mcts", config["blackbox_budget"]),
    ]

    total_runs = config["n_seeds"] * len(methods)
    current_run = 0

    for seed in range(config["n_seeds"]):
        for method, budget in methods:
            current_run += 1

            print(f"\n{'='*60}")
            print(f"[{current_run}/{total_runs}] Method: {method}, Budget: {budget}, Seed: {seed}")
            print(f"{'='*60}")

            # Set seeds
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Train on step demand
            train_env = create_env_with_pattern(config["train_pattern"], seed=seed)

            if method == "sd_mcts":
                wm, pn, vn, rewards = train_sd_mcts(
                    train_env,
                    env_steps=budget,
                    gamma=config["gamma"],
                    device=config["device"],
                    verbose=True,
                )
            else:
                wm, _, vn, rewards = train_blackbox_mcts(
                    train_env,
                    env_steps=budget,
                    gamma=config["gamma"],
                    device=config["device"],
                    verbose=True,
                )
                pn = None

            # Record training performance
            train_final_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)

            # Evaluate on all patterns
            print(f"\nEvaluating on {len(config['test_patterns'])} patterns...")
            for pattern in config["test_patterns"]:
                test_env = create_env_with_pattern(pattern, seed=seed + 10000)

                if method == "sd_mcts":
                    eval_result = evaluate_sd_mcts(
                        test_env, wm, None, vn,
                        n_episodes=config["eval_episodes"],
                        gamma=config["gamma"],
                        device=config["device"],
                        num_simulations=config["num_simulations"],
                    )
                else:
                    eval_result = evaluate_blackbox_mcts(
                        test_env, wm, vn,
                        n_episodes=config["eval_episodes"],
                        gamma=config["gamma"],
                        device=config["device"],
                        num_simulations=config["num_simulations"],
                    )

                results.append({
                    "method": method,
                    "budget": budget,
                    "seed": seed,
                    "pattern": pattern,
                    "cost_mean": eval_result["total_cost_mean"],
                    "cost_std": eval_result["total_cost_std"],
                    "train_final_reward": train_final_reward,
                })

                print(f"  {pattern}: {eval_result['total_cost_mean']:.1f} ± {eval_result['total_cost_std']:.1f}")

    return results


def generate_summary(results, config):
    """Generate summary tables."""
    lines = []

    lines.append("=" * 60)
    lines.append("SUMMARY: SD-MCTS vs BlackBox-MCTS Demand Transfer")
    lines.append("=" * 60)
    lines.append(f"\nSD-MCTS budget: {config['sd_budget']}")
    lines.append(f"BlackBox budget: {config['blackbox_budget']}")
    lines.append(f"Seeds: {config['n_seeds']}")
    lines.append(f"Eval episodes: {config['eval_episodes']}")

    # Table 1: Absolute costs
    lines.append("\n\n## Table 1: Absolute Costs (Mean ± SE)")
    lines.append("\n| Method | Budget | Step | Sinusoidal | Random Walk | Bursty |")
    lines.append("|--------|--------|------|------------|-------------|--------|")

    methods = [("sd_mcts", config["sd_budget"]), ("blackbox_mcts", config["blackbox_budget"])]

    for method, budget in methods:
        row = [method, str(budget)]
        for pattern in config["test_patterns"]:
            costs = [r["cost_mean"] for r in results
                     if r["method"] == method and r["pattern"] == pattern]
            if costs:
                mean = np.mean(costs)
                se = np.std(costs) / np.sqrt(len(costs)) if len(costs) > 1 else 0
                row.append(f"{mean:.0f}±{se:.0f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # Table 2: Transfer ratios
    lines.append("\n\n## Table 2: Transfer Ratio (Test Cost / Train Cost)")
    lines.append("\n| Method | Step→Sinus | Step→RandWalk | Step→Bursty | Avg OOD |")
    lines.append("|--------|------------|---------------|-------------|---------|")

    for method, budget in methods:
        # Get step (in-distribution) costs
        step_costs = [r["cost_mean"] for r in results
                      if r["method"] == method and r["pattern"] == "step"]
        step_mean = np.mean(step_costs) if step_costs else 1

        ratios = []
        row = [method]
        for pattern in ["sinusoidal", "random_walk", "bursty"]:
            costs = [r["cost_mean"] for r in results
                     if r["method"] == method and r["pattern"] == pattern]
            if costs:
                mean = np.mean(costs)
                ratio = mean / step_mean
                ratios.append(ratio)
                row.append(f"{ratio:.2f}x")
            else:
                row.append("N/A")

        avg_ood = np.mean(ratios) if ratios else 0
        row.append(f"**{avg_ood:.2f}x**")
        lines.append("| " + " | ".join(row) + " |")

    # Table 3: Training performance comparison
    lines.append("\n\n## Table 3: Training Performance (Final Avg Reward)")
    lines.append("\n| Method | Budget | Final Reward (Avg±SE) |")
    lines.append("|--------|--------|----------------------|")

    for method, budget in methods:
        rewards = [r["train_final_reward"] for r in results
                   if r["method"] == method and r["pattern"] == "step"]
        if rewards:
            mean = np.mean(rewards)
            se = np.std(rewards) / np.sqrt(len(rewards)) if len(rewards) > 1 else 0
            lines.append(f"| {method} | {budget} | {mean:.1f}±{se:.1f} |")

    # Key findings
    lines.append("\n\n## Key Findings")

    sd_ood_ratios = []
    bb_ood_ratios = []

    for pattern in ["sinusoidal", "random_walk", "bursty"]:
        sd_step = np.mean([r["cost_mean"] for r in results
                          if r["method"] == "sd_mcts" and r["pattern"] == "step"])
        sd_test = np.mean([r["cost_mean"] for r in results
                          if r["method"] == "sd_mcts" and r["pattern"] == pattern])
        bb_step = np.mean([r["cost_mean"] for r in results
                          if r["method"] == "blackbox_mcts" and r["pattern"] == "step"])
        bb_test = np.mean([r["cost_mean"] for r in results
                          if r["method"] == "blackbox_mcts" and r["pattern"] == pattern])

        if sd_step > 0 and bb_step > 0:
            sd_ood_ratios.append(sd_test / sd_step)
            bb_ood_ratios.append(bb_test / bb_step)

    if sd_ood_ratios and bb_ood_ratios:
        sd_avg = np.mean(sd_ood_ratios)
        bb_avg = np.mean(bb_ood_ratios)
        advantage = (bb_avg - sd_avg) / sd_avg * 100

        lines.append(f"\n- SD-MCTS average OOD transfer ratio: {sd_avg:.2f}x")
        lines.append(f"- BlackBox average OOD transfer ratio: {bb_avg:.2f}x")
        lines.append(f"- SD-MCTS generalization advantage: {advantage:.1f}% less degradation")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Demand Transfer Comparison Experiment")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--sd-budget", type=int, default=15000, help="SD-MCTS training budget")
    parser.add_argument("--bb-budget", type=int, default=30000, help="BlackBox training budget")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    # Update config
    CONFIG["n_seeds"] = args.seeds
    CONFIG["sd_budget"] = args.sd_budget
    CONFIG["blackbox_budget"] = args.bb_budget
    CONFIG["eval_episodes"] = args.eval_episodes
    CONFIG["device"] = args.device

    print("=" * 60)
    print("Demand Pattern Transfer Comparison")
    print("SD-MCTS vs BlackBox-MCTS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  SD-MCTS budget:    {CONFIG['sd_budget']}")
    print(f"  BlackBox budget:   {CONFIG['blackbox_budget']}")
    print(f"  Seeds:             {CONFIG['n_seeds']}")
    print(f"  Eval episodes:     {CONFIG['eval_episodes']}")
    print(f"  Test patterns:     {CONFIG['test_patterns']}")

    # Run experiment
    results = run_experiment(CONFIG)

    # Generate and print summary
    summary = generate_summary(results, CONFIG)
    print("\n" + summary)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = NEURIPS_DIR / "results" / "demand_transfer_comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
