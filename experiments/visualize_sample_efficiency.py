"""
Visualization for Sample Efficiency Comparison Results.

Generates:
1. Learning curves (log-scale X-axis) with 95% CI
2. Bar charts comparing final performance and collapse rates
3. Sample efficiency ratio table
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def load_results(results_path: str) -> Dict:
    """Load experiment results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def aggregate_seeds(seed_results: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Aggregate results across seeds."""
    # Get all step counts
    all_steps = set()
    for seed_data in seed_results.values():
        all_steps.update(seed_data.get("steps", []))
    steps = sorted(all_steps)

    # Interpolate to common steps
    rewards_matrix = []
    collapse_matrix = []

    for seed_data in seed_results.values():
        if not seed_data.get("steps"):
            continue
        seed_steps = np.array(seed_data["steps"])
        seed_rewards = np.array(seed_data["rewards"])
        seed_collapse = np.array(seed_data["collapse_rates"])

        # Interpolate
        interp_rewards = np.interp(steps, seed_steps, seed_rewards)
        interp_collapse = np.interp(steps, seed_steps, seed_collapse)

        rewards_matrix.append(interp_rewards)
        collapse_matrix.append(interp_collapse)

    rewards_matrix = np.array(rewards_matrix)
    collapse_matrix = np.array(collapse_matrix)

    return {
        "steps": np.array(steps),
        "rewards_mean": np.mean(rewards_matrix, axis=0),
        "rewards_std": np.std(rewards_matrix, axis=0),
        "rewards_ci95": 1.96 * np.std(rewards_matrix, axis=0) / np.sqrt(len(rewards_matrix)),
        "collapse_mean": np.mean(collapse_matrix, axis=0),
        "collapse_std": np.std(collapse_matrix, axis=0),
        "final_reward_mean": np.mean(rewards_matrix[:, -1]),
        "final_reward_std": np.std(rewards_matrix[:, -1]),
        "final_collapse_mean": np.mean(collapse_matrix[:, -1]),
        "final_collapse_std": np.std(collapse_matrix[:, -1]),
    }


def plot_learning_curves(
    scenario_results: Dict,
    scenario_name: str,
    output_dir: str,
):
    """Plot learning curves with log-scale X-axis."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "sdmcts": "#2ecc71",  # Green
        30000: "#e74c3c",     # Red
        100000: "#e67e22",    # Orange
        300000: "#9b59b6",    # Purple
        1000000: "#3498db",   # Blue
    }

    # SD-MCTS
    sdmcts_agg = aggregate_seeds(scenario_results["sdmcts"]["seeds"])
    if len(sdmcts_agg["steps"]) > 0:
        ax1.plot(sdmcts_agg["steps"], sdmcts_agg["rewards_mean"],
                 color=colors["sdmcts"], linewidth=2, label="SD-MCTS (30k)")
        ax1.fill_between(
            sdmcts_agg["steps"],
            sdmcts_agg["rewards_mean"] - sdmcts_agg["rewards_ci95"],
            sdmcts_agg["rewards_mean"] + sdmcts_agg["rewards_ci95"],
            color=colors["sdmcts"], alpha=0.2
        )
        # Extend SD-MCTS line as dashed
        max_budget = max(scenario_results["ppo"].keys())
        ax1.axhline(y=sdmcts_agg["final_reward_mean"], color=colors["sdmcts"],
                    linestyle="--", alpha=0.7, xmin=0.3)

        ax2.plot(sdmcts_agg["steps"], sdmcts_agg["collapse_mean"],
                 color=colors["sdmcts"], linewidth=2, label="SD-MCTS (30k)")

    # PPO at different budgets
    for budget, budget_data in scenario_results["ppo"].items():
        ppo_agg = aggregate_seeds(budget_data["seeds"])
        if len(ppo_agg["steps"]) > 0:
            label = f"PPO ({budget//1000}k)"
            ax1.plot(ppo_agg["steps"], ppo_agg["rewards_mean"],
                     color=colors.get(budget, "#7f8c8d"), linewidth=2, label=label)
            ax1.fill_between(
                ppo_agg["steps"],
                ppo_agg["rewards_mean"] - ppo_agg["rewards_ci95"],
                ppo_agg["rewards_mean"] + ppo_agg["rewards_ci95"],
                color=colors.get(budget, "#7f8c8d"), alpha=0.2
            )

            ax2.plot(ppo_agg["steps"], ppo_agg["collapse_mean"],
                     color=colors.get(budget, "#7f8c8d"), linewidth=2, label=label)

    # Format axes
    ax1.set_xscale("log")
    ax1.set_xlabel("Training Steps (log scale)", fontsize=12)
    ax1.set_ylabel("Average Reward", fontsize=12)
    ax1.set_title(f"{scenario_name}: Learning Curves", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Training Steps (log scale)", fontsize=12)
    ax2.set_ylabel("Collapse Rate", fontsize=12)
    ax2.set_title(f"{scenario_name}: Collapse Rate", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"learning_curve_{scenario_name}.png"), dpi=150)
    plt.close()


def plot_bar_comparison(
    scenario_results: Dict,
    scenario_name: str,
    output_dir: str,
):
    """Plot bar chart comparing final performance."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = ["SD-MCTS\n30k"]
    rewards = []
    collapses = []
    errors_r = []
    errors_c = []

    # SD-MCTS
    sdmcts_agg = aggregate_seeds(scenario_results["sdmcts"]["seeds"])
    rewards.append(sdmcts_agg["final_reward_mean"])
    collapses.append(sdmcts_agg["final_collapse_mean"])
    errors_r.append(sdmcts_agg["final_reward_std"])
    errors_c.append(sdmcts_agg["final_collapse_std"])

    # PPO at different budgets
    for budget in sorted(scenario_results["ppo"].keys()):
        ppo_agg = aggregate_seeds(scenario_results["ppo"][budget]["seeds"])
        methods.append(f"PPO\n{budget//1000}k")
        rewards.append(ppo_agg["final_reward_mean"])
        collapses.append(ppo_agg["final_collapse_mean"])
        errors_r.append(ppo_agg["final_reward_std"])
        errors_c.append(ppo_agg["final_collapse_std"])

    x = np.arange(len(methods))
    width = 0.6

    colors = ["#2ecc71"] + ["#e74c3c", "#e67e22", "#9b59b6", "#3498db"][:len(methods)-1]

    # Reward bars
    bars1 = ax1.bar(x, rewards, width, yerr=errors_r, color=colors, capsize=5, alpha=0.8)
    ax1.set_ylabel("Average Reward", fontsize=12)
    ax1.set_title(f"{scenario_name}: Final Performance", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add training cost annotations
    if len(methods) > 1:
        for i, budget in enumerate(sorted(scenario_results["ppo"].keys())):
            ratio = budget / 30000
            if ratio > 1:
                ax1.annotate(f"{ratio:.0f}x cost",
                             xy=(i+1, rewards[i+1] + errors_r[i+1] + 0.5),
                             ha="center", fontsize=9, color="#7f8c8d")

    # Collapse bars
    bars2 = ax2.bar(x, [c * 100 for c in collapses], width,
                    yerr=[e * 100 for e in errors_c], color=colors, capsize=5, alpha=0.8)
    ax2.set_ylabel("Collapse Rate (%)", fontsize=12)
    ax2.set_title(f"{scenario_name}: Safety (Lower is Better)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bar_comparison_{scenario_name}.png"), dpi=150)
    plt.close()


def compute_sample_efficiency_ratio(scenario_results: Dict) -> Dict[str, float]:
    """Compute sample efficiency ratio: N_PPO / 30k to match SD-MCTS performance."""

    sdmcts_agg = aggregate_seeds(scenario_results["sdmcts"]["seeds"])
    target_reward = sdmcts_agg["final_reward_mean"]

    ratios = {}

    for budget in sorted(scenario_results["ppo"].keys()):
        ppo_agg = aggregate_seeds(scenario_results["ppo"][budget]["seeds"])

        # Find first step where PPO reaches SD-MCTS target
        steps = ppo_agg["steps"]
        rewards = ppo_agg["rewards_mean"]

        reached = False
        for i, (s, r) in enumerate(zip(steps, rewards)):
            if r >= target_reward:
                ratios[budget] = s / 30000
                reached = True
                break

        if not reached:
            # PPO never reached SD-MCTS performance
            ratios[budget] = float("inf")

    return ratios


def generate_summary_table(results: Dict, output_dir: str):
    """Generate summary table in Markdown format."""

    lines = [
        "# Sample Efficiency Comparison Summary",
        "",
        "## Final Performance Comparison",
        "",
        "| Scenario | SD-MCTS (30k) | PPO (30k) | PPO (100k) | PPO (300k) | PPO (1M) |",
        "|----------|---------------|-----------|------------|------------|----------|",
    ]

    for scenario_name, scenario_results in results["scenarios"].items():
        sdmcts_agg = aggregate_seeds(scenario_results["sdmcts"]["seeds"])
        row = [scenario_name, f"{sdmcts_agg['final_reward_mean']:.2f}"]

        for budget in [30000, 100000, 300000, 1000000]:
            if budget in scenario_results["ppo"]:
                ppo_agg = aggregate_seeds(scenario_results["ppo"][budget]["seeds"])
                row.append(f"{ppo_agg['final_reward_mean']:.2f}")
            else:
                row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Collapse Rate Comparison",
        "",
        "| Scenario | SD-MCTS (30k) | PPO (30k) | PPO (100k) | PPO (300k) | PPO (1M) |",
        "|----------|---------------|-----------|------------|------------|----------|",
    ])

    for scenario_name, scenario_results in results["scenarios"].items():
        sdmcts_agg = aggregate_seeds(scenario_results["sdmcts"]["seeds"])
        row = [scenario_name, f"{sdmcts_agg['final_collapse_mean']*100:.1f}%"]

        for budget in [30000, 100000, 300000, 1000000]:
            if budget in scenario_results["ppo"]:
                ppo_agg = aggregate_seeds(scenario_results["ppo"][budget]["seeds"])
                row.append(f"{ppo_agg['final_collapse_mean']*100:.1f}%")
            else:
                row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Sample Efficiency Ratio",
        "",
        "How many more steps does PPO need to match SD-MCTS performance?",
        "",
        "| Scenario | PPO Steps to Match | Efficiency Ratio |",
        "|----------|-------------------|------------------|",
    ])

    for scenario_name, scenario_results in results["scenarios"].items():
        ratios = compute_sample_efficiency_ratio(scenario_results)

        # Find first budget that matches
        for budget in sorted(ratios.keys()):
            if ratios[budget] != float("inf"):
                lines.append(f"| {scenario_name} | {budget//1000}k | {ratios[budget]:.1f}x |")
                break
        else:
            lines.append(f"| {scenario_name} | >1M (never) | >33x |")

    with open(os.path.join(output_dir, "sample_efficiency_summary.md"), "w") as f:
        f.write("\n".join(lines))

    print("Summary saved to sample_efficiency_summary.md")


def main(results_path: str, output_dir: str = None):
    """Generate all visualizations."""

    results = load_results(results_path)

    if output_dir is None:
        output_dir = os.path.dirname(results_path)

    os.makedirs(output_dir, exist_ok=True)

    # Generate plots for each scenario
    for scenario_name, scenario_results in results["scenarios"].items():
        print(f"Generating plots for: {scenario_name}")
        plot_learning_curves(scenario_results, scenario_name, output_dir)
        plot_bar_comparison(scenario_results, scenario_name, output_dir)

    # Generate summary table
    generate_summary_table(results, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize sample efficiency results")
    parser.add_argument("results_path", type=str,
                        help="Path to sample_efficiency_results.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")

    args = parser.parse_args()
    main(args.results_path, args.output_dir)
