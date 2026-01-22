#!/usr/bin/env python
"""
Visualization for SD-MCTS vs MPC comparison experiments.

Generates:
1. Efficiency curve comparison (learned vs true)
2. State trajectory comparison (SD-MCTS vs MPC)
3. Bar chart with confidence intervals
"""
import os
import sys
import json
import argparse
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_ttest(data1: list, data2: list) -> dict:
    """Compute paired samples t-test (correct for same-seed comparison)."""
    data1, data2 = np.array(data1), np.array(data2)
    t_stat, p_value = stats.ttest_rel(data1, data2)
    n = len(data1)
    # Cohen's d_z for paired design: mean difference / std of differences
    diffs = data1 - data2
    diff_mean = np.mean(diffs)
    diff_std = np.std(diffs, ddof=1)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01
    }


def plot_efficiency_curves(true_alpha, true_theta, learned_alpha, learned_theta,
                           belief_alpha, belief_theta, output_path):
    """Plot efficiency curves: true vs learned vs belief."""
    fatigue = np.linspace(0, 25, 200)

    # True efficiency (one-sided Lorentzian)
    diff_true = np.maximum(0, fatigue - true_theta)
    eff_true = 1.0 / (1.0 + true_alpha * diff_true ** 2)

    # Learned efficiency
    diff_learned = np.maximum(0, fatigue - learned_theta)
    eff_learned = 1.0 / (1.0 + learned_alpha * diff_learned ** 2)

    # Belief (wrong) efficiency
    diff_belief = np.maximum(0, fatigue - belief_theta)
    eff_belief = 1.0 / (1.0 + belief_alpha * diff_belief ** 2)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(fatigue, eff_true, 'b-', linewidth=2.5, label=f'True (α={true_alpha}, θ={true_theta})')
    ax.plot(fatigue, eff_learned, 'g--', linewidth=2.5, label=f'Learned (α={learned_alpha:.3f}, θ={learned_theta:.2f})')
    ax.plot(fatigue, eff_belief, 'r:', linewidth=2, label=f'Wrong Belief (α={belief_alpha}, θ={belief_theta})')

    ax.axvline(x=true_theta, color='b', linestyle=':', alpha=0.5)
    ax.axvline(x=learned_theta, color='g', linestyle=':', alpha=0.5)

    ax.set_xlabel('Fatigue Level', fontsize=12)
    ax.set_ylabel('Efficiency Coefficient', fontsize=12)
    ax.set_title('Efficiency Degradation: True vs Learned vs Wrong Belief', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved efficiency curves to: {output_path}")


def plot_comparison_bars(results_dict, output_path):
    """Plot bar chart with confidence intervals and t-test significance."""
    methods = ['mpc_wrong', 'adaptive_mpc_h3', 'adaptive_mpc_h15', 'oracle_mpc', 'sd_mcts']
    labels = ['MPC\n(wrong)', 'Adaptive\nH=3', 'Adaptive\nH=15', 'Oracle\nMPC', 'SD-MCTS']
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4']

    means = []
    ci_lowers = []
    ci_uppers = []

    for method in methods:
        method_stats = results_dict['statistics'][method]
        means.append(method_stats['mean'])
        ci_lowers.append(method_stats['mean'] - method_stats['ci_95_lower'])
        ci_uppers.append(method_stats['ci_95_upper'] - method_stats['mean'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=colors, edgecolor='black', linewidth=1.2)
    ax.errorbar(x, means, yerr=[ci_lowers, ci_uppers], fmt='none', color='black', capsize=5, linewidth=2)

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + max(ci_uppers)*0.1,
                f'{mean:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Total Reward (higher is better)', fontsize=12)
    ax.set_title('SD-MCTS vs MPC Baselines: Efficiency Degradation Scenario', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # Compute actual t-test for SD-MCTS vs Oracle MPC
    sd_data = results_dict['results']['sd_mcts']
    oracle_data = results_dict['results']['oracle_mpc']
    ttest = compute_ttest(sd_data, oracle_data)

    sd_idx = methods.index('sd_mcts')
    oracle_idx = methods.index('oracle_mpc')
    y_max = max(means) + max(ci_uppers) * 1.5
    ax.annotate('', xy=(sd_idx, y_max), xytext=(oracle_idx, y_max),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.5))

    # Display actual p-value with significance stars
    p_val = ttest['p_value']
    if ttest['significant_01']:
        if p_val < 0.0001:
            sig_text = "**p<0.0001"
        else:
            sig_text = f"**p={p_val:.4f}"
    elif ttest['significant_05']:
        sig_text = f"*p={p_val:.4f}"
    else:
        sig_text = f"p={p_val:.3f} (n.s.)"
    ax.text((sd_idx + oracle_idx)/2, y_max + 50, sig_text, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison bars to: {output_path}")
    print(f"  SD-MCTS vs Oracle MPC: t={ttest['t_statistic']:.3f}, p={ttest['p_value']:.4f}, d={ttest['cohens_d']:.3f}")


def plot_improvement_chart(results_dict, output_path):
    """Plot improvement percentage vs MPC wrong."""
    methods = ['adaptive_mpc_h3', 'adaptive_mpc_h15', 'oracle_mpc', 'sd_mcts']
    labels = ['Adaptive\nH=3', 'Adaptive\nH=15', 'Oracle\nMPC', 'SD-MCTS']
    colors = ['#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4']

    mpc_wrong_mean = results_dict['statistics']['mpc_wrong']['mean']
    improvements = []

    for method in methods:
        method_mean = results_dict['statistics'][method]['mean']
        improvement = 100 * (method_mean - mpc_wrong_mean) / abs(mpc_wrong_mean)
        improvements.append(improvement)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(methods))
    bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=1.2)

    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, imp + 0.3,
                f'+{imp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Improvement vs MPC (wrong) %', fontsize=12)
    ax.set_title('Performance Improvement Over Baseline', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved improvement chart to: {output_path}")


def plot_state_trajectory(trajectory_file: str, output_path: str):
    """Plot state trajectory comparison between SD-MCTS and MPC.

    Args:
        trajectory_file: Path to trajectory JSON with format:
            {"sd_mcts": {"queue": [...], "backlog": [...], "fatigue": [...]},
             "oracle_mpc": {...}, "mpc_wrong": {...}}
        output_path: Where to save the figure
    """
    if not os.path.exists(trajectory_file):
        print(f"Trajectory file not found: {trajectory_file}")
        return

    with open(trajectory_file, 'r') as f:
        traj = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    methods = ['sd_mcts', 'oracle_mpc', 'mpc_wrong']
    labels = ['SD-MCTS', 'Oracle MPC', 'MPC (wrong)']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    linestyles = ['-', '--', ':']

    metrics = ['queue', 'backlog', 'fatigue']
    ylabels = ['Queue Length', 'Total Backlog', 'Cumulative Fatigue']

    for ax_idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[ax_idx]
        for method, label, color, ls in zip(methods, labels, colors, linestyles):
            if method in traj and metric in traj[method]:
                data = traj[method][metric]
                ax.plot(data, label=label, color=color, linestyle=ls, linewidth=1.5)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    axes[-1].set_xlabel('Time Step', fontsize=11)
    axes[0].set_title('State Trajectory Comparison', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to: {output_path}")


def plot_state_trajectory_from_dict(traj: Dict, output_path: str):
    """Plot state trajectory from dictionary data.

    Args:
        traj: Dictionary with method keys and trajectory values.
        output_path: Where to save the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    methods = ['sd_mcts', 'oracle_mpc', 'mpc_wrong']
    labels = ['SD-MCTS', 'Oracle MPC', 'MPC (wrong)']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    linestyles = ['-', '--', ':']

    metrics = ['queue', 'backlog', 'fatigue']
    ylabels = ['Queue Length', 'Total Backlog', 'Cumulative Fatigue']

    for ax_idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[ax_idx]
        for method, label, color, ls in zip(methods, labels, colors, linestyles):
            if method in traj and metric in traj[method]:
                data = traj[method][metric]
                ax.plot(data, label=label, color=color, linestyle=ls, linewidth=1.5)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    axes[-1].set_xlabel('Time Step', fontsize=11)
    axes[0].set_title('State Trajectory Comparison (Seed 0)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--trajectory", type=str, default=None, help="Path to trajectory JSON file")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results, 'r') as f:
        results = json.load(f)

    # 1. Efficiency curves
    plot_efficiency_curves(
        true_alpha=results['true_alpha'],
        true_theta=results['true_theta'],
        learned_alpha=results['learned_alpha'],
        learned_theta=results['learned_theta'],
        belief_alpha=results['belief_alpha'],
        belief_theta=results['belief_theta'],
        output_path=os.path.join(args.output_dir, 'efficiency_curves.png')
    )

    # 2. Comparison bars
    plot_comparison_bars(
        results_dict=results,
        output_path=os.path.join(args.output_dir, 'comparison_bars.png')
    )

    # 3. Improvement chart
    plot_improvement_chart(
        results_dict=results,
        output_path=os.path.join(args.output_dir, 'improvement_chart.png')
    )

    # 4. Trajectory plot (from embedded data or separate file)
    trajectory_data = None
    if args.trajectory:
        if os.path.exists(args.trajectory):
            with open(args.trajectory, 'r') as f:
                trajectory_data = json.load(f)
    elif 'trajectories' in results and results['trajectories']:
        trajectory_data = results['trajectories']

    if trajectory_data:
        plot_state_trajectory_from_dict(
            trajectory_data,
            output_path=os.path.join(args.output_dir, 'state_trajectory.png')
        )

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
