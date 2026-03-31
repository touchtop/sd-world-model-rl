#!/usr/bin/env python3
"""Generate Sample Efficiency Figure for NeurIPS paper.

Implements Gemini's visualization recommendations:
1. Main plot: Learning curves with log-scale X-axis
2. Inset bar chart: "Samples Required to Reach Return of -800"
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Style configuration for NeurIPS
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color scheme
COLORS = {
    'sd_mcts': '#2E86AB',      # Blue
    'dreamerv3': '#A23B72',    # Magenta
    'ppo': '#F18F01',          # Orange
    'dqn': '#C73E1D',          # Red
}

LABELS = {
    'sd_mcts': 'SD-MCTS (Ours)',
    'dreamerv3': 'DreamerV3',
    'ppo': 'PPO',
    'dqn': 'DQN',
}

MARKERS = {
    'sd_mcts': 'o',
    'dreamerv3': 's',
    'ppo': '^',
    'dqn': 'v',
}


def load_results(results_path: Path) -> dict:
    """Load and aggregate results by method and budget."""
    with open(results_path) as f:
        data = json.load(f)

    # Group by (method, budget)
    grouped = defaultdict(list)
    for r in data:
        if r.get('error'):
            continue
        key = (r['method'], r['sample_budget'])
        grouped[key].append(r['eval_return_mean'])

    # Compute mean and std
    results = {}
    for (method, budget), returns in grouped.items():
        if method not in results:
            results[method] = {'budgets': [], 'means': [], 'stds': [], 'ses': []}
        n = len(returns)
        std = np.std(returns, ddof=1) if n > 1 else 0.0
        results[method]['budgets'].append(budget)
        results[method]['means'].append(np.mean(returns))
        results[method]['stds'].append(std)
        results[method]['ses'].append(std / np.sqrt(n) if n > 0 else 0.0)

    # Sort by budget
    for method in results:
        idx = np.argsort(results[method]['budgets'])
        for key in ['budgets', 'means', 'stds', 'ses']:
            results[method][key] = [results[method][key][i] for i in idx]

    return results


def estimate_samples_to_threshold(results: dict, threshold: float = -800) -> dict:
    """Estimate samples needed to reach a performance threshold."""
    samples_needed = {}

    for method, data in results.items():
        budgets = np.array(data['budgets'])
        means = np.array(data['means'])

        # Find first budget where mean >= threshold
        above = means >= threshold
        if above.any():
            idx = np.where(above)[0][0]
            samples_needed[method] = budgets[idx]
        else:
            # Extrapolate or mark as "not reached"
            samples_needed[method] = None

    return samples_needed


def create_figure(results: dict, output_path: Path):
    """Create the main figure with inset."""
    fig, ax_main = plt.subplots(figsize=(7, 4.5))

    # --- Main Plot: Learning Curves ---
    methods_order = [
        method for method in ['sd_mcts', 'dreamerv3', 'ppo', 'dqn']
        if method in results
    ]

    for method in methods_order:
        if method not in results:
            continue
        data = results[method]
        budgets = np.array(data['budgets'])
        means = np.array(data['means'])
        ses = np.array(data['ses'])

        # Plot with error bands (standard error)
        ax_main.plot(budgets, means,
                     color=COLORS[method],
                     marker=MARKERS[method],
                     markersize=6,
                     linewidth=1.5,
                     label=LABELS[method])
        ax_main.fill_between(budgets, means - ses, means + ses,
                             color=COLORS[method], alpha=0.2)

    # Log scale for X-axis
    ax_main.set_xscale('log')
    ax_main.set_xticks([1000, 3000, 10000])
    ax_main.set_xticklabels(['1K', '3K', '10K'])

    ax_main.set_xlabel('Environment Steps (Sample Budget)')
    ax_main.set_ylabel('Evaluation Return')
    ax_main.set_title('Sample Efficiency Comparison on Beer Game')

    # Add horizontal line at threshold
    threshold = -800
    ax_main.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_main.text(10500, threshold + 200, f'Target: {threshold}', fontsize=6, color='gray', ha='left', va='bottom')

    # Legend - position in lower right
    ax_main.legend(loc='lower right', framealpha=0.9, fontsize=7)

    # Grid
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim([-12000, 500])

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Also save PDF for paper
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def create_efficiency_table(results: dict, output_path: Path):
    """Create a LaTeX table for the paper."""
    methods = [
        method for method in ['sd_mcts', 'dreamerv3', 'ppo', 'dqn']
        if method in results
    ]
    budgets = [1000, 3000, 10000]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Sample Efficiency Comparison on Beer Game (Return $\uparrow$, mean$\pm$SE, n=3)}",
        r"\label{tab:sample_efficiency}",
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        r"Budget & " + " & ".join([LABELS[m].replace(' (Ours)', r'$^\dagger$' if m == 'sd_mcts' else '') for m in methods]) + r" \\",
        r"\midrule",
    ]

    for budget in budgets:
        row = [f"{budget:,}"]
        best_mean = max(results[m]['means'][results[m]['budgets'].index(budget)]
                        for m in methods if budget in results[m]['budgets'])

        for method in methods:
            if budget in results[method]['budgets']:
                idx = results[method]['budgets'].index(budget)
                mean = results[method]['means'][idx]
                se = results[method]['ses'][idx]

                # Bold if best
                if abs(mean - best_mean) < 1:
                    row.append(f"\\textbf{{{mean:.0f}$\\pm${se:.0f}}}")
                else:
                    row.append(f"{mean:.0f}$\\pm${se:.0f}")
            else:
                row.append("--")

        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\footnotesize{$^\dagger$ Our method. Higher is better (less negative cost).}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample efficiency figures')
    parser.add_argument('--results', type=str,
                        default='../results/main/20260208_201101/results.json',
                        help='Path to results.json')
    parser.add_argument('--output', type=str, default='../figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = Path(__file__).parent / results_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Print summary
    print("\n=== Results Summary ===")
    for method in ['sd_mcts', 'dreamerv3', 'ppo', 'dqn']:
        if method in results:
            print(f"{LABELS[method]}:")
            for i, budget in enumerate(results[method]['budgets']):
                mean = results[method]['means'][i]
                se = results[method]['ses'][i]
                print(f"  {budget:>5}: {mean:>8.1f} ± {se:.1f}")

    # Create visualizations
    print("\n=== Generating Figures ===")
    create_figure(results, output_dir / 'figure1_sample_efficiency.png')
    create_efficiency_table(results, output_dir / 'table1_sample_efficiency.tex')

    print("\nDone!")


if __name__ == '__main__':
    main()
