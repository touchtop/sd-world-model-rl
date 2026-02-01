#!/usr/bin/env python3
"""Visualization for SD-MCTS Theorem Validation Experiments.

Generates publication-quality figures for:
1. Experiment A: VNet Ablation (Theorem 2) - Line plot with CI bands
2. Experiment C: Crossover Grid (Theorem 3) - 2D Heatmap with boundary
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.figsize': (7, 5),
    })


def load_vnet_ablation_results(json_path: str) -> Dict:
    """Load and process VNet ablation results."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    h_values = data['config']['h_values']
    methods = ['sd_mcts', 'mcts_zero', 'mcts_random']

    processed = {'h_values': h_values}

    for method in methods:
        means = []
        ci95s = []
        for H in h_values:
            # Filter runs for this method and H
            runs = [r for r in data['runs'] if r['method'] == method and r['H'] == H]
            successes = [r['success'] for r in runs]
            success_rate = 100 * np.mean(successes)
            # 95% CI using normal approximation
            n = len(successes)
            if n > 0:
                se = 100 * np.sqrt(np.mean(successes) * (1 - np.mean(successes)) / n)
                ci = 1.96 * se
            else:
                ci = 0
            means.append(success_rate)
            ci95s.append(ci)

        processed[method] = {'mean': means, 'ci95': ci95s}

    return processed


def plot_vnet_ablation(results: Dict, save_path: str, t_collapse: int = 12):
    """Plot Experiment A: VNet Ablation results."""
    set_publication_style()

    h_values = np.array(results['h_values'])

    styles = [
        ('sd_mcts', 'SD-MCTS', '#1f77b4', 'o', '-'),
        ('mcts_zero', 'MCTS-Zero', '#d62728', 's', '--'),
        ('mcts_random', 'MCTS-Random', '#7f7f7f', '^', ':'),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, label, color, marker, ls in styles:
        if key in results:
            data = results[key]
            mean = np.array(data['mean'])
            ci = np.array(data['ci95'])

            ax.plot(h_values, mean, label=label, color=color,
                    marker=marker, linestyle=ls, markersize=8, markeredgewidth=1.5)
            ax.fill_between(h_values, mean - ci, mean + ci, color=color, alpha=0.15)

    # Vertical line at T_collapse
    ax.axvline(x=t_collapse, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(t_collapse + 0.3, 50, f'$T_{{collapse}}={t_collapse}$',
            rotation=90, verticalalignment='center', fontsize=10, alpha=0.7)

    ax.set_xlabel('Search Depth $H$')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-5, 105)
    ax.set_xlim(min(h_values) - 0.5, max(h_values) + 0.5)
    ax.set_xticks(h_values)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def load_crossover_grid_results(json_path: str) -> Dict:
    """Load and process crossover grid results."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    h_values = sorted(set(g['H'] for g in data['grid']))
    delta_values = sorted(set(g['delta'] for g in data['grid']))

    # Build RPR matrix (delta x H)
    rpr_matrix = np.zeros((len(delta_values), len(h_values)))

    for g in data['grid']:
        i = delta_values.index(g['delta'])
        j = h_values.index(g['H'])
        rpr_matrix[i, j] = g['rpr_mean']

    return {
        'h_values': h_values,
        'delta_values': delta_values,
        'rpr_matrix': rpr_matrix,
        'oracle': data.get('oracle_baseline', {}),
    }


def plot_crossover_grid(results: Dict, save_path: str):
    """Plot Experiment C: Crossover Grid Heatmap."""
    set_publication_style()

    h_values = results['h_values']
    delta_values = results['delta_values']
    rpr_matrix = results['rpr_matrix']

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diverging colormap centered at RPR=1.0
    vmin = max(0, rpr_matrix.min() - 0.1)
    vmax = min(2, rpr_matrix.max() + 0.1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(rpr_matrix, cmap='RdBu', norm=norm,
                   origin='lower', aspect='auto',
                   extent=[-0.5, len(h_values)-0.5, -0.5, len(delta_values)-0.5])

    # Axis labels
    ax.set_xticks(range(len(h_values)))
    ax.set_xticklabels(h_values)
    ax.set_yticks(range(len(delta_values)))
    ax.set_yticklabels([f'{d:.0%}' for d in delta_values])

    ax.set_xlabel('Search Depth $H$')
    ax.set_ylabel(r'Parameter Error $\delta_\theta$')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Performance Ratio (RPR)')
    cbar.ax.axhline(1.0, color='black', linewidth=1.5)

    # Contour at RPR=1.0 (crossover boundary)
    X, Y = np.meshgrid(range(len(h_values)), range(len(delta_values)))
    try:
        cs = ax.contour(X, Y, rpr_matrix, levels=[1.0],
                        colors='black', linestyles='--', linewidths=2)
        ax.clabel(cs, inline=True, fmt='RPR=1', fontsize=9)
    except ValueError:
        pass  # No contour if all values on one side

    # Annotate cells with RPR values
    for i in range(len(delta_values)):
        for j in range(len(h_values)):
            val = rpr_matrix[i, j]
            color = 'white' if abs(val - 1.0) > 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_mock_data(output_dir: Path):
    """Create mock data for testing visualization."""
    # Mock VNet ablation data
    h_values = [2, 4, 6, 8, 10, 12, 14]
    vnet_data = {
        'config': {'h_values': h_values},
        'runs': []
    }

    for H in h_values:
        for seed in range(30):
            # SD-MCTS: high success across all H
            vnet_data['runs'].append({
                'method': 'sd_mcts', 'H': H, 'seed': seed,
                'success': np.random.random() < 0.95
            })
            # MCTS-Zero: step function at H=12
            prob = 0.05 if H < 12 else 0.90
            vnet_data['runs'].append({
                'method': 'mcts_zero', 'H': H, 'seed': seed,
                'success': np.random.random() < prob
            })
            # MCTS-Random: medium performance
            vnet_data['runs'].append({
                'method': 'mcts_random', 'H': H, 'seed': seed,
                'success': np.random.random() < 0.4 + H * 0.02
            })

    vnet_path = output_dir / 'vnet_ablation_mock.json'
    with open(vnet_path, 'w') as f:
        json.dump(vnet_data, f)
    print(f"Created mock data: {vnet_path}")

    # Mock crossover grid data
    h_values = [5, 10, 15, 20]
    delta_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    grid_data = {
        'config': {'h_values': h_values, 'delta_values': delta_values},
        'oracle_baseline': {'mean': 20.0},
        'grid': []
    }

    for H in h_values:
        for delta in delta_values:
            # RPR decreases with delta, increases with H
            rpr = 1.2 - delta * 0.8 + (H - 10) * 0.03
            rpr = max(0.3, min(1.8, rpr + np.random.normal(0, 0.05)))
            grid_data['grid'].append({
                'H': H, 'delta': delta, 'rpr_mean': rpr, 'rpr_std': 0.1
            })

    grid_path = output_dir / 'crossover_grid_mock.json'
    with open(grid_path, 'w') as f:
        json.dump(grid_data, f)
    print(f"Created mock data: {grid_path}")

    return vnet_path, grid_path


def main():
    parser = argparse.ArgumentParser(description="Visualize P0 experiment results")
    parser.add_argument("--vnet-results", type=str, help="Path to VNet ablation JSON")
    parser.add_argument("--grid-results", type=str, help="Path to crossover grid JSON")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--mock", action="store_true", help="Generate mock data for testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        print("Generating mock data...")
        vnet_path, grid_path = create_mock_data(output_dir.parent)
        args.vnet_results = str(vnet_path)
        args.grid_results = str(grid_path)

    # Plot VNet ablation
    if args.vnet_results:
        print(f"Loading VNet ablation results from: {args.vnet_results}")
        vnet_data = load_vnet_ablation_results(args.vnet_results)
        plot_vnet_ablation(vnet_data, str(output_dir / 'vnet_ablation.png'))

    # Plot crossover grid
    if args.grid_results:
        print(f"Loading crossover grid results from: {args.grid_results}")
        grid_data = load_crossover_grid_results(args.grid_results)
        plot_crossover_grid(grid_data, str(output_dir / 'crossover_heatmap.png'))

    print("Done!")


if __name__ == "__main__":
    main()
