#!/usr/bin/env python3
"""
Generate publication-quality figures for Causal Intervention (Zero-Shot Transfer).

Three-variant comparison:
1. SD (with graph editing)
2. SD_no_update (same SD model, NO graph editing — ablation control)
3. BB (BlackBox MLP, no adaptation)

Figures:
1. Bar chart comparing 3 variants' policy costs across interventions
2. Bar chart comparing 3 variants' RMSE errors across interventions
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# NeurIPS-style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

RESULTS_DIR = Path("/home/data/round/code/sd_world_model_rl/docs/neurips/results/causal_intervention")

COLORS = {
    'sd': '#2E7D32',          # Green for SD (edited)
    'sd_no_update': '#81C784', # Light green for SD_no_update
    'blackbox': '#C62828',     # Red for BlackBox
}

NAMES = {
    'none': 'No Intervention',
    'capacity_shock': 'Capacity Shock',
    'comm_severed': 'Comm Severed',
}

VARIANT_LABELS = {
    'sd': 'SD-MCTS\n(graph edited)',
    'sd_no_update': 'SD-MCTS\n(no editing)',
    'blackbox': 'BlackBox-MCTS',
}


def load_latest_results():
    """Find and load the latest results.json file."""
    if not RESULTS_DIR.exists():
        print(f"Error: Directory {RESULTS_DIR} does not exist.")
        return None

    subdirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not subdirs:
        print(f"Error: No result directories found in {RESULTS_DIR}")
        return None

    for latest_dir in subdirs:
        results_file = latest_dir / "results.json"
        if results_file.exists():
            print(f"Loading results from: {results_file}")
            with open(results_file, 'r') as f:
                data = json.load(f)
            # Check if this is a three-variant result
            first = data[0]
            inv_key = list(first.keys())[1]  # first intervention key after 'seed'
            if 'sd_no_update' in first[inv_key].get('policy', {}):
                return data, latest_dir
            else:
                print(f"  (skipping {latest_dir.name}: no sd_no_update data)")
                continue

    print(f"Error: No three-variant results.json found in {RESULTS_DIR}")
    return None


def has_sd_no_update(results):
    """Check if results contain sd_no_update variant."""
    try:
        return 'sd_no_update' in results[0]['none']['policy']
    except (KeyError, IndexError):
        return False


def plot_intervention_performance(results, output_dir):
    """Plot bar chart for policy returns (3 variants)."""
    interventions = ['none', 'capacity_shock', 'comm_severed']
    three_variant = has_sd_no_update(results)
    n_seeds = len(results)

    sd_means, sd_stds = [], []
    sd_nu_means, sd_nu_stds = [], []
    bb_means, bb_stds = [], []

    for inv in interventions:
        sd_vals = [-r[inv]["policy"]["sd"]["mean"] for r in results]
        bb_vals = [-r[inv]["policy"]["bb"]["mean"] for r in results]
        sd_means.append(np.mean(sd_vals))
        sd_stds.append(np.std(sd_vals) / np.sqrt(n_seeds))
        bb_means.append(np.mean(bb_vals))
        bb_stds.append(np.std(bb_vals) / np.sqrt(n_seeds))

        if three_variant:
            sd_nu_vals = [-r[inv]["policy"]["sd_no_update"]["mean"] for r in results]
            sd_nu_means.append(np.mean(sd_nu_vals))
            sd_nu_stds.append(np.std(sd_nu_vals) / np.sqrt(n_seeds))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(interventions))

    if three_variant:
        width = 0.25
        ax.bar(x - width, sd_means, width, yerr=sd_stds,
               label=VARIANT_LABELS['sd'], color=COLORS['sd'], alpha=0.9, capsize=4)
        ax.bar(x, sd_nu_means, width, yerr=sd_nu_stds,
               label=VARIANT_LABELS['sd_no_update'], color=COLORS['sd_no_update'],
               alpha=0.9, capsize=4, edgecolor=COLORS['sd'], linewidth=1.0, linestyle='--')
        ax.bar(x + width, bb_means, width, yerr=bb_stds,
               label=VARIANT_LABELS['blackbox'], color=COLORS['blackbox'], alpha=0.9, capsize=4)

        # Annotations
        for i, inv in enumerate(interventions):
            if inv != 'none':
                # Edit gain
                eg = (sd_nu_means[i] - sd_means[i]) / sd_nu_means[i] * 100
                ax.annotate(f'Edit gain\n{eg:.0f}%',
                            xy=(x[i] - width/2, sd_means[i]),
                            xytext=(x[i] - width - 0.15, sd_means[i] + 200),
                            fontsize=8, color=COLORS['sd'], fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=COLORS['sd'], lw=1.2))
                # Structural gain
                sg = bb_means[i] / sd_nu_means[i]
                mid_y = max(sd_nu_means[i], bb_means[i]) * 1.03
                ax.text(x[i] + width/2, mid_y, f'{sg:.1f}x\nStructural',
                        ha='center', fontsize=8, color='#666', fontweight='bold')
    else:
        width = 0.35
        ax.bar(x - width/2, sd_means, width, yerr=sd_stds,
               label='SD-MCTS', color=COLORS['sd'], alpha=0.9, capsize=5)
        ax.bar(x + width/2, bb_means, width, yerr=bb_stds,
               label='BlackBox-MCTS', color=COLORS['blackbox'], alpha=0.9, capsize=5)

    ax.set_ylabel('Total Cost (Lower is Better)')
    ax.set_title('Robustness to Causal Interventions: Two Sources of Advantage', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[i] for i in interventions])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_intervention_cost.png', dpi=300)
    plt.savefig(output_dir / 'fig_intervention_cost.pdf')
    print(f"Saved: {output_dir / 'fig_intervention_cost.png'}")
    plt.close()


def plot_intervention_rmse(results, output_dir):
    """Plot bar chart for RMSE prediction errors (3 variants)."""
    interventions = ['none', 'capacity_shock', 'comm_severed']
    three_variant = has_sd_no_update(results)
    n_seeds = len(results)

    sd_means, sd_stds = [], []
    sd_nu_means, sd_nu_stds = [], []
    bb_means, bb_stds = [], []

    for inv in interventions:
        sd_vals = [r[inv]["rmse"]["sd"] for r in results]
        bb_vals = [r[inv]["rmse"]["bb"] for r in results]
        sd_means.append(np.mean(sd_vals))
        sd_stds.append(np.std(sd_vals) / np.sqrt(n_seeds))
        bb_means.append(np.mean(bb_vals))
        bb_stds.append(np.std(bb_vals) / np.sqrt(n_seeds))

        if three_variant:
            sd_nu_vals = [r[inv]["rmse"]["sd_no_update"] for r in results]
            sd_nu_means.append(np.mean(sd_nu_vals))
            sd_nu_stds.append(np.std(sd_nu_vals) / np.sqrt(n_seeds))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(interventions))

    if three_variant:
        width = 0.25
        ax.bar(x - width, sd_means, width, yerr=sd_stds,
               label=VARIANT_LABELS['sd'], color=COLORS['sd'], alpha=0.9, capsize=4)
        ax.bar(x, sd_nu_means, width, yerr=sd_nu_stds,
               label=VARIANT_LABELS['sd_no_update'], color=COLORS['sd_no_update'],
               alpha=0.9, capsize=4, edgecolor=COLORS['sd'], linewidth=1.0, linestyle='--')
        ax.bar(x + width, bb_means, width, yerr=bb_stds,
               label=VARIANT_LABELS['blackbox'], color=COLORS['blackbox'], alpha=0.9, capsize=4)

        # Annotate RMSE values
        for i, inv in enumerate(interventions):
            # SD (graph edited) is always 0 — mark it explicitly so readers know it's not missing
            ax.plot(x[i] - width, 0, marker='^', markersize=8, color=COLORS['sd'],
                    zorder=5, clip_on=False)
            ax.text(x[i] - width, 0.4, '0.00', ha='center', fontsize=8,
                    color=COLORS['sd'], fontweight='bold')
            # SD_no_update values
            if sd_nu_means[i] > 0.5:
                ax.text(x[i], sd_nu_means[i] + 0.5, f'{sd_nu_means[i]:.1f}',
                        ha='center', fontsize=9, color=COLORS['sd'], fontweight='bold')
    else:
        width = 0.35
        ax.bar(x - width/2, sd_means, width, yerr=sd_stds,
               label='SD Model', color=COLORS['sd'], alpha=0.9, capsize=5)
        ax.bar(x + width/2, bb_means, width, yerr=bb_stds,
               label='BlackBox MLP', color=COLORS['blackbox'], alpha=0.9, capsize=5)

    ax.set_ylabel('RMSE (t=10..24 Evaluation Window)')
    ax.set_title('World Model Prediction Error Under Interventions', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[i] for i in interventions])
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_intervention_rmse.png', dpi=300)
    plt.savefig(output_dir / 'fig_intervention_rmse.pdf')
    print(f"Saved: {output_dir / 'fig_intervention_rmse.png'}")
    plt.close()


def main():
    data = load_latest_results()
    if not data:
        return

    results, output_dir = data
    plot_intervention_performance(results, output_dir)
    plot_intervention_rmse(results, output_dir)
    print("\nAll causal intervention figures generated successfully.")


if __name__ == "__main__":
    main()
