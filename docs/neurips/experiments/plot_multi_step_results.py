#!/usr/bin/env python3
"""
Generate publication-quality figures for multi-step rollout error analysis.

Figures:
1. Error growth curves (linear and log scale)
2. Relative error growth (normalized to H=1)
3. Error growth rate bar chart
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# NeurIPS-style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# Data paths
CSV_PATH = Path("/home/data/round/code/sd_world_model_rl/docs/neurips/results/multi_step_rollout/20260215_213008/error_curves.csv")
OUTPUT_DIR = Path("/home/data/round/code/sd_world_model_rl/docs/neurips/results/multi_step_rollout/20260215_213008/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Distribution names for display
DIST_NAMES = {
    'id': 'In-Distribution',
    'backlog_shock': 'Backlog Shock',
    'inventory_surplus': 'Inventory Surplus',
    'pipeline_imbalance': 'Pipeline Imbalance',
    'alternating_extreme': 'Alternating Extreme',
}

# Colors
COLORS = {
    'sd': '#2E7D32',      # Green for SD
    'blackbox': '#C62828',  # Red for BlackBox
    'id': '#1976D2',
    'backlog_shock': '#F57C00',
    'inventory_surplus': '#7B1FA2',
    'pipeline_imbalance': '#D32F2F',
    'alternating_extreme': '#388E3C',
}

def load_and_aggregate():
    """Load CSV and aggregate across seeds."""
    df = pd.read_csv(CSV_PATH)

    # Aggregate: mean across seeds
    agg = df.groupby(['distribution', 'model', 'horizon'])['rmse'].agg(['mean', 'std']).reset_index()
    agg.rename(columns={'mean': 'rmse', 'std': 'rmse_std'}, inplace=True)

    return agg

def plot_error_growth_curves(df):
    """Figure 1: Error growth curves for ID and Backlog Shock (main result)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Focus on main distributions
    main_dists = ['id', 'backlog_shock']

    for ax, dist in zip([ax1, ax2], main_dists):
        dist_data = df[df['distribution'] == dist]

        for model in ['sd', 'blackbox']:
            model_data = dist_data[dist_data['model'] == model].sort_values('horizon')
            horizons = model_data['horizon'].values
            rmse = model_data['rmse'].values
            label = 'SD-MCTS' if model == 'sd' else 'BlackBox-MCTS'
            color = COLORS[model]

            ax.plot(horizons, rmse, 'o-', label=label, color=color, markersize=5)

        ax.set_xlabel('Rollout Horizon (steps)')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{DIST_NAMES[dist]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 52)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_error_growth_main.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_error_growth_main.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'fig1_error_growth_main.png'}")
    plt.close()

def plot_error_growth_all_dists(df):
    """Figure 2: Error growth for all distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dist in DIST_NAMES.keys():
        dist_data = df[df['distribution'] == dist]

        # Only BlackBox (SD is always 0)
        bb_data = dist_data[dist_data['model'] == 'blackbox'].sort_values('horizon')
        horizons = bb_data['horizon'].values
        rmse = bb_data['rmse'].values

        ax.plot(horizons, rmse, 'o-', label=DIST_NAMES[dist],
                color=COLORS[dist], markersize=5, alpha=0.8)

    ax.set_xlabel('Rollout Horizon (steps)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Multi-Step Rollout Error: BlackBox-MCTS', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 52)

    # Add note about SD
    ax.text(0.98, 0.02, 'SD-MCTS error = 0.0000 across all horizons (not shown)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_error_growth_all.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_error_growth_all.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'fig2_error_growth_all.png'}")
    plt.close()

def plot_relative_error_growth(df):
    """Figure 3: Relative error growth (normalized to H=1)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for dist in ['id', 'backlog_shock']:  # Focus on main distributions
        dist_data = df[df['distribution'] == dist]
        bb_data = dist_data[dist_data['model'] == 'blackbox'].sort_values('horizon')

        horizons = bb_data['horizon'].values
        rmse = bb_data['rmse'].values

        # Normalize to H=1
        h1_rmse = rmse[horizons == 1][0]
        relative_rmse = rmse / h1_rmse

        ax.plot(horizons, relative_rmse, 'o-', label=DIST_NAMES[dist],
                color=COLORS[dist], markersize=6, linewidth=2)

    # Add reference line for exponential growth at 5% per step
    h_ref = np.array([1, 50])
    y_ref = np.exp(0.05 * (h_ref - 1))
    ax.plot(h_ref, y_ref, '--', color='gray', alpha=0.5, label='5% exp. growth reference')

    ax.set_xlabel('Rollout Horizon (steps)', fontsize=12)
    ax.set_ylabel('Relative RMSE (H=1 baseline)', fontsize=12)
    ax.set_title('Error Accumulation: BlackBox-MCTS', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 52)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_relative_error_growth.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_relative_error_growth.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'fig3_relative_error_growth.png'}")
    plt.close()

def plot_growth_rate_comparison():
    """Figure 4: Exponential growth rate comparison."""
    # Growth rate data from results
    growth_rates = {
        'ID': 0.0635,
        'Backlog Shock': 0.0534,
        'Inventory Surplus': 0.0424,
        'Pipeline Imbalance': 0.0289,
        'Alternating Extreme': 0.0368,
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    dists = list(growth_rates.keys())
    rates = [growth_rates[d] * 100 for d in dists]  # Convert to percentage

    colors = [COLORS[d.lower().replace(' ', '_')] for d in dists]

    bars = ax.bar(dists, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Exponential Growth Rate (% per step)', fontsize=12)
    ax.set_title('BlackBox-MCTS Error Growth Rate by Distribution', fontsize=14)
    ax.set_xlabel('Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add note about SD
    ax.text(0.98, 0.02, 'SD-MCTS growth rate = 0.00% (structural constraints)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color='green',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_growth_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_growth_rate_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'fig4_growth_rate_comparison.png'}")
    plt.close()

def plot_comparison_summary(df):
    """Figure 5: Side-by-side comparison of SD vs BB at key horizons."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    horizons = [1, 10, 20, 30, 40, 50]

    for ax, h in zip(axes, horizons):
        h_data = df[df['horizon'] == h]

        dists = []
        sd_vals = []
        bb_vals = []

        for dist in DIST_NAMES.keys():
            dist_h = h_data[h_data['distribution'] == dist]
            sd_val = dist_h[dist_h['model'] == 'sd']['rmse'].values[0]
            bb_val = dist_h[dist_h['model'] == 'blackbox']['rmse'].values[0]

            dists.append(DIST_NAMES[dist])
            sd_vals.append(sd_val)
            bb_vals.append(bb_val)

        x = np.arange(len(dists))
        width = 0.35

        ax.bar(x - width/2, sd_vals, width, label='SD-MCTS', color=COLORS['sd'], alpha=0.8)
        ax.bar(x + width/2, bb_vals, width, label='BlackBox-MCTS', color=COLORS['blackbox'], alpha=0.8)

        ax.set_ylabel('RMSE')
        ax.set_title(f'Horizon = {h}')
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace(' ', '\n') for d in dists], fontsize=8, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add ratio annotation
        for i, (sd, bb) in enumerate(zip(sd_vals, bb_vals)):
            if sd > 0:
                ratio = bb / sd
            else:
                ratio = float('inf')
            ax.text(i, max(sd, bb) * 1.05, f'{bb/sd:.0f}x' if sd > 0 else '∞',
                   ha='center', fontsize=7, color='red')

    plt.suptitle('RMSE Comparison: SD-MCTS vs BlackBox-MCTS', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_comparison_summary.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'fig5_comparison_summary.png'}")
    plt.close()

def main():
    """Generate all figures."""
    print("Loading data...")
    df = load_and_aggregate()

    print("\nGenerating figures...")
    print("  Figure 1: Error growth curves (main distributions)")
    plot_error_growth_curves(df)

    print("  Figure 2: Error growth (all distributions)")
    plot_error_growth_all_dists(df)

    print("  Figure 3: Relative error growth (log scale)")
    plot_relative_error_growth(df)

    print("  Figure 4: Growth rate comparison")
    plot_growth_rate_comparison()

    print("  Figure 5: Side-by-side comparison")
    plot_comparison_summary(df)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for dist in ['id', 'backlog_shock', 'inventory_surplus']:
        dist_data = df[df['distribution'] == dist]
        bb_h1 = dist_data[(dist_data['model'] == 'blackbox') & (dist_data['horizon'] == 1)]['rmse'].values[0]
        bb_h50 = dist_data[(dist_data['model'] == 'blackbox') & (dist_data['horizon'] == 50)]['rmse'].values[0]

        print(f"\n{DIST_NAMES[dist]}:")
        print(f"  BB H=1:  {bb_h1:.4f}")
        print(f"  BB H=50: {bb_h50:.4f}")
        print(f"  Growth:  {bb_h50/bb_h1:.1f}x")

if __name__ == "__main__":
    main()
