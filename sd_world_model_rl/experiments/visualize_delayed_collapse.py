#!/usr/bin/env python
"""Visualization for Delayed Collapse experiment."""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_delayed_collapse_bars(output_path: str):
    """Plot bar chart for delayed collapse scenario."""
    methods = ['MPC\n(wrong belief)', 'SD-MCTS', 'Optimal']
    rewards = [9.0, 15.0, 24.8]
    colors = ['#d62728', '#1f77b4', '#2ca02c']

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(methods))
    bars = ax.bar(x, rewards, color=colors, edgecolor='black', linewidth=1.2)

    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, reward + 0.5,
                f'{reward:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement annotation
    ax.annotate('', xy=(1, 15), xytext=(0, 9),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.5, 12.5, '+66.7%', ha='center', fontsize=11, color='green', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Delayed Collapse + Demand Surge Scenario', fontsize=14)
    ax.set_ylim(0, 30)
    ax.grid(True, axis='y', alpha=0.3)

    # Add collapse/bonus info
    ax.text(0, -3.5, 'Collapse: 100%\nSurge Bonus: 0%', ha='center', fontsize=9, color='#d62728')
    ax.text(1, -3.5, 'Collapse: 0%\nSurge Bonus: 100%', ha='center', fontsize=9, color='#1f77b4')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved delayed collapse bars to: {output_path}")


def plot_fatigue_trajectory(output_path: str):
    """Plot fatigue trajectory comparison."""
    T = 30
    T_s = 12
    F_true = 10.0
    F_req = 5.0
    alpha, beta = 1.0, 0.1

    # MPC trajectory (greedy a=1)
    mpc_fatigue = [0.0]
    mpc_alive = [True]
    for t in range(T):
        if mpc_alive[-1]:
            new_f = mpc_fatigue[-1] + (alpha + beta) * 1.0 - beta
            new_f = max(0, new_f)
            mpc_fatigue.append(new_f)
            mpc_alive.append(new_f < F_true)
        else:
            mpc_fatigue.append(mpc_fatigue[-1])
            mpc_alive.append(False)

    # SD-MCTS trajectory (safe intensity a=0.47)
    a_safe = (F_req + beta * T_s) / ((alpha + beta) * T_s)
    sdmcts_fatigue = [0.0]
    for t in range(T):
        if t < T_s:
            new_f = sdmcts_fatigue[-1] + (alpha + beta) * a_safe - beta
        else:
            new_f = sdmcts_fatigue[-1] + (alpha + beta) * 1.0 - beta
        new_f = max(0, new_f)
        sdmcts_fatigue.append(new_f)

    fig, ax = plt.subplots(figsize=(10, 5))

    t_range = np.arange(T + 1)
    ax.plot(t_range, mpc_fatigue, 'r-', linewidth=2, label='MPC (greedy a=1)')
    ax.plot(t_range, sdmcts_fatigue, 'b-', linewidth=2, label=f'SD-MCTS (a={a_safe:.2f})')

    # Thresholds
    ax.axhline(y=F_true, color='red', linestyle='--', alpha=0.7, label=f'Collapse F={F_true}')
    ax.axhline(y=F_req, color='green', linestyle='--', alpha=0.7, label=f'Surge Req F_req={F_req}')

    # Surge time
    ax.axvline(x=T_s, color='orange', linestyle=':', alpha=0.7, label=f'Surge T_s={T_s}')
    ax.fill_between([T_s-0.5, T_s+0.5], 0, 25, alpha=0.2, color='orange')

    # Mark collapse point
    collapse_t = next(i for i, a in enumerate(mpc_alive) if not a)
    ax.scatter([collapse_t], [F_true], color='red', s=100, zorder=5, marker='X')
    ax.annotate('MPC Collapse', xy=(collapse_t, F_true), xytext=(collapse_t+2, F_true+2),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

    # Mark surge bonus
    ax.scatter([T_s], [sdmcts_fatigue[T_s]], color='blue', s=100, zorder=5, marker='*')
    ax.annotate('Surge Bonus\n+15', xy=(T_s, sdmcts_fatigue[T_s]), xytext=(T_s+2, sdmcts_fatigue[T_s]-3),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10, color='blue')

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Fatigue', fontsize=12)
    ax.set_title('Fatigue Trajectory: MPC vs SD-MCTS', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved fatigue trajectory to: {output_path}")


def plot_reward_accumulation(output_path: str):
    """Plot cumulative reward comparison."""
    T = 30
    T_s = 12
    F_true = 10.0
    F_req = 5.0
    alpha, beta = 1.0, 0.1
    B = 15.0

    # MPC rewards (greedy, collapses at t=10)
    mpc_rewards = []
    fatigue = 0
    for t in range(T):
        if fatigue >= F_true:
            mpc_rewards.append(0)
        else:
            mpc_rewards.append(1.0)  # a=1
            fatigue += (alpha + beta) * 1.0 - beta

    # SD-MCTS rewards (safe intensity)
    a_safe = (F_req + beta * T_s) / ((alpha + beta) * T_s)
    sdmcts_rewards = []
    for t in range(T):
        if t < T_s:
            r = a_safe
            if t == T_s - 1:
                r += B  # Surge bonus
            sdmcts_rewards.append(r)
        else:
            sdmcts_rewards.append(1.0)

    # Cumulative rewards
    mpc_cumsum = np.cumsum(mpc_rewards)
    sdmcts_cumsum = np.cumsum(sdmcts_rewards)

    fig, ax = plt.subplots(figsize=(10, 5))

    t_range = np.arange(T)
    ax.plot(t_range, mpc_cumsum, 'r-', linewidth=2, label='MPC (greedy)')
    ax.plot(t_range, sdmcts_cumsum, 'b-', linewidth=2, label='SD-MCTS')

    # Mark surge event
    ax.axvline(x=T_s-1, color='orange', linestyle=':', alpha=0.7)
    ax.annotate('Surge Bonus\n(+15 for SD-MCTS)', xy=(T_s-1, sdmcts_cumsum[T_s-1]),
                xytext=(T_s+3, sdmcts_cumsum[T_s-1]-5),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=10)

    # Mark MPC plateau (after collapse)
    collapse_t = 10
    ax.annotate('MPC Collapse\n(no more rewards)', xy=(collapse_t, mpc_cumsum[collapse_t]),
                xytext=(collapse_t+3, mpc_cumsum[collapse_t]+3),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Reward Accumulation: MPC vs SD-MCTS', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved reward accumulation to: {output_path}")


if __name__ == "__main__":
    output_dir = "/home/data/round/code/sd_world_model_rl/results/figures"
    os.makedirs(output_dir, exist_ok=True)

    plot_delayed_collapse_bars(os.path.join(output_dir, "delayed_collapse_bars.png"))
    plot_fatigue_trajectory(os.path.join(output_dir, "delayed_collapse_trajectory.png"))
    plot_reward_accumulation(os.path.join(output_dir, "delayed_collapse_rewards.png"))

    print("\nAll delayed collapse figures saved.")
