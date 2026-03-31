#!/usr/bin/env python3
"""
NeurIPS Experiment: Demand Pattern Transfer (Zero-Shot Generalization)

P1 Priority: Test zero-shot transfer across different demand patterns.
Train on step demand, test on sinusoidal/random-walk/bursty without retraining.

Key hypothesis: SD-MCTS's explicit stock-flow structure enables robust
generalization to unseen demand patterns, while black-box models overfit.

Design:
- Reuses proven training infrastructure from run_beer_game_v2.py
- Injects custom demand patterns via lightweight wrapper
- Maintains all critical hyperparameters for consistency
"""
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add project paths (matching run_ablation_principled.py)
SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent.networks import ValueNetwork
from src.planning.mcts import mcts_search

from experiments.run_beer_game_v2 import (
    train_sd_mcts,
    evaluate_sd_mcts,
    compute_statistics,
    BeerGameSDModelWrapper,
    NormalizedValueNet,
    ReplayBuffer,
    Transition,
    make_beer_game_reward_fn,
)
from baselines.blackbox_world_model import BlackBoxWorldModel
from demand_patterns import create_demand_pattern

import torch.nn.functional as F
import torch.optim as optim
import time


# =============================================================================
# Experiment Configuration
# =============================================================================

CONFIG = {
    "experiment_name": "demand_transfer",
    "train_pattern": "step",
    "test_patterns": ["step", "sinusoidal", "random_walk", "bursty"],
    "methods": ["sd_mcts", "blackbox_mcts"],  # Compare SD vs BlackBox

    # Training config (matching run_ablation_principled.py)
    "train_budget": 30000,
    "order_levels": 3,  # Flat action space: 3^4 = 81 actions
    "num_simulations": 200,
    "gamma": 0.99,
    "device": "cpu",

    # Evaluation config
    "n_seeds": 5,
    "eval_episodes": 30,

    # Environment config
    "n_echelons": 4,
    "tau_info": 2,
    "tau_ship": 2,
    "T": 36,
    "h": 0.5,
    "p": 1.0,
}


# =============================================================================
# Environment Wrapper for Custom Demand Patterns
# =============================================================================

class DemandPatternEnv(BeerGameEnvFlat):
    """Beer Game environment with pluggable demand patterns.

    Injects custom demand generator while preserving all existing
    infrastructure (flat action space, metrics, etc.)
    """

    def __init__(self, demand_pattern, **kwargs):
        self.demand_pattern = demand_pattern
        super().__init__(**kwargs)

    def _get_customer_demand(self) -> int:
        """Override to use pluggable demand pattern."""
        return self.demand_pattern.get_demand()

    def reset(self, *, seed=None, options=None):
        """Reset both environment and demand pattern."""
        self.demand_pattern.reset()
        return super().reset(seed=seed, options=options)


def create_env_with_pattern(pattern_type: str, seed: int = None) -> DemandPatternEnv:
    """Create Beer Game environment with specific demand pattern."""
    demand_pattern = create_demand_pattern(pattern_type, seed=seed)

    return DemandPatternEnv(
        demand_pattern=demand_pattern,
        n_echelons=CONFIG["n_echelons"],
        tau_info=CONFIG["tau_info"],
        tau_ship=CONFIG["tau_ship"],
        order_levels=CONFIG["order_levels"],
        T=CONFIG["T"],
        h=CONFIG["h"],
        p=CONFIG["p"],
        seed=seed,
    )


# =============================================================================
# Training Functions (reuse patterns from run_ablation_principled.py)
# =============================================================================

def train_blackbox_mcts(
    env: BeerGameEnvFlat,
    env_steps: int = 5000,
    gamma: float = 0.99,
    device: str = 'cpu',
    verbose: bool = True,
):
    """Train BlackBox-MCTS on Beer Game.

    Directly adapted from run_ablation_principled.py train_and_eval()
    with variant_config["model"] = "mlp".
    """
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create MLP world model (no structural priors)
    world_model = BlackBoxWorldModel(
        state_dim=state_dim,
        action_dim=n_actions,
        hidden_dim=256,
        num_layers=3,
    ).to(device)
    model_opt = optim.Adam(world_model.parameters(), lr=1e-3)

    # Value network (same as SD-MCTS)
    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    # Return normalization (critical for stable training)
    return_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def get_return_stats():
        return return_stats['mean'], return_stats['std']

    normalized_vnet = NormalizedValueNet(value_net, get_return_stats)

    # Hyperparameters (matching run_ablation_principled.py)
    epsilon = 0.3
    random_steps = min(2000, max(500, env_steps // 3))
    num_sims = 200
    rollout_depth = 10

    episode_rewards = []
    steps = 0
    episode = 0

    if verbose:
        print(f"Training BlackBox-MCTS (steps={env_steps}, sims={num_sims})...")

    while steps < env_steps:
        state, _ = env.reset(seed=episode)
        episode_reward = 0.0
        episode_transitions = []

        for t in range(env.T):
            # Epsilon-greedy exploration
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model,
                    n_actions=n_actions,
                    num_simulations=num_sims,
                    value_net=normalized_vnet,
                    reward_fn=reward_fn,
                    rollout_depth=rollout_depth,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            transition = Transition(
                state=state.copy(), action=action, reward=reward,
                next_state=next_state.copy(), done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated:
                break

        # Compute returns for value learning
        G = 0.0
        returns = []
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        # Update return statistics
        for r in returns:
            return_stats['count'] += 1
            delta = r - return_stats['mean']
            return_stats['mean'] += delta / return_stats['count']
            if return_stats['count'] > 1:
                return_stats['std'] = max(1.0, np.sqrt(
                    ((return_stats['count'] - 1) * return_stats['std']**2 + delta * (r - return_stats['mean']))
                    / return_stats['count']
                ))

        # Train world model (5 updates per episode, matching SD-MCTS)
        if len(buffer) >= 32:
            for _ in range(5):
                batch = buffer.sample(32)
                states_t = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
                actions_t = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                next_states_t = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)

                pred_next = world_model(states_t, actions_t)
                loss = F.mse_loss(pred_next, next_states_t)
                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

        # Train value network (normalized returns)
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(
                np.array([t.state for t in episode_transitions]),
                dtype=torch.float32, device=device
            )
            normalized_returns = [(r - return_stats['mean']) / return_stats['std'] for r in returns]
            ep_returns = torch.tensor(normalized_returns, dtype=torch.float32, device=device)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode_rewards.append(episode_reward)
        episode += 1

        if verbose and episode % 20 == 0:
            avg_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            print(f"  Step {steps}: Avg10={avg_10:.2f}")

    return world_model, None, value_net, episode_rewards


def evaluate_blackbox_mcts(
    env: BeerGameEnvFlat,
    world_model: BlackBoxWorldModel,
    value_net: ValueNetwork,
    n_episodes: int = 10,
    gamma: float = 0.99,
    device: str = 'cpu',
    num_simulations: int = 200,
) -> dict:
    """Evaluate trained BlackBox-MCTS model."""
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)
    n_actions = env.action_space.n

    total_costs = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=1000 + ep)
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=world_model,
                n_actions=n_actions,
                num_simulations=num_simulations,
                value_net=value_net,
                reward_fn=reward_fn,
                rollout_depth=10,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        metrics = env.get_metrics()
        total_costs.append(metrics['total_cost'])

    cost_stats = compute_statistics(total_costs, "cost")

    return {
        'total_cost_mean': cost_stats['mean'],
        'total_cost_std': cost_stats['std'],
        'total_cost_se': cost_stats['se'],
        'total_costs': total_costs,
    }


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class TransferResult:
    """Result for one (method, train_pattern, test_pattern, seed) combination."""
    method: str
    train_pattern: str
    test_pattern: str
    seed: int
    train_cost_mean: float
    train_cost_std: float
    test_cost_mean: float
    test_cost_std: float
    transfer_ratio: float  # test_cost / train_cost (lower is better)
    cost_increase: float  # (test_cost - train_cost) / train_cost

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


def run_single_experiment(
    method: str,
    test_pattern: str,
    seed: int,
) -> TransferResult:
    """Run single demand transfer experiment.

    Flow:
    1. Train on step demand using train_sd_mcts()
    2. Evaluate on step demand (baseline)
    3. Evaluate on test_pattern (zero-shot)
    4. Compute transfer metrics
    """
    print(f"\n{'='*60}")
    print(f"Method: {method}, Test: {test_pattern}, Seed: {seed}")
    print(f"{'='*60}")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create training environment (step demand)
    train_env = create_env_with_pattern(CONFIG["train_pattern"], seed=seed)

    # Train agent based on method
    print(f"Training {method} on {CONFIG['train_pattern']} demand ({CONFIG['train_budget']} steps)...")

    if method == "sd_mcts":
        world_model, policy_net, value_net, episode_rewards = train_sd_mcts(
            env=train_env,
            env_steps=CONFIG["train_budget"],
            gamma=CONFIG["gamma"],
            device=CONFIG["device"],
            verbose=False,
        )
        # Evaluate using SD-MCTS evaluator
        print(f"Evaluating on {CONFIG['train_pattern']} (in-distribution)...")
        train_eval = evaluate_sd_mcts(
            env=train_env,
            world_model=world_model,
            policy_net=None,
            value_net=value_net,
            n_episodes=CONFIG["eval_episodes"],
            gamma=CONFIG["gamma"],
            device=CONFIG["device"],
            num_simulations=CONFIG["num_simulations"],
        )

        # Create test environment (different demand pattern)
        test_env = create_env_with_pattern(test_pattern, seed=seed + 10000)

        # Evaluate on test pattern (zero-shot transfer)
        print(f"Testing on {test_pattern} (zero-shot transfer)...")
        test_eval = evaluate_sd_mcts(
            env=test_env,
            world_model=world_model,
            policy_net=None,
            value_net=value_net,
            n_episodes=CONFIG["eval_episodes"],
            gamma=CONFIG["gamma"],
            device=CONFIG["device"],
            num_simulations=CONFIG["num_simulations"],
        )

    elif method == "blackbox_mcts":
        world_model, _, value_net, episode_rewards = train_blackbox_mcts(
            env=train_env,
            env_steps=CONFIG["train_budget"],
            gamma=CONFIG["gamma"],
            device=CONFIG["device"],
            verbose=False,
        )
        # Evaluate using BlackBox evaluator
        print(f"Evaluating on {CONFIG['train_pattern']} (in-distribution)...")
        train_eval = evaluate_blackbox_mcts(
            env=train_env,
            world_model=world_model,
            value_net=value_net,
            n_episodes=CONFIG["eval_episodes"],
            gamma=CONFIG["gamma"],
            device=CONFIG["device"],
            num_simulations=CONFIG["num_simulations"],
        )

        # Create test environment (different demand pattern)
        test_env = create_env_with_pattern(test_pattern, seed=seed + 10000)

        # Evaluate on test pattern (zero-shot transfer)
        print(f"Testing on {test_pattern} (zero-shot transfer)...")
        test_eval = evaluate_blackbox_mcts(
            env=test_env,
            world_model=world_model,
            value_net=value_net,
            n_episodes=CONFIG["eval_episodes"],
            gamma=CONFIG["gamma"],
            device=CONFIG["device"],
            num_simulations=CONFIG["num_simulations"],
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute transfer metrics
    train_cost = train_eval["total_cost_mean"]
    test_cost = test_eval["total_cost_mean"]
    transfer_ratio = test_cost / train_cost if train_cost != 0 else float('inf')
    cost_increase = (test_cost - train_cost) / train_cost if train_cost != 0 else float('inf')

    print(f"  Train Cost: {train_cost:.1f} ± {train_eval['total_cost_std']:.1f}")
    print(f"  Test Cost:  {test_cost:.1f} ± {test_eval['total_cost_std']:.1f}")
    print(f"  Transfer Ratio: {transfer_ratio:.2f}x")
    print(f"  Cost Increase: {cost_increase*100:+.1f}%")

    return TransferResult(
        method=method,
        train_pattern=CONFIG["train_pattern"],
        test_pattern=test_pattern,
        seed=seed,
        train_cost_mean=train_cost,
        train_cost_std=train_eval["total_cost_std"],
        test_cost_mean=test_cost,
        test_cost_std=test_eval["total_cost_std"],
        transfer_ratio=transfer_ratio,
        cost_increase=cost_increase,
    )


def run_full_experiment():
    """Run full demand transfer experiment."""
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = NEURIPS_DIR / "results" / "demand_transfer" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(CONFIG["methods"]) * len(CONFIG["test_patterns"]) * CONFIG["n_seeds"]
    current_run = 0

    for method in CONFIG["methods"]:
        for test_pattern in CONFIG["test_patterns"]:
            for seed in range(CONFIG["n_seeds"]):
                current_run += 1
                print(f"\n\n[Progress: {current_run}/{total_runs}]")

                try:
                    result = run_single_experiment(method, test_pattern, seed)
                    results.append(result.to_dict())

                    # Save incremental results
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "method": method,
                        "test_pattern": test_pattern,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Generate summary
    generate_summary(results, output_dir)

    print(f"\n\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(results: List[Dict], output_dir: Path):
    """Generate transfer performance summary tables."""
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("No valid results to summarize!")
        return

    lines = ["# Demand Pattern Transfer - Zero-Shot Generalization\n\n"]
    lines.append(f"**Train Pattern**: {CONFIG['train_pattern']}\n")
    lines.append(f"**Training Budget**: {CONFIG['train_budget']} steps\n")
    lines.append(f"**Seeds**: {CONFIG['n_seeds']}\n")
    lines.append(f"**Eval Episodes**: {CONFIG['eval_episodes']}\n\n")

    # Aggregate results by (method, test_pattern)
    summary = {}
    for r in valid_results:
        key = (r["method"], r["test_pattern"])
        if key not in summary:
            summary[key] = {
                "transfer_ratios": [],
                "cost_increases": [],
                "test_costs": [],
                "train_costs": [],
            }
        summary[key]["transfer_ratios"].append(r["transfer_ratio"])
        summary[key]["cost_increases"].append(r["cost_increase"])
        summary[key]["test_costs"].append(r["test_cost_mean"])
        summary[key]["train_costs"].append(r["train_cost_mean"])

    # Table 1: Transfer Ratios (lower is better)
    lines.append("## Table 1: Transfer Ratio (Test Cost / Train Cost)\n\n")
    lines.append("| Method | Step | Sinusoidal | Random Walk | Bursty | Avg |\n")
    lines.append("|--------|------|------------|-------------|--------|-----|\n")

    for method in CONFIG["methods"]:
        row = [method]
        ratios = []
        for pattern in CONFIG["test_patterns"]:
            key = (method, pattern)
            if key in summary:
                data = summary[key]["transfer_ratios"]
                stats = compute_statistics(data, f"{method}_{pattern}")
                ratio = stats["mean"]
                se = stats["se"]
                ratios.append(ratio)
                row.append(f"{ratio:.2f}±{se:.2f}")
            else:
                row.append("N/A")

        avg_ratio = np.mean(ratios) if ratios else 0
        row.append(f"**{avg_ratio:.2f}**")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n*Lower is better. 1.0 = same performance as training pattern.*\n\n")

    # Table 2: Absolute Costs
    lines.append("## Table 2: Absolute Costs (Mean ± SE)\n\n")
    lines.append("| Method | Step | Sinusoidal | Random Walk | Bursty |\n")
    lines.append("|--------|------|------------|-------------|--------|\n")

    for method in CONFIG["methods"]:
        row = [method]
        for pattern in CONFIG["test_patterns"]:
            key = (method, pattern)
            if key in summary:
                data = summary[key]["test_costs"]
                stats = compute_statistics(data, f"{method}_{pattern}")
                mean = stats["mean"]
                se = stats["se"]
                row.append(f"{mean:.1f}±{se:.1f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n")

    # Table 3: Cost Increase (relative to training)
    lines.append("## Table 3: Cost Increase vs Training Pattern (%)\n\n")
    lines.append("| Method | Sinusoidal | Random Walk | Bursty | Avg |\n")
    lines.append("|--------|------------|-------------|--------|-----|\n")

    for method in CONFIG["methods"]:
        row = [method]
        increases = []
        for pattern in ["sinusoidal", "random_walk", "bursty"]:
            key = (method, pattern)
            if key in summary:
                data = summary[key]["cost_increases"]
                stats = compute_statistics(data, f"{method}_{pattern}")
                inc = stats["mean"]
                se = stats["se"]
                increases.append(inc)
                row.append(f"{inc*100:+.1f}±{se*100:.1f}")
            else:
                row.append("N/A")

        avg_inc = np.mean(increases) if increases else 0
        row.append(f"**{avg_inc*100:+.1f}**")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n")

    # Key findings
    lines.append("## Key Findings\n\n")

    # Find best/worst transfer patterns
    for method in CONFIG["methods"]:
        method_ratios = {}
        for pattern in CONFIG["test_patterns"]:
            key = (method, pattern)
            if key in summary:
                method_ratios[pattern] = np.mean(summary[key]["transfer_ratios"])

        if method_ratios:
            best_pattern = min(method_ratios, key=method_ratios.get)
            worst_pattern = max(method_ratios, key=method_ratios.get)

            lines.append(f"**{method}**:\n")
            lines.append(f"- Best transfer: {best_pattern} ({method_ratios[best_pattern]:.2f}x)\n")
            lines.append(f"- Worst transfer: {worst_pattern} ({method_ratios[worst_pattern]:.2f}x)\n")

            # Compute average degradation on OOD patterns
            ood_patterns = [p for p in CONFIG["test_patterns"] if p != CONFIG["train_pattern"]]
            ood_ratios = [method_ratios[p] for p in ood_patterns if p in method_ratios]
            if ood_ratios:
                avg_ood_ratio = np.mean(ood_ratios)
                lines.append(f"- Avg OOD performance: {avg_ood_ratio:.2f}x training cost\n")

            lines.append("\n")

    # Write summary
    with open(output_dir / "summary.md", "w") as f:
        f.write("".join(lines))

    print(f"\nSummary written to: {output_dir / 'summary.md'}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demand Pattern Transfer Experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--budget", type=int, default=30000, help="Training budget (steps)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")

    args = parser.parse_args()

    # Update config
    CONFIG["n_seeds"] = args.seeds
    CONFIG["train_budget"] = args.budget
    CONFIG["device"] = args.device

    # Run experiment
    run_full_experiment()
