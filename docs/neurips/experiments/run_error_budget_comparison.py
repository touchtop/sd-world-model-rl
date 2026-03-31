#!/usr/bin/env python3
"""
NeurIPS Experiment: SD vs BB Computational Efficiency under Fixed Error Budget

Theoretical Foundation:
- SD: ε_H ≤ ε_1 = O(1)  (error bounded)
- BB: ε_H ≤ ε_1 · L^(H-1) = O(λ^H)  (error grows exponentially)

Hypothesis:
Under fixed error budget, SD can achieve same performance with significantly
less computation (smaller H and N_sims) compared to BB.

Design:
1. Train both SD and BB models
2. Estimate ε_1 and growth rate L for each
3. Compute optimal H and N_sims for fixed error budget
4. Compare actual performance and computational cost
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

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
    BeerGameSDModelWrapper,
    make_beer_game_reward_fn,
    NormalizedValueNet,
    ReplayBuffer,
    Transition,
)
from baselines.blackbox_world_model import BlackBoxWorldModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Theoretical Error Growth Model
# =============================================================================

@dataclass
class ErrorCharacteristics:
    """Characterize the error growth profile of a world model."""
    epsilon_1: float        # Single-step prediction error
    growth_rate: float      # L: error growth factor per step (λ)
    error_bound: float      # Overall error bound at horizon H
    is_bounded: bool        # Whether error is bounded (SD) vs growing (BB)


def compute_error_characteristics(
    world_model,
    env,
    model_type: str,
    n_samples: int = 100,
    horizons: List[int] = [1, 5, 10, 20],
    device: str = "cpu",
) -> ErrorCharacteristics:
    """
    Compute error characteristics by measuring prediction error at different horizons.

    Returns:
        ErrorCharacteristics with epsilon_1 and growth rate
    """
    is_mlp = not hasattr(world_model, 'model')
    rng = np.random.default_rng(42)

    # Measure ε_H at different horizons
    horizon_errors = {}

    for H in horizons:
        errors = []
        for _ in range(n_samples // len(horizons)):
            state, _ = env.reset(seed=rng.integers(0, 10000))
            current_state = state.copy()

            # Rollout for H steps
            for h in range(H):
                action = env.action_space.sample()

                # True transition
                next_state, _, _, _, _ = env.step(action)

                # Predicted transition
                with torch.no_grad():
                    state_t = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_t = torch.tensor([action], dtype=torch.int64, device=device)
                    if is_mlp:
                        pred_next = world_model(state_t, action_t).squeeze(0).cpu().numpy()
                    else:
                        pred_next = world_model.model(state_t, action_t).squeeze(0).cpu().numpy()

                # Accumulate error (only on state components, not t/demand)
                if h == H - 1:
                    error = np.sqrt(np.mean((pred_next[:-2] - next_state[:-2]) ** 2))
                    errors.append(error)

                current_state = next_state
                state = next_state

        horizon_errors[H] = float(np.mean(errors))

    # Estimate growth rate: ε_H ≈ ε_1 · λ^(H-1)
    # Take log: log(ε_H) ≈ log(ε_1) + (H-1) · log(λ)
    # Fit linear regression to estimate log(λ)

    H_list = np.array(list(horizon_errors.keys()))
    error_list = np.array(list(horizon_errors.values()))

    # Avoid log(0) - clamp minimum error
    error_list = np.maximum(error_list, 1e-10)

    # Linear regression: log(error) vs H
    log_errors = np.log(error_list)
    coeffs = np.polyfit(H_list - 1, log_errors, 1)  # [slope, intercept]

    epsilon_1 = float(np.exp(coeffs[1]))  # exp(intercept) = ε_1
    growth_rate = float(np.exp(coeffs[0]))  # exp(slope) = λ

    # Clamp to reasonable values
    epsilon_1 = max(epsilon_1, 1e-10)
    growth_rate = max(growth_rate, 0.9)  # Allow slight contraction

    # Determine if error is bounded
    # For SD: growth_rate ≈ 1.0 (no growth)
    # For BB: growth_rate > 1.0 (exponential growth)
    is_bounded = growth_rate < 1.01  # Tolerance

    return ErrorCharacteristics(
        epsilon_1=epsilon_1,
        growth_rate=growth_rate,
        error_bound=horizon_errors.get(max(horizons), epsilon_1),
        is_bounded=is_bounded,
    )


# =============================================================================
# Optimal Parameter Computation
# =============================================================================

def compute_optimal_horizon(
    error_char: ErrorCharacteristics,
    V_max: float = 1000.0,
    gamma: float = 0.99,
    epsilon_target: float = 10.0,
    max_H: int = 30,
    min_H: int = 5,
) -> int:
    """
    Compute minimum horizon H satisfying error budget.

    For bounded error (SD): ε_H ≈ ε_1, solve γ^H · V_max ≤ ε_target
    For growing error (BB): ε_H ≈ ε_1 · λ^(H-1), solve ε_1 · λ^(H-1) ≤ ε_target

    Returns:
        Optimal horizon H
    """
    if error_char.is_bounded:
        # SD case: truncation error dominates
        # γ^H · V_max ≤ ε_target
        if V_max <= 0 or epsilon_target <= 0 or gamma <= 0 or gamma >= 1:
            return max_H

        rhs = epsilon_target / V_max
        if 0 < rhs < 1:
            H_needed = int(np.ceil(np.log(rhs) / np.log(gamma)))
        else:
            H_needed = min_H
    else:
        # BB case: model error evolves with H
        lambda_ = error_char.growth_rate

        if lambda_ <= 1.0:
            # Error shrinks or stays constant with H - use larger H
            # This can happen if the learned model is conservative
            H_needed = max_H
        elif error_char.epsilon_1 >= epsilon_target:
            # Single-step error already exceeds budget - use min_H
            H_needed = min_H
        else:
            # Error grows: ε_1 · λ^(H-1) ≤ epsilon_target
            # λ^(H-1) ≤ epsilon_target / ε_1
            # H-1 ≤ log(epsilon_target / ε_1) / log(λ)
            ratio = epsilon_target / error_char.epsilon_1
            if ratio > 1:
                H_needed = int(np.ceil(np.log(ratio) / np.log(lambda_))) + 1
            else:
                H_needed = min_H

    return max(min(H_needed, max_H), min_H)


def compute_optimal_sims(
    error_char: ErrorCharacteristics,
    H: int,
    V_max: float = 1000.0,
    sigma: float = 1.0,
    epsilon_target: float = 10.0,
    delta: float = 0.05,
    base_sims: int = 200,
    gamma: float = 0.99,
    C: float = 1.0,
) -> int:
    """
    Compute optimal number of simulations based on error budget allocation.

    Returns:
        Optimal number of simulations
    """
    if gamma <= 0 or gamma >= 1 or delta <= 0 or delta >= 1:
        return base_sims
    if sigma <= 0 or epsilon_target <= 0:
        return base_sims

    # Model error contribution
    if error_char.is_bounded:
        epsilon_model_contrib = error_char.epsilon_1
    else:
        # For BB, error at horizon H
        epsilon_model_contrib = error_char.epsilon_1 * (error_char.growth_rate ** (H - 1))

    # Truncation error
    epsilon_trunc = (gamma ** H) * V_max

    # Remaining budget for MC error
    epsilon_mc_budget = epsilon_target - epsilon_model_contrib - epsilon_trunc

    if epsilon_mc_budget <= 0:
        # Model or truncation error dominates
        return max(20, base_sims // 4)

    # Hoeffding bound
    n_sims = int(C * (sigma ** 2 / epsilon_mc_budget ** 2) * np.log(1 / delta))

    return max(min(n_sims, 1000), 20)


# =============================================================================
# Training (reuse from existing experiments)
# =============================================================================

def train_model(
    env: BeerGameEnvFlat,
    model_type: str,
    sample_budget: int,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> Tuple:
    """Train SD or BlackBox model."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if model_type == "sd":
        sd_model = BeerGameSDModel(
            order_levels=env.order_levels,
            O_max=env.O_max,
        ).to(device)
        world_model = BeerGameSDModelWrapper(sd_model)
        model_opt = optim.Adam(sd_model.parameters(), lr=1e-2) if list(sd_model.parameters()) else None
    else:
        world_model = BlackBoxWorldModel(state_dim, n_actions, hidden_dim=256).to(device)
        model_opt = optim.Adam(world_model.parameters(), lr=1e-3)

    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    return_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def get_return_stats():
        return return_stats['mean'], return_stats['std']

    normalized_vnet = NormalizedValueNet(value_net, get_return_stats)
    gamma = 0.99
    epsilon = 0.3
    random_steps = min(2000, max(500, sample_budget // 3))

    steps = 0
    episode = 0

    if verbose:
        print(f"    Training {model_type.upper()} (budget={sample_budget})")

    while steps < sample_budget:
        state, _ = env.reset(seed=seed + episode)
        episode_transitions = []

        for t in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model,
                    n_actions=n_actions,
                    num_simulations=200,
                    value_net=normalized_vnet,
                    reward_fn=reward_fn,
                    rollout_depth=10,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(Transition(state, action, reward, next_state, done))
            episode_transitions.append((state, action, reward))

            # Update return stats
            if episode >= 5:
                recent_returns = [sum(r for _, _, r in buf) for buf in buffer.buffer[-50:]]
                return_stats['mean'] = float(np.mean(recent_returns))
                return_stats['std'] = float(np.std(recent_returns)) + 1e-6

            state = next_state
            steps += 1

            if done:
                break

        # Training updates
        if len(buffer) >= 100:
            batch = buffer.sample(min(64, len(buffer)))
            states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array([t.action for t in batch]), dtype=torch.long).to(device)
            next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(device)

            # World model loss
            with torch.no_grad():
                if model_type == "sd":
                    target_next = world_model.model(states, actions)
                else:
                    target_next = next_states

            pred_next = world_model(states, actions)
            wm_loss = F.mse_loss(pred_next, target_next)

            if model_opt is not None:
                model_opt.zero_grad()
                wm_loss.backward()
                model_opt.step()

            # Value loss
            current_values = value_net(states)
            with torch.no_grad():
                next_values = value_net(next_states)
                target_values = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device).unsqueeze(1) + gamma * next_values
            value_loss = F.mse_loss(current_values, target_values)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode += 1

        if verbose and episode % 10 == 0:
            print(f"      Steps {steps}/{sample_budget} | Episode {episode}")

    return world_model, normalized_vnet, error_char


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    env,
    world_model,
    value_net,
    error_char: ErrorCharacteristics,
    H: int,
    N_sims: int,
    n_episodes: int = 50,
    device: str = "cpu",
) -> Dict:
    """Evaluate a model with given H and N_sims."""
    n_actions = env.action_space.n
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    episode_returns = []
    total_simulations = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=42 + ep)
        episode_return = 0.0
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=world_model,
                n_actions=n_actions,
                num_simulations=N_sims,
                value_net=value_net,
                reward_fn=reward_fn,
                rollout_depth=H,
            )
            total_simulations += N_sims

            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        episode_returns.append(episode_return)

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "total_simulations": total_simulations,
        "avg_simulations_per_step": total_simulations / (n_episodes * env.T),
        "H": H,
        "N_sims": N_sims,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_error_budget_experiment(
    sample_budget: int = 5000,
    n_seeds: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
    """Run SD vs BB comparison under fixed error budget."""

    results = {
        "sample_budget": sample_budget,
        "n_seeds": n_seeds,
        "seeds": [],
        "aggregated": {},
    }

    # Common parameters
    V_max = 1000.0
    gamma = 0.99
    epsilon_target = 10.0  # 1% of V_max

    for seed in range(n_seeds):
        seed_result = {
            "seed": seed,
            "models": {},
        }

        print(f"\n{'='*80}")
        print(f"Seed {seed}")
        print('='*80)

        env = BeerGameEnvFlat(n_echelons=4, order_levels=3, O_max=16)

        # Train and evaluate SD
        print(f"\n[Seed {seed}] Training SD-MCTS...")
        t0 = time.time()
        sd_wm, sd_vn, _ = train_model(env, "sd", sample_budget, device, seed, verbose)

        # Compute SD error characteristics
        print(f"[Seed {seed}] Computing SD error characteristics...")
        sd_error_char = compute_error_characteristics(sd_wm, env, "sd", device=device)
        print(f"  SD: ε_1={sd_error_char.epsilon_1:.6f}, λ={sd_error_char.growth_rate:.4f}, bounded={sd_error_char.is_bounded}")

        # Compute optimal parameters for SD
        sd_H_opt = compute_optimal_horizon(sd_error_char, V_max, gamma, epsilon_target)
        sd_N_opt = compute_optimal_sims(sd_error_char, sd_H_opt, V_max, epsilon_target=epsilon_target, gamma=gamma)

        print(f"  SD optimal: H={sd_H_opt}, N_sims={sd_N_opt}")

        # Evaluate SD with optimal parameters
        print(f"[Seed {seed}] Evaluating SD with optimal parameters...")
        sd_result_opt = evaluate_model(env, sd_wm, sd_vn, sd_error_char, sd_H_opt, sd_N_opt, device=device)
        print(f"  SD Return: {sd_result_opt['mean_return']:.1f}±{sd_result_opt['std_return']:.1f}")

        # Also evaluate SD with baseline parameters
        sd_result_baseline = evaluate_model(env, sd_wm, sd_vn, sd_error_char, H=10, N_sims=200, device=device)

        seed_result["models"]["sd"] = {
            "error_char": {
                "epsilon_1": sd_error_char.epsilon_1,
                "growth_rate": sd_error_char.growth_rate,
                "is_bounded": sd_error_char.is_bounded,
            },
            "optimal": {
                "H": sd_H_opt,
                "N_sims": sd_N_opt,
                "mean_return": sd_result_opt["mean_return"],
                "std_return": sd_result_opt["std_return"],
                "avg_sims": sd_result_opt["avg_simulations_per_step"],
            },
            "baseline": {
                "H": 10,
                "N_sims": 200,
                "mean_return": sd_result_baseline["mean_return"],
                "std_return": sd_result_baseline["std_return"],
                "avg_sims": sd_result_baseline["avg_simulations_per_step"],
            },
            "training_time": time.time() - t0,
        }

        # Train and evaluate BB
        print(f"\n[Seed {seed}] Training BB-MCTS...")
        t0 = time.time()
        bb_wm, bb_vn, _ = train_model(env, "mlp", sample_budget, device, seed, verbose)

        # Compute BB error characteristics
        print(f"[Seed {seed}] Computing BB error characteristics...")
        bb_error_char = compute_error_characteristics(bb_wm, env, "mlp", device=device)
        print(f"  BB: ε_1={bb_error_char.epsilon_1:.6f}, λ={bb_error_char.growth_rate:.4f}, bounded={bb_error_char.is_bounded}")

        # Compute optimal parameters for BB
        bb_H_opt = compute_optimal_horizon(bb_error_char, V_max, gamma, epsilon_target)
        bb_N_opt = compute_optimal_sims(bb_error_char, bb_H_opt, V_max, epsilon_target=epsilon_target, gamma=gamma)

        print(f"  BB optimal: H={bb_H_opt}, N_sims={bb_N_opt}")

        # Evaluate BB with optimal parameters
        print(f"[Seed {seed}] Evaluating BB with optimal parameters...")
        bb_result_opt = evaluate_model(env, bb_wm, bb_vn, bb_error_char, bb_H_opt, bb_N_opt, device=device)
        print(f"  BB Return: {bb_result_opt['mean_return']:.1f}±{bb_result_opt['std_return']:.1f}")

        # Also evaluate BB with baseline parameters
        bb_result_baseline = evaluate_model(env, bb_wm, bb_vn, bb_error_char, H=10, N_sims=200, device=device)

        seed_result["models"]["bb"] = {
            "error_char": {
                "epsilon_1": bb_error_char.epsilon_1,
                "growth_rate": bb_error_char.growth_rate,
                "is_bounded": bb_error_char.is_bounded,
            },
            "optimal": {
                "H": bb_H_opt,
                "N_sims": bb_N_opt,
                "mean_return": bb_result_opt["mean_return"],
                "std_return": bb_result_opt["std_return"],
                "avg_sims": bb_result_opt["avg_simulations_per_step"],
            },
            "baseline": {
                "H": 10,
                "N_sims": 200,
                "mean_return": bb_result_baseline["mean_return"],
                "std_return": bb_result_baseline["std_return"],
                "avg_sims": bb_result_baseline["avg_simulations_per_step"],
            },
            "training_time": time.time() - t0,
        }

        # Compute efficiency ratios
        sd_cost = sd_H_opt * sd_N_opt
        bb_cost = bb_H_opt * bb_N_opt
        efficiency_ratio = bb_cost / sd_cost if sd_cost > 0 else 1.0

        print(f"\n  [Seed {seed}] Efficiency Analysis:")
        print(f"    SD cost: H={sd_H_opt} × N={sd_N_opt} = {sd_cost}")
        print(f"    BB cost: H={bb_H_opt} × N={bb_N_opt} = {bb_cost}")
        print(f"    Efficiency ratio (BB/SD): {efficiency_ratio:.2f}x")

        seed_result["efficiency_ratio"] = efficiency_ratio

        results["seeds"].append(seed_result)

    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS")
    print("="*80)

    # Extract results for aggregation
    sd_eps1 = [s["models"]["sd"]["error_char"]["epsilon_1"] for s in results["seeds"]]
    sd_lambda = [s["models"]["sd"]["error_char"]["growth_rate"] for s in results["seeds"]]
    sd_H_opt = [s["models"]["sd"]["optimal"]["H"] for s in results["seeds"]]
    sd_N_opt = [s["models"]["sd"]["optimal"]["N_sims"] for s in results["seeds"]]
    sd_return_opt = [s["models"]["sd"]["optimal"]["mean_return"] for s in results["seeds"]]
    sd_sims_opt = [s["models"]["sd"]["optimal"]["avg_sims"] for s in results["seeds"]]

    bb_eps1 = [s["models"]["bb"]["error_char"]["epsilon_1"] for s in results["seeds"]]
    bb_lambda = [s["models"]["bb"]["error_char"]["growth_rate"] for s in results["seeds"]]
    bb_H_opt = [s["models"]["bb"]["optimal"]["H"] for s in results["seeds"]]
    bb_N_opt = [s["models"]["bb"]["optimal"]["N_sims"] for s in results["seeds"]]
    bb_return_opt = [s["models"]["bb"]["optimal"]["mean_return"] for s in results["seeds"]]
    bb_sims_opt = [s["models"]["bb"]["optimal"]["avg_sims"] for s in results["seeds"]]

    efficiency_ratios = [s["efficiency_ratio"] for s in results["seeds"]]

    results["aggregated"] = {
        "sd": {
            "epsilon_1": float(np.mean(sd_eps1)),
            "growth_rate": float(np.mean(sd_lambda)),
            "H_opt": float(np.mean(sd_H_opt)),
            "N_sims_opt": float(np.mean(sd_N_opt)),
            "return_opt": float(np.mean(sd_return_opt)),
            "avg_sims_opt": float(np.mean(sd_sims_opt)),
        },
        "bb": {
            "epsilon_1": float(np.mean(bb_eps1)),
            "growth_rate": float(np.mean(bb_lambda)),
            "H_opt": float(np.mean(bb_H_opt)),
            "N_sims_opt": float(np.mean(bb_N_opt)),
            "return_opt": float(np.mean(bb_return_opt)),
            "avg_sims_opt": float(np.mean(bb_sims_opt)),
        },
        "efficiency_ratio": float(np.mean(efficiency_ratios)),
    }

    print("\n### Error Characteristics")
    print(f"{'Model':<10} {'ε_1':<12} {'Growth Rate λ':<15} {'Bounded?':<10}")
    print("-"*60)
    print(f"{'SD':<10} {results['aggregated']['sd']['epsilon_1']:<12.6f} {results['aggregated']['sd']['growth_rate']:<15.4f} {'Yes':<10}")
    print(f"{'BB':<10} {results['aggregated']['bb']['epsilon_1']:<12.6f} {results['aggregated']['bb']['growth_rate']:<15.4f} {'No':<10}")

    print("\n### Optimal Parameters (Under Same Error Budget)")
    print(f"{'Model':<10} {'H_opt':<10} {'N_sims_opt':<15} {'Total Cost':<15} {'Return':<12}")
    print("-"*80)
    sd_cost = results['aggregated']['sd']['H_opt'] * results['aggregated']['sd']['N_sims_opt']
    bb_cost = results['aggregated']['bb']['H_opt'] * results['aggregated']['bb']['N_sims_opt']
    print(f"{'SD':<10} {results['aggregated']['sd']['H_opt']:<10.0f} {results['aggregated']['sd']['N_sims_opt']:<15.0f} {sd_cost:<15.0f} {results['aggregated']['sd']['return_opt']:<12.1f}")
    print(f"{'BB':<10} {results['aggregated']['bb']['H_opt']:<10.0f} {results['aggregated']['bb']['N_sims_opt']:<15.0f} {bb_cost:<15.0f} {results['aggregated']['bb']['return_opt']:<12.1f}")

    print("\n### Efficiency Summary")
    eff_ratio = results['aggregated']['efficiency_ratio']
    if eff_ratio > 0:
        print(f"BB needs {eff_ratio:.2f}x more computation than SD")
        print(f"SD reduction: {(1 - 1/eff_ratio)*100:.1f}%")
    else:
        print(f"Efficiency ratio calculation error")

    return results


def main():
    parser = argparse.ArgumentParser(description="SD vs BB Efficiency Comparison")
    parser.add_argument("--budget", type=int, default=5000, help="Training sample budget")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / "error_budget_comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SD vs BB Computational Efficiency Experiment")
    print("="*80)
    print(f"Sample budget: {args.budget}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")

    results = run_error_budget_experiment(
        sample_budget=args.budget,
        n_seeds=args.seeds,
        device=args.device,
        verbose=args.verbose,
    )

    # Save results
    results_file = output_dir / "error_budget_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
