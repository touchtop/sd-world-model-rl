#!/usr/bin/env python3
"""
NeurIPS Experiment: OOD Initial State Generalization

Tests structural invariance: SD's hardcoded dynamics (conservation laws,
delays, shipping logic) are exact for ANY state, while BlackBox MLP
overfits to training distribution.

Design:
- Train both models on near-equilibrium initial states
- Evaluate on 4 OOD state distributions (backlog shock, inventory surplus, etc.)
- Measure: (1) world model prediction error, (2) MCTS policy performance
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


# =============================================================================
# OOD State Distributions
# =============================================================================

def sample_id_state(rng: np.random.Generator, n_echelons: int = 4) -> np.ndarray:
    """Sample in-distribution (near-equilibrium) initial state."""
    state = np.zeros(n_echelons * 6 + 2, dtype=np.float32)
    for i in range(n_echelons):
        base = i * 6
        state[base + 0] = rng.uniform(8, 16)    # Inventory
        state[base + 1] = 0.0                    # Backlog (zero at equilibrium)
        state[base + 2] = rng.uniform(3, 5)     # ship_0
        state[base + 3] = rng.uniform(3, 5)     # ship_1
        state[base + 4] = rng.uniform(3, 5)     # order_0
        state[base + 5] = rng.uniform(3, 5)     # order_1
    state[-2] = 0.0  # t=0
    state[-1] = 4.0  # initial demand
    return state


def sample_backlog_shock(rng: np.random.Generator, n_echelons: int = 4) -> np.ndarray:
    """OOD: Random echelons have severe backlog, depleted inventory."""
    state = sample_id_state(rng, n_echelons)
    shocked = rng.choice(n_echelons, size=2, replace=False)
    for i in shocked:
        base = i * 6
        state[base + 0] = rng.uniform(0, 2)      # Depleted inventory
        state[base + 1] = rng.uniform(24, 48)    # Severe backlog
    return state


def sample_inventory_surplus(rng: np.random.Generator, n_echelons: int = 4) -> np.ndarray:
    """OOD: Excessive inventory, near-zero pipeline."""
    state = np.zeros(n_echelons * 6 + 2, dtype=np.float32)
    for i in range(n_echelons):
        base = i * 6
        state[base + 0] = rng.uniform(32, 60)   # Excessive inventory
        state[base + 1] = 0.0                    # No backlog
        state[base + 2] = rng.uniform(0, 2)     # Near-zero pipeline
        state[base + 3] = rng.uniform(0, 2)
        state[base + 4] = rng.uniform(0, 2)
        state[base + 5] = rng.uniform(0, 2)
    state[-2] = 0.0
    state[-1] = 4.0
    return state


def sample_pipeline_imbalance(rng: np.random.Generator, n_echelons: int = 4) -> np.ndarray:
    """OOD: Imbalanced pipeline (some near-zero, some very high)."""
    state = sample_id_state(rng, n_echelons)
    for i in range(n_echelons):
        base = i * 6
        if rng.random() < 0.5:
            state[base + 2] = rng.uniform(0, 2)
            state[base + 3] = rng.uniform(30, 50)
            state[base + 4] = rng.uniform(0, 2)
            state[base + 5] = rng.uniform(30, 50)
        else:
            state[base + 2] = rng.uniform(30, 50)
            state[base + 3] = rng.uniform(0, 2)
            state[base + 4] = rng.uniform(30, 50)
            state[base + 5] = rng.uniform(0, 2)
    return state


def sample_alternating_extreme(rng: np.random.Generator, n_echelons: int = 4) -> np.ndarray:
    """OOD: Adjacent echelons alternate between backlog shock and surplus."""
    state = np.zeros(n_echelons * 6 + 2, dtype=np.float32)
    for i in range(n_echelons):
        base = i * 6
        if i % 2 == 0:  # Backlog shock
            state[base + 0] = rng.uniform(0, 2)
            state[base + 1] = rng.uniform(30, 50)
        else:  # Inventory surplus
            state[base + 0] = rng.uniform(40, 70)
            state[base + 1] = 0.0
        # Extreme pipeline imbalance
        state[base + 2] = rng.uniform(0, 3) if i % 2 == 0 else rng.uniform(25, 45)
        state[base + 3] = rng.uniform(25, 45) if i % 2 == 0 else rng.uniform(0, 3)
        state[base + 4] = rng.uniform(0, 3) if i % 2 == 0 else rng.uniform(25, 45)
        state[base + 5] = rng.uniform(25, 45) if i % 2 == 0 else rng.uniform(0, 3)
    state[-2] = 0.0
    state[-1] = 4.0
    return state


OOD_DISTRIBUTIONS = {
    "id": ("In-Distribution (near equilibrium)", sample_id_state),
    "backlog_shock": ("Backlog Shock (2 echelons)", sample_backlog_shock),
    "inventory_surplus": ("Inventory Surplus (all echelons)", sample_inventory_surplus),
    "pipeline_imbalance": ("Pipeline Imbalance", sample_pipeline_imbalance),
    "alternating_extreme": ("Alternating Extreme", sample_alternating_extreme),
}


# =============================================================================
# Environment with Custom Initial State
# =============================================================================

class OODStateEnv(BeerGameEnvFlat):
    """Beer Game environment with injectable initial state."""

    def __init__(self, initial_state_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._initial_state_fn = initial_state_fn
        self._custom_state = None

    def set_initial_state(self, state: np.ndarray):
        """Set custom initial state for next reset."""
        self._custom_state = state.copy()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self._custom_state is not None:
            self._restore_state(self._custom_state)
            self._custom_state = None
        elif self._initial_state_fn is not None:
            custom = self._initial_state_fn(self.rng, self.n_echelons)
            self._restore_state(custom)
        else:
            self._init_state()

        return self._get_obs(), {}

    def _restore_state(self, state: np.ndarray):
        """Restore internal state from state vector."""
        for i in range(self.n_echelons):
            base = i * 6
            self.inventory[i] = state[base + 0]
            self.backlog[i] = state[base + 1]
            self.incoming_pipeline[i, 0] = state[base + 2]
            self.incoming_pipeline[i, 1] = state[base + 3]
            self.order_pipeline[i, 0] = state[base + 4]
            self.order_pipeline[i, 1] = state[base + 5]
        self.t = int(state[-2])
        self.current_demand = state[-1]
        self.total_cost = 0.0


# =============================================================================
# Training (reuse patterns from ablation)
# =============================================================================

def train_model(
    env: BeerGameEnvFlat,
    model_type: str,  # "sd" or "mlp"
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
        is_mlp = False
    else:
        world_model = BlackBoxWorldModel(state_dim, n_actions, hidden_dim=256).to(device)
        model_opt = optim.Adam(world_model.parameters(), lr=1e-3)
        is_mlp = True

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
    last_log_steps = 0

    if verbose:
        print(f"    Training {model_type.upper()} (budget={sample_budget}, random_steps={random_steps})")

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

        # Update return statistics
        G = 0.0
        returns = []
        for tr in reversed(episode_transitions):
            G = tr.reward + gamma * G
            returns.insert(0, G)

        for r in returns:
            return_stats['count'] += 1
            delta = r - return_stats['mean']
            return_stats['mean'] += delta / return_stats['count']
            if return_stats['count'] > 1:
                return_stats['std'] = max(1.0, np.sqrt(
                    ((return_stats['count'] - 1) * return_stats['std']**2 + delta * (r - return_stats['mean']))
                    / return_stats['count']
                ))

        # Train world model
        if len(buffer) >= 32:
            for _ in range(5):
                batch = buffer.sample(32)
                states_t = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
                actions_t = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                next_states_t = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)

                if is_mlp:
                    pred_next = world_model(states_t, actions_t)
                else:
                    pred_next = world_model.model(states_t, actions_t)

                loss = F.mse_loss(pred_next, next_states_t)
                if model_opt:
                    model_opt.zero_grad()
                    loss.backward()
                    model_opt.step()

        # Train value network
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

        episode += 1

        # Progress logging
        if verbose and steps - last_log_steps >= 1000:
            print(f"      Steps {steps}/{sample_budget} | Episode {episode} | Recent return: {sum(returns)/len(returns):.1f}")
            last_log_steps = steps

    return world_model, value_net, normalized_vnet, return_stats


# =============================================================================
# Evaluation: World Model Prediction Error
# =============================================================================

def evaluate_prediction_error(
    world_model,
    env: OODStateEnv,
    dist_name: str,
    sample_fn,
    n_samples: int = 200,
    rollout_horizons: List[int] = [1, 5, 10, 20],
    device: str = "cpu",
) -> Dict:
    """Evaluate world model prediction error on OOD states."""
    rng = np.random.default_rng(42)
    results = {h: [] for h in rollout_horizons}

    is_mlp = not hasattr(world_model, 'model')

    for _ in range(n_samples):
        init_state = sample_fn(rng, env.n_echelons)
        env.set_initial_state(init_state)
        true_state, _ = env.reset()

        for horizon in rollout_horizons:
            pred_state = true_state.copy()
            actual_state = true_state.copy()
            env.set_initial_state(init_state)
            env.reset()

            cumulative_error = 0.0
            for step in range(horizon):
                action = env.action_space.sample()

                # Predict next state
                with torch.no_grad():
                    state_t = torch.tensor(pred_state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_t = torch.tensor([action], dtype=torch.int64, device=device)
                    if is_mlp:
                        pred_next = world_model(state_t, action_t).squeeze(0).cpu().numpy()
                    else:
                        pred_next = world_model.model(state_t, action_t).squeeze(0).cpu().numpy()

                # Actual next state
                actual_next, _, terminated, truncated, _ = env.step(action)

                # Compute error on state components (exclude t and demand)
                # Note: We exclude demand because (1) it's deterministic in this env,
                # (2) we're testing STRUCTURAL dynamics invariance, not demand prediction
                error = np.mean((pred_next[:-2] - actual_next[:-2]) ** 2)
                cumulative_error += error

                pred_state = pred_next
                actual_state = actual_next

                if terminated or truncated:
                    break

            results[horizon].append(cumulative_error / horizon)

    return {
        "dist": dist_name,
        "horizons": {h: {"mean": np.mean(results[h]), "std": np.std(results[h])} for h in rollout_horizons},
    }


# =============================================================================
# Evaluation: MCTS Policy Performance
# =============================================================================

def evaluate_policy_performance(
    world_model,
    value_net,
    env: OODStateEnv,
    dist_name: str,
    sample_fn,
    n_episodes: int = 50,
    device: str = "cpu",
    seed_offset: int = 0,
) -> Dict:
    """Evaluate MCTS policy performance on OOD initial states."""
    n_actions = env.action_space.n
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    episode_returns = []

    for ep in range(n_episodes):
        rng = np.random.default_rng(seed_offset + ep)
        init_state = sample_fn(rng, env.n_echelons)
        env.set_initial_state(init_state)
        state, _ = env.reset(seed=seed_offset + ep)

        episode_return = 0.0
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=world_model,
                n_actions=n_actions,
                num_simulations=200,
                value_net=value_net,
                reward_fn=reward_fn,
                rollout_depth=10,
            )

            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        episode_returns.append(episode_return)

    return {
        "dist": dist_name,
        "mean": np.mean(episode_returns),
        "std": np.std(episode_returns),
        "min": np.min(episode_returns),
        "max": np.max(episode_returns),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    sample_budget: int = 15000,
    n_seeds: int = 3,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Run full OOD generalization experiment."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = NEURIPS_DIR / "results" / "ood_state_generalization" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OOD Initial State Generalization Experiment")
    print("=" * 80)
    print(f"Sample budget: {sample_budget}")
    print(f"Seeds: {n_seeds}")
    print(f"OOD distributions: {list(OOD_DISTRIBUTIONS.keys())}")
    print(f"Output: {output_dir}")
    print()

    all_results = []

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Create training environment (ID distribution)
        train_env = OODStateEnv(
            initial_state_fn=sample_id_state,
            order_levels=3,
            T=36,
        )

        # Train SD model
        print(f"\n[Seed {seed}] Training SD-MCTS...")
        t0 = time.time()
        sd_wm, sd_vn, sd_nvn, sd_stats = train_model(
            train_env, "sd", sample_budget, device, seed, verbose=verbose
        )
        sd_time = time.time() - t0
        print(f"  SD training: {sd_time:.1f}s")

        # Train BlackBox model
        print(f"[Seed {seed}] Training BlackBox-MCTS...")
        t0 = time.time()
        bb_wm, bb_vn, bb_nvn, bb_stats = train_model(
            train_env, "mlp", sample_budget, device, seed, verbose=verbose
        )
        bb_time = time.time() - t0
        print(f"  BlackBox training: {bb_time:.1f}s")

        # Evaluate on each OOD distribution
        seed_results = {"seed": seed, "prediction_error": {}, "policy_performance": {}}

        for dist_key, (dist_desc, sample_fn) in OOD_DISTRIBUTIONS.items():
            print(f"\n  Evaluating on: {dist_desc}")

            # Create independent environments for each model to avoid randomness contamination
            sd_eval_env = OODStateEnv(order_levels=3, T=36)
            bb_eval_env = OODStateEnv(order_levels=3, T=36)

            # Prediction error
            sd_pred = evaluate_prediction_error(sd_wm, sd_eval_env, dist_key, sample_fn, device=device)
            bb_pred = evaluate_prediction_error(bb_wm, bb_eval_env, dist_key, sample_fn, device=device)

            seed_results["prediction_error"][dist_key] = {
                "sd": sd_pred,
                "blackbox": bb_pred,
            }

            # Policy performance (use independent environments with fixed seeds)
            sd_perf = evaluate_policy_performance(sd_wm, sd_nvn, sd_eval_env, dist_key, sample_fn, device=device, seed_offset=seed*1000)
            bb_perf = evaluate_policy_performance(bb_wm, bb_nvn, bb_eval_env, dist_key, sample_fn, device=device, seed_offset=seed*1000)

            seed_results["policy_performance"][dist_key] = {
                "sd": sd_perf,
                "blackbox": bb_perf,
            }

            # Print summary
            print(f"    Prediction Error (H=10): SD={sd_pred['horizons'][10]['mean']:.4f}, "
                  f"BB={bb_pred['horizons'][10]['mean']:.4f}")
            print(f"    Policy Return: SD={sd_perf['mean']:.1f}±{sd_perf['std']:.1f}, "
                  f"BB={bb_perf['mean']:.1f}±{bb_perf['std']:.1f}")

        all_results.append(seed_results)

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    # Build summary tables
    print("\n### Prediction Error (Horizon=10, lower is better)")
    print(f"{'Distribution':<25} {'SD Mean':<12} {'BB Mean':<12} {'Ratio (BB/SD)':<12}")
    print("-" * 65)

    for dist_key in OOD_DISTRIBUTIONS.keys():
        sd_errors = [r["prediction_error"][dist_key]["sd"]["horizons"][10]["mean"] for r in all_results]
        bb_errors = [r["prediction_error"][dist_key]["blackbox"]["horizons"][10]["mean"] for r in all_results]
        sd_mean = np.mean(sd_errors)
        bb_mean = np.mean(bb_errors)
        ratio = bb_mean / sd_mean if sd_mean > 1e-8 else float('inf')
        print(f"{dist_key:<25} {sd_mean:<12.4f} {bb_mean:<12.4f} {ratio:<12.2f}x")

    print("\n### Policy Performance (higher is better, cost = negative return)")
    print(f"{'Distribution':<25} {'SD Return':<15} {'BB Return':<15} {'SD Advantage':<12}")
    print("-" * 70)

    for dist_key in OOD_DISTRIBUTIONS.keys():
        sd_returns = [r["policy_performance"][dist_key]["sd"]["mean"] for r in all_results]
        bb_returns = [r["policy_performance"][dist_key]["blackbox"]["mean"] for r in all_results]
        sd_mean = np.mean(sd_returns)
        bb_mean = np.mean(bb_returns)
        # Positive = SD better, Negative = BlackBox better
        advantage = ((sd_mean - bb_mean) / abs(bb_mean)) * 100 if bb_mean != 0 else 0
        print(f"{dist_key:<25} {sd_mean:<15.1f} {bb_mean:<15.1f} {advantage:>+10.1f}%")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOD State Generalization Experiment")
    parser.add_argument("--budget", type=int, default=15000, help="Training sample budget")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose training output")
    args = parser.parse_args()

    run_experiment(
        sample_budget=args.budget,
        n_seeds=args.seeds,
        device=args.device,
        verbose=args.verbose,
    )
