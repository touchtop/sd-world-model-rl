#!/usr/bin/env python
"""
Full Comparison Experiment: SD-MCTS vs Oracle MPC vs Adaptive MPC vs MPC (wrong).

This script runs a comprehensive comparison of all baselines for the
efficiency degradation scenario. Results are saved with statistical analysis.

Usage:
    cd /home/data/round/code/sd_world_model_rl
    PYTHONPATH=.:$PYTHONPATH python experiments/run_full_comparison.py
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.m1_framework import ScenarioConfig, ModelBelief
from experiments.baselines.oracle_mpc import OracleMPC, run_oracle_mpc_episode
from experiments.baselines.adaptive_mpc import AdaptiveMPC, run_adaptive_mpc_episode
from src.world_model import TaskSDModel
from src.agent import PolicyNetwork, ValueNetwork
from src.planning import mcts_search
from train_muzero import collect_episode, ReplayBuffer


def run_mpc_wrong_episode(
    env, belief: ModelBelief, seed: int, episode_length: int,
    belief_alpha: float, belief_theta: float, n_workers: int,
    horizon: int = 3, gamma: float = 0.99, record_trajectory: bool = False
) -> Tuple[float, List[float], Dict]:
    """Run MPC with fixed wrong belief (original baseline).

    Uses H-step rollout with wrong belief parameters, matching
    the horizon structure of other MPC variants.
    """
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    trajectory = {"queue": [], "backlog": [], "fatigue": []} if record_trajectory else None
    worker_rates = np.array(belief.worker_rates, dtype=np.float32)

    for step in range(episode_length):
        if record_trajectory:
            trajectory["queue"].append(float(state[0]))
            trajectory["backlog"].append(float(np.sum(state[1:1+n_workers])))
            trajectory["fatigue"].append(float(state[-1]) if len(state) > 1 + n_workers else 0.0)

        best_action = n_workers
        best_value = float('-inf')

        for action in range(n_workers + 1):
            # H-step rollout with wrong belief
            value = _rollout_wrong(
                state, action, belief, n_workers, belief_alpha, belief_theta,
                worker_rates, horizon, gamma
            )
            if value > best_value:
                best_value = value
                best_action = action

        next_state, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rewards, trajectory or {}


def _rollout_wrong(
    state: np.ndarray, first_action: int, belief: ModelBelief,
    n_workers: int, belief_alpha: float, belief_theta: float,
    worker_rates: np.ndarray, horizon: int, gamma: float
) -> float:
    """H-step rollout with WRONG belief parameters."""
    total_reward = 0.0
    sim_state = state.copy()

    for t in range(horizon):
        action = first_action if t == 0 else _greedy_action_wrong(
            sim_state, belief, n_workers, belief_alpha, belief_theta, worker_rates
        )
        sim_state, reward = _simulate_step_wrong(
            sim_state, action, belief, n_workers, belief_alpha, belief_theta, worker_rates
        )
        total_reward += reward * (gamma ** t)

    return total_reward


def _greedy_action_wrong(
    state: np.ndarray, belief: ModelBelief, n_workers: int,
    belief_alpha: float, belief_theta: float, worker_rates: np.ndarray
) -> int:
    """Greedy action selection with wrong belief."""
    best_action = n_workers
    best_value = float('-inf')

    for action in range(n_workers + 1):
        _, reward = _simulate_step_wrong(
            state.copy(), action, belief, n_workers, belief_alpha, belief_theta, worker_rates
        )
        if reward > best_value:
            best_value = reward
            best_action = action

    return best_action


def _simulate_step_wrong(
    state: np.ndarray, action: int, belief: ModelBelief,
    n_workers: int, belief_alpha: float, belief_theta: float,
    worker_rates: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Simulate one step with WRONG belief parameters."""
    queue = state[0]
    backlogs = state[1:1+n_workers].copy()
    fatigue = state[-1] if len(state) > 1 + n_workers else 0.0

    queue += belief.arrival_rate
    if action < n_workers and queue > 0:
        queue -= 1
        backlogs[action] += 1

    # Use WRONG α and θ
    diff = max(0.0, fatigue - belief_theta)
    efficiency = 1.0 / (1.0 + belief_alpha * diff * diff)
    effective_rates = worker_rates * efficiency

    backlogs = np.maximum(0, backlogs - effective_rates)

    # Fatigue update (using wrong belief about threshold)
    max_backlog = np.max(backlogs) if backlogs.size > 0 else 0.0
    if max_backlog > 3.0:  # Wrong belief about accumulation threshold
        fatigue += 0.2
    else:
        fatigue = max(0.0, fatigue - 0.05)

    cost = queue + 0.1 * np.sum(backlogs ** 2)
    reward = -cost

    next_state = np.concatenate([[queue], backlogs, [fatigue]])
    return next_state.astype(np.float32), reward


def train_sd_mcts(
    scenario: ScenarioConfig, env_steps: int = 20000, verbose: bool = True
) -> Tuple[TaskSDModel, PolicyNetwork, ValueNetwork]:
    """Train SD-MCTS world model (from existing code)."""
    env = scenario.build_env(seed=42)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_workers = len(scenario.worker_rates)

    init_alpha = scenario.belief_efficiency_alpha or scenario.efficiency_alpha
    init_theta = scenario.belief_efficiency_threshold or scenario.efficiency_threshold

    world_model = TaskSDModel(
        n_workers=n_workers,
        fatigue_threshold=scenario.fatigue_threshold,
        fatigue_penalty=scenario.fatigue_penalty,
        learn_efficiency_degradation=scenario.efficiency_degradation,
        init_efficiency_alpha=init_alpha,
        init_efficiency_threshold=init_theta,
    )
    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    model_opt = optim.Adam(world_model.parameters(), lr=1e-2)
    buffer = ReplayBuffer(capacity=10000)

    if verbose:
        print("Training SD-MCTS world model...")

    steps = 0
    while steps < env_steps:
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=20, n_workers=n_workers
        )
        steps += len(transitions)

        for t in transitions:
            buffer.add(t)

        for _ in range(5):
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                mcts_policies = torch.tensor(np.array([t.mcts_policy for t in batch]), dtype=torch.float32)
                returns = torch.tensor([t.cumulative_return for t in batch], dtype=torch.float32)
                fatigue_states = states[:, -1] if states.shape[-1] > n_workers + 1 else None

                policy_logits = policy_net(states)
                policy_loss = -(mcts_policies * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
                policy_opt.zero_grad()
                policy_loss.backward()
                policy_opt.step()

                value_pred = value_net(states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, returns)
                value_opt.zero_grad()
                value_loss.backward()
                value_opt.step()

                pred_next = world_model(states, actions)
                mse_loss = F.mse_loss(pred_next, next_states)

                eff_loss = torch.tensor(0.0)
                if world_model.learn_efficiency_degradation and fatigue_states is not None:
                    pred_eff = world_model.get_efficiency_coef(fatigue_states)
                    true_alpha = scenario.efficiency_alpha
                    true_theta = scenario.efficiency_threshold
                    diff = F.relu(fatigue_states - true_theta)
                    true_eff = 1.0 / (1.0 + true_alpha * diff * diff)
                    error = pred_eff - true_eff
                    weights = torch.where(error < 0, 2.0, 0.5)
                    eff_loss = (weights * error * error).mean()

                loss = mse_loss + 0.5 * eff_loss
                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

        if verbose and steps % 5000 == 0:
            if hasattr(world_model, '_efficiency_alpha_raw'):
                alpha = F.softplus(world_model._efficiency_alpha_raw).item()
                theta = world_model._efficiency_threshold.item()
                print(f"  Step {steps}: α={alpha:.4f}, θ={theta:.2f}")

    return world_model, policy_net, value_net


def run_sd_mcts_episode(
    env, world_model, policy_net, value_net, n_actions: int,
    seed: int, episode_length: int, n_workers: int, record_trajectory: bool = False
) -> Tuple[float, List[float], Dict]:
    """Run SD-MCTS episode."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    trajectory = {"queue": [], "backlog": [], "fatigue": []} if record_trajectory else None

    for step in range(episode_length):
        if record_trajectory:
            trajectory["queue"].append(float(state[0]))
            trajectory["backlog"].append(float(np.sum(state[1:1+n_workers])))
            trajectory["fatigue"].append(float(state[-1]) if len(state) > 1 + n_workers else 0.0)

        action = mcts_search(
            state, world_model, n_actions=n_actions, num_simulations=50,
            policy_net=policy_net, value_net=value_net, n_workers=n_workers
        )
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rewards, trajectory or {}


def compute_statistics(rewards: List[float]) -> Dict[str, float]:
    """Compute mean, std, CI for rewards."""
    arr = np.array(rewards)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    ci_95 = 1.96 * se
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_95_lower": mean - ci_95,
        "ci_95_upper": mean + ci_95,
        "n": n,
    }


def run_comparison(
    n_seeds: int = 30,
    train_steps: int = 20000,
    episode_length: int = 200,
    adaptive_horizons: List[int] = [3, 15],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run full comparison experiment."""
    # Load scenario
    scenarios = [s for s in ScenarioConfig.generate_m1_matrix()
                 if s.name == "efficiency_degradation_fairness"]
    if not scenarios:
        raise ValueError("Scenario not found!")

    scenario = scenarios[0]
    n_workers = len(scenario.worker_rates)
    n_actions = n_workers + 1

    if verbose:
        print("=" * 80)
        print("Full Comparison: Efficiency Degradation Scenario")
        print("=" * 80)
        print(f"True α={scenario.efficiency_alpha}, θ={scenario.efficiency_threshold}")
        print(f"Belief α={scenario.belief_efficiency_alpha}, θ={scenario.belief_efficiency_threshold}")
        print(f"Seeds: {n_seeds}, Train steps: {train_steps}")
        print("=" * 80)

    # Phase 1: Train SD-MCTS
    if verbose:
        print("\nPhase 1: Training SD-MCTS")
        print("-" * 40)
    world_model, policy_net, value_net = train_sd_mcts(scenario, train_steps, verbose)

    learned_alpha = F.softplus(world_model._efficiency_alpha_raw).item()
    learned_theta = world_model._efficiency_threshold.item()
    if verbose:
        print(f"\nLearned: α={learned_alpha:.4f}, θ={learned_theta:.2f}")

    # Phase 2: Evaluate all methods
    if verbose:
        print("\nPhase 2: Evaluation")
        print("-" * 40)

    results = {
        "mpc_wrong": [],
        "oracle_mpc": [],
        "sd_mcts": [],
    }
    for h in adaptive_horizons:
        results[f"adaptive_mpc_h{h}"] = []

    # Trajectory data for visualization (record first seed only)
    trajectories = {}

    # Setup beliefs and baselines
    belief = ModelBelief(
        arrival_rate=scenario.arrival_rate,
        worker_rates=scenario.worker_rates,
        fatigue_threshold=scenario.fatigue_threshold,
        fatigue_penalty=scenario.fatigue_penalty,
        max_queue=scenario.max_queue,
        max_backlog=scenario.max_backlog,
    )
    belief_alpha = scenario.belief_efficiency_alpha or 0.001
    belief_theta = scenario.belief_efficiency_threshold or 12.0

    oracle = OracleMPC(
        n_workers=n_workers,
        arrival_rate=scenario.arrival_rate,
        worker_rates=scenario.worker_rates,
        horizon=15,
        max_queue=scenario.max_queue,
        max_backlog=scenario.max_backlog,
        efficiency_alpha=scenario.efficiency_alpha,
        efficiency_threshold=scenario.efficiency_threshold,
        fatigue_accum_threshold=scenario.fatigue_accumulation_threshold,
        fatigue_accum_rate=scenario.fatigue_accumulation_rate,
        fatigue_recovery_rate=scenario.fatigue_recovery_rate,
    )

    for seed in range(n_seeds):
        if verbose and seed % 10 == 0:
            print(f"  Seed {seed}/{n_seeds}...")

        record_traj = (seed == 0)  # Record trajectory for first seed

        # MPC (wrong belief)
        env = scenario.build_env(seed=seed)
        r, _, traj = run_mpc_wrong_episode(
            env, belief, seed, episode_length, belief_alpha, belief_theta, n_workers,
            record_trajectory=record_traj
        )
        results["mpc_wrong"].append(r)
        if record_traj:
            trajectories["mpc_wrong"] = traj

        # Oracle MPC
        env = scenario.build_env(seed=seed)
        r, _, oracle_traj = run_oracle_mpc_episode(env, oracle, seed, episode_length,
                                                    record_trajectory=record_traj)
        results["oracle_mpc"].append(r)
        if record_traj:
            trajectories["oracle_mpc"] = oracle_traj

        # SD-MCTS
        env = scenario.build_env(seed=seed)
        r, _, traj = run_sd_mcts_episode(
            env, world_model, policy_net, value_net, n_actions, seed, episode_length, n_workers,
            record_trajectory=record_traj
        )
        results["sd_mcts"].append(r)
        if record_traj:
            trajectories["sd_mcts"] = traj

        # Adaptive MPC variants
        for h in adaptive_horizons:
            adaptive = AdaptiveMPC(
                n_workers=n_workers,
                arrival_rate=scenario.arrival_rate,
                worker_rates=scenario.worker_rates,
                horizon=h,
                scenario_type="efficiency_degradation",
                init_alpha=belief_alpha,
                init_theta=belief_theta,
            )
            env = scenario.build_env(seed=seed)
            r, _, _, params = run_adaptive_mpc_episode(env, adaptive, seed, episode_length)
            results[f"adaptive_mpc_h{h}"].append(r)

    # Compute statistics
    stats = {}
    for method, rewards in results.items():
        stats[method] = compute_statistics(rewards)

    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Results Summary")
        print("=" * 80)
        for method in ["mpc_wrong"] + [f"adaptive_mpc_h{h}" for h in adaptive_horizons] + ["sd_mcts", "oracle_mpc"]:
            s = stats[method]
            print(f"{method:20s}: {s['mean']:8.2f} ± {s['std']:6.2f}  [95% CI: {s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}]")

        print("\n" + "-" * 40)
        print("Pairwise Comparisons (vs MPC wrong):")
        mpc_mean = stats["mpc_wrong"]["mean"]
        for method in [f"adaptive_mpc_h{h}" for h in adaptive_horizons] + ["sd_mcts", "oracle_mpc"]:
            improvement = stats[method]["mean"] - mpc_mean
            pct = 100 * improvement / abs(mpc_mean) if mpc_mean != 0 else 0
            print(f"  {method}: Δ={improvement:+.2f} ({pct:+.1f}%)")

    return {
        "scenario": scenario.name,
        "true_alpha": scenario.efficiency_alpha,
        "true_theta": scenario.efficiency_threshold,
        "belief_alpha": belief_alpha,
        "belief_theta": belief_theta,
        "learned_alpha": learned_alpha,
        "learned_theta": learned_theta,
        "n_seeds": n_seeds,
        "train_steps": train_steps,
        "episode_length": episode_length,
        "results": {k: list(v) for k, v in results.items()},
        "statistics": stats,
        "trajectories": trajectories,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Full comparison experiment")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--train-steps", type=int, default=20000, help="SD-MCTS training steps")
    parser.add_argument("--episode-length", type=int, default=200, help="Episode length")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    results = run_comparison(
        n_seeds=args.seeds,
        train_steps=args.train_steps,
        episode_length=args.episode_length,
        verbose=True,
    )

    # Save results
    output_file = args.output or f"results/full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save trajectory separately for visualization
    if results.get("trajectories"):
        traj_file = output_file.replace(".json", "_trajectory.json")
        with open(traj_file, "w") as f:
            json.dump(results["trajectories"], f, indent=2)
        print(f"Trajectory saved to: {traj_file}")


if __name__ == "__main__":
    main()
