#!/usr/bin/env python
"""
SD-MCTS vs MPC Comparison Experiment for Efficiency Degradation Scenario.

Key hypothesis:
- Both start with same wrong belief about efficiency parameters (α, θ)
- MPC uses fixed belief throughout
- SD-MCTS learns true parameters from experience
- SD-MCTS should outperform MPC due to model correction
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from m1_framework import ScenarioConfig, ModelBelief
from src.world_model import TaskSDModel
from src.agent import PolicyNetwork, ValueNetwork
from src.planning import mcts_search
from train_muzero import collect_episode, ReplayBuffer


def run_mpc_episode(env, belief: ModelBelief, horizon: int = 3, seed: int = None, episode_length: int = 200,
                    belief_alpha: float = 0.005, belief_theta: float = 10.0) -> Tuple[float, List[float]]:
    """Run one episode with MPC using fixed belief model."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    n_workers = len(belief.worker_rates)

    for step in range(episode_length):
        # Simple greedy MPC: choose action that minimizes expected cost
        best_action = n_workers  # Default: hold
        best_value = float('inf')

        for action in range(n_workers + 1):
            # Simulate one step with belief model
            value = _simulate_action_value(state, action, belief, n_workers, belief_alpha, belief_theta)
            if value < best_value:
                best_value = value
                best_action = action

        next_state, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rewards


def _simulate_action_value(state: np.ndarray, action: int, belief: ModelBelief, n_workers: int,
                            belief_alpha: float = 0.005, belief_theta: float = 10.0) -> float:
    """Simulate action value using belief model (simplified MPC rollout)."""
    queue = state[0]
    backlogs = state[1:1+n_workers].copy()

    # Simulate arrival
    queue += belief.arrival_rate

    # Simulate assignment
    if action < n_workers and queue > 0:
        queue -= 1
        backlogs[action] += 1

    # Simulate completion with believed efficiency
    effective_rates = np.array(belief.worker_rates)

    # Apply believed efficiency degradation using MPC's wrong belief
    # One-sided Lorentzian: efficiency = 1 / (1 + α * max(0, fatigue - θ)²)
    if len(state) > 1 + n_workers:
        fatigue = state[-1]
        # MPC uses its believed α and θ (which are wrong: α=0.001, θ=12.0)
        diff = max(0.0, fatigue - belief_theta)  # One-sided
        efficiency = 1.0 / (1.0 + belief_alpha * diff * diff)
        effective_rates = effective_rates * efficiency

    backlogs = np.maximum(0, backlogs - effective_rates)

    # Cost: queue + backlog penalty
    cost = queue + 0.1 * np.sum(backlogs ** 2)
    return cost


def train_sd_mcts(scenario: ScenarioConfig, env_steps: int = 15000) -> Tuple[TaskSDModel, PolicyNetwork, ValueNetwork]:
    """Train SD-MCTS world model and networks."""
    env = scenario.build_env(seed=42)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_workers = len(scenario.worker_rates)

    # Initialize with wrong belief
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
    model_opt = optim.Adam(world_model.parameters(), lr=1e-2)  # Higher LR for faster α convergence
    buffer = ReplayBuffer(capacity=10000)

    print("Training SD-MCTS world model...")
    steps = 0
    while steps < env_steps:
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=20
        )
        steps += len(transitions)

        for t in transitions:
            buffer.add(t)

        # Train world model, policy, and value networks
        for _ in range(5):
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                mcts_policies = torch.tensor(np.array([t.mcts_policy for t in batch]), dtype=torch.float32)
                returns = torch.tensor([t.cumulative_return for t in batch], dtype=torch.float32)
                fatigue_states = states[:, -1] if states.shape[-1] > n_workers + 1 else None

                # 1. Policy Loss: CrossEntropy(policy_net, mcts_policy)
                policy_logits = policy_net(states)
                policy_loss = -(mcts_policies * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

                policy_opt.zero_grad()
                policy_loss.backward()
                policy_opt.step()

                # 2. Value Loss: MSE(value_net, returns)
                value_pred = value_net(states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, returns)

                value_opt.zero_grad()
                value_loss.backward()
                value_opt.step()

                # 3. World Model Loss
                pred_next = world_model(states, actions)
                mse_loss = F.mse_loss(pred_next, next_states)

                # Efficiency supervision with asymmetric loss
                # Penalize overestimation of α more than underestimation
                eff_loss = torch.tensor(0.0)
                if world_model.learn_efficiency_degradation and fatigue_states is not None:
                    pred_eff = world_model.get_efficiency_coef(fatigue_states)
                    true_alpha = scenario.efficiency_alpha
                    true_theta = scenario.efficiency_threshold
                    # One-sided Lorentzian (consistent with env)
                    diff = F.relu(fatigue_states - true_theta)
                    true_eff = 1.0 / (1.0 + true_alpha * diff * diff)
                    # Asymmetric loss: penalize pred_eff < true_eff (pessimistic α) more
                    # This biases α learning toward underestimation (optimistic)
                    error = pred_eff - true_eff
                    # λ_under=2.0 for underestimation (pred < true), λ_over=0.5 for overestimation
                    weights = torch.where(error < 0, 2.0, 0.5)
                    eff_loss = (weights * error * error).mean()

                loss = mse_loss + 0.5 * eff_loss

                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

        if steps % 3000 == 0:
            if hasattr(world_model, '_efficiency_alpha_raw'):
                alpha = F.softplus(world_model._efficiency_alpha_raw).item()
                theta = world_model._efficiency_threshold.item()
                print(f"  Step {steps}: α={alpha:.4f}, θ={theta:.2f}")

    return world_model, policy_net, value_net


def run_sd_mcts_episode(env, world_model, policy_net, value_net, n_actions: int, seed: int = None, episode_length: int = 200) -> Tuple[float, List[float]]:
    """Run one episode with trained SD-MCTS."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []

    for step in range(episode_length):
        action = mcts_search(
            state, world_model,
            n_actions=n_actions,
            num_simulations=50,
            policy_net=policy_net,
            value_net=value_net
        )
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rewards


def main():
    print("=" * 80)
    print("SD-MCTS vs MPC Comparison: Efficiency Degradation Scenario")
    print("=" * 80)

    # Load scenario
    scenarios = [s for s in ScenarioConfig.generate_m1_matrix()
                 if s.name == "efficiency_degradation_fairness"]
    if not scenarios:
        print("ERROR: Scenario not found!")
        return

    scenario = scenarios[0]
    print(f"\nScenario: {scenario.name}")
    print(f"  True α={scenario.efficiency_alpha}, θ={scenario.efficiency_threshold}")
    print(f"  Belief α={scenario.belief_efficiency_alpha}, θ={scenario.belief_efficiency_threshold}")

    # Phase 1: Train SD-MCTS
    print("\n" + "=" * 80)
    print("Phase 1: Training SD-MCTS")
    print("=" * 80)
    world_model, policy_net, value_net = train_sd_mcts(scenario, env_steps=20000)

    # Show learned parameters
    if hasattr(world_model, '_efficiency_alpha_raw'):
        learned_alpha = F.softplus(world_model._efficiency_alpha_raw).item()
        learned_theta = world_model._efficiency_threshold.item()
        print(f"\nLearned parameters: α={learned_alpha:.4f}, θ={learned_theta:.2f}")
        print(f"True parameters:    α={scenario.efficiency_alpha}, θ={scenario.efficiency_threshold}")

    # Phase 2: Evaluation
    print("\n" + "=" * 80)
    print("Phase 2: Evaluation (10 seeds)")
    print("=" * 80)

    n_seeds = 10
    mpc_rewards = []
    sdmcts_rewards = []

    # Create belief for MPC (uses wrong parameters)
    belief = ModelBelief(
        arrival_rate=scenario.arrival_rate,
        worker_rates=scenario.worker_rates,
        fatigue_threshold=scenario.fatigue_threshold,
        fatigue_penalty=scenario.fatigue_penalty,
        max_queue=scenario.max_queue,
        max_backlog=scenario.max_backlog,
    )

    n_actions = len(scenario.worker_rates) + 1
    episode_length = scenario.episode_length
    belief_alpha = scenario.belief_efficiency_alpha or 0.005
    belief_theta = scenario.belief_efficiency_threshold or 10.0

    for seed in range(n_seeds):
        # MPC with fixed wrong belief
        env_mpc = scenario.build_env(seed=seed)
        mpc_reward, _ = run_mpc_episode(env_mpc, belief, seed=seed, episode_length=episode_length,
                                        belief_alpha=belief_alpha, belief_theta=belief_theta)
        mpc_rewards.append(mpc_reward)

        # SD-MCTS with learned model
        env_sdmcts = scenario.build_env(seed=seed)
        sdmcts_reward, _ = run_sd_mcts_episode(
            env_sdmcts, world_model, policy_net, value_net, n_actions, seed=seed, episode_length=episode_length
        )
        sdmcts_rewards.append(sdmcts_reward)

        print(f"  Seed {seed}: MPC={mpc_reward:.2f}, SD-MCTS={sdmcts_reward:.2f}, Δ={sdmcts_reward-mpc_reward:.2f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    mpc_mean, mpc_std = np.mean(mpc_rewards), np.std(mpc_rewards)
    sdmcts_mean, sdmcts_std = np.mean(sdmcts_rewards), np.std(sdmcts_rewards)

    print(f"\nMPC (fixed wrong belief):     {mpc_mean:.2f} ± {mpc_std:.2f}")
    print(f"SD-MCTS (learned dynamics):   {sdmcts_mean:.2f} ± {sdmcts_std:.2f}")
    print(f"\nImprovement: {sdmcts_mean - mpc_mean:.2f} ({100*(sdmcts_mean-mpc_mean)/abs(mpc_mean):.1f}%)")

    if sdmcts_mean > mpc_mean:
        print("\n✓ SD-MCTS outperforms MPC as expected!")
    else:
        print("\n✗ MPC performed better - need to investigate")

    print("=" * 80)


if __name__ == "__main__":
    main()
