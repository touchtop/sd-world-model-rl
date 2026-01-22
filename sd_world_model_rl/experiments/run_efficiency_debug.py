#!/usr/bin/env python
"""Debug version: Run efficiency degradation experiment with detailed logging."""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from m1_framework import ScenarioConfig
from src.world_model import TaskSDModel
from src.agent import PolicyNetwork, ValueNetwork
from train_muzero import collect_episode, ReplayBuffer


def train_with_logging(scenario, env_steps_budget=15000, log_interval=1000):
    """Train SD-MCTS with detailed efficiency parameter logging."""
    env = scenario.build_env(seed=42)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_workers = len(scenario.worker_rates)

    # Initialize world model with belief parameters
    init_alpha = scenario.belief_efficiency_alpha or scenario.efficiency_alpha
    init_theta = scenario.belief_efficiency_threshold or scenario.efficiency_threshold

    world_model = TaskSDModel(
        n_workers=n_workers,
        fatigue_threshold=scenario.fatigue_threshold,
        fatigue_penalty=scenario.fatigue_penalty,
        learn_fatigue_dynamics=(scenario.fatigue_collapse_threshold is not None),
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

    print("=" * 80)
    print("SD-MCTS Training with Efficiency Degradation Parameter Logging")
    print("=" * 80)
    print(f"True parameters:   α={scenario.efficiency_alpha}, θ={scenario.efficiency_threshold}")
    print(f"Initial belief:    α={init_alpha}, θ={init_theta}")
    print("-" * 80)
    print(f"{'Steps':>8} | {'α':>10} | {'θ':>10} | {'MSE Loss':>12} | {'Eff Loss':>10} | {'Fatigue':>8} | {'Efficiency':>10}")
    print("-" * 80)

    env_steps = 0
    total_mse_loss = 0.0
    total_eff_loss = 0.0
    loss_count = 0

    while env_steps < env_steps_budget:
        # Collect episode
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=20
        )
        env_steps += len(transitions)

        # Track fatigue and efficiency
        max_fatigue = 0.0
        min_efficiency = 1.0
        for t in transitions:
            buffer.add(t)
            if len(t.next_state) > n_workers + 1:
                fatigue = t.next_state[-1]
                max_fatigue = max(max_fatigue, fatigue)

        # Get last efficiency from env info
        if hasattr(env, '_last_info') and 'efficiency_coef' in env._last_info:
            min_efficiency = env._last_info['efficiency_coef']

        # Train with efficiency supervision
        for _ in range(5):
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)

                # Get fatigue from states
                fatigue_states = states[:, -1] if states.shape[-1] > n_workers + 1 else None

                pred_next = world_model(states, actions)
                mse_loss = F.mse_loss(pred_next, next_states)

                # Efficiency supervision: compare predicted vs actual efficiency
                eff_loss = torch.tensor(0.0)
                if world_model.learn_efficiency_degradation and fatigue_states is not None:
                    pred_eff = world_model.get_efficiency_coef(fatigue_states)
                    # Compute true efficiency for supervision
                    true_alpha = scenario.efficiency_alpha
                    true_theta = scenario.efficiency_threshold
                    excess = F.relu(fatigue_states - true_theta)
                    true_eff = torch.clamp(1.0 - true_alpha * excess * excess, min=0.0)
                    eff_loss = F.mse_loss(pred_eff, true_eff)

                # Combined loss
                loss = mse_loss + 0.5 * eff_loss

                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

                total_mse_loss += mse_loss.item()
                total_eff_loss += eff_loss.item()
                loss_count += 1

        # Log progress
        if env_steps % log_interval < len(transitions) or env_steps >= env_steps_budget:
            if hasattr(world_model, '_efficiency_alpha_raw'):
                current_alpha = F.softplus(world_model._efficiency_alpha_raw).item()
                current_theta = world_model._efficiency_threshold.item()
            else:
                current_alpha, current_theta = 0, 0

            avg_mse = total_mse_loss / max(loss_count, 1)
            avg_eff = total_eff_loss / max(loss_count, 1)
            print(f"{env_steps:>8} | {current_alpha:>10.5f} | {current_theta:>10.3f} | {avg_mse:>12.4f} | {avg_eff:>10.6f} | {max_fatigue:>8.2f} | {min_efficiency:>10.3f}")

    print("-" * 80)
    print(f"Final parameters: α={current_alpha:.5f} (target: {scenario.efficiency_alpha})")
    print(f"                  θ={current_theta:.3f} (target: {scenario.efficiency_threshold})")
    print("=" * 80)

    # Check gradients
    print("\nGradient analysis on final batch:")
    if len(buffer) >= 64:
        batch = buffer.sample(64)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        fatigue_states = states[:, -1] if states.shape[-1] > n_workers + 1 else None

        world_model.zero_grad()
        pred_next = world_model(states, actions)
        mse_loss = F.mse_loss(pred_next, next_states)

        if world_model.learn_efficiency_degradation and fatigue_states is not None:
            pred_eff = world_model.get_efficiency_coef(fatigue_states)
            true_alpha = scenario.efficiency_alpha
            true_theta = scenario.efficiency_threshold
            excess = F.relu(fatigue_states - true_theta)
            true_eff = torch.clamp(1.0 - true_alpha * excess * excess, min=0.0)
            eff_loss = F.mse_loss(pred_eff, true_eff)
            loss = mse_loss + 0.5 * eff_loss
        else:
            loss = mse_loss

        loss.backward()

        if hasattr(world_model, '_efficiency_alpha_raw'):
            grad_alpha = world_model._efficiency_alpha_raw.grad
            grad_theta = world_model._efficiency_threshold.grad
            print(f"  d(loss)/d(α_raw) = {grad_alpha.item() if grad_alpha is not None else 'None':.6f}")
            print(f"  d(loss)/d(θ) = {grad_theta.item() if grad_theta is not None else 'None':.6f}")

    # Analyze fatigue distribution
    print("\nFatigue distribution analysis:")
    fatigue_values = []
    for t in list(buffer.buffer)[-200:]:
        if len(t.state) > n_workers + 1:
            fatigue_values.append(t.state[-1])

    if fatigue_values:
        print(f"  Fatigue range: [{min(fatigue_values):.2f}, {max(fatigue_values):.2f}]")
        print(f"  Fatigue mean: {np.mean(fatigue_values):.2f}")
        print(f"  Fatigue std: {np.std(fatigue_values):.2f}")
        # Count samples in different regions
        below_theta = sum(1 for f in fatigue_values if f < scenario.efficiency_threshold)
        around_theta = sum(1 for f in fatigue_values if scenario.efficiency_threshold <= f < scenario.efficiency_threshold + 5)
        above_theta = sum(1 for f in fatigue_values if f >= scenario.efficiency_threshold + 5)
        print(f"  Samples below θ ({scenario.efficiency_threshold}): {below_theta} ({100*below_theta/len(fatigue_values):.1f}%)")
        print(f"  Samples around θ: {around_theta} ({100*around_theta/len(fatigue_values):.1f}%)")
        print(f"  Samples above θ+5: {above_theta} ({100*above_theta/len(fatigue_values):.1f}%)")

    return world_model


if __name__ == "__main__":
    scenarios = [s for s in ScenarioConfig.generate_m1_matrix()
                 if s.name == "efficiency_degradation_fairness"]
    if not scenarios:
        print("ERROR: Scenario 'efficiency_degradation_fairness' not found!")
        exit(1)

    scenario = scenarios[0]
    print(f"Scenario: {scenario.name}")
    print(f"  arrival_rate = {scenario.arrival_rate}")
    print(f"  worker_rates = {scenario.worker_rates}")
    print(f"  fatigue_accumulation_threshold = {scenario.fatigue_accumulation_threshold}")
    print(f"  efficiency_alpha = {scenario.efficiency_alpha} (true)")
    print(f"  efficiency_threshold = {scenario.efficiency_threshold} (true)")
    print()

    trained_model = train_with_logging(scenario, env_steps_budget=15000)
