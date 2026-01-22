#!/usr/bin/env python
"""Debug version: Run probabilistic collapse experiment with detailed logging."""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from m1_framework import (
    ScenarioConfig, SDMCTSWrapper, MPCPolicy, TaskAllocationEnv,
)
from src.world_model import TaskSDModel
from src.agent import PolicyNetwork, ValueNetwork
from train_muzero import collect_episode, train_step, ReplayBuffer

def train_with_logging(scenario, env_steps_budget=15000, log_interval=1000):
    """Train SD-MCTS with detailed sigmoid parameter logging."""
    env = scenario.build_env(seed=42)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_workers = len(scenario.worker_rates)

    # Initialize world model with belief parameters
    init_sigmoid_center = scenario.belief_sigmoid_center or scenario.collapse_sigmoid_center
    init_sigmoid_temp = scenario.belief_sigmoid_temp or scenario.collapse_sigmoid_temp

    world_model = TaskSDModel(
        n_workers=n_workers,
        fatigue_threshold=scenario.fatigue_threshold,
        fatigue_penalty=scenario.fatigue_penalty,
        learn_fatigue_dynamics=(scenario.fatigue_collapse_threshold is not None),
        learn_probabilistic_collapse=scenario.probabilistic_collapse,
        init_sigmoid_center=init_sigmoid_center,
        init_sigmoid_temp=init_sigmoid_temp,
        probabilistic_collapse_penalty=scenario.collapse_penalty,
    )
    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    model_opt = optim.Adam(world_model.parameters(), lr=1e-2)
    buffer = ReplayBuffer(capacity=10000)

    print("=" * 70)
    print("SD-MCTS Training with Sigmoid Parameter Logging")
    print("=" * 70)
    print(f"True parameters:   μ={scenario.collapse_sigmoid_center}, T={scenario.collapse_sigmoid_temp}")
    print(f"Initial belief:    μ={init_sigmoid_center}, T={init_sigmoid_temp}")
    print("-" * 70)
    print(f"{'Steps':>8} | {'μ':>8} | {'T':>8} | {'Model Loss':>12} | {'Collapses':>10} | {'Fatigue':>8}")
    print("-" * 70)

    env_steps = 0
    total_collapses = 0
    total_model_loss = 0.0
    loss_count = 0

    while env_steps < env_steps_budget:
        # Collect episode
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=20
        )
        env_steps += len(transitions)

        # Count collapses in this episode
        ep_collapses = 0
        max_fatigue = 0.0
        for t in transitions:
            buffer.add(t)
            # Count collapse events
            if t.collapse_occurred > 0.5:
                ep_collapses += 1
            # Track max fatigue
            if len(t.next_state) > n_workers + 1:
                fatigue = t.next_state[-1]
                max_fatigue = max(max_fatigue, fatigue)
        total_collapses += ep_collapses

        # Train with BCE loss for collapse supervision
        total_bce = 0.0
        for _ in range(5):
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                collapse_labels = torch.tensor([t.collapse_occurred for t in batch], dtype=torch.float32)

                pred_next = world_model(states, actions)
                mse_loss = F.mse_loss(pred_next, next_states)

                # BCE loss for explicit collapse supervision
                pred_collapse_prob = world_model.get_collapse_prob(states)
                bce_loss = F.binary_cross_entropy(pred_collapse_prob, collapse_labels)

                # T regularization to prevent divergence (log-normal prior)
                t_reg_loss = world_model.get_t_regularization_loss(lambda_t=0.1)

                # Combined loss: MSE + BCE + T_regularization
                loss = mse_loss + 0.5 * bce_loss + t_reg_loss

                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

                total_model_loss += mse_loss.item()
                total_bce += bce_loss.item()
                loss_count += 1

        # Log progress
        if env_steps % log_interval < len(transitions) or env_steps >= env_steps_budget:
            if hasattr(world_model, '_sigmoid_center'):
                current_mu = world_model._sigmoid_center.item()
                current_T = F.softplus(world_model._sigmoid_temp_raw).item() + 0.1
            else:
                current_mu, current_T = 0, 0

            avg_loss = total_model_loss / max(loss_count, 1)
            print(f"{env_steps:>8} | {current_mu:>8.3f} | {current_T:>8.3f} | {avg_loss:>12.4f} | {total_collapses:>10} | {max_fatigue:>8.2f}")

    print("-" * 70)
    print(f"Final parameters: μ={current_mu:.3f} (target: {scenario.collapse_sigmoid_center})")
    print(f"                  T={current_T:.3f} (target: {scenario.collapse_sigmoid_temp})")
    print("=" * 70)

    # Check gradients
    print("\nGradient analysis on final batch:")
    if len(buffer) >= 64:
        batch = buffer.sample(64)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        collapse_labels = torch.tensor([t.collapse_occurred for t in batch], dtype=torch.float32)

        world_model.zero_grad()
        pred_next = world_model(states, actions)
        mse_loss = F.mse_loss(pred_next, next_states)
        pred_collapse_prob = world_model.get_collapse_prob(states)
        bce_loss = F.binary_cross_entropy(pred_collapse_prob, collapse_labels)
        loss = mse_loss + 0.5 * bce_loss
        loss.backward()

        if hasattr(world_model, '_sigmoid_center'):
            grad_mu = world_model._sigmoid_center.grad
            grad_T = world_model._sigmoid_temp_raw.grad
            print(f"  d(loss)/d(μ) = {grad_mu.item() if grad_mu is not None else 'None':.6f}")
            print(f"  d(loss)/d(T_raw) = {grad_T.item() if grad_T is not None else 'None':.6f}")

    # Analyze prediction errors
    print("\nPrediction error analysis:")
    fatigue_errors = []
    for t in list(buffer.buffer)[-100:]:
        if len(t.state) > n_workers + 1:
            state = torch.tensor([t.state], dtype=torch.float32)
            action = torch.tensor([t.action], dtype=torch.int64)
            with torch.no_grad():
                pred = world_model(state, action)
            actual_fatigue = t.next_state[-1]
            pred_fatigue = pred[0, -1].item()
            fatigue_errors.append(pred_fatigue - actual_fatigue)

    if fatigue_errors:
        print(f"  Fatigue prediction error (pred - actual):")
        print(f"    Mean: {np.mean(fatigue_errors):.4f}")
        print(f"    Std:  {np.std(fatigue_errors):.4f}")
        print(f"    Range: [{min(fatigue_errors):.4f}, {max(fatigue_errors):.4f}]")

    return world_model


if __name__ == "__main__":
    scenarios = [s for s in ScenarioConfig.generate_m1_matrix()
                 if s.name == "probabilistic_collapse_fairness"]
    if not scenarios:
        print("ERROR: Scenario not found!")
        exit(1)

    scenario = scenarios[0]
    print(f"Scenario: {scenario.name}")
    print(f"  arrival_rate = {scenario.arrival_rate}")
    print(f"  worker_rates = {scenario.worker_rates}")
    print(f"  fatigue_accumulation_threshold = {scenario.fatigue_accumulation_threshold}")
    print()

    trained_model = train_with_logging(scenario, env_steps_budget=15000)
