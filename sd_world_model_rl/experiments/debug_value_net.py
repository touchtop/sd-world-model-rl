#!/usr/bin/env python3
"""Diagnostic: Check value network predictions and MCTS behavior."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.delayed_collapse_env import DelayedCollapseEnv
from run_horizon_ablation import train_sd_mcts, set_seed, depth_limited_mcts_search, make_reward_fn


def diagnose():
    # Setup
    true_F = 10.0
    belief_F = 50.0
    alpha = 1.0
    beta = 0.1
    F_req = 5.0
    T_s = 12
    B = 15.0
    T = 30

    print("=" * 60)
    print("Value Network Diagnostic")
    print("=" * 60)

    # Train
    set_seed(42)
    env = DelayedCollapseEnv(alpha=alpha, beta=beta, F=true_F, F_req=F_req, T_s=T_s, B=B, T=T, seed=42)
    world_model, policy_net, value_net = train_sd_mcts(env, true_F=true_F, init_F_belief=belief_F, env_steps=15000)

    F_learned = world_model.collapse_threshold.item()
    print(f"\nLearned F: {F_learned:.2f} (true: {true_F})")
    print(f"Learned alpha: {world_model.alpha.item():.3f}")
    print(f"Learned beta: {world_model.beta.item():.3f}")

    # Check value network predictions for various states
    print("\n" + "=" * 60)
    print("Value Network Predictions for Various States")
    print("=" * 60)
    print(f"{'State (fatigue, alive, surge, time_to_surge)':<45} {'V(s)'}")
    print("-" * 60)

    test_states = [
        [0.0, 1.0, 0.0, 12.0],   # Initial state
        [0.0, 1.0, 0.0, 11.0],   # Step 1
        [2.5, 1.0, 0.0, 6.0],    # Midway, low fatigue
        [5.0, 1.0, 0.0, 1.0],    # Before surge, at F_req
        [5.0, 1.0, 1.0, 0.0],    # At surge time
        [5.0, 1.0, 0.0, 0.0],    # After surge
        [8.0, 1.0, 0.0, 0.0],    # High fatigue, alive
        [10.0, 0.0, 0.0, 0.0],   # Collapsed
    ]

    with torch.no_grad():
        for state in test_states:
            state_t = torch.tensor([state], dtype=torch.float32)
            v = value_net(state_t).item()
            print(f"{str(state):<45} {v:>8.2f}")

    # Check policy network preferences
    print("\n" + "=" * 60)
    print("Policy Network Action Probabilities")
    print("=" * 60)

    initial_state = torch.tensor([[0.0, 1.0, 0.0, 12.0]], dtype=torch.float32)
    with torch.no_grad():
        logits = policy_net(initial_state)
        probs = torch.softmax(logits, dim=-1)[0].numpy()

    print("Initial state action probs:")
    for a, p in enumerate(probs):
        intensity = a / (env.n_actions - 1)
        print(f"  action={a} (intensity={intensity:.2f}): prob={p:.3f}")

    # Run single episode with detailed output
    print("\n" + "=" * 60)
    print("Single Episode Trace (SD-MCTS with H=10)")
    print("=" * 60)

    eval_env = DelayedCollapseEnv(alpha=alpha, beta=beta, F=true_F, F_req=F_req, T_s=T_s, B=B, T=T, seed=1000)
    state, _ = eval_env.reset(seed=1000)
    reward_fn = make_reward_fn(env.n_actions, F_req, B, F_collapse=F_learned)

    total_reward = 0.0
    print(f"{'Step':>4} {'Action':>6} {'Intensity':>9} {'Fatigue':>8} {'V(s)':>8} {'Reward':>7}")
    print("-" * 60)

    for step in range(T):
        with torch.no_grad():
            state_t = torch.tensor([state], dtype=torch.float32)
            v = value_net(state_t).item()

            action = depth_limited_mcts_search(
                state, world_model,
                n_actions=env.n_actions,
                num_simulations=100,
                max_depth=10,
                policy_net=policy_net,
                value_net=value_net,
                reward_fn=reward_fn,
            )

        next_state, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        intensity = action / (env.n_actions - 1)

        print(f"{step:>4} {action:>6} {intensity:>9.2f} {state[0]:>8.2f} {v:>8.2f} {reward:>7.2f}")

        if info.get("surge_bonus_earned"):
            print("       *** SURGE BONUS CAPTURED ***")
        if info.get("collapsed"):
            print("       *** COLLAPSED ***")
            break

        state = next_state
        if terminated or truncated:
            break

    print("-" * 60)
    print(f"Total reward: {total_reward:.2f}")

    # Now trace MCTS-Zero for comparison
    print("\n" + "=" * 60)
    print("Single Episode Trace (MCTS-Zero with H=14)")
    print("=" * 60)

    eval_env = DelayedCollapseEnv(alpha=alpha, beta=beta, F=true_F, F_req=F_req, T_s=T_s, B=B, T=T, seed=1000)
    state, _ = eval_env.reset(seed=1000)

    total_reward = 0.0
    print(f"{'Step':>4} {'Action':>6} {'Intensity':>9} {'Fatigue':>8} {'Reward':>7}")
    print("-" * 60)

    for step in range(T):
        with torch.no_grad():
            action = depth_limited_mcts_search(
                state, world_model,
                n_actions=env.n_actions,
                num_simulations=100,
                max_depth=14,
                policy_net=policy_net,
                value_net=None,  # MCTS-Zero
                reward_fn=reward_fn,
            )

        next_state, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        intensity = action / (env.n_actions - 1)

        print(f"{step:>4} {action:>6} {intensity:>9.2f} {state[0]:>8.2f} {reward:>7.2f}")

        if info.get("surge_bonus_earned"):
            print("       *** SURGE BONUS CAPTURED ***")
        if info.get("collapsed"):
            print("       *** COLLAPSED ***")
            break

        state = next_state
        if terminated or truncated:
            break

    print("-" * 60)
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    diagnose()
