#!/usr/bin/env python3
"""Quick comparison: SD-MCTS 10k vs PPO 100k on fatigue_threshold_10."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np

from experiments.run_sample_efficiency import ScenarioConfig, PPOAgent, train_ppo, evaluate_agent
from src.world_model.task_sd_model import TaskSDModel
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.planning.mcts import mcts_search
from train_muzero import collect_episode, train_step, ReplayBuffer

print("=" * 60)
print("Quick Comparison: SD-MCTS 10k vs PPO 100k")
print("Scenario: fatigue_threshold_10")
print("=" * 60)

# Get scenario
scenarios = [s for s in ScenarioConfig.get_m1_scenarios() if s.name == "fatigue_threshold_10"]
scenario = scenarios[0]

n_workers = len(scenario.env_kwargs.get("worker_rates", (1.0, 0.8)))
fatigue_threshold = scenario.env_kwargs.get("fatigue_threshold", None)
fatigue_penalty = scenario.env_kwargs.get("fatigue_penalty", 0.5)

print(f"\nScenario config:")
print(f"  fatigue_threshold: {fatigue_threshold}")
print(f"  arrival_rate: {scenario.env_kwargs.get('arrival_rate')}")

# ========== SD-MCTS Training (10k steps) ==========
print("\n" + "=" * 60)
print("[SD-MCTS] Training for 10000 steps...")
print("=" * 60)

env = scenario.build_env(seed=42)
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

world_model = TaskSDModel(
    n_workers=n_workers,
    fatigue_threshold=fatigue_threshold,
    fatigue_penalty=fatigue_penalty,
)
policy_net = PolicyNetwork(state_dim, n_actions, hidden_dim=64)
value_net = ValueNetwork(state_dim, hidden_dim=64)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
model_optimizer = optim.Adam(world_model.parameters(), lr=1e-3)

buffer = ReplayBuffer(capacity=10000)
env_steps = 0
sdmcts_rewards = []

torch.manual_seed(42)
np.random.seed(42)

while env_steps < 10000:
    transitions = collect_episode(
        env, world_model, n_actions=n_actions,
        policy_net=policy_net, value_net=value_net,
        num_simulations=20, gamma=0.99,
        n_workers=n_workers
    )
    for t in transitions:
        buffer.add(t)
    env_steps += len(transitions)

    if len(buffer) >= 64:
        for _ in range(5):
            train_step(
                policy_net, value_net, world_model, buffer,
                policy_optimizer, value_optimizer, model_optimizer,
                batch_size=64
            )

    if env_steps >= len(sdmcts_rewards) * 2000 + 2000:
        # Eval
        rewards = []
        for ep in range(5):
            eval_reward = 0
            state, _ = env.reset(seed=100+ep)
            for _ in range(100):
                action = mcts_search(
                    state, world_model, n_actions=n_actions,
                    num_simulations=50, gamma=0.99,
                    policy_net=policy_net, value_net=value_net,
                    n_workers=n_workers
                )
                state, reward, term, trunc, _ = env.step(action)
                eval_reward += reward
                if term or trunc:
                    break
            rewards.append(eval_reward)
        avg_reward = np.mean(rewards)
        sdmcts_rewards.append(avg_reward)
        print(f"  @ {env_steps:5d} steps: reward={avg_reward:.2f}")

sdmcts_final = sdmcts_rewards[-1] if sdmcts_rewards else 0

# ========== PPO Training (100k steps) ==========
print("\n" + "=" * 60)
print("[PPO] Training for 100000 steps...")
print("=" * 60)

env = scenario.build_env(seed=42)
torch.manual_seed(42)
np.random.seed(42)

ppo_agent = PPOAgent(
    state_dim=state_dim,
    action_dim=n_actions,
    hidden_dim=64,
    n_hidden_layers=2,
    lr=3e-4,
    gamma=0.99,
    eps_clip=0.2,
)

# Use the train_ppo function from run_sample_efficiency
ppo_results = train_ppo(
    env, ppo_agent,
    total_steps=100000,
    eval_interval=20000,
    eval_episodes=5,
)

ppo_final = ppo_results["rewards"][-1] if ppo_results["rewards"] else 0

# ========== Summary ==========
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\nSD-MCTS @ 10k steps:  reward = {sdmcts_final:.2f}")
print(f"PPO @ 100k steps:     reward = {ppo_final:.2f}")
print(f"\nSample efficiency ratio: 10x")

if sdmcts_final > ppo_final:
    improvement = (sdmcts_final - ppo_final) / abs(ppo_final) * 100
    print(f"✓ SD-MCTS with 10x LESS data outperforms PPO by {improvement:.1f}%")
else:
    gap = ppo_final - sdmcts_final
    print(f"  PPO still better by {gap:.2f} (but used 10x more samples)")
