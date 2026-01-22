#!/usr/bin/env python
"""SD-MCTS vs MPC Comparison: Delayed Collapse + Demand Surge Scenario.

Mathematical proof of SD-MCTS structural advantage:

MPC fails because:
1. Cannot model latent fatigue (not in its state representation)
2. Short horizon (H=1-3) cannot see surge at T_s
3. Greedy a_t=1 causes collapse before surge

SD-MCTS wins because:
1. Learns true collapse threshold F through world model
2. Value network propagates surge bonus backward
3. Discovers conservative a_safe policy

Expected results:
- MPC: Collapses at t=10, total reward ≈ 10
- SD-MCTS: Survives to surge, total reward ≈ 20+ (>100% improvement)
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/data/round/code/sd_world_model_rl')

from src.env.delayed_collapse_env import DelayedCollapseEnv, DelayedCollapseWorldModel
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.agent import PolicyNetwork, ValueNetwork
from src.planning import mcts_search


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


def run_mpc_episode(
    env: DelayedCollapseEnv,
    belief_model: DelayedCollapseWorldModel,
    horizon: int = 1,
    seed: int = None,
) -> Tuple[float, List[float], dict]:
    """Run one episode with MPC using fixed belief model.

    MPC uses 1-step greedy: maximize immediate reward.
    With wrong belief about F, it will greedily use a_t=1 and collapse.
    """
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    collapse_step = None

    for step in range(env.T):
        # MPC: Choose action that maximizes immediate reward
        # Since reward = a_t and MPC doesn't see collapse coming, it chooses max action
        best_action = env.n_actions - 1  # a_t = 1.0

        # Simple 1-step lookahead (optional, but MPC still fails)
        if horizon > 0:
            best_value = -float('inf')
            for action in range(env.n_actions):
                pred_reward = belief_model.predict_reward(state, action)
                if pred_reward > best_value:
                    best_value = pred_reward
                    best_action = action

        next_state, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        rewards.append(reward)

        if info.get("collapsed") and collapse_step is None:
            collapse_step = step

        state = next_state
        if terminated or truncated:
            break

    return total_reward, rewards, {"collapse_step": collapse_step}


def compute_reward_from_state(state: np.ndarray, action: int, n_actions: int,
                               F_req: float, B: float) -> float:
    """Compute reward for MCTS planning."""
    fatigue, alive, surge_flag, time_to_surge = state

    if alive < 0.5:
        return 0.0

    a_t = action / (n_actions - 1)
    reward = a_t

    # Surge bonus
    if time_to_surge < 1.5 and time_to_surge > 0.5 and fatigue <= F_req:
        reward += B

    return reward


def make_reward_fn(n_actions: int, F_req: float, B: float):
    """Create a reward function closure for MCTS."""
    def reward_fn(state: np.ndarray) -> float:
        fatigue, alive, surge_flag, time_to_surge = state
        if alive < 0.5:
            return 0.0
        # Base reward estimate (assume mid-level action)
        reward = 0.5
        # Surge bonus (if eligible)
        if time_to_surge < 1.5 and time_to_surge > 0.5 and fatigue <= F_req:
            reward += B
        return reward
    return reward_fn


def train_sd_mcts(
    env: DelayedCollapseEnv,
    true_F: float,
    init_F_belief: float = 50.0,
    env_steps: int = 5000,
) -> Tuple[DelayedCollapseSDModel, PolicyNetwork, ValueNetwork]:
    """Train SD-MCTS world model and networks."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize with wrong belief (same as MPC)
    world_model = DelayedCollapseSDModel(
        n_actions=n_actions,
        init_alpha=env.alpha,
        init_beta=env.beta,
        init_F=init_F_belief,  # Wrong belief about collapse threshold
        F_req=env.F_req,
        T_s=env.T_s,
        B=env.B,
    )

    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    model_opt = optim.Adam(world_model.parameters(), lr=1e-2)

    buffer = ReplayBuffer(capacity=5000)

    # Create reward function for MCTS
    reward_fn = make_reward_fn(n_actions, env.F_req, env.B)

    print("Training SD-MCTS...")
    steps = 0
    episode = 0

    while steps < env_steps:
        state, _ = env.reset(seed=episode)
        episode_reward = 0.0
        episode_transitions = []

        for t in range(env.T):
            # Use MCTS for action selection during training
            with torch.no_grad():
                # Simple epsilon-greedy for exploration during early training
                if steps < 1000 or np.random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    action = mcts_search(
                        state, world_model,
                        n_actions=n_actions,
                        num_simulations=20,
                        policy_net=policy_net,
                        value_net=value_net,
                        reward_fn=reward_fn,
                    )

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            transition = Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated:
                break

        # Compute returns for value learning
        returns = []
        G = 0.0
        gamma = 0.99
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        # Training updates
        for _ in range(5):
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                rewards_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32)

                # World model loss: predict next state
                pred_next = world_model(states, actions)
                model_loss = F.mse_loss(pred_next, next_states)

                # Add supervision for collapse threshold learning
                # If we observe a collapse (alive goes from 1 to 0), learn from it
                alive_current = states[:, 1]
                alive_next = next_states[:, 1]
                fatigue_next = next_states[:, 0]  # Use next_state fatigue (at collapse point)
                collapse_mask = (alive_current > 0.5) & (alive_next < 0.5)

                if collapse_mask.any():
                    # The fatigue at collapse should be near F (use next_state fatigue)
                    collapse_fatigue = fatigue_next[collapse_mask]
                    # Encourage learned F to match observed collapse point
                    F_learned = world_model.collapse_threshold
                    collapse_loss = F.mse_loss(
                        F_learned.expand(collapse_fatigue.shape[0]),
                        collapse_fatigue
                    )
                    model_loss = model_loss + 0.5 * collapse_loss

                model_opt.zero_grad()
                model_loss.backward()
                model_opt.step()

        # Value network training on episode returns
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(
                np.array([t.state for t in episode_transitions]),
                dtype=torch.float32
            )
            ep_returns = torch.tensor(returns, dtype=torch.float32)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode += 1

        if steps % 1000 == 0 or steps >= env_steps:
            F_learned = world_model.collapse_threshold.item()
            a_safe = world_model.get_safe_intensity()
            print(f"  Step {steps}: F_learned={F_learned:.2f} (true={true_F}), a_safe={a_safe:.3f}")

    return world_model, policy_net, value_net


def run_sd_mcts_episode(
    env: DelayedCollapseEnv,
    world_model: DelayedCollapseSDModel,
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    seed: int = None,
) -> Tuple[float, List[float], dict]:
    """Run one episode with trained SD-MCTS."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    collapse_step = None
    surge_bonus_earned = False

    # Create reward function for MCTS
    reward_fn = make_reward_fn(env.n_actions, env.F_req, env.B)

    for step in range(env.T):
        action = mcts_search(
            state, world_model,
            n_actions=env.n_actions,
            num_simulations=50,
            policy_net=policy_net,
            value_net=value_net,
            reward_fn=reward_fn,
        )

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)

        if info.get("collapsed") and collapse_step is None:
            collapse_step = step
        if info.get("surge_bonus_earned"):
            surge_bonus_earned = True

        state = next_state
        if terminated or truncated:
            break

    return total_reward, rewards, {
        "collapse_step": collapse_step,
        "surge_bonus_earned": surge_bonus_earned
    }


def run_optimal_episode(env: DelayedCollapseEnv, seed: int = None) -> Tuple[float, List[float], dict]:
    """Run with optimal policy for reference."""
    state, _ = env.reset(seed=seed)
    a_safe = env.get_optimal_intensity()
    a_safe_action = int(a_safe * (env.n_actions - 1))

    total_reward = 0.0
    rewards = []

    for step in range(env.T):
        # Use safe intensity before surge, max after
        if step < env.T_s:
            action = a_safe_action
        else:
            action = env.n_actions - 1  # Max intensity after surge

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)

        state = next_state
        if terminated or truncated:
            break

    return total_reward, rewards, {"policy": "optimal", "a_safe": a_safe}


def main():
    print("=" * 80)
    print("SD-MCTS vs MPC: Delayed Collapse + Demand Surge")
    print("=" * 80)

    # Environment parameters (Codex recommended)
    true_F = 10.0          # True collapse threshold
    belief_F = 50.0        # MPC believes it's much higher
    alpha = 1.0
    beta = 0.1
    F_req = 5.0
    T_s = 12
    B = 15.0
    T = 30

    print(f"\nScenario Configuration:")
    print(f"  True collapse threshold F = {true_F}")
    print(f"  MPC belief F = {belief_F} (5x overestimate)")
    print(f"  Fatigue rate α = {alpha}, recovery β = {beta}")
    print(f"  Surge at T_s = {T_s}, bonus B = {B} if fatigue ≤ {F_req}")
    print(f"  Episode length T = {T}")

    # Compute theoretical values
    T_collapse_greedy = int(true_F / alpha)
    a_safe = (F_req + beta * T_s) / ((alpha + beta) * T_s)
    print(f"\nTheoretical Analysis:")
    print(f"  MPC (greedy a=1) collapses at t = {T_collapse_greedy}")
    print(f"  Optimal safe intensity a_safe = {a_safe:.3f}")
    print(f"  Expected MPC reward ≈ {T_collapse_greedy}")
    print(f"  Expected optimal reward ≈ {T_s * a_safe + B:.1f}")

    # Create environment and belief model
    env = DelayedCollapseEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        T_s=T_s, B=B, T=T, seed=42
    )

    belief_model = DelayedCollapseWorldModel(
        belief_alpha=alpha, belief_beta=beta, belief_F=belief_F,
        belief_F_req=F_req, belief_T_s=T_s, belief_B=B,
        n_actions=env.n_actions
    )

    # Phase 1: Train SD-MCTS
    print("\n" + "=" * 80)
    print("Phase 1: Training SD-MCTS")
    print("=" * 80)

    world_model, policy_net, value_net = train_sd_mcts(
        env, true_F=true_F, init_F_belief=belief_F, env_steps=10000
    )

    F_learned = world_model.collapse_threshold.item()
    print(f"\nLearned parameters:")
    print(f"  F = {F_learned:.2f} (true = {true_F})")
    print(f"  α = {world_model.alpha.item():.3f} (true = {alpha})")
    print(f"  β = {world_model.beta.item():.3f} (true = {beta})")

    # Phase 2: Evaluation
    print("\n" + "=" * 80)
    print("Phase 2: Evaluation (10 seeds)")
    print("=" * 80)

    n_seeds = 10
    mpc_rewards = []
    sdmcts_rewards = []
    optimal_rewards = []
    mpc_collapses = 0
    sdmcts_collapses = 0
    sdmcts_surge_earned = 0

    for seed in range(n_seeds):
        # MPC
        mpc_reward, _, mpc_info = run_mpc_episode(env, belief_model, seed=seed)
        mpc_rewards.append(mpc_reward)
        if mpc_info["collapse_step"] is not None:
            mpc_collapses += 1

        # SD-MCTS
        sdmcts_reward, _, sdmcts_info = run_sd_mcts_episode(
            env, world_model, policy_net, value_net, seed=seed
        )
        sdmcts_rewards.append(sdmcts_reward)
        if sdmcts_info["collapse_step"] is not None:
            sdmcts_collapses += 1
        if sdmcts_info["surge_bonus_earned"]:
            sdmcts_surge_earned += 1

        # Optimal
        optimal_reward, _, _ = run_optimal_episode(env, seed=seed)
        optimal_rewards.append(optimal_reward)

        print(f"  Seed {seed}: MPC={mpc_reward:.2f}, SD-MCTS={sdmcts_reward:.2f}, "
              f"Optimal={optimal_reward:.2f}, Δ={sdmcts_reward-mpc_reward:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    mpc_mean, mpc_std = np.mean(mpc_rewards), np.std(mpc_rewards)
    sdmcts_mean, sdmcts_std = np.mean(sdmcts_rewards), np.std(sdmcts_rewards)
    optimal_mean = np.mean(optimal_rewards)

    print(f"\nMPC (fixed wrong belief):     {mpc_mean:.2f} ± {mpc_std:.2f}")
    print(f"SD-MCTS (learned dynamics):   {sdmcts_mean:.2f} ± {sdmcts_std:.2f}")
    print(f"Optimal (reference):          {optimal_mean:.2f}")

    print(f"\nCollapse statistics:")
    print(f"  MPC collapses: {mpc_collapses}/{n_seeds}")
    print(f"  SD-MCTS collapses: {sdmcts_collapses}/{n_seeds}")
    print(f"  SD-MCTS surge bonuses earned: {sdmcts_surge_earned}/{n_seeds}")

    improvement = (sdmcts_mean - mpc_mean) / abs(mpc_mean) * 100
    print(f"\nImprovement: {sdmcts_mean - mpc_mean:.2f} ({improvement:.1f}%)")

    if improvement > 30:
        print(f"\n✓ SD-MCTS shows SIGNIFICANT structural advantage (>{30}%)")
    elif improvement > 0:
        print(f"\n✓ SD-MCTS outperforms MPC, but advantage < 30%")
    else:
        print(f"\n✗ MPC performed better - need to investigate")

    print("=" * 80)


if __name__ == "__main__":
    main()
