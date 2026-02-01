#!/usr/bin/env python3
"""Experiment: Multi-Surge Horizon Ablation.

Tests Theorem 2 with multiple irregular surge events.
This increases planning complexity - value network should show
greater advantage over pure tree search.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.delayed_collapse_env import MultiSurgeEnv
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.agent import PolicyNetwork, ValueNetwork
from run_horizon_ablation import (
    set_seed, ReplayBuffer, Transition, MCTSNode, _puct_score
)


class MultiSurgeWorldModel(nn.Module):
    """World model for multi-surge environment."""

    def __init__(self, n_actions: int, surge_times: list, surge_bonuses: list,
                 init_alpha: float = 1.0, init_beta: float = 0.1,
                 init_F: float = 50.0, F_req: float = 5.0):
        super().__init__()
        self.n_actions = n_actions
        self.surge_times = surge_times
        self.surge_bonuses = surge_bonuses
        self.F_req = F_req

        self._alpha_raw = nn.Parameter(torch.tensor(np.log(init_alpha)))
        self._beta_raw = nn.Parameter(torch.tensor(np.log(init_beta)))
        self._F_raw = nn.Parameter(torch.tensor(np.log(init_F)))

    @property
    def alpha(self):
        return torch.exp(self._alpha_raw)

    @property
    def beta(self):
        return torch.exp(self._beta_raw)

    @property
    def collapse_threshold(self):
        return torch.exp(self._F_raw)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        fatigue = state[:, 0]
        alive = state[:, 1]
        surge_idx = state[:, 2]
        time_to_next = state[:, 3]

        a_t = action.float() / (self.n_actions - 1)

        # Fatigue dynamics
        new_fatigue = fatigue + (self.alpha + self.beta) * a_t - self.beta
        new_fatigue = torch.clamp(new_fatigue, min=0.0)

        # Collapse check
        new_alive = torch.where(new_fatigue >= self.collapse_threshold,
                                torch.zeros_like(alive), alive)

        # Update surge tracking
        new_time_to_next = torch.clamp(time_to_next - 1, min=0.0)
        # Increment surge_idx when time_to_next reaches 0
        surge_triggered = (time_to_next > 0.5) & (new_time_to_next < 0.5)
        new_surge_idx = surge_idx + surge_triggered.float()

        # Compute new time_to_next for next surge
        for i, st in enumerate(self.surge_times):
            mask = (new_surge_idx == i) & (surge_triggered)
            if i + 1 < len(self.surge_times):
                next_time = self.surge_times[i + 1] - self.surge_times[i]
                new_time_to_next = torch.where(mask, torch.full_like(new_time_to_next, next_time - 1), new_time_to_next)

        return torch.stack([new_fatigue, new_alive, new_surge_idx, new_time_to_next], dim=1)


def make_multi_surge_reward_fn(n_actions: int, F_req: float, surge_times: list,
                                surge_bonuses: list, F_collapse: float = None):
    """Create reward function for multi-surge environment."""
    def reward_fn(state: np.ndarray, action: int = None) -> float:
        fatigue, alive, surge_idx, time_to_next = state
        surge_idx = int(surge_idx)

        if F_collapse is not None and fatigue >= F_collapse:
            return -0.5

        if alive < 0.5:
            return 0.0

        if action is not None:
            a_t = action / (n_actions - 1)
        else:
            a_t = 0.5

        reward = a_t

        # Check if at surge time
        if surge_idx < len(surge_times) and time_to_next < 1.5 and time_to_next > 0.5:
            if fatigue <= F_req:
                reward += surge_bonuses[surge_idx]

        return reward

    return reward_fn


def depth_limited_mcts_search(
    root_state: np.ndarray,
    world_model,
    n_actions: int,
    num_simulations: int = 100,
    max_depth: int = 10,
    c: float = 1.4,
    gamma: float = 0.99,
    reward_fn=None,
    policy_net=None,
    value_net=None,
) -> int:
    """MCTS with explicit depth limit."""
    import math

    root_tensor = torch.tensor(root_state, dtype=torch.float32)
    root = MCTSNode(root_tensor, depth=0)

    for sim in range(num_simulations):
        node = root
        path = [node]

        while node.is_fully_expanded(n_actions) and node.children and node.depth < max_depth:
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        at_depth_limit = (node.depth >= max_depth)

        if not at_depth_limit and not node.is_fully_expanded(n_actions):
            if policy_net is not None:
                with torch.no_grad():
                    node_priors = F.softmax(policy_net(node.state.unsqueeze(0)), dim=-1)[0].numpy()
            else:
                node_priors = np.ones(n_actions) / n_actions

            unexpanded = [a for a in range(n_actions) if a not in node.children]
            states_batch = node.state.unsqueeze(0).expand(len(unexpanded), -1)
            actions_batch = torch.tensor(unexpanded, dtype=torch.int64)

            with torch.no_grad():
                next_states = world_model(states_batch, actions_batch)

            for i, action in enumerate(unexpanded):
                child_reward = reward_fn(next_states[i].detach().numpy(), action)
                child = MCTSNode(
                    next_states[i], parent=node, action=action,
                    prior=node_priors[action], reward=child_reward,
                    depth=node.depth + 1
                )
                node.children[action] = child

            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        if value_net is not None:
            with torch.no_grad():
                leaf_value = value_net(node.state.unsqueeze(0)).item()
        else:
            leaf_value = 0.0

        G = leaf_value
        for node in reversed(path):
            if node.parent is not None:
                G = node.reward + gamma * G
            node.N += 1
            node.W += G
            node.Q = node.W / node.N

    if not root.children:
        return 0

    visit_counts = np.zeros(n_actions)
    for action, child in root.children.items():
        visit_counts[action] = child.N

    return int(np.argmax(visit_counts))


def train_multi_surge(env, training_steps=15000):
    """Train world model and networks for multi-surge environment."""
    from src.planning import mcts_search

    state_dim = 4
    n_actions = env.n_actions

    world_model = MultiSurgeWorldModel(
        n_actions=n_actions,
        surge_times=env.surge_times,
        surge_bonuses=env.surge_bonuses,
        init_alpha=env.alpha,
        init_beta=env.beta,
        init_F=50.0,
        F_req=env.F_req,
    )

    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    model_opt = optim.Adam([
        {'params': [world_model._alpha_raw], 'lr': 1e-2},
        {'params': [world_model._beta_raw], 'lr': 5e-2},
        {'params': [world_model._F_raw], 'lr': 5e-2},
    ])

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_multi_surge_reward_fn(
        n_actions, env.F_req, env.surge_times, env.surge_bonuses
    )

    steps = 0
    episode = 0
    gamma = 0.99

    while steps < training_steps:
        state, _ = env.reset(seed=episode)
        episode_transitions = []

        for t in range(env.T):
            with torch.no_grad():
                if steps < 2000:
                    action = env.action_space.sample()
                elif np.random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    action = depth_limited_mcts_search(
                        state, world_model, n_actions=n_actions,
                        num_simulations=20, max_depth=10,
                        policy_net=policy_net, value_net=value_net,
                        reward_fn=reward_fn,
                    )

            next_state, reward, terminated, truncated, info = env.step(action)
            steps += 1

            transition = Transition(state.copy(), action, reward, next_state.copy(), terminated or truncated)
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated:
                break

        # Compute returns
        returns = []
        G = 0.0
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        # Training updates
        for _ in range(5):
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                batch = [t for t in batch if t.state[1] > 0.5]
                if len(batch) < 16:
                    continue

                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)

                pred_next = world_model(states, actions)
                model_loss = F.mse_loss(pred_next, next_states)

                # Learn collapse threshold from collapse events
                alive_current = states[:, 1]
                alive_next = next_states[:, 1]
                fatigue_next = next_states[:, 0]
                collapse_mask = (alive_current > 0.5) & (alive_next < 0.5)

                if collapse_mask.any():
                    collapse_fatigue = fatigue_next[collapse_mask]
                    F_learned = world_model.collapse_threshold.float()
                    collapse_loss = F.mse_loss(
                        F_learned.expand(collapse_fatigue.shape[0]),
                        collapse_fatigue
                    )
                    model_loss = model_loss + 0.5 * collapse_loss

                model_opt.zero_grad()
                model_loss.backward()
                model_opt.step()

        # Value network training
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(np.array([t.state for t in episode_transitions]), dtype=torch.float32)
            ep_returns = torch.tensor(returns, dtype=torch.float32)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode += 1

        if steps % 3000 == 0 or steps >= training_steps:
            F_learned = world_model.collapse_threshold.item()
            print(f"  Step {steps}: F={F_learned:.2f}, alpha={world_model.alpha.item():.3f}, beta={world_model.beta.item():.3f}")

    return world_model, policy_net, value_net


def main():
    # Multi-surge configuration - irregular spacing
    alpha = 1.0
    beta = 0.08  # Slightly slower recovery
    true_F = 8.0
    F_req = 4.5  # Moderately strict
    surge_times = [4, 8, 14]  # Irregular: gaps of 4, 4, 6
    surge_bonuses = [8.0, 12.0, 18.0]  # Last surge most valuable
    T = 18
    training_steps = 25000
    h_values = [1, 2, 3, 4, 5, 6, 8, 10]
    num_seeds = 50
    num_simulations = 30

    print("=" * 60)
    print("Multi-Surge Horizon Ablation Experiment")
    print("=" * 60)
    print(f"Surge times: {surge_times} (bonuses: {surge_bonuses})")
    print(f"F_req: {F_req}, T: {T}")
    print(f"H values: {h_values}")
    print()

    # Train
    print("Training...")
    set_seed(42)
    train_env = MultiSurgeEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        surge_times=surge_times, surge_bonuses=surge_bonuses, T=T, seed=42
    )
    world_model, policy_net, value_net = train_multi_surge(train_env, training_steps)

    print(f"\nLearned: F={world_model.collapse_threshold.item():.2f}, "
          f"alpha={world_model.alpha.item():.3f}, beta={world_model.beta.item():.3f}")
    print()

    reward_fn = make_multi_surge_reward_fn(
        train_env.n_actions, F_req, surge_times, surge_bonuses,
        F_collapse=world_model.collapse_threshold.item()
    )

    # Test
    for H in h_values:
        print(f"Testing H={H}...")
        sd_surges = []
        zero_surges = []

        for seed in range(num_seeds):
            # SD-MCTS
            eval_env = MultiSurgeEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                surge_times=surge_times, surge_bonuses=surge_bonuses, T=T, seed=seed+1000
            )
            state, _ = eval_env.reset(seed=seed+1000)
            surges_captured = 0
            for step in range(T):
                with torch.no_grad():
                    action = depth_limited_mcts_search(
                        state, world_model, n_actions=train_env.n_actions,
                        num_simulations=num_simulations, max_depth=H,
                        policy_net=policy_net, value_net=value_net, reward_fn=reward_fn,
                    )
                next_state, _, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_captured"):
                    surges_captured += 1
                if info.get("collapsed"):
                    break
                state = next_state
                if terminated or truncated:
                    break
            sd_surges.append(surges_captured)

            # MCTS-Zero
            eval_env = MultiSurgeEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                surge_times=surge_times, surge_bonuses=surge_bonuses, T=T, seed=seed+1000
            )
            state, _ = eval_env.reset(seed=seed+1000)
            surges_captured = 0
            for step in range(T):
                with torch.no_grad():
                    action = depth_limited_mcts_search(
                        state, world_model, n_actions=train_env.n_actions,
                        num_simulations=num_simulations, max_depth=H,
                        policy_net=policy_net, value_net=None, reward_fn=reward_fn,
                    )
                next_state, _, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_captured"):
                    surges_captured += 1
                if info.get("collapsed"):
                    break
                state = next_state
                if terminated or truncated:
                    break
            zero_surges.append(surges_captured)

        sd_mean = np.mean(sd_surges)
        zero_mean = np.mean(zero_surges)
        sd_all3 = sum(1 for s in sd_surges if s == 3) / num_seeds
        zero_all3 = sum(1 for s in zero_surges if s == 3) / num_seeds

        print(f"  SD-MCTS: avg={sd_mean:.2f}/3 surges, all_3={sd_all3:.0%}")
        print(f"  MCTS-Zero: avg={zero_mean:.2f}/3 surges, all_3={zero_all3:.0%}")


if __name__ == "__main__":
    main()
