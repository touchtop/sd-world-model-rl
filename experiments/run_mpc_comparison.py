#!/usr/bin/env python3
"""Experiment: SD-MCTS vs Oracle MPC (Theorem 2 & Corollary 2 Validation).

Phase 1: Validates effective horizon extension (Corollary 2)
- SD-MCTS (learned model + V_ψ) vs MPC (learned model + V=0)
- Same model, only terminal value differs

Phase 2: Validates "imperfect but far-sighted" > "perfect but short-sighted" (Theorem 3)
- SD-MCTS (learned model + V_ψ) vs Oracle MPC (true dynamics + V=0)
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.delayed_collapse_env import DelayedCollapseEnv
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.agent import PolicyNetwork, ValueNetwork


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# Oracle MPC Implementation
# ============================================================================

class OracleDynamics:
    """Wrapper for true environment dynamics."""

    def __init__(self, alpha: float, beta: float, F: float, n_actions: int):
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.n_actions = n_actions

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Predict next state and reward using true dynamics."""
        fatigue, alive, surge_flag, time_to_surge = state

        if alive < 0.5:
            new_state = np.array([fatigue, 0.0, 0.0, max(0, time_to_surge - 1)], dtype=np.float32)
            return new_state, 0.0

        a_t = action / (self.n_actions - 1)

        # True fatigue dynamics
        new_fatigue = fatigue + (self.alpha + self.beta) * a_t - self.beta
        new_fatigue = max(0.0, new_fatigue)

        # Collapse check
        if new_fatigue >= self.F:
            new_state = np.array([new_fatigue, 0.0, 0.0, max(0, time_to_surge - 1)], dtype=np.float32)
            return new_state, 0.0

        # Update time to surge
        new_time_to_surge = max(0.0, time_to_surge - 1)
        new_surge_flag = 1.0 if new_time_to_surge == 0 else 0.0

        new_state = np.array([new_fatigue, 1.0, new_surge_flag, new_time_to_surge], dtype=np.float32)

        # Reward (base)
        reward = a_t

        return new_state, reward


def mpc_search(
    state: np.ndarray,
    dynamics,  # OracleDynamics or learned world model
    n_actions: int,
    horizon: int,
    gamma: float = 0.99,
    F_req: float = 5.0,
    B: float = 15.0,
    use_learned_model: bool = False,
    num_samples: int = 500,  # Number of random trajectories to sample
) -> int:
    """MPC with H-step lookahead and V_terminal = 0.

    Uses Random Shooting: sample random action sequences, evaluate, pick best.
    Complexity: O(num_samples * H) instead of O(n_actions^H)
    """
    if use_learned_model:
        return _mpc_random_shooting_learned(
            state, dynamics, n_actions, horizon, gamma, F_req, B, num_samples
        )
    else:
        return _mpc_random_shooting(
            state, dynamics, n_actions, horizon, gamma, F_req, B, num_samples
        )


def _mpc_random_shooting(
    state: np.ndarray,
    dynamics: 'OracleDynamics',
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
    num_samples: int,
) -> int:
    """Random Shooting MPC with oracle dynamics."""
    best_value = -float('inf')
    best_first_action = 0

    for _ in range(num_samples):
        # Sample random action sequence
        action_seq = np.random.randint(0, n_actions, size=horizon)

        # Rollout and compute return
        s = state.copy()
        total_return = 0.0

        for t, a in enumerate(action_seq):
            if s[1] < 0.5:  # Dead
                break

            next_s, reward = dynamics.predict(s, a)

            # Check surge bonus
            time_to_surge = s[3]
            if time_to_surge < 1.5 and time_to_surge > 0.5:
                if next_s[0] <= F_req:
                    reward += B

            total_return += (gamma ** t) * reward
            s = next_s

        if total_return > best_value:
            best_value = total_return
            best_first_action = action_seq[0]

    return best_first_action


def _mpc_random_shooting_learned(
    state: np.ndarray,
    world_model,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
    num_samples: int,
) -> int:
    """Random Shooting MPC with learned world model."""
    best_value = -float('inf')
    best_first_action = 0

    state_tensor = torch.tensor(state, dtype=torch.float32)

    for _ in range(num_samples):
        action_seq = np.random.randint(0, n_actions, size=horizon)

        s = state_tensor.clone()
        total_return = 0.0

        for t, a in enumerate(action_seq):
            if s[1].item() < 0.5:
                break

            with torch.no_grad():
                action_tensor = torch.tensor([a], dtype=torch.int64)
                next_s = world_model(s.unsqueeze(0), action_tensor)[0]

            a_t = a / (n_actions - 1)
            reward = a_t

            time_to_surge = s[3].item()
            if time_to_surge < 1.5 and time_to_surge > 0.5:
                if next_s[0].item() <= F_req:
                    reward += B

            if next_s[1].item() < 0.5:
                reward = 0.0

            total_return += (gamma ** t) * reward
            s = next_s

        if total_return > best_value:
            best_value = total_return
            best_first_action = action_seq[0]

    return best_first_action


def _mpc_exact_dp(
    state: np.ndarray,
    dynamics,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
) -> int:
    """Exact DP for MPC (exponential complexity, use only for small H)."""

    def compute_value(s: np.ndarray, h: int) -> Tuple[float, int]:
        if h == 0:
            return 0.0, 0  # V_terminal = 0

        fatigue, alive, surge_flag, time_to_surge = s

        if alive < 0.5:
            return 0.0, 0

        best_value = -float('inf')
        best_action = 0

        for a in range(n_actions):
            next_state, reward = dynamics.predict(s, a)

            # Add surge bonus if applicable
            if time_to_surge < 1.5 and time_to_surge > 0.5:
                next_fatigue = next_state[0]
                if next_fatigue <= F_req:
                    reward += B

            future_value, _ = compute_value(next_state, h - 1)
            total_value = reward + gamma * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = a

        return best_value, best_action

    _, best_action = compute_value(state, horizon)
    return best_action


def _mpc_exact_dp_reduced(
    state: np.ndarray,
    dynamics,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
) -> int:
    """Exact DP with reduced action set for large H."""
    # Use subset of actions: {0, 3, 5, 7, 10} = 5 actions
    # This reduces complexity from 11^H to 5^H
    action_subset = [0, 3, 5, 7, 10]

    def compute_value(s: np.ndarray, h: int) -> Tuple[float, int]:
        if h == 0:
            return 0.0, 0

        fatigue, alive, surge_flag, time_to_surge = s

        if alive < 0.5:
            return 0.0, 0

        best_value = -float('inf')
        best_action = 0

        for a in action_subset:
            next_state, reward = dynamics.predict(s, a)

            if time_to_surge < 1.5 and time_to_surge > 0.5:
                next_fatigue = next_state[0]
                if next_fatigue <= F_req:
                    reward += B

            future_value, _ = compute_value(next_state, h - 1)
            total_value = reward + gamma * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = a

        return best_value, best_action

    _, best_action = compute_value(state, horizon)
    return best_action


def _mpc_beam_search(
    state: np.ndarray,
    dynamics,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
    beam_width: int,
) -> int:
    """Beam search MPC for larger horizons."""
    # Each beam entry: (cumulative_value, state, first_action, depth)
    initial_beams = []

    # Expand first level to get first actions
    for a in range(n_actions):
        next_state, reward = dynamics.predict(state, a)

        # Check surge bonus at first step
        time_to_surge = state[3]
        if time_to_surge < 1.5 and time_to_surge > 0.5:
            if next_state[0] <= F_req:
                reward += B

        if next_state[1] > 0.5:  # Still alive
            initial_beams.append((reward, next_state, a, 1))

    if not initial_beams:
        return 0

    # Sort and keep top beam_width
    beams = sorted(initial_beams, key=lambda x: -x[0])[:beam_width * 2]

    # Expand remaining levels
    for depth in range(1, horizon):
        new_beams = []
        for cum_value, s, first_action, d in beams:
            if s[1] < 0.5:  # Dead
                new_beams.append((cum_value, s, first_action, d))
                continue

            for a in range(n_actions):
                next_state, reward = dynamics.predict(s, a)

                # Check surge bonus
                time_to_surge = s[3]
                if time_to_surge < 1.5 and time_to_surge > 0.5:
                    if next_state[0] <= F_req:
                        reward += B

                new_value = cum_value + (gamma ** depth) * reward
                new_beams.append((new_value, next_state, first_action, depth + 1))

        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]

    if not beams:
        return 0

    # Return first action of best beam
    best_beam = max(beams, key=lambda x: x[0])
    return best_beam[2]


def _mpc_with_learned_model(
    state: np.ndarray,
    world_model: DelayedCollapseSDModel,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
    beam_width: int = 3,
) -> int:
    """MPC using learned world model with V_terminal = 0."""
    if horizon <= 8:
        return _mpc_learned_exact(state, world_model, n_actions, horizon, gamma, F_req, B)
    else:
        return _mpc_learned_exact_reduced(state, world_model, n_actions, horizon, gamma, F_req, B)


def _mpc_learned_exact(
    state: np.ndarray,
    world_model: DelayedCollapseSDModel,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
) -> int:
    """Exact DP MPC with learned model."""

    def compute_value(s: torch.Tensor, h: int) -> Tuple[float, int]:
        if h == 0:
            return 0.0, 0

        if s[1].item() < 0.5:
            return 0.0, 0

        best_value = -float('inf')
        best_action = 0

        for a in range(n_actions):
            with torch.no_grad():
                action_tensor = torch.tensor([a], dtype=torch.int64)
                next_state = world_model(s.unsqueeze(0), action_tensor)[0]

            a_t = a / (n_actions - 1)
            reward = a_t

            time_to_surge = s[3].item()
            if time_to_surge < 1.5 and time_to_surge > 0.5:
                if next_state[0].item() <= F_req:
                    reward += B

            if next_state[1].item() < 0.5:
                reward = 0.0

            future_value, _ = compute_value(next_state, h - 1)
            total_value = reward + gamma * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = a

        return best_value, best_action

    state_tensor = torch.tensor(state, dtype=torch.float32)
    _, best_action = compute_value(state_tensor, horizon)
    return best_action


def _mpc_learned_exact_reduced(
    state: np.ndarray,
    world_model: DelayedCollapseSDModel,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
) -> int:
    """Exact DP MPC with learned model and reduced action set."""
    action_subset = [0, 3, 5, 7, 10]

    def compute_value(s: torch.Tensor, h: int) -> Tuple[float, int]:
        if h == 0:
            return 0.0, 0

        if s[1].item() < 0.5:
            return 0.0, 0

        best_value = -float('inf')
        best_action = 0

        for a in action_subset:
            with torch.no_grad():
                action_tensor = torch.tensor([a], dtype=torch.int64)
                next_state = world_model(s.unsqueeze(0), action_tensor)[0]

            a_t = a / (n_actions - 1)
            reward = a_t

            time_to_surge = s[3].item()
            if time_to_surge < 1.5 and time_to_surge > 0.5:
                if next_state[0].item() <= F_req:
                    reward += B

            if next_state[1].item() < 0.5:
                reward = 0.0

            future_value, _ = compute_value(next_state, h - 1)
            total_value = reward + gamma * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = a

        return best_value, best_action

    state_tensor = torch.tensor(state, dtype=torch.float32)
    _, best_action = compute_value(state_tensor, horizon)
    return best_action


def _mpc_learned_beam(
    state: np.ndarray,
    world_model: DelayedCollapseSDModel,
    n_actions: int,
    horizon: int,
    gamma: float,
    F_req: float,
    B: float,
    beam_width: int,
) -> int:
    """Beam search MPC with learned model."""
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Each beam: (cumulative_value, state_tensor, first_action)
    initial_beams = []

    for a in range(n_actions):
        with torch.no_grad():
            action_tensor = torch.tensor([a], dtype=torch.int64)
            next_state = world_model(state_tensor.unsqueeze(0), action_tensor)[0]

        a_t = a / (n_actions - 1)
        reward = a_t

        time_to_surge = state_tensor[3].item()
        if time_to_surge < 1.5 and time_to_surge > 0.5:
            if next_state[0].item() <= F_req:
                reward += B

        if next_state[1].item() < 0.5:
            reward = 0.0

        initial_beams.append((reward, next_state, a))

    beams = sorted(initial_beams, key=lambda x: -x[0])[:beam_width * 2]

    for depth in range(1, horizon):
        new_beams = []
        for cum_value, s, first_action in beams:
            if s[1].item() < 0.5:
                new_beams.append((cum_value, s, first_action))
                continue

            for a in range(n_actions):
                with torch.no_grad():
                    action_tensor = torch.tensor([a], dtype=torch.int64)
                    next_state = world_model(s.unsqueeze(0), action_tensor)[0]

                a_t = a / (n_actions - 1)
                reward = a_t

                time_to_surge = s[3].item()
                if time_to_surge < 1.5 and time_to_surge > 0.5:
                    if next_state[0].item() <= F_req:
                        reward += B

                if next_state[1].item() < 0.5:
                    reward = 0.0

                new_value = cum_value + (gamma ** depth) * reward
                new_beams.append((new_value, next_state, first_action))

        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]

    if not beams:
        return 0

    best_beam = max(beams, key=lambda x: x[0])
    return best_beam[2]


# ============================================================================
# SD-MCTS Implementation (with value network)
# ============================================================================

class MCTSNode:
    __slots__ = ['state', 'parent', 'action', 'children', 'N', 'W', 'Q', 'P', 'reward', 'depth']

    def __init__(self, state: torch.Tensor, parent=None, action: int = None,
                 prior: float = 0.0, reward: float = 0.0, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior
        self.reward = reward
        self.depth = depth

    def is_fully_expanded(self, n_actions: int) -> bool:
        return len(self.children) == n_actions


def _puct_score(parent: MCTSNode, child: MCTSNode, c: float) -> float:
    import math
    exploration = c * child.P * math.sqrt(parent.N + 1) / (1 + child.N)
    return child.Q + exploration


def make_reward_fn(n_actions: int, F_req: float, B: float, F_collapse: float = None):
    def reward_fn(state: np.ndarray, action: int = None) -> float:
        fatigue, alive, surge_flag, time_to_surge = state
        if F_collapse is not None and fatigue >= F_collapse:
            return -0.5
        if alive < 0.5:
            return 0.0
        if action is not None:
            a_t = action / (n_actions - 1)
        else:
            a_t = 0.5
        reward = a_t
        if time_to_surge < 1.5 and time_to_surge > 0.5 and fatigue <= F_req:
            reward += B
        return reward
    return reward_fn


def sd_mcts_search(
    root_state: np.ndarray,
    world_model: DelayedCollapseSDModel,
    n_actions: int,
    num_simulations: int,
    max_depth: int,
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    reward_fn: Callable,
    c: float = 1.4,
    gamma: float = 0.99,
) -> int:
    """SD-MCTS with value network for leaf evaluation."""
    root_tensor = torch.tensor(root_state, dtype=torch.float32)
    root = MCTSNode(root_tensor, depth=0)

    for _ in range(num_simulations):
        node = root
        path = [node]

        # Selection
        while node.is_fully_expanded(n_actions) and node.children and node.depth < max_depth:
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        at_depth_limit = (node.depth >= max_depth)

        # Expansion
        if not at_depth_limit and not node.is_fully_expanded(n_actions):
            with torch.no_grad():
                priors = F.softmax(policy_net(node.state.unsqueeze(0)), dim=-1)[0].numpy()

            unexpanded = [a for a in range(n_actions) if a not in node.children]
            states_batch = node.state.unsqueeze(0).expand(len(unexpanded), -1)
            actions_batch = torch.tensor(unexpanded, dtype=torch.int64)

            with torch.no_grad():
                next_states = world_model(states_batch, actions_batch)

            for i, action in enumerate(unexpanded):
                child_reward = reward_fn(next_states[i].numpy(), action)
                child = MCTSNode(
                    next_states[i], parent=node, action=action,
                    prior=priors[action], reward=child_reward,
                    depth=node.depth + 1
                )
                node.children[action] = child

            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Leaf evaluation with value network
        with torch.no_grad():
            leaf_value = value_net(node.state.unsqueeze(0)).item()

        # Backpropagation
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


# ============================================================================
# Training
# ============================================================================

from dataclasses import dataclass
from typing import List

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


def train_sd_mcts(
    env: DelayedCollapseEnv,
    true_F: float,
    init_F_belief: float = 50.0,
    env_steps: int = 15000,
) -> Tuple[DelayedCollapseSDModel, PolicyNetwork, ValueNetwork]:
    """Train SD-MCTS world model, policy network, and value network."""
    from src.planning import mcts_search

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = DelayedCollapseSDModel(
        n_actions=n_actions,
        init_alpha=env.alpha,
        init_beta=env.beta,
        init_F=init_F_belief,
        F_req=env.F_req,
        T_s=env.T_s,
        B=env.B,
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
    reward_fn = make_reward_fn(n_actions, env.F_req, env.B)

    steps = 0
    episode = 0
    gamma = 0.99

    while steps < env_steps:
        state, _ = env.reset(seed=episode)
        episode_transitions = []

        for t in range(env.T):
            with torch.no_grad():
                if steps < 2000:
                    action = env.action_space.sample()
                elif np.random.random() < 0.2:
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

                alive_current = states[:, 1]
                alive_next = next_states[:, 1]
                fatigue_next = next_states[:, 0]
                collapse_mask = (alive_current > 0.5) & (alive_next < 0.5)

                if collapse_mask.any():
                    collapse_fatigue = fatigue_next[collapse_mask]
                    F_learned = world_model.collapse_threshold
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

        # Policy network training
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(np.array([t.state for t in episode_transitions]), dtype=torch.float32)
            ep_actions = torch.tensor([t.action for t in episode_transitions], dtype=torch.int64)
            ep_returns = torch.tensor(returns, dtype=torch.float32)

            logits = policy_net(ep_states)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, ep_actions.unsqueeze(1)).squeeze(1)

            advantages = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-8)
            policy_loss = -(selected_log_probs * advantages.detach()).mean()

            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

        episode += 1

        if steps % 3000 == 0 or steps >= env_steps:
            F_learned = world_model.collapse_threshold.item()
            print(f"  Step {steps}: F_learned={F_learned:.2f} (true={true_F})")

    return world_model, policy_net, value_net


# ============================================================================
# Main Experiment
# ============================================================================

import time

def main():
    parser = argparse.ArgumentParser(description="SD-MCTS vs Oracle MPC")
    parser.add_argument("--h-values", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12, 14])
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--training-steps", type=int, default=20000)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--mpc-samples", type=int, default=1000, help="Number of samples for MPC random shooting")
    args = parser.parse_args()

    # Environment config
    true_F = 10.0
    belief_F = 50.0
    alpha = 1.0
    beta = 0.1
    F_req = 5.0
    T_s = 12
    B = 15.0
    T = 30
    n_actions = 11

    print("=" * 70)
    print("Experiment: SD-MCTS vs Oracle MPC (Corollary 2 & Theorem 3)")
    print("=" * 70)
    print(f"H values: {args.h_values}")
    print(f"T_s (surge time): {T_s}")
    print(f"True F (collapse threshold): {true_F}")
    print(f"MPC samples: {args.mpc_samples}")
    print()

    # Train SD-MCTS
    print("Training SD-MCTS...")
    set_seed(42)
    train_env = DelayedCollapseEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        T_s=T_s, B=B, T=T, seed=42
    )
    world_model, policy_net, value_net = train_sd_mcts(
        train_env, true_F=true_F, init_F_belief=belief_F,
        env_steps=args.training_steps
    )

    F_learned = world_model.collapse_threshold.item()
    print(f"\nLearned F: {F_learned:.2f} (true: {true_F})")
    print()

    # Create Oracle dynamics
    oracle_dynamics = OracleDynamics(alpha=alpha, beta=beta, F=true_F, n_actions=n_actions)

    reward_fn = make_reward_fn(n_actions, F_req, B, F_collapse=F_learned)

    print("=" * 70)
    print("Phase 1: Corollary 2 - Effective Horizon Extension")
    print("SD-MCTS (learned model + V_ψ) vs MPC (learned model + V=0)")
    print("=" * 70)
    print(f"{'H':>3} | {'SD-MCTS':>12} | {'MPC(V=0)':>12} | {'SD Time':>10} | {'MPC Time':>10}")
    print("-" * 60)

    for H in args.h_values:
        sd_successes = 0
        mpc_learned_successes = 0
        sd_total_time = 0.0
        mpc_total_time = 0.0

        for seed in range(args.num_seeds):
            eval_env = DelayedCollapseEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                T_s=T_s, B=B, T=15, seed=seed + 1000
            )

            # SD-MCTS evaluation
            state, _ = eval_env.reset(seed=seed + 1000)
            sd_surge = False
            sd_collapsed = False

            t_start = time.perf_counter()
            for step in range(15):
                action = sd_mcts_search(
                    state, world_model, n_actions,
                    num_simulations=args.num_simulations,
                    max_depth=H,
                    policy_net=policy_net,
                    value_net=value_net,
                    reward_fn=reward_fn,
                )
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_bonus_earned"):
                    sd_surge = True
                if info.get("collapsed"):
                    sd_collapsed = True
                    break
                state = next_state
                if terminated or truncated:
                    break
            sd_total_time += time.perf_counter() - t_start

            if sd_surge and not sd_collapsed:
                sd_successes += 1

            # MPC with learned model evaluation
            eval_env = DelayedCollapseEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                T_s=T_s, B=B, T=15, seed=seed + 1000
            )
            state, _ = eval_env.reset(seed=seed + 1000)
            mpc_surge = False
            mpc_collapsed = False

            t_start = time.perf_counter()
            for step in range(15):
                action = mpc_search(
                    state, world_model, n_actions,
                    horizon=H, gamma=0.99, F_req=F_req, B=B,
                    use_learned_model=True, num_samples=args.mpc_samples,
                )
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_bonus_earned"):
                    mpc_surge = True
                if info.get("collapsed"):
                    mpc_collapsed = True
                    break
                state = next_state
                if terminated or truncated:
                    break
            mpc_total_time += time.perf_counter() - t_start

            if mpc_surge and not mpc_collapsed:
                mpc_learned_successes += 1

        sd_rate = sd_successes / args.num_seeds * 100
        mpc_rate = mpc_learned_successes / args.num_seeds * 100
        sd_avg_time = sd_total_time / args.num_seeds * 1000  # ms
        mpc_avg_time = mpc_total_time / args.num_seeds * 1000  # ms
        print(f"{H:>3} | {sd_rate:>10.0f}% | {mpc_rate:>10.0f}% | {sd_avg_time:>8.1f}ms | {mpc_avg_time:>8.1f}ms")

    print()
    print("=" * 70)
    print("Phase 2: Theorem 3 - 'Imperfect but Far-sighted' > 'Perfect but Short-sighted'")
    print("SD-MCTS (learned model + V_ψ) vs Oracle MPC (true dynamics + V=0)")
    print("=" * 70)
    print(f"{'H':>3} | {'SD-MCTS':>12} | {'Oracle MPC':>12} | {'SD Time':>10} | {'MPC Time':>10}")
    print("-" * 60)

    for H in args.h_values:
        sd_successes = 0
        oracle_mpc_successes = 0
        sd_total_time = 0.0
        mpc_total_time = 0.0

        for seed in range(args.num_seeds):
            eval_env = DelayedCollapseEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                T_s=T_s, B=B, T=15, seed=seed + 1000
            )

            # SD-MCTS evaluation
            state, _ = eval_env.reset(seed=seed + 1000)
            sd_surge = False
            sd_collapsed = False

            t_start = time.perf_counter()
            for step in range(15):
                action = sd_mcts_search(
                    state, world_model, n_actions,
                    num_simulations=args.num_simulations,
                    max_depth=H,
                    policy_net=policy_net,
                    value_net=value_net,
                    reward_fn=reward_fn,
                )
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_bonus_earned"):
                    sd_surge = True
                if info.get("collapsed"):
                    sd_collapsed = True
                    break
                state = next_state
                if terminated or truncated:
                    break
            sd_total_time += time.perf_counter() - t_start

            if sd_surge and not sd_collapsed:
                sd_successes += 1

            # Oracle MPC evaluation
            eval_env = DelayedCollapseEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                T_s=T_s, B=B, T=15, seed=seed + 1000
            )
            state, _ = eval_env.reset(seed=seed + 1000)
            oracle_surge = False
            oracle_collapsed = False

            t_start = time.perf_counter()
            for step in range(15):
                action = mpc_search(
                    state, oracle_dynamics, n_actions,
                    horizon=H, gamma=0.99, F_req=F_req, B=B,
                    use_learned_model=False, num_samples=args.mpc_samples,
                )
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_bonus_earned"):
                    oracle_surge = True
                if info.get("collapsed"):
                    oracle_collapsed = True
                    break
                state = next_state
                if terminated or truncated:
                    break
            mpc_total_time += time.perf_counter() - t_start

            if oracle_surge and not oracle_collapsed:
                oracle_mpc_successes += 1

        sd_rate = sd_successes / args.num_seeds * 100
        oracle_rate = oracle_mpc_successes / args.num_seeds * 100
        sd_avg_time = sd_total_time / args.num_seeds * 1000
        mpc_avg_time = mpc_total_time / args.num_seeds * 1000
        print(f"{H:>3} | {sd_rate:>10.0f}% | {oracle_rate:>10.0f}% | {sd_avg_time:>8.1f}ms | {mpc_avg_time:>8.1f}ms")

    print()
    print("=" * 70)
    print("Experiment Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
