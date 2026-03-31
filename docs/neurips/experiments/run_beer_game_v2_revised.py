#!/usr/bin/env python
"""Revised Beer Game SD-MCTS runner for paper Experiment 1.

This revision keeps the original Beer Game training recipe, but aligns the
training/evaluation protocol with the later Cartpole workflow:

1. No policy prior is injected into MCTS unless it is explicitly trained.
2. The value network is trained and evaluated with the same normalized-return
   wrapper, so MCTS sees a consistent value interface throughout.

The original script is preserved as `run_beer_game_v2.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import ValueNetwork
from src.env.beer_game_env import BeerGameEnv
from src.env.beer_game_env import BeerGameEnvFlat
from src.planning import mcts_search
from src.world_model.beer_game_sd_model import BeerGameSDModel


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: List[Transition] = []
        self.capacity = capacity

    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(
            len(self.buffer),
            min(batch_size, len(self.buffer)),
            replace=False,
        )
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


def make_beer_game_reward_fn(
    h: float = 0.5,
    p: float = 1.0,
    n_echelons: int = 4,
) -> Callable:
    """Create reward function compatible with mcts_search signature."""

    def reward_fn(state: np.ndarray, action: int = None) -> float:
        inventory = state[0::6][:n_echelons]
        backlog = state[1::6][:n_echelons]
        holding_cost = h * np.sum(np.maximum(0, inventory))
        backlog_cost = p * np.sum(backlog)
        return -(holding_cost + backlog_cost)

    return reward_fn


class BeerGameSDModelWrapper(torch.nn.Module):
    """Make BeerGameSDModel compatible with mcts_search."""

    def __init__(self, model: BeerGameSDModel):
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        params = list(self.model.parameters())
        device = params[0].device if params else torch.device("cpu")
        return self.model(states.to(device), actions.to(device))

    def parameters(self):
        return self.model.parameters()


@dataclass
class ValueStats:
    mean: float = 0.0
    std: float = 1.0
    count: int = 0


class NormalizedValueNet(torch.nn.Module):
    """Expose value predictions on the original return scale for MCTS."""

    def __init__(self, value_net: ValueNetwork, stats: ValueStats):
        super().__init__()
        self.value_net = value_net
        self.stats = stats

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized_value = self.value_net(states)
        return normalized_value * self.stats.std + self.stats.mean

    def parameters(self):
        return self.value_net.parameters()


def _update_return_stats(stats: ValueStats, returns: List[float]) -> None:
    for ret in returns:
        stats.count += 1
        delta = ret - stats.mean
        stats.mean += delta / stats.count
        if stats.count > 1:
            stats.std = max(
                1.0,
                np.sqrt(
                    ((stats.count - 1) * stats.std ** 2 + delta * (ret - stats.mean))
                    / stats.count
                ),
            )


def compute_statistics(data: List[float]) -> dict:
    arr = np.array(data)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    median = np.median(arr)
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "median": median,
        "n": n,
        "raw": arr.tolist(),
    }


def train_sd_mcts(
    env: BeerGameEnvFlat,
    env_steps: int = 5000,
    gamma: float = 0.99,
    device: str = "cpu",
    verbose: bool = True,
    world_model_kwargs: Optional[dict] = None,
) -> Tuple[BeerGameSDModel, ValueNetwork, ValueStats, List[float]]:
    """Train Beer Game SD-MCTS with train/eval-consistent normalized values."""

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    wm_kwargs = world_model_kwargs or {}
    world_model = BeerGameSDModel(
        order_levels=env.order_levels,
        O_max=env.O_max,
        **wm_kwargs,
    ).to(device)
    world_model_wrapper = BeerGameSDModelWrapper(world_model)
    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)

    wm_params = list(world_model.parameters())
    model_opt = optim.Adam(wm_params, lr=1e-2) if wm_params else None
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    episode_rewards: List[float] = []
    steps = 0
    episode = 0
    value_stats = ValueStats()
    normalized_value_net = NormalizedValueNet(value_net, value_stats)

    random_steps = min(500, max(50, env_steps // 4))
    epsilon = 0.3
    num_sims = 200

    if verbose:
        print(f"Training revised SD-MCTS (steps={env_steps}, sims={num_sims})...")

    while steps < env_steps:
        state, _ = env.reset(seed=episode)
        episode_reward = 0.0
        episode_transitions: List[Transition] = []

        for _ in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model_wrapper,
                    n_actions=n_actions,
                    num_simulations=num_sims,
                    policy_net=None,
                    value_net=normalized_value_net,
                    reward_fn=reward_fn,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
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

        returns: List[float] = []
        G = 0.0
        for transition in reversed(episode_transitions):
            G = transition.reward + gamma * G
            returns.insert(0, G)

        _update_return_stats(value_stats, returns)

        for _ in range(5):
            if len(buffer) < 32:
                continue
            batch = buffer.sample(32)
            states = torch.tensor(
                np.array([transition.state for transition in batch]),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.tensor(
                [transition.action for transition in batch],
                dtype=torch.int64,
                device=device,
            )
            next_states = torch.tensor(
                np.array([transition.next_state for transition in batch]),
                dtype=torch.float32,
                device=device,
            )

            if model_opt is not None:
                pred_next = world_model(states, actions)
                model_loss = F.mse_loss(pred_next, next_states)
                model_opt.zero_grad()
                model_loss.backward()
                model_opt.step()

        if episode_transitions:
            ep_states = torch.tensor(
                np.array([transition.state for transition in episode_transitions]),
                dtype=torch.float32,
                device=device,
            )
            ep_returns = torch.tensor(
                [(ret - value_stats.mean) / value_stats.std for ret in returns],
                dtype=torch.float32,
                device=device,
            )

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode_rewards.append(episode_reward)
        episode += 1

        if verbose and episode % 20 == 0:
            avg_10 = (
                np.mean(episode_rewards[-10:])
                if len(episode_rewards) >= 10
                else episode_reward
            )
            print(
                f"  Step {steps}: Avg10={avg_10:.2f}, "
                f"value_mean={value_stats.mean:.2f}, value_std={value_stats.std:.2f}"
            )

    return world_model, value_net, value_stats, episode_rewards


def evaluate_sd_mcts(
    env: BeerGameEnv,
    world_model: BeerGameSDModel,
    value_net: ValueNetwork,
    value_stats: ValueStats,
    n_episodes: int = 10,
    device: str = "cpu",
    num_simulations: int = 200,
) -> dict:
    """Evaluate revised SD-MCTS using the same normalized wrapper as training."""

    del device
    world_model_wrapper = BeerGameSDModelWrapper(world_model)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)
    n_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else np.prod(env.action_space.nvec)
    )
    normalized_value_net = NormalizedValueNet(value_net, value_stats)
    total_costs = []
    bullwhip_ratios = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=1000 + ep)
        done = False

        while not done:
            if hasattr(env.action_space, "n"):
                action = mcts_search(
                    root_state=state,
                    world_model=world_model_wrapper,
                    n_actions=n_actions,
                    num_simulations=num_simulations,
                    policy_net=None,
                    value_net=normalized_value_net,
                    reward_fn=reward_fn,
                )
            else:
                flat_action = mcts_search(
                    root_state=state,
                    world_model=world_model_wrapper,
                    n_actions=n_actions,
                    num_simulations=num_simulations,
                    policy_net=None,
                    value_net=normalized_value_net,
                    reward_fn=reward_fn,
                )
                action = decode_flat_action(
                    flat_action,
                    env.order_levels,
                    env.n_echelons,
                )

            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        metrics = env.get_metrics()
        total_costs.append(metrics["total_cost"])
        bullwhip_ratios.append(metrics["bullwhip_ratios"])

    cost_stats = compute_statistics(total_costs)
    bullwhip_r = [bw[0] for bw in bullwhip_ratios]
    bullwhip_f = [bw[-1] for bw in bullwhip_ratios]

    return {
        "total_cost_mean": cost_stats["mean"],
        "total_cost_std": cost_stats["std"],
        "total_cost_se": cost_stats["se"],
        "total_cost_ci_95": (cost_stats["ci_95_low"], cost_stats["ci_95_high"]),
        "total_cost_median": cost_stats["median"],
        "total_costs": total_costs,
        "bullwhip_mean": np.mean(bullwhip_ratios, axis=0),
        "bullwhip_std": np.std(bullwhip_ratios, axis=0),
        "bullwhip_r_stats": compute_statistics(bullwhip_r),
        "bullwhip_f_stats": compute_statistics(bullwhip_f),
    }


def decode_flat_action(
    flat_action: int,
    order_levels: int,
    n_echelons: int,
) -> np.ndarray:
    """Decode flat action index to per-echelon orders."""

    actions = np.zeros(n_echelons, dtype=int)
    remaining = flat_action
    for i in range(n_echelons - 1, -1, -1):
        base = order_levels ** i
        actions[i] = remaining // base
        remaining = remaining % base
    return actions
