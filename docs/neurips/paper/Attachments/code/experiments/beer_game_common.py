from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch

from src.agent.networks import ValueNetwork
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

    def __len__(self):
        return len(self.buffer)


def make_beer_game_reward_fn(h: float = 0.5, p: float = 1.0, n_echelons: int = 4) -> Callable:
    """Create reward function compatible with the MCTS interface."""

    def reward_fn(state: np.ndarray, action: int = None) -> float:
        inventory = state[0::6][:n_echelons]
        backlog = state[1::6][:n_echelons]
        holding_cost = h * np.sum(np.maximum(0, inventory))
        backlog_cost = p * np.sum(backlog)
        return -(holding_cost + backlog_cost)

    return reward_fn


class BeerGameSDModelWrapper(torch.nn.Module):
    """Adapter that makes BeerGameSDModel compatible with mcts_search."""

    def __init__(self, model: BeerGameSDModel):
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        params = list(self.model.parameters())
        device = params[0].device if params else torch.device("cpu")
        return self.model(states.to(device), actions.to(device))

    def parameters(self):
        return self.model.parameters()


class NormalizedValueNet(torch.nn.Module):
    """Denormalize value-network output before using it inside MCTS."""

    def __init__(self, value_net: ValueNetwork, get_stats_fn):
        super().__init__()
        self.value_net = value_net
        self.get_stats_fn = get_stats_fn

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized_value = self.value_net(states)
        mean, std = self.get_stats_fn()
        return normalized_value * std + mean

    def parameters(self):
        return self.value_net.parameters()
