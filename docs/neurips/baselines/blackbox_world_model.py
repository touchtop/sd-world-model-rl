"""
BlackBox World Model: MLP-based dynamics for ablation study.

This serves as the key baseline to isolate the contribution of SD structure.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class BlackBoxWorldModel(nn.Module):
    """MLP-based world model without structural priors."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        in_dim = state_dim + action_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state."""
        if action.dim() == 1:
            action_onehot = torch.zeros(state.shape[0], self.action_dim, device=state.device)
            action_onehot.scatter_(1, action.unsqueeze(1), 1)
        else:
            action_onehot = action

        x = torch.cat([state, action_onehot], dim=-1)
        delta = self.mlp(x)
        return state + delta  # Residual prediction

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """NumPy interface for MCTS."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            a = torch.LongTensor([action])
            next_s = self.forward(s, a)
            return next_s.squeeze(0).numpy()

    def get_violation_rate(self, states: np.ndarray) -> float:
        """
        Calculate physics violation rate.
        BlackBox model has no constraints, so we check for negative values
        in stock variables (indices 0, 1, 2 typically).
        """
        violations = np.sum(states[:, :3] < -1e-6)  # Allow small numerical errors
        total = states.shape[0] * 3
        return violations / total if total > 0 else 0.0


class BlackBoxMCTS:
    """MCTS using BlackBox world model instead of SD structure."""

    def __init__(
        self,
        world_model: BlackBoxWorldModel,
        value_net: nn.Module,
        action_dim: int,
        n_simulations: int = 50,
        c_puct: float = 1.0,
        gamma: float = 0.99,
        max_depth: int = 10,
    ):
        self.world_model = world_model
        self.value_net = value_net
        self.action_dim = action_dim
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.max_depth = max_depth

    def search(self, state: np.ndarray) -> int:
        """Run MCTS and return best action."""
        # Placeholder - will implement full MCTS logic
        # For now, delegates to the same MCTS implementation as SD-MCTS
        # but uses BlackBox world model for simulation
        raise NotImplementedError("Full implementation pending")


def train_blackbox_model(
    model: BlackBoxWorldModel,
    replay_buffer: list,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 100,
) -> dict:
    """Train BlackBox world model on collected transitions."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        np.random.shuffle(replay_buffer)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(replay_buffer) - batch_size + 1, batch_size):
            batch = replay_buffer[i:i+batch_size]
            states = torch.from_numpy(np.array([t[0] for t in batch], dtype=np.float32)).to(device)
            actions = torch.from_numpy(np.array([t[1] for t in batch], dtype=np.int64)).to(device)
            next_states = torch.from_numpy(np.array([t[2] for t in batch], dtype=np.float32)).to(device)

            optimizer.zero_grad()
            pred_next = model(states, actions)
            loss = criterion(pred_next, next_states)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            losses.append(epoch_loss / n_batches)

    return {"losses": losses}
