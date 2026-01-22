import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率分布"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作logits"""
        return self.net(state)

    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作概率分布"""
        return F.softmax(self.forward(state), dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """采样或选择动作"""
        probs = self.get_probs(state)
        if deterministic:
            return probs.argmax(dim=-1).item()
        return torch.multinomial(probs, 1).item()


class ValueNetwork(nn.Module):
    """价值网络：输入状态，输出状态价值"""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回状态价值"""
        return self.net(state).squeeze(-1)
