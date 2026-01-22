"""Learnable World Model for Delayed Collapse Environment.

SD-MCTS uses this model to learn the true fatigue dynamics,
including the real collapse threshold F and accumulation rate α.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DelayedCollapseSDModel(nn.Module):
    """Differentiable world model for delayed collapse dynamics.

    Learns:
    - α (alpha): fatigue accumulation rate
    - β (beta): recovery rate
    - F: collapse threshold

    State: [fatigue, alive, surge_flag, time_to_surge]
    Action: discrete intensity level (0 to n_actions-1)
    """

    def __init__(
        self,
        n_actions: int = 11,
        init_alpha: float = 1.0,
        init_beta: float = 0.1,
        init_F: float = 50.0,         # Start with wrong belief (like MPC)
        F_req: float = 5.0,           # Known requirement
        T_s: int = 12,                # Known surge time
        B: float = 15.0,              # Known bonus
    ):
        super().__init__()
        self.n_actions = n_actions
        self.F_req = F_req
        self.T_s = T_s
        self.B = B

        # Learnable parameters (use softplus to ensure positivity)
        # α: accumulation rate
        alpha_tensor = torch.tensor(float(init_alpha))
        self._alpha_raw = nn.Parameter(
            torch.log(torch.expm1(alpha_tensor).clamp_min(1e-6))
        )

        # β: recovery rate
        beta_tensor = torch.tensor(float(init_beta))
        self._beta_raw = nn.Parameter(
            torch.log(torch.expm1(beta_tensor).clamp_min(1e-6))
        )

        # F: collapse threshold (critical to learn correctly)
        F_tensor = torch.tensor(float(init_F))
        self._F_raw = nn.Parameter(
            torch.log(torch.expm1(F_tensor).clamp_min(1e-6))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw)

    @property
    def collapse_threshold(self) -> torch.Tensor:
        return F.softplus(self._F_raw)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state.

        Args:
            state: [batch, 4] - [fatigue, alive, surge_flag, time_to_surge]
            action: [batch] - discrete action indices

        Returns:
            next_state: [batch, 4]
        """
        fatigue = state[..., 0]
        alive = state[..., 1]
        time_to_surge = state[..., 3]

        # Convert action to intensity
        a_t = action.float() / (self.n_actions - 1)

        # Fatigue dynamics (differentiable)
        alpha = self.alpha
        beta = self.beta
        new_fatigue = fatigue + (alpha + beta) * a_t - beta
        new_fatigue = F.relu(new_fatigue)  # Cannot go negative

        # Collapse probability (soft threshold for differentiability)
        # Use sigmoid to create smooth transition around F
        collapse_prob = torch.sigmoid((new_fatigue - self.collapse_threshold) * 5.0)

        # Alive state: smooth transition
        new_alive = alive * (1.0 - collapse_prob)

        # Time to surge update
        new_time_to_surge = F.relu(time_to_surge - 1.0)

        # Surge flag
        new_surge_flag = torch.where(
            new_time_to_surge < 0.5,
            torch.ones_like(new_time_to_surge),
            torch.zeros_like(new_time_to_surge)
        )

        return torch.stack([
            new_fatigue,
            new_alive,
            new_surge_flag,
            new_time_to_surge
        ], dim=-1)

    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward for planning.

        This includes the surge bonus, which MPC cannot see in short horizon.
        """
        fatigue = state[..., 0]
        alive = state[..., 1]
        time_to_surge = state[..., 3]

        a_t = action.float() / (self.n_actions - 1)

        # Base reward (work intensity when alive)
        reward = a_t * alive

        # Surge bonus (when time_to_surge is about to hit 0)
        is_surge = (time_to_surge > 0.5) & (time_to_surge < 1.5)
        surge_eligible = fatigue <= self.F_req
        surge_bonus = torch.where(
            is_surge & surge_eligible,
            torch.full_like(fatigue, self.B),
            torch.zeros_like(fatigue)
        )

        return reward + surge_bonus * alive

    def get_safe_intensity(self) -> float:
        """Compute safe intensity based on learned parameters.

        This is what a well-trained SD-MCTS should discover.
        """
        with torch.no_grad():
            alpha = self.alpha.item()
            beta = self.beta.item()
            a_safe = (self.F_req + beta * self.T_s) / ((alpha + beta) * self.T_s)
            return min(1.0, a_safe)
