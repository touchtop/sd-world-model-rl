"""SD World Model for CartPole-v1.

Differentiable world model encoding Newtonian mechanics as structural prior.

Structure (hard-coded — the known physics):
    - Newton's 2nd law for inverted pendulum on a cart
    - Angular acceleration: θ̈ = f(θ, θ̇, F, m_c, m_p, l, g)
    - Linear acceleration:  ẍ  = f(θ, θ̇, θ̈, F, m_c, m_p, l)
    - Euler integration for state update

Parameters (learnable — the unknown θ*):
    - gravity (g): gravitational acceleration
    - masscart (m_c): mass of the cart
    - masspole (m_p): mass of the pole
    - length (l): half-length of the pole
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CartPoleSDModel(nn.Module):
    """Differentiable SD world model for CartPole dynamics.

    Encodes Newton's 2nd law for an inverted pendulum on a cart.
    The equations of motion are the structural prior f(s, a; θ);
    physical constants (g, m_c, m_p, l) are the learnable parameters θ.

    State layout (4 dimensions):
        [0] x       - cart position
        [1] x_dot   - cart velocity
        [2] theta   - pole angle (rad, 0 = upright)
        [3] theta_dot - pole angular velocity

    Action space: Discrete(2)
        0: push left  (force = -force_mag)
        1: push right (force = +force_mag)
    """

    def __init__(
        self,
        force_mag: float = 10.0,
        dt: float = 0.02,  # Gym CartPole uses tau=0.02
        # Initial guesses for learnable parameters (can be noisy)
        gravity_init: float = 9.8,
        masscart_init: float = 1.0,
        masspole_init: float = 0.1,
        length_init: float = 0.5,
        learn_params: bool = True,
        prior_noise: float = 0.0,  # Add noise to initial parameter guesses
    ):
        super().__init__()

        self.force_mag = force_mag
        self.dt = dt
        self.state_dim = 4

        # Store true values for reference
        self.register_buffer('true_gravity', torch.tensor(gravity_init))
        self.register_buffer('true_masscart', torch.tensor(masscart_init))
        self.register_buffer('true_masspole', torch.tensor(masspole_init))
        self.register_buffer('true_length', torch.tensor(length_init))

        if learn_params:
            # Add optional noise to initial parameter guesses
            g_init = gravity_init + prior_noise * np.random.uniform(-2.0, 2.0)
            mc_init = masscart_init + prior_noise * np.random.uniform(-0.3, 0.3)
            mp_init = masspole_init + prior_noise * np.random.uniform(-0.03, 0.03)
            l_init = length_init + prior_noise * np.random.uniform(-0.15, 0.15)

            # Use softplus parameterization to ensure positivity
            self._gravity_raw = nn.Parameter(
                torch.tensor(np.log(np.exp(max(g_init, 0.1)) - 1), dtype=torch.float32)
            )
            self._masscart_raw = nn.Parameter(
                torch.tensor(np.log(np.exp(max(mc_init, 0.1)) - 1), dtype=torch.float32)
            )
            self._masspole_raw = nn.Parameter(
                torch.tensor(np.log(np.exp(max(mp_init, 0.01)) - 1), dtype=torch.float32)
            )
            self._length_raw = nn.Parameter(
                torch.tensor(np.log(np.exp(max(l_init, 0.01)) - 1), dtype=torch.float32)
            )
        else:
            self.register_buffer('gravity', torch.tensor(gravity_init, dtype=torch.float32))
            self.register_buffer('masscart', torch.tensor(masscart_init, dtype=torch.float32))
            self.register_buffer('masspole', torch.tensor(masspole_init, dtype=torch.float32))
            self.register_buffer('length', torch.tensor(length_init, dtype=torch.float32))

        self.learn_params = learn_params

    @property
    def g(self) -> torch.Tensor:
        if self.learn_params:
            return F.softplus(self._gravity_raw)
        return self.gravity

    @property
    def mc(self) -> torch.Tensor:
        if self.learn_params:
            return F.softplus(self._masscart_raw)
        return self.masscart

    @property
    def mp(self) -> torch.Tensor:
        if self.learn_params:
            return F.softplus(self._masspole_raw)
        return self.masspole

    @property
    def l(self) -> torch.Tensor:
        if self.learn_params:
            return F.softplus(self._length_raw)
        return self.length

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state given current state and action.

        Implements the exact Newtonian equations of motion for an inverted
        pendulum on a cart, using Euler integration.

        This is the structural prior f(s, a; θ) from the paper.

        Args:
            state: (batch, 4) current state [x, x_dot, theta, theta_dot]
            action: (batch,) discrete action (0=left, 1=right)

        Returns:
            next_state: (batch, 4) predicted next state
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]

        # Decode discrete action to continuous force
        force = torch.where(
            action.long() == 1,
            torch.tensor(self.force_mag, device=state.device, dtype=state.dtype),
            torch.tensor(-self.force_mag, device=state.device, dtype=state.dtype),
        )

        # Get learnable physical parameters
        gravity = self.g
        masscart = self.mc
        masspole = self.mp
        length = self.l
        total_mass = masscart + masspole
        polemass_length = masspole * length

        # ========================================
        # STRUCTURAL PRIOR: Newton's 2nd Law
        # These equations are HARD-CODED (known physics)
        # Only the parameters (g, m_c, m_p, l) are learned
        # ========================================

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # Angular acceleration (from Lagrangian mechanics)
        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        theta_acc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )

        # Linear acceleration
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        # Euler integration (structural update rule)
        new_x = x + self.dt * x_dot
        new_x_dot = x_dot + self.dt * x_acc
        new_theta = theta + self.dt * theta_dot
        new_theta_dot = theta_dot + self.dt * theta_acc

        # ========================================
        # PROJECTION OPERATOR Π_C
        # Physical feasibility constraints
        # ========================================
        # Note: We do NOT clamp here during rollout to match Gym's behavior.
        # The termination check is done in the reward function.

        next_state = torch.stack([new_x, new_x_dot, new_theta, new_theta_dot], dim=1)
        return next_state

    def predict_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict reward for CartPole.

        CartPole gives +1 for every step the pole stays upright.
        Episode terminates when angle > 12° or position > 2.4.
        """
        if next_state is None:
            next_state = self.forward(state, action)

        x = next_state[:, 0]
        theta = next_state[:, 2]

        # Check termination conditions
        terminated = (
            (x.abs() > 2.4) |
            (theta.abs() > 12 * 2 * 3.14159 / 360)  # 12 degrees in radians
        )

        # +1 if alive, 0 if terminated
        reward = (~terminated).float()
        return reward

    def get_learned_params(self) -> dict:
        """Return current learned parameter values."""
        return {
            'gravity': self.g.item(),
            'masscart': self.mc.item(),
            'masspole': self.mp.item(),
            'length': self.l.item(),
            'true_gravity': self.true_gravity.item(),
            'true_masscart': self.true_masscart.item(),
            'true_masspole': self.true_masspole.item(),
            'true_length': self.true_length.item(),
        }


class CartPoleWorldModelWrapper:
    """Numpy wrapper for CartPoleSDModel for use with MCTS.

    Provides interface expected by mcts_search function.
    """

    def __init__(self, model: CartPoleSDModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass callable interface for mcts_search."""
        return self.model(state, action)

    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Predict next state (numpy interface)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.long, device=self.device)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)
            next_state = self.model.forward(state_t, action_t)
            return next_state.squeeze(0).cpu().numpy()

    def predict_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Predict reward (numpy interface)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.long, device=self.device)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)
            reward = self.model.predict_reward(state_t, action_t)
            return reward.squeeze().cpu().item()
