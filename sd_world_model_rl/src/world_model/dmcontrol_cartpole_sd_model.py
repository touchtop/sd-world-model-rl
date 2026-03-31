"""SD World Model for DMControl Cartpole-Swingup.

Differentiable world model encoding Newtonian mechanics as structural prior,
precisely matching DMControl's MuJoCo-based cartpole environment.

MuJoCo Config (from XML inspection):
    - Integrator: RK4 (model.opt.integrator = 1)
    - Timestep: 0.01
    - n_sub_steps: 1
    - Gravity: -9.81 (z-axis)
    - Cart mass: 1.0, Pole mass: 0.1
    - Pole CoM offset: 0.5 (half-length)
    - Actuator gear: 10
    - Joint damping: slider=0.0005, hinge=2e-6

Reward function (smooth/non-sparse):
    reward = upright * small_control * small_velocity * centered
    where:
        upright = (cos(θ) + 1) / 2
        centered = tolerance(cart_position, margin=2)  →  (1 + centered) / 2
        small_control = tolerance(action, margin=1, quadratic)  →  (4 + sc) / 5
        small_velocity = tolerance(angular_vel, margin=5)  →  (1 + sv) / 2

State: qpos=[x, θ], qvel=[ẋ, θ̇]   (θ=0 → upright, θ=π → hanging)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DMControlCartPoleSDModel(nn.Module):
    """Differentiable SD world model for DMControl cartpole dynamics.

    Uses RK4 integration to match MuJoCo, with damping terms.

    Internal state: [x, θ, ẋ, θ̇]
    Action: continuous [-1, 1], mapped to force via gear=10
    """

    def __init__(
        self,
        gear: float = 10.0,
        dt: float = 0.01,
        # True physical parameters (from MuJoCo XML)
        gravity_init: float = 9.81,
        masscart_init: float = 1.0,
        masspole_init: float = 0.1,
        length_init: float = 0.5,  # pole CoM offset (half-length)
        # Damping from XML (very small, fixed — not learned)
        slider_damping: float = 0.0005,
        hinge_damping: float = 2e-6,
        learn_params: bool = True,
        prior_noise: float = 0.0,
        mp_prior_noise: Optional[float] = None,
        mp_clamp_min: float = 0.01,
    ):
        super().__init__()

        self.gear = gear
        self.dt = dt
        self.state_dim = 4

        # Fixed damping coefficients (from MuJoCo XML, not learned)
        self.register_buffer('slider_damping',
                             torch.tensor(slider_damping, dtype=torch.float32))
        self.register_buffer('hinge_damping',
                             torch.tensor(hinge_damping, dtype=torch.float32))

        # Store true values
        self.register_buffer('true_gravity', torch.tensor(gravity_init))
        self.register_buffer('true_masscart', torch.tensor(masscart_init))
        self.register_buffer('true_masspole', torch.tensor(masspole_init))
        self.register_buffer('true_length', torch.tensor(length_init))

        if learn_params:
            g_init = gravity_init * (1 + prior_noise * np.random.uniform(-0.3, 0.3))
            mc_init = masscart_init * (1 + prior_noise * np.random.uniform(-0.3, 0.3))
            mp_noise = mp_prior_noise if mp_prior_noise is not None else prior_noise
            mp_init = masspole_init * (1 + mp_noise * np.random.uniform(-0.3, 0.3))
            l_init = length_init * (1 + prior_noise * np.random.uniform(-0.3, 0.3))

            self._mp_clamp_min = mp_clamp_min
            self._gravity_raw = nn.Parameter(
                torch.tensor(self._inv_softplus(max(g_init, 0.1)), dtype=torch.float32))
            self._masscart_raw = nn.Parameter(
                torch.tensor(self._inv_softplus(max(mc_init, 0.05)), dtype=torch.float32))
            self._masspole_raw = nn.Parameter(
                torch.tensor(self._inv_softplus(max(mp_init, mp_clamp_min)), dtype=torch.float32))
            self._length_raw = nn.Parameter(
                torch.tensor(self._inv_softplus(max(l_init, 0.01)), dtype=torch.float32))
        else:
            self.register_buffer('_g_fixed', torch.tensor(gravity_init, dtype=torch.float32))
            self.register_buffer('_mc_fixed', torch.tensor(masscart_init, dtype=torch.float32))
            self.register_buffer('_mp_fixed', torch.tensor(masspole_init, dtype=torch.float32))
            self.register_buffer('_l_fixed', torch.tensor(length_init, dtype=torch.float32))

        self.learn_params = learn_params
        self._mp_clamp_min_val = mp_clamp_min

    @staticmethod
    def _inv_softplus(x):
        return float(np.log(np.exp(x) - 1))

    @property
    def g(self):
        return F.softplus(self._gravity_raw).clamp_min(0.1) if self.learn_params else self._g_fixed

    @property
    def mc(self):
        return F.softplus(self._masscart_raw).clamp_min(0.05) if self.learn_params else self._mc_fixed

    @property
    def mp(self):
        return F.softplus(self._masspole_raw).clamp_min(self._mp_clamp_min_val) if self.learn_params else self._mp_fixed

    @property
    def l(self):
        return F.softplus(self._length_raw).clamp_min(0.05) if self.learn_params else self._l_fixed

    def _accelerations(self, x, theta, x_dot, theta_dot, force):
        """Compute accelerations from Barto et al. (1983) equations + damping.

        This is the STRUCTURAL PRIOR f(s, a; θ).
        """
        gravity = self.g
        masscart = self.mc
        masspole = self.mp
        length = self.l
        total_mass = masscart + masspole
        polemass_length = masspole * length

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # Include damping forces
        force_with_damping = force - self.slider_damping * x_dot
        theta_dot_damped = theta_dot  # hinge damping is negligible (2e-6)

        temp = (force_with_damping + polemass_length * theta_dot_damped ** 2 * sintheta) / total_mass
        denom = length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        denom = torch.clamp(denom, min=1e-4)
        theta_acc = (gravity * sintheta - costheta * temp) / denom
        # Subtract hinge damping torque (tiny but correct)
        theta_acc = theta_acc - self.hinge_damping * theta_dot / (masspole * length ** 2)

        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        return x_acc, theta_acc

    def _rk4_step(self, state, force):
        """Single RK4 integration step — matches MuJoCo's integrator."""
        x, theta, x_dot, theta_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        h = self.dt

        # k1
        ax1, ath1 = self._accelerations(x, theta, x_dot, theta_dot, force)
        k1_x, k1_th = x_dot, theta_dot
        k1_xd, k1_thd = ax1, ath1

        # k2
        x2 = x + 0.5 * h * k1_x
        th2 = theta + 0.5 * h * k1_th
        xd2 = x_dot + 0.5 * h * k1_xd
        thd2 = theta_dot + 0.5 * h * k1_thd
        ax2, ath2 = self._accelerations(x2, th2, xd2, thd2, force)
        k2_x, k2_th = xd2, thd2
        k2_xd, k2_thd = ax2, ath2

        # k3
        x3 = x + 0.5 * h * k2_x
        th3 = theta + 0.5 * h * k2_th
        xd3 = x_dot + 0.5 * h * k2_xd
        thd3 = theta_dot + 0.5 * h * k2_thd
        ax3, ath3 = self._accelerations(x3, th3, xd3, thd3, force)
        k3_x, k3_th = xd3, thd3
        k3_xd, k3_thd = ax3, ath3

        # k4
        x4 = x + h * k3_x
        th4 = theta + h * k3_th
        xd4 = x_dot + h * k3_xd
        thd4 = theta_dot + h * k3_thd
        ax4, ath4 = self._accelerations(x4, th4, xd4, thd4, force)
        k4_x, k4_th = xd4, thd4
        k4_xd, k4_thd = ax4, ath4

        # Update
        new_x = x + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        new_theta = theta + (h / 6) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        new_x_dot = x_dot + (h / 6) * (k1_xd + 2 * k2_xd + 2 * k3_xd + k4_xd)
        new_theta_dot = theta_dot + (h / 6) * (k1_thd + 2 * k2_thd + 2 * k3_thd + k4_thd)

        return torch.stack([new_x, new_theta, new_x_dot, new_theta_dot], dim=-1)

    def forward(self, state, action):
        """Predict next state using RK4 integration.

        Args:
            state: (..., 4) [x, θ, ẋ, θ̇]
            action: (..., 1) or (...,) in [-1, 1]
        Returns:
            next_state: (..., 4)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        force = action[..., 0] * self.gear
        return self._rk4_step(state, force)

    @staticmethod
    def obs_to_state(obs):
        """Convert DMControl observation to internal state.

        obs: [cart_pos, cos(θ), sin(θ), ẋ, θ̇] → state: [x, θ, ẋ, θ̇]
        Uses atan2(sin, cos) which is correct for the full circle.
        """
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                x = obs[0]
                theta = np.arctan2(obs[2], obs[1])
                return np.array([x, theta, obs[3], obs[4]], dtype=np.float32)
            else:
                x = obs[:, 0]
                theta = np.arctan2(obs[:, 2], obs[:, 1])
                return np.stack([x, theta, obs[:, 3], obs[:, 4]], axis=1).astype(np.float32)
        else:
            raise TypeError(f"Expected numpy array, got {type(obs)}")

    @staticmethod
    def state_to_sincos(state_tensor):
        """Convert state [x, θ, ẋ, θ̇] to sin/cos representation [x, cosθ, sinθ, ẋ, θ̇].

        Used for angle-aware loss computation.
        """
        x = state_tensor[..., 0]
        theta = state_tensor[..., 1]
        x_dot = state_tensor[..., 2]
        theta_dot = state_tensor[..., 3]
        return torch.stack([
            x, torch.cos(theta), torch.sin(theta), x_dot, theta_dot
        ], dim=-1)

    def predict_reward(self, state, action, next_state=None):
        """Predict reward matching DMControl's smooth cartpole-swingup reward.

        reward = upright * centered * small_control * small_velocity
        """
        if next_state is None:
            next_state = self.forward(state, action)

        theta = next_state[..., 1]
        x = next_state[..., 0]
        theta_dot = next_state[..., 3]

        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        ctrl = action[..., 0]

        # upright: (cos(θ) + 1) / 2  ∈ [0, 1]
        upright = (torch.cos(theta) + 1) / 2

        # centered: tolerance(x, margin=2, value_at_margin=0.1, gaussian)
        # sigma = margin / sqrt(-2*log(0.1)) = 2/2.1460 ≈ 0.9320
        centered_raw = torch.exp(-0.5 * (x / 0.9320) ** 2)
        centered = (1 + centered_raw) / 2

        # small_control: tolerance(ctrl, margin=1, quadratic) → (4 + sc) / 5
        sc_raw = torch.clamp(1 - (ctrl / 1.0) ** 2, min=0)
        small_control = (4 + sc_raw) / 5

        # small_velocity: tolerance(θ̇, margin=5, value_at_margin=0.1, gaussian)
        # sigma = 5/2.1460 ≈ 2.3300
        sv_raw = torch.exp(-0.5 * (theta_dot / 2.3300) ** 2)
        small_velocity = (1 + sv_raw) / 2

        return upright * centered * small_control * small_velocity

    def get_learned_params(self):
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


class DMControlCartPoleWrapper:
    """Numpy/MCTS-compatible wrapper."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(self, state, action):
        return self.model(state, action)

    def predict_next_state(self, state_np, action_val):
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            a = torch.tensor([action_val], dtype=torch.float32, device=self.device)
            return self.model.forward(s, a).squeeze(0).cpu().numpy()

    def predict_reward_np(self, state_np, action_val):
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            a = torch.tensor([action_val], dtype=torch.float32, device=self.device)
            return self.model.predict_reward(s, a).squeeze().cpu().item()
