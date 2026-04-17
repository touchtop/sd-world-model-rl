"""Structured world model for DMControl Walker-Walk.

This first version intentionally uses a conservative hybrid design:

1. State variables are organized into "stocks" (height + joint angles) and
   their rates (linear / angular velocities).
2. The stock update is explicit and kinematic: positions are integrated from
   their corresponding rates.
3. The rate update is decomposed into three parts:
   - passive flow: gravity / coupling / support effects inferred from state
   - control flow: direct actuator effect on the six leg joints
   - residual flow: unmodeled contact and higher-order effects

The goal is not to exactly recover MuJoCo mechanics in closed form. The goal
is to preserve a mechanically meaningful state layout and update skeleton so
that later versions can strengthen the explicit prior or attach residual nets.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    return torch.remainder(x + torch.pi, two_pi) - torch.pi


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class DMControlWalkerSDModel(nn.Module):
    """Structured-hybrid world model for ``dm_control.suite.walker.walk``.

    State layout (17 dims):
        stocks:
            [0]  height
            [1]  torso angle
            [2]  right hip
            [3]  right knee
            [4]  right ankle
            [5]  left hip
            [6]  left knee
            [7]  left ankle
        rates:
            [8]  z_dot
            [9]  x_dot
            [10] torso_dot
            [11] right hip dot
            [12] right knee dot
            [13] right ankle dot
            [14] left hip dot
            [15] left knee dot
            [16] left ankle dot

    Observation interface:
        ``walker.walk`` exposes:
            - orientations: 14 numbers = (xx, xz) for 7 bodies
            - height: torso height
            - velocity: 9 generalized velocities

        From the 7 body orientations we reconstruct torso angle and the
        6 relative joint angles by differencing adjacent body angles.
    """

    BODY_NAMES = (
        "torso",
        "right_thigh",
        "right_leg",
        "right_foot",
        "left_thigh",
        "left_leg",
        "left_foot",
    )

    JOINT_NAMES = (
        "torso",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    )

    ACTION_NAMES = (
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    )

    STOCK_DIM = 8
    RATE_DIM = 9
    STATE_DIM = 17
    ACTION_DIM = 6

    # Rates corresponding to the 8 stocks above.
    STOCK_RATE_IDX = (0, 2, 3, 4, 5, 6, 7, 8)
    ANGLE_SLICE = slice(1, 8)

    def __init__(
        self,
        dt: float = 0.025,
        hidden: int = 256,
        move_speed: float = 1.0,
        residual_scale: float = 5.0,
        passive_scale: float = 10.0,
    ):
        super().__init__()
        self.dt = dt
        self.move_speed = move_speed
        self.residual_scale = residual_scale
        self.passive_scale = passive_scale

        self.passive_flow = _mlp(self.STATE_DIM, self.RATE_DIM, hidden)
        self.residual_flow = _mlp(self.STATE_DIM + self.ACTION_DIM, self.RATE_DIM, hidden)

        # Direct actuator effect only enters the six actuated joint channels.
        self.control_gain = nn.Parameter(torch.ones(self.ACTION_DIM, dtype=torch.float32))
        self.rate_damping_raw = nn.Parameter(torch.full((self.RATE_DIM,), 0.1, dtype=torch.float32))

    @property
    def rate_damping(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rate_damping_raw)

    @staticmethod
    def flatten_obs(obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            np.asarray(obs["orientations"], dtype=np.float32).reshape(-1),
            np.asarray(obs["height"], dtype=np.float32).reshape(1),
            np.asarray(obs["velocity"], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)

    @staticmethod
    def _absolute_body_angles_from_pairs(orientations: np.ndarray) -> np.ndarray:
        pairs = orientations.reshape(7, 2)
        return np.arctan2(pairs[:, 1], pairs[:, 0]).astype(np.float32)

    @classmethod
    def obs_to_state(cls, obs: Union[Mapping[str, np.ndarray], np.ndarray, Sequence[float]]) -> np.ndarray:
        if isinstance(obs, Mapping):
            orientations = np.asarray(obs["orientations"], dtype=np.float32).reshape(14)
            height = float(np.asarray(obs["height"], dtype=np.float32).reshape(()))
            velocity = np.asarray(obs["velocity"], dtype=np.float32).reshape(9)
        else:
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            if flat.shape[0] != 24:
                raise ValueError(f"Expected walker flat observation of size 24, got {flat.shape[0]}")
            orientations = flat[:14]
            height = float(flat[14])
            velocity = flat[15:]

        body_angles = cls._absolute_body_angles_from_pairs(orientations)
        torso = body_angles[0]
        joint_angles = np.array([
            torso,
            body_angles[1] - body_angles[0],
            body_angles[2] - body_angles[1],
            body_angles[3] - body_angles[2],
            body_angles[4] - body_angles[0],
            body_angles[5] - body_angles[4],
            body_angles[6] - body_angles[5],
        ], dtype=np.float32)
        joint_angles = _wrap_angle_np(joint_angles)

        state = np.concatenate([
            np.array([height], dtype=np.float32),
            joint_angles,
            velocity.astype(np.float32),
        ]).astype(np.float32)
        return state

    @staticmethod
    def state_to_body_angles(state: torch.Tensor) -> torch.Tensor:
        torso = state[..., 1]
        right_thigh = _wrap_angle_torch(torso + state[..., 2])
        right_leg = _wrap_angle_torch(right_thigh + state[..., 3])
        right_foot = _wrap_angle_torch(right_leg + state[..., 4])
        left_thigh = _wrap_angle_torch(torso + state[..., 5])
        left_leg = _wrap_angle_torch(left_thigh + state[..., 6])
        left_foot = _wrap_angle_torch(left_leg + state[..., 7])
        return torch.stack(
            [torso, right_thigh, right_leg, right_foot, left_thigh, left_leg, left_foot],
            dim=-1,
        )

    def _control_flow(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(0)
        joint_control = self.control_gain.unsqueeze(0) * action
        zeros = torch.zeros(action.shape[0], 3, dtype=action.dtype, device=action.device)
        return torch.cat([zeros, joint_control], dim=-1)

    def _rate_acceleration(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        passive = torch.tanh(self.passive_flow(state)) * self.passive_scale
        residual_in = torch.cat([state, action], dim=-1)
        residual = torch.tanh(self.residual_flow(residual_in)) * self.residual_scale
        damping = self.rate_damping.unsqueeze(0) * state[:, self.STOCK_DIM:]
        return passive + self._control_flow(action) + residual - damping

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        stocks = state[:, :self.STOCK_DIM]
        rates = state[:, self.STOCK_DIM:]

        rate_acc = self._rate_acceleration(state, action)
        next_rates = rates + self.dt * rate_acc

        stock_rates = next_rates[:, list(self.STOCK_RATE_IDX)]
        next_stocks = stocks + self.dt * stock_rates
        next_height = torch.clamp(next_stocks[:, :1], min=0.0)
        next_angles = _wrap_angle_torch(next_stocks[:, self.ANGLE_SLICE])
        next_stocks = torch.cat([next_height, next_angles], dim=-1)

        return torch.cat([next_stocks, next_rates], dim=-1)

    @staticmethod
    def _lower_tolerance(
        x: torch.Tensor,
        lower: float,
        margin: float,
        value_at_margin: float = 0.1,
        sigmoid: str = "gaussian",
    ) -> torch.Tensor:
        in_bounds = x >= lower
        if margin <= 0:
            return in_bounds.to(dtype=x.dtype)

        d = torch.clamp((lower - x) / margin, min=0.0)
        if sigmoid == "gaussian":
            scale = float(np.sqrt(-2.0 * np.log(value_at_margin)))
            value = torch.exp(-0.5 * (d * scale) ** 2)
        elif sigmoid == "linear":
            scale = 1.0 - value_at_margin
            value = torch.clamp(1.0 - d * scale, min=0.0)
        else:
            raise ValueError(f"Unsupported sigmoid: {sigmoid}")
        return torch.where(in_bounds, torch.ones_like(value), value)

    def predict_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if next_state is None:
            next_state = self.forward(state, action)

        height = next_state[..., 0]
        torso = next_state[..., 1]
        x_dot = next_state[..., 9]

        standing = self._lower_tolerance(height, lower=1.2, margin=0.6, value_at_margin=0.1, sigmoid="gaussian")
        upright = (1.0 + torch.cos(torso)) / 2.0
        stand_reward = (3.0 * standing + upright) / 4.0

        if self.move_speed <= 0:
            return stand_reward

        move_reward = self._lower_tolerance(
            x_dot,
            lower=self.move_speed,
            margin=self.move_speed / 2.0,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return stand_reward * (5.0 * move_reward + 1.0) / 6.0

    def get_learned_params(self) -> Dict[str, object]:
        return {
            "dt": self.dt,
            "move_speed": self.move_speed,
            "control_gain": self.control_gain.detach().cpu().tolist(),
            "rate_damping": self.rate_damping.detach().cpu().tolist(),
            "passive_scale": self.passive_scale,
            "residual_scale": self.residual_scale,
        }


class DMControlWalkerWrapper:
    """Numpy / planning compatible wrapper."""

    def __init__(self, model: DMControlWalkerSDModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(state, action)

    def predict_next_state(self, state_np: np.ndarray, action_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            return self.model(state, action).squeeze(0).cpu().numpy()

    def predict_reward_np(self, state_np: np.ndarray, action_np: np.ndarray) -> float:
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            reward = self.model.predict_reward(state, action)
            return float(reward.squeeze().cpu().item())
