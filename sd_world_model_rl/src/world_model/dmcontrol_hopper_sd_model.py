"""Structured-hybrid world model for DMControl Hopper-Hop."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    return torch.remainder(x + torch.pi, two_pi) - np.pi


def _touch_raw_to_contact_np(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-np.clip(x, a_min=0.0, a_max=None))


def _touch_raw_to_contact_torch(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-torch.clamp(x, min=0.0))


def _inv_softplus_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class DMControlHopperSDModel(nn.Module):
    """Structured world model for ``dm_control.suite.hopper.hop``.

    Two representations are used:

    1. Raw observation layout (15 dims), matching dm_control state observations:
       [rootz, rooty, waist, hip, knee, ankle, x_dot, z_dot,
        rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot, touch_toe, touch_heel]

    2. Structured latent layout (17 dims), used by TD-MPC2 planning:
       [height, vx, vz, pitch, pitch_rate, compression, compression_rate,
        swing, swing_rate, foot_angle, foot_rate, support, stance_mix,
        impact, clearance, rootz, front_anchor]
    """

    POSITION_DIM = 6
    VELOCITY_DIM = 7
    TOUCH_DIM = 2
    RAW_OBS_DIM = 15
    STATE_DIM = 17
    ACTION_DIM = 4
    ANGLE_SLICE = slice(1, 6)
    STOCK_RATE_IDX = (1, 2, 3, 4, 5, 6)
    TORSO_TO_PELVIS_Z = -0.05
    PELVIS_TO_THIGH_Z = -0.20
    THIGH_TO_CALF_Z = -0.33
    CALF_TO_FOOT_Z = -0.32
    TOE_OFFSET_X = 0.17
    HEEL_OFFSET_X = -0.08
    CONTACT_MARGIN = 0.02
    CONTACT_SCALE = 0.02
    HEIGHT_RESIDUAL_SCALE = 0.05
    TOUCH_RESIDUAL_SCALE = 0.10
    TARGET_STAND_HEIGHT = 0.90
    CONTACT_VEL_SCALE = 0.35
    SUPPORT_BLEND_SCALE = 0.08
    HEIGHT_IDX = 0
    VX_IDX = 1
    VZ_IDX = 2
    PITCH_IDX = 3
    PITCH_RATE_IDX = 4
    COMP_IDX = 5
    COMP_RATE_IDX = 6
    SWING_IDX = 7
    SWING_RATE_IDX = 8
    FOOT_IDX = 9
    FOOT_RATE_IDX = 10
    SUPPORT_IDX = 11
    STANCE_IDX = 12
    IMPACT_IDX = 13
    CLEARANCE_IDX = 14
    ROOTZ_IDX = 15  # v4: raw rootz (slide joint) for asymmetric contact viability
    FRONT_ANCHOR_IDX = 16  # v4.3: explicit fore-body anchor mode for crawl/recovery
    RATE_LIKE_IDX = (VX_IDX, VZ_IDX, PITCH_RATE_IDX, COMP_RATE_IDX, SWING_RATE_IDX, FOOT_RATE_IDX, IMPACT_IDX)
    ENV_ACTION_REPEAT = 2

    # v4 asymmetric-contact viability thresholds (derived from Phase 0 diagnostic,
    # see docs/worklog/hopper_walker_extension_2026_04_11.md for data support).
    V4_ROOTZ_LOW_CENTER = -0.5     # sigmoid center for "body is down"
    V4_ROOTZ_LOW_SCALE = 0.1
    V4_PITCH_BACK_CENTER = -0.3    # pitch below this is "tilted backward"
    V4_PITCH_BACK_SCALE = 0.1
    V4_PITCH_FWD_CENTER = 0.0      # forward_lean admits any positive pitch smoothly
    V4_PITCH_FWD_SCALE = 0.15
    V4_PITCH_NEUTRAL_SPREAD = 0.3  # recovered_upright admits |pitch| up to ~1 rad
    V4_FRONT_ANCHOR_PITCH_CENTER = 0.8
    V4_FRONT_ANCHOR_PITCH_SCALE = 0.35
    V4_FRONT_ANCHOR_CLEARANCE_CENTER = 0.74
    V4_FRONT_ANCHOR_CLEARANCE_SCALE = 0.06

    def _front_anchor_from_geom(
        self,
        pitch: torch.Tensor,
        clearance: torch.Tensor,
        rootz: torch.Tensor,
    ) -> torch.Tensor:
        body_low = torch.sigmoid((self.V4_ROOTZ_LOW_CENTER - rootz) / self.V4_ROOTZ_LOW_SCALE)
        front_anchor_pitch = torch.sigmoid((pitch - self.V4_FRONT_ANCHOR_PITCH_CENTER) / self.V4_FRONT_ANCHOR_PITCH_SCALE)
        front_anchor_clearance = torch.sigmoid(((-clearance) - self.V4_FRONT_ANCHOR_CLEARANCE_CENTER) / self.V4_FRONT_ANCHOR_CLEARANCE_SCALE)
        return body_low * front_anchor_pitch * front_anchor_clearance

    def __init__(
        self,
        dt: float = 0.02,
        hidden: int = 256,
        hop_speed: float = 2.0,
        residual_scale: float = 5.0,
        passive_scale: float = 10.0,
    ):
        super().__init__()
        self.dt = dt
        self.hop_speed = hop_speed
        self.residual_scale = residual_scale
        self.passive_scale = passive_scale
        self.learned_passive_weight = 0.20
        self.learned_residual_weight = 0.15

        self.passive_flow = _mlp(self.STATE_DIM, len(self.RATE_LIKE_IDX), hidden)
        self.residual_flow = _mlp(self.STATE_DIM + self.ACTION_DIM, len(self.RATE_LIKE_IDX), hidden)
        self.touch_head = _mlp(self.STATE_DIM, 2, hidden)
        self.height_head = _mlp(self.STATE_DIM, 1, hidden)
        self.clearance_head = _mlp(self.STATE_DIM, 1, hidden)

        self.register_buffer(
            "base_control_gears",
            torch.tensor([30.0, 40.0, 30.0, 10.0], dtype=torch.float32) / 40.0,
        )
        self.register_buffer(
            "joint_neutral_pose",
            torch.tensor([0.0, -1.20, 1.20, 0.0], dtype=torch.float32),
        )
        self.register_buffer(
            "base_rate_damping",
            torch.tensor([0.05, 0.10, 0.08, 0.06, 0.06, 0.07, 0.05], dtype=torch.float32),
        )
        self.control_scale = nn.Parameter(torch.ones(self.ACTION_DIM, dtype=torch.float32))
        self.rate_damping_raw = nn.Parameter(_inv_softplus_torch(self.base_rate_damping.clone()))

    @property
    def rate_damping(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rate_damping_raw)

    @classmethod
    def flatten_obs(cls, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            np.asarray(obs["position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["velocity"], dtype=np.float32).reshape(-1),
            np.asarray(obs["touch"], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)

    @classmethod
    def obs_to_state(cls, obs: Union[Mapping[str, np.ndarray], np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compatibility helper for legacy scripts: normalized raw observation."""
        if isinstance(obs, Mapping):
            pos = np.asarray(obs["position"], dtype=np.float32).reshape(6)
            vel = np.asarray(obs["velocity"], dtype=np.float32).reshape(7)
            touch = np.asarray(obs["touch"], dtype=np.float32).reshape(2)
        else:
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            if flat.shape[0] != 15:
                raise ValueError(f"Expected hopper flat observation of size 15, got {flat.shape[0]}")
            pos = flat[:6]
            vel = flat[6:13]
            touch = flat[13:]
        pos = pos.copy()
        pos[2:] = _wrap_angle_np(pos[2:])
        touch = _touch_raw_to_contact_np(np.asarray(touch, dtype=np.float32))
        return np.concatenate([pos, vel, touch], dtype=np.float32)

    @staticmethod
    def _rotate_local(x_local: torch.Tensor, z_local: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_world = cos_a * x_local + sin_a * z_local
        z_world = -sin_a * x_local + cos_a * z_local
        return x_world, z_world

    def _absolute_angles(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torso = pos[:, 1]
        pelvis = torso + pos[:, 2]
        thigh = pelvis + pos[:, 3]
        calf = thigh + pos[:, 4]
        foot = calf + pos[:, 5]
        return torso, pelvis, thigh, calf, foot

    def _kinematic_features(self, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        torso, pelvis, thigh, calf, foot = self._absolute_angles(pos)
        root_z = pos[:, 0]

        pelvis_x, pelvis_z = self._rotate_local(
            torch.zeros_like(root_z),
            torch.full_like(root_z, self.TORSO_TO_PELVIS_Z),
            torso,
        )
        thigh_x, thigh_z = self._rotate_local(
            torch.zeros_like(root_z),
            torch.full_like(root_z, self.PELVIS_TO_THIGH_Z),
            pelvis,
        )
        calf_x, calf_z = self._rotate_local(
            torch.zeros_like(root_z),
            torch.full_like(root_z, self.THIGH_TO_CALF_Z),
            thigh,
        )
        foot_x, foot_z = self._rotate_local(
            torch.zeros_like(root_z),
            torch.full_like(root_z, self.CALF_TO_FOOT_Z),
            calf,
        )

        foot_body_x = pelvis_x + thigh_x + calf_x + foot_x
        foot_body_z = root_z + pelvis_z + thigh_z + calf_z + foot_z

        toe_dx, toe_dz = self._rotate_local(
            torch.full_like(root_z, self.TOE_OFFSET_X),
            torch.zeros_like(root_z),
            foot,
        )
        heel_dx, heel_dz = self._rotate_local(
            torch.full_like(root_z, self.HEEL_OFFSET_X),
            torch.zeros_like(root_z),
            foot,
        )
        return {
            "foot_body_x": foot_body_x,
            "foot_body_z": foot_body_z,
            "toe_z": foot_body_z + toe_dz,
            "heel_z": foot_body_z + heel_dz,
            "foot_angle": foot,
        }

    def _explicit_height_from_pos(self, pos: torch.Tensor) -> torch.Tensor:
        kin = self._kinematic_features(pos)
        return pos[:, 0] - kin["foot_body_z"]

    def _pos_rates_from_vel(self, vel: torch.Tensor) -> torch.Tensor:
        return torch.cat([vel[:, 1:2], vel[:, 2:]], dim=-1)

    def _contact_features(self, pos: torch.Tensor, vel: torch.Tensor, touch: torch.Tensor) -> Dict[str, torch.Tensor]:
        kin = self._kinematic_features(pos)
        pos_eps = pos + self.dt * self._pos_rates_from_vel(vel)
        kin_eps = self._kinematic_features(pos_eps)

        toe_vz = (kin_eps["toe_z"] - kin["toe_z"]) / self.dt
        heel_vz = (kin_eps["heel_z"] - kin["heel_z"]) / self.dt
        toe_geom = torch.sigmoid((self.CONTACT_MARGIN - kin["toe_z"]) / self.CONTACT_SCALE)
        heel_geom = torch.sigmoid((self.CONTACT_MARGIN - kin["heel_z"]) / self.CONTACT_SCALE)
        toe_impact = torch.sigmoid(-toe_vz / self.CONTACT_VEL_SCALE)
        heel_impact = torch.sigmoid(-heel_vz / self.CONTACT_VEL_SCALE)
        toe_support = torch.maximum(touch[:, 0], toe_geom * toe_impact)
        heel_support = torch.maximum(touch[:, 1], heel_geom * heel_impact)
        support = torch.clamp(0.65 * torch.maximum(toe_support, heel_support) + 0.35 * touch.max(dim=-1).values, 0.0, 1.0)
        stance_mix = torch.tanh((toe_support - heel_support) / self.SUPPORT_BLEND_SCALE)
        return {
            "support": support,
            "toe_support": toe_support,
            "heel_support": heel_support,
            "stance_mix": stance_mix,
            "toe_vz": toe_vz,
            "heel_vz": heel_vz,
            "kin": kin,
        }

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.RAW_OBS_DIM:
            raise ValueError(f"Expected raw hopper observation of size {self.RAW_OBS_DIM}, got {obs.shape[-1]}")

        leading = obs.shape[:-1]
        obs_flat = obs.reshape(-1, obs.shape[-1])

        pos = torch.cat([obs_flat[:, :2], _wrap_angle_torch(obs_flat[:, 2:6])], dim=-1)
        vel = obs_flat[:, 6:13]
        touch = _touch_raw_to_contact_torch(obs_flat[:, 13:15])
        contact = self._contact_features(pos, vel, touch)
        kin = contact["kin"]

        height = self._explicit_height_from_pos(pos)
        vx = vel[:, 0]
        vz = vel[:, 1]
        pitch = pos[:, 1]
        pitch_rate = vel[:, 2]
        compression = torch.relu(self.TARGET_STAND_HEIGHT - height)
        compression_rate = -vz
        swing = 0.15 * pos[:, 2] + 0.55 * pos[:, 3] - 0.35 * pos[:, 4] + 0.10 * pos[:, 5]
        swing_rate = 0.15 * vel[:, 3] + 0.55 * vel[:, 4] - 0.35 * vel[:, 5] + 0.10 * vel[:, 6]
        foot_angle = kin["foot_angle"]
        foot_rate = vel[:, 2] + vel[:, 3] + vel[:, 4] + vel[:, 5] + vel[:, 6]
        support = contact["support"]
        stance_mix = contact["stance_mix"]
        impact = torch.relu(-(contact["toe_vz"] + contact["heel_vz"]) * 0.5) * support
        clearance = torch.minimum(kin["toe_z"], kin["heel_z"])
        # v4 new structural dim: raw rootz (slide joint position).
        # Empirically separates success (rootz > -0.3) from failure basin (rootz < -0.8)
        # when combined with the unwrapped torso pitch in predict_plan_reward.
        rootz = pos[:, 0]
        front_anchor = self._front_anchor_from_geom(pitch, clearance, rootz)

        latent = torch.stack(
            [
                height,
                vx,
                vz,
                pitch,
                pitch_rate,
                compression,
                compression_rate,
                swing,
                swing_rate,
                foot_angle,
                foot_rate,
                support,
                stance_mix,
                impact,
                clearance,
                rootz,
                front_anchor,
            ],
            dim=-1,
        )
        return latent.reshape(*leading, self.STATE_DIM)

    def _explicit_rate_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        scaled_action = action * self.control_scale.unsqueeze(0)

        height = state[:, self.HEIGHT_IDX]
        vx = state[:, self.VX_IDX]
        vz = state[:, self.VZ_IDX]
        pitch = state[:, self.PITCH_IDX]
        pitch_rate = state[:, self.PITCH_RATE_IDX]
        compression = state[:, self.COMP_IDX]
        compression_rate = state[:, self.COMP_RATE_IDX]
        swing = state[:, self.SWING_IDX]
        swing_rate = state[:, self.SWING_RATE_IDX]
        foot = state[:, self.FOOT_IDX]
        foot_rate = state[:, self.FOOT_RATE_IDX]
        clearance = state[:, self.CLEARANCE_IDX]
        support = torch.clamp(state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        stance_mix = torch.clamp(state[:, self.STANCE_IDX], min=-1.0, max=1.0)
        impact = torch.clamp(state[:, self.IMPACT_IDX], min=0.0)
        front_anchor = torch.clamp(state[:, self.FRONT_ANCHOR_IDX], min=0.0, max=1.0)
        drive_support = torch.maximum(support, 0.75 * front_anchor)
        posture_support = torch.maximum(support, 0.90 * front_anchor)
        lift_support = torch.maximum(support, 0.35 * front_anchor)
        air = 1.0 - torch.maximum(support, 0.60 * front_anchor)

        drive_action = 0.20 * scaled_action[:, 0] + 0.55 * scaled_action[:, 1] - 0.50 * scaled_action[:, 2] + 0.25 * scaled_action[:, 3]
        posture_action = 0.30 * scaled_action[:, 0] - 0.10 * scaled_action[:, 1] + 0.05 * scaled_action[:, 2] + 0.05 * scaled_action[:, 3]
        foot_action = 0.05 * scaled_action[:, 0] + 0.10 * scaled_action[:, 1] - 0.15 * scaled_action[:, 2] + 0.35 * scaled_action[:, 3]
        contact_lock = torch.maximum(support, 0.70 * front_anchor) * torch.sigmoid((-clearance - 0.70) / 0.06)
        # Only sufficiently strong positive release should become vertical liftoff.
        release_lift = torch.relu(drive_action - 0.95) * (1.0 - 0.78 * contact_lock)

        x_acc = drive_support * (0.90 * drive_action + 0.25 * swing_rate + 0.15 * stance_mix - 0.18 * vx) + 0.35 * front_anchor * torch.relu(pitch) - 0.06 * air * vx
        z_acc = -4.8 - 1.6 * air + lift_support * (10.0 * compression + 4.0 * release_lift + 1.2 * impact - 1.4 * torch.relu(vz))
        pitch_acc = -2.5 * pitch - 0.75 * pitch_rate - 0.35 * swing + posture_support * (0.15 * posture_action + 0.08 * stance_mix) + 0.40 * front_anchor * torch.relu(pitch)
        compression_acc = lift_support * (1.4 * torch.relu(-vz) + 0.8 * impact - 3.2 * release_lift) - 0.90 * compression - 0.35 * compression_rate + 0.15 * air * compression + 0.20 * front_anchor * compression
        swing_acc = 0.80 * drive_action - 1.10 * swing - 0.40 * swing_rate - 0.15 * stance_mix + 0.10 * front_anchor * torch.relu(pitch)
        foot_acc = 0.90 * foot_action - 1.00 * foot - 0.35 * foot_rate + 0.10 * stance_mix
        impact_rate = -3.0 * impact + lift_support * torch.relu(-vz)

        return torch.cat(
            [
                x_acc.unsqueeze(-1),
                z_acc.unsqueeze(-1),
                pitch_acc.unsqueeze(-1),
                compression_acc.unsqueeze(-1),
                swing_acc.unsqueeze(-1),
                foot_acc.unsqueeze(-1),
                impact_rate.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _rate_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        explicit = self._explicit_rate_dynamics(state, action)
        passive = self.learned_passive_weight * torch.tanh(self.passive_flow(state)) * self.passive_scale
        residual = self.learned_residual_weight * torch.tanh(
            self.residual_flow(torch.cat([state, action], dim=-1))
        ) * self.residual_scale
        damping = self.rate_damping.unsqueeze(0) * state[:, list(self.RATE_LIKE_IDX)]
        return explicit + passive + residual - damping

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        cols = [state[:, i] for i in range(self.STATE_DIM)]
        cols[self.HEIGHT_IDX] = torch.clamp(cols[self.HEIGHT_IDX], min=0.0, max=2.5)
        cols[self.PITCH_IDX] = torch.clamp(cols[self.PITCH_IDX], min=-6.5, max=6.5)
        cols[self.COMP_IDX] = torch.clamp(cols[self.COMP_IDX], min=0.0, max=1.5)
        cols[self.SWING_IDX] = _wrap_angle_torch(cols[self.SWING_IDX])
        cols[self.FOOT_IDX] = _wrap_angle_torch(cols[self.FOOT_IDX])
        cols[self.SUPPORT_IDX] = torch.clamp(cols[self.SUPPORT_IDX], min=0.0, max=1.0)
        cols[self.STANCE_IDX] = torch.clamp(cols[self.STANCE_IDX], min=-1.0, max=1.0)
        cols[self.IMPACT_IDX] = torch.clamp(cols[self.IMPACT_IDX], min=0.0, max=2.0)
        cols[self.ROOTZ_IDX] = torch.clamp(cols[self.ROOTZ_IDX], min=-1.5, max=1.5)
        cols[self.FRONT_ANCHOR_IDX] = torch.clamp(cols[self.FRONT_ANCHOR_IDX], min=0.0, max=1.0)
        return torch.stack(cols, dim=-1)

    def _forward_one_substep(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        rate_dyn = self._rate_dynamics(state, action)
        next_rate_like = state[:, list(self.RATE_LIKE_IDX)] + self.dt * rate_dyn

        next_vx = next_rate_like[:, 0]
        next_vz = next_rate_like[:, 1]
        next_pitch_rate = next_rate_like[:, 2]
        next_comp_rate = next_rate_like[:, 3]
        next_swing_rate = next_rate_like[:, 4]
        next_foot_rate = next_rate_like[:, 5]
        next_impact = torch.clamp(next_rate_like[:, 6], min=0.0, max=2.0)

        height = state[:, self.HEIGHT_IDX]
        pitch = state[:, self.PITCH_IDX]
        compression = state[:, self.COMP_IDX]
        swing = state[:, self.SWING_IDX]
        foot = state[:, self.FOOT_IDX]
        support = torch.clamp(state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        stance = torch.clamp(state[:, self.STANCE_IDX], min=-1.0, max=1.0)
        front_anchor = torch.clamp(state[:, self.FRONT_ANCHOR_IDX], min=0.0, max=1.0)

        next_height = torch.clamp(height + self.dt * next_vz, min=0.0, max=2.5)
        next_pitch = torch.clamp(pitch + self.dt * next_pitch_rate, min=-6.5, max=6.5)
        integrated_comp = torch.clamp(compression + self.dt * next_comp_rate, min=0.0, max=1.5)
        target_comp = torch.relu(self.TARGET_STAND_HEIGHT - next_height)
        next_comp = 0.65 * integrated_comp + 0.35 * target_comp
        next_swing = _wrap_angle_torch(swing + self.dt * next_swing_rate)
        next_foot = _wrap_angle_torch(foot + self.dt * next_foot_rate)

        clearance_geom = next_height - 0.58 - 0.18 * next_comp - 0.05 * torch.abs(next_pitch) - 0.06 * torch.cos(next_foot)
        touch_residual = self.TOUCH_RESIDUAL_SCALE * torch.tanh(self.touch_head(state))
        contact_geom = torch.sigmoid((self.CONTACT_MARGIN - clearance_geom) / 0.04)
        impact_gate = torch.sigmoid((0.12 - next_vz) / 0.22)
        support_hold = 0.14 * support * torch.sigmoid((next_vz + 0.02) / 0.08) * torch.sigmoid((-clearance_geom - 0.50) / 0.10)
        next_support = torch.clamp(
            0.55 * support + 0.35 * contact_geom * impact_gate + 0.10 * torch.sigmoid(next_impact) + support_hold + touch_residual[:, 0],
            min=0.0,
            max=1.0,
        )
        next_stance = torch.tanh(0.55 * stance + 0.25 * next_swing - 0.08 * next_foot + touch_residual[:, 1])
        clearance_residual = self.HEIGHT_RESIDUAL_SCALE * torch.tanh(self.clearance_head(state).squeeze(-1))
        clearance_contact_penalty = 0.12 * next_support + 0.06 * torch.sigmoid(next_impact)
        next_clearance = clearance_geom - clearance_contact_penalty + clearance_residual
        # v4: propagate rootz via vertical velocity (bounded to a wide physical range).
        rootz = state[:, self.ROOTZ_IDX]
        next_rootz = torch.clamp(rootz + self.dt * next_vz, min=-1.5, max=1.5)
        next_front_anchor_geom = self._front_anchor_from_geom(next_pitch, next_clearance, next_rootz)
        next_rear_trap = torch.sigmoid((self.V4_ROOTZ_LOW_CENTER - next_rootz) / self.V4_ROOTZ_LOW_SCALE) * torch.sigmoid(
            -(next_pitch - self.V4_PITCH_BACK_CENTER) / self.V4_PITCH_BACK_SCALE
        )
        next_front_anchor = torch.clamp(
            0.55 * front_anchor + 0.45 * next_front_anchor_geom + 0.08 * support * torch.sigmoid((next_pitch - self.V4_FRONT_ANCHOR_PITCH_CENTER) / self.V4_FRONT_ANCHOR_PITCH_SCALE) - 0.10 * next_rear_trap,
            min=0.0,
            max=1.0,
        )

        return torch.stack(
            [
                next_height,
                next_vx,
                next_vz,
                next_pitch,
                next_pitch_rate,
                next_comp,
                next_comp_rate,
                next_swing,
                next_swing_rate,
                next_foot,
                next_foot_rate,
                next_support,
                next_stance,
                next_impact,
                next_clearance,
                next_rootz,
                next_front_anchor,
            ],
            dim=-1,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Match the DMControl wrapper, which advances the simulator twice per environment step.
        next_state = state
        for _ in range(self.ENV_ACTION_REPEAT):
            next_state = self._forward_one_substep(next_state, action)
        return next_state

    @staticmethod
    def _tolerance_interval(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return ((x >= lower) & (x <= upper)).to(x.dtype)

    @staticmethod
    def _lower_tolerance(
        x: torch.Tensor,
        lower: float,
        margin: float,
        value_at_margin: float = 0.5,
    ) -> torch.Tensor:
        in_bounds = x >= lower
        if margin <= 0:
            return in_bounds.to(dtype=x.dtype)
        d = torch.clamp((lower - x) / margin, min=0.0)
        scale = 1.0 - value_at_margin
        value = torch.clamp(1.0 - d * scale, min=0.0)
        return torch.where(in_bounds, torch.ones_like(value), value)

    def predict_height(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        height_residual = 0.02 * torch.tanh(self.height_head(state).squeeze(-1))
        return state[:, self.HEIGHT_IDX] + height_residual

    def predict_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Environment-aligned reward under the current DMControl wrapper.

        The DMControl environment wrapper applies each action for two internal
        simulator steps and accumulates both rewards into one environment step.
        The explicit reward therefore needs to match that two-step scale.
        """
        if next_state is None:
            next_state = self.forward(state, action)
        height = self.predict_height(next_state)
        x_dot = next_state[:, self.VX_IDX]

        standing = self._tolerance_interval(height, 0.6, 2.0)
        hopping = self._lower_tolerance(x_dot, lower=self.hop_speed, margin=self.hop_speed / 2.0, value_at_margin=0.5)

        return self.ENV_ACTION_REPEAT * standing * hopping

    def predict_viable_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Viability-aware reward for hybrid/explicit/planning use.

        Multiplies the true reward by a stability gate that approximates
        the probability of surviving beyond the current step.  This
        compensates for short planning horizons that cannot foresee falls.
        """
        if next_state is None:
            next_state = self.forward(state, action)
        true_reward = self.predict_reward(state, action, next_state)
        height = self.predict_height(next_state)
        pitch = next_state[:, self.PITCH_IDX]
        support = next_state[:, self.SUPPORT_IDX]
        clearance = next_state[:, self.CLEARANCE_IDX]

        height_gate = self._lower_tolerance(height, lower=0.45, margin=0.15, value_at_margin=0.0)
        upright = 0.5 * (1.0 + torch.cos(pitch))
        pitch_gate = torch.clamp((upright - 0.35) / 0.45, min=0.0, max=1.0)
        support_gate = torch.clamp((support - 0.08) / 0.22, min=0.0, max=1.0)
        clearance_gate = torch.clamp((clearance + 0.02) / 0.10, min=0.0, max=1.0)
        contact_gate = torch.maximum(support_gate, clearance_gate)
        viability = height_gate * pitch_gate * contact_gate

        return viability * true_reward

    def predict_plan_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Planning reward that prefers hop-relevant transitions.

        The earlier shaping over-rewarded safe, high-contact states with small
        forward drift. This version keeps a viability gate, but now also
        distinguishes three sub-regimes that Phase 0 exposed as behaviorally
        important: low-body forward crawl, compression arming, and release into
        takeoff. The intent is to reward the recovery sequence, not only the
        final airborne state.
        """
        if next_state is None:
            next_state = self.forward(state, action)

        true_reward = self.predict_reward(state, action, next_state)
        height = self.predict_height(next_state)
        x_dot = next_state[:, self.VX_IDX]
        z_dot = next_state[:, self.VZ_IDX]
        pitch = next_state[:, self.PITCH_IDX]
        prev_pitch = state[:, self.PITCH_IDX]
        compression = torch.clamp(next_state[:, self.COMP_IDX], min=0.0)
        support = torch.clamp(next_state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        prev_support = torch.clamp(state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        clearance = next_state[:, self.CLEARANCE_IDX]
        prev_clearance = state[:, self.CLEARANCE_IDX]
        next_rootz = next_state[:, self.ROOTZ_IDX]
        front_anchor = torch.clamp(next_state[:, self.FRONT_ANCHOR_IDX], min=0.0, max=1.0)
        prev_front_anchor = torch.clamp(state[:, self.FRONT_ANCHOR_IDX], min=0.0, max=1.0)

        # v4 asymmetric-contact viability: replace symmetric cos(pitch) gate with a
        # two-lane signed composition. Phase 0 showed that successful trajectories
        # occupy either (a) forward-leaning crawl (pitch > 0, body low) or (b)
        # recovered upright hopping (body high, |pitch| small). Failure is
        # unambiguously "body low AND pitch negative" (rear-trap). The gate
        # therefore admits lane (a) via forward_lean and lane (b) via
        # recovered_upright, taking their elementwise max, then subtracts
        # rear-trap probability multiplicatively and via an explicit penalty.
        height_gate = self._lower_tolerance(height, lower=0.45, margin=0.15, value_at_margin=0.0)
        body_low = torch.sigmoid((self.V4_ROOTZ_LOW_CENTER - next_rootz) / self.V4_ROOTZ_LOW_SCALE)
        body_high = 1.0 - body_low
        forward_lean = torch.sigmoid((pitch - self.V4_PITCH_FWD_CENTER) / self.V4_PITCH_FWD_SCALE)
        recovered_upright = body_high * torch.sigmoid((1.0 - torch.abs(pitch)) / self.V4_PITCH_NEUTRAL_SPREAD)
        viable_lane = torch.maximum(forward_lean, recovered_upright)
        rear_trap = body_low * torch.sigmoid(-(pitch - self.V4_PITCH_BACK_CENTER) / self.V4_PITCH_BACK_SCALE)
        viability = height_gate * viable_lane * (1.0 - rear_trap)
        safe_gate = height_gate * (1.0 - rear_trap)
        crawl_lane = body_low * forward_lean

        speed_progress = torch.clamp(x_dot / max(self.hop_speed, 1e-6), min=0.0, max=1.0)
        upward_progress = torch.clamp(z_dot / 0.60, min=0.0, max=1.0)
        support_relief = torch.clamp(prev_support - support, min=0.0, max=0.50) / 0.50
        liftoff_progress = torch.clamp((0.18 - support) / 0.18, min=0.0, max=1.0)
        clearance_gain = torch.clamp(clearance - prev_clearance, min=0.0, max=0.20) / 0.20
        sticky_contact = torch.clamp((support - 0.55) / 0.35, min=0.0, max=1.0) * torch.clamp((-z_dot) / 0.35, min=0.0, max=1.0)
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        compression_level = torch.clamp(compression / 0.35, min=0.0, max=1.0)
        release_score = (
            torch.clamp(-action[:, 1], min=0.0, max=1.0)
            + torch.clamp(action[:, 2], min=0.0, max=1.0)
            + torch.clamp(action[:, 3], min=0.0, max=1.0)
        ) / 3.0
        compression_drive = (
            torch.clamp(action[:, 1], min=0.0, max=1.0)
            + torch.clamp(-action[:, 2], min=0.0, max=1.0)
            + torch.clamp(-action[:, 3], min=0.0, max=1.0)
        ) / 3.0
        release_liftoff = torch.maximum(upward_progress, liftoff_progress)
        pitch_gain = torch.clamp(pitch - prev_pitch, min=0.0, max=0.45) / 0.45
        front_anchor_gain = torch.clamp(front_anchor - prev_front_anchor, min=0.0, max=0.35) / 0.35
        unsupported_low = body_low * (1.0 - front_anchor) * torch.sigmoid((self.V4_FRONT_ANCHOR_PITCH_CENTER - pitch) / self.V4_FRONT_ANCHOR_PITCH_SCALE)

        positive = viability * (
            0.10 * speed_progress
            + 0.12 * upward_progress
            + 0.14 * support_relief
            + 0.08 * liftoff_progress
            + 0.08 * clearance_gain
        )
        # v4.1: Phase 0 showed that successful runs often traverse a low-body,
        # forward-leaning crawl before recovering into lift-off. Reward this
        # sequence explicitly so the planner does not need to infer it only from
        # sparse final hopping reward.
        crawl_progress = safe_gate * front_anchor * grounded * speed_progress
        arming_progress = safe_gate * front_anchor * grounded * compression_level * compression_drive
        release_progress = safe_gate * front_anchor * compression_level * release_score * release_liftoff
        rear_escape_progress = height_gate * (0.7 * front_anchor + 0.3 * front_anchor_gain) * pitch_gain
        # v4: add an explicit penalty for entering the rear-trap basin so that the
        # planner actively avoids it, not only dampens reward there.
        penalty = 0.10 * sticky_contact + 0.20 * rear_trap + 0.12 * unsupported_low + 0.06 * crawl_lane * (1.0 - front_anchor)

        return (
            true_reward
            + positive
            + 0.10 * crawl_progress
            + 0.08 * arming_progress
            + 0.12 * release_progress
            + 0.10 * rear_escape_progress
            - penalty
        )

    def get_learned_params(self) -> Dict[str, object]:
        return {
            "dt": self.dt,
            "hop_speed": self.hop_speed,
            "base_control_gears": self.base_control_gears.detach().cpu().tolist(),
            "control_scale": self.control_scale.detach().cpu().tolist(),
            "joint_neutral_pose": self.joint_neutral_pose.detach().cpu().tolist(),
            "rate_damping": self.rate_damping.detach().cpu().tolist(),
            "learned_passive_weight": self.learned_passive_weight,
            "learned_residual_weight": self.learned_residual_weight,
            "passive_scale": self.passive_scale,
            "residual_scale": self.residual_scale,
            "latent_layout": [
                "height", "vx", "vz", "pitch", "pitch_rate", "compression", "compression_rate",
                "swing", "swing_rate", "foot_angle", "foot_rate", "support", "stance_mix",
                "impact", "clearance", "rootz", "front_anchor",
            ],
        }


class DMControlHopperWrapper:
    def __init__(self, model: DMControlHopperSDModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(state, action)

    def predict_next_state(self, state_np: np.ndarray, action_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            latent = self.model.encode_observation(state)
            return self.model(latent, action).squeeze(0).cpu().numpy()

    def predict_reward_np(self, state_np: np.ndarray, action_np: np.ndarray) -> float:
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            latent = self.model.encode_observation(state)
            return float(self.model.predict_reward(latent, action).squeeze().cpu().item())
