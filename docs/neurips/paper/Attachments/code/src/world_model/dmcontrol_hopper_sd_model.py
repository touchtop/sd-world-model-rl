"""Compact grey-box SD world model for DMControl Hopper-Hop."""

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


def _touch_raw_to_contact_np(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-np.clip(x, a_min=0.0, a_max=None))


def _touch_raw_to_contact_torch(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-torch.clamp(x, min=0.0))


def _inv_softplus_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


class DMControlHopperSDModel(nn.Module):
    """Compact structured SD core for ``dm_control.suite.hopper.hop``.

    The explicit tensor carries:

    1. Main stocks: ``H, Vx, Vz, Theta, Omega, A``.
    2. Auxiliary observed quantities needed for geometry/contact updates:
       ``waist, hip, knee, ankle, waist_dot, hip_dot, knee_dot, ankle_dot,
       touch_toe, touch_heel, z_min``.

    This keeps the SD core interpretable while preserving enough structure for
    rollout-time state judgement and shaping.
    """

    RAW_OBS_DIM = 15
    ACTION_DIM = 4
    STATE_DIM = 17
    ENV_ACTION_REPEAT = 2

    H_IDX = 0
    ROOTZ_IDX = H_IDX
    HEIGHT_IDX = H_IDX
    VX_IDX = 1
    VZ_IDX = 2
    PITCH_IDX = 3
    PITCH_RATE_IDX = 4
    ACTIVATION_IDX = 5
    WAIST_IDX = 6
    HIP_IDX = 7
    KNEE_IDX = 8
    ANKLE_IDX = 9
    WAIST_RATE_IDX = 10
    HIP_RATE_IDX = 11
    KNEE_RATE_IDX = 12
    ANKLE_RATE_IDX = 13
    TOE_TOUCH_IDX = 14
    HEEL_TOUCH_IDX = 15
    ZMIN_IDX = 16

    QPOS_SLICE = slice(WAIST_IDX, ANKLE_IDX + 1)
    QVEL_SLICE = slice(WAIST_RATE_IDX, ANKLE_RATE_IDX + 1)
    TOUCH_SLICE = slice(TOE_TOUCH_IDX, HEEL_TOUCH_IDX + 1)

    # 15-d raw observation layout from dm_control state observations:
    # [rootz, rooty, waist, hip, knee, ankle,
    #  x_dot, z_dot, rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot,
    #  touch_toe, touch_heel]
    KNEE_OBS_IDX = 4

    # Kinematic constants copied from the previous implementation so trace
    # tooling and geometry remain comparable.
    TORSO_BASE_Z = 1.0
    TORSO_TO_PELVIS_Z = -0.05
    PELVIS_TO_THIGH_Z = -0.20
    THIGH_TO_CALF_Z = -0.33
    CALF_TO_FOOT_Z = -0.32
    TORSO_GEOM_Z0 = -0.05
    TORSO_GEOM_Z1 = 0.20
    TORSO_GEOM_RADIUS = 0.0653
    NOSE_GEOM_X0 = 0.08
    NOSE_GEOM_Z0 = 0.13
    NOSE_GEOM_X1 = 0.15
    NOSE_GEOM_Z1 = 0.14
    NOSE_GEOM_RADIUS = 0.03
    PELVIS_GEOM_Z0 = 0.0
    PELVIS_GEOM_Z1 = -0.15
    PELVIS_GEOM_RADIUS = 0.065
    THIGH_GEOM_Z0 = 0.0
    THIGH_GEOM_Z1 = -0.33
    THIGH_GEOM_RADIUS = 0.04
    CALF_GEOM_Z0 = 0.0
    CALF_GEOM_Z1 = -0.32
    CALF_GEOM_RADIUS = 0.03
    FOOT_GEOM_X0 = -0.08
    FOOT_GEOM_Z0 = 0.0
    FOOT_GEOM_X1 = 0.17
    FOOT_GEOM_Z1 = 0.0
    FOOT_GEOM_RADIUS = 0.04
    TOE_OFFSET_X = 0.17
    HEEL_OFFSET_X = -0.08
    NOSE_CENTER_X = 0.5 * (NOSE_GEOM_X0 + NOSE_GEOM_X1)
    NOSE_CENTER_Z = 0.5 * (NOSE_GEOM_Z0 + NOSE_GEOM_Z1)

    STATE_GROUND_EPS = 0.03
    STATE_BODY_COLLISION_EPS = 0.0
    STATE_VZ_EPS = 0.05
    STATE_SUPPORT_TOUCH = 0.30
    STATE_REAR_PITCH_MAX = -0.75
    STATE_PRONE_PITCH_MIN = 0.75
    STATE_UPRIGHT_PITCH_LOW = -0.45
    STATE_UPRIGHT_PITCH_HIGH = 0.45
    STATE_UPRIGHT_HEIGHT_MIN = 0.32
    STATE_UPRIGHT_OMEGA_MAX = 3.5
    STATE_PREPARED_VZ_MAX = 0.40
    STATE_RISING_VZ_MIN = 0.05

    REAR_TO_UPRIGHT_REWARD_WEIGHT = 0.28
    PRONE_TO_UPRIGHT_REWARD_WEIGHT = 0.28
    UPRIGHT_TO_PREPARED_REWARD_WEIGHT = 0.40
    PREPARED_TO_RISING_REWARD_WEIGHT = 0.45
    FALLING_TO_UPRIGHT_REWARD_WEIGHT = 0.50
    FALLING_TO_PREPARED_REWARD_WEIGHT = 0.75

    PRIOR_GROUNDED_MIN = 0.80
    REAR_LYING_PITCH_MAX = -1.50
    PRONE_LYING_PITCH_MIN = 0.90
    PRIOR_KNEE_TARGET = 1.55
    PRIOR_FOOT_SUPPORT_TARGET = 0.99
    REAR_PROGRESS_SCALE = 0.85
    PRONE_SEVERITY_SCALE = 0.65

    KNEE_READY_LOW = 1.45
    KNEE_READY_HIGH = 1.95
    KNEE_READY_LOW_SCALE = 0.08
    KNEE_READY_HIGH_SCALE = 0.12

    CONTACT_MARGIN = 0.04
    CONTACT_SCALE = 0.012
    CONTACT_CARRY = 0.45
    CONTACT_CARRY_RISE = 0.10
    CONTACT_CARRY_FALL = 0.55
    CONTACT_VEL_LOW = 0.20
    CONTACT_VEL_SCALE = 0.18
    TOE_TOUCH_TARGET_MARGIN = 0.18
    HEEL_TOUCH_TARGET_MARGIN = 0.28
    TOUCH_TARGET_SCALE = 0.06
    TOUCH_TARGET_LOCAL_VEL_LOW = 0.02
    TOUCH_TARGET_LOCAL_VEL_SCALE = 0.10
    TOUCH_TARGET_STATIC_VEL_BAND = 0.10
    TOUCH_TARGET_STATIC_VEL_SCALE = 0.06
    GROUND_NORMAL_STIFFNESS = 420.0
    GROUND_NORMAL_DAMPING = 28.0
    GROUND_FRICTION_COEFF = 0.85
    GROUND_SLIP_V_SCALE = 0.35
    GROUND_SUPPORT_VZ_GAIN = 6.0
    GROUND_SUPPORT_GATE_CENTER = 0.20
    GROUND_SUPPORT_GATE_SCALE = 0.10
    GROUND_SUPPORT_SHARE_BLEND = 0.40
    GROUND_SUPPORT_PROJ_SCALE = 0.35
    AIRBORNE_ROOT_X_COUPLING = (0.10, -0.14, 0.08, -0.04)
    AIRBORNE_ROOT_TORQUE_COUPLING = (0.55, -0.75, 0.45, -0.20)
    TRANSITION_RESIDUAL_BASE_SCALE = (0.10, 0.18, 0.30)
    TOUCH_TRANSITION_RESIDUAL_BASE_SCALE = (0.05, 0.07)
    TRANSITION_RESIDUAL_EXTRA_DIM = 34
    DENSE_DOWN_V_CENTER = 0.10
    DENSE_RISE_V_CENTER = 0.10
    DENSE_V_SCALE = 0.10
    DENSE_LANDING_VZ_TARGET = -0.45
    DENSE_GROUND_VZ_TARGET = 0.0
    DENSE_RISE_VZ_TARGET = 0.55
    DENSE_LANDING_KNEE_TARGET = 1.55
    DENSE_GROUND_KNEE_TARGET = 1.65
    DENSE_RISE_KNEE_TARGET = 1.15
    DENSE_THETA_SIGMA = 0.35
    DENSE_OMEGA_SIGMA = 2.2
    DENSE_KNEE_SIGMA = 0.28
    DENSE_VX_SIGMA = 1.20
    DENSE_VZ_SIGMA = 0.30
    DENSE_HEIGHT_TARGET = 0.55
    DENSE_HEIGHT_SIGMA = 0.18
    DENSE_SUPPORT_SIGMA = 0.20
    DENSE_BODY_SIGMA = 0.20
    PRONE_RECOVER_GEOM_SCALE = 0.60
    PRONE_RECOVER_PITCH_SIGMA = 0.45
    PRONE_RECOVER_SPIN_SIGMA = 4.0
    PRONE_RECOVER_GROUND_SOFT_SCALE = 0.015
    PRONE_RECOVER_FOOT_SOFT_SCALE = 0.03
    PRONE_RECOVER_BODY_CLEAR_CENTER = 0.02
    PRONE_RECOVER_BODY_CLEAR_SCALE = 0.03
    PRONE_RECOVER_PROGRESS_WEIGHT = 0.20
    PRONE_RECOVER_CORE_ABS_WEIGHT = 0.05
    PRONE_RECOVER_TORSO_OVER_FOOT_SIGMA = 0.35
    PRONE_RECOVER_TORSO_Z_CENTER = 0.65
    PRONE_RECOVER_TORSO_Z_SCALE = 0.10
    PRONE_RECOVER_KNEE_LOW = 1.20
    PRONE_RECOVER_KNEE_HIGH = 2.10
    PRONE_RECOVER_KNEE_LOW_SCALE = 0.12
    PRONE_RECOVER_KNEE_HIGH_SCALE = 0.12
    STATE_NAMES = (
        "falling_air",
        "grounded_rear",
        "grounded_prone",
        "grounded_upright",
        "grounded_prepared",
        "rising_air",
    )
    STAGE_MODES = ("full", "landing", "ground", "rise", "prone_recover")

    def __init__(
        self,
        dt: float = 0.02,
        hidden: int = 256,
        hop_speed: float = 2.0,
        residual_scale: float = 5.0,
        passive_scale: float = 10.0,
        action_prior_enabled: bool = False,
        shaping_scale: float = 1.0,
        dense_pose_delta_weight: float = 0.40,
        dense_pose_abs_weight: float = 0.0,
        dense_pose_milestone_scale: float = 0.25,
        dense_pose_gamma: float = 0.99,
        recover_penalty_scale: float = 0.05,
        stage_mode: str = "full",
        use_transition_residual: bool = False,
        transition_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.dt = dt
        self.hop_speed = hop_speed
        self.action_prior_enabled = action_prior_enabled
        self.shaping_scale = float(shaping_scale)
        self.dense_pose_delta_weight = float(dense_pose_delta_weight)
        self.dense_pose_abs_weight = float(dense_pose_abs_weight)
        self.dense_pose_milestone_scale = float(dense_pose_milestone_scale)
        self.dense_pose_gamma = float(dense_pose_gamma)
        self.recover_penalty_scale = float(recover_penalty_scale)
        self.stage_mode = str(stage_mode)
        self.use_transition_residual = bool(use_transition_residual)
        self.transition_residual_scale = float(transition_residual_scale)
        if self.stage_mode not in self.STAGE_MODES:
            raise ValueError(f"Unknown Hopper stage_mode: {self.stage_mode}")
        _ = hidden, residual_scale, passive_scale

        self.register_buffer("gravity", torch.tensor(9.81, dtype=torch.float32))
        self.register_buffer("mass", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("inertia", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("rest_leg_length", torch.tensor(0.82, dtype=torch.float32))
        self.register_buffer(
            "joint_rest",
            torch.tensor([0.0, 0.0, 1.55, 0.0], dtype=torch.float32),
        )

        self.activation_w = nn.Parameter(torch.tensor([0.20, 0.20, 0.35, 0.15], dtype=torch.float32))
        self.activation_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.activation_tau_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(0.08, dtype=torch.float32)))

        self.leg_stiffness_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(6.0, dtype=torch.float32)))
        self.leg_damping_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(0.8, dtype=torch.float32)))
        self.leg_force_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(3.0, dtype=torch.float32)))

        self.contact_k_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(10.0, dtype=torch.float32)))
        self.contact_tau = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

        self.linear_damping_raw = nn.Parameter(
            _inv_softplus_torch(torch.tensor([0.10, 0.18], dtype=torch.float32))
        )
        self.angular_damping_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(0.12, dtype=torch.float32)))
        self.balance_theta_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(1.40, dtype=torch.float32)))
        self.balance_omega_raw = nn.Parameter(_inv_softplus_torch(torch.tensor(0.35, dtype=torch.float32)))

        self.joint_action_scale = nn.Parameter(torch.tensor([1.1, 1.2, 1.4, 0.9], dtype=torch.float32))
        self.joint_rate_damping_raw = nn.Parameter(
            _inv_softplus_torch(torch.tensor([1.0, 1.0, 1.2, 0.8], dtype=torch.float32))
        )
        self.joint_centering_raw = nn.Parameter(
            _inv_softplus_torch(torch.tensor([0.20, 0.20, 0.35, 0.20], dtype=torch.float32))
        )
        if self.use_transition_residual:
            residual_hidden = max(32, min(int(hidden), 128))
            self.transition_residual_head = nn.Sequential(
                nn.Linear(
                    self.STATE_DIM + self.ACTION_DIM + self.TRANSITION_RESIDUAL_EXTRA_DIM,
                    residual_hidden,
                ),
                nn.SiLU(),
                nn.Linear(residual_hidden, 3),
            )
            self.touch_transition_residual_head = nn.Sequential(
                nn.Linear(
                    self.STATE_DIM + self.ACTION_DIM + self.TRANSITION_RESIDUAL_EXTRA_DIM,
                    residual_hidden,
                ),
                nn.SiLU(),
                nn.Linear(residual_hidden, 2),
            )
            nn.init.zeros_(self.transition_residual_head[-1].weight)
            nn.init.zeros_(self.transition_residual_head[-1].bias)
            nn.init.zeros_(self.touch_transition_residual_head[-1].weight)
            nn.init.zeros_(self.touch_transition_residual_head[-1].bias)
        else:
            self.transition_residual_head = None
            self.touch_transition_residual_head = None
        self.register_buffer(
            "transition_residual_base_scale",
            torch.tensor(self.TRANSITION_RESIDUAL_BASE_SCALE, dtype=torch.float32),
        )
        self.register_buffer(
            "touch_transition_residual_base_scale",
            torch.tensor(self.TOUCH_TRANSITION_RESIDUAL_BASE_SCALE, dtype=torch.float32),
        )

    @property
    def activation_tau(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.activation_tau_raw) + 1e-4

    @property
    def leg_stiffness(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.leg_stiffness_raw)

    @property
    def leg_damping(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.leg_damping_raw)

    @property
    def leg_force_max(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.leg_force_raw)

    @property
    def contact_k(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.contact_k_raw)

    @property
    def linear_damping(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.linear_damping_raw)

    @property
    def angular_damping(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.angular_damping_raw)

    @property
    def balance_theta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.balance_theta_raw)

    @property
    def balance_omega(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.balance_omega_raw)

    @property
    def joint_rate_damping(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.joint_rate_damping_raw)

    @property
    def joint_centering(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.joint_centering_raw)

    @classmethod
    def flatten_obs(cls, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            np.asarray(obs["position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["velocity"], dtype=np.float32).reshape(-1),
            np.asarray(obs["touch"], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)

    @classmethod
    def obs_to_state(cls, obs: Union[Mapping[str, np.ndarray], np.ndarray, Sequence[float]]) -> np.ndarray:
        if isinstance(obs, Mapping):
            pos = np.asarray(obs["position"], dtype=np.float32).reshape(6)
            vel = np.asarray(obs["velocity"], dtype=np.float32).reshape(7)
            touch = np.asarray(obs["touch"], dtype=np.float32).reshape(2)
        else:
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            if flat.shape[0] != cls.RAW_OBS_DIM:
                raise ValueError(f"Expected hopper flat observation of size {cls.RAW_OBS_DIM}, got {flat.shape[0]}")
            pos = flat[:6]
            vel = flat[6:13]
            touch = flat[13:]
        pos = pos.copy()
        pos[1:] = _wrap_angle_np(pos[1:])
        touch = _touch_raw_to_contact_np(np.asarray(touch, dtype=np.float32))
        return np.concatenate([pos, vel, touch], dtype=np.float32)

    @staticmethod
    def _rotate_local(x_local: torch.Tensor, z_local: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_world = cos_a * x_local + sin_a * z_local
        z_world = -sin_a * x_local + cos_a * z_local
        return x_world, z_world

    def _absolute_angles(
        self,
        theta: torch.Tensor,
        waist: torch.Tensor,
        hip: torch.Tensor,
        knee: torch.Tensor,
        ankle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torso = theta
        pelvis = torso + waist
        thigh = pelvis + hip
        calf = thigh + knee
        foot = calf + ankle
        return torso, pelvis, thigh, calf, foot

    def _segment_lower_z(
        self,
        origin_x: torch.Tensor,
        origin_z: torch.Tensor,
        angle: torch.Tensor,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        radius: float,
    ) -> torch.Tensor:
        dx0, dz0 = self._rotate_local(
            torch.full_like(origin_z, x0),
            torch.full_like(origin_z, z0),
            angle,
        )
        dx1, dz1 = self._rotate_local(
            torch.full_like(origin_z, x1),
            torch.full_like(origin_z, z1),
            angle,
        )
        del origin_x
        lower = torch.minimum(origin_z + dz0, origin_z + dz1)
        return lower - radius

    def _segment_lower_point(
        self,
        origin_x: torch.Tensor,
        origin_z: torch.Tensor,
        angle: torch.Tensor,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx0, dz0 = self._rotate_local(
            torch.full_like(origin_z, x0),
            torch.full_like(origin_z, z0),
            angle,
        )
        dx1, dz1 = self._rotate_local(
            torch.full_like(origin_z, x1),
            torch.full_like(origin_z, z1),
            angle,
        )
        xw0 = origin_x + dx0
        zw0 = origin_z + dz0
        xw1 = origin_x + dx1
        zw1 = origin_z + dz1
        use_first = zw0 <= zw1
        lower_x = torch.where(use_first, xw0, xw1)
        lower_z = torch.where(use_first, zw0, zw1) - radius
        return lower_x, lower_z

    def _kinematic_features(
        self,
        torso_z: torch.Tensor,
        theta: torch.Tensor,
        waist: torch.Tensor,
        hip: torch.Tensor,
        knee: torch.Tensor,
        ankle: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        torso, pelvis, thigh, calf, foot = self._absolute_angles(theta, waist, hip, knee, ankle)

        torso_x = torch.zeros_like(torso_z)
        pelvis_dx, pelvis_dz = self._rotate_local(
            torch.zeros_like(torso_z),
            torch.full_like(torso_z, self.TORSO_TO_PELVIS_Z),
            torso,
        )
        pelvis_x = torso_x + pelvis_dx
        pelvis_z = torso_z + pelvis_dz

        thigh_dx, thigh_dz = self._rotate_local(
            torch.zeros_like(torso_z),
            torch.full_like(torso_z, self.PELVIS_TO_THIGH_Z),
            pelvis,
        )
        thigh_x = pelvis_x + thigh_dx
        thigh_z = pelvis_z + thigh_dz

        calf_dx, calf_dz = self._rotate_local(
            torch.zeros_like(torso_z),
            torch.full_like(torso_z, self.THIGH_TO_CALF_Z),
            thigh,
        )
        calf_x = thigh_x + calf_dx
        calf_z = thigh_z + calf_dz

        foot_dx, foot_dz = self._rotate_local(
            torch.zeros_like(torso_z),
            torch.full_like(torso_z, self.CALF_TO_FOOT_Z),
            calf,
        )
        foot_x = calf_x + foot_dx
        foot_z = calf_z + foot_dz

        toe_dx, toe_dz = self._rotate_local(
            torch.full_like(torso_z, self.TOE_OFFSET_X),
            torch.zeros_like(torso_z),
            foot,
        )
        heel_dx, heel_dz = self._rotate_local(
            torch.full_like(torso_z, self.HEEL_OFFSET_X),
            torch.zeros_like(torso_z),
            foot,
        )
        toe_x = foot_x + toe_dx
        toe_z = foot_z + toe_dz
        heel_x = foot_x + heel_dx
        heel_z = foot_z + heel_dz
        nose_dx, nose_dz = self._rotate_local(
            torch.full_like(torso_z, self.NOSE_CENTER_X),
            torch.full_like(torso_z, self.NOSE_CENTER_Z),
            torso,
        )
        nose_x = torso_x + nose_dx
        nose_z = torso_z + nose_dz

        torso_lower_x, torso_lower_z = self._segment_lower_point(
            torso_x, torso_z, torso,
            0.0, self.TORSO_GEOM_Z0, 0.0, self.TORSO_GEOM_Z1, self.TORSO_GEOM_RADIUS,
        )
        nose_lower_x, nose_lower_z = self._segment_lower_point(
            torso_x, torso_z, torso,
            self.NOSE_GEOM_X0, self.NOSE_GEOM_Z0, self.NOSE_GEOM_X1, self.NOSE_GEOM_Z1, self.NOSE_GEOM_RADIUS,
        )
        pelvis_lower_x, pelvis_lower_z = self._segment_lower_point(
            pelvis_x, pelvis_z, pelvis,
            0.0, self.PELVIS_GEOM_Z0, 0.0, self.PELVIS_GEOM_Z1, self.PELVIS_GEOM_RADIUS,
        )
        thigh_lower_x, thigh_lower_z = self._segment_lower_point(
            thigh_x, thigh_z, thigh,
            0.0, self.THIGH_GEOM_Z0, 0.0, self.THIGH_GEOM_Z1, self.THIGH_GEOM_RADIUS,
        )
        calf_lower_x, calf_lower_z = self._segment_lower_point(
            calf_x, calf_z, calf,
            0.0, self.CALF_GEOM_Z0, 0.0, self.CALF_GEOM_Z1, self.CALF_GEOM_RADIUS,
        )
        foot_lower_x, foot_lower_z = self._segment_lower_point(
            foot_x, foot_z, foot,
            self.FOOT_GEOM_X0, self.FOOT_GEOM_Z0, self.FOOT_GEOM_X1, self.FOOT_GEOM_Z1, self.FOOT_GEOM_RADIUS,
        )

        all_contact_z = torch.stack(
            [
                torso_lower_z,
                nose_lower_z,
                pelvis_lower_z,
                thigh_lower_z,
                calf_lower_z,
                foot_lower_z,
                toe_z,
                heel_z,
            ],
            dim=-1,
        )
        z_min = all_contact_z.min(dim=-1).values
        x_all = torch.stack(
            [
                nose_x,
                torso_x,
                pelvis_x,
                thigh_x,
                calf_x,
                foot_x,
                toe_x,
                heel_x,
            ],
            dim=-1,
        )
        x_min = x_all.min(dim=-1).values
        x_max = x_all.max(dim=-1).values

        return {
            "torso_x": torso_x,
            "torso_z": torso_z,
            "nose_x": nose_x,
            "nose_z": nose_z,
            "pelvis_x": pelvis_x,
            "pelvis_z": pelvis_z,
            "thigh_x": thigh_x,
            "thigh_z": thigh_z,
            "calf_x": calf_x,
            "calf_z": calf_z,
            "foot_x": foot_x,
            "foot_z": foot_z,
            "toe_x": toe_x,
            "toe_z": toe_z,
            "heel_x": heel_x,
            "heel_z": heel_z,
            "torso_lower_x": torso_lower_x,
            "torso_lower_z": torso_lower_z,
            "nose_lower_x": nose_lower_x,
            "nose_lower_z": nose_lower_z,
            "pelvis_lower_x": pelvis_lower_x,
            "pelvis_lower_z": pelvis_lower_z,
            "thigh_lower_x": thigh_lower_x,
            "thigh_lower_z": thigh_lower_z,
            "calf_lower_x": calf_lower_x,
            "calf_lower_z": calf_lower_z,
            "foot_lower_x": foot_lower_x,
            "foot_lower_z": foot_lower_z,
            "x_min": x_min,
            "x_max": x_max,
            "z_min": z_min,
            "hip_x": pelvis_x,
            "hip_z": pelvis_z,
            "foot_angle": foot,
        }

    def _contact_gate(
        self,
        touch_toe: torch.Tensor,
        touch_heel: torch.Tensor,
        kin: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        toe_touch_gate = torch.sigmoid(self.contact_k * (touch_toe - self.contact_tau))
        heel_touch_gate = torch.sigmoid(self.contact_k * (touch_heel - self.contact_tau))
        toe_geom_gate = torch.sigmoid((self.CONTACT_MARGIN - kin["toe_z"]) / self.CONTACT_SCALE)
        heel_geom_gate = torch.sigmoid((self.CONTACT_MARGIN - kin["heel_z"]) / self.CONTACT_SCALE)
        toe_gate = toe_touch_gate * toe_geom_gate
        heel_gate = heel_touch_gate * heel_geom_gate
        return torch.maximum(toe_gate, heel_gate)

    @staticmethod
    def _asymmetric_contact_update(
        prev_touch: torch.Tensor,
        target_touch: torch.Tensor,
        rise_carry: float,
        fall_carry: float,
    ) -> torch.Tensor:
        carry = torch.where(
            target_touch >= prev_touch,
            torch.full_like(prev_touch, rise_carry),
            torch.full_like(prev_touch, fall_carry),
        )
        return torch.clamp(carry * prev_touch + (1.0 - carry) * target_touch, min=0.0, max=1.0)

    def _touch_geom_gate(
        self,
        point_z: torch.Tensor,
        margin: float,
        scale: float,
    ) -> torch.Tensor:
        return torch.sigmoid((margin - point_z) / max(scale, 1e-6))

    def _touch_onset_gate(
        self,
        point_z: torch.Tensor,
        point_vz: torch.Tensor,
        root_vz: torch.Tensor,
        relaxed_margin: float,
    ) -> torch.Tensor:
        strict_near = self._touch_geom_gate(point_z, self.CONTACT_MARGIN, self.CONTACT_SCALE)
        relaxed_near = self._touch_geom_gate(point_z, relaxed_margin, self.TOUCH_TARGET_SCALE)
        local_downward = torch.sigmoid(
            (-point_vz - self.TOUCH_TARGET_LOCAL_VEL_LOW) / self.TOUCH_TARGET_LOCAL_VEL_SCALE
        )
        root_downward = torch.sigmoid((-root_vz - self.CONTACT_VEL_LOW) / self.CONTACT_VEL_SCALE)
        motion_gate = torch.maximum(local_downward, root_downward)
        return torch.maximum(strict_near, relaxed_near * motion_gate)

    def _contact_point(
        self,
        touch_toe: torch.Tensor,
        touch_heel: torch.Tensor,
        kin: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        toe = torch.clamp(touch_toe, min=0.0, max=1.0)
        heel = torch.clamp(touch_heel, min=0.0, max=1.0)
        denom = toe + heel + 1e-6
        toe_w = toe / denom
        heel_w = heel / denom
        default_x = 0.5 * (kin["toe_x"] + kin["heel_x"])
        default_z = 0.5 * (kin["toe_z"] + kin["heel_z"])
        use_default = (toe + heel) < 1e-4
        px = torch.where(
            use_default,
            default_x,
            toe_w * kin["toe_x"] + heel_w * kin["heel_x"],
        )
        pz = torch.where(
            use_default,
            default_z,
            toe_w * kin["toe_z"] + heel_w * kin["heel_z"],
        )
        return px, pz

    def _leg_mechanics(
        self,
        touch_toe: torch.Tensor,
        touch_heel: torch.Tensor,
        kin: Dict[str, torch.Tensor],
        next_kin_nominal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        p_contact_x, p_contact_z = self._contact_point(touch_toe, touch_heel, kin)
        r_leg_x = kin["hip_x"] - p_contact_x
        r_leg_z = kin["hip_z"] - p_contact_z
        leg_length = torch.sqrt(r_leg_x.square() + r_leg_z.square() + 1e-6)
        c_leg = torch.clamp(self.rest_leg_length - leg_length, min=0.0)
        if next_kin_nominal is None:
            d_c_leg = torch.zeros_like(c_leg)
        else:
            next_contact_x_free, next_contact_z_free = self._contact_point(touch_toe, touch_heel, next_kin_nominal)
            # During a grounded step the stance contact is approximately world-fixed
            # over one integration interval; letting the contact point move with the
            # next foot geometry underestimates leg compression and support.
            contact_anchor = torch.clamp(torch.maximum(touch_toe, touch_heel), min=0.0, max=1.0)
            next_contact_x = contact_anchor * p_contact_x + (1.0 - contact_anchor) * next_contact_x_free
            next_contact_z = contact_anchor * p_contact_z + (1.0 - contact_anchor) * next_contact_z_free
            next_r_leg_x = next_kin_nominal["hip_x"] - next_contact_x
            next_r_leg_z = next_kin_nominal["hip_z"] - next_contact_z
            next_leg_length = torch.sqrt(next_r_leg_x.square() + next_r_leg_z.square() + 1e-6)
            next_c_leg = torch.clamp(self.rest_leg_length - next_leg_length, min=0.0)
            d_c_leg = (next_c_leg - c_leg) / max(self.dt * self.ENV_ACTION_REPEAT, 1e-6)
        e_leg_x = r_leg_x / (leg_length + 1e-6)
        e_leg_z = r_leg_z / (leg_length + 1e-6)
        return {
            "p_contact_x": p_contact_x,
            "p_contact_z": p_contact_z,
            "r_leg_x": r_leg_x,
            "r_leg_z": r_leg_z,
            "leg_length": leg_length,
            "c_leg": c_leg,
            "d_c_leg": d_c_leg,
            "e_leg_x": e_leg_x,
            "e_leg_z": e_leg_z,
        }

    def _contact_reaction(
        self,
        touch_toe: torch.Tensor,
        touch_heel: torch.Tensor,
        kin: Dict[str, torch.Tensor],
        next_kin_nominal: Dict[str, torch.Tensor],
        vx: torch.Tensor,
        vz: torch.Tensor,
        leg_drive_x: torch.Tensor,
        leg_drive_z: torch.Tensor,
        step_dt: float,
    ) -> Dict[str, torch.Tensor]:
        toe_touch_gate = torch.sigmoid(self.contact_k * (touch_toe - self.contact_tau))
        heel_touch_gate = torch.sigmoid(self.contact_k * (touch_heel - self.contact_tau))
        toe_geom_gate = torch.sigmoid((self.CONTACT_MARGIN - kin["toe_z"]) / self.CONTACT_SCALE)
        heel_geom_gate = torch.sigmoid((self.CONTACT_MARGIN - kin["heel_z"]) / self.CONTACT_SCALE)
        toe_gate = toe_touch_gate * toe_geom_gate
        heel_gate = heel_touch_gate * heel_geom_gate

        toe_pen = torch.relu(self.CONTACT_MARGIN - kin["toe_z"])
        heel_pen = torch.relu(self.CONTACT_MARGIN - kin["heel_z"])
        toe_vz = (next_kin_nominal["toe_z"] - kin["toe_z"]) / max(step_dt, 1e-6)
        heel_vz = (next_kin_nominal["heel_z"] - kin["heel_z"]) / max(step_dt, 1e-6)
        toe_vx = vx + (next_kin_nominal["toe_x"] - kin["toe_x"]) / max(step_dt, 1e-6)
        heel_vx = vx + (next_kin_nominal["heel_x"] - kin["heel_x"]) / max(step_dt, 1e-6)

        toe_normal_base = toe_gate * torch.relu(
            self.GROUND_NORMAL_STIFFNESS * toe_pen - self.GROUND_NORMAL_DAMPING * toe_vz
        )
        heel_normal_base = heel_gate * torch.relu(
            self.GROUND_NORMAL_STIFFNESS * heel_pen - self.GROUND_NORMAL_DAMPING * heel_vz
        )
        contact_weight = toe_gate + heel_gate + 1e-6
        toe_share_contact = toe_gate / contact_weight
        support_mid_x = 0.5 * (kin["toe_x"] + kin["heel_x"])
        support_half_span = 0.5 * torch.abs(kin["toe_x"] - kin["heel_x"]) + 1e-4
        support_proj = (kin["hip_x"] - support_mid_x) / (self.GROUND_SUPPORT_PROJ_SCALE * support_half_span + 1e-6)
        toe_share_proj = torch.sigmoid(support_proj)
        toe_share = (
            (1.0 - self.GROUND_SUPPORT_SHARE_BLEND) * toe_share_contact
            + self.GROUND_SUPPORT_SHARE_BLEND * toe_share_proj
        )
        toe_share = torch.clamp(toe_share, min=0.05, max=0.95)
        heel_share = 1.0 - toe_share

        active_normal = torch.relu(leg_drive_z)
        total_gate = torch.clamp(toe_gate + heel_gate, min=0.0, max=1.0)
        support_gate = torch.sigmoid(
            (total_gate - self.GROUND_SUPPORT_GATE_CENTER) / self.GROUND_SUPPORT_GATE_SCALE
        )
        support_bias = support_gate * torch.relu(self.mass * self.gravity - self.GROUND_SUPPORT_VZ_GAIN * vz)
        toe_normal = toe_share * support_bias + toe_normal_base + toe_share * active_normal
        heel_normal = heel_share * support_bias + heel_normal_base + heel_share * active_normal

        toe_friction = -self.GROUND_FRICTION_COEFF * toe_normal * torch.tanh(toe_vx / self.GROUND_SLIP_V_SCALE)
        heel_friction = -self.GROUND_FRICTION_COEFF * heel_normal * torch.tanh(heel_vx / self.GROUND_SLIP_V_SCALE)
        toe_fx = toe_share * leg_drive_x + toe_friction
        heel_fx = heel_share * leg_drive_x + heel_friction

        total_fx = toe_fx + heel_fx
        total_fz = toe_normal + heel_normal
        toe_tau = (kin["toe_x"] - kin["torso_x"]) * toe_normal - (kin["toe_z"] - kin["torso_z"]) * toe_fx
        heel_tau = (kin["heel_x"] - kin["torso_x"]) * heel_normal - (kin["heel_z"] - kin["torso_z"]) * heel_fx
        return {
            "toe_gate": toe_gate,
            "heel_gate": heel_gate,
            "gate": total_gate,
            "support_gate": support_gate,
            "toe_share": toe_share,
            "heel_share": heel_share,
            "toe_normal": toe_normal,
            "heel_normal": heel_normal,
            "toe_fx": toe_fx,
            "heel_fx": heel_fx,
            "toe_vx": toe_vx,
            "heel_vx": heel_vx,
            "toe_vz": toe_vz,
            "heel_vz": heel_vz,
            "F_x": total_fx,
            "F_z": total_fz,
            "tau_contact": toe_tau + heel_tau,
        }

    @staticmethod
    def _feature_tanh(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.tanh(x / max(scale, 1e-6))

    def _transition_residual_inputs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_activation: torch.Tensor,
        joint_acc: torch.Tensor,
        kin: Dict[str, torch.Tensor],
        leg: Dict[str, torch.Tensor],
        contact: Dict[str, torch.Tensor],
        airborne_gate: torch.Tensor,
        base_delta_vx: torch.Tensor,
        base_delta_vz: torch.Tensor,
        base_delta_omega: torch.Tensor,
    ) -> torch.Tensor:
        # The residual sees the current structured situation and the base
        # model's own proposal, so it can learn the unexplained remainder
        # instead of refitting the entire transition law from raw state alone.
        force_scale = float(self.gravity.item())
        torque_scale = float((self.gravity * self.rest_leg_length).item())
        extra = [
            next_activation.unsqueeze(-1),
            self._feature_tanh(joint_acc, 8.0),
            self._feature_tanh(kin["toe_z"].unsqueeze(-1), 0.25),
            self._feature_tanh(kin["heel_z"].unsqueeze(-1), 0.25),
            self._feature_tanh(kin["hip_x"].unsqueeze(-1), 0.50),
            self._feature_tanh(kin["hip_z"].unsqueeze(-1), 1.00),
            self._feature_tanh(kin["torso_z"].unsqueeze(-1), 1.00),
            self._feature_tanh(kin["z_min"].unsqueeze(-1), 0.25),
            self._feature_tanh(leg["c_leg"].unsqueeze(-1), 0.20),
            self._feature_tanh(leg["d_c_leg"].unsqueeze(-1), 4.0),
            leg["e_leg_x"].unsqueeze(-1),
            leg["e_leg_z"].unsqueeze(-1),
            contact["gate"].unsqueeze(-1),
            contact["support_gate"].unsqueeze(-1),
            contact["toe_gate"].unsqueeze(-1),
            contact["heel_gate"].unsqueeze(-1),
            contact["toe_share"].unsqueeze(-1),
            contact["heel_share"].unsqueeze(-1),
            self._feature_tanh(contact["toe_normal"].unsqueeze(-1), force_scale),
            self._feature_tanh(contact["heel_normal"].unsqueeze(-1), force_scale),
            self._feature_tanh(contact["F_x"].unsqueeze(-1), force_scale),
            self._feature_tanh(contact["F_z"].unsqueeze(-1), force_scale),
            self._feature_tanh(contact["tau_contact"].unsqueeze(-1), torque_scale),
            self._feature_tanh(contact["toe_vx"].unsqueeze(-1), 4.0),
            self._feature_tanh(contact["heel_vx"].unsqueeze(-1), 4.0),
            self._feature_tanh(contact["toe_vz"].unsqueeze(-1), 4.0),
            self._feature_tanh(contact["heel_vz"].unsqueeze(-1), 4.0),
            self._feature_tanh(base_delta_vx.unsqueeze(-1), 0.50),
            self._feature_tanh(base_delta_vz.unsqueeze(-1), 0.50),
            self._feature_tanh(base_delta_omega.unsqueeze(-1), 1.50),
            airborne_gate.unsqueeze(-1),
        ]
        return torch.cat([state, action] + extra, dim=-1)

    def _activation_proxy(
        self,
        touch_toe: torch.Tensor,
        touch_heel: torch.Tensor,
        knee: torch.Tensor,
        qdot: torch.Tensor,
    ) -> torch.Tensor:
        support = torch.maximum(touch_toe, touch_heel)
        motion = torch.linalg.vector_norm(qdot, dim=-1)
        knee_gate = torch.sigmoid((knee - 1.15) / 0.22)
        motion_gate = torch.sigmoid((motion - 1.00) / 0.35)
        return torch.clamp(support * knee_gate * motion_gate, min=0.0, max=1.0)

    def _touch_target(
        self,
        point_z: torch.Tensor,
        point_vz: torch.Tensor,
        root_vz: torch.Tensor,
        relaxed_margin: float,
    ) -> torch.Tensor:
        return self._touch_onset_gate(point_z, point_vz, root_vz, relaxed_margin)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.RAW_OBS_DIM:
            raise ValueError(f"Expected raw hopper observation of size {self.RAW_OBS_DIM}, got {obs.shape[-1]}")

        leading = obs.shape[:-1]
        flat = obs.reshape(-1, obs.shape[-1])

        pos = torch.cat([flat[:, :1], _wrap_angle_torch(flat[:, 1:6])], dim=-1)
        vel = flat[:, 6:13]
        touch = _touch_raw_to_contact_torch(flat[:, 13:15])

        torso_z = self.TORSO_BASE_Z + pos[:, 0]
        theta = pos[:, 1]
        q = pos[:, 2:6]
        qdot = vel[:, 3:7]
        kin = self._kinematic_features(torso_z, theta, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        activation = self._activation_proxy(touch[:, 0], touch[:, 1], q[:, 2], qdot)

        latent = torch.stack(
            [
                torso_z,
                vel[:, 0],
                vel[:, 1],
                theta,
                vel[:, 2],
                activation,
                q[:, 0],
                q[:, 1],
                q[:, 2],
                q[:, 3],
                qdot[:, 0],
                qdot[:, 1],
                qdot[:, 2],
                qdot[:, 3],
                touch[:, 0],
                touch[:, 1],
                kin["z_min"],
            ],
            dim=-1,
        )
        return latent.reshape(*leading, self.STATE_DIM)

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        H = torch.clamp(state[:, self.H_IDX], min=0.0, max=3.5)
        vx = torch.clamp(state[:, self.VX_IDX], min=-10.0, max=10.0)
        vz = torch.clamp(state[:, self.VZ_IDX], min=-12.0, max=12.0)
        theta = _wrap_angle_torch(state[:, self.PITCH_IDX])
        omega = torch.clamp(state[:, self.PITCH_RATE_IDX], min=-20.0, max=20.0)
        activation = torch.clamp(state[:, self.ACTIVATION_IDX], min=0.0, max=1.0)
        waist = _wrap_angle_torch(state[:, self.WAIST_IDX])
        hip = _wrap_angle_torch(state[:, self.HIP_IDX])
        knee = _wrap_angle_torch(state[:, self.KNEE_IDX])
        ankle = _wrap_angle_torch(state[:, self.ANKLE_IDX])
        waist_rate = torch.clamp(state[:, self.WAIST_RATE_IDX], min=-20.0, max=20.0)
        hip_rate = torch.clamp(state[:, self.HIP_RATE_IDX], min=-20.0, max=20.0)
        knee_rate = torch.clamp(state[:, self.KNEE_RATE_IDX], min=-20.0, max=20.0)
        ankle_rate = torch.clamp(state[:, self.ANKLE_RATE_IDX], min=-20.0, max=20.0)
        touch_toe = torch.clamp(state[:, self.TOE_TOUCH_IDX], min=0.0, max=1.0)
        touch_heel = torch.clamp(state[:, self.HEEL_TOUCH_IDX], min=0.0, max=1.0)
        kin = self._kinematic_features(
            H,
            theta,
            waist,
            hip,
            knee,
            ankle,
        )
        z_min = torch.clamp(kin["z_min"], min=-0.5, max=3.0)
        return torch.stack(
            [
                H,
                vx,
                vz,
                theta,
                omega,
                activation,
                waist,
                hip,
                knee,
                ankle,
                waist_rate,
                hip_rate,
                knee_rate,
                ankle_rate,
                touch_toe,
                touch_heel,
                z_min,
            ],
            dim=-1,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        step_dt = self.dt * self.ENV_ACTION_REPEAT

        H = state[:, self.H_IDX]
        vx = state[:, self.VX_IDX]
        vz = state[:, self.VZ_IDX]
        theta = state[:, self.PITCH_IDX]
        omega = state[:, self.PITCH_RATE_IDX]
        activation = state[:, self.ACTIVATION_IDX]
        q = state[:, self.QPOS_SLICE]
        qdot = state[:, self.QVEL_SLICE]
        touch_toe = torch.clamp(state[:, self.TOE_TOUCH_IDX], min=0.0, max=1.0)
        touch_heel = torch.clamp(state[:, self.HEEL_TOUCH_IDX], min=0.0, max=1.0)

        kin = self._kinematic_features(H, theta, q[:, 0], q[:, 1], q[:, 2], q[:, 3])

        activation_des = torch.sigmoid((action * self.activation_w.unsqueeze(0)).sum(dim=-1) + self.activation_b)
        next_activation = activation + step_dt * (activation_des - activation) / self.activation_tau
        next_activation = torch.clamp(next_activation, min=0.0, max=1.0)

        joint_acc = (
            self.joint_action_scale.unsqueeze(0) * action
            - self.joint_rate_damping.unsqueeze(0) * qdot
            - self.joint_centering.unsqueeze(0) * (q - self.joint_rest.unsqueeze(0))
        )
        next_qdot = qdot + step_dt * joint_acc
        next_q = q + step_dt * next_qdot

        # Use actuation-updated geometry / activation so the current action can
        # affect root translation and rotation within the same model step.
        H_nominal = torch.clamp(H + step_dt * vz, min=0.0, max=3.5)
        theta_nominal = _wrap_angle_torch(theta + step_dt * omega)
        kin_actuated = self._kinematic_features(
            H_nominal,
            theta_nominal,
            next_q[:, 0],
            next_q[:, 1],
            next_q[:, 2],
            next_q[:, 3],
        )
        kin_future_nominal = self._kinematic_features(
            torch.clamp(H_nominal + step_dt * vz, min=0.0, max=3.5),
            _wrap_angle_torch(theta_nominal + step_dt * omega),
            next_q[:, 0] + step_dt * next_qdot[:, 0],
            next_q[:, 1] + step_dt * next_qdot[:, 1],
            next_q[:, 2] + step_dt * next_qdot[:, 2],
            next_q[:, 3] + step_dt * next_qdot[:, 3],
        )
        leg = self._leg_mechanics(
            touch_toe,
            touch_heel,
            kin_actuated,
            next_kin_nominal=kin_future_nominal,
        )
        leg_drive = torch.relu(
            self.leg_stiffness * leg["c_leg"]
            + self.leg_damping * leg["d_c_leg"]
            + self.leg_force_max * next_activation
        )
        leg_drive_x = leg_drive * leg["e_leg_x"]
        leg_drive_z = leg_drive * leg["e_leg_z"]
        contact = self._contact_reaction(
            touch_toe,
            touch_heel,
            kin_actuated,
            kin_future_nominal,
            vx,
            vz,
            leg_drive_x,
            leg_drive_z,
            step_dt,
        )
        G = contact["gate"]
        airborne_gate = 1.0 - G

        F_x = contact["F_x"]
        F_z = contact["F_z"]
        tau_contact = contact["tau_contact"]
        tau_bal = -self.balance_theta * theta - self.balance_omega * omega

        # In free flight, articulated-body coupling still lets joint torques
        # perturb the root frame even without ground reaction force. Keep this
        # as a small explicit prior instead of zeroing all action influence.
        air_ax = airborne_gate * (
            self.AIRBORNE_ROOT_X_COUPLING[0] * joint_acc[:, 0]
            + self.AIRBORNE_ROOT_X_COUPLING[1] * joint_acc[:, 1]
            + self.AIRBORNE_ROOT_X_COUPLING[2] * joint_acc[:, 2]
            + self.AIRBORNE_ROOT_X_COUPLING[3] * joint_acc[:, 3]
        )
        tau_air = airborne_gate * (
            self.AIRBORNE_ROOT_TORQUE_COUPLING[0] * joint_acc[:, 0]
            + self.AIRBORNE_ROOT_TORQUE_COUPLING[1] * joint_acc[:, 1]
            + self.AIRBORNE_ROOT_TORQUE_COUPLING[2] * joint_acc[:, 2]
            + self.AIRBORNE_ROOT_TORQUE_COUPLING[3] * joint_acc[:, 3]
        )
        next_vx_base = vx + step_dt * (F_x + air_ax - self.linear_damping[0] * vx) / self.mass
        next_vz_base = vz + step_dt * (F_z - self.mass * self.gravity - self.linear_damping[1] * vz) / self.mass
        next_omega_base = (
            omega
            + step_dt * (tau_contact + tau_air + tau_bal - self.angular_damping * omega) / self.inertia
        )
        if self.use_transition_residual:
            residual_in = self._transition_residual_inputs(
                state=state,
                action=action,
                next_activation=next_activation,
                joint_acc=joint_acc,
                kin=kin_actuated,
                leg=leg,
                contact=contact,
                airborne_gate=airborne_gate,
                base_delta_vx=next_vx_base - vx,
                base_delta_vz=next_vz_base - vz,
                base_delta_omega=next_omega_base - omega,
            )
            residual_delta = (
                self.transition_residual_scale
                * self.transition_residual_base_scale.unsqueeze(0)
                * torch.tanh(self.transition_residual_head(residual_in))
            )
            touch_residual_delta = (
                self.transition_residual_scale
                * self.touch_transition_residual_base_scale.unsqueeze(0)
                * torch.tanh(self.touch_transition_residual_head(residual_in))
            )
        else:
            residual_delta = torch.zeros(
                (state.shape[0], 3),
                dtype=state.dtype,
                device=state.device,
            )
            touch_residual_delta = torch.zeros(
                (state.shape[0], 2),
                dtype=state.dtype,
                device=state.device,
            )

        next_vx = next_vx_base + residual_delta[:, 0]
        next_vz = next_vz_base + residual_delta[:, 1]
        next_omega = next_omega_base + residual_delta[:, 2]
        next_H = torch.clamp(H + step_dt * next_vz, min=0.0, max=3.5)
        next_theta = _wrap_angle_torch(theta + step_dt * next_omega)

        next_kin = self._kinematic_features(
            next_H,
            next_theta,
            next_q[:, 0],
            next_q[:, 1],
            next_q[:, 2],
            next_q[:, 3],
        )
        toe_vz_next = (next_kin["toe_z"] - kin_actuated["toe_z"]) / max(step_dt, 1e-6)
        heel_vz_next = (next_kin["heel_z"] - kin_actuated["heel_z"]) / max(step_dt, 1e-6)
        toe_target = self._touch_target(
            next_kin["toe_z"],
            toe_vz_next,
            next_vz,
            self.TOE_TOUCH_TARGET_MARGIN,
        )
        heel_target = self._touch_target(
            next_kin["heel_z"],
            heel_vz_next,
            next_vz,
            self.HEEL_TOUCH_TARGET_MARGIN,
        )
        next_touch_toe = self._asymmetric_contact_update(
            touch_toe,
            toe_target,
            rise_carry=self.CONTACT_CARRY_RISE,
            fall_carry=self.CONTACT_CARRY_FALL,
        )
        next_touch_heel = self._asymmetric_contact_update(
            touch_heel,
            heel_target,
            rise_carry=self.CONTACT_CARRY_RISE,
            fall_carry=self.CONTACT_CARRY_FALL,
        )
        toe_touch_residual_gate = self._touch_geom_gate(
            next_kin["toe_z"],
            self.TOE_TOUCH_TARGET_MARGIN,
            self.TOUCH_TARGET_SCALE,
        )
        heel_touch_residual_gate = self._touch_geom_gate(
            next_kin["heel_z"],
            self.HEEL_TOUCH_TARGET_MARGIN,
            self.TOUCH_TARGET_SCALE,
        )
        next_touch_toe = torch.clamp(
            next_touch_toe + toe_touch_residual_gate * touch_residual_delta[:, 0],
            min=0.0,
            max=1.0,
        )
        next_touch_heel = torch.clamp(
            next_touch_heel + heel_touch_residual_gate * touch_residual_delta[:, 1],
            min=0.0,
            max=1.0,
        )

        next_state = torch.stack(
            [
                next_H,
                next_vx,
                next_vz,
                next_theta,
                next_omega,
                next_activation,
                next_q[:, 0],
                next_q[:, 1],
                next_q[:, 2],
                next_q[:, 3],
                next_qdot[:, 0],
                next_qdot[:, 1],
                next_qdot[:, 2],
                next_qdot[:, 3],
                next_touch_toe,
                next_touch_heel,
                next_kin["z_min"],
            ],
            dim=-1,
        )
        return self.project_state(next_state)

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

    @staticmethod
    def _soft_interval(
        x: torch.Tensor,
        low: float,
        high: float,
        low_scale: float,
        high_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if high_scale is None:
            high_scale = low_scale
        low_gate = torch.sigmoid((x - low) / low_scale)
        high_gate = torch.sigmoid((high - x) / high_scale)
        return low_gate * high_gate

    @staticmethod
    def _robust_quadratic(z: torch.Tensor) -> torch.Tensor:
        abs_z = torch.abs(z)
        return torch.where(abs_z <= 1.0, 0.5 * z.square(), abs_z - 0.5)

    @staticmethod
    def _angle_residual(x: torch.Tensor, target: float) -> torch.Tensor:
        target_t = torch.full_like(x, target)
        return _wrap_angle_torch(x - target_t)

    def _pose_score(
        self,
        terms: Sequence[tuple[torch.Tensor, float]],
    ) -> torch.Tensor:
        distance = torch.zeros_like(terms[0][0])
        for term, weight in terms:
            distance = distance + float(weight) * term
        return torch.exp(-distance)

    def _stage_enabled(self, stage_name: str) -> bool:
        if stage_name == "prone_recover":
            return self.stage_mode == "prone_recover"
        return self.stage_mode == "full" or self.stage_mode == stage_name

    @staticmethod
    def _prior_box(center: torch.Tensor, width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        width = torch.clamp(width, min=0.05, max=0.95)
        low = torch.clamp(center - width, min=-1.0, max=1.0)
        high = torch.clamp(center + width, min=-1.0, max=1.0)
        return low, torch.maximum(high, low)

    def predict_height(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return torch.clamp(state[:, self.H_IDX], min=0.0, max=3.5)

    def _state_reward_features(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        H = torch.clamp(state[:, self.H_IDX], min=0.0, max=3.5)
        vx = state[:, self.VX_IDX]
        vz = state[:, self.VZ_IDX]
        theta = state[:, self.PITCH_IDX]
        omega = state[:, self.PITCH_RATE_IDX]
        activation = torch.clamp(state[:, self.ACTIVATION_IDX], min=0.0, max=1.0)
        q = state[:, self.QPOS_SLICE]
        qdot = state[:, self.QVEL_SLICE]
        touch_toe = torch.clamp(state[:, self.TOE_TOUCH_IDX], min=0.0, max=1.0)
        touch_heel = torch.clamp(state[:, self.HEEL_TOUCH_IDX], min=0.0, max=1.0)
        stored_z_min = torch.clamp(state[:, self.ZMIN_IDX], min=-0.5, max=3.0)

        kin = self._kinematic_features(H, theta, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        leg = self._leg_mechanics(touch_toe, touch_heel, kin)
        G = self._contact_gate(touch_toe, touch_heel, kin)

        height = H
        z_min = kin["z_min"]
        any_contact_z = z_min
        support_contact = torch.maximum(touch_toe, touch_heel)
        full_foot_support = torch.minimum(touch_toe, touch_heel)
        foot_contact_z = torch.minimum(kin["toe_z"], kin["heel_z"])
        foot_grounded = (
            (foot_contact_z <= self.STATE_GROUND_EPS)
            | (support_contact >= self.STATE_SUPPORT_TOUCH)
        ).to(state.dtype)
        body_contact_points = torch.stack(
            [
                torch.stack([kin["torso_lower_x"], kin["torso_lower_z"]], dim=-1),
                torch.stack([kin["nose_lower_x"], kin["nose_lower_z"]], dim=-1),
                torch.stack([kin["pelvis_lower_x"], kin["pelvis_lower_z"]], dim=-1),
                torch.stack([kin["thigh_lower_x"], kin["thigh_lower_z"]], dim=-1),
                torch.stack([kin["calf_lower_x"], kin["calf_lower_z"]], dim=-1),
            ],
            dim=-2,
        )
        body_contact_zs = body_contact_points[..., 1]
        body_contact_idx = body_contact_zs.argmin(dim=-1, keepdim=True)
        body_contact_z = body_contact_zs.gather(-1, body_contact_idx).squeeze(-1)
        body_contact_x = body_contact_points[..., 0].gather(-1, body_contact_idx).squeeze(-1)
        body_collision = (body_contact_z <= self.STATE_BODY_COLLISION_EPS).to(state.dtype)
        no_body_collision = 1.0 - body_collision
        grounded_any = (any_contact_z <= self.STATE_GROUND_EPS).to(state.dtype)
        airborne = (any_contact_z > self.STATE_GROUND_EPS).to(state.dtype)
        airborne_true = airborne * (1.0 - foot_grounded) * no_body_collision
        support_contact_gate = (support_contact >= self.STATE_SUPPORT_TOUCH).to(state.dtype)
        full_foot_support_gate = (full_foot_support >= self.STATE_SUPPORT_TOUCH).to(state.dtype)
        upright_pitch = (
            (theta > self.STATE_UPRIGHT_PITCH_LOW)
            & (theta < self.STATE_UPRIGHT_PITCH_HIGH)
        ).to(state.dtype)
        nose_minus_foot_x = kin["nose_x"] - kin["foot_x"]
        nose_minus_support_x = kin["nose_x"] - leg["p_contact_x"]
        hip_minus_support_x = kin["hip_x"] - leg["p_contact_x"]
        posture_anchor_x = 0.5 * (kin["x_min"] + kin["x_max"])
        nose_minus_anchor_x = kin["nose_x"] - posture_anchor_x
        rear_geometry = grounded_any * (nose_minus_anchor_x < 0.0).to(state.dtype)
        prone_geometry = grounded_any * (nose_minus_anchor_x >= 0.0).to(state.dtype)
        abs_theta = torch.abs(theta)
        abs_omega = torch.abs(omega)

        qpos_knee = q[:, 2]
        knee_window = self._soft_interval(
            qpos_knee,
            low=self.KNEE_READY_LOW,
            high=self.KNEE_READY_HIGH,
            low_scale=self.KNEE_READY_LOW_SCALE,
            high_scale=self.KNEE_READY_HIGH_SCALE,
        )
        upright_height_gate = (height >= self.STATE_UPRIGHT_HEIGHT_MIN).to(state.dtype)
        stable_vz_gate = (torch.abs(vz) <= self.STATE_PREPARED_VZ_MAX).to(state.dtype)
        stable_omega_gate = (abs_omega <= self.STATE_UPRIGHT_OMEGA_MAX).to(state.dtype)
        upright_quality_gate = (
            foot_grounded
            * no_body_collision
            * full_foot_support_gate
            * upright_pitch
            * upright_height_gate
        )
        prepared_quality_gate = (
            foot_grounded
            * no_body_collision
            * full_foot_support_gate
            * upright_pitch
            * upright_height_gate
            * knee_window
            * stable_vz_gate
            * stable_omega_gate
        )
        landing_quality_gate = prepared_quality_gate
        rising_quality_gate = (
            airborne_true
            * (vz > self.STATE_RISING_VZ_MIN).to(state.dtype)
        )
        state_confident = torch.maximum(
            torch.maximum(rear_geometry, prone_geometry),
            torch.maximum(
                upright_quality_gate,
                torch.maximum(
                    prepared_quality_gate,
                    airborne_true * (
                        ((vz < -self.STATE_VZ_EPS) | (vz > self.STATE_RISING_VZ_MIN))
                    ).to(state.dtype),
                ),
            ),
        )
        grounded_prepared = prepared_quality_gate
        grounded_upright = upright_quality_gate * (1.0 - grounded_prepared)
        grounded_rear = grounded_any * (1.0 - grounded_prepared) * (1.0 - grounded_upright) * rear_geometry
        grounded_prone = grounded_any * (1.0 - grounded_prepared) * (1.0 - grounded_upright) * (1.0 - grounded_rear) * prone_geometry
        falling_air = airborne_true * (vz < -self.STATE_VZ_EPS).to(state.dtype)
        rising_air = rising_quality_gate

        posture_stability = torch.clamp((1.10 - abs_theta) / 1.10, min=0.0, max=1.0)
        spin_stability = torch.clamp((self.STATE_UPRIGHT_OMEGA_MAX - abs_omega) / self.STATE_UPRIGHT_OMEGA_MAX, min=0.0, max=1.0)
        forward_progress = torch.clamp(vx / max(self.hop_speed, 1e-6), min=0.0, max=1.0)
        upward_progress = torch.clamp(vz / 0.60, min=0.0, max=1.0)
        descent_progress = torch.clamp((-vz) / 0.60, min=0.0, max=1.0)
        backward_speed = torch.clamp(-vx / 0.50, min=0.0, max=1.0)
        low_body = torch.clamp((0.90 - height) / 0.30, min=0.0, max=1.0)
        vertical_settle = torch.clamp((0.40 - torch.abs(vz)) / 0.40, min=0.0, max=1.0)
        rotation_penalty = torch.clamp((abs_omega - self.STATE_UPRIGHT_OMEGA_MAX) / 2.5, min=0.0, max=1.0)

        landing_pose_score = self._pose_score(
            [
                (self._robust_quadratic(self._angle_residual(theta, 0.0) / self.DENSE_THETA_SIGMA), 0.32),
                (self._robust_quadratic(omega / self.DENSE_OMEGA_SIGMA), 0.20),
                (self._robust_quadratic((qpos_knee - self.DENSE_LANDING_KNEE_TARGET) / self.DENSE_KNEE_SIGMA), 0.18),
                (self._robust_quadratic((vz - self.DENSE_LANDING_VZ_TARGET) / self.DENSE_VZ_SIGMA), 0.15),
                (self._robust_quadratic((vx - self.hop_speed) / self.DENSE_VX_SIGMA), 0.15),
            ]
        )
        ground_pose_score = self._pose_score(
            [
                (self._robust_quadratic(self._angle_residual(theta, 0.0) / self.DENSE_THETA_SIGMA), 0.26),
                (self._robust_quadratic(omega / self.DENSE_OMEGA_SIGMA), 0.16),
                (self._robust_quadratic((qpos_knee - self.DENSE_GROUND_KNEE_TARGET) / self.DENSE_KNEE_SIGMA), 0.18),
                (self._robust_quadratic((vz - self.DENSE_GROUND_VZ_TARGET) / 0.22), 0.10),
                (self._robust_quadratic((height - self.DENSE_HEIGHT_TARGET) / self.DENSE_HEIGHT_SIGMA), 0.10),
                (self._robust_quadratic((full_foot_support - 1.0) / self.DENSE_SUPPORT_SIGMA), 0.12),
                (self._robust_quadratic(body_collision / self.DENSE_BODY_SIGMA), 0.08),
            ]
        )
        rise_pose_score = self._pose_score(
            [
                (self._robust_quadratic(self._angle_residual(theta, 0.0) / self.DENSE_THETA_SIGMA), 0.25),
                (self._robust_quadratic(omega / self.DENSE_OMEGA_SIGMA), 0.16),
                (self._robust_quadratic((qpos_knee - self.DENSE_RISE_KNEE_TARGET) / 0.35), 0.14),
                (self._robust_quadratic((vz - self.DENSE_RISE_VZ_TARGET) / self.DENSE_VZ_SIGMA), 0.22),
                (self._robust_quadratic(body_collision / self.DENSE_BODY_SIGMA), 0.08),
            ]
        )
        prone_recover_knee_score = self._soft_interval(
            qpos_knee,
            low=self.PRONE_RECOVER_KNEE_LOW,
            high=self.PRONE_RECOVER_KNEE_HIGH,
            low_scale=self.PRONE_RECOVER_KNEE_LOW_SCALE,
            high_scale=self.PRONE_RECOVER_KNEE_HIGH_SCALE,
        )
        prone_recover_geom_score = torch.clamp(
            1.0 - torch.abs(nose_minus_anchor_x) / self.PRONE_RECOVER_GEOM_SCALE,
            min=0.0,
            max=1.0,
        )
        prone_recover_pitch_score = torch.exp(
            -self._robust_quadratic(
                self._angle_residual(theta, 0.0) / self.PRONE_RECOVER_PITCH_SIGMA
            )
        )
        prone_recover_spin_score = torch.exp(
            -self._robust_quadratic(omega / self.PRONE_RECOVER_SPIN_SIGMA)
        )
        prone_recover_support_score = support_contact
        prone_recover_support_gate = support_contact
        prone_recover_grounded_soft = torch.sigmoid(
            (self.STATE_GROUND_EPS - any_contact_z) / self.PRONE_RECOVER_GROUND_SOFT_SCALE
        )
        prone_recover_foot_soft = torch.sigmoid(
            (self.STATE_GROUND_EPS - foot_contact_z) / self.PRONE_RECOVER_FOOT_SOFT_SCALE
        )
        prone_recover_body_clear_score = torch.sigmoid(
            (body_contact_z - self.PRONE_RECOVER_BODY_CLEAR_CENTER)
            / self.PRONE_RECOVER_BODY_CLEAR_SCALE
        )
        prone_recover_lift_score = torch.sigmoid(
            (H - self.PRONE_RECOVER_TORSO_Z_CENTER) / self.PRONE_RECOVER_TORSO_Z_SCALE
        )
        prone_recover_torso_over_foot_score = torch.exp(
            -self._robust_quadratic(
                (kin["torso_x"] - leg["p_contact_x"]) / self.PRONE_RECOVER_TORSO_OVER_FOOT_SIGMA
            )
        )
        prone_recover_progress_score = (
            0.40 * support_contact
            + 0.30 * prone_recover_body_clear_score
            + 0.30 * prone_recover_lift_score
        )
        prone_recover_core_score = (
            full_foot_support
            * prone_recover_body_clear_score
            * prone_recover_pitch_score
            * prone_recover_lift_score
        )
        prone_recover_progress_component = (
            self.PRONE_RECOVER_PROGRESS_WEIGHT
            * prone_recover_grounded_soft
            * prone_recover_progress_score
        )
        prone_recover_core_component = (
            (1.0 - self.PRONE_RECOVER_PROGRESS_WEIGHT)
            * prone_recover_core_score
        )
        # Backward-compatible diagnostics: old unfold/stand names now map to
        # the simplified progress/core decomposition.
        prone_recover_unfold_score = prone_recover_progress_score
        prone_recover_stand_score = prone_recover_core_score
        prone_recover_unfold_component = prone_recover_progress_component
        prone_recover_stand_component = prone_recover_core_component
        prone_recover_pose_core = prone_recover_progress_score + prone_recover_core_score
        prone_recover_spin_cap = 0.5 + 0.5 * prone_recover_spin_score
        prone_recover_pose_score = (
            prone_recover_unfold_component
            + prone_recover_stand_component
        )

        dense_down_gate_raw = airborne_true * torch.sigmoid(
            (-vz - self.DENSE_DOWN_V_CENTER) / self.DENSE_V_SCALE
        )
        dense_ground_gate_raw = foot_grounded * no_body_collision
        dense_rise_gate_raw = airborne_true * torch.sigmoid(
            (vz - self.DENSE_RISE_V_CENTER) / self.DENSE_V_SCALE
        )
        dense_gate_sum = dense_down_gate_raw + dense_ground_gate_raw + dense_rise_gate_raw + 1e-6
        dense_down_gate = dense_down_gate_raw / dense_gate_sum
        dense_ground_gate = dense_ground_gate_raw / dense_gate_sum
        dense_rise_gate = dense_rise_gate_raw / dense_gate_sum
        dense_landing_score = landing_pose_score * no_body_collision * stable_omega_gate
        dense_ground_score = ground_pose_score * full_foot_support_gate * no_body_collision
        dense_rise_score = rise_pose_score * forward_progress
        prone_recover_score = prone_recover_pose_score
        landing_component = dense_down_gate * dense_landing_score
        ground_component = dense_ground_gate * dense_ground_score
        rise_component = dense_rise_gate * dense_rise_score
        prone_recover_component = prone_recover_score
        dense_pose_potential = torch.zeros_like(landing_component)
        if self._stage_enabled("landing"):
            dense_pose_potential = dense_pose_potential + landing_component
        if self._stage_enabled("ground"):
            dense_pose_potential = dense_pose_potential + ground_component
        if self._stage_enabled("rise"):
            dense_pose_potential = dense_pose_potential + rise_component
        if self._stage_enabled("prone_recover"):
            dense_pose_potential = dense_pose_potential + prone_recover_component

        return {
            "torso_z": H,
            "rootz": H - self.TORSO_BASE_Z,
            "height": height,
            "vx": vx,
            "vz": vz,
            "pitch": theta,
            "pitch_rate": omega,
            "activation": activation,
            "touch_toe": touch_toe,
            "touch_heel": touch_heel,
            "z_min": z_min,
            "z_min_state": stored_z_min,
            "any_contact_z": any_contact_z,
            "foot_contact_z": foot_contact_z,
            "body_contact_x": body_contact_x,
            "body_contact_z": body_contact_z,
            "foot_grounded": foot_grounded,
            "body_collision": body_collision,
            "airborne_true": airborne_true,
            "qpos_waist": q[:, 0],
            "qpos_hip": q[:, 1],
            "qpos_knee": qpos_knee,
            "qpos_ankle": q[:, 3],
            "qvel_waist": qdot[:, 0],
            "qvel_hip": qdot[:, 1],
            "qvel_knee": qdot[:, 2],
            "qvel_ankle": qdot[:, 3],
            "G": G,
            "p_contact_x": leg["p_contact_x"],
            "p_contact_z": leg["p_contact_z"],
            "nose_x": kin["nose_x"],
            "nose_z": kin["nose_z"],
            "torso_x": kin["torso_x"],
            "pelvis_x": kin["pelvis_x"],
            "pelvis_z": kin["pelvis_z"],
            "thigh_x": kin["thigh_x"],
            "thigh_z": kin["thigh_z"],
            "calf_x": kin["calf_x"],
            "calf_z": kin["calf_z"],
            "foot_x": kin["foot_x"],
            "foot_z": kin["foot_z"],
            "toe_x": kin["toe_x"],
            "toe_z": kin["toe_z"],
            "heel_x": kin["heel_x"],
            "heel_z": kin["heel_z"],
            "x_min": kin["x_min"],
            "x_max": kin["x_max"],
            "posture_anchor_x": posture_anchor_x,
            "nose_minus_anchor_x": nose_minus_anchor_x,
            "nose_minus_foot_x": nose_minus_foot_x,
            "nose_minus_support_x": nose_minus_support_x,
            "hip_minus_support_x": hip_minus_support_x,
            "leg_length": leg["leg_length"],
            "c_leg": leg["c_leg"],
            "d_c_leg": leg["d_c_leg"],
            "e_leg_x": leg["e_leg_x"],
            "e_leg_z": leg["e_leg_z"],
            "posture_stability": posture_stability,
            "spin_stability": spin_stability,
            "forward_progress": forward_progress,
            "upward_progress": upward_progress,
            "descent_progress": descent_progress,
            "backward_speed": backward_speed,
            "support_contact": support_contact,
            "full_foot_support": full_foot_support,
            "support_contact_gate": support_contact_gate,
            "full_foot_support_gate": full_foot_support_gate,
            "upright_height_gate": upright_height_gate,
            "stable_vz_gate": stable_vz_gate,
            "stable_omega_gate": stable_omega_gate,
            "upright_quality_gate": upright_quality_gate,
            "prepared_quality_gate": prepared_quality_gate,
            "landing_quality_gate": landing_quality_gate,
            "rising_quality_gate": rising_quality_gate,
            "state_confident": state_confident,
            "low_body": low_body,
            "knee_window": knee_window,
            "vertical_settle": vertical_settle,
            "falling_air": falling_air,
            "grounded_rear": grounded_rear,
            "grounded_prone": grounded_prone,
            "grounded_upright": grounded_upright,
            "grounded_prepared": grounded_prepared,
            "rising_air": rising_air,
            "rotation_penalty": rotation_penalty,
            "landing_pose_score": landing_pose_score,
            "ground_pose_score": ground_pose_score,
            "rise_pose_score": rise_pose_score,
            "prone_recover_geom_score": prone_recover_geom_score,
            "prone_recover_pitch_score": prone_recover_pitch_score,
            "prone_recover_spin_score": prone_recover_spin_score,
            "prone_recover_support_score": prone_recover_support_score,
            "prone_recover_support_gate": prone_recover_support_gate,
            "prone_recover_grounded_soft": prone_recover_grounded_soft,
            "prone_recover_foot_soft": prone_recover_foot_soft,
            "prone_recover_body_clear_score": prone_recover_body_clear_score,
            "prone_recover_torso_over_foot_score": prone_recover_torso_over_foot_score,
            "prone_recover_lift_score": prone_recover_lift_score,
            "prone_recover_knee_score": prone_recover_knee_score,
            "prone_recover_progress_score": prone_recover_progress_score,
            "prone_recover_core_score": prone_recover_core_score,
            "prone_recover_unfold_score": prone_recover_unfold_score,
            "prone_recover_stand_score": prone_recover_stand_score,
            "prone_recover_progress_component": prone_recover_progress_component,
            "prone_recover_core_component": prone_recover_core_component,
            "prone_recover_unfold_component": prone_recover_unfold_component,
            "prone_recover_stand_component": prone_recover_stand_component,
            "prone_recover_pose_core": prone_recover_pose_core,
            "prone_recover_spin_cap": prone_recover_spin_cap,
            "prone_recover_pose_score": prone_recover_pose_score,
            "dense_down_gate": dense_down_gate,
            "dense_ground_gate": dense_ground_gate,
            "dense_rise_gate": dense_rise_gate,
            "dense_landing_component": landing_component,
            "dense_ground_component": ground_component,
            "dense_rise_component": rise_component,
            "dense_prone_recover_component": prone_recover_component,
            "dense_pose_potential": dense_pose_potential,
        }

    @staticmethod
    def _dominant_state_ids(feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        grounded = feat["any_contact_z"] <= DMControlHopperSDModel.STATE_GROUND_EPS
        airborne = ~grounded
        foot_grounded = feat["foot_grounded"] >= 0.5
        body_collision = feat["body_collision"] >= 0.5
        airborne_true = (
            (feat["airborne_true"] >= 0.5)
            & airborne
        )
        rear = grounded & (feat["nose_minus_anchor_x"] < 0.0)
        prone = grounded & ~rear
        upright_pitch = (
            (feat["pitch"] > DMControlHopperSDModel.STATE_UPRIGHT_PITCH_LOW)
            & (feat["pitch"] < DMControlHopperSDModel.STATE_UPRIGHT_PITCH_HIGH)
        )
        prepared = (
            foot_grounded
            & ~body_collision
            & (feat["full_foot_support"] >= DMControlHopperSDModel.STATE_SUPPORT_TOUCH)
            & upright_pitch
            & (feat["height"] >= DMControlHopperSDModel.STATE_UPRIGHT_HEIGHT_MIN)
            & (feat["qpos_knee"] >= DMControlHopperSDModel.KNEE_READY_LOW)
            & (feat["qpos_knee"] <= DMControlHopperSDModel.KNEE_READY_HIGH)
            & (torch.abs(feat["vz"]) <= DMControlHopperSDModel.STATE_PREPARED_VZ_MAX)
            & (torch.abs(feat["pitch_rate"]) <= DMControlHopperSDModel.STATE_UPRIGHT_OMEGA_MAX)
        )
        upright = (
            foot_grounded
            & ~body_collision
            & (feat["full_foot_support"] >= DMControlHopperSDModel.STATE_SUPPORT_TOUCH)
            & upright_pitch
            & (feat["height"] >= DMControlHopperSDModel.STATE_UPRIGHT_HEIGHT_MIN)
            & ~prepared
        )
        falling = airborne_true & (feat["vz"] < -DMControlHopperSDModel.STATE_VZ_EPS)
        rising = airborne_true & (feat["vz"] > DMControlHopperSDModel.STATE_RISING_VZ_MIN)
        state_ids = torch.full_like(feat["vz"], -1, dtype=torch.long)
        state_ids = torch.where(rear, torch.full_like(state_ids, 1), state_ids)
        state_ids = torch.where(prone, torch.full_like(state_ids, 2), state_ids)
        state_ids = torch.where(upright, torch.full_like(state_ids, 3), state_ids)
        state_ids = torch.where(prepared, torch.full_like(state_ids, 4), state_ids)
        state_ids = torch.where(rising, torch.full_like(state_ids, 5), state_ids)
        state_ids = torch.where(falling, torch.zeros_like(state_ids), state_ids)
        grounded_fallback = grounded & (state_ids == -1)
        state_ids = torch.where(
            grounded_fallback & (feat["nose_minus_anchor_x"] >= 0.0),
            torch.full_like(state_ids, 2),
            state_ids,
        )
        state_ids = torch.where(
            grounded_fallback & (feat["nose_minus_anchor_x"] < 0.0),
            torch.full_like(state_ids, 1),
            state_ids,
        )
        airborne_fallback = airborne & (state_ids == -1)
        state_ids = torch.where(airborne_fallback, torch.zeros_like(state_ids), state_ids)
        return state_ids

    def predict_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        if next_state is None:
            next_state = self.forward(state, action)
        true_reward = self.predict_reward(state, action, next_state)
        feat = self._state_reward_features(next_state)
        height_gate = self._lower_tolerance(feat["height"], lower=0.45, margin=0.15, value_at_margin=0.0)
        upright = 0.5 * (1.0 + torch.cos(feat["pitch"]))
        pitch_gate = torch.clamp((upright - 0.35) / 0.45, min=0.0, max=1.0)
        contact_gate = torch.clamp((0.03 - feat["any_contact_z"]) / 0.06, min=0.0, max=1.0)
        viability = height_gate * pitch_gate * (0.50 + 0.50 * contact_gate)
        return viability * true_reward

    def get_action_prior_bounds(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        feat = self._state_reward_features(state)
        batch = state.shape[0]
        low = torch.full((batch, self.ACTION_DIM), -1.0, dtype=state.dtype, device=state.device)
        high = torch.full_like(low, 1.0)
        state_ids = self._dominant_state_ids(feat)
        rear_mask = state_ids == 1
        prone_mask = state_ids == 2

        if not self.action_prior_enabled:
            stats = {
                "rear_prior_frac": 0.0,
                "prone_prior_frac": 0.0,
                "rear_state_frac": float(rear_mask.float().mean().item()),
                "prone_state_frac": float(prone_mask.float().mean().item()),
                "action_prior_enabled": 0.0,
            }
            return low, high, stats

        support_missing = torch.clamp(self.PRIOR_FOOT_SUPPORT_TARGET - feat["full_foot_support"], min=0.0, max=1.0)
        knee_excess = torch.clamp(feat["qpos_knee"] - self.PRIOR_KNEE_TARGET, min=0.0, max=1.5)
        rear_progress = torch.clamp(
            (feat["pitch"] - self.REAR_LYING_PITCH_MAX) / self.REAR_PROGRESS_SCALE,
            min=0.0,
            max=1.0,
        )
        prone_severity = torch.clamp(
            (feat["pitch"] - self.PRONE_LYING_PITCH_MIN) / self.PRONE_SEVERITY_SCALE,
            min=0.0,
            max=1.0,
        )

        if rear_mask.any():
            rear_center = torch.stack(
                [
                    torch.clamp(-0.35 - 0.35 * rear_progress, min=-0.95, max=-0.10),
                    torch.clamp(0.25 - 1.10 * rear_progress - 0.10 * support_missing, min=-1.0, max=0.60),
                    torch.clamp(-0.10 + 0.75 * rear_progress, min=-0.70, max=0.95),
                    torch.clamp(-0.25 - 0.35 * rear_progress, min=-1.0, max=0.40),
                ],
                dim=-1,
            )
            rear_width = torch.stack(
                [
                    torch.full_like(rear_progress, 0.18),
                    torch.full_like(rear_progress, 0.28),
                    torch.full_like(rear_progress, 0.34),
                    torch.full_like(rear_progress, 0.26),
                ],
                dim=-1,
            )
            rear_low, rear_high = self._prior_box(rear_center, rear_width)
            low[rear_mask] = rear_low[rear_mask]
            high[rear_mask] = rear_high[rear_mask]

        if prone_mask.any():
            prone_center = torch.stack(
                [
                    torch.clamp(0.72 + 0.18 * prone_severity, min=0.45, max=0.98),
                    torch.clamp(-0.88 + 0.18 * (1.0 - prone_severity), min=-1.0, max=-0.15),
                    torch.clamp(0.05 - 0.12 * knee_excess, min=-0.35, max=0.45),
                    torch.clamp(0.72 + 0.10 * support_missing, min=0.45, max=0.98),
                ],
                dim=-1,
            )
            prone_width = torch.stack(
                [
                    torch.full_like(prone_severity, 0.18),
                    torch.full_like(prone_severity, 0.24),
                    torch.full_like(prone_severity, 0.42),
                    torch.full_like(prone_severity, 0.18),
                ],
                dim=-1,
            )
            prone_low, prone_high = self._prior_box(prone_center, prone_width)
            low[prone_mask] = prone_low[prone_mask]
            high[prone_mask] = prone_high[prone_mask]

        stats = {
            "rear_prior_frac": float(rear_mask.float().mean().item()),
            "prone_prior_frac": float(prone_mask.float().mean().item()),
            "rear_state_frac": float(rear_mask.float().mean().item()),
            "prone_state_frac": float(prone_mask.float().mean().item()),
            "action_prior_enabled": 1.0,
        }
        return low, high, stats

    def predict_shaping_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if next_state is None:
            next_state = self.forward(state, action)
        prev_feat = self._state_reward_features(state)
        next_feat = self._state_reward_features(next_state)
        prev_state = self._dominant_state_ids(prev_feat)
        next_state_id = self._dominant_state_ids(next_feat)

        prev_conf = prev_feat["state_confident"]
        next_conf = next_feat["state_confident"]
        upright_quality = next_feat["upright_quality_gate"]
        prepared_quality = next_feat["prepared_quality_gate"]
        landing_quality = next_feat["landing_quality_gate"]
        rising_quality = next_feat["rising_quality_gate"]
        forward_gate = next_feat["forward_progress"]

        rear_to_upright = (
            ((prev_state == 1) & (next_state_id == 3)).to(state.dtype)
            * prev_conf
            * upright_quality
        )
        prone_to_upright = (
            ((prev_state == 2) & (next_state_id == 3)).to(state.dtype)
            * prev_conf
            * upright_quality
        )
        upright_to_prepared = (
            ((prev_state == 3) & (next_state_id == 4)).to(state.dtype)
            * prev_conf
            * prepared_quality
        )
        prepared_to_rising = (
            ((prev_state == 4) & (next_state_id == 5)).to(state.dtype)
            * prev_conf
            * next_conf
            * rising_quality
            * forward_gate
        )
        falling_to_upright = (
            ((prev_state == 0) & (next_state_id == 3)).to(state.dtype)
            * prev_conf
            * upright_quality
        )
        falling_to_prepared = (
            ((prev_state == 0) & (next_state_id == 4)).to(state.dtype)
            * prev_conf
            * landing_quality
        )

        milestone = torch.zeros_like(rear_to_upright)
        if self._stage_enabled("ground"):
            milestone = milestone + (
                self.REAR_TO_UPRIGHT_REWARD_WEIGHT * rear_to_upright
                + self.PRONE_TO_UPRIGHT_REWARD_WEIGHT * prone_to_upright
                + self.UPRIGHT_TO_PREPARED_REWARD_WEIGHT * upright_to_prepared
            )
        if self._stage_enabled("prone_recover"):
            milestone = milestone + (
                self.PRONE_TO_UPRIGHT_REWARD_WEIGHT * prone_to_upright
            )
        if self._stage_enabled("rise"):
            milestone = milestone + (
                self.PREPARED_TO_RISING_REWARD_WEIGHT * prepared_to_rising
            )
        if self._stage_enabled("landing"):
            milestone = milestone + (
                self.FALLING_TO_UPRIGHT_REWARD_WEIGHT * falling_to_upright
                + self.FALLING_TO_PREPARED_REWARD_WEIGHT * falling_to_prepared
            )
        dense_prev = prev_feat["dense_pose_potential"]
        dense_next = next_feat["dense_pose_potential"]
        dense_delta = self.dense_pose_gamma * dense_next - dense_prev
        if self._stage_enabled("prone_recover"):
            dense_reward = (
                self.dense_pose_delta_weight * dense_delta
                + self.PRONE_RECOVER_CORE_ABS_WEIGHT * next_feat["prone_recover_core_component"]
            )
            recover_penalty = torch.zeros_like(dense_reward)
        else:
            dense_reward = (
                self.dense_pose_delta_weight * dense_delta
                + self.dense_pose_abs_weight * dense_next
            )
            recover_penalty = torch.zeros_like(dense_reward)
        if self._stage_enabled("ground"):
            recover_mask = prev_feat["body_collision"] * (1.0 - prev_feat["airborne_true"])
            recover_penalty = recover_penalty + recover_mask * (
                0.20 * torch.abs(prev_feat["pitch"])
                + 0.05 * torch.abs(prev_feat["pitch_rate"])
            )
        total = (
            dense_reward
            + self.dense_pose_milestone_scale * milestone
            - self.recover_penalty_scale * recover_penalty
        )
        return self.shaping_scale * total

    def predict_plan_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if next_state is None:
            next_state = self.forward(state, action)
        true_reward = self.predict_reward(state, action, next_state)
        shaping_delta = self.predict_shaping_reward(state, action, next_state)
        return true_reward + shaping_delta

    def get_learned_params(self) -> Dict[str, object]:
        return {
            "dt": self.dt,
            "hop_speed": self.hop_speed,
            "mass": float(self.mass.item()),
            "inertia": float(self.inertia.item()),
            "rest_leg_length": float(self.rest_leg_length.item()),
            "activation_w": self.activation_w.detach().cpu().tolist(),
            "activation_b": float(self.activation_b.detach().cpu().item()),
            "activation_tau": float(self.activation_tau.detach().cpu().item()),
            "leg_stiffness": float(self.leg_stiffness.detach().cpu().item()),
            "leg_damping": float(self.leg_damping.detach().cpu().item()),
            "leg_force_max": float(self.leg_force_max.detach().cpu().item()),
            "contact_k": float(self.contact_k.detach().cpu().item()),
            "contact_tau": float(self.contact_tau.detach().cpu().item()),
            "linear_damping": self.linear_damping.detach().cpu().tolist(),
            "angular_damping": float(self.angular_damping.detach().cpu().item()),
            "balance_theta": float(self.balance_theta.detach().cpu().item()),
            "balance_omega": float(self.balance_omega.detach().cpu().item()),
            "joint_action_scale": self.joint_action_scale.detach().cpu().tolist(),
            "joint_rate_damping": self.joint_rate_damping.detach().cpu().tolist(),
            "joint_centering": self.joint_centering.detach().cpu().tolist(),
            "use_transition_residual": bool(self.use_transition_residual),
            "transition_residual_scale": self.transition_residual_scale,
            "transition_residual_base_scale": self.transition_residual_base_scale.detach().cpu().tolist(),
            "touch_transition_residual_base_scale": self.touch_transition_residual_base_scale.detach().cpu().tolist(),
            "action_prior_enabled": bool(self.action_prior_enabled),
            "dense_pose_delta_weight": self.dense_pose_delta_weight,
            "dense_pose_abs_weight": self.dense_pose_abs_weight,
            "dense_pose_milestone_scale": self.dense_pose_milestone_scale,
            "dense_pose_gamma": self.dense_pose_gamma,
            "recover_penalty_scale": self.recover_penalty_scale,
            "latent_layout": [
                "H",
                "vx",
                "vz",
                "theta",
                "omega",
                "A",
                "waist",
                "hip",
                "knee",
                "ankle",
                "waist_dot",
                "hip_dot",
                "knee_dot",
                "ankle_dot",
                "touch_toe",
                "touch_heel",
                "z_min",
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
