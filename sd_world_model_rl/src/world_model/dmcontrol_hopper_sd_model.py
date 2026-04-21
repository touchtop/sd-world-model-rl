"""Structured world model for DMControl Hopper-Hop with explicit core stocks."""

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
    """Explicit SD model built around a small set of core feedback stocks.

    Raw observation layout (15 dims), matching dm_control state observations:
    [rootz, rooty, waist, hip, knee, ankle, x_dot, z_dot,
     rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot, touch_toe, touch_heel]

    Structured latent layout (9 dims):
    [vx, vz, pitch, pitch_rate, compression, compression_rate, support, rootz, capture_memory]

    The first 8 dims are observable-state stocks that can be reconstructed from
    the current simulator observation. The final dim `capture_memory` is a true
    hidden stock: it stores whether fore-body capture has been established and
    should therefore not be supervised by one-frame observation matching.
    """

    POSITION_DIM = 6
    VELOCITY_DIM = 7
    TOUCH_DIM = 2
    RAW_OBS_DIM = 15
    STATE_DIM = 9
    OBSERVABLE_DIM = 8
    ACTION_DIM = 4

    VX_IDX = 0
    VZ_IDX = 1
    PITCH_IDX = 2
    PITCH_RATE_IDX = 3
    COMP_IDX = 4
    COMP_RATE_IDX = 5
    SUPPORT_IDX = 6
    ROOTZ_IDX = 7
    CAPTURE_IDX = 8
    RATE_LIKE_IDX = (VX_IDX, VZ_IDX, PITCH_RATE_IDX, COMP_RATE_IDX)

    ANGLE_SLICE = slice(1, 6)
    TORSO_TO_PELVIS_Z = -0.05
    PELVIS_TO_THIGH_Z = -0.20
    THIGH_TO_CALF_Z = -0.33
    CALF_TO_FOOT_Z = -0.32
    TOE_OFFSET_X = 0.17
    HEEL_OFFSET_X = -0.08
    CONTACT_MARGIN = 0.02
    CONTACT_SCALE = 0.02
    CONTACT_VEL_SCALE = 0.35
    TARGET_STAND_HEIGHT = 0.90
    ENV_ACTION_REPEAT = 2

    V4_ROOTZ_LOW_CENTER = -0.5
    V4_ROOTZ_LOW_SCALE = 0.1
    V4_PITCH_BACK_CENTER = -0.3
    V4_PITCH_BACK_SCALE = 0.1
    V4_PITCH_FWD_CENTER = 0.0
    V4_PITCH_FWD_SCALE = 0.15
    V4_PITCH_NEUTRAL_SPREAD = 0.3
    V4_CAPTURE_PITCH_CENTER = 0.8
    V4_CAPTURE_PITCH_SCALE = 0.35
    V4_CAPTURE_PITCH_LOW_CENTER = 0.55
    V4_CAPTURE_PITCH_LOW_SCALE = 0.22
    V4_CAPTURE_PITCH_HIGH_CENTER = 2.80
    V4_CAPTURE_PITCH_HIGH_SCALE = 0.50
    V4_CAPTURE_CLEARANCE_CENTER = 0.70
    V4_CAPTURE_CLEARANCE_SCALE = 0.08
    V4_CAPTURE_CENTER = 0.55
    V4_CAPTURE_SCALE = 0.08
    V4_HANDOFF_BODY_HIGH_CENTER = 0.10
    V4_HANDOFF_BODY_HIGH_SCALE = 0.10
    V4_PRECAPTURE_ROOTZ_CENTER = -0.32
    V4_PRECAPTURE_ROOTZ_SCALE = 0.18
    V4_PRECAPTURE_PITCH_CENTER = 0.10
    V4_PRECAPTURE_PITCH_SCALE = 0.25
    V4_RECOVERY_PITCH_CENTER = 0.45
    V4_RECOVERY_PITCH_SCALE = 0.20
    V4_RECOVERY_COMP_CENTER = 0.18
    V4_RECOVERY_COMP_SCALE = 0.08
    V4_RECOVERY_SUPPORT_PITCH_CENTER = 0.25
    V4_RECOVERY_SUPPORT_PITCH_SCALE = 0.30
    V4_NONCOLLAPSED_ROOTZ_CENTER = -0.70
    V4_NONCOLLAPSED_ROOTZ_SCALE = 0.08
    V4_NONCOLLAPSED_HEIGHT_CENTER = 0.14
    V4_NONCOLLAPSED_HEIGHT_SCALE = 0.06
    V4_BRIDGE_RISE_ROOTZ_CENTER = -0.82
    V4_BRIDGE_RISE_ROOTZ_SCALE = 0.16
    V4_STANDUP_PITCH_LOW_CENTER = 0.35
    V4_STANDUP_PITCH_LOW_SCALE = 0.18
    V4_STANDUP_PITCH_HIGH_CENTER = 1.70
    V4_STANDUP_PITCH_HIGH_SCALE = 0.28
    V4_RELEASE_PITCH_LOW_CENTER = 0.55
    V4_RELEASE_PITCH_LOW_SCALE = 0.18
    V4_RELEASE_PITCH_HIGH_CENTER = 1.55
    V4_RELEASE_PITCH_HIGH_SCALE = 0.24
    V4_ARM_CORRIDOR_GAIN = 0.50
    V4_ARM_BODY_HIGH_GAIN = 0.20

    def __init__(
        self,
        dt: float = 0.02,
        hidden: int = 256,
        hop_speed: float = 2.0,
        residual_scale: float = 5.0,
        passive_scale: float = 10.0,
    ):
        super().__init__()
        del hidden, residual_scale, passive_scale
        self.dt = dt
        self.hop_speed = hop_speed

        self.register_buffer(
            "base_rate_damping",
            torch.tensor([0.10, 0.12, 0.10, 0.10], dtype=torch.float32),
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
        clearance = torch.minimum(kin["toe_z"], kin["heel_z"])
        return {
            "support": support,
            "clearance": clearance,
            "toe_vz": toe_vz,
            "heel_vz": heel_vz,
        }

    def _height_from_compression(self, compression: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.TARGET_STAND_HEIGHT - compression, min=0.0, max=2.5)

    def _body_low_from_rootz(self, rootz: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self.V4_ROOTZ_LOW_CENTER - rootz) / self.V4_ROOTZ_LOW_SCALE)

    def _body_high_from_rootz(self, rootz: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._body_low_from_rootz(rootz)

    def _clearance_from_state(
        self,
        compression: torch.Tensor,
        pitch: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        height = self._height_from_compression(compression)
        return height - 0.58 - 0.22 * compression - 0.08 * torch.abs(pitch) - 0.06 * support

    def _capture_pitch_corridor_from_state(self, pitch: torch.Tensor) -> torch.Tensor:
        lower_gate = torch.sigmoid((pitch - self.V4_CAPTURE_PITCH_LOW_CENTER) / self.V4_CAPTURE_PITCH_LOW_SCALE)
        upper_gate = torch.sigmoid((self.V4_CAPTURE_PITCH_HIGH_CENTER - pitch) / self.V4_CAPTURE_PITCH_HIGH_SCALE)
        return lower_gate * upper_gate

    def _front_contact_geom_from_state(
        self,
        pitch: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        support: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        body_low = self._body_low_from_rootz(rootz)
        clearance = self._clearance_from_state(
            compression,
            pitch,
            support if support is not None else torch.zeros_like(pitch),
        )
        capture_pitch = self._capture_pitch_corridor_from_state(pitch)
        capture_clearance = torch.sigmoid(((-clearance) - self.V4_CAPTURE_CLEARANCE_CENTER) / self.V4_CAPTURE_CLEARANCE_SCALE)
        return body_low * capture_pitch * capture_clearance

    def _capture_commitment(self, capture_memory: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((capture_memory - self.V4_CAPTURE_CENTER) / self.V4_CAPTURE_SCALE)

    def _recovered_upright_from_state(self, pitch: torch.Tensor, rootz: torch.Tensor) -> torch.Tensor:
        body_high = self._body_high_from_rootz(rootz)
        return body_high * torch.sigmoid((1.0 - torch.abs(pitch)) / self.V4_PITCH_NEUTRAL_SPREAD)

    def _handoff_ready_from_state(self, pitch: torch.Tensor, rootz: torch.Tensor) -> torch.Tensor:
        body_high = self._body_high_from_rootz(rootz)
        return body_high * torch.sigmoid((1.8 - torch.abs(pitch)) / 0.45)

    def _noncollapsed_posture_from_state(self, compression: torch.Tensor, rootz: torch.Tensor) -> torch.Tensor:
        height = self._height_from_compression(compression)
        rootz_gate = torch.sigmoid((rootz - self.V4_NONCOLLAPSED_ROOTZ_CENTER) / self.V4_NONCOLLAPSED_ROOTZ_SCALE)
        height_gate = torch.sigmoid((height - self.V4_NONCOLLAPSED_HEIGHT_CENTER) / self.V4_NONCOLLAPSED_HEIGHT_SCALE)
        return rootz_gate * height_gate

    def _body_rise_attempt_from_state(self, rootz: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((rootz - self.V4_BRIDGE_RISE_ROOTZ_CENTER) / self.V4_BRIDGE_RISE_ROOTZ_SCALE)

    def _crawl_forward_progress_from_state(
        self,
        vx: torch.Tensor,
        support: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        capture_commitment = self._capture_commitment(capture_memory)
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        speed_gate = torch.clamp(vx / 0.60, min=0.0, max=1.0)
        return capture_commitment * grounded * speed_gate

    def _standup_pitch_corridor_from_state(self, pitch: torch.Tensor) -> torch.Tensor:
        lower_gate = torch.sigmoid((pitch - self.V4_STANDUP_PITCH_LOW_CENTER) / self.V4_STANDUP_PITCH_LOW_SCALE)
        upper_gate = torch.sigmoid((self.V4_STANDUP_PITCH_HIGH_CENTER - pitch) / self.V4_STANDUP_PITCH_HIGH_SCALE)
        return lower_gate * upper_gate

    @staticmethod
    def _release_engagement_from_phase(release_phase: torch.Tensor) -> torch.Tensor:
        return torch.clamp((release_phase - 0.08) / 0.18, min=0.0, max=1.0)

    def _release_pitch_corridor_from_state(self, pitch: torch.Tensor) -> torch.Tensor:
        lower_gate = torch.sigmoid((pitch - self.V4_RELEASE_PITCH_LOW_CENTER) / self.V4_RELEASE_PITCH_LOW_SCALE)
        upper_gate = torch.sigmoid((self.V4_RELEASE_PITCH_HIGH_CENTER - pitch) / self.V4_RELEASE_PITCH_HIGH_SCALE)
        return lower_gate * upper_gate

    def _recovery_support_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        capture_memory: torch.Tensor,
        rootz: torch.Tensor,
    ) -> torch.Tensor:
        recovery_support_gate = torch.sigmoid(
            (pitch - self.V4_RECOVERY_SUPPORT_PITCH_CENTER) / self.V4_RECOVERY_SUPPORT_PITCH_SCALE
        )
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        body_high = self._body_high_from_rootz(rootz)
        capture_commitment = self._capture_commitment(capture_memory)
        handoff_unlock = capture_commitment * torch.sigmoid(
            (body_high - self.V4_HANDOFF_BODY_HIGH_CENTER) / self.V4_HANDOFF_BODY_HIGH_SCALE
        )
        support_corridor = torch.maximum(
            grounded,
            0.45 * handoff_unlock * torch.clamp(0.25 + 0.75 * body_high, min=0.0, max=1.0),
        )
        return capture_commitment * recovery_support_gate * torch.clamp(0.45 + 0.55 * support_corridor, min=0.0, max=1.0)

    def _standup_window_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        capture_memory: torch.Tensor,
        rootz: torch.Tensor,
    ) -> torch.Tensor:
        capture_commitment = self._capture_commitment(capture_memory)
        recovery_support = self._recovery_support_from_state(pitch, support, capture_memory, rootz)
        body_low = self._body_low_from_rootz(rootz)
        recovery_pitch = torch.sigmoid((pitch - self.V4_RECOVERY_PITCH_CENTER) / self.V4_RECOVERY_PITCH_SCALE)
        compression_arm = torch.sigmoid((compression - self.V4_RECOVERY_COMP_CENTER) / self.V4_RECOVERY_COMP_SCALE)
        noncollapsed = self._noncollapsed_posture_from_state(compression, rootz)
        pitch_corridor = self._standup_pitch_corridor_from_state(pitch)
        return capture_commitment * recovery_support * body_low * recovery_pitch * compression_arm * noncollapsed * pitch_corridor

    def _arm_ready_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        capture_memory: torch.Tensor,
        rootz: torch.Tensor,
    ) -> torch.Tensor:
        capture_commitment = self._capture_commitment(capture_memory)
        recovery_support = self._recovery_support_from_state(pitch, support, capture_memory, rootz)
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        body_high = self._body_high_from_rootz(rootz)
        recovered_upright = self._recovered_upright_from_state(pitch, rootz)
        handoff_ready = self._handoff_ready_from_state(pitch, rootz)
        recovery_pitch = torch.sigmoid((pitch - self.V4_RECOVERY_PITCH_CENTER) / self.V4_RECOVERY_PITCH_SCALE)
        compression_arm = torch.sigmoid((compression - self.V4_RECOVERY_COMP_CENTER) / self.V4_RECOVERY_COMP_SCALE)
        noncollapsed = self._noncollapsed_posture_from_state(compression, rootz)
        standup_corridor = self._standup_pitch_corridor_from_state(pitch)
        arm_corridor = torch.maximum(
            torch.maximum(recovered_upright, 0.5 * handoff_ready),
            self.V4_ARM_CORRIDOR_GAIN * recovery_pitch * compression_arm * standup_corridor
            + self.V4_ARM_BODY_HIGH_GAIN * body_high,
        )
        support_lane = torch.clamp(grounded + 0.45 * handoff_ready, min=0.0, max=1.0)
        return capture_commitment * recovery_support * support_lane * noncollapsed * torch.clamp(arm_corridor, min=0.0, max=1.0)

    def _rear_recovery_phase_from_state(self, pitch: torch.Tensor, rootz: torch.Tensor) -> torch.Tensor:
        body_low = self._body_low_from_rootz(rootz)
        backward_tilt = torch.sigmoid(-(pitch - self.V4_PITCH_BACK_CENTER) / self.V4_PITCH_BACK_SCALE)
        return body_low * backward_tilt

    def _precapture_phase_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        capture_commitment = self._capture_commitment(capture_memory)
        soft_body_low = torch.sigmoid((self.V4_PRECAPTURE_ROOTZ_CENTER - rootz) / self.V4_PRECAPTURE_ROOTZ_SCALE)
        forward_gate = (
            torch.sigmoid((pitch - self.V4_PRECAPTURE_PITCH_CENTER) / self.V4_PRECAPTURE_PITCH_SCALE)
            * self._capture_pitch_corridor_from_state(pitch)
        )
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        contact_geom = self._front_contact_geom_from_state(pitch, compression, rootz, support)
        rear_recovery = self._rear_recovery_phase_from_state(pitch, rootz)
        return grounded * soft_body_low * forward_gate * torch.maximum(grounded, contact_geom) * (1.0 - capture_commitment) * (1.0 - 0.75 * rear_recovery)

    def _capture_crawl_phase_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        capture_commitment = self._capture_commitment(capture_memory)
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        body_low = self._body_low_from_rootz(rootz)
        capture_corridor = self._capture_pitch_corridor_from_state(pitch)
        rear_recovery = self._rear_recovery_phase_from_state(pitch, rootz)
        return grounded * body_low * capture_commitment * capture_corridor * (1.0 - 0.75 * rear_recovery)

    def _stable_capture_phase_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        precapture_phase = self._precapture_phase_from_state(pitch, support, compression, rootz, capture_memory)
        contact_geom = self._front_contact_geom_from_state(pitch, compression, rootz, support)
        grounded = torch.clamp((support - 0.12) / 0.40, min=0.0, max=1.0)
        capture_commitment = self._capture_commitment(capture_memory)
        build_gate = torch.clamp((0.75 - capture_commitment) / 0.55, min=0.0, max=1.0)
        return precapture_phase * torch.maximum(contact_geom, 0.45 * grounded) * build_gate

    def _standup_phase_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        standup_window = self._standup_window_from_state(pitch, support, compression, capture_memory, rootz)
        arm_ready = self._arm_ready_from_state(pitch, support, compression, capture_memory, rootz)
        standup_corridor = self._standup_pitch_corridor_from_state(pitch)
        rise_gate = torch.clamp((arm_ready - 0.04) / 0.20, min=0.0, max=1.0)
        return standup_window * torch.maximum(0.25 * standup_corridor, rise_gate)

    def _crawl_to_standup_bias_from_state(
        self,
        vx: torch.Tensor,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        capture_phase = self._capture_crawl_phase_from_state(pitch, support, rootz, capture_memory)
        recovery_support = self._recovery_support_from_state(pitch, support, capture_memory, rootz)
        noncollapsed = self._noncollapsed_posture_from_state(compression, rootz)
        body_rise_attempt = self._body_rise_attempt_from_state(rootz)
        crawl_forward_progress = self._crawl_forward_progress_from_state(vx, support, capture_memory)
        standup_corridor = self._standup_pitch_corridor_from_state(pitch)
        compression_arm = torch.sigmoid((compression - self.V4_RECOVERY_COMP_CENTER) / self.V4_RECOVERY_COMP_SCALE)
        bridge_posture = torch.maximum(noncollapsed, torch.maximum(body_rise_attempt, crawl_forward_progress))
        return capture_phase * recovery_support * bridge_posture * standup_corridor * torch.clamp(0.30 + 0.70 * compression_arm, min=0.0, max=1.0)

    def _release_phase_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        arm_ready = self._arm_ready_from_state(pitch, support, compression, capture_memory, rootz)
        handoff_ready = self._handoff_ready_from_state(pitch, rootz)
        noncollapsed = self._noncollapsed_posture_from_state(compression, rootz)
        compression_gate = torch.sigmoid((compression - (self.V4_RECOVERY_COMP_CENTER + 0.08)) / 0.06)
        pitch_gate = self._release_pitch_corridor_from_state(pitch)
        release_corridor = torch.maximum(arm_ready, 0.60 * handoff_ready)
        return release_corridor * compression_gate * pitch_gate * noncollapsed

    def _handoff_phase_from_state(
        self,
        pitch: torch.Tensor,
        support: torch.Tensor,
        compression: torch.Tensor,
        rootz: torch.Tensor,
        capture_memory: torch.Tensor,
    ) -> torch.Tensor:
        release_phase = self._release_phase_from_state(pitch, support, compression, rootz, capture_memory)
        handoff_ready = self._handoff_ready_from_state(pitch, rootz)
        body_high = self._body_high_from_rootz(rootz)
        release_engagement = self._release_engagement_from_phase(release_phase)
        release_bridge = torch.maximum(0.35 * release_phase, handoff_ready * release_engagement)
        return body_high * release_bridge

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

        height = self._explicit_height_from_pos(pos)
        vx = vel[:, 0]
        vz = vel[:, 1]
        pitch = pos[:, 1]
        pitch_rate = vel[:, 2]
        compression = torch.relu(self.TARGET_STAND_HEIGHT - height)
        compression_rate = -vz
        support = contact["support"]
        rootz = pos[:, 0]
        capture_seed = torch.clamp(
            0.65 * self._front_contact_geom_from_state(pitch, compression, rootz, support)
            + 0.35 * torch.clamp((support - 0.10) / 0.35, min=0.0, max=1.0) * self._capture_pitch_corridor_from_state(pitch),
            min=0.0,
            max=1.0,
        )

        latent = torch.stack(
            [
                vx,
                vz,
                pitch,
                pitch_rate,
                compression,
                compression_rate,
                support,
                rootz,
                capture_seed,
            ],
            dim=-1,
        )
        return latent.reshape(*leading, self.STATE_DIM)

    def _explicit_rate_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        scaled_action = action * self.control_scale.unsqueeze(0)

        vx = state[:, self.VX_IDX]
        vz = state[:, self.VZ_IDX]
        pitch = state[:, self.PITCH_IDX]
        pitch_rate = state[:, self.PITCH_RATE_IDX]
        compression = state[:, self.COMP_IDX]
        compression_rate = state[:, self.COMP_RATE_IDX]
        support = torch.clamp(state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        rootz = state[:, self.ROOTZ_IDX]
        capture_memory = torch.clamp(state[:, self.CAPTURE_IDX], min=0.0, max=1.0)

        # Action sign convention follows the validated v3/v4 hopper control prior:
        # negative hip with positive knee/ankle corresponds to a release/liftoff push.
        drive_action = 0.35 * scaled_action[:, 0] + 0.30 * scaled_action[:, 1] - 0.20 * scaled_action[:, 2] + 0.15 * scaled_action[:, 3]
        posture_action = 0.45 * scaled_action[:, 0] - 0.20 * scaled_action[:, 1] + 0.20 * scaled_action[:, 2] + 0.15 * scaled_action[:, 3]
        release_score = (
            torch.clamp(-scaled_action[:, 1], min=0.0, max=1.0)
            + torch.clamp(scaled_action[:, 2], min=0.0, max=1.0)
            + torch.clamp(scaled_action[:, 3], min=0.0, max=1.0)
        ) / 3.0
        release_drive = (
            0.25 * torch.clamp(scaled_action[:, 0], min=0.0, max=1.0)
            + 0.35 * torch.clamp(-scaled_action[:, 1], min=0.0, max=1.0)
            + 0.25 * torch.clamp(scaled_action[:, 2], min=0.0, max=1.0)
            + 0.15 * torch.clamp(scaled_action[:, 3], min=0.0, max=1.0)
        )
        rear_escape_drive = (
            torch.clamp(scaled_action[:, 0], min=0.0, max=1.0)
            + torch.clamp(-scaled_action[:, 1], min=0.0, max=1.0)
            + torch.clamp(scaled_action[:, 2], min=0.0, max=1.0)
            + torch.clamp(scaled_action[:, 3], min=0.0, max=1.0)
        ) / 4.0
        rear_collapse_drive = (
            torch.clamp(-scaled_action[:, 0], min=0.0, max=1.0)
            + torch.clamp(scaled_action[:, 1], min=0.0, max=1.0)
            + torch.clamp(-scaled_action[:, 2], min=0.0, max=1.0)
            + torch.clamp(-scaled_action[:, 3], min=0.0, max=1.0)
        ) / 4.0

        body_low = self._body_low_from_rootz(rootz)
        body_high = 1.0 - body_low
        rear_phase = self._rear_recovery_phase_from_state(pitch, rootz)
        stable_capture = self._stable_capture_phase_from_state(pitch, support, compression, rootz, capture_memory)
        capture_commitment = self._capture_commitment(capture_memory)
        recovery_support = self._recovery_support_from_state(pitch, support, capture_memory, rootz)
        standup_window = self._standup_window_from_state(pitch, support, compression, capture_memory, rootz)
        arm_ready = self._arm_ready_from_state(pitch, support, compression, capture_memory, rootz)
        release_phase = self._release_phase_from_state(pitch, support, compression, rootz, capture_memory)
        release_window = torch.maximum(standup_window, 0.60 * release_phase)
        clearance = self._clearance_from_state(compression, pitch, support)
        contact_geom = self._front_contact_geom_from_state(pitch, compression, rootz, support)
        grounded = torch.maximum(support, 0.30 * contact_geom)
        contact_lock = grounded * torch.sigmoid((-clearance - 0.70) / 0.06)
        release_lift = release_window * torch.relu(release_drive - 0.35) * (1.0 - 0.60 * contact_lock)
        recovery_release = release_window * (0.35 + 0.65 * release_score)

        drive_support = torch.clamp(0.80 * grounded + 0.20 * recovery_support, min=0.0, max=1.0)
        lift_support = grounded
        posture_support = torch.maximum(grounded, 0.20 * recovery_support)
        air = 1.0 - grounded
        impact = lift_support * torch.relu(-vz)
        upright_window = recovery_support * body_high * torch.sigmoid((1.0 - torch.abs(pitch)) / self.V4_PITCH_NEUTRAL_SPREAD)
        anchor_pitch_drive = 0.35 * recovery_support * torch.relu(pitch) * (1.0 - 0.70 * recovery_release)
        stance_impact_pitch = -2.8 * grounded * torch.relu(-vz) * torch.clamp(1.0 - 0.85 * contact_geom, min=0.0, max=1.0)
        front_capture_pitch = 3.4 * contact_geom * torch.relu(-vz) * torch.clamp(1.0 - support, min=0.0, max=1.0)
        stable_capture_recoil = stable_capture * torch.relu(-vz) * torch.clamp(1.0 - 0.55 * support, min=0.0, max=1.0)
        pitch_turnover = 3.2 * torch.maximum(
            grounded * torch.relu(-vz),
            release_lift + 0.50 * recovery_release,
        ) * (-torch.tanh(pitch_rate / 4.0))

        x_acc = drive_support * (0.95 * drive_action + 0.20 * torch.relu(pitch) - 0.20 * vx) + 0.25 * release_window * release_score - 0.06 * air * vx
        z_acc = -4.8 - 1.4 * air + lift_support * (9.5 * compression + 1.1 * impact - 1.3 * torch.relu(vz)) + 4.2 * recovery_release + 4.8 * release_lift + 1.4 * upright_window + 1.2 * stable_capture_recoil + 1.8 * rear_phase * rear_escape_drive - 1.2 * rear_phase * rear_collapse_drive
        pitch_acc = -1.8 * pitch - 0.65 * pitch_rate + posture_support * (0.22 * posture_action) + anchor_pitch_drive + stance_impact_pitch + front_capture_pitch + 1.8 * stable_capture_recoil + pitch_turnover - 0.60 * recovery_release * torch.clamp(pitch, min=0.0, max=2.5) - 0.45 * upright_window * pitch + 2.2 * rear_phase * rear_escape_drive - 1.6 * rear_phase * rear_collapse_drive
        compression_acc = lift_support * (1.25 * torch.relu(-vz) + 0.55 * impact - 2.4 * release_lift) - 0.90 * compression - 0.35 * compression_rate + 0.12 * air * compression + 0.10 * grounded * compression - 0.55 * recovery_release * compression

        return torch.cat(
            [
                x_acc.unsqueeze(-1),
                z_acc.unsqueeze(-1),
                pitch_acc.unsqueeze(-1),
                compression_acc.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _rate_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        explicit = self._explicit_rate_dynamics(state, action)
        damping = self.rate_damping.unsqueeze(0) * state[:, list(self.RATE_LIKE_IDX)]
        return explicit - damping

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        cols = [state[:, i] for i in range(self.STATE_DIM)]
        cols[self.PITCH_IDX] = _wrap_angle_torch(cols[self.PITCH_IDX])
        cols[self.COMP_IDX] = torch.clamp(cols[self.COMP_IDX], min=0.0, max=1.5)
        cols[self.SUPPORT_IDX] = torch.clamp(cols[self.SUPPORT_IDX], min=0.0, max=1.0)
        cols[self.ROOTZ_IDX] = torch.clamp(cols[self.ROOTZ_IDX], min=-1.5, max=1.5)
        cols[self.CAPTURE_IDX] = torch.clamp(cols[self.CAPTURE_IDX], min=0.0, max=1.0)
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

        pitch = state[:, self.PITCH_IDX]
        pitch_rate = state[:, self.PITCH_RATE_IDX]
        compression = state[:, self.COMP_IDX]
        support = torch.clamp(state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        rootz = state[:, self.ROOTZ_IDX]
        capture_memory = torch.clamp(state[:, self.CAPTURE_IDX], min=0.0, max=1.0)
        vz = state[:, self.VZ_IDX]
        capture_commitment = self._capture_commitment(capture_memory)

        release_score = (
            torch.clamp(-action[:, 1], min=0.0, max=1.0)
            + torch.clamp(action[:, 2], min=0.0, max=1.0)
            + torch.clamp(action[:, 3], min=0.0, max=1.0)
        ) / 3.0
        release_drive = (
            0.25 * torch.clamp(action[:, 0], min=0.0, max=1.0)
            + 0.35 * torch.clamp(-action[:, 1], min=0.0, max=1.0)
            + 0.25 * torch.clamp(action[:, 2], min=0.0, max=1.0)
            + 0.15 * torch.clamp(action[:, 3], min=0.0, max=1.0)
        )

        standup_window = self._standup_window_from_state(pitch, support, compression, capture_memory, rootz)
        release_phase = self._release_phase_from_state(pitch, support, compression, rootz, capture_memory)
        release_window = torch.maximum(standup_window, 0.60 * release_phase)
        recovery_release = release_window * (0.35 + 0.65 * release_score)
        release_lift = release_window * torch.relu(release_drive - 0.35)

        next_pitch = _wrap_angle_torch(pitch + self.dt * next_pitch_rate)
        next_rootz = torch.clamp(rootz + self.dt * next_vz, min=-1.5, max=1.5)
        next_comp_rate = 0.60 * next_comp_rate + 0.40 * (-next_vz)
        next_comp = torch.clamp(compression + self.dt * next_comp_rate, min=0.0, max=1.5)

        next_clearance = self._clearance_from_state(next_comp, next_pitch, support)
        contact_geom = torch.sigmoid((self.CONTACT_MARGIN - next_clearance) / 0.04)
        impact_gate = torch.sigmoid((0.12 - next_vz) / 0.22)
        contact_stick = torch.sigmoid((0.02 - next_vz) / 0.10) * torch.sigmoid((0.02 - next_clearance) / 0.08)
        support_hold = 0.18 * capture_commitment * support * contact_stick * torch.sigmoid((-next_clearance - 0.10) / 0.10)
        attach_rate = 0.55 * torch.maximum(contact_geom * impact_gate, contact_geom * contact_stick)
        liftoff_release = recovery_release * torch.sigmoid((next_vz - 0.04) / 0.10)
        detach_rate = 0.45 * torch.maximum(
            liftoff_release,
            (1.0 - contact_geom) * torch.sigmoid((next_vz - 0.04) / 0.10),
        )
        next_support = torch.clamp(
            support
            + attach_rate * (1.0 - support)
            + support_hold * (1.0 - support)
            - detach_rate * support,
            min=0.0,
            max=1.0,
        )

        next_front_contact = self._front_contact_geom_from_state(next_pitch, next_comp, next_rootz, next_support)
        next_rear_trap = self._rear_recovery_phase_from_state(next_pitch, next_rootz)
        next_precapture = self._precapture_phase_from_state(next_pitch, next_support, next_comp, next_rootz, capture_memory)
        next_stable_capture = self._stable_capture_phase_from_state(next_pitch, next_support, next_comp, next_rootz, capture_memory)
        release_decay = (1.0 - self._body_low_from_rootz(next_rootz)) * recovery_release
        support_commit = next_support * self._capture_pitch_corridor_from_state(next_pitch)
        next_capture = torch.clamp(
            0.72 * capture_memory
            + 0.28 * next_front_contact
            + 0.22 * next_precapture * (1.0 - capture_commitment)
            + 0.18 * next_stable_capture
            + 0.14 * support_commit
            - 0.18 * next_rear_trap
            - 0.10 * release_decay,
            min=0.0,
            max=1.0,
        )

        return self.project_state(
            torch.stack(
                [
                    next_vx,
                    next_vz,
                    next_pitch,
                    next_pitch_rate,
                    next_comp,
                    next_comp_rate,
                    next_support,
                    next_rootz,
                    next_capture,
                ],
                dim=-1,
            )
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

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
        return self._height_from_compression(torch.clamp(state[:, self.COMP_IDX], min=0.0))

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
        height = self.predict_height(next_state)
        pitch = next_state[:, self.PITCH_IDX]
        rootz = next_state[:, self.ROOTZ_IDX]
        capture = next_state[:, self.CAPTURE_IDX]

        height_gate = self._lower_tolerance(height, lower=0.45, margin=0.15, value_at_margin=0.0)
        forward_lean = torch.sigmoid((pitch - self.V4_PITCH_FWD_CENTER) / self.V4_PITCH_FWD_SCALE)
        recovered_upright = self._recovered_upright_from_state(pitch, rootz)
        rear_trap = self._rear_recovery_phase_from_state(pitch, rootz)
        viable_lane = torch.maximum(forward_lean * self._capture_commitment(capture), recovered_upright)
        return height_gate * viable_lane * (1.0 - rear_trap) * true_reward

    def predict_plan_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if next_state is None:
            next_state = self.forward(state, action)

        true_reward = self.predict_reward(state, action, next_state)
        x_dot = next_state[:, self.VX_IDX]
        z_dot = next_state[:, self.VZ_IDX]
        pitch = next_state[:, self.PITCH_IDX]
        prev_pitch = state[:, self.PITCH_IDX]
        compression = torch.clamp(next_state[:, self.COMP_IDX], min=0.0)
        prev_compression = torch.clamp(state[:, self.COMP_IDX], min=0.0)
        support = torch.clamp(next_state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        prev_support = torch.clamp(state[:, self.SUPPORT_IDX], min=0.0, max=1.0)
        prev_rootz = state[:, self.ROOTZ_IDX]
        next_rootz = next_state[:, self.ROOTZ_IDX]
        capture = torch.clamp(next_state[:, self.CAPTURE_IDX], min=0.0, max=1.0)
        prev_capture = torch.clamp(state[:, self.CAPTURE_IDX], min=0.0, max=1.0)

        capture_commitment = self._capture_commitment(capture)
        prev_capture_commitment = self._capture_commitment(prev_capture)
        recovery_support = self._recovery_support_from_state(pitch, support, capture, next_rootz)
        prev_recovery_support = self._recovery_support_from_state(prev_pitch, prev_support, prev_capture, prev_rootz)
        front_contact = self._front_contact_geom_from_state(pitch, compression, next_rootz, support)
        prev_front_contact = self._front_contact_geom_from_state(prev_pitch, prev_compression, prev_rootz, prev_support)
        height = self.predict_height(next_state)

        prev_body_low = self._body_low_from_rootz(prev_rootz)
        body_low = self._body_low_from_rootz(next_rootz)
        prev_forward_lean = torch.sigmoid((prev_pitch - self.V4_PITCH_FWD_CENTER) / self.V4_PITCH_FWD_SCALE)
        forward_lean = torch.sigmoid((pitch - self.V4_PITCH_FWD_CENTER) / self.V4_PITCH_FWD_SCALE)
        prev_recovered_upright = self._recovered_upright_from_state(prev_pitch, prev_rootz)
        prev_handoff_ready = self._handoff_ready_from_state(prev_pitch, prev_rootz)
        recovered_upright = self._recovered_upright_from_state(pitch, next_rootz)
        handoff_ready = self._handoff_ready_from_state(pitch, next_rootz)
        viable_lane = torch.maximum(forward_lean * torch.maximum(capture_commitment, front_contact), recovered_upright)
        prev_rear_trap = self._rear_recovery_phase_from_state(prev_pitch, prev_rootz)
        rear_trap = self._rear_recovery_phase_from_state(pitch, next_rootz)
        height_gate = self._lower_tolerance(height, lower=0.45, margin=0.15, value_at_margin=0.0)
        viability = height_gate * viable_lane * (1.0 - rear_trap)
        recovery_gate = self._lower_tolerance(next_rootz, lower=-0.95, margin=0.35, value_at_margin=0.0) * (1.0 - rear_trap)
        crawl_lane = body_low * forward_lean * capture_commitment

        speed_progress = torch.clamp(x_dot / max(self.hop_speed, 1e-6), min=0.0, max=1.0)
        upward_progress = torch.clamp(z_dot / 0.60, min=0.0, max=1.0)
        liftoff_progress = torch.clamp((0.18 - support) / 0.18, min=0.0, max=1.0)
        sticky_contact = torch.clamp((support - 0.55) / 0.35, min=0.0, max=1.0) * torch.clamp((-z_dot) / 0.35, min=0.0, max=1.0)
        low_motion = torch.clamp((0.08 - torch.abs(x_dot)) / 0.08, min=0.0, max=1.0) * torch.clamp((0.08 - torch.abs(z_dot)) / 0.08, min=0.0, max=1.0)
        frozen_capture = (
            torch.clamp((support - 0.80) / 0.20, min=0.0, max=1.0)
            * capture_commitment
            * low_motion
            * torch.clamp((pitch - 1.10) / 0.55, min=0.0, max=1.0)
        )
        compression_level = torch.clamp(compression / 0.35, min=0.0, max=1.0)
        release_score = (
            torch.clamp(-action[:, 1], min=0.0, max=1.0)
            + torch.clamp(action[:, 2], min=0.0, max=1.0)
            + torch.clamp(action[:, 3], min=0.0, max=1.0)
        ) / 3.0
        rear_escape_drive = (
            torch.clamp(action[:, 0], min=0.0, max=1.0)
            + torch.clamp(-action[:, 1], min=0.0, max=1.0)
            + torch.clamp(action[:, 2], min=0.0, max=1.0)
            + torch.clamp(action[:, 3], min=0.0, max=1.0)
        ) / 4.0
        rear_collapse_drive = (
            torch.clamp(-action[:, 0], min=0.0, max=1.0)
            + torch.clamp(action[:, 1], min=0.0, max=1.0)
            + torch.clamp(-action[:, 2], min=0.0, max=1.0)
            + torch.clamp(-action[:, 3], min=0.0, max=1.0)
        ) / 4.0
        release_liftoff = torch.maximum(upward_progress, liftoff_progress)
        pitch_gain = torch.clamp(pitch - prev_pitch, min=0.0, max=0.45) / 0.45
        capture_gain = torch.clamp(capture - prev_capture, min=0.0, max=0.35) / 0.35
        contact_gain = torch.clamp(front_contact - prev_front_contact, min=0.0, max=0.35) / 0.35

        prev_standup_window = self._standup_window_from_state(prev_pitch, prev_support, prev_compression, prev_capture, prev_rootz)
        standup_window = self._standup_window_from_state(pitch, support, compression, capture, next_rootz)
        prev_arm_ready = self._arm_ready_from_state(prev_pitch, prev_support, prev_compression, prev_capture, prev_rootz)
        arm_ready = self._arm_ready_from_state(pitch, support, compression, capture, next_rootz)
        prev_precapture_phase = self._precapture_phase_from_state(prev_pitch, prev_support, prev_compression, prev_rootz, prev_capture)
        prev_capture_phase = self._capture_crawl_phase_from_state(prev_pitch, prev_support, prev_rootz, prev_capture)
        prev_standup_phase = self._standup_phase_from_state(prev_pitch, prev_support, prev_compression, prev_rootz, prev_capture)
        prev_release_phase = self._release_phase_from_state(prev_pitch, prev_support, prev_compression, prev_rootz, prev_capture)
        prev_handoff_phase = self._handoff_phase_from_state(prev_pitch, prev_support, prev_compression, prev_rootz, prev_capture)
        prev_crawl_to_standup_bias = self._crawl_to_standup_bias_from_state(
            state[:, self.VX_IDX],
            prev_pitch,
            prev_support,
            prev_compression,
            prev_rootz,
            prev_capture,
        )
        crawl_to_standup_bias = self._crawl_to_standup_bias_from_state(
            x_dot,
            pitch,
            support,
            compression,
            next_rootz,
            capture,
        )
        prev_stable_capture_phase = self._stable_capture_phase_from_state(prev_pitch, prev_support, prev_compression, prev_rootz, prev_capture)
        stable_capture_phase = self._stable_capture_phase_from_state(pitch, support, compression, next_rootz, capture)

        arm_ready_gain = torch.clamp(arm_ready - prev_arm_ready, min=0.0, max=0.25) / 0.25
        body_rise = torch.clamp(prev_body_low - body_low, min=0.0, max=0.45) / 0.45
        upright_gain = torch.clamp(recovered_upright - prev_recovered_upright, min=0.0, max=0.45) / 0.45
        capture_release = torch.clamp(prev_capture - capture, min=0.0, max=0.35) / 0.35
        rootz_gain_progress = torch.clamp(next_rootz - prev_rootz, min=0.0, max=0.25) / 0.25
        handoff_ready_gain = torch.clamp(handoff_ready - prev_handoff_ready, min=0.0, max=0.20) / 0.20
        recovery_support_gain = torch.clamp(recovery_support - prev_recovery_support, min=-0.20, max=0.20) / 0.20
        crawl_to_standup_gain = torch.clamp(crawl_to_standup_bias - prev_crawl_to_standup_bias, min=0.0, max=0.25) / 0.25
        stable_capture_gain = torch.clamp(stable_capture_phase - prev_stable_capture_phase, min=0.0, max=0.30) / 0.30
        unsupported_low = body_low * (1.0 - capture_commitment) * torch.sigmoid((self.V4_CAPTURE_PITCH_CENTER - pitch) / self.V4_CAPTURE_PITCH_SCALE)
        rear_flight = rear_trap * torch.clamp(z_dot, min=0.0, max=1.0)

        rear_gate = recovery_gate * prev_rear_trap
        precapture_gate = recovery_gate * prev_precapture_phase
        crawl_gate = recovery_gate * prev_capture_phase
        standup_gate = recovery_gate * torch.maximum(prev_standup_phase, 0.40 * prev_crawl_to_standup_bias)
        release_gate = recovery_gate * prev_release_phase
        handoff_gate = recovery_gate * prev_handoff_phase
        premature_release = recovery_gate * torch.maximum(prev_rear_trap, prev_precapture_phase) * release_score

        positive = viability * (0.12 * speed_progress + 0.08 * upward_progress)
        rear_escape_progress = rear_gate * (
            0.50 * pitch_gain
            + 0.28 * capture_gain
            + 0.12 * torch.clamp(recovery_support_gain, min=0.0, max=1.0)
            + 0.10 * rear_escape_drive
        )
        precapture_progress = precapture_gate * (
            0.34 * capture_commitment
            + 0.24 * capture_gain
            + 0.22 * front_contact
            + 0.20 * contact_gain
        )
        stable_capture_progress = precapture_gate * (
            0.34 * stable_capture_phase
            + 0.24 * stable_capture_gain
            + 0.18 * capture_gain
            + 0.14 * contact_gain
            + 0.10 * torch.clamp(recovery_support_gain, min=0.0, max=1.0)
        )
        crawl_phase_now = self._capture_crawl_phase_from_state(pitch, support, next_rootz, capture)
        crawl_progress = crawl_gate * crawl_phase_now * (
            (0.70 - 0.22 * prev_crawl_to_standup_bias) * speed_progress
            + 0.18 * capture_gain
            + 0.12 * torch.clamp(recovery_support_gain, min=0.0, max=1.0)
        )
        standup_transfer_progress = crawl_gate * prev_crawl_to_standup_bias * (
            0.30 * body_rise
            + 0.24 * arm_ready_gain
            + 0.18 * rootz_gain_progress
            + 0.16 * upright_gain
            + 0.12 * crawl_to_standup_gain
        )
        standup_progress = standup_gate * (
            0.34 * body_rise
            + 0.28 * upright_gain
            + 0.22 * arm_ready_gain
            + 0.16 * torch.clamp(recovery_support_gain, min=0.0, max=1.0)
        )
        release_progress = release_gate * (
            0.40 * compression_level * release_score * release_liftoff
            + 0.24 * rootz_gain_progress
            + 0.20 * body_rise
            + 0.16 * upward_progress
        )
        handoff_progress = handoff_gate * (
            0.30 * capture_release
            + 0.24 * release_liftoff
            + 0.18 * rootz_gain_progress
            + 0.16 * handoff_ready_gain
            + 0.12 * speed_progress
        )

        penalty = (
            0.10 * sticky_contact
            + 0.16 * frozen_capture
            + 0.20 * rear_trap
            + 0.12 * rear_flight
            + 0.06 * unsupported_low * (1.0 - 0.75 * body_rise)
            + 0.05 * crawl_lane * (1.0 - capture_commitment)
            + 0.04 * crawl_gate * prev_crawl_to_standup_bias * sticky_contact * (1.0 - torch.maximum(body_rise, arm_ready_gain))
            + 0.12 * premature_release * (1.0 - torch.maximum(capture_commitment, 0.5 * body_rise))
            + 0.08 * rear_gate * rear_collapse_drive
        )

        return (
            true_reward
            + positive
            + 0.08 * rear_escape_progress
            + 0.08 * precapture_progress
            + 0.08 * stable_capture_progress
            + 0.10 * crawl_progress
            + 0.12 * standup_transfer_progress
            + 0.14 * standup_progress
            + 0.16 * release_progress
            + 0.18 * handoff_progress
            - penalty
        )

    def get_learned_params(self) -> Dict[str, object]:
        return {
            "dt": self.dt,
            "hop_speed": self.hop_speed,
            "control_scale": self.control_scale.detach().cpu().tolist(),
            "rate_damping": self.rate_damping.detach().cpu().tolist(),
            "observable_dim": self.OBSERVABLE_DIM,
            "latent_layout": [
                "vx",
                "vz",
                "pitch",
                "pitch_rate",
                "compression",
                "compression_rate",
                "support",
                "rootz",
                "capture_memory",
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
