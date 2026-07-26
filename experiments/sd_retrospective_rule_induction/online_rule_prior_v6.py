#!/usr/bin/env python3
"""Runtime MPPI reweighting for v6 retrospective-rule online smoke tests."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


JOINT_TO_DIM = {"waist": 0, "hip": 1, "knee": 2, "ankle": 3}
VALID_ACTION_FAMILIES = {
    "waist_pos",
    "waist_neg",
    "hip_pos",
    "hip_neg",
    "knee_pos",
    "knee_neg",
    "ankle_pos",
    "ankle_neg",
}
ACTION_FAMILY_INTERVENTIONS = {"soft_prefer_action_family", "avoid_action_family"}
STAGE_EFFECT_INTERVENTIONS = {"stage_effect_prior"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


class HopperTVAValueNet(nn.Module):
    """TVA value network architecture used by the Hopper TVA explainer."""

    def __init__(self, state_dim: int = 17, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _gumbel_softmax_sample(p: torch.Tensor, temperature: float = 1.0, dim: int = 0) -> torch.Tensor:
    logits = p.clamp_min(1e-12).log()
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    y_soft = ((logits + gumbels) / temperature).softmax(dim)
    return y_soft.argmax(-1)


def _load_replay_validated_rules(path: Path, allowed_statuses: set[str] | None = None) -> list[dict[str, Any]]:
    with path.open("r") as f:
        payload = json.load(f)
    statuses = allowed_statuses or {"replay_validated"}
    rules: list[dict[str, Any]] = []
    for rule in payload.get("rules", []):
        if rule.get("status") not in statuses:
            continue
        intervention = rule.get("intervention_type")
        if intervention in ACTION_FAMILY_INTERVENTIONS and str(rule.get("action_family", "")) not in VALID_ACTION_FAMILIES:
            continue
        if rule.get("source") == "v4_substate" and intervention == "soft_prefer_action_family":
            rules.append(rule)
        elif rule.get("source") == "v8_transition_replay_safety" and intervention in ACTION_FAMILY_INTERVENTIONS:
            rules.append(rule)
        elif rule.get("source") == "sd_diagnostic_toe_recovery" and intervention in ACTION_FAMILY_INTERVENTIONS:
            rules.append(rule)
        elif rule.get("source") == "sd_tva_broad_context_counterfactual" and intervention in ACTION_FAMILY_INTERVENTIONS:
            rules.append(rule)
        elif rule.get("source") in {"sd_stage_effect", "sd_diagnostic_toe_recovery"} and intervention in STAGE_EFFECT_INTERVENTIONS:
            rules.append(rule)
    return rules


class HopperRulePriorV6:
    """Evaluate replay-validated substate rules and score MPPI candidates."""

    def __init__(
        self,
        library_path: Path,
        device: torch.device,
        stage_mode: str = "full",
        weight: float = 0.05,
        opposite_penalty: float = 0.02,
        max_bonus: float = 0.10,
        mode: str = "prefer",
        preferred_cap: float = 0.70,
        cap_penalty: float = 0.02,
        allowed_statuses: set[str] | None = None,
        v8_continue_score_min: float = 0.60,
        sd_checkpoint: Path | None = None,
        stochastic_gate: bool = False,
        min_apply_prob: float = 0.40,
        max_apply_prob: float = 0.98,
        confidence_prob_power: float = 1.0,
        stochastic_eval: bool = False,
        stochastic_seed: int = 20260528,
        phase_match_mode: str = "current",
        enable_grounded_hop_phase: bool = False,
    ) -> None:
        self.library_path = Path(library_path)
        self.allowed_statuses = allowed_statuses or {"replay_validated"}
        self.rules = _load_replay_validated_rules(self.library_path, self.allowed_statuses)
        self.device = device
        self.weight = float(weight)
        self.opposite_penalty = float(opposite_penalty)
        self.max_bonus = float(max_bonus)
        self.mode = str(mode)
        if self.mode not in {"prefer", "penalty_only"}:
            raise ValueError(f"Unknown rule prior mode: {self.mode}")
        self.preferred_cap = float(preferred_cap)
        self.cap_penalty = float(cap_penalty)
        self.v8_continue_score_min = float(v8_continue_score_min)
        self.stochastic_gate = bool(stochastic_gate)
        self.min_apply_prob = max(0.0, min(1.0, float(min_apply_prob)))
        self.max_apply_prob = max(self.min_apply_prob, min(1.0, float(max_apply_prob)))
        self.confidence_prob_power = max(1e-6, float(confidence_prob_power))
        self.stochastic_eval = bool(stochastic_eval)
        self.stochastic_seed = int(stochastic_seed)
        # Keep rule-gate sampling separate from TD-MPC2/MPPI sampling so enabling
        # stochastic rule use does not perturb the planner's global RNG stream.
        self._gate_generator = torch.Generator(device="cpu")
        self._gate_generator.manual_seed(self.stochastic_seed)
        self.phase_match_mode = str(phase_match_mode)
        if self.phase_match_mode not in {"current", "hop_alias"}:
            raise ValueError(f"Unknown rule-prior phase match mode: {self.phase_match_mode}")
        self.enable_grounded_hop_phase = bool(enable_grounded_hop_phase)
        self.feature_model, self.feature_model_checkpoint = self._load_feature_model(stage_mode, sd_checkpoint)
        self.feature_model = self.feature_model.to(device)
        self.feature_model.eval()
        self._last_obs_signature: tuple[float, ...] | None = None
        self._last_simple_phase: str | None = None
        self._prev_simple_phase: str | None = None
        self.last_metrics: dict[str, float] = {
            "rule_prior_enabled": 1.0,
            "rule_prior_num_rules": float(len(self.rules)),
            "rule_prior_sd_checkpoint_loaded": 1.0 if self.feature_model_checkpoint else 0.0,
            "rule_prior_hit": 0.0,
            "rule_prior_bonus_mean": 0.0,
            "rule_prior_bonus_max": 0.0,
            "rule_prior_matched_rules": 0.0,
            "rule_prior_applied_rules": 0.0,
            "rule_prior_dropped_rules": 0.0,
            "rule_prior_apply_prob_mean": 0.0,
            "rule_prior_apply_multiplier_mean": 0.0,
            "rule_prior_grounded_hop_phase_enabled": 1.0 if self.enable_grounded_hop_phase else 0.0,
        }

    def _apply_probability(self, confidence: float) -> float:
        c = max(0.0, min(1.0, float(confidence)))
        shaped = c ** self.confidence_prob_power
        return self.min_apply_prob + (self.max_apply_prob - self.min_apply_prob) * shaped

    def _rule_apply_multiplier(self, confidence: float, eval_mode: bool, device: torch.device) -> tuple[float, float, bool]:
        if not self.stochastic_gate:
            return 1.0, 1.0, False
        prob = self._apply_probability(confidence)
        if eval_mode and not self.stochastic_eval:
            return prob, prob, False
        gate_sample = torch.rand((), generator=self._gate_generator)
        keep = bool((gate_sample < prob).item())
        return (1.0 if keep else 0.0), prob, not keep

    @staticmethod
    def _load_feature_model(stage_mode: str, checkpoint: Path | None) -> tuple[DMControlHopperSDModel, str | None]:
        model_kwargs: dict[str, Any] = {
            "action_prior_enabled": False,
            "shaping_scale": 0.0,
            "dense_pose_delta_weight": 0.0,
            "dense_pose_abs_weight": 0.0,
            "dense_pose_milestone_scale": 0.0,
            "recover_penalty_scale": 0.0,
            "stage_mode": stage_mode,
        }
        payload = None
        if checkpoint is not None and checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model_kwargs.update(payload.get("model_config", {}))
            model_kwargs["stage_mode"] = stage_mode
        model = DMControlHopperSDModel(**model_kwargs)
        if payload is None:
            return model, None
        src_state = payload.get("model_state", payload)
        target_state = model.state_dict()
        compatible = {
            k: v
            for k, v in src_state.items()
            if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
        }
        model.load_state_dict(compatible, strict=False)
        return model, str(checkpoint)

    def _substate_parts(self, obs: torch.Tensor) -> dict[str, str]:
        with torch.no_grad():
            state = self.feature_model.encode_observation(obs.to(self.device, dtype=torch.float32))
            feat = self.feature_model._state_reward_features(state)
            pitch_rate = feat["pitch_rate"][0]
            vx = feat["vx"][0]
            full_foot = feat["full_foot_support"][0]
            support_contact = feat["support_contact"][0]
            foot_grounded = feat["foot_grounded"][0]
            touch_toe = feat["touch_toe"][0]
            touch_heel = feat["touch_heel"][0]
            rotation_penalty = feat["rotation_penalty"][0]
            body_collision = feat["body_collision"][0]
            torso_z = feat["torso_z"][0]
            pitch = feat["pitch"][0]
            falling = feat["falling_air"][0] >= 0.5
            rising = feat["rising_air"][0] >= 0.5
            rear = feat["grounded_rear"][0] >= 0.5
            prone = feat["grounded_prone"][0] >= 0.5
            raw_hop = feat.get("grounded_hop", torch.zeros_like(feat["grounded_prone"]))[0] >= 0.5
            hop = raw_hop if self.enable_grounded_hop_phase else torch.zeros_like(raw_hop, dtype=torch.bool)
            upright = (feat["grounded_upright"][0] >= 0.5) | (feat["grounded_prepared"][0] >= 0.5)
            uprightness = torch.clamp(1.0 - torch.abs(pitch) / 0.45, min=0.0, max=1.0)
            height_quality = torch.clamp((torso_z - 0.45) / 0.45, min=0.0, max=1.0)
            forward_speed = torch.clamp(vx / 2.0, min=0.0, max=1.0)
            spin_control = torch.clamp(1.0 - torch.abs(pitch_rate) / 3.50, min=0.0, max=1.0)
            support_readiness = torch.maximum(support_contact, full_foot)
            non_collision = 1.0 - torch.clamp(body_collision, min=0.0, max=1.0)
            continue_score = (
                0.25 * uprightness
                + 0.20 * height_quality
                + 0.20 * forward_speed
                + 0.15 * spin_control
                + 0.10 * support_readiness
                + 0.10 * non_collision
            )

        omega_bin = "omega_pos" if pitch_rate.item() > 1.0 else "omega_neg" if pitch_rate.item() < -1.0 else "omega_mid"
        vx_bin = "vx_high" if vx.item() >= 0.8 else "vx_low" if vx.item() < 0.2 else "vx_mid"
        support_bin = "support_none" if full_foot.item() < 0.1 else "support_low" if full_foot.item() < 0.35 else "support_some"
        coarse_omega_bin = "omega_neg" if omega_bin == "omega_neg" else "omega_nonneg"
        if full_foot.item() >= 0.35:
            broad_support_bin = "support_full"
        elif touch_toe.item() >= 0.5 and touch_heel.item() < 0.2:
            broad_support_bin = "support_partial"
        elif touch_heel.item() >= 0.5 and touch_toe.item() < 0.2:
            broad_support_bin = "support_partial"
        elif max(touch_toe.item(), touch_heel.item()) >= 0.2:
            broad_support_bin = "support_partial"
        else:
            broad_support_bin = "support_none"
        phase = (
            "falling"
            if bool(falling.item())
            else "rear"
            if bool(rear.item())
            else "hop"
            if bool(hop.item())
            else "prone"
            if bool(prone.item())
            else "upright"
            if bool(upright.item())
            else "other"
        )
        simple_phase = (
            "air_falling"
            if bool(falling.item())
            else "air_rising"
            if bool(rising.item())
            else "ground_rear"
            if bool(rear.item())
            else "ground_front"
            if bool(prone.item())
            else "ground_upright"
            if bool(upright.item())
            else "other"
        )
        structural_success = (
            torso_z.item() >= 0.75
            and abs(pitch.item()) <= 0.45
            and vx.item() >= 0.80
            and abs(pitch_rate.item()) <= 3.50
            and body_collision.item() < 0.50
        )
        near_good = structural_success or continue_score.item() >= self.v8_continue_score_min
        if near_good:
            if full_foot.item() >= 0.35:
                refined_phase = "near_good_support"
            elif support_contact.item() >= 0.10 or foot_grounded.item() >= 0.10:
                refined_phase = "near_good_contact"
            else:
                refined_phase = "near_good_air"
        else:
            refined_phase = phase
        if body_collision.item() >= 0.5:
            risk_bin = "collision_risk"
        elif rotation_penalty.item() >= 0.5 or abs(pitch_rate.item()) >= 3.5:
            risk_bin = "rotation_risk"
        elif phase in {"falling", "rising"} and full_foot.item() < 0.1:
            risk_bin = "support_risk"
        else:
            risk_bin = "stable"
        toe_only_launch_trap = (
            touch_toe.item() >= 0.80
            and touch_heel.item() < 0.20
            and support_contact.item() >= 0.80
            and full_foot.item() < 0.35
            and vx.item() >= 0.80
            and torso_z.item() >= 0.55
            and not bool(upright.item())
        )
        toe_recovery_low_support = (
            touch_toe.item() >= 0.80
            and touch_heel.item() < 0.50
            and full_foot.item() < 0.35
            and not bool(upright.item())
            and (
                bool(prone.item())
                or torso_z.item() <= 0.72
                or rotation_penalty.item() >= 0.50
                or abs(pitch_rate.item()) >= 3.0
            )
        )
        toe_recovery_recoverable_low_support = toe_recovery_low_support and (
            bool(prone.item()) or bool(rear.item())
        )
        toe_recovery_unstable_low_support = toe_recovery_low_support and not toe_recovery_recoverable_low_support
        return {
            "omega_bin": omega_bin,
            "coarse_omega_bin": coarse_omega_bin,
            "vx_bin": vx_bin,
            "support_bin": support_bin,
            "broad_support_bin": broad_support_bin,
            "phase": phase,
            "simple_phase": simple_phase,
            "refined_phase": refined_phase,
            "risk_bin": risk_bin,
            "raw_grounded_hop": "1" if bool(raw_hop.item()) else "0",
            "grounded_hop_phase_enabled": "1" if self.enable_grounded_hop_phase else "0",
            "rotation_risk": "1" if (rotation_penalty.item() >= 0.5 or abs(pitch_rate.item()) >= 3.0) else "0",
            "toe_only_launch_trap": "1" if toe_only_launch_trap else "0",
            "toe_recovery_low_support": "1" if toe_recovery_low_support else "0",
            "toe_recovery_recoverable_low_support": "1" if toe_recovery_recoverable_low_support else "0",
            "toe_recovery_unstable_low_support": "1" if toe_recovery_unstable_low_support else "0",
        }

    @staticmethod
    def _obs_signature(obs: torch.Tensor) -> tuple[float, ...]:
        flat = obs.detach().float().view(-1).cpu()
        return tuple(round(float(x), 5) for x in flat.tolist())

    def _attach_temporal_parts(self, obs: torch.Tensor, parts: dict[str, str]) -> dict[str, str]:
        out = dict(parts)
        sig = self._obs_signature(obs)
        current = str(out.get("simple_phase", "unknown"))
        if sig != self._last_obs_signature:
            self._prev_simple_phase = self._last_simple_phase
            self._last_simple_phase = current
            self._last_obs_signature = sig
        out["prev_simple_phase"] = self._prev_simple_phase or "unknown"
        out["current_simple_phase"] = current
        return out

    def _state_matches(self, rule: dict[str, Any], parts: dict[str, str]) -> bool:
        key = str(rule.get("substate_key", ""))
        if not key:
            key = str(rule.get("rule_key", ""))
        if key.startswith("prev="):
            items: dict[str, str] = {}
            for token in key.split("|"):
                if "=" not in token:
                    return False
                name, value = token.split("=", 1)
                items[name] = value
            return (
                items.get("prev") == parts.get("prev_simple_phase")
                and items.get("cur") == parts.get("current_simple_phase")
                and items.get("omega") == parts.get("coarse_omega_bin")
                and items.get("vx") == parts.get("vx_bin")
                and items.get("support") == parts.get("broad_support_bin")
            )
        tokens = key.split("|")
        if len(tokens) == 5:
            event, omega_bin, vx_bin, support_bin, phase = tokens
            event_supported = event == "unsupported_high_torque" or (
                event == "rotation_risk" and parts.get("rotation_risk") == "1"
            )
            phase_matches = phase == parts["phase"]
            if self.phase_match_mode == "hop_alias" and phase == "hop":
                phase_matches = parts["phase"] in {"hop", "prone", "other"}
            return (
                event_supported
                and omega_bin == parts["omega_bin"]
                and vx_bin == parts["vx_bin"]
                and support_bin == parts["support_bin"]
                and phase_matches
            )
        if len(tokens) == 7:
            role, _quality_bin, omega_bin, vx_bin, support_bin, phase, risk_bin = tokens
            phase_matches = phase in {"phase_any", "*", "any"} or phase == parts["refined_phase"]
            return (
                role == "transition"
                and omega_bin == parts["omega_bin"]
                and vx_bin == parts["vx_bin"]
                and support_bin == parts["support_bin"]
                and phase_matches
                and risk_bin == parts["risk_bin"]
            )
        if key == "toe_recovery_low_support":
            return parts.get("toe_recovery_low_support") == "1"
        if key == "toe_recovery_recoverable_low_support":
            return parts.get("toe_recovery_recoverable_low_support") == "1"
        if key == "toe_recovery_unstable_low_support":
            return parts.get("toe_recovery_unstable_low_support") == "1"
        if key == "toe_only_launch_trap":
            return parts.get("toe_only_launch_trap") == "1"
        return False

    @staticmethod
    def _candidate_gate(rule: dict[str, Any], first_actions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        event = str(rule.get("state_event") or str(rule.get("substate_key", "")).split("|", 1)[0])
        key = str(rule.get("substate_key") or rule.get("rule_key") or "")
        if key.startswith("toe_recovery_") or key == "toe_only_launch_trap":
            return torch.ones(first_actions.shape[0], dtype=dtype, device=first_actions.device)
        if key.startswith("prev=") or rule.get("source") == "sd_tva_broad_context_counterfactual":
            return torch.ones(first_actions.shape[0], dtype=dtype, device=first_actions.device)
        if rule.get("rule_role") == "transition" or key.startswith("transition|"):
            return torch.ones(first_actions.shape[0], dtype=dtype, device=first_actions.device)
        high_torque_gate = (first_actions.abs().max(dim=-1).values >= 0.65).to(dtype=dtype)
        if event == "unsupported_high_torque":
            return high_torque_gate
        if event == "rotation_risk":
            return torch.ones_like(high_torque_gate)
        return high_torque_gate

    @staticmethod
    def _action_family(rule: dict[str, Any]) -> tuple[int | None, int]:
        family = str(rule.get("action_family", ""))
        if "_" not in family:
            return None, 0
        joint, sign = family.split("_", 1)
        if joint not in JOINT_TO_DIM:
            return None, 0
        return JOINT_TO_DIM[joint], 1 if sign == "pos" else -1

    @staticmethod
    def _default_stage_objective(rule: dict[str, Any]) -> dict[str, dict[str, float]]:
        goal = str(rule.get("stage_goal", ""))
        if goal == "recover_full_foot_support":
            return {
                "terminal_weights": {
                    "full_foot_support": 0.40,
                    "support_contact": 0.25,
                    "torso_z_quality": 0.15,
                    "rotation_penalty": -0.15,
                    "body_collision": -0.15,
                },
                "progress_weights": {
                    "full_foot_support": 0.45,
                    "support_contact": 0.25,
                    "torso_z_quality": 0.10,
                    "rotation_penalty": -0.15,
                    "body_collision": -0.10,
                },
                "mean_weights": {
                    "full_foot_support": 0.10,
                    "support_contact": 0.10,
                    "rotation_penalty": -0.10,
                    "body_collision": -0.10,
                },
            }
        if goal == "stabilize_toe_only_launch":
            return {
                "terminal_weights": {
                    "support_contact": 0.35,
                    "spin_control": 0.25,
                    "torso_z_quality": 0.20,
                    "pitch_control": 0.15,
                    "touch_heel": 0.10,
                    "full_foot_support": 0.10,
                    "rotation_penalty": -0.25,
                    "body_collision": -0.25,
                    "falling_air": -0.20,
                    "airborne_true": -0.10,
                },
                "progress_weights": {
                    "support_contact": 0.35,
                    "spin_control": 0.20,
                    "torso_z_quality": 0.10,
                    "pitch_control": 0.10,
                    "touch_heel": 0.10,
                    "full_foot_support": 0.05,
                    "rotation_penalty": -0.25,
                    "body_collision": -0.20,
                    "falling_air": -0.20,
                },
                "mean_weights": {
                    "support_contact": 0.15,
                    "spin_control": 0.10,
                    "rotation_penalty": -0.10,
                    "body_collision": -0.10,
                },
            }
        return {
            "terminal_weights": {},
            "progress_weights": {},
            "mean_weights": {},
        }

    @staticmethod
    def _metric_value(feat: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name == "torso_z_quality":
            return torch.clamp((feat["torso_z"] - 0.45) / 0.40, min=0.0, max=1.0)
        if name == "pitch_control":
            return torch.clamp(1.0 - torch.abs(feat["pitch"]) / 0.75, min=0.0, max=1.0)
        if name == "spin_control":
            return torch.clamp(1.0 - torch.abs(feat["pitch_rate"]) / 3.50, min=0.0, max=1.0)
        if name == "forward_speed":
            return torch.clamp(feat["vx"] / 2.0, min=0.0, max=1.0)
        if name not in feat:
            raise KeyError(f"Unknown stage-effect metric: {name}")
        return torch.clamp(feat[name], min=0.0, max=1.0)

    def _constraint_metric_value(
        self,
        init_feat: dict[str, torch.Tensor],
        rollout_feats: list[dict[str, torch.Tensor]],
        metric: str,
        scope: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        values = [self._metric_value(feat, metric).to(device=device, dtype=dtype) for feat in rollout_feats]
        if scope == "initial":
            return self._metric_value(init_feat, metric).to(device=device, dtype=dtype)
        if scope == "terminal":
            return values[-1]
        if scope == "progress":
            initial = self._metric_value(init_feat, metric).to(device=device, dtype=dtype)
            return values[-1] - initial
        stacked = torch.stack(values, dim=0)
        if scope == "mean":
            return stacked.mean(dim=0)
        if scope == "min":
            return stacked.min(dim=0).values
        if scope == "max":
            return stacked.max(dim=0).values
        raise ValueError(f"Unknown stage constraint scope: {scope}")

    def _stage_constraint_penalty(
        self,
        objective: dict[str, Any],
        init_feat: dict[str, torch.Tensor],
        rollout_feats: list[dict[str, torch.Tensor]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        penalty = torch.zeros_like(self._metric_value(rollout_feats[-1], "support_contact").to(device=device, dtype=dtype))
        for constraint in objective.get("stage_constraints", []) or []:
            metric = str(constraint.get("metric", ""))
            if not metric:
                continue
            scope = str(constraint.get("scope", "terminal"))
            op = str(constraint.get("op", "min"))
            threshold = float(constraint.get("threshold", 0.0))
            weight = float(constraint.get("penalty", 0.0))
            if weight <= 0.0:
                continue
            value = self._constraint_metric_value(init_feat, rollout_feats, metric, scope, dtype, device)
            if op == "min":
                violation = torch.relu(torch.as_tensor(threshold, dtype=dtype, device=device) - value)
            elif op == "max":
                violation = torch.relu(value - torch.as_tensor(threshold, dtype=dtype, device=device))
            else:
                raise ValueError(f"Unknown stage constraint op: {op}")
            penalty = penalty + weight * violation
        return penalty

    @staticmethod
    def _struct_state(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "_split_latent"):
            z_struct, _ = model._split_latent(z)
            return z_struct
        return z

    @staticmethod
    def _dynamics_core(model: torch.nn.Module) -> torch.nn.Module:
        return getattr(model, "_dynamics", model)

    def _rollout_stage_features(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        model: torch.nn.Module | None,
        z: torch.Tensor | None,
        task: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        num_samples = actions.shape[1]
        if model is not None and z is not None:
            state = z
            core = self._dynamics_core(model)
            if hasattr(core, "_state_reward_features"):
                init_feat = core._state_reward_features(self._struct_state(model, state))
                feats: list[dict[str, torch.Tensor]] = []
                for t in range(horizon):
                    state = model.next(state, actions[t], task)
                    feats.append(core._state_reward_features(self._struct_state(model, state)))
                return init_feat, feats

        state = self.feature_model.encode_observation(obs.to(self.device, dtype=torch.float32)).repeat(num_samples, 1)
        init_feat = self.feature_model._state_reward_features(state)
        feats = []
        for t in range(horizon):
            state = self.feature_model(state, actions[t].to(self.device, dtype=torch.float32))
            feats.append(self.feature_model._state_reward_features(state))
        return init_feat, feats

    def _stage_effect_score(
        self,
        rule: dict[str, Any],
        obs: torch.Tensor,
        actions: torch.Tensor,
        model: torch.nn.Module | None,
        z: torch.Tensor | None,
        task: torch.Tensor | None,
    ) -> torch.Tensor:
        horizon = max(1, min(int(rule.get("horizon", 5) or 5), int(actions.shape[0])))
        objective = rule.get("stage_objective") or rule.get("objective") or self._default_stage_objective(rule)
        init_feat, rollout_feats = self._rollout_stage_features(obs, actions, horizon, model, z, task)
        terminal_feat = rollout_feats[-1]
        score = torch.zeros(actions.shape[1], dtype=actions.dtype, device=actions.device)

        for name, weight in dict(objective.get("terminal_weights", {})).items():
            score = score + float(weight) * self._metric_value(terminal_feat, name).to(score.device, score.dtype)
        for name, weight in dict(objective.get("progress_weights", {})).items():
            terminal = self._metric_value(terminal_feat, name).to(score.device, score.dtype)
            initial = self._metric_value(init_feat, name).to(score.device, score.dtype)
            score = score + float(weight) * (terminal - initial)
        if objective.get("mean_weights"):
            for name, weight in dict(objective.get("mean_weights", {})).items():
                values = [self._metric_value(feat, name).to(score.device, score.dtype) for feat in rollout_feats]
                score = score + float(weight) * torch.stack(values, dim=0).mean(dim=0)

        if bool(rule.get("center_stage_score", True)):
            score = score - score.mean()
        if bool(rule.get("normalize_stage_score", True)):
            score = score / (score.abs().mean().clamp_min(1e-6))
        stage_weight = float(rule.get("stage_prior_weight", self.weight) or self.weight)
        stage_max_bonus = float(rule.get("stage_max_bonus", self.max_bonus) or self.max_bonus)
        stage_score = stage_weight * score
        constraint_penalty = self._stage_constraint_penalty(
            objective,
            init_feat,
            rollout_feats,
            actions.dtype,
            actions.device,
        )
        priority_weight = float(objective.get("constraint_priority_weight", rule.get("constraint_priority_weight", 0.0)) or 0.0)
        if priority_weight > 0.0:
            penalty_span = constraint_penalty.max() - constraint_penalty.min()
            if torch.isfinite(penalty_span) and penalty_span.item() > 1e-6:
                relative_penalty = (constraint_penalty - constraint_penalty.min()) / penalty_span.clamp_min(1e-6)
                stage_score = stage_score - priority_weight * relative_penalty.to(stage_score.device, stage_score.dtype)
            satisfied_bonus = float(
                objective.get("constraint_satisfied_bonus", rule.get("constraint_satisfied_bonus", 0.0)) or 0.0
            )
            if satisfied_bonus > 0.0:
                stage_score = stage_score + satisfied_bonus * (constraint_penalty <= 1e-6).to(stage_score.dtype)
        absolute_weight = float(objective.get("constraint_absolute_weight", rule.get("constraint_absolute_weight", 1.0)) or 0.0)
        stage_score = stage_score - absolute_weight * constraint_penalty
        return stage_score.clamp(min=-stage_max_bonus, max=stage_max_bonus)

    def score(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        model: torch.nn.Module | None = None,
        z: torch.Tensor | None = None,
        task: torch.Tensor | None = None,
        eval_mode: bool = False,
    ) -> torch.Tensor:
        """Return additive candidate values for sampled action sequences.

        The rule is applied only as soft candidate reweighting. It does not
        replace the selected action and does not alter the learned model.
        """
        bonus = torch.zeros(actions.shape[1], dtype=actions.dtype, device=actions.device)
        if not self.rules:
            self.last_metrics.update(
                rule_prior_hit=0.0,
                rule_prior_bonus_mean=0.0,
                rule_prior_bonus_max=0.0,
                rule_prior_matched_rules=0.0,
                rule_prior_applied_rules=0.0,
                rule_prior_dropped_rules=0.0,
                rule_prior_apply_prob_mean=0.0,
                rule_prior_apply_multiplier_mean=0.0,
                rule_prior_grounded_hop_phase_enabled=1.0 if self.enable_grounded_hop_phase else 0.0,
            )
            return bonus

        parts = self._attach_temporal_parts(obs, self._substate_parts(obs))
        hits = 0
        matched_rules = 0
        applied_rules = 0
        dropped_rules = 0
        apply_prob_sum = 0.0
        apply_multiplier_sum = 0.0
        first_actions = actions[0]
        for rule in self.rules:
            if not self._state_matches(rule, parts):
                continue
            matched_rules += 1
            rule_scale = float(rule.get("effective_prior_scale", rule.get("confidence", 1.0)) or 0.0)
            rule_scale = max(0.0, min(1.0, rule_scale))
            apply_multiplier, apply_prob, dropped = self._rule_apply_multiplier(rule_scale, eval_mode, actions.device)
            apply_prob_sum += apply_prob
            apply_multiplier_sum += apply_multiplier
            if dropped:
                dropped_rules += 1
                continue
            applied_rules += 1
            if rule.get("intervention_type") in STAGE_EFFECT_INTERVENTIONS:
                score = self._stage_effect_score(rule, obs, actions, model, z, task)
                bonus = bonus + apply_multiplier * rule_scale * score.to(dtype=actions.dtype, device=actions.device)
                hits += 1
                continue
            dim, sign = self._action_family(rule)
            if dim is None:
                continue
            signed = first_actions[:, dim] * float(sign)
            prefer = torch.relu(signed).clamp(max=1.0)
            oppose = torch.relu(-signed).clamp(max=1.0)
            if rule.get("intervention_type") == "avoid_action_family":
                score = -self.opposite_penalty * prefer
            elif self.mode == "prefer":
                score = self.weight * prefer - self.opposite_penalty * oppose
                if self.cap_penalty > 0:
                    score = score - self.cap_penalty * torch.relu(prefer - self.preferred_cap)
            else:
                score = -self.opposite_penalty * oppose
                if self.cap_penalty > 0:
                    score = score - self.cap_penalty * torch.relu(prefer - self.preferred_cap)
            candidate_gate = self._candidate_gate(rule, first_actions, actions.dtype)
            bonus = bonus + candidate_gate * apply_multiplier * rule_scale * score
            hits += 1

        bonus = bonus.clamp(min=-self.max_bonus, max=self.max_bonus)
        self.last_metrics.update(
            rule_prior_hit=1.0 if hits else 0.0,
            rule_prior_bonus_mean=float(bonus.mean().detach().cpu().item()),
            rule_prior_bonus_max=float(bonus.max().detach().cpu().item()) if bonus.numel() else 0.0,
            rule_prior_matched_rules=float(matched_rules),
            rule_prior_applied_rules=float(applied_rules),
            rule_prior_dropped_rules=float(dropped_rules),
            rule_prior_apply_prob_mean=float(apply_prob_sum / matched_rules) if matched_rules else 0.0,
            rule_prior_apply_multiplier_mean=float(apply_multiplier_sum / matched_rules) if matched_rules else 0.0,
            rule_prior_grounded_hop_phase_enabled=1.0 if self.enable_grounded_hop_phase else 0.0,
        )
        return bonus


class HopperTVATerminalPrior:
    """Low-weight TVA terminal compensation for TD-MPC2 MPPI candidates."""

    def __init__(
        self,
        sd_checkpoint: Path,
        tva_checkpoint: Path,
        device: torch.device,
        stage_mode: str = "full",
        weight: float = 0.02,
        max_bonus: float = 0.05,
        normalize: bool = True,
    ) -> None:
        self.device = device
        self.sd_checkpoint = Path(sd_checkpoint)
        self.tva_checkpoint = Path(tva_checkpoint)
        self.weight = float(weight)
        self.max_bonus = float(max_bonus)
        self.normalize = bool(normalize)
        self.feature_model, self.feature_model_checkpoint = HopperRulePriorV6._load_feature_model(stage_mode, self.sd_checkpoint)
        self.feature_model = self.feature_model.to(device).eval()
        self.value_net = self._load_value_net(self.tva_checkpoint).to(device).eval()
        self.last_metrics: dict[str, float] = {
            "tva_terminal_prior_enabled": 1.0,
            "tva_terminal_prior_value_mean": 0.0,
            "tva_terminal_prior_value_std": 0.0,
            "tva_terminal_prior_bonus_mean": 0.0,
            "tva_terminal_prior_bonus_max": 0.0,
        }

    def _load_value_net(self, checkpoint: Path) -> HopperTVAValueNet:
        if not checkpoint.exists():
            raise FileNotFoundError(f"TVA terminal checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        value_net = HopperTVAValueNet(
            state_dim=int(payload.get("state_dim", 17)),
            hidden=int(payload.get("hidden", 128)),
        )
        value_net.load_state_dict(payload.get("value_net_state", payload), strict=False)
        return value_net

    def score(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        num_samples = actions.shape[1]
        state = self.feature_model.encode_observation(obs.to(self.device, dtype=torch.float32)).repeat(num_samples, 1)
        with torch.no_grad():
            for t in range(actions.shape[0]):
                state = self.feature_model(state, actions[t].to(self.device, dtype=torch.float32))
            values = self.value_net(state).to(dtype=actions.dtype, device=actions.device)
        raw_values = values
        if self.normalize:
            centered = values - values.mean()
            values = centered / centered.abs().mean().clamp_min(1e-6)
        bonus = (self.weight * values).clamp(min=-self.max_bonus, max=self.max_bonus)
        self.last_metrics.update(
            tva_terminal_prior_value_mean=float(raw_values.mean().detach().cpu().item()),
            tva_terminal_prior_value_std=float(raw_values.std(unbiased=False).detach().cpu().item()),
            tva_terminal_prior_bonus_mean=float(bonus.mean().detach().cpu().item()),
            tva_terminal_prior_bonus_max=float(bonus.max().detach().cpu().item()) if bonus.numel() else 0.0,
        )
        return bonus


def _resolve_path(raw: str, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    try:
        import hydra

        return Path(hydra.utils.get_original_cwd()) / path
    except Exception:
        return Path.cwd() / path


def install_rule_prior(agent: torch.nn.Module, cfg: Any) -> HopperRulePriorV6 | None:
    """Install the v6 planner patch on a TD-MPC2 agent when explicitly enabled."""
    rule_enabled = bool(getattr(cfg, "rule_prior_enabled", False))
    tva_enabled = bool(getattr(cfg, "tva_terminal_prior_enabled", False))
    if not rule_enabled and not tva_enabled:
        return None
    library_path = _resolve_path(
        str(getattr(cfg, "rule_prior_library", "results/sd_retrospective_rule_induction/rules/library_v5.json")),
        "rule_prior_library",
    )
    sd_checkpoint_raw = str(getattr(cfg, "rule_prior_sd_checkpoint", "") or "").strip()
    sd_checkpoint = _resolve_path(sd_checkpoint_raw, "rule_prior_sd_checkpoint") if sd_checkpoint_raw else None
    prior = None
    if rule_enabled:
        prior = HopperRulePriorV6(
            library_path=library_path,
            device=agent.device,
            stage_mode=str(getattr(cfg, "sd_stage_mode", "full")),
            weight=float(getattr(cfg, "rule_prior_weight", 0.05)),
            opposite_penalty=float(getattr(cfg, "rule_prior_opposite_penalty", 0.02)),
            max_bonus=float(getattr(cfg, "rule_prior_max_bonus", 0.10)),
            mode=str(getattr(cfg, "rule_prior_mode", "prefer")),
            preferred_cap=float(getattr(cfg, "rule_prior_preferred_cap", 0.70)),
            cap_penalty=float(getattr(cfg, "rule_prior_cap_penalty", 0.02)),
            v8_continue_score_min=float(getattr(cfg, "rule_prior_v8_continue_score_min", 0.60)),
            sd_checkpoint=sd_checkpoint,
            stochastic_gate=_as_bool(getattr(cfg, "rule_prior_stochastic_gate", False)),
            min_apply_prob=float(getattr(cfg, "rule_prior_min_apply_prob", 0.40)),
            max_apply_prob=float(getattr(cfg, "rule_prior_max_apply_prob", 0.98)),
            confidence_prob_power=float(getattr(cfg, "rule_prior_confidence_prob_power", 1.0)),
            stochastic_eval=_as_bool(getattr(cfg, "rule_prior_stochastic_eval", False)),
            stochastic_seed=int(getattr(cfg, "rule_prior_stochastic_seed", 20260528)),
            phase_match_mode=str(getattr(cfg, "rule_prior_phase_match_mode", "current")),
            enable_grounded_hop_phase=_as_bool(getattr(cfg, "rule_prior_enable_grounded_hop_phase", False)),
            allowed_statuses={
                s.strip()
                for s in str(getattr(cfg, "rule_prior_allowed_statuses", "replay_validated")).split(",")
                if s.strip()
            },
        )
    tva_prior = None
    if tva_enabled:
        tva_sd_raw = str(getattr(cfg, "tva_terminal_prior_sd_checkpoint", "") or "").strip()
        tva_ckpt_raw = str(getattr(cfg, "tva_terminal_prior_checkpoint", "") or "").strip()
        if not tva_sd_raw:
            raise ValueError("tva_terminal_prior_enabled requires tva_terminal_prior_sd_checkpoint")
        if not tva_ckpt_raw:
            raise ValueError("tva_terminal_prior_enabled requires tva_terminal_prior_checkpoint")
        tva_prior = HopperTVATerminalPrior(
            sd_checkpoint=_resolve_path(tva_sd_raw, "tva_terminal_prior_sd_checkpoint"),
            tva_checkpoint=_resolve_path(tva_ckpt_raw, "tva_terminal_prior_checkpoint"),
            device=agent.device,
            stage_mode=str(getattr(cfg, "sd_stage_mode", "full")),
            weight=float(getattr(cfg, "tva_terminal_prior_weight", 0.02)),
            max_bonus=float(getattr(cfg, "tva_terminal_prior_max_bonus", 0.05)),
            normalize=bool(getattr(cfg, "tva_terminal_prior_normalize", True)),
        )
    agent._v6_rule_prior = prior
    agent._tva_terminal_prior = tva_prior
    original_get_plan_metrics = agent.get_plan_metrics

    def get_plan_metrics_with_rule_prior(self):
        metrics = original_get_plan_metrics()
        if self._v6_rule_prior is not None:
            metrics.update(self._v6_rule_prior.last_metrics)
        if self._tva_terminal_prior is not None:
            metrics.update(self._tva_terminal_prior.last_metrics)
        return metrics

    def _plan_with_rule_prior(self, obs, t0=False, eval_mode=False, task=None):
        z = self.model.encode(obs, task)
        prior_low, prior_high, prior_stats = None, None, {}
        if hasattr(self.model, "action_prior_bounds"):
            prior_low, prior_high, prior_stats = self.model.action_prior_bounds(z, task)
            if prior_low.ndim > 1:
                prior_low = prior_low[0]
                prior_high = prior_high[0]
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                if prior_low is not None:
                    pi_actions[t] = torch.max(torch.min(pi_actions[t], prior_high), prior_low)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)
            if prior_low is not None:
                pi_actions[-1] = torch.max(torch.min(pi_actions[-1], prior_high), prior_low)

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, : self.cfg.num_pi_trajs] = pi_actions

        rule_bonus = torch.zeros(self.cfg.num_samples, dtype=torch.float32, device=self.device)
        tva_bonus = torch.zeros(self.cfg.num_samples, dtype=torch.float32, device=self.device)
        nonfinite_frac = torch.zeros((), dtype=torch.float32, device=self.device)
        for _ in range(self.cfg.iterations):
            r = torch.randn(
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            if prior_low is not None:
                actions_sample = torch.max(
                    torch.min(actions_sample, prior_high.view(1, 1, -1)),
                    prior_low.view(1, 1, -1),
                )
            actions[:, self.cfg.num_pi_trajs :] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            raw_value = self._estimate_value(z, actions, task)
            if self._v6_rule_prior is not None:
                rule_bonus = self._v6_rule_prior.score(
                    obs,
                    actions,
                    model=self.model,
                    z=z,
                    task=task,
                    eval_mode=eval_mode,
                ).to(raw_value.device)
            else:
                rule_bonus = torch.zeros(self.cfg.num_samples, dtype=torch.float32, device=self.device)
            if self._tva_terminal_prior is not None:
                tva_bonus = self._tva_terminal_prior.score(obs, actions).to(raw_value.device)
            else:
                tva_bonus = torch.zeros(self.cfg.num_samples, dtype=torch.float32, device=self.device)
            nonfinite_frac = (~torch.isfinite(raw_value)).float().mean()
            value = raw_value.nan_to_num(0) + rule_bonus.view(-1, 1) + tva_bonus.view(-1, 1)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        rand_idx = _gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        if prior_low is not None:
            a = torch.max(torch.min(a, prior_high), prior_low)
        self._prev_mean.copy_(mean)
        self._last_plan_metrics = {
            "plan_value_mean": float(value.mean().item()),
            "plan_value_std": float(value.std(unbiased=False).item()),
            "plan_elite_value_mean": float(elite_value.mean().item()),
            "plan_elite_value_std": float(elite_value.std(unbiased=False).item()),
            "plan_std_mean": float(std.mean().item()),
            "plan_nonfinite_frac": float(nonfinite_frac.item()),
            "plan_prior_rear_frac": float(prior_stats.get("rear_prior_frac", 0.0)),
            "plan_prior_prone_frac": float(prior_stats.get("prone_prior_frac", 0.0)),
            "rule_prior_value_mean": float(rule_bonus.mean().item()),
            "rule_prior_value_max": float(rule_bonus.max().item()) if rule_bonus.numel() else 0.0,
            "tva_terminal_prior_value_mean": float(tva_bonus.mean().item()),
            "tva_terminal_prior_value_max": float(tva_bonus.max().item()) if tva_bonus.numel() else 0.0,
        }
        return a.clamp(-1, 1)

    agent.get_plan_metrics = types.MethodType(get_plan_metrics_with_rule_prior, agent)
    agent._plan = types.MethodType(_plan_with_rule_prior, agent)
    agent._plan_val = None
    print(
        "Installed v6 planner priors: "
        f"rules={len(prior.rules) if prior is not None else 0} "
        f"rule_enabled={rule_enabled} "
        f"tva_terminal_enabled={tva_enabled} "
        f"rule_library={library_path} "
        f"rule_sd_checkpoint={(prior.feature_model_checkpoint if prior is not None else None) or 'default'} "
        f"tva_checkpoint={(tva_prior.tva_checkpoint if tva_prior is not None else None) or 'none'}"
    )
    return prior
