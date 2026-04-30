from copy import deepcopy
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


class SDWorldModel(nn.Module):
	"""
	TD-MPC2-compatible wrapper that keeps the TD-MPC2 solver stack
	while replacing the latent transition / reward core with an SD model.

	Current implementation targets Hopper-Hop and uses a hybrid latent:
	structured state for explicit mechanics plus a learned residual subspace
	for value/planning-relevant factors that are hard to encode manually.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		assert cfg.world_model_kind == 'sd_hopper'
		assert cfg.obs == 'state', 'SD world model currently supports state observations only.'
		assert not cfg.multitask, 'SD world model is currently implemented for single-task use.'
		assert cfg.task == 'hopper-hop', 'sd_hopper adapter currently supports hopper-hop only.'
		assert cfg.action_dim == DMControlHopperSDModel.ACTION_DIM

		self.struct_dim = int(getattr(cfg, 'sd_struct_dim', DMControlHopperSDModel.STATE_DIM))
		self.latent_dim = int(getattr(cfg, 'sd_latent_dim', max(self.struct_dim, int(getattr(cfg, 'latent_dim', 512)))))
		assert self.struct_dim == DMControlHopperSDModel.STATE_DIM, f'sd_hopper currently expects {DMControlHopperSDModel.STATE_DIM} structured dims.'
		assert self.latent_dim >= self.struct_dim, 'sd_latent_dim must be >= sd_struct_dim.'
		self.resid_dim = self.latent_dim - self.struct_dim
		self.reward_mode = getattr(cfg, 'sd_reward_mode', 'learned')
		self.reward_env_scale = float(getattr(cfg, 'sd_env_reward_scale', 1.0))
		self.reward_hybrid_alpha = float(getattr(cfg, 'sd_reward_hybrid_alpha', 0.1))
		self.reward_curriculum_start = int(getattr(cfg, 'sd_reward_curriculum_start', 2500))
		self.reward_curriculum_end = int(getattr(cfg, 'sd_reward_curriculum_end', 4000))
		self.reward_curriculum_alpha = float(getattr(cfg, 'sd_reward_curriculum_alpha', 0.5))
		self.action_prior_enabled = bool(getattr(cfg, 'sd_action_prior_enabled', False))
		self.register_buffer('_reward_curriculum_step', torch.tensor(0, dtype=torch.long))

		self._encoder = nn.Identity()
		self._dynamics = DMControlHopperSDModel(
			dt=getattr(cfg, 'sd_dt', 0.02),
			hidden=getattr(cfg, 'sd_hidden_dim', 256),
			hop_speed=getattr(cfg, 'sd_hop_speed', 2.0),
			action_prior_enabled=self.action_prior_enabled,
			shaping_scale=float(getattr(cfg, 'sd_shaping_scale', 1.0)),
			dense_pose_delta_weight=float(getattr(cfg, 'sd_dense_pose_delta_weight', 0.40)),
			dense_pose_abs_weight=float(getattr(cfg, 'sd_dense_pose_abs_weight', 0.0)),
			dense_pose_milestone_scale=float(getattr(cfg, 'sd_dense_pose_milestone_scale', 0.25)),
			dense_pose_gamma=float(getattr(cfg, 'sd_dense_pose_gamma', getattr(cfg, 'discount_max', 0.99))),
			recover_penalty_scale=float(getattr(cfg, 'sd_recover_penalty_scale', 0.05)),
			stage_mode=str(getattr(cfg, 'sd_stage_mode', 'full')),
			use_transition_residual=bool(getattr(cfg, 'sd_use_transition_residual', False)),
			transition_residual_scale=float(getattr(cfg, 'sd_transition_residual_scale', 1.0)),
		)
		if self.resid_dim > 0:
			self._residual_encoder = layers.mlp(cfg.obs_shape['state'][0], 2*[cfg.mlp_dim], self.resid_dim)
			self._residual_dynamics = layers.mlp(self.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], self.resid_dim)
			self._struct_residual = layers.mlp(self.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], self.struct_dim)
			self.max_resid_scale = float(getattr(cfg, 'sd_max_resid_scale', 5.0))
		else:
			self._residual_encoder = None
			self._residual_dynamics = None
			self._struct_residual = None
		self._reward = layers.mlp(self.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(self.latent_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(self.latent_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([
			layers.mlp(self.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout)
			for _ in range(cfg.num_q)
		])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight] + [q[-1].weight for q in self._Qs.modules_list])

		self.register_buffer('log_std_min', torch.tensor(cfg.log_std_min))
		self.register_buffer('log_std_dif', torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.struct_residual_scale = float(getattr(cfg, 'sd_struct_residual_scale', 0.10))
		self.vx_residual_scale = float(getattr(cfg, 'sd_vx_residual_scale', 0.50))
		self.vz_residual_scale = float(getattr(cfg, 'sd_vz_residual_scale', 3.00))
		self.pitch_residual_scale = float(getattr(cfg, 'sd_pitch_residual_scale', 0.75))
		self.pitch_rate_residual_scale = float(getattr(cfg, 'sd_pitch_rate_residual_scale', 8.00))
		self.knee_residual_scale = float(getattr(cfg, 'sd_knee_residual_scale', 0.35))
		self.trend_protect_max_scale = float(getattr(cfg, 'sd_trend_protect_max_scale', 0.20))
		self.trend_protect_min_correction = float(getattr(cfg, 'sd_trend_protect_min_correction', 0.0))
		self._trend_protected_idx = (
			DMControlHopperSDModel.ROOTZ_IDX,
			DMControlHopperSDModel.TOE_TOUCH_IDX,
			DMControlHopperSDModel.HEEL_TOUCH_IDX,
			DMControlHopperSDModel.ZMIN_IDX,
		)
		self._coupling_bridge_idx = (
			DMControlHopperSDModel.VX_IDX,
			DMControlHopperSDModel.VZ_IDX,
			DMControlHopperSDModel.PITCH_IDX,
			DMControlHopperSDModel.PITCH_RATE_IDX,
		)
		self.init()

	def init(self):
		self._detach_Qs = deepcopy(self._Qs)
		self._target_Qs = deepcopy(self._Qs)
		for module in (self._detach_Qs, self._target_Qs):
			for p in module.parameters():
				p.requires_grad_(False)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		for target_p, detach_p in zip(self._target_Qs.parameters(), self._detach_Qs.parameters()):
			target_p.data.lerp_(detach_p.data, self.cfg.tau)

	def sync_detach_Q(self):
		self._detach_Qs.load_state_dict(self._Qs.state_dict())

	def load_supervised_dynamics(self, fp):
		"""
		Load a supervised Hopper SD checkpoint into the explicit dynamics core only.

		This intentionally does not touch reward / Q / pi heads, nor the optional
		latent residual heads owned by ``SDWorldModel``. The goal is to reuse the
		short-horizon transition fit while leaving TD-MPC2 downstream heads to be
		trained in the RL loop.
		"""
		state = torch.load(fp, map_location='cpu', weights_only=False)
		if isinstance(state, dict) and 'model_state' in state:
			src_state = state['model_state']
		else:
			src_state = state
		target_state = self._dynamics.state_dict()
		compatible = {
			k: v for k, v in src_state.items()
			if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
		}
		missing = sorted(set(target_state.keys()) - set(compatible.keys()))
		unexpected = sorted(k for k in src_state.keys() if k not in compatible)
		self._dynamics.load_state_dict(compatible, strict=False)
		return {
			'loaded_keys': len(compatible),
			'missing_keys': missing,
			'unexpected_keys': unexpected,
		}

	def _flatten_batch(self, x):
		if x.ndim == 1:
			x = x.unsqueeze(0)
		leading = x.shape[:-1]
		return x.reshape(-1, x.shape[-1]), leading

	@staticmethod
	def _restore_batch(x, leading):
		if not isinstance(leading, tuple):
			leading = tuple(leading)
		return x.reshape(*leading, x.shape[-1])

	def _split_latent(self, z):
		z_struct = z[..., :self.struct_dim]
		z_resid = z[..., self.struct_dim:] if self.resid_dim > 0 else None
		return z_struct, z_resid

	def _clamp_resid(self, z_resid):
		s = self.max_resid_scale
		return s * torch.tanh(z_resid / s)

	def _merge_latent(self, z_struct, z_resid=None):
		if self.resid_dim <= 0:
			return z_struct
		return torch.cat([z_struct, z_resid], dim=-1)

	def struct_residual_stats(self, z, a):
		if self.resid_dim <= 0 or self._struct_residual is None:
			zero = torch.zeros((), device=z.device, dtype=z.dtype)
			return zero, zero
		z_flat, _ = self._flatten_batch(z)
		a_flat, _ = self._flatten_batch(a)
		correction = torch.tanh(self._struct_residual(torch.cat([z_flat, a_flat], dim=-1)))
		return correction.square().mean(), correction.abs().mean()

	def struct_residual_penalty(self, z, a):
		penalty, _ = self.struct_residual_stats(z, a)
		return penalty


	def encode(self, obs, task):
		assert task is None
		z_struct = self._dynamics.encode_observation(obs)
		if self.resid_dim <= 0:
			return z_struct
		obs_flat, leading = self._flatten_batch(obs)
		z_resid = self._clamp_resid(self._residual_encoder(obs_flat))
		z_resid = self._restore_batch(z_resid, leading)
		return self._merge_latent(z_struct, z_resid)

	def next(self, z, a, task):
		assert task is None
		z_flat, leading = self._flatten_batch(z)
		a_flat, _ = self._flatten_batch(a)
		z_struct, z_resid = self._split_latent(z_flat)
		next_struct_base = self._dynamics(z_struct, a_flat)
		if (
			bool(getattr(self.cfg, 'sd_online_supervision_enabled', False))
			and bool(getattr(self.cfg, 'sd_detach_core_in_main_loss', True))
		):
			next_struct_base = next_struct_base.detach()
		if self.resid_dim > 0:
			hybrid = torch.cat([z_flat, a_flat], dim=-1)
			raw_correction = self._struct_residual(hybrid)
			base_delta = next_struct_base - z_struct

			# --- trend-protected dimensions ---
			p_idx = list(self._trend_protected_idx)
			p_delta = base_delta[:, p_idx]
			p_mag = p_delta.abs()
			p_sign = p_delta.sign()
			p_corr = torch.tanh(raw_correction[:, p_idx])
			scale_factor = 1.0 + self.trend_protect_max_scale * p_corr
			gated_mag = p_mag * scale_factor.abs()
			hard_floor = torch.maximum(
				(1.0 - self.trend_protect_max_scale) * p_mag,
				torch.full_like(p_mag, self.trend_protect_min_correction),
			)
			hard_ceil = (1.0 + self.trend_protect_max_scale) * p_mag + self.trend_protect_min_correction
			protected_signed = p_sign * torch.clamp(gated_mag, min=hard_floor, max=hard_ceil)

			# --- free dimensions ---
			free_mask = torch.ones(self.struct_dim, dtype=torch.bool, device=z_flat.device)
			free_mask[p_idx] = False
			free_scale = torch.full((self.struct_dim,), self.struct_residual_scale, device=z_flat.device, dtype=z_flat.dtype)
			free_scale[DMControlHopperSDModel.VX_IDX] = self.vx_residual_scale
			free_scale[DMControlHopperSDModel.VZ_IDX] = self.vz_residual_scale
			free_scale[DMControlHopperSDModel.PITCH_IDX] = self.pitch_residual_scale
			free_scale[DMControlHopperSDModel.PITCH_RATE_IDX] = self.pitch_rate_residual_scale
			free_scale[DMControlHopperSDModel.KNEE_IDX] = self.knee_residual_scale
			free_delta = free_scale.unsqueeze(0) * torch.tanh(raw_correction)

			next_struct = next_struct_base.clone()
			next_struct[:, p_idx] = z_struct[:, p_idx] + protected_signed
			next_struct[:, free_mask] = next_struct_base[:, free_mask] + free_delta[:, free_mask]
			next_struct = self._dynamics.project_state(next_struct)
			next_resid = self._clamp_resid(self._residual_dynamics(hybrid))
			next_z = self._merge_latent(next_struct, next_resid)
		else:
			next_z = next_struct_base
		return self._restore_batch(next_z, leading)

	def _scalar_mix_to_logits(self, learned_logits, explicit_scalar, alpha):
		"""Mix learned reward with explicit scalar in value space, return logits."""
		learned_scalar = math.two_hot_inv(learned_logits, self.cfg)
		mixed = (1.0 - alpha) * learned_scalar + alpha * explicit_scalar
		mixed_dist = math.two_hot(mixed, self.cfg)
		return torch.log(mixed_dist.clamp(min=1e-8))

	def _scalar_add_to_logits(self, learned_logits, shaping_delta, alpha):
		"""Add a small shaping delta on top of learned reward in value space."""
		learned_scalar = math.two_hot_inv(learned_logits, self.cfg)
		shifted = learned_scalar + alpha * shaping_delta
		shifted_dist = math.two_hot(shifted, self.cfg)
		return torch.log(shifted_dist.clamp(min=1e-8))

	def action_prior_bounds(self, z, task=None):
		"""Return state-conditional planning bounds from the explicit SD state."""
		del task
		z_flat, leading = self._flatten_batch(z)
		z_struct, _ = self._split_latent(z_flat)
		low, high, stats = self._dynamics.get_action_prior_bounds(z_struct)
		low = self._restore_batch(low, leading)
		high = self._restore_batch(high, leading)
		return low, high, stats

	def _reward_mix_alpha(self):
		if self.reward_mode == 'curriculum':
			return self._curriculum_alpha()
		if self.reward_mode == 'hybrid':
			return self.reward_hybrid_alpha
		return 0.0

	def shaping_reward(self, z, a, task, next_z=None):
		assert task is None
		z_flat, leading = self._flatten_batch(z)
		a_flat, _ = self._flatten_batch(a)
		if next_z is None:
			with torch.no_grad():
				next_z = self.next(z_flat, a_flat, task)
		else:
			next_z, _ = self._flatten_batch(next_z)
		z_struct, _ = self._split_latent(z_flat)
		next_struct, _ = self._split_latent(next_z)
		shaping_delta = self._dynamics.predict_shaping_reward(
			z_struct, a_flat, next_struct,
		).unsqueeze(-1)
		return self._restore_batch(shaping_delta, leading)

	def mixed_reward_target(self, env_reward, z, a, task, next_z=None):
		assert task is None
		reward_flat, leading = self._flatten_batch(env_reward)
		reward_flat = self.reward_env_scale * reward_flat
		if self.reward_mode == 'learned':
			info = {
				"reward_mix_alpha": 0.0,
				"reward_shaping_mean": 0.0,
				"reward_shaping_std": 0.0,
			}
			return self._restore_batch(reward_flat, leading), info
		shaping_delta = self.shaping_reward(z, a, task, next_z=next_z)
		shaping_flat, _ = self._flatten_batch(shaping_delta)
		alpha = self._reward_mix_alpha()
		target = reward_flat + alpha * shaping_flat
		info = {
			"reward_mix_alpha": float(alpha),
			"reward_shaping_mean": float(shaping_flat.mean().item()),
			"reward_shaping_std": float(shaping_flat.std(unbiased=False).item()),
		}
		return self._restore_batch(target, leading), info

	def reward(self, z, a, task, next_z=None):
		assert task is None
		z_flat, leading = self._flatten_batch(z)
		a_flat, _ = self._flatten_batch(a)
		learned_logits = self._reward(torch.cat([z_flat, a_flat], dim=-1))

		if self.reward_mode == 'learned' or self.training:
			reward = learned_logits
		else:
			if self.reward_mode == 'explicit':
				z_struct, _ = self._split_latent(z_flat)
				if next_z is None:
					with torch.no_grad():
						next_z = self.next(z_flat, a_flat, task)
				else:
					next_z, _ = self._flatten_batch(next_z)
				next_struct, _ = self._split_latent(next_z)
				explicit_reward = self._dynamics.predict_plan_reward(
					z_struct, a_flat, next_struct,
				).unsqueeze(-1)
				reward = math.two_hot(explicit_reward, self.cfg).clamp(min=1e-8).log()
			else:
				# Hybrid/curriculum shaping is already baked into the reward-head
				# supervision target. Re-adding it here would double count shaping
				# during planning/eval and make reward usage inconsistent.
				reward = learned_logits
		return self._restore_batch(reward, leading)

	def _curriculum_alpha(self):
		alpha_max = self.reward_curriculum_alpha
		alpha_min = float(getattr(self.cfg, 'sd_reward_curriculum_alpha_min', 0.1))
		if self.reward_curriculum_end <= self.reward_curriculum_start:
			return alpha_min if int(self._reward_curriculum_step.item()) >= self.reward_curriculum_end else alpha_max
		step = int(self._reward_curriculum_step.item())
		if step <= self.reward_curriculum_start:
			return alpha_max
		if step >= self.reward_curriculum_end:
			return alpha_min
		progress = (step - self.reward_curriculum_start) / max(1, self.reward_curriculum_end - self.reward_curriculum_start)
		return alpha_max + (alpha_min - alpha_max) * progress

	def termination(self, z, task, unnormalized=False):
		assert task is None
		if self._termination is None:
			raise RuntimeError('Termination head is disabled because episodic=false.')
		z_flat, leading = self._flatten_batch(z)
		logits = self._termination(z_flat)
		logits = self._restore_batch(logits, leading)
		if unnormalized:
			return logits
		return torch.sigmoid(logits)

	def pi(self, z, task):
		assert task is None
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)
		log_prob = math.gaussian_logprob(eps, log_std)
		scaled_log_prob = log_prob * eps.shape[-1]
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)
		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			'mean': mean,
			'log_std': log_std,
			'action_prob': torch.ones_like(log_prob),
			'entropy': -log_prob,
			'scaled_entropy': -log_prob * entropy_scale,
		}, batch_size=mean.shape[:-1])
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		assert task is None
		assert return_type in {'min', 'avg', 'all'}
		za = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(za)
		if return_type == 'all':
			return out
		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == 'min':
			return Q.min(0).values
		return Q.sum(0) / 2
