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
		self.latent_dim = int(getattr(cfg, 'sd_latent_dim', max(self.struct_dim, 32)))
		assert self.struct_dim == DMControlHopperSDModel.STATE_DIM, 'sd_hopper currently expects 15 structured dims.'
		assert self.latent_dim >= self.struct_dim, 'sd_latent_dim must be >= sd_struct_dim.'
		self.resid_dim = self.latent_dim - self.struct_dim
		self.reward_mode = getattr(cfg, 'sd_reward_mode', 'learned')
		self.reward_hybrid_alpha = float(getattr(cfg, 'sd_reward_hybrid_alpha', 0.1))
		self.reward_curriculum_start = int(getattr(cfg, 'sd_reward_curriculum_start', 2500))
		self.reward_curriculum_end = int(getattr(cfg, 'sd_reward_curriculum_end', 4000))
		self.reward_curriculum_alpha = float(getattr(cfg, 'sd_reward_curriculum_alpha', 0.5))
		self.register_buffer('_reward_curriculum_step', torch.tensor(0, dtype=torch.long))

		self._encoder = nn.Identity()
		self._dynamics = DMControlHopperSDModel(
			dt=getattr(cfg, 'sd_dt', 0.02),
			hidden=getattr(cfg, 'sd_hidden_dim', 256),
			hop_speed=getattr(cfg, 'sd_hop_speed', 2.0),
			residual_scale=getattr(cfg, 'sd_residual_scale', 5.0),
			passive_scale=getattr(cfg, 'sd_passive_scale', 10.0),
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
		self.struct_residual_scale = 0.10
		self.coupling_residual_scale = 0.20
		self.trend_protect_max_scale = 0.20
		self.trend_protect_min_correction = 0.02
		self._trend_protected_idx = (
			DMControlHopperSDModel.HEIGHT_IDX,
			DMControlHopperSDModel.COMP_IDX,
			DMControlHopperSDModel.SUPPORT_IDX,
			DMControlHopperSDModel.STANCE_IDX,
			DMControlHopperSDModel.IMPACT_IDX,
			DMControlHopperSDModel.CLEARANCE_IDX,
		)
		self._coupling_bridge_idx = (
			DMControlHopperSDModel.VZ_IDX,
			DMControlHopperSDModel.PITCH_IDX,
			DMControlHopperSDModel.PITCH_RATE_IDX,
			DMControlHopperSDModel.COMP_RATE_IDX,
			DMControlHopperSDModel.SWING_IDX,
			DMControlHopperSDModel.SWING_RATE_IDX,
			DMControlHopperSDModel.FOOT_IDX,
			DMControlHopperSDModel.FOOT_RATE_IDX,
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
			free_scale[list(self._coupling_bridge_idx)] = self.coupling_residual_scale
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

	def reward(self, z, a, task, next_z=None):
		assert task is None
		z_flat, leading = self._flatten_batch(z)
		a_flat, _ = self._flatten_batch(a)
		learned_logits = self._reward(torch.cat([z_flat, a_flat], dim=-1))

		if self.reward_mode == 'learned' or self.training:
			reward = learned_logits
		else:
			z_struct, _ = self._split_latent(z_flat)
			if next_z is None:
				with torch.no_grad():
					next_z = self.next(z_flat, a_flat, task)
			else:
				next_z, _ = self._flatten_batch(next_z)
			next_struct, _ = self._split_latent(next_z)
			shaping = self._dynamics.predict_plan_reward(
				z_struct, a_flat, next_struct,
			).unsqueeze(-1)
			alpha = self.reward_hybrid_alpha
			if self.reward_mode == 'explicit':
				alpha = 1.0
			elif self.reward_mode == 'curriculum':
				alpha = self._curriculum_alpha()
			if alpha <= 0.0:
				reward = learned_logits
			else:
				reward = self._scalar_mix_to_logits(learned_logits, shaping, alpha)
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
