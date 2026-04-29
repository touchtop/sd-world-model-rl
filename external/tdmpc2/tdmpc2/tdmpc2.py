import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.sd_world_model import SDWorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		world_model_kind = getattr(cfg, 'world_model_kind', 'tdmpc2')
		world_model_cls = SDWorldModel if world_model_kind == 'sd_hopper' else WorldModel
		self.model = world_model_cls(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		for _attr in ('_residual_encoder', '_residual_dynamics', '_struct_residual'):
			_mod = getattr(self.model, _attr, None)
			if _mod is not None and isinstance(_mod, torch.nn.Module):
				self.optim.add_param_group({'params': list(_mod.parameters()), 'lr': self.cfg.lr})
		self.sd_optim = None
		if bool(getattr(self.cfg, 'sd_online_supervision_enabled', False)) and hasattr(self.model, '_dynamics'):
			sd_lr = getattr(self.cfg, 'sd_online_supervision_lr', None)
			if sd_lr is None:
				sd_lr = self.cfg.lr
			self.sd_optim = torch.optim.Adam(
				self.model._dynamics.parameters(),
				lr=float(sd_lr),
				capturable=True,
			)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self.register_buffer(
			'_prev_mean',
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device),
		)
		self._last_plan_metrics = {}
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp, metadata=None):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
			metadata (dict | None): Optional training state payload.
		"""
		payload = {"model": self.model.state_dict()}
		if metadata:
			payload.update(metadata)
		torch.save(payload, fp)

	def get_plan_metrics(self):
		return dict(self._last_plan_metrics)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=self.device, weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		# Skip legacy api_model_conversion for SD checkpoints (their _Qs layout is modules_list.*)
		if not any('modules_list' in k for k in state_dict.keys()):
			state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			next_z = self.model.next(z, actions[t], task)
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task, next_z=next_z), self.cfg)
			z = next_z
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
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
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				if prior_low is not None:
					pi_actions[t] = torch.max(torch.min(pi_actions[t], prior_high), prior_low)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)
			if prior_low is not None:
				pi_actions[-1] = torch.max(torch.min(pi_actions[-1], prior_high), prior_low)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			if prior_low is not None:
				actions_sample = torch.max(
					torch.min(actions_sample, prior_high.view(1, 1, -1)),
					prior_low.view(1, 1, -1),
				)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			raw_value = self._estimate_value(z, actions, task)
			nonfinite_frac = (~torch.isfinite(raw_value)).float().mean()
			value = raw_value.nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
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
		}
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		}, batch_size=torch.Size([]))
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _sd_struct_loss_scales(self, target):
		"""Return per-channel scales for SD-structure losses from target batch statistics."""
		mode = str(getattr(self.cfg, 'sd_struct_loss_scale_mode', 'batch_std'))
		floor = float(getattr(self.cfg, 'sd_struct_loss_scale_floor', 0.1))
		if mode == 'none':
			return torch.ones(target.shape[-1], dtype=target.dtype, device=target.device)
		if mode != 'batch_std':
			raise ValueError(f'Unknown sd_struct_loss_scale_mode: {mode}')
		flat = target.detach().reshape(-1, target.shape[-1])
		scales = flat.std(dim=0, unbiased=False)
		return scales.clamp_min(floor)

	def _sd_struct_mse(self, pred, target, weights=None, scales=None):
		if scales is None:
			scales = self._sd_struct_loss_scales(target)
		while scales.ndim < pred.ndim:
			scales = scales.unsqueeze(0)
		err = ((pred - target) / scales) ** 2
		if weights is None:
			return err.mean()
		while weights.ndim < err.ndim:
			weights = weights.unsqueeze(-1)
		denom = weights.sum() * pred.shape[-1] + 1e-6
		return (err * weights).sum() / denom

	def _sd_latent_consistency_mse(self, pred_z, target_z):
		if not hasattr(self.model, '_split_latent'):
			return F.mse_loss(pred_z, target_z)
		pred_struct, pred_resid = self.model._split_latent(pred_z)
		true_struct, true_resid = self.model._split_latent(target_z)
		struct_loss = self._sd_struct_mse(pred_struct, true_struct)
		if pred_resid is None or true_resid is None:
			return struct_loss
		resid_loss = F.mse_loss(pred_resid, true_resid)
		struct_weight = float(getattr(self.cfg, 'sd_latent_consistency_struct_weight', 1.0))
		resid_weight = float(getattr(self.cfg, 'sd_latent_consistency_resid_weight', 1.0))
		denom = struct_weight + resid_weight
		if denom <= 0:
			return struct_loss
		return (struct_weight * struct_loss + resid_weight * resid_loss) / denom

	def _sd_transition_weights(self, struct_now, struct_next):
		dyn = self.model._dynamics
		flat_now = struct_now.reshape(-1, struct_now.shape[-1])
		flat_next = struct_next.reshape(-1, struct_next.shape[-1])
		with torch.no_grad():
			feat_now = dyn._state_reward_features(flat_now)
			feat_next = dyn._state_reward_features(flat_next)
			grounded_now = feat_now["any_contact_z"] <= dyn.STATE_GROUND_EPS
			grounded_next = feat_next["any_contact_z"] <= dyn.STATE_GROUND_EPS
			landing = (feat_now["airborne_true"] >= 0.5) & grounded_next
			takeoff = grounded_now & (feat_next["airborne_true"] >= 0.5)
			near_contact = (feat_now["any_contact_z"] > dyn.STATE_GROUND_EPS) & (feat_now["any_contact_z"] <= 0.10)
			high_omega = feat_now["pitch_rate"].abs() >= dyn.STATE_UPRIGHT_OMEGA_MAX
			critical = landing | takeoff | near_contact | high_omega
			phase_weight = float(getattr(self.cfg, 'sd_online_supervision_phase_weight', 1.0))
			weights = torch.ones_like(feat_now["pitch_rate"])
			if phase_weight > 1.0:
				weights = torch.where(critical, torch.full_like(weights, phase_weight), weights)
		return (
			weights.reshape(struct_now.shape[:-1]),
			critical.to(dtype=struct_now.dtype).reshape(struct_now.shape[:-1]),
		)

	def _sd_planning_weights(self, td_targets=None, reward_target=None, expected_shape=None):
		if not bool(getattr(self.cfg, 'sd_online_planning_weight_enabled', False)):
			return None
		mode = str(getattr(self.cfg, 'sd_online_planning_weight_mode', 'td_positive'))
		source = td_targets if mode.startswith('td_') else reward_target
		if source is None:
			return None
		with torch.no_grad():
			if mode in ('td_positive', 'reward_positive'):
				score = source.detach().clamp_min(0)
			elif mode in ('td_abs', 'reward_abs'):
				score = source.detach().abs()
			else:
				raise ValueError(f'Unknown sd_online_planning_weight_mode: {mode}')
			if score.ndim > 0 and score.shape[-1] == 1:
				score = score.squeeze(-1)
			if expected_shape is not None and score.shape != expected_shape:
				expected_numel = 1
				for dim in expected_shape:
					expected_numel *= int(dim)
				if score.numel() != expected_numel:
					raise ValueError(
						f'sd planning weights shape {tuple(score.shape)} cannot match {tuple(expected_shape)}'
					)
				score = score.reshape(expected_shape)
			scale = score.mean()
			if scale.item() <= 1e-6:
				return torch.ones_like(score)
			coef = float(getattr(self.cfg, 'sd_online_planning_weight_coef', 1.0))
			weights = 1.0 + coef * (score / scale - 1.0)
			return weights.clamp(min=0.0, max=float(getattr(self.cfg, 'sd_online_planning_weight_max', 5.0)))

	def _update_sd_supervision(self, obs, action, td_targets=None, reward_target=None):
		if self.sd_optim is None or not hasattr(self.model, '_dynamics'):
			return {}
		if isinstance(obs, TensorDict):
			return {}
		dyn = self.model._dynamics
		horizon = action.shape[0]
		true_struct = dyn.encode_observation(obs.reshape(-1, obs.shape[-1])).reshape(
			obs.shape[0], obs.shape[1], dyn.STATE_DIM,
		)
		true_now = true_struct[:-1]
		true_next = true_struct[1:]
		pred_next = torch.stack([dyn(true_struct[t], action[t]) for t in range(horizon)], dim=0)

		step_weights, critical_mask = self._sd_transition_weights(true_now, true_next)
		planning_weights = self._sd_planning_weights(
			td_targets=td_targets,
			reward_target=reward_target,
			expected_shape=step_weights.shape,
		)
		if planning_weights is not None:
			step_weights = step_weights * planning_weights
		struct_scales = self._sd_struct_loss_scales(true_next)
		one_step_loss = self._sd_struct_mse(pred_next, true_next, step_weights, scales=struct_scales)

		omega_idx = dyn.PITCH_RATE_IDX
		contact_slice = slice(dyn.TOE_TOUCH_IDX, dyn.HEEL_TOUCH_IDX + 1)
		omega_loss = self._sd_struct_mse(
			pred_next[..., omega_idx:omega_idx+1],
			true_next[..., omega_idx:omega_idx+1],
			step_weights,
			scales=struct_scales[omega_idx:omega_idx+1],
		)
		contact_loss = self._sd_struct_mse(
			pred_next[..., contact_slice],
			true_next[..., contact_slice],
			step_weights,
			scales=struct_scales[contact_slice],
		)

		rollout_pred = true_struct[0]
		for t in range(horizon):
			rollout_pred = dyn(rollout_pred, action[t])
		rollout_target = true_struct[horizon]
		rollout_weights, rollout_critical = self._sd_transition_weights(
			true_struct[0:1], rollout_target.unsqueeze(0),
		)
		if planning_weights is not None:
			rollout_weights = rollout_weights * planning_weights.mean(dim=0, keepdim=True)
		rollout_loss = self._sd_struct_mse(
			rollout_pred.unsqueeze(0),
			rollout_target.unsqueeze(0),
			rollout_weights,
			scales=struct_scales,
		)

		total_sd_loss = (
			float(getattr(self.cfg, 'sd_online_supervision_one_step_weight', 1.0)) * one_step_loss
			+ float(getattr(self.cfg, 'sd_online_supervision_rollout_weight', 1.0)) * rollout_loss
			+ float(getattr(self.cfg, 'sd_online_supervision_omega_weight', 0.0)) * omega_loss
			+ float(getattr(self.cfg, 'sd_online_supervision_contact_weight', 0.0)) * contact_loss
		)

		self.sd_optim.zero_grad(set_to_none=True)
		total_sd_loss.backward()
		sd_grad_norm = torch.nn.utils.clip_grad_norm_(dyn.parameters(), self.cfg.grad_clip_norm)
		self.sd_optim.step()
		self.sd_optim.zero_grad(set_to_none=True)

		return {
			"sd_online_supervision_loss": total_sd_loss.detach(),
			"sd_online_one_step_loss": one_step_loss.detach(),
			"sd_online_rollout_loss": rollout_loss.detach(),
			"sd_online_omega_loss": omega_loss.detach(),
			"sd_online_contact_loss": contact_loss.detach(),
			"sd_online_critical_frac": critical_mask.mean().detach(),
			"sd_online_rollout_critical_frac": rollout_critical.mean().detach(),
			"sd_online_planning_weight_mean": (
				planning_weights.mean().detach()
				if planning_weights is not None
				else torch.zeros((), device=self.device)
			),
			"sd_online_grad_norm": sd_grad_norm.detach(),
		}

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			reward_mix_info = {}
			z_target = self.model.encode(obs[:-1], task)
			next_z = self.model.encode(obs[1:], task)
			if hasattr(self.model, 'mixed_reward_target'):
				reward_target, reward_mix_info = self.model.mixed_reward_target(
					reward, z_target, action, task, next_z=next_z,
				)
			else:
				reward_target = reward
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = torch.zeros((), device=self.device)
		struct_consistency_loss = torch.zeros((), device=self.device)
		resid_consistency_loss = torch.zeros((), device=self.device)
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + self._sd_latent_consistency_mse(z, _next_z) * self.cfg.rho**t
			if hasattr(self.model, '_split_latent'):
				pred_struct, pred_resid = self.model._split_latent(z)
				true_struct, true_resid = self.model._split_latent(_next_z)
				struct_consistency_loss = struct_consistency_loss + self._sd_struct_mse(pred_struct, true_struct) * self.cfg.rho**t
				if pred_resid is not None and true_resid is not None:
					resid_consistency_loss = resid_consistency_loss + F.mse_loss(pred_resid, true_resid) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_target_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward_target.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_target_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		struct_consistency_loss = struct_consistency_loss / self.cfg.horizon
		resid_consistency_loss = resid_consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)

		# Auxiliary loss: train height_head to predict the true height from
		# the rollout latent.  Without this, height_head has no gradient and
		# its contribution to predict_reward / predict_plan_reward is either
		# random (after Bug 1 fix) or accidentally near-zero (current state).
		height_loss = 0.0
		height_coef = float(getattr(self.cfg, 'sd_height_loss_coef', 0.0))
		if height_coef > 0 and hasattr(self.model, '_dynamics') and hasattr(self.model._dynamics, 'height_head'):
			try:
				with torch.no_grad():
					obs_flat = obs[:-1].reshape(-1, obs.shape[-1])
					true_struct = self.model._dynamics.encode_observation(obs_flat)
					true_height = true_struct[:, self.model._dynamics.HEIGHT_IDX]
					true_height = true_height.reshape(self.cfg.horizon, self.cfg.batch_size)
				z_struct_rollout, _ = self.model._split_latent(_zs)
				z_struct_flat = z_struct_rollout.reshape(-1, z_struct_rollout.shape[-1])
				pred_height = self.model._dynamics.predict_height(z_struct_flat)
				pred_height = pred_height.reshape(self.cfg.horizon, self.cfg.batch_size)
				for t in range(self.cfg.horizon):
					height_loss = height_loss + F.mse_loss(pred_height[t], true_height[t]) * self.cfg.rho**t
				height_loss = height_loss / self.cfg.horizon
			except Exception:
				height_loss = 0.0

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss +
			height_coef * height_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)
		self.model.sync_detach_Q()

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Update shared SD transition core with an explicit replay-supervision step.
		sd_supervision_info = self._update_sd_supervision(obs, action, td_targets=td_targets, reward_target=reward_target)

		# Return training statistics
		self.model.eval()
		q_values = math.two_hot_inv(qs, self.cfg)
		reward_pred_values = math.two_hot_inv(reward_preds, self.cfg)
		extra_info = {}
		if hasattr(self.model, '_split_latent'):
			z_struct, z_resid = self.model._split_latent(zs)
			extra_info['struct_norm'] = z_struct.norm(dim=-1).mean()
			if z_resid is not None:
				extra_info['residual_norm'] = z_resid.norm(dim=-1).mean()
		if hasattr(self.model, '_dynamics') and hasattr(self.model._dynamics, 'predict_reward'):
			try:
				struct_now, _ = self.model._split_latent(_zs)
				struct_next, _ = self.model._split_latent(zs[1:])
				explicit_reward = self.model._dynamics.predict_reward(
					struct_now.reshape(-1, struct_now.shape[-1]),
					action.reshape(-1, action.shape[-1]),
					struct_next.reshape(-1, struct_next.shape[-1]),
				).reshape_as(reward)
				extra_info['explicit_reward_mean'] = explicit_reward.mean()
				extra_info['explicit_reward_std'] = explicit_reward.std(unbiased=False)
			except Exception:
				pass
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"struct_consistency_loss": struct_consistency_loss,
			"resid_consistency_loss": resid_consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"height_loss": height_loss if torch.is_tensor(height_loss) else torch.tensor(float(height_loss)),
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"td_target_mean": td_targets.mean(),
			"td_target_std": td_targets.std(unbiased=False),
			"q_mean": q_values.mean(),
			"q_std": q_values.std(unbiased=False),
			"reward_target_mean": reward_target.mean(),
			"reward_target_std": reward_target.std(unbiased=False),
			"reward_pred_mean": reward_pred_values.mean(),
			"reward_pred_std": reward_pred_values.std(unbiased=False),
		}, batch_size=torch.Size([]))
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		if extra_info:
			info.update(TensorDict(extra_info, batch_size=torch.Size([])))
		if reward_mix_info:
			info.update(TensorDict(reward_mix_info, batch_size=torch.Size([])))
		if sd_supervision_info:
			info.update(TensorDict(sd_supervision_info, batch_size=torch.Size([])))
		info = info.detach()
		return {
			k: v.mean().item() if torch.is_tensor(v) else v
			for k, v in info.items()
		}

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
			torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
