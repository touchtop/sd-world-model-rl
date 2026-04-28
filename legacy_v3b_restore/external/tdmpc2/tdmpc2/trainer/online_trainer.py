from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._total_updates = 0

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time if elapsed_time > 0 else 0.0,
			total_updates=self._total_updates,
			update_to_data_ratio=self._total_updates / max(self._step, 1),
		)

	def buffer_metrics(self):
		return dict(
			buffer_num_steps=self.buffer.num_steps,
			buffer_num_eps=self.buffer.num_eps,
			buffer_capacity=self.buffer.capacity,
			buffer_fill_ratio=self.buffer.fill_ratio,
		)

	def checkpoint_metadata(self, updates_this_step=0, train_metrics=None):
		metadata = dict(
			step=self._step,
			episode=self._ep_idx,
			total_updates=self._total_updates,
			updates_this_step=updates_this_step,
			optim=self.agent.optim.state_dict(),
			pi_optim=self.agent.pi_optim.state_dict(),
			scale=self.agent.scale.state_dict(),
			prev_mean=self.agent._prev_mean.detach().cpu(),
		)
		metadata.update(self.buffer_metrics())
		if train_metrics:
			metadata["train_metrics"] = dict(train_metrics)
		return metadata

	def save_full_state(self, fp):
		"""Save complete resumable state to fp: model + optim + buffer + counters.

		This is distinct from `logger.save_agent` (which only persists model state_dict
		plus passive metadata). `save_full_state` bundles the replay buffer episodes
		and all agent/trainer state so that training can be strictly resumed.
		"""
		import os as _os
		_os.makedirs(_os.path.dirname(str(fp)) or '.', exist_ok=True)
		state = {
			'model': self.agent.model.state_dict(),
			'target_Qs': self.agent.model._target_Qs.state_dict() if hasattr(self.agent.model, '_target_Qs') else None,
			'detach_Qs': self.agent.model._detach_Qs.state_dict() if hasattr(self.agent.model, '_detach_Qs') else None,
			'optim': self.agent.optim.state_dict(),
			'pi_optim': self.agent.pi_optim.state_dict(),
			'scale': self.agent.scale.state_dict(),
			'prev_mean': self.agent._prev_mean.detach().cpu(),
			'step': int(self._step),
			'episode': int(self._ep_idx),
			'total_updates': int(self._total_updates),
			'buffer_episodes': list(self.buffer._episodes),
			'buffer_num_steps': int(self.buffer._num_steps),
			'buffer_num_eps': int(self.buffer._num_eps),
			'buffer_num_eps_total': int(getattr(self.buffer, '_num_eps_total', self.buffer._num_eps)),
		}
		torch.save(state, fp)
		print(f'Saved full resumable state to {fp}')

	def load_full_state(self, fp):
		"""Restore complete state saved by `save_full_state`."""
		state = torch.load(fp, map_location='cpu', weights_only=False)
		device = self.agent.device
		model_sd = state['model']
		self.agent.model.load_state_dict(model_sd)
		if state.get('target_Qs') is not None and hasattr(self.agent.model, '_target_Qs'):
			self.agent.model._target_Qs.load_state_dict(state['target_Qs'])
		if state.get('detach_Qs') is not None and hasattr(self.agent.model, '_detach_Qs'):
			self.agent.model._detach_Qs.load_state_dict(state['detach_Qs'])
		self.agent.optim.load_state_dict(state['optim'])
		self.agent.pi_optim.load_state_dict(state['pi_optim'])
		self.agent.scale.load_state_dict(state['scale'])
		self.agent._prev_mean.copy_(state['prev_mean'].to(device))
		self._step = int(state['step'])
		self._ep_idx = int(state['episode'])
		self._total_updates = int(state['total_updates'])
		self.buffer._episodes = list(state['buffer_episodes'])
		self.buffer._num_steps = int(state['buffer_num_steps'])
		self.buffer._num_eps = int(state['buffer_num_eps'])
		self.buffer._num_eps_total = int(state.get('buffer_num_eps_total', state.get('episode', state['buffer_num_eps'])))
		print(f'Resumed from {fp}: step={self._step}, episode={self._ep_idx}, '
			  f'total_updates={self._total_updates}, buffer={self.buffer._num_steps} steps / {self.buffer._num_eps} eps')

	def aggregate_update_metrics(self, metrics_list):
		if not metrics_list:
			return {}
		keys = metrics_list[0].keys()
		out = {}
		for key in keys:
			values = [m[key] for m in metrics_list if key in m]
			out[key] = float(np.mean(values)) if values else 0.0
		return out

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
					torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_reward_std=np.nanstd(ep_rewards),
			episode_reward_min=np.nanmin(ep_rewards),
			episode_reward_max=np.nanmax(ep_rewards),
			episode_reward_list=';'.join(f'{r:.6f}' for r in ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			{
				'obs': obs,
				'action': action.unsqueeze(0),
				'reward': reward.unsqueeze(0),
				'terminated': terminated.unsqueeze(0),
			},
			batch_size=(1,),
		)
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		last_info = None
		while self._step <= self.cfg.steps:
			if self._step > 0 and self._step % self.cfg.eval_freq == 0:
				eval_next = True

			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				# Full-state checkpoints do not preserve the live env / partial episode,
				# so resumed training always starts from a fresh episode boundary.
				if self._step > 0 and last_info is not None and hasattr(self, '_tds') and len(self._tds) > 1:
					if last_info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=last_info['success'],
						episode_length=len(self._tds),
						episode_terminated=last_info['terminated'],
						updates_this_step=0,
					)
					train_metrics.update(self.common_metrics())
					train_metrics.update(self.buffer_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			last_info = info
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			updates_this_step = 0
			if hasattr(self.agent.model, '_reward_curriculum_step'):
				self.agent.model._reward_curriculum_step.fill_(self._step)
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.get('seed_pretrain_updates', self.cfg.seed_steps) or self.cfg.seed_steps)
					print('Pretraining agent on seed data...')
				else:
					num_updates = int(self.cfg.get('updates_per_step', 1) or 1)
				update_metrics = []
				for _ in range(num_updates):
					update_metrics.append(self.agent.update(self.buffer))
				updates_this_step = num_updates
				self._total_updates += num_updates
				train_metrics = self.aggregate_update_metrics(update_metrics)

			self._step += 1

			if self._step >= self.cfg.seed_steps and self._step % int(self.cfg.get('train_log_freq', 100) or 100) == 0:
				step_train_metrics = dict(train_metrics)
				step_train_metrics.update(self.common_metrics())
				step_train_metrics.update(self.buffer_metrics())
				step_train_metrics.update(self.agent.get_plan_metrics())
				step_train_metrics['updates_this_step'] = updates_this_step
				self.logger.log(step_train_metrics, 'train')

			checkpoint_freq = self.cfg.get('checkpoint_freq', None)
			checkpoint_due = checkpoint_freq and self._step % int(checkpoint_freq) == 0
			seed_checkpoint = self._step == self.cfg.seed_steps and self.cfg.get('checkpoint_at_seed', True)
			if checkpoint_due or seed_checkpoint:
				self.logger.save_agent(
					self.agent,
					identifier=f'{self._step}',
					metadata=self.checkpoint_metadata(updates_this_step=updates_this_step, train_metrics=train_metrics),
				)
				if bool(self.cfg.get('save_full_state', False)):
					from pathlib import Path as _P
					work_dir = _P(self.cfg.work_dir)
					self.save_full_state(work_dir / 'models' / f'{self._step}.full.pt')

		self.logger.finish(self.agent)
		if bool(self.cfg.get('save_full_state', False)):
			from pathlib import Path as _P
			work_dir = _P(self.cfg.work_dir)
			self.save_full_state(work_dir / 'models' / 'final.full.pt')
