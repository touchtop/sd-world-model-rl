import csv
from pathlib import Path
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

	def _should_export_eval_traces(self):
		return (
			bool(self.cfg.get('save_eval_traces', False))
			and getattr(self.cfg, 'task', None) == 'hopper-hop'
			and getattr(self.cfg, 'world_model_kind', None) == 'sd_hopper'
			and hasattr(getattr(self.agent.model, '_dynamics', None), 'encode_observation')
			and hasattr(self.env.unwrapped, 'physics')
		)

	def _eval_trace_out_dir(self):
		return Path(self.cfg.work_dir) / 'official_eval' / f'step_{int(self._step):07d}'

	@staticmethod
	def _physics_snapshot(physics):
		geom_names = ['torso', 'nose', 'pelvis', 'thigh', 'calf', 'foot']
		qpos_names = ['rootx', 'rootz', 'rooty', 'waist', 'hip', 'knee', 'ankle']
		qvel_names = ['rootx', 'rootz', 'rooty', 'waist', 'hip', 'knee', 'ankle']
		geom_xpos = physics.named.data.geom_xpos
		qpos = physics.named.data.qpos
		qvel = physics.named.data.qvel
		snap = {}
		for name in geom_names:
			snap[f'{name}_x'] = float(geom_xpos[name, 'x'])
			snap[f'{name}_z'] = float(geom_xpos[name, 'z'])
		for name in qpos_names:
			snap[f'qpos_{name}'] = float(qpos[name])
		for name in qvel_names:
			snap[f'qvel_{name}'] = float(qvel[name])
		return snap

	def _hopper_eval_state_features(self, snap, struct):
		model = self.agent.model._dynamics
		state = torch.tensor(struct, dtype=torch.float32).unsqueeze(0)
		rootz = float(snap['qpos_rootz'])
		pitch = float(snap['qpos_rooty'])
		support = float(struct[model.SUPPORT_IDX])
		compression = float(struct[model.COMP_IDX])
		vx = float(struct[model.VX_IDX])
		vz = float(struct[model.VZ_IDX])
		capture_memory = float(struct[model.CAPTURE_IDX])
		with torch.no_grad():
			height = float(model.predict_height(state)[0].item())
			body_low = float(model._body_low_from_rootz(state[:, model.ROOTZ_IDX])[0].item())
			body_high = 1.0 - body_low
			forward_lean = float(torch.sigmoid((state[:, model.PITCH_IDX] - model.V4_PITCH_FWD_CENTER) / model.V4_PITCH_FWD_SCALE)[0].item())
			recovered_upright = float(model._recovered_upright_from_state(state[:, model.PITCH_IDX], state[:, model.ROOTZ_IDX])[0].item())
			handoff_ready = float(model._handoff_ready_from_state(state[:, model.PITCH_IDX], state[:, model.ROOTZ_IDX])[0].item())
			rear_trap = float(model._rear_recovery_phase_from_state(state[:, model.PITCH_IDX], state[:, model.ROOTZ_IDX])[0].item())
			crawl_lane = body_low * forward_lean * float(model._capture_commitment(state[:, model.CAPTURE_IDX])[0].item())
			front_contact_geom = float(model._front_contact_geom_from_state(state[:, model.PITCH_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.SUPPORT_IDX])[0].item())
			capture_commitment = float(model._capture_commitment(state[:, model.CAPTURE_IDX])[0].item())
			recovery_support = float(model._recovery_support_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.CAPTURE_IDX], state[:, model.ROOTZ_IDX])[0].item())
			standup_window = float(model._standup_window_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.CAPTURE_IDX], state[:, model.ROOTZ_IDX])[0].item())
			arm_ready = float(model._arm_ready_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.CAPTURE_IDX], state[:, model.ROOTZ_IDX])[0].item())
			release_phase = float(model._release_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			handoff_phase = float(model._handoff_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			precapture_phase = float(model._precapture_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			stable_capture_phase = float(model._stable_capture_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			capture_phase = float(model._capture_crawl_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			crawl_to_standup_bias = float(model._crawl_to_standup_bias_from_state(state[:, model.VX_IDX], state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			standup_phase = float(model._standup_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
			release_ready = release_phase * min(max(compression / 0.35, 0.0), 1.0)
		unsupported_low = body_low * (1.0 - capture_commitment)
		foot_low = 1.0 if snap['foot_z'] < 0.08 else 0.0
		nose_low = 1.0 if snap['nose_z'] < 0.08 else 0.0
		front_support = max(foot_low, nose_low)
		launch_like = 1.0 if (vz > 0.20 and support < 0.18) else 0.0
		return {
			'rootz': rootz,
			'pitch': pitch,
			'body_low': body_low,
			'body_high': body_high,
			'forward_lean': forward_lean,
			'recovered_upright': recovered_upright,
			'handoff_ready': handoff_ready,
			'rear_trap': rear_trap,
			'crawl_lane': crawl_lane,
			'front_contact_geom': front_contact_geom,
			'capture_memory': capture_memory,
			'capture_commitment': capture_commitment,
			'recovery_support': recovery_support,
			'standup_window': standup_window,
			'arm_ready': arm_ready,
			'release_ready': release_ready,
			'rear_recovery_phase': rear_trap,
			'precapture_phase': precapture_phase,
			'stable_capture_phase': stable_capture_phase,
			'capture_phase': capture_phase,
			'crawl_to_standup_bias': crawl_to_standup_bias,
			'standup_phase': standup_phase,
			'release_phase': release_phase,
			'handoff_phase': handoff_phase,
			'unsupported_low': unsupported_low,
			'support': support,
			'height': height,
			'vx': vx,
			'vz': vz,
			'foot_low': foot_low,
			'nose_low': nose_low,
			'front_support': front_support,
			'launch_like': launch_like,
		}

	@staticmethod
	def _mean_of(rows, key):
		values = [float(r[key]) for r in rows]
		return float(sum(values) / max(len(values), 1))

	def _write_eval_trace_summary(self, summary_rows, out_dir):
		if not summary_rows:
			return
		summary_csv = out_dir / 'eval_episode_summary.csv'
		with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
			writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
			writer.writeheader()
			writer.writerows(summary_rows)

	def _summarize_eval_trace_episode(self, ep_idx, rows, ep_reward, ep_success, trace_csv):
		initial = rows[0]
		final = rows[-1]
		return {
			'episode_idx': ep_idx,
			'episode_reward': ep_reward,
			'episode_success': ep_success,
			'length': len(rows),
			'initial_pitch': initial['pitch'],
			'initial_rootz': initial['rootz'],
			'final_pitch': final['pitch'],
			'final_rootz': final['rootz'],
			'final_rootx': final['qpos_rootx'],
			'max_rootx': max(float(r['qpos_rootx']) for r in rows),
			'max_height': max(float(r['height']) for r in rows),
			'max_vx': max(float(r['vx']) for r in rows),
			'max_vz': max(float(r['vz']) for r in rows),
			'rear_trap_frac': self._mean_of(rows, 'rear_trap_gt_05'),
			'crawl_frac': self._mean_of(rows, 'crawl_gt_05'),
			'upright_frac': self._mean_of(rows, 'upright_gt_05'),
			'capture_commitment_frac': self._mean_of(rows, 'capture_commitment_gt_05'),
			'standup_frac': self._mean_of(rows, 'standup_gt_05'),
			'crawl_to_standup_frac': self._mean_of(rows, 'crawl_to_standup_gt_05'),
			'arm_ready_frac': self._mean_of(rows, 'arm_ready_gt_05'),
			'release_ready_frac': self._mean_of(rows, 'release_ready_gt_05'),
			'trace_csv': str(trace_csv),
		}

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		export_eval_traces = self._should_export_eval_traces()
		trace_summary_rows = []
		trace_csv_dir = None
		if export_eval_traces:
			trace_out_dir = self._eval_trace_out_dir()
			trace_csv_dir = trace_out_dir / 'episode_csv'
			trace_csv_dir.mkdir(parents=True, exist_ok=True)
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			trace_rows = []
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
					torch.compiler.cudagraph_mark_step_begin()
				if export_eval_traces:
					physics = self.env.unwrapped.physics
					snap = self._physics_snapshot(physics)
					obs_cuda = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
					with torch.no_grad():
						struct = self.agent.model._dynamics.encode_observation(obs_cuda)[0].detach().cpu().numpy()
					features = self._hopper_eval_state_features(snap, struct)
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				reward_value = float(reward.item())
				ep_reward += reward_value
				if export_eval_traces:
					action_np = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
					row = {
						'episode': i,
						'step': t,
						'reward': reward_value,
						'done': int(bool(done)),
					}
					row.update(snap)
					row.update(features)
					row['rear_trap_gt_05'] = 1.0 if features['rear_trap'] >= 0.5 else 0.0
					row['crawl_gt_05'] = 1.0 if features['crawl_lane'] >= 0.5 else 0.0
					row['upright_gt_05'] = 1.0 if features['recovered_upright'] >= 0.5 else 0.0
					row['capture_memory_gt_05'] = 1.0 if features['capture_memory'] >= 0.5 else 0.0
					row['capture_commitment_gt_05'] = 1.0 if features['capture_commitment'] >= 0.5 else 0.0
					row['standup_gt_05'] = 1.0 if features['standup_window'] >= 0.5 else 0.0
					row['crawl_to_standup_gt_05'] = 1.0 if features['crawl_to_standup_bias'] >= 0.5 else 0.0
					row['arm_ready_gt_05'] = 1.0 if features['arm_ready'] >= 0.5 else 0.0
					row['release_ready_gt_05'] = 1.0 if features['release_ready'] >= 0.5 else 0.0
					row['unsupported_low_gt_05'] = 1.0 if features['unsupported_low'] >= 0.5 else 0.0
					for j in range(action_np.size):
						row[f'action_{j}'] = float(action_np[j])
					trace_rows.append(row)
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
			if export_eval_traces:
				trace_csv = trace_csv_dir / f'episode_{i:02d}.csv'
				with open(trace_csv, 'w', newline='', encoding='utf-8') as f:
					writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
					writer.writeheader()
					writer.writerows(trace_rows)
				trace_summary_rows.append(self._summarize_eval_trace_episode(i, trace_rows, ep_reward, info['success'], trace_csv))
		if export_eval_traces:
			self._write_eval_trace_summary(trace_summary_rows, self._eval_trace_out_dir())
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
		while self._step <= self.cfg.steps:
			if self._step > 0 and self._step % self.cfg.eval_freq == 0:
				eval_next = True

			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'],
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
