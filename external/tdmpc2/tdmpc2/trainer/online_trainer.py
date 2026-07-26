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
		self._hopper_trace_probe = None

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
			sd_optim=self.agent.sd_optim.state_dict() if getattr(self.agent, 'sd_optim', None) is not None else None,
			scale=self.agent.scale.state_dict(),
			prev_mean=self.agent._prev_mean.detach().cpu(),
		)
		metadata.update(self.buffer_metrics())
		if train_metrics:
			metadata["train_metrics"] = dict(train_metrics)
		return metadata

	def _save_step_checkpoint(self, step_identifier, updates_this_step=0, train_metrics=None):
		self.logger.save_agent(
			self.agent,
			identifier=f'{step_identifier}',
			metadata=self.checkpoint_metadata(
				updates_this_step=updates_this_step,
				train_metrics=train_metrics,
			),
		)
		if bool(self.cfg.get('save_full_state', False)):
			work_dir = Path(self.cfg.work_dir)
			self.save_full_state(work_dir / 'models' / f'{step_identifier}.full.pt')

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
			'sd_optim': self.agent.sd_optim.state_dict() if getattr(self.agent, 'sd_optim', None) is not None else None,
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

		def _optimizer_group_count(payload):
			if isinstance(payload, dict):
				return len(payload.get('param_groups', []))
			return 'unknown'

		model_sd = state['model']
		self.agent.model.load_state_dict(model_sd)
		if state.get('target_Qs') is not None and hasattr(self.agent.model, '_target_Qs'):
			self.agent.model._target_Qs.load_state_dict(state['target_Qs'])
		if state.get('detach_Qs') is not None and hasattr(self.agent.model, '_detach_Qs'):
			self.agent.model._detach_Qs.load_state_dict(state['detach_Qs'])
		try:
			self.agent.optim.load_state_dict(state['optim'])
		except (ValueError, KeyError, RuntimeError) as exc:
			old_groups = _optimizer_group_count(state.get('optim'))
			new_groups = _optimizer_group_count(self.agent.optim.state_dict())
			print(
				'Warning: skipped main optimizer state restore due to state mismatch '
				f'(old_param_groups={old_groups}, new_param_groups={new_groups}): {exc}'
			)
		self.agent.pi_optim.load_state_dict(state['pi_optim'])
		if state.get('sd_optim') is not None and getattr(self.agent, 'sd_optim', None) is not None:
			try:
				self.agent.sd_optim.load_state_dict(state['sd_optim'])
			except (ValueError, KeyError, RuntimeError) as exc:
				old_groups = _optimizer_group_count(state.get('sd_optim'))
				new_groups = _optimizer_group_count(self.agent.sd_optim.state_dict())
				print(
					'Warning: skipped SD optimizer state restore due to state mismatch '
					f'(old_param_groups={old_groups}, new_param_groups={new_groups}): {exc}'
				)
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
			and hasattr(self.env.unwrapped, 'physics')
		)

	def _hopper_trace_model(self):
		# Hopper eval traces should be comparable across sd_hopper and vanilla
		# tdmpc2. Use the same explicit state probe in both cases, instead of
		# relying on whichever world model the agent happens to use.
		if self._hopper_trace_probe is None:
			from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel
			self._hopper_trace_probe = DMControlHopperSDModel(
				stage_mode=str(getattr(self.cfg, 'sd_stage_mode', 'full'))
			).cpu()
			self._hopper_trace_probe.eval()
		return self._hopper_trace_probe

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

	def _hopper_eval_state_features(self, obs):
		model = self._hopper_trace_model()
		obs_cpu = obs.detach().cpu().to(dtype=torch.float32).unsqueeze(0)
		with torch.no_grad():
			state = model.encode_observation(obs_cpu)
			feat = model._state_reward_features(state)
			state_id = int(model._dominant_state_ids(feat)[0].item())
			height = float(model.predict_height(state)[0].item())
		return {
			'torso_z': float(feat['torso_z'][0].item()) if 'torso_z' in feat else float(feat['rootz'][0].item()),
			'rootz': float(feat['rootz'][0].item()),
			'vx': float(feat['vx'][0].item()),
			'vz': float(feat['vz'][0].item()),
			'pitch': float(feat['pitch'][0].item()),
			'pitch_rate': float(feat['pitch_rate'][0].item()),
			'activation': float(feat['activation'][0].item()) if 'activation' in feat else 0.0,
			'touch_toe': float(feat['touch_toe'][0].item()),
			'touch_heel': float(feat['touch_heel'][0].item()),
			'z_min': float(feat['z_min'][0].item()),
			'any_contact_z': float(feat['any_contact_z'][0].item()) if 'any_contact_z' in feat else float(feat['z_min'][0].item()),
			'foot_contact_z': float(feat['foot_contact_z'][0].item()) if 'foot_contact_z' in feat else 0.0,
			'body_contact_z': float(feat['body_contact_z'][0].item()) if 'body_contact_z' in feat else 0.0,
			'foot_grounded': float(feat['foot_grounded'][0].item()) if 'foot_grounded' in feat else 0.0,
			'body_collision': float(feat['body_collision'][0].item()) if 'body_collision' in feat else 0.0,
			'airborne_true': float(feat['airborne_true'][0].item()) if 'airborne_true' in feat else 0.0,
			'G': float(feat['G'][0].item()) if 'G' in feat else 0.0,
			'p_contact_x': float(feat['p_contact_x'][0].item()) if 'p_contact_x' in feat else 0.0,
			'p_contact_z': float(feat['p_contact_z'][0].item()) if 'p_contact_z' in feat else 0.0,
			'nose_x_model': float(feat['nose_x'][0].item()) if 'nose_x' in feat else 0.0,
			'pelvis_x_model': float(feat['pelvis_x'][0].item()) if 'pelvis_x' in feat else 0.0,
			'foot_x_model': float(feat['foot_x'][0].item()) if 'foot_x' in feat else 0.0,
			'x_min_model': float(feat['x_min'][0].item()) if 'x_min' in feat else 0.0,
			'x_max_model': float(feat['x_max'][0].item()) if 'x_max' in feat else 0.0,
			'nose_minus_foot_x': float(feat['nose_minus_foot_x'][0].item()) if 'nose_minus_foot_x' in feat else 0.0,
			'nose_minus_anchor_x': float(feat['nose_minus_anchor_x'][0].item()) if 'nose_minus_anchor_x' in feat else 0.0,
			'nose_minus_support_x': float(feat['nose_minus_support_x'][0].item()) if 'nose_minus_support_x' in feat else 0.0,
			'hip_minus_support_x': float(feat['hip_minus_support_x'][0].item()) if 'hip_minus_support_x' in feat else 0.0,
			'leg_length': float(feat['leg_length'][0].item()) if 'leg_length' in feat else 0.0,
			'c_leg': float(feat['c_leg'][0].item()) if 'c_leg' in feat else 0.0,
			'd_c_leg': float(feat['d_c_leg'][0].item()) if 'd_c_leg' in feat else 0.0,
			'e_leg_x': float(feat['e_leg_x'][0].item()) if 'e_leg_x' in feat else 0.0,
			'e_leg_z': float(feat['e_leg_z'][0].item()) if 'e_leg_z' in feat else 0.0,
			'qpos_waist': float(feat['qpos_waist'][0].item()) if 'qpos_waist' in feat else 0.0,
			'qpos_hip': float(feat['qpos_hip'][0].item()) if 'qpos_hip' in feat else 0.0,
			'qpos_knee': float(feat['qpos_knee'][0].item()),
			'qpos_ankle': float(feat['qpos_ankle'][0].item()) if 'qpos_ankle' in feat else 0.0,
			'qvel_waist': float(feat['qvel_waist'][0].item()) if 'qvel_waist' in feat else 0.0,
			'qvel_hip': float(feat['qvel_hip'][0].item()) if 'qvel_hip' in feat else 0.0,
			'qvel_knee': float(feat['qvel_knee'][0].item()) if 'qvel_knee' in feat else 0.0,
			'qvel_ankle': float(feat['qvel_ankle'][0].item()) if 'qvel_ankle' in feat else 0.0,
			'height': height,
			'posture_stability': float(feat['posture_stability'][0].item()),
			'spin_stability': float(feat['spin_stability'][0].item()),
			'forward_progress': float(feat['forward_progress'][0].item()),
			'upward_progress': float(feat['upward_progress'][0].item()),
			'descent_progress': float(feat['descent_progress'][0].item()),
			'backward_speed': float(feat['backward_speed'][0].item()),
			'support_contact': float(feat['support_contact'][0].item()) if 'support_contact' in feat else 0.0,
			'full_foot_support': float(feat['full_foot_support'][0].item()),
			'support_contact_gate': float(feat['support_contact_gate'][0].item()),
			'full_foot_support_gate': float(feat['full_foot_support_gate'][0].item()),
			'stable_vz_gate': float(feat['stable_vz_gate'][0].item()) if 'stable_vz_gate' in feat else 0.0,
			'stable_omega_gate': float(feat['stable_omega_gate'][0].item()) if 'stable_omega_gate' in feat else 0.0,
			'upright_quality_gate': float(feat['upright_quality_gate'][0].item()) if 'upright_quality_gate' in feat else 0.0,
			'prepared_quality_gate': float(feat['prepared_quality_gate'][0].item()) if 'prepared_quality_gate' in feat else 0.0,
			'landing_quality_gate': float(feat['landing_quality_gate'][0].item()) if 'landing_quality_gate' in feat else 0.0,
			'rising_quality_gate': float(feat['rising_quality_gate'][0].item()) if 'rising_quality_gate' in feat else 0.0,
			'state_confident': float(feat['state_confident'][0].item()) if 'state_confident' in feat else 0.0,
			'landing_pose_score': float(feat['landing_pose_score'][0].item()) if 'landing_pose_score' in feat else 0.0,
			'ground_pose_score': float(feat['ground_pose_score'][0].item()) if 'ground_pose_score' in feat else 0.0,
			'rise_pose_score': float(feat['rise_pose_score'][0].item()) if 'rise_pose_score' in feat else 0.0,
			'prone_recover_geom_score': float(feat['prone_recover_geom_score'][0].item()) if 'prone_recover_geom_score' in feat else 0.0,
			'prone_recover_pitch_score': float(feat['prone_recover_pitch_score'][0].item()) if 'prone_recover_pitch_score' in feat else 0.0,
			'prone_recover_spin_score': float(feat['prone_recover_spin_score'][0].item()) if 'prone_recover_spin_score' in feat else 0.0,
			'prone_recover_support_score': float(feat['prone_recover_support_score'][0].item()) if 'prone_recover_support_score' in feat else 0.0,
			'prone_recover_support_gate': float(feat['prone_recover_support_gate'][0].item()) if 'prone_recover_support_gate' in feat else 0.0,
			'prone_recover_grounded_soft': float(feat['prone_recover_grounded_soft'][0].item()) if 'prone_recover_grounded_soft' in feat else 0.0,
			'prone_recover_foot_soft': float(feat['prone_recover_foot_soft'][0].item()) if 'prone_recover_foot_soft' in feat else 0.0,
			'prone_recover_body_clear_score': float(feat['prone_recover_body_clear_score'][0].item()) if 'prone_recover_body_clear_score' in feat else 0.0,
			'prone_recover_torso_over_foot_score': float(feat['prone_recover_torso_over_foot_score'][0].item()) if 'prone_recover_torso_over_foot_score' in feat else 0.0,
			'prone_recover_lift_score': float(feat['prone_recover_lift_score'][0].item()) if 'prone_recover_lift_score' in feat else 0.0,
			'prone_recover_knee_score': float(feat['prone_recover_knee_score'][0].item()) if 'prone_recover_knee_score' in feat else 0.0,
			'prone_recover_progress_score': float(feat['prone_recover_progress_score'][0].item()) if 'prone_recover_progress_score' in feat else 0.0,
			'prone_recover_core_score': float(feat['prone_recover_core_score'][0].item()) if 'prone_recover_core_score' in feat else 0.0,
			'prone_recover_unfold_score': float(feat['prone_recover_unfold_score'][0].item()) if 'prone_recover_unfold_score' in feat else 0.0,
			'prone_recover_stand_score': float(feat['prone_recover_stand_score'][0].item()) if 'prone_recover_stand_score' in feat else 0.0,
			'prone_recover_progress_component': float(feat['prone_recover_progress_component'][0].item()) if 'prone_recover_progress_component' in feat else 0.0,
			'prone_recover_core_component': float(feat['prone_recover_core_component'][0].item()) if 'prone_recover_core_component' in feat else 0.0,
			'prone_recover_unfold_component': float(feat['prone_recover_unfold_component'][0].item()) if 'prone_recover_unfold_component' in feat else 0.0,
			'prone_recover_stand_component': float(feat['prone_recover_stand_component'][0].item()) if 'prone_recover_stand_component' in feat else 0.0,
			'prone_recover_pose_core': float(feat['prone_recover_pose_core'][0].item()) if 'prone_recover_pose_core' in feat else 0.0,
			'prone_recover_spin_cap': float(feat['prone_recover_spin_cap'][0].item()) if 'prone_recover_spin_cap' in feat else 0.0,
			'prone_recover_pose_score': float(feat['prone_recover_pose_score'][0].item()) if 'prone_recover_pose_score' in feat else 0.0,
			'dense_down_gate': float(feat['dense_down_gate'][0].item()) if 'dense_down_gate' in feat else 0.0,
			'dense_ground_gate': float(feat['dense_ground_gate'][0].item()) if 'dense_ground_gate' in feat else 0.0,
			'dense_rise_gate': float(feat['dense_rise_gate'][0].item()) if 'dense_rise_gate' in feat else 0.0,
			'dense_prone_recover_component': float(feat['dense_prone_recover_component'][0].item()) if 'dense_prone_recover_component' in feat else 0.0,
			'dense_pose_potential': float(feat['dense_pose_potential'][0].item()) if 'dense_pose_potential' in feat else 0.0,
			'low_body': float(feat['low_body'][0].item()),
			'knee_window': float(feat['knee_window'][0].item()),
			'vertical_settle': float(feat['vertical_settle'][0].item()),
			'falling_air': float(feat['falling_air'][0].item()),
			'grounded_rear': float(feat['grounded_rear'][0].item()),
			'grounded_prone': float(feat['grounded_prone'][0].item()),
			'grounded_hop': float(feat['grounded_hop'][0].item()) if 'grounded_hop' in feat else 0.0,
			'grounded_upright': float(feat['grounded_upright'][0].item()),
			'grounded_prepared': float(feat['grounded_prepared'][0].item()),
			'rising_air': float(feat['rising_air'][0].item()),
			'dominant_state_id': state_id,
			'rotation_penalty': float(feat['rotation_penalty'][0].item()),
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
			'initial_torso_z': initial['torso_z'],
			'initial_rootz': initial['rootz'],
			'initial_qpos_rootz': initial['qpos_rootz'],
			'final_pitch': final['pitch'],
			'final_torso_z': final['torso_z'],
			'final_rootz': final['rootz'],
			'final_qpos_rootz': final['qpos_rootz'],
			'final_rootx': final['qpos_rootx'],
			'max_rootx': max(float(r['qpos_rootx']) for r in rows),
			'max_torso_z': max(float(r['torso_z']) for r in rows),
			'max_height': max(float(r['height']) for r in rows),
			'max_vx': max(float(r['vx']) for r in rows),
			'max_vz': max(float(r['vz']) for r in rows),
			'max_full_foot_support': max(float(r['full_foot_support']) for r in rows),
			'falling_air_frac': self._mean_of(rows, 'falling_air_gt_05'),
			'grounded_rear_frac': self._mean_of(rows, 'grounded_rear_gt_05'),
			'grounded_prone_frac': self._mean_of(rows, 'grounded_prone_gt_05'),
			'grounded_hop_frac': self._mean_of(rows, 'grounded_hop_gt_05'),
			'grounded_upright_frac': self._mean_of(rows, 'grounded_upright_gt_05'),
			'grounded_prepared_frac': self._mean_of(rows, 'grounded_prepared_gt_05'),
			'rising_air_frac': self._mean_of(rows, 'rising_air_gt_05'),
			'full_foot_support_frac': self._mean_of(rows, 'full_foot_support_gt_05'),
			'backward_speed_frac': self._mean_of(rows, 'backward_speed_gt_05'),
			'rotation_penalty_frac': self._mean_of(rows, 'rotation_penalty_gt_05'),
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
					features = self._hopper_eval_state_features(obs)
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				reward_value = float(reward.item()) if hasattr(reward, 'item') else float(reward)
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
					row['rootx'] = snap['qpos_rootx']
					row['falling_air_gt_05'] = 1.0 if features['falling_air'] >= 0.5 else 0.0
					row['grounded_rear_gt_05'] = 1.0 if features['grounded_rear'] >= 0.5 else 0.0
					row['grounded_prone_gt_05'] = 1.0 if features['grounded_prone'] >= 0.5 else 0.0
					row['grounded_upright_gt_05'] = 1.0 if features['grounded_upright'] >= 0.5 else 0.0
					row['grounded_prepared_gt_05'] = 1.0 if features['grounded_prepared'] >= 0.5 else 0.0
					row['rising_air_gt_05'] = 1.0 if features['rising_air'] >= 0.5 else 0.0
					row['grounded_hop_gt_05'] = 1.0 if features.get('grounded_hop', 0.0) >= 0.5 else 0.0
					row['full_foot_support_gt_05'] = 1.0 if features['full_foot_support'] >= 0.5 else 0.0
					row['backward_speed_gt_05'] = 1.0 if features['backward_speed'] >= 0.5 else 0.0
					row['rotation_penalty_gt_05'] = 1.0 if features['rotation_penalty'] >= 0.5 else 0.0
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
		last_info = None
		while self._step <= self.cfg.steps:
			if self._step > 0 and self._step % self.cfg.eval_freq == 0:
				eval_next = True

			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					self._save_step_checkpoint(
						step_identifier=self._step,
						updates_this_step=0,
						train_metrics=train_metrics,
					)
					eval_next = False

				if self._step > 0 and last_info is not None and len(getattr(self, '_tds', [])) > 1:
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
				last_info = None

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
				self._save_step_checkpoint(
					step_identifier=self._step,
					updates_this_step=updates_this_step,
					train_metrics=train_metrics,
				)

		self.logger.finish(self.agent)
		if bool(self.cfg.get('save_full_state', False)):
			from pathlib import Path as _P
			work_dir = _P(self.cfg.work_dir)
			self.save_full_state(work_dir / 'models' / 'final.full.pt')
