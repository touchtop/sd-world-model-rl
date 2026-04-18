import random

import torch
from tensordict.tensordict import TensorDict


class Buffer():
	"""
	Minimal replay buffer for TD-MPC2 training.

	The upstream implementation uses torchrl's ReplayBuffer + SliceSampler.
	On this machine, torchrl version constraints conflict with the P100-compatible
	PyTorch build, so we keep a small local implementation that preserves the
	required interface: `add()` and `sample()`.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		replay_capacity = cfg.get('replay_capacity', None)
		self._capacity = int(replay_capacity) if replay_capacity is not None else int(cfg.buffer_size)
		self._num_eps = 0
		self._num_eps_total = 0
		self._episodes = []
		self._num_steps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer in environment steps."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes currently stored."""
		return self._num_eps

	@property
	def num_steps(self):
		"""Return the number of replay steps currently stored."""
		return self._num_steps

	@property
	def fill_ratio(self):
		"""Return replay occupancy ratio."""
		return 0.0 if self._capacity <= 0 else self._num_steps / self._capacity

	def _episode_len(self, td):
		return int(td['reward'].shape[0])

	def _evict_if_needed(self):
		while self._episodes and self._num_steps > self._capacity:
			old = self._episodes.pop(0)
			self._num_steps -= self._episode_len(old)
			self._num_eps -= 1

	def add(self, td):
		"""Add one full episode TensorDict to the buffer."""
		td = td.detach().cpu()
		self._episodes.append(td)
		self._num_steps += self._episode_len(td)
		self._num_eps += 1
		self._num_eps_total += 1
		if self._num_eps_total == 1:
			print(f'Buffer capacity: {self._capacity:,}')
			print('Using CPU memory for storage.')
		self._evict_if_needed()
		return self._num_eps_total

	def _sample_window(self):
		h = self.cfg.horizon + 1
		candidates = [ep for ep in self._episodes if self._episode_len(ep) >= h]
		if not candidates:
			raise RuntimeError('Replay buffer does not contain any episode long enough to sample from.')
		ep = random.choice(candidates)
		start = random.randint(0, self._episode_len(ep) - h)
		return ep[start:start+h]

	def _stack_obs(self, windows):
		first_obs = windows[0].get('obs')
		if isinstance(first_obs, TensorDict):
			keys = list(first_obs.keys())
			return TensorDict(
				{k: torch.stack([w.get('obs').get(k) for w in windows], dim=1) for k in keys},
				batch_size=(self.cfg.horizon + 1, self.cfg.batch_size),
			)
		return torch.stack([w.get('obs') for w in windows], dim=1)

	def sample(self):
		"""Sample a batch of contiguous subsequences and move them to the training device."""
		windows = [self._sample_window() for _ in range(self.cfg.batch_size)]
		td = torch.stack(windows, dim=1)
		obs = self._stack_obs(windows)
		if isinstance(obs, TensorDict):
			obs = obs.to(self._device, non_blocking=True)
		else:
			obs = obs.contiguous().to(self._device, non_blocking=True)
		action = td.get('action')[1:].contiguous().to(self._device, non_blocking=True)
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous().to(self._device, non_blocking=True)
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous().to(self._device, non_blocking=True)
		else:
			terminated = torch.zeros_like(reward, device=self._device)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous().to(self._device, non_blocking=True)
		return obs, action, reward, terminated, task
