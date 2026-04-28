import gymnasium as gym


class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps
		self.elapsed_steps = 0
	
	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self.elapsed_steps = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.elapsed_steps += 1
		done = done or self.elapsed_steps >= self.max_episode_steps
		return obs, reward, done, info
