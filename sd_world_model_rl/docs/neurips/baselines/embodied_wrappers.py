"""
Embodied Environment Wrappers for DreamerV3 Integration.

This module provides wrappers to run our custom environments with DreamerV3.
DreamerV3 requires specific observation format with is_first, is_last, is_terminal.
"""

import numpy as np
import sys
from pathlib import Path

# Add IJOC src path for environment imports
_ijoc_src = Path(__file__).parent.parent.parent / "ijoc"
if _ijoc_src.exists() and str(_ijoc_src) not in sys.path:
    sys.path.insert(0, str(_ijoc_src))

try:
    import embodied
    import elements
    EMBODIED_AVAILABLE = True
except ImportError:
    EMBODIED_AVAILABLE = False
    print("Warning: embodied not available. DreamerV3 wrappers disabled.")


class EmbodiedEnvWrapper:
    """
    Wrapper to convert Gym/Gymnasium environments to embodied.Env interface.

    DreamerV3 requires observations as dicts with specific keys:
    - 'vector' or 'image': the actual observation
    - 'reward': float32 scalar
    - 'is_first': bool, True only on reset
    - 'is_last': bool, True on episode end (including timeouts)
    - 'is_terminal': bool, True only on true terminal states (not timeouts)
    """

    def __init__(self, env, obs_key: str = "vector"):
        """
        Args:
            env: Gym/Gymnasium environment instance
            obs_key: Key name for observation in the dict ('vector' for MLP, 'image' for CNN)
        """
        self._env = env
        self._obs_key = obs_key
        self._done = True

        # Get observation shape
        if hasattr(env.observation_space, 'shape'):
            self._obs_shape = env.observation_space.shape
            self._obs_dtype = np.float32
        else:
            raise ValueError("Environment must have Box observation space")

        # Get action space info
        if hasattr(env.action_space, 'n'):
            # Discrete
            self._act_shape = ()
            self._act_dtype = np.int32
            self._num_actions = env.action_space.n
            self._is_discrete = True
        elif hasattr(env.action_space, 'nvec'):
            # MultiDiscrete - flatten to single action for DreamerV3
            # DreamerV3 handles this via action repeat or we need special handling
            self._act_shape = (len(env.action_space.nvec),)
            self._act_dtype = np.int32
            self._num_actions = env.action_space.nvec
            self._is_discrete = True
            self._is_multi_discrete = True
        else:
            raise ValueError("Only Discrete and MultiDiscrete action spaces supported")

    @property
    def obs_space(self):
        """Return observation space in embodied format."""
        if not EMBODIED_AVAILABLE:
            return None

        spaces = {
            self._obs_key: elements.Space(
                dtype=self._obs_dtype,
                shape=self._obs_shape,
                low=-np.inf,
                high=np.inf,
            ),
            'reward': elements.Space(dtype=np.float32, shape=()),
            'is_first': elements.Space(dtype=bool, shape=()),
            'is_last': elements.Space(dtype=bool, shape=()),
            'is_terminal': elements.Space(dtype=bool, shape=()),
        }
        return spaces

    @property
    def act_space(self):
        """Return action space in embodied format."""
        if not EMBODIED_AVAILABLE:
            return None

        if hasattr(self, '_is_multi_discrete') and self._is_multi_discrete:
            # For MultiDiscrete, we need multiple action keys or flatten
            # Using single flattened action for simplicity
            spaces = {
                'action': elements.Space(
                    dtype=self._act_dtype,
                    shape=self._act_shape,
                    low=0,
                    high=self._num_actions - 1,
                ),
            }
        else:
            spaces = {
                'action': elements.Space(
                    dtype=self._act_dtype,
                    shape=self._act_shape,
                    low=0,
                    high=self._num_actions - 1,
                ),
            }
        return spaces

    def reset(self, seed=None, **kwargs):
        """Reset environment and return initial observation."""
        if seed is not None:
            obs = self._env.reset(seed=seed, **kwargs)
        else:
            obs = self._env.reset(**kwargs)

        # Handle new gymnasium API that returns (obs, info)
        if isinstance(obs, tuple):
            obs, info = obs

        self._done = False

        return {
            self._obs_key: np.asarray(obs, dtype=self._obs_dtype),
            'reward': np.float32(0.0),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    def step(self, action):
        """Execute action and return observation."""
        # Extract action from dict
        if isinstance(action, dict):
            act = action['action']
        else:
            act = action

        # Convert to numpy if needed
        act = np.asarray(act)

        # Handle MultiDiscrete: action shape should match
        if hasattr(self, '_is_multi_discrete') and self._is_multi_discrete:
            if act.shape != self._act_shape:
                # If scalar action received, need to handle somehow
                # For now, assume correct shape
                pass

        # Step environment
        result = self._env.step(act)

        # Handle both old (obs, reward, done, info) and new (obs, reward, terminated, truncated, info) APIs
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        self._done = done

        # is_terminal: True only for true terminal states (not timeouts)
        # is_last: True for any episode end
        is_terminal = terminated and not truncated
        is_last = done

        return {
            self._obs_key: np.asarray(obs, dtype=self._obs_dtype),
            'reward': np.float32(reward),
            'is_first': False,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def close(self):
        """Close environment."""
        if hasattr(self._env, 'close'):
            self._env.close()

    def __getattr__(self, name):
        """Forward unknown attributes to wrapped env."""
        # Avoid infinite recursion for our own attributes
        if name.startswith('_') or name in ('obs_space', 'act_space', 'reset', 'step', 'close'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._env, name)


class BeerGameEmbodied(EmbodiedEnvWrapper):
    """Embodied wrapper specifically for Beer Game environment."""

    def __init__(self, **kwargs):
        from src.env.beer_game_env import BeerGameEnvFlat
        # Use order_levels=3 for consistency with SD-MCTS/BlackBox-MCTS
        if 'order_levels' not in kwargs:
            kwargs['order_levels'] = 3
        env = BeerGameEnvFlat(**kwargs)
        super().__init__(env, obs_key="vector")


class DelayedCollapseEmbodied(EmbodiedEnvWrapper):
    """Embodied wrapper specifically for Delayed Collapse environment."""

    def __init__(self, **kwargs):
        from src.env.delayed_collapse_env import DelayedCollapseEnv
        env = DelayedCollapseEnv(**kwargs)
        super().__init__(env, obs_key="vector")


def make_env(env_name: str, **kwargs):
    """Factory function to create embodied-wrapped environments."""
    if env_name == "beer_game":
        return BeerGameEmbodied(**kwargs)
    elif env_name == "delayed_collapse":
        return DelayedCollapseEmbodied(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


# Test function
def test_wrapper():
    """Test the embodied wrapper."""
    print("Testing EmbodiedEnvWrapper...")

    # Test with Beer Game
    try:
        env = BeerGameEmbodied()
        print(f"  obs_space: {env.obs_space}")
        print(f"  act_space: {env.act_space}")

        obs = env.reset()
        print(f"  reset obs keys: {obs.keys()}")
        print(f"  is_first: {obs['is_first']}")

        # Take a random action
        if hasattr(env._env.action_space, 'nvec'):
            action = {'action': np.array([4, 4, 4, 4])}  # Middle action for each echelon
        else:
            action = {'action': 0}

        obs = env.step(action)
        print(f"  step obs keys: {obs.keys()}")
        print(f"  is_first: {obs['is_first']}, is_last: {obs['is_last']}")

        env.close()
        print("  Beer Game wrapper: OK")
    except Exception as e:
        print(f"  Beer Game wrapper: FAILED - {e}")

    # Test with Delayed Collapse
    try:
        env = DelayedCollapseEmbodied()
        obs = env.reset()
        print(f"  Delayed Collapse reset: OK")
        env.close()
    except Exception as e:
        print(f"  Delayed Collapse wrapper: FAILED - {e}")

    print("Testing complete.")


if __name__ == "__main__":
    test_wrapper()
