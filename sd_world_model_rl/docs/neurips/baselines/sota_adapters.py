"""
SOTA Baseline Adapters for NeurIPS experiments.

Provides unified interface for DreamerV3, TD-MPC2, and other SOTA methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np


class BaselineAgent(ABC):
    """Abstract base class for all baseline agents."""

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Select action given state."""
        pass

    @abstractmethod
    def train(self, replay_buffer: List[Tuple]) -> Dict:
        """Train on collected data."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return agent name for logging."""
        pass


class DreamerV3Adapter(BaselineAgent):
    """
    Adapter for DreamerV3.

    Reference: https://github.com/danijar/dreamerv3
    Paper: Mastering Diverse Domains through World Models (2023)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or self._default_config()
        self._initialized = False

    def _default_config(self) -> Dict:
        """Default DreamerV3 config for discrete action tasks."""
        return {
            "model_size": "small",  # small/medium/large/xlarge
            "batch_size": 16,
            "batch_length": 64,
            "train_ratio": 512,
            "gamma": 0.997,
            "actor_lr": 3e-5,
            "critic_lr": 3e-5,
            "model_lr": 1e-4,
        }

    def _lazy_init(self):
        """Lazy initialization of DreamerV3 components."""
        if self._initialized:
            return

        try:
            # Try to import dreamerv3
            # Note: Requires separate installation
            import dreamerv3
            self.agent = dreamerv3.Agent(
                obs_space={"state": (self.state_dim,)},
                act_space={"action": self.action_dim},
                config=self.config,
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "DreamerV3 not installed. Please install from: "
                "https://github.com/danijar/dreamerv3"
            )

    def act(self, state: np.ndarray) -> int:
        self._lazy_init()
        obs = {"state": state}
        action = self.agent.policy(obs)
        return int(action["action"])

    def train(self, replay_buffer: List[Tuple]) -> Dict:
        self._lazy_init()
        # Convert replay buffer to DreamerV3 format
        # Implementation depends on DreamerV3 API
        raise NotImplementedError("DreamerV3 training integration pending")

    def get_name(self) -> str:
        return "DreamerV3"


class TDMPC2Adapter(BaselineAgent):
    """
    Adapter for TD-MPC2.

    Reference: https://github.com/nicklashansen/tdmpc2
    Paper: TD-MPC2: Scalable, Robust World Models (2024)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or self._default_config()
        self._initialized = False

    def _default_config(self) -> Dict:
        """Default TD-MPC2 config."""
        return {
            "model_size": 5,  # Model size in millions of parameters
            "horizon": 3,
            "mpc_iterations": 6,
            "population_size": 512,
            "elite_fraction": 0.1,
        }

    def _lazy_init(self):
        if self._initialized:
            return

        try:
            # Note: TD-MPC2 requires separate installation
            import tdmpc2
            self.agent = tdmpc2.TDMPC2(
                obs_dim=self.state_dim,
                action_dim=self.action_dim,
                **self.config,
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "TD-MPC2 not installed. Please install from: "
                "https://github.com/nicklashansen/tdmpc2"
            )

    def act(self, state: np.ndarray) -> int:
        self._lazy_init()
        action = self.agent.act(state)
        # TD-MPC2 outputs continuous, discretize for our tasks
        return int(np.round(action * (self.action_dim - 1)))

    def train(self, replay_buffer: List[Tuple]) -> Dict:
        self._lazy_init()
        raise NotImplementedError("TD-MPC2 training integration pending")

    def get_name(self) -> str:
        return "TD-MPC2"


class MuZeroAdapter(BaselineAgent):
    """
    Adapter for MuZero.

    Reference: https://github.com/werner-duvaud/muzero-general
    Paper: Mastering Atari, Go, Chess and Shogi (2020)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or self._default_config()
        self._initialized = False

    def _default_config(self) -> Dict:
        """Default MuZero config for our tasks."""
        return {
            "num_simulations": 50,
            "discount": 0.997,
            "batch_size": 128,
            "td_steps": 10,
            "num_unroll_steps": 5,
            "lr_init": 0.003,
        }

    def _lazy_init(self):
        if self._initialized:
            return

        try:
            # Using muzero-general implementation
            from muzero_general import MuZero
            self.agent = MuZero(
                observation_shape=(self.state_dim,),
                action_space_size=self.action_dim,
                **self.config,
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "MuZero not installed. Please install muzero-general from: "
                "https://github.com/werner-duvaud/muzero-general"
            )

    def act(self, state: np.ndarray) -> int:
        self._lazy_init()
        return self.agent.select_action(state)

    def train(self, replay_buffer: List[Tuple]) -> Dict:
        self._lazy_init()
        raise NotImplementedError("MuZero training integration pending")

    def get_name(self) -> str:
        return "MuZero"


class PPOAdapter(BaselineAgent):
    """
    Adapter for PPO using Stable-Baselines3.

    This is a model-free baseline for comparison.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.model = None

    def act(self, state: np.ndarray) -> int:
        if self.model is None:
            return np.random.randint(self.action_dim)
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def train(self, env, total_timesteps: int = 100000) -> Dict:
        try:
            from stable_baselines3 import PPO
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                **self.config,
            )
            self.model.learn(total_timesteps=total_timesteps)
            return {"status": "trained"}
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed. "
                "pip install stable-baselines3"
            )

    def get_name(self) -> str:
        return "PPO"


def create_baseline(name: str, state_dim: int, action_dim: int, **kwargs) -> BaselineAgent:
    """Factory function to create baseline agents."""
    baselines = {
        "dreamerv3": DreamerV3Adapter,
        "tdmpc2": TDMPC2Adapter,
        "muzero": MuZeroAdapter,
        "ppo": PPOAdapter,
    }

    if name.lower() not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")

    return baselines[name.lower()](state_dim, action_dim, **kwargs)
