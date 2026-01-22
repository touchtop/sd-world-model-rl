"""Delayed Collapse + Demand Surge Environment.

Mathematical design for SD-MCTS structural advantage:
- Latent fatigue f_t accumulates silently (no immediate penalty)
- Hard collapse at f_t >= F (absorbing state)
- Demand surge at T_s with bonus B if f_t <= F_req

MPC fails structurally because:
1. Cannot model latent fatigue (not in its state representation)
2. Horizon H < T_s means it cannot see the surge event
3. Greedy a_t=1 maximizes immediate reward but causes collapse before surge

SD-MCTS wins by:
1. Learning fatigue dynamics through world model
2. Value network propagates surge bonus backward in time
3. Discovers conservative policy: a_t = a_safe < 1 before surge
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


class DelayedCollapseEnv(gym.Env):
    """Environment with delayed collapse and demand surge.

    State: (latent_fatigue, alive_flag, surge_flag)
    Action: work_intensity in [0, 1] (discretized to n_actions levels)

    Dynamics:
    - f_{t+1} = f_t + (α + β) * a_t - β  (fatigue accumulation/recovery)
    - Collapse when f_t >= F (irreversible, absorbing state)
    - Surge at t = T_s: bonus B if f_t <= F_req, else penalty P

    Reward:
    - Normal step: r_t = a_t (immediate throughput)
    - Surge step: r_t = a_t + B (if f_t <= F_req) or a_t - P
    - Collapsed: r_t = 0
    """

    def __init__(
        self,
        alpha: float = 1.0,           # Fatigue accumulation rate
        beta: float = 0.1,            # Recovery rate (when resting)
        F: float = 10.0,              # Hard collapse threshold
        F_req: float = 5.0,           # Fatigue requirement for surge bonus
        T_s: int = 12,                # Surge time step
        B: float = 15.0,              # Surge bonus
        P: float = 0.0,               # Surge penalty (if not meeting F_req)
        T: int = 30,                  # Episode length
        n_actions: int = 11,          # Discretize [0,1] into n_actions levels
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.F_req = F_req
        self.T_s = T_s
        self.B = B
        self.P = P
        self.T = T
        self.n_actions = n_actions
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # State: [fatigue, alive, surge_indicator, time_to_surge]
        # time_to_surge helps the agent anticipate the surge
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([F * 2, 1.0, 1.0, float(T_s)], dtype=np.float32),
            dtype=np.float32
        )

        # Action: discrete work intensity levels
        self.action_space = spaces.Discrete(n_actions)

        self.fatigue = 0.0
        self.alive = True
        self.t = 0

    def _get_action_intensity(self, action: int) -> float:
        """Convert discrete action to continuous intensity [0, 1]."""
        return action / (self.n_actions - 1)

    def _get_obs(self) -> np.ndarray:
        surge_flag = 1.0 if self.t == self.T_s else 0.0
        time_to_surge = max(0.0, self.T_s - self.t)
        return np.array([
            min(self.fatigue, self.F * 2),  # Clamp for observation space
            1.0 if self.alive else 0.0,
            surge_flag,
            time_to_surge
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.fatigue = 0.0
        self.alive = True
        self.t = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.alive:
            # Absorbing state: no reward, no state change
            self.t += 1
            terminated = self.t >= self.T
            return self._get_obs(), 0.0, terminated, False, {"collapsed": True}

        a_t = self._get_action_intensity(action)

        # Fatigue dynamics
        self.fatigue = self.fatigue + (self.alpha + self.beta) * a_t - self.beta
        self.fatigue = max(0.0, self.fatigue)  # Cannot go negative

        # Check collapse
        if self.fatigue >= self.F:
            self.alive = False
            self.t += 1
            terminated = self.t >= self.T
            return self._get_obs(), 0.0, terminated, False, {"collapsed": True, "collapse_step": self.t}

        # Compute reward
        reward = a_t  # Base reward: work intensity

        # Surge event
        is_surge = (self.t == self.T_s - 1)  # Check before incrementing t
        if is_surge:
            if self.fatigue <= self.F_req:
                reward += self.B
            else:
                reward -= self.P

        self.t += 1
        terminated = self.t >= self.T

        info = {
            "fatigue": self.fatigue,
            "alive": self.alive,
            "is_surge": is_surge,
            "surge_bonus_earned": is_surge and self.fatigue <= self.F_req,
        }

        return self._get_obs(), reward, terminated, False, info

    def get_optimal_intensity(self) -> float:
        """Compute the optimal constant intensity to meet surge requirement.

        Constraint: f_{T_s} = (α + β) * Σa_t - β * T_s <= F_req
        For constant a: (α + β) * a * T_s - β * T_s <= F_req
        => a <= (F_req + β * T_s) / ((α + β) * T_s)
        """
        a_safe = (self.F_req + self.beta * self.T_s) / ((self.alpha + self.beta) * self.T_s)
        return min(1.0, a_safe)


class DelayedCollapseWorldModel:
    """Simplified world model for delayed collapse environment.

    This is what MPC would use (with wrong beliefs).
    """

    def __init__(
        self,
        belief_alpha: float = 1.0,
        belief_beta: float = 0.1,
        belief_F: float = 50.0,      # MPC thinks collapse threshold is much higher
        belief_F_req: float = 5.0,
        belief_T_s: int = 12,
        belief_B: float = 15.0,
        n_actions: int = 11,
    ):
        self.alpha = belief_alpha
        self.beta = belief_beta
        self.F = belief_F
        self.F_req = belief_F_req
        self.T_s = belief_T_s
        self.B = belief_B
        self.n_actions = n_actions

    def predict_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict next state given current state and action."""
        fatigue, alive, surge_flag, time_to_surge = state

        if alive < 0.5:
            # Already collapsed
            return np.array([fatigue, 0.0, 0.0, max(0, time_to_surge - 1)], dtype=np.float32)

        a_t = action / (self.n_actions - 1)

        # Fatigue dynamics (using belief parameters)
        new_fatigue = fatigue + (self.alpha + self.beta) * a_t - self.beta
        new_fatigue = max(0.0, new_fatigue)

        # Check collapse (using belief threshold - MPC thinks it's safe)
        new_alive = 1.0 if new_fatigue < self.F else 0.0

        # Update time to surge
        new_time_to_surge = max(0.0, time_to_surge - 1)
        new_surge_flag = 1.0 if new_time_to_surge == 0 else 0.0

        return np.array([new_fatigue, new_alive, new_surge_flag, new_time_to_surge], dtype=np.float32)

    def predict_reward(self, state: np.ndarray, action: int) -> float:
        """Predict reward (using belief model)."""
        fatigue, alive, surge_flag, time_to_surge = state

        if alive < 0.5:
            return 0.0

        a_t = action / (self.n_actions - 1)
        reward = a_t

        # MPC doesn't properly account for surge (short horizon)
        # In 1-step lookahead, it won't see the surge bonus

        return reward
