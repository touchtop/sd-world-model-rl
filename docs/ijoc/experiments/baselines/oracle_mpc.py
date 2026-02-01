"""
Oracle MPC Baseline.

Oracle MPC uses TRUE environment parameters (not beliefs) for planning.
This serves as a theoretical upper bound on MPC performance.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class OracleMPC:
    """MPC with perfect knowledge of environment parameters.

    This baseline demonstrates the best possible MPC performance when
    all parameters are known exactly. It serves as a theoretical upper bound
    for comparison with learning-based methods.

    Attributes:
        n_workers: Number of workers in the environment.
        horizon: Planning horizon (number of steps to look ahead).
        gamma: Discount factor for future rewards.
        true_params: Dictionary of true environment parameters.
    """

    def __init__(
        self,
        n_workers: int,
        arrival_rate: float,
        worker_rates: Tuple[float, ...],
        horizon: int = 15,
        gamma: float = 0.99,
        # Environment constraints
        max_queue: int = 50,
        max_backlog: int = 50,
        # Efficiency degradation parameters (true values)
        efficiency_alpha: Optional[float] = None,
        efficiency_threshold: Optional[float] = None,
        # Fatigue dynamics parameters (true values)
        fatigue_accum_threshold: float = 0.0,
        fatigue_accum_rate: float = 0.5,
        fatigue_recovery_rate: float = 0.2,
        # Delayed collapse parameters (true values)
        collapse_threshold: Optional[float] = None,
        fatigue_alpha: Optional[float] = None,
        fatigue_beta: Optional[float] = None,
        # Peak reward parameters
        surge_time: Optional[int] = None,
        surge_fatigue_req: Optional[float] = None,
        surge_bonus: Optional[float] = None,
    ):
        self.n_workers = n_workers
        self.arrival_rate = arrival_rate
        self.worker_rates = np.array(worker_rates, dtype=np.float32)
        self.horizon = horizon
        self.gamma = gamma

        # Environment constraints
        self.max_queue = max_queue
        self.max_backlog = max_backlog

        # Store true parameters
        self.efficiency_alpha = efficiency_alpha
        self.efficiency_threshold = efficiency_threshold
        self.collapse_threshold = collapse_threshold
        self.fatigue_alpha = fatigue_alpha
        self.fatigue_beta = fatigue_beta
        self.surge_time = surge_time
        self.surge_fatigue_req = surge_fatigue_req
        self.surge_bonus = surge_bonus

        # Fatigue dynamics
        self.fatigue_accum_threshold = fatigue_accum_threshold
        self.fatigue_accum_rate = fatigue_accum_rate
        self.fatigue_recovery_rate = fatigue_recovery_rate

        # Determine scenario type
        self.scenario_type = self._detect_scenario_type()

    def _detect_scenario_type(self) -> str:
        """Detect which scenario we're in based on parameters."""
        if self.efficiency_alpha is not None:
            return "efficiency_degradation"
        elif self.collapse_threshold is not None:
            return "delayed_collapse"
        else:
            return "basic"

    def select_action(self, state: np.ndarray, current_step: int = 0) -> int:
        """Select best action using H-step lookahead with true parameters.

        Args:
            state: Current environment state.
            current_step: Current timestep (for surge timing).

        Returns:
            Best action index.
        """
        best_action = self.n_workers  # Default: hold
        best_value = float('-inf')

        for action in range(self.n_workers + 1):
            value = self._rollout(state, action, current_step)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _rollout(self, state: np.ndarray, first_action: int, current_step: int) -> float:
        """Simulate H-step rollout with true parameters.

        Args:
            state: Starting state.
            first_action: Action to take at first step.
            current_step: Current timestep in episode.

        Returns:
            Discounted cumulative reward.
        """
        total_reward = 0.0
        sim_state = state.copy()
        sim_step = current_step

        for t in range(self.horizon):
            action = first_action if t == 0 else self._greedy_action(sim_state, sim_step)
            sim_state, reward = self._simulate_step(sim_state, action, sim_step)
            total_reward += reward * (self.gamma ** t)
            sim_step += 1

        return total_reward

    def _greedy_action(self, state: np.ndarray, current_step: int) -> int:
        """Select greedy action for rollout continuation."""
        best_action = self.n_workers
        best_value = float('-inf')

        for action in range(self.n_workers + 1):
            next_state, reward = self._simulate_step(state.copy(), action, current_step)
            if reward > best_value:
                best_value = reward
                best_action = action

        return best_action

    def _simulate_step(
        self, state: np.ndarray, action: int, current_step: int
    ) -> Tuple[np.ndarray, float]:
        """Simulate one step using TRUE parameters.

        Args:
            state: Current state [queue, backlog_0, ..., backlog_n, (fatigue)].
            action: Action to take.
            current_step: Current timestep.

        Returns:
            (next_state, reward) tuple.
        """
        if self.scenario_type == "efficiency_degradation":
            return self._simulate_efficiency_degradation(state, action)
        elif self.scenario_type == "delayed_collapse":
            return self._simulate_delayed_collapse(state, action, current_step)
        else:
            return self._simulate_basic(state, action)

    def _simulate_basic(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """Basic simulation without fatigue effects."""
        queue = state[0]
        backlogs = state[1:1 + self.n_workers].copy()

        # Arrivals (expected value) with queue constraint
        queue = min(self.max_queue, queue + self.arrival_rate)

        # Assignment
        if action < self.n_workers and queue >= 1.0:
            queue = max(0.0, queue - 1.0)
            backlogs[action] = min(self.max_backlog, backlogs[action] + 1)

        # Completion
        backlogs = np.maximum(0, backlogs - self.worker_rates)

        # Reward (negative cost)
        cost = queue + 0.1 * np.sum(backlogs ** 2)
        reward = -cost

        next_state = np.concatenate([[queue], backlogs])
        return next_state.astype(np.float32), reward

    def _simulate_efficiency_degradation(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """Simulate with efficiency degradation (one-sided Lorentzian)."""
        queue = state[0]
        backlogs = state[1:1 + self.n_workers].copy()
        fatigue = state[-1] if len(state) > 1 + self.n_workers else 0.0

        # Arrivals with queue constraint
        queue = min(self.max_queue, queue + self.arrival_rate)

        # Assignment with backlog constraint
        if action < self.n_workers and queue >= 1.0:
            queue = max(0.0, queue - 1.0)
            backlogs[action] = min(self.max_backlog, backlogs[action] + 1)

        # Compute efficiency using TRUE α and θ
        # One-sided Lorentzian: efficiency = 1 / (1 + α * max(0, fatigue - θ)²)
        diff = max(0.0, fatigue - self.efficiency_threshold)
        efficiency = 1.0 / (1.0 + self.efficiency_alpha * diff * diff)
        effective_rates = self.worker_rates * efficiency

        # Completion
        backlogs = np.maximum(0, backlogs - effective_rates)

        # Fatigue update using TRUE parameters
        max_backlog = np.max(backlogs) if backlogs.size > 0 else 0.0
        if max_backlog > self.fatigue_accum_threshold:
            fatigue += self.fatigue_accum_rate
        else:
            fatigue = max(0.0, fatigue - self.fatigue_recovery_rate)

        # Reward (negative cost)
        cost = queue + 0.1 * np.sum(backlogs ** 2)
        reward = -cost

        next_state = np.concatenate([[queue], backlogs, [fatigue]])
        return next_state.astype(np.float32), reward

    def _simulate_delayed_collapse(
        self, state: np.ndarray, action: int, current_step: int
    ) -> Tuple[np.ndarray, float]:
        """Simulate delayed collapse scenario."""
        fatigue = state[0]
        alpha = self.fatigue_alpha
        beta = self.fatigue_beta

        # Action effect on fatigue
        # f_{t+1} = f_t + (α + β) * a - β
        if action == 1:  # work
            new_fatigue = fatigue + alpha
        else:  # rest
            new_fatigue = max(0.0, fatigue - beta)

        # Check collapse (using TRUE threshold)
        collapsed = new_fatigue >= self.collapse_threshold

        # Base reward
        if collapsed:
            reward = 0.0  # System collapsed
        else:
            reward = float(action)  # Reward for working

        # Surge bonus (if applicable)
        if self.surge_time is not None and current_step == self.surge_time:
            if not collapsed and fatigue <= self.surge_fatigue_req:
                reward += self.surge_bonus

        next_state = np.array([new_fatigue], dtype=np.float32)
        return next_state, reward


def run_oracle_mpc_episode(
    env,
    oracle: OracleMPC,
    seed: Optional[int] = None,
    episode_length: int = 200,
    record_trajectory: bool = False,
) -> Tuple[float, List[float], Dict]:
    """Run one episode with Oracle MPC.

    Args:
        env: Environment instance.
        oracle: OracleMPC instance with true parameters.
        seed: Random seed for environment reset.
        episode_length: Maximum episode length.
        record_trajectory: Whether to record state trajectory.

    Returns:
        (total_reward, rewards_list, trajectory_dict) tuple.
    """
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    trajectory = {"queue": [], "backlog": [], "fatigue": []} if record_trajectory else None
    n_workers = oracle.n_workers

    for step in range(episode_length):
        if record_trajectory:
            trajectory["queue"].append(float(state[0]))
            trajectory["backlog"].append(float(np.sum(state[1:1+n_workers])))
            trajectory["fatigue"].append(float(state[-1]) if len(state) > 1 + n_workers else 0.0)

        action = oracle.select_action(state, current_step=step)
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rewards, trajectory or {}
