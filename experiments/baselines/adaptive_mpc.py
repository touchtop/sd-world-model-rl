"""
Adaptive MPC Baseline.

Adaptive MPC starts with wrong beliefs and learns parameters online
using RLS (Recursive Least Squares). This provides a fair comparison
with SD-MCTS, as both methods learn from experience.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from .rls_estimator import EfficiencyDegradationEstimator, DelayedCollapseEstimator


class AdaptiveMPC:
    """MPC with online parameter learning via RLS.

    Unlike Oracle MPC (which uses true parameters) or fixed-belief MPC
    (which never updates), Adaptive MPC learns parameters from observations.
    This provides a fair baseline for comparing learning methods.

    Attributes:
        n_workers: Number of workers.
        horizon: Planning horizon.
        estimator: Parameter estimator (RLS-based).
    """

    def __init__(
        self,
        n_workers: int,
        arrival_rate: float,
        worker_rates: Tuple[float, ...],
        horizon: int = 3,
        gamma: float = 0.99,
        scenario_type: str = "efficiency_degradation",
        # Environment constraints
        max_queue: int = 50,
        max_backlog: int = 50,
        # Initial beliefs (can be wrong)
        init_alpha: float = 0.001,
        init_theta: float = 12.0,
        # For delayed collapse
        init_F: float = 50.0,
        init_fatigue_alpha: float = 1.0,
        init_fatigue_beta: float = 0.1,
        # RLS settings
        forgetting_factor: float = 0.99,
        # Fatigue dynamics (initial beliefs, can be wrong)
        init_fatigue_accum_threshold: float = 3.0,
        init_fatigue_accum_rate: float = 0.2,
        init_fatigue_recovery_rate: float = 0.05,
    ):
        self.n_workers = n_workers
        self.arrival_rate = arrival_rate
        self.worker_rates = np.array(worker_rates, dtype=np.float32)
        self.horizon = horizon
        self.gamma = gamma
        self.scenario_type = scenario_type

        # Environment constraints
        self.max_queue = max_queue
        self.max_backlog = max_backlog

        # Fatigue dynamics beliefs (can be learned/updated)
        self.fatigue_accum_threshold = init_fatigue_accum_threshold
        self.fatigue_accum_rate = init_fatigue_accum_rate
        self.fatigue_recovery_rate = init_fatigue_recovery_rate

        # Initialize appropriate estimator
        if scenario_type == "efficiency_degradation":
            self.estimator = EfficiencyDegradationEstimator(
                init_alpha=init_alpha,
                init_theta=init_theta,
                forgetting_factor=forgetting_factor,
            )
        elif scenario_type == "delayed_collapse":
            self.estimator = DelayedCollapseEstimator(
                init_alpha=init_fatigue_alpha,
                init_beta=init_fatigue_beta,
                init_F=init_F,
                forgetting_factor=forgetting_factor,
            )
        else:
            self.estimator = None

        # Track last state for parameter updates
        self._last_state = None
        self._last_action = None

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray, info: Dict[str, Any]):
        """Update parameter estimates from transition.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.
            info: Additional info from environment.
        """
        if self.scenario_type == "efficiency_degradation":
            self._update_efficiency_degradation(state, info)
        elif self.scenario_type == "delayed_collapse":
            self._update_delayed_collapse(state, action, next_state, info)

    def _update_efficiency_degradation(self, state: np.ndarray, info: Dict[str, Any]):
        """Update efficiency degradation parameters."""
        if len(state) <= 1 + self.n_workers:
            return  # No fatigue in state

        fatigue = state[-1]
        efficiency = info.get("efficiency_coef", 1.0)

        if efficiency < 1.0:  # Only update when degradation is observable
            self.estimator.update(fatigue, efficiency)

    def _update_delayed_collapse(
        self, state: np.ndarray, action: int, next_state: np.ndarray, info: Dict[str, Any]
    ):
        """Update delayed collapse parameters."""
        fatigue_before = state[0]
        fatigue_after = next_state[0]
        collapsed = info.get("collapse_triggered", False)

        self.estimator.update(action, fatigue_before, fatigue_after, collapsed)

    def select_action(self, state: np.ndarray, current_step: int = 0) -> int:
        """Select action using MPC with current parameter estimates.

        Args:
            state: Current state.
            current_step: Current timestep.

        Returns:
            Best action.
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
        """H-step rollout with current parameter estimates."""
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
        """Greedy action for rollout continuation."""
        best_action = self.n_workers
        best_value = float('-inf')

        for action in range(self.n_workers + 1):
            _, reward = self._simulate_step(state.copy(), action, current_step)
            if reward > best_value:
                best_value = reward
                best_action = action

        return best_action

    def _simulate_step(
        self, state: np.ndarray, action: int, current_step: int
    ) -> Tuple[np.ndarray, float]:
        """Simulate one step with LEARNED parameters."""
        if self.scenario_type == "efficiency_degradation":
            return self._simulate_efficiency_degradation(state, action)
        elif self.scenario_type == "delayed_collapse":
            return self._simulate_delayed_collapse(state, action, current_step)
        else:
            return self._simulate_basic(state, action)

    def _simulate_basic(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """Basic simulation."""
        queue = state[0]
        backlogs = state[1:1 + self.n_workers].copy()

        # Apply queue constraint
        queue = min(self.max_queue, queue + self.arrival_rate)
        if action < self.n_workers and queue >= 1.0:
            queue = max(0.0, queue - 1.0)
            backlogs[action] = min(self.max_backlog, backlogs[action] + 1)

        backlogs = np.maximum(0, backlogs - self.worker_rates)
        cost = queue + 0.1 * np.sum(backlogs ** 2)

        next_state = np.concatenate([[queue], backlogs])
        return next_state.astype(np.float32), -cost

    def _simulate_efficiency_degradation(
        self, state: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """Simulate with LEARNED efficiency parameters."""
        queue = state[0]
        backlogs = state[1:1 + self.n_workers].copy()
        fatigue = state[-1] if len(state) > 1 + self.n_workers else 0.0

        # Apply queue constraint
        queue = min(self.max_queue, queue + self.arrival_rate)
        if action < self.n_workers and queue >= 1.0:
            queue = max(0.0, queue - 1.0)
            backlogs[action] = min(self.max_backlog, backlogs[action] + 1)

        # Use learned α and θ
        alpha_hat, theta_hat = self.estimator.get_params()
        diff = max(0.0, fatigue - theta_hat)
        efficiency = 1.0 / (1.0 + alpha_hat * diff * diff)
        effective_rates = self.worker_rates * efficiency

        backlogs = np.maximum(0, backlogs - effective_rates)

        # Fatigue update using current beliefs
        max_backlog = np.max(backlogs) if backlogs.size > 0 else 0.0
        if max_backlog > self.fatigue_accum_threshold:
            fatigue += self.fatigue_accum_rate
        else:
            fatigue = max(0.0, fatigue - self.fatigue_recovery_rate)

        cost = queue + 0.1 * np.sum(backlogs ** 2)
        next_state = np.concatenate([[queue], backlogs, [fatigue]])
        return next_state.astype(np.float32), -cost

    def _simulate_delayed_collapse(
        self, state: np.ndarray, action: int, current_step: int
    ) -> Tuple[np.ndarray, float]:
        """Simulate delayed collapse with LEARNED parameters."""
        fatigue = state[0]
        alpha_hat, beta_hat, F_hat = self.estimator.get_params()

        # Dynamics with learned α, β
        if action == 1:
            new_fatigue = fatigue + alpha_hat
        else:
            new_fatigue = max(0.0, fatigue - beta_hat)

        # Check collapse with learned F
        collapsed = new_fatigue >= F_hat

        if collapsed:
            reward = 0.0
        else:
            reward = float(action)

        next_state = np.array([new_fatigue], dtype=np.float32)
        return next_state, reward

    def get_learned_params(self) -> Dict[str, float]:
        """Get current learned parameter estimates."""
        if self.scenario_type == "efficiency_degradation":
            alpha, theta = self.estimator.get_params()
            return {"alpha": alpha, "theta": theta}
        elif self.scenario_type == "delayed_collapse":
            alpha, beta, F = self.estimator.get_params()
            return {"alpha": alpha, "beta": beta, "F": F}
        return {}


def run_adaptive_mpc_episode(
    env,
    adaptive_mpc: AdaptiveMPC,
    seed: Optional[int] = None,
    episode_length: int = 200,
) -> Tuple[float, List[float], List[Dict[str, Any]], Dict[str, float]]:
    """Run one episode with Adaptive MPC.

    Args:
        env: Environment instance.
        adaptive_mpc: AdaptiveMPC instance.
        seed: Random seed.
        episode_length: Maximum episode length.

    Returns:
        (total_reward, rewards_list, info_list, final_params) tuple.
    """
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    infos = []

    for step in range(episode_length):
        action = adaptive_mpc.select_action(state, current_step=step)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update parameter estimates
        adaptive_mpc.update(state, action, next_state, info)

        total_reward += reward
        rewards.append(reward)
        infos.append(info)
        state = next_state

        if terminated or truncated:
            break

    final_params = adaptive_mpc.get_learned_params()
    return total_reward, rewards, infos, final_params
