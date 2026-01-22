import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Sequence, Tuple

BACKLOG_PENALTY_WEIGHT = 0.1  # 降低权重以平衡二次项的数值范围
COLLAPSE_PENALTY = 50.0  # 累积疲劳崩溃惩罚


def compute_reward(
    state: np.ndarray,
    n_workers: Optional[int] = None,
    fatigue: Optional[float] = None,
    collapse_threshold: Optional[float] = None,
) -> float:
    """统一奖励函数：负的队列长度和积压量，加上崩溃惩罚"""
    queue = state[0]
    if n_workers is not None:
        backlogs = state[1:1 + n_workers]
    else:
        backlogs = state[1:]
    base_cost = queue + BACKLOG_PENALTY_WEIGHT * np.sum(backlogs ** 2)
    if fatigue is not None and collapse_threshold is not None:
        if fatigue >= collapse_threshold:
            base_cost += COLLAPSE_PENALTY
    return -base_cost


class TaskAllocationEnv(gym.Env):
    """动态任务分配环境：任务到达队列，分配给多个Worker处理"""

    def __init__(
        self,
        max_queue: int = 50,
        max_backlog: int = 50,
        arrival_rate: float = 0.7,
        worker_rates: tuple = (1.0, 0.8),
        fatigue_threshold: float = None,  # 新增：疲劳阈值
        fatigue_penalty: float = 0.5,     # 新增：疲劳时效率降低比例
        fatigue_accumulation_threshold: Optional[float] = None,
        fatigue_accumulation_rate: float = 1.0,
        fatigue_collapse_threshold: Optional[float] = None,
        collapse_penalty: float = 0.3,
        fatigue_recovery_rate: float = 0.0,
        # NEW: Delayed collapse mechanism
        collapse_delay: int = 0,  # Steps between threshold breach and actual collapse
        collapse_duration: int = 10,  # How long collapse lasts
        # NEW: Probabilistic collapse mechanism
        probabilistic_collapse: bool = False,  # Use stochastic collapse instead of deterministic
        collapse_sigmoid_center: float = 10.0,  # μ: center of sigmoid (true fatigue threshold)
        collapse_sigmoid_temp: float = 2.0,  # T: temperature (steepness of probability curve)
        # NEW: Efficiency degradation mechanism (continuous learning signal)
        efficiency_degradation: bool = False,  # Use continuous efficiency loss instead of binary collapse
        efficiency_alpha: float = 0.01,  # Quadratic coefficient: efficiency = 1 - α*(fatigue - θ)²
        efficiency_threshold: float = 5.0,  # θ: fatigue threshold where degradation starts
        arrival_rate_schedule: Optional[Sequence[Tuple[int, float]]] = None,
        worker_rate_schedule: Optional[Sequence[Tuple[int, Tuple[float, ...]]]] = None,
        observe_fatigue: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self.max_queue = max_queue
        self.max_backlog = max_backlog
        self._base_arrival_rate = arrival_rate
        self.arrival_rate = arrival_rate
        self._base_worker_rates = np.array(worker_rates, dtype=np.float32)
        self.base_worker_rates = self._base_worker_rates.copy()
        self.worker_rates = self.base_worker_rates.copy()
        self.n_workers = len(worker_rates)
        self.fatigue_threshold = fatigue_threshold
        self.fatigue_penalty = fatigue_penalty
        self.fatigue_accumulation_threshold = fatigue_accumulation_threshold
        self.fatigue_accumulation_rate = fatigue_accumulation_rate
        self.fatigue_collapse_threshold = fatigue_collapse_threshold
        self.collapse_penalty = collapse_penalty
        self.fatigue_recovery_rate = fatigue_recovery_rate
        self.cumulative_fatigue = 0.0
        self._collapse_triggered = False
        # Delayed collapse state
        self.collapse_delay = collapse_delay
        self.collapse_duration = collapse_duration
        self._collapse_timer = 0  # Countdown to collapse (hidden from observation)
        self._collapse_remaining = 0  # How many steps collapse lasts
        # Probabilistic collapse parameters
        self.probabilistic_collapse = probabilistic_collapse
        self.collapse_sigmoid_center = collapse_sigmoid_center
        self.collapse_sigmoid_temp = collapse_sigmoid_temp
        # NEW: Efficiency degradation mechanism (continuous alternative to binary collapse)
        self.efficiency_degradation = efficiency_degradation
        self.efficiency_alpha = efficiency_alpha
        self.efficiency_threshold = efficiency_threshold
        self.observe_fatigue = observe_fatigue
        self.arrival_rate_schedule = (
            sorted(list(arrival_rate_schedule), key=lambda x: x[0])
            if arrival_rate_schedule else None
        )
        self.worker_rate_schedule = (
            sorted(list(worker_rate_schedule), key=lambda x: x[0])
            if worker_rate_schedule else None
        )
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._step_count = 0

        if self.worker_rate_schedule is not None:
            for _, rates in self.worker_rate_schedule:
                if len(rates) != self.n_workers:
                    raise ValueError("worker_rate_schedule length must match n_workers")

        # State: [queue_length, worker_1_load, worker_2_load, ..., (cumulative_fatigue)]
        obs_dim = 1 + self.n_workers + (1 if self.observe_fatigue else 0)
        low = np.zeros(obs_dim, dtype=np.float32)
        fatigue_max = float(self.fatigue_collapse_threshold or 100.0)
        high_list = [float(max_queue)] + [float(max_backlog)] * self.n_workers
        if self.observe_fatigue:
            high_list.append(fatigue_max)
        high = np.array(high_list, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: 0~n_workers-1 -> assign to worker_i, n_workers -> hold
        self.action_space = spaces.Discrete(self.n_workers + 1)
        self.state = None

    def _apply_schedules(self) -> None:
        if self.arrival_rate_schedule:
            rate = self._base_arrival_rate
            for step, value in self.arrival_rate_schedule:
                if self._step_count >= step:
                    rate = value
                else:
                    break
            self.arrival_rate = rate
        else:
            self.arrival_rate = self._base_arrival_rate

        if self.worker_rate_schedule:
            rates = self._base_worker_rates
            for step, value in self.worker_rate_schedule:
                if self._step_count >= step:
                    rates = np.array(value, dtype=np.float32)
                else:
                    break
            self.base_worker_rates = rates.copy()
        else:
            self.base_worker_rates = self._base_worker_rates.copy()

    def _get_obs(self) -> np.ndarray:
        """Build observation, optionally including cumulative fatigue."""
        if self.observe_fatigue:
            # Clamp fatigue to observation space bounds
            fatigue_max = float(self.fatigue_collapse_threshold or 100.0)
            clamped_fatigue = min(self.cumulative_fatigue, fatigue_max)
            return np.concatenate([self.state, [clamped_fatigue]]).astype(np.float32)
        return self.state.copy()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self.arrival_rate = self._base_arrival_rate
        self.base_worker_rates = self._base_worker_rates.copy()
        self.worker_rates = self.base_worker_rates.copy()
        self.cumulative_fatigue = 0.0
        self._collapse_triggered = False
        self._collapse_timer = 0
        self._collapse_remaining = 0
        self._apply_schedules()
        self.state = np.zeros(1 + self.n_workers, dtype=np.float32)
        self.worker_rates = self.base_worker_rates.copy()
        return self._get_obs(), {}

    def _update_cumulative_fatigue(self, backlogs: np.ndarray) -> None:
        if self.fatigue_accumulation_threshold is None or self.fatigue_collapse_threshold is None:
            return
        overload_metric = float(np.max(backlogs)) if backlogs.size else 0.0
        if overload_metric > self.fatigue_accumulation_threshold:
            self.cumulative_fatigue += self.fatigue_accumulation_rate
        elif self.fatigue_recovery_rate > 0.0:
            self.cumulative_fatigue = max(0.0, self.cumulative_fatigue - self.fatigue_recovery_rate)

        # Probabilistic collapse logic (stochastic)
        if self.probabilistic_collapse:
            if not self._collapse_triggered:
                # P(collapse) = sigmoid((fatigue - μ) / T)
                z = (self.cumulative_fatigue - self.collapse_sigmoid_center) / max(self.collapse_sigmoid_temp, 0.01)
                z = np.clip(z, -100.0, 100.0)  # Numerical stability
                prob_collapse = 1.0 / (1.0 + np.exp(-z))
                if self.rng.random() < prob_collapse:
                    self._collapse_triggered = True
                    self._collapse_remaining = self.collapse_duration
            return

        # Delayed collapse logic (deterministic)
        if self.collapse_delay > 0:
            # With delay: start countdown when threshold breached
            if not self._collapse_triggered and self.cumulative_fatigue >= self.fatigue_collapse_threshold:
                if self._collapse_timer == 0:
                    self._collapse_timer = self.collapse_delay  # Start countdown
            # Tick down the timer
            if self._collapse_timer > 0:
                self._collapse_timer -= 1
                if self._collapse_timer == 0:
                    # Timer expired -> trigger collapse
                    self._collapse_triggered = True
                    self._collapse_remaining = self.collapse_duration
        else:
            # Original immediate collapse logic
            if not self._collapse_triggered and self.cumulative_fatigue >= self.fatigue_collapse_threshold:
                self._collapse_triggered = True
                self._collapse_remaining = self.collapse_duration

    def step(self, action: int):
        self._apply_schedules()
        queue = self.state[0]
        backlogs = self.state[1:].copy()

        # 任务到达
        arrivals = self.rng.poisson(self.arrival_rate)
        queue = min(self.max_queue, queue + arrivals)

        # 分配任务
        if action < self.n_workers and queue > 0:
            queue -= 1
            backlogs[action] = min(self.max_backlog, backlogs[action] + 1)

        # Worker疲劳机制：积压超过阈值时效率下降
        effective_rates = self.base_worker_rates.copy()
        if self.fatigue_threshold is not None:
            for i in range(self.n_workers):
                if backlogs[i] > self.fatigue_threshold:
                    effective_rates[i] = effective_rates[i] * self.fatigue_penalty

        # Collapse effect (with optional duration limit)
        if self._collapse_triggered:
            if self._collapse_remaining > 0:
                effective_rates = effective_rates * self.collapse_penalty
                self._collapse_remaining -= 1
                if self._collapse_remaining == 0:
                    # Collapse ends, reset for potential future collapse
                    self._collapse_triggered = False
                    self.cumulative_fatigue = 0.0
            elif self.collapse_duration == 0:
                # Permanent collapse (original behavior)
                effective_rates = effective_rates * self.collapse_penalty

        # NEW: Efficiency degradation (continuous alternative to binary collapse)
        # One-sided Lorentzian: efficiency = 1 / (1 + α * max(0, fatigue - θ)²)
        # Only degrades when fatigue > threshold, full efficiency below threshold
        efficiency_coef = 1.0
        if self.efficiency_degradation:
            diff = max(0.0, self.cumulative_fatigue - self.efficiency_threshold)
            efficiency_coef = 1.0 / (1.0 + self.efficiency_alpha * diff * diff)
            effective_rates = effective_rates * efficiency_coef

        self.worker_rates = effective_rates

        # Worker处理任务
        backlogs = np.maximum(0.0, backlogs - effective_rates)

        self.state = np.concatenate([[queue], backlogs]).astype(np.float32)
        # 累积疲劳在当前状态后更新，坍塌效应从下一步开始生效
        self._update_cumulative_fatigue(backlogs)
        reward = compute_reward(
            self.state,
            n_workers=len(self.worker_rates),
            fatigue=self.cumulative_fatigue,
            collapse_threshold=self.fatigue_collapse_threshold,
        )
        self._step_count += 1
        info = {
            "cumulative_fatigue": float(self.cumulative_fatigue),
            "collapse_triggered": self._collapse_triggered,
            "efficiency_coef": float(efficiency_coef),
        }
        return self._get_obs(), reward, False, False, info


class TaskAllocationEnvWithFatigue(TaskAllocationEnv):
    """带疲劳机制的任务分配环境（便捷包装）"""

    def __init__(
        self,
        max_queue: int = 50,
        max_backlog: int = 50,
        arrival_rate: float = 0.7,
        worker_rates: tuple = (1.0, 0.8),
        fatigue_threshold: float = 15.0,  # 默认阈值15
        fatigue_penalty: float = 0.5,      # 效率降至50%
        seed: int = None,
    ):
        super().__init__(
            max_queue=max_queue,
            max_backlog=max_backlog,
            arrival_rate=arrival_rate,
            worker_rates=worker_rates,
            fatigue_threshold=fatigue_threshold,
            fatigue_penalty=fatigue_penalty,
            seed=seed,
        )
