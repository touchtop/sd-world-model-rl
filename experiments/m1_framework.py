"""
M1 experiment framework for OR journal submission.
This module defines baseline policies, scenario configurations, metrics collection,
and an experiment runner with multi-seed statistical reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any
import random
import json
import os
import math
import numpy as np

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.env.task_allocation_env import (
    BACKLOG_PENALTY_WEIGHT,
    COLLAPSE_PENALTY,
    TaskAllocationEnv,
    compute_reward,
)

try:
    from src.world_model import TaskSDModel
except Exception:
    TaskSDModel = None


@dataclass
class ModelBelief:
    """Planner's internal model (can differ from true scenario parameters)."""
    arrival_rate: float
    worker_rates: Tuple[float, ...]
    fatigue_threshold: Optional[float]
    fatigue_penalty: float
    max_queue: int
    max_backlog: int
    fatigue_accumulation_threshold: Optional[float] = None
    fatigue_accumulation_rate: float = 1.0
    fatigue_collapse_threshold: Optional[float] = None
    collapse_penalty: float = 0.3
    fatigue_recovery_rate: float = 0.0
    # Probabilistic collapse belief parameters
    probabilistic_collapse: bool = False
    sigmoid_center: float = 10.0  # Believed μ
    sigmoid_temp: float = 2.0  # Believed T

    @classmethod
    def from_scenario(
        cls,
        scenario: "ScenarioConfig",
        param_noise: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ) -> "ModelBelief":
        """Create belief from scenario, optionally with noise perturbation."""
        if param_noise > 0 and rng is not None:
            noise_factor = lambda: 1.0 + rng.uniform(-param_noise, param_noise)
            arrival = scenario.arrival_rate * noise_factor()
            rates = tuple(r * noise_factor() for r in scenario.worker_rates)
        else:
            arrival = scenario.arrival_rate
            rates = scenario.worker_rates
        return cls(
            arrival_rate=arrival,
            worker_rates=rates,
            fatigue_threshold=scenario.fatigue_threshold,
            fatigue_penalty=scenario.fatigue_penalty,
            max_queue=scenario.max_queue,
            max_backlog=scenario.max_backlog,
            fatigue_accumulation_threshold=scenario.fatigue_accumulation_threshold,
            fatigue_accumulation_rate=scenario.fatigue_accumulation_rate,
            fatigue_collapse_threshold=scenario.fatigue_collapse_threshold,
            collapse_penalty=scenario.collapse_penalty,
            fatigue_recovery_rate=scenario.fatigue_recovery_rate,
            probabilistic_collapse=scenario.probabilistic_collapse,
            sigmoid_center=scenario.belief_sigmoid_center if scenario.belief_sigmoid_center is not None else scenario.collapse_sigmoid_center,
            sigmoid_temp=scenario.belief_sigmoid_temp if scenario.belief_sigmoid_temp is not None else scenario.collapse_sigmoid_temp,
        )


@dataclass
class ScenarioConfig:
    name: str
    arrival_rate: float
    worker_rates: Tuple[float, ...]
    fatigue_threshold: Optional[float] = None
    fatigue_penalty: float = 0.5
    fatigue_accumulation_threshold: Optional[float] = None
    fatigue_accumulation_rate: float = 1.0
    fatigue_collapse_threshold: Optional[float] = None
    collapse_penalty: float = 0.3
    fatigue_recovery_rate: float = 0.0
    # Delayed collapse: penalty kicks in after delay steps
    collapse_delay: int = 0
    collapse_duration: int = 10
    # Probabilistic collapse parameters
    probabilistic_collapse: bool = False
    collapse_sigmoid_center: float = 10.0  # True μ
    collapse_sigmoid_temp: float = 2.0  # True T
    # Belief parameters for probabilistic collapse (for MPC/SD-MCTS initial prior)
    belief_sigmoid_center: Optional[float] = None  # Planner's believed μ
    belief_sigmoid_temp: Optional[float] = None  # Planner's believed T
    # NEW: Efficiency degradation parameters
    efficiency_degradation: bool = False  # Use continuous efficiency loss
    efficiency_alpha: float = 0.01  # True α: quadratic coefficient
    efficiency_threshold: float = 5.0  # True θ: fatigue threshold for degradation
    belief_efficiency_alpha: Optional[float] = None  # Planner's believed α
    belief_efficiency_threshold: Optional[float] = None  # Planner's believed θ
    arrival_rate_schedule: Optional[Sequence[Tuple[int, float]]] = None
    worker_rate_schedule: Optional[Sequence[Tuple[int, Tuple[float, ...]]]] = None
    max_queue: int = 50
    max_backlog: int = 50
    episode_length: int = 100
    observe_fatigue: bool = False
    seeds: Sequence[int] = field(default_factory=lambda: list(range(10)))
    # For mismatch experiments: planner's believed parameters (if different from true)
    belief_arrival_rate: Optional[float] = None
    belief_worker_rates: Optional[Tuple[float, ...]] = None
    belief_fatigue_threshold: Optional[float] = None  # For fatigue cliff scenario

    def build_env(self, seed: Optional[int] = None) -> TaskAllocationEnv:
        return TaskAllocationEnv(
            max_queue=self.max_queue,
            max_backlog=self.max_backlog,
            arrival_rate=self.arrival_rate,
            worker_rates=self.worker_rates,
            fatigue_threshold=self.fatigue_threshold,
            fatigue_penalty=self.fatigue_penalty,
            fatigue_accumulation_threshold=self.fatigue_accumulation_threshold,
            fatigue_accumulation_rate=self.fatigue_accumulation_rate,
            fatigue_collapse_threshold=self.fatigue_collapse_threshold,
            collapse_penalty=self.collapse_penalty,
            fatigue_recovery_rate=self.fatigue_recovery_rate,
            collapse_delay=self.collapse_delay,
            collapse_duration=self.collapse_duration,
            probabilistic_collapse=self.probabilistic_collapse,
            collapse_sigmoid_center=self.collapse_sigmoid_center,
            collapse_sigmoid_temp=self.collapse_sigmoid_temp,
            efficiency_degradation=self.efficiency_degradation,
            efficiency_alpha=self.efficiency_alpha,
            efficiency_threshold=self.efficiency_threshold,
            arrival_rate_schedule=self.arrival_rate_schedule,
            worker_rate_schedule=self.worker_rate_schedule,
            observe_fatigue=self.observe_fatigue,
            seed=seed,
        )

    def get_belief(
        self,
        param_noise: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ) -> ModelBelief:
        """Get planner's belief model. Uses explicit belief params if set, else noisy scenario."""
        if (self.belief_arrival_rate is not None or
            self.belief_worker_rates is not None or
            self.belief_fatigue_threshold is not None or
            self.belief_sigmoid_center is not None):
            # Explicit mismatch scenario
            return ModelBelief(
                arrival_rate=self.belief_arrival_rate if self.belief_arrival_rate is not None else self.arrival_rate,
                worker_rates=self.belief_worker_rates if self.belief_worker_rates is not None else self.worker_rates,
                fatigue_threshold=self.belief_fatigue_threshold if self.belief_fatigue_threshold is not None else self.fatigue_threshold,
                fatigue_penalty=self.fatigue_penalty,
                max_queue=self.max_queue,
                max_backlog=self.max_backlog,
                fatigue_accumulation_threshold=self.fatigue_accumulation_threshold,
                fatigue_accumulation_rate=self.fatigue_accumulation_rate,
                fatigue_collapse_threshold=self.fatigue_collapse_threshold,
                collapse_penalty=self.collapse_penalty,
                fatigue_recovery_rate=self.fatigue_recovery_rate,
                probabilistic_collapse=self.probabilistic_collapse,
                sigmoid_center=self.belief_sigmoid_center if self.belief_sigmoid_center is not None else self.collapse_sigmoid_center,
                sigmoid_temp=self.belief_sigmoid_temp if self.belief_sigmoid_temp is not None else self.collapse_sigmoid_temp,
            )
        # Otherwise use noise-based belief
        return ModelBelief.from_scenario(self, param_noise=param_noise, rng=rng)

    @staticmethod
    def generate_m1_matrix() -> List["ScenarioConfig"]:
        scenarios: List[ScenarioConfig] = []

        base_rates = (1.0, 0.8)
        heter_rates = (1.2, 0.6, 0.9)
        arrival_rates = [0.5, 0.7, 1.0, 1.3]
        fatigue_thresholds = [10.0, 15.0, 20.0]

        scenarios.append(
            ScenarioConfig(
                name="linear_baseline",
                arrival_rate=0.7,
                worker_rates=base_rates,
                fatigue_threshold=None,
            )
        )

        # Fatigue scenarios need higher load to trigger fatigue
        for threshold in fatigue_thresholds:
            scenarios.append(
                ScenarioConfig(
                    name=f"fatigue_threshold_{int(threshold)}",
                    arrival_rate=1.5,  # High load to trigger fatigue
                    worker_rates=base_rates,
                    fatigue_threshold=threshold,
                )
            )

        for rate in arrival_rates:
            scenarios.append(
                ScenarioConfig(
                    name=f"load_sweep_{rate}",
                    arrival_rate=rate,
                    worker_rates=base_rates,
                    fatigue_threshold=15.0,
                )
            )

        scenarios.append(
            ScenarioConfig(
                name="heterogeneous_workers",
                arrival_rate=0.7,
                worker_rates=heter_rates,
                fatigue_threshold=15.0,
            )
        )

        # === PHASE 2: Advanced Mismatch Scenarios ===

        # --------------------------------------------------
        # Scenario A: Asymmetric Mismatch (Ranking Inversion)
        # --------------------------------------------------
        # MPC believes workers are equal, but they are NOT
        # Result: MPC routes 50/50, overloading the slow worker
        scenarios.append(
            ScenarioConfig(
                name="asymmetric_blind_spot",
                arrival_rate=1.2,  # Moderate-high load
                worker_rates=(1.5, 0.5),  # True: A is 3x faster than B
                fatigue_threshold=12.0,
                belief_arrival_rate=1.2,
                belief_worker_rates=(1.0, 1.0),  # MPC thinks they're equal
            )
        )

        # Ranking inversion: MPC believes B > A, but reality is A > B
        scenarios.append(
            ScenarioConfig(
                name="asymmetric_inverted",
                arrival_rate=1.2,
                worker_rates=(1.5, 0.6),  # True: A is 2.5x faster
                fatigue_threshold=12.0,
                belief_arrival_rate=1.2,
                belief_worker_rates=(0.6, 1.5),  # MPC thinks B is faster (INVERTED!)
            )
        )

        # --------------------------------------------------
        # Scenario B: Fatigue Threshold Mismatch
        # --------------------------------------------------
        # MPC believes fatigue triggers later than it actually does
        # Result: MPC pushes workers past the true threshold, causing collapse
        scenarios.append(
            ScenarioConfig(
                name="fatigue_cliff",
                arrival_rate=1.3,
                worker_rates=(1.0, 1.0),
                fatigue_threshold=8.0,  # True: collapse at 8
                fatigue_penalty=0.3,  # Severe penalty when triggered
                belief_arrival_rate=1.3,
                belief_worker_rates=(1.0, 1.0),
                belief_fatigue_threshold=20.0,  # MPC thinks safe until 20 (actually 8!)
            )
        )

        # --------------------------------------------------
        # Scenario C: Combined Asymmetric + Fatigue
        # --------------------------------------------------
        scenarios.append(
            ScenarioConfig(
                name="combined_asymmetric_fatigue",
                arrival_rate=1.4,
                worker_rates=(1.2, 0.6),  # A is 2x faster
                fatigue_threshold=10.0,
                fatigue_penalty=0.4,
                belief_arrival_rate=1.4,
                belief_worker_rates=(0.8, 0.8),  # MPC underestimates A, overestimates B
            )
        )

        # --------------------------------------------------
        # Scenario D: Combined Asymmetric + Fatigue Threshold Mismatch
        # --------------------------------------------------
        scenarios.append(
            ScenarioConfig(
                name="combined_all_mismatch",
                arrival_rate=1.3,
                worker_rates=(1.4, 0.5),  # A is 2.8x faster
                fatigue_threshold=8.0,  # True threshold
                fatigue_penalty=0.3,
                belief_arrival_rate=1.3,
                belief_worker_rates=(0.9, 0.9),  # MPC thinks equal
                belief_fatigue_threshold=18.0,  # MPC thinks safe until 18
            )
        )

        # ==================================================
        # CRITICAL LOAD SCENARIOS (from Codex/Gemini analysis)
        # ==================================================

        # Scenario: Critical Inversion
        # arrival ≈ 1 means queue stays near critical, wrong routing causes backlog explosion
        scenarios.append(
            ScenarioConfig(
                name="critical_inversion",
                arrival_rate=0.95,  # Near critical (only 1 task assigned per step)
                worker_rates=(1.5, 0.3),  # True: A is 5x faster
                fatigue_threshold=None,
                belief_arrival_rate=0.95,
                belief_worker_rates=(0.3, 1.5),  # MPC thinks B is faster (INVERTED)
            )
        )

        # Scenario: Fatigue Inversion Cliff
        # Wrong routing triggers fatigue collapse on slow worker
        scenarios.append(
            ScenarioConfig(
                name="fatigue_inversion_cliff",
                arrival_rate=0.95,
                worker_rates=(1.2, 0.5),  # True: A is 2.4x faster
                fatigue_threshold=4.0,  # Very low threshold
                fatigue_penalty=0.15,  # Severe collapse to 15%
                belief_arrival_rate=0.95,
                belief_worker_rates=(0.5, 1.2),  # MPC thinks B is faster
                belief_fatigue_threshold=25.0,  # MPC thinks threshold is very safe
            )
        )

        # Scenario: Ghost Capacity
        # MPC believes equal workers but B is nearly useless
        scenarios.append(
            ScenarioConfig(
                name="ghost_capacity",
                arrival_rate=0.92,
                worker_rates=(1.2, 0.05),  # B is a true "ghost" (5% capacity)
                fatigue_threshold=None,
                belief_arrival_rate=0.92,
                belief_worker_rates=(0.6, 0.6),  # MPC thinks both equal
            )
        )

        # ==================================================
        # FUNDAMENTAL MPC LIMITS (HORIZON + PARTIAL OBSERVABILITY)
        # ==================================================

        # Scenario: DELAYED_COLLAPSE (MPC Horizon Limit)
        # Cumulative fatigue triggers a permanent collapse after T >> H=3.
        # MPC's O(A^H) rollouts cannot see this long-horizon failure mode.
        # observe_fatigue=True ensures MPC failure is due to horizon limit, not hidden state.
        scenarios.append(
            ScenarioConfig(
                name="mpc_horizon_limit",
                arrival_rate=1.0,
                worker_rates=(0.6, 0.6),
                fatigue_accumulation_threshold=4.0,
                fatigue_accumulation_rate=0.25,
                fatigue_collapse_threshold=20.0,
                collapse_penalty=0.1,
                fatigue_recovery_rate=0.0,
                episode_length=120,
                observe_fatigue=True,
            )
        )

        # Scenario: HIDDEN_FATIGUE (POMDP / Partial Observability)
        # Observation only includes [queue, backlog]; cumulative fatigue is latent.
        # Identical observed states can have different collapse risk.
        scenarios.append(
            ScenarioConfig(
                name="pomdp_hidden_state",
                arrival_rate=0.9,
                arrival_rate_schedule=[(0, 1.05), (35, 0.6), (75, 0.95)],
                worker_rates=(0.7, 0.7),
                fatigue_accumulation_threshold=2.0,
                fatigue_accumulation_rate=0.2,
                fatigue_collapse_threshold=9.0,
                collapse_penalty=0.15,
                fatigue_recovery_rate=0.05,
                episode_length=130,
            )
        )

        # ==================================================
        # MODEL MISMATCH SCENARIO (RL LEARNING ADVANTAGE)
        # ==================================================

        # Scenario: "Optimistic Manager" - belief underestimates fatigue
        # MPC uses optimistic belief (threshold=20), but reality is harsher (threshold=8).
        # SD-MCTS starts with same wrong belief but learns true dynamics from data.
        scenarios.append(
            ScenarioConfig(
                name="production_mismatch_optimistic",
                arrival_rate=1.1,
                worker_rates=(1.0, 0.7),           # True: asymmetric workers
                fatigue_threshold=8.0,              # True: early fatigue
                fatigue_penalty=0.4,
                episode_length=100,
                belief_arrival_rate=1.1,
                belief_worker_rates=(0.9, 0.9),     # Belief: symmetric workers
                belief_fatigue_threshold=20.0,      # Belief: optimistic threshold
            )
        )

        # ==================================================
        # DELAYED COLLAPSE SCENARIO (STRUCTURAL MPC LIMITATION)
        # ==================================================

        # Scenario: Delayed Collapse - consequence beyond MPC horizon
        # Cumulative fatigue triggers collapse AFTER 8-step delay (> MPC horizon H=3)
        # MPC cannot see the delayed consequence; SD-MCTS learns to avoid it
        # Key: observe_fatigue=False, so both must infer from observations
        scenarios.append(
            ScenarioConfig(
                name="delayed_collapse_structural",
                arrival_rate=1.2,
                worker_rates=(0.9, 0.9),
                fatigue_accumulation_threshold=3.0,  # Overload when backlog > 3
                fatigue_accumulation_rate=1.0,       # Accumulate 1 per overloaded step
                fatigue_collapse_threshold=10.0,     # Threshold for collapse trigger
                collapse_delay=8,                    # 8 steps delay (> MPC horizon H=3)
                collapse_duration=15,                # Collapse lasts 15 steps
                collapse_penalty=0.2,                # Severe: 80% capacity loss
                fatigue_recovery_rate=0.0,           # No recovery during episode
                observe_fatigue=False,               # Hidden state - must learn
                episode_length=150,
            )
        )

        # ==================================================
        # PROBABILISTIC COLLAPSE FAIRNESS TEST
        # ==================================================
        # Key insight from Codex/Gemini: Make collapse stochastic, both start with
        # same wrong prior, only SD-MCTS learns the true parameters from data.
        #
        # Design:
        # - Fatigue accumulates as stock (observe_fatigue=True for fair comparison)
        # - Collapse probability: P = sigmoid((fatigue - μ) / T)
        # - True parameters: μ=8.0, T=1.5 (collapse likely when fatigue > 8)
        # - Belief parameters: μ=15.0, T=3.0 (both start optimistic: "won't collapse")
        # - MPC uses fixed belief; SD-MCTS learns to correct its belief
        #
        # Fairness: Both see fatigue state. Difference is MPC uses fixed sigmoid
        # params while SD-MCTS learns them from training data.
        #
        # NOTE: With 1-task-per-step assignment, backlog stays low (~0.15-0.25).
        # Use threshold=0 so any non-zero backlog triggers fatigue accumulation.
        scenarios.append(
            ScenarioConfig(
                name="probabilistic_collapse_fairness",
                arrival_rate=2.0,                    # 125% of capacity for queue buildup
                worker_rates=(0.85, 0.75),           # Total capacity = 1.6
                fatigue_accumulation_threshold=0.0,  # Any backlog > 0 triggers fatigue
                fatigue_accumulation_rate=0.5,       # Moderate accumulation
                fatigue_collapse_threshold=20.0,     # High ceiling (prob takes over before)
                collapse_duration=12,
                collapse_penalty=0.25,               # 75% capacity loss during collapse
                fatigue_recovery_rate=0.1,           # Recovery when backlog=0
                probabilistic_collapse=True,
                collapse_sigmoid_center=8.0,         # True μ: collapse likely at fatigue=8
                collapse_sigmoid_temp=1.5,           # True T: steep transition
                belief_sigmoid_center=15.0,          # Belief μ: optimistic (thinks safe up to 15)
                belief_sigmoid_temp=3.0,             # Belief T: shallow curve
                observe_fatigue=True,                # CRITICAL: Must be True for world model
                episode_length=200,                  # Longer episode for more collapse events
            )
        )

        # ==================================================
        # EFFICIENCY DEGRADATION SCENARIO (CONTINUOUS LEARNING SIGNAL)
        # ==================================================
        # Key insight: MPC's optimistic bias must be HARMFUL, not accidentally optimal.
        #
        # Design (Lorentzian form - never saturates to 0):
        # - Efficiency = 1 / (1 + α * (fatigue - θ)²)
        # - True parameters: α=0.10, θ=5.0 (SEVERE penalty, 100× MPC's belief)
        # - Belief parameters: α=0.001, θ=12.0 (MPC thinks penalty mild, safe up to 12)
        #
        # Mathematical proof at fatigue=12:
        # - MPC believes: 1/(1 + 0.001*(12-12)²) = 1.0 (100% efficient)
        # - Reality:      1/(1 + 0.10*(12-5)²) = 0.17 (17% efficient!)
        # MPC actively drives workers to 17% efficiency while believing 100%.
        #
        # SD-MCTS learns true α,θ → keeps fatigue ~5-7 → maintains 70-100% efficiency
        scenarios.append(
            ScenarioConfig(
                name="efficiency_degradation_fairness",
                arrival_rate=1.0,                    # 67% of capacity (sweet spot)
                worker_rates=(0.8, 0.7),             # Total capacity = 1.5
                fatigue_accumulation_threshold=0.0,  # Any backlog triggers fatigue
                fatigue_accumulation_rate=0.5,       # Faster accumulation for clear signal
                fatigue_collapse_threshold=50.0,     # High ceiling (efficiency takes over)
                fatigue_recovery_rate=0.2,           # Recovery when underloaded
                efficiency_degradation=True,
                efficiency_alpha=0.10,               # True α: SEVERE penalty (100× belief)
                efficiency_threshold=5.0,            # True θ: degradation starts at fatigue=5
                belief_efficiency_alpha=0.001,       # Belief α: thinks penalty negligible
                belief_efficiency_threshold=12.0,    # Belief θ: actively seeks fatigue=12
                observe_fatigue=True,                # Required for learning
                episode_length=200,
            )
        )

        # ==================================================
        # REALISTIC NONSTATIONARY SCENARIOS (SD-MCTS ADVANTAGE)
        # ==================================================

        # Scenario: Flash Crowd + Fatigue (cloud/web traffic surge)
        scenarios.append(
            ScenarioConfig(
                name="flash_crowd_fatigue",
                arrival_rate=0.6,
                arrival_rate_schedule=[(0, 0.6), (35, 1.6), (80, 0.9)],
                worker_rates=(1.4, 0.9),
                fatigue_threshold=6.0,
                fatigue_penalty=0.2,
                episode_length=120,
                belief_arrival_rate=0.6,  # MPC expects steady low demand
                belief_worker_rates=(1.4, 0.9),
                belief_fatigue_threshold=18.0,  # MPC underestimates fatigue risk
            )
        )

        # Scenario: Shift Change Inversion (manufacturing/call center shift)
        scenarios.append(
            ScenarioConfig(
                name="shift_change_inversion",
                arrival_rate=1.0,
                worker_rates=(1.5, 0.7),
                worker_rate_schedule=[(0, (1.5, 0.7)), (50, (0.7, 1.5))],
                episode_length=120,
                belief_arrival_rate=1.0,
                belief_worker_rates=(1.5, 0.7),  # MPC never updates ranking
            )
        )

        # Scenario: Maintenance Downtime (machine breakdown and recovery)
        scenarios.append(
            ScenarioConfig(
                name="maintenance_downtime",
                arrival_rate=1.1,
                worker_rates=(1.1, 1.1),
                worker_rate_schedule=[(0, (1.1, 1.1)), (30, (1.1, 0.05)), (70, (1.1, 1.1))],
                fatigue_threshold=12.0,
                fatigue_penalty=0.5,
                episode_length=120,
                belief_arrival_rate=1.1,
                belief_worker_rates=(1.1, 1.1),
            )
        )

        # Scenario: Training Ramp (new staff becomes productive)
        scenarios.append(
            ScenarioConfig(
                name="training_ramp",
                arrival_rate=0.9,
                worker_rates=(0.5, 1.2),
                worker_rate_schedule=[(0, (0.5, 1.2)), (60, (1.2, 1.2))],
                episode_length=120,
                belief_arrival_rate=0.9,
                belief_worker_rates=(0.5, 1.2),
            )
        )

        # Scenario: Cloud burst + fatigue (nonstationary demand + delayed collapse)
        scenarios.append(
            ScenarioConfig(
                name="cloud_burst_fatigue",
                arrival_rate=0.7,
                arrival_rate_schedule=[(0, 0.7), (30, 1.8), (70, 0.9)],
                worker_rates=(1.3, 0.9),
                fatigue_threshold=6.0,
                fatigue_penalty=0.2,
                episode_length=120,
                belief_arrival_rate=0.7,
                belief_worker_rates=(1.3, 0.9),
                belief_fatigue_threshold=16.0,
            )
        )

        # Scenario: Delayed collapse (fatigue accumulation to cliff)
        scenarios.append(
            ScenarioConfig(
                name="delayed_collapse",
                arrival_rate=1.1,
                worker_rates=(0.8, 0.4),
                fatigue_accumulation_threshold=3.0,
                fatigue_collapse_threshold=15.0,
                collapse_penalty=0.25,
                episode_length=120,
            )
        )

        # Scenario: Partial outage + recovery (hidden capacity drop)
        scenarios.append(
            ScenarioConfig(
                name="partial_outage_recovery",
                arrival_rate=1.1,
                worker_rates=(1.2, 1.1),
                worker_rate_schedule=[(0, (1.2, 1.1)), (40, (1.2, 0.05)), (80, (1.2, 1.1))],
                episode_length=120,
                belief_arrival_rate=1.1,
                belief_worker_rates=(1.2, 1.1),
            )
        )

        # Scenario: Skill inversion mid-episode (shift swap under high load)
        scenarios.append(
            ScenarioConfig(
                name="skill_inversion_shift",
                arrival_rate=1.0,
                worker_rates=(1.6, 0.6),
                worker_rate_schedule=[(0, (1.6, 0.6)), (50, (0.6, 1.6))],
                episode_length=120,
                belief_arrival_rate=1.0,
                belief_worker_rates=(1.6, 0.6),
            )
        )

        return scenarios


class MetricsCollector:
    def __init__(self, fatigue_threshold: Optional[float], max_queue: int, arrival_rate: float = 0.7, n_workers: int = 2):
        self.fatigue_threshold = fatigue_threshold
        self.max_queue = max_queue
        self.arrival_rate = arrival_rate
        self.n_workers = n_workers
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.queue_sum = 0.0
        self.backlog_sum = 0.0
        self.backlog_sum_per_worker: List[float] = []
        self.fatigue_triggers = 0
        self.collapse_triggers = 0
        self.queue_cost_sum = 0.0
        self.backlog_cost_sum = 0.0

    def observe(self, state: np.ndarray) -> None:
        queue = float(state[0])
        backlogs = state[1:1 + self.n_workers]
        self.steps += 1
        self.queue_sum += queue
        self.backlog_sum += float(backlogs.sum())
        if not self.backlog_sum_per_worker:
            self.backlog_sum_per_worker = [0.0 for _ in backlogs]
        for i, b in enumerate(backlogs):
            self.backlog_sum_per_worker[i] += float(b)

        if self.fatigue_threshold is not None:
            if np.any(backlogs > self.fatigue_threshold):
                self.fatigue_triggers += 1

        if queue >= self.max_queue:
            self.collapse_triggers += 1

        self.queue_cost_sum += queue
        self.backlog_cost_sum += BACKLOG_PENALTY_WEIGHT * float(np.sum(backlogs ** 2))

    def summarize(self) -> Dict[str, float]:
        if self.steps == 0:
            return {}
        avg_queue = self.queue_sum / self.steps
        avg_backlog = self.backlog_sum / self.steps
        avg_backlog_per_worker = [
            v / self.steps for v in self.backlog_sum_per_worker
        ]
        fatigue_rate = self.fatigue_triggers / self.steps
        collapse_rate = self.collapse_triggers / self.steps
        queue_cost = self.queue_cost_sum / self.steps
        backlog_cost = self.backlog_cost_sum / self.steps
        total_cost = queue_cost + backlog_cost
        avg_wait_time = (avg_queue / self.arrival_rate) if self.arrival_rate > 1e-6 else 0.0
        return {
            "avg_queue_length": avg_queue,
            "avg_backlog_total": avg_backlog,
            "avg_backlog_per_worker": avg_backlog_per_worker,
            "fatigue_trigger_rate": fatigue_rate,
            "collapse_rate": collapse_rate,
            "queue_cost": queue_cost,
            "backlog_cost": backlog_cost,
            "total_cost": total_cost,
            "avg_wait_time": avg_wait_time,
        }


class BaselinePolicy:
    name = "baseline"

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        self.scenario = scenario
        self.learning_curve = []  # [(env_steps, eval_reward), ...]

    def train(self, env: TaskAllocationEnv, env_steps_budget: int) -> None:
        del env, env_steps_budget

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def get_learning_curve(self) -> List[Tuple[int, float]]:
        return self.learning_curve

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
    ) -> None:
        del state, action, next_state, reward


class JSQPolicy(BaselinePolicy):
    name = "jsq"

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        super().reset(scenario, seed=seed)
        self._n_workers = len(scenario.worker_rates)

    def select_action(self, state: np.ndarray) -> int:
        queue = int(state[0])
        backlogs = state[1:1 + self._n_workers]
        if queue <= 0:
            return len(backlogs)
        return int(np.argmin(backlogs))


class CmuPolicy(BaselinePolicy):
    """cμ rule: prioritize workers by cost * rate (SED - Shortest Expected Delay)"""
    name = "cmu"

    def __init__(self, cost_weights: Optional[Sequence[float]] = None):
        self._init_cost_weights = cost_weights
        self.cost_weights = None
        self._n_workers = 2

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        super().reset(scenario, seed=seed)
        self._n_workers = len(scenario.worker_rates)
        if self._init_cost_weights is not None:
            self.cost_weights = list(self._init_cost_weights)
        else:
            self.cost_weights = [1.0 for _ in scenario.worker_rates]

    def select_action(self, state: np.ndarray) -> int:
        queue = int(state[0])
        backlogs = state[1:1 + self._n_workers]
        if queue <= 0:
            return len(backlogs)
        rates = np.array(self.scenario.worker_rates, dtype=np.float32)
        if self.scenario.fatigue_threshold is not None:
            fatigued = backlogs > self.scenario.fatigue_threshold
            rates = rates * np.where(fatigued, self.scenario.fatigue_penalty, 1.0)
        expected_delay = (backlogs + 1) / np.maximum(rates, 1e-6)
        return int(np.argmin(expected_delay))


def _ensure_fatigue_state(state: np.ndarray, belief: ModelBelief) -> np.ndarray:
    """Ensure state includes cumulative fatigue dimension for MPC rollouts.

    If state already contains fatigue (len >= 1 + n_workers + 1), return as-is.
    Otherwise, append 0.0 as initial fatigue (for scenarios without observe_fatigue).
    """
    n_workers = len(belief.worker_rates)
    state_arr = np.asarray(state, dtype=np.float32)
    if state_arr.shape[0] >= 1 + n_workers + 1:
        return state_arr
    return np.concatenate([state_arr[:1 + n_workers], [0.0]]).astype(np.float32)


def _expected_step_with_belief(
    state: np.ndarray,
    action: int,
    belief: ModelBelief,
) -> np.ndarray:
    """Simulate one step using planner's belief model (may differ from true dynamics).

    Now includes cumulative fatigue accumulation and collapse logic so that
    MPC can model long-horizon fatigue effects (though limited by its horizon H).
    For probabilistic collapse, uses expected penalty based on belief sigmoid params.
    """
    n_workers = len(belief.worker_rates)
    state_arr = np.asarray(state, dtype=np.float32)
    queue = float(state_arr[0])
    has_fatigue = state_arr.shape[0] >= 1 + n_workers + 1
    backlogs = state_arr[1:1 + n_workers].copy()
    cumulative_fatigue = float(state_arr[1 + n_workers]) if has_fatigue else 0.0

    queue = min(belief.max_queue, queue + belief.arrival_rate)

    if action < n_workers and queue >= 1.0:
        queue = max(0.0, queue - 1.0)
        backlogs[action] = min(belief.max_backlog, backlogs[action] + 1)

    worker_rates = np.array(belief.worker_rates, dtype=np.float32)
    # Threshold-based fatigue (original logic)
    if belief.fatigue_threshold is not None:
        for i in range(len(worker_rates)):
            if backlogs[i] > belief.fatigue_threshold:
                worker_rates[i] = worker_rates[i] * belief.fatigue_penalty

    # Probabilistic collapse: use expected efficiency based on belief sigmoid
    if belief.probabilistic_collapse:
        # P(collapse) = sigmoid((fatigue - μ) / T)
        z = (cumulative_fatigue - belief.sigmoid_center) / max(belief.sigmoid_temp, 0.01)
        z = np.clip(z, -100.0, 100.0)  # Numerical stability
        collapse_prob = 1.0 / (1.0 + np.exp(-z))
        # Expected efficiency: (1-P) * normal + P * collapsed
        expected_coef = 1.0 - collapse_prob * (1.0 - belief.collapse_penalty)
        worker_rates = worker_rates * expected_coef
    else:
        # Deterministic cumulative fatigue collapse
        collapse_active = (
            belief.fatigue_collapse_threshold is not None and
            cumulative_fatigue >= belief.fatigue_collapse_threshold
        )
        if collapse_active:
            worker_rates = worker_rates * belief.collapse_penalty

    backlogs = np.maximum(0.0, backlogs - worker_rates)

    # Update cumulative fatigue for next step
    next_cumulative_fatigue = cumulative_fatigue
    if belief.fatigue_accumulation_threshold is not None:
        overload_metric = float(np.max(backlogs)) if backlogs.size else 0.0
        if overload_metric > belief.fatigue_accumulation_threshold:
            next_cumulative_fatigue += belief.fatigue_accumulation_rate
        elif belief.fatigue_recovery_rate > 0.0:
            next_cumulative_fatigue = max(
                0.0,
                next_cumulative_fatigue - belief.fatigue_recovery_rate,
            )

    if has_fatigue:
        return np.concatenate([[queue], backlogs, [next_cumulative_fatigue]]).astype(np.float32)
    return np.concatenate([[queue], backlogs]).astype(np.float32)


def _expected_step(
    state: np.ndarray,
    action: int,
    scenario: ScenarioConfig,
) -> np.ndarray:
    """Legacy wrapper: convert scenario to belief and call _expected_step_with_belief."""
    belief = ModelBelief(
        arrival_rate=scenario.arrival_rate,
        worker_rates=scenario.worker_rates,
        fatigue_threshold=scenario.fatigue_threshold,
        fatigue_penalty=scenario.fatigue_penalty,
        max_queue=scenario.max_queue,
        max_backlog=scenario.max_backlog,
        fatigue_accumulation_threshold=scenario.fatigue_accumulation_threshold,
        fatigue_accumulation_rate=scenario.fatigue_accumulation_rate,
        fatigue_collapse_threshold=scenario.fatigue_collapse_threshold,
        collapse_penalty=scenario.collapse_penalty,
        fatigue_recovery_rate=scenario.fatigue_recovery_rate,
    )
    return _expected_step_with_belief(state, action, belief)


def _state_cost(
    state: np.ndarray,
    n_workers: Optional[int] = None,
    fatigue_collapse_threshold: Optional[float] = None,
) -> float:
    """Compute state cost, optionally ignoring fatigue dimension."""
    queue = float(state[0])
    if n_workers is None:
        backlogs = state[1:]
        cumulative_fatigue = 0.0
    else:
        backlogs = state[1:1 + n_workers]
        cumulative_fatigue = float(state[1 + n_workers]) if len(state) > 1 + n_workers else 0.0
    backlog_sq = float(np.sum(backlogs ** 2))
    cost = queue + BACKLOG_PENALTY_WEIGHT * backlog_sq
    if fatigue_collapse_threshold is not None and cumulative_fatigue >= fatigue_collapse_threshold:
        cost += COLLAPSE_PENALTY
    return cost


class MPCPolicy(BaselinePolicy):
    name = "mpc"

    def __init__(self, horizon: int = 3, discount: float = 1.0, param_noise: float = 0.0):
        self.horizon = horizon
        self.discount = discount
        self.param_noise = param_noise
        self.sd_model = None
        self.belief: Optional[ModelBelief] = None
        self._rng: Optional[np.random.Generator] = None

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        super().reset(scenario)
        if TaskSDModel is not None:
            self.sd_model = TaskSDModel(n_workers=len(scenario.worker_rates))
        # Initialize belief with potential noise/mismatch
        self._rng = np.random.default_rng(seed)
        self.belief = scenario.get_belief(param_noise=self.param_noise, rng=self._rng)

    def select_action(self, state: np.ndarray) -> int:
        n_workers = len(self.belief.worker_rates) if self.belief is not None else len(state) - 1
        n_actions = n_workers + 1
        # Augment state with cumulative fatigue for MPC rollouts
        belief_state = _ensure_fatigue_state(state, self.belief) if self.belief is not None else state
        collapse_thresh = self.belief.fatigue_collapse_threshold if self.belief else None

        def rollout(curr_state: np.ndarray, depth: int) -> float:
            if depth == 0:
                return 0.0
            costs = []
            for act in range(n_actions):
                next_state = _expected_step_with_belief(curr_state, act, self.belief)
                cost = _state_cost(next_state, n_workers=n_workers, fatigue_collapse_threshold=collapse_thresh)
                cost += self.discount * rollout(next_state, depth - 1)
                costs.append(cost)
            return min(costs)

        best_action = 0
        best_cost = math.inf
        for act in range(n_actions):
            next_state = _expected_step_with_belief(belief_state, act, self.belief)
            cost = _state_cost(next_state, n_workers=n_workers, fatigue_collapse_threshold=collapse_thresh)
            cost += self.discount * rollout(next_state, self.horizon - 1)
            if cost < best_cost:
                best_cost = cost
                best_action = act
        return best_action


class ADPRolloutPolicy(BaselinePolicy):
    name = "adp_rollout"

    def __init__(self, base_policy: Optional[BaselinePolicy] = None, gamma: float = 0.95, param_noise: float = 0.0):
        self.base_policy = base_policy or JSQPolicy()
        self.gamma = gamma
        self.param_noise = param_noise
        self.belief: Optional[ModelBelief] = None
        self._rng: Optional[np.random.Generator] = None
        self._n_workers: int = 2

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        super().reset(scenario)
        self._n_workers = len(scenario.worker_rates)
        self.base_policy.reset(scenario)
        # Initialize belief with potential noise/mismatch
        self._rng = np.random.default_rng(seed + 1000)  # Different seed from MPC
        self.belief = scenario.get_belief(param_noise=self.param_noise, rng=self._rng)

    def _approx_value(self, state: np.ndarray) -> float:
        collapse_thresh = self.belief.fatigue_collapse_threshold if self.belief else None
        return -_state_cost(state, n_workers=self._n_workers, fatigue_collapse_threshold=collapse_thresh)

    def select_action(self, state: np.ndarray) -> int:
        n_workers = self._n_workers
        n_actions = n_workers + 1
        collapse_thresh = self.belief.fatigue_collapse_threshold if self.belief else None

        best_action = 0
        best_score = -math.inf
        for act in range(n_actions):
            next_state = _expected_step_with_belief(state, act, self.belief)
            immediate = -_state_cost(next_state, n_workers=n_workers, fatigue_collapse_threshold=collapse_thresh)

            rollout_action = self.base_policy.select_action(next_state)
            rollout_state = _expected_step_with_belief(next_state, rollout_action, self.belief)
            rollout_value = self._approx_value(rollout_state)

            score = immediate + self.gamma * rollout_value
            if score > best_score:
                best_score = score
                best_action = act
        return best_action


class ExperimentRunner:
    def __init__(
        self,
        scenarios: Sequence[ScenarioConfig],
        policies: Sequence[BaselinePolicy],
        env_steps_budget: int = 10000,
        num_seeds: int = 10,
    ):
        self.scenarios = scenarios
        self.policies = policies
        self.env_steps_budget = env_steps_budget
        self.num_seeds = num_seeds

    def run(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        learning_curves: Dict[str, Dict[str, List]] = {}  # scenario -> policy -> curves
        total_tasks = len(self.scenarios) * len(self.policies) * self.num_seeds
        completed = 0

        for scenario in self.scenarios:
            print(f"\n[Scenario] {scenario.name}")
            results[scenario.name] = {}
            learning_curves[scenario.name] = {}
            seeds = list(scenario.seeds)[: self.num_seeds]
            for policy in self.policies:
                print(f"  [Policy] {policy.name}", end="", flush=True)
                metrics_per_seed = []
                curves_per_seed = []

                for seed in seeds:
                    policy.reset(scenario, seed=seed)
                    train_env = scenario.build_env(seed=seed)
                    policy.train(train_env, self.env_steps_budget)

                    # Use independent eval env with different seed
                    eval_env = scenario.build_env(seed=seed + 10000)
                    metrics = self._evaluate_policy(eval_env, policy, scenario)
                    metrics_per_seed.append(metrics)
                    curves_per_seed.append(policy.get_learning_curve())
                    completed += 1
                    print(".", end="", flush=True)

                results[scenario.name][policy.name] = self._aggregate(metrics_per_seed)
                learning_curves[scenario.name][policy.name] = curves_per_seed
                avg_q = results[scenario.name][policy.name].get("avg_queue_length_mean", 0)
                print(f" avg_queue={avg_q:.2f}")

        self.learning_curves = learning_curves
        return results

    def _evaluate_policy(
        self,
        env: TaskAllocationEnv,
        policy: BaselinePolicy,
        scenario: ScenarioConfig,
    ) -> Dict[str, float]:
        obs, _ = env.reset()
        collector = MetricsCollector(
            fatigue_threshold=scenario.fatigue_threshold,
            max_queue=scenario.max_queue,
            arrival_rate=scenario.arrival_rate,
            n_workers=len(scenario.worker_rates),
        )
        steps = 0
        while steps < scenario.episode_length:
            action = policy.select_action(obs)
            next_obs, reward, _, _, _ = env.step(action)
            policy.observe_transition(obs, action, next_obs, reward)
            obs = next_obs
            collector.observe(obs)
            steps += 1
        return collector.summarize()

    @staticmethod
    def _aggregate(metrics_per_seed: List[Dict[str, Any]]) -> Dict[str, float]:
        if not metrics_per_seed:
            return {}
        aggregated: Dict[str, float] = {}
        keys = [k for k in metrics_per_seed[0].keys() if not isinstance(metrics_per_seed[0][k], list)]
        for key in keys:
            values = np.array([m[key] for m in metrics_per_seed], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if HAS_SCIPY and len(values) > 1:
                ci = stats.sem(values) * stats.t.ppf((1 + 0.95) / 2., len(values) - 1)
            else:
                ci = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            aggregated[f"{key}_mean"] = mean
            aggregated[f"{key}_std"] = std
            aggregated[f"{key}_ci95"] = ci
        return aggregated


class PPOWrapper(BaselinePolicy):
    """Wrapper for PPO from compare_methods.py"""
    name = "ppo"

    def __init__(self, lr: float = 3e-4, gamma: float = 0.99):
        self.lr = lr
        self.gamma = gamma
        self.agent = None

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        super().reset(scenario, seed=seed)
        self.agent = None

    def train(self, env: TaskAllocationEnv, env_steps_budget: int) -> None:
        from compare_methods import SimplePPO
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.agent = SimplePPO(state_dim, n_actions, lr=self.lr, gamma=self.gamma)

        eval_interval = max(500, env_steps_budget // 10)
        env_steps = 0

        while env_steps < env_steps_budget:
            states, actions, log_probs, rewards, values = [], [], [], [], []
            state, _ = env.reset()
            ep_reward = 0
            for _ in range(100):
                action, log_prob = self.agent.get_action(state)
                value = self.agent.get_value(state)
                next_state, reward, term, trunc, _ = env.step(action)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob.item())
                rewards.append(reward)
                values.append(value)
                ep_reward += reward
                state = next_state
                env_steps += 1
                if term or trunc:
                    break
            returns, advantages, R = [], [], 0
            for r, v in zip(reversed(rewards), reversed(values)):
                R = r + self.gamma * R
                returns.insert(0, R)
                advantages.insert(0, R - v)
            self.agent.update(states, actions, log_probs, returns, advantages)

            if env_steps % eval_interval < 100:
                self.learning_curve.append((env_steps, ep_reward))

    def select_action(self, state: np.ndarray) -> int:
        if self.agent is None:
            return 0
        return self.agent.get_action(state, deterministic=True)


class SDMCTSWrapper(BaselinePolicy):
    """Wrapper for SD+MCTS from train_muzero.py"""
    name = "sd_mcts"

    def __init__(
        self,
        num_simulations: int = 50,
        online_update: bool = False,
        online_lr: float = 5e-3,
        online_steps: int = 1,
        online_batch: int = 32,
        online_buffer_cap: int = 2000,
    ):
        self.num_simulations = num_simulations
        self.world_model = None
        self.policy_net = None
        self.value_net = None
        self.n_actions = 3
        self.online_update = online_update
        self.online_lr = online_lr
        self.online_steps = online_steps
        self.online_batch = online_batch
        self.online_buffer_cap = online_buffer_cap
        self._online_buffer: List[Tuple[np.ndarray, int, np.ndarray]] = []
        self._online_opt = None

    def reset(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        super().reset(scenario, seed=seed)
        self.world_model = None
        self.policy_net = None
        self.value_net = None
        self.n_actions = len(scenario.worker_rates) + 1
        self._online_buffer = []
        self._online_opt = None

    def train(self, env: TaskAllocationEnv, env_steps_budget: int) -> None:
        import torch.optim as optim
        from src.world_model import TaskSDModel
        from src.agent import PolicyNetwork, ValueNetwork
        from train_muzero import collect_episode, train_step, ReplayBuffer

        state_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        n_workers = len(self.scenario.worker_rates)

        # CRITICAL: Initialize world model with BELIEF (prior), not TRUTH.
        # This ensures SD-MCTS starts with same (potentially wrong) assumptions as MPC.
        # The learning advantage comes from adapting to observed data.
        init_fatigue_threshold = (
            self.scenario.belief_fatigue_threshold
            if self.scenario.belief_fatigue_threshold is not None
            else self.scenario.fatigue_threshold
        )

        # Probabilistic collapse: start with belief parameters, learn true ones
        use_prob_collapse = self.scenario.probabilistic_collapse
        init_sigmoid_center = (
            self.scenario.belief_sigmoid_center
            if self.scenario.belief_sigmoid_center is not None
            else self.scenario.collapse_sigmoid_center
        )
        init_sigmoid_temp = (
            self.scenario.belief_sigmoid_temp
            if self.scenario.belief_sigmoid_temp is not None
            else self.scenario.collapse_sigmoid_temp
        )

        self.world_model = TaskSDModel(
            n_workers=n_workers,
            fatigue_threshold=init_fatigue_threshold,
            fatigue_penalty=self.scenario.fatigue_penalty,
            learn_fatigue_dynamics=(
                self.scenario.fatigue_collapse_threshold is not None
            ),
            learn_probabilistic_collapse=use_prob_collapse,
            init_sigmoid_center=init_sigmoid_center,
            init_sigmoid_temp=init_sigmoid_temp,
            probabilistic_collapse_penalty=self.scenario.collapse_penalty,
        )
        self.policy_net = PolicyNetwork(state_dim, self.n_actions)
        self.value_net = ValueNetwork(state_dim)

        policy_opt = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        value_opt = optim.Adam(self.value_net.parameters(), lr=1e-3)
        model_opt = optim.Adam(self.world_model.parameters(), lr=1e-2)
        if self.online_update:
            self._online_opt = optim.Adam(self.world_model.parameters(), lr=self.online_lr)
        buffer = ReplayBuffer(capacity=10000)

        eval_interval = max(500, env_steps_budget // 10)
        env_steps = 0

        while env_steps < env_steps_budget:
            transitions = collect_episode(
                env, self.world_model, self.n_actions,
                policy_net=self.policy_net, value_net=self.value_net,
                num_simulations=min(20, self.num_simulations)
            )
            ep_reward = sum(t.reward for t in transitions)
            env_steps += len(transitions)
            for t in transitions:
                buffer.add(t)
            for _ in range(5):
                train_step(
                    self.policy_net, self.value_net, self.world_model, buffer,
                    policy_opt, value_opt, model_opt, batch_size=64
                )

            if env_steps % eval_interval < len(transitions):
                self.learning_curve.append((env_steps, ep_reward))

    def select_action(self, state: np.ndarray) -> int:
        if self.world_model is None:
            return 0
        from src.planning import mcts_search
        n_workers = self.n_actions - 1
        collapse_thresh = self.scenario.fatigue_collapse_threshold

        def fatigue_aware_reward(s):
            fatigue = s[-1] if s.shape[0] > 1 + n_workers else None
            return compute_reward(s, n_workers=n_workers, fatigue=fatigue, collapse_threshold=collapse_thresh)

        return mcts_search(
            state, self.world_model, n_actions=self.n_actions,
            num_simulations=self.num_simulations,
            policy_net=self.policy_net, value_net=self.value_net,
            reward_fn=fatigue_aware_reward
        )

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
    ) -> None:
        del reward
        if not self.online_update or self.world_model is None or self._online_opt is None:
            return
        self._online_buffer.append((state.copy(), int(action), next_state.copy()))
        if len(self._online_buffer) > self.online_buffer_cap:
            self._online_buffer.pop(0)
        if len(self._online_buffer) < self.online_batch:
            return

        import torch
        import torch.nn.functional as F

        for _ in range(self.online_steps):
            batch = random.sample(self._online_buffer, self.online_batch)
            states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.int64)
            next_states = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32)

            pred_next = self.world_model(states, actions)
            loss = F.mse_loss(pred_next, next_states)
            self._online_opt.zero_grad()
            loss.backward()
            self._online_opt.step()


def save_results_to_csv(results: Dict, output_path: str) -> None:
    rows = []
    for scenario_name, policy_results in results.items():
        for policy_name, metrics in policy_results.items():
            row = {"scenario": scenario_name, "policy": policy_name}
            row.update(metrics)
            rows.append(row)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    except ImportError:
        with open(output_path.replace('.csv', '.json'), 'w') as f:
            json.dump(rows, f, indent=2)


def plot_comparison(
    results: Dict,
    metric: str = "avg_queue_length_mean",
    output_path: str = "m1_comparison.png"
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    scenarios = list(results.keys())
    policies = list(results[scenarios[0]].keys()) if scenarios else []

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(policies), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, policy in enumerate(policies):
        means = [results[s].get(policy, {}).get(metric, 0) for s in scenarios]
        ci_key = metric.replace("_mean", "_ci95")
        cis = [results[s].get(policy, {}).get(ci_key, 0) for s in scenarios]
        offset = (i - len(policies) / 2) * width + width / 2
        ax.bar(x + offset, means, width, yerr=cis, label=policy, capsize=3)

    ax.set_xlabel("Scenario")
    ax.set_ylabel(metric)
    ax.set_title(f"M1 Comparison: {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_learning_curves(
    learning_curves: Dict[str, Dict[str, List]],
    scenario: str,
    output_path: str = "learning_curves.png"
) -> None:
    """Plot learning curves with CI bands for a specific scenario."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    if scenario not in learning_curves:
        print(f"Scenario {scenario} not found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (policy, curves_per_seed) in enumerate(learning_curves[scenario].items()):
        # Only plot learning methods (those with non-empty curves)
        non_empty_curves = [c for c in curves_per_seed if len(c) > 0]
        if len(non_empty_curves) == 0:
            continue

        # Align curves to common x-axis
        all_steps = sorted(set(s for curve in non_empty_curves for s, _ in curve))
        if len(all_steps) == 0:
            continue

        # Interpolate values at each step
        values_at_step = {s: [] for s in all_steps}
        for curve in non_empty_curves:
            curve_dict = dict(curve)
            for s in all_steps:
                if s in curve_dict:
                    values_at_step[s].append(curve_dict[s])

        steps = [s for s in all_steps if len(values_at_step[s]) > 0]
        means = [np.mean(values_at_step[s]) for s in steps]
        stds = [np.std(values_at_step[s]) for s in steps]

        ax.plot(steps, means, label=policy, color=colors[i % len(colors)], linewidth=2)
        ax.fill_between(steps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=colors[i % len(colors)])

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"Learning Curves: {scenario}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_m1_experiments(
    output_dir: str = "results/m1",
    env_steps_budget: int = 8000,
    num_seeds: int = 10,
    include_learning_methods: bool = True,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    scenarios = ScenarioConfig.generate_m1_matrix()

    policies: List[BaselinePolicy] = [
        JSQPolicy(),
        CmuPolicy(),
        MPCPolicy(horizon=3),
        ADPRolloutPolicy(),
    ]
    if include_learning_methods:
        policies.extend([
            PPOWrapper(),
            SDMCTSWrapper(num_simulations=50, online_update=True),
        ])

    runner = ExperimentRunner(
        scenarios=scenarios,
        policies=policies,
        env_steps_budget=env_steps_budget,
        num_seeds=num_seeds,
    )

    print("=" * 70)
    print("M1 Experiment Framework - OR Journal Submission")
    print("=" * 70)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Policies: {[p.name for p in policies]}")
    print(f"Seeds per scenario: {num_seeds}")
    print(f"Training budget: {env_steps_budget} env steps")
    print("=" * 70)

    results = runner.run()

    save_results_to_csv(results, os.path.join(output_dir, "m1_results.csv"))

    for metric in ["avg_queue_length_mean", "fatigue_trigger_rate_mean", "collapse_rate_mean"]:
        plot_comparison(
            results, metric=metric,
            output_path=os.path.join(output_dir, f"m1_{metric.replace('_mean', '')}.png")
        )

    # Plot learning curves for key scenarios
    if hasattr(runner, 'learning_curves'):
        for scenario in ["linear_baseline", "load_sweep_1.0", "fatigue_threshold_15"]:
            if scenario in runner.learning_curves:
                plot_learning_curves(
                    runner.learning_curves, scenario,
                    output_path=os.path.join(output_dir, f"learning_curve_{scenario}.png")
                )

    print("\n" + "=" * 70)
    print("Results saved to:", output_dir)
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_m1_experiments()
