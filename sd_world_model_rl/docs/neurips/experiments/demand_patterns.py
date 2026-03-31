"""
Demand Pattern Generators for Zero-Shot Transfer Experiments.

Provides diverse demand patterns to test generalization beyond training distribution.
"""
import numpy as np
from typing import Callable


class DemandPattern:
    """Base class for demand pattern generators."""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self):
        """Reset internal state."""
        self.t = 0

    def get_demand(self) -> int:
        """Get demand for current timestep."""
        raise NotImplementedError


class StepDemand(DemandPattern):
    """Classic Beer Game step demand (training distribution)."""

    def __init__(self, low: int = 4, high: int = 8, step_time: int = 5,
                 noise_std: float = 0.0, seed: int = None):
        super().__init__(seed)
        self.low = low
        self.high = high
        self.step_time = step_time
        self.noise_std = noise_std

    def get_demand(self) -> int:
        base = self.low if self.t < self.step_time else self.high

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std)
            demand = max(0, int(round(base + noise)))
        else:
            demand = base

        self.t += 1
        return demand


class SinusoidalDemand(DemandPattern):
    """Sinusoidal demand pattern (periodic fluctuation)."""

    def __init__(self, mean: float = 6.0, amplitude: float = 3.0,
                 period: int = 12, noise_std: float = 0.5, seed: int = None):
        super().__init__(seed)
        self.mean = mean
        self.amplitude = amplitude
        self.period = period
        self.noise_std = noise_std

    def get_demand(self) -> int:
        # Sinusoidal with phase starting at peak
        phase = 2 * np.pi * self.t / self.period
        base = self.mean + self.amplitude * np.sin(phase)

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std)
            demand = max(0, int(round(base + noise)))
        else:
            demand = max(0, int(round(base)))

        self.t += 1
        return demand


class RandomWalkDemand(DemandPattern):
    """Random walk demand (non-stationary drift)."""

    def __init__(self, initial: float = 6.0, drift: float = 0.0,
                 step_std: float = 1.0, min_demand: int = 2,
                 max_demand: int = 15, seed: int = None):
        super().__init__(seed)
        self.initial = initial
        self.drift = drift
        self.step_std = step_std
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.current_level = initial

    def reset(self):
        super().reset()
        self.current_level = self.initial

    def get_demand(self) -> int:
        # Random walk with optional drift and bounds
        step = self.rng.normal(self.drift, self.step_std)
        self.current_level += step

        # Reflect at boundaries to prevent unbounded growth
        self.current_level = np.clip(
            self.current_level,
            self.min_demand,
            self.max_demand
        )

        demand = max(0, int(round(self.current_level)))
        self.t += 1
        return demand


class BurstySpikeDemand(DemandPattern):
    """Bursty spike demand (sudden unpredictable surges)."""

    def __init__(self, base_mean: float = 4.0, base_std: float = 1.0,
                 spike_prob: float = 0.15, spike_mean: float = 15.0,
                 spike_std: float = 3.0, seed: int = None):
        super().__init__(seed)
        self.base_mean = base_mean
        self.base_std = base_std
        self.spike_prob = spike_prob
        self.spike_mean = spike_mean
        self.spike_std = spike_std

    def get_demand(self) -> int:
        # Decide if this is a spike period
        is_spike = self.rng.random() < self.spike_prob

        if is_spike:
            demand = self.rng.normal(self.spike_mean, self.spike_std)
        else:
            demand = self.rng.normal(self.base_mean, self.base_std)

        demand = max(0, int(round(demand)))
        self.t += 1
        return demand


class TrendingDemand(DemandPattern):
    """Linear trending demand (seasonal growth)."""

    def __init__(self, initial: float = 4.0, trend: float = 0.1,
                 noise_std: float = 1.0, max_demand: int = 20,
                 seed: int = None):
        super().__init__(seed)
        self.initial = initial
        self.trend = trend
        self.noise_std = noise_std
        self.max_demand = max_demand

    def get_demand(self) -> int:
        # Linear trend with noise
        base = min(self.initial + self.trend * self.t, self.max_demand)

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std)
            demand = max(0, int(round(base + noise)))
        else:
            demand = max(0, int(round(base)))

        self.t += 1
        return demand


def create_demand_pattern(pattern_type: str, seed: int = None) -> DemandPattern:
    """Factory function to create demand patterns."""
    patterns = {
        'step': StepDemand,
        'sinusoidal': SinusoidalDemand,
        'random_walk': RandomWalkDemand,
        'bursty': BurstySpikeDemand,
        'trending': TrendingDemand,
    }

    if pattern_type not in patterns:
        raise ValueError(
            f"Unknown pattern: {pattern_type}. "
            f"Available: {list(patterns.keys())}"
        )

    return patterns[pattern_type](seed=seed)
