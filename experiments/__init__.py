"""Experiment package.

M1 framework symbols are imported lazily so ICLR supplementary modules can
be used without pulling matplotlib or task-allocation dependencies.
"""

_M1_NAMES = {
    "ScenarioConfig",
    "MetricsCollector",
    "BaselinePolicy",
    "JSQPolicy",
    "CmuPolicy",
    "MPCPolicy",
    "ADPRolloutPolicy",
    "PPOWrapper",
    "SDMCTSWrapper",
    "ExperimentRunner",
    "run_m1_experiments",
    "save_results_to_csv",
    "plot_comparison",
}


def __getattr__(name):
    if name in _M1_NAMES:
        from . import m1_framework

        return getattr(m1_framework, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_M1_NAMES)
