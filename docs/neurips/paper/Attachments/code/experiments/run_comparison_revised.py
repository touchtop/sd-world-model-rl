#!/usr/bin/env python3
"""Revised Experiment 1 runner for the paper.

This keeps the original comparison script intact and only replaces the SD-MCTS
runner with a train/eval-consistent version aligned with the later Cartpole
workflow. Other baselines can be reused from the original script when needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = NEURIPS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat

from experiments.run_ablation_principled import train_and_eval


@dataclass
class ExperimentConfig:
    sample_budgets: List[int] = field(default_factory=lambda: [1000, 3000, 10000])
    n_seeds: int = 3
    eval_episodes: int = 10
    mcts_simulations: int = 200
    output_dir: str | None = None
    methods: List[str] = field(default_factory=lambda: ["sd_mcts"])


@dataclass
class RunMetrics:
    method: str
    env: str
    sample_budget: int
    seed: int
    eval_return_mean: float = 0.0
    eval_return_std: float = 0.0
    train_returns: List[float] = field(default_factory=list)
    total_time: float = 0.0
    env_time: float = 0.0
    train_time: float = 0.0
    plan_time: float = 0.0
    total_steps: int = 0
    total_episodes: int = 0
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "env": self.env,
            "sample_budget": self.sample_budget,
            "seed": self.seed,
            "eval_return_mean": self.eval_return_mean,
            "eval_return_std": self.eval_return_std,
            "total_time": self.total_time,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "error": self.error,
        }


def run_sd_mcts_revised(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
) -> RunMetrics:
    device = "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics = RunMetrics(
        method="sd_mcts",
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )
    env = BeerGameEnvFlat(order_levels=3, seed=seed)
    start_time = time.time()

    try:
        variant = {
            "model": "sd",
            "use_vnet": True,
            "use_mcts": True,
            "prior": None,
        }
        scenario = {
            "env": {
                "tau_info": 2,
                "tau_ship": 2,
                "T": 36,
                "demand_std": 0.0,
            },
            "train": {
                "sample_budget": sample_budget,
            },
        }

        eval_mean, eval_std, train_time, total_steps = train_and_eval(
            env=env,
            variant_config=variant,
            scenario_config=scenario,
            seed=seed,
            eval_episodes=eval_episodes,
            mcts_simulations=mcts_simulations,
            device=device,
        )

        metrics.eval_return_mean = eval_mean
        metrics.eval_return_std = eval_std
        metrics.total_time = time.time() - start_time
        metrics.train_time = train_time
        metrics.total_steps = total_steps

    except Exception as exc:
        metrics.error = str(exc)
        metrics.total_time = time.time() - start_time
        import traceback

        traceback.print_exc()

    return metrics


METHOD_RUNNERS = {
    "sd_mcts": run_sd_mcts_revised,
}


def _unsupported_baseline(method: str):
    def runner(*args, **_kwargs):
        sample_budget = int(args[0]) if len(args) > 0 else 0
        seed = int(args[1]) if len(args) > 1 else 0
        metric = RunMetrics(method=method, env="beer_game", sample_budget=sample_budget, seed=seed)
        metric.error = (
            f"{method} rerun is not bundled in this supplemental package; use "
            "data/beer_game/dreamerv3_ppo_historical_results.json for the "
            "paper aggregate, or install the public baseline separately."
        )
        return metric

    return runner


for _method in ("dreamerv3", "ppo"):
    METHOD_RUNNERS[_method] = _unsupported_baseline(_method)


def print_summary(results: List[Dict[str, Any]], experiment_type: str = "main") -> None:
    """Print a compact aggregate table for the bundled SD-MCTS rerun."""
    print("\n" + "=" * 60)
    print(f"Summary: {experiment_type}")
    print("=" * 60)

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for item in results:
        if item.get("error"):
            continue
        key = (item["method"], item["sample_budget"])
        grouped.setdefault(key, []).append(item)

    for (method, budget), rows in sorted(grouped.items()):
        values = [float(row["eval_return_mean"]) for row in rows]
        print(
            f"{method:12s} budget={budget:6d} "
            f"mean={np.mean(values):8.1f} stderr={np.std(values) / max(len(values), 1) ** 0.5:6.1f}"
        )


def run_experiments(config: ExperimentConfig) -> List[Dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        output_dir = NEURIPS_DIR / "results" / "main_revised" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Methods: {config.methods}")
    print(f"Output: {output_dir}")

    results: List[Dict[str, Any]] = []
    total_runs = len(config.sample_budgets) * config.n_seeds * len(config.methods)
    current = 0

    for budget in config.sample_budgets:
        for seed in range(config.n_seeds):
            for method in config.methods:
                current += 1
                print(f"\n[{current}/{total_runs}] {method.upper()} | budget={budget} | seed={seed}")
                runner = METHOD_RUNNERS[method]
                if method == "dreamerv3":
                    metric = runner(budget, seed, output_dir, config.eval_episodes)
                elif method in ("ppo", "dqn"):
                    metric = runner(budget, seed, config.eval_episodes)
                else:
                    metric = runner(budget, seed, config.eval_episodes, config.mcts_simulations)

                results.append(metric.to_dict())
                if metric.error:
                    print(f"  ERROR: {metric.error}")
                else:
                    print(
                        f"  Return: {metric.eval_return_mean:.1f} +/- "
                        f"{metric.eval_return_std:.1f} | Time: {metric.total_time:.1f}s"
                    )

                with open(output_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "sample_budgets": config.sample_budgets,
                "n_seeds": config.n_seeds,
                "eval_episodes": config.eval_episodes,
                "mcts_simulations": config.mcts_simulations,
                "methods": config.methods,
            },
            f,
            indent=2,
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Revised Sample Efficiency Experiments")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 3000, 10000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--mcts-sims", type=int, default=200)
    parser.add_argument("--methods", type=str, nargs="+", default=["sd_mcts"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = ExperimentConfig(
        sample_budgets=args.budgets,
        n_seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        mcts_simulations=args.mcts_sims,
        output_dir=args.output,
        methods=args.methods,
    )

    print("=" * 60)
    print("Revised Sample Efficiency Experiment")
    print("=" * 60)
    print(f"Budgets: {config.sample_budgets}")
    print(f"Seeds: {config.n_seeds}")
    print(f"Methods: {config.methods}")
    print("=" * 60)

    results = run_experiments(config)
    print_summary(results, "main")


if __name__ == "__main__":
    main()
