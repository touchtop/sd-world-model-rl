#!/usr/bin/env python3
"""Sample Efficiency Comparison: SD-MCTS vs DreamerV3 vs BlackBox-MCTS.

Reuses proven IJOC implementation for SD-MCTS training.
Collects both steps and wall-clock time for comprehensive analysis.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Use IJOC's proven implementation
SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.planning.mcts import mcts_search

# Import IJOC's proven training functions
from experiments.run_beer_game_v2 import (
    train_sd_mcts,
    evaluate_sd_mcts,
    BeerGameSDModelWrapper,
    make_beer_game_reward_fn,
)

# NeurIPS-specific baselines
sys.path.insert(0, str(NEURIPS_DIR))
from baselines.blackbox_world_model import BlackBoxWorldModel, train_blackbox_model


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    env_name: str = "beer_game"
    sample_budgets: List[int] = field(default_factory=lambda: [1000, 3000, 10000])
    n_seeds: int = 3
    eval_episodes: int = 10
    mcts_simulations: int = 200
    output_dir: Optional[str] = None


@dataclass
class RunMetrics:
    """Metrics from a single run."""
    method: str
    env: str
    sample_budget: int
    seed: int
    # Performance (negative cost, higher is better)
    eval_return_mean: float = 0.0
    eval_return_std: float = 0.0
    train_returns: List[float] = field(default_factory=list)
    # Time breakdown
    total_time: float = 0.0
    env_time: float = 0.0
    train_time: float = 0.0
    plan_time: float = 0.0
    # Additional
    total_steps: int = 0
    total_episodes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'env': self.env,
            'sample_budget': self.sample_budget,
            'seed': self.seed,
            'eval_return_mean': self.eval_return_mean,
            'eval_return_std': self.eval_return_std,
            'total_time': self.total_time,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'error': self.error,
        }


def run_sd_mcts(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
    device: str = 'cpu',
) -> RunMetrics:
    """Run SD-MCTS using IJOC's proven implementation."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics = RunMetrics(
        method="sd_mcts",
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )

    # Create environment (flat action space as in IJOC)
    env = BeerGameEnvFlat(order_levels=3, seed=seed)  # 3^4 = 81 actions

    start_time = time.time()

    try:
        # Use IJOC's proven training function
        world_model, policy_net, value_net, train_returns = train_sd_mcts(
            env=env,
            env_steps=sample_budget,
            gamma=0.99,
            device=device,
            verbose=True,
        )

        train_time = time.time() - start_time
        metrics.train_returns = train_returns
        metrics.total_episodes = len(train_returns)

        # Evaluation using IJOC's function
        eval_start = time.time()
        eval_results = evaluate_sd_mcts(
            env=env,
            world_model=world_model,
            policy_net=policy_net,
            value_net=value_net,
            n_episodes=eval_episodes,
            num_simulations=mcts_simulations,
            device=device,
        )
        eval_time = time.time() - eval_start

        # Convert cost to return (negative cost)
        metrics.eval_return_mean = -eval_results['total_cost_mean']
        metrics.eval_return_std = eval_results['total_cost_std']
        metrics.total_time = time.time() - start_time
        metrics.train_time = train_time
        metrics.plan_time = eval_time
        metrics.total_steps = sample_budget

    except Exception as e:
        metrics.error = str(e)
        metrics.total_time = time.time() - start_time
        import traceback
        traceback.print_exc()

    return metrics


def run_blackbox_mcts(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
    device: str = 'cpu',
) -> RunMetrics:
    """Run BlackBox-MCTS (ablation baseline without SD structure)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics = RunMetrics(
        method="blackbox_mcts",
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )

    env = BeerGameEnvFlat(order_levels=3, seed=seed)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    start_time = time.time()
    env_time = 0.0
    train_time = 0.0

    try:
        # Create BlackBox world model (MLP, no SD structure)
        bb_model = BlackBoxWorldModel(state_dim, n_actions, hidden_dim=256).to(device)
        reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

        # Data collection with random policy
        replay_buffer = []
        total_steps = 0
        train_returns = []

        while total_steps < sample_budget:
            t0 = time.time()
            state, _ = env.reset(seed=seed + len(train_returns))
            env_time += time.time() - t0

            episode_return = 0.0
            done = False

            while not done and total_steps < sample_budget:
                # Random action during data collection
                action = env.action_space.sample()

                t0 = time.time()
                next_state, reward, terminated, truncated, _ = env.step(action)
                env_time += time.time() - t0

                replay_buffer.append((state, action, next_state, reward, terminated or truncated))
                state = next_state
                episode_return += reward
                total_steps += 1
                done = terminated or truncated

            train_returns.append(episode_return)

        # Train BlackBox model
        t0 = time.time()
        train_blackbox_model(bb_model, replay_buffer, epochs=200)
        train_time = time.time() - t0

        # Evaluation with MCTS using trained BlackBox model
        eval_returns = []
        for ep in range(eval_episodes):
            state, _ = env.reset(seed=1000 + ep)
            episode_return = 0.0
            done = False

            while not done:
                action = mcts_search(
                    root_state=state,
                    world_model=bb_model,
                    n_actions=n_actions,
                    num_simulations=mcts_simulations,
                    reward_fn=reward_fn,
                )
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                done = terminated or truncated

            eval_returns.append(episode_return)

        metrics.train_returns = train_returns
        metrics.eval_return_mean = float(np.mean(eval_returns))
        metrics.eval_return_std = float(np.std(eval_returns))
        metrics.total_time = time.time() - start_time
        metrics.env_time = env_time
        metrics.train_time = train_time
        metrics.total_steps = total_steps
        metrics.total_episodes = len(train_returns)

    except Exception as e:
        metrics.error = str(e)
        metrics.total_time = time.time() - start_time
        import traceback
        traceback.print_exc()

    return metrics


def run_dreamerv3(
    sample_budget: int,
    seed: int,
    logdir: Path,
) -> RunMetrics:
    """Run DreamerV3 via subprocess."""
    metrics = RunMetrics(
        method="dreamerv3",
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )

    script_path = SCRIPT_DIR / "train_dreamerv3_full.py"
    run_logdir = logdir / f"dreamerv3_{sample_budget}_{seed}"

    # Update embodied_wrappers path for DreamerV3
    baselines_init = NEURIPS_DIR / "baselines" / "__init__.py"
    if not baselines_init.exists():
        baselines_init.write_text("")

    cmd = [
        sys.executable, str(script_path),
        "--env", "beer_game",
        "--steps", str(sample_budget),
        "--seed", str(seed),
        "--logdir", str(run_logdir),
        "--batch-size", "16",
        "--batch-length", "32",  # Must be <= episode length (36 for Beer Game)
        "--train-ratio", "4.0",
        "--random-steps", str(min(200, sample_budget // 10)),
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            metrics.error = result.stderr[-1000:] if result.stderr else "Unknown error"
            return metrics
    except subprocess.TimeoutExpired:
        metrics.error = "Timeout"
        return metrics
    except Exception as e:
        metrics.error = str(e)
        return metrics

    # Load results
    summary_path = run_logdir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        metrics.eval_return_mean = summary.get("final_return_mean", 0.0)
        metrics.eval_return_std = summary.get("final_return_std", 0.0)
        metrics.total_time = summary.get("total_time", time.time() - start_time)
        metrics.env_time = summary.get("env_time", 0.0)
        metrics.train_time = summary.get("train_time", 0.0)
        metrics.total_steps = summary.get("total_steps", sample_budget)
        metrics.total_episodes = summary.get("total_episodes", 0)
    else:
        metrics.error = f"Summary file not found: {summary_path}"

    return metrics


def run_experiments(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Run all experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        output_dir = NEURIPS_DIR / "results" / "comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = []
    total_runs = len(config.sample_budgets) * config.n_seeds * 3
    current = 0

    for budget in config.sample_budgets:
        for seed in range(config.n_seeds):
            # SD-MCTS (our method)
            current += 1
            print(f"\n[{current}/{total_runs}] SD-MCTS | budget={budget} | seed={seed}")
            m = run_sd_mcts(budget, seed, config.eval_episodes, config.mcts_simulations, device)
            results.append(m.to_dict())
            if m.error:
                print(f"  ERROR: {m.error}")
            else:
                print(f"  Return: {m.eval_return_mean:.1f} +/- {m.eval_return_std:.1f} | Time: {m.total_time:.1f}s")

            # BlackBox-MCTS (ablation)
            current += 1
            print(f"\n[{current}/{total_runs}] BlackBox-MCTS | budget={budget} | seed={seed}")
            m = run_blackbox_mcts(budget, seed, config.eval_episodes, config.mcts_simulations, device)
            results.append(m.to_dict())
            if m.error:
                print(f"  ERROR: {m.error}")
            else:
                print(f"  Return: {m.eval_return_mean:.1f} +/- {m.eval_return_std:.1f} | Time: {m.total_time:.1f}s")

            # DreamerV3
            current += 1
            print(f"\n[{current}/{total_runs}] DreamerV3 | budget={budget} | seed={seed}")
            m = run_dreamerv3(budget, seed, output_dir)
            results.append(m.to_dict())
            if m.error:
                print(f"  ERROR: {m.error}")
            else:
                print(f"  Return: {m.eval_return_mean:.1f} +/- {m.eval_return_std:.1f} | Time: {m.total_time:.1f}s")

            # Save intermediate results
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            'env_name': config.env_name,
            'sample_budgets': config.sample_budgets,
            'n_seeds': config.n_seeds,
            'eval_episodes': config.eval_episodes,
            'mcts_simulations': config.mcts_simulations,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary table."""
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error"):
            continue
        method = r["method"]
        budget = r["sample_budget"]
        grouped[budget][method].append(r["eval_return_mean"])

    print("\n" + "="*80)
    print("SAMPLE EFFICIENCY SUMMARY (Higher Return = Better)")
    print("="*80)
    print(f"{'Budget':<10} {'SD-MCTS':<22} {'BlackBox-MCTS':<22} {'DreamerV3':<22}")
    print("-"*80)

    for budget in sorted(grouped.keys()):
        row = f"{budget:<10}"
        for method in ["sd_mcts", "blackbox_mcts", "dreamerv3"]:
            vals = grouped[budget].get(method, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0
                row += f"{mean:>9.1f} +/- {std:<8.1f}"
            else:
                row += f"{'N/A':<22}"
        print(row)
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Sample Efficiency Comparison")
    parser.add_argument("--env", default="beer_game", help="Environment name")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 3000, 10000],
                        help="Sample budgets to test")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = ExperimentConfig(
        env_name=args.env,
        sample_budgets=args.budgets,
        n_seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        mcts_simulations=args.mcts_sims,
        output_dir=args.output,
    )

    print("="*60)
    print("Sample Efficiency Comparison Experiment")
    print("="*60)
    print(f"Environment: {config.env_name}")
    print(f"Sample budgets: {config.sample_budgets}")
    print(f"Seeds: {config.n_seeds}")
    print(f"MCTS simulations: {config.mcts_simulations}")
    print(f"Methods: SD-MCTS, BlackBox-MCTS, DreamerV3")
    print("="*60)

    results = run_experiments(config)
    print_summary(results)


if __name__ == "__main__":
    main()
