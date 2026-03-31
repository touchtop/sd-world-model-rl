#!/usr/bin/env python3
"""Sample Efficiency Comparison: SD-MCTS vs SOTA baselines.

Experiment Modes:
- main: Sample efficiency comparison (SD-MCTS vs DreamerV3 vs PPO vs DQN)
- ablation: Component ablation (SD-MCTS vs BlackBox-MCTS)

Usage:
    # Main experiment
    python run_comparison.py --experiment main --budgets 1000 3000 --seeds 3

    # Ablation experiment
    python run_comparison.py --experiment ablation --budgets 1000 --seeds 3
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

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.planning.mcts import mcts_search

from experiments.run_beer_game_v2 import (
    train_sd_mcts,
    evaluate_sd_mcts,
    BeerGameSDModelWrapper,
    make_beer_game_reward_fn,
)

from baselines.blackbox_world_model import BlackBoxWorldModel, train_blackbox_model

# Experiment method groups
MAIN_METHODS = ['sd_mcts', 'dreamerv3', 'ppo', 'dqn']
ABLATION_METHODS = ['sd_mcts', 'blackbox_mcts']


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_type: str = "main"  # "main" or "ablation"
    env_name: str = "beer_game"
    sample_budgets: List[int] = field(default_factory=lambda: [1000, 3000, 10000])
    n_seeds: int = 3
    eval_episodes: int = 10
    mcts_simulations: int = 200
    output_dir: Optional[str] = None

    @property
    def methods(self) -> List[str]:
        if self.experiment_type == "main":
            return MAIN_METHODS
        elif self.experiment_type == "ablation":
            return ABLATION_METHODS
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")


@dataclass
class RunMetrics:
    """Metrics from a single run."""
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


# =============================================================================
# SD-MCTS (Our Method)
# =============================================================================

def run_sd_mcts(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
) -> RunMetrics:
    """Run SD-MCTS using IJOC's proven implementation."""
    # Force CPU: serial MCTS with batch_size=1 is faster on CPU
    device = 'cpu'

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

        eval_start = time.time()
        eval_results = evaluate_sd_mcts(
            env=env,
            world_model=world_model,
            policy_net=policy_net,
            value_net=value_net,
            n_episodes=eval_episodes,
            num_simulations=mcts_simulations,
            device=device,
            # Note: evaluate_sd_mcts already uses seed=1000+ep internally
        )
        eval_time = time.time() - eval_start

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


# =============================================================================
# BlackBox-MCTS (Ablation: MLP world model)
# =============================================================================

def run_blackbox_mcts(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
) -> RunMetrics:
    """Run BlackBox-MCTS with on-policy data collection.

    To fairly compare with SD-MCTS, we use:
    1. Initial random exploration phase (warmup_steps)
    2. Train initial model
    3. On-policy data collection using MCTS (not random actions)
    4. Interleaved training during on-policy phase
    """
    # Force CPU: non-batched, non-parallel MCTS doesn't benefit from GPU
    device = 'cpu'

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
        bb_model = BlackBoxWorldModel(state_dim, n_actions, hidden_dim=256).to(device)
        reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

        replay_buffer = []
        total_steps = 0
        train_returns = []

        # Phase 1: Warmup with random actions (similar to DreamerV3's random_steps)
        warmup_steps = min(200, sample_budget // 5)
        steps_per_train = 200  # Interleaved training interval
        model_trained = False  # Track if model has been trained

        state, _ = env.reset(seed=seed)

        while total_steps < sample_budget:
            episode_return = 0.0
            done = False

            while not done and total_steps < sample_budget:
                # Use random during warmup, then MCTS (on-policy)
                if total_steps < warmup_steps or not model_trained:
                    action = env.action_space.sample()
                else:
                    # On-policy: use trained model for action selection
                    action = mcts_search(
                        root_state=state,
                        world_model=bb_model,
                        n_actions=n_actions,
                        num_simulations=mcts_simulations,  # Same as SD-MCTS for fair comparison
                        reward_fn=reward_fn,
                    )

                t0 = time.time()
                next_state, reward, terminated, truncated, _ = env.step(action)
                env_time += time.time() - t0

                replay_buffer.append((state, action, next_state, reward, terminated or truncated))
                state = next_state
                episode_return += reward
                total_steps += 1

                # Interleaved training (online RL)
                if total_steps % steps_per_train == 0 and len(replay_buffer) >= 64:
                    t0 = time.time()
                    # Adaptive epochs: more epochs when buffer is small
                    epochs = max(10, min(50, 1000 // len(replay_buffer) * 10))
                    train_blackbox_model(bb_model, replay_buffer, epochs=epochs, batch_size=64)
                    train_time += time.time() - t0
                    model_trained = True

                done = terminated or truncated

            train_returns.append(episode_return)
            if done:
                state, _ = env.reset(seed=seed + len(train_returns))

        # Final training pass with more epochs
        if len(replay_buffer) >= 64:
            t0 = time.time()
            final_epochs = max(50, min(200, 2000 // max(1, len(replay_buffer) // 100)))
            train_blackbox_model(bb_model, replay_buffer, epochs=final_epochs, batch_size=64)
            train_time += time.time() - t0

        bb_model.eval()

        # Evaluation with MCTS
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


# =============================================================================
# DreamerV3 (SOTA MBRL Baseline)
# =============================================================================

def run_dreamerv3(
    sample_budget: int,
    seed: int,
    logdir: Path,
    eval_episodes: int = 10,
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

    # Adaptive hyperparameters for low-sample regime
    # Balance between learning quality and training speed
    if sample_budget <= 2000:
        batch_size = 8
        batch_length = 32  # Full trajectory context (Beer Game episode=36)
        train_ratio = 2.0  # Balanced: enough updates without excessive overhead
        random_steps = min(100, max(50, sample_budget // 10))
    else:
        batch_size = 16
        batch_length = 32
        train_ratio = 4.0  # Standard ratio for larger budgets
        random_steps = min(200, sample_budget // 10)

    cmd = [
        sys.executable, str(script_path),
        "--env", "beer_game",
        "--steps", str(sample_budget),
        "--seed", str(seed),
        "--logdir", str(run_logdir),
        "--batch-size", str(batch_size),
        "--batch-length", str(batch_length),
        "--train-ratio", str(train_ratio),
        "--random-steps", str(random_steps),
        "--eval-episodes", str(eval_episodes),
        "--eval-seed-base", "1000",
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


# =============================================================================
# PPO (Model-Free Baseline)
# =============================================================================

def run_ppo(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
) -> RunMetrics:
    """Run PPO using stable-baselines3."""
    metrics = RunMetrics(
        method="ppo",
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )

    start_time = time.time()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy

        env = BeerGameEnvFlat(order_levels=3, seed=seed)
        eval_env = BeerGameEnvFlat(order_levels=3, seed=1000)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,  # Reduced for low-sample regime
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
            seed=seed,
            device='cpu',  # Force CPU for compatibility
        )

        model.learn(total_timesteps=sample_budget)
        metrics.train_time = time.time() - start_time
        metrics.total_steps = int(getattr(model, "num_timesteps", sample_budget))

        # Evaluation with consistent seeds
        eval_returns = []
        for ep in range(eval_episodes):
            eval_env_ep = BeerGameEnvFlat(order_levels=3, seed=1000 + ep)
            obs, _ = eval_env_ep.reset()
            episode_return = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env_ep.step(action)
                episode_return += reward
                done = terminated or truncated
            eval_returns.append(episode_return)

        metrics.eval_return_mean = float(np.mean(eval_returns))
        metrics.eval_return_std = float(np.std(eval_returns))
        metrics.total_time = time.time() - start_time
        metrics.total_steps = sample_budget

    except ImportError:
        metrics.error = "stable-baselines3 not installed. Run: pip install stable-baselines3"
    except Exception as e:
        metrics.error = str(e)
        metrics.total_time = time.time() - start_time
        import traceback
        traceback.print_exc()

    return metrics


# =============================================================================
# DQN (Model-Free Baseline)
# =============================================================================

def run_dqn(
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
) -> RunMetrics:
    """Run DQN using stable-baselines3.

    Note: Beer Game uses discrete action space; DQN is the appropriate
    off-policy model-free baseline.
    """
    metrics = RunMetrics(
        method="dqn",
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )

    start_time = time.time()

    try:
        from stable_baselines3 import DQN

        env = BeerGameEnvFlat(order_levels=3, seed=seed)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=min(50, sample_budget // 10),  # Lower threshold for low-sample
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=max(100, sample_budget // 10),  # Adaptive for small budgets
            verbose=0,
            seed=seed,
            device='cpu',  # Force CPU for compatibility
        )

        model.learn(total_timesteps=sample_budget)
        metrics.train_time = time.time() - start_time
        metrics.total_steps = int(getattr(model, "num_timesteps", sample_budget))

        # Evaluation with consistent seeds
        eval_returns = []
        for ep in range(eval_episodes):
            eval_env = BeerGameEnvFlat(order_levels=3, seed=1000 + ep)
            obs, _ = eval_env.reset()
            episode_return = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_return += reward
                done = terminated or truncated
            eval_returns.append(episode_return)

        metrics.eval_return_mean = float(np.mean(eval_returns))
        metrics.eval_return_std = float(np.std(eval_returns))
        metrics.total_time = time.time() - start_time

    except ImportError:
        metrics.error = "stable-baselines3 not installed. Run: pip install stable-baselines3"
    except Exception as e:
        metrics.error = str(e)
        metrics.total_time = time.time() - start_time
        import traceback
        traceback.print_exc()

    return metrics


# =============================================================================
# Experiment Runner
# =============================================================================

METHOD_RUNNERS = {
    'sd_mcts': run_sd_mcts,
    'blackbox_mcts': run_blackbox_mcts,
    'dreamerv3': run_dreamerv3,
    'ppo': run_ppo,
    'dqn': run_dqn,
}


def run_experiments(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Run all experiments based on configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        output_dir = NEURIPS_DIR / "results" / config.experiment_type / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment type: {config.experiment_type}")
    print(f"Methods: {config.methods}")
    print(f"Output: {output_dir}")

    results = []
    methods = config.methods
    total_runs = len(config.sample_budgets) * config.n_seeds * len(methods)
    current = 0

    for budget in config.sample_budgets:
        for seed in range(config.n_seeds):
            for method in methods:
                current += 1
                print(f"\n[{current}/{total_runs}] {method.upper()} | budget={budget} | seed={seed}")

                runner = METHOD_RUNNERS[method]

                # Different methods have different signatures
                if method == 'dreamerv3':
                    m = runner(budget, seed, output_dir, config.eval_episodes)
                elif method in ('ppo', 'dqn'):
                    m = runner(budget, seed, config.eval_episodes)
                else:  # sd_mcts, blackbox_mcts
                    m = runner(budget, seed, config.eval_episodes, config.mcts_simulations)

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
            'experiment_type': config.experiment_type,
            'env_name': config.env_name,
            'sample_budgets': config.sample_budgets,
            'n_seeds': config.n_seeds,
            'eval_episodes': config.eval_episodes,
            'mcts_simulations': config.mcts_simulations,
            'methods': config.methods,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")

    return results


def print_summary(results: List[Dict[str, Any]], experiment_type: str):
    """Print summary table."""
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error"):
            continue
        method = r["method"]
        budget = r["sample_budget"]
        grouped[budget][method].append(r["eval_return_mean"])

    methods = MAIN_METHODS if experiment_type == "main" else ABLATION_METHODS

    print(f"\n{'='*80}")
    print(f"SAMPLE EFFICIENCY SUMMARY ({experiment_type.upper()})")
    print(f"{'='*80}")

    header = f"{'Budget':<10}"
    for method in methods:
        header += f"{method:<18}"
    print(header)
    print("-"*80)

    for budget in sorted(grouped.keys()):
        row = f"{budget:<10}"
        for method in methods:
            vals = grouped[budget].get(method, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0
                row += f"{mean:>7.1f}±{std:<8.1f}"
            else:
                row += f"{'N/A':<18}"
        print(row)
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Sample Efficiency Experiments")
    parser.add_argument("--experiment", type=str, default="main",
                        choices=["main", "ablation"],
                        help="Experiment type: 'main' for sample efficiency, 'ablation' for component analysis")
    parser.add_argument("--env", default="beer_game", help="Environment name")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 3000, 10000],
                        help="Sample budgets to test")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_type=args.experiment,
        env_name=args.env,
        sample_budgets=args.budgets,
        n_seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        mcts_simulations=args.mcts_sims,
        output_dir=args.output,
    )

    print("="*60)
    print(f"Sample Efficiency Experiment: {config.experiment_type.upper()}")
    print("="*60)
    print(f"Environment: {config.env_name}")
    print(f"Sample budgets: {config.sample_budgets}")
    print(f"Seeds: {config.n_seeds}")
    print(f"Methods: {config.methods}")
    print("="*60)

    results = run_experiments(config)
    print_summary(results, config.experiment_type)


if __name__ == "__main__":
    main()
