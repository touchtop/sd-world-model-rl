#!/usr/bin/env python3
"""
Experiment B: Multi-Step Rollout Error Analysis

Demonstrates that SD's structural constraints bound error propagation,
while BlackBox's error compounds exponentially with rollout depth.

Key insight: Even if single-step error is small for BB, multi-step
accumulation causes divergence. SD's conservation laws prevent this.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent.networks import ValueNetwork
from src.planning.mcts import mcts_search

from experiments.beer_game_common import (
    BeerGameSDModelWrapper,
    make_beer_game_reward_fn,
    NormalizedValueNet,
    ReplayBuffer,
    Transition,
)
from baselines.blackbox_world_model import BlackBoxWorldModel

# Import OOD sampling from main experiment
from experiments.run_ood_state_generalization import (
    OODStateEnv,
    sample_id_state,
    sample_backlog_shock,
    sample_inventory_surplus,
    sample_pipeline_imbalance,
    sample_alternating_extreme,
    OOD_DISTRIBUTIONS,
    train_model,
)


# =============================================================================
# Multi-Step Rollout Error Analysis
# =============================================================================

def evaluate_multi_step_error(
    world_model,
    env: OODStateEnv,
    sample_fn,
    horizons: List[int] = [1, 2, 5, 10, 15, 20, 30, 40, 50],
    n_samples: int = 100,
    n_actions_per_step: int = 5,  # Try multiple actions and average
    device: str = "cpu",
) -> Dict:
    """Evaluate prediction error across multiple rollout horizons.

    For each horizon, we:
    1. Start from an OOD initial state
    2. Roll out H steps using random actions (autoregressive: predictions feed back)
    3. Compare cumulative prediction error vs ground truth

    Key metrics:
    - RMSE: Root mean squared error on state components
    - NRMSE: RMSE normalized by state range (inventory/backlog/pipeline values)

    NOTE: SD model error ≈ 0 is EXPECTED because:
    - SD uses identical conservation laws as the environment
    - Both use: I' = I + arrivals - shipments, B' = B + orders - shipments
    - This demonstrates structural invariance, not data leakage
    """
    rng = np.random.default_rng(42)
    is_mlp = not hasattr(world_model, 'model')

    results = {h: [] for h in horizons}

    for sample_idx in range(n_samples):
        init_state = sample_fn(rng, env.n_echelons)
        env.set_initial_state(init_state)
        true_state, _ = env.reset()

        for horizon in horizons:
            # Roll out exactly H steps
            pred_state = true_state.copy()
            env.set_initial_state(init_state)
            env.reset()

            squared_errors = []
            for step in range(horizon):
                action = env.action_space.sample()

                # Predict next state
                with torch.no_grad():
                    state_t = torch.tensor(pred_state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_t = torch.tensor([action], dtype=torch.int64, device=device)
                    if is_mlp:
                        pred_next = world_model(state_t, action_t).squeeze(0).cpu().numpy()
                    else:
                        pred_next = world_model.model(state_t, action_t).squeeze(0).cpu().numpy()

                # Actual next state
                actual_next, _, terminated, truncated, _ = env.step(action)

                # Squared error on state components (exclude t, demand)
                se = np.mean((pred_next[:-2] - actual_next[:-2]) ** 2)
                squared_errors.append(se)

                pred_state = pred_next

                if terminated or truncated:
                    break

            # RMSE over the horizon
            rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0
            results[horizon].append(rmse)

    # Compute statistics
    summary = {}
    for h in horizons:
        values = results[h]
        summary[h] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Additional: Compute NRMSE (normalized by state range)
    # State components: inventory [0, ~100], backlog [0, ~100], pipeline [0, ~50]
    # Approximate range: 100
    state_range = 100.0
    summary["nrmse"] = {}
    for h in horizons:
        values = results[h]
        summary["nrmse"][h] = {
            "mean": float(np.mean(values) / state_range),
            "std": float(np.std(values) / state_range),
        }

    return summary


def compute_error_growth_rate(
    errors: List[float],
    horizons: List[int],
) -> Dict:
    """Analyze error growth pattern.

    Returns:
    - growth_rate: Exponential growth rate (error ~ exp(rate * horizon))
    - linearity_r2: R² for linear fit (error ~ rate * horizon)
    """
    from scipy import stats

    means = [errors[h]["mean"] for h in horizons]

    # Linear fit: error = a * h + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(horizons, means)

    # Exponential fit: log(error) = a * h + b
    log_means = np.log(np.array(means) + 1e-8)
    exp_slope, exp_intercept, exp_r2, exp_p, exp_std_err = stats.linregress(
        horizons, log_means
    )

    return {
        "linear_slope": float(slope),
        "linear_r2": float(r_value ** 2),
        "exp_growth_rate": float(exp_slope),
        "exp_r2": float(exp_r2),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_multi_step_experiment(
    sample_budget: int = 15000,
    n_seeds: int = 3,
    horizons: List[int] = None,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Run multi-step rollout error analysis."""
    if horizons is None:
        horizons = [1, 2, 5, 10, 15, 20, 30, 40, 50]

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = NEURIPS_DIR / "results" / "multi_step_rollout" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Experiment B: Multi-Step Rollout Error Analysis")
    print("=" * 80)
    print(f"Sample budget: {sample_budget}")
    print(f"Seeds: {n_seeds}")
    print(f"Horizons: {horizons}")
    print(f"Output: {output_dir}")
    print()

    all_results = []

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Create training environment
        train_env = OODStateEnv(
            initial_state_fn=sample_id_state,
            order_levels=3,
            T=36,
        )

        # Train models
        print(f"\n[Seed {seed}] Training models...")
        t0 = time.time()
        sd_wm, sd_vn, sd_nvn, sd_stats = train_model(
            train_env, "sd", sample_budget, device, seed, verbose=verbose
        )
        sd_time = time.time() - t0

        t0 = time.time()
        bb_wm, bb_vn, bb_nvn, bb_stats = train_model(
            train_env, "mlp", sample_budget, device, seed, verbose=verbose
        )
        bb_time = time.time() - t0

        print(f"  Training times: SD={sd_time:.1f}s, BB={bb_time:.1f}s")

        # Evaluate on each OOD distribution
        seed_results = {
            "seed": seed,
            "multi_step_error": {},
            "error_growth": {},
        }

        for dist_key, (dist_desc, sample_fn) in OOD_DISTRIBUTIONS.items():
            print(f"\n  Evaluating on: {dist_desc}")

            sd_eval_env = OODStateEnv(order_levels=3, T=50)  # Longer T for multi-step
            bb_eval_env = OODStateEnv(order_levels=3, T=50)

            sd_errors = evaluate_multi_step_error(
                sd_wm, sd_eval_env, sample_fn, horizons=horizons,
                n_samples=50, device=device
            )
            bb_errors = evaluate_multi_step_error(
                bb_wm, bb_eval_env, sample_fn, horizons=horizons,
                n_samples=50, device=device
            )

            seed_results["multi_step_error"][dist_key] = {
                "sd": sd_errors,
                "blackbox": bb_errors,
            }

            # Compute growth rates
            sd_growth = compute_error_growth_rate(sd_errors, horizons)
            bb_growth = compute_error_growth_rate(bb_errors, horizons)

            seed_results["error_growth"][dist_key] = {
                "sd": sd_growth,
                "blackbox": bb_growth,
            }

            # Print summary
            print(f"    H={horizons[-1]}: SD={sd_errors[horizons[-1]]['mean']:.4f}, "
                  f"BB={bb_errors[horizons[-1]]['mean']:.4f}, "
                  f"Ratio={bb_errors[horizons[-1]]['mean']/max(sd_errors[horizons[-1]]['mean'], 1e-8):.1f}x")
            print(f"    SD exp_growth={sd_growth['exp_growth_rate']:.4f}, "
                  f"BB exp_growth={bb_growth['exp_growth_rate']:.4f}")

        all_results.append(seed_results)

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    # Build summary table for each horizon
    print("\n### Multi-Step Error Growth (RMSE)")
    print(f"{'Horizon':<10} {'ID SD':<10} {'ID BB':<10} {'ID Ratio':<10} | "
          f"{'Shock SD':<10} {'Shock BB':<10} {'Shock Ratio':<10}")
    print("-" * 80)

    for h in horizons:
        id_sd = np.mean([r["multi_step_error"]["id"]["sd"][h]["mean"] for r in all_results])
        id_bb = np.mean([r["multi_step_error"]["id"]["blackbox"][h]["mean"] for r in all_results])
        shock_sd = np.mean([r["multi_step_error"]["backlog_shock"]["sd"][h]["mean"] for r in all_results])
        shock_bb = np.mean([r["multi_step_error"]["backlog_shock"]["blackbox"][h]["mean"] for r in all_results])

        id_ratio = id_bb / max(id_sd, 1e-8)
        shock_ratio = shock_bb / max(shock_sd, 1e-8)

        print(f"{h:<10} {id_sd:<10.4f} {id_bb:<10.4f} {id_ratio:<10.1f}x | "
              f"{shock_sd:<10.4f} {shock_bb:<10.4f} {shock_ratio:<10.1f}x")

    # Growth rate summary
    print("\n### Error Growth Rate Analysis (exponential fit)")
    print(f"{'Distribution':<20} {'SD Growth Rate':<18} {'BB Growth Rate':<18} {'Ratio':<10}")
    print("-" * 70)

    for dist_key in OOD_DISTRIBUTIONS.keys():
        sd_rates = [r["error_growth"][dist_key]["sd"]["exp_growth_rate"] for r in all_results]
        bb_rates = [r["error_growth"][dist_key]["blackbox"]["exp_growth_rate"] for r in all_results]
        sd_mean = np.mean(sd_rates)
        bb_mean = np.mean(bb_rates)
        ratio = bb_mean / max(abs(sd_mean), 1e-8)

        print(f"{dist_key:<20} {sd_mean:<18.4f} {bb_mean:<18.4f} {ratio:<10.1f}x")

    # Save results
    results_file = output_dir / "multi_step_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save CSV for plotting
    import csv
    csv_file = output_dir / "error_curves.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "distribution", "model", "horizon", "rmse"])
        for r in all_results:
            for dist_key in OOD_DISTRIBUTIONS.keys():
                for h in horizons:
                    for model in ["sd", "blackbox"]:
                        model_key = "sd" if model == "sd" else "blackbox"
                        rmse = r["multi_step_error"][dist_key][model_key][h]["mean"]
                        writer.writerow([r["seed"], dist_key, model, h, rmse])
    print(f"CSV saved to: {csv_file}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Step Rollout Error Analysis")
    parser.add_argument("--budget", type=int, default=15000, help="Training sample budget")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose training output")
    args = parser.parse_args()

    run_multi_step_experiment(
        sample_budget=args.budget,
        n_seeds=args.seeds,
        device=args.device,
        verbose=args.verbose,
    )
