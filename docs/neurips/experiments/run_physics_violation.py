"""
NeurIPS Experiment: Physics Violation Analysis

P0 Priority: Compare physics/constraint violation rates between
structured (SD-MCTS) and unstructured (BlackBox) world models.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


CONFIG = {
    "experiment_name": "physics_violation",
    "environments": ["beer_game"],
    "methods": ["sd_mcts", "blackbox_mcts"],
    "test_conditions": [
        "in_distribution",
    ],
    "n_seeds": 5,
    "eval_episodes": 30,
}


def check_physics_violations(states: np.ndarray, env_name: str) -> Dict:
    """
    Check for physics violations in state trajectories.

    Violations include:
    - Negative inventory/stock values
    - Capacity overflow
    - Conservation law violations
    """
    violations = {
        "negative_stock": 0,
        "capacity_overflow": 0,
        "conservation_violation": 0,
        "total_states": len(states),
    }

    for state in states:
        # Check negative values (stocks should be non-negative)
        if np.any(state < -1e-6):
            violations["negative_stock"] += 1

        # Check capacity bounds (environment-specific)
        if env_name == "beer_game":
            max_inventory = 100
            if np.any(state[:4] > max_inventory):
                violations["capacity_overflow"] += 1

    # Calculate rates
    n = violations["total_states"]
    violations["negative_rate"] = violations["negative_stock"] / n if n > 0 else 0
    violations["overflow_rate"] = violations["capacity_overflow"] / n if n > 0 else 0
    violations["total_violation_rate"] = (
        violations["negative_stock"] + violations["capacity_overflow"]
    ) / n if n > 0 else 0

    return violations


def create_ood_environment(env_name: str, condition: str):
    """Create environment with OOD conditions for testing."""
    if env_name == "beer_game":
        from env.beer_game_env import BeerGameEnv

        if condition == "in_distribution":
            return BeerGameEnv()
        elif condition == "demand_shift_1.5x":
            return BeerGameEnv(demand_mean=6.0)  # 1.5x of default 4.0
        elif condition == "demand_shift_2x":
            return BeerGameEnv(demand_mean=8.0)  # 2x of default
        elif condition == "parameter_drift_30pct":
            return BeerGameEnv(lead_time=3)  # 30% longer lead time
    elif env_name == "delayed_collapse":
        from env.delayed_collapse_env import DelayedCollapseEnv

        if condition == "in_distribution":
            return DelayedCollapseEnv()
        elif "demand_shift" in condition:
            factor = float(condition.split("_")[-1].replace("x", ""))
            return DelayedCollapseEnv(surge_bonus=15 * factor)
        elif condition == "parameter_drift_30pct":
            return DelayedCollapseEnv(collapse_threshold=13)  # 30% higher

    raise ValueError(f"Unknown env/condition: {env_name}/{condition}")


def run_violation_experiment(
    env_name: str,
    method_name: str,
    condition: str,
    seed: int,
) -> Dict:
    """Run single violation analysis experiment."""
    np.random.seed(seed)

    # Create environment
    train_env = create_ood_environment(env_name, "in_distribution")
    test_env = create_ood_environment(env_name, condition)

    # Create and train agent
    from run_sample_efficiency import create_agent, train_agent
    agent = create_agent(method_name, train_env)
    train_agent(agent, train_env, sample_budget=10000)

    # Evaluate and collect state trajectories
    all_states = []
    returns = []

    for ep in range(CONFIG["eval_episodes"]):
        state = test_env.reset()
        episode_states = [state.copy()]
        episode_return = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = test_env.step(action)
            episode_states.append(next_state.copy())
            state = next_state
            episode_return += reward

        all_states.extend(episode_states)
        returns.append(episode_return)

    # Analyze violations
    violations = check_physics_violations(np.array(all_states), env_name)

    return {
        "env": env_name,
        "method": method_name,
        "condition": condition,
        "seed": seed,
        "eval_return_mean": np.mean(returns),
        "eval_return_std": np.std(returns),
        **violations,
    }


def run_full_experiment():
    """Run full physics violation experiment."""
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / "physics_violation" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_name in CONFIG["environments"]:
        for method_name in CONFIG["methods"]:
            for condition in CONFIG["test_conditions"]:
                for seed in range(CONFIG["n_seeds"]):
                    print(f"{env_name} / {method_name} / {condition} / seed {seed}")

                    try:
                        result = run_violation_experiment(
                            env_name, method_name, condition, seed
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        results.append({
                            "env": env_name,
                            "method": method_name,
                            "condition": condition,
                            "seed": seed,
                            "error": str(e),
                        })

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Generate summary table
    generate_summary(results, output_dir)

    print(f"\nResults saved to: {output_dir}")
    return results


def generate_summary(results: List[Dict], output_dir: Path):
    """Generate summary table of violation rates."""
    summary = {}

    for r in results:
        if "error" in r:
            continue

        key = (r["method"], r["condition"])
        if key not in summary:
            summary[key] = []
        summary[key].append(r["total_violation_rate"])

    # Format as table
    lines = ["# Physics Violation Rate Summary\n"]
    lines.append("| Method | Condition | Violation Rate (mean ± std) |")
    lines.append("|--------|-----------|----------------------------|")

    for (method, condition), rates in sorted(summary.items()):
        mean_rate = np.mean(rates) * 100
        std_rate = np.std(rates) * 100
        lines.append(f"| {method} | {condition} | {mean_rate:.2f}% ± {std_rate:.2f}% |")

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_full_experiment()
