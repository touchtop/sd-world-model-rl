#!/usr/bin/env python3
"""Ablation Study for SD-MCTS (Full Cross-Product Matrix).

Design based on Codex + Gemini + Claude consensus:
- All variants tested on all scenarios (full cross-product)
- 3 core scenarios in main table, 3 control scenarios in appendix
- Paired seeds enable statistical testing with higher power

Usage:
    python run_ablation_principled.py --seeds 5
    python run_ablation_principled.py --suite appendix --seeds 5
    python run_ablation_principled.py --describe
"""

import argparse
import copy
import json
import os
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

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
from baselines.blackbox_world_model import BlackBoxWorldModel, train_blackbox_model


# =============================================================================
# Scenario and Variant Definitions
# =============================================================================

# Core scenarios (main table in paper)
SCENARIOS_CORE = {
    "standard": {
        "desc": "Standard Beer Game: short horizon, moderate noise",
        "env": {"tau_info": 2, "tau_ship": 2, "T": 36, "demand_std": 0.5},
        "train": {"sample_budget": 3000},
    },
    "low_sample_shift": {
        "desc": "Low samples + distribution shift (SD advantage)",
        "env": {"tau_info": 2, "tau_ship": 2, "T": 36, "demand_std": 1.0},
        "train": {"sample_budget": 1000},
        "eval": {"demand_high": 10},
    },
    "long_horizon_delay": {
        "desc": "Long horizon + large delays (VNet + MCTS advantage)",
        "env": {"tau_info": 2, "tau_ship": 2, "T": 100, "demand_std": 0.5},
        "train": {"sample_budget": 10000},
    },
}

# Control scenarios (appendix — validate advantage boundaries)
SCENARIOS_APPENDIX = {
    "high_sample_stationary": {
        "desc": "CONTROL: High samples + stationary (SD ≈ MLP expected)",
        "env": {"tau_info": 2, "tau_ship": 2, "T": 36, "demand_std": 0.0},
        "train": {"sample_budget": 10000},
    },
    "short_horizon": {
        "desc": "CONTROL: Horizon = rollout depth (VNet redundant)",
        "env": {"tau_info": 2, "tau_ship": 2, "T": 20, "demand_std": 0.0},
        "train": {"sample_budget": 3000, "mcts_depth": 20},
    },
    "no_delay": {
        "desc": "CONTROL: Short delay (Greedy ≈ MCTS expected)",
        "env": {"tau_info": 1, "tau_ship": 1, "T": 36, "demand_std": 0.0},
        "train": {"sample_budget": 3000},
    },
}

VARIANTS = {
    "sd_mcts": {
        "desc": "Full SD-MCTS with value network",
        "model": "sd", "use_vnet": True, "use_mcts": True, "prior": None,
    },
    "blackbox_mlp": {
        "desc": "MLP world model + MCTS (no structural prior)",
        "model": "mlp", "use_vnet": True, "use_mcts": True, "prior": None,
    },
    "no_vnet": {
        "desc": "SD-MCTS without value network (MC rollout only)",
        "model": "sd", "use_vnet": False, "use_mcts": True, "prior": None,
    },
    "greedy_1step": {
        "desc": "SD model with greedy 1-step (no tree search)",
        "model": "sd", "use_vnet": False, "use_mcts": False, "prior": None,
    },
    "wrong_prior": {
        "desc": "SD with wrong demand prior (step time offset +10)",
        "model": "sd", "use_vnet": True, "use_mcts": True,
        "prior": {"demand_step_time_offset": 10},
    },
}

# Narrative guidance (not used to filter experiments)
ANALYSIS_HINTS = {
    ("blackbox_mlp", "low_sample_shift"): "SD >> MLP expected: structural prior helps in low-sample regime",
    ("no_vnet", "long_horizon_delay"): "VNet >> NoVNet expected: terminal value essential for long horizons",
    ("greedy_1step", "long_horizon_delay"): "MCTS >> Greedy expected: lookahead needed with large delays",
    ("wrong_prior", "long_horizon_delay"): "Wrong demand prior should hurt most under long horizons",
    ("blackbox_mlp", "high_sample_stationary"): "CONTROL: SD ≈ MLP with sufficient data",
    ("no_vnet", "short_horizon"): "CONTROL: VNet ≈ NoVNet when rollout covers horizon",
    ("greedy_1step", "no_delay"): "CONTROL: MCTS ≈ Greedy without delay",
}


@dataclass
class AblationResult:
    variant: str
    scenario: str
    seed: int
    component: str
    expected: str
    eval_return_mean: float = 0.0
    eval_return_std: float = 0.0
    train_time: float = 0.0
    total_steps: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# World Model Factories
# =============================================================================

def create_sd_model(env: BeerGameEnvFlat, prior: Optional[Dict] = None, device: str = 'cpu'):
    kwargs = {
        "order_levels": env.order_levels,
        "O_max": env.O_max,
        "tau_info": env.tau_info,
        "tau_ship": env.tau_ship,
    }
    if prior:
        if "demand_step_time_offset" in prior:
            kwargs["demand_step_time"] = env.demand_step_time + prior["demand_step_time_offset"]
    model = BeerGameSDModel(**kwargs).to(device)
    return BeerGameSDModelWrapper(model)


def create_mlp_model(env: BeerGameEnvFlat, hidden_dim: int = 256, device: str = 'cpu'):
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    return BlackBoxWorldModel(state_dim, n_actions, hidden_dim=hidden_dim).to(device)


# =============================================================================
# Training Functions
# =============================================================================

def train_and_eval(
    env: BeerGameEnvFlat,
    variant_config: Dict,
    scenario_config: Dict,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
    device: str = 'cpu',
) -> Tuple[float, float, float, int]:
    """Train and evaluate a single ablation configuration."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    sample_budget = scenario_config.get("train", {}).get("sample_budget", 3000)
    rollout_depth = scenario_config.get("train", {}).get("mcts_depth", 10)

    # Create world model
    if variant_config["model"] == "sd":
        world_model = create_sd_model(env, variant_config.get("prior"), device)
        sd_model = world_model.model
        model_opt = optim.Adam(sd_model.parameters(), lr=1e-2) if list(sd_model.parameters()) else None
        is_mlp = False
    else:
        world_model = create_mlp_model(env, device=device)
        model_opt = optim.Adam(world_model.parameters(), lr=1e-3)
        is_mlp = True

    # Value network
    value_net = None
    value_opt = None
    if variant_config["use_vnet"]:
        value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)
        value_opt = optim.Adam(value_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    return_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def get_return_stats():
        return return_stats['mean'], return_stats['std']

    normalized_vnet = NormalizedValueNet(value_net, get_return_stats) if value_net else None
    use_mcts = variant_config["use_mcts"]
    gamma = 0.99
    epsilon = 0.3
    random_steps = min(2000, max(500, sample_budget // 3))

    steps = 0
    episode = 0
    start_time = time.time()

    while steps < sample_budget:
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_transitions = []

        for t in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            elif use_mcts:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model,
                    n_actions=n_actions,
                    num_simulations=mcts_simulations,
                    value_net=normalized_vnet,
                    reward_fn=reward_fn,
                    rollout_depth=rollout_depth,
                )
            else:
                action = _greedy_action(state, world_model, n_actions, reward_fn, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            transition = Transition(
                state=state.copy(), action=action, reward=reward,
                next_state=next_state.copy(), done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated:
                break

        # Update return statistics
        G = 0.0
        returns = []
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        for r in returns:
            return_stats['count'] += 1
            delta = r - return_stats['mean']
            return_stats['mean'] += delta / return_stats['count']
            if return_stats['count'] > 1:
                return_stats['std'] = max(1.0, np.sqrt(
                    ((return_stats['count'] - 1) * return_stats['std']**2 + delta * (r - return_stats['mean']))
                    / return_stats['count']
                ))

        # Train world model
        if len(buffer) >= 32:
            for _ in range(5):
                batch = buffer.sample(32)
                states_t = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
                actions_t = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                next_states_t = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)

                if is_mlp:
                    pred_next = world_model(states_t, actions_t)
                    loss = F.mse_loss(pred_next, next_states_t)
                    model_opt.zero_grad()
                    loss.backward()
                    model_opt.step()
                elif model_opt:
                    pred_next = world_model.model(states_t, actions_t)
                    loss = F.mse_loss(pred_next, next_states_t)
                    model_opt.zero_grad()
                    loss.backward()
                    model_opt.step()

        # Train value network
        if value_opt and len(episode_transitions) > 0:
            ep_states = torch.tensor(
                np.array([t.state for t in episode_transitions]),
                dtype=torch.float32, device=device
            )
            normalized_returns = [(r - return_stats['mean']) / return_stats['std'] for r in returns]
            ep_returns = torch.tensor(normalized_returns, dtype=torch.float32, device=device)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode += 1

    train_time = time.time() - start_time

    # Evaluation
    eval_env_kwargs = {}
    if "eval" in scenario_config:
        eval_env_kwargs = scenario_config["eval"]

    eval_vnet = value_net if (value_net and not is_mlp) else normalized_vnet

    eval_returns = []
    for ep in range(eval_episodes):
        eval_env = BeerGameEnvFlat(
            order_levels=env.order_levels,
            tau_info=env.tau_info,
            tau_ship=env.tau_ship,
            T=env.T,
            demand_std=scenario_config.get("env", {}).get("demand_std", 0.0),
            demand_high=eval_env_kwargs.get("demand_high", env.demand_high) if eval_env_kwargs else env.demand_high,
            seed=1000 + ep,
        )

        state, _ = eval_env.reset(seed=1000 + ep)
        episode_return = 0.0
        done = False

        while not done:
            if use_mcts:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model,
                    n_actions=n_actions,
                    num_simulations=mcts_simulations,
                    value_net=eval_vnet,
                    reward_fn=reward_fn,
                    rollout_depth=rollout_depth,
                )
            else:
                action = _greedy_action(state, world_model, n_actions, reward_fn, device)

            state, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += reward
            done = terminated or truncated

        eval_returns.append(episode_return)

    return float(np.mean(eval_returns)), float(np.std(eval_returns)), train_time, steps


def _greedy_action(state: np.ndarray, world_model, n_actions: int, reward_fn, device: str) -> int:
    best_action = 0
    best_reward = float('-inf')

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for a in range(n_actions):
        action_tensor = torch.tensor([a], dtype=torch.int64, device=device)
        with torch.no_grad():
            if hasattr(world_model, 'model'):
                next_state = world_model.model(state_tensor, action_tensor).squeeze(0).cpu().numpy()
            else:
                next_state = world_model(state_tensor, action_tensor).squeeze(0).cpu().numpy()
        reward = reward_fn(next_state)
        if reward > best_reward:
            best_reward = reward
            best_action = a

    return best_action


# =============================================================================
# Job Generation
# =============================================================================

def generate_jobs(scenarios: dict, num_seeds: int, seed0: int = 0) -> List[Dict]:
    """Generate full cross-product jobs (all variants × all scenarios)."""
    jobs = []
    for scenario in scenarios:
        for variant in VARIANTS:
            for seed in range(seed0, seed0 + num_seeds):
                hint = ANALYSIS_HINTS.get((variant, scenario), "")
                jobs.append({
                    "variant": variant,
                    "scenario": scenario,
                    "seed": seed,
                    "hint": hint,
                })
    return jobs


# =============================================================================
# Main
# =============================================================================

def run_job(job: Dict, scenarios: dict, eval_episodes: int = 10) -> AblationResult:
    """Run a single ablation job."""
    variant = job["variant"]
    scenario = job["scenario"]
    seed = job["seed"]

    scenario_config = scenarios[scenario]
    variant_config = VARIANTS[variant]

    env_kwargs = {"order_levels": 3, "seed": seed}
    env_kwargs.update(scenario_config.get("env", {}))
    env = BeerGameEnvFlat(**env_kwargs)

    result = AblationResult(
        variant=variant,
        scenario=scenario,
        seed=seed,
        component=job.get("hint", ""),
        expected=job.get("hint", ""),
    )

    try:
        mean, std, train_time, steps = train_and_eval(
            env, variant_config, scenario_config, seed, eval_episodes
        )
        result.eval_return_mean = mean
        result.eval_return_std = std
        result.train_time = train_time
        result.total_steps = steps
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Ablation Study (Full Cross-Product)")
    parser.add_argument("--suite", choices=["core", "appendix", "all"], default="core",
                        help="core: 3 main scenarios; appendix: 3 control scenarios; all: both")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()

    if args.describe:
        print("=" * 70)
        print("ABLATION STUDY — FULL CROSS-PRODUCT DESIGN")
        print("=" * 70)

        print("\n=== CORE SCENARIOS (main table) ===")
        for name, cfg in SCENARIOS_CORE.items():
            print(f"  {name}: {cfg['desc']}")
        print(f"\n=== CONTROL SCENARIOS (appendix) ===")
        for name, cfg in SCENARIOS_APPENDIX.items():
            print(f"  {name}: {cfg['desc']}")
        print(f"\n=== VARIANTS ({len(VARIANTS)}) ===")
        for name, cfg in VARIANTS.items():
            print(f"  {name}: {cfg['desc']}")

        n_core = len(SCENARIOS_CORE) * len(VARIANTS)
        n_appendix = len(SCENARIOS_APPENDIX) * len(VARIANTS)
        print(f"\nCore table: {len(VARIANTS)} variants × {len(SCENARIOS_CORE)} scenarios = {n_core} configs")
        print(f"Appendix:   {len(VARIANTS)} variants × {len(SCENARIOS_APPENDIX)} scenarios = {n_appendix} configs")
        print(f"Total runs (5 seeds): {(n_core + n_appendix) * 5}")
        return

    # Select scenarios
    if args.suite == "core":
        scenarios = SCENARIOS_CORE
    elif args.suite == "appendix":
        scenarios = SCENARIOS_APPENDIX
    else:
        scenarios = {**SCENARIOS_CORE, **SCENARIOS_APPENDIX}

    jobs = generate_jobs(scenarios, args.seeds, args.seed0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else NEURIPS_DIR / "results" / "ablation" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"ABLATION STUDY — {args.suite.upper()}")
    print("=" * 70)
    print(f"Scenarios: {list(scenarios.keys())}")
    print(f"Variants:  {list(VARIANTS.keys())}")
    print(f"Jobs: {len(jobs)} ({len(VARIANTS)} variants × {len(scenarios)} scenarios × {args.seeds} seeds)")
    print(f"Output: {output_dir}", flush=True)
    print("=" * 70)

    results = []
    for i, job in enumerate(jobs):
        print(f"\n[{i+1}/{len(jobs)}] {job['variant']} | {job['scenario']} | seed={job['seed']}", flush=True)
        result = run_job(job, scenarios, args.eval_episodes)
        results.append(result.to_dict())

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Return: {result.eval_return_mean:.1f} ± {result.eval_return_std:.1f} | Time: {result.train_time:.1f}s", flush=True)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Cross-product summary table
    print_cross_product_table(results, scenarios)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


def print_cross_product_table(results: List[Dict], scenarios: dict):
    """Print full cross-product table (variants × scenarios)."""
    from collections import defaultdict

    # Aggregate by (variant, scenario) -> list of returns
    agg = defaultdict(list)
    for r in results:
        if not r.get("error"):
            agg[(r["variant"], r["scenario"])].append(r["eval_return_mean"])

    scenario_names = list(scenarios.keys())
    variant_names = list(VARIANTS.keys())

    # Header
    col_w = 22
    print(f"\n{'=' * 70}")
    print("CROSS-PRODUCT TABLE (Mean ± SE)")
    print(f"{'=' * 70}")
    header = f"{'Method':<20}"
    for s in scenario_names:
        header += f" {s:>{col_w}}"
    print(header)
    print("-" * (20 + (col_w + 1) * len(scenario_names)))

    for v in variant_names:
        row = f"{v:<20}"
        for s in scenario_names:
            vals = agg.get((v, s), [])
            if vals:
                mean = np.mean(vals)
                se = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                row += f" {mean:>+10.1f} ± {se:<7.1f}"
            else:
                row += f" {'N/A':>{col_w}}"
        print(row)

    # Paired t-test: each ablation vs sd_mcts
    print(f"\n{'=' * 70}")
    print("PAIRED T-TESTS vs sd_mcts (Bonferroni α = 0.05/{})".format(
        (len(variant_names) - 1) * len(scenario_names)))
    print(f"{'=' * 70}")

    from scipy import stats as scipy_stats
    n_comparisons = (len(variant_names) - 1) * len(scenario_names)
    alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    print(f"{'Variant':<20} {'Scenario':<22} {'Δ Mean':>10} {'Cohen d':>10} {'p-value':>10} {'Sig':>6}")
    print("-" * 80)

    for s in scenario_names:
        full_data = {}
        for r in results:
            if not r.get("error") and r["variant"] == "sd_mcts" and r["scenario"] == s:
                full_data[r["seed"]] = r["eval_return_mean"]

        for v in variant_names:
            if v == "sd_mcts":
                continue
            ablated_data = {}
            for r in results:
                if not r.get("error") and r["variant"] == v and r["scenario"] == s:
                    ablated_data[r["seed"]] = r["eval_return_mean"]

            common_seeds = sorted(set(full_data.keys()) & set(ablated_data.keys()))
            if len(common_seeds) < 2:
                continue

            full_vals = np.array([full_data[seed] for seed in common_seeds])
            ablated_vals = np.array([ablated_data[seed] for seed in common_seeds])
            diff = full_vals - ablated_vals
            delta = np.mean(diff)
            cohens_d = delta / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            t_stat, p_val = scipy_stats.ttest_rel(full_vals, ablated_vals)

            sig = "**" if p_val < alpha_corrected else ("*" if p_val < 0.05 else "")
            hint = ANALYSIS_HINTS.get((v, s), "")
            print(f"{v:<20} {s:<22} {delta:>+10.1f} {cohens_d:>10.2f} {p_val:>10.4f} {sig:>6}")
            if hint:
                print(f"  → {hint}")


if __name__ == "__main__":
    main()
