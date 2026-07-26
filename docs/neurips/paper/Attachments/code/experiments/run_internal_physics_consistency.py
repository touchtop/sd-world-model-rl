#!/usr/bin/env python3
"""Formal physics-consistency experiment for Table 2.

This script measures physics violations on *search-tree internal imagined
states* under a controlled SD-MCTS vs BB-MCTS comparison:

- same Beer Game environment and training budget
- same MCTS planner
- same value-network architecture and usage
- only the world model structure changes (SD vs black-box MLP)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = NEURIPS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NEURIPS_DIR))

from src.agent.networks import ValueNetwork
from src.env.beer_game_env import BeerGameEnvFlat
from src.planning.mcts import mcts_search

from baselines.blackbox_world_model import train_blackbox_model
from experiments.run_ablation_principled import create_mlp_model, create_sd_model
from experiments.beer_game_common import (
    NormalizedValueNet,
    ReplayBuffer,
    Transition,
    make_beer_game_reward_fn,
)


DEFAULT_CONFIG = {
    "sample_budget": 3000,
    "n_seeds": 3,
    "eval_episodes": 10,
    "mcts_simulations": 200,
    "rollout_depth": 10,
    "random_steps_min": 500,
    "epsilon": 0.3,
    "env": {
        "order_levels": 3,
        "tau_info": 2,
        "tau_ship": 2,
        "T": 36,
        "demand_std": 0.0,
    },
}


@dataclass
class SeedResult:
    method: str
    seed: int
    sample_budget: int
    eval_return_mean: float
    eval_return_std: float
    imagined_state_count: int
    imagined_state_violation_rate: float
    train_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "sample_budget": self.sample_budget,
            "eval_return_mean": self.eval_return_mean,
            "eval_return_std": self.eval_return_std,
            "imagined_state_count": self.imagined_state_count,
            "imagined_state_violation_rate": self.imagined_state_violation_rate,
            "train_time_sec": self.train_time_sec,
        }


def create_env(seed: int) -> BeerGameEnvFlat:
    return BeerGameEnvFlat(
        order_levels=DEFAULT_CONFIG["env"]["order_levels"],
        tau_info=DEFAULT_CONFIG["env"]["tau_info"],
        tau_ship=DEFAULT_CONFIG["env"]["tau_ship"],
        T=DEFAULT_CONFIG["env"]["T"],
        demand_std=DEFAULT_CONFIG["env"]["demand_std"],
        seed=seed,
    )


def check_imagined_state_violations(states: np.ndarray) -> Dict[str, float]:
    """Check whether imagined states violate non-negativity constraints.

    Only the first 24 dimensions are physical stock/backlog/pipeline variables.
    The final two dimensions are timestep and current demand metadata.
    """
    if len(states) == 0:
        return {
            "total_states": 0,
            "violating_states": 0,
            "violation_rate": 0.0,
        }

    physical = states[:, :24]
    violating_rows = np.any(physical < -1e-6, axis=1)
    violating_states = int(np.sum(violating_rows))
    total_states = int(len(states))
    return {
        "total_states": total_states,
        "violating_states": violating_states,
        "violation_rate": violating_states / total_states,
    }


def build_models(method: str, env: BeerGameEnvFlat, device: str) -> Tuple[Any, Any, Any, bool]:
    """Create world model, optimizer, value net, optimizer."""
    if method == "sd_mcts":
        world_model = create_sd_model(env, prior=None, device=device)
        sd_model = world_model.model
        model_opt = optim.Adam(sd_model.parameters(), lr=1e-2) if list(sd_model.parameters()) else None
        is_mlp = False
    elif method == "bb_mcts":
        world_model = create_mlp_model(env, device=device)
        model_opt = optim.Adam(world_model.parameters(), lr=1e-3)
        is_mlp = True
    else:
        raise ValueError(f"Unknown method: {method}")

    state_dim = env.observation_space.shape[0]
    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    return world_model, model_opt, value_net, value_opt, is_mlp


def train_method(
    method: str,
    seed: int,
    sample_budget: int,
    mcts_simulations: int,
    rollout_depth: int,
    device: str = "cpu",
):
    """Train one method under a controlled protocol shared across methods."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = create_env(seed)
    world_model, model_opt, value_net, value_opt, is_mlp = build_models(method, env, device)

    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)
    n_actions = env.action_space.n
    gamma = 0.99
    epsilon = DEFAULT_CONFIG["epsilon"]
    random_steps = min(2000, max(DEFAULT_CONFIG["random_steps_min"], sample_budget // 3))

    return_stats = {"mean": 0.0, "std": 1.0, "count": 0}

    def get_return_stats():
        return return_stats["mean"], return_stats["std"]

    normalized_vnet = NormalizedValueNet(value_net, get_return_stats)
    buffer = ReplayBuffer(capacity=5000)

    steps = 0
    episode = 0
    start_time = time.time()

    while steps < sample_budget:
        state, _ = env.reset(seed=seed + episode)
        episode_transitions: List[Transition] = []

        for _ in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model,
                    n_actions=n_actions,
                    num_simulations=mcts_simulations,
                    value_net=normalized_vnet,
                    reward_fn=reward_fn,
                    rollout_depth=rollout_depth,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            transition = Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)
            steps += 1
            state = next_state

            if terminated or truncated or steps >= sample_budget:
                break

        # Update return normalization statistics.
        G = 0.0
        returns: List[float] = []
        for transition in reversed(episode_transitions):
            G = transition.reward + gamma * G
            returns.insert(0, G)

        for ret in returns:
            return_stats["count"] += 1
            delta = ret - return_stats["mean"]
            return_stats["mean"] += delta / return_stats["count"]
            if return_stats["count"] > 1:
                return_stats["std"] = max(
                    1.0,
                    np.sqrt(
                        (
                            (return_stats["count"] - 1) * return_stats["std"] ** 2
                            + delta * (ret - return_stats["mean"])
                        )
                        / return_stats["count"]
                    ),
                )

        if len(buffer) >= 32:
            for _ in range(5):
                batch = buffer.sample(32)
                states_t = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
                actions_t = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                next_states_t = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)

                if is_mlp:
                    pred_next = world_model(states_t, actions_t)
                else:
                    pred_next = world_model.model(states_t, actions_t)
                model_loss = F.mse_loss(pred_next, next_states_t)
                model_opt.zero_grad()
                model_loss.backward()
                model_opt.step()

        if len(episode_transitions) > 0:
            ep_states = torch.tensor(
                np.array([t.state for t in episode_transitions]),
                dtype=torch.float32,
                device=device,
            )
            normalized_returns = [
                (ret - return_stats["mean"]) / return_stats["std"] for ret in returns
            ]
            ep_returns = torch.tensor(normalized_returns, dtype=torch.float32, device=device)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode += 1

    train_time = time.time() - start_time
    return world_model, normalized_vnet, train_time


def evaluate_method(
    world_model,
    value_net,
    sample_budget: int,
    mcts_simulations: int,
    rollout_depth: int,
    eval_episodes: int,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate method and collect internal imagined-state violations."""
    base_env = create_env(seed=0)
    reward_fn = make_beer_game_reward_fn(h=base_env.h, p=base_env.p, n_echelons=base_env.n_echelons)
    n_actions = base_env.action_space.n

    all_imagined_states: List[np.ndarray] = []
    eval_returns: List[float] = []

    for episode in range(eval_episodes):
        env = create_env(seed=1000 + episode)
        state, _ = env.reset(seed=1000 + episode)
        episode_return = 0.0
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=world_model,
                n_actions=n_actions,
                num_simulations=mcts_simulations,
                value_net=value_net,
                reward_fn=reward_fn,
                rollout_depth=rollout_depth,
                imagined_state_recorder=all_imagined_states.append,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        eval_returns.append(episode_return)

    imagined_stats = check_imagined_state_violations(np.array(all_imagined_states, dtype=np.float32))
    return float(np.mean(eval_returns)), float(np.std(eval_returns)), imagined_stats


def summarize_results(results: List[SeedResult]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for method in sorted({result.method for result in results}):
        method_results = [result for result in results if result.method == method]
        returns = np.array([result.eval_return_mean for result in method_results], dtype=np.float64)
        violations = np.array(
            [result.imagined_state_violation_rate for result in method_results],
            dtype=np.float64,
        )
        counts = np.array([result.imagined_state_count for result in method_results], dtype=np.int64)
        n = len(method_results)
        summary[method] = {
            "n_seeds": n,
            "sample_budget": method_results[0].sample_budget if method_results else 0,
            "imagined_state_count_mean": float(np.mean(counts)) if n else 0.0,
            "imagined_state_violation_rate_mean": float(np.mean(violations)) if n else 0.0,
            "imagined_state_violation_rate_std": float(np.std(violations, ddof=1)) if n > 1 else 0.0,
            "imagined_state_violation_rate_se": float(np.std(violations, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "eval_return_mean": float(np.mean(returns)) if n else 0.0,
            "eval_return_std": float(np.std(returns, ddof=1)) if n > 1 else 0.0,
            "eval_return_se": float(np.std(returns, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        }
    return summary


def run_experiment(
    sample_budget: int,
    n_seeds: int,
    eval_episodes: int,
    mcts_simulations: int,
    rollout_depth: int,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[SeedResult] = []

    for method in ("sd_mcts", "bb_mcts"):
        for seed in range(n_seeds):
            print(
                f"[{method}] seed={seed} | budget={sample_budget} | "
                f"eval_eps={eval_episodes} | mcts_sims={mcts_simulations}",
                flush=True,
            )
            world_model, normalized_vnet, train_time = train_method(
                method=method,
                seed=seed,
                sample_budget=sample_budget,
                mcts_simulations=mcts_simulations,
                rollout_depth=rollout_depth,
            )
            eval_mean, eval_std, imagined_stats = evaluate_method(
                world_model=world_model,
                value_net=normalized_vnet,
                sample_budget=sample_budget,
                mcts_simulations=mcts_simulations,
                rollout_depth=rollout_depth,
                eval_episodes=eval_episodes,
            )
            seed_result = SeedResult(
                method=method,
                seed=seed,
                sample_budget=sample_budget,
                eval_return_mean=eval_mean,
                eval_return_std=eval_std,
                imagined_state_count=imagined_stats["total_states"],
                imagined_state_violation_rate=imagined_stats["violation_rate"],
                train_time_sec=train_time,
            )
            results.append(seed_result)
            print(
                f"  return={eval_mean:.1f} +/- {eval_std:.1f} | "
                f"viol={imagined_stats['violation_rate'] * 100:.2f}% | "
                f"states={imagined_stats['total_states']}",
                flush=True,
            )

    payload = {
        "config": {
            "sample_budget": sample_budget,
            "n_seeds": n_seeds,
            "eval_episodes": eval_episodes,
            "mcts_simulations": mcts_simulations,
            "rollout_depth": rollout_depth,
            "env": DEFAULT_CONFIG["env"],
        },
        "results": [result.to_dict() for result in results],
        "summary": summarize_results(results),
    }
    with open(output_dir / "results.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Internal physics consistency experiment")
    parser.add_argument("--budget", type=int, default=DEFAULT_CONFIG["sample_budget"])
    parser.add_argument("--seeds", type=int, default=DEFAULT_CONFIG["n_seeds"])
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_CONFIG["eval_episodes"])
    parser.add_argument("--mcts-sims", type=int, default=DEFAULT_CONFIG["mcts_simulations"])
    parser.add_argument("--rollout-depth", type=int, default=DEFAULT_CONFIG["rollout_depth"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output)
        if args.output
        else NEURIPS_DIR / "results" / "physics_consistency" / timestamp
    )

    payload = run_experiment(
        sample_budget=args.budget,
        n_seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        mcts_simulations=args.mcts_sims,
        rollout_depth=args.rollout_depth,
        output_dir=output_dir,
    )

    print(f"\nSaved results to: {output_dir}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
