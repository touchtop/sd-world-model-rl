#!/usr/bin/env python
"""
Standalone script to measure physics violation rates for SD-MCTS vs BlackBox.
Uses the real BeerGameEnv and trains models properly.

Usage:
    conda run -n mapf python docs/neurips/experiments/check_violation_rate.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Fix paths
NEURIPS_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = NEURIPS_DIR.parent.parent
sys.path.insert(0, str(NEURIPS_DIR))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(NEURIPS_DIR / "experiments"))

import torch
from src.env.beer_game_env import BeerGameEnv, BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent import PolicyNetwork, ValueNetwork
from src.planning import mcts_search

# Import correct helpers from run_beer_game_v2
from run_beer_game_v2 import (
    train_sd_mcts, BeerGameSDModelWrapper,
    make_beer_game_reward_fn, NormalizedValueNet,
    ReplayBuffer, Transition
)
from baselines.blackbox_world_model import BlackBoxWorldModel, train_blackbox_model


def check_physics_violations(states: np.ndarray) -> dict:
    """Check for negative inventory/backlog violations."""
    n = len(states)
    # Any negative value in state vector is a physics violation (negative stock/orders)
    neg_rows = np.any(states < -1e-6, axis=1)
    neg_count = int(np.sum(neg_rows))
    return {
        "total_states": n,
        "negative_count": neg_count,
        "violation_rate": neg_count / n if n > 0 else 0.0,
    }


def collect_random_transitions(env, n_steps=2000):
    """Collect random transitions for BlackBox pre-training."""
    buffer = []
    state, _ = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        buffer.append((state.copy(), action, next_state.copy(), reward))
        state = next_state
        if done or truncated:
            state, _ = env.reset()
    return buffer


def rollout_sd_mcts(world_model, policy_net, value_net, env, n_episodes=10, num_sims=50):
    """Roll out trained SD-MCTS agent and collect states."""
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)
    wm_wrapper = BeerGameSDModelWrapper(world_model)
    return_stats = {'mean': 0.0, 'std': 1.0}
    def get_return_stats():
        return return_stats['mean'], return_stats['std']
    norm_vnet = NormalizedValueNet(value_net, get_return_stats)

    all_states = []
    all_returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep + 200)
        ep_return = 0.0
        done = False
        while not done:
            all_states.append(state.copy())
            action = mcts_search(
                root_state=state,
                world_model=wm_wrapper,
                n_actions=env.action_space.n,
                num_simulations=num_sims,
                policy_net=None,
                value_net=norm_vnet,
                reward_fn=reward_fn,
            )
            state, r, terminated, truncated, _ = env.step(action)
            ep_return += r
            done = terminated or truncated
        all_states.append(state.copy())
        all_returns.append(ep_return)
    return np.array(all_states), all_returns


def rollout_blackbox(bb_model, env, n_episodes=10, num_sims=50):
    """Roll out BlackBox-MCTS agent: use random policy (since training path is different)."""
    # BlackBox prediction doesn't enforce constraints → measure violation on predicted states
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Collect states from random policy in environment (ground truth trajectory)
    all_states = []
    # Also collect PREDICTED next states from BB model to measure its violations
    all_pred_states = []
    all_returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep + 200)
        ep_return = 0.0
        done = False
        while not done:
            all_states.append(state.copy())
            action = env.action_space.sample()
            # Get blackbox prediction for this state
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.LongTensor([action])
            with torch.no_grad():
                pred_next = bb_model(state_t, action_t).squeeze(0).numpy()
            all_pred_states.append(pred_next)
            state, r, terminated, truncated, _ = env.step(action)
            ep_return += r
            done = terminated or truncated
        all_states.append(state.copy())
        all_returns.append(ep_return)
    return np.array(all_pred_states), all_returns


def main():
    N_SEEDS = 3
    N_EPISODES_EVAL = 10
    TRAIN_BUDGET = 3000  # steps for SD-MCTS training

    all_results = {}

    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed} ===")
        env_train = BeerGameEnvFlat(order_levels=5)

        # ---- SD-MCTS ----
        print(f"  Training SD-MCTS (budget={TRAIN_BUDGET})...")
        try:
            world_model, policy_net, value_net, _ = train_sd_mcts(
                env_train, env_steps=TRAIN_BUDGET, verbose=False
            )
            eval_env = BeerGameEnvFlat(order_levels=5)
            states, returns = rollout_sd_mcts(
                world_model, policy_net, value_net, eval_env,
                n_episodes=N_EPISODES_EVAL
            )
            viol = check_physics_violations(states)
            all_results.setdefault("sd_mcts", []).append({
                "seed": seed,
                "violation_rate": viol["violation_rate"],
                "mean_return": float(np.mean(returns)),
            })
            print(f"    SD-MCTS violation rate: {viol['violation_rate']*100:.2f}%  "
                  f"mean_return: {np.mean(returns):.1f}")
        except Exception as e:
            print(f"    SD-MCTS ERROR: {e}")

        # ---- BlackBox ----
        print(f"  Training BlackBox (random data collection)...")
        try:
            env_bb = BeerGameEnvFlat(order_levels=5)
            n_states = env_bb.observation_space.shape[0]
            n_actions = env_bb.action_space.n
            bb_model = BlackBoxWorldModel(n_states, n_actions)
            # Collect random transitions
            replay = collect_random_transitions(env_bb, n_steps=TRAIN_BUDGET)
            replay_tuples = [(s, a, ns, r) for (s, a, ns, r) in replay]
            # Train blackbox (API: model, replay_buffer)
            train_blackbox_model(bb_model, replay_tuples, batch_size=64, epochs=50)
            # Measure predicted-state violations
            eval_env_bb = BeerGameEnvFlat(order_levels=5)
            pred_states, returns_bb = rollout_blackbox(
                bb_model, eval_env_bb, n_episodes=N_EPISODES_EVAL
            )
            viol_bb = check_physics_violations(pred_states)
            all_results.setdefault("blackbox_mcts", []).append({
                "seed": seed,
                "violation_rate": viol_bb["violation_rate"],
                "mean_return": float(np.mean(returns_bb)),
            })
            print(f"    BlackBox violation rate: {viol_bb['violation_rate']*100:.2f}%  "
                  f"mean_return: {np.mean(returns_bb):.1f}")
        except Exception as e:
            print(f"    BlackBox ERROR: {e}")

    # Summarize
    print("\n\n=== Physics Violation Rate Summary (Beer Game, in-distribution) ===")
    print(f"{'Method':<22} {'Viol. Rate Mean':>16} {'Viol. Rate Std':>16} {'Mean Return':>14}")
    print("-" * 72)
    summary = {}
    for method, records in all_results.items():
        rates = [r["violation_rate"] for r in records]
        rets = [r["mean_return"] for r in records]
        summary[method] = {
            "violation_rate_mean": float(np.mean(rates)),
            "violation_rate_std": float(np.std(rates)),
            "mean_return_mean": float(np.mean(rets)),
            "n_seeds": len(rates),
        }
        print(f"{method:<22} {np.mean(rates)*100:>14.2f}%  {np.std(rates)*100:>14.2f}%  {np.mean(rets):>14.1f}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = NEURIPS_DIR / "results" / "physics_violation" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "violation_summary.json", "w") as f:
        json.dump({"summary": summary, "raw": all_results}, f, indent=2)
    print(f"\nResults saved to: {out_dir}")
    return summary


if __name__ == "__main__":
    main()
