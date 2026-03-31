#!/usr/bin/env python3
"""Test sd_mcts on long_horizon scenario with detailed logging."""

import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent.networks import ValueNetwork
from src.planning.mcts import mcts_search
from experiments.run_beer_game_v2 import (
    BeerGameSDModelWrapper, make_beer_game_reward_fn,
    NormalizedValueNet, ReplayBuffer, Transition,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30000, help="Sample budget")
    parser.add_argument("--horizon", type=int, default=100, help="Episode length T")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcts-sims", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    print(f"Testing sd_mcts on long_horizon (T={args.horizon}) with sample_budget={args.budget}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BeerGameEnvFlat(order_levels=3, tau_info=2, tau_ship=2, T=args.horizon,
                          demand_std=0.0, seed=args.seed)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = BeerGameSDModel(order_levels=env.order_levels, O_max=env.O_max)
    world_model_wrapper = BeerGameSDModelWrapper(world_model)

    # World model optimizer (for learnable demand parameters)
    wm_params = list(world_model.parameters())
    model_opt = optim.Adam(wm_params, lr=1e-2) if wm_params else None

    value_net = ValueNetwork(state_dim, hidden_dim=128)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    return_stats = {"mean": 0.0, "std": 1.0, "count": 0}
    def get_return_stats():
        return return_stats["mean"], return_stats["std"]
    normalized_vnet = NormalizedValueNet(value_net, get_return_stats)

    gamma = 0.99
    epsilon = 0.2  # Match delayed_collapse: 20% exploration after warmup
    random_steps = min(1000, args.budget // 5)  # Match delayed_collapse: 1000 steps warmup

    steps = 0
    episode = 0
    episode_rewards = []
    start_time = time.time()

    while steps < args.budget:
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        episode_transitions = []
        ep_start = time.time()

        for t in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state, world_model=world_model_wrapper,
                    n_actions=n_actions, num_simulations=args.mcts_sims,
                    value_net=normalized_vnet, reward_fn=reward_fn,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            transition = Transition(state=state.copy(), action=action, reward=reward,
                                    next_state=next_state.copy(), done=terminated or truncated)
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated or steps >= args.budget:
                break

        # Train world model
        if model_opt and len(buffer) >= 32:
            for _ in range(5):
                batch = buffer.sample(32)
                states_t = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions_t = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states_t = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)

                pred_next = world_model(states_t, actions_t)
                loss = F.mse_loss(pred_next, next_states_t)
                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

        # Compute returns for statistics tracking
        G = 0.0
        returns = []
        for tr in reversed(episode_transitions):
            G = tr.reward + gamma * G
            returns.insert(0, G)
        for r in returns:
            return_stats["count"] += 1
            delta = r - return_stats["mean"]
            return_stats["mean"] += delta / return_stats["count"]
            if return_stats["count"] > 1:
                return_stats["std"] = max(1.0, np.sqrt(
                    ((return_stats["count"]-1)*return_stats["std"]**2 + delta*(r-return_stats["mean"]))
                    / return_stats["count"]))

        # Train value network with normalized MC returns (required for NormalizedValueNet)
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(np.array([t.state for t in episode_transitions]), dtype=torch.float32)
            # Normalize returns for stable learning (value_net outputs normalized values)
            normalized_returns = [(r - return_stats["mean"]) / return_stats["std"] for r in returns]
            ep_returns = torch.tensor(normalized_returns, dtype=torch.float32)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode_rewards.append(episode_reward)
        episode += 1
        ep_time = time.time() - ep_start

        # Compute world model prediction error for diagnosis
        wm_error = 0.0
        if len(buffer) >= 32:
            diag_batch = buffer.sample(32)
            with torch.no_grad():
                ds = torch.tensor(np.array([t.state for t in diag_batch]), dtype=torch.float32)
                da = torch.tensor([t.action for t in diag_batch], dtype=torch.int64)
                dns = torch.tensor(np.array([t.next_state for t in diag_batch]), dtype=torch.float32)
                pred = world_model(ds, da)
                wm_error = F.mse_loss(pred, dns).item()

        print(f"[Ep {episode:3d}] steps={steps:5d}/{args.budget} | return={episode_reward:8.1f} | "
              f"avg10={np.mean(episode_rewards[-10:]):8.1f} | wm_err={wm_error:.3f} | "
              f"time={ep_time:.1f}s | elapsed={time.time()-start_time:.0f}s")

    train_time = time.time() - start_time
    print("=" * 70)
    print(f"Training: {episode} episodes, {steps} steps, {train_time:.1f}s")

    print("\nEvaluating...")
    eval_returns = []
    for ep in range(args.eval_episodes):
        state, _ = env.reset(seed=1000 + ep)
        episode_return = 0.0
        done = False
        while not done:
            action = mcts_search(root_state=state, world_model=world_model_wrapper,
                                n_actions=n_actions, num_simulations=args.mcts_sims,
                                value_net=normalized_vnet, reward_fn=reward_fn)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        eval_returns.append(episode_return)
        print(f"  Eval {ep+1}: {episode_return:.1f}")

    print("=" * 70)
    print(f"Result: {np.mean(eval_returns):.1f} ± {np.std(eval_returns):.1f}")


if __name__ == "__main__":
    main()
