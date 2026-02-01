#!/usr/bin/env python3
"""Experiment A: Value Network Ablation (Theorem 2 Validation).

Compares SD-MCTS vs MCTS-Zero (no value network) vs MCTS-Random (untrained VNet)
across different search depths to validate effective horizon extension.

Expected results:
- MCTS-Zero: Step function, ~0% success for H < T_collapse, jumps to ~100% at H >= T_collapse
- SD-MCTS: Flat high curve, ~100% success even at small H (value network compensates)
"""
import json
import sys
import time
import copy
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.delayed_collapse_env import DelayedCollapseEnv
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.agent import PolicyNetwork, ValueNetwork
from src.planning import mcts_search


class RandomValueNetwork(nn.Module):
    """每次调用返回真正随机值的价值网络，用于 mcts_random 消融实验。"""

    def __init__(self, value_scale: float = 10.0):
        super().__init__()
        self.value_scale = value_scale

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0] if state.dim() > 1 else 1
        return torch.randn(batch_size) * self.value_scale


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_reward_fn(n_actions: int, F_req: float, B: float, F_collapse: float = None):
    def reward_fn(state: np.ndarray, action: int = None) -> float:
        fatigue, alive, surge_flag, time_to_surge = state
        # Collapse penalty
        if F_collapse is not None and fatigue >= F_collapse:
            return -0.375
        if alive < 0.5:
            return 0.0
        # Base reward = action intensity (trade-off: high intensity = high reward but high fatigue)
        if action is not None:
            a_t = action / (n_actions - 1)
        else:
            a_t = 0.5  # fallback
        reward = a_t
        # Surge bonus
        if time_to_surge < 1.5 and time_to_surge > 0.5 and fatigue <= F_req:
            reward += B
        return reward
    return reward_fn


def train_sd_mcts(
    env: DelayedCollapseEnv,
    true_F: float,
    init_F_belief: float = 50.0,
    env_steps: int = 10000,
) -> Tuple[DelayedCollapseSDModel, PolicyNetwork, ValueNetwork]:
    """Train SD-MCTS world model and networks (same as run_delayed_collapse.py)."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize with wrong belief about F (same as MPC)
    world_model = DelayedCollapseSDModel(
        n_actions=n_actions,
        init_alpha=env.alpha,
        init_beta=env.beta,
        init_F=init_F_belief,  # Wrong belief about collapse threshold
        F_req=env.F_req,
        T_s=env.T_s,
        B=env.B,
    )

    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)
    # Differential learning rates
    model_opt = optim.Adam([
        {'params': [world_model._alpha_raw], 'lr': 1e-2},
        {'params': [world_model._beta_raw], 'lr': 5e-2},  # Higher lr for beta
        {'params': [world_model._F_raw], 'lr': 5e-2},     # Higher lr for F convergence
    ])

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_reward_fn(n_actions, env.F_req, env.B)

    steps = 0
    episode = 0
    gamma = 0.99

    while steps < env_steps:
        state, _ = env.reset(seed=episode)
        episode_transitions = []

        for t in range(env.T):
            with torch.no_grad():
                # Ensure action diversity for beta gradient (a_t needs range [0,1])
                if steps < 2000:
                    # Early exploration: uniform random for diverse gradients
                    action = env.action_space.sample()
                elif np.random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    action = mcts_search(
                        state, world_model,
                        n_actions=n_actions,
                        num_simulations=20,
                        policy_net=policy_net,
                        value_net=value_net,
                        reward_fn=reward_fn,
                    )

            next_state, reward, terminated, truncated, info = env.step(action)
            steps += 1

            transition = Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated:
                break

        # Compute returns for value learning
        returns = []
        G = 0.0
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        # Training updates
        for _ in range(5):
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                # Filter out post-collapse transitions (absorbing state)
                batch = [t for t in batch if t.state[1] > 0.5]
                if len(batch) < 16:
                    continue

                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)

                # World model loss: predict next state
                pred_next = world_model(states, actions)
                model_loss = F.mse_loss(pred_next, next_states)

                # Add supervision for collapse threshold learning
                alive_current = states[:, 1]
                alive_next = next_states[:, 1]
                fatigue_next = next_states[:, 0]
                collapse_mask = (alive_current > 0.5) & (alive_next < 0.5)

                if collapse_mask.any():
                    collapse_fatigue = fatigue_next[collapse_mask]
                    F_learned = world_model.collapse_threshold
                    collapse_loss = F.mse_loss(
                        F_learned.expand(collapse_fatigue.shape[0]),
                        collapse_fatigue
                    )
                    model_loss = model_loss + 0.5 * collapse_loss

                model_opt.zero_grad()
                model_loss.backward()
                model_opt.step()

        # Value network training on episode returns
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(
                np.array([t.state for t in episode_transitions]),
                dtype=torch.float32
            )
            ep_returns = torch.tensor(returns, dtype=torch.float32)

            value_pred = value_net(ep_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, ep_returns)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

        episode += 1

        if steps % 2000 == 0 or steps >= env_steps:
            F_learned = world_model.collapse_threshold.item()
            a_safe = world_model.get_safe_intensity()
            print(f"  Step {steps}: F_learned={F_learned:.2f} (true={true_F}), a_safe={a_safe:.3f}")

    return world_model, policy_net, value_net


def run_episode(
    env: DelayedCollapseEnv,
    world_model: DelayedCollapseSDModel,
    policy_net: PolicyNetwork,
    value_net: Optional[ValueNetwork],
    num_simulations: int,
    seed: int,
    rollout_depth: int = 10,
) -> Tuple[float, bool, bool]:
    """Run single episode and return (reward, collapsed, surge_captured)."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    collapsed = False
    surge_captured = False

    reward_fn = make_reward_fn(env.n_actions, env.F_req, env.B,
                               F_collapse=world_model.collapse_threshold.item())

    for step in range(env.T):
        with torch.no_grad():
            action = mcts_search(
                state, world_model,
                n_actions=env.n_actions,
                num_simulations=num_simulations,
                policy_net=policy_net,
                value_net=value_net,
                reward_fn=reward_fn,
                rollout_depth=rollout_depth,
            )

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get("collapsed"):
            collapsed = True
        if info.get("surge_bonus_earned"):
            surge_captured = True

        state = next_state
        if terminated or truncated:
            break

    return total_reward, collapsed, surge_captured


def main():
    parser = argparse.ArgumentParser(description="VNet Ablation Experiment (Theorem 2)")
    parser.add_argument("--h-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14])
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--results-dir", type=str, default="results/vnet_ablation")
    args = parser.parse_args()

    # Environment config - Original successful configuration
    true_F = 10.0
    belief_F = 50.0
    alpha = 1.0
    beta = 0.1
    F_req = 5.0
    T_s = 12       # Surge timing
    B = 15.0
    T = 30

    print("=" * 70)
    print("Experiment A: Value Network Ablation (Theorem 2 Validation)")
    print("=" * 70)
    print(f"H values: {args.h_values}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Training steps: {args.training_steps}")
    print(f"T_collapse (true F): {true_F}, T_surge: {T_s}")
    print()

    results = {
        "experiment": "vnet_ablation",
        "config": {
            "h_values": args.h_values,
            "num_seeds": args.num_seeds,
            "true_F": true_F,
            "belief_F": belief_F,
            "T_s": T_s,
            "training_steps": args.training_steps,
        },
        "runs": [],
        "summary": {},
    }

    # Train networks
    print("Training SD-MCTS networks...")
    set_seed(42)
    env = DelayedCollapseEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        T_s=T_s, B=B, T=T, seed=42
    )
    world_model, policy_net, value_net = train_sd_mcts(
        env, true_F=true_F, init_F_belief=belief_F, env_steps=args.training_steps
    )

    F_learned = world_model.collapse_threshold.item()
    print(f"\nLearned F: {F_learned:.2f} (true: {true_F})")
    print(f"Learned alpha: {world_model.alpha.item():.3f} (true: {alpha})")
    print(f"Learned beta: {world_model.beta.item():.3f} (true: {beta})")
    print()

    # Create random value network (returns truly random values each call)
    random_value_net = RandomValueNetwork(value_scale=10.0)

    # Methods with their (value_net, rollout_depth) config
    # Use short rollout_depth=3, so T_surge=12 is beyond direct rollout visibility
    # Key: sd_mcts has value network to estimate long-term value despite short rollout
    methods = {
        "sd_mcts": (value_net, 3),
        "mcts_zero": (None, 3),
        "mcts_random": (random_value_net, 3),
    }

    for H in args.h_values:
        print(f"Testing H={H}...")
        for method_name, (vnet, rollout_d) in methods.items():
            successes = 0
            rewards = []

            for seed in range(args.num_seeds):
                eval_env = DelayedCollapseEnv(
                    alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                    T_s=T_s, B=B, T=T, seed=seed + 1000
                )

                total_reward, collapsed, surge_captured = run_episode(
                    eval_env, world_model, policy_net, vnet,
                    num_simulations=max(H * 5, 20),
                    seed=seed + 1000,
                    rollout_depth=rollout_d,
                )

                success = (not collapsed) and surge_captured
                successes += int(success)
                rewards.append(total_reward)

                results["runs"].append({
                    "method": method_name,
                    "H": H,
                    "seed": seed,
                    "reward": total_reward,
                    "success": success,
                    "collapsed": collapsed,
                    "surge_captured": surge_captured,
                })

            success_rate = successes / args.num_seeds
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            key = f"{method_name}_H{H}"
            results["summary"][key] = {
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
            }

            print(f"  {method_name}: success={success_rate:.1%}, reward={mean_reward:.2f}±{std_reward:.2f}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"vnet_ablation_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
