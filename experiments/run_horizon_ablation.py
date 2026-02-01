#!/usr/bin/env python3
"""Experiment: True Horizon Ablation (Theorem 2 Validation).

This experiment correctly validates Theorem 2 by explicitly controlling
the MCTS tree search depth H, rather than simulation count.

Key difference from run_vnet_ablation.py:
- Fixed num_simulations (computational budget)
- Variable max_depth (planning horizon H)
- Tree expansion stops at depth H, then leaf evaluation

Leaf node evaluation strategies:
- SD-MCTS: Trained value network V_ψ(s) estimates future returns
- MCTS-Zero: V = 0 (pure tree search, no lookahead beyond depth H)
- MCTS-Random: Random values (noise baseline)

Expected results:
- MCTS-Zero: Step function at H ≈ T_surge (needs explicit search to "see" surge)
- SD-MCTS: Flat high curve (value network extends effective horizon beyond H)
- MCTS-Random: Low performance (random values provide no useful guidance)
"""
import json
import sys
import time
import copy
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.delayed_collapse_env import DelayedCollapseEnv
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.agent import PolicyNetwork, ValueNetwork


class RandomValueNetwork(nn.Module):
    """Random value network for ablation."""

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
        if F_collapse is not None and fatigue >= F_collapse:
            return -0.375
        if alive < 0.5:
            return 0.0
        if action is not None:
            a_t = action / (n_actions - 1)
        else:
            a_t = 0.5
        reward = a_t
        if time_to_surge < 1.5 and time_to_surge > 0.5 and fatigue <= F_req:
            reward += B
        return reward
    return reward_fn


# ============================================================================
# Depth-Limited MCTS Implementation
# ============================================================================

class MCTSNode:
    """MCTS node with depth tracking."""

    __slots__ = ['state', 'parent', 'action', 'children', 'N', 'W', 'Q', 'P', 'reward', 'depth']

    def __init__(self, state: torch.Tensor, parent: 'MCTSNode' = None,
                 action: int = None, prior: float = 0.0, reward: float = 0.0, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, MCTSNode] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior
        self.reward = reward
        self.depth = depth

    def is_fully_expanded(self, n_actions: int) -> bool:
        return len(self.children) == n_actions


def _puct_score(parent: MCTSNode, child: MCTSNode, c: float) -> float:
    exploration = c * child.P * math.sqrt(parent.N + 1) / (1 + child.N)
    return child.Q + exploration


def depth_limited_mcts_search(
    root_state: np.ndarray,
    world_model,
    n_actions: int,
    num_simulations: int = 100,
    max_depth: int = 10,
    c: float = 1.4,
    gamma: float = 0.99,
    reward_fn: Callable = None,
    policy_net=None,
    value_net=None,
) -> int:
    """MCTS with explicit depth limit for Theorem 2 validation.

    Args:
        max_depth: Maximum tree search depth H. Tree expansion stops here.
        num_simulations: Fixed computational budget (same for all H).
        value_net: If provided, used for leaf evaluation; otherwise rollout.
    """
    root_tensor = torch.tensor(root_state, dtype=torch.float32)
    root = MCTSNode(root_tensor, depth=0)

    for sim in range(num_simulations):
        node = root
        path = [node]

        # Selection: traverse tree using PUCT until leaf or depth limit
        while node.is_fully_expanded(n_actions) and node.children and node.depth < max_depth:
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Check if at depth limit
        at_depth_limit = (node.depth >= max_depth)

        # Expansion: only if not at depth limit
        if not at_depth_limit and not node.is_fully_expanded(n_actions):
            if policy_net is not None:
                with torch.no_grad():
                    node_priors = F.softmax(policy_net(node.state.unsqueeze(0)), dim=-1)[0].numpy()
            else:
                node_priors = np.ones(n_actions) / n_actions

            unexpanded = [a for a in range(n_actions) if a not in node.children]
            states_batch = node.state.unsqueeze(0).expand(len(unexpanded), -1)
            actions_batch = torch.tensor(unexpanded, dtype=torch.int64)

            with torch.no_grad():
                next_states = world_model(states_batch, actions_batch)

            for i, action in enumerate(unexpanded):
                child_reward = reward_fn(next_states[i].detach().numpy(), action)
                child = MCTSNode(
                    next_states[i], parent=node, action=action,
                    prior=node_priors[action], reward=child_reward,
                    depth=node.depth + 1
                )
                node.children[action] = child

            # Select best child for evaluation
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Leaf evaluation
        # Option A: value_net provides estimate, otherwise 0 (pure tree search)
        if value_net is not None:
            with torch.no_grad():
                leaf_value = value_net(node.state.unsqueeze(0)).item()
        else:
            # Zero terminal value - tests pure tree depth without lookahead
            leaf_value = 0.0

        # Backpropagation
        G = leaf_value
        for node in reversed(path):
            if node.parent is not None:
                G = node.reward + gamma * G
            node.N += 1
            node.W += G
            node.Q = node.W / node.N

    if not root.children:
        return 0

    # Select action with highest visit count
    visit_counts = np.zeros(n_actions)
    for action, child in root.children.items():
        visit_counts[action] = child.N

    return int(np.argmax(visit_counts))


# ============================================================================
# Training (same as run_vnet_ablation.py)
# ============================================================================

def train_sd_mcts(
    env: DelayedCollapseEnv,
    true_F: float,
    init_F_belief: float = 50.0,
    env_steps: int = 10000,
    use_value_net: bool = True,  # If False, trains MCTS-Zero style (no value network)
) -> Tuple[DelayedCollapseSDModel, PolicyNetwork, Optional[ValueNetwork]]:
    """Train SD-MCTS world model and networks.

    Args:
        use_value_net: If True, train with value network (SD-MCTS).
                       If False, train without value network (MCTS-Zero).
    """
    from src.planning import mcts_search

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = DelayedCollapseSDModel(
        n_actions=n_actions,
        init_alpha=env.alpha,
        init_beta=env.beta,
        init_F=init_F_belief,
        F_req=env.F_req,
        T_s=env.T_s,
        B=env.B,
    )

    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim) if use_value_net else None

    policy_opt = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3) if use_value_net else None
    model_opt = optim.Adam([
        {'params': [world_model._alpha_raw], 'lr': 1e-2},
        {'params': [world_model._beta_raw], 'lr': 5e-2},
        {'params': [world_model._F_raw], 'lr': 5e-2},
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
                if steps < 2000:
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

        # Compute returns
        returns = []
        G = 0.0
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        # Training updates
        for _ in range(5):
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                batch = [t for t in batch if t.state[1] > 0.5]
                if len(batch) < 16:
                    continue

                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)

                pred_next = world_model(states, actions)
                model_loss = F.mse_loss(pred_next, next_states)

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

        # Value network training (only if using value network)
        if use_value_net and len(episode_transitions) > 0:
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

        # Policy network training from MCTS visit counts (simplified: use returns as targets)
        if len(episode_transitions) > 0:
            ep_states = torch.tensor(
                np.array([t.state for t in episode_transitions]),
                dtype=torch.float32
            )
            ep_actions = torch.tensor([t.action for t in episode_transitions], dtype=torch.int64)
            ep_returns = torch.tensor(returns, dtype=torch.float32)

            # Simple policy gradient with baseline
            logits = policy_net(ep_states)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, ep_actions.unsqueeze(1)).squeeze(1)

            # Normalize returns as advantage
            advantages = (ep_returns - ep_returns.mean()) / (ep_returns.std() + 1e-8)
            policy_loss = -(selected_log_probs * advantages.detach()).mean()

            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

        episode += 1

        if steps % 2000 == 0 or steps >= env_steps:
            F_learned = world_model.collapse_threshold.item()
            a_safe = world_model.get_safe_intensity()
            mode = "SD-MCTS" if use_value_net else "MCTS-Zero"
            print(f"  [{mode}] Step {steps}: F_learned={F_learned:.2f} (true={true_F}), a_safe={a_safe:.3f}")

    return world_model, policy_net, value_net


def run_episode(
    env: DelayedCollapseEnv,
    world_model: DelayedCollapseSDModel,
    policy_net: PolicyNetwork,
    value_net: Optional[ValueNetwork],
    num_simulations: int,
    max_depth: int,
    seed: int,
) -> Tuple[float, bool, bool]:
    """Run single episode with depth-limited MCTS."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    collapsed = False
    surge_captured = False

    reward_fn = make_reward_fn(env.n_actions, env.F_req, env.B,
                               F_collapse=world_model.collapse_threshold.item())

    for step in range(env.T):
        with torch.no_grad():
            action = depth_limited_mcts_search(
                state, world_model,
                n_actions=env.n_actions,
                num_simulations=num_simulations,
                max_depth=max_depth,
                policy_net=policy_net,
                value_net=value_net,
                reward_fn=reward_fn,
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
    parser = argparse.ArgumentParser(description="True Horizon Ablation (Theorem 2)")
    parser.add_argument("--h-values", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14])
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--training-steps", type=int, default=15000)
    parser.add_argument("--num-simulations", type=int, default=100,
                        help="Fixed simulation count for all H values")
    parser.add_argument("--results-dir", type=str, default="results/horizon_ablation")
    args = parser.parse_args()

    # Environment config
    true_F = 10.0
    belief_F = 50.0
    alpha = 1.0          # Standard fatigue accumulation
    beta = 0.1           # Standard recovery rate
    F_req = 5.0          # Standard surge requirement
    T_s = 12             # Surge at step 11
    B = 15.0
    T_train = 30
    T_eval = 15

    print("=" * 70)
    print("Experiment: True Horizon Ablation (Theorem 2 Validation)")
    print("=" * 70)
    print(f"Tree depth H values: {args.h_values}")
    print(f"Fixed num_simulations: {args.num_simulations}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Training steps: {args.training_steps}")
    print(f"T_collapse (true F): {true_F}, T_surge: {T_s}")
    print()

    results = {
        "experiment": "horizon_ablation",
        "config": {
            "h_values": args.h_values,
            "num_simulations": args.num_simulations,
            "num_seeds": args.num_seeds,
            "true_F": true_F,
            "belief_F": belief_F,
            "T_s": T_s,
            "training_steps": args.training_steps,
        },
        "runs": [],
        "summary": {},
    }

    # Train SD-MCTS networks (with value network)
    print("Training SD-MCTS networks (with value network)...")
    set_seed(42)
    train_env = DelayedCollapseEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        T_s=T_s, B=B, T=T_train, seed=42
    )
    sd_world_model, sd_policy_net, sd_value_net = train_sd_mcts(
        train_env, true_F=true_F, init_F_belief=belief_F,
        env_steps=args.training_steps, use_value_net=True
    )

    print(f"\n[SD-MCTS] Learned F: {sd_world_model.collapse_threshold.item():.2f}")
    print(f"[SD-MCTS] Learned alpha: {sd_world_model.alpha.item():.3f}, beta: {sd_world_model.beta.item():.3f}")

    # Train MCTS-Zero networks (without value network) - independent training!
    print("\nTraining MCTS-Zero networks (without value network)...")
    set_seed(42)
    train_env = DelayedCollapseEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        T_s=T_s, B=B, T=T_train, seed=42
    )
    zero_world_model, zero_policy_net, _ = train_sd_mcts(
        train_env, true_F=true_F, init_F_belief=belief_F,
        env_steps=args.training_steps, use_value_net=False
    )

    print(f"\n[MCTS-Zero] Learned F: {zero_world_model.collapse_threshold.item():.2f}")
    print(f"[MCTS-Zero] Learned alpha: {zero_world_model.alpha.item():.3f}, beta: {zero_world_model.beta.item():.3f}")
    print()

    random_value_net = RandomValueNetwork(value_scale=10.0)

    # Methods: (world_model, policy_net, value_net, name)
    methods = {
        "sd_mcts": (sd_world_model, sd_policy_net, sd_value_net),
        "mcts_zero": (zero_world_model, zero_policy_net, None),
    }

    for H in args.h_values:
        print(f"Testing max_depth H={H} (fixed simulations={args.num_simulations})...")
        for method_name, (wm, pnet, vnet) in methods.items():
            successes = 0
            rewards = []

            for seed in range(args.num_seeds):
                eval_env = DelayedCollapseEnv(
                    alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                    T_s=T_s, B=B, T=T_eval, seed=seed + 1000
                )

                total_reward, collapsed, surge_captured = run_episode(
                    eval_env, wm, pnet, vnet,
                    num_simulations=args.num_simulations,
                    max_depth=H,
                    seed=seed + 1000,
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
    out_path = results_dir / f"horizon_ablation_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
