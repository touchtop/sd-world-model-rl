"""
Ablation Study: Isolating the Contribution of Structured Knowledge.

Variants:
1. SD-MCTS (Full): Complete SD equations - baseline best
2. SD-MCTS (Random): Random initialized parameters - should degrade
3. Pure MCTS: No world model learning - should degrade significantly
4. SD-MCTS (Incorrect): Wrong SD structure - should be worst
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.task_allocation_env import TaskAllocationEnv
from src.env.delayed_collapse_env import DelayedCollapseEnv
from src.world_model.task_sd_model import TaskSDModel
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.planning.mcts import mcts_search
from src.agent.networks import PolicyNetwork, ValueNetwork
from train_muzero import collect_episode, train_step, ReplayBuffer


# ============================================================================
# Ablation World Models
# ============================================================================

class FrozenSDModel(nn.Module):
    """SD model with frozen (non-learnable) parameters."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        # Freeze all parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, state, action):
        return self.base_model(state, action)

    def parameters(self):
        return iter([])  # No learnable parameters


class RandomWorldModel(nn.Module):
    """Pure neural network world model (no SD structure)."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # One-hot encode action
        if action.dim() == 0:
            action = action.unsqueeze(0)
        batch_size = state.shape[0] if state.dim() > 1 else 1
        action_onehot = torch.zeros(batch_size, self.n_actions)
        action_onehot.scatter_(1, action.view(-1, 1), 1.0)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.cat([state, action_onehot], dim=-1)
        delta = self.net(x)
        return state + delta  # Residual prediction


class IncorrectSDModel(nn.Module):
    """SD model with systematically wrong structure."""

    def __init__(self, n_workers: int = 2):
        super().__init__()
        self.n_workers = n_workers

        # WRONG: Use additive instead of multiplicative fatigue effect
        self.wrong_fatigue_coef = nn.Parameter(torch.tensor(0.5))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        next_state = state.clone()

        # WRONG dynamics: fatigue adds to queue instead of reducing efficiency
        queue = state[:, 0]
        backlogs = state[:, 1:1+self.n_workers]

        # Incorrect: queue grows with fatigue (completely wrong)
        fatigue_effect = self.wrong_fatigue_coef * backlogs.sum(dim=1)
        next_state[:, 0] = queue + fatigue_effect

        return next_state


# ============================================================================
# Ablation Experiment Runner
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    training_steps: int = 30000
    eval_interval: int = 5000
    eval_episodes: int = 10
    seeds: List[int] = None
    num_simulations: int = 50
    output_dir: str = "results/ablation"

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(range(10))


def create_ablation_model(
    variant: str,
    env_type: str,
    state_dim: int,
    n_actions: int,
    n_workers: int = 2,
) -> nn.Module:
    """Create world model for ablation variant."""

    if variant == "full":
        # Full SD model (baseline)
        if env_type == "task_allocation":
            return TaskSDModel(n_workers=n_workers)
        else:
            return DelayedCollapseSDModel(n_actions=n_actions, init_F=50.0)

    elif variant == "random_init":
        # SD model with very wrong initial parameters
        if env_type == "task_allocation":
            model = TaskSDModel(n_workers=n_workers)
            # Randomize parameters
            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.randn_like(param) * 2.0
            return model
        else:
            model = DelayedCollapseSDModel(n_actions=n_actions, init_F=100.0)
            with torch.no_grad():
                model._alpha_raw.data = torch.tensor(5.0)
                model._F_raw.data = torch.tensor(100.0)
            return model

    elif variant == "frozen":
        # SD model with frozen (wrong) parameters
        if env_type == "task_allocation":
            base = TaskSDModel(n_workers=n_workers)
        else:
            base = DelayedCollapseSDModel(n_actions=n_actions, init_F=50.0)
        return FrozenSDModel(base)

    elif variant == "pure_nn":
        # Pure neural network (no SD structure)
        return RandomWorldModel(state_dim, n_actions)

    elif variant == "incorrect":
        # Incorrect SD structure
        return IncorrectSDModel(n_workers=n_workers)

    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_ablation_variant(
    env,
    variant: str,
    config: AblationConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run single ablation variant."""

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_workers = getattr(env, 'n_workers', 2)
    env_type = "delayed_collapse" if isinstance(env, DelayedCollapseEnv) else "task_allocation"

    # Create models
    world_model = create_ablation_model(variant, env_type, state_dim, n_actions, n_workers)
    policy_net = PolicyNetwork(state_dim, n_actions, hidden_dim=64)
    value_net = ValueNetwork(state_dim, hidden_dim=64)

    buffer = ReplayBuffer(capacity=10000)

    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

    # Only optimize world model if it has learnable parameters
    model_params = list(world_model.parameters())
    if len(model_params) > 0:
        model_optimizer = optim.Adam(model_params, lr=1e-3)
    else:
        model_optimizer = None

    results = {"steps": [], "rewards": [], "collapse_rates": []}
    env_steps = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    while env_steps < config.training_steps:
        # Collect episode
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=config.num_simulations, gamma=0.99,
            n_workers=n_workers if env_type == "task_allocation" else None
        )

        for t in transitions:
            buffer.add(t)
        env_steps += len(transitions)

        # Train
        if len(buffer) >= 64:
            for _ in range(5):
                # Custom train step for ablation (may not have model optimizer)
                if len(buffer) < 64:
                    continue

                batch = buffer.sample(64)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
                mcts_policies = torch.tensor(np.array([t.mcts_policy for t in batch]), dtype=torch.float32)
                returns = torch.tensor([t.cumulative_return for t in batch], dtype=torch.float32)

                # Policy loss
                policy_logits = policy_net(states)
                policy_loss = -(mcts_policies * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Value loss
                value_pred = value_net(states)
                value_loss = nn.MSELoss()(value_pred, returns)
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                # Model loss (if learnable)
                if model_optimizer is not None:
                    pred_next = world_model(states, actions)
                    model_loss = nn.MSELoss()(pred_next, next_states)
                    model_optimizer.zero_grad()
                    model_loss.backward()
                    model_optimizer.step()

        # Evaluate
        if env_steps >= len(results["steps"]) * config.eval_interval + config.eval_interval:
            eval_rewards = []
            eval_collapses = 0

            for _ in range(config.eval_episodes):
                state, _ = env.reset()
                ep_reward = 0
                collapsed = False

                for _ in range(100):
                    action = mcts_search(
                        state, world_model, n_actions=n_actions,
                        num_simulations=config.num_simulations, gamma=0.99,
                        policy_net=policy_net, value_net=value_net,
                        n_workers=n_workers if env_type == "task_allocation" else None
                    )
                    state, reward, term, trunc, info = env.step(action)
                    ep_reward += reward
                    if term or trunc:
                        if info.get("collapse_triggered", False) or info.get("collapsed", False):
                            collapsed = True
                        break

                eval_rewards.append(ep_reward)
                if collapsed:
                    eval_collapses += 1

            results["steps"].append(env_steps)
            results["rewards"].append(np.mean(eval_rewards))
            results["collapse_rates"].append(eval_collapses / config.eval_episodes)

            print(f"    {variant} @ {env_steps}: reward={np.mean(eval_rewards):.2f}")

    return results


def run_ablation_study(config: AblationConfig, scenario_name: str = "delayed_collapse"):
    """Run full ablation study."""

    os.makedirs(config.output_dir, exist_ok=True)

    variants = ["full", "random_init", "frozen", "pure_nn"]  # "incorrect" removed - too unstable

    all_results = {
        "config": asdict(config),
        "scenario": scenario_name,
        "variants": {},
    }

    for variant in variants:
        print(f"\n{'='*40}")
        print(f"Variant: {variant}")
        print(f"{'='*40}")

        all_results["variants"][variant] = {"seeds": {}}

        for seed in config.seeds:
            print(f"\n  Seed {seed}")

            # Create environment
            if scenario_name == "delayed_collapse":
                env = DelayedCollapseEnv(
                    alpha=1.0, beta=0.1, F=10.0, F_req=5.0,
                    T_s=12, B=15.0, T=30, n_actions=5, seed=seed
                )
            else:
                env = TaskAllocationEnv(
                    arrival_rate=1.5, worker_rates=(1.0, 0.8),
                    fatigue_threshold=15.0, seed=seed
                )

            results = run_ablation_variant(env, variant, config, seed)
            all_results["variants"][variant]["seeds"][str(seed)] = results

    # Save results
    output_path = os.path.join(config.output_dir, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Generate summary
    print("\n" + "="*50)
    print("ABLATION STUDY SUMMARY")
    print("="*50)

    for variant in variants:
        final_rewards = [
            all_results["variants"][variant]["seeds"][str(s)]["rewards"][-1]
            for s in config.seeds
            if all_results["variants"][variant]["seeds"][str(s)]["rewards"]
        ]
        if final_rewards:
            print(f"{variant:15s}: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--scenario", type=str, default="delayed_collapse",
                        choices=["delayed_collapse", "task_allocation"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    config = AblationConfig(
        training_steps=args.steps,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )

    if args.quick:
        config.training_steps = 5000
        config.eval_interval = 1000
        config.seeds = [0, 1]

    run_ablation_study(config, args.scenario)
