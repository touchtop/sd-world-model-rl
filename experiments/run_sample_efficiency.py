"""
Sample Efficiency Comparison: SD-MCTS vs PPO

Experimental design to address reviewer concern:
"If you don't use structured knowledge and just let PPO run for 1M steps, would it be worse?"

Protocol:
- SD-MCTS: 30k steps (fixed)
- PPO: 30k / 100k / 300k / 1M steps
- 10 random seeds per configuration
- 9 scenarios from M1 matrix
- Learning curves logged every 5k (SD-MCTS) or 10k (PPO) steps
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.task_allocation_env import TaskAllocationEnv
from src.env.delayed_collapse_env import DelayedCollapseEnv
from src.world_model.task_sd_model import TaskSDModel
from src.world_model.delayed_collapse_model import DelayedCollapseSDModel
from src.planning.mcts import mcts_search
from src.agent.networks import PolicyNetwork, ValueNetwork
from train_muzero import collect_episode, train_step, ReplayBuffer, Transition


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ScenarioConfig:
    """Scenario configuration for sample efficiency experiments."""
    name: str
    env_type: str  # "task_allocation" or "delayed_collapse"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_m1_scenarios() -> List["ScenarioConfig"]:
        """Generate M1 scenario matrix (9 scenarios)."""
        scenarios = []

        # 1. Linear baseline (no fatigue)
        scenarios.append(ScenarioConfig(
            name="linear_baseline",
            env_type="task_allocation",
            env_kwargs={"arrival_rate": 0.7, "worker_rates": (1.0, 0.8)}
        ))

        # 2-4. Fatigue threshold scenarios
        for threshold in [10.0, 15.0, 20.0]:
            scenarios.append(ScenarioConfig(
                name=f"fatigue_threshold_{int(threshold)}",
                env_type="task_allocation",
                env_kwargs={
                    "arrival_rate": 1.5,
                    "worker_rates": (1.0, 0.8),
                    "fatigue_threshold": threshold,
                }
            ))

        # 5-8. Load sweep scenarios
        for load in [0.5, 0.7, 1.0, 1.3]:
            scenarios.append(ScenarioConfig(
                name=f"load_sweep_{load}",
                env_type="task_allocation",
                env_kwargs={
                    "arrival_rate": load,
                    "worker_rates": (1.0, 0.8),
                    "fatigue_threshold": 15.0,
                }
            ))

        # 9. Heterogeneous workers
        scenarios.append(ScenarioConfig(
            name="heterogeneous_workers",
            env_type="task_allocation",
            env_kwargs={
                "arrival_rate": 0.7,
                "worker_rates": (1.2, 0.6, 0.9),
                "fatigue_threshold": 15.0,
            }
        ))

        return scenarios

    @staticmethod
    def get_delayed_collapse_scenario() -> "ScenarioConfig":
        """Get the delayed collapse scenario (Scenario 2 from paper)."""
        return ScenarioConfig(
            name="delayed_collapse",
            env_type="delayed_collapse",
            env_kwargs={
                "alpha": 1.0,
                "beta": 0.1,
                "F": 10.0,
                "F_req": 5.0,
                "T_s": 12,
                "B": 15.0,
                "T": 30,
                "n_actions": 5,
            }
        )

    def build_env(self, seed: Optional[int] = None):
        """Build environment from config."""
        kwargs = {**self.env_kwargs}
        if seed is not None:
            kwargs["seed"] = seed

        if self.env_type == "task_allocation":
            return TaskAllocationEnv(**kwargs)
        elif self.env_type == "delayed_collapse":
            return DelayedCollapseEnv(**kwargs)
        else:
            raise ValueError(f"Unknown env type: {self.env_type}")


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Training budgets
    sdmcts_budget: int = 30000
    ppo_budgets: List[int] = field(default_factory=lambda: [30000, 100000, 300000, 1000000])

    # Evaluation
    eval_interval_sdmcts: int = 5000
    eval_interval_ppo: int = 10000
    eval_episodes: int = 10

    # Seeds
    seeds: List[int] = field(default_factory=lambda: list(range(10)))

    # Network architecture (aligned for fairness)
    hidden_dim: int = 64
    n_hidden_layers: int = 2

    # PPO hyperparameters
    ppo_lr_init: float = 3e-4
    ppo_lr_decay: bool = True  # Linear decay to 0 for long training
    ppo_gamma: float = 0.99
    ppo_eps_clip: float = 0.2
    ppo_grad_clip: float = 0.5  # For long training stability
    ppo_batch_size: int = 64

    # SD-MCTS hyperparameters
    sdmcts_lr: float = 1e-3
    sdmcts_num_simulations: int = 20  # Training (match paper)
    sdmcts_eval_simulations: int = 50  # Evaluation (match paper)
    sdmcts_gamma: float = 0.99

    # Output
    output_dir: str = "results/sample_efficiency"


# ============================================================================
# PPO Implementation with Long Training Support
# ============================================================================

class PPOAgent:
    """PPO agent with support for long training (LR decay, gradient clipping)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        grad_clip: float = 0.5,
        lr_decay: bool = False,
        total_steps: int = 1000000,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_init = lr
        self.total_steps = total_steps
        self.current_step = 0

        # Build policy network
        policy_layers = []
        in_dim = state_dim
        for _ in range(n_hidden_layers):
            policy_layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        policy_layers.append(nn.Linear(in_dim, action_dim))
        policy_layers.append(nn.Softmax(dim=-1))
        self.policy = nn.Sequential(*policy_layers)

        # Build value network
        value_layers = []
        in_dim = state_dim
        for _ in range(n_hidden_layers):
            value_layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        value_layers.append(nn.Linear(in_dim, 1))
        self.value = nn.Sequential(*value_layers)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

    def get_lr(self) -> float:
        """Get current learning rate (with optional decay)."""
        if not self.lr_decay:
            return self.lr_init
        progress = min(1.0, self.current_step / self.total_steps)
        return self.lr_init * (1.0 - progress)

    def update_lr(self):
        """Update optimizer learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """Get action from policy."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy(state_t)
        if deterministic:
            return probs.argmax(dim=-1).item()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state: np.ndarray) -> float:
        """Get state value."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.value(state_t).item()

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        old_log_probs: List[float],
        returns: List[float],
        advantages: List[float],
    ):
        """PPO update step."""
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Update learning rate
        self.update_lr()

        # PPO epochs
        for _ in range(10):
            probs = self.policy(states_t)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # Policy loss (clipped surrogate)
            ratio = (new_log_probs - old_log_probs_t).exp()
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.value(states_t).squeeze()
            value_loss = nn.MSELoss()(values, returns_t)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.grad_clip
                )

            self.optimizer.step()


# ============================================================================
# Training Functions
# ============================================================================

def train_ppo(
    env,
    agent: PPOAgent,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int = 10,
    episode_length: int = 100,
) -> Dict[str, List]:
    """Train PPO agent and record learning curve."""

    results = {
        "steps": [],
        "rewards": [],
        "collapse_rates": [],
    }

    env_steps = 0

    while env_steps < total_steps:
        # Collect episode
        states, actions, log_probs, rewards, values = [], [], [], [], []
        state, _ = env.reset()
        ep_reward = 0
        collapsed = False

        for _ in range(episode_length):
            action, log_prob = agent.get_action(state)
            value = agent.get_value(state)
            next_state, reward, term, trunc, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value)
            ep_reward += reward

            state = next_state
            env_steps += 1
            agent.current_step = env_steps

            if term or trunc:
                if info.get("collapse_triggered", False) or info.get("collapsed", False):
                    collapsed = True
                break

        # Compute returns and advantages
        returns, advantages = [], []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + agent.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - v)

        # Update agent
        agent.update(states, actions, log_probs, returns, advantages)

        # Evaluate periodically
        if env_steps >= len(results["steps"]) * eval_interval + eval_interval:
            eval_reward, eval_collapse = evaluate_agent(
                env, agent, eval_episodes, episode_length
            )
            results["steps"].append(env_steps)
            results["rewards"].append(eval_reward)
            results["collapse_rates"].append(eval_collapse)
            print(f"  PPO @ {env_steps:>7d} steps: reward={eval_reward:.2f}, collapse={eval_collapse:.1%}")

    return results


def train_ppo_with_checkpoints(
    env,
    agent: PPOAgent,
    checkpoints: List[int],
    eval_interval: int,
    eval_episodes: int = 10,
    episode_length: int = 100,
) -> Dict[int, Dict[str, List]]:
    """Train PPO once to max checkpoint, record results at each checkpoint.

    Args:
        checkpoints: List of step counts to record (e.g., [30000, 100000, 300000, 1000000])

    Returns:
        Dict mapping checkpoint -> {"steps": [...], "rewards": [...], "collapse_rates": [...]}
    """
    checkpoints = sorted(checkpoints)
    max_steps = checkpoints[-1]

    # Results for each checkpoint
    checkpoint_results = {cp: {"steps": [], "rewards": [], "collapse_rates": []} for cp in checkpoints}

    # Current training state
    current_results = {"steps": [], "rewards": [], "collapse_rates": []}
    env_steps = 0
    checkpoint_idx = 0

    while env_steps < max_steps:
        # Collect episode
        states, actions, log_probs, rewards, values = [], [], [], [], []
        state, _ = env.reset()

        for _ in range(episode_length):
            action, log_prob = agent.get_action(state)
            value = agent.get_value(state)
            next_state, reward, term, trunc, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value)

            state = next_state
            env_steps += 1
            agent.current_step = env_steps

            if term or trunc:
                break

        # Compute returns and advantages
        returns, advantages = [], []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + agent.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - v)

        # Update agent
        agent.update(states, actions, log_probs, returns, advantages)

        # Evaluate periodically
        if env_steps >= len(current_results["steps"]) * eval_interval + eval_interval:
            eval_reward, eval_collapse = evaluate_agent(
                env, agent, eval_episodes, episode_length
            )
            current_results["steps"].append(env_steps)
            current_results["rewards"].append(eval_reward)
            current_results["collapse_rates"].append(eval_collapse)
            print(f"  PPO @ {env_steps:>7d} steps: reward={eval_reward:.2f}, collapse={eval_collapse:.1%}")

        # Check if we've reached a checkpoint
        while checkpoint_idx < len(checkpoints) and env_steps >= checkpoints[checkpoint_idx]:
            cp = checkpoints[checkpoint_idx]
            # Force evaluation at checkpoint if not just evaluated
            if not current_results["steps"] or current_results["steps"][-1] != env_steps:
                eval_reward, eval_collapse = evaluate_agent(
                    env, agent, eval_episodes, episode_length
                )
                current_results["steps"].append(env_steps)
                current_results["rewards"].append(eval_reward)
                current_results["collapse_rates"].append(eval_collapse)
                print(f"  PPO @ {env_steps:>7d} steps: reward={eval_reward:.2f}, collapse={eval_collapse:.1%} [checkpoint]")
            # Copy current results up to this checkpoint
            checkpoint_results[cp] = {
                "steps": current_results["steps"].copy(),
                "rewards": current_results["rewards"].copy(),
                "collapse_rates": current_results["collapse_rates"].copy(),
            }
            checkpoint_idx += 1

    return checkpoint_results


def train_sdmcts(
    env,
    world_model,
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    total_steps: int,
    eval_interval: int,
    num_simulations: int = 20,
    eval_simulations: int = 50,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eval_episodes: int = 10,
    episode_length: int = 100,
) -> Dict[str, List]:
    """Train SD-MCTS agent and record learning curve."""

    results = {
        "steps": [],
        "rewards": [],
        "collapse_rates": [],
    }

    buffer = ReplayBuffer(capacity=10000)
    n_actions = env.action_space.n
    n_workers = getattr(env, 'n_workers', None)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    model_optimizer = optim.Adam(world_model.parameters(), lr=lr)

    env_steps = 0

    while env_steps < total_steps:
        # Collect episode using MCTS
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=num_simulations, gamma=gamma,
            n_workers=n_workers
        )

        for t in transitions:
            buffer.add(t)
        env_steps += len(transitions)

        # Train
        if len(buffer) >= 64:
            for _ in range(5):
                train_step(
                    policy_net, value_net, world_model, buffer,
                    policy_optimizer, value_optimizer, model_optimizer,
                    batch_size=64
                )

        # Evaluate periodically
        if env_steps >= len(results["steps"]) * eval_interval + eval_interval:
            eval_reward, eval_collapse = evaluate_sdmcts(
                env, world_model, policy_net, value_net,
                n_actions, eval_simulations, gamma, eval_episodes, episode_length
            )
            results["steps"].append(env_steps)
            results["rewards"].append(eval_reward)
            results["collapse_rates"].append(eval_collapse)
            print(f"  SD-MCTS @ {env_steps:>7d} steps: reward={eval_reward:.2f}, collapse={eval_collapse:.1%}")

    return results


def evaluate_agent(env, agent, num_episodes: int, episode_length: int) -> Tuple[float, float]:
    """Evaluate PPO agent."""
    rewards = []
    collapses = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        collapsed = False

        for _ in range(episode_length):
            action = agent.get_action(state, deterministic=True)
            state, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc:
                if info.get("collapse_triggered", False) or info.get("collapsed", False):
                    collapsed = True
                break

        rewards.append(ep_reward)
        if collapsed:
            collapses += 1

    return np.mean(rewards), collapses / num_episodes


def evaluate_sdmcts(
    env, world_model, policy_net, value_net,
    n_actions, num_simulations, gamma,
    num_episodes: int, episode_length: int
) -> Tuple[float, float]:
    """Evaluate SD-MCTS agent."""
    rewards = []
    collapses = 0
    n_workers = getattr(env, 'n_workers', None)

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        collapsed = False

        for _ in range(episode_length):
            action = mcts_search(
                state, world_model, n_actions=n_actions,
                num_simulations=num_simulations, gamma=gamma,
                policy_net=policy_net, value_net=value_net,
                n_workers=n_workers
            )
            state, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc:
                if info.get("collapse_triggered", False) or info.get("collapsed", False):
                    collapsed = True
                break

        rewards.append(ep_reward)
        if collapsed:
            collapses += 1

    return np.mean(rewards), collapses / num_episodes


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(config: ExperimentConfig, scenarios: List[ScenarioConfig]):
    """Run full sample efficiency experiment."""

    os.makedirs(config.output_dir, exist_ok=True)

    all_results = {
        "config": asdict(config),
        "scenarios": {},
    }

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"{'='*60}")

        scenario_results = {
            "sdmcts": {"seeds": {}},
            "ppo": {budget: {"seeds": {}} for budget in config.ppo_budgets},
        }

        for seed in config.seeds:
            print(f"\n--- Seed {seed} ---")

            # Build environment
            env = scenario.build_env(seed=seed)
            state_dim = env.observation_space.shape[0]
            n_actions = env.action_space.n

            # ===== Train SD-MCTS =====
            print(f"\n[SD-MCTS] Training for {config.sdmcts_budget} steps...")

            # Initialize models
            if scenario.env_type == "task_allocation":
                n_workers = len(scenario.env_kwargs.get("worker_rates", (1.0, 0.8)))
                fatigue_threshold = scenario.env_kwargs.get("fatigue_threshold", None)
                fatigue_penalty = scenario.env_kwargs.get("fatigue_penalty", 0.5)
                world_model = TaskSDModel(
                    n_workers=n_workers,
                    fatigue_threshold=fatigue_threshold,
                    fatigue_penalty=fatigue_penalty,
                )
            else:
                world_model = DelayedCollapseSDModel(
                    n_actions=n_actions,
                    init_F=50.0,  # Wrong belief
                )

            policy_net = PolicyNetwork(state_dim, n_actions, hidden_dim=config.hidden_dim)
            value_net = ValueNetwork(state_dim, hidden_dim=config.hidden_dim)

            torch.manual_seed(seed)
            np.random.seed(seed)

            sdmcts_results = train_sdmcts(
                env, world_model, policy_net, value_net,
                total_steps=config.sdmcts_budget,
                eval_interval=config.eval_interval_sdmcts,
                num_simulations=config.sdmcts_num_simulations,
                eval_simulations=config.sdmcts_eval_simulations,
                gamma=config.sdmcts_gamma,
                lr=config.sdmcts_lr,
                eval_episodes=config.eval_episodes,
            )
            scenario_results["sdmcts"]["seeds"][seed] = sdmcts_results

            # ===== Train PPO once to max budget, record at checkpoints =====
            max_budget = max(config.ppo_budgets)
            print(f"\n[PPO] Training once to {max_budget//1000}k steps (checkpoints: {[b//1000 for b in config.ppo_budgets]}k)...")

            # Re-create environment for fresh start
            env = scenario.build_env(seed=seed)

            torch.manual_seed(seed)
            np.random.seed(seed)

            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=n_actions,
                hidden_dim=config.hidden_dim,
                n_hidden_layers=config.n_hidden_layers,
                lr=config.ppo_lr_init,
                gamma=config.ppo_gamma,
                eps_clip=config.ppo_eps_clip,
                grad_clip=config.ppo_grad_clip,
                lr_decay=config.ppo_lr_decay,
                total_steps=max_budget,
            )

            ppo_checkpoint_results = train_ppo_with_checkpoints(
                env, agent,
                checkpoints=config.ppo_budgets,
                eval_interval=config.eval_interval_ppo,
                eval_episodes=config.eval_episodes,
            )

            for budget in config.ppo_budgets:
                scenario_results["ppo"][budget]["seeds"][seed] = ppo_checkpoint_results[budget]

        all_results["scenarios"][scenario.name] = scenario_results

        # Save intermediate results
        with open(os.path.join(config.output_dir, "sample_efficiency_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {config.output_dir}/sample_efficiency_results.json")
    print(f"{'='*60}")

    return all_results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample Efficiency Comparison Experiment")
    parser.add_argument("--scenarios", type=str, default="m1",
                        choices=["m1", "delayed_collapse", "all"],
                        help="Scenario set to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Random seeds")
    parser.add_argument("--ppo-budgets", type=int, nargs="+",
                        default=[30000, 100000, 300000, 1000000],
                        help="PPO training budgets")
    parser.add_argument("--output-dir", type=str, default="results/sample_efficiency",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced budgets")

    args = parser.parse_args()

    # Build config
    config = ExperimentConfig(
        seeds=args.seeds,
        ppo_budgets=args.ppo_budgets,
        output_dir=args.output_dir,
    )

    if args.quick:
        config.sdmcts_budget = 5000
        config.ppo_budgets = [5000, 10000]
        config.eval_interval_sdmcts = 1000
        config.eval_interval_ppo = 2000
        config.seeds = [0, 1]

    # Build scenarios
    if args.scenarios == "m1":
        scenarios = ScenarioConfig.get_m1_scenarios()
    elif args.scenarios == "delayed_collapse":
        scenarios = [ScenarioConfig.get_delayed_collapse_scenario()]
    else:
        scenarios = ScenarioConfig.get_m1_scenarios()
        scenarios.append(ScenarioConfig.get_delayed_collapse_scenario())

    # Run
    run_experiment(config, scenarios)
