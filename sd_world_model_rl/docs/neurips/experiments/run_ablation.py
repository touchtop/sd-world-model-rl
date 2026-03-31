#!/usr/bin/env python3
"""Ablation Study for SD-MCTS.

Four ablation variants:
1. sd_mcts (full): Complete SD-MCTS with all components
2. no_vnet: SD-MCTS without value network (zero terminal value)
3. no_mcts: SD model with greedy 1-step lookahead (no tree search)
4. blackbox_mcts: MLP world model + MCTS (no structural prior)

Usage:
    python run_ablation.py --budgets 1000 3000 --seeds 3
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.planning.mcts import mcts_search

from experiments.run_beer_game_v2 import (
    BeerGameSDModelWrapper,
    make_beer_game_reward_fn,
    NormalizedValueNet,
    ReplayBuffer,
    Transition,
)

from baselines.blackbox_world_model import BlackBoxWorldModel, train_blackbox_model

ABLATION_METHODS = ['sd_mcts', 'no_vnet', 'no_mcts', 'blackbox_mcts']


@dataclass
class AblationMetrics:
    """Metrics from a single ablation run."""
    method: str
    env: str
    sample_budget: int
    seed: int
    eval_return_mean: float = 0.0
    eval_return_std: float = 0.0
    train_returns: List[float] = field(default_factory=list)
    total_time: float = 0.0
    train_time: float = 0.0
    total_steps: int = 0
    total_episodes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'env': self.env,
            'sample_budget': self.sample_budget,
            'seed': self.seed,
            'eval_return_mean': self.eval_return_mean,
            'eval_return_std': self.eval_return_std,
            'total_time': self.total_time,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'error': self.error,
        }


# =============================================================================
# SD-MCTS Full (baseline for ablation)
# =============================================================================

def train_and_eval_sd_mcts(
    env: BeerGameEnvFlat,
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
    use_value_net: bool = True,
    device: str = 'cpu',
) -> Tuple[float, float, List[float], float]:
    """Train and evaluate SD-MCTS with optional value network."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize models
    world_model = BeerGameSDModel(
        order_levels=env.order_levels,
        O_max=env.O_max,
    ).to(device)
    world_model_wrapper = BeerGameSDModelWrapper(world_model)

    policy_net = PolicyNetwork(state_dim, n_actions, hidden_dim=128).to(device)
    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device) if use_value_net else None

    # Optimizers
    wm_params = list(world_model.parameters())
    model_opt = optim.Adam(wm_params, lr=1e-2) if wm_params else None
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3) if value_net else None

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    episode_rewards = []
    steps = 0
    episode = 0

    # Running statistics for return normalization
    return_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def get_return_stats():
        return return_stats['mean'], return_stats['std']

    # Wrapped value net for MCTS
    normalized_value_net = None
    if value_net is not None:
        normalized_value_net = NormalizedValueNet(value_net, get_return_stats)

    random_steps = min(500, sample_budget // 4)
    epsilon = 0.3
    gamma = 0.99

    start_time = time.time()

    while steps < sample_budget:
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_transitions = []

        for t in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model_wrapper,
                    n_actions=n_actions,
                    num_simulations=mcts_simulations,
                    policy_net=None,
                    value_net=normalized_value_net,  # None if no_vnet
                    reward_fn=reward_fn,
                )

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            transition = Transition(
                state=state.copy(), action=action, reward=reward,
                next_state=next_state.copy(), done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated or steps >= sample_budget:
                break

        # Compute returns
        returns = []
        G = 0.0
        for t in reversed(episode_transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)

        # Update return statistics
        for r in returns:
            return_stats['count'] += 1
            delta = r - return_stats['mean']
            return_stats['mean'] += delta / return_stats['count']
            if return_stats['count'] > 1:
                return_stats['std'] = max(1.0, np.sqrt(
                    ((return_stats['count'] - 1) * return_stats['std']**2 + delta * (r - return_stats['mean']))
                    / return_stats['count']
                ))

        # World model training
        for _ in range(5):
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)

                if model_opt is not None:
                    pred_next = world_model(states, actions)
                    model_loss = F.mse_loss(pred_next, next_states)
                    model_opt.zero_grad()
                    model_loss.backward()
                    model_opt.step()

        # Value network training
        if value_opt is not None and len(episode_transitions) > 0:
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

        episode_rewards.append(episode_reward)
        episode += 1

    train_time = time.time() - start_time

    # Evaluation
    eval_returns = []
    for ep in range(eval_episodes):
        state, _ = env.reset(seed=1000 + ep)
        episode_return = 0.0
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=world_model_wrapper,
                n_actions=n_actions,
                num_simulations=mcts_simulations,
                policy_net=None,
                value_net=normalized_value_net,
                reward_fn=reward_fn,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        eval_returns.append(episode_return)

    return float(np.mean(eval_returns)), float(np.std(eval_returns)), episode_rewards, train_time


# =============================================================================
# No-MCTS: Greedy 1-step lookahead
# =============================================================================

def train_and_eval_no_mcts(
    env: BeerGameEnvFlat,
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    device: str = 'cpu',
) -> Tuple[float, float, List[float], float]:
    """Train SD model but use greedy 1-step lookahead instead of MCTS."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = BeerGameSDModel(
        order_levels=env.order_levels,
        O_max=env.O_max,
    ).to(device)
    world_model_wrapper = BeerGameSDModelWrapper(world_model)

    wm_params = list(world_model.parameters())
    model_opt = optim.Adam(wm_params, lr=1e-2) if wm_params else None

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    episode_rewards = []
    steps = 0
    episode = 0
    gamma = 0.99

    random_steps = min(500, sample_budget // 4)
    epsilon = 0.3

    start_time = time.time()

    def greedy_action(state: np.ndarray) -> int:
        """Select action by 1-step greedy lookahead using world model."""
        best_action = 0
        best_reward = float('-inf')

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for a in range(n_actions):
            action_tensor = torch.tensor([a], dtype=torch.int64, device=device)
            with torch.no_grad():
                next_state = world_model(state_tensor, action_tensor).squeeze(0).cpu().numpy()
            reward = reward_fn(state, a, next_state)
            if reward > best_reward:
                best_reward = reward
                best_action = a

        return best_action

    while steps < sample_budget:
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_transitions = []

        for t in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            transition = Transition(
                state=state.copy(), action=action, reward=reward,
                next_state=next_state.copy(), done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated or steps >= sample_budget:
                break

        # World model training
        for _ in range(5):
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)

                if model_opt is not None:
                    pred_next = world_model(states, actions)
                    model_loss = F.mse_loss(pred_next, next_states)
                    model_opt.zero_grad()
                    model_loss.backward()
                    model_opt.step()

        episode_rewards.append(episode_reward)
        episode += 1

    train_time = time.time() - start_time

    # Evaluation with greedy
    eval_returns = []
    for ep in range(eval_episodes):
        state, _ = env.reset(seed=1000 + ep)
        episode_return = 0.0
        done = False

        while not done:
            action = greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        eval_returns.append(episode_return)

    return float(np.mean(eval_returns)), float(np.std(eval_returns)), episode_rewards, train_time


# =============================================================================
# BlackBox-MCTS (from run_comparison.py)
# =============================================================================

def train_and_eval_blackbox_mcts(
    env: BeerGameEnvFlat,
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
    device: str = 'cpu',
) -> Tuple[float, float, List[float], float]:
    """Train BlackBox MLP world model + MCTS."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    bb_model = BlackBoxWorldModel(state_dim, n_actions, hidden_dim=256).to(device)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    replay_buffer = []
    total_steps = 0
    train_returns = []

    warmup_steps = min(200, sample_budget // 5)
    steps_per_train = 200
    model_trained = False

    start_time = time.time()
    state, _ = env.reset(seed=seed)

    while total_steps < sample_budget:
        episode_return = 0.0
        done = False

        while not done and total_steps < sample_budget:
            if total_steps < warmup_steps or not model_trained:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=bb_model,
                    n_actions=n_actions,
                    num_simulations=mcts_simulations,
                    reward_fn=reward_fn,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.append((state, action, next_state, reward, terminated or truncated))
            state = next_state
            episode_return += reward
            total_steps += 1

            if total_steps % steps_per_train == 0 and len(replay_buffer) >= 64:
                epochs = max(10, min(50, 1000 // len(replay_buffer) * 10))
                train_blackbox_model(bb_model, replay_buffer, epochs=epochs, batch_size=64)
                model_trained = True

            done = terminated or truncated

        train_returns.append(episode_return)
        if done:
            state, _ = env.reset(seed=seed + len(train_returns))

    # Final training
    if len(replay_buffer) >= 64:
        final_epochs = max(50, min(200, 2000 // max(1, len(replay_buffer) // 100)))
        train_blackbox_model(bb_model, replay_buffer, epochs=final_epochs, batch_size=64)

    train_time = time.time() - start_time
    bb_model.eval()

    # Evaluation
    eval_returns = []
    for ep in range(eval_episodes):
        state, _ = env.reset(seed=1000 + ep)
        episode_return = 0.0
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=bb_model,
                n_actions=n_actions,
                num_simulations=mcts_simulations,
                reward_fn=reward_fn,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        eval_returns.append(episode_return)

    return float(np.mean(eval_returns)), float(np.std(eval_returns)), train_returns, train_time


# =============================================================================
# Main Runner
# =============================================================================

def run_ablation_experiment(
    method: str,
    sample_budget: int,
    seed: int,
    eval_episodes: int = 10,
    mcts_simulations: int = 200,
) -> AblationMetrics:
    """Run a single ablation experiment."""
    device = 'cpu'
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics = AblationMetrics(
        method=method,
        env="beer_game",
        sample_budget=sample_budget,
        seed=seed,
    )

    env = BeerGameEnvFlat(order_levels=3, seed=seed)
    start_time = time.time()

    try:
        if method == 'sd_mcts':
            mean, std, train_returns, train_time = train_and_eval_sd_mcts(
                env, sample_budget, seed, eval_episodes, mcts_simulations,
                use_value_net=True, device=device
            )
        elif method == 'no_vnet':
            mean, std, train_returns, train_time = train_and_eval_sd_mcts(
                env, sample_budget, seed, eval_episodes, mcts_simulations,
                use_value_net=False, device=device
            )
        elif method == 'no_mcts':
            mean, std, train_returns, train_time = train_and_eval_no_mcts(
                env, sample_budget, seed, eval_episodes, device=device
            )
        elif method == 'blackbox_mcts':
            mean, std, train_returns, train_time = train_and_eval_blackbox_mcts(
                env, sample_budget, seed, eval_episodes, mcts_simulations, device=device
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        metrics.eval_return_mean = mean
        metrics.eval_return_std = std
        metrics.train_returns = train_returns
        metrics.train_time = train_time
        metrics.total_episodes = len(train_returns)
        metrics.total_steps = sample_budget

    except Exception as e:
        metrics.error = str(e)
        import traceback
        traceback.print_exc()

    metrics.total_time = time.time() - start_time
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SD-MCTS Ablation Study")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 3000],
                        help="Sample budgets to test")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations")
    parser.add_argument("--methods", type=str, nargs="+", default=ABLATION_METHODS,
                        help="Methods to run")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = NEURIPS_DIR / "results" / "ablation" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SD-MCTS ABLATION STUDY")
    print("=" * 70)
    print(f"Methods: {args.methods}")
    print(f"Budgets: {args.budgets}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    results = []
    total_runs = len(args.budgets) * args.seeds * len(args.methods)
    current = 0

    for budget in args.budgets:
        for seed in range(args.seeds):
            for method in args.methods:
                current += 1
                print(f"\n[{current}/{total_runs}] {method.upper()} | budget={budget} | seed={seed}")

                m = run_ablation_experiment(
                    method=method,
                    sample_budget=budget,
                    seed=seed,
                    eval_episodes=args.eval_episodes,
                    mcts_simulations=args.mcts_sims,
                )

                results.append(m.to_dict())

                if m.error:
                    print(f"  ERROR: {m.error}")
                else:
                    print(f"  Return: {m.eval_return_mean:.1f} ± {m.eval_return_std:.1f} | Time: {m.total_time:.1f}s")

                # Save intermediate results
                with open(output_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            'methods': args.methods,
            'sample_budgets': args.budgets,
            'n_seeds': args.seeds,
            'eval_episodes': args.eval_episodes,
            'mcts_simulations': args.mcts_sims,
        }, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'=' * 70}")

    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error"):
            continue
        grouped[r["sample_budget"]][r["method"]].append(r["eval_return_mean"])

    header = f"{'Budget':<10}"
    for method in args.methods:
        header += f"{method:<18}"
    print(header)
    print("-" * 70)

    for budget in sorted(grouped.keys()):
        row = f"{budget:<10}"
        for method in args.methods:
            vals = grouped[budget].get(method, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0
                row += f"{mean:>7.1f}±{std:<8.1f}"
            else:
                row += f"{'N/A':<18}"
        print(row)

    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
