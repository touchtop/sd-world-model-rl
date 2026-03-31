#!/usr/bin/env python3
"""CartPole SD-MCTS Benchmark Experiment.

Demonstrates that SD-MCTS generalizes beyond custom-designed environments
to the classic CartPole-v1 control benchmark, where Newton's laws of motion
serve as the structural prior.

Usage:
    python scripts/run_cartpole_benchmark.py --budget 500 --seeds 3
    python scripts/run_cartpole_benchmark.py --budget 500 --seeds 3 --prior_noise 0.3
"""
import argparse
import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from src.world_model.cartpole_sd_model import CartPoleSDModel, CartPoleWorldModelWrapper


# ============================================================
# Value Network for Terminal Value Alignment (TVA)
# ============================================================
class CartPoleValueNet(nn.Module):
    """Simple value network for CartPole leaf node evaluation."""
    def __init__(self, state_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_value_net(value_net, data, gamma=0.99, epochs=100, lr=1e-3):
    """Train value network on collected trajectory data using MC returns."""
    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    states = data['states']
    rewards = data['rewards']

    # Compute Monte Carlo returns (backwards)
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0.0
    ep_boundaries = data.get('ep_boundaries', [])
    for t in reversed(range(len(rewards))):
        if t in ep_boundaries:
            G = 0.0  # Reset at episode boundary
        G = rewards[t] + gamma * G
        returns[t] = G

    returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        pred = value_net(states)
        loss = loss_fn(pred, returns_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# ============================================================
# Data Collection
# ============================================================
def collect_trajectories(env, n_steps, policy='random'):
    """Collect trajectories from CartPole environment."""
    states, actions, next_states, rewards = [], [], [], []
    ep_boundaries = []
    state, _ = env.reset()
    episodes_completed = 0

    for step in range(n_steps):
        if policy == 'random':
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        states.append(state.copy())
        actions.append(action)
        next_states.append(next_state.copy())
        rewards.append(reward)

        if terminated or truncated:
            ep_boundaries.append(step)
            state, _ = env.reset()
            episodes_completed += 1
        else:
            state = next_state

    return {
        'states': torch.tensor(np.array(states), dtype=torch.float32),
        'actions': torch.tensor(np.array(actions), dtype=torch.int64),
        'next_states': torch.tensor(np.array(next_states), dtype=torch.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'ep_boundaries': ep_boundaries,
        'episodes': episodes_completed,
    }


def train_sd_model(model, data, epochs=100, lr=1e-2):
    """Train the SD world model parameters via MSE loss on collected data."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    states = data['states']
    actions = data['actions']
    next_states = data['next_states']

    best_loss = float('inf')
    for epoch in range(epochs):
        pred = model(states, actions)
        loss = loss_fn(pred, next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

    return best_loss


# ============================================================
# MCTS with proper deep tree expansion
# ============================================================
class MCTSNode:
    __slots__ = ['state', 'parent', 'action', 'children', 'N', 'W', 'Q']

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0


def is_terminal(state):
    """Check if CartPole state is terminal."""
    x = state[0].item() if isinstance(state, torch.Tensor) else state[0]
    theta = state[2].item() if isinstance(state, torch.Tensor) else state[2]
    return abs(x) > 2.4 or abs(theta) > (12 * 2 * math.pi / 360)


def mcts_search_cartpole(
    root_state,
    world_model_wrapper,
    n_actions=2,
    num_simulations=100,
    c=1.5,
    gamma=0.99,
    value_net=None,
):
    """MCTS search for CartPole with proper deep tree expansion.

    Each simulation traverses the tree via UCB, expands ONE new node,
    evaluates it (via value_net or heuristic), and backpropagates.
    This builds a progressively deeper tree over simulations.
    """
    root_tensor = torch.tensor(root_state, dtype=torch.float32)

    if is_terminal(root_tensor):
        return 0  # Any action, we're done

    root = MCTSNode(root_tensor)

    for sim in range(num_simulations):
        node = root
        path = [node]
        depth = 0

        # ---- SELECTION: descend tree via UCB ----
        while node.children and len(node.children) == n_actions:
            # All children expanded, select best via UCB
            best_child = None
            best_score = -float('inf')
            for child in node.children.values():
                if child.N == 0:
                    score = float('inf')  # Unvisited children get priority
                else:
                    exploit = child.Q
                    explore = c * math.sqrt(math.log(node.N + 1) / (child.N + 1))
                    score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)
            depth += 1

            # Stop if terminal
            if is_terminal(node.state):
                break

        # ---- EXPANSION: add one new child ----
        if not is_terminal(node.state) and len(node.children) < n_actions:
            # Find an unexpanded action
            for a in range(n_actions):
                if a not in node.children:
                    action_t = torch.tensor([a], dtype=torch.int64)
                    with torch.no_grad():
                        next_state = world_model_wrapper(
                            node.state.unsqueeze(0), action_t
                        )[0]
                    child = MCTSNode(next_state, parent=node, action=a)
                    node.children[a] = child
                    node = child
                    path.append(node)
                    break

        # ---- EVALUATION: estimate leaf value ----
        if is_terminal(node.state):
            leaf_value = 0.0
        elif value_net is not None:
            with torch.no_grad():
                leaf_value = value_net(node.state.unsqueeze(0)).item()
        else:
            # Heuristic evaluation: simulate with bang-bang controller
            rollout_state = node.state
            rollout_return = 0.0
            discount = 1.0
            for step in range(50):  # 50-step heuristic rollout
                if is_terminal(rollout_state):
                    break
                # Bang-bang controller: push toward pole balance
                theta = rollout_state[2].item()
                theta_dot = rollout_state[3].item()
                x_pos = rollout_state[0].item()
                x_vel = rollout_state[1].item()
                score = theta + 0.5 * theta_dot + 0.05 * x_pos + 0.05 * x_vel
                ha = 1 if score > 0 else 0
                action_t = torch.tensor([ha], dtype=torch.int64)
                with torch.no_grad():
                    rollout_state = world_model_wrapper(
                        rollout_state.unsqueeze(0), action_t
                    )[0]
                rollout_return += discount * 1.0  # +1 for surviving
                discount *= gamma
            leaf_value = rollout_return

        # ---- BACKPROPAGATION ----
        G = leaf_value
        for node_bp in reversed(path):
            if node_bp.parent is not None:
                # Add step reward (+1 for being alive)
                r = 0.0 if is_terminal(node_bp.state) else 1.0
                G = r + gamma * G
            node_bp.N += 1
            node_bp.W += G
            node_bp.Q = node_bp.W / node_bp.N

    if not root.children:
        return 0

    # Select action with highest visit count
    best_action = max(root.children.keys(), key=lambda a: root.children[a].N)
    return best_action


# ============================================================
# Evaluation
# ============================================================
def evaluate_sd_mcts(env, world_model, n_episodes=10, num_simulations=100,
                     max_steps=500, value_net=None):
    """Evaluate SD-MCTS agent on CartPole."""
    wrapper = CartPoleWorldModelWrapper(world_model)
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = mcts_search_cartpole(
                state, wrapper, n_actions=2,
                num_simulations=num_simulations,
                value_net=value_net,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    return np.array(episode_rewards)


def evaluate_random(env, n_episodes=10, max_steps=500):
    """Evaluate random policy on CartPole."""
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    return np.array(episode_rewards)


def evaluate_heuristic(env, n_episodes=10, max_steps=500):
    """Evaluate the heuristic (bang-bang) policy on CartPole."""
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            # Same heuristic as rollout policy
            theta = state[2]
            theta_dot = state[3]
            x = state[0]
            x_dot = state[1]
            score = theta + 0.5 * theta_dot + 0.05 * x + 0.05 * x_dot
            action = 1 if score > 0 else 0
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    return np.array(episode_rewards)


# ============================================================
# Main Experiment
# ============================================================
def run_experiment(budget, seed, prior_noise=0.0, num_sims=100,
                   eval_episodes=10, use_value_net=True):
    """Run a single CartPole SD-MCTS experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make('CartPole-v1')
    env.reset(seed=seed)

    # Create SD world model with Newtonian structural prior
    model = CartPoleSDModel(
        learn_params=True,
        prior_noise=prior_noise,
    )

    print(f"\n  [Seed {seed}] Initial params: g={model.g.item():.3f}, "
          f"m_c={model.mc.item():.3f}, m_p={model.mp.item():.3f}, l={model.l.item():.3f}")

    # Phase 1: Collect trajectories from random policy
    exploration_budget = min(budget, max(200, budget // 2))
    data = collect_trajectories(env, exploration_budget, policy='random')
    print(f"  [Seed {seed}] Collected {exploration_budget} transitions "
          f"({data['episodes']} episodes)")

    # Phase 2: System Identification — learn physical parameters
    train_loss = train_sd_model(model, data, epochs=200, lr=1e-2)
    params = model.get_learned_params()
    print(f"  [Seed {seed}] After SI: g={params['gravity']:.3f}, "
          f"m_c={params['masscart']:.3f}, m_p={params['masspole']:.3f}, "
          f"l={params['length']:.3f} (loss={train_loss:.6f})")

    # Phase 3: Train value network (TVA component)
    value_net = None
    if use_value_net:
        value_net = CartPoleValueNet()
        vnet_loss = train_value_net(value_net, data, epochs=200)
        print(f"  [Seed {seed}] Value network loss: {vnet_loss:.4f}")

    # Phase 4: Iterative refinement — use SD-MCTS to collect better data
    remaining_budget = budget - exploration_budget
    if remaining_budget > 50:
        wrapper = CartPoleWorldModelWrapper(model)
        refinement_data = collect_trajectories(
            env, remaining_budget,
            policy=lambda s: mcts_search_cartpole(
                s, wrapper, n_actions=2, num_simulations=max(20, num_sims // 4),
                value_net=value_net,
            )
        )
        # Merge and retrain
        combined_data = {
            'states': torch.cat([data['states'], refinement_data['states']]),
            'actions': torch.cat([data['actions'], refinement_data['actions']]),
            'next_states': torch.cat([data['next_states'], refinement_data['next_states']]),
            'rewards': np.concatenate([data['rewards'], refinement_data['rewards']]),
            'ep_boundaries': data['ep_boundaries'] + [
                b + len(data['rewards']) for b in refinement_data['ep_boundaries']
            ],
        }
        train_loss = train_sd_model(model, combined_data, epochs=100, lr=5e-3)
        if use_value_net:
            vnet_loss = train_value_net(value_net, combined_data, epochs=100)
        params = model.get_learned_params()
        print(f"  [Seed {seed}] After refinement: g={params['gravity']:.3f}, "
              f"m_c={params['masscart']:.3f}, m_p={params['masspole']:.3f}, "
              f"l={params['length']:.3f} (loss={train_loss:.6f})")

    # Phase 5: Final evaluation
    eval_env = gym.make('CartPole-v1')
    eval_env.reset(seed=seed + 1000)
    rewards = evaluate_sd_mcts(
        eval_env, model, n_episodes=eval_episodes,
        num_simulations=num_sims,
        value_net=value_net,
    )
    eval_env.close()
    env.close()

    return rewards, params


def main():
    parser = argparse.ArgumentParser(description='CartPole SD-MCTS Benchmark')
    parser.add_argument('--budget', type=int, default=500,
                        help='Total interaction budget (environment steps)')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--prior_noise', type=float, default=0.3,
                        help='Noise added to initial parameter guesses (0=oracle)')
    parser.add_argument('--num_sims', type=int, default=100,
                        help='MCTS simulations per decision')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--no_value_net', action='store_true',
                        help='Disable value network (heuristic rollout only)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with reduced settings')
    args = parser.parse_args()

    if args.quick:
        args.num_sims = 30
        args.eval_episodes = 5

    print("=" * 70)
    print("CartPole-v1 SD-MCTS Benchmark Experiment")
    print("=" * 70)
    print(f"  Budget:       {args.budget} steps")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Prior noise:  {args.prior_noise}")
    print(f"  MCTS sims:    {args.num_sims}")
    print(f"  Value net:    {'OFF' if args.no_value_net else 'ON'}")
    print(f"  Eval episodes:{args.eval_episodes}")
    print("=" * 70)

    # Run SD-MCTS experiments
    all_rewards = []
    all_params = []
    start_time = time.time()

    for seed_idx in range(args.seeds):
        rewards, params = run_experiment(
            budget=args.budget,
            seed=seed_idx * 42,
            prior_noise=args.prior_noise,
            num_sims=args.num_sims,
            eval_episodes=args.eval_episodes,
            use_value_net=not args.no_value_net,
        )
        all_rewards.append(rewards)
        all_params.append(params)

    elapsed = time.time() - start_time
    all_rewards = np.concatenate(all_rewards)

    # Baselines
    print("\n--- Baselines ---")
    baseline_env = gym.make('CartPole-v1')
    random_rewards = evaluate_random(baseline_env, n_episodes=args.eval_episodes * args.seeds)
    heuristic_rewards = evaluate_heuristic(baseline_env, n_episodes=args.eval_episodes * args.seeds)
    baseline_env.close()

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Mean':>7} {'±Std':>7} {'Min':>5} {'Max':>5}")
    print("-" * 65)
    print(f"{'SD-MCTS (Ours, budget=' + str(args.budget) + ')':<35} "
          f"{all_rewards.mean():>7.1f} {all_rewards.std():>7.1f} "
          f"{all_rewards.min():>5.0f} {all_rewards.max():>5.0f}")
    print(f"{'Heuristic (no learning)':<35} "
          f"{heuristic_rewards.mean():>7.1f} {heuristic_rewards.std():>7.1f} "
          f"{heuristic_rewards.min():>5.0f} {heuristic_rewards.max():>5.0f}")
    print(f"{'Random Policy':<35} "
          f"{random_rewards.mean():>7.1f} {random_rewards.std():>7.1f} "
          f"{random_rewards.min():>5.0f} {random_rewards.max():>5.0f}")
    print("-" * 65)

    # Published baselines
    print(f"\n{'--- Published Baselines (from literature) ---':^65}")
    print(f"{'DreamerV3 (500 steps)':<35} {'~50-100':>7}")
    print(f"{'DreamerV3 (5k steps)':<35} {'~500':>7}")
    print(f"{'PPO (10k steps)':<35} {'~200':>7}")
    print(f"{'DQN (10k steps)':<35} {'~200':>7}")

    # Parameter recovery
    print(f"\n{'--- Parameter Recovery ---':^65}")
    print(f"{'Parameter':<12} {'True':>8} {'Learned':>10} {'Error':>8}")
    print("-" * 40)
    for key in ['gravity', 'masscart', 'masspole', 'length']:
        true_val = all_params[0][f'true_{key}']
        learned_vals = [p[key] for p in all_params]
        mean_l = np.mean(learned_vals)
        err_pct = abs(mean_l - true_val) / true_val * 100
        print(f"{key:<12} {true_val:>8.3f} {mean_l:>10.3f} {err_pct:>7.1f}%")

    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
