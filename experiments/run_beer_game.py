"""Beer Game Experiments for SD-MCTS.

Implements the supply chain bullwhip effect benchmark (Sterman 1989).

Experiment flow:
1. Train SD-MCTS on Beer Game environment
2. Evaluate against baselines (Base-Stock, Anchoring, MPC)
3. Compute metrics (Total Cost, Bullwhip Ratio)
4. Generate visualizations
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.beer_game_env import BeerGameEnv, BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel, BeerGameWorldModelWrapper
from src.baselines.beer_game_baselines import (
    BaseStockPolicy,
    AnchoringAdjustmentPolicy,
    OracleMPCPolicy,
    ConstantOrderPolicy,
    ReactivePolicy,
)


class ValueNetwork(nn.Module):
    """Value network for Beer Game state evaluation."""

    def __init__(self, state_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ReplayBuffer:
    """Simple replay buffer for experience storage."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


def make_beer_game_reward_fn(h: float = 0.5, p: float = 1.0, n_echelons: int = 4):
    """Create reward function for MCTS."""
    def reward_fn(state: np.ndarray, action=None) -> float:
        # Parse inventory and backlog from state
        inventory = state[0::6][:n_echelons]
        backlog = state[1::6][:n_echelons]

        holding_cost = h * np.sum(np.maximum(0, inventory))
        backlog_cost = p * np.sum(backlog)

        return -(holding_cost + backlog_cost)

    return reward_fn


def beer_game_mcts_search(
    root_state: np.ndarray,
    world_model: BeerGameWorldModelWrapper,
    value_net: nn.Module,
    n_actions: int,
    num_simulations: int = 50,
    c: float = 1.4,
    gamma: float = 0.99,
    device: str = 'cpu',
) -> int:
    """Simplified MCTS for Beer Game with flat action space.

    Uses progressive widening: expands ONE action per simulation.
    """
    from src.planning.mcts import MCTSNode, _puct_score

    root_tensor = torch.tensor(root_state, dtype=torch.float32, device=device)
    root = MCTSNode(root_tensor)

    reward_fn = make_beer_game_reward_fn()
    uniform_prior = 1.0 / n_actions

    for sim in range(num_simulations):
        node = root
        path = [node]

        # Selection: descend until we find a node that needs expansion or is terminal
        while node.children and node.is_fully_expanded(n_actions):
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Expansion: expand ONE random unexpanded action (progressive widening)
        if not node.is_fully_expanded(n_actions):
            unexpanded = [a for a in range(n_actions) if a not in node.children]
            # Pick one action randomly
            action = np.random.choice(unexpanded)

            next_state = world_model.predict_next_state(
                node.state.cpu().numpy(),
                np.array(action)
            )
            child_reward = reward_fn(next_state)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
            child = MCTSNode(
                next_state_tensor,
                parent=node,
                action=action,
                prior=uniform_prior,
                reward=child_reward
            )
            node.children[action] = child
            node = child
            path.append(node)

        # Leaf evaluation
        with torch.no_grad():
            leaf_value = value_net(node.state.unsqueeze(0)).item()

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

    # Select best action by visit count
    visit_counts = np.zeros(n_actions)
    for action, child in root.children.items():
        visit_counts[action] = child.N

    return int(np.argmax(visit_counts))


def collect_episode(
    env: BeerGameEnvFlat,
    world_model: BeerGameWorldModelWrapper,
    value_net: nn.Module,
    num_simulations: int = 50,
    device: str = 'cpu',
    verbose: bool = False,
    episode_num: int = 0,
) -> Tuple[List, float]:
    """Collect one episode using MCTS policy."""
    transitions = []
    state, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        action = beer_game_mcts_search(
            state,
            world_model,
            value_net,
            n_actions=env.action_space.n,
            num_simulations=num_simulations,
            device=device,
        )

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        transitions.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state
        step += 1

        if verbose and step % 6 == 0:
            print(f"  Ep {episode_num} Step {step}/36: reward={reward:.1f}, cost={info['period_cost']:.1f}", flush=True)

    return transitions, total_reward


def train_sd_mcts(
    env: BeerGameEnvFlat,
    num_episodes: int = 100,
    num_simulations: int = 50,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr_model: float = 0.01,
    lr_value: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[BeerGameSDModel, nn.Module, List[float]]:
    """Train SD-MCTS on Beer Game."""
    # Initialize models
    world_model_torch = BeerGameSDModel(
        order_levels=env.order_levels,
        O_max=env.O_max,
    ).to(device)
    world_model = BeerGameWorldModelWrapper(world_model_torch, device)

    value_net = ValueNetwork(state_dim=env.observation_space.shape[0], hidden_dim=128).to(device)

    # Optimizers
    model_optimizer = optim.Adam(world_model_torch.parameters(), lr=lr_model)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)

    # Replay buffer
    buffer = ReplayBuffer(capacity=10000)

    # Training loop
    episode_rewards = []

    for episode in range(num_episodes):
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes} starting...", flush=True)

        # Collect episode
        transitions, ep_reward = collect_episode(
            env, world_model, value_net,
            num_simulations=num_simulations,
            device=device,
            verbose=verbose,
            episode_num=episode + 1,
        )
        episode_rewards.append(ep_reward)

        if verbose:
            print(f"Episode {episode + 1}/{num_episodes} done: reward={ep_reward:.2f}", flush=True)

        # Add to buffer
        for t in transitions:
            buffer.push(*t)

        # Training updates
        if len(buffer) >= batch_size:
            # Sample batch
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

            # World model update
            model_optimizer.zero_grad()
            predicted_next = world_model_torch(states_t, actions_t)
            model_loss = nn.MSELoss()(predicted_next, next_states_t)
            model_loss.backward()
            model_optimizer.step()

            # Value network update (TD learning)
            value_optimizer.zero_grad()
            with torch.no_grad():
                next_values = value_net(next_states_t)
                td_targets = rewards_t + gamma * next_values * (1 - dones_t)
            values = value_net(states_t)
            value_loss = nn.MSELoss()(values, td_targets)
            value_loss.backward()
            value_optimizer.step()

            if verbose:
                print(f"  Training: model_loss={model_loss.item():.4f}, value_loss={value_loss.item():.4f}", flush=True)

    return world_model_torch, value_net, episode_rewards


def evaluate_policy(
    env: BeerGameEnv,
    policy,
    n_episodes: int = 10,
    verbose: bool = False,
) -> Dict:
    """Evaluate a policy on Beer Game."""
    total_costs = []
    bullwhip_ratios = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        policy.reset() if hasattr(policy, 'reset') else None

        done = False
        while not done:
            action = policy.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        metrics = env.get_metrics()
        total_costs.append(metrics['total_cost'])
        bullwhip_ratios.append(metrics['bullwhip_ratios'])

        if verbose:
            print(f"Episode {ep + 1}: Cost = {metrics['total_cost']:.2f}")

    return {
        'total_cost_mean': np.mean(total_costs),
        'total_cost_std': np.std(total_costs),
        'total_costs': total_costs,  # Raw data for statistical tests
        'bullwhip_mean': np.mean(bullwhip_ratios, axis=0),
        'bullwhip_std': np.std(bullwhip_ratios, axis=0),
        'bullwhip_ratios': bullwhip_ratios,  # Raw data for statistical tests
    }


def compute_statistical_tests(results: Dict, baseline_name: str = 'Base-Stock') -> Dict:
    """Compute pairwise statistical significance tests.

    Uses Welch's t-test for unequal variances.
    Returns p-values for each method vs baseline.
    """
    from scipy import stats

    if baseline_name not in results or 'total_costs' not in results[baseline_name]:
        return {}

    baseline_costs = results[baseline_name]['total_costs']
    p_values = {}

    for name, res in results.items():
        if name == baseline_name or 'total_costs' not in res:
            continue

        # Welch's t-test (unequal variance)
        t_stat, p_val = stats.ttest_ind(
            res['total_costs'],
            baseline_costs,
            equal_var=False
        )
        p_values[name] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant_0.05': p_val < 0.05,
            'significant_0.01': p_val < 0.01,
        }

    return p_values


def run_experiments(args):
    """Run Beer Game experiments."""
    print("=" * 60)
    print("Beer Game Experiments for SD-MCTS")
    print("=" * 60)

    # Create environments
    env_train = BeerGameEnvFlat(
        order_levels=args.order_levels,
        T=args.episode_length,
        demand_std=args.demand_std,
        seed=args.seed,
    )
    env_eval = BeerGameEnv(
        order_levels=args.order_levels,
        T=args.episode_length,
        demand_std=args.demand_std,
        seed=args.seed + 1000,
    )

    results = {}

    # Train SD-MCTS
    if args.train:
        print("\n[1] Training SD-MCTS...")
        world_model, value_net, rewards = train_sd_mcts(
            env_train,
            num_episodes=args.train_episodes,
            num_simulations=args.num_simulations,
            device=args.device,
            verbose=True,
        )

        # Save models
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(world_model.state_dict(), save_dir / 'beer_game_world_model.pt')
        torch.save(value_net.state_dict(), save_dir / 'beer_game_value_net.pt')

        results['training_rewards'] = rewards
        print(f"Models saved to {save_dir}")

    # Evaluate baselines
    print("\n[2] Evaluating Baselines...")

    baselines = {
        'Base-Stock': BaseStockPolicy(
            order_levels=args.order_levels,
            O_max=env_eval.O_max,
        ),
        'Anchoring-Adjustment': AnchoringAdjustmentPolicy(
            order_levels=args.order_levels,
            O_max=env_eval.O_max,
        ),
        'Constant-4': ConstantOrderPolicy(
            order_rate=4,
            order_levels=args.order_levels,
            O_max=env_eval.O_max,
        ),
        'Reactive': ReactivePolicy(
            order_levels=args.order_levels,
            O_max=env_eval.O_max,
        ),
    }

    if args.mpc:
        baselines['Oracle-MPC'] = OracleMPCPolicy(
            horizon=args.mpc_horizon,
            n_samples=args.mpc_samples,
            order_levels=args.order_levels,
            O_max=env_eval.O_max,
            seed=args.seed,
        )

    for name, policy in baselines.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_policy(
            env_eval,
            policy,
            n_episodes=args.eval_episodes,
            verbose=args.verbose,
        )
        print(f"  Total Cost: {results[name]['total_cost_mean']:.2f} ± "
              f"{results[name]['total_cost_std']:.2f}")
        print(f"  Bullwhip (Retailer): {results[name]['bullwhip_mean'][0]:.2f}")
        print(f"  Bullwhip (Factory): {results[name]['bullwhip_mean'][-1]:.2f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Total Cost':>15} {'Bullwhip (R)':>15} {'Bullwhip (F)':>15}")
    print("-" * 70)
    for name, res in results.items():
        if isinstance(res, dict) and 'total_cost_mean' in res:
            print(f"{name:<25} {res['total_cost_mean']:>15.2f} "
                  f"{res['bullwhip_mean'][0]:>15.2f} "
                  f"{res['bullwhip_mean'][-1]:>15.2f}")

    # Statistical significance tests
    print("\n[3] Statistical Significance Tests (vs Base-Stock)...")
    p_values = compute_statistical_tests(results, baseline_name='Base-Stock')
    if p_values:
        print(f"\n{'Method':<25} {'t-statistic':>15} {'p-value':>15} {'Sig.(α=0.05)':>15}")
        print("-" * 70)
        for name, stats in p_values.items():
            sig_marker = "***" if stats['significant_0.01'] else ("*" if stats['significant_0.05'] else "")
            print(f"{name:<25} {stats['t_statistic']:>15.3f} "
                  f"{stats['p_value']:>15.4f} {sig_marker:>15}")
        results['statistical_tests'] = p_values

    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'beer_game_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {save_dir / 'beer_game_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Beer Game Experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')

    # Environment
    parser.add_argument('--order-levels', type=int, default=5,
                        help='Number of discrete order levels (action space = levels^4)')
    parser.add_argument('--episode-length', type=int, default=36)

    # Training
    parser.add_argument('--train', action='store_true', help='Train SD-MCTS')
    parser.add_argument('--train-episodes', type=int, default=100)
    parser.add_argument('--num-simulations', type=int, default=50)

    # Evaluation
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--demand-std', type=float, default=2.0,
                        help='Demand noise std (0 = deterministic, 2 = moderate noise)')
    parser.add_argument('--mpc', action='store_true', help='Include MPC baseline')
    parser.add_argument('--mpc-horizon', type=int, default=8)
    parser.add_argument('--mpc-samples', type=int, default=100)

    # Output
    parser.add_argument('--save-dir', type=str, default='results/beer_game')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_experiments(args)


if __name__ == '__main__':
    main()
