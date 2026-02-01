#!/usr/bin/env python
"""Diagnostic script for Beer Game SD-MCTS debugging.

This script runs a single episode with detailed logging to identify:
1. World model prediction accuracy
2. Value network prediction vs actual returns
3. MCTS action selection behavior
4. State evolution (inventory, backlog)
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.beer_game_env import BeerGameEnv, BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.agent import PolicyNetwork, ValueNetwork
from src.planning import mcts_search, MCTSResult

REWARD_SCALE = 1000.0


def make_beer_game_reward_fn(h: float = 0.5, p: float = 1.0, n_echelons: int = 4):
    def reward_fn(state: np.ndarray, action: int = None) -> float:
        inventory = state[0::6][:n_echelons]
        backlog = state[1::6][:n_echelons]
        holding_cost = h * np.sum(np.maximum(0, inventory))
        backlog_cost = p * np.sum(backlog)
        return -(holding_cost + backlog_cost) / REWARD_SCALE
    return reward_fn


class BeerGameSDModelWrapper(torch.nn.Module):
    def __init__(self, model: BeerGameSDModel):
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.model(states, actions)

    def parameters(self):
        return self.model.parameters()


def parse_state(state: np.ndarray, n_echelons: int = 4):
    """Parse state vector into readable components."""
    inventory = state[0::6][:n_echelons]
    backlog = state[1::6][:n_echelons]
    pipeline = state[2::6][:n_echelons]
    incoming = state[3::6][:n_echelons]
    outgoing = state[4::6][:n_echelons]
    last_order = state[5::6][:n_echelons]
    t = state[-2]
    demand = state[-1]
    return {
        'inventory': inventory,
        'backlog': backlog,
        'pipeline': pipeline,
        'incoming': incoming,
        'outgoing': outgoing,
        'last_order': last_order,
        't': t,
        'demand': demand,
    }


def decode_action(flat_action: int, order_levels: int, n_echelons: int = 4):
    """Decode flat action to per-echelon orders."""
    orders = []
    remaining = flat_action
    for i in range(n_echelons):
        orders.append(remaining % order_levels)
        remaining //= order_levels
    return np.array(orders)


def run_diagnostic():
    print("=" * 70)
    print("Beer Game SD-MCTS Diagnostic")
    print("=" * 70)

    # Create environment
    env = BeerGameEnvFlat(order_levels=3, T=36, demand_std=0.0, seed=42)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_echelons = env.n_echelons
    order_levels = env.order_levels

    print(f"State dim: {state_dim}, Actions: {n_actions}")
    print(f"Order levels: {order_levels}, Echelons: {n_echelons}")
    print(f"Order quantities mapping: {env.order_quantities}")

    # Create models (untrained)
    world_model = BeerGameSDModel(order_levels=order_levels, O_max=env.O_max)
    world_model_wrapper = BeerGameSDModelWrapper(world_model)
    policy_net = PolicyNetwork(state_dim, n_actions, hidden_dim=128)
    value_net = ValueNetwork(state_dim, hidden_dim=128)

    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=n_echelons)

    # Check world model order quantities
    print(f"\nWorld model order quantities: {world_model.order_quantities.numpy()}")
    print(f"Env order quantities: {env.order_quantities}")

    # Run single episode with detailed logging
    print("\n" + "=" * 70)
    print("Running single episode with detailed logging")
    print("=" * 70)

    state, _ = env.reset(seed=42)
    total_reward = 0.0

    for t in range(min(10, env.T)):  # First 10 steps only
        parsed = parse_state(state, n_echelons)

        print(f"\n--- Step {t} ---")
        print(f"  Time: {parsed['t']:.0f}, Demand: {parsed['demand']:.1f}")
        print(f"  Inventory: {parsed['inventory']}")
        print(f"  Backlog:   {parsed['backlog']}")
        print(f"  Pipeline:  {parsed['pipeline']}")

        # Get MCTS action with details
        result = mcts_search(
            root_state=state,
            world_model=world_model_wrapper,
            n_actions=n_actions,
            num_simulations=30,
            policy_net=policy_net,
            value_net=value_net,
            reward_fn=reward_fn,
            gamma=0.99,
            c=2.0,
            return_details=True,
        )

        mcts_action = result.action
        mcts_policy = result.policy
        root_value = result.root_value

        decoded_action = decode_action(mcts_action, order_levels, n_echelons)
        order_quantities = [env.order_quantities[a] for a in decoded_action]

        print(f"  MCTS action: {mcts_action} -> orders: {order_quantities}")
        print(f"  MCTS root value: {root_value:.4f}")
        print(f"  MCTS policy top-5: {np.argsort(mcts_policy)[-5:][::-1]} probs: {np.sort(mcts_policy)[-5:][::-1]}")

        # Value network prediction
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            value_pred = value_net(state_tensor).item()
        print(f"  Value net prediction: {value_pred:.4f}")

        # Compare random action
        random_action = env.action_space.sample()
        random_decoded = decode_action(random_action, order_levels, n_echelons)
        random_orders = [env.order_quantities[a] for a in random_decoded]
        print(f"  Random action: {random_action} -> orders: {random_orders}")

        # World model prediction vs actual
        with torch.no_grad():
            action_tensor = torch.tensor([mcts_action], dtype=torch.int64)
            pred_next = world_model(state_tensor, action_tensor).numpy()[0]

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(mcts_action)
        total_reward += reward

        # Compare prediction vs actual
        pred_error = np.mean(np.abs(pred_next - next_state))
        print(f"  Reward: {reward:.2f} (normalized: {reward/REWARD_SCALE:.4f})")
        print(f"  World model prediction error (MAE): {pred_error:.4f}")

        # Check specific state components
        pred_parsed = parse_state(pred_next, n_echelons)
        actual_parsed = parse_state(next_state, n_echelons)
        print(f"  Inv error: {np.abs(pred_parsed['inventory'] - actual_parsed['inventory'])}")
        print(f"  Backlog error: {np.abs(pred_parsed['backlog'] - actual_parsed['backlog'])}")

        state = next_state
        if terminated or truncated:
            break

    print(f"\n{'=' * 70}")
    print(f"Total reward (first 10 steps): {total_reward:.2f}")
    print(f"{'=' * 70}")

    # Now test: what if we use a simple heuristic (order = demand)?
    print("\n" + "=" * 70)
    print("Comparison: Heuristic (order = 4) vs MCTS")
    print("=" * 70)

    # Heuristic episode
    state, _ = env.reset(seed=42)
    heuristic_reward = 0.0
    for t in range(env.T):
        # Simple heuristic: order 4 for all echelons (middle level for order_levels=3)
        action = 1 * (order_levels**0) + 1 * (order_levels**1) + 1 * (order_levels**2) + 1 * (order_levels**3)
        next_state, reward, terminated, truncated, info = env.step(action)
        heuristic_reward += reward
        state = next_state
        if terminated or truncated:
            break

    print(f"Heuristic (order=4) total reward: {heuristic_reward:.2f}")

    # Random episode
    state, _ = env.reset(seed=42)
    random_reward = 0.0
    for t in range(env.T):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        random_reward += reward
        state = next_state
        if terminated or truncated:
            break

    print(f"Random total reward: {random_reward:.2f}")

    # MCTS episode (untrained, WITH value_net)
    state, _ = env.reset(seed=42)
    mcts_reward = 0.0
    for t in range(env.T):
        action = mcts_search(
            root_state=state,
            world_model=world_model_wrapper,
            n_actions=n_actions,
            num_simulations=300,  # Increased from 30
            policy_net=policy_net,
            value_net=value_net,
            reward_fn=reward_fn,
            gamma=0.99,
            c=2.0,
        )
        next_state, reward, terminated, truncated, info = env.step(action)
        mcts_reward += reward
        state = next_state
        if terminated or truncated:
            break

    print(f"MCTS (300 sims, with value_net) total reward: {mcts_reward:.2f}")

    # MCTS episode WITHOUT value_net (pure rollout)
    state, _ = env.reset(seed=42)
    mcts_no_vnet_reward = 0.0
    for t in range(env.T):
        action = mcts_search(
            root_state=state,
            world_model=world_model_wrapper,
            n_actions=n_actions,
            num_simulations=300,  # Increased from 30
            policy_net=None,  # No policy prior
            value_net=None,   # No value network - use rollout
            reward_fn=reward_fn,
            gamma=0.99,
            c=2.0,
        )
        next_state, reward, terminated, truncated, info = env.step(action)
        mcts_no_vnet_reward += reward
        state = next_state
        if terminated or truncated:
            break

    print(f"MCTS (300 sims, no value_net) total reward: {mcts_no_vnet_reward:.2f}")


if __name__ == '__main__':
    run_diagnostic()
