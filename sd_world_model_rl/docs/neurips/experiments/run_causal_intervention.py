#!/usr/bin/env python3
"""
NeurIPS Experiment: Causal Intervention / Topology Transfer

Demonstrates that SD's causal graph structure allows for zero-shot 
adaptation to causal interventions (e.g., node capacity shocks, severed links),
whereas BlackBox (MLP) implicitly bakes these structures into its weights,
causing catastrophic failure when the causal graph changes.

Interventions:
1. capacity_shock: Factory capacity drops from 20 to 2 at t>=15.
2. comm_severed: Wholesaler (echelon 1) communication is severed (orders=0) for t in [10, 20].
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

from src.env.beer_game_env import BeerGameEnvFlat
from src.world_model.beer_game_sd_model import BeerGameSDModel, BeerGameWorldModelWrapper
from src.agent.networks import ValueNetwork
from src.planning.mcts import mcts_search

from experiments.run_beer_game_v2 import (
    BeerGameSDModelWrapper,
    make_beer_game_reward_fn,
    NormalizedValueNet,
    ReplayBuffer,
    Transition,
)
from baselines.blackbox_world_model import BlackBoxWorldModel
from run_ood_state_generalization import sample_id_state


# =============================================================================
# Intervention Environments
# =============================================================================

class BeerGameEnvIntervention(BeerGameEnvFlat):
    """Beer Game environment with causal interventions."""

    def __init__(self, intervention_type: str = "none", **kwargs):
        super().__init__(**kwargs)
        self.intervention_type = intervention_type

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Decode action to per-echelon orders
        multi_action = self._decode_action(action)
        
        # Apply Causal Interventions BEFORE original step logic
        
        # 1. Communication Severed (Edge removal)
        if self.intervention_type == "comm_severed":
            if 10 <= self.t <= 20:
                multi_action[1] = 0  # Echelon 1 (Wholesaler) cannot place orders
                
        # 2. Capacity Shock (Node capacity constraint change)
        original_cap = self.production_capacity
        if self.intervention_type == "capacity_shock":
            if self.t >= 15:
                self.production_capacity = 2  # Severe drop in factory capacity
                
        # Execute step
        obs, reward, terminated, truncated, info = super(BeerGameEnvFlat, self).step(multi_action)
        
        # Restore original capacity parameter
        self.production_capacity = original_cap
        
        return obs, reward, terminated, truncated, info


class BeerGameSDModelIntervention(BeerGameSDModel):
    """SD World Model updated with the causal intervention prior."""

    def __init__(self, intervention_type: str = "none", **kwargs):
        super().__init__(**kwargs)
        self.intervention_type = intervention_type

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device

        echelon_states = state[:, :self.n_echelons * 6].view(batch_size, self.n_echelons, 6)
        inventory = echelon_states[:, :, 0]
        backlog = echelon_states[:, :, 1]
        ship_0 = echelon_states[:, :, 2]
        ship_1 = echelon_states[:, :, 3]
        order_0 = echelon_states[:, :, 4]
        order_1 = echelon_states[:, :, 5]

        t = state[:, -2]
        current_demand = state[:, -1]

        if action.dim() == 1:
            orders = self._decode_flat_action(action)
        else:
            orders = self.order_quantities[action.long()]

        # === APPLY INTERVENTION PRIOR TO ORDERS ===
        if self.intervention_type == "comm_severed":
            # If 10 <= t <= 20, Wholesaler orders are 0
            is_severed = (t >= 10) & (t <= 20)
            orders[:, 1] = torch.where(is_severed, torch.zeros_like(orders[:, 1]), orders[:, 1])

        # Step 1-3
        arrivals = ship_0.clone()
        new_ship_0 = ship_1.clone()
        new_ship_1 = torch.zeros_like(ship_1)

        incoming_orders = torch.zeros(batch_size, self.n_echelons, device=device)
        incoming_orders[:, 0] = current_demand
        for i in range(1, self.n_echelons):
            incoming_orders[:, i] = order_0[:, i - 1]

        # Step 4-5
        new_inventory = inventory + arrivals
        new_backlog = backlog + incoming_orders
        shipments = torch.minimum(new_inventory, new_backlog)
        new_inventory = new_inventory - shipments
        new_backlog = new_backlog - shipments

        # Step 6
        for i in range(self.n_echelons - 1):
            new_ship_1[:, i] = shipments[:, i + 1]

        # === APPLY INTERVENTION PRIOR TO CAPACITY ===
        if self.intervention_type == "capacity_shock":
            dynamic_cap = torch.where(t >= 15, torch.tensor(2.0, device=device), torch.tensor(float(self.production_capacity), device=device))
            factory_production = torch.minimum(orders[:, -1], dynamic_cap)
        else:
            factory_production = torch.clamp(orders[:, -1], max=self.production_capacity)
            
        new_ship_1[:, -1] = factory_production

        # Step 7
        new_order_0 = order_1.clone()
        new_order_1 = torch.zeros_like(order_1)
        new_order_1 = new_order_1 + orders

        new_inventory = torch.clamp(new_inventory, min=0)
        new_backlog = torch.clamp(new_backlog, min=0)
        new_ship_0 = torch.clamp(new_ship_0, min=0)
        new_ship_1 = torch.clamp(new_ship_1, min=0)
        new_order_0 = torch.clamp(new_order_0, min=0)
        new_order_1 = torch.clamp(new_order_1, min=0)

        new_echelon_states = torch.stack([
            new_inventory, new_backlog, new_ship_0, new_ship_1, new_order_0, new_order_1
        ], dim=2)

        next_demand = self.get_demand(t + 1)
        new_state = torch.cat([
            new_echelon_states.view(batch_size, -1),
            (t + 1).unsqueeze(1),
            next_demand.unsqueeze(1),
        ], dim=1)

        return new_state


# =============================================================================
# Training / Evaluation
# =============================================================================

def train_model(
    env: BeerGameEnvFlat,
    model_type: str,
    sample_budget: int,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> Tuple:
    """Train SD or BlackBox model on the base (no intervention) environment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if model_type == "sd":
        # We use our custom intervention SD model, but initialize with "none" for training
        sd_model = BeerGameSDModelIntervention(
            intervention_type="none",
            order_levels=env.order_levels,
            O_max=env.O_max,
        ).to(device)
        world_model = BeerGameSDModelWrapper(sd_model)
        model_opt = optim.Adam(sd_model.parameters(), lr=1e-2) if list(sd_model.parameters()) else None
        is_mlp = False
    else:
        world_model = BlackBoxWorldModel(state_dim, n_actions, hidden_dim=256).to(device)
        model_opt = optim.Adam(world_model.parameters(), lr=1e-3)
        is_mlp = True

    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)
    value_opt = optim.Adam(value_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=5000)
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    return_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def get_return_stats():
        return return_stats['mean'], return_stats['std']

    normalized_vnet = NormalizedValueNet(value_net, get_return_stats)
    gamma = 0.99
    epsilon = 0.3
    random_steps = min(2000, max(500, sample_budget // 3))

    steps = 0
    episode = 0
    last_log_steps = 0

    if verbose:
        print(f"    Training {model_type.upper()} (budget={sample_budget}, random_steps={random_steps})")

    while steps < sample_budget:
        state, _ = env.reset(seed=seed + episode)
        episode_transitions = []

        for t in range(env.T):
            if steps < random_steps or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = mcts_search(
                    root_state=state,
                    world_model=world_model,
                    n_actions=n_actions,
                    num_simulations=200,
                    value_net=normalized_vnet,
                    reward_fn=reward_fn,
                    rollout_depth=10,
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1

            transition = Transition(
                state=state.copy(), action=action, reward=reward,
                next_state=next_state.copy(), done=terminated or truncated,
            )
            episode_transitions.append(transition)
            buffer.add(transition)

            state = next_state
            if terminated or truncated:
                break

        # Compute returns
        G = 0.0
        returns = []
        for tr in reversed(episode_transitions):
            G = tr.reward + gamma * G
            returns.insert(0, G)

        for r in returns:
            return_stats['count'] += 1
            delta = r - return_stats['mean']
            return_stats['mean'] += delta / return_stats['count']
            if return_stats['count'] > 1:
                return_stats['std'] = max(1.0, np.sqrt(
                    ((return_stats['count'] - 1) * return_stats['std']**2 + delta * (r - return_stats['mean']))
                    / return_stats['count']
                ))

        # Train models
        if len(buffer) >= 32:
            for _ in range(5):
                batch = buffer.sample(32)
                states_t = torch.tensor(np.array([tr.state for tr in batch]), dtype=torch.float32, device=device)
                actions_t = torch.tensor([tr.action for tr in batch], dtype=torch.int64, device=device)
                next_states_t = torch.tensor(np.array([tr.next_state for tr in batch]), dtype=torch.float32, device=device)

                if is_mlp:
                    pred_next = world_model(states_t, actions_t)
                else:
                    pred_next = world_model.model(states_t, actions_t)

                loss = F.mse_loss(pred_next, next_states_t)
                if model_opt:
                    model_opt.zero_grad()
                    loss.backward()
                    model_opt.step()

        if len(episode_transitions) > 0:
            ep_states = torch.tensor(np.array([tr.state for tr in episode_transitions]), dtype=torch.float32, device=device)
            norm_returns = [(r - return_stats['mean']) / return_stats['std'] for r in returns]
            ep_returns = torch.tensor(norm_returns, dtype=torch.float32, device=device)

            val_pred = value_net(ep_states).squeeze(-1)
            val_loss = F.mse_loss(val_pred, ep_returns)

            value_opt.zero_grad()
            val_loss.backward()
            value_opt.step()

        episode += 1

        # Progress logging
        if verbose and steps - last_log_steps >= 1000:
            print(f"      Steps {steps}/{sample_budget} | Episode {episode} | Recent return: {sum(returns)/len(returns):.1f}")
            last_log_steps = steps

    return world_model, value_net, normalized_vnet, return_stats


def evaluate_policy(
    world_model,
    value_net,
    env: BeerGameEnvIntervention,
    n_episodes: int = 20,
    seed_offset: int = 0,
) -> Dict:
    """Evaluate MCTS policy on the given environment."""
    n_actions = env.action_space.n
    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)

    returns = []
    
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_offset + ep)
        episode_return = 0.0
        done = False

        while not done:
            action = mcts_search(
                root_state=state,
                world_model=world_model,
                n_actions=n_actions,
                num_simulations=200,
                value_net=value_net,
                reward_fn=reward_fn,
                rollout_depth=10,
            )
            state, reward, term, trunc, _ = env.step(action)
            episode_return += reward
            done = term or trunc

        returns.append(episode_return)

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "returns": returns
    }


def evaluate_prediction_rmse(
    world_model,
    env: BeerGameEnvIntervention,
    horizon: int = 15,
    warmup_steps: int = 10,
) -> float:
    """Measure world model RMSE over a multi-step rollout.
    
    Args:
        warmup_steps: Number of steps to advance the environment before
            starting the prediction evaluation. This ensures the RMSE
            window covers intervention periods (e.g., capacity_shock at t>=15,
            comm_severed at t in [10,20]). With warmup=10 and horizon=15,
            the evaluation covers t=10..24.
    """
    state, _ = env.reset(seed=42)
    is_mlp = not hasattr(world_model, 'model')
    device = next(world_model.parameters()).device

    # Warmup: advance environment to the desired starting time step
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        state, _, term, trunc, _ = env.step(action)
        if term or trunc:
            return 0.0  # Episode ended during warmup

    cumulative_error = 0.0
    pred_state = state.copy()
    steps_evaluated = 0
    
    for step in range(horizon):
        action = env.action_space.sample()
        
        with torch.no_grad():
            state_t = torch.tensor(pred_state, dtype=torch.float32, device=device).unsqueeze(0)
            action_t = torch.tensor([action], dtype=torch.int64, device=device)
            if is_mlp:
                pred_next = world_model(state_t, action_t).squeeze(0).cpu().numpy()
            else:
                pred_next = world_model.model(state_t, action_t).squeeze(0).cpu().numpy()

        actual_next, _, term, trunc, _ = env.step(action)
        
        # State idx 0-23 are I, B, pipelines
        error = np.mean((pred_next[:24] - actual_next[:24]) ** 2)
        cumulative_error += error
        steps_evaluated += 1
        
        pred_state = pred_next
        if term or trunc:
            break
            
    return float(np.sqrt(cumulative_error / max(steps_evaluated, 1)))


# =============================================================================
# Main Experiment Controller
# =============================================================================

def run_causal_intervention_experiment(
    sample_budget: int = 10000,
    n_seeds: int = 3,
    eval_episodes: int = 20,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    verbose: bool = False,
):
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = NEURIPS_DIR / "results" / "causal_intervention" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Causal Intervention / Topology Transfer Experiment (Zero-Shot)")
    print("=" * 80)
    print(f"Sample budget: {sample_budget}")
    print(f"Seeds: {n_seeds}")
    print(f"Interventions: ['none', 'capacity_shock', 'comm_severed']")
    
    interventions = ["none", "capacity_shock", "comm_severed"]
    all_results = []

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Train on "none" (standard setup)
        train_env = BeerGameEnvIntervention(intervention_type="none", order_levels=3, T=36)
        
        print("Training SD-MCTS (no constraints)...")
        sd_wm, sd_vn, sd_nvn, _ = train_model(train_env, "sd", sample_budget, device, seed, verbose=verbose)
        print("Training BlackBox-MCTS...")
        bb_wm, bb_vn, bb_nvn, _ = train_model(train_env, "mlp", sample_budget, device, seed, verbose=verbose)

        seed_results = {"seed": seed}

        for inv in interventions:
            print(f"\n  Evaluating Intervention: '{inv}'")
            eval_env = BeerGameEnvIntervention(intervention_type=inv, order_levels=3, T=36)
            
            # --- Zero-Shot Adaptation ---
            # SD Model gets the graph/rule update immediately
            sd_wm.model.intervention_type = inv 
            # BlackBox has NO mechanism to update its embedded graph!

            # RMSE Error Evaluation
            sd_rmse = evaluate_prediction_rmse(sd_wm, eval_env, horizon=15)
            # Recreate eval_env to match same sequence
            eval_env = BeerGameEnvIntervention(intervention_type=inv, order_levels=3, T=36) 
            bb_rmse = evaluate_prediction_rmse(bb_wm, eval_env, horizon=15)

            # SD_no_update control: SD model WITHOUT graph editing (keeps intervention_type="none")
            sd_wm.model.intervention_type = "none"  # Revert to no-update
            eval_env_ctrl = BeerGameEnvIntervention(intervention_type=inv, order_levels=3, T=36)
            sd_no_update_rmse = evaluate_prediction_rmse(sd_wm, eval_env_ctrl, horizon=15)
            
            # Policy Evaluation
            print(f"    Running MCTS eval for {eval_episodes} episodes (3 variants)...")
            # 1. SD with graph update
            sd_wm.model.intervention_type = inv  # Re-apply update for policy eval
            sd_perf = evaluate_policy(sd_wm, sd_nvn, eval_env, n_episodes=eval_episodes, seed_offset=seed*100)
            # 2. BB (no adaptation)
            bb_perf = evaluate_policy(bb_wm, bb_nvn, eval_env, n_episodes=eval_episodes, seed_offset=seed*100)
            # 3. SD_no_update control (SD model, but no graph editing)
            sd_wm.model.intervention_type = "none"  # Keep as "none" during eval
            eval_env_ctrl2 = BeerGameEnvIntervention(intervention_type=inv, order_levels=3, T=36)
            sd_no_update_perf = evaluate_policy(sd_wm, sd_nvn, eval_env_ctrl2, n_episodes=eval_episodes, seed_offset=seed*100)
            
            # Restore SD to updated state for next iteration
            sd_wm.model.intervention_type = inv
            
            print(f"    [RMSE] SD: {sd_rmse:.2f} | SD_no_upd: {sd_no_update_rmse:.2f} | BB: {bb_rmse:.2f}")
            print(f"    [RET ] SD: {sd_perf['mean']:.1f} ± {sd_perf['std']:.1f} | SD_no_upd: {sd_no_update_perf['mean']:.1f} ± {sd_no_update_perf['std']:.1f} | BB: {bb_perf['mean']:.1f} ± {bb_perf['std']:.1f}")
            
            seed_results[inv] = {
                "rmse": {"sd": sd_rmse, "sd_no_update": sd_no_update_rmse, "bb": bb_rmse},
                "policy": {"sd": sd_perf, "sd_no_update": sd_no_update_perf, "bb": bb_perf}
            }
            
        all_results.append(seed_results)

    # Aggregation and Save
    print("\n" + "=" * 80)
    print("AGGREGATED CAUSAL INTERVENTION RESULTS")
    print("=" * 80)
    
    print(f"{'Intervention':<20} | {'SD Return':<12} | {'SD_noUpd Ret':<14} | {'BB Return':<12} | {'SD RMSE':<8} | {'SD_noUpd RMSE':<14} | {'BB RMSE':<8}")
    print("-" * 110)
    
    for inv in interventions:
        sd_ret = np.mean([r[inv]["policy"]["sd"]["mean"] for r in all_results])
        sd_nu_ret = np.mean([r[inv]["policy"]["sd_no_update"]["mean"] for r in all_results])
        bb_ret = np.mean([r[inv]["policy"]["bb"]["mean"] for r in all_results])
        sd_rmse = np.mean([r[inv]["rmse"]["sd"] for r in all_results])
        sd_nu_rmse = np.mean([r[inv]["rmse"]["sd_no_update"] for r in all_results])
        bb_rmse = np.mean([r[inv]["rmse"]["bb"] for r in all_results])
        
        print(f"{inv:<20} | {sd_ret:<12.1f} | {sd_nu_ret:<14.1f} | {bb_ret:<12.1f} | {sd_rmse:<8.2f} | {sd_nu_rmse:<14.2f} | {bb_rmse:<8.2f}")

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run_causal_intervention_experiment(
        sample_budget=args.budget,
        n_seeds=args.seeds,
        eval_episodes=10,
        device=args.device,
        verbose=args.verbose,
    )
