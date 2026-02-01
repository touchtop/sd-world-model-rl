#!/usr/bin/env python3
"""Cross-validation grid search for optimal beta and F_req combination."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.delayed_collapse_env import DelayedCollapseEnv
from run_horizon_ablation import train_sd_mcts, set_seed, depth_limited_mcts_search, make_reward_fn, RandomValueNetwork


def evaluate_config(alpha, beta, F_req, T_s, true_F, belief_F, B,
                    T_train, T_eval, training_steps, h_values, num_seeds, num_simulations):
    """Evaluate a single parameter configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: beta={beta}, F_req={F_req}")
    print(f"{'='*60}")

    # Train
    set_seed(42)
    train_env = DelayedCollapseEnv(
        alpha=alpha, beta=beta, F=true_F, F_req=F_req,
        T_s=T_s, B=B, T=T_train, seed=42
    )
    world_model, policy_net, value_net = train_sd_mcts(
        train_env, true_F=true_F, init_F_belief=belief_F, env_steps=training_steps
    )

    print(f"Learned: F={world_model.collapse_threshold.item():.2f}, "
          f"alpha={world_model.alpha.item():.3f}, beta={world_model.beta.item():.3f}")

    results = {}
    for H in h_values:
        sd_success = 0
        zero_success = 0

        for seed in range(num_seeds):
            eval_env = DelayedCollapseEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                T_s=T_s, B=B, T=T_eval, seed=seed + 1000
            )
            reward_fn = make_reward_fn(eval_env.n_actions, F_req, B,
                                       F_collapse=world_model.collapse_threshold.item())

            # Test SD-MCTS
            state, _ = eval_env.reset(seed=seed + 1000)
            surge_captured = False
            collapsed = False
            for step in range(T_eval):
                with torch.no_grad():
                    action = depth_limited_mcts_search(
                        state, world_model, n_actions=eval_env.n_actions,
                        num_simulations=num_simulations, max_depth=H,
                        policy_net=policy_net, value_net=value_net, reward_fn=reward_fn
                    )
                next_state, _, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_bonus_earned"):
                    surge_captured = True
                if info.get("collapsed"):
                    collapsed = True
                    break
                state = next_state
                if terminated or truncated:
                    break
            if surge_captured and not collapsed:
                sd_success += 1

            # Test MCTS-Zero
            eval_env = DelayedCollapseEnv(
                alpha=alpha, beta=beta, F=true_F, F_req=F_req,
                T_s=T_s, B=B, T=T_eval, seed=seed + 1000
            )
            state, _ = eval_env.reset(seed=seed + 1000)
            surge_captured = False
            collapsed = False
            for step in range(T_eval):
                with torch.no_grad():
                    action = depth_limited_mcts_search(
                        state, world_model, n_actions=eval_env.n_actions,
                        num_simulations=num_simulations, max_depth=H,
                        policy_net=policy_net, value_net=None, reward_fn=reward_fn
                    )
                next_state, _, terminated, truncated, info = eval_env.step(action)
                if info.get("surge_bonus_earned"):
                    surge_captured = True
                if info.get("collapsed"):
                    collapsed = True
                    break
                state = next_state
                if terminated or truncated:
                    break
            if surge_captured and not collapsed:
                zero_success += 1

        sd_rate = sd_success / num_seeds
        zero_rate = zero_success / num_seeds
        results[H] = {"sd_mcts": sd_rate, "mcts_zero": zero_rate}
        print(f"  H={H}: SD-MCTS={sd_rate:.0%}, MCTS-Zero={zero_rate:.0%}")

    return results


def check_ideal(results, h_values):
    """Check if results meet ideal criteria."""
    # Check SD-MCTS
    sd_ok = all(results[H]["sd_mcts"] >= 0.9 for H in h_values)

    # Find critical H for MCTS-Zero
    critical_h = None
    for H in sorted(h_values):
        if results[H]["mcts_zero"] >= 0.5:
            critical_h = H
            break

    print(f"\n  >> SD-MCTS all >=90%: {sd_ok}")
    print(f"  >> MCTS-Zero critical H: {critical_h}")

    # Ideal: SD high, critical H >= 4
    is_ideal = sd_ok and critical_h is not None and critical_h >= 4
    is_good = sd_ok and critical_h is not None and critical_h >= 3

    return is_ideal, is_good, critical_h


def main():
    # Fixed parameters
    alpha = 1.0
    true_F = 10.0
    belief_F = 50.0
    T_s = 14
    B = 15.0
    T_train = 30
    T_eval = 18
    training_steps = 20000
    h_values = [1, 2, 3, 4, 5, 6, 7, 8]
    num_seeds = 50
    num_simulations = 30

    # Grid search candidates
    beta_candidates = [0.03, 0.05, 0.08]
    F_req_candidates = [4.0, 4.5, 5.0, 5.5]

    print("="*60)
    print("Cross-validation Grid Search")
    print("="*60)
    print(f"beta candidates: {beta_candidates}")
    print(f"F_req candidates: {F_req_candidates}")
    print(f"H values: {h_values}")

    best_config = None
    best_critical_h = 0

    for beta in beta_candidates:
        for F_req in F_req_candidates:
            results = evaluate_config(
                alpha=alpha, beta=beta, F_req=F_req, T_s=T_s,
                true_F=true_F, belief_F=belief_F, B=B,
                T_train=T_train, T_eval=T_eval,
                training_steps=training_steps, h_values=h_values,
                num_seeds=num_seeds, num_simulations=num_simulations
            )

            is_ideal, is_good, critical_h = check_ideal(results, h_values)

            if is_ideal:
                print(f"\n*** IDEAL CONFIG FOUND: beta={beta}, F_req={F_req}, critical_H={critical_h} ***")
                return

            if is_good and (critical_h or 0) > best_critical_h:
                best_critical_h = critical_h
                best_config = (beta, F_req, critical_h)
                print(f"\n  >> New best config: beta={beta}, F_req={F_req}, critical_H={critical_h}")

    print("\n" + "="*60)
    print("Grid Search Complete")
    print("="*60)
    if best_config:
        print(f"Best config: beta={best_config[0]}, F_req={best_config[1]}, critical_H={best_config[2]}")
    else:
        print("No good config found.")


if __name__ == "__main__":
    main()
