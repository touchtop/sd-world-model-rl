#!/usr/bin/env python3
"""Quick test for demand transfer experiment setup.

Verifies:
1. Demand patterns generate correctly
2. Environment wrapper works
3. Can import training infrastructure
4. Minimal training run executes without errors
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

import numpy as np
import torch

from demand_patterns import create_demand_pattern
from run_demand_transfer import create_env_with_pattern, CONFIG
from experiments.run_beer_game_v2 import train_sd_mcts


def test_demand_patterns():
    """Test all demand pattern generators."""
    print("="*60)
    print("Test 1: Demand Pattern Generators")
    print("="*60)

    patterns = ["step", "sinusoidal", "random_walk", "bursty"]

    for pattern_type in patterns:
        print(f"\nTesting {pattern_type}...")
        pattern = create_demand_pattern(pattern_type, seed=42)

        # Generate 20 demand values
        demands = []
        for _ in range(20):
            d = pattern.get_demand()
            demands.append(d)

        print(f"  Generated: {demands[:10]}... (showing first 10)")
        print(f"  Mean: {np.mean(demands):.2f}, Std: {np.std(demands):.2f}")
        print(f"  Min: {min(demands)}, Max: {max(demands)}")

        # Test reset
        pattern.reset()
        d_after_reset = pattern.get_demand()
        print(f"  After reset: {d_after_reset}")

    print("\n✅ All demand patterns work!")


def test_environment_wrapper():
    """Test DemandPatternEnv wrapper."""
    print("\n" + "="*60)
    print("Test 2: Environment Wrapper")
    print("="*60)

    for pattern_type in ["step", "sinusoidal"]:
        print(f"\nTesting {pattern_type} environment...")
        env = create_env_with_pattern(pattern_type, seed=42)

        # Check spaces
        print(f"  State dim: {env.observation_space.shape[0]}")
        print(f"  Action space: {env.action_space.n} (flat)")

        # Run a few steps
        state, _ = env.reset(seed=42)
        demands = []
        for _ in range(10):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            demands.append(env.current_demand)
            state = next_state
            if terminated or truncated:
                break

        print(f"  Demands observed: {demands}")
        print(f"  Episode reward: ~{reward:.1f} (last step)")

    print("\n✅ Environment wrapper works!")


def test_training_minimal():
    """Test minimal training run (100 steps)."""
    print("\n" + "="*60)
    print("Test 3: Minimal Training Run")
    print("="*60)

    print("\nTraining SD-MCTS for 100 steps...")

    np.random.seed(42)
    torch.manual_seed(42)

    env = create_env_with_pattern("step", seed=42)

    world_model, policy_net, value_net, episode_rewards = train_sd_mcts(
        env=env,
        env_steps=100,
        gamma=0.99,
        device='cpu',
        verbose=True,
    )

    print(f"\n  Training completed!")
    print(f"  Episodes: {len(episode_rewards)}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Final 3 rewards: {episode_rewards[-3:]}")

    print("\n✅ Training infrastructure works!")


def test_full_integration():
    """Test one complete transfer experiment (step → sinusoidal)."""
    print("\n" + "="*60)
    print("Test 4: Full Integration (Mini)")
    print("="*60)

    from run_demand_transfer import run_single_experiment

    # Temporarily reduce budget for quick test
    original_budget = CONFIG["train_budget"]
    original_eval = CONFIG["eval_episodes"]
    CONFIG["train_budget"] = 500
    CONFIG["eval_episodes"] = 3

    print("\nRunning mini transfer experiment (step → sinusoidal)...")
    print("  Training budget: 500 steps")
    print("  Eval episodes: 3")

    try:
        result = run_single_experiment(
            method="sd_mcts",
            test_pattern="sinusoidal",
            seed=42,
        )

        print(f"\n  ✅ Experiment completed!")
        print(f"  Train cost: {result.train_cost_mean:.1f}")
        print(f"  Test cost: {result.test_cost_mean:.1f}")
        print(f"  Transfer ratio: {result.transfer_ratio:.2f}x")

    finally:
        # Restore config
        CONFIG["train_budget"] = original_budget
        CONFIG["eval_episodes"] = original_eval

    print("\n✅ Full integration works!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Demand Transfer Experiment - Setup Verification")
    print("="*60 + "\n")

    try:
        test_demand_patterns()
        test_environment_wrapper()
        test_training_minimal()
        test_full_integration()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nReady to run full experiment:")
        print("  python run_demand_transfer.py --seeds 5 --budget 30000")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
