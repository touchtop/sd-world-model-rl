#!/usr/bin/env python3
"""Quick verification for OOD state generalization experiment."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

import numpy as np

from run_ood_state_generalization import (
    OODStateEnv,
    OOD_DISTRIBUTIONS,
    sample_id_state,
    sample_backlog_shock,
    train_model,
    evaluate_prediction_error,
)


def test_ood_distributions():
    """Test OOD state sampling functions."""
    print("Test 1: OOD state distributions...")
    rng = np.random.default_rng(42)

    for name, (desc, sample_fn) in OOD_DISTRIBUTIONS.items():
        state = sample_fn(rng, n_echelons=4)
        assert state.shape == (26,), f"{name}: wrong shape {state.shape}"
        assert state[-2] == 0.0, f"{name}: t should be 0"
        assert np.all(state[:-2] >= 0), f"{name}: negative values"
        print(f"  {name}: OK (I=[{state[0]:.1f}, ...], B=[{state[1]:.1f}, ...])")

    print("  PASSED\n")


def test_ood_env():
    """Test OODStateEnv with custom initial states."""
    print("Test 2: OODStateEnv...")
    rng = np.random.default_rng(42)

    env = OODStateEnv(order_levels=3, T=36)

    # Test with backlog shock
    shock_state = sample_backlog_shock(rng, 4)
    env.set_initial_state(shock_state)
    obs, _ = env.reset()

    assert np.allclose(obs, shock_state, atol=1e-5), "State not restored correctly"
    print(f"  Backlog shock state restored: I[0]={obs[0]:.1f}, B[0]={obs[1]:.1f}")

    # Test step
    action = env.action_space.sample()
    next_obs, reward, done, _, _ = env.step(action)
    assert next_obs.shape == (26,), "Wrong obs shape after step"
    print(f"  Step OK: reward={reward:.1f}")

    print("  PASSED\n")


def test_minimal_training():
    """Test minimal training runs without error."""
    print("Test 3: Minimal training (500 steps)...")

    env = OODStateEnv(initial_state_fn=sample_id_state, order_levels=3, T=36)

    # SD
    sd_wm, sd_vn, _, _ = train_model(env, "sd", sample_budget=500, seed=0)
    print(f"  SD model trained")

    # BlackBox
    bb_wm, bb_vn, _, _ = train_model(env, "mlp", sample_budget=500, seed=0)
    print(f"  BlackBox model trained")

    print("  PASSED\n")


def test_prediction_error():
    """Test prediction error evaluation."""
    print("Test 4: Prediction error evaluation...")

    env = OODStateEnv(initial_state_fn=sample_id_state, order_levels=3, T=36)

    # Quick training
    sd_wm, _, _, _ = train_model(env, "sd", sample_budget=500, seed=0)
    bb_wm, _, _, _ = train_model(env, "mlp", sample_budget=500, seed=0)

    eval_env = OODStateEnv(order_levels=3, T=36)

    # Evaluate on ID
    sd_result = evaluate_prediction_error(
        sd_wm, eval_env, "id", sample_id_state,
        n_samples=10, rollout_horizons=[1, 5]
    )
    bb_result = evaluate_prediction_error(
        bb_wm, eval_env, "id", sample_id_state,
        n_samples=10, rollout_horizons=[1, 5]
    )

    print(f"  SD pred error (H=5): {sd_result['horizons'][5]['mean']:.4f}")
    print(f"  BB pred error (H=5): {bb_result['horizons'][5]['mean']:.4f}")

    # Evaluate on OOD (backlog shock)
    sd_ood = evaluate_prediction_error(
        sd_wm, eval_env, "backlog_shock", sample_backlog_shock,
        n_samples=10, rollout_horizons=[1, 5]
    )
    bb_ood = evaluate_prediction_error(
        bb_wm, eval_env, "backlog_shock", sample_backlog_shock,
        n_samples=10, rollout_horizons=[1, 5]
    )

    print(f"  SD OOD pred error (H=5): {sd_ood['horizons'][5]['mean']:.4f}")
    print(f"  BB OOD pred error (H=5): {bb_ood['horizons'][5]['mean']:.4f}")

    print("  PASSED\n")


if __name__ == "__main__":
    test_ood_distributions()
    test_ood_env()
    test_minimal_training()
    test_prediction_error()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
