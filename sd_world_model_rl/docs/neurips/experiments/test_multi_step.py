#!/usr/bin/env python3
"""Quick verification for multi-step rollout experiment."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SCRIPT_DIR.parent
IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
sys.path.insert(0, str(IJOC_DIR))
sys.path.insert(0, str(NEURIPS_DIR))

import numpy as np

from run_multi_step_rollout import (
    evaluate_multi_step_error,
    OODStateEnv,
    sample_id_state,
    sample_backlog_shock,
)
from run_ood_state_generalization import train_model


def test_multi_step_evaluation():
    """Test multi-step error evaluation."""
    print("Test: Multi-step error evaluation...")

    env = OODStateEnv(order_levels=3, T=50)

    # Quick train
    sd_wm, _, _, _ = train_model(env, "sd", sample_budget=500, seed=0, verbose=False)
    bb_wm, _, _, _ = train_model(env, "mlp", sample_budget=500, seed=0, verbose=False)

    # Evaluate on ID
    print("  Evaluating SD on ID...")
    sd_id = evaluate_multi_step_error(
        sd_wm, env, sample_id_state,
        horizons=[1, 5, 10, 20], n_samples=10, device="cpu"
    )

    print("  Evaluating BB on ID...")
    bb_id = evaluate_multi_step_error(
        bb_wm, env, sample_id_state,
        horizons=[1, 5, 10, 20], n_samples=10, device="cpu"
    )

    # Evaluate on OOD
    print("  Evaluating SD on Backlog Shock...")
    sd_shock = evaluate_multi_step_error(
        sd_wm, env, sample_backlog_shock,
        horizons=[1, 5, 10, 20], n_samples=10, device="cpu"
    )

    print("  Evaluating BB on Backlog Shock...")
    bb_shock = evaluate_multi_step_error(
        bb_wm, env, sample_backlog_shock,
        horizons=[1, 5, 10, 20], n_samples=10, device="cpu"
    )

    # Print results
    print("\n  Results (RMSE):")
    print(f"  {'Horizon':<10} {'SD ID':<12} {'BB ID':<12} {'SD Shock':<12} {'BB Shock':<12}")
    print("  " + "-" * 60)
    for h in [1, 5, 10, 20]:
        print(f"  {h:<10} {sd_id[h]['mean']:<12.4f} {bb_id[h]['mean']:<12.4f} "
              f"{sd_shock[h]['mean']:<12.4f} {bb_shock[h]['mean']:<12.4f}")

    # Verify expected patterns
    print("\n  Pattern verification:")

    # SD should have near-zero error on all horizons
    for h in [1, 5, 10, 20]:
        assert sd_id[h]['mean'] < 0.01, f"SD ID error at h={h} too high: {sd_id[h]['mean']}"
    print("    ✓ SD ID error ≈ 0 across all horizons")

    # BB error should grow with horizon
    assert bb_id[20]['mean'] > bb_id[1]['mean'], "BB error should grow with horizon"
    print("    ✓ BB error grows with horizon")

    # BB OOD error should be higher than ID
    assert bb_shock[10]['mean'] > bb_id[10]['mean'], "BB OOD error > ID error"
    print("    ✓ BB OOD error > ID error")

    # BB/SD ratio should increase with horizon
    ratio_5 = bb_id[5]['mean'] / max(sd_id[5]['mean'], 1e-8)
    ratio_20 = bb_id[20]['mean'] / max(sd_id[20]['mean'], 1e-8)
    print(f"    BB/SD ratio: H=5 → {ratio_5:.1f}x, H=20 → {ratio_20:.1f}x")

    print("\n  PASSED\n")


if __name__ == "__main__":
    test_multi_step_evaluation()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
