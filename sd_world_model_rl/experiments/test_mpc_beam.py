#!/usr/bin/env python3
"""Test cases for MPC beam search implementation."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_mpc_comparison import (
    OracleDynamics, _mpc_beam_search, _mpc_exact_dp
)


def test_oracle_dynamics():
    """Test that OracleDynamics correctly predicts state transitions."""
    print("=" * 60)
    print("Test 1: OracleDynamics state transitions")
    print("=" * 60)

    dynamics = OracleDynamics(alpha=1.0, beta=0.1, F=10.0, n_actions=11)

    # Initial state: fatigue=0, alive=1, surge_flag=0, time_to_surge=12
    state = np.array([0.0, 1.0, 0.0, 12.0], dtype=np.float32)

    print(f"Initial state: {state}")

    # Test action 0 (rest): a_t = 0/10 = 0
    # new_fatigue = 0 + (1.0 + 0.1) * 0 - 0.1 = -0.1 -> clamped to 0
    next_state, reward = dynamics.predict(state, action=0)
    print(f"Action 0 (rest): next_state={next_state}, reward={reward:.2f}")
    assert next_state[0] == 0.0, f"Expected fatigue=0, got {next_state[0]}"
    assert reward == 0.0, f"Expected reward=0, got {reward}"

    # Test action 10 (max work): a_t = 10/10 = 1.0
    # new_fatigue = 0 + (1.0 + 0.1) * 1.0 - 0.1 = 1.0
    next_state, reward = dynamics.predict(state, action=10)
    print(f"Action 10 (max work): next_state={next_state}, reward={reward:.2f}")
    assert abs(next_state[0] - 1.0) < 0.01, f"Expected fatigue=1.0, got {next_state[0]}"
    assert abs(reward - 1.0) < 0.01, f"Expected reward=1.0, got {reward}"

    # Test action 5 (moderate): a_t = 5/10 = 0.5
    # new_fatigue = 0 + (1.0 + 0.1) * 0.5 - 0.1 = 0.55 - 0.1 = 0.45
    next_state, reward = dynamics.predict(state, action=5)
    print(f"Action 5 (moderate): next_state={next_state}, reward={reward:.2f}")
    assert abs(next_state[0] - 0.45) < 0.01, f"Expected fatigue=0.45, got {next_state[0]}"

    print("✓ OracleDynamics test passed\n")


def test_mpc_sees_surge():
    """Test that MPC can see and plan for surge at H >= T_s."""
    print("=" * 60)
    print("Test 2: MPC should see surge bonus at H >= 12")
    print("=" * 60)

    dynamics = OracleDynamics(alpha=1.0, beta=0.1, F=10.0, n_actions=11)

    # State at t=11 (one step before surge): time_to_surge = 1
    # If agent takes action that keeps fatigue <= 5, should get bonus
    state_near_surge = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    print(f"State near surge (t=11): {state_near_surge}")
    print(f"  time_to_surge = 1 (surge happens this step)")

    # With H=1, MPC should choose action that gets surge bonus
    # Any action that keeps fatigue <= F_req=5 should be chosen
    action_h1 = _mpc_beam_search(
        state_near_surge, dynamics, n_actions=11,
        horizon=1, gamma=0.99, F_req=5.0, B=15.0, beam_width=5
    )
    print(f"  H=1 beam search chooses action: {action_h1}")

    # Verify: with fatigue=0, any action should keep fatigue < 5
    # Max fatigue after action 10: 0 + 1.1 * 1.0 - 0.1 = 1.0 < 5 ✓
    # So optimal is action 10 (max work + surge bonus = 1.0 + 15 = 16)

    # Let's trace what happens
    for a in range(11):
        next_state, base_reward = dynamics.predict(state_near_surge, a)
        surge_bonus = 15.0 if next_state[0] <= 5.0 else 0.0
        total_reward = base_reward + surge_bonus
        if a in [0, 5, 10]:
            print(f"  Action {a}: fatigue={next_state[0]:.2f}, base_r={base_reward:.2f}, surge={surge_bonus:.0f}, total={total_reward:.2f}")

    print()

    # Now test from initial state (time_to_surge=12)
    # MPC needs H >= 12 to "see" the surge
    initial_state = np.array([0.0, 1.0, 0.0, 12.0], dtype=np.float32)
    print(f"Initial state: {initial_state}")

    # Test with different H values
    for H in [1, 6, 11, 12, 14]:
        action = _mpc_beam_search(
            initial_state, dynamics, n_actions=11,
            horizon=H, gamma=0.99, F_req=5.0, B=15.0, beam_width=5
        )
        a_t = action / 10
        print(f"  H={H}: action={action} (intensity={a_t:.1f})")

    print()


def test_beam_search_path_tracing():
    """Trace beam search to see what paths it's considering."""
    print("=" * 60)
    print("Test 3: Trace beam search paths")
    print("=" * 60)

    dynamics = OracleDynamics(alpha=1.0, beta=0.1, F=10.0, n_actions=11)
    initial_state = np.array([0.0, 1.0, 0.0, 12.0], dtype=np.float32)

    # Manually trace beam search for H=12
    n_actions = 11
    beam_width = 5
    gamma = 0.99
    F_req = 5.0
    B = 15.0

    # Initial expansion
    print("Depth 0 -> 1 expansion:")
    initial_beams = []
    for a in range(n_actions):
        next_state, reward = dynamics.predict(initial_state, a)
        time_to_surge = initial_state[3]
        # Check surge at depth 0->1: time_to_surge=12, not in (0.5, 1.5)
        # So no surge bonus here
        if next_state[1] > 0.5:
            initial_beams.append((reward, next_state.copy(), a, 1))
            if a in [0, 5, 10]:
                print(f"  Action {a}: reward={reward:.2f}, fatigue={next_state[0]:.2f}")

    # Sort and keep top beams
    beams = sorted(initial_beams, key=lambda x: -x[0])[:beam_width * 2]
    print(f"\nTop beams after depth 1: {[(b[0], b[2]) for b in beams[:5]]}")

    # The problem: high-intensity actions have higher immediate reward
    # but will collapse before reaching surge

    # Let's trace the best beam forward
    print("\nTracing top beam forward:")
    best_beam = beams[0]
    cum_value, state, first_action, depth = best_beam
    print(f"  First action: {first_action}, cumulative={cum_value:.2f}")

    for d in range(1, 12):
        # Always take action 10 (what greedy would do)
        next_state, reward = dynamics.predict(state, 10)
        time_to_surge = state[3]

        # Check surge
        if time_to_surge < 1.5 and time_to_surge > 0.5:
            if next_state[0] <= F_req:
                reward += B
                print(f"  Depth {d}: SURGE! fatigue={next_state[0]:.2f}, reward+B={reward:.2f}")

        cum_value += (gamma ** d) * reward

        if next_state[1] < 0.5:
            print(f"  Depth {d}: COLLAPSED! fatigue={next_state[0]:.2f}")
            break

        if d >= 9:
            print(f"  Depth {d}: fatigue={next_state[0]:.2f}, time_to_surge={next_state[3]:.0f}, cum={cum_value:.2f}")

        state = next_state

    print()


def test_safe_intensity_path():
    """Test that a safe intensity path can reach and capture surge."""
    print("=" * 60)
    print("Test 4: Safe intensity path analysis")
    print("=" * 60)

    dynamics = OracleDynamics(alpha=1.0, beta=0.1, F=10.0, n_actions=11)

    # Calculate safe intensity
    # Need fatigue <= F_req=5 at step 11
    # fatigue = sum of (alpha+beta)*a_t - beta = (1.1*a - 0.1) per step
    # After 11 steps with constant a: fatigue = 11 * (1.1*a - 0.1) = 12.1*a - 1.1
    # Need 12.1*a - 1.1 <= 5 => a <= 6.1/12.1 ≈ 0.504

    safe_intensity = 5.0 / 11  # roughly 0.45
    safe_action = int(round(safe_intensity * 10))  # action 5

    print(f"Safe intensity: {safe_intensity:.3f}, safe action: {safe_action}")

    # Simulate path with safe action
    state = np.array([0.0, 1.0, 0.0, 12.0], dtype=np.float32)
    total_reward = 0.0
    gamma = 0.99

    print("\nSimulating safe path (action=5):")
    for step in range(12):
        next_state, reward = dynamics.predict(state, safe_action)
        time_to_surge = state[3]

        # Check surge
        if time_to_surge < 1.5 and time_to_surge > 0.5:
            if next_state[0] <= 5.0:
                reward += 15.0
                print(f"  Step {step}: SURGE CAPTURED! fatigue={next_state[0]:.2f}")

        total_reward += (gamma ** step) * reward

        if step >= 9 or step == 0:
            print(f"  Step {step}: fatigue={next_state[0]:.2f}, time_to_surge={next_state[3]:.0f}")

        state = next_state

    print(f"\nTotal reward with safe path: {total_reward:.2f}")

    # Now compare with greedy path
    state = np.array([0.0, 1.0, 0.0, 12.0], dtype=np.float32)
    total_reward_greedy = 0.0

    print("\nSimulating greedy path (action=10):")
    for step in range(12):
        next_state, reward = dynamics.predict(state, 10)

        if next_state[1] < 0.5:
            print(f"  Step {step}: COLLAPSED! fatigue={next_state[0]:.2f}")
            break

        time_to_surge = state[3]
        if time_to_surge < 1.5 and time_to_surge > 0.5:
            if next_state[0] <= 5.0:
                reward += 15.0

        total_reward_greedy += (gamma ** step) * reward
        state = next_state

    print(f"Total reward with greedy path: {total_reward_greedy:.2f}")
    print(f"\nSafe path reward ({total_reward:.2f}) > Greedy path reward ({total_reward_greedy:.2f}): {total_reward > total_reward_greedy}")
    print()


def test_beam_search_bug():
    """Identify the bug in beam search."""
    print("=" * 60)
    print("Test 5: Identify beam search bug")
    print("=" * 60)

    # The issue: beam search keeps beams with highest CUMULATIVE value
    # But at each step, it only adds the current reward, not considering future
    # So greedy high-intensity paths dominate early

    # Look at the beam search code:
    # new_value = cum_value + (gamma ** depth) * reward
    # This only considers immediate reward, not future potential

    # For beam search to work, it needs to either:
    # 1. Use much larger beam_width to keep diverse paths
    # 2. Or use a heuristic to estimate future value

    # Let's test with larger beam_width
    dynamics = OracleDynamics(alpha=1.0, beta=0.1, F=10.0, n_actions=11)
    initial_state = np.array([0.0, 1.0, 0.0, 12.0], dtype=np.float32)

    print("Testing beam search with different beam_width:")
    for bw in [3, 5, 11, 22, 50]:
        action = _mpc_beam_search(
            initial_state, dynamics, n_actions=11,
            horizon=12, gamma=0.99, F_req=5.0, B=15.0, beam_width=bw
        )
        print(f"  beam_width={bw}: action={action}")

    print()


if __name__ == "__main__":
    test_oracle_dynamics()
    test_mpc_sees_surge()
    test_beam_search_path_tracing()
    test_safe_intensity_path()
    test_beam_search_bug()

    print("=" * 60)
    print("DIAGNOSIS: Beam search problem")
    print("=" * 60)
    print("""
The beam search has a fundamental problem:
- At each depth, it keeps beams with highest cumulative reward SO FAR
- High-intensity actions have higher immediate reward
- So all beams converge to high-intensity paths
- These paths collapse before reaching the surge

SOLUTION OPTIONS:
1. Use exact DP (slow but correct)
2. Use much larger beam_width (11^2 = 121 to keep all 2-step paths)
3. Use different MPC formulation (e.g., random shooting, CEM)
4. Accept that MPC with small horizon is fundamentally greedy
""")
