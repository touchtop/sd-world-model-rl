#!/usr/bin/env python
"""Run probabilistic collapse fairness test to demonstrate RL learning advantage.

Key Design:
- Fatigue accumulates as observable stock (observe_fatigue=True)
- Collapse is stochastic: P = sigmoid((fatigue - μ) / T)
- True parameters: μ=8.0, T=1.5 (collapse likely when fatigue > 8)
- Belief parameters: μ=15.0, T=3.0 (both MPC and SD-MCTS start optimistic)

Fairness Principle:
- Both MPC and SD-MCTS see the same fatigue state
- MPC uses FIXED belief sigmoid params (μ=15, T=3)
- SD-MCTS LEARNS sigmoid params from training → converges to (μ≈8, T≈1.5)
"""
from m1_framework import (
    ScenarioConfig, ExperimentRunner, SDMCTSWrapper, MPCPolicy,
    JSQPolicy, CmuPolicy, ADPRolloutPolicy, PPOWrapper,
)

if __name__ == "__main__":
    # Filter for the probabilistic collapse fairness scenario
    scenarios = [
        s for s in ScenarioConfig.generate_m1_matrix()
        if s.name == "probabilistic_collapse_fairness"
    ]

    if not scenarios:
        print("ERROR: probabilistic_collapse_fairness scenario not found!")
        exit(1)

    scenario = scenarios[0]
    print("=" * 70)
    print("Probabilistic Collapse Fairness Test")
    print("=" * 70)
    print("\nScenario Design:")
    print(f"  True collapse sigmoid center (μ) = {scenario.collapse_sigmoid_center}")
    print(f"  True collapse sigmoid temp (T) = {scenario.collapse_sigmoid_temp}")
    print(f"  Belief sigmoid center = {scenario.belief_sigmoid_center} (optimistic)")
    print(f"  Belief sigmoid temp = {scenario.belief_sigmoid_temp}")
    print(f"  observe_fatigue = {scenario.observe_fatigue} (hidden state)")
    print("\nHypothesis:")
    print("  - Both MPC and SD-MCTS see fatigue state (observe_fatigue=True)")
    print("  - Both start with SAME wrong belief (μ=15, T=3)")
    print("  - MPC uses FIXED belief -> underestimates collapse risk")
    print("  - SD-MCTS LEARNS true parameters -> correctly estimates risk")
    print("=" * 70)

    policies = [
        JSQPolicy(),
        CmuPolicy(),
        MPCPolicy(horizon=3),
        ADPRolloutPolicy(),
        PPOWrapper(),
        SDMCTSWrapper(num_simulations=50, online_update=True),
    ]

    runner = ExperimentRunner(
        scenarios=scenarios,
        policies=policies,
        env_steps_budget=15000,  # More training for learning dynamics
        num_seeds=10,
    )

    results = runner.run()

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    for scenario_name, policy_results in results.items():
        print(f"\n[{scenario_name}]")
        for policy_name, metrics in policy_results.items():
            avg = metrics.get("avg_queue_mean", 0)
            std = metrics.get("avg_queue_std", 0)
            print(f"  {policy_name}: avg_queue={avg:.2f} ± {std:.2f}")
