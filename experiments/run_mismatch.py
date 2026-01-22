#!/usr/bin/env python
"""Run delayed collapse scenario to demonstrate RL structural advantage over MPC."""
from m1_framework import (
    ScenarioConfig, ExperimentRunner, SDMCTSWrapper, MPCPolicy,
    JSQPolicy, CmuPolicy, ADPRolloutPolicy, PPOWrapper,
)

if __name__ == "__main__":
    # Filter for the delayed collapse scenario
    scenarios = [
        s for s in ScenarioConfig.generate_m1_matrix()
        if s.name == "delayed_collapse_structural"
    ]

    if not scenarios:
        print("ERROR: delayed_collapse_structural scenario not found!")
        exit(1)

    print("=" * 70)
    print("Delayed Collapse Experiment: Structural MPC Limitation")
    print("=" * 70)
    print("\nScenario Design:")
    print(f"  fatigue_collapse_threshold = {scenarios[0].fatigue_collapse_threshold}")
    print(f"  collapse_delay = {scenarios[0].collapse_delay} steps (> MPC horizon H=3)")
    print(f"  collapse_duration = {scenarios[0].collapse_duration} steps")
    print(f"  collapse_penalty = {scenarios[0].collapse_penalty} (80% capacity loss)")
    print(f"  observe_fatigue = {scenarios[0].observe_fatigue} (hidden state)")
    print("\nHypothesis:")
    print("  - MPC (H=3) cannot see collapse 8 steps ahead -> keeps pushing")
    print("  - SD-MCTS learns delayed consequence -> avoids overload")
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
        env_steps_budget=10000,  # More training for learning delayed dynamics
        num_seeds=10,
    )

    results = runner.run()

    # Print summary manually
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    for scenario_name, policy_results in results.items():
        print(f"\n[{scenario_name}]")
        for policy_name, metrics in policy_results.items():
            avg = metrics.get("avg_queue_mean", 0)
            std = metrics.get("avg_queue_std", 0)
            print(f"  {policy_name}: avg_queue={avg:.2f} ± {std:.2f}")
