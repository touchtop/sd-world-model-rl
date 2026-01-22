#!/usr/bin/env python
"""Run only methodology-focused scenarios (mpc_horizon_limit, pomdp_hidden_state)."""
from m1_framework import (
    ScenarioConfig, ExperimentRunner, SDMCTSWrapper, MPCPolicy,
    JSQPolicy, CmuPolicy, ADPRolloutPolicy, PPOWrapper,
)

if __name__ == "__main__":
    scenarios = [
        s for s in ScenarioConfig.generate_m1_matrix()
        if s.name in ["mpc_horizon_limit", "pomdp_hidden_state"]
    ]
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
        env_steps_budget=8000,
        num_seeds=10,
    )
    results = runner.run()
    print("\n" + "=" * 70)
    print("Results Summary:")
    for scenario_name, policy_results in results.items():
        print(f"\n[{scenario_name}]")
        for policy_name, metrics in policy_results.items():
            avg_q = metrics.get("avg_queue_length_mean", 0)
            std_q = metrics.get("avg_queue_length_std", 0)
            print(f"  {policy_name}: avg_queue={avg_q:.2f} ± {std_q:.2f}")
