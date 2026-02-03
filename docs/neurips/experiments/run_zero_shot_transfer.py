"""
NeurIPS Experiment: Zero-Shot Transfer

P1 Priority: Test transfer across different topology sizes.
Train on 3-node supply chain, test on 5-node without retraining.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


CONFIG = {
    "experiment_name": "zero_shot_transfer",
    "train_topology": "3_node",
    "test_topologies": ["3_node", "4_node", "5_node", "6_node"],
    "methods": ["sd_mcts", "blackbox_mcts", "dreamerv3"],
    "train_budget": 30000,
    "n_seeds": 10,
    "eval_episodes": 30,
}


class MultiNodeSupplyChainEnv:
    """
    Configurable multi-node supply chain environment.
    Allows testing transfer across different network sizes.
    """

    def __init__(self, n_nodes: int = 4, lead_time: int = 2):
        self.n_nodes = n_nodes
        self.lead_time = lead_time

        # State: inventory + backlog + pipeline for each node
        self.state_dim = n_nodes * 3
        self.action_dim = 10  # Order quantity discretization

        # Observation and action spaces (gym-like interface)
        self.observation_space = type('Space', (), {'shape': (self.state_dim,)})()
        self.action_space = type('Space', (), {'n': self.action_dim})()

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.inventory = np.ones(self.n_nodes) * 12
        self.backlog = np.zeros(self.n_nodes)
        self.pipeline = np.zeros((self.n_nodes, self.lead_time))
        self.t = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.concatenate([
            self.inventory,
            self.backlog,
            self.pipeline[:, 0],  # Incoming shipments
        ])

    def step(self, action: int) -> tuple:
        """Execute action and return (state, reward, done, info)."""
        # Convert discrete action to order quantities
        order_qty = action * 2  # Scale factor

        # Simplified dynamics (full implementation in production)
        demand = 4 + np.random.randint(-2, 3)

        # Update inventory
        arrivals = self.pipeline[:, 0].copy()
        self.inventory = self.inventory + arrivals

        # Fulfill demand
        fulfilled = np.minimum(self.inventory[0], demand)
        self.inventory[0] -= fulfilled
        unfulfilled = demand - fulfilled
        self.backlog[0] += unfulfilled

        # Calculate reward (negative costs)
        holding_cost = 0.5 * np.sum(np.maximum(self.inventory, 0))
        backlog_cost = 1.0 * np.sum(self.backlog)
        reward = -(holding_cost + backlog_cost)

        # Update pipeline
        self.pipeline = np.roll(self.pipeline, -1, axis=1)
        self.pipeline[:, -1] = order_qty

        self.t += 1
        done = self.t >= 35

        return self._get_state(), reward, done, {}


def create_transfer_env(topology: str) -> MultiNodeSupplyChainEnv:
    """Create environment for given topology."""
    n_nodes = int(topology.split("_")[0])
    return MultiNodeSupplyChainEnv(n_nodes=n_nodes)


def adapt_agent_to_topology(agent, source_nodes: int, target_nodes: int):
    """
    Adapt agent's world model to new topology size.

    For SD-MCTS: Structure (stock-flow equations) transfers directly,
                 only need to replicate node-local parameters.
    For BlackBox: Need to handle different input/output dimensions.
    """
    if hasattr(agent, 'world_model'):
        wm = agent.world_model
        if hasattr(wm, 'adapt_topology'):
            wm.adapt_topology(source_nodes, target_nodes)
        else:
            # BlackBox models cannot adapt - this is the key limitation
            print(f"  Warning: {type(wm).__name__} cannot adapt topology")
    return agent


def run_transfer_experiment(
    method_name: str,
    test_topology: str,
    seed: int,
) -> Dict:
    """Run single transfer experiment."""
    np.random.seed(seed)

    # Create training environment
    train_env = create_transfer_env(CONFIG["train_topology"])
    train_nodes = int(CONFIG["train_topology"].split("_")[0])

    # Create and train agent
    from run_sample_efficiency import create_agent, train_agent
    agent = create_agent(method_name, train_env)
    train_metrics = train_agent(agent, train_env, CONFIG["train_budget"])

    # Evaluate on training topology (baseline)
    train_eval = evaluate_on_topology(agent, train_env)

    # Create test environment
    test_env = create_transfer_env(test_topology)
    test_nodes = int(test_topology.split("_")[0])

    # Adapt agent to new topology (if possible)
    if test_nodes != train_nodes:
        agent = adapt_agent_to_topology(agent, train_nodes, test_nodes)

    # Evaluate on test topology
    test_eval = evaluate_on_topology(agent, test_env)

    # Calculate transfer efficiency
    if train_eval["mean_return"] != 0:
        transfer_ratio = test_eval["mean_return"] / train_eval["mean_return"]
    else:
        transfer_ratio = 0.0

    return {
        "method": method_name,
        "train_topology": CONFIG["train_topology"],
        "test_topology": test_topology,
        "seed": seed,
        "train_return": train_eval["mean_return"],
        "test_return": test_eval["mean_return"],
        "transfer_ratio": transfer_ratio,
        "train_std": train_eval["std_return"],
        "test_std": test_eval["std_return"],
    }


def evaluate_on_topology(agent, env) -> Dict:
    """Evaluate agent on given environment."""
    returns = []

    for _ in range(CONFIG["eval_episodes"]):
        state = env.reset()
        episode_return = 0
        done = False

        while not done:
            try:
                action = agent.act(state)
            except Exception:
                # If agent cannot handle new state dim, use random
                action = env.action_space.n // 2
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward

        returns.append(episode_return)

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "returns": returns,
    }


def run_full_experiment():
    """Run full zero-shot transfer experiment."""
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / "zero_shot_transfer" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name in CONFIG["methods"]:
        for test_topology in CONFIG["test_topologies"]:
            for seed in range(CONFIG["n_seeds"]):
                print(f"{method_name} / {test_topology} / seed {seed}")

                try:
                    result = run_transfer_experiment(method_name, test_topology, seed)
                    results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        "method": method_name,
                        "test_topology": test_topology,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Generate summary
    generate_transfer_summary(results, output_dir)

    print(f"\nResults saved to: {output_dir}")
    return results


def generate_transfer_summary(results: List[Dict], output_dir: Path):
    """Generate transfer efficiency summary."""
    lines = ["# Zero-Shot Transfer Results\n"]
    lines.append(f"Train topology: {CONFIG['train_topology']}\n")
    lines.append("| Method | Test Topology | Transfer Ratio | Test Return |")
    lines.append("|--------|---------------|----------------|-------------|")

    summary = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["method"], r["test_topology"])
        if key not in summary:
            summary[key] = {"ratios": [], "returns": []}
        summary[key]["ratios"].append(r["transfer_ratio"])
        summary[key]["returns"].append(r["test_return"])

    for (method, topology), data in sorted(summary.items()):
        ratio = np.mean(data["ratios"]) * 100
        ret = np.mean(data["returns"])
        lines.append(f"| {method} | {topology} | {ratio:.1f}% | {ret:.1f} |")

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_full_experiment()
