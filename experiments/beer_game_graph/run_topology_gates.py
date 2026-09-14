#!/usr/bin/env python
"""G-BT0..G-BT3 topology gates. Does not overwrite published T1 tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.env.beer_game_env import BeerGameEnv, BeerGameEnvSequential
from src.env.graph_beer_game import GraphBeerGameEnv, GraphBeerGameEnvSequential, load_supply_graph, serial_topology, t1_matches_beer_game
from src.planning.period_aware_mcts import SequentialDecisionWorldModel, make_sequential_reward_fn, period_aware_mcts_search
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.world_model.graph_beer_game_sd_model import GraphBeerGameSDModel, theta_dimension, build_graph_sd_model


def gate_bt0():
    env = BeerGameEnvSequential(seed=3)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.shape[0] == obs.shape[0]
    joint = BeerGameEnv(seed=3)
    joint.reset(seed=3)
    actions = [1, 2, 3, 4]
    for a in actions[:-1]:
        o, r, _, _, info = env.step(a)
        assert r == 0.0 and info["substep"]
    o, r, _, _, info = env.step(actions[-1])
    jo, jr, _, _, _ = joint.step(np.array(actions))
    assert np.allclose(o[:-1], jo) and np.isclose(r, jr) and info["substep"] is False
    return True


def gate_bt1():
    return bool(t1_matches_beer_game(seed=11, steps=12))


def gate_bt2():
    classic = BeerGameSDModel(learn_demand=False)
    graph = serial_topology(4)
    generic = GraphBeerGameSDModel(graph, learn_demand=False)
    rng = np.random.default_rng(4)
    for _ in range(6):
        state = rng.random((2, 26)).astype(np.float32)
        state[:, 0::6] = np.abs(state[:, 0::6]) * 10
        action = rng.integers(0, 9, size=(2, 4))
        s = torch.tensor(state)
        a = torch.tensor(action)
        with torch.no_grad():
            if not torch.allclose(classic(s, a), generic(s, a), atol=1e-5):
                return False
    return True


def gate_bt3_smoke():
    """Smoke, not Table 1/2 reproduction. Confirms sequential MCTS runs on T1."""
    env = BeerGameEnv(seed=0)
    model = BeerGameSDModel(learn_demand=False)
    wrapper = SequentialDecisionWorldModel(model, joint_dim=26, period_length=4)

    def reward_fn(state, action=None):
        joint = np.asarray(state).reshape(-1)[:26]
        inv = joint[0::6][:4]
        back = joint[1::6][:4]
        return -float(0.5 * np.sum(np.maximum(0, inv)) + np.sum(back))

    seq_reward = make_sequential_reward_fn(reward_fn, period_length=4, joint_dim=26)
    root = np.concatenate([env.reset(seed=0)[0], [0.0], np.zeros(4, dtype=np.float32)])
    action = period_aware_mcts_search(
        root_state=root,
        world_model=wrapper,
        n_actions=env.order_levels,
        num_simulations=8,
        gamma=0.99,
        reward_fn=seq_reward,
        period_length=4,
        substep_index=26,
        rollout_depth=4,
    )
    return int(action) in range(env.order_levels)


def scale_audit():
    rows = []
    root = Path(__file__).parent / "topologies"
    for name in ("t1_serial4.yaml", "t2_serial8.yaml", "t3_fork.yaml", "t4_dag.yaml", "t5_scale.yaml"):
        graph = load_supply_graph(root / name)
        env = GraphBeerGameEnv(graph, seed=0, serial_layout=False)
        env.reset(seed=0)
        rows.append(
            {
                "id": graph.id,
                "n_nodes": graph.n_nodes,
                "n_edges": len(graph.edges),
                "n_retail": len(graph.retail_nodes),
                "theta_shared": theta_dimension(len(graph.retail_nodes), True),
                "theta_node_specific": theta_dimension(len(graph.retail_nodes), False),
                "state_dim": env.state_dim,
                "sd_param_count": {
                    "shared": sum(p.numel() for p in build_graph_sd_model(graph, shared_demand=True).parameters()),
                    "node_specific": sum(p.numel() for p in build_graph_sd_model(graph, shared_demand=False).parameters()),
                },
                "node_demand": {n.id: n.demand for n in graph.nodes},
                "source_capacity": {n.id: n.capacity for n in graph.source_nodes},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = {
        "G-BT0": gate_bt0(),
        "G-BT1": gate_bt1(),
        "G-BT2": gate_bt2(),
        "G-BT3_smoke": gate_bt3_smoke(),
        "scale_audit": scale_audit(),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    if not all(report[k] for k in ("G-BT0", "G-BT1", "G-BT2", "G-BT3_smoke")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
