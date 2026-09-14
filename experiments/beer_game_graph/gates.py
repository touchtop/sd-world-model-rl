"""Planner-level gates G-BT5 (seeded determinism, paired streams) and G-BT6 (root coverage).

Run: python -m experiments.beer_game_graph.gates --out results/FILE.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.env.graph_beer_game import GraphBeerGameEnv
from experiments.beer_game_graph.common import (
    NormalizedValueNet, SequentialPlanner, build_world_model, load_topology, planner_seed,
)
from src.agent.networks import ValueNetwork


def make(topology, arm, sims, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    g = load_topology(topology)
    env = GraphBeerGameEnv(g, seed=seed, serial_layout=False)
    model = build_world_model(arm, g, env.state_dim)
    vnet = NormalizedValueNet(ValueNetwork(env.state_dim, 128), {"mean": 0.0, "std": 1.0}, env.state_dim)
    return env, SequentialPlanner(env, model, sims, "value", vnet)


def gate_bt5(topology="T4", sims=64):
    env, planner = make(topology, "sd", sims)
    s, _ = env.reset(seed=0)
    a1, d1 = planner.act(s, seed=3, period=7, return_details=True)
    a2, d2 = planner.act(s, seed=3, period=7, return_details=True)
    same = bool(np.array_equal(a1, a2)) and all(np.array_equal(x.policy, y.policy) for x, y in zip(d1, d2))
    # Paired streams: SD and BB planners draw the same planner seeds at the same (seed, period, substep).
    seeds_sd = [planner_seed(3, 7, k) for k in range(env.graph.n_nodes)]
    _, bb = make(topology, "bb", sims)
    bb.act(s, seed=3, period=7)
    seeds_bb = [planner_seed(3, 7, k) for k in range(env.graph.n_nodes)]
    return {"deterministic": same, "paired_seeds_equal": seeds_sd == seeds_bb, "action": a1.tolist()}


def gate_bt6(topology="T4", sims=64):
    env, planner = make(topology, "sd", sims)
    coverage = []
    for seed in range(3):
        s, _ = env.reset(seed=seed)
        for period in range(6):
            _, details = planner.act(s, seed=seed, period=period, return_details=True)
            coverage.append(min(int((d.policy > 0).sum()) for d in details))
            s = env.step(planner.act(s, seed=seed, period=period))[0]
    return {"min_root_children": min(coverage), "n_actions": env.graph.order_levels,
            "all_expanded": min(coverage) == env.graph.order_levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--simulations", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(1)
    report = {"simulations": args.simulations, "G-BT5": gate_bt5(sims=args.simulations),
              "G-BT6": {t: gate_bt6(t, args.simulations) for t in ("T1", "T4")}}
    report["G-BT5_pass"] = report["G-BT5"]["deterministic"] and report["G-BT5"]["paired_seeds_equal"]
    report["G-BT6_pass"] = all(v["all_expanded"] for v in report["G-BT6"].values())
    print(json.dumps(report, indent=1))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(report, indent=1))
    if not (report["G-BT5_pass"] and report["G-BT6_pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
