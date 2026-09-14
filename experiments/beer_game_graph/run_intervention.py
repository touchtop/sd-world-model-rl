"""§8.6 item 3: zero-shot causal interventions on T4 (SD edits its graph; SD-no-update and BB cannot).

Run: python -m experiments.beer_game_graph.run_intervention --main results/MAIN --budget 10000 --seed 0 --out results/DIR
Loads the ``sd`` / ``bb`` checkpoints of the main matrix; no retraining.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.env.graph_beer_game import GraphBeerGameEnv
from experiments.beer_game_graph.common import (
    SequentialPlanner, build_world_model, job_name, load_topology, physical_mask, provenance, write_json,
)
from experiments.beer_game_graph.run_arm import evaluate

INTERVENTIONS = {
    "none": None,
    "capacity_shock": {"type": "capacity_shock", "node": "s0", "factor": 0.5, "t_start": 15},
    "comm_severed": {"type": "comm_severed", "edge": ["w1", "r1"], "t_start": 10, "t_end": 20},
}


def load(main: Path, topology: str, arm: str, budget: int, seed: int, g, dim):
    ckpt = torch.load(main / (job_name(topology, arm, budget, seed) + ".pt"))
    model = build_world_model(arm, g, dim)
    model.load_state_dict(ckpt["model"])
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", required=True)
    ap.add_argument("--topology", default="T4")
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out) / f"{args.topology}_interventions_b{args.budget}_s{args.seed}.json"
    if out.exists():
        print(f"skip existing {out}"); return
    g = load_topology(args.topology)
    dim = GraphBeerGameEnv(g, serial_layout=False).state_dim
    mask = physical_mask(g)
    sd = load(Path(args.main), args.topology, "sd", args.budget, args.seed, g, dim)
    bb = load(Path(args.main), args.topology, "bb", args.budget, args.seed, g, dim)
    t0 = time.time()
    record = {"topology": args.topology, "budget": args.budget, "seed": args.seed, "simulations": args.simulations,
              "leaf": "rollout", "interventions": INTERVENTIONS, "results": {}}
    for name, iv in INTERVENTIONS.items():
        record["results"][name] = {}
        for arm, model, edit in (("sd", sd, True), ("sd_no_update", sd, False), ("bb", bb, False)):
            env = GraphBeerGameEnv(g, seed=args.seed, serial_layout=False)
            env.intervention = iv
            model.intervention = iv if edit else None
            planner = SequentialPlanner(env, model, args.simulations, "rollout", None, record_imagined=True)
            res = evaluate(env, lambda s, p, e: planner.act(s, args.seed, p, 10_000 + e),
                           args.seed, args.eval_episodes, model, mask, planner)
            record["results"][name][arm] = res
            print(f"[{out.stem}] {name:15s} {arm:13s} return {res['mean_return']:9.1f} rmse8 {res['rollout_rmse']['8']:.2f}", flush=True)
    sd.intervention = bb.intervention = None
    record["provenance"] = provenance([Path(__file__), Path("src/env/graph_beer_game.py"),
                                       Path("src/world_model/graph_beer_game_sd_model.py")])
    record["seconds"] = time.time() - t0
    write_json(out, record)


if __name__ == "__main__":
    main()
