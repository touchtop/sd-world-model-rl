"""§8.6 item 2: zero-shot topology transfer T1 → T3 of the shared demand profile.

Run: python -m experiments.beer_game_graph.run_transfer --main results/MAIN --budget 1000 --seed 0 --out results/DIR
θ learned on T1 (perturbed prior, main-matrix checkpoint) is copied into a fresh
T3 model and planned with zero T3 interactions; controls are the T3-trained sd
and bb jobs at the same budget (read from the main matrix records).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.env.graph_beer_game import GraphBeerGameEnv
from experiments.beer_game_graph.common import (
    SequentialPlanner, build_world_model, job_name, load_topology, physical_mask, provenance, write_json,
)
from experiments.beer_game_graph.run_arm import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", required=True)
    ap.add_argument("--source", default="T1")
    ap.add_argument("--target", default="T3")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(1)
    main_dir = Path(args.main)
    out = Path(args.out) / f"{args.source}to{args.target}_transfer_b{args.budget}_s{args.seed}.json"
    if out.exists():
        print(f"skip existing {out}"); return
    gs, gt = load_topology(args.source), load_topology(args.target)
    ds = GraphBeerGameEnv(gs, serial_layout=False).state_dim
    src_ckpt = torch.load(main_dir / (job_name(args.source, "sd", args.budget, args.seed) + ".pt"))
    src = build_world_model("sd", gs, ds); src.load_state_dict(src_ckpt["model"])
    env = GraphBeerGameEnv(gt, seed=args.seed, serial_layout=False)
    model = build_world_model("sd", gt, env.state_dim)
    with torch.no_grad():
        for k in ("_demand_low_raw", "_demand_high_raw", "_demand_step_time"):
            getattr(model, k).copy_(getattr(src, k))
    t0 = time.time()
    planner = SequentialPlanner(env, model, args.simulations, "rollout", None, record_imagined=True)
    res = evaluate(env, lambda s, p, e: planner.act(s, args.seed, p, 10_000 + e),
                   args.seed, args.eval_episodes, model, physical_mask(gt), planner)
    controls = {}
    for arm in ("sd", "bb", "sd_oracle"):
        p = main_dir / (job_name(args.target, arm, args.budget, args.seed) + ".json")
        if p.exists():
            controls[arm] = json.loads(p.read_text())["eval"]["mean_return"]
    write_json(out, {"source": args.source, "target": args.target, "budget": args.budget, "seed": args.seed,
                     "theta_transferred": model.get_learned_params(), "eval": res,
                     "target_controls_mean_return": controls, "seconds": time.time() - t0,
                     "provenance": provenance([Path(__file__), Path("src/world_model/graph_beer_game_sd_model.py")])})
    print(f"[{out.stem}] transferred {res['mean_return']:.1f} controls {controls}", flush=True)


if __name__ == "__main__":
    main()
