"""T5 scale audit (§8.6 item 4): prediction error, feasibility, parameter counts,
per-call timing across T1–T5, and T1→T5 shared-θ transfer from a perturbed prior.

Run: python -m experiments.beer_game_graph.run_t5_audit --out results/FILE.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.env.graph_beer_game import GraphBeerGameEnv
from experiments.beer_game_graph.common import (
    BlackBoxWorldModel, TOPOLOGY_FILES, build_world_model, load_topology, physical_mask, provenance,
    rollout_rmse, violation_rate,
)


def random_trajectory(g, seed, steps):
    env = GraphBeerGameEnv(g, seed=seed, serial_layout=False)
    s, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    S, A = [s.copy()], []
    for _ in range(steps):
        a = rng.integers(0, g.order_levels, g.n_nodes)
        s = env.step(a)[0]
        S.append(s.copy()); A.append(a)
    return np.stack(S), np.stack(A)


def fit(model, S, A, lr, epochs=200):
    params = list(model.parameters())
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)
    s, a, ns = torch.tensor(S[:-1]), torch.tensor(A), torch.tensor(S[1:])
    for _ in range(epochs):
        loss = F.mse_loss(model(s, a), ns)
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=108, help="random training periods (3 episodes)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    report = {"budget_periods": args.budget, "seeds": args.seeds, "topologies": {}}
    for name in TOPOLOGY_FILES:
        g = load_topology(name)
        env = GraphBeerGameEnv(g, serial_layout=False)
        mask = physical_mask(g)
        rows = {"state_dim": env.state_dim, "n_nodes": g.n_nodes, "n_edges": len(g.edges),
                "theta_shared": 3, "theta_node_specific": 3 * len(g.retail_nodes),
                "bb_params": sum(p.numel() for p in BlackBoxWorldModel(env.state_dim, g.n_nodes, g.order_levels).parameters()),
                "per_seed": []}
        for seed in range(args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            S, A = random_trajectory(g, seed, args.budget)
            Se, Ae = random_trajectory(g, 100 + seed, g.T)
            row = {"seed": seed}
            for arm, lr in (("sd", 1e-2), ("sd_node", 1e-2), ("bb", 1e-3)):
                m = build_world_model(arm, g, env.state_dim)
                t0 = time.time(); fit(m, S, A, lr); row[f"{arm}_fit_seconds"] = time.time() - t0
                row[f"{arm}_rmse"] = rollout_rmse(m, Se, Ae, mask)
                with torch.no_grad():
                    pred = [m(torch.tensor(Se[t])[None], torch.tensor(Ae[t])[None])[0].numpy() for t in range(len(Ae))]
                row[f"{arm}_violation"] = violation_rate(pred, mask)["rate"]
                if hasattr(m, "get_learned_params"):
                    row[f"{arm}_theta"] = m.get_learned_params()
                x = torch.tensor(np.repeat(Se[:1], 1, 0)); a = torch.tensor(Ae[:1])
                with torch.no_grad():
                    m(x, a); t0 = time.time()
                    for _ in range(100):
                        m(x, a)
                row[f"{arm}_ms_per_call"] = (time.time() - t0) * 10
            rows["per_seed"].append(row)
        report["topologies"][name] = rows
    # T1 -> T5 transfer: θ learned on T1 from a perturbed prior, moved to the T5 model.
    transfer = []
    g1, g5 = load_topology("T1"), load_topology("T5")
    d1, d5 = GraphBeerGameEnv(g1, serial_layout=False).state_dim, GraphBeerGameEnv(g5, serial_layout=False).state_dim
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        m1 = build_world_model("sd", g1, d1)
        init = m1.get_learned_params()
        S, A = random_trajectory(g1, seed, args.budget); fit(m1, S, A, 1e-2)
        m5 = build_world_model("sd", g5, d5)
        with torch.no_grad():
            for k in ("_demand_low_raw", "_demand_high_raw", "_demand_step_time"):
                getattr(m5, k).copy_(getattr(m1, k))
        Se, Ae = random_trajectory(g5, 100 + seed, g5.T)
        transfer.append({"seed": seed, "t1_theta_init": init, "t1_theta_learned": m1.get_learned_params(),
                         "t5_rmse_transferred": rollout_rmse(m5, Se, Ae, physical_mask(g5))})
    report["t1_to_t5_transfer"] = transfer
    report["provenance"] = provenance([Path(__file__), Path("src/world_model/graph_beer_game_sd_model.py")])
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, r in report["topologies"].items():
        ms = np.mean([x["sd_ms_per_call"] for x in r["per_seed"]])
        print(f"{name}: dim {r['state_dim']} sd_ms {ms:.2f} sd_rmse8 {np.mean([x['sd_rmse']['8'] for x in r['per_seed']]):.3f} "
              f"bb_rmse8 {np.mean([x['bb_rmse']['8'] for x in r['per_seed']]):.2f} bb_viol {np.mean([x['bb_violation'] for x in r['per_seed']]):.2f}")


if __name__ == "__main__":
    main()
