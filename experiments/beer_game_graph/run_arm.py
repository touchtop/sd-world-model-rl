"""Single job of the §8 protocol: one topology × arm × budget × seed → one JSON.

Run: python -m experiments.beer_game_graph.run_arm --topology T1 --arm sd --budget 1000 --seed 0 --out results/DIR
Training mirrors the paper's Beer Game loop (epsilon-greedy warm-up, replay
model updates, return-normalised value net); evaluation is 10 full episodes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agent.networks import ValueNetwork
from src.env.graph_beer_game import GraphBeerGameEnv
from experiments.beer_game_graph.baselines import SDMPC
from experiments.beer_game_graph.common import (
    ARMS, BATCH, BB_LR, EPSILON, GAMMA, MODEL_UPDATES_PER_EPISODE, REPLAY, SD_LR, VALUE_LR,
    BaseStockPolicy, NormalizedValueNet, SequentialPlanner, build_world_model, job_name,
    load_topology, physical_mask, provenance, random_steps_for, rollout_rmse,
    update_return_stats, violation_rate, write_json,
)


def train(arm, env, model, planner, value_net, stats, budget, seed, log):
    g = env.graph
    params = list(model.parameters()) if arm != "sd_oracle" else []
    model_opt = optim.Adam(params, lr=BB_LR if arm == "bb" else SD_LR) if params else None
    value_opt = optim.Adam(value_net.parameters(), lr=VALUE_LR)
    buffer, steps, episode, random_steps = [], 0, 0, random_steps_for(budget)
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    theta_log, ep_returns = [], []
    while steps < budget:
        state, _ = env.reset(seed=seed + episode)
        transitions = []
        for period in range(g.T):
            if steps < random_steps or rng.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = planner.act(state, seed, period, episode)
            nxt, reward, term, trunc, _ = env.step(action)
            transitions.append((state.copy(), np.asarray(action), reward, nxt.copy()))
            buffer = (buffer + [transitions[-1]])[-REPLAY:]
            steps += 1
            state = nxt
            if term or trunc or steps >= budget:
                break
        returns, G = [], 0.0
        for _, _, r, _ in reversed(transitions):
            G = r + GAMMA * G
            returns.insert(0, G)
        update_return_stats(stats, returns)
        if model_opt is not None and len(buffer) >= BATCH:
            for _ in range(MODEL_UPDATES_PER_EPISODE):
                idx = rng.choice(len(buffer), BATCH, replace=False)
                s = torch.tensor(np.stack([buffer[i][0] for i in idx]))
                a = torch.tensor(np.stack([buffer[i][1] for i in idx]))
                ns = torch.tensor(np.stack([buffer[i][3] for i in idx]))
                loss = F.mse_loss(model(s, a), ns)
                model_opt.zero_grad(); loss.backward(); model_opt.step()
        s = torch.tensor(np.stack([t[0] for t in transitions]))
        target = torch.tensor([(r - stats["mean"]) / stats["std"] for r in returns], dtype=torch.float32)
        vloss = F.mse_loss(value_net(s), target)
        value_opt.zero_grad(); vloss.backward(); value_opt.step()
        ep_returns.append(float(sum(t[2] for t in transitions)))
        if hasattr(model, "get_learned_params"):
            theta_log.append({"steps": steps, **{k: v for k, v in model.get_learned_params().items()
                                                   if k in ("demand_low", "demand_high", "demand_step_time")}})
        episode += 1
        if episode % 10 == 0:
            log(f"steps {steps}/{budget} ep {episode} return {ep_returns[-1]:.1f}")
    return {"episodes": episode, "train_returns": ep_returns, "theta_trajectory": theta_log,
            "random_steps": random_steps}


def evaluate(env, act, seed, n_episodes, model=None, mask=None, planner=None):
    g = env.graph
    returns, trajectories, rmse = [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed * 1000 + ep)
        states, actions, total = [state.copy()], [], 0.0
        orders = []
        for period in range(g.T):
            action = act(state, period, ep)
            state, r, term, trunc, info = env.step(action)
            total += r
            states.append(state.copy()); actions.append(np.asarray(action)); orders.append(info["orders"])
            if term or trunc:
                break
        returns.append(total)
        orders = np.array(orders)
        trajectories.append({"return": total, "order_var": orders.var(0).tolist(),
                             "final_backlog": float(env.backlog.sum())})
        if model is not None:
            rmse.append(rollout_rmse(model, np.stack(states), np.stack(actions), mask))
    out = {"returns": returns, "mean_return": float(np.mean(returns)), "episodes": trajectories}
    if rmse:
        out["rollout_rmse"] = {h: float(np.mean([r[h] for r in rmse])) for h in rmse[0]}
    if planner is not None:
        out["imagined_violation"] = violation_rate(planner.imagined, mask)
        out["model_calls"] = planner.calls
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--leaf", default="rollout", choices=("rollout", "hybrid", "value"))
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--tag", default="", help="job-name suffix for ablation variants")
    ap.add_argument("--allocation", default="backlog_proportional", choices=("backlog_proportional", "priority"),
                    help="environment allocation rule (model always assumes proportional: rule-mismatch ablation)")
    ap.add_argument("--prior-step-offset", type=float, default=0.0,
                    help="add to the SD prior's demand_step_time before training (prior-mismatch ablation)")
    ap.add_argument("--resume-checkpoints", action="store_true",
                    help="evaluate an existing .pt checkpoint when the final JSON is absent")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out) / (job_name(args.topology, args.arm, args.budget, args.seed, args.leaf) + args.tag + ".json")
    if out.exists():
        print(f"skip existing {out}"); return
    log = lambda m: print(f"[{out.stem}] {m}", flush=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    g = load_topology(args.topology)
    env = GraphBeerGameEnv(g, seed=args.seed, serial_layout=False, allocation=args.allocation)
    mask = physical_mask(g)
    t0 = time.time()
    record = {"topology": args.topology, "arm": args.arm, "budget": args.budget, "seed": args.seed,
              "simulations": args.simulations, "leaf": args.leaf, "state_dim": env.state_dim, "tag": args.tag,
              "allocation": args.allocation, "prior_step_offset": args.prior_step_offset,
              "node_demand": {n.id: n.demand for n in g.nodes}}
    if args.arm == "basestock":
        policy = BaseStockPolicy(g)
        record["eval"] = evaluate(env, lambda s, p, e: policy.act(s), args.seed, args.eval_episodes)
    elif args.arm == "sd_mpc":
        # Planner-only change: reuse the matching sd job's learned model and value net.
        ckpt = torch.load(Path(args.out) / (job_name(args.topology, "sd", args.budget, args.seed, args.leaf) + ".pt"))
        model = build_world_model("sd", g, env.state_dim); model.load_state_dict(ckpt["model"])
        value_net = ValueNetwork(env.state_dim, hidden_dim=128); value_net.load_state_dict(ckpt["value_net"])
        record.update({"theta_final": model.get_learned_params(), "source_job": ckpt["job"]})
        terminal = None if args.leaf == "rollout" else NormalizedValueNet(value_net, ckpt["stats"], env.state_dim)
        # ~parity in joint-model calls: MCTS ≈ (1 + 4) calls per simulation, MPC = 4 per sample.
        mpc = SDMPC(env, model, terminal, num_samples=(args.simulations * g.n_nodes * 5) // 4)
        record["mpc"] = {"num_samples": mpc.num_samples, "horizon_periods": mpc.horizon}
        record["eval"] = evaluate(env, lambda s, p, e: mpc.act(s, args.seed, p, 10_000 + e),
                                  args.seed, args.eval_episodes, model, mask)
        record["eval"]["model_calls"] = mpc.calls
    else:
        resumed = False
        checkpoint = out.with_suffix(".pt")
        if args.resume_checkpoints and checkpoint.exists():
            try:
                ckpt = torch.load(checkpoint, map_location="cpu")
                model = build_world_model(args.arm, g, env.state_dim)
                model.load_state_dict(ckpt["model"])
                value_net = ValueNetwork(env.state_dim, hidden_dim=128)
                value_net.load_state_dict(ckpt["value_net"])
                stats = dict(ckpt["stats"])
            except Exception as exc:
                log(f"checkpoint resume unavailable ({type(exc).__name__}: {exc}); retraining")
            else:
                record["theta_dim"] = getattr(model, "theta_dim", lambda: None)()
                record["model_params"] = sum(p.numel() for p in model.parameters())
                if hasattr(model, "get_learned_params"):
                    record["theta_final"] = model.get_learned_params()
                record["resume"] = {"checkpoint": str(checkpoint),
                                     "source_job": ckpt.get("job"),
                                     "training_history_available": False}
                record["train"] = {"resumed_from_checkpoint": str(checkpoint),
                                    "history_available": False}
                eval_planner = SequentialPlanner(
                    env, model, args.simulations, args.leaf,
                    NormalizedValueNet(value_net, stats, env.state_dim), record_imagined=True)
                t1 = time.time()
                record["eval"] = evaluate(
                    env, lambda s, p, e: eval_planner.act(s, args.seed, p, 10_000 + e),
                    args.seed, args.eval_episodes, model, mask, eval_planner)
                record["eval_seconds"] = time.time() - t1
                resumed = True
        if not resumed:
            model = build_world_model(args.arm, g, env.state_dim)
            if args.prior_step_offset and hasattr(model, "_demand_step_time"):
                with torch.no_grad():
                    model._demand_step_time += args.prior_step_offset
            record["theta_dim"] = getattr(model, "theta_dim", lambda: None)()
            record["model_params"] = sum(p.numel() for p in model.parameters())
            if hasattr(model, "get_learned_params"):
                record["theta_init"] = model.get_learned_params()
            value_net = ValueNetwork(env.state_dim, hidden_dim=128)
            stats = {"mean": 0.0, "std": 1.0, "count": 0}
            planner = SequentialPlanner(env, model, args.simulations, args.leaf,
                                        NormalizedValueNet(value_net, stats, env.state_dim))
            record["train"] = train(args.arm, env, model, planner, value_net, stats, args.budget, args.seed, log)
            record["train_seconds"] = time.time() - t0
            record["train_model_calls"] = planner.calls
            if hasattr(model, "get_learned_params"):
                record["theta_final"] = model.get_learned_params()
            torch.save({"model": model.state_dict(), "value_net": value_net.state_dict(), "stats": dict(stats),
                        "job": out.stem}, (out.parent.mkdir(parents=True, exist_ok=True), out.with_suffix(".pt"))[1])
            eval_planner = SequentialPlanner(env, model, args.simulations, args.leaf,
                                             NormalizedValueNet(value_net, stats, env.state_dim), record_imagined=True)
            t1 = time.time()
            record["eval"] = evaluate(env, lambda s, p, e: eval_planner.act(s, args.seed, p, 10_000 + e),
                                      args.seed, args.eval_episodes, model, mask, eval_planner)
            record["eval_seconds"] = time.time() - t1
    record["provenance"] = provenance([Path(__file__), Path("experiments/beer_game_graph/common.py"),
                                       Path("src/env/graph_beer_game.py"),
                                       Path("src/world_model/graph_beer_game_sd_model.py"),
                                       Path("src/planning/period_aware_mcts.py"), Path("src/planning/mcts.py")])
    record["seconds"] = time.time() - t0
    write_json(out, record)
    log(f"done: mean return {record['eval']['mean_return']:.1f} in {record['seconds']:.0f}s")


if __name__ == "__main__":
    main()
