"""External baselines for the §8 protocol: PPO (stable-baselines3) and SD-MPC.

PPO: python -m experiments.beer_game_graph.baselines ppo --topology T1 --budget 1000 --seed 0 --out results/DIR
SD-MPC is invoked through ``run_arm --arm sd_mpc`` and reuses the learned SD
model and value network of the matching ``sd`` job (planner-only change).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.env.graph_beer_game import GraphBeerGameEnv, SupplyGraph
from experiments.beer_game_graph.common import (
    GAMMA, job_name, load_topology, planner_seed, provenance, write_json,
)


class BeerGameGym(gym.Env):
    """Gymnasium wrapper on the generic layout with demand-scaled observations."""

    def __init__(self, graph: SupplyGraph, seed: int = 0):
        self.env = GraphBeerGameEnv(graph, seed=seed, serial_layout=False)
        self.action_space = gym.spaces.MultiDiscrete([graph.order_levels] * graph.n_nodes)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.env.state_dim,), dtype=np.float32)
        self.scale = np.full(self.env.state_dim, 4.0 * max(n.demand for n in graph.nodes), dtype=np.float32)
        self.scale[-1 - len(graph.retail_nodes)] = graph.T
        self.reward_scale = 1.0 / (10.0 * sum(n.demand for n in graph.nodes))
        self._episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed if seed is not None else self._episode)
        self._episode += 1
        return obs / self.scale, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(np.asarray(action))
        info["raw_reward"] = r
        return obs / self.scale, r * self.reward_scale, term, trunc, info


def run_ppo(topology: str, budget: int, seed: int, out: Path, eval_episodes: int = 10, device: str = "cpu"):
    from stable_baselines3 import PPO
    import stable_baselines3

    path = out / (job_name(topology, "ppo", budget, seed) + ".json")
    if path.exists():
        print(f"skip existing {path}"); return
    g = load_topology(topology)
    torch.manual_seed(seed); np.random.seed(seed)
    env = BeerGameGym(g, seed=seed)
    n_steps = min(budget, 10 * g.T)
    t0 = time.time()
    model = PPO("MlpPolicy", env, seed=seed, device=device, n_steps=n_steps, batch_size=g.T,
                n_epochs=10, gamma=GAMMA, learning_rate=3e-4, verbose=0)
    model.learn(total_timesteps=budget)
    train_seconds = time.time() - t0
    returns = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            total += info["raw_reward"]; done = term or trunc
        returns.append(total)
    write_json(path, {
        "topology": topology, "arm": "ppo", "budget": budget, "seed": seed, "leaf": "rollout",
        "state_dim": env.env.state_dim, "sb3_version": stable_baselines3.__version__,
        "ppo": {"n_steps": n_steps, "batch_size": g.T, "n_epochs": 10, "lr": 3e-4, "gamma": GAMMA},
        "train_seconds": train_seconds, "model_params": sum(p.numel() for p in model.policy.parameters()),
        "eval": {"returns": returns, "mean_return": float(np.mean(returns))},
        "provenance": provenance([Path(__file__), Path("src/env/graph_beer_game.py")]),
        "seconds": time.time() - t0,
    })
    print(f"[{path.stem}] done: mean return {np.mean(returns):.1f} in {time.time() - t0:.0f}s", flush=True)


class SDMPC:
    """Receding-horizon random shooting on the learned joint SD model with terminal V.

    Same model, value network, horizon (4 periods) and simulation budget as the
    MCTS arm; only the planner changes (rebuttal Q4 attribution).
    """

    def __init__(self, env: GraphBeerGameEnv, model, value_net, num_samples: int, horizon: int = 4):
        self.env, self.model, self.value_net = env, model, value_net
        self.num_samples, self.horizon = num_samples, horizon
        self.n, self.levels = env.graph.n_nodes, env.graph.order_levels
        self.calls = 0

    def act(self, state: np.ndarray, seed: int, period: int, episode: int = 0) -> np.ndarray:
        rng = np.random.default_rng(planner_seed(seed, period, 0, episode))
        K, H = self.num_samples, self.horizon
        plans = rng.integers(0, self.levels, size=(K, H, self.n))
        s = torch.tensor(np.repeat(state[None], K, 0), dtype=torch.float32)
        ret, disc = torch.zeros(K), 1.0
        with torch.no_grad():
            for h in range(H):
                s = self.model(s, torch.tensor(plans[:, h]))
                ret += disc * torch.tensor([self.env.period_reward(x) for x in s.numpy()])
                disc *= GAMMA
            if self.value_net is not None:
                ret += disc * self.value_net(s)
        self.calls += K * H
        return plans[int(torch.argmax(ret))][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("ppo",))
    ap.add_argument("--topology", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    run_ppo(args.topology, args.budget, args.seed, Path(args.out), device=args.device)


if __name__ == "__main__":
    main()
