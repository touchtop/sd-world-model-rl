#!/usr/bin/env python3
"""Prototype training script for DMControl Walker-Walk structured model.

This script is intentionally limited to the first construction stage:
random-data collection + one-step model fitting. It does not yet include
planning or a terminal value network. The purpose is to validate whether the
state skeleton and update decomposition are learnable on real trajectories.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from dm_control import suite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.world_model.dmcontrol_walker_sd_model import DMControlWalkerSDModel


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_trajectories(env, n_steps: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    model_cls = DMControlWalkerSDModel
    data = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
    }

    ts = env.reset()
    obs = ts.observation
    for _ in range(n_steps):
        state = model_cls.obs_to_state(obs)
        action = rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)
        ts = env.step(action)
        next_obs = ts.observation
        next_state = model_cls.obs_to_state(next_obs)

        data["states"].append(state)
        data["actions"].append(action)
        data["rewards"].append(float(ts.reward))
        data["next_states"].append(next_state)
        data["dones"].append(bool(ts.last()))

        obs = next_obs
        if ts.last():
            ts = env.reset()
            obs = ts.observation

    for k in data:
        data[k] = np.asarray(data[k], dtype=np.float32)
    return data


def _state_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_pos = pred[:, :8]
    pred_vel = pred[:, 8:]
    tgt_pos = target[:, :8]
    tgt_vel = target[:, 8:]

    pos_loss = F.smooth_l1_loss(pred_pos[:, :1], tgt_pos[:, :1])

    pred_ang = pred_pos[:, 1:]
    tgt_ang = tgt_pos[:, 1:]
    ang_err = torch.atan2(torch.sin(pred_ang - tgt_ang), torch.cos(pred_ang - tgt_ang))
    ang_loss = F.smooth_l1_loss(ang_err, torch.zeros_like(ang_err))

    vel_loss = F.smooth_l1_loss(pred_vel, tgt_vel)
    return pos_loss + ang_loss + vel_loss


def rollout_return(model, init_state, action_seqs, gamma: float = 0.99):
    """Evaluate batched action sequences in the learned world model."""
    if init_state.dim() == 1:
        init_state = init_state.unsqueeze(0)
    pop, horizon, _ = action_seqs.shape
    state = init_state.expand(pop, -1)
    returns = torch.zeros(pop, dtype=state.dtype, device=state.device)
    discount = torch.ones(pop, dtype=state.dtype, device=state.device)
    for t in range(horizon):
        action = action_seqs[:, t]
        next_state = model(state, action)
        reward = model.predict_reward(state, action, next_state)
        returns = returns + discount * reward
        discount = discount * gamma
        state = next_state
    return returns


def plan_action_cem(
    model,
    state,
    horizon: int = 12,
    population: int = 256,
    elites: int = 32,
    iterations: int = 4,
    gamma: float = 0.99,
):
    """Simple CEM planner for 6D continuous actions."""
    device = state.device
    action_dim = 6
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    with torch.no_grad():
        for _ in range(iterations):
            noise = torch.randn(population, horizon, action_dim, device=device)
            action_seqs = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, -1.0, 1.0)
            returns = rollout_return(model, state, action_seqs, gamma=gamma)
            elite_idx = torch.topk(returns, k=elites, dim=0).indices
            elite = action_seqs[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(0.05)
    return mean[0].detach()


def _state_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pos_err = pred[:, :1] - target[:, :1]
    ang_err = torch.atan2(
        torch.sin(pred[:, 1:8] - target[:, 1:8]),
        torch.cos(pred[:, 1:8] - target[:, 1:8]),
    )
    vel_err = pred[:, 8:] - target[:, 8:]
    err = torch.cat([pos_err, ang_err, vel_err], dim=-1)
    return float(torch.sqrt((err ** 2).mean()).item())


def train_sd_model(model, data, epochs: int = 200, batch_size: int = 256, lr: float = 1e-3,
                   device: str = "cpu"):
    n = len(data["states"])
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(0.1 * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    model.to(device)
    states = torch.tensor(data["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        perm = np.random.permutation(train_idx)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            batch = perm[start:start + batch_size]
            s = states[batch]
            a = actions[batch]
            ns = next_states[batch]
            r = rewards[batch]

            pred_ns = model(s, a)
            pred_r = model.predict_reward(s, a, pred_ns)

            loss = _state_prediction_loss(pred_ns, ns) + 0.2 * F.smooth_l1_loss(pred_r, r)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            train_loss_sum += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vs = states[val_idx]
            va = actions[val_idx]
            vns = next_states[val_idx]
            vr = rewards[val_idx]
            pred_vns = model(vs, va)
            pred_vr = model.predict_reward(vs, va, pred_vns)
            val_loss = _state_prediction_loss(pred_vns, vns) + 0.2 * F.smooth_l1_loss(pred_vr, vr)

        best_val = min(best_val, float(val_loss.item()))
        if float(val_loss.item()) <= best_val:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            log(
                f"epoch {epoch + 1:4d} | "
                f"train={train_loss_sum / max(n_batches, 1):.4f} | "
                f"val={float(val_loss.item()):.4f} | "
                f"best={best_val:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return best_val


def evaluate_model(model, data, rollout_horizons=(1, 5, 10), device: str = "cpu"):
    model.to(device)
    states = torch.tensor(data["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=device)
    dones = np.asarray(data["dones"], dtype=bool)

    with torch.no_grad():
        pred_next = model(states, actions)
    metrics = {
        "one_step_rmse": _state_rmse(pred_next, next_states),
    }

    valid_starts = []
    max_h = max(rollout_horizons)
    for start in range(len(states) - max_h):
        if dones[start:start + max_h].any():
            continue
        valid_starts.append(start)

    if not valid_starts:
        return metrics

    starts = valid_starts[: min(128, len(valid_starts))]
    for horizon in rollout_horizons:
        errs = []
        for start in starts:
            pred = states[start:start + 1]
            for t in range(horizon):
                pred = model(pred, actions[start + t:start + t + 1])
            target = next_states[start + horizon - 1:start + horizon]
            errs.append(_state_rmse(pred, target))
        metrics[f"rollout_{horizon}_rmse"] = float(np.mean(errs))
    return metrics


def evaluate_agent(
    env,
    model,
    n_episodes: int = 3,
    max_steps: int = 1000,
    planner_horizon: int = 12,
    planner_population: int = 256,
    planner_elites: int = 32,
    planner_iterations: int = 4,
    gamma: float = 0.99,
    device: str = "cpu",
):
    returns = []
    model.eval()
    model.to(device)
    for ep in range(n_episodes):
        ts = env.reset()
        ep_ret = 0.0
        for _ in range(max_steps):
            state_np = DMControlWalkerSDModel.obs_to_state(ts.observation)
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
            action = plan_action_cem(
                model,
                state,
                horizon=planner_horizon,
                population=planner_population,
                elites=planner_elites,
                iterations=planner_iterations,
                gamma=gamma,
            )
            ts = env.step(action.cpu().numpy())
            ep_ret += float(ts.reward or 0.0)
            if ts.last():
                break
        returns.append(ep_ret)
        log(f"eval episode {ep}: return={ep_ret:.1f}")
    return np.asarray(returns, dtype=np.float32)


def evaluate_random(env, n_episodes: int = 3, max_steps: int = 1000, seed: int = 0):
    rng = np.random.RandomState(seed)
    returns = []
    for ep in range(n_episodes):
        ts = env.reset()
        ep_ret = 0.0
        for _ in range(max_steps):
            action = rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)
            ts = env.step(action)
            ep_ret += float(ts.reward or 0.0)
            if ts.last():
                break
        returns.append(ep_ret)
    return np.asarray(returns, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Prototype Walker-Walk SD model trainer")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--planner_horizon", type=int, default=12)
    parser.add_argument("--planner_population", type=int, default=256)
    parser.add_argument("--planner_elites", type=int, default=32)
    parser.add_argument("--planner_iterations", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = suite.load("walker", "walk", task_kwargs={"random": args.seed})
    log(f"Collecting {args.budget} steps of random Walker-Walk data")
    data = collect_trajectories(env, args.budget, seed=args.seed)

    model = DMControlWalkerSDModel()
    log("Training structured Walker model")
    best_val = train_sd_model(model, data, epochs=args.epochs, device=args.device)
    metrics = evaluate_model(model, data, device=args.device)
    eval_env = suite.load("walker", "walk", task_kwargs={"random": args.seed + 1000})
    random_returns = evaluate_random(eval_env, n_episodes=args.eval_episodes, seed=args.seed + 2000)
    learned_returns = evaluate_agent(
        eval_env,
        model,
        n_episodes=args.eval_episodes,
        planner_horizon=args.planner_horizon,
        planner_population=args.planner_population,
        planner_elites=args.planner_elites,
        planner_iterations=args.planner_iterations,
        device=args.device,
    )
    log(f"Best validation loss: {best_val:.4f}")
    for key, value in metrics.items():
        log(f"{key}: {value:.4f}")
    log(
        f"random return: mean={float(random_returns.mean()):.1f} "
        f"std={float(random_returns.std()):.1f}"
    )
    log(
        f"learned-MPC return: mean={float(learned_returns.mean()):.1f} "
        f"std={float(learned_returns.std()):.1f}"
    )
    log(f"Learned params: {model.get_learned_params()}")


if __name__ == "__main__":
    main()
