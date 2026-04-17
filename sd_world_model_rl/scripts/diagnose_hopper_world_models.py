"""Diagnostic comparison for Hopper world models under a shared TD-MPC2 solver.

This script compares:
1. Original TD-MPC2 world model
2. SD Hopper world model integrated into TD-MPC2

The comparison is intentionally restricted to quantities that are meaningful
under the two different latent designs:
- one-step reward prediction error
- multi-step cumulative reward rollout bias
- TD-MPC2 latent consistency error
- SD structured-state prediction error
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TDMPC2_ROOT = REPO_ROOT / "external" / "tdmpc2" / "tdmpc2"
if str(TDMPC2_ROOT) not in sys.path:
    sys.path.append(str(TDMPC2_ROOT))

from common import math  # type: ignore  # noqa: E402
from common.parser import parse_cfg  # type: ignore  # noqa: E402
from envs import make_env  # type: ignore  # noqa: E402
from tdmpc2 import TDMPC2  # type: ignore  # noqa: E402


if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: "cpu"  # type: ignore[attr-defined]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg(world_model_kind: str, seed: int, with_env: bool = True) -> object:
    hydra.utils.get_original_cwd = lambda: str(REPO_ROOT)
    cfg_path = TDMPC2_ROOT / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    overrides = {
        "task": "hopper-hop",
        "obs": "state",
        "seed": seed,
        "world_model_kind": world_model_kind,
        "model_size": None if world_model_kind == "sd_hopper" else 5,
        "steps": 3000,
        "enable_wandb": False,
        "save_video": False,
        "save_agent": False,
        "save_csv": False,
        "compile": False,
        "eval_episodes": 5,
        "eval_freq": 1000,
        "num_samples": 64,
        "num_elites": 8,
        "num_pi_trajs": 16,
        "horizon": 3,
        "iterations": 3,
        "batch_size": 256,
        "mlp_dim": 512,
        "num_q": 5,
        "dropout": 0.01,
        "lr": 3e-4,
        "exp_name": f"diag_{world_model_kind}",
    }
    cfg = OmegaConf.merge(cfg, overrides)
    cfg = parse_cfg(cfg)
    if with_env:
        env = make_env(cfg)
        del env
    return cfg


def load_agent(world_model_kind: str, checkpoint: str, seed: int) -> TDMPC2:
    cfg = build_cfg(world_model_kind, seed)
    agent = TDMPC2(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    agent.model.load_state_dict(state_dict, strict=True)
    agent.model.eval()
    return agent


def collect_random_rollouts(seed: int, num_episodes: int, max_steps: int) -> List[Dict[str, np.ndarray]]:
    cfg = build_cfg("tdmpc2", seed, with_env=True)
    env = make_env(cfg)
    episodes: List[Dict[str, np.ndarray]] = []
    rng = np.random.default_rng(seed)
    for _ in range(num_episodes):
        obs = env.reset().float()
        obs_seq = [obs.numpy().copy()]
        act_seq: List[np.ndarray] = []
        rew_seq: List[float] = []
        for _step in range(max_steps):
            action = rng.uniform(low=-1.0, high=1.0, size=env.action_space.shape).astype(np.float32)
            next_obs, reward, done, _info = env.step(torch.from_numpy(action))
            act_seq.append(action.copy())
            rew_seq.append(float(reward))
            obs_seq.append(next_obs.numpy().copy())
            obs = next_obs.float()
            if done:
                break
        episodes.append(
            {
                "obs": np.asarray(obs_seq, dtype=np.float32),
                "actions": np.asarray(act_seq, dtype=np.float32),
                "rewards": np.asarray(rew_seq, dtype=np.float32),
            }
        )
    return episodes


def reward_pred(agent: TDMPC2, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    z = agent.model.encode(obs, task=None)
    pred = agent.model.reward(z, action, task=None)
    return math.two_hot_inv(pred, agent.cfg).squeeze(-1)


def tdmpc2_latent_error(agent: TDMPC2, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    z = agent.model.encode(obs, task=None)
    next_z_pred = agent.model.next(z, action, task=None)
    next_z_true = agent.model.encode(next_obs, task=None)
    return torch.sqrt(torch.mean((next_z_pred - next_z_true) ** 2, dim=-1))


def sd_struct_error(agent: TDMPC2, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    z = agent.model.encode(obs, task=None)
    next_z_pred = agent.model.next(z, action, task=None)
    next_z_true = agent.model.encode(next_obs, task=None)
    return torch.sqrt(torch.mean((next_z_pred - next_z_true) ** 2, dim=-1))


def rollout_reward_bias(agent: TDMPC2, obs0: torch.Tensor, actions: torch.Tensor, true_rewards: torch.Tensor) -> float:
    z = agent.model.encode(obs0.unsqueeze(0), task=None)
    pred_rewards = []
    for t in range(actions.shape[0]):
        a = actions[t:t+1]
        pred_r = math.two_hot_inv(agent.model.reward(z, a, task=None), agent.cfg).squeeze().item()
        pred_rewards.append(pred_r)
        z = agent.model.next(z, a, task=None)
    pred_return = float(np.sum(pred_rewards))
    true_return = float(true_rewards.sum().item())
    return abs(pred_return - true_return)


def rollout_state_or_latent_error(agent: TDMPC2, world_model_kind: str, obs_seq: torch.Tensor, actions: torch.Tensor) -> float:
    z = agent.model.encode(obs_seq[0:1], task=None)
    for t in range(actions.shape[0]):
        z = agent.model.next(z, actions[t:t+1], task=None)
    target = agent.model.encode(obs_seq[actions.shape[0]:actions.shape[0]+1], task=None)
    return float(torch.sqrt(torch.mean((z - target) ** 2)).item())


def summarize(arr: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(x.mean()) if x.size else float("nan"),
        "std": float(x.std()) if x.size else float("nan"),
        "n": int(x.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", required=True)
    parser.add_argument("--tdmpc2_ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_default_device(args.device)

    episodes = collect_random_rollouts(args.seed, args.episodes, args.max_steps)
    sd_agent = load_agent("sd_hopper", args.sd_ckpt, args.seed)
    td_agent = load_agent("tdmpc2", args.tdmpc2_ckpt, args.seed)

    reward_mae = {"sd_hopper": [], "tdmpc2": []}
    one_step_err = {"sd_hopper": [], "tdmpc2": []}
    rollout_reward = {
        "sd_hopper": {1: [], 3: [], 5: []},
        "tdmpc2": {1: [], 3: [], 5: []},
    }
    rollout_dyn = {
        "sd_hopper": {1: [], 3: [], 5: []},
        "tdmpc2": {1: [], 3: [], 5: []},
    }

    for ep in episodes:
        obs = torch.tensor(ep["obs"], dtype=torch.float32, device=args.device)
        actions = torch.tensor(ep["actions"], dtype=torch.float32, device=args.device)
        rewards = torch.tensor(ep["rewards"], dtype=torch.float32, device=args.device)
        if actions.numel() == 0:
            continue

        for t in range(actions.shape[0]):
            obs_t = obs[t:t+1]
            act_t = actions[t:t+1]
            next_obs_t = obs[t+1:t+2]
            true_r = rewards[t:t+1]

            sd_r = reward_pred(sd_agent, obs_t, act_t)
            td_r = reward_pred(td_agent, obs_t, act_t)
            reward_mae["sd_hopper"].append(abs(float(sd_r.item() - true_r.item())))
            reward_mae["tdmpc2"].append(abs(float(td_r.item() - true_r.item())))

            one_step_err["sd_hopper"].append(float(sd_struct_error(sd_agent, obs_t, act_t, next_obs_t).item()))
            one_step_err["tdmpc2"].append(float(tdmpc2_latent_error(td_agent, obs_t, act_t, next_obs_t).item()))

        for horizon in (1, 3, 5):
            if actions.shape[0] < horizon:
                continue
            for t in range(actions.shape[0] - horizon + 1):
                obs_t = obs[t]
                act_seq = actions[t:t+horizon]
                rew_seq = rewards[t:t+horizon]
                rollout_reward["sd_hopper"][horizon].append(
                    rollout_reward_bias(sd_agent, obs_t, act_seq, rew_seq)
                )
                rollout_reward["tdmpc2"][horizon].append(
                    rollout_reward_bias(td_agent, obs_t, act_seq, rew_seq)
                )
                rollout_dyn["sd_hopper"][horizon].append(
                    rollout_state_or_latent_error(sd_agent, "sd_hopper", obs[t:t+horizon+1], act_seq)
                )
                rollout_dyn["tdmpc2"][horizon].append(
                    rollout_state_or_latent_error(td_agent, "tdmpc2", obs[t:t+horizon+1], act_seq)
                )

    print("=== One-step reward MAE ===")
    for k in ("tdmpc2", "sd_hopper"):
        print(k, summarize(reward_mae[k]))
    print("\n=== One-step transition error ===")
    print("tdmpc2 (latent RMSE)", summarize(one_step_err["tdmpc2"]))
    print("sd_hopper (struct RMSE)", summarize(one_step_err["sd_hopper"]))
    print("\n=== Rollout reward bias ===")
    for horizon in (1, 3, 5):
        print(f"H={horizon}")
        for k in ("tdmpc2", "sd_hopper"):
            print(" ", k, summarize(rollout_reward[k][horizon]))
    print("\n=== Rollout transition error ===")
    for horizon in (1, 3, 5):
        print(f"H={horizon}")
        print(" ", "tdmpc2 latent", summarize(rollout_dyn["tdmpc2"][horizon]))
        print(" ", "sd_hopper struct", summarize(rollout_dyn["sd_hopper"][horizon]))


if __name__ == "__main__":
    main()
