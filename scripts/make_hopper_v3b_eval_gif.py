import os
import sys
from pathlib import Path

# `egl` 在当前机器的提权环境里初始化失败；这里强制回到 `glfw`，
# 与此前本项目在无显示 warning 下仍可正常创建 Hopper 任务的口径保持一致。
os.environ["MUJOCO_GL"] = "glfw"

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
TDMPC2_ROOT = REPO_ROOT / "external" / "tdmpc2" / "tdmpc2"
if str(TDMPC2_ROOT) not in sys.path:
    sys.path.insert(0, str(TDMPC2_ROOT))

import hydra.utils  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs.dmcontrol import make_env as make_dmcontrol_env  # noqa: E402
from envs.wrappers.tensor import TensorWrapper  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402


def build_cfg():
    hydra.utils.get_original_cwd = lambda: str(REPO_ROOT)
    cfg = OmegaConf.load(TDMPC2_ROOT / "config.yaml")
    overrides = {
        "task": "hopper-hop",
        "obs": "state",
        "seed": 3,
        "world_model_kind": "sd_hopper",
        "model_size": None,
        "mlp_dim": 256,
        "num_q": 3,
        "compile": False,
        "save_video": False,
        "enable_wandb": False,
        "exp_name": "sd_hopper_50k_v3b_hh005_reval_gif",
        "checkpoint": str(
            REPO_ROOT
            / "logs"
            / "hopper-hop"
            / "3"
            / "sd_hopper_50k_v3b_hh005"
            / "models"
            / "final.pt"
        ),
        "sd_reward_mode": "curriculum",
        "sd_reward_curriculum_alpha": 0.5,
        "sd_reward_curriculum_alpha_min": 0.25,
        "sd_reward_curriculum_start": 10000,
        "sd_reward_curriculum_end": 30000,
        "sd_max_resid_scale": 5.0,
        "sd_height_loss_coef": 0.05,
        "eval_episodes": 10,
    }
    for key, value in overrides.items():
        cfg[key] = value
    cfg = parse_cfg(cfg)

    env = TensorWrapper(make_dmcontrol_env(cfg))
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return cfg, env


def save_gif(episodes, out_path, top_k=3, fps=15, hold_frames=8):
    chosen = sorted(episodes, key=lambda x: x["reward"], reverse=True)[:top_k]
    stitched = []
    for idx, ep in enumerate(chosen):
        stitched.extend(ep["frames"])
        if idx != len(chosen) - 1:
            stitched.extend([ep["frames"][-1]] * hold_frames)
    imageio.mimsave(out_path, stitched, fps=fps)
    return chosen


def main():
    assert torch.cuda.is_available(), "CUDA is required for replaying this checkpoint."
    cfg, env = build_cfg()
    set_seed(cfg.seed)

    agent = TDMPC2(cfg)
    payload = torch.load(cfg.checkpoint, map_location="cuda:0")
    model_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    agent.model.load_state_dict(model_state, strict=True)
    agent.model.eval()

    out_dir = REPO_ROOT / "results" / "hopper_v3b_eval_gif"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    summary = []
    for ep_idx in range(cfg.eval_episodes):
        obs = env.reset()
        done = False
        t = 0
        ep_reward = 0.0
        frames = [env.render()]
        while not done:
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward.item())
            frames.append(env.render())
            t += 1
        mp4_path = out_dir / f"episode_{ep_idx:02d}_R_{ep_reward:.2f}.mp4"
        imageio.mimsave(mp4_path, frames, fps=15)
        episodes.append({"index": ep_idx, "reward": ep_reward, "frames": frames})
        summary.append((ep_idx, ep_reward, t))
        print(f"episode={ep_idx:02d} reward={ep_reward:.6f} length={t}")

    gif_path = out_dir / "hopper_v3b_top3_eval.gif"
    chosen = save_gif(episodes, gif_path)
    print(f"gif={gif_path}")
    print("top_episodes=" + ",".join(f"{ep['index']}:{ep['reward']:.6f}" for ep in chosen))

    summary_path = out_dir / "episode_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for ep_idx, reward, length in summary:
            f.write(f"episode={ep_idx:02d} reward={reward:.6f} length={length}\n")
        f.write(f"gif={gif_path}\n")
        f.write("top_episodes=" + ",".join(f"{ep['index']}:{ep['reward']:.6f}" for ep in chosen) + "\n")


if __name__ == "__main__":
    main()
