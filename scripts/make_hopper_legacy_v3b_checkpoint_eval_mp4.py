import argparse
import os
import sys
from pathlib import Path

# 恢复版 v3b checkpoint 渲染固定走 glfw + xvfb，与既有 Hopper 回放口径一致。
os.environ["MUJOCO_GL"] = "glfw"

import imageio
import torch
from omegaconf import OmegaConf

MAIN_REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = MAIN_REPO_ROOT / "legacy_v3b_restore"
LEGACY_TDMPC2_ROOT = LEGACY_ROOT / "external" / "tdmpc2" / "tdmpc2"
MAIN_TDMPC2_ROOT = MAIN_REPO_ROOT / "external" / "tdmpc2" / "tdmpc2"

# 先放 legacy overlay，再放主线 fallback，避免误走当前 v4 的调用链。
for path in [LEGACY_TDMPC2_ROOT, MAIN_TDMPC2_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import hydra.utils  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs.dmcontrol import make_env as make_dmcontrol_env  # noqa: E402
from envs.wrappers.tensor import TensorWrapper  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--task", default="hopper-hop")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--world_model_kind", default="sd_hopper")
    parser.add_argument("--model_size", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--num_q", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--num_elites", type=int, default=64)
    parser.add_argument("--num_pi_trajs", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--sd_reward_mode", default="curriculum")
    parser.add_argument("--sd_reward_curriculum_alpha", type=float, default=0.5)
    parser.add_argument("--sd_reward_curriculum_alpha_min", type=float, default=0.1)
    parser.add_argument("--sd_reward_curriculum_start", type=int, default=10000)
    parser.add_argument("--sd_reward_curriculum_end", type=int, default=30000)
    parser.add_argument("--sd_max_resid_scale", type=float, default=5.0)
    parser.add_argument("--sd_height_loss_coef", type=float, default=0.05)
    return parser.parse_args()


def build_cfg(args):
    hydra.utils.get_original_cwd = lambda: str(LEGACY_ROOT)
    cfg = OmegaConf.load(LEGACY_TDMPC2_ROOT / "config.yaml")
    overrides = {
        "task": args.task,
        "obs": "state",
        "seed": args.seed,
        "world_model_kind": args.world_model_kind,
        "model_size": args.model_size,
        "mlp_dim": args.mlp_dim,
        "num_q": args.num_q,
        "num_samples": args.num_samples,
        "num_elites": args.num_elites,
        "num_pi_trajs": args.num_pi_trajs,
        "horizon": args.horizon,
        "iterations": args.iterations,
        "compile": False,
        "save_video": False,
        "enable_wandb": False,
        "eval_episodes": args.episodes,
        "checkpoint": str(args.checkpoint),
        "exp_name": f"{args.checkpoint.stem}_legacy_v3b_replay_eval",
        "sd_reward_mode": args.sd_reward_mode,
        "sd_reward_curriculum_alpha": args.sd_reward_curriculum_alpha,
        "sd_reward_curriculum_alpha_min": args.sd_reward_curriculum_alpha_min,
        "sd_reward_curriculum_start": args.sd_reward_curriculum_start,
        "sd_reward_curriculum_end": args.sd_reward_curriculum_end,
        "sd_max_resid_scale": args.sd_max_resid_scale,
        "sd_height_loss_coef": args.sd_height_loss_coef,
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


def load_agent(cfg):
    agent = TDMPC2(cfg)
    payload = torch.load(cfg.checkpoint, map_location="cuda:0", weights_only=False)
    model_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    agent.model.load_state_dict(model_state, strict=True)
    agent.model.eval()
    return agent


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required for replaying Hopper checkpoints."

    cfg, env = build_cfg(args)
    set_seed(cfg.seed)
    agent = load_agent(cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep_idx in range(cfg.eval_episodes):
        obs = env.reset()
        done = False
        t = 0
        ep_reward = 0.0
        frames = [env.render()]
        while not done:
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            obs, reward, done, _info = env.step(action)
            ep_reward += float(reward.item())
            frames.append(env.render())
            t += 1
        mp4_path = args.out_dir / f"episode_{ep_idx:02d}_R_{ep_reward:.2f}.mp4"
        imageio.mimsave(mp4_path, frames, fps=args.fps)
        rows.append((ep_idx, ep_reward, t, mp4_path))
        print(f"episode={ep_idx:02d} reward={ep_reward:.6f} length={t} mp4={mp4_path}")

    summary_path = args.out_dir / "episode_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint={args.checkpoint}\n")
        f.write(
            "config="
            f"task={args.task},seed={args.seed},world_model_kind={args.world_model_kind},"
            f"mlp_dim={args.mlp_dim},num_q={args.num_q},"
            f"sd_reward_mode={args.sd_reward_mode},"
            f"alpha={args.sd_reward_curriculum_alpha},"
            f"alpha_min={args.sd_reward_curriculum_alpha_min},"
            f"alpha_start={args.sd_reward_curriculum_start},"
            f"alpha_end={args.sd_reward_curriculum_end}\n"
        )
        for ep_idx, reward, length, mp4_path in rows:
            f.write(f"episode={ep_idx:02d} reward={reward:.6f} length={length} mp4={mp4_path}\n")


if __name__ == "__main__":
    main()
