import os
import sys
import argparse
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"

import imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dm_control import suite  # noqa: E402
from scripts.run_dmcontrol_cartpole import (  # noqa: E402
    DMControlCartPoleSDModel,
    DMControlCartPoleWrapper,
    SwingupValueNet,
    flatten_obs,
    get_discrete_actions,
    mcts_search_swingup,
    obs_to_state,
)


CKPT_PATH = REPO_ROOT / "results" / "si_robust_ckpts" / "T4_wd0p0001_sn0p001_seed42.pt"
EVAL_PATH = REPO_ROOT / "results" / "si_robust_eval" / "T4_wd0p0001_sn0p001_seed42_eval.pt"
OUT_DIR = REPO_ROOT / "results" / "cartpole_best_eval_mp4"

NUM_SIMS = 200
ACTION_REPEAT = 2
EVAL_EPISODES = 5
MAX_STEPS = 1000
RENDER_W = 640
RENDER_H = 480
FPS = 20


def load_artifacts():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    eval_state = torch.load(EVAL_PATH, map_location="cpu", weights_only=False)

    model = DMControlCartPoleSDModel(learn_params=True, prior_noise=0.3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vnet = SwingupValueNet(hidden=128)
    vnet.load_state_dict(ckpt["vnet"])
    vnet.eval()

    actions = get_discrete_actions(11)
    target_returns = np.array(eval_state["returns"], dtype=np.float32)
    return model, vnet, actions, target_returns


def render_episode(env, model, vnet, actions):
    ts = env.reset()
    obs = flatten_obs(ts.observation)
    state = obs_to_state(obs)

    frames = [env.physics.render(height=RENDER_H, width=RENDER_W, camera_id=0)]
    ep_ret = 0.0
    env_step = 0

    learned_g_over_l = model.g.item() / max(model.l.item(), 1e-3)
    wrapper = DMControlCartPoleWrapper(model)

    while env_step < MAX_STEPS:
        action_val = mcts_search_swingup(
            state,
            wrapper,
            actions,
            num_simulations=NUM_SIMS,
            value_net=vnet,
            action_repeat=ACTION_REPEAT,
            g_over_l=learned_g_over_l,
        )
        for _ in range(ACTION_REPEAT):
            ts = env.step(np.array([action_val], dtype=np.float32))
            ep_ret += float(ts.reward if ts.reward is not None else 0.0)
            env_step += 1
            frames.append(env.physics.render(height=RENDER_H, width=RENDER_W, camera_id=0))
            if ts.last() or env_step >= MAX_STEPS:
                break
        if ts.last():
            break
        obs = flatten_obs(ts.observation)
        state = obs_to_state(obs)

    return ep_ret, frames, env_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--start_episode", type=int, default=0)
    args = parser.parse_args()

    model, vnet, actions, target_returns = load_artifacts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = suite.load("cartpole", "swingup", task_kwargs={"random": 42 + 1000})
    rows = []
    total_episodes = args.start_episode + args.episodes
    for ep_idx in range(total_episodes):
        ep_ret, frames, steps = render_episode(env, model, vnet, actions)
        if ep_idx < args.start_episode:
            continue
        out_path = OUT_DIR / f"episode_{ep_idx:02d}_R_{ep_ret:.2f}.mp4"
        imageio.mimsave(out_path, frames, fps=FPS)
        target = float(target_returns[ep_idx]) if ep_idx < len(target_returns) else float("nan")
        rows.append((ep_idx, ep_ret, target, steps, out_path))
        print(f"episode={ep_idx:02d} reward={ep_ret:.6f} target_cache={target:.6f} steps={steps} mp4={out_path}")

    with open(OUT_DIR / "episode_summary.txt", "w", encoding="utf-8") as f:
        for ep_idx, ep_ret, target, steps, out_path in rows:
            f.write(
                f"episode={ep_idx:02d} reward={ep_ret:.6f} "
                f"target_cache={target:.6f} steps={steps} mp4={out_path}\n"
            )


if __name__ == "__main__":
    main()
