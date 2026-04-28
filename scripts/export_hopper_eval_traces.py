import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

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
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--task", default="hopper-hop")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--world_model_kind", default="sd_hopper")
    p.add_argument("--model_size", type=int, default=None)
    p.add_argument("--mlp_dim", type=int, default=256)
    p.add_argument("--num_q", type=int, default=3)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--num_elites", type=int, default=16)
    p.add_argument("--num_pi_trajs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--sd_struct_dim", type=int, default=DMControlHopperSDModel.STATE_DIM)
    p.add_argument("--sd_latent_dim", type=int, default=512)
    p.add_argument("--sd_reward_mode", default="curriculum")
    p.add_argument("--sd_reward_curriculum_alpha", type=float, default=0.5)
    p.add_argument("--sd_reward_curriculum_alpha_min", type=float, default=0.25)
    p.add_argument("--sd_reward_curriculum_start", type=int, default=10000)
    p.add_argument("--sd_reward_curriculum_end", type=int, default=30000)
    p.add_argument("--sd_max_resid_scale", type=float, default=5.0)
    p.add_argument("--sd_stage_mode", default="full")
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_cfg(args):
    hydra.utils.get_original_cwd = lambda: str(REPO_ROOT)
    cfg = OmegaConf.load(TDMPC2_ROOT / "config.yaml")
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
        "exp_name": f"{args.checkpoint.stem}_trace_eval",
        "dropout": args.dropout,
        "sd_struct_dim": args.sd_struct_dim,
        "sd_latent_dim": args.sd_latent_dim,
        "sd_reward_mode": args.sd_reward_mode,
        "sd_reward_curriculum_alpha": args.sd_reward_curriculum_alpha,
        "sd_reward_curriculum_alpha_min": args.sd_reward_curriculum_alpha_min,
        "sd_reward_curriculum_start": args.sd_reward_curriculum_start,
        "sd_reward_curriculum_end": args.sd_reward_curriculum_end,
        "sd_max_resid_scale": args.sd_max_resid_scale,
        "sd_stage_mode": args.sd_stage_mode,
    }
    for k, v in overrides.items():
        cfg[k] = v
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


def load_agent(cfg, device):
    agent = TDMPC2(cfg)
    payload = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    agent.model.load_state_dict(model_state, strict=True)
    agent.model.eval()
    return agent


def flatten_obs(obs):
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy().reshape(-1)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def physics_snapshot(physics):
    geom_names = ["torso", "nose", "pelvis", "thigh", "calf", "foot"]
    qpos_names = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]
    qvel_names = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]
    geom_xpos = physics.named.data.geom_xpos
    qpos = physics.named.data.qpos
    qvel = physics.named.data.qvel
    snap = {}
    for name in geom_names:
        snap[f"{name}_x"] = float(geom_xpos[name, "x"])
        snap[f"{name}_z"] = float(geom_xpos[name, "z"])
    for name in qpos_names:
        snap[f"qpos_{name}"] = float(qpos[name])
    for name in qvel_names:
        snap[f"qvel_{name}"] = float(qvel[name])
    return snap


def hopper_probe_features(probe, obs_np):
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        state = probe.encode_observation(obs_t)
        feat = probe._state_reward_features(state)
        height = float(probe.predict_height(state)[0].item())
    return {
        "torso_z": float(feat["torso_z"][0].item()),
        "rootz": float(feat["rootz"][0].item()),
        "vx": float(feat["vx"][0].item()),
        "vz": float(feat["vz"][0].item()),
        "pitch": float(feat["pitch"][0].item()),
        "pitch_rate": float(feat["pitch_rate"][0].item()),
        "touch_toe": float(feat["touch_toe"][0].item()),
        "touch_heel": float(feat["touch_heel"][0].item()),
        "z_min": float(feat["z_min"][0].item()),
        "any_contact_z": float(feat["any_contact_z"][0].item()),
        "qpos_knee": float(feat["qpos_knee"][0].item()),
        "height": height,
        "posture_stability": float(feat["posture_stability"][0].item()),
        "spin_stability": float(feat["spin_stability"][0].item()),
        "forward_progress": float(feat["forward_progress"][0].item()),
        "upward_progress": float(feat["upward_progress"][0].item()),
        "descent_progress": float(feat["descent_progress"][0].item()),
        "backward_speed": float(feat["backward_speed"][0].item()),
        "full_foot_support": float(feat["full_foot_support"][0].item()),
        "support_contact_gate": float(feat["support_contact_gate"][0].item()),
        "full_foot_support_gate": float(feat["full_foot_support_gate"][0].item()),
        "low_body": float(feat["low_body"][0].item()),
        "knee_window": float(feat["knee_window"][0].item()),
        "vertical_settle": float(feat["vertical_settle"][0].item()),
        "falling_air": float(feat["falling_air"][0].item()),
        "grounded_rear": float(feat["grounded_rear"][0].item()),
        "grounded_prone": float(feat["grounded_prone"][0].item()),
        "grounded_upright": float(feat["grounded_upright"][0].item()),
        "grounded_prepared": float(feat["grounded_prepared"][0].item()),
        "rising_air": float(feat["rising_air"][0].item()),
        "rotation_penalty": float(feat["rotation_penalty"][0].item()),
    }


def mean_of(rows, key):
    return float(sum(float(r[key]) for r in rows) / max(len(rows), 1))


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required for Hopper trace export."
    device = torch.device(args.device)

    cfg, env = build_cfg(args)
    set_seed(cfg.seed)
    agent = load_agent(cfg, device)
    probe = DMControlHopperSDModel(stage_mode=args.sd_stage_mode).cpu()
    probe.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ep_idx in range(args.episodes):
        obs = env.reset()
        done = False
        t = 0
        ep_reward = 0.0
        rows = []

        while not done:
            obs_np = flatten_obs(obs)
            snap = physics_snapshot(env.unwrapped.physics)
            features = hopper_probe_features(probe, obs_np)

            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            act_np = flatten_obs(action)
            next_obs, reward, done, _info = env.step(action)
            reward_val = float(reward.item())
            ep_reward += reward_val

            row = {
                "step": t,
                "reward": reward_val,
                "done": int(done),
            }
            row.update(snap)
            row.update(features)
            row["rootx"] = snap["qpos_rootx"]
            row["falling_air_gt_05"] = 1.0 if features["falling_air"] >= 0.5 else 0.0
            row["grounded_rear_gt_05"] = 1.0 if features["grounded_rear"] >= 0.5 else 0.0
            row["grounded_prone_gt_05"] = 1.0 if features["grounded_prone"] >= 0.5 else 0.0
            row["grounded_upright_gt_05"] = 1.0 if features["grounded_upright"] >= 0.5 else 0.0
            row["grounded_prepared_gt_05"] = 1.0 if features["grounded_prepared"] >= 0.5 else 0.0
            row["rising_air_gt_05"] = 1.0 if features["rising_air"] >= 0.5 else 0.0
            row["full_foot_support_gt_05"] = 1.0 if features["full_foot_support"] >= 0.5 else 0.0
            row["backward_speed_gt_05"] = 1.0 if features["backward_speed"] >= 0.5 else 0.0
            row["rotation_penalty_gt_05"] = 1.0 if features["rotation_penalty"] >= 0.5 else 0.0
            for j in range(act_np.size):
                row[f"action_{j}"] = float(act_np[j])
            rows.append(row)

            obs = next_obs
            t += 1

        csv_path = args.out_dir / f"episode_{ep_idx:02d}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        initial = rows[0]
        final = rows[-1]
        summary_rows.append({
            "episode_idx": ep_idx,
            "episode_reward": ep_reward,
            "length": t,
            "initial_pitch": initial["pitch"],
            "initial_torso_z": initial["torso_z"],
            "initial_rootz": initial["rootz"],
            "initial_qpos_rootz": initial["qpos_rootz"],
            "final_pitch": final["pitch"],
            "final_torso_z": final["torso_z"],
            "final_rootz": final["rootz"],
            "final_qpos_rootz": final["qpos_rootz"],
            "final_rootx": final["qpos_rootx"],
            "max_rootx": max(float(r["qpos_rootx"]) for r in rows),
            "max_torso_z": max(float(r["torso_z"]) for r in rows),
            "max_height": max(float(r["height"]) for r in rows),
            "max_vx": max(float(r["vx"]) for r in rows),
            "max_vz": max(float(r["vz"]) for r in rows),
            "max_full_foot_support": max(float(r["full_foot_support"]) for r in rows),
            "falling_air_frac": mean_of(rows, "falling_air_gt_05"),
            "grounded_rear_frac": mean_of(rows, "grounded_rear_gt_05"),
            "grounded_prone_frac": mean_of(rows, "grounded_prone_gt_05"),
            "grounded_upright_frac": mean_of(rows, "grounded_upright_gt_05"),
            "grounded_prepared_frac": mean_of(rows, "grounded_prepared_gt_05"),
            "rising_air_frac": mean_of(rows, "rising_air_gt_05"),
            "full_foot_support_frac": mean_of(rows, "full_foot_support_gt_05"),
            "backward_speed_frac": mean_of(rows, "backward_speed_gt_05"),
            "rotation_penalty_frac": mean_of(rows, "rotation_penalty_gt_05"),
            "trace_csv": str(csv_path),
        })
        print(f"episode={ep_idx:02d} reward={ep_reward:.6f} len={t} csv={csv_path}")

    summary_path = args.out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
