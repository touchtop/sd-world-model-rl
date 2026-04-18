"""Phase 0: per-step state logging for v3b SD-hopper checkpoint.

Replays N episodes under a trained checkpoint and logs, for each step,
the raw DMControl physics data (geom positions, qpos, qvel), the 15-dim
SD-encoded structured state, the action, and the immediate reward.

The goal is to quantify the failure-basin (pelvis-contact / backward
sitting) vs success-basin (forward-leaning with alternating nose/foot
support) separation, so v4 structural quantities have empirically
grounded thresholds instead of guesses.

Output: one CSV per episode under results/phase0_state_log/.
"""
import os
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "glfw")

import csv
import sys
from pathlib import Path

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


CHECKPOINT = Path("/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_50k_v3b_hh005/models/final.pt")
OUT_DIR = REPO_ROOT / "results" / "phase0_state_log"
NUM_EPISODES = 10

GEOM_NAMES = ["torso", "nose", "pelvis", "thigh", "calf", "foot"]
QPOS_NAMES = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]
STRUCT_FIELD_NAMES = [
    "height", "vx", "vz", "pitch", "pitch_rate",
    "compression", "compression_rate",
    "swing", "swing_rate", "foot_angle", "foot_rate",
    "support", "stance_mix", "impact", "clearance",
    "rootz",
]


def build_cfg(struct_dim_override: int | None = None):
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
        "exp_name": "phase0_state_log",
        "checkpoint": str(CHECKPOINT),
        "sd_reward_mode": "curriculum",
        "sd_reward_curriculum_alpha": 0.5,
        "sd_reward_curriculum_alpha_min": 0.25,
        "sd_reward_curriculum_start": 10000,
        "sd_reward_curriculum_end": 30000,
        "sd_max_resid_scale": 5.0,
        "sd_height_loss_coef": 0.05,
        "eval_episodes": NUM_EPISODES,
    }
    if struct_dim_override is not None:
        overrides["sd_struct_dim"] = struct_dim_override
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


def physics_snapshot(physics):
    geom_xpos = physics.named.data.geom_xpos
    qpos = physics.named.data.qpos
    qvel = physics.named.data.qvel
    snap = {}
    for name in GEOM_NAMES:
        snap[f"{name}_x"] = float(geom_xpos[name, "x"])
        snap[f"{name}_z"] = float(geom_xpos[name, "z"])
    for name in QPOS_NAMES:
        snap[f"qpos_{name}"] = float(qpos[name])
        snap[f"qvel_{name}"] = float(qvel[name])
    return snap


def main():
    assert torch.cuda.is_available(), "CUDA is required."
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
    # Peek checkpoint to detect the struct_dim it was trained with. The v4
    # branch bumps DMControlHopperSDModel.STATE_DIM to 16 (adds rootz); older
    # v3b checkpoints were trained with STATE_DIM=15. The model class layout
    # is fixed per branch, so a mismatched checkpoint cannot be loaded here.
    # If you need to replay a pre-v4 checkpoint, run this script from the
    # master branch or add a legacy adapter.
    _peek = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    _peek_sd = _peek["model"] if isinstance(_peek, dict) and "model" in _peek else _peek
    _ckpt_struct_dim = None
    _passive_w = _peek_sd.get("_dynamics.passive_flow.0.weight")
    if _passive_w is not None:
        _ckpt_struct_dim = int(_passive_w.shape[1])
    del _peek, _peek_sd
    from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel as _Cls
    if _ckpt_struct_dim is not None and _ckpt_struct_dim != _Cls.STATE_DIM:
        raise RuntimeError(
            f"Checkpoint struct_dim={_ckpt_struct_dim} does not match current "
            f"DMControlHopperSDModel.STATE_DIM={_Cls.STATE_DIM}. "
            "Replay from a branch whose class matches the checkpoint (e.g. master "
            "for v3b/fixrerun checkpoints)."
        )
    cfg, env = build_cfg(struct_dim_override=_ckpt_struct_dim)
    set_seed(cfg.seed)

    agent = TDMPC2(cfg)
    payload = torch.load(cfg.checkpoint, map_location="cuda:0", weights_only=False)
    model_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    agent.model.load_state_dict(model_state, strict=True)
    agent.model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_fields = ["episode", "step"] + [f"{g}_x" for g in GEOM_NAMES] + [f"{g}_z" for g in GEOM_NAMES]
    base_fields += [f"qpos_{n}" for n in QPOS_NAMES] + [f"qvel_{n}" for n in QPOS_NAMES]
    base_fields += [f"struct_{n}" for n in STRUCT_FIELD_NAMES]
    base_fields += ["action_0", "action_1", "action_2", "action_3", "reward", "cumulative_reward", "done"]

    summary_rows = []

    for ep_idx in range(NUM_EPISODES):
        csv_path = OUT_DIR / f"episode_{ep_idx:02d}.csv"
        rows = []
        obs = env.reset()
        done = False
        t = 0
        cum_reward = 0.0

        while not done:
            physics = env.unwrapped.physics
            snap = physics_snapshot(physics)

            obs_cuda = obs.to("cuda:0", non_blocking=True).unsqueeze(0)
            with torch.no_grad():
                struct = agent.model._dynamics.encode_observation(obs_cuda)[0].cpu().numpy()
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            action_np = action.detach().cpu().numpy().astype(np.float32).reshape(-1)

            obs_next, reward, done_flag, info = env.step(action)
            cum_reward += float(reward.item())

            row = {"episode": ep_idx, "step": t}
            row.update(snap)
            for name, value in zip(STRUCT_FIELD_NAMES, struct):
                row[f"struct_{name}"] = float(value)
            for i in range(4):
                row[f"action_{i}"] = float(action_np[i]) if i < action_np.size else 0.0
            row["reward"] = float(reward.item())
            row["cumulative_reward"] = cum_reward
            row["done"] = int(bool(done_flag))
            rows.append(row)

            obs = obs_next
            done = bool(done_flag)
            t += 1

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=base_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        summary_rows.append({
            "episode": ep_idx,
            "length": t,
            "cumulative_reward": cum_reward,
            "final_pelvis_z": rows[-1]["pelvis_z"],
            "final_nose_z": rows[-1]["nose_z"],
            "final_torso_z": rows[-1]["torso_z"],
            "final_pitch": rows[-1]["qpos_rooty"],
            "min_pelvis_z": min(r["pelvis_z"] for r in rows),
            "min_nose_z": min(r["nose_z"] for r in rows),
            "max_rootx": max(r["qpos_rootx"] for r in rows),
            "pelvis_low_frac": sum(1 for r in rows if r["pelvis_z"] < 0.4) / len(rows),
            "nose_low_frac": sum(1 for r in rows if r["nose_z"] < 0.4) / len(rows),
            "backward_pitch_frac": sum(1 for r in rows if r["qpos_rooty"] < -0.3) / len(rows),
            "forward_pitch_frac": sum(1 for r in rows if r["qpos_rooty"] > 0.3) / len(rows),
        })
        print(f"episode={ep_idx:02d} len={t} reward={cum_reward:.3f} "
              f"final(pelvis={rows[-1]['pelvis_z']:.2f}, nose={rows[-1]['nose_z']:.2f}, "
              f"pitch={rows[-1]['qpos_rooty']:+.2f}, rootx={rows[-1]['qpos_rootx']:+.2f})")

    summary_path = OUT_DIR / "summary.csv"
    summary_fields = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"\nSummary written: {summary_path}")


if __name__ == "__main__":
    main()
