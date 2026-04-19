"""Closed-loop diagnosis driver for v4 Hopper experiments.

This script operationalizes the current research workflow:

1. Replay a checkpoint under evaluation policy.
2. Log per-step physics and structured state traces.
3. Classify zero / low-score episodes by dominant failure basin.
4. Render only the most informative episodes to MP4.
5. Emit a compact diagnosis report and the next hypothesis to test.

It is intentionally narrow: it targets the current v4 Hopper work only,
so that failure-mode evidence can drive the next model revision instead
of relying on aggregate reward alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "glfw")

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


GEOM_NAMES = ["torso", "nose", "pelvis", "thigh", "calf", "foot"]
QPOS_NAMES = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]
QVEL_NAMES = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {x}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--exp_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--task", default="hopper-hop")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--render_failures", type=int, default=4)
    parser.add_argument("--render_successes", type=int, default=2)
    parser.add_argument("--single_pass", type=str2bool, default=False)
    parser.add_argument("--world_model_kind", default="sd_hopper")
    parser.add_argument("--model_size", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=256)
    parser.add_argument("--num_q", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_elites", type=int, default=16)
    parser.add_argument("--num_pi_trajs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sd_reward_mode", default="curriculum")
    parser.add_argument("--sd_reward_curriculum_alpha", type=float, default=0.5)
    parser.add_argument("--sd_reward_curriculum_alpha_min", type=float, default=0.25)
    parser.add_argument("--sd_reward_curriculum_start", type=int, default=10000)
    parser.add_argument("--sd_reward_curriculum_end", type=int, default=30000)
    parser.add_argument("--sd_max_resid_scale", type=float, default=5.0)
    parser.add_argument("--sd_height_loss_coef", type=float, default=0.05)
    parser.add_argument("--sd_use_struct_residual", type=str2bool, default=False)
    parser.add_argument("--zero_reward_threshold", type=float, default=0.05)
    parser.add_argument("--low_reward_threshold", type=float, default=1.0)
    return parser.parse_args()


def resolve_checkpoint(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.resolve()
    if args.exp_dir is None:
        raise ValueError("either --checkpoint or --exp_dir must be provided")
    models_dir = args.exp_dir / "models"
    candidates: List[Path] = []
    for name in ("final.pt", "final.full.pt"):
        path = models_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(sorted(models_dir.glob("*.pt")))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {models_dir}")

    def sort_key(path: Path):
        stem = path.stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    return sorted(set(candidates), key=sort_key)[-1].resolve()


def default_out_dir(checkpoint: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / "closed_loop" / f"{checkpoint.stem}_{stamp}"


def build_cfg(args, checkpoint: Path):
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
        "checkpoint": str(checkpoint),
        "exp_name": f"{checkpoint.stem}_closed_loop_eval",
        "sd_reward_mode": args.sd_reward_mode,
        "sd_reward_curriculum_alpha": args.sd_reward_curriculum_alpha,
        "sd_reward_curriculum_alpha_min": args.sd_reward_curriculum_alpha_min,
        "sd_reward_curriculum_start": args.sd_reward_curriculum_start,
        "sd_reward_curriculum_end": args.sd_reward_curriculum_end,
        "sd_max_resid_scale": args.sd_max_resid_scale,
        "sd_height_loss_coef": args.sd_height_loss_coef,
        "sd_use_struct_residual": args.sd_use_struct_residual,
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
    device = "cuda:0"
    agent = TDMPC2(cfg)
    payload = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    agent.model.load_state_dict(model_state, strict=True)
    agent.model.eval()
    return agent


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def physics_snapshot(physics) -> Dict[str, float]:
    geom_xpos = physics.named.data.geom_xpos
    qpos = physics.named.data.qpos
    qvel = physics.named.data.qvel
    snap: Dict[str, float] = {}
    for name in GEOM_NAMES:
        snap[f"{name}_x"] = float(geom_xpos[name, "x"])
        snap[f"{name}_z"] = float(geom_xpos[name, "z"])
    for name in QPOS_NAMES:
        snap[f"qpos_{name}"] = float(qpos[name])
    for name in QVEL_NAMES:
        snap[f"qvel_{name}"] = float(qvel[name])
    return snap


def v4_state_features(model, snap: Dict[str, float], struct: np.ndarray) -> Dict[str, float]:
    rootz = float(snap["qpos_rootz"])
    pitch = float(snap["qpos_rooty"])
    body_low = sigmoid((model.V4_ROOTZ_LOW_CENTER - rootz) / model.V4_ROOTZ_LOW_SCALE)
    body_high = 1.0 - body_low
    forward_lean = sigmoid((pitch - model.V4_PITCH_FWD_CENTER) / model.V4_PITCH_FWD_SCALE)
    recovered_upright = body_high * sigmoid((1.0 - abs(pitch)) / model.V4_PITCH_NEUTRAL_SPREAD)
    rear_trap = body_low * sigmoid(-(pitch - model.V4_PITCH_BACK_CENTER) / model.V4_PITCH_BACK_SCALE)
    crawl_lane = body_low * forward_lean
    support = float(struct[model.SUPPORT_IDX])
    clearance = float(struct[model.CLEARANCE_IDX])
    height = float(struct[model.HEIGHT_IDX])
    vx = float(struct[model.VX_IDX])
    vz = float(struct[model.VZ_IDX])
    front_anchor_pitch = sigmoid((pitch - model.V4_FRONT_ANCHOR_PITCH_CENTER) / model.V4_FRONT_ANCHOR_PITCH_SCALE)
    front_anchor_clearance = sigmoid(((-clearance) - model.V4_FRONT_ANCHOR_CLEARANCE_CENTER) / model.V4_FRONT_ANCHOR_CLEARANCE_SCALE)
    front_anchor_geom = body_low * front_anchor_pitch * front_anchor_clearance
    if struct.shape[0] > model.FRONT_ANCHOR_IDX:
        front_anchor = float(struct[model.FRONT_ANCHOR_IDX])
    else:
        front_anchor = front_anchor_geom
    unsupported_low = body_low * (1.0 - front_anchor) * sigmoid((model.V4_FRONT_ANCHOR_PITCH_CENTER - pitch) / model.V4_FRONT_ANCHOR_PITCH_SCALE)
    foot_low = 1.0 if snap["foot_z"] < 0.08 else 0.0
    nose_low = 1.0 if snap["nose_z"] < 0.08 else 0.0
    front_support = max(foot_low, nose_low)
    launch_like = 1.0 if (vz > 0.20 and support < 0.18) else 0.0
    return {
        "rootz": rootz,
        "pitch": pitch,
        "body_low": body_low,
        "body_high": body_high,
        "forward_lean": forward_lean,
        "recovered_upright": recovered_upright,
        "rear_trap": rear_trap,
        "crawl_lane": crawl_lane,
        "front_anchor_pitch": front_anchor_pitch,
        "front_anchor_clearance": front_anchor_clearance,
        "front_anchor_geom": front_anchor_geom,
        "front_anchor": front_anchor,
        "unsupported_low": unsupported_low,
        "support": support,
        "clearance": clearance,
        "height": height,
        "vx": vx,
        "vz": vz,
        "foot_low": foot_low,
        "nose_low": nose_low,
        "front_support": front_support,
        "launch_like": launch_like,
    }


def mean_of(rows: Iterable[Dict[str, float]], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return float(sum(values) / max(len(values), 1))


@dataclass
class EpisodeResult:
    episode_idx: int
    episode_reward: float
    length: int
    failure_mode: str
    summary: Dict[str, float]


def classify_episode(summary: Dict[str, float], zero_reward_threshold: float, low_reward_threshold: float) -> str:
    reward = summary["episode_reward"]
    if reward <= zero_reward_threshold:
        if summary["rear_trap_frac"] >= 0.25:
            return "rear_trap_zero"
        if summary["crawl_frac"] >= 0.25 and summary["launch_like_frac"] < 0.05:
            return "crawl_stall_zero"
        if summary["upright_frac"] >= 0.15 and summary["max_height"] >= 0.65:
            return "upright_no_drive_zero"
        return "flat_zero"
    if reward < low_reward_threshold:
        if summary["rear_trap_frac"] >= 0.20:
            return "rear_trap_low"
        if summary["crawl_frac"] >= 0.20 and summary["launch_like_frac"] < 0.10:
            return "crawl_stall_low"
        return "weak_progress"
    return "positive"


def summarize_episode(ep_idx: int, rows: List[Dict[str, float]], total_reward: float, zero_reward_threshold: float, low_reward_threshold: float) -> EpisodeResult:
    initial = rows[0]
    final = rows[-1]
    summary = {
        "episode": ep_idx,
        "episode_reward": total_reward,
        "length": len(rows),
        "initial_pitch": initial["pitch"],
        "initial_rootz": initial["rootz"],
        "initial_body_low": initial["body_low"],
        "rear_trap_frac": mean_of(rows, "rear_trap_gt_05"),
        "crawl_frac": mean_of(rows, "crawl_gt_05"),
        "upright_frac": mean_of(rows, "upright_gt_05"),
        "front_anchor_frac": mean_of(rows, "front_anchor_gt_05"),
        "unsupported_low_frac": mean_of(rows, "unsupported_low_gt_05"),
        "front_support_frac": mean_of(rows, "front_support"),
        "nose_low_frac": mean_of(rows, "nose_low"),
        "foot_low_frac": mean_of(rows, "foot_low"),
        "launch_like_frac": mean_of(rows, "launch_like"),
        "mean_support": mean_of(rows, "support"),
        "mean_clearance": mean_of(rows, "clearance"),
        "mean_vx": mean_of(rows, "vx"),
        "mean_vz": mean_of(rows, "vz"),
        "max_height": max(float(r["height"]) for r in rows),
        "max_vx": max(float(r["vx"]) for r in rows),
        "max_vz": max(float(r["vz"]) for r in rows),
        "max_rootx": max(float(r["qpos_rootx"]) for r in rows),
        "final_rootx": final["qpos_rootx"],
        "final_pitch": final["pitch"],
        "final_rootz": final["rootz"],
    }
    mode = classify_episode(summary, zero_reward_threshold, low_reward_threshold)
    return EpisodeResult(ep_idx, total_reward, len(rows), mode, summary)


def rollout_pass(env, agent, args, out_dir: Path, render_episode_ids: set[int]) -> List[EpisodeResult]:
    csv_dir = out_dir / "episode_csv"
    video_dir = out_dir / "videos"
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model = agent.model._dynamics
    results: List[EpisodeResult] = []
    device = "cuda:0"

    for ep_idx in range(args.episodes):
        obs = env.reset()
        done = False
        t = 0
        ep_reward = 0.0
        rows: List[Dict[str, float]] = []
        frames = [env.render()] if ep_idx in render_episode_ids else None

        while not done:
            physics = env.unwrapped.physics
            snap = physics_snapshot(physics)
            obs_cuda = obs.to(device, non_blocking=True).unsqueeze(0)
            with torch.no_grad():
                struct = agent.model._dynamics.encode_observation(obs_cuda)[0].detach().cpu().numpy()
            features = v4_state_features(model, snap, struct)
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            action_np = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
            obs_next, reward, done_flag, _info = env.step(action)
            ep_reward += float(reward.item())

            row: Dict[str, float] = {"episode": ep_idx, "step": t, "reward": float(reward.item()), "done": int(bool(done_flag))}
            row.update(snap)
            row.update(features)
            row["rear_trap_gt_05"] = 1.0 if features["rear_trap"] >= 0.5 else 0.0
            row["crawl_gt_05"] = 1.0 if features["crawl_lane"] >= 0.5 else 0.0
            row["upright_gt_05"] = 1.0 if features["recovered_upright"] >= 0.5 else 0.0
            row["front_anchor_gt_05"] = 1.0 if features["front_anchor"] >= 0.5 else 0.0
            row["unsupported_low_gt_05"] = 1.0 if features["unsupported_low"] >= 0.5 else 0.0
            for i in range(action_np.size):
                row[f"action_{i}"] = float(action_np[i])
            rows.append(row)

            obs = obs_next
            done = bool(done_flag)
            t += 1
            if frames is not None:
                frames.append(env.render())

        result = summarize_episode(ep_idx, rows, ep_reward, args.zero_reward_threshold, args.low_reward_threshold)
        result.summary["rendered"] = 1 if frames is not None else 0
        results.append(result)

        with open(csv_dir / f"episode_{ep_idx:02d}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if frames is not None:
            mp4_path = video_dir / f"episode_{ep_idx:02d}_{result.failure_mode}_R_{ep_reward:.2f}.mp4"
            imageio.mimsave(mp4_path, frames, fps=args.fps)
            result.summary["video"] = str(mp4_path)

        print(
            f"episode={ep_idx:02d} reward={ep_reward:.4f} mode={result.failure_mode} "
            f"rear={result.summary['rear_trap_frac']:.2f} crawl={result.summary['crawl_frac']:.2f} "
            f"upright={result.summary['upright_frac']:.2f} max_rootx={result.summary['max_rootx']:.2f}"
        )

    return results


def choose_render_episodes(results: List[EpisodeResult], render_failures: int, render_successes: int) -> set[int]:
    failures = [r for r in results if r.failure_mode != "positive"]
    failures = sorted(failures, key=lambda r: (r.episode_reward, -r.summary["rear_trap_frac"], -r.summary["crawl_frac"]))
    positives = [r for r in results if r.failure_mode == "positive"]
    positives = sorted(positives, key=lambda r: (-r.episode_reward, -r.summary["max_rootx"]))
    selected = {r.episode_idx for r in failures[:render_failures]}
    selected.update(r.episode_idx for r in positives[:render_successes])
    return selected


def aggregate_diagnosis(results: List[EpisodeResult]) -> Dict[str, object]:
    mode_counter = Counter(r.failure_mode for r in results)
    zero_like = [r for r in results if r.failure_mode.endswith("zero")]
    low_like = [r for r in results if r.failure_mode.endswith("low")]
    positives = [r for r in results if r.failure_mode == "positive"]
    all_rewards = [r.episode_reward for r in results]
    zero_init_backward_frac = 0.0
    if zero_like:
        zero_init_backward_frac = sum(1 for r in zero_like if r.summary["initial_pitch"] < 0.0) / len(zero_like)

    dominant_mode = mode_counter.most_common(1)[0][0]
    next_hypothesis = "mixed_failure_modes"
    next_action = "继续扩展状态日志，并对零分 episode 做定点视频检查。"
    if mode_counter["rear_trap_zero"] >= max(2, len(zero_like) // 2):
        next_hypothesis = "rear_trap_recovery"
        next_action = "优先补前支撑/后仰恢复机制，而不是继续放大一般性 shaping。"
    elif mode_counter["crawl_stall_zero"] + mode_counter["crawl_stall_low"] >= 2:
        next_hypothesis = "crawl_to_release_gap"
        next_action = "优先补从低身前倾爬行到释放起跳的结构量，例如 front_support/front_clearance 或 release trigger。"
    elif mode_counter["upright_no_drive_zero"] >= 1:
        next_hypothesis = "upright_without_drive"
        next_action = "优先增强立起后驱动与起跳释放，而不是再修 rear_trap。"

    return {
        "checkpoint_reward_mean": float(sum(all_rewards) / max(len(all_rewards), 1)),
        "checkpoint_reward_max": float(max(all_rewards) if all_rewards else 0.0),
        "mode_counts": dict(mode_counter),
        "dominant_mode": dominant_mode,
        "zero_episode_count": len(zero_like),
        "low_episode_count": len(low_like),
        "positive_episode_count": len(positives),
        "zero_init_backward_frac": zero_init_backward_frac,
        "next_hypothesis": next_hypothesis,
        "next_action": next_action,
    }


def write_summary_files(out_dir: Path, checkpoint: Path, args, results: List[EpisodeResult], diagnosis: Dict[str, object]):
    summary_rows = [r.summary | {"failure_mode": r.failure_mode} for r in results]
    summary_csv = out_dir / "episode_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    diagnosis_json = out_dir / "diagnosis.json"
    args_jsonable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with open(diagnosis_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "args": args_jsonable,
                "diagnosis": diagnosis,
                "episodes": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    report_md = out_dir / "report.md"
    lines = [
        "# v4 Hopper Closed-Loop Diagnosis",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- mean reward: `{diagnosis['checkpoint_reward_mean']:.4f}`",
        f"- max reward: `{diagnosis['checkpoint_reward_max']:.4f}`",
        f"- dominant mode: `{diagnosis['dominant_mode']}`",
        f"- next hypothesis: `{diagnosis['next_hypothesis']}`",
        "",
        "## Aggregate",
        "",
        f"- mode counts: `{json.dumps(diagnosis['mode_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- zero episode count: `{diagnosis['zero_episode_count']}`",
        f"- low episode count: `{diagnosis['low_episode_count']}`",
        f"- positive episode count: `{diagnosis['positive_episode_count']}`",
        f"- zero episodes with initial backward pitch fraction: `{diagnosis['zero_init_backward_frac']:.2f}`",
        "",
        "## Interpretation",
        "",
        diagnosis["next_action"],
        "",
        "## Episodes",
        "",
        "| episode | reward | mode | init_pitch | rear_frac | crawl_frac | upright_frac | max_rootx | video |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        video_name = Path(row["video"]).name if "video" in row else ""
        lines.append(
            f"| {int(row['episode'])} | {row['episode_reward']:.3f} | {row['failure_mode']} | "
            f"{row['initial_pitch']:.3f} | {row['rear_trap_frac']:.2f} | {row['crawl_frac']:.2f} | "
            f"{row['upright_frac']:.2f} | {row['max_rootx']:.2f} | {video_name} |"
        )

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    next_smoke = out_dir / "next_smoke.sh"
    exp_name = f"TODO_patchlabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    next_smoke_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{REPO_ROOT}"
mkdir -p results/closed_loop_smokes
env MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}} /home/anaconda3/envs/sdmcts_p100/bin/python -u external/tdmpc2/tdmpc2/train.py \\
  task=hopper-hop world_model_kind=sd_hopper model_size=null obs=state steps=5000 seed={args.seed} \\
  exp_name={exp_name} \\
  enable_wandb=false save_video=false save_agent=true save_csv=true \\
  eval_episodes=10 eval_freq=1000 compile=false mlp_dim={args.mlp_dim} num_q={args.num_q} \\
  sd_reward_mode={args.sd_reward_mode} sd_reward_curriculum_alpha={args.sd_reward_curriculum_alpha} \\
  sd_reward_curriculum_alpha_min={args.sd_reward_curriculum_alpha_min} sd_reward_curriculum_start={args.sd_reward_curriculum_start} \\
  sd_reward_curriculum_end={args.sd_reward_curriculum_end} sd_max_resid_scale={args.sd_max_resid_scale} \\
  sd_height_loss_coef={args.sd_height_loss_coef} sd_use_struct_residual={'true' if args.sd_use_struct_residual else 'false'}
"""
    with open(next_smoke, "w", encoding="utf-8") as f:
        f.write(next_smoke_cmd)


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required for Hopper replay."

    checkpoint = resolve_checkpoint(args)
    out_dir = args.out_dir.resolve() if args.out_dir is not None else default_out_dir(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, env = build_cfg(args, checkpoint)
    set_seed(cfg.seed)
    agent = load_agent(cfg)

    print(f"[closed-loop] checkpoint={checkpoint}")
    print(f"[closed-loop] out_dir={out_dir}")

    if args.single_pass:
        final_results = rollout_pass(env, agent, args, out_dir, render_episode_ids=set(range(args.episodes)))
    else:
        dry_results = rollout_pass(env, agent, args, out_dir, render_episode_ids=set())
        render_episode_ids = choose_render_episodes(dry_results, args.render_failures, args.render_successes)

        env.close()
        cfg, env = build_cfg(args, checkpoint)
        set_seed(cfg.seed)
        agent = load_agent(cfg)
        final_results = rollout_pass(env, agent, args, out_dir, render_episode_ids=render_episode_ids)
    diagnosis = aggregate_diagnosis(final_results)
    write_summary_files(out_dir, checkpoint, args, final_results, diagnosis)

    print(f"[closed-loop] diagnosis={json.dumps(diagnosis, ensure_ascii=False, sort_keys=True)}")
    print(f"[closed-loop] report={out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
