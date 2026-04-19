"""Fast simulator-backed semantic checks for v4 Hopper structure changes.

This script is designed to catch logic regressions in structural quantities
without running training. It rolls the DMControl Hopper simulator for a short,
fixed budget, mines physically interpretable states, and checks whether the
SD world model assigns consistent semantic values.

Primary use cases:
1. Verify that encoded torso pitch preserves the raw simulator angle.
2. Verify that rear-trap states receive high rear_trap and low front_anchor.
3. Verify that upright states are not misclassified as rear-trap.
4. Optionally inspect whether a forward-anchor-like state is discoverable under
   short scripted/random interaction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "glfw")

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TDMPC2_ROOT = REPO_ROOT / "external" / "tdmpc2" / "tdmpc2"
if str(TDMPC2_ROOT) not in sys.path:
    sys.path.insert(0, str(TDMPC2_ROOT))

import hydra.utils  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs.dmcontrol import make_env as make_dmcontrol_env  # noqa: E402
from envs.wrappers.tensor import TensorWrapper  # noqa: E402
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--steps_per_episode', type=int, default=120)
    p.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'results' / 'structural_sanity')
    p.add_argument('--rear_pitch_thresh', type=float, default=-2.5)
    p.add_argument('--rear_front_anchor_max', type=float, default=0.2)
    p.add_argument('--rear_rear_trap_min', type=float, default=0.7)
    p.add_argument('--upright_pitch_abs_max', type=float, default=0.3)
    p.add_argument('--upright_height_min', type=float, default=0.7)
    p.add_argument('--upright_rear_trap_max', type=float, default=0.1)
    return p.parse_args()


def build_cfg(seed: int):
    hydra.utils.get_original_cwd = lambda: str(REPO_ROOT)
    cfg = OmegaConf.load(TDMPC2_ROOT / 'config.yaml')
    overrides = {
        'task': 'hopper-hop',
        'obs': 'state',
        'seed': seed,
        'world_model_kind': 'sd_hopper',
        'model_size': None,
        'mlp_dim': 256,
        'num_q': 3,
        'compile': False,
        'save_video': False,
        'enable_wandb': False,
        'exp_name': 'v4_structural_sanity_check',
        'sd_reward_mode': 'curriculum',
        'sd_reward_curriculum_alpha': 0.5,
        'sd_reward_curriculum_alpha_min': 0.25,
        'sd_reward_curriculum_start': 10000,
        'sd_reward_curriculum_end': 30000,
        'sd_max_resid_scale': 5.0,
        'sd_height_loss_coef': 0.05,
        'sd_use_struct_residual': False,
    }
    for k, v in overrides.items():
        cfg[k] = v
    cfg = parse_cfg(cfg)
    env = TensorWrapper(make_dmcontrol_env(cfg))
    return cfg, env


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def physics_snapshot(physics):
    geom_xpos = physics.named.data.geom_xpos
    qpos = physics.named.data.qpos
    return {
        'nose_z': float(geom_xpos['nose', 'z']),
        'foot_z': float(geom_xpos['foot', 'z']),
        'rootx': float(qpos['rootx']),
        'rootz_raw': float(qpos['rootz']),
        'pitch_raw': float(qpos['rooty']),
    }


def semantic_features(model: DMControlHopperSDModel, struct: np.ndarray, snap: dict):
    rootz = float(struct[model.ROOTZ_IDX])
    pitch = float(struct[model.PITCH_IDX])
    height = float(struct[model.HEIGHT_IDX])
    vx = float(struct[model.VX_IDX])
    vz = float(struct[model.VZ_IDX])
    support = float(struct[model.SUPPORT_IDX])
    clearance = float(struct[model.CLEARANCE_IDX])
    front_anchor = float(struct[model.FRONT_ANCHOR_IDX])
    body_low = sigmoid((model.V4_ROOTZ_LOW_CENTER - rootz) / model.V4_ROOTZ_LOW_SCALE)
    body_high = 1.0 - body_low
    forward_lean = sigmoid((pitch - model.V4_PITCH_FWD_CENTER) / model.V4_PITCH_FWD_SCALE)
    recovered_upright = body_high * sigmoid((1.0 - abs(pitch)) / model.V4_PITCH_NEUTRAL_SPREAD)
    rear_trap = body_low * sigmoid(-(pitch - model.V4_PITCH_BACK_CENTER) / model.V4_PITCH_BACK_SCALE)
    unsupported_low = body_low * (1.0 - front_anchor) * sigmoid((model.V4_FRONT_ANCHOR_PITCH_CENTER - pitch) / model.V4_FRONT_ANCHOR_PITCH_SCALE)
    nose_low = 1.0 if snap['nose_z'] < 0.08 else 0.0
    foot_low = 1.0 if snap['foot_z'] < 0.08 else 0.0
    front_support = max(nose_low, foot_low)
    return {
        'pitch_struct': pitch,
        'height': height,
        'vx': vx,
        'vz': vz,
        'support': support,
        'clearance': clearance,
        'front_anchor': front_anchor,
        'body_low': body_low,
        'recovered_upright': recovered_upright,
        'rear_trap': rear_trap,
        'unsupported_low': unsupported_low,
        'nose_low': nose_low,
        'foot_low': foot_low,
        'front_support': front_support,
    }


def scripted_action(ep_idx: int, step: int, rng: np.random.Generator) -> np.ndarray:
    mode = ep_idx % 4
    if mode == 0:
        return np.zeros(4, dtype=np.float32)
    if mode == 1:
        phase = step % 20
        if phase < 10:
            return np.array([0.2, 0.8, -0.7, -0.6], dtype=np.float32)
        return np.array([0.3, -0.8, 0.8, 0.8], dtype=np.float32)
    if mode == 2:
        return rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
    return np.array([-0.3, 0.4, -0.4, 0.7], dtype=np.float32)


def candidate_type(row: dict, args) -> str | None:
    if row['pitch_raw'] < args.rear_pitch_thresh and row['nose_low'] > 0.5 and row['front_support'] > 0.5:
        return 'rear_nose_support'
    if abs(row['pitch_raw']) < args.upright_pitch_abs_max and row['height'] > args.upright_height_min:
        return 'upright_recovered'
    if row['pitch_raw'] > 1.0 and row['front_support'] > 0.5 and row['front_anchor'] > 0.3:
        return 'forward_anchor_like'
    return None


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg, env = build_cfg(args.seed)
    set_seed(cfg.seed)
    model = DMControlHopperSDModel().eval()
    rng = np.random.default_rng(args.seed)

    rows = []
    pitch_abs_err = []
    candidates = {'rear_nose_support': [], 'upright_recovered': [], 'forward_anchor_like': []}

    for ep in range(args.episodes):
        obs = env.reset()
        for t in range(args.steps_per_episode):
            physics = env.unwrapped.physics
            snap = physics_snapshot(physics)
            struct = model.encode_observation(obs.unsqueeze(0))[0].detach().cpu().numpy()
            feats = semantic_features(model, struct, snap)
            row = {
                'episode': ep,
                'step': t,
                **snap,
                **feats,
            }
            rows.append(row)
            pitch_abs_err.append(abs(row['pitch_struct'] - row['pitch_raw']))
            ctype = candidate_type(row, args)
            if ctype is not None and len(candidates[ctype]) < 5:
                candidates[ctype].append(row)

            action_np = scripted_action(ep, t, rng)
            action = torch.tensor(action_np, dtype=torch.float32)
            obs, _, done, _ = env.step(action)
            if bool(done):
                break

    checks = {}
    rear_rows = candidates['rear_nose_support']
    if rear_rows:
        checks['rear_nose_support'] = {
            'covered': True,
            'count': len(rear_rows),
            'mean_front_anchor': float(np.mean([r['front_anchor'] for r in rear_rows])),
            'mean_rear_trap': float(np.mean([r['rear_trap'] for r in rear_rows])),
            'pass': bool(np.mean([r['front_anchor'] for r in rear_rows]) <= args.rear_front_anchor_max and np.mean([r['rear_trap'] for r in rear_rows]) >= args.rear_rear_trap_min),
        }
    else:
        checks['rear_nose_support'] = {'covered': False, 'count': 0, 'pass': False}

    upright_rows = candidates['upright_recovered']
    if upright_rows:
        checks['upright_recovered'] = {
            'covered': True,
            'count': len(upright_rows),
            'mean_recovered_upright': float(np.mean([r['recovered_upright'] for r in upright_rows])),
            'mean_rear_trap': float(np.mean([r['rear_trap'] for r in upright_rows])),
            'pass': bool(np.mean([r['recovered_upright'] for r in upright_rows]) >= 0.5 and np.mean([r['rear_trap'] for r in upright_rows]) <= args.upright_rear_trap_max),
        }
    else:
        checks['upright_recovered'] = {'covered': False, 'count': 0, 'pass': False}

    forward_rows = candidates['forward_anchor_like']
    if forward_rows:
        checks['forward_anchor_like'] = {
            'covered': True,
            'count': len(forward_rows),
            'mean_front_anchor': float(np.mean([r['front_anchor'] for r in forward_rows])),
            'mean_rear_trap': float(np.mean([r['rear_trap'] for r in forward_rows])),
            'pass': bool(np.mean([r['front_anchor'] for r in forward_rows]) >= 0.3 and np.mean([r['rear_trap'] for r in forward_rows]) <= 0.2),
        }
    else:
        checks['forward_anchor_like'] = {'covered': False, 'count': 0, 'pass': True}

    covered_check_names = [name for name, info in checks.items() if info.get('covered', False)]
    uncovered_check_names = [name for name, info in checks.items() if not info.get('covered', False)]
    covered_pass = all(checks[name]['pass'] for name in covered_check_names)

    report = {
        'episodes': args.episodes,
        'steps_per_episode': args.steps_per_episode,
        'total_samples': len(rows),
        'pitch_abs_err_mean': float(np.mean(pitch_abs_err)) if pitch_abs_err else None,
        'pitch_abs_err_max': float(np.max(pitch_abs_err)) if pitch_abs_err else None,
        'checks': checks,
        'coverage_warnings': uncovered_check_names,
        'candidate_examples': {k: v for k, v in candidates.items()},
        'overall_pass': bool((np.max(pitch_abs_err) if pitch_abs_err else 1.0) < 1e-4 and covered_pass),
    }

    with open(args.out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
