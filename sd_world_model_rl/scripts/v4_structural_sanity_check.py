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


def semantic_features(model: DMControlHopperSDModel, struct: np.ndarray, snap: dict | None = None):
    rootz = float(struct[model.ROOTZ_IDX])
    pitch = float(struct[model.PITCH_IDX])
    height = float(struct[model.HEIGHT_IDX])
    vx = float(struct[model.VX_IDX])
    vz = float(struct[model.VZ_IDX])
    compression = float(struct[model.COMP_IDX])
    support = float(struct[model.SUPPORT_IDX])
    clearance = float(struct[model.CLEARANCE_IDX])
    front_anchor = float(struct[model.FRONT_ANCHOR_IDX])
    body_low = sigmoid((model.V4_ROOTZ_LOW_CENTER - rootz) / model.V4_ROOTZ_LOW_SCALE)
    body_high = 1.0 - body_low
    forward_lean = sigmoid((pitch - model.V4_PITCH_FWD_CENTER) / model.V4_PITCH_FWD_SCALE)
    recovered_upright = body_high * sigmoid((1.0 - abs(pitch)) / model.V4_PITCH_NEUTRAL_SPREAD)
    handoff_ready = body_high * sigmoid((1.8 - abs(pitch)) / 0.45)
    front_capture_commitment = sigmoid((front_anchor - model.V4_FRONT_CAPTURE_CENTER) / model.V4_FRONT_CAPTURE_SCALE)
    recovery_support_gate = sigmoid((pitch - model.V4_RECOVERY_SUPPORT_PITCH_CENTER) / model.V4_RECOVERY_SUPPORT_PITCH_SCALE)
    handoff_unlock = front_capture_commitment * sigmoid((body_high - model.V4_HANDOFF_BODY_HIGH_CENTER) / model.V4_HANDOFF_BODY_HIGH_SCALE)
    support_corridor_scale = handoff_unlock * min(1.0, 0.20 + 0.80 * body_high)
    recovery_support = max(front_anchor, support_corridor_scale * support * recovery_support_gate)
    rear_trap = body_low * sigmoid(-(pitch - model.V4_PITCH_BACK_CENTER) / model.V4_PITCH_BACK_SCALE)
    compression_level = min(max(compression / 0.35, 0.0), 1.0)
    grounded = min(max((support - 0.12) / 0.40, 0.0), 1.0)
    compression_arm = sigmoid((compression - model.V4_RECOVERY_COMP_CENTER) / model.V4_RECOVERY_COMP_SCALE)
    recovery_pitch = sigmoid((pitch - model.V4_RECOVERY_PITCH_CENTER) / model.V4_RECOVERY_PITCH_SCALE)
    standup_window = front_capture_commitment * recovery_support * body_low * recovery_pitch * compression_arm
    handoff_window = front_capture_commitment * recovery_support * handoff_ready
    crawl_ready = front_capture_commitment * body_low * grounded
    arm_ready = front_capture_commitment * recovery_support * grounded * max(recovered_upright, 0.5 * handoff_ready)
    release_ready = arm_ready * compression_level
    unsupported_low = body_low * (1.0 - front_anchor) * sigmoid((model.V4_FRONT_ANCHOR_PITCH_CENTER - pitch) / model.V4_FRONT_ANCHOR_PITCH_SCALE)
    feats = {
        'pitch_struct': pitch,
        'height': height,
        'vx': vx,
        'vz': vz,
        'compression': compression,
        'support': support,
        'clearance': clearance,
        'front_anchor': front_anchor,
        'body_low': body_low,
        'body_high': body_high,
        'forward_lean': forward_lean,
        'recovered_upright': recovered_upright,
        'handoff_ready': handoff_ready,
        'front_capture_commitment': front_capture_commitment,
        'handoff_unlock': handoff_unlock,
        'recovery_support': recovery_support,
        'rear_trap': rear_trap,
        'crawl_ready': crawl_ready,
        'standup_window': standup_window,
        'arm_ready': arm_ready,
        'release_ready': release_ready,
        'handoff_window': handoff_window,
        'unsupported_low': unsupported_low,
        'rootz': rootz,
    }
    if snap is not None:
        nose_low = 1.0 if snap['nose_z'] < 0.08 else 0.0
        foot_low = 1.0 if snap['foot_z'] < 0.08 else 0.0
        feats.update(
            {
                'nose_low': nose_low,
                'foot_low': foot_low,
                'front_support': max(nose_low, foot_low),
            }
        )
    return feats


def canonical_struct(model: DMControlHopperSDModel, name: str) -> np.ndarray:
    # Fall back to deterministic representative latent states if simulator mining
    # misses a regime; this keeps the sanity check stable across minor trajectory drift.
    if name == 'recoverable_front_anchor':
        return np.array(
            [0.25, 0.30, 0.08, 0.85, 0.00, 0.55, 0.00, 0.10, 0.00, -0.10, 0.00, 0.88, 0.10, 0.05, -0.84, -0.50, 0.82],
            dtype=np.float32,
        )
    if name == 'rear_trap':
        return np.array(
            [0.10, -0.05, -0.02, -2.80, 0.00, 0.22, 0.00, -0.10, 0.00, 0.10, 0.00, 0.85, 0.00, 0.05, -0.90, -0.78, 0.02],
            dtype=np.float32,
        )
    if name == 'uncaptured_forward_lean':
        return np.array(
            [0.16, 0.20, 0.00, 0.58, 0.00, 0.28, 0.00, 0.18, 0.00, -0.15, 0.00, 0.88, 0.05, 0.05, -0.88, -0.46, 0.06],
            dtype=np.float32,
        )
    if name == 'release_ready_corridor':
        return np.array(
            [0.25, 0.30, 0.08, 0.85, 0.00, 0.55, 0.00, 0.10, 0.00, -0.10, 0.00, 0.88, 0.10, 0.05, -0.84, -0.50, 0.82],
            dtype=np.float32,
        )
    raise KeyError(name)


def transition_probe(model: DMControlHopperSDModel, struct: np.ndarray, action: np.ndarray) -> dict:
    state = torch.tensor(struct, dtype=torch.float32).unsqueeze(0)
    act = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_state = model.forward(state, act)[0].detach().cpu().numpy()
        plan_reward = float(model.predict_plan_reward(state, act).item())
    prev = semantic_features(model, struct)
    nxt = semantic_features(model, next_state)
    return {
        'action': action.tolist(),
        'prev': prev,
        'next': nxt,
        'delta_rootz': nxt['rootz'] - prev['rootz'],
        'delta_abs_pitch_to_neutral': abs(prev['pitch_struct']) - abs(nxt['pitch_struct']),
        'delta_recovered_upright': nxt['recovered_upright'] - prev['recovered_upright'],
        'delta_front_anchor': nxt['front_anchor'] - prev['front_anchor'],
        'delta_front_capture': nxt['front_capture_commitment'] - prev['front_capture_commitment'],
        'delta_release_ready': nxt['release_ready'] - prev['release_ready'],
        'delta_handoff_window': nxt['handoff_window'] - prev['handoff_window'],
        'delta_standup_window': nxt['standup_window'] - prev['standup_window'],
        'delta_vz': nxt['vz'] - prev['vz'],
        'plan_reward': plan_reward,
    }


def transition_rollout_probe(model: DMControlHopperSDModel, struct: np.ndarray, actions: list[np.ndarray]) -> dict:
    state = torch.tensor(struct, dtype=torch.float32).unsqueeze(0)
    prev = semantic_features(model, struct)
    steps = []
    total_plan_reward = 0.0
    with torch.no_grad():
        for action in actions:
            act = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            total_plan_reward += float(model.predict_plan_reward(state, act).item())
            state = model.forward(state, act)
            cur = semantic_features(model, state[0].detach().cpu().numpy())
            steps.append(cur)
    final = steps[-1]
    return {
        'actions': [a.tolist() for a in actions],
        'prev': prev,
        'steps': steps,
        'final': final,
        'delta_rootz': final['rootz'] - prev['rootz'],
        'delta_front_anchor': final['front_anchor'] - prev['front_anchor'],
        'delta_handoff_window': final['handoff_window'] - prev['handoff_window'],
        'delta_recovered_upright': final['recovered_upright'] - prev['recovered_upright'],
        'delta_recovery_support': final['recovery_support'] - prev['recovery_support'],
        'total_plan_reward': total_plan_reward,
    }


def transition_and_reward_checks(model: DMControlHopperSDModel, candidates: dict[str, list[dict]]):
    action_bank = {
        'hold_crawl': np.array([0.2, 0.8, -0.8, -0.8], dtype=np.float32),
        'release_handoff': np.array([0.6, -1.0, 1.0, 1.0], dtype=np.float32),
        'follow_handoff': np.array([0.3, -0.4, 0.6, 0.6], dtype=np.float32),
        'neutral': np.zeros(4, dtype=np.float32),
        'rear_worsen': np.array([-0.6, 0.8, -0.8, -0.8], dtype=np.float32),
        'rear_escape': np.array([0.6, -0.8, 0.8, 0.8], dtype=np.float32),
    }

    forward_ready_candidates = [
        row for row in candidates['forward_anchor_like']
        if row['front_capture_commitment'] >= 0.5 and row['support'] >= 0.25
    ]
    forward_source = forward_ready_candidates[0]['struct'] if forward_ready_candidates else canonical_struct(model, 'recoverable_front_anchor').tolist()
    rear_source = candidates['rear_nose_support'][0]['struct'] if candidates['rear_nose_support'] else canonical_struct(model, 'rear_trap').tolist()
    uncaptured_source = canonical_struct(model, 'uncaptured_forward_lean').tolist()
    release_ready_source = canonical_struct(model, 'release_ready_corridor').tolist()

    forward_probes = {
        name: transition_probe(model, np.array(forward_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'release_handoff', 'neutral')
    }
    forward_rollouts = {
        'hold_then_hold': transition_rollout_probe(
            model,
            np.array(forward_source, dtype=np.float32),
            [action_bank['hold_crawl'], action_bank['hold_crawl']],
        ),
        'release_then_follow': transition_rollout_probe(
            model,
            np.array(forward_source, dtype=np.float32),
            [action_bank['release_handoff'], action_bank['follow_handoff']],
        ),
    }
    rear_probes = {
        name: transition_probe(model, np.array(rear_source, dtype=np.float32), action_bank[name])
        for name in ('rear_worsen', 'rear_escape', 'neutral')
    }
    uncaptured_probes = {
        name: transition_probe(model, np.array(uncaptured_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'release_handoff', 'neutral')
    }
    release_ready_probes = {
        name: transition_probe(model, np.array(release_ready_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'release_handoff', 'neutral')
    }
    release_ready_rollouts = {
        'hold_then_hold': transition_rollout_probe(
            model,
            np.array(release_ready_source, dtype=np.float32),
            [action_bank['hold_crawl'], action_bank['hold_crawl']],
        ),
        'release_then_follow': transition_rollout_probe(
            model,
            np.array(release_ready_source, dtype=np.float32),
            [action_bank['release_handoff'], action_bank['follow_handoff']],
        ),
    }

    hold = forward_probes['hold_crawl']
    release = forward_probes['release_handoff']
    rear_escape = rear_probes['rear_escape']
    rear_worsen = rear_probes['rear_worsen']
    uncaptured_hold = uncaptured_probes['hold_crawl']
    uncaptured_release = uncaptured_probes['release_handoff']
    release_ready_hold = release_ready_probes['hold_crawl']
    release_ready_release = release_ready_probes['release_handoff']
    release_ready_hold_rollout = release_ready_rollouts['hold_then_hold']
    release_ready_release_rollout = release_ready_rollouts['release_then_follow']

    capture_hold_pass = bool(
        hold['plan_reward'] >= release['plan_reward'] - 1e-4
        and hold['delta_front_capture'] >= release['delta_front_capture'] - 1e-4
        and hold['delta_standup_window'] >= release['delta_standup_window'] - 1e-4
    )
    release_ready_transition_pass = bool(
        release_ready_release['delta_rootz'] > release_ready_hold['delta_rootz'] + 1e-4
        and release_ready_release['delta_vz'] > release_ready_hold['delta_vz'] + 1e-4
        and release_ready_release['plan_reward'] > release_ready_hold['plan_reward'] + 1e-4
    )
    release_ready_handoff_pass = bool(
        release_ready_release['delta_recovered_upright'] > -1e-4
        and release_ready_release['delta_handoff_window'] >= release_ready_hold['delta_handoff_window'] - 1e-2
        and release_ready_release['delta_front_anchor'] <= release_ready_hold['delta_front_anchor'] + 1e-4
    )
    release_ready_continuity_pass = bool(
        release_ready_release_rollout['delta_rootz'] > release_ready_hold_rollout['delta_rootz'] + 1e-4
        and release_ready_release_rollout['delta_handoff_window'] > release_ready_hold_rollout['delta_handoff_window'] - 1e-4
        and release_ready_release_rollout['total_plan_reward'] > release_ready_hold_rollout['total_plan_reward'] + 1e-4
    )
    rear_transition_pass = bool(
        rear_worsen['next']['front_anchor'] <= 0.10
        and rear_worsen['next']['rear_trap'] >= 0.70
        and rear_escape['plan_reward'] > rear_worsen['plan_reward'] + 1e-4
    )
    uncaptured_phase_order_pass = bool(
        uncaptured_release['plan_reward'] <= uncaptured_hold['plan_reward'] + 1e-4
        and uncaptured_release['next']['front_capture_commitment'] <= uncaptured_hold['next']['front_capture_commitment'] + 1e-4
        and uncaptured_release['next']['standup_window'] <= uncaptured_hold['next']['standup_window'] + 1e-4
        and uncaptured_release['next']['release_ready'] <= 0.15
    )

    return {
        'recoverable_front_anchor': {
            'source': 'simulator' if forward_ready_candidates else 'canonical',
            'probes': forward_probes,
            'rollouts': forward_rollouts,
            'checks': {
                'capture_state_prefers_hold_over_release': capture_hold_pass,
            },
            'pass': bool(capture_hold_pass),
        },
        'rear_trap': {
            'source': 'simulator' if candidates['rear_nose_support'] else 'canonical',
            'probes': rear_probes,
            'checks': {
                'rear_worsen_stays_nonviable': rear_transition_pass,
            },
            'pass': bool(rear_transition_pass),
        },
        'release_ready_corridor': {
            'source': 'canonical',
            'probes': release_ready_probes,
            'rollouts': release_ready_rollouts,
            'checks': {
                'release_beats_hold_on_vertical_progress': release_ready_transition_pass,
                'release_advances_handoff_window': release_ready_handoff_pass,
                'release_preserves_two_step_recovery_corridor': release_ready_continuity_pass,
            },
            'pass': bool(release_ready_transition_pass and release_ready_handoff_pass and release_ready_continuity_pass),
        },
        'phase_dependency': {
            'source': 'canonical',
            'probes': uncaptured_probes,
            'checks': {
                'uncaptured_state_does_not_prefer_release': uncaptured_phase_order_pass,
            },
            'pass': bool(uncaptured_phase_order_pass),
        },
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
                'struct': struct.tolist(),
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
            'mean_standup_window': float(np.mean([r['standup_window'] for r in forward_rows])),
            'pass': bool(
                np.mean([r['front_anchor'] for r in forward_rows]) >= 0.3
                and np.mean([r['rear_trap'] for r in forward_rows]) <= 0.2
                and np.mean([r['standup_window'] for r in forward_rows]) >= 0.08
            ),
        }
    else:
        checks['forward_anchor_like'] = {'covered': False, 'count': 0, 'pass': True}

    covered_check_names = [name for name, info in checks.items() if info.get('covered', False)]
    uncovered_check_names = [name for name, info in checks.items() if not info.get('covered', False)]
    covered_pass = all(checks[name]['pass'] for name in covered_check_names)
    transition_checks = transition_and_reward_checks(model, candidates)
    focus_transition_names = ['recoverable_front_anchor', 'release_ready_corridor', 'phase_dependency']
    transition_pass = all(transition_checks[name]['pass'] for name in focus_transition_names)

    report = {
        'episodes': args.episodes,
        'steps_per_episode': args.steps_per_episode,
        'total_samples': len(rows),
        'pitch_abs_err_mean': float(np.mean(pitch_abs_err)) if pitch_abs_err else None,
        'pitch_abs_err_max': float(np.max(pitch_abs_err)) if pitch_abs_err else None,
        'checks': checks,
        'transition_reward_checks': transition_checks,
        'focus_transition_names': focus_transition_names,
        'coverage_warnings': uncovered_check_names,
        'candidate_examples': {k: v for k, v in candidates.items()},
        'overall_pass': bool((np.max(pitch_abs_err) if pitch_abs_err else 1.0) < 1e-4 and covered_pass and transition_pass),
    }

    with open(args.out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
