"""Fast simulator-backed semantic checks for v4 Hopper structure changes.

This script is designed to catch logic regressions in structural quantities
without running training. It rolls the DMControl Hopper simulator for a short,
fixed budget, mines physically interpretable states, and checks whether the
SD world model assigns consistent semantic values.

Primary use cases:
1. Verify that encoded torso pitch preserves the raw simulator angle.
2. Verify that rear-trap states receive high rear_trap and low capture_memory.
3. Verify that upright states are not misclassified as rear-trap.
4. Optionally inspect whether a forward-capture-like state is discoverable under
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
from common.sd_world_model import SDWorldModel  # noqa: E402
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
    p.add_argument('--rear_capture_max', type=float, default=0.2)
    p.add_argument('--rear_rear_trap_min', type=float, default=0.7)
    p.add_argument('--upright_pitch_abs_max', type=float, default=0.3)
    p.add_argument('--upright_height_min', type=float, default=0.7)
    p.add_argument('--upright_rear_trap_max', type=float, default=0.1)
    p.add_argument('--transition_one_step_rmse_max', type=float, default=0.35)
    p.add_argument('--transition_three_step_rmse_max', type=float, default=0.50)
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
        'sd_height_loss_coef': 0.0,
        'sd_use_struct_residual': False,
        'sd_use_local_residual': True,
        'sd_local_residual_scale': 0.20,
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


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict_next_struct(world_model: SDWorldModel, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return world_model.next(state, action, task=None)


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
    state = torch.tensor(struct, dtype=torch.float32).unsqueeze(0)
    rootz = float(struct[model.ROOTZ_IDX])
    pitch = float(struct[model.PITCH_IDX])
    vx = float(struct[model.VX_IDX])
    vz = float(struct[model.VZ_IDX])
    compression = float(struct[model.COMP_IDX])
    support = float(struct[model.SUPPORT_IDX])
    capture_memory = float(struct[model.CAPTURE_IDX])

    with torch.no_grad():
        height = float(model.predict_height(state)[0].item())
        body_low = float(model._body_low_from_rootz(state[:, model.ROOTZ_IDX])[0].item())
        body_high = 1.0 - body_low
        forward_lean = float(torch.sigmoid((state[:, model.PITCH_IDX] - model.V4_PITCH_FWD_CENTER) / model.V4_PITCH_FWD_SCALE)[0].item())
        recovered_upright = float(model._recovered_upright_from_state(state[:, model.PITCH_IDX], state[:, model.ROOTZ_IDX])[0].item())
        handoff_ready = float(model._handoff_ready_from_state(state[:, model.PITCH_IDX], state[:, model.ROOTZ_IDX])[0].item())
        front_contact_geom = float(model._front_contact_geom_from_state(state[:, model.PITCH_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.SUPPORT_IDX])[0].item())
        capture_commitment = float(model._capture_commitment(state[:, model.CAPTURE_IDX])[0].item())
        recovery_support = float(model._recovery_support_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.CAPTURE_IDX], state[:, model.ROOTZ_IDX])[0].item())
        rear_trap = float(model._rear_recovery_phase_from_state(state[:, model.PITCH_IDX], state[:, model.ROOTZ_IDX])[0].item())
        standup_window = float(model._standup_window_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.CAPTURE_IDX], state[:, model.ROOTZ_IDX])[0].item())
        arm_ready = float(model._arm_ready_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.CAPTURE_IDX], state[:, model.ROOTZ_IDX])[0].item())
        handoff_window = float(model._handoff_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
        precapture_phase = float(model._precapture_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
        capture_phase = float(model._capture_crawl_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
        standup_phase = float(model._standup_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
        crawl_to_standup_bias = float(model._crawl_to_standup_bias_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
        release_phase = float(model._release_phase_from_state(state[:, model.PITCH_IDX], state[:, model.SUPPORT_IDX], state[:, model.COMP_IDX], state[:, model.ROOTZ_IDX], state[:, model.CAPTURE_IDX])[0].item())
        release_ready = release_phase * min(max(compression / 0.35, 0.0), 1.0)
        unsupported_low = body_low * (1.0 - capture_commitment)
    feats = {
        'pitch_struct': pitch,
        'height': height,
        'vx': vx,
        'vz': vz,
        'compression': compression,
        'support': support,
        'front_contact_geom': front_contact_geom,
        'capture_memory': capture_memory,
        'body_low': body_low,
        'body_high': body_high,
        'forward_lean': forward_lean,
        'recovered_upright': recovered_upright,
        'handoff_ready': handoff_ready,
        'capture_commitment': capture_commitment,
        'recovery_support': recovery_support,
        'rear_trap': rear_trap,
        'standup_window': standup_window,
        'arm_ready': arm_ready,
        'release_ready': release_ready,
        'handoff_window': handoff_window,
        'rear_recovery_phase': rear_trap,
        'precapture_phase': precapture_phase,
        'capture_phase': capture_phase,
        'crawl_to_standup_bias': crawl_to_standup_bias,
        'standup_phase': standup_phase,
        'release_phase': release_phase,
        'handoff_phase': handoff_window,
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
            [0.12, 0.10, 0.85, 0.00, 0.38, 0.00, 0.82, -0.50, 0.18],
            dtype=np.float32,
        )
    if name == 'rear_trap':
        return np.array(
            [0.10, -0.05, -2.80, 0.00, 0.22, 0.00, 0.85, -0.78, 0.02],
            dtype=np.float32,
        )
    if name == 'uncaptured_forward_lean':
        return np.array(
            [0.16, 0.20, 0.58, 0.00, 0.28, 0.00, 0.88, -0.46, 0.06],
            dtype=np.float32,
        )
    if name == 'release_ready_corridor':
        return np.array(
            [0.25, 0.30, 0.95, 0.10, 0.42, 0.00, 0.88, -0.50, 0.82],
            dtype=np.float32,
        )
    if name == 'mature_capture_crawl':
        return np.array(
            [0.18, 0.05, 1.30, 0.00, 0.34, 0.00, 0.88, -0.74, 0.88],
            dtype=np.float32,
        )
    raise KeyError(name)


def transition_probe(world_model: SDWorldModel, model: DMControlHopperSDModel, struct: np.ndarray, action: np.ndarray) -> dict:
    state = torch.tensor(struct, dtype=torch.float32).unsqueeze(0)
    act = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_state_t = predict_next_struct(world_model, state, act)
        next_state = next_state_t[0].detach().cpu().numpy()
        plan_reward = float(model.predict_plan_reward(state, act, next_state_t).item())
    prev = semantic_features(model, struct)
    nxt = semantic_features(model, next_state)
    return {
        'action': action.tolist(),
        'prev': prev,
        'next': nxt,
        'delta_rootz': nxt['rootz'] - prev['rootz'],
        'delta_abs_pitch_to_neutral': abs(prev['pitch_struct']) - abs(nxt['pitch_struct']),
        'delta_recovered_upright': nxt['recovered_upright'] - prev['recovered_upright'],
        'delta_capture_memory': nxt['capture_memory'] - prev['capture_memory'],
        'delta_capture_commitment': nxt['capture_commitment'] - prev['capture_commitment'],
        'delta_release_ready': nxt['release_ready'] - prev['release_ready'],
        'delta_handoff_window': nxt['handoff_window'] - prev['handoff_window'],
        'delta_standup_window': nxt['standup_window'] - prev['standup_window'],
        'delta_vz': nxt['vz'] - prev['vz'],
        'plan_reward': plan_reward,
    }


def transition_rollout_probe(world_model: SDWorldModel, model: DMControlHopperSDModel, struct: np.ndarray, actions: list[np.ndarray]) -> dict:
    state = torch.tensor(struct, dtype=torch.float32).unsqueeze(0)
    prev = semantic_features(model, struct)
    steps = []
    total_plan_reward = 0.0
    with torch.no_grad():
        for action in actions:
            act = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            next_state = predict_next_struct(world_model, state, act)
            total_plan_reward += float(model.predict_plan_reward(state, act, next_state).item())
            state = next_state
            cur = semantic_features(model, state[0].detach().cpu().numpy())
            steps.append(cur)
    final = steps[-1]
    return {
        'actions': [a.tolist() for a in actions],
        'prev': prev,
        'steps': steps,
        'final': final,
        'delta_rootz': final['rootz'] - prev['rootz'],
        'delta_capture_memory': final['capture_memory'] - prev['capture_memory'],
        'delta_handoff_window': final['handoff_window'] - prev['handoff_window'],
        'delta_recovered_upright': final['recovered_upright'] - prev['recovered_upright'],
        'delta_recovery_support': final['recovery_support'] - prev['recovery_support'],
        'total_plan_reward': total_plan_reward,
    }


def transition_and_reward_checks(world_model: SDWorldModel, model: DMControlHopperSDModel, candidates: dict[str, list[dict]]):
    action_bank = {
        'hold_crawl': np.array([0.2, 0.8, -0.8, -0.8], dtype=np.float32),
        'release_handoff': np.array([0.6, -1.0, 1.0, 1.0], dtype=np.float32),
        'follow_handoff': np.array([0.3, -0.4, 0.6, 0.6], dtype=np.float32),
        'standup_nudge': np.array([0.30, -0.20, 0.35, 0.35], dtype=np.float32),
        'neutral': np.zeros(4, dtype=np.float32),
        'rear_worsen': np.array([-0.6, 0.8, -0.8, -0.8], dtype=np.float32),
        'rear_escape': np.array([0.6, -0.8, 0.8, 0.8], dtype=np.float32),
    }

    forward_source = canonical_struct(model, 'recoverable_front_anchor').tolist()
    rear_source = candidates['rear_nose_support'][0]['struct'] if candidates['rear_nose_support'] else canonical_struct(model, 'rear_trap').tolist()
    uncaptured_source = canonical_struct(model, 'uncaptured_forward_lean').tolist()
    release_ready_source = canonical_struct(model, 'release_ready_corridor').tolist()
    mature_capture_source = canonical_struct(model, 'mature_capture_crawl').tolist()

    forward_probes = {
        name: transition_probe(world_model, model, np.array(forward_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'release_handoff', 'neutral')
    }
    forward_rollouts = {
        'hold_then_hold': transition_rollout_probe(
            world_model,
            model,
            np.array(forward_source, dtype=np.float32),
            [action_bank['hold_crawl'], action_bank['hold_crawl']],
        ),
        'release_then_follow': transition_rollout_probe(
            world_model,
            model,
            np.array(forward_source, dtype=np.float32),
            [action_bank['release_handoff'], action_bank['follow_handoff']],
        ),
    }
    rear_probes = {
        name: transition_probe(world_model, model, np.array(rear_source, dtype=np.float32), action_bank[name])
        for name in ('rear_worsen', 'rear_escape', 'neutral')
    }
    uncaptured_probes = {
        name: transition_probe(world_model, model, np.array(uncaptured_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'release_handoff', 'neutral')
    }
    release_ready_probes = {
        name: transition_probe(world_model, model, np.array(release_ready_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'release_handoff', 'neutral')
    }
    mature_capture_probes = {
        name: transition_probe(world_model, model, np.array(mature_capture_source, dtype=np.float32), action_bank[name])
        for name in ('hold_crawl', 'standup_nudge', 'neutral')
    }
    release_ready_rollouts = {
        'hold_then_hold': transition_rollout_probe(
            world_model,
            model,
            np.array(release_ready_source, dtype=np.float32),
            [action_bank['hold_crawl'], action_bank['hold_crawl']],
        ),
        'release_then_follow': transition_rollout_probe(
            world_model,
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
    mature_capture_hold = mature_capture_probes['hold_crawl']
    mature_capture_standup = mature_capture_probes['standup_nudge']
    release_ready_hold_rollout = release_ready_rollouts['hold_then_hold']
    release_ready_release_rollout = release_ready_rollouts['release_then_follow']

    capture_hold_pass = bool(
        hold['plan_reward'] >= release['plan_reward'] - 1e-4
        and hold['delta_capture_commitment'] >= release['delta_capture_commitment'] - 1e-4
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
        and release_ready_release['delta_capture_memory'] <= release_ready_hold['delta_capture_memory'] + 1e-4
    )
    release_ready_continuity_pass = bool(
        release_ready_release_rollout['delta_rootz'] > release_ready_hold_rollout['delta_rootz'] + 1e-4
        and release_ready_release_rollout['delta_recovered_upright'] > release_ready_hold_rollout['delta_recovered_upright'] + 1e-4
        and release_ready_release_rollout['delta_handoff_window'] >= release_ready_hold_rollout['delta_handoff_window'] - 1e-2
        and release_ready_release_rollout['final']['handoff_window'] >= 0.015
        and release_ready_release_rollout['final']['arm_ready'] >= 0.01
        and release_ready_release_rollout['total_plan_reward'] > release_ready_hold_rollout['total_plan_reward'] + 1e-4
    )
    rear_transition_pass = bool(
        rear_worsen['next']['capture_memory'] <= 0.10
        and rear_worsen['next']['rear_trap'] >= 0.70
        and rear_escape['plan_reward'] > rear_worsen['plan_reward'] + 1e-4
    )
    uncaptured_phase_order_pass = bool(
        uncaptured_release['plan_reward'] <= uncaptured_hold['plan_reward'] + 1e-4
        and uncaptured_release['next']['capture_commitment'] <= uncaptured_hold['next']['capture_commitment'] + 1e-4
        and uncaptured_release['next']['standup_window'] <= uncaptured_hold['next']['standup_window'] + 1e-4
        and uncaptured_release['next']['release_ready'] <= 0.15
    )
    mature_capture_progress_pass = bool(
        mature_capture_standup['plan_reward'] > mature_capture_hold['plan_reward'] + 1e-4
        and mature_capture_standup['next']['capture_commitment'] >= mature_capture_hold['next']['capture_commitment'] - 0.05
        and (
            mature_capture_standup['delta_rootz'] > mature_capture_hold['delta_rootz'] + 1e-4
            or mature_capture_standup['delta_standup_window'] > mature_capture_hold['delta_standup_window'] + 1e-4
            or mature_capture_standup['delta_recovered_upright'] > mature_capture_hold['delta_recovered_upright'] + 1e-4
        )
    )

    return {
        'recoverable_front_anchor': {
            'source': 'canonical',
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
        'mature_capture_crawl': {
            'source': 'canonical',
            'probes': mature_capture_probes,
            'checks': {
                'mature_capture_prefers_standup_nudge_over_hold': mature_capture_progress_pass,
            },
            'pass': bool(mature_capture_progress_pass),
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


def classify_pitch_rate_failure(example: dict) -> str:
    pitch = float(example['pitch'])
    vz = float(example['vz'])
    support = float(example['support'])
    capture = float(example['capture_memory'])
    compression = float(example['compression'])
    rear_trap = float(example.get('rear_trap', 0.0))
    front_contact = float(example.get('front_contact_geom', 0.0))
    release_phase = float(example.get('release_phase', 0.0))
    standup_phase = float(example.get('standup_phase', 0.0))

    if rear_trap >= 0.70:
        return 'rear_trap_tumble'
    if capture >= 0.80 and pitch >= 2.0 and (release_phase >= 0.20 or standup_phase >= 0.40):
        return 'captured_release_turn'
    if front_contact >= 0.40 and support <= 0.35 and pitch >= 0.30 and vz <= -0.30:
        return 'front_capture_tumble'
    if support >= 0.75 and compression <= 0.40 and pitch >= 0.0 and vz <= -0.50:
        return 'high_support_rebound'
    if support >= 0.45 and capture <= 0.40 and vz <= -1.00:
        return 'uncaptured_landing_recoil'
    return 'mixed_or_unclear'


def transition_alignment_checks(world_model: SDWorldModel, model: DMControlHopperSDModel, rows: list[dict], args) -> dict:
    labels = [
        'vx',
        'vz',
        'pitch',
        'pitch_rate',
        'compression',
        'compression_rate',
        'support',
        'rootz',
    ]
    one_step_rmses = []
    three_step_rmses = []
    one_step_errors = []
    three_step_errors = []
    one_step_examples = []

    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row['episode']), []).append(row)

    for ep_rows in by_episode.values():
        ep_rows = sorted(ep_rows, key=lambda r: int(r['step']))
        for row in ep_rows:
            if 'action' not in row or 'next_struct' not in row:
                continue
            state = torch.tensor(row['struct'], dtype=torch.float32).unsqueeze(0)
            action = torch.tensor(row['action'], dtype=torch.float32).unsqueeze(0)
            true_next = torch.tensor(row['next_struct'], dtype=torch.float32).unsqueeze(0)
            pred_next = predict_next_struct(world_model, state, action)
            obs_dim = model.OBSERVABLE_DIM
            err = (pred_next[:, :obs_dim] - true_next[:, :obs_dim]).squeeze(0).detach().cpu().numpy()
            one_step_errors.append(err)
            one_step_examples.append({
                'episode': int(row['episode']),
                'step': int(row['step']),
                'pitch': float(row['pitch_struct']),
                'vz': float(row['vz']),
                'support': float(row['support']),
                'capture_memory': float(row['capture_memory']),
                'compression': float(row['compression']),
                'rear_trap': float(row['rear_trap']),
                'front_contact_geom': float(row['front_contact_geom']),
                'standup_phase': float(row['standup_phase']),
                'release_phase': float(row['release_phase']),
                'handoff_phase': float(row['handoff_phase']),
                'action': row['action'],
                'pred': pred_next[:, :obs_dim].squeeze(0).detach().cpu().tolist(),
                'target': true_next[:, :obs_dim].squeeze(0).detach().cpu().tolist(),
                'err': err.tolist(),
            })
            rmse = torch.sqrt(torch.mean((pred_next[:, :obs_dim] - true_next[:, :obs_dim]) ** 2)).item()
            one_step_rmses.append(rmse)

        for i in range(max(0, len(ep_rows) - 3)):
            window = ep_rows[i:i+4]
            if any('action' not in r or 'next_struct' not in r for r in window[:3]):
                continue
            state = torch.tensor(window[0]['struct'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = state
                for row in window[:3]:
                    action = torch.tensor(row['action'], dtype=torch.float32).unsqueeze(0)
                    pred = predict_next_struct(world_model, pred, action)
            true_final = torch.tensor(window[2]['next_struct'], dtype=torch.float32).unsqueeze(0)
            obs_dim = model.OBSERVABLE_DIM
            err = (pred[:, :obs_dim] - true_final[:, :obs_dim]).squeeze(0).detach().cpu().numpy()
            three_step_errors.append(err)
            rmse = torch.sqrt(torch.mean((pred[:, :obs_dim] - true_final[:, :obs_dim]) ** 2)).item()
            three_step_rmses.append(rmse)

    one_step_mean = float(np.mean(one_step_rmses)) if one_step_rmses else float('inf')
    three_step_mean = float(np.mean(three_step_rmses)) if three_step_rmses else float('inf')
    one_step_per_dim = {}
    three_step_per_dim = {}
    worst_examples = {}
    priority_dims = []
    pitch_rate_mode_counts = {}
    if one_step_errors:
        arr = np.stack(one_step_errors)
        for i, label in enumerate(labels[:arr.shape[1]]):
            col = arr[:, i]
            one_step_per_dim[label] = {
                'rmse': float(np.sqrt(np.mean(col ** 2))),
                'mae': float(np.mean(np.abs(col))),
                'bias': float(np.mean(col)),
            }
            ranked = sorted(
                (
                    {
                        **one_step_examples[j],
                        'abs_err': float(abs(col[j])),
                        'signed_err': float(col[j]),
                        'pred_value': float(one_step_examples[j]['pred'][i]),
                        'target_value': float(one_step_examples[j]['target'][i]),
                    }
                    for j in range(len(one_step_examples))
                ),
                key=lambda x: x['abs_err'],
                reverse=True,
            )[:5]
            worst_examples[label] = ranked
        priority_dims = [
            name for name, _ in sorted(
                ((name, stats['rmse']) for name, stats in one_step_per_dim.items()),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        for example in worst_examples.get('pitch_rate', []):
            mode = classify_pitch_rate_failure(example)
            example['mode'] = mode
            pitch_rate_mode_counts[mode] = pitch_rate_mode_counts.get(mode, 0) + 1
    if three_step_errors:
        arr = np.stack(three_step_errors)
        for i, label in enumerate(labels[:arr.shape[1]]):
            col = arr[:, i]
            three_step_per_dim[label] = {
                'rmse': float(np.sqrt(np.mean(col ** 2))),
                'mae': float(np.mean(np.abs(col))),
                'bias': float(np.mean(col)),
            }
    return {
        'one_step_observable_rmse_mean': one_step_mean,
        'three_step_observable_rmse_mean': three_step_mean,
        'one_step_per_dim': one_step_per_dim,
        'three_step_per_dim': three_step_per_dim,
        'priority_dims': priority_dims,
        'one_step_worst_examples': worst_examples,
        'pitch_rate_failure_modes': pitch_rate_mode_counts,
        'checks': {
            'one_step_rmse_ok': bool(one_step_mean <= args.transition_one_step_rmse_max),
            'three_step_rmse_ok': bool(three_step_mean <= args.transition_three_step_rmse_max),
        },
        'pass': bool(one_step_mean <= args.transition_one_step_rmse_max and three_step_mean <= args.transition_three_step_rmse_max),
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
    if row['pitch_raw'] > 1.0 and row['front_support'] > 0.5 and row['capture_memory'] > 0.3:
        return 'forward_anchor_like'
    return None


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg, env = build_cfg(args.seed)
    set_seed(cfg.seed)
    world_model = SDWorldModel(cfg).eval()
    model = world_model._dynamics.eval()
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
                'obs': obs.detach().cpu().numpy().tolist(),
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
            obs_next, _, done, _ = env.step(action)
            next_struct = model.encode_observation(obs_next.unsqueeze(0))[0].detach().cpu().numpy()
            row['action'] = action_np.tolist()
            row['next_struct'] = next_struct.tolist()
            obs = obs_next
            if bool(done):
                break

    checks = {}
    rear_rows = candidates['rear_nose_support']
    if rear_rows:
        checks['rear_nose_support'] = {
            'covered': True,
            'count': len(rear_rows),
            'mean_capture_memory': float(np.mean([r['capture_memory'] for r in rear_rows])),
            'mean_rear_trap': float(np.mean([r['rear_trap'] for r in rear_rows])),
            'pass': bool(np.mean([r['capture_memory'] for r in rear_rows]) <= args.rear_capture_max and np.mean([r['rear_trap'] for r in rear_rows]) >= args.rear_rear_trap_min),
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
            'mean_capture_memory': float(np.mean([r['capture_memory'] for r in forward_rows])),
            'mean_rear_trap': float(np.mean([r['rear_trap'] for r in forward_rows])),
            'mean_standup_window': float(np.mean([r['standup_window'] for r in forward_rows])),
            'pass': bool(
                np.mean([r['capture_memory'] for r in forward_rows]) >= 0.3
                and np.mean([r['rear_trap'] for r in forward_rows]) <= 0.2
                and np.mean([r['standup_window'] for r in forward_rows]) >= 0.08
            ),
        }
    else:
        checks['forward_anchor_like'] = {'covered': False, 'count': 0, 'pass': True}

    covered_check_names = [name for name, info in checks.items() if info.get('covered', False)]
    uncovered_check_names = [name for name, info in checks.items() if not info.get('covered', False)]
    covered_pass = all(checks[name]['pass'] for name in covered_check_names)
    transition_checks = transition_and_reward_checks(world_model, model, candidates)
    alignment_check = transition_alignment_checks(world_model, model, rows, args)
    focus_transition_names = ['recoverable_front_anchor', 'rear_trap', 'mature_capture_crawl', 'release_ready_corridor', 'phase_dependency', 'transition_alignment']
    transition_pass = all(transition_checks[name]['pass'] for name in focus_transition_names if name in transition_checks) and alignment_check['pass']

    report = {
        'episodes': args.episodes,
        'steps_per_episode': args.steps_per_episode,
        'total_samples': len(rows),
        'pitch_abs_err_mean': float(np.mean(pitch_abs_err)) if pitch_abs_err else None,
        'pitch_abs_err_max': float(np.max(pitch_abs_err)) if pitch_abs_err else None,
        'checks': checks,
        'transition_reward_checks': transition_checks,
        'transition_alignment': alignment_check,
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
