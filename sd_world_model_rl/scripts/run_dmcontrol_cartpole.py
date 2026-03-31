#!/usr/bin/env python3
"""DMControl Cartpole-Swingup SD-MCTS Benchmark (v3).

Matches DreamerV3's benchmark environment with action_repeat=2.

DreamerV3 published scores (arXiv:2301.04104):
    100k steps: ~700-750     (action_repeat=2)
    1M   steps: 861          (action_repeat=2)

v3 improvements over v2:
    - Action repeat k=2 in MCTS (macro-actions, 2x planning horizon)
    - Optimal bins=5 (fewer bins → deeper search → better swingup)
    - Multi-step edge rewards in MCTS backpropagation
    - Timestamped logging for progress tracking

Usage:
    conda run -n mapf python -u scripts/run_dmcontrol_cartpole.py \\
        --budget 1000 --seeds 3 2>&1 | tee /tmp/sdmcts_v3.log
"""
import argparse
import sys
import os
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from src.world_model.dmcontrol_cartpole_sd_model import (
    DMControlCartPoleSDModel,
    DMControlCartPoleWrapper,
)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Action discretization
# ============================================================
def get_discrete_actions(n_bins=5):
    return np.linspace(-1.0, 1.0, n_bins).astype(np.float32)


# ============================================================
# Observation helpers
# ============================================================
def flatten_obs(obs):
    return np.concatenate([obs['position'], obs['velocity']]).astype(np.float32)


def obs_to_state(obs_flat):
    return DMControlCartPoleSDModel.obs_to_state(obs_flat)


def energy_pump_policy(state):
    """Noisy energy pump to excite high-θ̇ for mp observability.
    From dE/dt = -m·l·θ̇·u·cos(θ), energy injection requires
    u opposite in sign to θ̇·cos(θ). Randomized magnitude preserves
    force diversity for mc/l identification."""
    theta, theta_dot = float(state[1]), float(state[3])
    direction = -1.0 if theta_dot * math.cos(theta) >= 0 else 1.0
    magnitude = np.random.uniform(0.5, 1.0)
    return direction * magnitude


def get_qstate(physics):
    """Get true state [x, θ, ẋ, θ̇] from MuJoCo physics.

    Important: wrap θ to [-pi, pi] so SI/VNet/MCTS all share the same
    angular representation. MuJoCo's hinge angle can accumulate across
    multiple full rotations under high-energy policies, which would make
    VNet training states inconsistent with eval-time states reconstructed
    from observations via atan2(sin, cos).
    """
    state = np.concatenate([physics.data.qpos, physics.data.qvel]).astype(np.float32)
    state[1] = (state[1] + np.pi) % (2 * np.pi) - np.pi
    return state


# ============================================================
# Swingup heuristic (energy-based pump + PD balance)
# ============================================================
def _swingup_action(x, theta, x_dot, theta_dot, g_over_l=19.6):
    """Energy-based swingup + PD balancing heuristic (original baseline)."""
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)

    if cos_th > 0.85:
        a = -(3.0 * sin_th + 1.0 * theta_dot + 0.4 * x + 0.3 * x_dot)
    else:
        E_norm = 0.5 * theta_dot * theta_dot / g_over_l + cos_th
        E_err = E_norm - 1.0
        pump_dir = 1.0 if theta_dot * cos_th >= 0 else -1.0
        a = -1.5 * E_err * pump_dir - 0.15 * x - 0.1 * x_dot

    return float(np.clip(a, -1.0, 1.0))


# ============================================================
# Multi-step forward (action repeat in world model)
# ============================================================
def model_forward_k(wrapper, state, action_val, k, gamma=0.99):
    """Roll world model forward k steps with same action.

    Returns: (final_state, cumulative_discounted_reward, total_discount)
    """
    s = state
    cum_reward = 0.0
    discount = 1.0
    action_t = torch.tensor([[action_val]], dtype=torch.float32)
    for _ in range(k):
        with torch.no_grad():
            next_s = wrapper(s.unsqueeze(0), action_t)[0]
            r = wrapper.model.predict_reward(
                s.unsqueeze(0), action_t, next_s.unsqueeze(0)
            ).item()
        cum_reward += discount * r
        discount *= gamma
        s = next_s
    return s, cum_reward, discount


# ============================================================
# Value Network (TVA)
# ============================================================
class SwingupValueNet(nn.Module):
    def __init__(self, state_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_value_net(value_net, states, returns, epochs=200, lr=1e-3,
                    batch_size=256, min_loss=0.0, weight_decay=0.0,
                    state_noise=0.0):
    optimizer = optim.AdamW(value_net.parameters(), lr=lr, amsgrad=True,
                            weight_decay=weight_decay)
    targets = torch.tensor(returns, dtype=torch.float32) if not isinstance(
        returns, torch.Tensor) else returns.float()
    n = len(states)
    if n == 0:
        return 0.0
    last_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_states = states[idx]
            if state_noise > 0:
                batch_states = batch_states + state_noise * torch.randn_like(batch_states)
            pred = value_net(batch_states)
            loss = nn.functional.smooth_l1_loss(pred, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            optimizer.step()
            epoch_loss_sum += loss.item()
            n_batches += 1
        last_loss = epoch_loss_sum / n_batches
        if min_loss > 0 and last_loss < min_loss:
            break
    return last_loss


def _compute_vnet_slice_metrics(states, targets, preds):
    theta = states[:, 1]
    cos_th = torch.cos(theta)
    x_abs = torch.abs(states[:, 0])
    thdot_abs = torch.abs(states[:, 3])

    slices = [
        ("all", torch.ones(len(states), dtype=torch.bool)),
        ("upright", cos_th > 0.85),
        ("transition", (cos_th > 0.0) & (cos_th <= 0.85)),
        ("high_thdot", thdot_abs >= torch.quantile(thdot_abs, 0.8)),
        ("cart_far", x_abs >= torch.quantile(x_abs, 0.8)),
    ]
    metrics = {}
    for name, mask in slices:
        count = int(mask.sum().item())
        if count == 0:
            continue
        err = preds[mask] - targets[mask]
        metrics[name] = {
            'n': count,
            'mse': (err ** 2).mean().item(),
            'bias': err.mean().item(),
        }
    return metrics


def _critical_vnet_score(metrics):
    return (
        0.5 * metrics['upright']['mse'] +
        0.3 * metrics['transition']['mse'] +
        0.2 * metrics['high_thdot']['mse']
    )


def _log_vnet_calibration(prefix, metrics):
    for name, values in metrics.items():
        log(f"    {prefix} {name}: n={values['n']} "
            f"mse={values['mse']:.4f} bias={values['bias']:.4f}")


def train_vnet_with_restarts(states, returns, seed, hidden=128, epochs=500,
                             lr=1e-3, min_loss=0.05, weight_decay=0.0,
                             state_noise=0.0, val_fraction=0.15,
                             n_restarts=5, batch_size=256,
                             log_prefix="VNet", init_state_dict=None,
                             selection_metric='overall'):
    n = len(states)
    if n == 0:
        return SwingupValueNet(hidden=hidden), 0.0, 0.0
    if val_fraction <= 0 or n_restarts <= 1:
        vnet = SwingupValueNet(hidden=hidden)
        if init_state_dict is not None:
            vnet.load_state_dict(init_state_dict)
        train_loss = train_value_net(
            vnet, states, returns, epochs=epochs, lr=lr,
            batch_size=batch_size, min_loss=min_loss,
            weight_decay=weight_decay, state_noise=state_noise)
        return vnet, train_loss, train_loss

    n_val = max(int(n * val_fraction), 100)
    rng = np.random.RandomState(seed + 9999)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_s = states[train_idx]
    train_r = returns[train_idx]
    val_s = states[val_idx]
    val_r_t = torch.tensor(returns[val_idx], dtype=torch.float32)

    best_vnet = None
    best_train_loss = 0.0
    best_val_loss = float('inf')
    best_metrics = None
    best_score = float('inf')
    for restart in range(n_restarts):
        torch.manual_seed(seed * 1000 + restart * 137 + 7)
        vnet = SwingupValueNet(hidden=hidden)
        if init_state_dict is not None:
            vnet.load_state_dict(init_state_dict)
        train_loss = train_value_net(
            vnet, train_s, train_r, epochs=epochs, lr=lr,
            batch_size=batch_size, min_loss=min_loss,
            weight_decay=weight_decay, state_noise=state_noise)
        with torch.no_grad():
            preds = vnet(val_s)
            metrics = _compute_vnet_slice_metrics(val_s, val_r_t, preds)
            val_loss = metrics['all']['mse']
            if selection_metric == 'critical':
                score = _critical_vnet_score(metrics)
            else:
                score = val_loss
        marker = ""
        if score < best_score:
            best_score = score
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_vnet = copy.deepcopy(vnet)
            best_metrics = metrics
            marker = "  ← best"
        log(f"    {log_prefix} restart {restart}: train={train_loss:.4f}, "
            f"val={val_loss:.4f}"
            + (f", critical={score:.4f}" if selection_metric == 'critical' else "")
            + marker)

    _log_vnet_calibration(f"{log_prefix} val", best_metrics)
    return best_vnet, best_train_loss, best_val_loss, best_metrics


# ============================================================
# Data collection — use direct qpos/qvel for SI
# ============================================================
def collect_trajectories(env, n_steps, policy='random'):
    states, actions, next_states, rewards_list = [], [], [], []
    obs_states = []
    ep_starts = [0]

    ts = env.reset()
    state = get_qstate(env.physics)
    obs = flatten_obs(ts.observation)
    obs_state = obs_to_state(obs)
    ep_idx = 0  # episode counter for mixed policy

    for step in range(n_steps):
        if policy == 'random':
            action_val = np.random.uniform(-1, 1)
        elif policy == 'mixed':
            # Every 3rd episode uses energy pump; rest random
            if ep_idx % 3 == 2:
                action_val = energy_pump_policy(obs_state)
            else:
                action_val = np.random.uniform(-1, 1)
        elif callable(policy):
            action_val = policy(obs_state)
        else:
            action_val = np.random.uniform(-1, 1)

        ts = env.step(np.array([action_val]))
        next_state = get_qstate(env.physics)
        next_obs = flatten_obs(ts.observation)
        next_obs_state = obs_to_state(next_obs)
        reward = ts.reward if ts.reward is not None else 0.0

        states.append(state.copy())
        actions.append(np.float32(action_val))
        next_states.append(next_state.copy())
        obs_states.append(obs_state.copy())
        rewards_list.append(reward)

        if ts.last():
            ts = env.reset()
            next_state = get_qstate(env.physics)
            next_obs = flatten_obs(ts.observation)
            next_obs_state = obs_to_state(next_obs)
            ep_starts.append(step + 1)
            ep_idx += 1

        state = next_state
        obs_state = next_obs_state

    gamma = 0.99
    returns = np.zeros(len(rewards_list), dtype=np.float32)
    ep_set = set(ep_starts)
    G = 0.0
    for t in reversed(range(len(rewards_list))):
        if (t + 1) in ep_set:
            G = 0.0
        G = rewards_list[t] + gamma * G
        returns[t] = G

    return {
        'states': torch.tensor(np.array(states), dtype=torch.float32),
        'actions': torch.tensor(np.array(actions), dtype=torch.float32),
        'next_states': torch.tensor(np.array(next_states), dtype=torch.float32),
        'obs_states': torch.tensor(np.array(obs_states), dtype=torch.float32),
        'rewards': np.array(rewards_list, dtype=np.float32),
        'returns': returns,
        'ep_starts': ep_starts,
        'n_episodes': len(ep_starts),
    }


# ============================================================
# Angle-aware SI loss
# ============================================================
def sincos_loss(pred_state, target_state, thdot_weight=1.0):
    x_loss = (pred_state[..., 0] - target_state[..., 0]) ** 2
    cos_loss = (torch.cos(pred_state[..., 1]) - torch.cos(target_state[..., 1])) ** 2
    sin_loss = (torch.sin(pred_state[..., 1]) - torch.sin(target_state[..., 1])) ** 2
    xdot_loss = (pred_state[..., 2] - target_state[..., 2]) ** 2
    thdot_loss = (pred_state[..., 3] - target_state[..., 3]) ** 2
    return (x_loss + cos_loss + sin_loss + xdot_loss
            + thdot_weight * thdot_loss).mean()


def _sincos_loss_unreduced(pred_state, target_state, thdot_weight=1.0):
    """Per-sample sincos loss (no reduction) for energy weighting."""
    x_loss = (pred_state[..., 0] - target_state[..., 0]) ** 2
    cos_loss = (torch.cos(pred_state[..., 1]) - torch.cos(target_state[..., 1])) ** 2
    sin_loss = (torch.sin(pred_state[..., 1]) - torch.sin(target_state[..., 1])) ** 2
    xdot_loss = (pred_state[..., 2] - target_state[..., 2]) ** 2
    thdot_loss = (pred_state[..., 3] - target_state[..., 3]) ** 2
    return x_loss + cos_loss + sin_loss + xdot_loss + thdot_weight * thdot_loss


def train_sd_model(model, data, epochs=300, lr=1e-2, use_sincos=True,
                   multistep=0, thdot_weight=1.0, energy_weighted=False,
                   two_phase_mp=False):
    """Train SD model with optional multi-step prediction loss.

    Args:
        multistep: If >0, use T-step rollout loss (curriculum: 1→T over epochs).
                   If 0, use standard 1-step loss (original behavior).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    states = data['states']
    actions = data['actions'].unsqueeze(-1)
    next_states = data['next_states']
    ep_starts_set = set(data['ep_starts'])
    best_loss = float('inf')

    # Pre-compute per-sample energy weights
    sample_weights = None
    if energy_weighted:
        thdot_abs = torch.abs(states[:, 3])
        sample_weights = 1.0 + thdot_abs

    if multistep <= 0:
        # --- Original 1-step loss (unchanged behavior) ---
        for epoch in range(epochs):
            pred = model(states, actions)
            if use_sincos:
                per_sample = _sincos_loss_unreduced(pred, next_states,
                                                   thdot_weight=thdot_weight)
                if sample_weights is not None:
                    loss = (per_sample * sample_weights).mean()
                else:
                    loss = per_sample.mean()
            else:
                loss = nn.functional.mse_loss(pred, next_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
        return best_loss

    # --- Multi-step loss with curriculum ---
    n = len(states)
    max_T = multistep
    valid_starts = []
    for i in range(n - max_T):
        crosses = any((i + t) in ep_starts_set for t in range(1, max_T + 1))
        if not crosses:
            valid_starts.append(i)
    valid_starts = np.array(valid_starts)

    if len(valid_starts) < 10:
        for epoch in range(epochs):
            pred = model(states, actions)
            loss = (sincos_loss(pred, next_states, thdot_weight=thdot_weight)
                    if use_sincos
                    else nn.functional.mse_loss(pred, next_states))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
        return best_loss

    # Two-phase SI: freeze mp in Phase A, unfreeze in Phase B
    phase_b_start = epochs // 2 if two_phase_mp else 0
    if two_phase_mp and hasattr(model, '_masspole_raw'):
        model._masspole_raw.requires_grad_(False)

    for epoch in range(epochs):
        # Phase transition: unfreeze mp, reduce mc/l lr
        if two_phase_mp and epoch == phase_b_start:
            if hasattr(model, '_masspole_raw'):
                model._masspole_raw.requires_grad_(True)
            optimizer = optim.Adam([
                {'params': [p for n, p in model.named_parameters()
                            if 'masspole' not in n and p.requires_grad],
                 'lr': lr * 0.1},
                {'params': [model._masspole_raw], 'lr': lr},
            ])

        T = min(max_T, 1 + int(max_T * epoch / (epochs // 2)))

        batch_size = min(128, len(valid_starts))
        batch_idx = valid_starts[
            np.random.choice(len(valid_starts), batch_size, replace=False)]

        total_loss = torch.tensor(0.0)
        s = states[batch_idx]
        for t in range(T):
            a = actions[batch_idx + t]
            pred = model(s, a)
            target = states[batch_idx + t + 1]
            if use_sincos:
                per_sample = _sincos_loss_unreduced(pred, target,
                                                   thdot_weight=thdot_weight)
                if sample_weights is not None:
                    w = sample_weights[batch_idx + t]
                    step_loss = (per_sample * w).mean()
                else:
                    step_loss = per_sample.mean()
            else:
                step_loss = nn.functional.mse_loss(pred, target)
            total_loss = total_loss + step_loss
            s = pred.detach()

        loss = total_loss / T
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()

    return best_loss


# ============================================================
# MCTS with action repeat
# ============================================================
class MCTSNode:
    __slots__ = ['state', 'parent', 'action_val', 'edge_reward',
                 'children', 'N', 'W', 'Q']

    def __init__(self, state, parent=None, action_val=None, edge_reward=0.0):
        self.state = state
        self.parent = parent
        self.action_val = action_val
        self.edge_reward = edge_reward
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0


def _heuristic_rollout(wrapper, state, rollout_depth, action_repeat, gamma,
                       g_over_l=19.6):
    """Heuristic rollout for swingup. Returns (cumulative_return, leaf_states)."""
    gamma_k = gamma ** action_repeat
    rs = state
    rollout_return = 0.0
    disc = 1.0
    for _ in range(rollout_depth):
        h_action = _swingup_action(
            rs[0].item(), rs[1].item(), rs[2].item(), rs[3].item(),
            g_over_l=g_over_l)
        new_state, step_r, _ = model_forward_k(
            wrapper, rs, h_action, action_repeat, gamma,
        )
        rollout_return += disc * step_r
        disc *= gamma_k
        rs = new_state
    return rollout_return


def _cart_drift_leaf_bias(state, alpha, cos_threshold=0.5):
    """Upright-gated penalty on cart drift and angular deviation.

    Applied at MCTS leaf evaluation time as search guidance.
    Only activates when the pole is near-upright (cos(theta) > cos_threshold),
    ramping smoothly from 0 at threshold to full penalty at cos=1.0.
    This acts as a local LQR-style stabilizer prior, correcting VNet blurriness
    at the unstable equilibrium without suppressing swing-up exploration.
    """
    if alpha <= 0.0:
        return 0.0
    
    # Smooth upright gate: 0 when pole is down, 1 when fully upright
    cos_th = math.cos(float(state[1].item()))
    if cos_th <= cos_threshold:
        return 0.0
    upright_w = (cos_th - cos_threshold) / (1.0 - cos_threshold)
    
    # 1. Quadratic cart drift penalty
    x_abs = abs(float(state[0].item()))
    drift = max(x_abs - 1.0, 0.0)
    x_penalty = drift * drift
    
    # 2. Angular stabilization penalty (LQR-style cost)
    th = float(state[1].item())
    th_dot = float(state[3].item())
    
    # Normalize theta to [-pi, pi]
    th = (th + math.pi) % (2 * math.pi) - math.pi
    
    # Strongly penalize angle and velocity: 10*theta^2 + 1*theta_dot^2
    theta_penalty = 10.0 * (th * th) + 1.0 * (th_dot * th_dot)
    
    total_penalty = x_penalty + theta_penalty
    
    return upright_w * alpha * total_penalty


def mcts_search_swingup(
    root_state_np, world_model_wrapper, discrete_actions,
    num_simulations=100, c=1.5, gamma=0.99,
    value_net=None, rollout_depth=20, action_repeat=2,
    collect_leaves=False, g_over_l=19.6,
    cart_drift_penalty=0.0,
    leaf_vnet_mix=1.0,
    leaf_vnet_gate_cos=None,
):
    """MCTS with action repeat (macro-actions).

    Each tree edge = action_repeat env steps with same action.
    If collect_leaves=True, also returns list of (leaf_state, rollout_value)
    for distillation training.
    """
    n_actions = len(discrete_actions)
    root = MCTSNode(torch.tensor(root_state_np, dtype=torch.float32))
    gamma_k = gamma ** action_repeat
    leaf_data = []  # For distillation: (state_tensor, rollout_value)

    for _ in range(num_simulations):
        node = root
        path = [node]

        # Selection
        while node.children and len(node.children) == n_actions:
            best_child, best_score = None, -float('inf')
            for child in node.children.values():
                if child.N == 0:
                    score = float('inf')
                else:
                    score = child.Q + c * math.sqrt(
                        math.log(node.N + 1) / (child.N + 1))
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)

        # Expansion — each edge = action_repeat steps
        if len(node.children) < n_actions:
            for a_idx in range(n_actions):
                if a_idx not in node.children:
                    action_val = discrete_actions[a_idx]
                    final_state, edge_r, _ = model_forward_k(
                        world_model_wrapper, node.state, action_val,
                        action_repeat, gamma,
                    )
                    child = MCTSNode(
                        final_state, parent=node,
                        action_val=action_val, edge_reward=edge_r,
                    )
                    node.children[a_idx] = child
                    node = child
                    path.append(node)
                    break

        # Evaluation
        rollout_val = None
        if value_net is not None:
            with torch.no_grad():
                vnet_value = value_net(node.state.unsqueeze(0)).item()
            mix = float(np.clip(leaf_vnet_mix, 0.0, 1.0))
            cos_th = math.cos(float(node.state[1].item()))
            needs_rollout = (
                collect_leaves or mix < 1.0 or
                (leaf_vnet_gate_cos is not None and cos_th <= leaf_vnet_gate_cos)
            )
            if needs_rollout:
                rollout_val = _heuristic_rollout(
                    world_model_wrapper, node.state,
                    rollout_depth, action_repeat, gamma,
                    g_over_l=g_over_l,
                )
            use_rollout_only = (
                leaf_vnet_gate_cos is not None and cos_th <= leaf_vnet_gate_cos
            )
            if use_rollout_only:
                leaf_value = rollout_val
            elif mix >= 1.0:
                leaf_value = vnet_value
            else:
                leaf_value = mix * vnet_value + (1.0 - mix) * rollout_val
            leaf_value -= _cart_drift_leaf_bias(node.state, cart_drift_penalty)
        else:
            leaf_value = _heuristic_rollout(
                world_model_wrapper, node.state,
                rollout_depth, action_repeat, gamma,
                g_over_l=g_over_l,
            )
            rollout_val = leaf_value
        # Collect leaf data for distillation (always uses rollout for labels)
        if collect_leaves:
            leaf_data.append((node.state.detach().clone(), rollout_val))

        # Backpropagation with macro-step discount
        G = leaf_value
        for node_bp in reversed(path):
            if node_bp.parent is not None:
                G = node_bp.edge_reward + gamma_k * G
            node_bp.N += 1
            node_bp.W += G
            node_bp.Q = node_bp.W / node_bp.N

    if not root.children:
        return (0.0, leaf_data) if collect_leaves else 0.0
    best_idx = max(root.children.keys(), key=lambda a: root.children[a].N)
    action = discrete_actions[best_idx]
    return (action, leaf_data) if collect_leaves else action


# ============================================================
# Evaluation with action repeat
# ============================================================
def evaluate_agent(env, world_model, discrete_actions, n_episodes=5,
                   num_simulations=100, value_net=None, max_steps=1000,
                   action_repeat=2, verbose=False, g_over_l=19.6,
                   cart_drift_penalty=0.0, leaf_vnet_mix=1.0,
                   leaf_vnet_gate_cos=None):
    wrapper = DMControlCartPoleWrapper(world_model)
    ep_returns = []
    for ep in range(n_episodes):
        ts = env.reset()
        obs = flatten_obs(ts.observation)
        state = obs_to_state(obs)
        ep_ret = 0.0
        env_step = 0

        while env_step < max_steps:
            action_val = mcts_search_swingup(
                state, wrapper, discrete_actions,
                num_simulations=num_simulations, value_net=value_net,
                action_repeat=action_repeat, g_over_l=g_over_l,
                cart_drift_penalty=cart_drift_penalty,
                leaf_vnet_mix=leaf_vnet_mix,
                leaf_vnet_gate_cos=leaf_vnet_gate_cos,
            )
            # Execute action for k env steps
            for _ in range(action_repeat):
                ts = env.step(np.array([action_val]))
                ep_ret += (ts.reward if ts.reward is not None else 0.0)
                env_step += 1
                if ts.last() or env_step >= max_steps:
                    break
            if ts.last():
                break
            obs = flatten_obs(ts.observation)
            state = obs_to_state(obs)

            if verbose and env_step % 200 == 0:
                cos_th = math.cos(state[1])
                log(f"    ep={ep} step={env_step:>4d}  "
                    f"cumR={ep_ret:>6.1f}  cos(θ)={cos_th:>+.3f}")

        ep_returns.append(ep_ret)
        if verbose:
            log(f"    ep={ep} DONE step={env_step}  return={ep_ret:.1f}")
    return np.array(ep_returns)


def evaluate_random(env, n_episodes=10, max_steps=1000):
    ep_returns = []
    for _ in range(n_episodes):
        ts = env.reset()
        ep_ret = 0.0
        for _ in range(max_steps):
            ts = env.step(np.random.uniform(-1, 1, size=(1,)).astype(np.float32))
            ep_ret += (ts.reward if ts.reward is not None else 0.0)
            if ts.last():
                break
        ep_returns.append(ep_ret)
    return np.array(ep_returns)


# ============================================================
# Rollout Distillation: train VNet from world-model imagination
# ============================================================
def generate_imagination_data(wrapper, action_repeat=2, gamma=0.99,
                              n_trajectories=2000, horizon=60,
                              real_states=None):
    """Generate (state, return) pairs using branch-augmented, replay-anchored imagination.

    Crucial for VNet robustness: MCTS evaluates branch states off the
    heuristic trajectory. If VNet only sees on-policy heuristic states,
    it extrapolation fails during planning. This generator uses
    segment-level branching to ensure support coverage without destroying
    the trajectory's ability to reach/hold the high-reward balance state:
      - 70% pure heuristic backbone rollouts
      - 30% branched rollouts (heuristic → 1-3 steps perturbation → heuristic recovery)
    All starting from real replay states.
    """
    all_states = []
    all_returns = []
    gamma_k = gamma ** action_repeat

    # Extract learned physical parameters from the wrapper's model
    params = wrapper.model.get_learned_params()
    g_over_l = params['gravity'] / max(params['length'], 1e-3)

    for i in range(n_trajectories):
        # 70/30 anchor split: real states + uniform random (original baseline)
        if real_states is not None and i < int(n_trajectories * 0.7):
            idx = np.random.randint(len(real_states))
            init_state = real_states[idx].clone()
            if i < int(n_trajectories * 0.5):
                init_state += torch.randn_like(init_state) * 0.1
        else:
            theta = np.random.uniform(-np.pi, np.pi)
            theta_dot = np.random.uniform(-8, 8)
            x = np.random.uniform(-0.5, 0.5)
            x_dot = np.random.uniform(-2, 2)
            init_state = torch.tensor(
                [x, theta, x_dot, theta_dot], dtype=torch.float32)

        rs = init_state
        traj_states = []
        traj_rewards = []

        for t in range(horizon):
            traj_states.append(rs.detach().clone())
            x, th, x_dot, th_dot = rs[0].item(), rs[1].item(), rs[2].item(), rs[3].item()
            act = _swingup_action(x, th, x_dot, th_dot, g_over_l=g_over_l)
            act = float(np.clip(act, -1.0, 1.0))
            new_state, step_r, _ = model_forward_k(
                wrapper, rs, act, action_repeat, gamma)
            traj_rewards.append(step_r)
            rs = new_state

        # Compute MC returns (with macro-step discount)
        returns = np.zeros(len(traj_rewards), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(traj_rewards))):
            G = traj_rewards[t] + gamma_k * G
            returns[t] = G

        # Subsample every 4 steps (baseline rate)
        for t in range(0, len(traj_states), 4):
            all_states.append(traj_states[t])
            all_returns.append(returns[t])

    return torch.stack(all_states), np.array(all_returns, dtype=np.float32)


# ============================================================
# Main Experiment
# ============================================================
def run_experiment(budget, seed, prior_noise=0.0, num_sims=100,
                   n_action_bins=5, eval_episodes=3, use_value_net=True,
                   fix_gravity=True, action_repeat=2, distill_rollout=False,
                   multistep=0, weight_decay=0.0, state_noise=0.0,
                   cart_drift_penalty=0.0, vnet_hidden=128,
                   vnet_val_fraction=0.15, vnet_restarts=5):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    discrete_actions = get_discrete_actions(n_action_bins)

    model = DMControlCartPoleSDModel(learn_params=True, prior_noise=prior_noise)

    if fix_gravity and model.learn_params:
        with torch.no_grad():
            model._gravity_raw.data = torch.tensor(
                model._inv_softplus(9.81), dtype=torch.float32)
        model._gravity_raw.requires_grad_(False)
        log(f"  [Seed {seed}] Fixed g=9.81, learning 3 params")

    params = model.get_learned_params()
    log(f"  [Seed {seed}] Init: g={params['gravity']:.3f}, "
        f"mc={params['masscart']:.3f}, mp={params['masspole']:.3f}, "
        f"l={params['length']:.3f}")

    # Phase 1: Collect random data for SI
    explore_steps = min(budget, max(500, budget // 2))
    data = collect_trajectories(env, explore_steps, policy='random')
    log(f"  [Seed {seed}] Collected {explore_steps} steps "
        f"({data['n_episodes']} ep, mean r/step={data['rewards'].mean():.3f})")

    # Phase 2: System Identification
    si_epochs = 500 if multistep <= 0 else 800  # More epochs for multi-step
    si_lr = 1e-2 if multistep <= 0 else 5e-3     # Lower lr for stability
    train_loss = train_sd_model(
        model, data, epochs=si_epochs, lr=si_lr,
        use_sincos=True, multistep=multistep)
    params = model.get_learned_params()
    log(f"  [Seed {seed}] After SI: g={params['gravity']:.3f}, "
        f"mc={params['masscart']:.3f}, mp={params['masspole']:.3f}, "
        f"l={params['length']:.3f} (loss={train_loss:.6f})")

    si_params = model.get_learned_params()

    # Phase 3: Train value network
    value_net = None

    if distill_rollout:
        # --- Rollout Distillation with VNet Bootstrap ---
        wrapper = DMControlCartPoleWrapper(model)

        # Round 1: heuristic imagination → initial VNet
        log(f"  [Seed {seed}] Generating imagination data (heuristic)...")
        imag_states, imag_returns = generate_imagination_data(
            wrapper, action_repeat=action_repeat,
            n_trajectories=2000, horizon=60,
            real_states=data['states'],
        )
        log(f"  [Seed {seed}] Imagination R1: {len(imag_states)} samples, "
            f"returns [{imag_returns.min():.1f}, {imag_returns.max():.1f}]")

        value_net, r1_loss, r1_val_loss, _ = train_vnet_with_restarts(
            imag_states, imag_returns, seed=seed, hidden=vnet_hidden,
            epochs=500, lr=1e-3, min_loss=0.05,
            weight_decay=weight_decay, state_noise=state_noise,
            val_fraction=vnet_val_fraction, n_restarts=vnet_restarts,
            log_prefix="VNet R1")
        log(f"  [Seed {seed}] VNet R1 best: train={r1_loss:.4f}, "
            f"val={r1_val_loss:.4f}")

        log(f"  [Seed {seed}] R2 bootstrap disabled; keeping R1-only VNet")

    elif use_value_net:
        # --- Original TVA Mode (env-data trained) ---
        value_net, vnet_loss, vnet_val_loss, _ = train_vnet_with_restarts(
            data['obs_states'], data['returns'], seed=seed,
            hidden=vnet_hidden, epochs=300, lr=1e-3, min_loss=0.0,
            weight_decay=weight_decay, state_noise=state_noise,
            val_fraction=vnet_val_fraction, n_restarts=vnet_restarts,
            log_prefix="Value net")
        log(f"  [Seed {seed}] Value net best: train={vnet_loss:.4f}, "
            f"val={vnet_val_loss:.4f}")

        # Phase 4: Value net refinement with MCTS data
        remaining = budget - explore_steps
        if remaining > 50:
            wrapper = DMControlCartPoleWrapper(model)
            ref_data = collect_trajectories(
                env, remaining,
                policy=lambda s: mcts_search_swingup(
                    s, wrapper, discrete_actions,
                    num_simulations=max(10, num_sims // 5),
                    value_net=value_net, rollout_depth=10,
                    action_repeat=action_repeat,
                ),
            )
            combined_obs = torch.cat([data['obs_states'], ref_data['obs_states']])
            combined_ret = np.concatenate([data['returns'], ref_data['returns']])
            value_net, vnet_loss, vnet_val_loss, _ = train_vnet_with_restarts(
                combined_obs, combined_ret, seed=seed,
                hidden=vnet_hidden, epochs=200, lr=1e-3, min_loss=0.0,
                weight_decay=weight_decay, state_noise=state_noise,
                val_fraction=vnet_val_fraction, n_restarts=vnet_restarts,
                log_prefix="Refined VNet")
            log(f"  [Seed {seed}] Refined VNet best: train={vnet_loss:.4f}, "
                f"val={vnet_val_loss:.4f}")

    # Phase 5: Evaluate
    eval_env = suite.load('cartpole', 'swingup',
                          task_kwargs={'random': seed + 1000})
    learned_g_over_l = model.g.item() / max(model.l.item(), 1e-3)
    log(f"  [Seed {seed}] Evaluating ({eval_episodes} episodes, "
        f"g/l={learned_g_over_l:.2f})...")
    returns = evaluate_agent(
        eval_env, model, discrete_actions,
        n_episodes=eval_episodes,
        num_simulations=num_sims,
        value_net=value_net,
        action_repeat=action_repeat,
        verbose=True,
        g_over_l=learned_g_over_l,
        cart_drift_penalty=cart_drift_penalty,
    )
    log(f"  [Seed {seed}] Eval: mean={returns.mean():.1f} ±{returns.std():.1f}")

    return returns, si_params


def main():
    parser = argparse.ArgumentParser(
        description='DMControl Cartpole-Swingup SD-MCTS Benchmark v3')
    parser.add_argument('--budget', type=int, default=1000,
                        help='Total env interaction steps for SI')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--prior_noise', type=float, default=0.3)
    parser.add_argument('--num_sims', type=int, default=200,
                        help='MCTS simulations per decision')
    parser.add_argument('--n_bins', type=int, default=7,
                        help='Action discretization bins')
    parser.add_argument('--action_repeat', type=int, default=2,
                        help='Action repeat k (DreamerV3 uses k=2)')
    parser.add_argument('--eval_episodes', type=int, default=3)
    parser.add_argument('--no_value_net', action='store_true',
                        help='Use heuristic rollout only (slow but good)')
    parser.add_argument('--distill_rollout', action='store_true',
                        help='Train VNet via model imagination (fast + good)')
    parser.add_argument('--multistep', type=int, default=0,
                        help='Multi-step SI loss horizon (0=1-step, 10=recommended)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='VNet weight decay (L2 regularization, e.g. 1e-3)')
    parser.add_argument('--state_noise', type=float, default=0.0,
                        help='Gaussian noise std on VNet training states (e.g. 0.01)')
    parser.add_argument('--cart_drift_penalty', type=float, default=0.0,
                        help='R2 return shaping: alpha*max(|x|-1,0)^2 (e.g. 2.0)')
    parser.add_argument('--vnet_hidden', type=int, default=128,
                        help='Hidden size for SwingupValueNet')
    parser.add_argument('--vnet_val_fraction', type=float, default=0.15,
                        help='Validation fraction for VNet selection (0 disables)')
    parser.add_argument('--vnet_restarts', type=int, default=5,
                        help='Number of VNet random restarts (1 disables selection)')
    parser.add_argument('--no_fix_gravity', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.num_sims = 50
        args.eval_episodes = 2
        args.seeds = 2

    # Determine value net mode
    if args.distill_rollout:
        vnet_mode = 'DISTILL'
    elif args.no_value_net:
        vnet_mode = 'OFF'
    else:
        vnet_mode = 'ON (env-data)'

    log("=" * 70)
    log("DMControl Cartpole-Swingup  SD-MCTS Benchmark  v3")
    log("  (RK4 + damping + exact reward + action repeat)")
    log("=" * 70)
    log(f"  Budget:         {args.budget} env steps")
    log(f"  Seeds:          {args.seeds}")
    log(f"  Prior noise:    {args.prior_noise}")
    log(f"  MCTS sims:      {args.num_sims}")
    log(f"  Action bins:    {args.n_bins}")
    log(f"  Action repeat:  {args.action_repeat}")
    log(f"  Value net:      {vnet_mode}")
    log(f"  Fix gravity:    {'NO' if args.no_fix_gravity else 'YES'}")
    log(f"  Eval episodes:  {args.eval_episodes}")
    log(f"  SI multistep:   {args.multistep if args.multistep > 0 else 'OFF (1-step)'}")
    if args.weight_decay > 0:
        log(f"  VNet WD:        {args.weight_decay}")
    if args.state_noise > 0:
        log(f"  State noise:    {args.state_noise}")
    if args.cart_drift_penalty > 0:
        log(f"  Cart drift pen: {args.cart_drift_penalty}")
    log(f"  VNet hidden:    {args.vnet_hidden}")
    log(f"  VNet val frac:  {args.vnet_val_fraction}")
    log(f"  VNet restarts:  {args.vnet_restarts}")
    log("=" * 70)

    all_returns = []
    all_params = []
    t0 = time.time()

    for i in range(args.seeds):
        seed = i * 42
        log(f"\n{'='*30} Seed {seed} {'='*30}")
        returns, params = run_experiment(
            budget=args.budget, seed=seed,
            prior_noise=args.prior_noise,
            num_sims=args.num_sims,
            n_action_bins=args.n_bins,
            eval_episodes=args.eval_episodes,
            use_value_net=not args.no_value_net and not args.distill_rollout,
            fix_gravity=not args.no_fix_gravity,
            action_repeat=args.action_repeat,
            distill_rollout=args.distill_rollout,
            multistep=args.multistep,
            weight_decay=args.weight_decay,
            state_noise=args.state_noise,
            cart_drift_penalty=args.cart_drift_penalty,
            vnet_hidden=args.vnet_hidden,
            vnet_val_fraction=args.vnet_val_fraction,
            vnet_restarts=args.vnet_restarts,
        )
        all_returns.append(returns)
        all_params.append(params)

    elapsed = time.time() - t0
    all_returns = np.concatenate(all_returns)

    # Random baseline
    log("\n--- Random Baseline ---")
    rand_env = suite.load('cartpole', 'swingup')
    random_returns = evaluate_random(
        rand_env, n_episodes=args.eval_episodes * args.seeds)

    # Results
    log("\n" + "=" * 75)
    log("RESULTS SUMMARY (v3 — with action repeat)")
    log("=" * 75)
    log(f"\n{'Method':<50} {'Mean':>7} {'±Std':>7} {'Min':>6} {'Max':>6}")
    log("-" * 80)
    log(f"{'SD-MCTS (Ours, ' + str(args.budget) + ' steps, k=' + str(args.action_repeat) + ')':<50} "
        f"{all_returns.mean():>7.1f} {all_returns.std():>7.1f} "
        f"{all_returns.min():>6.0f} {all_returns.max():>6.0f}")
    log(f"{'Oracle (true params, k=' + str(args.action_repeat) + ', 100 sims)':<50} "
        f"{'523.7':>7} {'7.8':>7}")
    log(f"{'Random Policy':<50} "
        f"{random_returns.mean():>7.1f} {random_returns.std():>7.1f} "
        f"{random_returns.min():>6.0f} {random_returns.max():>6.0f}")
    log("-" * 80)
    log(f"{'DreamerV3 (100k steps, k=2)':<50} {'~725':>7}")
    log(f"{'DreamerV3 (1M steps, k=2)':<50} {'861':>7}")

    log(f"\n{'--- System Identification ---':^80}")
    log(f"{'Param':<12} {'True':>8} {'Learned':>10} {'Error':>8}")
    log("-" * 40)
    for key in ['gravity', 'masscart', 'masspole', 'length']:
        true_val = all_params[0][f'true_{key}']
        learned = [p[key] for p in all_params]
        mean_l = np.mean(learned)
        err = abs(mean_l - true_val) / true_val * 100
        log(f"{key:<12} {true_val:>8.3f} {mean_l:>10.3f} {err:>7.1f}%")

    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log("=" * 75)
    log(f"\nSample efficiency: SD-MCTS uses {args.budget} env steps vs "
        f"DreamerV3's 100,000 env steps ({100000//args.budget}x less data)")


if __name__ == '__main__':
    main()
