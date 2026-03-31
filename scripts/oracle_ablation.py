#!/usr/bin/env python3
"""Oracle Ablation v2: Action Repeat experiment.

Tests the impact of action repeat (macro-actions) on MCTS planning.
With action repeat k, each MCTS tree edge = k env steps with same action.

DreamerV3 uses action_repeat=2 on DMControl cartpole-swingup.

Usage:
    conda run -n mapf python scripts/oracle_ablation.py 2>&1 | tee /tmp/oracle_v2.log
"""
import sys
import os
import time
import math
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from src.world_model.dmcontrol_cartpole_sd_model import (
    DMControlCartPoleSDModel,
    DMControlCartPoleWrapper,
)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_discrete_actions(n_bins=5):
    return np.linspace(-1.0, 1.0, n_bins).astype(np.float32)


def flatten_obs(obs):
    return np.concatenate([obs['position'], obs['velocity']]).astype(np.float32)


def obs_to_state(obs_flat):
    return DMControlCartPoleSDModel.obs_to_state(obs_flat)


# ============================================================
# Multi-step forward (action repeat in the world model)
# ============================================================
def model_forward_k(wrapper, state, action_val, k):
    """Roll the world model forward k steps with the same action.

    Returns: (final_state, cumulative_reward, gamma^k discount)
    """
    s = state
    gamma = 0.99
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
# MCTS with action repeat
# ============================================================
class MCTSNode:
    __slots__ = ['state', 'parent', 'action_val', 'edge_reward',
                 'children', 'N', 'W', 'Q']

    def __init__(self, state, parent=None, action_val=None, edge_reward=0.0):
        self.state = state
        self.parent = parent
        self.action_val = action_val
        self.edge_reward = edge_reward  # cumulative reward along the k-step edge
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0


def mcts_search(root_np, wrapper, da, num_sims=100, c=1.5, gamma=0.99,
                action_repeat=1, rollout_depth=20):
    """MCTS with action repeat (macro-actions)."""
    na = len(da)
    root = MCTSNode(torch.tensor(root_np, dtype=torch.float32))

    # Macro-step discount
    gamma_k = gamma ** action_repeat

    for _ in range(num_sims):
        node = root
        path = [node]

        # Selection
        while node.children and len(node.children) == na:
            best, best_s = None, -float('inf')
            for ch in node.children.values():
                if ch.N == 0:
                    s = float('inf')
                else:
                    s = ch.Q + c * math.sqrt(math.log(node.N + 1) / (ch.N + 1))
                if s > best_s:
                    best_s = s
                    best = ch
            node = best
            path.append(node)

        # Expansion — each edge = k steps with same action
        if len(node.children) < na:
            for ai in range(na):
                if ai not in node.children:
                    final_state, edge_r, _ = model_forward_k(
                        wrapper, node.state, da[ai], action_repeat
                    )
                    ch = MCTSNode(
                        final_state, parent=node,
                        action_val=da[ai], edge_reward=edge_r,
                    )
                    node.children[ai] = ch
                    node = ch
                    path.append(node)
                    break

        # Heuristic rollout (also uses action repeat)
        rs = node.state
        ret = 0.0
        disc = 1.0
        for _ in range(rollout_depth):
            th = rs[1].item()
            thd = rs[3].item()
            cos_th = math.cos(th)
            if cos_th > 0.5:
                ha = float(np.clip(-(math.sin(th) + 0.3 * thd) * 2, -1, 1))
            else:
                ha = float(np.clip(thd * 2, -1, 1))
            new_state, step_r, step_disc = model_forward_k(
                wrapper, rs, ha, action_repeat
            )
            ret += disc * step_r
            disc *= gamma_k
            rs = new_state

        # Backpropagation
        G = ret
        for nb in reversed(path):
            if nb.parent is not None:
                G = nb.edge_reward + gamma_k * G
            nb.N += 1
            nb.W += G
            nb.Q = nb.W / nb.N

    if not root.children:
        return 0.0
    return da[max(root.children.keys(), key=lambda a: root.children[a].N)]


# ============================================================
# Evaluation with action repeat in env
# ============================================================
def evaluate(env, wrapper, da, num_sims, action_repeat=1,
             n_episodes=2, max_steps=1000, label=""):
    """Evaluate MCTS with action repeat."""
    all_returns = []

    for ep in range(n_episodes):
        ts = env.reset()
        state = obs_to_state(flatten_obs(ts.observation))
        ep_ret = 0.0
        env_step = 0

        while env_step < max_steps:
            # MCTS decides one macro-action
            action = mcts_search(
                state, wrapper, da,
                num_sims=num_sims, action_repeat=action_repeat,
                rollout_depth=max(10, 30 // action_repeat),
            )

            # Execute the action for k env steps
            for _ in range(action_repeat):
                ts = env.step(np.array([action]))
                reward = ts.reward if ts.reward is not None else 0.0
                ep_ret += reward
                env_step += 1

                if ts.last() or env_step >= max_steps:
                    break

            if ts.last():
                break

            state = obs_to_state(flatten_obs(ts.observation))

            # Progress log every ~100 env steps
            if env_step % 100 == 0 and env_step > 0:
                cos_th = math.cos(state[1])
                log(f"  {label} ep={ep} envStep={env_step:>4d}  "
                    f"cumR={ep_ret:>6.1f}  cos(θ)={cos_th:>+.3f}  "
                    f"a={action:>+.2f}")

        all_returns.append(ep_ret)
        log(f"  {label} ep={ep} DONE envSteps={env_step}  return={ep_ret:.1f}")

    return np.array(all_returns)


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 70)
    log("ORACLE ABLATION v2: Action Repeat Experiment")
    log("=" * 70)

    oracle = DMControlCartPoleSDModel(learn_params=False)
    wrapper = DMControlCartPoleWrapper(oracle)

    n_episodes = 2
    max_steps = 1000  # Full episode

    # ---- Part 1: Action Repeat sweep (5 bins, 100 sims) ----
    log("\n" + "=" * 70)
    log("Part 1: Action Repeat Sweep (5 bins, 100 sims, 1000 steps/ep)")
    log("=" * 70)

    da5 = get_discrete_actions(5)
    results = {}

    for k in [1, 2, 4, 8]:
        label = f"k={k}"
        log(f"\n--- Action Repeat k={k}  "
            f"(effective horizon per sim = {k}×depth steps) ---")
        env = suite.load('cartpole', 'swingup', task_kwargs={'random': 0})
        t0 = time.time()
        rets = evaluate(
            env, wrapper, da5, num_sims=100,
            action_repeat=k, n_episodes=n_episodes,
            max_steps=max_steps, label=label,
        )
        elapsed = time.time() - t0
        results[k] = rets
        log(f"  k={k} RESULT: Mean={rets.mean():.1f} ±{rets.std():.1f}  "
            f"Min={rets.min():.0f} Max={rets.max():.0f}  ({elapsed:.0f}s)")

    # ---- Part 2: Best k with more sims ----
    log("\n" + "=" * 70)
    log("Part 2: Best k + more sims (5 bins, 1000 steps/ep)")
    log("=" * 70)

    for k, sims in [(4, 200), (8, 200)]:
        label = f"k={k},sims={sims}"
        log(f"\n--- k={k}, sims={sims} ---")
        env = suite.load('cartpole', 'swingup', task_kwargs={'random': 0})
        t0 = time.time()
        rets = evaluate(
            env, wrapper, da5, num_sims=sims,
            action_repeat=k, n_episodes=n_episodes,
            max_steps=max_steps, label=label,
        )
        elapsed = time.time() - t0
        results[(k, sims)] = rets
        log(f"  {label} RESULT: Mean={rets.mean():.1f} ±{rets.std():.1f}  "
            f"({elapsed:.0f}s)")

    # ---- Random ----
    log("\n--- Random Baseline (1000 steps) ---")
    env = suite.load('cartpole', 'swingup')
    rr = []
    for _ in range(n_episodes):
        ts = env.reset()
        r = 0
        for _ in range(max_steps):
            ts = env.step(np.random.uniform(-1, 1, size=(1,)).astype(np.float32))
            r += (ts.reward if ts.reward is not None else 0)
            if ts.last():
                break
        rr.append(r)
    log(f"  Random RESULT: Mean={np.mean(rr):.1f}")

    # ---- Summary ----
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"{'Config':<35} {'Mean':>8} {'±Std':>8} {'Max':>6}")
    log("-" * 60)
    for key, rets in results.items():
        if isinstance(key, tuple):
            lbl = f"Oracle k={key[0]}, sims={key[1]}"
        else:
            lbl = f"Oracle k={key}, sims=100"
        log(f"{lbl:<35} {rets.mean():>8.1f} {rets.std():>8.1f} {rets.max():>6.0f}")
    log(f"{'Random':<35} {np.mean(rr):>8.1f}")
    log(f"{'DreamerV3 (100k, k=2)':<35} {'~725':>8}")
    log(f"{'DreamerV3 (1M, k=2)':<35} {'861':>8}")
    log("=" * 70)


if __name__ == '__main__':
    main()
