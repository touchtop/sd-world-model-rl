#!/usr/bin/env python3
"""Diagnose Seed 42 vs Seed 0: VNet quality, imagination coverage, eval trace.

Usage:
    python -u scripts/diagnose_seed42.py 2>&1 | tee /tmp/diagnose_seed42.log
"""
import sys, os, math, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from scripts.run_dmcontrol_cartpole import (
    DMControlCartPoleSDModel, DMControlCartPoleWrapper,
    get_discrete_actions, flatten_obs, obs_to_state,
    collect_trajectories, train_sd_model,
    SwingupValueNet, train_value_net,
    generate_imagination_data, model_forward_k,
    MCTSNode, _swingup_action,
)
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def train_seed(seed):
    """Reproduce SI + VNet R1 for a given seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    model = DMControlCartPoleSDModel(learn_params=True, prior_noise=0.3)
    with torch.no_grad():
        model._gravity_raw.data = torch.tensor(
            model._inv_softplus(9.81), dtype=torch.float32)
    model._gravity_raw.requires_grad_(False)

    data = collect_trajectories(env, 5000, policy='random')
    train_sd_model(model, data, epochs=800, lr=5e-3,
                   use_sincos=True, multistep=10)
    params = model.get_learned_params()
    log(f"  Seed {seed} SI: mc={params['masscart']:.3f}, "
        f"mp={params['masspole']:.3f}, l={params['length']:.3f}")

    wrapper = DMControlCartPoleWrapper(model)
    imag_states, imag_returns = generate_imagination_data(
        wrapper, action_repeat=2, n_trajectories=2000, horizon=60,
        real_states=data['states'])

    value_net = SwingupValueNet()
    r1_loss = train_value_net(
        value_net, imag_states, imag_returns, epochs=500, lr=1e-3)
    log(f"  Seed {seed} R1 loss: {r1_loss:.4f}")
    return model, value_net, imag_states, imag_returns


def part1_coverage(imag_states, label):
    """Imagination data state coverage."""
    cos_th = np.cos(imag_states[:, 1].numpy())
    theta_dots = imag_states[:, 3].numpy()
    xs = imag_states[:, 0].numpy()

    log(f"\n=== Imagination Coverage: {label} ({len(imag_states)} samples) ===")
    bins = [(-1, -0.5), (-0.5, 0.0), (0.0, 0.5),
            (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]
    for lo, hi in bins:
        n = int(np.sum((cos_th >= lo) & (cos_th < hi)))
        pct = n / len(cos_th) * 100
        log(f"  cos(θ) [{lo:+.2f},{hi:+.2f}): {n:>5d} ({pct:>5.1f}%)")

    near = cos_th > 0.95
    n_near = int(near.sum())
    log(f"  Near-upright (cos>0.95): {n_near} ({n_near/len(cos_th)*100:.1f}%)")
    if n_near > 0:
        log(f"    θ̇: [{theta_dots[near].min():.1f}, {theta_dots[near].max():.1f}]"
            f"  x: [{xs[near].min():.2f}, {xs[near].max():.2f}]")

    # Returns distribution for near-upright
    near_mask = cos_th > 0.8
    n_m = int(near_mask.sum())
    log(f"  Balance zone (cos>0.8): {n_m} ({n_m/len(cos_th)*100:.1f}%)")


def part2_vnet_grid(v0, v42):
    """Compare VNet predictions side by side on diagnostic grid."""
    log(f"\n=== VNet Grid Comparison ===")
    log(f"  {'θ':>6} {'θ̇':>6} {'x':>4} | {'V(s0)':>8} {'V(s42)':>8} {'Δ':>8}")
    log(f"  {'-'*48}")
    for theta in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, math.pi]:
        for theta_dot in [0.0, 1.0, 3.0, -1.0]:
            s = torch.tensor([[0.0, theta, 0.0, theta_dot]],
                             dtype=torch.float32)
            with torch.no_grad():
                val0 = v0(s).item()
                val42 = v42(s).item()
            log(f"  {theta:>6.2f} {theta_dot:>+6.1f} {0.0:>4.1f} | "
                f"{val0:>8.2f} {val42:>8.2f} {val0-val42:>+8.2f}")


def part3_eval_trace(model, value_net, discrete_actions, seed):
    """One eval episode with per-step MCTS tracing (every 10 env steps)."""
    log(f"\n=== Eval Trace: Seed {seed} (R1 VNet only) ===")
    wrapper = DMControlCartPoleWrapper(model)
    eval_env = suite.load('cartpole', 'swingup',
                          task_kwargs={'random': seed + 1000})
    ts = eval_env.reset()
    obs = flatten_obs(ts.observation)
    state = obs_to_state(obs)
    ep_ret = 0.0
    action_repeat = 2
    n_actions = len(discrete_actions)
    gamma = 0.99
    gamma_k = gamma ** action_repeat

    log(f"  {'step':>4} {'cosθ':>6} {'θ':>7} {'θ̇':>7} {'x':>6} "
        f"{'act':>5} {'V(s)':>7} {'cumR':>7} | top-3")

    for dec in range(500):  # up to 1000 env steps
        env_step = dec * action_repeat
        cos_th = math.cos(state[1])

        with torch.no_grad():
            v_est = value_net(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            ).item()

        # MCTS with tree info capture
        root = MCTSNode(torch.tensor(state, dtype=torch.float32))
        for _ in range(200):
            node = root
            path = [node]
            while node.children and len(node.children) == n_actions:
                best_child, best_score = None, -float('inf')
                for child in node.children.values():
                    if child.N == 0:
                        score = float('inf')
                    else:
                        score = child.Q + 1.5 * math.sqrt(
                            math.log(node.N + 1) / (child.N + 1))
                    if score > best_score:
                        best_score = score
                        best_child = child
                node = best_child
                path.append(node)
            if len(node.children) < n_actions:
                for a_idx in range(n_actions):
                    if a_idx not in node.children:
                        av = discrete_actions[a_idx]
                        fs, er, _ = model_forward_k(
                            wrapper, node.state, av, action_repeat, gamma)
                        child = MCTSNode(fs, parent=node,
                                         action_val=av, edge_reward=er)
                        node.children[a_idx] = child
                        node = child
                        path.append(node)
                        break
            with torch.no_grad():
                lv = value_net(node.state.unsqueeze(0)).item()
            G = lv
            for nb in reversed(path):
                if nb.parent is not None:
                    G = nb.edge_reward + gamma_k * G
                nb.N += 1
                nb.W += G
                nb.Q = nb.W / nb.N

        # Top-3 actions
        ci = sorted([(discrete_actions[i], c.N, c.Q)
                      for i, c in root.children.items()],
                     key=lambda x: -x[1])
        best_idx = max(root.children.keys(),
                       key=lambda a: root.children[a].N)
        chosen = discrete_actions[best_idx]
        top3 = " ".join(f"{a:+.2f}:{n:>3d}" for a, n, _ in ci[:3])

        # Log every 5 decisions (10 env steps)
        if dec % 5 == 0:
            log(f"  {env_step:>4d} {cos_th:>+6.3f} {state[1]:>+7.3f} "
                f"{state[3]:>+7.2f} {state[0]:>+6.2f} "
                f"{chosen:>+5.2f} {v_est:>7.1f} {ep_ret:>7.1f} | {top3}")

        # Execute
        for _ in range(action_repeat):
            ts = eval_env.step(np.array([chosen]))
            ep_ret += (ts.reward if ts.reward is not None else 0.0)
            if ts.last():
                break
        if ts.last():
            break
        obs = flatten_obs(ts.observation)
        state = obs_to_state(obs)

    log(f"  TOTAL return: {ep_ret:.1f}")


if __name__ == '__main__':
    try:
        seeds_data = {}
        for seed in [0, 42]:
            log(f"\n{'='*25} Training Seed {seed} {'='*25}")
            m, v, ims, imr = train_seed(seed)
            seeds_data[seed] = (m, v, ims, imr)

        # Part 1: coverage
        for seed in [0, 42]:
            part1_coverage(seeds_data[seed][2], f"Seed {seed}")

        # Part 2: VNet grid
        part2_vnet_grid(seeds_data[0][1], seeds_data[42][1])

        # Part 3: eval trace
        discrete_actions = get_discrete_actions(11)
        for seed in [0, 42]:
            part3_eval_trace(
                seeds_data[seed][0], seeds_data[seed][1],
                discrete_actions, seed)

        log("\nDiagnostic complete.")
    except Exception:
        traceback.print_exc()
