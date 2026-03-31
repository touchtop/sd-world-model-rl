#!/usr/bin/env python3
"""Sweep cart_drift_penalty alpha with robust VNet pipeline.

Pipeline: SI → R1-Only MultiRestart (no R2 bootstrap) → Alpha Sweep Eval.

The R2 bootstrap is deliberately omitted because our diagnostic experiments
proved it creates a "deadly triad" feedback loop (MCTS exploits VNet
overestimates → biased R2 data → VNet degrades further), making results
highly seed-sensitive.  R1-MultiRestart with validation-based selection
recovers 3/4 previously failing seeds.

Usage:
    python -u scripts/sweep_alpha.py 2>&1 | tee /tmp/sweep_robust.log
"""
import sys, os, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from scripts.run_dmcontrol_cartpole import (
    DMControlCartPoleSDModel, DMControlCartPoleWrapper,
    SwingupValueNet, get_discrete_actions, flatten_obs, obs_to_state,
    collect_trajectories, train_sd_model, generate_imagination_data,
    train_value_net, mcts_search_swingup, evaluate_agent, model_forward_k,
    log,
)

# ── Configuration ──────────────────────────────────────────────
SEEDS = [0, 10, 21, 30, 42, 55, 63, 72, 84, 95]   # 10 seeds
BUDGET = 10000
PRIOR_NOISE = 0.3
NUM_SIMS = 200
N_BINS = 11
ACTION_REPEAT = 2
MULTISTEP = 10
EVAL_EPISODES = 5

# VNet multi-restart settings
N_RESTARTS = 5
VAL_FRACTION = 0.15

# Alpha sweep range
ALPHA_VALUES = [0.0, 5.0]
# ───────────────────────────────────────────────────────────────


def train_seed(seed):
    """Train SI model + R1-MultiRestart VNet (no R2 bootstrap).

    Returns (model, value_net, discrete_actions).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    discrete_actions = get_discrete_actions(N_BINS)

    # ── SI Model ───────────────────────────────────────────────
    model = DMControlCartPoleSDModel(learn_params=True, prior_noise=PRIOR_NOISE)
    with torch.no_grad():
        model._gravity_raw.data = torch.tensor(
            model._inv_softplus(9.81), dtype=torch.float32)
    model._gravity_raw.requires_grad_(False)

    explore_steps = min(BUDGET, max(500, BUDGET // 2))
    data = collect_trajectories(env, explore_steps, policy='random')
    log(f"  [Seed {seed}] Collected {explore_steps} steps "
        f"({data['n_episodes']} ep)")

    train_loss = train_sd_model(
        model, data, epochs=800, lr=5e-3,
        use_sincos=True, multistep=MULTISTEP)
    params = model.get_learned_params()
    log(f"  [Seed {seed}] SI: mc={params['masscart']:.3f}, "
        f"mp={params['masspole']:.3f}, l={params['length']:.3f} "
        f"(loss={train_loss:.6f})")

    # ── R1 Imagination Data ────────────────────────────────────
    wrapper = DMControlCartPoleWrapper(model)
    imag_states, imag_returns = generate_imagination_data(
        wrapper, action_repeat=ACTION_REPEAT,
        n_trajectories=2000, horizon=60,
        real_states=data['states'],
    )
    log(f"  [Seed {seed}] R1: {len(imag_states)} samples, "
        f"returns [{imag_returns.min():.1f}, {imag_returns.max():.1f}]")

    # ── R1-MultiRestart VNet (no R2) ──────────────────────────
    n = len(imag_states)
    n_val = max(int(n * VAL_FRACTION), 100)
    rng = np.random.RandomState(seed + 9999)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_s, train_r = imag_states[train_idx], imag_returns[train_idx]
    val_s = imag_states[val_idx]
    val_r_t = torch.from_numpy(imag_returns[val_idx]).float()

    best_vnet = None
    best_val_loss = float('inf')

    for restart in range(N_RESTARTS):
        torch.manual_seed(seed * 1000 + restart * 137 + 7)
        vnet = SwingupValueNet()
        t_loss = train_value_net(
            vnet, train_s, train_r, epochs=500, lr=1e-3, min_loss=0.05)

        with torch.no_grad():
            preds = vnet(val_s).squeeze()
            v_loss = ((preds - val_r_t) ** 2).mean().item()

        log(f"    restart {restart}: train={t_loss:.4f}, val={v_loss:.4f}"
            f"{'  ← best' if v_loss < best_val_loss else ''}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_vnet = copy.deepcopy(vnet)

    log(f"  [Seed {seed}] VNet best val_loss: {best_val_loss:.4f}")
    return model, best_vnet, discrete_actions


def evaluate_alpha(seed, model, value_net, discrete_actions, alpha):
    """Evaluate one (seed, alpha) pair."""
    eval_env = suite.load('cartpole', 'swingup',
                          task_kwargs={'random': seed + 1000})
    returns = evaluate_agent(
        eval_env, model, discrete_actions,
        n_episodes=EVAL_EPISODES,
        num_simulations=NUM_SIMS,
        value_net=value_net,
        action_repeat=ACTION_REPEAT,
        verbose=False,
        cart_drift_penalty=alpha,
    )
    return returns


def main():
    t0 = time.time()
    log("=" * 70)
    log("ROBUST ALPHA SWEEP (R1-MultiRestart, no R2)")
    log(f"  Seeds: {SEEDS}")
    log(f"  Alpha range: {ALPHA_VALUES}")
    log(f"  VNet restarts: {N_RESTARTS}")
    log(f"  Eval episodes per alpha: {EVAL_EPISODES}")
    log("=" * 70)

    # ── Phase 1: Train all seeds ───────────────────────────────
    trained = {}
    for seed in SEEDS:
        log(f"\n{'='*30} Training Seed {seed} {'='*30}")
        model, vnet, actions = train_seed(seed)
        trained[seed] = (model, vnet, actions)

    train_time = time.time() - t0
    log(f"\nTraining complete ({train_time:.0f}s). "
        f"Starting alpha sweep...\n")

    # ── Phase 2: Sweep alpha values ────────────────────────────
    results = {a: {} for a in ALPHA_VALUES}

    for alpha in ALPHA_VALUES:
        log(f"--- alpha={alpha:.1f} ---")
        for seed in SEEDS:
            model, vnet, actions = trained[seed]
            returns = evaluate_alpha(seed, model, vnet, actions, alpha)
            results[alpha][seed] = returns
            log(f"  Seed {seed}: {returns.mean():.1f} ±{returns.std():.1f}  "
                f"[{returns.min():.0f}, {returns.max():.0f}]")

        all_means = [results[alpha][s].mean() for s in SEEDS]
        overall = np.mean(all_means)
        log(f"  >> alpha={alpha:.1f}  overall={overall:.1f}  "
            f"per-seed={[f'{m:.0f}' for m in all_means]}")

    # ── Phase 3: Summary table ─────────────────────────────────
    log("\n" + "=" * 70)
    log("SWEEP RESULTS SUMMARY")
    log("=" * 70)

    header = f"{'alpha':>6s}"
    for s in SEEDS:
        header += f"  {'S'+str(s):>7s}"
    header += f"  {'Mean':>7s}  {'Median':>7s}  {'IQM':>7s}  {'<700':>5s}"
    log(header)
    log("-" * len(header))

    best_alpha, best_mean = 0.0, 0.0
    for alpha in ALPHA_VALUES:
        means = [results[alpha][s].mean() for s in SEEDS]
        overall = np.mean(means)
        median = np.median(means)
        # IQM: mean of middle 50%
        sorted_m = sorted(means)
        q1_idx = len(sorted_m) // 4
        q3_idx = len(sorted_m) - q1_idx
        iqm = np.mean(sorted_m[q1_idx:q3_idx]) if q3_idx > q1_idx else overall
        n_fail = sum(1 for m in means if m < 700)

        row = f"{alpha:6.1f}"
        for m in means:
            row += f"  {m:7.0f}"
        row += f"  {overall:7.1f}  {median:7.1f}  {iqm:7.1f}  {n_fail:5d}"
        log(row)

        if overall > best_mean:
            best_mean = overall
            best_alpha = alpha

    log("-" * len(header))
    log(f"Best alpha: {best_alpha:.1f} (overall mean={best_mean:.1f})")

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
