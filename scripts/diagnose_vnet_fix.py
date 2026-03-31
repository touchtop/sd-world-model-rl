#!/usr/bin/env python3
"""VNet fix validation: test R1-Only vs R1-MultiRestart on failing seeds.

Tests 3 VNet training configurations on failing seeds to find the minimal
fix for seed-dependent VNet collapse:

  1. Baseline  — current R1 + R2 bootstrap (the broken pipeline)
  2. R1-Only   — train VNet on imagination data only, skip R2
  3. R1-Multi  — train 5 VNets with different inits, pick best by val loss

Usage:
    python -u scripts/diagnose_vnet_fix.py 2>&1 | tee /tmp/vnet_fix.log
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
TEST_SEEDS = [21, 63]           # failing seeds to diagnose
HEALTHY_SEEDS = [0, 84]         # healthy seeds for sanity check
ALL_SEEDS = HEALTHY_SEEDS + TEST_SEEDS

BUDGET = 10000
PRIOR_NOISE = 0.3
NUM_SIMS = 200
N_BINS = 11
ACTION_REPEAT = 2
MULTISTEP = 10
EVAL_EPISODES = 5

N_RESTARTS = 5                  # number of VNet restarts for multi-restart
VAL_FRACTION = 0.15             # fraction of imagination data held out for val
# ───────────────────────────────────────────────────────────────


def train_si_and_collect(seed):
    """Train SI model only. Returns (model, wrapper, data, discrete_actions)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    discrete_actions = get_discrete_actions(N_BINS)

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

    wrapper = DMControlCartPoleWrapper(model)
    return model, wrapper, data, discrete_actions


def generate_r1_data(wrapper, data):
    """Generate R1 imagination data."""
    imag_states, imag_returns = generate_imagination_data(
        wrapper, action_repeat=ACTION_REPEAT,
        n_trajectories=2000, horizon=60,
        real_states=data['states'],
    )
    return imag_states, imag_returns


def train_vnet_baseline(wrapper, data, discrete_actions, seed, imag_s, imag_r):
    """Config 1: Baseline R1 + R2 (current pipeline)."""
    torch.manual_seed(seed)
    value_net = SwingupValueNet()
    r1_loss = train_value_net(
        value_net, imag_s, imag_r, epochs=500, lr=1e-3, min_loss=0.05)
    log(f"    R1 loss: {r1_loss:.4f}")

    # R2 bootstrap
    bs_states, bs_returns = [], []
    gamma, gamma_k = 0.99, 0.99 ** ACTION_REPEAT
    obs_cos = torch.cos(data['states'][:, 1])
    upright_idx = torch.where(obs_cos > 0.5)[0]

    for ti in range(200):
        if len(upright_idx) > 0 and ti < 100:
            idx = upright_idx[np.random.randint(len(upright_idx))].item()
        else:
            idx = np.random.randint(len(data['states']))
        rs = data['states'][idx].clone()
        if ti % 2 == 0:
            rs = rs + torch.randn_like(rs) * 0.05
        traj_s, traj_r = [], []
        for _ in range(30):
            traj_s.append(rs.detach().clone())
            act = mcts_search_swingup(
                rs.numpy(), wrapper, discrete_actions,
                num_simulations=max(30, NUM_SIMS // 4),
                value_net=value_net, action_repeat=ACTION_REPEAT,
            )
            rs_new, step_r, _ = model_forward_k(
                wrapper, rs, act, ACTION_REPEAT, gamma)
            traj_r.append(step_r)
            rs = rs_new
        rets = np.zeros(len(traj_r), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(traj_r))):
            G = traj_r[t] + gamma_k * G
            rets[t] = G
        for t in range(0, len(traj_s), 3):
            bs_states.append(traj_s[t])
            bs_returns.append(rets[t])

    bs_s = torch.stack(bs_states)
    bs_r = np.array(bs_returns, dtype=np.float32)
    log(f"    R2: {len(bs_s)} samples, returns [{bs_r.min():.1f}, {bs_r.max():.1f}]")

    all_s = torch.cat([imag_s, bs_s])
    all_r = np.concatenate([imag_r, bs_r])
    r2_loss = train_value_net(
        value_net, all_s, all_r, epochs=500, lr=1e-3, min_loss=0.05)
    log(f"    R2 loss: {r2_loss:.4f}")
    return value_net


def train_vnet_r1_only(seed, imag_s, imag_r):
    """Config 2: R1 only, skip R2 entirely."""
    torch.manual_seed(seed)
    value_net = SwingupValueNet()
    r1_loss = train_value_net(
        value_net, imag_s, imag_r, epochs=500, lr=1e-3, min_loss=0.05)
    log(f"    R1-only loss: {r1_loss:.4f}")
    return value_net


def train_vnet_r1_multi_restart(seed, imag_s, imag_r):
    """Config 3: Train N_RESTARTS VNets with different inits, pick best by val loss."""
    n = len(imag_s)
    n_val = max(int(n * VAL_FRACTION), 100)
    # Fixed random split (same across restarts for fair comparison)
    rng = np.random.RandomState(seed + 9999)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_s = imag_s[train_idx]
    train_r = imag_r[train_idx]
    val_s = imag_s[val_idx]
    val_r_np = imag_r[val_idx]
    val_r_t = torch.from_numpy(val_r_np).float()

    best_vnet = None
    best_val_loss = float('inf')

    for restart in range(N_RESTARTS):
        torch.manual_seed(seed * 1000 + restart * 137 + 7)
        vnet = SwingupValueNet()
        train_loss = train_value_net(
            vnet, train_s, train_r, epochs=500, lr=1e-3, min_loss=0.05)

        # Evaluate on validation set
        with torch.no_grad():
            preds = vnet(val_s).squeeze()
            val_loss = ((preds - val_r_t) ** 2).mean().item()

        log(f"    Restart {restart}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_vnet = copy.deepcopy(vnet)

    log(f"    >> Best restart val_loss: {best_val_loss:.4f}")
    return best_vnet


def evaluate_vnet(model, value_net, seed):
    """Evaluate a (model, vnet) pair."""
    discrete_actions = get_discrete_actions(N_BINS)
    eval_env = suite.load('cartpole', 'swingup',
                          task_kwargs={'random': seed + 1000})
    returns = evaluate_agent(
        eval_env, model, discrete_actions,
        n_episodes=EVAL_EPISODES,
        num_simulations=NUM_SIMS,
        value_net=value_net,
        action_repeat=ACTION_REPEAT,
        verbose=False,
    )
    return returns


def main():
    t0 = time.time()
    log("=" * 70)
    log("VNET FIX VALIDATION EXPERIMENT")
    log(f"  Test seeds (failing):  {TEST_SEEDS}")
    log(f"  Healthy seeds (check): {HEALTHY_SEEDS}")
    log(f"  Configs: Baseline(R1+R2), R1-Only, R1-MultiRestart(×{N_RESTARTS})")
    log("=" * 70)

    # ── Phase 1: Train SI + generate R1 data for all seeds ─────
    seed_data = {}
    for seed in ALL_SEEDS:
        log(f"\n{'='*25} SI + R1 Data for Seed {seed} {'='*25}")
        model, wrapper, data, actions = train_si_and_collect(seed)
        imag_s, imag_r = generate_r1_data(wrapper, data)
        log(f"  [Seed {seed}] R1 data: {len(imag_s)} samples, "
            f"returns [{imag_r.min():.1f}, {imag_r.max():.1f}]")
        seed_data[seed] = {
            'model': model, 'wrapper': wrapper, 'data': data,
            'actions': actions, 'imag_s': imag_s, 'imag_r': imag_r,
        }

    si_time = time.time() - t0
    log(f"\nSI + R1 data generation complete ({si_time:.0f}s)")

    # ── Phase 2: Train VNets with 3 configurations per seed ────
    configs = ['Baseline(R1+R2)', 'R1-Only', 'R1-MultiRestart']
    vnets = {}  # vnets[(seed, config)] = value_net

    for seed in ALL_SEEDS:
        d = seed_data[seed]
        log(f"\n{'='*25} VNet Training for Seed {seed} {'='*25}")

        # Config 1: Baseline
        log(f"  [Seed {seed}] Config: Baseline (R1+R2)")
        vnets[(seed, 'Baseline(R1+R2)')] = train_vnet_baseline(
            d['wrapper'], d['data'], d['actions'], seed,
            d['imag_s'], d['imag_r'])

        # Config 2: R1-Only
        log(f"  [Seed {seed}] Config: R1-Only")
        vnets[(seed, 'R1-Only')] = train_vnet_r1_only(
            seed, d['imag_s'], d['imag_r'])

        # Config 3: R1-MultiRestart
        log(f"  [Seed {seed}] Config: R1-MultiRestart (×{N_RESTARTS})")
        vnets[(seed, 'R1-MultiRestart')] = train_vnet_r1_multi_restart(
            seed, d['imag_s'], d['imag_r'])

    train_time = time.time() - t0
    log(f"\nAll VNet training complete ({train_time:.0f}s)")

    # ── Phase 3: Evaluate all (seed, config) pairs ─────────────
    log("\n" + "=" * 70)
    log("EVALUATION RESULTS")
    log("=" * 70)

    results = {}
    for seed in ALL_SEEDS:
        model = seed_data[seed]['model']
        for config in configs:
            vnet = vnets[(seed, config)]
            log(f"\n  Evaluating Seed {seed} / {config}...")
            rets = evaluate_vnet(model, vnet, seed)
            mean_r, std_r = rets.mean(), rets.std()
            results[(seed, config)] = (mean_r, std_r)
            log(f"  >> Seed {seed} / {config}: "
                f"{mean_r:.1f} ±{std_r:.1f}  "
                f"[{rets.min():.0f}, {rets.max():.0f}]")

    # ── Phase 4: Summary Table ─────────────────────────────────
    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)

    header = f"{'Seed':>6s}"
    for c in configs:
        header += f"  {c:>20s}"
    log(header)
    log("-" * (6 + 22 * len(configs)))

    for seed in ALL_SEEDS:
        tag = " ✓" if seed in HEALTHY_SEEDS else " ✗"
        row = f"{seed:>4d}{tag}"
        for config in configs:
            m, s = results[(seed, config)]
            row += f"  {m:>15.1f}±{s:<4.1f}"
        log(row)

    # Per-config averages
    log("-" * (6 + 22 * len(configs)))
    row_all = f"{'Mean':>6s}"
    row_fail = f"{'Fail':>6s}"
    for config in configs:
        vals_all = [results[(s, config)][0] for s in ALL_SEEDS]
        vals_fail = [results[(s, config)][0] for s in TEST_SEEDS]
        row_all += f"  {np.mean(vals_all):>15.1f}     "
        row_fail += f"  {np.mean(vals_fail):>15.1f}     "
    log(row_all)
    log(row_fail)

    total = time.time() - t0
    log(f"\nTotal time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main()
