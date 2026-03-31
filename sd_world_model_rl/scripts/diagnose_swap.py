#!/usr/bin/env python3
"""Component-swap diagnostic: isolate SI vs VNet as root cause of seed failure.

Trains a healthy seed (0) and a failing seed (21), then evaluates all 4
combinations of (SI_model, VNet) to determine which component drives failure.

Interpretation:
  - If failure follows SI_model → System Identification is root cause
  - If failure follows VNet    → Value Network is root cause

Also prints SI parameter comparison and multi-step prediction RMSE.

Usage:
    python -u scripts/diagnose_swap.py 2>&1 | tee /tmp/diagnose_swap.log
"""
import sys, os, time, math
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
HEALTHY_SEED = 0
FAILING_SEED = 21
BUDGET = 10000
PRIOR_NOISE = 0.3
NUM_SIMS = 200
N_BINS = 11
ACTION_REPEAT = 2
MULTISTEP = 10
EVAL_EPISODES = 5
# ───────────────────────────────────────────────────────────────

# Ground truth parameters for dm_control cartpole
# (from dm_control/suite/cartpole.py source)
GROUND_TRUTH = {'masscart': 1.0, 'masspole': 0.1, 'length': 0.5}


def train_seed(seed):
    """Train SI model + VNet for one seed. Returns (model, value_net, data)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    discrete_actions = get_discrete_actions(N_BINS)

    model = DMControlCartPoleSDModel(learn_params=True, prior_noise=PRIOR_NOISE)
    with torch.no_grad():
        model._gravity_raw.data = torch.tensor(
            model._inv_softplus(9.81), dtype=torch.float32)
    model._gravity_raw.requires_grad_(False)

    # Phase 1: Collect random data
    explore_steps = min(BUDGET, max(500, BUDGET // 2))
    data = collect_trajectories(env, explore_steps, policy='random')
    log(f"  [Seed {seed}] Collected {explore_steps} steps "
        f"({data['n_episodes']} ep)")

    # Phase 2: System Identification
    train_loss = train_sd_model(
        model, data, epochs=800, lr=5e-3,
        use_sincos=True, multistep=MULTISTEP)
    params = model.get_learned_params()
    log(f"  [Seed {seed}] SI: mc={params['masscart']:.3f}, "
        f"mp={params['masspole']:.3f}, l={params['length']:.3f} "
        f"(loss={train_loss:.6f})")

    # Phase 3: VNet R1
    wrapper = DMControlCartPoleWrapper(model)
    imag_states, imag_returns = generate_imagination_data(
        wrapper, action_repeat=ACTION_REPEAT,
        n_trajectories=2000, horizon=60,
        real_states=data['states'],
    )
    log(f"  [Seed {seed}] R1: {len(imag_states)} samples, "
        f"returns [{imag_returns.min():.1f}, {imag_returns.max():.1f}]")

    value_net = SwingupValueNet()
    r1_loss = train_value_net(
        value_net, imag_states, imag_returns, epochs=500, lr=1e-3,
        min_loss=0.05)
    log(f"  [Seed {seed}] R1 loss: {r1_loss:.4f}")

    # Phase 4: VNet R2 bootstrap
    log(f"  [Seed {seed}] Bootstrap (MCTS-guided)...")
    bs_states, bs_returns = [], []
    gamma = 0.99
    gamma_k = gamma ** ACTION_REPEAT
    n_bs_traj = 200
    bs_horizon = 30

    obs_cos = torch.cos(data['states'][:, 1])
    upright_idx = torch.where(obs_cos > 0.5)[0]

    for ti in range(n_bs_traj):
        if len(upright_idx) > 0 and ti < n_bs_traj // 2:
            idx = upright_idx[np.random.randint(len(upright_idx))].item()
        else:
            idx = np.random.randint(len(data['states']))
        rs = data['states'][idx].clone()
        if ti % 2 == 0:
            rs = rs + torch.randn_like(rs) * 0.05
        traj_s, traj_r = [], []
        for _ in range(bs_horizon):
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

    bs_states_t = torch.stack(bs_states)
    bs_returns_a = np.array(bs_returns, dtype=np.float32)
    log(f"  [Seed {seed}] R2: {len(bs_states_t)} samples, "
        f"returns [{bs_returns_a.min():.1f}, {bs_returns_a.max():.1f}]")

    all_s = torch.cat([imag_states, bs_states_t])
    all_r = np.concatenate([imag_returns, bs_returns_a])
    r2_loss = train_value_net(
        value_net, all_s, all_r, epochs=500, lr=1e-3, min_loss=0.05)
    log(f"  [Seed {seed}] R2 loss: {r2_loss:.4f}")

    return model, value_net, discrete_actions, data


def measure_si_accuracy(model, seed, steps=30):
    """Measure multi-step prediction RMSE on fresh real rollouts."""
    env = suite.load('cartpole', 'swingup',
                     task_kwargs={'random': seed + 2000})
    wrapper = DMControlCartPoleWrapper(model)

    # Collect a short trajectory from the real environment
    data = collect_trajectories(env, 500, policy='random')
    states = data['states']  # (T, 4)
    actions = data['actions']  # (T,)

    errors_1step = []
    errors_kstep = {5: [], 10: [], 20: [], 30: []}

    for start in range(0, len(states) - steps - 1, 5):
        # 1-step error
        s = states[start]
        a = actions[start].item()
        pred = wrapper.predict_next_state(s.numpy(), a)
        real = states[start + 1].numpy()
        errors_1step.append(np.linalg.norm(pred - real))

        # Multi-step error
        s_pred = s.numpy().copy()
        for k in range(min(steps, len(states) - start - 1)):
            a_k = actions[start + k].item()
            s_pred = wrapper.predict_next_state(s_pred, a_k)
            actual_k = states[start + k + 1].numpy()
            kk = k + 1
            if kk in errors_kstep:
                errors_kstep[kk].append(np.linalg.norm(s_pred - actual_k))

    rmse_1 = np.mean(errors_1step)
    rmse_k = {k: np.mean(v) if v else float('nan')
              for k, v in errors_kstep.items()}
    return rmse_1, rmse_k


def evaluate_combo(si_model, value_net, seed_label, eval_seed=999):
    """Evaluate a (SI_model, VNet) combination."""
    discrete_actions = get_discrete_actions(N_BINS)
    eval_env = suite.load('cartpole', 'swingup',
                          task_kwargs={'random': eval_seed})
    returns = evaluate_agent(
        eval_env, si_model, discrete_actions,
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
    log("COMPONENT SWAP DIAGNOSTIC")
    log(f"  Healthy seed: {HEALTHY_SEED}")
    log(f"  Failing seed: {FAILING_SEED}")
    log("=" * 70)

    # ── Phase 1: Train both seeds ──────────────────────────────
    log(f"\n{'='*30} Training Seed {HEALTHY_SEED} (Healthy) {'='*30}")
    model_H, vnet_H, _, data_H = train_seed(HEALTHY_SEED)

    log(f"\n{'='*30} Training Seed {FAILING_SEED} (Failing) {'='*30}")
    model_F, vnet_F, _, data_F = train_seed(FAILING_SEED)

    train_time = time.time() - t0
    log(f"\nTraining complete ({train_time:.0f}s)")

    # ── Phase 2: SI Parameter Comparison ───────────────────────
    log("\n" + "=" * 70)
    log("SI PARAMETER COMPARISON")
    log("=" * 70)
    params_H = model_H.get_learned_params()
    params_F = model_F.get_learned_params()

    log(f"{'Param':<12s}  {'Ground Truth':>12s}  "
        f"{'Seed {0} (H)':>12s}  {'Seed {1} (F)':>12s}  "
        f"{'Err_H (%)':>10s}  {'Err_F (%)':>10s}".format(
            HEALTHY_SEED, FAILING_SEED))
    log("-" * 72)
    for p in ['masscart', 'masspole', 'length']:
        gt = GROUND_TRUTH[p]
        vh = params_H[p]
        vf = params_F[p]
        eh = abs(vh - gt) / gt * 100
        ef = abs(vf - gt) / gt * 100
        log(f"{p:<12s}  {gt:12.4f}  {vh:12.4f}  {vf:12.4f}  "
            f"{eh:9.1f}%  {ef:9.1f}%")

    # ── Phase 3: Multi-step Prediction RMSE ────────────────────
    log("\n" + "=" * 70)
    log("MULTI-STEP PREDICTION RMSE")
    log("=" * 70)

    rmse1_H, rmsek_H = measure_si_accuracy(model_H, HEALTHY_SEED)
    rmse1_F, rmsek_F = measure_si_accuracy(model_F, FAILING_SEED)

    log(f"{'Horizon':<10s}  {'Seed {0} (H)':>14s}  "
        f"{'Seed {1} (F)':>14s}  {'Ratio F/H':>10s}".format(
            HEALTHY_SEED, FAILING_SEED))
    log("-" * 52)
    log(f"{'1-step':<10s}  {rmse1_H:14.6f}  {rmse1_F:14.6f}  "
        f"{rmse1_F / max(rmse1_H, 1e-10):10.2f}x")
    for k in sorted(rmsek_H.keys()):
        ratio = rmsek_F[k] / max(rmsek_H[k], 1e-10)
        log(f"{f'{k}-step':<10s}  {rmsek_H[k]:14.6f}  {rmsek_F[k]:14.6f}  "
            f"{ratio:10.2f}x")

    # ── Phase 4: Component Swap Evaluation ─────────────────────
    log("\n" + "=" * 70)
    log("COMPONENT SWAP EVALUATION")
    log("(using a fixed eval seed=999 for fair comparison)")
    log("=" * 70)

    combos = [
        ("SI_H + V_H", model_H, vnet_H, "baseline healthy"),
        ("SI_F + V_H", model_F, vnet_H, "if bad → SI is root cause"),
        ("SI_H + V_F", model_H, vnet_F, "if bad → VNet is root cause"),
        ("SI_F + V_F", model_F, vnet_F, "baseline failing"),
    ]

    results = {}
    for label, si, vn, hint in combos:
        log(f"\n  Evaluating {label} ({hint})...")
        rets = evaluate_combo(si, vn, label)
        mean_r = rets.mean()
        std_r = rets.std()
        results[label] = (mean_r, std_r, rets)
        log(f"  >> {label}: {mean_r:.1f} ±{std_r:.1f}  "
            f"[{rets.min():.0f}, {rets.max():.0f}]")

    # ── Phase 5: Diagnosis Summary ─────────────────────────────
    log("\n" + "=" * 70)
    log("DIAGNOSIS SUMMARY")
    log("=" * 70)

    hh = results["SI_H + V_H"][0]
    fh = results["SI_F + V_H"][0]
    hf = results["SI_H + V_F"][0]
    ff = results["SI_F + V_F"][0]

    si_impact = hh - fh   # drop when swapping SI to failing
    vn_impact = hh - hf   # drop when swapping VNet to failing

    log(f"  SI_H + V_H (both healthy):  {hh:.1f}")
    log(f"  SI_F + V_H (bad SI only):   {fh:.1f}  "
        f"(drop = {si_impact:.1f})")
    log(f"  SI_H + V_F (bad VNet only): {hf:.1f}  "
        f"(drop = {vn_impact:.1f})")
    log(f"  SI_F + V_F (both bad):      {ff:.1f}")

    log("")
    if si_impact > vn_impact * 2:
        log("  >>> VERDICT: SI (System Identification) is the PRIMARY root cause.")
        log("      The world model parameters are significantly misestimated,")
        log("      causing MCTS to plan against wrong physics.")
    elif vn_impact > si_impact * 2:
        log("  >>> VERDICT: VNet (Value Network) is the PRIMARY root cause.")
        log("      The value function is miscalibrated, misleading MCTS")
        log("      leaf evaluation.")
    else:
        log("  >>> VERDICT: MIXED — both SI and VNet contribute to failure.")
        log(f"      SI impact: {si_impact:.1f},  VNet impact: {vn_impact:.1f}")

    total = time.time() - t0
    log(f"\nTotal time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main()
