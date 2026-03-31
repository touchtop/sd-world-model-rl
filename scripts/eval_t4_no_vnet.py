#!/usr/bin/env python3
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from scripts.run_dmcontrol_cartpole import evaluate_agent, get_discrete_actions, log
from src.world_model.dmcontrol_cartpole_sd_model import DMControlCartPoleSDModel
from scripts.sweep_si_robust import _ckpt_path

SEEDS = [0, 10, 21, 42, 84]
TIER = 'T4'
N_EPISODES = 5
NUM_SIMS = 200
ACTION_REPEAT = 2
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'si_robust_eval', 'T4_no_vnet_5seeds.json')


def load_model(seed):
    ckpt = torch.load(_ckpt_path(TIER, seed), map_location='cpu', weights_only=False)
    model = DMControlCartPoleSDModel(
        learn_params=True, prior_noise=0.3,
        mp_prior_noise=0.15,
        mp_clamp_min=0.03,
    )
    model.load_state_dict(ckpt['model'])
    return model, ckpt['params']


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    actions = get_discrete_actions(11)
    rows = []

    log('=' * 70)
    log('T4 + no-VNet evaluation (5 seeds)')
    log(f'  Seeds: {SEEDS}')
    log('=' * 70)

    for seed in SEEDS:
        model, params = load_model(seed)
        g_over_l = model.g.item() / max(model.l.item(), 1e-3)
        env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed + 1000})
        returns = evaluate_agent(
            env, model, actions,
            n_episodes=N_EPISODES,
            num_simulations=NUM_SIMS,
            value_net=None,
            action_repeat=ACTION_REPEAT,
            verbose=False,
            g_over_l=g_over_l,
        )
        row = {
            'seed': seed,
            'returns': returns.tolist(),
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'min': float(returns.min()),
            'max': float(returns.max()),
            'masscart': float(params['masscart']),
            'masspole': float(params['masspole']),
            'length': float(params['length']),
        }
        rows.append(row)
        log(f"Seed {seed}: {returns.mean():.1f} ±{returns.std():.1f} [{returns.min():.0f}, {returns.max():.0f}]")

    means = np.array([r['mean'] for r in rows], dtype=np.float32)
    sorted_m = np.sort(means)
    q1, q3 = len(sorted_m) // 4, len(sorted_m) - len(sorted_m) // 4
    iqm = float(sorted_m[q1:q3].mean()) if q3 > q1 else float(means.mean())
    summary = {
        'tier': TIER,
        'mode': 'no_vnet',
        'seeds': SEEDS,
        'n_episodes': N_EPISODES,
        'num_sims': NUM_SIMS,
        'rows': rows,
        'aggregate': {
            'mean': float(means.mean()),
            'median': float(np.median(means)),
            'iqm': iqm,
            'n_below_700': int(np.sum(means < 700)),
        },
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    log('-' * 70)
    log(f"Mean={summary['aggregate']['mean']:.1f}  Median={summary['aggregate']['median']:.1f}  IQM={summary['aggregate']['iqm']:.1f}  <700={summary['aggregate']['n_below_700']}/{len(SEEDS)}")
    log(f'Results saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
