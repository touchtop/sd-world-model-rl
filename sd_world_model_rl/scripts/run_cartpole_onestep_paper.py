#!/usr/bin/env python3
"""Paper-facing 1-step no-MCTS baseline for Cartpole shared-prior checkpoints."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dm_control import suite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_dmcontrol_cartpole as cartpole
from scripts import sweep_si_robust as sweep

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / 'results' / 'si_robust_summaries'


def _build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate 1-step no-MCTS baseline using formal shared-prior checkpoints.'
    )
    parser.add_argument('--tier', default='T4', choices=sweep.TIERS.keys())
    parser.add_argument('--seeds', type=int, nargs='+', default=list(sweep.PAPER_SEEDS))
    parser.add_argument('--experiment_tag', required=True,
                        help='Existing formal shared-prior experiment tag to load checkpoints from')
    parser.add_argument('--summary_path', default=None,
                        help='Optional JSON summary output path')
    parser.add_argument('--action_bins', type=int, default=11,
                        help='Discrete action bins for one-step lookahead')
    parser.add_argument('--action_repeat', type=int, default=sweep.ACTION_REPEAT,
                        help='Action repeat for env execution and one-step forward')
    parser.add_argument('--eval_episodes', type=int, default=sweep.EVAL_EPISODES,
                        help='Evaluation episodes per seed')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Max env steps per episode')
    return parser


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _summary_path(args):
    if args.summary_path:
        return Path(args.summary_path).resolve()
    name = f'{args.tier}_{args.experiment_tag}_onestep_summary.json'
    return (SUMMARY_DIR / name).resolve()


def _load_checkpoint(tier, seed, experiment_tag):
    ckpt_path = sweep._ckpt_path(
        tier,
        seed,
        weight_decay=0.0,
        state_noise=0.0,
        selector_mode='shared_prior',
        prior_tag=experiment_tag,
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'missing checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = cartpole.DMControlCartPoleSDModel(
        learn_params=True,
        prior_noise=0.3,
        mp_prior_noise=0.15,
        mp_clamp_min=0.03,
    )
    model.load_state_dict(state['model'])
    vnet = cartpole.SwingupValueNet()
    vnet.load_state_dict(state['vnet'])
    vnet.eval()
    return ckpt_path, state, model, vnet


def _one_step_action(state_np, wrapper, actions, value_net, action_repeat):
    root = torch.tensor(state_np, dtype=torch.float32)
    gamma_k = 0.99 ** action_repeat
    candidates = []
    for action in actions:
        next_state, edge_reward, _ = cartpole.model_forward_k(
            wrapper, root, float(action), action_repeat, gamma=0.99,
        )
        with torch.no_grad():
            next_v = float(value_net(next_state.unsqueeze(0)).item())
        score = edge_reward + gamma_k * next_v
        candidates.append((score, float(action), float(edge_reward), next_v))
    candidates.sort(key=lambda row: (-row[0], -row[2]))
    return candidates[0][1]


def _evaluate_seed_onestep(seed, model, vnet, actions, action_repeat, eval_episodes, max_steps):
    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed + 1000})
    wrapper = cartpole.DMControlCartPoleWrapper(model)
    returns = []
    for _ in range(eval_episodes):
        ts = env.reset()
        obs = cartpole.flatten_obs(ts.observation)
        state = cartpole.obs_to_state(obs)
        ep_ret = 0.0
        env_step = 0
        while env_step < max_steps:
            action_val = _one_step_action(state, wrapper, actions, vnet, action_repeat)
            for _ in range(action_repeat):
                ts = env.step(np.array([action_val], dtype=np.float32))
                ep_ret += (ts.reward if ts.reward is not None else 0.0)
                env_step += 1
                if ts.last() or env_step >= max_steps:
                    break
            if ts.last():
                break
            obs = cartpole.flatten_obs(ts.observation)
            state = cartpole.obs_to_state(obs)
        returns.append(ep_ret)
    return np.array(returns, dtype=np.float32)


def _aggregate(seed_results):
    means = np.array([row['eval']['mean'] for row in seed_results], dtype=np.float32)
    sorted_means = sorted(float(m) for m in means)
    q1 = len(sorted_means) // 4
    q3 = len(sorted_means) - len(sorted_means) // 4
    iqm = float(np.mean(sorted_means[q1:q3])) if q3 > q1 else float(means.mean())
    return {
        'mean': float(means.mean()),
        'median': float(np.median(means)),
        'iqm': iqm,
        'num_below_700': int(sum(float(m) < 700.0 for m in means)),
        'count': int(len(means)),
    }


def _print_header(args, summary_path):
    sweep.log('=' * 70)
    sweep.log('CARTPOLE 1-STEP NO-MCTS PAPER BASELINE')
    sweep.log('=' * 70)
    sweep.log(f'  Tier: {args.tier}')
    sweep.log(f'  Paper seeds: {list(args.seeds)}')
    sweep.log(f'  Source experiment tag: {args.experiment_tag}')
    sweep.log(f'  Action bins: {args.action_bins}')
    sweep.log(f'  Action repeat: {args.action_repeat}')
    sweep.log(f'  Eval episodes: {args.eval_episodes}')
    sweep.log(f'  Max steps: {args.max_steps}')
    sweep.log(f'  Summary path: {summary_path}')
    sweep.log('=' * 70)


def _print_seed_summary(seed_results, aggregate):
    sweep.log('\n' + '=' * 70)
    sweep.log('1-STEP NO-MCTS RESULTS SUMMARY')
    sweep.log('=' * 70)
    header = f"{'Seed':>6s}  {'Return':>7s}  {'±Std':>7s}  {'Val':>7s}  {'mp':>7s}  {'mp_err%':>8s}"
    sweep.log(header)
    sweep.log('-' * len(header))
    for row in seed_results:
        sweep.log(
            f"{row['seed']:>6d}  {row['eval']['mean']:>7.1f}  {row['eval']['std']:>7.1f}  "
            f"{row['val_loss']:>7.4f}  {row['si_params']['masspole']:>7.3f}  {row['mp_err_percent']:>7.1f}%"
        )
    sweep.log('-' * len(header))
    sweep.log(
        f"Mean={aggregate['mean']:.1f}  Median={aggregate['median']:.1f}  "
        f"IQM={aggregate['iqm']:.1f}  <700={aggregate['num_below_700']}/{aggregate['count']}"
    )


def main():
    args = _build_parser().parse_args()
    summary_path = _summary_path(args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    actions = cartpole.get_discrete_actions(args.action_bins)

    _print_header(args, summary_path)

    t0 = time.time()
    seed_results = []
    for seed in args.seeds:
        sweep.log(f"\n{'=' * 30} Evaluating Seed {seed} {'=' * 30}")
        ckpt_path, ckpt_state, model, vnet = _load_checkpoint(args.tier, seed, args.experiment_tag)
        returns = _evaluate_seed_onestep(
            seed,
            model,
            vnet,
            actions,
            args.action_repeat,
            args.eval_episodes,
            args.max_steps,
        )
        params = ckpt_state['params']
        mp_err = abs(params['masspole'] - 0.1) / 0.1 * 100
        seed_results.append({
            'seed': int(seed),
            'checkpoint_path': ckpt_path,
            'selected_cfg': _jsonable(ckpt_state.get('selected_cfg') or {}),
            'si_params': _jsonable(params),
            'mp_err_percent': float(mp_err),
            'val_loss': float(ckpt_state['val_loss']),
            'eval': {
                'returns': returns.tolist(),
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'min': float(returns.min()),
                'max': float(returns.max()),
            },
        })
        sweep.log(
            f"  Seed {seed}: {returns.mean():.1f} ±{returns.std():.1f}  "
            f"[{returns.min():.0f}, {returns.max():.0f}]"
        )

    aggregate = _aggregate(seed_results)
    _print_seed_summary(seed_results, aggregate)

    summary = {
        'runner': 'run_cartpole_onestep_paper.py',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tier': args.tier,
        'paper_seeds': list(args.seeds),
        'source_experiment_tag': args.experiment_tag,
        'policy': {
            'name': 'one_step_no_mcts',
            'action_bins': int(args.action_bins),
            'action_repeat': int(args.action_repeat),
            'score_rule': 'immediate_reward + gamma^k * V(next_state)',
        },
        'evaluation': {
            'eval_episodes': int(args.eval_episodes),
            'max_steps': int(args.max_steps),
        },
        'per_seed': seed_results,
        'aggregate': aggregate,
        'artifacts': {
            'summary_path': str(summary_path),
        },
        'total_runtime_sec': float(time.time() - t0),
    }
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    sweep.log(f'\nSummary saved to {summary_path}')
    sweep.log(f'Total time: {summary["total_runtime_sec"]:.1f}s')


if __name__ == '__main__':
    main()
