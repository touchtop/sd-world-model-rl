#!/usr/bin/env python3
"""Formal paper-facing runner for cartpole shared_prior experiments."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import sweep_si_robust as sweep

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / 'results' / 'si_robust_summaries'


def _build_parser():
    parser = argparse.ArgumentParser(
        description='Formal paper-facing runner for cartpole shared_prior experiments.'
    )
    parser.add_argument('--tier', default='T4', choices=sweep.TIERS.keys())
    parser.add_argument('--seeds', type=int, nargs='+', default=list(sweep.PAPER_SEEDS),
                        help='Paper seeds to finetune/evaluate')
    parser.add_argument('--dev_seeds', type=int, nargs='+', default=None,
                        help='Development seeds for shared prior pretraining')
    parser.add_argument('--mode', default='serial', choices=('serial', 'parallel'),
                        help='Execution mode for the paper-facing pipeline')
    parser.add_argument('--parallel_gpus', type=int, default=0,
                        help='Number of GPUs for staged parallel execution')
    parser.add_argument('--weight_decay', '--weight-decay', dest='weight_decay',
                        type=float, default=0.0,
                        help='Paper-facing finetune weight decay')
    parser.add_argument('--state_noise', '--state-noise', dest='state_noise',
                        type=float, default=0.0,
                        help='Paper-facing finetune state noise')
    parser.add_argument('--prior_weight_decay', '--prior-weight-decay',
                        dest='prior_weight_decay', type=float, default=1e-4,
                        help='Shared prior pretraining weight decay')
    parser.add_argument('--prior_state_noise', '--prior-state-noise',
                        dest='prior_state_noise', type=float, default=1e-3,
                        help='Shared prior pretraining state noise')
    parser.add_argument('--prior_epochs', '--prior-epochs', dest='prior_epochs',
                        type=int, default=500,
                        help='Epochs for shared prior pretraining')
    parser.add_argument('--finetune_epochs', '--finetune-epochs', dest='finetune_epochs',
                        type=int, default=500,
                        help='Epochs for per-seed finetuning')
    parser.add_argument('--selection_metric', '--selection-metric',
                        dest='selection_metric',
                        default='overall', choices=('overall', 'critical'),
                        help='Validation metric used for restart selection')
    parser.add_argument('--experiment_tag_override', '--experiment-tag-override',
                        dest='experiment_tag_override', default=None,
                        help='Override the experiment tag used for target ckpt/eval artifacts')
    parser.add_argument('--experiment_tag_suffix', '--experiment-tag-suffix',
                        dest='experiment_tag_suffix', default='',
                        help='Suffix appended to the canonical experiment tag')
    parser.add_argument('--summary_path', '--summary-path', dest='summary_path',
                        default=None,
                        help='Optional JSON summary path')
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


def _validate_args(args, dev_seeds):
    if not dev_seeds:
        raise ValueError('shared_prior paper runner requires at least one development seed')
    overlap = sorted(set(dev_seeds) & set(sweep.PAPER_SEEDS))
    if overlap:
        raise ValueError(f'dev_seeds must exclude paper seeds {overlap}')
    target_overlap = sorted(set(dev_seeds) & set(args.seeds))
    if target_overlap:
        raise ValueError(f'dev_seeds must exclude target seeds {target_overlap}')
    missing = sorted(set(dev_seeds) - set(sweep.SEEDS))
    if missing:
        raise ValueError(f'dev_seeds must be a subset of configured seeds, got {missing}')
    if args.mode == 'parallel' and args.parallel_gpus < 2:
        raise ValueError('parallel mode requires --parallel_gpus >= 2')


def _prior_source_info(tier, dev_seeds, prior_tag, prior_weight_decay,
                       prior_state_noise, prior_epochs):
    prior_ckpt = sweep._prior_ckpt_path(tier, prior_tag)
    if os.path.exists(prior_ckpt):
        return {
            'reused': True,
            'source_kind': 'canonical',
            'source_path': prior_ckpt,
            'canonical_path': prior_ckpt,
        }
    for legacy_path in sweep._legacy_prior_ckpt_paths(
        tier, dev_seeds, prior_weight_decay, prior_state_noise, prior_epochs
    ):
        if os.path.exists(legacy_path):
            return {
                'reused': True,
                'source_kind': 'legacy',
                'source_path': legacy_path,
                'canonical_path': prior_ckpt,
            }
    return {
        'reused': False,
        'source_kind': 'new_train',
        'source_path': prior_ckpt,
        'canonical_path': prior_ckpt,
    }


def _summary_path(args, experiment_tag):
    if args.summary_path:
        return Path(args.summary_path).resolve()
    return (SUMMARY_DIR / f'{args.tier}_{experiment_tag}_summary.json').resolve()


def _experiment_tag(args, prior_tag):
    if args.experiment_tag_override:
        return args.experiment_tag_override
    tag = sweep._shared_experiment_tag(
        prior_tag,
        args.weight_decay,
        args.state_noise,
        args.finetune_epochs,
        args.selection_metric,
    )
    return f'{tag}{args.experiment_tag_suffix}'


def _stage_logs(tier, dev_seeds, seeds, experiment_tag):
    return {
        'train_prior': sweep._stage_log_path(tier, 'train_prior', prior_tag=experiment_tag),
        'prepare_dev': {
            seed: sweep._stage_log_path(tier, 'prepare_dev', seed=seed, prior_tag=experiment_tag)
            for seed in dev_seeds
        },
        'prepare_target': {
            seed: sweep._stage_log_path(tier, 'prepare_target', seed=seed, prior_tag=experiment_tag)
            for seed in seeds
        },
        'finetune': {
            seed: sweep._stage_log_path(tier, 'finetune', seed=seed, prior_tag=experiment_tag)
            for seed in seeds
        },
        'eval': {
            seed: sweep._stage_log_path(tier, 'eval', seed=seed, prior_tag=experiment_tag)
            for seed in seeds
        },
    }


def _load_seed_artifacts(tier, seed, experiment_tag, weight_decay, state_noise):
    ckpt_path = sweep._ckpt_path(
        tier, seed, weight_decay, state_noise,
        selector_mode='shared_prior', prior_tag=experiment_tag,
    )
    eval_path = sweep._eval_path(
        tier, seed, weight_decay, state_noise,
        selector_mode='shared_prior', prior_tag=experiment_tag,
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'missing checkpoint artifact: {ckpt_path}')
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f'missing eval artifact: {eval_path}')
    ckpt_state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    eval_state = torch.load(eval_path, map_location='cpu', weights_only=False)
    returns = np.array(eval_state['returns'], dtype=np.float32)
    params = ckpt_state['params']
    mp_err = abs(params['masspole'] - 0.1) / 0.1 * 100
    return {
        'seed': seed,
        'checkpoint_path': ckpt_path,
        'eval_path': eval_path,
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
    }


def _aggregate(seed_results):
    means = np.array([row['eval']['mean'] for row in seed_results], dtype=np.float32)
    if len(means) == 0:
        return {
            'mean': None,
            'median': None,
            'iqm': None,
            'num_below_700': 0,
            'count': 0,
        }
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


def _print_header(args, cfg, dev_seeds, prior_tag, experiment_tag, summary_path):
    sweep.log('=' * 70)
    sweep.log('CARTPOLE SHARED_PRIOR PAPER RUNNER')
    sweep.log('=' * 70)
    sweep.log(f'  Tier: {args.tier}')
    sweep.log(f'  Mode: {args.mode}')
    sweep.log(f'  Config: {cfg}')
    sweep.log(f'  Paper seeds: {list(args.seeds)}')
    sweep.log(f'  Dev seeds: {dev_seeds}')
    sweep.log(f'  Finetune weight_decay: {args.weight_decay:g}')
    sweep.log(f'  Finetune state_noise: {args.state_noise:g}')
    sweep.log(f'  Prior weight_decay: {args.prior_weight_decay:g}')
    sweep.log(f'  Prior state_noise: {args.prior_state_noise:g}')
    sweep.log(f'  Prior epochs: {args.prior_epochs}')
    sweep.log(f'  Finetune epochs: {args.finetune_epochs}')
    sweep.log(f'  Selection metric: {args.selection_metric}')
    sweep.log(f'  Prior tag: {prior_tag}')
    sweep.log(f'  Experiment tag: {experiment_tag}')
    sweep.log(f'  Summary path: {summary_path}')
    sweep.log('=' * 70)


def _print_seed_summary(seed_results, aggregate):
    sweep.log('\n' + '=' * 70)
    sweep.log('PAPER RESULTS SUMMARY')
    sweep.log('=' * 70)
    header = (
        f"{'Seed':>6s}  {'Return':>7s}  {'±Std':>7s}  {'Val':>7s}  "
        f"{'mp':>7s}  {'mp_err%':>8s}"
    )
    sweep.log(header)
    sweep.log('-' * len(header))
    for row in seed_results:
        sweep.log(
            f"{row['seed']:>6d}  {row['eval']['mean']:>7.1f}  {row['eval']['std']:>7.1f}  "
            f"{row['val_loss']:>7.4f}  {row['si_params']['masspole']:>7.3f}  "
            f"{row['mp_err_percent']:>7.1f}%"
        )
    sweep.log('-' * len(header))
    sweep.log(
        f"Mean={aggregate['mean']:.1f}  Median={aggregate['median']:.1f}  "
        f"IQM={aggregate['iqm']:.1f}  <700={aggregate['num_below_700']}/{aggregate['count']}"
    )


def _run_serial(args, cfg, dev_seeds, prior_tag, experiment_tag):
    dev_payloads = []
    for seed in dev_seeds:
        sweep.log(f"\n{'=' * 24} Preparing Dev Seed {seed} {'=' * 24}")
        sweep.prepare_seed_payload_cache(seed, cfg, args.tier)
        dev_payloads.append(sweep._load_seed_payload(args.tier, seed))

    prior_state = sweep._train_shared_prior(
        dev_payloads,
        args.tier,
        prior_tag,
        args.prior_weight_decay,
        args.prior_state_noise,
        args.prior_epochs,
    )

    seed_runtime = {}
    for seed in args.seeds:
        t_seed = time.time()
        sweep.log(f"\n{'=' * 30} Training Seed {seed} {'=' * 30}")
        sweep.prepare_seed_payload_cache(seed, cfg, args.tier)
        payload = sweep._load_seed_payload(args.tier, seed)
        model, vnet, actions, params, selected_cfg = sweep.train_seed(
            seed,
            cfg,
            tier=args.tier,
            weight_decay=args.weight_decay,
            state_noise=args.state_noise,
            selector_mode='shared_prior',
            prior_state=prior_state,
            prior_tag=experiment_tag,
            finetune_epochs=args.finetune_epochs,
            payload=payload,
            selection_metric=args.selection_metric,
        )
        returns = sweep.evaluate_seed(
            seed,
            model,
            vnet,
            actions,
            tier=args.tier,
            weight_decay=selected_cfg['weight_decay'],
            state_noise=selected_cfg['state_noise'],
            selector_mode='shared_prior',
            prior_tag=selected_cfg.get('prior_tag'),
            payload=payload,
        )
        sweep.log(
            f"  Seed {seed}: {returns.mean():.1f} ±{returns.std():.1f}  "
            f"[{returns.min():.0f}, {returns.max():.0f}]"
        )
        seed_runtime[seed] = float(time.time() - t_seed)
    return seed_runtime


def _run_parallel(args, cfg, dev_seeds, prior_tag, experiment_tag):
    gpu_count = max(1, args.parallel_gpus)
    sweep.log(f'Launching paper shared-prior pipeline with {gpu_count} parallel workers')

    def launch_group(stage, stage_seeds, extra_args=None, log_stage=None):
        procs = []
        extra_args = extra_args or []
        log_stage = log_stage or stage
        for idx, seed in enumerate(stage_seeds):
            device = f'cuda:{idx % gpu_count}' if args.parallel_gpus > 0 else None
            log_path = sweep._stage_log_path(args.tier, log_stage, seed=seed, prior_tag=experiment_tag)
            cmd = [
                sys.executable, os.path.abspath(sweep.__file__),
                '--tier', args.tier,
                '--selector_mode', 'shared_prior',
                '--stage', stage,
                '--stage_seed', str(seed),
                '--prior_weight_decay', str(args.prior_weight_decay),
                '--prior_state_noise', str(args.prior_state_noise),
                '--prior_epochs', str(args.prior_epochs),
                '--weight_decay', str(args.weight_decay),
                '--state_noise', str(args.state_noise),
                '--finetune_epochs', str(args.finetune_epochs),
                '--selection_metric', args.selection_metric,
                '--parallel_gpus', str(args.parallel_gpus),
                *extra_args,
            ]
            procs.append((
                sweep._run_subprocess(cmd, log_path, env=sweep._device_env(device)),
                f'{log_stage} seed={seed}',
                log_path,
            ))
        sweep._wait_for_group(procs)

    launch_group('prepare_dev', dev_seeds)

    prior_log = sweep._stage_log_path(args.tier, 'train_prior', prior_tag=experiment_tag)
    prior_cmd = [
        sys.executable, os.path.abspath(sweep.__file__),
        '--tier', args.tier,
        '--selector_mode', 'shared_prior',
        '--stage', 'train_prior',
        '--dev_seeds', *[str(seed) for seed in dev_seeds],
        '--prior_weight_decay', str(args.prior_weight_decay),
        '--prior_state_noise', str(args.prior_state_noise),
        '--prior_epochs', str(args.prior_epochs),
        '--weight_decay', str(args.weight_decay),
        '--state_noise', str(args.state_noise),
        '--finetune_epochs', str(args.finetune_epochs),
        '--selection_metric', args.selection_metric,
        '--parallel_gpus', str(args.parallel_gpus),
    ]
    sweep._wait_for_group([(
        sweep._run_subprocess(
            prior_cmd,
            prior_log,
            env=sweep._device_env('cuda:0' if args.parallel_gpus > 0 else None),
        ),
        'train_prior',
        prior_log,
    )])

    launch_group('prepare_dev', args.seeds, log_stage='prepare_target')
    launch_group('finetune_target', args.seeds, ['--prior_tag_override', experiment_tag], log_stage='finetune')
    launch_group('eval_target', args.seeds, ['--prior_tag_override', experiment_tag], log_stage='eval')
    return {}


def main():
    args = _build_parser().parse_args()
    cfg = sweep.TIERS[args.tier]
    dev_seeds = args.dev_seeds if args.dev_seeds is not None else sweep._default_dev_seeds(sweep.SEEDS)
    _validate_args(args, dev_seeds)

    prior_tag = sweep._prior_tag(
        dev_seeds,
        args.prior_weight_decay,
        args.prior_state_noise,
        args.prior_epochs,
    )
    experiment_tag = _experiment_tag(args, prior_tag)
    summary_path = _summary_path(args, experiment_tag)
    prior_info = _prior_source_info(
        args.tier,
        dev_seeds,
        prior_tag,
        args.prior_weight_decay,
        args.prior_state_noise,
        args.prior_epochs,
    )

    _print_header(args, cfg, dev_seeds, prior_tag, experiment_tag, summary_path)

    target_cache_before = {
        seed: {
            'checkpoint_reused': os.path.exists(sweep._ckpt_path(
                args.tier, seed, args.weight_decay, args.state_noise,
                selector_mode='shared_prior', prior_tag=experiment_tag,
            )),
            'eval_reused': os.path.exists(sweep._eval_path(
                args.tier, seed, args.weight_decay, args.state_noise,
                selector_mode='shared_prior', prior_tag=experiment_tag,
            )),
        }
        for seed in args.seeds
    }

    t0 = time.time()
    if args.mode == 'serial':
        seed_runtime = _run_serial(args, cfg, dev_seeds, prior_tag, experiment_tag)
    else:
        seed_runtime = _run_parallel(args, cfg, dev_seeds, prior_tag, experiment_tag)

    prior_state = torch.load(prior_info['source_path'], map_location='cpu', weights_only=False)
    seed_results = []
    for seed in args.seeds:
        row = _load_seed_artifacts(
            args.tier,
            seed,
            experiment_tag,
            args.weight_decay,
            args.state_noise,
        )
        row.update(target_cache_before[seed])
        if seed in seed_runtime:
            row['runtime_sec'] = seed_runtime[seed]
        seed_results.append(row)

    aggregate = _aggregate(seed_results)
    _print_seed_summary(seed_results, aggregate)

    summary = {
        'runner': 'run_cartpole_sharedprior_paper.py',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': args.mode,
        'tier': args.tier,
        'paper_seeds': list(args.seeds),
        'dev_seeds': dev_seeds,
        'selector_mode': 'shared_prior',
        'selection_metric': args.selection_metric,
        'finetune': {
            'weight_decay': args.weight_decay,
            'state_noise': args.state_noise,
            'epochs': args.finetune_epochs,
            'experiment_tag': experiment_tag,
        },
        'prior': {
            'prior_tag': prior_tag,
            'weight_decay': args.prior_weight_decay,
            'state_noise': args.prior_state_noise,
            'epochs': args.prior_epochs,
            'reused': prior_info['reused'],
            'source_kind': prior_info['source_kind'],
            'source_path': prior_info['source_path'],
            'canonical_path': prior_info['canonical_path'],
            'train_loss': float(prior_state['train_loss']),
            'val_loss': float(prior_state['val_loss']),
            'dev_seeds': _jsonable(prior_state['dev_seeds']),
            'n_samples': int(prior_state['n_samples']),
        },
        'per_seed': seed_results,
        'aggregate': aggregate,
        'artifacts': {
            'summary_path': str(summary_path),
            'stage_logs': _stage_logs(args.tier, dev_seeds, args.seeds, experiment_tag)
            if args.mode == 'parallel' else None,
        },
        'total_runtime_sec': float(time.time() - t0),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    sweep.log(f'\nSummary saved to {summary_path}')
    sweep.log(f'Total time: {summary["total_runtime_sec"]:.1f}s ({summary["total_runtime_sec"] / 60:.1f} min)')


if __name__ == '__main__':
    main()
