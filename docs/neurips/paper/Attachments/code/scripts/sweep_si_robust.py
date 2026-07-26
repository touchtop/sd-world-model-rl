#!/usr/bin/env python3
"""Sweep SI robustness interventions for Cartpole-Swingup.

Incremental testing of 4 interventions to improve masspole identifiability:
  T0: Baseline (all off) — reproduce sweep_robust.log (~615 mean)
  T1: Excitation-aware data collection (mixed policy)
  T2: T1 + theta_dot weighted SI loss
  T3: T2 + tighter mp prior (clamp_min=0.03, noise ±15%)
  T4: T3 + two-phase SI (freeze mp first half)

Usage:
    python -u scripts/sweep_si_robust.py --tier T4 2>&1 | tee /tmp/sweep_si_T4.log
    python -u scripts/sweep_si_robust.py --tier T1 --si-only 2>&1 | tee /tmp/sweep_si_T1_si.log
"""
import sys, os, time, argparse, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from scripts.run_dmcontrol_cartpole import (
    DMControlCartPoleSDModel, DMControlCartPoleWrapper,
    SwingupValueNet, get_discrete_actions, flatten_obs, obs_to_state,
    collect_trajectories, train_sd_model, generate_imagination_data,
    train_value_net, train_vnet_with_restarts, evaluate_agent,
    model_forward_k, mcts_search_swingup, log,
)

SEEDS = [0, 10, 21, 30, 42, 55, 63, 72, 84, 95]
BUDGET = 10000
NUM_SIMS = 200
ACTION_REPEAT = 2
EVAL_EPISODES = 5
N_RESTARTS = 5
VAL_FRACTION = 0.15
SELECTOR_TIE_TOL = 0.01
ADAPTIVE_CANDIDATES = ((1e-4, 0.0), (1e-4, 1e-3))
ADAPTIVE_TAG = 'adaptivev1'
PAPER_SEEDS = (0, 42, 84)
SHARED_PRIOR_TAG = 'sharedpriorv1'

# Tier definitions
TIERS = {
    'T0': dict(explore_policy='random', thdot_weight=1.0,
               energy_weighted=False, mp_prior_noise=None,
               mp_clamp_min=0.01, two_phase_mp=False),
    'T1': dict(explore_policy='mixed', thdot_weight=1.0,
               energy_weighted=False, mp_prior_noise=None,
               mp_clamp_min=0.01, two_phase_mp=False),
    'T2': dict(explore_policy='mixed', thdot_weight=5.0,
               energy_weighted=True, mp_prior_noise=None,
               mp_clamp_min=0.01, two_phase_mp=False),
    'T3': dict(explore_policy='mixed', thdot_weight=5.0,
               energy_weighted=True, mp_prior_noise=0.15,
               mp_clamp_min=0.03, two_phase_mp=False),
    'T4': dict(explore_policy='mixed', thdot_weight=5.0,
               energy_weighted=True, mp_prior_noise=0.15,
               mp_clamp_min=0.03, two_phase_mp=True),
}


CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'si_robust_ckpts')
EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'si_robust_eval')
PAYLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'si_robust_payloads')


def _float_tag(value):
    return f'{value:g}'.replace('-', 'm').replace('.', 'p')


def _seed_list_tag(seeds):
    return '-'.join(str(seed) for seed in seeds)


def _exp_tag(weight_decay=None, state_noise=None, selector_mode='single',
             prior_tag=None):
    if selector_mode == 'adaptive':
        return ADAPTIVE_TAG
    if selector_mode == 'shared_prior':
        return prior_tag if prior_tag is not None else SHARED_PRIOR_TAG
    return f'wd{_float_tag(weight_decay)}_sn{_float_tag(state_noise)}'


def _prior_tag(dev_seeds, prior_weight_decay, prior_state_noise,
               prior_epochs):
    return (
        f'{SHARED_PRIOR_TAG}_dev{_seed_list_tag(dev_seeds)}'
        f'_pwd{_float_tag(prior_weight_decay)}'
        f'_psn{_float_tag(prior_state_noise)}'
        f'_pe{prior_epochs}'
    )


def _shared_experiment_tag(prior_tag, finetune_weight_decay,
                           finetune_state_noise, finetune_epochs,
                           selection_metric='overall'):
    return (
        f'{prior_tag}'
        f'_fwd{_float_tag(finetune_weight_decay)}'
        f'_fsn{_float_tag(finetune_state_noise)}'
        f'_fe{finetune_epochs}'
        f'_sel{selection_metric}'
    )


def _ckpt_path(tier, seed, weight_decay=None, state_noise=None,
               selector_mode='single', prior_tag=None):
    return os.path.join(
        CKPT_DIR,
        f'{tier}_{_exp_tag(weight_decay, state_noise, selector_mode, prior_tag)}_seed{seed}.pt')


def _eval_path(tier, seed, weight_decay=None, state_noise=None,
               selector_mode='single', prior_tag=None):
    return os.path.join(
        EVAL_DIR,
        f'{tier}_{_exp_tag(weight_decay, state_noise, selector_mode, prior_tag)}_seed{seed}_eval.pt')


def _prior_ckpt_path(tier, prior_tag):
    return os.path.join(CKPT_DIR, f'{tier}_{prior_tag}_prior.pt')


def _legacy_prior_ckpt_paths(tier, dev_seeds, prior_weight_decay,
                             prior_state_noise, prior_epochs):
    legacy_base = (
        f'{SHARED_PRIOR_TAG}_dev{_seed_list_tag(dev_seeds)}'
        f'_pwd{_float_tag(prior_weight_decay)}'
        f'_psn{_float_tag(prior_state_noise)}'
        f'_pe{prior_epochs}'
    )
    return [
        os.path.join(
            CKPT_DIR,
            f'{tier}_{legacy_base}_fwd{_float_tag(prior_weight_decay)}'
            f'_fsn{_float_tag(prior_state_noise)}_fe500_prior.pt'),
        os.path.join(
            CKPT_DIR,
            f'{tier}_{legacy_base}_fwd{_float_tag(prior_weight_decay)}'
            f'_fsn{_float_tag(prior_state_noise)}_fe500_seloverall_prior.pt'),
    ]


def _payload_path(tier, seed):
    return os.path.join(PAYLOAD_DIR, f'{tier}_seed{seed}_payload.pt')


def _stage_log_path(tier, stage, seed=None, prior_tag=None):
    suffix = f'_seed{seed}' if seed is not None else ''
    tag = f'_{prior_tag}' if prior_tag is not None else ''
    return os.path.join('/tmp', f'si_robust_{tier}_{stage}{suffix}{tag}.log')


def _device_env(device):
    env = os.environ.copy()
    if device is not None and device.startswith('cuda:'):
        env['CUDA_VISIBLE_DEVICES'] = device.split(':', 1)[1]
    return env


def _train_single_candidate(seed, imag_states, imag_returns, weight_decay,
                            state_noise, log_prefix, epochs=500,
                            init_state_dict=None,
                            selection_metric='overall'):
    vnet, train_loss, val_loss, val_metrics = train_vnet_with_restarts(
        imag_states, imag_returns, seed=seed, hidden=128,
        epochs=epochs, lr=1e-3, min_loss=0.05,
        weight_decay=weight_decay, state_noise=state_noise,
        val_fraction=VAL_FRACTION, n_restarts=N_RESTARTS,
        log_prefix=log_prefix, init_state_dict=init_state_dict,
        selection_metric=selection_metric)
    return {
        'vnet': vnet,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'weight_decay': weight_decay,
        'state_noise': state_noise,
        'selection_metric': selection_metric,
    }


def _default_dev_seeds(seeds):
    return [seed for seed in seeds if seed not in PAPER_SEEDS]


def _prepare_seed_payload(seed, cfg):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = suite.load('cartpole', 'swingup', task_kwargs={'random': seed})
    actions = get_discrete_actions(11)

    model = DMControlCartPoleSDModel(
        learn_params=True, prior_noise=0.3,
        mp_prior_noise=cfg['mp_prior_noise'],
        mp_clamp_min=cfg['mp_clamp_min'],
    )
    with torch.no_grad():
        model._gravity_raw.data = torch.tensor(
            model._inv_softplus(9.81), dtype=torch.float32)
    model._gravity_raw.requires_grad_(False)

    explore_steps = min(BUDGET, max(500, BUDGET // 2))
    data = collect_trajectories(env, explore_steps,
                                policy=cfg['explore_policy'])
    log(f"  [Seed {seed}] Collected {explore_steps} steps "
        f"({data['n_episodes']} ep)")

    max_thdot = torch.abs(data['states'][:, 3]).max().item()
    log(f"  [Seed {seed}] max|θ̇| in data: {max_thdot:.1f}")

    train_loss = train_sd_model(
        model, data, epochs=800, lr=5e-3,
        use_sincos=True, multistep=10,
        thdot_weight=cfg['thdot_weight'],
        energy_weighted=cfg['energy_weighted'],
        two_phase_mp=cfg['two_phase_mp'],
    )
    params = model.get_learned_params()
    mp_err = abs(params['masspole'] - 0.1) / 0.1 * 100
    log(f"  [Seed {seed}] SI: mc={params['masscart']:.3f}, "
        f"mp={params['masspole']:.3f} (err={mp_err:.1f}%), "
        f"l={params['length']:.3f} (loss={train_loss:.6f})")

    wrapper = DMControlCartPoleWrapper(model)
    imag_states, imag_returns = generate_imagination_data(
        wrapper, action_repeat=ACTION_REPEAT,
        n_trajectories=2000, horizon=60,
        real_states=data['states'],
    )
    log(f"  [Seed {seed}] R1: {len(imag_states)} samples, "
        f"returns [{imag_returns.min():.1f}, {imag_returns.max():.1f}]")
    return {
        'seed': seed,
        'model': model.state_dict(),
        'actions': actions.tolist(),
        'params': params,
        'imag_states': imag_states,
        'imag_returns': imag_returns,
    }


def _load_seed_payload(tier, seed):
    payload_path = _payload_path(tier, seed)
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f'missing payload cache: {payload_path}')
    payload = torch.load(payload_path, map_location='cpu', weights_only=False)
    model = DMControlCartPoleSDModel(
        learn_params=True, prior_noise=0.3,
        mp_prior_noise=payload['cfg']['mp_prior_noise'],
        mp_clamp_min=payload['cfg']['mp_clamp_min'],
    )
    model.load_state_dict(payload['model'])
    return {
        'seed': payload['seed'],
        'model': model,
        'actions': np.array(payload['actions'], dtype=np.float32),
        'params': payload['params'],
        'imag_states': payload['imag_states'],
        'imag_returns': payload['imag_returns'],
    }


def prepare_seed_payload_cache(seed, cfg, tier):
    payload_path = _payload_path(tier, seed)
    if os.path.exists(payload_path):
        log(f"  [Seed {seed}] Payload cache exists: {payload_path}")
        return payload_path
    payload = _prepare_seed_payload(seed, cfg)
    os.makedirs(PAYLOAD_DIR, exist_ok=True)
    torch.save({
        'seed': seed,
        'cfg': cfg,
        'model': payload['model'],
        'actions': payload['actions'],
        'params': payload['params'],
        'imag_states': payload['imag_states'],
        'imag_returns': payload['imag_returns'],
    }, payload_path)
    log(f"  [Seed {seed}] Payload cache saved to {payload_path}")
    return payload_path


def _train_shared_prior(dev_payloads, tier, prior_tag, prior_weight_decay,
                        prior_state_noise, prior_epochs):
    prior_ckpt = _prior_ckpt_path(tier, prior_tag)
    if os.path.exists(prior_ckpt):
        log(f"Loading shared prior {prior_ckpt}")
        state = torch.load(prior_ckpt, map_location='cpu', weights_only=False)
        return state

    dev_seed_ids = None
    if dev_payloads:
        if isinstance(dev_payloads[0], int):
            dev_seed_ids = list(dev_payloads)
        else:
            dev_seed_ids = [payload['seed'] for payload in dev_payloads]
    if dev_seed_ids is not None:
        legacy_paths = _legacy_prior_ckpt_paths(
            tier, dev_seed_ids, prior_weight_decay,
            prior_state_noise, prior_epochs)
        for legacy_path in legacy_paths:
            if os.path.exists(legacy_path):
                log(f"Loading shared prior {legacy_path}")
                return torch.load(legacy_path, map_location='cpu', weights_only=False)
    if dev_payloads and isinstance(dev_payloads[0], int):
        dev_payloads = [_load_seed_payload(tier, seed) for seed in dev_payloads]

    imag_states = torch.cat([payload['imag_states'] for payload in dev_payloads], dim=0)
    imag_returns = np.concatenate([payload['imag_returns'] for payload in dev_payloads])
    log(f"Training shared prior on {len(dev_payloads)} dev seeds, "
        f"{len(imag_states)} total samples")
    prior = _train_single_candidate(
        seed=0,
        imag_states=imag_states,
        imag_returns=imag_returns,
        weight_decay=prior_weight_decay,
        state_noise=prior_state_noise,
        log_prefix="Shared prior",
        epochs=prior_epochs,
    )
    os.makedirs(CKPT_DIR, exist_ok=True)
    state = {
        'prior_tag': prior_tag,
        'vnet': prior['vnet'].state_dict(),
        'train_loss': prior['train_loss'],
        'val_loss': prior['val_loss'],
        'dev_seeds': [payload['seed'] for payload in dev_payloads],
        'n_samples': int(len(imag_states)),
        'weight_decay': prior_weight_decay,
        'state_noise': prior_state_noise,
        'epochs': prior_epochs,
    }
    torch.save(state, prior_ckpt)
    log(f"Shared prior saved to {prior_ckpt}")
    return state


def _select_candidate(candidates):
    ordered = sorted(candidates, key=lambda c: (c['val_loss'], c['state_noise']))
    best = ordered[0]
    if len(ordered) == 1:
        return best
    second = ordered[1]
    if abs(best['val_loss'] - second['val_loss']) > SELECTOR_TIE_TOL:
        return best
    return min((best, second), key=lambda c: c['state_noise'])


def train_seed(seed, cfg, tier='T4', weight_decay=0.0, state_noise=0.0,
               selector_mode='single', prior_state=None, prior_tag=None,
               finetune_epochs=500, payload=None,
               selection_metric='overall'):
    ckpt = _ckpt_path(tier, seed, weight_decay, state_noise, selector_mode,
                      prior_tag=prior_tag)
    if os.path.exists(ckpt):
        log(f"  [Seed {seed}] Loading checkpoint {ckpt}")
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = DMControlCartPoleSDModel(
            learn_params=True, prior_noise=0.3,
            mp_prior_noise=cfg['mp_prior_noise'],
            mp_clamp_min=cfg['mp_clamp_min'],
        )
        model.load_state_dict(state['model'])
        vnet = SwingupValueNet()
        vnet.load_state_dict(state['vnet'])
        actions = get_discrete_actions(11)
        params = state['params']
        selected_cfg = state.get('selected_cfg')
        if selected_cfg is not None:
            log(f"  [Seed {seed}] Selected config (cached): {selected_cfg}")
        log(f"  [Seed {seed}] SI: mc={params['masscart']:.3f}, "
            f"mp={params['masspole']:.3f}, l={params['length']:.3f} (cached)")
        return model, vnet, actions, params, selected_cfg

    payload = _prepare_seed_payload(seed, cfg) if payload is None else payload
    model = payload['model']
    actions = payload['actions']
    params = payload['params']
    imag_states = payload['imag_states']
    imag_returns = payload['imag_returns']

    if selector_mode == 'adaptive':
        candidate_runs = []
        for cand_wd, cand_sn in ADAPTIVE_CANDIDATES:
            log(f"  [Seed {seed}] Candidate: wd={cand_wd:g}, sn={cand_sn:g}")
            candidate_runs.append(_train_single_candidate(
                seed, imag_states, imag_returns,
                cand_wd, cand_sn,
                log_prefix=f"VNet wd={cand_wd:g} sn={cand_sn:g}",
                selection_metric=selection_metric))
        selected = _select_candidate(candidate_runs)
        best_vnet = selected['vnet']
        best_train_loss = selected['train_loss']
        best_val_loss = selected['val_loss']
        selected_cfg = {
            'selector_mode': selector_mode,
            'weight_decay': selected['weight_decay'],
            'state_noise': selected['state_noise'],
            'val_loss': selected['val_loss'],
        }
        log(f"  [Seed {seed}] Selector chose wd={selected['weight_decay']:g}, "
            f"sn={selected['state_noise']:g} by val_loss={selected['val_loss']:.4f}")
    elif selector_mode == 'shared_prior':
        selected = _train_single_candidate(
            seed, imag_states, imag_returns, weight_decay, state_noise,
            log_prefix="VNet finetune", epochs=finetune_epochs,
            init_state_dict=prior_state['vnet'],
            selection_metric=selection_metric)
        best_vnet = selected['vnet']
        best_train_loss = selected['train_loss']
        best_val_loss = selected['val_loss']
        selected_cfg = {
            'selector_mode': selector_mode,
            'weight_decay': weight_decay,
            'state_noise': state_noise,
            'val_loss': best_val_loss,
            'val_metrics': selected['val_metrics'],
            'selection_metric': selection_metric,
            'prior_tag': prior_tag,
            'prior_dev_seeds': prior_state['dev_seeds'],
            'prior_val_loss': prior_state['val_loss'],
        }
        log(f"  [Seed {seed}] Finetuned from shared prior {prior_tag}: "
            f"train={best_train_loss:.4f}, val={best_val_loss:.4f}")
        candidate_runs = None
    else:
        selected = _train_single_candidate(
            seed, imag_states, imag_returns, weight_decay, state_noise,
            log_prefix="VNet R1", selection_metric=selection_metric)
        best_vnet = selected['vnet']
        best_train_loss = selected['train_loss']
        best_val_loss = selected['val_loss']
        selected_cfg = {
            'selector_mode': selector_mode,
            'weight_decay': weight_decay,
            'state_noise': state_noise,
            'val_loss': best_val_loss,
            'val_metrics': selected['val_metrics'],
            'selection_metric': selection_metric,
        }
        log(f"  [Seed {seed}] VNet best: train={best_train_loss:.4f}, "
            f"val={best_val_loss:.4f} (weight_decay={weight_decay:g}, "
            f"state_noise={state_noise:g})")
        candidate_runs = None

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'vnet': best_vnet.state_dict(),
        'params': params,
        'val_loss': best_val_loss,
        'tier': tier,
        'seed': seed,
        'weight_decay': selected_cfg['weight_decay'],
        'state_noise': selected_cfg['state_noise'],
        'selector_mode': selector_mode,
        'prior_tag': prior_tag,
        'selection_metric': selection_metric,
        'selected_cfg': selected_cfg,
        'candidate_runs': None if candidate_runs is None else [
            {
                'weight_decay': c['weight_decay'],
                'state_noise': c['state_noise'],
                'train_loss': c['train_loss'],
                'val_loss': c['val_loss'],
            } for c in candidate_runs
        ],
    }, ckpt)
    log(f"  [Seed {seed}] Checkpoint saved to {ckpt}")

    return model, best_vnet, actions, params, selected_cfg


def evaluate_seed(seed, model, vnet, actions, tier='T4', weight_decay=0.0,
                  state_noise=0.0, selector_mode='single', prior_tag=None,
                  payload=None):
    eval_ckpt = _eval_path(tier, seed, weight_decay, state_noise, selector_mode,
                           prior_tag=prior_tag)
    if os.path.exists(eval_ckpt):
        state = torch.load(eval_ckpt, map_location='cpu', weights_only=False)
        returns = np.array(state['returns'], dtype=np.float32)
        log(f"  [Seed {seed}] Loaded eval cache {eval_ckpt}")
        return returns

    eval_env = suite.load('cartpole', 'swingup',
                          task_kwargs={'random': seed + 1000})
    if payload is not None:
        actions = payload['actions']
    g_over_l = model.g.item() / max(model.l.item(), 1e-3)
    returns = evaluate_agent(
        eval_env, model, actions,
        n_episodes=EVAL_EPISODES,
        num_simulations=NUM_SIMS,
        value_net=vnet,
        action_repeat=ACTION_REPEAT,
        verbose=False,
        g_over_l=g_over_l,
    )
    os.makedirs(EVAL_DIR, exist_ok=True)
    torch.save({
        'tier': tier,
        'seed': seed,
        'weight_decay': weight_decay,
        'state_noise': state_noise,
        'selector_mode': selector_mode,
        'prior_tag': prior_tag,
        'returns': returns.tolist(),
        'mean': float(returns.mean()),
        'std': float(returns.std()),
    }, eval_ckpt)
    log(f"  [Seed {seed}] Saved eval cache {eval_ckpt}")
    return returns


def _run_subprocess(cmd, log_path, env=None):
    with open(log_path, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc


def _wait_for_group(procs):
    for proc, label, log_path in procs:
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f'{label} failed (exit={ret}), see {log_path}')


def _launch_parallel_shared_prior(args, cfg, seeds, dev_seeds,
                                  prior_tag, experiment_tag):
    gpu_count = max(1, args.parallel_gpus)
    log(f"Launching shared-prior pipeline with {gpu_count} parallel workers")

    dev_group = []
    for idx, seed in enumerate(dev_seeds):
        device = f'cuda:{idx % gpu_count}' if args.parallel_gpus > 0 else None
        log_path = _stage_log_path(args.tier, 'prepare_dev', seed=seed, prior_tag=experiment_tag)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--tier', args.tier,
            '--selector_mode', 'shared_prior',
            '--stage', 'prepare_dev',
            '--stage_seed', str(seed),
            '--prior_weight_decay', str(args.prior_weight_decay),
            '--prior_state_noise', str(args.prior_state_noise),
            '--prior_epochs', str(args.prior_epochs),
            '--weight_decay', str(args.weight_decay),
            '--state_noise', str(args.state_noise),
            '--finetune_epochs', str(args.finetune_epochs),
            '--selection_metric', args.selection_metric,
            '--parallel_gpus', str(args.parallel_gpus),
        ]
        dev_group.append((_run_subprocess(cmd, log_path, env=_device_env(device)),
                          f'prepare_dev seed={seed}', log_path))
    _wait_for_group(dev_group)

    prior_log = _stage_log_path(args.tier, 'train_prior', prior_tag=experiment_tag)
    prior_cmd = [
        sys.executable, os.path.abspath(__file__),
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
    _wait_for_group([(_run_subprocess(prior_cmd, prior_log, env=_device_env('cuda:0' if args.parallel_gpus > 0 else None)),
                      'train_prior', prior_log)])

    target_group = []
    for idx, seed in enumerate(seeds):
        device = f'cuda:{idx % gpu_count}' if args.parallel_gpus > 0 else None
        log_path = _stage_log_path(args.tier, 'prepare_target', seed=seed, prior_tag=experiment_tag)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--tier', args.tier,
            '--selector_mode', 'shared_prior',
            '--stage', 'prepare_dev',
            '--stage_seed', str(seed),
            '--prior_weight_decay', str(args.prior_weight_decay),
            '--prior_state_noise', str(args.prior_state_noise),
            '--prior_epochs', str(args.prior_epochs),
            '--weight_decay', str(args.weight_decay),
            '--state_noise', str(args.state_noise),
            '--finetune_epochs', str(args.finetune_epochs),
            '--selection_metric', args.selection_metric,
            '--parallel_gpus', str(args.parallel_gpus),
        ]
        target_group.append((_run_subprocess(cmd, log_path, env=_device_env(device)),
                             f'prepare_target seed={seed}', log_path))
    _wait_for_group(target_group)

    finetune_group = []
    for idx, seed in enumerate(seeds):
        device = f'cuda:{idx % gpu_count}' if args.parallel_gpus > 0 else None
        log_path = _stage_log_path(args.tier, 'finetune', seed=seed, prior_tag=experiment_tag)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--tier', args.tier,
            '--selector_mode', 'shared_prior',
            '--stage', 'finetune_target',
            '--stage_seed', str(seed),
            '--prior_tag_override', prior_tag,
            '--prior_weight_decay', str(args.prior_weight_decay),
            '--prior_state_noise', str(args.prior_state_noise),
            '--prior_epochs', str(args.prior_epochs),
            '--weight_decay', str(args.weight_decay),
            '--state_noise', str(args.state_noise),
            '--finetune_epochs', str(args.finetune_epochs),
            '--selection_metric', args.selection_metric,
            '--parallel_gpus', str(args.parallel_gpus),
        ]
        finetune_group.append((_run_subprocess(cmd, log_path, env=_device_env(device)),
                               f'finetune seed={seed}', log_path))
    _wait_for_group(finetune_group)

    eval_group = []
    for idx, seed in enumerate(seeds):
        device = f'cuda:{idx % gpu_count}' if args.parallel_gpus > 0 else None
        log_path = _stage_log_path(args.tier, 'eval', seed=seed, prior_tag=experiment_tag)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--tier', args.tier,
            '--selector_mode', 'shared_prior',
            '--stage', 'eval_target',
            '--stage_seed', str(seed),
            '--prior_tag_override', prior_tag,
            '--prior_weight_decay', str(args.prior_weight_decay),
            '--prior_state_noise', str(args.prior_state_noise),
            '--prior_epochs', str(args.prior_epochs),
            '--weight_decay', str(args.weight_decay),
            '--state_noise', str(args.state_noise),
            '--finetune_epochs', str(args.finetune_epochs),
            '--selection_metric', args.selection_metric,
            '--parallel_gpus', str(args.parallel_gpus),
        ]
        eval_group.append((_run_subprocess(cmd, log_path, env=_device_env(device)),
                           f'eval seed={seed}', log_path))
    _wait_for_group(eval_group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', default='T4', choices=TIERS.keys())
    parser.add_argument('--si-only', action='store_true',
                        help='Only run SI+VNet training, skip MCTS eval')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Override seed list (default: all 10)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='VNet weight decay for this sweep run')
    parser.add_argument('--state_noise', type=float, default=0.0,
                        help='Gaussian noise std on VNet training states')
    parser.add_argument('--selector_mode', default='single',
                        choices=('single', 'adaptive', 'shared_prior'),
                        help='VNet config selection mode')
    parser.add_argument('--dev_seeds', type=int, nargs='+', default=None,
                        help='Development seeds for shared prior pretraining')
    parser.add_argument('--stage', default='full',
                        choices=('full', 'prepare_dev', 'train_prior', 'finetune_target', 'eval_target'),
                        help='Execution stage for shared-prior orchestration')
    parser.add_argument('--stage_seed', type=int, default=None,
                        help='Seed for stage-specific execution')
    parser.add_argument('--prior_tag_override', default=None,
                        help='Explicit prior tag for staged execution')
    parser.add_argument('--parallel_gpus', type=int, default=0,
                        help='Number of GPUs available for staged parallel launcher (0=CPU/default)')
    parser.add_argument('--prior_weight_decay', type=float, default=1e-4,
                        help='Weight decay for shared prior training')
    parser.add_argument('--selection_metric', default='overall',
                        choices=('overall', 'critical'),
                        help='Validation metric used to select the best VNet restart')
    parser.add_argument('--prior_state_noise', type=float, default=1e-3,
                        help='State noise for shared prior training')
    parser.add_argument('--prior_epochs', type=int, default=500,
                        help='Epochs for shared prior training')
    parser.add_argument('--finetune_epochs', type=int, default=500,
                        help='Epochs for per-seed finetuning from shared prior')
    args = parser.parse_args()

    cfg = TIERS[args.tier]
    if args.selector_mode == 'shared_prior':
        seeds = args.seeds if args.seeds else list(PAPER_SEEDS)
        dev_seeds = args.dev_seeds if args.dev_seeds else _default_dev_seeds(SEEDS)
    else:
        seeds = args.seeds if args.seeds else SEEDS
        dev_seeds = args.dev_seeds if args.dev_seeds else _default_dev_seeds(seeds)
    prior_tag = None
    prior_state = None
    t0 = time.time()
    log("=" * 70)
    log(f"SI ROBUSTNESS SWEEP — Tier {args.tier}")
    log(f"  Config: {cfg}")
    log(f"  Seeds: {seeds}")
    log(f"  SI-only: {args.si_only}")
    log(f"  VNet weight_decay: {args.weight_decay:g}")
    log(f"  VNet state_noise: {args.state_noise:g}")
    log(f"  Selector mode: {args.selector_mode}")
    if args.selector_mode == 'shared_prior':
        log(f"  Dev seeds: {dev_seeds}")
        log(f"  Paper seeds excluded from prior: {PAPER_SEEDS}")
        log(f"  Prior weight_decay: {args.prior_weight_decay:g}")
        log(f"  Prior state_noise: {args.prior_state_noise:g}")
        log(f"  Prior epochs: {args.prior_epochs}")
        log(f"  Finetune epochs: {args.finetune_epochs}")
        log(f"  Selection metric: {args.selection_metric}")
    log("=" * 70)

    if args.selector_mode == 'shared_prior':
        if not dev_seeds:
            raise ValueError('shared_prior mode requires at least one development seed')
        overlap = sorted(set(dev_seeds) & set(PAPER_SEEDS))
        if overlap:
            raise ValueError(f'dev_seeds must exclude paper seeds {overlap}')
        target_overlap = sorted(set(dev_seeds) & set(seeds))
        if target_overlap:
            raise ValueError(f'dev_seeds must exclude target seeds {target_overlap}')
        missing = sorted(set(dev_seeds) - set(SEEDS))
        if missing:
            raise ValueError(f'dev_seeds must be a subset of configured seeds, got {missing}')
        prior_tag = _prior_tag(
            dev_seeds, args.prior_weight_decay,
            args.prior_state_noise, args.prior_epochs)
        experiment_tag = args.prior_tag_override or _shared_experiment_tag(
            prior_tag, args.weight_decay, args.state_noise,
            args.finetune_epochs, args.selection_metric)

        if args.stage == 'full' and args.parallel_gpus > 1:
            _launch_parallel_shared_prior(
                args, cfg, seeds, dev_seeds, prior_tag, experiment_tag)
            return

        if args.stage == 'prepare_dev':
            if args.stage_seed is None:
                raise ValueError('prepare_dev stage requires --stage_seed')
            prepare_seed_payload_cache(args.stage_seed, cfg, args.tier)
            return

        if args.stage == 'train_prior':
            _train_shared_prior(
                dev_seeds, args.tier, prior_tag,
                args.prior_weight_decay, args.prior_state_noise,
                args.prior_epochs)
            return

        if args.stage == 'finetune_target':
            if args.stage_seed is None:
                raise ValueError('finetune_target stage requires --stage_seed')
            prior_state = _train_shared_prior(
                dev_seeds, args.tier, prior_tag,
                args.prior_weight_decay, args.prior_state_noise,
                args.prior_epochs)
            prepare_seed_payload_cache(args.stage_seed, cfg, args.tier)
            payload = _load_seed_payload(args.tier, args.stage_seed)
            model, vnet, actions, params, selected_cfg = train_seed(
                args.stage_seed, cfg, tier=args.tier,
                weight_decay=args.weight_decay,
                state_noise=args.state_noise,
                selector_mode=args.selector_mode,
                prior_state=prior_state,
                prior_tag=experiment_tag,
                finetune_epochs=args.finetune_epochs,
                payload=payload,
                selection_metric=args.selection_metric)
            log(f"  [Seed {args.stage_seed}] Finetune complete: "
                f"{selected_cfg['val_loss']:.4f}")
            return

        if args.stage == 'eval_target':
            if args.stage_seed is None:
                raise ValueError('eval_target stage requires --stage_seed')
            prepare_seed_payload_cache(args.stage_seed, cfg, args.tier)
            payload = _load_seed_payload(args.tier, args.stage_seed)
            model, vnet, actions, params, selected_cfg = train_seed(
                args.stage_seed, cfg, tier=args.tier,
                weight_decay=args.weight_decay,
                state_noise=args.state_noise,
                selector_mode=args.selector_mode,
                prior_state=None,
                prior_tag=experiment_tag,
                finetune_epochs=args.finetune_epochs,
                payload=payload,
                selection_metric=args.selection_metric)
            returns = evaluate_seed(
                args.stage_seed, model, vnet, actions,
                tier=args.tier,
                weight_decay=selected_cfg['weight_decay'],
                state_noise=selected_cfg['state_noise'],
                selector_mode=args.selector_mode,
                prior_tag=selected_cfg.get('prior_tag'),
                payload=payload)
            log(f"  [Seed {args.stage_seed}] Eval complete: {returns.mean():.1f} ±{returns.std():.1f}")
            return

        dev_payloads = []
        for seed in dev_seeds:
            log(f"\n{'='*24} Preparing Dev Seed {seed} {'='*24}")
            prepare_seed_payload_cache(seed, cfg, args.tier)
            dev_payloads.append(_load_seed_payload(args.tier, seed))
        prior_state = _train_shared_prior(
            dev_payloads, args.tier, prior_tag,
            args.prior_weight_decay, args.prior_state_noise,
            args.prior_epochs)

    # Phase 1: Train all seeds
    trained = {}
    si_results = {}
    for seed in seeds:
        log(f"\n{'='*30} Training Seed {seed} {'='*30}")
        payload = _load_seed_payload(args.tier, seed) if args.selector_mode == 'shared_prior' else None
        model, vnet, actions, params, selected_cfg = train_seed(
            seed, cfg, tier=args.tier,
            weight_decay=args.weight_decay,
            state_noise=args.state_noise,
            selector_mode=args.selector_mode,
            prior_state=prior_state,
            prior_tag=experiment_tag if args.selector_mode == 'shared_prior' else None,
            finetune_epochs=args.finetune_epochs,
            payload=payload,
            selection_metric=args.selection_metric)
        trained[seed] = (model, vnet, actions, selected_cfg)
        si_results[seed] = params

    train_time = time.time() - t0
    log(f"\nTraining complete ({train_time:.0f}s).")

    # SI summary
    log("\n" + "=" * 70)
    log("SI PARAMETER SUMMARY")
    log("=" * 70)
    header = f"{'Seed':>6s}  {'mc':>7s}  {'mp':>7s}  {'l':>7s}  {'mp_err%':>8s}  {'SI_loss':>10s}"
    log(header)
    log("-" * len(header))
    mp_errs = []
    for seed in seeds:
        p = si_results[seed]
        mp_err = abs(p['masspole'] - 0.1) / 0.1 * 100
        mp_errs.append(mp_err)
        log(f"{seed:>6d}  {p['masscart']:>7.3f}  {p['masspole']:>7.3f}  "
            f"{p['length']:>7.3f}  {mp_err:>7.1f}%")
    mp_errs = np.array(mp_errs)
    log(f"\nmp error: mean={mp_errs.mean():.1f}%, "
        f"std={mp_errs.std():.1f}%, max={mp_errs.max():.1f}%")
    mp_vals = [f'{si_results[s]["masspole"]:.3f}' for s in seeds]
    log(f"mp values: {mp_vals}")

    if args.si_only:
        log(f"\nTotal time: {time.time()-t0:.0f}s")
        return

    # Phase 2: Evaluate
    log(f"\nStarting MCTS evaluation...")
    eval_results = {}
    for seed in seeds:
        model, vnet, actions, selected_cfg = trained[seed]
        returns = evaluate_seed(
            seed, model, vnet, actions,
            tier=args.tier,
            weight_decay=selected_cfg['weight_decay'],
            state_noise=selected_cfg['state_noise'],
            selector_mode=args.selector_mode,
            prior_tag=selected_cfg.get('prior_tag'))
        eval_results[seed] = returns
        log(f"  Seed {seed}: {returns.mean():.1f} ±{returns.std():.1f}  "
            f"[{returns.min():.0f}, {returns.max():.0f}]  "
            f"(selected wd={selected_cfg['weight_decay']:g}, "
            f"sn={selected_cfg['state_noise']:g})")

    # Final summary
    log("\n" + "=" * 70)
    log("FINAL RESULTS SUMMARY")
    log("=" * 70)
    header = f"{'Seed':>6s}  {'Return':>7s}  {'mp':>7s}  {'mp_err%':>8s}"
    log(header)
    log("-" * len(header))
    all_means = []
    for seed in seeds:
        r = eval_results[seed].mean()
        p = si_results[seed]
        mp_err = abs(p['masspole'] - 0.1) / 0.1 * 100
        all_means.append(r)
        log(f"{seed:>6d}  {r:>7.1f}  {p['masspole']:>7.3f}  {mp_err:>7.1f}%")

    all_means = np.array(all_means)
    sorted_m = sorted(all_means)
    q1, q3 = len(sorted_m) // 4, len(sorted_m) - len(sorted_m) // 4
    iqm = np.mean(sorted_m[q1:q3]) if q3 > q1 else all_means.mean()
    n_fail = sum(1 for m in all_means if m < 700)

    log("-" * len(header))
    log(f"Mean={all_means.mean():.1f}  Median={np.median(all_means):.1f}  "
        f"IQM={iqm:.1f}  <700={n_fail}/{len(seeds)}")

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
