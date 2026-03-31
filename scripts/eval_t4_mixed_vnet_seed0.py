#!/usr/bin/env python3
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dm_control import suite
from scripts.run_dmcontrol_cartpole import evaluate_agent, get_discrete_actions, SwingupValueNet, log
from src.world_model.dmcontrol_cartpole_sd_model import DMControlCartPoleSDModel
from scripts.sweep_si_robust import _ckpt_path

SEED = 0
TIER = 'T4'
WITH_VNET_EPISODES = 5
WITH_VNET_SIMS = 200
MIXED_EPISODES = 5
MIXED_SIMS = 200
NO_VNET_EPISODES = 2
NO_VNET_SIMS = 50
ACTION_REPEAT = 2
LEAF_VNET_MIX = 0.25
LEAF_VNET_GATE_COS = 0.85


def load_model_vnet():
    ckpt = torch.load(_ckpt_path(TIER, SEED), map_location='cpu', weights_only=False)
    model = DMControlCartPoleSDModel(
        learn_params=True, prior_noise=0.3,
        mp_prior_noise=0.15,
        mp_clamp_min=0.03,
    )
    model.load_state_dict(ckpt['model'])
    vnet = SwingupValueNet()
    vnet.load_state_dict(ckpt['vnet'])
    return model, vnet


def run_case(label, value_net, n_episodes, num_sims,
             leaf_vnet_mix=1.0, leaf_vnet_gate_cos=None):
    log(f"Running {label} (episodes={n_episodes}, sims={num_sims})...")
    model, _ = load_model_vnet()
    env = suite.load('cartpole', 'swingup', task_kwargs={'random': SEED + 1000})
    actions = get_discrete_actions(11)
    g_over_l = model.g.item() / max(model.l.item(), 1e-3)
    returns = evaluate_agent(
        env, model, actions,
        n_episodes=n_episodes,
        num_simulations=num_sims,
        value_net=value_net,
        action_repeat=ACTION_REPEAT,
        verbose=False,
        g_over_l=g_over_l,
        leaf_vnet_mix=leaf_vnet_mix,
        leaf_vnet_gate_cos=leaf_vnet_gate_cos,
    )
    log(f"{label}: {returns.mean():.1f} ±{returns.std():.1f} {returns}")


def main():
    log('=' * 70)
    log('Seed 0 mixed-VNet ablation')
    log('=' * 70)

    model, vnet = load_model_vnet()
    run_case('with-VNet', vnet,
             n_episodes=WITH_VNET_EPISODES,
             num_sims=WITH_VNET_SIMS)
    run_case(f'mixed-gated (mix={LEAF_VNET_MIX}, gate={LEAF_VNET_GATE_COS})',
             vnet,
             n_episodes=MIXED_EPISODES,
             num_sims=MIXED_SIMS,
             leaf_vnet_mix=LEAF_VNET_MIX,
             leaf_vnet_gate_cos=LEAF_VNET_GATE_COS)
    run_case('no-VNet quick', None,
             n_episodes=NO_VNET_EPISODES,
             num_sims=NO_VNET_SIMS)


if __name__ == '__main__':
    main()
