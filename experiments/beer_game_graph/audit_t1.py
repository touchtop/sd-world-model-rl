"""Audit T1 compatibility without changing historical Beer Game experiments.

Run: python -m experiments.beer_game_graph.audit_t1 --out results/t1_audit
The frozen design's G-BT0/1/2 require matched transitions, not a favourable
return. Paired RNG streams isolate the representation change in the MCTS check.
This check does not reproduce the paper's training curves or OOD tables.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.env.beer_game_env import BeerGameEnv, BeerGameEnvSequential
from src.env.graph_beer_game import GraphBeerGameEnv, load_supply_graph
from src.world_model.beer_game_sd_model import BeerGameSDModel
from src.world_model.graph_beer_game_sd_model import GraphBeerGameSDModel
from experiments.run_beer_game_v2 import make_beer_game_reward_fn, decode_flat_action
from src.planning.mcts import mcts_search


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    topology = Path('experiments/beer_game_graph/topologies/t1_serial4.yaml')
    # The paper uses three choices per node: 3**4 = 81 joint actions.
    graph = replace(load_supply_graph(topology), order_levels=3)
    report = {'complete': False, 'order_levels': 3, 'joint_actions': 81,
              'episode_weeks': graph.T, 'seeds': list(range(args.seeds)),
              'simulations': args.simulations, 'gamma': .99,
              'rollout_depth': 4, 'leaf': 'none (compatibility check only)',
              'fixed_action_episodes': [], 'closed_loop': [],
              'G-BT3': {'status': 'not_reproduced',
                        'reason': 'No Table 1 training or Table 2 intervention rerun in this audit.'}}
    def save():
        (out / 'audit.json').write_text(json.dumps(report, indent=2))
    save()
    models = []
    for prior in (False, True):
        seed_all(42)
        classic = BeerGameSDModel(order_levels=3, prior_belief=prior)
        seed_all(42)
        structured = GraphBeerGameSDModel(graph, prior_belief=prior)
        models.append((classic, structured))
    for seed in range(args.seeds):
        for pattern in ('random', 'zero', 'maximum', 'alternating'):
            env = BeerGameEnv(order_levels=3, seed=seed)
            generic = GraphBeerGameEnv(graph, seed=seed)
            sequential = BeerGameEnvSequential(order_levels=3, seed=seed)
            state, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(state, generic.reset(seed=seed)[0])
            sequential.reset(seed=seed)
            rng = np.random.default_rng(seed)
            total = 0.
            for week in range(graph.T):
                action = (rng.integers(0, 3, 4) if pattern == 'random' else
                          np.full(4, 0 if pattern == 'zero' else
                                  (2 if pattern == 'maximum' else 2 * (week % 2))))
                # All joint actions at reachable states, including flat decoding.
                states = torch.tensor(np.repeat(state[None], 81, axis=0))
                actions = torch.arange(81)
                for classic, structured in models:
                    with torch.no_grad():
                        assert torch.equal(classic(states, actions), structured(states, actions))
                old = env.step(action)
                new = generic.step(action)
                np.testing.assert_array_equal(old[0], new[0])
                assert old[1:4] == new[1:4]
                for index, part in enumerate(action):
                    seq = sequential.step(int(part))
                    if index < 3:
                        assert seq[1] == 0 and not seq[2] and not seq[3]
                np.testing.assert_array_equal(seq[0][:-1], old[0])
                assert seq[1:4] == old[1:4]
                state = old[0]
                total += old[1]
            report['fixed_action_episodes'].append({'seed': seed, 'pattern': pattern, 'return': total})
    report.update({'G-BT0': True, 'G-BT1': True, 'G-BT2': True})
    save()
    print('G-BT0/1/2 passed on full episodes', flush=True)
    # Closed-loop compatibility is not a fresh learning/performance experiment.
    classic, structured = models[0]
    for seed in range(3):
        env, generic = BeerGameEnv(order_levels=3), GraphBeerGameEnv(graph)
        state, _ = env.reset(seed=seed)
        other, _ = generic.reset(seed=seed)
        trajectory = []
        for week in range(graph.T):
            chosen = []
            for model, root in ((classic, state), (structured, other)):
                seed_all(seed * 1000 + week)
                chosen.append(int(mcts_search(
                    root_state=root, world_model=model, n_actions=81,
                    num_simulations=args.simulations, gamma=.99,
                    reward_fn=make_beer_game_reward_fn(), value_net=None,
                    rollout_depth=4)))
            assert chosen[0] == chosen[1]
            action = decode_flat_action(chosen[0], 3, 4)
            old, new = env.step(action), generic.step(action)
            np.testing.assert_array_equal(old[0], new[0])
            assert old[1:4] == new[1:4]
            state, other = old[0], new[0]
            trajectory.append({'week': week, 'flat_action': chosen[0], 'reward': old[1]})
        report['closed_loop'].append({'seed': seed, 'trajectory': trajectory,
                                     'return': sum(r['reward'] for r in trajectory)})
        save()
        print(f'paired MCTS seed {seed} passed', flush=True)
    sources = [topology, Path(__file__), Path('src/env/beer_game_env.py'),
               Path('src/env/graph_beer_game.py'), Path('src/world_model/beer_game_sd_model.py'),
               Path('src/world_model/graph_beer_game_sd_model.py'), Path('src/planning/mcts.py')]
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report['versions'] = {'torch': torch.__version__, 'numpy': np.__version__}
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
