import numpy as np
import torch

from src.planning.mcts import mcts_search
from src.planning.period_aware_mcts import (
    SequentialDecisionWorldModel,
    default_period_boundary_fn,
    edge_gamma,
    make_sequential_reward_fn,
    period_aware_mcts_search,
)


class ConstModel(torch.nn.Module):
    def forward(self, state, action):
        nxt = state.clone()
        nxt[..., 0] = nxt[..., 0] + 1.0
        return nxt


def test_period_length_one_delegates():
    model = ConstModel()

    def reward_fn(state, action=None):
        return -float(np.asarray(state).reshape(-1)[0])

    root = np.zeros(3, dtype=np.float32)
    np.random.seed(0)
    a = period_aware_mcts_search(root, model, n_actions=2, num_simulations=6, reward_fn=reward_fn, period_length=1)
    np.random.seed(0)
    b = mcts_search(root, model, n_actions=2, num_simulations=6, reward_fn=reward_fn)
    assert int(a) == int(b)


def test_intermediate_edges_do_not_discount():
    is_boundary = default_period_boundary_fn(4, substep_index=-1)
    parent = np.array([0.0, 0.0, 1.0])  # not last substep
    child = np.array([0.0, 0.0, 2.0])
    assert edge_gamma(parent, child, 0.5, is_boundary) == 1.0
    parent_last = np.array([0.0, 0.0, 3.0])
    assert edge_gamma(parent_last, child, 0.5, is_boundary) == 0.5


def test_sequential_world_model_executes_on_last_substep():
    joint = ConstModel()
    wrap = SequentialDecisionWorldModel(joint, joint_dim=2, period_length=2)
    state = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]])
    mid = wrap(state, torch.tensor([3]))
    assert float(mid[0, 2]) == 1.0
    assert float(mid[0, 3]) == 3.0
    end = wrap(mid, torch.tensor([4]))
    assert float(end[0, 2]) == 0.0
    assert float(end[0, 0]) == 2.0


def test_sequential_reward_zero_until_period_end():
    def base(state, action=None):
        return 7.0

    fn = make_sequential_reward_fn(base, period_length=4, joint_dim=2)
    mid = np.array([0.0, 0.0, 2.0, 0, 0, 0, 0])
    end = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0])
    assert fn(mid, 1) == 0.0
    assert fn(end, 1) == 7.0
