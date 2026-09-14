"""Period-aware MCTS wrapper.

Existing ``mcts_search`` multiplies ``gamma`` on every tree edge. Sequential
Beer Game and multi-region epidemic decisions are cycles of substeps: only the
period-boundary edge should discount. This module wraps that contract without
changing ``mcts.py`` numerics.

When ``period_length == 1`` the call is delegated to ``mcts_search``.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.planning.mcts import (
    MCTSNode,
    MCTSResult,
    _puct_score,
    mcts_search,
)


def _as_numpy(state) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state
    return state.detach().cpu().numpy()


def default_period_boundary_fn(period_length: int, substep_index: int = -1):
    """True iff the parent state is the last substep of a period."""

    def is_boundary(parent_state, child_state=None) -> bool:
        if period_length <= 1:
            return True
        parent = _as_numpy(parent_state)
        sub = int(round(float(parent[substep_index])))
        return sub == period_length - 1

    return is_boundary


def edge_gamma(parent_state, child_state, gamma: float, is_boundary: Callable) -> float:
    return float(gamma) if is_boundary(parent_state, child_state) else 1.0


class SequentialDecisionWorldModel(nn.Module):
    """Lift a joint-action world model to sequential per-node decisions.

    State layout: ``[joint_state | current_substep | pending_actions...]``.
    Intermediate substeps only write the pending action and increment the
    substep index. The last substep executes the joint model and resets.
    """

    def __init__(self, joint_model: nn.Module, joint_dim: int, period_length: int):
        super().__init__()
        self.joint_model = joint_model
        self.joint_dim = int(joint_dim)
        self.period_length = int(period_length)
        self.state_dim = self.joint_dim + 1 + self.period_length

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = action.view(-1).long()
        if state.shape[0] == 1:  # tree-search fast path: no batched masking
            sub = int(round(float(state[0, self.joint_dim])))
            out = state.clone()
            out[0, self.joint_dim + 1 + sub] = float(action[0])
            if sub == self.period_length - 1:
                out[0, : self.joint_dim] = self.joint_model(state[:, : self.joint_dim], out[:, self.joint_dim + 1:].long())[0]
                out[0, self.joint_dim:] = 0.0
            else:
                out[0, self.joint_dim] = sub + 1
            return out
        batch = state.shape[0]
        joint = state[:, : self.joint_dim]
        substep = state[:, self.joint_dim]
        pending = state[:, self.joint_dim + 1 :].clone()
        idx = torch.arange(batch, device=state.device)
        sub_i = torch.round(substep).long().clamp(0, self.period_length - 1)
        pending[idx, sub_i] = action.to(pending.dtype)

        is_last = sub_i == (self.period_length - 1)
        next_joint = joint.clone()
        if bool(is_last.any()):
            planned = pending[is_last].long()
            next_joint[is_last] = self.joint_model(joint[is_last], planned)

        next_sub = torch.where(is_last, torch.zeros_like(substep), substep + 1)
        next_pending = torch.where(is_last.unsqueeze(1), torch.zeros_like(pending), pending)
        return torch.cat([next_joint, next_sub.unsqueeze(1), next_pending], dim=1)


def pack_sequential_state(joint_state: np.ndarray, substep: int, pending: np.ndarray) -> np.ndarray:
    joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    pending = np.asarray(pending, dtype=np.float32).reshape(-1)
    return np.concatenate([joint_state, [float(substep)], pending]).astype(np.float32)


def unpack_sequential_state(state: np.ndarray, joint_dim: int, period_length: int):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    joint = state[:joint_dim]
    substep = int(round(float(state[joint_dim])))
    pending = state[joint_dim + 1 : joint_dim + 1 + period_length]
    return joint, substep, pending


def make_sequential_reward_fn(
    base_reward_fn: Callable,
    period_length: int,
    joint_dim: int,
) -> Callable:
    """Reward is 0 on intermediate substeps; period-end uses the joint reward."""

    def reward_fn(state: np.ndarray, action=None) -> float:
        if period_length <= 1:
            return float(base_reward_fn(state, action))
        echelon = int(round(float(np.asarray(state).reshape(-1)[joint_dim])))
        if echelon == 0:
            return float(base_reward_fn(np.asarray(state).reshape(-1)[:joint_dim], action))
        return 0.0

    return reward_fn


def period_aware_mcts_search(
    root_state: np.ndarray,
    world_model,
    n_actions: int = 3,
    num_simulations: int = 50,
    c: float = 1.4,
    gamma: float = 0.99,
    reward_fn: Callable = None,
    policy_net=None,
    value_net=None,
    return_details: bool = False,
    n_workers: int = None,
    rollout_depth: int = 10,
    imagined_state_recorder: Optional[Callable[[np.ndarray], None]] = None,
    period_length: int = 1,
    substep_index: int = -1,
    is_period_boundary: Optional[Callable] = None,
    value_rollout_depth: int = 0,
):
    """MCTS with discount only on period-boundary edges.

    ``period_length == 1`` delegates to ``mcts_search`` so existing single-step
    environments keep bit-identical backup. With ``value_net``, the leaf is
    evaluated as ``value_rollout_depth`` random substeps, completion of the
    current period, then a bootstrap ``V`` at the period boundary.
    """
    if period_length <= 1 and is_period_boundary is None:
        return mcts_search(
            root_state=root_state,
            world_model=world_model,
            n_actions=n_actions,
            num_simulations=num_simulations,
            c=c,
            gamma=gamma,
            reward_fn=reward_fn,
            policy_net=policy_net,
            value_net=value_net,
            return_details=return_details,
            n_workers=n_workers,
            rollout_depth=rollout_depth,
            imagined_state_recorder=imagined_state_recorder,
        )

    if reward_fn is None:
        raise ValueError("period_aware_mcts_search requires an explicit reward_fn")

    if is_period_boundary is None:
        is_period_boundary = default_period_boundary_fn(period_length, substep_index)

    root_tensor = torch.tensor(root_state, dtype=torch.float32)
    root = MCTSNode(root_tensor)

    for _ in range(num_simulations):
        node = root
        path = [node]

        while node.children:
            if not node.is_fully_expanded(n_actions):
                if len(node.children) < max(1, int(1.0 * (node.N ** 0.5))):
                    break
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        if not node.is_fully_expanded(n_actions):
            if policy_net is not None:
                with torch.no_grad():
                    node_priors = F.softmax(policy_net(node.state.unsqueeze(0)), dim=-1)[0].cpu().numpy()
            else:
                node_priors = np.ones(n_actions) / n_actions

            unexpanded = [a for a in range(n_actions) if a not in node.children]
            unexpanded_priors = np.array([node_priors[a] for a in unexpanded], dtype=np.float64)
            unexpanded_priors = unexpanded_priors / unexpanded_priors.sum()
            action_to_expand = unexpanded[int(np.random.choice(len(unexpanded), p=unexpanded_priors))]

            with torch.no_grad():
                next_state = world_model(
                    node.state.unsqueeze(0),
                    torch.tensor([action_to_expand], dtype=torch.int64),
                )[0]

            next_state_np = next_state.detach().cpu().numpy()
            if imagined_state_recorder is not None:
                imagined_state_recorder(next_state_np.copy())
            child_reward = reward_fn(next_state_np, action_to_expand)
            child = MCTSNode(
                next_state,
                parent=node,
                action=action_to_expand,
                prior=float(node_priors[action_to_expand]),
                reward=child_reward,
            )
            node.children[action_to_expand] = child
            node = child
            path.append(node)

        if value_net is not None:
            # V is defined on period-boundary states only: optional random
            # rollout, complete the period, accumulate rewards, then bootstrap.
            rollout_state = node.state
            rollout_return = 0.0
            discount = 1.0
            steps_done = 0
            while steps_done < value_rollout_depth or (
                period_length > 1 and int(round(float(rollout_state[substep_index]))) != 0
            ):
                steps_done += 1
                rand_action = int(np.random.randint(n_actions))
                parent_np = rollout_state.detach().cpu().numpy()
                with torch.no_grad():
                    next_state = world_model(
                        rollout_state.unsqueeze(0),
                        torch.tensor([rand_action], dtype=torch.int64),
                    )[0]
                next_state_np = next_state.detach().cpu().numpy()
                if imagined_state_recorder is not None:
                    imagined_state_recorder(next_state_np.copy())
                rollout_return += discount * reward_fn(next_state_np, rand_action)
                discount *= edge_gamma(parent_np, next_state_np, gamma, is_period_boundary)
                rollout_state = next_state
            with torch.no_grad():
                leaf_value = rollout_return + discount * value_net(rollout_state.unsqueeze(0)).item()
        else:
            rollout_state = node.state
            rollout_return = 0.0
            discount = 1.0
            for _ in range(rollout_depth):
                rand_action = int(np.random.randint(n_actions))
                parent_np = rollout_state.detach().cpu().numpy()
                with torch.no_grad():
                    next_state = world_model(
                        rollout_state.unsqueeze(0),
                        torch.tensor([rand_action], dtype=torch.int64),
                    )[0]
                next_state_np = next_state.detach().cpu().numpy()
                if imagined_state_recorder is not None:
                    imagined_state_recorder(next_state_np.copy())
                r = reward_fn(next_state_np, rand_action)
                g = edge_gamma(parent_np, next_state_np, gamma, is_period_boundary)
                rollout_return += discount * r
                discount *= g
                rollout_state = next_state
            leaf_value = rollout_return

        G = leaf_value
        for node in reversed(path):
            if node.parent is not None:
                g = edge_gamma(
                    node.parent.state,
                    node.state,
                    gamma,
                    is_period_boundary,
                )
                G = node.reward + g * G
            node.N += 1
            node.W += G
            node.Q = node.W / node.N

    if not root.children:
        if return_details:
            uniform = np.ones(n_actions) / n_actions
            return MCTSResult(0, uniform, 0.0)
        return 0

    visit_counts = np.zeros(n_actions)
    q_values = np.zeros(n_actions)
    for action, child in root.children.items():
        visit_counts[action] = child.N
        q_values[action] = child.Q
    policy = visit_counts / visit_counts.sum()
    best_action = int(np.argmax(visit_counts))
    if return_details:
        return MCTSResult(best_action, policy, root.Q, q_values, root_state)
    return best_action
