import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Callable, Optional

from src.env.task_allocation_env import BACKLOG_PENALTY_WEIGHT, compute_reward


def make_reward_fn(n_workers: int):
    """创建正确的奖励函数，传入 n_workers 避免将 fatigue 计入 backlog"""
    def reward_fn(state: np.ndarray) -> float:
        return compute_reward(state, n_workers=n_workers)
    return reward_fn


def default_reward_fn(state: np.ndarray, n_workers: int = None) -> float:
    """默认奖励函数：与环境保持一致"""
    return compute_reward(state, n_workers=n_workers)


class MCTSNode:
    """MCTS树节点"""

    __slots__ = ['state', 'parent', 'action', 'children', 'N', 'W', 'Q', 'P', 'reward']

    def __init__(self, state: torch.Tensor, parent: 'MCTSNode' = None, action: int = None, prior: float = 0.0, reward: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, MCTSNode] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior  # 策略先验概率
        self.reward = reward  # 从父节点转移到此节点的即时奖励

    def is_fully_expanded(self, n_actions: int) -> bool:
        return len(self.children) == n_actions


def _puct_score(parent: MCTSNode, child: MCTSNode, c: float) -> float:
    """PUCT分数（AlphaZero/MuZero风格）：Q + c * P * sqrt(N_parent) / (1 + N_child)"""
    exploration = c * child.P * math.sqrt(parent.N + 1) / (1 + child.N)
    return child.Q + exploration


class MCTSResult:
    """MCTS搜索结果"""
    __slots__ = ['action', 'policy', 'root_value', 'q_values', 'state']

    def __init__(self, action: int, policy: np.ndarray, root_value: float,
                 q_values: np.ndarray = None, state: np.ndarray = None):
        self.action = action
        self.policy = policy
        self.root_value = root_value
        self.q_values = q_values  # 各动作的Q值
        self.state = state  # 决策时的状态


def mcts_search(
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
):
    """MuZero风格MCTS搜索

    Args:
        world_model: SD世界模型，用于状态转移预测
        policy_net: 策略网络，用于先验概率（可选）
        value_net: 价值网络，用于叶节点评估（可选，否则用reward_fn）
        return_details: 如果True，返回MCTSResult
        n_workers: 工人数量，用于正确计算奖励（避免将fatigue计入backlog）
    """
    if reward_fn is None:
        if n_workers is not None:
            reward_fn = make_reward_fn(n_workers)
        else:
            reward_fn = default_reward_fn

    root_tensor = torch.tensor(root_state, dtype=torch.float32)
    root = MCTSNode(root_tensor)

    for sim in range(num_simulations):
        node = root
        path = [node]

        # Selection: 沿PUCT最大路径下降
        while node.is_fully_expanded(n_actions) and node.children:
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Expansion: 扩展所有未访问动作
        if not node.is_fully_expanded(n_actions):
            # 当前节点的策略先验（而非根节点）
            if policy_net is not None:
                with torch.no_grad():
                    node_priors = F.softmax(policy_net(node.state.unsqueeze(0)), dim=-1)[0].numpy()
            else:
                node_priors = np.ones(n_actions) / n_actions

            unexpanded = [a for a in range(n_actions) if a not in node.children]
            states_batch = node.state.unsqueeze(0).expand(len(unexpanded), -1)
            actions_batch = torch.tensor(unexpanded, dtype=torch.int64)

            with torch.no_grad():
                next_states = world_model(states_batch, actions_batch)

            for i, action in enumerate(unexpanded):
                # 计算转移即时奖励
                child_reward = reward_fn(next_states[i].detach().numpy())
                # 子节点先验来自当前节点的策略分布
                child = MCTSNode(next_states[i], parent=node, action=action, prior=node_priors[action], reward=child_reward)
                node.children[action] = child

            # 用PUCT在新扩展的子节点中选择评估分支
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Leaf evaluation: 使用Value网络，或设为0避免双重计数
        if value_net is not None:
            with torch.no_grad():
                leaf_value = value_net(node.state.unsqueeze(0)).item()
        else:
            # 没有价值网络时，用0作为叶节点的未来回报估计
            # 即时奖励已在node.reward中，避免重复计入
            leaf_value = 0.0

        # Backpropagation: 先加即时奖励再更新W，使Q表示Q(s,a)而非V(s')
        G = leaf_value
        for node in reversed(path):
            if node.parent is not None:
                G = node.reward + gamma * G  # 先加即时奖励
            node.N += 1
            node.W += G
            node.Q = node.W / node.N

    if not root.children:
        if return_details:
            uniform = np.ones(n_actions) / n_actions
            return MCTSResult(0, uniform, 0.0)
        return 0

    # 计算基于访问次数的策略分布
    visit_counts = np.zeros(n_actions)
    q_values = np.zeros(n_actions)
    for action, child in root.children.items():
        visit_counts[action] = child.N
        q_values[action] = child.Q
    policy = visit_counts / visit_counts.sum()

    # 选择访问次数最多的动作（更稳定）
    best_action = int(np.argmax(visit_counts))
    root_value = root.Q

    if return_details:
        return MCTSResult(best_action, policy, root_value, q_values, root_state)
    return best_action
