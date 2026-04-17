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
    rollout_depth: int = 10,
    imagined_state_recorder: Optional[Callable[[np.ndarray], None]] = None,
):
    """MuZero风格MCTS搜索

    Args:
        world_model: SD世界模型，用于状态转移预测
        policy_net: 策略网络，用于先验概率（可选）
        value_net: 价值网络，用于叶节点评估（可选，否则用reward_fn）
        return_details: 如果True，返回MCTSResult
        n_workers: 工人数量，用于正确计算奖励（避免将fatigue计入backlog）
        imagined_state_recorder: 若提供，则记录搜索过程中生成的内部想象状态
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

        # Selection: 沿PUCT最大路径下降（只要有子节点就可以选择）
        while node.children:
            # 如果未完全扩展，以一定概率扩展新动作（渐进拓宽）
            if not node.is_fully_expanded(n_actions):
                # Progressive widening: 根据访问次数决定是否扩展
                # 公式: 当 |children| < C * N^alpha 时扩展（C=1, alpha=0.5）
                if len(node.children) < max(1, int(1.0 * (node.N ** 0.5))):
                    break  # 扩展新动作
            # 在已扩展的子节点中用PUCT选择
            node = max(node.children.values(), key=lambda ch: _puct_score(node, ch, c))
            path.append(node)

        # Expansion: 渐进扩展 - 每次只扩展一个动作
        if not node.is_fully_expanded(n_actions):
            # 获取策略先验
            if policy_net is not None:
                with torch.no_grad():
                    node_priors = F.softmax(policy_net(node.state.unsqueeze(0)), dim=-1)[0].cpu().numpy()
            else:
                node_priors = np.ones(n_actions) / n_actions

            unexpanded = [a for a in range(n_actions) if a not in node.children]

            # 选择一个动作扩展：按先验概率加权随机选择
            unexpanded_priors = np.array([node_priors[a] for a in unexpanded])
            unexpanded_priors = unexpanded_priors / unexpanded_priors.sum()
            action_to_expand = unexpanded[np.random.choice(len(unexpanded), p=unexpanded_priors)]

            # 扩展单个动作
            with torch.no_grad():
                next_state = world_model(
                    node.state.unsqueeze(0),
                    torch.tensor([action_to_expand], dtype=torch.int64)
                )[0]

            next_state_np = next_state.detach().cpu().numpy()
            if imagined_state_recorder is not None:
                imagined_state_recorder(next_state_np.copy())
            child_reward = reward_fn(next_state_np, action_to_expand)
            child = MCTSNode(next_state, parent=node, action=action_to_expand,
                           prior=node_priors[action_to_expand], reward=child_reward)
            node.children[action_to_expand] = child
            node = child
            path.append(node)

        # Leaf evaluation: 使用Value网络，或Monte Carlo rollout
        if value_net is not None:
            with torch.no_grad():
                leaf_value = value_net(node.state.unsqueeze(0)).item()
        else:
            # Monte Carlo rollout: 随机策略模拟rollout_depth步
            rollout_state = node.state
            rollout_return = 0.0
            discount = 1.0
            for _ in range(rollout_depth):
                rand_action = np.random.randint(n_actions)
                with torch.no_grad():
                    next_state = world_model(
                        rollout_state.unsqueeze(0),
                        torch.tensor([rand_action], dtype=torch.int64)
                    )[0]
                next_state_np = next_state.detach().cpu().numpy()
                if imagined_state_recorder is not None:
                    imagined_state_recorder(next_state_np.copy())
                r = reward_fn(next_state_np, rand_action)
                rollout_return += discount * r
                discount *= gamma
                rollout_state = next_state
            leaf_value = rollout_return

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
