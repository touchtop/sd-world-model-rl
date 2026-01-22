"""
MuZero-Lite 训练脚本

训练流程:
1. 用MCTS收集数据 (state, mcts_policy, cumulative_return)
2. 训练Policy Network模仿MCTS策略
3. 训练Value Network预测回报
4. 训练SD World Model预测状态转移
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from typing import List

from src.env import TaskAllocationEnv
from src.world_model import TaskSDModel
from src.planning import mcts_search
from src.agent import PolicyNetwork, ValueNetwork


@dataclass
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    mcts_policy: np.ndarray
    cumulative_return: float = 0.0
    collapse_occurred: float = 0.0  # 1.0 if collapse triggered this step


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


def collect_episode(env, world_model, n_actions, policy_net=None, value_net=None, num_simulations=30, gamma=0.99, n_workers=None):
    """收集一个episode的数据，包含MCTS策略分布"""
    transitions = []
    state, _ = env.reset()

    for _ in range(100):  # 最大步数
        result = mcts_search(
            state, world_model,
            n_actions=n_actions,
            num_simulations=num_simulations,
            gamma=gamma,
            policy_net=policy_net,
            value_net=value_net,
            return_details=True,
            n_workers=n_workers
        )

        next_state, reward, terminated, truncated, info = env.step(result.action)

        # Extract collapse signal from environment info
        collapse_occurred = 1.0 if info.get('collapse_triggered', False) else 0.0

        transitions.append(Transition(
            state=state.copy(),
            action=result.action,
            next_state=next_state.copy(),
            reward=reward,
            mcts_policy=result.policy.copy(),
            collapse_occurred=collapse_occurred,
        ))

        state = next_state
        if terminated or truncated:
            break

    # 计算累积回报 (从后往前)
    cumulative = 0.0
    for t in reversed(transitions):
        cumulative = t.reward + gamma * cumulative
        t.cumulative_return = cumulative

    return transitions


def train_step(
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    world_model: TaskSDModel,
    buffer: ReplayBuffer,
    policy_optimizer: optim.Optimizer,
    value_optimizer: optim.Optimizer,
    model_optimizer: optim.Optimizer,
    batch_size: int = 64,
):
    """执行一次训练步骤"""
    if len(buffer) < batch_size:
        return {}

    batch = buffer.sample(batch_size)

    states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
    actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
    next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
    mcts_policies = torch.tensor(np.array([t.mcts_policy for t in batch]), dtype=torch.float32)
    returns = torch.tensor([t.cumulative_return for t in batch], dtype=torch.float32)

    # 1. Policy Loss: CrossEntropy(policy_net, mcts_policy)
    policy_logits = policy_net(states)
    policy_loss = -(mcts_policies * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # 2. Value Loss: MSE(value_net, returns)
    value_pred = value_net(states)
    value_loss = nn.MSELoss()(value_pred, returns)

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # 3. World Model Loss: MSE(predicted_next_state, actual_next_state)
    pred_next = world_model(states, actions)
    model_loss = nn.MSELoss()(pred_next, next_states)

    # 4. BCE Loss for collapse supervision (if world model supports it)
    bce_loss = torch.tensor(0.0)
    if hasattr(world_model, 'learn_probabilistic_collapse') and world_model.learn_probabilistic_collapse:
        collapse_labels = torch.tensor([t.collapse_occurred for t in batch], dtype=torch.float32)
        pred_collapse_prob = world_model.get_collapse_prob(states)
        # Weighted BCE: pos_weight=20.0 to handle rare collapse events
        bce_loss = nn.BCELoss()(pred_collapse_prob, collapse_labels)
        # Alternative weighted version for very sparse collapses:
        # pos_weight = torch.tensor(20.0)
        # bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, collapse_labels)
        model_loss = model_loss + 0.5 * bce_loss  # Weighted combination

    model_optimizer.zero_grad()
    model_loss.backward()
    model_optimizer.step()

    result = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'model_loss': model_loss.item(),
    }
    if hasattr(world_model, 'learn_probabilistic_collapse') and world_model.learn_probabilistic_collapse:
        result['bce_loss'] = bce_loss.item()
    return result


def evaluate(env, policy_net, num_episodes=5):
    """用训练好的策略网络评估"""
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(100):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy_net.get_action(state_t, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def main():
    print("=" * 60)
    print("MuZero-Lite 训练")
    print("=" * 60)

    # 初始化
    env = TaskAllocationEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = TaskSDModel(n_workers=2)
    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    model_optimizer = optim.Adam(world_model.parameters(), lr=1e-2)

    buffer = ReplayBuffer(capacity=10000)

    # 训练参数
    num_iterations = 20
    episodes_per_iter = 3
    train_steps_per_iter = 10

    print(f"\n配置: {num_iterations} iterations, {episodes_per_iter} episodes/iter")
    print(f"State dim: {state_dim}, Actions: {n_actions}")
    print(f"真实Worker处理率: {env.worker_rates}")

    # 训练循环
    for iteration in range(num_iterations):
        # 收集数据（使用当前的policy_net和value_net指导MCTS）
        for _ in range(episodes_per_iter):
            transitions = collect_episode(
                env, world_model, n_actions,
                policy_net=policy_net,
                value_net=value_net,
                num_simulations=20
            )
            for t in transitions:
                buffer.add(t)

        # 训练
        losses = {'policy_loss': 0, 'value_loss': 0, 'model_loss': 0}
        for _ in range(train_steps_per_iter):
            step_losses = train_step(
                policy_net, value_net, world_model,
                buffer,
                policy_optimizer, value_optimizer, model_optimizer
            )
            for k, v in step_losses.items():
                losses[k] += v / train_steps_per_iter

        # 评估
        eval_reward = evaluate(env, policy_net, num_episodes=3)

        print(f"Iter {iteration+1:2d}: "
              f"policy_loss={losses['policy_loss']:.4f}, "
              f"value_loss={losses['value_loss']:.4f}, "
              f"model_loss={losses['model_loss']:.4f}, "
              f"eval_reward={eval_reward:.2f}")

    # 最终评估对比
    print("\n" + "=" * 60)
    print("最终评估对比")
    print("=" * 60)

    # 用训练好的策略网络
    policy_reward = evaluate(env, policy_net, num_episodes=10)
    print(f"训练后Policy Network: {policy_reward:.2f}")

    # MCTS (使用训练后的世界模型和网络)
    mcts_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(100):
            action = mcts_search(
                state, world_model,
                n_actions=n_actions,
                num_simulations=30,
                policy_net=policy_net,
                value_net=value_net
            )
            state, reward, _, _, _ = env.step(action)
            ep_reward += reward
        mcts_rewards.append(ep_reward)
    print(f"MCTS+SD (训练后): {np.mean(mcts_rewards):.2f}")

    # 随机策略
    random_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(100):
            action = np.random.randint(0, n_actions)
            state, reward, _, _, _ = env.step(action)
            ep_reward += reward
        random_rewards.append(ep_reward)
    print(f"随机策略: {np.mean(random_rewards):.2f}")

    print(f"\nSD模型参数: {world_model.completion_coef.detach().numpy()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
