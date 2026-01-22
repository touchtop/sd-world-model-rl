"""
对比实验：Model-Free RL (PPO) vs Model-Based RL (SD+MCTS)

实验设计：
- 固定真实环境交互次数
- 对比两种方法的学习曲线和最终性能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from src.env import TaskAllocationEnv
from src.world_model import TaskSDModel
from src.planning import mcts_search
from src.agent import PolicyNetwork, ValueNetwork
from train_muzero import collect_episode, train_step, ReplayBuffer


class SimplePPO:
    """简化版PPO实现"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

    def get_action(self, state, deterministic=False):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state_t)
        if deterministic:
            return probs.argmax(dim=-1).item()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.value(state_t).item()

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(10):  # PPO epochs
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.value(states).squeeze()
            value_loss = nn.MSELoss()(values, returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train_ppo_baseline(total_timesteps=6000):
    """训练PPO基线"""
    print("\n[PPO Baseline] 开始训练...")

    env = TaskAllocationEnv()
    eval_env = TaskAllocationEnv(seed=999)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = SimplePPO(state_dim, n_actions)

    eval_steps = []
    eval_rewards = []
    env_steps = 0

    while env_steps < total_timesteps:
        # 收集一个episode
        states, actions, log_probs, rewards, values = [], [], [], [], []
        state, _ = env.reset()

        for _ in range(100):
            action, log_prob = agent.get_action(state)
            value = agent.get_value(state)

            next_state, reward, term, trunc, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value)

            state = next_state
            env_steps += 1

            if term or trunc:
                break

        # 计算returns和advantages
        returns = []
        advantages = []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + agent.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - v)

        # 更新
        agent.update(states, actions, log_probs, returns, advantages)

        # 评估
        if env_steps >= (len(eval_steps) + 1) * 300:
            rews = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                total = 0
                for _ in range(100):
                    action = agent.get_action(obs, deterministic=True)
                    obs, reward, term, trunc, _ = eval_env.step(action)
                    total += reward
                    if term or trunc:
                        break
                rews.append(total)
            eval_steps.append(env_steps)
            eval_rewards.append(np.mean(rews))
            print(f"  Step {env_steps}: eval_reward={np.mean(rews):.2f}")

    # 最终评估
    final_rewards = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        total = 0
        for _ in range(100):
            action = agent.get_action(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(action)
            total += reward
            if term or trunc:
                break
        final_rewards.append(total)

    return {
        'eval_steps': eval_steps,
        'eval_rewards': eval_rewards,
        'final_reward': np.mean(final_rewards),
        'total_env_steps': env_steps,
    }


def train_sd_mcts(total_env_steps=6000):
    """训练SD+MCTS方法"""
    print("\n[SD+MCTS] 开始训练...")

    env = TaskAllocationEnv()
    eval_env = TaskAllocationEnv(seed=999)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = TaskSDModel(n_workers=2)
    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    model_optimizer = optim.Adam(world_model.parameters(), lr=1e-2)

    buffer = ReplayBuffer(capacity=10000)

    eval_steps = []
    eval_rewards = []
    env_steps_used = 0

    while env_steps_used < total_env_steps:
        # 收集数据（使用当前网络指导MCTS）
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net,
            value_net=value_net,
            num_simulations=20
        )
        env_steps_used += len(transitions)
        for t in transitions:
            buffer.add(t)

        # 训练
        for _ in range(5):
            train_step(
                policy_net, value_net, world_model,
                buffer,
                policy_optimizer, value_optimizer, model_optimizer,
                batch_size=64
            )

        # 评估（使用MCTS而非policy_net）
        if env_steps_used >= (len(eval_steps) + 1) * 300:
            rewards = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                total = 0
                for _ in range(100):
                    action = mcts_search(
                        obs, world_model,
                        n_actions=n_actions,
                        num_simulations=50,
                        policy_net=policy_net,
                        value_net=value_net
                    )
                    obs, reward, term, trunc, _ = eval_env.step(action)
                    total += reward
                    if term or trunc:
                        break
                rewards.append(total)
            eval_steps.append(env_steps_used)
            eval_rewards.append(np.mean(rewards))
            print(f"  Step {env_steps_used}: eval_reward={np.mean(rewards):.2f}")

    # 最终评估（使用MCTS）
    final_rewards = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        total = 0
        for _ in range(100):
            action = mcts_search(
                obs, world_model,
                n_actions=n_actions,
                num_simulations=30,
                policy_net=policy_net,
                value_net=value_net
            )
            obs, reward, term, trunc, _ = eval_env.step(action)
            total += reward
            if term or trunc:
                break
        final_rewards.append(total)

    return {
        'eval_steps': eval_steps,
        'eval_rewards': eval_rewards,
        'final_reward': np.mean(final_rewards),
        'total_env_steps': env_steps_used,
        'sd_params': world_model.completion_coef.detach().numpy(),
        'arrival_rate': world_model.arrival_rate.item(),
    }


def main():
    print("=" * 70)
    print("对比实验: Model-Free RL (PPO) vs Model-Based RL (SD+MCTS)")
    print("=" * 70)

    TOTAL_ENV_STEPS = 10000

    print(f"\n实验配置: 真实环境交互次数 = {TOTAL_ENV_STEPS}")
    print(f"评估环境: TaskAllocationEnv, Workers=[1.0, 0.8]")

    # 训练两种方法
    ppo_results = train_ppo_baseline(total_timesteps=TOTAL_ENV_STEPS)
    sd_results = train_sd_mcts(total_env_steps=TOTAL_ENV_STEPS)

    # 打印结果
    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)

    print(f"\n[PPO Baseline]")
    print(f"  总环境交互: {ppo_results['total_env_steps']}")
    print(f"  最终奖励: {ppo_results['final_reward']:.2f}")

    print(f"\n[SD+MCTS]")
    print(f"  总环境交互: {sd_results['total_env_steps']}")
    print(f"  最终奖励: {sd_results['final_reward']:.2f}")
    print(f"  学到的SD参数: completion={sd_results['sd_params']}, arrival={sd_results['arrival_rate']:.3f}")
    print(f"  真实参数: Worker速率=[1.0, 0.8], arrival=0.7")

    # 性能对比
    if ppo_results['final_reward'] != 0:
        improvement = (sd_results['final_reward'] - ppo_results['final_reward']) / abs(ppo_results['final_reward']) * 100
        print(f"\n[对比] SD+MCTS 相对PPO: {improvement:+.1f}%")

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_results['eval_steps'], ppo_results['eval_rewards'], 'b-o', label='PPO (Model-Free)', markersize=4)
    plt.plot(sd_results['eval_steps'], sd_results['eval_rewards'], 'r-s', label='SD+MCTS (Model-Based)', markersize=4)
    plt.xlabel('Environment Steps')
    plt.ylabel('Evaluation Reward')
    plt.title('Learning Curve: PPO vs SD+MCTS')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_results.png', dpi=150)
    print(f"\n学习曲线已保存: comparison_results.png")

    print("=" * 70)


if __name__ == "__main__":
    main()
