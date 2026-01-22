"""
疲劳机制测试：验证Worker疲劳使前瞻规划变得必要

假设：在有疲劳惩罚的环境中，需要预测性地分配任务以避免
任何worker积压过高导致效率下降，SD模型可以捕捉这种非线性动态
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.env import TaskAllocationEnv, TaskAllocationEnvWithFatigue
from src.world_model import TaskSDModel
from src.planning import mcts_search
from src.agent import PolicyNetwork, ValueNetwork
from train_muzero import collect_episode, train_step, ReplayBuffer
from compare_methods import SimplePPO


def train_ppo(env, total_timesteps=5000):
    """训练PPO"""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = SimplePPO(state_dim, n_actions)

    env_steps = 0
    while env_steps < total_timesteps:
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

        returns, advantages = [], []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + agent.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - v)

        agent.update(states, actions, log_probs, returns, advantages)

    return agent


def train_sd_mcts(env, total_env_steps=5000):
    """训练SD+MCTS（支持疲劳机制）"""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 创建带疲劳机制的世界模型
    world_model = TaskSDModel(
        n_workers=2,
        arrival_rate=env.arrival_rate,
        fatigue_threshold=getattr(env, 'fatigue_threshold', None),
        fatigue_penalty=getattr(env, 'fatigue_penalty', 0.5),
    )
    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    model_optimizer = optim.Adam(world_model.parameters(), lr=0.05)

    buffer = ReplayBuffer(capacity=10000)
    env_steps_used = 0

    while env_steps_used < total_env_steps:
        transitions = collect_episode(
            env, world_model, n_actions,
            policy_net=policy_net, value_net=value_net,
            num_simulations=20
        )
        env_steps_used += len(transitions)
        for t in transitions:
            buffer.add(t)

        for _ in range(5):
            train_step(
                policy_net, value_net, world_model,
                buffer,
                policy_optimizer, value_optimizer, model_optimizer,
                batch_size=64
            )

    return policy_net, value_net, world_model


def evaluate_with_mcts(policy_net, value_net, world_model, env, num_episodes=10):
    """使用MCTS直接评估"""
    rewards = []
    n_actions = env.action_space.n

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        for _ in range(100):
            action = mcts_search(
                state, world_model,
                n_actions=n_actions,
                num_simulations=50,
                policy_net=policy_net,
                value_net=value_net
            )
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                break

        rewards.append(ep_reward)

    return np.mean(rewards), np.std(rewards)


def evaluate_ppo(agent, env, num_episodes=10):
    """评估PPO"""
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        for _ in range(100):
            action = agent.get_action(state, deterministic=True)
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                break

        rewards.append(ep_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    print("=" * 70)
    print("疲劳机制测试: 验证非线性动态使前瞻规划变得必要")
    print("=" * 70)

    TRAIN_STEPS = 8000

    # 测试不同疲劳阈值
    thresholds = [None, 20, 15, 10]
    results = {'threshold': [], 'ppo': [], 'sd_mcts': []}

    for threshold in thresholds:
        print(f"\n--- 疲劳阈值: {threshold if threshold else '无(基线)'} ---")

        if threshold is None:
            env = TaskAllocationEnv(arrival_rate=0.7, seed=42)
        else:
            env = TaskAllocationEnvWithFatigue(
                arrival_rate=0.7,
                fatigue_threshold=threshold,
                fatigue_penalty=0.5,
                seed=42
            )

        print(f"  训练PPO ({TRAIN_STEPS} steps)...")
        ppo_agent = train_ppo(env, total_timesteps=TRAIN_STEPS)

        print(f"  训练SD+MCTS ({TRAIN_STEPS} steps)...")
        policy_net, value_net, world_model = train_sd_mcts(env, total_env_steps=TRAIN_STEPS)

        # 评估
        test_env = TaskAllocationEnv(
            arrival_rate=0.7,
            fatigue_threshold=threshold,
            fatigue_penalty=0.5 if threshold else 1.0,
            seed=999
        )

        ppo_reward, ppo_std = evaluate_ppo(ppo_agent, test_env)
        sd_reward, sd_std = evaluate_with_mcts(policy_net, value_net, world_model, test_env)

        print(f"  PPO: {ppo_reward:.1f} ± {ppo_std:.1f}")
        print(f"  SD+MCTS: {sd_reward:.1f} ± {sd_std:.1f}")

        # 显示学到的参数
        print(f"  SD学习的参数: completion={world_model.completion_coef.detach().numpy()}, arrival={world_model.arrival_rate.item():.3f}")
        print(f"  真实参数: worker_rates={env.base_worker_rates}, arrival=0.7")

        results['threshold'].append(threshold if threshold else 'None')
        results['ppo'].append(ppo_reward)
        results['sd_mcts'].append(sd_reward)

    # 分析结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'阈值':<10} {'PPO':<15} {'SD+MCTS':<15} {'差异':<15}")
    print("-" * 55)
    for i, th in enumerate(results['threshold']):
        ppo = results['ppo'][i]
        sd = results['sd_mcts'][i]
        diff = sd - ppo
        better = "SD更优" if diff > 0 else "PPO更优"
        print(f"{str(th):<10} {ppo:<15.1f} {sd:<15.1f} {diff:+.1f} ({better})")

    # 绘图
    x = range(len(thresholds))
    labels = [str(t) if t else 'None' for t in thresholds]

    plt.figure(figsize=(10, 6))
    width = 0.35
    plt.bar([i - width/2 for i in x], results['ppo'], width, label='PPO', color='blue', alpha=0.7)
    plt.bar([i + width/2 for i in x], results['sd_mcts'], width, label='SD+MCTS', color='green', alpha=0.7)
    plt.xlabel('Fatigue Threshold')
    plt.ylabel('Reward (higher is better)')
    plt.title('PPO vs SD+MCTS with Worker Fatigue Mechanism')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fatigue_test.png', dpi=150)
    print(f"\n图表已保存: fatigue_test.png")


if __name__ == "__main__":
    main()
