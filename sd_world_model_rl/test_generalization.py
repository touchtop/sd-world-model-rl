"""
泛化能力测试：训练时用arrival_rate=0.7，测试时用不同的arrival_rate

假设：SD模型学到的"物理规律"比PPO更具泛化性
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.env import TaskAllocationEnv
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
    """训练SD+MCTS"""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    world_model = TaskSDModel(n_workers=2, arrival_rate=env.arrival_rate)
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


def evaluate_agent(agent_type, agent_data, env, num_episodes=10):
    """评估agent在给定环境上的表现"""
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0

        for _ in range(100):
            if agent_type == 'ppo':
                action = agent_data.get_action(state, deterministic=True)
            elif agent_type == 'sd_policy':
                policy_net = agent_data
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = policy_net.get_action(state_t, deterministic=True)
            elif agent_type == 'sd_mcts':
                policy_net, value_net, world_model = agent_data
                action = mcts_search(
                    state, world_model,
                    n_actions=env.action_space.n,
                    num_simulations=30,
                    policy_net=policy_net,
                    value_net=value_net
                )

            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                break

        rewards.append(ep_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    print("=" * 70)
    print("泛化能力测试: 训练arrival_rate=0.7, 测试不同arrival_rate")
    print("=" * 70)

    # 训练环境
    train_env = TaskAllocationEnv(arrival_rate=0.7, seed=42)
    TRAIN_STEPS = 8000

    print(f"\n训练配置: {TRAIN_STEPS} steps, arrival_rate=0.7")

    # 训练两种方法
    print("\n[1] 训练PPO...")
    ppo_agent = train_ppo(train_env, total_timesteps=TRAIN_STEPS)

    print("[2] 训练SD+MCTS...")
    policy_net, value_net, world_model = train_sd_mcts(train_env, total_env_steps=TRAIN_STEPS)

    print(f"\n学到的SD参数:")
    print(f"  completion_coef: {world_model.completion_coef.detach().numpy()} (真实: {train_env.worker_rates})")
    print(f"  arrival_rate: {world_model.arrival_rate.item():.3f} (训练时真实: 0.7)")

    # 测试不同arrival_rate
    test_rates = [0.5, 0.7, 1.0, 1.3, 1.5]
    print(f"\n测试arrival_rates: {test_rates}")

    results = {'rate': [], 'ppo': [], 'sd_policy': [], 'sd_mcts': []}

    for rate in test_rates:
        test_env = TaskAllocationEnv(arrival_rate=rate, seed=999)

        ppo_reward, _ = evaluate_agent('ppo', ppo_agent, test_env)
        sd_policy_reward, _ = evaluate_agent('sd_policy', policy_net, test_env)
        sd_mcts_reward, _ = evaluate_agent('sd_mcts', (policy_net, value_net, world_model), test_env)

        results['rate'].append(rate)
        results['ppo'].append(ppo_reward)
        results['sd_policy'].append(sd_policy_reward)
        results['sd_mcts'].append(sd_mcts_reward)

        print(f"  rate={rate}: PPO={ppo_reward:.1f}, SD_Policy={sd_policy_reward:.1f}, SD_MCTS={sd_mcts_reward:.1f}")

    # 计算泛化分数（OOD表现相对于训练分布的保持程度）
    print("\n" + "=" * 70)
    print("泛化分析")
    print("=" * 70)

    train_idx = test_rates.index(0.7)
    for method in ['ppo', 'sd_policy', 'sd_mcts']:
        train_perf = results[method][train_idx]
        ood_perfs = [results[method][i] for i in range(len(test_rates)) if test_rates[i] != 0.7]
        avg_ood = np.mean(ood_perfs)
        degradation = (train_perf - avg_ood) / abs(train_perf) * 100 if train_perf != 0 else 0
        print(f"{method:12s}: 训练分布={train_perf:.1f}, OOD平均={avg_ood:.1f}, 性能下降={degradation:.1f}%")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(results['rate'], results['ppo'], 'b-o', label='PPO', markersize=8)
    plt.plot(results['rate'], results['sd_policy'], 'r-s', label='SD+Policy', markersize=8)
    plt.plot(results['rate'], results['sd_mcts'], 'g-^', label='SD+MCTS', markersize=8)
    plt.axvline(x=0.7, color='gray', linestyle='--', label='Training rate')
    plt.xlabel('Arrival Rate (test)')
    plt.ylabel('Reward')
    plt.title('Generalization Test: Performance vs Arrival Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('generalization_test.png', dpi=150)
    print(f"\n图表已保存: generalization_test.png")


if __name__ == "__main__":
    main()
