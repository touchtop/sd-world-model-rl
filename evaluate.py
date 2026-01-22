import numpy as np
import torch
import matplotlib.pyplot as plt
from src.env import TaskAllocationEnv
from src.world_model import TaskSDModel
from src.planning import mcts_search


def random_policy(state, n_actions):
    return np.random.randint(0, n_actions)


def greedy_policy(state, n_actions):
    """贪心策略：分配给负载最低的Worker，若队列空则hold"""
    if state[0] <= 0:
        return n_actions - 1  # hold
    backlogs = state[1:]
    return int(np.argmin(backlogs))


def evaluate_policy(env, policy_fn, steps=200, seed=42):
    """评估策略的累积奖励和状态轨迹"""
    state, _ = env.reset(seed=seed)
    rewards = []
    states = [state.copy()]

    for _ in range(steps):
        action = policy_fn(state, env.action_space.n)
        state, reward, _, _, _ = env.step(action)
        rewards.append(reward)
        states.append(state.copy())

    return np.array(rewards), np.array(states)


def mcts_policy_factory(world_model, num_simulations=30):
    """创建MCTS策略函数"""
    def policy(state, n_actions):
        return mcts_search(state, world_model, n_actions=n_actions, num_simulations=num_simulations)
    return policy


def evaluate_world_model_accuracy(env, world_model, steps=100, seed=42):
    """评估世界模型预测精度"""
    state, _ = env.reset(seed=seed)
    errors = []

    for _ in range(steps):
        action = np.random.randint(0, env.action_space.n)

        # 世界模型预测
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.int64)
        with torch.no_grad():
            pred_next = world_model(state_t, action_t)[0].numpy()

        # 真实环境
        true_next, _, _, _, _ = env.step(action)

        # 计算预测误差
        error = np.abs(pred_next - true_next).mean()
        errors.append(error)
        state = true_next

    return np.array(errors)


def train_and_evaluate():
    print("=" * 60)
    print("SD World Model + MCTS 验证实验")
    print("=" * 60)

    env = TaskAllocationEnv(seed=42)
    world_model = TaskSDModel(n_workers=2)

    # --- 1. 训练世界模型 ---
    print("\n[1] 训练SD世界模型...")
    from main import collect_trajectory, train_world_model

    print(f"    真实Worker处理率: {env.worker_rates}")
    print(f"    初始模型参数: {world_model.completion_coef.detach().numpy()}")

    losses = []
    for i in range(5):
        states, actions, next_states = collect_trajectory(env, world_model, steps=100)
        loss = train_world_model(world_model, states, actions, next_states, epochs=50)
        losses.append(loss)
        print(f"    Iter {i+1}: loss={loss:.4f}, params={world_model.completion_coef.detach().numpy()}")

    print(f"    最终模型参数: {world_model.completion_coef.detach().numpy()}")

    # --- 2. 世界模型预测精度 ---
    print("\n[2] 世界模型预测精度...")
    errors = evaluate_world_model_accuracy(env, world_model, steps=100)
    print(f"    平均预测误差 (MAE): {errors.mean():.4f}")
    print(f"    最大预测误差: {errors.max():.4f}")

    # --- 3. 策略对比 ---
    print("\n[3] 策略对比 (200 steps)...")

    mcts_policy = mcts_policy_factory(world_model, num_simulations=30)

    rewards_random, states_random = evaluate_policy(env, random_policy, steps=200)
    rewards_greedy, states_greedy = evaluate_policy(env, greedy_policy, steps=200)
    rewards_mcts, states_mcts = evaluate_policy(env, mcts_policy, steps=200)

    print(f"    随机策略 累积奖励: {rewards_random.sum():.2f}")
    print(f"    贪心策略 累积奖励: {rewards_greedy.sum():.2f}")
    print(f"    MCTS策略 累积奖励: {rewards_mcts.sum():.2f}")

    # --- 4. 可视化 ---
    print("\n[4] 生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 4.1 训练损失
    axes[0, 0].plot(range(1, len(losses)+1), losses, 'o-', color='blue')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('World Model Training Loss')
    axes[0, 0].grid(True)

    # 4.2 累积奖励对比
    cumsum_random = np.cumsum(rewards_random)
    cumsum_greedy = np.cumsum(rewards_greedy)
    cumsum_mcts = np.cumsum(rewards_mcts)

    axes[0, 1].plot(cumsum_random, label='Random', alpha=0.7)
    axes[0, 1].plot(cumsum_greedy, label='Greedy', alpha=0.7)
    axes[0, 1].plot(cumsum_mcts, label='MCTS+SD', alpha=0.7)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].set_title('Policy Comparison: Cumulative Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 4.3 队列长度对比
    axes[1, 0].plot(states_random[:, 0], label='Random', alpha=0.7)
    axes[1, 0].plot(states_greedy[:, 0], label='Greedy', alpha=0.7)
    axes[1, 0].plot(states_mcts[:, 0], label='MCTS+SD', alpha=0.7)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Queue Length')
    axes[1, 0].set_title('Queue Length Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4.4 Worker负载 (MCTS策略)
    axes[1, 1].plot(states_mcts[:, 1], label='Worker 1', alpha=0.7)
    axes[1, 1].plot(states_mcts[:, 2], label='Worker 2', alpha=0.7)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Backlog')
    axes[1, 1].set_title('Worker Backlogs (MCTS+SD Policy)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    print(f"    图表已保存: evaluation_results.png")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
