import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env import TaskAllocationEnv
from src.world_model import TaskSDModel
from src.planning import mcts_search


def collect_trajectory(env: TaskAllocationEnv, world_model: TaskSDModel, steps: int = 50):
    """收集真实环境轨迹"""
    states, actions, next_states = [], [], []
    state, _ = env.reset()

    for _ in range(steps):
        action = mcts_search(state, world_model, n_actions=env.action_space.n, num_simulations=30)
        next_state, _, _, _, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = next_state

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.int64),
        torch.tensor(np.array(next_states), dtype=torch.float32),
    )


def train_world_model(
    model: TaskSDModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-2,
):
    """训练SD世界模型参数"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pred = model(states, actions)
        loss = loss_fn(pred, next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate(env: TaskAllocationEnv, world_model: TaskSDModel, steps: int = 20):
    """评估Agent表现"""
    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(steps):
        action = mcts_search(state, world_model, n_actions=env.action_space.n, num_simulations=50)
        state, reward, _, _, _ = env.step(action)
        total_reward += reward

    return total_reward


def main():
    env = TaskAllocationEnv()
    world_model = TaskSDModel(n_workers=2)

    print("=== SD World Model + MCTS Training ===")
    print(f"Initial SD params: {world_model.completion_coef.data.numpy()}")

    for iteration in range(5):
        # 收集轨迹
        states, actions, next_states = collect_trajectory(env, world_model, steps=100)

        # 训练世界模型
        loss = train_world_model(world_model, states, actions, next_states)
        print(f"Iter {iteration+1}: loss={loss:.4f}, params={world_model.completion_coef.data.numpy()}")

    # 评估
    reward = evaluate(env, world_model, steps=50)
    print(f"\nEval total reward: {reward:.2f}")
    print(f"Final SD params: {world_model.completion_coef.data.numpy()}")
    print(f"True worker rates: {env.worker_rates}")


if __name__ == "__main__":
    main()
