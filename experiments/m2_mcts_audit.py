"""M2 MCTS Audit: Policy Regime Map and Q-value Analysis

收集 SD-MCTS 在不同状态下的决策数据，用于:
1. Policy Regime Map: 动作分布 vs (queue, backlog, 距疲劳阈值距离)
2. Q-value Audit: 关键决策点的各动作价值对比
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.task_allocation_env import TaskAllocationEnv
from src.world_model.task_sd_model import TaskSDModel
from src.planning.mcts import mcts_search
from src.agent.networks import PolicyNetwork, ValueNetwork


@dataclass
class MCTSDecisionRecord:
    """单次 MCTS 决策记录"""
    step: int
    queue: float
    backlogs: List[float]
    distance_to_threshold: float  # min(threshold - backlog) for all workers
    total_load: float  # queue + sum(backlogs)
    action: int
    policy: List[float]  # 各动作的选择概率 (基于访问次数)
    q_values: List[float]  # 各动作的Q值
    root_value: float


def collect_mcts_decisions(
    scenario_name: str,
    arrival_rate: float = 1.0,
    fatigue_threshold: float = 15.0,
    worker_rates: Tuple[float, ...] = (1.0, 0.8),
    num_episodes: int = 5,
    episode_length: int = 100,
    num_simulations: int = 50,
    seed: int = 42,
) -> List[MCTSDecisionRecord]:
    """收集 MCTS 决策数据"""

    # 创建环境
    env = TaskAllocationEnv(
        max_queue=50,
        max_backlog=50,
        arrival_rate=arrival_rate,
        worker_rates=worker_rates,
        fatigue_threshold=fatigue_threshold,
        fatigue_penalty=0.5,
        seed=seed,
    )

    n_workers = len(worker_rates)
    state_dim = 1 + n_workers
    n_actions = n_workers + 1  # 分配给各worker + hold

    # 创建模型
    world_model = TaskSDModel(
        n_workers=n_workers,
        arrival_rate=arrival_rate,
        fatigue_threshold=fatigue_threshold,
        fatigue_penalty=0.5,
        learn_arrival_rate=True,
    )
    policy_net = PolicyNetwork(state_dim, n_actions)
    value_net = ValueNetwork(state_dim)

    # 简单训练 (复用现有逻辑) - 增加训练步数
    _quick_train(env, world_model, policy_net, value_net, steps=5000)

    records = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep * 100)

        for step in range(episode_length):
            # 使用 MCTS 获取详细决策信息
            result = mcts_search(
                root_state=state,
                world_model=world_model,
                n_actions=n_actions,
                num_simulations=num_simulations,
                policy_net=policy_net,
                value_net=value_net,
                return_details=True,
            )

            queue = float(state[0])
            backlogs = state[1:].tolist()

            # 计算距疲劳阈值的距离
            if fatigue_threshold is not None:
                distance_to_threshold = min(fatigue_threshold - b for b in backlogs)
            else:
                distance_to_threshold = float('inf')

            total_load = queue + sum(backlogs)

            record = MCTSDecisionRecord(
                step=ep * episode_length + step,
                queue=queue,
                backlogs=backlogs,
                distance_to_threshold=distance_to_threshold,
                total_load=total_load,
                action=result.action,
                policy=result.policy.tolist(),
                q_values=result.q_values.tolist() if result.q_values is not None else [],
                root_value=result.root_value,
            )
            records.append(record)

            # 执行动作
            state, _, _, _, _ = env.step(result.action)

    return records


def _quick_train(env, world_model, policy_net, value_net, steps: int = 2000):
    """快速训练模型 (简化版)"""
    optimizer = torch.optim.Adam(
        list(world_model.parameters()) +
        list(policy_net.parameters()) +
        list(value_net.parameters()),
        lr=1e-3
    )

    state, _ = env.reset()

    for step in range(steps):
        action = env.action_space.sample()
        next_state, reward, _, _, _ = env.step(action)

        # 训练世界模型
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.int64)
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        pred_next = world_model(state_t, action_t)
        wm_loss = torch.nn.functional.mse_loss(pred_next, next_state_t)

        # 训练价值网络
        value_pred = value_net(state_t)
        value_target = torch.tensor([[reward]], dtype=torch.float32)
        value_loss = torch.nn.functional.mse_loss(value_pred, value_target)

        loss = wm_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if step % 100 == 0:
            state, _ = env.reset()


def run_m2_audit(output_dir: str = "results/m2_insights"):
    """运行 M2 MCTS 审计实验"""
    os.makedirs(output_dir, exist_ok=True)

    scenarios = [
        # 低负载场景
        {"name": "low_load", "arrival_rate": 0.5, "fatigue_threshold": 15.0},
        # 中负载场景
        {"name": "medium_load", "arrival_rate": 0.9, "fatigue_threshold": 15.0},
        # 高负载场景 - 提高到达率使队列积压
        {"name": "high_load", "arrival_rate": 1.5, "fatigue_threshold": 15.0},
        # 极低阈值场景 - 更容易触发疲劳
        {"name": "low_threshold", "arrival_rate": 1.2, "fatigue_threshold": 5.0},
        # 临界场景 - 高到达率 + 低阈值
        {"name": "critical", "arrival_rate": 1.5, "fatigue_threshold": 8.0},
    ]

    all_records = {}

    for scenario in scenarios:
        print(f"Collecting data for scenario: {scenario['name']}...")
        records = collect_mcts_decisions(
            scenario_name=scenario["name"],
            arrival_rate=scenario["arrival_rate"],
            fatigue_threshold=scenario["fatigue_threshold"],
            num_episodes=3,
            episode_length=100,
            num_simulations=50,
        )
        all_records[scenario["name"]] = [asdict(r) for r in records]
        print(f"  Collected {len(records)} decisions")

    # 保存数据
    output_path = os.path.join(output_dir, "mcts_audit_data.json")
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nData saved to {output_path}")

    return all_records


if __name__ == "__main__":
    run_m2_audit()
