# Oracle MPC 与 Adaptive MPC 基线实验设计

## 概述

本文档描述 SD-MCTS 与 Oracle MPC、Adaptive MPC 基线的对比实验设计。目的是：
1. 通过 Oracle MPC 确立理论性能上界
2. 通过 Adaptive MPC 进行公平的"学习能力"对比
3. 完整展示 SD-MCTS 的优势条件和贡献

**文档日期**：2026-01-20

---

## 1. 实验动机

### 1.1 当前实验的局限性

| 问题 | 说明 |
|------|------|
| **基线不公平** | MPC 使用错误参数 + 短视野，胜之不武 |
| **无理论上界** | 不知道"完美信息"下能做多好 |
| **学习对比缺失** | MPC 不学习，无法对比"学习方法"差异 |

### 1.2 新增基线的作用

```
性能
  ↑
  │  ┌─────────────────────────────────────────────────┐
  │  │  Oracle MPC (真实参数 + 长视野)                │  ← 理论上界
  │  └─────────────────────────────────────────────────┘
  │          ↑
  │          │ 差距 A：SD-MCTS 能逼近多少？
  │          ↓
  │  ┌─────────────────────────────────────────────────┐
  │  │  SD-MCTS (从错误信念学习)                      │  ← 我们的方法
  │  └─────────────────────────────────────────────────┘
  │          ↑
  │          │ 差距 B：结构化学习 vs 统计估计
  │          ↓
  │  ┌─────────────────────────────────────────────────┐
  │  │  Adaptive MPC (RLS 参数估计 + MPC)             │  ← 公平对比
  │  └─────────────────────────────────────────────────┘
  │          ↑
  │          │ 差距 C：学习 vs 不学习
  │          ↓
  │  ┌─────────────────────────────────────────────────┐
  │  │  MPC (固定错误信念)                            │  ← 当前基线
  │  └─────────────────────────────────────────────────┘
  │
  └────────────────────────────────────────────────────→
```

---

## 2. Oracle MPC 设计

### 2.1 定义

Oracle MPC 是一个**理论基线**，假设拥有完美的环境知识：
- 真实的动力学参数（α, θ, F 等）
- 足够长的规划视野覆盖关键事件
- 无学习过程，直接使用真实参数

### 2.2 规格

#### 效率退化场景 (efficiency_degradation_fairness)

| 参数 | Oracle MPC | MPC (wrong) | 说明 |
|------|------------|-------------|------|
| α (效率衰减) | 0.10 (真实) | 0.001 | Oracle 用真实值 |
| θ (衰减阈值) | 5.0 (真实) | 12.0 | Oracle 用真实值 |
| 规划视野 H | 10 | 3 | 覆盖疲劳累积效应 |

#### 延迟崩溃场景 (delayed_collapse)

| 参数 | Oracle MPC | MPC (wrong) | 说明 |
|------|------------|-------------|------|
| F (崩溃阈值) | 10.0 (真实) | 50.0 | Oracle 用真实值 |
| α (疲劳累积) | 1.0 (真实) | 1.0 | 相同 |
| β (疲劳恢复) | 0.1 (真实) | 0.1 | 相同 |
| 规划视野 H | 15 | 3 | 覆盖峰值时刻 T_s=12 |

### 2.3 实现伪代码

```python
class OracleMPC:
    """使用真实参数的 MPC 基线"""

    def __init__(self, true_params: dict, horizon: int = 15):
        self.params = true_params  # 真实参数（非学习）
        self.horizon = horizon

    def select_action(self, state: np.ndarray) -> int:
        """H 步前瞻搜索"""
        best_action = self.n_workers  # hold
        best_value = float('-inf')

        for action in range(self.n_workers + 1):
            value = self._rollout(state, action, self.horizon)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _rollout(self, state, first_action, horizon):
        """使用真实参数进行 H 步模拟"""
        total_reward = 0
        sim_state = state.copy()

        for t in range(horizon):
            action = first_action if t == 0 else self._greedy_action(sim_state)
            sim_state, reward = self._simulate_step(sim_state, action, self.params)
            total_reward += reward * (self.gamma ** t)

        return total_reward
```

### 2.4 预期结果

| 场景 | Oracle MPC 预期 | 与 Optimal 差距 |
|------|----------------|-----------------|
| 效率退化 | ≈ -7500 | < 5% |
| 延迟崩溃 | ≈ 24.0 | < 5% |

---

## 3. Adaptive MPC 设计

### 3.1 定义

Adaptive MPC 是一个**公平对比基线**，与 SD-MCTS 一样：
- 从错误的初始信念开始
- 通过在线观测学习真实参数
- 使用学到的参数进行 MPC 规划

**区别**：使用经典统计方法（RLS/EKF）而非梯度下降

### 3.2 学习机制：递归最小二乘 (RLS)

#### 效率退化参数估计

目标：从观测 (fatigue, efficiency) 估计 α 和 θ

模型：
$$
\text{efficiency} = \frac{1}{1 + \alpha \cdot \max(0, \text{fatigue} - \theta)^2}
$$

变换为线性形式：
$$
y = \frac{1}{\text{efficiency}} - 1 = \alpha \cdot (\text{fatigue} - \theta)^2 \quad \text{when fatigue} > \theta
$$

RLS 更新规则：
$$
\hat{\theta}_{t+1} = \hat{\theta}_t + K_t (y_t - \hat{y}_t)
$$
$$
K_t = P_t \phi_t / (1 + \phi_t^T P_t \phi_t)
$$
$$
P_{t+1} = (I - K_t \phi_t^T) P_t
$$

#### 延迟崩溃参数估计

目标：从观测 (action, fatigue_next - fatigue) 估计 α, β, F

模型：
$$
\Delta f = (\alpha + \beta) \cdot a - \beta
$$

这是关于 (α, β) 的线性模型，可直接用 RLS。

崩溃阈值 F 的估计：
- 观测到崩溃时，记录 fatigue 值
- 用多次观测的最小值作为 F̂

### 3.3 规格

| 参数 | Adaptive MPC | SD-MCTS | 说明 |
|------|-------------|---------|------|
| 初始 α | 0.001 (错误) | 0.001 (错误) | 相同起点 |
| 初始 θ | 12.0 (错误) | 12.0 (错误) | 相同起点 |
| 学习方法 | RLS | 梯度下降 | 核心差异 |
| 规划视野 | H=3 / H=15 | Value Net (∞) | 对比条件 |
| 规划方法 | 前向模拟 | MCTS | 核心差异 |

### 3.4 实现伪代码

```python
class AdaptiveMPC:
    """带在线参数估计的 MPC"""

    def __init__(self, init_alpha, init_theta, horizon=3):
        # 初始信念（与 SD-MCTS 相同的错误值）
        self.alpha_hat = init_alpha  # 0.001
        self.theta_hat = init_theta  # 12.0
        self.horizon = horizon

        # RLS 状态
        self.P = np.eye(2) * 100.0  # 协方差矩阵（高初始不确定性）
        self.lambda_rls = 0.99  # 遗忘因子

    def update(self, fatigue: float, observed_efficiency: float):
        """RLS 参数更新"""
        if fatigue <= self.theta_hat:
            # 低于阈值时无信息量
            return

        # 构造观测
        diff = fatigue - self.theta_hat
        y = 1.0 / observed_efficiency - 1.0  # 变换后的观测

        # 构造回归向量 [∂y/∂α, ∂y/∂θ]
        # y = α * diff², ∂y/∂α = diff², ∂y/∂θ = -2α*diff
        phi = np.array([diff**2, -2 * self.alpha_hat * diff])

        # RLS 更新
        P_phi = self.P @ phi
        gain = P_phi / (self.lambda_rls + phi @ P_phi)

        prediction = self.alpha_hat * diff**2
        error = y - prediction

        # 更新参数（带约束）
        delta = gain * error
        self.alpha_hat = max(1e-6, self.alpha_hat + delta[0] * 0.1)  # 缓慢更新
        self.theta_hat = max(0.0, self.theta_hat + delta[1] * 0.1)

        # 更新协方差
        self.P = (self.P - np.outer(gain, phi @ self.P)) / self.lambda_rls

    def select_action(self, state: np.ndarray) -> int:
        """使用当前估计的参数进行 MPC"""
        params = {'alpha': self.alpha_hat, 'theta': self.theta_hat}

        best_action = self.n_workers
        best_value = float('-inf')

        for action in range(self.n_workers + 1):
            value = self._rollout(state, action, params, self.horizon)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
```

### 3.5 Adaptive MPC 变体

| 变体 | 视野 | 目的 |
|------|------|------|
| Adaptive MPC (H=3) | 短 | 公平对比学习能力（视野相同时） |
| Adaptive MPC (H=15) | 长 | 控制视野变量，仅对比学习方法 |

### 3.6 预期结果

| 场景 | Adaptive MPC (H=3) | Adaptive MPC (H=15) | SD-MCTS |
|------|-------------------|---------------------|---------|
| 效率退化 | 中等（视野限制） | 较好 | 最好（预期） |
| 延迟崩溃 | 差（看不到峰值） | 较好 | 最好（预期） |

---

## 4. 完整实验矩阵

### 4.1 效率退化场景

| 方法 | 初始 α | 初始 θ | 学习 | 视野 | 预期排名 |
|------|--------|--------|------|------|----------|
| MPC (wrong) | 0.001 | 12.0 | ❌ | H=3 | 5 (最差) |
| Adaptive MPC (H=3) | 0.001 | 12.0 | ✅ RLS | H=3 | 4 |
| Adaptive MPC (H=15) | 0.001 | 12.0 | ✅ RLS | H=15 | 3 |
| SD-MCTS | 0.001 | 12.0 | ✅ 梯度 | ∞ (VNet) | 2 |
| Oracle MPC | 0.10 | 5.0 | ❌ | H=15 | 1 (最好) |

### 4.2 延迟崩溃场景

| 方法 | 初始 F | 学习 | 视野 | 预期排名 |
|------|--------|------|------|----------|
| MPC (wrong) | 50.0 | ❌ | H=3 | 5 (崩溃) |
| Adaptive MPC (H=3) | 50.0 | ✅ RLS | H=3 | 4 (视野不足) |
| Adaptive MPC (H=15) | 50.0 | ✅ RLS | H=15 | 3 |
| SD-MCTS | 50.0 | ✅ 梯度 | ∞ (VNet) | 2 |
| Oracle MPC | 10.0 | ❌ | H=15 | 1 (≈Optimal) |

---

## 5. 评估指标

### 5.1 主要指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **累积奖励** | Σ r_t | 主要性能指标 |
| **参数恢复误差** | \|θ̂ - θ_true\| / θ_true | 学习准确度 |
| **收敛速度** | 达到 90% 最终性能的步数 | 样本效率 |
| **稳定性** | 奖励标准差 | 跨 seed 一致性 |

### 5.2 统计检验

| 检验 | 用途 |
|------|------|
| **Welch's t-test** | 两方法均值差异显著性 |
| **置信区间** | 95% CI 不重叠则显著 |
| **效应量 (Cohen's d)** | 差异的实际意义 |

### 5.3 实验规模

| 参数 | 值 | 说明 |
|------|-----|------|
| Seeds | 30 | 满足统计检验要求 |
| Episode 长度 | 200 (效率) / 30 (崩溃) | 场景特定 |
| 训练步数 | 20k (SD-MCTS) | 学习收敛 |
| 重复次数 | 每 seed 评估 10 episodes | 减少方差 |

---

## 6. 实现计划

### 6.1 文件结构

```
experiments/
├── baselines/
│   ├── oracle_mpc.py        # Oracle MPC 实现
│   ├── adaptive_mpc.py      # Adaptive MPC 实现
│   └── rls_estimator.py     # RLS 参数估计器
├── run_full_comparison.py   # 完整对比实验脚本
└── analyze_results.py       # 统计分析与可视化
```

### 6.2 开发顺序

| 阶段 | 任务 | 产出 |
|------|------|------|
| 1 | 实现 Oracle MPC | `oracle_mpc.py` |
| 2 | 验证 Oracle ≈ Optimal | 确认理论上界 |
| 3 | 实现 RLS 估计器 | `rls_estimator.py` |
| 4 | 实现 Adaptive MPC | `adaptive_mpc.py` |
| 5 | 运行完整对比 | 30 seeds × 5 methods |
| 6 | 统计分析 | 表格 + 图表 |

### 6.3 预计工作量

| 阶段 | 估计代码行数 | 复杂度 |
|------|-------------|--------|
| Oracle MPC | ~100 LOC | 低 |
| RLS 估计器 | ~80 LOC | 中 |
| Adaptive MPC | ~150 LOC | 中 |
| 实验脚本 | ~200 LOC | 低 |
| 分析脚本 | ~150 LOC | 低 |
| **总计** | ~680 LOC | - |

---

## 7. 预期贡献

完成此实验后，我们可以回答：

| 问题 | 回答来源 |
|------|----------|
| SD-MCTS 能达到多接近最优？ | SD-MCTS vs Oracle MPC |
| 学习比不学习好多少？ | (SD-MCTS, Adaptive) vs MPC (wrong) |
| 结构化学习比统计估计好多少？ | SD-MCTS vs Adaptive MPC |
| Value Net 带来多少视野优势？ | SD-MCTS vs Adaptive MPC (H=15) |
| 何时 SD-MCTS 优势最大？ | 延迟惩罚 vs 即时反馈场景对比 |

---

## 8. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| RLS 不收敛或不稳定 | 中 | 使用遗忘因子、参数约束 |
| Adaptive MPC 超越 SD-MCTS | 低 | 分析原因，调整 SD-MCTS 训练 |
| 计算时间过长 | 中 | 并行化，减少 seeds 初步验证 |
| 结果无统计显著性 | 低 | 增加 seeds，检查效应量 |

---

## 附录 A：参数速查表

### 效率退化场景

| 参数 | 符号 | 真实值 | 错误信念 |
|------|------|--------|----------|
| 效率衰减系数 | α | 0.10 | 0.001 |
| 衰减阈值 | θ | 5.0 | 12.0 |
| 到达率 | λ | 1.0 | - |
| Worker 处理率 | μ | (0.8, 0.7) | - |

### 延迟崩溃场景

| 参数 | 符号 | 真实值 | 错误信念 |
|------|------|--------|----------|
| 崩溃阈值 | F | 10.0 | 50.0 |
| 疲劳累积率 | α | 1.0 | 1.0 |
| 疲劳恢复率 | β | 0.1 | 0.1 |
| 峰值时刻 | T_s | 12 | - |
| 峰值奖励 | B | 15.0 | - |

---

## 附录 B：参考文献

1. Ljung, L. (1999). System Identification: Theory for the User.
2. Aström, K. J., & Wittenmark, B. (2013). Adaptive Control.
3. Rawlings, J. B., & Mayne, D. Q. (2009). Model Predictive Control: Theory and Design.
4. Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play (AlphaZero).
5. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).
