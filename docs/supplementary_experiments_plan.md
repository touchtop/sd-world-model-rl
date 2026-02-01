# SD-MCTS 定理验证补充实验方案

**文档版本**: v1.0
**创建日期**: 2026-01-23
**目标**: 为 Full Paper 标准设计针对定理2、定理3的机制级验证实验

---

## 一、背景与动机

### 1.1 当前实验覆盖状态

| 定理 | 核心主张 | 当前验证方式 | 不足之处 |
|------|----------|--------------|----------|
| **定理1** | 样本复杂度 $O(d/\epsilon^2)$ | 附录C: 与PPO对比 | ✅ 已有针对性实验 |
| **定理2** | 价值网络实现近似无限视界 | 场景二: 间接验证 | ⚠️ 缺少消融实验 |
| **定理3** | "不完美+深远" > "完美+短视" | 场景一: SD-MCTS > Oracle MPC | ⚠️ 缺少边界验证 |

### 1.2 Full Paper 标准要求

- **机制隔离**: 通过消融实验直接证明各组件的贡献
- **定量验证**: 实验数据应与理论公式预测吻合
- **边界条件**: 验证定理成立的条件边界

---

## 二、定理回顾

### 2.1 定理2：有效视界扩展 (Effective Horizon Extension)

**数学表述**:
$$
|V^*(s) - \hat{V}_{\text{MCTS}}(s)| \leq \underbrace{\gamma^H \epsilon_V}_{\text{价值网络误差}} + \underbrace{(L_r + \gamma^H L_V) \cdot \epsilon_{\text{model}} \cdot \Gamma_H}_{\text{模型误差累积}}
$$

其中 $\Gamma_H = \frac{L_f^H - 1}{L_f - 1}$（当 $L_f \leq 1$ 时为 $H$）。

**物理意义**:
- 价值网络误差项 $\gamma^H \epsilon_V$ 随搜索深度 $H$ 指数衰减
- MCTS 叶节点使用 $V_\psi(s)$ 而非零终端，可将"有效视界"从 $H$ 扩展至近似无穷

### 2.2 定理3：视界-误差权衡 (Horizon-Error Tradeoff)

**数学表述**:
SD-MCTS 优于 Oracle MPC 当且仅当：
$$
\underbrace{\frac{\gamma^H R_{\max}}{1-\gamma}}_{\text{MPC 截断偏差}} > \underbrace{L_\theta \delta_\theta + \gamma^H \epsilon_V}_{\text{SD-MCTS 总误差}}
$$

**物理意义**:
- 左侧: Oracle MPC 因视界截断损失的价值（随 $H$ 减小而增大）
- 右侧: SD-MCTS 因参数误差 $\delta_\theta$ 和价值网络误差 $\epsilon_V$ 损失的价值
- 在长时间常数系统中，截断偏差主导，故"不完美但深远"胜过"完美但短视"

---

## 三、补充实验设计

### 3.1 实验A：价值网络消融（定理2 核心验证）

#### 3.1.1 实验目标

直接隔离价值网络 $V_\psi(s)$ 对有效视界扩展的贡献，验证其将远期成本反向传播的能力。

#### 3.1.2 实验设计

| 要素 | 设计 |
|------|------|
| **环境** | 场景二（延迟崩溃 $t=10$，激增奖金 $t=12$） |
| **实验组** | **SD-MCTS**: MCTS + 学习世界模型 + $V_\psi(s)$ |
| **对照组A** | **MCTS-Zero**: MCTS + 学习世界模型 + $V_{terminal}=0$ |
| **对照组B** | **MCTS-Random**: MCTS + 学习世界模型 + 随机初始化VNet（未训练） |
| **自变量** | 搜索深度 $H \in \{2, 4, 6, 8, 10, 12, 14\}$ |
| **因变量** | 成功率（定义：避免崩溃 AND 捕获激增奖金） |
| **种子数** | 每组 30 seeds |

#### 3.1.3 预期结果

```
成功率
  ^
1.0|                    ████████████████  SD-MCTS (平坦高位)
   |                   ╱
0.8|                  ╱
   |                 ╱
0.6|                ╱
   |               ╱
0.4|              ╱
   |             ╱
0.2|────────────╱                        MCTS-Zero (阶跃函数)
   |
0.0+──────────────────────────────────→ H
     2   4   6   8  10  12  14
                    ↑
              崩溃时刻 T_collapse
```

- **MCTS-Zero**: 阶跃函数，$H < 12$ 时成功率 ~0%，$H \geq 12$ 时跳变至 ~100%
- **SD-MCTS**: 平坦曲线，即使 $H=4$ 也维持 >95% 成功率
- **MCTS-Random**: 介于两者之间，取决于随机初始化

#### 3.1.4 理论验证逻辑

若结果符合预期，则直接证明：
1. 价值网络成功将 $t=12$ 的崩溃成本反向传播至 $t=0$ 的状态价值
2. $V_\psi(s)$ 有效替代了 $\gamma^H R_{max}/(1-\gamma)$ 的截断损失
3. 定理2的"有效视界扩展"机制成立

#### 3.1.5 可视化

- **主图**: 折线图，X轴=搜索深度H，Y轴=成功率，三条线（SD-MCTS/MCTS-Zero/MCTS-Random）
- **辅图**: 典型轨迹对比（疲劳曲线 + 累积奖励）
- **标注**: 在 $H=T_{collapse}$ 处添加垂直虚线

---

### 3.2 实验B：视界-误差网格（定理2 定量验证）

#### 3.2.1 实验目标

定量验证误差累积公式，测量经验误差与理论预测的吻合程度。

#### 3.2.2 实验设计

| 要素 | 设计 |
|------|------|
| **环境** | 场景二（状态空间小，可用DP计算真实 $V^*$） |
| **自变量1** | 规划视界 $H \in \{1, 3, 5, 10, 15, 20\}$ |
| **自变量2** | 模型误差 $\epsilon_{model} \in \{0.0, 0.05, 0.1, 0.2\}$（通过扰动参数实现） |
| **因变量** | 经验价值误差 $\|V^* - \hat{V}_{MCTS}\|$ |
| **控制条件** | 价值网络固定（同一训练检查点）；环境参数固定 |
| **基准计算** | 动态规划求解真实 $V^*(s)$ |
| **种子数** | 每组 30 seeds |

#### 3.2.3 预期结果

误差曲线应呈 U 型：
- $H$ 小时：$\gamma^H \epsilon_V$ 项主导（价值网络误差未充分衰减）
- $H$ 大时：$\epsilon_{model} \cdot \Gamma_H$ 项主导（模型误差累积）
- 存在最优 $H^*$ 使总误差最小

#### 3.2.4 统计分析

1. **非线性回归**: 拟合 `error = a·γ^H + b·ε_model·Γ_H`
2. **报告指标**: $R^2$、参数估计值及 95% 置信区间
3. **理论对比**: 将拟合曲线与定理2公式预测叠加

#### 3.2.5 可视化

- **主图**: 3D 曲面图或等高线图（X=H, Y=ε_model, Z=error）
- **辅图**: 固定 ε_model，绘制 error vs H 曲线族

---

### 3.3 实验C：交叉点相位图（定理3 核心验证）

#### 3.3.1 实验目标

验证定理3预测的性能交叉边界，绘制 $(H, \delta_\theta)$ 空间中 SD-MCTS 优于/劣于 Oracle MPC 的区域。

#### 3.3.2 实验设计

| 要素 | 设计 |
|------|------|
| **环境** | 场景一（效率衰减） |
| **自变量1** | 规划视界 $H \in \{5, 10, 15, 20\}$ |
| **自变量2** | 参数误差 $\delta_\theta \in \{0\%, 10\%, 25\%, 50\%, 100\%\}$ |
| **实验方法** | SD-MCTS 使用 $\hat{\theta} = \theta^* \cdot (1 + \delta)$ |
| **基准** | Oracle MPC（真实 $\theta^*$，$H=15$） |
| **因变量** | 相对性能比 RPR |
| **种子数** | 每格 10 seeds（共 20 格） |

#### 3.3.3 性能指标定义

$$
\text{RPR}(H, \delta_\theta) = \frac{R_{\text{SD-MCTS}}(H, \delta_\theta) - R_{\text{Random}}}{R_{\text{Oracle}}(H=15) - R_{\text{Random}}}
$$

- RPR > 1: SD-MCTS 优于 Oracle MPC
- RPR < 1: SD-MCTS 劣于 Oracle MPC
- RPR = 1: 交叉点

#### 3.3.4 预期结果

热力图中应存在对角线边界：

```
δ_θ (参数误差)
  ^
100%|  ██  ██  ░░  ░░     ██ = 红区 (SD-MCTS 劣)
 50%|  ██  ░░  ░░  ░░     ░░ = 蓝区 (SD-MCTS 优)
 25%|  ░░  ░░  ░░  ░░
 10%|  ░░  ░░  ░░  ░░
  0%|  ░░  ░░  ░░  ░░
    +─────────────────→ H (视界)
       5   10  15  20
```

- **蓝区（右下）**: 高 H + 低 δ_θ → SD-MCTS 胜（验证定理3）
- **红区（左上）**: 低 H + 高 δ_θ → 误差过大，Oracle MPC 胜
- **边界线**: 应与理论预测 $\gamma^H R_{max}/(1-\gamma) = L_\theta \delta_\theta + \gamma^H \epsilon_V$ 吻合

#### 3.3.5 统计分析

1. **逻辑回归**: 拟合交叉边界 $P(\text{SD-MCTS wins}) = \sigma(w_1 H + w_2 \delta_\theta + b)$
2. **理论边界叠加**: 将定理3不等式的等号边界绘制在热力图上
3. **边界偏差分析**: 计算经验边界与理论边界的偏离程度

#### 3.3.6 可视化

- **主图**: 2D 热力图，使用 diverging colormap (红-白-蓝)
- **叠加**: 理论边界线（虚线）
- **标注**: RPR=1 等值线（实线）

---

### 3.4 实验D：架构贡献分解矩阵

#### 3.4.1 实验目标

分解"结构化世界模型 (SD)"与"价值网络 (VNet)"的各自贡献，验证两者的互补性。

#### 3.4.2 实验设计

| 配置名称 | 世界模型 | 终端价值 | 验证目标 |
|----------|----------|----------|----------|
| **Oracle MPC** | 真实参数 $\theta^*$ | $V=0$ | 纯规划上界（完美模型+有限视界） |
| **Learned MPC** | 学习的 SD 模型 | $V=0$ | 隔离系统辨识质量 |
| **MuZero-Style** | 黑盒 MLP | 学习的 $V_\psi$ | 隔离结构化知识价值 |
| **SD-MCTS** | 学习的 SD 模型 | 学习的 $V_\psi$ | 完整方法（结构+价值） |

#### 3.4.3 预期结果与假设验证

| 比较 | 预期结果 | 验证假设 |
|------|----------|----------|
| SD-MCTS > MuZero-Style | 显著 | 结构化知识（SD方程）提升样本效率和泛化 |
| SD-MCTS > Learned MPC | 显著 | 价值网络扩展有效规划视界 |
| SD-MCTS > Oracle MPC | 显著 | 深度规划补偿参数误差（定理3核心） |
| Learned MPC > MuZero-Style | 中等 | 结构化模型即使无VNet也优于黑盒 |

#### 3.4.4 实验参数

| 要素 | 设计 |
|------|------|
| **环境** | 场景一（效率衰减） |
| **训练步数** | 30,000 步（SD方法）；300,000 步（MuZero-Style，验证样本效率差异） |
| **种子数** | 50 seeds |
| **评估** | 训练后评估 100 episodes |

#### 3.4.5 可视化

- **主图**: 分组柱状图 + 95% 置信区间
- **辅图**: 学习曲线对比（reward vs training steps）
- **表格**: 成对 t 检验结果 + Cohen's d

---

## 四、实现规范

### 4.1 代码复用分析

| 模块 | 路径 | 复用状态 |
|------|------|----------|
| MCTS 核心 | `src/planning/mcts.py` | ✅ 已支持 `value_net=None` |
| 世界模型 | `src/world_model/delayed_collapse_model.py` | ✅ 可直接使用 |
| Oracle MPC | `experiments/baselines/oracle_mpc.py` | ✅ 已支持 `horizon` 参数 |
| 消融框架 | `experiments/run_ablation.py` | ✅ 已有 `FrozenSDModel` 等变体 |
| 实验配置 | `experiments/m1_framework.py` | ✅ `ScenarioConfig` 可扩展 |

### 4.2 需新增/修改的代码

#### 4.2.1 MCTS 修改 (`src/planning/mcts.py`)

```python
# 新增参数
class MCTS:
    def __init__(self, ..., max_depth: int = None, use_value_net: bool = True):
        self.max_depth = max_depth  # 限制搜索深度
        self.use_value_net = use_value_net  # 控制是否使用价值网络

    def _evaluate_leaf(self, state):
        if not self.use_value_net or self.value_net is None:
            return 0.0  # 零终端价值
        return self.value_net(state)
```

#### 4.2.2 新增实验脚本

| 脚本 | 对应实验 | 主要功能 |
|------|----------|----------|
| `experiments/run_vnet_ablation.py` | 实验A | 价值网络消融，扫描搜索深度 |
| `experiments/run_horizon_error_grid.py` | 实验B | 视界-误差网格实验 |
| `experiments/run_crossover_grid.py` | 实验C | 交叉点相位图实验 |
| `experiments/run_component_ablation.py` | 实验D | 架构贡献分解 |

### 4.3 配置文件模板

```yaml
# configs/theorem_validation.yaml
experiment_a:
  name: "vnet_ablation"
  env: "delayed_collapse"
  variants:
    - name: "sd_mcts"
      use_value_net: true
    - name: "mcts_zero"
      use_value_net: false
    - name: "mcts_random"
      use_value_net: true
      pretrained: false
  search_depths: [2, 4, 6, 8, 10, 12, 14]
  seeds: 30

experiment_c:
  name: "crossover_grid"
  env: "efficiency_decay"
  horizons: [5, 10, 15, 20]
  param_errors: [0.0, 0.1, 0.25, 0.5, 1.0]
  seeds_per_cell: 10
```

---

## 五、统计规范

### 5.1 显著性检验

| 场景 | 检验方法 | 阈值 |
|------|----------|------|
| 两组对比 | Welch's t-test（正态）或 Mann-Whitney U（非正态） | $p < 0.05$ (*), $p < 0.01$ (**) |
| 多组对比 | One-way ANOVA + Tukey HSD | $\alpha = 0.05$ |
| 效应量 | Cohen's d | $d > 0.8$ (大效应) |

### 5.2 误差报告

| 图表类型 | 误差表示 |
|----------|----------|
| 折线图 | 阴影区域 = 95% CI (bootstrapped) |
| 柱状图 | 误差棒 = SEM |
| 热力图 | 每格显示 mean ± std |

### 5.3 可复现性

- 所有实验固定随机种子列表：`seeds = list(range(42, 42 + n_seeds))`
- 记录完整环境配置（Python版本、PyTorch版本、CUDA版本）
- 保存所有检查点和原始数据

---

## 六、论文整合建议

### 6.1 章节结构

在 **Section 5 (Results & Analysis)** 中新增：

```
5.1 Core Performance Comparison       [现有]
5.2 Strategy Emergence                 [现有]
5.3 Mechanism Analysis & Ablation      [新增]
    5.3.1 Effective Horizon Isolation  → 实验A
    5.3.2 Horizon-Error Quantification → 实验B
    5.3.3 Phase Transition Boundary    → 实验C
    5.3.4 Component Contribution       → 实验D
5.4 Theoretical Verification           [现有，更名]
```

### 6.2 图表编号建议

| 图号 | 内容 | 对应实验 |
|------|------|----------|
| Figure 7 | 价值网络消融：成功率 vs 搜索深度 | 实验A |
| Figure 8 | 视界-误差网格：3D 曲面图 | 实验B |
| Figure 9 | 交叉点相位图：热力图 + 理论边界 | 实验C |
| Figure 10 | 架构贡献分解：柱状图 | 实验D |

### 6.3 图表标题规范

**不佳示例**:
> "Figure 7: Success rate vs search depth."

**推荐示例**:
> "Figure 7: **Empirical Validation of Effective Horizon Extension (Theorem 2).** While MCTS-Zero (gray) exhibits step-function failure for search depths $H < T_{collapse}=12$, SD-MCTS (blue) maintains >95% success rate even at $H=4$, confirming the value network successfully propagates future collapse costs to early states as predicted by Eq. (14)."

---

## 七、时间与资源估算

### 7.1 计算资源

| 实验 | GPU 类型 | 预计时间 | 并行度 |
|------|----------|----------|--------|
| 实验A | RTX 3090 | 4 小时 | 7 × 3 × 30 = 630 runs |
| 实验B | RTX 3090 | 12 小时 | 6 × 4 × 30 = 720 runs |
| 实验C | RTX 3090 | 20 小时 | 4 × 5 × 10 = 200 runs |
| 实验D | RTX 3090 | 8 小时 | 4 × 50 = 200 runs |
| **总计** | - | **~44 小时** | - |

### 7.2 优先级排序

| 优先级 | 实验 | 理由 |
|--------|------|------|
| **P0** | 实验A | 定理2 的直接、可视化验证；实现简单 |
| **P0** | 实验C | 定理3 的完整边界验证；核心创新点 |
| **P1** | 实验D | 补充理解各组件贡献；审稿人常问 |
| **P2** | 实验B | 定理2 的定量验证；可选（若时间紧张可省略） |

---

## 八、风险与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| MCTS-Zero 在 $H \geq 12$ 时仍失败 | 低 | 高 | 检查崩溃机制实现；增加搜索预算 |
| 交叉边界与理论预测偏差大 | 中 | 中 | 分析 $L_\theta$ 估计误差；调整理论常数 |
| MuZero-Style 训练不收敛 | 中 | 低 | 增加训练步数；调整网络架构 |
| 计算资源不足 | 低 | 中 | 优先运行 P0 实验；减少种子数 |

---

## 附录：实验-定理映射总表

| 定理 | 核心声明 | 验证实验 | 关键证据 |
|------|----------|----------|----------|
| **定理2** | 价值网络实现近似无限视界 | 实验A (消融) | 平坦曲线 vs 阶跃函数 |
| **定理2** | 误差累积公式 | 实验B (网格) | 曲线形态与公式吻合 |
| **定理3** | 存在性能交叉点 | 实验C (相位图) | 对角线边界 |
| **定理3** | "不完美+深远" > "完美+短视" | 实验D (分解) | SD-MCTS > Oracle MPC |
| (两者) | 结构化知识价值 | 实验D (分解) | SD-MCTS >> MuZero-Style |

---

*本文档由 Claude (Codex + Gemini 协作分析) 生成*
