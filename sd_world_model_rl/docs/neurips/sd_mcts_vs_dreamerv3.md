# SD-MCTS vs DreamerV3: 基于 DMControl Cartpole-Swingup 的系统对比

> 基准代码: `scripts/run_dmcontrol_cartpole.py` + `src/world_model/dmcontrol_cartpole_sd_model.py`
>
> 对照论文: Hafner et al., "Mastering Diverse Domains through World Models", 2023 (arXiv:2301.04104)

---

## 1 实验设置对齐

| 维度 | DreamerV3 | SD-MCTS (Ours) | 是否一致 |
|------|-----------|----------------|---------|
| 环境 | `dm_control.suite.cartpole.swingup` | 同 | ✓ |
| 观测空间 | 本体感知 5 维 `[x, cos θ, sin θ, ẋ, θ̇]` | 同（经 `obs_to_state` 转换为 `[x, θ, ẋ, θ̇]`） | ✓ |
| 动作空间 | 连续 `[-1, 1]` | 离散 7 bins `linspace(-1, 1, 7)` | ✗ |
| Action repeat | k = 2 | k = 2 | ✓ |
| Episode 长度 | 1000 env steps (500 决策步) | 同 | ✓ |
| 物理引擎 | MuJoCo (RK4, dt=0.01, gear=10) | 同配置 | ✓ |
| Reward 函数 | `upright × centered × small_control × small_velocity` | 同公式，匹配 `tolerance` 的精确 σ | ✓ |
| 随机种子 | 多种子评估 | Seeds = {0, 42, 84} | ✓ |

---

## 2 世界模型对比

### 2.1 架构

| | DreamerV3 RSSM | SD-MCTS 结构动力学模型 |
|---|---|---|
| **建模范式** | 黑盒隐空间动态 | 白盒牛顿力学方程 |
| **状态空间** | 隐空间 `h_t` ∈ ℝ^512 (确定性) + `z_t` ∈ {32×32} (离散随机) | 物理空间 `[x, θ, ẋ, θ̇]` ∈ ℝ^4 |
| **转移模型** | GRU + MLP (posterior / prior) | Barto (1983) 运动方程 + RK4 积分器 |
| **已知结构** | 无 — 从数据学习全部动态 | 显式编码: 牛顿第二定律、杆铰链耦合、RK4 数值格式 |
| **未知成分** | 全部 (~8M 参数†) | 仅 3 个标量: `masscart`, `masspole`, `length` |
| **辅助头** | 解码器 + reward head + continue head | 解析 reward 函数 (无需学习) |
| **积分器** | 隐式 (GRU 单步) | 显式 RK4 (匹配 MuJoCo `model.opt.integrator = 1`) |

### 2.2 参数量

```
SD-MCTS 世界模型:        3 个可学习参数
DreamerV3 世界模型:  ~8M 个可学习参数 (Size S 估算†)

倍数差异: ~2.7M×
```

完整系统（含规划组件）:

```
SD-MCTS (WM + VNet):             17,284 参数
DreamerV3 (WM + Actor + Critic): ~10M 参数 (Size S 估算†)

倍数差异: ~580×
```

> **†参数量说明**: DreamerV3 论文未公布逐模块参数量表。此处基于论文 Table B.1
> 公开的架构超参 (Size S: GRU 512, MLP 512×2, discrete 32×32 等) 进行层级估算，
> 属于数量级参考而非精确值。SD-MCTS 的 17,284 参数可从代码精确计算
> (世界模型 3 + VNet 17,281)。

### 2.3 学习方式

| | DreamerV3 | SD-MCTS |
|---|---|---|
| **训练信号** | 观测重建 + reward 预测 + continue 预测 (联合 ELBO) | 状态预测 MSE / sincos loss (纯系统辨识) |
| **训练数据量** | 100k–1M env steps | 2,500 env steps (随机探索) |
| **训练方式** | 在线持续学习 (replay buffer + 梯度步) | 离线一次性辨识 (500–800 epochs SGD) |
| **多步一致性** | 隐式 (RSSM 自回归) | 显式 multi-step loss with curriculum (1→T steps) |
| **收敛指标** | 多头损失收敛 | SI loss ≈ 1e-6 (近似完美匹配 MuJoCo) |

### 2.4 可解释性

DreamerV3 的隐空间不可直接解读。SD-MCTS 的学习结果可直接验证:

```
参数         真实值     学习值     误差
gravity      9.810      9.810     0.0%  (固定)
masscart     1.000      1.016     1.6%
masspole     0.100      0.114    14.2%
length       0.500      0.513     2.7%
```

---

## 3 规划算法对比

### 3.1 架构差异

| | DreamerV3 | SD-MCTS |
|---|---|---|
| **规划方式** | **Actor-Critic in Imagination** | **MCTS 在线树搜索** |
| **策略表示** | 神经网络 Actor π_φ(a\|s)，连续 Gaussian 输出 | 无显式策略网络；MCTS 搜索结果即策略 |
| **动作选择** | 单次前向推理 (摊销规划) | 每步 200 次 MCTS simulation |
| **价值函数** | Critic V_ψ, TD(λ) + symlog 变换 | VNet, 2 轮 imagination distillation |
| **动作空间** | 连续 (SampleDist) | 离散 7 bins |
| **在线代价** | O(1) 前向推理 | O(sims × depth × k) 模型调用 |

### 3.2 Imagination 训练

DreamerV3 在 imagination 中交替优化 Actor 和 Critic (紧密迭代循环):

```
for each gradient step:
    1. 从 replay buffer 采样真实状态 s_0
    2. 用 Actor 在 RSSM 中 rollout H=15 步 imagination trajectory
    3. 计算 TD(λ) returns (λ=0.95, symlog 变换)
    4. 更新 Critic: 最小化 value 预测误差
    5. 更新 Actor: 最大化 imagined returns (REINFORCE + entropy)
```

SD-MCTS 的 VNet 训练分为两轮:

```
Round 1: 启发式 imagination
    1. 从真实数据 + 扰动采样初始状态
    2. 用能量泵送启发式 (Åström controller) rollout 60 步
    3. 计算 MC returns, 训练 VNet (mini-batch, smooth L1)

Round 2: MCTS bootstrap
    1. 从真实数据 (含近 upright 分层) 采样初始状态
    2. 用 VNet-guided MCTS rollout 30 步
    3. MC returns → 合并 Round 1 数据重训 VNet
```

### 3.3 关键差异的影响

| 差异 | 对 SD-MCTS 的影响 |
|------|-----------------|
| 离散 vs 连续动作 | 7 bins 分辨率 ≈ 0.286，精细平衡受限；DreamerV3 可输出任意精度 |
| 搜索 vs 摊销 | MCTS 推理慢 (每步 ~0.5s) 但无需 Actor 训练；DreamerV3 推理快 (<1ms) 但需持续训练 |
| 2 轮 bootstrap vs 紧密迭代 | VNet 质量弱于 DreamerV3 Critic；但 MCTS 搜索本身可补偿 VNet 不足 |
| 无 Actor 网络 | 无需在 imagination 中学习策略分布，避免策略梯度方差问题 |

---

## 4 Sample Efficiency 对比

### 4.1 数据使用

| | DreamerV3 | SD-MCTS |
|---|---|---|
| 世界模型训练 | 100k–1M env steps (持续在线学习) | **2,500 env steps** (一次性随机采集) |
| Actor-Critic 训练 | 与世界模型共享数据 (replay buffer) | VNet 纯 imagination 数据 (零额外 env 交互) |
| **总 env 交互量** | **100k–1M** | **2,500** (在 `--distill_rollout` 模式下，budget 5000 中半数用于 SI) |
| **Env 交互倍率** | 基准 | **40–400× 更少** |

### 4.2 性能对比

| 方法 | Env Steps | Return | 来源 |
|------|-----------|--------|------|
| **SD-MCTS (Ours)** | **5,000** | **335 ± 39** (v3 基线) | 实验结果 |
| DreamerV3 | 5,000 | ≈ 20 (接近 random) | Figure 4 学习曲线推读 |
| DreamerV3 | 100,000 | ≈ 700–750 | Figure 4 学习曲线推读 |
| DreamerV3 (proprio) | 500,000 | 850.0 | Table O.1 (主参考) |
| Random Policy | — | 17 ± 10 | 实验结果 |

> **版本说明**: 本文统一采用 DreamerV3 proprio (本体感知) 500K 步结果 850.0 作为
> 渐近性能参考。该设定与 SD-MCTS 的观测模态一致 (均为本体感知，无视觉输入)。
> DreamerV3 visual 1M 结果 (819.1, Table O.2) 因观测模态不同，不纳入直接对比。

### 4.3 为什么 SD-MCTS 样本效率更高

核心原因在于 **假设空间的结构性压缩**:

- DreamerV3 的假设空间是 **任意连续函数** 的 ~8M 维参数化近似。从 ODE 角度看，RSSM 需在无先验约束下学会满足牛顿第二定律的动态转移，这是一个对样本量有指数级依赖的非参数学习问题。
- SD-MCTS 将假设空间 **从任意函数压缩为已知 ODE 族的参数流形**。Barto (1983) 方程 + RK4 积分器将结构完全固定，待辨识量仅为 ODE 系数 {masscart, masspole, length}。
- 这一压缩使学习问题从 **函数逼近** (需要百万级样本覆盖状态-动作空间) 退化为 **参数估计** (仅需几千个观测点即可收敛到满意精度)。

信息论视角: 设真实系统的 Kolmogorov 复杂度为 K，SD 模型的描述长度为 K_SD ≈ 3×32 bits (3 个 float32 参数)。DreamerV3 的 RSSM 需 K_RSSM ≈ 8M×32 bits 来表达同一动态。根据 MDL (Minimum Description Length) 原则，更短的描述长度对应更低的样本复杂度需求。这从信息论角度解释了两种方法在样本效率上的阶差异。

因此，SD-MCTS 在 5k 步时世界模型已收敛 (SI loss ≈ 1e-6)，而 DreamerV3 在同样数据量下尚处于接近随机的水平——这不是偶然的性能差异，而是假设空间维度差异的必然结果。

---

## 5 Reward 模型对比

| | DreamerV3 | SD-MCTS |
|---|---|---|
| **Reward 来源** | Reward head (MLP, 从数据学习) | 解析函数 (硬编码 DMControl 公式) |
| **Reward head 参数** | ≈ 1M | 0 |
| **精度** | 受限于网络拟合质量和数据量 | 精确匹配 (tolerance sigma 已校正) |
| **是否需要学习** | 是 (需足够 reward 信号) | 否 (直接从物理状态计算) |

SD-MCTS 在 MCTS 规划中使用解析 reward 函数:

```python
reward = upright × centered × small_control × small_velocity
```

其中每个分量精确匹配 DMControl 的 `rewards.tolerance` 函数 (Gaussian sigmoid, `value_at_margin=0.1`):
- `centered`: σ = 2/√(−2 ln 0.1) ≈ 0.932
- `small_velocity`: σ = 5/√(−2 ln 0.1) ≈ 2.330
- `small_control`: quadratic sigmoid, margin=1

---

## 6 不同层面的对比定性总结

### 6.1 SD-MCTS 的结构性优势

1. **极端样本效率**: 3 维系统辨识 vs ~8M 端到端学习 → 40–400× 数据节省
2. **可解释性**: 学到的参数有物理含义，可直接验证和 debug
3. **零 reward 学习**: 解析 reward 函数无需从数据拟合
4. **结构保证**: RK4 积分器 + 守恒律 → 多步预测不发散
5. **可编辑性**: 物理参数变化时，直接修改对应标量即可（零样本迁移）

### 6.2 SD-MCTS 的结构性劣势

1. **需要领域知识**: 必须预先知道运动方程的解析形式；DreamerV3 无需任何先验
2. **离散动作空间**: MCTS 需要离散化，损失精细控制能力
3. **在线推理慢**: 每步 200 次 MCTS simulation vs DreamerV3 单次前向推理
4. **VNet 弱于 DreamerV3 Critic**: 2 轮 bootstrap vs 数千步紧密迭代
5. **泛化受限**: 方程形式变化时需要重写模型；DreamerV3 自动适应

### 6.3 公平性声明

本对比中，SD-MCTS 与 DreamerV3 的差异 **不仅限于世界模型**。以下混杂因素需在论文中明确声明:

1. **规划算法不同**: MCTS 在线树搜索 vs Actor-Critic 摊销规划。MCTS 每步消耗 200 次模拟换取更优决策，DreamerV3 依赖离线训练的 Actor 做 O(1) 推理。
2. **动作空间不同**: SD-MCTS 使用 7-bin 离散化 (分辨率 0.286)，DreamerV3 使用连续 Gaussian 输出。离散化限制了精细控制精度，但也简化了搜索空间。
3. **价值函数训练范式不同**: SD-MCTS 的 VNet 仅经 2 轮 imagination distillation (启发式 + MCTS bootstrap)，DreamerV3 的 Critic 经数千步与 Actor 紧密联合迭代优化 (TD(λ) + symlog)。
4. **Reward 信号来源不同**: SD-MCTS 使用硬编码的解析 reward 函数 (零噪声)，DreamerV3 从数据中学习 reward head (引入拟合误差)。
5. **表征学习缺失**: SD-MCTS 直接在物理状态空间操作，无需学习表征；DreamerV3 需同时学习状态表征和动态模型。

因此，性能差异不能完全归因于世界模型选择。但核心 claim 成立: **在结构已知的领域，用方程先验替代数百万参数的端到端学习，可将 sample complexity 降低 1–2 个数量级。** 上述混杂因素中，(4) 和 (5) 同样源于结构先验带来的简化，而 (1)–(3) 是独立的工程选择。

---

## 7 论文叙事建议

### 推荐定位

> *"当系统动力学的数学形式已知时，将其作为归纳偏置注入世界模型，可将可学习参数从数百万压缩至个位数，实现极端样本效率。在 DMControl cartpole-swingup 基准上，SD-MCTS 仅用 5,000 步环境交互即可支撑有效在线规划，而同样数据量下的 DreamerV3 尚处于随机策略水平。"*

### 不应过度声称

- ✗ "SD-MCTS 在所有指标上优于 DreamerV3" — 绝对性能仍有差距 (335 vs 850)
- ✗ "在完全相同的算法框架下对比" — 规划算法不同 (MCTS vs Actor-Critic)
- ✗ "通用的 MBRL 改进" — 仅适用于结构已知领域

### 应当强调

- ✓ **参数量对比** (3 vs ~8M) 作为核心证据
- ✓ **Sample efficiency 的阶差异** (5k vs 100k) 而非绝对分数
- ✓ **可解释性** — 学到的参数可直接与物理真值对比
- ✓ 这是 **结构先验价值的量化证据**，不是要取代 DreamerV3

---

## 附录 A: DreamerV3 参考数据来源

| 数据 | 来源 |
|------|------|
| Proprio 500K: 850.0 | DreamerV3 论文 Table O.1 (主参考) |
| Action repeat = 2 | 论文正文: "Control spans 20 tasks with an action repeat of 2" |
| 5k 步 ≈ random | 学习曲线推读 + 社区报告 (无精确表格值) |
| 100k 步 ≈ 700–750 | 学习曲线推读 (无精确表格值) |
| Visual 1M: 819.1 | DreamerV3 论文 Table O.2 (不同观测模态，仅供参考) |

## 附录 B: 代码文件索引

| 文件 | 内容 |
|------|------|
| `src/world_model/dmcontrol_cartpole_sd_model.py` | SD 世界模型 (RK4 + Barto 方程 + 精确 reward) |
| `scripts/run_dmcontrol_cartpole.py` | 实验主脚本 (SI + VNet distillation + MCTS + 评估) |
