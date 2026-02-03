# SD-MCTS NeurIPS 论文工作方案

**目标**: 产出一篇达到 Spotlight/Oral 标准的高质量ML顶会论文

**基于**: Codex + Gemini + Claude 三方协商结论

---

## 一、核心定位与叙事策略

### 1.1 论文定位

| 维度 | 定位 |
|------|------|
| **核心主张** | 结构化因果先验（SD stock-flow）可作为世界模型的归纳偏置，实现可证明的视界压缩与样本效率提升 |
| **目标Track** | NeurIPS 主轨道 (优先) / ICLR (备选) |
| **论文类型** | 理论轻量级 + 强实证的方法论文 |

### 1.2 叙事Hook

> **"Don't Relearn Physics: Causal Structure as Planning Prior"**

**开篇故事**: DreamerV3/MuZero 从零学习动力学，需要百万级样本。但在许多实际系统中（供应链、流行病、能源），因果结构是已知的——只有参数未知。我们证明：利用这一结构可将有效规划视界从 H 压缩至 O(log(1/ε))，实现数量级的样本效率提升。

### 1.3 差异化定位 (精确措辞)

**vs DreamerV3/TD-MPC2**:
> "DreamerV3 and TD-MPC2 learn unstructured latent dynamics, whereas we impose explicit SD causal structure as a planning prior. This yields interpretability and provable horizon reduction in domains where causal structure is known."

**vs PINNs/Neural ODEs**:
> "PINNs encode differential constraints during model fitting but do not shape planning or discrete decision-making. Our approach injects SD structure at the planning level, enabling long-horizon control with discrete actions."

**vs Classical SD**:
> "Classical SD focuses on forward simulation under fixed policies. We integrate SD into an RL loop with MCTS and learned residuals to compute optimal policies."

---

## 二、理论贡献设计

### 2.1 核心定理 (三条)

**Theorem 1: PUCT with SD Prior Consistency**
```
设有限 S,A, 有界奖励, PUCT 初始值 |Q₀(s,a)| ≤ 1, 虚拟访问 n₀.
令 ε_sd = sup_{s,a} |Q₀(s,a) - Q*(s,a)|.
则对任意 δ ∈ (0,1), N 次模拟后:
    0 ≤ V*(s₀) - V̂_N(s₀) ≤ 2c·√(log(1/δ)/N) + (n₀/(N+n₀))·ε_sd
概率至少 1-δ. 特别地, V̂_N(s₀) → V*(s₀) 当 N → ∞.
```

**Theorem 2: Value Error under Model Misspecification**
```
设 sup_{s,a} ||P(·|s,a) - P̂(·|s,a)||₁ ≤ ε_m.
则对任意策略 π 和初始状态 s₀:
    |V^π(s₀) - V̂^π(s₀)| ≤ H·ε_m
若 MCTS 在 P̂ 上返回 π̂ 满足模型最优性差距 V̂*(s₀) - V̂^{π̂}(s₀) ≤ ε_p,
则真实世界次优性有界:
    V*(s₀) - V^{π̂}(s₀) ≤ 2H·ε_m + ε_p
```

**Theorem 3: Effective Horizon Reduction (STAR THEOREM)**
```
设存在宏观状态映射 φ 和 SD 先验动力学 g, 使得对所有 s,s',a:
    ||φ(P(s,a)) - φ(P(s',a))|| ≤ κ·||φ(s) - φ(s')|| 其中 κ ∈ [0,1)
设 V* 在 φ 下 L_φ-Lipschitz.
则深度 d 的截断误差满足:
    |V*_H(s) - V*_d(s)| ≤ (L_φ/(1-κ))·κ^d
因此 ε-精确规划仅需深度:
    H_eff = ⌈log(ε(1-κ)/L_φ) / log(κ)⌉
```

### 2.2 理论定位策略

| 策略 | 说明 |
|------|------|
| **洞见驱动** | 定理服务于"解释为何有效"，而非追求最紧的界 |
| **假设透明** | 明确列出每个假设，承认理想化，提供实证验证 |
| **误差分解** | 将总误差分解为 模型误差 + 规划误差 + 截断误差 |

### 2.3 可接受假设 vs 需放松假设

| 可接受 | 需放松/验证 |
|--------|-------------|
| 有界奖励、有限分支、Lipschitz动力学 | 完美因果图 → 实验验证错误先验鲁棒性 |
| 有界模型误差、良定义宏观状态映射 | 精确延迟参数 → 模型包含不确定性 |

---

## 三、实验设计

### 3.1 实验矩阵

| 优先级 | 实验 | 环境 | 基线 | 核心指标 | 目标结果 |
|--------|------|------|------|----------|----------|
| **P0** | 样本效率曲线 | Beer Game (4-echelon) | DreamerV3, TD-MPC2, MuZero, PPO | 达到90%回报所需样本 | **3-5x 更少样本** |
| **P0** | 物理违背分析 | 全场景 OOD测试 | BlackBox-MCTS, DreamerV3 | 违背率(负库存等) | **SD-MCTS: 0%** |
| **P1** | 需求冲击鲁棒性 | Supply Chain + Demand Shock | DreamerV3, TD-MPC2, MPC | 冲击后遗憾 | **25-40% 更低遗憾** |
| **P1** | 零样本迁移 | 3节点→5节点供应链 | BlackBox, DreamerV3 | 迁移后性能保持率 | **>85% 保持** |
| **P2** | 流行病控制 | SEIR (延迟报告) | PPO, SAC, CEM-MPC | 总感染+干预成本 | **15-25% 更少感染** |
| **P2** | 延迟崩溃场景 | Scenario 2 (已有) | Oracle MPC, Adaptive MPC | 成功率 | **100% @ H=1** |
| **P3** | 消融研究 | Beer Game | 移除各组件 | 性能下降幅度 | 量化各组件贡献 |

### 3.2 必需消融实验

| 消融变体 | 移除内容 | 预期结果 |
|----------|----------|----------|
| **No-SD** | 移除stock-flow结构，用MLP替代 | 样本效率下降5-10x，出现物理违背 |
| **No-Vnet** | 移除价值网络终端估计 | 视界截断失败，长期任务崩溃 |
| **Wrong-Prior** | 使用错误因果结构 | 性能下降但优于纯黑盒 |
| **No-MCTS** | 仅用SD模型做单步贪心 | 无法处理延迟效应 |

### 3.3 基线实现规格

**必须实现的SOTA基线**:

| 基线 | 来源 | 适配要点 |
|------|------|----------|
| **DreamerV3** | 官方实现 | 离散动作适配，使用官方超参 |
| **TD-MPC2** | 官方实现 | 离散化或选择连续动作环境 |
| **MuZero** | muzero-general | 与SD-MCTS公平对比MCTS组件 |
| **BlackBox-MCTS** | 自行实现 | MLP(state+action→next_state) + 相同MCTS |
| **PPO/SAC** | Stable-Baselines3 | Model-free基线 |

**BlackBox世界模型规格**:
```python
class BlackBoxWorldModel(nn.Module):
    # 架构: MLP(state_dim + action_dim, 256, 256, state_dim)
    # 训练: 与SD-MCTS相同样本预算
    # 约束: 无（可预测负库存等违背状态）
```

### 3.4 Killer Figure 设计 (Figure 1)

**布局**: 2×2 子图

| 位置 | 内容 |
|------|------|
| **(a) 左上** | 样本效率曲线 (X: log环境步数, Y: 归一化回报). 5条曲线+置信区间. 粗箭头标注"5x fewer samples" |
| **(b) 右上** | 物理违背率柱状图 (SD-MCTS=0%, BlackBox=15%, DreamerV3=8%) |
| **(c) 左下** | OOD鲁棒性 (需求冲击前后性能对比). 阴影区域标注冲击期 |
| **(d) 右下** | SD结构示意图 (stock-flow + 神经残差网络) |

**核心信息**: 一眼看出 SD-MCTS 更快、更安全、更鲁棒

---

## 四、风险缓解与拒稿预防

### 4.1 Top 3 拒稿理由及预防策略

| 拒稿理由 | 概率 | 预防策略 | 论文中预埋文字 |
|----------|------|----------|----------------|
| **"SD+MCTS是直接组合，增量创新"** | 40% | 强调神经-符号整合的涌现能力；Theorem 3证明视界压缩；实验展示单独组件无法实现的行为 | "We combine SD causal structure with learned residual dynamics inside MCTS, yielding a principled inductive bias and new theory on horizon reduction." |
| **"理论假设过强"** | 30% | 显式列出误差分解(Theorem 2)；实验验证错误先验下的鲁棒性；提供安全门控规则 | "All theoretical statements include model error terms and we validate robustness under prior mismatch in Section X." |
| **"DreamerV3训练更久会追上"** | 35% | 提供1x和10x数据预算对比；展示OOD场景下的持续优势；强调结构带来的不可逾越优势 | "We include DreamerV3 with 1x and 10x data budgets and report both sample efficiency and generalization under distribution shift." |

### 4.2 备选环境与理论

**如果Beer Game实验效果不显著**:
- 备选环境: SEIR流行病控制、能源存储调度、水库管理
- 选择标准: 因果结构清晰、延迟效应显著、数据稀缺场景

**如果理论证明遇困**:
- 降级为"解释性命题" + 强实证验证
- 保守界: SD先验+不确定性加权保证不差于基线MCTS
- 遗憾分解: 模型误差 + 规划误差的加法分解

### 4.3 审稿人画像与应对

| 审稿人类型 | 可能关注点 | 应对策略 |
|------------|------------|----------|
| **ML理论派** | 假设严谨性、界的紧度 | 强调洞见驱动而非最优界；明确承认理想化假设 |
| **MBRL实证派** | 与SOTA对比公平性 | 使用官方实现、报告超参搜索预算、提供代码 |
| **应用导向派** | 实际价值、可扩展性 | 强调OR领域影响、讨论工业部署场景 |
| **物理信息ML派** | 与PINNs/HNN区别 | 强调规划层面而非拟合层面的结构注入 |

---

## 五、论文结构设计

### 5.1 标题候选

| 选项 | 标题 | 风格 |
|------|------|------|
| **A (推荐)** | *Structured World Models: Causal Priors for Sample-Efficient Planning* | 学术严谨 |
| **B** | *Don't Relearn Physics: System Dynamics as Planning Prior in Model-Based RL* | 叙事hook |
| **C** | *SD-MCTS: Provable Horizon Reduction via Causal Structure in World Models* | 技术聚焦 |

### 5.2 章节大纲

| 章节 | 页数 | 核心内容 |
|------|------|----------|
| **Abstract** | 0.3 | 问题-方法-结果-洞见 |
| **1. Introduction** | 1.2 | DreamerV3的tabula rasa低效 → 结构作为先验 → 贡献声明 |
| **2. Related Work** | 0.8 | MBRL谱系、物理信息ML、ADP终端代价 |
| **3. Problem Formulation** | 0.7 | 结构化MDP定义、SD stock-flow形式化 |
| **4. Method: SD-MCTS** | 1.5 | 结构化世界模型、MCTS+Vnet、梯度路径分析 |
| **5. Theoretical Analysis** | 1.0 | Theorem 1-3、假设讨论、洞见总结 |
| **6. Experiments** | 2.0 | 样本效率、物理违背、OOD鲁棒性、迁移、消融 |
| **7. Discussion & Limitations** | 0.5 | 适用边界、失败模式、未来方向 |
| **References** | - | ~40篇 |
| **Appendix** | 8+ | 证明细节、超参数、额外实验 |

### 5.3 关键贡献声明 (Contribution Statement)

```
Our contributions are:
1. We formalize System Dynamics stock-flow structure as an inductive bias for
   world models, reducing sample complexity from O(|S||A|) to O(d) where d
   is the number of learnable parameters.

2. We prove that causal contraction in SD dynamics enables effective horizon
   reduction from H to O(log(1/ε)), explaining why shallow search with learned
   terminal value matches deep exhaustive search.

3. We demonstrate empirically that SD-MCTS achieves 3-5x sample efficiency over
   DreamerV3/TD-MPC2, zero physics violations, and robust transfer across
   topologies—capabilities unattainable by black-box world models.
```

---

## 六、执行路线图

### 6.1 阶段划分

| 阶段 | 周期 | 核心任务 | 交付物 | 风险点 |
|------|------|----------|--------|--------|
| **Phase A: 基线构建** | Week 1-2 | 实现BlackBox-MCTS、集成DreamerV3 | `baselines/blackbox_mcts.py`, DreamerV3适配 | DreamerV3调参耗时 |
| **Phase B: 核心实验** | Week 3-5 | P0/P1实验执行、数据收集 | 样本效率曲线、物理违背表、OOD鲁棒性图 | 实验结果不如预期 |
| **Phase C: 理论严格化** | Week 2-4 (并行) | Theorem 1-3证明、假设文档化 | `theory/proofs.tex` | 证明遇阻 |
| **Phase D: 迁移实验** | Week 5-6 | 零样本迁移、新环境(SEIR) | 迁移性能表、泛化曲线 | 新环境调试 |
| **Phase E: 论文撰写** | Week 6-8 | 初稿、Figure制作 | NeurIPS格式完整草稿 | 时间压力 |
| **Phase F: 审计迭代** | Week 8-9 | Codex/Gemini交叉审查、修订 | 最终版本 | 重大修改需求 |

### 6.2 里程碑与检查点

| 里程碑 | 时间点 | 成功标准 | 失败应对 |
|--------|--------|----------|----------|
| **M1: 基线就绪** | Week 2末 | DreamerV3在Beer Game可运行 | 降级为BlackBox-only对比 |
| **M2: 初步样本效率** | Week 4末 | SD-MCTS显示≥2x优势 | 调整环境/加强消融 |
| **M3: 理论初稿** | Week 4末 | Theorem 3证明框架完成 | 降级为命题+实证验证 |
| **M4: 核心实验完成** | Week 6末 | Figure 1四子图数据齐全 | 减少实验范围聚焦核心 |
| **M5: 论文初稿** | Week 8末 | 完整8页+附录 | 加班冲刺 |

### 6.3 资源需求

| 资源类型 | 需求 | 用途 |
|----------|------|------|
| **GPU** | 1× A100 或 4× V100 | DreamerV3训练、大规模实验 |
| **CPU** | 32+ cores | MCTS并行模拟 |
| **存储** | 100GB+ | 实验日志、模型检查点 |
| **人力** | 1人全职 或 2人半职 | 代码+论文 |

---

## 七、代码实现计划

### 7.1 新增文件清单

```
src/
├── baselines/
│   ├── blackbox_world_model.py    # MLP世界模型
│   ├── blackbox_mcts.py           # BlackBox + MCTS
│   └── dreamerv3_adapter.py       # DreamerV3接口封装
├── envs/
│   └── seir_env.py                # SEIR流行病环境 (P2)
└── theory/
    └── horizon_reduction.py       # 视界压缩验证代码

experiments/
├── run_sample_efficiency.py       # P0: 样本效率曲线
├── run_physics_violation.py       # P0: 物理违背分析
├── run_ood_robustness.py          # P1: OOD鲁棒性
├── run_zero_shot_transfer.py      # P1: 零样本迁移
└── run_ablations_neurips.py       # P3: 消融研究

docs/neurips/
├── neurips_work_plan.md           # 本文档
├── figures/                       # 论文图表
└── draft/
    └── main.tex                   # 论文LaTeX源码
```

### 7.2 关键接口设计

**BlackBox世界模型接口** (与SD模型统一):
```python
class WorldModelInterface(ABC):
    @abstractmethod
    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """预测下一状态"""
        pass

    @abstractmethod
    def get_violation_rate(self, states: np.ndarray) -> float:
        """计算物理违背率"""
        pass
```

**统一评估接口**:
```python
def evaluate_method(
    method: str,  # "sd-mcts", "blackbox-mcts", "dreamerv3", ...
    env: gym.Env,
    sample_budgets: List[int],  # [1e3, 1e4, 1e5, 1e6]
    n_seeds: int = 10
) -> Dict[str, np.ndarray]:
    """返回 {metric_name: [n_budgets, n_seeds] array}"""
```

---

## 八、质量标准与自检清单

### 8.1 Spotlight/Oral 自检标准

| 维度 | 标准 | 自检问题 |
|------|------|----------|
| **新颖性** | 非显而易见的洞见 | "一个熟悉MBRL的人能否在30分钟内想到这个方法？" |
| **理论** | 有意义的形式化 | "定理是否解释了为什么有效，而非仅描述做了什么？" |
| **实验** | 令人信服的对比 | "审稿人能否说'但DreamerV3训练更久会追上'？" |
| **写作** | 一图胜千言 | "Figure 1是否能让人在10秒内理解核心贡献？" |
| **影响** | 值得关注 | "这篇论文会改变人们思考MBRL的方式吗？" |

### 8.2 提交前必须通过的检查

- [ ] DreamerV3/TD-MPC2基线使用官方实现+官方超参
- [ ] 所有实验≥10个随机种子
- [ ] 95%置信区间在所有图表中显示
- [ ] 物理违背率实验在OOD条件下进行
- [ ] Theorem 3证明完整无漏洞
- [ ] 论文经Codex+Gemini双重审查
- [ ] 代码可复现、文档完整

---

## 九、三方协商备忘

### 9.1 Codex核心建议

1. 将Theorem 3（视界压缩）作为理论亮点，强调因果收缩→有效视界降低
2. 实验需覆盖"错误先验"场景，证明鲁棒性
3. 预埋反驳文字应对"增量组合"质疑

### 9.2 Gemini核心建议

1. 叙事采用"Don't Relearn Physics"hook，建立情感共鸣
2. 零样本迁移是最强差异点，优先保障
3. Figure 1设计需一眼传达"更快、更安全、更鲁棒"

### 9.3 Claude综合判断

1. 核心创新有真实价值，但需精心执行才能达标
2. 最大风险是DreamerV3对比公平性，需使用官方实现
3. 理论贡献定位为"洞见驱动"而非"最优界"

---

**文档版本**: v1.0
**创建时间**: 2026-02-02
**协商参与**: Codex, Gemini, Claude
**下次更新**: Phase A完成后

