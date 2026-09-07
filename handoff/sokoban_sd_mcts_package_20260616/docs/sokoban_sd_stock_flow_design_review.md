# Sokoban/Boxoban Stock-Flow SD Design Revised Review

## 审查范围

本复审基于修订后的 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)、[sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)、[sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md) 与 [sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md) 完成。

本次复审调用 `academic-research-suite` 的 `academic-paper-reviewer` 工作流，并按 `methodology_reviewer_agent`、`devils_advocate_reviewer_agent` 和 `editorial_synthesizer_agent` 的角色规范在当前 Codex 会话内执行。未调用外部独立模型，也未把文档发送到外部 API。

## 修订后设计快照

修订版首版设计为：

```text
核心存量: G_t, U_t, M_t, R_t, C_t, I_t
状态转移: x_{t+1} = clip_X(x_t + B f_theta(x_t, a_t, c, h_t, b_t))
核心流量: 非负 f_theta，经已知 incidence matrix B 作用于存量
核心反馈: 目标压力反馈, 拥塞-机动性-风险反馈, 不确定性反馈
首版动作: 合法 single-push option
规划接口: SD-MCTS over learned abstract dynamics
扩展项: goal-room pressure, plan progress, structural slack, macro duration, path-channel feedback
```

## Methodology Reviewer 复审

### 总体判断

修订版已经从“宽泛机制清单”收缩为可实现的系统动力学世界模型框架。其方法设计能够回答当前研究问题：如何在抽象高层空间状态下，用结构已知、部分参数未知的 stock-flow-feedback 模型刻画面向任务目标的内生状态转移，并由 SD-MCTS 消费该学习到的抽象动力学。

### 存量逐项复审

| 存量或项 | 必要性 | 合理性 | 冲突检查 | 复审结论 |
| --- | --- | --- | --- | --- |
| `G_t` | 必须 | 任务完成存量可直接观测，且与奖励、终止概率相关。 | 已允许 `goal_unfill_flow`，未假设单调增加。 | 通过。实现时需区分普通目标占用和稳定目标占用。 |
| `U_t` | 必须 | 未解决压力不再等同于 `1-G_t`，引入匹配残差和分配不确定性后具有独立意义。 | 与 `G_t` 的近似守恒关系已保留，但允许偏离项。 | 通过。需要在数据 schema 中记录数量分量和难度分量。 |
| `M_t` | 必须 | 表示后续可操作能力，符合 Sokoban 可恢复性机制。 | 与 `R_t` 的分工已明确：`M_t` 是 option viability，`R_t` 是空间连通容量。 | 通过。需报告二者相关性，防止训练中共线。 |
| `R_t` | 条件必要但首版可保留 | 玩家可达域是 single-push 可执行性的基础。 | 与 `M_t` 的边界已清楚；若不可辨识，可后续合并。 | 通过，带实现期相关性审计。 |
| `C_t` | 必须 | 拥塞可解释死锁前的渐进恶化。 | `P_t` 已降级为派生特征，避免目标区压力与拥塞重复建模。 | 通过。实现时需固定局部窗口、入口和通道代理。 |
| `I_t` | 必须 | 当前状态风险压力或概率是 Sokoban 与普通网格任务的关键差异。 | 已与动作后果风险 `risk_t` 分离；risk relief 仅在风险压力语义下成立。 | 通过。风险标签必须分级，不能只来自启发式规则。 |
| `P_t` | 非首版核心 | 作为 goal-room pressure 派生特征合理。 | 不再与 `C_t` 竞争核心存量地位。 | 通过。满足独立预测增益后再升级。 |
| `L_t` | 非首版核心 | 作为 plan-progress probe 或辅助头合理。 | 避免把 MCTS 内部搜索进度混入环境状态。 | 通过。不得作为首版环境存量。 |
| `S_t` | 非首版核心 | 作为 structural slack 诊断合理。 | 不再与 `M_t/R_t/C_t/I_t` 重叠成不可辨识潜变量。 | 通过。是否升级应由 aliasing 诊断决定。 |

### 流量与 incidence matrix 复审

| 结构项 | 必要性 | 合理性 | 冲突检查 | 复审结论 |
| --- | --- | --- | --- | --- |
| `f_theta >= 0` | 必须 | 非负流量符合 stock-flow 建模习惯。 | 通过 `B` 表示方向，避免 inflow/outflow 任意抵消。 | 通过。 |
| `B f_theta` | 必须 | 已显式解决上一版流量双计风险。 | 每个流量对存量的符号固定，便于审计。 | 通过。实现需把 `B` 固化到配置或 schema。 |
| `goal_fill_flow` / `goal_unfill_flow` | 必须 | 覆盖目标完成与回退。 | 已同时作用 `G_t` 和 `U_t`，避免另设重复 completion 流。 | 通过。 |
| `assignment_relief_flow` / `assignment_worsening_flow` | 必须 | 解释未直接改变目标占用但改变任务压力的动作。 | 与 `goal_fill_flow` 的边界已说明。 | 通过。实现需记录残差变化来源。 |
| `mobility_*` 与 `reachability_*` | 必须 | 区分可操作能力与玩家空间容量。 | 与拥塞释放通过反馈关联，而不是同一流量同时全部修改。 | 通过，需相关性审计。 |
| `congestion_*` | 必须 | 拥塞作为中间机制合理。 | 目标房间压力作为子项进入，未单独核心化。 | 通过。 |
| `risk_*` | 必须 | 风险累积与风险缓解均有语义边界。 | 已声明已证明 dead-end 不能被 relief 恢复。 | 通过。 |

### 反馈复审

| 反馈 | 必要性 | 合理性 | 冲突检查 | 复审结论 |
| --- | --- | --- | --- | --- |
| 目标压力反馈 | 必须 | `U_t` 与 goal-room 派生压力调节目标相关流量，符合任务组织机制。 | 完成释放作为条件分支处理，未新增重复回路。 | 通过。 |
| 拥塞-机动性-风险反馈 | 必须 | 体现 Sokoban 中拥塞、可操作性和不可逆风险的内生耦合。 | 已限定反馈影响未来流量和规划权重，不改写已发生转移。 | 通过。 |
| 不确定性反馈 | 必须 | MBRL 和 SD-MCTS 必须处理模型误差。 | 已区分 epistemic 与 aleatoric uncertainty。 | 通过。实现时需选择校准方法。 |
| 计划通道反馈 | 非首版核心 | 作为 DRC/path-channel 扩展有价值。 | 不再污染首版核心状态。 | 通过。 |
| 结构余量反馈 | 非首版核心 | 作为后续可恢复性扩展有价值。 | 不再引入不可辨识存量。 | 通过。 |

### 编码器、预测头和学习目标复审

| 结构项 | 复审结论 | 需要实现检查 |
| --- | --- | --- |
| 可观测存量 `o_t` | 合理，保证存量语义可审计。 | 每个 `z_t` 特征需标注用途：stock proxy、flow gate、risk label 或 diagnostic。 |
| 有界残差 `delta_t` | 合理，修复上一版 residual 任意覆盖存量的问题。 | 报告 residual activity 和 residual ablation。 |
| 隐藏上下文 `h_t` | 合理，允许吸收六存量外的信息。 | 不得把 `h_t` 解释为已命名 SD 存量；需做 black-box leakage 消融。 |
| `x_{t+1}` target | 合理。 | 使用 stop-gradient target encoder 或显式代理标签，避免目标漂移。 |
| `risk_t` head | 合理。 | 日志中必须与 `I_t` 分列；风险标签必须分级。 |
| `sigma_t` head | 合理。 | 至少采用误差分桶、NLL、ensemble disagreement 或 conformal calibration 之一。 |
| `feasibility_t` head | 必要。 | 用 raw-state handle 检查 single-push option 是否真实可执行。 |
| `L_aliasing_diagnostic` | 必要。 | 报告抽象邻域内真实后果方差，不应把它当普通损失掩盖问题。 |

### Methodology 结论

修订版在方法上可作为首版实现基线。其主要改进是：核心存量可辨识性增强，流量方向可审计，反馈回路收缩到必要机制，SD-MCTS 接口不再被启发式搜索污染。剩余风险主要是实现层面的标签构建、uncertainty calibration、state aliasing 和 residual 覆盖问题。

## Devil's Advocate 复审

### 最强反方意见

修订版虽然避免了九存量过度复杂，但仍可能被真实评审质疑为“把经典 Sokoban 启发式重新包装成 SD 世界模型”。这种质疑的核心不在文档表述，而在实现证据：如果训练标签主要来自 matching lower bound、dead-square、PDB warning 或旧 best-first 轨迹，而 learned transition 展开没有真正替代精确 push dynamics，那么模型即使输出 `G_t/U_t/M_t/R_t/C_t/I_t` 和 `flows_t`，也可能只是启发式代理的可解释外壳。要抵消这一攻击，正式实验必须证明学习参数、反馈回路和 SD-MCTS learned rollout 分别提供增益，并且这些增益不能由 `no-learned-parameter` 或 `heuristic-search diagnostic` 复现。

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Location |
| --- | --- | --- | --- |
| C1 | 实现可证伪性 | 修订文档已经要求 learned transition provider，但若实现阶段未记录 provider，正式结果仍可能退回精确转移加启发式评分。 | SD-MCTS 接口、诊断与消融 |

#### MAJOR

| # | Dimension | Issue Description | Location |
| --- | --- | --- | --- |
| M1 | 标签循环 | 风险标签若主要由 dead-square、matching 或启发式 warning 产生，会证明模型拟合了启发式，而不是学习任务后果。 | 学习目标、风险校准 |
| M2 | 抽象状态别名 | 六存量可能把后果不同的箱子排列映射到相近 `x_t`，导致 MCTS imagined branch 不可执行或价值错误。 | 编码器、SD-MCTS 接口 |
| M3 | 残差覆盖 | `h_t` 和 `delta_t` 若解释过强，可能让显式 SD 存量只成为展示层。 | 编码器 |
| M4 | `M_t/R_t` 共线 | 机动性和可达容量仍可能高度相关，训练后未必可辨识。 | 存量定义 |
| M5 | 不确定性误用 | 若不区分 epistemic 与 aleatoric uncertainty，探索项可能鼓励进入不可恢复坏状态。 | 不确定性反馈 |

#### MINOR

| # | Dimension | Issue Description | Location |
| --- | --- | --- | --- |
| m1 | 单位细化 | `U_t` 的数量分量和难度分量需要在实现 schema 中列出具体归一化。 | 核心存量定义 |
| m2 | goal-room 派生项 | `P_t` 降级后仍需给出目标区入口压力的可复现计算方式。 | 派生特征 |
| m3 | macro 扩展边界 | 第二阶段 macro 引入时需新增 duration 和低层执行失败标签。 | 替换路线 |

### Devil's Advocate 结论

文档层面没有发现必须推翻修订版的逻辑矛盾。最强攻击点已经从“设计本身不符合 SD-MCTS”转移为“实现和实验能否证明不是启发式包装”。因此复审建议保留修订版，但把 transition provider、风险标签分级、aliasing 方差和 no-learned-parameter 消融设为硬性验收门槛。

## Editorial Synthesis 复审

### 上一轮问题追踪矩阵

| 上一轮问题 | 修订状态 | 仍需实现检查 |
| --- | --- | --- |
| 九存量过度复杂，`P_t/L_t/S_t` 不可辨识 | 已解决。首版收缩为 `G_t,U_t,M_t,R_t,C_t,I_t`，三者降级为派生或扩展。 | 后续若升级三者，需给出独立监督和增益消融。 |
| `G_t` 与 `U_t` 冗余 | 基本解决。`U_t` 改为 unresolved pressure，包含匹配残差和分配不确定性。 | schema 需分列数量项与难度项。 |
| `M_t` 与 `R_t` 重叠 | 部分解决。语义分工已明确。 | 训练后报告相关性和合并消融。 |
| `C_t` 与 `P_t` 重叠 | 已解决。`P_t` 降级为 `C_t` 派生输入。 | goal-room proxy 需可复现。 |
| `I_t` 与 `risk_t` 混淆 | 已解决。`I_t` 是状态风险压力，`risk_t` 是动作后果或分支风险。 | 日志和损失必须分列。 |
| 流量双计 | 已解决。引入 `x_{t+1}=clip_X(x_t+B f_t)`。 | `B` 需固化并进入实验配置。 |
| 反馈过多且重复输入 | 已解决。收缩为三类核心反馈。 | 反馈增益与普通输入特征需在模型接口分开。 |
| learned residual 破坏可解释存量 | 基本解决。改为可观测存量、有界残差和隐藏上下文。 | 报告 residual activity 和 residual ablation。 |
| 抽象状态别名 | 部分解决。已加入 raw-state handle、belief support 和 aliasing 诊断。 | 必须在正式实验中实际报告。 |
| 风险标签循环 | 部分解决。已要求三类标签。 | 需要构建真实转移或短 rollout 标签，降低纯启发式标签占比。 |
| SD-MCTS 被精确转移污染 | 部分解决。已要求 learned transition provider audit。 | 实现时必须记录每次 rollout provider。 |
| macro 过早进入首版动作 | 已解决。首版限定 single-push option。 | 第二阶段引入 macro 时需新增 duration 和 adapter 失败审计。 |

### Editorial Decision

**决定：通过作为首版模型设计基线，附带实现期强制检查项。**

理由如下。第一，修订版已经明确 `SD = stock-flow-feedback`，并把核心状态转移写成带 incidence matrix 的可学习动力学，而不是静态启发式打分。第二，首版模型复杂度已降至六个核心存量，降低了不可辨识风险。第三，SD-MCTS 接口已明确要求 learned abstract transition、raw-state handle 或 belief support、feasibility 检查和 provider 日志，能够与旧 best-first 诊断草稿划清边界。

仍需保留的强制检查项包括：

1. 固化 `B` 矩阵和每个流量的符号约束；
2. 分级构建风险标签，并报告各标签来源占比；
3. 对 `I_t` 与 `risk_t` 分别记录指标；
4. 报告 `M_t/R_t` 相关性、state aliasing 方差和 residual activity；
5. 在 SD-MCTS rollout 日志中记录 transition provider；
6. 完成 `no-learned-parameter`、`no-feedback`、`black-box latent MBRL` 和 `heuristic-search diagnostic` 消融；
7. 将旧 `eval_medium_train000_50.json` 继续标注为 smoke/diagnostic baseline，不作为 SD-MCTS 结果。

### 最终复审结论

修订后的设计没有破坏用户设定的研究框架。它以系统动力学的存量、流量和反馈为核心，以结构已知、部分参数未知的可学习世界模型为对象，以 `MBRL + SD-MCTS` 为整体框架，并把 Sokoban/Boxoban 定位为方法论研究场景。当前文档层面已经避免把世界模型退化为无需学习的启发函数；下一阶段的关键在于实现和实验审计必须严格执行上述强制检查项。
