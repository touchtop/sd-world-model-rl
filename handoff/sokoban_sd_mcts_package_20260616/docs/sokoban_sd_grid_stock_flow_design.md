# Sokoban 格点级 SD 设计（grid stock-flow design v1）

本设计取代 `sokoban_sd_stock_flow_design.md`（六存量 learned flow 版）作为 Sokoban 线 SD 核心的现行定义。旧设计及其 stage-1/2/3 产物整体冻结为负结果与诊断方法素材，不再演进。

## 1. 研究定位

Sokoban 的物理转移精确已知且无未知参数，SD 框架的参数辨识支柱在此退化为平凡情形；本场景隔离检验另外两根支柱：结构约束搜索（Π_C）与终端价值对齐（TVA），以及零样本拓扑干预。

设计遵循"三槽位原则"——SD 的抽象性只允许进入三个非状态槽位，全部锚定在精确的格点存量-流量基底上：

1. **事件粒度抽象**：以 push 为流量事件，玩家区域内行走被折叠（时间抽象，保解性）。
2. **反馈/约束抽象**：求解动力机制（不可逆性、自由度耗散）表达为存量构型对流量容量的结构性门控，进入 Π_C；每条反馈律必须可独立审计。
3. **价值抽象**：剩余价值 `V_psi(x)` 是不闭合的低维泛函，由 TVA 学习；旧六标量降级为其输入特征。

判据依据：低维、闭合（Markov）、价值充分三者在 Sokoban 上不可兼得（aliasing 审计 0.39 与 oracle abstract 0.0 成功率为实证）。状态层必须保持闭合与价值充分，即格点级。

## 2. 核心对象与记号

关卡 `L = (C, W, F, T, N)`：格子集合 `C`，墙 `W`，floor `F = C \ W`，目标格 `T ⊆ F`，箱子数 `N`。方向集 `D = {上, 下, 左, 右}`。有向边集 `E = {(c, c+d) : c ∈ F, c+d ∈ F, d ∈ D}`。

## 3. 存量

| 存量 | 定义 | 守恒律 |
|---|---|---|
| 箱子占用 `b ∈ {0,1}^F` | `b_c = 1` 当且仅当格 `c` 上有箱子 | `Σ_c b_c = N`（严格质量守恒） |
| 玩家占用 `p ∈ {0,1}^F` | 玩家位置 one-hot | `Σ_c p_c = 1` |

状态 `x = (b, p)`。二者均通过时间冻结测试（冻结时可盘点、只能被流量改变），且 `x` 与原始可解性等价类一一对应（经第 5 节正则化后），零 aliasing。

**明确排除**：goal_completion、congestion、mobility、reachable_capacity、unresolved_box_pressure、irreversibility_risk 等均为 `x` 的瞬时泛函（辅助变量），不得作为存量（见第 8 节）。

## 4. 流量与关联矩阵形式

流量事件 = 一次 push `e = (c, d)`：一单位箱子质量沿边 `(c, c+d)` 转移，玩家从 `c−d` 迁移至 `c`。

箱子存量更新写为论文 §4.2 的标准形式：

```text
b_{t+1} = b_t + B f_t
```

其中 `B ∈ {−1, 0, +1}^{F×E}` 为已知带符号关联矩阵（边 `(c, c')` 的列在 `c` 处为 `−1`、`c'` 处为 `+1`），`f_t ∈ {0,1}^E` 为 one-hot 非负边流量。玩家更新为已知确定性算子 `p_{t+1} = 1_c`（push 后玩家位于箱子原格）。

性质：单步误差恒为 0，多步 rollout 误差恒为 0，无 clip、无残差、无可学习参数。这是论文 Proposition 2 误差界的极限情形（`ε_model = 0`），规划偏差完全由截断偏差构成——即 Proposition 3 中 TVA 的主场。

## 5. 状态恒等与玩家区域正则化

push 粒度下玩家在其可达连通域内的具体位置不影响后续可行流量。状态恒等采用正则化键：

```text
κ(x) = (b, min_cell(Reach(b, p)))
```

`Reach(b, p)` 为玩家在当前箱子布局下的可达域（flood fill）。搜索去重、revisit 检测、价值缓存均以 `κ(x)` 为键。

## 6. 可行域与投影 Π_C

流量 `e = (c, d)` 在状态 `x` 下可行，当且仅当：

1. **容量**：`c+d ∈ F` 且 `b_{c+d} = 0`；
2. **源**：`b_c = 1`；
3. **站位**：`c−d ∈ F`、`b_{c−d} = 0` 且 `c−d ∈ Reach(b, p)`；
4. **不变集排除**：`b_{t+1}` 不落入 Tier-A 死锁不变集（第 7 节）。

Π_C 的实现形态为动作集限制 + 死态终止：搜索树只在可行流量上生长；判定为 proven dead 的状态作为终止节点，价值取 `V_dead`。

## 7. 反馈律（双层防火墙）

每条反馈律的形式：存量构型模式 → 对（当前或全部未来）流量容量的影响。按可靠性分两层，**层级边界即防火墙**：

### Tier-A：已证可靠，进入 Π_C

| 律 | 机制 | 可靠性来源 |
|---|---|---|
| F1 占用门控 | `b_{c'} = 1` 或 `c' ∈ W` ⇒ 所有流入 `c'` 的流量容量为 0 | 规则定义，构造性成立 |
| F2 死格律 | `D = {c ∈ F : 自任意目标格的 reverse-pull 不可达}`；箱子进入 `D \ T` ⇒ 该箱所有未来有效出流容量恒为 0 ⇒ 状态吸收死 | 静态可证，每关预计算一次 |
| F3 冻结律 | 箱子两轴均被墙/已冻结箱锁死且不在目标上 ⇒ 吸收死。现行采用保守"静态墙"判据；完整递归版须先通过零误杀审计方可升入 Tier-A | 需审计（验收门限 G3） |
| F4 目标吸收 | `Σ_{c∈T} b_c = N` ⇒ 终止 solved | 任务定义 |

### Tier-B：启发式，禁止进入 Π_C / reward / prior

congestion、mobility、reachable capacity、目标区整理压力、匹配下界等全部辅助量。**消费方式仅限**：`V_psi` 输入特征、诊断面板、离线审计指标。任何 Tier-B 量要影响在线规划（门控、奖励、先验），必须作为独立学习组件走显式准入门限（不劣于结构基线 + 默认关闭 + closed-loop smoke 非零），不得手工上线。

该防火墙是对 stage-3 启发式堆砌教训的制度化回应。

## 8. 辅助变量（readouts）

自 `x` 瞬时计算的泛函 `a(x)`，包括但不限于：目标完成率、未上目标箱压力、可行流量数（mobility）、可达域占比、拥塞度、不可逆风险代理、匹配下界、reverse-pull 距离场。

角色：`V_psi` 输入特征、可解释性面板、审计指标。旧六存量的 proxy 计算代码原样保留并归入此层。

## 9. 奖励与价值接口（SD-MCTS / TVA 合同）

- **转移 provider**：`exact_grid_stock_flow_v1`，提供 `legal_flows(x)`、`step(x, f)`（含逐步守恒断言）、`terminal(x) ∈ {solved, dead, none}`、`auxiliaries(x)`；无可学习参数。
- **奖励**：固定且极简——每次 push 代价 `−c_push`；solved 终止 `+R_goal`；proven dead 终止 `V_dead`。**禁止任何 shaping**：全部长程信号必须经由 `V_psi` 流入规划，使 TVA 的贡献可被干净归因。
- **叶值**：深度 `H` 截断处调用 `V_psi(φ(x), a(x))`。
- **TVA 训练**：按论文 §4.3，target 为根节点统计 `V_search(s) = Σ_a π_mcts(a|s) Q_H(s,a)` 加 `λ_td` TD 项；在真实状态空间的搜索统计上采集，迭代自举。
- **核心实验**：`H × budget × (TVA on/off)` 的解题率矩阵，对应"短视界 + TVA ≥ 长视界纯 MCTS"命题。
- **零样本干预**：改墙 = 改边容量 / `B` 的支撑集；改目标 = 改吸收集 `T` 与死格集 `D`；动力学无需重训，直接复跑 MCTS。

## 10. 验收门限（SD 部分）

| # | 门限 | 判据 |
|---|---|---|
| G1 | 守恒审计 | 全部 rollout 中 `Σ b_c ≡ N`、`Σ p_c ≡ 1`，违例率 0% |
| G2 | 转移保真 | `B f_t` 关联矩阵形式与 `step_state` 在采样关卡全量合法 push 上逐位一致 |
| G3 | 剪枝可靠性 | 在可全搜索小关卡集上，F2/F3 零误杀（不剪除任何可解状态）；同时报告剪枝功效（捕获死态比例） |
| G4 | 黑盒对照 | 同数据训练 latent MLP 转移模型，多步 rollout 误差曲线对照 SD ≡ 0；物理违例率表 |
| G5 | 状态恒等 | `κ(x)` 与 flood-fill 区域等价性的性质测试通过 |

当前实现中的 G1/G2/G5 由 `validate_sokoban_grid_stock_flow_design.py` 执行。2026-06-11 smoke 结果为 `grid_sd_gate=PASS`：`3` 个关卡、`15` 个状态、`64` 个合法 push 转移、`308` 个可达玩家位置，守恒、`Bf` 转移和 canonical key 违例均为 `0`。G3 的“零误杀”全搜索版本和 G4 black-box latent 对照仍是后续正式审计项。

## 10.5 验收门限（TVA 部分）

TVA 必须在进入 closed-loop SD-MCTS 前单独验证。独立 TVA 审计的对象是真实格点状态上的截断剩余价值，而不是六存量 abstract rollout 统计。当前脚本为 `audit_sokoban_grid_tva.py`，其合同为：

```text
transition_provider = exact_grid_stock_flow_v1
state_input = exact_grid_observation_plus_auxiliary_readouts
teacher_usage = offline_value_label_only
teacher_used_online = false
closed_loop_evaluation = false
legacy_six_stock_rollout_used = false
```

门限包括：valid 样本数、正值样本覆盖、valid 相关性和 root successor sorting 覆盖。通过门限后，审计脚本可导出 ridge checkpoint，供 `evaluate_sokoban_grid_sdmcts.py --leaf-value tva` 读取；closed-loop 脚本只消费 checkpoint，不调用 teacher。

2026-06-11 smoke 输出 `grid_tva_independent_audit_smoke_seed0.json`，`tva_independent_gate=PASS`：valid count `16`，MAE `1.3878170000076102`，corr `0.9736159851420181`，sign accuracy `1.0`，root sorting top-tier hit `5/5`。该结果只表示 TVA 管线具备独立审计与 leaf checkpoint 消费能力；是否带来规划增益必须由后续 `H × budget × (TVA on/off)` 矩阵判断。

## 11. 与现有代码的映射

| 现有代码 | 在新设计中的角色 |
|---|---|
| `SokobanPushDynamicsModel.step_state` / `encode_tensor_state`（`src/sokoban/sd_model.py`） | 精确流量施加；补 `B f_t` 形式审计（G2） |
| `SokobanGoalDirectedSDModel.reachable_cells` / `legal_pushes` | F1 + 站位前置条件 |
| dead-square 预计算、`frozen_boxes` / `is_deadlocked` | F2 / F3（F3 完整版须过 G3 后升 Tier-A） |
| `matching_lower_bound`、pull 距离场 | Tier-B 辅助变量 |
| 六存量 proxy 计算 | Tier-B 辅助变量（`V_psi` 特征） |
| stage-1/2 learned flow、stage-3 手工 prior/option 旋钮 | 冻结归档，不迁移 |

最小新增：`B` 矩阵构造与守恒断言、`κ` 正则化、Tier-A/B 分层接口、G1–G5 审计脚本。

当前新增实现为：`SokobanGridStockFlowModel`、`validate_sokoban_grid_stock_flow_design.py`、`audit_sokoban_grid_tva.py`、`evaluate_sokoban_grid_sdmcts.py`。`evaluate_sokoban_grid_sdmcts.py` 支持 `--require-independent-gates`，可强制 SD 与 TVA 审计先通过再运行 closed-loop。

## 12. 明确排除项

1. 不在抽象标量上做 rollout（任何形式的 learned abstract transition）。
2. 不做 reward shaping。
3. 不手工上线任何 Tier-B 量为在线 prior / 门控。
4. 不向动力学层人为引入可学习参数以凑"参数辨识"叙事。

## 13. 开放问题

1. `R_goal`、`V_dead`、`c_push`、`γ` 的数值尺度需与 TVA target 尺度联动设定，待 TVA 协议设计时一并定稿。
2. `H` 扫描范围（候选 `{1, 2, 4, 8}`）与 budget 网格，待规划实验设计定稿。
3. move-level 严格流量展开（玩家逐格行走作为流量事件）是否写入论文附录作为形式化补全。
