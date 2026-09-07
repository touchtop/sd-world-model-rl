# Sokoban/Boxoban Stock-Flow SD World Model Design

## 研究定位

本设计用于研究抽象高层状态空间下的系统动力学世界模型构造方法。Sokoban/Boxoban 只作为便于观察任务目标、不可逆状态、组合约束和规划表征的离散场景，不把求解该题本身作为研究目标。

本设计中的 `SD` 指系统动力学。世界模型由存量、流量和反馈构成，并在“结构已知、部分参数未知”的形式下学习任务目标状态转移的内生机制。它服务于 `MBRL + SD-MCTS` 框架，规划发生在学习到的抽象 SD 状态动力学上，而不是在静态启发函数上做 best-first search。

当前 [src/sokoban/search.py](/home/data/round/code/sd_world_model_rl/src/sokoban/search.py) 与 50 关卡评测只保留为诊断性草稿。该草稿说明玩家可达域、推箱动作、匹配下界和死锁证据可以作为结构观测代理，但不能替代可学习的 SD 世界模型。

## 核心对象

令原始 Sokoban 状态为 `s_t`，静态关卡结构为 `c`，首版高层动作 `a_t` 为一个合法 single-push option，即一次可执行推箱动作及其隐含玩家移动。tunnel macro、goal macro、relevance cut 和 path-channel 机制暂作为结构代理、诊断项或第二阶段扩展，不进入首版动作集合。

编码器把原始状态投影到抽象 SD 状态：

```text
o_t, h_t = E_phi(s_t, c)
x_t = clip_X(o_t + delta_phi(s_t, c, o_t))
```

其中 `o_t` 是可观测存量代理，`delta_phi` 是有界残差，`h_t` 是单独的隐藏上下文。残差只允许在预设幅度内修正可观测存量，不允许任意覆盖存量语义；隐藏上下文可以服务预测，但不得被解释为额外核心存量。

首版核心存量固定为六个：

```text
x_t = [G_t, U_t, M_t, R_t, C_t, I_t]
```

其中 `GoalRoomPressureStock P_t`、`PlanProgressStock L_t` 和 `StructuralSlackStock S_t` 不再作为首版核心存量。三者分别降级为派生特征、辅助预测头或第二阶段扩展。

## 核心存量定义

| 存量 | 单位与范围 | 可观测标签来源 | 设计用途 |
| --- | --- | --- | --- |
| `GoalCompletionStock G_t` | 目标完成比例，`[0, 1]`。首版可用 `boxes_on_goals / num_boxes`，可选稳定目标占用版本。 | 当前箱子在目标上的数量、目标区占用。 | 预测任务进展、奖励和终止概率；允许被推出目标导致回退。 |
| `UnresolvedBoxStock U_t` | 未解决压力，建议归一到 `[0, 1]`，必要时保留截断前压力值。 | 未上目标箱子比例、匹配残差、目标分配不确定性。 | 表示剩余组织和运输压力；与 `G_t` 近似守恒，但不等同于 `1-G_t`。 |
| `MobilityStock M_t` | 可恢复操作能力，`[0, 1]`。 | 合法 single-push 数、可行 push 数、短 rollout 中后续可推性。 | 表示后续仍能调整布局的能力，不直接等同于当前动作数。 |
| `ReachableCapacityStock R_t` | 玩家可达空间容量，`[0, 1]`。 | 玩家可达格比例、入口连通性、cut-square 可通过性。 | 连接低层可达域和 single-push option 可执行性。 |
| `CongestionStock C_t` | 拥塞压力，`[0, 1]`。 | 局部箱密度、通道占用、目标入口占用、合法动作减少率。 | 描述死锁前的渐进结构恶化，并调节机动性和风险流。 |
| `IrreversibilityRiskStock I_t` | 当前状态风险压力或风险概率，`[0, 1]`。 | 可证明 dead-end、短 rollout failure、启发式 warning 的分级标签。 | 表示当前状态已接近不可恢复区域的程度；不同于动作后果风险 `risk_t`。 |

`P_t`、`L_t`、`S_t` 的首版处理如下：

| 项 | 首版角色 | 可升级条件 |
| --- | --- | --- |
| `GoalRoomPressureStock P_t` | `C_t` 的派生特征和流量输入，例如目标房间入口压力、填充顺序冲突。 | 已实现可靠 goal-room/entrance 分解，并能证明它提供超出 `C_t` 的独立预测信息。 |
| `PlanProgressStock L_t` | 辅助头或 probe，用于预测 path-channel、assignment route consistency 或短 rollout 进展。 | 存在独立监督信号，且不会把 MCTS 内部搜索进度混入环境世界模型状态。 |
| `StructuralSlackStock S_t` | 诊断指标，用替代目标数、可绕行路径数、局部 pattern 余量估计。 | aliasing 诊断显示六存量不足以区分可恢复困难状态与不可恢复死局。 |

设计上不允许把 `matching_lower_bound`、`risk_score` 或 `heuristic_score` 直接等同于存量；它们只能作为观测代理、流量输入、风险标签或诊断项。

## Stock-Flow 转移

首版转移采用带 incidence matrix 的 stock-flow 形式：

```text
f_t = softplus(H_theta(x_t, a_t, c, h_t, b_t)) * gate(a_t, c, z_t)
x_{t+1} = clip_X(x_t + B f_t)
b_t = feedback_theta(x_t, a_t, c, y_t, sigma_t)
```

其中 `f_t` 是非负流量向量，`B` 是已知的流量到存量影响矩阵，`gate` 是由合法动作、目标区结构、通道、dead-square、trap/invariant 和其他可核验结构产生的门控。`theta` 只学习流量强度、阈值、非线性耦合和校准参数，不改变已声明的存量语义和流量符号。

核心流量与存量影响如下。表中的 `+`、`-` 表示 `B` 中的固定方向，实际幅度由非负流量值学习。

| 流量 | `G_t` | `U_t` | `M_t` | `R_t` | `C_t` | `I_t` | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `goal_fill_flow` | `+` | `-` |  |  | 条件 `+/-` |  | 箱子进入目标或稳定目标区；若堵塞目标入口，可同时增加拥塞。 |
| `goal_unfill_flow` | `-` | `+` |  |  | 条件 `+/-` |  | 箱子从目标移出；区分必要回退与破坏性回退。 |
| `assignment_relief_flow` |  | `-` |  |  |  |  | 匹配残差或目标分配不确定性下降，但未必立即增加 `G_t`。 |
| `assignment_worsening_flow` |  | `+` |  |  |  | 条件 `+` | 路线一致性恶化或箱子进入更差分配区域。 |
| `mobility_loss_flow` |  |  | `-` |  | 条件 `+` | 条件 `+` | 后续可操作能力下降。 |
| `mobility_recovery_flow` |  |  | `+` |  |  |  | 通道释放或可推选择恢复；不直接降低 `I_t`。 |
| `reachability_loss_flow` |  |  |  | `-` | 条件 `+` | 条件 `+` | 玩家可达域或关键入口连通性下降。 |
| `reachability_recovery_flow` |  |  |  | `+` |  |  | 玩家可达域恢复或入口重新打开。 |
| `congestion_generation_flow` |  |  |  |  | `+` | 条件 `+` | 通道、目标入口或局部窗口拥塞累积。 |
| `congestion_release_flow` |  |  |  |  | `-` |  | 拥塞释放；对 `M_t/R_t` 的改善通过反馈影响后续流量。 |
| `risk_accumulation_flow` |  |  |  |  |  | `+` | dead-square 接近、trap/invariant、局部 PDB 死锁证据或 rollout failure。 |
| `risk_relief_flow` |  |  |  |  |  | `-` | 只在 `I_t` 表示风险压力或概率时成立；已证明 dead-end 不允许通过该流恢复。 |

该矩阵用于避免双计。例如目标完成导致的 `U_t` 下降由 `goal_fill_flow` 负责；若同一动作还降低匹配残差，只能由 `assignment_relief_flow` 表示额外压力释放，训练和日志中必须能区分二者。

## 核心反馈回路

反馈函数不直接返回“动作好坏”，而是调节后续流量、风险惩罚和不确定性使用方式。首版只保留三类核心反馈：

1. 目标压力反馈：`U_t` 和派生 goal-room pressure 调节 `goal_fill_flow`、`goal_unfill_flow`、`assignment_relief_flow` 与目标区拥塞门控。完成释放反馈属于该回路的条件分支，而不是独立第七类机制。
2. 拥塞-机动性-风险反馈：`C_t` 上升会提高未来 `mobility_loss_flow`、`reachability_loss_flow` 和 `risk_accumulation_flow` 的敏感度；`M_t/R_t` 恢复会降低后续风险累积倾向，但不改写已经发生的真实转移。
3. 不确定性反馈：`sigma_t` 分为 epistemic 与 aleatoric 两类使用。epistemic uncertainty 可提高 SD-MCTS 的验证性探索；aleatoric uncertainty 主要降低价值置信或提高风险折扣，不应无条件鼓励探索。

`L_t` 对应的计划通道反馈和 `S_t` 对应的结构余量反馈暂不进入首版核心回路。它们可以作为 probe 记录，以便判断是否有必要升级。

## 已知结构与待学习参数

已知结构来自关卡拓扑和文献机制，只限定因果连接、结构门控和诊断标签，不固定动作价值：

1. 静态墙、目标、箱子数量和玩家位置；
2. 玩家可达域、合法 single-push option 和低层可执行性；
3. reverse-pull 拓扑、箱子到目标的可达性、目标区入口和 cut-square；
4. dead-square、静态墙角、trap/invariant、局部 PDB 或多箱 pattern 的风险证据；
5. tunnel/goal macro、relevance、path-channel 等第二阶段结构代理。

待学习参数包括：

1. 各流量函数的权重、阈值和非线性耦合；
2. 反馈增益与条件门控校准；
3. 结构代理到真实后果之间的校准参数；
4. single-push 后果、失败模式和可执行性预测；
5. 风险与不确定性校准头；
6. 有界残差与隐藏上下文的映射。

这种划分保证 SD 的结构可解释，但预测能力来自学习，而不是来自手写启发式。

## 编码器

编码器 `E_phi` 由三部分组成：

```text
z_t = deterministic_features(s_t, c)
o_t = stock_proxy_projector(z_t)
delta_t = rho * tanh(residual_phi(s_t, c, z_t))
h_t = hidden_context_phi(s_t, c, z_t)
x_t = clip_X(o_t + delta_t)
```

`z_t` 包含可复现实验可计算的结构观测，例如目标占用、未完成箱子数、玩家可达面积、合法推箱数、可行推箱数、目标区入口压力、dead-square 证据、局部拥塞、匹配残差和短路径一致性。每个特征都应标注为 stock proxy、flow gate、risk label 或 diagnostic。

`delta_t` 只作为有界校正，建议逐存量设置小幅上限 `rho` 并报告 residual activity。`h_t` 用于吸收六存量无法表达的上下文，但不进入解释性 stock vector。若 `h_t` 对预测贡献过大，应通过消融报告，不得把它包装为已解释的系统动力学机制。

## 世界模型预测头

给定 `x_t`、`a_t`、`c` 和 `h_t`，SD 世界模型首版必须输出：

1. `x_{t+1}`：下一抽象存量状态；
2. `flows_t`：非负流量 `f_t`、incidence matrix 后的净变化 `B f_t` 和边界截断信息；
3. `progress_t` 或 `reward_t`：任务进展或环境奖励预测；
4. `terminal_or_success_t`：完成概率；
5. `risk_t`：动作后果或分支风险预测，与状态存量 `I_t` 区分记录；
6. `sigma_t`：转移、进展、风险和可执行性不确定性；
7. `feasibility_t`：从当前 raw-state handle 执行该 single-push option 的可行性；
8. `diagnostics_t`：与文献机制对应的可解释观测代理。

`duration_t` 在首版 single-push 设置下为常量或可省略。引入 macro option 后再作为正式预测头。`diagnostics_t` 可以包含 matching lower bound、deadlock evidence、macro candidate、relevance proxy 和 path-channel probe，但这些字段不构成核心状态转移方程。

## SD-MCTS 接口

SD-MCTS 在抽象 SD 状态空间中规划：

```text
raw Sokoban state s_t
  -> encoder E_phi(s_t, c) = x_t, h_t
  -> node stores x_t distribution, uncertainty, raw-state handle or belief support
  -> learned SD transition predicts x_{t+1}, flows, progress, risk, uncertainty
  -> MCTS expands legal single-push options under learned abstract dynamics
  -> feasibility adapter checks whether the chosen option can be executed in raw state
  -> environment transition produces s_{t+1}
  -> replay updates theta and phi
```

MCTS 节点不得只保存单个启发式值，也不得只保存 `x_t` 而丢失低层可执行性信息。节点至少保存 `x_t` 或其分布、uncertainty、raw-state handle 或 belief support。这样可以降低抽象状态 aliasing 导致“抽象后继可行、真实后继不可执行”的风险。

选择公式可以使用预测价值、风险惩罚和不确定性项：

```text
score(n, a) =
  Q_hat(n, a)
  + c_puct * prior_theta(n, a) * sqrt(N(n)) / (1 + N(n, a))
  - lambda_risk * risk_theta(n, a)
  + lambda_unc * epistemic_bonus(sigma_theta(n, a))
```

这里的 `prior_theta` 来自学习到的 SD 世界模型或策略头。若使用文献机制产生动作代理，它只能作为 prior 的输入特征，不能替代模型学习和 MCTS 规划。正式实验日志必须记录 rollout transition provider；若 rollout 展开调用的是精确 push dynamics 加手工评分，该结果只能归入 `heuristic-search diagnostic`。

## 学习目标

训练数据来自真实环境交互、离线 Boxoban 轨迹、短 rollout 标签或诊断求解器生成的候选动作后果。核心损失包括：

```text
L =
  L_stock_transition
  + L_flow_balance
  + L_progress_reward
  + L_terminal_success
  + L_risk_calibration
  + L_uncertainty_calibration
  + L_structure_regularization
  + L_aliasing_diagnostic
  + L_planning_consistency_aux
```

其中：

1. `L_stock_transition` 使用下一真实状态编码的 stop-gradient target，或使用显式可观测代理标签监督核心存量；
2. `L_flow_balance` 约束 `x_{t+1} - x_t` 与 `B f_t` 一致，并保持非负、稀疏和门控约束；
3. `L_progress_reward` 预测目标进展和环境奖励，避免只依赖稀疏 solved reward；
4. `L_terminal_success` 预测完成状态；
5. `L_risk_calibration` 使用分级风险标签：可证明 dead-end、短 rollout failure、heuristic warning；
6. `L_uncertainty_calibration` 使用误差分桶、NLL、ensemble disagreement 或 conformal calibration 之一；
7. `L_structure_regularization` 保持近似守恒、边界、单调或互斥关系，但作为软约束报告 violation；
8. `L_aliasing_diagnostic` 报告同一抽象 `x_t` 或邻近桶内真实后果方差，不作为普通监督损失掩盖问题；
9. `L_planning_consistency_aux` 只作为辅助项，核心训练必须来自真实转移或短 rollout 后果。

动作排序标签可以作为辅助损失，但不能成为唯一训练目标。否则世界模型会退化为启发式打分器。

## 诊断与消融

建议采用以下诊断矩阵：

1. `SD-MCTS full`：完整 stock-flow-feedback 模型；
2. `no-feedback`：移除反馈回路，只保留独立流量；
3. `no-learned-parameter`：只保留手写结构代理，检验是否退化为启发式；
4. `black-box latent MBRL`：无显式存量、流量和反馈；
5. `heuristic-search diagnostic`：旧 best-first 草稿，仅作为诊断对照；
6. `risk-head only`：只使用风险预测，不使用完整 SD 动力学；
7. `flow-without-MCTS`：直接贪心使用流量预测，检验 SD-MCTS 的增益；
8. `oracle abstract transition`：在小规模关卡上用真实抽象转移替代学习模型，分离模型误差和规划误差；
9. `learned-transition-provider audit`：逐 rollout 记录后继是否来自 learned SD transition；
10. `state-aliasing audit`：统计同一抽象状态邻域内真实 single-push 后果方差。

评估指标应分为三类：世界模型指标、规划指标和方法论指标。世界模型指标包括抽象状态预测误差、流量守恒误差、风险校准、不确定性校准和 aliasing 方差。规划指标包括 SD-MCTS 成功率、样本效率、额外 thinking time 效应、预算敏感性和失败类型。方法论指标包括可解释存量曲线、反馈回路触发频率、结构代理与学习参数的消融差异。

## 与启发式搜索的边界

本设计不否认启发式搜索可以求解 Sokoban。边界在于研究目标不同：

1. 启发式搜索通常把领域知识压缩为排序、剪枝或下界；
2. 本设计要求世界模型显式预测抽象状态的 stock-flow-feedback 转移；
3. 启发式字段只能作为观测代理、结构先验、诊断标签或辅助输入；
4. 核心规划接口是 SD-MCTS over learned abstract dynamics；
5. 旧 best-first 草稿只能用于检查结构代理是否有信号，不能作为论文主线基准。

若某个实验在没有学习参数、没有流量、没有反馈、没有 SD-MCTS 的条件下取得较好求解率，该实验应被报告为启发式或符号诊断结果，而不是 SD 世界模型结果。

## 文献支撑边界

本设计使用本地已下载并可读文献作为结构依据。Junghanns 与 Schaeffer 的 deadlock、domain knowledge 和 relevance cuts 文献支持不可逆风险、推箱动作、匹配压力、宏动作和相关性结构；Pereira 等 pattern database 文献支持多箱交互和实例依赖抽象；Lipovetzky、Muise 与 Geffner 的 traps/invariants/dead-ends 文献支持把不可恢复性作为结构风险；Racaniere 等、Guez 等、Orseau 等、Martin 与 Sandholm、Taufeeque 等文献支持把 Sokoban/Boxoban 作为规划表征、额外计算、policy prior 和抽象想象规划的研究场景。

上述文献支持的是结构先验、观测代理、风险标签和规划接口，不支持把这些机制直接写成固定启发函数。Botea、Muller 与 Schaeffer 2002 年 PDF 的本地抽取文本存在严重乱码，因此只作为题录和后续引用链背景，不用其乱码正文支撑具体机制。

## 2026-06-11 修订：专家抽象动力反馈的消费层定位

阶段 3 失败不再简单归因为六个核心存量偏离任务目标。目标一致性 quick 审计显示，六存量对离线 teacher value 的验证集 `R^2=0.8859336361147274`，teacher solved AUC 为 `0.9247685185185185`，说明当前核心 stock 具有任务目标相关性。更直接的问题是规划端如何消费这些 stock、flow 和反馈。

从人类专家视角看，Sokoban 状态不是被压缩为单个静态启发值，而是被抽象为若干可推演的动力机制：箱子是否仍可分配到目标、推箱后是否保留玩家可达性、目标入口和走廊是否被占用、推入瓶颈是否消耗不可逆 slack、当前路径是否在重复状态附近循环。上述机制不应成为替代 SD-MCTS 的 best-first 搜索，而应作为 SD-MCTS 的反馈消费层，用于 reward shaping、action prior、leaf value 和循环反馈。

据此，阶段 3 新增 `expert_abstract_dynamics_feedback_v1`。其构成如下：

1. 状态输入仍以六个核心存量为主体：目标完成、未解决箱子压力、机动性、可达容量、拥塞、不可逆风险；
2. 结构反馈包括匹配压力、合法/可行 push 余量、目标入口阻塞、目标邻接箱子压力、走廊压力、关节点压力和死方邻接压力；
3. 闭环反馈包括已访问真实状态的重复计数，用于惩罚 planned successor 回到已经访问过的状态；
4. action prior 只作为 MCTS prior，不改变 learned provider 给出的抽象 stock transition；
5. expert terminal value 是非 TVA 对照，不替代 search-value TVA，也不使用 best-first teacher 在线打分。

该修订的 contract 是：

```text
transition_provider = learned_stock_flow_checkpoint_v1
planner_feedback = expert_abstract_dynamics_feedback_v1
planner_feedback_changes_transition_provider = false
expert_feedback_teacher_used_online = false
expert_feedback_best_first_used_online = false
```

可复现实验显示，该反馈在 held-out `6` 关卡、budget `24`、depth `4` 下将 `learned_sdmcts` 从旧 stock consumption 的 `0/6` 提升到 `1/6`，并保持 learned provider fraction 为 `1.0`。但 budget `48` 和 budget `200` 下成功率仍为 `0/6`。因此它被记录为“规划消费层方向性改善”，不是稳定阶段 3 通过模型。

后续若继续丰富 SD 设计，应优先处理预算敏感性和长程反馈尺度，而不是直接增加核心存量。进入下一阶段前，新增反馈或 option 至少需要在相同 held-out split 上报告 budget `24/48/200` 三档结果；若只在窄预算上出现非零成功率，则只能作为诊断性中间结果。

### 后续诊断修订

2026-06-11 追加诊断显示，expert terminal value 不应把 deadlocked proxy 作为 root stop 条件。`is_deadlocked` 在当前实现中是风险代理，可能在仍有合法 push 时触发；若在 root 直接停止，会破坏 SD-MCTS 展开。因此修订为只在 solved 或 no-legal root 停止，deadlock 仍作为强负值反馈进入叶值和即时回报。该修订消除了 budget `48` 中的 `planner_returned_no_action`，但没有产生成功率。

随后加入 viability feedback：当 planned successor 的 `viable_push_count` 为 `0` 或 `1` 时，对 reward 和 prior 施加惩罚。该机制是对“保留未来可恢复推箱余量”的专家动力判断建模。实验显示它使轨迹更保守，但 budget `48` 仍为 `0/6`，失败主要为 `no_legal_actions` 与 `max_pushes`。因此，当前模型不应继续只增加负反馈，而应引入正向目标导向结构，例如目标区域进入 option、箱子-目标路线 option 或 learned option prior。
