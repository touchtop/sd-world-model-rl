# Decisions

## 决策 1

**决策**  
将 Hopper 当前主线的研究目标正式拆分为 `model fidelity` 与 `planning usefulness` 两条线。

**背景**  
当前 `SD transition core` 的监督指标、在线 `consistency_loss`、policy replay 误差与闭环行为形态已经显示：动力学拟合更准确，并不自动意味着规划质量同步提升；反之，闭环 return 改善也不自动意味着 `SD` 已被良好拟合。

**选项**  
1. 继续把 `SD` 拟合质量与规划质量视为同一问题。  
2. 在分析与验证口径上显式拆分二者。

**结论**
采用选项 2。

**影响**
后续 Hopper 文档、实验汇报与诊断脚本应分别回答：

1. 模型在 replay 分布上的动力学保真度如何；
2. 模型对动作排序、规划稳定性与样本效率的帮助如何。

## 决策 2

**决策**
若后续继续修改训练方式，优先采用“共享 transition core + 独立在线 SD 监督支路”，而不是设计两个竞争的 dynamics 头。

**背景**
当前在线 RL 联合损失会把 world model 拉向“对当前策略有用”的近似，而不会自动维护 `SD` 在所有关键相位上的短视界精度。单靠继续拉长 RL 预算，未能保证 `consistency_loss` 与关键相位误差持续改善。

**选项**  
1. 保持现有纯联合损失，不额外约束 `SD`。  
2. 构造两个独立 dynamics 头，一个服务监督拟合，一个服务规划。  
3. 保持单一共享 dynamics core，并增加独立的在线 `SD` 监督更新路径。

**结论**  
采用选项 3。

**影响**  
后续若落地实现：

1. planner 与 world model 接口仍保持唯一；
2. `SD` 的短视界监督目标可被显式维护；
3. 训练解释上可清楚区分“联合 RL 目标”与“独立 dynamics fidelity 目标”。

## 决策 3

**决策**  
后续 Hopper 线不再把“继续加长 RL 步数”作为改善 `SD` 精度的首选手段。

**背景**  
当前 `seed3` 接续实验中，`consistency_loss` 在中期后反弹；`50k -> 75k` 的 replay 误差没有表现出稳定持续改善；而闭环行为改善主要表现为“后仰 -> 前倾推进”的恢复，而非 `hop` 形成。

**选项**  
1. 继续主要依赖更长 RL 训练预算。  
2. 优先修改训练目标与诊断策略，再决定是否加长预算。

**结论**  
采用选项 2。

**影响**  
下一阶段重点应放在：

1. 在线 `SD` 监督支路设计，
2. 关键相位加权误差，
3. 同分布 replay 诊断与动作排序诊断，

而不是先投入更大规模闭环训练。

## 决策 4

**决策**  
第一版在线 `SD` 监督支路先采用“单一共享 transition core + 独立优化器 + 1-step 与当前视界 rollout 监督”的最小实现。

**背景**  
方法论上已经确定，不应引入两个相互竞争的 dynamics 头；但若一次性同时改 replay 接口、长视界监督和 planner 接线，变量会过多，难以判断训练异常究竟来自目标设计还是工程实现。

**选项**  
1. 直接实现长视界、多采样路径的完整在线监督版本。  
2. 先保持当前 replay 采样接口不变，仅在共享 `SD transition core` 上追加独立在线监督更新。  
3. 改为双 dynamics 头，分别服务 RL 与监督拟合。

**结论**  
采用选项 2。

**影响**  
当前实现具有以下边界：

1. 监督视界先限制为 `1-step + 3-step`；
2. 关键相位优先覆盖 `landing / takeoff / near_contact / high_omega`；
3. 后续如需 `5-step` 或更长视界，应单独修改 replay 采样接口，而不是在本轮最小实现中混入。

## 决策 5

**决策**  
Hopper 主线保留 `sd_latent_dim=512`，并将 TD-MPC2 主 `consistency_loss` 拆成结构态块与 learned residual latent 块，默认按块等权合成。

**背景**  
显式 17 维 `SD` 结构态不应被要求完整覆盖 Hopper 的所有复杂动力机制，因此需要保留 learned residual latent 作为补偿子空间。但如果继续对完整 512 维 latent 直接做统一均方误差，495 维 residual 块会因维度数量远大于 17 维结构态而主导一致性损失。

**选项**  
1. 将 `sd_latent_dim` 固定为 `17`，完全关闭外层 residual latent。  
2. 保留 `sd_latent_dim=512`，但仍按完整 latent 维度直接做统一均方误差。  
3. 保留 `sd_latent_dim=512`，将结构态块和 residual 块分开计损失，并用显式块级权重合成。

**结论**  
采用选项 3。默认设置为 `sd_latent_consistency_struct_weight=1.0` 与 `sd_latent_consistency_resid_weight=1.0`。

**影响**  
该决策保留 residual latent 对未建模动力机制的补偿能力，同时避免高维 residual 因维度数量自然压过显式结构态。结构态块内部不使用人工固定通道系数，而是通过 `sd_struct_loss_scale_mode=batch_std` 依据 replay batch 目标结构态的逐通道标准差归一化；`sd_struct_loss_scale_floor=0.1` 仅作为数值稳定下限。

## 决策 6

**决策**  
下一版本先实现默认关闭的 planning-aware `SD` 样本权重入口，而不是直接把 `SD` 训练目标改为完整动作排序一致性损失。

**背景**  
当前 `pureenv_sdinit_online_sup` 与上一组 `pureenv_sdinit` 对比显示，平均结构态预测监督未必能提升强 seed 的闭环收益，说明 `SD` 保真目标与 planner 动作排序目标之间仍存在错配。但完整动作排序一致性损失需要额外构造动作 bank、真实短 rollout 标签或离线动作优劣诊断，工程变量较多。

**选项**  
1. 继续只优化平均结构态预测误差。  
2. 先根据 TD 目标或奖励目标对在线 `SD` 监督样本进行均值归一化重加权。  
3. 立即实现完整 action-ranking consistency loss。

**结论**  
采用选项 2。该入口默认关闭，仅在显式实验脚本中启用；启用后，权重按 batch 内 planning signal 均值归一化，并用 `sd_online_planning_weight_max` 截断。

**影响**  
该决策提供了低风险的第一步验证：若 planning-weighted `SD` 监督改善样本效率，则说明目标错配至少部分来自样本重要性；若无明显改善，则后续应转向完整动作排序一致性目标，而不是继续提高 `sd_online_supervision_lr`。

## 决策 7

**决策**
将下一阶段方法论术语暂定为 `Decision-Oriented System-Dynamics-Guided World Model`，中文为“面向决策的系统动力学引导世界模型”。

**背景**
当前实验提示，平均结构态预测误差下降并不必然带来更好的闭环规划效果。这说明 `SD` 世界模型的核心目标不应是完备复刻环境动力学，而应是形成足以支撑正确动作选择的结构化抽象。这里的 `SD` 明确指系统动力学（System Dynamics），其设计指导来自系统工程和系统仿真，而不是结构动力学。该思想与人类智能中的世界理解具有相似性：有效表征通常是行动相关、容量受限且带有归纳偏置的，而不是对世界的无损副本。

**选项**
1. 使用“SD 偏置世界模型”作为主术语。
2. 使用 `SD-biased world model` 或“SD 归纳偏置世界模型”。
3. 使用 `Decision-Oriented System-Dynamics-Guided World Model` 或“面向决策的系统动力学引导世界模型”。

**结论**
采用选项 3 作为主术语。选项 2 可作为解释性表述，用于说明 `SD` 在模型中提供系统动力学归纳偏置；选项 1 不作为主术语，以避免“偏置”被误读为有害统计偏差。

**影响**
后续论文写作、方法文档和实验解释应围绕“世界模型服务决策”展开。评价指标除 replay 分布上的状态预测误差外，还应强调动作排序质量、关键相位判断、规划稳定性和闭环样本效率。

## 决策 8

**决策**
将“`SD` 刻画精度何时足以支撑决策”的数学与统计论证列为未来方法论工作，而不纳入当前实验主线。

**背景**
在面向决策的系统动力学引导世界模型中，关键问题不再是对真实环境做无损复刻，而是判断模型误差是否已经小到不会改变关键动作排序、价值排序或可行性判断。该问题涉及状态预测误差到价值误差的传导、价值误差到动作选择的传导，以及不同误差在 replay 分布、规划 imagined 分布和真实闭环访问分布下的统计含义。

**选项**
1. 当前即尝试建立完整理论误差界。
2. 当前先训练和验证规划支撑能力，将理论充分性证明作为未来工作。
3. 完全不讨论 `SD` 刻画精度与决策支撑能力之间的数学关系。

**结论**
采用选项 2。当前主线继续聚焦 planning-aware 训练、动作排序一致性、闭环样本效率和关键相位行为改善。

**影响**
该问题将在后续论文或扩展工作中作为方法论创新方向展开。可能的理论目标包括：给出 `SD` 模型误差不改变最优动作排序的充分条件，刻画关键相位误差对规划价值的敏感性，或建立任务相关模型误差与闭环性能退化之间的统计界。

## 决策 9

**决策**
修复 `SDWorldModel` 与 TD-MPC2 的在线监督接口，使共享 `SD transition core` 的优化器归属唯一化，并取消趋势保护中的绝对最小修正下限。

**背景**
静态审查指出，启用 `sd_online_supervision_enabled=true` 时，`model._dynamics` 同时被 TD-MPC2 主优化器和 `sd_optim` 管理，会导致同一参数被两个 Adam 优化器用不同动量状态更新。审查还指出，趋势保护维度中的固定 `0.02` 最小修正下限对 `ROOTZ / ZMIN` 等米制维度存在单位错配风险，并可能在小信号阶段制造虚假漂移。另有一处 learned reward 模式未统一使用 `sd_env_reward_scale`。

**选项**
1. 保持原实现，仅观察训练曲线是否稳定。
2. 将 online SD 监督损失并入 TD-MPC2 主损失，使用单一优化器。
3. 启用 online supervision 时，将 `_dynamics` 从主优化器移除，使其只由 `sd_optim` 更新；同时移除趋势保护绝对下限，并修复 learned reward 目标尺度。

**结论**
采用选项 3。该方案保留“主 RL 目标 + 独立 SD 监督目标”的方法分工，同时使参数、目标和优化器归属保持一致。趋势保护仍保留相对比例约束，但默认不再强制绝对最小改变量。复审后进一步明确：启用 online supervision 时，主 consistency 路径默认 detach `_dynamics` 输出，使 `consistency_loss` 不再训练 `SD core`，而只训练外层 residual heads 与 TD-MPC2 下游头；`SD core` 由独立在线 `SD` 监督支路训练。

**影响**
后续启用 online supervision 的实验不再存在 `_dynamics` 双 Adam 更新问题，也不再为 `_dynamics` 计算随后被丢弃的主 consistency 梯度。旧 full checkpoint 在恢复时可能因主优化器参数组变化而无法完全载入 optimizer state；当前恢复逻辑已改为跳过不兼容的 optimizer state 并继续恢复模型、buffer、计数器和兼容优化器，告警中记录旧参数组数与新参数组数。外层结构态 residual 仍作为必要补偿保留，但 residual 尺度已配置化，并新增默认关闭的 `sd_struct_residual_penalty_coef` 与训练日志指标 `struct_residual_activity` 作为后续消融入口。

## 决策 10

**决策**
Sokoban 新研究线采用新增独立包 `src/sokoban/` 与新增实验目录 `experiments/sokoban_sd_world_model/`，不修改既有 Beer Game、Cartpole、Hopper 模型或实验入口。

**背景**
Sokoban 的任务动力机制与连续控制任务不同。其关键在于可达域、推箱宏动作、箱子-目标匹配、dead-square 与不可逆死锁，而不是直接观测网格的逐像素转移。与此同时，当前仓库已有 NeurIPS 论文相关的多条实验线，必须保证既有成果仍可按原路径复现。

**选项**
1. 直接修改现有 `src/env`、`src/world_model` 和 `src/planning` 包入口，将 Sokoban 接入原有模块导出链。
2. 在旧包路径下新增 Sokoban 文件，但不修改入口。
3. 新增独立 `src/sokoban/` 包，并在该包内组织环境、SD 模型和搜索器。

**结论**
采用选项 3。

**影响**
Sokoban 研究线与既有实验线隔离，避免旧 `__init__` 导入链和可选依赖影响新实验。模型内部仍采用继承机制：`SokobanPushDynamicsModel` 作为基础推箱动力学类，`SokobanGoalDirectedSDModel` 作为派生类加入目标导向 SD 机制。该结构保留了后续将 Sokoban adapter 接入 MCTS 或 TD-MPC2 风格接口的空间。

## 决策 11

**决策**
Sokoban 后续主线改为“文献驱动的 stock-flow-feedback SD 世界模型与替换设计”，不继续把旧 `src/sokoban/search.py` 的 best-first 求解器作为主要交付对象。

**背景**
首轮 50 关卡结果显示，当前玩家可达域、reverse-pull 距离、匹配下界和死锁剪枝可以显著改变搜索行为。但文献矩阵显示，经典 Sokoban 求解依赖 deadlock tables、多箱 pattern/PDB、tunnel/goal macros、relevance cuts、goal-room 结构、policy-guided heuristic search 等机制。当前草稿只覆盖其中一部分，直接围绕 solved rate 继续调参会把诊断性原型误认为完整求解器路线。

**选项**
1. 继续沿用当前 best-first search，并围绕启发式权重和剪枝策略优化求解率。
2. 删除当前代码，重新从完整经典求解器实现开始。
3. 保留当前代码为诊断草稿，先建立文献矩阵、机制输出 schema 和替换设计，再决定搜索消费端。

**结论**
采用选项 3。

**影响**
后续文档和实验报告应把 `eval_medium_train000_50.json` 标注为 smoke/diagnostic baseline，而不是完整 Boxoban benchmark。新增机制应优先落到存量、流量、反馈、风险和不确定性接口中，再由 SD-MCTS 在学习到的抽象动力学上消费。best-first search 只保留为启发式诊断对照。

## 决策 12

**决策**
Botea、Muller 与 Schaeffer 2002 年 PDF 的本地乱码抽取文本不作为具体设计证据。

**背景**
该 PDF 由旧 Ghostscript 生成，`pdftotext -layout -enc UTF-8` 与 `pdftotext -raw -enc UTF-8` 均未恢复可读正文。文件题录和后续论文引用链可核验，但当前本地正文无法支持逐条机制摘录。

**选项**
1. 继续从乱码文本中推断机制内容。
2. 只保留题录和后续可读论文复述中可核验的信息。
3. 完全移除该文献。

**结论**
采用选项 2。

**影响**
文献矩阵中该文献被标为 `B` 级证据。后续若获得可读版本，可新增条目补充机制依据；在此之前，不用乱码正文支撑具体实现选择。

## 决策 13

**决策**
Sokoban/Boxoban SD 主线采用“结构已知、部分参数未知”的可学习 stock-flow-feedback 世界模型，并以 SD-MCTS 作为规划基准。旧启发式搜索、匹配下界、dead-square、macro 和 relevance 等机制仅作为结构先验、观测代理、流量门控、风险标签或诊断对照。

**背景**
此前文档虽然已将 best-first 求解器降级为诊断草稿，但仍使用了 `policy_score`、`heuristic_score`、`risk_score` 和 action-ranking schema 等表述，容易把 SD 世界模型退化为静态启发函数。研究目标实际上是刻画复杂系统面向任务目标状态转移的内生机制，而不是证明启发式搜索可以求解 Sokoban。

**选项**
1. 继续沿用机制字段加动作评分的接口，由 best-first、MCTS 或神经模型共同消费。
2. 完全删除旧 Sokoban 代码与评测结果，只保留概念设计。
3. 保留旧代码为诊断性原型，新增 stock-flow-feedback 设计，将正式路线限定为可学习 SD 世界模型和 SD-MCTS。

**结论**
采用选项 3。

**影响**
后续实现必须至少输出存量、流量、反馈、不确定性和下一抽象状态；SD-MCTS 必须调用学习到的抽象动力学展开节点。任何只输出动作排序分数、没有学习流量、没有反馈回路或没有 SD-MCTS 的版本，均只能作为启发式诊断结果报告。

## 决策 14

**决策**
Sokoban/Boxoban 首版 SD 世界模型采用六个核心存量、已知 incidence matrix、三类核心反馈和合法 single-push option。`GoalRoomPressureStock`、`PlanProgressStock` 与 `StructuralSlackStock` 暂不作为首版核心存量。

**背景**
细粒度结构复审指出，九存量设计虽然覆盖面较广，但 `P_t` 与 `C_t`、`L_t` 与规划器内部搜索进度、`S_t` 与 `M_t/R_t/C_t/I_t` 存在较强重叠。若首版直接实现九存量，会增加不可辨识、标签循环和启发式退化风险。流量层面也需要明确 `B f_t` 的 incidence matrix，避免多个流量同时改变同一存量而产生双计。

**选项**
1. 保留九个存量并同时实现全部反馈。
2. 首版收缩为六个核心存量，`P_t/L_t/S_t` 降级为派生特征、辅助头或第二阶段扩展。
3. 进一步压缩为单一启发式或风险分数。

**结论**
采用选项 2。首版核心存量为 `G_t, U_t, M_t, R_t, C_t, I_t`；核心反馈为目标压力反馈、拥塞-机动性-风险反馈和不确定性反馈；动作集合限定为合法 single-push option。状态转移写为 `x_{t+1}=clip_X(x_t+B f_theta(...))`，其中 `f_theta` 为非负流量，`B` 为已知结构矩阵。

**影响**
后续实现和论文叙述应以六核心存量作为首版主线。goal-room、macro、relevance、path-channel 和 structural slack 机制可以继续作为结构代理、诊断项、辅助预测头或第二阶段扩展，但不得被写成首版核心状态转移的必要组成。正式 SD-MCTS 还必须记录 learned transition provider、raw-state handle 或 belief support、风险标签分级和 state aliasing 诊断，以防模型退回启发式搜索或不可执行抽象规划。

## 决策 15

**决策**
冻结 Sokoban/Boxoban 当前诊断性最小 learned stock-flow baseline，并暂缓将 feedback gate、feedback target 和 `reach_congestion` 加权损失纳入默认模型。

**背景**
`academic-research-suite` 审查指出，当前最可靠、可审计、可复现的模型不是最全面的 feedback 版本，而是六核心存量、固定 incidence matrix、非负 learned flow、合法 single-push action features、level-holdout split、uniform stock loss 和两步 rollout supervision 组成的 `rollout_sup_base`。复测显示该基线三种子主指标与旧结果精确一致，复现判定为 `REPRODUCIBLE`。同时，审计显示验证集上 clipping 修正较频繁，stock-only 与 stock-action alias bucket 仍存在明显 next-delta 方差，说明继续堆叠复杂损失或反馈结构会掩盖当前更基础的抽象状态与流量分解问题。

**选项**
1. 以 feedback-gated 模型作为默认基线，并继续调 feedback target 和损失权重。
2. 冻结 `rollout_sup_base` 为诊断性最小基线，只将 feedback gate、feedback target 和 loss reweighting 作为单因素消融。
3. 立即接入 SD-MCTS，以规划结果反向判断当前世界模型是否足够。

**结论**
采用选项 2。冻结基线的 `baseline_id` 为 `sokoban_six_stock_nonnegative_flow_rollout_sup_base_v1`。该基线仍是 diagnostic learned stock-flow baseline，不是 formal SD-MCTS 结果；正式 SD-MCTS 前必须完成 learned transition provider audit、stock proxy 审计、flow decomposition 审计、clipping 审计和 aliasing bucket 审计。

**影响**
后续实验应以该冻结基线作为对照。任何新增机制只有在同一 level-holdout、同一 rollout 指标、同一 provider 边界和同一审计字段下稳定优于该基线，才可进入默认模型。当前 feedback gate、feedback target、`reach_congestion` 加权损失、macro 扩展和不确定性反馈均不得直接写作主线默认改进。

## 决策 16

**决策**
Sokoban/Boxoban 下一阶段不直接接入 learned-transition-provider SD-MCTS，而是先围绕冻结基线执行单因素 stock-flow 诊断修订。优先修订顺序为：成对相反流量共激活约束或 paired/net-flow 参数化、`reachable_capacity` 与 `congestion` 代理修订、alias-aware 辅助特征。

**背景**
增强审计版冻结基线与旧 `rollout_sup_base` 主指标精确一致，说明新增审计没有改变训练路径。完整 stock-flow 诊断覆盖 `30` 个关卡、`566` 个抽象状态和 `3435` 条 transition，显示 exact diagnostic flow labels 在固定 incidence matrix 下 `max_l1=0.0`，但 learned prediction 侧存在明显问题：valid any-stock clipped fraction 均值为 `0.9277460177739462`；stock-action alias max next-delta variance 最高为 `0.11775519031141866`；六组成对相反流量预测共激活比例均为 `1.0`。同时，`reachable_capacity` 与 `congestion` 的 exact delta nonzero fraction 分别为 `0.37147016011644834` 和 `0.6954876273653566`，说明二者是有效变化维度，但当前代理仍较粗。

**选项**
1. 立即接入 SD-MCTS，用规划表现倒逼判断当前世界模型是否可用。
2. 继续堆叠 feedback gate、feedback target 或 `reach_congestion` loss 权重。
3. 保持冻结基线为对照，先做单因素诊断修订，并要求每个修订通过同一 provider audit、level-holdout 和两步 rollout 指标比较。

**结论**
采用选项 3。

**影响**
formal SD-MCTS 继续暂缓。下一轮模型修订必须说明它针对的是 clipping、相反流共激活、stock proxy 还是 aliasing 中的哪一个问题；不得一次性引入多个复杂项。每个候选修订均需生成可复现 JSON、与冻结基线做 summary 比较，并更新方法、复现和研究日志。只有当审计指标和 rollout 指标均不退化，且 provider audit 仍清楚区分 diagnostic provider 与 learned provider 时，才允许进入 SD-MCTS 接口实现。

## 决策 17

**决策**
阶段 1 第一轮 `paired_net` 修订不作为达标模型；保留成对互斥思想，但下一轮必须加入流量幅度或 raw-next 边界约束。

**背景**
`paired_net` 将每对相反流量由一个有符号净流量展开为两个非负流量，从结构上消除了相反流共激活。三种子实验显示，flow pair predicted coactivation mean 和 max 均为 `0.0`，valid flow MAE 从冻结基线的 `0.04738566527764002` 降至 `0.03719972570737203`。但是，valid rollout stock MAE mean 从 `0.048756017039219536` 升至 `0.05166300448278586`，超过阶段 1 门限 `0.05119`；valid rollout step-2 stock MAE 从 `0.056363618622223534` 升至 `0.05988542859752973`，超过门限 `0.05918`；valid mean abs clip correction 从 `0.017344274868567783` 升至 `0.028677965203921`，未达到 `<=0.01214` 的要求。

**选项**
1. 将 `paired_net` 纳入默认模型，因为它解决了相反流共激活。
2. 完全放弃成对互斥，回到独立 `softplus` 流量。
3. 将 `paired_net` 记录为阶段 1 部分通过结果，下一轮在成对互斥基础上增加幅度控制或边界约束。

**结论**
采用选项 3。

**影响**
当前不进入阶段 2，也不接入 learned-transition-provider SD-MCTS。下一轮阶段 1 的候选实现必须同时满足三个方向：保留相反流共激活为零或接近零，恢复 rollout 非退化门限，并降低 clip correction。若下一轮仍只改善 flow MAE 或共激活，但无法降低 clipping，则不能作为主线推进。

## 决策 18

**决策**
阶段 1 通过模型采用 `paired_net + capacity projection + final-step rollout calibration`，作为进入阶段 2 的 learned stock-flow 世界模型候选；暂不再继续阶段 1 的 rollout 权重搜索。

**背景**
第一轮 `paired_net` 已消除相反流共激活，但 clipping 和两步 rollout 未达标。后续候选显示，容量投影 mask `[1,1,1,0,0,1]` 可将 valid mean abs clip correction 降至接近零，并保持相反流互斥，但普通 `rollout_final_step_weight=1.0` 的三种子 step-2 均值仍为 `0.05938937390844027`，略高于门限 `0.05918`。逐存量 final-step 权重 `[1,1,1.2,1.6,1.6,1]` 与延长到 `300` epoch 均未改善关键 seed。最终，保留容量投影并将 `rollout_final_step_weight` 设为 `1.2` 后，三种子 gate 判定为 `PASS`。

**选项**
1. 将普通 `paired_net + capacity projection` 作为阶段 1 模型，并放宽 step-2 门限。
2. 继续搜索逐存量 rollout 权重、训练 epoch 或 `reach_congestion` 权重。
3. 采用 `paired_net + capacity_projection_stock_mask=1,1,1,0,0,1 + rollout_final_step_weight=1.2`，固定为阶段 1 通过模型。

**结论**
采用选项 3。阶段 1 gate 文件 [flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json) 显示 `verdict=PASS`、`failed_checks=[]`。观测指标为：valid rollout stock MAE mean `0.05017088601986567`，valid rollout step-2 stock MAE `0.058157226691643395`，valid mean abs clip correction `9.097307914392635e-07`，valid any-stock clipped fraction `0.0005817335719863573`，flow pair predicted coactivation mean/max `0.0/0.0`，incidence balance max L1 `0.0`。

**影响**
阶段 1 已完成，不应继续围绕同一 gate 做额外权重搜索。下一阶段可以推进 learned transition provider 与 SD-MCTS 接口，但必须保留 provider audit，明确区分 exact diagnostic labels、learned transition rollout 与 heuristic diagnostic path。当前通过结果仍是 learned stock-flow world-model gate，不是 formal SD-MCTS 规划结果。

## 决策 19

**决策**
阶段 2 采用最小 learned-transition-provider SD-MCTS 接口审计作为通过阶段 1 后的下一交付单元。该阶段接受 raw Sokoban state handle 作为合法动作支持，但 SD-MCTS 节点的抽象 stock transition、reward 和 value 必须来自 learned stock-flow checkpoint。

**背景**
阶段 1 已通过 learned stock-flow gate，但旧结果文件只保存指标，不保存可加载权重。如果直接进入规划评测，会无法审计 rollout transition 是否真正来自 learned model，也容易退回 exact single-push diagnostic provider 或启发式搜索。另一方面，六维抽象 stock 不包含箱子位置和玩家位置，无法独立枚举下一层合法推箱动作；因此 formal 接口需要显式区分 raw-state handle 与 learned abstract transition provider。

**选项**
1. 继续只使用阶段 1 JSON 指标，不保存 checkpoint，手工复述模型效果。
2. 直接实现完整 Boxoban 求解评测，将 raw exact dynamics 与启发式评分混合使用。
3. 先补可加载 checkpoint，并新增 learned provider SD-MCTS smoke：raw state 只用于动作枚举，抽象转移和规划评分均来自 learned provider，同时记录 provider audit。

**结论**
采用选项 3。`train_sokoban_flow_smoke.py` 新增默认关闭的 `--checkpoint-out`；`audit_stage2_learned_provider_sdmcts.py` 加载 checkpoint 并以 `learned_stock_flow_checkpoint_v1` 作为唯一 MCTS abstract transition provider。阶段 2 seed0 smoke 在 checkpoint 记录的 held-out levels 上得到 `verdict=PASS`、`failed_checks=[]`、learned provider fraction `1.0`、`42` 个 root、`793` 条 MCTS 展开 transition、depth-1 stock MAE `0.05060150917204054`、depth-2 stock MAE `0.056905325794544735`、incidence balance max L1 `1.1748737760441941e-07`。

**影响**
阶段 2 已确认 learned stock-flow 模型可以被 SD-MCTS 最小接口消费。后续可以进入正式规划实验设计，但不得把当前 smoke 解释为 Boxoban 求解 benchmark。下一阶段需要在相同 provider audit 下加入多种子、多预算、成功率、失败类型、不确定性/风险校准、no-learned-provider 与 heuristic diagnostic 消融。

## 决策 20

**决策**
阶段 3 规划 smoke 记录为有效负结果：learned provider 接线通过，但当前六 stock 抽象 reward/value 与短视界 SD-MCTS 规划层不足以解出 held-out Sokoban 关卡。下一轮不继续单纯提高 simulation 数，而是优先修订规划目标、终端价值、风险/循环惩罚和长程 value bootstrap。

**背景**
阶段 2 只验证了 learned transition provider 能进入 MCTS 展开。阶段 3 首次执行闭环规划评测，比较 `learned_sdmcts`、`oracle_abstract_sdmcts`、`zero_flow_sdmcts`、greedy heuristic diagnostic 和 best-first heuristic diagnostic。小预算 `16/48` simulations、深度 `3`、最多 `30` push 下，SD-MCTS 相关方法成功率均为 `0.0`。较大预算 `200` simulations、深度 `5`、最多 `60` push 后，`learned_sdmcts`、`oracle_abstract_sdmcts` 与 `zero_flow_sdmcts` 仍均为 `0.0`。同时，旧 best-first diagnostic 在同一 `6` 个 held-out levels、`5000` expansion 内成功率为 `1.0`。

**选项**
1. 继续提高 MCTS simulation 数和深度，尝试用搜索预算弥补当前 reward/value 缺陷。
2. 将 best-first heuristic diagnostic 作为主结果，弱化 learned SD-MCTS 的失败。
3. 将本轮记录为规划层负结果：provider 接口保留，下一轮集中修订 SD-MCTS 的目标函数、终端/死锁价值、循环惩罚、长程 bootstrap 和动作抽象。

**结论**
采用选项 3。oracle abstract SD-MCTS 也失败，说明仅有更准确的一步/短视界 stock transition 不足以产生 Sokoban 求解行为；当前主要瓶颈在规划消费层，而不只是 learned provider 误差。best-first diagnostic 仅作为关卡可解性和启发式上界参照，不能替代 SD-MCTS 主线。

**影响**
后续阶段应在保留 provider audit 的前提下，设计 planning-layer 修订实验。可进入下一轮的必要改动包括：显式 solved terminal reward/value、dead-end 与 revisit penalty、风险校准、长视界 value bootstrap、goal/macro/relevance 作为动作抽象或反馈输入的受控接入，以及 learned/no-learned/oracle/heuristic 的同口径对照。仅增加仿真预算而不修改规划目标，不作为优先路线。

## 决策 21

**决策**
将 Sokoban TVA v1 冻结为“最小可审计终端价值接入基线”，但不将其判定为阶段 3 通过模型。下一轮 TVA 研究应转向 search-value alignment、on-policy failed-state relabeling、revisit-aware target 和动作选择消融，而不是继续只扩大 MCTS budget。

**背景**
用户要求复核 NeurIPS/Beer Game 中 TVA 的设计，并为当前 Sokoban SD-MCTS 实现 TVA 后重新测试。NeurIPS 材料中的 TVA 公式以 `V_psi(s_leaf)` 补偿截断树搜索，并可由 search-value target 与 1-step TD target 共同训练；Beer Game 简化实现中，价值网络为 MLP，输入状态，输出标量值，MCTS leaf evaluation 调用该网络，训练可使用 Monte Carlo return 或 TD target。当前 Sokoban 阶段 3 负结果显示，原脚本只使用固定 `stock_value(node.stock)` 作为叶值，不具备可学习 terminal value。

本轮新增 `sokoban_tva.py`、`train_sokoban_tva.py`，并在 `evaluate_stage3_sdmcts_planning.py` 中加入 `--terminal-value-mode stock|zero|tva` 与 `--tva-checkpoint`。TVA v1 复用阶段 2 checkpoint 的 train/valid level split，只在 train levels 上拟合，不在线使用 teacher；输入为六个 SD stock、反馈派生量和高层结构标量，输出为反归一化剩余价值。训练结果显示 valid MAE 为 `1.1535100936889648`，valid corr 为 `0.9297597211519081`，但 `solved` 类平均预测低于目标，`teacher_budget_unsolved` 类被预测为正值。

阶段 3 TVA 复测中，learned provider usage gate 和 terminal value usage gate 均通过：小预算 smoke 中 TVA 调用 `36944` 次，大预算 `200`、深度 `5` 中 TVA 调用 `76600` 次，且 learned provider fraction 均为 `1.0`。但是 `learned_sdmcts` 在 held-out `6` 个关卡上成功率仍为 `0.0`；大预算下 `oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 也仍为 `0.0`。失败类型主要为 `state_revisit_limit` 与 `no_legal_actions`。

**选项**
1. 将 TVA v1 作为阶段 3 通过模型，继续用更大 budget 尝试获得成功率。
2. 回退到旧固定 `stock_value`，暂缓 TVA 接入。
3. 冻结 TVA v1 为最小可审计接入基线，保留其代码和复测结果，同时承认它没有通过阶段 3，并将下一轮集中在 TVA target 与 planning consumption 修订。

**结论**
采用选项 3。TVA v1 证明了价值网络已按 SD-MCTS 叶节点接口接入，且没有破坏 learned transition provider 基准；但它没有解决闭环求解问题，不能作为成功规划结果写入主线。

**影响**
后续研究应以 TVA v1 为对照，新增改动必须报告 `terminal_value_usage_gate`、learned provider fraction、成功率、失败类型和叶节点类型分布。优先方向为：使用 MCTS search statistics 构造 `V_search` 对齐目标；把当前 failed closed-loop trajectories 中的状态重新标注为训练样本；将重复状态和 no-legal 失败纳入 value target；比较 root action selection by visits 与 by value；评估 stock-only TVA 与 stock+structural TVA 的抽象充分性。诊断 best-first teacher 仍只能用于离线标注或可解性参照，不能在线替代 SD-MCTS。

## 决策 22

**决策**
将 TVA v2 failed-relabel 记录为“失败形态诊断有效、阶段 3 成功率未通过”的负结果；不继续围绕同一失败重标注方案做无门限调权重。下一轮若继续推进 TVA，应增加正例保护或 search-value target，而不是单纯加重 closed-loop 失败负样本。

**背景**
TVA v2 在 TVA v1 基础上加入 train-level failed closed-loop state relabeling、样本权重和 `root_action_selection=visits|value` 消融。训练集增加 `600` 个 closed-loop failed rollout 样本，来源包括 `state_revisit_limit`、`no_legal_actions`、`planner_returned_no_action` 和 `max_pushes`。训练指标显示 v2 对 proven dead-end 与 teacher budget unsolved 的负值区分更强，但 `solved` 类平均预测从正目标附近被压到负值：valid solved target mean 为 `4.7114176750183105`，pred mean 为 `-1.2775442600250244`。

阶段 3 复测在 held-out `6` 个关卡、`learned_sdmcts`、budget `200`、深度 `5`、seed `0` 下执行。`visits` 与 `value` 两种 root action selection 均通过 learned provider usage gate 和 TVA usage gate，TVA 调用均为 `17600` 次，provider id 均为 `sokoban_tva_value_v2_failed_relabel`。但是两条复测成功率均为 `0.0`，平均 push 数均为 `14.166666666666666`，失败计数均为 `planner_returned_no_action=3`、`state_revisit_limit=3`。

**选项**
1. 继续提高 closed-loop failed rollout 样本权重或扩大失败增强样本，尝试进一步减少循环。
2. 将 v2 判定为阶段 3 通过，因为它减少了平均 push 数并改变了失败类型。
3. 将 v2 判定为负结果：它提供了失败形态诊断，但没有通过成功率门限，且损害 solved 类价值估计；下一步必须补正例保护或引入 search-value alignment。

**结论**
采用选项 3。v2 的价值在于证明 failed closed-loop relabeling 会改变 SD-MCTS 的 closed-loop 行为，但当前实现把可解状态也压低，导致 planning 不再有效推进到解。root action selection by value 没有相对 visits 带来差异，说明当前瓶颈不只是根动作 tie-breaking。

**影响**
后续 TVA 路线应保留 v1 和 v2 作为对照。若继续训练 v3，必须显式满足两个门限：一是 valid solved 类预测均值不得被压为负值，且 solved MAE 相对 v2 明显下降；二是阶段 3 held-out `learned_sdmcts` 至少在相同 budget/depth 下出现非零成功率或在 oracle abstract 对照中稳定提升。若只改变失败类型或缩短失败轨迹而成功率仍为 `0.0`，不得判定为阶段 3 通过。

## 决策 23

**决策**
将后续正式 TVA 路线修订为 `sokoban_search_value_tva_v1`：训练目标必须由 SD-MCTS 根节点搜索统计构造 `V_search`，当前 v1/v2 仅保留为 terminal value proxy 对照，不再称为严格论文语义的 TVA。

**背景**
用户指出此前要求参照论文和已成功实验代码设计 TVA，但实现只复用了 MCTS 叶节点 value interface，没有按论文 TVA 的逻辑语义训练。复核 NeurIPS 论文可知，TVA 的核心不是“存在一个叶节点价值网络”，而是通过 MCTS root statistics 构造 `V_search(s)=sum_a pi_mcts(a|s) Q_H(s,a)`，并以 search alignment loss 训练 `V_psi`，可再加入 1-step TD target 与 target network。当前 `sokoban_tva_value_v1` 使用 best-first diagnostic teacher 的静态剩余价值，`sokoban_tva_value_v2_failed_relabel` 使用 closed-loop failure relabel 负样本，二者只与论文 TVA 的接口位置一致，不与其训练语义严格一致。

本修订方案已提交 `academic-research-suite` 审查，审查记录见 [sokoban_tva_search_value_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_tva_search_value_design_review.md)。审查指出，若不实现 `V_search=sum pi_mcts Q_H`，则仍不是论文 TVA；若 best-first teacher 进入在线 action choice 或 root scoring，则主线会退化为启发式搜索。审查同时要求在实现前增加 root child `q_value` 的 depth-1 语义审计。

**选项**
1. 继续在 v2 failed-relabel 上调样本权重或扩充失败状态样本。
2. 将 v1/v2 视作 TVA 的充分实现，只补充文档解释。
3. 停止在错误语义上调参，将 v1/v2 降级为 proxy 对照，新增 search-value TVA，并以 MCTS root statistics 作为主训练目标。

**结论**
采用选项 3。下一步实现必须新增 search stats collection、`V_search` target builder、可选 TD target network 和 checkpoint contract。若首轮设 `lambda_td=0`，必须标注为 TD 消融；完整 TVA 需要记录 TD 项与 target network 策略。

**影响**
后续阶段 3 通过判定增加硬 gate：checkpoint 必须记录 `target_source=sd_mcts_root_statistics`、`mcts_root_statistics_used=true`、`teacher_used_online=false`、`transition_provider_modified=false`；target collection 和 closed-loop eval 中 learned provider fraction 必须为 `1.0`；depth-1 `Q_H` 语义审计必须通过；finite `V_search` fraction、root child coverage、solved/dead-end/budget-unsolved 分组指标、held-out success rate、失败类型和 oracle abstract 对照必须同步报告。若 held-out `learned_sdmcts` 成功率仍为 `0.0`，不能判定阶段 3 通过。

## 决策 24

**决策**
`sokoban_search_value_tva_v1` 记录为“严格 TVA 语义实现已完成但阶段 3 未通过”的负结果；不继续围绕 positive search target 样本权重做无门限搜索。

**背景**
按决策 23 实现 [train_sokoban_search_value_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py)。该脚本从 SD-MCTS root statistics 构造 `V_search=sum_a pi_mcts(a|s)Q_H(s,a)`，checkpoint contract 记录 `target_source=sd_mcts_root_statistics`、`mcts_root_statistics_used=true`、`teacher_used_online=false`、`teacher_used_for_training_target=false`、`transition_provider_modified=false`。depth-1 `Q_H` 语义审计覆盖 `80` 个状态，最大误差为 `1.7763568394002505e-15`，判定为 `PASS`。

首轮 `lambda_td=0` search-alignment 训练得到 `873` 个 root 样本，其中 train `728`、valid `145`，finite `V_search` fraction 为 `0.5014359563469271`。valid MAE 为 `1.3938531875610352`，valid corr 为 `0.5508857019779951`。但 valid positive search target 组目标均值为 `0.8352776765823364`，预测均值为 `-0.8174604773521423`，不满足正例保护 gate。随后进行只基于 `V_search>0` 的正例权重消融：权重 `4.0` 时 valid positive pred mean 为 `-0.5202222466468811`，权重 `10.0` 时为 `-0.1770382821559906`，仍未转正。

阶段 3 复测显示，`sokoban_search_value_tva_v1` 与 `sokoban_search_value_tva_v1_posw10` 均通过 provider usage gate 和 terminal value usage gate，但 held-out `6` 关卡上成功率仍为 `0.0`。posw10 版本的 `root_action_selection=value` 消融也仍为 `0.0`，说明失败不是根动作选择规则单独造成的。

**选项**
1. 继续提高 positive target 样本权重，直到验证正例均值转正。
2. 将 search-value TVA 判定为阶段 3 通过，因为它满足了 TVA 语义和 usage gate。
3. 记录为语义合规但规划负结果，停止无门限权重搜索，下一步转向 target 质量、state aliasing、action abstraction 和 TD target network。

**结论**
采用选项 3。严格 TVA 语义的实现已经完成，原先“只用接口不按逻辑语义”的问题已被修复；但该修复没有带来 held-out 求解成功率，也没有通过正例保护 gate。因此不能将其判定为阶段 3 通过。

**影响**
后续工作不应继续只调 positive-target 权重。下一步应审查 `V_search` target 的分布与质量，特别是 positive target 稀疏性、valid positive target 泛化、root state alias bucket 内 target 方差，以及 oracle abstract 仍失败的规划层原因。若继续推进 TVA 完整版，应加入论文中的 TD target 与 target network，并把 action abstraction、goal-room、macro、relevance 作为 SD feedback 或 option 的受控扩展，而不是作为启发式在线替代。

## 决策 25

**决策**
当前六存量 SD 抽象通过 quick 目标一致性审计，暂不将阶段 3 失败归因为“SD 存量完全偏离任务目标”。后续修订优先集中在长程规划消费、动作抽象、终端价值和 search-value 泛化；goal-room、macro、relevance 等结构只能作为受控 SD feedback、option 或 probe 接入，不得作为在线启发式替代。

**背景**
用户要求用 skill 审核当前 SD 刻画元素与任务目标的一致性，并明确任务世界模型应针对目标抽象。本轮按 `academic-research-suite/experiment-agent` 验证口径新增 [audit_sokoban_sd_goal_alignment.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py)。该脚本不训练 transition model，不改 provider，不运行 closed-loop SD-MCTS；best-first diagnostic teacher 只用于离线标签，以审计六存量、反馈/结构特征和候选 goal probes 是否能区分 teacher value、teacher solved 标签和一步动作后果。

full 与 medium 配置因离线 teacher 搜索耗时较高停止，不作为证据。quick 配置覆盖 `30` 个关卡，状态级 train `231`、valid `60`，动作后果级 train `66`、valid `17`，teacher 标签计数为 `solved=156`、`teacher_budget_unsolved=52`、`proven_dead_end=83`。结果文件为 [sd_goal_alignment_audit_30_levelsplit_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_quick_seed0.json)。

quick 审计中，六存量对验证集 teacher value 的 ridge 回归 `R^2=0.8859336361147274`，teacher solved AUC `0.9247685185185185`，六存量 alias bucket 的 within/global target variance ratio 为 `0.022069573453746945`。加入 existing feedback/structural 后验证 `R^2` 分别为 `0.8965768263764594` 与 `0.8987603377865637`；加入 goal probes 后的 `all_existing_plus_goal_probes` 验证 `R^2=0.8947046578760309`，相对六存量只增加 `0.00877102176130351`。动作后果审计中，固定 stock Q 与 teacher best action 的验证 top-1 agreement 为 `0.8235294117647058`。但固定 `stock_value` readout 对 teacher value 的验证 `R^2` 仅为 `0.07533288704528096`，且阶段 3 oracle abstract SD-MCTS 仍为 `0.0` 成功率。

**选项**
1. 将阶段 3 失败直接归因为六存量抽象错误，立即扩展核心存量。
2. 将 quick 审计 PASS 解释为当前 SD 设计充分，继续只调 MCTS budget 或 TVA 权重。
3. 将 quick 审计解释为“六存量与任务目标方向一致但不保证长程规划充分”，冻结为当前目标一致性证据，并把下一步放在规划消费、动作抽象和价值 target 质量审计。

**结论**
采用选项 3。当前六存量不是任意状态压缩，也不是与任务目标无关的启发式包装；它们在 quick 目标一致性审计中能解释大部分离线 teacher value 方差，并能较好地区分 solved/unsolved 标签。然而，该结果不能推翻阶段 3 负结果，因为局部 stock Q/readout 的长程价值解释力仍弱，oracle abstract SD-MCTS 也未能闭环求解。

**影响**
下一阶段不应以“再加更多核心存量”为默认路线，也不应把 quick 审计 PASS 当作成功规划证明。优先工作为：扩大目标一致性审计到 medium/full 可完成配置；审查 `V_search` positive target 和 root state aliasing；将 goal-room、macro、relevance、route consistency 作为受控 SD feedback、option 或辅助 probe 做单因素接入；引入 TD target network 时同步报告 target 质量和 held-out success rate。若新增结构不能降低 root/value aliasing 或提升 oracle/learned SD-MCTS 成功率，不得进入默认 SD 核心。

## 决策 26

**决策**
将 `expert_abstract_dynamics_feedback_v1` 记录为阶段 3 规划消费层的方向性修订，而不是稳定通过模型。该修订保留六存量和 learned transition provider，只在 SD-MCTS reward、action prior、terminal value 与闭环重复状态反馈中消费人类专家式抽象动力信号。

**背景**
用户要求从第一性原理审视人类专家如何从 Sokoban 直接观测中抽象思考。人类专家通常不会只看静态评分，而会维护目标完成、箱子到目标的可分配性、可行动余量、玩家可达性、入口和瓶颈占用、不可逆风险以及重复状态等动力机制。本轮据此修改 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，新增 `--planner-feedback-mode expert` 与 `--terminal-value-mode expert`。该模式不修改 `learned_stock_flow_checkpoint_v1`，不在线使用 best-first teacher，结果文件记录 `planner_feedback_changes_transition_provider=false`、`expert_feedback_teacher_used_online=false` 和 `expert_feedback_best_first_used_online=false`。

同预算同深度对照显示，旧 `stock` consumption 在 held-out `6` 关卡、budget `24`、depth `4` 下成功率为 `0.0`，而 expert feedback provider 对照文件 [stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json) 中，`learned_sdmcts` 成功率为 `0.16666666666666666`，解出 level `000.txt:6`，`learned_provider_fraction=1.0`，planner feedback usage gate 为 `PASS`。同一文件中 `oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率为 `0.0`。

但后续稳定性复测显示，`learned_sdmcts` 在 budget `48`、depth `4` 下成功率回到 `0.0`；budget `200`、depth `5` 和 budget `200`、depth `4` 下也均为 `0.0`。`puct_c=0.25` 与 expert prior uniform mix `0.5` 均未恢复 budget `48/200` 成功率。

**选项**
1. 将 budget `24` 的非零成功率判定为阶段 3 已通过。
2. 因 budget `48/200` 失败而丢弃 expert feedback 修订，回到 stock consumption。
3. 将该修订记录为方向性改善：保留代码和 quick 非零成功率证据，同时明确稳定阶段 3 仍未通过，下一步专门处理预算敏感性和长程反馈尺度。

**结论**
采用选项 3。expert feedback 证明人类专家式抽象动力机制可以作为 SD-MCTS 消费层信号产生非零闭环成功率，且没有破坏 learned provider 合约；但它尚未形成预算稳定的规划策略，不能替代后续 TVA、action abstraction 或 option-level SD feedback 的系统修订。

**影响**
后续阶段 3 门限调整为：同一 held-out split 上至少报告 budget `24`、`48` 和 `200`；`learned_sdmcts` 必须在 `48` 与 `200` 中至少一个预算保持非零成功率，且不能只在一个窄预算偶然成功；learned provider fraction 必须为 `1.0`；planner feedback usage gate 必须为 `PASS`；teacher 与 best-first diagnostic 不得在线参与 action choice 或 root scoring。若新增反馈只在 budget `24` 成功而在更高预算退化，应记录为预算敏感的中间结果，不作为阶段 3 稳定通过。

## 决策 27

**决策**
将 deadlocked-root 展开修复和 viability feedback 记录为 expert feedback 的必要诊断修订，但不继续围绕惩罚项做无门限调参。下一步应转向目标导向 option/action abstraction 或 learned option prior，而不是继续提高 deadlock、revisit 或 viability 惩罚。

**背景**
budget `48/200` 的 expert feedback 复测显示明显预算不稳定。读取失败轨迹后发现，`ExpertAbstractTerminalValueEvaluator.should_stop` 会在 `model.is_deadlocked(raw_state)` 时停止展开，导致合法 push 仍存在的 root 可能返回 `planner_returned_no_action`。修复后，只在 solved 或 no-legal root 停止，deadlock 保留为强负值。修复结果显示，budget `48` 的 `planner_returned_no_action` 消失，但成功率仍为 `0.0`，失败变为 `no_legal_actions=5`、`max_pushes=1`。

随后新增 `--expert-viability-penalty`，对后继 `viable_push_count=0` 或 `1` 的动作施加 reward/prior 惩罚。该修订保持 budget `24` 的 `1/6` 成功率，但 budget `48` 仍为 `0/6`。`root_action_selection=visits` 消融将 budget `48` mean pushes 提高到 `32.0`，失败变为 `max_pushes=3`、`no_legal_actions=3`，但仍未解出关卡。

**选项**
1. 继续调高 viability、deadlock 或 revisit 惩罚，直到 budget `48` 出现成功。
2. 回退 deadlocked-root 修复和 viability feedback，因为它们没有使 budget `48` 通过。
3. 保留这些修复作为必要诊断改进，同时承认仅靠惩罚项不能产生稳定目标导向规划，下一步进入 option/action abstraction。

**结论**
采用选项 3。deadlocked-root 修复解决了一个真实的 MCTS 展开问题；viability feedback 使失败轨迹更保守，但没有形成求解策略。当前失败说明，SD-MCTS 需要更明确的高层动作或目标区域进入机制，才能把专家抽象动力反馈转化为稳定解，而不是只靠负反馈避免坏状态。

**影响**
后续实验应新增受控 option/action abstraction，例如目标房间进入、箱子-目标分配路线、有限宏推箱序列或 route-consistency option prior。每个 option 仍必须通过 learned provider 预测抽象 stock transition，并报告 provider fraction、feedback usage gate、budget `24/48/200` 成功率和 failure type。若 option 直接调用 best-first 搜索生成在线动作，则必须标记为 diagnostic，不得作为 SD-MCTS 主线。

## 决策 28

**决策**
将 `target2_prior` 记录为受控高层动作先验的合规中间结果，但不作为阶段 3 通过模型。完整 `target2` 额外 action 子节点暂不进入默认路线。

**背景**
阶段 3 expert feedback 的失败已转向“缺少稳定目标导向 option/长程价值”。本轮实现 `target2` 系列动作抽象：`target2` 只允许同一箱子的局部两步推进，不调用 best-first，不进行在线路径搜索；option 内若作为边展开，每个 primitive transition 仍调用 selected transition provider。初始 `target2` 额外子节点在 2-level smoke 中通过 provider 与 action abstraction gate，但会扩大 root 分支并压制原本可解 level `000.txt:6` 的成功轨迹，因此不作为默认路线。

随后实现 `target2_prior`，不新增 root 子节点，只把同一箱子的两步目标推进潜力折入 primitive action prior，并记录 `target2_prior_scored_roots`。该模式在 2-level smoke 中恢复 `1/2` 成功率。held-out `6` 关卡 budget `24` 下，结果文件 [stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json) 显示 `learned_sdmcts` 成功率为 `0.16666666666666666`，与 viability expert baseline 持平；budget `48` 文件 [stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json) 仍为 `0.0`。三 provider 对照 [stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json) 中，learned 为 `1/6`，oracle abstract 与 zero-flow 均为 `0/6`，provider、feedback 和 action abstraction gate 均通过。

**选项**
1. 采用完整 `target2` 额外 action 子节点作为默认动作抽象。
2. 将 `target2_prior` 判定为阶段 3 通过，因为它合规且在 budget `24` 保持非零成功率。
3. 保留 `target2_prior` 为合规中间机制，记录预算 `48` 未通过，并将下一步转向 root value 选择偏移、route consistency、goal-room entry 或 learned option prior。

**结论**
采用选项 3。`target2_prior` 满足“结构先验不替代 learned provider、不调用在线求解器”的协议，但没有改善阶段 3 稳定性门限。完整 `target2` 额外子节点当前还会损害低预算有效搜索，应暂保留为诊断实现。

**影响**
后续动作抽象必须满足更严格的进入条件：budget `24` 不低于当前 `1/6`；budget `48` 必须恢复非零成功率或显著降低成功关卡的 root value 偏移；provider fraction 必须为 `1.0`；action abstraction gate 必须记录真实使用；不得使用 best-first 在线生成动作。若新增机制只改变 prior 而不提高 budget `48`，只能作为 probe 或辅助特征保留。

## 决策 29

**决策**
`route_consistency` 与 `goal_room_order` 作为阶段 3 规划层诊断机制保留，但不进入默认模型。后续不继续手工调 goal-room 权重，转向 learned option prior 或 root value target 审计。

**背景**
对 level `000.txt:6` 的 budget `24/48` 分歧进行逐步渲染后发现，budget `48` 失败并非 provider 接线问题，而是 root value 在关键第二步偏向“立即新增箱子上目标”的局部收益；成功轨迹需要先整理已在目标区的箱子，将其推向端点目标。该现象符合 Sokoban 专家对目标区打包顺序的抽象判断。

本轮实现两个默认关闭的诊断参数：`--route-consistency-weight` 和 `--goal-room-order-weight`。前者尝试延续上一被推动物理箱子的路线意图，2-level budget `48` 仍为 `0.0` 成功率。后者对目标区内部向端点目标整理给正反馈，并对存在可整理目标区箱子时的新箱入目标施加 delay penalty。2-level budget `48` 中，权重 `2.0` 和 `3.0` 均恢复 `0.5` 成功率，但 held-out `6` 关卡 budget `24` 中，权重 `1.5/2.0/3.0` 均退化为 `0.0`，低于当前 target2_prior/expert baseline 的 `1/6`。

**选项**
1. 将 `goal_room_order_weight=2.0` 或 `3.0` 纳入默认，因为它修复了 2-level budget `48`。
2. 继续密集搜索 goal-room 权重，直到同时满足 budget `24/48`。
3. 将 route/goal-room 机制记录为诊断性局部信号，不进入默认；后续改为学习 option prior 或审计 root value target。

**结论**
采用选项 3。goal-room packing 解释了一个真实失败机制，但当前手工反馈不可跨关卡稳定泛化。将其纳入默认会破坏 quick 基线，不符合阶段 3 进入条件。

**影响**
后续阶段 3 的下一步应以 learned option prior、route-consistency target 或 `Q_H` 偏差审计为主，而不是继续增加手工权重。任何新机制必须在 held-out `6` 关卡上同时满足：budget `24` 不低于 `1/6`，budget `48` 恢复非零成功率，provider usage gate 为 `PASS`，feedback/action abstraction gate 为 `PASS`，且 best-first 不在线参与 action choice。

## 决策 30

**决策**
将 root-Q bias audit、audit-proxy prior 和 learned action prior v1 记录为阶段 3 规划消费层的诊断性负结果；learned action prior v1 不进入默认模型，不扩大到 held-out `6` 关卡完整门限测试。

**背景**
root-Q bias audit 量化了 level `000.txt:6` budget `24/48` 的关键分歧：budget `48` 失败动作相对成功动作 root `Q` 高 `1.0426520933606815`，prior 高 `0.6487923854590042`，访问数比为 `4.8`。成功动作属于目标区内部整理，失败动作属于立即新增箱子上目标。该审计支持“短期目标完成收益压过必要整理动作”的失败机制。

随后实现 `--audit-proxy-prior-weight`，但它只保持 budget `24` 的 `1/6`，没有恢复 2-level budget `48`。进一步实现 learned action prior v1，训练数据来自 stage-2 provider train split 的 closed-loop SD-MCTS trace，在线只作为 action-prior multiplier，不调用 best-first，不修改 learned transition provider。训练样本共 `443`，其中 train positive 仅 `6`，内部 valid 无正样本。2-level budget `48` 中，learned prior 权重 `0.5` 与 `0.1` 均为 `0/2`。

**选项**
1. 继续搜索 learned action prior 权重或扩大模型容量，直到 2-level budget `48` 出现成功。
2. 因 learned prior v1 接口合规而直接运行 held-out `6` 关卡完整测试。
3. 将 root-Q bias audit 作为失败机制证据保留，将 audit-proxy 与 learned action prior v1 作为负结果冻结；后续只有在训练 target 具有足够正例和内部 holdout 判别力时，才重新进入 learned prior 或 option prior 训练。

**结论**
采用选项 3。当前 learned prior v1 的问题不是接口，而是 target 质量不足和正例极稀。继续调权重会把少量训练成功轨迹过拟合成 prior，不能形成可审计的阶段推进。

**影响**
后续 learned prior 或 option prior 的进入条件调整为：训练数据必须来自 train split 或明确的离线 target，不得使用 held-out valid 成功轨迹作为训练标签；训练集和内部 holdout 必须同时含正负样本；内部 holdout AUC 或排序指标必须高于随机；随后先通过 2-level budget `48` 非零成功率，再扩展到 held-out `6` 关卡，且 budget `24` 不低于 `1/6`、budget `48` 非零、provider usage gate 和 feedback/action prior usage gate 同时通过。

## 决策 31

**决策**
当前不将 goal-room packing 升级为新的核心 SD stock，也不启动阶段 1/2 provider 重训。该机制继续作为局部失败诊断、option target 或分层审计变量保留。

**背景**
root-Q bias 和 goal-room delay 实验证明，目标区内部整理确实解释 level `000.txt:6` 的 budget `48` 失败。但是否应进入 SD 核心结构，需要满足更高证据标准：候选 stock 不仅要有单步转移信号，还应在 held-out teacher value 或 alias 审计中带来稳定增益。

新增 `audit_sokoban_goal_room_stock_probe.py` 后，quick 审计显示 goal-room 特征单步 transition nonzero fraction 为 `0.16944106925880922`，说明其可推演；但六 stock valid `R^2=0.8859336464419086`，加入 goal-room 特征后 `R^2=0.8850849477314626`，增益为 `-0.0008486987104460164`，未达到 `0.02` 的候选 stock 门限。

**选项**
1. 立即将 goal-room packing 作为第七个核心 stock，扩展 incidence matrix 并重训阶段 1/2。
2. 完全丢弃 goal-room 机制，因为候选 stock gate 未通过。
3. 不扩展核心 stock，将 goal-room packing 保留为局部诊断和 option/search-target 分层变量；只有在更强审计中证明其跨关卡价值增益后，再考虑核心结构修订。

**结论**
采用选项 3。goal-room packing 解释具体失败，但当前证据不足以改写核心 SD 结构。直接扩展 stock 会破坏已通过的六存量阶段 1/2 provider，而没有相应的 held-out value 增益支持。

**影响**
后续 SD 核心扩展门限调整为：候选 stock 必须同时具备足够单步转移信号、held-out teacher value 增益、alias 方差下降或 action consequence 排序增益，并能通过阶段 1 flow gate 与阶段 2 learned provider audit。未满足这些条件前，候选机制只能作为 planner probe、option target 或离线分层审计变量。

## 决策 32

**决策**
将 `goal_room_conflict_prior` 记录为 root-only 诊断性负结果，不进入默认阶段 3 配置，也不扩展到 held-out `6` 关卡完整门限测试。

**背景**
root-Q bias audit 显示 level `000.txt:6` 的 budget `48` 失败发生在第二步：失败动作偏向立即新增箱子上目标，而成功动作需要先把目标区内部箱子从中间目标整理到端点目标。已有 `goal_room_order_weight` 能在 2-level budget `48` 修复局部失败，但会破坏 held-out `6` 关卡 budget `24` 的 `1/6` quick 基线；goal-room stock probe 又显示该机制暂不足以升级为核心 stock。

因此本轮实现更窄的 `--goal-room-conflict-prior-weight`：只在 root action prior 层使用 `goal_room_order_bonus - goal_entry_delay_penalty`，不改变 reward、terminal value 或 learned transition provider。该机制的 planning contract 记录 `goal_room_conflict_prior_root_only=true`、`goal_room_conflict_prior_changes_transition_provider=false`、`goal_room_conflict_prior_best_first_used_online=false` 和 `goal_room_conflict_prior_changes_reward=false`。

2-level budget `48` smoke 中，`weight=1.0` 与 `weight=2.0` 均为 `0/2`，虽通过 provider、feedback 和 action abstraction gate，但未达到入场门限。level `000.txt:6` 上该 prior 确实触发并改变第二步选择，但只是把失败前缀从 `[1, 7, 8, 4, 4]` 改为 `[1, 12, 15, 4, 4]`，未恢复成功前缀 `[1, 3, 12, 15, 4]`。

**选项**
1. 继续提高 conflict prior 权重，直到 2-level budget `48` 出现成功。
2. 将该 prior 与旧 `goal_room_order_weight` 组合，进行更细密手工调参。
3. 冻结该路线为诊断负结果；保留代码和结果合同，但后续转向 learned search-value、option target 或目标区分配承诺建模。

**结论**
采用选项 3。该 prior 合同上符合 SD-MCTS 协议，但没有恢复最低 smoke 成功率。继续调手工权重会把具体关卡的 root 分歧变成规则搜索，偏离“可学习 SD 世界模型 + SD-MCTS 消费层”的研究框架。

**影响**
后续阶段 3 推进不再沿 goal-room hand-prior 权重搜索。若重新引入目标区机制，必须满足至少一个更强条件：形成 learned option/search-value target，并在内部 holdout 有可检验排序能力；或者作为候选 stock 通过 held-out value/alias/action consequence 增益门限；或者在不降低 budget `24` 当前 `1/6` 的前提下，使 2-level 和 held-out budget `48` 均恢复非零成功率。

## 决策 33

**决策**
将 `sokoban_teacher_q_action_prior_v1` 记录为“训练信号可学但阶段 3 smoke 未通过”的负结果，不进入默认配置，也不扩展到 held-out `6` 关卡完整测试。

**背景**
原 learned action prior v1 使用 closed-loop SD-MCTS solved-trace 标签，训练正例过稀，内部 valid 无正样本。本轮改为审计 stage-2 train split 的离线 successor teacher-Q action target。审计只使用 train levels，并在 train levels 内划分内部 holdout；best-first teacher 只用于 successor state 离线标注，不在线参与 SD-MCTS，也不修改 transition provider。

target signal quick 审计通过：train best-action labels `36`，内部 holdout best-action labels `14`，valid rank states `14`，ridge valid top-1 agreement `0.42857142857142855`，valid Q corr `0.7964238411550628`。随后训练 teacher-Q prior。初始 final epoch 过拟合，因此修订训练脚本，按内部 holdout rank 选择 best epoch。best epoch 为 `9`，内部 holdout valid top-1 agreement `0.6428571428571429`，valid Q corr `0.8467250504899865`。

但该 checkpoint 作为 action-prior multiplier 接入 SD-MCTS 后，2-level budget `48` 仍为 `0/2`。`learned_action_prior_weight=0.1` 和 `0.5` 均通过 provider usage gate、planner feedback usage gate、action abstraction gate 和 learned action prior usage gate，但没有恢复成功率。level `000.txt:6` 的前缀在 `0.1` 下仍为 `[1, 7, 8, 4, 4, 8]`，在 `0.5` 下为 `[1, 7, 8, 4, 4, 9]`，均未恢复成功前缀 `[1, 3, 12, 15, 4]`。

**选项**
1. 继续搜索 teacher-Q prior 权重或 max multiplier，直到 2-level budget `48` 成功。
2. 因内部 holdout 排序表现较好，直接扩展到 held-out `6` 关卡完整测试。
3. 冻结该 prior 为阶段 3 负结果；保留 target 审计和训练证据，将下一步转向 search-value/TVA 或 option-level rollout target。

**结论**
采用选项 3。该 prior 证明动作级离线 target 可学习，但 root prior multiplier 的作用形式不足以解决当前长程备份偏差。继续调权重会偏向规则化搜索，不符合“先用可靠、可审计模型逐步丰富”的路线。

**影响**
后续 learned prior 的门限调整为：不仅要求 target signal gate 和内部 holdout 排序通过，还必须先在 2-level budget `48` 恢复非零成功率，才能进入 held-out `6` 关卡。若 action-Q target 再利用，应优先作为 TVA/search-value target、option rollout value 或高层 option 选择目标，而不是单独作为 primitive root prior multiplier。

## 决策 34

**决策**
将 `sokoban_search_value_tva_stage3_td_v2_quick` 记录为“TVA 语义与阶段 3 planner 对齐更完整，但最低 smoke 未通过”的负结果；不进入默认配置，也不扩展到 held-out `6` 关卡完整测试。

**背景**
复核 NeurIPS 论文第 4.4 节后，确认 TVA 不应只训练一个接口兼容的 terminal value 网络，而应让 `V_psi` 对齐实际 MCTS 消费层产生的 root statistics。旧 `sokoban_search_value_tva_v1` 虽然使用 `V_search=sum_a pi_mcts Q_H`，但采集目标来自简化的 stock-reward primitive MCTS，与阶段 3 当前最强中间线 `expert feedback + target2_prior` 不一致；同时其 TD 项未启用，终止后继在 TD 中置零。

本轮修订 `train_sokoban_search_value_tva.py`，新增 `--target-planner-mode stage3`，复用阶段 3 的 learned transition provider、`expert_abstract_dynamics_feedback_v1` 和 `target2_prior` 采集 `V_search`。同时启用 `lambda_td=0.25` 与 target network，并将 TD target 改为 `r+discount*Vbar_or_terminal(s_next)`。`SokobanTvaTerminalValueEvaluator` 也被修订为不因 deadlock proxy 阻断展开，但对 solved、deadlocked、no-legal leaf 给出显式边界值。所有修订均不改变 learned stock-flow transition provider，也不在线调用 teacher 或 best-first。

quick 训练采集 `217` 个 root 样本，train `173`、valid `44`。训练 valid corr 为 `0.9664165047048058`，valid MAE 为 `1.0694197416305542`。但 positive search target 仍然稀少：全样本正值 target 为 `12`，valid positive 仅 `1`，且该正例 target mean `0.14891120791435242`、pred mean `-2.7556140422821045`。

2-level budget `48` smoke 中，`root_action_selection=value` 与 `visits` 均为 `0/2`。两个结果均通过 provider usage gate、terminal value usage gate、planner feedback usage gate 和 action abstraction usage gate。level `000.txt:6` 中，`value` 版本仍走向既有失败前缀 `[1, 7, ...]`，没有恢复成功前缀 `[1, 3, 12, 15, 4]`。

**选项**
1. 将该 TVA v2 扩展到 held-out `6`，因为它的内部 valid corr 较高且语义更合规。
2. 继续调 `lambda_td`、正例权重或训练轮数，直到 2-level budget `48` 成功。
3. 将其冻结为负结果；保留代码与 contract，下一步转向 positive search-value target 覆盖、option-level value target 或更明确的高层 option。

**结论**
采用选项 3。该修订解决了旧 TVA 与阶段 3 planner 语义不一致的问题，但没有满足最低阶段门限。内部回归相关性高不能替代 closed-loop smoke；正向 search-value target 稀疏和跨关卡泛化不足仍是关键瓶颈。

**影响**
后续 TVA 路线的进入条件调整为：target collection 必须说明是否与实际 planner 消费层一致；TD 项若启用，必须报告 target network 和终止后继价值来源；候选 checkpoint 必须先在 2-level budget `48` 非零成功，才可扩展到 held-out `6`。在该门限未满足前，不继续围绕该 checkpoint 做无界 `lambda_td` 或正例权重搜索。

## 决策 35

**决策**
将 `sokoban_stage3_planner_tva_v2_quick` 与 `sokoban_stage3_planner_tva_v2_expanded` 作为 stage3-planner TVA v2 的复核负结果冻结；不进入默认配置，不扩展到 held-out `6` 关卡完整测试。

**背景**
决策 34 已记录 `sokoban_search_value_tva_stage3_td_v2_quick` 的语义修订与 2-level smoke 失败。上下文恢复后，为核对当前代码状态并排除样本过小造成的误判，重新以更清晰的命名训练两个 checkpoint。二者均使用 `target_planner_mode=stage3`、`planner_feedback_mode=expert`、`action_abstraction_mode=target2_prior`、`bootstrap_terminal_value_mode=expert`、`lambda_td=0.25`，并保持 `teacher_used_online=false`、`best_first_used_online=false`、`transition_provider_modified=false`。

quick 版采集 `81` 个样本，valid corr `0.9321286812910057`，valid MAE `1.2824660539627075`，但正值 search target 只有 `1` 个。2-level budget `48` smoke 成功率为 `0.0`，provider、terminal value、planner feedback 和 action abstraction usage gate 均为 `PASS`。

expanded 版仅扩大离线 root-state 覆盖，采集 `356` 个样本，正值 search target `15` 个；valid corr `0.9534958319265285`，valid MAE `0.9219829440116882`。但 valid positive 仍只有 `1` 个，target mean `0.14891120791435242`，pred mean `-2.613969326019287`。对应 2-level budget `48` smoke 成功率仍为 `0.0`，四类 usage gate 均为 `PASS`。

**选项**
1. 因 expanded 版 valid corr 较高，继续扩展到 held-out `6` 完整门限测试。
2. 继续调 `lambda_td`、正例权重、训练轮数或采样规模，直到 2-level budget `48` 成功。
3. 按既定门限冻结该路线，记录为语义合规但 closed-loop 失败；下一步转向 option-level value target 或更明确的高层动作/分配承诺建模。

**结论**
采用选项 3。两个复核 checkpoint 都证明 stage3-planner TVA v2 可以学习大部分负值 search target 排序，但不能稳定识别稀有正值 search target，也不能在最低 2-level budget `48` smoke 中产生非零成功率。内部回归相关性不能替代 closed-loop 成功门限。

**影响**
后续不再围绕该 TVA v2 checkpoint 做无界超参数搜索。若继续使用 TVA，应先改变 target 生成机制，而不是继续调当前 primitive-root `V_search`。可选方向包括：option-level search value、目标区分配承诺 value、成功/失败分层采样、或将 action-Q 离线 target 作为 auxiliary loss 而非 root prior multiplier。任何新版本仍必须先通过 2-level budget `48` 非零成功率，才能扩展到 held-out `6`。

## 决策 36

**决策**
将 `sokoban_teacher_q_state_value_v1_quick` 冻结为 teacher-Q state-value terminal leaf probe 的诊断负结果；不进入默认阶段 3 配置，也不扩展到 held-out `6` 关卡完整测试。

**背景**
在 teacher-Q action prior 作为 root prior multiplier 失败后，进一步检验同一离线 action-Q 信号是否更适合作为状态级终端叶值。新增 `train_sokoban_teacher_q_state_value.py`，只使用 stage-2 provider train split，并在 train split 内部划分 holdout。状态 target 定义为 `max_a Q_teacher(s,a)`，其中 best-first teacher 只用于 successor state 的离线标注，不在线参与 SD-MCTS，不使用 held-out valid split 训练，也不修改 learned stock-flow transition provider。contract 标注 `planner_consumption=terminal_leaf_value_only` 与 `strict_search_statistics_tva=false`，因此该方法不是严格 search-statistics TVA，而是 teacher-Q 聚合状态价值 probe。

quick 训练中，action-Q target signal gate 通过：train best-action labels `36`，内部 holdout best-action labels `14`，valid rank states `14`，valid top-1 agreement `0.42857142857142855`，valid Q corr `0.7964238411550628`。state-value gate 也通过：train state samples `33`，valid state samples `14`，valid corr `0.3278236921586032`，valid MAE `3.12015962600708`。但分组校准暴露风险：valid positive target 组表现较好，MAE `0.8567870259284973`，corr `0.9382455304567155`；valid nonpositive target 组被严重高估，target mean `-2.052501916885376`，pred mean `4.085488319396973`，MAE `6.137990474700928`。

2-level budget `48` smoke 结果为 `0/2`，两个关卡均达到 `max_pushes`。provider usage gate、terminal value usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`，说明失败不是接口未接入，而是该 terminal leaf value 对 closed-loop 规划没有带来最低收益。

**选项**
1. 因 positive target 分组相关性较高，继续调训练权重或扩大样本后重跑。
2. 直接扩展到 held-out `6`，以确认是否存在偶然低估。
3. 按既定门限冻结该路线；保留其训练和 smoke 结果作为 teacher-Q state-value 的诊断证据，后续转向 option-level rollout target、分配承诺 value 或带非正状态校准的 value target。

**结论**
采用选项 3。`max_a Q_teacher(s,a)` 聚合保留了局部可行动作的乐观信号，但会高估预算未解或失败状态；内部 positive 相关性不能替代 closed-loop smoke。继续调该 checkpoint 容易变成无界价值校准，而不是推进 SD 世界模型和 SD-MCTS 的结构性设计。

**影响**
teacher-Q 状态价值若再次进入阶段 3，必须增加 nonpositive/budget-unsolved 校准 gate，并说明其 target 与论文 TVA、option value 或 planner terminal value 的层级关系。任何重启版本仍必须先满足 2-level budget `48` 非零成功率、四类 usage gate 通过、transition provider unchanged 和 no online teacher/best-first contract，才可扩展到 held-out `6`。

## 决策 37

**决策**
将当前 `target2_local_structural_option` 的显式 option 消费和 option-target learned-prior 入场尝试冻结为诊断负结果；保留 option-level 抽象方向，但下一步必须重新定义 option target，而不是继续调 target2 结构分数、prior multiplier 或执行模式。

**背景**
在 primitive action prior、search-value TVA 和 teacher-Q state-value 均未通过阶段 3 smoke 后，继续检查是否应转向 option-level target。`audit_sokoban_option_target_signal.py` 将 primitive push 与 `target2_local_structural_option` 放在同一离线 teacher-Q 排序空间中，候选 option 的 target 为 option 内 stock reward 累积加上 successor state 的离线 teacher value。该审计只使用 stage-2 train split 及内部 holdout；best-first teacher 不在线参与 SD-MCTS，不使用 held-out valid split 训练，也不修改 learned stock-flow transition provider。

quick 审计采集 train option rows `254`、valid option rows `100`，候选来源为 primitive push `245` 条、target2 option `109` 条。option target signal gate 为 `FAIL`，唯一失败项是 valid top-1 agreement 低于 `0.40`。观测值为：train best option labels `33`，valid best option labels `14`，valid rank states `14`，valid top-1 agreement `0.2857142857142857`，valid Q corr `0.7794633473898425`，valid teacher-Q std `4.823039134104247`，train best target2 option labels `22`，valid teacher-best option state fraction `0.5`。

该审计说明 option-level 信号存在，但现有学习/结构消费形式未达门限。特别是，`fixed_stock_return` baseline 的 valid top-1 agreement 为 `0.5714285714285714`，`exact_option_score` 为 `0.5`，均高于 ridge option-Q 的 `0.2857142857142857` 和 target2 structural score 的 `0.21428571428571427`。这表明当前 target2 结构分数不足以作为默认 option prior，直接训练 learned option prior 也缺少入场证据。

随后复测显式 target2 option 的 closed-loop 消费。2-level budget `48` 下，`option_execution_mode=first_push` 的结果为 `0/2`，action abstraction usage gate 为 `PASS`，`option_expansions=337`，`option_primitives=674`；`option_execution_mode=full_option` 仍为 `0/2`，usage gate 为 `PASS`，`option_expansions=281`，`option_primitives=562`。provider usage gate 和 planner feedback usage gate 均通过，说明 failure 不是 option 没有展开，也不是 learned provider 未被调用。

**选项**
1. 继续提高 target2 option prior multiplier，或扩大 target2 option 数量。
2. 因 teacher-best option 覆盖达到 `0.5`，直接训练 learned option prior 并上线。
3. 冻结当前 target2 option 消费为负结果；保留 option-level 抽象方向，但要求下一版重新定义 option target，并先通过离线 top-1 gate 和 2-level closed-loop gate。

**结论**
采用选项 3。当前证据支持“人类专家式高层 option 抽象是必要方向”，但不支持“现有 target2 两步结构分数已经足够”。继续调 target2 prior 或执行模式会把局部结构假设硬推成在线规则，而不是改进 SD 世界模型的任务目标状态推演。

**影响**
下一版 option 方法必须把 option target 与任务目标动力机制重新对齐，例如采用 SD rollout return、目标区分配承诺、箱子-目标路线承诺、非正状态校准和跨关卡 option top-1 gate。入场条件为：离线 valid top-1 agreement 不低于 `0.40`、valid Q corr 不低于 `0.20`、teacher-best option 覆盖充足、best-first 不在线使用、transition provider unchanged；随后 2-level budget `48` 必须非零成功，才可扩展到 held-out `6`。

## 决策 38

**决策**
阶段 3 后续调试新增强制前置门限：TVA checkpoint 必须先通过离线截断价值准确性审计，才允许进入 SD-MCTS closed-loop 候选测试。当前 `sokoban_stage3_planner_tva_v2_quick` 与 `sokoban_stage3_planner_tva_v2_expanded` 在该审计下冻结为截断价值校准负结果。

**背景**
用户指出，SD transition provider 可独立测试状态推演准确性，但 SD-MCTS 在 TVA 缺席时无法独立调试，因为 MCTS 展开步长远短于实际求解步长，缺少终端价值补偿必然陷入局部最优。因此，不能继续把无 TVA 或 TVA 未校准的 SD-MCTS smoke 当作主要调参依据。用户进一步建议用离线启发式搜索获得真实解或高置信剩余步数，从而直接评估 TVA 对截断误差的补偿能力。

本轮新增 `audit_sokoban_tva_truncation_value.py`。该脚本只用 offline best-first diagnostic teacher 标注状态剩余价值，不在线参与 SD-MCTS，不修改 learned transition provider、planner feedback 或 action prior。smoke 审计覆盖 `46` 个状态，其中 high-confidence 样本 `43` 个，teacher counts 为 `solved=35`、`proven_dead_end=8`、`teacher_budget_unsolved=3`。

审计结果显示，`expert_value` high-confidence MAE 为 `5.125799790019716`，`stock_value` 为 `5.408689795334134`。`sokoban_stage3_planner_tva_v2_quick` high-confidence MAE 为 `8.333739054839088`，positive bias 为 `-8.761932504277338`；`sokoban_stage3_planner_tva_v2_expanded` high-confidence MAE 为 `7.423524176047102`，positive bias 为 `-7.659244627712222`。二者相关性很高，但符号准确率均只有 `0.18604651162790697`，说明 TVA 学到了排序趋势，却没有学到真实截断剩余价值的尺度和正负边界。

**选项**
1. 继续将当前 TVA v2 作为 closed-loop SD-MCTS 候选，通过调 budget、prior 或 option 弥补。
2. 因 TVA 与 teacher value 相关性高，直接扩展到 held-out `6` 关卡。
3. 冻结当前 TVA v2 为截断价值负结果；后续先修订 TVA target 或校准方式，只有通过离线截断价值 accuracy gate 后，才进入 2-level budget `48` closed-loop smoke。

**结论**
采用选项 3。当前 TVA v2 的问题不是接口缺失，而是价值补偿方向错误：它对可解正值状态严重负偏，无法为短视界 SD-MCTS 提供终端补偿。继续调闭环 budget 或 prior 会把 TVA 校准问题转嫁到规划层，不符合 SD-MCTS-TVA 协作协议。

**影响**
后续阶段 3 的进入顺序固定为：一，SD transition provider 独立状态推演审计；二，TVA 截断价值准确性审计；三，带 learned provider、planner feedback/action abstraction 和通过 accuracy gate 的 TVA 运行 closed-loop SD-MCTS。无 TVA 或 TVA accuracy gate 失败的 closed-loop 结果只能标记为截断偏差诊断，不作为阶段 3 候选。

## 决策 39

**决策**
将 `option_value_prior_weight=0.1` 的 target2 option-value prior 冻结为 planner-prior 诊断负结果；不继续搜索权重，不进入 held-out `6` 关卡扩展测试，也不作为下一阶段默认配置。

**背景**
option target 审计显示，`fixed_stock_return` 在内部 holdout 上的 valid top-1 agreement 为 `0.5714285714285714`，高于当前 ridge option-Q probe 和 `target2_structural_score`。为检验该信号是否能以低风险方式被 SD-MCTS 消费，在 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py) 中新增默认关闭的 `--option-value-prior-weight`。该机制只在 planner action prior 层使用本地 option 内 SD stock rollout return，对候选 target2 option 进行 prior boost；raw state exact step 只用于 action support 和 stock readout，不在线调用 best-first teacher，不修改 `learned_stock_flow_checkpoint_v1`，也不改变 planner feedback 或 terminal value。

2-level budget `48` first-push smoke 输出 [stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json)。结果为 `0/2`，但 provider usage gate、planner feedback usage gate、action abstraction usage gate 和 option value prior usage gate 均为 `PASS`；`option_value_prior_rows=3070`，`option_value_prior_positive_rows=1264`。

随后只运行一个补充的 full-option 执行模式测试，输出 [stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，失败类型为 `max_pushes=2`；四类 usage gate 仍通过，`option_value_prior_rows=3168`，`option_value_prior_positive_rows=1350`。

**选项**
1. 继续搜索 `option_value_prior_weight` 或 prior multiplier，直到 2-level budget `48` 出现成功。
2. 因 usage gate 已通过且 option-value prior 大量触发，将其扩展到 held-out `6` 关卡。
3. 冻结该 planner prior 为负结果；保留“option-level SD rollout return 有离线信号”的诊断结论，但下一步转向重新设计 TVA 截断价值或目标区分配/承诺 option target。

**结论**
采用选项 3。该机制证明 option-value prior 确实被在线消费，但没有恢复最低 closed-loop 成功率。继续调权重会把一个尚未通过门限的 planner prior 变成手工搜索偏置，不符合“从简化、可靠、可审计的 SD 模型逐步丰富”的路线。

**影响**
后续 option 方向的门限保持不变：先通过离线 option target top-1 gate、非正状态校准和 TVA/截断价值相关审计，再进入 2-level budget `48` closed-loop。`option_value_prior` 当前只作为诊断性 planner-consumption 代码保留，其 contract 必须继续标注 `option_value_prior_changes_transition_provider=false`、`option_value_prior_best_first_used_online=false` 和 `option_value_prior_teacher_used_online=false`。

## 决策 40

**决策**
将 `sokoban_teacher_q_state_value_v1_quick` 重新定位为“截断价值校准诊断上界”，而不是严格 TVA 成功模型；阶段 3 不因该 checkpoint 通过离线截断价值 gate 而进入 held-out `6` 扩展测试。

**背景**
按决策 38，阶段 3 候选必须先通过 TVA 截断价值准确性审计。本轮将已有 checkpoint 放入同一审计：`sokoban_tva_value_v1`、`sokoban_tva_value_v2_failed_relabel`、`sokoban_teacher_q_state_value_v1_quick` 和 `sokoban_stage3_planner_tva_v2_expanded`。审计输出 [tva_truncation_value_audit_existing_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_existing_seed0.json)。

结果显示，严格 stage3 search-statistics TVA v2 仍失败：`sokoban_stage3_planner_tva_v2_expanded` 的 high-confidence MAE 为 `7.423524176047102`，差于 `expert_value` 基线 `5.125799790019716`，positive bias 为 `-7.659244627712222`。teacher-distilled `sokoban_tva_value_v1` 的 MAE 为 `1.2229815195856601`，但 positive bias 为 `-1.335852713344545`，未过 `-1.0` 门限。相反，`sokoban_teacher_q_state_value_v1_quick` 在同一审计下无失败项：high-confidence MAE `0.5409354691144549`，corr `0.9944298990860484`，sign accuracy `1.0`，positive bias `0.003251509906797726`，nonpositive bias `0.8333237171173096`。

但该 checkpoint 的 contract 明确标注 `strict_search_statistics_tva=false`，target 为 `max_a Q_teacher(s,a)`，来源是 stage-2 train split 的离线 successor teacher-Q 聚合。它不符合 NeurIPS 论文中由实际 MCTS root statistics 构造 `V_search=sum_a pi_mcts Q_H` 的 TVA 语义。因此，其通过截断价值 gate 只能说明“抽象状态特征足以承载截断叶值校准信号”，不能说明严格 TVA 已经成功。

为验证闭环消费是否仅受 option 执行方式限制，补跑 `terminal-value-mode=tva`、`tva-checkpoint=sokoban_teacher_q_state_value_v1_quick`、`action-abstraction-mode=target2`、`option-execution-mode=full_option` 的 2-level budget `48` smoke，输出 [stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，但 provider usage gate、terminal value usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`。

**选项**
1. 将 `sokoban_teacher_q_state_value_v1_quick` 改称为 TVA 成功模型，并扩展到 held-out `6`。
2. 因其 closed-loop 仍失败，完全忽略截断价值审计通过结果。
3. 将其作为诊断上界保留：证明叶值校准可由当前抽象特征承载，但严格 TVA target 和高层 option/备份消费仍需重新设计。

**结论**
采用选项 3。该 checkpoint 通过离线截断价值 gate 是有价值证据，但不能越过语义边界。阶段 3 当前失败不再能简单归因于“抽象特征无法表达叶值”；更准确的定位是：严格 search-statistics TVA 未学到真实截断价值，而 teacher-Q 叶值即使较准，也没有和当前 target2 option/action backup 协同产生闭环成功。

**影响**
后续路线分为两条且必须保持区分。第一，严格 TVA 路线必须改变 `V_search` target 生成或校准方式，使其也通过离线截断价值 gate，然后才进入 closed-loop。第二，teacher-Q 状态价值只能作为离线上界、辅助 loss 或诊断标签来源，不能直接写作论文 TVA 主结果。任何进入阶段 3 的候选仍需满足：learned provider fraction `1.0`、terminal value usage gate `PASS`、无在线 teacher/best-first、2-level budget `48` 非零成功率。

## 决策 41

**决策**
将 `sokoban_teacher_q_state_value_v1_quick` 重新归类为“通过截断价值 accuracy gate 的 terminal value compensator”，允许其作为后续 SD-MCTS 消费层诊断的叶值补偿器；但它仍不是严格 search-statistics TVA，也没有通过 2-level budget `48` closed-loop 入场门限。

**背景**
决策 36 曾基于训练脚本内部 valid 分组和 closed-loop smoke 将该 checkpoint 冻结为 teacher-Q state-value 负结果。新增截断价值审计后，对所有已有 TVA/terminal-value checkpoint 进行同批横向比较。结果显示，`sokoban_teacher_q_state_value_v1_quick` 是唯一通过 gate 的候选：high-confidence MAE `0.5409354691144549`，corr `0.9944298990860484`，sign accuracy `1.0`，positive bias `0.003251509906797726`，nonpositive bias `0.8333237171173096`。该证据比旧训练内部分组更直接，因为它评估的是 TVA 在 MCTS 截断叶节点上应补偿的真实剩余价值。

随后按新协议运行 2-level budget `48` SD-MCTS smoke，输出 `stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json`。结果为 `0/2`，mean pushes `40.0`；learned provider fraction `1.0`，TVA evaluations `3840`，planner feedback 和 target2_prior usage gate 均为 `PASS`。

**选项**
1. 因该 checkpoint 通过 accuracy gate，直接扩展到 held-out `6`。
2. 因 closed-loop 仍失败，完全废弃 teacher-Q state-value。
3. 将其保留为准确终端补偿器，用于定位 SD-MCTS 消费层问题；但不扩展 held-out，也不称为严格 search-statistics TVA。

**结论**
采用选项 3。该 checkpoint 证明“准确叶值补偿”在当前抽象特征上是可获得的；但闭环仍失败，说明阶段 3 主要瓶颈已转移到 MCTS 消费层，包括 root Q 备份、动作抽象、先验和 reward/terminal-value 尺度协同。

**影响**
后续不应继续重训同类 teacher-Q state-value，除非新的训练目标能解释为什么会改善 closed-loop 消费。下一步应围绕通过 accuracy gate 的叶值补偿器诊断 root policy：比较成功/失败前缀上的 root child Q、visit 分配、prior 权重、终端价值贡献和即时 feedback reward，判断当前 SD-MCTS 为什么没有选择 teacher 剩余价值支持的动作。

## 决策 42

**决策**
将 `sokoban_teacher_q_state_value_v1_quick` 保留为准确终端补偿器和 root sorting 诊断工具，但不把当前 `expert feedback + target2_prior + value root selection` 组合视为阶段 3 通过配置。下一步优先审查 root backup 尺度、learned/abstract stock transition 与 route-level option commitment 的协同，而不是继续重训同类 teacher-Q state-value。

**背景**
按决策 41，已有 accuracy-gate passed terminal compensator 后，阶段 3 的关键问题转为该叶值是否被 MCTS 正确消费。新增 `audit_sokoban_tva_root_sorting.py`，固定已有 2-level smoke 的 `trace_sample` 根状态，在 `zero_leaf`、`stock_leaf` 和 `teacherq_tva` 三种 terminal value 设置下比较 root policy。离线 teacher 只在规划后标注 root child successor 的剩余价值，不在线参与 SD-MCTS，不修改 transition provider。

审计结果显示，`teacherq_tva` 确实改善排序方向：相对 `zero_leaf`，教师最佳动作在 root Q 排名中的平均名次改善 `1.4`，平均 `q_teacher_corr` 从 `0.7418640397326196` 提升到 `0.8084293031527132`；相对 `stock_leaf`，名次改善 `1.3`，相关性从 `0.7096232078500353` 提升到 `0.8084293031527132`。但在 teacher-best tier 口径下，`teacherq_tva`、`zero_leaf` 和 `stock_leaf` 的 root Q top-tier 命中均为 `6/10`，critical miss 均为 `4/10`。因此，TVA 影响存在但不足以改变最终 root 选择集合。

一步备份分解进一步显示，使用 learned provider stock 的 `expert_reward + gamma * TVA` 一步分数虽然与 teacher value 相关，但 top-1 为 `0/10` 进入 teacher-best tier；使用 exact stock readout 的同一分解为 `6/10`。这说明 learned abstract transition 的 stock 预测会在局部备份中扰动动作排序。然而相同 TVA 与 planner 设置下，oracle abstract transition 2-level smoke 仍为 `0/2`，说明 provider 误差不是唯一瓶颈。

**选项**
1. 因 `teacherq_tva` 提高相关性，继续扩展 held-out `6` 或调大 budget。
2. 因 top-tier 命中没有净提升，废弃该 terminal compensator。
3. 保留该 compensator 作为诊断工具；同时把后续工作转向 root backup、reward/terminal 尺度、transition-provider stock calibration 与更明确的 route/goal commitment option。

**结论**
采用选项 3。该 compensator 已证明准确叶值可以进入并影响 MCTS 排序，但当前 planner 消费层没有把该影响稳定转化为 teacher-best tier 选择。oracle abstract 对照也失败，说明阶段 3 不能只靠替换 provider 或只靠重训同类 leaf value 解决。

**影响**
后续阶段 3 的低成本调试顺序调整为：一，继续使用 root sorting audit 作为前置诊断；二，检查 learned provider stock calibration 对 one-step backup top-1 的影响；三，审查 expert reward 与 TVA leaf value 尺度；四，若设计新 option，应以 route/goal commitment 和 teacher-best tier top-1 gate 为准入标准。任何新配置仍需先通过 2-level budget `48` 非零成功率，才可扩展到 held-out `6`。

## 决策 43

**决策**
不新增简单 reward/TVA leaf scale 参数作为阶段 3 候选，也不继续围绕现有两步 `target2` option 调执行模式。后续应把改进重点转向 learned stock transition 的价值消费校准，以及更明确的多步 route/goal commitment option。

**背景**
为检查 root backup 尺度是否可解释当前排序错配，新增 `audit_sokoban_tva_backup_scale.py`。该脚本只读取 root sorting 审计中已有的一步分解结果，离线扫描 `reward_scale * expert_reward + gamma * leaf_scale * TVA_leaf`，不重跑 MCTS、不调用 teacher、不修改 planner 或 provider。

扫描结果显示，在 learned provider stock 下，最佳尺度组合只达到 teacher-best tier `3/10`，critical miss `7/10`，低于当前完整 root Q 的 `6/10`。在 exact stock readout 诊断下，最佳组合可达 `8/10`，critical miss `2/10`，说明尺度调整只有在 stock transition 更准确时才有效。随后补跑 oracle abstract provider + 显式 `target2` option + full-option execution 的 2-level smoke，结果仍为 `0/2`，尽管 action abstraction usage gate 通过且 `option_expansions=350`。

**选项**
1. 新增 `terminal_value_scale` 或 `reward_scale` 参数，并按尺度扫描结果直接调闭环。
2. 继续调 `target2` 的 prior multiplier、执行模式或 option 数量。
3. 冻结简单尺度和两步 target2 执行模式为负结果，转向 provider stock calibration 与更强的 route/goal commitment option。

**结论**
采用选项 3。learned provider 下没有可上线的简单尺度候选；oracle + target2 full-option 仍失败，说明当前两步局部 option 不足以表达 Sokoban 任务所需的路线承诺。继续调比例或执行模式会把问题退化为手工搜索偏置。

**影响**
后续若修改 planner，应满足更强的前置证据：在 root sorting audit 中，新机制必须提升 teacher-best tier 命中或显著降低 critical miss；若修改 provider，应优先报告 value-consumption 相关的 stock calibration，而不仅是平均 transition error；若设计 option，应表示目标区分配、箱子-目标路线和多步承诺，并先通过离线 option target/top-tier gate。

## 决策 44

**决策**
冻结“逐存量仿射 provider 校准”和“目标绑定单调 route-goal option 直接上线”为诊断负结果；不把二者接入在线 SD-MCTS。后续若继续 route/goal commitment 方向，应先构造更明确的 learned option target、first-push commitment target 或目标分配 target，并通过离线 teacher-best tier gate 后再进入 2-level budget `48` smoke。

**背景**
在 root sorting 审计显示 learned provider stock 读数会扰动一步备份排序后，新增 provider stock calibration 审计。该审计只使用 stage-2 provider train split 拟合逐存量仿射变换，并在固定 root sorting 状态上比较 raw provider、calibrated provider 与 exact stock readout。结果显示，raw provider 的 teacher-best tier 命中为 `0/10`，critical miss 为 `10/10`；calibrated provider 仅改善到 `1/10` 和 `9/10`；exact stock readout 为 `6/10` 和 `4/10`。因此，主要差距不能用全局 scale/bias 后处理解释。

随后新增 route-goal commitment option 离线审计。该候选显式绑定一个箱子和一个目标，要求同一箱子沿静态 reverse-pull 距离单调接近该目标。quick 与 length-6 两个版本均覆盖全部 `10` 个固定 root，但 gate 均失败，失败项为 `route_goal_beats_nonroute_roots_min_2`；route-goal 最佳候选在 `0/10` 个 root 上严格优于 primitive/target2 的最佳候选。与此同时，route-only structural top-1 的 first-push tier 命中为 `8/10`，说明该机制更像 first-push commitment 信号，而不是可直接上线的显式长 option。

**选项**
1. 将 calibrated provider 作为 wrapper 接入 SD-MCTS，并运行 2-level smoke。
2. 将 route-goal monotone option 作为新的 MCTS child 或 prior 直接接入 planner。
3. 冻结二者为诊断负结果，只保留其证据：provider 误差需要状态条件化或机制级修订，route-goal 机制应转化为 learned target 或 commitment feature，而不是直接手工上线。

**结论**
采用选项 3。仿射校准的排序改善过小，不满足进入闭环的最低证据；route-goal option 虽有较高 first-push tier 命中，但没有提供超过 non-route 候选的 option-level teacher-Q 上界。直接上线会把诊断信号误用为手工 planner 偏置。

**影响**
阶段 3 后续路线继续保持“离线门限先行”。任何 provider 修订必须报告 value-consumption root sorting 指标，而不仅是平均 stock MAE；任何 route/goal option 修订必须先证明其候选或学习分数能提升 teacher-best tier 命中、降低 critical miss 或提供新的 teacher-Q 上界。未满足该条件前，不扩展 held-out `6` 关卡，不运行大预算闭环，不把该机制写作阶段 3 候选。

## 决策 45

**决策**
保留 first-push commitment target 作为后续可学习目标和 route/goal commitment 研究方向，但当前不训练 learned first-push prior，不接入在线 SD-MCTS，也不把 route-goal structural score 手工上线。

**背景**
route-goal monotone option 直接作为 MCTS child 的审计失败，但其结构 top-1 的 first-push tier 命中较高。为检验该信号是否可转化为学习目标，新增 `audit_sokoban_first_push_commitment_signal.py`。该审计将 primitive push、target2 local structural option 和 route-goal monotone option 放入同一离线候选集合，并用

```text
Q_commit(s, a_first) = max_o Q_teacher(s, o), first(o) = a_first
```

把 option-level teacher-Q 聚合回第一步 primitive push。teacher 只用于 stage-2 train split 内部的 option successor 离线标注，不在线进入 SD-MCTS，不参与候选生成，也不修改 transition provider、planner feedback、TVA 或 MCTS child 集合。

holdout-50 smoke 和 12-level 扩展审计均通过数据充分性门限。扩展审计采集 train states `22`、valid states `17`，train best first-push labels `25`，valid best first-push labels `25`；valid top-tier hit 为 `0.47058823529411764`，valid top-1 exact agreement 为 `0.47058823529411764`，valid Q corr 为 `0.8319232992067614`，valid route-goal available fraction 为 `0.38181818181818183`，valid teacher-best route-goal source labels 为 `11`。

但完整对照显示，当前 75 维 ridge commitment score 还不是 planner-ready。扩展审计中，route-goal structural score 的 valid top-tier hit 为 `0.7647058823529411`，route-goal count baseline 也为 `0.7647058823529411`，均高于 ridge 的 `0.47058823529411764`；ridge 的 mean teacher regret `2.2574965213611655` 也高于 route-goal structural score 的 `1.1701865437235404`。因此，数据充分性 gate 通过只能说明信号存在，不能说明学习器已经优于简单结构基线。

**选项**
1. 因 gate 为 `PASS`，直接训练 learned first-push prior 并作为 prior multiplier 接入 SD-MCTS。
2. 将 route-goal structural score 或 route-goal count 直接作为在线启发式 prior。
3. 保留 first-push commitment target，新增相对基线的 planner-readiness 要求；只有学习分数在内部 holdout 上不劣于主要在线可计算结构基线，并满足足够 state 覆盖、top-tier hit、critical miss 或 regret 门限后，才允许进入默认关闭的 prior smoke。

**结论**
采用选项 3。该结果支持 route/goal commitment 是有研究价值的高层目标抽象，但不支持把当前 ridge probe 或结构分数直接上线。直接接入会把 SD-MCTS 退化为手工 first-push 启发式，且与“先从简化、可靠、可审计的 SD 模型逐步丰富”的路线不一致。

**影响**
下一阶段若继续 first-push commitment，应先改进离线学习器，例如减少特征维度、增强正则化、按 route-goal/primitive 分层校准或引入更明确的目标分配特征。进入在线 SD-MCTS 的最低条件为：一，仍保持 no online teacher/best-first、transition provider unchanged；二，内部 holdout valid rank states 至少达到当前 `17` 的量级；三，learned score 的 top-tier hit、critical miss 或 regret 不劣于 route-goal structural/count 等在线可计算结构基线；四，prior 只能默认关闭，并先通过 2-level budget `48` 非零成功率 smoke，之后才可扩展 held-out `6`。

## 决策 46

**决策**
Sokoban 新主线采用 exact grid stock-flow SD provider `exact_grid_stock_flow_v1`，冻结六存量 learned-flow 作为负结果和诊断材料；SD 与 TVA 必须分别通过独立门禁后，才允许进入 closed-loop SD-MCTS。

**背景**
阶段 3 的多轮结果显示，六存量 learned-flow 路线同时存在抽象不闭合、动作后果排序不足和闭环消费失败。`oracle_abstract_sdmcts` 仍无法恢复成功率，stock-only aliasing 显示同一六维抽象存在不可约后继差异，root sorting 审计中 learned provider 一步备份 top-1 不能进入 teacher-best tier。复盘后确认，Sokoban 的物理 push 转移精确已知，SD 层不应继续学习一个有损低维转移；求解抽象性应进入事件粒度、反馈/约束和 TVA，而不是进入可 rollout 状态。

本轮实现了 `SokobanGridStockFlowModel`。其存量为格点箱子占用 `b` 和玩家占用 `p`，flow 为一次合法 push 对应的一条有向 floor edge，转移为 `b_next=b+Bf`。同时新增 SD 独立审计 `validate_sokoban_grid_stock_flow_design.py`、TVA 独立审计 `audit_sokoban_grid_tva.py` 和 exact-grid SD-MCTS 闭环入口 `evaluate_sokoban_grid_sdmcts.py`。闭环入口支持 `--require-independent-gates`，可强制读取 SD 与 TVA 审计文件，门禁未通过则拒绝运行。

当前 smoke 结果为：SD 审计 `grid_sd_gate=PASS`，检查 `3` 个关卡、`15` 个状态、`64` 个合法 push 转移，全部守恒与 `Bf` 一致性违例为 `0`；TVA 审计 `tva_independent_gate=PASS`，valid count `16`，MAE `1.3878170000076102`，corr `0.9736159851420181`，sign accuracy `1.0`，root sorting top-tier hit `5/5`；闭环 zero-leaf 与 TVA-leaf 的 3-level smoke 均为 `1/3`。

**选项**
1. 继续在六存量 learned-flow provider 上调 prior、TVA、option 和 feedback，以求恢复闭环成功率。
2. 在 exact push 动力学上保留一个人为 learned transition 参数，使 Sokoban 叙事表面上仍是“学习 SD 参数”。
3. 将 SD 核心改为 exact grid stock-flow，学习只进入 TVA 和后续可审计的 option/commitment 组件；所有闭环候选必须先通过 SD 与 TVA 独立门禁。

**结论**
采用选项 3。该选择使 Sokoban 章节回到论文框架的可检验分工：SD 提供零误差可行流形和可审计反馈约束，MCTS 在真实状态上展开，TVA 负责补偿有限深度截断偏差。六存量 learned-flow 的产物保留为“为什么不能在有损抽象上规划”的负结果证据，不再作为主线调参对象。

**影响**
后续 Sokoban 实验的默认入口改为 `evaluate_sokoban_grid_sdmcts.py`。任何 TVA checkpoint 必须先由 `audit_sokoban_grid_tva.py` 或同等级独立审计证明离线有效；任何 SD provider 修改必须先通过 `validate_sokoban_grid_stock_flow_design.py` 的守恒、`Bf` 和 canonical key 门禁。正式结果应报告 `H x budget x TVA on/off` 矩阵，并明确区分“接口通过 smoke”和“TVA 相对 zero-leaf 有性能增益”这两类结论。

## 决策 47

**决策**
GFT-SD 第二层资源开发采用 10 关正式预算作为内循环门槛；`K`（区域容量裕度）保留为当前正向候选，`Q`（顺序依赖债务）第一切片只保留为 active readout 和审计变量，不进入推荐在线先验配置。

**背景**
50 关 heldout 单轮成本过高，不适合作为每个二层资源的开发内循环。当前同一 10 关正式预算切片已用于比较默认 GFT、`K` 和 `Q`：默认 GFT 为 `6/10`，状态分布 `deadlocked=1,max_steps=2,no_action=1,solved=6`；`K` 在 `capacity_weight=0.25` 下为 `7/10`，未新增 deadlocked；`Q` 在 `ordering_weight=0.25` 与 `0.1` 下均为 `6/10`，且均使 `level_index=5` 从 `max_steps` 退化为 `deadlocked`，deadlocked 从 `1` 增至 `2`。

`Q` 的读出覆盖并非为空：50 关 heldout 初始动作审计中，`223` 个初始合法动作有 `60` 个触发非零 `Q` 变化，`38/50` 关至少有一个动作触发变化。但该覆盖没有转化为 10 关闭环收益，且引入新增死锁风险。随后运行 `K+Q` 组合交互测试，`capacity_weight=0.25,ordering_weight=0.05` 为 `6/10`，没有新增死锁，但把 `K` 单独解出的 `level_index=5` 重新推回 `max_steps`，因此也未通过保持 `K` 增益的组合门槛。

**选项**
1. 继续在 `Q` 上做更密集的权重搜索，以寻找可能的正向点。
2. 将 `Q` 与 `K` 组合后直接作为下一候选，绕过单因素失败结论。
3. 保留 `Q` 第一切片为默认关闭的读出和审计变量，不作为推荐先验；后续优先改进偏序构图或推进目标承诺读出，仍以 10 关正式预算为开发门槛。

**结论**
采用选项 3。当前 `Q` 第一切片的结构解释是合理的，但在线消费证据不足；继续小步调权容易把问题转化为手工先验搜索。当前推荐候选仍为 `K` 的 `capacity_weight=0.25`，完整 50 关最强结果仍以默认 GFT/grid-equivalent prior 的 `41/50` 为准。

**影响**
后续 `K/Q/C` 或其他二层资源进入在线规划前，必须先通过同一 10 关正式预算门槛：默认权重为 `0` 时不改变当前最佳复现路径；非零权重至少不劣于默认线；不得新增明显死锁退化。50 关 heldout 只用于阶段性确认，不再作为每个资源切片的常规开发循环。

## 决策 48

**决策**
GFT-SD 动作先验采用统一资源变化接口；`P`（运输进度）不再在 GFT 规划器层通过旧 grid `sd_delta` 快路径特殊处理。现有 `sd_heuristic` 叶值保留为零学习历史基线，但不作为最终 TVA-MCTS 架构原则。

**背景**
当前 `41/50` 最强结果中，`P` 通过 matching lower bound 同时进入叶值和动作先验，这是历史实现路径形成的有效基线。但 GFT-SD 的目标是统一二层资源系统：`P/M/A/K/D/Q/C/I` 均由底账状态 `y_t` 读出，并通过 `Delta z_t` 进入动作先验、TVA 特征与诊断。若保留 `P` 在规划器中的特殊通道，后续 `K/Q/C` 的组合测试和 TVA-MCTS 重建会继续继承不一致接口。

**选项**
1. 保留 `P` 的特殊快路径，以最大限度复用已验证 grid `sd_delta` 实现。
2. 立即删除 `sd_heuristic` 叶值，强制所有叶节点价值等待 TVA 接入。
3. 在动作先验层统一调用 `SokobanGFTSDModel.gft_action_prior_rows()`，保留 `sd_heuristic` 作为显式历史基线和对照项。

**结论**
采用选项 3。该选择将在线先验消费位置统一到 GFT 模型层，同时保留当前可复现最强路径所需的零学习叶值基线。默认 3 关等价 gate 显示，统一接口后 aggregate、逐关字段和 root trace mismatch 均为 `0`，说明该修订改变架构边界而不改变默认行为。

**影响**
后续组合测试中，`P/K/Q/C` 等资源均应通过同一 `gft_action_prior_rows()` 接口进入动作先验，不再在规划器层手写资源分量。未来 TVA 接入时，应把统一 GFT 资源向量作为 TVA 特征来源之一，并将 `sd_heuristic` 作为需被 TVA 超越的零学习对照，而不是最终叶值设计。

## 决策 49

**决策**
`C`（目标承诺清晰度）第一切片作为 active readout 和 TVA/诊断候选特征保留，但不作为单独推荐在线先验配置。`K+C` 组合通过 10 关不劣于门槛，但相对 `K` 单独没有新增收益；当前二层资源推荐候选仍为更简洁的 `K` 单独 `capacity_weight=0.25`。

**背景**
`C` 采用箱子到目标的静态 reverse-pull 成本矩阵，经 Sinkhorn 归一化得到软箱子-目标分配，再由归一化熵计算承诺清晰度。50 关 heldout 初始动作审计显示，`223` 个初始合法动作中有 `155` 个触发非零 `C` 变化，`32/50` 关至少有一个动作触发变化，说明该读出具有覆盖度。

但 10 关正式预算门槛显示，`commitment_weight=0.5` 和 `0.25` 均为 `6/10`，未超过默认 GFT 的 `6/10`，且均使 `level_index=5` 从 `max_steps` 退化为 `deadlocked`，deadlocked 从 `1` 增至 `2`。同一门槛下，`K` 单独为 `7/10` 且未新增 deadlocked；`K+Q` 组合为 `6/10`，抵消了 `K` 在 `level_index=5` 上的增益。补充 `K+C` 组合后，`capacity_weight=0.25,commitment_weight=0.25` 为 `7/10`，deadlocked 仍为 `1`，逐关 solved/status/pushes 与 `K` 单独完全一致。

**选项**
1. 继续对 `C` 做更密集的权重搜索，寻找可能不触发死锁的弱权重点。
2. 将 `C` 与 `K` 的等价组合视为新推荐配置，因为它至少不劣于 `K`。
3. 保留 `C` 为默认关闭的读出和 TVA/诊断特征；当前推荐仍取更简洁的 `K` 单独配置，后续若使用 `C`，优先作为 TVA 输入或学习目标，而不是继续扩大手工在线先验权重搜索。

**结论**
采用选项 3。当前 `C` 读出有覆盖和解释意义，但直接进入动作先验会引入新增死锁风险；与 `K` 组合后虽不再退化，但也没有超过 `K` 单独配置。继续搜索手工权重容易偏离统一资源接口的研究目的，也会增加 10 关门槛成本。

**影响**
短期推荐配置保持为默认 GFT/grid-equivalent prior 或 `K` 单独候选；`Q`、`C` 单独先验不进入推荐在线配置，`K+C` 仅记录为与 `K` 等价的组合结果。后续若重建 TVA-MCTS，`C` 应作为统一 GFT 资源向量的一部分进入 TVA 特征，并通过 TVA 训练和叶值消费来验证其价值，而不是直接扩大手工 prior 权重。

## 决策 50

**决策**
按 `sokoban_gptpro.md` 完整补齐 GFT-SD 结构接口，但默认在线配置保持行为等价；新补齐的运输成本扩展、软死锁压力、瓶颈 overlap、系统动力学奖励、系统动力学值函数和 TVA 特征接口先作为显式结构与诊断通道存在。当前推荐候选仍为默认 GFT/grid-equivalent prior 与 `K`（区域容量裕度）单独 `capacity_weight=0.25`。

**背景**
此前 GFT-SD 已实现两层 OO 架构、八个资源库存和统一动作先验，但与 `sokoban_gptpro.md` 的完整定义仍有缺口：`P` 只实现为 matching lower bound，`D` 只有硬死锁债务，资源流量、`r_SD`、`V_SD(z)` 和 TVA 特征向量没有作为一等接口落盘，`I` 的动作相关性也缺少瓶颈集合与动作影响域 overlap 的显式记录。

本轮补齐上述接口。`P` 成本矩阵实现为 `d_g+lambda_turn*turns+lambda_access*access_risk+lambda_zone*zone_risk`，默认三个 `lambda` 为 `0`，因此默认退化为原 matching lower bound。`C` 改为读取同一成本矩阵做 Sinkhorn 分配熵。`D` 改为硬死锁债务加软死锁压力，软压力由机动性损失、匹配脆弱性、容量压力、可达性损失和顺序违例构成，默认五个 `alpha` 为 `1.0`。新增 `gft_resource_vector()`、`gft_resource_flow()`、`gft_sd_reward()`、`gft_sd_value()`、`gft_tva_feature_dict()` 和 `framework_contract.gft_spec_alignment`。

验证显示，默认 3 关 gate 与上一版 aggregate、逐关字段和 root trace mismatch 均为 `0`；默认 10 关正式预算仍为 `6/10`，与旧默认输出 aggregate、逐关字段和 root trace mismatch 均为 `0`；`K=0.25` 10 关正式预算仍为 `7/10`，与旧 `K=0.25` 输出 aggregate、逐关字段和 root trace mismatch 均为 `0`。

**选项**
1. 将新补齐的运输成本扩展、软死锁和瓶颈 overlap 立即以非零权重接入默认在线先验。
2. 只实现文档中已产生性能收益的部分，继续把奖励、值函数和 TVA 特征接口留到未来。
3. 完整补齐设计文档接口，但默认权重保持当前最强行为路径；任何新增非零消费都必须按同一 10 关正式预算门槛单独验证。

**结论**
采用选项 3。该选择同时满足设计文档完整性和可复现性约束：结构层已与文档对齐，结果层不把未经门槛验证的新手工权重混入当前最强结论。旧六存量 learned-flow 负结果仍不得作为否定 GFT-SD 两层设计的证据；它只说明低维抽象不能脱离精确底账层承担闭合转移 provider。

**影响**
后续 TVA-MCTS 重建应优先读取 `gft_tva_feature_dict()` 和 `gft_sd_value()`，而不是继续为单个资源建立特殊通道。若要启用非零 `lambda_turn/access/zone`、软死锁在线权重或瓶颈 overlap，必须先完成结构审计、默认等价回归，再进入 10 关正式预算门槛；只有不劣于默认线且不新增明显死锁退化，才可安排 50 关阶段性确认。

## 决策 51

**决策**
GFT-SD 全因素测试采用“无严重恶化则保留”的判据。当前二层资源结构完整保留；手工在线先验的 10 关开发候选更新为运输成本三项组合 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25`，以及在其上加入 `ordering_weight=0.05` 和可选 `commitment_weight=0.05`。瓶颈 overlap 的当前手工先验默认关闭，但保留为诊断读出和未来可学习权重候选。

**背景**
用户指出，若 GFT-SD 的八个资源因素可能存在协同作用，则逐项切片不应采用“单因素未涨分即删除”的筛选逻辑。该逻辑会把当前手工接入方式、当前权重和当前搜索预算的局限误判为结构因素本身无效，尤其不适合尚未进入可学习权重和 TVA 叶值阶段的方案。因而，本轮补充单因素、强底座逐项叠加、全包和弱瓶颈收口测试。

统一 10 关正式预算中，默认 GFT 为 `6/10`，`K=0.25`、`P_access=0.25`、`P_zone=0.25` 均为 `7/10`。运输成本三项组合 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25` 达到 `8/10`，同时解出默认失败的 `level_index=5` 和 `level_index=6`。在该强底座上加入 `ordering_weight=0.05` 仍为 `8/10`，再加入 `commitment_weight=0.05` 仍为 `8/10`。相反，加入 `K/M/A/D/C` 的若干手工权重通常回到 `7/10`，加入瓶颈 overlap `0.10` 回到 `6/10`，弱瓶颈 `0.02` 也未保持 `8/10`。保守全包和低权重全包结果为 `7/10` 或 `6/10`，没有显示完整结构崩塌，但显示手工瓶颈 overlap 当前不适合默认开启。

**选项**
1. 沿用早期单因素筛选逻辑，删除或边缘化 `Q/C/I` 等单独未涨分的资源。
2. 直接把所有因素以保守权重全部默认开启，忽略组合中暴露的稀释和瓶颈 overlap 退化。
3. 保留完整二层结构；把单因素测试定位为安全门和诊断；以组合证据选择当前手工在线候选，并把未形成正向证据但无严重破坏的因素留到可学习权重和 TVA 阶段。

**结论**
采用选项 3。当前最强 50 关正式结果仍是默认 GFT/grid-equivalent prior 的 `41/50`；本轮只更新 10 关开发候选，不直接改写 50 关 heldout 结论。当前 10 关最好候选为运输成本三项组合、运输成本三项加顺序依赖债务、运输成本三项加顺序依赖债务和目标承诺清晰度，三者均为 `8/10`。`K/M/A/D/C` 等因素不因当前手工叠加回到 `7/10` 而被删除；瓶颈 overlap 不进入当前手工推荐配置，但仍作为 `I` 的结构组成保留。

**影响**
后续默认叙述必须区分“结构保留”和“手工在线默认开启”。正式模型结构仍包含 `P/M/A/K/D/Q/C/I` 八个资源、显式资源流量、系统动力学奖励、系统动力学值函数和 TVA 特征接口。若进行下一轮 50 关阶段性确认，应优先测试 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25`，以及加入 `ordering_weight=0.05,commitment_weight=0.05` 的候选。若后续重建 TVA-MCTS，`Q/C/I` 等因素应通过统一特征和可学习权重重新评估，而不是被早期手工先验切片排除。

## 决策 52

**决策**
GFT-TVA v1 作为严格 TVA 接入基线保留，但不进入推荐在线配置，也不安排 50 关正式确认。当前正式最强结果仍为 `sd_heuristic + GFT action prior` 的 50 关 `41/50`；当前 10 关开发最好候选仍为强运输成本组合及其 `Q/C` 扩展的 `8/10`。

**背景**
设计文档已经明确 TVA 的位置：`V_leaf(z_h)=V_SD(z_h)+V_TVA(z_h)`，其中 TVA 学习显式 SD 世界模型无法覆盖的长期可解性残差。本轮实现了 GFT-TVA v1：训练脚本从 GFT-MCTS 根节点统计构造 `V_search=sum_a p(a)Q_H(s,a)`，以 `V_search-V_SD` 为 ridge 残差目标；闭环入口支持 GFT-TVA checkpoint，并在 `--leaf-value tva` 下执行 `V_SD+V_TVA`。训练与闭环均不使用 diagnostic teacher 或 best-first teacher。

实现过程中发现，`gft_sd_value()` 的默认资源量纲和非有限运输进度惩罚不能直接作为 MCTS 回报。为避免未校准值函数破坏搜索，闭环叶值把显式 `V_SD` 投影到 `[deadend_value, goal_value]`，训练残差使用同一投影。开发版训练使用关卡 `8-47`，与 10 关评测 `48-57` 不重叠；样本数 `204`，train `169`、valid `35`，gate 为 `PASS`，但 valid corr 仅 `0.056`。

闭环结果显示，强 GFT 先验加 `sd_heuristic` 叶值为 `8/10`；替换为显式 `gft_sd` 叶值后仅 `1/10`，主要失败为 `no_action`；使用 `V_SD+TVA` 后为 `0/10`，`no_action=9`。这说明当前 `V_SD` 默认权重和量纲尚未成为 planner-ready 叶值，ridge TVA 残差也未能修复该问题。

**选项**
1. 因为 GFT-TVA 接口已接通，继续扩大训练集或直接跑 50 关，期待规模改善。
2. 放弃 TVA，回到纯手工先验和 `sd_heuristic` 叶值。
3. 保留 GFT-TVA v1 为严格接入负结果；后续先校准显式 `V_SD` 或改做 `sd_heuristic` 锚定残差，再重新进入 10 关门槛。

**结论**
采用选项 3。GFT-TVA v1 证明了统一 GFT 特征、search-value 目标、checkpoint 合同和闭环叶值消费已经连通；但闭环效果显著劣于当前强基线，不能作为性能候选。该结果不否定 TVA 架构，只否定当前“默认 `V_SD` + ridge 残差”的第一实现。

**影响**
后续若继续 TVA，应先完成 `V_SD` 的 `beta` 权重/值尺度校准，使 `gft_sd` 叶值至少接近 `sd_heuristic` 的 10 关表现；或先训练 `sd_heuristic_residual` 作为过渡，验证残差学习能否在强基线上提供增益。任何新 TVA 变体必须先通过同一 10 关开发门槛，再考虑 50 关阶段性确认。

## 决策 53

**决策**
GFT-TVA 后续训练优先锚定 `discounted_progress` 口径的显式 GFT-SD 叶值；`linear_projected` 口径保留为负结果对照，不作为下一版 TVA 的默认基底。第二层资源中的区域容量裕度、顺序依赖债务和目标承诺清晰度可以以保守权重进入叶值候选，但在扩大权重扫或重训 TVA 前，应优先补充第二层特征缓存。

**背景**
GFT-TVA v1 闭环失败的直接原因之一是显式 `V_SD` 叶值本身不可用：`linear_projected` 模式下，强 GFT 先验 10 关仅 `1/10`，`V_SD+TVA` 进一步降为 `0/10`。这不能被解释为 TVA 架构无效，因为残差学习的基底已经明显偏离当前强对照。

本轮新增 `discounted_progress` 叶值模式，把 GFT 运输进度读出映射到 MCTS 折扣回报尺度，并在叶值资源权重全为 `0` 时跳过完整 TVA 特征构造。10 关强底座结果恢复为 `8/10`，且与 `sd_heuristic` 强底座逐关 status 和 pushes 完全一致。进一步加入保守第二层叶值校正，即区域容量裕度、顺序依赖债务和目标承诺清晰度权重均为 `0.05`，仍保持 `8/10`，失败关卡不变，但运行时间从 `488.25s` 增至 `550.80s`。

**选项**
1. 继续沿用 `linear_projected` 叶值训练 TVA，通过扩大训练样本或模型容量弥补闭环失败。
2. 完全放弃显式 GFT-SD 叶值，改回只围绕 `sd_heuristic` 做残差学习。
3. 将 `discounted_progress` 作为显式 GFT-SD 叶值的当前校准口径，在该基底上重新训练 TVA；同时保留 `sd_heuristic` 作为强对照，不把第二层资源的保守叶值校正视为最终权重。

**结论**
采用选项 3。`discounted_progress` 证明显式 GFT-SD 叶值可以恢复到当前强 10 关对照，因而具备作为 TVA 残差基底的资格。`linear_projected` 的负结果继续保留，但其结论限定为“未校准资源线性值不适合作为 MCTS 叶值”，不能外推为 GFT-SD 或 TVA 结构无效。

**影响**
下一版 TVA 训练若继续使用 `V_leaf=V_SD+V_TVA`，应以 `--gft-sd-leaf-mode discounted_progress` 收集 search-value 残差。区域容量裕度、顺序依赖债务和目标承诺清晰度在叶值中以 `0.05` 保守权重共存，不构成严重退化证据；但由于非零资源权重会增加在线计算成本，后续在扩大权重组合、训练 TVA 或安排 50 关确认前，应先实现或验证第二层特征缓存。正式 50 关最强结果暂不改变，仍为 `sd_heuristic + GFT action prior` 的 `41/50`。

## 决策 54

**决策**
将 GFT-SD 第二层特征缓存作为后续 TVA 训练和闭环评估的默认工程前提。缓存修复完成后，可以回到 `discounted_progress` 显式叶值基底上的 TVA 训练；短期不再因非零第二层叶值权重的重复特征组装成本而阻塞。

**背景**
显式叶值校准显示，`discounted_progress` 零资源权重可以恢复到强 10 关对照的 `8/10`；加入区域容量裕度、顺序依赖债务和目标承诺清晰度各 `0.05` 后仍为 `8/10`，但运行时间从 `488.25s` 增至 `550.80s`。这表明第二层因素进入叶值本身没有造成严重退化，但在线计算存在重复构造资源向量和 TVA 特征合同的问题。

本轮在模型层缓存 `gft_resource_vector(state)` 与 `gft_tva_feature_dict(state)`，在规划器层缓存 `gft_sd_leaf_value(state)` 与 GFT-TVA 预测值。缓存键使用 canonical state，保持与现有 SD 叶值缓存一致。低预算 smoke 通过；10 关保守第二层叶值组合复测仍为 `8/10`，与缓存前 aggregate 和逐关关键字段完全一致，wall time 从 `550.80s` 降至 `481.41s`。

**选项**
1. 不引入缓存，继续接受每次非零第二层叶值权重带来的额外耗时。
2. 只缓存底层资源分量，不缓存完整资源向量、TVA 特征字典和叶值。
3. 在模型层和规划器层同时做行为保持型缓存，并用同一 10 关门槛确认行为等价和耗时收益。

**结论**
采用选项 3。该修复不改变 GFT-SD 结构、不改变结果 JSON 合同、不改变 TVA 特征顺序，也不改变当前正式最强 50 关结论。缓存后同配置行为完全一致，耗时降至低于上一轮零资源权重基底的水平。

**影响**
下一步可重新训练 GFT-TVA，训练基底应采用 `--gft-sd-leaf-mode discounted_progress`，并可保留区域容量裕度、顺序依赖债务和目标承诺清晰度的保守叶值权重作为候选。若后续仍需进一步降耗，应审计 MCTS 展开形态、动作先验计算和候选扩展策略，而不是继续围绕 TVA 特征合同组装做局部优化。

## 决策 55

**决策**
GFT-TVA v2 作为“discounted-progress 基底下的全强度 ridge 残差”记录为闭环负结果，不进入推荐配置，不安排 50 关确认。下一步 TVA 工作应转向残差安全校准，包括残差缩放、残差裁剪、非正残差门控或排序保持型 safety gate。

**背景**
v1 的主要问题是显式 `V_SD` 基底本身不可用。经过 `discounted_progress` 校准和第二层特征缓存后，显式 `V_SD` 在 10 关强底座上恢复到 `8/10`。因此本轮重训 GFT-TVA v2，训练目标仍为 `V_search-V_SD`，但 `V_SD` 改为 `discounted_progress` 并加入保守第二层叶值权重 `K/Q/C=0.05`。

v2 离线指标明显优于 v1：valid corr 从 `0.056` 提高到 `0.668`，说明新基底使残差更可拟合。但 10 关闭环中，显式 `V_SD` 为 `8/10`，`V_SD+TVA v2` 仅为 `1/10`，状态分布为 `deadlocked=8,max_steps=1,solved=1`。逐关看，v2 只保留 `level_index=4` 解出，并将原本可解的 `1,2,3,5,6,7,8` 推入 deadlocked。初始状态残差预测范围约 `[-0.695,-0.114]`，均值约 `-0.425`，没有单点数值爆炸；问题在于负残差在闭环中持续改变状态排序。

**选项**
1. 因为 v2 离线 valid corr 较高，直接扩大训练集或跑 50 关确认。
2. 放弃 TVA，长期保留 `sd_heuristic` 或显式 `V_SD` 作为唯一叶值。
3. 保留 v2 为负结果，承认全强度残差不安全；下一步先加入残差缩放、裁剪或门控，使 TVA 至少不破坏显式 `V_SD` 的 10 关表现。

**结论**
采用选项 3。离线相关性不能替代闭环门槛；当前 v2 证明了 residual signal 可拟合，但没有证明 residual 可安全消费。该结论不否定 TVA 架构，只否定“无缩放、无裁剪、无门控的 ridge 残差直接全强度加入叶值”。

**影响**
后续 TVA 变体必须先以 10 关门槛证明不低于显式 `V_SD` 的 `8/10`，并不得显著增加 deadlocked。推荐的下一小步不是增加训练样本，而是在评估入口中引入默认关闭的 TVA 残差安全参数，例如残差缩放系数、残差裁剪区间或只允许非负/低不确定性残差介入的 gate。正式 50 关最强结论仍不变，仍以 `sd_heuristic + GFT action prior` 的 `41/50` 为准。

## 决策 56

**决策**
将 GFT-TVA 的默认实现口径从残差 TVA 修订为直接终端价值 TVA。后续新训练默认使用 `target_mode=direct_value`，即 `V_psi` 直接学习 `V_search`；残差模式 `gft_sd_residual` 仅作为可复现负结果和对照实验保留。闭环默认叶值为 `V_leaf=V_psi`，并通过 `--tva-value-blend` 支持受控的 `V_SD` 兜底混合。

**背景**
GFT-TVA v2 证明 `discounted_progress` 显式 `V_SD` 可以作为稳定叶值，但全强度 ridge 残差 `V_leaf=V_SD+V_TVA` 在 10 关中从 `8/10` 降至 `1/10`，并显著增加 deadlocked。复核 NeurIPS 论文公式以及 Beer Game、Cartpole 已完成实现后，确认更一致的 TVA 语义是让 `V_psi(s)` 直接对齐搜索值 `V_search(s)`，并在叶节点直接消费该价值网络，而不是默认学习相对 `V_SD` 的残差。

本轮实现完成两处修订。训练入口 `train_sokoban_gft_tva.py` 新增并默认采用 `--target-mode direct_value`；历史名称 `value` 作为兼容别名，写入 checkpoint 时规范为 `direct_value`。评估入口 `evaluate_sokoban_gft_sdmcts.py` 对 `direct_value` checkpoint 默认返回 `V_psi`；当 `--tva-value-blend=alpha<1` 时，执行 `V_leaf=(1-alpha)V_SD+alpha V_psi`。纯直接价值模式不要求运行时显式 `V_SD` 基底与 checkpoint 合同一致；只有残差模式或混合模式使用 `V_SD` 时才检查 `gft_sd_leaf_mode` 和 `gft_sd_leaf_weights`。

小规模 smoke 已通过：direct-value 训练输出 gate 为 `PASS`，checkpoint 记录 `target_mode=direct_value`、`direct_value_formula=V_psi=V_search` 和 `leaf_formula=V_leaf=V_psi`；1 关闭环 smoke 分别验证了纯 `V_psi` 与 `alpha=0.5` 混合叶值路径可执行并落盘。该 smoke 只验证接口，不作为性能结论。

**选项**
1. 沿用残差 TVA 作为默认路线，并通过缩放、裁剪或门控修复 v2 负结果。
2. 完全删除残差实现，只保留直接价值 TVA。
3. 将直接终端价值 TVA 设为默认主线；保留残差实现作为负结果复现和对照；通过 `--tva-value-blend` 提供显式 `V_SD` 的可控兜底。

**结论**
采用选项 3。直接终端价值 TVA 与论文和既有成功实验更一致，应成为后续 Sokoban GFT-TVA 的默认口径。残差 v2 的负结果继续保留，但不再作为否定直接价值 TVA 的依据。显式 `V_SD` 继续作为系统动力学读出、零学习叶值对照和混合兜底，不被删除。

**影响**
下一轮正式实验应训练 direct-value checkpoint，并在 10 关开发切片上比较 `--tva-value-blend 1.0` 与保守混合值；只有通过 10 关门槛后才考虑 50 关确认。文档和结果解读中必须明确区分 `V_SD` 读出、直接终端价值 `V_psi` 和残差 `V_TVA`，不得把残差负结果外推为直接价值 TVA 的失败证据。当前正式 50 关最强结果仍为 `sd_heuristic + GFT action prior` 的 `41/50`。

## 决策 57

**决策**
GFT-TVA direct-value v3 记录为当前实现形态下的闭环负结果，不进入推荐配置，不安排 50 关确认。下一步 TVA 工作不再围绕同一 ridge state-value checkpoint 做无界 `alpha` 混合扫描，而应优先修订学习器容量和 target 构造。

**背景**
决策 56 将 GFT-TVA 默认口径修订为直接终端价值 TVA。随后训练 direct-value v3：训练目标为 `V_psi=V_search`，训练关卡为 `8-47`，评测切片仍为 `48-57`。离线指标通过 gate，valid MAE 为 `0.752`，valid corr 为 `0.710`，target 为正值搜索值，train target mean 为 `3.838`。这说明当前模型并非没有拟合离线 search-value target。

闭环结果为负。纯 `V_leaf=V_psi` 在 10 关中为 `1/10`，状态分布 `max_steps=8,no_action=1,solved=1`，耗时 `real=1504.46s`。保守混合 `V_leaf=0.75V_SD+0.25V_psi` 同样为 `1/10`，状态分布 `max_steps=7,no_action=2,solved=1`，耗时 `real=1360.77s`。两者只保留 `level_index=4` 解出；显式 `V_SD` 对照在同配置下为 `8/10`。失败形态与 residual v2 不同，direct-value v3 主要表现为长时间 `max_steps/no_action`，而不是大量 deadlocked。

**选项**
1. 因为 direct-value v3 离线相关性较高，继续跑更多 `alpha` 值或直接扩大到 50 关。
2. 放弃直接终端价值 TVA，回退到残差 TVA 或永久只保留显式 `V_SD`。
3. 将 direct-value v3 的当前具体实现冻结为负结果；保留直接终端价值 TVA 的接口口径；下一步改变学习器和 target，而不是继续调同一 checkpoint 的消费权重。

**结论**
采用选项 3。direct-value v3 证明“论文一致口径 + 当前 ridge/GFT 特征/search-value target”仍不足以在闭环中稳定改善 MCTS。该结果不否定直接终端价值 TVA 的架构位置，但否定当前具体实现进入推荐在线配置。

**影响**
后续 TVA 若继续推进，应优先考虑非线性价值模型、on-policy 闭环状态再标注、成功/失败分层采样、TD bootstrapping、动作排序辅助损失或局部结构辅助目标。任何新版本仍需先在同一 10 关门槛上不低于显式 `V_SD` 的 `8/10`，且不得显著增加耗时或 no-action。当前正式 50 关最强结果仍为 `sd_heuristic + GFT action prior` 的 `41/50`。

## 决策 58

**决策**
将 GFT-TVA 独立截断价值审计设为后续 TVA 接入 MCTS 的前置门槛。GFT-TVA direct-value v3 在该审计下判定为 `FAIL`，因此不进入推荐配置，也不再通过混合权重继续尝试闭环接入。

**背景**
用户指出，TVA 应像 SD 一样先被独立验证，确认其准确性后再接入规划框架。此前 direct-value v3 只有训练/验证集拟合、接口 smoke 和 10 关闭环结果；这些证据不能区分 TVA 本身不准确、标签构造不合适和 MCTS 消费方式不合适。

本轮新增 `audit_sokoban_gft_tva_truncation_value.py`。该脚本用显式 `V_SD` 叶值的 GFT-SD-MCTS 采样实际叶节点状态，TVA 不参与动作选择；随后使用离线 exact-state best-first teacher 从叶节点继续搜索，生成截断剩余价值标签。高置信标签只包括 teacher solved、proven dead-end 和 no legal actions；teacher budget unsolved 单独报告，不作为高置信 gate。

10 关正式审计采集 `120` 个叶节点，其中 high-confidence `49` 个。TVA 在高置信点值 MAE 上略好于显式 `V_SD`，`0.805` 对 `0.813`，但排序准确率低于显式 `V_SD`，`0.800` 对 `0.827`，因此 gate 为 `FAIL`。高预算复核采集 `80` 个叶节点，其中 high-confidence `41` 个；TVA MAE `0.822` 优于显式 `V_SD` 的 `0.881`，但排序准确率 `0.821` 仍低于显式 `V_SD` 的 `0.866`，gate 仍为 `FAIL`。预算未解组中 TVA 保持明显正值预测，显示其对难解叶节点偏乐观。

**选项**
1. 因为 TVA 的高置信 MAE 略好于显式 `V_SD`，允许进入闭环，继续调 `alpha` 或预算。
2. 因为独立审计 FAIL，放弃 TVA 整条路线。
3. 将独立审计设为前置门槛；冻结 direct-value v3 的当前实现；保留 TVA 架构，但要求下一版先改善叶节点排序和难解状态校准。

**结论**
采用选项 3。TVA 的用途是补偿 MCTS 截断价值，而 MCTS 对 value 的主要消费是候选状态排序。direct-value v3 虽然点值 MAE 略好，但排序不如显式 `V_SD`，因此不能认为已独立准确。该结论解释了其闭环 `1/10` 和大量 `max_steps/no_action`，也说明继续调同一 checkpoint 的混合权重没有优先级。

**影响**
后续 TVA 版本必须先通过独立截断价值审计，再进入 10 关闭环门槛。训练集 valid corr、valid MAE 或接口 usage gate 不能替代该审计。下一步若继续 TVA，应优先处理叶节点排序和预算未解状态校准，例如非线性模型、on-policy 叶节点再标注、难解/失败状态分层采样、TD bootstrapping 或动作排序辅助损失。当前正式 50 关最强结果仍为 `sd_heuristic + GFT action prior` 的 `41/50`。

## 决策 59

**决策**
`teacher_budget_unsolved` 作为删失风险标签处理，不作为确定负终端价值回归标签。当前 GFT 特征线性风险探针未通过跨关卡可分性门槛，因此不训练或接入基于现有特征的 TVA 风险门控。

**背景**
独立截断价值审计显示，direct-value v3 对 budget-unsolved 叶节点明显偏乐观，但这些状态并非高置信不可解。用户指出，如果最高预算或纯搜索也无法求解，则无法获得可靠终端价值标签。因而，不能把 budget-unsolved 简单标成真实负值，否则会把搜索预算限制误写成环境事实。

本轮新增 `audit_sokoban_gft_tva_censored_risk.py`。该脚本读取 high-budget 截断价值审计结果，把 teacher solved 标为 low-risk，把 teacher budget unsolved、proven dead-end 和 no legal actions 标为 high-risk，并用 GFT 特征、TVA 预测和显式 `V_SD` 叶值做按关卡留一风险可分性探针。该探针只用于判断是否存在稳定风险信号，不输出可在线接入模型。

结果显示，样本为 `80` 个叶节点，其中 solved low-risk `41` 个、budget-unsolved high-risk `39` 个。按关卡留一的 GFT 特征 ridge 风险探针 AUC 为 `0.490`，balanced accuracy 为 `0.515`，gate 为 `FAIL`。原始 `V_SD` 和 `V_psi` 对 budget-unsolved 有一定方向性 AUC，但方向是值越高越像 budget-unsolved，与终端价值语义相反。

**选项**
1. 将 budget-unsolved 直接标为固定负值，训练下一版单标量 TVA。
2. 基于当前 GFT 特征训练线性风险头，并用于闭环门控。
3. 将 budget-unsolved 视为删失风险信号；冻结当前线性风险探针为负结果；后续先改采样、标签和模型容量。

**结论**
采用选项 3。当前证据说明 budget-unsolved 是重要风险来源，但现有 GFT 特征线性头不能跨关卡稳定识别它。把它硬标成负值或直接训练风险门控，都不具备充分证据。

**影响**
下一版 TVA 应把 solved、proven dead-end/no-legal 和 budget-unsolved 分开处理。可行方向包括提高 teacher 标注预算、使用多个离线求解器交叉确认、收集更多高置信负例、on-policy 叶节点采样、非线性模型、多头价值-风险结构和排序辅助损失。任何风险门控进入 MCTS 前，必须先在按关卡 heldout 的删失风险探针或独立截断价值审计中通过。

## 决策 60

**决策**
GFT-TVA v4 采用实际叶节点监督和删失风险处理作为后续训练基线；其中校准锚定版本通过独立截断价值审计，但闭环未超过显式 `V_SD`，因此不进入推荐配置，也不安排 50 关确认。

**背景**
direct-value v3 的独立审计显示，训练/验证相关性和点值 MAE 不足以证明 TVA 可在线消费；排序低于显式 `V_SD` 会破坏 MCTS。删失风险探针进一步说明，`teacher_budget_unsolved` 不能硬标为负值，当前线性风险头也不能跨关卡稳定识别预算未解状态。

本轮新增 `train_sokoban_gft_leaf_tva.py`。训练数据来自 GFT-SD-MCTS 实际 `leaf_value()` 调用的叶节点；离线 teacher solved、proven dead-end 和 no legal actions 作为高置信价值标签；teacher budget unsolved 只作为删失风险标签。评估器支持新的 `mlp_guarded_v1` JSON checkpoint，价值公式为 `value_head-risk_penalty*risk_probability` 与显式 `V_SD` 锚定项的组合。

全强度 MLP 风险校正版训练 gate 通过，训练样本 `323`，验证价值 MAE `0.811`、风险 AUC `0.776`，但独立截断价值审计失败：TVA 高置信 MAE `0.909`，差于显式 `V_SD` 的 `0.849`；pairwise ordering accuracy `0.583`，低于显式 `V_SD` 的 `0.809`。因此该非线性风险校正不能进入闭环。

校准锚定版本只使用训练集 teacher-solved 叶节点估计 `V_SD` 的系统偏置，得到 `anchor_bias=-0.2277`。该版本通过独立截断价值审计：TVA high-confidence MAE `0.761`，优于显式 `V_SD` 的 `0.849`；pairwise ordering accuracy 与显式 `V_SD` 相同，均为 `0.809`。但是闭环中纯校准锚定 TVA 为 `7/10`，比显式 `V_SD` 的 `8/10` 少解一关；`--tva-value-blend=0.25` 恢复为 `8/10`，但耗时 `509.54s`，高于显式 `V_SD` 的 `478.90s`，没有形成新收益。

**选项**
1. 因为校准锚定版本通过独立审计，将其提升为新默认 TVA 并扩展到 50 关。
2. 因为闭环没有超过显式 `V_SD`，放弃叶节点监督 TVA 路线。
3. 保留叶节点监督、删失风险处理和校准锚定 checkpoint 作为 TVA v4 基线；不进入推荐在线配置；下一步优先做标签缓存、排序保持残差和风险/不确定性建模。

**结论**
采用选项 3。v4 证明了新的训练与审计机制是可运行的，并产生了第一个通过独立截断价值审计的 GFT-TVA checkpoint；但它尚未带来闭环成功率或效率收益。通过独立审计是进入闭环测试的必要条件，不是成为推荐配置的充分条件。

**影响**
当前正式 50 关最强结果仍为 `sd_heuristic + GFT action prior` 的 `41/50`；当前 10 关显式 `V_SD` 对照仍为 `8/10`。后续 TVA 版本必须同时满足独立截断价值审计和 10 关闭环不低于显式 `V_SD`，才可考虑 50 关确认。下一步应避免重复昂贵 teacher 标注，优先把叶节点样本与标签缓存成可复用数据集，并在该数据集上测试排序保持残差、按 root 的 pairwise loss、风险/不确定性头和更稳健的预算未解处理。

## 决策 61

**决策**
将显式 GFT-SD 叶值版本确认为新的 Sokoban 50 关正式最强配置。当前正式结果从 `41/50` 更新为 `44/50`；TVA 仍不进入推荐配置，后续 TVA 必须超过该显式 SD 叶值线才可更新主结果。

**背景**
此前 10 关门槛显示，显式 `V_SD` 的 `discounted_progress` 口径已经恢复到强 `sd_heuristic` 对照的 `8/10`；加入保守二层叶值校正 `K/Q/C=0.05` 后仍为 `8/10`。该结果说明 `linear_projected` 的早期负结果不能代表显式 SD 叶值整体不可用。用户随后要求将显式 SD 叶值版本扩展到 50 关。

本轮运行 50 关 heldout，配置为 `leaf_value=gft_sd`、`gft_sd_leaf_mode=discounted_progress`、叶值二层校正 `capacity=0.05,ordering=0.05,commitment=0.05`，动作先验使用 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25,ordering_weight=0.05,commitment_weight=0.05`，其余预算与既有正式线一致：`horizon=20,budget=4096,max_steps=80,revisit_penalty=1.0,transposition,temperature=2.0,uniform_mix=0.1`。输出结果为 `44/50`，状态分布 `deadlocked=1,max_steps=2,no_action=3,solved=44`。

相对旧正式最强 `41/50`，新增解出 `level_index=5,6,25,30`，退化 `level_index=40`，净增 `+3`。死锁数保持 `1`，max_steps 从 `5` 降到 `2`。该收益主要来自减少搜索耗尽型失败，而非增加死锁风险。

**选项**
1. 因该配置仍未接入 TVA，继续把 `41/50` 作为主结果，只把 `44/50` 作为开发记录。
2. 将显式 GFT-SD 叶值版本提升为新的正式最强配置，并把 TVA 后续目标提高到超过 `44/50`。
3. 同时运行 TVA 50 关确认，以便一次性决定是否再替换主结果。

**结论**
采用选项 2。该 50 关结果已经使用正式 heldout 切片和正式预算，且相对旧最强线有净增益、死锁不增加，因此应更新主结果。TVA v4 虽然通过过独立审计的校准锚定版本存在，但 10 关闭环未超过显式 `V_SD`，没有资格直接进入 50 关主结果确认。

**影响**
Sokoban 当前正式主结果为显式 GFT-SD 叶值 + 强 GFT action prior 的 `44/50`。论文和方法叙述中，`41/50` 应作为被显式 `V_SD` 叶值版本超越的历史强基线。后续 TVA、搜索形态修复或更多二层资源权重学习，必须以 `44/50` 作为当前闭环门槛；仅恢复到 `8/10` 开发切片或通过独立审计不再足以更新主结果。

## 决策 62

**决策**
当前 `44/50` 的未解关卡不解释为 `sokoban_gptpro.md` 八个 GFT-SD 存量之外缺少一组新的并列抽象层因素。下一步优先统一激活和审计已有八存量，尤其是箱子机动性 `M`、玩家推位可达性 `A`、死锁债务 `D` 和动作相关性 `I`，并补充进度停滞或循环债务的历史审计。

**背景**
用户要求细致分析未求解算例，判断是否对应未纳入的 SD 抽象层因素。本轮对当前正式最强 `44/50` 结果中的 6 个失败关卡进行定向重跑，扩大 trace 到 100 步并开启 GFT 读数。失败关卡为 `level_index=0`、`9`、`14`、`40`、`45`、`46`。

`level_index=9` 和 `45` 的最终状态中，区域容量裕度、死锁债务和顺序依赖债务已经强烈报警。例如 `level_index=9` 最终 `M=0,A=1,K=-4,D=5.080,Q=2,C=0`，且为 matching deadlock；`level_index=45` 最终 `M=0,K=-4,D=3.597,Q=6`，所有普通合法推箱都会被保守死锁门剪掉。该类失败说明已有 `K/D/Q` 信号介入不足，而不是抽象层没有表达。

`level_index=14` 和 `46` 最终不是正式 matching deadlock，但 `viable_push_count=0`，表现为准死锁。`level_index=14` 最终 `M=0,A=1,D=2.437,Q=1`；`level_index=46` 最终 `M=0,A=2,D=1.907,K=0,Q=0,C=0.968`。该类失败主要由机动性和可达性坍缩解释，当前实现已经读出 `M/A/D`，但推荐配置没有把这些因素作为主叶值和主动作先验分量消费。

`level_index=0` 和 `40` 是后期振荡。`level_index=0` 的 80 步轨迹只有 8 个唯一底账状态，后 20 步持续左右推同一箱子；`level_index=40` 的 80 步轨迹只有 20 个唯一底账状态，且该关在旧 `41/50` 的动作先验加旧叶值配置中 18 步解出。该类失败指向进度停滞和循环债务不足，设计文档没有把它列为第九个并列库存，更适合由动作相关性 `I`、revisit 惩罚、历史循环债务或 TVA 长期排序承载。

**选项**
1. 在八个 GFT-SD 存量之外新增一组独立抽象存量，例如循环、互锁或走廊宏动作存量。
2. 将失败解释为现有八存量已读出但在线消费不完整，优先补齐 `M/A/D/I` 的叶值、动作先验和 TVA 审计接口。
3. 暂不处理失败归因，直接继续 TVA 或搜索器改造。

**结论**
采用选项 2。现有失败样本的大部分风险已经能由八个 GFT-SD 存量读出；主要缺口是部分读数尚未以统一范式进入 MCTS 叶值、动作先验或 TVA 排序。只有振荡和进度停滞具有历史依赖特征，但它更像动作相关性、revisit 机制和长期价值排序的不足，而不是需要立即扩展 `z_t=(P,M,A,K,D,Q,C,I)` 的并列维度。

**影响**
后续实验不应因为单因素暂未提升而删除 `M/A/D/I`。更合适的推进顺序是：先以 10 关开发切片做保守小权重激活和审计，检查 `M/A/D/I` 是否缓解 `14/46` 的准死锁、`9/45` 的入口锁定和 `0/40` 的振荡；若没有严重退化，再进入组合测试或可学习权重。若后续确实要新增因素，应作为 `I` 的瓶颈影响域、`D` 的软死锁子项或 TVA 的长期残差特征先行试验，而不是立即改变八存量主合同。

## 决策 63

**决策**
`M/A/D/I` 的首轮在线激活不更新 Sokoban 推荐配置。玩家推位可达性 `A` 可作为中性候选保留；当前一阶箱子机动性 `M`、整体死锁债务 `D` 和瓶颈重叠动作相关性 `I` 不按本轮在线公式继续组合或扩展到 50 关。

**背景**
根据决策 62，下一步以 10 关开发切片检查已有八存量中尚未充分消费的 `M/A/D/I`。本轮在当前显式 GFT-SD 叶值配置上只改权重，不改模型代码，不改核心转移 provider。基线为 `8/10`，失败为 `level_index=0:max_steps`、`9:deadlocked`。

箱子机动性单项 `gft_mobility_weight=0.10,gft_sd_leaf_mobility_weight=0.05` 得到 `7/10`，新增 `level_index=6:no_action`。玩家推位可达性单项 `gft_access_weight=0.10,gft_sd_leaf_access_weight=0.05` 保持 `8/10`，失败形态不变。死锁债务单项 `gft_deadlock_weight=0.05,gft_sd_leaf_deadlock_weight=0.05` 降至 `6/10`，新增 `level_index=5:max_steps`、`6:no_action`。动作相关性单项 `gft_bottleneck_weight=0.02` 也降至 `6/10`，新增 `level_index=5:max_steps`、`6:max_steps`。降权的 `M+A` 组合仍为 `7/10`。

随后只将不退化的 `A` 单项拿到 6 个正式失败关卡上复核。结果没有新增解出：`0` 仍为 `max_steps`，`9` 仍为 `deadlocked`，`14` 从 `no_action` 变为 `max_steps`，`40` 仍为 `max_steps`，`45` 和 `46` 仍为 `no_action`。

**选项**
1. 因为 `A` 不退化，将其加入当前 50 关推荐配置。
2. 因为 `M/D/I` 单项退化，删除这些抽象层因素。
3. 保留 `A` 作为中性候选，但不更新推荐配置；将 `M/D/I` 的当前在线公式判为不合格，后续改做分量重标定、局部互锁/循环机制或 TVA 排序特征。

**结论**
采用选项 3。`A` 通过了 10 关“不严重退化”筛查，但没有改善正式失败关卡，因此不足以进入 50 关确认。`M/D/I` 的负结果不能解释为抽象层因素无效；它们在失败审计中确实读出风险，但当前在线消费方式会破坏可解边界样本排序。

**影响**
当前正式主结果仍为显式 GFT-SD 叶值 + 强 GFT action prior 的 `44/50`。下一步不应继续对本轮 `M/D/I` 权重做简单组合。更高优先级的方向是：针对 `0/40` 设计历史循环和进度停滞审计；针对 `14/46` 将一阶机动性改为局部互锁、多步解锁性或关键通道可释放性；针对 `D/I` 做分量级权重重标定或转为 TVA/离线排序特征，先通过 10 关和失败关卡定向审计后再考虑闭环主线。

## 决策 64

**决策**
建立在线轨迹 GFT-TVA 作为下一阶段 TVA 训练机制，但不将其 smoke checkpoint 作为推荐配置。在线 TVA 必须先通过独立截断价值审计、风险审计和 10 关闭环门槛，才可挑战当前 `44/50` 主结果。

**背景**
用户提出从第一性原理看，显式 SD 抽象层不可能完整概括所有长期价值，因此 TVA 应负责学习显式 SD 无法覆盖的长期残差。同时，TVA 不应在闭环中起负作用；若起负作用，应视为设计或训练不当。讨论后形成约束：成功轨迹可作为高置信正价值标签，死锁和无动作失败可作为高置信负价值标签，步数耗尽未解只能作为删失风险标签，不能默认硬标为真实不可解。

本轮新增 `train_sokoban_gft_online_tva.py`。该脚本使用当前 GFT-SD-MCTS 在线闭环运行关卡并记录实际访问的根状态和最终状态，再训练带价值头、风险头和显式 `V_SD` 锚定项的 `mlp_guarded_v1` checkpoint。默认 `max_steps` 只训练风险头，`anchor_weight=0.75`，避免未审计 TVA 直接替换显式 `V_SD`。

两个 smoke 验证了机制。低预算 5 关均为 `max_steps`，产生 `94` 个风险样本但无价值样本，训练 gate 按 `min_value_train_samples` 失败；这证明步数耗尽默认不会被写成确定负价值。已知可解单关产生 `13` 个 solved 价值样本，训练 gate 通过，并且生成 checkpoint 可被评估入口以 `--leaf-value tva` 读取执行。带 `--require-independent-gates` 的正式评估会因缺少独立审计路径被拦截，符合安全接入约束。

**选项**
1. 直接将在线 TVA smoke checkpoint 接入闭环并跑 10 关或 50 关。
2. 继续沿用此前离线 root search-value 或离线叶节点 teacher TVA，不新增在线轨迹机制。
3. 将在线轨迹 TVA 确认为下一阶段训练机制；先完善采样和审计，再进入闭环性能测试。

**结论**
采用选项 3。在线轨迹采样解决的是分布错配问题，比单纯扩大离线样本更符合当前失败模式。但 smoke checkpoint 只证明接口可运行，不证明长期价值排序可靠。尤其是 `max_steps` 删失风险、历史循环状态和同一根节点候选排序仍需独立审计。

**影响**
后续 TVA 路线转向在线轨迹数据集构建。下一步应在训练关卡上采集同时包含 solved、deadlocked/no_action 和 max_steps 的在线轨迹，形成可复用缓存；然后运行独立截断价值审计、删失风险审计和 10 关闭环。未通过这些门槛前，在线 TVA 不进入推荐配置，也不更新当前 `44/50` 主结果。

## 决策 65

**决策**
在线轨迹 GFT-TVA 的 `alpha=0.25` 混合叶值候选进入“受限候选”状态，但不进入推荐配置，不更新当前 `44/50` 主结果。失败轨迹的负价值标签默认只作用于最终失败状态，失败轨迹中间状态只作为风险样本。

**背景**
在线轨迹 TVA 的首个混合数据集来自非 heldout `train/001.txt` 前 40 关，包含 `834` 个状态样本，轨迹状态为 `max_steps=648,solved=134,no_action=52`。原始标注策略把 `no_action/deadlocked` 失败轨迹的所有中间状态都标为负价值。独立截断价值审计显示该策略会让 TVA 对高置信正价值叶状态严重低估，无法通过价值审计。

因此实现新增 `--failure-value-label-scope {final,all}`，默认 `final`。修订后，成功轨迹继续为整条轨迹提供正价值标签；失败轨迹只有最终失败状态提供负价值标签；步数耗尽仍只作为删失风险标签。使用该策略训练的候选在纯 TVA 叶值下仍不合格，但在 `V_leaf=0.75V_SD+0.25V_TVA` 的混合形式下通过独立截断价值审计：高置信 MAE `0.489`，优于显式 `V_SD` 的 `0.552`；排序准确率 `0.962`，优于显式 `V_SD` 的 `0.949`。

删失风险审计没有通过。按关卡留一风险探针 AUC 为 `0.442`，balanced accuracy 为 `0.504`，risk recall 为 `0.238`。这说明当前风险头不能用于闭环门控或剪枝。低预算 10 关闭环诊断显示候选有正向信号：`train/002.txt` 前 10 关从显式 `V_SD` 的 `5/10` 到混合 TVA 的 `6/10`，`train/003.txt` 前 10 关从 `4/10` 到 `6/10`。但这些切片不是正式 heldout，预算也低于正式配置。

**选项**
1. 因混合叶值通过价值审计且低预算闭环有增益，直接扩展到 50 关。
2. 因风险审计失败，冻结在线 TVA 路线。
3. 将该版本记录为受限候选：允许继续做更大非 heldout 审计和接近正式预算的 10 关确认，但不得作为推荐配置或正式主结果。

**结论**
采用选项 3。当前证据说明在线轨迹 TVA 的标签策略修订有效，且保守混合叶值可能改善显式 `V_SD` 的叶节点估计；但风险头尚不可靠，闭环证据仍是小切片和低预算诊断。因此，该版本是下一步 TVA 的候选基线，而不是已完成的主结果。

**影响**
后续在线 TVA 实验应沿用 `failure_value_label_scope=final` 作为默认安全标注策略，并将 `alpha=0.25` 混合叶值作为当前候选对照。下一步必须至少完成更大独立截断价值审计、风险头重训或风险审计修订、以及接近正式预算的 10 关非 heldout 或开发切片验证。正式主结果仍为显式 GFT-SD 叶值的 `44/50`；在线 TVA 只有在审计和闭环均超过该线时才可更新推荐配置。

## 决策 66

**决策**
删失风险审计失败不再作为禁止在线 TVA 闭环效果测试的硬门槛。对于当前 `alpha=0.25` 混合叶值候选，应区分两件事：风险头不能作为剪枝或安全门控使用；但通过独立截断价值审计的混合叶值可以进行闭环效果测试。当前在线轨迹 GFT-TVA 混合叶值记录为新的实测最好候选 `45/50`，但在正式推荐前仍需复跑或复核门槛元数据和风险/效率问题。

**背景**
决策 65 记录了当前在线轨迹 GFT-TVA 候选的状态：`alpha=0.25` 混合叶值通过独立截断价值审计，但删失风险审计失败。用户指出，风险审计可能过于严格，仍应测试闭环效果。本轮据此运行正式预算测试。

10 关结果显示，`alpha=0.10` 为 `7/10`，`alpha=0.25` 为 `8/10`。`alpha=0.25` 与显式 `V_SD` 基线总数持平，但解出原来 deadlocked 的 `level_index=9`，同时丢掉 `level_index=6`。随后 50 关 heldout 探索性测试得到 `45/50`，相对显式 `V_SD` 的 `44/50` 净增一关。新增解出 `level_index=9,40,46`，退化 `6,25`；死锁数保持 `1`，max_steps 保持 `2`，no_action 从 `3` 降到 `2`，mean pushes solved 从 `15.68` 降到 `15.16`，但 wall time 从约 `867s` 增加到约 `994s`。

本轮还发现评估器的独立门槛读取逻辑只支持旧 grid TVA 审计字段，不支持当前 GFT-TVA 截断价值审计字段。该问题已修复，并用 `--require-independent-gates` smoke 验证通过。因此，50 关探索性结果的行为有效，但结果文件中的独立门槛元数据是在修复前生成的，正式提升推荐配置前应复跑或等价复核。

**选项**
1. 因风险审计失败，忽略 `45/50` 闭环结果，继续把在线 TVA 阻断在审计阶段。
2. 因 `45/50` 超过 `44/50`，立即把在线 TVA 混合叶值提升为正式推荐配置。
3. 将风险审计失败从“禁止效果测试的硬门”降级为“禁止风险门控的限制”；承认 `45/50` 是新的实测最好候选，但在正式推荐前补齐门槛元数据、复核耗时和风险头使用边界。

**结论**
采用选项 3。闭环结果说明风险审计确实不能简单作为否定混合叶值的证据；但风险审计失败仍说明当前风险头不能安全地承担剪枝、屏蔽或强门控职责。当前 `45/50` 是重要正向信号，应成为下一步主候选，而不是直接作为已完全审计完成的最终配置。

**影响**
后续主线应围绕 `alpha=0.25` 在线轨迹 GFT-TVA 混合叶值展开。优先事项为：第一，用修复后的独立门槛读取逻辑复跑或复核 50 关结果；第二，分析退化关卡 `6` 和 `25`，确认是否来自风险惩罚过强、价值混合权重过大或训练标签覆盖不足；第三，评估耗时增加是否来自 TVA 预测缓存、特征计算或搜索路径改变；第四，在风险审计通过前，不得将风险头用作独立门控。当前显式 `V_SD` 的 `44/50` 仍是已完全记录的正式基线，在线 TVA 的 `45/50` 是新的实测最好候选。

## 决策 67

**决策**
在线轨迹 GFT-TVA 的动态消费规则不进入推荐配置，也不扩展到 50 关。新增的动态叶值混合、动作先验软惩罚和动态根动作选择开关保留为默认关闭的审计入口；当前主候选仍为固定 `alpha=0.25` 混合叶值的 `45/50` 探索性结果。

**背景**
对 `45/50` 候选的退化关卡分析显示，第 25 关存在明确的强负 TVA 修正：错误候选子状态 `risk=0.999,V_TVA-V_SD=-2.388`，但固定 `alpha=0.25` 不足以压过显式 SD 资源改善和动作先验；第 6 关则属于局部读数看似安全、后续入口锁死的延迟失效。用户同意按“候选子状态风险和 TVA 修正动态调整消费强度”的策略测试。

本轮实现了三类默认关闭开关。动态叶值混合在高风险强负修正时提高 `alpha`；动作先验软惩罚在同一条件下降低候选动作的 GFT 先验分数；动态根动作选择默认仍按访问次数出手，但当访问次数最高动作触发强负 TVA 信号且另一个候选均值更高时，允许切换到均值更高的动作。所有机制均不修改底层转移 provider，也不把风险头作为剪枝门控。

10 关正式预算消融结果为负。宽阈值动态叶值为 `5/10`，加入已访问状态额外叶值惩罚后为 `6/10`；收紧阈值、加入动作先验软惩罚并全局按均值选择为 `7/10`；收紧阈值、加入动作先验软惩罚和动态根动作选择为 `6/10`。这些结果均低于固定混合 TVA 和显式 `V_SD` 的 `8/10`。

局部正信号是第 25 关可被收紧阈值、动作先验软惩罚和根动作按均值选择修复，17 步解出。这说明 TVA 价值头已经读出该振荡分支的长期风险；但同一消费规则在 10 关上引入第 5、8 关等新增退化，不能作为主线。

**选项**
1. 因第 25 关被修复，将动态 TVA 消费扩展到 50 关。
2. 因 10 关整体退化，删除动态 TVA 消费代码。
3. 保留代码为默认关闭的审计和后续校准入口，但不更新推荐配置，不进行 50 关扩展。

**结论**
采用选项 3。第 25 关证明 TVA 信号方向有用，但手工动态消费规则不稳定。当前证据支持继续改进 TVA 排序目标或根动作级 pairwise 校准，而不支持把本轮动态规则作为在线策略。

**影响**
当前 Sokoban 结果排序保持不变：显式 GFT-SD 叶值 `44/50` 是已完整记录的正式主线，固定 `alpha=0.25` 在线轨迹 GFT-TVA `45/50` 是新的实测最好候选但仍需复核门槛元数据；动态 TVA 消费结果不进入主表。后续若继续处理第 25 关，应优先训练能直接约束根动作排序的 TVA 目标，而不是继续放大风险触发阈值或重复状态惩罚。

## 决策 68

**决策**
状态门控 GFT-TVA 混合记录为新的实测最好候选，50 关 heldout 探索性结果为 `46/50`。固定全局 `alpha=0.25` 的在线 TVA 退为对照候选；显式 GFT-SD 叶值 `44/50` 仍是已完整审计的正式主线。状态门控候选在正式推荐前仍需独立门槛元数据复跑或等价复核。

**背景**
用户澄清“动态 TVA”应理解为根据当前状态选择不同 TVA 权重，必要时权重取 `0`，而不是只在高风险负修正时放大 TVA。本轮据此新增 `--tva-state-gated-blend`。直接价值型 GFT-TVA 叶值改为：

```text
V_leaf(s)=(1-alpha(s))*V_SD(s)+alpha(s)*V_TVA(s)
```

当前候选默认 `alpha(s)=0`，仅在 `risk_probability<=0.30` 且 `V_TVA-V_SD>=0.25` 时使用 `alpha=0.25`。这保留了第 9、40 关中低风险正修正带来的收益，并避免固定混合在第 6 关的退化。第 25 关仍需要根动作选择兜底：若访问次数最高的根候选满足 `risk_probability>=0.95,V_TVA-V_SD<=-1.0`，且另一个候选均值更高，则允许切换到均值更高的候选。该兜底不修改转移 provider，不剪枝，也不把风险头作为安全门控。

正式预算 10 关结果显示，宽松状态门控为 `9/10`，优于显式 `V_SD` 和固定混合 TVA 的 `8/10`。严格正向门控可以修复第 25 关但 10 关降回 `8/10`，因此不作为候选。宽松状态门控加动态根选择同时保持 10 关 `9/10` 并修复第 25 关。50 关探索性测试得到 `46/50`：相对显式 `V_SD` 新增解出 `9,40`，无退化；相对固定 TVA 恢复 `6,25`，丢掉 `46`，净增 `+1`。当前失败关卡为 `0:max_steps`、`14:no_action`、`45:no_action`、`46:no_action`。

**选项**
1. 立即把状态门控 TVA 提升为正式推荐配置。
2. 因风险审计仍失败，忽略 `46/50` 闭环结果。
3. 将状态门控 TVA 记录为新的实测最好候选，但保持探索性标签，后续补齐独立门槛元数据和风险/效率复核。

**结论**
采用选项 3。状态门控直接落实了“TVA 可按状态退回 `alpha=0`”的策略，并在 50 关上超过固定混合候选。风险审计失败不否定混合叶值效果，但仍限制风险头不能用于剪枝、屏蔽或安全门控；本次 50 关也尚未带修复后的正式门槛元数据。

**影响**
后续主线应以状态门控 GFT-TVA 作为优先候选。下一步优先事项为：第一，复跑或等价复核带独立门槛元数据的 `46/50`；第二，分析第 46 关为何固定混合能解而状态门控不能解；第三，继续校准风险头和状态门控条件，避免把风险头提升为硬门控。论文方法叙述中，`44/50` 是正式显式 SD 主线，`45/50` 是固定混合 TVA 对照，`46/50` 是当前最强探索候选。

## 决策 69

**决策**
保留状态门控 GFT-TVA 的 `46/50` 作为当前最强单次配置；将“状态门控失败后用固定 `alpha=0.60` 重跑失败关卡”的 `49/50` 记录为探索性组合策略正信号，不把它提升为单次主配置。

**背景**
当前 `46/50` 状态门控配置剩余失败为 `0:max_steps`、`14:no_action`、`45:no_action`、`46:no_action`。单关固定 TVA 混合比例扫描显示，`alpha=0.60` 可以解出 `14/45/46`，但不能解出第 `0` 关。该结果说明固定高权重 TVA 可以补偿状态门控在部分准死锁和入口锁定样本上的保守性。

固定高权重不能直接替代状态门控。正式预算 10 关开发切片中，固定 `alpha=0.60` 与 `alpha=0.65` 均为 `7/10`，失败为 `0:max_steps`、`5:deadlocked`、`6:no_action`，低于状态门控的 `9/10`。新增的动态叶值软惩罚和运输阶段混合规则也未形成稳健单次收益，只保留为默认关闭的审计开关。

在此基础上构造失败后重跑组合策略：先运行当前状态门控 `46/50` 配置；若某关失败，则仅对失败关卡使用固定 `alpha=0.60` 重跑；只有重跑解出时才替换结果。该组合得到 `49/50`，触发重跑 4 关，修复 3 关，唯一剩余失败为 `level_index=0:max_steps`。

**选项**
1. 将固定 `alpha=0.60` 作为新的单次主配置。
2. 将失败后重跑组合策略的 `49/50` 直接作为新的主结果。
3. 保留 `46/50` 为当前单次最好配置，把 `49/50` 作为组合策略探索结果，后续若要正式化，需要先定义预先触发规则并重新评估时间成本和外推性。

**结论**
采用选项 3。固定高权重 TVA 在目标失败样本上有明确局部收益，但 10 关开发切片显示其全局稳定性不足。失败后重跑组合策略证明两种 TVA 消费模式互补，但它不是一次固定配置下的闭环 MCTS 运行，不能与单次配置直接等价。

**影响**
后续主线分为两层。单次配置层面继续以状态门控 GFT-TVA 为基线，优先处理第 `0` 关循环停滞和第 `14/45/46` 近终局准死锁的状态识别。组合策略层面可以把“状态门控优先、失败后固定高权重重跑”发展为预先声明的组合搜索方法，但必须单独报告触发次数、额外耗时和失败后重跑规则，避免把组合收益误写为单次搜索器收益。

## 决策 70

**决策**
当前不把 phase060 这类 `P/C` 阈值规则提升为动态 TVA 单次配置。该规则可作为离线诊断和后续特征设计依据，但不能用来声称已经把 `49/50` 组合策略压缩为单次动态 TVA。

**背景**
用户提出疑问：既然固定不同 alpha 可以把组合结果推到 `49/50`，是否可以直接用动态 TVA 在单次搜索中实现同样结果。为此，本轮新增离线轨迹审计脚本 [audit_sokoban_dynamic_tva_alpha_features.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_dynamic_tva_alpha_features.py)，重放已有状态门控、固定高权重和 phase060 结果，不重新运行 MCTS。

被检验的 phase060 规则为：当 `P>=20`，或 `P<=10` 且 `C>=0.25` 时使用 `alpha=0.60`；否则回退到当前状态门控的低风险正修正规则。该规则在 `14/45/46` 三个正例的初始状态均触发，并且 phase060 或固定 `alpha=0.60` 可解出这三关，说明它确实捕捉到一部分修复窗口。

但同一规则不能区分反例。第 `5` 关和第 `6` 关在状态门控下本来可解，初始状态不触发 phase060，但可解轨迹中后段都会进入 `P<=10,C>=0.25` 区域；固定高权重和 phase060 动态版本分别导致 `deadlocked/no_action` 或 `deadlocked`。第 `0` 关则不触发 phase060，固定高权重也无法修复，属于循环停滞难例。

**选项**
1. 直接将 phase060 规则作为新的单次动态 TVA 配置，并报告其与 fallback 组合策略等价。
2. 放弃动态 TVA alpha 路线，只保留 `46/50` 状态门控和 `49/50` 组合结果。
3. 将 phase060 审计记录为负结果：它解释了正例窗口，但反例误触发过多；继续保留动态 TVA 方向，但要求新增入口可释放性、局部互锁、重复状态率或候选动作级排序校准后再测试。

**结论**
采用选项 3。当前证据支持“高 TVA 权重在部分准死锁状态有用”，但不支持“仅凭 `P` 与 `C` 阈值即可安全决定何时提高 alpha”。`49/50` 仍是组合策略正信号，不是单次动态 TVA 已掌握的结果。

**影响**
后续若继续追求单次 `49/50`，应先把反例 `5/6` 与正例 `14/45/46` 放在同一审计集中建模，并加入当前 `P/C` 阈值以外的状态或动作级证据。论文和结果表述中不得把失败后固定 alpha 重跑的收益写成状态门控 TVA 的单次闭环收益。

## 决策 71

**决策**
将带前缀与根状态保护的动态运输阶段 TVA 记录为当前最强单次探索候选，50 关 heldout 单次运行结果为 `49/50`。显式 GFT-SD 叶值 `44/50` 仍保留为已完整审计的正式主线；该动态 TVA 结果在独立复核前不替代正式主线。

**背景**
决策 70 已确认，单纯 phase060 规则虽然覆盖 `14/45/46` 的高权重 TVA 修复窗口，但会在 `5/6` 的可解轨迹后段误触发并导致退化。因此，本轮没有继续扩大 `P/C` 阈值，而是增加两个保护条件：第一，`--tva-transport-phase-max-closed-loop-step 5` 只允许运输阶段高权重在闭环前缀步触发；第二，`--tva-transport-phase-require-root-eligible` 要求当前根状态本身满足同一运输阶段谓词，避免树内深层叶节点脱离当前根状态阶段后单独触发。

该规则与当前状态门控 GFT-TVA 共同使用。默认状态仍为 `alpha=0`，低风险正修正状态仍使用 `alpha=0.25`；只有在前缀保护和根状态保护同时满足，且候选状态满足 `P>=20` 或 `P<=10,C>=0.25` 时，才使用运输阶段 `alpha=0.60`。底层转移 provider 仍为精确格点 stock-flow，风险头和 TVA 价值头不用于剪枝或安全门控。

单关复核显示，该规则解出目标修复样本 `14/45/46`，同时保持反例 `5/6` solved。10 关开发切片为 `9/10`，唯一失败仍为 `0:max_steps`。50 关 heldout 单次运行输出 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json)，aggregate 为 `49/50`，状态分布 `max_steps=1,solved=49`，唯一失败为 `level_index=0:max_steps`，`/usr/bin/time` 为 `real=1738.16s,user=1738.00s,sys=6.74s`。

**选项**
1. 将该 `49/50` 立即提升为正式主线。
2. 因该规则来自当前失败与反例集，继续只保留 `46/50` 状态门控候选，不记录本轮单次结果。
3. 将该 `49/50` 记录为最强单次探索候选，但要求独立门槛元数据复跑、风险审计和外推复核后才更新正式主线。

**结论**
采用选项 3。该结果回答了“能否把组合策略正信号压缩为单次动态 TVA”的核心问题：在加入闭环前缀和根状态阶段保护后，可以得到一次固定配置下的 `49/50`，且不依赖失败后重跑或结果替换。但该规则仍是从当前正反例审计中诱导出来的，不能直接等同于已完成正式方法验证。

**影响**
后续结果表述应区分三层：`44/50` 是正式主线，`49/50` 动态运输阶段 TVA 是最强单次探索候选，`49/50` 失败后固定 alpha 重跑是组合策略对照。下一步应优先做带独立门槛元数据的等价复跑，并检查第 `0` 关循环停滞是否需要独立的重复状态率或进度停滞机制，而不是继续扩大运输阶段高权重 TVA 的触发窗口。

## 决策 72

**决策**
将带前缀与根状态保护的动态运输阶段 GFT-TVA 从探索候选提升为 Sokoban 当前正式主线。正式主结果更新为 50 关 heldout 单次配置 `49/50`；显式 GFT-SD `44/50` 降为零 TVA 学习权重正式对照，状态门控 GFT-TVA `46/50` 保留为较保守的单次对照，失败后固定 `alpha=0.60` 重跑的 `49/50` 保留为组合策略对照。

**背景**
决策 71 要求在升级前完成独立门槛元数据复跑、关键消融、风险口径审计和外层复核。本轮冻结规则，不再调整阈值或按失败关卡重跑。冻结规则为：默认 `alpha=0`，低风险正修正状态 `alpha=0.25`；在 `step<=5` 的闭环前缀内，当前根状态和候选叶状态均满足 `P>=20` 或 `P<=10,C>=0.25` 时，使用运输阶段 `alpha=0.60`。

正式复跑输出 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json)。该运行启用 `--require-independent-gates`，`grid_sd_gate=PASS`、`tva_independent_gate=PASS`、`failed_checks=[]`，material passport 为 `VERIFIED`。aggregate 为 `49/50`，唯一失败为 `level_index=0:max_steps`，且结果文件不含 `primary_result`、`fallback_used_count` 或 `fallback_results`，因此不是失败后组合替换。

关键消融显示，未保护 phase060 在 10 关开发切片上为 `7/10`，新增 `5:deadlocked` 与 `6:deadlocked`；仅前缀保护仍为 `7/10`，新增 `5:deadlocked` 与 `6:no_action`；仅根状态保护恢复为 `9/10`；完整保护规则保持 `9/10`，同时第 `5` 关解长回到 24 步。该结果说明根状态阶段保护是必要保护项，前缀保护提供额外保守约束。

外层切片复核使用 `medium/train/001.txt` 前 10 关，并使用同一 independent gate 约束。显式 GFT-SD 与状态门控 GFT-TVA 对照均为 `6/10`，动态运输阶段 GFT-TVA 为 `7/10`，未观察到相对对照的系统性退化。

**选项**
1. 因已完成独立门槛和消融，将动态运输阶段 GFT-TVA 提升为正式主线。
2. 因外层复核仅为 10 关切片，继续把该结果保留为探索候选。
3. 要求第二个完整 50 关 heldout 复核后才允许任何主线更新。

**结论**
采用选项 1。当前证据已满足本阶段主线升级标准：冻结规则复跑稳定复现 `49/50`，独立 SD 与 TVA gate 均通过，关键消融解释了保护项的必要性，外层切片没有显示退化。第二个完整 50 关复核仍有价值，但不作为本阶段升级的硬前置条件。

**影响**
论文和项目文档中的 Sokoban 当前正式主线应更新为“动态运输阶段 GFT-TVA，单次 `49/50`”。需要同时报告边界：该方法仍使用 exact-grid stock-flow 转移，TVA 只进入叶值混合和窄根动作选择，不做剪枝；第 `0` 关循环停滞仍未解决；外层复核目前为 10 关切片；失败后固定 alpha 重跑的 `49/50` 仍只能作为组合策略对照，不能与单次配置混同。

## 决策 73

**决策**
将 Boxoban `unfiltered/test/000.txt` 全 1000 关结果记录为动态运输阶段 GFT-TVA 正式主线的公开外部基准：`989/1000`，success rate `0.989`。该结果用于与使用 Boxoban `unfiltered test` 的已发表成果对齐；50 关 heldout 的 `49/50` 仍保留为内部主线开发和方法审计结果。

**背景**
此前 `49/50` 来自 `medium/train/000.txt` 的 offset `48` 到 `97`，适合内部 heldout 和机制审计，但不够直接对应公开论文的测试口径。Boxoban README 明确给出 `unfiltered test` 为 `1000` 关，并说明该测试集被 Orseau et al. 使用。因此，在不调整动态运输阶段 TVA 规则的前提下，本轮将冻结正式主线评估于完整 `unfiltered/test/000.txt`。

运行仍启用 independent gates，并使用 20 个不重叠 50 关分片并行完成。为保证分片与单进程全量等价，分片 seed 使用 `offset*1009`，对应评估脚本内部的 `seed + local_level_index*1009` 随机流。汇总文件 [gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json) 标记为 `VERIFIED`，`grid_sd_gate=PASS`、`tva_independent_gate=PASS`、`failed_checks=[]`，且没有 `primary_result`、`fallback_used_count` 或 `fallback_results` 字段。

结果为 `989/1000`，状态分布 `solved=989,max_steps=7,deadlocked=4`。11 个失败关卡为 `199:deadlocked`、`367:max_steps`、`525:deadlocked`、`546:max_steps`、`664:max_steps`、`698:max_steps`、`706:deadlocked`、`758:max_steps`、`785:deadlocked`、`892:max_steps`、`977:max_steps`。

**选项**
1. 只报告内部 50 关 `49/50`，暂不进入公开测试集。
2. 将 `unfiltered/test` 全 1000 关结果作为主要外部基准，同时保留内部 50 关结果用于方法演化叙述。
3. 在看到 `unfiltered/test` 结果后继续调参，再重新报告测试集成绩。

**结论**
采用选项 2。`unfiltered/test` 完整 1000 关是当前最小公开可比测试口径，且本次未基于测试集结果调整规则。选项 3 会破坏测试集作为外部评估的意义；后续调参应回到训练或验证切分，再另行冻结后复测。

**影响**
论文和报告中的外部基准可以写为“Boxoban unfiltered test, 1000 levels, 989 solved, 98.9%”。同时必须说明方法口径：该结果来自 push-level GFT-SD-MCTS、exact-grid 转移 provider 和动态运输阶段 GFT-TVA 叶值混合，不是端到端 model-free RL agent 的同协议训练结果。若后续要与 Guez et al. 的 medium/hard 线进一步对齐，还需要单独评估 `medium/valid` 或 `hard`。

## 决策 74

**决策**
在 NeurIPS 论文补充实验中加入 Boxoban/Sokoban 训练与审计规模对照表，并将本文 `989/1000` 的公开基准定位表述为“接近强已发表结果 / competitive with SOTA-class search and model-free planning baselines”，而不是严格 SOTA。

**背景**
用户要求补充一张小表，列出本文方法使用的训练/审计样本量、TVA 训练状态数、teacher 调用量，并与 Guez et al. 这类 model-free planning 的大规模训练步数对照。文献复核显示，Guez et al. (2019) 的 DRC(3,3) 使用 Boxoban unfiltered large train set `900k` 关，并训练至 `1e9` environment steps 后在 unfiltered test 上约 `99%` solved。与此同时，Orseau et al. (2018) 的 LevinTS 和 Orseau and Lelis (2021) 的 PHS/PHS* 属于策略引导或策略+启发式搜索路线，均在 1000 关测试口径上报告过 `1000/1000`。

本文当前外部基准为 Boxoban `unfiltered/test/000.txt` 完整 `1000` 关的 `989/1000`，结果强于或接近 Guez et al. DRC 的约 `99%` 数值，但低于已发表搜索方法的满分结果。另一方面，本文使用的是 exact-grid stock-flow 转移、push-level MCTS 和 GFT/TVA 叶值评估；可学习 TVA 训练来自 40 关、`1378` 个 raw 状态、`834` 个去重状态，其中 value 训练状态 `114` 个；独立审计为 `120` 个叶状态、`78` 个高置信 teacher 标签。teacher 不参与训练或在线测试，仅用于独立审计。

**选项**
1. 在论文中宣称 Boxoban `unfiltered/test` SOTA。
2. 完全不与已发表 Boxoban 结果对比，只把该实验作为内部扩展。
3. 增加训练/审计规模对照表，并将结论限制为：在公开测试集上达到接近强已发表结果的性能，同时展示显式结构化世界模型、少量 TVA 状态训练和可解释机制读数的优势。

**结论**
采用选项 3。该表已经写入 [full_draft_zh.tex](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/full_draft_zh.tex) 的 Boxoban 补充外部基准小节。表中同时列出本文、Guez et al. DRC、Orseau et al. LevinTS 和 Orseau and Lelis PHS*，并明确训练/审计规模、teacher 或搜索监督口径和测试结果。

**影响**
论文主张应改为：Boxoban 结果支持 SD-MCTS 的较广泛适用性，说明在规则可显式编码的离散规划任务中，结构化世界模型加少量 TVA 校准可以接近强已发表求解器；该结果也支持解释性和潜在样本效率优势。但不能写成严格超过所有已发表方法，因为已有搜索方法在该 1000 关口径达到 `1000/1000`。如果后续要主张更强 SOTA，需要在更难的 `medium/valid`、`hard`、更大关卡或统一训练/测试预算下追加评测。

## 决策 75

**决策**
将 XSokoban first 第 1 关失败解释为结构分布外推、目标房间长程承诺不足和 TVA/GFT 校准范围越界的组合结果，而不是简单归因为 `max_steps=80` 太短或单次搜索预算不足。Boxoban `unfiltered/test` 的 `989/1000` 继续作为标准 10x10、4 箱同分布公开基准；XSokoban first-40 作为更强 OOD 压力测试单独报告。

**背景**
当前正式主线在 Boxoban `unfiltered/test/000.txt` 全 1000 关上为 `989/1000`。同时，XSokoban first 第 1 关在相同最佳配置下，`max_steps=80` 与 `max_steps=1000` 均未解出。用户要求分析 XSokoban first 是否与 Boxoban unfiltered/test 在结构上已有显著不同，并解释二者性能差异。

本轮结构统计显示，Boxoban `unfiltered/test` 固定为 `10x10`、4 箱，中位地板数 `32`、中位初始合法 push 数 `8`、中位 dead-square 地板占比 `0.333`、中位割点数 `3`。XSokoban first-40 的中位尺寸为 `15x18`、中位地板数 `157`、中位箱数 `15.5`、最大箱数 `32`、中位初始合法 push 数 `4`、中位 dead-square 地板占比 `0.515`、中位割点数 `14`。XSokoban 第 1 关本身为 `11x19`、6 箱、`139` 个地板格、初始合法 push 数 `3`、dead-square 地板占比 `0.705`、割点数 `12`。

XSokoban 第 1 关 `max_steps=1000` 结果仍为 `max_steps`，最终只完成 `boxes_on_goals=1`，且前 20 步 trace 已出现同一箱子的重复推动和多个单 child root。初始 GFT 读数 `P=97.35,M=3,A=3,D=2.75,C=0`，最终 `P=81.45,M=2,A=2,D=2.877,Q=2,C=0.123`，说明模型读出了高运输工作量和软死锁/顺序压力，但当前 action prior 与叶值混合没有把这些读数转化为多箱目标房间填充策略。

**选项**
1. 将 XSokoban 失败解释为步数上限不足，继续放大 `max_steps`。
2. 将 XSokoban first 视为与 Boxoban `hard` 或 `unfiltered/test` 同类，只报告一个统一 Sokoban 成功率。
3. 将 XSokoban first 作为 OOD 压力测试，明确它在尺寸、箱数、目标集中、瓶颈结构和 TVA/GFT 校准分布上均不同于 Boxoban `unfiltered/test`。

**结论**
采用选项 3。`max_steps=1000` 未解出已经排除了单纯 80 步上限解释；XSokoban first 与 Boxoban `unfiltered/test` 的结构差异足以构成分布外测试。当前方法在 Boxoban 上的强表现主要来自 4 箱同分布下的精确转移、GFT 运输匹配和 TVA 校准；XSokoban 失败则指向多箱目标房间的 box-goal/room-order commitment 缺失。

**影响**
论文和报告中应把 XSokoban 结果写成“OOD 压力测试暴露的泛化边界”，不能用它否定 Boxoban `989/1000` 的标准测试意义，也不能用 Boxoban `989/1000` 推断模型已能处理更大尺寸、多箱、目标房间式 Sokoban。后续若继续推进 XSokoban，应优先建立参考解锚点、GFT 时间序列审计、多箱 room-order option 和 XSokoban 分布上的 TVA 校准，而不是只增加闭环步数。

## 决策 76

**决策**
将 XSokoban first 第 1 关失败的直接机制修正为“当前 SD/GFT/TVA 决策层把多箱目标房间问题过度压缩为短期运输进展最大化，缺少可执行的 box-goal/room-order commitment”，而不是仅以数据集结构差异或报告口径解释失败。

**背景**
用户指出前一版分析偏向已知事实和结果报告，而没有充分解释 SD 模型自身为什么在 XSokoban first 上表现不佳。为此新增 [analyze_xsokoban_sd_failure_mechanism.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_xsokoban_sd_failure_mechanism.py)，对既有 `max_steps=1000` 失败日志保存的前 20 步 trace 做机制诊断。

诊断显示，`20` 个保存 root 中有 `12` 个 `child_count=1`，被选动作访问占比中位数为 `1.0`；`b5:right` 出现 `14/20` 次；被选后继按当前一阶 leaf value 排名第一的步数为 `17/20`。前缀资源变化为 `P:97.35 -> 81.45`，但 `Q:0 -> 2`、`D:2.75 -> 2.877`、`C:0 -> 0.123`。配置层面，root prior 主要奖励 `progress` 和 `goal`，`mobility/access/capacity/deadlock` 权重均为 `0`；GFT leaf 的 `deadlock` 权重为 `0`；TVA trace 特征 `max_abs_z` 均值约 `20.37`，但 OOD gate 关闭。

**选项**
1. 继续保留“XSokoban 与 Boxoban 结构不同”作为主要解释。
2. 将失败归因为 `max_steps` 或 MCTS budget 仍不足。
3. 将失败归因为当前 SD/GFT/TVA 决策目标错配：短期 transport progress 被过度奖励，room-order commitment、软死锁风险、容量/可达性压力和 OOD TVA 保护没有被足够强地消费。

**结论**
采用选项 3。结构差异是触发条件，但不是充分解释；保存 trace 表明当前 MCTS 多数时候在服从自身价值函数，被选前缀也确实降低了 `P`。失败的关键在于该价值函数没有把 XSokoban 目标房间的填充顺序和长程承诺编码成强约束或高权重风险，因此错误前缀在局部看起来仍是好动作。

**影响**
后续 XSokoban 方向的第一优先级应是机制修复，而不是继续扩大 `max_steps`。候选方向包括显式 room-slot/goal-order option、对错误深推的早期惩罚、将 D/K/A/M 作为可消费的强 prior 或 leaf 风险项、启用并校准 TVA OOD gate，以及用参考解轨迹审计每一步是否满足目标房间填充顺序。

## 决策 77

**决策**
采用 OO 继承机制派生 `SokobanRoomGFTSDModel` 作为 XSokoban/Hard 难题专用 SD 版本。该派生类只增加 room-slot、frontier、entry、uncommitted occupancy 和语义 child 代理接口，不修改 `SokobanGFTSDModel`、现有 planner 入口或 Boxoban `989/1000` 正式主线。

**背景**
专家建议指出，XSokoban first 的失败核心不是搜索预算不足，而是当前 SD 层没有把 box-goal/room-order commitment 建模为可执行决策变量。用户明确要求通过 OO 继承派生新的难题 SD 版本，确保不会影响现有成果。因此，直接在原 `SokobanGFTSDModel` 或 `evaluate_sokoban_gft_sdmcts.py` 中改写默认行为不符合本阶段约束。

本轮实现的 A1/A2 诊断显示，XSokoban 第 1 关保存 trace 中 20 个被选后继全部处于 `feature_max_abs_z>3` 的 OOD 区域，其中 3 个存在当前 alpha 实际生效的 OOD 正向 TVA 修正。Room-GFT 诊断识别到 1 个 6-slot goal room、3 个 entry/support cell；前缀中没有 frontier fill，却出现 uncommitted entry、non-frontier fill、room-order debt 和 blocked-slot 增量。这些信号支持先做派生 hard-instance 机制，而不是继续只调原主线权重。

**选项**
1. 在现有 `SokobanGFTSDModel` 中直接加入 room-order 逻辑，并让所有评估默认使用。
2. 保持原模型完全不动，只写外部分析报告，不增加可复用代码接口。
3. 派生 `SokobanRoomGFTSDModel`，以继承方式复用原 GFT-SD 与 exact grid-flow 转移，新增难题诊断和后续 semantic planner 所需接口。

**结论**
采用选项 3。该选择同时满足机制推进和主线隔离：Boxoban 正式结果仍由原评估入口和原模型合同复现；XSokoban/Hard 修订线获得独立 provider id、独立诊断脚本和独立输出文件。

**影响**
后续 A3/A4/A5 均应沿 `room_gft_*` 命名空间推进，结果写入 `results/sokoban_sd_world_model/room_gft_*`，文档写入独立报告或当前研究文档。只有在派生 hard-instance 版本通过独立诊断和回归测试后，才讨论是否将其中某些机制回迁到通用 Sokoban 主线。

## 决策 78

**决策**
将 A3 实现为独立 `RoomSemanticGFTSDMCTSPlanner`，使 MCTS 节点携带 room-slot commitment，并允许同一物理 push 按不同 slot 承诺展开为多个 semantic children。该 planner 只用于派生 hard-instance 路线，不替换现有 Boxoban GFT-SD-MCTS 主线。

**背景**
专家建议指出，当前 trace 中多个 `child_count=1` 并不意味着战略分支只有一个，而是物理状态合并压缩了“这个箱子应承诺给哪个 slot”的决策。原 planner 使用 `action.flat_index` 作为 root child key，同一物理 push 无法产生多个 child；即使在 value 层惩罚 room-order 错误，也不能让 MCTS 在同一物理后继下比较不同 slot 承诺。

本轮新增 `RoomSemanticAction`、`RoomCommitmentState` 和 `RoomSemanticGFTSDMCTSPlanner`。节点 key 改为 `gft_canonical_key(state) + commitment_state.key`，物理局面相同但 slot commitment 不同的节点不再合并。A3 还加入默认 `6` 步 entry 距离的 commitment boundary，使 slot 承诺可以在进入 goal-room 前展开。

A3 诊断显示，旧失败 trace 上 semantic child 中位数为 `1.0`、最大值为 `4`；被选物理动作拥有多个 slot 选项的步数为 `1`。短闭环 semantic smoke 执行 `20` 步仍未解出，状态为 `max_steps`，最终 `boxes_on_goals=0`，但 semantic child 中位数达到 `4.0`、最大值 `6`，并形成了持久 room-slot commitment。

**选项**
1. 直接修改原 `GFTSDMCTSPlanner`，让所有 Sokoban 评估默认使用 semantic children。
2. 只在报告中说明需要 semantic children，不实现 planner。
3. 新增独立派生 planner，在 hard-instance 诊断和后续实验中显式调用。

**结论**
采用选项 3。A3 已证明 semantic child 和 commitment state 可以接入 MCTS，但短闭环 smoke 未解出 XSokoban 第 1 关，因此它是机制通过而非性能通过。该结果不改变 Boxoban `989/1000` 的正式主线，也不能写成 XSokoban 求解结果。

**影响**
下一步 A4 应在该派生 planner 上把 `P` 从任意 goal 距离进展改为 slot-schedule constrained progress，并强化非 frontier 填充、blocked slot 和无承诺 entry 的 barrier。长走廊内部在承诺确定后仍可能退回单一后继，因此还需要 forced-corridor option 或 corridor contraction 来减少错误单调推进。

## 决策 79

**决策**
将 A4 实现为 `RoomSlotProgressGFTSDMCTSPlanner` 子类，在 A3 语义承诺 planner 之上叠加 slot-schedule constrained progress、承诺不可达惩罚和 pass-through 例外。该 A4 子类只用于 XSokoban/Hard 派生难题线，不替换 A3 诊断 planner，也不回写 Boxoban 正式主线。

**背景**
A3 已能把同一物理 push 拆成不同 slot commitment 的 semantic children，但保存 trace 和短闭环 smoke 仍显示，一旦进入长走廊并形成承诺，planner 仍可能沿局部运输进展推进到错误行。旧失败 trace 的关键入房窗口中，`b5:down` 把已承诺箱子推到与承诺 slot 不可达的行；随后 `b5:right` 填入浅层非前沿 slot，导致 `q_settled` 和 `k_blocked` 上升。

本轮新增 `room_slot_progress_summary`，用反向推箱距离衡量已承诺箱子到承诺 slot 的可推进展，并按 frontier slot 顺序为未承诺箱子构造运输代理。A4 还加入 `committed_pass_through_blocker` 例外：若已承诺箱子进入的是目标深槽的浅层 blocker，且到目标 slot 的距离继续下降，则该中间态不按 settled 错填处理。

诊断输出 [room_gft_xsokoban_a4_slot_progress.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a4_slot_progress.json) 显示，旧 trace 的 20 个 root 上，旧 GFT 与 A4 都有 15 步将保存物理动作排第一，因此 A4 不是全局替换前缀。关键入房窗口中，step `18` 的旧动作 `b5:down` 从旧 GFT rank `3` 降为 A4 rank `4`；A4 首选 `b5:right|slot3@[7,17]`，标记为 pass-through，transport delta 为 `+1`、barrier delta 为 `+2`。

短闭环 A4 smoke 使用 `budget=256,horizon=8,smoke_steps=40,boundary_distance=6`，结果仍为 `max_steps`，最终 `boxes_on_goals=0`。该 smoke 表明 A4 修复的是入房处的局部排序机制，而不是完整求解能力；当前仍存在承诺箱子在走廊中来回推动的问题。

**选项**
1. 直接修改 A3 `RoomSemanticGFTSDMCTSPlanner`，让 A3 诊断和 A4 评分共用同一 planner。
2. 新增 A4 子类，在保留 A3 机制诊断可复现性的前提下叠加 slot-schedule progress。
3. 将 A4 结果立即作为 XSokoban 求解配置，并扩大到 first-40 或 Boxoban hard。

**结论**
采用选项 2。A4 需要改变 prior 和 leaf 评分，若直接写入 A3 planner，会混淆“语义 child 机制是否工作”和“槽位顺序评分是否有效”两个问题。A4 当前只证明局部机制成立，尚未解决短闭环，因此不进入更大测试集或主线结果。

**影响**
后续 A5 应优先处理 A4 smoke 暴露的新瓶颈：长走廊内的重复承诺状态和来回推动。候选方向包括 corridor contraction、forced-corridor macro option、同一承诺箱子反向推动惩罚、以及以参考解或 teacher trace 校准 room-slot 承诺序列。在这些机制通过前，不应把 A4 描述为 XSokoban 解题率改进。

## 决策 80

**决策**
将 A5 实现为 `RoomLoopGuardGFTSDMCTSPlanner` 子类，在 A4 槽位顺序 planner 之上叠加承诺单调 loop guard。A5 记录为有效局部机制但负/部分结果：它消除了 A4 的走廊回退和直接反向推动，但没有解出 XSokoban 第 1 关，不进入正式求解配置。

**背景**
A4 短闭环 smoke 在 `budget=256,horizon=8,smoke_steps=40,boundary_distance=6` 下仍为 `max_steps`，最终 `boxes_on_goals=0`，事后统计显示被选轨迹中有 `15` 次承诺距离回退和 `3` 次直接反向推动。该现象说明 A4 的 slot-schedule progress 虽能修正入房窗口的局部排序，但无法阻止已承诺箱子在长走廊中来回移动。

本轮 A5 在派生 planner 中加入三类软惩罚：已承诺箱子到承诺 slot 的反向推箱距离回退、直接撤销上一步 push、以及近期重复 commitment state。根动作选择还加入保护：若访问数最高的 child 触发 loop penalty，而存在未触发候选，则在未触发候选中按原 `select_by` 规则出手。A5 不剪枝合法动作，不修改 `SokobanRoomGFTSDModel` 静态 room 分解，也不改变 exact grid-flow 转移。

诊断输出 [room_gft_xsokoban_a5_loop_guard.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a5_loop_guard.json) 显示，A5 与 A4 的首个动作差异发生在 step `11`。A5 将被选轨迹中的承诺距离回退、直接反向推动和重复 commitment state 均降为 `0`，并在 step `18` 完成 `1` 个目标；但随后在第 `22` 步进入 `no_action`，最终 `boxes_on_goals=1`，`final_slot_schedule_cost=46.0`。

**选项**
1. 继续只调 A5 的 loop penalty 权重，期待通过更强软惩罚解出该关。
2. 将 A5 因为完成 1 个目标而作为 XSokoban 修复结果。
3. 将 A5 记录为局部机制通过但整体负/部分结果，并把下一阶段转向 corridor contraction、forced-corridor macro 或参考解/teacher trace 锚定的多箱 room-slot 调度。

**结论**
采用选项 3。A5 证明 A4 暴露的走廊回退可以被承诺单调性和根选择保护消除；同时也证明，仅靠更强的局部 loop guard 不足以产生后续多箱 room-slot 调度。A5 的 `no_action` 结局说明，单箱填入后，当前 commitment state、frontier slot 代理和物理可达性尚未形成可继续推进的多箱策略。

**影响**
后续不应把 A5 写成 XSokoban 求解能力提升，也不应优先继续堆叠软惩罚。A6 应转向更结构化的推进机制：第一，corridor contraction 或 forced-corridor macro，把长走廊内的单调推进压缩为可审计 option；第二，引入参考解或 teacher trace，对多箱 room-slot 承诺序列做锚定；第三，在这些机制通过前，所有 `room_gft_*` 结果仍保持派生难题线标签，不影响 Boxoban `989/1000` 正式主线。

## 决策 81

**决策**
将 A6 实现为 `RoomCorridorMacroGFTSDMCTSPlanner` 子类，在 A5 loop guard 之上增加 root-level forced-corridor macro。A6 记录为走廊压缩机制通过但整体负/部分结果：它减少闭环决策次数，但没有增加目标完成数，也没有解出 XSokoban 第 1 关。

**背景**
A5 的主要剩余问题是完成第一个目标后进入 `no_action`，但 A5 前半段仍需要逐步重新规划同一承诺箱子的长走廊推进。因此，A6 先验证一个更结构化但保守的 corridor contraction：树内仍保持 A5 single-push semantic MCTS；根节点选中动作后，若同一已承诺箱子存在唯一合法、非死锁、到承诺 slot 反向推箱距离严格下降的后续 push，则把连续 push 作为一个宏执行。该机制只属于 `room_gft_*` 派生难题线，不修改 `SokobanGFTSDModel`、Boxoban 评估入口或 `989/1000` 正式主线。

诊断输出 [room_gft_xsokoban_a6_corridor_macro.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a6_corridor_macro.json) 显示，A6 在 step `9` 触发一次 corridor macro，将承诺箱子从 `[7,8]` 连续推进到承诺 slot `[7,17]`，额外压缩 `8` 个 push。A5 为 `no_action`、执行 `22` 个 push、`boxes_on_goals=1`；A6 为 `no_action`、`14` 个闭环决策、执行 `22` 个 push、`boxes_on_goals=1`。A6 的承诺距离回退和直接反向推动仍为 `0`，final slot schedule cost 仍为 `46.0`。

**选项**
1. 将 A6 作为 XSokoban 修复结果，因为它压缩了长走廊推进。
2. 继续扩大 corridor macro 的触发范围，例如允许多分支单调 push 或跨多个箱子自动推进。
3. 将 A6 记录为局部机制通过但整体负/部分结果，下一阶段转向参考解/teacher trace 锚定的多箱 room-slot 调度。

**结论**
采用选项 3。A6 证明单箱 forced-corridor macro 可以减少重复规划开销，但失败状态与 A5 相同，说明核心瓶颈已经不在“一个已承诺箱子如何穿过长走廊”，而在“第一个箱子填入后，后续箱子的 room-slot 顺序、释放动作和可达性如何协调”。继续扩大单箱宏触发范围可能掩盖多箱调度缺陷，并增加不可审计的自动推进风险。

**影响**
后续 A7 应优先建立多箱 room-slot 调度锚点。可行方向包括：第一，使用 XSokoban 参考解或 teacher trace 生成 slot-order/box-release 监督信号；第二，在 `RoomCommitmentState` 中加入阶段性 room fill plan，而不仅记录单箱承诺；第三，把当前 A6 macro 作为执行层 option 保留，但不再把它作为主要改进方向。A6 不改变 Boxoban 正式主线，也不能报告为 XSokoban 解题率改进。

## 决策 82

**决策**
将 A7 实现为 `RoomTraceAnchoredGFTSDMCTSPlanner` 子类，在 A6 corridor macro 之上加入 teacher/reference trace anchor。A7 记录为机制定位和修复方向验证：在参考调度显式给定时可解出 XSokoban 第 1 关，但它不是无 teacher 的通用求解配置。

**背景**
A5/A6 均在完成第一个目标后进入 `no_action`，且 A6 已证明单箱走廊压缩不能改变最终失败状态。该现象把瓶颈定位到宏后的多箱 room-slot 调度、释放动作和槽位顺序选择。为验证当前 exact transition、room commitment 和 corridor macro 是否仍能承载完整解，本轮在 A7 诊断脚本内生成带父指针的 local exact-state teacher trace，并将其转化为 root state 到下一步参考动作及未来 goal slot 的映射。

诊断输出 [room_gft_xsokoban_a7_trace_anchor.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a7_trace_anchor.json) 显示，teacher 在 `7248` 个展开内找到 `97` push 解。A7 使用该参考轨迹后，在 `a7_budget=64,horizon=8,a7_max_pushes=120` 下解出该关，执行 `97` 个 push、`63` 个闭环决策、触发 `6` 次 corridor macro，最终 `boxes_on_goals=6`、slot schedule cost 为 `0.0`。A6 对照仍为 `no_action`、执行 `22` 个 push、`boxes_on_goals=1`。

**选项**
1. 继续只扩大 A6 corridor macro 的触发范围，期待自动跨越多箱调度问题。
2. 将 A7 的 teacher trace anchor 作为正式求解器，并报告 XSokoban 第 1 关已由本方法自主解出。
3. 将 A7 作为机制诊断：用 teacher/reference trace 验证结构承载能力，并把下一阶段目标转为从参考轨迹中抽取可自主生成的 room fill plan、释放动作约束和价值审计目标。

**结论**
采用选项 3。A7 的正结果证明，当前派生结构的转移、承诺状态和宏执行层不是阻塞点；只要多箱调度被正确锚定，该关可以沿可行轨迹完成。与此同时，A7 每个命中参考状态的出手都使用了 teacher trace，因此不能把它表述为无 teacher 泛化能力或求解率改进。

**影响**
后续应从 A7 trace 中抽取无参考解版本可消费的结构：目标房间填充阶段、箱子释放顺序、slot commitment 更新规则、宏触发边界和错误调度的负例。所有相关代码仍应保留在 `room_gft_*` 派生命名空间内，不修改 `evaluate_sokoban_gft_sdmcts.py` 或 Boxoban `989/1000` 正式主线。

## 决策 83

**决策**
将 teacher 轨迹作为 Room-GFT 派生线的缺失机制诊断材料，而不是作为在线求解器或训练标签直接注入主线。下一阶段优先设计无 teacher 的 room fill plan / slot queue / box release value，而不是继续只调 A6 corridor macro 或 loop penalty 权重。

**背景**
A7 已证明，在参考调度显式给定时，当前 exact transition、room commitment 和 corridor macro 可以完成 XSokoban 第 1 关。但这不能说明 A6 已具备自主调度能力。为进一步定位缺失，本轮新增 [room_gft_xsokoban_teacher_trace_gap.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_teacher_trace_gap.json)，尝试对 XSokoban first 前 3 关生成 teacher 轨迹并在 teacher 状态上重算 A6 评分。

默认 `100000` expansions 或 `90s`/level 的 teacher 预算下，第 1 关解出，解长 `97` push、展开 `7248`；第 2 关和第 3 关分别为 10 箱和 11 箱，在 `90s` 内未解出。因此，当前完整 gap 统计主要来自第 1 关，另两关作为 teacher 轨迹库建设的预算边界。第 1 关上，A6 prior 将 teacher 动作排第 1 的比例为 `0.598`，排入前 3 的比例为 `0.856`，最大 rank 为 `13`；MCTS 每 10 步采样的 teacher 动作匹配比例为 `0.600`。gap tags 显示 `setup_or_release_underrewarded=20`、`teacher_low_prior_rank=14`、`short_transport_competes=16`，并有 `9` 段长度至少为 3 的同向 corridor run。

**选项**
1. 将 teacher trace 直接作为 A7 式在线锚定，继续扩大可解关卡数。
2. 继续调 A6 的 corridor macro、loop penalty 或 prior 权重，期待同一机制自发得到多箱调度。
3. 将 teacher trace 用作诊断样本，从中抽取无 teacher 的 room fill plan、slot queue、box release value 和长程 tradeoff 校准目标。

**结论**
采用选项 3。A7/A8 诊断共同说明，A6 的阻塞点不在单步转移，也不只在单箱走廊推进，而在缺少可自主生成和维护的多箱目标房间填充计划。teacher trace 最有价值的用途是暴露哪些正确动作在当前 SD 评分中被低估，以及这些动作对应哪些缺失机制。

**影响**
后续实现应新增派生类或派生 planner，显式加入 room fill phase、frontier/deep slot queue、box release/setup value 和 schedule-guarded corridor macro。任何继续使用 teacher trace 直接指导出手的实验必须标注为 reference-conditioned diagnostic，不得与无 teacher 求解率混写。

## 决策 84

**决策**
高预算 fast teacher 运行后，暂不宣称已获得 XSokoban first 前三关的综合 teacher 轨迹。当前证据口径固定为：第 1 关有完整 teacher 轨迹和 gap 诊断；第 2、3 关在更高预算下仍未解出，只作为 teacher 轨迹库构建的预算边界。

**背景**
为响应“增加 teacher 预算，争取求解前三关，并在综合证据下补充 SD 所缺信息”的要求，本轮在 `analyze_room_gft_xsokoban_teacher_trace_gap.py` 中加入 fast teacher 配置：Hungarian matching 下界、Tier-A deadlock 剪枝、关闭 viable-push 子项、heuristic weight `5.0`，并将每关预算提高到 `1000000` expansions 或 `600s`。该配置仍只用于离线诊断，不注入 Room-GFT/A6 或 Boxoban 主线的在线求解。

运行结果写入 [room_gft_xsokoban_teacher_trace_gap_fast_budget.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_teacher_trace_gap_fast_budget.json) 与 [room_gft_xsokoban_teacher_trace_gap_fast_budget.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_teacher_trace_gap_fast_budget.md)。第 1 关解出，解长 `109` push、展开 `822845` 个状态、生成 `3666199` 个后继、deadlock 剪枝 `535324`，耗时约 `483.69s`。第 2 关和第 3 关仍在 `600s` 时间上限内未解出，分别展开 `533130` 和 `784388` 个状态。

第 1 关 fast teacher 轨迹上的 gap 方向与默认 teacher 轨迹一致：teacher 动作 rank-1 比例为 `0.697`，rank<=3 比例为 `0.899`，`setup_or_release_underrewarded=16`、`teacher_low_prior_rank=11`、`short_transport_competes=14`，并包含 `11` 段长度至少为 3 的同向走廊推进。

**选项**
1. 继续扩大 teacher 预算，直到获得前三关完整轨迹后再修订 SD 缺失判断。
2. 将第 2、3 关未解出解释为当前 SD 结构已足够，只需改进 MCTS/TVA 消费端。
3. 固定当前证据边界：用第 1 关完整轨迹抽取缺失机制，同时把第 2、3 关记录为 teacher 侧搜索能力不足的边界，后续先实现无 teacher 的 room fill plan、slot queue、box release/setup value 与 schedule-guarded corridor macro。

**结论**
采用选项 3。高预算 fast teacher 未能产生前三关完整轨迹，因此不能写成“综合前三关轨迹”结论。它也不能证明 SD 端不缺信息；相反，第 1 关两条不同 teacher 轨迹均指向相同缺口，说明当前 Room-GFT/A6 仍缺少自主目标房间填充计划、箱子释放与准备动作价值、通道保持价值，以及由全局调度约束的长走廊宏触发边界。

**影响**
下一阶段不应只在 TVA 或 MCTS 末端调权。应继续保持 `room_gft_*` 派生命名空间和 OO 继承边界，在新的 hard-instance SD 版本中显式加入可更新的 room fill phase / slot queue、box release/setup value、channel preservation 与 schedule-guarded corridor macro。任何使用 teacher 轨迹直接锚定动作的实验仍须标注为 reference-conditioned diagnostic，不得作为独立求解率报告。

## 决策 85

**决策**
将 A8 实现为无 teacher 在线动作锚定的 `SokobanRoomFillPlanGFTSDModel` 与 `RoomFillPlanGFTSDMCTSPlanner` 派生版本。A8 记录为 SD 补强后的部分正结果：它让 XSokoban 第 1 关从 A6 的 `boxes_on_goals=1` 推进到 `boxes_on_goals=2`，但仍未解出，不进入正式求解配置。

**背景**
第 1 关 teacher 轨迹说明，A6/A7 之间的差距不是单步转移或 corridor macro 执行能力，而是缺少自主 room fill plan、active slot、释放/准备动作价值和通道保持状态。进一步重建 teacher 轨迹后可见，浅槽在可行解中多次作为中转和通道，而非简单的最终填充目标。因此，直接把深槽优先写成硬约束会误伤可行前缀；更合适的做法是将 fill phase、active assigned box、channel/staging 和 active-box 可推性作为 SD 状态项。

本轮新增 `SokobanRoomFillPlanGFTSDModel`，继承 `SokobanRoomGFTSDModel`，只增加当前状态可计算的 fill-plan 摘要，不保存 teacher 状态表，也不修改 exact grid-flow 转移。新增 `RoomFillPlanGFTSDMCTSPlanner`，继承 A6 的 `RoomCorridorMacroGFTSDMCTSPlanner`，在 prior 和 leaf 中消费 fill-plan barrier、active commitment、channel staging、early channel occupancy、active transport、active box release 和 active monotone push 信号；它不继承 A7 的 reference action anchor。

诊断输出 [room_gft_xsokoban_a8_fill_plan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a8_fill_plan.json) 显示，默认 `budget=256,horizon=8,max_steps=120` 下，A6 为 `no_action`、执行 `22` push、`boxes_on_goals=1`、final slot cost `46.0`；A8 为 `no_action`、执行 `48` push、`boxes_on_goals=2`、final slot cost `43.0`，触发 `4` 次 corridor macro。较大预算输出 [room_gft_xsokoban_a8_fill_plan_h12_b1024.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a8_fill_plan_h12_b1024.json) 使用 `budget=1024,horizon=12,max_steps=160`，结局仍为 `boxes_on_goals=2`、`no_action`。

**选项**
1. 将 A8 作为 XSokoban 第 1 关修复结果报告。
2. 放弃 SD 侧补强，转向只提高 MCTS 预算或调 TVA 权重。
3. 将 A8 记录为部分正结果，并继续在同一 OO 派生线中补充更细的释放队列、active-box 局部可达性和多箱协同调度；TVA 只作为这些状态的审计/校准配合。

**结论**
采用选项 3。A8 证明 teacher 轨迹归纳出的 SD 状态能够产生方向性改善，但较大预算复核不改变 `2` 个目标后的 `no_action`，说明当前剩余瓶颈不是简单搜索深度不足，而是第三个 deep slot 之前的释放队列和局部可达性仍未被充分建模。

**影响**
后续 A9 应继续保留 `room_gft_*` 命名空间和继承链，在 A8 的 active slot / active box 基础上加入“哪些非 active push 是释放 active box 的准备动作”的因果状态，尤其是 active box 被阻塞时的局部可达性、通道占用和候选释放动作价值。TVA 可配合生成审计样本，但不应把 teacher 动作直接注入在线求解。

## 决策 86

**决策**
将 A9 实现为无 teacher 在线动作锚定的 `SokobanRoomReleasePlanGFTSDModel` 与 `RoomReleasePlanGFTSDMCTSPlanner` 派生版本。A9 记录为 release-plan 诊断增强：它显式定位 active box 的站位格阻塞，但尚未把 XSokoban 第 1 关解出，不进入正式求解配置。

**背景**
A8 在第 1 关上相对 A6 多完成一个目标，但在第三个 deep slot 前仍进入 `no_action`。终态分析显示，active slot 为 `(7,17)`，active assigned box 为 `(6,5)`；若要使该箱子向 slot 接近，需要向下推至 `(7,5)`，而玩家站位格 `(5,5)` 被另一箱占据。A8 只能给出 `active_box_blocked_count=1`，不能区分是站位格被占、玩家不可达、目标格被占，还是几何上不存在单调推进。

本轮新增 `SokobanRoomReleasePlanGFTSDModel`，继承 A8 模型，枚举 active assigned box 的候选单调推进，并记录所需站位格、站位格阻塞、目标格阻塞、玩家可达性和 Tier-A deadlock 状态。新增 `RoomReleasePlanGFTSDMCTSPlanner`，继承 A8 planner，在 prior 和 leaf 中消费 release barrier、release need、stand blocker、unreachable stand 和 available monotone push 信号；它不继承 A7 reference anchor，也不读取 teacher 动作表。

诊断输出 [room_gft_xsokoban_a9_release_plan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a9_release_plan.json) 显示，默认 `budget=256,horizon=8,max_steps=120` 下，A8 为 `no_action`、执行 `48` push、`boxes_on_goals=2`、final slot cost `43.0`；A9 为 `no_action`、执行 `44` push、`boxes_on_goals=2`、final slot cost `42.0`、final release barrier `8.0`。较大预算输出 [room_gft_xsokoban_a9_release_plan_h12_b1024.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a9_release_plan_h12_b1024.json) 使用 `budget=1024,horizon=12,max_steps=160`，A9 仍为 `no_action`、`boxes_on_goals=2`、release barrier `8.0`。

**选项**
1. 将 A9 作为 XSokoban 第 1 关修复结果报告。
2. 认为 SD 侧信息已经充分，后续只调 MCTS budget 或普通 prior 权重。
3. 将 A9 记录为诊断增强而非求解成功：SD 侧已能读出具体站位格阻塞，但仍需要 TVA 审计或更强 schedule guard，使 planner 提前保持和释放关键站位格。

**结论**
采用选项 3。A9 证明 A8 末端的 active-box blocked 状态可以进一步解释为具体的站位格释放问题：`(6,5)` 向 `(7,17)` 的单调推进需要站位 `(5,5)`，该格被箱子占据，因而 `active_release_need_count=1`、`active_release_stand_blocker_count=1`、`active_release_monotone_available_count=0`。但 A9 默认和大预算复核均未解出，说明仅在当前一步 prior/leaf 中加入 release-plan 信号仍不足以形成稳定的提前释放策略。

**影响**
后续应将 A9 release-plan 状态纳入 TVA 审计和校准样本，重点覆盖 stand-cell blocker 出现之前的前缀状态、释放动作低估状态和错误占用关键站位格的负例。若继续扩展 MCTS，应优先考虑 schedule-guarded access preservation 或 release macro，而不是把 teacher 动作表在线注入。

## 决策 87

**决策**
将 A10 实现为 `SokobanRoomAccessPlanGFTSDModel` 与 `RoomAccessPlanGFTSDMCTSPlanner` 派生版本。A10 记录为 XSokoban first 第 1 关的无 teacher 在线动作锚定单关正结果：默认 120 push 完成 5/6，延长到 200 push 后解出该关；该结果不外推为 first-40 或 hard 集解题率。

**背景**
A9 能在终态准确定位 active box `(6,5)` 推向 slot `(7,17)` 所需站位格 `(5,5)` 被箱子占据，但该诊断发生得太晚。A9 在 slot 切换到 `(7,17)` 后才看到阻塞，已经无法生成有效释放动作。因此，需要把后续 deep slot 的站位格保护提前纳入 SD 状态。

本轮新增 `SokobanRoomAccessPlanGFTSDModel`，继承 A9 模型，对后续 open deep slot 进行 lookahead assignment，并枚举每个 assigned box 的候选单调推进、所需站位格和 blocker 状态。新增 `RoomAccessPlanGFTSDMCTSPlanner`，继承 A9 planner，在 prior/leaf 和根选择保护中消费 future stand-cell blocker 信号。

初始 `lookahead_slots=3` 版本过强，默认 120 push 只完成 1 个目标，表现为过度保护 access 状态导致循环。收窄后的正式 A10 采用 `lookahead_slots=2`，并只把“未来站位格被箱子占据”作为强信号，一般 `stand_cell_unreachable_by_player` 仅保留在摘要中。诊断输出 [room_gft_xsokoban_a10_access_plan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan.json) 显示，`budget=256,horizon=8,max_steps=120` 下，A10 为 `max_steps`、执行 `120` push、`boxes_on_goals=5`、final slot cost `1.0`。延长步数输出 [room_gft_xsokoban_a10_access_plan_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan_s200.json) 显示，相同配置仅将 `smoke_steps` 改为 `200` 后，A10 为 `solved`、执行 `127` push、`90` 个闭环决策、`boxes_on_goals=6`。

**选项**
1. 将 A10 直接报告为 XSokoban first-40 或 hard 集求解能力改进。
2. 放弃 access-plan 路线，认为 A9 的 release-plan 已经足够，只需增加 MCTS budget。
3. 将 A10 记录为单关机制正结果：lookahead access-plan 能修复第 1 关 A9 的视野过短问题，但仍需多关复核、消融和 TVA 审计后才能形成整体结论。

**结论**
采用选项 3。A10 的 200 步复核证明，前瞻站位格保护是第 1 关从 A9 失败走向可解的关键机制之一；同时，`lookahead_slots=3` 的负结果说明 access preservation 过强会导致循环，因此该机制需要受控使用，不能简单扩大前瞻深度或权重。

**影响**
后续应在 XSokoban first 前若干关上做 A10 多关 smoke，并对 `lookahead_slots=1/2/3`、future stand blocker root guard、leaf blocker penalty 做消融。若目标是形成论文或团队交接包中的正式 hard-instance 结论，应先区分单关机制正结果、reference-conditioned A7 结果和无 teacher 多关求解率结果。

## 决策 88

**决策**
将 A10 在 XSokoban first 第 2 关上的结果记录为泛化边界复核：A10 规则和实现可在第二关直接运行，但当前 `budget=256,horizon=8,smoke_steps=200,lookahead_slots=2` 配置未解出第二关，不改变 A10 仅为第 1 关单关正结果的表述。

**背景**
A10 的 lookahead access-plan 源自第 1 关 teacher 轨迹诊断，但其状态变量是从当前局面计算得到的后续 deep slot 站位格、release/access barrier 和 blocker 摘要，并未保存第 1 关动作表或逐状态参考轨迹。因此需要在第二关上验证该机制是否至少可机械外推。诊断脚本新增 `--level-offset` 参数后，对 XSokoban first 第 2 关运行同一 A10 派生模型和 planner。输出 [room_gft_xsokoban_a10_access_plan_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan_l002_s200.json) 显示，A9 为 `no_action`、执行 `34` push、`boxes_on_goals=1`；A10 为 `no_action`、执行 `57` push、`53` 个闭环决策、`boxes_on_goals=1`，最终 slot schedule cost 同为 `52`。A10 终态不是 Tier-A 硬死锁，且仍有合法 push。

**选项**
1. 将 A10 第 1 关成功外推为 XSokoban first 多关求解能力。
2. 因第二关未解出而否定 A10 的第 1 关机制意义。
3. 将第二关记录为泛化边界：A10 不是第 1 关硬编码，但当前规则仍不足以处理第二关的多箱 release/access 调度。

**结论**
采用选项 3。第二关结果说明 A10 的规则层不依赖第 1 关，但只补入 future stand-cell blocker 仍不足以解决 10 箱关卡中的多箱释放、通道保持和跨 slot 调度问题。第 1 关结果仍可作为单关机制正结果保留；第二关结果则作为下一版 A11 的诊断依据。

**影响**
后续 hard-instance 派生线不应只扩大 `smoke_steps`。由于第二关失败为 `no_action` 而非步数截断，下一步更应补充多箱 release queue、release macro、跨 slot access preservation 或 TVA 前缀校准，并继续保持无 teacher 在线动作锚定的边界。

## 决策 89

**决策**
将 A11 实现为 `SokobanRoomSchedulePlanGFTSDModel` 与 `RoomSchedulePlanGFTSDMCTSPlanner`，用于把 XSokoban 第 2 关暴露出的多箱调度缺口显式补到 SD 表示侧。A11 当前记录为负结果诊断：调度图表示成立，但 MCTS 消费端尚未形成求解能力。

**背景**
A10 在第二关中不是因步数截断失败，而是在仍有合法 push 且无 Tier-A 硬死锁时进入 `no_action`。这说明 planner 缺少“哪些箱子需要先被释放、哪些站位格需要保护、哪些动作会破坏后续 slot”的多箱调度状态。按照模块分工，该信息应先在 SD 派生模型中显式化，再由 MCTS 消费，而不是只在 MCTS 中堆叠隐式动作权重。

本轮新增 A11 SD 摘要：关键站位格、未来关键站位格、阻塞箱、不可达站位格、release dependency、release action 和 destructive action。首个 raw A11 使用 `lookahead_slots=3`，在第二关上首步偏离 A10 并最终 `no_action`、`boxes_on_goals=0`、`no_legal_pushes=true`。原因是消费端把 destructive action count 下降误判为收益。随后 guarded A11 将前瞻收窄为 `2`，加入 immediate critical/future stand-cell entry penalty，并加 summary cache。guarded A11 在 `budget=256,horizon=8,smoke_steps=200` 下为 `max_steps`、执行 `200` push、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`，仍未解出。

**选项**
1. 将 A11 作为第二关求解改进报告。
2. 因 A11 未解出而回退到只调 A10 的 access-plan 权重。
3. 保留 A11 的 SD 表示层，记录当前 MCTS 消费端为负结果，并将下一步聚焦于 release macro、dependency-stall loop guard 或 TVA 前缀校准。

**结论**
采用选项 3。A11 证明“调度缺口应补在 SD 表示侧”的模块判断是可实现的，但当前 prior/leaf/root guard 不能充分消费这个调度图。raw A11 过强，guarded A11 过弱且循环，二者都不能作为求解结果报告。

**影响**
后续 A12 应在 A11 dependency graph 上构造可执行 release macro 或二箱循环惩罚，而不是再扩大普通 MCTS budget。TVA 可围绕 A11 中 `dependency_count` 长期不变、`release_action_count=0` 和重复动作对生成审计样本。A11 及后续版本仍不得继承 A7 reference action anchor，teacher 轨迹只可用于离线诊断。

## 决策 90

**决策**
将 A12 实现为 `RoomScheduleStallGuardGFTSDMCTSPlanner`，作为 A11 schedule-plan 的消费端 dependency-stall guard。A12 记录为负结果诊断：它打断 A11 的 200 push 循环，但 6 push 后进入 `no_action`，未形成 XSokoban first 第 2 关求解进展。

**背景**
A11 guarded 在第二关上能够避免 raw A11 的早期 `no_action`，但在 200 push 内没有完成目标，并长期重复 `b9:up` 与 `b5:down`。该现象说明仅依赖 dependency/barrier delta 不能识别“调度状态没有实质改善”的循环。因此，本轮在 A11 planner 之上加入 recent moved edge、recent schedule signature、重复 commitment state、反向近期边和 dependency/schedule 非改善检测。该改动只发生在 MCTS 消费端，继续使用 A11 的 SD schedule graph，不改变 Boxoban 主线，也不接入 teacher 动作表。

诊断输出 [room_gft_xsokoban_a12_stall_guard_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a12_stall_guard_l002_s200.json) 显示，同一 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2` 下，A11 guarded 为 `max_steps`、执行 `200` push、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`；A12 为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `40.5`、dependency count `2`、future dependency count `1`、release action count `0`。A12 首步改为 `b6:right`，第三个决策 `b7:right` 被标记为 stalled，随后进入无动作状态。

**选项**
1. 将 A12 作为 A11 循环修复并继续扩大步数或 MCTS budget。
2. 回退 A11/A12 的 SD schedule graph，只保留 A10 access-plan。
3. 将 A12 记录为负结果诊断：dependency-stall guard 能暴露循环消费缺口，但当前缺少 release/access recovery macro 和玩家可达域恢复信号。

**结论**
采用选项 3。A12 证明二箱循环确实可以被消费端 guard 识别和打断，但其替代动作把 planner 带入更早的 `no_action`。因此，A12 不是第二关改进，也不能作为求解配置报告。

**影响**
后续 A13 不应继续只调 stall penalty、recent window 或普通 MCTS budget。更合适的方向是在 A11/A12 的 dependency graph 上构造可执行 release/access recovery macro，显式回答“哪个箱子应被移开以恢复当前或未来站位格可达性”；TVA 可围绕 A12 的 no-action 前缀和 A11 循环前缀做审计/校准。A12 仍保持无 teacher 在线动作锚定边界。

## 决策 91

**决策**
将 A13 实现为 `RoomScheduleAccessRecoveryGFTSDMCTSPlanner`，作为 A12 的 access-recovery guard。A13 记录为部分改善但仍为负结果：它避免 A12 的 6 push 早期 `no_action`，但在 200 push 内仍未完成目标，并形成低合法余量循环。

**背景**
A12 说明 dependency-stall guard 可以打断 A11 的二箱长循环，但会把 planner 导入只有极少合法推箱动作、最终 `no_action` 的局面。A12 终态的 schedule graph 显示，当前站位格不可达，未来 target 被箱子占据，且 `release_actions=[]`。因此，本轮在 A12 planner 之上增加 access-recovery margin：在 unresolved dependency 存在时，惩罚非阻塞合法推箱余量崩塌、`no_legal_pushes`、target blocker 增加和不可达站位未恢复，奖励 release action 出现、target blocker 减少、不可达站位减少和 dependency 总数下降。

诊断输出 [room_gft_xsokoban_a13_access_recovery_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a13_access_recovery_l002_s200.json) 显示，同一 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2` 下，A12 为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`；A13 为 `max_steps`、执行 `200` push、`182` 个闭环决策、`boxes_on_goals=0`、final slot cost `29`、final schedule barrier `87.5`、dependency count `2`、target blocker count `2`、release action count `0`。A12/A13 首个决策差异为 step `1`。

**选项**
1. 将 A13 作为第二关修复结果，并继续扩大 `smoke_steps` 或 MCTS budget。
2. 回退 A13，只保留 A12 的 dependency-stall guard。
3. 将 A13 记录为部分改善的负结果：access-recovery guard 能推迟早期 `no_action` 并暴露低余量循环，但后续必须实现显式 release/access recovery macro。

**结论**
采用选项 3。A13 说明“合法推箱余量”和 “target blocker / unreachable stand” 是必要的消费端诊断信号，但当前实现只是评分和根保护，不能主动构造恢复路径。后半段 `b6:down / b9:up` 类循环表明，当所有剩余候选都处在低余量状态时，guard 已无安全替代动作。

**影响**
后续 A14 应从 guard 转向 macro：在 A11/A13 的 dependency graph 上，针对具体 `stand_cell_unreachable_by_player` 或 `target_occupied_by_box` 依赖，生成短程恢复候选，要求每一步保持 Tier-A 非死锁，并以 target blocker 减少、stand reachability 改善、release action 出现或 legal-margin 恢复作为停止条件。TVA 可继续围绕 A13 低余量循环前缀形成审计样本；teacher/reference 动作仍不得在线注入。

## 决策 92

**决策**
将 A14 实现为 `RoomScheduleRecoveryMacroGFTSDMCTSPlanner`，作为 A13 之后的 bounded release/access recovery macro。A14 记录为负结果：宏执行机制可用，并显著改变闭环轨迹，但未解出 XSokoban first 第 2 关，也未产生目标填充。

**背景**
A13 已说明 access-recovery guard 可以避免 A12 的早期 `no_action`，但在后半段形成低合法余量循环。按照决策 91，下一步应从 guard 转向 macro。本轮新增 A14：在 MCTS 根动作和原 corridor macro 后，搜索最多 `4` 个合法非 Tier-A 死锁 push，若短序列改善合法余量、target blocker、不可达站位、release action、dependency 总数或 slot cost，则作为 recovery macro 追加执行。该实现仍使用 A11 的 schedule graph，不读取 teacher 动作表。

诊断输出 [room_gft_xsokoban_a14_recovery_macro_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a14_recovery_macro_l002_s200.json) 显示，同一 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2` 下，A13 为 `max_steps`、执行 `200` push、`182` 个闭环决策、`boxes_on_goals=0`、final slot cost `29`、dependency count `2`、target blocker count `2`；A14 为 `max_steps`、执行 `200` push、`73` 个闭环决策、`boxes_on_goals=0`、final slot cost `33`、final schedule barrier `72.0`、dependency count `3`、future dependency count `2`、target blocker count `1`、release action count `0`。A14 recovery macro 触发 `48` 次，累计追加 `116` 个 push。

**选项**
1. 将 A14 作为第二关改进结果，并继续扩大 macro 搜索节点数或追加 push 上限。
2. 放弃 recovery macro，回退到 A13 的单步 access-recovery guard。
3. 将 A14 记录为负结果诊断：宏执行能力成立，但目标函数过宽；下一步必须绑定具体 dependency row。

**结论**
采用选项 3。A14 证明可执行短宏动作可以接入当前 OO 派生 planner，并减少闭环决策数；但它追逐一般合法余量和局部 schedule 指标，未能服务于具体站位格或 target blocker 的解除。因此它不是求解配置，也不是第二关成功。

**影响**
后续 A15 应实现 dependency-bound recovery macro：入口选择必须来自 A11/A14 schedule graph 中的具体依赖行，尤其是 `stand_cell_unreachable_by_player` 和 `target_occupied_by_box`；宏搜索目标应优先解除该依赖，而非一般合法余量增长。复现报告应继续记录 dependency row、被选 blocker、宏前后 stand reachability、target occupancy、release action count 和 slot cost。TVA 可围绕 A14 高频宏循环生成负例样本。

## 决策 93

**决策**
将 A15 实现为 `RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner`，作为 A14 之后的 dependency-bound recovery macro。A15 记录为负结果诊断：宏目标已绑定具体 schedule dependency row，但仍未解出 XSokoban first 第 2 关，并较早进入 `no_action`。

**背景**
A14 已证明短程 recovery macro 可以接入当前 OO 派生 planner，但其目标函数过宽，主要追逐一般合法余量、target blocker 和 slot cost 的局部变化。按照决策 92，A15 将宏入口绑定到 A11/A14 schedule graph 中的具体依赖行，候选类型限定为 `stand_cell_unreachable_by_player` 与 `target_occupied_by_box`。宏评分优先奖励该依赖行消失、所选 dependency kind 解除、站位格恢复可达、target blocker 被移开和对应 release action 出现；无依赖进展的候选即使改善一般 legal margin 也不能单独触发。

诊断输出 [room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.json) 显示，同一 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2` 下，A14 为 `max_steps`、执行 `200` push、`73` 个闭环决策、`boxes_on_goals=0`、final slot cost `33`、dependency count `3`；A15 为 `no_action`、执行 `36` push、`21` 个闭环决策、`boxes_on_goals=0`、final slot cost `42`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。A15 dependency-bound macro 触发 `3` 次，累计追加 `9` 个 push，`row_disappeared_count=5`、`stand_resolved_count=3`、`target_resolved_count=0`。

**选项**
1. 将 A15 视为 A14 改进，并继续提高 dependency-bound macro 的权重、节点数或追加 push 上限。
2. 回退到 A14 的一般 recovery macro，因为 A15 更早进入 `no_action`。
3. 将 A15 记录为负结果诊断：依赖行级目标是必要收窄，但仍不足以保证全局 slot-fill 可行性；下一步应加入宏后置条件和 TVA 审计。

**结论**
采用选项 3。A15 证明具体 dependency row 可以作为宏搜索锚点，并能产生可解释的行级进展；但行级依赖消失或局部站位恢复可达并不等价于目标填充进展。A15 的早期 `no_action` 表明，当前宏仍可能把局部依赖解析作为充分条件，忽视了 slot-fill phase、后续合法动作保持和新阻塞风险。

**影响**
后续不应只扩大 A15 的搜索预算。下一版应把 dependency resolution 作为必要条件，同时加入全局后置条件：slot schedule cost 不恶化或有明确补偿、目标填充或 room-fill phase 有单调进展、关键站位/target 不新增破坏、非阻塞合法推箱余量不坍塌。TVA 审计应优先纳入 A15 中“row disappeared 但最终 no_action”的状态，校准局部依赖解析与全局装填可行性之间的差异。

## 决策 94

**决策**
将 A16 实现为 `RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner`，作为 A15 之后的 feasibility-guarded dependency macro。A16 记录为轻量负结果：全局后置条件能避免 A15 的早期 `no_action` 形态，但未产生目标填充，并转入长程循环。

**背景**
A15 证明 dependency-bound macro 可以解析局部 schedule dependency row，但这些局部进展没有稳定转化为 slot-fill 可行性，最终 36 push 后进入 `no_action`。因此，本轮在 A15 之上加入全局后置条件：slot cost 不回退，schedule barrier 不大幅回退，target blocker、critical stand box、destructive action 和 dependency total 不增加，且非阻塞合法推箱余量保持。只有依赖进展和全局后置条件同时成立时，短宏才可执行。

默认 200 push 级别复核在 `recovery_macro_max_nodes=400` 和 `80` 下均因诊断成本过高被中止，说明 A16 的候选拒绝机制需要额外优化。已完成的轻量 smoke 使用 `budget=128,smoke_steps=80,recovery_macro_max_nodes=20`，输出 [room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.json)。A15 对照为 `no_action`、执行 `36` push、`boxes_on_goals=0`、final slot cost `42`、dependency count `2`；A16 为 `max_steps`、执行 `80` push、`78` 个闭环决策、`boxes_on_goals=0`、final slot cost `50`、dependency count `1`、target blocker count `1`、release action count `0`。A16 feasibility-guarded macro 触发 `1` 次，追加 `2` 个 push，最终宏验收行无全局 violation，并解析一个 target dependency。

**选项**
1. 将 A16 作为 A15 的成功修复，并继续扩大 200 push 完整预算。
2. 回退到 A15，因为 A16 的完整配置计算成本过高。
3. 将 A16 记录为轻量负结果：后置条件方向成立，但根选择层仍存在无目标进展的长程循环，且完整诊断成本需要优化。

**结论**
采用选项 3。A16 修正了 A15 的一个关键错误，即局部 dependency resolution 不能作为宏执行充分条件；但它没有解决第 2 关，也没有产生 goal progress。后半段 `b6:down / b9:up` 循环说明，仅过滤宏执行结果不足以约束根选择层的长期调度停滞。

**影响**
后续不应只继续堆叠宏后置条件。下一版应把 schedule-level no-progress 作为根选择或 leaf 层约束，尤其是对“dependency count 维持 1、slot cost 维持约 50、无 release action、动作对反复出现”的状态施加惩罚；TVA 审计应纳入 A16 中后置条件通过但目标/slot phase 无进展的前缀状态。同时需要优化 schedule summary 缓存或宏候选剪枝，否则完整 200 push 诊断成本过高。

## 决策 95

**决策**
将 A17 实现为 `RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner`，作为 A16 之后的 schedule-phase no-progress root guard。A17 记录为负结果：它打断 A16 的长程循环前缀，但过早进入 `no_action`，未形成目标填充。

**背景**
A16 说明全局后置条件能避免 A15 的早期 `no_action`，但轻量 smoke 后半段转入 `b6:down / b9:up` 类长循环，表现为 dependency、slot phase 和 release action 长时间无实质进展。因此，本轮新增 A17，在根候选上显式记录 phase progress、no-progress、stagnant context、重复 schedule signature、重复/反向 moved edge，并在 prior、root guard 和 leaf 中惩罚无进展重复调度。

诊断输出 [room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.json) 显示，同一轻量配置 `budget=128,smoke_steps=80,recovery_macro_max_nodes=20` 下，A16 为 `max_steps`、执行 `80` push、`boxes_on_goals=0`、final slot cost `50`、dependency count `1`；A17 为 `no_action`、执行 `22` push、`19` 个闭环决策、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。A17 中 `phase_no_progress_selected_count=14`、`phase_stagnant_context_selected_count=13`，终态 `no_legal_pushes=true`。

**选项**
1. 将 A17 作为 A16 循环修复，并继续提高 no-progress penalty。
2. 回退 A17，只保留 A16 的 feasibility-guarded macro。
3. 将 A17 记录为负结果诊断：no-progress guard 能定位根选择层问题，但需要可恢复性判据或 TVA 校准，而不是继续加大硬惩罚。

**结论**
采用选项 3。A17 的确改变了 A16 的循环前缀，但把失败从长程循环提前为 `no_action`。这说明“无 phase progress”不是充分的负判据；Sokoban 中许多准备动作短期内不会降低 dependency 或 slot cost，但仍可能保持后续可恢复性。

**影响**
后续应把 A17 trace 作为 TVA/根选择校准数据：重点区分无进展但可恢复的准备动作、无进展且造成低合法余量或 `no_action` 的动作。若继续做规则版，应新增可恢复性读数，例如后继非阻塞合法 push 集规模、关键站位是否仍可重开、release action 是否能在短 horizon 内出现，而不是只依赖 immediate phase delta。

## 决策 96

**决策**
将 A18 实现为 `RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner`，作为 A17 之后的 recoverability-calibrated root guard。A18 记录为负结果：短程可恢复性探测方向成立，但当前一层、局部 slot-cost 判据产生 false positive，使 planner 更早进入 `no_action`。

**背景**
A17 说明单纯惩罚无 phase progress 会过约束根选择，因为 Sokoban 中存在短期不降低 dependency 或 slot cost 的准备动作。为区分准备动作和破坏性动作，本轮在 A17 上加入短程 recoverability probe：对无即时 phase progress 且存在 unresolved dependency 的候选，从 child 状态向前枚举最多 `16` 个深度 `1` 的合法后继，若能看到 goal、slot cost、schedule barrier、dependency、target blocker 或 release action 进展，且 child 状态合法余量充足，则给予 `recoverable_no_progress` bonus。

诊断输出 [room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s80_b128_n16.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s80_b128_n16.json) 显示，同一轻量配置 `budget=128,smoke_steps=80,recovery_macro_max_nodes=20` 下，A17 为 `no_action`、执行 `22` push、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`；A18 为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`、target blocker count `1`、release action count `0`。A18 的首步 `b7:right` 被标为 `recoverable_no_progress=true`，因为一层探测中的 `b6:right` 使 slot cost 从 `52` 降到 `51`，但 dependency、target blocker 和 release action 均无改善。实际闭环前缀为 `b7:right,b6:right,b6:right`，随后进入 `no_action`。

**选项**
1. 将 A18 作为 A17 修复，并继续提高 recoverable no-progress bonus。
2. 回退到 A17 的硬 no-progress guard。
3. 将 A18 记录为 false-positive recoverability 诊断：下一版必须把可恢复性收紧为结构进展或多步 survivability，而不是继续调权重。

**结论**
采用选项 3。A18 证明“可恢复性”是正确的诊断维度，但当前 probe 的充分条件过弱。一层内的 slot-cost 局部改善不能代表依赖可释放、关键站位可恢复或后续合法动作可保持。

**影响**
后续 A19 若继续规则线，应派生新的 survivable recoverability planner，而非修改 A18。候选 no-progress 动作只有在满足 dependency total 下降、target blocker 下降、release action 出现，或通过两步以上合法余量保持检查时，才应被视为可恢复；仅有 slot-cost delta 的 probe 应降权或要求额外 survivability 条件。TVA 审计应把 A18 的首步 false positive 作为负例，校准“局部 slot-cost 恢复”和“真实释放进展”之间的差异。

## 决策 97

**决策**
将 A19 实现为 `RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`，作为 A18 之后的结构进展型 survivable recoverability guard。A19 记录为机制级部分改善：它消除了 A18 的首步 slot-only false positive，并在 40 push 机制级 smoke 中完成 1 个目标，但仍未解出 XSokoban first 第 2 关。

**背景**
A18 的首步 `b7:right` 被标为 `recoverable_no_progress=true`，原因是一层 probe 看到后继 `b6:right` 可带来 slot-cost 局部下降；但该 probe 没有减少 dependency、target blocker，也没有增加 release action，实际轨迹很快进入 `no_action`。因此，本轮在 A18 之上新增 A19，把 recoverability 的充分条件收紧为结构进展：goal 增加、dependency total 下降、target blocker 下降或 release action 增加。slot-cost 或 schedule-barrier 局部改善只作为诊断项，不再单独产生 recoverable bonus。

沿用 A18/A17 的 `budget=128,smoke_steps=80` 并设置 `recoverability_probe_depth=2` 时，`max_nodes=64` 与 `16` 两个配置均因计算成本过高被中止；堆栈显示耗时来自多层 MCTS prior 的 recoverability probe 与 legal-push 可达域计算，不是运行时错误。因此，本轮正式记录机制级 smoke：`budget=64,smoke_steps=40,recoverability_probe_max_nodes=8`，并用同参 A18 作为对照。诊断输出 [room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.json) 显示，同参 A18 为 `no_action`、执行 `6` push、`boxes_on_goals=0`；A19 为 `max_steps`、执行 `40` push、`boxes_on_goals=1`、final slot cost `41`，终态 `no_legal_pushes=false`。

**选项**
1. 将 A19 作为第二关求解配置，并直接扩大 `smoke_steps` 与 MCTS budget。
2. 回退 A18，因为 A19 完整 80-step 诊断成本过高。
3. 将 A19 记录为机制级部分改善：结构进展型 recoverability 是正确修复方向，但必须先降低 probe 成本并补后续 release/action 生成，才能扩大测试。

**结论**
采用选项 3。A19 已证明 A18 的关键误判来自 slot-only recoverability，并通过结构进展判据修复了首步选择；但它只在机制级 smoke 中完成 1 个目标，不能报告为第二关解决。完整配置成本过高也说明当前实现不适合直接扩展到 first-40 或更大预算。

**影响**
后续应优先优化 A19 的计算边界：将 recoverability probe 限制在 root candidates，或按 canonical state、commitment key、lookahead slots 缓存 probe 结果，避免在 MCTS 内部每层候选重复计算。机制上，A19 之后的主要缺口转向一个目标完成后的 dependency/release 调度：需要显式生成 release action，或把 A19 中 1-goal 后仍有 `dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0` 的前缀纳入 TVA 校准样本。

## 决策 98

**决策**
将 A20 实现为 `RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`，作为 A19 之后的 root-only/cache survivable recoverability planner。A20 记录为计算边界修复和机制级部分改善：它完成了 A19 无法完成的 80-step 复核，并在 XSokoban first 第 2 关上达到 2 个目标，但仍未解出该关。

**背景**
A19 修复了 A18 的首步 slot-only false positive，但完整 `budget=128,smoke_steps=80` 诊断因计算成本过高被中止。堆栈显示主要成本来自 MCTS 内部多层节点候选上的 recoverability probe 与 legal-push 可达域计算。因此，本轮将 recoverability probe 限制到当前闭环决策的 root candidates；非根节点不再计算该 probe，并为 root candidates 加入 planner 内缓存。该改动只改变计算边界，不改变 A19 的结构进展型 recoverability 判据。

诊断输出 [room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.json) 显示，A20 使用 `budget=128,smoke_steps=80,recoverability_probe_depth=2,recoverability_probe_max_nodes=8,root_recoverability_only=true` 可完整落盘。A19 机制级对照为 `max_steps`、执行 `40` push、`boxes_on_goals=1`、final slot cost `41`；A20 为 `max_steps`、执行 `80` push、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`，终态 `no_legal_pushes=false`。

**选项**
1. 将 A20 作为第二关求解配置，并继续直接扩大步数或预算。
2. 回退 A19，因为 A20 改变了内部节点的 recoverability shaping。
3. 将 A20 记录为计算边界修复和机制级部分改善：后续重点转向两个目标后的 release/action 生成和调度校准。

**结论**
采用选项 3。A20 已证明 root-only recoverability 是可行的成本边界：它保留根选择层的诊断与约束，并避免在 MCTS 内部重复 probe。但 A20 仍只是机制级部分改善，不能报告为第二关解决。

**影响**
后续不应继续优先加深 recoverability probe。更合适的 A21 方向是在 A20 的 2-goal 后继状态上生成 release action 或构造 release macro，重点处理 `dependency_count=2,future_dependency_count=1,release_action_count=0` 的状态；TVA 也应把 A20 后半段 slot cost 回升、目标数停在 2 的前缀纳入校准样本。若要扩大到 first-40，应先确认 A20 的 root-only 成本在多关上可控。

## 决策 99

**决策**
将 A21 实现为 `RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner`，作为 A20 之后的 stand-lock guard 诊断。A21 记录为负结果：它确认了不可释放 critical stand blocker 是有效风险信号，但单纯在 MCTS prior、root guard 和 leaf 中惩罚该风险，会削弱 A20 已有的目标填充前缀，不能替代显式 release/action 生成。

**背景**
A20 在 XSokoban first 第 2 关上完成 2 个目标后仍停在 `dependency_count=2,future_dependency_count=1,release_action_count=0`。对 A20 后继状态的 replay 显示，剩余依赖中存在 `stand_cell_occupied_by_box`：关键站位格被箱子占据，其中包括已在目标/角落中的箱子，且没有可执行的释放 push。因此，本轮在 A20 上派生 A21，统计 locked stand blocker、goal-locked blocker、assigned-goal reassignment 与 release option，并将其作为根选择和 leaf 惩罚信号。

诊断输出 [room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.json) 显示，同一 `budget=128,smoke_steps=80,recoverability_probe_depth=2,recoverability_probe_max_nodes=8` 下，A20 为 `max_steps`、执行 `80` push、`boxes_on_goals=2`、final slot cost `60`、target blocker count `0`、release action count `0`；A21 为 `max_steps`、执行 `80` push、`boxes_on_goals=0`、final slot cost `48`、target blocker count `1`、release action count `0`。A20/A21 首个决策差异发生在 step `3`；A21 中 `locked_stand_blocker_selected_count=73`，但没有产生新的 release action。

**选项**
1. 将 A21 作为 A20 的修复，并继续提高 stand-lock penalty 或 root guard 强度。
2. 回退到 A20，忽略 stand-lock 诊断，因为 A21 的目标数下降。
3. 将 A21 记录为负结果诊断：stand-lock 信号应作为显式 release macro 的入口和验收特征，而不是单独作为强 guard。

**结论**
采用选项 3。A21 证明 stand-lock 风险确实存在，但也证明只惩罚风险不足以生成解决风险的动作。由于 A21 不新增 release/action 生成器，它只能改变根选择偏好；一旦所有可选轨迹都存在某种 locked stand risk，强 guard 会把 planner 推入低目标数、低 slot cost 但无填充进展的循环。

**影响**
后续 A22 若继续规则线，应继承 A21 或 A20，并显式生成 dependency-bound release macro：入口绑定具体 `stand_cell_occupied_by_box` 依赖，候选短序列以释放该 stand cell、保留/提高目标数、避免新增 target blocker、维持可恢复合法余量为验收条件。TVA 校准样本应纳入 A21 中 `locked_stand_blocker_selected_count` 高但目标数为 0 的前缀，以及 A20 中目标数为 2 但 release action 仍为 0 的前缀。

## 决策 100

**决策**
将 A22 实现为 `RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner`，作为 A21 之后的阶段化 release/action 版本。A22 记录为阶段化机制验证和部分成功：它避免了 A21 的全局 stand-lock 退化，并保留 A20 的 2-goal 前缀，但尚未触发真正的 release-stage macro，也未解出 XSokoban first 第 2 关。

**背景**
A21 确认 stand-lock 风险是真实诊断信号，但将该信号作为全局 root/leaf guard 会过早偏离 A20 的有效装填轨迹，目标数从 A20 的 `2` 降至 `0`。这与当前理论定位一致：SD 世界模型刻画的是决策机制，而复杂问题中的决策机制很可能需要按阶段和里程碑消费不同摘要。因此，本轮在 A21 上派生 A22，将 stand-lock 从全局强惩罚改为 release 阶段入口信号；默认完成至少 `2` 个目标后才进入 release 阶段。

诊断输出 [room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.json) 显示，同一 `budget=128,smoke_steps=80,recoverability_probe_depth=2,recoverability_probe_max_nodes=8` 下，A20 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=60`，A21 为 `max_steps,80 pushes,77 decisions,boxes_on_goals=0,final_slot_cost=48`，A22 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=60`。A20/A22 首个决策差异为 `None`，A21/A22 首个决策差异为 step `3`。A22 的 `release_stage_selected_count=32`，但 `phased_release_macro_trigger_steps=0`，仅有 `pre_release_compatible_macro_trigger_steps=1`。

**选项**
1. 将 A22 作为第二关求解配置，并直接扩大步数或预算。
2. 回退 A21 的全局 stand-lock guard，继续调高或调低 penalty。
3. 将 A22 记录为阶段化机制验证：保留阶段化消费方向，下一步专门修订 release-stage dependency row 选择和候选生成。

**结论**
采用选项 3。A22 已证明 A21 的失败主要来自信号消费阶段错误：stand-lock 在准备和装填阶段不应作为全局强负例，而应在 release/recovery 阶段转化为具体释放目标。但 A22 没有触发 true release macro，说明当前 release 阶段入口后的候选生成或依赖绑定仍不足。

**影响**
后续应继续沿 A22 的阶段化架构推进，而不是回到全局 stand-lock guard。下一步重点是 release-stage macro 的生成与验收：优先选择仍阻塞关键站位的 `stand_cell_occupied_by_box` dependency row，生成能移动对应阻塞箱、保持目标数不回退、避免新增 target blocker、维持可恢复合法余量的短序列。TVA 或根选择校准样本应覆盖 A22 中 `release` 阶段已出现但 `phased_release_macro_trigger_steps=0` 的状态。

## 决策 101

**决策**
将 A23 实现为 `RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner`，作为 A22 之后的 pre-release corridor macro 边界诊断。A23 记录为负结果：它能截断 A22 的早期过度 corridor macro，但不能生成替代调度，最终仍与 A22 同为 2 个目标、0 个 release action。

**背景**
A22 的 release-stage macro 未触发，并非单纯因为 release 宏搜索预算太小。回放第一个 release 状态显示，关键 `stand_cell_occupied_by_box` 依赖中的 stand blocker 已经被局部几何夹住；扩大 release suffix 到 `max_nodes=2000,max_extra_pushes=6` 仍未找到可接受释放序列。继续回溯发现，互锁源头在 fill 阶段的 corridor macro：A20/A22 在 step `2` 一次闭环中执行 4 个单调 push，随后进入低合法余量和前置依赖未解除的状态，step `3` 的 `b3:up` 进一步制造不可释放 stand-lock。因此，本轮在 A22 上派生 A23，只改变 corridor macro 的执行边界。

诊断输出 [room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.json) 显示，同一 `budget=128,smoke_steps=80` 下，A22 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=60,release_action_count=0`，A23 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=60,release_action_count=0`。A23 触发 `2` 次 `pre_release_bottleneck_truncated_corridor`，共节省 `8` 个宏内 push；A22/A23 首个决策差异为 step `3`，但 true phased release macro 仍为 `0`。

**选项**
1. 将 A23 作为 A22 的修复，并继续调低 corridor 截断阈值。
2. 回退 A23，认为 corridor macro 与失败无关。
3. 将 A23 记录为负结果诊断：宏边界确实参与互锁形成，但真正缺口在根选择层的替代调度生成。

**结论**
采用选项 3。A23 说明 A22 的失败不是单一 release-stage 后处理问题，而是 fill 阶段的宏执行已经越过了必要前置依赖检查。但仅截断宏不会自动产生正确动作；planner 仍会在后续根选择中回到相同互锁路径。

**影响**
后续不应继续只调整 `pre_release_corridor_guard_legal_margin` 或截断步数。下一版应面向根选择层或 TVA：把“候选动作会把 target/stand 前置依赖转化为不可释放 stand-lock”的状态作为强负例，同时奖励能维持可恢复空间、降低 target blocker 或打开 stand reachability 的替代动作。A23 的截断位置和 A22 的 step `2`/step `3` 前缀可作为该校准的核心样本。

## 决策 102

**决策**
将 A24 实现为 `RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner`，作为 A23 之后的 pre-release root-trap guard 诊断。A24 记录为负结果：root-trap 信号能识别并避开直接 stand-lock 转化，但单步 root guard 发生得太晚，不能替代多步前置依赖解除调度。

**背景**
A23 证明 corridor macro 截断能改变早期互锁前缀，但最终仍回到 2-goal、0-release-action 终态。A23 的 step `4` 仍选择 `b3:up`，该动作把当前 target/stand 前置依赖转化为不可释放的 critical stand blocker。因此，本轮在 A23 之上新增根层风险摘要，比较候选动作前后的 `target_blocker_count`、`unreachable_stand_count`、critical stand box、`locked_stand_blocker_count`、`release_action_count` 与非阻塞合法推箱数，并在 severe trap 候选存在替代项时优先避开。

诊断输出 [room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.json) 显示，同一 `budget=128,smoke_steps=80` 下，A23 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=60,target_blocker_count=0,release_action_count=0`；A24 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=48,target_blocker_count=1,release_action_count=0`。A23/A24 首个决策差异为 step `4`。A24 改选 `b7:right`，避开了该步 `b3:up` 的 `locked_stand_blocker_delta=1` 和 `precondition_to_lock_conversion=true`；但 step `5` 进入只有 `b3:up` 可选的 recovery 状态，随后仍形成 severe trap 与长循环。

**选项**
1. 将 A24 作为 A23 的修复，并继续提高 root-trap penalty。
2. 回退 A24，认为 stand-lock 转化诊断无效。
3. 将 A24 记录为负结果诊断：root-trap 信号有效，但必须前移为阶段里程碑或多步调度约束。

**结论**
采用选项 3。A24 的首个分歧说明诊断信号本身没有错；它确实把直接制造不可释放 stand-lock 的 `b3:up` 排到低优先级。但替代动作 `b7:right` 只是在局部避免风险，没有打开 target/stand 前置依赖，也没有保持足够合法余量，导致下一步被迫执行同一个坏动作。

**影响**
后续不应继续只调 A24 的 penalty 或 guard 选择键。下一版应把 root-trap 信号前移到更早阶段：在 step `2`/step `3` 之前，将“即将进入只有坏动作可选的低余量瓶颈”作为里程碑失败，并生成多步替代调度。可选实现方向包括阶段级 milestone guard、基于短前瞻的 trap-predecessor 识别，或在 TVA 中补充“避免进入单动作 severe-trap 状态”的训练/审计样本。

## 决策 103

**决策**
将 A25 实现为 `RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner`，作为 A24 之后的 SD-consumer 泛化测试派生类。A25 记录为负结果：它能把 A24 的失败证据前移到 step `2` 并避开 severe stand-lock 路径，但仍不能产生目标进展，说明手工原型泛化不足以替代可学习 TVA/value/ranking consumer。

**背景**
A24 已经证明 root-trap 信号可以识别直接 stand-lock 转化，但单步 root guard 发生得太晚。用户提出 DQN 对表格型 Q-learning 的关键启发：函数逼近使未见状态组合可以共享已学经验。对应到本项目，SD 不应退回端到端黑箱，而应保留显式结构表示，同时让消费端具备跨结构组合泛化能力。因此，本轮在 A24 上通过 OO 继承新增 A25，不改动既有 Boxoban 989/1000 主线，也不改变 A8-A24 的行为。

诊断输出 [room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.json) 显示，同一 `budget=128,smoke_steps=80` 下，A24 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=48,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`；A25 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=49,dependency_count=1,future_dependency_count=0,target_blocker_count=0,release_action_count=0`。A24/A25 首个决策差异为 step `2`。A25 中 `stand_lock_risk_selected_count=0`、`locked_stand_blocker_selected_count=0`、`sd_generalization_selected_severe_count=0`、`sd_generalization_generalized_failure_selected_count=0`、`sd_generalization_trap_predecessor_selected_count=0`，但目标完成数仍为 `0`，并在 fill 阶段形成 `b9:up / b6:down` 循环。

**选项**
1. 将 A25 作为 A24 修复，并继续增加手工 failure/success prototype。
2. 回退 A25，认为 SD 泛化消费方向无效。
3. 将 A25 记录为接口级负结果：保留 SD 泛化消费接口，但下一阶段改为可学习 TVA/value/ranking consumer，并显式纳入 phase progress 与 milestone value。

**结论**
采用选项 3。A25 说明泛化消费方向有效地改变了早期根选择，并避免了 A24 的已知 severe trap；但它也暴露出手工原型的核心缺陷：当前评分更善于识别“不要进入陷阱”，却不能判断“是否朝求解里程碑推进”。这与 DQN 类比一致，真正缺少的是可学习函数对结构组合的价值插值，而不是继续扩充人工规则表。

**影响**
后续不应把 A25 报告为第 2 关求解改进，也不应沿着手工原型继续堆叠规则。A25 应作为 learned SD-consumer 的最小接口原型保留：输入为 dependency、target blocker、stand reachability、release action、legal margin、trap predecessor、phase/stage 等 SD 特征；输出应同时服务 action prior、leaf value 或 ranking。训练或审计样本应覆盖 A24 的 severe trap、A25 的安全无进展循环、A20/A23 的 2-goal 无 release 前缀，以及 teacher/reference 离线轨迹中的 milestone progress。在线求解仍不得查询 teacher/reference 动作。

## 决策 104

**决策**
将 A26-A28 作为 `RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner` 之后的 learned SD consumer 测试线保留。A26 负责建立线性 SD value/ranking 接口，A27 负责识别伪里程碑循环，A28 负责在 root choice 中消费该标签。A28 记录为阶段性改善而非求解配置：它将 XSokoban first 第 2 关从 A26/A27 的 1 个目标推进到 3 个目标和 release 阶段，但仍未解出，release action 仍为 0。

**背景**
A25 证明结构泛化可以避开 A24 severe trap，但停在 0-goal 安全无进展循环。按照用户提出的 DQN 类比，本轮不继续扩充手工原型，而是保持显式 SD 建模，将“可泛化经验共享”落到消费端。为避免影响既有成果，A26、A27、A28 均通过 OO 继承派生：A26 继承 A25，A27 继承 A26，A28 继承 A27；Boxoban 主线、A11 schedule-plan SD 和 online teacher/reference 边界均不改变。

诊断输出显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A25 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=49,release_action_count=0`；A26 为 `max_steps,80 pushes,39 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`；A27 与 A26 行为一致，但将 `sd_value_milestone_selected_count` 从 `31` 降为 `6`，并标记 `sd_value_pseudo_milestone_loop_selected_count=25`；A28 为 `max_steps,80 pushes,45 decisions,boxes_on_goals=3,final_slot_cost=26,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A27/A28 首个决策差异为 step `14`，A28 的 `sd_value_pseudo_milestone_loop_selected_count=0`，`sd_value_inferred_guard_override_count=5`。

**选项**
1. 将 A28 作为第二关求解配置，并直接扩大步数或预算。
2. 回退 A26-A28，认为 learned SD consumer 方向无效。
3. 将 A26-A28 记录为阶段性改善：保留 learned SD consumer 接口与 root 标签消费机制，下一步转向 release 阶段动作生成和值排序。

**结论**
采用选项 3。A26-A28 不是完整求解方案，但它们给出了比 A25 更强的因果证据链：SD value 接口能产生目标进展，伪里程碑标签能正确解释 A26 的循环，root escape 消费该标签后能进一步推进到 3 个目标和 release 阶段。因此 learned SD consumer 方向可行，但当前线性评分与 root guard 仍不足以解决 release 阶段。

**影响**
后续不应把主要精力放在继续惩罚 fill-stage 伪里程碑上，也不应将 A28 报告为已解出 XSokoban 第 2 关。下一版应在 A28 基础上派生，重点处理 release 阶段：生成可执行 `stand_cell_occupied_by_box` 释放动作、在 release-stage ranking 中奖励打开 release action 和降低 future dependency、约束已完成目标不回退，并将 A28 后半段 `release_action_count=0` 的循环状态纳入 TVA/审计样本。在线求解仍保持 teacher/reference 为 false。

## 决策 105

**决策**
将 A29 实现为 A28 之后的 release-action macro 负诊断版本。A29 保留为 `RoomScheduleReleaseActionMacroGFTSDMCTSPlanner`，但不作为求解配置报告；其主要结论是 release 阶段宏入口已接入，问题在当前约束下找不到可采纳释放后缀。

**背景**
A28 已将 XSokoban first 第 2 关推进到 3 个目标和 release 阶段，但 `release_action_count=0`。此前判断认为下一缺口在 release 阶段动作生成，而不是 fill 阶段伪里程碑。因此，本轮在 A28 上通过 OO 继承新增 A29，只在 release 阶段尝试针对 `stand_cell_occupied_by_box` 依赖生成短释放后缀，并继续保持 Boxoban 主线和 online teacher/reference 边界不变。

诊断输出 [room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A28 与 A29 轨迹完全一致：`max_steps,80 pushes,45 decisions,boxes_on_goals=3,final_slot_cost=26,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A28/A29 首个决策差异为 `None`。

同时，A29 不是简单“宏没有运行”。`release_stage_decision_count=20`，`release_action_macro_reason_counts={'not_release_stage':25,'no_release_dependency':1,'no_admissible_release_suffix':19}`。在 release 阶段，宏入口反复绑定到 `stand_cell_occupied_by_box` 依赖；关键站位格和 blocker cell 均为 `[5,1]`，`stand_release_option_count=0`，`locked_stand_blocker_count=1`，`release_action_count=0`。

**选项**
1. 将 A29 的 0 触发解释为代码未接入，继续调试入口。
2. 继续增加 A28/A29 的 root value 惩罚，试图靠 ranking 避开 release 循环。
3. 将 A29 记录为负诊断：入口已接入但当前短后缀搜索与目标不回退约束无法生成可采纳释放动作，下一步转向受控目标回退、更长释放序列或显式 blocker relocation 生成。

**结论**
采用选项 3。A29 已提供足够证据说明 release-action macro 通道本身接入成功，但候选生成和验收条件过窄。当前剩余瓶颈不是“没有识别 release 阶段”，也不是“没有绑定依赖”，而是 `stand_cell_occupied_by_box` 依赖被 goal-locked stand blocker 锁住时，缺少可接受的多步释放或临时回退机制。

**影响**
后续 A30/A29b 应继续沿 OO 继承派生，不修改 Boxoban 正式主线。优先方向为：允许受控的临时目标回退并设置恢复验收、扩大 release suffix 搜索深度与候选生成范围，或新增显式 `goal_locked_stand_blocker_relocation` generator。A29 不得写成 XSokoban 第 2 关求解改进；它的论文和交接表述应为 release 阶段候选生成缺口的可复现负诊断。

## 决策 106

**决策**
将 A30 实现为 A29 之后的 controlled release-regression 与 goal-lock 截断诊断版本。A30 记录为负结果：它能识别并截断会制造 goal-locked stand blocker 的 release 宏，但没有生成替代调度，终态从 A29 的 3 goals 退化为 2 goals。

**背景**
A29 显示 release macro 入口已接入，但在 `[5,1]` 的 `stand_cell_occupied_by_box` 依赖上找不到可采纳后缀。为检验是否只是目标不回退约束过强，A30 首先放宽 release suffix：搜索深度增至 `8`、节点上限增至 `640`，并允许最多 `1` 个目标的受控临时回退。但第一版 A30 与 A29 完全同轨迹，说明事后释放没有产生直接 release 进展。

进一步重放 step `26` 后状态发现，`[5,1]` 上的 blocker 已是 goal-locked box，且该箱没有任何合法 push。因此，A30 增加第二个机制：若继承 release 宏会制造或保留 `goal_locked_stand_blocker_count>0` 且 `release_action_count=0`，则截断该宏并记录 `controlled_release_truncated_goal_lock`。

最终诊断输出 [room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A30 为 `max_steps,80 pushes,52 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`。A29/A30 首个决策差异为 step `27`，`controlled_release_goal_lock_truncation_count=1`，但 `release_action_macro_triggered_count=0`、`release_action_macro_opened_release_count=0`。

**选项**
1. 将 A30 作为改进，因为它截断了 goal-locking macro。
2. 回退 A30，认为 A29 的 3-goal 前缀更好。
3. 将 A30 记录为负诊断：保留它揭示的 goal-locking macro 证据，但下一步改为生成替代调度，而不是只截断危险宏。

**结论**
采用选项 3。A30 的截断机制本身有效：step `25` 的原 `4` push 宏末态会产生 `goal_locked_stand_blocker_count=1`、`locked_stand_blocker_count=1`、`release_action_count=0`，A30 截断后的保留前缀避免了该锁定。但截断没有提供替代动作序列，后续进入 2-goal release 循环，目标数低于 A29。

**影响**
A30 不应报告为求解改进，也不应简单回退为“无效”。它把问题边界前移到 release 宏生成之前：需要在进入 goal-locking macro 前生成能同时保持目标进展、避免 goal-lock、并打开 release/dependency 进展的替代候选。下一版应优先派生 A31/A30b，围绕 `goal_locking_macro_predecessor` 做 root 候选生成或 release-stage relocation generator；单纯放宽事后目标回退已经不足。

## 决策 107

**决策**
将 A31 实现为 A30 之后的 goal-lock predecessor release replacement 诊断版本。A31 记录为机制推进而非求解改进：它能把 A30 的危险宏截断信号前移到 root 替代动作选择，并打开 release action；但第二关仍停在 2 goals，后续缺口转为 release action 打开后的循环抑制和下一阶段目标推进。

**背景**
A30 已证明 step `25` 的继承宏会制造 `goal_locked_stand_blocker_count=1`、`locked_stand_blocker_count=1`、`release_action_count=0` 的后继状态，并且只做截断会使终态从 A29 的 3 goals 退化为 2 goals。因此，下一轮不应继续事后释放 `[5,1]`，而应在该宏被采用前生成替代 release 调度候选。

A31 通过 OO 继承新增 `RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner`，继承 A30。该版本在 A30 返回 `controlled_release_truncated_goal_lock=true` 时扫描当前 root 已展开候选，筛选不降低目标数且不会立即产生 goal-locked stand blocker 的替代动作，并优先选择能打开 release action、降低 dependency/target blocker、改善合法余量或降低 slot schedule cost 的候选。在线求解仍保持 `teacher_actions_used_online=false`。

诊断输出 [room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A31 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`。A30/A31 首个决策差异为 step `25`。A31 的 `goal_lock_predecessor_override_count=2`、`goal_lock_predecessor_structural_progress_count=2`、`goal_lock_predecessor_opened_release_count=2`，两次均将原 `b6:down` 替换为 `b6:up`，替代后 `goal_delta=0` 且 `release_action_count_delta=2`。

**选项**
1. 将 A31 作为求解改进，因为它打开了 release action。
2. 回退 A31，认为 A30/A29 的目标数或前缀更有价值。
3. 将 A31 记录为机制推进：保留 root 前置替代机制，但下一轮重点处理 release action 打开后的循环和值排序。

**结论**
采用选项 3。A31 相对 A30 明确推进了机制边界：A30 只能截断坏宏，A31 能在 root 层用替代动作打开 release action。但 A31 没有恢复 A29 的 3 goals，也没有解出第二关；step `25` 后仍在 release 阶段围绕 `b3:down` 类 committed-slot 动作循环，dependency 和 target blocker 没有继续消解。

**影响**
A31 不应写作 XSokoban 第 2 关求解配置，也不应简单回退。后续 A32/A31b 应继承 A31，针对 `release_action_count>0` 的状态增加 root ranking 与循环抑制：识别 release-open 之后的 committed-slot 伪里程碑，奖励继续降低 dependency/target blocker 或增加 goals 的动作，并避免在 release-open 状态反复选择无结构进展的宏。A31 的 step `25`、step `30` 替代样本和后续 `b3:down` 循环应作为 TVA/审计负例。

## 决策 108

**决策**
将 A32 实现为 A31 之后的 learned SD consumer 接口版本，并将 learned scorer 的 root hard override 默认关闭。A32 保留为“显式 SD 表示 + 可学习消费者 + exact MCTS”的接口原型，不作为 XSokoban 第 2 关求解配置报告。

**背景**
前序 A24-A31 已形成一条可解释诊断链，但也暴露出继续添加手工规则的边际收益下降。用户明确要求回到第一性原理和通用机制：显式 SD 负责机制建模，神经网络负责跨状态组合的表达与经验复用，MCTS 负责精确动力学下的搜索。因此，本轮在 A31 上通过 OO 继承新增 `RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner`，并固定 `44` 维 learned SD consumer 特征契约。

A32 离线样本由 A28-A31 trace 构造，输出 [room_gft_a32_learned_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_dataset.json) 与 [room_gft_a32_learned_sd_consumer_linear_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_linear_checkpoint.json)。当前样本数为 `6845`，线性 ridge trace 蒸馏训练内 `MAE=0.1717,corr=0.9922`。该结果只证明接口自洽，不证明模型具备泛化求解能力。

诊断显示，若允许 weak checkpoint 直接覆盖 root choice，则行为显著退化。开启 `--learned-sd-consumer-root-override` 的 30-push 对照 [room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json) 为 `status=no_action,executed_pushes=6,closed_loop_decision_steps=4,boxes_on_goals=0`，A31/A32 首个决策差异发生在 step `1`。

关闭 hard override 后，A32 soft-consumer 80-push 复核 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json) 与 A31 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A32 首个决策差异为 `None`。同时，learned scorer 在 `42` 个决策中均被加载，score 范围为 `[-3.9273,3.7098]`，说明接口已接通但未改变闭环行为。

**选项**
1. 继续让 learned scorer 默认硬覆盖 root choice，期望后续调权修复早期崩溃。
2. 完全移除 A32 learned consumer，回到 A31 手工机制链。
3. 将 learned scorer 默认作为软 prior/value 消费；保留 hard override 仅作显式诊断开关，并把 A32 作为后续神经 TVA/ranking consumer 的稳定接口。

**结论**
采用选项 3。弱 trace 蒸馏 checkpoint 不应直接接管根动作；否则会把 learned consumer 变成一条未经校准的硬规则，并破坏 exact MCTS 的稳定性。A32 的价值在于固定可学习消费者的输入、输出、trace 和复现边界，而不是立即提升第 2 关解题率。

**影响**
后续难题线应停止继续扩展 A24-A31 式零散规则，把这些版本转为样本来源、失败模式库和审计标签。下一步应训练更正式的 neural/value/ranking consumer，样本需覆盖 A31 release-open 后的 `b3:down` 循环、A28 的 3-goal release 前缀、A24/A25 的 severe trap 与安全无进展循环，并报告 soft prior、leaf value、ranking override 的单因素消融。在线求解仍必须保持 `teacher_actions_used_online=false`，Boxoban 989/1000 主线不得被修改。

## 决策 109

**决策**
将 A33 实现为 A32 之后的 neural SD consumer 诊断版本。A33 记录为神经消费者接口推进而非求解改进；默认保持 soft prior/value 消费，root hard override 继续关闭。后续重点转为训练目标与 release-stage 价值校准，而不是继续添加手工规则或开启 hard override。

**背景**
A32 固定了 learned SD consumer 的特征契约和 checkpoint 前向，但线性 trace 蒸馏 soft prior 在 80-push 下与 A31 完全同轨迹，hard override 则早期 `no_action`。因此需要检验真正的神经函数逼近是否能以软消费方式改变搜索轨迹，同时保持显式 SD 表示、exact MCTS 和 online teacher 禁用边界。

A33 通过 OO 继承新增 `RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner`，继承 A32。训练脚本 [train_room_gft_neural_sd_consumer_a33.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_neural_sd_consumer_a33.py) 从 A24-A32 trace 构造 `12462` 个候选样本，按 root group 切分训练/验证，并训练 `[32,16]` MLP。输出 [room_gft_a33_neural_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_mlp_checkpoint.json) 的验证指标为 `MAE=0.0696,corr=0.9986,target_top1_accuracy=0.8642,selected_top1_accuracy=0.5389`。

诊断输出 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A33 为 `max_steps,80 pushes,46 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`。A31/A33 首个决策差异为 step `14`，说明 neural soft prior 已改变闭环搜索轨迹；但终态目标数和 slot cost 未超过 A31/A32。

hard-override 对照 [room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json) 仍为 `status=no_action,executed_pushes=6,boxes_on_goals=0`，A31/A33 首个决策差异为 step `1`。这说明即使 MLP 分数比 A32 线性分数更平滑，未经校准的 root hard override 仍会破坏闭环稳定性。

**选项**
1. 将 A33 作为求解改进，因为它改变了 A31/A32 轨迹并增加了 release 替代触发。
2. 开启 A33 root hard override，进一步放大神经模型对动作选择的控制权。
3. 将 A33 记录为 neural consumer 接口推进：保留 soft prior，关闭 hard override，下一步改进训练目标和 release-stage 标签。

**结论**
采用选项 3。A33 已证明神经消费者能进入闭环并改变轨迹，但当前训练目标仍是启发式 SD progress target，缺少 solved-distance、milestone、release-after-open 真进展等监督。它不能被报告为第 2 关求解突破，也不应通过 hard override 强行接管根动作。

**影响**
后续 A34/A33b 应优先改进数据和目标，而不是继续新增手写规则。可行方向包括：引入离线 teacher/search 轨迹中的 solved-distance 或 milestone 标签；对 A31/A33 release-open 后的循环构造 hard negative；把目标拆成 value、trap risk、release-after-open progress 三个头；或在 leaf value 层而非 root hard override 中接入神经评分。所有新实验仍需报告 `teacher_actions_used_online=false`、`mainline_mutation=false`，并保持 Boxoban 989/1000 主线隔离。

## 决策 110

**决策**
将 A34 实现为 A33 之后的 future-window milestone target 诊断版本。A34 记录为负结果：milestone target 的理论方向更接近“未来求解进展”，但当前单一 scalar 目标尺度偏正、判别力不足，soft prior 未改变 A31/A32 轨迹，hard override 仍早期退化。

**背景**
A33 已证明 neural soft prior 能改变闭环轨迹，但仍未求解第 2 关。其核心问题是训练目标仍主要拟合即时 SD progress，而不直接刻画 release-open 后的未来目标推进。因此，本轮不继续新增手写规则，而是改为训练 A34 milestone checkpoint：每个 trace step 向前看 `8` 个闭环决策，奖励未来窗口内的 goal、release、dependency、target blocker 和 slot 里程碑，惩罚 release-open stagnation、无未来里程碑和目标回退。

A34 通过 OO 继承新增 `RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner`，继承 A33。训练输出 [room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json) 的验证指标为 `MAE=0.3929,corr=0.9371,target_top1_accuracy=0.8318,selected_top1_accuracy=0.5831`。该目标比 A33 更难拟合，但 selected-top1 高于 A33，说明它更贴近部分历史 planner 选择。

诊断输出 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A34 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`，与 A31/A32 完全同轨迹，A31/A34 首个决策差异为 `None`。score 范围 `[-0.5022,3.0]`，多个前缀分数达到 clip 上限。

hard-override 对照 [room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json) 仍为 `status=no_action,executed_pushes=6,boxes_on_goals=0`，A31/A34 首个决策差异为 step `1`。

**选项**
1. 将 A34 作为改进，因为目标更贴近 future milestone。
2. 通过提高权重或启用 hard override 强行放大 A34 的影响。
3. 将 A34 记录为负诊断：保留 future-window 目标思路，但下一步转向目标校准、多头分解或引入更强离线 solved-distance 标签。

**结论**
采用选项 3。A34 证明“只把即时 target 换成未来窗口 scalar target”还不足以解决难题；目标尺度和语义需要进一步校准。当前模型对前缀普遍给出偏正分数，无法在 soft prior 中形成有效差异；hard override 仍破坏闭环稳定性。

**影响**
后续 A35/A34b 不应继续沿单一 scalar score 直接加权。优先方向为：将 consumer 输出拆成 value、release-after-open true progress、loop/trap risk 多头；将 A31/A34 后半段 `b3:down` release-open 循环作为明确 hard negative；引入离线 teacher/search 的 solved-distance 或 remaining-plan-length 标签；或把 learned value 接入 leaf value/rollout evaluation，而不是 root hard override。在线求解仍保持 `teacher_actions_used_online=false`，并继续隔离 Boxoban 主线。

## 决策 111

**决策**
将 A35 实现为 A34 之后的 multi-head SD consumer 诊断版本。A35 记录为接口推进但闭环持平：四头目标可训练、可审计，组合目标验证排序优于 A34，但默认 soft prior 在 80-push 复核中仍与 A31/A34 完全同轨迹，未突破第二关的 2-goal release-open 循环。

**背景**
A34 说明单一 future-window milestone scalar 存在机制混叠和尺度偏正问题。用户同时明确要求沿“显式 SD 结构表示 + 神经函数逼近 + exact MCTS”的方向推进，并避免把模型变成大量零散规则堆砌。因此本轮采用 OO 继承新增 `RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner`，继承 A34，只扩展 checkpoint 消费方式，不新增 Sokoban 动作规则，不修改 Boxoban 主线。

A35 训练脚本 [train_room_gft_multihead_sd_consumer_a35.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_multihead_sd_consumer_a35.py) 从 A24-A34 trace 构造 `13928` 个候选样本，按 root group 稳定切分训练/验证，并分别训练 value、release-progress、loop-risk、trap-risk 四个 `[32,16]` MLP。输出 [room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json) 的组合目标验证指标为 `MAE=0.2662,corr=0.9612,target_top1_accuracy=0.8799,selected_top1_accuracy=0.6193`。

诊断输出 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A35 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A35 首个决策差异为 `None`。multi-head scorer 加载 `42` 次，score 范围 `[-0.3828,2.5846]`。

30-push 小预算 smoke [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json) 在 step `14` 偏离 A31，但结果只到 `1` goal。这说明多头 soft prior 已能影响早期 root ranking，但该影响未稳定转化为求解进展。

**选项**
1. 将 A35 作为方法改进，因为多头验证指标优于 A34。
2. 继续提高 A35 权重或启用 root hard override，以强制多头模型改变根动作。
3. 将 A35 记录为多头接口诊断：保留四头表达，但下一步转向目标校准、hard negative 强化和消费位置调整。

**结论**
采用选项 3。A35 解决了 A34 在诊断解释上的一部分问题，但没有解决闭环控制问题。后段重复 `b3:down` 时，value 头为负、loop-risk 头较高，但 release-progress 头仍较高，组合后仍给出约 `0.1639` 的正分，说明当前多头合成仍会奖励 release-open 循环中的伪进展。

**影响**
后续 A36/A35b 不应继续增加更多零散规则或简单放大权重。优先方向为：把 release-open 后的重复 committed-slot 循环作为更强 hard negative，重新校准 release-progress 与 loop-risk 的相对权重；把多头输出接入 leaf value 或 rollout evaluation，而不是仅作为 root soft prior；必要时引入离线 teacher/search 的 remaining-plan-length 或 solved-distance 标签。在线求解仍必须保持 `teacher_actions_used_online=false`、`mainline_mutation=false`，并继续隔离 Boxoban 989/1000 主线。

## 决策 112

**决策**
将 A36 实现为 A35 之后的最小 agenda progress-certificate gate 诊断版本。A36 记录为负诊断：release opened/consumed 区分和证书字段已经接入，但仅靠局部 delta 构造 certificate 仍不足以识别 release-open 后的重复 `b3:down` 循环。下一步应进入 active agenda / blocker-set 层，而不是继续调 A35/A36 的 release 权重。

**背景**
专家 27 点建议指出，当前 Room-GFT/SD 的主要缺口不是“再少一个规则”，而是缺少 active obligation 与可执行任务议程。为避免一次性重写 MCTS，本轮先实现其中最小命题：release-progress 只有在产生 progress certificate 时才是真进展。A36 通过 OO 继承新增 `RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner`，继承 A35，复用 A35 multi-head checkpoint，不训练新模型，不改动作生成，不改 exact dynamics，不在线调用 teacher。

A36 在线为每个候选记录 `release_opened`、`release_consumed`、`stale_open_release`、`pseudo_progress`、`certified_progress`、`agenda_tier`、`release_head_raw` 与 `release_head_effective`。若打开 release 但未消费，则将 release head 截断到 `0.35`；若没有证书，则将 release head 置零并施加伪进展惩罚。

诊断输出 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A36 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A36 首个决策差异为 `None`。证书计数为 `loaded=42,certified=39,pseudo=1,stale_open_release=0,release_opened=4,release_consumed=0,agenda_loop=0`。

关键负证据是后段重复 `b3:down`。step `31` 至 step `41` 的重复动作仍被局部证书标记为 `certified_progress=True,tier=6`，`release_head_raw=2.3367` 且 `release_head_effective=2.3367`，合成分数约 `0.2864`。这说明局部 slot/dependency/target delta 仍会把形式进展误判为真实 obligation progress。

**选项**
1. 继续调低 release-progress 权重或调高 loop-risk 权重。
2. 将 A36 作为有效修复，因为它已经截断了部分 opened-not-consumed release head。
3. 将 A36 记录为最小证书负诊断：保留证书字段和 opened/consumed 区分，但下一步必须引入 active agenda 与 blocker-set。

**结论**
采用选项 3。A36 验证了 Agenda-GFT 方向的必要性，但也证明最小局部 certificate 不够。它能记录和部分门控 opened-not-consumed release，却不能判断某个动作是否服务于“当前必须完成的义务”。因此，真正需要的是 `score(s,g,a)` 中的 `g`，以及当前义务的 blocker set，而不是进一步调线性权重。

**影响**
后续 A37 应实现 active agenda 最小版本，至少覆盖 `FillFrontierSlot`、`ReleaseBlocker` 和 `ConsumeRelease` 三类义务，并让 certificate 绑定 active obligation 的 blocker reduction。A36 的后段 `b3:down` 重复段应作为 A37 的首要负例：如果 active blocker 没有减少、opened release 没有被消费，即使局部 slot/release 指标为正，也应标为 off-agenda 或 pseudo progress。在线求解仍保持 `teacher_actions_used_online=false`、`mainline_mutation=false`，Boxoban 989/1000 主线继续隔离。

## 决策 113

**决策**
将 A37 实现为 A36 之后的最小 active-agenda / blocker-set certificate 诊断版本。A37 记录为有效诊断但非闭环修复：它能把 A36 后段重复 `b3:down` 从 `certified_progress=True` 改判为 off-agenda release-loop，但同一 80-push 复核仍与 A31/A36 完全同轨迹。

**背景**
A36 证明局部 progress certificate 不足以处理 release-open 后的循环，因为局部 dependency/target/slot delta 会把形式上的 release closure 误认为真实进展。专家 27 点建议要求从 `score(s,a)` 走向 `score(s,g,a)`，其中 `g` 是当前 active obligation。因此本轮通过 OO 继承新增 `RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner`，继承 A36，复用 A35 multi-head checkpoint，不训练新模型，不改动作生成，不改 exact dynamics，不在线调用 teacher/reference。

A37 定义最小 agenda 类型：`release_blocker`、`consume_release`、`fill_frontier_slot`。其中 `consume_release` 只在 `before_release_action_count>0` 时激活；release 阶段但尚未打开 release 的状态仍归为 `release_blocker`。A37 为每个候选记录 active blocker before/after、blocker delta、active agenda progress、off-agenda action、release loop 和 agenda reason，并用这些字段覆盖 A36 的 `certified_progress`、`pseudo_progress`、`agenda_tier` 与 `release_head_effective`。

诊断输出 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A37 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A37 首个决策差异为 `None`。计数为 `active_agenda_loaded_count=42,active_agenda_progress_count=25,off_agenda=17,pseudo=17,stale_open_release=13,release_loop=13,release_consumed=1`，类型分布为 `{release_blocker:28,consume_release:14}`。

关键证据是 step `31` 至 step `41`。A37 将重复 `b3:down` 标记为 `active_agenda_type=consume_release,active_agenda_reason=repeated_release_closure,active_agenda_progress=False,active_agenda_off_agenda_action=True,active_agenda_release_loop=True,certified_progress=False,tier=1,release_head_effective=0`，learned score 约 `-2.4741`。这正好修正了 A36 的 `certified_progress=True,tier=6,release_head_effective=2.3367` 误判。

同时，step `31` 的 top-child 证据显示，`b3:down` 尽管已被 A37 判为 off-agenda/release-loop，仍保有 `prior_score=266.0073,room_sd_value.score=150,visits=120`；其他候选的 prior 大多为负且 visits 为 `1`。因此，失败原因已从“证书层看不出伪进展”转移为“soft prior 不能约束旧启发式 prior 与 MCTS 访问优势”。

**选项**
1. 继续放大 A37 的 off-agenda/release-loop 惩罚。
2. 将 A37 作为闭环修复，因为它已经把 repeated release closure 置为负分。
3. 将 A37 记录为 active-agenda 诊断推进：保留证书纠错，但下一步把 active agenda 接入 hard gate、agenda-option 候选或 leaf/rollout value。

**结论**
采用选项 3。A37 已完成 A36 之后最小 active agenda 的诊断目标，证明“当前义务”这一状态变量确实必要；但它也证明，仅把 active agenda 作为 root soft prior 的附加分不足以改变闭环。继续调惩罚会把问题退回线性权重试错，不能解决消费位置不匹配。

**影响**
后续 A38 不应继续扩大零散规则或只调 A37 分数。优先方向为：在 root 选择前对 active-agenda off-agenda/release-loop 候选做 hard gate 或强降级；将 active agenda 变成显式 option，使 MCTS 先选择义务再选择动作；或把 active agenda certificate 接入 leaf value/rollout evaluation，让搜索树内部而非仅 root prior 消费该信号。在线求解仍保持 `teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 114

**决策**
后续 hard Sokoban 派生实验必须显式对齐专家 27 点 Agenda-GFT 建议，不得在未经论证的方向上临时扩展。每个新版本进入实现前，应说明它对应 27 点中的哪一项或哪一个最小实验路线，并说明它如何由 A36/A37 的证据自然推出。

**背景**
A36 已对应专家建议中的 E1：给 release-progress 加 progress certificate gate。A37 已对应 E2：引入 active agenda，不改图网络。A37 结果显示，active-agenda 证书已经能把重复 `b3:down` 改判为 off-agenda release-loop，但 root 仍选择该动作，说明下一缺口位于 active agenda 的 MCTS 消费位置，而不是 SD 证书本身。

专家 27 点中与下一阶段直接相关的既定路线包括：E3 leaf value 使用 agenda potential，第 8 点 learned consumer 不应只做 root soft prior，第 13 点 root prior/leaf value/rollout evaluation 分工，第 14 点 agenda-option tree，第 16 点 lexicographic tier 替代单一线性合成分数。相反，第 21 点明确不建议继续堆局部规则、只调 A35/A37 权重、只扩大 MCTS budget、直接端到端学习 44 维到 action，或在线调用 teacher。

**选项**
1. 继续按局部失败现象即时增加修补规则。
2. 直接转入新的大模型、图网络或 teacher imitation。
3. 建立版本准入约束：每个后续实验必须映射到 27 点既定路线，并保留 OO 继承、主线隔离、teacher online 禁用和文档同步。

**结论**
采用选项 3。A38 起不再接受“看起来可能有效但未在 27 点或 A36/A37 证据链中论证”的新方向。若确需偏离既定路线，必须先形成单独分析和决策记录，再进入代码实现。

**影响**
A38 的优先候选应限定在 A37 证据自然推出的消费端问题：最保守方向是 E3，即将 active-agenda potential 接入 leaf value；若采用 hard gate 或 tier，则必须明确对应第 16 点，并说明它只是把 A37 已有 `agenda_tier/off_agenda/release_loop` 从 soft prior 转为比较层约束，而不是新增零散规则。所有实验继续保持 `teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 115

**决策**
将 A38 记录为 Agenda-GFT E3 的负诊断版本：active-agenda potential 已接入 MCTS leaf value 且可被 trace 审计，但在 XSokoban first 第 2 关 80-push 复核中未改变 A31/A37 的闭环轨迹。A38 不作为求解改进报告，后续不得通过继续调 leaf potential 权重来替代新的机制决策。

**背景**
A37 已证明 active agenda 能把 release-open 后的重复 `b3:down` 改判为 off-agenda release-loop，但 root 仍选择该动作。根据决策 114，A38 只能执行专家 27 点中已讨论的 E3：让 leaf value 使用 agenda potential，检验 active agenda 是否需要进入 MCTS 树内消费端。

A38 通过 OO 继承新增 `RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner`，继承 A37。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增动作生成规则，不接入 online teacher/reference，不引入 hard gate、agenda-option tree 或图网络。leaf adjustment 由 `neutral_offset + bonus - penalty` 显式构成，并记录 leaf eval、loop eval、stale release eval 等审计字段。

诊断输出 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A38 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A38 首个决策差异为 `None`。A38 的 leaf potential 加载 `42` 次，leaf eval 总数为 `5302`，adjustment 范围为 `[-14.31,3.93]`，其中 release-loop leaf eval 为 `129`，stale-release leaf eval 为 `74`。

关键证据是 step `31`。A38 对重复 `b3:down` 所处段落给出 `agenda_leaf_adjustment=-14.31`，同时 A37 证书已将其标记为 `active_agenda_type=consume_release,active_agenda_progress=False,active_agenda_off_agenda_action=True,active_agenda_release_loop=True,certified_progress=False,tier=1`。但是该候选仍有 `prior_score=266.0073,room_sd_value.score=150,visits=120`，继续成为 root top-child。

**选项**
1. 继续放大 A38 的 leaf loop penalty 或 stale-release penalty。
2. 将 A38 作为闭环改进，因为 leaf value 已经产生负修正。
3. 将 A38 记录为 E3 负诊断：保留 leaf potential 与审计字段，但承认单纯 additive leaf potential 不足以约束 root/top-child 选择；下一步必须在专家 27 点内另行决策消费结构。

**结论**
采用选项 3。A38 完成了 E3 的最小可验证实验，证明 active-agenda 信号可以进入 leaf value 并产生可观测负修正；但 additive leaf potential 没有改变实际闭环动作，说明问题不是再调一个 leaf 权重即可解决。当前失败点仍是旧启发式 prior 与 MCTS visits 对 root/top-child 的支配。

**影响**
A38 之后的候选不能继续沿“调 leaf potential 参数”推进。若继续实验，应在专家 27 点中选择并记录新的机制入口：第 16 点 lexicographic tier / hard downgrade，或第 14 点 agenda-option tree。任何后续版本仍须保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`，并在实现前说明其与 27 点的对应关系。

## 决策 116

**决策**
将 A39 实现为 Agenda-GFT 第 16 点的证据限定版本：只在 release-open 后的 `consume_release` 低 tier release-loop/off-agenda 场景中启用 lexicographic tier / hard downgrade，而不采用全局 tier 接管。A39 记录为有效机制推进但非求解突破。

**背景**
A37 已能识别 release-open 后的 repeated release closure，A38 已将该信号接入 leaf value，但两者都无法改变 root 选择。A38 的 step `31` 证据显示，tier-1 release-loop 候选仍可凭旧启发式 prior 和 visits 成为 top-child。因此，A39 按专家 27 点第 16 点，把 `agenda_tier` 从软分数转为 root choice 前的层级比较。

初始思路若做成全局 tier 接管，会在 fill 阶段过早改写 root 选择，超出 A37/A38 已验证的证据范围。为避免从第 16 点漂移为未经论证的全局动作选择器，A39 默认采用 `release_loop_only`：只有父类选择满足 `active_agenda_type=consume_release`、`agenda_tier<=1`，且存在 release-loop、off-agenda 或 stale-open-release 时才触发；替代候选也限定为同一 active agenda 下的非 release-loop `consume_release` root edge。

A39 通过 OO 继承新增 `RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner`，继承 A38。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增动作生成规则，不接入 online teacher/reference，不引入 graph model 或 agenda-option tree。诊断输出 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A39 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A39 首个决策差异为 step `26`，`agenda_tier_choice_override_count=20`。

关键正证据是 step `26`：父类候选 `b3:down` 为 `tier=1,off-agenda=true,release-loop=true`，且有 `visits=110,prior_score=292.9614,room_sd_value.score=150`；A39 改选 `b4:up`，该候选为 `tier=7,active_agenda_progress=true,off-agenda=false,release-loop=false`，尽管它只有 `visits=1`。这证明 lexicographic tier 可以压过旧 prior 和 visits，确实改变了 A37/A38 的失败模式。

同时，A39 没有解出第二关。它将连续 `b3:down` 循环打散后，后段大量选择 tier-2 off-agenda `consume_release` 候选，仍无法持续产生目标推进或消解 blocker。

**选项**
1. 采用全局 tier 接管，使所有阶段的高 tier 候选均可覆盖父类 root 选择。
2. 继续只调 A38 leaf potential 或 A37 soft prior 惩罚。
3. 采用证据限定的 `release_loop_only` tier choice，将 A39 记录为第 16 点的最小机制推进，并把下一缺口定位到可持续 agenda-option / milestone 候选不足。

**结论**
采用选项 3。全局 tier 接管会把 A39 扩展到 A37/A38 未论证的阶段，容易造成早期退化；只调软分数或 leaf value 已被 A37/A38 否定。`release_loop_only` 版本正好检验已确认的失败点，并证明 root 层级消费是必要机制。

**影响**
A39 不应写作 XSokoban 第 2 关求解配置，也不应继续通过调 tier 阈值来追求偶然成功。后续若继续，应在专家 27 点内评估第 14 点 agenda-option tree，或先补充 active agenda 的可执行里程碑候选生成，使 tier-5/7 真进展候选能够持续出现。所有后续实验仍须保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 117

**决策**
将 A40 实现为 Agenda-GFT 第 14 点的保守 agenda-option evaluation 版本，而不是完整 option commitment 版本。A40 记录为有正诊断信号但非闭环修复：它能在 A39 后段识别短程可完成 `consume_release` 的 tier-7 option，但由于只执行 first edge、不承诺执行 suffix，80-push 复核与 A39 完全同轨迹。

**背景**
A39 证明 lexicographic tier 可以改变 A37/A38 的 release-loop root 选择，但后段仍大量进入 tier-2 off-agenda `consume_release` 候选。专家 27 点第 14 点建议将搜索拓扑从 push tree 改为 agenda node、option node、push node。为避免一次性重写 MCTS 节点状态，A40 先做最小可审计版本：在 root 处为候选第一步运行 bounded local option evaluator，使用真实 legal push 搜索判断该第一步是否能在短程内完成当前 active agenda。

A40 通过 OO 继承新增 `RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner`，继承 A39。该版本不训练新 checkpoint，不新增动作生成规则，不在线调用 teacher/reference，不改变 exact dynamics，也不把 option suffix 作为宏动作执行。默认局部树预算为 `agenda_option_tree_max_pushes=5,agenda_option_tree_max_nodes=96,agenda_option_tree_max_root_candidates=10`。深层 commitment 采用 `RoomSlotOption(kind="none")` 的保守推进，避免在 A40 中引入新的 slot-option 分支生成机制。

诊断输出 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A40 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A39/A40 首个决策差异为 `None`。option tree 加载 `53` 次，override `0` 次，selected progress `22` 次，selected done `22` 次，selected loop `0` 次，总评价 option 数 `186`，最大 selected depth 为 `4`。

关键证据是 step `29` 以后。A40 找到三步左右的 `consume_release` 完成路径，例如 `b6:up -> b9:left -> b4:down`，其局部评价为 `agenda_done=true,release_consumed=true,active_blocker_delta=5,dependency_total_delta=3,target_blocker_count_delta=2,after_active_blocker_count=0`。但是闭环只执行 `b6:up`，下一步重新规划后可能偏离该 suffix。step `30` 和 step `31` 的 `b3:left -> b9:left -> b4:down` 也显示同类现象。

**选项**
1. 将 A40 改为直接执行发现的 suffix，作为新的宏动作。
2. 继续调 A40 的局部 option 分数、节点预算或 root override 阈值，追求直接改变 A39 轨迹。
3. 将 A40 记录为保守 option-evaluation 诊断：承认短程 option 信号存在，但下一步必须设计显式 option commitment 或将 active agenda/option 写入 MCTS 节点状态。

**结论**
采用选项 3。选项 1 会把 A40 从“评价 option”变成新的宏动作生成器，超出本轮已论证边界；选项 2 仍停留在 root 分数调参，不能解决“第一步之后不再承诺同一 option”的问题。A40 的证据表明，当前系统并非完全缺少短程义务完成信息，而是缺少持续履行同一 obligation-option 的机制。

**影响**
后续 A41 若继续推进，应仍限定在专家第 14 点和第 6 点附近：将 active option 或 agenda commitment 写入闭环状态/MCTS 节点，使同一 option 在若干步内保持有效，直到 agenda 完成、出现 irreversible risk、遇到 room/commitment boundary 或形成新 frontier。不得将该结果解释为“只需扩大 MCTS budget”或“只需继续调 root 分数”。所有后续实验仍须保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 118

**决策**
将 A41 实现为 Agenda-GFT 第 14 点和第 6 点之间的受限 option commitment 诊断版本。A41 记录为机制验证但非求解突破：它能执行 A40 已发现的 tier-7 `consume_release` suffix，并把 release/dependency/target blocker 清零；但 80-push 复核仍停在 `2` goals，后续缺口转移到 release 完成后的 fill 阶段里程碑和目标选择。

**背景**
A40 已证明短程 `consume_release` agenda-done option 可见，但闭环只执行第一步，下一轮 replanning 可能偏离该 option。因此 A41 不再调 root score，也不扩大 A40 局部树预算，而是检验同一 option 是否需要跨若干 push 保持承诺。该方向直接来自专家 27 点第 14 点的 agenda-option tree 和第 6 点的 agenda state persistence。

A41 通过 OO 继承新增 `RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner`，继承 A40。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。承诺触发条件为：A40 的 selected option 存在、`agenda_tier>=7`、`agenda_done=true`、option 长度不少于 `2` 个 push，且父层 corridor/release 宏没有先行生成多步宏。触发前会重新验证每个 suffix push 的合法性、deadlock child 和 path loop。

诊断输出 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A41 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=41,dependency_count=0,target_blocker_count=0,release_action_count=0`。A40/A41 首个决策差异为 step `30`。commitment 加载 `53` 次，触发 `1` 次，首个触发为 step `29`，额外执行 `2` 个 suffix push。

关键正证据是 step `29`：A41 执行 `b6:up -> b9:left -> b4:down`，最终 progress 为 `agenda_done=true,release_consumed=true,release_action_count_delta=-2,active_blocker_delta=5,dependency_total_delta=3,target_blocker_count_delta=2,after_active_blocker_count=0`。这说明 A40 的 option 不是评分幻觉，缺失的承诺机制确实可以把短程义务完成。

同时，A41 的负证据也很明确：release 义务完成后，系统进入 fill 阶段但未在 80-push 内推进到第三个目标，`slot_schedule_cost` 长期停在 `41`。因此，单纯对 `consume_release` suffix 做承诺不能求解第二关。

**选项**
1. 将 A41 作为求解方向继续强化，扩大承诺长度或降低触发阈值。
2. 因最终 slot cost 高于 A40 而回退到 A40，不再承认 option commitment 的必要性。
3. 将 A41 记录为必要但不充分的机制验证：承认承诺机制能完成 release 义务，但下一步问题已经转移到阶段化 fill 里程碑、slot-distance/value 表达和阶段切换后的目标选择。

**结论**
采用选项 3。选项 1 会把 A41 扩展为未经论证的 suffix 执行器，容易重新走向局部规则堆叠；选项 2 则忽略了 A41 已经把 A40 的 release-open/dependency/target blocker 状态推进到清零。A41 的结果应解释为“当前 release obligation 的执行持续性缺口已被验证，但更高层阶段目标仍缺失”。

**影响**
后续版本不应继续只围绕 `consume_release` 增加更长承诺或更多特例。若继续推进，应仍显式对齐专家 27 点：要么把 active agenda/option 真正写入 MCTS 节点状态并在阶段切换时重建 agenda，要么补充 release 完成后的 fill-frontier milestone/value 表达和可复用函数近似；所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 119

**决策**
将 A42 实现为 A41 之后的 post-release fill-frontier option 诊断版本。A42 记录为局部机制推进但非求解突破：它能在 clean fill 阶段完成一个 frontier slot，并将最终 slot cost 从 A41 的 `41` 降到 `35`；但 80-push 复核仍停在 `2` goals，失败点转移到 frontier completion 之后的新 release-blocker option 和阶段状态持久化。

**背景**
A41 已经验证 `consume_release` option commitment 可以把 release、dependency 和 target blocker 清零，但随后进入 fill 阶段后长期停在 `slot_schedule_cost=41`。该证据说明，下一问题不再是继续扩大 `consume_release` suffix，而是 release 完成后的 fill frontier milestone 是否能作为独立 agenda boundary 被识别和消费。该方向对应专家 27 点中关于 frontier obligation、agenda-option tree 和 next agenda milestone 的建议。

A42 通过 OO 继承新增 `RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner`，继承 A41。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。触发条件被限制在 clean post-release fill context：`stage=fill`、`release_action_count=0`、`dependency_total=0`、`target_blocker_count=0`、已有有效填充 slot，并且父类候选对 `fill_frontier_slot` 处于低 tier 或 off-agenda。候选必须不回退目标，且完成当前 frontier slot 或产生足够 slot 进展。

诊断输出 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A42 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A41/A42 首个决策差异为 step `30`。`fill_frontier_option_override_count=1`，`fill_frontier_option_selected_completion_count=1`，`fill_frontier_option_selected_slot_progress_count=1`。

关键正证据是 step `30`：A42 改选 `b4:down`，并通过既有 corridor macro 执行 `b4:down -> b6:down -> b6:left`。局部评价为 `frontier_completed=true,slot_progress=true,goal_delta=0,slot_schedule_cost_delta=20,valid_filled_slot_count_delta=1`。这说明 A41 暴露出的 fill-frontier 消费缺口确实存在，并且可以通过 agenda option 机制被补上。

同时，A42 的负证据同样明确：该 frontier completion 打开了新的 release/release-blocker 局面。step `31` 以后，已选动作持续被标为 `release_blocker,tier=2,active_agenda_progress=false,active_agenda_off_agenda_action=true`，且 `release_action_count=0,dependency_count=2,target_blocker_count=1,slot_schedule_cost=35` 长期不变。因此 A42 并未解决第 2 关。

**选项**
1. 将 A42 作为求解方向继续扩大，降低 fill-frontier 触发阈值或允许更多全局 fill 接管。
2. 因 A42 未提升目标数而回退到 A41，不再承认 fill-frontier milestone 的必要性。
3. 将 A42 记录为必要但不充分的机制推进：承认 post-release fill frontier 可以作为独立 agenda boundary 被消费，但下一步必须处理 frontier completion 之后的新 release-blocker option、agenda state 持久化和里程碑生成。

**结论**
采用选项 3。选项 1 会把 A42 扩展为未经论证的全局 fill 接管器，容易重新走向零散规则堆叠；选项 2 则忽略了 A42 已经使 frontier slot 完成并降低 slot cost。A42 的结果应解释为“当前 fill frontier boundary 的消费机制已被验证，但后续新 release-blocker 义务仍缺少可持续 option”。

**影响**
后续版本不应继续只围绕 fill 阈值或单步 slot cost 调参。若继续推进，应仍显式对齐专家 27 点：优先处理 frontier completion 之后的 release-blocker option commitment、active agenda state 持久化或更通用的 agenda-node/option-node/push-node 结构。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 120

**决策**
将 A43 实现为 A42 之后的 post-frontier release-blocker option 负诊断版本。A43 记录为有效排除项而非求解突破：在收紧后的 release-blocker true progress 口径下，80-push 复核与 A42 完全同轨迹，说明当前短程 option tree 未找到能打开 release 或减少 blocker 的 high-tier release-blocker option。

**背景**
A42 已证明 post-release fill frontier 可以作为独立 agenda boundary 被消费，但 frontier completion 后进入新的 `release_blocker` off-agenda 停滞：`release_action_count=0,dependency_count=2,target_blocker_count=1,slot_schedule_cost=35` 长期不变。根据专家 27 点中的 agenda-option tree 和 agenda state persistence 建议，A43 首先检验是否已有短程 release-blocker option 可被 root 选择消费。

A43 通过 OO 继承新增 `RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner`，继承 A42。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。触发条件被限制在 post-frontier release-blocker 语境：`stage=release`、`release_action_count=0`、仍有 active blocker、已有至少 `2` 个有效 filled slot，且父类候选为 release-blocker 低 tier 或 off-agenda。

实现中曾检查较宽松口径：只要 bounded option tree 内出现 `slot_schedule_cost_delta=1`，就可能被 A40 的原始 progress 逻辑标为 tier-5 release-blocker progress。该版本在 step `34` 发生 1 次 override，并将最终 slot cost 从 `35` 降到 `30`，但没有打开 release、没有降低 dependency、target blocker 或 locked blocker。该结果被判定为口径污染，而非有效 release-blocker 进展。

正式 A43 因此收紧 release-blocker true progress：只有目标推进、打开 release action、减少 active blocker、减少 dependency/target blocker、或减少 locked/goal-locked stand blocker，才可被视为 high-tier release-blocker option。诊断输出 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A43 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，与 A42 完全同轨迹，A42/A43 首个决策差异为 `None`。

关键负证据是：A43 的 release-blocker option 字段加载 `51` 次，其中 `31` 次为 `parent_not_release_blocker_trigger`，step `31` 至 step `50` 的 `20` 次触发均为 `no_high_tier_release_blocker_option`。这些触发点没有任何 `release_blocker_option_override`，`release_blocker_option_selected_progress_count=0`，`release_blocker_option_selected_release_open_count=0`。

**选项**
1. 接受宽松版本，将 slot cost 微降视为 release-blocker 进展，并继续沿该口径调参。
2. 因 A43 未改变轨迹而完全放弃 agenda-option tree 路线。
3. 将 A43 记录为负诊断：保留 release-blocker option evaluator 和审计字段，但承认当前短程预算内缺少真实 release-blocker 进展；下一步应转向更长程/分阶段里程碑、active blocker 分解或 agenda state 持久化。

**结论**
采用选项 3。选项 1 会把 release-blocker 义务重新混同为 slot cost 微调，违背 A42 暴露出的真实问题；选项 2 则忽略了 A40-A42 已验证 agenda-option 和 option commitment 的必要性。A43 的意义是排除“已有短程 release-blocker option 只是没有被 root 消费”这一假设。

**影响**
后续版本不应继续通过放宽 release-blocker progress、调 root score 或扩大同一短程 evaluator 的权重来追求偶然差异。若继续推进，应仍显式对齐专家 27 点：优先考虑把 active blocker 拆成可执行子义务、引入阶段里程碑状态、将 agenda/option 写入 MCTS 节点或闭环状态，或设计更长程但有边界的 option commitment。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 121

**决策**
将 A44 实现为 A43 之后的 active-blocker obligation 诊断版本。A44 记录为关系结构定位推进但非求解突破：它把 aggregate release-blocker count 拆为具体 A11 schedule dependency row，并在 `5` 个触发点发现 blocker-specific true progress suffix；但 80-push 复核仍与 A43 完全同轨，说明缺口已转移到 active-blocker option commitment 或 agenda state persistence。

**背景**
A43 的负诊断表明，当前 `5` push、`96` node 的 aggregate release-blocker option tree 内没有能打开 release 或减少 blocker 的 high-tier option。但 A43 的摘要仍只处理 aggregate blocker count，可能遮蔽了专家 27 点强调的 relation chain，即 `release -> blocker -> slot -> room`。因此 A44 不继续调 A43 分数，也不放宽 release-blocker progress，而是检验具体 blocker identity 是否已经能从现有 schedule dependency 中读出并形成可审计子义务。

A44 通过 OO 继承新增 `RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner`，继承 A43。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。它将 `stand_cell_occupied_by_box`、`stand_cell_unreachable_by_player`、`target_occupied_by_box` 三类 A11 dependency kind 作为 active blocker obligation 的 selected kinds，并对最多 `4` 个 obligation、最多 `12` 个 root candidate 运行 `6` push、`128` node 的 bounded evaluator。

诊断输出 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A44 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，与 A43 完全同轨迹，A43/A44 首个决策差异为 `None`。

关键证据是：A44 的 post-frontier 段触发 `20` 次，其中 `5` 次为 `parent_active_blocker_obligation_already_best_tier`，`15` 次为 `no_high_tier_active_blocker_obligation_option`。在这 `5` 个触发点中，A44 发现 high-tier blocker-specific true progress，例如 future dependency row `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6]` 的 suffix 能使 `remaining_selected_kind_delta=1` 且 `stand_resolved=true`。但父类当前 first edge 已达到同一 tier，A44 没有发生 override，`active_blocker_obligation_override_count=0`。

**选项**
1. 将 A44 改为 hard override，只要 top suffix 存在 true progress 就覆盖父类 root 选择。
2. 因 A44 未改变轨迹而判定 blocker-specific relation 无效，回到 aggregate release-blocker 口径。
3. 将 A44 记录为关系结构正诊断：承认具体 blocker 子义务和局部 suffix 已可见，但下一步必须处理 suffix commitment 或 agenda state persistence。

**结论**
采用选项 3。选项 1 会把 A44 从诊断改成未经验证的 suffix 执行器，而且 A44 证据显示父类 first edge 已经同 tier，不是 root 选择错误；选项 2 则忽略了 A44 已经发现的 blocker-specific true progress。A44 的结果应解释为“SD 端并非完全缺少 blocker identity，消费端缺少对同一 blocker obligation 的持续承诺”。

**影响**
后续版本不应继续通过放宽 A44 真进展口径、调整 root 分数或扩大同一 evaluator 权重来追求偶然差异。若继续推进，应仍显式对齐专家 27 点：优先把 active blocker option 作为可验证 suffix 承诺执行，或将 active obligation/option 写入闭环状态和 MCTS 节点，使系统在若干 push 内保持同一 blocker-specific agenda，直到 selected kind 消失、stand/target blocker 解除、遇到风险边界或出现新 frontier。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 122

**决策**
将 A45 实现为 A44 之后的 active-blocker option commitment 诊断版本。A45 记录为可执行性验证但非求解突破：它能在闭环中执行 A44 发现的 blocker-specific high-tier suffix，并产生真实子义务进展；但 80-push 复核仍停在 `2` goals，说明下一缺口不能再简单归因于“suffix 完全没有被持久执行”。

**背景**
A44 已排除“SD 端完全缺少 blocker identity”这一强假设，并在 `5` 个触发点发现了 high-tier blocker-specific true progress suffix。A44 的限制是只执行 root first edge，不承诺 suffix。根据专家 27 点中的第 6 点和第 14 点，下一步应检验同一 blocker obligation 的短程 option 是否可以作为闭环 commitment 被执行和维护。

A45 通过 OO 继承新增 `RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner`，继承 A44。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。触发条件为：A44 `selected_option` 存在、`active_blocker_obligation_tier>=8`、`active_blocker_true_progress=true`、suffix 长度不少于 `2` 个 push，且父层 corridor macro 默认没有先行生成多步宏。触发前会重新验证每个 suffix push 的合法性、deadlock child、path loop 和最终 active-blocker progress。

诊断输出 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A45 为 `max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A44/A45 首个决策差异为 step `32`。commitment 加载 `47` 次，触发 `2` 次，首个触发为 step `31`，额外执行 `4` 个 suffix push，验证 push 总数为 `6`。

关键正证据是 step `31` 与 step `41`。A45 在 step `31` 将 A44 的 active-blocker suffix 作为四步承诺执行，最终 progress 为 `active_blocker_obligation_tier=8,active_blocker_true_progress=true,active_blocker_progress_kinds=[selected_kind_reduced,stand_resolved]`；step `41` 再次执行两步承诺并达到同类 true progress。这证明 A44 的 suffix 不是离线评价幻觉，可以被当前 exact transition 闭环合法消费。

同时，A45 的负证据也明确：commitment 被执行以后，系统仍未推进到第三个目标，最终 dependency 与 target blocker 未消除，release action 仍为 `0`，且 final slot cost 相比 A44 还从 `35` 变为 `36`。因此，当前失败不能仅靠继续执行同类 suffix 解决。

**选项**
1. 将 A45 继续扩大为更长、更频繁的 active-blocker suffix 执行器。
2. 因 A45 未提升目标数而否定 A44/A45 的 blocker-specific obligation 路线。
3. 将 A45 记录为必要但不充分的机制验证：承认 blocker-specific suffix 可以闭环执行，但下一步必须处理 commitment 后的 active obligation 更新、阶段/里程碑重建和更高层 agenda-node/option-node/push-node 结构。

**结论**
采用选项 3。选项 1 会把 A45 扩展为未经论证的局部后缀执行器，容易重新走向规则堆叠；选项 2 则忽略了 A45 已经把 A44 的局部 true progress 在闭环中实际执行出来。A45 的结果应解释为“active-blocker commitment 是可执行机制，但不是当前第二关求解的充分条件”。

**影响**
后续版本不应继续通过降低 A45 触发阈值、延长同类 suffix、放宽 true-progress 口径或增加局部 blocker 规则来追求偶然收益。若继续推进，应仍显式对齐专家 27 点：优先把 active obligation/option 写入可持久更新的 agenda state，或推进更通用的 agenda-node/option-node/push-node 结构，使系统在承诺后能重建下一阶段义务，而不是只重复执行局部 suffix。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 123

**决策**
将 A46 实现为 A45 之后的 residual active-agenda 负诊断版本。A46 记录为残余义务重建有效但不足以求解：它能在 A45 commitment 后把同一 blocker relation 的 `target_occupied_by_box` 重建为 target-only residual agenda，但在 `8` push、`192` node 的 residual option 空间内未找到 target progress，因此 80-push 复核仍与 A45 完全同轨。

**背景**
A45 已证明 blocker-specific suffix 可以被闭环合法执行，但两次 commitment 都只完成 `stand_resolved`，并留下 `target_occupied_by_box`。因此，A45 后的下一假设是：失败可能来自 commitment 后没有显式携带残余 agenda，导致系统转回其他局部义务或低 tier 动作。该假设直接对应专家 27 点中的第 6 点 active agenda state、第 11.2 点 distance to next agenda milestone 和第 14 点 agenda-option tree。

A46 通过 OO 继承新增 `RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner`，继承 A45。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。A46 的触发条件被限制为：A45 commitment 已触发，最终 `unresolved_target=true`，且最后一个 commitment row 的 `remaining_selected_kinds` 仍包含 `target_occupied_by_box`。此时 A46 保持原 dependency relation 不变，将 `selected_dependency_kinds` 收缩为 `["target_occupied_by_box"]`，并运行一次 residual option 搜索。只有 dependency row 消失、target blocker 解除或 selected blocker cell 离开，才允许追加 residual push；否则动作保持 A45 原样。

诊断输出 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A46 为 `max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，与 A45 完全同轨迹，A45/A46 首个决策差异为 `None`。

关键证据是：A46 在 step `31` 和 step `41` 都正确重建 target-only residual agenda，残余 relation 为 `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6],candidate_direction=down`，残余 kind 为 `target_occupied_by_box`。两次 residual search 均达到 `expanded_nodes=192` 上限，合计评估 `384` 个 residual option，但没有发现 target progress，`residual_active_agenda_triggered_count=0`、`residual_active_agenda_target_progress_count=0`，原因分布为 `{'source_commitment_not_triggered': 45, 'no_target_progress_residual_option': 2}`。

**选项**
1. 继续扩大 A46 residual 搜索预算，尝试用更深 BFS 找到 target blocker 解除路径。
2. 因 A46 未改变轨迹而否定 active agenda / residual obligation 路线。
3. 将 A46 记录为负诊断：承认残余义务显式重建是必要审计环节，但当前 target blocker 的下一里程碑不在该有限 residual option 空间内；下一步应转向更一般的 agenda-node/option-node/push-node 拓扑、中间里程碑定义或 agenda distance 函数近似。

**结论**
采用选项 3。选项 1 已被临时预算诊断的运行代价和 A46 的 `192` node 上限负结果削弱，继续盲目加深会违背专家意见中“当前失败不是搜索预算单纯不足”的判断；选项 2 则忽略了 A46 已经把残余 target obligation 从 A45 的部分 commitment 中明确恢复出来。A46 的意义是排除“只要承诺后显式携带 target 残余义务即可推进”的解释。

**影响**
后续版本不应继续通过降低 A46 触发阈值、把非 target 变化算作 residual progress、或继续堆叠局部 blocker 规则来追求偶然收益。若继续推进，应显式对齐专家 27 点：把 MCTS 拓扑从 push tree 推向 agenda-node/option-node/push-node，定义 target residual 的可执行中间里程碑，或训练/使用可复用的 agenda-distance/value 表达。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 124

**决策**
将 A47 实现为 A46 之后的 target-prerequisite agenda 负诊断版本。A47 记录为前提义务显式化但非求解突破：它能把 target residual blocker 的直接 push 前提分解为 `ClearTargetPushStand`，并识别 required stand 被 parent assigned box 占用；但在 `8` push、`256` node 的 prerequisite option 空间内未找到真实前提进展，因此 80-push 复核仍与 A46 完全同轨。

**背景**
A46 已排除“承诺后没有显式重建 target residual agenda”这一单一解释，但它仍只把 residual target blocker 当作直接待清除对象。后续审计显示，target blocker `[3,6]` 自身没有合法 push：向下推它需要 stand `[2,6]`，而 `[2,6]` 被 parent assigned box 占用；向上会推入 `[2,6]` 的箱子；左右方向不是地板。因此下一假设是：失败可能来自 target residual 没有被继续分解为可执行前提义务和 blocker set。该假设对应专家 27 点中的未完成义务集合、frontier obligations、每个义务的 blocker set、第 6 点 active agenda state 和第 14 点 agenda-option tree。

A47 通过 OO 继承新增 `RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner`，继承 A46。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。触发条件被限制为：A46 的 residual agenda 未找到 target progress，且 target blocker 没有合法 push；若存在一个可清空 parent target cell 的 blocker push，其 push target 为空地板但 required stand 被箱子占用，则生成 `ClearTargetPushStand` 前提义务。只有 target blocker 已移动、target blocker push 已合法，或 required stand 已清空且可达，才允许追加 prerequisite suffix。

诊断输出 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A47 为 `max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，与 A46 完全同轨迹，A46/A47 首个决策差异为 `None`。

关键证据是：A47 在 step `31` 和 step `41` 都构造出 `ClearTargetPushStand` 前提义务。两次义务均指向 parent residual relation `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6],candidate_direction=down`，并记录 `required_stand_cell=[2,6]`、`required_stand_blocker_is_parent_assigned_box=true`、`target_blocker_legal_push_count=0`。两次 prerequisite search 均达到 `expanded_nodes=256` 上限，合计评估 `512` 个 option，但 `target_prerequisite_agenda_triggered_count=0`、`target_prerequisite_agenda_committed_push_count=0`，原因分布为 `{'source_not_target_residual_failure': 45, 'no_target_prerequisite_progress_option': 2}`。

**选项**
1. 将 A47 改为坐标级或 relation 特例规则，强制优先移动 `[2,6]` 上的箱子。
2. 因 A47 未改变轨迹而否定 residual/prerequisite agenda 路线。
3. 将 A47 记录为负诊断：承认前提义务和 blocker set 已可显式读出，但当前短程 prerequisite option 无法完成；下一步应转向更高层 agenda-node/option-node/push-node 拓扑、阶段里程碑或可复用 agenda-distance/value 表达。

**结论**
采用选项 3。选项 1 会把当前发现退化为局部坐标规则，违背简洁、可泛化的模型要求；选项 2 则忽略了 A47 已经把 target residual 的下一层前提义务结构化读出。A47 的意义是排除“只要把 target residual 分解成前提 blocker set 即可推进”的解释。

**影响**
后续版本不应把 `[2,6]`、`[3,6]` 等具体坐标写入规则，也不应继续单纯扩大 A47 的 BFS 预算来追求偶然差异。若继续推进，应仍显式对齐专家 27 点：将 active agenda/option 写入节点或闭环状态，形成 agenda-node/option-node/push-node 拓扑，或引入可复用的 agenda-distance/value 函数，使系统能学习和复用“清空前提站位、释放 target blocker、再推进 slot”的阶段性经验。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

## 决策 125

**决策**
将 A48 实现为 A47 之后的 assignment-conflict agenda 负诊断版本。A48 记录为 assignment/prerequisite 冲突显式化但非求解突破：它能把 `ClearTargetPushStand` 的 required stand blocker 识别为 parent assigned box，并读出 assigned box 的移动性与到 slot 的 pull-distance；但在 `8` push、`320` node 的 conflict option 空间内未找到冲突解除，因此 80-push 复核仍与 A47 完全同轨。

**背景**
A47 已排除“target residual 没有被分解成前提 blocker set”这一单一解释，但它进一步显示 required stand `[2,6]` 的占用者正是 parent residual 中分配给 slot `[2,1]` 的 assigned box。也就是说，当前不是普通站位阻塞，而是当前 slot assignment 与 target blocker release prerequisite 发生冲突。该假设对应专家 27 点中的未完成义务集合、每个义务的 blocker set、第 6 点 active agenda state 和第 14 点 agenda-option tree。

A48 通过 OO 继承新增 `RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner`，继承 A47。该版本复用 A35 multi-head checkpoint，不训练新模型，不新增 teacher/reference 在线锚定，不改变 exact dynamics，也不修改 Boxoban 989/1000 主线。触发条件被限制为：A47 的 target prerequisite agenda 未找到 progress，且 `required_stand_blocker_is_parent_assigned_box=true`。此时 A48 生成 `ResolveAssignedBoxPrerequisiteConflict` 义务，记录 assigned box cell、assigned slot cell、required stand cell、target blocker cell、assigned box 到 slot 的 pull-distance、assigned box 四向 push candidates、合法 push 数和朝 slot 合法 push 数。只有 required stand 清空且可达、target blocker push 合法、或 assigned box 离开 required stand 并解除占用，才允许追加 suffix。

诊断输出 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json) 显示，同一 `budget=128,horizon=8,smoke_steps=80` 下，A48 为 `max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，与 A47 完全同轨迹，A47/A48 首个决策差异为 `None`。

关键证据是：A48 在 step `31` 和 step `41` 都构造出 assignment-conflict obligation。两次义务均记录 `assigned_box_cell=[2,6]`、`assigned_slot_cell=[2,1]`、`required_stand_cell=[2,6]`、`target_blocker_cell=[3,6]`、`assigned_box_pull_distance=9.0`、`assigned_box_legal_push_count=0`、`assigned_box_toward_slot_push_count=0`。两次 conflict search 均达到 `expanded_nodes=320` 上限，合计评估 `640` 个 option，但最佳 progress 仍为 tier `2`：`assignment_conflict_done=false`、`assigned_box_mobility_opened=false`、`assigned_box_toward_slot_opened=false`、`assigned_box_pull_distance_delta=0.0`。原因分布为 `{'source_not_target_prerequisite_failure': 45, 'no_assignment_conflict_done_option': 2}`。

**选项**
1. 将 A48 扩展为专门移动 `[2,6]` assigned box 或其相邻箱子的局部规则。
2. 因 A48 未改变轨迹而否定 assignment/prerequisite conflict 结构。
3. 将 A48 记录为负诊断：承认 conflict relation、assigned box mobility 和 pull-distance 已可显式读出，但当前短程局部 option 无法解除；下一步应转向更高层 agenda-node/option-node/push-node 拓扑、阶段里程碑或可复用 agenda-distance/value 表达。

**结论**
采用选项 3。选项 1 会把当前结构发现退化为局部坐标和相邻格规则，违背简洁、可泛化的模型要求；选项 2 则忽略了 A48 已经把 A47 的前提阻塞提升为 assignment/prerequisite conflict。A48 的意义是排除“只要读出 assignment conflict 即可通过短程 option 推进”的解释。

**影响**
后续版本不应继续围绕 `[2,6]`、相邻箱子或同类短程 BFS 写规则，也不应简单扩大 A48 搜索预算。若继续推进，应仍显式对齐专家 27 点：把 active agenda/option 写入节点或闭环状态，形成可跨阶段持久的 agenda-node/option-node/push-node 拓扑，或引入可复用 agenda-distance/value，使模型能够表达和复用“释放 assigned box 移动性、清空 prerequisite stand、释放 target blocker、再推进 slot”的多阶段机制。所有后续实验继续保持 OO 继承、Boxoban 主线隔离、`teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。
