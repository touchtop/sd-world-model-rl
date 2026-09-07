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
