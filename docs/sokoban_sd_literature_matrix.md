# Sokoban SD Literature Matrix

## 使用原则

本矩阵用于把 Sokoban/Boxoban 文献中的可核验机制转写为 stock-flow-feedback SD 世界模型设计约束。文献机制只提供结构先验、观测代理、流量门控、风险标签或诊断项，不直接构成最终启发函数。证据等级分为三类：

1. `A`：本地 PDF 抽取文本可读，机制可直接从正文核验；
2. `B`：题录、摘要或参考文献链可核验，但本地正文抽取不完整或不可读；
3. `C`：检索结果误匹配或仅可作为背景，不进入设计依据。

`results/sokoban_sd_world_model/literature_scan.json` 中的 arXiv 搜索结果包含明显误匹配，例如与 Sokoban 无关的新近预印本，因此不作为设计依据。当前依据优先来自本地 PDF 与可读抽取文本。

## 证据矩阵

| 文献 | 证据等级 | 可核验机制 | 对 SD 设计的约束 |
| --- | --- | --- | --- |
| Junghanns and Schaeffer, 1998, `Sokoban: Evaluating Standard Single-Agent Search Techniques in the Presence of Deadlock` | A | Sokoban 存在不可逆死锁；push-level 表示隐含玩家移动；最小二分匹配可作为下界；deadlock tables、macro moves 和 move ordering 对搜索关键。 | 支持 `IrreversibilityRiskStock`、`MobilityStock`、`ReachableCapacityStock` 和 push-level option 结构。匹配下界和 deadlock evidence 作为风险流量输入或标签，不作为最终启发函数。 |
| Junghanns and Schaeffer, 2001, `Sokoban: Enhancing General Single-Agent Search Methods Using Domain Knowledge` | A | Rolling Stone 组合使用 transposition、matching lower bound、deadlock tables、tunnel macros、goal macros、goal cuts、relevance cuts 和 pattern searches；多项增强可带来数量级搜索树缩减。 | 支持把 goal-room、tunnel、relevance 和 pattern 机制作为结构门控与诊断代理。当前第一版只覆盖少量代理，不能声称实现完整 SD 世界模型。 |
| Junghanns and Schaeffer, 2001, `Sokoban: Relevance Cuts` | A | Relevance cuts 用于限制与当前相关性弱的推箱序列，体现扩展后果不是只由局部合法性决定。 | 支持把相关性作为 single-push option 的流量门控、prior 输入和 plan-progress probe；首版不把 relevance 写成硬剪枝或核心存量。 |
| Botea, Muller and Schaeffer, 2002, `Using Abstraction for Planning in Sokoban` | B | 本地 PDF 题录可核验，但正文抽取乱码；Guez 等后续文献将其列为 Sokoban planning/abstraction 相关工作。 | 可作为“抽象规划是既有路线”的背景，不直接用其乱码正文支持具体机制。 |
| Pereira, Ritt and Buriol, 2013, `Finding Optimal Solutions to Sokoban Using Instance Dependent Pattern Databases` | A | Sokoban 中箱子到目标映射事先不固定；实例分解和 PDB 可捕获箱子、玩家位置交互；PDB 能识别部分多箱死锁。 | 支持把多箱 pattern、实例依赖目标分配和局部死锁证据纳入 `UnresolvedBoxStock`、`IrreversibilityRiskStock`、风险流量和 structural slack 诊断。 |
| Pereira et al., 2016, `Improved Heuristic and Tie-Breaking for Optimally Solving Sokoban` | A | EMM 是标准启发式；死格、cut squares、中间目标状态、多个中间目标和 tie-breaking 会影响最优搜索。 | 支持目标区填充顺序、goal-room pressure 派生特征和风险标签。EMM 和 tie-breaking 只作为诊断或辅助标签。 |
| Lipovetzky, Muise and Geffner, 2016, `Traps, Invariants, and Dead-Ends` | A | traps 是条件不变量；与目标 mutex 的 trap 可表示 dead-end；预处理 trap 可用于 dead-end pruning。 | 支持 `risk_accumulation_flow`、`risk_relief_flow` 和 `IrreversibilityRiskStock`，并要求风险标签区分可证明 dead-end 与启发式风险。 |
| Racaniere et al., 2017, `Imagination-Augmented Agents` | A | 通过模型想象轨迹可提高数据效率与泛化，但近似模型误差会影响规划。 | 支持在学习到的抽象世界模型中 rollout，并要求报告模型误差和规划误差。 |
| Guez et al., 2019, `An Investigation of Model-Free Planning` | A | Boxoban/Sokoban 可检验组合泛化、数据效率和额外 thinking time；DRC 在 Sokoban 上表现出 planning-like 行为。 | 支持把 Boxoban 作为规划表征基准；评测应包含泛化、预算扩展和额外计算时间维度，而不只报告固定 50 关卡 solved rate。 |
| Orseau et al., 2018, `Single-Agent Policy Tree Search With Guarantees` | A | policy 可指导单智能体树搜索，并给出搜索保证。 | 支持把 SD 世界模型输出接入 tree search prior，但 prior 应来自学习模型或其头部，不等同于静态启发式标量。 |
| Orseau and Lelis, 2021, `Policy-Guided Heuristic Search with Guarantees` | A | policy 与 heuristic 可联合指导搜索；Sokoban/Boxoban 实验中二者质量共同决定搜索效率。 | 支持在消融中分离 policy prior、value、risk 和 heuristic diagnostic，防止混淆 SD-MCTS 与启发式搜索。 |
| Mirza et al., 2020, `Physically Embedded Planning Problems` | A | MuJoBan 将 Sokoban 嵌入物理控制，显示抽象规划与低层执行解耦会造成额外难度。 | 支持显式记录抽象 option 与低层执行 adapter，避免高层 SD 状态与真实执行脱节。 |
| Martin and Sandholm, 2023, `AI planning in the imagination` | A | 可在学习到的抽象搜索空间上进行高层 MCTS，动作可代表 compound behavior。 | 直接支持 `SD-MCTS over learned abstract SD states` 的接口。 |
| Taufeeque et al., 2024, `Planning in a recurrent neural network that plays Sokoban` | A | DRC 中存在可预测未来动作的内部计划表示，且网络会通过 pacing 获得额外计算。 | 支持 thinking-time evaluation、计划通道 probe 和第二阶段 plan-progress 辅助头；首版不把该内部计划表征混入环境核心存量。 |
| Taufeeque et al., 2025, `Path Channels and Plan Extension Kernels` | A | DRC path channels 存储未来移动；扩展 kernel 体现双向计划扩展、障碍抑制和回溯；路径通道干预显著影响求解率。 | 支持把 path/goal/box 通道作为 plan-progress probe、路线一致性辅助标签和后续反馈扩展。 |

## 机制到 stock-flow-feedback 的映射

| 机制 | 当前草稿状态 | SD 设计用法 |
| --- | --- | --- |
| 玩家可达域 | 已实现 flood fill | 作为 `ReachableCapacityStock` 和 `MobilityStock` 的观测代理，并作为 option 可执行性的结构门控。 |
| push-level 动作 | 已实现合法推箱动作 | 作为首版合法 single-push option 集合；tunnel/goal macro 只在审计后扩展。 |
| reverse-pull 距离 | 已实现静态墙图上的单箱距离 | 作为目标压力、路线一致性和风险流量输入，不直接定义价值函数。 |
| 最小匹配下界 | 已实现二分匹配 DP | 作为 `UnresolvedBoxStock` 的观测代理、流量输入或辅助标签，不直接定义价值函数。 |
| dead-square | 已实现不可达目标格剪枝 | 作为 `risk_accumulation_flow` 的高置信结构证据；硬剪枝只限诊断草稿。 |
| 静态墙角死锁 | 已实现保守判据 | 作为不可逆风险标签和过剪枝审计对象。 |
| 多箱死锁 | 未实现 | 作为局部 pattern 或 PDB 风险标签，进入 `IrreversibilityRiskStock`。 |
| tunnel/goal macro | 未实现 | 作为第二阶段 option 候选、goal-room pressure 派生特征和持续时间预测输入。 |
| relevance cuts | 未实现 | 作为流量门控、prior 输入、plan-progress probe 或 structural slack 诊断，不直接硬剪枝。 |
| policy + heuristic 分离 | 未实现 | 在正式 SD-MCTS 中分离 learned prior、value、risk 和 heuristic diagnostic。 |
| plan/path channels | 未实现 | 作为 plan-progress probe、辅助监督或第二阶段扩展。 |

## 下一步验收标准

1. 文献矩阵中的每个机制均应对应一个存量代理、流量输入、反馈输入、风险标签、诊断项或明确的“不实现原因”；
2. 正式模型必须输出存量、流量、反馈、不确定性和下一抽象状态；
3. 任一硬剪枝机制必须有过剪枝审计，至少报告被剪枝状态中可解反例的抽查结果；
4. 任一规划结果必须同时报告世界模型误差、流量守恒误差、风险校准、solved rate、预算耗尽数和失败类型；
5. 进入神经训练前，应先完成短 rollout 或真实交互标签构建，避免只用启发式代理训练世界模型。
