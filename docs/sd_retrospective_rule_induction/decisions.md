# Decisions

## 决策 17

**决策**  
TVA 先作为规则价值的 shadow update 信号接入，不直接改变在线规则库或 planner prior。

**背景**  
TVA 离线验证显示，其 episode-level 价值信号与真实回报高度相关，且在 good-label 记录中的实际动作 top-3 比例高于 bad-label 记录。但短视界反事实排序仍受动作探测集合、SD rollout 误差和局部外推影响，尚不足以直接替代现有 confidence 更新。

**选项**  
1. 将 TVA 评分直接并入在线规则置信度更新，并影响下一周期动作先验。  
2. 只生成 TVA shadow update 报告，与现有 confidence 更新和后续规则效应审计对照。  

**结论**  
采用选项 2。

**影响**  
后续周期可用 `--enable-tva-shadow-update` 生成 TVA 建议的增强、削弱和退休结果，但训练仍沿用原规则库。只有当 TVA shadow 建议在多周期、多种子上与实际规则效应审计一致后，才考虑进入正式 confidence 更新或在线 prior。

## 决策 15

**决策**  
第八版规则提取不转换原方向，而是在现有失败恢复规则基础上新增“成功延续”和“阶段衔接”规则通道。

**背景**  
`v72` 三种子共享规则闭环训练至 `200k` 后，最终规则库中已有 `34` 条可参与在线先验的 `replay_validated / rolling_validated` 规则，训练日志显示规则先验确实启用。但最终 eval 轨迹审查显示，规则命中主要集中于 `rear`、`falling`、`vx_low`、`rotation_risk` 或 `unsupported_high_torque` 等恢复状态；若干较理想姿态片段，例如 `omega_mid / vx_high / support_none / falling / rotation_risk=0`，没有被当前规则库覆盖。

**选项**  
1. 转换研究方向，放弃失败恢复规则，改为只提取成功姿态规则。  
2. 保留失败恢复规则，在原框架中新增成功延续与阶段衔接规则。  

**结论**  
采用选项 2。

**影响**  
后续规则库需要记录 `rule_role`，区分 `recovery`、`continue` 和 `transition`。失败恢复规则仍用于坏状态风险筛选；成功延续规则用于保持局部好姿态；阶段衔接规则用于解释飞行、落地、支撑和再起跳之间的中断。在线接入仍保持轻量 MPPI 候选排序，不执行硬动作替换。

## 决策 16

**决策**  
第八版规则提取采用“通用引擎 + Hopper 适配器”设计边界。

**背景**  
当前实现为了推动 Hopper 实验，显式使用了 `rear/prone/falling/upright`、`pitch/vx/support`、`waist/hip/knee/ankle` 等 Hopper 专用语义。该做法有利于第一阶段可解释、可调试，但若这些谓词直接嵌入规则闭环主流程，将削弱未来迁移到 Walker、Cartpole 或其他结构化控制任务的可能性。

**选项**  
1. 继续在主规则流程中直接硬编码 Hopper 谓词和动作族。  
2. 将任务无关流程抽象为通用引擎，将 Hopper 谓词、阶段和动作族封装为适配器。  

**结论**  
采用选项 2。

**影响**  
后续代码实现应优先拆分 `StateSchema`、`PhaseDetector`、`QualityMetrics`、`ActionFamilyMapper`、`SegmentMiner`、`CounterfactualEvaluator`、`RuleInducer` 和 `RuleValidator`。Hopper v8 可以继续显式使用当前结构态字段，但这些字段应通过适配器进入通用流程，而不是散落在规则归纳、验证和更新主逻辑中。

## 决策 1

**决策**  
将“基于 SD 的反向归因与规则归纳”作为独立研究线，而不是直接并入当前 Hopper TD-MPC2 在线训练主线。

**背景**  
当前 mechanism-oriented `SD` 主线已经证明结构态前向推演具有样本效率价值，但新设想涉及失败归因、反事实分析和规则干预。若在第一阶段直接接入在线训练，将难以区分规则质量、归因可靠性和 TD-MPC2 训练波动之间的影响。

**选项**  
1. 直接将规则归纳模块嵌入 TD-MPC2 在线训练。  
2. 先建立离线独立研究线，验证归因和规则可靠性后再考虑软干预接入。  

**结论**  
采用选项 2。

**影响**  
第一阶段只读取既有 Hopper 轨迹和评估结果，不修改 TD-MPC2 主训练逻辑。规则树先作为诊断与候选知识产物保存，后续再评估是否作为动作先验、风险惩罚或 replay 加权机制接入训练。

## 决策 2

**决策**  
新研究线的代码与结果采用独立目录，并在第一阶段禁止修改 TD-MPC2 主训练代码和既有 Hopper 运行脚本。

**背景**  
该方向需要读取当前 Hopper 主线产物，但研究对象已经从在线前向规划扩展到离线反向归因与规则归纳。若代码混入 `external/tdmpc2/` 或复用现有 `run_hopper_v5_*` 脚本，会使后续实验难以判断结果来自原主线方法、归因模块，还是训练入口变化。

**选项**  
1. 在现有 TD-MPC2 目录中直接加入归因与规则代码。  
2. 在 `experiments/sd_retrospective_rule_induction/` 与 `results/sd_retrospective_rule_induction/` 下建立独立离线工具链。  

**结论**  
采用选项 2。

**影响**  
第一阶段新增脚本只读已有 Hopper 日志，输出写入独立结果目录。后续若规则需要接入在线训练，应新建 `run_hopper_v6_*` 脚本和显式默认关闭的规则开关，而不是改写已有主线脚本。

## 决策 3

**决策**  
规则归纳采用高门槛原则，允许每轮只生成少量规则，甚至不生成新规则。

**背景**  
本研究线的目标不是最大化规则数量，而是模拟经验总结过程，将可重复、可解释、可迁移的状态-动作原则沉淀为规则。当前 v3 审查显示，`8` 条 v2 候选中仅 `1` 条通过可信候选门槛，其余规则主要存在好轨迹误伤率偏高或同事件动作偏好冲突的问题。若强行保留这些规则，后续在线接入时可能固化局部策略偏差。

**选项**  
1. 每轮都强制生成固定数量的新规则。  
2. 允许空产出，将每轮结果分为新增、强化、修订、否决或保持观测。  

**结论**  
采用选项 2。

**影响**  
后续规则库增长应是稀疏、保守和可审计的。没有新增规则并不表示该轮无效；若该轮削弱了错误候选、提示需要细分状态条件，或强化了已有规则，也应视为有效经验总结。

## 决策 4

**决策**  
第五版规则库只允许 `replay_validated` 子状态规则进入后续在线软先验候选池，粗规则和未通过回放的候选规则暂不接入在线训练。

**背景**  
v4 离线回放显示，v3 粗规则 `unsupported_high_torque -> prefer hip_neg` 虽然通过了关键状态反事实审查，但在逐帧 trace 中命中大量好轨迹片段。v5 子状态回放进一步显示，只有 `unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos` 同时具备较低好轨迹命中率、较高差轨迹命中率和跨 seed、跨周期覆盖。

**选项**  
1. 将所有已有规则作为在线先验接入。  
2. 仅将 `replay_validated` 子状态规则纳入在线软先验候选池，其余规则继续离线观察或拆分。  

**结论**  
采用选项 2。

**影响**  
后续若开展在线 smoke，应新建独立 v6 入口，并保持规则开关默认关闭。在线干预应先采用低强度动作候选重加权，不进行硬动作替换。`needs_split` 与 `candidate` 规则只用于诊断和离线更新，不参与训练决策。

## 决策 5

**决策**  
第六版在线 smoke 采用独立训练入口和默认关闭的运行时规则先验补丁，不修改 TD-MPC2 主线代码。

**背景**  
当前规则库只有一条子状态规则通过回放审查，尚不足以支持大规模在线接入。若直接修改 `external/tdmpc2/` 或既有 Hopper v5 脚本，将难以区分规则先验、训练入口变化和原有 SD 主线之间的影响。

**选项**  
1. 直接在 TD-MPC2 主线 planner 中加入规则逻辑。  
2. 新建 v6 训练入口，在 agent 构造后按显式开关安装规则先验补丁。  

**结论**  
采用选项 2。

**影响**  
规则关闭时，v6 入口可作为同入口对照；规则开启时，只对 MPPI 候选价值排序施加低强度偏置。该实现不会替换动作、不会改变环境奖励、不会改变 replay 写入和优化器状态。后续评估应采用成对 smoke，而不是只看单次开启规则后的训练曲线。

## 决策 6

**决策**  
基于第六版 quick smoke，暂不增强当前 `hip_pos` 在线规则先验，也不进入更长步数的同形态规则开启训练。

**背景**  
quick smoke 显示，规则开启后确实改变了目标子状态下的动作分布，但最终评估没有获得回报改善，并且轨迹诊断指标变差。规则开启线在目标子状态下全部选择 `hip_pos`，同时出现更低的 forward displacement、更低的 full-foot support、更高的 backward-speed fraction 和更高的 rotation-penalty fraction。

**选项**  
1. 维持当前规则先验形式，直接扩大步数或提高权重。  
2. 暂停增强当前规则，先降低干预强度或增加门控条件后再验证。  

**结论**  
采用选项 2。

**影响**  
当前规则库中的 `replay_validated` 状态仍表示其通过离线回放审查，但不再等价于“可直接在线推高动作族”。后续应把在线状态单独标注为需要进一步门控或降权，避免把离线失败定位能力误读为在线控制收益。

## 决策 7

**决策**  
将下一轮在线规则接入从 `prefer` 模式改为 `penalty_only` 模式。

**背景**  
第六版 quick smoke 中，`prefer` 模式使目标子状态下的动作全部转向饱和 `hip_pos`，但最终评估轨迹更差。第六点一版离线审查显示，penalty-only 会惩罚失败 trace 中全部饱和 `hip_pos` 命中点，同时也会惩罚规则关闭 trace 中相反或过饱和的高风险动作。

**选项**  
1. 保持 `prefer` 模式，只降低权重。  
2. 改为 `penalty_only` 模式，不再正向奖励 `hip_pos`，只惩罚相反或过饱和动作。  

**结论**  
采用选项 2。

**影响**  
规则在线语义从“推荐动作族”降级为“风险筛选器”。这保留了离线失败归因得到的状态定位价值，同时降低了把局部反事实动作族误用为在线控制指令的风险。

## 决策 8

**决策**  
不以 3k quick smoke 的终点评估回报否决 penalty-only 规则；先进行 20k 阶段性训练对比，再决定是否增加 support-recovery gate 或扩大训练。

**背景**  
penalty-only quick smoke 消除了 v6 prefer 模式中的饱和 `hip_pos` 失败形态，并降低了 backward-speed 与 rotation-penalty 指标，但最终回报仍为 `0.0`，且未恢复 full-foot support。由于 Hopper 任务奖励稀疏、完成过程阶段化依赖强，3k 终点评估只能作为局部安全性与行为信号检查，不能直接作为长程训练效果结论。

**选项**  
1. 直接扩大 penalty-only 到长训练，不做阶段性中间诊断。  
2. 先做 20k 成对 pilot，综合回报与阶段诊断指标，再决定是否加门控或扩展步数。  

**结论**  
采用选项 2。

**影响**  
下一轮先比较 `rule off` 与 `penalty_only` 在 20k 内的阶段性表现。若 penalty-only 在阶段指标上不退化或出现局部改善，再考虑加入 full-foot support 恢复趋势、低 backward-speed、低 rotation-penalty 或接触可行性等门控条件；若出现稳定退化，则回到离线规则拆分。

## 决策 9

**决策**  
20k pilot 的后续推进不直接扩大当前 penalty-only 规则，而是先增加与支撑恢复或阶段进展相关的门控条件，并继续保持纯 TD-MPC2 学习主体。

**背景**  
修正后的 20k pilot 已确认不再训练 SD，运行参数为 `world_model_kind=tdmpc2`、`sd_online_supervision_enabled=false`、`sd_supervised_init_from=null`。结果显示，penalty-only 在 10k 和 15k 出现中间正信号，评估回报分别达到 `0.148496` 和 `0.644448`，且 backward-speed 与 rotation-penalty 指标低于 rule-off；但 20k 末端该信号未保持，且 full-foot support 恢复不足。

该结果需要放在规则库规模中解释。当前在线规则先验只接入 `1` 条通过回放验证的早期规则，而 Hopper 任务具有明显的阶段化依赖。单条早期规则在小预算下不能覆盖起身、支撑恢复、前向推进和稳定落足等完整过程，因此不应以“未在 20k 内彻底解决 Hopper”作为否定标准。

**选项**  
1. 直接将当前 penalty-only 配置扩大到 50k 或更多 seed。  
2. 保持纯 TD-MPC2 主体，先为规则先验加入支撑恢复、阶段进展或接触可行性门控，再进入更长训练。  

**结论**  
采用选项 2。

**影响**  
当前结果被视为“单条早期规则存在阶段性信号但尚不稳定”，而不是稳定收益或明确失败。下一版规则先验不应重新开启在线 SD 监督；若需要使用 SD，应作为离线归因或显式分离的实验变量单独评估。

## 决策 10

**决策**  
在教师核心规则初始化分支中，新增“能力入口追踪”作为离线审计组件，优先分析高质量能力如何进入 replay 分布，而不是只从最终高质量轨迹提取维持规则。

**背景**  
`tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1` 显示 seed1 很早进入高质量 hop 轨迹，而 seed2、seed3 明显滞后。仅从最终高分轨迹提取核心规则，可以解释标准姿态下应如何维持动作结构，但不足以解释 agent 如何从早期失败吸引子跨入可持续 hop 的入口状态。若目标是形成可泛化的方法论，需要追踪能力形成过程，并从跨阶段入口中提炼一致规则。

**选项**  
1. 继续只分析最终高质量轨迹，将规则设计集中在标准 hop 姿态维持上。  
2. 新增离线能力入口追踪，自动识别首次跨越阈值的训练周期，并以同期未跨入口种子作为对照，寻找阶段入口候选动作和状态条件。  

**结论**  
采用选项 2。

**影响**  
第一版入口追踪只读 retrospective train traces 和既有规则审计报告，输出写入 `results/sd_retrospective_rule_induction/ability_entry/`，不修改在线规则库或训练流程。该决策将后续规则生成重点从“最终姿态规则”扩展为“阶段入口规则 + 最终维持规则”的组合，但入口候选在上线前仍需经过 SD/TVA 反事实推演、跨周期后果验证和 shadow 审计。

## 决策 11

**决策**  
阶段入口规则的下一版表达优先采用“效果空间规则”，暂不恢复动作序列式或阶段成果式在线规则。

**背景**  
能力入口追踪显示，seed1 早期成功入口中存在若干动作族差异，但 SD/TVA 反事实审计表明这些单动作族并不稳定地成为最佳候选。此前阶段成果式规则虽然可以表达动作过程，但在线触发、终止、冲突维护和信心削弱均更复杂，且效果不明确。为保持现有规则库的简洁性，需要把复杂阶段过程投影到更适合维护的数学空间中。

**选项**  
1. 恢复动作序列式或阶段成果式规则，用显式状态机表达阶段过程。  
2. 保持规则对象简洁，将规则从动作族偏好改写为效果空间方向偏好，即 `context_key -> prefer_effect_direction`。  

**结论**  
采用选项 2。

**影响**  
规则维护机制仍可沿用字典、置信度增强/削弱和冲突审计等现有结构；复杂性转移到 SD/TVA 支持的短视界效果映射与方向向量学习中。该方向暂以离线审计为准，不直接接入在线训练。只有当效果方向在跨周期、跨 seed 或 shadow eval 中表现稳定时，才考虑转为在线 prior。

## 决策 10

**决策**  
第一版在线闭环测试采用“TD-MPC2 主训练线 + SD 接续总结线”的解耦结构，暂不实现 SD 与 TD-MPC2 的端到端协同嵌入。

**背景**  
当前研究目标是先确认方法论有效性：SD 是否能够在训练周期结束后总结经验、进行失败归因，并将稳定经验压缩为可审计规则。若在同一阶段把 SD 嵌入 TD-MPC2 在线世界模型或共享训练梯度，将难以区分收益来自周期总结机制、规则库积累，还是来自新的协同模型结构。

**选项**  
1. 在闭环第一版中把 SD 直接嵌入 TD-MPC2 在线训练，使其参与主训练梯度和规划模型。  
2. 保持 TD-MPC2 主训练线独立；SD 解释器仅在周期总结阶段从上一轮 checkpoint 接续训练，并通过规则库影响下一周期探索。  

**结论**  
采用选项 2。

**影响**  
后续闭环 runner 应明确区分两类 checkpoint：TD-MPC2 训练 checkpoint 与 SD 解释器 checkpoint。SD 可跨周期接续训练，但不参与 TD-MPC2 主训练优化器、在线 replay 更新或世界模型梯度。下一阶段若考虑嵌入协同，应作为新的实验变量单独设计。

## 决策 11

**决策**  
第七版 `10k x 5` 闭环结果不作为规则归纳失败处理，下一步保留闭环框架，优先补充候选规则拒绝原因诊断和周期轨迹覆盖统计。

**背景**  
第一轮闭环实验完成了 `5` 个周期，总预算为 `50000` 环境步。TD-MPC2 主训练线保持 `world_model_kind=tdmpc2`、`sd_online_supervision_enabled=false`、`sd_supervised_init_from=null`，SD 从第 2 周期起均成功加载上一周期解释器 checkpoint。官方评估仅在 10k 出现 `0.182293`，后续评估点为 `0.0`；训练 episode 在第 4 周期和第 5 周期出现间歇性正回报，但未形成稳定评估收益。5 个周期的规则库更新均为 `no_rule_added`。

该结果说明闭环执行链路已经成立，但当前单 seed、10k 周期和保守规则准入条件下，周期总结尚未产生足以进入在线规则库的新规则。根据既定方法论，规则产出允许稀疏甚至为空，尤其在 Hopper 这类奖励稀疏、阶段依赖强的任务中，不能把单轮空产出直接解释为方法失败。

**选项**  
1. 降低规则准入阈值，使每个周期更容易产生新规则。  
2. 保持当前保守准入策略，先增加候选规则拒绝原因统计、周期轨迹覆盖诊断和阶段指标报告。  

**结论**  
采用选项 2。

**影响**  
下一步不直接放大既有单条规则，也不以降低阈值换取表面规则数量。闭环框架继续保留，重点转向解释为何没有规则进入在线库：是否因为没有形成稳定候选、候选跨轨迹一致性不足、子状态拆分后样本不足、回放验证未通过，或好轨迹误伤风险过高。只有在这些拒绝原因被明确后，才调整规则阈值、周期长度、多 seed 规模或阶段门控条件。

## 决策 12

**决策**  
将闭环规则状态拆分为“周期候选”和“上线验证”两层，后续不再要求单周期、单 seed 直接产出 `replay_validated` 规则。

**背景**  
第七版 `10k x 5` 规则漏斗审计显示，每个周期的本周期 `library_v5.json` 均为空，闭环输出中的 `5` 条规则均继承自初始 `offline_v5`。直接卡点在规则生成前：第 2 至第 5 周期每轮只有 `3` 条官方评估 episode、`9` 条反事实状态记录，且全部为 bad 标签；v2 粗规则和 v4 子状态规则均主要因支持样本不足而未生成。同时，当前单 seed 运行与 v3/v5 的跨 seed 可靠性准入存在结构性不匹配。

**选项**  
1. 降低 `replay_validated` 准入阈值，使单周期更容易直接上线新规则。  
2. 保持 `replay_validated` 作为上线状态，但新增周期候选状态，用于保存单周期近邻候选并跨周期累计证据。  

**结论**  
采用选项 2。

**影响**  
后续闭环总结应输出未达上线阈值但有诊断价值的 `cycle_candidate` 或 `cycle_supported` 规则，并记录其支持数、优势、good/bad 覆盖和被拒绝原因。在线动作先验仍只读取 `replay_validated` 规则。这样可以保留保守上线原则，同时避免把早期周期的弱信号全部丢弃。

## 决策 13

**决策**  
第七点一版闭环采用 `5k` 触发频率，但周期总结使用滚动窗口和阶段感知轨迹分层，而不是把每个 `5k` 周期作为独立证据窗口。

**背景**  
第七版 `10k x 5` 审计显示，规则未新增的核心原因是每轮证据供给不足。若直接把周期改为 `5k` 且仍只读取当前周期 official eval，证据会进一步稀疏。因此，`5k` 更适合作为闭环反馈频率，而不是规则归纳的独立证据窗口。

同时，Hopper 早期回报常为 `0`，纯回报分层无法在失败轨迹中区分 near-good 与 bad。阶段诊断指标显示，即使 reward 为 `0`，不同轨迹在 `final_rootx`、`max_rootx`、支撑恢复、后退速度和旋转惩罚上仍存在明显差异。

**选项**  
1. 直接把 `cycle_steps` 从 `10000` 改为 `5000`，其他逻辑不变。  
2. 使用 `cycle_steps=5000` 触发闭环，但每次总结读取最近 `2` 至 `3` 个周期，并使用阶段感知分层选择 good/near-good 与 bad 对照。  

**结论**  
采用选项 2。

**影响**  
第七点一版推荐设置为 `cycle_steps=5000`、`cycles=10`、`summary_window_cycles=3`、`trajectory_score_mode=stage`，总预算仍为 `50000`。规则候选可跨周期累计并升级为 `rolling_validated`；在线先验默认仍只读取 `replay_validated`，若 pilot 需要测试滚动验证规则的闭环作用，可显式允许 `replay_validated,rolling_validated`。

## 决策 14

**决策**  
第七点一版 `5k x 10` pilot 的主要结论应表述为“闭环机制可靠运行并产生滚动验证规则”，而不是“已经取得 Hopper 性能收益”。

**背景**  
本轮实验完整运行到约 `50k` 环境步，10 个周期均正常触发官方评估，SD 解释器持续接续加载，阶段分层和滚动窗口稳定提供轨迹。规则库从初始 `1` 条在线规则扩展为 `1` 条 `replay_validated` 加 `3` 条 `rolling_validated` 在线可读规则，说明闭环归因和跨周期候选累计开始发挥作用。

同时，官方评估回报全程为 `0.0`。训练 episode 中出现间歇性非零回报，但尚未转化为稳定评估收益。因此，若将本轮解释为任务性能改善，会超过当前证据。

**选项**  
1. 将本轮作为性能有效性证明，直接推进更大规模闭环训练。  
2. 将本轮作为机制可靠性和规则生成有效性的 pilot，下一步增加同预算纯 TD-MPC2 对照或多 seed 复现。  

**结论**  
采用选项 2。

**影响**  
后续汇报应强调：第七点一版已经解决第七版中“规则库长期停留在初始规则”的问题，但尚未证明最终任务收益。下一阶段实验应围绕对照和复现展开，而不是继续只看单 seed 单运行。

## 决策 15

**决策**  
第七点一版 `100k` 接续结果应解释为“闭环可恢复且规则库继续演化，但尚未形成稳定评估收益”；下一步优先做同预算纯 TD-MPC2 对照和冲突感知规则整理，而不是直接增强规则权重。

**背景**  
本轮从 `cycle010` 的 TD-MPC2 checkpoint、SD 解释器 checkpoint 和闭环规则库接续到 `cycle020`，最终步数为 `100019`。接续链路完整运行，最终规则库包含 `23` 条规则，其中在线可读规则为 `1` 条 `replay_validated` 与 `7` 条 `rolling_validated`。训练 episode 在 `cycle019` 达到约 `64.175` 的局部高回报，显示训练分布中出现更强阶段性行为信号。

同时，官方评估回报在 `50009` 至 `100019` 区间均为 `0.0`。最终规则库中同一 `unsupported_high_torque|omega_neg|vx_low|support_none|prone` 子状态下同时存在多个不同偏好动作族，说明当前子状态表达仍然过粗，规则之间存在潜在冲突。

**选项**  
1. 将 `100k` 接续中的训练 episode 高回报视为方法有效性证明，并直接提高规则先验权重或扩大允许规则集合。  
2. 将本轮视为闭环恢复性和规则演化证据；性能有效性仍待对照验证，并先处理规则冲突与子状态过粗问题。  

**结论**  
采用选项 2。

**影响**  
后续实验不应仅依赖单 seed 闭环曲线做性能判断。优先补充同预算纯 TD-MPC2 对照，检查 `90k` 附近训练 episode 高回报是否超出基线波动；同时对最终规则库进行冲突感知合并或子状态再拆分，尤其处理 prone 子状态下 `hip_neg`、`waist_pos`、`knee_pos` 和 `waist_neg` 并存的问题。

## 决策 16

**决策**  
在进入规则内化之前，先保留显式 eval-time 规则补丁作为诊断工具，并修正规则事件匹配与审计口径。

**背景**  
100k checkpoint 的 `rule_off / rule_on` 成对 eval 显示，两组评估回报均为 `0.0`，但 `rule_on` 明显改变了规则命中状态下的动作方向：相反方向动作比例从 `40.1%` 降至 `3.5%`。这说明当前规则补丁在 eval 期确实生效，问题不只是规则没有被用于推理。

同时，检查发现在线 prior 的 `_state_matches()` 当前只允许 `unsupported_high_torque` 事件命中，导致 `rotation_risk` 规则虽然被加载但不会实际作用。这使“加载规则数”与“有效可命中规则数”不一致。

**选项**  
1. 直接进入规则内化，把当前规则作为 actor 监督或蒸馏目标。  
2. 先修正显式规则 eval 工具链，区分已加载规则和可实际命中规则，再决定内化目标。  

**结论**  
采用选项 2。

**影响**  
后续规则内化前，应先完成多事件规则匹配修正或显式限定在线规则库只包含当前实现可支持的事件。否则，内化目标可能包含未在真实 planner 中起作用的规则，导致诊断和训练目标不一致。

## 决策 17

**决策**  
将冲突规则处理从离散状态升级为连续信心机制，并允许信心值缩放在线动作先验强度。

**背景**  
`cycle020` 规则库中，同一子状态下存在多个动作族方向，尤其是 `unsupported_high_torque|omega_neg|vx_low|support_none|prone` 中同时出现 `waist_pos` 和 `waist_neg`。若继续只使用 `rolling_validated` 等离散状态，在线 prior 会把这些规则近似等强处理，导致冲突信号进入 planner 排序。

用户提出：规则应能被不断加强、削弱甚至消除；矛盾规则出现时，后来证据可先削弱原规则，原规则消除后若反向证据仍持续，则自然形成反向规则；规则信心也可影响先验动作引导权重。

**选项**  
1. 继续保留离散状态，只在报告中手动标注冲突规则。  
2. 引入连续 `confidence`，用正证据、冲突证据和缺席衰减动态更新规则强度，并将 `confidence` 作为在线 prior 缩放因子。  

**结论**  
采用选项 2。

**影响**  
规则库后续应同时记录 `confidence`、`effective_prior_scale`、`positive_evidence_cycles`、`negative_evidence_cycles` 和 `conflict_group`。在线 prior 默认兼容旧规则库；只有显式使用 confidence 规则库时，规则强度才按信心缩放。该机制使冲突规则可以通过证据竞争自然淘汰，而不是由人工直接删除。

## 决策 18

**决策**  
在重新训练前，先修正在线 prior 的多事件匹配，并优先考虑从 `100k` checkpoint 接续训练，而不是从头重跑。

**背景**  
使用 confidence 规则库进行不重训练 eval 后，三条评估回报仍为 `0.0`，但动作审计与阶段指标较原始 `cycle020_rule_on` 更干净：相反方向动作检查从 `3.5%` 降为 `0.0%`，`full_foot_support_frac` 从 `0.0027` 升至 `0.0793`，`backward_speed_frac` 从 `0.2807` 降至 `0.2060`。

同时，当前在线 prior 匹配逻辑仍只实际支持 `unsupported_high_torque` 事件，导致 confidence 库中的 `rotation_risk` active 规则虽被安装但不能命中。若在此时直接重训，仍会带着已知实现限制进入新实验。

**选项**  
1. 立即从头重新训练闭环实验。  
2. 先修正多事件匹配，再用 confidence 规则库进行 eval 或从 `100k` checkpoint 接续训练。  

**结论**  
采用选项 2。

**影响**  
下一步不从头重训。若修正多事件匹配后 confidence eval 继续显示更清晰的阶段信号，可从 `100k final.full.pt` 接续到 `150k` 或 `200k`，并以 confidence 规则库作为初始规则库。这样能把新增变量限制在规则 prior 机制上，避免重新训练引入额外随机波动。

## 决策 19

**决策**  
多事件匹配修正后，下一轮若继续实验，应从 `100k` checkpoint 接续短预算训练，而不是再次做同 checkpoint eval 或从头训练。

**背景**  
多事件匹配修正后，`rotation_risk` confidence active 规则已经实际命中，动作方向审计无相反方向；但 eval 回报仍为 `0.0`，阶段指标与修正前基本一致。由此可判断，显式 eval-time 规则 prior 已经较干净地发挥局部作用，但同一 checkpoint 上继续只做 eval 难以产生新的策略分布。

**选项**  
1. 继续在同一 checkpoint 上调整 eval-time prior 权重。  
2. 从 `100k final.full.pt` 接续短预算训练，观察 confidence prior 是否能在训练分布中积累支撑恢复与低后退速度信号。  

**结论**  
采用选项 2。

**影响**  
后续实验建议从 `100k` 接续到 `150k`，使用 confidence 规则库作为初始规则库，并允许 `confidence_active` 进入在线 prior。实验解释仍应保持谨慎：若训练 episode 和阶段指标改善但 eval 仍为零，应视为局部信号，而不是任务成功。

## 决策 20

**决策**  
后续 Hopper 闭环实验将周期内 `3` 条 eval 仅作为 quick eval，不再作为主要性能结论；关键 checkpoint 必须追加更宽 eval，并同时做 rule-on / rule-off 对照。

**背景**  
`100k -> 150k` confidence prior 接续实验中，训练 episode 在后半段明显增强，`cycle030` 最高训练回报达到 `145.26`，但周期内 `3` 条 official eval 在最终轮只有 `0.03`。用户指出该 train/eval 严重不一致不符合此前经验，并怀疑 eval 实现存在漏洞。

对同一 `150k final.full.pt` 做不训练诊断后发现，默认 eval 动作模式扩展到 `9` 条 episode 时，均值为 `64.03`，回报列表中包含多条 `140+` 轨迹。因此，最终 checkpoint 不是完全无效；周期内 `eval_episodes=3` 的固定前三条初始状态样本过少，导致严重低估。

同时，同一 checkpoint 关闭 rule prior 后，`9` 条 eval 均值为 `88.29`，高于 rule-on 的 `64.03`。因此，当前 confidence prior 在更宽评估样本上没有净收益，不能继续按“规则有效但回报尚未体现”解释。

**选项**  
1. 继续沿用 `eval_episodes=3`，将最终轮 near-zero eval 解释为方法失败。  
2. 将 `eval_episodes=3` 定义为 quick eval，并对关键 checkpoint 追加 `9` 条或更多 eval；同时用 rule-on / rule-off 对照判断规则 prior 净效应。  

**结论**  
采用选项 2。

**影响**  
后续报告不得仅用周期内 3-episode eval 判定 Hopper 性能。闭环 runner 可继续保留低成本 quick eval，但关键节点必须追加更宽评估并报告 `mean/std/min/max/list`。规则 prior 方面，下一步优先审查 active 规则在 `rule_on` 相对 `rule_off` 中造成负贡献的状态和动作，而不是增加规则权重或继续接续训练。

## 决策 21

**决策**  
规则 prior 的净效应判断采用保存 checkpoint 的标准 `10` 条 episode 系统 sweep；不得从单个 checkpoint 或临时 `9` 条 episode 诊断直接断言规则 prior 有害或有效。

**背景**  
150k checkpoint 的临时 `9` 条 episode 诊断显示 rule-off 均值高于 rule-on，但该结果只覆盖单个 checkpoint。用户指出，一个 checkpoint 上分数被拉低不能断言规则 prior 有害，并建议利用已保存的多个 checkpoint 按 DeepMind Control 常用的 `10` 条 episode 口径系统比较。

随后对 `cycle020` 至 `cycle030` 进行 `10` 条 episode rule-on / rule-off sweep。结果显示，`cycle020-028` 的差值多为小幅波动，`cycle029` 出现 rule-on 明显正向，`cycle030` 出现 rule-on 明显负向。

**选项**  
1. 保留 150k 单点诊断结论，将当前 confidence prior 直接标记为有害。  
2. 采用跨 checkpoint 标准 `10` 条 episode sweep，把规则 prior 效果解释为阶段依赖，并进一步审计正负差值最大的 checkpoint。  

**结论**  
采用选项 2。

**影响**  
后续对规则 prior 的表述应使用“尚未显示稳定净收益”“存在阶段依赖的正负局部效应”，而不是“规则有害”。下一步优先比较 `cycle029` 和 `cycle030` 的 episode-level 轨迹、规则命中状态和动作方向，解释为什么同一规则机制在相邻 checkpoint 上分别产生正负大幅差值。

## 决策 22

**决策**  
Explainability-v0 规则短接续实验完成后，暂不把该规则库视为可扩展长训练配置；下一步优先改进规则识别和上线判据，而不是提高 prior 权重或直接延长同一配置。

**背景**  
`100k -> 110k` 短接续实验显示，工程闭环能够正常运行：SD explainer 连续接续拟合，规则归因、候选生成、闭环规则库更新、在线 planner prior 均正常执行。训练日志中，规则 prior 命中行比例从 `cycle021` 的 `16/50` 上升到 `cycle022` 的 `24/50`，最终规则库保留 `12` 条 `explain_active` 规则，并新增 `1` 条仅供观察的 `cycle_candidate`。

同时，周期内 quick eval 均为 `0.0`，最终 checkpoint 的 `10` 条 episode 对照仅显示弱正向信号：`explain_on=15.955`，`rule_off=8.822`，差值 `+7.13`。收益集中在少数 episode，绝对性能仍低。

**选项**  
1. 将该结果解释为 explainability 规则已经有效，直接提高 prior 权重或延长同一配置。  
2. 将该结果解释为弱兼容信号，保留工程管线，优先审计规则识别、阶段覆盖、反事实样本不确定性和 rolling validation 阈值。  

**结论**  
采用选项 2。

**影响**  
后续推进应围绕“如何识别更关键、更稳定的规则”展开。特别需要检查 explainability 分数是否受少量 SD counterfactual 支配，`rear`、`omega_pos/mid`、`hip/ankle` 相关规则是否真正解释高低回报分岔，以及候选规则未上线的主要原因是支持不足、跨周期不一致，还是条件过细导致覆盖稀疏。

## 决策 23

**决策**  
`110k -> 200k` explainability-v0 接续实验作为中期适配观察，不作为 Hopper 成功或失败的最终判据；下一步优先改进规则识别和规则覆盖，而不是仅延长同一规则库或提高 prior 权重。

**背景**  
Hopper-hop 任务存在复杂动力学、稀疏奖励和阶段化恢复依赖，`200k` 预算仍明显低于通常稳定解决该任务所需的预算。因此，短期或中期训练结果更适合用于判断工程闭环、显式 prior 生效性、训练侧恢复片段和规则库演化，而不是直接要求达到任务级成功。

本轮从 `110k` 接续至 `200k` 后，训练 episode 的最高回报逐步提升到 `237.4`，`cycle039` 的训练 episode 均值达到 `115.3`，说明训练侧恢复片段增强。周期内 quick eval 仍大多为 `0.0`，但 `cycle032` 和 `cycle038` 分别出现 `58.57` 与 `74.19`。最终 `10` 条 episode 对照中，`explain_on=89.203`，`rule_off=88.514`，差值仅 `+0.690`，且成功 episode 集合没有变化。

规则库方面，最终为 `11` 条 `explain_active`、`3` 条 `rolling_validated`、`14` 条 `cycle_candidate`。后续周期没有新增在线规则，说明当前主要瓶颈在候选规则是否关键、是否稳定、是否覆盖关键阶段，而不是规则 prior 未被安装。

**选项**  
1. 继续沿同一规则库直接延长训练，并将更长预算作为主要变量。  
2. 将本轮作为中期适配观察，先改进解释力指标、候选筛选、冲突削弱和阶段覆盖，再进入更大预算训练。  

**结论**  
采用选项 2。

**影响**  
后续实验应围绕规则识别质量展开：检查候选规则为何未上线，评估 `rear`、`omega_pos/mid`、`hip/ankle` 等关键恢复阶段是否仍覆盖不足，并将解释力中的已发生轨迹解释、SD 反事实增益和跨周期稳定性结合为更严格的上线判据。更长训练可以保留为后续验证，但不应替代规则识别机制本身的改进。

## 决策 24

**决策**  
新的 Hopper 闭环实验采用三种子共享规则库，并将 5k 总结周期与 25k 正式评估周期分离。

**背景**  
前序实验显示，5k 周期总结对 SD 回归、失败归因和规则更新是必要的；但 5k 周期内低样本 quick eval 容易产生误导，不应作为性能判断依据。用户明确要求不要进行无意义 quick eval，但 SD 回归、归因和规则更新仍应保持 5k 周期。同时，单 seed 规则证据不足是此前候选难以上线的重要原因，因此本轮需要在多种子间共享规则。

**选项**  
1. 将训练周期、规则总结和 official eval 都统一设为 25k。  
2. 保持 5k 规则总结周期，但不做 5k official eval；每 25k 做一次 `10` 条 episode official eval 和正式 checkpoint；三个 seed 共享同一规则库。  

**结论**  
采用选项 2。

**影响**  
闭环总结的数据源改为训练 replay，而不是 5k quick eval trace。每个 5k 周期结束后，从三个 seed 的 `final.full.pt` 中导出 replay episode，合并后训练共享 SD explainer，并更新共享规则库。5k 周期末 `final.full.pt` 仅作为接续状态，`25k/50k/75k/100k` 才作为正式 checkpoint 与评估点。该设置更符合方法论目标：高频总结经验、低频稳定评估性能，并让规则证据在跨 seed 层面累计。

## 决策 25

**决策**  
v8 成功延续与阶段衔接规则采用精炼阶段标签，并且候选规则归纳只使用退化前的因果上游锚点。

**背景**  
v72 `200k` 最终 eval 的成功片段挖掘显示，许多正回报、高躯干、高前向速度片段被既有 SD 阶段检测器标记为 `prone`。若直接沿用该标签，v8 规则会与 v7 的普通失败恢复规则混在一起，难以表达“局部好姿态尚未成功衔接到下一支撑阶段”的语义。同时，若把 `degradation_onset` 帧也用于候选规则归纳，容易把退化已经发生后的动作误当作退化原因。

**选项**  
1. 完全沿用原始 SD 阶段标签，并在所有关键帧上归纳候选动作规则。  
2. 在保留原始阶段记录的同时，为 v8 规则键引入 `near_good_air / near_good_contact / near_good_support` 精炼阶段；候选规则只从 `last_success / pre_degradation / post_success` 等上游锚点提取，`degradation_onset` 只用于诊断。  

**结论**  
采用选项 2。

**影响**  
v8 规则候选将更明确地区分“已经失败后的恢复”与“局部好姿态的阶段衔接失败”。当前产物仍属于离线观测候选，不能直接上线为在线 prior；后续需要通过 SD 反事实目标 `S_continue` 和 `S_transition` 验证这些候选动作族是否具有可解释的决策收益。

## 决策 26

**决策**  
v8 转接规则在进入任何在线 prior 前，必须先经过 held-out 诊断和冲突削弱；冲突未解或被削弱的规则不得直接上线。

**背景**  
v8 离线流程已经同时产生观测层 `avoid` 候选和 SD 反事实层 `soft_prefer` 候选。初始合并结果显示，部分规则在同一 `rule_key / action_family` 上出现“观测上伴随退化，但 SD 反事实上优于实际动作”的冲突，也存在同一关节相反方向互相竞争的情况。若这些规则未经处理即进入在线 prior，可能在 planner 排序中相互抵消或误伤局部好姿态。

**选项**  
1. 将 `offline_v8_supported` 规则直接作为在线 prior 候选。  
2. 先在 episode-level held-out 子集上验证每条规则，再按同一规则键、同一关节执行冲突削弱；只有稳定支持且冲突已解决的规则才进入下一步误伤率验证。  

**结论**  
采用选项 2。

**影响**  
v8 当前产物定位为离线规则候选库，而不是在线规则库。`heldout_supported` 规则可进入下一步 replay 或跨 checkpoint 验证；`heldout_conflict_unresolved` 规则保持观察；`heldout_weakened_by_conflict` 规则应降低置信或暂不参与 prior。该决策延续此前规则信心可增强、削弱和消除的原则。

## 决策 27

**决策**  
v8 transition 规则的 replay 安全判据改为好姿态误伤率，而不是沿用 recovery 规则的坏片段命中率。

**背景**  
replay 安全审查显示，v8 transition 规则命中的几乎全部是 `near_good_air / near_good_contact` 片段。这是规则设计的直接结果：该类规则本来用于解释局部好姿态如何延续或如何衔接到下一支撑阶段。若继续沿用 v7 recovery 规则的审查标准，即要求规则主要命中坏片段，则会错误地把成功延续规则判为不合格。

**选项**  
1. 沿用 recovery 规则的回放审查标准，要求规则主要命中坏片段。  
2. 对 v8 transition 规则单独使用好姿态误伤率：`avoid` 规则检查是否压制好姿态中实际使用的动作族，`soft_prefer` 规则检查是否偏向好姿态实际动作的相反方向。  

**结论**  
采用选项 2。

**影响**  
v8 transition 规则的上线前审查将以“低误伤局部好姿态”为必要条件。能够解释退化但会压制好片段常用动作的规则，应标记为 `replay_false_intervention_risk` 或保持观察。当前高风险例子为 `near_good_contact|stable->avoid:knee_neg`，其 replay 好姿态误伤率为 `0.322`，不得直接进入在线 prior。

## 决策 28

**决策**  
第二轮在线测试优先使用 `v8_safe` 单独先验进行短接续 smoke；暂不把旧 v7 当前规则与 v8 safe 规则混合作为默认配置。

**背景**  
`200k` 三种子 checkpoint 的 eval-only 对照显示，`v8_safe` 三种子均值为 `144.026`，几乎等同于 `rule_off=144.137`，未表现出明显推理期破坏性。相比之下，`v7_current=129.935`，主要由于 seed3 从 `220.729` 降至 `174.007`。这说明旧 v7 当前规则在该 checkpoint 口径下存在阶段性负效应，而 v8 safe 窄候选至少通过了进入短在线测试的最低门槛。

**选项**  
1. 将旧 v7 规则和 v8 safe 规则直接混合，进入第二轮在线测试。  
2. 先用 v8 safe 单独先验做短接续 smoke，确认其训练期接入、规则命中和 near-good 转接片段变化，再决定是否与其他规则族组合。  

**结论**  
采用选项 2。

**影响**  
后续推进应从“尽快完成规则设计并进入第二轮在线测试”出发，不再继续扩展离线规则分析。第二轮 smoke 应保持短预算、低权重、低评估频率和标准 `10ep` 终点评估；主要判断 v8 safe 是否能在训练分布中正常接入且不破坏现有策略，而不是要求短期解决 Hopper。

## 决策 29

**决策**  
第二轮在线 smoke 结果解释为 v8 safe 规则闭环的工程与短程兼容性验证通过；后续应在保持低强度先验和标准评估口径的前提下，优先改进候选规则压缩、状态条件泛化和 validation 效率。

**背景**  
`200k -> 210k` 三种子短接续 smoke 已完整跑通。`cycle041` 与 `cycle042` 均完成训练、`10ep` formal eval、replay trace 导出、SD 接续训练、反事实归因、规则评分、规则验证和共享规则库更新。`205k` 三种子均值为 `178.394`，`210k` 三种子均值为 `180.607`，高于 `200k` eval-only v8_safe 起点 `144.026`，但训练预算仍远低于 Hopper 稳定成功所需规模。

规则先验命中率较低且平均先验项接近零：`cycle041` 合计命中率为 `0.100`，`cycle042` 为 `0.087`。这说明规则确实参与了规划排序，但其行为仍属于轻量显式先验。最终规则库包含 `9` 条规则，其中 `7` 条为 `v8_replay_safe_candidate`，`2` 条为 `cycle_candidate`。与此同时，`cycle042` 生成 `84` 条 substate 候选和 `95` 条中间库规则，导致 validation 耗时明显上升。

**选项**  
1. 将本次短程上升解释为 v8 safe 已有效，直接提高 prior 权重或扩大同一配置训练。  
2. 将本次结果解释为工程闭环和短程兼容性通过，保留低强度先验，先改进候选压缩、状态条件泛化、跨周期验证和 validation 效率，再进入更长在线测试。  

**结论**  
采用选项 2。

**影响**  
后续实验可以继续使用 `v8_replay_safe_candidate` 作为在线门控状态，但不应因为一次 `10k` smoke 的均值上升而直接提高干预强度。下一阶段应围绕规则质量推进：减少冗余 substate 候选，区分真正可迁移的转接条件与单周期偶然候选，提高 validation 速度，并继续用标准 `10ep` formal eval 做周期性判断。

## 决策 30

**决策**  
Hopper 规则方法的正式有效性测试不再以接续训练均值增长为主要证据；规则生成通过逻辑审查和 smoke test 后，应从头训练 `100k`，并与官方 TD-MPC2 同预算结果和轨迹审查共同对照。

**背景**  
`210k -> 250k` 接续实验在 `225k` 处给出三种子均值 `182.681`，与 `210k` 的 `180.607` 基本持平略升。但该比较缺乏因果解释力：TD-MPC2 即使不加载任何规则，也会随训练预算推进而改变性能。因此，仅观察 `rule_on` 多种子均值是否增长，最多说明规则闭环未明显破坏训练，不能证明规则相对于基础 TD-MPC2 产生了收益。

**选项**  
1. 继续把接续训练分数增长作为主要判断依据，并将 `210k -> 250k` 线跑完后解释规则收益。  
2. 中止接续训练，把该线定位为工程 smoke 和安全性观察；后续规则版本先通过逻辑审查与 smoke test，再从头训练 `100k`，与官方 TD-MPC2 同预算结果和轨迹审查共同比较。  

**结论**  
采用选项 2。

**影响**  
接续训练仍可用于检查规则 prior 是否能加载、SD 是否能接续拟合、规则库是否能滚动更新，但不得单独作为规则有效性的主证据。正式实验必须包含从头 `100k` 训练、标准 `10ep` eval、官方 TD-MPC2 同预算参考和轨迹审查。由于官方参考记录没有 `50k` 口径，`50k` 不作为主测试预算。若需要进一步确认因果机制，可补充同起点 `rule_on / rule_off` 对照，但其定位是机制解释，而不是替代正式从头训练比较。

## 决策 31

**决策**  
后续 Hopper 闭环变体放宽规则观察层，而不直接降低在线上线门槛。

**背景**  
空规则库正式 `100k` 主实验在 `cycle001-010` 中持续生成候选规则，但上线规则数量较少：`cycle003` 首次形成 `1` 条 `v8_replay_safe_candidate`，`cycle009-010` 增至 `2` 条。该现象与当前保守门控有关，包括每周期最多携带 `3` 条候选、晋升要求 `good_share <= 0.25`，以及超过该阈值后候选容易被削弱退出滚动池。Hopper 任务存在阶段依赖和恢复动作在好坏轨迹中共同出现的情况，过早削弱候选可能导致有价值规则无法跨周期积累证据。

**选项**  
1. 直接降低 `v8_replay_safe_candidate` 上线门槛，使更多规则进入在线 prior。  
2. 保持上线门槛不变，但增加候选携带数量，并将“削弱退出”的好轨迹占比阈值与“上线晋升”的好轨迹占比阈值分离。  

**结论**  
采用选项 2。

**影响**  
后续新变体默认采用 `max_cycle_candidates = 6`、`rolling_max_good_share = 0.25`、`rolling_weaken_max_good_share = 0.40`。`good_share <= 0.25` 仍是上线必要条件；`0.25 < good_share <= 0.40` 的规则只保留为观察候选，不参与在线 prior。当前正在运行的正式 `100k` 主实验不回溯修改，仍按启动时记录的参数解释。

## 决策 32

**决策**  
后续 Hopper 闭环变体将每周期直接上线规则数量上限从 `1` 条放宽为 `2` 条。

**背景**  
当前闭环更新器原始策略为 `at_most_one_rule_per_cycle`，即使一个周期内有多条规则已经满足在线状态，也只选择排序最高的一条合并到共享规则库。随着观察层候选数增加，若仍保持单周期只合并 `1` 条，规则库增长可能继续偏慢，尤其是在三种子共享、每周期约 `30` 条新训练轨迹、总结窗口约 `90` 条轨迹的设置下，单条上线吞吐量可能低于证据产生速度。

**选项**  
1. 保持每周期最多直接上线 `1` 条，继续最大限度避免规则库增长过快。  
2. 将直接上线数量参数化，并将后续默认上限设为 `2` 条。  
3. 取消直接上线数量上限，只要满足状态门槛就全部合并。  

**结论**  
采用选项 2。

**影响**  
`closed_loop_rule_update.py` 新增 `max_online_rules_per_cycle` 参数，报告策略从 `at_most_one_rule_per_cycle` 改为 `bounded_online_rules_per_cycle`。后续多种子共享规则脚本默认 `max_online_rules_per_cycle = 2`。该调整放宽的是上线吞吐量，不改变单条规则必须满足 `v8_replay_safe_candidate` 等在线状态门槛的要求。当前正在运行的正式 `100k` 主实验不回溯修改。

## 决策 33

**决策**  
正式 `100k` 空规则库闭环实验的主要结论应表述为“工程闭环成立，但当前规则识别与覆盖不足，尚未超过官方 TD-MPC2 `100k` 基线”。

**背景**  
`hopper_v8formal_100k_empty_seed123_20260508` 已完成 `20` 个周期。最终三种子 `100k / 10ep` 均值为 `4.366770`，低于官方 TD-MPC2 Hopper `100k` 均值 `13.63`。最终规则库包含 `21` 条规则，其中 `4` 条为 `v8_replay_safe_candidate` 并参与在线 prior。终点评估轨迹审查显示，规则确实命中并使部分动作方向与规则对齐，但高分 episode 主要来自 seed2，且规则命中步数占比较低。

**选项**  
1. 将当前结果解释为规则方法在 `100k` 预算下失败，并停止该方向。  
2. 将当前结果解释为工程管线通过、规则识别仍不足；下一阶段聚焦规则价值标准、关键转接状态和更高覆盖的规则提取。  
3. 忽略官方基线差距，只强调 seed2 中的局部高分片段。  

**结论**  
采用选项 2。

**影响**  
后续工作不应把当前 `100k` 结果作为性能优势证据，也不应仅因低预算未超过 TD-MPC2 而否定方法论。下一轮在线测试应使用已经实现的观察层放宽和每周期上线吞吐量参数，并在正式训练前先用轨迹审查确认新增规则是否覆盖高回报片段之前的关键转接条件。

## 决策 34

**决策**  
在下一轮正式 `100k` 重跑前，将 v8 transition 规则族接入每周期在线闭环；仅放宽原 v4 失败恢复规则的候选与上线数量不足以构成有希望的修订。

**背景**  
上一轮正式 `100k` 终点评估显示，最终上线规则确实命中并影响动作方向，但高分 episode 中规则命中步数占比通常只有 `0.012-0.080`，且规则主要集中于 `rear/prone` 恢复。旧 `cycle020` 轨迹审查显示，短暂高分片段存在，但这些片段几乎不被现有 v4 规则覆盖。单纯提高 `max_cycle_candidates` 或 `max_online_rules_per_cycle` 只能让同类恢复规则更快上线，不能解决关键阶段转接缺失。

**选项**  
1. 仅使用已修订的观察层放宽参数重跑 `100k`。  
2. 在重跑前补充 v8 transition 规则闭环，使每周期也总结短暂高分片段和 transition gap。  
3. 暂不重跑，继续只做离线规则分析。  

**结论**  
采用选项 2。

**影响**  
新一轮 `hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2` 实验仍从空规则库开始，避免高预算规则泄漏；但每周期总结除原 v4 失败恢复规则外，还会尝试生成 replay-safe transition 规则。正式结论仍需以 `100k / 10ep` 三种子结果、官方 TD-MPC2 `100k` 基线和终点轨迹审查共同判断。

## 决策 35

**决策**  
中止 `hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2` 的继续训练，不再将本轮推进到原计划 `100k`。

**背景**  
该实验原计划从空规则库运行 `100k`，每 `5k` 周期执行 SD 拟合、失败恢复规则更新和 v8 transition 规则更新，每 `25k` 保存 checkpoint 并执行标准 `10ep` eval。实际运行中，`25k`、`50k`、`75k` 三个完整评估节点均已落盘。`75k` 三种子均值为 `4.493884349172974`，仍未形成相对于既有官方 TD-MPC2 `100k` 参考的明显性能信号。更关键的是，截至 `cycle015` 的 `28` 条规则全部来自 `v4_substate` 恢复规则，其中 `5` 条上线为 `v8_replay_safe_candidate`；v8 transition 分支虽然在 `450` 条 trace 中找到 `55` 个 transition-gap 片段，但没有生成通过 validation 和 replay safety 的 transition 规则。

**选项**  
1. 继续运行到 `100k`，保留原计划完整预算结果。  
2. 在 `75k` 完整评估节点和 `cycle016` 部分训练已存在的情况下中止，将本轮作为规则识别失败模式诊断。  
3. 继续运行但临时降低 v8 transition 规则门槛，使更多 transition 候选上线。  

**结论**  
采用选项 2。

**影响**  
本轮实验不作为“v8 transition 规则改善性能”的正式正结果。其有效结论是：闭环工程管线、跨种子共享规则、在线规则 prior 加载和恢复类规则滚动上线均已成立；但 transition 规则提取与验证链条尚未把近成功片段转化为可上线规则。后续应先修订 transition anchor、观测动作证据聚合、SD 反事实目标和 replay safety 门槛，再重启新的从头在线测试。已落盘的 `25k`、`50k`、`75k` checkpoints 和 eval 可用于诊断，不应与完整 `100k` 官方基线作最终性能结论。

## 决策 36

**决策**  
v8 transition 规则默认使用 `phase_any` 聚合键，而不是完整 `refined_phase` 精细键。

**背景**  
`hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2` 的 `cycle015` 中，v8 分支已生成 `133` 条 transition action record 和 `78` 条 SD 反事实 record，但原精细规则键将证据拆为 `37` 个 group，最终候选规则为 `0`。离线审查显示，若保留 `omega/vx/support/risk` 而弱化 `refined_phase`，同一批证据可压缩为 `17` 个 group，并产生观测候选、SD 反事实候选、held-out supported 规则和 replay-safe 规则。

**选项**  
1. 继续使用完整 `refined_phase` 键，保持最强状态特异性。  
2. 使用 `phase_any` 聚合键，保留速度、角速度、支撑和风险条件，但跨 `falling/near_good_air/near_good_contact` 聚合证据。  
3. 进一步移除 `risk_bin` 或 `support_bin`，形成更粗规则。  

**结论**  
采用选项 2。

**影响**  
后续 v8 transition 规则键形如 `transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|support_risk`。该修改已同步到 transition action mining、replay safety 和在线 rule prior 匹配逻辑。离线 smoke 显示，`cycle015` 真实产物可导出 `1` 条 replay-safe transition 规则，false intervention rate 为 `0.0769`。该结果支持进入下一轮从头在线测试，但仍需在正式训练中观察其是否带来更高覆盖和更稳定的姿态转接。

## 决策 37

**决策**  
以空规则库从头启动 `hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2`，作为 `phase_any` transition 规则键的正式在线闭环测试。

**背景**  
上一轮 `v8trans` 实验在 `75k` 后中止，主要原因是精细 `refined_phase` 规则键导致 transition 证据过度稀疏，未能形成上线 transition 规则。离线 smoke 显示，将 v8 transition 规则键改为 `phase_any` 后，同一批真实 `cycle015` 产物可生成 `1` 条 replay-safe transition 规则，且 replay false intervention rate 低于当前安全阈值。因此，有必要重新从空规则库启动正式闭环，观察该规则键是否能在在线过程中稳定产生并应用 transition 规则。

**选项**  
1. 在旧 `v8trans` 实验上接续训练，利用已形成的恢复类规则继续推进。  
2. 从空规则库重新运行 `100k`，只保留代码层面的 `phase_any` 规则键修订。  
3. 直接用离线 smoke 产生的规则作为初始规则库启动新实验。  

**结论**  
采用选项 2。

**影响**  
新实验不继承旧 checkpoint、旧规则库或 smoke 规则，避免规则泄漏。评估仍采用三种子、每 `5k` 周期总结、每 `25k` checkpoint 与 `10ep` formal eval 的口径。若本轮能产生更多 replay-safe transition 规则，后续可进一步审查其对近成功姿态片段和动作方向的影响；若仍不能产生规则，则说明瓶颈不再主要是 phase token 稀疏，而需要继续修订 transition anchor、SD 反事实目标或 replay safety 审查。

## 决策 38

**决策**  
中止 `hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2` 的继续训练，将本轮定位为规则发现漏斗诊断，而不是推进至完整 `100k`。

**背景**  
该实验在 `cycle005 / 25k` 的标准 `10ep` formal eval 中均值约为 `0.0578`，但该节点不能直接判定规则无效，因为规则 prior 从 `cycle006` 才加载。`cycle006` 训练日志显示三种子均安装 `3` 条规则，`rule_prior_hit` 分别约为 `0.04 / 0.08 / 0.12`；轨迹审查显示 seed2、seed3 中规则方向动作比例上升、相反方向动作比例下降。因此规则 prior 已有局部行为作用。问题在于状态覆盖率只有约 `3.2%`，且 `cycle006` 与 `cycle007` 的普通恢复分支和 v8 transition 分支均没有稳定扩展上线规则。`cycle008` 启动时仍沿用旧的稀疏归因设置，继续运行主要会重复低覆盖恢复规则。

**选项**  
1. 继续运行至 `100k`，保留完整正式预算结果。  
2. 在确认规则有局部作用但规则库未能扩展后中止，将计算资源转回规则发现层修订。  
3. 临时降低 `v8_replay_safe_candidate` 门槛，使更多 transition 候选上线。  

**结论**  
采用选项 2。

**影响**  
本轮结果不作为 `phase_any` transition 规则改善性能的正式结果。其有效结论是：在线 prior 加载和轻量惩罚机制能够改变部分动作方向，但当前规则发现漏斗仍过窄。后续应先扩大反事实状态抽样、增加周期候选携带量，并将高 `good_share` 但高 support 或高 advantage 的规则保留为可拆分候选，而不是直接拒绝。在线门槛暂不降低。

## 决策 39

**决策**  
下一版闭环修订放宽规则发现层的采样和候选保留：`max_states_per_episode` 提高到 `10`，`max_cycle_candidates` 提高到 `12`，同时高 `good_share` 的高支持候选优先保留为候选而非直接拒绝。

**背景**  
`phase_any` 正式重跑在三种子、`25k-35k` 之间已经经历大量早期失败轨迹，但上线规则仍只有少数恢复类规则。离线补跑显示，旧状态更新逻辑会把部分高 support 或高 advantage 的规则因 `good_share` 偏高而拒绝；这在 Hopper 中可能过于激进，因为同一恢复或转接动作可能同时出现在好坏轨迹中，真正问题是状态条件需要继续拆分，而不是规则信号必然错误。

**选项**  
1. 直接降低在线上线门槛，增加 `v8_replay_safe_candidate` 数量。  
2. 保持在线门槛，放宽发现层采样和候选保留，使更多候选跨周期接受拆分与验证。  
3. 放弃当前恢复规则和 transition 规则分支，重新设计完整规则语言。  

**结论**  
采用选项 2。

**影响**  
后续从头重跑或接续诊断应显式记录这两个参数，不再沿用旧的 `max_states_per_episode=3` 和 `max_cycle_candidates=6` 作为主口径。该修改不会使未经验证的规则直接进入在线 prior；它只提高 SD 反事实和滚动证据积累的覆盖面。

## 决策 40

**决策**  
将 `hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1` 记为规则发现漏斗修订后的有效阶段性结果，但暂不将其表述为 Hopper 任务级成功。

**背景**  
本轮从空规则库完成 `100k` 三种子闭环训练，最终 `10ep` formal eval 均值为 `82.447`，高于既有 TD-MPC2 official `100k` 参考均值 `13.63`。同时，最终规则库形成 `20` 条可上线规则，其中 `7` 条来自 v8 transition replay safety，说明 widened funnel 解决了上一轮 transition 分支长期不上线的关键瓶颈。但终点轨迹仍以 prone/rear 状态为主，seed2 得分较低，较理想姿态主要表现为短片段而非稳定步态。

**选项**  
1. 将本轮作为明确性能正结果，直接进入更大预算训练。  
2. 将本轮作为规则发现和闭环机制的正向信号，先补做 transition 规则命中与姿态片段审查。  
3. 忽略本轮性能信号，重新回到完全离线规则设计。  

**结论**  
采用选项 2。

**影响**  
后续工作应保留 widefunnel 设置作为新的默认候选，但下一步不是直接扩大训练预算，而是审查 v8 transition 规则是否实际作用于高分片段前后的关键状态。如果规则命中与姿态改善一致，再启动下一轮更长预算或多种子在线测试；如果不一致，则继续修订 transition 状态谓词和 replay safety 审查。

## 决策 41

**决策**  
引入 rule-effect confidence 后处理机制，将规则作用审查中的方向一致率、反效果率和命中情况转化为规则信心，并默认只让 `confidence_active` 规则进入后续在线 prior。

**背景**  
widefunnel 终点审查显示，transition 规则不是整体有效或整体无效，而是内部明显分化。`waist_pos` 鼓励和 `waist_neg` 避免规则与高分 episode 高度共现且 harmful rate 低；`avoid:hip_neg` 与 `avoid:knee_pos` 高频命中但 harmful rate 超过 `0.64`。如果规则库只能用固定上线状态表示这些规则，就会把可靠规则和问题规则一并注入 planner。

**选项**  
1. 直接删除 harmful rate 高的规则。  
2. 保留原始规则库不变，只在文档中标注问题规则。  
3. 输出 confidence-weighted 规则库，使问题规则降为观察状态，可靠规则保持激活。  

**结论**  
采用选项 3。

**影响**  
新增 `apply_rule_effect_confidence.py`，将规则作用审查结果写入 `confidence`、`confidence_status` 和 `effective_prior_scale`。该机制不改写原始实验产物，后续在线测试可显式加载 confidence 规则库并设置 `rule_prior_allowed_statuses=confidence_active`。偶发不良规则由此不再必须人工删除，而是可被削弱、观察和后续再评估。

## 决策 42

**决策**  
将 rule-effect confidence 机制推进到下一轮在线闭环设计，而不是停留在终点后处理。

**背景**  
基于 widefunnel `cycle019` 实际生效规则库构造的 confidence 规则库，在同 checkpoint eval-only 对比中得到弱正向结果：`rule_off=76.201`、`original_cycle019=77.916`、`confidence_active=79.511`。confidence 规则库相对原规则均值提高 `+1.595`，相对 rule-off 提高 `+3.310`。该测试同时确认，剔除两条高 harmful transition avoid 规则后，显式规则 prior 没有发生明显退化。

**选项**  
1. 继续只在实验结束后做 confidence 后处理，不影响训练过程。  
2. 直接把 confidence 后处理结果作为新一轮初始规则库。  
3. 将 confidence 更新纳入每周期闭环规则库更新，使规则在训练过程中动态增强、削弱和退出。  

**结论**  
采用选项 3。

**影响**  
下一轮从头在线测试前，应把 rule-effect audit 或其轻量替代指标接入周期总结流程。在线 prior 默认只读取 `confidence_active`，规则强度按 `effective_prior_scale` 缩放；`confidence_observing` 规则保留证据但不参与动作排序。该设计使少量错误规则不再成为闭环实验的致命风险。

## 决策 43

**决策**  
从空规则库启动 `hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2`，作为 rule-effect confidence 在线闭环机制的正式 `100k` 三种子测试。

**背景**  
widefunnel 版本已经证明，放宽规则发现漏斗后可以形成更多可上线规则，且 v8 transition 分支不再长期为零。终点规则作用审查进一步显示，不同规则的实际行为效应存在明显分化：部分腰部 transition 规则与高分片段方向一致，而部分 hip/knee avoid 规则 harmful rate 较高。后处理 confidence smoke 显示，削弱高 harmful 规则后没有破坏原有显式规则 prior，并获得弱正向信号。因此，下一步应检验 confidence 是否能在训练过程中动态更新，而不是只在终点后处理。

**选项**  
1. 继续在 widefunnel 终点后处理 confidence，不进入新训练。  
2. 把 cycle019 后处理生成的 confidence 规则库作为初始规则库接续或重训。  
3. 从空规则库重新训练 `100k`，在每周期规则总结后执行 rule-effect audit 与 confidence 更新，下一周期只加载 `confidence_active`。  

**结论**  
采用选项 3。

**影响**  
新实验不继承旧规则，避免把旧实验产物作为先验泄漏到规则价值验证中。每 `5k` steps 执行 SD 回归、规则归纳、规则效应审查和 confidence 更新；每 `25k` steps 保存 checkpoint 并做标准 `10ep` formal eval。恢复规则与 v8 transition 规则的候选携带量均设为 `12`。后续分析将以规则生成质量、规则状态演化、方向一致率、harmful rate 和近成功片段覆盖为主，回报曲线作为辅助证据。

## 决策 44

**决策**  
修订 rule-effect confidence 的在线门控，使新规则先观察、有害规则可硬下线，并在上线前消除同状态相反有效动作方向冲突。

**背景**  
`hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2` 的 100k 结果明显低于官方 TD-MPC2 参考。对最终 checkpoint 做 rule-off / rule-on 诊断后发现，关闭规则后 seed2、seed3 仍低，说明训练期规则机制可能已经造成策略偏置；同时 seed1 在开启规则后下降，说明 eval-time 显式规则仍存在额外负效应。进一步审查发现，部分 `effect_harmful_weakened` 规则仍保持 `confidence_active`，方向一致性指标不直接衡量回报收益，且同一状态下可同时上线相反有效动作方向。

**选项**  
1. 保持现有 confidence 机制，仅在文档中标记问题。  
2. 完全关闭规则 prior，回到纯 TD-MPC2。  
3. 修订 confidence 门控，保留规则闭环主线，但显著提高上线安全性。  

**结论**  
采用选项 3。

**影响**  
新规则默认信心改为 `0.50`，低于 active 阈值；充分命中且 harmful rate 超过 `0.45` 的规则将立即降到 active 阈值以下；正向增强需要同时具备方向一致和好状态或高分 episode 上下文；同状态、同关节的相反有效方向在输出 confidence 规则库前进行仲裁。该修改会降低早期规则上线速度，但更符合“先确认规则价值，再影响训练分布”的方法论目标。

## 决策 45

**决策**  
先将“脚尖可发力但尚未形成全足支撑时延迟弹射”的规则作为离线 SD 诊断候选，而不是立即写入在线规则库。

**背景**  
100k 高分轨迹审查显示，Hopper agent 常在低身位或俯身恢复阶段借脚尖接触产生推进，但随后多数不能完成平衡恢复。该现象可以由人工观察直接提出规则假设，但研究目标是确认 SD 是否也能通过结构态推演给出一致诊断。

**选项**  
1. 直接把该规则作为人工种子规则写入在线 prior。  
2. 完全等待自动规则发现，不引入该失败模式的专门诊断。  
3. 先做离线 SD 诊断脚本，验证 SD 是否支持该规则，再决定是否接入 confidence 闭环。  

**结论**  
采用选项 3。

**影响**  
新增 `diagnose_hopper_toe_recovery_rule.py`，默认使用 100k `cycle020` SD explainer 和高分轨迹。该脚本输出 `offline_sd_diagnostic_candidate`，不改变训练、checkpoint 或现有规则库。若后续接入在线训练，该规则应接受 confidence 机制的增强、削弱和退休，而不是作为不可变硬规则。

## 决策 46

**决策**  
将脚尖恢复 SD 诊断规则以 `toe_recovery_low_support->soft_prefer:waist_pos` 的形式接入 100k 至 200k 三种子接续实验。

**背景**  
离线诊断显示，100k 高分轨迹中 `192` 个脚尖恢复低支撑状态的下一步释放率为 `1.000`，后续俯身率接近 `1.000`，SD 在 `98` 个状态中支持延迟弹射和支撑恢复规则。当前在线 prior 只能表达状态谓词到动作族的轻量排序偏置，尚不能直接表达“选择能改善支撑恢复的候选动作”这一效果型规则。

**选项**  
1. 直接上线多个 SD 偏好动作族，包括 `waist_pos`、`knee_pos` 和 `knee_neg`。  
2. 暂不接入在线训练，只保留离线诊断结果。  
3. 只上线无同关节反向冲突且支持数最高的 `waist_pos` 投影，并继续交由 confidence 机制审查。  

**结论**  
采用选项 3。

**影响**  
新规则库 `hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json` 在原 100k confidence 规则库基础上增加一条 `source=sd_diagnostic_toe_recovery` 的 `confidence_active` 规则。`online_rule_prior_v6.py` 增加该规则来源和 `toe_recovery_low_support` 谓词匹配。新实验从三个 100k checkpoint 接续至 200k，用于观察该 SD 诊断规则在已有闭环框架中的增量影响。

## 决策 47

**决策**  
修补 `toe_recovery_low_support` 的 rule-effect audit 口径，并停止旧接续实验后以 `auditfix` 新目录重新从 100k 接续至 200k。

**背景**  
`hopper_v8formal_200k_confidence_toerecovery_seed123_20260513_r1` 的早期周期显示，脚尖恢复规则在在线 prior 中仍为 `confidence_active`，但 `rule_effect_audit.json` 没有该规则的逐条统计，导致 confidence 更新把它解释为审计缺席并进行缓慢衰减。对 replay trace 的独立统计表明，满足 `toe_recovery_low_support` 的状态并不少见，因此问题来自审计脚本未同步该新规则族，而不是轨迹中没有对应失败窗口。

**选项**  
1. 保持旧实验继续运行，仅在分析时人工校正该规则的置信度解释。  
2. 在旧实验中途修补审计脚本并继续接续。  
3. 修补审计脚本，用旧周期回放验证该规则能进入逐条审计，然后以新实验目录从 100k checkpoint 重新接续。  

**结论**  
采用选项 3。

**影响**  
`audit_hopper_v8_widefunnel_rule_effects.py` 现在加载 `sd_diagnostic_toe_recovery` 来源规则，支持 `toe_recovery_low_support` 状态匹配，并对该规则取消高扭矩门控。旧 `r1` 实验作为不完整审计口径的诊断产物保留，不再作为正式结果解释。新实验目录为 `hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2`，用于检验脚尖恢复规则在完整审计和 confidence 更新闭环中的实际作用。

## 决策 48

**决策**  
新增 `stage_effect_prior` 规则类型，用于表达“小阶段进程”而非单步动作族偏好。

**背景**  
最终代 eval 审查显示，`toe_recovery_low_support->soft_prefer:waist_pos` 能在触发状态下改变动作方向分布，但不能稳定恢复全足支撑。这说明脚尖恢复低支撑问题不是单一动作方向可充分表达的控制原则，而是需要在短视界内完成支撑恢复、姿态稳定和旋转风险降低的阶段性过程。

**选项**  
1. 继续强化 `waist_pos` 单步动作族规则。  
2. 推翻现有规则维护框架，重新设计规则系统。  
3. 保留现有规则库和 confidence 生命周期，新增阶段式规则执行器和审计器。  

**结论**  
采用选项 3。

**影响**  
`online_rule_prior_v6.py` 增加 `stage_effect_prior` 在线候选序列打分；`audit_hopper_v8_widefunnel_rule_effects.py` 增加阶段目标回放审计；`build_hopper_toe_recovery_stage_rule.py` 可在既有规则库基础上追加 `toe_recovery_low_support->stage_effect:recover_full_foot_support`。该类型复用 `confidence_active` 等现有状态，因此可被当前闭环更新机制增强、削弱或退休。

## 决策 49

**决策**  
将脚尖恢复阶段规则的触发键从 `toe_only_launch_trap` 修正回 `toe_recovery_low_support`，并重启 100k 至 200k 三种子接续实验。

**背景**  
`hopper_v8formal_200k_toelaunch_stage_seed123_20260513_r1` 的 `cycle021` 至 `cycle023` 审计显示，`toe_only_launch_trap->stage_effect:stabilize_toe_only_launch` 连续零命中。对同一批 replay trace 的独立统计显示，原 `toe_recovery_low_support` 谓词仍命中 `56949` 帧，而 `toe_only_launch_trap` 谓词为零。主要原因是新谓词额外加入的 `torso_z >= 0.55` 与当前 replay 字段口径不匹配，导致训练回放中的相关状态被全部排除。

**选项**  
1. 继续运行 `toelaunch_stage` 实验，并在结果解释中注明规则未触发。  
2. 仅移除 `torso_z >= 0.55` 条件，保留 `toe_only_launch_trap` 新触发键。  
3. 恢复为原先大量命中的 `toe_recovery_low_support` 触发键，只将规则表达从单步动作偏好升级为阶段结果评分。  

**结论**  
采用选项 3。

**影响**  
`build_hopper_toe_recovery_stage_rule.py` 现在生成 `toe_recovery_low_support->stage_effect:stabilize_toe_only_launch`。修正版初始规则库为 `hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json`，并已通过 replay audit smoke 确认命中 `56949` 帧。旧 `toelaunch_stage` 实验作为触发谓词偏移的诊断产物保留，不再作为阶段规则有效性实验解释。新实验目录为 `hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1`。

## 决策 50

**决策**  
将统一的脚尖恢复阶段规则拆分为可恢复子阶段 active 规则和失稳子阶段 observing 规则。

**背景**  
`hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1` 运行至 `cycle033` 后，统一规则 `toe_recovery_low_support->stage_effect:stabilize_toe_only_launch` 的触发次数持续较高，但方向一致率由早期约 `0.63` 下降到约 `0.55`。进一步按触发后的结构态变化分析发现，该规则确实定位了脚尖低支撑窗口，但不同阶段后果不同：`prone/rear` 状态中支撑保持和旋转控制更常改善；`falling_air/toe_support` 状态中支撑丢失、旋转加剧和跌落更常出现。

**选项**  
1. 保留统一规则，继续依靠 confidence 机制整体增强或削弱。  
2. 直接删除脚尖恢复阶段规则，避免噪声影响在线训练。  
3. 保留诊断问题，但将触发谓词按结构阶段拆分；仅让可恢复子阶段进入在线 prior，失稳子阶段先观察。  

**结论**  
采用选项 3。

**影响**  
`online_rule_prior_v6.py` 与 `audit_hopper_v8_widefunnel_rule_effects.py` 新增 `toe_recovery_recoverable_low_support` 和 `toe_recovery_unstable_low_support` 两个一致谓词。`build_hopper_toe_recovery_stage_rule.py` 生成新规则库 `hopper_165k_cycle033_confidence_plus_toe_recovery_split_stage_effect.json`，其中 `toe_recovery_recoverable_low_support->stage_effect:stabilize_recoverable_toe_support` 为 `confidence_active`，`toe_recovery_unstable_low_support->stage_effect:observe_unstable_toe_support` 为 `confidence_observing`。旧 `toerecovery_stagefix` 实验已停止，作为统一阶段规则的诊断产物保留。

## 决策 51

**决策**  
停止旧拆分阶段规则实验，并将可恢复脚尖规则从软阶段成果规则修订为阶段约束式规则。

**背景**  
`hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1` 运行至 `cycle037 / 185k` 后，轨迹审查显示 `toe_recovery_recoverable_low_support` 虽然大量触发，但触发后的支撑、脚跟/全足接触、旋转和跌落状态没有相对旧统一规则形成明显改善。进一步审查发现三类问题：规则目标偏软，允许候选序列不恢复全足支撑也获得分数；规则写入 `horizon=5`，但当前 TD-MPC2 在线规划实际为 `horizon=3`；审计中的 `support_contact` 未回退读取 replay CSV 中的 `foot_grounded`，可能导致 confidence 证据口径偏差。

**选项**  
1. 继续运行旧拆分实验至 200k，仅在结果解释中注明阶段规则未形成明显引导。  
2. 放弃阶段成果规则，回到单步动作族规则。  
3. 停止旧实验，修正审计口径，并将可恢复脚尖规则改为带显式支撑、脚跟/全足进展、跌落和旋转约束的阶段约束式规则。  

**结论**  
采用选项 3。

**影响**  
`audit_hopper_v8_widefunnel_rule_effects.py` 修正 `stage_metric("support_contact")` 字段别名；`online_rule_prior_v6.py` 新增 `stage_constraints` 候选序列惩罚；`build_hopper_toe_recovery_stage_rule.py` 生成 `toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support`，默认 `horizon=3`。新规则库为 `results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json`。旧拆分实验保留为诊断产物，不再继续作为正式阶段约束规则实验解释。

## 决策 52

**决策**  
阶段约束规则不进入 185k 至 190k 短接续训练，先修正规则评分与结构目标转化机制。

**背景**  
`cycle037 / 185k` eval-only 烟测显示，`toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support` 能运行、能触发，也能在离线候选排序中产生一定结构信号；但真实 eval 轨迹中，rule-on 和 stage-only 均没有改善触发后的支撑保持、全足恢复和旋转控制。stage-only 相比 rule-off 的 `expected_good_rate` 从 `0.0021` 降至 `0.0008`，`mean_min_support` 从 `0.5867` 降至 `0.4846`，`mean_max_rotation_penalty` 从 `0.2105` 升至 `0.2676`。这说明当前规则未通过“触发后结构改善”门槛。

**选项**  
1. 仍然从 185k 接续至 190k，用训练过程观察 confidence 是否自然削弱该规则。  
2. 只看 reward 均值接近，直接进入更长接续训练。  
3. 停止进入训练，把 eval-only 结构烟测作为前置门槛，先修正规则评分和转化机制。  

**结论**  
采用选项 3。

**影响**  
新增只读对比脚本 `compare_hopper_stage_constraint_evalonly.py` 和 stage-only 规则库 `hopper_185k_cycle037_toe_recovery_stage_constraint_only.json`。阶段约束规则后续修订必须先在 eval-only stage-only 口径下证明触发后结构指标改善，再进入闭环接续训练。该决策避免把尚不可控的规则写入训练过程，防止策略在不良 prior 作用下继续固化。

## 决策 53

**决策**  
在线阶段规则必须支持显式加载同周期 SD explainer；约束优先规则仍暂不进入接续训练。

**背景**  
进一步审查发现，离线候选排序审计使用 `cycle037` 已拟合 SD explainer，而普通 TD-MPC2 checkpoint 的在线 eval prior 若未额外传参，只能回退到默认 `DMControlHopperSDModel`。该口径差异会导致“离线 SD 认为更好”的候选排序信号不能直接解释在线行为。为此新增 `rule_prior_sd_checkpoint` 参数，并将阶段约束评分改为约束优先形式。离线候选排序中，top candidate 的约束满足率由旧版 `0.7017` 提升到 `0.8367`。但显式加载 `cycle037` SD explainer 的 stage-only eval-only 仍未改善真实触发后结构：`expected_good_rate` 为 `0.0013`，低于 rule-off 的 `0.0021`；`mean_min_support` 为 `0.4933`，低于 rule-off 的 `0.5867`。

**选项**  
1. 认为离线候选排序改善已经足够，直接接续训练。  
2. 只保留 `rule_prior_sd_checkpoint` 修补，放弃约束优先规则。  
3. 保留两项工程修订，但不进入训练；下一步审计真实 MPPI 候选池中的 raw value、rule bonus、elite 选择和最终动作，确认转化失败发生在候选分布、打分权重还是环境后果上。  

**结论**  
采用选项 3。

**影响**  
`online_rule_prior_v6.py` 支持 `rule_prior_sd_checkpoint`，`eval_hopper_v7_rule_prior.py` 报告 SD checkpoint 加载状态，`build_hopper_toe_recovery_stage_rule.py` 生成约束优先规则。当前阶段约束规则虽然在 SD 离线候选排序中更合理，但仍未通过真实 eval 结构烟测，因此不启动 185k 至 190k 接续训练。后续工作转向真实 planner 候选级审计。

## 决策 54

**决策**  
暂时将主线实验切回旧的单动作/动作族置信规则方案，并从 200k 接续训练至 300k。

**背景**  
阶段约束式规则的离线 SD 候选排序已有改善，但显式加载同周期 SD explainer 后，真实 eval 轨迹仍未表现出触发后支撑、脚跟/全足恢复和旋转控制的结构改善。因此，阶段式规则当前更适合作为诊断分支，而不是继续写入训练过程。相比之下，旧的 `confidence_toerecovery_auditfix` 方案在 200k formal eval 中取得 `174.4901` 的三种子均值，略高于官方 TD-MPC2 200k 均值 `166.6333`，且其规则生成、共享、confidence 更新和在线轻量 prior 管线均已打通。

**选项**  
1. 继续修订阶段约束规则，并等待其通过真实 eval 结构烟测后再训练。  
2. 放弃规则训练主线，仅做候选级审计和离线诊断。  
3. 阶段规则保留为诊断分支，主线回到旧的单动作/动作族规则方案，从 200k 接续到 300k，观察策略网络和规则自学习是否能长期改善姿态。  

**结论**  
采用选项 3。

**影响**  
新增启动脚本 `scripts/launch_hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1.sh`。实验从 `hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/cycle_040` 的规则库、SD checkpoint 和三种子 TD-MPC2 checkpoint 接续，运行至 `cycle060 / 300k`。在线 prior 只允许 `confidence_active` 规则，以 `penalty_only`、`rule_prior_weight=0.05` 的低强度方式介入 MPPI 候选排序。阶段约束式规则不进入该训练线。

## 决策 55

**决策**  
TVA 规则价值信号继续保持审计/影子层，不进入 live confidence update；shadow update 采用 gated 口径。

**背景**  
TVA episode-level 价值信号已经能够区分高低回报轨迹，反事实动作排序也有局部解释力。但第一版 shadow update 的滞后预测未显示稳定跨周期预测关系，mismatch 诊断还发现 TVA 抽样覆盖相对 rule-effect 全量命中过低、部分增强来自高熵最优动作族、部分削弱与 rule-effect 一致率存在冲突。若此时进入在线更新，会把尚未充分验证的局部反事实信号写入训练分布。

**选项**  
1. 直接将 TVA shadow delta 写入正式 confidence 更新，并启动重训练。  
2. 放弃 TVA，只保留既有 rule-effect confidence。  
3. 保留 TVA，但限定为 gated shadow update：要求足够命中、TVA/effect 覆盖率、低 best-family entropy、明确 target-opposite margin 和方向支持比例间隔；通过报告继续验证，不改变在线 prior。  

**结论**  
采用选项 3。

**影响**  
[apply_tva_shadow_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_shadow_update.py) 新增 `--min-directional-rate-margin`，并将 best-family entropy 门控同时用于增强与削弱。闭环入口同步暴露 `--tva-shadow-min-directional-rate-margin`。以 `cycle041/050/060` 回填验证后，gated 版本的方向冲突为 `0`，mismatch 审计为 `0`；但 `cycle041/050` 无 strengthened/weakened 样本，尚不足以证明跨周期预测力。因此后续仍应先改进 TVA counterfactual 覆盖与验证样本量，再决定是否进入正式在线更新。

## 决策 56

**决策**  
TVA counterfactual 审计新增 `label_event` 覆盖增强模式；TVA shadow 默认门控收紧为 `min_effect_hit_ratio=0.003` 与 `max_best_family_entropy=0.65`。

**背景**  
300 条分层 TVA 审计在严格 gated 口径下几乎不产生 `cycle041/050` 影子改动，无法检验 TVA 是否具有跨周期规则价值预测力。将审计样本扩展为每轮 `1200` 条并按 `label × event` 分层后，三轮 matched state-rule pairs 提升到约 `8k`，稳定增强/削弱候选重新出现。但基础门控下 mismatch 也重新出现，每轮约 `5` 条，说明覆盖增强需要配合更严格的证据质量门控。

**选项**  
1. 保持 300 条 label 分层审计，接受 shadow 样本不足。  
2. 直接采用 1200 条 `label_event` 审计和原门控，允许更多 shadow 改动。  
3. 增加 `label_event` 审计模式，但将 TVA shadow 默认门控收紧，继续作为审计/影子层验证。  

**结论**  
采用选项 3。

**影响**  
[audit_hopper_tva_counterfactuals.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_counterfactuals.py) 支持 `--sample-mode label_event` 并缓存 trace CSV；闭环入口新增 `--tva-audit-sample-mode`，默认仍为 `stratified`。严格门控回填结果为：`cycle041` 中 `1` 条增强、`3` 条削弱、`3` 条 mismatch；`cycle050` 中 `4` 条增强、`5` 条削弱、`1` 条 mismatch；`cycle060` 中 `7` 条增强、`4` 条削弱、`0` 条 mismatch。该结果支持继续沿 TVA 规则价值验证方向推进，但仍不足以进入在线 confidence 更新。

## 决策 57

**决策**  
TVA shadow 规则进入 live confidence update 前，必须通过真实轨迹短窗口后果层验证；当前验证结果不支持上线。

**背景**  
`label_event1200_strict_gated` 回填后，TVA shadow 已形成跨周期稳定候选。但 rule-effect mismatch 和 `actual_alignment_but_tva_opposite` 现象说明，仅凭 TVA 反事实排序和实际动作同向统计，仍无法判断规则价值是否可靠。为此新增后果层审计，比较被增强规则的同向动作与相反动作、被削弱规则的相反动作与同向动作在真实轨迹后续窗口中的结构后果。

**选项**  
1. 将严格门控 TVA shadow 直接作为下一轮在线 confidence 更新依据。  
2. 放弃 TVA shadow，只保留现有 rule-effect confidence。  
3. 保留 TVA shadow 作为候选价值信号，但增加真实后果层验证作为上线前门槛。  

**结论**  
采用选项 3。

**影响**  
新增 [audit_tva_shadow_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_consequences.py)。以 `horizon=10` 审计 `cycle041/050/060` 后，后果层支持率分别为 `0.250 / 0.444 / 0.333`；`horizon=3/5` 中 `cycle060` 提升到 `0.667`，但 `cycle041/050` 仍为 `0.250 / 0.444`。该结果表明 TVA shadow 中存在可用信号，但尚未达到上线标准。后续应将后果层支持纳入 shadow 候选筛查，而不是直接重训练。

## 决策 58

**决策**  
在 TVA live update 之前，采用 consequence-gated shadow update 作为新的候选筛查口径。

**背景**  
严格 TVA shadow 门控虽然减少了 mismatch，但仍保留了若干在真实轨迹短窗口后果中不被支持的规则，尤其是 `hip_pos` 增强和 `hip_neg` 削弱。后果层验证显示，只有一部分 TVA shadow 候选在真实轨迹中满足期望方向优于相反方向。因此，应将后果层验证从报告提升为 shadow 后处理门控。

**选项**  
1. 忽略后果层验证，直接使用 strict-gated TVA shadow。  
2. 将所有后果层不支持的规则永久删除。  
3. 生成 consequence-gated shadow report：只保留后果层支持的 shadow 改动，其他改动回退为原始 confidence，并保留拒绝原因。  

**结论**  
采用选项 3。

**影响**  
新增 [apply_tva_consequence_gate.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_consequence_gate.py)。以 `horizon=5` 后果审计作为门控依据后，`cycle041/050/060` 的保留改动分别为 `1/4/4`，拒绝改动分别为 `3/5/7`，mismatch 降为 `1/0/0`。该结果说明 consequence gate 有助于过滤不稳定 TVA 信号，但仍属于审计/影子层，不进入在线规则库或训练。

## 决策 59

**决策**  
将 TVA consequence gate 接入闭环入口，但默认关闭，并保持 shadow-only 输出。

**背景**  
离线回填表明 consequence-gated TVA shadow 能过滤一批不稳定规则，但尚未证明其在完整在线周期流程中能持续稳定地产生候选。下一阶段需要采集周期内自动生成的 shadow-only 报告，而不是立即把 TVA 候选写入在线 prior。

**选项**  
1. 继续只用离线回填脚本，暂不接入闭环。  
2. 直接让 consequence-gated TVA shadow 替换正式规则库。  
3. 接入闭环脚本，但只写出 `tva_shadow_consequence_audit` 与 `tva_shadow_consequence_gated_update` 报告，训练仍使用原 confidence 规则库。  

**结论**  
采用选项 3。

**影响**  
`run_hopper_v7_closed_loop.py` 和 `run_hopper_v7_multiseed_shared_rules.py` 新增 `--enable-tva-consequence-gate` 及相关参数。dry-run 已确认命令顺序为 `TVA audit -> TVA shadow update -> consequence audit -> consequence-gated shadow update`，且最终传给下一周期的 shared rule library 未切换为 TVA 输出。后续可启动 shadow-only 闭环采集实验，用于观察 consequence-gated TVA 规则候选的跨周期稳定性。

## 决策 60

**决策**  
TVA terminal prior 的在线启用时机改为由 TVA-SD 接口审计阈值决定，而不是由固定训练步数或 checkpoint 是否存在决定。

**背景**  
非门控 `rule+TVA terminal prior` 从头训练在 `200k` 下明显低于历史 rule-only 结果。代码审查显示，TVA terminal prior 对每次 MPPI 规划的所有候选序列生效，并在候选集合内归一化后以固定尺度加入 planner value；该介入范围明显大于原有稀疏规则先验。即使 TVA 在真实状态上已有一定价值信号，若其在 SD 推演末端状态上的估值尚未通过接口校准，也可能过早改变探索分布。

**选项**  
1. 按经验步数延迟启用 TVA，例如 `50k` 或 `100k` 后启用。  
2. 保持从 `cycle002` 起只要 checkpoint 存在即启用。  
3. 在每个总结周期运行 TVA-SD 接口审计，只有上一周期审计通过阈值，下一周期才启用 TVA terminal prior。  

**结论**  
采用选项 3。

**影响**  
`run_hopper_v7_multiseed_shared_rules.py` 新增 `--enable-tva-terminal-prior-gate` 及相关阈值参数。门控默认检查 `h=1,3` 下的 `oracle_pearson`、`sd_pearson`、`sd_r2` 与 `sd_rmse / baseline_rmse`，并将结果写入 `cycle_xxx/reports/tva_terminal_prior_gate.json`。新增启动脚本 `scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh`，用于从头验证 gated TVA terminal prior。该设计保留 TVA 训练、TVA 审计和 shadow 报告，但在接口可靠性不足时阻止 TVA 进入在线动作排序。

## 决策 61

**决策**  
将 TVA terminal prior 接口门控从单一强门槛修订为强/弱分层门控。

**背景**  
`hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 到 `75k` 时仍未通过强门控，但历史报告显示 `cycle007`、`cycle016`、`cycle017` 等周期已出现方向性相关信号。强门控要求相关性、R2 和 RMSE ratio 同时达标，适合作为“正常权重上线”的标准，但对早期低权重试用可能过于保守。

**选项**  
1. 直接降低原强门控阈值。  
2. 取消门控，让 TVA 从 checkpoint 可用后直接进入 planner。  
3. 保留强门控，并新增弱门控：只在方向性证据通过时，以显著更低的 `weight` 和 `max_bonus` 进入动作排序。  

**结论**  
采用选项 3。

**影响**  
`run_hopper_v7_multiseed_shared_rules.py` 新增 `--enable-tva-terminal-prior-weak-gate`、弱门控阈值、弱 TVA 权重与接续初始 policy 参数。强门控通过时使用 `weight=0.02, max_bonus=0.05`；弱门控通过时使用 `weight=0.005, max_bonus=0.01`。门控报告新增 `strong_passed`、`weak_passed` 与 `decision.mode`，其中 `passed` 表示下一周期是否以强或弱模式允许 TVA terminal prior 上线。该修订使 TVA 能在方向性信号出现时进行可控、可审计的轻量介入，同时保留强门控作为正常权重上线标准。

## 决策 62

**决策**  
在 Hopper 底层状态标签中新增 `grounded_hop`，并收缩 `grounded_prone` 的语义范围。

**背景**  
`175k` seed1 最高分轨迹 `episode_03` 的回报达到 `491.6247`。视频抽帧显示，该轨迹视觉上已经基本呈现标准 Hopper hop 循环，但旧姿态审计仍报告 `grounded_prone_frac=0.614`。代码审查确认，旧 `grounded_prone` 主要由接地、非 rear、非严格 upright/prepared 和 `nose_minus_anchor_x >= 0` 共同决定，因此会把正常 hop 的屈膝前倾支撑误记为 prone。

**选项**  
1. 保持旧标签，仅在报告中人工解释 `grounded_prone` 的局限。  
2. 放宽 `grounded_upright`，把正常 hop 支撑并入 upright。  
3. 新增中间状态 `grounded_hop`，保留 `grounded_upright` 的严格姿态语义，并将正常 hop 支撑从 `grounded_prone` 中剥离。  

**结论**  
采用选项 3。

**影响**  
`DMControlHopperSDModel.STATE_NAMES` 追加 `grounded_hop`，阈值为 `height >= 0.55`、`abs(pitch) <= 1.10`、`vx >= 0.75` 且无身体碰撞。为兼容旧 CSV，`falling/rear/prone/upright/prepared/rising` 的原有编号保持不变，`grounded_hop` 使用追加编号。评估 trace 导出、replay trace 导出、在线 rule prior、v8 success mining、widefunnel rule-effect audit 和 posture comparison 均同步识别 `grounded_hop`。后续报告应同时列出 `grounded_hop_frac` 与 `grounded_prone_frac`，避免把标准 hop 的前倾蓄力和接地推进误解释为趴地失败。旧 CSV 文件不自动改写；需要比较旧实验时，应重新导出或重算标签。

## 决策 63

**决策**  
将 `grounded_hop` 从核心排他状态划分中降级为诊断性派生特征。

**背景**  
修订状态标签后的从头实验 `tdmpc2_v8formal_200k_tva_terminal_prior_gated_hoplabel_fromscratch_seed123_20260519_r2` 在 `75k` 前明显慢于上一条高分线。同期 TVA terminal prior 尚未通过门控并实际未上线，因此早期退化更可能来自状态标签修订对 SD 反事实归因、轨迹分层或规则维护输入分布的影响。审计显示，新线规则命中率更高，但命中与回报的相关性更差，transition rule 的命中-回报相关性由正转负。进一步检查发现，早期 replay summary 中 `grounded_hop_frac` 基本为 0，问题并非大量失败轨迹被直接导出为 hop，而是 SD 内部把 `grounded_hop` 加入 `state_confident` 和 `_dominant_state_ids`，并让其在 `grounded_prone` 之前排他占位，可能扰动早期恢复状态的语义。

**选项**  
1. 保持 `grounded_hop` 的排他状态地位，继续等待长训练观察。  
2. 完全删除 `grounded_hop`，回到旧标签体系。  
3. 保留 `grounded_hop` 作为姿态审计和高分轨迹解释特征，但不让它进入 `state_confident`、不从 `grounded_prone` 中排他扣除，也不写入 `_dominant_state_ids` 的核心状态编号。  

**结论**  
采用选项 3。

**影响**  
`grounded_hop` 仍可用于报告标准 hop 片段，避免将高分轨迹的前倾接地推进误读为趴地失败；但早期 SD 归因、状态编号和恢复型规则主干不再被 hop 标签重划分。新增 `100k` 从头验证线 `tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1`，用于检查第 `25k/50k/75k/100k` 的 formal eval、规则命中-回报相关性、transition rule 相关性和 seed1 是否恢复早期爬升趋势。

## 决策 64

**决策**  
TVA 上下文动作竞争暂不进入 live confidence update，也不直接替换在线 active 规则库；其下一阶段用途限定为规则价值的附加审计信号。

**背景**  
`cycle040` 最终规则库中仍存在同一状态上下文下多个动作族并存、甚至同关节相反方向并存的问题。为避免继续依赖 Hopper 姿态硬编码，新增 TVA 上下文动作竞争审计：在同一规则上下文内，用 SD rollout 与 TVA terminal value 比较不同动作族，标记 `tva_supported` 与 `tva_suspect` 规则。该审计在 `cycle040` 中报告 `17` 个上下文、`7` 个冲突上下文、`21` 条 TVA-supported 规则和 `18` 条 TVA-suspect 规则，并能解释若干同上下文相反动作族冲突。

随后生成 TVA context competition shadow library，将 active 规则由 `61` 条压缩为 `51` 条。但 eval-only 对照显示，影子库在同一 `200k` checkpoint 上没有带来即时收益：`rule_off`、原规则库、影子库三种子均值分别为 `125.407`、`127.866`、`120.461`。行为审计进一步显示，原规则库与影子库的 selected-gate 命中率分别为 `0.6159` 和 `0.6092`，加权有害方向比例约为 `0.3011` 和 `0.3016`，没有改善。更关键的是，部分被 TVA 影子库降权的原 active 规则在 eval 中方向一致率很高，例如 rear 恢复状态下的 `waist_pos` 和 `hip_neg` 规则。这说明单纯终端价值竞争可能误伤低回报但必要的恢复型规则。

**选项**  
1. 将 TVA 上下文竞争影子库直接作为下一轮在线训练规则库。  
2. 放弃 TVA 上下文竞争，只保留原有 rule-effect confidence。  
3. 保留 TVA 上下文竞争，但仅作为附加审计特征；未来进入更新前必须与 rule-effect、真实短窗口后果、规则角色分类共同门控。  

**结论**  
采用选项 3。

**影响**  
新增 [audit_tva_context_action_competition.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_context_action_competition.py)、[build_tva_context_competition_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_competition_shadow_library.py)、[build_tva_context_role_gated_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_role_gated_shadow_library.py) 和 [eval_hopper_tva_context_competition_shadow_200k.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py)。当前输出均为只读审计或 eval-only 产物，不进入闭环训练。

role/effect-gated 版本将第一版影子库的 active 数从 `51` 恢复为 `61`，只改变 `7` 条规则 confidence，并在 eval-only 中取得三种子均值 `126.738`，接近原规则库 `127.866`，明显高于第一版 TVA 影子库 `120.461`。该结果支持“TVA 上下文竞争必须与真实 rule-effect 和规则角色联合门控”的修订方向，但尚未证明性能提升。下一轮规则维护修订应继续增加规则角色区分：恢复型规则不能仅因终端 TVA 价值较低被降权；推进型或高回报上下文规则才适合使用 winner-take-most 的同上下文动作族竞争。

## 决策 65

**决策**  
TVA context role-gated shadow 即使通过短窗口后果门控，也暂不进入 live confidence update 或正式在线重训练。

**背景**  
为进一步验证 TVA 上下文竞争的规则价值，已对 `cycle040 / 200k` 的 role/effect-gated shadow 执行通用短窗口后果审计。该审计在 `7` 条 confidence 改动中只支持 `2` 条，并生成 consequence-gated shadow 规则库。随后在同一 `200k` 三种子 checkpoint 上执行 `10 episode` eval-only：consequence-gated shadow 的三种子均值为 `124.106`，低于原规则库 `127.866`、role-gated shadow `126.738` 和 rule-off `125.407`。补充 rule-effect 审计显示其 selected-gate 命中比例为 `0.6203`，与原规则库和 role-gated shadow 基本相同；补充姿态审计显示 `near_good_frac=0.2203`、`structural_success_frac=0.0655`，均低于原规则库。

**选项**  
1. 将 consequence-gated shadow 作为下一轮在线训练规则库。  
2. 放弃 TVA 上下文竞争和短窗口后果门控，回到原 rule-only confidence。  
3. 保留该链条为 shadow-only 规则价值筛查层，但不进入 live update；后续优先加强跨周期后果验证和同上下文动作族归并。  

**结论**  
采用选项 3。

**影响**  
当前证据说明，TVA 上下文竞争、规则角色门控和短窗口后果门控可以形成可审计的筛查链，但在该 checkpoint 上没有证明可改善行为。负结果不是由于规则未触发，而是保留改动的行为杠杆不足，且短窗口后果支持未转化为长期姿态和回报提升。因此，该机制继续作为只读或 eval-only 产物保留。后续若要进入在线规则维护，必须先满足更强条件：跨周期后果方向稳定、对相同上下文冲突动作族给出一致归并、并在 eval-only 行为审计中至少不劣于原规则库。

## 决策 66

**决策**  
新增跨周期后果稳定性汇总和同上下文动作族归并工具，但二者当前均限定为 shadow/eval-only；只有产生 active 规则变动且通过后续行为审计时，才考虑正式评估或上线。

**背景**  
单周期 consequence-gated shadow 的负结果表明，短窗口后果支持不足以单独证明规则价值。现有规则库仍存在同一状态上下文下多个相反动作族并存的问题，但直接按 TVA 终端价值压缩规则会误伤恢复型规则。因此，需要一个更保守的中间层：一方面要求规则后果在多个周期中稳定，另一方面只在同一上下文、同一关节、相反动作族之间做局部归并。

**选项**  
1. 直接放宽 role-gated / consequence-gated shadow，使更多规则发生变化。  
2. 启动完整 eval 或重训，观察现有 consequence-gated shadow 的长期效果。  
3. 先实现跨周期后果汇总和同上下文动作族归并，只生成 shadow 产物，并用现有 `cycle040` 数据做 smoke。  

**结论**  
采用选项 3。

**影响**  
新增 [summarize_rule_shadow_cross_cycle_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/summarize_rule_shadow_cross_cycle_consequences.py) 和 [build_tva_context_consolidated_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_consolidated_shadow_library.py)。前者在当前只有一个输入 audit 时报告 `stable_supported_rules=0`，符合“至少两个周期支持才算稳定”的设定。后者在 `cycle040` 上发现 `13` 个相反动作族候选组，但只有 `1` 个组达到归并资格，仅将一条 observing 规则 `unsupported_high_torque|omega_pos|vx_mid|support_none|prone->prefer:ankle_pos` 从 `0.481741` 降至 `0.240870`，active 规则数保持 `61`。该结果说明新工具没有大规模改写规则库，适合作为下一阶段规则价值审计层继续积累证据。

## 决策 67

**决策**  
新增“教师核心规则初始化”作为独立实验分支，允许从已确认标准 Hop 姿态的高质量轨迹中提取少量高可信核心规则，但必须与纯在线自学习规则库分开命名、记录和比较。

**背景**  
`300k` 规则审计显示，现有自动规则库已经高频命中，并与高分轨迹存在强相关，但同一状态上下文中仍存在多个动作族并存甚至相反方向冲突的问题。进一步检查发现，当前高分轨迹尚未稳定形成标准 Hop 姿态，而此前 `hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 的 `seed1` 在 `175k/200k` 已产生接近或超过 `500` 分、姿态明显更标准的轨迹。这些轨迹可作为结构化显式世界模型的高质量经验来源。

**选项**  
1. 继续只依赖失败归因和在线 replay 自学习规则，不使用任何高质量教师轨迹。  
2. 直接模仿高分策略网络或将高分 checkpoint 作为训练起点。  
3. 仅从高质量轨迹中提取少量可解释核心规则，作为独立初始规则库，并保持后续 SD/TVA/规则闭环自动运行。  

**结论**  
采用选项 3。

**影响**  
新增 [build_hopper_teacher_core_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_teacher_core_rules.py)，并生成两个初始规则库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_hop_only_confidence.json
```

宽核心库包含 `24` 条规则，在教师轨迹上的 selected-gate 命中率为 `0.6190`，方向一致比例约 `0.97`；`hop_only` 库包含 `12` 条规则，在教师轨迹上的 selected-gate 命中率为 `0.4596`，但在当前 `300k` eval 中几乎不触发。该结果说明宽核心库更适合作为早期可触发初始化，`hop_only` 库更适合作为标准姿态锚点。后续实验应以 `teacher_core_init` 明确标记该分支，不应将其解释为纯从零自学习规则发现。该设计不改变方法论主线：教师轨迹只提供初始核心规则，之后规则强化、削弱、冲突归并和 TVA 审计仍应在线自动完成。

## 决策 68

**决策**  
能力入口效果空间方向暂不进入 online prior 或 shadow prior；下一阶段数学对象从全局效果方向转为局部反事实遗憾。

**背景**  
能力入口追踪确认 seed1 较早进入高质量 Hop 能力区间，但入口窗口中的单动作族候选不能直接解释成功。随后设计了 `context_key -> prefer_effect_direction` 的效果空间规则，以避免重新引入难以控制的动作序列式规则。该方向在组内具备解释力，但跨周期迁移不足。最小稳定审计与判别式方向审计均未产生 stable tier。

进一步审计发现，原始 `delta_z(s,a)` 混合了动作后果、状态基线和周期尺度差异。因此新增局部动作效果优势空间：

```text
A_z(s, a_actual) = delta_z(s, a_actual)
                 - mean_{a' in probe_bank} delta_z(s, a')
```

全量审计使用 `474` 条去重窗口，跳过数为 `0`。该方法改善了 `high_entry` 的部分指标，例如 `ridge_lda / high_entry` 的 transfer 正 margin 比例为 `1.000`、平均 AUC 为 `0.576`，但仍未满足全部 transfer 的 AUC 均不低于 `0.60` 的稳定标准；`low_entry` 仍明显不稳定，`ridge_lda` 平均 AUC 为 `0.482`，平均方向余弦为 `-0.129`。

**选项**  
1. 放宽稳定阈值，将局部优势方向作为 shadow prior。  
2. 继续增加更复杂的分类器或非线性方向模型。  
3. 保留效果空间为诊断层，停止将入口能力压缩为单一线性方向；转向局部反事实遗憾和价值势函数。  

**结论**  
采用选项 3。

**影响**  
现有效果空间、判别式方向和局部优势空间结果均仅作为离线诊断保存，不写入正式规则库，也不接入在线 planner。下一步规则价值判断应优先使用 SD 推演与 TVA 价值势函数构造：

```text
U_h(s, a) = Phi(SD_h(s, a))
Regret_h(s, a) = max_{a' in probe_bank} U_h(s, a') - U_h(s, a)
Adv_h(s, a) = U_h(s, a) - mean_{a' in probe_bank} U_h(s, a')
```

其中 `Phi` 应由 TVA 或其他可迁移价值势函数提供，避免 Hopper 姿态硬编码。规则只有在同一结构化上下文中表现为低遗憾、高优势、低最优动作熵，并通过跨周期后果稳定性检查时，才进入候选规则库。该决策保持规则维护结构简洁，把复杂性限制在离线数学审计层。

## 决策 69

**决策**  
`Phi=TVA` 的局部反事实遗憾当前保留为离线诊断，不进入规则生成或在线 prior。

**背景**  
为测试局部反事实遗憾路线，新增 `Phi/Regret` 势函数审计。该审计以 TVA 作为 `Phi`，并通过同周期 SD 推演定义 `U_h(s,a)`、`Regret_h(s,a)` 和 `Adv_h(s,a)`。该方法意图避免继续学习不稳定的全局效果方向，而改为在同一状态候选动作集合中比较局部机会成本。

全量审计使用 `474` 条去重能力入口窗口，跳过数为 `0`。第一层结果显示，`Phi(s)` 对 episode 级好坏和低阈值入口有排序信号：全体窗口中 `Phi(s)` 与 episode reward 的 Spearman 为 `0.607`，与未来前向位移的 Spearman 为 `0.376`；`low_entry` 中与未来前向位移的 Spearman 为 `0.419`。但 `high_entry` 中 `Phi(s)` 与短窗口未来奖励、前向位移和前向速度的 Spearman 接近或低于 `0`。

第二层和第三层未通过。实际动作优势 AUC 在 `h=1 / 3 / 5` 分别为 `0.502 / 0.481 / 0.469`，低遗憾 AUC 分别为 `0.481 / 0.460 / 0.434`。上下文聚合没有产生 stable-like candidate。局部较强候选虽然存在，例如 `h=3` 下 `omega_neg|vx_mid|support_none|falling|support_risk -> hip_pos` 的 best-family rate 为 `0.889`、entropy 为 `0.388`，但跨周期最小 best rate 为 `0.0`，且该动作族在真实观察动作中几乎不出现。

**选项**  
1. 将局部强候选放宽为 shadow prior，用后续训练检验。  
2. 继续保留 `Phi/Regret` 为只读审计，先诊断 `high_entry` 校准和上下文混合问题。  
3. 放弃 TVA 作为 `Phi`，回到旧规则 confidence 机制。  

**结论**  
采用选项 2。

**影响**  
当前 `Phi=TVA` 证明了价值势函数路线有诊断价值，但尚未证明可以生成规则。后续规则生成仍不使用该信号。若继续推进，应先满足两项前置条件：`Phi` 在目标上下文或分层入口中具备稳定后果排序；`Regret_h / Adv_h` 在同一结构化上下文中产生跨周期低熵动作族，并通过真实后果审计。该决策保持框架简洁，避免在证据不足时把 TVA 反事实 winner 直接转化为在线规则。

## 决策 70

**决策**  
TVA 效果空间的参数训练和核 SVM 当前均不进入规则筛选流程；该方向暂作为负结果记录。

**背景**  
为避免继续增加规则结构复杂度，测试了是否可以将复杂性限制在离线数学筛选器中。新增审计脚本以 `Phi=TVA` 构造 `z_phi(s,a)`，包括 `U_h`、`Adv_h`、`-Regret_h` 和 `DeltaPhi_h`，并比较 `logistic_l2`、`linear_svm` 与 `rbf_svm`。其中 RBF SVM 只用于判断 TVA 效果空间是否存在非线性可分结构，不作为在线规则机制。

全量审计使用 `474` 条入口窗口，跳过数为 `0`。leave-one-cycle 结果显示，三类模型均未超过随机水平。global 平均 AUC 为 `0.322 / 0.320 / 0.347`；`low_entry` 平均 AUC 为 `0.284 / 0.283 / 0.376`；`high_entry` 平均 AUC 为 `0.332 / 0.305 / 0.304`。没有模型产生 stable-like context。RBF SVM 相比线性模型没有实质改善。

**选项**  
1. 放宽候选门槛，将局部 top family rate 较高的上下文作为 shadow prior。  
2. 调整 SVM 超参数或继续尝试更复杂核方法。  
3. 将本轮作为负结果收束，不上线规则；后续回到更基础的标签与上下文定义审查。  

**结论**  
采用选项 3。

**影响**  
当前证据不支持“TVA 效果空间中存在可由简单参数训练或核方法稳定抽取的规则结构”。因此不生成新规则、不接入在线 prior、不启动重训。该负结果说明，问题更可能来自能力入口标签混合、上下文划分过粗或 TVA 势函数与短窗口动作价值不一致，而不是缺少一个更强分类器。后续若继续推进，应优先保持框架简洁，重新审查目标标签和上下文，而不是堆叠更复杂的模型。

## 决策 71

**决策**  
能力入口规则提取暂不继续增加模型复杂度；下一步优先修订入口标签口径和上下文定义。

**背景**  
TVA 效果空间、局部遗憾、线性参数训练和 RBF SVM 均未产生稳定规则候选。为判断失败原因是否来自基础数据口径，新增能力入口标签与上下文基础审查。审查不使用 SD/TVA 反事实，也不训练新模型，只分析现有 `entry_candidate_segments.json`。

结果显示，原始 `568` 条窗口中，唯一窗口标识为 `413` 个，重复率为 `0.273`；`29` 个窗口存在角色冲突，即同一窗口在不同阈值事件中可同时作为 `entry_success` 和 `matched_failure`。`entry_success` 中 `42.0%` 来自入口前一周期，真正入口当周期只占 `24.8%`。该设计适合追踪能力形成过程，但会稀释规则训练标签。

标签消融显示，严格化后信号增强：原始 future reward AUC 为 `0.663`，只保留入口当周期后为 `0.727`，入口当周期且去除角色冲突窗口后为 `0.755`；future rootx AUC 从 `0.753` 提升到 `0.812`。说明入口标签并非无效，而是当前规则学习口径过宽。

上下文审查显示，当前 `context_key` 过粗。样本数不少于 `12` 的原始上下文中，`16/16` 均为 ambiguous；使用 `refined_phase` 后仍有 `13/15` ambiguous。主要问题是同一上下文跨多个周期和阈值，角色熵与动作族熵高，dominant action family 的跨周期最小命中率经常为 `0`。

**选项**  
1. 在当前标签和上下文上继续尝试更复杂的模型或核方法。  
2. 暂停模型复杂化，先修订标签与上下文，再重跑轻量审计。  
3. 放弃能力入口规则提取，回到旧规则机制。  

**结论**  
采用选项 2。

**影响**  
当前能力入口数据保留为诊断资料，不直接生成规则。后续若继续使用该数据，应采用更严格的规则学习口径：优先使用入口当周期或入口后周期，去除角色冲突重复窗口，避免同一窗口跨阈值重复加权；上下文定义需要能区分当前混在一起的局部控制机制。只有在基础标签和上下文审查通过后，才重新运行 TVA 效果空间或轻量线性筛选审计。

## 决策 72

**决策**  
严格入口标签可以作为后续能力入口审计的默认候选口径，但它不能单独恢复规则提取；下一步重点转向上下文定义。

**背景**  
根据基础审查，已派生 `no_pre_entry` 与 `entry_cycle_only` 两版严格入口数据。两者均移除 success/failure 角色冲突窗口，并对同一窗口跨阈值重复只保留一个最高阈值实例。`no_pre_entry` 保留 `269` 条窗口，`entry_cycle_only` 保留 `135` 条窗口。

严格标签改善了真实后果可分性。原始口径的 future reward AUC 为 `0.663`，future rootx AUC 为 `0.753`；`no_pre_entry` 分别提升到 `0.714 / 0.799`，`entry_cycle_only` 分别为 `0.712 / 0.800`。这说明入口标签修订方向正确。

但是，严格标签没有使数学规则审计通过。TVA 效果空间 SVM 在两版严格数据上仍无稳定候选：`no_pre_entry` 的 global AUC 约为 `0.323-0.359`，`entry_cycle_only` 约为 `0.235-0.397`，stable context 均为 `0`。局部动作效果优势空间有所改善，例如 `no_pre_entry` 下 `ridge_lda / high_entry` transfer AUC 达到 `0.601`，但仍未通过 stable tier。

**选项**  
1. 基于严格标签继续调 SVM 或核方法超参数。  
2. 将严格标签作为必要改进保留，但承认其不足，转向上下文定义修订。  
3. 回到原始宽入口标签，因为样本更多。  

**结论**  
采用选项 2。

**影响**  
后续能力入口规则审计应默认使用严格标签或同时报告严格标签结果；原始宽标签只作为能力形成过程追踪资料。由于严格标签仍不能生成稳定规则，下一步不再优先尝试新分类器，而应审查并修订 `context_key`，使其能区分当前混合在一起的局部控制机制。只有上下文纯度改善后，才重新运行轻量的局部优势或线性筛选审计。

## 决策 73

**决策**  
严格入口标签下的上下文变体审计保留为下一步规则诊断入口，但当前候选不进入在线规则库，也不启动重训。

**背景**  
严格标签重跑后，真实后果可分性增强，但 TVA 效果空间 SVM 和局部动作优势空间仍未产生稳定规则。为避免继续增加模型复杂度，新增上下文变体审计，只测试不同上下文定义是否能恢复 `context_key -> prefer_action_family` 形式的简单候选。

审计显示，`no_pre_entry` 口径下出现一条主要候选：

```text
low_entry::omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward
  -> prefer waist_neg
```

该候选在统计层面较强：成功侧 `waist_neg` 命中率为 `1.000`，失败侧为 `0.000`，动作族 AUC 为 `1.000`，后续奖励差为 `+6.818`，后续 root-x 差为 `+0.364`，成功样本跨 `cycle007/017/018`。但 SD/TVA 反事实支持不足：成功侧 `waist_neg` 的 SD top-3 为 `2/6`，TVA top-3 为 `4/6`；失败侧 TVA 明确不支持 `waist_neg`，但 SD 仍有 `3/5` 将其放入 top-3。

后续分层检查显示，该“不足”不能被简单解释为候选无效。早期 `cycle007` 的成功样本中，`waist_neg` 的 SD/TVA top-3 支持均为 `0/2`；但在 `cycle017/018` 的成功样本中，TVA top-3 支持为 `4/4`，`cycle018` 中 SD 与 TVA 均为 `2/2` best。这说明早期 SD/TVA 未充分训练会削弱反事实审计的可信度。

**选项**  
1. 将该候选作为 shadow prior，依靠后续训练检验。  
2. 继续增加上下文特征或分类器复杂度，争取产生更多候选。  
3. 保留该候选为离线诊断对象，先分析 SD/TVA 分歧和真实后果稳定性，不进入在线规则库。  

**结论**  
采用选项 3。

**影响**  
当前证据说明，入口标签修订和上下文修订确实能找回少量候选规则形态，但尚未证明这些候选具有足够可靠的模型支持。后续规则提取应默认使用 `no_pre_entry` 严格标签作为主要审计口径，同时把候选上下文的 SD/TVA 分歧作为下一步诊断对象。反事实证据需要按模型成熟度解释：早期不支持只作为弱反证，后期成熟模型的一致支持才可作为更强证据。`entry_cycle_only` 可作为更严格的对照口径，但当前样本量不足，不宜作为唯一规则学习口径。

## 决策 74

**决策**  
效果空间规则可以采用与动作规则一致的 shadow confidence 维护框架，但当前只作为离线审计，不进入在线规则库。

**背景**  
动作规则已经采用 confidence 强化、削弱和冲突处理机制。为保持框架一致性，并允许效果空间规则随训练深入自然优化，新增效果空间 shadow confidence 审计。规则形式保持简洁：

```text
context_key -> prefer_effect_direction
```

该规则不直接绑定某个动作族，而是在候选动作排序时偏好能让 SD/TVA 推演后果朝该效果方向移动的动作。由于早期 SD/TVA 未充分训练，confidence 更新中加入 maturity weight，使早期反事实证据权重较低，后期模型较成熟时证据权重较高。

严格 `no_pre_entry` 数据上，审计使用 `269` 条窗口、跳过数为 `0`，生成 `3` 条 shadow 效果规则。最高信心值为 `0.633`，前一轮重点候选 `low_entry::omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward` 的信心为 `0.626`。该候选的主要增强来自 `cycle017/018` 的成功窗口，而早期证据因 maturity weight 较低只产生小幅影响。

**选项**  
1. 将效果空间规则直接转为在线 prior，与动作规则共同作用。  
2. 放弃效果空间规则，继续只维护动作规则。  
3. 先建立 shadow confidence 审计，验证信心演化、模型成熟度加权和诱导动作族稳定性，再决定是否上线。  

**结论**  
采用选项 3。

**影响**  
效果空间规则被纳入统一规则维护思想，但仍处于离线 shadow 层。当前信心值温和上升，说明机制可运行；但部分候选的诱导动作族熵较高，尚不能证明其能稳定转化为在线动作选择。后续若要接入训练，应至少满足三个条件：confidence 稳定超过预设阈值，后期成熟模型证据与真实后果审计一致，效果方向诱导出的候选动作族不过度发散。

## 决策 75

**决策**  
在继续规则归纳或重训前，先校正 Hopper 高质量轨迹中的 hop 标签和正式评估摘要口径。

**背景**  
200k 三种子实验中，seed1 已出现稳定高分 hop 轨迹。对该轨迹进行标签校验时发现，正式评估 summary 中 `grounded_hop_frac` 为 `0.0`，但原始逐帧特征中 `grounded_hop >= 0.5` 的比例约为 `0.199`。原因是正式评估曾通过 `dominant_state_id == 6` 写入 `grounded_hop_gt_05`，而 SD 模型的 dominant state 分配逻辑没有实际产生状态编号 `6`。

同时，`grounded_prone` 在高质量 hop 轨迹中大量出现，说明该标签在 Hopper 中不能直接解释为失败，而应解释为前倾接地支撑。若继续使用旧标签口径，规则审计可能把正常 hop 周期误判为 prone 或 rotation risk，从而扭曲后续规则质量判断。

**选项**  
1. 不改代码，只在后续报告中人工说明旧摘要的局限。  
2. 重新设计完整姿态标签体系，再继续规则归纳。  
3. 进行最小代码修正：补全 `grounded_hop` 与 dominant state 编号，正式评估摘要改为按原始特征阈值写入，并调整审计脚本的 hop/prone 优先级。

**结论**  
采用选项 3。

**影响**  
该修正不改变已训练策略，也不改变历史轨迹本身，只改变后续标签解释、正式评估摘要和审计口径。旧 summary 中的 `grounded_hop_frac` 不再作为可靠证据；既有 episode CSV 可用原始特征重新审计。后续规则质量分析需要重新检查哪些规则原本被归入 prone 或 rotation risk，实质上可能对应正常 hop 周期。新的在线实验应在该标签口径通过审计后再启动。

## 决策 76

**决策**  
修正标签后，下一轮规则初始化不应无差别沿用全部旧规则；应优先保留 transition 规则和经方向审查的教师核心规则，对方向可疑的旧 prone/falling 规则降权或重新归因。

**背景**  
修正 hop/prone 标签后，对 `cycle039` 规则库在 `cycle040` 正式评估轨迹上的命中进行了重新审查。审计显示，`71` 条在线规则中有 `23` 条显式 `prone` 恢复规则、`27` 条 hop/air-cycle 规则、`12` 条方向可疑规则、`8` 条最终 eval 未命中。

显式 `prone` 规则并未大规模转移为 hop 规则，它们仍主要命中前倾接地恢复或支撑片段。这说明旧规则不是简单被错误标签污染，而是包含大量局部恢复规则。另一方面，`v8_transition_replay_safety` 规则数量只有 `8` 条，却贡献 `7747` 次 selected hit，平均 good-state hit rate 为 `0.688`，方向一致率为 `0.734`，transition 命中密度与回报相关为 `0.821`。这些规则更接近高分 hop 周期中的有效约束。

教师核心规则整体比闭环或 legacy 规则健康，但仍存在个别方向可疑项。因此，修正标签后的下一轮实验需要使用更谨慎的规则初始口径。

**选项**  
1. 继续无差别沿用全部旧规则库，以最大化早期规则覆盖。  
2. 丢弃旧规则库，只保留重新自动生成的规则。  
3. 保留 transition 规则和经方向审查的教师核心规则；对旧 prone/falling 中 harmful 比例偏高的规则降权、重新归因或要求 TVA/SD 后果审计通过。

**结论**  
采用选项 3。

**影响**  
下一轮重训或接续实验应先生成一个修正标签口径下的候选初始规则库，而不是直接使用 `cycle039` 全量规则库。该规则库应保留简洁性：仍采用单动作规则和 transition 安全规则，不引入复杂阶段式规则；但在初始权重和上线资格上区分 transition、高质量教师核心规则和方向可疑恢复规则。

## 决策 77

**决策**  
规则先验新增可选的信心校准随机门控；训练时可按规则信心概率启用规则，正式评估默认使用确定性期望缩放。

**背景**  
多种子训练中表现不一致，根本上与不同种子早期采集到的样本分布有关。规则虽然能共享跨种子经验，但如果命中即确定性生效，可能在弱种子上过早压制仍有价值的探索。由于规则本身来自有限回放数据和反事实归因，其信心值应同时影响先验强度和先验使用概率。

当前在线规则 prior 已经是轻量 MPPI 候选重排序，不会直接替代动作；但旧实现中，只要状态命中，规则每次都会进入候选排序。新机制在该位置增加概率门控，使规则成为“有置信度的经验建议”，而非确定命令。

**选项**  
1. 保持旧实现，只用 `effective_prior_scale` 缩放规则强度。  
2. 改为命中规则时按概率直接执行规则动作。  
3. 保持 MPPI 软排序介入方式不变，在每条命中规则进入排序前按 `p_apply(confidence)` 进行随机门控；eval 默认不用随机采样，而使用期望缩放。

**结论**  
采用选项 3。

**影响**  
该机制默认关闭，不影响旧实验复现。开启后，高信心规则仍会频繁发挥作用，但不会完全锁死探索；低信心规则影响较小，却仍可被偶尔采样以收集强化或削弱证据。后续重训应同时记录规则匹配数、实际启用数、dropout 数和平均启用概率，用于判断规则利用与探索比例是否健康。

## 决策 78

**决策**  
下一轮在线 smoke 使用修正标签初始规则库 `library_corrected_label_initial_v1.json`，而非 `cycle039` 全量规则库。

**背景**  
修正标签规则审计显示，全量旧规则库中存在方向可疑的 prone/falling 恢复规则。若直接用于新随机门控实验，将难以区分随机门控本身的效果与旧规则噪声的影响。为保持验证清晰性，构建了紧凑初始规则库，只保留非可疑 transition 安全规则和方向健康的教师核心规则。

该库共 `12` 条规则，包括 `7` 条 `v8_transition_replay_safety` 和 `5` 条 `teacher_core_standard_hop`。轻量 smoke 显示，在三种子正式 eval 抽样状态上，训练式随机门控实际启用率为 `0.680`，接近平均启用概率 `0.649`；eval 期望缩放不产生随机 dropout，符合设计。

**选项**  
1. 使用 `cycle039` 全量规则库启动新实验。  
2. 完全不使用初始规则，只依靠后续自学习规则。  
3. 使用修正标签初始规则库启动短 smoke，再视规则命中和训练信号决定是否扩展到 100k/200k。

**结论**  
采用选项 3。

**影响**  
下一步应先运行短在线 smoke，而不是直接启动长周期重训。观察重点为：规则实际启用率是否接近设定概率、三种子是否都能获得规则覆盖、早期训练是否未被规则过度锁定，以及规则效果审计是否继续支持这些保留规则。

## 决策 79

**决策**  
修正标签后的正式从零训练应使用 cold-start 初始规则库，而不是直接使用紧凑审计库 `library_corrected_label_initial_v1.json`。

**背景**  
`library_corrected_label_initial_v1.json` 的设计目标是降低规则噪声并验证随机门控机制，因此只保留非方向可疑 transition 安全规则和方向健康的教师核心规则。该库在成熟高分轨迹上较干净，但 10k online smoke 已显示其早期覆盖稀疏。将该库直接作为 200k 从零训练初始规则，会丢失高质量教师轨迹中的 hop 入口和部分恢复覆盖，使实验退化为弱规则冷启动。

修正标签本身仍是必要的。问题不在于重新生成规则库，而在于不能把成熟轨迹审计中的“是否健康命中”直接等同于“是否适合作为冷启动入口先验”。冷启动初始库需要同时满足修正标签一致性和早期探索覆盖。

**选项**  
1. 继续使用 `library_corrected_label_initial_v1.json` 作为正式初始库。  
2. 回退到未修正标签的原教师核心规则库。  
3. 在修正标签口径下重建 cold-start 初始库：保留教师核心入口覆盖，用审计结果校准信心；非可疑 transition 规则作为安全补充；明确无效动作族过滤。

**结论**  
采用选项 3。

**影响**  
后续正式重训应使用 `library_corrected_label_coldstart_initial_v2.json`。`library_corrected_label_initial_v1.json` 只作为紧凑审计库或 smoke 对照保留。新库不回退标签修正，而是将标签审计从“硬删除规则”改为“信心校准与风险记录”，以便规则在在线训练中继续接受强化、削弱或退休。

## 决策 80

**决策**  
新一轮 cold-start 三种子训练保留随机门控原阈值 `p_min=0.10, p_max=0.95, gamma=1.0`，不在同一轮中额外优化门控阈值。

**背景**  
本轮修正的核心变量是初始规则库：由紧凑审计库改为修正标签 cold-start 规则库，并新增无效 `zero` 动作族过滤。新规则库已经通过信心分层降低方向可疑规则影响，低信心规则在随机门控下自然具有较低启用概率。若同时修改随机门控阈值，将难以区分性能变化来自规则库覆盖修正，还是来自门控概率变化。

**选项**  
1. 同时提高 `p_min` 或降低 `p_max`，更主动地改变利用/探索比例。  
2. 暂时关闭随机门控，恢复确定性规则 prior。  
3. 保持既有随机门控阈值，只改变规则库与 `zero` 过滤，后续根据命中、启用、dropout 和规则效果审计再单独调整。

**结论**  
采用选项 3。

**影响**  
当前实验可解释为“修正标签 cold-start 规则库 + 既有随机门控 + TVA 主线口径”。若后续发现低信心规则仍过度干扰，可在独立实验中调整 `p_min / p_max / gamma`。

## 决策 81

**决策**  
`zero` 不再作为 action-family 规则进入在线动作先验或闭环规则晋升池。

**背景**  
`zero` 表示没有主导关节方向，而当前 `online_rule_prior_v6` 的 action-family prior 依赖 `joint_sign` 形式映射到具体动作维度。`prefer:zero` 无法被解释为可执行的关节方向建议，会导致规则统计中出现命中但不产生实际 prior 的无效规则。

**选项**  
1. 保留 `prefer:zero`，允许在线 prior 在执行时自然跳过。  
2. 将 `zero` 强行解释为所有关节动作趋近 0。  
3. 当前 action-family 规则中过滤 `zero`；若未来需要表达“整体降幅”，另建独立规则类型。

**结论**  
采用选项 3。

**影响**  
归因过程仍可在分析层面观察到低动作幅度现象，但不能把 `prefer:zero` 作为普通单动作族规则上线。在线 prior 和闭环规则更新均使用同一有效动作族集合。

## 决策 82

**决策**  
停止 `corrected_coldstart_stochgate_tva` 线，启动 teacher-core 最小改动重训线：初始规则库回到较成功线的 `24` 条 teacher-core 规则，保留当前标签修订、`zero` 过滤、TVA 主线口径和随机门控。

**背景**  
`corrected_coldstart_stochgate_tva` 线在 `75k` 的正式评估仍显著低于 `teacher_core_init` 线。进一步对照显示，最新初始规则库并非只引入标签修订和随机门控，还对 teacher-core 入口规则进行了 corrected-label confidence 后处理：若干 falling/prone 入口规则被压到 `0.35` 或 `0.65`，再经随机门控后期望有效强度降至约 `0.14` 或 `0.42`。此外，该库还预置了 `7` 条 transition 规则。上述处理没有被独立验证为优于上一条较成功配置，因而不符合在成功方案上进行最小扰动改进的原则。

**选项**  
1. 继续运行 `corrected_coldstart_stochgate_tva` 到 `200k`，等待后续周期自行恢复。  
2. 在当前有问题的规则库上继续调整随机门控阈值或新增补丁。  
3. 回到较成功线的 teacher-core 初始规则强度，只保留已有明确依据的修订：标签解释修正、`zero`/invalid action 过滤、TVA 接口门控和在线规则维护；随机门控作为唯一新增启用机制保留，以便单独观察其影响。  

**结论**  
采用选项 3。

**影响**  
新增脚本 [launch_hopper_v8formal_200k_teacher_core_stochgate_tva_minimal_seed123_20260524_r1.sh](/home/data/round/code/sd_world_model_rl/scripts/launch_hopper_v8formal_200k_teacher_core_stochgate_tva_minimal_seed123_20260524_r1.sh)。该线不再使用 `library_corrected_label_coldstart_initial_v2.json`，而直接使用 `library_teacher_core_confidence.json` 作为初始库。若新线恢复早期上升趋势，则说明主要问题来自 corrected-label confidence 后处理；若仍未恢复，则下一步应在同一 teacher-core 初始库上关闭随机门控，做更干净的因果对照。

## 决策 83

**决策**  
停止 `teacher_core_stochgate_tva_minimal` 长线，启动当前代码下的 teacher-core 无随机门控 `100k` 对照线。

**背景**  
`teacher_core_stochgate_tva_minimal` 线在 `100k` 的正式评估为 overall mean `3.517`、max `22.770`，明显低于较成功的 `teacher_core_init` 原线在 `75k` 和 `100k` 的表现。进一步检查显示，最初 `24` 条 teacher-core 规则在随机门控线中并未更严重地退休或削弱；到 `cycle020` 仍有 `22` 条 active、`2` 条 observing，平均信心为 `0.781`。因此，退化更可能来自随机门控在执行层降低了规则先验的稳定作用，而不是规则库维护本身失败。

**选项**  
1. 继续运行随机门控线到 `200k`，等待后续训练自然恢复。  
2. 直接调整随机门控概率参数，例如提高 `p_min` 或改变 `gamma`。  
3. 保持当前代码、标签修订、TVA 主线和 teacher-core 初始规则不变，仅关闭随机门控，先跑到 `100k` 做干净因果对照。

**结论**  
采用选项 3。

**影响**  
新增脚本 [launch_hopper_v8formal_100k_teacher_core_nostoch_tva_current_seed123_20260524_r1.sh](/home/data/round/code/sd_world_model_rl/scripts/launch_hopper_v8formal_100k_teacher_core_nostoch_tva_current_seed123_20260524_r1.sh)。若该线在 `50k-100k` 恢复 seed1 的早期 hop 入口，则随机门控应改为分层机制：高置信 teacher-core 核心规则保持确定性介入，新生成或低置信规则再按信心随机门控。若该线仍不恢复，则继续审查当前代码修订与原 `teacher_core_init` 运行环境之间的差异。

## 决策 84

**决策**  
对 `teacher_core_standard_hop` 初始规则加入冲突保护，并启动 `50k` 三种子短程重训验证。

**背景**  
当前代码无随机门控对照线没有缺少 `24` 条 teacher-core 初始规则，但在 `cycle001` 后，关键入口规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_neg
```

被冲突机制从 `confidence_active, confidence=1.0` 降为 `confidence_observing, confidence=0.5`，原因是同状态同关节出现了新生成的反向 `prefer:knee_pos` 候选。原较成功线未发生该降权。该现象说明：早期 5k 噪声分布中的自学习候选能够过早削弱教师入口规则，进而妨碍策略进入可扩张的高质量片段分布。

**选项**  
1. 不保护教师规则，继续让所有规则按同一冲突仲裁机制竞争。  
2. 完全冻结教师规则，使其永不被削弱。  
3. 只在冲突仲裁中保护教师规则；若反向候选具有足够多命中、显著更高信心和显著更低有害率，仍允许其覆盖教师规则。

**结论**  
采用选项 3。

**影响**  
`apply_rule_effect_confidence.py` 新增教师核心冲突保护。默认保护 `rule_origin == teacher_core_standard_hop` 的规则；反向候选必须满足 `selected_gate_hits >= 1000`、信心优势不低于 `0.25`、有害率优势不低于 `0.20`，才可在冲突仲裁中覆盖教师规则。`closed_loop_rule_update.py` 同步保留教师来源元数据，避免同 ID 周期证据覆盖后丢失教师身份。

新增短程重训脚本：

```text
scripts/launch_hopper_v8formal_50k_teacher_core_protected_tva_current_seed123_20260525_r1.sh
```

该实验保持 teacher-core 初始规则、无随机门控、TVA terminal prior、v8 transition 规则和 rule-effect confidence 等主线配置，只改变教师冲突保护。预算为 `50k`，规则更新周期仍为 `5k`，正式评估在 `25k` 和 `50k` 进行。第一判断标准不是短期回报绝对值，而是关键教师入口规则是否不再在 `cycle001` 被冲突降权，以及早期是否更容易形成高质量 transition 片段。

## 决策 85

**决策**  
将 `grounded_hop` 从默认在线 phase 匹配中解耦，保留为诊断标签；若需要让其覆盖 `prone`，必须显式开启 `rule_prior_enable_grounded_hop_phase`。

**背景**  
早中期训练轨迹复核显示，`grounded_hop` 在高质量轨迹审计中有价值，但作为单帧在线 phase 标签过于宽松。历史教师核心初始化线中，`cycle020` 的 `grounded_hop` 触发帧几乎全部来自旧口径 `prone`，且多数伴随低全足支撑和后续短窗旋转风险。这说明当前静态标签无法稳定区分标准起跳前瞬间与恢复失败中的不稳定前冲。

**选项**  
1. 保持当前逻辑，让 `grounded_hop` 默认在在线 phase 中覆盖 `prone`。  
2. 删除 `grounded_hop` 标签，完全回到旧标签体系。  
3. 保留 `grounded_hop` 作为审计和后续转移上下文特征，但默认不参与在线 phase 覆盖；需要实验性启用时通过显式开关打开。  

**结论**  
采用选项 3。

**影响**  
`online_rule_prior_v6.py` 新增 `enable_grounded_hop_phase` 参数，默认关闭。`run_hopper_v7_multiseed_shared_rules.py` 和 `run_hopper_v6_5k_rule_prior_smoke.sh` 新增对应透传参数。默认训练语义恢复为 `falling -> rear -> prone -> upright -> rising -> other` 的在线 phase 解释；`grounded_hop` 仍可用于轨迹审计、报告导出和后续“前序状态 -> 当前状态 -> 后续结果”的转移上下文规则提取。

## 决策 86

**决策**  
新增离线转移上下文规则候选提取器，但候选规则默认不得直接上线。

**背景**  
`grounded_hop` 标签复核表明，单帧静态状态不足以区分标准 Hop 起跳前瞬间与不稳定前冲。规则提取需要引入前序状态和后续结果，但若直接把复杂阶段序列规则接入在线 prior，会增加不可控性。因此，需要先以离线工具形式实现最小可审查版本。

**选项**  
1. 继续沿用单帧 phase 规则，不引入转移上下文。  
2. 直接实现新的在线阶段式规则类型。  
3. 先实现离线转移上下文候选提取器，输出观察性候选；候选经 SD/TVA 反事实验证后，才允许进入规则库。  

**结论**  
采用选项 3。

**影响**  
新增脚本：

```text
experiments/sd_retrospective_rule_induction/extract_hopper_transition_context_rules.py
```

脚本构造 `past dominant phase -> current candidate phase -> future outcome` 记录，并按 `prev_phase/current_phase/omega_bin/vx_bin/support_bin/risk_bin` 聚合动作族后果。输出状态为 `transition_context_candidate`，明确表示观察性候选。该实现保持最终规则仍为单动作族形式，但将前件扩展为转移上下文，从而为后续 SD/TVA 反事实验证提供候选池。

## 决策 87

**决策**  
将 `8` 条通过低分对照和跨周期稳定性审计的 `SD/TVA` 反事实规则作为新一轮三种子在线训练的初始化教师规则，并继续启用教师规则保护。

**背景**  
观察性高分轨迹候选在联合反事实验证中暴露出动作共现不等于决策价值的问题。新的生成方式改为由 `SD` 短视界推演收益与 `TVA` 终端补偿共同选择动作族，并通过低分对照与跨周期稳定性筛选出 `retain: 8` 的规则子集。该子集相较旧教师核心规则更直接体现结构化世界模型的诊断优势，适合作为新一轮方法论验证的初始化规则库。

**选项**  
1. 继续使用旧 `teacher_core_standard_hop` 规则库，保持历史较优实验条件。  
2. 将 `13` 条反事实生成候选全部上线，提高覆盖面。  
3. 只上线 `retain: 8` 子集，并保留教师规则保护与后续信心更新。  

**结论**  
采用选项 3。

**影响**  
新增规则库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_counterfactual_teacher_retain8_confidence.json
```

新增训练线：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1.sh
```

本轮实验从零开始至 `200k`，三种子共享规则库，`SD` 拟合、`TVA` 训练、反事实归因和规则更新仍保持 `5k` 周期，正式 checkpoint 与完整 eval 仍保持 `25k` 周期。`TVA terminal prior` 继续采用接口审计门控，第一周期无有效 `SD/TVA` checkpoint 时保持关闭。

## 决策 88

**决策**  
教师规则不再因低命中、零命中或审计缺失而衰减；仅允许由强反例削弱。

**背景**  
当前反事实教师规则由高质量轨迹中的稳定上下文提取，并经过 `SD/TVA` 联合反事实、低分对照和跨周期稳定性审计。在线训练早期或种子分布不平衡时，教师上下文可能很少出现。若将低命中解释为负证据，会把“尚未进入高质量上下文”误判为“教师规则无效”，从而削弱本应作为入口引导的教师先验。

**选项**  
1. 维持原机制，教师规则与普通规则一样因零命中、低命中和审计缺失而衰减。  
2. 完全冻结教师规则，使其永远保持 `confidence_active`。  
3. 区分证据不足与强反例：教师规则在低证据下保持信心；一致性证据可增强；只有达到强有害阈值的反例才能削弱至 active 阈值以下。  

**结论**  
采用选项 3。

**影响**  
`apply_rule_effect_confidence.py` 新增教师低证据保护。受 `protected_teacher_origins` 覆盖的教师规则在 `stats is None`、`hits == 0` 或 `hits < min_effect_hits` 时保持原信心，不再应用 `absence_decay` 或 `low_hit_decay`。当命中足够但未达到强反例标准时，教师规则也不因普通有害率波动被削弱；若命中数不少于 `hard_harmful_min_hits` 且有害率不低于 `hard_harmful_threshold`，则仍可被降至 active 阈值以下。

## 决策 89

**决策**  
记录当前规则语言多方言问题，未来需统一规则上下文表达；统一到何种机制暂不决定。

**背景**  
当前在线 prior 与效果审计使用同一匹配入口，但不同规则来源采用不同规则 key 语义。反事实教师规则采用 `prev=...|cur=...` 的转移上下文；普通在线归纳规则多采用 `event|omega|vx|support|phase` 的单帧子状态；v8 transition 规则采用 `transition|...` 结构；诊断规则另有 `toe_recovery_*` 格式。该设计使多类规则可以共存，但也导致教师规则与普通自学习规则在上下文表达能力上不一致。

**选项**  
1. 统一到 `prev->cur` 转移上下文。  
2. 统一到 v8 transition replay safety 规则。  
3. 保持多方言共存。  
4. 暂不指定具体统一目标，先将问题记录为后续研究议题。  

**结论**  
采用选项 4。

**影响**  
短期内不改动当前运行实验，也不立即重构规则归纳管线。后续需要比较不同上下文表达在规则覆盖、可解释性、反事实可验证性、在线可控性和跨任务泛化性上的差异，再决定统一方向。

## 决策 90

**决策**  
未来统一规则语言时，倾向于将 `prev->cur` 作为抽象规则状态的默认组成。

**背景**  
从严格 MDP 角度看，完整状态应具有无前效性；若 `s_t` 已经是充分状态，则历史路径不应改变最优决策。但当前规则层使用的是粗粒度抽象标签，而不是完整仿真状态或充分潜在状态。相同当前标签可能对应不同动态来源，例如空中落地、失败恢复或起跳前准备。此时 `prev->cur` 并不是引入额外历史因果，而是在抽象状态不充分时补充阶段语境。

**选项**  
1. 只在教师规则中保留 `prev->cur`，普通在线规则继续使用单帧状态。  
2. 将 `prev->cur` 视为可选补丁，仅在发现单帧状态不足时使用。  
3. 将 `prev->cur` 作为统一规则语言的默认字段，允许 `prev=unknown`，但规则前件始终具备转移语境表达能力。  

**结论**  
采用选项 3 作为后续设计方向。

**影响**  
该决策暂不改变当前运行实验。后续重构规则归纳管线时，教师规则和在线自学习规则应统一为类似：

```text
prev_abstract_state -> current_abstract_state + context_features -> action_prior
```

的规则前件形式。该形式保留抽象状态的可解释性，同时以低成本增强阶段识别能力，更适合复杂任务中的可泛化规则学习。

## 决策 91

**决策**  
终止当前 `150k` 已验证线，并以教师低证据保护和统一 `prev->cur` 规则语言启动新一轮 `200k` 三种子重训。

**背景**  
当前 `hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1` 在 `150k` 正式评估达到三种子均值 `186.85`，超过原始 TD-MPC2 `200k` 均值 `166.63`。尽管 seed2 仍未跟上，但 seed1 与 seed3 已经稳定进入较高分区间，说明教师规则、闭环规则维护与 TVA 门控先验的组合具有可验证效果。继续当前线至 `200k` 的边际信息低于按新机制从头重训。

**选项**  
1. 继续当前线至 `200k`，保持所有历史削弱和旧普通规则语言。  
2. 从头重训，并同时加入随机门控或模式定理思想。  
3. 从头重训，只加入已明确的两项机制修订：教师规则低证据保护、普通在线规则统一 `prev->cur`。  

**结论**  
采用选项 3。

**影响**  
新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_seed123_20260527_r1.sh
```

本轮不加入规则触发时保留探索空间，也不加入模式定理思想，以保证每项修订可归因。实验协议继续保持 `5k` 闭环更新、`25k` 正式 eval/checkpoint、三种子共享规则、TVA terminal prior 门控。

## 决策 92

**决策**  
在当前 `prev->cur` 保护线确认 `100k` 有效信号后，终止该线后续训练，并以保守信心随机门控启动新一轮 `200k` 三种子重训。

**背景**  
`hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_seed123_20260527_r1` 在 `100k` 正式评估达到三种子均值 `125.962`，高于本地同口径历史 `100k` 主要对照线；其中 seed2 达到 `304.221`，seed3 达到 `65.766`，seed1 仍未打开。该结果说明教师规则保护、统一 `prev->cur` 在线规则归纳和 TVA terminal prior 门控的组合已形成有效信号。下一步研究重点转为在保持规则先验价值的同时，为不同种子保留适当探索空间，减少早期分布锁定。

**选项**  
1. 继续当前线至 `200k`，不引入随机门控。  
2. 使用既有默认强随机门控 `p_min=0.10, p_max=0.95` 直接重训。  
3. 使用保守信心随机门控 `p_min=0.40, p_max=0.98, gamma=1.0` 重训。  

**结论**  
采用选项 3。

**影响**  
规则命中后的执行概率采用：

```text
p_apply(rule) = 0.40 + 0.58 * confidence
```

正式评估仍不采样，而使用期望缩放；训练阶段通过规则 dropout 保留探索空间。该修订只改变规则触发后的在线排序介入频率，不改变规则库生成、信心维护、TVA terminal prior 或策略网络训练目标。

## 决策 93

**决策**  
终止 `p_min=0.40, p_max=0.98` 随机门控线，并以近确定性随机门控重新启动 `200k` 三种子训练。

**背景**  
`hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate040098_seed123_20260527_r1` 在 `75k` 正式评估中得到：

```text
seed1 mean = 0.000, max = 0.000
seed2 mean = 3.954, max = 19.297
seed3 mean = 0.000, max = 0.001
overall mean = 1.318, max = 19.297
```

同口径上一轮无随机门控的 `prevcur_protect` 线在 `75k` 达到 `overall mean = 73.011, max = 250.571`。由于教师规则仍在线、规则库持续扩展、规则命中比例正常上升，当前弱表现更可能来自较宽随机门控削弱了教师规则在早期训练中的连续引导，而不是规则库失效。

**选项**  
1. 完全取消随机门控，回到无 dropout 的 `prevcur_protect` 线。  
2. 保留当前 `p_min=0.40, p_max=0.98` 继续至更长周期。  
3. 将探索窗口限缩到极小范围，采用 `p_min=0.95, p_max=0.995, gamma=1.0` 重训。  

**结论**  
采用选项 3。

**影响**  
新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1.sh
```

该实验保留三种子、`5k` 闭环更新、`25k` 正式 `10ep` eval/checkpoint、共享规则库、教师规则保护、统一 `prev->cur` 在线规则归纳、TVA terminal prior 门控和 TVA 审计。唯一关键变化是将规则触发后的训练期随机 dropout 限缩为近确定性执行，以检验极小探索窗口是否能保留教师引导价值，同时避免完全锁死规则覆盖区域。

## 决策 94

**决策**  
终止近确定性随机门控实验，并修订随机门控实现，使其使用独立随机数生成器，不再消耗 TD-MPC2/MPPI 主随机数流。

**背景**  
近确定性门控线 `hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1` 在 `75k` 正式评估中仍明显弱于无随机门控有效线。排查显示教师规则未丢失、规则库持续扩展、规则命中后实际丢弃比例很低；同时代码层面发现门控采样使用全局 `torch.rand`。因此，即便 `p_apply=0.95-0.995`，门控也会改变 MPPI 采样、Gumbel 选择和最终动作探索噪声的随机序列，使该实验无法作为“极小探索窗口”机制的可靠检验。

**选项**  
1. 继续当前近确定性门控线，等待更长周期结果。  
2. 完全取消随机门控，回到无门控方案。  
3. 保留随机门控思想，但先隔离其随机数流，再重新进行可归因验证。  

**结论**  
采用选项 3，并终止当前近确定性门控线。

**影响**  
`HopperRulePriorV6` 新增独立 CPU `torch.Generator`，由 `rule_prior_stochastic_seed` 初始化；多种子 launcher 按 `base_seed + cycle * 1000 + seed` 派生训练周期内的门控种子。该修订使规则门控采样仍可复现，但不再推进 TD-MPC2/MPPI 的全局随机数流。后续随机门控实验应基于该实现重新评估，而不再使用已终止的 `stochgate0950995` 结果作为机制结论。

## 决策 95

**决策**  
在下一轮在线重训前，先独立发展 rich 初始化教师规则提取，不直接改动训练管线。

**背景**  
当前 `retain8` 教师规则库能够验证 SD/TVA 反事实教师规则的基本可行性，但其规则覆盖较窄，且当前 `300k` 紧窗口线显示早期入口路径仍具有明显随机敏感性。若直接修改在线规则维护或探索门控，容易把教师规则质量、随机路径和训练机制混在一起。更稳妥的做法是先在离线层面从高质量轨迹的成长链路中提取更丰富的入口规则，再独立审查其语义和反事实可靠性。

**选项**  
1. 继续当前 `retain8` 教师库，只通过更长训练观察是否自然改善。  
2. 立即修改在线规则维护机制，例如加入阶段规则或更复杂的探索门控。  
3. 新建离线 rich 初始化教师规则提取流程，先产出候选库和审计报告，再决定是否用于重训。  

**结论**  
采用选项 3。

**影响**  
新增脚本：

```text
experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_rich_init_library.py
```

第一版 rich 初始化库位于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
```

该库由 `30` 条规则构成，其中 `22` 条为 `confidence_active`，`8` 条为 `confidence_observing`。低分对照中为 `supported` 的规则暂不激活，只保留为观察项。该决策不改变当前在线训练；后续需先审查 rich 规则的语义覆盖、早期失败轨迹触发和与旧 `retain8` 库的冲突，再决定是否以其作为新一轮初始化教师规则库。

## 决策 96

**决策**  
终止当前 `retain8` 初始化长线，并以 rich 初始化教师规则库启动新的 `300k` 三种子重训。

**背景**  
rich 初始化教师规则库已完成独立候选生成、高质量成长链路审计、低分对照审计和早期触发审计。结果显示，该库在高质量成长链路上覆盖较密，在早期失败轨迹中只窄幅命中入口/转接状态，且相较 `retain8` 新增了 `15` 条 active 入口规则。当前 `retain8` 长线虽仍在推进，但已显示早期入口路径敏感性；继续运行与启动 rich 初始化重训相比，边际方法论信息较低。

**选项**  
1. 继续 `retain8` 长线至 `300k`，暂不验证 rich 初始化库。  
2. 在现有 `retain8` checkpoint 上接入 rich 规则继续训练。  
3. 中止 `retain8` 长线，从零启动 rich 初始化教师规则三种子 `300k` 重训。  

**结论**  
采用选项 3。

**影响**  
新增启动脚本：

```text
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1.sh
```

新实验名：

```text
hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1
```

本轮保持 `5k` 闭环更新、`25k` 标准 `10ep` eval/checkpoint、三种子共享规则库、TVA terminal prior gate、v8 transition rules、rule-effect confidence 和近确定性随机门控不变。核心变化为：

```text
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1
```

同时将通用 `retain8` launcher 中的 `TEACHER_RULES` 和 `PROTECTED_TEACHER_ORIGINS` 改为可由环境变量覆盖，以便后续同协议替换教师库时不复制整段长命令。

## 决策 97

**决策**  
取消规则触发时的探索保留窗口，并以无随机门控方式重跑 rich 初始化教师规则 `300k` 三种子实验。

**背景**  
rich 初始化教师规则结合 `0.95-0.995` 近确定性随机门控的 `300k` 实验已经完成。该线最终均值略高于公开原始 TD-MPC2 `300k`，但优势不大，且三种子方差显著更高，主要由单个强 seed 拉高。随机门控的原始目的在于缓解多种子间的不平衡；既然该目标没有实现，继续保留该机制会把规则先验效果与训练期随机 dropout 混在一起，不利于判断 rich 初始化规则和教师保护机制本身的贡献。

**选项**  
1. 继续保留 `0.95-0.995` 随机门控，并尝试更长训练或更多种子。  
2. 调整随机门控概率，例如进一步放宽或收紧探索窗口。  
3. 取消随机门控，保持其余协议不变，重跑 clean ablation。  

**结论**  
采用选项 3。

**影响**  
新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh
```

新实验名：

```text
hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1
```

本轮保持 rich 初始化教师规则库、教师规则保护、`prev->cur` 在线规则归纳、TVA terminal prior gate、v8 transition rules、rule-effect confidence、`5k` 闭环更新和 `25k` 正式评估口径不变。唯一目标变化是移除 `--enable-rule-prior-stochastic-gate` 及其概率、信心幂和随机种子参数。该结果将用于判断此前随机门控是否对 rich 初始化规则线形成干扰。

## 决策 98

**决策**  
将已完成的无随机门控 rich 初始化 `300k` 三种子实验从 `cycle060` 接续训练至 `500k`。

**背景**  
无随机门控 rich 初始化线在 `300k` 正式评估中得到 `344.610 / 290.076 / 280.835`，均值 `305.174`，标准差 `28.140`。该结果在同预算下已经高于公开原始 TD-MPC2 `300k` 均值 `259.733`，且方差更小；但三种子是否都能进入较理想姿态仍需通过更长训练观察。

**选项**  
1. 停止于 `300k`，直接整理同预算结果。  
2. 从零启动新的 `500k` 三种子训练。  
3. 从 `300k` 终点接续至 `500k`，保留现有共享规则库、SD/TVA checkpoint 和各 seed full-state checkpoint。  

**结论**  
采用选项 3。

**影响**  
新增启动脚本：

```text
scripts/launch_hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.sh
```

新实验名：

```text
hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1
```

该实验从 `cycle061_target305060` 运行至 `cycle100_target500099`，保持无随机门控、rich 初始化教师规则、教师保护、TVA terminal prior gate、v8 transition rules、rule-effect confidence、`5k` 闭环更新和 `25k` 正式评估口径不变。由于 `cycle060` 的 TVA gate 为 strong 通过，续训首个周期显式使用 `initial_tva_terminal_prior_policy=strong`，以维持连续训练语义。该 `500k` 结果用于评估延长训练后的能力上限；严格同预算结论仍以 `300k` 对照为主。

## 决策 99

**决策**  
从非灰因果无随机门控 rich 初始化线的 `400k` 正式评估点严格接续至 `500k`，不使用已落盘但未正式评估的 `405080` checkpoint。

**背景**  
`resume300k_20260603_r1` 续训线的实验名为 `500k`，但实际正式评估只到 `cycle080_target400079`；`cycle081_target405080` 仅有训练 checkpoint，没有对应正式 `eval.csv`。若直接从 `405080` 继续，会使后续结果无法被清晰地解释为从 `400k` 正式评估点接续。

**选项**  
1. 从 `405080` 最新 checkpoint 继续，减少重复训练。  
2. 从 `400079` 正式评估点继续，重新运行 `cycle081` 至 `cycle100`。  
3. 先补 `405080` 评估，再从 `405080` 继续。  

**结论**  
采用选项 2。

**影响**  
新增独立运行目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume400k_20260606_r2
```

该运行从 `cycle081_target405080` 开始，到 `cycle100_target500099` 结束。初始规则库、SD checkpoint 和 TVA checkpoint 使用 `resume300k_20260603_r1/cycle_080` 产物，三种子均从 `cycle080_target400079/models/final.full.pt` 恢复。普通 `nohup` 启动的 `r1` 未能稳定保留后台进程，未生成 checkpoint；有效运行使用 `setsid` 启动的 `r2`。
