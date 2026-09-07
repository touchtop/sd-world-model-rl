# SD 反向归因与规则归纳研究日志

## 2026-05-08

### phase-any 正式重跑的早期规则作用审查与中断

对 `hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2` 进行了早期在线观察。该实验从空规则库启动，三种子共享规则，周期长度 `5k`，每 `25k` 执行一次标准 `10ep` formal eval。`cycle005 / 25k` formal eval 已落盘，三种子均值分别为：

```text
seed1 = 0.0
seed2 = 0.17335563196102158
seed3 = 0.0
mean  = 0.05778521065367386
```

该节点不作为停止依据，因为规则 prior 从 `cycle006` 才开始真正加载。`cycle006` 三种子训练日志显示均安装 `3` 条 `v8_replay_safe_candidate` 规则，规划指标中的 `rule_prior_hit` 分别约为：

```text
seed1 = 0.04
seed2 = 0.08
seed3 = 0.12
```

因此，规则 prior 并非没有进入动作排序。但由于当前采用 `penalty_only`，`rule_prior_bonus_max` 为 `0.0`，其作用表现为对相反方向或过饱和候选的轻量惩罚，而不是正向奖励。

进一步审查 `cycle005` 与 `cycle006` 的 retrospective train traces 后确认，规则对动作方向存在可测影响。三条上线规则在 `cycle006` 中的总状态命中率约为 `3.217%`，低于 `cycle005` 的 `3.519%`，说明状态覆盖本身没有扩大；但 seed2、seed3 的对齐动作比例上升、相反动作比例下降。例如 seed3 中：

```text
rotation_risk|omega_neg|vx_mid|support_none|rear->prefer:waist_pos
  cycle005 align = 53.6%, oppose = 39.1%
  cycle006 align = 59.9%, oppose = 31.4%

unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:hip_neg
  cycle005 align = 48.5%, oppose = 37.8%
  cycle006 align = 51.9%, oppose = 31.6%

rotation_risk|omega_neg|vx_low|support_none|rear->prefer:hip_neg
  cycle005 align = 49.3%, oppose = 37.1%
  cycle006 align = 52.6%, oppose = 34.2%
```

该结果说明当前问题不是“规则没有作用”，而是“规则作用范围过窄，且仍主要集中于早期 `rear / support_none / rotation_risk` 恢复状态”。`cycle006` 总结中，普通恢复分支生成 `24` 条子状态规则、`639` 条反事实状态记录，但闭环合并结果为 `no_rule_added`；v8 transition 分支识别 `14` 个 transition gap、`34` 条 transition action records 和 `20` 条 SD transition counterfactual records，但候选规则仍为 `0`。

随后 `cycle007` 在中断命令生效前已完整跑完并完成总结。该周期最终共享规则库为 `11` 条规则，其中 `4` 条为 `v8_replay_safe_candidate`，但 transition 分支仍未产生上线规则。由于本次主控进程仍沿用旧启动参数 `max_cycle_candidates=6` 且反事实抽样仍为 `max_states_per_episode=3`，继续运行会主要重复低覆盖恢复规则，而不能有效检验已修订的规则发现设置。因此在 `cycle008` 刚启动后终止该实验。终止后确认没有该实验相关进程，GPU1-3 已释放。

本轮结论为：规则 prior 已体现局部行为作用，不能视为工程失效；但当前规则发现层仍过于稀疏，transition 规则链条没有稳定产生上线规则。下一轮正式重跑前，应先采用更宽的反事实抽样与候选保留设置，并修订高 `good_share` 候选的处理逻辑，避免大量可能有价值的广覆盖规则在早期被直接拒绝。

### widefunnel 从头重跑启动

在完成上述中断和规则发现漏斗修订后，已启动新的从头 `100k` 三种子共享规则闭环实验。新实验命名为：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1
exp_prefix = tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1
```

该实验与已中断的 `r2` 保持相同在线门槛和评估口径，但显式放宽规则发现层：

```text
max_states_per_episode = 10
max_cycle_candidates   = 12
max_online_rules_per_cycle = 2
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = v8_replay_safe_candidate
formal_eval_episodes = 10
formal_eval_freq = 25000
cycle_steps = 5000
```

启动后确认 `cycle001` 三个种子均在 GPU1-3 上运行，第一周期按预期从空规则库开始，`rule_prior_enabled=false`。本轮后续观察重点不放在 `cycle001` 回报，而放在每周期总结后候选数量、`cycle_candidate` 携带数量、`v8_replay_safe_candidate` 增长以及 transition 分支是否能形成上线规则。

## 2026-05-07

### 第八版成功延续与阶段衔接规则设计

在 `v72` 三种子共享规则闭环训练至 `200k` 后，对最终 `10ep` 评估轨迹进行复核。三种子官方评估回报分别为 seed1 `99.55`、seed2 `187.30`、seed3 `193.55`，均值约 `160.13`。仓库内 TD-MPC2 官方 CSV 的 `200k` 三种子均值为 `166.63`，因此当前闭环规则实验在均值上略低于官方 TD-MPC2，但 seed3 高于官方对应 seed。

轨迹姿态复核显示，最终评估中存在局部较理想姿态片段，例如 seed2 episode 09 frame 282 处 `pitch=-0.111`、`torso_z=0.836`、`vx=1.726`、单步 reward `1.573`。但严格的 `grounded_upright / grounded_prepared` 持续片段很少，说明当前策略尚未将局部好姿态稳定连接为完整跳跃周期。

进一步审查最终规则库 [library_closed_loop_cycle_040.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_200k_seed123_resume100k_20260507_r2/cycle_040/rules/library_closed_loop_cycle_040.json) 后确认：共有 `59` 条规则，其中 `34` 条 `replay_validated / rolling_validated` 规则参与在线先验。最终 eval 轨迹中约 `14.7%` 的动作步命中至少一条规则，但命中主要集中于 `rear`、`falling`、`vx_low`、`rotation_risk` 或 `unsupported_high_torque` 等恢复状态。前述高质量姿态片段所在的 `omega_mid / vx_high / support_none / falling / rotation_risk=0` 子状态未被当前规则覆盖。

据此形成第八版设计结论：现有失败恢复规则应保留，但必须新增“成功延续”和“阶段衔接”规则通道。新规则不再只问“坏状态下如何恢复”，还要问“好姿态为何出现、为何中断、哪些动作有助于把局部好姿态延伸为落地支撑与再起跳”。同时，为保持方法未来可迁移性，设计采用“通用引擎 + Hopper 适配器”边界，Hopper 的 `pitch/vx/support/waist/hip/knee/ankle` 等只作为当前任务适配器，而不是规则闭环方法本体。

新增设计文档：

- [hopper_v8_success_extension_rule_design.md](/home/data/round/code/sd_world_model_rl/docs/sd_retrospective_rule_induction/hopper_v8_success_extension_rule_design.md)

随后实现第八版第一步片段挖掘脚本：

- [mine_hopper_v8_success_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py)

该脚本按“通用流程 + Hopper 适配器”的形式组织，当前 Hopper 适配器使用 `torso_z`、`pitch`、`pitch_rate`、`vx`、`support_contact`、`full_foot_support`、`dense_pose_potential` 和逐步 reward 等字段识别成功候选片段，并在片段后 `25` 步窗口内标注退化类型。脚本同时读取最终 v7 规则库，检查成功片段是否已被现有恢复规则覆盖。

已执行：

```bash
python experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py --use-default-v72-200k
```

本轮从 `30` 条最终评估 trace 中提取 `356` 个成功候选片段，其中 `349` 个被标注为阶段衔接缺口，只有 `7` 个在 `25` 步窗口内未检测到退化。候选原因统计显示，`356` 个片段均含正回报，`306` 个片段含结构性好姿态帧，`34` 个片段含 `dense_pose_potential >= 0.65` 的帧。退化类型主要为 `entered_prone`、`rotation_risk` 和 `support_failure`。现有 v7 规则只覆盖 `2` 个候选片段，说明局部有分片段与现有失败恢复规则之间存在明显覆盖缺口。

输出报告：

- [hopper_v8_success_segments_200k_v72.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.md)

## 2026-05-05

### 第一部分：背景与设想

当前 Hopper 主线已经证明，系统动力学引导的世界模型（System-Dynamics-Guided World Model, SD world model）不应仅被理解为一个用于替代黑盒动力学的前向预测模块。mechanism-oriented `SD` 版本在纯环境奖励、独立在线 `SD` 监督、相位/接触/角速度机制监督、以及关闭 `TD-positive` 样本重加权的条件下，已经在 `100k` 与 `200k` 节点显著超过仓库内公开 CSV 口径的 TD-MPC2 与 DreamerV3 基线；`300k` 接续实验也已经完成落盘，三种子 `300001` 评估分别约为 `251.59`、`273.91`、`239.83`，均值约为 `255.11`。这些结果说明，显式结构态与机制监督能够改善样本效率和长期训练表现。

但现有主线仍主要把 `SD` 用于“前向状态推演”：给定当前状态与动作，预测下一步或短视界结构态演化，并通过这种预测改进 TD-MPC2 的规划、动作排序和训练稳定性。该用法虽然有效，却只利用了 `SD` 的一部分价值。系统动力学结构态本身具有可解释性：高度、速度、躯干角、角速度、足端接触、支撑相位、起跳相位、落地相位等变量并不是无语义 latent，而是可被人类研究者直接解释的任务机制。因此，`SD` 不仅可以回答“采取该动作后系统会怎样演化”，也应能够回答“这条轨迹为什么失败”“成功轨迹在哪些关键状态下做对了什么”“失败轨迹在哪些关键状态下做错了什么”。

由此形成新的研究设想：将 `SD` 从 TD-MPC2 的在线前向规划接口中暂时抽离出来，作为一个独立的经验总结与失败归因模块。TD-MPC2 仍按常规方式独立训练；每经过一个固定训练周期，例如 `25k` 或 `50k` 环境步，系统导出该周期内的训练轨迹、评估轨迹和 replay 片段。随后，离线或半离线地使用这些轨迹进一步拟合独立 `SD` 解释器，使其贴合当前策略分布下的结构态演化。完成拟合后，该 `SD` 解释器不再只是预测未来，而是对该周期内的好轨迹、差轨迹和随机抽样轨迹进行反向分析，识别轨迹中的关键决策点、失败前兆、相位转换错误和成功恢复模式。

这一过程试图模仿人类总结经验的方式。人在完成一个复杂动作任务时，往往不是只靠下一步预测，而是在周期性复盘中总结“何时做对了”“何时做错了”“哪些状态下不能这样做”“哪些动作只有在特定准备条件下才有效”。对应到 Hopper 中，反向归因模块应能发现类似原则：在脚底尚未形成充分支撑且躯干角速度仍向失败方向扩大时，不应急于施加推进；在接近落地且垂直速度过大时，应优先形成稳定支撑而非继续腾空；在已经具备近直立支撑和正向速度时，可以允许更强的推进或起跳动作。这类原则不应由人工硬编码，而应由 `SD` 对好坏轨迹的结构态差异、短视界反事实和跨周期一致性进行归纳得到。

新方向的核心产物不是一个更低均方误差的动力学模型，而是一套可以逐步生长的任务规则树。规则树可以包含状态条件、相位条件、动作类别偏好、风险判据和例外条件。例如，规则可以表达为：“若处于 grounded 相位，躯干角已经明显后仰且角速度继续后仰，则抑制会进一步增大后仰角速度的动作候选”；也可以表达为：“若处于 near-landing 相位，足端接触概率低且下落速度高，则优先选择增加支撑接触稳定性的动作类别”。这些规则不必在第一阶段直接输出确定动作，而可以先输出动作采样权重、候选动作风险分数、探索优先级或 replay 片段重要性。

因此，该研究线暂定为 `SD Retrospective Attribution and Rule Induction`，中文可称为“基于系统动力学的反向归因与规则归纳”。其方法论主张是：`SD` 不只是服务于前向 imagined rollout 的世界模型，也可以成为将连续交互经验压缩为可解释、可复用、可审计任务规则的经验总结器。若该方向成立，未来的强化学习训练过程将不再只依赖随机探索、价值函数更新和前向规划，而可以周期性地从自身历史经验中总结结构化规则，再用这些规则约束、偏置或解释后续探索。

### 初始系统边界

第一阶段应保持该方向与现有 TD-MPC2 主线解耦。TD-MPC2 不因该模块而改变训练目标、优化器、replay 机制或 planner 接口；新模块只读取已经落盘的轨迹、评估汇总和 checkpoint 结果。这样可以避免在尚未验证归因可靠性之前，将不成熟规则直接注入在线训练并固化早期局部策略偏差。

第一阶段的输入包括：

1. 每个训练周期的 `eval.csv`、`eval_full.csv`、`train.csv`；
2. 官方评估目录中的 `eval_episode_summary.csv`；
3. 可选的逐 episode trace CSV；
4. 对应周期的策略 checkpoint 或 SD checkpoint；
5. 当前 Hopper `SD` 结构态定义和相位诊断逻辑。

第一阶段的输出包括：

1. 轨迹分层结果：好轨迹、差轨迹、中间轨迹和随机轨迹；
2. 关键事件标注：落地失败、支撑不足、后仰扩大、推进时机错误、起跳准备不足、恢复成功等；
3. 反事实诊断结果：在同一结构态附近替换动作、微扰动作或采用高分轨迹相似动作后，短视界结构态是否改善；
4. 候选规则树：以可读条件表达任务原则，并记录覆盖率、误杀率、好轨迹保留率和差轨迹预警率；
5. 周期性经验报告：说明该周期新增了哪些稳定原则，哪些旧规则被削弱、修订或保留。

### 初始研究假设

本方向依赖以下研究假设。

第一，Hopper 的成功与失败并非完全由不可解释的高维策略细节决定，而是在若干结构态变量和相位转换上存在可重复模式。当前 mechanism-oriented `SD` 主线中，高分仍主要表现为向前推进、短暂腾空和部分充分支撑片段，稳定充分支撑步态尚未完全收敛；这恰好提供了反向归因的对象，即分析哪些结构态条件使推进转化为有效得分，哪些条件使推进退化为失稳或倒地。

第二，`SD` 结构态足以承载一部分任务原则。规则不必覆盖全部低层连续控制细节，但应能覆盖训练中最常见、最有代价的失败机制。例如，足端接触、躯干角速度、垂直速度和支撑相位可能足以解释相当一部分“接触前推进”“落地后支撑不足”“躯干姿态恶化后仍施加错误扭矩”等失败。

第三，反向归因应以短视界反事实作为核心证据，而不能仅依赖好坏轨迹的相关性差异。若某一状态-动作模式只是在失败轨迹中常见，但替换该动作并不能改善 `SD` 预测的短视界结构态，则该模式不应被直接归纳为因果规则。反之，若某一相位下的动作替换稳定降低失败风险，则该模式可作为候选原则进入规则树。

第四，规则树应先作为软干预使用。早期规则可能存在覆盖不足、误杀成功探索、固化局部策略等风险，因此初始接入方式应优先选择动作采样偏置、风险打分、replay 采样权重或诊断报告，而不是直接替代 TD-MPC2 的动作选择。只有在跨 seed、跨周期、跨 checkpoint 验证后，规则才适合逐步提高干预强度。

### 第一阶段工作目标

第一阶段目标不是立即提出完整的新算法，而是建立可审计的离线研究闭环：

1. 从已有 Hopper 轨迹中抽取跨周期、跨 seed 的轨迹集合；
2. 建立独立 `SD` 解释器训练与评估脚本；
3. 建立好坏轨迹分层与关键相位标注；
4. 建立短视界反事实分析流程；
5. 输出第一版候选规则树及其离线验证指标；
6. 判断规则是否具备进入在线训练软干预的条件。

若第一阶段成立，第二阶段再考虑将规则树接入 TD-MPC2，优先作为 MPPI 采样先验或候选动作风险惩罚；第三阶段再评估是否可以在高置信状态下执行规则动作，逐步减少随机探索。

### 与现有 Hopper 主线的关系

该方向不是否定当前 mechanism-oriented `SD` 主线，而是对其进行方法论扩展。现有主线回答的是：结构化 `SD` 能否改善 TD-MPC2 的前向建模、规划和样本效率。新方向回答的是：结构化 `SD` 能否进一步承担经验复盘、失败归因和规则形成，从而把交互数据转化为可解释任务知识。

二者在工程上应暂时分离，在概念上保持连续。当前 Hopper 主线提供轨迹、结构态定义、相位诊断和已验证的 `SD` 建模基础；新方向在这些基础上增加反向归因、反事实比较和规则归纳。若后续规则树能够稳定改善训练，新方向可成为未来“决策导向系统动力学世界模型”的下一层方法论：从前向规划走向前向规划与反向总结共同驱动的经验自我改进。

### 代码隔离方案

为避免新方向与既有 Hopper 主线互相干扰，已建立独立代码与结果目录：

1. 离线代码目录：[experiments/sd_retrospective_rule_induction](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction)；
2. 独立结果目录：[results/sd_retrospective_rule_induction](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction)；
3. 初始配置文件：[hopper_offline_v1.yaml](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/configs/hopper_offline_v1.yaml)；
4. 初始只读索引脚本：[build_hopper_manifest.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_manifest.py)。

该隔离方案采用三条约束。第一，输入只读，只从 `logs/hopper-hop/`、既有 Hopper 诊断结果和 `src/world_model/dmcontrol_hopper_sd_model.py` 读取数据或定义，不移动、不清洗、不覆盖原始训练日志。第二，输出独立，所有 manifest、独立 `SD` 解释器、归因结果、候选规则树和周期报告均写入 `results/sd_retrospective_rule_induction/`。第三，不改主线，第一阶段不修改 `external/tdmpc2/`、现有 `run_hopper_v5_*` 脚本或 Hopper 在线训练配置。

当前新增的 `build_hopper_manifest.py` 只负责索引已有 `100k/200k/300k` mechanism-oriented `SD` Hopper 日志，并将运行目录、`eval.csv`、`eval_full.csv`、`train.csv` 与 `official_eval/step_*/eval_episode_summary.csv` 记录到新结果目录中的 JSON manifest。该脚本不训练模型，不写入 `logs/hopper-hop/`，也不被任何 TD-MPC2 训练入口自动调用。

### 第一版离线闭环实现

在上述隔离边界下，已实现第一版离线闭环。该版本暂不重新训练独立 `SD` 解释器，而是直接复用官方评估 trace CSV 中已经落盘的 `SD` 派生结构态字段进行归因，包括 `full_foot_support`、`grounded_rear`、`grounded_prone`、`grounded_upright`、`grounded_prepared`、`falling_air`、`rotation_penalty`、`vx`、`vz`、`pitch_rate` 与动作列。这样做的目的，是先验证“轨迹分层 -> 结构态归因 -> 规则归纳 -> 离线验证”这一经验总结闭环是否可运行、可审计、且不干扰主训练线。

第一版新增脚本如下：

1. [run_offline_v1.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_offline_v1.py)：顺序执行完整离线管线；
2. [stratify_trajectories.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/stratify_trajectories.py)：按周期内 episode return 的 `20% / 80%` 分位数标记差轨迹与好轨迹，并加入固定随机种子的随机抽样轨迹；
3. [attribute_failures.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/attribute_failures.py)：从逐帧 trace 中提取无支撑高幅动作、落地准备不足、高旋转风险、后仰/俯伏陷阱、充分支撑准备和有效推进等结构态事件；
4. [induce_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/induce_rules.py)：从归因特征中生成第一版单条件决策列表形式的候选规则树；
5. [validate_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/validate_rules.py)：在全部样本、训练样本和 hold-out seed 上计算风险规则与成功规则的召回率和误警率。

已执行：

```bash
python experiments/sd_retrospective_rule_induction/run_offline_v1.py
```

本轮共索引 `9` 条 Hopper 运行，覆盖 `100k/200k/300k` 三种子 mechanism-oriented `SD` 目录；从 `360` 条官方评估 episode 中选择 `258` 条进入第一版归因集合，所有 trace 均成功读取，未出现缺失。候选规则输出为：

- [hopper_offline_v1_rules.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/hopper_offline_v1_rules.md)
- [hopper_offline_v1_rule_validation.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_offline_v1_rule_validation.json)

第一版保留了 `3` 条规则：

1. 风险规则：`prone_or_rear_trap_frac >= 0.9760`，精确率 `0.940`，召回率 `0.504`。该规则说明后仰或俯伏接地片段极高时，轨迹高度可能属于差轨迹；
2. 成功规则：`mean_vx >= 0.8886`，精确率 `0.600`，召回率 `0.542`。该规则说明平均正向速度较高与好轨迹相关；
3. 成功规则：`max_rootx >= 17.7820`，精确率 `0.600`，召回率 `0.542`。该规则说明最大前向位移较高与好轨迹相关。

离线验证结果显示，在全部 `258` 条样本中，风险规则命中 `125` 条差轨迹中的 `63` 条，差轨迹召回率为 `0.504`，对好轨迹的误警率为 `0.0`；成功规则命中 `72` 条好轨迹中的 `39` 条，好轨迹召回率为 `0.542`，对差轨迹的误接收率为 `0.056`。在 hold-out seed `3` 上，风险规则对差轨迹召回率为 `0.377`，对好轨迹误警率仍为 `0.0`；但该 hold-out 集合中好轨迹仅 `1` 条，因此成功规则的 hold-out 统计暂不充分。

该结果支持如下初步判断：第一版离线闭环已经能够从已有轨迹中自动提炼出与人工观察一致的失败和成功原则，即“长期后仰/俯伏接地是强失败信号”“正向推进距离和速度是强成功信号”。但当前规则仍然偏粗，尚未达到可直接接入在线训练的程度。下一步应优先加入更细粒度的短视界反事实动作比较，并将 `fit_sd_explainer.py` 落地，使规则不只来自 trace 相关性，还来自 `SD` 对候选动作后果的局部反事实预测。

### 短视界反事实动作比较

在第一版离线闭环上，已补充真正的短视界反事实动作比较。新增脚本 [counterfactual_actions.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/counterfactual_actions.py) 对每条已选轨迹最多抽取 `3` 个关键状态。对每个状态，脚本从 trace 行恢复 Hopper 的原始 observation，使用 `DMControlHopperSDModel` 进行 `5` 步短视界 rollout，并比较以下动作：

1. 该状态下策略实际执行的动作；
2. 零动作；
3. 四个动作维度上幅度为 `0.35 / 0.70 / 1.00` 的正负单关节探测动作。

短视界结构态得分奖励足底支撑、接地近直立、准备支撑和正向速度，惩罚后仰/俯伏陷阱、下落和高角速度。该得分不作为环境回报替代物，而是作为离线归因中的结构态比较指标。

为避免全量反事实运行过慢，脚本已采用批量 rollout：同一状态下的实际动作和全部探测动作一次性送入 `SD` 模型，并行完成 `5` 步 rollout。完整管线已重新运行：

```bash
python experiments/sd_retrospective_rule_induction/run_offline_v1.py
```

反事实输出为：

- [hopper_offline_v1_counterfactuals.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.json)
- [hopper_offline_v1_counterfactuals.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.md)

本轮共生成 `774` 个关键状态反事实记录。总体上，实际动作在 `26` 个探测动作中的平均排名约为 `14.91`，进入前三的比例约为 `0.243`，最佳反事实动作相对实际动作的平均结构态得分提升约为 `0.089`。差轨迹中的平均改进空间约为 `0.109`，好轨迹约为 `0.077`，说明差轨迹在短视界结构态意义上确实存在更大的可替代动作空间。

该结果将“失败归因”从纯相关性统计推进到局部反事实比较：不仅能看到哪些结构态模式与失败相关，也能在同一状态下比较替代动作是否被 `SD` 预测为更优。当前反事实仍使用默认结构模型，而非周期拟合后的独立解释器；因此下一步应实现 `fit_sd_explainer.py`，并比较“默认结构模型反事实”与“周期拟合解释器反事实”的一致性。

### 第二版：独立解释器与状态-动作级规则

在第一版反事实闭环基础上，已实现 `offline_v2`。该版本新增独立 `SD` 解释器拟合，并将轨迹级统计规则推进为状态-动作级规则。新增入口如下：

1. [fit_sd_explainer.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/fit_sd_explainer.py)：从第一版已选轨迹中抽样转移，拟合独立 `SD` 解释器；
2. [run_offline_v2.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_offline_v2.py)：顺序执行解释器拟合、解释器反事实和状态-动作级规则归纳；
3. [induce_state_action_rules_v2.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/induce_state_action_rules_v2.py)：从短视界反事实记录中归纳 `state_event -> action_family` 规则。

已执行：

```bash
python experiments/sd_retrospective_rule_induction/run_offline_v2.py --epochs 3 --max-transitions 50000
```

本轮解释器使用 `50,000` 条 trace 转移训练，按 episode 切分得到 `45,010` 条训练转移与 `4,990` 条验证转移。解释器采用 `use_transition_residual=true`、`transition_residual_scale=1.0`，训练 `3` 个 epoch。验证指标显示，`one_step_struct_rmse=1.4452`、`rollout_1_rmse=0.9650`、`rollout_3_rmse=1.6169`、`rollout_5_rmse=1.8415`、`vx_mae=0.1706`、`vz_mae=0.3042`、`pitch_mae=0.0363`、`pitch_rate_mae=1.6037`、`full_foot_support_mae=0.2137`。对应产物为：

- [hopper_offline_v2_sd_explainer.pt](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/sd_explainers/hopper_offline_v2_sd_explainer.pt)
- [hopper_offline_v2_sd_explainer_summary.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/sd_explainers/hopper_offline_v2_sd_explainer_summary.json)

随后使用该解释器重新执行 `774` 个关键状态的短视界反事实。相较默认结构模型，解释器反事实下实际动作平均排名从约 `14.91` 改善到约 `13.90`，进入前三的比例从约 `0.243` 提高到约 `0.288`，最佳反事实动作相对实际动作的平均结构态得分提升从约 `0.089` 增加到约 `0.106`。这说明经轨迹拟合后的解释器对当前策略分布给出了更清晰的局部动作差异。

基于解释器反事实，第二版生成了 `8` 条状态-动作级候选规则，输出为：

- [hopper_offline_v2_state_action_rules.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/hopper_offline_v2_state_action_rules.md)
- [hopper_offline_v2_state_action_rules.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/hopper_offline_v2_state_action_rules.json)

排名靠前的规则包括：

1. `unsupported_high_torque -> prefer hip_pos`，支持数 `74`，平均反事实优势 `0.1262`；
2. `unsupported_high_torque -> prefer knee_pos`，支持数 `47`，平均反事实优势 `0.1236`；
3. `rotation_risk -> prefer knee_neg`，支持数 `41`，平均反事实优势 `0.1357`；
4. `unsupported_high_torque -> prefer hip_neg`，支持数 `28`，平均反事实优势 `0.1823`；
5. `unsupported_high_torque -> prefer waist_neg`，支持数 `21`，平均反事实优势 `0.1822`。

该结果说明，反向归因已经从“这条轨迹整体失败”推进到“在某类状态事件下，哪类动作族更可能在短视界结构态上改善后果”。但当前规则仍是候选规则，尚未经过在线策略影响验证，也未验证动作族之间是否存在相互冲突。下一步应进行规则一致性审查：按 seed、训练周期和事件类型分别统计同一状态事件下推荐动作族是否稳定，并检查高支持规则是否会误杀好轨迹中的关键动作。

### 第三版：规则一致性审查

根据上述要求，已实现 [analyze_rule_consistency.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/analyze_rule_consistency.py)，并将其纳入 [run_offline_v2.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_offline_v2.py)。该脚本对第二版状态-动作级规则进行三类审查：

1. 跨 seed 与跨周期覆盖：检查规则支持样本是否集中于少数 seed 或少数训练阶段；
2. 好轨迹误伤风险：统计规则推荐动作族在好轨迹中出现的比例；
3. 同事件动作冲突：统计同一 `state_event` 下最优动作族分布的归一化熵，判断是否需要进一步细分状态条件。

已基于第二版解释器反事实结果运行一致性审查，输出为：

- [hopper_offline_v3_rule_consistency.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_offline_v3_rule_consistency.md)
- [hopper_offline_v3_reliable_rules.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/hopper_offline_v3_reliable_rules.md)

本轮审查对 `8` 条 v2 候选规则进行筛选，仅保留 `1` 条 v3 可信候选：

- `unsupported_high_torque -> prefer hip_neg`，等级 `B`，支持数 `28`，覆盖 `3` 个 seed 与 `9` 个周期，平均反事实优势 `0.1823`，好轨迹误伤率 `0.071`，同事件冲突熵 `0.659`。

其余规则被降为待细分候选。主要原因有两类：第一，`unsupported_high_torque -> prefer hip_pos / knee_pos` 虽然支持数较高，但好轨迹误伤率分别为 `0.392` 与 `0.319`，说明这些动作族也经常出现在好轨迹的局部反事实偏好中，不能作为泛化软先验直接使用；第二，`rotation_risk` 相关规则的同事件冲突熵约为 `0.804`，说明该事件内部动作偏好分散，应继续按 `pitch_rate` 符号、`falling_air`、`full_foot_support`、`vx` 等变量细分状态条件。

该审查结果给出了比 v2 更保守的结论：当前已有一条可进入“在线软先验候选池”的规则，但仍不宜直接做大规模在线接入。下一步应围绕两个方向推进：一是对 `unsupported_high_torque` 和 `rotation_risk` 做子状态切分，降低同事件动作冲突；二是在离线 replay 中模拟规则打分，验证 `hip_neg` 偏好是否会保留好轨迹关键动作，同时压制差轨迹中的低质量动作。

### 第四版：子状态切分与离线回放验证

为检验 v3 规则是否需要进一步细化，已实现 `offline_v4` 的三项工具：

1. [analyze_substate_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/analyze_substate_rules.py)：对 `unsupported_high_torque` 与 `rotation_risk` 按 `pitch_rate` 符号、`vx` 区间、`full_foot_support` 区间和主导相位进行子状态切分，并重新统计动作族偏好；
2. [replay_rule_scoring.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/replay_rule_scoring.py)：在已选轨迹的逐帧 trace 上回放 v3 可信规则，统计实际动作与规则偏好的一致、相反和无关比例；
3. [build_rule_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_rule_library.py)：建立版本化规则库 [library.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/library.json)，区分 `reliable` 与 `candidate` 状态。

子状态切分输出为：

- [hopper_offline_v4_substate_rules.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/hopper_offline_v4_substate_rules.md)

本轮共识别 `33` 个子状态，生成 `4` 条 v4 子状态候选规则，且这些候选的支持样本均来自差轨迹：

1. `unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos`，支持数 `9 / 28`，平均反事实优势 `0.4234`；
2. `unsupported_high_torque|omega_pos|vx_low|support_none|falling -> prefer waist_pos`，支持数 `6 / 7`，平均反事实优势 `0.2181`；
3. `unsupported_high_torque|omega_neg|vx_low|support_none|falling -> prefer knee_neg`，支持数 `7 / 10`，平均反事实优势 `0.1889`；
4. `unsupported_high_torque|omega_pos|vx_high|support_none|rear -> prefer knee_neg`，支持数 `9 / 18`，平均反事实优势 `0.1785`。

该结果说明，v2/v3 中 `unsupported_high_torque` 下动作族冲突并不完全是噪声，而是包含可进一步拆分的子状态结构。不同 `pitch_rate` 符号、速度区间和相位条件下，短视界最优动作族确实可能不同。

离线回放输出为：

- [hopper_offline_v4_replay_rule_scoring.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_offline_v4_replay_rule_scoring.md)

对 v3 可信规则 `unsupported_high_torque -> prefer hip_neg` 的逐帧回放显示，该规则在已选 trace 中命中 `80,620` 个状态，实际动作与 `hip_neg` 对齐比例为 `0.133`，相反比例为 `0.197`。命中状态中包含 `36,965` 个差轨迹片段、`23,646` 个好轨迹片段和 `20,009` 个中间轨迹片段；相反动作也出现在好轨迹中 `5,768` 次。该结果提示，v3 规则虽然通过了关键状态反事实审查，但若直接在所有 `unsupported_high_torque` 状态上在线施加，会覆盖大量好轨迹片段，因此仍不应直接接入训练。

当前规则库 [library.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/library.json) 包含 `5` 条规则：`1` 条 v3 `reliable` 规则和 `4` 条 v4 `candidate` 子状态规则。该版本化规则库明确标注 `online_enabled=false`。下一步应把 v4 子状态规则继续做回放验证，并考虑将 v3 粗规则降级为 `needs_split`，只把通过回放的子状态规则保留为未来在线软先验候选。

### 规则稀疏产出原则

在讨论 v3 审查结果时，进一步明确一条方法论原则：每轮总结只生成少量规则，甚至不生成新规则，是正常且合理的。该研究线模拟的是经验总结过程，而不是规则数量最大化过程。真实经验总结中，大多数周期只是在验证、强化、修订或否决已有判断，只有少数周期会形成新的稳定原则。

因此，后续每轮离线总结的有效产物不只包括“新增规则”，还包括：

1. 强化已有规则，即已有规则获得更多跨 seed、跨周期证据；
2. 修订或拆分规则，即粗粒度事件下动作偏好冲突较高，需要细化状态条件；
3. 否决规则，即候选规则在好轨迹中误伤率过高；
4. 保持观测，即当前周期没有足够证据改变规则库。

该原则已同步写入 [methods.md](/home/data/round/code/sd_world_model_rl/docs/sd_retrospective_rule_induction/methods.md) 与 [decisions.md](/home/data/round/code/sd_world_model_rl/docs/sd_retrospective_rule_induction/decisions.md)。其核心理由是：若规则未来可能接入在线训练，错误规则的风险高于没有规则；规则归纳模块宁可空产出，也不应把不稳定相关性包装成任务知识。

### 第五版：子状态规则回放审查与规则库状态更新

在 v4 发现粗规则覆盖过宽后，进一步实现 `offline_v5`，目标不是生成更多规则，而是对已有子状态候选进行逐帧回放审查，并据此更新版本化规则库状态。新增工具包括：

1. [validate_substate_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/validate_substate_rules.py)：在已选 trace 的逐帧状态上回放 v4 子状态规则，统计命中数、好轨迹命中率、差轨迹命中率、动作对齐率、相反率以及跨 seed、跨周期覆盖；
2. [update_rule_library_status.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/update_rule_library_status.py)：综合 v4 粗规则回放与 v5 子状态回放结果，生成 [library_v5.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/library_v5.json) 和 [library_v5.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/library_v5.md)。

子状态规则回放输出为：

- [hopper_offline_v5_substate_rule_validation.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_offline_v5_substate_rule_validation.md)
- [hopper_offline_v5_substate_rule_validation.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_offline_v5_substate_rule_validation.json)

本轮共审查 `4` 条 v4 子状态候选。其中：

1. `unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos` 命中 `2,029` 个逐帧状态，好轨迹命中率 `0.053`，差轨迹命中率 `0.887`，覆盖 `3` 个 seed 与 `12` 个周期，升级为 `replay_validated`；
2. `unsupported_high_torque|omega_pos|vx_low|support_none|falling -> prefer waist_pos` 命中 `878` 个状态，但好轨迹命中率 `0.270`，保持 `candidate`；
3. `unsupported_high_torque|omega_neg|vx_low|support_none|falling -> prefer knee_neg` 命中 `715` 个状态，但好轨迹命中率 `0.274`，保持 `candidate`；
4. `unsupported_high_torque|omega_pos|vx_high|support_none|rear -> prefer knee_neg` 命中 `253` 个状态，样本量偏低，保持 `candidate`。

规则库 v5 包含 `5` 条规则，且继续保持 `online_enabled=false`。状态分布为：`1` 条 `needs_split`、`1` 条 `replay_validated`、`3` 条 `candidate`。其中 v3 粗规则 `unsupported_high_torque -> prefer hip_neg` 已由 `reliable` 降级为 `needs_split`，原因是其逐帧回放命中大量好轨迹片段，不适合作为无门控在线规则直接使用。

该结果支持此前的规则稀疏产出原则：v5 没有扩大规则数量，而是完成了一次规则库质量控制。下一步若进入在线 smoke，应只允许 `replay_validated` 子状态规则进入软干预候选池，并采用默认关闭、显式开启、低强度 MPPI 候选重加权的方式验证其影响；粗规则和未通过回放的候选规则不应进入在线训练。

### 第六版：在线软先验 smoke 入口

在 v5 规则库质量控制基础上，已实现第一版在线 soft-prior smoke 入口。该版本继续遵循独立研究线约束，不修改 `external/tdmpc2/`、不改写既有 `run_hopper_v5_*` 脚本，也不复用旧的 `sd_action_prior_enabled` 边界先验。新增文件为：

1. [online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py)：读取 [library_v5.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/rules/library_v5.json) 中 `replay_validated` 且来自 `v4_substate` 的规则，并在运行时为 TD-MPC2 agent 安装 MPPI 候选价值重加权补丁；
2. [train_hopper_v6_rule_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py)：独立训练入口，主体流程与 TD-MPC2 原始训练入口保持一致，只在 agent 构造完成后根据显式开关调用 `install_rule_prior`；
3. [run_hopper_v6_5k_rule_prior_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v6_5k_rule_prior_smoke.sh)：5k smoke 启动脚本，`RULE_PRIOR_ENABLED` 默认值为 `false`。

当前 v6 只允许一条规则进入在线候选池：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos
```

在线执行时，规则不直接替换动作，也不改写环境奖励、replay、优化器或模型结构。补丁仅在 MPPI 每轮候选序列估值后，对第一步动作属于高力矩候选且当前观测满足子状态条件的序列添加小幅价值偏置。默认参数为：`rule_prior_weight=0.05`，`rule_prior_opposite_penalty=0.02`，`rule_prior_max_bonus=0.10`。这些数值的作用是让规则先验参与候选排序，但不压过 TD-MPC2 学到的价值估计。

已完成静态检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/*.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

下一步实验应采用成对 smoke：先运行 `RULE_PRIOR_ENABLED=false` 的同入口对照，再运行 `RULE_PRIOR_ENABLED=true` 的规则先验版本。比较重点不是一次性证明收益，而是检查规则命中率、价值偏置幅度、评估轨迹是否出现误伤好轨迹片段，以及训练曲线是否出现异常退化。

#### 第六版 quick smoke 结果

已执行一轮成对 quick smoke，输出记录为 [hopper_v6_rule_prior_quick_smoke_20260505.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v6_rule_prior_quick_smoke_20260505.md)。由于默认 5k smoke 的 `seed_pretrain_updates=2500` 和中间评估耗时较长，本轮采用较窄设置：`steps=3000`，`eval_freq=3000`，`eval_episodes=1`，`seed_pretrain_updates=20`，`num_samples=64`，`num_elites=8`，`num_pi_trajs=8`，`iterations=3`。

成对运行目录为：

1. `logs/hopper-hop/1/sd_hopper_v6_3k_rule_prior_off_seed1_finalonly_20260505`
2. `logs/hopper-hop/1/sd_hopper_v6_3k_rule_prior_on_seed1_finalonly_20260505`

最终评估中，两条线的 `episode_reward` 均为 `0.0`，`episode_success` 均为 `0.0`。但规则开启线确实改变了命中子状态下的动作分布：规则关闭线在最终 eval trace 中命中目标子状态 `54` 次，其中 `hip_pos` 为 `16` 次、`hip_neg` 为 `25` 次，平均 hip 动作为 `-0.1053`；规则开启线命中 `41` 次，`hip_pos` 为 `41` 次、`hip_neg` 为 `0` 次，平均 hip 动作为 `1.0000`。

该行为改变没有带来正向效果。规则关闭线最终 `final_rootx=0.5113`、`max_rootx=0.6496`、`max_full_foot_support=0.9731`、`backward_speed_frac=0.074`、`rotation_penalty_frac=0.060`；规则开启线最终 `final_rootx=-0.2067`、`max_rootx=0.1633`、`max_full_foot_support=0.0000`、`backward_speed_frac=0.202`、`rotation_penalty_frac=0.150`。因此，当前 v6 规则不应继续增强在线干预强度。

该结果提示：离线回放验证能够说明规则定位失败片段，但尚不能保证该规则作为在线第一步动作偏好有益。下一步应先削弱或重构在线干预形式，例如降低 `rule_prior_weight`、加入动作幅度上限、要求接触可恢复门控，或把规则用于候选惩罚诊断而非正向推高 `hip_pos`。

#### 第六点一版：penalty-only 规则干预审查

根据 quick smoke 的失败结果，已实现 v6.1 的两项改动。第一，[online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py) 新增 `rule_prior_mode` 参数，支持 `prefer` 与 `penalty_only` 两种模式；第二，新增 [simulate_rule_prior_variants_v61.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/simulate_rule_prior_variants_v61.py)，在已落盘 eval trace 上审查不同规则干预形式会作用于哪些动作片段。

`penalty_only` 模式不再奖励 `hip_pos`。当目标子状态命中时，它只惩罚与规则动作族方向相反的高力矩动作，并可惩罚超过 `rule_prior_preferred_cap` 的同向过饱和动作。当前默认参数为：`rule_prior_mode=prefer`，`rule_prior_preferred_cap=0.70`，`rule_prior_cap_penalty=0.02`；因此既有 v6 结果仍可按原行为复现。若显式设置 `RULE_PRIOR_MODE=penalty_only`，则不再产生正向动作族奖励。

离线审查输出为：

- [hopper_v6p1_rule_prior_variant_audit.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit.md)
- [hopper_v6p1_rule_prior_variant_audit.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit.json)

审查结果显示，规则关闭 trace 中目标子状态命中 `54` 次，其中 `opposed_count=25`，`preferred_saturated_count=8`，penalty-only 会惩罚 `35` 个实际动作、保持 `19` 个动作不变；规则开启失败 trace 中目标子状态命中 `41` 次，全部为正向饱和 `hip_pos`，penalty-only 会惩罚全部 `41` 个实际动作。相比 `prefer` 模式对失败 trace 中 `41` 个动作全部给正分，penalty-only 更符合当前诊断目标：它把规则从“推高 hip_pos”改为“抑制相反或过饱和动作”。

该结果支持进入一轮更保守的在线 smoke，但不支持继续使用原 `prefer` 模式。下一轮在线验证应采用同一 final-only 快速配置，新增一条 `RULE_PRIOR_MODE=penalty_only` 线，并与已经落盘的 `rule off` 结果比较。

#### penalty-only quick smoke 结果

已执行 `RULE_PRIOR_MODE=penalty_only` 的同配置 quick smoke，输出记录为 [hopper_v6p1_penalty_only_quick_smoke_20260505.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v6p1_penalty_only_quick_smoke_20260505.md)。运行目录为：

```text
logs/hopper-hop/1/sd_hopper_v6p1_3k_rule_prior_penalty_only_seed1_20260505
```

最终评估中，penalty-only 线的 `episode_reward=0.0`、`episode_success=0.0`，仍未取得任务收益。但相较 v6 `prefer` 线，它消除了最明显的失败形态：目标子状态命中从 `41` 降至 `6`，饱和 `hip_pos` 从 `41` 降至 `0`，平均 hip 动作从 `1.0000` 降至 `0.4113`，`backward_speed_frac` 从 `0.202` 降至 `0.010`，`rotation_penalty_frac` 从 `0.150` 降至 `0.016`。

与 rule-off 基线相比，penalty-only 线仍较弱：`final_rootx=0.1970`，低于 rule-off 的 `0.5113`；`max_full_foot_support=0.0000`，低于 rule-off 的 `0.9731`。因此，penalty-only 可视为比 `prefer` 更安全的风险筛选形式，但 3k quick smoke 尚不能判断其是否能在更长训练中改善阶段转换。

需要补充说明的是，Hopper 任务具有奖励稀疏、完成过程阶段化依赖和早期探索高度不稳定等特征，不能仅以 3k 终点评估回报作为规则是否有效的判断标准。上述 quick smoke 的功能应限定为局部行为信号和明显负效应筛查：它表明 `prefer` 模式会导致饱和 `hip_pos`，而 `penalty_only` 能消除该失败形态；但它不足以否定 penalty-only 作为长程风险筛选器的潜在价值。

下一步应启动阶段性训练对比，采用相同 seed、相同训练预算，对比 `rule off` 与 `penalty_only`。判断指标除 `episode_reward` 外，还应包括 `final_rootx / max_rootx`、`max_full_foot_support`、`grounded_rear_frac`、`backward_speed_frac`、`rotation_penalty_frac`、目标子状态命中次数、规则惩罚次数以及阶段转换轨迹。若 penalty-only 在 20k pilot 中未出现明显退化，再考虑扩展到 50k 或补 seed。

#### 纯 TD-MPC2 第六点一版 20k pilot 结果

用户指出 20k pilot 初始运行速度异常，原因核查后确认：最初启动的两条 20k 线误用了 `world_model_kind=sd_hopper`，且仍包含在线 SD 监督训练配置。该组运行已停止，不纳入结果解释。随后脚本入口已改为可显式指定 `WORLD_MODEL_KIND=tdmpc2` 与 `SD_ONLINE_SUPERVISION_ENABLED=false`；当 `WORLD_MODEL_KIND != sd_hopper` 时，不再要求或加载 SD 初始化 checkpoint，`sd_supervised_init_from` 置为 `null`。

修正后的成对 20k pilot 已完成，报告记录为 [hopper_v6p1_tdmpc2_20k_penalty_only_vs_off_seed1_20260505.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v6p1_tdmpc2_20k_penalty_only_vs_off_seed1_20260505.md)。运行目录为：

```text
logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_off_seed1_20260505
logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_penalty_only_seed1_20260505
```

日志确认两条线均使用 `world_model_kind=tdmpc2`、`sd_online_supervision_enabled=false`、`sd_supervised_init_from=null`。因此，规则开启线较慢主要来自外接规则先验在规划阶段进行子状态匹配和动作惩罚计算，而非在线训练 SD。

评估回报显示，规则关闭线在 5k、10k、15k、20k 分别为 `0.000000`、`0.000199`、`0.011152`、`0.004151`；penalty-only 线分别为 `0.000000`、`0.148496`、`0.644448`、`0.000000`。这说明 penalty-only 在 10k 和 15k 出现中间正信号，但未在 20k 末端保持。

阶段指标进一步说明该信号具有不稳定性。15k 时，penalty-only 的 `final_rootx=0.3936`，高于 rule-off 的 `-1.2706`，且 `backward_speed_frac=0.0307`、`rotation_penalty_frac=0.0193`，低于 rule-off 的 `0.2080` 和 `0.1040`。但 penalty-only 仍未恢复 full-foot support，15k 的 `max_full_foot_support=0.0000`；20k 时其 `final_rootx=-0.3622`、`max_rootx=0.1641`，低于 rule-off 的 `1.2532` 和 `1.7839`。

该结果不支持将 penalty-only 直接判定为稳定改进，但也不支持仅凭 20k 末端回报否定该方向。更合理的解释是：当前规则先验能够短暂抑制 backward-speed 和 rotation-penalty 等失败形态，但缺少与支撑恢复、阶段进展相关的门控条件，因而不能稳定转化为 Hopper 的完整动作阶段。

该解释还应结合当前规则库规模理解。在线规则先验目前只使用 `1` 条通过回放验证的早期规则，没有理由期待单条早期规则在 `20k` 这类较小训练预算下彻底解决 Hopper。当前 pilot 的有效判断对象应限定为：这条规则是否能产生可检测的局部阶段信号，是否避免稳定负效应，以及是否为后续规则库扩展和阶段门控提供依据。

#### 在线闭环测试的方法论边界

已明确下一阶段闭环测试的研究目标：先验证解耦的周期总结机制是否有效，暂不进入 SD 与 TD-MPC2 的嵌入协同。闭环中 TD-MPC2 仍作为主学习线独立训练；SD 作为解释器线，在每个周期结束后读取该周期落盘轨迹，并从上一周期 SD checkpoint 接续训练，而不是每轮从全新 SD 开始。

周期总结阶段包括：收集轨迹、接续拟合 SD 解释器、执行失败归因和反事实比较、生成候选规则、只保留最显著的 `1` 条规则进入回放审查。若无稳定候选，则本周期不新增规则。通过审查的规则写入下一周期规则库，再以保守的在线规则先验影响后续探索。

该边界的研究含义是：第一版闭环只检验“SD 总结经验并提炼规则”的方法论价值，不把 SD 训练混入 TD-MPC2 在线优化器、replay 更新或世界模型梯度。后续若要测试 SD 与 TD-MPC2 的协同嵌入，应作为下一阶段的独立变量重新设计。

#### 第七版闭环 runner 实现

已实现第一版在线闭环入口 [run_hopper_v7_closed_loop.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py)。周期长度由 `--cycle-steps` 控制，不绑定 `10k` 或 `20k`；闭环周期数由 `--cycles` 控制。

该入口每个周期先调用独立 v6 训练入口运行纯 TD-MPC2，并显式设置 `WORLD_MODEL_KIND=tdmpc2`、`SD_ONLINE_SUPERVISION_ENABLED=false`。从第二周期开始，runner 通过 `resume_from` 加载上一周期 `final.full.pt`，同时加载上一周期闭环规则库。为支持这一点，已在 [run_hopper_v6_5k_rule_prior_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v6_5k_rule_prior_smoke.sh) 中新增 `RESUME_FROM` 环境变量透传。

周期总结阶段复用既有离线组件：manifest 构建、轨迹分层、失败归因、SD 解释器拟合、反事实比较、规则归纳、一致性审查、子状态拆分和回放验证。已在 [fit_sd_explainer.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/fit_sd_explainer.py) 中新增 `--init-checkpoint`，用于从上一周期 SD 解释器 checkpoint 接续训练。

新增 [closed_loop_rule_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py)，用于执行“每周期最多上线一条规则”的规则库更新策略。默认只从本周期规则库中选择 `status == replay_validated`、`source == v4_substate`、`intervention_type == soft_prefer_action_family` 的候选规则；若无符合条件的候选，则本周期不新增规则。

已完成最小参数展开检查：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --cycles 1 --cycle-steps 1000 --exp-prefix drycheck_v7 --dry-run
```

已完成静态检查：

```bash
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
python -m py_compile experiments/sd_retrospective_rule_induction/fit_sd_explainer.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
```

## 2026-05-06

### 第七版 10k x 5 闭环实验结果

已完成第一版在线闭环测试，周期长度为 `10000` 环境步，总预算为 `50000` 环境步，seed 为 `1`。实验报告已整理至 [hopper_v7_closed_loop_10k5_seed1_20260505.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_seed1_20260505.md)。

本轮运行目录为：

```text
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle001_target10000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle002_target20000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle003_target30000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle004_target40000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle005_target50000
```

闭环总结结果位于：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v7_10k5_seed1_20260505
```

日志确认 TD-MPC2 主训练线保持纯训练设置：`world_model_kind=tdmpc2`、`sd_online_supervision_enabled=false`、`sd_supervised_init_from=null`。从第二周期起，TD-MPC2 从上一周期 `final.full.pt` 恢复，SD 解释器从上一周期 `sd_explainer_cycle_*.pt` 接续加载。SD 第 2 至第 5 周期均加载了 `30` 个兼容 tensor，说明接续训练路径按预期工作。

官方评估回报为：5k `0.000000`，10k `0.182293`，15k `0.000000`，25k `0.000000`，35k `0.000000`，45k `0.000000`。由于恢复训练后的步数触发存在偏移，后续周期评估点记录为 `15001`、`25001`、`35001`、`45001`；50k 保存了最终 checkpoint，但未触发对应官方评估。

训练 episode 中存在间歇性正信号。第 4 周期最大训练 episode 回报为 `10.552876`，第 5 周期最大训练 episode 回报为 `7.907880`。官方评估阶段诊断中，第 4 周期 35k 评估出现 `final_rootx=0.6564`、`max_rootx=1.0277`、`full_foot_support_frac=0.2567` 的阶段信号，但该信号未在第 5 周期评估中保持。

规则库更新方面，5 个周期均为 `no_rule_added`。最终闭环规则库仍包含 `5` 条规则，未新增通过保守准入条件的在线规则。该结果符合此前设定的高门槛原则：证据不足时允许不生成新规则，避免把不稳定相关性写入在线规则库。

本轮结论为：闭环基础设施、TD-MPC2 跨周期恢复、SD 接续训练和规则库版本化均已验证；性能层面尚未出现稳定官方评估收益；方法论下一步应聚焦于周期总结证据供给和候选规则拒绝原因诊断，而不是直接增强当前单条在线规则。

### 第七版规则未新增漏斗审计

针对“为何 5 个周期均未新增规则”的问题，已新增审计脚本 [audit_closed_loop_rule_funnel.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_closed_loop_rule_funnel.py)，并生成报告 [hopper_v7_closed_loop_10k5_rule_funnel_audit_20260506.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_rule_funnel_audit_20260506.md)。

审计结论为：本轮没有新增规则的直接卡点不在闭环更新器，而在周期规则生成之前。每个周期的 `cycle_*/rules/library_v5.json` 均为空规则库；闭环输出中保留的 `5` 条规则全部继承自初始 `offline_v5` 基础库。`closed_loop_rule_update.py` 每轮均报告 `num_eligible_candidates=0`，因此执行 `no_rule_added`。

漏斗统计显示，第 1 周期选中 `6` 条官方评估 episode，其中 `1` 条 good、`5` 条 bad，形成 `18` 条反事实状态记录；第 2 至第 5 周期每轮仅选中 `3` 条 official eval episode，且全部为 bad，形成 `9` 条反事实状态记录。v2 粗事件规则要求同一 `event -> action_family` 至少 `8` 个支持样本且平均优势不少于 `0.10`；v4 子状态规则要求同一 `substate -> action_family` 至少 `5` 个支持样本、平均优势不少于 `0.12` 且 good share 不高于 `0.25`。本轮 5 个周期在 v2 和 v4 两层均主要被支持样本不足卡住。

更深层的结构性问题包括：第一，周期总结主要读取官方评估轨迹，没有利用训练 episode 中出现的非零回报或阶段诊断较好的片段；第二，第 2 至第 5 周期没有 good/near-good 对照，难以抽取“哪些决策相对更好”；第三，当前运行只有 seed 1，而 v3/v5 可靠性准入仍保留跨 seed 要求，即使产生周期候选，也难以在单周期内升级为 `replay_validated`。

因此，下一步应将规则库状态拆分为周期候选与累计验证两层。周期内允许产生 `cycle_candidate` 或 `cycle_supported`，用于记录单周期内的近邻候选；只有累计跨周期或跨 seed 后，才升级为 `replay_validated` 并进入在线规则先验。同时，周期总结应纳入训练 episode 中的非零回报片段、较高 `final_rootx / max_rootx` 片段和支撑恢复片段，补足当前仅靠官方评估轨迹导致的样本不足。

### 周期候选规则层实现

已实现第一版周期候选规则层。主要改动为：

1. [analyze_substate_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/analyze_substate_rules.py) 新增 `cycle_candidate` 与 `cycle_supported` 状态。严格上线支持阈值仍为 `min_support=5`、`min_advantage=0.12`、`max_good_share=0.25`；周期候选保留阈值为 `candidate_min_support=2`、`candidate_min_advantage=0.02`、`candidate_max_good_share=0.50`。
2. [build_rule_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_rule_library.py) 保留子状态规则自身状态，不再统一改写为普通 `candidate`。
3. [update_rule_library_status.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/update_rule_library_status.py) 在回放证据不足时保留 `cycle_candidate` 或 `cycle_supported`，只有满足原有高门槛时才升级为 `replay_validated`。
4. [closed_loop_rule_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py) 在“每周期最多上线一条 `replay_validated` 规则”之外，额外携带最多 `3` 条周期候选证据。在线动作先验仍只读取 `replay_validated`，因此周期候选不会直接干预训练。

已在既有 `hopper_v7_10k5_seed1_20260505` 输出上重放规则生成段，不重新训练 TD-MPC2，也不重新拟合 SD。结果显示，5 个周期共产生 `6` 条 `cycle_candidate`，其中 1 条在后续周期以同 id 更新，最终闭环规则库包含 `10` 条规则：初始 `offline_v5` 规则 `5` 条，累计周期候选证据 `5` 条。可上线 `replay_validated` 规则仍为 `1` 条，因此原先关于在线动作先验强度和效果的解释不变。

最终候选层摘要为：

```text
cycle001: 1 条 cycle_candidate
cycle002: 1 条 cycle_candidate
cycle003: 1 条 cycle_candidate
cycle004: 1 条 cycle_candidate
cycle005: 2 条 cycle_candidate
```

本次静态检查已通过：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/analyze_substate_rules.py experiments/sd_retrospective_rule_induction/build_rule_library.py experiments/sd_retrospective_rule_induction/update_rule_library_status.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/audit_closed_loop_rule_funnel.py
```

### 第七点一版闭环改造

为使闭环实验更可靠地运行，已实现第七点一版闭环改造。该版本不再把“5k 周期”理解为独立证据窗口，而是采用高频触发和滚动证据窗口结合的方式：可设置 `cycle_steps=5000`、`summary_window_cycles=3`，即每 `5000` 步总结一次，但每次总结读取最近最多 `15000` 步范围内的周期运行。

[stratify_trajectories.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/stratify_trajectories.py) 新增阶段感知分层。除原始 `reward` 模式外，新增 `stage` 与 `hybrid`。`stage` 分数综合 `final_rootx`、`max_rootx`、支撑恢复、直立准备、后退速度惩罚、旋转惩罚和俯伏/后仰比例。该设计用于处理 Hopper 早期 reward 全为 `0` 时无法区分 near-good 和 bad 轨迹的问题。

已用既有 `cycle_005` official eval 做快速验证：3 条评估 episode 的 reward 均为 `0`，原始分层会全部落入同类；阶段分层可选出 1 条 near-good 和 1 条 bad，其中 near-good 轨迹具有更高 `final_rootx` 与 `max_rootx`。这说明阶段分层能够为 SD 总结提供更有用的对照样本。

[closed_loop_rule_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py) 新增跨周期候选累计字段：`evidence_cycles`、`cycles_covered`、`cumulative_support` 和 `cumulative_mean_advantage`。满足默认 `rolling_min_cycles=2`、`rolling_min_support=6`、`rolling_min_advantage=0.05`、`rolling_max_good_share=0.25` 时，候选可升级为 `rolling_validated`。

[online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py) 与 [run_hopper_v6_5k_rule_prior_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v6_5k_rule_prior_smoke.sh) 新增 `rule_prior_allowed_statuses` / `RULE_PRIOR_ALLOWED_STATUSES`。默认仍为 `replay_validated`；若 v7.1 pilot 需要让滚动验证规则进入在线先验，可显式设为 `replay_validated,rolling_validated`。`cycle_candidate` 与 `cycle_supported` 仍不会被在线读取。

已完成验证：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/stratify_trajectories.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --seed 1 --cycles 2 --cycle-steps 5000 --summary-window-cycles 3 --trajectory-score-mode stage --rule-prior-allowed-statuses replay_validated,rolling_validated --exp-prefix drycheck_v71 --dry-run
```

### 第七点一版 5k x 10 闭环 pilot

已完成 `tdmpc2_v71a_closed_loop_5k10_seed1_20260506` 闭环实验。该实验采用 `cycle_steps=5000`、`cycles=10`、`summary_window_cycles=3`、`trajectory_score_mode=stage`、`rule_prior_allowed_statuses=replay_validated,rolling_validated`，总预算约 `50000` 环境步。为修正恢复训练后的评估触发偏移，启用了 `--align-resumed-eval`，实际评估步数为 `5000`、`10001`、`15002`、`20003`、`25004`、`30005`、`35006`、`40007`、`45008`、`50009`。

10 个周期均正常完成，并均在目标步数触发官方评估。官方评估回报全程为 `0.0`。训练 episode 中存在间歇性非零回报，周期最大值分别为 `5.672`、`0.105`、`1.172`、`8.600`、`1.103`、`4.188`、`4.870`、`5.436`、`8.257`、`7.264`。该结果说明训练过程中存在阶段性行为信号，但尚未形成稳定官方评估收益。

滚动窗口轨迹供给按预期运行：第 1 周期选择 `3/3` 条轨迹，第 2 周期选择 `6/6` 条轨迹，第 3 至第 10 周期均选择 `9/9` 条轨迹，且每轮均包含 `bad`、`middle`、`good` 三类阶段分层标签。SD 解释器从第 2 周期起均从上一周期 checkpoint 接续加载 `30` 个 tensor；第 3 至第 10 周期每轮使用 `4491` 条转移样本。

规则库方面，所有周期的闭环更新 `action` 均为 `no_rule_added`，表示没有新增满足 `replay_validated` 准入条件的规则。但滚动候选累计开始产生作用：cycle005 后在线规则数由 `1` 增至 `2`，cycle006 后增至 `4`。最终规则库包含 `13` 条规则，其中在线可读规则为 `1` 条 `replay_validated` 和 `3` 条 `rolling_validated`。最稳定的滚动规则为 `unsupported_high_torque|omega_neg|vx_low|support_none|prone->prefer:knee_pos`，覆盖 cycle004 至 cycle009 中的 6 个周期，累计支持数为 `14`，累计平均优势为 `0.385`。

本轮结论为：第七点一版闭环的工程链路已经可靠运行到 50k，且相较 `10k x 5` 版本，规则库已从初始单条在线规则扩展到包含滚动验证规则的在线先验集合。性能层面仍不能声称成功解决 Hopper；下一步应设置同预算纯 TD-MPC2 对照或多 seed 复现，以判断滚动规则先验是否带来整体训练曲线差异。

正式报告已写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_closed_loop_5k10_seed1_20260506.md
```

### 最终轮规则动作分布审计

已检查 cycle010 official eval 逐帧轨迹，重点验证在线规则命中子状态下动作方向是否按预期变化。审计报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_final_eval_rule_action_audit_20260506.md
```

结果显示，最终三个滚动规则共同命中的 `unsupported_high_torque|omega_neg|vx_low|support_none|prone` 子状态在最终轮出现 `61` 帧。按 `hip_neg`、`waist_pos`、`knee_pos` 三个偏好方向的关节符号统计，偏好方向占 `86.3%`，相反方向占 `13.7%`。作为对照，在规则上线前的 cycle001 至 cycle005，该同一子状态的相反方向占比约为 `35.8%` 至 `55.8%`；cycle007 至 cycle010 在 4 条在线规则启用后，相反方向占比降至 `4.7%`、`7.1%`、`7.6%`、`13.7%`。

初始 `replay_validated` 规则命中的 `unsupported_high_torque|omega_pos|vx_mid|support_none|rear` 子状态在最终轮出现 `5` 帧，`hip_pos` 方向为 `5/5`，相反方向为 `0/5`。若按幅度最大的主导动作族统计，偏好动作族占比仍不高；这与当前 `penalty_only` 设置一致，因为该设置主要惩罚相反动作，并不直接奖励偏好动作成为主导动作。

本次审计结论为：在线动作先验确实改变了命中子状态下的动作方向，尤其是减少了规则惩罚的相反方向动作。但官方评估回报仍为 `0.0`，因此不能直接解释为任务收益。后续应通过同预算纯 TD-MPC2 对照和多 seed 复现，区分“周期和预算尚不足以转化为回报”与“规则提炼仍不够准确或覆盖不足”两种解释。

### 第七点一版 100k 接续训练

已将 `tdmpc2_v71a_closed_loop_5k10_seed1_20260506` 从 `50k` 接续训练至约 `100k`。为支持该实验，[run_hopper_v7_closed_loop.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py) 增加 `--start-cycle` 与 `--resume-from` 参数，使闭环 runner 可以从任意周期恢复，并显式接收上一周期 TD-MPC2 checkpoint、SD 解释器 checkpoint 与闭环规则库。

接续命令从 `cycle010` 的 `final.full.pt`、`sd_explainer_cycle_010.pt` 和 `library_closed_loop_cycle_010.json` 开始，运行 `cycle011` 至 `cycle020`。最终落盘为：

```text
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/rules/library_closed_loop_cycle_020.json
```

官方评估回报在 `50009` 至 `100019` 区间仍为 `0.0`。训练 episode 后半段出现更强局部信号，尤其是 `cycle019` 最大训练回报达到约 `64.175`，`cycle020` 最大训练回报达到约 `26.711`。该信号说明训练分布中存在阶段性成功片段，但尚未转化为稳定评估策略。

规则库继续演化。`cycle020` 最终规则库包含 `23` 条规则，其中 `replay_validated=1`、`rolling_validated=7`、`candidate=3`、`cycle_candidate=11`、`needs_split=1`。在线规则先验在 `cycle010` 至 `cycle020` 均启用，在线可读规则数从 `4` 至 `8` 间变化，最终为 `7` 条。该结果说明闭环总结机制仍在积累规则证据。

同时，最终规则库暴露出同一粗子状态下动作族冲突的问题。`unsupported_high_torque|omega_neg|vx_low|support_none|prone` 下同时存在 `hip_neg`、`waist_pos`、`knee_pos` 和 `waist_neg` 等偏好。该现象说明当前子状态仍需继续细分，而不是简单增加规则权重或把所有滚动规则等价视为稳定控制原则。

正式报告已写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_closed_loop_100k_continuation_seed1_20260506.md
```

### 100k checkpoint 显式规则 eval 对照

为区分“规则未内化”与“规则在 eval 期也显式无效”，已对 `cycle020` 最终 checkpoint 进行不重训练的成对评估。新增入口为：

```text
experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py
```

该入口加载 `final.full.pt`，复用训练期 `install_rule_prior()` 补丁，并调用 `OnlineTrainer.eval()` 导出 official eval trace。评估按操作指南使用 `MUJOCO_GL=glfw`。由于该 checkpoint 采用闭环训练的小模型规格，eval 命令中显式设置 `mlp_dim=256`、`num_q=3` 和 `dropout=0.01`。

已完成两组评估：

```text
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_rule_off_seed1_20260506/official_eval/step_0100020/
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_rule_on_seed1_20260506/official_eval/step_0100020/
```

两组 `episode_reward` 均为 `0.0;0.0;0.0`。`rule_on` 成功加载 `8` 条在线可读规则，但未带来评估回报改善。

逐帧动作审计显示，显式规则确实在 eval 期改变了动作方向。按当前实现可实际命中的规则计算，`rule_off` 的规则命中动作检查中偏好方向为 `56.1%`、相反方向为 `40.1%`；`rule_on` 中偏好方向为 `86.1%`、相反方向降至 `3.5%`。因此，当前问题不应表述为“eval 时规则没有作用”，而应表述为“规则作为显式 planner 排序偏置有效改变了局部动作方向，但不足以生成完整任务收益，且没有被策略网络内化”。

本次同时发现一个实现限制：`online_rule_prior_v6.py` 的 `_state_matches()` 当前只允许 `unsupported_high_torque` 事件命中。因此，`cycle020` 规则库中的 `rotation_risk|omega_neg|vx_low|support_none|prone -> prefer knee_pos` 虽然被加载，但不会实际生效。下一步应修正规则 prior 的事件匹配逻辑或在报告中区分“已加载规则”和“可命中规则”。

正式报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_explicit_rule_eval_seed1_20260506.md
```

### 规则信心重放机制

根据规则冲突处理的新设想，已实现离线信心重放工具：

```text
experiments/sd_retrospective_rule_induction/replay_rule_confidence.py
```

该工具读取 `cycle001-020` 的 `library_v5.json`，将每轮子状态规则作为证据更新规则信心。冲突组定义为 `substate_key + joint`，因此只在同一子状态、同一关节、相反符号之间互相削弱；不同关节规则不互相削弱。该设计用于保留 `hip_neg + knee_pos` 这类可能互补的动作族，同时处理 `waist_pos / waist_neg` 这类直接冲突。

当前默认参数为：`positive_lr=0.35`、`conflict_lr=0.85`、`absence_decay=0.98`、`active_threshold=0.35`、`retired_threshold=0.05`。重放结果写入：

```text
results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json
results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.md
```

结果显示，共生成 `19` 条信心规则，其中 `confidence_active=5`、`confidence_observing=6`、`confidence_retired=8`。典型冲突组结果如下：

```text
unsupported_high_torque|omega_neg|vx_low|support_none|prone|joint:knee:
  knee_pos confidence = 0.611
  knee_neg confidence = 0.000

unsupported_high_torque|omega_neg|vx_low|support_none|prone|joint:waist:
  waist_neg confidence = 0.364
  waist_pos confidence = 0.000
```

这说明信心机制可以将原先并存的矛盾规则整理为竞争关系：后来持续出现的证据会削弱旧规则，旧规则降至退休阈值后不再作为在线强规则。

同时，[online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py) 已补充读取 `effective_prior_scale` / `confidence` 的能力。旧规则库没有该字段时默认比例为 `1.0`，不改变既有实验；若后续显式使用 confidence 规则库，则先验动作引导强度会按规则信心缩放。

正式报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_confidence_replay_seed1_20260506.md
```

### Confidence 规则库不重训练 eval

已使用同一个 `100k final.full.pt` 进行 confidence 规则库 eval，不重训练。规则库替换为：

```text
results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json
```

本轮只允许 `confidence_active`，共安装 `5` 条规则。评估输出为：

```text
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_conf_active_seed1_20260506/official_eval/step_0100020/
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_conf_active_seed1_20260506.json
```

三条 eval 回报仍为 `0.0;0.0;0.0`。但与原 `cycle020_rule_on` 相比，动作方向和部分阶段指标有所改善：

```text
cycle020_rule_on:
  opposed action checks = 3.5%
  full_foot_support_frac = 0.0027
  backward_speed_frac = 0.2807

confidence_active:
  opposed action checks = 0.0%
  full_foot_support_frac = 0.0793
  backward_speed_frac = 0.2060
```

该结果说明 confidence 规则库确实减少了冲突规则进入在线 prior 后的互相抵消，使显式动作引导更一致；但它仍未带来回报收益。当前更合理的下一步不是从头重训，而是先修正在线 prior 的多事件匹配，使 `rotation_risk` 类 confidence active 规则真正可命中，再决定是否从 `100k` checkpoint 接续训练。

正式报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_confidence_eval_seed1_20260506.md
```

### 多事件在线规则匹配修正

已修正 [online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py) 的事件匹配逻辑。此前 `_state_matches()` 只允许 `unsupported_high_torque` 命中，导致 `rotation_risk` 规则虽被加载但不可实际作用。修正后：

1. `_substate_parts()` 增加 `rotation_risk` 标记，条件为 `rotation_penalty >= 0.5` 或 `abs(pitch_rate) >= 3.0`；
2. `_state_matches()` 支持 `unsupported_high_torque` 与 `rotation_risk`；
3. `_candidate_gate()` 按事件选择候选门控：`unsupported_high_torque` 保留高力矩候选门控，`rotation_risk` 对当前旋转风险状态下的候选动作统一施加 confidence 缩放后的轻量惩罚。

静态检查已通过：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py
```

随后使用同一 `100k final.full.pt` 和同一 confidence 规则库重跑不训练 eval，实验名为：

```text
tdmpc2_v71a_100k_eval_conf_active_multievent_seed1_20260506
```

结果仍为 `0.0;0.0;0.0`。但动作审计确认 `rotation_risk` 规则已经实际命中：`rotation_risk` 共 `12` 次动作检查，偏好方向为 `100%`，相反方向为 `0%`。整体阶段指标与修正前几乎相同，说明多事件修正使实现口径更一致，但尚未单独带来 eval 回报改善。

正式报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_confidence_multievent_eval_seed1_20260506.md
```

### Confidence prior 100k 至 150k 接续与 eval 诊断

已从 `100k final.full.pt` 接续运行至 `150k`，实验名为：

```text
tdmpc2_v71a_conf_closed_loop_5k_seed1_20260506
```

接续实验使用 `cycle_steps=5000`、`start_cycle=21`、`cycles=30`、`summary_window_cycles=3`、`trajectory_score_mode=stage`。主训练保持 `world_model_kind=tdmpc2`，`sd_online_supervision_enabled=false`；规则先验只允许 `confidence_active`，并以 `penalty_only` 方式介入 MPPI 候选排序。

为避免 confidence 规则在闭环合并时被旧式 `cycle_candidate` 或 `rolling_validated` 状态覆盖，[closed_loop_rule_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py) 已增加 confidence 元数据保留逻辑。同名规则被周期证据更新时，原有 `confidence`、`confidence_status` 和 `effective_prior_scale` 会被保留，本周期旧式状态写入 `latest_cycle_rule_status`。

最终落盘为：

```text
logs/hopper-hop/1/tdmpc2_v71a_conf_closed_loop_5k_seed1_20260506_cycle030_target150029/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_030/sd_explainers/sd_explainer_cycle_030.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_030/rules/library_closed_loop_cycle_030.json
```

周期内 `3` 条 official eval 的结果为：`cycle027=32.69`、`cycle028=1.16`、`cycle029=25.61`、`cycle030=0.03`。训练 episode 的最高回报从 `cycle021` 的 `66.04` 上升到 `cycle030` 的 `145.26`，且非零训练 episode 数从 `2/10` 上升到 `7/10`。该信号说明训练分布中阶段性成功片段明显增加，但周期内 3-episode eval 仍不稳定。

针对 train/eval 严重不一致，已对同一 `150k final.full.pt` 做不训练诊断。将 eval 从 `3` 条扩展到 `9` 条后，默认 eval 动作模式下均值为 `64.03`，回报列表为：

```text
0.000000;0.000000;0.000000;141.175848;145.752106;0.000000;41.827946;94.016391;153.523193
```

这说明最终 checkpoint 并非完全无效；周期内 `eval_episodes=3` 的固定前三条初始状态过少且偏难，导致最终周期 eval 严重低估。

同时，关闭规则 prior 后，同一 checkpoint 的 `9` 条 eval 均值为 `88.29`，高于 rule-on 的 `64.03`。因此，当前 confidence rule prior 没有形成净收益，甚至可能拖低更广评估样本上的表现。后续应先审计 `rule_on` 与 `rule_off` 的轨迹差异，而不是继续增加规则权重。

正式诊断报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_150k_eval_diagnostics_seed1_20260506.md
```

### 100k 至 150k checkpoint 的 10-episode rule on/off 系统 sweep

为避免从单个 checkpoint 或非标准 episode 数得出过强结论，已对 `100k` 至 `150k` 保存 checkpoint 进行标准 `10` 条 episode 的 rule-on / rule-off 成对评估。评估脚本为 [eval_hopper_v71a_checkpoint_sweep.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_v71a_checkpoint_sweep.py)，每个 checkpoint 分别加载同一个 TD-MPC2 `final.full.pt`，并仅切换 `rule_prior_enabled=false/true`。`rule_on` 使用 `confidence_active` 规则、`penalty_only` 模式和训练期一致的显式 planner 排序补丁。

输出写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506/
results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506.md
```

主要结果如下：

```text
cycle  off_mean  on_mean  on-off
020      1.077    1.525   +0.448
021      1.057    1.057   +0.000
022     19.431   19.046   -0.385
023     21.556   22.502   +0.946
024      3.183    3.866   +0.683
025     30.370   31.311   +0.941
026     34.785   34.157   -0.629
027     34.806   35.172   +0.367
028     38.305   37.268   -1.037
029     50.888   70.566  +19.678
030     82.414   57.692  -24.723
```

该结果修正了先前 `9` 条 episode 的临时诊断口径。整体看，`cycle020-028` 的 on/off 差异大多远小于 episode 间方差，`cycle029` 显示 rule-on 明显正向，`cycle030` 则显示 rule-on 明显负向。因此，当前不能从单个 `150k` checkpoint 断言规则先验整体有害，也不能说已有 confidence 规则库已经稳定带来净收益。更准确的表述是：显式规则先验在不同训练阶段的影响具有 checkpoint 依赖性，后期存在正负两类显著局部效应，需要进一步做 episode-level 与 rule-hit 轨迹审计。

### Cycle029 与 Cycle030 的 rule on/off 行为审计

已复用 `10` 条 episode sweep 保存的逐步轨迹，对最后两轮 `cycle029` 与 `cycle030` 做 rule-on / rule-off 行为审计。审计不重新运行 eval，而是从 `episode_csv` 重算在线规则的子状态命中、动作方向和阶段指标。报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_cycle029_030_behavior_on_off_audit_seed1_20260506.md
results/sd_retrospective_rule_induction/reports/hopper_v71a_cycle029_030_behavior_on_off_audit_seed1_20260506.json
```

审计显示，On/Off 行为差异确实明显，但不表现为“规则稳定压低某一不良动作后回报单调改善”。在 `cycle029`，rule-on 的平均前向速度从 `0.108` 提高到 `0.214`，后退比例从 `0.269` 降至 `0.212`，非零回报 episode 从 `7/10` 增至 `8/10`，`>=100` 回报 episode 从 `3/10` 增至 `5/10`。在 `cycle030`，rule-on 的平均前向速度从 `0.318` 降至 `0.222`，后退比例从 `0.136` 升至 `0.156`，`>=100` 回报 episode 从 `6/10` 降至 `3/10`。

episode 级别看，差异主要来自早期轨迹分岔。`cycle030` 的 episode 0 与 episode 2 在 rule-off 下分别得到 `104.43` 和 `128.11`，而 rule-on 下均为 `0.0`；对应的 rule-on 轨迹几乎没有进入规则所描述的 `prone` 子状态，反而长期停留在 `rear` 姿态。相反，`cycle029` 的 episode 2 与 episode 7 在 rule-on 下分别从低回报或零回报跃升到 `102.71` 和 `115.95`，并且进入更多规则命中状态。

因此，当前更合理的解释是：规则先验确实会改变行为，但其影响主要通过早期动作排序改变后续状态分布；当压低某类局部动作后，轨迹可能进入更好或更差的姿态分支。局部“不良行为”减少并不必然转化为回报改善，因为 Hopper 的得分依赖连续阶段链条，而当前规则只覆盖 `omega_neg/vx_low/support_none/prone` 与少量 `falling` 子状态。

### 规则关键性覆盖审计

为检验“规则是否只覆盖了少量且并非关键的状态区域”，新增 [audit_hopper_v71a_rule_keyness.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py)。该脚本复用 `cycle029` 与 `cycle030` 的 `10` 条 episode on/off trace，不重新评估 checkpoint，统计 active confidence 规则的覆盖足迹、高低回报 episode 的状态分布，以及 on/off 差异最大的 episode 是否落在规则覆盖区域内。

报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_rule_keyness_audit_cycle029_030_seed1_20260506.md
results/sd_retrospective_rule_induction/reports/hopper_v71a_rule_keyness_audit_cycle029_030_seed1_20260506.json
```

审计结果显示，`cycle030` 在线可读 active 规则只有 `5` 条，但只对应 `3` 个唯一子状态：

```text
rotation_risk|omega_neg|vx_low|support_none|prone
unsupported_high_torque|omega_neg|vx_low|support_none|falling
unsupported_high_torque|omega_neg|vx_low|support_none|prone
```

动作族只包含 `knee_pos`、`waist_neg`、`waist_pos`，涉及关节只有 `knee` 和 `waist`，不覆盖 `rear`、`upright`、`omega_pos/mid`、`vx_mid/high`、`hip` 或 `ankle` 决策。

在 `cycle029-030` 的全部 `20000` 个 eval step 中，active 规则覆盖 `3333` 步，覆盖率为 `0.167`。其中 `rear` 阶段占全部步数 `0.291`，但 active 规则对 `rear` 的覆盖率为 `0.0`。高回报 episode（`>=100`）的平均规则覆盖率为 `0.279`、`rear` 占比为 `0.055`、`prone` 占比为 `0.865`；低回报 episode（`<=1`）的平均规则覆盖率仅为 `0.063`、`rear` 占比为 `0.427`、`prone` 占比为 `0.535`。

最大差异 episode 也呈现相同模式。`cycle030` 的 episode 0 与 episode 2 在 rule-off 下分别为 `104.43` 和 `128.11`，rule-on 下均为 `0.0`；rule-on 版本只被 active 规则覆盖 `6` 步和 `4` 步，并长期停留于 `rear`。相反，`cycle029` 中 rule-on 获益最大的 episode 2 与 episode 7 分别被 active 规则覆盖 `114` 步和 `129` 步，并转入更多 `prone` 恢复状态。

因此，本轮审计支持以下解释：当前规则并非完全无效，而是局部有效但任务关键覆盖不足。规则先验在轨迹进入 `omega_neg/vx_low/support_none/prone` 分支后可能产生正向作用；但一旦早期动作排序把轨迹推入 `rear` 等未覆盖区域，规则库无法继续提供恢复指导，因而出现显著分化。

### 解释力 v0 离线审计

为把规则价值从支持数和经验频率转向“解释力”，新增 [audit_hopper_v71a_explainability_v0.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_v71a_explainability_v0.py)。该脚本只做离线审计，不改变训练和在线 prior。候选池包括 active confidence 规则、周期候选规则、已退休规则和 counterfactual 诊断组。

初始分数定义为：

```text
Expl_v0 =
  0.35 * past_separation
+ 0.15 * past_label_power
+ 0.25 * cf_value_gain
+ 0.15 * transition_gain
+ 0.10 * basin_coverage
- 0.10 * conflict_cost
- 0.05 * complexity_cost
```

其中，`past_separation` 使用已发生 trace 中规则对齐动作与反向动作后的 `50` 步阶段结果差异；`past_label_power` 衡量高回报与低回报 episode 的规则对齐差异；`cf_value_gain` 与 `basin_coverage` 来自 SD counterfactual 记录；`transition_gain` 衡量规则对齐后是否更可能进入下一恢复阶段；`conflict_cost` 和 `complexity_cost` 分别约束同组反向规则和过细规则。

报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.md
results/sd_retrospective_rule_induction/reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.json
```

本轮共评分 `42` 条规则或诊断组。排名靠前的规则主要不是当前 active 规则，而是当前未上线的 `rear`、`omega_pos/mid`、`hip/ankle` 相关规则。例如：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> hip_pos    Expl=0.382
unsupported_high_torque|omega_neg|vx_low|support_none|rear -> waist_neg   Expl=0.365
unsupported_high_torque|omega_pos|vx_mid|support_none|prone -> waist_pos  Expl=0.358
unsupported_high_torque|omega_pos|vx_low|support_none|prone -> waist_neg  Expl=0.330
```

相比之下，当前 `5` 条 active 规则中，只有 `unsupported_high_torque|omega_neg|vx_low|support_none|prone -> waist_neg` 得到正分 `0.136`；`knee_pos` 相关 active 规则为负分，`falling -> waist_pos` 也为负分。这说明当前 confidence 规则库中存在“按早期支持保留但当前解释力不足”的规则，同时也存在“当前未上线但能解释后期分岔”的候选。

该结果支持下一步方法调整：规则更新不应再主要由支持数、good share 和平均反事实优势驱动，而应引入解释力分数作为候选排序和 confidence 更新依据。第一版仍保留为审计指标，不直接改变训练，以避免在样本量和权重尚未稳定前扰动闭环。

### Explainability v0 规则库 eval-only smoke

为检验解释力规则是否值得进入接续训练，已先构造 eval-only 规则库，而不是直接重跑训练。新增 [build_hopper_v71a_explainability_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_v71a_explainability_rules.py)，从解释力 v0 报告中选择 `Expl_v0 >= 0.15`、已发生 trace 命中不少于 `100` 步的规则，并按同一冲突组只保留最高分规则。规则状态标记为 `explain_active`，`effective_prior_scale` 按 `Expl_v0 / 0.40` 归一化。

生成规则库：

```text
results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json
results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.md
```

该规则库共 `12` 条，补入了 `rear`、`omega_pos/mid` 和 `hip` 分支，同时移除了原先负解释力的 `knee_pos` active 规则。

随后新增 [eval_hopper_v71a_explainability_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_v71a_explainability_rules.py)，对 `cycle029` 和 `cycle030` checkpoint 做 `10` 条 episode eval-only 对照。对照结果为：

```text
cycle029:
  rule_off       = 50.888
  confidence_on  = 70.566
  explain_on     = 48.565

cycle030:
  rule_off       = 82.414
  confidence_on  = 57.692
  explain_on     = 50.615
```

输出报告：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506.md
```

该 smoke 未通过。解释力规则在 `cycle029` 没有保住原 confidence 规则的正向收益，在 `cycle030` 也没有修复 rule-on 相对 rule-off 的负效应。episode 级别看，`cycle029` 中原 confidence 规则显著改善的 episode 7 在 explain 规则下仍为 `0.0`；`cycle030` 中 rule-off 高分的 episode 0 与 episode 2 在 explain 规则下仍接近 `0.0`。

因此，当前不应进入解释力规则接续训练。更合理的下一步是修正解释力到在线规则库的映射：第一，降低或重新校准仅由少量 SD counterfactual 支撑的高分规则权重；第二，把 `past_label_power` 和已发生轨迹分岔解释力作为上线必要条件，而不是只参与加权；第三，对 `rear` 规则做单独的小集合 ablation，避免一次性上线 `12` 条高权重规则导致 planner 排序过度偏移。

补充解释：上述 eval-only smoke 只检验“在旧 confidence 规则训练出的 checkpoint 上，直接替换为 explainability 规则是否无缝适配”。该结果不能解释为解释力规则本身无效。旧 checkpoint 的模型、value 估计、replay 分布和 planner 习惯是在旧规则干预下形成的；旧规则已覆盖的局部行为可能没有被策略网络充分内化，而旧规则未覆盖的区域则承担了更多训练适配。因此，新规则直接插入旧 checkpoint 会产生分布错配。解释力规则的真实效果需要在同一规则机制参与训练一段时间后再判断。

基于该认识，新增 [run_hopper_v71a_explainability_short_resume.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v71a_explainability_short_resume.py)，用于配置一个短接续验证入口。默认从 `100k` checkpoint 接续 `5k x 2` 周期，使用 explainability v0 规则库作为初始规则库，允许 `explain_active` 进入在线 prior，但将基础惩罚权重降至 `0.01`，避免规则切换过猛。该入口目前先作为可复现实验配置，不等同于长预算正式训练。

### Explainability-v0 规则短接续训练

已执行 explainability-v0 规则短接续训练。实验从 `100020` 步的 `cycle020` full-state checkpoint 恢复，初始 SD explainer 从同周期 SD checkpoint 接续加载，初始规则库为 `12` 条 `explain_active` 规则。训练配置为 `5k x 2` 周期，即 `cycle021` 和 `cycle022`，在线 prior 只允许 `explain_active` 状态参与，采用 `penalty_only`，`rule_prior_opposite_penalty=0.01`，`rule_prior_cap_penalty=0.01`。

训练与总结均正常结束，最终产物为：

```text
logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/sd_explainers/sd_explainer_cycle_022.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/rules/library_closed_loop_cycle_022.json
```

周期内 `3` 条 quick eval 在两轮末尾均为 `0.0`。训练 episode 中，`cycle021` 有 `3/10` 个非零回合，最大回报 `6.94`；`cycle022` 有 `3/10` 个非零回合，最大回报 `78.87`。这说明短接续过程中出现了高回报片段，但未稳定转化为周期内 quick eval 性能。

SD 接续拟合按预期执行。`cycle021` 从旧闭环 `cycle020` SD checkpoint 初始化，加载 `30` 个张量，使用 `1497` 条 transition；`cycle022` 从 `cycle021` SD checkpoint 初始化，加载 `30` 个张量，使用 `2994` 条 transition。两轮最终验证损失均约为 `0.3068`，结构预测误差与此前同量级。

规则更新方面，`cycle021` 产生 `1` 条子状态候选，`cycle022` 产生 `3` 条子状态候选，但闭环更新器均未新增在线规则。最终规则库包含 `13` 条规则，其中 `12` 条仍为 `explain_active`，另有 `1` 条 `cycle_candidate`：

```text
unsupported_high_torque|omega_neg|vx_low|support_none|falling->prefer:knee_pos
```

该候选的累计支持为 `4`，平均优势为 `0.183`，但因 `support_below_online_threshold` 仍不参与在线 prior。训练日志中，规则 prior 命中行比例从 `cycle021` 的 `16/50` 上升到 `cycle022` 的 `24/50`，说明 explainability 规则确实在规划候选排序阶段被触发。

随后对最终 checkpoint 追加标准 `10` 条 episode 对照评估。结果为：

```text
rule_off:
  mean = 8.822
  list = 0.000;0.000;0.000;21.066;15.689;0.000;0.000;0.000;51.463;0.000

explain_on:
  mean = 15.955
  list = 0.000;0.000;0.000;80.774;48.390;0.000;0.000;0.000;30.383;0.000
```

`explain_on - rule_off` 的均值差为 `+7.13`，主要来自第 `4`、第 `5` 条 episode 的改善，同时第 `9` 条 episode 下降。因此，该短接续结果显示了弱正向兼容信号，但绝对回报仍低，且收益集中于少数初始状态。当前结论应表述为：工程闭环已经在 explainability 规则机制下打通，规则先验确实生效；但第一版解释力规则尚未形成稳定任务级收益，下一步仍需优先改进规则识别、规则上线阈值和阶段覆盖。

### Explainability-v0 规则中期接续观察

根据 Hopper 任务较复杂、短预算不应作为成败判据的判断，已将 explainability-v0 规则短接续实验继续放大为中期观察。实验从 `110k` checkpoint 接续到 `200k`，周期长度仍为 `5k`，共执行 `cycle023` 至 `cycle040`。该实验目的不是在 `200k` 预算下证明任务成功，而是观察同一规则机制在稍长训练中是否保持稳定、是否形成训练侧恢复片段、以及最终显式 prior 开关是否出现较大净效应。

最终产物为：

```text
logs/hopper-hop/1/tdmpc2_v71a_explain_long_5k_seed1_20260506_cycle040_target200039/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/sd_explainers/sd_explainer_cycle_040.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/rules/library_closed_loop_cycle_040.json
```

训练侧信号相对 `100k -> 110k` 明显增强。各周期训练 episode 中的最高回报从早期的 `86.8`、`94.2`、`111.4` 逐步上升到后期的 `229.2`、`232.6`、`237.4`；`cycle039` 中 `6/10` 个训练 episode 为非零，均值达到 `115.3`。周期内 `3` 条 quick eval 仍不稳定，大多数周期为 `0.0`，但 `cycle032` 出现 `58.57`，`cycle038` 出现 `74.19`。因此，较合理的表述是：训练侧恢复片段随预算增强，quick eval 出现偶发正信号，但尚未稳定转化为任务级性能。

规则更新方面，`cycle023-040` 均未新增在线规则。闭环更新器每轮仍产生子状态候选，最终候选数累计到 `14` 条，但由于支持不足、跨周期证据不足或在线阈值未满足，均保持候选状态。最终规则库状态为：

```text
explain_active = 11
rolling_validated = 3
cycle_candidate = 14
```

`explain_active` 从 `12` 降为 `11` 的原因是其中一条 `unsupported_high_torque|omega_neg|vx_low|support_none|prone->prefer:knee_neg` 在后期重新以 `cycle_candidate` 证据出现，但本轮支持数仅为 `2`，未达到在线阈值，因此不再参与在线 prior。这符合信心信号可削弱规则的设计方向。

最终 checkpoint 已追加标准 `10` 条 episode 的 `rule_off / explain_on` 对照。结果为：

```text
rule_off:
  mean = 88.514
  list = 219.547;0.000;0.000;231.711;212.722;0.000;0.000;0.000;221.155;0.000

explain_on:
  mean = 89.203
  list = 223.137;0.000;0.000;232.392;214.622;0.000;0.000;0.000;221.880;0.000
```

均值差为 `+0.690`。规则 prior 对已经成功恢复的 episode 有轻微同向提升，但没有改变哪些初始状态可以恢复。因此，本轮中期观察不支持“规则 prior 已经带来稳定任务级收益”，也不支持“规则 prior 明显有害”。更直接的结论是：工程闭环和显式 prior 机制已经能够长期运行，当前瓶颈主要是规则识别质量、阶段覆盖和上线阈值，而不是规则注入本身。

详细摘要记录于：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_long_final_10ep_seed1_20260506/summary.md
```

### 三种子共享规则闭环实验启动

根据后续实验设计，已新增三种子共享规则闭环入口 [run_hopper_v7_multiseed_shared_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py)。该入口将训练、总结和评估解耦：训练与 SD 回归、失败归因、反事实分析、规则更新仍保持 `5k` 周期；正式 checkpoint 与完整 `10` 条 episode official eval 仅在 `25k`、`50k`、`75k`、`100k` 保存点触发。

为避免 5k 总结依赖 quick eval，新增 [export_hopper_replay_traces.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py)，从每个 seed 的 `final.full.pt` 中导出 replay buffer episode，写入 `retrospective_train_traces/`。同时更新 [build_hopper_manifest.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_manifest.py) 与 [stratify_trajectories.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/stratify_trajectories.py)，使闭环总结能同时识别 official eval trace 与训练 replay trace。该改动保留 official eval 的独立地位，不把 5k 训练 replay 误记为正式评估结果。

启动前发现 TD-MPC2 默认会在 seed pretrain 点额外保存 checkpoint。为符合“正式 checkpoint 每 `25k` 一次”的口径，已在 [run_hopper_v6_5k_rule_prior_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v6_5k_rule_prior_smoke.sh) 增加 `CHECKPOINT_AT_SEED` 环境开关，并在多种子 runner 中固定为 `false`。5k 周期末仍保存 `final.full.pt`，但它只作为下一周期接续状态，不作为正式性能报告 checkpoint。

正式实验已以 `r3` 前缀启动：

```text
exp_prefix = tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3
seeds = 1,2,3
gpus = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_allowed_statuses = replay_validated,rolling_validated
```

第 `1` 个周期已完成端到端验证。三种子各导出 `10` 条训练 replay episode，共 `30` 条输入轨迹；分层筛选选中 `19` 条用于归因；共享 SD explainer 使用 `9481` 条 transition 完成拟合；反事实记录 `57` 个状态；子状态规则生成 `6` 条候选。闭环更新器本周期未上线新规则，输出共享规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_001/rules/library_closed_loop_cycle_001.json
```

第 `2` 个周期已从三个 seed 的 `cycle001` `final.full.pt` 接续启动，并读取同一个共享规则库。后续应重点检查 `cycle005` 是否按预期产生 `25k` official eval 与正式 step checkpoint。

### 三种子共享规则 100k 实验完成

三种子共享规则闭环实验 `tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3` 已完成 `20` 个周期，共 `100k` 环境步。训练、SD 回归、失败归因、反事实分析与规则更新保持 `5k` 周期；正式 checkpoint 与 `10` 条 episode official eval 仅在 `25k`、`50k`、`75k`、`100k` 执行。该运行未使用 5k quick eval 作为性能判断，而是从 replay buffer 导出轨迹供规则总结使用。

正式评估结果如下：

```text
25k:
  seed1 = 0.001563
  seed2 = 0.001151
  seed3 = 0.000154
  mean3 = 0.000956

50k:
  seed1 = 0.009328
  seed2 = 12.111750
  seed3 = 0.000025
  mean3 = 4.040368

75k:
  seed1 = 32.651904
  seed2 = 31.847339
  seed3 = 0.925306
  mean3 = 21.808183

100k:
  seed1 = 65.843415
  seed2 = 84.924441
  seed3 = 55.056809
  mean3 = 68.608222
```

`100k` 的 `10` 条 episode 明细显示三个种子均出现多次非零恢复片段：seed1 最大回报 `168.791`，seed2 最大回报 `182.089`，seed3 最大回报 `120.039`。这仍不能表述为 Hopper 任务已经成功解决，但相对 `25k` 至 `75k` 的变化，说明共享规则闭环在较短预算内形成了跨种子正向训练信号。

规则方面，最终共享规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_020/rules/library_closed_loop_cycle_020.json
```

最终规则库包含 `31` 条规则，其中 `replay_validated = 6`、`rolling_validated = 4`、`cycle_supported = 1`、`cycle_candidate = 20`。在线 prior 实际可用规则为 `10` 条，`cycle020` 训练日志确认三个种子均加载 `rules=10`，模式为 `penalty_only`，权重为 `0.05`。因此，前期“规则门槛过高导致退化为纯 TD-MPC2”的担忧在 `50k` 后得到缓解；规则上线仍保守，但并未停滞。

最终新增规则为：

```text
unsupported_high_torque|omega_pos|vx_low|support_none|rear->prefer:knee_pos
```

本轮结果支持以下阶段性结论：第一，三种子共享规则、5k 总结、25k 正式评估的工程管线已经稳定打通；第二，规则 prior 以轻量方式介入规划排序，并在 `100k` 内伴随跨种子性能上升；第三，当前结果仍不能证明规则本身相对纯 TD-MPC2 的因果收益，后续需要同预算纯 TD-MPC2 多种子基线、规则 prior 开关复评、以及更细的规则覆盖与冲突分析。

### 100k 最终轨迹规则动作方向检查

对 `cycle020` 的 `100k` official eval 轨迹进行了规则动作方向检查。由于 `cycle020` 训练与评估时实际加载的是 `cycle019` 规则库，因此统计使用：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_019/rules/library_closed_loop_cycle_019.json
```

该规则库在 `cycle020` 日志中确认为 `rules=10`，模式为 `penalty_only`，权重为 `0.05`。因此，规则机制并不直接奖励偏好动作，而是压低相反方向动作，并轻微压制超过 `preferred_cap=0.7` 的过强偏好动作。

在 `30` 条 official eval episode、共 `15000` 个时间步中，按规则逐条命中统计得到：

```text
rule-hit count = 2066
preferred direction count = 1568
opposite direction count = 26
neutral / weak count = 472
penalized actual count = 46
```

即在规则命中处，偏好方向动作占约 `75.9%`，相反方向动作占约 `1.3%`。这说明最终评估轨迹中，规则要求的相反动作确实被明显压低；偏好动作则主要通过“不被惩罚”的相对排序方式被保留，而非被直接加分。

主要规则的方向一致性如下：

```text
rotation_risk|omega_neg|vx_low|support_none|rear->prefer:waist_neg:
  hits = 109, prefer = 100, oppose = 0

unsupported_high_torque|omega_neg|vx_low|support_none|rear->prefer:knee_pos:
  hits = 950, prefer = 702, oppose = 5

unsupported_high_torque|omega_pos|vx_low|support_none|rear->prefer:knee_neg:
  hits = 928, prefer = 711, oppose = 11

rotation_risk|omega_neg|vx_mid|support_none|rear->prefer:knee_pos:
  hits = 57, prefer = 41, oppose = 4
```

存在一个局部反例：

```text
rotation_risk|omega_neg|vx_low|support_some|prone->prefer:hip_neg:
  hits = 5, prefer = 0, oppose = 5
```

该规则在最终 official eval 中命中很少，但命中时方向完全相反，后续应作为低支持、高风险规则单独复查。

同口径检查 `cycle020` 导出的训练 replay 轨迹时，结果更混合：`30752` 个 rule-hit 中，偏好方向 `16776`，相反方向 `8880`，中性 `5096`。该差异符合 replay buffer 包含训练周期探索样本和历史样本的性质。因此，判断最终策略行为时应优先使用 official eval；训练 replay 更适合作为规则提炼和失败归因的数据来源。

## 2026-05-07

### v8 成功片段与转接动作证据

新增 v8 成功延续与阶段衔接规则设计文档：

```text
docs/sd_retrospective_rule_induction/hopper_v8_success_extension_rule_design.md
```

该设计将现有失败恢复规则保留为 `recovery` 角色，并新增 `continue` 与 `transition` 角色。方法边界被设定为“通用流程 + 任务适配器”：Hopper 当前可使用 `pitch`、`vx`、`support`、`falling/prone`、`waist/hip/knee/ankle` 等字段，但这些字段不构成方法本体，未来迁移时应替换适配器。

新增成功片段挖掘脚本：

```text
experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py
```

默认复核 v72 三种子 `200k` 最终 official eval 轨迹后，输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.md
```

共分析 `30` 条轨迹，得到 `356` 个成功候选片段，其中 `349` 个为转接缺口，`7` 个为成功延续片段。只有 `2` 个片段被现有 v7 active 规则覆盖。该结果支持以下判断：当前规则库主要解释失败恢复，而对局部好姿态和阶段衔接失败覆盖不足。

随后新增转接动作证据脚本：

```text
experiments/sd_retrospective_rule_induction/mine_hopper_v8_transition_actions.py
```

该脚本从成功片段报告中抽取 `last_success`、`pre_degradation` 和 `degradation_onset` 三类关键帧，记录动作族、精炼阶段键、`S_continue` 与 `S_transition` 的下一步变化。候选规则归纳只使用 `last_success / pre_degradation / post_success` 等因果上游锚点，`degradation_onset` 只用于诊断；动作族支持默认只统计主导动作维度。

脚本输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.md
```

当前得到 `715` 条关键状态-动作记录、`30` 个规则键组和 `25` 条离线观测候选规则。候选全部为 `transition` 角色下的 `avoid_action_family`，主要集中在 `omega_pos / vx_high / support_none / near_good_air` 或 `near_good_contact` 状态。该结果说明，局部好姿态向下一支撑阶段衔接失败时，存在跨 seed 重复出现的动作风险信号；但这些候选仍只是观测证据，后续需要使用 SD 反事实目标 `S_continue` 和 `S_transition` 验证其决策收益。

进一步新增 v8 转接状态反事实脚本：

```text
experiments/sd_retrospective_rule_induction/counterfactual_hopper_v8_transition_actions.py
```

该脚本读取 v8 动作证据，加载 `cycle_040` 共享 SD explainer，并在 `last_success / pre_degradation` 状态上比较实际动作与固定动作探测集。默认目标为：

```text
S_joint = 0.5 * S_continue + 0.5 * S_transition
```

默认运行输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.md
```

本次评估 `300` 条关键状态，得到 `11` 条 SD 反事实候选规则，主要为 `near_good_air / near_good_contact` 转接状态下的 `soft_prefer:knee_neg`。该结果与观测层 `avoid` 候选互补：观测层指出哪些实际动作之后常伴随退化，SD 反事实层指出哪些探测动作在结构目标上优于实际动作。下一步应将二者合并为 v8 候选规则归纳器，并保留冲突削弱机制，避免观测 `avoid` 与反事实 `prefer` 被分别上线后互相抵消。

最后新增 v8 转接规则合并脚本：

```text
experiments/sd_retrospective_rule_induction/induce_hopper_v8_transition_rules.py
```

该脚本将观测层 `avoid_action_family` 候选与 SD 反事实层 `soft_prefer_action_family` 候选按同一 `rule_key / action_family` 合并，并显式标记同向或反向冲突。默认输出为：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_rules_200k_v72.json
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_rules_200k_v72.md
```

当前合并得到 `36` 条离线规则，其中 `24` 条为 `offline_v8_supported`，`12` 条为 `offline_v8_conflict`。该结果说明 v8 规则提取已经形成从片段识别、动作证据、SD 反事实到候选规则合并的第一条离线闭环。当前规则仍不能接入在线 prior；下一步需要增加 held-out 轨迹验证、冲突削弱和同一规则键下的动作族互斥选择。

新增 held-out 验证与冲突削弱脚本：

```text
experiments/sd_retrospective_rule_induction/validate_hopper_v8_transition_rules.py
```

默认 held-out 子集为 `episode % 3 == 0`。该脚本分别验证观测 `avoid` 规则在 held-out 子集中的下一步退化率和结构分数下降，以及 SD `soft_prefer` 规则在 held-out 子集中的反事实优势；随后在同一 `rule_key`、同一关节内比较证据强度，对冲突规则执行保留、削弱或保持未解冲突。

默认运行输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_rule_validation_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_rule_validation_200k_v72.md
```

结果为 `heldout_supported = 24`、`heldout_weak = 4`、`heldout_conflict_unresolved = 4`、`heldout_weakened_by_conflict = 4`。这说明 v8 候选库中相当一部分规则在 episode-level held-out 子集上仍有同向证据；同时，仍有一部分 `avoid` 与 `soft_prefer` 证据互相竞争，不能直接进入在线 prior。下一步应在训练 replay 或其他 checkpoint eval 上进行误伤率与跨周期稳定性验证。

继续新增 replay 安全审查脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v8_transition_replay_safety.py
```

该脚本只读取 `heldout_supported` 规则，并在 `cycle_040` 三种子训练 replay 中重新扫描逐帧命中。默认输入为三种子各 `400` 条 replay episode，共 `1200` 条轨迹、`600000` 步。输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_replay_safety_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_replay_safety_200k_v72.md
```

审查结果显示，`24` 条 held-out supported 规则中，`7` 条为 `replay_safety_supported`，`16` 条为 `replay_observe`，`1` 条为 `replay_false_intervention_risk`。唯一高风险规则为：

```text
transition|transition_gap|omega_pos|vx_high|support_none|near_good_contact|stable->avoid:knee_neg
```

该规则在 replay 中命中 `7891` 步，全部属于 near-good 命中，其中 `32.2%` 的好姿态命中步实际主导动作就是 `knee_neg`。因此，该规则虽然可能解释部分退化片段，但若作为 `avoid` prior 上线，可能压制成功片段中的常用动作。

本次审查还修正了 v8 transition 规则的安全判据。由于该类规则目标本来就是局部好姿态转接，不能再要求规则主要命中坏片段；安全性应主要看是否低误伤好姿态中的实际动作。按该判据，低误伤候选包括若干 `avoid:hip_neg`、部分 `avoid:waist_neg` 和 `soft_prefer:knee_neg`。下一步应把 `replay_safety_supported` 规则导出为独立候选层，并继续做跨 checkpoint eval 或短接续 smoke，而不是直接把全部 held-out supported 规则上线。

新增 replay-safe 候选导出脚本：

```text
experiments/sd_retrospective_rule_induction/export_hopper_v8_replay_safe_rules.py
```

默认筛选 `replay_false_intervention_rate <= 0.10` 的 `replay_safety_supported` 规则，并输出：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.json
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.md
```

当前导出 `7` 条规则，其中 `6` 条为 `avoid_action_family`，`1` 条为 `soft_prefer_action_family`。这形成了 v8 第一版可供下一阶段验证的窄候选层；该层仍不是在线规则库，后续需要在独立 checkpoint eval 或短接续训练中验证其净效应。

### v8 replay-safe 规则 eval-only 对照

为避免离线分析继续扩展而偏离主线，新增最小 eval-only 脚本：

```text
experiments/sd_retrospective_rule_induction/eval_hopper_v8_safe_prior.py
```

该脚本只比较三种模式：

```text
rule_off
v7_current
v8_safe
```

其中 `v8_safe` 读取 `7` 条 `v8_replay_safe_candidate`，并使用 `penalty_only` 轻量先验。为支持该测试，`online_rule_prior_v6.py` 做了向后兼容扩展：旧 v7 五段 `substate_key` 逻辑保持不变；当规则来源为 `v8_transition_replay_safety` 时，允许读取七段 `rule_key`，并支持 `avoid_action_family` 对目标动作族本身施加惩罚。

本次 `200k` 三种子、每种模式 `10` 条 episode 的 eval-only 结果为：

```text
rule_off:
  seed1 = 75.264
  seed2 = 136.417
  seed3 = 220.729
  mean3 = 144.137

v7_current:
  seed1 = 75.234
  seed2 = 140.564
  seed3 = 174.007
  mean3 = 129.935

v8_safe:
  seed1 = 73.647
  seed2 = 138.945
  seed3 = 219.486
  mean3 = 144.026
```

结果文件为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_safe_eval_only_200k_v72_20260508.json
results/sd_retrospective_rule_induction/reports/hopper_v8_safe_eval_only_200k_v72_20260508.md
```

该结果支持进入第二轮在线 smoke：`v8_safe` 没有出现明显推理期破坏性，三种子均值几乎与 `rule_off` 持平；旧 `v7_current` 在 seed3 上明显拉低，暂不应作为第二轮默认主先验。下一步应以 `v8_safe` 单独先验进行短接续训练验证，先观察规则命中、near-good 转接片段和最终 `10ep` 对照，而不是继续扩大离线规则分析。

### v8 replay-safe 第二轮在线 smoke

启动并完成 `200k -> 210k` 的三种子短接续 smoke，实验目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_smoke_200k_to210k_seed123_20260508
```

本次实验从 v72 三种子 `200k` checkpoint 接续，初始规则库为：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.json
```

协议为：`5k` 周期执行 SD 接续回归、失败归因和共享规则库更新；`5k` 周期末同时保存 checkpoint 并执行 `10` 条 episode 的 formal eval。在线 prior 只允许 `v8_replay_safe_candidate` 状态规则，模式为 `penalty_only`，`rule_prior_opposite_penalty = 0.02`，`rule_prior_cap_penalty = 0.02`，`rule_prior_preferred_cap = 0.70`。

`cycle041` 从 `200040` 接续至 `205040`。三种子 `10ep` eval 结果为：

```text
seed1 = 103.086
seed2 = 206.638
seed3 = 225.459
mean3 = 178.394
```

`cycle041` 后处理完整跑通，生成 `sd_explainer_cycle_041.pt`、反事实归因、规则一致性报告、replay scoring、`library_v5.json` 和闭环规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_smoke_200k_to210k_seed123_20260508/cycle_041/rules/library_closed_loop_cycle_041.json
```

该闭环库共 `8` 条规则，其中 `7` 条保持 `v8_replay_safe_candidate`，`1` 条为 `cycle_candidate`。`cycle042` 实际加载其中 `7` 条可上线规则，说明状态门控按预期工作。

`cycle042` 从 `205041` 接续至 `210041`。三种子 `10ep` eval 结果为：

```text
seed1 = 102.736
seed2 = 207.326
seed3 = 231.758
mean3 = 180.607
```

与 `200k` eval-only 起点 `v8_safe mean3 = 144.026` 相比，`205k` 和 `210k` formal eval 均值分别为 `178.394` 和 `180.607`。该结果是短程正向信号，但仍不能解释为 Hopper 已被解决；其主要意义是 v8 safe 规则在训练期接入、三种子共享规则、SD 接续训练、归因总结和规则库更新的工程闭环已经完整跑通。

规则先验命中统计显示，介入仍然较轻。`cycle041` 三种子合计 `150` 条训练日志记录中，`rule_prior_hit` 比例为 `0.100`，非零先验项比例为 `0.053`，平均先验项为 `-0.000656`。`cycle042` 中对应比例为 `0.087`、`0.073` 和 `-0.000899`。这说明规则确实参与了规划排序，但当前强度仍属于低权重、低覆盖的显式先验。

`cycle042` 后处理也完整结束，最终闭环规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_smoke_200k_to210k_seed123_20260508/cycle_042/rules/library_closed_loop_cycle_042.json
```

该库共 `9` 条规则，其中 `7` 条为 `v8_replay_safe_candidate`，`2` 条为 `cycle_candidate`。新增候选为 `rotation_risk|omega_pos|vx_high|support_none|rear->prefer:ankle_pos`，尚未进入在线 prior。`cycle042` 的中间规则候选显著增多：`state_action_rules_v2 = 26`、`reliable_rules_v3 = 11`、`substate_rules_v4 = 84`、`library_v4 = 95`。因此，规则验证步骤耗时明显增加，后续需要把候选压缩或 validation 加速作为工程优化点。

随后对第二轮 smoke 的规则更新机制做了最小工程修正。规则审计报告写入：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8safe_smoke_rule_audit_210k_20260508.md
```

本次修改包括三点。第一，在 `analyze_substate_rules.py` 中增加候选压缩，默认 `max_rules = 48`、`max_rules_per_substate = 2`、`max_rules_per_event_action = 8`，使大量相近 substate 候选在进入 validation 前被截断。以 cycle042 产物离线重放时，`substate_rules_v4` 从 `84` 条压缩为 `48` 条。

第二，在 `validate_substate_rules.py` 中将逐条规则扫描轨迹改为按 `substate_key` 建索引后扫描轨迹。新实现会缓存 trace CSV，并对每个状态事件计算一次 substate key，再更新匹配规则。使用 cycle042 压缩规则重放时，新旧实现的 `hits`、`aligned`、`opposed`、`good_hit_rate`、`bad_hit_rate` 等统计完全一致；全量 `84` 条规则的索引版 validation 约 `21.5s` 完成，压缩后 `48` 条规则约 `20.8s` 完成。

第三，在 `closed_loop_rule_update.py` 中修正滚动候选逻辑：已有候选若在新周期再次出现，会优先被更新，而不是被新的高分候选挤出 top-N；当 `required_status` 以 `v8_` 开头时，满足滚动阈值的候选晋升为同一在线可读状态，例如 `v8_replay_safe_candidate`；若候选跨周期重复出现但 `good_share` 超过安全阈值，则削弱为 `cycle_weakened` 并退出滚动晋升池。用 cycle042 离线重放时，`rotation_risk|omega_pos|vx_mid|support_some|rear->prefer:waist_pos` 累计到 `cycles_covered = 2`、`cumulative_support = 6`、`cumulative_mean_advantage = 0.149`，但 `good_share = 0.5`，因此被削弱而不是晋升。

### v8 replay-safe 续训至 250k 的首个在线周期

在完成候选压缩、索引式 validation 和滚动候选修正后，启动 `210k -> 250k` 的三种子接续实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_cont_210k_to250k_seed123_20260508
```

该实验从 `cycle042` 的三种子 checkpoint 和共享规则库接续，仍保持 `5k` 周期执行 SD 接续回归、反事实归因和规则更新；checkpoint 与完整 `10ep` formal eval 改为每 `25k` 执行一次。初始规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_smoke_200k_to210k_seed123_20260508/cycle_042/rules/library_closed_loop_cycle_042.json
```

初始 SD explainer 为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_smoke_200k_to210k_seed123_20260508/cycle_042/sd_explainers/sd_explainer_cycle_042.pt
```

`cycle043` 从 `210k` 接续至约 `215k`，三种子训练已完成，并完整生成 SD 接续拟合、反事实归因、规则提取、replay scoring、validation 和闭环规则库。该周期不执行 formal eval，因为当前协议只在每 `25k` 保存点进行完整 `10ep` eval。

真实在线周期验证了候选压缩机制已经生效。`cycle043` 的 `substate_rules_v4` 在压缩前有 `58` 条，压缩后保留 `34` 条；压缩参数为 `max_rules = 48`、`max_rules_per_substate = 2`、`max_rules_per_event_action = 8`。validation 报告显示索引式验证缓存了 `524` 条 trace，并处理 `34` 条规则、`2` 类状态事件。

`cycle043` 的闭环规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_cont_210k_to250k_seed123_20260508/cycle_043/rules/library_closed_loop_cycle_043.json
```

该库共 `12` 条规则，其中 `7` 条保持 `v8_replay_safe_candidate`，`5` 条为 `cycle_candidate`。本轮闭环更新结果为 `no_rule_added`：共有 `4` 条周期证据候选，但没有候选满足上线门槛；已有周期候选继续作为滚动证据保留。随后 `cycle044` 已加载 `cycle043` 规则库继续训练，说明真实在线闭环在修正后可以无中断进入下一周期。

`cycle044` 从约 `215k` 接续至约 `220k`，三种子训练完成并保存 `final.full.pt`。该周期同样未执行 formal eval，其作用是继续积累 SD 归因与规则证据。后处理完整生成 `sd_explainer_cycle_044.pt`、反事实归因、规则候选、validation 报告和闭环规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_cont_210k_to250k_seed123_20260508/cycle_044/rules/library_closed_loop_cycle_044.json
```

`cycle044` 的候选压缩从 `80` 条降至 `48` 条，达到当前全局上限；validation 缓存 `1053` 条 trace，处理 `48` 条规则和 `2` 类状态事件。闭环规则库共 `13` 条规则，其中 `8` 条为 `v8_replay_safe_candidate`，`5` 条为 `cycle_candidate`。

本周期 `closed_loop_update.action` 仍为 `no_rule_added`，表示没有直接从当期候选中新加入一条上线规则；但滚动验证将 `rotation_risk|omega_neg|vx_mid|support_some|rear->prefer:knee_neg` 晋升为 `v8_replay_safe_candidate`。这说明修正后的跨周期证据累积机制已经在真实在线运行中生效：规则可以不是单周期直接上线，而是在多个周期积累足够支持后进入在线 prior。随后 `cycle045` 已使用 `cycle044` 规则库继续训练，下一关键节点为约 `225k` 的标准 `10ep` formal eval。

`cycle045` 到达约 `225k` 时，三种子均保存 `225000` 与 `225044` checkpoint，并写出标准 `10ep` formal eval：

```text
seed1 = 126.698
seed2 = 253.115
seed3 = 168.229
mean3 = 182.681
```

与同口径历史结果相比，`200k` eval-only v8_safe 均值为 `144.026`，`205k` 为 `178.394`，`210k` 为 `180.607`，本次 `225k` 为 `182.681`。因此，`210k -> 225k` 续训没有出现明显崩塌，并维持了第二轮 smoke 后的分数平台；但增幅很小，不能解释为规则已经带来稳定任务级收益。后续仍应继续观察到 `250k`，并结合 `cycle045` 及后续规则库变化判断规则质量是否持续改善。

随后根据实验解释逻辑重新审查该续训线。仅观察 `rule_on` 多种子均值是否随训练预算增长，不能证明规则有效，因为 TD-MPC2 本身也会随预算推进而改变性能。该线最多说明闭环工程没有明显崩塌，不能作为相对于 TD-MPC2 的因果收益证据。

因此，`210k -> 250k` 接续实验在 `cycle045` 后处理中止。中止时已完成约 `225k` formal eval 和 checkpoint 保存，但 `cycle045` 的 SD 反事实及后续规则库更新未作为正式结果继续解释。后续实验口径改为：规则生成机制通过逻辑审查和 smoke test 后，应从头开始重跑 `100k`，并与官方 TD-MPC2 同预算结果进行对比，同时执行轨迹审查。由于官方参考记录没有 `50k` 口径，`50k` 不再作为主测试预算。接续训练可作为工程调试手段，但不再作为规则有效性的主要证据。

### Hopper 100k 正式协议整理

已新增正式协议文档：

```text
docs/sd_retrospective_rule_induction/hopper_100k_formal_protocol.md
```

该协议将正式主测试定义为空规则库从头闭环训练至 `100k`：不传入 `--initial-resume`、`--initial-sd-checkpoint` 或 `--initial-rule-library`，第一周期默认关闭规则 prior，后续周期只读取闭环生成并满足上线门控的规则。这样可以避免把 `200k` 或其他高预算轨迹中离线挖出的规则泄漏到 `0-100k` 主测试中。

若需要测试一个预先离线审查过的规则库，协议将其定义为 seeded-rule 变体，必须单独报告，并明确规则来源。该变体不能与空库主测试混合解释。

官方 TD-MPC2 `100k` 参考值固定记录为：

```text
TD-MPC2 official, 100k: mean = 13.63, population_std = 10.41
```

来源文件为：

```text
docs/neurips/paper/Attachments/data/hopper/mechanism_oriented_200k/hopper_public_baseline_comparison.csv
external/tdmpc2/results/tdmpc2/hopper-hop.csv
```

同时对 `run_hopper_v7_multiseed_shared_rules.py` 做了一个最小元数据修正：空初始规则库创建时会把本次运行的 `rule_required_status` 写入 `status_meaning`，但不覆盖已有 `replay_validated / rolling_validated` 解释。语法检查已通过：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

随后整理正式 100k readiness 报告：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_formal_100k_readiness_20260508.md
```

结论为：当前 v8-safe 闭环实现已经通过工程 smoke，可作为从头 `100k` 正式主测试的候选实现；但主测试必须采用空初始规则库，不能把从 `200k` 高预算轨迹中挖出的 v8-safe 规则作为 `0-100k` 主实验的初始规则。若要测试预置规则库，只能作为 seeded-rule 变体单独报告。

### Hopper v8 formal empty-library 100k run 启动

按正式协议启动空规则库从头 `100k` 主测试：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_empty_seed123_20260508
exp_prefix  = tdmpc2_v8formal_100k_empty_seed123_20260508
seeds       = 1,2,3
gpus        = 1,2,3
cycles      = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
```

启动命令未传入 `--initial-resume`、`--initial-sd-checkpoint` 或 `--initial-rule-library`。脚本创建的空初始规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_empty_seed123_20260508/shared_rules/library_cycle_000_empty.json
```

`cycle001` 三种子均从零开始，日志确认：

```text
world_model_kind = tdmpc2
sd_online_supervision_enabled = false
rule_prior_enabled = false
resume_from = none
eval_episodes = 10
checkpoint_at_seed = false
```

`cycle001` 已完成三种子训练和 replay 导出。每个 seed 导出 `10` 条 replay episode，manifest 索引 `3` 个 run，轨迹选择为 `17 / 30`，归因 `17` 条且 `missing = 0`。第一轮 SD explainer 拟合使用 `8483` 条 transition，划分为 `7485` train 和 `998` validation。反事实分析生成 `51` 条 state record，规则管线生成 `1` 条 state-action 候选和 `4` 条 substate 候选。闭环更新结果为 `no_rule_added`，输出规则库为空，符合空库首轮预期。

`cycle002` 已从三个 `cycle001 final.full.pt` 接续启动。日志确认 prior 入口已安装，但当前 `rules = 0`：

```text
Installed v6 rule prior: rules=0 mode=penalty_only weight=0.05
```

这属于预期状态：正式主测试允许 prior 入口从第二周期开始接入，但只有当本次 `0-100k` 运行内部生成并晋升出 `v8_replay_safe_candidate` 后，才会产生实际动作先验。

`cycle002` 已完成三种子训练和周期总结。该周期导出每个 seed `20` 条 replay episode，manifest 索引 `cycle001-002` 共 `6` 个 run，轨迹选择为 `47 / 90`，归因 `47` 条且无缺失。SD explainer 从 `cycle001` checkpoint 接续初始化，使用 `23453` 条 transition。反事实分析生成 `141` 条 state record，规则管线生成 `12` 条 substate 候选。闭环库包含 `3` 条 `cycle_candidate`，尚无 `v8_replay_safe_candidate` 上线规则：

```text
rotation_risk|omega_pos|vx_mid|support_some|rear->prefer:knee_pos
unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:hip_neg
unsupported_high_torque|omega_neg|vx_low|support_none|rear->prefer:hip_neg
```

`cycle003` 已从三个 `cycle002 final.full.pt` 接续启动。当前解释为：空库正式测试已开始自然积累候选规则，但尚未形成可上线规则；这符合保守滚动门控的预期，不应视为运行失败。

`cycle003` 已完成周期总结，并首次从本次 `0-100k` 运行内部晋升出 `1` 条可上线规则：

```text
unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:hip_neg
status = v8_replay_safe_candidate
```

该规则由 `cycle002` 中携带的候选跨周期重复出现后通过滚动验证晋升。`cycle003` 闭环规则库状态为：

```text
v8_replay_safe_candidate = 1
cycle_candidate = 2
```

`cycle004` 已从 `cycle003` checkpoint 接续启动，并且三个 seed 的训练日志均确认规则 prior 实际加载：

```text
Installed v6 rule prior: rules=1 mode=penalty_only weight=0.05
```

因此，本次空库正式主测试已经通过一个关键运行检查：规则不是从外部高预算规则库预置，而是在 `0-100k` 运行内部由闭环归因、候选保留和滚动验证自然生成，并从 `cycle004` 开始参与 planner 排序。

`cycle004` 已完成周期总结。该周期索引 `cycle002-004` 共 `9` 个 run，轨迹选择为 `130 / 270`，归因 `130` 条且无缺失。SD explainer 从 `cycle003` checkpoint 接续初始化，使用 `50000` 条 transition，反事实分析生成 `390` 条 state record。规则管线生成 `4` 条 state-action 候选、`1` 条 reliable rule 和 `16` 条 substate 候选，但闭环更新结果为 `no_rule_added`。`cycle004` 闭环库包含 `1` 条 `v8_replay_safe_candidate` 和 `5` 条 `cycle_candidate`；进入 `cycle005` 时仍只有 `1` 条规则参与动作先验。

`cycle005` 已完成 `25k` 正式保存点，并写出标准 `10ep` eval。三个 seed 均保存 `25000.full.pt`，随后在 `25004` 步完成 eval 和 `final.full.pt` 保存。首次 `25k / 10ep` 评估结果为：

```text
seed1 = 0.000000
seed2 = 0.009519
seed3 = 0.000096
mean  = 0.003205
population_std = 0.004465
sample_std     = 0.005468
```

该结果仅记录为早期过程信号，不作为本方法有效性的正式结论。正式判断仍以 `100k` 终点评估、规则库演化、动作先验命中情况和轨迹审查共同决定。

`cycle005` 周期总结已完成。该周期索引 `cycle003-005` 共 `9` 个 run，轨迹选择为 `178 / 390`，归因 `178` 条且无缺失。SD explainer 从 `cycle004` checkpoint 接续初始化，使用 `50000` 条 transition，反事实分析生成 `534` 条 state record。规则管线生成 `5` 条 state-action 候选、`0` 条 reliable rule 和 `20` 条 substate 候选；闭环更新结果仍为 `no_rule_added`。`cycle005` 闭环库状态为：

```text
v8_replay_safe_candidate = 1
cycle_candidate = 4
cycle_supported = 1
cycle_weakened = 1
```

因此，进入 `cycle006` 时仍只有 `unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:hip_neg` 作为上线规则参与动作先验，其余规则仍处于候选、被支持或被削弱状态。

`cycle010` 已到达 `50k` 正式保存点并写出标准 `10ep` eval。三个 seed 均保存 `50000.full.pt`、`50009.full.pt` 和 `final.full.pt`。`50k / 10ep` 评估结果为：

```text
seed1 = 0.615573
seed2 = 0.000051
seed3 = 0.000612
mean  = 0.205412
population_std = 0.290028
sample_std     = 0.355210
```

其中 seed1 的 `10ep` 均值主要来自单个 episode 的非零回报，seed2 和 seed3 仍接近零。因此该节点只能说明相较 `25k` 出现了弱过程信号，不能解释为稳定性能提升。`cycle009` 进入 `cycle010` 前的闭环规则库已有 `2` 条 `v8_replay_safe_candidate` 上线规则：

```text
unsupported_high_torque|omega_neg|vx_low|support_none|rear->prefer:knee_pos
rotation_risk|omega_pos|vx_low|support_some|prone->prefer:hip_pos
```

`cycle010` 的周期总结已进入 replay 导出、轨迹选择和 SD 接续拟合阶段，后续仍需检查本周期是否新增或削弱上线规则。

### 规则观察层放宽实现

针对 `cycle001-010` 中候选规则持续产生但上线数量偏少的问题，已将后续变体的候选观察层与在线上线层分离。修改包括：

```text
experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

`closed_loop_rule_update.py` 新增 `--rolling-weaken-max-good-share`。`rolling_max_good_share` 继续作为晋升为在线状态的门槛，默认仍为 `0.25`；`rolling_weaken_max_good_share` 控制候选何时被削弱退出滚动池。多种子共享规则脚本的后续默认值调整为：

```text
max_cycle_candidates          = 6
rolling_max_good_share        = 0.25
rolling_weaken_max_good_share = 0.40
```

因此，`good_share` 位于 `0.25-0.40` 的候选不会上线，但可以继续跨周期积累证据。该修改不影响当前正在运行的正式 `100k` 主实验；该实验仍按启动时的 `max_cycle_candidates = 3`、`rolling_max_good_share = 0.25` 解释。

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

## 2026-05-13

### 阶段式规则类型的初版接入

在 `hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2` 最终代正式 eval 中，`toe_recovery_low_support->soft_prefer:waist_pos` 已能改变触发状态下的 `action_0` 分布，但未能稳定形成全足支撑恢复。该结果表明，脚尖恢复问题更适合表达为“小阶段进程规则”，而不是单步动作族规则。

据此新增 `stage_effect_prior` 规则类型。该类型仍使用现有规则库、`confidence_active / confidence_observing / confidence_retired` 状态和 rule-effect confidence 生命周期，但在线执行方式改为对 MPPI 候选动作序列进行短视界效果评分。首个目标为：

```text
toe_recovery_low_support -> stage_effect_prior: recover_full_foot_support
```

实现内容包括：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

`online_rule_prior_v6.py` 现在允许 `source=sd_stage_effect` 且 `intervention_type=stage_effect_prior` 的规则进入在线 prior。该规则不读取 `action_family`，而是在当前状态命中后，对 `actions[horizon, num_samples, action_dim]` 做短视界 SD/TD-MPC2 world-model 推演，并按阶段目标计算候选序列 bonus。若在线环境可提供 TD-MPC2 的 SD world model，则使用已训练的 `model.next` 和 `_dynamics._state_reward_features`；否则退回本地 `DMControlHopperSDModel` 做 smoke 级推演。

`audit_hopper_v8_widefunnel_rule_effects.py` 增加阶段式审计口径。对实际轨迹中的命中状态，审计脚本向后查看 `horizon` 步，计算阶段目标分数；若分数超过 `audit_positive_threshold`，记为有效；若低于 `audit_harmful_threshold`，记为有害；其余为中性。输出字段仍保留 `selected_gate_hits`、`direction_consistent_rate`、`direction_harmful_rate`、`good_state_hit_rate` 和 `high_episode_hit_rate`，因此可直接被 `apply_rule_effect_confidence.py` 使用。

最小验证结果：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py

stage prior smoke:
  rules = 1
  hit = 1.0
  bonus_abs_max = 0.000838

seed2 final eval stage audit smoke:
  selected_gate_hits = 1629
  stage_effective = 650
  stage_harmful = 820
  direction_consistent_rate = 0.3990
  direction_harmful_rate = 0.5034
```

最后一项结果不是新规则的失败结论，而是对旧最终策略轨迹的回放审计：它说明现有高分脚尖推进轨迹并未真正完成阶段式支撑恢复，因此阶段式规则审计能把“动作方向看似生效但阶段目标没有完成”的差异暴露出来。

## 2026-05-13

### 脚尖恢复阶段的 SD 诊断规则

新增离线诊断脚本：

```text
experiments/sd_retrospective_rule_induction/diagnose_hopper_toe_recovery_rule.py
```

该脚本用于检验一个具体失败归因假设：在 Hopper 高分轨迹中，agent 经常处于低身位或俯身恢复状态，脚尖已经可发力但脚跟和全足支撑尚未形成；此时若立即弹射，后续常继续俯身或失衡。诊断流程不修改训练 checkpoint 或规则库，而是读取 `cycle020` 的 SD explainer 和 100k 高分轨迹，抽取 `touch_toe>=0.80`、`touch_heel<0.50`、`full_foot_support<0.35`、尚未 `grounded_prepared/upright` 且存在低身位、俯身或旋转风险的状态。随后用 SD 对实际动作和探针动作进行 `15` 步结构态推演，比较候选动作是否更有利于支撑恢复、准备姿态、俯身降低和旋转控制。

默认诊断命令为：

```text
python experiments/sd_retrospective_rule_induction/diagnose_hopper_toe_recovery_rule.py
```

默认输入为：

```text
sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

trace_source =
logs/hopper-hop/*/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/
```

默认输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_100k_toe_recovery_sd_diagnostic.json
results/sd_retrospective_rule_induction/reports/hopper_100k_toe_recovery_sd_diagnostic.md
```

本次诊断自动选择 100k 高分 official eval 与 retrospective replay 轨迹中的前 `8` 条，均来自 seed3，回报范围约为 `222.716` 至 `245.375`。共抽取 `192` 个目标状态，观测上这些状态的下一步释放率为 `1.000`，后续 `15` 步进入或保持俯身的比例为 `0.984`，后续出现全足支撑或准备姿态的比例为 `0.000`。这说明该状态谓词确实捕捉到了“脚尖可发力但尚未恢复支撑就进入弹射”的失败窗口。

SD 反事实诊断形成了一条候选规则：

```text
toe_recovery_low_support->delay_release_and_restore_full_foot_support
```

在 `192` 个目标状态中，`98` 个状态由 SD 支持该规则，支持率为 `0.510`。支持样本的平均结构态优势为 `0.903`，平均 `support_contact/full_foot_support` 增益为 `0.160`，平均准备姿态增益为 `0.061`，平均俯身降低为 `0.194`，平均旋转风险降低为 `0.084`。SD 最常偏好的动作族为 `waist_pos`、`knee_pos`、`knee_neg`，其中 `waist_pos` 的支持数为 `40`，`knee_pos` 为 `17`，`knee_neg` 为 `14`。

该结果支持方法论判断：这条规则虽然可以由人工观察直接写出，但 SD 也能在给定状态谓词和恢复目标后，通过短视界结构态推演给出一致的诊断证据。需要保守解释的是，当前脚本尚未证明自动状态谓词发现已经完成；它验证的是“SD 能否对人工锁定的失败窗口形成预期规则”。下一步若要接入在线训练，应将其作为可被 confidence 机制增强、削弱或退休的 `offline_sd_diagnostic_candidate`，而不是不可变的手工硬规则。

### 脚尖恢复规则接入 100k 至 200k 接续实验

已中止原 `hopper_v8formal_200k_confidence_candidate_seed123_20260512_r1` 接续实验。该实验在中止时已有主控进程和 `cycle040` 三个训练进程运行；由于下一步需要检验新 SD 诊断规则的增量作用，继续原实验不再符合当前对照目的。

新增在线 prior 兼容逻辑：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
```

修改内容为：允许 `source=sd_diagnostic_toe_recovery` 且 `intervention_type=soft_prefer_action_family/avoid_action_family` 的规则被加载；在在线特征抽取中增加 `toe_recovery_low_support` 谓词。该谓词要求脚尖接触强、脚跟接触弱、全足支撑不足、尚未进入准备或直立状态，并伴随俯身、低身位或旋转风险。该修改只增加新规则族的匹配，不改变既有 v4 substate 和 v8 transition 规则行为。

将离线 SD 诊断规则转成当前轻量在线 prior 可执行的一条动作族投影：

```text
toe_recovery_low_support->soft_prefer:waist_pos
```

输出规则库为：

```text
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json
```

之所以只激活 `waist_pos`，是因为 15 步 SD 诊断中虽然还出现 `knee_pos` 和 `knee_neg`，但二者在当前动作族 prior 下会形成直接方向冲突；`waist_pos` 是支持数最高且无同关节反向冲突的投影。该规则以 `confidence_active` 进入初始规则库，`confidence=0.75`，`effective_prior_scale=0.75`，后续仍接受 rule-effect confidence 的增强、削弱或退休。

已启动新的三种子 100k 至 200k 接续实验：

```text
output_root =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_seed123_20260513_r1

exp_prefix =
tdmpc2_v8formal_200k_confidence_toerecovery_seed123_20260513_r1

initial_rule_library =
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

initial_resumes =
seed1/seed2/seed3 的 100k cycle020 final.full.pt
```

训练配置沿用上一轮 100k 至 200k 接续口径：`cycle_steps=5000`，`start_cycle=21`，`cycles=40`，每 `25k` 做标准 `10ep` formal eval 和 checkpoint 保存，SD 拟合、归因、规则更新仍每 `5k` 执行。启动确认显示，三个种子均从 `step=100020` 恢复，并加载 `48` 条 `confidence_active` 规则，其中包含新增脚尖恢复规则。

### 脚尖恢复规则审计缺口修补与重跑

检查 `hopper_v8formal_200k_confidence_toerecovery_seed123_20260513_r1` 后发现，`toe_recovery_low_support->soft_prefer:waist_pos` 规则在在线 prior 中已被加载并保持 `confidence_active`，但 `cycle021` 和 `cycle022` 的 `rule_effect_audit.json` 中没有该规则的逐条记录。原因不是训练轨迹中缺少对应状态，而是 `audit_hopper_v8_widefunnel_rule_effects.py` 尚未同步支持 `sd_diagnostic_toe_recovery` 来源、`toe_recovery_low_support` 状态谓词和该规则的无高扭矩门控。

已修补审计脚本，使其在加载规则、状态匹配和候选动作门控三个位置与 `online_rule_prior_v6.py` 对齐。由于 replay trace 使用 `rootz`、`foot_grounded` 等字段，而在线结构特征使用 `torso_z`、`support_contact`，审计脚本同时增加了字段别名读取，避免同一语义特征在 eval trace 与 replay trace 之间出现缺省值偏差。

修补后使用合成样例执行 smoke test，确认 `toe_recovery_low_support` 状态、规则匹配、门控和 `waist_pos` 动作方向判断均可命中。随后用旧接续实验的真实 `cycle021` 和 `cycle022` replay trace 重跑审计，结果显示该规则已进入逐条审计：

```text
cycle021:
  state_hits = 51034
  selected_gate_hits = 51034
  same = 27717
  opposite = 15324
  direction_consistent_rate = 0.6997
  direction_harmful_rate = 0.3003

cycle022:
  state_hits = 53847
  selected_gate_hits = 53847
  same = 29384
  opposite = 16003
  direction_consistent_rate = 0.7028
  direction_harmful_rate = 0.2972
```

使用修补后的 `cycle022` 审计结果执行 confidence 更新，脚尖恢复规则不再触发 `missing_from_effect_audit_absence_decay`，而是记录为 `effect_observed_no_large_change`，保持 `confidence_active`，置信度保持在 `0.735`。这说明此前置信度下降属于审计缺口造成的伪缺席信号。

已停止旧口径接续实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_seed123_20260513_r1
```

并以修补后的审计口径从 100k checkpoint 重新启动 100k 至 200k 三种子接续实验：

```text
output_root =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2

exp_prefix =
tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2

start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_allowed_statuses = confidence_active
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
```

启动确认显示主控进程和 seed1、seed2、seed3 三个 `cycle021` 训练进程均已运行，且均从 `tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019` 的 `final.full.pt` 恢复，并加载包含脚尖恢复规则的初始规则库。

## 2026-05-10

### confidence mid r2 的空介入问题与候选置信度修订

`hopper_v8formal_100k_confidence_mid_seed123_20260510_r2` 在 `cycle001` 完成后被中止。该周期并非完全没有规则信号：v4 substate 分支产生 `10` 条规则，其中 `7` 条为 `cycle_candidate`；但本轮上线口径只允许 `confidence_active`，而 rule-effect confidence 后处理未覆盖 `cycle_candidate`。因此这些候选规则虽被保留在规则库中，却无法进入置信度系统，导致 `cycle002` 虽显示 `rule_prior_enabled=true`，实际安装规则数为 `0`。

该现象说明问题不在于完全缺少可提取规则，而在于门控链条过窄。继续运行会退化为空规则库训练，不符合本轮验证规则价值的目的。

已修订 `run_hopper_v7_closed_loop.py`：rule-effect audit 和 confidence update 现在同时覆盖 `cycle_candidate`、`cycle_supported`、`rolling_validated`、`v8_replay_safe_candidate`、`confidence_active` 和 `confidence_observing`。训练期在线 prior 仍只读取 `confidence_active`。因此，候选规则不会被无条件放开，而是先接受行为回放审查和 confidence 状态转换，只有达到 active 门槛后才作用于下一周期规划排序。

使用 r2 `cycle001` 的真实产物进行离线 smoke，结果为：

```text
input rules = 7 cycle_candidate
rule-effect audit rules = 7
gate_hit_frac = 0.3417
confidence_active = 4
confidence_observing = 3
```

该结果表明修订后的口径能够避免“有候选但无法上线”的空介入问题，同时保留 harmful rate、冲突仲裁和 confidence 衰减机制。

已按该口径从头启动 `hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3`。`cycle001` 完成后，规则更新结果为：

```text
substate rules = 14
library_closed_loop_cycle_001_confidence rules = 5
confidence_active = 3
confidence_observing = 2
rule-effect gate_hit_frac = 0.2004
```

两条 `falling` 状态下腰部相反方向规则因 active conflict 仲裁降为 `confidence_observing`；三条 `rear` 状态下 `waist_pos` 规则保持 `confidence_active`。`cycle002` 三个种子训练日志均确认实际安装 `rules=3`，说明该修订已消除空规则库训练问题。

`r3` 已完成 `100k` 三种子闭环训练。最终规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/rules/library_closed_loop_cycle_020_confidence.json
```

最终规则状态为：

```text
total rules = 97
confidence_active = 47
confidence_observing = 29
confidence_retired = 21
v4_substate rules = 89
v8_transition_replay_safety rules = 8
active v8_transition_replay_safety rules = 4
```

标准 `10ep` formal eval 结果如下：

```text
25k:  seed1=0.0016, seed2=0.0094,  seed3=0.4287,  mean=0.1466
50k:  seed1=0.6521, seed2=0.2165,  seed3=51.9109, mean=17.5932
75k:  seed1=0.9074, seed2=0.0498,  seed3=53.8062, mean=18.2544
100k: seed1=0.7227, seed2=42.7913, seed3=70.1488, mean=37.8876
```

该结果表明，候选规则进入 confidence 系统后，闭环机制不再空转，规则数量、active 数量与 transition 规则均持续增长；最终均值高于既有 `100k` official TD-MPC2 参考均值 `13.63`。但结果仍高度依赖种子，seed1 仍未形成有效策略。因此，当前结论应表述为：规则机制产生了可测的正向训练信号，但尚未形成跨种子稳定收益。

对 `cycle020` 最终规则命中进行审查后，规则库总体表现为健康但仍需细化。`rule_effect_audit` 覆盖 `600` 条 replay episode，任一规则 gate 命中比例为 `0.4396`，规则命中密度与回报相关为 `0.2736`，transition 命中密度与回报相关为 `0.9354`。按最终状态过滤后，`47` 条 active 规则的总 gate hits 为 `224050`，平均方向一致率为 `0.7049`，平均 harmful rate 为 `0.2951`；active 中 harmful rate 超过 `0.45` 的规则只有 `1` 条，且没有 active 规则超过 `0.50`。相对地，observing 规则平均 harmful rate 为 `0.4440`，retired 审计命中的规则平均 harmful rate 为 `0.6317`，说明 confidence 机制确实在把高风险规则推向观察或退休状态。

transition 规则是最强的正向信号。最终 active transition 规则为 `4` 条，均来自 `v8_transition_replay_safety`，其中高回报 episode 命中率在 `0.484-0.675` 之间。按 episode 回报分桶，transition 命中密度从低回报 episode 的 `0.008` 上升到 `100+` 回报 episode 的 `0.379`，与回报呈强相关。该结果支持“阶段转接规则比单纯失败恢复规则更接近 Hopper 有效行为”的判断。

剩余风险主要有三点。第一，最终 active 规则中仍有一条边界规则 `rotation_risk|omega_pos|vx_mid|support_none|prone->prefer:waist_neg`，harmful rate 为 `0.455`，需要下一轮优先削弱或拆分。第二，若干高命中 v4 substate 规则覆盖大量 `falling` 状态，但高分 episode 命中率较低，说明其更多是恢复/失败区间规则，不足以解释成功步态。第三，seed1 的规则命中密度并不低但回报仍低，说明规则命中本身不是充分条件，还需要检查 transition 规则是否真正覆盖关键转接窗口。

## 2026-05-12

### 从 100k 接续至 200k

基于 `r3` 的 `100k` 结果，后续目标调整为验证自主规则框架能否在轻量介入下显著提高训练效率，而不是继续细化 Hopper 规则本身。由于 `100k` 均值 `37.89` 已明显高于既有 TD-MPC2 official `100k` 参考均值 `13.63`，且最终规则命中审查显示 active 规则总体健康，因此决定直接接续训练到 `200k`，观察较长预算下的趋势。

已启动接续实验：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_candidate_seed123_20260512_r1
exp_prefix = tdmpc2_v8formal_200k_confidence_candidate_seed123_20260512_r1
start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
```

接续输入为：

```text
initial_rule_library = r3/cycle_020/rules/library_closed_loop_cycle_020_confidence.json
initial_sd_checkpoint = r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
seed1 resume = r3/cycle020 seed1 final.full.pt
seed2 resume = r3/cycle020 seed2 final.full.pt
seed3 resume = r3/cycle020 seed3 final.full.pt
```

启动验证结果为：三种子均从 `step=100020`、`episode=200`、`total_updates=98019` 恢复；`cycle021` 三个训练日志均显示安装 `47` 条规则：

```text
Installed v6 rule prior: rules=47 mode=penalty_only weight=0.05
```

因此，该实验确认为从 `100k` 结果接续，而非重新开始；已有规则库和 SD 解释器均被延用。

### confidence 中间门控 r2 启动

`hopper_v8formal_100k_confidence_safe_seed123_20260510_r1` 运行至 `cycle006` 后中止。中止原因是：截至 `25k`，`cycle001` 至 `cycle005` 均未形成任何 `confidence_active` 规则，导致训练在四分之一预算内基本退化为纯 TD-MPC2，不能有效检验规则闭环价值。

已完成的 `25k` formal eval 为：

```text
seed1 = 0.0685
seed2 = 0.0301
seed3 = 0.2159
mean  = 0.1049
```

基于 r1 `cycle005` 的真实规则库和 rule-effect audit 做参数 smoke 后，采用中间门控：

```text
confidence_default = 0.56
confidence_active_threshold = 0.55
confidence_positive_context_threshold = 0.0
confidence_harmful_threshold = 0.45
confidence_hard_harmful_threshold = 0.50
confidence_hard_harmful_min_hits = 100
```

该口径在 r1 `cycle005` smoke 中产生 `2` 条 `confidence_active`，而不是严格口径的 `0` 条或旧口径的过多规则上线。相反有效动作方向冲突仲裁和硬下线机制保持启用。

随后启动新的从头 `100k` 三种子实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_mid_seed123_20260510_r2
```

启动命令为：

```text
setsid -f bash -c 'python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_100k_confidence_mid_seed123_20260510_r2 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_mid_seed123_20260510_r2 --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses confidence_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --v8-max-cycle-transition-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --confidence-default 0.56 --confidence-active-threshold 0.55 --confidence-positive-context-threshold 0.0 --confidence-harmful-threshold 0.45 --confidence-hard-harmful-threshold 0.50 --confidence-hard-harmful-min-hits 100 --confidence-hard-harmful-margin 0.05 --confidence-conflict-keep-confidence-margin 0.15 --confidence-conflict-keep-harmful-margin 0.10 --confidence-conflict-demote-margin 0.05 --enable-v8-transition-rules --enable-rule-effect-confidence > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_mid_seed123_20260510_r2/launcher.log 2>&1'
```

启动确认结果为：主控进程和 seed1、seed2、seed3 三个训练进程均存在；第一周期仍使用空规则库，且 `rule_prior_enabled=false`。

### rule-effect confidence 安全门控修订

100k 终点诊断后，规则机制审查确认当前 confidence 更新存在三类风险：有害规则可在多周期内继续 active，方向一致性不等于性能收益，同状态相反有效动作方向可共同上线并压制必要的大动作。因此，对 confidence 更新做安全修订。

代码修改包括：

```text
experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

修订内容为：新规则 `default_confidence=0.50`；若 `selected_gate_hits>=100` 且 `direction_harmful_rate>=0.45`，规则触发 `effect_harmful_deactivated` 并降到 active 阈值以下；方向一致规则只有在 `good_state_hit_rate` 或 `high_episode_hit_rate` 达到 `0.05` 时才允许增强；同一状态同一关节的相反有效动作方向在输出前仲裁，没有明确优势时降为 `confidence_observing`。`avoid:hip_neg` 与 `soft_prefer:hip_pos` 在仲裁中视为同一有效方向。

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 1 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --exp-prefix dry --output-root /tmp/hopper_dry --enable-rule-effect-confidence --dry-run
python experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_020/rules/library_closed_loop_cycle_020.json --audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_020/reports/rule_effect_audit.json --output results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/library_cycle020_confidence_patched_smoke.json --target-status v8_replay_safe_candidate --target-status confidence_active --target-status confidence_observing --target-status confidence_retired
```

旧 `cycle020` 离线 smoke 结果为：active 规则数由 `31` 降为 `24`，`effect_harmful_deactivated=5`，`active_conflict_demoted=2`，active 规则中的相反有效方向冲突为 `0`。

### confidence 安全门控 100k 从头重跑启动

基于上述修订，启动新的从头 `100k` 三种子闭环实验。该实验不接续旧 checkpoint，不继承旧规则库，用于检验修订后的规则机制是否能避免训练期规则偏置继续写入策略。

实验目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1
```

启动命令为：

```text
setsid -f bash -c 'python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_100k_confidence_safe_seed123_20260510_r1 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1 --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses confidence_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --v8-max-cycle-transition-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --confidence-default 0.50 --confidence-active-threshold 0.65 --confidence-positive-context-threshold 0.05 --confidence-hard-harmful-threshold 0.45 --confidence-hard-harmful-min-hits 100 --confidence-hard-harmful-margin 0.05 --confidence-conflict-keep-confidence-margin 0.15 --confidence-conflict-keep-harmful-margin 0.10 --confidence-conflict-demote-margin 0.05 --enable-v8-transition-rules --enable-rule-effect-confidence > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1/launcher.log 2>&1'
```

启动后确认结果为：主控进程和 seed1、seed2、seed3 三个训练进程均存在；`cycle001` 三个训练进程分别使用 GPU1、GPU2、GPU3；第一周期 `rule_prior_enabled=false`，并加载空规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1/shared_rules/library_cycle_000_empty.json
```

### v8 transition phase-any widefunnel 100k 完成

`hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1` 已完成全部 `20` 个周期，终点为 `100019` steps。该实验从空规则库启动，三种子共享规则库，SD 回归、失败归因和规则更新每 `5k` 执行一次，checkpoint 与标准 `10ep` formal eval 每 `25k` 执行一次。

最终 `100k / 10ep` 评估结果为：

```text
seed1 = 72.312993
seed2 = 9.343264
seed3 = 165.685288
mean  = 82.447182
population_std = 64.227375
```

按既有公开对照文件，TD-MPC2 official `100k` Hopper 参考均值为 `13.63`，`200k` 参考均值为 `166.63`。因此，本轮在 `100k` 三种子均值上高于 official `100k` 参考，但仍不能解释为 Hopper 已稳定解决：seed 间差异很大，且终点轨迹仍显示大量 prone/rear 状态。

最终规则库为：

```text
rules total = 78
v8_replay_safe_candidate = 20
cycle_candidate = 37
cycle_supported = 8
cycle_weakened = 13
```

其中实际可被在线 prior 读取的 `20` 条规则包括 `13` 条 `v4_substate` 恢复规则和 `7` 条 `v8_transition_replay_safety` 规则；干预类型为 `16` 条 `soft_prefer_action_family` 和 `4` 条 `avoid_action_family`。这说明本轮 widefunnel 设置确实缓解了此前 transition 分支长期不上线的问题，规则发现链条不再退化为少量恢复规则的重复使用。

终点轨迹审查产物为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_final_audit_20260509.json
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_final_audit_20260509.md
```

审查覆盖三种子共 `30` 条 official eval episode。总体轨迹统计为：

```text
reward_sum = 2473.415459
structural_success_frac = 0.0474
dense_pose_high_frac = 0.0005
positive_reward_frac = 0.1882
mean_continue_score = 0.2716
rear_frac = 0.2680
prone_frac = 0.6598
upright_frac = 0.0001
```

最好的片段出现在 seed1 的若干高分 episode 中，例如 seed1 episode3 的 `frames=138-156`，长度 `19`，片段内 `structural_success_frac=0.474`、`mean_torso_z=0.785`、`mean_abs_pitch=0.224`、`mean_vx=1.061`。这说明终点策略已经出现了短暂较理想姿态和正向速度片段，但尚未把这些片段稳定衔接为完整步态。

当前阶段性解释为：widefunnel 修订使规则发现和 transition 规则上线显著改善，且 `100k` formal eval 出现高于 official `100k` 参考的均值信号；但由于 seed2 仍低、轨迹总体 prone 比例高、理想姿态片段较短，下一步应继续审查 v8 transition 规则在高分片段前后的命中与动作方向，而不是直接宣称任务级性能提升。

### widefunnel 终点规则作用审查

补充实现只读审查脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

该脚本按 `online_rule_prior_v6.py` 的状态匹配逻辑复核最终 official eval 轨迹，并区分两类规则作用：`soft_prefer_action_family` 以“减少相反方向动作”为有效方向，`avoid_action_family` 以“减少被禁止动作同向出现”为有效方向。该审查不能替代 rule-on/off 反事实 eval，但可以判断规则是否在最终行为中实际命中、是否与已选动作方向一致、是否集中在高分 episode。

需要特别说明的是，`cycle020` 训练和 final eval 实际加载的是 `cycle019` 规则库，而不是 `cycle020` 总结后形成的最终规则库。训练日志中 `rule_prior_num_rules=16`，与 `cycle020/cycle_config.json` 中的 `shared_rule_library_before_cycle` 一致。因此，规则作用审查采用：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/cycle_019/rules/library_closed_loop_cycle_019.json
```

审查输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_cycle019_active_20260509.json
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_cycle019_active_20260509.md
```

主要结果如下：

```text
online rules used by final eval = 16
v4_substate rules = 9
v8_transition_replay_safety rules = 7
any-rule selected-gate hit fraction = 0.1415
rule-hit-density / reward correlation = 0.9718
transition-hit-density / reward correlation = 0.9730
```

按命中动作方向统计，`v4_substate` 规则总体较可靠：`1512` 次有效命中中，方向一致率约 `89.6%`。v8 transition 规则命中更密集，共 `5971` 次有效命中，但总体方向一致率约 `63.8%`，内部差异很大。

可靠的 transition 规则主要是：

```text
transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->soft_prefer:waist_pos
  hits = 1373, harmful = 10.5%, high_episode_hit_rate = 94.8%

transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->avoid:waist_neg
  hits = 1373, harmful = 10.5%, high_episode_hit_rate = 94.8%

transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:waist_pos
  hits = 428, harmful = 14.5%, high_episode_hit_rate = 63.3%
```

需要警惕的 transition avoid 规则为：

```text
transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->avoid:hip_neg
  hits = 1373, harmful = 64.6%

transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->avoid:knee_pos
  hits = 1373, harmful = 65.8%
```

这两条规则高频命中且大多出现在高分 episode，但最终动作仍经常选择被它们压制的方向，说明它们要么权重过轻而未能改变 MPPI 排序，要么规则本身把高分片段中的必要动作误判为应避免动作。由于同一状态下还同时存在 `waist_pos` 鼓励和 `waist_neg` 避免规则，本轮 transition 规则的正向信号主要来自腰部方向，而 hip/knee 的 avoid 规则应在下一轮进入降权、削弱或拆分审查。

### rule-effect confidence 更新实现

为把上述规则作用审查转化为可执行的信心更新，新增：

```text
experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
```

该脚本读取规则库和 rule-effect audit 报告，输出新的 confidence-weighted 规则库，不修改原始实验产物。第一版更新规则如下：

```text
1. 命中充分且方向一致率超过 positive_threshold 的规则增强。
2. 命中充分且 harmful rate 超过 harmful_threshold 的规则削弱。
3. 低命中规则执行轻微衰减。
4. 未命中规则执行 absence_decay。
5. 输出 status = confidence_active / confidence_observing / confidence_retired。
6. effective_prior_scale = confidence，供 online_rule_prior_v6.py 缩放规则强度。
```

本轮使用 `cycle019` 实际生效规则库和终点规则作用审查报告执行：

```text
python experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
```

输出为：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8phaseany_widefunnel_cycle019_effect_confidence_20260509.json
results/sd_retrospective_rule_induction/rules/hopper_v8phaseany_widefunnel_cycle019_effect_confidence_20260509.md
```

结果为：

```text
updated rules = 16
confidence_active = 14
confidence_observing = 2
confidence_retired = 0
```

两条被削弱为 `confidence_observing` 的规则正是此前审查中 harmful rate 最高的 transition avoid 规则：

```text
transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->avoid:knee_pos
  confidence = 0.595, harmful = 0.658

transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->avoid:hip_neg
  confidence = 0.611, harmful = 0.646
```

与此同时，`waist_pos` 鼓励和 `waist_neg` 避免规则保持 `confidence_active`。这说明“规则可以被增强、削弱和暂时退出在线激活集”的机制已经可以在当前 widefunnel 产物上运行。下一步若要进入新一轮在线测试，可将 `rule_prior_allowed_statuses` 设为 `confidence_active`，并加载该 confidence 规则库；`confidence_observing` 规则仍保留证据历史，但默认不再参与 MPPI 排序。

### confidence 规则库 eval-only 对比

为检验 rule-effect confidence 后处理是否改善显式规则 prior，新增 eval-only 对比脚本：

```text
experiments/sd_retrospective_rule_induction/eval_hopper_v8_widefunnel_confidence.py
```

该脚本不训练，只重载 widefunnel `100k` 三种子 checkpoint，并在同一 eval-only 口径下比较三种 planner 配置：

```text
rule_off           = 不加载规则 prior
original_cycle019  = 加载 cycle019 实际生效规则库，允许 v8_replay_safe_candidate
confidence_active  = 加载 rule-effect confidence 规则库，只允许 confidence_active
```

执行命令为：

```text
python experiments/sd_retrospective_rule_induction/eval_hopper_v8_widefunnel_confidence.py
```

输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509.md
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/
```

三种子 `10ep` 均值如下：

```text
rule_off:
  seed1 = 54.920
  seed2 = 9.511
  seed3 = 164.172
  mean  = 76.201

original_cycle019:
  seed1 = 53.823
  seed2 = 11.267
  seed3 = 168.659
  mean  = 77.916

confidence_active:
  seed1 = 58.463
  seed2 = 10.110
  seed3 = 169.962
  mean  = 79.511
```

因此，`confidence_active` 相对 `original_cycle019` 的三种子均值变化为 `+1.595`，相对 `rule_off` 为 `+3.310`。分种子看，confidence 相对原规则在 seed1 为 `+4.640`，seed2 为 `-1.157`，seed3 为 `+1.303`。

该结果应解释为弱正向 smoke：削弱两条高 harmful transition avoid 规则后，显式规则 prior 没有退化，且相对原规则略有改善。但提升幅度小于种子间和 episode 间方差，不能作为任务级性能结论。它支持下一步把 rule-effect confidence 机制接入新一轮从头在线闭环测试，而不是仅在后处理层使用。

### rule-effect confidence 在线闭环 100k 重跑启动

已启动新的从头 `100k` 三种子在线闭环实验，用于检验规则效应置信机制在训练过程中的实际价值。本轮不继承旧 checkpoint、旧规则库或终点 confidence 规则库，而是从空规则库开始，使规则生成、规则作用审查、信心增强或削弱、在线 prior 加载均在同一闭环内发生。

首次启动的 `r1` 运行在 `cycle001` 完成后被中止。中止原因是参数审查发现 v8 transition 规则的候选携带量使用了独立参数 `v8_max_cycle_transition_candidates`，默认值仍为 `6`，而本轮协议要求恢复规则与 transition 规则均使用 `12`。该问题已通过代码修正解决，并以新目录 `r2` 重新从空库启动，避免两种口径混入同一结果目录。

运行目录与实验名前缀为：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2
exp_prefix = tdmpc2_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2
```

核心参数如下：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = empty
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = confidence_active
rule_prior_mode = penalty_only
rule_prior_opposite_penalty = 0.01
rule_prior_cap_penalty = 0.01
max_online_rules_per_cycle = 2
max_cycle_candidates = 12
v8_max_cycle_transition_candidates = 12
max_states_per_episode = 10
rolling_weaken_max_good_share = 0.40
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
```

启动后确认主控进程和三种子训练进程均存在。`cycle001` 按预期从空规则库启动，训练命令中 `rule_prior_enabled=false`、`resume_from=null`，规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/shared_rules/library_cycle_000_empty.json
```

`r2` 启动后再次确认主控进程和三种子训练进程均存在，`cycle001` 配置为 `rule_prior_enabled=false`、`resume_from=null`，并从空规则库开始。`run_config.json` 已记录 `v8_max_cycle_transition_candidates = 12`。

### confidence 在线闭环 50k 节点与 transition 候选诊断

`hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2` 已完成 `cycle010 / 50k` 节点并进入 `cycle011`。标准 `10ep` formal eval 在 `25k` 和 `50k` 均值仍低：

```text
25k overall mean = 0.1998
50k overall mean = 0.1892
```

规则系统本身仍在运行并增长。`cycle010` confidence 规则库包含：

```text
total = 50
confidence_active = 16
confidence_observing = 2
cycle_candidate = 25
cycle_weakened = 7
```

所有 active 规则仍来自 `v4_substate` 恢复规则。`cycle010` 的 confidence 更新显示 `8` 条规则因 harmful 证据被削弱，`3` 条规则因一致性证据被增强，说明规则信心机制并非单向累积，而是在持续校正。

本轮最重要的变化是 v8 transition 分支首次形成候选规则。`cycle010` 的 transition 统计为：

```text
success fragments = 30
transition records = 74
observational candidates = 1
v8 transition rules = 1
validation status = heldout_weak
replay-safe exported rules = 0
```

该候选规则为：

```text
transition|transition_gap|omega_pos|vx_high|support_none|phase_any|support_risk -> avoid:waist_neg
```

观测层证据显示，该规则来自 `5` 条支持记录、覆盖 `2` 个种子，指向在 `omega_pos / vx_high / support_none / support_risk` 的 transition gap 状态中避免 `waist_neg`。观测统计为 `mean_continue_delta_next = -0.0787`、`mean_transition_delta_next = -0.0908`、`next_degradation_rate = 1.0`，因此可形成 `offline_v8_supported`。

未能上线的原因在 held-out validation，而不是 replay safety 本身。held-out 子集只有 `1` 条支持记录、覆盖 `1` 个种子，`mean_continue_delta_next = -0.0231`，未达到 avoid 规则所需的 `avoid_max_delta = -0.03`，因此状态为 `heldout_weak`。后续 replay safety 只允许 `heldout_supported` 进入，故该规则没有被导出为 `v8_replay_safe_candidate`。

该结果说明 transition 分支的瓶颈已经从“完全不能产生候选”推进为“候选能产生，但 held-out 支持不足”。下一步应优先审查 held-out 划分和 transition 候选验证口径，而不是简单降低在线门槛。可能方向包括：增加 transition anchor 的 held-out 样本量、对 `pre_degradation / last_success / post_success` 分开验证、或为观测强但 held-out 样本过少的规则设置观察态，而不是直接丢弃。

本轮观察重点限定在规则价值本身：每周期规则生成数量和来源、`confidence_active / confidence_observing / confidence_retired` 的演化、规则效应审查中的方向一致率与 harmful rate、transition 规则是否覆盖高分或近高分片段前后的关键状态，以及规则 prior 是否在动作方向统计中产生可解释影响。`25k / 50k / 75k / 100k` 的标准 `10ep` 回报只作为辅助指标，不作为单独判断规则机制成败的依据。

### v8 transition phase-any 100k 从头重跑启动

在 `phase_any` 规则键离线 smoke 通过后，启动新的从头三种子 `100k` 正式实验。该实验不继承旧规则库或旧 checkpoint，以避免将上一轮未完成实验中的规则状态带入新评估。

实验入口为：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2
exp_prefix = tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = empty
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = v8_replay_safe_candidate
enable_v8_transition_rules = true
```

首次使用 `nohup` 后台启动时，进程立即退出且 `launcher.log` 为空。随后以前台最小诊断确认脚本、MuJoCo/CUDA 环境和参数解析均可正常进入训练；该失败更可能来自当前执行器对普通后台子进程的清理，而非实验脚本错误。因此正式运行改用 `setsid -f bash -c ...` 脱离方式启动。

启动核查显示主控进程和三条 TD-MPC2 子训练均已存活，`cycle_001` 已创建，三种子训练日志已落盘，GPU 1、2、3 已出现训练占用。第一周期使用空规则库，故 `rule_prior_enabled=false`；后续周期将在规则库产生 `v8_replay_safe_candidate` 后才启用显式规则 prior。

`cycle001` 已完成训练和总结链条。三种子各导出 `10` 条 replay 轨迹，轨迹筛选阶段选中 `17 / 30` 条用于归因；SD explainer 使用 `8483` 条 transitions，其中训练集 `7485` 条、验证集 `998` 条。v4 子状态分支产生 `2` 条候选，其中最终共享库保留 `1` 条 `cycle_candidate`：

```text
rotation_risk|omega_neg|vx_mid|support_none|rear->prefer:waist_pos
```

该规则因 `support = 2` 且 `mean_advantage = 0.0255`，未达到在线阈值，因此没有转为 `v8_replay_safe_candidate`。v8 transition 分支在 `30` 条 traces 中未发现成功片段，transition-gap fragments、transition action records、SD counterfactual candidates 和 replay-safe rules 均为 `0`。`cycle001` 最终 `online_enabled=false`，`cycle002` 已从 `cycle001` checkpoint 接续启动；虽然训练命令传入了共享规则库，但在线 prior 只允许 `v8_replay_safe_candidate`，因此当前尚无有效规则干预。

`cycle002` 已完成训练和总结链条，`cycle003` 已接续启动。SD explainer 从 `cycle001` checkpoint 接续初始化，使用 `23453` 条 transitions，其中训练集 `20958` 条、验证集 `2495` 条。v4 子状态分支产生 `6` 条候选，但闭环共享库仍只保留上一周期的候选规则：

```text
rotation_risk|omega_neg|vx_mid|support_none|rear->prefer:waist_pos
```

该规则累计 `support = 5`、累计 `mean_advantage = 0.0231`，仍低于在线阈值，最终状态保持为 `cycle_candidate`，`online_enabled=false`。

v8 transition 分支相比 `cycle001` 开始出现非零信号：在 `60` 条 traces 中发现 `2` 个 success fragments 和 `2` 个 transition-gap fragments，涉及：

```text
transition|transition_gap|omega_mid|vx_mid|support_none|falling|support_risk
transition|transition_gap|omega_neg|vx_mid|support_none|falling|support_risk
```

但 action evidence 仅有 `5` 条 records、`4` 个 groups，观测候选为 `0`；SD counterfactual records 为 `3`，候选为 `0`，因此未形成 v8 transition rules 或 replay-safe rules。该结果说明 `phase_any` 修改后 transition 分支已经能在在线早期捕捉到少量片段，但截至 `10k` 仍不足以形成可上线规则。

`cycle003` 已完成训练和总结链条，`cycle004` 已接续启动。三种子均正常保存 `final.full.pt`。SD explainer 从 `cycle002` checkpoint 接续初始化，使用 `44411` 条 transitions，其中训练集 `39920` 条、验证集 `4491` 条。v4 子状态分支产生 `12` 条候选，闭环共享库保留 `3` 条 `cycle_candidate`：

```text
rotation_risk|omega_neg|vx_mid|support_none|rear->prefer:waist_pos
unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:hip_neg
rotation_risk|omega_neg|vx_low|support_none|rear->prefer:hip_neg
```

三条规则均未达到 `v8_replay_safe_candidate` 状态，`cycle003` 最终仍为 `online_enabled=false`。其中第一条累计 `support = 5`、累计 `mean_advantage = 0.0231`；第二条 `support = 2`、`mean_advantage = 0.0623`；第三条 `support = 5`、`mean_advantage = 0.0223`。当前主要受 support 和 online advantage 阈值限制。

v8 transition 分支继续增强但仍未上线：在 `90` 条 traces 中发现 `5` 个 success fragments 和 `5` 个 transition-gap fragments；transition action evidence 为 `13` 条 records、`6` 个 groups；SD counterfactual records 为 `8`，候选仍为 `0`。因此 `v8_transition_rules`、validation 和 replay-safe export 均为空。该阶段的积极信号是 success fragment 数量随训练进展增加；限制因素仍是跨种子、跨规则键的证据聚合不足，尚不能支持在线动作 prior。

### v8 transition 闭环重跑的中止与阶段性判断

`hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2` 原计划运行 `100k`，但在用户确认后中止继续推进。由于中止执行时主控已经越过 `50k` 节点并推进至 `cycle016` 训练阶段，完整可复查节点实际包括 `25k`、`50k` 和 `75k`。`cycle016` 仅保留部分训练日志，不作为正式评估节点。

正式 `10ep` eval 结果如下：

```text
25k:
  seed1 = 0.3403018239652738
  seed2 = 0.003030696837231517
  seed3 = 0.0
  mean  = 0.1144441736008351

50k:
  seed1 = 0.8833430181723088
  seed2 = 0.031516402238048614
  seed3 = 0.24276511929929256
  mean  = 0.3858748465698833

75k:
  seed1 = 10.606520211859607
  seed2 = 1.895721105489065
  seed3 = 0.97941173017025
  mean  = 4.493884349172974
```

截至 `cycle015`，闭环规则库包含 `28` 条规则，其中 `5` 条为 `v8_replay_safe_candidate`、`3` 条为 `cycle_supported`、`17` 条为 `cycle_candidate`、`3` 条为 `cycle_weakened`。所有规则来源仍为 `v4_substate`。线上规则为：

```text
unsupported_high_torque|omega_neg|vx_low|support_none|rear->prefer:hip_neg
unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:waist_pos
rotation_risk|omega_neg|vx_low|support_none|rear->prefer:hip_pos
rotation_risk|omega_neg|vx_low|support_none|rear->prefer:waist_neg
unsupported_high_torque|omega_neg|vx_mid|support_none|rear->prefer:knee_pos
```

v8 transition 分支在 `cycle015` 已能从 `450` 条 replay trace 中识别 `55` 个 transition-gap 片段，但 `v8_transition_rule_validation` 和 `v8_transition_replay_safe_rules` 仍均为 `0` 条规则。这说明当前瓶颈不是训练预算本身，而是 transition 片段尚未被可靠转化为可上线规则。

因此，本轮结果不继续解释为“再跑到 `100k` 可能自然显著改善”。更稳妥的阶段性结论是：工程闭环和规则 prior 加载均已验证，恢复类规则可滚动上线并影响训练过程，但规则覆盖仍主要局限于 `rear` 恢复附近，尚未形成面向姿态维持和阶段转接的有效规则族。后续应优先修订 transition 规则提取与验证机制，再启动下一轮正式在线测试。

### v8 transition 规则键聚合修订

对 `cycle015` 产物进行分层审查后，确认 v8 transition 链条的主要截断点是规则键过细。原实现使用：

```text
role | quality_bin | omega_bin | vx_bin | support_bin | refined_phase | risk_bin
```

作为 transition 规则键。`cycle015` 中虽然已有 `133` 条 transition action record 和 `78` 条 SD 反事实 record，但被拆分为 `37` 个精细 rule-key group，导致单个动作族通常达不到 `min_support = 5`。去除 `refined_phase` 的精细拆分后，相同数据可压缩为 `17` 个 transition group，其中多个 group 已跨 `2-3` 个 seed，且动作族证据达到候选阈值。

已修订代码：

```text
experiments/sd_retrospective_rule_induction/mine_hopper_v8_transition_actions.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_transition_replay_safety.py
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
```

修订后，v8 transition 规则默认使用显式 wildcard phase token：

```text
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|support_risk
```

该键仍保留 `omega`、`vx`、`support` 和 `risk`，但不再把 `falling`、`near_good_air`、`near_good_contact` 等阶段别名拆成不同规则。`audit_hopper_v8_transition_replay_safety.py` 同步支持 `phase_any` replay 匹配，`online_rule_prior_v6.py` 同步支持 `phase_any` 在线状态匹配。

在 `cycle015` 真实产物上进行离线 smoke，结果如下：

```text
transition action records: 133
rule-key groups: 17
observational candidates: 2
SD counterfactual records: 78
SD counterfactual candidates: 4
merged transition rules: 6
heldout_supported rules: 3
replay_safety_supported rules: 1
exported v8 replay-safe transition rules: 1
```

导出的 replay-safe transition 规则为：

```text
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|support_risk->soft_prefer:waist_pos
```

该规则在 replay safety 中命中 `3603` 步，其中 good hits 为 `13`，false intervention rate 为 `0.0769`，低于当前 `0.10` 上限。该 smoke 说明，上一轮 v8 transition 规则为 `0` 并非完全缺少信号，而是规则键粒度过细造成证据稀释。下一轮在线测试应使用该聚合键版本重新进入从头训练。

### 正式 100k 空规则库闭环实验完成

正式从头 `100k` Hopper 闭环实验已完成，初始规则库为空，三种子共享规则，SD 回归、失败归因和规则更新每 `5k` 执行一次，checkpoint 与标准 `10ep` eval 每 `25k` 执行一次。最终实验目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_empty_seed123_20260508
```

最终 `100k / 10ep` 评估结果为：

```text
seed1 = 0.839243
seed2 = 12.245380
seed3 = 0.015687
mean  = 4.366770
population_std = 5.581155
sample_std     = 6.835490
```

官方 TD-MPC2 Hopper `100k` 参考均值为 `13.63`，来源为：

```text
docs/neurips/paper/Attachments/data/hopper/mechanism_oriented_200k/hopper_public_baseline_comparison.csv
```

因此，当前规则闭环版本在正式 `100k` 口径下未超过官方 TD-MPC2 基线。该结果不解释为方法论失效，而解释为当前规则机制尚未形成足够稳定、关键和高覆盖的行为约束。工程管线已完整跑通，包括训练、SD 接续拟合、归因、规则生成、共享规则库更新、在线 prior 加载和标准评估。

最终 `cycle020` 规则库包含 `21` 条规则，其中：

```text
v8_replay_safe_candidate = 4
cycle_candidate = 10
cycle_supported = 2
cycle_weakened = 5
```

上线规则为：

```text
rotation_risk|omega_pos|vx_low|support_some|prone->prefer:hip_pos
rotation_risk|omega_neg|vx_low|support_none|rear->prefer:hip_pos
rotation_risk|omega_neg|vx_low|support_none|rear->prefer:knee_pos
unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:waist_pos
```

### 最终轨迹与规则命中审查

新增只读审查脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v8formal_100k_final.py
```

该脚本读取最终官方 eval 的逐步轨迹和 `cycle020` 规则库，统计姿态片段、上线规则命中、动作方向对齐和相反方向动作。输出报告为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_empty_seed123_final_audit_20260508.json
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_empty_seed123_final_audit_20260508.md
```

审查结果显示，最终 `30` 条 eval episode 中，较高回报 episode 全部来自 seed2。整体轨迹的 `structural_success_frac = 0.0007`，`dense_pose_high_frac = 0.0005`，`mean_continue_score = 0.1805`，说明确实出现了少量相对较好的姿态片段，但远未形成稳定站立或持续前进。seed2 的高分片段多表现为短窗口内 `torso_z` 接近 `0.75-0.80`、`vx` 接近或超过 `0.8`，但绝对 pitch 和角速度仍偏大，因此更接近“短暂可奖励运动片段”，而不是稳定成功姿态。

上线规则在最终 eval 中确实发生命中并影响动作方向统计：

```text
rotation_risk|omega_pos|vx_low|support_some|prone->prefer:hip_pos
  hits = 1, aligned = 1, opposed = 0

rotation_risk|omega_neg|vx_low|support_none|rear->prefer:hip_pos
  hits = 676, aligned = 580, opposed = 16

rotation_risk|omega_neg|vx_low|support_none|rear->prefer:knee_pos
  hits = 676, aligned = 563, opposed = 16

unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:waist_pos
  hits = 686, aligned = 386, opposed = 203
```

该结果表明，规则 prior 不是未生效；更合理的解释是，当前上线规则主要覆盖 `rear/prone` 恢复中的局部动作排序，且并未覆盖形成稳定 Hopper 步态所需的关键转接条件。尤其是 seed2 的高分 episode 中，命中规则步数占比通常只有 `0.012-0.080`，说明规则对行为有局部影响，但难以主导整条轨迹的阶段转移。

### v8 transition 规则闭环接入与新一轮 100k 重跑

对修订后的脚本进行分析后，认为仅放宽 `max_cycle_candidates`、`rolling_weaken_max_good_share` 和 `max_online_rules_per_cycle` 有助于增加规则吞吐量，但不足以解决上一轮正式实验暴露出的主问题：规则主要覆盖 `rear/prone` 局部恢复，缺少高分片段之前和之后的关键转接规则。因此，在重跑前进一步将 v8 transition 规则族接入每周期闭环。

代码修订包括：

```text
experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

`closed_loop_rule_update.py` 现在允许 `v8_transition_replay_safety` 来源的 `avoid_action_family` 和 `soft_prefer_action_family` 规则进入闭环合并，并在排序时兼容 `heldout_strength`、`replay_hits` 和 `replay_good_hit_rate`。`run_hopper_v7_closed_loop.py` 的 `summarize_cycle` 增加 v8 transition 分支：在原有失败恢复规则更新后，继续挖掘成功片段、提取转接动作证据、运行 SD 反事实、归纳 transition 规则、执行 held-out validation、执行 replay safety 审查、导出 `v8_replay_safe_candidate`，再合并到本周期共享规则库。

在旧正式实验 `cycle020` 的真实轨迹上执行离线 smoke，结果为：

```text
success fragments = 109 / 600 traces
transition gaps = 108
counterfactual candidates = 4
heldout_supported = 1
replay_safety_supported = 1
exported v8 safe transition rules = 1
```

产生的 replay-safe transition 规则为：

```text
transition|transition_gap|omega_pos|vx_high|support_none|near_good_contact|support_risk->soft_prefer:hip_pos
```

该规则命中近成功接触但支撑风险仍高的转接状态，属于上一版 `rear/prone` 恢复规则没有覆盖的状态区域。因此，修订后的脚本具有进入新一轮正式测试的价值。

已启动新的从头 `100k` 三种子空库实验：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2
exp_prefix = tdmpc2_v8formal_100k_v8trans_empty_seed123_20260508_r2
seeds = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = empty
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = v8_replay_safe_candidate
enable_v8_transition_rules = true
```

首次沙箱内启动失败，原因为 `torch.cuda.is_available()` 为 false；随后以非沙箱权限重启。`cycle001` 已完成训练、replay 导出、SD 拟合、失败恢复规则更新和 v8 transition 分支。该周期只发现 `2` 个成功片段，未产生 transition 上线规则，符合早期训练阶段预期。`cycle002` 已接续启动。

### 每周期直接上线数量放宽

在观察层放宽的基础上，进一步将每周期直接上线规则数量上限参数化。此前 `closed_loop_rule_update.py` 固定执行 `at_most_one_rule_per_cycle`，即候选中即使有多条规则已满足在线状态，也只合并排序最高的一条。现改为：

```text
policy = bounded_online_rules_per_cycle
max_online_rules_per_cycle = 2
```

该参数已接入 `run_hopper_v7_closed_loop.py` 和 `run_hopper_v7_multiseed_shared_rules.py`。后续变体默认每周期最多直接合并 `2` 条满足 `required_status` 的规则；不满足 `v8_replay_safe_candidate` 等在线状态门槛的候选仍不会直接上线。该修改不影响当前正在运行的正式 `100k` 主实验。

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

## 2026-05-13

### 阶段效果规则的触发后状态审计

对 `toe_recovery_low_support->stage_effect:recover_full_foot_support` 进行了规则开启后的初步因果烟测。第一次评估使用 `MUJOCO_GL=egl`，因本机无法初始化 headless EGL 而未能创建 `hopper-hop` 环境；随后改用操作指南规定的 `MUJOCO_GL=glfw`。第二次评估暴露出阶段效果规则在普通 TD-MPC2 checkpoint 上调用 SD 专属 `_state_reward_features` 的接口问题，因此将 `online_rule_prior_v6.py` 修改为：仅当 checkpoint 内部模型提供结构特征接口时使用其短视界推演；否则回退到本地 `DMControlHopperSDModel` 结构模型对候选动作序列进行阶段目标评分。

修复后启动 `seed2` 和 `seed3` 的规则开启评估。由于阶段评分对每个 MPPI 候选序列额外执行短视界结构推演，完整 `10ep` 评估耗时显著增加；本次先中止完整评估，并对已经落盘的 `seed2` 前 `8` 条轨迹进行触发后 `5` 步审计。该结果只能作为因果烟测，不作为正式性能评估。

审计对象：

```text
rule_library = results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_recovery_stage_effect.json
trace_dir = logs/hopper-hop/2/tdmpc2_stage_effect_rule_eval_h5_seed2_20260513_glfw_r2/official_eval/step_0000000/episode_csv
audit = results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/stage_rule_on_h5_seed2_partial_audit.json
```

阶段规则审计结果：

```text
state_hits = 1057
stage_effective = 264
stage_harmful = 687
neutral = 106
direction_consistent_rate = 0.2498
direction_harmful_rate = 0.6500
```

直接按触发时刻与 `t+5` 状态差分统计，平均变化为：

```text
delta_full_foot_support = 0.0000
delta_support_contact = -0.7460
delta_torso_z_quality = -0.0721
delta_rotation_penalty = +0.1313
delta_body_collision = +0.0636
delta_stage_score = -0.2265
stage_score_positive_rate = 0.1178
```

因此，当前实现下不能认为该阶段效果规则已经使触发后的真实状态稳定朝“恢复全足支撑”方向发展。更合理的解释是：规则能够被加载并参与候选排序，但其当前阶段目标、权重、触发窗口或与其他规则的叠加关系尚不足以把实际闭环行为推向预期阶段结果。该发现支持后续将“规则是否触发”和“触发后是否完成阶段目标”作为两个独立验证项。

### toe-only 弹射陷阱规则修订与 eval-only 烟测

基于轨迹统计确认，`touch_toe >= 0.8`、`touch_heel < 0.2`、`support_contact >= 0.8` 且即时奖励较高的脚尖弹射片段具有明显短期收益，但在 `10-20` 步内几乎必然发生支撑丢失、下落或碰撞。旧 `seed2` 最终评估中，该类事件 `989` 次，`10` 步内 bad rate 为 `99.4%`，`20` 步内 fall/collision rate 为 `99.0%`；旧 `seed3` 最终评估中，该类事件 `888` 次，`10` 步内 bad rate 为 `99.8%`，`20` 步内 fall/collision rate 为 `98.0%`。该结果支持将其视为“短期奖励陷阱”，而不是可靠步态。

据此将阶段规则从 `toe_recovery_low_support->stage_effect:recover_full_foot_support` 修订为：

```text
toe_only_launch_trap->stage_effect:stabilize_toe_only_launch
```

新谓词只命中脚尖强支撑、heel 缺失、前向速度较高、非直立准备的弹射窗口；阶段目标改为优先保持 `support_contact`、降低 `rotation_penalty` 和 `body_collision`、提高 `spin_control`、`pitch_control` 与 `torso_z_quality`，并将 `touch_heel` 与 `full_foot_support` 作为弱正项。修订后的规则库为：

```text
results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_launch_stabilize_stage_effect.json
```

在 `seed2` 200k checkpoint 上执行 `3ep` eval-only 烟测，结果为：

```text
normal rule:
  episode_reward_mean = 238.39
  toe_launch_trap events = 322
  bad5 = 95.3%
  fall10 = 86.6%
  fall20 = 100.0%

strong rule:
  episode_reward_mean = 235.13
  toe_launch_trap events = 327
  bad5 = 96.3%
  fall10 = 84.4%
  fall20 = 100.0%
```

强介入版本使用 `confidence=0.90`、`stage_prior_weight=0.16`、`stage_max_bonus=0.25`，评估命令同时设置 `rule_prior_max_bonus=0.25`。结果表明，仅在最终 checkpoint 的 planner 排序阶段加入单条阶段规则，尚不足以从已经形成的脚尖弹射吸引子中恢复出稳定全足或可控支撑步态。该规则更适合作为后续训练期塑形信号或更高层候选序列筛选规则，而不是单独承担最终 eval-time 行为修正。

### toe-only 阶段规则的 100k 至 200k 接续训练启动

为避免将 `200k` 后验规则泄漏回 `100k` 起点，重新以 `100k cycle020` confidence 规则库为基底生成初始规则库：

```text
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_launch_stabilize_stage_effect.json
```

该规则库包含 `98` 条规则，其中 `48` 条为 `confidence_active`；新增阶段规则为：

```text
toe_only_launch_trap->stage_effect:stabilize_toe_only_launch
confidence = 0.75
stage_prior_weight = 0.08
stage_max_bonus = 0.12
status = confidence_active
```

已启动三种子 `100k -> 200k` 接续实验：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toelaunch_stage_seed123_20260513_r1
exp_prefix = tdmpc2_v8formal_200k_toelaunch_stage_seed123_20260513_r1
start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_allowed_statuses = confidence_active
rule_required_status = v8_replay_safe_candidate
```

三种子均从 `tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt` 恢复。启动日志确认 `cycle021` 目标步数为 `105020`，恢复点为 `step=100020`，并安装 `48` 条 active 规则。训练 CSV 中已出现 `rule_prior_hit=1` 的记录，说明规则 prior 已实际进入在线规划排序。

## 2026-05-13

### toe-only 触发谓词偏移的纠正

对 `hopper_v8formal_200k_toelaunch_stage_seed123_20260513_r1` 的早期周期进行审查后发现，新增阶段规则 `toe_only_launch_trap->stage_effect:stabilize_toe_only_launch` 在 `cycle021`、`cycle022`、`cycle023` 的 rule-effect audit 中均为 `state_hits=0`。该现象并非脚尖恢复相关状态消失，而是规则触发键从原先大量命中的 `toe_recovery_low_support` 改为更窄的 `toe_only_launch_trap` 后，额外条件排除了全部训练回放状态。

使用 `cycle023` 的真实 replay trace 直接统计，旧谓词 `toe_recovery_low_support` 仍命中 `56949` 帧，而新谓词 `toe_only_launch_trap` 命中 `0` 帧。分解检查显示，主要问题是新谓词包含 `torso_z >= 0.55`，但当前 replay 导出中的 `rootz/qpos_rootz` 均低于该阈值；同时 `vx >= 0.80` 也进一步收窄了触发范围。因此，原 `toelaunch_stage` 实验不能作为“阶段效果规则是否有效”的干净检验。

已停止该无效口径实验的主控进程和残留后处理进程。随后将 [build_hopper_toe_recovery_stage_rule.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py) 修订为生成：

```text
toe_recovery_low_support->stage_effect:stabilize_toe_only_launch
```

该规则沿用原有 `toe_recovery_low_support` 触发谓词，只将阶段目标改为稳定脚尖低支撑弹射窗口，优先保持可控支撑、降低旋转和碰撞风险，并将 `touch_heel/full_foot_support` 作为弱正项。生成时会移除旧的 `toe_recovery_low_support->soft_prefer:waist_pos`、`toe_recovery_low_support->stage_effect:recover_full_foot_support` 以及错误的 `toe_only_launch_trap->stage_effect:stabilize_toe_only_launch`，避免同一诊断问题被多条规则叠加解释。

新 100k 起点规则库为：

```text
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json
```

该规则库包含 `98` 条规则，其中新增阶段规则为 `confidence_active`，`confidence=0.75`，`stage_prior_weight=0.08`，`stage_max_bonus=0.12`。使用上一轮已落盘的 `cycle023` replay trace 做 rule-effect audit smoke，结果为：

```text
state_hits = 56949
stage_effective = 35405
stage_harmful = 20603
direction_consistent_rate = 0.6217
direction_harmful_rate = 0.3618
```

该 smoke 仅用于确认触发口径已经恢复，不用于声称规则收益。随后已启动新的三种子 `100k -> 200k` 接续实验：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1
exp_prefix = tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1
initial_rule_library = results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json
start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
```

启动日志确认 `cycle021` 的 seed1、seed2、seed3 训练进程均存在，并从 `tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt` 恢复。

## 2026-05-14

### 脚尖恢复阶段规则的结构阶段拆分

使用 `hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1` 的 `cycle033` 训练回放，对 `toe_recovery_low_support->stage_effect:stabilize_toe_only_launch` 进行结构态后果分析。分析口径不以即时回报为主判据，而是检查规则触发后 `horizon=5` 内的支撑、脚跟/全足接触、旋转、跌落和高度变化。

结果显示，该规则触发状态确实对应脚尖强接触、脚跟和全足支撑缺失、非直立的低支撑窗口，但不同结构阶段的后果明显分化。`prone/rear` 等可恢复阶段的有效率较高，而 `falling_air/toe_support` 等失稳阶段的有害率较高。因此，原统一规则被判定为“诊断问题成立，但在线干预谓词过宽”。

为避免后续周期混入新旧代码口径，已停止旧实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1
```

随后修改 [online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py) 和 [audit_hopper_v8_widefunnel_rule_effects.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py)，新增两个与在线和审计一致的触发键：

```text
toe_recovery_recoverable_low_support
toe_recovery_unstable_low_support
```

其中 `toe_recovery_recoverable_low_support` 要求满足原 `toe_recovery_low_support`，且 `grounded_prone` 或 `grounded_rear` 成立；`toe_recovery_unstable_low_support` 覆盖剩余低支撑脚尖恢复窗口，主要对应 `falling_air/toe_support` 等更不稳定状态。

[build_hopper_toe_recovery_stage_rule.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py) 现在生成拆分规则库：

```text
results/sd_retrospective_rule_induction/rules/hopper_165k_cycle033_confidence_plus_toe_recovery_split_stage_effect.json
```

该规则库移除旧的统一脚尖恢复阶段规则，新增：

```text
toe_recovery_recoverable_low_support->stage_effect:stabilize_recoverable_toe_support
status = confidence_active

toe_recovery_unstable_low_support->stage_effect:observe_unstable_toe_support
status = confidence_observing
```

在 `cycle033` 同一批三种子训练轨迹上重跑审计，结果为：

```text
recoverable:
  state_hits = 41809
  stage_effective = 27899
  stage_harmful = 13067
  direction_consistent_rate = 0.6673
  direction_harmful_rate = 0.3125

unstable:
  state_hits = 36650
  stage_effective = 12897
  stage_harmful = 23348
  direction_consistent_rate = 0.3519
  direction_harmful_rate = 0.6371
```

该结果支持拆分设计：可恢复子阶段适合继续作为在线轻量 prior，失稳子阶段暂不直接干预，只进入观察状态，由后续 confidence 机制决定是否增强、削弱或退役。

已基于 `cycle033 / 165k` checkpoint 启动拆分阶段规则接续实验：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1
exp_prefix = tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1
start_cycle = 34
cycles = 40
cycle_steps = 5000
initial_rule_library = results/sd_retrospective_rule_induction/rules/hopper_165k_cycle033_confidence_plus_toe_recovery_split_stage_effect.json
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1/cycle_033/sd_explainers/sd_explainer_cycle_033.pt
```

三种子均从旧统一规则实验 `cycle033_target165032/models/final.full.pt` 恢复。启动日志显示 `cycle034` 目标步数为 `170033`，三个训练进程均已启动，并安装 `56` 条 `confidence_active` 规则。`toe_recovery_unstable_low_support` 初始为 `confidence_observing`，因此不会进入本周期在线 prior，只参与后续审计和置信度更新。

### 阶段成果规则未形成预期引导的原因审查

对 `hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1` 的 `cycle034` 至 `cycle037` 训练回放进行触发后结构态审查。分析仍以规则触发后的状态演化为主，不以即时回报作为主要判据。

结果显示，`toe_recovery_recoverable_low_support` 的触发位置基本正确，但触发后的真实短视界结构没有明显改善。以 `cycle037` 为例，触发状态通常为 `touch_toe≈0.98`、`touch_heel≈0`、`full_foot_support≈0`、`foot_grounded≈0.98`。在触发后 `horizon=5` 内，平均 `foot_grounded` 下降约 `0.421`，严格满足“脚跟和全足支撑有进展、支撑不丢失、跌落和旋转不增加”的比例仅约 `2.2%`。该现象与旧统一规则线 `cycle033` 相比没有实质改善，说明拆分触发谓词虽有必要，但不足以让规则形成预期控制后果。

机制审查得到三个主要原因。第一，阶段目标仍偏软，候选序列可以在保持脚尖接触或改善部分姿态指标时获得分数，而不必真正恢复脚跟或全足支撑。第二，规则库中写入的 `horizon=5` 与当前 TD-MPC2 在线规划 `horizon=3` 不一致，在线执行时 `_stage_effect_score()` 会把规则视界截断为 `actions.shape[0]`，实际只看 3 步；这会削弱“先稳定、再恢复支撑”的阶段规则。第三，SD 对该规则依赖的关键结构量仍有较大误差，近期 `support_contact_mae≈0.43-0.45`、`full_foot_support_mae≈0.23-0.28`、`pitch_rate_mae≈1.9-2.7`，因此基于短视界预测的候选排序可能无法稳定对应真实后果。

同时发现审计口径存在一处会高估阶段规则有效性的实现问题：`stage_metric()` 对 `support_contact` 未像状态谓词那样回退读取 `foot_grounded`，导致部分 replay CSV 中的支撑语义在阶段分数中被缺省为 `0`。该问题不影响在线 prior 的触发，但会影响 confidence 更新所依据的 stage-effect audit。

据此已停止旧拆分阶段实验的主控进程与 `cycle038` 三个训练进程，停止点为 `cycle037 / 185k` 已完成、`cycle038 / 190k` 刚启动后。旧实验目录保留为诊断产物：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1
```

随后完成三项修订。第一，[audit_hopper_v8_widefunnel_rule_effects.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py) 的 `stage_metric("support_contact")` 现在按 `support_contact -> foot_grounded` 的顺序读取字段，使阶段审计与状态谓词使用一致语义。第二，[online_rule_prior_v6.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py) 的 `stage_effect_prior` 增加 `stage_constraints`，可对短视界内支撑丢失、跌落、旋转风险和缺少脚跟/全足恢复进展施加未归一化约束惩罚。第三，[build_hopper_toe_recovery_stage_rule.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py) 将可恢复脚尖规则改为：

```text
toe_recovery_recoverable_low_support -> stage_effect_prior: constrain_recoverable_toe_support
horizon = 3
```

其阶段目标从软性 `stabilize_recoverable_toe_support` 改为约束式目标：保留短视界最小支撑和终端支撑，要求 `touch_heel` 与 `full_foot_support` 至少有小幅进展，并惩罚进入 `falling_air` 或高 `rotation_penalty` 的候选序列。新规则库已生成：

```text
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json
```

使用旧 `cycle037` 轨迹对新规则库做只读审计，结果显示可恢复规则不再被乐观地解释为稳定有效：

```text
toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support
state_hits = 45039
stage_effective = 22366
stage_harmful = 22203
direction_consistent_rate = 0.4966
direction_harmful_rate = 0.4930
```

该 smoke 的含义是：新审计口径已能把旧行为中“触发后没有恢复支撑、甚至丢失支撑”的部分识别为有害，而不是继续将其解释为有效。后续若使用该规则重训，应先观察它是否在在线排序中显著降低这类有害后果；若第一周期后仍保持约 `0.49` 的 harmful rate，则 confidence 机制应自然削弱或退出该规则。

### 阶段约束规则 eval-only 结构烟测

在不重训 checkpoint 的条件下，使用 `cycle037 / 185k` 三个 TD-MPC2 checkpoint 对阶段约束规则进行了 eval-only 验证。该验证分为两步：第一，使用完整 `confidence_active` 规则库开启 rule prior，检查当前规则系统在最终动作排序中的整体效果；第二，构造只包含 `toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support` 的最小规则库，隔离检验新增阶段约束规则本身的作用。

候选排序离线审计显示，新规则在 SD 短视界候选中确实会把结构上更好的候选排到更前。600 个触发状态中，约束规则 top candidate 的约束满足率为 `0.7017`，高于实际动作的 `0.5633`，也高于旧软阶段规则的 `0.6350`。因此，规则的离线排序方向并非完全无信号。

但 eval-only 真实轨迹烟测没有显示触发后结构改善。完整规则库 rule-on 与 rule-off 的 30 episode 对比结果为：

```text
rule-off:
  reward_mean = 122.2854
  triggers = 5841
  expected_good_rate = 0.0021
  rule_constraint_satisfied_rate = 0.0021
  mean_min_support = 0.5867
  mean_max_rotation_penalty = 0.2105

rule-on:
  reward_mean = 120.6674
  triggers = 5323
  expected_good_rate = 0.0017
  rule_constraint_satisfied_rate = 0.0019
  mean_min_support = 0.4737
  mean_max_rotation_penalty = 0.2785
```

只启用阶段约束规则的 stage-only 对比结果同样不支持继续训练：

```text
rule-off:
  reward_mean = 122.2854
  triggers = 5841
  expected_good_rate = 0.0021
  rule_constraint_satisfied_rate = 0.0021
  mean_min_support = 0.5867
  mean_full_foot_progress = 0.0038

stage-only:
  reward_mean = 116.2903
  triggers = 5305
  expected_good_rate = 0.0008
  rule_constraint_satisfied_rate = 0.0008
  mean_min_support = 0.4846
  mean_full_foot_progress = 0.0015
```

分项约束进一步定位了失败原因。stage-only 后 `falling_air` 约束基本保持正常，`falling_ok_rate` 为 `0.9448`；但 `min_support_ok_rate` 从 `0.6025` 降至 `0.5020`，`terminal_support_ok_rate` 从 `0.7367` 降至 `0.6746`，`rotation_ok_rate` 从 `0.7764` 降至 `0.7176`，`full_foot_progress_ok_rate` 从 `0.0039` 降至 `0.0015`。这说明当前规则并没有把触发后的行为推向“保持支撑并恢复全足”的目标，反而更常伴随支撑丢失和旋转风险上升。

该结果被解释为：当前阶段约束规则具有可运行性和局部候选排序信号，但尚不具备可控的在线结构引导能力。其主要缺口不在于是否能命中状态，而在于短视界 SD 排序与真实环境后果之间仍存在转化失败；同时，当前约束权重和评分流程仍可能让候选通过部分相对指标获得正分，而未真正满足脚跟、全足和支撑保持目标。因此，本轮不进入 185k 至 190k 的接续训练，应先修正规则评分与审计一致性。

新增报告与脚本：

```text
experiments/sd_retrospective_rule_induction/compare_hopper_stage_constraint_evalonly.py
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_toe_recovery_stage_constraint_only.json
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_evalonly_20260514/post_trigger_structure_compare.json
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_stageonly_evalonly_20260514/post_trigger_structure_compare_stage_only_vs_off.json
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_stageonly_evalonly_20260514/rule_effect_audit_stage_only.json
```

### 约束优先评分与 SD checkpoint 加载复测

针对上一轮结果，完成两项进一步修订。第一，`stage_effect_prior` 新增 `constraint_priority_weight`、`constraint_satisfied_bonus` 和 `constraint_absolute_weight` 字段。若 `constraint_priority_weight > 0`，在线 prior 会先按候选序列的约束违背量形成相对惩罚，再叠加原有软阶段目标；软目标只在约束水平相近的候选之间发挥细分作用。第二，`online_rule_prior_v6.py` 新增 `rule_prior_sd_checkpoint` 参数，使普通 TD-MPC2 checkpoint 的 eval-only 测试可以显式加载同周期拟合的 SD explainer，而不是回退到未接续训练的本地结构模型。`eval_hopper_v7_rule_prior.py` 同步记录 `rule_prior_sd_checkpoint_loaded`，便于审计确认。

基于 `cycle037 / 185k` 规则库重新生成：

```text
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint_priority.json
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_toe_recovery_stage_constraint_priority_only.json
```

离线候选排序审计显示，约束优先评分确实改善了 SD 内部候选排序：

```text
old stage_constraint_rule:
  top_constraint_satisfied_rate = 0.7017
  mean_top_min_support = 0.6697
  mean_top_touch_heel_progress = 0.5822
  mean_top_full_foot_progress = 0.5741

priority stage_constraint_rule:
  top_constraint_satisfied_rate = 0.8367
  mean_top_min_support = 0.6868
  mean_top_touch_heel_progress = 0.6010
  mean_top_full_foot_progress = 0.5930
```

随后进行了两组 stage-only eval-only 烟测。第一组使用 priority 规则但不显式加载 SD checkpoint，结果为：

```text
stage_priority_only:
  reward_mean = 121.1224
  expected_good_rate = 0.0010
  rule_constraint_satisfied_rate = 0.0008
  mean_min_support = 0.4337
  mean_max_rotation_penalty = 0.2906
```

第二组显式加载 `cycle037` SD explainer：

```text
rule_prior_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/cycle_037/sd_explainers/sd_explainer_cycle_037.pt

stage_priority_sdckpt:
  reward_mean = 117.5056
  expected_good_rate = 0.0013
  rule_constraint_satisfied_rate = 0.0011
  mean_min_support = 0.4933
  mean_max_rotation_penalty = 0.2180
```

三种子报告均显示 `rule_prior_sd_checkpoint_loaded = true`，说明复测确实使用了训练过的 SD explainer。与 rule-off 相比，`stage_priority_sdckpt` 仍未改善触发后结构目标：

```text
rule-off:
  reward_mean = 122.2854
  expected_good_rate = 0.0021
  mean_min_support = 0.5867
  support_not_lost_rate = 0.5814

stage_priority_sdckpt:
  reward_mean = 117.5056
  expected_good_rate = 0.0013
  mean_min_support = 0.4933
  support_not_lost_rate = 0.4881
```

该结果说明，先前确实存在“在线 prior 未显式加载接续 SD explainer”的工程缺口，但修补后仍不足以让阶段约束规则形成真实结构改善。当前主要问题转为：离线人工候选集合上的 SD 排序改善，不能可靠转化为 TD-MPC2 实际 MPPI 候选分布中的最终动作变化。下一步不应进入接续训练，而应审计真实 planner 候选集合：记录触发状态下 raw value、rule bonus、elite 选择和最终动作，判断规则是否在真实候选池中选到了 SD 认为满足约束、且环境后果也更好的序列。

新增报告：

```text
results/sd_retrospective_rule_induction/reports/hopper_185k_cycle037_stage_constraint_priority_candidate_ranking.json
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_priority_stageonly_evalonly_20260514/post_trigger_structure_compare_stage_priority_only_vs_off.json
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_priority_sdckpt_evalonly_20260514/post_trigger_structure_compare_stage_priority_sdckpt_vs_off.json
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_priority_sdckpt_evalonly_20260514/rule_effect_audit_stage_priority_sdckpt.json
```

### 回到单动作/动作族规则线并接续至 300k

基于阶段约束规则尚未通过真实 eval 结构烟测的结果，本轮将主线验证重新切回旧的单动作/动作族置信规则方案。该方案对应 `confidence_toerecovery_auditfix` 分支，即继续使用自学习的 `confidence_active` 动作规则和 SD 诊断生成的 toe-recovery 动作族规则，不使用后续阶段约束式规则作为在线 prior。

接续起点为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2
```

该起点在 `cycle040 / 200k` 的 formal eval 为：

```text
seed1 = 51.4377
seed2 = 231.8195
seed3 = 240.2131
mean  = 174.4901
```

对应官方 TD-MPC2 200k 记录为：

```text
seed1 = 138.4
seed2 = 215.8
seed3 = 145.7
mean  = 166.6333
```

因此，该旧方案在 200k 均值上略高于官方 TD-MPC2 记录，且其规则机制已完整打通，适合作为继续观察“策略网络学习 + 规则自学习机制”能否逐渐改善 Hopper 姿态的主线。

已新增固定启动脚本：

```text
scripts/launch_hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1.sh
```

本轮实验目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1
```

接续输入为：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/cycle_040/rules/library_closed_loop_cycle_040_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/cycle_040/sd_explainers/sd_explainer_cycle_040.pt

initial_resumes =
logs/hopper-hop/1/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/models/final.full.pt
logs/hopper-hop/2/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/models/final.full.pt
logs/hopper-hop/3/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/models/final.full.pt
```

运行协议保持旧线口径：`start_cycle=41`，`cycles=60`，`cycle_steps=5000`，formal eval 与 checkpoint 每 `25000` 步执行一次，formal eval 为 `10` episodes。规则 prior 使用 `confidence_active` 状态规则，`rule_prior_mode=penalty_only`，`rule_prior_weight=0.05`，`rule_prior_opposite_penalty=0.01`，`rule_prior_cap_penalty=0.01`。SD explainer 从 `cycle040` 继续拟合，每 5k 周期进行一次 SD 回归、归因和规则库更新。

启动检查结果如下：

```text
tmux session = hopper300k_oldscheme
cycle041 target_steps = 205040
seed1/seed2/seed3 resumed from step = 200040
Installed v6 rule prior: rules = 59
rule_prior_allowed_statuses = confidence_active
sd_online_supervision_enabled = false
```

该检查说明，本轮训练已经回到旧的单动作/动作族规则线，并且不是阶段式规则分支，也不是在线监督 SD 训练脚本。

## 2026-05-15

### TVA 审计型接入实现

根据规则价值评估应逐步推进、每项修订均需可验证、且未来应具备自动化和跨问题泛化能力的要求，本日新增 `TVA` 旁路审计接入。该实现不将 Hopper 特定姿态指标写入规则判断标准，而是在每个总结周期中使用 `SD` 反事实推演后的终端状态，并由 `TVA` 价值网络给出未来价值读出。

为降低重复实现风险，当前实现复用 `scripts/run_dmcontrol_hopper.py` 中已有的 `HopperValueNet`、`build_tva_targets` 和 `train_value_net`。规则归纳目录只新增两个薄封装脚本：

```text
experiments/sd_retrospective_rule_induction/train_hopper_tva_explainer.py
experiments/sd_retrospective_rule_induction/audit_hopper_tva_counterfactuals.py
```

`train_hopper_tva_explainer.py` 使用当前周期选中的轨迹、当前周期拟合完成的 `SD explainer` 和既有 TVA 目标构造函数训练价值网络。脚本支持 `--init-checkpoint`，因此可像 `SD explainer` 一样在周期之间接续训练。`audit_hopper_tva_counterfactuals.py` 则对现有 `counterfactual_actions.py` 生成的动作探测集合进行 TVA 终端价值评分，输出 `tva_counterfactual_audit.json/md`，但不更新规则库。

闭环入口 [run_hopper_v7_closed_loop.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py) 和 [run_hopper_v7_multiseed_shared_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py) 新增 `--enable-tva-audit` 及 TVA 相关参数。默认情况下该开关关闭，现有训练、`SD` 拟合、反事实归因、规则归纳和 confidence 更新流程不受影响。启用后，每周期在 `SD explainer` 拟合之后训练 TVA，并在普通反事实分析之后生成 TVA 旁路审计。为控制计算成本，`--tva-audit-max-records` 默认设为 `300`；若需全量审计，可显式设为 `0`。

验证结果如下：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/train_hopper_tva_explainer.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_tva_counterfactuals.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

语法检查通过。随后使用 `cycle060` 已有轨迹和 `SD explainer` 进行小样本 smoke：`train_hopper_tva_explainer.py` 在 `200` 条 transition、`1` 个 epoch 下成功写出 `/tmp/hopper_tva_smoke.pt` 与 `/tmp/hopper_tva_smoke_summary.json`；`audit_hopper_tva_counterfactuals.py` 在 `5` 条反事实记录上成功写出 `/tmp/hopper_tva_cf_audit_smoke.json`。另执行 `run_hopper_v7_multiseed_shared_rules.py --dry-run --enable-tva-audit`，确认 TVA 训练与审计命令已按预期插入周期流程，且默认仍不改变规则库更新路径。

本次实现中曾发现两个工程细节：第一，当前结构态维度为 `17`，而旧 `HopperValueNet` 默认参数为 `15`，因此封装脚本改为根据实际编码后的状态维度实例化价值网络；第二，完整 `cycle060` TVA 反事实审计在 CPU 上成本较高，因此加入 `--tva-audit-max-records` 预算参数，使第一阶段验证保持轻量可控。

### TVA 第一、第二层离线验证

为验证 `TVA` 是否具备进入规则价值判断的基础，本日新增 [audit_hopper_tva_value_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_value_signal.py)，用于检查 TVA 状态价值与真实 episode return、trajectory label 的关系。该脚本不影响训练与规则库，只读取 `selection`、`SD checkpoint` 和 `TVA checkpoint`，输出 episode-level 的 TVA 均值、最大值、前段均值、后段均值，以及与回报的 Pearson / Spearman 相关。

随后对旧单动作/动作族规则主线的 `cycle041 / cycle050 / cycle060` 做离线验证。输出目录为：

```text
results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515
```

验证配置如下：

```text
TVA train transitions per cycle = 5000
TVA epochs per cycle = 10
TVA checkpoint continuation = cycle041 -> cycle050 -> cycle060
value audit episodes per cycle = 300
value audit rows per episode = 20
counterfactual audit records per cycle = 300
counterfactual audit sampling = stratified by label
```

汇总结果显示，TVA 已经具备明确的 episode-level 价值信号：

| cycle | reward-TVA Pearson | reward-TVA Spearman | bad TVA mean | good TVA mean |
| --- | ---: | ---: | ---: | ---: |
| `041` | `0.860` | `0.719` | `-16.65` | `58.42` |
| `050` | `0.972` | `0.717` | `-21.56` | `66.40` |
| `060` | `0.949` | `0.654` | `-25.03` | `78.20` |

这说明，在当前数据与训练预算下，TVA 能稳定地区分高回报轨迹与低回报轨迹，可作为规则价值判断的候选信号。

第二层反事实排序审计也给出有用但尚不充分的信号。分层抽样后，good-label 记录中实际动作进入 TVA top-3 的比例高于 bad-label 记录：

| cycle | bad top-3 | good top-3 | bad TVA advantage | good TVA advantage |
| --- | ---: | ---: | ---: | ---: |
| `041` | `0.248` | `0.412` | `9.38` | `13.63` |
| `050` | `0.170` | `0.360` | `14.89` | `10.66` |
| `060` | `0.210` | `0.400` | `10.06` | `14.16` |

该结果说明 TVA 对动作排序具有一定解释力，但还不能直接替代现有 confidence 更新：即使在 good-label 记录中，TVA 仍经常认为探测动作存在较大增益，说明其短视界反事实评分仍受动作探测集合、SD rollout 误差和 TVA 局部外推影响。因此，下一步不应直接将 TVA 接入在线 prior，而应先实现 `shadow update`：在每个周期旁路生成 TVA 建议的规则增强/削弱/退休结果，并与当前 confidence 更新和后续规则效应审计进行对照。

### TVA shadow update 第一版

本日新增 [apply_tva_shadow_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_shadow_update.py)，用于将 TVA 反事实评分转化为规则级影子置信度建议。该脚本只生成独立报告，不修改源规则库，不改变在线 planner prior，也不改变训练路径。

实现上，`audit_hopper_tva_counterfactuals.py` 同步补充了 `probe_actions`、`family`、`best_action_family` 与 `actual_action_family` 字段，使每个 probe 动作可以在动作族层面被审查。`apply_tva_shadow_update.py` 读取 TVA 审计结果后，复用现有 rule-effect 审计中的 `row_parts`、`state_matches` 和 `selected_gate_applies`，在相同状态匹配与在线门控条件下检查每条规则。若规则动作族在触发状态中进入 TVA family top-k 且优于实际动作，则记为正向证据；若相反动作族进入 top-k 且优于规则动作族，则记为负向证据。脚本据此写出 `shadow_confidence`、`shadow_confidence_status`、`shadow_update_reason` 与各项 TVA 组件。

闭环入口 [run_hopper_v7_closed_loop.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py) 和 [run_hopper_v7_multiseed_shared_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py) 新增 `--enable-tva-shadow-update`。该开关默认关闭，并要求 `--enable-tva-audit` 已启用。启用后，每周期会在规则库生成之后额外写出 `reports/tva_shadow_update.json/md`，但返回给下一周期训练的规则库路径不变。

验证步骤如下：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/audit_hopper_tva_counterfactuals.py \
  experiments/sd_retrospective_rule_induction/apply_tva_shadow_update.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

语法检查通过。随后使用 `cycle060` 既有旧规则主线数据执行 shadow smoke，正式输出为 [hopper_tva_shadow_update_cycle_060.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/hopper_tva_shadow_update_cycle_060.json) 与同名 Markdown 报告。该次抽样审计共匹配 `2356` 个 state-rule 对，规则级结果为：`180` 条规则中 `16` 条被影子增强、`19` 条被影子削弱、`145` 条保持不变；由于本脚本默认不将抽样缺席解释为规则无效，缺席和低命中不会造成置信度衰减。代表性结果显示，`rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` 从 `0.000` 影子增强至 `0.065`，而 `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` 从 `0.560` 影子削弱至 `0.488`。

另执行 `run_hopper_v7_multiseed_shared_rules.py --dry-run --enable-tva-audit --enable-tva-shadow-update`，确认闭环命令序列中已在 TVA 审计之后、下一周期规则库返回之前写出 shadow update 报告。最后用 `cycle060` 的 `SD explainer` 与 TVA checkpoint 对 `1` 条记录重新运行 `audit_hopper_tva_counterfactuals.py`，确认新 schema 中 `probe_actions` 完整写出，样本记录包含 `25` 个 probe actions。

### TVA shadow update 回填验证

在第一版 shadow update 实现后，对旧单动作/动作族规则主线的 `cycle041 / cycle050 / cycle060` 执行回填验证。输出文件位于：

```text
results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/
```

新增 [summarize_tva_shadow_validation.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/summarize_tva_shadow_validation.py)，用于汇总每轮 TVA shadow 与同周期 [rule_effect_audit.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/cycle_060/reports/rule_effect_audit.json) 的关系。生成的主报告为 [tva_shadow_validation_summary.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/tva_shadow_validation_summary.md)。

回填结果如下：

| cycle | rules | matched state-rule pairs | strengthened | weakened | unchanged | mean shadow delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `041` | `143` | `2068` | `11` | `14` | `118` | `-0.0014` |
| `050` | `157` | `2285` | `10` | `15` | `132` | `-0.0016` |
| `060` | `180` | `2356` | `16` | `19` | `145` | `-0.0013` |

跨周期稳定信号主要有两类。第一类是持续削弱的 `hip_neg` 规则，例如 `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` 在 `041/050/060` 三轮均被削弱，平均影子变化为 `-0.0715`；`rotation_risk|omega_pos|vx_high|support_none|falling->prefer:hip_neg` 同样三轮持续削弱，平均影子变化为 `-0.0700`。第二类是幅度较小但持续增强的恢复动作规则，例如 `unsupported_high_torque|omega_neg|vx_low|support_none|falling->prefer:ankle_neg` 与 `rotation_risk|omega_neg|vx_low|support_none|falling->prefer:ankle_neg` 在三轮中均被轻微增强。

该结果说明 TVA shadow 并未退化为随机扰动：其跨周期最强削弱集中在同一组动作方向规则上，且方向稳定。但它也没有简单复刻现有 rule-effect 审计。例如部分被 TVA 持续削弱的规则在 rule-effect 审计中仍有较高方向一致率，这说明 rule-effect 的“实际动作与规则同向”并不等价于“该动作族具有反事实价值”。因此，TVA 确实提供了现有统计之外的信号，但当前证据还不足以直接进入在线规则置信度更新。

据此，下一步不宜立即重训练验证性能，而应先做“TVA shadow 与后续周期真实规则效应”的滞后对齐检查：用 `cycle041` 的 shadow 建议对照 `cycle042-050` 的规则效应变化，用 `cycle050` 的 shadow 建议对照 `cycle051-060` 的规则效应变化。只有当 TVA shadow 对后续效果具有稳定预测性时，才进入 `TVA-gated live update`。

### TVA shadow 滞后预测验证

根据上述判断，本日新增 [audit_tva_shadow_lag_effects.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_lag_effects.py)，用于检查 TVA shadow 建议是否能够预测后续周期的 rule-effect 表现。默认窗口为：

```text
cycle041 shadow -> cycle042-050 rule-effect audits
cycle050 shadow -> cycle051-060 rule-effect audits
```

输出文件为 [tva_shadow_lag_effects.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/tva_shadow_lag_effects.md) 与同名 JSON。该分析只检验预测关联，不检验因果收益，因为 TVA shadow 并未参与这些后续周期的训练或规划。

滞后验证结果如下：

| anchor -> future | group | count | seen later | future consistent / harmful | future - anchor consistent / harmful |
| --- | --- | ---: | ---: | ---: | ---: |
| `041 -> 042-050` | strengthened | `11` | `8` | `0.593 / 0.407` | `-0.004 / +0.004` |
| `041 -> 042-050` | weakened | `14` | `11` | `0.630 / 0.370` | `-0.008 / +0.008` |
| `041 -> 042-050` | unchanged | `118` | `79` | `0.647 / 0.353` | `-0.001 / +0.001` |
| `050 -> 051-060` | strengthened | `10` | `6` | `0.537 / 0.463` | `-0.001 / +0.001` |
| `050 -> 051-060` | weakened | `15` | `11` | `0.689 / 0.311` | `-0.002 / +0.002` |
| `050 -> 051-060` | unchanged | `132` | `80` | `0.673 / 0.327` | `+0.003 / -0.003` |

该结果不支持将 TVA shadow 直接升级为 live update。原因有三点。第一，strengthened 与 weakened 组在后续窗口中的 rule-effect 差异没有形成稳定的同向预测关系；第二，后续相对锚点的一致率和有害率变化幅度很小，多数在 `0.001-0.008` 范围；第三，在 `050 -> 051-060` 窗口中，weakened 组后续方向有害率反而低于 strengthened 组。由此可见，当前 TVA shadow 已能提供局部反事实动作价值信号，但尚不能证明其对后续在线规则效应具有可靠预测力。

因此，当前不应进入重训练或在线更新阶段。下一步应集中分析 TVA shadow 与 rule-effect 指标不一致的原因：可能是 rule-effect 的“实际动作同向/反向”指标不足以表达长期价值，也可能是 TVA probe action set 与 planner 实际候选分布不一致，或是 TVA 价值网络在局部反事实状态上仍存在外推误差。

### TVA shadow 与 rule-effect 不一致诊断

为解释上述不一致，本日新增 [audit_tva_shadow_mismatches.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_mismatches.py)。该脚本定位两类规则：`tva_weakened_effect_consistent`，即 TVA shadow 削弱但 rule-effect 方向一致率较高；`tva_strengthened_effect_harmful`，即 TVA shadow 增强但 rule-effect 方向有害率较高。随后脚本回到 TVA 反事实记录，重建每条规则在 TVA 样本中的实际动作族、TVA 最优动作族、规则动作族和相反动作族的关系。

输出文件为 [tva_shadow_mismatch_audit.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/tva_shadow_mismatch_audit.md)。三轮诊断结果如下：

| cycle | mismatches | strengthened but effect harmful | weakened but effect consistent |
| ---: | ---: | ---: | ---: |
| `041` | `8` | `4` | `4` |
| `050` | `12` | `4` | `8` |
| `060` | `9` | `1` | `8` |

主要发现如下。第一，TVA 样本覆盖相对 rule-effect 全量命中极低。若干关键规则的 TVA/effect hit ratio 只有 `0.00039-0.00128`，例如 `cycle060` 中 `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` 的 TVA 命中为 `42`，而 rule-effect selected gate hits 为 `107445`。这说明当前 `300` 条分层 TVA counterfactual audit 只适合作为旁路抽样信号，不足以作为正式置信度更新依据。

第二，若干被 TVA 持续削弱的 `hip_neg` 规则呈现清晰的“实际动作同向但 TVA 认为相反动作更优”模式。例如 `cycle060` 中 `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` 的 rule-effect 一致率为 `0.676`，但在 TVA 样本内，TVA 最优动作族为相反 `hip_pos` 的比例达到 `0.714`。这说明 rule-effect 的“实际动作与规则同向”并不等价于“该动作族具有反事实价值”；它可能只是记录了当前策略经常这样做。

第三，部分 TVA 增强但 rule-effect 有害率较高的规则，TVA 最优动作族分布较散。例如若干 `ankle_neg` 或 `waist_neg` 规则虽然获得小幅 shadow 增强，但其 TVA best family entropy 较高，说明该增强可能来自少量局部 top-k 命中，而不是稳定动作族偏好。

据此形成当前判断：TVA shadow 的问题不是“没有信号”，而是“信号尚未达到在线更新所需的覆盖率和稳定性”。下一版不应重训练，而应先修订 TVA shadow 判据，至少加入三类门控：一是 per-rule TVA 最小命中与 TVA/effect 覆盖率下限；二是 best-family entropy 上限，避免分散最优动作族造成伪增强；三是 target-vs-opposite margin，要求规则动作族相对相反动作族具有明确优势后才允许增强。削弱规则也应区分“TVA 明确支持相反动作”与“规则动作族未被 probe 支持”两种情况，前者可以削弱，后者只应标记为证据不足。

### TVA shadow gated update 修订与复核

根据上述诊断，本日修订 [apply_tva_shadow_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_shadow_update.py)，将 TVA shadow update 从第一版直接加权改为更保守的 gated 口径。新增门控包括：`--min-effect-hit-ratio 0.001`、`--max-best-family-entropy 0.75`、`--target-opposite-margin 0.50` 和 `--min-directional-rate-margin 0.05`。其中方向比例间隔用于避免“相反动作族支持比例更高，但少数目标动作族高优势样本抬高均值后仍被增强”的错误；best-family entropy 同时用于增强与削弱，避免高熵 TVA 最优动作族导致伪增强或伪削弱。

闭环入口 [run_hopper_v7_closed_loop.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py) 与 [run_hopper_v7_multiseed_shared_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py) 同步暴露 `--tva-shadow-min-directional-rate-margin`，默认值为 `0.05`。该参数只影响 `reports/tva_shadow_update.json/md` 的影子建议，不改变当前在线规则库、planner prior 或训练路径。

语法检查如下：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/apply_tva_shadow_update.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

随后重新回填 `cycle041 / cycle050 / cycle060` 的 gated shadow update，并重新生成 [tva_shadow_validation_summary_gated.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/tva_shadow_validation_summary_gated.md)、[tva_shadow_lag_effects_gated.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/tva_shadow_lag_effects_gated.md) 与 [tva_shadow_mismatch_audit_gated.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/tva_shadow_mismatch_audit_gated.md)。最终结果如下：

| cycle | rules | strengthened | weakened | unchanged | directional conflicts | mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `041` | `143` | `0` | `0` | `143` | `0` | `0` |
| `050` | `157` | `0` | `0` | `157` | `0` | `0` |
| `060` | `180` | `2` | `3` | `175` | `0` | `0` |

`cycle060` 保留的两条增强规则为 `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` 与 `transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:hip_pos`，TVA same/opposite 分别为 `0.725/0.039` 和 `0.712/0.034`，best-family entropy 分别为 `0.456` 和 `0.444`。三条削弱规则均满足相反方向支持比例更高且 best-family entropy 低于阈值。

该复核表明，gated TVA shadow 已消除第一版中主要的方向矛盾和高熵伪信号。但由于 `cycle041/050` 在保守门控下不再产生影子改动，当前滞后预测报告没有足够 strengthened/weakened 样本来证明跨周期预测力。因此，TVA 仍应停留在审计/影子层；下一步应扩大或改进 TVA counterfactual 覆盖，再检验 gated 信号是否能在更多周期中形成可预测的规则价值样本。

### TVA counterfactual 覆盖增强复核

为检验上一节中“样本不足”是否是主要限制，本日修订 [audit_hopper_tva_counterfactuals.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_counterfactuals.py)，新增 `--sample-mode label_event`。该模式按 `label × event` 分层抽取反事实状态，避免原 `stratified` 模式只按 good/bad/middle label 分层而使少数事件类型覆盖不足。脚本同时增加 trace CSV 缓存，避免扩大审计样本后重复读取同一轨迹文件。闭环入口同步新增 `--tva-audit-sample-mode`，默认仍为 `stratified`，因此既有实验行为不变。

smoke 测试使用 `cycle060`、`60` 条样本验证通过：样本均分到 `12` 个 `label|event` 组，每条记录写出 `25` 个 `probe_actions`，并包含 `actual_action_family` 与 `best_action_family`。随后对 `cycle041 / cycle050 / cycle060` 使用 `label_event` 模式各审计 `1200` 条样本，并生成 `label_event1200_gated` 与 `label_event1200_strict_gated` 两组 shadow 报告。

覆盖增强后的基础 gated 结果如下：

| cycle | source records | TVA records | matched pairs | strengthened | weakened | unchanged | mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `041` | `4980` | `1200` | `7956` | `8` | `7` | `128` | `5` |
| `050` | `18020` | `1200` | `7957` | `6` | `9` | `142` | `5` |
| `060` | `21560` | `1200` | `8525` | `8` | `9` | `163` | `5` |

与 300 条抽样相比，TVA selected-gate 覆盖显著增加，并恢复了跨周期稳定的候选信号。例如 `transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` 在三轮均被增强，平均 delta 为 `+0.1259`；若干 `omega_pos/vx_high/support_none/falling` 下的 `hip_neg` 或 `ankle_pos` 规则在三轮均被削弱。但 mismatch 也随之重新出现，说明覆盖增强本身不能解决 TVA 与 rule-effect 指标的语义冲突。

进一步使用更严格门控 `--min-effect-hit-ratio 0.003` 与 `--max-best-family-entropy 0.65` 复核，结果如下：

| cycle | strengthened | weakened | unchanged | mismatches |
| ---: | ---: | ---: | ---: | ---: |
| `041` | `1` | `3` | `139` | `3` |
| `050` | `4` | `5` | `148` | `1` |
| `060` | `7` | `4` | `169` | `0` |

严格门控保留了跨周期增强候选，包括 `transition...hip_pos`、`rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` 与 `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_pos`，同时将 mismatch 总数从 `15` 降至 `4`。据此，`apply_tva_shadow_update.py` 与闭环入口的默认审计阈值已调整为 `min_effect_hit_ratio=0.003`、`max_best_family_entropy=0.65`。该调整仍只影响 TVA shadow 报告，不改变在线规则库。

当前结论为：TVA 的问题不再主要是“完全没有规则价值信号”，而是“TVA 反事实价值与 rule-effect 实际动作统计存在可解释但尚未闭合的语义差异”。特别是部分 `hip_neg` 削弱规则呈现 `actual_alignment_but_tva_opposite`，即策略实际经常同向执行，但 TVA 认为相反动作族更有终端价值。这类信号可能正是 TVA 的新增价值，也可能是 TVA 局部外推误差；在未建立更直接的后果验证前，仍不应进入 live update。

### TVA shadow 后果层验证

为检验 TVA shadow 建议是否不仅在反事实价值中成立，也能在真实轨迹后续窗口中得到支持，本日新增 [audit_tva_shadow_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_consequences.py)。该脚本读取严格门控后的 shadow update 与 `label_event1200` TVA counterfactual audit，回到每条记录的 `trace_csv + step`，复用 `row_parts`、`state_matches`、`selected_gate_applies` 和 `selected_action_relation`，按 `same / opposite / neutral` 统计真实动作后的短窗口结构后果。

后果分数 `outcome_score` 只用于审计，主要由终端 continuation score、continuation 变化、最小支撑接触、最小旋转控制、最大碰撞和最大跌落组成。对于 TVA 增强规则，期望 `same` 动作族后果优于 `opposite`；对于 TVA 削弱规则，期望 `opposite` 后果优于 `same`。该口径不能作为因果估计，因为真实动作并非随机分配，但可以检验 TVA 建议是否至少与已发生轨迹中的短期结构后果相容。

使用 `label_event1200_strict_gated` 的三轮 shadow 规则进行验证，`horizon=10` 结果如下：

| cycle | changed rules | compared rules | consequence-supported | support rate |
| ---: | ---: | ---: | ---: | ---: |
| `041` | `4` | `4` | `1` | `0.250` |
| `050` | `9` | `9` | `4` | `0.444` |
| `060` | `11` | `9` | `3` | `0.333` |

为排除窗口长度影响，又执行 `horizon=3` 与 `horizon=5` 的敏感性检查：

| horizon | cycle041 | cycle050 | cycle060 |
| ---: | ---: | ---: | ---: |
| `3` | `0.250` | `0.444` | `0.667` |
| `5` | `0.250` | `0.444` | `0.667` |
| `10` | `0.250` | `0.444` | `0.333` |

按规则聚合后，较稳定得到后果层支持的规则主要包括：`rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` 的增强、`rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_neg` 的削弱，以及 `unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` 的增强。相反，`rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` 的增强在 `cycle050/060` 与三个 horizon 上均被真实后果反向；若干 `hip_neg` 削弱规则也显示同向实际动作后果更好。这说明 TVA 对局部反事实动作族的偏好仍不能直接转化为可靠规则价值。

据此，本阶段结论进一步收紧：TVA shadow 已能产生候选规则价值信号，但只有一部分通过真实后果层验证。下一步不应启动 live confidence update 或重训练；应把“后果层支持”加入 TVA shadow 的候选筛查流程，或至少将其作为进入在线更新前的必要审计门槛。

### TVA consequence-gated shadow update

根据后果层验证结果，本日新增 [apply_tva_consequence_gate.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_consequence_gate.py)，将真实轨迹短窗口后果审计纳入 TVA shadow 候选筛查。该脚本读取原始 strict-gated shadow update 与 consequence audit，生成新的 consequence-gated shadow 报告。未通过后果门控的 shadow 改动会回退到原始 confidence，并将 `shadow_update_reason` 改为 `tva_shadow_rejected_by_consequence_gate`；源规则库和在线 planner prior 不被修改。

本次采用 `horizon=5` 的 consequence audit 作为门控依据，原因是它与 TVA counterfactual audit 的默认 rollout horizon 一致。默认门槛为：期望方向与相反方向均至少 `10` 个真实轨迹样本，且期望方向的 `outcome_score` 均值不低于相反方向。

门控前后对比如下：

| cycle | strict-gated changed | consequence kept | consequence rejected | final strengthened | final weakened | mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `041` | `4` | `1` | `3` | `0` | `1` | `1` |
| `050` | `9` | `4` | `5` | `2` | `2` | `0` |
| `060` | `11` | `4` | `7` | `2` | `2` | `0` |

保留下来的主要候选包括：`unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` 的削弱，`unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` 与 `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` 的增强，`rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_neg` 与 `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_neg` 的削弱，以及 `transition...omega_pos|vx_high...hip_pos` 在 `cycle060` 的增强。

后果门控后，先前反复出现的 `hip_pos` 增强和 `hip_neg` 削弱大多被过滤掉，说明这些 TVA 反事实建议虽然有局部价值信号，但缺少真实轨迹短窗口后果支持。`cycle050 -> 051-060` 滞后检查中，保留下来的 `knee_pos` 增强规则后续方向一致率约为 `0.669`，较锚点高约 `0.007`；但该证据仍是预测关联，不是因果收益。

当前结论为：TVA 规则价值信号经过覆盖增强、严格门控和后果门控后，已形成少量更可信的候选；但候选数量仍少，且尚未经过在线训练闭环验证。因此下一步可设计一次 audit-only 或 shadow-only 的闭环采集实验，让 TVA consequence gate 在周期内持续产出候选，但仍不改变在线 prior；待跨周期稳定性进一步确认后，再讨论是否进入小规模 live update。

### TVA consequence gate 闭环入口接入

根据上述结论，本日将 consequence gate 接入现有闭环脚本，但保持默认关闭和 audit-only。修改点包括：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

`CyclePaths` 新增两个周期输出：`reports/tva_shadow_consequence_audit.json` 与 `reports/tva_shadow_consequence_gated_update.json`。闭环入口新增参数 `--enable-tva-consequence-gate`、`--tva-consequence-horizon`、`--tva-consequence-action-threshold`、`--tva-consequence-min-outcome-margin` 和 `--tva-consequence-min-relation-count`。启用 `--enable-tva-audit --enable-tva-shadow-update --enable-tva-consequence-gate` 后，每周期命令链变为：

```text
TVA counterfactual audit
-> TVA strict shadow update
-> TVA shadow consequence audit
-> TVA consequence-gated shadow update
```

该接入不改变 `finalize_cycle_library` 的返回值，下一周期训练仍读取原 confidence 规则库；若只启用 consequence gate 而未启用 TVA shadow update，则会打印跳过提示。

验证包括两部分。第一，执行语法检查：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  experiments/sd_retrospective_rule_induction/audit_tva_shadow_consequences.py \
  experiments/sd_retrospective_rule_induction/apply_tva_consequence_gate.py
```

第二，执行 `run_hopper_v7_multiseed_shared_rules.py --dry-run`，确认命令序列中已按预期出现 `audit_hopper_tva_counterfactuals.py`、`apply_tva_shadow_update.py`、`audit_tva_shadow_consequences.py` 和 `apply_tva_consequence_gate.py`，且最终 shared rule library 仍指向 `library_closed_loop_cycle_001.json`，不是 TVA shadow 输出。

### TVA consequence gate shadow-only 短程闭环采集启动

为观察 TVA consequence-gated shadow 候选是否能在真实闭环周期中稳定自动产生，本日启动短程采集实验。该实验从旧方案 `300k / cycle060` 产物接续，仅运行 `cycle061` 至 `cycle063` 三个 `5000` 步周期，总步数至约 `315k`。在线训练仍使用原 confidence 规则库作为 planner prior；TVA audit、TVA shadow update 与 consequence gate 均保持 shadow-only，不写入下一周期实际规则库。

实验启动前确认以下接续输入存在：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/cycle_060/rules/library_closed_loop_cycle_060_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/cycle_060/sd_explainers/sd_explainer_cycle_060.pt

initial_tva_checkpoint =
results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/hopper_tva_cycle_060.pt
```

首次使用 `r2` 运行名进行后台启动时，`nohup bash` 在当前 shell 环境中出现空日志、无结果目录、无进程的异常。随后使用 `bash -x` 验证脚本参数链正确，并确认训练可进入 `cycle061`；该调试启动产生的子进程已停止，`r2` 目录仅作为调试残留，不作为正式结果使用。

正式采集使用新的 `r3` 运行名：

```text
script =
scripts/launch_hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.sh

output_root =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3

launcher_log =
logs/sd_retrospective_rule_induction/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.debug.log
```

由于 `nohup bash` 在本环境中未能可靠保留进程，正式启动采用：

```text
setsid bash -x scripts/launch_hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.sh \
  > logs/sd_retrospective_rule_induction/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.debug.log 2>&1 < /dev/null &
```

启动后已确认主控进程与三个 seed 训练进程存在，`cycle061` 的三个训练日志均显示从 `step=300060` 恢复，并安装 `76` 条 confidence-active 规则。`run_config.json` 已确认包含：

```text
enable_tva_audit = true
tva_audit_sample_mode = label_event
tva_audit_max_records = 1200
enable_tva_shadow_update = true
enable_tva_consequence_gate = true
tva_consequence_horizon = 5
tva_consequence_min_relation_count = 10
tva_consequence_min_outcome_margin = 0.0
```

本次采集的判读重点不是 `315k` 的短程性能变化，而是每周期是否完整生成 `tva_shadow_consequence_audit.json` 与 `tva_shadow_consequence_gated_update.json`，以及 consequence gate 是否继续过滤掉缺少真实后果支持的 TVA shadow 候选。

### TVA consequence gate shadow-only 短程闭环采集完成

`hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3` 已完整运行至 `cycle063 / 315k`，主控进程正常退出。最终输出包括：

```text
final shared rule library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/rules/library_closed_loop_cycle_063_confidence.json

final shared SD checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/sd_explainers/sd_explainer_cycle_063.pt

final shared TVA checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/tva/hopper_tva_cycle_063.pt
```

三轮 TVA consequence-gated shadow 结果如下：

| cycle | rules | original changed | kept changed | rejected changed | strengthened | weakened | unchanged |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `061` | `183` | `0` | `0` | `0` | `0` | `0` | `183` |
| `062` | `183` | `0` | `0` | `0` | `0` | `0` | `183` |
| `063` | `184` | `7` | `3` | `4` | `0` | `3` | `181` |

`cycle063` 的 consequence audit 显示 `changed rules = 7`，`support rate among compared = 0.714`。经 consequence gate 保留的三条改动均为削弱信号：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:hip_pos
rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_neg
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:hip_pos
```

对应 shadow delta 分别约为 `-0.096`、`-0.032`、`-0.083`。四条未通过者被标记为 `tva_shadow_rejected_by_consequence_gate`，其中部分规则虽有 TVA 方向信号，但缺少足够的相反方向真实样本，或真实短窗口后果方向不支持，因此被回退为原始 confidence。

本次短程采集表明，TVA consequence gate 可以在完整闭环中稳定运行，并且不会污染在线规则库；它在前两轮保持零改动，在第三轮只保留后果支持的少量削弱候选。这一结果支持继续将 TVA 作为规则价值审计与候选筛查层使用，但仍不足以直接进入 live confidence update。下一步应围绕 `cycle063` 的 7 条候选做定性审查，判断被保留的三条削弱是否符合轨迹后果和动作语义，再决定是否扩大 shadow-only 周期或设计更小规模的可控 live update。

### TVA shadow 候选定性审查

已对 `cycle063` 的 7 条 TVA shadow 候选进行定性审查，并形成报告：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/reports/tva_shadow_candidate_review.md
```

审查结论如下。第一，`unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:hip_pos` 的削弱较可信。TVA same/opposite 比例为 `0.048 / 0.738`，真实短窗口中相反动作关系的 outcome score、reward、terminal support、falling risk 和 terminal forward speed 均优于同向关系。该候选与“后仰、无支撑、正向旋转状态下继续鼓励 hip_pos 可能加剧失衡”的解释一致。

第二，`transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` 的削弱方向基本合理，但结构指标混合。相反动作关系具有更好的 good-label 组成、更低 falling risk 和更高 terminal forward speed，但 terminal support 不优于同向关系。因此它适合被解释为“削弱过度信任 hip_pos transition 规则”的信号，而不是形成反向规则的充分证据。

第三，`rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_neg` 的削弱是低优先级清理信号。其原 confidence 已接近零，后果 margin 仅 `+0.0054`，且相反关系并未在 terminal support 和 forward speed 上占优。该候选不应被视为强行为原则。

四条被拒绝候选总体符合门控预期：两个候选虽有正向 outcome margin，但同向或相反关系样本数低于 `10`；两个低速 falling 条件下的 hip 规则则出现 TVA 建议与真实短窗口后果相反，门控正确回退。该审查进一步说明 consequence gate 的主要作用不是增加候选数量，而是阻止 TVA 反事实价值信号直接越过真实后果验证。

当前结论维持不变：TVA consequence gate 已可作为规则价值审计层继续运行，但仍不宜直接接入 live confidence update。下一步更合适的是继续扩大 shadow-only 覆盖，或在离线回放层面设计“只应用 TVA-kept 削弱”的反事实 replay 检查，而不是立即重训。

### TVA-kept shadow eval-only 对照启动

为避免直接进入重训，本日新增 eval-only 对照工具：

```text
experiments/sd_retrospective_rule_induction/build_tva_kept_shadow_library.py
experiments/sd_retrospective_rule_induction/eval_hopper_tva_kept_shadow_315k.py
```

首先基于 `cycle063` 的原 confidence 规则库与 consequence-gated shadow 报告生成 eval-only 规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/rules/library_closed_loop_cycle_063_confidence_tva_kept_shadow_eval.json
```

本次 consequence gate 保留了 3 条 shadow 削弱，但其中 `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_neg` 在原规则库中已经是 `confidence_retired`，不会被 `rule_prior_allowed_statuses=confidence_active` 读取。因此 eval-only 副本实际改动 2 条在线可作用规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:hip_pos
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:hip_pos
```

随后启动三种子、三模式、每模式 `10` 条 episode 的 checkpoint eval-only 对照：

```text
rule_off
original
tva_kept_shadow
```

启动命令为：

```text
setsid /home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/eval_hopper_tva_kept_shadow_315k.py \
  --eval-episodes 10 --no-resume \
  > logs/sd_retrospective_rule_induction/hopper_315k_tva_kept_shadow_eval_20260516.log 2>&1 < /dev/null &
```

该评估只加载 `315k` TD-MPC2 checkpoint，不训练策略、不修改原规则库、不更新 SD 或 TVA。输出目录为：

```text
results/sd_retrospective_rule_induction/reports/hopper_315k_tva_kept_shadow_eval_20260516
```

预期判读口径为：若 `tva_kept_shadow` 相对 `original` 的得分和规则命中统计没有明显变化，则说明当前保留的 TVA 削弱虽有审计信号，但对实际 planner 排序影响很轻；若出现稳定改善或恶化，则进一步审查对应 eval 轨迹中的规则命中和动作方向变化。

## 2026-05-16

### TVA-kept shadow eval-only 对照与行为方向审计

`315k` checkpoint 的三种子、三模式、每模式 `10` 条 episode 的 eval-only 对照已完成。输出目录为：

```text
results/sd_retrospective_rule_induction/reports/hopper_315k_tva_kept_shadow_eval_20260516
```

三种模式的标准 eval 均值如下：

| mode | seed means | mean |
| --- | --- | ---: |
| `rule_off` | `145.481; 324.274; 299.538` | `256.431` |
| `original` | `168.801; 318.133; 290.249` | `259.061` |
| `tva_kept_shadow` | `168.825; 316.852; 288.372` | `258.016` |

`original` 相对 `rule_off` 的均值差为 `+2.631`，`tva_kept_shadow` 相对 `original` 的均值差为 `-1.045`。因此，在本次 eval-only 条件下，TVA consequence gate 保留的削弱信号没有带来可见的短期得分收益，但也未造成显著退化。

随后使用 `audit_hopper_v8_widefunnel_rule_effects.py` 对 `original` 与 `tva_kept_shadow` 的 eval 轨迹进行规则行为方向审计。审计输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_315k_tva_kept_shadow_eval_20260516/original_eval_rule_effect_audit.json
results/sd_retrospective_rule_induction/reports/hopper_315k_tva_kept_shadow_eval_20260516/tva_kept_shadow_eval_rule_effect_audit.json
```

两条在线可作用的 TVA-kept 削弱规则表现如下：

| rule | mode | hits | same | opposite | neutral | consistent% | harmful% | good_state% | high_ep% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:hip_pos` | `original` | `64` | `32` | `31` | `1` | `51.6` | `48.4` | `7.8` | `89.1` |
| `unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:hip_pos` | `tva_kept_shadow` | `68` | `29` | `38` | `1` | `44.1` | `55.9` | `2.9` | `86.8` |
| `transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` | `original` | `600` | `333` | `256` | `11` | `57.3` | `42.7` | `53.3` | `99.8` |
| `transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` | `tva_kept_shadow` | `598` | `317` | `273` | `8` | `54.3` | `45.7` | `52.7` | `99.5` |

审计显示，`tva_kept_shadow` 的确改变了 planner 最终动作排序：第一条规则的同向比例下降 `7.44` 个百分点，第二条规则的同向比例下降 `2.99` 个百分点，对应反向比例同步上升。也就是说，TVA-kept 削弱不是只停留在规则库数值上，而是已经转化为可观察的行为差异。

但该行为差异仍然很轻，且没有形成得分提升。尤其是第一条规则在 `tva_kept_shadow` 中的 `good_state%` 从 `7.8%` 降至 `2.9%`，说明被削弱规则更多命中在非理想状态；第二条规则虽有较高命中数，但其方向变化只有约 `3` 个百分点。由此可判断，当前 TVA consequence gate 的 kept 削弱具有行为可达性和一定解释力，但不足以支持直接进入 live confidence update 或重训。

当前结论为：TVA consequence gate 可继续作为规则价值审计层保留；下一阶段不宜把单次 `cycle063` 的轻量削弱直接上线，而应继续扩大 shadow-only 覆盖，或设计一个可回滚的小规模 live-update smoke，并限定为“只削弱 confidence-active 规则”。判断重点仍应放在规则价值，即候选规则是否经过 TVA、真实短窗口后果和显式行为方向三重验证，而不是短期 eval 均值本身。

### 315k 至 350k TVA consequence-gated shadow-only 扩展实验启动

为检验 TVA-kept 削弱是否具有跨周期稳定性，本日新增独立启动脚本：

```text
scripts/launch_hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1.sh
```

该实验从 `315k / cycle063` 接续至 `350k / cycle070`，新增 `7` 个 `5k` 周期。实验不启用 TVA live update，不修改在线规则库，只继续执行 TVA audit、TVA shadow update 与 consequence gate，用于扩大规则价值审计覆盖。输入起点如下：

```text
initial rule library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/rules/library_closed_loop_cycle_063_confidence.json

initial SD =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/sd_explainers/sd_explainer_cycle_063.pt

initial TVA =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3/cycle_063/tva/hopper_tva_cycle_063.pt
```

三种子 TD-MPC2 checkpoint 均从 `tdmpc2_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3_cycle063_target315062` 的 `final.full.pt` 恢复。新实验输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1
```

启动命令为：

```text
setsid bash -x scripts/launch_hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1.sh \
  > logs/sd_retrospective_rule_induction/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1.debug.log 2>&1 < /dev/null &
```

启动后已确认主控进程和三个 seed 训练进程存在，`cycle064` 的目标步数为 `320063`。三种子训练日志均显示从 `step=315063` 恢复，并安装 `77` 条 `confidence_active` 规则；`SD_ONLINE_SUPERVISION_ENABLED=false`，规则仍只通过低强度 MPPI candidate reweighting 起作用。

本次扩展实验的主要判读口径为：每周期记录 `tva_shadow_update.json`、`tva_shadow_consequence_audit.json` 与 `tva_shadow_consequence_gated_update.json`；观察 kept/rejected 候选是否跨周期稳定，尤其是已出现的 hip_pos 削弱信号是否继续被真实短窗口后果支持。若多周期保持同向且行为方向审计可见，再考虑进入可回滚的小规模 live-update smoke。

### 350k shadow-only 扩展实验阶段进展

`hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1` 已完成 `cycle064` 与 `cycle065` 的训练、回放导出、SD 接续拟合、TVA 接续训练、规则更新、rule-effect confidence 更新，以及 TVA consequence-gated shadow 审计。当前正在运行 `cycle066` 三种子训练，目标步数为 `330065`。

已完成周期的关键结果如下：

| cycle | confidence rules | active / observing / retired | TVA original changed | kept changed | rejected changed | note |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| `064` | `186` | `77 / 29 / 80` | `2` | `1` | `1` | kept 为 `omega_neg vx_low rear -> prefer:waist_pos` 的削弱 |
| `065` | `189` | `80 / 28 / 81` | `5` | `2` | `3` | kept 为一条削弱与一条低置信增强 |

`cycle065` 中 consequence gate 保留的两条 shadow 改动为：

```text
unsupported_high_torque|omega_neg|vx_low|support_none|falling->prefer:hip_pos
  delta = -0.060
  reason = tva_shadow_harmful_deactivated
  consequence margin = +0.0160

rotation_risk|omega_neg|vx_low|support_some|rear->prefer:waist_neg
  delta = +0.0047
  reason = tva_shadow_consistent_strengthened
  consequence margin = +0.0163
```

其中第二条原状态为 `confidence_retired`，即使被 shadow 增强，在当前 `rule_prior_allowed_statuses=confidence_active` 口径下仍不会直接上线；第一条是在线可作用规则的轻量削弱。`cycle065` 中此前关注的 `omega_pos / vx_mid / rear -> hip_pos` 相关削弱被 consequence gate 拒绝，说明 TVA 信号仍会被真实短窗口后果层约束，而不是机械放大。

当前训练日志显示，`cycle066` 三个 seed 均已从 `cycle065` 的 `final.full.pt` 正常恢复，最近训练片段约在 `328k-329k` 步附近；seed2 与 seed3 的单 episode 回报多在 `285-307` 区间，seed1 波动较大，出现过 `216-256` 以及一次 `0` 回报。该信息仅作为运行状态记录，不作为规则价值判断依据。后续仍以 `cycle066-070` 的 kept/rejected 稳定性和行为方向审计为主要判据。

### TVA 终端补偿能力验证

为避免继续停留在规则候选细节中，本日将 TVA 验证问题收束为一个直接命题：TVA 是否能较准确刻画 SD 推演若干步后终端状态的剩余价值补偿。为此新增审计脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_tva_terminal_compensation.py
```

该脚本读取同一周期的 `selected_trajectories.json`、SD checkpoint 与 TVA checkpoint。对每个采样起点 `s_t`，使用真实动作序列让 SD 推演 `h` 步，得到 `s'_{t+h}`，然后比较 `TVA(s'_{t+h})` 与真实轨迹中从 `t+h` 开始的折扣剩余目标。脚本同时报告 `TVA(s_{t+h})` 的结果，作为价值网络本身在真实终端状态上的上界参照。这样可以区分两类误差：一是 TVA 价值网络本身拟合不准，二是 SD 推演后的状态分布导致终端补偿失准。

在 `cycle070` 上完成一轮小样本验证：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/reports/tva_terminal_compensation_audit_tiny.json
```

该验证使用 `10` 条 episode、每条最多 `5` 个起点、`h = 1, 3, 5`，共 `150` 个终端补偿样本。结果如下：

| h | mode | n | target std | MAE | RMSE | baseline RMSE | Pearson | Spearman | R2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `oracle_terminal` | `50` | `56.771` | `21.118` | `30.440` | `56.771` | `0.859` | `0.743` | `0.713` |
| `1` | `sd_terminal` | `50` | `56.771` | `21.234` | `30.672` | `56.771` | `0.848` | `0.724` | `0.708` |
| `3` | `oracle_terminal` | `50` | `61.294` | `17.290` | `23.122` | `61.294` | `0.952` | `0.818` | `0.858` |
| `3` | `sd_terminal` | `50` | `61.294` | `21.557` | `29.625` | `61.294` | `0.886` | `0.718` | `0.766` |
| `5` | `oracle_terminal` | `50` | `57.232` | `18.540` | `27.591` | `57.232` | `0.913` | `0.776` | `0.768` |
| `5` | `sd_terminal` | `50` | `57.232` | `22.992` | `32.525` | `57.232` | `0.829` | `0.702` | `0.677` |

初步结论为：TVA 对终端补偿不是无信号。即使在 SD 推演终端状态上，`h=1/3/5` 的 Pearson 分别约为 `0.848 / 0.886 / 0.829`，RMSE 明显低于仅预测样本均值的 baseline RMSE。这说明 TVA 已经具备基本排序与补偿能力，可以作为正式实验中的候选机制继续验证。

同时，该结果还不支持直接把 TVA 作为强权重终端价值项。原因有二：第一，小样本验证的绝对误差仍不小，`sd_terminal` 的 MAE 约为 `21-23`；第二，当前 TVA 训练目标不是纯环境回报，而是 `true_returns + shaped_returns + 0.65 * adjust_returns`，因此它更像“结构化后果价值”而非严格的环境回报价值。若后续进入正式实验，较稳妥的方式是低系数接入、保留 consequence gate 和行为审计，并把 TVA 权重作为可控参数。

本日 `350k` shadow-only 扩展实验已完整结束。最终输出为：

```text
final rule library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/rules/library_closed_loop_cycle_070_confidence.json

final SD =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/sd_explainers/sd_explainer_cycle_070.pt

final TVA =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/tva/hopper_tva_cycle_070.pt
```

`cycle064-070` 的 TVA consequence gate 汇总如下：

| cycle | original changed | kept | rejected | strengthened | weakened |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `064` | `2` | `1` | `1` | `0` | `1` |
| `065` | `5` | `2` | `3` | `1` | `1` |
| `066` | `4` | `2` | `2` | `0` | `2` |
| `067` | `2` | `2` | `0` | `0` | `2` |
| `068` | `2` | `2` | `0` | `1` | `1` |
| `069` | `2` | `2` | `0` | `0` | `2` |
| `070` | `13` | `4` | `9` | `1` | `3` |

`cycle070` 出现 `13` 条原始 shadow 改动，但 consequence gate 仅保留 `4` 条、拒绝 `9` 条，说明后果门控仍在发挥筛选作用。结合终端补偿审计，当前更合理的推进方向不是继续增加 shadow-only 细节，而是设计一个低权重、可回滚的 TVA 正式接入实验：TVA 作为 SD 规划末端补偿或规则价值审计权重进入在线流程，保留显式开关、低系数、标准 TD-MPC2 对照、policy on/off eval 与行为审计。

### TVA 终端先验 375k 接续实验启动

依据上述终端补偿审计结果，本日启动低权重 TVA 终端先验正式接入实验。实验目的不是验证 TVA 单独能否直接提升短期回报，而是检验：当 SD 可对候选动作序列做短视界推演时，TVA 是否能为推演终端提供可用的后果补偿，并由此影响在线 MPPI 候选排序和后续学习轨迹。

启动脚本为：

```text
scripts/launch_hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1.sh
```

输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1
```

实验从 `350k / cycle070` 接续，使用上一轮 shadow-only 实验的最终规则库、SD checkpoint、TVA checkpoint 和三种子 TD-MPC2 `final.full.pt`：

```text
rules = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/rules/library_closed_loop_cycle_070_confidence.json
SD = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/sd_explainers/sd_explainer_cycle_070.pt
TVA = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1/cycle_070/tva/hopper_tva_cycle_070.pt
```

核心新增参数为：

```text
--enable-tva-terminal-prior
--tva-terminal-prior-weight 0.02
--tva-terminal-prior-max-bonus 0.05
```

原规则先验保持轻量口径：`rule_prior_mode=penalty_only`、`rule_prior_weight=0.05`、`rule_prior_opposite_penalty=0.01`、`rule_prior_allowed_statuses=confidence_active`。训练周期仍为 `5000` 步，formal checkpoint 和 10 episode formal eval 周期仍为 `25000` 步。该设计保留了逐步验证和可回滚原则：TVA 只在规划候选排序末端作为弱终端补偿项出现，不直接写入策略网络或 TD-MPC2 价值网络。

启动后日志确认 `cycle071` 三个 seed 均从 `350070` 步恢复，且均显示：

```text
Installed v6 planner priors: rules=80 rule_enabled=True tva_terminal_enabled=True
```

后续观察重点为：第一，`cycle071-075` 是否能稳定完成训练、SD 拟合、TVA 训练与规则审计；第二，TVA 终端先验是否在行为审计中产生可观察的动作排序影响；第三，`375k` formal eval 与轨迹姿态是否相对 `350k` 起点出现一致改善或退化。

### 375k TVA 终端先验最终审计

本日 `hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1` 已完整结束。最终产物为：

```text
final rule library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/rules/library_closed_loop_cycle_075_confidence.json

final SD =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/sd_explainers/sd_explainer_cycle_075.pt

final TVA =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/tva/hopper_tva_cycle_075.pt
```

最终三种子 formal eval 为 `271.2 / 300.3 / 313.0`，均值为 `294.8`。最终规则库共有 `192` 条规则，其中 `confidence_active = 81`、`confidence_observing = 23`、`confidence_retired = 88`。`cycle075` 常规闭环更新未新增普通规则，但 transition replay-safe 分支合并了两条在线候选规则；随后 confidence 审计将规则库更新为上述最终状态。

对最终 official eval 轨迹运行 active 规则作用审计，输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/reports/final_official_eval_rule_effect_audit.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/reports/final_official_eval_rule_effect_audit.md
```

审计结果显示，`30` 条 final official eval episode 中，任一 active 规则命中并进入候选排序门控的帧比例为 `0.7794`，规则命中密度与 episode reward 的相关系数为 `0.6977`，transition 规则命中密度与 reward 的相关系数为 `0.2198`。因此，规则机制在最终评估中并非空转；它确实频繁参与了动作候选排序，并与高回报 episode 存在正相关。

按规则来源分组后，`sd_diagnostic_toe_recovery` 规则 `toe_recovery_low_support->soft_prefer:waist_pos` 命中 `4389` 次，方向一致率约 `0.701`，高质量状态命中率约 `0.794`。`v8_transition_replay_safety` 规则共 `14` 条，命中 `4656` 次，按 same/opposite 加权的一致率约 `0.644`，高质量状态命中率约 `0.741`。相比之下，`v4_substate` 规则共 `66` 条，命中 `33470` 次，但加权一致率约 `0.473`，存在若干高命中且有害率较高的规则，例如 `omega_pos / vx_high / prone -> prefer:ankle_neg` 相关规则。这说明当前 active 规则库不是简单“越多越好”，后续应继续强化 consequence/TVA 对规则质量的筛选。

为避免只看得分，新增并运行只读姿态比较脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py
```

对 `350k` shadow-only 终点与 `375k` TVA terminal prior 终点进行同口径比较，输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/reports/final_official_eval_posture_comparison.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/reports/final_official_eval_posture_comparison.md
```

主要结果如下：

| 指标 | 350k shadow-only | 375k TVA terminal prior |
| --- | ---: | ---: |
| formal eval mean reward | `244.972` | `294.865` |
| reward std | `104.481` | `60.932` |
| structural_success_frac | `0.1124` | `0.1277` |
| positive_reward_frac | `0.4032` | `0.4601` |
| mean_continue_score | `0.3622` | `0.4107` |
| mean_vx | `0.8773` | `1.1259` |
| mean_abs_pitch | `0.9367` | `0.7939` |
| body_collision_mean | `0.2011` | `0.1751` |
| mean_full_foot_support | `0.00650` | `0.00575` |
| toe_only_launch_frac | `0.2008` | `0.2339` |
| prone phase frac | `0.6637` | `0.7787` |

该结果支持一个有限但明确的判断：TVA 终端先验接入后，`350k -> 375k` 阶段出现了得分均值上升、波动下降、速度提高、身体碰撞下降和姿态角改善；但它没有把策略推向标准 Hopper 的全足支撑跳跃。最终高分轨迹仍主要表现为更稳定、更高效的俯身或脚尖弹射式前进，`full_foot_support` 仍接近零，`toe_only_launch_frac` 反而增加。

因此，当前机制的可验证价值在于：规则/TVA prior 能够以轻量方式实际影响规划排序，并改善已有行为模式的效率与稳定性；当前不足在于：规则价值函数仍未充分区分“即时有效但长期姿态不可持续”的脚尖弹射与“阶段性恢复全足支撑”的更长期目标。后续改进应优先围绕规则质量与延迟后果判断展开，而不是简单增加 Hopper 手写规则或继续提高 prior 权重。

### 200k rule+TVA terminal prior 从头训练启动

为检验 TVA terminal prior 是否能在完整学习过程中改变数据分布、规则形成和轨迹姿态，本日启动一条从空规则库、空 TD-MPC2 checkpoint、空 SD checkpoint 和空 TVA checkpoint 开始的新线。该线不继承 `350k` 规则库或策略 checkpoint，避免把后期成熟规则与策略吸引子混入从头训练判断。

启动脚本为：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1.sh
```

输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1
```

主日志为：

```text
logs/sd_retrospective_rule_induction/hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1.debug.log
```

实验协议如下：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 40
cycle_steps = 5000
target_steps = 200k
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_opposite_penalty = 0.01
rule_prior_allowed_statuses = confidence_active
tva_terminal_prior_weight = 0.02
tva_terminal_prior_max_bonus = 0.05
```

规则闭环仍按每 `5k` 周期执行 SD 拟合、反事实归因、v8 transition 规则提取、rule-effect confidence 更新、TVA 训练、TVA counterfactual audit、TVA shadow update 与 TVA consequence gate。formal eval 和 checkpoint 仍按每 `25k` 执行一次。

由于从头训练的 `cycle001` 尚无可用 SD 和 TVA checkpoint，在线 TVA terminal prior 在第一周期自动关闭；`cycle001` 结束后会生成 `sd_explainer_cycle_001.pt` 与 `hopper_tva_cycle_001.pt`，从 `cycle002` 起 TVA terminal prior 才能以低权重进入 MPPI 候选排序。为支持该从头口径，`run_hopper_v7_multiseed_shared_rules.py` 已修订为：当 `--enable-tva-terminal-prior` 打开但当前周期尚无 SD/TVA checkpoint 时，不再报错退出，而是将本周期 `TVA_TERMINAL_PRIOR_ENABLED=false`，待 checkpoint 具备后自动启用。

启动检查显示，主控进程 PID 为 `63581`，`cycle001` 三个 seed 均已启动：

```text
seed1 -> gpu1
seed2 -> gpu2
seed3 -> gpu3
cycle001 target_steps = 5000
initial_rule_library = shared_rules/library_cycle_000_empty.json
rule_prior_enabled = false
tva_terminal_prior_enabled = false
```

后续判断重点为：第一，`25k / 50k / 75k / 100k / ... / 200k` formal eval 是否相对历史 rule-only 主线出现同口径改善；第二，TVA terminal prior 线是否更早形成高质量 active 规则；第三，最终轨迹姿态是否不仅提高回报，还改善 `structural_success_frac`、`continue_score_ge_060_frac`、碰撞控制和支撑结构，而不是单纯提高脚尖弹射效率。

### 200k rule+TVA terminal prior 从头训练完成及 TVA 独立审计

`hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1` 已完成至 `200k`。最终共享规则库、SD checkpoint 和 TVA checkpoint 分别为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1/cycle_040/sd_explainers/sd_explainer_cycle_040.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_fromscratch_seed123_20260516_r1/cycle_040/tva/hopper_tva_cycle_040.pt
```

三种子、每种子 `10ep` 的 formal eval 曲线如下：

| step | mean | seed1 | seed2 | seed3 | max episode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25k | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 50k | 0.0367 | 0.0573 | 0.0528 | 0.0000 | 0.5733 |
| 75k | 1.1194 | 0.5633 | 0.0845 | 2.7105 | 13.6841 |
| 100k | 0.5777 | 1.4212 | 0.1616 | 0.1503 | 6.7561 |
| 125k | 22.6465 | 33.6075 | 0.0252 | 34.3069 | 93.4951 |
| 150k | 15.9795 | 46.9411 | 0.9974 | 0.0000 | 122.0641 |
| 175k | 44.2920 | 82.4117 | 0.0748 | 50.3894 | 170.9918 |
| 200k | 72.7535 | 135.0705 | 0.0003 | 83.1896 | 214.9372 |

最终规则库包含 `156` 条规则，其中 `confidence_active=63`、`confidence_observing=40`、`confidence_retired=53`。最终 `rule_effect_audit` 的 `gate_hit_frac=0.7155`，说明规则机制确实在线命中；但 `200k` 均值低于历史 rule-only `200k` 记录和外部 TD-MPC2 原始 `200k` 记录，主要受 seed2 几乎完全失败影响。

为避免只用在线回报判断 TVA terminal prior，补做了轻量 TVA terminal compensation 审计。该审计比较两种信号：`oracle_terminal` 表示在真实轨迹的 `s_{t+h}` 上评估 TVA，主要检验 TVA 拟合本身；`sd_terminal` 表示先用 SD 从 `s_t` 按真实动作推演 `h` 步，再在推演末端评估 TVA，直接对应规划时的终端补偿用途。审计输出保存为各周期的：

```text
cycle_010/reports/tva_terminal_compensation_audit_light.json
cycle_020/reports/tva_terminal_compensation_audit_light.json
cycle_030/reports/tva_terminal_compensation_audit_light.json
cycle_040/reports/tva_terminal_compensation_audit_light.json
```

关键结果如下：

| cycle | h | oracle Pearson | sd Pearson | oracle R2 | sd R2 | SD state RMSE | sd RMSE / baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 0.744 | 0.712 | 0.471 | 0.403 | 0.916 | 0.773 |
| 10 | 3 | 0.731 | 0.667 | 0.455 | 0.326 | 1.168 | 0.821 |
| 10 | 5 | 0.556 | 0.453 | 0.143 | 0.002 | 1.501 | 0.999 |
| 20 | 1 | 0.772 | 0.674 | 0.507 | 0.360 | 0.787 | 0.800 |
| 20 | 3 | 0.641 | 0.597 | 0.376 | 0.330 | 1.054 | 0.818 |
| 20 | 5 | 0.746 | 0.655 | 0.535 | 0.407 | 1.073 | 0.770 |
| 30 | 1 | 0.599 | 0.604 | 0.241 | 0.333 | 1.110 | 0.817 |
| 30 | 3 | 0.649 | 0.450 | 0.279 | -0.039 | 1.478 | 1.019 |
| 30 | 5 | 0.427 | 0.209 | 0.004 | -0.427 | 1.369 | 1.194 |
| 40 | 1 | 0.807 | 0.868 | 0.607 | 0.738 | 1.108 | 0.512 |
| 40 | 3 | 0.667 | 0.597 | 0.373 | 0.268 | 1.405 | 0.855 |
| 40 | 5 | 0.675 | 0.153 | 0.369 | -0.527 | 1.736 | 1.236 |

该结果支持如下判断：TVA 在真实末端状态上的价值排序并非无效，`oracle_terminal` 多数节点具有正相关；但是一旦接入 SD 推演，较长视界的 `sd_terminal` 信号不稳定，尤其在 cycle 30 和 cycle 40 的 `h=5` 上出现负 `R2`，且误差超过均值基线。这说明本轮失败不应被简单归因于“TVA 完全没有价值信号”，更可能是 SD 推演误差与 TVA 终端补偿耦合后，在较长视界上产生了不可靠排序。

同时，TVA counterfactual audit 显示，最终周期 `cycle040` 的 `actual_top3_rate=0.288`、`mean_actual_rank=13.52`、`mean_tva_advantage_over_actual=4.107`，说明 TVA 经常认为探测动作中存在优于实际动作的替代选择；但 shadow consequence gate 在后期多数周期拒绝或不产生变更，例如 `cycle036-040` 的 kept changed rules 分别为 `0,0,0,0,0`。这表明 TVA 的价值信号尚未稳定转化为可被后果门控接受的规则信心更新。

阶段性结论为：TVA 可以保留为有潜力的方向，但不宜直接把当前 `h=5` SD terminal compensation 作为正式增强项继续放大训练。后续应优先检查 TVA 的目标构造、归一化尺度、训练数据分布，以及 terminal prior 的有效视界；在未证明 `sd_terminal` 信号随训练稳定改善之前，应避免仅凭在线回报波动判断其价值。

### TVA-SD 接口校准审计

为避免将上述现象误写为“TVA 与 SD 耦合导致 TVA 自身失效”，新增 `audit_hopper_tva_sd_interface.py`。该脚本将问题重新定义为接口校准：TVA 在真实末端状态 `s_{t+h}` 上的估值是否与其在 SD 推演末端状态 `SD(s_t, a_t, ..., a_{t+h-1})` 上的估值一致，以及不一致主要来自哪些状态分量。

脚本输出如下：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_tva_sd_interface.py
```

本次在 `cycle020 / 100k`、`cycle030 / 150k` 和 `cycle040 / 200k` 上使用轻量配置运行：

```text
horizons = 1,3,5
max_episodes = 20
max_starts_per_episode = 2
device = cpu
```

输出文件为：

```text
cycle_020/reports/tva_sd_interface_audit_light.json
cycle_030/reports/tva_sd_interface_audit_light.json
cycle_040/reports/tva_sd_interface_audit_light.json
```

核心结果如下：

| cycle | h | oracle r | sd r | oracle R2 | sd R2 | mean abs(delta V) | state RMSE | angle RMSE | velocity RMSE | touch RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1 | 0.718 | 0.723 | 0.472 | 0.461 | 2.003 | 0.872 | 0.398 | 1.205 | 0.413 |
| 20 | 3 | 0.632 | 0.705 | 0.347 | 0.413 | 3.637 | 1.082 | 0.517 | 1.482 | 0.605 |
| 20 | 5 | 0.665 | 0.550 | 0.438 | 0.260 | 5.135 | 1.358 | 0.587 | 1.884 | 0.741 |
| 30 | 1 | 0.444 | 0.397 | -0.196 | -0.192 | 3.846 | 0.986 | 0.500 | 1.345 | 0.573 |
| 30 | 3 | 0.526 | 0.115 | -0.010 | -0.795 | 9.819 | 1.919 | 0.779 | 2.759 | 0.746 |
| 30 | 5 | 0.393 | 0.351 | -0.099 | -0.235 | 9.176 | 1.778 | 0.676 | 2.547 | 0.763 |
| 40 | 1 | 0.605 | 0.602 | 0.317 | 0.320 | 6.049 | 1.269 | 0.654 | 1.788 | 0.596 |
| 40 | 3 | 0.686 | 0.569 | 0.421 | 0.279 | 8.757 | 1.547 | 0.739 | 2.151 | 0.847 |
| 40 | 5 | 0.718 | 0.551 | 0.497 | 0.218 | 10.283 | 2.153 | 0.864 | 3.120 | 0.772 |

该审计支持更精确的解释：TVA 在真实末端状态上的价值信号仍然存在，且 `cycle040 h=5` 的 oracle Pearson 为 `0.718`、oracle R2 为 `0.497`；但随着视界从 `h=1` 增至 `h=5`，`|sd_value - oracle_value|` 明显增加，`cycle040` 从 `6.049` 增至 `10.283`。同时，速度分量误差随视界增长最明显，`cycle040` 的 `velocity_rmse` 从 `1.788` 增至 `3.120`，大于角度和接触误差。`|delta V|` 与整体状态误差、速度误差均呈正相关，说明价值偏移主要来自 SD 推演末端状态与真实末端状态之间的动力学偏差，而不是 TVA 自身在真实状态上完全无信号。

由此形成的下一步判断为：TVA terminal prior 不应被取消，但当前 `h=5` 用法需要校准。更稳妥的方向是先将 online terminal prior 限制在 `h=1` 或 `h<=3`，并进一步考虑在 TVA 训练中加入 SD-rollout terminal state 的一致性校准，使 `V(SD terminal)` 显式对齐 `V(true terminal)`。该修订应作为单独可验证改动，不应与规则阈值、prior 权重或阶段规则同时调整。

### TVA terminal prior 接口门控实现

为避免 TVA 进入在线训练的时机依赖经验步数，闭环入口新增 `--enable-tva-terminal-prior-gate`。启用后，每个周期结束时会运行 `audit_hopper_tva_sd_interface.py`，并根据上一周期的接口审计结果决定下一周期是否允许 TVA terminal prior 进入 MPPI 候选排序。

代码改动如下：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
```

门控报告写入：

```text
cycle_xxx/reports/tva_sd_interface_audit_gate.json
cycle_xxx/reports/tva_terminal_prior_gate.json
```

下一周期的 `cycle_config.json` 记录如下字段：

```text
tva_terminal_prior_requested
tva_terminal_prior_gate_enabled
tva_terminal_prior_allowed_this_cycle
tva_terminal_prior_gate_report
```

第一版门控阈值为：

```text
horizons = 1,3
min_samples = 30
min_oracle_pearson = 0.55
min_sd_pearson = 0.55
min_sd_r2 = 0.20
max_sd_rmse_ratio = 0.90
max_abs_delta_v = disabled
```

该设计的研究意图是：TVA 只有在真实末端估值和 SD 推演末端估值均显示出基本可靠性时，才进入在线动作排序。若未通过门控，TVA 仍会继续训练、审计和生成 shadow 报告，但不会影响 TD-MPC2 的数据生成分布。

新增 gated 从头训练启动脚本为：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
```

该脚本沿用非门控 `rule+TVA terminal prior` 线的三种子、`5k` 总结周期、`25k` formal eval 和 `200k` 总预算，只新增接口门控参数。静态检查已通过：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/audit_hopper_tva_sd_interface.py
bash -n scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
```

此外，用 `--dry-run` 验证了门控流程：`cycle001` 在无上一周期审计报告时保持 `tva_terminal_prior_allowed=false`，周期总结后会生成门控审计命令和 `tva_terminal_prior_gate.json`，下一周期是否启用 TVA 将由该报告决定。

### gated TVA terminal prior 从头训练启动

已启动 gated TVA terminal prior 从头训练线：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
```

输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1
```

启动命令通过 `setsid` 后台运行，主日志为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1/launcher.log
```

启动检查显示，主控进程和 `cycle001` 的三种子训练进程均已存在。三个 seed 的 `cycle001` 均显示：

```text
rule_prior_enabled = false
tva_terminal_prior_enabled = false
```

该状态符合门控设计：第一周期没有上一周期的 SD/TVA checkpoint 和接口审计报告，因此 TVA terminal prior 不进入在线排序；`cycle001` 总结完成后会生成 `tva_terminal_prior_gate.json`，再决定 `cycle002` 是否允许启用 TVA terminal prior。

## 2026-05-18

### TVA terminal prior 强/弱分层门控修订

`hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 在 `75k` 前后的接口审计显示，TVA-SD 终端补偿尚未达到强门控要求，但已有若干周期出现方向性信号。回放已有 `tva_sd_interface_audit_gate.json` 后，采用弱门控阈值时，`cycle001`、`cycle005`、`cycle007`、`cycle016` 与 `cycle017` 可进入 `weak` 模式；其中 `cycle016` 和 `cycle017` 是当前接续训练最相关的信号。

为避免从“完全关闭 TVA”直接跳到“正常权重 TVA”，闭环入口已修订为强/弱分层门控：

```text
strong: weight=0.02, max_bonus=0.05
weak:   weight=0.005, max_bonus=0.01
off:    TVA 只训练和审计，不进入在线 planner
```

弱门控参数为：

```text
tva_terminal_weak_gate_horizon_mode = any
tva_terminal_weak_gate_min_samples = 30
tva_terminal_weak_gate_min_oracle_pearson = 0.50
tva_terminal_weak_gate_min_sd_pearson = 0.50
tva_terminal_weak_gate_min_sd_r2 = 0.00
tva_terminal_weak_gate_max_sd_rmse_ratio = 1.00
```

代码和脚本更新如下：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_weakgated_resume_cycle018_seed123_20260518_r1.sh
```

门控报告新增：

```text
strong_passed
weak_passed
decision.mode
decision.weight
decision.max_bonus
```

其中 `passed` 表示下一周期是否允许 TVA terminal prior 以强或弱模式上线。为支持从已有运行安全接续，新增 `--initial-tva-terminal-prior-policy`。接续脚本从 `cycle017` 的规则库、SD checkpoint、TVA checkpoint 和三种子 TD-MPC2 checkpoint 启动，并在 `cycle018` 初始采用 `weak` policy。

静态检查已通过：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/audit_hopper_tva_sd_interface.py
bash -n scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
bash -n scripts/launch_hopper_v8formal_200k_tva_terminal_prior_weakgated_resume_cycle018_seed123_20260518_r1.sh
```

修改前启动的旧主控在 `cycle018` 仍使用 `tva_terminal_prior_enabled=false`，因此已在该周期早段中止，并使用弱门控接续脚本从 `cycle017` checkpoint 重启。新主日志为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1/launcher_weakgated_resume_cycle018.log
```

重启检查显示 `cycle018/cycle_config.json` 记录：

```text
tva_terminal_prior_allowed_this_cycle = true
tva_terminal_prior_policy_this_cycle.mode = weak
tva_terminal_prior_policy_this_cycle.weight = 0.005
tva_terminal_prior_policy_this_cycle.max_bonus = 0.01
```

三个训练进程启动命令均显示：

```text
tva_terminal_prior_enabled = true
tva_terminal_prior_weight = 0.005
tva_terminal_prior_max_bonus = 0.01
```

## 2026-05-19

对 `hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 的 `175k` seed1 最高分轨迹进行了视频渲染与抽帧复核。轨迹 `episode_03` 的回报为 `491.6247`，渲染文件为：

```text
results/hopper_trace_renders/v8formal_tva_terminal_prior_seed1_175k_episode03_R491p62.mp4
```

抽帧图保存为：

```text
results/hopper_trace_renders/frames_seed1_175k_ep03_R491p62/contact_1fps.png
results/hopper_trace_renders/frames_seed1_175k_ep03_R491p62/critical_frames_labeled.png
```

视觉复核显示，该轨迹已经呈现较标准的 Hopper hop 循环，主要不足更接近蓄力和发力幅度仍可提高，而不是趴地前行。此前自动审计中 `grounded_prone_frac=0.614` 与视频判断不一致。对照代码后确认，旧 `grounded_prone` 标签在接地且未满足严格 `upright/prepared` 时，以 `nose_minus_anchor_x >= 0` 作为主要兜底几何条件；正常 hop 的屈膝前倾支撑也容易满足该条件，导致高质量支撑帧被误记为 prone。

据此校正底层状态标签：新增 `grounded_hop`，用于表示无身体碰撞、具备前向速度、躯干高度不低且前倾角仍在正常 hop 范围内的接地推进状态；`grounded_prone` 被收缩为不满足 `grounded_hop/upright/prepared/rear` 的前向接地失败或恢复状态。阈值为：

```text
STATE_HOP_HEIGHT_MIN = 0.55
STATE_HOP_PITCH_ABS_MAX = 1.10
STATE_HOP_VX_MIN = 0.75
```

更新文件包括：

```text
src/world_model/dmcontrol_hopper_sd_model.py
external/tdmpc2/tdmpc2/trainer/online_trainer.py
scripts/export_hopper_eval_traces.py
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py
experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py
```

静态检查已通过：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  src/world_model/dmcontrol_hopper_sd_model.py \
  external/tdmpc2/tdmpc2/trainer/online_trainer.py \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py \
  experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py \
  scripts/export_hopper_eval_traces.py
```

用校正后的结构模型重算 `491.6247` 轨迹的状态占比，结果为：

```text
grounded_prone = 0.046
grounded_hop = 0.568
grounded_upright = 0.018
grounded_prepared = 0.002
falling_air = 0.322
rising_air = 0.040
```

旧标签下同一轨迹的 `grounded_prone` 占比为 `0.614`。因此，本次校正确认将主要误判部分从 `grounded_prone` 转移至 `grounded_hop`。对 seed2、seed3 的低分失败轨迹复核显示，低分轨迹仍主要标记为 `grounded_rear`，`grounded_hop=0.0`，未观察到明显把失败轨迹误标为 hop 的情况。

同步修订 `experiments/hopper_reward_regression/reward_regression_suite.py` 中与状态编号相关的回归用例：原 `fake_rear_now_prone` 调整为 `front_lean_lift_now_hop`，原 `prone_to_upright_positive_reward` 调整为 `hop_to_upright_positive_reward`。回归测试通过：

```text
PYTHONPATH=/home/data/round/code/sd_world_model_rl \
  /home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/hopper_reward_regression/test_reward_regression.py

Ran 2 tests in 0.151s
OK
```

同日，对 `hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 的最终收尾状态进行了检查。cycle040 的三种子训练与 `200k` 正式评估均已完成并落盘，但闭环控制脚本在最终 replay trace 导出阶段中断。中断原因不是训练或评估失败，而是 `experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py` 在新增 `grounded_hop` 后，行级 CSV 的 `FIELDNAMES` 和 summary CSV 的 `SUMMARY_FIELDS` 未同步加入 `grounded_hop` 与 `grounded_hop_frac`，导致 `csv.DictWriter` 检测到额外字段并报错。

已修复该字段一致性问题：

```text
experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py
```

修复后通过静态检查：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py
```

并用 `200k` seed1 checkpoint 做最小导出验证，成功写出 `400` 条 replay episode。随后补齐 cycle040 三个 seed 的实际 replay 导出：

```text
logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1_cycle040_target200039/retrospective_train_traces/step_0200040/eval_episode_summary.csv
logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1_cycle040_target200039/retrospective_train_traces/step_0200040/eval_episode_summary.csv
logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1_cycle040_target200039/retrospective_train_traces/step_0200040/eval_episode_summary.csv
```

每个 seed 均导出 `400` 条 replay episode。该修复只影响最终后处理导出字段，不改变已经完成的训练、checkpoint 或正式评估结果。

基于上述修复，启动修订状态标签后的从零重训正式线，目标为 `200k`，三种子共享规则库，协议沿用上一条 gated TVA terminal prior 线：每 `5k` 执行 SD 拟合、失败归因、规则更新和 TVA 评估，每 `25k` 保存 checkpoint 并做 `10 episode` 正式评估。新实验名为：

```text
tdmpc2_v8formal_200k_tva_terminal_prior_gated_hoplabel_fromscratch_seed123_20260519_r2
```

输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hoplabel_fromscratch_seed123_20260519_r2
```

启动脚本为：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_hoplabel_fromscratch_seed123_20260519_r2.sh
```

`r1` 名称曾用于短前台诊断启动行为，并被 `timeout` 主动中止；为避免残留 cycle001 文件与正式结果混用，正式运行改用 `r2` 名称。启动检查显示，控制进程与 seed1/2/3 三个 cycle001 训练进程均已运行；cycle001 中 `tva_terminal_prior_policy={'allowed': False, 'mode': 'off', 'weight': 0.0, 'max_bonus': 0.0}`，规则库为 `shared_rules/library_cycle_000_empty.json`，符合从零训练设定。

## 2026-05-19

对状态标签修订后的从头实验 `tdmpc2_v8formal_200k_tva_terminal_prior_gated_hoplabel_fromscratch_seed123_20260519_r2` 进行了早期退化诊断。该线运行至约 `90k` 前被主动中止，已有结果保留用于事后分析。中止原因不是程序错误，而是 `75k` 前性能明显弱于上一条高分线，且审计显示问题更可能来自规则机制输入分布变化，而非 TVA terminal prior。

检查结果显示，`25k/50k/75k` formal eval 中 seed1 分别约为 `0.000/0.001/0.851`，显著低于上一条 `tdmpc2_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 的同口径 seed1 结果。与此同时，TVA terminal prior 在早期周期未通过门控，训练命令中 `tva_terminal_prior_policy` 处于 `off`；因此，早期退化不能解释为 TVA 直接进入 planner 后造成的性能下降。

规则审计显示，第 `15` 周期新线的规则 selected-gate 命中率约为 `0.494`，高于旧线约 `0.365`，但 transition rule 命中与回报相关性由旧线的正相关转为负相关。规则内容也从旧线较多 `waist_pos` 相关的 rear/falling 恢复规则，转向更混合的 `knee_pos/hip_neg/waist_neg/prone/falling/rear` 规则。该结果说明规则并非未生效，而是更宽的规则覆盖没有转化为有利行为。

进一步检查发现，新线早期 replay summary 中 `grounded_hop_frac` 基本为 `0`，因此问题不是大量早期失败轨迹被直接导出为 `grounded_hop`。更可疑的机制是，上一轮状态标签修订将 `grounded_hop` 加入 SD 内部 `state_confident` 与 `_dominant_state_ids`，并在 `grounded_prone` 之前排他占位。该设计虽然修正了高分 hop 轨迹被误判为 prone 的审计问题，但可能在早期低质量轨迹的 SD 反事实评分和规则维护中改变恢复状态语义。

据此执行最小修订：`grounded_hop` 保留为诊断性姿态特征，但不再进入核心排他状态划分。具体代码修订为：从 `state_confident` 中移除 `grounded_hop`，不再用 `(1 - grounded_hop)` 从 `grounded_prone` 中扣除，并从 `_dominant_state_ids` 中移除 hop 状态编号。该修订保留高分 hop 轨迹报告能力，同时减少对早期恢复型规则主干的扰动。

新增 `100k` 从头验证脚本：

```text
scripts/launch_hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1.sh
```

实验名为：

```text
tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1
```

输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1
```

该线仍使用三种子共享规则、`5k` 总结周期、`25k` checkpoint 与 `10 episode` formal eval、gated TVA terminal prior、TVA shadow/consequence 审计。主要观察指标为：第 `25k/50k/75k/100k` formal eval、规则命中-回报相关性、transition rule 相关性、规则内容是否恢复为更明确的恢复型规则，以及 seed1 是否重新出现早期爬升趋势。

完成编译检查：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  src/world_model/dmcontrol_hopper_sd_model.py \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

随后启动 `100k` 复核线。启动检查显示，控制进程与 seed1/2/3 三个 cycle001 训练进程均已运行；cycle001 使用空规则库，`rule_prior_enabled=false`，且 `tva_terminal_prior_policy={'allowed': False, 'mode': 'off', 'weight': 0.0, 'max_bonus': 0.0}`，符合从零训练设定。

## 2026-05-20

`tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1` 已完成 `100k`。正式评估显示，三种子在 `100k` 均出现非零信号，其中 seed1 mean `46.330`、max `98.523`，seed2 mean `51.921`、max `133.595`，seed3 mean `4.352`、max `19.740`。规则审计在后期持续转健康，`cycle020` 的 `gate_hit_reward_corr=0.322`，`transition_gate_hit_reward_corr=0.845`。TVA terminal prior 在 `cycle018` 之后开始通过门控，其中 `cycle020` 门控报告对下一周期给出 `strong` 决策。

基于用户要求，启动严格接续至 `200k` 的新线。该线不从头重训，而是使用 `100k` 终点产物作为初始输入：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1/cycle_020/rules/library_closed_loop_cycle_020_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

initial_tva_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1/cycle_020/tva/hopper_tva_cycle_020.pt

initial_resumes =
1:logs/hopper-hop/1/tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1_cycle020_target100019/models/final.full.pt
2:logs/hopper-hop/2/tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1_cycle020_target100019/models/final.full.pt
3:logs/hopper-hop/3/tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1_cycle020_target100019/models/final.full.pt
```

由于 `cycle020` 的 TVA terminal prior gate 对下一周期给出 `strong` 决策，接续线显式设置：

```text
initial_tva_terminal_prior_policy = strong
```

接续实验名为：

```text
tdmpc2_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1
```

输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1
```

启动脚本为：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1.sh
```

该线从 `cycle021` 运行至 `cycle040`，总步数从 `100k` 接续到 `200k`。协议保持不变：每 `5k` 执行 SD 拟合、归因、规则更新、TVA 训练与门控审计，每 `25k` 保存 checkpoint 并执行 `10 episode` formal eval。

启动前完成脚本语法与入口编译检查：

```text
bash -n scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1.sh
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

随后启动接续线。启动检查显示，控制进程和 seed1/2/3 三个 `cycle021` 训练进程均已运行；三个 seed 均从 `100k` 的 `final.full.pt` 恢复，规则库为 `cycle020` confidence library，`tva_terminal_prior_policy={'allowed': True, 'mode': 'strong', 'weight': 0.02, 'max_bonus': 0.05}`，符合严格接续设定。

`tdmpc2_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1` 已完成 `200k`。正式评估结果为：seed1 mean `121.579`、max `220.344`，seed2 mean `143.229`、max `252.527`，seed3 mean `86.607`、max `225.937`，三种子均值 `117.138`。同口径官方 TD-MPC2 原始记录 `external/tdmpc2/results/tdmpc2/hopper-hop.csv` 中 `200k` 为 seed1 `138.4`、seed2 `215.8`、seed3 `145.7`，三种子均值 `166.633`。因此，该接续线在 `200k` 未超过官方 TD-MPC2，且低于本线 `175k` 均值 `120.906`。

随后对 `200k` official eval 的三种子 `30` 条轨迹进行了独立姿态与规则命中审计，输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/final200k_official_posture_audit.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/final200k_official_rule_effect_audit.json
```

姿态审计显示，最终 official eval 的 `near_good_frac=0.206`、`structural_success_frac=0.058`、`toe_only_launch_frac=0.124`、`mean_full_foot_support=0.0038`、`phase_prone_frac=0.565`、`phase_rear_frac=0.347`、`phase_hop_frac=0.000`。高分轨迹同样主要处于 `prone` 与脚尖发力模式：`reward >= 180` 的 `16` 条轨迹平均 `phase_prone_frac=0.853`、`toe_only_launch_frac=0.226`、`mean_full_foot_support=0.0027`，说明当前高分来源仍不是稳定全足支撑 hop。

规则审计显示，official eval 中 active/observing 规则 selected-gate 命中率为 `0.679`，episode 级规则命中密度与回报相关性为 `0.853`，transition 命中密度与回报相关性为 `0.620`。这与训练 replay 审计中的总体相关性较弱形成差异，说明最终评估中规则确实更密集地作用于高分轨迹；但这些高分轨迹本身仍是 prone/toe-launch 模式。因此，本轮结果不能解释为规则未生效，而应解释为：规则有效地强化或伴随了当前局部高回报模式，但该模式不是目标中的标准 hop。

进一步检查最终规则库发现，多处同一状态上下文仍保留多个动作族规则，部分存在相反方向动作。例如 `unsupported_high_torque|omega_pos|vx_high|support_none|prone` 同时包含 `waist_pos/waist_neg`、`knee_pos/knee_neg` 等规则；其中部分规则 confidence 已降为 observing，但仍参与本次 official eval 审计。该结果提示，当前 confidence 维护虽能退休明显不良规则，但尚未充分压缩同一上下文下的互斥动作族，导致规则库更像“局部动作偏置集合”，而不是阶段目标明确的教师型先验。

由此形成当前判断：工程闭环已经打通，规则在最终评估中可验证地生效；失败点不在规则没有进入 planner，而在规则价值标准仍过多继承了当前策略分布中的局部高回报模式。下一步改进应聚焦于规则价值定义，而不是继续增加 Hopper 特定硬编码。可验证方向包括：将规则上线标准从单步动作一致性进一步提升到阶段后果解释力；引入对 `full_support_drive`、稳定 hop 循环和持续平衡的后果约束；并在同一状态上下文内加入互斥动作族归并或 winner-take-most 机制，避免同一上下文保留多个相反动作偏置。

### TVA 上下文动作竞争审计与影子规则库

为避免将 Hopper 姿态目标硬编码进规则价值标准，新增只读审计 [audit_tva_context_action_competition.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_context_action_competition.py)。该审计以规则状态条件作为上下文，在同一上下文内汇总 SD rollout + TVA terminal value 对动作族的排序，从而判断规则动作族是否在本上下文中得到 TVA 支持。该审计不把 `full_foot_support`、标准 hop 姿态或其他人工目标作为筛选标准；这些结构态只作为规则上下文和事后解释字段。

在 `cycle040` 最终规则库上运行该审计，输入为：

```text
library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json

tva_counterfactual_audit =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_counterfactual_audit.json
```

输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_context_action_competition_audit.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_context_action_competition_audit.md
```

审计结果显示：共读取 active/observing 规则 `95` 条，报告上下文 `17` 个，匹配上下文记录 `1547` 条；其中 `7` 个上下文存在同关节相反动作族冲突，`21` 条规则被标记为 `tva_supported`，`18` 条规则被标记为 `tva_suspect`。典型例子是 `unsupported_high_torque|omega_neg|vx_low|support_none|falling`：TVA 排序支持 `ankle_neg`、`knee_pos`、`waist_neg`，并将同上下文中的 `ankle_pos`、`waist_pos` 标记为可疑。该结果说明 TVA 上下文竞争能够在不引入 Hopper 姿态硬编码的条件下，对同状态冲突动作族给出可解释区分。

随后新增 [build_tva_context_competition_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_competition_shadow_library.py)，将上述审计结果转化为只读影子规则库。该脚本不修改原规则库，只生成 eval/shadow 副本。默认策略为：`tva_supported` 小幅增强，`tva_suspect` 按 `suspect_decay=0.50` 削弱并可降为 observing，冲突但证据不足的规则保守软衰减。对 `cycle040` 运行后输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_competition_shadow.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_competition_shadow.md
```

影子库统计为：confidence 发生变化的规则 `34` 条；原 active 规则由 `61` 条降至 `51` 条；`11` 条原 active 被降为 observing，`1` 条 observing 被提升为 active。该结果表明，TVA 上下文竞争确实能压缩部分同上下文冲突规则，但其是否适合直接替换在线规则库仍需 eval-only 验证。

为此新增 [eval_hopper_tva_context_competition_shadow_200k.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py)，在同一 `200k` 三种子 checkpoint 上比较三种 eval-only 模式：`rule_off`、原 `cycle040 confidence` 规则库、TVA 上下文竞争影子规则库。每个 seed 每种模式执行 `10 episode`，不训练、不更新 replay、不修改规则库。汇总输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_competition_shadow_eval_20260520.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_competition_shadow_eval_20260520.md
```

eval-only 结果为：

```text
rule_off:
  seed means = 120.908 / 123.477 / 131.834
  mean = 125.407

original:
  seed means = 104.559 / 125.235 / 153.804
  mean = 127.866

tva_context_shadow:
  seed means = 103.536 / 124.591 / 133.257
  mean = 120.461
```

因此，影子库在直接替换 active 规则库的 eval-only 口径下没有带来即时收益，且均值低于原规则库和 rule-off。进一步姿态审计显示，三种模式仍均为局部 prone/toe-launch 行为：`phase_hop_frac=0.0`；`toe_only_launch_frac` 分别为 `0.1351 / 0.1378 / 0.1298`；`mean_full_foot_support` 分别为 `0.0372 / 0.0098 / 0.0141`。TVA 影子库略微降低了 toe-launch 和 prone 占比，但也降低了 `near_good_frac` 与回报。

行为 rule-effect 审计进一步解释了该结果。原规则库与 TVA 影子库的 selected-gate 命中率分别为 `0.6159` 与 `0.6092`，总体规则命中强度没有显著下降；加权有害方向比例分别约为 `0.3011` 与 `0.3016`，也没有改善。被 TVA 影子库降权的 `11` 条原 active 规则中，多条在 eval 中具有较高方向一致率，例如 `rotation_risk|omega_neg|vx_low|support_none|rear->prefer:waist_pos` 的 selected-gate hits 为 `459`，方向一致率约 `0.996`；`rotation_risk|omega_neg|vx_low|support_none|rear->prefer:hip_neg` 的 hits 为 `459`，方向一致率约 `0.989`。这些规则主要作用于低分 rear 恢复状态，虽然不对应高回报结构，但可能承担基础恢复功能。TVA 上下文竞争将其降权，说明单纯按终端价值竞争可能误伤“低回报但必要的恢复型规则”。

本轮结论为：TVA 上下文动作竞争是有价值的规则审计信号，尤其适合识别同一上下文中的动作族冲突；但当前版本不适合直接替换 live confidence update 或 active 规则库。下一步应将 TVA 上下文竞争作为附加证据，与真实 rule-effect、一致方向统计、短窗口后果和规则角色区分联合使用。特别是应区分“恢复型规则”和“高回报推进型规则”：前者不能仅因终端 TVA 价值不高而被降权，后者才更适合采用 winner-take-most 的上下文动作族竞争。

基于上述负结果，继续实现保守版 [build_tva_context_role_gated_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_role_gated_shadow_library.py)。该脚本仍读取 TVA 上下文竞争审计，但额外读取真实 eval rule-effect 审计。其核心变化是：TVA suspicion 不能单独降权行为上一致的恢复规则；只有当 TVA 可疑信号同时伴随真实动作方向有害、方向一致性不足或有效证据不足时，才允许削弱。规则角色不由 Hopper 特定姿态硬编码决定，而由 rule-effect 统计划分：高一致但低 good/high-episode 命中的规则标记为 `recovery_consistent` 并受保护，高 good/high-episode 上下文规则标记为 `progress_context`。

在同一 `cycle040` 上运行 role/effect-gated 影子更新后，输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_shadow.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_shadow.md
```

该版本仅改变 `7` 条规则的 confidence，active 数保持 `61` 条不变。role 统计为：`progress_context=35`、`recovery_consistent=10`、`neutral=3`、`low_evidence=13`、`unknown=34`。其中第一版 TVA 影子库误降权的若干 rear 恢复规则被标记为 `tva_suspect_but_recovery_consistent_protected` 或 `tva_suspect_but_effect_consistent`，不再被降为 observing。

随后对 role-gated 影子库执行同一 `200k` checkpoint eval-only。结果为：

```text
rule_off mean              = 125.407
original mean              = 127.866
tva_context_shadow mean    = 120.461
tva_context_role_gated mean = 126.738
```

姿态审计显示，`tva_context_role_gated` 的 `near_good_frac=0.2267`、`toe_only_launch_frac=0.1363`，基本接近原规则库的 `0.2288` 与 `0.1378`，明显好于第一版影子库的 `near_good_frac=0.2128`。rule-effect 审计显示，role-gated 版本 selected-gate 命中率为 `0.6200`，加权方向一致率约 `0.7010`，加权有害方向比例约 `0.2990`，略好于原规则库的 `0.6989 / 0.3011`。

该结果说明，TVA 上下文竞争作为附加价值信号是可用的，但必须经过真实行为证据和规则角色门控。当前 role-gated 版本只恢复到原规则库附近，尚未形成明确提升；因此仍不应直接进入正式重训。下一步应将该机制继续保持 shadow-only，并进一步加入真实短窗口后果验证或跨周期 replay 验证，确认它能稳定筛除有害 progress 规则，而不削弱恢复型规则。

## 2026-05-21

### TVA 上下文竞争的短窗口后果门控评估

为检验 role/effect-gated 影子库中的 TVA 改动是否有真实轨迹短窗口后果支持，本日对 `cycle040` 的 role-gated shadow 执行通用规则后果审计，并据此生成 consequence-gated shadow 规则库。相关输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_short_window_consequence_20260521.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_short_window_consequence_20260521.md

results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_consequence_shadow.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_consequence_shadow.md
```

短窗口后果审计显示，role-gated 影子库中共有 `7` 条规则发生 confidence 变化，其中只有 `2` 条通过后果支持检查，支持率为 `2/7 = 0.286`。通过者为：

```text
rotation_risk|omega_neg|vx_low|support_none|falling->prefer:ankle_neg
unsupported_high_torque|omega_neg|vx_mid|support_none|falling->prefer:hip_pos
```

其余 `5` 条因真实短窗口后果中相反方向更优或后果证据不足而被回退至原始 confidence。生成的 consequence-gated shadow 规则库因此保留 active 规则数 `61`，但仅实际保留上述 `2` 条 TVA 增强，其余 role-gated 改动被撤销。

随后在同一 `200k` 三种子 checkpoint 上执行 `10 episode` eval-only，输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_shadow_eval_20260521.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_shadow_eval_20260521.md
```

结果如下：

```text
tva_context_role_consequence:
  seed means = 104.311 / 137.204 / 130.803
  mean = 124.106
```

与此前同口径对照相比，`rule_off` 均值为 `125.407`，原规则库均值为 `127.866`，role-gated shadow 均值为 `126.738`。因此，consequence-gated shadow 没有带来即时收益，且略低于 rule-off 与原规则库。

为进一步确认该结果不是由规则未触发造成，补充执行了规则命中与姿态行为审计：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_rule_effects_20260521.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_rule_effects_20260521.md

results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_posture_comparison_20260521.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_posture_comparison_20260521.md
```

规则审计显示，consequence-gated shadow 的 selected-gate 命中比例为 `0.6203`，与原规则库 `0.6159` 和 role-gated shadow `0.6200` 基本相同；episode 级规则命中密度与回报相关性为 `0.7081`。因此，该负结果不能解释为规则 prior 没有作用，而应解释为保留下来的 TVA 改动没有产生更优动作排序。

姿态比较也支持这一判断。四种模式的核心指标为：

```text
                reward_mean  near_good  structural  toe_launch  full_support_drive
original        127.866      0.2288     0.0682      0.1378      0.0001
rule_off        125.407      0.2237     0.0695      0.1351      0.0008
role_gated      126.738      0.2267     0.0691      0.1363      0.0004
role_conseq     124.106      0.2203     0.0655      0.1341      0.0002
```

consequence-gated shadow 略微降低了脚尖弹射比例，但同时降低了 `near_good_frac`、`structural_success_frac` 与回报，没有表现为更接近标准 hop 的行为改进。被保留的两条增强规则在 eval 中确实可触发，但影响有限：`falling -> ankle_neg` 的方向一致率从原规则库 `0.627` 降至 `0.573`，`falling -> hip_pos` 的方向一致率从 `0.739` 升至 `0.811`；两者对整体姿态和回报均未形成可见增益。

当前结论为：TVA 上下文动作竞争、规则角色门控和短窗口后果门控构成了可审计的规则价值筛查链，但该链在 `cycle040 / 200k` 产物上尚未证明可提升行为质量。其失败点不是命中不足，而是保留候选的行为杠杆过小，且短窗口后果支持不等价于长期姿态改进。因此，该机制继续保持 shadow-only，不进入正式在线规则库或重训练。下一步若继续推进，应优先改进“规则价值的跨周期后果验证”和“同一上下文下动作族冲突的稳定归并”，而不是简单放大 TVA 改动权重。

## 2026-05-22

### 教师核心规则初始化分支

基于 `300k` 规则审计暴露出的同一状态动作族冲突问题，新增高质量教师轨迹核心规则提取工具 [build_hopper_teacher_core_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_teacher_core_rules.py)。该工具从高回报 official eval 轨迹中提取少量方向稳定、跨 episode 支持充分的动作族偏好，输出可被现有 `online_rule_prior_v6` 直接读取的规则库。为保持实现兼容，规则的 `source` 字段仍写为 `v4_substate`；为保持实验口径清晰，额外写入 `rule_origin=teacher_core_standard_hop`、`teacher_source=hopper_high_quality_official_eval` 和教师轨迹证据字段。

教师轨迹来源为此前已渲染并人工复核过的标准 Hop 线：

```text
logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1_cycle040_target200039/official_eval/step_0200039/episode_csv
```

该目录包含 `10` 条 `495.23` 至 `530.25` 分的 official eval 轨迹，均值为 `514.55`。姿态审计输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_high_quality_teacher_175k_200k_posture_audit.json
results/sd_retrospective_rule_induction/reports/hopper_high_quality_teacher_175k_200k_posture_audit.md
```

其中 `200k seed1` 教师组的关键指标为：`mean_reward=514.55`、`phase_hop_frac=0.4728`、`body_collision_mean=0.0188`、`toe_only_launch_frac=0.1262`。同时保留此前 `175k seed1 episode03 R491.62` 的视频与抽帧：

```text
results/hopper_trace_renders/v8formal_tva_terminal_prior_seed1_175k_episode03_R491p62.mp4
results/hopper_trace_renders/frames_seed1_175k_ep03_R491p62/critical_frames_labeled.png
results/hopper_trace_renders/frames_seed1_175k_ep03_R491p62/contact_1fps.png
```

已生成两个教师核心规则库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_hop_only_confidence.json
```

宽核心库包含 `24` 条规则，覆盖 `12` 个子状态。筛选条件为：教师 episode 回报不低于 `490`，行级 `vx>=0.50`、`body_collision<=0.20`、`abs(pitch)<=1.10`，动作阈值 `0.25`，跨 episode 支持不少于 `6`，方向一致比例不低于 `0.72`，相反方向比例不高于 `0.28`。在教师轨迹上的审计结果为：规则 selected-gate 命中率 `0.6190`，高分 episode 的规则命中密度约 `0.98`，方向一致比例约 `0.97`。该结果说明宽核心库能较好复现教师动作结构。

保守 `hop_only` 核心库只保留排他 `phase=hop` 帧，包含 `12` 条规则。该库在教师轨迹上的 selected-gate 命中率为 `0.4596`，但在当前 `300k` eval 轨迹上几乎不触发。这一现象说明当前 `300k` 轨迹虽然存在 raw `grounded_hop` 片段，但排他 phase 仍多被判为 `prone`，因此 `hop_only` 库更适合作为高可信标准姿态锚点，而宽核心库更适合作为早期可触发的教师初始化。

当前结论为：使用标准 Hop 高分轨迹提取核心规则不属于作弊，但必须明确标记为“教师核心规则初始化”，并与纯在线自学习规则库分开比较。该分支体现了结构化显式世界模型的优势：高质量经验可以被转写为可审计、可修改、可迁移的规则，而不是仅被隐式吸收到策略网络参数中。下一步实验应以该教师核心库作为独立初始规则库，检验其是否提升从头训练中进入标准 Hop 姿态的效率；同时继续修订同上下文动作族竞争和信心削弱机制，避免教师宽规则在非教师姿态分布上产生持续冲突。

### 教师核心初始化 200k 从头实验启动

新增启动脚本 [launch_hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1.sh](/home/data/round/code/sd_world_model_rl/scripts/launch_hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1.sh)，以宽核心教师规则库作为初始规则库启动独立闭环实验：

```text
exp_prefix: tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1
output_root: results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1
initial_rule_library: results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
seeds: 1,2,3
cycles: 40
cycle_steps: 5000
formal_eval_freq: 25000
formal_eval_episodes: 10
```

本实验从零训练 TD-MPC2，不接续任何策略 checkpoint；通过 `--enable-rule-prior-cycle1` 使教师核心规则从第一个训练周期即参与在线动作排序。TVA terminal prior 仍遵循现有门控流程：初始无 SD/TVA checkpoint，因此第一个周期不启用 TVA terminal prior，后续由周期性 SD 拟合、TVA 训练和 TVA-SD gate 决定是否上线。实验已启动，当前处于 `cycle_001 / target_steps=5000` 的三种子训练阶段。第一轮 formal eval 将在 `25k` 保存点执行。

### 跨周期后果汇总与同上下文动作族归并工具

基于上述结论，本日新增两个只读/影子工具，用于把下一步重点从“单周期 TVA 改动”推进到“跨周期稳定性”和“同一上下文动作族冲突归并”。

第一，新增 [summarize_rule_shadow_cross_cycle_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/summarize_rule_shadow_cross_cycle_consequences.py)。该脚本读取一个或多个 `audit_rule_shadow_short_window_consequences.py` 产物，按规则 `id` 聚合后果支持次数、比较次数、平均 outcome margin 和支持率。默认只有在至少 `2` 个 audit 中可比较、至少 `2` 个 audit 支持、支持率不低于 `0.67` 且平均 margin 非负时，才标记为 `stable_consequence_supported`。该设计用于避免单周期偶然轨迹片段直接触发 live update。

对当前 `cycle040` 单一后果审计执行 smoke，输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_cross_cycle_consequence_smoke_20260521.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_cross_cycle_consequence_smoke_20260521.md
```

结果为：输入 audits=`1`，rules seen=`7`，stable supported rules=`0`。其中两条规则在单周期内为正向支持，但由于没有第二个周期验证，均未被标记为稳定支持。这符合预期，也说明当前数据还不能支撑将这些 TVA 改动写入正式规则库。

第二，新增 [build_tva_context_consolidated_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_consolidated_shadow_library.py)。该脚本读取原规则库、TVA context action competition audit、真实 rule-effect audit，并可选读取 short-window consequence audit，在同一 `context_key + joint` 下比较相反动作族。综合分数由 TVA 上下文竞争、真实方向一致性和短窗口后果三部分构成；只有当赢家相对次优动作族的分数差超过阈值、赢家具备足够真实 effect hits 和 TVA context records，且 loser 不是受保护的 recovery-consistent 规则时，才在影子库中削弱 loser。默认不增强 winner，以避免从一个极端转向另一个极端。

在 `cycle040` 上运行后输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_consolidated_shadow.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_consolidated_shadow.md
```

smoke 结果显示：共发现 `13` 个同上下文同关节相反动作族候选组，其中只有 `1` 个组达到归并资格，实际改变 `1` 条规则 confidence，active 数保持 `61` 不变。唯一被削弱规则为：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|prone->prefer:ankle_pos
confidence: 0.481741 -> 0.240870
```

同组赢家为：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|prone->prefer:ankle_neg
```

该赢家在 TVA 上下文竞争中得到强支持，且在原 eval rule-effect 中有 `621` 次 selected-gate hits；loser 无对应 effect hits，并被 TVA 标记为 suspect。由于 loser 原本已经是 `confidence_observing`，本次归并不会改变 active 规则数量，只会进一步降低其未来重新上线概率。该结果表明，当前保守归并机制可以识别明确冲突，但不会大规模改写规则库。

同时更新 [eval_hopper_tva_context_competition_shadow_200k.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py)，新增 `tva_context_consolidated` eval-only 模式，并通过 dry-run 确认命令会加载上述 consolidated shadow 规则库。鉴于当前影子库只改变一条 observing 规则，暂不启动完整 `10ep x 3 seeds` eval；该 eval 模式保留给后续当 consolidation 产生 active 规则变动或跨周期稳定支持候选时使用。

## 2026-05-23

### 能力入口追踪离线审计

基于教师核心初始化实验中 seed1 与 seed2、seed3 的早期分叉现象，新增离线工具 [trace_hopper_ability_entry_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/trace_hopper_ability_entry_segments.py)。该工具从 retrospective train traces 中自动识别每个 seed 首次跨越 `R>1`、`R>10`、`R>50`、`R>100` 的周期，并回到具体 episode 的局部窗口，与同期未跨入口 seed 的高分失败 episode 进行对照。该工具不修改规则库、不接入在线训练，目的在于追踪高质量能力形成过程，为后续阶段入口规则提供可审计候选。

已对如下实验线运行第一版审计：

```text
tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1
```

输出位置为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_trace_report.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_trace_report.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_actions.json
```

第一版结果确认，seed1 的能力入口明显早于 seed2、seed3：`R>1` 首次出现在 `cycle004 / 20k`，`R>10` 首次出现在 `cycle007 / 35k`，`R>50` 与 `R>100` 首次出现在 `cycle012 / 60k`。seed2 分别在 `cycle017 / 85k`、`cycle018 / 90k`、`cycle025 / 125k` 才跨过前三个阈值，且未在前 `25` 个周期跨过 `R>100`；seed3 分别在 `cycle008 / 40k`、`cycle021 / 105k`、`cycle023 / 115k` 跨过前三个阈值，亦未跨过 `R>100`。

窗口对照显示，seed1 在 `cycle012` 的 `R>50 / R>100` 入口窗口中，`knee_neg` 和 `ankle_neg` 等动作族在成功窗口中的占比高于同期失败窗口，并伴随更高的后续窗口奖励和前向位移。该结果目前仅构成观察性候选，不足以直接上线为规则；后续应将这些入口候选送入 SD/TVA 反事实推演，并进行跨周期后果验证。

### 能力入口候选的 SD/TVA 反事实审计

新增 [audit_hopper_ability_entry_counterfactuals.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_counterfactuals.py)，将能力入口窗口送入同周期 SD explainer 与 TVA value net 进行短视界反事实审计。该脚本在每个窗口 anchor 上比较真实动作、零动作和四个关节正负单关节探针，使用 `5` 步 SD rollout 后的结构态 desirability 与 TVA 终端价值分别排序动作族。该步骤仍为只读审计，不修改规则库或训练流程。

已完成 smoke 与全量审计。smoke 使用 `80` 个窗口，未出现缺失 checkpoint 或 trace；全量审计使用 `568` 个窗口，跳过数为 `0`。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit.md
```

全量聚合结果显示，入口成功窗口中，观察性目标动作族进入 SD top-3 的比例为 `0.260`，进入 TVA top-3 的比例为 `0.298`；成为 SD 最优动作族的比例为 `0.099`，成为 TVA 最优动作族的比例为 `0.061`。同期失败窗口中，SD top-3 比例反而为 `0.474`，TVA top-3 比例为 `0.294`。因此，单纯将入口窗口中出现频率较高的动作族解释为“可上线规则”并不充分。

更具体地看，若只看 seed1 在 `cycle012 / 60k` 的 `R>50 / R>100` 入口，先前观察性候选 `knee_neg` 虽然具有较高后续奖励，但其 SD top-3 支持率仅约 `0.167`，TVA top-3 支持率仅约 `0.083`；`ankle_neg` 的 SD/TVA top-3 支持率也只处于较低水平。相反，较晚出现的 seed2 `cycle025 / R>50 / hip_pos` 得到较强支持，SD top-3 为 `0.750`，TVA top-3 为 `0.625`。这说明当前入口发现机制已经能够定位“能力进入 replay 的时间点”，但从入口窗口到规则的转换仍不能采用单动作族频率压缩，可能需要显式建模动作序列、阶段上下文或窗口级后果。

当前结论为：能力入口追踪是有效的离线诊断层，但第一版候选还不能直接转为在线规则。下一步应围绕“入口动作序列 / 阶段上下文”的候选表达进行设计，并继续保持 SD/TVA 反事实审计作为上线前筛查。

### 效果空间规则原型

为保持规则维护机制简洁，并避免重新引入较难控制的动作序列式规则，新增 [audit_hopper_ability_entry_effect_space.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_effect_space.py)。该脚本将规则候选表达为 `context_key -> prefer_effect_direction`，其中 `effect_direction` 由入口成功窗口与同期失败窗口在低维效果空间中的短视界后果差异给出：

```text
v_entry = mean(delta_z | entry_success_actual)
        - mean(delta_z | matched_failure_actual)
```

效果向量 `z` 由同周期 SD rollout 和 TVA value net 构造，包含 TVA 值、结构态 desirability、速度、支撑、姿态和旋转控制等分量。候选动作通过 `dot(delta_z_candidate, normalize(v_entry))` 排序，因此在线语义仍可保持为轻量排序先验，而不是显式动作序列控制。

已完成 smoke 与全量审计。smoke 使用平衡抽样的 `80` 个窗口，得到 `8` 个方向，跳过数为 `0`。全量审计使用 `568` 个窗口，跳过数为 `0`，输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit.md
```

结果显示，效果空间方向比单动作族候选具有更清晰的区分信号。全局 `AUC(success alignment > failure alignment)` 为 `0.620`，alignment margin 为 `3.3387`。早期入口中，`R>1 / cycle008` 的 AUC 为 `0.685`，`R>1 / cycle004` 为 `0.656`，`R>50 / cycle012` 与 `R>100 / cycle012` 均为 `0.632`。这些数值尚不足以直接上线，但表明“把阶段入口过程映射为效果方向”比“保存单一动作族”更符合当前数据结构。

当前结论为：效果空间规则是值得继续推进的简洁机制。下一步应检查这些方向是否具备跨周期稳定性，并设计 shadow 版本的 `prefer_effect_direction` prior，使其在不改变规则库核心结构的前提下接受在线排序审计。

### 效果空间方向跨周期稳定性检查

新增 [audit_hopper_ability_entry_effect_space_stability.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_effect_space_stability.py)，用于检验效果空间方向是否具备跨周期可复用性。该脚本重新计算 `568` 个入口窗口的 `delta_z`，从每个可用入口组学习 `v_entry`，再将其迁移到其他入口组，报告同阈值迁移、跨阈值迁移、leave-one 全局迁移和方向余弦相似度。该步骤仍为离线审计，不修改规则库或训练流程。

输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_stability_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_stability_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_stability.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_stability.md
```

全量结果显示，组内检查仍有正信号：`in_sample` 的正 margin 比例为 `1.000`，平均 AUC 为 `0.608`。但跨周期迁移明显不足：`same_threshold_transfer` 的正 margin 比例为 `0.333`，平均 AUC 为 `0.482`；`leave_one_global` 的正 margin 比例为 `0.143`，平均 AUC 为 `0.456`。方向余弦也显示同阈值方向并不稳定，均值为 `-0.2387`。

唯一较稳定的正向迁移来自早期低阈值入口：`R>1 / cycle004 -> cycle008` 的 AUC 为 `0.684`，反向 `cycle008 -> cycle004` 的 AUC 为 `0.622`，两者方向余弦为 `0.9251`。相比之下，`R>50 / cycle012` 与 `R>50 / cycle023` 的方向余弦为 `-0.9572`，迁移 AUC 分别为 `0.422` 与 `0.375`，说明高阈值入口在不同周期可能代表不同阶段或不同局部策略，而不能被当前单一 `threshold + cycle` 上下文稳定表达。

当前结论为：效果空间方向具备组内解释力，但跨周期稳定性尚不足，不应直接进入 shadow prior。下一步应优先改进上下文划分或效果空间归一化，例如区分早期起步入口与高阈值维持入口、按状态流形而非回报阈值聚类，或对 `tva` 分量进行跨周期尺度校准。

### 最小稳定效果审计

为避免效果空间机制继续复杂化，新增 [audit_hopper_ability_entry_minimal_stable_effect.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_minimal_stable_effect.py)。该脚本只采用两个最小修订：按周期对 `delta_z` 做 robust median/IQR 归一化，并将入口上下文粗分为 `low_entry` 与 `high_entry`。其中 `low_entry` 包含 `R>1` 与 `R>10` 入口，`high_entry` 包含 `R>50` 与 `R>100` 入口。该脚本采用严格 stable/unstable 输出，不生成在线 prior。

输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_minimal_stable_effect_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_minimal_stable_effect_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_minimal_stable_effect_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_minimal_stable_effect_audit.md
```

全量审计中，原始 `568` 条窗口记录经去重后保留 `474` 条，跳过数为 `0`，可用 tier-cycle 组为 `6`。严格稳定标准为：全部 transfer 均为正 margin，全部 transfer 的 AUC 均不低于 `0.60`，且 tier 内方向余弦均值为正。

结果显示，`low_entry` 与 `high_entry` 均未通过稳定标准。`low_entry` 的 transfer 正 margin 比例为 `0.667`，AUC 不低于 `0.60` 的比例为 `0.250`，平均 AUC 为 `0.479`；`high_entry` 的 transfer 正 margin 比例为 `0.000`，平均 AUC 为 `0.449`。虽然 `low_entry` 中部分迁移仍较强，例如 `cycle004 -> cycle008` 的 AUC 为 `0.802`、`cycle008 -> cycle004` 的 AUC 为 `0.782`，但加入 `cycle007` 与 `cycle018` 后整体不稳定。

当前结论为：最小修订没有产生可上线的稳定效果方向，因此不应继续推进 shadow prior。效果空间路线保留为诊断工具；在线训练层面应继续维持简洁、可控的规则机制，把当前不稳定的阶段入口细节交还 TD-MPC2 主体训练解决，除非后续出现更明确、更局部且跨周期稳定的方向证据。

### 判别式效果空间方向审计

为检验是否存在更稳定的数学方向估计，新增 [audit_hopper_ability_entry_effect_space_discriminant.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_effect_space_discriminant.py)。该脚本沿用最小稳定审计的数据口径，即周期内 robust 归一化、`low_entry / high_entry` 轻分层、严格 stable/unstable 输出，但将方向学习方法扩展为三类：

```text
mean_diff
ridge_lda
sign_consensus_ridge_lda
```

其中 `ridge_lda` 使用正则化 Fisher/LDA 方向，`sign_consensus_ridge_lda` 在 LDA 前先保留跨训练入口组符号一致的效果维度。该设计用于检查效果空间中是否存在更稳定的线性判别方向，而不是引入更复杂的在线规则结构。

已完成 smoke 与全量审计。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_discriminant_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_discriminant_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_discriminant_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_discriminant_audit.md
```

全量结果仍未产生稳定 tier。`ridge_lda` 相比 `mean_diff` 略微提高了 transfer 中 AUC 不低于 `0.60` 的比例，但没有通过严格稳定标准：`low_entry` 的 transfer 正 margin 比例为 `0.500`，AUC 不低于 `0.60` 的比例为 `0.417`，平均 AUC 为 `0.485`；`high_entry` 的 transfer 正 margin 比例仍为 `0.000`，平均 AUC 为 `0.465`。`sign_consensus_ridge_lda` 没有进一步改善，且 leave-one 表现更弱。

当前结论为：效果空间中确实存在局部判别信号，尤其 `low_entry` 的若干 leave-one 检查可达到 `0.66-0.77` 的 AUC；但该信号无法稳定迁移到所有早期入口周期，更不能支持高阈值入口。因此，按照简洁、有效、可控的原则，效果空间规则暂不进入 online 或 shadow prior。该线保留为方法论诊断工具，后续只有在出现更局部且跨周期稳定的入口子类时再恢复规则化尝试。

### 局部动作效果优势空间审计

为进一步排除 `delta_z` 混入状态基线与周期尺度差异的问题，新增 [audit_hopper_ability_entry_local_advantage_space.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_local_advantage_space.py)。该脚本对同一 anchor 状态下的候选动作集合计算平均效果，并将真实动作效果改写为局部优势：

```text
A_z(s, a_actual) = delta_z(s, a_actual)
                 - mean_{a' in probe_bank} delta_z(s, a')
```

该处理的研究意图是回到动作选择的局部第一性问题：如果状态已经给定，真实动作相对于同一状态下其他可选动作，究竟带来了什么额外后果。脚本继续沿用 `low_entry / high_entry` 分层、周期内 robust median/IQR 归一化、`mean_diff / ridge_lda` 两类线性方向，以及跨周期 transfer、leave-one 和方向余弦稳定性检查。该审计只读既有 SD/TVA checkpoint 与轨迹，不修改规则库、不启动训练。

已完成语法检查、`80` 窗口 smoke 和全量审计。smoke 输出 `73` 条记录、跳过数为 `0`；全量输出 `474` 条记录、跳过数为 `0`。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_local_advantage_space_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_local_advantage_space_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_local_advantage_space_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_local_advantage_space_audit.md
```

结果显示，局部优势空间对 `high_entry` 有一定改善，但没有形成可上线规则。`ridge_lda / high_entry` 的 transfer 正 margin 比例为 `1.000`，AUC 不低于 `0.60` 的比例为 `0.500`，平均 AUC 为 `0.576`，平均方向余弦为 `0.364`；`mean_diff / high_entry` 的正 margin 比例同为 `1.000`，但 AUC 不低于 `0.60` 的比例为 `0.000`，平均 AUC 为 `0.557`。`low_entry` 仍不稳定：`ridge_lda` 的 transfer 正 margin 比例为 `0.500`，AUC 不低于 `0.60` 的比例为 `0.083`，平均 AUC 为 `0.482`，平均方向余弦为 `-0.129`。

当前结论为：局部动作效果优势空间更接近动作因果后果，能够缓解部分高阈值入口的状态基线问题；但它仍不能把能力入口压缩成跨周期稳定的单一线性效果方向。因此，效果空间规则仍不应进入 online prior 或 shadow prior。下一步不应继续堆叠更复杂分类器，而应转向更本质的局部反事实遗憾对象：使用 SD 推演与 TVA 价值势函数估计 `U_h(s,a)`，再比较同一状态动作集合内的 `Regret_h(s,a)` 与 `Adv_h(s,a)`。只有在同一结构化上下文中出现低遗憾、高优势、低动作熵且跨周期稳定的动作族时，才具备进入规则候选库的依据。

### Phi/Regret 势函数审计

为测试 TVA 是否可以作为局部反事实遗憾中的 `Phi`，新增 [audit_hopper_phi_regret_potential.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_phi_regret_potential.py)。该脚本定义：

```text
Phi(s) = TVA(s)
U_h(s, a) = Phi(SD_h(s, a))
Regret_h(s, a) = max_{a' in probe_bank} U_h(s, a') - U_h(s, a)
Adv_h(s, a) = U_h(s, a) - mean_{a' in probe_bank} U_h(s, a')
```

审计分三层进行：第一层检查当前 `Phi(s_t)` 与真实未来窗口后果的排序关系；第二层检查经 SD 推演后的 `U_h / Adv_h / Regret_h` 是否能区分 entry success 与 matched failure；第三层在 `tier + context_key` 内聚合最优动作族，检查是否产生低熵、跨周期的低遗憾候选。该脚本只读已有入口窗口、SD explainer 和 TVA checkpoint，不生成规则、不修改在线训练。

已完成语法检查、`80` 窗口 smoke 和全量审计。smoke 输出 `73` 条记录、跳过数为 `0`；全量输出 `474` 条记录、跳过数为 `0`。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_phi_regret_potential_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_phi_regret_potential_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_phi_regret_potential_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_phi_regret_potential_audit.md
```

第一层结果显示，TVA 作为 `Phi(s)` 具备一定排序信号，但该信号主要存在于低阈值入口和 episode 级尺度。全体窗口中，`Phi(s)` 与 episode reward 的 Spearman 为 `0.607`，与未来前向位移的 Spearman 为 `0.376`，与未来平均前向速度的 Spearman 为 `0.353`；`low_entry` 中与未来前向位移的 Spearman 为 `0.419`。但 `high_entry` 中，`Phi(s)` 与短窗口未来奖励、前向位移和前向速度的 Spearman 分别为 `-0.040 / -0.041 / -0.034`，说明高阈值入口不能直接按当前统一势函数解释。

第二层结果显示，SD 推演后的 `U_h / Regret_h` 尚不能解释入口成功与失败差异。全体窗口下，实际动作优势 AUC 在 `h=1 / 3 / 5` 分别为 `0.502 / 0.481 / 0.469`，低遗憾 AUC 分别为 `0.481 / 0.460 / 0.434`。`low_entry` 的实际动作优势 AUC 也仅为 `0.493 / 0.510 / 0.503`，没有形成稳定区分。

第三层结果显示，默认门槛下没有任何上下文动作族成为 stable-like candidate。部分上下文有局部信号，例如 `h=3` 下 `omega_neg|vx_mid|support_none|falling|support_risk -> hip_pos` 的 best-family rate 为 `0.889`，best-family entropy 为 `0.388`；但其跨周期最小 best rate 为 `0.0`，且该动作族在真实观察动作中没有出现，不能作为规则上线依据。多个候选都表现出类似问题：总体最优比例较高，但至少一个周期完全不支持，说明不是阈值单纯过严，而是同一结构化上下文下最优动作族仍随训练周期变化。

当前结论为：`Phi=TVA` 在状态排序层面有价值，但当前入口窗口上的 `U_h / Regret_h` 尚不能生成可靠规则。后续不应把该审计结果接入在线 prior，而应先分析两类问题：一是 `high_entry` 的 TVA 势函数校准是否需要单独处理；二是 `context_key` 是否仍过粗，导致不同局部控制机制被混在同一上下文中。

### TVA 效果空间参数化筛选与核方法审计

为检验是否可以在不增加规则形态复杂度的前提下，通过参数训练或核方法从 TVA 效果空间中筛选规则，新增 [audit_hopper_tva_effect_space_svm.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_effect_space_svm.py)。该脚本构造：

```text
z_phi(s,a) = [
  Phi(s),
  U_h(s,a),
  Adv_h(s,a),
  -Regret_h(s,a),
  U_h(s,a)-Phi(s)
] for h in {1,3,5}
```

其中 `Phi=TVA`，`U_h(s,a)=TVA(SD_h(s,a))`。审计模型包括 `logistic_l2`、`linear_svm` 和 `rbf_svm`。前两者用于测试是否可训练出线性规则筛选器；`rbf_svm` 只作为核方法上限诊断，不作为上线对象。验证采用 leave-one-cycle，并在 held-out 周期上用训练后的评分函数对候选动作族排序，聚合为 `tier + context_key -> top_family` 候选。

已完成语法检查、`80` 窗口 smoke 和全量审计。smoke 输出 `73` 条记录、跳过数为 `0`；全量输出 `474` 条记录、跳过数为 `0`。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_tva_effect_space_svm_audit_smoke.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_tva_effect_space_svm_audit_smoke.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_tva_effect_space_svm_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_tva_effect_space_svm_audit.md
```

全量结果为负。global leave-one-cycle 平均 AUC 分别为：`logistic_l2=0.322`，`linear_svm=0.320`，`rbf_svm=0.347`；`low_entry` 中分别为 `0.284 / 0.283 / 0.376`；`high_entry` 中分别为 `0.332 / 0.305 / 0.304`。没有任何模型产生 stable-like context。虽然部分上下文的 top family rate 达到 `0.5-0.6`，但跨周期最小 best rate 多为 `0`，best-family entropy 偏高，不能作为规则。

当前结论为：当前 TVA 效果空间不支持直接通过参数训练或核方法筛选规则。RBF SVM 没有明显优于线性模型，说明问题不是单纯“线性投影不足”；更可能是当前入口标签混合了多种机制，或 TVA 势函数与短窗口入口动作价值之间存在口径错配。因此，现阶段不应继续沿该数据口径放宽模型或上线规则，应回到标签和上下文定义的简化诊断。

### 能力入口标签与上下文基础审查

根据前述负结果，新增 [audit_hopper_entry_label_context_foundations.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_entry_label_context_foundations.py)，回到不依赖 SD/TVA 反事实和任何新模型的基础审查。该脚本只读取 `entry_candidate_segments.json`，检查入口标签是否在真实后续窗口中可分，以及当前 `context_key` 是否足以承载跨周期动作族规则。

已完成语法检查和全量审查。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_label_context_foundation_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_label_context_foundation_audit.md
```

审查结果显示，原始窗口为 `568` 条，唯一窗口标识为 `413` 个，重复率为 `0.273`。其中 `29` 个窗口存在角色冲突，即同一窗口在不同阈值事件中同时作为某个事件的 `entry_success` 和另一个事件的 `matched_failure`。同时，`entry_success` 中 `42.0%` 来自入口前一周期，`33.2%` 来自入口后一周期，真正入口当周期只占 `24.8%`。这源于入口追踪工具为观察能力形成过程而纳入入口前后周期 top episode；该设计适合诊断，但不适合直接作为规则训练标签。

标签消融显示，严格化标签可以显著增强真实后果可分性：

```text
raw:
  future_reward AUC = 0.663
  future_rootx AUC = 0.753

no_pre_entry:
  future_reward AUC = 0.720
  future_rootx AUC = 0.801

entry_cycle_only:
  future_reward AUC = 0.727
  future_rootx AUC = 0.811

entry_cycle_only + role_consistent_uids:
  future_reward AUC = 0.755
  future_rootx AUC = 0.812
```

因此，当前入口标签不是没有信号，而是被前周期扩展、重复阈值事件和角色冲突稀释。后续规则候选提取若继续使用能力入口数据，应优先改用严格标签：只保留入口当周期或入口后周期，去除角色冲突窗口，并避免同一窗口跨阈值重复加权。

上下文审查显示，当前上下文语言仍不足。原始 `context_key` 下，样本数不少于 `12` 的上下文有 `16` 个，其中 `16/16` 均被标记为 ambiguous；使用 `refined_phase` 后，仍有 `13/15` 个 ambiguous。主要表现为：同一上下文横跨多个周期和多个阈值，success/failure 角色熵高，动作族熵高，且 dominant action family 的跨周期最小命中率常为 `0`。例如 `omega_pos|vx_high|support_none|other|rotation_risk` 横跨 `cycle004/007/008/012/017/018/023/025`，角色熵为 `0.994`，动作族熵为 `0.790`，dominant family 的跨周期最小命中率为 `0`。

当前结论为：此前模型和核方法无法提取稳定规则，并非主要因为模型能力不足，而是由于基础标签和上下文尚未形成规则学习所需的稳定对象。下一步不应继续复杂化模型；应先修订能力入口数据口径，再考虑是否重新运行轻量规则筛选审计。

### 严格入口标签重跑数学审计

根据基础审查结论，新增 [build_hopper_strict_entry_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_strict_entry_segments.py)，从原始入口数据派生两版严格标签数据集：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments_strict_no_pre_entry.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments_strict_entry_cycle_only.json
```

`no_pre_entry` 保留入口当周期和入口后一周期；`entry_cycle_only` 仅保留入口当周期。两者均移除 success/failure 角色冲突窗口，并对同一窗口跨阈值重复只保留一个最高阈值实例。生成结果为：`no_pre_entry` 保留 `269` 条窗口，`entry_cycle_only` 保留 `135` 条窗口。

随后在两版严格数据上重跑基础标签审查、TVA 效果空间 SVM 审计和局部动作效果优势空间审计。综合报告写入：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/strict_entry_label_math_audit_summary.md
```

结果显示，严格标签确实改善了基础后果信号。`no_pre_entry` 的 future reward AUC 为 `0.714`，future rootx AUC 为 `0.799`；`entry_cycle_only` 分别为 `0.712` 和 `0.800`。这相比原始口径的 `0.663 / 0.753` 有明显提升。

但是，严格标签没有恢复 TVA 效果空间 SVM 可分性。`no_pre_entry` 中，global leave-one-cycle AUC 为：`logistic_l2=0.331`，`linear_svm=0.323`，`rbf_svm=0.359`；`entry_cycle_only` 中分别为 `0.235 / 0.241 / 0.397`。两版数据均没有产生 stable context。

局部动作效果优势空间有一定改善但仍不稳定。`no_pre_entry` 中，`ridge_lda / low_entry` 的 transfer AUC 从原始口径的 `0.482` 提升到 `0.536`，`ridge_lda / high_entry` 从 `0.576` 提升到 `0.601`；但 AUC>=0.60 的比例和方向余弦仍不足以通过 stable tier。`entry_cycle_only` 中 high entry 轻微改善，low entry 因样本减少未改善。

当前结论为：修订入口标签能缓解噪声，但不能单独解决规则提取问题。剩余瓶颈主要是上下文定义过粗，同一 `context_key` 下仍混合多个局部控制机制。后续不应回到复杂模型，而应在严格标签基础上修订上下文语言。

## 2026-05-23

在严格入口标签基础上新增上下文变体审计脚本 [audit_hopper_strict_context_variants.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_strict_context_variants.py)。该脚本比较原始 anchor phase、anchor refined phase、加入 anchor type、窗口主导 refined phase、支撑机制、风险机制和姿态机制等上下文定义，只检查是否出现 `context_key -> prefer_action_family` 形式的 rule-like 候选，不训练新模型、不写入规则库。

已完成语法检查，并分别在两版严格入口数据上运行：

```text
entry_candidate_segments_strict_no_pre_entry.json
entry_candidate_segments_strict_entry_cycle_only.json
```

输出为：

```text
entry_strict_context_variant_audit_no_pre_entry.json/md
entry_strict_context_variant_audit_entry_cycle_only.json/md
strict_context_variant_summary.md
```

`no_pre_entry` 口径下出现一条主要候选：`low_entry::omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward -> prefer waist_neg`。该候选有 `11` 条窗口，成功 `6` 条、失败 `5` 条；成功侧 `waist_neg` 命中率为 `1.000`，失败侧为 `0.000`，后续奖励差为 `+6.818`，后续 root-x 差为 `+0.364`，成功样本跨 `cycle007/017/018`。`entry_cycle_only` 因样本量较小，没有产生 stable-like 候选。

随后在 `no_pre_entry` 严格数据上重跑 SD/TVA 入口反事实审计，输出为 `entry_counterfactual_audit_strict_no_pre_entry.json/md`。候选规则的统计后果信号较强，但模型支持混合：成功侧 `waist_neg` 的 SD top-3 为 `2/6`，TVA top-3 为 `4/6`；失败侧 `waist_neg` 的 TVA top-3 为 `0/5`，SD top-3 为 `3/5`。据此，当前结果被记录为“上下文修订找回候选形态，但尚未达到可上线规则证据强度”。下一步应分析 SD 与 TVA 在该候选上下文中的分歧，而不是继续扩大分类器复杂度。

进一步按周期分层检查后，确认早期模型成熟度会影响反事实解释。候选上下文中，`cycle007` 的成功样本已有较高真实后续奖励，但 `waist_neg` 的 SD/TVA top-3 支持均为 `0/2`；而 `cycle017/018` 的成功样本中，TVA top-3 支持为 `4/4`，`cycle018` 中 SD 与 TVA 均为 `2/2` best。该结果支持一项修正：早期 SD/TVA 反事实不支持只能作为弱证据，不宜直接否定统计后果较强的候选规则。

同日，根据“效果空间规则也维护信心值”的设计，新增 [audit_hopper_effect_confidence_shadow.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_effect_confidence_shadow.py)。该脚本在严格 `no_pre_entry` 入口数据上构造 `context_key -> prefer_effect_direction` 的 shadow 规则，按周期用 maturity-weighted evidence 更新 confidence，不写入在线规则库。

已完成语法检查和全量审计。输出为：

```text
entry_effect_confidence_shadow_strict_no_pre_entry.json
entry_effect_confidence_shadow_strict_no_pre_entry.md
```

审计使用 `269` 条窗口，跳过数为 `0`，生成 `3` 条效果空间 shadow 规则。最高信心为 `0.633`，对应 `low_entry::omega_mid|vx_low|support_none|near_good_air|stable|first_positive_reward`；前一轮重点候选 `low_entry::omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward` 的信心为 `0.626`。该候选在 `cycle007` 获得低权重小幅增强，`cycle017/018` 的成功窗口带来主要增强，符合早期模型证据降权、后期模型证据增强的预期。当前仍不进入 online prior，因为候选的诱导动作族熵偏高，尚未证明效果方向能稳定转化为动作选择。

### Hopper 高质量轨迹标签校验与最小修正

在三种子 200k 实验完成后，对 seed1 高质量正式评估轨迹进行了状态标签校验。校验目标不是重新判断局部得分高低，而是确认当前规则生成和姿态审计所依赖的底层标签是否能正确表达“较标准 hop”状态。报告写入：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/label_calibration_seed1_high_quality_eval.md
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/label_calibration_seed1_high_quality_eval.json
```

校验发现，seed1 的高分轨迹已经表现出持续向前跳跃行为，但既有正式评估摘要中的 `grounded_hop_frac` 为 `0.0`。进一步检查原始逐帧特征后发现，`grounded_hop >= 0.5` 的比例约为 `0.199`。不一致的原因是 `online_trainer.py` 先前通过 `dominant_state_id == 6` 写入 `grounded_hop_gt_05`，而 `DMControlHopperSDModel._dominant_state_ids` 实际没有分配状态编号 `6`。因此，正式摘要低估了 hop 状态。

同时，`grounded_prone` 在高质量 hop 轨迹中也大量出现。该标签更准确地说是“前倾接地支撑”，不能直接等同于失败姿态。`rotation_penalty` 也不能单独作为失败标记，因为高角速度经常是正常 hop 周期的一部分。

已进行最小代码修正：

```text
src/world_model/dmcontrol_hopper_sd_model.py
external/tdmpc2/tdmpc2/trainer/online_trainer.py
experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py
```

修正内容包括：在 SD 状态特征中显式加入 `grounded_hop`；在 `_dominant_state_ids` 中为 hop 分配状态编号 `6`；正式评估摘要中的 `_gt_05` 字段改为直接使用原始特征阈值；审计和规则匹配脚本中将 hop 判定置于 prone 判定之前。语法检查已通过。

基于旧轨迹的快速 sanity check 显示：逐帧 `grounded_hop >= 0.5` 比例为 `0.199`，旧摘要中的 hop 比例为 `0.0`，按新口径计算的 hop 比例为 `0.199`。因此，后续规则质量审查应使用修正后的原始特征口径或重新生成正式评估摘要；旧摘要中的 `grounded_hop_frac` 不再作为可靠依据。

随后使用修正后的标签优先级，对 200k 三种子正式评估 episode CSV 重跑只读姿态审计和规则命中审计。输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/posture_comparison_corrected_labels.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/posture_comparison_corrected_labels.md
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/rule_effect_audit_official_eval_using_cycle039_rules_corrected_labels.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/rule_effect_audit_official_eval_using_cycle039_rules_corrected_labels.md
```

姿态审计显示，seed1/seed2/seed3 的 `near_good_frac` 分别为 `0.7804 / 0.5170 / 0.4734`，`phase_hop_frac` 分别为 `0.1990 / 0.3428 / 0.3152`，说明修正后高分轨迹中的 hop 成分能够被指标捕捉。seed1 的 `mean_vx = 1.3693`、`body_collision_mean = 0.0418`，与其 579.1 的平均回报一致，支持其作为标签校验锚点。

规则命中审计使用 `cycle039` 规则库重新解释 `cycle040` 正式评估轨迹。修正口径下共有 `71` 条规则参与审计，任一规则 selected gate hit 比例为 `0.6933`。旧口径下同一审计约为 `0.7669`，下降说明部分旧 `prone` 命中在新口径下被重新解释为 hop 或非 prone 状态。该结果不否定既有训练效果，因为历史训练时仍使用旧口径；但后续规则质量审查和新训练应使用修正口径。

### 修正标签口径下的规则质量审查

为进一步判断旧 `prone / rotation_risk` 规则是否实际覆盖正常 hop 周期，新增只读审计脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_corrected_label_rule_quality.py
```

该脚本读取修正标签后的规则命中审计和 `cycle039` 规则库，按规则来源、相位、hop/air-cycle 覆盖、方向一致性和潜在有害比例进行汇总，不修改规则库。输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.md
```

审计覆盖 `71` 条在线规则，其中 `23` 条被解释为显式 `prone` 恢复规则，`27` 条被解释为 hop/air-cycle 规则，`12` 条方向可疑，`8` 条在最终 eval 中未命中。结果显示，显式 `prone` 规则在修正口径下仍主要命中 `prone` 帧，而不是被大规模转移到 `hop`。因此，旧规则的问题不是简单的 hop/prone 标签互换；更准确地说，`prone` 在 Hopper 中常表示前倾接地恢复或支撑片段，不能直接等同于失败。

分组结果显示，`v8_transition_replay_safety` 规则只有 `8` 条，但 selected hit 总数为 `7747`，平均 good-state hit rate 为 `0.688`，平均方向一致率为 `0.734`；其 transition hit density 与回报相关为 `0.821`。相比之下，`v4_substate` 规则有 `63` 条，selected hit 总数为 `16121`，但平均 good-state hit rate 只有 `0.155`，平均方向一致率为 `0.542`。这说明当前最接近“正常 hop 周期控制”的有效信号主要来自 transition 规则，而不是大量旧的显式 prone 恢复规则。

教师核心规则共 `8` 条，selected hit 总数为 `3104`，平均 good-state hit rate 为 `0.515`，方向一致率为 `0.716`，整体健康度高于闭环或 legacy 规则。但部分教师或旧规则在修正标签下仍表现出较高 harmful/opposite 比例，例如若干 `prone->prefer:waist_pos/knee_neg` 或 `falling->prefer:hip_pos/waist_neg` 规则。因此，下一轮实验不宜直接把全部旧规则作为强先验，应保留 transition 规则和经过方向审查的教师核心规则，同时对方向可疑规则降权、重新归因或要求 TVA/SD 后果审计通过。

### 信心校准随机规则门控实现

为缓解多种子训练中的规则利用/探索不平衡，在线规则 prior 新增信心校准随机门控。实现位置为：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
scripts/run_hopper_v6_5k_rule_prior_smoke.sh
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

机制形式为：

```text
p_apply(rule) = p_min + (p_max - p_min) * confidence^gamma
```

其中 `confidence` 来自规则的 `effective_prior_scale` 或 `confidence` 字段。训练阶段若开启 `rule_prior_stochastic_gate`，命中规则会按 `p_apply` 随机决定是否参与本次 MPPI 候选排序；正式 eval 默认不采样，而使用 `confidence * p_apply` 的期望缩放，以避免评估噪声。

新增参数默认关闭：

```text
rule_prior_stochastic_gate = false
rule_prior_min_apply_prob = 0.10
rule_prior_max_apply_prob = 0.95
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

多种子 launcher 可通过 `--enable-rule-prior-stochastic-gate` 开启，并透传上述参数。运行日志将新增 `rule_prior_matched_rules`、`rule_prior_applied_rules`、`rule_prior_dropped_rules`、`rule_prior_apply_prob_mean` 和 `rule_prior_apply_multiplier_mean`，用于检查规则使用概率与实际启用比例。语法检查已通过：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

### 修正标签初始规则库与随机门控 smoke

新增初始规则库构建脚本：

```text
experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_initial_library.py
```

该脚本读取 `cycle039` 全量 confidence 规则库和修正标签规则质量审计，只保留两类规则：非方向可疑的 `v8_transition_replay_safety` 规则，以及方向健康的 `teacher_core_standard_hop` 规则。输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.md
```

生成结果为 `12` 条规则，其中 `7` 条为 transition 安全规则，`5` 条为教师核心健康规则。方向可疑规则、未命中规则和旧恢复规则中未通过该口径的规则均未进入初始强先验库。

随后新增只读 smoke 脚本：

```text
experiments/sd_retrospective_rule_induction/smoke_hopper_stochastic_rule_gate.py
```

该脚本从三种子 200k 正式 eval 轨迹中按 `stride=5` 抽取 `1500` 个状态，加载 `library_corrected_label_initial_v1.json`，并比较三种模式：旧确定性规则、训练式随机门控、eval 期望缩放。输出为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_initial_v1.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_initial_v1.md
```

结果显示，`1500` 个抽样状态中有 `843` 个状态命中规则，累计匹配规则数为 `940`。旧确定性模式 applied 为 `940/940`；训练式随机门控 applied 为 `639/940`，dropout 为 `301/940`，实际启用率 `0.680`，接近平均启用概率 `0.649`；eval 期望缩放 applied 为 `940/940`，dropout 为 `0`，平均 multiplier 为 `0.649`。这说明随机门控和 eval 期望缩放均按设计工作，可以进入在线短 smoke。

### 修正标签初始规则库与随机门控在线 smoke

启动并完成三种子从零在线 smoke：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_10k_corrected_initial_stochgate_seed123_20260523_r1
```

实验设置为 `seeds=1,2,3`，`cycle_steps=5000`，`cycles=2`，总预算 `10k`。第一周期即启用 `library_corrected_label_initial_v1.json` 中的 `12` 条 `confidence_active` 初始规则，并开启信心校准随机门控：

```text
p_apply = 0.10 + 0.85 * confidence
```

正式 eval 中未启用随机采样，而使用期望缩放，以避免评估阶段额外噪声。TVA terminal prior 未启用，本轮仅检验规则库与随机门控的在线闭环健康性。

结果显示，三种子均成功加载 `12` 条规则并写出 `rule_prior_matched_rules`、`rule_prior_applied_rules`、`rule_prior_dropped_rules`、`rule_prior_apply_prob_mean` 等指标。`cycle001` 完成回放导出、轨迹选择、SD 拟合、反事实归因、规则生成与共享规则库更新；`cycle002` 以 `cycle001` 的 SD checkpoint 继续拟合，`init_loaded_tensors=30`，符合 SD 接续训练设计。两轮均未新增 `confidence_active` 规则。

在线训练中的规则覆盖较稀疏。`cycle001` 仅 seed3 命中并应用 `1` 次规则；`cycle002` 三个种子各命中 `1` 次，其中 seed2 应用 `1` 次，seed1 与 seed3 各随机丢弃 `1` 次，命中时平均应用概率均为 `0.576`。因此，本轮 smoke 证明了随机门控机制已经真实进入在线规划排序并按信心概率保留探索空间，但也暴露出当前高质量初始规则主要覆盖成熟策略状态，对从零早期探索入口覆盖不足。

离线总结路径正常。`cycle001` 的 SD 拟合使用 `17465` 个 transition，验证损失降至 `0.695`；`cycle002` 使用 `41916` 个 transition 并从上一轮 SD 初始化，验证损失降至 `0.530`。完整摘要写入：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_10k_corrected_initial_stochgate_seed123_20260523_r1/cycle_002/reports/online_stochastic_gate_10k_smoke_summary.md
```

本轮结论是：工程闭环已经通过在线 smoke，下一步若进入较长周期实验，应把观察重点放在规则覆盖随训练阶段是否自然增加，以及是否需要由回放数据和 TVA/SD 后果信号自动提取早期入口规则，而不是继续只依赖成熟高分轨迹规则。

### 启动修正标签初始规则库与随机门控 200k 三种子实验

基于在线 smoke 通过的结论，启动新的从零三种子 `200k` 实验。实验目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_corrected_initial_stochgate_seed123_20260523_r2
```

实验设置为 `seeds=1,2,3`，`cycle_steps=5000`，`cycles=40`，保存 checkpoint 和 formal eval 的周期均为 `25000`，每次 formal eval 使用 `10` 个 episode。初始规则库使用：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.json
```

第一周期即启用规则先验，并启用信心校准随机门控：

```text
rule_prior_allowed_statuses = confidence_active
rule_required_status = confidence_active
p_apply = 0.10 + 0.85 * confidence
```

TVA terminal prior 未启用，以便将本轮观察重点限定在修正标签规则库、规则自学习闭环和信心随机门控本身。运行通过 `tmux` 会话 `hopper_stochgate_200k_r2` 启动，避免长周期实验受交互会话中断影响。启动检查显示三种子均加载 `12` 条初始规则，`rule_prior_enabled=true`，`rule_prior_stochastic_gate=true`。

注：同名前缀 `r1` 曾被 dry-run 写入周期配置和占位日志，未作为正式训练结果使用。

### 修正 200k 实验的 TVA terminal prior 口径

检查历史配置后确认，上一轮高均值 200k 实验 `hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1` 启用了 TVA terminal prior、强/弱接口门控、TVA audit、TVA shadow update、TVA consequence gate、v8 transition rules 和 rule-effect confidence。此前启动的 `hopper_v8formal_200k_corrected_initial_stochgate_seed123_20260523_r2` 沿用了 10k smoke 的隔离设置，只启用修正标签初始规则库与随机门控，未启用上述 TVA 和规则维护主线机制，因此不适合作为正式 200k 对照。

已停止 `r2`，并以新目录重新启动正式实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_corrected_initial_stochgate_tva_seed123_20260523_r3
```

`r3` 保留修正标签初始规则库和信心随机门控，同时恢复上一轮高均值实验的 TVA 与规则闭环口径：

```text
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
enable_tva_audit = true
enable_tva_shadow_update = true
enable_tva_consequence_gate = true
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
rule_required_status = v8_replay_safe_candidate
```

第一周期训练日志中的 `tva_terminal_prior_enabled=false` 属于预期行为，因为从零训练的第一周期尚无上一周期 SD/TVA checkpoint。完成 `cycle001` 的 SD/TVA 拟合与接口门控审计后，后续周期才会按门控结果决定是否启用 weak 或 strong TVA terminal prior。

## 2026-05-24

### 修正标签冷启动初始规则库

复查研究日志后确认，`library_corrected_label_initial_v1.json` 的构建目标是低噪声规则审计和随机门控 smoke，而不是从零训练的完整冷启动先验。该库只保留 `7` 条 transition 安全规则和 `5` 条教师核心健康规则，在修正标签口径下较干净，但删除了大量高质量教师轨迹中的 hop 入口、falling 恢复和 prone 前倾支撑规则。10k online smoke 已经暴露其从零早期覆盖不足，因此将其直接用于 200k 正式冷启动实验属于协议口径偏差。

据此新增冷启动专用构建脚本：

```text
experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_coldstart_library.py
```

该脚本使用三类历史留存产物：高质量教师核心规则库、`cycle039` 闭环规则库、修正标签规则质量审计。新库仍在修正标签口径下生成，但以冷启动覆盖为目标：教师核心规则不因成熟评估窗口中未命中而被删除，而是保留为冷启动入口规则；修正标签审计用于调节信心和记录风险；transition 规则只保留非方向可疑项；无法被 `online_rule_prior_v6` 映射的动作族被过滤。

生成的新库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.md
```

规则库共 `31` 条规则，其中 `24` 条为 `teacher_core_standard_hop`，`7` 条为 `v8_transition_replay_safety`。阶段覆盖为 `11` 条 hop、`5` 条 falling、`8` 条 prone、`7` 条 transition。所有规则均为 `confidence_active`，但信心按修正标签审计校准：审计健康规则保留原信心，成熟评估窗口未命中的教师入口规则上限为 `0.90`，mixed/low-support 规则上限为 `0.65`，direction-suspect 教师规则上限为 `0.35`。该处理保留了入口覆盖，同时允许随机门控和后续 rule-effect confidence 继续削弱或退休问题规则。

验证包括语法检查和随机门控 smoke。成熟 200k eval 轨迹上，新库 `1500` 个抽样状态中命中 `996` 步、匹配 `1642` 条规则；训练式随机门控启用率为 `0.700`。25k 早期 eval 轨迹上，新库命中 `78` 步、匹配 `119` 条规则；旧紧凑库在同一轨迹上仅命中 `45` 步、匹配 `65` 条规则。该结果说明新库没有退回旧标签，而是在修正标签口径下恢复了更合理的冷启动覆盖。

### 冷启动规则库三种子重训启动

修订 `zero` 动作族处理后，启动新的三种子从零 `200k` 实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1
```

新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1.sh
```

本轮使用 `library_corrected_label_coldstart_initial_v2.json` 作为初始规则库，保留上一条高均值实验的 TVA terminal prior、强/弱门控、TVA audit、TVA shadow update、TVA consequence gate、v8 transition rules 和 rule-effect confidence。随机门控继续使用：

```text
p_apply = 0.10 + 0.85 * confidence
```

本轮未调整随机门控阈值，原因是主要变量应集中在 cold-start 规则库修正和 `zero` 过滤。新库已通过规则信心分层降低可疑规则强度，若同时修改随机门控阈值，会削弱对规则库修正效果的解释性。若本轮后续审计显示低信心规则仍过度干扰，再单独调整 `p_min / p_max / gamma`。

`zero` 修订位置为：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py
```

在线 prior 加载时会过滤不能映射到 Hopper 关节方向的动作族；闭环规则更新时也不再把此类规则晋升或保留为有效候选。因此，即使归因过程偶然产生 `prefer:zero`，也不会进入训练时 MPPI 候选排序先验。

已停止上一条使用紧凑规则库的 `hopper_stochgate_tva_200k_r3` tmux 会话，并启动新会话：

```text
hopper_corrected_coldstart_tva_200k_r1
```

启动检查显示 `run_config.json` 已写入新初始规则库、`enable_rule_prior_stochastic_gate=true`、`enable_tva_terminal_prior=true`、`enable_tva_audit=true`、`enable_tva_shadow_update=true`、`enable_tva_consequence_gate=true`。三种子均已进入 `cycle001` 训练日志。

### teacher-core 最小改动重训线

`corrected_coldstart_stochgate_tva` 线在 `75k` 的正式评估显著低于上一条较成功的 `teacher_core_init` 线。对比后确认，差异不只是标签修订和随机门控：`library_corrected_label_coldstart_initial_v2.json` 还对多条 teacher-core 入口规则进行了大幅信心降权，并额外预置了 `7` 条 transition 规则。该后处理不符合“在较成功训练结果上改进时应保持最小扰动”的原则。

因此停止旧会话：

```text
tmux kill-session -t hopper_corrected_coldstart_tva_200k_r1
```

新增最小改动重训脚本：

```text
scripts/launch_hopper_v8formal_200k_teacher_core_stochgate_tva_minimal_seed123_20260524_r1.sh
```

新线使用原始较成功线的 `24` 条 teacher-core 初始规则：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
```

该规则库不经过 corrected-label confidence 后处理，不预置 transition 规则；当前代码中的标签修订、`zero`/invalid action-family 过滤、TVA terminal prior 强/弱门控、TVA audit、TVA shadow update、TVA consequence gate、v8 transition online 生成和 rule-effect confidence 仍保持开启。随机门控保留原阈值：

```text
p_apply = 0.10 + 0.85 * confidence
```

由于 teacher-core 规则 confidence 基本接近 `1.0`，该门控对核心规则的期望启用概率接近 `0.95`，可用于检验随机门控在“高置信核心规则不被降权”的条件下是否仍会破坏训练。若该线恢复到 `teacher_core_init` 的早期趋势，则问题主要来自 corrected-label confidence 后处理；若仍显著退化，再单独关闭随机门控进行对照。

### teacher-core 当前代码无随机门控对照线

`teacher_core_stochgate_tva_minimal` 线在 `100k` 的正式评估未恢复到 `teacher_core_init` 的早期趋势。三种子 `10ep` 正式评估为：

```text
25k  overall mean 0.198, max 5.909
50k  overall mean 0.332, max 9.442
75k  overall mean 0.067, max 1.523
100k overall mean 3.517, max 22.770
```

与较成功的 `teacher_core_init` 原线相比，主要差异集中在 seed1：原线 seed1 在 `75k` 已达到 `202.7` 的正式评估均值，而随机门控线 seed1 在 `100k` 仍只有 `1.245`。对照规则库后确认，最初 `24` 条 teacher-core 规则并未在随机门控线中更严重地退休或削弱；到 `cycle020` 仍有 `22` 条 active、`2` 条 observing，平均信心为 `0.781`。因此，本轮退化更可能来自执行层：随机门控使命中规则在训练期被间歇性丢弃，且 eval 期望缩放降低了候选排序中的有效惩罚强度。

为验证这一判断，已停止随机门控长线：

```text
hopper_teacher_core_stochgate_tva_minimal_200k_r1
```

并新增当前代码下的无随机门控干净对照脚本：

```text
scripts/launch_hopper_v8formal_100k_teacher_core_nostoch_tva_current_seed123_20260524_r1.sh
```

该线使用同一 `24` 条 teacher-core 初始规则、当前标签解释修订、`zero`/invalid action-family 过滤、TVA terminal prior 强/弱门控、TVA audit、TVA shadow update、TVA consequence gate、v8 transition online 生成和 rule-effect confidence；唯一关闭项是 `rule_prior_stochastic_gate`。预算先设为 `100k`，每 `25k` 保存 checkpoint 并做 `10ep` 正式评估。若 `100k` 前恢复 seed1 的早期入口趋势，则可将退化主要归因于随机门控执行方式；若仍未恢复，则需继续检查当前代码修订与原 `teacher_core_init` 运行环境之间的其他差异。

### 后续方法论设想：借鉴遗传算法模式定理的规则凝练机制

在当前多 episode 闭环归因框架下，提出一个后续方法论方向：规则学习过程可借鉴遗传算法中的模式定理（schema theorem）。该思想不要求照搬遗传算法，而是借鉴其核心逻辑：短的、低阶的、适应度高于平均水平的模式，会在选择过程中获得更多保留和扩张机会。

映射到当前规则归因框架中，episode 轨迹可视为个体，局部状态-动作-后果结构可视为模式，规则的解释力、SD 反事实后果优势、TVA terminal value 改善、跨轨迹和跨周期稳定性可视为适应度信号。由此，规则不应仅被理解为完整状态标签到单一动作族的映射，而可被理解为带通配符、可泛化、可特化的结构模式。例如：

```text
局部状态条件 + 通配符 + 动作族 + 短期/中期效果谓词
```

该方向与当前研究主线一致：目标不是从单条高分轨迹中抽取孤立规则，而是在多条 episode、多种子、多周期中识别能够稳定解释成功或失败后果的结构性决策模式。未来可考虑建立 schema-rule 机制，使规则适应度由以下成分共同决定：

```text
schema_fitness =
  解释力
  + SD 反事实后果优势
  + TVA terminal value 改善
  + 跨 seed / 跨周期稳定性
  - 复杂度惩罚
  - 冲突惩罚
```

该设想也为随机门控问题提供了新的解释：若某些 teacher-core 规则已表现为高适应度核心 schema，则不宜在训练中被随机丢弃，而应获得更稳定的介入机会；随机门控更适合低置信、新生成、待验证的 schema。后续在当前无随机门控对照实验结束后，可再讨论是否将现有规则库机制扩展为“核心 schema 确定性介入、候选 schema 概率性探索”的分层结构。

### teacher-core 当前代码无随机门控对照结果分析

当前代码下的 teacher-core 无随机门控 `100k` 对照线已完成：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_teacher_core_nostoch_tva_current_seed123_20260524_r1
```

正式评估结果为：

```text
25k  overall mean 0.004,  max 0.127
50k  overall mean 2.512,  max 30.930
75k  overall mean 13.060, max 101.173
100k overall mean 18.052, max 137.717
```

该结果高于保留随机门控的 teacher-core 最小改动线，但仍显著低于原较成功线 `teacher_core_init` 在同预算下的趋势：

```text
teacher_core_init:
25k  overall mean 0.010,  max 0.420
50k  overall mean 6.910,  max 37.670
75k  overall mean 67.610, max 226.670
100k overall mean 104.970, max 376.650
```

因此，随机门控确实是退化来源之一，但不是全部原因。进一步对照显示，启动脚本除预算周期外基本一致，当前线与原成功线的关键差异应来自运行时代码和规则维护语义。代码时间也显示：原成功线启动后，`src/world_model/dmcontrol_hopper_sd_model.py` 和 `experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py` 均发生过修改，因此当前线不是对原成功线的严格复现，而是当前标签解释、无效动作过滤和 TVA/rule-prior 接口下的复跑。

第 `20` 周期的规则审计显示，当前线并非规则完全不命中。原成功线与当前线分别为：

```text
teacher_core_init cycle020:
  reward_mean 28.628, reward_max 383.964
  state_hit_frac 0.575, selected_gate_hit_frac 0.534
  gate_hit_reward_corr 0.089
  transition_gate_hit_reward_corr 0.906

teacher_core_nostoch_current cycle020:
  reward_mean 4.712, reward_max 139.599
  state_hit_frac 0.501, selected_gate_hit_frac 0.453
  gate_hit_reward_corr 0.015
  transition_gate_hit_reward_corr 0.634
```

这说明当前机制仍在触发规则，但规则命中与高回报轨迹之间的关系明显变弱。初始 `24` 条 teacher-core 规则在两条线中并未整体崩溃：到 `cycle020`，两条线均有 `21` 条处于 active，但当前线额外退休或显著削弱了部分 falling/recovery 入口规则。例如：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_neg
  teacher_core_init: selected_gate_hits 9239, confidence 0.500, observing
  current_nostoch:   selected_gate_hits 0,    confidence 0.031, retired

unsupported_high_torque|omega_neg|vx_high|support_none|falling->prefer:ankle_neg
  teacher_core_init: selected_gate_hits 2039, confidence 0.984, active
  current_nostoch:   selected_gate_hits 986,  confidence 0.323, observing
```

更关键的差异出现在 v8 transition 规则。原成功线在 `cycle020` 的回放安全 transition 规则为 `18` 条，其中 `8` 条为 `soft_prefer_action_family`，`10` 条为 `avoid_action_family`；当前线只有 `11` 条，其中 `1` 条为 `soft_prefer_action_family`，`10` 条为 `avoid_action_family`。也就是说，当前线仍保留了一些“避免坏动作”的负向规则，但明显缺少“推动进入好动作模式”的建设性规则。

该差异不是单纯发生在候选生成阶段。`cycle020` 的 transition 验证中，当前线仍产生了 `7` 条 soft-prefer 候选，但回放安全筛选后只保留 `1` 条；原成功线则从 `14` 条 soft-prefer 候选中保留 `8` 条。当前线保留下来的唯一 soft-prefer 规则为：

```text
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|support_risk->soft_prefer:hip_pos
```

其回放 good-hit rate 仅约 `0.006`；而原成功线中多条 soft-prefer transition 规则的 good-hit rate 达到 `0.56-0.77`，例如：

```text
transition|transition_gap|omega_neg|vx_high|support_none|phase_any|rotation_risk->soft_prefer:hip_pos
  hits 5354, good_hit_rate 0.768

transition|transition_gap|omega_mid|vx_high|support_none|phase_any|support_risk->soft_prefer:knee_neg
  hits 1926, good_hit_rate 0.571

transition|transition_gap|omega_neg|vx_high|support_none|phase_any|support_risk->soft_prefer:knee_neg
  hits 1107, good_hit_rate 0.560
```

成功片段层面的分布也支持这一解释。`cycle020` 中，原成功线提取到 `996` 个 transition-gap 成功片段，片段 score 最大约 `4.59`；当前线只有 `575` 个片段，score 最大约 `1.36`。两条线的片段都主要来自 seed1，但当前线的片段质量明显不足，导致后续规则归纳更偏向冲突、弱支持或保守的 avoid 规则。

阶段性结论为：当前机制的问题不应再被理解为“规则没有生效”，而应理解为“规则生效后没有稳定推动策略进入可扩张的高质量片段分布”。其直接表现是：规则命中率仍较高，但建设性 transition soft-prefer 规则不足，且当前标签/筛选语义下部分 teacher-core falling/recovery 入口规则被过早削弱。后续若继续修订，应优先验证当前代码能否复现原成功线的规则生成结构，而不是继续在最终回报上做小幅比较。

### 当前代码对原成功线回放数据的规则生成复现审查

为区分“当前代码规则生成语义发生破坏”和“当前低分线未进入足够好的数据分布”两种解释，使用当前代码重新审计原 `teacher_core_init` 成功线 `cycle020` 的回放数据。审查输出独立写入：

```text
results/sd_retrospective_rule_induction/diagnostics/current_code_reaudit_teacher_core_init_cycle020_20260525
```

该审查不重训、不覆盖原实验目录，只复用原成功线的 `cycle020` 回放轨迹、训练 replay export 和 SD explainer checkpoint，依次重新执行：

```text
mine_hopper_v8_success_segments.py
mine_hopper_v8_transition_actions.py
counterfactual_hopper_v8_transition_actions.py
induce_hopper_v8_transition_rules.py
validate_hopper_v8_transition_rules.py
audit_hopper_v8_transition_replay_safety.py
```

复现结果如下：

```text
success fragments: 996
transition records: 2333
observational candidates: 30
counterfactual candidates: 14
induced transition rules: 44
validated statuses:
  heldout_supported 18
  heldout_conflict_unresolved 15
  heldout_weak 5
  heldout_weakened_by_conflict 5
  heldout_no_support 1
replay-safe transition rules: 18
replay-safe intervention counts:
  soft_prefer_action_family 8
  avoid_action_family 10
```

与原成功线 `cycle020` 存档相比，重新生成的 transition rule ID 集合完全一致；规则类型、动作族、状态分布和 replay safety 汇总完全一致。唯一差异为一条规则的 `heldout_strength` 存在约 `3e-8` 的浮点级差异，不影响任何状态判断或规则保留结果。

该复现审查支持以下判断：当前代码仍能从原成功线的高质量回放分布中生成并保留建设性 `soft_prefer` transition 规则。因此，当前无随机门控低分线的问题更可能来自在线训练早期未进入足够高质量、可扩张的片段分布，而不是当前 transition 规则生成代码本身无法提取建设性规则。后续应重点分析为何当前线在相同 teacher-core 初始规则下没有形成原成功线那样的 seed1 早期入口分布，以及现有规则维护是否在高质量片段出现前过早削弱了关键入口规则。

### 教师核心规则保护修订与短程重训启动

进一步核对显示，新实验并未缺少关键初始规则。原 `teacher_core_init` 线与当前无随机门控线均使用同一初始规则库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
```

并且 `cycle001` 中 `24` 条 teacher-core 规则均存在。差异在于，当前线的第一个 confidence update 已经将关键入口规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_neg
```

从 `confidence=1.0, confidence_active` 降为 `confidence=0.5, confidence_observing`，原因是 `active_conflict_demoted`。同一状态键下出现了早期自学习的反向候选：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_pos
```

该现象说明，问题不是教师规则未加载，而是早期低预算数据产生的冲突候选可直接削弱教师入口规则。由于教师核心规则本来就是经过高质量轨迹和审计选择后用于指导训练的先验，不能与普通自学习候选完全等权竞争。

已修订：

```text
experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py
```

修订内容为：`rule_origin == teacher_core_standard_hop` 的规则在冲突仲裁中受到保护；反向候选必须满足足够命中、显著更高信心和显著更低有害率，才可覆盖教师规则。否则教师规则保持当前状态并记录 `teacher_core_conflict_protected`。同时，闭环规则更新保留教师来源元数据，避免同 ID 周期证据覆盖后丢失教师身份。

离线重放当前无随机门控线 `cycle001` 的 confidence update 后，关键规则保持为：

```text
confidence 1.0
confidence_status confidence_active
last_effect_confidence_reason teacher_core_conflict_protected
```

反向 `prefer:knee_pos` 候选仍被降为观察态。

新增短程重训脚本：

```text
scripts/launch_hopper_v8formal_50k_teacher_core_protected_tva_current_seed123_20260525_r1.sh
```

该实验三种子从零训练至 `50k`，规则周期保持 `5k`，正式评估在 `25k` 和 `50k` 进行。判断重点为：关键教师入口规则是否不再在早期被冲突降权，以及是否更早形成高质量 transition 片段；短期回报绝对值只作为辅助信号。

### 教师核心规则保护短程重训结果

`hopper_v8formal_50k_teacher_core_protected_tva_current_seed123_20260525_r1` 已完成 `50k` 三种子短程重训。该实验使用同一 `teacher_core` 初始规则库，关闭随机规则门控，保留 TVA terminal prior 的门控机制，规则总结周期为 `5k`，正式评估在 `25k` 与 `50k` 执行。

直接验证结果如下：

```text
25k formal eval:
  seed1 mean/max 0.211 / 2.108
  seed2 mean/max 0.014 / 0.116
  seed3 mean/max 1.172 / 11.720
  overall mean/median/max 0.466 / 0.000 / 11.720

50k formal eval:
  seed1 mean/max 0.809 / 8.090
  seed2 mean/max 0.047 / 0.468
  seed3 mean/max 13.392 / 63.123
  overall mean/median/max 4.749 / 0.000 / 63.123
```

同口径历史对比如下：

```text
25k overall mean:
  protected_r1 0.466
  nostoch_current 0.004
  teacher_core_init 0.015
  stochgate_minimal 0.198

50k overall mean:
  protected_r1 4.749
  nostoch_current 2.512
  teacher_core_init 6.905
  stochgate_minimal 0.332
```

因此，保护修订在早期没有造成回报坍塌，并且相对前一条无随机门控当前线有更好的早期信号；但 `50k` 绝对回报仍处于低水平，不能解释为已经形成稳定 hop 策略。

规则机制审查结果如下：

```text
cycle001: rules=34 active=31 fragments=1   v8_safe=0
cycle005: rules=62 active=42 fragments=13  v8_safe=0
cycle009: rules=82 active=52 fragments=92  v8_safe=1
cycle010: rules=89 active=53 fragments=138 v8_safe=4
```

关键入口教师规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_neg
```

在 `cycle001` 至 `cycle010` 始终保持 `confidence=1.0, confidence_active`，后期原因变为 `effect_consistent_strengthened`。这说明本次修订确实阻止了早期反向候选对关键教师规则的错误降权。部分教师规则仍可因自身效果审计而从活跃态降为观察态，例如 `omega_neg/vx_high/support_none/falling -> prefer:ankle_neg`，该行为符合“保护冲突，不冻结规则”的设计目标。

后段机制也出现了可验证进展。`cycle006` 的 TVA terminal prior 首次通过弱门控并在线一周期；`cycle009` 与 `cycle010` 均上线同一条 replay-safe v8 transition 规则：

```text
transition|transition_gap|omega_mid|vx_high|support_none|phase_any|support_risk->soft_prefer:waist_pos
```

同时，`cycle010` 中 v8 transition 提取已经达到 `9` 条候选，其中 `4` 条为 `heldout_supported`，回放安全审查中 `1` 条为 `replay_safety_supported`、`3` 条保持观察。TVA shadow consequence gate 在 `cycle010` 中保留 `2` 条变动、拒绝 `1` 条变动。

阶段性结论为：教师核心冲突保护是必要且有效的，当前主要瓶颈已经从“教师规则是否被保留”转向“高质量片段如何更高效地转化为可上线规则”以及“TVA terminal prior 门控为何仍不稳定”。下一步不应继续围绕同一保护机制反复调整，而应针对 transition 规则上线效率、回放安全审查的保守性、TVA 门控稳定性做独立审计，并保持每项修订均可单独验证。

## 2026-05-25

### 教师核心保护线接续至 100k

基于 `50k` 正式评估均值约为 `4.749` 且规则机制未出现坍塌的事实，继续采用同一实验口径从 `50k` 接续训练至 `100k`。该接续不是新的机制修订，而是对已完成 `50k` 教师核心保护线进行中程观察，用于判断早期弱正信号是否能够在更长预算下转化为更稳定的姿态改善、规则上线与 TVA 门控信号。

新增启动脚本如下：

```text
scripts/launch_hopper_v8formal_100k_teacher_core_protected_tva_current_seed123_20260525_r1_resume50k.sh
```

接续配置保持三种子、`5k` 规则总结周期、`25k` 正式评估与 checkpoint 周期、每次正式评估 `10` 条轨迹。接续起点使用 `cycle010` 的三种子 TD-MPC2 checkpoint、共享规则库、SD checkpoint 与 TVA checkpoint；训练从 `cycle011` 开始，至 `cycle020` 结束，对应 `75k` 与 `100k` 两次正式评估。

TVA terminal prior 的初始策略设为 `off`，原因是 `50k` 结束时强/弱门控并未稳定通过；后续每个周期仍按既有门控重新审计并自动决定是否上线。该设置避免将未通过门控的 TVA 估值强行带入接续训练，同时保留其在后续周期自然上线的可能。

### 教师核心保护线与较成功线的差异复查

`teacher_core_protected` 接续至 `100k` 后，正式评估均值与保护前的当前代码无随机门控线几乎没有差别：

```text
teacher_core_nostoch_current 100k: overall mean 18.052, max 137.717
teacher_core_protected       100k: overall mean 19.075, max 191.695
```

因此，教师核心冲突保护不能解释为已经恢复性能。它修复的是早期反向候选过快削弱教师规则的问题，而不是高质量 hop 入口分布本身。

与较成功的 `teacher_core_init` 线相比，差异集中在高质量片段形成速度和片段质量，而不是规则库总量。三条线在 `100k` 的正式评估如下：

```text
teacher_core_init:
  seed1 mean/max 313.529 / 376.650
  seed2 mean/max   1.375 /  13.667
  seed3 mean/max   0.000 /   0.001
  overall mean 104.968

teacher_core_nostoch_current:
  seed1 mean/max  54.112 / 137.717
  seed2 mean/max   0.044 /   0.426
  seed3 mean/max   0.000 /   0.000
  overall mean 18.052

teacher_core_protected:
  seed1 mean/max   1.105 /  11.046
  seed2 mean/max   0.047 /   0.423
  seed3 mean/max  56.074 / 191.695
  overall mean 19.075
```

训练轨迹也显示同一结构。`teacher_core_init` 的 seed1 在 `50k` 后已经进入可扩张的高回报片段；当前保护线则主要由 seed3 形成中等质量片段，seed1、seed2 未进入类似入口分布：

```text
100k train replay summary:
  teacher_core_init seed1: max episode reward 383.96, >50 reward episodes 74
  nostoch_current   seed1: max episode reward 139.60, >50 reward episodes 24
  protected         seed3: max episode reward 178.62, >50 reward episodes 47
  protected         seed1: max episode reward  11.90, >50 reward episodes  0
```

v8 成功片段进一步说明当前线的问题不是片段数量不足，而是片段质量不足：

```text
cycle020 v8_success_fragments:
  teacher_core_init: n=996,  max_score=4.59, max_len=126, max_sum_reward=112.33
  nostoch_current:   n=575,  max_score=1.36, max_len=18,  max_sum_reward=13.23
  protected:         n=1106, max_score=1.66, max_len=24,  max_sum_reward=15.38
```

规则库数量也不支持“上线规则太少”这一解释：

```text
cycle020 confidence library:
  teacher_core_init: rules=130, active=64, teacher_core active=21, v8 transition active=5
  protected:         rules=128, active=74, teacher_core active=21, v8 transition active=6
```

TVA terminal prior 也不是早期差异的主要解释。`teacher_core_init` 到 `cycle020` 才通过弱门控，而 `protected` 在 `cycle015` 已通过弱门控、`cycle020` 通过强门控；但保护线仍未恢复 `teacher_core_init` 的 seed1 入口。这说明 TVA 门控本身看起来更健康，但没有把早期低质量片段直接转化为高质量 hop 入口。

阶段性判断为：当前代码能够维护规则，也能够在有片段时生成 transition 规则；但在线训练没有稳定进入 `teacher_core_init` 中出现过的高质量入口分布。教师核心保护修订是必要修复，但不是恢复性能的充分条件。下一步应优先做“最小复现”方向的排查：固定初始 `24` 条 teacher-core 规则、关闭随机门控、关闭新增保护以外的非必要后处理差异，逐项恢复到 `teacher_core_init` 的运行语义，尤其复查 `online_rule_prior_v6.py` 和 `dmcontrol_hopper_sd_model.py` 在 `2026-05-23` 至 `2026-05-24` 后的标签解释、无效动作过滤、规则匹配和候选排序语义。若最小复现能恢复 seed1 入口，再逐项加入保护、TVA 门控和后续机制；若仍不能恢复，则应把差异更多归因于种子随机性和早期采样分布，并转向核心 schema 的稳定入口引导设计。

### 教师核心 hop 阶段兼容匹配诊断

对 `teacher_core_init` 高质量回放和当前低分线进行只读覆盖审计后发现，当前阶段匹配语义可能低估了教师核心规则中的 `hop` 条件。审计脚本为：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_teacher_core_prior_coverage.py
```

审计输出保存于：

```text
results/sd_retrospective_rule_induction/diagnostics/teacher_core_minrepro_prior_coverage_20260525/teacher_core_prior_coverage_cycle020.json
```

在历史 `teacher_core_init` seed1 `cycle020` 高质量轨迹中，当前精确阶段匹配下的 gated match rate 约为 `0.1066`，而将教师规则中的 `hop` 视为兼容 `{hop, prone, other}` 后，gated match rate 增至约 `0.1948`。在当前保护线 seed3 `cycle020` 轨迹中，同一改动将 gated match rate 从约 `0.0647` 增至约 `0.1095`。这说明标签修订后，部分教师核心 `hop` 规则仍与轨迹中的物理片段相关，但被在线阶段标签排除。

据此新增在线参数：

```text
rule_prior_phase_match_mode = current | hop_alias
```

`hop_alias` 仅对 `rule_origin == teacher_core_standard_hop` 的五段式 substate 规则生效；当规则阶段为 `hop` 时，允许在线状态阶段处于 `{hop, prone, other}`。该改动不新增规则、不改变动作族、不改变信心机制，只用于验证“教师轨迹标签与当前在线标签语义漂移”是否为较成功线无法复现的重要原因。相关实现包括：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

同时在闭环入口中新增 `--disable-teacher-core-conflict-protection`，用于后续必要时关闭教师核心冲突保护做最小复现对照；当前 `hop_alias` 线仍保留教师核心保护。

已启动新的三种子 `100k` 最小复现诊断线：

```text
scripts/launch_hopper_v8formal_100k_teacher_core_hopalias_tva_current_seed123_20260525_r2.sh
tmux session: hopper_hopalias_100k_r2
result root: results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_teacher_core_hopalias_tva_current_seed123_20260525_r2
```

该线使用原始 `24` 条 teacher-core 初始规则、关闭随机门控、保留 TVA terminal prior 门控、保留 rule-effect confidence 和 v8 transition 自学习，并设置 `rule_prior_phase_match_mode=hop_alias`。第一周期已经完成闭环更新：`cycle001` 的规则效果审计 selected-gate 命中率约为 `0.3003`，最终规则库 `32` 条，其中 `30` 条为 `confidence_active`，`24/24` 条教师核心规则仍为 active。首轮没有新增在线规则，v8 success fragment 仅 `1` 条，最大 fragment score 约为 `0.267`，低于历史 `teacher_core_init cycle001` 的约 `0.459`。因此，当前不能仅凭首轮断言修复有效；后续判断重点为 `25k` 与 `50k` 是否恢复 seed1 早期入口、高质量 fragment 长度和回放最高回报。

### hop_alias 诊断线 25k 结果与严格最小复现线启动

`hopper_v8formal_100k_teacher_core_hopalias_tva_current_seed123_20260525_r2` 已运行至 `25k` 第一个正式评估点。结果如下：

```text
seed1 mean/max = 0.041 / 0.407
seed2 mean/max = 0.000 / 0.000
seed3 mean/max = 0.000 / 0.000
overall mean/max = 0.014 / 0.407
```

该结果与历史 `teacher_core_init 25k` 的整体均值相近，但明显低于 `protected_current 25k` 的 `overall mean/max = 0.466 / 11.720`，且训练回放到 `cycle005` 仍未出现 `>10` 或 `>50` 的高质量片段。由此判断，`hop_alias` 虽改善了教师规则覆盖解释，但没有恢复历史成功线的早期入口分布。

进一步比较发现，显式启动参数层面，历史成功线与当前 `hop_alias` 线除预算长度外几乎一致；当前线相对历史语义新增了两个可验证差异：第一，`rule_prior_phase_match_mode=hop_alias` 改变了教师核心 `hop` 规则的在线匹配语义；第二，当前代码默认启用后加入的 teacher-core 冲突保护，而历史成功线运行时尚无该保护。规则库对比显示，历史成功线在 `cycle005` 中曾将关键 falling 入口教师规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_neg
```

降为 `confidence_observing`，而当前保护机制会倾向于保留此类教师规则为 active。这说明 teacher-core 保护虽然防止早期噪声破坏教师规则，但也可能改变了历史成功线中的早期规则竞争和动作排序分布。

据此中止低信号 `hop_alias` 线，并启动更严格的最小复现线：

```text
scripts/launch_hopper_v8formal_100k_teacher_core_strict_minrepro_noprotect_currentphase_seed123_20260525_r1.sh
tmux session: hopper_strict_minrepro_100k_r1
result root: results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_teacher_core_strict_minrepro_noprotect_currentphase_seed123_20260525_r1
```

该线使用原始 `24` 条 teacher-core 初始规则、关闭随机门控、保留 TVA terminal prior 门控与 v8 transition 自学习，但显式设置：

```text
rule_prior_phase_match_mode = current
disable_teacher_core_conflict_protection = true
```

因此，它比 `hop_alias` 线更接近历史 `teacher_core_init` 的运行语义。后续判断点仍为 `25k` formal eval、训练回放中是否出现 `>10 / >50` 片段，以及 `cycle005` 附近关键 falling 入口规则是否像历史线一样被自然降权。

### 严格最小复现线进度与早期差异回溯

对 `hopper_v8formal_100k_teacher_core_strict_minrepro_noprotect_currentphase_seed123_20260525_r1` 进行了运行状态检查。当前进程仍在 `tmux` 会话 `hopper_strict_minrepro_100k_r1` 中运行，已完成 `cycle004` 的训练回放导出、SD 接续拟合、TVA 拟合、规则归因、规则库更新和 TVA 门控审计，并已进入 `cycle005 / target_steps=25004` 的三种子训练。正式 `25k` eval 文件尚未落盘；`cycle005` 目录中目前只有 `train.csv` 和空的 `models/` 目录，因此尚不能给出 25k 正式评估结果。

早期 replay 指标显示，严格最小复现线到 `cycle004` 仍未出现 `>10` 或 `>50` 的训练片段：

```text
strict_minrepro:
  cycle001 max=0.037, mean=0.0016, >10=0, fragments=2, gate_hit_frac=0.3793
  cycle002 max=0.039, mean=0.0015, >10=0, fragments=3, gate_hit_frac=0.4406
  cycle003 max=0.039, mean=0.0017, >10=0, fragments=5, gate_hit_frac=0.4169
  cycle004 max=0.150, mean=0.0033, >10=0, fragments=8, gate_hit_frac=0.4059
```

与历史较成功线早期相比，`cycle001-cycle003` 的绝对回报差距不大，但 `cycle004` 已出现方向性差异：

```text
teacher_core_init:
  cycle001 max=0.174, mean=0.0060, >10=0, fragments=1, gate_hit_frac=0.3095
  cycle002 max=0.174, mean=0.0036, >10=0, fragments=3, gate_hit_frac=0.3799
  cycle003 max=0.276, mean=0.0067, >10=0, fragments=4, gate_hit_frac=0.4301
  cycle004 max=6.549, mean=0.0599, >10=0, fragments=7, gate_hit_frac=0.4391
  cycle005 max=6.549, mean=0.0538, >10=0, fragments=10, gate_hit_frac=0.4386
```

该对比说明，当前严格最小复现线尚未恢复历史成功线在 `20k-25k` 附近出现的早期入口迹象。`hop_alias` 线虽然在 `cycle002-cycle005` 维持过 `max=9.041` 的回放片段，但其正式 25k eval 和后续片段扩张仍较弱；严格线则连这一弱入口也未复现。

进一步回溯研究日志和规则库后，初步定位到一个比启动参数更实质的早期演化差异。历史成功线中，关键教师入口规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_neg
```

在 `cycle002` 被自然降为 `confidence_observing`，原因为 `active_conflict_demoted`；同一状态键下出现了自学习反向候选：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_pos
```

而严格最小复现线虽然显式传入 `--disable-teacher-core-conflict-protection`，该 `knee_neg` 教师规则到 `cycle004` 仍保持 `confidence=1.0, confidence_active`。严格线中被冲突降权的是同状态下的 `waist_neg`，对应反向候选为 `waist_pos`。因此，差异不是“是否开启教师保护”这一单一开关，而是更早的候选生成/规则竞争路径已经发生改变。

启动脚本对比显示，严格线与历史成功线除预算长度外，显式参数差异主要是将当前默认语义写明：关闭随机门控、`rule_prior_phase_match_mode=current`、关闭 teacher-core 冲突保护。两者使用同一个初始 `24` 条 teacher-core 规则库。因此，目前更可能的差异来源是运行时代码和早期数据分布的耦合变化，而不是初始规则库缺失。

代码时间与未提交差异也支持这一判断：`src/world_model/dmcontrol_hopper_sd_model.py` 在历史成功线启动后加入了 `grounded_hop` 标签及 `STATE_HOP_*` 阈值；`online_rule_prior_v6.py`、`run_hopper_v7_multiseed_shared_rules.py` 在 2026-05-25 又加入了 phase matching 相关逻辑。虽然当前代码已经能从历史成功线 `cycle020` 的高质量回放中复现原 transition 规则集合，但它未必能复现历史线上训练早期的状态标签、候选动作归因和冲突仲裁路径。下一步若 25k 仍无入口迹象，应优先做只读离线复现审查：用当前代码分别重放历史线与严格线 `cycle001-cycle005` 的选轨、counterfactual、substate rule 和 confidence update，定位 `knee_pos` 与 `waist_pos` 分歧是在轨迹分布、状态标签、反事实排序，还是冲突仲裁阶段产生。

### 严格最小复现线 100k 完成结果

`hopper_v8formal_100k_teacher_core_strict_minrepro_noprotect_currentphase_seed123_20260525_r1` 已完成全部 `20` 个周期，终点为 `100019` steps。该线使用原始 `24` 条 teacher-core 初始规则，关闭随机门控，显式使用 `rule_prior_phase_match_mode=current`，并关闭 teacher-core 冲突保护；SD、TVA 和规则归因仍按 `5k` 周期接续更新。

正式 `10ep` 评估结果如下：

```text
25k:
  seed1 mean/max 0.022 / 0.221
  seed2 mean/max 0.005 / 0.053
  seed3 mean/max 0.001 / 0.010
  overall mean/max 0.009 / 0.221

50k:
  seed1 mean/max 0.008 / 0.083
  seed2 mean/max 0.002 / 0.017
  seed3 mean/max 0.070 / 0.695
  overall mean/max 0.027 / 0.695

75k:
  seed1 mean/max 0.008 / 0.081
  seed2 mean/max 0.074 / 0.699
  seed3 mean/max 0.000 / 0.000
  overall mean/max 0.027 / 0.699

100k:
  seed1 mean/max 0.007 / 0.075
  seed2 mean/max 0.073 / 0.579
  seed3 mean/max 7.452 / 32.661
  overall mean/max 2.511 / 32.661
```

训练 replay 侧也未恢复历史成功线的高质量入口分布。`cycle020` 共导出 `600` 条 replay episode，整体均值约 `0.171`，最高回报仍为 `8.566`，没有 `>10`、`>50` 或 `>100` 的 episode。最终规则命中率不低，`cycle020` 的 selected gate hit fraction 约为 `0.4995`，但规则命中未能转化为可扩张的高质量 hop 片段。

最终规则库包含 `120` 条规则，其中 `64` 条为 `confidence_active`、`34` 条为 `confidence_observing`、`22` 条为 `confidence_retired`；`24` 条 teacher-core 规则仍在库中。`cycle020` 首次有 `2` 条 v8 transition replay-safety 规则合并入库，但其中一条 `omega_neg/vx_mid/support_none/rotation_risk -> soft_prefer:hip_neg` 在同周期 confidence 更新后被降为 `confidence_observing` 且信心约 `0.053`，另一条 `omega_neg/vx_low/support_none/rotation_risk -> soft_prefer:hip_neg` 保持 `confidence_active`、信心 `0.56`。TVA terminal prior 到 `cycle020` 通过强门控，但该通过发生在终点周期，未对本轮训练形成实质在线作用。

该结果说明，严格恢复显式启动语义并关闭 teacher-core 保护仍不能复现历史 `teacher_core_init` 在 `50k-100k` 的 seed1 入口趋势。当前差异更可能来自早期数据分布、状态标签解释、反事实候选排序和 confidence 冲突路径的耦合，而不是单一启动参数或初始规则库差异。下一步应停止围绕该线继续盲目接续，优先做 `teacher_core_init` 与 strict-minrepro 的 `cycle001-cycle005` 离线并排复现审查，定位早期从 `knee_pos` 冲突路径转向 `waist_pos` 冲突路径的具体发生环节。

### grounded_hop 标签早中期轨迹复核

对历史教师核心初始化线 `hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1` 与最新严格最小复现线 `hopper_v8formal_100k_teacher_core_strict_minrepro_noprotect_currentphase_seed123_20260525_r1` 的早期、中期训练回放轨迹进行了重新标注审查。审查未直接依赖已导出的 `grounded_hop` 列，而是从逐帧 15 维 Hopper 原始观测重新调用当前 SD 特征函数，比较无 hop 旧 phase 口径与当前包含 hop 的 phase 口径。

结果保存于：

```text
results/sd_retrospective_rule_induction/reports/grounded_hop_label_early_mid_recomputed_audit_20260526.json
results/sd_retrospective_rule_induction/reports/grounded_hop_label_early_mid_recomputed_audit_20260526.md
```

主要发现为：`grounded_hop` 在最早期几乎不触发，但到历史线中期开始逐渐触发；触发帧几乎全部由旧口径 `prone` 改判为 `hop`，且大量伴随低全足支撑、旋转风险或后续短窗旋转风险。例如历史线 `cycle020` 中 `grounded_hop` 占比为 `0.0274`，其中 `99.9%` 来自旧 `prone`，`95.6%` 为低全足支撑，后续 20 帧内出现旋转风险的 hop 片段比例为 `99.2%`。最新严格线中触发比例较低，但同样主要来自旧 `prone`，且低全足支撑比例接近 `99%`。

据此修订此前判断：高质量轨迹校正确认了 `grounded_hop` 作为诊断特征的必要性，但当前门控条件不足以保证其可作为跨训练阶段的在线 phase 标签。后续应将 `grounded_hop` 暂时限制为审计特征，或另设更严格的 `stable_grounded_hop` 在线门控；在此之前，不宜让 `grounded_hop` 在规则先验 phase 匹配中直接覆盖 `prone`。

### grounded_hop 在线 phase 解耦实现

按早中期轨迹复核结论，对在线规则先验进行了最小修订：`grounded_hop` 保留为 SD 诊断特征，但默认不再参与 `online_rule_prior_v6.py` 的 phase 覆盖。新增参数：

```text
rule_prior_enable_grounded_hop_phase = false | true
```

默认值为 `false`。默认口径下，在线 phase 顺序恢复为不含 hop 覆盖的旧语义；只有显式启用该参数时，`grounded_hop` 才会在 `rear` 和 `prone` 之间作为 `hop` phase 参与规则匹配。`run_hopper_v7_multiseed_shared_rules.py` 与 `run_hopper_v6_5k_rule_prior_smoke.sh` 已同步透传该参数，并在训练指标中记录 `rule_prior_grounded_hop_phase_enabled`。

本次修订的目的不是删除 `grounded_hop`，而是将“静态审计标签”和“在线规则语境标签”分离。后续若需要利用标准 Hop 起跳前信息，应通过带前序状态和后续结果的转移上下文规则实现，而不是仅依赖单帧 `grounded_hop`。

### 转移上下文规则候选提取器

新增离线脚本：

```text
experiments/sd_retrospective_rule_induction/extract_hopper_transition_context_rules.py
```

该脚本从 replay trace 的原始观测重新计算 SD 特征，构造 `past dominant phase -> current candidate phase -> future outcome` 记录。规则前件只使用可在线观察的 `prev_phase/current_phase/omega_bin/vx_bin/support_bin/risk_bin`；未来窗口只用于估计观察性后果，不作为在线匹配条件。输出规则状态为 `transition_context_candidate`，表示尚未经过 SD/TVA 反事实验证，不得直接并入在线规则库。

完成两项 smoke 审查：

```text
results/sd_retrospective_rule_induction/reports/hopper_transition_context_grounded_hop_hist_best_cyc005_020_20260526.json
results/sd_retrospective_rule_induction/reports/hopper_transition_context_grounded_hop_hist_best_cyc005_020_20260526.md
results/sd_retrospective_rule_induction/reports/hopper_transition_context_grounded_hop_strict_minrepro_cyc005_020_20260526.json
results/sd_retrospective_rule_induction/reports/hopper_transition_context_grounded_hop_strict_minrepro_cyc005_020_20260526.md
```

历史教师核心初始化线 `cycle005-cycle020` 聚焦 `grounded_hop_candidate` 时得到 `11504` 条转移上下文记录、`106` 个上下文和 `19` 条观察性候选规则；最新严格最小复现线同口径仅得到 `222` 条记录、`31` 个上下文和 `3` 条候选规则。历史线的大上下文主要集中在 `prev=prone|cur=grounded_hop_candidate`，并进一步由 `support` 与 `risk` 分裂为不同情境。这说明转移上下文提取器已经能把单帧 `grounded_hop` 拆解为更细的来源和风险语境，但当前候选仍是观察性相关，下一步应接入 SD/TVA 反事实验证，而不是直接上线。

### 高分轨迹教师转移上下文候选库 v1

基于历史教师核心初始化线中已经确认存在较标准 Hop 姿态的高分 official eval 轨迹，生成第一版教师转移上下文候选库。输入轨迹为 seed1 的 `175k` 与 `200k` official eval：

```text
logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle035_target175034/official_eval/step_0175034/episode_csv/episode_*.csv
logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv/episode_*.csv
```

输出如下：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_candidates.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_candidates.md
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_candidates_loose.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_candidates_loose.md
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_grounded_hop_candidates.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_grounded_hop_candidates.md
```

严格阈值版本基于 `9440` 条上下文记录、`255` 个上下文，仅形成 `2` 条观察性候选规则；宽松版本形成 `31` 条候选；聚焦 `grounded_hop_candidate` 的版本基于 `1810` 条记录、`41` 个上下文，形成 `3` 条候选。所有候选均标记为 `transition_context_candidate`，不具备直接上线资格。

本次结果同时暴露一个方法问题：即便在高分标准 Hop 轨迹中，当前 SD 静态 phase 与风险标签仍频繁将局部帧解释为 `prone/falling/rotation_risk/unstable`。因此，高分轨迹教师候选库已经可以作为后续 SD/TVA 反事实验证的候选池，但还不能被视为高可信教师规则库。下一步应优先修订转移上下文中的 outcome/风险评分，使其更贴合周期性 Hop 轨迹，而不是把当前观察性候选直接用于在线训练。

### 高分轨迹规则稀疏性审查

针对“低分轨迹各不相同、高分轨迹相对趋于一致，但高分轨迹中提取出来的规则很少”这一现象，对 `teacher_transition_context_v1` 严格候选库进行了瓶颈审查。审查报告保存于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/rule_scarcity_audit_20260526.md
```

严格候选库共有 `9440` 条动作记录和 `255` 个转移上下文。筛选路径为：`255` 个上下文中只有 `61` 个达到 `min_context_support=20`，其中只有 `24` 个上下文的胜出动作族达到 `min_family_support=5`，最终只有 `2` 个上下文的动作族优势超过 `min_score_advantage=0.04`。上下文支持数的中位数仅为 `4`，上四分位数为 `18`，说明高分行为在当前符号表示下被显著切碎。

更关键的是，当前 outcome 表示与周期性 Hop 的实际语义不匹配。高分轨迹中的 `9440` 条记录有 `88.7%` 的未来窗口被标为 `unstable`，`60.4%` 的当前状态被标为 `rotation_risk`，`61.5%` 被标为 `support_none`，而 `support_full` 仅为 `10.4%`。这些比例与轨迹的高分和较标准 Hop 姿态并不矛盾，而是表明当前短窗口评分函数把周期性 Hop 中正常的腾空、快速旋转和接触切换大量解释为风险或不稳定。

据此修订解释：高分轨迹并非不一致，稀疏规则也不说明高质量轨迹缺乏经验规律。更合理的解释是，高质量 Hop 的一致性存在于周期和效果层面，而当前规则提取器要求在静态转移上下文内找到单一动作族的局部显著优势。该问题定义过窄，既会把一致的周期行为切成多个风险上下文，也会削弱多关节协调动作在单动作族统计中的优势。因此，后续不宜简单降低规则阈值，而应优先校准未来效果表示：保持最终规则形式简洁，但将候选动作评价改为周期感知或 TVA 辅助的效果空间判断，再由 SD/TVA 反事实验证筛选可上线规则。

### 周期感知 outcome 校准

针对高分标准 Hop 轨迹被大量标为 `unstable` 的问题，对 `extract_hopper_transition_context_rules.py` 增加可回退的 outcome 口径：

```text
--outcome-mode legacy
--outcome-mode cycle_v1
```

`legacy` 保持原有短窗判据；`cycle_v1` 将未来后果改为周期感知效果评分，主要依据未来窗口内的前向位移、平均前向速度、非坍塌、保持高度和接触周期延续，并且只在旋转/高角速度与低进展或碰撞共同出现时施加较强惩罚。该修订只改变离线候选动作的后果评分，不改变规则前件，也不直接影响在线规则库。

审查报告保存于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/cycle_v1_calibration_audit_20260526.md
```

同一批 `9440` 条高分教师轨迹记录中，旧口径将 `8374` 条标为 `unstable`；`cycle_v1` 将 `9069` 条标为 `cycle_continue`，仅 `310` 条标为 `unstable`。这确认了旧判据对周期性 Hop 存在明显误标：高角速度、腾空和低全足支撑不应被单独等同于失控。

基于 `cycle_v1` 重新生成教师转移候选库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_candidates_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_candidates_cycle_v1_loose.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_transition_context_grounded_hop_candidates_cycle_v1.json
```

严格阈值下候选规则由旧口径的 `2` 条增加到 `6` 条，宽松口径为 `39` 条，聚焦 `grounded_hop_candidate` 为 `5` 条。该结果说明校准修复了主要误标，但尚未完全解决上下文碎片化问题；上下文中位支持数仍为 `4`。因此，这些规则仍保持 `transition_context_candidate` 状态，后续需要用 SD/TVA 反事实或失败轨迹对照验证，不能仅因来自高分轨迹而直接上线。

### 教师规则覆盖瓶颈审查

对 `cycle_v1` 严格教师候选的覆盖面进行了离线抽象粒度审查。报告保存于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/coverage_bottleneck_audit_20260526.md
```

当前完整上下文由 `prev_phase/current_phase/omega_bin/vx_bin/support_bin/risk_bin` 构成。六条严格规则的上下文覆盖为 `844/9440=8.9%`，实际胜出动作支持为 `86/9440=0.9%`。这说明当前规则是高置信局部偏好，而不是高覆盖教师知识。

将同一批记录用不同抽象粒度重新聚合后发现，若保留完整上下文，覆盖为 `8.9%`；若将支撑和风险粗化，覆盖约为 `30.4%`；若去掉 `risk_bin`，覆盖约为 `43.4%`；若使用 `prev/current/vx/coarse_support`，覆盖约为 `34.8%`。这表明覆盖偏窄主要来自上下文过细，尤其是 `risk_bin`、细粒度支撑类型和细粒度角速度共同切碎了周期性 Hop 行为。

同时，粗化后反复出现的规则包括 `prev=prone|cur=falling|vx_high|support_none -> waist_neg`、`prev=falling|cur=prone|vx_high|support_full -> ankle_neg`、`prev=falling|cur=grounded_hop_candidate|vx_high|support_none -> ankle_neg` 等。这些规则在语义上更像高质量 Hop 周期维持规则，但因前件更宽，必须经过更严格的 SD/TVA 反事实验证。

据此形成下一步建议：规则提取应采用两层结构。第一层用较宽的效果上下文定义候选覆盖，例如 `prev_phase/current_phase/vx_bin/coarse_support/optional coarse_omega`；第二层将细粒度 `omega/support/risk` 作为诊断修饰项，用于验证阶段削弱、拒绝或分裂候选，而不是在候选生成阶段过早切碎。这样可以保持在线规则形式简洁，同时避免高分教师知识被过细标签分割。

### broad_context_v1 候选提取实现

按“宽口径候选提取、SD/TVA 反事实验证、信心机制在线维护”的方法论方向，对 `extract_hopper_transition_context_rules.py` 增加可回退的上下文模式：

```text
--context-mode full
--context-mode broad_context_v1
```

`full` 保持原有完整上下文，`broad_context_v1` 使用：

```text
prev_phase
current_phase
coarse_omega_bin
vx_bin
coarse_support_bin
```

其中 `risk_bin`、细支撑类型和细角速度不再参与候选生成，但仍保留在记录的 `parts` 中，供后续反事实验证和诊断分层使用。该实现只影响离线候选提取，不改变 SD 特征、不改变在线规则库，也不影响既有 `full` 模式结果。

基于同一批高分教师轨迹生成宽口径候选库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_broad_context_v1_candidates_cycle_v1.json
```

审查报告保存于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/broad_context_v1_audit_20260526.md
```

结果显示，在相同严格阈值下，`full` 模式为 `9440` 条记录、`255` 个上下文、`6` 条规则，唯一规则上下文覆盖为 `8.9%`；`broad_context_v1` 为 `9440` 条记录、`97` 个上下文、`6` 条规则，唯一规则上下文覆盖提升至 `39.7%`。其中最大覆盖候选为：

```text
prev=prone|cur=falling|omega=omega_nonneg|vx=vx_high|support=support_none -> waist_neg
```

其上下文覆盖为 `2115/9440=22.4%`。但该上下文内多个动作族得分接近，例如 `waist_neg`、`ankle_neg` 和 `hip_pos` 的均值差异较小。因此，宽口径提取应被理解为高召回候选机制，而非直接上线机制。下一步应开发 broad-candidate validator，对每条宽候选在细粒度 `omega/support/risk` 子区域中进行 SD/TVA 反事实验证，并输出 `validated_broad_rule`、`validated_with_exclusions`、`split_required` 或 `rejected`。

### simple_v1 互斥状态标签重构

根据状态标签应保持过程无关的原则，对 `extract_hopper_transition_context_rules.py` 增加可回退的 phase schema：

```text
--phase-schema legacy
--phase-schema simple_v1
```

`simple_v1` 使用六个互斥标签：

```text
air_rising
air_falling
ground_front
ground_rear
ground_upright
other
```

`grounded_hop` 不再作为互斥 phase 标签使用，仅保留为诊断特征。过去窗口与当前帧均使用同一套 `simple_v1` 标签，避免旧口径中 `prev_phase` 与 `current_phase` 语义不一致的问题。`ground_front` 只表示当前几何/接触形态，不表示失败趴地，也不表示成功 Hop；成功或失败由过去上下文、速度/支撑修饰项、未来效果和后续反事实验证决定。

完成高分教师轨迹与低分对照轨迹验证，报告保存于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/simple_v1_phase_schema_audit_20260526.md
```

高分教师轨迹为 `20` 条 official eval 轨迹，回报范围 `491.805-614.571`，均值 `571.935`。在 `9440` 条记录中，当前 phase 分布为 `ground_front 57.6%`、`air_falling 30.8%`、`air_rising 8.6%`、`ground_rear 1.9%`、`other 1.2%`，`cycle_v1` 后果中 `cycle_continue` 占 `96.1%`。

低分对照轨迹为 strict-minrepro 100k 的 `30` 条 official eval 轨迹，回报范围 `0.000-32.661`，均值 `2.511`。在 `14124` 条记录中，当前 phase 分布为 `ground_rear 66.3%`、`ground_front 32.6%`、`air_falling 1.0%`、`ground_upright 0.1%`，`cycle_v1` 后果中 `unstable` 占 `99.9%`。

该结果说明，同一套过程无关状态标签可以同时描述高分周期性 Hop 与低分失败轨迹；二者差异体现在状态分布、速度/支撑修饰项和未来效果上，而不是状态标签名称本身。基于 `simple_v1 + broad_context_v1 + cycle_v1` 生成的新候选库如下：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_simple_v1_broad_context_candidates_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_simple_v1_full_context_candidates_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_low_score_simple_v1_broad_context_candidates_cycle_v1.json
```

高分宽口径候选为 `9440` 条记录、`82` 个上下文、`6` 条候选规则，规则上下文覆盖为 `39.7%`；低分宽口径候选为 `14124` 条记录、`98` 个上下文、`5` 条候选规则，规则上下文覆盖为 `6.8%`。后续教师核心规则应优先从高分 `simple_v1` 宽候选中做 SD/TVA 反事实验证。

### broad-candidate validator v1

新增观察性对照验证器：

```text
experiments/sd_retrospective_rule_induction/validate_hopper_broad_context_candidates.py
```

该验证器不将规则上线，也不宣称因果有效。其作用是在进入 SD/TVA 反事实验证前，对宽口径候选进行高分教师轨迹、低分对照轨迹和细粒度诊断子区域的一致性审查。候选提取仍使用宽上下文；细粒度 `omega/support/risk` 仅用于诊断是否需要排除或分裂。

对 `library_teacher_simple_v1_broad_context_candidates_cycle_v1.json` 运行验证，输出保存于：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/validation_teacher_simple_v1_broad_context_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/validation_teacher_simple_v1_broad_context_cycle_v1.md
```

结果为 `6` 条候选中 `5` 条进入 `counterfactual_ready`，`1` 条被拒绝：

```text
counterfactual_ready: 5
rejected_low_teacher_cycle_continuation: 1
```

被拒绝规则为：

```text
prev=ground_front|cur=ground_front|omega=omega_nonneg|vx=vx_high|support=support_none -> knee_neg
```

该规则在高分教师轨迹中上下文支持为 `81`，但 `cycle_continue_rate=0.840`、`unstable_rate=0.160`，且主要细分子区域 `omega=omega_pos|support=support_none|risk=rotation_risk` 的不稳定率为 `0.196`，未达到作为教师候选的稳定性要求。

通过观察性对照、进入反事实验证队列的候选为：

```text
prev=ground_front|cur=other|omega=omega_nonneg|vx=vx_high|support=support_none -> waist_neg
prev=air_falling|cur=ground_front|omega=omega_nonneg|vx=vx_high|support=support_full -> ankle_neg
prev=air_falling|cur=ground_front|omega=omega_neg|vx=vx_high|support=support_partial -> waist_neg
prev=ground_front|cur=air_falling|omega=omega_nonneg|vx=vx_high|support=support_none -> waist_neg
prev=ground_front|cur=air_rising|omega=omega_neg|vx=vx_high|support=support_none -> ankle_pos
```

其中最大覆盖候选为 `ground_front -> air_falling, vx_high, support_none -> waist_neg`，高分教师上下文支持 `2115`，目标动作族支持 `296`，观察优势 `0.0420`，`cycle_continue_rate=0.996`，`unstable_rate=0.004`；低分对照中同上下文支持仅 `24`，且 `unstable_rate=1.000`。这说明该候选有较强的教师轨迹一致性与低分对照区分度，但仍需 SD/TVA 反事实验证确认 `waist_neg` 相对于相近动作族确实有因果优势。

### SD/TVA 联合反事实验证口径修订

新增并修订宽候选反事实验证脚本：

```text
experiments/sd_retrospective_rule_induction/validate_hopper_broad_context_counterfactuals.py
```

初版脚本将 `SD` 终端结构评分与 `TVA` 终端价值分别排序，容易被误读为二者独立投票。根据当前方法设定，反事实价值应采用“`SD` 短视界推演收益 + `TVA` 终端价值补偿”的联合口径。因此脚本已修订为：对同一状态下的有限动作探针执行 `horizon=5` 的 `SD` 推演，逐步累计 `model.predict_plan_reward`，并在终端加入 `gamma^h * TVA(s_h)`，其中 `gamma=0.99`。`SD` 终端结构评分与单独 `TVA` 排序仅保留为审计字段。

同一批 `counterfactual_ready` 候选、同一组高分 trace、同一 `cycle_040` 的 `SD/TVA` checkpoint 已完成重跑，输出为：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/counterfactual_teacher_simple_v1_broad_context_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/counterfactual_teacher_simple_v1_broad_context_cycle_v1.md
```

重跑后五条候选在联合评分下仍均为 `not_supported`。其 `joint top3` 比例分别为 `0.025`、`0.146`、`0.115`、`0.052`、`0.073`，说明问题不只是候选动作没有成为第一名，而是多数状态下连联合评分前三也难以进入。该结果否定的是“高分轨迹中观察到的单动作族可直接作为教师规则”的假设，而不是否定高分轨迹或转移上下文提取本身。后续若继续使用该资料，应优先在同一宽上下文中直接用 `SD + TVA` 联合反事实排序生成候选动作族，再审查这些候选是否稳定、简洁和可解释。

### 反事实驱动候选生成

根据上述结论，新增反事实驱动候选生成脚本：

```text
experiments/sd_retrospective_rule_induction/generate_hopper_broad_context_counterfactual_rules.py
```

该脚本保留 `simple_v1 + broad_context_v1 + cycle_v1` 的状态与上下文口径，但将动作族选择从观察频率改为 `SD rollout return + TVA terminal value` 的联合反事实排序。对每个高支持宽上下文，脚本采样最多 `96` 个状态，使用 `zero` 与四个关节正负方向、三种幅度的动作探针进行 `horizon=5` 推演，统计各动作族进入联合评分前三与第一的比例。

已执行：

```text
python experiments/sd_retrospective_rule_induction/generate_hopper_broad_context_counterfactual_rules.py \
  --source-library results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_teacher_simple_v1_broad_context_candidates_cycle_v1.json \
  --sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/sd_explainers/sd_explainer_cycle_040.pt \
  --tva-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/tva/hopper_tva_cycle_040.pt \
  --output results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/counterfactual_generated_teacher_simple_v1_broad_context_cycle_v1.json \
  --device cpu
```

输出保存为：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/counterfactual_generated_teacher_simple_v1_broad_context_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/counterfactual_generated_teacher_simple_v1_broad_context_cycle_v1.md
```

本轮共读取 `9440` 条高分教师轨迹记录、`82` 个宽上下文；其中 `19` 个上下文达到测试支持门槛，生成 `13` 条 `counterfactual_rule_candidate`，另有 `6` 个上下文保留为 `weak_counterfactual_pattern`。主要候选包括：

```text
prev=ground_front|cur=air_rising|omega=omega_nonneg|vx=vx_high|support=support_none -> hip_neg
  joint_top3=1.000, joint_best=0.906

prev=ground_front|cur=air_falling|omega=omega_nonneg|vx=vx_high|support=support_none -> hip_neg
  joint_top3=0.969, joint_best=0.729

prev=air_falling|cur=ground_front|omega=omega_neg|vx=vx_high|support=support_partial -> knee_neg
  joint_top3=0.854, joint_best=0.719

prev=air_falling|cur=ground_front|omega=omega_nonneg|vx=vx_high|support=support_full -> hip_pos
  joint_top3=0.698, joint_best=0.573
```

与此前五条观察性候选相比，新的候选动作族明显不同。例如，原先 `ground_front -> air_falling, support_none` 的观察候选为 `waist_neg`，但联合反事实候选为 `hip_neg`；原先 `air_falling -> ground_front, support_full` 的观察候选为 `ankle_neg`，但联合反事实候选为 `hip_pos`。这说明宽上下文本身仍能承载结构，但动作方向应由反事实动作价值生成，而不能由高分轨迹中的实际动作共现直接决定。

当前候选仍保持离线状态，尚未进入初始教师规则库。后续需要进行低分对照、跨周期稳定性和规则库转换审计，避免把单次 `cycle_040` 的 `SD/TVA` 偏好直接写入在线训练。

### 反事实生成候选的低分对照与跨周期稳定性审计

新增候选可靠性审计脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_counterfactual_generated_rule_reliability.py
```

该脚本固定已经生成的 `13` 条反事实候选，不重新生成规则，只在指定 trace 集合与指定 `SD/TVA` checkpoint 下复核目标动作族是否仍稳定进入联合评分前列。

已完成两项审计：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/reliability_low_strict_minrepro_100k_cycle040_sdtva.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/reliability_high_teacher_cycle035_sdtva.json
```

低分对照使用 strict-minrepro 100k 的 `30` 条 official eval 轨迹，共 `14124` 条记录、`98` 个上下文。结果为：

```text
no_overlap: 6
not_supported: 3
weak: 3
supported: 1
```

其中 `supported` 的规则为：

```text
prev=ground_front|cur=air_falling|omega=omega_neg|vx=vx_high|support=support_none -> hip_neg
```

该规则在低分轨迹中上下文支持 `20`，`joint_top3=0.900`、`joint_best=0.550`，但对应未来 `good=0.000`、`unstable=1.000`。因此它不能作为成功教师规则直接保留，应进入 `reject_or_defer`。

跨周期稳定性使用同一高分教师轨迹，但将 `SD/TVA` 从 `cycle040` 换为 `cycle035`。结果为：

```text
supported: 12
not_supported: 1
```

唯一跨周期不稳定规则为：

```text
prev=ground_front|cur=ground_front|omega=omega_neg|vx=vx_high|support=support_none -> knee_pos
```

该规则在 `cycle040` 中 `joint_top3=0.677`，但在 `cycle035` 中降至 `0.406`，且目标动作族第一比例仅 `0.052`，应暂缓或剔除。

综合低分对照与跨周期稳定性后，当前 `13` 条候选可分为：

```text
retain: 8
caution: 3
reject_or_defer: 2
```

`retain` 规则为：

```text
prev=ground_front|cur=air_rising|omega=omega_nonneg|vx=vx_high|support=support_none -> hip_neg
prev=ground_front|cur=other|omega=omega_nonneg|vx=vx_high|support=support_none -> hip_neg
prev=air_falling|cur=air_falling|omega=omega_nonneg|vx=vx_high|support=support_none -> hip_neg
prev=air_falling|cur=ground_front|omega=omega_neg|vx=vx_high|support=support_full -> knee_neg
prev=air_falling|cur=ground_front|omega=omega_neg|vx=vx_high|support=support_partial -> knee_neg
prev=air_falling|cur=ground_front|omega=omega_nonneg|vx=vx_high|support=support_none -> hip_neg
prev=air_falling|cur=ground_front|omega=omega_nonneg|vx=vx_high|support=support_full -> hip_pos
prev=ground_front|cur=ground_front|omega=omega_nonneg|vx=vx_high|support=support_partial -> hip_pos
```

该审计支持当前修订方向：宽上下文可以保留，动作族应由 `SD/TVA` 联合反事实生成；但在进入教师规则库前，仍必须通过低分对照和跨周期稳定性筛查。下一步可基于 `retain` 子集构造一个只读教师候选库，并再执行一次规则语义审查与覆盖审查。

### 反事实教师规则初始化库与新一轮在线训练

已将 `retain: 8` 的反事实教师规则转换为在线可读取的初始化规则库：

```text
experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_init_library.py
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_counterfactual_teacher_retain8_confidence.json
```

该规则库使用 `source=sd_tva_broad_context_counterfactual`、`rule_origin=sd_tva_counterfactual_teacher_core_v1`、`status=confidence_active`。规则保护机制已扩展为可配置的 `protected_teacher_origins`，本轮实验同时保护 `teacher_core_standard_hop` 与 `sd_tva_counterfactual_teacher_core_v1`。保护只用于避免弱冲突证据过早降级教师规则，不表示规则被冻结；后续若出现足够强的有害证据，仍可通过信心机制削弱。

为支持 `prev=...|cur=...` 的宽转移上下文规则，在线规则先验与行为审计脚本已补齐时间上下文匹配逻辑。行为审计按每条 episode 轨迹独立维护 `prev_simple_phase`，避免跨 episode 串联导致命中统计失真。

新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1.sh
```

语法检查与 dry-run 已通过：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_init_library.py
bash scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1.sh --cycles 1 --dry-run
```

正式三种子 `200k` 训练已启动：

```text
logs/launch/hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1.out
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1
```

首轮训练日志显示三种子均读取到 `rules=8`，且 `rule_prior_enabled=true`、`rule_prior_allowed_statuses=confidence_active`。由于第一周期尚无已拟合的在线 `SD/TVA` checkpoint，`TVA terminal prior` 在 `cycle001` 按门控策略保持关闭；后续周期完成 `SD` 拟合和 `TVA` 训练后再由接口审计决定是否启用。

## 2026-05-27

### 教师规则信心维护语义修订

当前 `200k` 三种子实验中，初始 `8` 条反事实教师规则仍保留在规则库内，但部分规则因低命中、零命中或普通有害率波动降为 `confidence_observing`。该现象暴露出一个语义问题：教师规则来自高质量轨迹，并已通过 `SD/TVA` 联合反事实、低分对照和跨周期稳定性审计；因此，在线训练分布尚未覆盖教师上下文时，不应将低命中解释为规则错误。

已修订 `apply_rule_effect_confidence.py` 的教师规则更新逻辑。受保护教师来源包括当前实验使用的：

```text
teacher_core_standard_hop
sd_tva_counterfactual_teacher_core_v1
```

修订后，教师规则在 `stats is None`、`hits == 0` 或 `hits < min_effect_hits` 时保持信心不变；一致性证据仍可增强；只有命中数不少于 `hard_harmful_min_hits` 且有害率不低于 `hard_harmful_threshold` 的强反例才可削弱教师规则。该改动使教师规则保护从“只保护冲突仲裁”扩展为“同时保护低证据情形”，但不冻结教师规则，也不取消强反例证伪机制。

该修订主要面向后续从头重训。当前正在运行的实验中，既有周期已经发生的低命中削弱不作回补；未来使用同一教师规则库重新启动训练时，教师规则不会再因未触发或少量样本而被动衰减。

### 规则语言多方言问题记录

当前在线 prior 与规则审计使用同一匹配入口，但根据规则 key 格式进入不同语义分支：

```text
prev=...|cur=...        -> 转移上下文规则，主要由反事实教师规则使用；
event|omega|vx|support|phase -> v4 单帧子状态规则，主要由普通在线归纳规则使用；
transition|...          -> v8 transition replay safety 规则；
toe_recovery_*          -> 诊断型规则。
```

因此，命中系统本身并非教师规则与普通规则完全分离，但规则语言实际上存在多套方言。当前 `prev->cur` 机制已经贯穿教师规则提取、在线匹配和效果审计；普通在线规则归纳仍主要使用单帧状态规则。这会带来方法论上的不一致：高可信教师规则能够区分“从哪个状态进入当前状态”，而普通自学习规则多数仍只能识别当前状态。

该问题已记录为未来统一方向。统一目标不是简单指定为 `prev->cur`，也不预设必须采用 v8 transition 或更复杂阶段规则；后续需研究哪一种上下文表达能够在保持规则维护简洁性的同时，提高跨阶段决策识别能力，并具备问题间泛化潜力。

### `prev->cur` 与 MDP 状态充分性的关系

围绕 `prev->cur` 是否应作为通用规则语言组成，进一步讨论了其与 MDP / Bellman 状态无前效假设之间的关系。

理论上，在严格 MDP 中，若 `s_t` 是完整马尔可夫状态，则到达 `s_t` 的历史路径不应影响最优后续决策。贝尔曼递推依赖当前状态和动作：

```text
V(s_t) = max_a E[r_t + gamma V(s_{t+1}) | s_t, a]
```

而不显式依赖 `s_{t-1}->s_t`。因此，如果规则层所使用的状态表达本身已经充分，`prev->cur` 不应提供额外决策信息。

但当前规则系统使用的并不是完整仿真状态或充分潜在状态，而是粗粒度抽象标签，例如：

```text
air_falling
air_rising
ground_front
ground_rear
ground_upright
other
```

这类标签具有可解释性和可泛化性，但显然会压缩大量动力学信息。相同的当前标签可能来自不同过程，例如从空中落地进入 `ground_front`，从失败恢复进入 `ground_front`，或起跳前短暂进入 `ground_front`。这些情形在粗标签上相同，但后续合理动作和风险并不完全相同。

因此，`prev->cur` 并不是对 MDP 无前效性的否定，而是在抽象规则状态不充分时，对被压缩掉的运动方向、阶段进程和接触变化信息进行低成本补充。它可以被理解为规则层的有限历史状态扩展，服务于抽象状态的可判别性。

进一步讨论后形成如下判断：对于目标为可泛化通用机制的规则系统，不宜把 `prev->cur` 仅视为某些任务中的临时补丁。复杂问题中，完整 MDP 状态表达往往难以获得，而粗标签知识、诊断标签或可解释状态摘要则更容易获得。在这种条件下，`prev->cur` 作为默认语境字段更可靠，也不会带来明显计算开销。它可能带来“前效暗示”的表面风险，但该风险已经通过上述解释澄清：历史并非作为额外因果变量使用，而是作为抽象状态不充分时的语境补全。

由此，未来统一规则语言时，倾向于将规则前件统一为：

```text
prev_abstract_state -> current_abstract_state + context_features -> action_prior
```

其中 `prev_abstract_state` 可取 `unknown`，但字段始终存在。这样，教师规则与在线自学习规则可共享同一上下文表达，不再区分“单帧规则”和“转移上下文规则”。该设计保持规则形式简洁，同时增强阶段识别能力，并更符合跨任务泛化目标。

### `150k` 机制有效性确认与当前线终止

`hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1` 已运行至 `150k` 正式评估。结果为：

```text
seed1 mean = 271.43, max = 286.97
seed2 mean =   2.71, max =  13.29
seed3 mean = 286.41, max = 302.83
overall mean = 186.85, max = 302.83
```

该结果已超过 `external/tdmpc2/results/tdmpc2/hopper-hop.csv` 中原始 TD-MPC2 `200k` 三种子均值 `166.63`。虽然 seed2 仍明显滞后，但 seed1 与 seed3 在 `150k` 前已经稳定进入较高分区间，足以作为当前机制有效性的初步证据。因此，当前实验已按研究判断终止，保留已落盘的 `150k` 结果与 `cycle030` 中间产物，不再继续至 `200k`。

### 统一 `prev->cur` 普通在线规则归纳

基于前述规则语言讨论，已将普通在线归纳的 `v4_substate` 规则从单帧子状态改为统一 `prev->cur` 抽象转移上下文。修订内容包括：

```text
analyze_substate_rules.py:
  生成 prev=<上一抽象状态>|cur=<当前抽象状态>|omega=<粗角速度>|vx=<速度档>|support=<粗支撑> 形式的 substate_key；

validate_substate_rules.py:
  在每条 episode 内维护上一帧抽象状态，并按同一 prev->cur key 进行回放验证；

online_rule_prior_v6.py:
  对 substate_key 中的 prev=... 使用转移上下文候选门控；

audit_hopper_v8_widefunnel_rule_effects.py:
  对 substate_key 中的 prev=... 使用转移上下文审计门控。
```

使用 `cycle029` 历史数据执行离线 smoke test：

```text
python -m py_compile ...
python analyze_substate_rules.py ... --output /tmp/substate_prevcur_rules_v4.json
python validate_substate_rules.py ... --rules /tmp/substate_prevcur_rules_v4.json --output /tmp/substate_prevcur_validation_v5.json
```

结果显示，新生成的 `48` 条普通候选规则全部为 `prev=...|cur=...` 形式，离线验证 `48` 条均有命中。因此，教师规则与普通在线自学习规则已在规则前件表达上统一到 `prev->cur` 机制。

### 新一轮 `200k` 重训启动准备

新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_seed123_20260527_r1.sh
```

新实验保留上一轮已验证的主线设置：三种子、`5k` 规则更新周期、`25k` 正式 `10ep` eval/checkpoint、共享规则库、`TVA terminal prior` 接口门控、无随机门控。新增差异为：

```text
教师规则低证据保护；
普通在线规则统一 prev->cur 前件表达；
不加入触发规则时保留探索空间；
不加入模式定理思想。
```

dry-run 已通过，确认路径、参数和闭环步骤可展开。

### 新一轮 `200k` 重训正式启动

已清理 dry-run 残留目录，并正式启动：

```text
hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_seed123_20260527_r1
```

启动口径为三种子 `1,2,3`，总周期 `40`，每周期 `5000` 环境步，总预算 `200k`；每 `25k` 保存 checkpoint 并执行 `10ep` 正式评估；SD 回归、反事实归因、规则更新仍按 `5k` 周期执行。初始教师规则库沿用：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_counterfactual_teacher_retain8_confidence.json
```

本次实验启用教师规则低证据保护，普通在线规则统一采用 `prev->cur` 前件表达；保留 `TVA terminal prior` 门控接口，但第 1 周期由于尚无 SD/TVA checkpoint，实际策略为关闭；未启用随机门控，也未加入模式定理机制。启动后已确认 `run_config.json` 中 `cycles=40`，三个 seed 的训练日志均进入真实训练，未出现 dry-run 标记。

### `100k` 有效信号确认与保守随机门控重训

`hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_seed123_20260527_r1` 已完成 `100k` 正式 `10ep` 评估：

```text
seed1 mean =   7.900, max =  17.197
seed2 mean = 304.221, max = 399.889
seed3 mean =  65.766, max = 136.634
overall mean = 125.962, max = 399.889
```

该结果高于本地同口径历史 `100k` 主要对照线，说明教师规则低证据保护、统一 `prev->cur` 在线规则前件、TVA terminal prior 门控和当前闭环规则维护组合已形成有效信号。由于 seed1 仍未打开，后续研究重点转向在规则先验利用和探索保留之间取得更好平衡。

已终止该线继续训练，保留 `100k` checkpoint、正式 eval 和 `cycle020` 中间结果。随后将信心随机门控默认参数从早期强随机口径调整为保守口径：

```text
rule_prior_min_apply_prob = 0.40
rule_prior_max_apply_prob = 0.98
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

该设置使教师规则和高信心规则通常保持较高启用率，同时让新上线或边界信心规则保留更多探索空间。新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate040098_seed123_20260527_r1.sh
```

新实验除启用 `--enable-rule-prior-stochastic-gate` 与上述概率参数外，保持上一条有效线的协议不变：三种子、`5k` 闭环更新、`25k` 正式 `10ep` eval/checkpoint、共享规则库、教师规则保护、统一 `prev->cur` 在线规则归纳、TVA terminal prior 门控和 TVA 审计。

该实验已正式启动：

```text
hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate040098_seed123_20260527_r1
```

启动检查显示 `run_config.json` 中 `enable_rule_prior_stochastic_gate=true`，`rule_prior_min_apply_prob=0.4`，`rule_prior_max_apply_prob=0.98`，`rule_prior_confidence_prob_power=1.0`，`rule_prior_stochastic_eval=false`。三种子 `cycle001` 训练日志均已进入 TD-MPC2 训练过程，并显示已安装 `8` 条教师规则；第 1 周期由于尚无 SD/TVA checkpoint，TVA terminal prior 按门控策略暂为关闭。

### `0.40-0.98` 随机门控负向结果与近确定性门控重训

`hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate040098_seed123_20260527_r1` 已完成 `75k` 正式 `10ep` 评估：

```text
seed1 mean = 0.000, max = 0.000
seed2 mean = 3.954, max = 19.297
seed3 mean = 0.000, max = 0.001
overall mean = 1.318, max = 19.297
```

该结果显著弱于上一轮无随机门控 `prevcur_protect` 线在 `75k` 的 `overall mean = 73.011, max = 250.571`。同期审计显示教师规则仍全部在线，规则库持续扩展，规则命中比例正常上升，因此当前结果被记录为“较宽随机门控削弱早期连续教师引导”的负向证据，而不是规则生成或教师保护机制失效。

已终止该线继续训练，保留至 `75k` 的 checkpoint、正式 eval、规则库和审计文件。随后新增近确定性随机门控启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1.sh
```

新线仅调整规则触发后的训练期执行概率：

```text
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

其余协议保持不变：三种子、总预算 `200k`、`5k` 闭环更新、`25k` 正式 `10ep` eval/checkpoint、共享规则库、教师规则低证据保护、统一 `prev->cur` 在线规则归纳、TVA terminal prior 门控和 TVA 审计。

## 2026-05-28

### 近确定性随机门控线的差异排查

对 `hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1` 与历史较成功线 `hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_seed123_20260527_r1` 进行了配置、规则库、训练回放分布和运行时代码排查。

显式配置差异仅限于规则随机门控：

```text
enable_rule_prior_stochastic_gate: false -> true
rule_prior_min_apply_prob: 0.10 -> 0.95
rule_prior_max_apply_prob: 0.95 -> 0.995
```

其中无随机门控线的 `min/max_apply_prob` 为未启用参数，不参与实际行为。两条线使用同一个 `8` 条教师规则初始库，且教师规则在已检查周期内均保持 `8/8 confidence_active`。近确定性随机门控线在 `75k` 的正式 `10ep` 评估为：

```text
seed1 mean = 0.823, max = 8.227
seed2 mean = 0.468, max = 4.115
seed3 mean = 0.049, max = 0.490
overall mean = 0.446, max = 8.227
```

同期无随机门控线在 `75k` 为 `overall mean = 73.011, max = 250.571`，说明近确定性门控并未恢复较成功线的早期入口趋势。

进一步比较训练回放分布发现，关键分叉发生在第 `6-10` 个闭环周期，而非正式评估末端。无随机门控线从 `cycle007-cycle010` 开始出现 `18.068、59.410、85.508、96.554` 等较高回报训练轨迹，回放均值逐步升至 `3.979`；近确定性随机门控线至 `cycle016` 仍基本受限在 `max ~= 11.705`，回放均值约 `0.537`。这表明当前问题主要是早期探索数据分布未进入高回报入口，而不是教师规则丢失。

规则命中审计显示，近确定性随机门控的实际丢弃比例很低：`cycle005-cycle016` 中命中规则后的平均丢弃约为 `0-0.04` 条/规划步，规则实际启用率通常接近 `0.97-1.00`。因此，`0.95-0.995` 参数本身并未大量关闭规则。

代码层面发现一个更重要的实现因素：`online_rule_prior_v6.py` 中 `_rule_apply_multiplier` 在 `stochastic_gate=true` 时，无论 `prob` 多接近 `1`，都会在训练模式下调用 `torch.rand` 进行伯努利采样。该采样发生在 MPPI 规划循环内部、最终探索噪声和候选采样之前，会消耗全局 PyTorch 随机数流。因此，即使规则几乎总是被保留，开启随机门控也会改变 TD-MPC2 原有 MPPI 采样、Gumbel 选择和最终动作探索噪声的随机序列。该行为足以解释为什么近确定性门控线与无门控线在早期即进入不同轨迹分布。

当前结论为：随机门控思想本身仍可作为“规则触发时保留探索空间”的机制，但现有实现并不是对成功线的局部、可控扰动；它同时改变了规划器的随机数流，导致难以判断性能下降来自探索保留机制本身，还是来自随机数流重排引发的轨迹路径分叉。后续若继续验证随机门控，应先将门控采样与 TD-MPC2/MPPI 主随机数流隔离，例如使用独立生成器或可复现的状态哈希门控；在此之前，不宜把当前近确定性门控线作为机制有效性的可靠负例。

### 终止近确定性门控线并隔离门控随机数流

已终止当前 `stochgate0950995` 实验进程。终止前该线已推进至 `cycle017` 后处理/训练附近，尚未完成 `100k` 正式评估；保留已落盘的 `75k` 正式 eval、checkpoint、规则库、SD/TVA 拟合和审计文件。

随后完成随机门控实现修订。`HopperRulePriorV6` 现在在初始化时创建私有 CPU `torch.Generator`，由 `rule_prior_stochastic_seed` 初始化；训练阶段 `_rule_apply_multiplier` 使用该私有生成器采样规则门控，不再调用全局 `torch.rand`。多种子 launcher 新增 `--rule-prior-stochastic-seed`，并按 `base_seed + cycle * 1000 + seed` 派生每个训练进程的门控种子；`run_hopper_v6_5k_rule_prior_smoke.sh` 同步透传 `RULE_PRIOR_STOCHASTIC_SEED`。

本次修订涉及：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

验证已完成：

```text
py_compile online_rule_prior_v6.py run_hopper_v7_multiseed_shared_rules.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
global_rng_unchanged= True
```

该修订使后续随机门控实验成为对规则触发机制本身的可归因检验，而不是对 TD-MPC2/MPPI 随机路径的隐式扰动。

### 隔离 RNG 后的近确定性随机门控重启

已启动隔离随机数流后的近确定性门控三种子 `200k` 重训：

```text
hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1.sh
```

本轮保持上一条近确定性门控线的主要协议：三种子 `1,2,3`，总预算 `200k`，`5k` 闭环更新，`25k` 正式 `10ep` eval/checkpoint，共享规则库，教师规则低证据保护，统一 `prev->cur` 在线规则归纳，TVA terminal prior 门控和 TVA 审计。关键差异是使用已修订的规则门控独立随机数流，并显式设置：

```text
rule_prior_stochastic_seed = 20260528
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

启动检查显示 `run_config.json` 已落盘，主控进程和三种子训练进程均已进入 `cycle001`。三种子的训练命令分别收到派生门控种子：

```text
seed1 cycle001: rule_prior_stochastic_seed = 20261529
seed2 cycle001: rule_prior_stochastic_seed = 20261530
seed3 cycle001: rule_prior_stochastic_seed = 20261531
```

该线用于重新检验“规则触发时保留极小探索窗口”本身的效果，而不再混入 MPPI 主随机数流被扰动的问题。

### 终止 0.95-0.995 紧窗口线并启动 0.90-0.995 放宽窗口线

隔离 RNG 后的 `0.95-0.995` 近确定性随机门控线已在 `75k` 后继续推进至 `cycle016` 训练阶段。已落盘的 `75k` 正式 `10ep` 评估为：

```text
seed1 mean = 0.005, max = 0.049
seed2 mean = 0.000, max = 0.000
seed3 mean = 185.868, max = 218.869
overall mean = 61.958, max = 218.869
```

该结果说明随机数流隔离后框架已经恢复基本训练能力，且 `seed3` 已进入较高回报区域；同时，多种子间不平衡仍然显著，`seed1/seed2` 未能有效启动。由于该线的探索释放窗口很窄，继续至 `100k` 的边际信息价值有限，故按当前实验判断终止该线，保留已落盘的 checkpoint、正式评估、规则库、SD/TVA 拟合及审计文件。

随后新建并启动放宽探索窗口三种子 `200k` 重训：

```text
hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0900995_rngiso_seed123_20260528_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0900995_rngiso_seed123_20260528_r1.sh
```

新线保持教师规则库、`prev->cur` 在线规则归纳、教师规则强反例保护、`5k` 闭环更新、`25k` 正式 `10ep` eval/checkpoint、TVA terminal prior 门控、TVA 审计、共享规则库和独立门控随机数流不变。唯一实质改动是将规则触发后的训练期应用概率下界从 `0.95` 放宽至 `0.90`：

```text
rule_prior_min_apply_prob = 0.90
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_seed = 20260528
rule_prior_stochastic_eval = false
```

该实验用于检验在不显著削弱教师规则连续引导的前提下，略微扩大规则触发后的自由探索空间，是否能缓解多种子早期入口不平衡。

启动后复查发现，`r1` 主控及三种子训练进程在 `cycle001` seed data pretraining 开始处退出，未产生模型文件，也未进入闭环后处理；训练日志未出现 Python traceback。为避免复用半截 `cycle001` 目录，保留该启动失败痕迹，后续正式运行改用 `r2` 实验名：

```text
hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0900995_rngiso_seed123_20260528_r2
```

`r2` 仅改变实验名和输出目录，协议参数与 `r1` 完全一致；启动方式改为 `setsid` 后台脱离，以避免启动会话结束影响子进程。

### 终止 0.90-0.995 放宽窗口线并回到 0.95-0.995 紧窗口 300k

`0.90-0.995` 放宽窗口线已推进至 `100k` 正式评估之后，并在 `cycle023` 附近停止。其正式评估显示，放宽窗口没有解决三种子整体平衡问题：

```text
25k overall mean = 0.479, max = 14.208
50k overall mean = 0.492, max = 8.018
75k overall mean = 1.004, max = 10.684
100k overall mean = 25.065, max = 148.517
```

其中 `seed1/seed3` 在后期出现较明显训练入口，但 `seed2` 始终未能启动。与此前 `0.95-0.995` 紧窗口线相比，放宽窗口只在部分口径上降低了强种子早期冲高造成的离散度，并未真正提高弱种子启动概率；同时早期总体均值明显较低。因此，当前阶段不再继续该线。

后续回到 `0.95-0.995` 紧窗口，并将总预算扩大至 `300k`，以检验两点：第一，隔离随机数流后的紧窗口早期效果是否可复现；第二，在较长训练预算下，规则教师与在线规则维护是否能继续推动整体姿态和回报提升。新线为：

```text
hopper_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1.sh
```

该线保持三种子、共享规则库、`5k` 闭环更新、`25k` 正式 `10ep` eval/checkpoint、教师规则强反例保护、统一 `prev->cur` 在线规则归纳、TVA terminal prior 门控和 TVA 审计不变；主要差异为：

```text
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
cycles = 60
total_budget = 300k
```

### 独立 rich 初始化教师规则提取

为改善早期入口路径敏感性，新增离线 rich 初始化教师规则提取步骤。该步骤不修改正在运行的训练过程，也不写入在线闭环规则库；其目的在于先独立检验能否从高质量轨迹的中早期成长链路中提取更丰富的教师规则。

新增脚本：

```text
experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_rich_init_library.py
```

该脚本不再像 `retain8` 版本那样硬编码保留规则集合，而是根据候选规则、同分布高质量审计和低分对照审计自动选择 active / observing 教师规则。

本次第一版输入为 `teacher_core_init` seed1 在 `100k、125k、150k、175k、200k` 的 official eval 轨迹。这些轨迹覆盖从中等回报进入稳定高分 hop 的成长链路，用于近似追溯“进入成功轨道”的入口上下文。SD/TVA 使用已有成熟 checkpoint：

```text
sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_035/sd_explainers/sd_explainer_cycle_035.pt
tva_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_035/tva/hopper_tva_cycle_035.pt
```

离线候选生成采用宽上下文规则形式：

```text
prev=<previous_simple_phase>|cur=<current_simple_phase>|omega=<coarse_omega>|vx=<vx_bin>|support=<coarse_support> -> prefer:<action_family>
```

第一版输出：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/counterfactual_generated_teacher_rich_simple_v1_broad_context_cycle_v1.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/reliability_high_teacher_growth_chain_sdtva.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/reliability_low_strict_minrepro_100k_sdtva.json
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
```

结果摘要：

```text
候选生成：49 个上下文，30 条 counterfactual_rule_candidate，2 条 weak_counterfactual_pattern。
高质量成长链路审计：28 supported，2 weak。
低分对照审计：12 no_overlap，8 not_supported，4 weak，6 supported。
最终 rich 初始化库：30 条规则，其中 22 条 confidence_active，8 条 confidence_observing。
```

低分对照中为 `supported` 的规则暂不激活，只作为观察规则保留。该处理保持教师初始化规则的解释性和谨慎性：高质量轨迹提供成功入口候选，SD/TVA 联合反事实提供动作后果判断，低分对照用于排除明显混杂规则。

### Rich 初始化教师规则触发审计

对 `library_counterfactual_teacher_rich_init_confidence.json` 中的 `22` 条 `confidence_active` 规则进行了离线触发审计。审计只读取轨迹，不改变训练过程。

规则覆盖摘要：

```text
action_family:
  hip_neg = 11
  hip_pos = 4
  knee_pos = 4
  knee_neg = 2
  ankle_neg = 1

prev phase:
  ground_front = 10
  air_falling = 9
  ground_rear = 3

current phase:
  ground_front = 12
  air_falling = 4
  air_rising = 2
  ground_rear = 2
  other = 2

support:
  support_none = 12
  support_partial = 6
  support_full = 4
```

与旧 `retain8` 教师库相比，rich active 规则中有 `7` 条重合，新增 `15` 条 active 规则。新增部分主要覆盖 `vx_mid`、`ground_rear` 和若干 `air_falling -> ground_front`、`ground_front -> air_falling` 转接上下文。

早期失败轨迹触发审计：

```text
audit = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/audit_rich_active_on_current300_early_25_75.json
episodes = 90
episode_reward_mean = 3.006
episode_reward_max = 92.112
any_rule_selected_gate_hit_frac = 0.0494
gate_hit_reward_corr = 0.5585

25k mean_reward = 0.001, mean_hit_density = 0.0484
50k mean_reward = 0.452, mean_hit_density = 0.0380
75k mean_reward = 8.567, mean_hit_density = 0.0618
```

高质量成长链路触发审计：

```text
audit = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/audit_rich_active_on_teacher_growth_chain.json
episodes = 50
episode_reward_mean = 479.566
episode_reward_max = 614.571
any_rule_selected_gate_hit_frac = 0.4320
gate_hit_reward_corr = 0.7429
```

该结果说明 rich 教师规则在高质量成长链路中覆盖较密，在早期失败轨迹中只窄幅命中入口/转接状态。早期审计中的动作方向一致率不能直接解释为规则质量，因为这些轨迹并未使用 rich 规则训练；相反，部分高命中规则在失败轨迹中与实际动作方向相反，正是规则先验可能介入修正的场景。下一步应重点审查这些高命中规则是否存在语义冲突，以及若用于重训，需将 `sd_tva_counterfactual_teacher_rich_init_v1` 加入教师保护来源。

### 终止 retain8 线并启动 rich 初始化教师规则重训

已终止正在运行的 `retain8` 初始化三种子长线：

```text
hopper_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1
```

该线停止时已完成 `150k` 正式评估，并进入 `cycle032` 训练阶段。已落盘结果保留，不删除。

随后启动 rich 初始化教师规则三种子 `300k` 重训：

```text
hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1.sh
```

启动日志：

```text
logs/launch/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1.out
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1
```

相对被终止的 `retain8` 线，本轮只替换初始化教师规则库，并补充教师保护来源：

```text
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1
```

其余核心协议保持不变：

```text
seeds = 1,2,3
cycles = 60
cycle_steps = 5000
formal_eval_freq = 25000
formal_eval_episodes = 10
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
TVA terminal prior gate = enabled
v8 transition rules = enabled
rule-effect confidence = enabled
```

启动检查显示主控进程和三种子 `cycle001_target5000` 训练进程均已进入运行状态。`cycle001` 尚无 SD/TVA checkpoint，因此 TVA terminal prior policy 为 `off`，这与从零训练的预期一致。

## 2026-05-31

### 终止随机门控作为主要改进假设，并启动无随机门控 rich 初始化消融

`hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1` 已完成 `300k` 正式评估。最终 `step=300059` 三种子结果为：

```text
seed1 = 131.397
seed2 = 209.762
seed3 = 517.310
mean = 286.156
std = 166.552
```

与公开原始 TD-MPC2 `300k` 三种子结果 `271.0 / 316.3 / 191.9`、均值 `259.733`、标准差 `51.407` 相比，本轮 rich 初始化随机门控线只有小幅均值优势，且主要由 seed3 拉高，三种子方差明显更大。因此，近确定性随机门控 `0.95-0.995` 没有解决多种子间的不平衡，也不能作为当前性能提升的稳定来源。

基于该结果，取消规则触发时的探索保留窗口，启动同协议无随机门控消融。新增脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh
```

新实验为：

```text
hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1
```

相对于上一轮 rich 初始化随机门控线，本轮保持以下配置不变：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 60
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_allowed_statuses = confidence_active
TVA terminal prior gate = enabled
v8 transition rules = enabled
rule-effect confidence = enabled
```

唯一目标改动是移除以下随机门控参数：

```text
--enable-rule-prior-stochastic-gate
--rule-prior-min-apply-prob 0.95
--rule-prior-max-apply-prob 0.995
--rule-prior-confidence-prob-power 1.0
--rule-prior-stochastic-seed 20260528
```

启动前已完成 `bash -n`、参数展开 dry-run 和无随机门控参数检索检查。该 dry-run 曾在输出目录中生成 `cycle_001` 至 `cycle_060` 的占位配置和 `DRY RUN` 训练日志；正式运行会从 `cycle001` 起逐步覆盖这些文件，因此后续进度判断应以主控进程、GPU 训练进程和当前周期真实训练日志为准，而不是仅以未来周期目录是否存在为准。由于普通 `nohup ... &` 后台进程会被当前工具会话清理，正式启动改用 `setsid` 脱离会话：

```text
setsid env PYTHONUNBUFFERED=1 bash scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh > logs/launch/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.out 2>&1 < /dev/null &
```

启动 PID 为 `735645`。启动检查显示主控进程已进入 `run_hopper_v7_multiseed_shared_rules.py`，三种子训练进程分别为 `735652`、`735653`、`735654`，运行于 GPU1、GPU2、GPU3。`cycle001` 中 `rule_prior_stochastic_gate=false`，且因尚无 SD/TVA checkpoint，`tva_terminal_prior_policy=off`。当前 GPU1、GPU2、GPU3 已出现训练进程和显存占用，实验正式进入 `cycle001_target5000`。

随后复核确认 `cycle001_target5000` 三种子均已写出 `models/final.full.pt`，主控进入 `cycle001` 的 replay 导出、manifest 构建、轨迹分层、归因和 `fit_sd_explainer` 后处理阶段。此时 GPU1、GPU2、GPU3 暂时空闲属于预期现象，因为该阶段主要在 CPU 上运行。

`cycle001` 后处理随后正常完成，TVA-SD interface gate 未通过强/弱门控，因此 `cycle002` 的 `tva_terminal_prior_policy` 仍为 `off`。主控已启动 `cycle002_target10001` 三种子训练，三条线均从各自 `cycle001_target5000/models/final.full.pt` 恢复，规则库切换为 `cycle001/rules/library_closed_loop_cycle_001_confidence.json`，且训练命令继续显示 `rule_prior_stochastic_gate=false`。

## 2026-06-01

### 无随机门控 rich 初始化线推进至 100k

`hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1` 已超过 `100k` 正式评估点，并继续推进至 `cycle023_target115022`。截至当前检查，后台主控 PID `735645` 仍在运行，三种子训练进程正在 GPU1、GPU2、GPU3 上执行 `cycle023`。训练命令继续显示 `rule_prior_stochastic_gate=false`。

正式评估结果如下：

```text
25k:  seed1 = 0.000000, seed2 = 0.044779, seed3 = 0.034680, mean = 0.026486
50k:  seed1 = 0.460186, seed2 = 3.042969, seed3 = 0.000157, mean = 1.167771
75k:  seed1 = 3.040382, seed2 = 11.054194, seed3 = 28.849342, mean = 14.314639
100k: seed1 = 0.000000, seed2 = 54.524605, seed3 = 70.497707, mean = 41.674104
```

该结果显示，无随机门控线在 `100k` 时 seed2 与 seed3 已出现入口信号，但 seed1 仍未启动，整体均值仍低于此前 `prevcur_protect` 有效线的 `100k` 表现。`cycle022` 的 TVA terminal prior gate 首次以 weak 模式通过，`cycle023` 起 TVA terminal prior 以 `weight=0.005`、`max_bonus=0.01` 进入训练。

## 2026-06-03

### 无随机门控 rich 初始化线从 300k 接续至 500k

`hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1` 已完成 `cycle060_target300059`，主控进程正常退出，最终共享规则库、SD checkpoint、TVA checkpoint 和三种子 TD-MPC2 full-state checkpoint 均已落盘。

`300k` 正式评估结果为：

```text
seed1 = 344.609655
seed2 = 290.076124
seed3 = 280.834895
mean = 305.173558
std = 28.139585
```

该结果相对于公开原始 TD-MPC2 `300k` 的 `271.0 / 316.3 / 191.9`、均值 `259.733`、标准差 `51.407`，已在同预算下显示均值和稳定性优势。为检验延长训练后是否三种子均能进入较理想姿态，新增 `500k` 接续实验：

```text
hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1
```

新增启动脚本：

```text
scripts/launch_hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.sh
```

续训从 `cycle061_target305060` 开始，到 `cycle100_target500099` 结束。三种子均从 `300k` 终点的 `final.full.pt` 恢复；共享规则库、SD checkpoint 和 TVA checkpoint 使用 `cycle060` 产物。由于 `cycle060` 的 TVA terminal prior gate 为 `strong` 通过，续训首个周期显式设置：

```text
initial_tva_terminal_prior_policy = strong
```

该设置用于保持真实接续语义，避免默认 `auto` 在首个续训周期因缺少当前实验内上一周期 gate report 而临时关闭 TVA terminal prior。

启动命令为：

```text
setsid env PYTHONUNBUFFERED=1 bash scripts/launch_hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.sh > logs/launch/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.out 2>&1 < /dev/null &
```

启动后实际主控进程为 `1968796`。检查显示 `cycle061` 的三种子训练进程已分别运行于 GPU1、GPU2、GPU3，训练日志确认均从 `cycle060_target300059/models/final.full.pt` 恢复，恢复状态为 `step=300060`，目标步数为 `305060`。`cycle061/cycle_config.json` 记录 `tva_terminal_prior_policy_this_cycle = strong`，且训练命令继续显示 `rule_prior_stochastic_gate=false`。

## 2026-06-06

### 无随机门控 rich 初始化线从 400k 严格接续至 500k

复核 `resume300k_20260603_r1` 续训结果后确认：该实验命名为 `500k` 续训线，但实际最新 TD-MPC2 full-state checkpoint 已落盘至 `cycle081_target405080`，正式 `eval.csv` 只落盘至 `cycle080_target400079`。为保持正式评估锚点一致，本次继续实验不使用未正式评估的 `405080` checkpoint，而是从 `cycle080_target400079` 严格接续。

接续输入为：

```text
initial_rule_library = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/rules/library_closed_loop_cycle_080_confidence.json
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/sd_explainers/sd_explainer_cycle_080.pt
initial_tva_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/tva/hopper_tva_cycle_080.pt
initial_resume(seed1) = logs/hopper-hop/1/tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079/models/final.full.pt
initial_resume(seed2) = logs/hopper-hop/2/tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079/models/final.full.pt
initial_resume(seed3) = logs/hopper-hop/3/tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079/models/final.full.pt
```

`cycle080` 的 TVA terminal prior gate 为 strong 通过，因此首个续训周期继续显式设置 `initial_tva_terminal_prior_policy=strong`。新实验独立于既有 `resume300k` 文件系统，输出目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume400k_20260606_r2
```

实验前缀为：

```text
tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume400k_20260606_r2
```

首次尝试使用普通 `nohup` 启动的 `resume400k_20260606_r1` 未能稳定脱离工具会话，三种子仅写入 `400100` 的首条训练记录，未生成 checkpoint，故不作为有效实验。正式启动改用 `setsid`，主控 PID 为 `1920918`。启动检查显示主控进程 PPID 已变为 `1`，三种子训练进程分别为 `1920920`、`1920921`、`1920922`，运行于 GPU1、GPU2、GPU3。训练日志确认三条线均从 `step=400080, episode=800, total_updates=398079, buffer=400800 steps / 800 eps` 恢复，当前周期为 `cycle081_target405080`，最终目标为 `cycle100_target500099`。
