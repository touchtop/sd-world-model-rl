# Methods

## 当前方法草案

本研究线暂定为 `SD Retrospective Attribution and Rule Induction`，目标是使用系统动力学结构态对强化学习轨迹进行反向失败归因，并将跨轨迹、跨周期一致出现的经验模式归纳为可执行规则树。

第一阶段采用离线流程：

1. TD-MPC2 独立训练并落盘轨迹；
2. 周期性收集好轨迹、差轨迹和随机轨迹；
3. 第一版先复用已落盘 trace CSV 中的 `SD` 派生结构态字段，后续再使用该周期轨迹拟合独立 `SD` 解释器；
4. 基于结构态事件、相位标注和短视界候选反事实点进行失败归因；
5. 将稳定归因结果压缩为候选规则树；
6. 使用 held-out 轨迹评估规则覆盖率、误杀率、好轨迹保留率和差轨迹预警率。

第一阶段不改变 TD-MPC2 在线训练过程，不直接将规则注入 planner。

## 代码与结果隔离

第一阶段代码位于：

- [experiments/sd_retrospective_rule_induction](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction)

第一阶段结果位于：

- [results/sd_retrospective_rule_induction](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction)

当前已落地的完整第一版入口为 [run_offline_v1.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_offline_v1.py)。该入口顺序执行 manifest 构建、轨迹分层、结构态归因、规则归纳和离线验证，不修改原始日志、checkpoint 或 TD-MPC2 训练代码。

后续实现遵循单一职责拆分：

1. manifest 构建；
2. 轨迹分层；
3. 失败归因；
4. 规则归纳；
5. 规则离线验证；
6. 独立 `SD` 解释器拟合。

在规则经过离线验证前，不新增在线训练脚本；若未来需要接入训练，应另建 `run_hopper_v6_*` 系列脚本，并使规则相关开关默认关闭。

## 第一版归因特征

第一版归因不重新训练模型，而是基于逐帧 trace 中已经保存的结构态字段构造事件特征：

1. `unsupported_high_torque_frac`：动作幅度较高，但 `full_foot_support` 与 `foot_grounded` 均不足，表示无充分支撑下的高幅动作；
2. `landing_unprepared_frac`：处于下落阶段且后续短视界未形成足底支撑，表示落地准备不足；
3. `rotation_risk_frac`：`rotation_penalty` 或 `pitch_rate` 较高，表示姿态旋转风险；
4. `prone_or_rear_trap_frac`：`grounded_prone` 或 `grounded_rear` 高，表示俯伏或后仰接地陷阱；
5. `prepared_support_frac`：`full_foot_support` 与 `grounded_prepared / grounded_upright` 同时较高，表示可用于推进的准备支撑；
6. `effective_drive_frac`：正向速度、足底支撑与中高动作幅度同时出现，表示有效推进片段。

规则归纳暂采用单条件决策列表。该形式并非最终规则树结构，而是第一版用于验证信号质量的最小可解释规则族。

## 短视界反事实比较

第一版已加入 [counterfactual_actions.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/counterfactual_actions.py)。该脚本对每条已选轨迹抽取最多 `3` 个关键状态，并在每个状态上比较实际动作与固定动作探测集合。

动作探测集合包括：

1. 实际执行动作；
2. 零动作；
3. 四个动作维度的正负单关节动作，幅度为 `0.35 / 0.70 / 1.00`。

每个动作使用 `DMControlHopperSDModel` rollout `5` 步，并用结构态得分比较短视界结果。该得分奖励足底支撑、接地近直立、准备支撑和正向速度，惩罚后仰/俯伏陷阱、下落和高角速度。该得分只用于离线归因，不替代环境回报，也不接入 TD-MPC2 在线训练。

## 第二版独立解释器

第二版新增 [fit_sd_explainer.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/fit_sd_explainer.py)。该脚本从第一版选中的 trace CSV 中抽取 `(obs, action, next_obs)` 转移，并使用现有 Hopper SD 监督损失拟合独立解释器。解释器产物不被 TD-MPC2 自动读取，只供离线反事实分析使用。

当前第二版默认口径：

1. `max_transitions=50000`；
2. `epochs=3`；
3. `batch_size=512`；
4. `lr=5e-4`；
5. `use_transition_residual=true`；
6. `transition_residual_scale=1.0`；
7. episode-level `90% / 10%` 训练验证切分。

解释器拟合后，`counterfactual_actions.py` 通过 `--model-checkpoint` 加载该解释器，并重新生成短视界反事实结果。

## TVA 审计型接入

为避免将 Hopper 特定姿态指标硬编码为规则价值标准，后续规则价值评估将逐步引入 `TVA`（Terminal Value Alignment）信号。第一版接入只作为旁路审计，不改变在线训练、不改变 planner 动作排序、不改变现有 confidence 更新。

当前实现复用 [scripts/run_dmcontrol_hopper.py](/home/data/round/code/sd_world_model_rl/scripts/run_dmcontrol_hopper.py) 中已经存在的 `HopperValueNet`、`build_tva_targets` 与 `train_value_net`，仅在规则归纳目录中新增薄封装：

1. [train_hopper_tva_explainer.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/train_hopper_tva_explainer.py)：在每个总结周期中，使用该周期选中轨迹和已拟合的 SD explainer 构造 TVA 目标，并从上一周期 TVA checkpoint 接续训练价值网络；
2. [audit_hopper_tva_counterfactuals.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_counterfactuals.py)：对既有 SD 反事实动作集合进行 TVA 终端价值评分，输出 `tva_counterfactual_audit.json/md`，供后续比较当前结构态启发式分数与 TVA 分数的一致性。

闭环入口新增 `--enable-tva-audit` 开关，默认关闭。启用后，周期流程变为：

```text
TD-MPC2 周期训练
-> replay trace 导出
-> SD explainer 接续拟合
-> TVA value net 接续训练
-> SD 反事实动作分析
-> TVA 反事实旁路审计
-> 原规则归纳与 confidence 更新
```

`TVA` 审计默认最多评分 `300` 条反事实记录，可通过 `--tva-audit-max-records 0` 显式执行全量审计。该限制只控制审计成本，不影响 TVA 训练样本数；TVA 训练样本数由 `--tva-max-transitions` 控制，默认 `50000`。审计抽样默认采用按 label 分层的 `stratified` 模式；为提高规则覆盖，可显式设置 `--tva-audit-sample-mode label_event`，按 `label × event` 分层抽样。该模式仍只改变 TVA 审计覆盖，不改变 SD 反事实生成、规则归纳或在线训练。

TVA 第三层接入采用 shadow update，而不是在线更新。新增 [apply_tva_shadow_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_shadow_update.py)，读取既有规则库与 `tva_counterfactual_audit.json`，复用 [audit_hopper_v8_widefunnel_rule_effects.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py) 中的状态匹配与 selected-gate 判定，判断每条规则在 TVA 审计样本中是否得到动作族层面的支持。

具体口径为：`audit_hopper_tva_counterfactuals.py` 现在为每个 probe action 写出 `family` 与完整 `probe_actions` 排序；shadow update 将同一动作族的不同幅度合并为该 family 的最高 TVA 值。若规则动作族在触发状态中进入 family top-k 且其 TVA 值高于实际动作，则记为正向证据；若相反动作族进入 family top-k 且优于规则动作族，则记为负向证据。脚本据此只生成 `shadow_confidence`、`shadow_confidence_status`、`shadow_update_reason` 与审计组件，不覆盖原有 `confidence`、`status` 或 `effective_prior_scale`。

经滞后验证与 mismatch 诊断后，TVA shadow update 已改为 gated 口径。默认门控包括：每条规则至少 `20` 个 TVA selected-gate 命中；若提供同周期 rule-effect audit，则 TVA/effect 命中比例不低于 `0.003`；增强和削弱均要求 best-family entropy 不高于 `0.65`，避免分散的最优动作族被解释为稳定规则价值；增强要求规则动作族支持比例至少比相反动作族高 `0.05`，且平均 target-opposite margin 不低于 `0.50`；削弱要求相反动作族支持比例至少比规则动作族高 `0.05`，且平均 opposite-target margin 不低于 `0.50`。对应参数为 `--min-hits`、`--min-effect-hit-ratio`、`--max-best-family-entropy`、`--min-directional-rate-margin` 与 `--target-opposite-margin`。该 gated 版本仍为审计/影子层，不进入在线 confidence 更新。

闭环入口新增 `--enable-tva-shadow-update`，默认关闭，且只有在 `--enable-tva-audit` 打开时才会写出 `reports/tva_shadow_update.json/md`。当前默认将缺席证据和低命中证据设为不衰减，因为 `TVA` 反事实审计通常是分层抽样样本，未命中不能等价解释为规则长期无效。后续若改为全量周期审计，可显式调整 `--absence-decay` 与 `--low-hit-decay`。

为比较 TVA shadow 信号与现有规则效应审计的关系，新增 [summarize_tva_shadow_validation.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/summarize_tva_shadow_validation.py)。该脚本读取多周期 `tva_shadow_update.json` 与对应 `rule_effect_audit.json`，输出每周期增强、削弱、不变规则数量，增强/削弱组在 rule-effect 审计中的一致率、有害率、高回报命中率，以及跨周期持续增强或持续削弱的规则列表。该汇总仍属于审计层，不作为在线规则更新依据。

为进一步检查 TVA shadow 是否具备跨周期预测力，新增 [audit_tva_shadow_lag_effects.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_lag_effects.py)。该脚本以某一周期的 shadow update 为锚点，读取后续若干周期的 `rule_effect_audit.json`，比较 strengthened、weakened 与 unchanged 组在后续窗口中的方向一致率、有害率、高回报命中率和好状态命中率。该分析只检验预测关联，不检验因果收益，因为 shadow update 并未在线启用。

为解释 TVA shadow 与 rule-effect 指标不一致的来源，新增 [audit_tva_shadow_mismatches.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_mismatches.py)。该脚本定位两类规则：一类是 TVA shadow 削弱但 rule-effect 方向一致率较高，另一类是 TVA shadow 增强但 rule-effect 方向有害率较高。随后脚本回到 TVA counterfactual records，重建实际动作族、TVA 最优动作族、规则动作族和相反动作族之间的关系，并报告 TVA 样本命中数与 rule-effect 全量命中数的比例。该诊断用于区分三类问题：抽样覆盖不足、实际动作同向但 TVA 认为相反动作更优、TVA 最优动作族分布过散。

为进一步避免仅依赖 TVA 反事实值或 rule-effect 的动作同向统计，新增 [audit_tva_shadow_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_shadow_consequences.py)。该脚本读取 TVA shadow update 与 TVA counterfactual audit，回到每条记录的真实轨迹 `trace_csv + step`，复用同一套状态匹配、selected-gate 和动作族方向判定，并比较规则动作族同向、相反、中性动作在后续短窗口中的真实结构后果。后果指标包括 continuation score、支撑接触、旋转控制、跌落、碰撞和窗口奖励等，汇总为只读的 `outcome_score`。对于被增强规则，期望同向动作的后果优于相反动作；对于被削弱规则，期望相反动作的后果优于同向动作。该分析仍不是因果估计，因为实际动作不是随机分配的，但它可作为 TVA live update 前的后果层筛查。

新增 [apply_tva_consequence_gate.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_tva_consequence_gate.py)，将上述后果层审计作为 TVA shadow 的后处理门控。该脚本读取原 shadow update 和 consequence audit，只生成新的 consequence-gated shadow 报告；未通过后果门控的 shadow 改动会回退到原始 confidence，并标记为 `tva_shadow_rejected_by_consequence_gate`。默认门槛为：期望方向与相反方向均至少 `10` 个真实轨迹样本，且期望方向的 `outcome_score` 均值不低于相反方向。该步骤仍不修改在线规则库。

闭环入口新增 `--enable-tva-consequence-gate`，默认关闭，且依赖 `--enable-tva-audit` 与 `--enable-tva-shadow-update`。启用后，每个周期在 `tva_shadow_update.json/md` 之后额外写出 `reports/tva_shadow_consequence_audit.json/md` 与 `reports/tva_shadow_consequence_gated_update.json/md`。该输出只用于审计与后续分析，闭环训练返回给下一周期的规则库路径仍为原 confidence 规则库。相关参数包括 `--tva-consequence-horizon`、`--tva-consequence-action-threshold`、`--tva-consequence-min-outcome-margin` 与 `--tva-consequence-min-relation-count`。

为直接验证 TVA 是否适合作为 SD 推演的终端补偿，新增 [audit_hopper_tva_terminal_compensation.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_terminal_compensation.py)。该脚本对真实轨迹中的采样起点 `s_t`，使用实际动作序列让 SD 推演 `h` 步，得到 `s'_{t+h}`，并比较 `TVA(s'_{t+h})` 与真实轨迹从 `t+h` 开始的折扣剩余目标。同时，脚本也报告 `TVA(s_{t+h})` 与同一目标的误差，作为价值网络在真实终端状态上的上界参照。默认报告 `MAE`、`RMSE`、baseline RMSE、Pearson、Spearman 和 `R2`，用于区分 TVA 拟合误差与 SD rollout 分布误差。该审计只读 checkpoint 和轨迹，不改变规则库或训练过程。

### TVA 终端先验正式接入

在终端补偿审计显示 TVA 具备基本排序信号后，在线 prior 新增 `HopperTVATerminalPrior`。该组件不修改 TD-MPC2 网络结构，也不改变 replay、reward 或 value 训练目标；它仅在 MPPI 候选动作序列完成原始评分后，使用上一周期 SD explainer 从当前观测和候选动作序列推演至短视界终端，再使用上一周期 TVA 对终端结构状态评分。TVA 分数经中心化和归一化后，以低权重加入候选排序：

```text
tva_terminal_prior_enabled = true
tva_terminal_prior_weight = 0.02
tva_terminal_prior_max_bonus = 0.05
tva_terminal_prior_normalize = true
```

该设置表示 TVA 只作为 SD 推演终端补偿的弱先验，用于检验其是否能在在线规划中提供比单动作规则更连续的后果判断。由于 `cycle070` 小样本审计仍显示 `sd_terminal` 的绝对误差不低，正式接入采用低权重、上限裁剪和显式开关，保留完全回滚到原 rule prior 的能力。当前正式实验从 `350k / cycle070` 接续到 `375k / cycle075`，每 `5000` 步继续执行 SD 回归、TVA 训练和规则审计，每 `25000` 步执行 10 episode formal eval。

从头训练时，第一周期尚不存在可用于 SD 推演与 TVA 终端评分的 checkpoint。因此，`run_hopper_v7_multiseed_shared_rules.py` 采用延迟启用策略：若 `--enable-tva-terminal-prior` 已打开但当前周期没有 `sd_checkpoint` 或 `tva_checkpoint`，本周期训练环境写入 `TVA_TERMINAL_PRIOR_ENABLED=false`，训练结束后仍照常执行 SD 拟合与 TVA 训练；从下一周期具备 checkpoint 后，TVA terminal prior 自动进入 MPPI 候选排序。该处理避免为了从头实验人为注入外部 SD/TVA 初始化，也避免第一周期因缺少终端补偿模型而中断。

当前从头 `200k` rule+TVA terminal prior 线使用同一低权重设置：

```text
tva_terminal_prior_enabled = true
tva_terminal_prior_weight = 0.02
tva_terminal_prior_max_bonus = 0.05
tva_terminal_prior_normalize = true
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_eval_episodes = 10
```

该线从空规则库、空 SD 和空 TVA 开始，主要用于与既有 rule-only 历史线比较规则形成速度、规则健康度、formal eval 回报和终点姿态结构。

### `grounded_hop` 诊断化修订

`grounded_hop` 现作为高分 Hopper 姿态审计与报告特征保留，但不再参与核心排他状态划分。具体而言，`grounded_hop` 仍由 `height >= 0.55`、`abs(pitch) <= 1.10`、`vx >= 0.75` 且无身体碰撞定义；但其不再进入 `state_confident`，不再从 `grounded_prone` 中排他扣除，也不再写入 `_dominant_state_ids` 的状态编号。

该修订的研究意图是区分两类问题：一是高分 hop 轨迹的姿态报告不应被旧 `grounded_prone` 误导；二是早期低质量轨迹中的恢复语义不应被从高分轨迹校准而来的中间标签过早改写。后续从头验证线采用同一规则、SD、TVA 门控协议，仅将 `grounded_hop` 降级为诊断特征，以观察早期规则相关性和 formal eval 是否恢复到上一条高分线的趋势。

对应实验线为：

```text
tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1
```

该线总预算为 `100k`，周期为 `5k`，每 `25k` 执行 `10 episode` formal eval，并继续进行 SD 拟合、反事实归因、规则更新、TVA 训练和 gated TVA terminal prior 审计。

### TVA-kept shadow eval-only 检查

为检验 consequence-gated shadow 候选是否会对显式 planner prior 产生可观察影响，新增两个 eval-only 工具：

```text
experiments/sd_retrospective_rule_induction/build_tva_kept_shadow_library.py
experiments/sd_retrospective_rule_induction/eval_hopper_tva_kept_shadow_315k.py
```

`build_tva_kept_shadow_library.py` 读取原 confidence 规则库与 `tva_shadow_consequence_gated_update.json`，只对 consequence gate 保留且 `shadow_confidence` 与原 `confidence` 不同的规则写入 eval-only 副本。写入方式为将该规则的 `confidence` 和 `effective_prior_scale` 改为 `shadow_confidence`，同时记录 `confidence_before_tva_kept_shadow_eval`、`tva_kept_shadow_delta` 和 gate 元数据。原始规则库不被修改。

`eval_hopper_tva_kept_shadow_315k.py` 复用 [eval_hopper_v7_rule_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py)，在同一 `315k` checkpoint 上比较三种模式：

```text
rule_off:
  rule_prior_enabled = false

original:
  rule_prior_enabled = true
  rule_prior_library = cycle063 confidence library
  rule_prior_allowed_statuses = confidence_active

tva_kept_shadow:
  rule_prior_enabled = true
  rule_prior_library = eval-only TVA-kept shadow library
  rule_prior_allowed_statuses = confidence_active
```

三种模式均使用 `eval_episodes=10`、`eval_action_mode=eval`、`penalty_only`、`rule_prior_weight=0.05`、`rule_prior_opposite_penalty=0.01`、`rule_prior_preferred_cap=0.70`、`rule_prior_cap_penalty=0.01`。该检查不训练策略，不更新 SD 或 TVA，只检验被 TVA consequence gate 保留的 shadow 削弱在推理期显式 planner 排序中是否产生可观察差异。

### 能力入口追踪

为避免只从最终高质量轨迹中提取“维持规则”，新增离线能力入口追踪工具 [trace_hopper_ability_entry_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/trace_hopper_ability_entry_segments.py)。该工具从多种子 retrospective train traces 中自动识别每个种子首次跨越若干回报阈值的周期，并回到具体 episode 的局部窗口，比较成功入口窗口与同期未跨入口种子的失败窗口。

当前第一版使用 `R>1`、`R>10`、`R>50`、`R>100` 作为能力入口阈值；这些阈值不被视为最终价值标准，而是用于定位训练过程中“能力进入 replay 分布”的时间点。每个入口周期默认读取前后各 `1` 个周期，并在每个周期选取最多 `3` 条 episode。局部窗口长度为 `16` 帧，后续后果观察视界为 `30` 帧，动作族阈值为 `0.20`。输出包括：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_trace_report.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_trace_report.md
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_actions.json
```

该工具复用既有 Hopper 结构态适配器和动作族定义，但其主逻辑保持为“阈值入口、成功窗口、同期失败对照、候选动作差异”四步。当前产物只作为离线审计，不修改规则库、不启动训练、不改变在线 planner prior。后续若将入口候选转化为规则，应继续经过 SD/TVA 反事实推演、跨周期后果验证和在线 shadow 审计。

为执行入口候选的第一层反事实筛查，新增 [audit_hopper_ability_entry_counterfactuals.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_counterfactuals.py)。该脚本读取 `entry_candidate_segments.json`，在每个窗口 anchor 上加载同周期 SD explainer 与 TVA value net，并使用固定动作探针集合进行 `5` 步短视界 rollout。探针集合与既有短视界反事实脚本一致，包括真实动作、零动作以及四个动作维度在 `0.35 / 0.70 / 1.00` 幅度下的正负单关节动作。

该审计同时报告两类支持信号：一是 SD 结构态短视界 desirability 分数，二是 TVA 终端价值分数。每个窗口的目标动作族暂取入口窗口内的 `dominant_action_family`，审计其是否进入 SD/TVA 的 top-3 动作族，或是否成为最佳动作族。该设计只检验“观察到的入口动作族是否被同周期模型支持”，不把结果直接写入规则库。

当前第一版输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit.md
```

由于同一高质量入口 episode 可同时服务于多个阈值，例如 `R>50` 与 `R>100`，报告中允许存在同一 trace 在不同阈值下重复出现。该重复表示不同能力入口定义共享同一实际轨迹证据，而不是数据读取错误。后续若转为规则归纳，应按 `trace_csv + anchor_step + threshold` 或按阶段入口定义去重。

### 效果空间规则原型

为避免重新引入难以控制的动作序列式规则，新增 [audit_hopper_ability_entry_effect_space.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_effect_space.py)。该脚本将入口规则从“偏好某个动作族”改写为“偏好使短视界后果向量朝入口方向移动的动作”。规则对象因此仍可保持简洁形式：

```text
context_key -> prefer_effect_direction
```

第一版效果向量 `z` 由同周期 SD rollout 和 TVA value net 共同构造，包含 `tva`、结构态 desirability、前向速度、垂直速度、躯干高度、支撑接触、全足支撑、直立/准备支撑、hop、非俯伏/后仰、非下落、旋转控制和俯仰控制等低维分量。对每个入口事件，方向向量定义为：

```text
v_entry = mean(delta_z | entry_success_actual)
        - mean(delta_z | matched_failure_actual)
```

随后，候选动作不再按动作族频率排序，而是按 `dot(delta_z_candidate, normalize(v_entry))` 排序。该处理把阶段过程的复杂性放入效果空间映射中，而不是放入规则维护结构中。当前第一版仅做离线验证，不修改规则库、不改变 planner。

输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_space_audit.md
```

该原型的评价指标包括成功入口窗口与匹配失败窗口的 actual-action alignment margin，以及 `AUC(success alignment > failure alignment)`。若某个 `context_key -> prefer_effect_direction` 在多周期中维持正 margin 和高 AUC，才可进入后续 shadow 规则候选。

为检查方向是否只是组内拟合结果，新增 [audit_hopper_ability_entry_effect_space_stability.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_effect_space_stability.py)。该脚本重新计算入口窗口的 `delta_z`，并执行三类稳定性检查：

1. `same_threshold_transfer`：从某个阈值入口周期学习方向，迁移到同阈值的其他入口周期；
2. `leave_one_global`：从除目标组以外的所有组学习方向，再评估目标组；
3. `direction_cosine`：比较不同入口组方向向量之间的余弦相似度。

该检查用于决定 `prefer_effect_direction` 是否可进入 shadow prior。当前门槛暂按审计解释口径设定：若迁移后保持正 alignment margin，且 `AUC(success alignment > failure alignment) >= 0.55`，则视为具备初步稳定信号；若同阈值方向余弦为负或 leave-one 结果退化，则应继续改进效果空间或上下文划分。

为避免效果空间路线继续复杂化，新增 [audit_hopper_ability_entry_minimal_stable_effect.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_minimal_stable_effect.py)。该脚本只加入两个最小修订：一是按周期对 `delta_z` 做 robust median/IQR 归一化，二是将入口上下文粗分为 `low_entry` 与 `high_entry` 两类，其中 `low_entry` 对应 `threshold <= 10`，`high_entry` 对应 `threshold >= 50`。

该脚本采用严格 stable/unstable 输出。某一 tier 只有在全部跨周期 transfer 均为正 margin、全部 transfer 的 AUC 均不低于 `0.60`，且该 tier 内方向余弦均值为正时，才标记为 `stable`。若未通过，则不生成效果空间 prior，相关问题继续交还 TD-MPC2 主体训练或仅保留为离线诊断。

在最小稳定审计未通过后，进一步新增 [audit_hopper_ability_entry_effect_space_discriminant.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_effect_space_discriminant.py)，用于测试更强但仍可控的数学方向估计。该脚本在相同数据口径下比较三种方法：`mean_diff`、正则化 Fisher/LDA 方向 `ridge_lda`，以及带跨组符号一致性筛选的 `sign_consensus_ridge_lda`。所有方法仍输出 `context -> prefer_effect_direction` 的离线稳定性审计，不生成在线 prior。

`ridge_lda` 使用如下方向：

```text
w = (Sigma_pooled + lambda I)^(-1) (mu_success - mu_failure)
```

其中 `Sigma_pooled` 为成功与失败窗口的组内协方差，`lambda` 采用按协方差 trace 缩放的 ridge 项。`sign_consensus_ridge_lda` 先检查不同训练入口组中每个效果维度的 `success - failure` 符号是否一致，再只保留一致性达到门槛的维度进入 LDA。该步骤用于判断是否存在比简单均值差更稳定的效果方向；若仍不通过严格迁移门槛，则停止向在线 prior 推进。

## 第二版状态-动作规则

第二版新增 [induce_state_action_rules_v2.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/induce_state_action_rules_v2.py)。规则形式为：

```text
state_event -> prefer_action_family
```

其中 `state_event` 来自第一版事件归因，例如 `unsupported_high_torque`、`rotation_risk`、`prone_or_rear_trap`；`action_family` 来自短视界反事实中的最优动作族，例如 `hip_pos`、`knee_neg`。规则记录支持样本数、事件覆盖率、平均反事实优势、实际动作平均排名和置信度。

## 第三版规则一致性审查

第三版新增 [analyze_rule_consistency.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/analyze_rule_consistency.py)。该脚本不生成新的反事实，而是审查第二版候选规则是否具备作为在线软先验候选的最低可靠性。

审查指标包括：

1. `seeds_covered`：规则支持样本覆盖的 seed 数；
2. `cycles_covered`：规则支持样本覆盖的训练周期数；
3. `good_false_prefer_rate`：规则偏好的动作族在好轨迹中出现的比例，用于衡量误伤风险；
4. `same_event_conflict_entropy`：同一状态事件下不同最优动作族分布的归一化熵，用于判断是否需要细分状态条件；
5. `mean_advantage`：该规则对应样本中最佳反事实动作相对实际动作的平均结构态优势。

当前默认可信候选门槛为：支持数不少于 `20`，覆盖至少 `2` 个 seed，平均反事实优势不少于 `0.10`，好轨迹误伤率不超过 `0.25`。

## 第四版子状态与回放

第四版新增两个离线验证步骤。

第一，子状态切分。粗粒度事件如 `unsupported_high_torque` 会继续按以下结构态离散化：

1. `pitch_rate` 符号：`omega_pos / omega_neg / omega_mid`；
2. `vx` 区间：`vx_low / vx_mid / vx_high`；
3. `full_foot_support` 区间：`support_none / support_low / support_some`；
4. 主导相位：`falling / rear / prone / upright / other`。

子状态规则形式为：

```text
state_event|omega_bin|vx_bin|support_bin|phase -> prefer_action_family
```

第二，离线回放。对 v3 可信规则在已选 trace 的逐帧状态上进行打分，统计实际动作与规则推荐动作族是否对齐、相反或无关，并按好轨迹、差轨迹和中间轨迹分组。该步骤用于判断规则若作为在线软先验，是否可能误伤好轨迹中的关键动作。

第四版仍不接入在线训练。若粗规则在回放中命中大量好轨迹，应降级为 `needs_split`，优先保留更细的子状态候选。

## 第五版规则库状态更新

第五版不新增反事实样本，而是对 v4 子状态规则进行逐帧回放审查，并把审查结论写回版本化规则库。该步骤的研究意图是区分“在关键状态反事实中看起来有效”的候选规则与“在完整轨迹分布中主要命中失败片段”的候选规则。

子状态回放统计以下指标：

1. `hits`：规则在逐帧 trace 中命中的状态数；
2. `good_hit_rate`：命中状态中来自好轨迹的比例，用于估计误伤成功行为的风险；
3. `bad_hit_rate`：命中状态中来自差轨迹的比例，用于估计规则是否主要定位失败片段；
4. `alignment_rate`：实际动作已落在规则推荐动作族中的比例；
5. `opposed_rate`：实际动作与规则推荐动作族方向相反的比例；
6. `seeds_covered` 与 `cycles_covered`：规则命中是否跨 seed、跨训练周期，而不是只来自单次训练偶然样本。

第五版规则库采用以下状态：

1. `candidate`：已有局部反事实支持，但逐帧回放证据尚不足；
2. `replay_validated`：逐帧回放显示规则主要命中差轨迹，跨 seed 覆盖较好，可进入在线软先验候选池；
3. `needs_split`：粗规则在完整轨迹中覆盖过宽，尤其命中较多好轨迹片段，必须继续细分状态条件；
4. `rejected`：离线回放未能支持规则，或误伤风险过高。

当前 v5 默认升级条件为：逐帧命中数不少于 `500`，好轨迹命中率不高于 `0.10`，差轨迹命中率不低于 `0.60`，且覆盖至少 `2` 个 seed。未满足条件的子状态规则保持 `candidate`，而不是被删除；后续周期若提供更多证据，可继续修订其状态。

v5 仍不直接接入在线训练。若后续执行在线 smoke，允许进入软干预候选池的仅限 `replay_validated` 规则，并且干预形式应限制为低强度动作候选重加权，而不是硬替换动作。

## 第六版在线软先验

第六版将 v5 规则库中通过回放审查的子状态规则作为在线软先验候选，验证“离线经验总结能否温和影响在线探索”。该版本只做 smoke，不改变 TD-MPC2 的训练目标、优化器、replay 机制、环境奖励或模型结构。

规则接入采用运行时补丁方式完成。独立训练入口在构造原始 TD-MPC2 agent 后，根据 `rule_prior_enabled` 显式开关安装规划补丁；开关关闭时，训练路径与原始 TD-MPC2 入口保持一致。该设计用于保证 v6 结果可以与 v5 主线训练和无规则对照清晰区分。

当前在线候选池只读取规则库中同时满足以下条件的规则：

1. `status == replay_validated`；
2. `source == v4_substate`；
3. `intervention_type == soft_prefer_action_family`。

当前满足条件的规则为：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos
```

在线重加权过程如下。首先，使用 `DMControlHopperSDModel` 从当前观测中计算结构态特征，并离散化为 `pitch_rate` 符号、`vx` 区间、`full_foot_support` 区间和主导相位。其次，在 MPPI 候选动作序列估值后，仅对当前子状态命中规则、且第一步动作为高力矩候选的序列添加小幅偏置。若候选动作在规则指定关节方向上为正向，则增加 `rule_prior_weight * relu(action_dim)`；若同一关节方向相反，则扣除 `rule_prior_opposite_penalty * relu(-action_dim)`。偏置被限制在 `[-rule_prior_max_bonus, rule_prior_max_bonus]` 内。

默认参数为：

1. `rule_prior_enabled=false`；
2. `rule_prior_weight=0.05`；
3. `rule_prior_opposite_penalty=0.02`；
4. `rule_prior_max_bonus=0.10`。

该干预属于候选动作重加权，而不是硬规则控制。其预期作用是轻微提高已验证动作族在特定失败子状态下进入 elite 集合的概率，同时保留 MPPI 和策略网络原有的探索能力。

## 第六点一版规则干预审查

第六版 quick smoke 显示，正向 `prefer` 模式会把目标子状态下的动作强烈推向饱和 `hip_pos`，但该行为没有带来收益。因此第六点一版将在线规则干预拆分为两种模式：

1. `prefer`：保留第六版行为，对规则动作族方向给小幅正向价值偏置，并对相反方向给小幅惩罚；
2. `penalty_only`：不再给正向奖励，只惩罚相反方向动作和同向过饱和动作。

`penalty_only` 的研究含义是把规则从“动作建议”降级为“风险筛选”。该模式承认离线规则能够定位失败片段，但暂不假设其推荐动作族可以直接改善在线控制。形式上，对目标子状态命中的第一步候选动作，若其在规则关节上与推荐方向相反，则施加 `rule_prior_opposite_penalty`；若其与推荐方向相同但幅度超过 `rule_prior_preferred_cap`，则施加 `rule_prior_cap_penalty`。该惩罚仍被限制在 `rule_prior_max_bonus` 范围内。

第六点一版新增离线审查脚本，用已落盘 eval trace 统计每种干预模式会作用于哪些实际动作。该审查不声称能够替代在线训练，只用于在重新运行在线 smoke 前排除明显会造成动作饱和的规则形式。

## 规则产出原则

本研究线不要求每个训练周期或每轮离线分析都生成新规则。规则归纳采用高门槛原则，允许少量产出甚至空产出。每轮经验总结的有效结果可以分为五类：

1. 新增规则：跨 seed、跨周期稳定，反事实优势明显，且好轨迹误伤率低；
2. 强化已有规则：未产生新规则，但已有规则获得更多支持样本、更低误伤率或更高反事实优势；
3. 修订或拆分规则：粗粒度事件下动作偏好冲突较高，需要继续细分状态条件；
4. 否决规则：候选规则在好轨迹中频繁出现，不能作为训练干预规则；
5. 保持观测：当前周期未提供足够证据改变规则库。

因此，规则库的增长应是稀疏的、保守的和可审计的。若规则未来可能接入在线训练，错误规则的风险高于没有规则；规则归纳模块宁可不产出，也不应把不稳定相关性包装成任务知识。

## 闭环方法论边界

下一阶段在线闭环测试验证的是解耦的周期总结机制，而不是 SD 与 TD-MPC2 的端到端协同嵌入。TD-MPC2 主学习线持续训练策略、价值函数和自身世界模型；SD 解释器线只在周期总结阶段使用落盘轨迹接续训练，并用于失败归因、反事实比较和规则提炼。两条学习线通过版本化规则库交互，而不共享在线梯度、优化器状态或 replay 更新。

闭环中 SD 解释器采用接续训练，而不是每轮从全新模型开始。第一个周期可从已有离线解释器 checkpoint 或结构模型初始化；后续周期从上一周期的 `sd_explainer_cycle_*.pt` 加载，并使用新周期轨迹或累计轨迹缓冲继续拟合。该设计使 SD 表示能够积累经验总结能力，同时避免把 SD 训练混入 TD-MPC2 在线主体。

每个总结周期最多提炼 `1` 条最显著规则。若该周期缺少稳定、一致且回放支持的候选规则，则不新增规则。通过回放审查的规则进入下一周期规则库；未通过的候选只记录为证据，不参与在线规则先验。

## 第七版在线闭环

第七版实现第一版在线闭环 runner：[run_hopper_v7_closed_loop.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py)。该入口把周期训练和周期总结串联起来，但仍遵循上述方法论边界。周期长度由 `--cycle-steps` 控制，可以设为 `10000`、`20000` 或其他预算，不作为方法假设的一部分。

每个周期执行以下步骤：

1. 使用纯 TD-MPC2 入口训练到本周期目标步数，强制 `WORLD_MODEL_KIND=tdmpc2`、`SD_ONLINE_SUPERVISION_ENABLED=false`；
2. 若不是第一周期，则从上一周期 `final.full.pt` 恢复 TD-MPC2 训练状态，并加载上一周期闭环规则库；
3. 用该周期 run 名称构建只读 manifest，并选择好轨迹、差轨迹和随机轨迹；
4. 从上一周期 SD 解释器 checkpoint 接续拟合本周期 SD 解释器；
5. 用本周期 SD 解释器执行短视界反事实、规则归纳、一致性审查、子状态拆分和回放验证；
6. 调用 [closed_loop_rule_update.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py)，从本周期规则库中最多选择 `1` 条满足准入条件的规则合并到下一周期规则库。

第七版复用 v6 的运行时规则先验补丁，默认在线干预形式为 `penalty_only`。第一周期默认不启用规则先验，用于生成初始周期轨迹；可通过 `--enable-rule-prior-cycle1` 显式改变该行为。

## 第七版 10k x 5 结果解释口径

第一轮闭环实验采用 `cycle_steps=10000`、`cycles=5`，总预算为 `50000` 环境步。该设置的作用是验证周期总结机制能否在较短周期内稳定执行，而不是把 `10k` 作为 Hopper 的充分学习周期假设。

结果解释采用三层口径。

第一，基础设施层面检查闭环是否完整运行，包括 TD-MPC2 从上一周期 `final.full.pt` 恢复、SD 从上一周期 `sd_explainer_cycle_*.pt` 接续训练、周期规则库是否传递到下一周期，以及规则先验是否只通过 `penalty_only` 形式进入规划候选排序。

第二，行为信号层面同时读取官方评估、训练 episode 和阶段诊断。由于 Hopper 奖励稀疏且过程阶段化明显，官方评估回报为零不等于训练中完全没有有效信号；但训练 episode 中的偶发高回报也不能直接解释为稳定策略改进。解释时需同时关注 `final_rootx`、`max_rootx`、`max_full_foot_support`、`full_foot_support_frac`、`backward_speed_frac` 和 `rotation_penalty_frac`。

第三，规则归纳层面把“未新增规则”视为合法输出。每周期最多新增 `1` 条规则只是上限，不是产出要求。若候选规则未通过一致性审查、子状态拆分或回放验证，则规则库保持不变。该策略服务于可审计性和在线安全性，避免把偶然轨迹相关性写入后续探索。

第一轮 `10k x 5` 的结果表明，闭环执行链路成立，SD 接续训练成立，但当前保守阈值下未产生新规则，官方评估收益也未稳定形成。后续方法改进应优先补充候选规则拒绝原因统计和周期轨迹覆盖，而不是直接降低规则准入门槛或增强既有单条规则权重。

## 规则未新增漏斗审计方法

闭环实验后新增规则未新增漏斗审计，用于区分三类不同问题：

1. 周期总结是否形成候选规则；
2. 候选规则是否通过一致性、子状态和回放验证；
3. 闭环更新器是否从本周期规则库中选择上线规则。

审计脚本为 [audit_closed_loop_rule_funnel.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_closed_loop_rule_funnel.py)。该脚本不重新训练模型，只读取每个 `cycle_*` 下已经落盘的 manifest、反事实记录、v2 规则、v3 一致性审查、v4 子状态规则、v5 回放验证和闭环规则库更新结果。

审计指标包括：

1. 轨迹供给：选中 episode 数、good/bad 标签分布、seed 覆盖；
2. 反事实供给：状态记录数、事件分布、事件平均优势；
3. v2 粗规则漏斗：同一 `event -> action_family` 的最大支持数、最大平均优势，以及是否达到 `min_support=8` 与 `min_advantage=0.10`；
4. v4 子状态漏斗：同一 `substate -> action_family` 的最大支持数、最大平均优势、good share，以及是否达到 `min_support=5`、`min_advantage=0.12` 与 `max_good_share=0.25`；
5. v5 和闭环更新：本周期 `library_v5` 是否产生规则、是否存在 `replay_validated` 候选、闭环更新器是否执行 `added_new_rule` 或 `updated_existing_rule`。

第七版 `10k x 5` 审计显示，所有周期均在 v2 和 v4 规则生成之前被支持样本不足卡住。第 2 至第 5 周期每轮只有 `3` 条官方评估 episode 和 `9` 条反事实状态记录，且全部为 bad 标签。该设置适合定位失败状态，但不足以支持稳定规则归纳。

后续闭环方法应把“周期内候选”和“可上线规则”分离。单周期、单 seed 下可产生 `cycle_candidate` 或 `cycle_supported`，用于积累证据和诊断方向；`replay_validated` 则保留为跨周期或跨 seed 证据充分后的上线状态。该调整不等于降低在线规则准入门槛，而是避免在证据积累阶段把所有未达最终门槛的近邻候选直接丢失。

## 第八版成功延续与阶段衔接规则

第八版规则提取方法不替换第七版失败恢复规则，而是在同一闭环中新增“成功延续”和“阶段衔接”规则通道。该修订来自 `200k` 最终评估轨迹审查：当前规则先验已经启用并覆盖约一部分恢复状态，但最终较理想姿态片段主要落在 `omega_mid / vx_high / support_none / falling / rotation_risk=0` 等子状态上，现有规则库未覆盖这些片段。

第八版方法将规则分为三类：

1. `recovery`：失败恢复规则，沿用当前 `rear / prone / falling / rotation_risk / unsupported_high_torque` 口径；
2. `continue`：成功延续规则，从局部好姿态片段中提取，目标是保持已有姿态优势；
3. `transition`：阶段衔接规则，从成功片段后的退化窗口中提取，目标是解释和改善飞行、落地、支撑、再起跳之间的转换。

第八版采用“通用引擎 + Hopper 适配器”边界。通用引擎只定义 `StateSchema`、`PhaseDetector`、`QualityMetrics`、`ActionFamilyMapper`、`SegmentMiner`、`CounterfactualEvaluator`、`RuleInducer`、`RuleValidator` 和 `RuleLibraryUpdater`。Hopper 适配器负责将 `pitch`、`pitch_rate`、`vx`、`torso_z`、`full_foot_support`、`falling`、`rear`、`prone`、`waist/hip/knee/ankle` 等任务语义映射到通用接口。

初始 Hopper 成功片段判据为：

```text
torso_z >= 0.75
abs(pitch) <= 0.45
vx >= 0.80
abs(pitch_rate) <= 3.50
body_collision < 0.50
```

若 `dense_pose_potential >= 0.65` 或单步环境 reward 为正，也可作为候选入口。该判据只用于片段挖掘，不作为在线规则上线条件。成功中断片段定义为：出现局部好姿态后，在短窗口内出现明显旋转风险、躯干高度下降、速度退化、进入 `rear/prone`、落地支撑失败或身体碰撞。

第八版新增规则键为：

```text
rule_role|quality_bin|omega_bin|vx_bin|support_bin|phase|risk_bin -> prefer/avoid action_family
```

其中 `rule_role` 取 `recovery / continue / transition`，`quality_bin` 取 `bad / near_good / good_fragment / transition_gap`，`risk_bin` 取 `rotation_risk / support_risk / collision_risk / stable`。成功延续规则可以生成 `prefer` 或 `avoid`；在线接入时，`avoid` 优先采用 `penalty_only`，`prefer` 需要更高跨 seed 证据后才进入正向加权。

完整设计见 [hopper_v8_success_extension_rule_design.md](/home/data/round/code/sd_world_model_rl/docs/sd_retrospective_rule_induction/hopper_v8_success_extension_rule_design.md)。

第八版第一步已实现片段挖掘脚本：

- [mine_hopper_v8_success_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py)

该脚本只读取 Hopper trace CSV，不训练模型、不修改 TD-MPC2 checkpoint、不写入 `logs/`。默认复核当前 v72 三种子共享规则 `200k` 最终评估轨迹，输出成功候选片段、成功中断片段、阶段衔接缺口、候选原因分布、退化类型分布，以及这些片段是否已被现有 v7 规则库覆盖。

默认运行命令为：

```bash
python experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py --use-default-v72-200k
```

默认输出为：

- [hopper_v8_success_segments_200k_v72.json](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.json)
- [hopper_v8_success_segments_200k_v72.md](/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.md)

## 周期候选规则层

周期候选规则层用于保存单周期内的弱但有诊断价值的规则证据。该层不改变在线动作先验的准入条件；在线先验仍只读取 `status == replay_validated` 的规则。

子状态规则现在分为三类：

1. `cycle_candidate`：达到候选保留阈值，但未达到单周期支持阈值。当前候选保留阈值为 `support >= 2`、`mean_advantage >= 0.02`、`good_share <= 0.50`。
2. `cycle_supported`：达到单周期子状态挖掘阈值，但尚未通过累计回放和跨 seed 审查。当前单周期阈值为 `support >= 5`、`mean_advantage >= 0.12`、`good_share <= 0.25`。
3. `replay_validated`：通过回放验证且满足上线要求的规则。当前要求仍包括足够逐帧命中、低好轨迹命中率、高差轨迹命中率和跨 seed 覆盖。

闭环规则更新器对这三类规则采用不同处理。`replay_validated` 规则仍遵循“每周期最多上线一条”的策略；`cycle_candidate` 和 `cycle_supported` 只作为证据携带，默认每周期最多携带 `3` 条。由于在线规则先验过滤 `replay_validated`，周期候选不会直接影响 TD-MPC2 规划。

该设计的研究意义是把“经验记录”和“在线干预”分离。单周期弱信号被保存下来，便于跨周期累计、合并和审查；但只有累计证据充分后，规则才会进入动作先验。

## 第七点一版闭环方案

第七点一版用于修正第七版暴露出的证据供给问题。周期触发频率改为 `5000` 环境步，但周期总结不再只读取当前周期，而是读取最近若干周期的滚动窗口。推荐第一轮设置为 `cycle_steps=5000`、`cycles=10`、`summary_window_cycles=3`，即总预算仍为 `50000`，每 5k 总结一次，每次总结最多覆盖最近 15k 的官方评估轨迹。

轨迹分层由纯回报分层扩展为阶段感知分层。`stratify_trajectories.py` 新增 `--score-mode` 参数：

1. `reward`：沿用原始 episode reward；
2. `stage`：使用 `final_rootx`、`max_rootx`、`max_full_foot_support`、`full_foot_support_frac`、`grounded_upright_frac`、`grounded_prepared_frac`、`backward_speed_frac`、`rotation_penalty_frac`、`grounded_prone_frac` 和 `grounded_rear_frac` 构造阶段分数；
3. `hybrid`：回报与阶段分数相加。

Hopper 闭环建议使用 `stage` 或 `hybrid`。当官方评估回报全为 `0` 时，阶段分数仍可选出 near-good 轨迹，为失败归因提供对照样本。已在既有第 5 周期官方评估上验证：reward 全为 `0` 的 3 条 episode 中，阶段分层可选出 1 条 bad 和 1 条 good/near-good。

候选规则跨周期累计由 `closed_loop_rule_update.py` 完成。`cycle_candidate` 和 `cycle_supported` 会记录 `evidence_cycles`、`cycles_covered`、`cumulative_support` 和 `cumulative_mean_advantage`。当候选满足默认 `rolling_min_cycles=2`、`rolling_min_support=6`、`rolling_min_advantage=0.05`、`rolling_max_good_share=0.25` 时，可升级为 `rolling_validated`。

在线规则先验默认仍只读取 `replay_validated`。若要在 v7.1 pilot 中允许闭环内部滚动验证规则影响后续探索，可显式设置 `rule_prior_allowed_statuses=replay_validated,rolling_validated`。该设置仍不会读取 `cycle_candidate` 或 `cycle_supported`。

## 第七点一版 5k x 10 pilot 解释口径

第七点一版 pilot 的解释分为三个层次。

第一，可靠性层面检查闭环是否完整运行。必要条件包括：恢复训练后仍能在目标步数触发官方评估，滚动 manifest 能够稳定索引最近若干周期，阶段感知分层在官方回报为零时仍能产生 good/near-good 与 bad 对照，SD 解释器能够从上一周期 checkpoint 接续加载，规则库能够传递到下一周期，在线先验只读取允许状态的规则。

第二，规则层面区分 `no_rule_added` 与“规则库无变化”。`no_rule_added` 仅表示本周期没有新增满足 `replay_validated` 准入条件的规则；若候选规则通过跨周期累计达到 `rolling_validated`，则仍可进入在线先验。第七点一版 pilot 显式允许 `replay_validated,rolling_validated` 进入在线先验，因此在线规则数可在 `no_rule_added` 的同时增加。

第三，性能层面不以单周期官方回报作为唯一判断。Hopper 早期具有奖励稀疏和阶段依赖特征，官方评估为零时仍需记录训练 episode 回报、阶段分数、支撑恢复、前向位移和规则先验命中。但这些信号只能说明存在阶段性学习或归因信号，不能直接解释为稳定任务收益。稳定效果判断需要同预算纯 TD-MPC2 对照或多 seed 复现。

本轮 `5k x 10` pilot 的最终规则库包含 `1` 条 `replay_validated` 与 `3` 条 `rolling_validated` 在线可读规则。该结果被视为闭环归因机制开始产生可上线弱规则，而不是 Hopper 性能已经改善的证据。

## 第七点一版 100k 接续解释口径

`100k` 接续实验沿用第七点一版闭环方案，但从 `cycle010` 的 TD-MPC2 checkpoint、SD 解释器 checkpoint 与规则库恢复，而不是从头开始。该设置用于检验闭环系统的可恢复性和较长预算下的阶段性信号，不改变方法论边界：TD-MPC2 主训练线保持纯 TD-MPC2，SD 解释器只在周期总结阶段接续训练，规则先验仍以 `penalty_only` 方式轻量介入候选动作排序。

该实验的解释应分为三层。

第一，恢复可靠性。若 `cycle011` 至 `cycle020` 能连续触发评估、保存 TD-MPC2 checkpoint、训练 SD 解释器并传递规则库，则说明闭环 runner 已具备中断后接续训练能力。

第二，规则库演化。规则库增长或滚动规则升级说明经验总结机制仍在工作；但同一子状态下出现多个互相不同的偏好动作族时，应优先解释为子状态过粗或证据冲突，而不是规则更完备。

第三，性能判断。官方评估回报仍是稳定任务收益的主要证据。训练 episode 中的非零回报和阶段分数可作为局部行为信号，但不能替代评估策略表现。若训练 episode 出现更高回报而官方评估保持为零，应记录为“训练分布存在阶段性成功片段，但尚未稳定泛化到评估策略”。

`100k` 接续结果显示，闭环机制可以可靠恢复并继续积累规则；训练 episode 在 `90k` 附近出现更强局部信号；但官方评估回报仍为 `0.0`。因此，下一步方法改进不应直接提高规则权重，而应优先进行纯 TD-MPC2 同预算对照、规则冲突感知合并、以及对 prone 子状态的更细粒度拆分。

## 显式规则 eval 对照方法

为区分规则是否只在训练时影响 planner 排序，新增不重训练的 checkpoint eval 对照。该对照加载同一 `final.full.pt`，并分别设置 `rule_prior_enabled=false` 与 `rule_prior_enabled=true`。`rule_on` 使用训练期相同的 `install_rule_prior()`，因此评估的是规则在推理期显式介入 planner 排序的效果，而不是策略网络内化后的效果。

该对照的判读口径如下。

第一，若 `rule_on` 与 `rule_off` 回报和轨迹均无差异，则说明规则补丁没有实际影响推理期动作，需要检查规则匹配、状态离散化、动作门控或 planner 接入位置。

第二，若 `rule_on` 改变命中子状态下的动作方向，但回报仍无改善，则说明规则在推理期显式有效，但规则覆盖、规则准确性或任务阶段链条仍不足；这时不能把失败归因于“eval 时规则未启用”。

第三，若 `rule_on` 有效而 `rule_off` 无效，则说明规则尚未充分内化到策略网络。后续应考虑 actor 规则一致性损失、planner-with-rule 蒸馏或规则惩罚进入训练目标。

100k 显式规则 eval 对照属于第二种情况。`rule_on` 将规则命中动作检查中的相反方向比例从 `40.1%` 降至 `3.5%`，但三条 eval 回报仍为 `0.0`。因此，当前主要问题不是 eval 时规则没有作用，而是显式局部规则不足以完成完整 Hopper 任务，且策略网络本身仍未内化这些规则。

当前实现还需修正一个审计口径：`online_rule_prior_v6.py` 中 `_state_matches()` 只允许 `unsupported_high_torque` 事件命中，导致 `rotation_risk` 规则被加载但不会实际生效。后续多事件规则接入前，应将“规则加载数”和“可实际命中规则数”分开记录。

## 规则信心机制

为处理冲突规则，新增规则信心重放机制。该机制把规则库从静态状态集合扩展为动态证据系统：每条规则维护 `confidence`，每轮总结证据可以增强、削弱或淘汰规则。该机制不要求后来规则直接覆盖旧规则，而是通过同一冲突组内的证据竞争实现渐进替代。

当前冲突组定义为：

```text
conflict_group = substate_key + joint
```

因此，同一子状态、同一关节、相反符号的规则互相削弱，例如 `waist_pos` 与 `waist_neg`；不同关节的规则不互相削弱，例如 `hip_neg` 与 `knee_pos`，因为它们可能共同构成一个有效动作族组合。

信心更新包括三类操作。

第一，正证据增强。同一规则在本周期再次出现时，根据支持数、平均反事实优势和好轨迹占比计算证据强度，并增加 `confidence`。

第二，冲突证据削弱。同一 `conflict_group` 中相反动作族出现时，既有规则按冲突证据强度削弱。若旧规则被削弱至退休阈值以下，而反向规则继续获得支持，则反向规则自然成为新的候选规则。

第三，缺席衰减。某条规则在后续周期长期没有新证据时，`confidence` 按固定比例缓慢衰减，避免早期偶然规则永久保留。

信心状态分为：

```text
confidence_active
confidence_observing
confidence_retired
```

在线动作先验已补充读取 `effective_prior_scale` 或 `confidence` 的能力。旧规则库没有该字段时默认比例为 `1.0`，不改变既有实验；若显式使用 confidence 规则库，则每条规则的实际先验强度为基础权重乘以该信心比例。

在 `cycle001-020` 上重放后，规则库从原先强弱不分的滚动规则转化为 `5` 条 active、`6` 条 observing、`8` 条 retired。典型结果是：`unsupported_high_torque|omega_neg|vx_low|support_none|prone|joint:knee` 中，`knee_pos` 信心升至 `0.611`，而早期 `knee_neg` 被后续证据削弱至 `0.0`；`joint:waist` 中，`waist_neg` 保留 `0.364` 信心，`waist_pos` 被削弱至 `0.0`。这与“矛盾规则先削弱旧规则，旧规则消失后反向规则自然形成”的设计一致。

## 多事件在线规则匹配

在线规则 prior 已从单事件匹配扩展为多事件匹配。当前支持：

1. `unsupported_high_torque`：仍要求子状态 bins 匹配，并且只对首步高力矩候选动作施加规则先验；
2. `rotation_risk`：要求子状态 bins 匹配，并要求当前状态满足 `rotation_penalty >= 0.5` 或 `abs(pitch_rate) >= 3.0`。该事件不再使用高力矩候选门控，而是在旋转风险状态下对候选动作统一施加 confidence 缩放后的轻量惩罚。

该设计的含义是：`unsupported_high_torque` 规则仍被视为“高力矩无支撑时的动作筛选”；`rotation_risk` 规则则被视为“旋转风险状态下的方向性风险筛选”。两者共享相同的动作族偏好和 `penalty_only` 计算，但候选门控不同。

多事件修正后的 confidence eval 显示，`rotation_risk` active 规则已经实际命中，且未产生相反方向动作。但三条 eval 回报仍为 `0.0`，阶段指标与修正前基本一致。因此，多事件匹配修正解决的是实现一致性问题，而不是直接证明规则 prior 已足以解决 Hopper。

## Confidence prior 接续后的 eval 诊断口径

`100k -> 150k` 接续实验表明，Hopper 闭环评估不能继续只依赖 `eval_episodes=3` 的周期内 quick eval。该设置在运行中每轮只测试固定随机种子下的前三条初始状态；当策略处于阶段性恢复但尚未完全稳定时，前三条初始状态可能全部失败，从而把有效 checkpoint 误判为近零。

正式判断 checkpoint 表现时，应至少报告以下指标：

```text
episode_reward_mean
episode_reward_std
episode_reward_min
episode_reward_max
episode_reward_list
```

关键 checkpoint 建议使用 `eval_episodes >= 9`。周期内 quick eval 可保留 `3` 条用于快速检测是否崩溃，但不能作为主要性能结论。对于 Hopper 当前阶段，`3` 条 quick eval 只能视为低成本哨兵信号；若 train episode 出现连续非零回报或阶段指标明显改善，应追加更宽评估。

新增 eval-only 诊断开关：

```text
eval_action_mode=eval
eval_action_mode=train_like
```

`train_like` 不训练、不更新 replay buffer，只强制评估时调用 `agent.act(..., eval_mode=False)`，用于判断 train/eval 差异是否主要来自 planner 动作模式。150k 诊断显示 `eval_action_mode=eval` 的 9-episode 均值为 `64.03`，`train_like` 为 `63.69`，两者接近。因此，本轮严重不一致主要来自 3-episode 统计不足，而不是 eval 动作模式本身。

规则 prior 的效果应单独审计。同一 `150k final.full.pt` 上，开启 confidence prior 的 9-episode eval 均值为 `64.03`，关闭 prior 为 `88.29`。这说明当前 confidence prior 在更宽评估样本上不是净正向。后续不应提高规则权重，而应按 episode 和子状态审查 rule prior 的负贡献来源，尤其检查 active 规则是否在某些初始姿态下压制了必要的恢复动作。

## Checkpoint 级 rule on/off sweep

关键 checkpoint 的规则先验效果应采用标准 `10` 条 episode 成对 sweep，而不是只比较单个 checkpoint 或临时 `9` 条 episode 诊断。成对 sweep 的基本单位为同一个 `final.full.pt` checkpoint 下的两次不训练评估：

```text
rule_off: rule_prior_enabled = false
rule_on:  rule_prior_enabled = true,
          rule_prior_allowed_statuses = confidence_active,
          rule_prior_mode = penalty_only
```

两组评估保持相同任务、随机种子、模型参数、MPPI 参数、`eval_action_mode=eval` 和 `eval_episodes=10`。该设置用于估计显式规则 prior 在推理期 planner 排序中的净效应，不用于判断策略网络是否已经内化规则。

结果解释采用三层口径。第一，若多数 checkpoint 的 `on-off` 差值接近 `0` 且远小于 episode 间方差，则说明规则 prior 已实际接入但净效应弱。第二，若某些 checkpoint 出现大幅正向或负向差值，应记录为阶段依赖的局部效应，而不是立即概括为方法有效或有害。第三，若同一训练后期连续多个 checkpoint 呈同方向大幅差值，才进入规则准确性、覆盖范围或权重机制的因果审计。

`100k -> 150k` 的标准 `10` 条 episode sweep 显示：`cycle020-028` 差异大多较小，`cycle029` 为 `+19.68`，`cycle030` 为 `-24.72`。因此，当前结论应是规则 prior 的效果具有 checkpoint 依赖性；下一步应对 `cycle029` 与 `cycle030` 分别做 episode-level rule-hit 审计，比较哪些初始状态因规则 prior 获益，哪些初始状态被规则 prior 压制。

## 规则关键性覆盖审计

规则是否“有效”不应只由命中后动作方向是否改变判断，还需检查规则是否覆盖任务链条中的关键分岔状态。当前审计采用三类指标。

第一，规则足迹。统计在线可读规则的数量、唯一子状态数、事件类型、阶段、动作族和关节覆盖。若规则集中在少数阶段或少数关节，即使局部有效，也不应解释为任务级规则完备。

第二，轨迹覆盖。将 eval trace 的每一步按在线 prior 相同的子状态离散化重新编码，计算 active 规则覆盖的 step 比例，以及各阶段内覆盖率。该指标用于区分“规则没有命中”和“规则命中了但收益不稳定”。

第三，高低回报对比。将 episode 按回报分为高回报、低回报和中间组，比较规则覆盖率、`rear/prone/falling/upright` 阶段占比、最终前向位移和最大前向位移。若低回报 episode 主要落在规则未覆盖阶段，则说明当前规则库缺少关键恢复规则，而不是已有规则本身必然错误。

`cycle029-030` 关键性审计显示，当前 active 规则只有 `5` 条、`3` 个唯一子状态，全部集中在 `omega_neg/vx_low/support_none` 下的 `prone` 或 `falling`，动作族只覆盖 `knee_pos`、`waist_neg`、`waist_pos`。这些规则覆盖全部 eval step 的 `16.7%`，对 `rear` 阶段覆盖为 `0.0`。低回报 episode 的 `rear` 占比显著高于高回报 episode。因此，后续规则归纳应优先补足 `rear` 恢复、`omega_pos/mid`、`vx_mid/high`、以及 `hip/ankle` 相关动作族，而不是单纯提高现有规则权重。

## 解释力 v0

规则价值的第一版解释力定义为离线审计指标，不直接修改在线 prior。目标是判断一条规则是否同时满足两类要求：第一，能充分解释已发生轨迹中的好坏差异；第二，能通过 SD 反事实更有信心地说明该规则为什么可能带来决策效益扩张。

初始公式为：

```text
Expl_v0(r) =
  0.35 * past_separation(r)
+ 0.15 * past_label_power(r)
+ 0.25 * cf_value_gain(r)
+ 0.15 * transition_gain(r)
+ 0.10 * basin_coverage(r)
- 0.10 * conflict_cost(r)
- 0.05 * complexity_cost(r)
```

各分量含义如下。

`past_separation` 使用已发生 eval trace 中命中规则子状态的步，比较实际动作符合规则偏好与违反规则偏好后 `50` 步阶段结果的差异。阶段结果不是稀疏回报，而是由前向位移、速度、支撑、姿态、后退和旋转风险组成的稠密恢复指标。

`past_label_power` 比较高回报 episode 与低回报 episode 中规则对齐程度的差异，用于衡量规则是否解释了轨迹标签分化。

`cf_value_gain` 使用 SD counterfactual 记录中规则偏好动作相对实际动作的预测阶段收益。该项对应解释力中的“为什么该规则可能带来决策效益”。

`transition_gain` 衡量规则对齐动作是否更容易推动阶段转换，例如 `rear -> prone/upright`、`prone -> forward_prone/upright` 或 `falling -> recoverable`。

`basin_coverage` 衡量规则在相近反事实状态中出现正收益的比例，用于区分偶然有效和局部区域有效。

`conflict_cost` 对同一子状态、同一关节的反向规则进行扣分。`complexity_cost` 对过细条件进行轻度扣分，避免规则退化为轨迹记忆。

解释力 v0 不以出现频率作为价值本身；支持数只作为原始审计信息保留。若后续使用解释力更新 confidence，应增加不确定性估计，避免少量 counterfactual 样本导致过高置信。

在 `cycle029-030` 上，解释力 v0 显示当前 active 规则不是解释力最高的一组。高分规则集中在当前未上线的 `rear`、`omega_pos/mid`、`hip/ankle` 或反向 `knee_neg` 相关候选上，而多条 active `knee_pos` 规则获得负分。该结果说明，现有 confidence 机制仍在保留早期局部规则，尚未充分反映后期轨迹分岔的解释力。

## 解释力规则上线前验证

解释力分数不能直接等同于在线先验强度。任何解释力规则库进入接续训练前，应先通过 eval-only smoke。该 smoke 使用同一 checkpoint 比较三种条件：

```text
rule_off
confidence_on
explain_on
```

若 `explain_on` 无法同时满足两点，则不进入训练：第一，不应破坏已有 confidence 规则在正向 checkpoint 上的收益；第二，应至少缓解负向 checkpoint 中 rule-on 相对 rule-off 的主要损失。

`cycle029/030` 第一版 explainability-v0 规则库没有通过该 smoke。该规则库一次性上线 `12` 条 `explain_active` 规则，其中多条高分规则主要由少量 SD counterfactual 样本驱动，并被归一化到较高 prior scale。结果显示，`cycle029` 从原 confidence_on 的 `70.566` 降至 `48.565`，`cycle030` 从 rule_off 的 `82.414` 降至 `50.615`。因此，解释力 v0 当前应保留为审计和候选排序工具，不能直接作为在线 prior 规则库。

后续映射策略应更保守：`past_label_power` 或已发生轨迹分岔解释力应作为上线必要条件；counterfactual-only 高分规则应先进入 observing；`rear` 相关规则应做小集合 ablation，而不是与全部高解释力规则同时上线。

## 规则切换与分布错配

eval-only 规则替换实验只能检验零训练适配性，不能直接判断新规则机制的训练价值。旧 checkpoint 的 replay buffer、value 估计和 planner 行为分布是在旧 confidence 规则先验下形成的。若旧规则已经显式处理某些局部状态，模型可能较少通过数据学习这些状态下的完整恢复策略；若新规则覆盖不同子状态或相反动作族，则会在一个已经适配旧先验的模型上改变 planner 排序，造成分布错配。

因此，解释力规则的正式判断应采用短接续训练，而不是只看直接替换 eval。短接续验证需满足：

1. 从较早 checkpoint 接续，避免在末端 checkpoint 上强行切换规则；
2. 降低 explainability 规则的基础 prior 权重，避免一次性高强度改写 planner；
3. 至少训练一个完整周期以上，让 replay 和 value 估计有机会适配新规则；
4. 训练后仍以 `rule_off / confidence_on / explain_on` 三路 eval 判断规则机制，而非只看周期内 quick eval；
5. 同步审计规则覆盖是否从 `prone/falling` 扩展到 `rear`、`omega_pos/mid`、`hip/ankle`。

在完成该短接续验证前，解释力 v0 的定位仍是“规则识别与候选排序指标”，不是可直接替换旧 prior 的最终规则库。

## Explainability-v0 短接续验证口径

短接续验证的目的不是证明规则库已经解决 Hopper，而是区分两类问题：第一，解释力规则是否能在同机制训练中正常接入并被 planner 使用；第二，经过少量训练适配后，最终 checkpoint 上的显式规则 prior 是否至少不再表现为明显分布错配。

该验证采用以下判读口径。周期内 `3` 条 eval 只作为 quick eval，用于发现明显崩溃或明显跃迁，不作为主要性能结论。关键 checkpoint 需要追加 `10` 条 episode 的 `rule_off / explain_on` 对照。若 `explain_on` 均值略高但绝对回报低，结论应是“存在弱兼容信号”，而不是“规则机制有效”。若 `explain_on` 只改善少数 episode、同时压低另一些 episode，则应进一步做 episode-level 轨迹审计，分析规则覆盖阶段与初始状态依赖。

`100k -> 110k` explainability-v0 短接续结果符合弱兼容信号口径。在线 prior 命中行比例在第二周期上升，最终 `10` 条 episode 对照中 `explain_on` 均值为 `15.955`，`rule_off` 均值为 `8.822`，差值为 `+7.13`。但周期内 quick eval 仍为 `0.0`，且标准对照中的非零收益集中在少数 episode。因此，当前方法论判断应聚焦于规则识别质量和阶段覆盖，而不是继续提高先验权重或据此声称任务级性能改善。

后续规则识别应优先检查三点：第一，解释力分数是否过度依赖少量 SD counterfactual；第二，`rear`、`omega_pos/mid`、`hip/ankle` 规则是否在高低回报分岔中具有稳定解释力；第三，新增候选在 rolling validation 中未被上线时，是支持不足、跨周期不一致，还是规则条件过细导致覆盖稀疏。

## 中期接续观察口径

Hopper-hop 的完整成功通常需要远高于 `100k-200k` 的训练预算，因此中期接续实验不以短预算任务成功作为主要判据。`110k -> 200k` explainability-v0 接续实验采用以下解释口径。

第一，周期内 `eval_episodes=3` 仅作为 quick eval。该指标用于发现明显崩溃或偶发恢复信号，不作为主要性能结论。若 quick eval 大多为零但训练 episode 中持续出现更高回报片段，应记录为训练侧恢复片段增强，而不是直接判定方法失败。

第二，关键 checkpoint 必须使用标准 `10` 条 episode 的 `rule_off / explain_on` 对照。该对照用于估计显式规则 prior 在同一模型上的净推理效应。若 `explain_on` 只轻微改变已经成功的 episode，而不改变成功 episode 集合，则说明规则已在 planner 排序中发挥作用，但尚未改变恢复 basin。

第三，规则库状态应与性能一起报告。`explain_active` 表示进入在线 prior 的规则；`rolling_validated` 和 `cycle_candidate` 仅作为证据积累，不参与在线动作先验。若候选持续增加但 active 规则不增加，说明闭环瓶颈在规则识别、证据稳定性或上线阈值，而不是训练脚本未运行。

第四，规则信心允许被削弱。若既有 active 规则在后续周期只以低支持证据重新出现，并未满足在线阈值，可降为候选或不再参与 prior。该机制用于处理冲突规则和早期规则过时问题，避免规则库只增不减。

在该口径下，`110k -> 200k` 结果被解释为中期适配信号：训练侧高回报片段增强，quick eval 偶发非零，最终 `10` 条 episode 中 `explain_on` 相对 `rule_off` 差值很小。下一阶段方法重点仍应放在解释力指标、阶段覆盖和规则上线机制，而不是单纯延长同一规则库或提高 prior 权重。

## 三种子共享规则闭环口径

多种子共享规则实验采用“训练周期”和“正式评估周期”分离的设计。训练仍每 `5k` 步中断一次，以便进行 SD 回归、失败归因、SD 反事实分析、候选规则生成和共享规则库更新。正式 checkpoint 与 official eval 不再每个 5k 周期执行，而只在 `25k` 的整数倍触发，每次使用 `10` 条 episode。

该设计的动机是避免低样本 quick eval 主导判断，同时不牺牲闭环方法所需的高频经验总结。5k 周期末保存的 `final.full.pt` 是接续训练状态，属于内部运行状态；`25k/50k/75k/100k` 保存点才是正式评估 checkpoint。

由于 5k 周期不再产生 official eval trace，周期总结的数据源改为训练 replay。每个 seed 周期结束后，从 `final.full.pt` 中导出 replay buffer episode，并写入 `retrospective_train_traces/`。这些轨迹用于 SD 回归和失败归因，但不作为 official eval 性能结果。manifest 中将其标记为 `train_replay` 来源，以避免与正式评估混淆。

共享规则库的更新顺序如下。每个周期先并行运行三个 seed 的训练；三个 seed 均完成后，分别导出 replay trace；随后将三种子的轨迹合并到同一个 manifest，训练一个共享 SD explainer，并基于合并归因结果更新同一个规则库。下一周期三个 seed 均读取该共享规则库作为在线 prior 来源。在线 prior 当前允许读取 `replay_validated,rolling_validated`，但不读取 `cycle_candidate`。

该设置将“跨 seed 证据”提前放入闭环内部，而不是事后汇总单 seed 规则。若规则在多个 seed 中反复出现，其 rolling validation 更容易获得稳定证据；若只在单 seed 或单一初始状态下出现，则应保持候选状态。

## v8 成功延续与阶段衔接规则

v8 不替换 v7 的失败恢复规则，而是在同一方法论框架中新增 `continue` 和 `transition` 两类规则角色。`continue` 关注局部好姿态如何被保持，`transition` 关注局部好姿态为什么未能衔接到落地、支撑和再起跳。当前 Hopper 实现仍使用 Hopper 专用结构态字段，但以任务适配器形式组织，避免把 Hopper 阶段和关节名称固化为通用方法本身。

第一步脚本 [mine_hopper_v8_success_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py) 读取已落盘 trace CSV，提取成功候选片段、成功延续片段和转接缺口片段，并检查这些片段是否被现有 v7 active 规则覆盖。当前默认复核 v72 三种子 `200k` 最终 official eval，输出位于：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_success_segments_200k_v72.md
```

该复核共分析 `30` 条最终 eval 轨迹，得到 `356` 个成功候选片段，其中 `349` 个属于转接缺口，只有 `2` 个被现有 v7 规则覆盖。该结果说明，当前规则库主要覆盖失败恢复区域，而对局部好姿态和阶段衔接失败的解释不足。

第二步脚本 [mine_hopper_v8_transition_actions.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/mine_hopper_v8_transition_actions.py) 在片段基础上抽取关键状态-动作证据。脚本记录每个片段的 `last_success`、`pre_degradation` 和 `degradation_onset` 三类锚点，但候选规则归纳只使用 `last_success / pre_degradation / post_success` 等因果上游锚点，避免把退化已经发生后的动作误作退化原因。动作族归纳默认只使用主导动作维度，避免同一状态下多个非弱动作被同时计入支持。

v8 动作证据引入精炼阶段标签。逐帧记录仍保留原始 SD 阶段，例如 `prone`、`falling`；但规则键中若当前帧满足成功候选条件，则根据支撑情况改写为 `near_good_air`、`near_good_contact` 或 `near_good_support`。该处理用于避免正回报、高躯干、高速度的转接片段被旧阶段检测器简单归为普通 `prone` 失败状态。

当前动作证据输出位于：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_actions_200k_v72.md
```

默认口径下，从 `356` 个片段中抽取 `715` 条关键状态-动作记录，聚合为 `30` 个 v8 规则键组，并产生 `25` 条离线观测候选规则。所有候选均为 `transition` 角色下的 `avoid_action_family`，主要集中在 `omega_pos / vx_high / support_none / near_good_air` 或 `near_good_contact` 状态。该结果应解释为“局部好姿态转接失败中存在可重复动作风险信号”，不能直接作为在线 prior 上线条件。下一步仍需加入 SD 反事实目标 `S_continue` 和 `S_transition`，判断这些 `avoid` 候选是否具有反事实优势，而不仅是已发生轨迹中的下一步退化相关。

第三步脚本 [counterfactual_hopper_v8_transition_actions.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/counterfactual_hopper_v8_transition_actions.py) 使用 SD 解释器在 v8 上游关键状态上进行动作反事实比较。默认读取 `cycle_040` 的共享 SD explainer，对每个规则键最多抽取 `40` 条 `last_success / pre_degradation` 记录，比较实际动作与固定动作探测集，并按 `objective=joint` 排序，其中：

```text
S_joint = 0.5 * S_continue + 0.5 * S_transition
```

默认输出位于：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_counterfactuals_200k_v72.md
```

本次默认运行评估 `300` 条关键状态，产生 `11` 条 `offline_sd_counterfactual_candidate`。候选以 `knee_neg` 为主，覆盖 `near_good_air / near_good_contact` 下的 `omega_pos/mid/neg, vx_high, support_none` 转接状态。该结果与观测证据形成互补：观测层主要给出若干动作族在好片段后伴随退化的 `avoid` 信号，SD 反事实层则给出若干可尝试保持或改善 `S_joint` 的 `soft_prefer` 信号。二者尚未合并为在线规则；后续应设计规则归纳器，将观测 `avoid`、SD `prefer`、跨 seed 支持、冲突关系和好片段误伤率统一压缩为 v8 候选规则库。

第四步脚本 [induce_hopper_v8_transition_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/induce_hopper_v8_transition_rules.py) 将观测层和 SD 反事实层候选合并为同一离线规则文件：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_rules_200k_v72.json
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_rules_200k_v72.md
```

当前合并得到 `36` 条离线 v8 转接规则，其中 `24` 条为 `offline_v8_supported`，`12` 条为 `offline_v8_conflict`。冲突包括两类：同一规则键、同一动作族在观测层表现为 `avoid`，但在 SD 反事实层表现为 `soft_prefer`；以及同一关节的相反方向在两层证据中互相竞争。该结果说明 v8 已经形成可审计候选库，但尚未达到在线接入条件。后续应先处理冲突削弱、同一规则键下动作族互斥、以及 held-out 轨迹误伤率验证。

第五步脚本 [validate_hopper_v8_transition_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/validate_hopper_v8_transition_rules.py) 对 v8 转接候选执行 episode-level held-out 诊断和冲突削弱。默认 held-out 子集为 `episode % 3 == 0`，验证结果输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_rule_validation_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_rule_validation_200k_v72.md
```

当前结果为：`36` 条候选中，`24` 条为 `heldout_supported`，`4` 条为 `heldout_weak`，`4` 条为 `heldout_conflict_unresolved`，`4` 条为 `heldout_weakened_by_conflict`。该结果用于把规则库更新器的输入分层：`heldout_supported` 可进入下一步 replay/held-out 误伤率审查；`heldout_conflict_unresolved` 只能保持观察；`heldout_weakened_by_conflict` 应降低置信或暂不参与 prior。由于候选仍来自同一最终 eval 池，该验证只能作为保守诊断，不能替代跨周期、跨训练分布验证。

第六步脚本 [audit_hopper_v8_transition_replay_safety.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_v8_transition_replay_safety.py) 将 `heldout_supported` 规则投放到 `cycle_040` 三种子训练 replay 中做误伤率审查。默认输入为三种子各 `400` 条 replay episode，共 `1200` 条轨迹、`600000` 步。输出位于：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_replay_safety_200k_v72.json
results/sd_retrospective_rule_induction/reports/hopper_v8_transition_replay_safety_200k_v72.md
```

该步骤采用不同于 recovery 规则的安全判据。v8 transition 规则的目标本来就是 `near_good_air / near_good_contact` 等局部好姿态，因此不能要求规则主要命中坏片段。误伤风险定义为：`avoid_action_family` 是否会压制 replay 中好姿态下实际使用的主导动作族；`soft_prefer_action_family` 是否会在好姿态下偏向实际主导动作的相反方向。

当前 replay 安全审查结果为：`24` 条 `heldout_supported` 规则中，`7` 条为 `replay_safety_supported`，`16` 条为 `replay_observe`，`1` 条为 `replay_false_intervention_risk`。唯一高风险规则为：

```text
transition|transition_gap|omega_pos|vx_high|support_none|near_good_contact|stable->avoid:knee_neg
```

该规则在 replay 中命中 `7891` 步，全部为 near-good 命中，但好姿态中实际主导动作为 `knee_neg` 的比例达到 `0.322`，因此若作为 `avoid` prior 可能明显压制成功片段中的常用动作。低误伤候选主要包括 `avoid:hip_neg`、部分 `avoid:waist_neg` 和 `soft_prefer:knee_neg`。这些结果说明，v8 规则不能只按“能解释退化”上线，还必须通过好片段误伤审查。

第七步脚本 [export_hopper_v8_replay_safe_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/export_hopper_v8_replay_safe_rules.py) 将 `replay_safety_supported` 规则导出为独立窄候选库：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.json
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.md
```

当前导出 `7` 条规则，其中 `avoid_action_family = 6`，`soft_prefer_action_family = 1`。该文件仍不是在线规则库，只是下一步跨 checkpoint eval 或短接续 smoke 的候选输入。默认筛选条件为 `replay_false_intervention_rate <= 0.10`。

第八步脚本 [eval_hopper_v8_safe_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/eval_hopper_v8_safe_prior.py) 对 `200k` 三种子 checkpoint 执行 `10` 条 episode 的 eval-only 对照。三种模式为：

```text
rule_off
v7_current
v8_safe
```

其中 `v7_current` 读取 `cycle_040` 旧共享规则库中的 `replay_validated,rolling_validated` 规则，`v8_safe` 只读取 `hopper_v8_transition_replay_safe_rules_200k_v72.json` 中的 `v8_replay_safe_candidate`。该步骤用于判断 v8 窄候选是否存在明显推理期破坏性，而不是证明其训练价值。

结果输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8_safe_eval_only_200k_v72_20260508.json
results/sd_retrospective_rule_induction/reports/hopper_v8_safe_eval_only_200k_v72_20260508.md
```

三种子均值如下：

```text
rule_off   = 144.137
v7_current = 129.935
v8_safe    = 144.026
```

因此，v8 safe 窄候选没有表现出明显 eval-only 破坏性；同时，旧 v7 当前规则在该口径下低于 rule_off，主要由 seed3 下滑贡献。下一步应进入第二轮在线测试的短 smoke，优先测试 `v8_safe` 单独先验，而不是把旧 v7 和 v8 直接混合。

## v8 replay-safe 在线 smoke 协议

第二轮在线 smoke 使用 `v8_safe` 单独先验，并从 v72 三种子 `200k` checkpoint 接续。该测试的目的不是在 `10k` 预算内解决 Hopper，而是验证 v8 transition replay-safe 规则能否在训练期以低强度先验接入，并确认 SD 接续训练、归因总结、规则更新和多种子共享规则库是否构成稳定闭环。

实验采用两段 `5k` 周期：

```text
cycle041: 200040 -> 205040
cycle042: 205041 -> 210041
```

每个周期均执行三种子并行训练、`10ep` formal eval、replay trace 导出、共享 SD explainer 接续训练、反事实归因、规则评分、规则状态更新和闭环规则库合并。初始规则库为：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8_transition_replay_safe_rules_200k_v72.json
```

在线 prior 采用以下门控和强度：

```text
rule_prior_allowed_statuses = v8_replay_safe_candidate
rule_required_status        = v8_replay_safe_candidate
rule_prior_mode             = penalty_only
rule_prior_weight           = 0.05
rule_prior_opposite_penalty = 0.02
rule_prior_cap_penalty      = 0.02
rule_prior_preferred_cap    = 0.70
```

该设置只让通过 replay-safe 审查的规则参与 MPPI 候选排序。`cycle_candidate` 规则会被保留在闭环库中，但不进入在线 prior。`online_enabled` 字段在当前训练入口中不是硬门控；实际门控由 `rule_prior_allowed_statuses` 决定。因此，判断规则是否上线应以训练日志中的 `Installed v6 rule prior: rules=...` 和 CSV 中的 `rule_prior_*` 指标为准。

本次 smoke 的 formal eval 结果为：

```text
200k eval-only v8_safe mean3 = 144.026
205k cycle041 mean3         = 178.394
210k cycle042 mean3         = 180.607
```

三种子明细为：

```text
205k: seed1 = 103.086, seed2 = 206.638, seed3 = 225.459
210k: seed1 = 102.736, seed2 = 207.326, seed3 = 231.758
```

该结果只作为短程兼容信号：v8 safe 单独先验没有在本次接续中造成明显破坏，并在短程内维持或略微提高 `10ep` eval 均值。由于 Hopper 的长期成功通常需要远高于该 smoke 的训练预算，不能将该结果解释为任务级成功。

规则命中统计表明，该先验仍属于轻量介入。`cycle041` 合计命中率为 `0.100`，非零先验项比例为 `0.053`，平均先验项为 `-0.000656`；`cycle042` 合计命中率为 `0.087`，非零先验项比例为 `0.073`，平均先验项为 `-0.000899`。因此，当前 prior 的作用主要是少量场景下的候选排序惩罚，而不是强制策略替换。

最终闭环规则库位于：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8safe_smoke_200k_to210k_seed123_20260508/cycle_042/rules/library_closed_loop_cycle_042.json
```

该库包含 `9` 条规则，其中 `7` 条为 `v8_replay_safe_candidate`，`2` 条为 `cycle_candidate`。`cycle_candidate` 仅作为后续滚动验证对象，不参与当前在线 prior。`cycle042` 中间候选数量达到 `substate_rules_v4 = 84`、`library_v4 = 95`，显示规则提取已能产生较多候选；同时，validation 成本开始成为闭环周期的主要工程开销。后续方法应优先改进候选压缩、状态条件泛化和 validation 加速，而不是简单提高 prior 权重。

### 候选压缩与滚动晋升

闭环规则更新现采用“先压缩、再验证、再滚动更新”的顺序。

`analyze_substate_rules.py` 在输出 substate 候选前执行保守压缩：

```text
max_rules = 48
max_rules_per_substate = 2
max_rules_per_event_action = 8
```

排序依据仍为 `mean_advantage` 和 `support`，但同一 substate 和同一状态事件-动作族组合不能无限累积候选。该设计用于抑制规则库膨胀，使 validation 面向少数解释力较强、覆盖较清晰的候选。

`validate_substate_rules.py` 不再按“每条规则扫描全部轨迹”的方式执行，而是将规则按 `substate_key` 建索引，并对每条轨迹逐帧计算相关状态事件的 substate key。该实现保持原统计定义不变，但显著降低重复扫描成本。

`closed_loop_rule_update.py` 的滚动更新规则如下。已有 `cycle_candidate / cycle_supported` 若在新周期再次出现，会优先占用携带名额，以便累积证据；新候选只使用剩余额度。候选满足 `rolling_min_cycles`、`rolling_min_support`、`rolling_min_advantage` 且 `good_share <= rolling_max_good_share` 时晋升。若当前实验要求的在线状态以 `v8_` 开头，晋升状态即为该在线状态，例如 `v8_replay_safe_candidate`；否则沿用旧的 `rolling_validated`。

若候选跨周期满足基本支持和优势阈值，但 `good_share` 超过安全阈值，则标记为 `cycle_weakened`。该状态保留审计记录，但不参与在线 prior，也不继续作为滚动晋升对象。该机制实现了规则信号可以增强、削弱和退出的基本闭环。

### 观察层放宽与上线层分离

`cycle001-010` 正式空库运行显示，候选规则能够持续产生，但上线规则增长较慢。为避免闭环过早退化为少量早期规则，同时不降低在线 prior 的安全门槛，后续变体将观察层与上线层分离：

```text
max_online_rules_per_cycle    = 2
max_cycle_candidates          = 6
rolling_max_good_share        = 0.25
rolling_weaken_max_good_share = 0.40
```

其中 `max_online_rules_per_cycle` 表示每周期从已经满足在线状态的候选中最多合并 `2` 条，而不是旧实现中的 `1` 条。`rolling_max_good_share` 仍是晋升为 `v8_replay_safe_candidate` 的必要门槛；`rolling_weaken_max_good_share` 仅控制候选何时被削弱并退出滚动池。也就是说，`good_share` 位于 `0.25-0.40` 的规则不会上线，但可继续作为观察候选跨周期积累证据。该调整放宽的是候选保留和上线吞吐量，不改变在线 prior 只读取 `v8_replay_safe_candidate` 的原则。

## v8 replay-safe 续训协议

`210k -> 250k` 续训实验沿用第二轮 smoke 的低强度显式规则先验，但将 formal eval 与 checkpoint 频率降为每 `25k` 一次，避免在短周期内过度解释 eval 噪声。SD 回归、失败归因、候选规则提取、规则验证和共享规则库更新仍保持每 `5k` 一次，以维持“训练一段时间后总结一次”的闭环结构。

当前协议为：

```text
cycle_steps                = 5000
formal_eval_freq           = 25000
formal_checkpoint_freq     = 25000
formal_eval_episodes       = 10
rule_prior_allowed_statuses = v8_replay_safe_candidate
rule_required_status        = v8_replay_safe_candidate
rule_prior_mode             = penalty_only
rule_prior_opposite_penalty = 0.02
rule_prior_cap_penalty      = 0.02
rule_prior_preferred_cap    = 0.70
sd_epochs                  = 3
sd_max_transitions          = 50000
```

该续训从上一阶段 `cycle042` 的三种子 TD-MPC2 checkpoint、共享规则库和 SD explainer 接续。SD explainer 在每个周期使用上一轮 checkpoint 初始化，因此概念上属于接续拟合，而不是每轮从零训练。

真实在线 `cycle043` 显示，候选压缩在闭环中按预期生效：`substate_rules_v4` 从压缩前 `58` 条变为 `34` 条，低于全局上限 `48`，并受每个 substate 与每个状态事件-动作族上限约束。validation 使用索引式 trace 缓存，报告中记录 `num_cached_traces = 524`。闭环规则库保留 `7` 条可上线 `v8_replay_safe_candidate`，同时携带 `5` 条 `cycle_candidate` 作为后续滚动证据；本周期没有新规则上线。

`cycle044` 进一步验证了滚动晋升机制。该周期 `substate_rules_v4` 从压缩前 `80` 条降至 `48` 条，validation 缓存 `1053` 条 trace。闭环更新没有从单周期候选中直接加入新规则，但将跨周期重复出现且满足累计阈值的 `rotation_risk|omega_neg|vx_mid|support_some|rear->prefer:knee_neg` 晋升为 `v8_replay_safe_candidate`。因此，当前在线规则增长主要依赖跨周期证据累积，而不是单周期高分候选直接上线。

## 正式有效性测试口径

接续训练只用于工程调试和安全性观察，不作为规则方法有效性的主要证据。原因是 TD-MPC2 本身会随训练预算推进而改变性能，单独观察 `rule_on` 续训分数是否增长，不能区分规则收益与基础算法自身学习进展。

规则生成机制进入正式测试前必须先通过两类前置检查。第一，逻辑审查：规则的状态谓词、动作方向、冲突削弱、信心更新和上线门控应与 Hopper 任务机制一致，且不应明显压制已知好姿态片段。第二，smoke test：规则 prior 能被训练入口正确加载，规划排序指标中能观察到规则命中和惩罚项，短程运行不出现明显崩塌或工程中断。

通过前置检查后，正式测试应从头开始训练，而不是从既有 TD-MPC2 checkpoint 接续。默认预算统一为 `100k`；由于官方 TD-MPC2 参考记录没有 `50k` 口径，`50k` 不作为主测试预算。评估采用标准 `10` 条 episode 口径，并在固定 checkpoint 上保存可复查轨迹。正式结论应同时包含三部分：

```text
1. 与官方 TD-MPC2 同任务、同预算或最接近预算结果的数值对比。
2. 多种子均值与方差，不以单 checkpoint 或单 seed 作为主要证据。
3. 轨迹审查：检查规则命中子状态、动作分布变化、姿态改善片段和失败模式是否与规则解释一致。
```

若需要因果归因，应补充同起点、同预算、同种子的 `rule_off` 对照线，或在同一 checkpoint 上执行推理期 rule on/off 对照。但这些对照用于解释机制，不替代从头训练与官方 TD-MPC2 口径的主比较。

正式 100k 协议单独记录于：

```text
docs/sd_retrospective_rule_induction/hopper_100k_formal_protocol.md
```

主测试采用空规则库从头闭环生成规则，第一周期不启用规则 prior，避免把高预算轨迹中离线挖出的规则泄漏到 `0-100k` 训练。若确需测试预先审查过的规则库，应作为 seeded-rule 变体单独报告，并明确规则来源。

## 终点评估轨迹审查方法

正式 `100k` 终点评估后，使用只读脚本对官方 eval 逐步轨迹进行复核：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v8formal_100k_final.py
```

该审查不重新运行环境、不改变 checkpoint，也不参与训练。其目的在于回答两个问题：第一，最终轨迹中是否出现了相对理想的 Hopper 姿态片段；第二，最终上线规则是否在这些轨迹中真实命中并改变动作方向统计。

姿态片段采用三类信号识别：结构化成功条件、密集姿态势能阈值和正即时奖励。结构化成功条件为：

```text
torso_z >= 0.75
abs(pitch) <= 0.45
vx >= 0.80
abs(pitch_rate) <= 3.50
body_collision < 0.50
```

同时记录 `dense_pose_potential >= 0.65` 和 `reward > 0` 的连续片段。连续片段最小长度默认设为 `3` 帧，用于过滤孤立噪声。

规则命中复现在线 prior 的子状态匹配逻辑。对 `rotation_risk` 规则，要求 `rotation_penalty >= 0.5` 或 `abs(pitch_rate) >= 3.0`；对 `unsupported_high_torque` 规则，额外统计实际动作最大绝对值是否达到 `0.65`。动作方向采用四维动作列：

```text
waist -> action_0
hip   -> action_1
knee  -> action_2
ankle -> action_3
```

若规则要求的有符号动作值不低于 `0.2`，记为对齐；不高于 `-0.2`，记为相反；中间区间记为中性。该统计只能说明规则与最终动作方向的对应关系，不能单独证明规则带来回报提升。

## 在线闭环中的 v8 transition 规则接入

上一轮正式 `100k` 轨迹审查显示，失败恢复规则能够命中并改变局部动作方向，但主要集中在 `rear/prone` 恢复状态，难以解释或维持短暂高分姿态片段。因此，后续闭环在原有失败恢复规则后增加 v8 transition 规则族。该规则族不替代原有恢复规则，而是补充近成功片段的阶段衔接信息。

每个周期的总结顺序调整为：

```text
1. 训练 5k steps，并导出三种子 replay traces。
2. 使用当前窗口轨迹接续拟合 SD explainer。
3. 按原 v4 子状态机制提取失败恢复规则，并更新 recovery library。
4. 从 replay traces 中挖掘成功片段和 transition gap。
5. 在 transition anchor 上提取观测动作证据。
6. 使用本周期 SD explainer 对 transition anchor 做反事实动作比较。
7. 合并观测层和 SD 反事实层，形成 v8 transition 候选规则。
8. 执行 held-out validation 和 train-replay safety 审查。
9. 仅将 `v8_replay_safe_candidate` 合并到最终共享规则库。
```

在线 prior 仍只读取 `v8_replay_safe_candidate`。`v8_transition_replay_safety` 来源允许两类干预：`soft_prefer_action_family` 表示轻度鼓励某动作族，`avoid_action_family` 表示轻度压制某动作族。二者均只作用于 MPPI 候选排序，不替换最终动作，也不直接改写策略网络。

该设计的判断依据是：若规则只从失败状态中总结，容易持续产生“如何从坏姿态恢复”的局部规则；若同时从短暂高分片段及其退化前后总结，则有机会形成“如何维持或完成阶段转接”的规则。该补充仍属于通用方法论的一部分，只是当前实现使用 Hopper 的结构化状态特征来定义成功片段、转接风险和动作族。

## v8 transition 闭环重跑后的方法修订方向

`hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2` 在 `75k` 完整评估节点后中止。该结果对方法设计的主要含义不是“规则 prior 未生效”，而是“当前 transition 规则提取链条未能形成上线规则”。截至 `cycle015`，线上规则仍全部来自 `v4_substate` 恢复规则，主要覆盖 `rear`、`unsupported_high_torque`、`rotation_risk` 等失败恢复子状态；v8 transition 分支虽然可持续识别 transition-gap 片段，但未能通过 held-out validation 和 replay safety 形成 `v8_replay_safe_candidate`。

后续方法修订应保持三项约束。第一，不直接降低在线规则门槛以换取更多规则上线，避免将未经验证的 transition 规则注入 MPPI 排序。第二，将重点从“增加恢复类规则数量”转向“提高 transition anchor 的有效性”，包括更精细地区分近成功接触、腾空准备、支撑风险解除和速度保持等阶段。第三，SD 反事实目标应更明确地面向阶段转接收益，而不只比较短 horizon 内的局部姿态或风险指标。

因此，下一轮在线测试前应先完成 transition 规则链条的离线审查：检查成功片段是否覆盖真实高分姿态前后的关键窗口，检查观测动作证据是否在 seed 间一致，检查 SD 反事实评分是否能区分“维持好姿态”与“从坏姿态恢复”，并检查 replay safety 是否过度保守。只有在这些检查显示 transition 候选能够稳定产生后，才进入新的从头闭环训练。

## v8 transition 规则键的 phase-any 聚合

后续 v8 transition 规则默认采用 `phase_any` 聚合键：

```text
role | quality_bin | omega_bin | vx_bin | support_bin | phase_any | risk_bin
```

该设计保留了与转接控制直接相关的速度、角速度、支撑和风险状态，同时弱化 `refined_phase` 的精细划分。其理由是，短暂高分或近高分片段常在 `falling`、`near_good_air`、`near_good_contact` 之间快速切换；若把这些阶段别名作为硬匹配条件，跨 seed 证据会被拆散，导致每条规则的 support 不足。`phase_any` 并不是完全泛化到所有状态，它仍要求 `role=transition`、同一 `quality_bin`、同一 `omega_bin`、同一 `vx_bin`、同一 `support_bin` 和同一 `risk_bin`。

在线 prior 和 replay safety 均按相同语义解释 `phase_any`。在 replay safety 中，同一规则会匹配对应 `role/quality/omega/vx/support/risk` 下的所有 refined phase；在在线 prior 中，`phase_any` 允许当前状态的 `refined_phase` 为任意值，但其他 token 必须完全一致。该修改只作用于 v8 transition 规则，不改变 v4 substate 恢复规则。

该聚合键的离线 smoke 标准为：在既有真实轨迹上，transition action evidence、SD counterfactual、held-out validation 和 replay safety 至少能形成一条 `v8_replay_safe_candidate`，且 false intervention rate 不超过当前安全阈值。`cycle015` 的离线 smoke 已满足该标准，导出规则为：

```text
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|support_risk->soft_prefer:waist_pos
```

## 规则发现漏斗的早期停止口径

`phase_any` 正式重跑的早期观察显示，在线规则 prior 可以在规划排序中产生局部行为影响，但若连续周期没有扩展规则库，则继续训练会逐渐退化为少量恢复规则的重复使用。后续类似实验采用以下早期停止口径：

```text
1. 不以早期 formal eval 回报作为单独停止依据。
2. 先检查 rule_prior_hit、rule_prior_bonus_mean 和轨迹动作方向统计，确认规则是否实际作用于动作排序。
3. 若规则已有局部作用，但状态覆盖率低、上线规则连续周期不增长、transition 分支持续为 0，则停止当前运行并回到规则发现层修订。
4. 若规则既没有规划命中，也没有轨迹动作方向变化，则立即中断并审查在线 prior 加载或匹配逻辑。
```

该口径将“规则是否生效”和“规则是否足够关键”区分开来。前者是工程问题，后者是规则识别问题。`hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2` 属于后者：`cycle006` 的规则命中率约为 `4%-12%`，部分种子中相反方向动作减少，但三条规则总状态覆盖率约 `3.2%`，且 `cycle006-cycle007` 仍未产生 transition 上线规则。

## 规则发现漏斗的下一版修订

后续闭环默认提高反事实状态抽样和候选保留容量：

```text
max_states_per_episode = 10
max_cycle_candidates   = 12
```

该修改针对的是规则发现层，而不是在线 prior 门槛。在线 prior 仍只读取 `v8_replay_safe_candidate`。研究意图是增加每周期被 SD 反事实比较的状态数量，并让更多跨周期候选保留下来，从而避免三种子、每周期约 `150` 条 episode 经历后仍只形成极少数上线规则。

高 `good_share` 候选的处理也应与在线拒绝分离。若某条规则在好轨迹中占比偏高，但同时具备较高 support 或明显 mean advantage，不应直接 `rejected`；应保留为 `cycle_candidate` 或等待后续拆分。其原因是 Hopper 的恢复动作和阶段转接动作可能同时出现在好坏轨迹中，高 `good_share` 更可能表示状态谓词过粗，而不必然表示规则无效。只有当后续拆分、回放安全性或反事实优势均不能支持时，才应退出候选池。

## widefunnel 100k 结果的解释口径

`hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1` 完成后，方法层面的主要结论是：放宽规则发现漏斗能够改善规则产出，尤其能够使 v8 transition 分支形成可上线规则。最终规则库中有 `20` 条 `v8_replay_safe_candidate`，其中 `7` 条来自 `v8_transition_replay_safety`，不再是纯恢复规则库。

该结果应与性能结论分开解释。终点 `100k / 10ep` 三种子均值为 `82.447`，高于既有 TD-MPC2 official `100k` 参考均值 `13.63`，但 seed 间差异较大，且轨迹审查显示总体 `prone_frac=0.6598`、`upright_frac=0.0001`。因此，当前更稳妥的解释是：规则发现机制和在线闭环已产生更强信号，策略出现短暂较好姿态片段，但尚未形成稳定 Hopper 步态。

后续分析应优先补充 v8 transition 规则的 episode-level 命中审查，特别是检查 transition 规则是否集中命中高分片段前后的支撑风险解除、速度保持和旋转风险控制窗口。若这些规则确实在高分片段前后命中，并减少相反方向动作，则可认为规则方向与阶段转接有初步一致性；若命中主要发生在无效 prone 状态，则需要继续细化 transition 状态谓词。

本轮补充的规则作用审查确认了两点。第一，final eval 使用的是 `cycle019` 规则库中的 `16` 条规则，`cycle020` 总结后新增到 `20` 条的规则尚未作用于该 final eval。第二，transition 规则的作用并不均匀：`waist_pos` 鼓励和 `waist_neg` 避免规则与高分 episode 高度共现，且动作方向大体一致；但 `avoid:hip_neg` 和 `avoid:knee_pos` 在同一 `omega_pos/vx_high/support_none/rotation_risk` 状态下出现高 harmful rate，说明该状态谓词对 hip/knee 动作而言过粗。

后续规则更新不宜简单按“transition 规则整体有效或无效”处理。更合适的策略是保留或增强腰部方向规则，同时对高 harmful rate 的 hip/knee avoid 规则降低信心、进入拆分池或要求更强的 rule-on/off 证据后再上线。该结论与用户提出的“规则信心可被增强、削弱和消除”机制一致。

## rule-effect confidence 第一版

规则信心更新新增一条后处理路径：用终点或周期内 rule-effect audit 的行为统计更新规则库信心，而不是只依赖规则生成时的 support、advantage 和 replay safety。实现脚本为：

```text
experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
```

该脚本的输入为规则库和规则作用审查报告，输出为新的 confidence-weighted 规则库。核心字段为：

```text
confidence
confidence_status
effective_prior_scale
effect_confidence_history
```

在线 prior 已支持读取 `effective_prior_scale` 或 `confidence`，因此该机制无需改写 planner，只需在后续实验中将 `rule_prior_allowed_statuses` 设置为 `confidence_active` 并加载 confidence 规则库。

第一版更新逻辑采用保守解释。若规则命中充分且方向一致率高，则增强或保持高信心；若命中充分且 harmful rate 高，则显著削弱；若低命中或未命中，则按 `low_hit_decay` 或 `absence_decay` 缓慢衰减。规则不会因单次审查被物理删除，而是进入 `confidence_observing` 或 `confidence_retired`。这使偶发不良规则不再必然成为闭环实验的致命风险，但仍保留可复查证据。

在 widefunnel `cycle019` 实际生效规则库上，第一版更新将 `16` 条规则分为 `14` 条 `confidence_active` 和 `2` 条 `confidence_observing`。两条观察规则分别是 `avoid:knee_pos` 和 `avoid:hip_neg` 的高频 transition avoid 规则；腰部方向 transition 规则保持激活。该结果符合当前人工审查结论。

同 checkpoint 的 eval-only 对比显示，`confidence_active` 规则库相对原 `cycle019` 规则库有轻微均值改善。三种子 `10ep` 均值为：`rule_off=76.201`、`original_cycle019=77.916`、`confidence_active=79.511`。因此，第一版 confidence 更新通过了弱 smoke test：它没有破坏原有规则 prior，并在剔除两条高 harmful 规则后带来小幅正向变化。

该结果仍需谨慎解释。由于本测试只是在已训练 checkpoint 上替换显式 planner prior，不能说明从头训练中的长期收益；且均值差异小于 Hopper 短预算评估的方差。后续正式测试应把 confidence 更新嵌入每周期规则库更新，使规则在训练过程中动态增强、削弱和退出，而不只在终点后处理。

### rule-effect confidence 安全修订

`hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2` 的终点规则开关诊断显示，关闭 eval-time 规则后部分种子仍低分，说明训练期规则偏置可能已经写入 checkpoint；同时 seed1 在 rule-on 时明显下降，说明显式规则 prior 仍存在额外负干预。基于该结果，confidence 机制修订为更保守的在线门控。

修订后的 `apply_rule_effect_confidence.py` 包含四项变化。第一，新规则默认信心从 `1.0` 降为 `0.50`，低于 `confidence_active_threshold=0.65`，避免未被充分验证的规则直接上线。第二，当 `selected_gate_hits >= 100` 且 `direction_harmful_rate >= 0.45` 时，触发 `effect_harmful_deactivated`，将规则信心压到 active 阈值以下。第三，只有当方向一致率超过阈值且 `good_state_hit_rate` 或 `high_episode_hit_rate` 至少达到 `0.05` 时，才允许 positive delta 增强规则；否则记为 `effect_consistent_but_low_context`，不再仅凭“不反向”强化规则。第四，对同一状态、同一关节的相反有效动作方向进行上线前仲裁；若没有明确更可靠的一侧，则降为 `confidence_observing`。其中 `avoid:hip_neg` 与 `soft_prefer:hip_pos` 被视为同一有效方向，因为二者在 `penalty_only` 下都惩罚负向 hip 动作。

在旧 `cycle020` 规则库和审计报告上进行离线 smoke，修订后 active 规则数由 `31` 降为 `24`，`effect_harmful_deactivated` 为 `5` 条，`active_conflict_demoted` 为 `2` 条，active 规则中的相反有效方向冲突为 `0`。该 smoke 只验证机制按预期过滤问题规则，不构成性能结论。

## rule-effect confidence 在线闭环口径

下一轮正式在线测试采用从空规则库开始的 `100k` 三种子闭环，而不是把既有 confidence 规则库作为初始知识注入。该设计使本轮只检验一种机制变量：规则在训练过程中被生成、审查、增强、削弱或退出后，是否能形成比静态规则库更可靠的显式动作先验。

每个周期的规则库更新顺序为：

```text
1. 三种子各训练 5000 steps，并导出 replay traces。
2. 使用本周期轨迹接续拟合 SD explainer。
3. 运行 v4 substate 恢复规则归纳和 v8 transition 规则归纳。
4. 仅将通过 replay safety 的 `v8_replay_safe_candidate` 作为本周期可上线候选。
5. 对当前周期 eval 或 replay trace 执行 rule-effect audit，统计命中、方向一致、相反方向和 harmful 证据。
6. 使用 rule-effect confidence 更新规则库，写入 `confidence`、`confidence_status`、`effective_prior_scale` 和 `effect_confidence_history`。
7. 下一周期在线 prior 只读取 `confidence_active`，并按 `effective_prior_scale` 缩放轻量惩罚强度。
```

该机制保留 `confidence_observing` 和 `confidence_retired` 规则的历史证据，但默认不让它们参与 MPPI 候选排序。若某条规则长期未命中，其信心缓慢衰减；若命中充分且方向一致率较高，其信心增强或保持；若命中充分但 harmful rate 较高，其信心显著削弱并退出在线激活集。

本轮的核心评价对象不是短期 Hopper 得分，而是规则系统自身是否形成可复查的价值信号。主要指标包括：上线规则的数量和来源、transition 规则占比、规则状态演化、规则命中密度与高分或近高分片段的关系、方向一致率和 harmful rate 的变化，以及规则 prior 对最终动作方向的实际影响。标准 `10ep` eval 回报用于与 TD-MPC2 official 口径保持可比，但不单独作为规则价值判断标准。

### 候选规则进入 confidence 系统的口径

后续在线闭环不再将 `cycle_candidate` 长期停留在不可上线状态。其方法学理由是：Hopper 早期训练中成功片段稀少，若只允许 `v8_replay_safe_candidate` 进入 confidence 更新，则大量由失败恢复归因产生的候选规则无法接受行为审查，也无法被增强、削弱或淘汰，实验会退化为空规则库训练。

修订后的口径为：`cycle_candidate`、`cycle_supported`、`rolling_validated`、`v8_replay_safe_candidate`、`confidence_active` 和 `confidence_observing` 均进入 rule-effect audit 与 confidence update；但在线 prior 仍只读取 `confidence_active`。因此，候选规则必须先通过回放命中、方向一致性、有害率和冲突仲裁形成 active 状态，才会在下一周期影响动作排序。该设计介于两个极端之间：既不要求早期规则必须达到过强的 replay-safe transition 标准，也不允许未经 confidence 审查的候选规则直接控制 planner。

## 脚尖恢复延迟规则的离线诊断

为检验“阶段转换过早”这一规则模板，新增 `diagnose_hopper_toe_recovery_rule.py`。该脚本不参与训练闭环，只用于离线判断 SD 是否支持如下规则假设：

```text
当 Hopper 处于低身位或俯身恢复阶段，脚尖已接触但脚跟和全足支撑尚未形成时，
不应立即进入脚尖弹射；应优先选择能继续恢复支撑、降低俯身和控制旋转的动作。
```

状态谓词采用可解释的结构态阈值：

```text
touch_toe >= 0.80
touch_heel < 0.50
full_foot_support < 0.35
grounded_prepared < 0.50
grounded_upright < 0.50
grounded_prone、low_body、低 torso_z 或 rotation_penalty 至少一项成立
```

SD 诊断采用固定动作探针库，动作幅度默认为 `0.35`、`0.70`、`1.00`。对每个目标状态，分别用实际动作和探针动作进行 `15` 步 SD 推演，并用恢复目标函数排序：

```text
1.40 * max(full_foot_support, support_contact)
+ 1.10 * max(grounded_prepared, grounded_upright)
+ 0.65 * torso_z_quality
+ 0.45 * controlled_spin
+ 0.20 * forward_speed
- 1.15 * max(grounded_prone, grounded_rear)
- 0.80 * falling_air
- 0.65 * rotation_penalty
```

该目标函数故意让支撑恢复和姿态控制高于短期前进速度。若最佳探针动作相对实际动作的恢复分数优势至少为 `0.05`，且在支撑、准备姿态、俯身降低或旋转降低中至少一项有 `0.05` 以上改善，则该状态记为支持“延迟弹射、恢复支撑”规则。规则整体通过阈值为：支持样本不少于 `8`，且支持率不低于 `0.35`。

该脚本的作用边界是明确的：它验证 SD 是否能对人工锁定的失败窗口给出规则化诊断证据，但尚不等同于完整自动规则发现。若后续接入在线训练，该规则应进入 confidence 机制，由行为命中、方向一致、反效果率和高分上下文继续增强、削弱或退休。

### 在线接入方式

当前在线 prior 的执行单元是“状态谓词 + 动作族”，而脚尖恢复诊断规则本身是效果型规则。因此，首次在线测试采用保守投影：只将 15 步 SD 诊断中支持数最高、且不与同关节反向动作冲突的 `waist_pos` 转成在线规则：

```text
toe_recovery_low_support -> soft_prefer_action_family: waist_pos
```

该规则并不直接写死“禁止起跳”，也不在 planner 内部额外调用 SD 做二次推演；它仍遵循既有轻量 prior 机制，仅在 MPPI 候选排序阶段对相反方向动作施加小惩罚，并按 `effective_prior_scale` 缩放。该设计保留了当前框架的轻量介入特征，同时将 SD 诊断得到的阶段性原则注入训练过程。

未同时激活 `knee_pos` 和 `knee_neg`。原因是二者虽然都在 SD 诊断中出现，但在当前动作族 prior 表达下属于同关节相反方向，若同时上线会产生直接冲突。后续若要完整表达效果型规则，应将在线 prior 从“动作族匹配”扩展为“候选动作短视界结构态效果评分”，届时可直接依据支撑恢复、俯身降低和旋转控制打分，而不必压缩成单一动作方向。

### 脚尖恢复规则的审计口径

`toe_recovery_low_support` 规则必须在在线 prior 和 rule-effect audit 中使用一致的状态谓词。审计端的谓词为：

```text
touch_toe >= 0.80
touch_heel < 0.50
full_foot_support < 0.35
grounded_upright < 0.50
且满足以下至少一项：
  grounded_prone >= 0.50
  torso_z/rootz <= 0.72
  rotation_penalty >= 0.50
  abs(pitch_rate) >= 3.0
```

该规则不要求候选动作达到高扭矩门控。方法学理由是：脚尖恢复低支撑本身已经是专门状态谓词，若再套用 `unsupported_high_torque` 的高扭矩门控，会把大量真实发生的低幅恢复动作排除出审计，造成规则未命中的伪信号。

审计脚本同时兼容在线结构特征和回放 CSV 字段。`torso_z` 可由 `torso_z`、`rootz` 或 `qpos_rootz` 读取；`support_contact` 可由 `support_contact` 或 `foot_grounded` 读取。该处理不改变在线 prior，仅保证 eval trace 与 retrospective replay trace 在行为审计中使用相同语义。

该规则的 confidence 更新仍沿用一般 rule-effect confidence 机制。若规则被审计命中且方向一致率不足以超过正向增强阈值，但 harmful rate 也低于削弱阈值，则记录为 `effect_observed_no_large_change`，保持当前信心；若充分命中且相反方向比例升高，则按一般 harmful 规则削弱或退出。

### 阶段式效果规则

为避免将阶段性控制原则过度压缩为单个动作维度，新增 `stage_effect_prior` 规则类型。该规则类型的基本结构为：

```text
rule_key: 当前结构状态谓词
intervention_type: stage_effect_prior
stage_goal: 阶段目标
horizon: 短视界长度
stage_objective: 终端状态、进展量和窗口均值的加权目标
confidence / status / effective_prior_scale: 沿用现有 confidence 机制
```

首个实现目标为 `recover_full_foot_support`。默认目标函数由三部分组成：

```text
terminal_weights:
  + full_foot_support
  + support_contact
  + torso_z_quality
  - rotation_penalty
  - body_collision

progress_weights:
  + delta full_foot_support
  + delta support_contact
  + delta torso_z_quality
  - delta rotation_penalty
  - delta body_collision

mean_weights:
  + mean full_foot_support
  + mean support_contact
  - mean rotation_penalty
  - mean body_collision
```

在线执行时，规则不再读取 `action_family`，而是对 MPPI 采样得到的整段候选动作序列进行短视界推演。若当前状态满足 `rule_key`，则对每条候选序列计算阶段目标得分，经中心化和尺度归一化后乘以 `stage_prior_weight`、`effective_prior_scale`，再作为 additive bonus 进入候选排序。该机制仍保持轻量介入：它不替换动作、不修改策略网络，也不改变 TD-MPC2 的主损失。

具体推演后端按可用信息选择。若当前 world model 的动力学核心提供 `_state_reward_features`，则直接使用 checkpoint 内部模型进行短视界推演和结构特征读取；若当前 checkpoint 是普通 TD-MPC2，动力学核心不含 SD 结构特征接口，则回退到本地 `DMControlHopperSDModel` 结构模型，以 Hopper 观测量编码出的结构态作为起点，对候选动作序列执行同一阶段目标评分。该回退保证阶段效果规则可在普通 TD-MPC2 checkpoint 的 eval-only 测试中运行，但其含义是“结构模型先验评分”，不是“已训练 TD-MPC2 latent model 自动具备 SD 解释接口”。

审计口径也相应从“单步动作方向是否一致”改为“命中状态后的实际短视界轨迹是否完成阶段目标”。若实际轨迹在 `horizon` 步后的阶段分数高于 `audit_positive_threshold`，则记为有效；若低于 `audit_harmful_threshold`，则记为有害；否则记为中性。审计输出仍映射为 `direction_consistent_rate` 和 `direction_harmful_rate`，以便复用现有 confidence 更新、缺席衰减和硬下线机制。

### toe-only 弹射陷阱规则

在最终评估轨迹中，脚尖强支撑、高即时奖励片段表现出稳定的短期奖励陷阱特征。因此新增 `toe_only_launch_trap` 谓词：

```text
touch_toe >= 0.80
touch_heel < 0.20
support_contact >= 0.80
full_foot_support < 0.35
vx >= 0.80
torso_z >= 0.55
grounded_upright < 0.50
```

该谓词不以即时 reward 作为在线条件，因为 planner 运行时不直接读取环境即时奖励；其语义是用结构态近似定位“脚尖强支撑、向前弹射、heel 缺失且尚未进入直立准备”的局部状态。对应阶段目标为 `stabilize_toe_only_launch`，优先项为保持 `support_contact`、降低旋转和碰撞、维持高度和姿态控制；`touch_heel` 与 `full_foot_support` 仅作为弱正项。该设计避免把所有脚尖支撑状态都强行解释为“立即恢复全足支撑”，同时仍保留向更稳定支撑过渡的偏置。

## 脚尖恢复阶段规则的触发口径修订

后续审查表明，上述 `toe_only_launch_trap` 谓词不应作为当前在线训练的主触发键。原因是该谓词在 `toe_recovery_low_support` 基础上额外加入 `torso_z >= 0.55` 与 `vx >= 0.80` 等条件，而当前 replay 导出中的 `rootz/qpos_rootz` 字段口径使 `torso_z >= 0.55` 在相关训练回放中全部不成立。结果是阶段规则在 `cycle021` 至 `cycle023` 连续零命中，无法检验阶段效果规则本身。

修订后的阶段效果规则采用：

```text
toe_recovery_low_support -> stage_effect_prior: stabilize_toe_only_launch
```

其中触发谓词恢复为既有 `toe_recovery_low_support`：

```text
touch_toe >= 0.80
touch_heel < 0.50
full_foot_support < 0.35
grounded_upright < 0.50
and (
  grounded_prone >= 0.50
  or torso_z <= 0.72
  or rotation_penalty >= 0.50
  or abs(pitch_rate) >= 3.0
)
```

阶段目标仍使用 `stabilize_toe_only_launch`，即优先保持 `support_contact`、控制旋转和碰撞、维持姿态与高度，并仅弱鼓励 `touch_heel/full_foot_support`。该修订的研究含义是：规则升级只改变“干预表达形式”，从单步动作族偏好改为短视界阶段结果评分；触发条件不再同步收窄，以保证与前一版大量命中的脚尖恢复诊断问题保持同一分析对象。

为避免叠加混淆，生成修正规则库时会移除旧的：

```text
toe_recovery_low_support->soft_prefer:waist_pos
toe_recovery_low_support->stage_effect:recover_full_foot_support
toe_only_launch_trap->stage_effect:stabilize_toe_only_launch
```

当前修正版规则库为：

```text
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json
```

在上一轮 `cycle023` replay trace 上的审计 smoke 中，该规则命中 `56949` 帧，说明触发口径已恢复到可观察状态。该 smoke 不作为性能证据，只作为规则进入闭环训练前的逻辑审查。

由于在线 prior 最终会经过 `rule_prior_max_bonus` 全局裁剪，阶段规则强度由三项共同决定：规则内 `stage_prior_weight`、规则内 `stage_max_bonus`、运行命令中的 `rule_prior_max_bonus`。若 `stage_max_bonus` 大于全局上限，实际效果仍会被全局上限截断。

## 脚尖恢复规则的结构阶段拆分

后续结构态后果分析表明，统一的 `toe_recovery_low_support` 阶段规则虽然能定位脚尖低支撑问题，但作为在线干预谓词过宽。该谓词同时覆盖可恢复的 `prone/rear` 状态和已经更接近失稳的 `falling_air/toe_support` 状态；前者在短视界内更常表现为支撑保持、旋转降低和跌落风险降低，后者则更常表现为支撑丢失、旋转加剧和进入跌落。

因此，当前方法将该诊断问题拆分为两个结构阶段触发键：

```text
toe_recovery_recoverable_low_support:
  toe_recovery_low_support
  and (grounded_prone >= 0.50 or grounded_rear >= 0.50)

toe_recovery_unstable_low_support:
  toe_recovery_low_support
  and not toe_recovery_recoverable_low_support
```

在线 prior 与 rule-effect audit 使用同一套谓词，避免再次出现训练端和审计端口径不一致。可恢复子阶段对应 active 规则：

```text
toe_recovery_recoverable_low_support -> stage_effect_prior: stabilize_recoverable_toe_support
```

其阶段目标仍以保持可控支撑、降低旋转、避免跌落和碰撞为主，并弱鼓励脚跟接触和全足支撑恢复。失稳子阶段对应 observing 规则：

```text
toe_recovery_unstable_low_support -> stage_effect_prior: observe_unstable_toe_support
```

该规则初始不进入在线动作排序，只在闭环审计中积累证据。若后续显示该阶段目标可稳定改善结构后果，confidence 机制可以将其增强为 active；若持续表现为有害或缺乏支持，则应保持观察或退休。该设计保留“从 SD 诊断到规则闭环”的主线，同时避免把噪声较大的失稳窗口直接变成在线干预。

## 阶段约束式规则修订

拆分阶段规则的后续审查显示，单纯的阶段成果加权分数仍不够可控。`toe_recovery_recoverable_low_support` 虽然大量触发，但旧目标允许候选序列通过局部姿态指标、脚尖接触或旋转控制获得正分，而不必真正完成脚跟接触、全足支撑恢复和支撑保持。由此产生的现象是：规则被 confidence 审计解释为相当比例有效，但触发后的实际轨迹仍经常出现 `foot_grounded` 下降、`falling_air` 或旋转风险上升。

为使阶段规则更接近“教师型小阶段目标”，`stage_effect_prior` 现在支持 `stage_constraints`。该字段是阶段目标的一部分，不改变规则库和 confidence 生命周期，只改变候选序列的短视界打分。每个约束包含：

```text
metric: 结构态指标名称
scope: initial / terminal / progress / mean / min / max
op: min 或 max
threshold: 约束阈值
penalty: 约束违背惩罚强度
```

在线执行时，普通 `terminal_weights`、`progress_weights` 和 `mean_weights` 仍先计算阶段分数，并可继续执行居中和归一化；随后 `stage_constraints` 的违背惩罚以未归一化形式扣除。这样做的理由是：支撑丢失、跌落和旋转风险不应被候选集合内的相对归一化抵消。最终分数仍受规则内 `stage_max_bonus` 和运行时 `rule_prior_max_bonus` 裁剪，因此该规则仍是轻量 planner prior，而不是硬替换动作。

当前脚尖恢复可恢复子阶段使用以下规则：

```text
toe_recovery_recoverable_low_support
  -> stage_effect_prior: constrain_recoverable_toe_support
```

触发谓词保持不变：

```text
toe_recovery_low_support
and (grounded_prone >= 0.50 or grounded_rear >= 0.50)
```

但在线目标调整为更硬的阶段约束：短视界内 `support_contact` 的最小值和终端值不能过低，`touch_heel` 与 `full_foot_support` 至少应有小幅正进展，`falling_air` 和 `rotation_penalty` 不应越过风险阈值。规则默认 `horizon=3`，与当前 TD-MPC2 在线规划视界一致；若后续提升 TD-MPC2 planning horizon，可再同步提高该规则视界。

审计脚本同步修正 `support_contact` 的字段读取。由于在线结构特征直接提供 `support_contact`，而 replay CSV 中常以 `foot_grounded` 表示接触支撑，`stage_metric("support_contact")` 现在按 `support_contact -> foot_grounded` 的顺序读取。该修订避免把真实脚尖接触误记为阶段目标中的零支撑，也使 confidence 更新基于与状态谓词一致的结构语义。

该修订的实验解释边界是：阶段约束式规则仍不是对 Hopper 的手工控制器。它只在 MPPI 候选排序最后阶段改变候选序列相对分数，并继续接受 rule-effect confidence 的增强、削弱和退休。若重训后触发状态仍无法减少支撑丢失和跌落风险，则应解释为当前谓词、SD 短视界预测或轻量 prior 强度不足，而不是直接把短期回报变化作为唯一判断依据。

## 阶段约束规则的 eval-only 结构烟测方法

阶段约束式规则进入接续训练前，采用 eval-only 结构烟测作为门槛。该烟测不更新 TD-MPC2 checkpoint，不训练 SD explainer，只在已有 checkpoint 的 MPPI 候选排序阶段打开或关闭规则 prior，并用逐 episode trace 检查触发后真实结构态变化。

当前烟测包含三种口径：

```text
rule-off:
  rule_prior_enabled = false

full rule-on:
  rule_prior_enabled = true
  rule_prior_library = hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json

stage-only:
  rule_prior_enabled = true
  rule_prior_library = hopper_185k_cycle037_toe_recovery_stage_constraint_only.json
```

`full rule-on` 用于判断当前完整规则库的实际行为，`stage-only` 用于隔离新增阶段约束规则，避免旧动作族规则和 transition 规则造成归因混淆。三种口径均使用 `eval_episodes=10`、三种子、`horizon=3`、`num_samples=64`、`num_elites=8`、`num_pi_trajs=8`、`iterations=3`。

结构烟测的触发谓词为：

```text
toe_recovery_recoverable_low_support:
  toe_recovery_low_support
  and (grounded_prone >= 0.50 or grounded_rear >= 0.50)
```

触发后检查 `horizon=3` 内的真实轨迹窗口。`expected_good` 与 `rule_constraint_satisfied` 均要求：窗口内最小 `support_contact` 不低于 `0.55`，终端 `support_contact` 不低于 `0.65`，终端 `touch_heel` 相对初始至少增加 `0.05`，终端 `full_foot_support` 相对初始至少增加 `0.03`，窗口最大 `falling_air` 不超过 `0.25`，窗口最大 `rotation_penalty` 不超过 `0.35`。同时记录各分项通过率，用于判断失败来自支撑保持、脚跟/全足进展、跌落风险还是旋转风险。

该烟测的判据不是短期 reward 是否上升，而是规则触发后结构目标是否按预期改善。若 stage-only 相比 rule-off 不能提高 `expected_good_rate`、`rule_constraint_satisfied_rate`，且降低 `min_support_ok_rate` 或提高旋转风险，则不进入接续训练，优先修正规则目标、约束权重或 SD 排序转化机制。

## 约束优先阶段评分

阶段约束式规则在普通软惩罚之外新增约束优先评分。该机制通过 `stage_objective` 中的三个字段控制：

```text
constraint_priority_weight: 基于候选集合内约束违背量排序的相对惩罚权重
constraint_satisfied_bonus: 完全满足全部约束时的附加奖励
constraint_absolute_weight: 原始约束违背惩罚的绝对权重
```

在线评分顺序为：

```text
1. 计算 terminal/progress/mean 软阶段目标；
2. 对软阶段目标按原规则执行居中和归一化；
3. 乘以 stage_prior_weight；
4. 计算每条候选序列的 stage_constraints 违背量；
5. 若 constraint_priority_weight > 0，则按候选集合内违背量从小到大形成相对排序惩罚；
6. 若候选完全满足全部约束，则加入 constraint_satisfied_bonus；
7. 扣除 constraint_absolute_weight * raw_constraint_penalty；
8. 最后仍受 stage_max_bonus 和运行时 rule_prior_max_bonus 裁剪。
```

该机制仍保持轻量 planner prior，不替代 TD-MPC2 动作选择；但其研究意图是使支撑保持、脚跟/全足进展、跌落和旋转约束先于普通姿态软分数发挥作用。当前 `constrain_recoverable_toe_support` 使用：

```text
constraint_priority_weight = 0.18
constraint_satisfied_bonus = 0.03
constraint_absolute_weight = 1.0
stage_prior_weight = 0.04
```

## 在线阶段规则的 SD checkpoint 加载

普通 TD-MPC2 checkpoint 不包含可直接读取 `_state_reward_features` 的 SD 动力学核心。若未额外指定 SD explainer，`stage_effect_prior` 只能回退到本地 `DMControlHopperSDModel` 结构模型。这会造成离线候选审计和在线 eval prior 使用不同 SD 参数，从而削弱规则验证的可解释性。

为统一口径，在线 prior 新增：

```text
rule_prior_sd_checkpoint = <path/to/sd_explainer_cycle_xxx.pt>
```

若该参数存在，`online_rule_prior_v6.py` 会加载 checkpoint 中的 `model_config` 与兼容的 `model_state`，并在 planner 候选评分中使用该 SD explainer。`eval_hopper_v7_rule_prior.py` 在报告中记录：

```text
rule_prior_sd_checkpoint
rule_prior_sd_checkpoint_loaded
```

后续 eval-only 和接续训练若依赖 SD 反事实规则，均应显式传入对应周期的 SD checkpoint；否则只能解释为“默认结构模型 prior”，不能解释为“接续训练后的 SD prior”。

## 单动作/动作族规则线的 300k 接续观察

阶段约束式规则未通过真实 eval 结构烟测前，主线实验回到旧的单动作/动作族规则方案。该方案的核心假设是：规则不直接替代策略，也不作为硬控制器，而是在 TD-MPC2 的 MPPI 候选排序末端施加低强度动作先验；同时，规则库通过每 5k 周期的 SD 回归、反事实归因、规则提取和 rule-effect confidence 审计持续增强、削弱或退休。

当前接续实验从 `cycle040 / 200k` 开始，到 `cycle060 / 300k` 结束。训练、SD 总结和规则更新仍以 `5000` 步为周期，formal checkpoint 和完整 eval 仍以 `25000` 步为周期。这样保留了“短周期总结、长周期评估”的研究口径：规则库可以频繁吸收新轨迹证据，但性能判断不依赖单个短周期的即时回报波动。

在线 prior 仅读取 `confidence_active` 规则，参数为：

```text
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_opposite_penalty = 0.01
rule_prior_cap_penalty = 0.01
rule_prior_preferred_cap = 0.70
```

该设置表示规则只在最终动作候选排序阶段施加轻量惩罚或上限约束，不改变 TD-MPC2 的世界模型训练目标，不开启 SD 在线监督，也不将规则直接写入策略网络。若长期训练后姿态改善，应解释为策略在规则先验影响下逐渐学习到更优行为分布；若 reward 上升但姿态未改善，则需进一步区分回报利用、规则覆盖不足和姿态评价指标不足。

本轮观察重点包括：

```text
1. 250k 与 300k formal eval 是否相对 200k 继续改善；
2. 高分轨迹中是否减少趴地、脚尖急躁弹射和失衡跌倒；
3. 规则命中是否保持健康，即 active 规则是否有触发证据且未被 confidence 审计持续削弱；
4. 新增规则是否仍以动作族和 transition 规则为主，而非误混入阶段约束式规则。
```

## 终点姿态比较审计

为避免以 Hopper 回报作为唯一判断标准，新增 [audit_hopper_posture_comparison.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py)。该脚本读取一个或多个 official eval `episode_csv` 目录，并按实验组聚合三类信号：第一，episode reward 的均值、最大值、最小值与标准差；第二，结构姿态指标，包括 `structural_success_frac`、`dense_pose_high_frac`、`continue_score_ge_060_frac`、`mean_continue_score`、`mean_vx`、`mean_abs_pitch`、`mean_abs_omega` 和 `body_collision_mean`；第三，支撑与动作模式指标，包括 `mean_full_foot_support`、`full_foot_support_gt_035_frac`、`toe_only_low_support_frac`、`toe_only_launch_frac`、`full_support_drive_frac` 以及各阶段占比。

其中 `structural_success` 沿用 v8 审计口径：`torso_z >= 0.75`、`|pitch| <= 0.45`、`vx >= 0.80`、`|pitch_rate| <= 3.50` 且 `body_collision < 0.50`。`continuation_score` 使用姿态直立性、高度、前向速度、旋转控制、支撑准备和无碰撞状态的加权和，用于衡量轨迹是否具备继续形成稳定跳跃的结构条件。`toe_only_launch` 定义为 `touch_toe >= 0.80`、`touch_heel < 0.50`、`full_foot_support < 0.35` 且 `vx >= 0.80`，用于标记脚尖低支撑下的前向弹射模式。

该审计只读轨迹，不改变 checkpoint、规则库或训练过程。其研究作用是将“得分是否上升”拆解为“姿态是否改善、支撑方式是否改善、是否只是更高效利用脚尖弹射”的可复查指标。当前 `350k -> 375k` TVA terminal prior 分析即采用该脚本生成：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/reports/final_official_eval_posture_comparison.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1/cycle_075/reports/final_official_eval_posture_comparison.md
```

## TVA-SD 接口校准审计

为区分 TVA 本身的价值估计能力与其进入 SD 推演链条后的使用误差，新增 [audit_hopper_tva_sd_interface.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_sd_interface.py)。该脚本沿用 TVA 训练中的 `build_tva_targets` 目标构造，并在同一批轨迹起点上比较两类终端估值：

```text
oracle_terminal = V(s_{t+h})
sd_terminal = V(SD(s_t, a_t, ..., a_{t+h-1}))
```

其中 `oracle_terminal` 用于检验 TVA 在真实编码状态上的价值信号，`sd_terminal` 用于检验 TVA 作为 SD rollout 终端补偿时的实际可用性。二者的差异不被解释为 TVA 或 SD 单体失效，而被解释为接口校准问题，即价值网络对 SD 推演状态的局部偏差是否足够稳健。

脚本输出每个视界 `h` 的 Pearson、Spearman、R2、RMSE、`sd_value - oracle_value` 分布，以及 SD 末端状态相对真实末端状态的分量误差：

```text
height_rmse
angles_rmse
velocities_rmse
touch_rmse
state_rmse
```

同时计算 `|sd_value - oracle_value|` 与上述状态误差的相关性，用于判断价值偏移主要来自姿态角、速度、接触信号还是整体状态偏差。该审计是只读诊断，不改变 TVA、SD、TD-MPC2 checkpoint 或规则库。

当前轻量审计默认使用：

```text
horizons = 1,3,5
max_episodes = 20
max_starts_per_episode = 2
device = cpu
```

该设置用于快速判断方向性。若需要形成正式统计结论，应提高 `max_episodes` 与 `max_starts_per_episode`，并固定随机种子复跑。

## TVA 终端先验的接口门控

TVA terminal prior 不再只依赖是否存在 SD/TVA checkpoint 启用。为避免不成熟或接口未校准的 TVA 全局影响 MPPI 候选排序，闭环入口新增可选门控：

```text
--enable-tva-terminal-prior-gate
```

该门控在每个总结周期结束后运行 `audit_hopper_tva_sd_interface.py`，并将结果写入：

```text
cycle_xxx/reports/tva_sd_interface_audit_gate.json
cycle_xxx/reports/tva_terminal_prior_gate.json
```

下一周期训练启动时，只有上一周期 `tva_terminal_prior_gate.json` 的 `passed=true`，才会设置：

```text
TVA_TERMINAL_PRIOR_ENABLED=true
```

否则即使 `--enable-tva-terminal-prior` 已打开，也会在该周期保持：

```text
TVA_TERMINAL_PRIOR_ENABLED=false
```

第一版门控使用 `h=1,3`，不使用 `h=5` 作为上线门槛，因为当前在线 TD-MPC2 规划视界更接近短视界，且前述审计显示 `h=5` 的 SD terminal 价值偏移更不稳定。默认阈值为：

```text
tva_terminal_gate_horizons = 1,3
tva_terminal_gate_min_samples = 30
tva_terminal_gate_min_oracle_pearson = 0.55
tva_terminal_gate_min_sd_pearson = 0.55
tva_terminal_gate_min_sd_r2 = 0.20
tva_terminal_gate_max_sd_rmse_ratio = 0.90
tva_terminal_gate_max_abs_delta_v = 0.0  # 禁用该尺度阈值
```

这些阈值的研究含义为：TVA 在真实末端状态上必须先具备基本价值信号，SD 推演末端上的 TVA 估值也必须优于均值基线，并且误差不能接近或超过基线 RMSE。`max_abs_delta_v` 默认不启用，因为不同周期的 TVA 目标尺度仍存在漂移；待价值尺度归一化进一步稳定后，可再将其作为补充门槛。

该机制只决定 TVA 是否进入在线 planner 排序，不影响 TVA 训练、TVA counterfactual audit、TVA shadow update 或 consequence gate。其作用是把 TVA 的上线时机从经验步数改为可审计阈值。

## TVA 上下文动作竞争审计

为使规则价值判断更贴近“同一状态下应偏向哪个动作族”的问题，新增只读审计 [audit_tva_context_action_competition.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_tva_context_action_competition.py)。该审计读取当前规则库和 `tva_counterfactual_audit.json`，以规则的状态条件作为上下文，在同一上下文内汇总 SD rollout + TVA terminal value 对不同动作族的排序。

该审计不使用 Hopper 特定姿态目标作为规则筛选条件。Hopper 的相位、支撑和姿态标签只用于定位规则上下文和事后解释；规则动作族是否被支持，取决于其在同一上下文内相对于其他动作族、实际动作族和相反动作族的 TVA 终端价值表现。核心指标包括：

```text
best_rate: 该动作族成为同上下文最佳 TVA 动作族的比例
top2_rate: 该动作族进入同上下文前两名的比例
mean_advantage_over_actual: 该动作族相对真实动作的平均 TVA 优势
mean_opposite_margin: 该动作族相对相反动作族的平均 TVA margin
tva_supported / tva_suspect: 基于上述指标形成的只读规则价值标记
```

该设计用于区分两类现象：第一，某条规则在训练或评估中经常触发；第二，该规则的动作族在同一状态上下文下确实比其他候选动作族更有终端价值支持。只有第二类信号才适合作为未来规则维护的候选依据。

在只读审计基础上，新增 [build_tva_context_competition_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_competition_shadow_library.py)。该脚本不修改原规则库，而是生成影子规则库：`tva_supported` 规则获得小幅 confidence 增强，`tva_suspect` 规则按比例削弱并可降至 observing，同一上下文冲突但证据不充分的规则可保守软衰减。该影子库用于 eval-only 或下一轮 shadow-only 对照，不直接进入在线闭环训练。

默认参数为：

```text
support_lr = 0.10
suspect_decay = 0.50
conflict_ambiguous_decay = 0.90
active_threshold = 0.55
retired_threshold = 0.05
demote_suspect_below_active = true
```

该机制的研究含义是：规则价值不再只由出现频率、动作一致性或当前策略回报相关性决定，而由“同一状态上下文内，SD 推演后由 TVA 判断的动作族后果竞争”提供附加证据。该附加证据仍是影子层，需通过 eval-only 行为审计和后续闭环实验验证后，才能考虑进入正式 confidence update。

第一版影子过滤显示，单独依赖 TVA 上下文竞争可能误伤低回报但必要的恢复型规则。因此新增 role/effect-gated 版本 [build_tva_context_role_gated_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_role_gated_shadow_library.py)。该版本把 TVA 竞争结果与真实轨迹 rule-effect 证据联合使用：若规则在真实轨迹中方向一致率高、但主要命中低 good/high-episode 上下文，则标记为 `recovery_consistent`，TVA suspicion 不能单独削弱该规则；若规则命中较多高 good/high-episode 上下文，则标记为 `progress_context`，可更严格使用 TVA 动作族竞争；若真实行为审计显示方向有害或一致性不足，则 TVA suspicion 才能触发削弱。

该门控仍不编码 Hopper 的目标姿态。`recovery_consistent` 与 `progress_context` 来自真实规则命中、方向一致率、好状态命中率和高回报 episode 命中率，而不是来自人工指定的“全足支撑为好”或“prone 为坏”。其作用是防止终端价值信号把“必要的中间恢复动作”误判为无价值，同时保留 TVA 对高回报上下文中动作族冲突的筛查能力。

在 role/effect-gated 之后，新增通用短窗口后果门控 [audit_rule_shadow_short_window_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_rule_shadow_short_window_consequences.py) 与 [apply_rule_shadow_consequence_gate.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/apply_rule_shadow_consequence_gate.py)。前者比较源规则库与任意 shadow 规则库，找出 confidence 发生变化的规则，并在真实轨迹中统计规则同向动作、相反动作和中性动作后续若干步的结构后果；后者只保留通过短窗口后果检查的 shadow 改动，其余改动回退至源规则库 confidence。

该后果门控的输入可以是 TVA shadow、TVA 上下文竞争 shadow 或其他未来规则价值候选，因此不依赖 Hopper 特定规则类型。默认判据为：期望关系与对照关系均至少具备 `min_relation_count=10` 个真实轨迹样本，且期望方向的短窗口 `outcome_score` 不低于对照方向。对被增强规则，期望方向是规则同向动作；对被削弱规则，期望方向是相反动作。当前 `outcome_score` 仍由通用结构后果近似，包括 continuation score、支撑接触、旋转控制、碰撞、跌落和窗口奖励；它不是因果估计，只作为 shadow 候选进入在线规则库前的后果层过滤。

在 `cycle040 / 200k` 的 TVA context role-gated shadow 上，该门控只保留 `2/7` 条 confidence 改动。后续 eval-only 显示 consequence-gated shadow 的三种子均值为 `124.106`，低于原规则库 `127.866` 和 role-gated shadow `126.738`。因此，短窗口后果门控目前只证明了“能够过滤缺少后果支持的 TVA 改动”，尚未证明“保留的改动足以改善策略行为”。在进入 live confidence update 前，还需要跨周期后果稳定性、行为审计和同上下文动作族冲突归并共同支持。

为支持跨周期稳定性检查，新增 [summarize_rule_shadow_cross_cycle_consequences.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/summarize_rule_shadow_cross_cycle_consequences.py)。该脚本聚合多个短窗口后果审计，对同一规则统计可比较周期数、后果支持周期数、支持率和平均 outcome margin。默认稳定支持条件为：至少 `2` 个 audit 可比较、至少 `2` 个 audit 支持、支持率不低于 `0.67`，且平均 margin 不低于 `0`。该条件不是最终上线标准，而是防止单周期偶然后果信号直接进入 live update 的最低稳定性筛查。

为支持同一上下文动作族归并，新增 [build_tva_context_consolidated_shadow_library.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_tva_context_consolidated_shadow_library.py)。该脚本在 `context_key + joint` 层面寻找相反动作族冲突，并用三类通用信号计算动作族分数：

```text
score =
  tva_weight * TVA context competition score
  + effect_weight * real rule-effect score
  + consequence_weight * short-window consequence score
```

默认权重为 `0.45 / 0.35 / 0.20`。TVA 分数来自同上下文动作族的 `best_rate`、`top2_rate`、`mean_opposite_margin` 以及 supported/suspect 标记；effect 分数来自真实轨迹中的方向一致率、有害率和命中数；consequence 分数来自短窗口 outcome margin。只有当赢家分数优势超过 `min_score_gap=0.12`、赢家有至少 `50` 个真实 selected-gate hits、TVA context records 至少为 `20`，且 loser 不是受保护的 `recovery_consistent` 规则时，才在影子库中削弱 loser。默认不增强 winner。

该归并机制仍不编码 Hopper 姿态目标；它只回答“在同一结构化上下文中，同一关节的相反动作族是否有足够证据支持其中一个”。输出仍为 shadow/eval-only 规则库。`cycle040` smoke 中，该机制仅使一条 observing 规则 `ankle_pos` 从 `0.481741` 降至 `0.240870`，active 规则数不变，说明当前默认阈值保持保守。

### TVA 终端先验的强/弱分层门控

`hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1` 在 `75k` 前后显示，TVA-SD 接口审计已经多次出现方向性相关信号，但未能同时满足强门控中的 R2 与 RMSE ratio 要求。因此，门控被修订为分层形式：

```text
strong: 正常 TVA terminal prior，weight=0.02, max_bonus=0.05
weak:   低权重方向性 TVA terminal prior，weight=0.005, max_bonus=0.01
off:    TVA 继续训练和审计，但不进入在线动作排序
```

强门控仍沿用原阈值，要求 `h=1,3` 均满足 `oracle_pearson >= 0.55`、`sd_pearson >= 0.55`、`sd_r2 >= 0.20`、`sd_rmse_ratio <= 0.90`。弱门控用于“已有方向性证据但幅值校准尚不稳定”的阶段，默认采用：

```text
tva_terminal_weak_gate_horizon_mode = any
tva_terminal_weak_gate_min_samples = 30
tva_terminal_weak_gate_min_oracle_pearson = 0.50
tva_terminal_weak_gate_min_sd_pearson = 0.50
tva_terminal_weak_gate_min_sd_r2 = 0.00
tva_terminal_weak_gate_max_sd_rmse_ratio = 1.00
tva_terminal_weak_prior_weight = 0.005
tva_terminal_weak_prior_max_bonus = 0.01
```

门控报告中的 `passed` 表示下一周期是否允许 TVA terminal prior 以任一模式上线；`strong_passed` 和 `weak_passed` 分别记录强门控与弱门控是否通过；`decision.mode` 记录实际采用 `strong`、`weak` 或 `off`。该设计避免将未完全校准的 TVA 直接按正常权重纳入 planner，同时也避免在已有方向性信号时完全退化为 rule-only。

## Hopper 底层状态标签校正

`grounded_prone` 不再作为所有“接地、前倾、未达到严格 upright/prepared”的兜底标签。对 `175k` seed1 高分轨迹的视频复核表明，标准 hop 的屈膝蓄力和前倾推进阶段会频繁满足旧 `prone_geometry` 条件，但视觉上并非趴地失败。因此，底层状态 schema 增加：

```text
grounded_hop
```

其条件为：

```text
grounded_any
no_body_collision
height >= 0.55
abs(pitch) <= 1.10
vx >= 0.75
```

为兼容旧 CSV 中的 `dominant_state_id=3/4/5`，新增状态采用追加编号，而不改变旧编号：

```text
0 falling_air
1 grounded_rear
2 grounded_prone
3 grounded_upright
4 grounded_prepared
5 rising_air
6 grounded_hop
```

语义上，`grounded_hop` 位于 `grounded_prone` 与 `grounded_upright` 之间，用于表示正常 hop 中的接地推进、屈膝支撑和前倾蓄力阶段；在实现中，它保留为追加状态以避免破坏旧日志解释。`grounded_prone` 被收缩为不满足 `grounded_hop/upright/prepared/rear` 的前向接地失败或恢复状态。

该修订的研究含义为：规则生成仍然是自动完成的，但自动规则依赖的状态语言得到校准。后续失败归因、transition 规则、rule-effect 审计和姿态报告不应再将高 `grounded_prone_frac` 直接解释为趴地前行；应同时报告 `grounded_hop_frac`，并区分“正常 hop 前倾支撑”和“失败 prone/rear trap”。已有旧轨迹 CSV 不会自动重写；若需要用新标签审查旧 checkpoint，应重新导出轨迹或用结构模型从 CSV 中的 qpos/qvel/state 字段重算标签。

## 教师核心规则初始化

教师核心规则初始化用于检验结构化显式世界模型能否将高质量经验轨迹转写为少量、可审计、可迁移的动作先验。该机制不替代纯在线自学习规则线，而作为独立实验口径保存。其核心原则为：只从已确认具有标准 Hop 姿态的高回报轨迹中提取少量高可信规则，不将全部教师行为克隆为规则库。

当前实现为 [build_hopper_teacher_core_rules.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_teacher_core_rules.py)。输入为一个或多个 official eval `episode_csv` 目录；脚本先按 episode 回报筛选教师轨迹，再按行级姿态和运动质量筛选帧，最后在 `substate_key -> action_family` 层面统计方向稳定性。默认兼容现有 Hopper v8 规则先验，规则键采用：

```text
unsupported_high_torque | omega_bin | vx_bin | support_bin | phase
```

动作族采用现有四关节符号：

```text
waist_pos / waist_neg
hip_pos / hip_neg
knee_pos / knee_neg
ankle_pos / ankle_neg
```

筛选标准包括：同一规则至少在 `min_episodes` 个教师 episode 中出现，支持帧数不低于 `min_support`，同向动作占同向/反向动作总数的比例不低于 `min_directional_share`，反向比例不高于 `max_opposite_share`。同一 `substate + joint` 下只保留方向稳定性最高的动作族，以避免把相反动作同时写入教师核心库。输出规则均标记为 `status=confidence_active`，但额外记录 `rule_origin=teacher_core_standard_hop`、`teacher_source=hopper_high_quality_official_eval`、`teacher_episode_support`、`directional_share` 和 `opposite_share`。

为兼容现有在线 prior，输出规则的 `source` 字段暂保持为 `v4_substate`。该字段是工程兼容字段，不表示规则来源于失败归因自学习；真实来源由 `rule_origin` 与报告文件说明。使用该库的实验应命名为 `teacher_core_init` 或同义标签，不应与 `fromscratch` 纯自学习规则混记。

当前已生成两个库：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_hop_only_confidence.json
```

其中宽核心库保留 `24` 条规则，覆盖 `hop/prone/falling` 等教师高分轨迹中的多个阶段，适合测试教师初始化是否能更早引导行为分布进入标准 Hop；`hop_only` 库保留 `12` 条 `phase=hop` 稳态规则，可信度更高但早期触发率低，适合作为标准姿态锚点或后续混合库中的高置信核心。

## 局部动作效果优势空间

能力入口效果空间的第一版方向学习使用 `delta_z(s,a)`，其中 `z` 由 SD 短视界推演和 TVA 终端价值共同构造。后续稳定性审计显示，`delta_z` 同时包含动作后果、状态基线和周期尺度差异，因此一个周期内可解释的方向并不必然能迁移到另一个周期。

为更接近动作选择的第一性对象，新增 [audit_hopper_ability_entry_local_advantage_space.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_local_advantage_space.py)。该脚本将效果向量改写为同一状态、同一探针动作集合下的局部优势：

```text
A_z(s, a_actual) = delta_z(s, a_actual)
                 - mean_{a' in probe_bank} delta_z(s, a')
```

其中 `probe_bank` 仍为真实动作、零动作，以及各动作维度在 `0.35 / 0.70 / 1.00` 幅度下的正负单关节探针。该定义的研究含义是：不再直接问某个动作把状态推进到了哪里，而是问在同一状态可选动作集合中，真实动作相对候选动作平均水平带来了怎样的局部后果优势。随后仍只使用 `mean_diff` 与 `ridge_lda` 两类线性方向，并继续采用 `low_entry / high_entry` 分层、周期内 robust median/IQR 归一化和跨周期 transfer/leave-one 稳定性检查。

当前全量审计使用 `474` 条去重窗口，跳过数为 `0`。结果显示，局部优势空间改善了 `high_entry` 的方向一致性，但仍未达到可上线标准：`ridge_lda` 在 `high_entry` 中 transfer 正 margin 比例为 `1.000`，AUC 不低于 `0.60` 的比例为 `0.500`，平均 AUC 为 `0.576`；在 `low_entry` 中 transfer 正 margin 比例为 `0.500`，AUC 不低于 `0.60` 的比例为 `0.083`，平均 AUC 为 `0.482`。`mean_diff` 与 `ridge_lda` 均没有产生 stable tier。

因此，局部动作效果优势空间目前被记录为诊断层，而不进入 online prior 或 shadow prior。该结果提示，问题不只是状态基线混入，而是“能力入口”本身不是单一全局线性效果方向：早期逃逸、恢复、推进和姿态维持在不同状态流形上可能要求不同的局部价值判断。

下一步更合适的数学对象应从“方向分类”转为“局部反事实遗憾”。候选形式为：

```text
U_h(s, a) = Phi(SD_h(s, a))
Regret_h(s, a) = max_{a' in probe_bank} U_h(s, a') - U_h(s, a)
Adv_h(s, a) = U_h(s, a) - mean_{a' in probe_bank} U_h(s, a')
```

其中 `Phi` 应优先由 TVA 或后续学习到的终端价值势函数提供，而不是由 Hopper 特定姿态硬编码。规则只有在同一结构化上下文中，某个动作族持续表现出低遗憾、高优势、低最优动作熵，并且跨周期后果稳定时，才可进入候选规则库。该形式仍保持 `context_key -> prefer_action_family` 或 `context_key -> prefer_low_regret_action` 的简洁维护结构，把复杂性限制在离线价值审计层。

## Phi/Regret 势函数审计

为检验 TVA 是否可以作为局部反事实遗憾中的价值势函数，新增 [audit_hopper_phi_regret_potential.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_phi_regret_potential.py)。该脚本仍只读已有入口窗口、SD explainer 和 TVA checkpoint，不生成规则、不修改在线 prior。

审计对象定义为：

```text
Phi(s) = TVA(s)
U_h(s, a) = Phi(SD_h(s, a))
Regret_h(s, a) = max_{a' in probe_bank} U_h(s, a') - U_h(s, a)
Adv_h(s, a) = U_h(s, a) - mean_{a' in probe_bank} U_h(s, a')
```

其中 `probe_bank` 沿用既有单关节正负探针集合，默认 horizon 为 `1 / 3 / 5`。审计分三层输出：

1. `phi_state_calibration`：检查当前 `Phi(s_t)` 是否与真实未来窗口奖励、前向位移、前向速度和 episode 回报同向；
2. `actual_action_separation`：检查真实动作经 SD 推演得到的 `U_h / Adv_h / Regret_h` 是否能区分 entry success 与 matched failure；
3. `family_regret_contexts`：在 `tier + context_key` 内聚合 TVA 最优动作族，报告 best-family entropy、跨周期最小 best rate 和观察性后果差异。

当前全量审计使用 `474` 条去重入口窗口，跳过数为 `0`。结果显示，TVA 作为 `Phi(s)` 对低阈值入口和 episode 级好坏具备排序信号：全体窗口中 `Phi(s)` 与 episode reward 的 Spearman 为 `0.607`，与未来前向位移的 Spearman 为 `0.376`；`low_entry` 中与未来前向位移的 Spearman 为 `0.419`。但 `high_entry` 中 `Phi(s)` 与短窗口后果关系较弱甚至为负，说明高阈值入口不应直接按同一势函数解释。

经 SD 推演后的 `U_h / Regret_h` 仍未形成稳定规则信号。三个 horizon 下，entry success 相对 matched failure 的实际动作优势 AUC 均接近随机：`h=1` 为 `0.502`，`h=3` 为 `0.481`，`h=5` 为 `0.469`；低遗憾 AUC 分别为 `0.481 / 0.460 / 0.434`。上下文聚合中没有任何候选满足默认稳定门槛。部分上下文具有较高总体 best-family rate，例如 `h=3` 下 `omega_neg|vx_mid|support_none|falling|support_risk -> hip_pos` 的 best rate 为 `0.889`、entropy 为 `0.388`，但跨周期最小 best rate 为 `0.0`，且该动作族在真实观察动作中几乎不出现，因此不能转为规则。

据此，当前 `Phi=TVA` 的价值势函数可作为诊断信号，但尚不能直接驱动低遗憾规则生成。后续若继续推进，应先改进两点：一是对 `high_entry` 进行单独校准或使用更局部的上下文势函数；二是要求候选动作族不仅在 TVA 反事实中低遗憾，还要在跨周期和真实后果审计中稳定。

## TVA 效果空间参数化筛选与核方法审计

为避免继续增加规则形态复杂度，新增 [audit_hopper_tva_effect_space_svm.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_tva_effect_space_svm.py)。该脚本只测试“TVA 效果空间中是否存在可由简单参数训练或核方法发现的稳定结构”，不修改规则库、不接入在线 planner。

每个窗口的实际动作或候选动作族被表示为：

```text
z_phi(s,a) = [
  Phi(s),
  U_h(s,a),
  Adv_h(s,a),
  -Regret_h(s,a),
  U_h(s,a)-Phi(s)
] for h in {1,3,5}
```

其中 `Phi=TVA`，`U_h(s,a)=TVA(SD_h(s,a))`。审计模型包括：

```text
logistic_l2
linear_svm
rbf_svm
```

`logistic_l2` 与 `linear_svm` 用于测试是否能形成可解释的线性规则筛选器；`rbf_svm` 仅作为非线性可分性的上限诊断。验证采用 leave-one-cycle：在同一 tier 的其他周期训练，在目标周期测试；同时报告 global leave-one-cycle 作为补充。模型只学习区分 entry success 与 matched failure 的实际动作效果特征，随后在 held-out 周期上对所有候选动作族评分，并聚合为 `tier + context_key -> top_family` 候选。

当前全量审计使用 `474` 条入口窗口，跳过数为 `0`。结果显示，参数训练和核方法均未发现稳定结构。global leave-one-cycle 平均 AUC 分别为：`logistic_l2=0.322`，`linear_svm=0.320`，`rbf_svm=0.347`；`low_entry` 中分别为 `0.284 / 0.283 / 0.376`；`high_entry` 中分别为 `0.332 / 0.305 / 0.304`。没有任何模型产生 stable-like context。

因此，当前数据不支持将 TVA 效果空间通过参数训练或核 SVM 直接转化为规则筛选器。RBF SVM 没有显著优于线性模型，说明问题不是简单的线性投影不足，而更可能来自当前入口标签、上下文划分或 TVA 势函数目标与短窗口能力入口之间的错配。

## 能力入口标签与上下文基础审查

为回到更基础的数据口径，新增 [audit_hopper_entry_label_context_foundations.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_entry_label_context_foundations.py)。该脚本不调用 SD、TVA 或任何新模型，只读取 `entry_candidate_segments.json`，审查两类问题：

1. `entry_success / matched_failure` 标签是否在真实后续窗口中具有可分性；
2. 当前 `context_key` 是否足以形成跨周期稳定的动作族偏好。

当前审查发现，原始入口窗口共 `568` 条，但唯一窗口标识只有 `413` 个，重复率为 `0.273`；其中 `29` 个窗口在不同阈值事件中出现角色冲突，即同一窗口可同时被标为某个入口的 `entry_success` 和另一个入口的 `matched_failure`。此外，`entry_success` 中有 `42.0%` 来自入口前一周期，因为原入口追踪工具为观察能力形成过程而纳入了入口前后各一个周期的 top episode。

这些设计对诊断能力形成过程有价值，但不适合直接训练规则筛选器。消融结果显示，标签严格化后后续后果信号显著增强：

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

因此，后续若继续使用能力入口数据生成规则，应采用更严格的标签口径：只使用入口当周期或入口后周期，去除角色冲突重复窗口，并避免同一窗口因不同阈值事件被重复加权。

上下文审查显示，当前 `context_key = omega_bin + vx_bin + support_bin + phase + risk_bin` 仍过粗。在最少 `12` 条样本的上下文中，`16/16` 个原始上下文被标记为 ambiguous；使用 `refined_phase` 后仍有 `13/15` 个上下文 ambiguous。主要原因是同一上下文经常横跨多个周期和多个阈值，角色熵高、动作族熵高，且 dominant action family 的跨周期最小命中率常为 `0`。这解释了为什么线性模型、局部遗憾和 RBF SVM 都难以产生稳定规则：它们面对的不是单一机制，而是混合上下文。

当前基础审查的结论是：能力入口标签有真实信号，但需要严格化；当前上下文语言不足以承载动作族规则。后续应优先修订入口标签和上下文定义，而不是继续增加模型复杂度。

## 严格入口标签数据集

为检验修订标签体系是否足以恢复数学规则信号，新增 [build_hopper_strict_entry_segments.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_strict_entry_segments.py)。该脚本从原始 `entry_candidate_segments.json` 派生严格入口数据，不覆盖旧数据。当前生成两版：

```text
entry_candidate_segments_strict_no_pre_entry.json
entry_candidate_segments_strict_entry_cycle_only.json
```

其中 `no_pre_entry` 保留入口当周期和入口后一周期；`entry_cycle_only` 仅保留入口当周期。两者均移除 success/failure 角色冲突窗口，并在同一窗口跨阈值重复出现时只保留一个最高阈值实例。样本量由原始 `568` 条窗口变为：

```text
no_pre_entry:     269
entry_cycle_only: 135
```

严格标签增强了真实后果可分性：`no_pre_entry` 的 future reward AUC 为 `0.714`，future rootx AUC 为 `0.799`；`entry_cycle_only` 分别为 `0.712` 和 `0.800`。但在这些严格数据上重跑 TVA 效果空间 SVM 后，仍没有恢复可用规则信号，global AUC 仍约为 `0.24-0.40`，stable context 数为 `0`。

局部动作效果优势空间有所改善但未通过稳定标准。`no_pre_entry` 下，`ridge_lda / low_entry` 的 transfer AUC 从原始口径的 `0.482` 提升到 `0.536`，`ridge_lda / high_entry` 从 `0.576` 提升到 `0.601`；但两者仍没有形成 stable tier。`entry_cycle_only` 中 high entry 有轻微提升，low entry 因样本减少未改善。

因此，严格入口标签可以减少噪声，但不能单独解决规则提取问题。当前剩余瓶颈仍是上下文定义过粗，同一 `context_key` 下混合多个局部控制机制。后续若继续推进，应在严格标签基础上修订上下文定义，再重跑轻量审计。

## 严格入口上下文变体审计

为检验上下文修订是否能恢复规则形态候选，新增 [audit_hopper_strict_context_variants.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_strict_context_variants.py)。该脚本仍保持只读，不使用新分类器，不写入规则库。它在严格入口数据上比较多种上下文定义，包括原始 anchor phase、anchor refined phase、加入 anchor type、窗口主导 refined phase，以及由支撑、跌落、旋转和姿态幅度构成的低维机制上下文。输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_strict_context_variant_audit_no_pre_entry.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_strict_context_variant_audit_entry_cycle_only.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/strict_context_variant_summary.md
```

审计标准仍对应简洁规则形式：

```text
context_key -> prefer_action_family
```

`stable-like` 只表示某一上下文在成功侧动作族集中、失败侧同动作族较少、后续奖励和位移差为正、并且成功样本至少跨两个周期出现；它不是上线规则标准。

`no_pre_entry` 口径下出现一条主要候选：

```text
low_entry::omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward
  -> prefer waist_neg
```

该候选包含 `11` 条窗口，成功 `6` 条、失败 `5` 条；成功侧 `waist_neg` 命中率为 `1.000`，失败侧为 `0.000`，动作族 AUC 为 `1.000`，后续奖励差为 `+6.818`，后续 root-x 差为 `+0.364`，成功样本横跨 `cycle007/017/018`。较窄的 `entry_cycle_only` 口径没有产生 stable-like 候选，说明入口后一周期对当前能力入口规则审计仍有必要。

随后使用 [audit_hopper_ability_entry_counterfactuals.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_ability_entry_counterfactuals.py) 在 `no_pre_entry` 严格数据上重跑 SD/TVA 反事实审计，输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_counterfactual_audit_strict_no_pre_entry.json
```

反事实检查显示，该候选尚不能直接转为规则。成功侧 `waist_neg` 的 SD top-3 支持为 `2/6`，TVA top-3 支持为 `4/6`；失败侧 `waist_neg` 的 TVA top-3 为 `0/5`，但 SD top-3 为 `3/5`。因此，当前结论是：严格标签与上下文修订能够找回候选规则形态，但仍未形成足够强的 SD/TVA 共同支持。后续应围绕该类候选做局部后果与模型分歧诊断，而不是继续增加分类器复杂度。

该反事实结论需要加入模型成熟度限定。按周期分层后，早期 `cycle007` 的成功样本虽表现出 `waist_neg` 的统计后果优势，但 SD/TVA 均未将其排入 top-3；到 `cycle017/018` 后，TVA 对成功侧 `waist_neg` 的 top-3 支持为 `4/4`，`cycle018` 中 SD 和 TVA 均将其排为最优。因此，早期 SD/TVA 的不支持不应被视为强反证。后续入口规则审计应区分早期弱模型证据和后期较成熟模型证据，必要时将反事实证据按周期成熟度加权。

### 效果空间规则的 shadow confidence

为使效果空间规则与动作规则维护机制保持一致，新增 [audit_hopper_effect_confidence_shadow.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/audit_hopper_effect_confidence_shadow.py)。该脚本仍为离线审计，不写入在线规则库。规则形式为：

```text
context_key -> prefer_effect_direction
confidence_t -> confidence_{t+1}
```

脚本在严格 `no_pre_entry` 入口数据上重算 SD/TVA 短视界效果向量，并对每个上下文累计估计 success-minus-failure 效果方向。随后按训练周期回放证据，使用中性初值 `0.50` 更新 shadow confidence：

```text
confidence = 0.5 + (confidence - 0.5) * decay
           + learning_rate * maturity_weight * evidence_strength * (evidence_score - 0.5)
```

其中 `maturity_weight` 根据周期从早期低权重逐步升高，用于表达早期 SD/TVA 尚未充分训练；`evidence_score` 来自同上下文内 success/failure 的效果方向 alignment AUC、alignment margin，或单一角色窗口中的弱 alignment 证据。默认参数为：

```text
initial_confidence = 0.50
learning_rate = 0.30
decay_to_neutral = 0.98
maturity_floor = 0.20
context_variant = anchor_refined_type
horizon = 5
```

当前输出为：

```text
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_confidence_shadow_strict_no_pre_entry.json
results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_effect_confidence_shadow_strict_no_pre_entry.md
```

严格 `no_pre_entry` 数据共 `269` 条窗口，跳过数为 `0`，生成 `3` 条 shadow 效果空间规则候选。最高信心值为 `0.633`，其次为 `0.626` 和 `0.544`。前述 `omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward` 候选的信心从 `0.50` 温和升至 `0.626`，其后期 `cycle017/018` 成功窗口带来了主要增强；早期 `cycle004/007/008` 因 maturity weight 较低，影响较弱。

该结果支持“效果空间规则也可维护信心值”的设计，但当前仍是 shadow-only。原因是部分候选虽然 alignment 证据较强，但诱导出的 best effect family 熵较高，说明效果方向尚未稳定压缩为明确动作选择。后续若要接入在线 prior，应继续要求：信心值稳定超过阈值、跨周期不依赖单一偶然窗口、诱导动作族不过度发散，并通过真实后果审计。

## Hopper 姿态标签校准口径

高质量正式评估轨迹被用作姿态标签校准锚点。该用法的依据是：校准目标不是从高质量轨迹中直接手工写入任务规则，而是检查底层状态标签是否与已观察到的稳定 hop 行为一致。若标签不能识别明显的 hop 片段，后续自动规则归纳、效果空间审计和姿态对比都会受到系统性噪声影响。

当前 `grounded_hop` 的最小定义为：接地、无躯干碰撞、高度不低于 `STATE_HOP_HEIGHT_MIN = 0.55`、绝对俯仰角不高于 `STATE_HOP_PITCH_ABS_MAX = 1.10`、前向速度不低于 `STATE_HOP_VX_MIN = 0.75`，并且不属于 `grounded_upright` 或 `grounded_prepared`。该定义只用于区分已形成的前向接地跳跃片段，不试图完整描述所有标准 hop 技术细节。

正式评估摘要的 `_gt_05` 字段采用原始特征阈值生成：

```text
feature_gt_05 = 1.0 if feature >= 0.5 else 0.0
```

该口径替代先前依赖 `dominant_state_id` 的摘要写法。原因是 dominant state 是互斥主状态，适合粗略阶段统计，但不适合作为所有布尔特征的唯一来源；尤其在 hop 状态未被分配编号时，会导致摘要与原始逐帧特征不一致。

审计和规则匹配中的阶段优先级调整为先识别 `hop`，再识别 `grounded_prone`。这并不表示前倾接地一定是良性状态，而是承认在 Hopper 中，高质量 hop 周期也可能满足旧的 prone 几何条件。因此，`grounded_prone` 在当前文档中解释为“前倾接地支撑”，不得单独作为失败标签；`rotation_penalty` 也不得单独作为失败标签，必须结合支撑、位移、后续跌倒或 TVA/SD 后果判断使用。

旧实验轨迹中的原始 CSV 已包含可重算特征，因此可用于修正后审计。旧正式评估 summary CSV 中由旧逻辑生成的 `grounded_hop_frac` 不再作为可靠汇总值；若需要摘要层面对比，应重新运行正式评估，或从 episode CSV 按上述阈值重新计算。

修正后审计与历史训练之间需要区分两个口径。历史 200k 训练实际使用旧阶段解释，因此其训练效果不能直接归因于修正后标签；修正后审计只用于重新解释已有轨迹和准备后续实验。后续若重新训练，`online_rule_prior_v6.py` 中的阶段匹配也会采用 hop 优先级，因此同一规则库的在线命中分布可能与历史训练不同。

## 修正标签下的规则质量分层

修正标签后，规则质量审查不再只看命中数量，而按以下维度分层：

```text
rule_origin: teacher_core_standard_hop / closed_loop_or_legacy
source: v4_substate / v8_transition_replay_safety
phase interpretation: explicit_prone_recovery / hop_cycle / mixed_hop_air_cycle / mixed_or_recovery
direction health: healthy_direction / direction_suspect / mixed_or_low_support / inactive
```

其中 `explicit_prone_recovery` 表示规则在修正阶段口径下仍几乎全部命中 `prone` 帧；`hop_cycle` 和 `mixed_hop_air_cycle` 表示规则主要覆盖 hop、rising 或 falling 周期。`direction_suspect` 暂定义为 selected hit 不少于 `100` 且 harmful/opposite 比例不低于 `0.45`。该阈值只作为审计筛选标准，不直接决定规则上线或下线。

当前 200k `cycle040` 审计结论为：显式 `prone` 规则仍主要是恢复或前倾接地支撑规则，不应简单视为正常 hop 规则；`v8_transition_replay_safety` 规则虽然数量少，但与高分 hop 周期更一致，是后续规则保留和重训初始化时的优先候选；方向可疑规则需要重新归因或降权，不应在修正标签口径下直接作为强先验。

## 信心校准的随机规则门控

规则来自有限回放分布和 SD/TVA 归因证据，因此即使规则信心较高，也不应被解释为确定性命令。为减少多种子训练中的早期分布锁定风险，在线规则先验新增可选随机门控：

```text
p_apply(rule) = p_min + (p_max - p_min) * confidence^gamma
```

其中 `confidence` 使用规则的 `effective_prior_scale`，若缺失则使用 `confidence` 字段，并被限制在 `[0, 1]`。`p_min` 保证低信心规则仍有稳定的基础启用比例，`p_max` 保证高信心规则也不会完全消除探索，`gamma` 控制中等信心规则的保守程度。

该机制只改变规则是否在某次 MPPI 候选排序中生效，不改变规则库、不改变策略网络、不直接替代动作。若规则被采样启用，其原有 soft prior 仍按 `effective_prior_scale` 缩放；若被 dropout，则本次排序中不施加该规则。正式评估默认不进行随机采样，而使用期望缩放：

```text
effective_scale_eval = confidence * p_apply(rule)
```

这样训练阶段保留探索，评估阶段避免因随机门控引入额外方差。若需要研究随机 eval 本身，可显式开启 `rule_prior_stochastic_eval`。

当前新增参数及默认值为：

```text
rule_prior_stochastic_gate = false
rule_prior_min_apply_prob = 0.40
rule_prior_max_apply_prob = 0.98
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

默认关闭，因此旧实验命令和旧结果口径保持可复现。当前默认概率采用保守门控口径：教师规则和高信心规则通常保持较高启用率，新上线或边界信心规则保留更多探索空间。开启后，日志额外记录 `rule_prior_matched_rules`、`rule_prior_applied_rules`、`rule_prior_dropped_rules`、`rule_prior_apply_prob_mean` 和 `rule_prior_apply_multiplier_mean`，用于检查规则利用与探索比例是否符合预期。

## 修正标签初始规则库

修正标签后的下一轮实验不直接复用 `cycle039` 全量规则库，而使用紧凑初始库：

```text
library_corrected_label_initial_v1.json
```

构建标准为：

```text
keep_transition:
  source == v8_transition_replay_safety
  health != direction_suspect
  selected_gate_hits > 0

keep_teacher_core:
  rule_origin == teacher_core_standard_hop
  health == healthy_direction
  selected_gate_hits > 0
```

所有保留规则被设为 `confidence_active`，但其 `confidence` 和 `effective_prior_scale` 保留源规则库中的值。这样，规则库在结构上较紧凑，而信心校准随机门控仍可根据规则既有信心调节使用概率和先验强度。

当前库共 `12` 条规则：`7` 条 transition 安全规则和 `5` 条教师核心健康规则。该库用于验证框架，不声称已经覆盖 Hopper 完整任务规则；其目标是以较低噪声的初始规则测试随机门控和后续自学习更新是否能改善多种子一致性。

## 修正标签冷启动初始规则库

`library_corrected_label_initial_v1.json` 只适合作为低噪声 smoke 库，不应作为完整冷启动先验。后续从零训练应使用冷启动专用库：

```text
library_corrected_label_coldstart_initial_v2.json
```

该库的构建原则为：修正标签口径负责重新解释规则质量，但不直接删除高质量教师轨迹中的入口覆盖。教师核心规则全部保留为冷启动入口候选；若修正标签审计显示健康，则保留原信心；若在成熟评估窗口未命中，则认为缺少成熟窗口证据但保留教师轨迹证据，信心上限设为 `0.90`；若为 `mixed_or_low_support`，信心上限设为 `0.65`；若为 `direction_suspect`，信心上限设为 `0.35`。transition 安全规则只保留修正标签审计中非方向可疑且有 selected hit 的规则。

该库继续使用 `confidence_active`，由 `effective_prior_scale` 同时控制先验强度和随机门控启用概率。这样，规则不会被解释为确定命令；低信心规则仍可进入训练并接受后续 rule-effect confidence 的强化、削弱或退休。动作族必须属于可被 `online_rule_prior_v6` 映射的关节方向族，不能映射的规则不得进入初始库。

当前版本共 `31` 条规则：`24` 条教师核心规则和 `7` 条 transition 安全规则。该库保留 `hop / falling / prone / transition` 覆盖结构，用于修正标签后的从零训练；旧 `v1` 紧凑库保留为审计和 smoke 对照，不再作为正式冷启动默认初始库。

## 无效动作族过滤

Hopper 在线动作先验只支持可映射到具体关节维度和方向的动作族：

```text
waist_pos / waist_neg
hip_pos / hip_neg
knee_pos / knee_neg
ankle_pos / ankle_neg
```

`zero` 表示无主导动作族，不能映射到某个关节方向。若将 `prefer:zero` 作为普通规则保留，在线 prior 中无法产生有效候选重排序，同时还会污染规则命中和启用统计。因此，`online_rule_prior_v6.py` 在加载 action-family 规则时过滤无效动作族，`closed_loop_rule_update.py` 在规则晋升和候选证据保留阶段也使用同一口径。该处理不否定“少动”可能有价值，只是当前单动作族 prior 不能表达“整体降幅”这一复合动作约束；若未来需要表达，应作为新的规则类型单独实现，而不是复用 `prefer:zero`。

## 教师核心规则冲突保护

`rule_origin == teacher_core_standard_hop` 的初始规则被视为教师先验，而不是普通闭环自学习候选。其研究含义是：这些规则来自高质量教师轨迹和先前审计，目的是在训练早期提供稳定入口；新生成规则若来自低预算早期分布，不应在单周期冲突中直接削弱教师规则。

当前冲突保护只作用于 `apply_rule_effect_confidence.py` 的同状态、同关节、相反有效动作方向仲裁。若冲突组中存在教师核心规则，默认不允许普通候选将其降为观察态，除非普通候选同时满足：

```text
selected_gate_hits >= 1000
challenger_confidence - teacher_confidence >= 0.25
teacher_harmful_rate - challenger_harmful_rate >= 0.20
```

上述阈值使反向规则必须具备足够多的实际命中、显著更高信心和显著更低有害率，才能覆盖教师规则。否则，教师规则保持当前信心状态，并记录 `teacher_core_conflict_protected`；反向候选继续按普通冲突逻辑降为观察态或保留为待验证证据。

该保护不阻止教师规则被自身效果审计逐步削弱。若教师规则在多周期中直接表现为高有害率，仍可通过 rule-effect confidence 的有害证据更新而降低信心。保护对象仅是早期冲突仲裁，目的是避免噪声候选过早破坏训练入口。

`closed_loop_rule_update.py` 同步保留教师来源元数据，例如 `rule_origin`、`teacher_source`、`teacher_examples`、`teacher_episode_support` 和 `mean_teacher_episode_reward`。这样，同 ID 规则被周期证据更新时不会丢失教师身份，后续周期的冲突保护仍可生效。

## 教师核心规则的阶段兼容匹配

高质量教师轨迹中的核心规则包含显式阶段标签，例如 `hop`、`prone` 和 `falling`。状态标签修订后，同一物理姿态在在线轨迹中可能被更保守地归入 `prone` 或 `other`，从而导致教师规则虽然仍保留在规则库中，但在在线动作排序阶段命中不足。为区分“规则本身无效”和“规则阶段标签与在线标签语义不一致”，在线先验新增可选参数：

```text
rule_prior_phase_match_mode = current | hop_alias
```

`current` 使用原始精确阶段匹配，是默认口径；`hop_alias` 仅对 `rule_origin == teacher_core_standard_hop` 的五段式 substate 规则生效。当教师规则阶段为 `hop` 时，在线匹配允许当前状态阶段落在 `{hop, prone, other}` 中。该处理不是新的 Hopper 专用行为规则，而是对教师轨迹标签和当前在线标签之间的兼容性诊断；它只扩大已有教师规则的触发条件，不改变动作族、不新增规则、不改变信心更新逻辑。

该参数由 `online_rule_prior_v6.py` 实现，并通过 `run_hopper_v7_multiseed_shared_rules.py`、`run_hopper_v6_5k_rule_prior_smoke.sh` 传入训练。若后续实验显示该模式恢复了较成功线中的早期入口分布，则说明此前退化的重要来源可能是状态标签修订后的规则匹配语义漂移；若不能恢复，则应继续检查候选排序、无效动作过滤和早期采样分布等其他差异。

## grounded_hop 的在线 phase 解耦

`grounded_hop` 最初用于刻画标准 Hop 中离地前的接地推进瞬间。该标签适合作为高质量轨迹审计特征，但早中期训练轨迹复核显示，单帧静态条件不足以区分“稳定起跳前脚尖发力”和“恢复失败中的不稳定前冲”。因此，`grounded_hop` 不再默认覆盖在线规则先验中的 `prone` phase。

当前在线 prior 新增参数：

```text
rule_prior_enable_grounded_hop_phase = false | true
```

默认值为 `false`。默认口径下，SD 仍计算并可导出 `grounded_hop`，但 `online_rule_prior_v6.py` 的 phase 匹配按旧顺序解释：

```text
falling -> rear -> prone -> upright -> rising -> other
```

只有显式启用 `rule_prior_enable_grounded_hop_phase=true` 时，在线 phase 才允许：

```text
falling -> rear -> hop -> prone -> upright -> rising -> other
```

该处理的研究含义是：静态姿态标签与在线规则语境分离。`grounded_hop` 可继续用于诊断、轨迹审计和后续转移上下文规则构造，但在完成跨阶段验证前，不作为默认在线规则匹配标签。后续若需要把该信息用于规则提取，应通过“前序状态 -> 当前候选状态 -> 后续结果”的转移上下文，而不是仅依赖单帧 `grounded_hop`。

## 转移上下文规则候选提取

为避免单帧状态标签混淆不同阶段语义，新增离线候选提取脚本：

```text
experiments/sd_retrospective_rule_induction/extract_hopper_transition_context_rules.py
```

该脚本从已导出的 Hopper replay trace 中读取逐帧原始观测，重新调用 SD 特征函数，构造如下上下文：

```text
past dominant phase -> current candidate phase -> future outcome
```

其中，规则前件只包含可在线观察的过去与当前信息：

```text
prev_phase
current_phase
omega_bin
vx_bin
support_bin
risk_bin
```

`future_outcome` 仅作为规则提取证据，不作为在线匹配条件。未来窗口用于估计候选动作族对应的观察性后果，包括 `controlled_rise`、`support_recovery`、`unstable` 和 `neutral`。转移得分综合短窗内 rising 比例、前向位移、平均前向速度、全足支撑、旋转风险、falling 风险和高角速度风险。

当前输出规则状态为：

```text
transition_context_candidate
```

该状态明确表示观察性候选，不得直接合并到在线规则库。后续上线前必须经过 SD/TVA 反事实验证，确认同一转移上下文内该动作族相对于替代动作确实改善后续状态。

该设计保持规则维护框架简洁：最终候选仍是单动作族规则，但其前件从单帧 phase 扩展为转移上下文。这样可区分“稳定支撑进入起跳前瞬间”和“prone/rotation 风险中不稳定前冲”这两类在单帧上相似、但训练含义不同的状态。

## 高分轨迹教师转移候选库

`teacher_transition_context_v1` 使用已经确认存在较标准 Hop 姿态的高分 official eval 轨迹生成教师候选规则。该库的目标不是直接替换在线规则库，而是提供一组来自高质量行为的转移上下文候选，供后续 SD/TVA 反事实验证。

当前输出目录为：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/
```

其中：

```text
library_teacher_transition_context_candidates.json
library_teacher_transition_context_candidates_loose.json
library_teacher_transition_context_grounded_hop_candidates.json
```

均为观察性候选库。严格版本强调较高支持和优势阈值，宽松版本用于保留更多候选以供后续审查，`grounded_hop` 版本只聚焦起跳前候选状态。三者均不得直接作为 `initial_rule_library` 用于训练；只有通过失败轨迹对照和 SD/TVA 反事实验证后的子集，才可升级为教师规则。

当前版本还显示，已有 SD 静态 phase 与风险标签对高分周期性 Hop 轨迹仍不够贴合：许多高质量轨迹片段会被标记为 `prone/falling/rotation_risk/unstable`。因此，教师候选提取的下一步不是放宽上线门槛，而是校准转移 outcome 与风险评分，使其能识别“周期性 Hop 中的正常高角速度/腾空/落地循环”和“失败恢复中的失控前冲”之间的差异。

后续方法调整以规则稀疏性审查为依据：高分轨迹中的一致性主要存在于周期效果层面，而不是静态上下文内的单动作族局部优势层面。因此，`teacher_transition_context_v1` 不应通过简单降低 `min_context_support`、`min_family_support` 或 `min_score_advantage` 来获得更多候选。更合适的方向是在保持规则最终形式简洁的前提下，重新定义候选动作的未来效果表示，例如使用更长或周期对齐的窗口、TVA 终端补偿、非坍塌的前向推进等指标，使正常 Hop 的腾空、快速角速度和接触切换不被直接等同于不稳定。该调整仍应保持离线候选与在线规则分离，只有通过 SD/TVA 反事实验证的候选才可升级为教师规则。

当前转移上下文提取器支持两种 outcome 口径：

```text
legacy
cycle_v1
```

`legacy` 是原短窗口径，直接惩罚 `rotation_penalty`、`falling_air`、`high_spin` 和低全足支撑。`cycle_v1` 是校准后的周期感知口径，其规则前件保持不变，仍为 `prev_phase/current_phase/omega_bin/vx_bin/support_bin/risk_bin`；变化仅发生在候选动作的未来效果评分。`cycle_v1` 的核心指标为未来窗口内的前向位移、平均前向速度、非坍塌、保持高度和接触周期延续。旋转和高角速度不再单独构成失控证据，只有在伴随碰撞、低高度、低速度或低前向进展时才形成较强惩罚。

该口径的研究含义是：高分标准 Hop 的教师轨迹应被视为周期效果原型，而不是由单帧风险标签逐帧判定。`cycle_v1` 仍是 Hopper 实现层的校准函数，不等同于可泛化的最终方法；未来跨任务版本应将同一思想抽象为“由任务进度、非坍塌/非失败约束和 TVA 终端价值共同定义的效果空间评分”。

## 宽口径候选与反事实验证

为保持方法论框架简洁，当前规则提取采用“宽口径候选、严格验证”的分工。候选生成阶段不追求一次性形成精确规则，而是从高质量轨迹或回放数据中提取覆盖较高、语义清晰的候选经验；候选是否进入规则库由后续 SD/TVA 反事实验证决定。

转移上下文提取器支持两种 context 口径：

```text
full
broad_context_v1
```

`full` 使用 `prev_phase/current_phase/omega_bin/vx_bin/support_bin/risk_bin`，适合诊断和对照审查。`broad_context_v1` 使用 `prev_phase/current_phase/coarse_omega_bin/vx_bin/coarse_support_bin`，去除 `risk_bin` 和细支撑类型，以避免在候选生成阶段过早碎片化。细粒度 `omega/support/risk` 仍保留为诊断修饰项，用于验证阶段判断候选是否需要降信心、排除子区域或分裂。

该方法的原则为：

```text
候选提取：宽、简洁、高召回
反事实验证：严格、因果化、高精度
在线维护：信心增强/削弱，允许后续修正
```

在 Hopper 当前实现中，`broad_context_v1` 是面向周期性 Hop 的具体版本；在更一般的问题中，对应的是先在低维效果空间中提取候选，再用结构化世界模型和价值网络验证动作后果，而不是依赖大量任务特定的细标签直接生成规则。

## 过程无关状态标签

规则提取中的互斥状态标签应描述当前可观测形态，而不应编码“成功 Hop”“失败恢复”或“进入 Hop”等过程语义。过程信息由过去窗口、速度/支撑/角速度等修饰项、未来效果和 SD/TVA 反事实验证表达。

当前提取器支持两种 phase schema：

```text
legacy
simple_v1
```

`legacy` 保留旧口径，用于复现和对照。`simple_v1` 是新的默认研究口径，包含六个互斥状态：

```text
air_rising
air_falling
ground_front
ground_rear
ground_upright
other
```

映射关系为：

```text
falling_air -> air_falling
rising_air -> air_rising
grounded_rear -> ground_rear
grounded_prone -> ground_front
grounded_upright or grounded_prepared -> ground_upright
otherwise -> other
```

`grounded_hop` 不再作为互斥状态标签，仅作为诊断特征保留。若某一帧在高分 Hop 周期和失败恢复过程中具有一致的观测形态，它应落入同一个状态标签；其规则价值由上下文和后果验证决定。

在 `simple_v1` 下，高分教师轨迹主要表现为 `ground_front`、`air_falling` 和 `air_rising` 的周期性分布，而低分对照轨迹主要停留在 `ground_rear` 与 `ground_front` 且缺少有效腾空周期。该差异来自数据分布和未来效果，不来自状态名中的过程假设。

## 宽候选观察性验证

`validate_hopper_broad_context_candidates.py` 用于在 SD/TVA 反事实验证前筛查宽口径候选。该脚本读取宽候选库，重新抽取高分教师轨迹和低分对照轨迹中的同口径记录，并对每条候选计算：

```text
高分教师上下文支持
目标动作族支持
目标动作族相对基线优势
高分 future outcome 稳定性
低分对照重叠情况
细粒度 omega/support/risk 子区域稳定性
```

输出状态包括：

```text
counterfactual_ready
counterfactual_with_contrast_caution
split_required_before_counterfactual
rejected_*
```

其中 `counterfactual_ready` 只表示观察性对照通过，下一步值得进行 SD/TVA 反事实验证；它不表示规则已经可以进入在线规则库。该分层设计避免把宽候选的高召回误解为因果有效，同时保留框架简洁性。

## SD/TVA 联合反事实验证

宽候选进入因果化验证时，采用联合动作价值口径，而不是 `SD` 与 `TVA` 的独立投票。对每个候选上下文中的真实状态，验证器构造有限动作探针集合，包括 `zero` 以及 `waist/hip/knee/ankle` 四个关节的正负方向，幅度为 `0.35`、`0.70` 和 `1.00`。该集合不声称覆盖连续动作空间，而是作为可解释的单关节动作族审计集。

每个探针动作通过 `SD` 重复推演 `horizon=5` 步，并计算：

```text
U(s, a) = sum_t gamma^t r_SD(s_t, a, s_{t+1}) + gamma^h TVA(s_h)
```

其中 `r_SD` 使用 `model.predict_plan_reward`，`gamma=0.99`。候选规则不要求目标动作族成为第一名，而是要求其在同一上下文的联合动作族排序中稳定处于前列。当前实现以 `joint top3` 作为主要审计指标：若目标动作族经常进入前三，说明规则具有反事实支持；若很少进入前三，则说明该动作族更可能只是高分轨迹中的伴随模式。

`SD` 终端结构评分和单独 `TVA` 终端价值仍在报告中保留，但只作为解释联合结果的辅助审计字段，不作为主要上线依据。

## 反事实驱动候选生成

观察性候选验证显示，“高分轨迹中某上下文下常见的动作族”不能直接等同于教师规则。为此新增反事实驱动候选生成口径：高分轨迹仍用于提供宽上下文和好后果样本，但动作族不再由轨迹中的实际动作频率决定，而由 `SD` 短视界推演收益与 `TVA` 终端价值的联合排序决定。

当前实现为：

```text
experiments/sd_retrospective_rule_induction/generate_hopper_broad_context_counterfactual_rules.py
```

流程为：

```text
高质量轨迹 -> 宽上下文分组 -> 每个上下文采样状态
-> 对每个状态执行动作探针反事实推演
-> 统计各动作族进入联合评分 top3 和 top1 的比例
-> 生成单动作族候选规则
```

默认门槛为：上下文支持不少于 `40`，好后果率不少于 `0.75`，不稳定率不超过 `0.25`，候选动作族 `joint top3 >= 0.60`，`joint best >= 0.15`，且 best-family 熵不超过 `0.85`。这些门槛只用于离线候选筛查，不表示规则可直接上线。生成的候选仍需低分对照、跨周期稳定性和在线规则库转换审计。

## 反事实候选可靠性审计

反事实生成候选进入规则库前，需固定候选规则并在不同数据分布或不同 `SD/TVA` checkpoint 下复核。当前实现为：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_counterfactual_generated_rule_reliability.py
```

该脚本不重新生成规则，只评估既有规则在指定 trace 集合和指定 `SD/TVA` checkpoint 下的 `joint top3`、`joint best`、上下文重叠、好后果率和不稳定率。其用途包括：

```text
低分对照：若规则在低分不稳定轨迹中同样强支持，则不能作为成功教师规则；
跨周期稳定：若规则在相邻周期 SD/TVA 下不再进入前列，则暂缓进入规则库；
覆盖审查：若低分轨迹无对应上下文，可视为该规则主要覆盖高质量行为区域。
```

当前离线筛查采用三类解释：`retain` 表示高分相邻周期仍支持，且低分对照无重叠或不支持；`caution` 表示高分相邻周期支持，但低分不稳定轨迹中有弱支持；`reject_or_defer` 表示低分不稳定轨迹中强支持，或跨周期稳定性不足。

## 反事实教师规则库在线化

通过可靠性筛查的 `retain` 规则进入在线训练前，需转换为闭环规则库格式。当前转换脚本为：

```text
experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_init_library.py
```

转换后的规则保持单动作族干预形式：

```text
prev=<previous_simple_phase>|cur=<current_simple_phase>|omega=<coarse_omega>|vx=<vx_bin>|support=<broad_support> -> prefer:<action_family>
```

在线字段固定为：

```text
source = sd_tva_broad_context_counterfactual
rule_origin = sd_tva_counterfactual_teacher_core_v1
status = confidence_active
intervention_type = soft_prefer_action_family
```

信心值由原始 `joint_top3_rate` 和 `joint_best_rate` 初始化，用于影响先验强度和后续信心更新。教师规则保护不改变动作排序计算本身，只在冲突仲裁中保留高可信教师来源，避免早期低样本反向规则过早覆盖教师规则。强负证据仍可通过规则效果审计和信心更新削弱规则。

宽转移上下文规则依赖 `prev_simple_phase` 与 `current_simple_phase`。在线先验在每次观测评分时维护上一帧简单状态；离线行为审计则在每条 episode 内重新初始化上一状态为 `unknown`，以保证命中统计只反映同一轨迹内的局部转移。

## 教师规则信心维护

反事实教师规则来自高质量轨迹中的稳定上下文，并已经通过 `SD/TVA` 联合反事实、低分对照和跨周期稳定性审计。因此，在线训练中的低命中、零命中或审计缺失不再被解释为负证据；这些情况只说明当前训练分布尚未充分覆盖相应教师上下文。

当前维护口径为：

```text
普通规则：零命中或低命中按 absence_decay / low_hit_decay 衰减；
教师规则：零命中、低命中或审计缺失保持原信心；
教师规则：一致性证据可继续增强；
教师规则：只有达到 hard_harmful_min_hits 且 harmful_rate 达到 hard_harmful_threshold 的强反例才可削弱至 active 阈值以下。
```

该修订保留教师规则的可证伪性，但将“证据不足”和“反证充分”区分开。教师规则仍可能因强反例退出在线动作先验，但不会因未触发或少量样本而被动衰减。

## 规则触发时的近确定性探索窗口

`prev->cur` 教师保护线在 `100k` 显示有效信号后，曾测试保守随机门控：

```text
rule_prior_min_apply_prob = 0.40
rule_prior_max_apply_prob = 0.98
rule_prior_confidence_prob_power = 1.0
```

该设置在 `75k` 正式评估中明显弱于无随机门控的上一轮有效线，说明较宽的规则 dropout 可能过早破坏教师规则的连续引导。后续将探索窗口限缩为近确定性门控：

```text
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

该口径仍保留极小概率的自由探索，但规则命中后绝大多数情况下继续参与最终动作排序。正式评估不采样，仍使用确定性策略；随机性只作用于训练阶段的规则先验应用。

后续排查发现，早期实现中的随机门控采样直接使用 `torch.rand`，会消耗 TD-MPC2/MPPI 的全局随机数流。即使 `p_apply` 接近 `1`，开启门控也会改变候选动作采样、Gumbel 选择和最终探索噪声的后续随机序列，使该实验不再是对无门控有效线的局部扰动。

当前修订为：规则门控使用 `HopperRulePriorV6` 内部独立的 CPU `torch.Generator`，由 `rule_prior_stochastic_seed` 初始化；多种子闭环 launcher 按 `base_seed + cycle * 1000 + seed` 派生每个周期、每个 seed 的门控种子。该设计保持门控本身可复现，同时不改变 TD-MPC2/MPPI 主随机数流。正式评估仍默认不采样，使用期望缩放。

在隔离随机数流后，`0.95-0.995` 紧探索窗口线验证了框架可以恢复训练能力，但多种子入口仍不均衡。后续重训将探索窗口适度放宽为：

```text
rule_prior_min_apply_prob = 0.90
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
```

该设置仍属于近确定性规则门控。其目的不是重新引入大幅随机 dropout，而是在教师规则和在线规则命中时保留约 `0.5%-10%` 的训练期自由探索空间，以观察是否能改善早期入口覆盖和多种子不平衡。除该概率下界外，教师规则来源、规则信心维护、TVA terminal prior 门控、`5k` 闭环更新和 `25k` 正式评估协议均保持不变。

实际记录中，`stochgate0900995_rngiso_seed123_20260528_r1` 为短暂启动失败线，未形成有效模型或闭环结果；有效运行线使用相同协议的 `stochgate0900995_rngiso_seed123_20260528_r2`。

后续 `0.95-0.995` rich 初始化长线完成 `300k` 后显示，随机门控未能解决多种子不平衡，且最终均值优势主要来自单一强种子。因此，当前方法记录将该探索窗口从主要改进假设中移除，并新增无随机门控 clean ablation。该消融保持 rich 初始化教师规则、教师保护、统一 `prev->cur` 在线规则归纳、TVA terminal prior 门控、v8 transition rules、rule-effect confidence、`5k` 闭环更新和 `25k` 正式评估协议不变，仅取消训练期规则触发 dropout。

无随机门控口径为：

```text
enable_rule_prior_stochastic_gate = false
rule_prior_stochastic_eval = false
```

启动脚本中不再传入：

```text
--enable-rule-prior-stochastic-gate
--rule-prior-min-apply-prob
--rule-prior-max-apply-prob
--rule-prior-confidence-prob-power
--rule-prior-stochastic-seed
```

注意：底层训练命令仍会打印 `rule_prior_min_apply_prob`、`rule_prior_max_apply_prob` 等默认配置值，但当 `rule_prior_stochastic_gate=false` 时这些概率不参与规则先验应用。该实验的归因解释应以门控开关是否启用为准，而不是以默认概率字段是否存在为准。

`stochgate0900995_rngiso_seed123_20260528_r2` 的在线结果表明，全局统一将应用概率下界放宽至 `0.90` 未能稳定改善三种子启动平衡。因此，后续长周期验证回到 `0.95-0.995` 紧窗口，并将闭环周期数从 `40` 增至 `60`，即总预算从 `200k` 增至 `300k`。该设置用于检验紧窗口早期效果的可复现性，以及较长预算下规则库是否继续改善整体策略。

## Rich 初始化教师规则提取

为降低早期入口路径对随机样本的敏感性，当前方法新增一个独立于在线训练的 rich 初始化教师规则提取步骤。该步骤只读取既有高质量轨迹、SD checkpoint、TVA checkpoint 和低分对照轨迹，输出一个可选的初始化教师规则库；它不影响正在运行的训练，也不改变在线规则维护机制。

该步骤遵循三段式原则：

```text
高质量成长链路轨迹 -> 宽上下文 SD/TVA 反事实候选 -> 高质量稳定性审计 + 低分对照审计 -> 初始化教师规则库
```

候选规则仍保持简单的单动作族形式：

```text
prev=<previous_simple_phase>|cur=<current_simple_phase>|omega=<coarse_omega>|vx=<vx_bin>|support=<coarse_support> -> prefer:<action_family>
```

这里的 `prev->cur` 不被解释为破坏 MDP 的历史依赖，而是作为粗状态标签不足时的补充上下文。它用于区分“同一粗状态来自成功 hop 周期”与“同一粗状态来自失败恢复过程”的不同决策语境。

新脚本为：

```text
experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_rich_init_library.py
```

与旧的 `build_counterfactual_teacher_init_library.py` 相比，新脚本不再硬编码 `retain8` 规则集合，而是依据以下条件动态选择：

```text
source joint_top3_rate >= min_source_top3
source joint_best_rate >= min_source_best
high-quality audit status in allowed_high_status
high-quality audit top3 >= min_high_top3
low-score contrast status not in blocking_low_status
```

满足条件的规则写为 `confidence_active`，未满足但仍有记录价值的近邻规则写为 `confidence_observing`。该设计使初始化教师库可以更丰富，但不会把低分对照中同样被 SD/TVA 支持的混杂规则直接上线。

第一版 rich 初始化库使用 `teacher_core_init` seed1 的 `100k、125k、150k、175k、200k` official eval 轨迹，目的是覆盖从中等回报到稳定高分 hop 的成长链路。其输出为：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
```

该库包含 `30` 条规则，其中 `22` 条为 `confidence_active`，`8` 条为 `confidence_observing`。该库尚未用于在线重训；后续使用前应先审查其语义覆盖、与既有 `retain8` 教师规则的重合和冲突，以及在早期失败轨迹上的触发情况。

## 500k 接续训练口径

在无随机门控 rich 初始化线完成 `300k` 后，采用同一闭环协议从 `cycle060` 接续至 `cycle100`，用于评估延长训练后多种子是否均能进入较理想姿态。该实验不重新从零训练，也不改变规则先验、TVA、v8 transition rules 或 rule-effect confidence 的核心设置。

接续输入为：

```text
initial_rule_library = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/rules/library_closed_loop_cycle_060_confidence.json
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/sd_explainers/sd_explainer_cycle_060.pt
initial_tva_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/tva/hopper_tva_cycle_060.pt
initial_resume(seed1) = logs/hopper-hop/1/tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059/models/final.full.pt
initial_resume(seed2) = logs/hopper-hop/2/tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059/models/final.full.pt
initial_resume(seed3) = logs/hopper-hop/3/tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059/models/final.full.pt
```

由于 `cycle060` 的 TVA terminal prior gate 为 strong 通过，续训首个周期设置 `initial_tva_terminal_prior_policy=strong`。其后仍由每个新周期的 TVA-SD interface audit 决定下一周期的 strong、weak 或 off policy。随机门控继续关闭，即 `enable_rule_prior_stochastic_gate=false`。

该结果的解释分为两层：`300k` 同预算结果可与公开 TD-MPC2 `300k` 直接比较；`500k` 续训结果用于表征延长训练后的能力上限和姿态稳定性。若需严格声明 `500k` 同预算优于原始算法，应补充原始 TD-MPC2 的同预算 `500k` 对照。

## 400k 正式评估点严格接续口径

`resume300k_20260603_r1` 续训线已产生 `cycle081_target405080` 的训练 checkpoint，但正式评估文件只落盘至 `cycle080_target400079`。因此，若需要从“400k 正式评估点”继续运行，应使用 `cycle080_target400079` 的三种子 full-state checkpoint，而不是使用 `405080` 的未评估 checkpoint。

严格接续口径如下：

```text
start_cycle = 81
cycles = 100
cycle_steps = 5000
cycle081 target_steps = 405080
cycle100 target_steps = 500099
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
```

共享规则库、SD checkpoint 和 TVA checkpoint 均取 `resume300k_20260603_r1/cycle_080` 产物。`initial_tva_terminal_prior_policy` 设为 `strong`，用于继承 `cycle080` 的 TVA terminal prior gate 判定。其余机制保持无随机门控 rich 初始化线不变：`rule_prior_mode=penalty_only`，`rule_prior_allowed_statuses=confidence_active`，`rule_required_status=v8_replay_safe_candidate`，启用 TVA audit、TVA shadow update、TVA consequence gate、TVA terminal prior gate、v8 transition rules 和 rule-effect confidence。
