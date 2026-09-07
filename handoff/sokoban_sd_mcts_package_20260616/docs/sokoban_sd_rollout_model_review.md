# Sokoban/Boxoban SD Flow Smoke Review

## 审查范围

本审查面向当前 Sokoban/Boxoban 六核心存量 SD 模型实现、验证性实验结果和复现记录。审查材料包括：

1. [validate_sokoban_sd_design.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py)
2. [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py)
3. [flow_rollout_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_30_levelsplit_summary.json)
4. [flow_rollout_weighted_feedback_target_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_weighted_feedback_target_30_levelsplit_summary.json)
5. [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
6. [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)
7. [research_log.md](/home/data/round/code/sd_world_model_rl/docs/research_log.md)

本次审查调用 `academic-research-suite`，并按 `methodology_reviewer_agent`、`devils_advocate_reviewer_agent`、`editorial_synthesizer_agent` 与 `experiment-agent validate` 的规范在当前 Codex 会话内执行。未调用外部独立模型，也未将代码、结果或文档发送到外部 API。

## 审查结论

当前最可靠、可审计、可复现的基线应冻结为：

```text
six stocks + fixed incidence matrix + nonnegative learned flows
+ legal single-push action features
+ level-holdout split
+ uniform stock loss
+ two-step rollout supervision
+ explicit diagnostic provider audit
```

对应实验组为 `rollout_sup_base`。该组不是完整 SD-MCTS 结果，而是进入 learned-transition-provider SD-MCTS 前的最小 learned stock-flow 诊断基线。反馈门控、feedback target 和 `reachable_capacity/congestion` 加权损失均应保留为扩展实验项，暂不进入默认基线。

该结论与“从简化但可靠、可审计、可复现的 SD 模型开始逐渐丰富”的研究推进原则一致。当前证据不支持一上来采用更全面的反馈门控、辅助目标和加权损失组合，因为这些复杂项尚未带来稳定增益，且会增加解释和审计成本。

## Methodology Review

### 结构约束审查

当前实现保留了首版系统动力学模型的必要骨架。`validate_sokoban_sd_design.py` 固化六个存量、十二个流量和已知 incidence matrix；`diagnostic_flows(delta)` 将观测到的存量变化分解为非负流量；验证脚本逐 transition 检查 `B f_t` 对抽象存量变化的重构误差。设计验证结果显示 `levels_checked=5`、`levels_passed=5`、`transitions_checked=200`、`range_violations=0`、`nonnegative_flow_violations=0`、`max_balance_l1=0.0`，说明 schema、符号约束和 provider/aliasing 审计字段可以稳定生成。

`train_sokoban_flow_smoke.py` 中，`FlowNet` 通过 `softplus` 输出非负流量，`predict_next_stock()` 使用固定矩阵 `INCIDENCE_MATRIX` 并执行 `x_{t+1}=clip(x_t+B f_t)`。因此，当前 learned flow smoke 没有退化为直接预测下一状态的黑箱回归，也没有把世界模型简化为启发式动作分数。

当前代码也明确记录 `design_scope.sd_mcts_enabled=false` 和 `transition_labels_from=exact_single_push_diagnostic_provider`。这一点是重要优点：结果边界被写入输出文件，避免把诊断标签误报为 learned SD-MCTS rollout。

### 实验设计审查

实验口径基本满足当前阶段要求。随机 transition split 已被 level-holdout split 取代，三种子 `30` 关卡结果可复现地显示 learned nonnegative flow 优于 zero-flow baseline。一步 level-holdout smoke 中，验证集 stock MAE 均值为 `0.0516518789033095`，zero-flow 均值为 `0.09816546738147736`，相对改善均值为 `0.4691395196414155`。

多步 rollout consistency 是当前最有价值的新增验证。`rollout_sup_base` 相对 `one_step_train_base_rollout_eval` 将 valid rollout stock MAE 均值从 `0.05358633274833361` 降至 `0.048756017039219536`，step-2 stock MAE 从 `0.062035638839006424` 降至 `0.056363618622223534`。因此，rollout supervision 能改善反复应用 learned stock-flow transition 后的短视界一致性，应纳入最小基线。

### 复杂项审查

反馈门控结构在代码上是可运行的，但当前证据不足以支持其作为默认基线。带 rollout supervision 时，`rollout_sup_feedback` 的 valid rollout stock MAE 均值为 `0.04945573459068934`，高于 `rollout_sup_base` 的 `0.048756017039219536`；step-2 stock MAE 也从 `0.056363618622223534` 上升到 `0.057083974281946816`。因此，反馈门控可以作为机制候选保留，但不能作为已经验证的核心改进。

`feedback_target_loss_weight=0.1` 有弱正信号，但不足以改变基线选择。`uniform_feedback_target010` 相对 `uniform_feedback` 的 valid rollout stock MAE 从 `0.04945573459068934` 降至 `0.04896984373529752`，但仍未超过 `uniform_base` 的 `0.048756017039219536`。该结果说明 feedback target 可作为后续诊断入口，而不是默认训练项。

`reach_congestion` 加权损失为负结果。`reach_congestion_base` 相对 `uniform_base` 的 valid rollout stock MAE 从 `0.048756017039219536` 升至 `0.05112618890901407`，step-2 stock MAE 从 `0.056363618622223534` 升至 `0.05892247830828031`。该结果说明 `reachable_capacity` 与 `congestion` 的误差不是简单提高损失权重即可解决的问题，更可能涉及 stock proxy、流量分解或抽象状态别名。

## Devil's Advocate Review

### 最强反方意见

当前模型仍可能被质疑为“用 exact single-push diagnostic provider 生成标签，再学习其压缩投影”。这种批评不是说当前 smoke 无价值，而是指出它尚未证明 learned SD transition 能在 SD-MCTS 中替代精确推箱动力学。只要 rollout 标签和动作序列仍由 exact provider 生成，当前结果就只能证明六存量与非负流量存在可学习信号，不能证明 formal MBRL + SD-MCTS 已成立。

第二个强攻击点是抽象状态别名。设计验证 smoke 已记录 `max_alias_next_delta_variance=0.4044025054643195`，而多步误差集中在 `reachable_capacity` 与 `congestion`。这意味着不同原始箱子排列可能映射到相近六存量状态，但后继变化差异较大。若不先审计 alias bucket，直接把当前模型接入 MCTS，可能产生不可执行或后果偏差较大的 imagined branch。

第三个攻击点是复杂项先行。反馈门控、反馈目标和加权损失看似更接近完整 SD，但当前结果并未证明其必要性。若将这些项写入基线，审稿人可以合理质疑模型复杂度超过证据支持范围，并且无法判断性能变化来自系统动力学机制、网络容量、辅助目标还是损失权重。

### 主要风险

| 风险 | 严重性 | 依据 | 处理建议 |
| --- | --- | --- | --- |
| 将诊断 provider 结果误报为 formal SD-MCTS | Critical | 所有当前 flow 结果均标注 `transition_labels_from=exact_single_push_diagnostic_provider`，且 `sd_mcts_enabled=false`。 | 继续将当前结果命名为 diagnostic learned flow baseline；formal 结果必须记录 learned transition provider。 |
| 过早把 feedback gate 设为基线 | Major | `rollout_sup_feedback` 未优于 `rollout_sup_base`。 | feedback gate 保留为 ablation，不进入默认模型。 |
| 使用加权损失掩盖 proxy 问题 | Major | `reach_congestion` 加权后总体 rollout 误差上升。 | 先审查 proxy、flow decomposition 和 aliasing，而不是继续调权重。 |
| 一步 stock MAE 掩盖 flow 分解错误 | Major | `action_only_base` stock MAE 较低但 flow MAE 较高。 | 主指标必须包含 flow MAE、rollout MAE 和 per-stock MAE。 |
| 抽象状态别名影响 MCTS 展开 | Major | design smoke 已出现较高 alias next-delta variance。 | 在接入 SD-MCTS 前生成 alias bucket 审计表。 |

## Experiment Validation

按 `experiment-agent validate` 口径，当前结果属于 `ANALYZED`，不能标为 `VERIFIED formal SD-MCTS`。主要原因是：当前实验可以本地复现，随机种子、数据路径、环境和输出文件已记录；但其验证对象是诊断性 learned flow，而不是完整的 MBRL 闭环或 SD-MCTS 规划。

当前统计解释应保持描述性，不进行显著性夸大。三种子均值和标准差足以支持 smoke 阶段的方向性判断，但不足以支持正式性能主张。最稳妥的表述是：

1. learned nonnegative flow 在 level-holdout 一步预测中稳定优于 zero-flow；
2. rollout supervision 改善两步一致性；
3. feedback gate 未显示稳定增益；
4. feedback target 仅有弱正信号；
5. reach/congestion 加权损失为负结果；
6. 当前主要误差集中于 `reachable_capacity` 与 `congestion`。

11 类方法学谬误扫描中，当前最相关的是选择偏差、过拟合、确认偏差、look-elsewhere effect 和因果过度解释。现有文档已通过 level-holdout、三种子、负结果记录和 provider 边界标注降低这些风险；后续仍需避免把 smoke 结果写成 formal benchmark 结论。

## Editorial Synthesis

### 决定

**当前模型代码与实验结果通过诊断性基线审查，但不通过 formal SD-MCTS 结果审查。**

当前应冻结的基线为 `uniform_base + rollout supervision`，即六核心存量、固定 incidence matrix、非负 learned flow、合法 single-push action features、level-holdout split 和两步 rollout supervision。该基线简单、可靠、可审计、可复现，适合作为后续逐步丰富的起点。

### 暂不纳入基线的项目

| 项目 | 当前状态 | 审查决定 |
| --- | --- | --- |
| `FeedbackFlowNet` gate | 可运行，但未带来稳定 rollout 增益。 | 保留为 ablation。 |
| feedback target | 弱正信号，但未超过 base。 | 保留为候选扩展。 |
| `reach_congestion` stock loss | 负结果。 | 暂停推进，先审查 proxy 与 aliasing。 |
| action-only 低 stock MAE | flow MAE 更差。 | 作为指标陷阱证据，不作为模型改进。 |
| SD-MCTS 接入 | 尚未使用 learned transition provider。 | 暂缓，等待审计门槛通过。 |

### 下一阶段强制门槛

进入 formal learned-transition-provider SD-MCTS 前，建议先完成以下门槛：

1. 冻结 `rollout_sup_base` 为最小诊断基线，并保存配置摘要；
2. 生成 `reachable_capacity` 与 `congestion` 的 proxy 审计，包括原始特征组成、delta 分布和 per-level 误差；
3. 生成 alias bucket 审计，列出同一抽象状态桶内后继 delta 方差最大的样本；
4. 检查 diagnostic flow decomposition，确认高误差 stock 是否来自不可辨识流量分解；
5. 保持 feedback gate、feedback target 和 loss reweighting 为单因素 ablation；
6. 接入 SD-MCTS 前实现 learned transition provider audit，日志必须区分 learned provider、exact diagnostic provider 和 heuristic diagnostic path；
7. formal 实验同时报告 world-model 指标、provider 审计、规划指标和 no-feedback/no-learned-parameter/heuristic diagnostic 消融。

## Post-review Execution

审查结论确认后，已按 `rollout_sup_base` 冻结最小基线并重新复测。训练脚本新增 `baseline_contract`、provider audit、逐存量/逐流量误差、clip audit 和 aliasing audit；新增汇总脚本对三种子结果与旧 `rollout_sup_base` 做复现比较。上述改动只增加审计输出，不改变训练目标或网络结构。

复测结果写入 [flow_baseline_retest_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_summary.json)。与旧结果比较，`valid_stock_mae`、`valid_flow_mae`、`valid_rollout_stock_mae_mean`、step-1/step-2 stock MAE 和 flow MAE 均为 exact match，复现判定为 `REPRODUCIBLE`。

新增审计显示，valid any-stock clipped fraction 均值为 `0.9277460177739462`，valid mean abs clip correction 均值为 `0.017344274868567783`；stock-only alias max next-delta variance 均值为 `0.38876330852508545`，stock-action alias max next-delta variance 均值为 `0.1035704289873441`。该执行结果强化了原审查结论：冻结基线可复现，但进入 SD-MCTS 前应先处理 clipping、stock proxy、flow decomposition 和 aliasing 问题。

## Stock-flow Diagnostic Follow-up

按上述审查结论，已继续执行更细粒度的 stock-flow 诊断，而不是增加模型复杂度或接入 SD-MCTS。新增 [audit_sokoban_stock_flow_diagnostics.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_stock_flow_diagnostics.py)，并在 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 中补充 per-stock 裁剪、delta 分布、flow 分布和相反流共激活审计。该轮结果写入：

1. [flow_baseline_diagnostics_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_summary.json)
2. [stock_flow_diagnostics_30_levelsplit.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json)

增强审计版冻结基线与旧 `rollout_sup_base` 主指标 exact match，复现判定仍为 `REPRODUCIBLE`。完整诊断覆盖 `30` 个关卡、`566` 个抽象状态和 `3435` 条 transition。exact diagnostic flow labels 的 incidence balance `max_l1=0.0`，说明当前诊断流量分解本身满足固定矩阵守恒。

细节审查结论如下：

| 审查项 | 结果 | 判断 |
| --- | --- | --- |
| incidence matrix 与 diagnostic flow labels | `max_l1=0.0` | 结构守恒成立，当前问题不在标签矩阵本身。 |
| `clip_X` 使用 | valid any-stock clipped fraction 均值 `0.9277460177739462` | 预测侧过度依赖边界裁剪，进入 SD-MCTS 前必须处理。 |
| 相反流共激活 | 六组 paired flows 的预测共激活比例均为 `1.0` | `softplus` 非负流可能产生补偿性相反流，削弱流量解释性。 |
| stock-action aliasing | 最高 next-delta variance `0.11775519031141866` | 当前六存量加动作特征仍非充分抽象，MCTS imagined branch 风险较高。 |
| `reachable_capacity` | delta nonzero fraction `0.37147016011644834` | 该存量必要，但当前 `reachable_count/floor_count` 代理可能过粗。 |
| `congestion` | delta nonzero fraction `0.6954876273653566` | 该存量必要，但当前合法推箱数归一化难以区分局部拥塞机制。 |

该 follow-up 审查进一步支持冻结基线策略：当前不应将 feedback gate、feedback target、loss reweighting 或 SD-MCTS 接入作为默认推进方向。下一步应按单因素方式修订模型，使每个修订明确对应一个诊断问题，并在同一 level-holdout、同一两步 rollout 和同一 provider audit 下与冻结基线比较。

## Stage-1 Paired Flow Review

阶段 1 第一轮实现了 `paired_net` 流量参数化。该版本保持六存量、十二非负流量和固定 incidence matrix，不改变 stock proxy，不启用 feedback gate，也不接入 SD-MCTS。其唯一结构变更是将每对相反流量由一个有符号净流量展开，从而保证每对流量至多一个方向激活。

审查材料包括：

1. [flow_stage1_paired_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_summary.json)
2. [flow_stage1_paired_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_gate.json)

阶段 1 gate 判定为 `PARTIAL`。该版本通过了相反流共激活和 incidence balance 门限：flow pair predicted coactivation mean 为 `0.0`，max 为 `0.0`，incidence balance max L1 为 `0.0`。同时，valid flow MAE 从冻结基线的 `0.04738566527764002` 降至 `0.03719972570737203`，说明 paired/net-flow 对 diagnostic flow target 更友好。

但该版本没有通过阶段 1 总门限。valid rollout stock MAE mean 为 `0.05166300448278586`，高于门限 `0.05119`；valid rollout step-2 stock MAE 为 `0.05988542859752973`，高于门限 `0.05918`；valid mean abs clip correction 为 `0.028677965203921`，高于门限 `0.01214`；valid any-stock clipped fraction 为 `0.9618482192357382`，高于门限 `0.75`。

审查结论为：`paired_net` 证明了相反流共激活可以由结构参数化消除，但单独使用会增加边界裁剪并造成轻微 rollout 退化。该版本不应进入阶段 2，也不应接入 SD-MCTS。下一轮阶段 1 应保留成对互斥思想，同时加入流量幅度控制或 raw-next boundary penalty，使共激活、rollout 非退化和裁剪下降三个门限同时成立。

## Stage-1 Capacity Projection Review

第二轮阶段 1 修订在 `paired_net` 基础上加入容量一致性投影。该投影不改变六存量、十二非负流量或固定 incidence matrix，也不直接输出动作评分；它只在网络预测流量后缩放会导致指定存量越出 `[0,1]` 的入流或出流组。达标候选使用 `capacity_projection_stock_mask=1,1,1,0,0,1`，保护 `goal_completion`、`unresolved_box_pressure`、`mobility` 和 `irreversibility_risk`，同时保留 `reachable_capacity` 与 `congestion` 的流量自由度。为消除普通投影候选仅剩的 step-2 轻微超限，训练时将 rollout 最后一步 stock loss 的标量权重设为 `1.2`。

审查材料包括：

1. [flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json)
2. [flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json)

阶段 1 gate 判定为 `PASS`。通过项覆盖全部阶段 1 门限：valid rollout stock MAE mean 为 `0.05017088601986567`，低于 `0.05119`；valid rollout step-2 stock MAE 为 `0.058157226691643395`，低于 `0.05918`；valid mean abs clip correction 为 `9.097307914392635e-07`，低于 `0.01214`；valid any-stock clipped fraction 为 `0.0005817335719863573`，低于 `0.75`；flow pair predicted coactivation mean 和 max 均为 `0.0`；incidence balance max L1 为 `0.0`。

负结果也已记录。普通容量投影候选仅剩 step-2 超限，判定为 `PARTIAL`；逐存量 final-step 权重、`rollout_loss_weight=1.2/1.5`、`rollout_final_step_weight=1.5` 和 `epochs=300` 均未形成更稳健的候选。因此，审查结论为：`paired_net + capacity projection + rollout_final_step_weight=1.2` 是当前阶段 1 的最小达标模型，后续不应继续围绕阶段 1 gate 调参。

该结论仍有边界。当前结果证明 learned stock-flow model 的阶段 1 结构门限已通过，但尚未证明 learned transition provider 已被 SD-MCTS 消费。进入下一阶段时，必须新增 provider audit，记录 MCTS 展开使用的是 learned transition、exact diagnostic label 还是 heuristic diagnostic path。

## 最终结论

当前阶段不应追求看似全面的 SD 模型。最合理路线是先冻结简化可靠的 learned stock-flow baseline，再围绕高误差存量、抽象别名和 provider 审计逐项扩展。只有当某个扩展项在同一 level-holdout、同一 rollout 指标和同一 provider 边界下稳定优于该基线时，才应进入主模型。
