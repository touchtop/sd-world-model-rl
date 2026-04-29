# Hopper SD 接入 TD-MPC2 的验证与消融计划

## 1. 文档目的

本文档将当前 Hopper `SD transition core` 的下一阶段工作，收敛为一套明确的闭环验证与消融计划。目标不是继续手工调奖励，也不是继续无边界修改 SD 公式，而是回答一个更具体的问题：

> 当前 `contactfix5 + internal transition residual + supervised shortloss010` 这条主线，是否真的改善了 TD-MPC2 在 Hopper-Hop 上的短视界规划质量、样本效率和闭环训练稳定性。

本文档吸收以下事实前提：

1. 当前活跃实现已回到纯环境奖励口径，`sd_reward_mode=learned`；
2. 当前闭环主线推荐 `sd_latent_dim=512`，保留外层 learned residual latent，用于补足显式 SD 结构态未覆盖的动力机制；
3. 当前最强证据仍来自监督拟合，而不是闭环 RL；
4. 当前不再建议恢复自定义结构化 reward shaping；
5. 下一阶段应转向严格的诊断与 ablation，而不是继续混合更多机制。

## 2. 当前阶段结论

当前最准确的阶段性结论是：

1. Hopper `SD dynamics` 的短视界预测能力已经显著强于早期版本；
2. 该版本已经具备接入 TD-MPC2 做系统性闭环验证的条件；
3. 但当前证据主要仍是监督拟合证据，还不能说闭环 RL 已明确受益；
4. 后续重点应是检验 `SD transition core` 是否真的改善 TD-MPC2，而不是继续讨论奖励塑形。

因此，下一阶段的实验目标应表述为：

> 验证改进后的 `SD transition core` 是否提升 TD-MPC2 的 transition accuracy、action ranking quality、规划稳定性与样本效率。

### 2.1 2026-04-29 方法论收敛

截至 `2026-04-29`，当前 Hopper 线还补充形成了一条更高层的方法论结论：

> `SD dynamics` 的拟合准确性，与 TD-MPC2 的规划受益，不应再被预设为天然一致。

原因如下：

1. `SD` 拟合目标是最小化真实转移误差；
2. TD-MPC2 规划目标是把真实长期回报更高的动作排在前面；
3. 对 Hopper-Hop 而言，`landing / takeoff / high_omega / near_contact` 这类任务相关相位误差，对规划的危害远大于普通 grounded 段的小误差；
4. 当前在线训练使用的是联合 world-model 损失，而不是纯 `SD` 系统辨识损失，因此只会把模型推向“对当前策略分布有用”，不会自动维护“全相位低误差”的 `SD` 拟合器。

当前经验现象已经支持这一判断：

1. `consistency_loss` 在 `10k-50k` 区间下降，但在 `50k` 之后反弹；
2. `50k -> 75k` replay 诊断中，`support/contact` 局部指标略有改善，但 `rollout_3_rmse`、`pitch_rate_mae`、`high_omega` 相位误差没有同步下降；
3. `100k` 闭环评估虽然已出现“全部 episode 非零”，但行为改善主要表现为“后仰失败态 -> 前倾推进态”的恢复，而不是 `hop` 行为链形成。

因此，下一阶段必须把验证对象正式拆成两条线：

1. `model fidelity`
2. `planning usefulness`

并且在训练设计上优先考虑：

1. 保持单一共享的 `SD transition core`；
2. 保留当前 RL 联合目标；
3. 额外增加一条独立的在线 `SD` 监督支路，用于持续维护 `1 / 3 / 5` 步 rollout 误差与关键相位保真度。

### 2.2 2026-04-28 状态更新

截至 `2026-04-28`，本计划中的前 3 项诊断已有以下进展：

1. `phase-sliced error` 已完成，结果见
   [phase_sliced_error.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/phase_sliced_error.json)
2. `internal residual audit` 已完成，结果见
   [residual_usage_phase_audit.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/residual_usage_phase_audit.json)
3. `policy-distribution validation` 已完成第一轮，结果见
   [policy_distribution_error_25k_replay.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/policy_distribution_error_25k_replay.json)
4. `action ranking diagnostic` 已完成第一轮，结果见
   [seed2_25k_falling/summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_ranking/seed2_25k_falling/summary.json)
   与
   [seed2_25k_filtered/summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_ranking/seed2_25k_filtered/summary.json)
   ，以及更聚焦的
   [seed2_25k_prelanding/summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_ranking/seed2_25k_prelanding/summary.json)
5. `touch onset timing` 已完成第一轮，结果见
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding/summary.json)
   与
   [learnable_sd_v5_touch_onset_2026_04_28.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_touch_onset_2026_04_28.md)
6. `touch onset` 最小修补已完成一轮复核，结果见
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding_contactfix3/summary.json)
7. 实验性 `touch residual` 已完成一轮最小监督微调与复核，结果见
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding_touchresid_tracee5/summary.json)

当前最新事实是：

1. 随机验证集到 policy replay 的 distribution shift 明显存在；
2. 在同一 `seed2@25k replay` 上，RL 当前模型相较监督初始化模型已明显改善 `3/5` 步 rollout 误差；
3. `landing / takeoff / high_vz_impact / high_omega` 仍是当前最主要的薄弱相位；
4. 在 `early falling` 与更聚焦的 `pre-landing falling` 状态上，当前 action ranking 仍明显错误；
5. `pre-landing` 上模型对 `grounded_any` 的几何接地时刻并非完全失真，但 `heel_touch / foot_grounded / airborne_false` 明显存在晚开与缺失；
6. 最小显式修补后，`heel_touch / foot_grounded` 的“完全缺失”确有下降，但同时引入了更多伪开启，说明问题不只是 `carry` 太慢；
7. 实验性 `touch residual` 在最小 one-step trace 微调后，确实能减少 `heel_touch / foot_grounded` 的缺失，但会明显破坏 `grounded_any` 几何主链；
8. 这表明当前 `falling -> landing` 的关键薄弱点，已经可以进一步收敛到“接触开启语义与显式几何 proxy 不完全对齐”，且单独优化 `touch` 会对几何主链产生负迁移；
9. 在大多数 `rear/prone` 失败簇状态上，真实短视界目标已近似退化为常数，说明这部分状态不适合作为主要排序验证集。

## 3. 总体原则

后续所有 Hopper SD 验证应遵循以下原则：

1. 保持 `pure-env` 口径，不重新引入 shaping；
2. 保持 `sd_latent_dim=512`，但在一致性损失中将 17 维结构态块与 learned residual latent 块按块加权，避免高维 residual 仅因维度数更多而掩盖显式结构贡献；
3. 先补短平快诊断，再拉长 50k/100k 闭环预算；
4. 所有闭环结论必须来自对照实验，不允许只汇报单条 full model 结果；
5. 所有关键结论都要区分：
   1. 监督拟合改善
   2. 闭环 TD-MPC2 改善
6. 不再把 `consistency_loss` 的下降视为“规划一定改善”的充分证据；
7. 后续若修改训练方式，优先采用“共享 dynamics core + 独立在线 SD 监督支路”，而不是引入两个互相竞争的 dynamics 头。

## 4. 诊断优先级

下一步优先做 5 类诊断。优先级从高到低如下。

### 4.1 P0：phase-sliced 预测误差

不能只看整体 `RMSE / MAE`。必须按相位拆开误差，至少包括：

1. `airborne`
2. `grounded`
3. `landing`
4. `takeoff`
5. `toe-only contact`
6. `heel-only contact`
7. `full-foot contact`
8. `high-omega`
9. `high-vz-impact`

每个相位至少统计：

1. `H_mae`
2. `vx_mae`
3. `vz_mae`
4. `theta_mae`
5. `omega_mae`
6. `touch_toe_mae`
7. `touch_heel_mae`
8. `z_min_mae`

目标是回答：

> 当前模型是否只是“整体平均误差不错”，但在 landing/takeoff 等 TD-MPC2 真正敏感的时刻仍然很差。

### 4.2 P0：internal residual 贡献与饱和率审计

必须量化 internal residual 到底承担了多少动力学责任。至少记录：

1. `base_delta_vx` vs `residual_delta_vx`
2. `base_delta_vz` vs `residual_delta_vz`
3. `base_delta_omega` vs `residual_delta_omega`
4. `residual_ratio = |residual| / (|base_delta| + eps)`
5. `tanh_saturation_rate`
6. `phase-wise residual_ratio`

目标是回答：

> 当前是“SD 主干已经较准，residual 只做小修补”，还是“residual 已经承担与主干同阶的关键动力学责任”。

这决定后续论文与报告中的叙事边界。

### 4.3 P0：policy-distribution validation

当前监督数据主要来自随机动作采样。这对初始化有价值，但不代表在 TD-MPC2 真正访问的状态-动作分布上也一样准确。

必须补一个 policy-induced replay 验证：

1. 从当前 `5k pure-env sdinit` 或后续 `10k` 闭环实验的 replay 中抽样；
2. 在这些状态-动作对上重新评估：
   1. `1-step error`
   2. `3-step rollout error`
   3. `5-step rollout error`
   4. `contact error`
   5. `reward prediction error`

目标是回答：

> 当前监督初始化是否仅在随机数据分布上有效，而在 planner/policy 真正关心的分布上迅速退化。

### 4.4 P0：action ranking diagnostic

这是最接近 TD-MPC2 规划需求的诊断。对真实 replay 中采样的状态 `s_t`：

1. 采样若干候选动作 `a_i`；
2. 用 `SDWorldModel` 预测 `next-state / reward / short-horizon value`；
3. 用真实环境 1~3 步 rollout 计算真实动作优劣；
4. 比较模型排序与真实排序的一致性。

建议统计：

1. `Spearman rank correlation`
2. `top-k action hit rate`
3. `model-chosen action true return`
4. `random action true return`

目标是回答：

> 即使 RMSE 还可以，当前 SD 是否真的能把“更好的动作”和“更差的动作”排对。

### 4.5 P1：显式官方 reward 近似对照

当前 `reward_mode=learned` 是合理的，因为要避免结构化 shaping 混入。但 Hopper 官方 reward 本身不是人为塑形，而是任务定义的一部分。

因此建议加入一个对照，不作为主线替换，只作为瓶颈诊断：

1. `A`: learned reward head
2. `B`: explicit official reward approximation from z
3. `C`: learned reward + explicit reward auxiliary target

目标是回答：

> 如果 dynamics 已经显著改善，而闭环仍起不来，瓶颈是否在 reward learning 而不是 transition model。

## 5. 闭环实验矩阵

### 5.1 阶段一：5k / 10k smoke

这一阶段不以最终 return 为目标，而只看稳定性与是否存在明显负迁移。

至少比较：

1. `vanilla TD-MPC2`
2. `SD dynamics random init`
3. `SD dynamics supervised init`
4. `SD dynamics supervised init + frozen dynamics first N steps`

记录：

1. `episode return`
2. `consistency / struct_consistency`
3. `reward_loss`
4. `Q loss`
5. `policy entropy / action scale`
6. `residual magnitude`
7. `phase-sliced contact error`

这一阶段的成功标准不是高分，而是：

1. 不数值爆炸；
2. 不出现明显 action collapse；
3. 不出现 reward/Q loss 异常；
4. 不出现比 vanilla 明显更差的早期稳定性。

### 5.2 阶段二：50k / 100k 样本效率对比

这是第一批可以开始看 return 的实验。最小建议矩阵如下：

1. `A`: vanilla TD-MPC2 baseline
2. `B`: SD, no supervised init
3. `C`: SD, supervised init
4. `D`: SD, supervised init, internal residual disabled
5. `E`: SD, supervised init, internal residual enabled

固定：

1. `3 seeds`
2. 相同 replay budget
3. 相同 `eval_freq / checkpoint_freq`
4. 相同 `horizon / num_samples / num_elites / num_pi_trajs`

这一阶段回答的问题是：

1. `SD` 结构本身是否有用；
2. `supervised init` 是否有用；
3. `internal residual` 是否有用；
4. 是否只是训练噪声。

### 5.3 阶段三：冻结与微调策略对照

考虑到监督初始化后，RL 早期 replay 质量通常较差，需要比较：

1. `full fine-tune`
2. `freeze dynamics for first 5k`
3. `freeze dynamics for first 10k`
4. `smaller dynamics lr fine-tune`

这一阶段回答的问题是：

> 监督初始化后的 SD 是否会在早期被劣质 replay 迅速带偏。

### 5.4 阶段四：policy-distribution 再拟合

如果前面诊断显示：

1. policy-distribution 下 model error 明显高于随机数据；
2. 或闭环训练中 consistency error 持续升高；

则再进入轻量数据聚合流程：

1. 随机数据监督初始化；
2. 运行 5k/10k policy；
3. 用 policy replay 再做 supervised fine-tune；
4. 再接回 TD-MPC2。

这里的重点是修正 distribution shift，而不是重新发明新奖励或继续手调 SD 公式。

## 6. 最小 ablation 集合

如果算力有限，建议最先保证下面这组最小 ablation：

1. `vanilla TD-MPC2`
2. `SD + no init`
3. `SD + supervised init`
4. `SD + supervised init + residual off`
5. `SD + supervised init + residual on`

如果这 5 组都做不到，就不应过早对 Hopper SD 的有效性下强结论。

## 7. 结论表述边界

在所有这些实验完成前，后续文档、汇报与论文叙事都应遵守以下边界：

### 7.1 当前可以说的

1. 当前 `SD core` 的短视界 transition accuracy 已明显改善；
2. 当前模型边界比旧版 `reward shaping + high-dim residual` 系统清楚得多；
3. 当前版本已经具备接入 TD-MPC2 做严格闭环验证的条件；
4. 显式结构提供了几何、接触和动作响应上的强先验。

### 7.2 当前不能说的

1. 不能说 Hopper 已经被解决；
2. 不能说 SD 已明确优于黑盒 world model；
3. 不能说显式主干本身已经足够准确；
4. 不能把监督拟合结果直接当成闭环 RL 结果。

### 7.3 若 residual 贡献过大时的保守表述

如果 residual 审计发现：

1. residual 与 base dynamics 同阶；
2. 或 residual 大量饱和；

则更准确的表述应是：

> 显式 SD 主干提供了结构化几何与接触坐标系，internal residual 完成了关键短视界根部运动校正。

而不是：

> 显式 SD 主干本身已完成高精度动力学刻画。

## 8. 当前执行顺序

当前推荐执行顺序如下：

1. 固定当前 `contactfix5 + internal residual + shortloss010` 作为基准 checkpoint；
2. 先做：
   1. `phase-sliced error`
   2. `residual audit`
   3. `policy-distribution validation`
   4. `action ranking diagnostic`
3. 同步推进 `5k/10k` 闭环 smoke；
4. 若数值稳定，再拉 `50k/100k` 多 seed ablation；
5. 暂不恢复任何自定义结构化 reward shaping。

## 9. 最终一句话

当前最应该做的，不是继续手工修改 Hopper SD 公式，而是：

> 用严格的诊断和 ablation，检验这版 `SD transition core` 是否真的改善了 TD-MPC2 的闭环规划与学习。
