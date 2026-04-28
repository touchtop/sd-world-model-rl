# Hopper SD 接触开启时序诊断记录（2026-04-28）

## 目标

在 `action ranking` 已确认 `pre-landing falling_air` 上动作排序明显错误之后，继续缩小问题范围，只检查一件事：

> 从同一真实前序状态、沿同一真实动作序列 rollout 时，模型预测的首次有效接地时刻，是否系统性晚于真实 trace。

本轮不重新讨论奖励，不讨论最终得分，只关注 `falling -> landing` 的接触开启时序。

## 新增脚本

新增脚本：

- [analyze_hopper_touch_onset_timing.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_touch_onset_timing.py)

脚本流程如下：

1. 从 `official_eval/episode_csv` 读取真实 trace；
2. 过滤 `falling_air` 且未来 `K` 步内会落地的样本；
3. 以 trace 当前帧为初态，沿 trace 中记录的真实动作序列 rollout 当前 `SDWorldModel`；
4. 逐条比较真实与预测的首次事件时刻：
   1. `toe_touch >= 0.5`
   2. `heel_touch >= 0.5`
   3. `foot_grounded >= 0.5`
   4. `any_contact_z <= 0.03`
   5. `airborne_true < 0.5`
5. 汇总：
   1. onset timing error
   2. 逐步 classification accuracy
   3. 早开 / 晚开 / 缺失比例
   4. 真实接地前后 `vx / vz / omega / contact` 误差

## 运行设置

使用：

- checkpoint：
  [25000.pt](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/2/sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1/models/25000.pt)
- trace：
  [seed2 25k official_eval episode_csv](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/2/sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1/official_eval/step_0025000/episode_csv)

参数：

1. `state_name_filter = falling_air`
2. `require_trace_landing_within = 6`
3. `horizon = 6`
4. `samples_per_trace = 2`

结果目录：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding/summary.json)
- [records.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding/records.json)

## 关键结果

样本数：

1. `num_samples = 20`

### 1. `grounded_any` 几何接地时刻并不算特别差

`any_contact_z <= 0.03` 的首次时刻：

1. `exact_match_rate = 0.55`
2. `pred_early_rate = 0.15`
3. `pred_late_rate = 0.25`
4. `onset_error mean = +0.105`
5. `p50 = 0`
6. `p95 = 1`

这说明：

> 仅从“几何最低点是否过地面阈值”来看，模型对接地时刻并非完全失真，偏差通常在 `±1` 步量级。

### 2. `heel_touch` 与 `foot_grounded` 明显滞后，且大量缺失

`heel_touch >= 0.5`：

1. `pred_missing_rate = 0.55`
2. `both_missing_rate = 0.30`
3. 只有 `1` 条样本同时存在真实与预测 onset，且 `pred_late = +2`

`foot_grounded >= 0.5`：

1. `pred_missing_rate = 0.50`
2. `both_missing_rate = 0.30`
3. 稀少的非缺失样本里，`onset_error mean = +3.5`

这说明：

> 模型往往已经在几何上“碰地”，但 `touch_heel / foot_grounded` 仍不开启，或者开启得太晚。

### 3. `airborne_false` 也系统性偏晚

`airborne_true < 0.5`：

1. `exact_match_rate = 0.25`
2. `pred_late_rate = 0.55`
3. `onset_error mean = +1.05`
4. `p50 = +1`
5. `p95 ≈ +3.1`

这说明：

> 由接触派生出的“空中结束”判别，通常比真实 trace 晚约 `1` 步，尾部可晚 `3~4` 步。

### 4. 真实接地窗口附近，`vz / omega / heel_touch` 误差仍较大

围绕真实 `grounded_any onset ±1` 的绝对误差：

1. `vx_mae mean ≈ 0.37`
2. `vz_mae mean ≈ 1.35`
3. `omega_mae mean ≈ 3.26`
4. `any_contact_z_mae mean ≈ 0.043`
5. `touch_heel_mae mean ≈ 0.52`

这里最关键的是：

> 几何接地已经大体跟上，但在接地前后最敏感的窗口内，`vz / omega` 误差仍明显偏大，而 `heel_touch` 几乎经常完全错过。

## 典型样本

例如 `episode=0, step=7`：

1. 真实第 `1` 步就 `heel_touch=0.997`、`foot_grounded=1.0`、`airborne_true=0.0`
2. 预测第 `1~3` 步 `heel_touch` 仍几乎是 `0`
3. 预测直到第 `4` 步 `airborne_true` 才从 `1` 变成 `0`
4. 同时预测 `any_contact_z` 只是逐步下降，到第 `4` 步才接近接地

这类样本说明：

> 当前模型更像是在先等“几何 z 足够低”，然后才逐步把接触与离地判别打开；而真实系统里，`heel touch` 往往更早、更陡峭地开启。

## 代码层面的对应怀疑

当前实现中，最可疑的链条仍然是：

1. `_touch_target(point_z, vz)` 只看 `point_z` 与全局 `vz`
2. `next_touch_toe/heel = carry * touch + (1-carry) * target`
3. `_contact_gate = touch_gate * geom_gate`

这意味着：

1. 如果 `touch` 还没开，`contact_gate` 会被额外压住；
2. 如果 `touch_target` 只依赖全局 `vz`，就无法利用更直接的 `toe_vz / heel_vz` 近地信息；
3. `CONTACT_CARRY` 让接触信号具有明显惯性，更容易在 onset 边界上偏慢。

本轮结果与这一结构怀疑是一致的。

## 结论

本轮可以收敛出一个明确结论：

1. `pre-landing falling_air` 上，模型并非完全错失几何接地时刻；
2. 但真正对 planner 更敏感的 `heel_touch / foot_grounded / airborne_false` 确实存在系统性滞后与缺失；
3. 因此，当前 `falling -> landing` 的主要问题，不只是连续状态误差，而是接触开启语义没有及时建立。

这足以解释为什么：

1. `pre-landing` 动作排序仍为负相关；
2. planner 可能偏好真实环境里来不及建立支撑的动作；
3. `landing` 相位仍是当前最主要的结构薄弱点。

## 下一步

下一步应直接检查并修订接触更新机制，而不是继续泛化调参：

1. 让 `touch onset` 直接利用 `toe_vz / heel_vz`；
2. 减少 `touch_gate * geom_gate` 对已有 touch 状态的过度依赖；
3. 重新审视 `CONTACT_CARRY` 是否让接触开启过慢；
4. 修订后重复运行本脚本，确认 `heel_touch / foot_grounded / airborne_false` 的晚开比例是否下降。

## 代码修补与复核

随后对接触更新链条做了一个最小修补：

1. `touch target` 改为直接使用局部足端竖直速度，而不只看全局 `vz`；
2. `next_touch_toe/heel` 改为“快开慢关”的非对称 carry；
3. 前掌与后跟分开处理，后跟允许更早开启，前掌保持更严格门槛；
4. 未保留“同一步接触反力提前开启”的激进版本，因为它会直接破坏 `grounded_any` 的几何接地时序。

修补后的结果目录：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding_contactfix3/summary.json)

与基线相比，关键变化如下：

1. `grounded_any` 保持不变：
   1. `exact_match_rate = 0.55 -> 0.55`
   2. `onset_error mean = +0.105 -> +0.105`
2. `heel_touch` 的“完全缺失”明显减少，但仍未达到可接受水平：
   1. `pred_missing_rate = 0.55 -> 0.30`
   2. `both_missing_rate = 0.30 -> 0.10`
   3. `onset_error count = 1 -> 6`
   4. `onset_error mean = +2.0 -> +1.33`
3. `foot_grounded` 也从“多数缺失”变成“部分可触发”，但引入了更多伪开启：
   1. `pred_missing_rate = 0.50 -> 0.25`
   2. `spurious_pred_rate = 0.10 -> 0.35`
4. `airborne_false` 略有改善，但幅度有限：
   1. `pred_missing_rate = 0.05 -> 0.00`
   2. `onset_error mean = +1.05 -> +0.90`
5. `toe_touch` 出现了新的伪开启，需要保守看待：
   1. `spurious_pred_rate = 0.00 -> 0.20`

这说明：

> “局部足端速度 + 快开慢关”这条方向本身是对的，因为它确实减少了 `heel_touch / foot_grounded` 的完全缺失；但当前显式几何与 touch 语义之间仍有系统性错位，继续只靠手工阈值放宽，已经开始以伪开启为代价换取召回。

因此，当前更准确的结论应改为：

1. 原问题不是单一的 `CONTACT_CARRY` 太慢；
2. 更深层的问题是：`touch` 语义与当前显式几何 proxy 并不严格对齐；
3. 最小显式修补可以缓解晚开，但不足以在不引入明显伪开启的前提下彻底修好。

后续若继续推进这条线，优先级应改为：

1. 评估是否为 `touch_toe / touch_heel` 增加受限 residual 或专门的 onset 校正头；
2. 或者重新审视当前 heel/toe 几何 proxy 是否整体偏高，导致 `touch` 只能靠放宽阈值硬拉。

## `touch residual` 实验

随后按上述判断，在 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中新增了实验性 `touch_transition_residual_head`：

1. 主 residual 仍只修 `vx / vz / omega`；
2. 新增 `2` 维 `touch residual`，只修 `touch_toe / touch_heel`；
3. 两个新增通道均为零初始化；
4. 再乘 `touch_transition_residual_base_scale = [0.05, 0.07]`；
5. 还额外乘近地几何 gate，避免明显离地时任意改写接触语义。

由于旧 RL checkpoint 上新增头默认全零，不能直接说明这条机制是否有效，因此又补做了一次最小监督微调：

1. 初始化：
   [hopper_sd_supervised.pt](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt)
2. 数据：
   `seed2@25k official_eval episode_csv`
3. 训练：
   仅 `5` 个 epoch；
4. 损失：
   只保留一步状态损失，加 `touch_loss_weight=0.5`，不再带 rollout 项；
5. 结果：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_touchresid_trace_seed2eval25k_e5_cpu_touchloss050_onestep/summary.json)

然后对这份新 dynamics checkpoint 重新运行 onset 诊断：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_touch_onset/seed2_25k_prelanding_touchresid_tracee5/summary.json)

与基线、`contactfix3` 的对比如下：

1. `heel_touch`
   1. `pred_missing_rate: 0.55 -> 0.45`
   2. `spurious_pred_rate: 0.10 -> 0.25`
   3. `onset_error mean: 2.0 -> 1.0`
2. `foot_grounded`
   1. `pred_missing_rate: 0.50 -> 0.35`
   2. `spurious_pred_rate: 0.10 -> 0.25`
   3. `onset_error mean: 3.5 -> 1.6`
3. 但 `grounded_any` 被明显破坏：
   1. `pred_missing_rate: 0.05 -> 0.55`
   2. `exact_match_rate: 0.55 -> 0.20`
   3. `onset_error mean: 0.105 -> 0.444`
4. `airborne_false` 也没有形成稳定收益：
   1. `pred_missing_rate: 0.05 -> 0.40`
   2. `onset_error mean: 1.05 -> 0.92`

因此，这轮 `touch residual` 实验的结论应明确写成：

> 它确实能减少一部分 `heel_touch / foot_grounded` 的完全缺失，但当前最小 one-step trace 微调会把 `grounded_any` 主链带坏，因此这版不能作为接受版本进入主基线。

更具体地说，问题不在于“新增 residual 完全无效”，而在于：

1. one-step `touch` 校正可以学到局部接触语义；
2. 但如果没有同时约束几何接地与多步滚动一致性，就会出现“touch 稍好、几何更坏”的负迁移；
3. 因而下一轮若继续推进这条线，必须把 `grounded_any / any_contact_z` 的保持约束与 `touch` 校正一起纳入，而不能只做局部 one-step 微调。
