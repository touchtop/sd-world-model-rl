# Hopper SD 中 `vx / pitch_rate` 建模专项审查

## 2026-04-28 修复后复核

本文件前述主体内容记录的是修复前基线。2026-04-28 已完成两轮结构修补，并据此补充复核结论。

### 已落地的结构修补

第一轮修补：

1. `F_leg / F_x / F_z` 改为使用动作更新后的 `next_q`、`next_qdot`、`next_activation`；
2. `next_omega` 中加入接触点相对躯干的力矩项 `tau_contact`。

第二轮修补：

1. 保留上述 grounded 接触力矩链；
2. 额外加入小幅的“空中惯性耦合先验”，使 `joint_acc` 在 `G` 很低时仍可显式影响 `vx` 与 `omega`；
3. 未新增可学习参数，以避免旧 checkpoint 的 strict load 链路被打断。

对应代码位于：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py)

### 修复后动作敏感性结果

正式复核结果见：

- 修复前基线：
  [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000/summary.json)
- 第一轮修补后：
  [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000_after_fix/summary.json)
- 第二轮修补后：
  [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000_after_fix2/summary.json)

最关键的变化如下：

1. 修复前，20 个样本内的 `sd_delta_vx` 唯一值个数恒为 `1`，`sd_delta_pitch_rate` 唯一值个数也恒为 `1`；
2. 第一轮修补后，grounded 样本已恢复动作区分，但大量 airborne 样本仍塌缩；
3. 第二轮修补后，20 个样本中 `sd_delta_pitch_rate` 的唯一值个数全部为 `24`，`sd_delta_vx` 的唯一值个数分布为 `{22: 13, 23: 2, 24: 5}`。

聚合指标的变化为：

1. `vx.dir_match_mean`：`0.16875 -> 0.21458 -> 0.44583`
2. `pitch_rate.dir_match_mean`：`0.33125 -> 0.33125 -> 0.36667`
3. `pitch.dir_match_mean`：`0.44792 -> 0.44792 -> 0.45625`

这说明：

> 当前问题已不再是“动作根本传不到根部动力学”，而是“动作作用链已接通，但量纲、符号与耦合强度仍需进一步校准”。

### 修复后应如何定位剩余问题

修复后的剩余风险已从“结构性失敏”降级为“定量不准”。当前更准确的判断是：

1. grounded 相位的动作作用链已经具备；
2. airborne 相位也已具备显式动作响应；
3. 但 `pitch_rate` 的方向一致率仍偏低，说明空中惯性耦合先验只是补齐接口，不代表已接近真实动力学。

因此，本文件原结论中的“这是结构性作用链缺失，而不是训练不足”，在 2026-04-28 之后应更新为：

> 结构性作用链缺失问题已修复；下一阶段的主问题转为参数标定与多步误差控制。

### 第一轮监督校准结果

在修复作用链后，监督拟合脚本也同步修正了两个问题：

1. 旧版 `structured_state_loss()` 错把 `omega` 与 `activation` 混入角度环绕误差；
2. 旧版默认 `rollout_weight=0.0`，实际上并未对多步 rollout 做训练约束。

修订后：

1. 仅对真实角度通道 `theta, waist, hip, knee, ankle` 使用 wrapped residual；
2. 显式加大 `vx` 与 `omega` 的监督权重；
3. 将 `activation` 降为弱约束；
4. 默认启用 `rollout_horizon=5`、`rollout_weight=0.25`。

结果见：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_after_fix2_calib1/summary.json)

与修复后但未校准的基线相比：

1. `vx_mae`：`0.09437 -> 0.08113`
2. `pitch_rate_mae`：`0.88123 -> 0.86727`
3. `rollout_10_rmse`：`1.24607 -> 1.17632`
4. `rollout_20_rmse`：`1.21126 -> 1.08886`

但与此同时：

1. `one_step_struct_rmse`：`0.64091 -> 0.64322`
2. `vz_mae`：`0.38670 -> 0.39966`

因此这轮校准的最准确解读是：

> 新损失已经开始把模型往“更好的根部动作响应、以及更稳的长时 rollout”方向推，但它同时牺牲了部分一步整体误差与 `vz` 通道精度。

换言之，当前已不适合再把“单一总体 RMSE 最低”作为唯一优化目标。后续应把根部关键量与多步稳定性视为主要指标。

## 0. 目的

本文只审查两条最关键的根部动力学通道：

1. 前向速度 `vx`
2. 躯干俯仰角速度 `pitch_rate = omega`

审查问题是：

> 当前 `DMControlHopperSDModel` 对这两部分的建模，是否足以给 TD-MPC2 提供可靠的一步动作响应。

审查依据包括：

1. 代码静态分析；
2. 已运行的动作敏感性诊断结果：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000/summary.json)
3. 明细数据：
   [action_sensitivity_details.csv](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000/action_sensitivity_details.csv)

---

## 1. 结论摘要

结论非常明确：

> 当前模型中，`action` 对 `next_vx` 和 `next_omega` 在同一步内几乎没有直接作用；这不是“参数没学好”，而是更新方程的结构就把这条作用链切断了。

更具体地说：

1. `next_vx`、`next_vz` 使用的地面反作用力来自“当前时刻”的腿几何与 activation，而不是动作更新后的量；
2. `next_omega` 完全不依赖 `action`、也不依赖接触力矩，只依赖当前 `theta` 与 `omega` 的被动回正项；
3. 因此，在同一初始状态下，改变动作后，SD 的一步 `vx / vz / pitch_rate / pitch` 响应几乎是常数；
4. 这与真实环境中“动作符号一变，`vx` 与 `omega` 响应方向也随之改变”的行为不一致。

这解释了此前动作敏感性实验中最差的两项：

1. `vx` 方向一致率仅 `0.16875`
2. `pitch_rate` 方向一致率仅 `0.33125`

---

## 2. 代码层面的根因

### 2.1 `vx` 更新链

当前 `vx` 的更新为：

```text
next_vx = vx + dt * (F_x - damping_x * vx) / mass
```

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:812)

其中：

```text
F_x = F_leg * e_leg_x
```

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:807)

而 `F_leg` 的定义是：

```text
F_leg =
    G * relu(
        leg_stiffness * c_leg
        + leg_damping * d_c_leg
        + leg_force_max * activation
    )
```

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:802)

问题在于，这里的所有量都是“动作更新之前”的量：

1. `activation` 是当前状态中的 `activation`，不是 `next_activation`
2. `c_leg / d_c_leg / e_leg_x` 来自当前 `kin` 与当前 `touch`
3. 它们在计算 `F_leg` 时都没有使用 `next_q`、`next_qdot` 或 `next_activation`

而动作真正更新的是：

1. `next_activation`
2. `next_qdot`
3. `next_q`

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:790)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:794)

但这些更新后的量，并没有反馈到同一步的 `F_x`，只会进入：

1. `next_q`
2. `next_touch`
3. 下一步的几何状态

因此：

> 当前 `action` 对 `next_vx` 的同一步影响被结构性延迟到了下一步。

### 2.2 `pitch_rate` 更新链

当前 `omega` 的更新为：

```text
tau_bal = -balance_theta * theta - balance_omega * omega
next_omega = omega + dt * (tau_bal - angular_damping * omega) / inertia
```

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:810)
- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:814)

这意味着：

1. `next_omega` 完全不依赖 `action`
2. `next_omega` 也不依赖 `F_x / F_z`
3. `next_omega` 也不依赖接触点相对躯干的力矩臂

因此：

> 当前 `pitch_rate` 的一步更新不是“动作驱动的姿态动力学”，而只是“被动回正 + 阻尼”。

这不是近似过粗，而是作用链缺失。

### 2.3 `pitch` 也随之失真

当前：

```text
next_theta = theta + dt * next_omega
```

见：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py:816)

因为 `next_omega` 同一步与动作无关，所以：

1. `pitch` 的一步动作响应也几乎不依赖动作；
2. 它只能在更晚一步通过接触和几何变化间接体现动作影响。

这解释了为什么 `pitch` 的方向一致率也偏低。

---

## 3. 动作敏感性结果如何佐证这一点

正式动作敏感性结果见：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000/summary.json)

其中最关键的聚合指标为：

1. `vx.dir_match_mean = 0.16875`
2. `pitch.dir_match_mean = 0.447916...`
3. `pitch_rate.dir_match_mean = 0.33125`

更关键的不是“数值差”，而是明细里暴露出的结构性现象。

### 3.1 同一状态下，SD 的 `vx` 响应对所有动作都相同

对 20 个样本状态逐一统计，在每个样本内部，24 个动作对应的：

1. `sd_delta_vx`
2. `sd_delta_pitch_rate`
3. `sd_delta_vz`

唯一值个数都恒为 `1`。

这意味着：

> 在固定状态下，无论动作怎么变，SD 的一步 `vx / vz / pitch_rate` 响应都是常数。

这与真实环境显然不符。

### 3.2 具体样本中，动作正负翻转时，环境响应翻转，SD 不翻转

以样本 `sample_idx=0` 为例：

1. `waist_neg_0.25` 与 `waist_pos_0.25`
2. `hip_neg_0.25` 与 `hip_pos_0.25`
3. `knee_neg_0.25` 与 `knee_pos_0.25`

在环境中：

1. `env_delta_pitch_rate` 会随动作正负翻转而翻转；
2. `env_delta_vx` 也会因动作不同而变化方向或幅度。

但在 SD 中：

1. `sd_delta_vx` 两边都相同；
2. `sd_delta_pitch_rate` 两边也完全相同。

这说明不是“某些动作学偏了”，而是：

> 当前方程没有把动作符号信息传到这两条根部动力学通道。

### 3.3 分动作维度看，问题在所有关节上都存在

从明细聚合可见：

1. `vx` 对所有关节维度的方向一致率都很低；
2. `pitch_rate` 对所有关节维度都很低；
3. `waist` 在 `pitch_rate` 上的绝对误差尤其大。

这进一步说明：

1. 不是某一个 action 维度 mapping 错了；
2. 而是整条“action -> root dynamics”作用链没有进入根部更新。

---

## 4. 状态设计是否也是原因

### 4.1 对当前问题，状态缺失不是主因

当前最突出的错误，不是因为状态里缺了某个量，而是因为：

1. 已有状态量没有被正确用于根部更新；
2. 动作更新后的关节/接触变化没有进入同一步 `vx / omega` 计算。

因此，对本轮问题排序应当是：

1. **首先修方程作用链**
2. **其次再考虑状态是否还需补量**

### 4.2 但 `vx` 之后仍可能需要额外信息

即便把同一步作用链修好，`vx` 通道仍可能偏弱，因为当前状态与方程都没有显式表示：

1. 接触切向速度或滑移状态；
2. 地面切向反作用 / 摩擦模式；
3. 更直接的“支撑相位”或 stick-slip 信息。

因此：

1. 当前 `vx` 问题的第一层根因是“动作未进入同一步更新”；
2. 第二层根因才是“切向接触建模仍不够”。

---

## 5. 问题优先级排序

### P0：`action` 未进入同一步根部平动更新

当前 `next_vx` 与 `next_vz` 使用的 `F_leg` 是基于当前状态计算的，动作影响被延后到下一步。

影响：

1. planner 在短视野 rollout 中几乎看不到“动作如何立刻改变速度”；
2. `vx` 和 `vz` 的一步动作敏感性结构被破坏。

### P0：`action` 未进入同一步俯仰角速度更新

当前 `next_omega` 只由 `theta / omega` 的被动回正项给出。

影响：

1. 动作无法立刻改变俯仰角速度；
2. `pitch_rate` 的符号响应几乎必然错误；
3. `pitch` 的一步响应也随之失真。

### P1：`tau_bal` 过强地替代了真实动作-接触力矩链

即便作为先验，`tau_bal` 也不应成为 `omega` 的唯一驱动项。

影响：

1. 模型会把姿态变化解释为“总想回正”的被动系统；
2. 而不是“动作通过接触和几何关系改变俯仰”的系统。

### P1：切向接触 / 摩擦缺失会继续限制 `vx`

即便修正同一步作用链，`vx` 建模仍可能受限于没有显式切向接触机制。

### P2：`project_state()` 裁剪不是本轮主要根因

硬裁剪仍然是风险，但它无法解释“同一状态下不同动作给出完全相同的 `sd_delta_vx / sd_delta_pitch_rate`”这一现象，因此不是本轮首要问题。

---

## 6. 最小修订建议

本轮先给建议，不直接改代码。

### 建议 1：让根部动力学使用动作更新后的有效腿力

至少应让：

1. `F_leg`
2. `F_x`
3. `F_z`

依赖于：

1. `next_activation`
2. `next_q / next_qdot` 或中点近似几何

而不是只依赖当前状态。

目标不是“做更复杂的积分器”，而是先恢复：

> 当前动作能够在当前一步影响 `next_vx / next_vz`。

### 建议 2：为 `next_omega` 增加接触力矩项

最小版本可以采用：

```text
tau_contact = r_x * F_z - r_z * F_x
next_omega = omega + dt * (tau_contact + tau_bal - damping * omega) / inertia
```

其中 `r` 可用接触点相对躯干或相对髋部的力臂近似。

核心要求只有一个：

> `omega` 的一步更新必须显式依赖动作诱发的受力变化。

### 建议 3：将 `tau_bal` 降为弱先验，而不是唯一来源

`tau_bal` 可以保留为被动稳定项，但不应继续替代真实的动作-接触-姿态耦合。

### 建议 4：修完 P0 后再评估是否要补切向接触状态

若修完后：

1. `pitch_rate` 明显改善；
2. 但 `vx` 仍差；

则下一步再考虑为 `vx` 单独补：

1. 接触切向速度；
2. 滑移 / stick-slip 代理量；
3. 更合理的切向地面力近似。

---

## 7. 最终结论

当前 `vx / pitch_rate` 的建模问题，核心不在“参数没学好”，也不在“状态维度略少”，而在于：

> 现有更新方程没有把动作对根部动力学的同一步影响接入进去。

因此，这不是一个需要靠更多训练慢慢弥补的误差，而是一个必须先修正结构作用链的建模问题。

在修正前，TD-MPC2 planner 即使拿到再好的状态，也难以从 SD rollout 中看到“动作立刻如何改变前向速度与俯仰角速度”，这会直接削弱其规划价值。

## 2026-04-28 追加：受限内部残差对照

在完成作用链修补与纯结构监督校准后，又新增了一条 Hopper internal residual 对照。其设计边界与 `sd-mcts-pcsp` 中的 `transition residual` 一致：

1. 残差位于 transition 层，而不是 readout 层；
2. 残差只修正 `vx / vz / omega` 的一步增量；
3. 残差零初始化、`tanh` 限幅、全局尺度受控；
4. 残差默认关闭；
5. 不直接修改 `theta / q / touch / z_min` 等投影后量。

实现位置：

- [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py)

监督对照结果：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid1/summary.json)

与当前无残差最佳口径 `calib2` 比较：

1. `one_step_struct_rmse`：`0.64415 -> 0.62297`
2. `rollout_5_rmse`：`1.10797 -> 1.06285`
3. `rollout_10_rmse`：`1.17789 -> 1.12732`
4. `rollout_20_rmse`：`1.09166 -> 1.04595`
5. `vx_mae`：`0.09129 -> 0.06626`
6. `vz_mae`：`0.39855 -> 0.24343`
7. `pitch_rate_mae`：`0.86442 -> 0.73091`

这组结果说明：

> 当前 Hopper SD 的主要剩余误差，已经可以被“小幅、受约束的内部 transition residual”明显吸收；继续坚持纯结构化不再是最优工程路线。

需要保持的边界是：

1. 当前版本仍只是 `state + action -> tiny residual`，尚未使用历史窗口；
2. 它尚未接回 TD-MPC2 主训练，只完成了监督对照；
3. 其可接受性来自“increment residual 的受约束角色”，而不是重新放开自由黑盒旁路。

### 残差尺度首轮排序

随后对同一受限 residual 继续做了首轮尺度扫描：

1. `0.5x residual`
   - [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_transitionresid0p5/summary.json)
2. `1.0x residual`
   - [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid1/summary.json)
3. `2.0x residual`
   - [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid2p0/summary.json)

结果表现出清晰的单调趋势：

1. `0.5x`：`vx_mae = 0.07188`，`vz_mae = 0.31851`，`pitch_rate_mae = 0.79232`
2. `1.0x`：`vx_mae = 0.06626`，`vz_mae = 0.24343`，`pitch_rate_mae = 0.73091`
3. `2.0x`：`vx_mae = 0.06182`，`vz_mae = 0.13085`，`pitch_rate_mae = 0.63742`

同时，一步与多步整体误差也同步下降：

1. `one_step_struct_rmse`：`0.63235 -> 0.62297 -> 0.60854`
2. `rollout_20_rmse`：`1.06794 -> 1.04595 -> 1.00090`

据此，当前最准确的工程判断是：

> 在现有受限结构下，残差强度至少在 `0.5x -> 2.0x` 区间内仍未过冲；`2.0x residual` 是截至目前最强的监督基线。

## 2026-04-28 再追加：grounded 接触支撑回填后的中间结论

在完成 `4.0x residual` 责任分析后，又对 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 做了第二轮 grounded 主干修补。该轮不改 `STATE_DIM`，也不扩大 residual 作用域，只改 `forward()` 中的 grounded 接触力学：

1. 将原先“单接触点腿弹簧承担全部地面反力”的近似，改为“前掌/后跟分点法向支撑 + 切向滑移摩擦 + 分点力矩”；
2. 在此基础上，再加入一层最小的 grounded 支撑基线，使接触门打开时，主干先具备抵消重力的能力，再由足部几何与腿部驱动作修正。

对应的快速监督 smoke 结果为：

1. 无残差第一轮：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix_noresid_cpu/summary.json)
2. `4.0x residual` 第一轮：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix_resid4_cpu/summary.json)
3. 无残差第二轮：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix2_noresid_cpu/summary.json)
4. `4.0x residual` 第二轮：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix2_resid4_cpu/summary.json)
5. 第二轮 residual 责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix2_resid4_cpu/residual_usage_analysis.json)

这些 smoke 的最重要信号不是整体 RMSE，而是 `vz`：

1. 无残差 `vz_mae`：`0.3665 -> 0.3365`
2. `4.0x residual` `vz_mae`：`0.1621 -> 0.1753`
3. 第二轮 residual 责任分析中，`vz.mean_abs_base_delta`：`0.3615 -> 0.3239`

这说明：

1. grounded 接触支撑回填已经开始把一部分 `vz` 责任从 residual 拉回主干；
2. 但 residual 仍在大量 grounded 样本上以反向、同量级方式修正 `vz`；
3. 因此当前主问题虽然已经聚焦，但尚未解决。

本文件据此给出的更新判断是：

> 当前 `vx / pitch_rate` 问题已经不应再单独理解为“动作作用链问题”，而应理解为“grounded 接触支撑与滑移建模仍偏弱，进而连带污染 `vx / vz / omega` 三个根部通道”。

## 2026-04-28 再追加：支撑门陡化比接触锚定更有效

在 grounded 主干继续回填时，又单独比较了两类修补：

1. `_leg_mechanics()` 中的短时世界系接触锚定；
2. grounded 支撑基线从“线性门”改为“更陡的软门”。

结果表明：

1. 接触锚定对同口径 `12×250×30` 无残差 smoke 几乎没有带来可分辨改善；
2. 支撑门陡化后，无残差 `vz_mae` 从 `0.3365` 下降到 `0.3144`，`rollout_20_rmse` 从 `2.1232` 下降到 `2.0515`。

对应结果：

1. 锚定后同口径无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix3b_noresid_cpu/summary.json)
2. 支撑门陡化后无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_noresid_cpu/summary.json)
3. 支撑门陡化后 `4.0x residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_resid4_cpu/summary.json)
4. 支撑门陡化后责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_resid4_cpu/residual_usage_analysis.json)

并且 `vz` 责任分布继续朝正确方向移动：

1. `vz.mean_abs_base_delta`：`0.3239 -> 0.2941`
2. grounded `vz.mean_sign_match_with_base`：`0.0556 -> 0.2265`

这说明 residual 虽仍显著，但已经不再像之前那样几乎纯粹“整段反向对消”。

## 2026-04-28 再追加：feature-context residual 主要修复了 `vx` 的“输入过贫”，但未同时解决 `omega` 的长时稳定性

在 `contactfix5` 口径上，又对 transition residual 做了单独增强：残差头不再只读取裸 `state + action`，而是额外读取当前接触与主干提议量，包括 `joint_acc`、当前接触几何、`c_leg / d_c_leg / e_leg`、`toe/heel` 支撑分配、法向/切向接触量、滑移速度，以及主干自己的 `base_delta_vx / vz / omega`。

对应结果：

1. 旧版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_cpu/summary.json)
2. 新版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary.json)
3. 新版责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/residual_usage_analysis.json)

最关键的变化是：

1. `vx_mae`：`0.13342 -> 0.08705`
2. `vz_mae`：`0.17875 -> 0.15467`
3. `pitch_rate_mae`：`0.77846 -> 0.74535`
4. `rollout_20_rmse`：`1.64983 -> 1.67611`

因此，这轮增强带来的判断应分成两部分。

第一，`vx`：

1. `vx` 一步误差显著下降；
2. `vx.mean_ratio_resid_over_base` 已收敛到约 `1.52`；
3. 这说明 `vx` 通道此前确实存在“residual 看不到当前接触上下文，只能粗暴旁路修正”的问题。

第二，`omega`：

1. `pitch_rate_mae` 虽有改善，但改善幅度明显小于 `vx / vz`；
2. `omega.mean_abs_resid_delta` 仍约为 `0.56`，`mean_ratio_resid_over_base` 仍约为 `7.86`；
3. `rollout_20_rmse` 还略有回升。

据此，当前最准确的工程判断是：

> 把 residual 输入从“裸状态”升级为“结构上下文 + 主干提议量”是正确方向，而且已经证明 `vx` 的剩余误差很大一部分只是 residual 缺少对口信息。但 `omega` 的主要瓶颈并不只是输入过贫，还包含更强的多步累积偏差或俯仰力矩链建模误差。

也就是说，当前 `vx` 问题已经更像“局部余量补偿问题”，而 `omega` 问题仍然更像“局部修正 + 长时一致性”共同作用的复合问题。

对当前 TD-MPC2 Hopper 配置，还需要补充一条口径约束：由于规划和训练 rollout 的真实 `horizon = 3`，这里对 `vx / pitch_rate` 的主验收标准也应落在 `1 / 3 / 5` 步，而不是把 `10 / 20` 步误差当作主要否决项。后两者仍应保留，但只作为漂移诊断。

## 2026-04-28 再追加：短视界关键通道损失进一步压低了 `pitch_rate`，但也轻微拉回了 `vx / vz`

在不改模型结构的前提下，又新增了一条纯训练目标修订：对 `3 / 5` 步 rollout 的 `vx / vz / omega` 增加附加监督，其中 `omega` 权重略高。

对应结果：

1. 原 `feature-context residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary.json)
2. 短视界损失版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss_cpu/summary.json)

最关键的变化是：

1. `rollout_3_rmse`：`1.18643 -> 1.18081`
2. `rollout_5_rmse`：`1.45345 -> 1.44230`
3. `pitch_rate_mae`：`0.74535 -> 0.73885`
4. `vx_mae`：`0.08705 -> 0.09226`
5. `vz_mae`：`0.15467 -> 0.16231`

因此，这轮结果进一步支持了一个更细的判断：

> `omega` 的确能通过“更短视界、更有针对性的监督”继续被压低；但在当前权重下，这种压低会开始侵占 `vx / vz` 的单步精度预算。

换言之，当前问题已不再是“这类短视界损失有没有用”，而是“该把短视界一致性权重推到多大，才不会为了 `omega` 牺牲太多 `vx / vz`”。

围绕这一点，又继续做了 `0.10 / 0.15 / 0.25` 三个权重的同口径扫描。结果表明：

1. `0.10` 已经能把 `rollout_3/5` 继续压低，同时几乎不损伤 `vx`，`vz` 只轻微回退；
2. `0.15` 虽然把 `pitch_rate_mae` 压得更低，但 `vz` 回退更明显；
3. `0.25` 已经进入“为了 `omega` 牺牲过多 `vz`”的区间。

因此，若当前目标是给 TD-MPC2 提供更可靠的 `horizon=3` rollout，而不是单独追 `omega` 最低点，那么 `short_rollout_weight = 0.10` 是目前更合适的工程折中。
