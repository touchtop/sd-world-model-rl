# Hopper SD 与规划目标解耦记录（2026-04-29）

## 背景

在当前 `pure-env + sd_latent_dim=17 + internal transition residual + supervised init` 主线上，已经观察到如下事实：

1. `SD dynamics` 的独立监督短视界指标相较早期版本已有明显改善；
2. 在线 RL 训练中，`consistency_loss` 在前半程下降，但并未持续单调下降；
3. `100k` 闭环行为已从“后仰后僵死/乱滚”发展为“后仰后可恢复到前倾推进”，但行为主模态仍为爬行，而不是 `upright -> prepared -> rise -> landing`；
4. `75k` policy replay 诊断显示：常规 grounded 段短视界误差已进入可用区间，但 `landing / takeoff / high_omega / high_vz_impact` 仍是显著薄弱相位。

因此，本轮工作的核心问题不再是“是否继续设计奖励”，而是：

> 当前是否应继续把 `SD dynamics` 的拟合目标与 TD-MPC2 的闭环规划目标混合为一个单一优化问题。

## 高层方法论判断

本轮分析收敛到以下方法论结论：

> “动力学拟合准确”与“规划必然受益”不是同一目标，不应预设二者天然一致。

理由如下。

### 1. 目标函数不同

`SD dynamics` 的拟合目标是：给定真实 `(s_t, a_t)`，尽量准确预测 `s_{t+1}` 或短视界 rollout。

TD-MPC2 的规划目标是：在候选动作之间，尽量把真实长期回报更高的动作排序在前。

二者只有在误差类型、分布和任务敏感方向高度对齐时才可能自然一致。对 Hopper-Hop 这种强接触、强非线性任务，不能默认这种一致性成立。

### 2. 规划真正关心的是“任务相关误差”，不是平均 RMSE

对动力学监督而言，平均 `MSE / RMSE / MAE` 是自然目标。

对规划而言，不同误差的重要性完全不同。例如：

1. `landing` 时刻的 `omega` 偏差可能直接改变动作优劣排序；
2. `takeoff` 时刻的接触开启/关闭时序偏差会系统性破坏规划可行性判断；
3. 某些常规 grounded 段的小误差虽然能降低平均 `RMSE`，却未必改变动作选择。

因此，平均 `consistency_loss` 改善并不能自动推出规划一定改善；反之，某些面向动作排序的局部修正，也不必然使整体拟合指标同步最优。

### 3. 当前联合损失只会优化“闭环有用的 world model”，不会自动维护“全相位精确的 SD 拟合器”

在线 RL 训练中的总损失见 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py)：

1. `consistency_loss`
2. `reward_loss`
3. `value_loss`
4. `termination_loss`
5. `height_loss`（当前基本关闭）

这说明训练目标不是纯 `SD` 参数辨识，而是一个联合 world-model 目标。其结果是：

1. 模型会优先适配当前 policy / planner 频繁访问的分布；
2. 高频普通样本会压过低频关键相位；
3. 对平均一致性有利的调整，不必然对 `landing / takeoff / high_omega` 有利。

这不是实现错误，而是联合目标的自然结果。

### 4. 当前经验现象已经支持“目标不一致”这一判断

当前最直接的证据包括：

1. `consistency_loss` 在 `10k-50k` 区间下降，但在 `50k` 之后反弹；
2. `50k -> 75k` 的 replay 诊断中，`support_contact` 略有改善，但 `rollout_3_rmse`、`pitch_rate_mae`、`high_omega omega_mae` 没有同步改善；
3. `100k` 闭环评估中所有 episode 均非零，但行为改善主要表现为“后仰失败态 -> 前倾推进态”的恢复子程序，而不是 `hop` 行为链形成。

这组现象与“联合目标偏向当前闭环可用性，而非持续压低 SD 误差”完全一致。

## 当前证据整理

### 1. 在线一致性损失并未持续单调下降

基于 [train.csv](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_probe/train.csv) 的分段均值：

1. `10k-15k`: `consistency_loss ≈ 0.920`
2. `15k-25k`: `≈ 0.875`
3. `25k-35k`: `≈ 0.866`
4. `35k-50k`: `≈ 0.838`
5. `50k-65k`: `≈ 0.886`
6. `65k-75k`: `≈ 0.926`
7. `75k-87.2k`: `≈ 1.006`

这表明：

> 当前联合 RL 训练并不会持续把 `SD` 误差单调压低，而是在中期以后出现了反弹。

### 2. `75k` replay 上的 dynamics 质量仍有明显结构性薄弱点

诊断文件：

- [diagnostics_20260429_seed3_resume75k_policy_error.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/diagnostics_20260429_seed3_resume75k_policy_error.json)

关键结果：

1. `one_step_struct_rmse = 1.1488`
2. `rollout_3_rmse = 1.1839`
3. `rollout_5_rmse = 1.2154`
4. `pitch_rate_mae = 1.3819`
5. `landing omega_mae = 3.3107`
6. `takeoff dominant_state_match = 0.3222`
7. `high_vz_impact omega_mae = 3.5503`

这说明：

1. 常规 grounded 段和短视界常规 rollout 已进入“可工作”区间；
2. 但接触转换与高角速度相位仍远未达到“低误差”水平；
3. 规划最敏感的相位正是当前最不稳定的相位。

### 3. `100k` 闭环行为改善主要体现为姿态重组，而不是跳跃链条形成

在 [step_0100001](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_probe/official_eval/step_0100001) 中，10 条评估轨迹全部非零，均分约 `67.61`。但状态统计显示：

1. `grounded_prone_gt_05 ≈ 0.772`
2. `grounded_upright_gt_05 = 0`
3. `grounded_prepared_gt_05 = 0`
4. `full_foot_support_gt_05 ≈ 0.0038`

这说明：

> 当前行为改善主要来自“后仰/坐地失败态 -> 前倾推进态”的恢复能力，而不是 `hop` 动作链的形成。

## 工程决策

本轮据此做出如下高层决策。

### 决策 1：正式将验证目标拆分为两条线

后续 Hopper 主线不再把“SD 拟合质量”与“TD-MPC2 规划表现”视为同一个验证对象，而拆为：

1. `model fidelity`
2. `planning usefulness`

分别回答：

1. `SD` 是否准确描述当前 replay 分布上的短视界动力学；
2. 当前 `SD` 是否足以改善动作排序、规划稳定性和样本效率。

### 决策 2：若进一步改训练方式，应优先采用“共享 transition core + 独立在线 SD 监督支路”

不建议引入两个互相竞争的 dynamics 头。更合理的方向是：

1. 保持单一共享的 `encode / next` transition core；
2. RL 仍使用现有 world-model 联合目标；
3. 额外增加一条面向 replay 真实转移的在线 `SD` 监督更新支路；
4. 该支路优先压：
   1. `1 / 3 / 5` 步 rollout 误差，
   2. `omega / vz / contact`,
   3. `landing / takeoff / near_contact / high_omega` 的 phase-weighted 误差。

这样既能保持 planner 使用的世界模型唯一且清晰，又能避免联合 RL 目标持续侵蚀 `SD` 的短视界保真度。

### 决策 3：后续不再把“持续加长 RL 训练步数”视为首选修复手段

当前证据表明，仅靠继续增加 RL 步数，并不能保证 `SD` 精度继续改善。后续优先级应为：

1. 先把在线 `SD` 监督目标独立出来；
2. 再比较是否改善关键相位误差；
3. 再判断闭环是否真正从“稳定爬行”跃迁到“可跳跃”。

## 后续最小执行方案

### 阶段 A：文档与验证口径收敛

1. 在研究文档中明确区分 `model fidelity` 与 `planning usefulness`；
2. 将当前 `75k` replay 误差、`100k` 行为形态、一致性损失反弹事实固定为基线；
3. 不再用单一 `consistency_loss` 讲述全部方法结论。

### 阶段 B：训练实现改造

建议后续实现一个最小版本的在线 SD 监督支路：

1. 输入：当前 replay buffer 中的真实 `(obs_t, a_t, obs_{t+1:t+H})`
2. 模型：共享现有 `SDWorldModel._dynamics`
3. 优化器：可与 RL 主优化器分离，也可交替更新
4. 损失：
   1. 一步结构态误差
   2. `1 / 3 / 5` 步 rollout 误差
   3. `landing / takeoff / near_contact / high_omega` phase-weighted 误差

### 阶段 C：受控对照实验

最小闭环对照应包括：

1. 仅 RL 联合目标
2. RL 联合目标 + 在线 SD 监督支路
3. 若资源允许，再比较 `phase-weighted` 与 `uniform` 的在线 `SD` 监督

评价时同时看：

1. `consistency_loss` 是否仍反弹；
2. `75k / 100k` replay 上 `landing / takeoff / high_omega` 是否改善；
3. `100k` 行为是否仍主要停留在爬行模态。

## 小结

本轮工作的关键结论不是“当前 `SD` 无用”，而是：

> 当前 Hopper 线已经足够清楚地表明：`SD` 拟合与规划受益不是天然一致的两个目标。若希望继续验证显式结构化 world model 的价值，就必须把这两条目标在方法论和训练设计上显式拆开。

这一定义了下一阶段工作的正确方向。
