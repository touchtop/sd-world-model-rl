# Learnable SD v5: TD-MPC2 接口复核与修复记录（2026-04-30）

## 背景

本轮复核针对当前激活路径：

1. `world_model_kind=sd_hopper`
2. `sd_use_transition_residual=true`
3. `sd_online_supervision_enabled=true`
4. `sd_reward_mode=learned`

复核重点是 `SDWorldModel` 与 TD-MPC2 主训练循环之间的参数归属、损失归属和规划接口一致性。

## 已处理问题

### 1. `_dynamics` 双 Adam 管理

原实现中，`model._dynamics.parameters()` 同时属于 TD-MPC2 主优化器 `self.optim` 和在线 `SD` 监督优化器 `self.sd_optim`。这会使同一组参数被两个 Adam 优化器分别维护动量和二阶矩，导致有效学习率、动量历史和权重更新不可解释。

修复后，若 `sd_online_supervision_enabled=true`，`_dynamics` 从主优化器参数组中移除，只由 `sd_optim` 管理。主优化器继续管理 encoder、reward、Q、residual heads 等 TD-MPC2 侧参数。主梯度裁剪也同步改为只裁剪主优化器实际拥有的参数集合。

该修复对应文件：

1. [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py)

### 2. 趋势保护维度的绝对最小修正下限

原实现中，`trend_protect_min_correction=0.02` 被直接用于 `ROOTZ / TOE_TOUCH / HEEL_TOUCH / ZMIN` 等受保护维度。该值对接触概率维度和米制高度维度含义不同，尤其在高度变化很小的阶段可能引入单位错配的虚假漂移。

修复后，`sd_trend_protect_min_correction` 显式配置化，并将默认值改为 `0.0`。趋势保护仍保留相对比例约束 `sd_trend_protect_max_scale=0.20`，但不再强制任何维度产生绝对最小变化。

该修复对应文件：

1. [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py)
2. [config.yaml](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/config.yaml)

### 3. 在线 SD 监督目标与规划转移的归属关系

当前在线 `SD` 监督仍直接作用于 `_dynamics(true_struct, action)`，不经过外层 `struct_residual` 与 residual latent。该设计保留，因为其目的不是监督完整 world model，而是锚定共享系统动力学 core。

本轮修复通过优化器归属完成边界收紧：`_dynamics` 只由在线 `SD` 监督更新，外层 residual heads 由 TD-MPC2 主损失更新。这样，`SD core` 与 residual 补偿的职责更清楚，避免同一 `_dynamics` 参数同时被主一致性目标和独立监督目标用不同 Adam 状态拉扯。

复审后补充一项实现收紧：启用 `sd_online_supervision_enabled=true` 且 `sd_detach_core_in_main_loss=true` 时，`SDWorldModel.next()` 中 `_dynamics` 输出的 `next_struct_base` 在主 consistency 路径中 detach 后再参与 residual 合成。这样，`consistency_loss` 不再给 `_dynamics` 计算随后被丢弃的梯度，也避免读者误解为主 consistency 仍在训练 `SD core`。此时主 consistency 的结构态误差主要训练外层 `struct_residual`，`SD core` 由独立在线 `SD` 监督训练。

### 4. 结构态 residual 覆盖风险

结构态 residual 没有被移除，因为当前 Hopper 动力机制复杂，显式系统动力学结构态不应被要求完整覆盖所有规划相关因素。残差保留是主线设计的一部分。

为提高可审计性，本轮将各通道 residual cap 显式配置化，并新增默认关闭的 `sd_struct_residual_penalty_coef`。该正则项作用于 `tanh(raw_struct_residual)` 的平方均值，后续可用于检查 residual 是否过度覆盖系统动力学 core。当前默认关闭，是为了不在未完成 smoke 与消融前引入新的行为变量。

对应新增配置包括：

1. `sd_struct_residual_scale`
2. `sd_vx_residual_scale`
3. `sd_vz_residual_scale`
4. `sd_pitch_residual_scale`
5. `sd_pitch_rate_residual_scale`
6. `sd_knee_residual_scale`
7. `sd_struct_residual_penalty_coef`
8. `sd_detach_core_in_main_loss`

训练日志中新增 `struct_residual_activity=mean(abs(tanh(raw_struct_residual)))`。该指标不直接代表物理误差，但可以反映外层结构态 residual 是否长期处在高饱和修正区间。

### 5. learned reward 目标尺度

原实现中，`mixed_reward_target()` 已先计算 `reward_flat = sd_env_reward_scale * env_reward`，但在 `sd_reward_mode=learned` 分支返回了未缩放的 `env_reward`。当前主线 `sd_env_reward_scale=1.0` 时不显性影响结果，但未来切换 reward scale、hybrid 或 curriculum 时会形成静默不一致。

修复后，learned 模式也统一返回已经缩放的 `reward_flat`。

## 断点恢复兼容性

由于主优化器参数组发生变化，旧 full checkpoint 中的 `optim` 状态可能无法严格载入。当前恢复逻辑改为：若主优化器或 `sd_optim` 的 state dict 因参数组不匹配而报错，则跳过对应 optimizer state，并继续恢复模型、target Q、detach Q、policy optimizer、scale、planner 均值、replay buffer 和训练计数器。

该行为用于兼容接口修复前后的 full checkpoint，但不改变模型权重恢复逻辑。复审后，异常捕获范围从 `ValueError` 扩展为 `(ValueError, KeyError, RuntimeError)`，告警中同时输出旧 optimizer 参数组数与新 optimizer 参数组数，便于事后定位。

## 下一次 smoke 观测点

下一次 5k 或 10k smoke 应明确记录以下指标：

1. `struct_consistency_loss / resid_consistency_loss` 的相对量级；
2. `struct_residual_activity` 是否长期接近高值；
3. `sd_online_supervision_loss`、`sd_online_one_step_loss`、`sd_online_rollout_loss` 是否继续下降；
4. `sd_online_planning_weight_mean` 是否维持在接近 1 的范围；
5. 训练中 episode return 是否仍出现起身、姿态转正或高分片段。

若 residual activity 持续偏高且闭环行为主要依赖 residual 兜底，应将 `sd_vz_residual_scale` 与 `sd_pitch_rate_residual_scale` 的默认值从历史设置同步收紧一档，例如先试 `vz=1.0`、`pitch_rate=3.0`，并可配合 `sd_struct_residual_penalty_coef=0.05` 做小预算消融。当前不直接收紧默认值，是为了避免在没有 smoke 证据前把“优化器归属修复”和“残差容量收缩”两个变量混在一起。

## 当前边界

本轮没有把 online `SD` 监督改为监督完整 `model.next()`。原因是当前方法论仍需要区分系统动力学 core 的保真锚点与 TD-MPC2 外层 residual 的规划补偿作用。若后续证据显示 residual 覆盖风险仍显著，应优先通过 `sd_struct_residual_penalty_coef` 和 residual scale 消融处理，而不是取消 residual 机制。

当前正在运行的 planning-aware 100k 三种子实验是在本轮接口修复前启动的，其进程内代码不会自动吸收这些修复。该实验可作为旧接口版本的对照记录；若需要评估本轮修复后的效果，应使用新 `exp_name` 重新启动 smoke 或正式三种子实验。
