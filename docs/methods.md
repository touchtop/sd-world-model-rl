# Methods

## Hopper：SD 拟合与规划验证的当前方法口径

### 目标

当前 Hopper 主线的研究目标不再是设计奖励塑形，而是检验：

1. 当前 `SD transition core` 是否能够在 TD-MPC2 使用的状态-动作分布上保持足够的短视界动力学保真度；
2. 该 core 是否能够改善 TD-MPC2 的动作排序、规划稳定性与闭环样本效率。

因此，方法上必须明确区分以下两个对象：

1. `model fidelity`
2. `planning usefulness`

二者相关，但不再预设天然一致。

### 当前活跃训练口径

当前闭环 Hopper 主线采用如下设置：

1. `world_model_kind=sd_hopper`
2. `sd_struct_dim=17`
3. `sd_latent_dim=512`
4. `sd_use_transition_residual=true`
5. `sd_transition_residual_scale=4.0`
6. `sd_reward_mode=learned`
7. `sd_action_prior_enabled=false`
8. 不使用旧的结构化奖励塑形
9. `consistency_loss` 中的结构态与残差态默认按块等权，而不是按维度数加权
10. SD 结构态一致性损失使用 batch 目标标准差做逐通道归一化

对应脚本为：

- [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)
- [run_hopper_v5_200k_pureenv_sdinit_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh)

### 当前在线训练的损失结构

在 RL 闭环中，`SDWorldModel` 不是独立按监督目标训练，而是作为 TD-MPC2 world model 与 reward/Q 一起联合优化。当前总损失由以下部分组成：

1. `consistency_loss`
2. `reward_loss`
3. `value_loss`
4. `termination_loss`
5. `height_loss`（当前配置中基本关闭）

这意味着在线训练优化的是“对当前 TD-MPC2 闭环有用的 world model”，而不是“平均动力学误差最小的 SD 拟合器”。

截至 `2026-04-29` 的修复版本中，若 `world_model_kind=sd_hopper`，`consistency_loss` 不再直接对完整 512 维 latent 做统一均方误差，而是先拆成 17 维显式结构态与 learned residual latent 两个块。结构态块按当前 batch 中目标结构态的逐通道标准差进行归一化，并设置标量下限防止近常量通道数值爆炸；残差块保留普通均方误差。两个块默认等权合成，由 `sd_latent_consistency_struct_weight=1.0` 与 `sd_latent_consistency_resid_weight=1.0` 控制。

这一处理保留了 learned residual latent 对未建模动力机制的补偿作用，但避免 495 维残差块仅因维度更多而在 `consistency_loss` 中压过 17 维结构态。结构态的 batch 标准差归一化用于避免 `omega / qdot / vz` 等大尺度通道主导梯度，使 `H / touch_toe / touch_heel / z_min` 等接触相位相关通道在优化目标中保持可见。

### 诊断方法

当前优先采用以下诊断：

1. `policy-distribution error`
2. `phase-sliced error`
3. `action ranking diagnostic`
4. `touch onset timing`
5. `consistency_loss` 的阶段趋势分析

其中：

1. `policy-distribution error` 用于判断随机监督初始化与当前 replay 分布之间的偏移；
2. `phase-sliced error` 用于识别 `landing / takeoff / high_omega / near_contact` 等关键相位；
3. `action ranking diagnostic` 用于判断模型是否能支持规划真正需要的动作排序；
4. `consistency_loss` 仅作为在线联合训练中的粗粒度诊断，不再被视为对“规划可用性”的充分代理。

### 当前方法论结论

截至 `2026-04-29`，当前主线采用以下解释框架：

1. 若 `consistency_loss` 下降，不自动推出规划质量提升；
2. 若 return 提升，不自动推出 `SD` 已被良好拟合；
3. 对 Hopper 而言，`landing / takeoff / high_omega / high_vz_impact` 的误差优先级高于平均 grounded 段误差；
4. 规划真正依赖的是任务相关的短视界动力学与动作排序，而不是统一意义上的平均 RMSE。

### 下一阶段推荐方法

若继续改训练方式，当前推荐方法不是构造两个竞争的 dynamics 头，而是：

1. 保持单一共享的 `SD transition core`；
2. 保留现有 RL 联合目标；
3. 额外增加一条独立的在线 `SD` 监督支路，直接作用于同一 core；
4. 该支路优先优化：
   1. 一步结构态误差，
   2. `1 / 3 / 5` 步 rollout 误差，
   3. `landing / takeoff / near_contact / high_omega` 相位加权误差。

这样可以在不污染 planner 接口的前提下，显式维护 `SD` 的短视界保真度。

### 当前已落地的最小实现

截至 `2026-04-29`，上述方向已先落地最小版本，其实现位于 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py)。

实现原则如下：

1. 不新增第二个 dynamics 头；
2. 继续复用当前 `SDWorldModel._dynamics` 作为唯一 transition core；
3. 在 TD-MPC2 每次 `_update()` 的主模型更新与 target-Q 更新之后，再执行一次独立的 `sd_optim` 监督更新；
4. 监督数据直接来自当前 replay batch 的真实 `obs, action, obs_next`。

### 当前在线监督支路的损失构成

当前最小在线监督支路由以下部分组成：

1. `sd_online_one_step_loss`：基于真实结构态的 `1-step` 转移误差；
2. `sd_online_rollout_loss`：基于同一 batch 的短视界 rollout 误差；
3. `sd_online_omega_loss`：对 `pitch_rate / omega` 通道的附加监督；
4. `sd_online_contact_loss`：对 `touch_toe, touch_heel` 接触通道的附加监督。

总损失为：

1. `sd_online_supervision_one_step_weight * one_step_loss`
2. `+ sd_online_supervision_rollout_weight * rollout_loss`
3. `+ sd_online_supervision_omega_weight * omega_loss`
4. `+ sd_online_supervision_contact_weight * contact_loss`

上述 `one_step_loss`、`rollout_loss`、`omega_loss` 与 `contact_loss` 均使用同一批目标结构态的逐通道标准差做尺度归一化。相位加权用于提高 `landing / takeoff / near_contact / high_omega` 的样本权重，通道尺度归一化用于校正不同物理量单位和数值范围造成的梯度偏置，二者作用不同。

当前尺度口径为：`sd_struct_loss_scale_mode=batch_std`，`sd_struct_loss_scale_floor=0.1`。即每个 replay batch 内先按目标结构态估计通道标准差，再用 `0.1` 作为统一下限。该尺度来自当前训练分布的目标统计量，不是手工指定的固定物理通道系数。

### 当前 rollout 监督视界

当前最小实现先支持：

1. `1-step`
2. `3-step`

原因不是方法论限制，而是当前 [buffer.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/buffer.py) 的 `sample()` 只返回 `horizon+1` 长度窗口，而当前 Hopper 训练配置中 `horizon=3`。因此，在不改采样接口的前提下，当前最自然、最干净的最小实现就是 `1-step + 3-step`。

若后续需要显式加入 `5-step` 监督，应同步修改 replay 采样窗口长度或额外增加独立监督采样路径。

### 当前关键相位加权

当前在线监督不是完全均匀加权，而是对以下相位提高权重：

1. `landing`
2. `takeoff`
3. `near_contact`
4. `high_omega`

其动机是：这些相位对 TD-MPC2 的动作排序与短视界规划最敏感，而它们也是此前 replay 诊断中误差最顽固的部分。

### 下一版本：planning-aware SD 权重入口

截至 `2026-04-30`，下一版本已加入默认关闭的 planning-aware 样本权重入口。该入口不直接改变 planner，也不把 `SD` 训练目标改成最大化回报，而是在现有结构态监督锚点上，根据 TD 目标或奖励目标对 replay 样本加权。

新增配置为：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

可选模式包括 `td_positive`、`td_abs`、`reward_positive` 与 `reward_abs`。启用后，权重会同时作用于 `1-step` 结构态监督和当前短视界 rollout 监督。权重按当前 batch 的 planning signal 均值归一化，即 `sd_online_planning_weight_coef=1.0` 时，未截断前的平均权重约为 `1`，避免该机制退化为整体提高 `SD` 学习率。若正值模式下当前 batch 没有正 signal，则回退为均匀权重。

该入口用于验证：相比平均状态预测误差，按 planning signal 重加权的 `SD` 监督是否更能改善动作排序和闭环样本效率。

该版本仍不是完整的 action-ranking consistency loss。完整排序目标需要额外构造动作 bank，并用真实短 rollout 或高保真离线诊断得到动作优劣标签；当前权重入口只是为该方向提供低风险的第一步。
