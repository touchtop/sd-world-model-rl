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
3. `sd_latent_dim=17`
4. `sd_use_transition_residual=true`
5. `sd_transition_residual_scale=4.0`
6. `sd_reward_mode=learned`
7. `sd_action_prior_enabled=false`
8. 不使用旧的结构化奖励塑形

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
