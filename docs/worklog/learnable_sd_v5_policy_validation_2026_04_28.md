# Hopper SD Policy-Distribution 验证记录（2026-04-28）

## 目标

在当时的 `pure-env + sd_latent_dim=17 + internal transition residual + supervised init` 诊断口径上，补齐一项此前缺失的关键验证：

> 当前 `SD dynamics` 在随机动作监督数据上表现较好，但在 TD-MPC2 policy 真正访问到的 replay 分布上，误差是否显著恶化；若恶化，在线 RL 微调是否已经开始修正这种 distribution shift。

本轮不再讨论结构化奖励，不再修改 reward 设计，只围绕 `transition core` 的可用性进行验证。

## 新增脚本

新增脚本：

- [analyze_hopper_sd_policy_distribution_error.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_sd_policy_distribution_error.py)

用途如下：

1. 读取一个 Hopper dynamics checkpoint。支持两类输入：
   1. 监督初始化 checkpoint `hopper_sd_supervised.pt`
   2. TD-MPC2 的 `25000.pt / 25000.full.pt` 中的 `_dynamics.*` 权重
2. 读取一个或多个 `full.pt`，直接复用其中保存的 replay episode
3. 在 policy 访问过的 `(obs_t, action_t, obs_{t+1})` 分布上评估：
   1. `one_step_struct_rmse`
   2. `rollout_1 / 3 / 5 rmse`
   3. `vx / vz / pitch / pitch_rate / contact` 等关键量误差
   4. `airborne_true / landing / takeoff / high_omega / high_vz_impact / near_contact` 的 phase-sliced 误差

这样后续不需要重采样环境，即可持续比较：

1. 随机监督初始化模型
2. RL 早期/中期在线微调后的 dynamics

## 既有随机验证集结论

来自监督初始化 summary 的关键指标为：

- `one_step_struct_rmse = 0.7184`
- `rollout_1_rmse = 0.6743`
- `rollout_3_rmse = 1.1788`
- `rollout_5_rmse = 1.4369`
- `vx_mae = 0.0864`
- `vz_mae = 0.1564`
- `pitch_rate_mae = 0.7394`

phase-sliced 诊断已经表明：

1. `airborne_true` 相位明显比 grounded 更难；
2. `high_omega` 相位误差显著放大；
3. `landing / high_vz_impact` 在随机验证集里样本过少，统计不足。

因此，随机验证集已经提示：

> 当前 SD core 适合继续做闭环验证，但最敏感的空中与接触转换相位仍然是风险点。

## 25k policy replay 验证：监督初始化模型

使用三种子 `100k pureenv_sdinit` 在 `25k` 时保存的三个 `full.pt` replay，合并得到：

- `73,500` 条 transitions
- `147` 个 episodes

在该 policy 分布上，用监督初始化 checkpoint 直接评估，得到：

- `one_step_struct_rmse = 1.3873`
- `rollout_1_rmse = 1.1775`
- `rollout_3_rmse = 1.6014`
- `rollout_5_rmse = 1.8625`
- `vx_mae = 0.1246`
- `vz_mae = 0.2213`
- `pitch_rate_mae = 1.4527`

与随机验证集相比，误差出现了明显恶化，尤其是：

1. `one_step_struct_rmse` 由 `0.7184` 升至 `1.3873`
2. `pitch_rate_mae` 由 `0.7394` 升至 `1.4527`
3. `rollout_5_rmse` 由 `1.4369` 升至 `1.8625`

这说明：

> 监督初始化模型在随机动作分布上可用，但在 policy replay 分布上存在明确的 distribution shift。

### 关键相位

在三种子合并 replay 上，监督初始化模型的关键相位误差如下：

- `airborne_true`：`vz_mae = 0.5016`，`omega_mae = 2.5991`
- `landing`：`dominant_state_match = 0.7359`，`vz_mae = 0.9285`，`omega_mae = 3.2547`
- `takeoff`：`dominant_state_match = 0.5714`
- `high_omega`：`vx_mae = 0.2070`，`vz_mae = 0.3976`，`omega_mae = 2.9081`
- `high_vz_impact`：`vz_mae = 1.0662`，`omega_mae = 3.4135`

可见当前最脆弱的仍然是：

1. 空中相位，
2. 落地冲击相位，
3. 高角速度相位，
4. 起跳/离地判别。

## 同分布对照：seed2 replay 上，监督初始化模型 vs RL 25k 模型

为了避免“replay 分布不同”导致误读，又在完全相同的 `seed2@25k replay` 上做了同分布对照。

### 监督初始化模型

- `one_step_struct_rmse = 1.4311`
- `rollout_1_rmse = 1.2190`
- `rollout_3_rmse = 1.7066`
- `rollout_5_rmse = 1.8591`
- `vx_mae = 0.1240`
- `vz_mae = 0.2308`
- `pitch_rate_mae = 1.4572`

### seed2 当前 RL 模型

- `one_step_struct_rmse = 1.2628`
- `rollout_1_rmse = 1.0847`
- `rollout_3_rmse = 1.2627`
- `rollout_5_rmse = 1.2967`
- `vx_mae = 0.1246`
- `vz_mae = 0.2065`
- `pitch_rate_mae = 1.2133`

### 对照结论

在线 RL 微调已经开始修正 replay 分布上的误差，尤其体现在：

1. `rollout_3_rmse`：`1.7066 -> 1.2627`
2. `rollout_5_rmse`：`1.8591 -> 1.2967`
3. `pitch_rate_mae`：`1.4572 -> 1.2133`
4. `vz_mae`：`0.2308 -> 0.2065`

这表明：

> 监督初始化没有白费，且在线训练并非只是在学 Q / policy，下游 replay 也在反向修正 dynamics 对 policy 分布的适配性。

但仍需注意两点。

第一，`vx_mae` 基本没有改善，说明平动前向推进仍不稳定。

第二，某些关键相位仍明显偏弱。例如在 `seed2@25k replay` 上：

- `landing dominant_state_match` 只有 `0.6857`
- `takeoff dominant_state_match` 只有 `0.1967`
- `high_vz_impact omega_mae` 仍有 `2.8085`

因此当前更准确的表述应为：

> RL 已经对 policy 分布误差进行了部分纠偏，但接触转换与离地判别仍是当前 SD core 的主要瓶颈。

## 25k 闭环行为状态

三种子 `100k pureenv_sdinit` 的第一次正式评估均已落盘：

- seed1：`eval mean = 0.1867`
- seed2：`eval mean = 0.4930`
- seed3：`eval mean = 0.0553`

从 `eval_episode_summary.csv` 可见：

1. `grounded_upright_frac = 0`
2. `grounded_prepared_frac = 0`
3. 绝大多数 episode 仍集中在 `grounded_rear / grounded_prone`

因此到 `25k` 为止，可以明确说明：

> 当前闭环训练尚未把行为推进到可用的 upright / prepared 区域。

但结合上文的 policy-distribution 验证，也可以同时说明：

> 这不是“SD 完全失效”，而更像是：当前 dynamics 已能在 replay 分布上部分自修正，但最关键的起跳、落地、高角速度接触转换仍不足以支撑成功 hop 行为。

## 当前阶段结论

本轮验证给出三条比之前更清楚的结论。

第一，`policy-distribution shift` 是真实存在的，不能再只依赖随机动作监督指标。

第二，在线 RL 微调已经能够明显改善 `3/5` 步 rollout 误差，这说明当前 `SD + internal residual` 主线具备继续闭环优化的价值。

第三，真正限制 Hopper-Hop 成功行为的仍然是接触转换相关相位，尤其是：

1. `landing`
2. `takeoff`
3. `high_vz_impact`
4. `high_omega`

因此，下一阶段重点应继续放在：

1. policy-distribution 下的 phase-sliced 误差持续跟踪，
2. action ranking diagnostic，
3. 接触转换与离地判别的进一步建模检查，

而不是重新回到 reward shaping 路线。
