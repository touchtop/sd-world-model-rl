# Hopper SD World Model 设计与当前训练结果

## 1. 文档目的

本文档整理当前 Hopper `SD world model` 的独立设计口径，覆盖以下内容：

1. 当前活跃实现的目标与边界；
2. `DMControlHopperSDModel` 的 17 维结构状态定义；
3. 主干一步动力学推进、几何与接触近似；
4. 内部 `transition residual` 的设计、输入、输出与训练边界；
5. `SDWorldModel` 如何接入 TD-MPC2 的 `encode / next / reward / Q / pi` 接口；
6. 当前监督拟合脚本、最佳 checkpoint、学习到的关键参数与短视界结果；
7. 当前已知局限，以及为什么当前推荐口径是 `contactfix5 + internal residual + shortloss010`。

本文档只讨论当前 `SD dynamics` 主线，不再讨论旧的结构化奖励设计。

## 2. 当前活跃口径

当前 Hopper 主线的事实口径如下：

1. 显式动力学主干文件为 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py)；
2. TD-MPC2 外层包装文件为 [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py)；
3. 当前最佳监督初始化 checkpoint 为 [hopper_sd_supervised.pt](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt)；
4. 当前推荐闭环实验口径脚本为 [run_hopper_v5_5k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_5k_pureenv_sdinit.sh) 与 [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)；
5. 当前闭环验证不再混入 shaping，`sd_reward_mode=learned`，仅保留环境奖励监督；
6. 当前闭环推荐 `sd_latent_dim=17`，即关闭外层 latent residual，只保留 `DMControlHopperSDModel` 内部的 `transition residual`。

这里需要明确区分两类“残差”：

1. 外层 `SDWorldModel` 自带的 latent residual heads：
   `self._residual_encoder / self._residual_dynamics / self._struct_residual`
2. 内层 `DMControlHopperSDModel` 的一步物理转移残差：
   `transition_residual_head`

当前活跃主线只保留第 2 类。由于 `sd_latent_dim=17`，当前闭环口径中 `SDWorldModel.resid_dim = 0`，外层 latent residual 实际关闭。

## 3. 设计目标

当前设计的核心目标不是“解释所有 Hopper 细节”，而是提供一个可被 TD-MPC2 直接调用的、短视界可用的 world model core。更具体地说，当前目标是：

1. `encode(obs) -> z` 能把环境状态稳定映射到一个固定 17 维结构态；
2. `next(z, a) -> z'` 在 1 步与 3 步视界内有足够精度；
3. `z` 中的关键量对规划有直接意义，尤其是 `vx / vz / pitch / omega / contact`；
4. 保留显式几何与接触语义，使误差来源可以被继续分析；
5. 不要求完全纯结构化；如果主干仍有系统误差，允许用受限 residual 只补最难的通道。

因此，这一版的立场是：

> SD 仍是主干，residual 只是受约束的局部误差补偿器。

## 4. 17 维结构状态定义

当前 `STATE_DIM = 17`。状态布局由 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中的 `latent_layout` 与索引常量共同确定。

### 4.1 状态表

| 索引 | 名称 | 含义 |
| --- | --- | --- |
| 0 | `H` | 躯干高度，代码中同时视作 `rootz / torso_z / height` |
| 1 | `vx` | 水平速度 |
| 2 | `vz` | 垂直速度 |
| 3 | `theta` | 主躯干 pitch |
| 4 | `omega` | 主躯干角速度 |
| 5 | `A` | 激活代理量 `activation` |
| 6 | `waist` | 腰关节角 |
| 7 | `hip` | 髋关节角 |
| 8 | `knee` | 膝关节角 |
| 9 | `ankle` | 踝关节角 |
| 10 | `waist_dot` | 腰关节角速度 |
| 11 | `hip_dot` | 髋关节角速度 |
| 12 | `knee_dot` | 膝关节角速度 |
| 13 | `ankle_dot` | 踝关节角速度 |
| 14 | `touch_toe` | 前掌触地接触强度，已从原始传感器值映射到 `[0, 1]` |
| 15 | `touch_heel` | 后掌触地接触强度，已映射到 `[0, 1]` |
| 16 | `z_min` | 全身代表几何点的最低高度，由当前几何重算 |

### 4.2 从环境观测到结构态

环境原始状态观测是 15 维：

1. `position[6]`
2. `velocity[7]`
3. `touch[2]`

编码过程见 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中 `encode_observation()` 与 `obs_to_state()`：

1. 所有关节角经过角度包裹；
2. `touch` 通过 `1 - exp(-clip(x, 0, +inf))` 压到 `[0, 1]`；
3. `activation` 不直接来自观测，而由当前 `touch`、`knee`、关节角速度构造；
4. `z_min` 不是观测原值，而是由当前几何链重新计算。

因此，当前 17 维状态已经不是简单的原始观测拷贝，而是：

> 原始可观测量 + 显式派生几何/接触量

## 5. 主干几何与接触建模

### 5.1 刚体链与代表几何

当前主干保留了一套固定的 Hopper 几何近似常量，例如：

1. `TORSO_TO_PELVIS_Z`
2. `PELVIS_TO_THIGH_Z`
3. `THIGH_TO_CALF_Z`
4. `CALF_TO_FOOT_Z`
5. `NOSE / TORSO / PELVIS / THIGH / CALF / FOOT` 各自的几何半径与局部段参数

在每一步中，模型会先通过 `_absolute_angles()` 与 `_rotate_local()` 计算：

1. `torso / pelvis / thigh / calf / foot` 的绝对方向；
2. `toe / heel / nose / torso / pelvis / thigh / calf / foot` 的世界坐标；
3. 各几何段的最低点；
4. `x_min / x_max / z_min`。

这部分的作用不是为了追求高精度外观重建，而是为了让：

1. 接触近似有明确几何基础；
2. `z_min`、`body collision`、`foot support` 等特征可被一致导出；
3. 状态标签、动作先验、奖励探针仍可复用同一套结构量。

### 5.2 足部接触与支撑分配

当前接触建模不是“是否触地”的硬判别，而是由几部分共同构成：

1. `touch_toe / touch_heel` 的软接触门；
2. `toe_z / heel_z` 的几何近地门；
3. 地面穿透深度 `pen`；
4. 接触点垂直速度 `toe_vz / heel_vz`；
5. 由 `toe_gate / heel_gate` 派生出的总支撑门 `gate`；
6. 基于接触与支撑投影的 `toe_share / heel_share` 分配。

`_contact_reaction()` 中会进一步计算：

1. `toe_normal / heel_normal`
2. `toe_fx / heel_fx`
3. `F_x / F_z`
4. `tau_contact`
5. `toe_vx / heel_vx / toe_vz / heel_vz`

这些量既被主干显式动力学使用，也被内部 residual 读取。

### 5.3 支撑腿压缩与驱动

当前主干保留了一条显式“支撑腿力学”链：

1. 用接触点和髋位置计算腿向量；
2. 由 `rest_leg_length` 得到压缩量 `c_leg`；
3. 用下一时刻名义几何近似 `d_c_leg`；
4. 以 `leg_stiffness * c_leg + leg_damping * d_c_leg + leg_force_max * activation` 形成 `leg_drive`；
5. 再沿腿方向拆成 `leg_drive_x / leg_drive_z`。

这条链的作用是提供一个可解释的“腿像弹簧-阻尼-受激驱动器”的主干先验，而不是把根部平动完全交给黑盒 MLP。

## 6. 一步动力学推进

`forward(state, action)` 的当前流程可以概括为：

1. 解包当前 17 维结构态；
2. 更新激活代理 `activation`；
3. 由动作、关节阻尼、关节回中项得到 `joint_acc`；
4. 更新 `qdot` 与 `q`；
5. 用更新后的关节位形计算 actuation-aware 几何；
6. 由腿压缩与触地状态得到接触力 `F_x / F_z` 与接触力矩 `tau_contact`；
7. 加上空中动作耦合项 `air_ax / tau_air` 与平衡项 `tau_bal`；
8. 先得到无 residual 的基线：
   `next_vx_base / next_vz_base / next_omega_base`
9. 如启用内部 `transition residual`，主 residual 只修正上述三个量；另有一个实验性 `touch residual` 仅修正 `touch_toe / touch_heel`。
10. 再积分得到 `next_H` 与 `next_theta`；
11. 由新几何重算 `touch_toe / touch_heel / z_min`；
12. 统一经 `project_state()` 投影回合法范围。

当前主干最重要的工程取舍是：

1. `q / qdot / activation / contact / z_min` 保持显式；
2. 最难拟合的 `vx / vz / omega` 允许小范围 residual 修正；
3. `project_state()` 最终仍重新用几何链计算 `z_min`，而不是相信 latent 里旧值。

## 7. 内部 transition residual 设计

### 7.1 设计目标

内部 residual 的目标不是重写整套动力学，而是只补主干最难拟合的误差。当前分两部分：

1. 主 residual：`vx`
2. 主 residual：`vz`
3. 主 residual：`omega`
4. 实验性 `touch residual`：`touch_toe`
5. 实验性 `touch residual`：`touch_heel`

这是一个非常关键的边界。当前 residual 不直接改：

1. `H`
2. `theta`
3. `q / qdot`
4. `touch`
5. `z_min`

因此，显式主干仍然负责几何闭环与接触语义。

### 7.2 网络结构与初始化

当 `use_transition_residual=true` 时，`DMControlHopperSDModel` 会创建：

1. 一个两层 MLP `transition_residual_head`
2. 一个两层 MLP `touch_transition_residual_head`
2. 输入维度：
   `STATE_DIM + ACTION_DIM + TRANSITION_RESIDUAL_EXTRA_DIM`
3. `transition_residual_head` 输出维度：`3`
4. `touch_transition_residual_head` 输出维度：`2`

关键约束如下：

1. 最后一层权重和偏置全零初始化；
2. 输出经 `tanh` 限幅；
3. 再乘 `transition_residual_base_scale = [0.10, 0.18, 0.30]`；
4. `touch residual` 再乘 `touch_transition_residual_base_scale = [0.05, 0.07]`；
5. 再乘全局系数 `transition_residual_scale`；
6. `touch residual` 还会乘一个近地几何 gate，避免在明显离地时任意篡改接触语义。

因此实际残差项是：

`residual_delta = transition_residual_scale * base_scale * tanh(head(input))`

`touch_residual_delta = transition_residual_scale * touch_base_scale * tanh(touch_head(input))`

这意味着初始时 residual 精确为 0，训练初期不会破坏主干显式动力学。

需要说明的是：`touch residual` 当前仍处于实验阶段。现有最小监督微调结果表明，它可以改善局部 `heel_touch / foot_grounded` 缺失，但如果缺少同时约束 `grounded_any / any_contact_z` 的多步一致性，容易对几何主链造成负迁移。因此当前推荐基线仍应以“主 residual 已验证、touch residual 待进一步约束”来表述。

### 7.3 当前 residual 输入

当前 residual 输入不是裸 `state + action`，而是：

1. 当前 17 维结构状态 `state`
2. 当前 4 维动作 `action`
3. 主干已算出的 `next_activation`
4. `joint_acc`
5. 当前几何特征：
   `toe_z, heel_z, hip_x, hip_z, torso_z, z_min`
6. 当前腿力学特征：
   `c_leg, d_c_leg, e_leg_x, e_leg_z`
7. 当前接触特征：
   `gate, support_gate, toe_gate, heel_gate, toe_share, heel_share`
8. 当前接触力与力矩：
   `toe_normal, heel_normal, F_x, F_z, tau_contact`
9. 当前足端速度：
   `toe_vx, heel_vx, toe_vz, heel_vz`
10. 主干自己的基线提议量：
    `base_delta_vx, base_delta_vz, base_delta_omega`
11. `airborne_gate`

这条设计的核心思想是：

> residual 读取“当前结构上下文 + 主干自己的解释”，只学习尚未解释掉的那部分余量。

因此它不是一个无约束的替代模型，而是一个针对主干误差的校正器。

### 7.4 当前 active residual 口径

当前监督与闭环主线使用：

1. `use_transition_residual=true`
2. `transition_residual_scale=4.0`

对应脚本见：

1. [run_hopper_v5_5k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_5k_pureenv_sdinit.sh)
2. [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)

## 8. SDWorldModel 外层接口与 TD-MPC2 接入

### 8.1 encode / next

`SDWorldModel` 是 TD-MPC2 的正式 world model 适配层。当前关键事实是：

1. `encode(obs) -> z` 直接调用 `DMControlHopperSDModel.encode_observation()`；
2. `next(z, a) -> z'` 直接调用 `DMControlHopperSDModel.forward()`；
3. 在当前活跃口径 `sd_latent_dim=17` 下，`z` 就是 17 维结构态本身；
4. 因此当前规划和训练 rollout 都在显式 SD 状态空间内进行，不再带外层 latent residual 子空间。

### 8.2 reward / Q / pi / termination

外层 [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py) 仍保留 TD-MPC2 标准接口：

1. `reward(z, a, task, next_z=None)`
2. `Q(z, a, task, ...)`
3. `pi(z, task)`
4. `termination(z, task)`

当前纯环境奖励主线中：

1. `reward_mode=learned`
2. `mixed_reward_target()` 直接返回环境奖励；
3. `reward()` 在训练和评估时都只使用 learned reward head；
4. 不再在规划阶段额外叠加 shaping。

这意味着当前我们验证的是：

> 改进后的 SD dynamics 能否作为 TD-MPC2 的 transition core 直接工作，而不是靠奖励设计补救。

### 8.3 外层 latent residual 当前是否启用

`SDWorldModel` 仍然保留外层 residual 机制，条件是：

1. `sd_latent_dim > sd_struct_dim`

这时会创建：

1. `_residual_encoder`
2. `_residual_dynamics`
3. `_struct_residual`

但当前 active line 明确设置：

1. `sd_struct_dim=17`
2. `sd_latent_dim=17`

因此：

1. `resid_dim = 0`
2. 外层 latent residual 头全部关闭

这也是当前文档需要特别强调的一点：当前主线的“残差”专指 `DMControlHopperSDModel` 内部的 transition residual，而不是旧的外层 latent residual。

## 9. 独立监督拟合脚本

### 9.1 脚本目的

监督脚本 [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py) 的目的不是训练 TD-MPC2，而是单独回答：

> 当前 SD dynamics 能否把模拟器的一步和短视界转移拟合到足以支持规划的程度。

### 9.2 数据与训练方式

当前脚本流程为：

1. 在 `dm_control.suite.hopper.hop` 中用随机动作采样数据；
2. 按 episode 级切分训练/验证集；
3. 监督目标是 `encode_observation(raw_obs_t)` 到 `encode_observation(raw_obs_{t+1})` 的结构态转移；
4. 主损失是加权 `smooth_l1` 的 `structured_state_loss`；
5. 可选附加：
   1. 多步 rollout loss
   2. 面向 `vx / vz / omega` 的短视界附加损失

当前状态损失权重对关键根部量更高：

1. `vx` 权重 `3.0`
2. `vz` 权重 `3.0`
3. `omega` 权重 `3.0`
4. `theta` 权重 `1.5`

这反映了当前工程优先级：先让规划最依赖的根部运动量更准。

### 9.3 当前最佳监督口径

当前最佳监督基线是：

1. `contactfix5` 主干
2. `use_transition_residual=true`
3. `transition_residual_scale=4.0`
4. residual 使用 feature-context 输入
5. `short_rollout_horizons=[3, 5]`
6. `short_rollout_weight=0.10`

对应输出目录为：

[run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu)

最佳 checkpoint 为：

[hopper_sd_supervised.pt](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt)

## 10. 当前最佳监督结果

### 10.1 主结果

当前最佳 `summary.json` 在：

[summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/summary.json)

关键指标如下：

| 指标 | 数值 |
| --- | --- |
| `one_step_struct_rmse` | `0.7184` |
| `rollout_1_rmse` | `0.6743` |
| `rollout_3_rmse` | `1.1788` |
| `rollout_5_rmse` | `1.4369` |
| `rollout_10_rmse` | `1.7100` |
| `rollout_20_rmse` | `1.6566` |
| `height_mae` | `0.00567` |
| `vx_mae` | `0.08640` |
| `vz_mae` | `0.15638` |
| `pitch_mae` | `0.03091` |
| `pitch_rate_mae` | `0.73936` |
| `support_contact_mae` | `0.23627` |
| `full_foot_support_mae` | `0.04425` |
| `body_contact_z_mae` | `0.00712` |

### 10.2 与上一基线比较

相对未加短视界附加损失的 `feature-context residual` 基线：

[summary_short_horizon_eval.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary_short_horizon_eval.json)

改进方向是清楚的：

1. `rollout_3_rmse`: `1.1864 -> 1.1788`
2. `rollout_5_rmse`: `1.4534 -> 1.4369`
3. `vx_mae`: `0.08705 -> 0.08640`
4. `pitch_rate_mae`: `0.74535 -> 0.73936`

代价是：

1. `vz_mae`: `0.15467 -> 0.15638`

因此当前工程判断是：

> `short_rollout_weight = 0.10` 是当前最平衡的监督口径，既改善了 TD-MPC2 真正使用的 `1/3/5` 步区间，又没有像更大权重那样明显破坏 `vx / vz`。

## 11. 当前学习到的关键参数

当前最佳监督模型导出的关键参数如下，来源同样是 [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/summary.json)。

### 11.1 激活与腿部参数

1. `activation_w = [0.2150, 0.2307, 0.2189, 0.1369]`
2. `activation_b = -0.1285`
3. `activation_tau = 0.0859`
4. `leg_stiffness = 5.8846`
5. `leg_damping = 0.7798`
6. `leg_force_max = 2.9043`
7. `rest_leg_length = 0.82`

### 11.2 接触与阻尼参数

1. `contact_k = 9.9720`
2. `contact_tau = 0.3702`
3. `linear_damping = [0.1147, 0.1943]`
4. `angular_damping = 0.1271`
5. `balance_theta = 1.3919`
6. `balance_omega = 0.3685`

### 11.3 关节驱动参数

1. `joint_action_scale = [1.3481, 1.4983, 1.6570, 1.2127]`
2. `joint_rate_damping = [1.0696, 1.1698, 1.2922, 0.9466]`
3. `joint_centering = [0.2107, 0.2354, 0.3942, 0.2571]`

### 11.4 residual 参数

1. `use_transition_residual = true`
2. `transition_residual_scale = 4.0`
3. `transition_residual_base_scale = [0.10, 0.18, 0.30]`
4. `touch_transition_residual_base_scale = [0.05, 0.07]`

这些数值说明当前监督训练并未把主干参数推到异常量级，整体仍处在较合理的可解释范围内。

## 12. 闭环接入与当前证据

### 12.1 监督初始化接线

训练入口 [train.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/train.py) 已支持：

1. `sd_supervised_init_from=<checkpoint>`

`SDWorldModel.load_supervised_dynamics()` 只会加载：

1. `_dynamics` 的权重

不会加载：

1. reward head
2. Q head
3. pi head
4. 外层 latent residual 头

这保证了监督拟合只初始化显式动力学，而 TD-MPC2 下游头仍在 RL 中重新学习。

### 12.2 当前纯环境闭环 smoke

当前纯环境闭环 smoke 实验为：

[sd_hopper_5k_pureenv_sdinit_seed3_20260428](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_5k_pureenv_sdinit_seed3_20260428)

其结果是：

1. 训练稳定；
2. `2500` 与 `5000` 都没有明显行为突破；
3. `5000` 时只有极小非零分；
4. 状态仍主要停留在失败地面姿态。

这说明：

1. 监督初始化后的 SD 没有把闭环训练弄坏；
2. 但 Hopper 在纯环境奖励下仍然很难，仅凭 `5k` 预算还看不到决定性收益。

## 13. 当前已知局限

当前版本虽然已经比早先主线显著更好，但还存在明确限制。

### 13.1 长视界仍会漂

当前验收主指标已经明确转向 `1 / 3 / 5` 步，而不是 `20` 步。这不是偶然，而是因为：

1. TD-MPC2 Hopper 当前 `horizon = 3`
2. 当前长视界 rollout 仍然会累积漂移

因此当前版本的准确表述应是：

> 已逐步逼近“短视界规划可用”，但远未达到“长视界自由仿真替代器”。

### 13.2 `omega` 仍是最顽固通道

尽管 `pitch_rate_mae` 已有改善，但它仍显著高于 `height`、`vx` 等更稳定通道。这意味着：

1. 躯干旋转链仍未被完全刻画；
2. 接触力矩、空中耦合项和关节驱动链还可能继续优化。

### 13.3 支撑相关量仍偏难

`support_contact_mae` 仍在 `0.23` 左右，说明：

1. 虽然 `full_foot_support_mae` 已不大；
2. 但更细粒度的支撑质量与接触转换仍不够精确。

### 13.4 闭环 RL 证据仍在积累

当前最强证据仍来自独立监督拟合，而不是长期闭环 RL 已显著超过基线。也就是说：

1. 现在可以说 `SD core` 比之前更可信；
2. 但还不能说它已经完全证明能稳定替代黑盒 world model。

## 14. 当前结论

截至 `2026-04-28`，对 Hopper SD 主线最准确的工程结论是：

1. 当前活跃实现是 `contactfix5` 主干加受限内部 `transition residual`；
2. 当前闭环主线禁用了外层 latent residual，`sd_latent_dim=17`；
3. residual 只修 `vx / vz / omega`，没有破坏显式几何闭环；
4. residual 输入已从裸 `state + action` 升级为“结构上下文 + 接触特征 + 主干提议量”；
5. `short_rollout_weight=0.10` 的监督口径是当前最优折中；
6. 当前最佳 checkpoint 已足以支撑继续做纯环境闭环验证；
7. 但当前证据仍只能支持“短视界可用性显著增强”，还不能宣称 Hopper SD 已完全解决。

一句话概括：

> 当前版本已经不是“纯拍脑袋的显式模型”，而是一套以显式几何和接触为主干、以受限 residual 补关键根部通道、并经短视界监督拟合校准的 Hopper SD world model。
