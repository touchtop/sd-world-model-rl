# Hopper Hop 可学习系统动力学模型设计：v5 极简实现版

## 0. 文档目的

本文档描述当前仓库中 `world_model_kind=sd_hopper` 的真实实现口径，覆盖三层内容：

1. 显式 `SD` 核，即 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py)；
2. 外层 `SDWorldModel` 包装，即 [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py)；
3. `TD-MPC2` 规划器中对结构化奖励与动作先验的接入方式，即 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py)。

本文档不再混入“未来建议但尚未实现”的口径。若代码与旧设计讨论不一致，以当前代码行为为准。

---

## 1. 总体结构

当前 `v5` 实现不是“纯 6 维显式模型”，而是三层耦合结构。

### 1.1 显式结构态

显式结构态维度固定为 `17`，由 `DMControlHopperSDModel` 负责编码、递推和投影。

其中：

1. 主存量仍是 `H, Vx, Vz, Theta, Omega, A` 六个；
2. 另外 `11` 维是 rollout 所必需的辅助递推槽位，而不是额外概念性主存量。

### 1.2 残差潜变量

外层 `SDWorldModel` 默认采用混合 latent：

1. 前 `17` 维为显式结构态；
2. 后 `sd_latent_dim - 17` 维为可学习残差态。

当前实验脚本实际使用：

1. `sd_struct_dim=17`
2. `sd_latent_dim=512`

因此当前主线实际 latent 为：

1. 显式结构态 `17` 维；
2. 残差态 `495` 维。

### 1.3 三种奖励通道

当前实现中同时存在三种奖励相关量：

1. 环境真实奖励的显式近似 `predict_reward`
2. 稠密姿态势函数加里程碑迁移的结构化塑形 `predict_shaping_reward`
3. 奖励头 `_reward` 学到的可学习奖励 logits

这三者并不在所有阶段等价使用，后文会分别说明。

### 1.4 规划中的条件动作先验

当前代码仍保留条件动作先验接口，但默认配置已切到禁用态：

1. `SDWorldModel` 会继续暴露 `action_prior_bounds` 接口；
2. 默认返回全范围 `[-1, 1]`，即不主动收缩规划器动作盒；
3. 只有显式传入 `sd_action_prior_enabled=true` 时，才重新启用 `rear/prone` 条件动作盒。

因此，当前“结构化先验”的主通道是奖励混合监督；动作盒仍可用于后续对照实验，但不再默认主导规划。

---

## 2. 原始观测与显式状态

### 2.1 原始观测

`dm_control.suite.hopper.hop` 的 state observation 在当前实现中按 15 维平铺使用：

```text
[rootz, rooty, waist, hip, knee, ankle,
 x_dot, z_dot, rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot,
 touch_toe, touch_heel]
```

其中：

1. `rootz` 是根部竖直位置；
2. `rooty` 是躯干俯仰角；
3. `waist, hip, knee, ankle` 是四个关节角；
4. `x_dot, z_dot, rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot` 是对应速度；
5. `touch_toe, touch_heel` 是前掌、后跟接触传感器原始值。

在编码阶段：

1. 所有关节角会先 wrap 到 `[-pi, pi]`；
2. 接触原始值会通过 `1 - exp(-clamp(x, 0))` 压到 `[0, 1)`。

### 2.2 显式结构态 17 维布局

当前显式结构态张量定义为：

```text
0  H                torso 高度（世界坐标）
1  Vx               水平速度
2  Vz               竖直速度
3  Theta            躯干俯仰角
4  Omega            躯干俯仰角速度
5  A                激活态
6  waist            腰关节角
7  hip              髋关节角
8  knee             膝关节角
9  ankle            踝关节角
10 waist_dot        腰关节角速度
11 hip_dot          髋关节角速度
12 knee_dot         膝关节角速度
13 ankle_dot        踝关节角速度
14 touch_toe        前掌接触强度
15 touch_heel       后跟接触强度
16 z_min            当前姿态下所有几何代表点的最小 z 值
```

其中：

1. `H` 不是原始观测里的 `rootz`，而是 `TORSO_BASE_Z + rootz`；
2. `z_min` 不是直接观测，而是由当前姿态几何计算得到；
3. 关节角与关节角速度被显式保留，是因为当前实现里的接触、腿长、状态判定与动作先验都需要它们。

### 2.3 状态投影

每一步显式递推后都会经过 `project_state` 投影：

1. `H ∈ [0, 3.5]`
2. `Vx ∈ [-10, 10]`
3. `Vz ∈ [-12, 12]`
4. `Theta` wrap 到 `[-pi, pi]`
5. `Omega ∈ [-20, 20]`
6. `A ∈ [0, 1]`
7. 四个关节角全部 wrap 到 `[-pi, pi]`
8. 四个关节角速度全部裁到 `[-20, 20]`
9. `touch_toe, touch_heel ∈ [0, 1]`
10. `z_min` 不是沿用输入槽位，而是基于投影后的 `H, Theta, waist, hip, knee, ankle` 重新做一次几何计算，再裁到 `[-0.5, 3.0]`

因此，当前显式核确实带有“按构造投影”的结构约束，并且 `z_min` 在显式核与外层 residual 回填后都保持几何一致。

---

## 3. 几何与派生量

### 3.1 几何链

当前实现显式写死了一套二维几何常数，用于从：

1. `H, Theta`
2. `waist, hip, knee, ankle`

递推出：

1. `torso, pelvis, thigh, calf, foot` 的世界坐标；
2. `toe, heel` 的世界坐标；
3. 各几何段的最低点；
4. `z_min`。

这一部分不是从 simulator 读取 site 坐标，而是用固定几何参数在 `SD` 核内部做正向运动学近似。

### 3.2 接触门控

接触门控定义为“接触强度门 × 几何近地门”的乘积：

```math
G_{toe} = \sigma(k_c (touch_{toe} - \tau_c)) \cdot \sigma((z_{margin} - z_{toe}) / s_z)
```

```math
G_{heel} = \sigma(k_c (touch_{heel} - \tau_c)) \cdot \sigma((z_{margin} - z_{heel}) / s_z)
```

```math
G = \max(G_{toe}, G_{heel})
```

其中：

1. `k_c` 对应 `contact_k`
2. `tau_c` 对应 `contact_tau`
3. `z_margin` 对应 `CONTACT_MARGIN`
4. `s_z` 对应 `CONTACT_SCALE`

因此 `G` 仍是软接触门，但已经不再允许“仅凭 carry-touch 残留就持续施力”。

### 3.3 等效接触点

等效接触点由前掌与后跟的接触强度加权得到：

```math
p_{contact} =
\frac{touch_{toe} p_{toe} + touch_{heel} p_{heel}}
{touch_{toe} + touch_{heel} + \varepsilon}
```

若两者都几乎为零，则退化为 `toe` 与 `heel` 的中点。

### 3.4 腿部几何与力学量

当前显式派生量包括：

1. `r_leg = p_hip - p_contact`
2. `leg_length = ||r_leg||`
3. `c_leg = clamp(rest_leg_length - leg_length, min=0)`
4. `d_c_leg = (next_c_leg - c_leg) / (dt * action_repeat)`
5. `e_leg = r_leg / ||r_leg||`

其中 `d_c_leg` 不是由历史缓存直接差分，而是通过当前名义下一步姿态 `next_kin_nominal` 计算出来。

### 3.5 编码阶段的激活代理

编码原始观测时，`A` 不是直接从动作积分得到，而是先由当前观测构造一个激活代理：

```math
A_{enc} = clamp(support \cdot knee\_gate \cdot motion\_gate, 0, 1)
```

其中：

1. `support = max(touch_toe, touch_heel)`
2. `knee_gate = sigmoid((knee - 1.15) / 0.22)`
3. `motion_gate = sigmoid((||qdot|| - 1.00) / 0.35)`

因此，编码时的 `A` 表示“当前姿态下的激活倾向”，不是纯动作历史。

---

## 4. 显式动力学更新

### 4.1 时间步长

当前显式核使用：

1. `dt = 0.02`
2. `ENV_ACTION_REPEAT = 2`

所以实际一步递推使用：

```math
\Delta t = 0.04
```

### 4.2 激活动态

动作先被压成一个标量激活目标：

```math
A^{des} = \sigma((u \odot w_A).sum + b_A)
```

然后按一阶迟滞更新：

```math
A_{t+1} = clamp\left(A_t + \Delta t \frac{A^{des} - A_t}{\tau_A}, 0, 1\right)
```

其中：

1. `w_A` 对应 `activation_w`
2. `b_A` 对应 `activation_b`
3. `tau_A` 对应 `activation_tau`

### 4.3 关节子系统

当前实现不是只递推躯干六维，而是同时递推四个关节角与四个角速度：

```math
\ddot q =
s_u \odot u
- d_q \odot \dot q
- k_q \odot (q - q_{rest})
```

```math
\dot q_{t+1} = \dot q_t + \Delta t \ddot q
```

```math
q_{t+1} = q_t + \Delta t \dot q_{t+1}
```

其中：

1. `s_u` 对应 `joint_action_scale`
2. `d_q` 对应 `joint_rate_damping`
3. `k_q` 对应 `joint_centering`
4. `q_rest` 固定为 `[0, 0, 1.55, 0]`

### 4.4 腿推力

腿推力定义为：

```math
F_{leg} = G \cdot ReLU(k_{leg} c_{leg} + b_{leg} d_{c,leg} + F_{max} A_t)
```

再沿腿方向分解：

```math
F_x = F_{leg} e_{leg}^{(x)}, \quad F_z = F_{leg} e_{leg}^{(z)}
```

### 4.5 姿态恢复

姿态力矩是最简 `PD` 形式：

```math
\tau_{bal} = -K_\theta \Theta - K_\omega \Omega
```

### 4.6 平动与转动更新

当前递推为：

```math
Vx_{t+1} = Vx_t + \Delta t \frac{F_x - d_x Vx_t}{m}
```

```math
Vz_{t+1} = Vz_t + \Delta t \frac{F_z - mg - d_z Vz_t}{m}
```

```math
\Omega_{t+1} = \Omega_t + \Delta t \frac{\tau_{bal} - d_\theta \Omega_t}{I}
```

```math
H_{t+1} = clamp(H_t + \Delta t \, Vz_{t+1}, 0, 3.5)
```

```math
\Theta_{t+1} = wrap(\Theta_t + \Delta t \, \Omega_{t+1})
```

### 4.7 接触强度递推

下一步前掌、后跟接触不是直接由几何硬阈值得到，而是带平滑记忆的更新：

```math
touch' = clamp(\rho \, touch + (1-\rho) \, target, 0, 1)
```

其中：

1. `rho = CONTACT_CARRY = 0.45`
2. `target` 由“近地 + 向下或准静态”构成：
   - `near_ground`
   - `downward`
   - `quasi_static`

因此接触信号在显式核中也是连续强度，不是离散接触。

---

## 5. 显式核中的可学习参数

当前显式核实际学习的参数包括：

### 5.1 激活相关

1. `activation_w`
2. `activation_b`
3. `activation_tau`

### 5.2 腿力学相关

1. `leg_stiffness`
2. `leg_damping`
3. `leg_force_max`

### 5.3 接触门控相关

1. `contact_k`
2. `contact_tau`

### 5.4 躯干阻尼与平衡相关

1. `linear_damping[2]`
2. `angular_damping`
3. `balance_theta`
4. `balance_omega`

### 5.5 关节子系统相关

1. `joint_action_scale[4]`
2. `joint_rate_damping[4]`
3. `joint_centering[4]`

当前固定不学习的量包括：

1. `mass = 1.0`
2. `inertia = 1.0`
3. `gravity = 9.81`
4. `rest_leg_length = 0.82`
5. 几何长度常数

---

## 6. 外层 SDWorldModel 包装

### 6.1 混合 latent

`SDWorldModel` 的编码方式为：

1. 前 `17` 维由显式核 `encode_observation` 产生；
2. 若 `latent_dim > 17`，则剩余维度由 `_residual_encoder(obs)` 生成；
3. 两部分拼接成总 latent。

### 6.2 下一步 latent 更新

下一步 latent 由三部分组成：

1. 显式核给出的 `next_struct_base`
2. 结构残差 `_struct_residual`
3. 残差态递推 `_residual_dynamics`

当前实现不是“显式核完全决定结构态”。相反，结构态会被 `_struct_residual` 做受限修正。

### 6.3 结构残差的保护机制

当前有一组 trend-protected 维度：

1. `H / ROOTZ`
2. `touch_toe`
3. `touch_heel`
4. `z_min`

对这些维度，结构残差不能任意改写方向和幅度，而是只能在显式核基准增量附近做受限放缩。

其余“自由维度”允许按不同尺度做 `tanh` 修正，其中：

1. `vx`
2. `vz`
3. `pitch`
4. `pitch_rate`
5. `knee`

拥有单独更大的修正尺度。

因此，当前真实实现不是“无 residual 的纯灰箱核”，而是“显式灰箱核 + 受限结构残差 + 高维残差子空间”。

---

## 7. 六个互斥状态的当前实现

当前状态不是软门值相乘后直接参与奖励，而是先由 `_state_reward_features` 计算，再用 `_dominant_state_ids` 取 dominant state。

状态顺序固定为：

1. `0 = falling_air`
2. `1 = grounded_rear`
3. `2 = grounded_prone`
4. `3 = grounded_upright`
5. `4 = grounded_prepared`
6. `5 = rising_air`

高层状态标签当前恢复为闭环六状态，不再保留未命名高层标签。

具体做法是：

1. 候选量仍由 `_state_reward_features` 输出；
2. 但 `_dominant_state_ids` 不再要求“最大值 `>= 0.5`”才给状态；
3. 对真正满足判据的主状态，按闭环状态机口径直接收束为唯一高层状态；
4. 对未满足严格 `rear/prone/upright/prepared` 判据的 grounded 帧，按 `pitch` 符号回填到 `rear/prone`，保证主状态闭环；
5. 对未命中 `falling/rising` 的 airborne 帧，回填到 `falling_air`，避免出现空中无标签；
6. 同时不再把“任意部位接地”和“足部形成可控支撑”混为一谈，而是额外导出：
   - `foot_grounded`
   - `body_collision`
   - `airborne_true`
   - `state_confident`

这样处理的目的，是既保持高层状态闭环，便于轨迹统计与 milestone 计算，又尽量让严格状态判据继续由 `upright/prepared/airborne_true` 主导。

### 7.1 关键中间量

当前状态判定与奖励质量门主要依赖以下量：

1. `any_contact_z = z_min`
2. `support_contact = max(touch_toe, touch_heel)`
3. `full_foot_support = min(touch_toe, touch_heel)`
4. `foot_contact_z = min(toe_z, heel_z)`
5. `body_contact_z = min(torso_lower_z, nose_lower_z, pelvis_lower_z, thigh_lower_z, calf_lower_z)`
6. `foot_grounded = 1(foot_contact_z <= 0.03 or support_contact >= 0.30)`
7. `body_collision = 1(body_contact_z <= 0.00)`
8. `airborne_true = 1(z_min > 0.03) * (1 - foot_grounded) * (1 - body_collision)`
9. `upright_pitch = Theta ∈ (-0.45, 0.45)`
10. `nose_minus_foot_x = nose_x - foot_x`
11. `nose_minus_support_x = nose_x - p_contact_x`
12. `hip_minus_support_x = hip_x - p_contact_x`
13. `rear_geometry = grounded_any * 1(nose_minus_foot_x < 0)`
14. `prone_geometry = grounded_any * 1(nose_minus_foot_x >= 0)`
15. `upright_height_gate = H >= 0.32`
16. `knee_window = soft_interval(q_knee, 1.45, 1.95)`
17. `stable_vz_gate = 1(|Vz| <= 0.40)`
18. `stable_omega_gate = 1(|Omega| <= 3.5)`

同时：

1. `grounded_any` 当且仅当 `z_min <= 0.03`
2. `airborne` 当且仅当 `z_min > 0.03`
3. `support_contact_gate` 当且仅当 `support_contact >= 0.30`
4. `full_foot_support_gate` 当且仅当 `full_foot_support >= 0.30`

### 7.2 六状态定义

当前实现中的状态定义为：

```text
grounded_prepared
= foot_grounded
* (1 - body_collision)
* full_foot_support_gate
* upright_pitch
* upright_height_gate
* knee_window
* stable_vz_gate
* stable_omega_gate
```

```text
grounded_upright
= foot_grounded
* (1 - body_collision)
* (1 - grounded_prepared)
* full_foot_support_gate
* upright_pitch
* upright_height_gate
* stable_omega_gate
```

```text
grounded_rear
= grounded_any
* (1 - grounded_prepared)
* (1 - grounded_upright)
* rear_geometry
```

```text
grounded_prone
= grounded_any
* (1 - grounded_prepared)
* (1 - grounded_upright)
* (1 - grounded_rear)
* prone_geometry
```

```text
falling_air = airborne_true * 1(vz < -0.05)
```

```text
rising_air = airborne_true * 1(vz > 0.05)
```

因此：

1. `prepared` 明确要求整足支撑、姿态基本竖直、躯干高度不过低、膝角进入窗口、垂直与角速度不能过大、且不存在身体碰撞；
2. `upright` 是 `prepared` 之外的站立接地态，当前实现只要求整足支撑、躯干姿态基本竖直、躯干高度不过低、且不存在身体碰撞；不再额外要求角速度门；
3. `rear/prone` 仍允许发生身体碰撞，因此用于描述失败着地或倒地恢复阶段；但二者不再按主躯干 `pitch` 区分，而改按整体前后几何区分；
4. `air` 不再仅由 `z_min` 定义，而是额外要求既无足部支撑，也无身体碰撞；
5. grounded fallback 不再按 `pitch` 符号回填，而改按 `nose_minus_foot_x` 的符号回填；
6. 六状态标签仍保持闭环六选一；但若某一步只满足回退标签而不满足上述质量条件，则 `state_confident = 0`，该步不触发正向 shaping。

---

## 8. 环境真实奖励的显式近似

当前 `predict_reward` 对 Hopper 官方奖励的实现为：

```math
true\_reward = action\_repeat \cdot standing(H') \cdot hopping(Vx')
```

其中：

1. `standing(H') = 1(H' \in [0.6, 2.0])`
2. `hopping(Vx')` 由 `_lower_tolerance` 给出，要求前向速度不低于 `hop_speed=2.0`，并允许 margin 衰减
3. 最终乘上 `ENV_ACTION_REPEAT = 2`

这就是当前显式环境奖励近似，不包含结构化塑形。

另外，`predict_viable_reward` 还定义了一个更保守的可行性奖励近似：

1. 在 `true_reward` 基础上再乘高度、姿态、接地可行性门；
2. 当前主要用于诊断，不是主训练目标。

---

## 9. 奖励塑形：dense pose 主导，里程碑迁移辅助

### 9.1 奖励形式

当前结构化奖励不再只依赖六状态硬迁移，而改为：

```math
shaping
=
dense\_reward
+ milestone\_scale \cdot milestone
```

即：

1. 主信号改为三相位稠密姿态势函数；
2. 六状态硬迁移奖励保留，但降级为 milestone；
3. 最终 shaping 仍统一乘全局 `shaping_scale`；
4. 当前默认系数为：
   - `dense_pose_delta_weight = 0.40`
   - `dense_pose_abs_weight = 0.00`
   - `dense_pose_milestone_scale = 0.25`
   - `dense_pose_gamma = 0.995`
   - `recover_penalty_scale = 0.05`

### 9.2 三相位 dense pose

当前实现中，dense pose 使用三个软相位 gate：

1. `dense_down_gate`
2. `dense_ground_gate`
3. `dense_rise_gate`

其中：

1. `down` 由 `airborne_true` 与向下速度软门给出；
2. `ground` 仅在“足部已形成地面支撑且不存在身体碰撞”时开启；
3. `rise` 由 `airborne_true` 与向上速度软门给出；
4. 三者会做归一化，因此任一步的相位权重和约为 `1`。

这样处理的目的，是阻断 prone / 贴地失败姿态通过 `ground` 相位长期占有 dense reward 的通路。

### 9.3 三个姿态分数

当前实现定义了三个连续姿态分数：

1. `landing_pose_score`
2. `ground_pose_score`
3. `rise_pose_score`

它们都采用“加权归一化残差 -> robust quadratic -> `exp(-distance)`”的形式，输出位于 `(0, 1]`。

当前第一版只使用一组高可信特征：

1. `pitch`
2. `pitch_rate`
3. `qpos_knee`
4. `vx`
5. `vz`
6. `height`
7. `full_foot_support`
8. `body_collision`

其中：

1. `landing_pose_score` 主要刻画“空中下降时姿态回正、角速度收敛、膝部进入着陆准备窗口、速度方向合理”；
2. `ground_pose_score` 主要刻画“接地后姿态回正、整足支撑形成、身体不再碰地、膝部进入起跳准备窗口”；
3. `rise_pose_score` 主要刻画“起跳后向上、姿态稳定，且膝部不再保持压缩”；前向性由相位外的 `forward_progress` 单独控制，不再在 `rise_pose_score` 内重复编码。

### 9.4 稠密势函数

当前实现中的 dense pose potential 为：

```math
\Phi(s)=
dense\_down\_gate \cdot (landing\_pose\_score \cdot (1-body\_collision) \cdot stable\_omega\_gate)
+ dense\_ground\_gate \cdot (ground\_pose\_score \cdot full\_foot\_support\_gate \cdot (1-body\_collision))
+ dense\_rise\_gate \cdot (rise\_pose\_score \cdot forward\_progress)
```

并据此定义：

```math
dense\_reward =
\alpha_{\Delta}(\gamma \Phi(s_{t+1})-\Phi(s_t))
+ \alpha_{abs}\Phi(s_{t+1})
```

其中：

1. 差分项负责奖励“朝当前相位目标姿态靠近”，并使用 `\gamma \Phi_{t+1}-\Phi_t` 保持 potential-based 语义；
2. 当前主线默认已将小绝对项降为 `0`，避免失败姿态形成占有奖励；
3. `landing/ground/rise` 三个相位都显式带质量门，避免仅凭相位 gate 获得 dense 收益；
4. `rise` 相位仍与 `forward_progress` 绑定，避免原地摆出漂亮空中姿态；
5. 额外加入小尺度 `recover_penalty`，在“身体碰地且未真正离地”的失败姿态下提供弱负向塑形，防止 prone / rear 静止停留形成 0 奖励平台。

### 9.5 六个 milestone 迁移项

当前实现仍保留六项硬迁移奖励：

1. `grounded_rear -> grounded_upright`，权重 `0.28`，乘 `upright_quality_gate`
2. `grounded_prone -> grounded_upright`，权重 `0.28`，乘 `upright_quality_gate`
3. `grounded_upright -> grounded_prepared`，权重 `0.40`，乘 `prepared_quality_gate`
4. `grounded_prepared -> rising_air`，权重 `0.45`，乘 `rising_quality_gate * forward_progress`
5. `falling_air -> grounded_upright`，权重 `0.50`，乘 `upright_quality_gate`
6. `falling_air -> grounded_prepared`，权重 `0.75`，乘 `landing_quality_gate`

对应函数为 [predict_shaping_reward](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py)。

这些项的作用已从“主奖励”降级为“里程碑确认”。

### 9.6 当前不再使用的项

当前实现中，以下量虽然仍保留为特征导出，但不直接作为独立塑形项进入奖励：

1. `forward_progress`
2. `backward_speed`
3. `rotation_penalty`
4. `low_body`
5. `vertical_settle`

它们目前主要用于导出、分析和诊断。

---

## 10. 奖励塑形如何接入训练与规划

### 10.1 训练阶段：混合监督

训练时，奖励头 `_reward` 的监督目标不是始终等于环境奖励。

在 `reward_mode != learned` 时，`SDWorldModel.mixed_reward_target` 使用：

```math
reward\_target = env\_reward + \alpha \cdot shaping\_delta
```

其中：

1. `shaping_delta` 来自“dense pose 稠密势函数 + milestone”；
2. `alpha` 由 `reward_mode` 决定；
3. 当前实验脚本使用 `reward_mode=curriculum`。

### 10.2 当前实验中的 curriculum 口径

当前 `10k`/`50k` 相关脚本使用：

1. `sd_reward_mode=curriculum`
2. `sd_reward_curriculum_alpha=0.5`
3. `sd_reward_curriculum_alpha_min=0.25`
4. `sd_reward_curriculum_start=10000`
5. `sd_reward_curriculum_end=30000`

因此当前这条线的奖励头监督目标为：

1. `10000` 步之前，`alpha = 0.5`
2. `10000` 到 `30000` 间线性衰减
3. `30000` 之后，`alpha = 0.25`

### 10.3 规划与评估阶段

当前 `reward()` 的行为是：

1. 若 `reward_mode=learned`，直接使用 learned reward logits；
2. 若处于训练态，即使是 `curriculum/hybrid`，也仍返回 learned reward logits；
3. 若 `reward_mode=explicit` 且处于评估/规划态，才直接用 `predict_plan_reward = true_reward + shaping_reward`；
4. 对于 `curriculum/hybrid`，规划与评估不会再次把塑形直接加到 reward logits 上，避免双重计入。

因此，当前主线的真实口径是：

1. 训练时通过混合监督把结构化塑形注入 reward head；
2. 规划时仍主要使用 learned reward head；
3. 并不是在规划时再次显式把 shaping 直接叠加进去。

---

## 11. 动作裁剪：条件动作先验盒

### 11.1 发生位置

当前动作裁剪不是训练时对策略网络输出做硬限制，而是规划器中的条件动作先验：

1. `SDWorldModel.action_prior_bounds` 从显式结构态读取动作盒；
2. `TDMPC2._plan` 在采样、策略轨迹生成和最终动作选择时，把动作裁到该动作盒内。

### 11.2 默认行为

当前默认配置 `sd_action_prior_enabled=false`。因此默认行为是：

1. `low = [-1, -1, -1, -1]`
2. `high = [1, 1, 1, 1]`

即不施加额外动作约束。

同时仍保留以下统计字段，便于后续做 ablation：

1. `rear_state_frac`
2. `prone_state_frac`
3. `rear_prior_frac`
4. `prone_prior_frac`

其中在默认禁用配置下，前两者反映状态占比，后两者固定为 `0`。

### 11.3 触发条件

只有在显式开启 `sd_action_prior_enabled=true` 后，且强制六选一的 dominant state 命中以下状态时，才收缩动作盒：

1. `grounded_rear`
2. `grounded_prone`

这两个状态分别对应后仰着地和前伏着地恢复阶段。

### 11.4 rear 状态下的动作盒

以下内容仅在 `sd_action_prior_enabled=true` 时生效。

当 `rear_mask` 触发时，动作盒中心由以下量调节：

1. `rear_progress`
2. `support_missing`

然后通过 `_prior_box(center, width)` 形成每一维的 `low/high`。

当前 rear 四维中心大致为：

1. 第 1 维偏负
2. 第 2 维从偏正向更负移动
3. 第 3 维随 `rear_progress` 上升
4. 第 4 维偏负

宽度固定在约 `0.18 ~ 0.34` 区间。

### 11.5 prone 状态下的动作盒

以下内容仅在 `sd_action_prior_enabled=true` 时生效。

当 `prone_mask` 触发时，动作盒中心由以下量调节：

1. `prone_severity`
2. `support_missing`
3. `knee_excess`

当前 prone 四维中心大致为：

1. 第 1 维强正
2. 第 2 维强负
3. 第 3 维接近中性、随膝过弯略向负移
4. 第 4 维强正

宽度固定在约 `0.18 ~ 0.42` 区间。

### 11.6 裁剪发生的具体节点

在 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py) 中，条件动作盒会作用于三处：

1. `num_pi_trajs` 的策略轨迹动作；
2. `MPPI` 采样得到的 `actions_sample`；
3. 最终选出的环境执行动作 `a`。

最后一步仍会再经过一次全局 `[-1, 1]` clamp，因此最终环境动作始终合法。

### 11.7 作用边界

这套动作先验在当前主线中的性质应表述为：

1. 接口仍保留；
2. 默认禁用，不再作为主训练通道；
3. 若重新启用，它也不是手写控制器，只是在 `rear/prone` 两类明显失衡状态下缩小规划器动作搜索盒；
4. 盒内具体取值仍由 `TD-MPC2` 规划和价值评估决定。

---

## 12. 当前实现与早期 v5 讨论的主要差异

与早期“无 residual 的极简灰箱第一版”相比，当前真实实现已经发生了三点重要变化：

1. 不是只保留 6 维主状态，而是显式递推 17 维结构态；
2. 不是 residual 默认关闭，而是当前主线默认使用 `17 + 495` 的混合 latent；
3. 奖励塑形不只停留在文档层，而是已经通过混合监督进入 reward head 训练；动作先验接口虽仍保留，但当前默认禁用。

因此，当前版本更准确的概括应是：

> 一个以 17 维显式灰箱 Hopper 结构态为前端、以高维残差 latent 为后端、并带有六状态硬迁移塑形与条件动作先验的混合结构化世界模型。

---

## 13. 当前实现的一句话总结

> 当前 `v5` 的核心不是“完全靠手工结构替代学习”，而是“用 17 维可解释灰箱显式态稳定约束 Hopper 的接触、姿态和腿部力学，再把高维残差、奖励头学习与规划器搜索建立在这层结构之上”。
