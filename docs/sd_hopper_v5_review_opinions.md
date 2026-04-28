# Hopper Hop 可学习系统动力学 World Model v5 实现评审意见

**评审对象**：`learnable_sd_hopper_design_v5_minimal.md` 中描述的 `world_model_kind=sd_hopper` 实现  
**适用场景**：将可学习系统动力学（learnable SD）结构用于 Hopper Hop 的有模型强化学习（Model-Based RL / TD-MPC2）  
**评审重点**：结构正确性、物理一致性、可学习机制层、规划器是否可能利用非物理漏洞、奖励塑形是否改变任务目标

---

## 1. 总体判断

当前 v5 实现**没有明显的方向性错误**，可以作为 Hopper Hop 的结构化 world model 继续实验。它已经不再是一个纯粹的 6 维系统动力学模型，而是一个更工程化的混合模型：

```text
17 维显式灰箱结构态
+ 495 维 residual latent
+ 结构残差修正
+ 六状态迁移奖励塑形
+ 条件动作先验盒
+ TD-MPC2 learned reward / value / planning
```

因此，建议将它定义为：

> **SD-constrained hybrid latent world model**  
> 即：以系统动力学结构为约束前端，以高维残差潜变量补偿真实 MuJoCo 接触动力学误差的混合结构化世界模型。

这个方向是合理的。对于 Hopper 这类强接触、强非线性的系统，纯手写 SD 模型通常过弱；加入关节状态、接触变量、`z_min`、residual latent 和 TD-MPC2 规划器，是合理的工程推进。

但从有模型强化学习角度看，当前版本最需要防范的不是“SD 结构不够复杂”，而是：

> **planner 可能利用显式 SD 核中的非物理自由度，而不是真正学习 Hopper 的稳定跳跃机制。**

当前最值得优先检查的风险包括：

1. 接触门控可能允许空中施力；
2. 腿部阻尼项符号可能导致非物理能量注入；
3. `z_min` 作为派生几何量，可能被 residual 独立改写，破坏几何一致性；
4. 奖励塑形不是 potential-based，且后期 alpha 没有衰减到 0，可能改变最终优化目标；
5. 条件动作先验盒是强干预，可能在状态误判时造成动作锁死。

---

## 2. 当前实现的准确定位

### 2.1 不是纯 SD 模型

当前模型不是：

```text
少数几个手写 stock-flow 方程
+ 少量可学习物理参数
```

而是：

```text
显式 SD 结构态
+ residual latent
+ residual dynamics
+ learned reward head
+ learned value head
+ MPPI / TD-MPC2 planning
+ 手写动作先验
+ 手写状态迁移塑形
```

所以它更像一个**结构化世界模型框架**，而不是传统意义上的纯系统动力学仿真器。

### 2.2 显式结构层仍然有价值

尽管 residual latent 很大，17 维显式结构态仍然有重要作用：

- 将高度、速度、姿态、关节、接触等关键机制显式暴露出来；
- 为奖励塑形、动作先验、接触判断提供可解释中间变量；
- 给 residual model 一个物理上更合理的预测基准；
- 允许对模型内部机制做诊断，而不仅仅看 return。

### 2.3 但需要证明 SD 核的实际贡献

由于 residual latent 是 495 维，最终性能提升可能来自三类来源：

1. SD 显式结构本身；
2. 高维 residual latent；
3. 奖励塑形和动作先验。

因此，如果实验目标是证明 learnable SD 有贡献，需要做 ablation，而不能只报告 full model 的最终 return。

---

## 3. 关键问题一：接触门控可能允许空中施力

### 3.1 当前实现

当前接触门控大致为：

```math
support = \max(touch_{toe}, touch_{heel})
```

```math
G = \sigma(k_c(support - \tau_c))
```

腿部推力为：

```math
F_{leg} = G \cdot ReLU(k_{leg} c_{leg} - b_{leg} \dot c_{leg} + F_{max} A_t)
```

其中 `touch_toe` 和 `touch_heel` 不是瞬时几何接触，而是带有平滑记忆的递推变量：

```math
touch' = \rho touch + (1 - \rho) target
```

并且当前有：

```text
CONTACT_CARRY = 0.45
```

### 3.2 风险

如果 touch 在离地后仍然保留一段时间，那么 `G` 也可能继续为正。此时模型可能预测出：

```text
几何上已经离地，但腿部仍然可以对地面施力。
```

这在普通 one-step prediction 中可能只是小误差，但在 Model-Based RL 中非常危险。Planner 会主动搜索 world model 的漏洞。如果模型允许残余空中接触力，planner 可能学出真实 MuJoCo 中不存在的策略，例如：

```text
在空中通过残余 touch 门控继续“蹬地”加速或修正姿态。
```

这会造成典型的 model exploit。

### 3.3 建议修改

接触门控不要只依赖 touch，至少应乘以几何近地门：

```math
G = G_{touch} \cdot G_{geom}
```

其中：

```math
G_{touch} = \sigma(k_t(support - \tau_t))
```

```math
G_{geom} = \sigma(k_z(z_{margin} - z_{min}))
```

更细粒度的版本是分别对 toe 和 heel 建模：

```math
G_{toe} = \sigma(k_z(z_{margin} - z_{toe})) \cdot \sigma(k_t(touch_{toe} - \tau_t))
```

```math
G_{heel} = \sigma(k_z(z_{margin} - z_{heel})) \cdot \sigma(k_t(touch_{heel} - \tau_t))
```

```math
G = \max(G_{toe}, G_{heel})
```

最低限度应满足：

```text
当 z_min > contact_margin 时，G 应接近 0。
```

### 3.4 建议增加一致性损失

可以加入接触一致性损失：

```math
\mathcal{L}_{contact} = \|touch \cdot \mathbf{1}(z_{min} > margin)\|^2
```

含义是：

```text
几何上已经明显离地时，touch 不应该仍然保持高值。
```

### 3.5 必做诊断

在 model rollout 中记录：

```text
z_min
touch_toe
touch_heel
G
F_leg
```

检查是否出现：

```text
z_min > 0.05 但 G 高
z_min > 0.05 但 F_leg 高
```

如果出现，这就是优先级最高的 bug 或设计漏洞。

---

## 4. 关键问题二：腿部阻尼项符号可能反了

### 4.1 当前实现

当前文档中定义：

```math
c_{leg} = rest\_leg\_length - leg\_length
```

因此：

- `c_leg > 0` 表示腿部被压缩；
- `d_c_leg > 0` 表示压缩正在增加，即落地压缩阶段；
- `d_c_leg < 0` 表示压缩正在减少，即伸展回弹阶段。

当前腿推力写作：

```math
F_{leg} = G \cdot ReLU(kc - b\dot c + F_{max}A)
```

### 4.2 风险

如果 `d_c_leg` 的方向确实是：

```math
\dot c = \frac{c_{next} - c}{\Delta t}
```

那么 `-b dot c` 会导致：

- 压缩阶段 `dot c > 0`：阻尼项减小支撑力；
- 伸展阶段 `dot c < 0`：阻尼项增加回弹力。

这更像是**负阻尼**，可能在伸展阶段向系统注入能量。

被动阻尼应该抵抗相对运动，而不是放大运动。对于压缩量定义为 `c = L0 - L` 的 Kelvin-Voigt 型模型，通常更自然的形式是：

```math
F_{leg} = kc + b\dot c
```

这样：

- 压缩时 `dot c > 0`，阻尼增加支撑力，抵抗继续压缩；
- 伸展时 `dot c < 0`，阻尼减少回弹力，抵抗继续伸展；
- 阻尼整体做负功，即耗散能量。

### 4.3 建议修改

先确认代码中 `d_c_leg` 的实际符号。如果确实表示压缩速度，则建议改为：

```math
F_{leg} = G \cdot ReLU(kc + b\dot c + F_{max}A)
```

如果想保留 `-b` 的形式，则应把变量定义为腿长变化率：

```math
\dot L = -\dot c
```

然后写成：

```math
F_{leg} = G \cdot ReLU(kc - b\dot L + F_{max}A)
```

核心不是符号形式，而是必须保证被动阻尼满足：

```math
P_{damping} \le 0
```

也就是阻尼不能凭空给系统加能量。

### 4.4 必做诊断

记录被动阻尼功率，例如：

```math
P_{damping} = F_{damping} \cdot v_{relative}
```

检查其是否系统性为正。如果阻尼项在大量样本中表现为正功，则说明它正在注入能量，而不是耗散能量。

---

## 5. 关键问题三：`z_min` 不应作为可被 residual 长期独立改写的状态

### 5.1 当前实现

`z_min` 被放在 17 维结构态的第 16 维，并参与：

- grounded / airborne 判断；
- 六状态分类；
- shaping reward；
- action prior bounds；
- contact update；
- `grounded_any` / `airborne` 判定。

文档中指出，`z_min` 是由当前姿态几何计算得到的派生量。但在外层 `SDWorldModel` 中，结构态会被 `_struct_residual` 修正。虽然 `z_min` 属于 trend-protected 维度，但它仍然是 latent slot 中的一个独立数值。

### 5.2 风险

`z_min` 本应由以下变量唯一决定：

```text
H, Theta, waist, hip, knee, ankle
```

如果 residual 可以独立改写 `z_min`，模型内部可能出现几何不一致：

```text
H 和关节角表示身体仍在空中，
但 z_min 被 residual 修成接地。
```

或反过来：

```text
几何上已经穿地或接地，
但 z_min 被 residual 修成未接地。
```

这会直接影响状态分类、奖励塑形和动作裁剪，因此不是单纯的预测误差问题，而是结构一致性问题。

### 5.3 建议修改

最干净的做法是：

```text
不要把 z_min 当作可被 residual 独立修正的自由状态。
每次 transition 后，根据 H, Theta, q 重新计算 z_min。
```

伪代码：

```python
next_struct = explicit_step(struct, action)
next_struct = residual_correct_allowed_dims(next_struct)
next_struct.z_min = recompute_z_min(
    next_struct.H,
    next_struct.Theta,
    next_struct.waist,
    next_struct.hip,
    next_struct.knee,
    next_struct.ankle,
)
```

如果确实希望 residual 修正几何误差，建议让 residual 修正几何参数或 body offset，而不是直接修正 `z_min`。

### 5.4 最低限度约束

如果保留 `z_min` 作为 latent slot，至少加入一致性损失：

```math
\mathcal{L}_{zmin} = \|z_{min}^{slot} - z_{min}^{kinematics}(H, \Theta, q)\|^2
```

### 5.5 必做诊断

每一步比较：

```text
z_min_slot
recomputed_z_min(H, Theta, q)
```

如果二者长期偏离，则说明 residual 已经破坏几何一致性。

---

## 6. 关键问题四：奖励塑形可能改变最终优化目标

### 6.1 当前实现

训练阶段 reward head 的监督目标为：

```math
reward\_target = env\_reward + \alpha \cdot shaping\_delta
```

当前 curriculum 中：

```text
alpha: 0.5 -> 0.25
```

也就是说，即使训练后期，reward head 仍然学习的是带塑形的奖励，而不是纯环境奖励。

### 6.2 风险

六状态迁移塑形鼓励：

```text
grounded_rear -> grounded_upright
grounded_prone -> grounded_upright
grounded_upright -> grounded_prepared
grounded_prepared -> rising_air
falling_air -> grounded_upright
falling_air -> grounded_prepared
```

这些迁移有助于恢复、准备、起跳和落地，但它们不等价于官方 Hopper Hop 奖励。官方任务更核心的是：

```text
站立高度足够 + 前向速度足够
```

如果 alpha 后期不衰减到 0，planner 最终优化的 learned reward head 可能永久带有塑形偏差。潜在问题包括：

- 反复制造 prepared / rising / landing transition，而不是稳定高速前进；
- 高冲击落地后只要状态被判为 prepared 就获得塑形收益；
- 最终策略偏向恢复循环，而不是最优 hopping gait。

### 6.3 建议方案 A：拆成两个 reward head

建议训练两个 reward head：

```text
env_reward_head      学真实环境奖励
shaping_reward_head  学结构化塑形奖励
```

规划时使用：

```text
early training: env_reward + alpha * shaping_reward
late training:  env_reward only
```

这样既保留早期结构引导，又不会污染最终任务目标。

### 6.4 建议方案 B：alpha 最终衰减到 0

把：

```text
alpha_min = 0.25
```

改为：

```text
alpha_min = 0.0
```

或者只在低可行状态、恢复状态中启用 shaping，而不是全局保留。

### 6.5 建议方案 C：改成 potential-based shaping

定义状态势函数：

```math
\Phi(s)
```

然后塑形奖励写成：

```math
F(s,a,s') = \gamma \Phi(s') - \Phi(s)
```

potential-based shaping 更不容易改变最优策略。

### 6.6 落地迁移建议加入冲击约束

当前：

```text
falling_air -> grounded_prepared
falling_air -> grounded_upright
```

建议加入落地质量条件：

```text
|Vz'| 不应过大
|Omega'| 不应过大
腿部压缩不应过度
落地后不应立即失稳
```

或者增加冲击惩罚：

```math
r_{impact} = -\lambda \cdot \mathbf{1}_{landing} \cdot \max(0, -Vz)^2
```

---

## 7. 关键问题五：条件动作先验盒可能造成恢复锁死

### 7.1 当前实现

当状态被判为：

```text
grounded_rear
grounded_prone
```

时，模型会给出状态条件动作盒，TD-MPC2 会在三处裁剪动作：

1. 策略轨迹动作；
2. MPPI 采样动作；
3. 最终环境执行动作。

### 7.2 优点

这个设计有明确价值：

- rear/prone 是明显失衡状态；
- 缩小动作搜索空间可以提升早期恢复成功率；
- 它不是直接给动作，而是约束 planner 搜索范围。

### 7.3 风险

但它仍然是强结构干预。风险包括：

#### 风险 1：状态误判导致错误裁剪

如果模型把某状态误判为 `grounded_prone`，planner 的动作空间会被强行缩窄。此时更好的恢复动作可能被裁掉。

#### 风险 2：限制非预设恢复策略

Hopper 有时可能需要非直觉动作恢复。硬动作盒可能过早限制探索，使策略只学到手写先验允许的恢复方式。

#### 风险 3：最终性能上限被压低

硬 prior 可能提升早期 sample efficiency，但降低最终 asymptotic return。

### 7.4 建议修改：soft prior

将硬裁剪改成 soft penalty：

```math
r_{prior} = -\lambda \|a - a_{center}(s)\|^2
```

或将其加入 planner 的 action proposal prior，而不是最终动作硬裁剪。

### 7.5 建议修改：混合采样

在 MPPI 中采用混合采样：

```text
80% samples from prior box
20% samples from full [-1, 1]
```

这样保留先验引导，同时不完全关闭探索。

### 7.6 建议修改：confidence gating

只有当状态分类置信度足够高时才触发动作盒：

```text
grounded_prone_score > 0.8
```

而不是 dominant state 只要超过 `0.5` 就硬裁剪。

---

## 8. 中等风险问题

### 8.1 时间步长必须和 replay transition 对齐

当前显式核使用：

```text
dt = 0.02
action_repeat = 2
Delta t = 0.04
```

这只有在 replay buffer 中一条 transition 确实对应两个 dm_control control steps 时才是正确的。

建议从数据中直接估算：

```math
\Delta H_{data} \approx Vz_t \Delta t
```

分别测试：

```text
Delta t = 0.02
Delta t = 0.04
```

哪个更贴近真实 transition。

### 8.2 `A_enc` 与 rollout 中的 `A_t` 语义不完全一致

编码阶段：

```math
A_{enc} = support \cdot knee\_gate \cdot motion\_gate
```

rollout 阶段：

```math
A_{t+1} = A_t + \Delta t \frac{A^{des}_t - A_t}{\tau_A}
```

前者是从当前姿态/运动推断出的激活倾向，后者是由历史 action 积分出来的 actuator-like latent state。两者语义不完全一致。

建议：

- 明确将 `A` 定义为 latent mechanism state；
- 使用 `obs_t, previous_action, obs_{t-1}` 推断初始 `A`；
- 做 encoded A 与 rollout A 的 multi-step consistency test。

### 8.3 高维 residual 可能掩盖 SD 结构

当前 latent 为：

```text
17 structural dims + 495 residual dims
```

这并非错误，但它意味着 full model 的成功不一定来自 SD 核。

建议做 ablation：

```text
A. pure TD-MPC2 latent model
B. 17-dim SD only
C. 17-dim SD + struct residual only
D. 17-dim SD + residual latent only
E. full v5
```

### 8.4 `predict_reward` 的硬阈值可能不平滑

当前显式 reward 近似中：

```text
standing(H') = 1(H' in [0.6, 2.0])
```

如果它只用于诊断，问题不大；如果用于 reward supervision 或 planning，硬阈值可能带来不连续目标。

建议改为平滑形式：

```math
standing(H) = \sigma(k(H - 0.6))
```

如果需要上界：

```math
standing(H) = \sigma(k(H - 0.6)) \cdot \sigma(k(2.0 - H))
```

---

## 9. 当前实现中合理的部分

### 9.1 17 维结构态比 6 维更合理

Hopper 的接触、腿长、姿态恢复和动作先验都依赖关节状态。只保留：

```text
H, Vx, Vz, Theta, Omega, A
```

会太抽象。保留 waist/hip/knee/ankle 及其速度是合理的。

### 9.2 residual latent 的存在不是问题

MuJoCo Hopper 的接触动力学很难被手写 SD 完整复刻。Residual latent 是合理补偿。真正要防范的是：

```text
residual 破坏结构一致性。
```

而不是 residual 本身。

### 9.3 训练时注入 shaping、规划时不重复叠加 shaping 是合理的

当前设计避免了 double counting：

```text
训练 reward head 时注入 shaping；
规划时不再显式额外加一次 shaping。
```

这个设计比“训练加一次、规划再加一次”安全。

问题只在于：alpha 后期是否应该保留。

### 9.4 动作先验不是手写控制器

条件动作盒没有直接决定动作，而是缩小搜索空间。盒内动作仍由 TD-MPC2 规划器和价值评估决定。

但由于它作用于最终环境执行动作，仍建议做 softening 或 confidence gating。

---

## 10. 建议优先级

如果只能先改 5 处，建议按以下顺序：

### P0-1：接触门控乘几何近地门

目的：防止空中施力和 model exploit。

```text
G = G_touch * G_geom
```

### P0-2：检查并修正腿部阻尼符号

目的：保证阻尼耗能，而不是注入能量。

```text
若 c = L0 - L，则阻尼项通常应与 dot c 同号进入支撑力。
```

### P0-3：每步重新计算 `z_min`

目的：保证几何一致性。

```text
z_min = forward_kinematics_min_z(H, Theta, q)
```

不要让 residual 长期独立改写 `z_min`。

### P1-1：拆分 env reward head 与 shaping reward head

目的：避免最终任务目标被永久塑形污染。

```text
env_reward_head
shaping_reward_head
```

### P1-2：把 hard action box 改成 soft/confidence-gated prior

目的：降低状态误判时的动作锁死风险。

```text
confidence > threshold 时才强约束；
否则使用 soft penalty 或混合采样。
```

---

## 11. 建议验证实验清单

### 实验 1：空中接触力检查

记录：

```text
z_min
touch_toe
touch_heel
G
F_leg
```

检查：

```text
z_min > 0.05 但 G 高
z_min > 0.05 但 F_leg 高
```

### 实验 2：阻尼功率检查

记录被动阻尼功率，确认：

```math
P_{damping} \le 0
```

如果出现系统性正功，说明阻尼项可能正在注入能量。

### 实验 3：`z_min` 几何一致性检查

比较：

```text
z_min_slot
recomputed_z_min(H, Theta, q)
```

如果偏差明显，说明 residual 破坏了结构一致性。

### 实验 4：reward head 分解误差

分别评估：

```text
env reward prediction error
shaping reward prediction error
mixed reward prediction error
```

不要只看 mixed reward。

### 实验 5：action prior ablation

比较：

```text
no action prior
hard action prior
soft action prior
confidence-gated action prior
```

关注 early return 和 final return 是否存在 trade-off。

### 实验 6：residual dominance ablation

比较：

```text
17 SD only
17 SD + struct residual only
17 SD + residual latent only
full model
pure TD-MPC2 baseline
```

记录：

```text
structural prediction error
contact prediction error
reward prediction error
planning return
real env return
```

### 实验 7：时间步长一致性测试

用 replay buffer 检查：

```math
H_{t+1} - H_t \approx Vz_t \Delta t
```

比较：

```text
Delta t = 0.02
Delta t = 0.04
```

### 实验 8：encoded A 与 rollout A 一致性测试

比较多步 rollout 中：

```text
encoded A_{t+k}
rolled-out A_{t+k}
```

如果分布明显不一致，需要重新定义或重新推断 `A`。

---

## 12. 建议的最小修补版本

一个不大改框架、但能显著降低风险的最小修补版本如下：

```text
1. transition 后重新由几何计算 z_min；
2. contact gate = touch gate * geometry gate；
3. 修正或验证 leg damping sign；
4. 增加 contact consistency loss 与 z_min consistency loss；
5. reward head 拆成 env / shaping 两个输出；
6. hard action prior 改为 confidence-gated hard prior，或混合采样 prior。
```

这些修改不会推翻当前 v5 的主体结构，但能显著减少 planner 利用非物理通道的风险。

---

## 13. 最终结论

当前 v5 实现可以继续作为 Hopper Hop 的可学习 SD world model 进行实验。它的核心优势是把 Hopper 的高度、速度、姿态、关节、触地和恢复状态显式结构化，并通过 residual latent 补偿真实接触动力学的复杂性。

但在正式作为主模型或论文实验结论前，建议优先处理三类本质风险：

```text
1. contact / z_min / geometry 一致性；
2. force / damping / energy 物理一致性；
3. reward shaping / action prior 对最终策略目标的干预。
```

最重要的一句话是：

> **这个 v5 不是本质上错误；它的问题在于 planner 可能利用非物理自由度。只要先修 contact gate、damping sign 和 z_min consistency，并通过 ablation 证明 SD 核的贡献，它就是一个值得继续推进的结构化有模型强化学习方案。**
