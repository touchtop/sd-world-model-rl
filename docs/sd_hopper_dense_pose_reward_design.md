# Hopper Hop 结构化奖励改造设计文档：从硬状态迁移到相位条件密集姿态奖励

## 0. 文档目的

本文档整理当前 Hopper Hop 可学习 SD / world model 实验中的奖励引导问题，并给出一套新的结构化奖励设计建议。

核心观点是：

> 当前基于 `prev_state -> next_state` 的硬迁移奖励，对 Hopper 这种多自由度连续控制系统过于稀疏、离散且容易被阈值边界支配。
> 更合理的方向是：在不同运动相位中定义可解释的“目标姿态流形”，并用当前姿态到目标姿态的连续距离构造密集奖励。

本文重点讨论三类目标姿态：

1. **下降相位 landing-prep 姿态**：空中下降时的标准着地准备姿态；
2. **地面相位 ground-prepared 姿态**：着地后准备起跳的标准压缩/稳定姿态；
3. **上升相位 rise-extension 姿态**：起跳后空中上升时的舒展与稳定姿态。

---

## 1. 当前奖励失效的可能原因

### 1.1 当前奖励是硬状态迁移奖励

当前状态判别有 6 个宏观状态：

```text
falling_air
grounded_rear
grounded_prone
grounded_upright
grounded_prepared
rising_air
```

当前结构化塑形奖励只在以下离散迁移发生时给奖励：

```text
grounded_rear     -> grounded_upright
grounded_prone    -> grounded_upright
grounded_upright  -> grounded_prepared
grounded_prepared -> rising_air
falling_air       -> grounded_upright
falling_air       -> grounded_prepared
```

这意味着，如果一个动作没有让系统跨过硬状态边界，那么该动作在结构化奖励上就是 0。

### 1.2 多自由度连续控制中的关键动作往往发生在状态内部

Hopper 的控制难点不是只在“状态切换的一帧”发生，而是在整个连续过程中发生：

```text
下降时如何摆腿和调整躯干
触地前如何控制角速度
触地时如何吸收冲击
地面相位如何压缩蓄力
起跳时如何产生前向和向上的速度
空中如何保持姿态并准备下一次落地
```

这些动作通常不会立刻触发状态标签变化，但它们决定了后续是否能稳定跳跃。

所以当前硬迁移奖励容易出现：

```text
大多数 transition 没有 shaping signal
planner 不知道“变好一点”是否有价值
只有跨过阈值的一瞬间才有奖励
训练信号高方差、低密度
```

### 1.3 当前状态边界本身也比较脆弱

当前最终状态标签主要由以下变量决定：

```text
z_min
pitch
vz
height
knee
touch_toe
touch_heel
full_foot_support
```

这使得奖励容易受到阈值影响。例如：

```text
z_min 从 0.029 变到 0.031，状态可能从 grounded 跳到 rising_air
vz 从 -0.051 变到 -0.049，状态可能从 falling_air 跳到 rising_air
knee 从 1.44 变到 1.45，状态可能进入 prepared
```

如果奖励只依赖这些硬切换，model-based planner 很可能学习到“刷边界”，而不是真正改善运动机制。

### 1.4 旧奖励没有充分利用连续诊断量

当前实现中保留了不少连续诊断量，例如：

```text
posture_stability
spin_stability
forward_progress
upward_progress
descent_progress
vertical_settle
rotation_penalty
low_body
```

但这些量没有成为主要塑形奖励。它们本来可以提供密集反馈，却没有被充分利用。

---

## 2. 新想法是否正确？

你的新想法是：

> 建立三个理想姿态：
>
> 1. 空中下降的标准着地准备姿态；
> 2. 地面的标准 prepared 姿态；
> 3. 空中上升的标准舒展姿态；
>    然后在不同相位中计算当前姿态与对应标准姿态的距离，从而获得密集奖励信号。

这个方向是正确的。

更准确地说，它应该表述为：

> 从“奖励硬状态迁移”改为“奖励相位内朝目标姿态流形靠近”。

这比单纯的状态迁移奖励更适合 Hopper，因为 Hopper 的控制变量是连续的，姿态变化也是连续的。只要当前姿态比上一帧更接近当前相位的理想姿态，就可以获得反馈。

---

## 3. 不建议简单使用姿态 L2 距离

虽然“三个目标姿态”的方向是对的，但实现时不建议直接做：

```text
reward = - ||current_joint_angles - target_joint_angles||_2
```

原因如下。

### 3.1 角度变量需要 wrap

角度是周期变量。比如：

```text
-3.13 和 3.13
```

在数值上相差约 6.26，但物理上非常接近。

因此角度差必须使用：

```python
wrap_angle(current - target)
```

而不是直接相减。

### 3.2 不同变量尺度不同

下面这些量不能直接等权相加：

```text
theta
omega
hip
knee
ankle
vx
vz
touch_toe
touch_heel
leg_compression
foot_x_relative
```

它们量纲不同、数值范围不同、对任务影响也不同。需要使用尺度归一化和权重。

### 3.3 成功姿态不是一个点，而是一个区域

Hopper 的成功跳跃不对应唯一的关节角组合，而对应一族可行姿态。比如地面 prepared 姿态可以有不同程度的膝盖压缩，只要稳定、有支撑、能起跳即可。

因此更准确的目标不是“目标点”，而是：

```text
目标姿态流形
或者
目标姿态原型 + 容忍范围
```

---

## 4. 推荐的基本数学形式

### 4.1 特征向量

先从状态中提取姿态特征：

\[
\phi(s)
\]

建议包含：

```text
theta
omega
waist
hip
knee
ankle
waist_dot
hip_dot
knee_dot
ankle_dot
H
vx
vz
leg_compression
foot_x_relative_to_hip
foot_z_relative_to_ground
touch_toe
touch_heel
body_collision
```

不要求三个相位都使用全部特征，可以为每个相位设置不同权重。

### 4.2 加权归一化距离

对第 \(i\) 个相位，定义目标原型：

\[
\mu_i
\]

定义容忍尺度：

\[
\sigma_i
\]

定义权重：

\[
w_i
\]

则距离为：

\[
D_i(s)
======

\sum_k
w_{i,k}
\rho
\left(
\frac{\Delta(\phi_k(s), \mu_{i,k})}{\sigma_{i,k}}
\right)
\]

其中：

- \(\Delta\) 对角度变量使用 wrapped difference；
- 对非角度变量使用普通差值；
- \(\rho\) 可以是平方损失、Huber 损失或 clipped quadratic；
- \(\sigma\) 表示该变量允许偏离目标多少；
- \(w\) 表示该变量在当前相位中的重要程度。

### 4.3 距离转成姿态分数

可以使用指数分数：

\[
S_i(s) = \exp(-D_i(s))
\]

也可以使用更平滑的形式：

\[
S_i(s) = 1 - \tanh(D_i(s))
\]

推荐先用：

\[
S_i(s) = \exp(-D_i(s))
\]

因为它简单、单调，并且自然位于 \([0,1]\)。

---

## 5. 三个目标姿态的具体含义

## 5.1 空中下降：landing-prep 姿态

### 目标

下降相位的目标不是“好看”，而是：

```text
准备以可控姿态触地
降低落地冲击后的翻倒概率
保证脚部先接触而不是躯干/腿段刮地
让落脚点相对身体处于合理位置
降低角速度
```

### 建议重点特征

```text
theta                    躯干接近竖直
omega                    角速度较小
knee                     适度弯曲，准备吸收冲击
ankle                    足部姿态适合接地
foot_x_relative_to_hip   落脚点不能太前或太后
foot_z_relative          足部接近地面但不穿地
vz                       明显下降，但不要过大
vx                       保持正向
```

### 不建议过度强调

```text
touch_toe / touch_heel
```

因为下降相位中尚未真正触地，接触信号可能为 0 或存在模型滞后。

### 示例目标语义

```text
theta ≈ 0
omega ≈ 0
knee = moderately bent
ankle = landing-ready
foot_x_rel = slightly forward or neutral
vz < 0 but not extremely negative
vx > 0
```

---

## 5.2 地面：ground-prepared 姿态

### 目标

地面 prepared 姿态应该表示：

```text
脚稳定支撑
躯干基本竖直
腿部适度压缩
垂直速度接近 0 或准备向上
角速度可控
没有身体其他部位碰地
处于可起跳姿态
```

### 建议重点特征

```text
theta                    接近竖直
omega                    接近 0
knee                     在可起跳压缩区间
leg_compression          适度压缩
touch_toe/touch_heel     整足支撑
vz                       接近 0 或略向上
height                   不能过低
body_collision           必须低
```

### 关键注意事项

ground-prepared reward 不能让 agent 永远停在地面 prepared 姿态。

否则可能出现：

```text
蹲着不跳
稳定站着
反复微调膝盖刷 prepared 分数
```

因此 ground-prepared 更适合使用“潜势差分”奖励，而不是纯状态占有奖励。

---

## 5.3 空中上升：rise-extension 姿态

### 目标

上升相位的标准舒展姿态是合理的，但要修正一个物理理解：

> Hopper 在空中不能通过腿部舒展直接获得更远距离。
> 飞行距离主要由起跳瞬间的 \(V_x\)、\(V_z\) 和角动量决定。

所以 rise-extension 的作用不是“在空中伸腿就飞得更远”，而是：

```text
保持躯干稳定
降低角速度
保持良好飞行姿态
为后续下降和落地准备
避免空中乱摆腿
```

同时必须把 rise-extension 与起跳质量绑定：

```text
vx 为正且较大
vz 为正
theta 稳定
omega 较小
腿部舒展合理
```

### 建议重点特征

```text
theta                    接近竖直
omega                    小
vx                       正向且尽量接近 hop speed
vz                       正向
knee / leg_extension     合理舒展
ankle                    不极端
foot trajectory          不乱摆
```

### 不建议

不要只奖励“腿伸直”。如果只奖励腿伸直，agent 可能学到：

```text
原地小跳 + 空中摆出漂亮姿态
```

但没有真正完成 Hopper Hop 的前向移动任务。

---

## 6. 使用软相位门控，而不是硬状态标签

不建议直接写：

```python
if state == falling_air:
    reward = score_land_prep
elif state == grounded_prepared:
    reward = score_ground_prep
elif state == rising_air:
    reward = score_rise_extend
```

因为当前硬状态边界本身比较脆弱，而且 `rising_air` 这类标签可能包含空中顶点或轻微下降阶段。

更推荐定义连续相位 gate。

### 6.1 空中下降 gate

\[
g_ {down}
=========

g_{air}
\cdot
\sigma
\left(
\frac{-v_z - v_{fall}}{s_v}
\right)
\]

其中：

```text
g_air 表示处于空中
v_fall 是下降速度阈值
s_v 是平滑尺度
```

### 6.2 地面支撑 gate

\[
g_ {ground}
===========

g_{foot}
\cdot
g_{support}
\]

其中：

```text
g_foot      表示脚接近或接触地面
g_support   表示 toe/heel 接触强度足够
```

建议不要只用 `z_min`，因为 `z_min` 是全身最低点，会把脚部接地和身体碰地混在一起。

### 6.3 空中上升 gate

\[
g_ {rise}
=========

g_{air}
\cdot
\sigma
\left(
\frac{v_z - v_{rise}}{s_v}
\right)
\]

### 6.4 gate 归一化

\[
\tilde g_i
==========

\frac{g_i}{g_{down}+g_{ground}+g_{rise}+\epsilon}
\]

这样可以避免多个 gate 同时激活导致奖励尺度飘动。

---

## 7. 推荐奖励形式：相位姿态潜势

### 7.1 相位姿态潜势

定义：

\[
\Phi(s)
=======

\tilde g_{down} S_{land}(s)
+
\tilde g_{ground} S_{ground}(s)
+
\tilde g_{rise} S_{rise}(s)
\]

其中：

```text
S_land    是当前状态到 landing-prep 姿态的相似度
S_ground  是当前状态到 ground-prepared 姿态的相似度
S_rise    是当前状态到 rise-extension 姿态的相似度
```

### 7.2 推荐使用势函数差分

为了防止 agent 停在某个高分姿态，建议主要使用：

\[
r_ {dense}
==========

\alpha_{\Delta}
\left[
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right]
\]

这表示：

```text
奖励“变得更接近当前相位目标”
而不是奖励“待在目标附近不动”
```

这对 ground-prepared 尤其重要。

### 7.3 可加入少量绝对项

如果发现差分项信号太弱，可以加入少量绝对项：

\[
r_ {dense}
==========

\alpha_{\Delta}
\left[
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right]
+
\alpha_{abs}
\Phi(s_{t+1})
\]

建议：

```text
alpha_delta > alpha_abs
```

例如：

```text
alpha_delta = 1.0
alpha_abs   = 0.1 ~ 0.3
```

---

## 8. 质量门：防止姿态奖励脱离任务目标

### 8.1 forward gate

Hopper Hop 的目标是前向跳跃，而不是原地摆姿态。

因此上升相位与起跳相关奖励应乘以：

\[
g_ {forward}
============

clip
\left(
\frac{V_x}{V_{hop}}, 0, 1
\right)
\]

或者平滑版本：

\[
g_ {forward}
============

\sigma
\left(
\frac{V_x - V_{min}}{s_x}
\right)
\]

其中 \(V_{hop}\) 可以取官方 hop speed 目标附近，例如 2.0。

### 8.2 landing quality gate

下降/落地相关奖励应检查落地质量：

```text
落地后 |vz| 不能过大
落地后 |omega| 不能过大
不能 body_collision
不能明显穿地
不能高冲击反弹
```

可以定义：

\[
g_ {landing}
============

g_{stable\_vz}
\cdot
g_{stable\_omega}
\cdot
g_{no\_body\_collision}
\]

### 8.3 support quality gate

地面 prepared 需要真正脚部支撑，而不是身体其他部位碰地：

\[
g_ {support}
============

g_{toe}
\cdot
g_{heel}
\cdot
g_{no\_body\_collision}
\]

建议显式区分：

```text
foot_grounded
body_collision
```

而不是只用 `z_min`。

### 8.4 takeoff quality gate

prepared 到 rising 的奖励不应该只看是否离地，而应看起跳质量：

```text
vx_next > 0
vz_next > v_rise
theta_next stable
omega_next not too high
```

---

## 9. 保留原状态迁移奖励，但降级为 milestone

不建议完全删除原来的硬迁移奖励。它可以作为“里程碑奖励”保留，但不应再作为主要引导信号。

推荐总结构：

\[
r_ {shape}
==========

r_{dense\_pose}
+
\beta r_{milestone}
\]

其中：

```text
r_dense_pose   是主要密集奖励
r_milestone    是旧的六项状态迁移奖励
beta           小于旧版本权重
```

例如：

```text
beta = 0.1 ~ 0.3
```

旧迁移奖励可以继续表达：

```text
恢复
准备
起跳
落地
```

但密集姿态奖励负责提供大部分连续引导。

---

## 10. 三个 prototype 的来源

### 10.1 第一阶段：人工初始化

初期可以先人工定义三个 prototype：

```text
mu_land_prep
mu_ground_prep
mu_rise_extend
```

每个 prototype 包含：

```text
theta
omega
hip
knee
ankle
vx
vz
leg_compression
foot_x_rel
touch_toe
touch_heel
```

并配套：

```text
sigma_land_prep
sigma_ground_prep
sigma_rise_extend
weights_land_prep
weights_ground_prep
weights_rise_extend
```

### 10.2 第二阶段：从高回报轨迹中估计

当出现较好轨迹后，建议从高回报 episode 中按相位提取数据，估计：

\[
\mu_i = \mathbb{E}[\phi(s) \mid phase=i, high\ return]
\]

\[
\Sigma_i = Cov[\phi(s) \mid phase=i, high\ return]
\]

然后使用 Mahalanobis 距离：

\[
D_i(s)
======

(\phi(s)-\mu_i)^T
\Sigma_i^{-1}
(\phi(s)-\mu_i)
\]

如果不想直接估计协方差，可以先用 EMA 更新 prototype：

\[
\mu_i \leftarrow (1-\eta)\mu_i + \eta \bar{\phi}_i
\]

注意：

```text
只用高回报轨迹更新 prototype
不要让失败轨迹带偏目标姿态
更新速率 eta 要小
prototype 更新最好有 warmup
```

### 10.3 第三阶段：可学习但受约束的 prototype

如果把 prototype 设计为可学习参数，要加约束：

```text
knee 目标范围不能漂到不可起跳区域
theta 目标不能偏离竖直太多
touch 目标必须符合相位语义
sigma 不能无限变大
```

否则模型可能通过“移动目标”来降低损失，而不是真正学会跳跃。

---

## 11. 最小可实现版本

下面是一个建议的最小版本，不需要一次性重写全部系统。

### Step 1：实现姿态特征提取

```python
phi = {
    "theta": theta,
    "omega": omega,
    "hip": hip,
    "knee": knee,
    "ankle": ankle,
    "vx": vx,
    "vz": vz,
    "leg_compression": c_leg,
    "foot_x_rel": foot_x - hip_x,
    "touch_toe": touch_toe,
    "touch_heel": touch_heel,
}
```

### Step 2：定义三个 prototype 和 tolerance

```python
mu_land_prep = {...}
mu_ground_prep = {...}
mu_rise_extend = {...}

sigma_land_prep = {...}
sigma_ground_prep = {...}
sigma_rise_extend = {...}

weights_land_prep = {...}
weights_ground_prep = {...}
weights_rise_extend = {...}
```

### Step 3：定义姿态分数

```python
def wrapped_angle_diff(x, y):
    return (x - y + math.pi) % (2 * math.pi) - math.pi

def robust_loss(z):
    # Huber-like or clipped quadratic
    return torch.where(
        z.abs() < 1.0,
        0.5 * z.pow(2),
        z.abs() - 0.5,
    )

def pose_score(phi, mu, sigma, weights, angle_keys):
    d = 0.0
    for k in phi.keys():
        if k in angle_keys:
            diff = wrapped_angle_diff(phi[k], mu[k])
        else:
            diff = phi[k] - mu[k]
        z = diff / (sigma[k] + 1e-6)
        d = d + weights[k] * robust_loss(z)
    return torch.exp(-d)
```

### Step 4：定义软相位 gate

```python
air_gate = sigmoid((z_min - ground_eps) / z_scale)

down_gate = air_gate * sigmoid((-vz - v_fall) / v_scale)
rise_gate = air_gate * sigmoid((vz - v_rise) / v_scale)

foot_gate = sigmoid((ground_eps - foot_z_min) / z_scale)
support_gate = min(touch_toe, touch_heel)
ground_gate = foot_gate * support_gate
```

如果暂时没有 `foot_z_min`，可以先用 `z_min`，但这只是临时方案。长期建议区分脚部接地和身体碰撞。

### Step 5：定义势函数

```python
score_land = pose_score(phi, mu_land_prep, sigma_land_prep, weights_land_prep)
score_ground = pose_score(phi, mu_ground_prep, sigma_ground_prep, weights_ground_prep)
score_rise = pose_score(phi, mu_rise_extend, sigma_rise_extend, weights_rise_extend)

score_rise = score_rise * forward_gate
score_ground = score_ground * support_quality_gate
score_land = score_land * landing_prep_quality_gate

gate_sum = down_gate + ground_gate + rise_gate + 1e-6

Phi = (
    down_gate * score_land
    + ground_gate * score_ground
    + rise_gate * score_rise
) / gate_sum
```

### Step 6：定义 dense reward

```python
r_dense = alpha_delta * (gamma * Phi_next - Phi_now)
r_dense = r_dense + alpha_abs * Phi_next
```

### Step 7：与旧 milestone 奖励合并

```python
r_shape = r_dense + beta * r_milestone
```

---

## 12. 建议的训练接入方式

### 12.1 reward head 使用双头或分解监督

如果当前 reward head 只学习：

```text
env_reward + alpha * shaping
```

那么 shaping 可能污染最终优化目标。

更建议拆成：

```text
env_reward_head
shape_reward_head
```

规划时使用：

```text
planned_reward =
    env_reward_head
  + alpha_schedule * shape_reward_head
```

其中：

```text
alpha_schedule 早期较大，后期逐渐降到 0 或很小
```

### 12.2 alpha 后期应衰减

如果最终目标仍然是官方 Hopper Hop reward，那么建议：

```text
alpha -> 0
```

或者至少：

```text
alpha -> very small
```

不要长期保持较大的 shaping 权重，否则可能出现“姿态很好但不向前跳”的局部最优。

### 12.3 可以分阶段启用

推荐 curriculum：

```text
阶段 1：恢复和姿态稳定
阶段 2：prepared 和起跳
阶段 3：前向速度和周期跳跃
阶段 4：弱化 shaping，强化 env reward
```

---

## 13. 需要重点防范的新 reward hacking

### 13.1 停在 prepared 不起跳

风险：

```text
ground-prepared 分数高
agent 选择一直保持该姿态
```

解决：

```text
使用潜势差分
prepared 持续时间过长后衰减
prepared 后进入 rising 才给 milestone
ground reward 不要太大
```

### 13.2 空中摆姿态但没有前向速度

风险：

```text
rise-extension 分数高
但 vx 很低，实际是原地小跳
```

解决：

```text
rise score 乘 forward_gate
takeoff milestone 检查 vx_next
后期降低纯姿态权重
```

### 13.3 漂亮下降但高速砸地

风险：

```text
landing-prep 姿态接近目标
但落地冲击过大
```

解决：

```text
landing quality gate
impact penalty
落地后 |vz| 和 |omega| 约束
body_collision penalty
```

### 13.4 prototype 被失败轨迹带偏

风险：

```text
从低质量轨迹估计目标姿态
导致目标越来越偏
```

解决：

```text
只用高回报轨迹更新
EMA 慢速更新
设置物理合理范围
保留人工先验
```

### 13.5 planner 利用 world model 漏洞刷姿态分

风险：

```text
模型预测中姿态分数变高
但真实环境中不可实现
```

解决：

```text
真实环境 rollout 验证
短 horizon + model uncertainty penalty
奖励分数 clipping
监控 model-predicted shape return 与 real env return 相关性
```

---

## 14. 建议的诊断实验

### 14.1 稀疏性诊断

统计旧 shaping reward：

```text
每个 episode 中非零 shaping transition 的比例
每种 milestone 出现次数
平均每 1000 环境步的 shaping 次数
```

如果非零比例很低，就支持“奖励过稀疏”的判断。

### 14.2 密集奖励相关性

训练前或离线 replay 上计算：

```text
dense pose return
env return
old milestone return
```

画相关性：

```text
dense pose return vs env return
old milestone return vs env return
```

如果 dense pose return 与 env return 更相关，说明新奖励更有引导价值。

### 14.3 三相位可视化

按相位随机抽帧：

```text
down_gate 最高的帧
ground_gate 最高的帧
rise_gate 最高的帧
```

检查它们是否符合人类理解。

### 14.4 prototype 距离热图

对每个 episode 画：

```text
D_land(t)
D_ground(t)
D_rise(t)
gate_down(t)
gate_ground(t)
gate_rise(t)
Phi(t)
env_reward(t)
```

这能看出奖励是否在正确时刻起作用。

### 14.5 ablation

至少比较：

```text
A. env reward only
B. old hard transition shaping
C. dense pose shaping only
D. dense pose + milestone
E. dense pose + milestone + prototype update
```

关键指标：

```text
sample efficiency
final return
fall frequency
successful hop cycles
average vx
landing stability
model exploit frequency
```

---

## 15. 与当前状态机设计的关系

新的密集奖励不必完全替代当前状态机。

建议关系如下：

```text
当前六状态机：
    用于诊断、milestone、可视化、少量阶段推进奖励

新的三相位姿态奖励：
    用于主要密集引导

当前连续诊断量：
    从“导出/分析用途”升级为 reward feature
```

也就是说，不需要废弃：

```text
falling_air
grounded_upright
grounded_prepared
rising_air
```

而是把它们从主要奖励信号降级为：

```text
事件标签
里程碑
debug 工具
```

真正提供连续梯度的是三相位目标姿态距离。

---

## 16. 推荐优先级

如果只做最小改动，建议按以下顺序：

1. **增加三个 prototype 的姿态分数**
   先离线计算，不接入训练，验证分数是否符合直觉。
2. **实现软相位 gate**
   不再完全依赖 hard state id。
3. **用势函数差分构造 dense reward**
   避免 prepared 停滞。
4. **给 rise 加 forward gate，给 landing 加 impact / stability gate**
   防止姿态奖励偏离 Hopper Hop 目标。
5. **把旧迁移奖励降权为 milestone**
   不再让硬状态切换承担主要引导。
6. **逐步把 alpha 衰减到 0 或很小**
   确保最终策略仍优化官方任务。

---

## 17. 一句话结论

当前奖励效果弱的一个很可能原因，确实是：

> 只奖励进入下一状态，对 Hopper 这种多自由度连续控制系统过于稀疏。

用三个相位条件目标姿态构造密集奖励，是合理且值得实现的改进方向。

但建议不要把它实现成简单的姿态 L2 距离，而应实现为：

```text
软相位 gate
+ 加权归一化姿态距离
+ 目标姿态流形
+ 势函数差分
+ forward / landing / support 质量门
+ 低权重 milestone 奖励
```

这样它既能提供密集反馈，又不容易偏离 Hopper Hop 的真实目标。
