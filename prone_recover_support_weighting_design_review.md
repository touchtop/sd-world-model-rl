# Prone Recover 势函数调整方案评审意见

## 1. 背景与当前观察

当前实验将训练目标大幅简化为单一子任务：从前倾伏地姿态恢复到站起姿态。实验中暂时排除了 Hopper Hop 的完整环境奖励、状态迁移 milestone 以及其他复杂 shaping，只保留 prone recover 相关的 dense pose potential。

你观察到的现象可以概括为两点：

1. 在 5k 这样极小预算下，单独姿态训练已经让 agent 开始获得一些环境分数。此前在同等预算下几乎没有得分，这说明 prone_recover 姿态势函数确实进入了学习通道，并且它内在地促进了部分环境奖励。
2. 但 agent 主要学到的是“抬身 / 展开 / 爬行式改善”，而不是“足部主支撑下真正站起来”。也就是说，当前奖励有信号，但信号还没有把“站到脚上”塑造成必要条件。

因此，问题不应被理解为“奖励完全无效”，而应理解为：

> 姿态奖励已经推动行为变化，但高势值区域没有被足部主支撑严格约束住。

这意味着下一步不应简单地否定当前设计，而应沿着当前有效信号继续收紧目标结构。

---

## 2. 当前问题的核心判断

当前 prone_recover 势函数大致可以被理解为：

```text
姿态展开 / 躯干抬升 / pitch 变正 / 几何居中 / 膝角合适
```

这些因素确实能推动 agent 从前伏状态开始活动。但如果这些因素可以在没有足部支撑的情况下形成中高势值平台，那么 agent 就没有足够强的动力继续把身体支撑转移到 toe / heel 上。

更关键的是，当前 reward 中加入了一个小的 absolute occupancy 项。如果这个 absolute 项作用于整个 Phi，而 Phi 又包含不依赖足部支撑的 unfold 通道，那么 agent 会在“抬身但不真正站起”的状态上持续获利。

因此，主矛盾不是简单的：

```text
support 权重不够高
```

而是：

```text
无足部支撑的 unfold 改善也能形成过高势值；
占有奖励没有只奖励足部支撑后的站立部分。
```

一句话概括：

> 当前奖励已经让 agent 愿意动，但还没有让 agent 必须站到脚上。

---

## 3. 对你提出的三条调整原则的判断

你提出的调整方向是：

1. 保留 unfold，但只让它负责起步；
2. 把足部支撑变成高势值的更强必要条件；
3. 把 absolute occupancy 奖励只作用于支撑成立后的恢复部分，而不是作用于整个 Phi。

我认为这三条是可行的，而且是当前最应该做的方向。

更准确地说，下一版奖励设计应遵循：

> unfold 负责让 agent 从前伏姿态中开始尝试恢复；support 负责决定 agent 是否能够进入高势值区，并决定是否能够持续获得占有奖励。

这比简单地把 `support_score` 的加法权重从 0.20 提到 0.40 更合理。

---

## 4. 为什么不能只简单提高 support 的加法权重

假设把当前加法式势函数改成：

```text
Phi = 0.40 support + 0.20 pitch + 0.20 lift + 0.10 geom + 0.10 knee
```

这看起来提高了足部支撑的重要性，但仍然有漏洞。因为在加法结构中，agent 可以通过其他项补偿 support 缺失：

```text
support 较低
pitch 较好
lift 较高
geom 较好
knee 较合适
```

这种状态仍可能获得不低的总势值。

如果目标是让 agent 真正产生“必须站起来”的意愿，那么 support 不应该只是一个加法项，而应成为高势值区域的门控项或上限项。

推荐思路是：

```text
无足部支撑时，Phi 只能达到较低上限；
有弱足部支撑时，Phi 可以继续上升，但仍有限；
有强足部支撑、身体离地、躯干竖直时，Phi 才能进入高分区。
```

也就是说，support 应该改变势函数的拓扑，而不仅仅改变一个系数。

---

## 5. 推荐的双通道势函数结构

建议把 prone_recover 势函数拆成两个分支：

```text
Phi_unfold：起步 / 展开通道
Phi_stand：足部支撑站起通道
```

总势函数：

```math
\Phi(s) = \Phi_{unfold}(s) + \Phi_{stand}(s)
```

这两个通道的职责不同：

| 通道 | 作用 | 是否需要足部支撑 | 是否允许形成高势值 |
|---|---|---:|---:|
| `Phi_unfold` | 让 agent 从前伏状态愿意动起来 | 不强制 | 不允许 |
| `Phi_stand` | 奖励真正站到脚上的恢复姿态 | 强制依赖 | 允许 |

---

## 6. Phi_unfold：起步通道

### 6.1 设计目标

`Phi_unfold` 只负责解决早期探索问题：

```text
从趴着不动 → 抬身、展开、尝试调整躯干和腿部
```

它不应该单独代表“站起来”。

### 6.2 建议公式

```math
\Phi_{unfold}(s) = U_{cap} \cdot g_{ground}(s) \cdot U(s)
```

其中：

```text
U_cap = 0.25 ~ 0.35
```

`U(s)` 可以由当前已有姿态指标组成：

```math
U(s) =
0.35 \cdot pitch\_score
+ 0.30 \cdot lift\_score
+ 0.20 \cdot geom\_score
+ 0.15 \cdot knee\_score
```

也可以保持更接近当前实现的权重，但应明确设置上限。

### 6.3 关键约束

`Phi_unfold` 的最大值应被限制在较低区间，例如：

```text
0.25 ~ 0.35
```

这样 agent 可以因为抬身和展开获得初始正反馈，但如果没有足部支撑，就无法进入高势值区。

---

## 7. Phi_stand：足部支撑站起通道

### 7.1 设计目标

`Phi_stand` 表示真正有价值的恢复状态：

```text
脚部承担支撑
身体其他部位逐渐离地
躯干接近竖直
高度足够
膝角处于可站立 / 可后续起跳的区间
角速度不过大
```

这部分应决定 agent 是否能够获得高势值和持续占有奖励。

### 7.2 建议公式

```math
\Phi_{stand}(s)
=
(1-U_{cap})
\cdot g_{foot}(s)
\cdot g_{support}(s)
\cdot g_{body\_clear}(s)
\cdot Q_{stand}(s)
```

其中：

```text
g_foot：脚部近地 / 脚部接触门
g_support：toe / heel 支撑质量门
g_body_clear：身体其他部位离地程度
Q_stand：站立姿态质量
```

### 7.3 建议的 Q_stand

```math
Q_{stand}(s) =
0.35 \cdot pitch\_score
+ 0.25 \cdot lift\_score
+ 0.15 \cdot knee\_score
+ 0.15 \cdot torso\_over\_foot\_score
+ 0.10 \cdot spin\_cap
```

其中 `spin_cap` 不作为主奖励，而作为质量上限项或弱质量项。这样不会让 agent 只追求低角速度，但可以避免它高速甩过竖直姿态拿高分。

---

## 8. support 门控建议

当前已有：

```text
support_contact = max(touch_toe, touch_heel)
full_foot_support = min(touch_toe, touch_heel)
```

可以组合为：

```math
S(s) = 0.5 \cdot \max(touch_{toe}, touch_{heel})
+ 0.5 \cdot \min(touch_{toe}, touch_{heel})
```

但不建议直接线性使用 `S`。建议改成非线性门控：

### 方案 A：平方门

```math
g_{support}(s) = S(s)^2
```

这会显著压低弱接触的贡献：

```text
S = 0.3 → S² = 0.09
S = 0.6 → S² = 0.36
S = 0.9 → S² = 0.81
```

### 方案 B：sigmoid 门

```math
g_{support}(s) = \sigma\left(\frac{S(s)-0.35}{0.08}\right)
```

两者都可以。若希望先简单验证，建议先用 `S^2`。

---

## 9. body_clear 是必要补充

仅仅提高 foot support 仍然不够。因为可能出现：

```text
toe / heel 有接触，但躯干、鼻尖、大腿或小腿仍然贴地。
```

这不是足部主支撑站起。

当前系统中的 `z_min` 是全身几何代表点的最低高度，不是脚部专用接地量。因此，如果只用 `z_min` 判断 grounded，就会把“身体某处碰地”和“脚部支撑”混在一起。

建议显式拆出：

```text
foot_min_z
body_min_z
```

然后定义：

```math
g_{body\_clear}(s)
=
\sigma\left(\frac{body\_min\_z - 0.04}{0.02}\right)
```

如果当前还没有 `body_min_z`，可以优先从几何链中排除 toe / heel / foot，只取 torso、nose、pelvis、thigh、calf 的最低点构造。

这样可以区分：

```text
脚在地上，身体离地   → 好
脚在地上，身体也趴地 → 不好
```

---

## 10. torso_over_foot 比 nose 居中更接近站起语义

此前 `geom_score` 使用鼻尖相对整体包络中心的位置。这对检测“从前伏中展开”可能有帮助，但它并不是最直接的站起指标。

对于足部支撑站起，更关键的是：

```text
躯干 / 质心是否位于支撑点附近或上方。
```

因此建议在 `Phi_stand` 中新增：

```math
torso\_over\_foot\_score
=
\exp\left(
- robust\left(\frac{x_{torso} - x_{contact}}{0.35}\right)
\right)
```

其中 `x_contact` 可以由 toe / heel 接触强度加权：

```math
x_{contact} =
\frac{touch_{toe} x_{toe} + touch_{heel} x_{heel}}
{touch_{toe} + touch_{heel} + \epsilon}
```

如果 toe / heel 都接近 0，则退化为 toe/heel 中点。

`nose_x` 居中仍可保留在 `Phi_unfold` 中，用于起步展开；但不建议在最终站起质量中占最高权重。

---

## 11. 推荐的最终奖励形式

建议奖励分成两部分：

1. 总势函数差分，推动从低势值走向高势值；
2. 只作用于 `Phi_stand` 的占有奖励，鼓励站到脚上后保持。

公式：

```math
r_t
=
w_{\Delta}
\left[
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right]
+
w_{abs}\Phi_{stand}(s_{t+1})
```

推荐初值：

```text
w_delta = 0.40
gamma = 0.995
w_abs = 0.02 ~ 0.06
U_cap = 0.25 ~ 0.35
```

如果当前 agent 明显停留在“抬身爬行”平台，建议先用：

```text
U_cap = 0.25
w_abs = 0.04
```

这意味着：

```text
最多 25% 势值来自无支撑展开；
剩下 75% 必须通过足部支撑、身体离地和站立质量获得；
持续占有奖励只奖励 Phi_stand，不奖励 unfold。
```

---

## 12. 一个完整可实现草案

### 12.1 分数定义

```math
U(s) =
0.35 \cdot pitch\_score
+ 0.30 \cdot lift\_score
+ 0.20 \cdot geom\_score
+ 0.15 \cdot knee\_score
```

```math
S(s) =
0.5 \cdot \max(touch_{toe}, touch_{heel})
+ 0.5 \cdot \min(touch_{toe}, touch_{heel})
```

```math
g_{support}(s) = S(s)^2
```

```math
Q_{stand}(s) =
0.35 \cdot pitch\_score
+ 0.25 \cdot lift\_score
+ 0.15 \cdot knee\_score
+ 0.15 \cdot torso\_over\_foot\_score
+ 0.10 \cdot spin\_cap
```

### 12.2 势函数

```math
\Phi_{unfold}(s) =
0.25 \cdot g_{ground}(s) \cdot U(s)
```

```math
\Phi_{stand}(s) =
0.75
\cdot g_{foot}(s)
\cdot g_{support}(s)
\cdot g_{body\_clear}(s)
\cdot Q_{stand}(s)
```

```math
\Phi(s) = \Phi_{unfold}(s) + \Phi_{stand}(s)
```

### 12.3 奖励

```math
r_t =
0.40 \cdot
\left[
0.995 \cdot \Phi(s_{t+1}) - \Phi(s_t)
\right]
+ 0.04 \cdot \Phi_{stand}(s_{t+1})
```

---

## 13. 课程化建议

如果担心一开始 foot support 太强导致信号断掉，可以采用课程化：

```text
0k ~ 2k:   U_cap = 0.35, w_abs = 0.02
2k ~ 5k:   U_cap 线性降到 0.25, w_abs 升到 0.04
5k 之后:  U_cap = 0.20 ~ 0.25, w_abs = 0.04 ~ 0.06
```

这样前期允许 agent 通过 unfold 得到探索信号，后期逐渐强迫它把支撑转移到脚上。

---

## 14. 需要警惕的风险

### 14.1 support 门太强导致早期信号断裂

如果 `Phi_stand` 强依赖 support，而 `Phi_unfold` 又被压得太低，agent 可能重新回到“不知道怎么动”的状态。

解决方式：

```text
不要删掉 unfold；
只限制 unfold 上限；
必要时课程化降低 U_cap。
```

### 14.2 support 高不等于站起

足部接触高不代表身体已经离地，也不代表躯干在脚上方。

解决方式：

```text
加入 body_clear；
加入 torso_over_foot；
避免只看 touch_toe / touch_heel。
```

### 14.3 abs 奖励错误强化爬行平台

如果 abs 奖励仍然挂在总 Phi 上，agent 可能继续在 unfold 高但 support 低的姿态上吃持续奖励。

解决方式：

```text
abs 只挂 Phi_stand；
不要挂 Phi_unfold 或 Phi_total。
```

### 14.4 high Phi 是否真实代表站起

如果高 Phi 帧仍然出现身体贴地或弱支撑，则说明势函数仍有漏洞。

解决方式：

```text
做高 Phi 帧渲染；
检查 support、body_clear、torso_over_foot。
```

---

## 15. 建议的诊断实验

### 15.1 高 Phi 支撑质量检查

筛选：

```text
Phi > 0.7
Phi > 0.8
```

统计：

```text
support_score
full_foot_support
body_clear
torso_over_foot
pitch_score
lift_score
```

理想结果：

```text
高 Phi 帧应主要是强足部支撑、身体离地、躯干竖直。
```

如果 `Phi > 0.7` 但 `support_score < 0.3` 仍然常见，说明设计没有收住。

---

### 15.2 Phi_unfold 与 Phi_stand 分解曲线

对轨迹绘制：

```text
Phi_unfold
Phi_stand
Phi_total
r_delta
r_abs_stand
```

理想曲线：

```text
初期 Phi_unfold 先上升；
随后 Phi_stand 开始上升；
最终高 Phi 主要来自 Phi_stand。
```

如果整条轨迹都靠 `Phi_unfold` 得分，说明 unfold 上限仍然太高，或者 stand 门太难进入。

---

### 15.3 abs 奖励来源检查

记录每一步：

```text
r_delta
r_abs_stand
Phi_stand
support_score
body_clear
```

如果 agent 仍然趴地或爬行时 `r_abs_stand` 很高，说明 `Phi_stand` 定义不够严格。

---

### 15.4 脚接触但身体趴地检查

筛选：

```text
support_score > 0.6
body_clear < 0.3
Phi > 0.6
```

如果这种状态很多，说明需要加强 `body_clear` 或 `torso_over_foot`。

---

### 15.5 与环境回报的相关性

比较：

```text
episode_return_env
episode_return_dense
episode_mean_Phi_stand
episode_max_Phi_stand
```

理想情况：

```text
Phi_stand 上升应比 Phi_unfold 上升更能解释环境回报上升。
```

如果只有 `Phi_unfold` 上升而环境回报停滞，说明 agent 仍然停在“会动但没站起来”的阶段。

---

## 16. 最小可执行改动清单

如果只做最小修改，建议按以下顺序：

1. 将总 Phi 拆成 `Phi_unfold` 与 `Phi_stand`；
2. 给 `Phi_unfold` 设置上限，建议先取 `U_cap = 0.25`；
3. 让 `Phi_stand` 强依赖 `support_score^2`；
4. 增加 `body_clear`，防止脚接触但身体仍趴地；
5. 将 absolute reward 从 `Phi_total` 改为只作用于 `Phi_stand`；
6. 导出并绘制 `Phi_unfold / Phi_stand / Phi_total`；
7. 渲染检查高 `Phi_total` 和高 `Phi_stand` 的帧。

---

## 17. 最终结论

这次观察说明当前 prone_recover 姿态奖励并不是无效的。相反，它已经在 5k 极小预算下推动了行为变化，并且间接带来了环境分数。

但当前主要问题是：

```text
agent 学会了抬身和展开，却没有形成必须站到脚上的动力。
```

因此，下一步不应简单地继续调大原公式中的 support 加法权重，而应重构势函数层级：

```text
unfold 负责起步；
support / body_clear / torso_over_foot 负责高势值；
absolute occupancy 只奖励足部支撑后的 stand 分支。
```

推荐的核心公式是：

```math
\Phi = \Phi_{unfold} + \Phi_{stand}
```

```math
r_t =
0.40 \cdot [0.995\Phi(s_{t+1}) - \Phi(s_t)]
+ 0.04 \cdot \Phi_{stand}(s_{t+1})
```

其中：

```text
Phi_unfold 最高约 0.25；
Phi_stand 最高约 0.75；
Phi_stand 必须由足部支撑、身体离地和站立姿态共同决定。
```

这样可以把 agent 从“愿意动”进一步推向“必须站到脚上”，也更符合 prone_recover 子任务的真实目标。
