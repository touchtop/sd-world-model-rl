# Hopper v8 成功延续规则提取设计

## 设计目标

本设计用于扩展现有 `SD Retrospective Attribution and Rule Induction` 闭环。现有 v7 规则主要从失败轨迹和高风险状态中提炼恢复规则，例如在 `rear`、`falling`、`rotation_risk` 或 `unsupported_high_torque` 状态下抑制或偏置某类动作。200k 最终轨迹审查显示，这类规则已经能够进入在线规划排序，但覆盖范围主要集中在坏状态恢复；最终 eval 中较理想的姿态片段，如 `omega_mid / vx_high / support_none / falling / rotation_risk=0`，没有被当前规则库覆盖。

v8 的目标不是替换失败恢复规则，而是在原框架中新增“成功延续与阶段衔接”规则通道。该通道关注局部好片段为什么能够出现、为什么未能延续、以及哪些动作有助于把短暂好姿态转化为稳定落地、支撑和再起跳。

## 通用方法边界

v8 继续采用“通用引擎 + Hopper 适配器”的边界。通用引擎只定义流程和接口，不直接写入 Hopper 专用相位名称或动作关节名称。

通用引擎包括以下模块：

1. `StateSchema`：声明任务可解释状态变量；
2. `PhaseDetector`：将逐帧结构态映射为阶段、风险状态和接近成功状态；
3. `QualityMetrics`：定义局部好、阶段衔接好和全局好；
4. `ActionFamilyMapper`：把连续动作或离散动作映射为可比较动作族；
5. `SegmentMiner`：从轨迹中提取失败片段、成功片段、成功中断片段和阶段转换片段；
6. `CounterfactualEvaluator`：在片段关键点上进行短视界动作反事实；
7. `RuleInducer`：将稳定证据压缩为规则；
8. `RuleValidator`：跨轨迹、跨 seed、跨周期验证规则；
9. `RuleLibraryUpdater`：按置信度强化、削弱、替换或保留规则。

Hopper 适配器负责实现具体语义，例如 `pitch`、`pitch_rate`、`vx`、`torso_z`、`full_foot_support`、`falling`、`rear`、`prone`、`waist/hip/knee/ankle` 等。未来迁移到 Walker、Cartpole 或其他控制任务时，应替换适配器，而不是改写通用闭环流程。

## 规则类型

v8 将规则分为三类，每类规则进入同一规则库，但保留 `rule_role` 字段。

第一类为失败恢复规则。该类沿用 v7 设计，处理已经进入明显坏状态或高风险状态的片段。其目标是减少旋转风险、后退、俯伏、后仰和无支撑高力矩动作。

第二类为成功延续规则。该类从局部高质量片段中提取，处理已经接近可用姿态的状态。Hopper 中的初始适配判据为：`torso_z` 较高、`abs(pitch)` 较小、`vx` 为正且较高、`abs(pitch_rate)` 不高、短视界 dense pose 或环境 reward 为正。规则目标不是“从坏姿态恢复”，而是“保持已有姿态优势并避免立即退化”。

第三类为阶段衔接规则。该类关注相邻阶段之间的转换，例如从下落到可控落地、从接地压缩到准备支撑、从准备支撑到起跳。Hopper 中优先关注 `falling -> landing/support`、`support -> rising` 和 `near-good flight -> next support`。该类规则用于解释为什么局部好片段不能延伸为完整跳跃周期。

## Hopper 初始适配口径

Hopper v8 先使用现有 trace CSV 中已保存的结构态字段，不改变 TD-MPC2 训练代码。初始字段如下：

1. 姿态变量：`pitch`、`pitch_rate`、`torso_z`、`height`；
2. 推进变量：`vx`、`vz`、`reward`、`qpos_rootx`；
3. 支撑变量：`touch_toe`、`touch_heel`、`full_foot_support`、`support_contact`、`foot_grounded`、`body_collision`；
4. 阶段变量：`falling_air`、`grounded_rear`、`grounded_prone`、`grounded_upright`、`grounded_prepared`、`rising_air`；
5. 姿态质量变量：`dense_pose_potential`、`landing_pose_score`、`ground_pose_score`、`rise_pose_score`、`posture_stability`、`spin_stability`；
6. 动作变量：`action_0` 至 `action_3`，分别映射为 `waist`、`hip`、`knee`、`ankle` 动作族。

初始成功片段判据采用宽松门控，避免只识别已经稳定成功的状态：

```text
torso_z >= 0.75
abs(pitch) <= 0.45
vx >= 0.80
abs(pitch_rate) <= 3.50
body_collision < 0.50
```

若 `dense_pose_potential >= 0.65` 或单步环境 reward 为正，也可作为成功片段补充入口。该判据只用于候选片段挖掘，不直接作为规则上线条件。

成功中断片段定义为：在短窗口内出现上述成功片段，但随后 `K` 步内出现以下任一退化：

1. `abs(pitch)` 明显增大；
2. `pitch_rate` 进入高旋转风险；
3. `torso_z` 快速下降；
4. `vx` 快速下降或转为后退；
5. 进入 `rear/prone`；
6. 下一次接地未形成足部支撑或出现身体碰撞。

阶段衔接片段定义为：某一阶段内局部指标较好，但进入下一阶段失败。例如飞行姿态较好但落地支撑失败、落地接触出现但未形成可再起跳准备、或有正向速度但支撑相位无法维持。

## 反事实目标

v7 的反事实目标主要衡量短视界恢复收益。v8 新增两类目标。

成功延续目标衡量当前动作是否延续已有好状态。Hopper 初始评分可写为：

```text
S_continue =
  0.25 * uprightness
+ 0.20 * height_quality
+ 0.20 * forward_speed
+ 0.15 * spin_control
+ 0.10 * support_readiness
+ 0.10 * non_collision
```

阶段衔接目标衡量从当前阶段到下一阶段的可达性。Hopper 初始评分可写为：

```text
S_transition =
  0.25 * landing_pose_score
+ 0.20 * ground_pose_score
+ 0.20 * rise_pose_score
+ 0.15 * support_contact
+ 0.10 * positive_vx
+ 0.10 * low_rotation_penalty
```

两个评分均只用于离线反事实排序。最终规则仍需通过跨轨迹、跨 seed、跨周期验证，不能只凭单次评分上线。

## 规则语法

v8 规则保留 v7 子状态键，但增加 `quality_bin` 和 `rule_role`。

初始 Hopper 子状态键为：

```text
rule_role|quality_bin|omega_bin|vx_bin|support_bin|phase|risk_bin -> prefer/avoid action_family
```

其中：

1. `rule_role` 取 `recovery`、`continue`、`transition`；
2. `quality_bin` 取 `bad`、`near_good`、`good_fragment`、`transition_gap`；
3. `omega_bin` 沿用 `omega_pos / omega_neg / omega_mid`；
4. `vx_bin` 沿用 `vx_low / vx_mid / vx_high`；
5. `support_bin` 沿用 `support_none / support_low / support_some`；
6. `phase` 沿用 `falling / rear / prone / upright / rising / other`；
7. `risk_bin` 取 `rotation_risk / support_risk / collision_risk / stable`。

成功延续规则可以生成 `prefer` 或 `avoid` 两种干预语义。若某动作族在好片段中稳定提高 `S_continue`，生成 `prefer`；若某动作族稳定导致姿态退化，生成 `avoid`。在线接入时，`avoid` 应优先采用 penalty-only，`prefer` 需要更高置信度。

## 置信度与冲突处理

v8 继续保留规则置信度机制，并将冲突处理作为规则库更新的一部分。每条规则记录以下量：

1. `support`：本周期支持样本数；
2. `cumulative_support`：跨周期累计支持；
3. `mean_advantage`：相对实际动作或对照动作的平均反事实优势；
4. `continuation_gain`：成功延续目标收益；
5. `transition_gain`：阶段衔接目标收益；
6. `good_preservation_rate`：规则命中后好片段保持为好片段的比例；
7. `failure_prevention_rate`：规则命中后避免进入失败片段的比例；
8. `conflict_penalty`：同一子状态、同一关节上相反方向规则的惩罚；
9. `effective_prior_scale`：在线先验实际使用权重。

同一子状态、同一关节上的相反方向规则不应并存为同等强度。新证据先削弱旧规则置信度；旧规则置信度降至阈值以下后，若反向证据仍持续出现，再形成反向规则。该机制保证规则库可以随训练阶段变化而修订，而不是永久固化早期经验。

## 在线接入原则

v8 设计阶段不直接提高在线干预强度。规则进入在线先验时遵循以下顺序：

1. `recovery/avoid` 规则优先以 penalty-only 接入；
2. `continue/avoid` 规则优先以 penalty-only 接入，用于避免破坏好姿态；
3. `continue/prefer` 规则必须具备更高跨 seed 证据后才能正向加权；
4. `transition` 规则先用于离线诊断和候选排序，待验证后再接入；
5. 所有规则均只影响 MPPI 候选排序，不替换最终动作。

该原则保持 v6/v7 的轻量介入思想，同时补上好片段延续与阶段衔接的规则来源。

## 初始实现步骤

第一步，新增 Hopper 成功片段与阶段衔接片段挖掘脚本。该脚本读取 `official_eval/episode_csv/*.csv` 和回放导出的训练轨迹，输出 `success_fragments.json`、`transition_gaps.json` 和对应 Markdown 报告。

第二步，扩展反事实脚本，使其支持 `objective=recovery|continue|transition`。同一关键状态可同时计算恢复收益、成功延续收益和阶段衔接收益。

第三步，新增 v8 规则归纳脚本，将 `rule_role`、`quality_bin` 和 `risk_bin` 写入规则键。该脚本不删除 v7 规则，只生成新增规则族。

第四步，扩展回放验证，对成功片段统计规则是否保持好姿态，对失败片段统计规则是否降低退化概率，并单独计算好片段误伤率。

第五步，扩展闭环规则更新器，使其能把 `recovery`、`continue`、`transition` 三类规则纳入同一置信度更新框架，并处理同一关节反向规则冲突。

第六步，在不重训的情况下，先用 200k 最终 eval 轨迹和已有周期轨迹离线验证 v8 是否能解释已观察到的好姿态片段。只有当离线结果显示新规则覆盖好片段并能解释退化原因后，再考虑进入新的闭环训练。

## 预期判断标准

v8 第一版不以 Hopper 200k 或 300k 回报立即提升为唯一目标。设计验证应优先看以下信号：

1. 是否能系统识别最终 eval 中的高质量姿态片段；
2. 是否能识别这些片段之后的退化原因；
3. 是否能产生不同于失败恢复规则的成功延续或阶段衔接规则；
4. 新规则是否覆盖 `omega_mid / vx_high / falling / support_none / stable` 等当前空白子状态；
5. 新规则是否在高质量片段中低误伤；
6. 新规则是否能跨 seed 或跨周期重复出现；
7. 冲突规则是否能通过置信度机制被削弱、替换或保留为待观察状态。

若上述信号成立，即使短期训练回报尚未提升，也可说明方法论从“失败归因”扩展到了“成功保持与阶段衔接归因”。
