## 2026-04-25

### Hopper Hop：将 v5 方向收缩为可落地的极简灰箱 SD

在重新审阅 `docs/learnable_sd_hopper_design.md` 后，确认其总体方向正确，即应将 Hopper Hop 建模为“接触切换、腿部压缩、主动注能、姿态稳定”共同构成的灰箱闭环系统；但若按完整版直接实现，第一版会同时学习过多机制量与参数，存在明显的不可辨识性和训练不稳定风险。

因此，`v5` 的落地方向被收缩为一版极简灰箱设计，并形成新文档：

- `docs/learnable_sd_hopper_design_v5_minimal.md`

### 本次收缩的核心结论

1. 当前主线 SD 的问题不主要是阈值或权重，而是显式层中混入了过多面向状态判定、奖励塑形和诊断的任务量，导致其更像任务解释器，而不像稳定的机制模型。
2. `v5` 第一版应先定义最小 stock-flow 闭环，再让状态判断与奖励建立在其上，而不是反过来。
3. 第一版显式机制状态只保留 `A`，即电机激活迟滞；`F_leg`、`tau_bal`、`Loss` 仅作为代数量或诊断量，不作为独立 stock。
4. 第一版派生层只保留 `G` 与 `C_leg` 两个核心量，分别对应接触相位与腿部压缩。
5. `m` 与 `I` 在第一版中固定，不学习，以降低 `V/P`、`Omega/L` 双重表征带来的冗余与混淆。

### 关于结构化先验信息的口径

本轮同时统一了一条口径：

若 simulator 内部的几何坐标、body/site 坐标、接触量能够稳定获得，且训练、规划、评估三条链路口径一致，则这些量可以作为结构化先验使用，不应被机械地视为“不可用特权信息”。

若同一信息可由观测量与固定几何关系稳健推回，则优先推回；若推回误差明显，则允许直接读取 simulator 几何量。

### 对后续实现的约束

`v5` 第一版实现时，应遵循以下约束：

1. 先实现无 residual 的灰箱核；
2. 先验证 `G`、`C_leg`、`A`、`tau_bal` 的语义是否与轨迹一致；
3. 在无 residual 版本已经能够稳定重构主要相位后，再决定是否接入小 residual；
4. 状态标签与结构化奖励不再写进 SD 主状态定义，只允许读取其输出。

### 动力方程口径进一步定稿

在继续讨论后，`v5` 第一版的实现口径进一步收紧为：

1. 主存量固定为 `H, Vx, Vz, Theta, Omega, A`；
2. 接触门控 `G` 由 `touch_toe / touch_heel` 的软门控给出；
3. 腿部压缩 `C_leg` 由 `hip-contact` 几何长度定义；
4. 接触点 `p_contact` 由 toe 与 heel 的接触强度加权平均得到；
5. 腿推力方向不再单独学习，而直接采用腿向量单位方向 `e_leg`；
6. 激活需求 `A_des` 第一版只由动作经 `w_u, b_u` 压缩得到，不把高度误差、速度误差、姿态误差直接混入；
7. `F_leg` 采用“接触门控 × 弹性项 + 阻尼项 + 主动推力项”的结构；
8. `tau_bal` 采用最简 `PD` 形式；
9. `m, I, g, L0` 第一版固定，不学习。

这样处理后，保留下来的动力学主干与完整版保持一致，只是把第一版不必承担的分支机制和额外自由度收掉。

同时明确了一条实现口径：

虽然 `v5` 的主存量只有 `H, Vx, Vz, Theta, Omega, A` 六个，但实际代码中的结构态张量会额外缓存关节角、关节角速度、触地量和 `z_min` 等辅助槽位，以保证在 rollout 时不依赖 simulator 仍能持续计算接触、几何与状态判据。这些辅助槽位不改变主存量定义，只是实现层面的可递推缓存。

### 对现有状态与奖励设计的结论

本轮还同步梳理了当前六状态与硬迁移奖励在 `v5` 下是否需要推倒重来。结论如下：

1. 六状态集合总体可沿用；
2. 当前硬迁移奖励结构总体可沿用；
3. 真正需要改变的不是状态集合和迁移语义，而是判据输入来源；
4. 这些判据在 `v5` 中应优先改由 `H, Theta, Vz, G, C_leg, q_knee` 及必要几何接触量给出；
5. 状态判断与奖励在 `v5` 中属于上层分析/引导模块，不再混入 SD 主状态定义。

### 10k smoke 在 5000 评估阶段提前退出的直接原因

在检查 `sd_hopper_10k_v5minimal_landingboost_smoke_20260425_run1` 的目录结构后，确认训练主循环实际已运行至 `step=5000`，且 `5000.pt` 与 `5000.full.pt` 已完成落盘，但 `eval.csv` 未生成，`official_eval/step_0005000/episode_csv/` 目录为空。

沿执行链排查后，确认问题不在启动脚本参数，`scripts/run_hopper_v5_10k_smoke.sh` 中 `steps=10000` 配置正确。直接原因位于 `external/tdmpc2/tdmpc2/trainer/online_trainer.py`：

1. `_hopper_eval_state_features` 被误缩进到 `_physics_snapshot` 函数体内部；
2. 因此该函数并未注册为 `OnlineTrainer` 的类方法；
3. 当 `step=5000` 后下一回合进入 `eval()`，且 `save_eval_traces=true` 时，会走到 `self._hopper_eval_state_features(obs)`；
4. 此处会立即触发属性缺失异常，导致官方评估中断，因此不会写出 `eval.csv`，也不会导出 episode csv。

已将该函数恢复为类级方法，并完成两项静态校验：

1. `python -m py_compile external/tdmpc2/tdmpc2/trainer/online_trainer.py` 通过；
2. 通过导入 `OnlineTrainer` 确认 `_hopper_eval_state_features` 已重新出现在类属性列表中。

这一修复解释了“为何 5000 权重已存在而 5000 评估未完成”：因为训练循环会先按 `checkpoint_freq=5000` 落盘，再在下一次 episode 边界触发 `eval()`。

### v5 设计文档已按当前实现重新对齐

在继续推进 `v5` 试验时，再次审查了 `docs/learnable_sd_hopper_design_v5_minimal.md` 与当前代码的一致性。确认旧版文档中仍混有若干“早期目标口径”，与当前实现已不再一致，主要包括：

1. 旧文档仍把 `v5` 描述为“无 residual 的第一版灰箱核”；
2. 旧文档未准确写出当前 `17` 维显式结构态；
3. 旧文档未把奖励塑形的真实接入方式写清，即训练时通过 mixed reward target 进入 reward head，而不是简单理解为规划阶段直接显式相加；
4. 旧文档未把 `rear/prone` 两类状态下的条件动作先验盒裁剪写入；
5. 旧文档对六状态判据、膝窗口、整足支撑与躯干高度门的口径也未完全跟上当前实现。

据此，文档已整体重写，改为“当前实现版”描述，并明确区分三层：

1. 显式 `SD` 核；
2. 外层 `SDWorldModel` 的 `17 + residual` 混合 latent；
3. `TD-MPC2` 规划器中的条件动作先验与 reward-head 混合监督。

重写后，文档已补齐以下当前真实实现细节：

1. 原始 15 维 observation 布局；
2. 17 维显式结构态布局；
3. 几何正向运动学、`z_min`、接触点、腿压缩和腿方向的计算；
4. 激活迟滞、关节子系统、腿推力、姿态力矩与接触递推；
5. 六状态互斥判据与 dominant-state 机制；
6. 六项硬迁移奖励及其固定权重；
7. `reward_mode=curriculum` 下的 mixed reward target；
8. `rear/prone` 两类状态下的条件动作盒裁剪及其在规划器中的作用位置。

这次修订的目的是统一讨论口径：后续若再讨论 `v5` 的 `SD`、奖励塑形或动作裁剪，应以当前文档对应的代码实现为准，不再把旧版“目标草图”与现行实现混用。

### 根据评审意见落实四项优先修补

在进一步审阅 `docs/sd_hopper_v5_review_opinions.md` 后，按已确认优先级落实了四项代码修补，并同步收紧了当前主线口径。

第一，接触门控 `G` 不再只由 `touch_toe / touch_heel` 决定，而改为“触地强度门 × 几何近地门”的乘积，再对 toe 与 heel 取最大值。这样处理的原因是，原实现中接触强度带有 carry 记忆，若只看 `touch`，planner 可能利用离地后的残余接触信号继续施力；加入 `toe_z / heel_z` 的近地门后，只有在几何上确实接近地面时，接触门控才会保持开启。

第二，腿部阻尼项符号已调整。原实现为 `k*c - b*dc + active`，这会在腿部回弹阶段额外放大推力，存在向系统注入非物理能量的风险。现改为 `k*c + b*dc + active`，使阻尼在压缩阶段增强支撑，在伸展阶段减弱回弹，方向与压缩变量 `c_leg = L0 - L` 的定义保持一致。

第三，`z_min` 不再被视为可自由漂移的结构槽位，而是在 `project_state` 中依据投影后的 `H, Theta, waist, hip, knee, ankle` 重新做一次几何计算并回填。这样一来，无论是显式核自身递推，还是外层结构 residual 先修改了关节和姿态后再调用 `project_state`，最终落盘的 `z_min` 都与当前姿态保持一致。

第四，条件动作先验盒已降级为默认禁用。当前实现仍保留 `action_prior_bounds` 接口和相关统计字段，但默认返回全范围 `[-1, 1]`，仅在显式传入 `sd_action_prior_enabled=true` 时才重新启用 `rear/prone` 条件动作盒。这样处理的原因是，现有动作盒边界来自高分轨迹的启发式估计，其正确性尚不足以支撑继续作为主导引导机制；保留接口则是为了后续做单独 ablation。

本轮修补后，`v5` 的当前主线口径可概括为：

1. 奖励混合监督仍是当前结构化引导的主通道；
2. 接触力的开启必须同时满足接触强度与几何近地两条条件；
3. 几何派生量 `z_min` 已恢复为严格的姿态派生量，而不是可独立漂移的自由槽位；
4. 条件动作先验接口保留，但默认不再参与主线训练。

### 10k 重跑时发现并修复的额外实现错误

在按固定脚本重启 `v5 10k smoke` 时，训练能够正常走到 `2500` 随机采样结束并完成 `2500.pt / 2500.full.pt` 落盘，但在进入第一次参数更新后立即因 autograd 报错退出。错误类型为：

`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

结合本轮新增修改点排查后，最直接的触发源位于 `project_state`。此前为保证 `z_min` 与姿态几何一致，曾在函数尾部先 `stack` 出中间 `projected` 张量，再用列赋值方式把重算后的 `z_min` 回写进 `projected[:, ZMIN_IDX]`。这类列级 inplace 回填会把参与后续梯度图的 view 就地改写，进而在第一次反向传播时触发版本冲突。

因此，本轮已将 `project_state` 改为纯函数式拼装：

1. 先逐项得到投影后的 `H, Vx, Vz, Theta, ... touch_toe, touch_heel`；
2. 再基于这些投影后的姿态量计算 `kin["z_min"]`；
3. 最后一次性 `torch.stack([... , z_min], dim=-1)` 返回。

这样处理后，既保留了“`z_min` 必须由当前姿态重算”的几何一致性要求，也避免了 inplace 写回对梯度图的破坏。

### 固定 v5 的 50k 三种子主实验入口

在准备继续推进当前 `v5` 主线时，又核对了一遍现有 `50k` 启动脚本。确认 `scripts/run_hopper_v429_50k.sh` 仍固定为较早主线口径，其中 `sd_struct_dim=9`，已不再对应当前 `v5` 的 `17` 维显式结构态实现。因此，若继续直接复用该脚本，会把“已修订的 v5 代码”与“旧版 50k 参数口径”混用。

为避免再次在正式实验启动时手写长命令或误用旧脚本，本轮新增：

- `scripts/run_hopper_v5_50k.sh`

其作用仅是把当前已验证通过的 `v5 10k smoke` 训练参数平移到 `50k` 主实验口径，并固定：

1. `steps=50000`
2. `eval_freq=25000`
3. `checkpoint_freq=25000`
4. `eval_episodes=10`
5. `sd_struct_dim=17`
6. `sd_latent_dim=512`
7. `save_full_state=true`
8. `save_eval_traces=true`

同时，`docs/HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md` 中的 `50k` 入口也已同步补充为“当前 `v5` 推荐脚本优先、`v429` 旧脚本仅保留作旧口径复现”。

### 新增结构化奖励整体放大 10 倍的 10k 对照入口

在继续观察当前 `v5 50k` 三种子主实验的同时，为单独验证“若显著增强结构化吸引，策略会更快进入哪些行为模式”，本轮新增了一条对照口径：只放大结构化奖励，不改状态判据、不改单项权重、不改训练主框架。

具体实现方式为：

1. 在 `DMControlHopperSDModel` 中新增全局参数 `shaping_scale`；
2. `predict_shaping_reward()` 先按原 6 项硬迁移权重求和，再统一乘以该系数；
3. `SDWorldModel` 通过 `cfg.sd_shaping_scale` 将该参数传入显式核；
4. 默认值保持 `1.0`，因此不会影响当前主线实验；
5. 对照实验专用脚本 `scripts/run_hopper_v5_10k_structx10.sh` 将其固定为 `10.0`。

这样处理的理由是，若直接逐项改 6 个奖励系数，会同时改变各迁移之间的相对比例，不利于解释“只是整体吸引过弱”这一假设；而全局放大系数只改变结构化奖励的总量级，不改变其内部语义结构，更适合作为这一步的最小对照实验。

### 去除 `dominant_state >= 0.5` 的历史缓冲层，恢复闭环六状态标签

在继续用 `official_eval` 轨迹核对当前 `v5` 的六状态行为时，又暴露出一条与研究目标不一致的实现残留：虽然六个状态的语义判据此前已按“直接观测优先、闭环互斥”的原则逐项讨论清楚，但代码在把候选量收束为高层状态标签时，仍额外保留了一层历史缓冲：

1. 先取六个状态候选量的最大者；
2. 只有当最大值 `>= 0.5` 时，才赋予该步一个 dominant state；
3. 否则记为 `-1`，在轨迹分析脚本中表现为 `none`。

这层门槛的直接后果是：

1. 高层状态机重新出现“无标签空洞”；
2. 评估导出的 `*_gt_05` 列不再等同于闭环状态标签，而只是“某软量是否过门”；
3. `predict_shaping_reward()` 中的状态迁移虽然名义上是“硬迁移”，其上游却仍受到这层额外缓冲影响。

这与当前研究目标不符。因为本轮主线要求的是：

1. 六个高层状态闭环；
2. 每一步必须落入且仅落入其中一个状态；
3. 不再额外保留“都不像就算未定义”的缓冲技巧。

据此，本轮只做一处最小修订，不新增任何状态、不新增任何新阈值：

1. `_state_reward_features()` 继续保留现有候选量与诊断量；
2. `_dominant_state_ids()` 不再使用 `max_vals >= 0.5`；
3. 改为按既定高层语义直接强制六选一：
   - grounded 由 `z_min <= 0.03` 判定；
   - airborne 为其补集；
   - grounded 内依次判 `prepared / rear / prone`，其余 grounded 统一归入 `upright`；
   - airborne 内若 `vz < -0.05` 判为 `falling_air`，否则归入 `rising_air`。
4. `online_trainer.py` 的 `*_gt_05` 导出列同步改为依据唯一 `dominant_state_id` 写 one-hot，不再各自按 `>= 0.5` 重算。

这样处理后：

1. 高层状态标签重新回到“每步唯一、无空洞”的闭环状态机口径；
2. soft 候选量仍保留在导出中，供后续诊断使用；
3. 但奖励触发、动作先验状态命中与轨迹 one-hot 标签，均统一服从同一套高层状态标签。

### 按专家审核意见收紧六状态判据，并给关键迁移加质量门

在继续核对当前 `v5` 六状态实现时，又把专家对“状态判别与结构化奖励”的审核意见与现有代码逐条对照了一遍。结论是：当前六状态主链方向没有错，但最危险的漏洞集中在三处：

1. `grounded_upright` 过宽，容易把不稳定接地误判为站稳；
2. `rising_air` 过宽，容易把顶点或阈值边缘离地误判为起跳；
3. `z_min` 同时承担“任意部位接地”和“有效足部支撑”的语义，导致 shaping 容易奖励错误迁移。

考虑到当前主线已明确保持“闭环六状态，不再扩第七态”，本轮不引入新的主状态，而采用最小修补方案：保留六状态标签框架，但把“状态质量门”和“奖励屏蔽层”补进实现。

具体改动如下：

1. 在 `_state_reward_features()` 中新增并导出：
   - `foot_grounded`
   - `body_collision`
   - `airborne_true`
   - `upright_quality_gate`
   - `prepared_quality_gate`
   - `landing_quality_gate`
   - `rising_quality_gate`
   - `state_confident`
2. 其中：
   - `foot_grounded` 用“足部几何接地”与“足部接触强度”联合定义；
   - `body_collision` 用躯干、鼻尖、骨盆、大腿、小腿等非足部几何最低点定义；
   - `airborne_true` 仅在“既无足部支撑，也无身体碰撞”的离地条件下成立。
3. `grounded_upright` 不再只是“剩余 grounded 状态”，而是至少要求：
   - 足部形成支撑；
   - pitch 落在 upright 区间；
   - 高度不过低；
   - 角速度不过大；
   - 不存在身体碰撞。
4. `grounded_prepared` 在上述基础上进一步要求：
   - 整足支撑；
   - 膝角进入窗口；
   - `|vz|` 与 `|omega|` 足够小。
5. `falling_air` 与 `rising_air` 不再仅看 `z_min`，而是改为依赖 `airborne_true`；
   - `rising_air` 额外要求 `vz > 0.05`，不再用“不是 falling 就算 rising”的宽定义。
6. `predict_shaping_reward()` 保留原 6 条硬迁移结构，但不再裸触发：
   - `rear/prone -> upright` 乘 `upright_quality_gate`
   - `upright -> prepared` 乘 `prepared_quality_gate`
   - `prepared -> rising` 乘 `rising_quality_gate * forward_progress`
   - `falling -> prepared` 乘 `landing_quality_gate`
7. 若某一步仅被闭环标签“兜底”收纳，但未通过质量门，则该步 `state_confident = 0`，不会触发正向 shaping。

这样处理的理由是：

1. 继续保留六状态主框架，避免再次把状态图复杂化；
2. 但把“标签闭环”和“奖励可信度”拆开，阻止 planner 只刷阈值边界；
3. 让 `upright / prepared / rising` 这些真正参与正向塑形的状态先变得更干净、更接近动作语义；
4. 同时把 `foot_grounded / body_collision / airborne_true` 一并导出，便于后续直接在轨迹级验证每个判据是否与视频一致。

### 将 `upright` 的足部条件从“单侧支撑”收紧为“整足支撑”

在继续核对 `episode_07` 的轨迹与状态判据时，又暴露出一条语义不够自洽的残留：虽然 `prepared` 已明确要求 `full_foot_support`，但 `upright` 仍只要求 `support_contact = max(touch_toe, touch_heel)` 过门。这意味着仅前掌或仅后跟支撑，也可能被解释为“已站稳但尚未预备”。

从当前 Hopper 的接触表征与动力机制看，这样的分层并不自然。因为：

1. `touch_toe / touch_heel` 只是两个接触强度代理量，不是连续的足底压力分布；
2. 在该表征下，常见情况往往是：
   - 要么只有前掌或后跟一侧接地；
   - 要么两端同时高起来，此时基本已经属于较完整的整足支撑；
3. 因而“低门槛整足支撑”和“高门槛整足支撑”并不是两个稳定可分的物理阶段，继续让 `upright` 使用 `support_contact` 只会把大量脆弱单点支撑姿态误吸进来。

据此，本轮进一步收紧：

1. `upright_quality_gate` 不再使用 `support_contact_gate`；
2. 改为与 `prepared` 一样要求 `full_foot_support_gate`；
3. 这样 `upright` 与 `prepared` 的主要区别就只保留在：
   - 是否进入膝角准备窗口；
   - 垂直速度与角速度是否达到更严格的预备条件。

这样处理后，`upright` 的语义更接近“已经真正整足站住”，而 `prepared` 则表示“已经整足站住且完成起跳预备”，两者之间的层次关系也更符合当前任务的物理直觉。
