# 2026-04-26

## Hopper Hop：将 v5 结构化奖励从稀疏硬迁移改为 dense pose 主导

在继续审阅 `docs/sd_hopper_dense_pose_reward_design.md`、当前 `v5` 六状态实现和近几轮 `10k` 轨迹后，确认当前主线的瓶颈已经不主要是状态语义错误，而是奖励信号过于稀疏。

已有实现虽然已经把：

1. `grounded_upright`
2. `grounded_prepared`
3. `rising_air`

这些标签逐步收紧到较干净的语义，但 `predict_shaping_reward()` 仍主要依赖“命中某次状态迁移”才给奖励。对 Hopper 这种多自由度连续控制问题，这意味着：

1. 大量“正在变好但尚未跨阈值”的动作没有结构化反馈；
2. `rear/prone -> upright` 恢复阶段尤其缺少连续引导；
3. 即使状态语义逐步正确，训练仍可能长期学不到恢复链条。

据此，本轮将奖励主结构改为：

```text
dense pose reward
+ 小权重 milestone reward
```

而不是继续把六状态硬迁移当作唯一主通道。

## 本轮实现口径

### 1. 保留六状态，不推翻现有状态机

当前仍保留六个互斥高层状态：

1. `falling_air`
2. `grounded_rear`
3. `grounded_prone`
4. `grounded_upright`
5. `grounded_prepared`
6. `rising_air`

它们继续用于：

1. 诊断与轨迹导出；
2. 小权重里程碑奖励；
3. 后续状态统计与人工核对。

即，本轮不是“废弃状态机”，而是把状态机从主奖励降级为辅助标签层。

### 2. dense pose 只实现最小三相位

第一版不引入完整原型学习，也不一次性启用大量代理量，只保留三个软相位：

1. `dense_down_gate`
2. `dense_ground_gate`
3. `dense_rise_gate`

其中：

1. `down` 由 `airborne_true` 和向下速度软门给出；
2. `rise` 由 `airborne_true` 和向上速度软门给出；
3. `ground` 直接对所有 grounded 姿态开启。

这里特别没有把 `ground` gate 绑死到“整足支撑已形成”。原因是若地面 gate 只在已站稳时开启，那么当前最需要被拉回来的 `rear/prone` 失败姿态仍然几乎得不到 dense 信号。

### 3. dense pose 只使用一组高可信特征

本轮只使用下列量构造姿态分数：

1. `pitch`
2. `pitch_rate`
3. `qpos_knee`
4. `vx`
5. `vz`
6. `height`
7. `full_foot_support`
8. `body_collision`

理由是：

1. 这些量要么是直接观测，要么是当前几何链中已反复核对过的稳定派生量；
2. 不再把大量仍缺乏充分把握的代理量一并塞入奖励。

### 4. 当前 dense pose 的三类姿态分数

已在 `DMControlHopperSDModel._state_reward_features()` 中新增：

1. `landing_pose_score`
2. `ground_pose_score`
3. `rise_pose_score`
4. `dense_pose_potential`

三类分数统一采用：

```text
加权归一化残差
-> robust quadratic
-> exp(-distance)
```

这样做的目的，是避免纯阈值奖励，同时保持公式简洁、单调、可解释。

### 5. dense reward 的最终形式

当前 `predict_shaping_reward()` 已改为：

```text
shaping
= dense_pose_delta_weight * (Phi_next - Phi_prev)
+ dense_pose_abs_weight   * Phi_next
+ dense_pose_milestone_scale * milestone
```

默认系数为：

1. `sd_dense_pose_delta_weight = 0.40`
2. `sd_dense_pose_abs_weight = 0.05`
3. `sd_dense_pose_milestone_scale = 0.25`

这意味着：

1. 主信号来自“是否朝目标姿态更接近”；
2. 绝对姿态项只保留弱吸引；
3. 原六项硬迁移奖励继续存在，但只承担里程碑确认作用。

## 同步修改的工程项

### 1. 配置项

`external/tdmpc2/tdmpc2/config.yaml` 已新增：

1. `sd_dense_pose_delta_weight`
2. `sd_dense_pose_abs_weight`
3. `sd_dense_pose_milestone_scale`

并由 `SDWorldModel` 传入显式核。

### 2. 轨迹导出字段

`OnlineTrainer._hopper_eval_state_features()` 已新增导出：

1. `landing_pose_score`
2. `ground_pose_score`
3. `rise_pose_score`
4. `dense_down_gate`
5. `dense_ground_gate`
6. `dense_rise_gate`
7. `dense_pose_potential`

这样后续做轨迹分析时，可以直接从 `official_eval` 轨迹 csv 观察 dense 奖励是否在正确时刻激活，不再只能看最终状态标签。

## 本轮静态校验

本轮完成了以下校验：

1. `python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py`
2. `python -m py_compile external/tdmpc2/tdmpc2/common/sd_world_model.py`
3. `python -m py_compile external/tdmpc2/tdmpc2/trainer/online_trainer.py`
4. 直接构造张量调用 `encode_observation -> _state_reward_features -> forward -> predict_shaping_reward`，确认输出有限；
5. 用一条既有 `official_eval` 失败轨迹离线重算，确认 `dense_pose_potential` 不再全为 0。

在 `episode_09` 这类典型前伏失败轨迹上，离线重算得到：

1. `ground_pose_score` 平均约 `0.084`
2. `dense_ground_gate` 平均约 `0.98`
3. `dense_pose_potential` 平均约 `0.084`

这说明：即使尚未形成 `upright/prepared`，dense reward 也已经能够在 grounded failure 区间提供连续信号，而不再完全依赖一次硬迁移命中。

## 当前结论

本轮之后，`v5` 的结构化奖励口径已变为：

1. 六状态继续保留，但不再承担主要塑形职责；
2. 主要塑形来自三相位 dense pose potential；
3. milestone 只作为小权重事件确认；
4. 训练仍通过 mixed reward target 注入 reward head，规划阶段不额外显式重复叠加 shaping。

下一步更合适的工作，不再是继续细修旧的六项硬迁移权重，而是直接做 `10k` 级 smoke，检验 dense pose 是否开始在失败轨迹中诱导出更明确的姿态回正与状态迁移。

## 补充修订：收紧 `body_collision` 判据

在对 `seed1 @ 25k eval` 的 [episode_05.csv](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/1/sd_hopper_100000_v5densepose_seed1_20260426_run1/official_eval/step_0025000/episode_csv/episode_05.csv) 逐帧核对后，发现：

1. 初始较正常落地阶段，前掌与后掌在多帧上同时有高接触，`full_foot_support` 计算正确；
2. 这些帧未被判为 `upright` 的主要原因，不是脚部支撑不足，而是 `body_collision` 被误触发；
3. 误触发来源是小腿下缘 `calf_lower_z`，其数值约为 `0.009`，表示“小腿离地约 1 cm”，并非真实身体碰地；
4. 原实现直接复用 `STATE_GROUND_EPS = 0.03` 作为 `body_collision` 阈值，会把这类“离地很近但未碰地”的正常整足落地误判为身体碰撞。

因此本轮将 `body_collision` 的判据从：

```text
body_contact_z <= 0.03
```

收紧为：

```text
body_contact_z <= 0.00
```

并在代码中拆出独立常量：

1. `STATE_GROUND_EPS = 0.03` 继续仅用于“是否接地”的宽松判定；
2. `STATE_BODY_COLLISION_EPS = 0.00` 专用于“非足部几何是否真实碰地”的严格判定。

这样处理后，像 `episode_05` 早期这类“整足支撑、姿态较直、但小腿下缘离地很近”的帧，不再被系统性挡在 `upright` 之外。

## 补充修订：按 25k prone 局部最优诊断收紧 dense pose

在汇总 `sd_hopper_100000_v5densepose_seed{1,2,3}_20260426_run1` 的 `25k eval` 轨迹并复核代码后，确认当前 dense pose 的第一版实现仍存在三类结构性问题：

1. `dense_ground_gate` 直接绑定 `grounded_any`，会让 prone / 贴地失败姿态长期处于 ground 相位；
2. dense reward 仍包含 `Phi_next - Phi_prev + abs(Phi_next)` 的占有奖励成分，且缺少 `gamma` 折扣；
3. `_dominant_state_ids()` 仍把语义不明确的 grounded 失败帧强行回填到 `rear/prone`，污染 milestone 统计。

据此，本轮进一步做了以下修订：

1. `dense_ground_gate_raw` 从 `grounded_any` 收紧为 `foot_grounded * (1 - body_collision)`；
2. `landing` dense 分数显式乘 `no_body_collision * stable_omega_gate`；
3. `ground` dense 分数显式乘 `full_foot_support_gate * no_body_collision`；
4. `rise_pose_score` 内部去掉 `vx` 残差，前向性只保留在外层 `forward_progress`；
5. `dense_delta` 从 `Phi_next - Phi_prev` 改为 `gamma * Phi_next - Phi_prev`；
6. 默认 `dense_pose_abs_weight` 从 `0.05` 下调为 `0.00`；
7. `_dominant_state_ids()` 取消 `grounded_fallback -> rear/prone` 的强制回填；未满足六主状态严格判据的 grounded 失败帧，内部保留 `-1`。

本轮之后，dense pose 的目标重新收束为：

1. prone / body-drag 失败姿态不再通过 `ground` 相位持续获得 dense 正收益；
2. dense 主信号回到“朝合法目标姿态靠近”的差分项；
3. milestone 统计不再被 fallback 回填污染，后续轨迹分析可以更直接地区分“真 rear/prone”与“未命名失败态”。

## 补充修订：修复默认配置与闭环状态回归，并加入 recover penalty

在对新一轮代码修订做二次审查后，又补了三处问题：

1. 默认配置中的 `sd_struct_dim` 仍残留为 `9`，与当前 `STATE_DIM=17` 的硬约束不一致；
2. 上一轮将 `_dominant_state_ids()` 改成默认 `-1` 后，重新引入了“无标签空洞”，会让 `online_trainer` 的六状态统计列失真，也会切断 milestone 恢复链路；
3. 即使 dense ground 已被收紧，失败着地姿态仍可能落入 `dense=0, milestone=0, env_reward≈0` 的平台区，缺少“尝试恢复优于原地不动”的弱驱动。

据此，本轮进一步修订为：

1. `external/tdmpc2/tdmpc2/config.yaml` 的默认 `sd_struct_dim` 改回 `17`；
2. `_dominant_state_ids()` 恢复闭环六状态：
   - grounded fallback 按 `pitch` 符号回填到 `rear/prone`
   - airborne fallback 回填到 `falling_air`
3. 在 `predict_shaping_reward()` 里加入小尺度 `recover_penalty`：

```text
recover_mask = body_collision * (1 - airborne_true)
recover_penalty = recover_mask * (0.20 * |pitch| + 0.05 * |pitch_rate|)
total -= recover_penalty_scale * recover_penalty
```

默认 `recover_penalty_scale` 在初版中设为 `0.05`，其作用不是手写恢复控制，而只是让 prone / rear 静止停留不再成为 0 奖励平台。

关于 `airborne fallback -> falling_air`，这里补记一条与早先“按 `vz` 符号区分 falling / rising”口径的差异说明：

1. 严格的 `rising_air` 仍只在 `airborne_true=1` 且 `vz > 0.05` 时成立；
2. 但对未命中 `falling/rising` 严格判据的空中剩余帧，当前实现统一回填到 `falling_air`，而不是再按 `vz` 非负直接回填到 `rising_air`；
3. 这样做是有意收紧：其目标是避免语义不清的空中过渡帧被误记为 `rising_air`，从而错误触发 `grounded_prepared -> rising_air` 这类 milestone；
4. 因此，这一条不应理解为回归，而应理解为“为保证 milestone 语义保守正确，对空中 fallback 做的单侧收缩”。

## 补充修订：提高 `recover_penalty_scale` 到 0.10

在首轮 `5k smoke` 中，`dense_ground_gate` 漏洞已被堵住，且 `reward_shaping_mean` 已稳定为负，但 `5000 eval` 的高层状态仍主要停留在：

1. `grounded_rear_frac ≈ 0.59`
2. `grounded_prone_frac ≈ 0.39`
3. `grounded_upright_frac ≈ 0.0002`

这说明当前 `recover_penalty_scale = 0.05` 虽已打破“失败姿态 = 0 reward”平台，但对失败着地带的驱离力度仍偏弱。

据此，本轮将：

```text
sd_recover_penalty_scale: 0.05 -> 0.10
```

并先重跑一轮 `5k smoke`。此处先取区间下界 `0.10`，而不是直接升到 `0.15`，目的是避免过强惩罚在超小预算阶段反向压制刚出现的恢复尝试。

## 补充修订：`rear/prone` 改为整体前后几何判据

在复核 `100k seed3` 的高分轨迹时，发现大量被标成 `grounded_rear` 的片段，并不是“整体后仰”，而是从完全伏地抬起上半身时出现的局部主躯干后仰。

进一步对照 `episode_csv` 可见：

1. 高分轨迹中的这些片段，`nose_x` 仍明显位于 `foot_x` 前方；
2. 真正的后仰失败轨迹中，`nose_x` 会落到 `foot_x` 之后；
3. 因而，单用 `pitch` 或 grounded fallback 的 `pitch` 符号来区分 `rear/prone`，会把“前伏抬身”误记为 `rear`。

据此，本轮将 `rear/prone` 的判据从“主躯干 pitch 主导”改为“整体前后几何主导”：

1. `_kinematic_features()` 补充导出：
   - `nose_x/nose_z`
   - `x_min/x_max`
2. `_state_reward_features()` 补充导出：
   - `nose_minus_foot_x = nose_x - foot_x`
   - `nose_minus_support_x = nose_x - p_contact_x`
   - `hip_minus_support_x = hip_x - p_contact_x`
3. `grounded_rear` 改为：
   - 已着地；
   - 不属于 `upright/prepared`；
   - `nose_minus_foot_x < 0`
4. `grounded_prone` 改为：
   - 已着地；
   - 不属于 `upright/prepared`；
   - `nose_minus_foot_x >= 0`
5. `_dominant_state_ids()` 中 grounded fallback 也同步改为按 `nose_minus_foot_x` 的符号回填，而不再按 `pitch` 符号回填。

这次修订的目标是：

1. `rear` 表示真正意义上的整体后仰失败；
2. `prone` 覆盖前伏与前伏抬身阶段；
3. `upright` 与 `prepared` 继续保留足部充分支撑、无身体碰撞、主躯干近竖直等严格条件，不在本轮重写。

## 补充修订：建立奖励/状态回归测试目录

为避免后续继续凭口头记忆反复核对真假 `rear`、关键状态迁移与 shaping 是否被误改，本轮新增独立测试目录：

```text
experiments/hopper_reward_regression/
```

其中包含：

1. `reward_regression_suite.py`：从真实 `episode_csv` 重建状态，并直接调用当前 `DMControlHopperSDModel` 重算硬状态与 shaping reward；
2. `run_reward_regression.py`：执行回归并写出 JSON 报告；
3. `test_reward_regression.py`：测试入口，同时兼容 `pytest` 与标准库 `unittest`；
4. `results/latest_report.json`：最近一次回归结果。

当前首批样例全部来自已有真实轨迹，覆盖以下语义：

1. 旧“假 rear”样例现在应判为 `prone`；
2. 真正整体后仰样例仍应判为 `rear`；
3. 一条短暂直立片段仍应判为 `upright`；
4. `prone -> upright` 迁移应给正 shaping；
5. `falling -> prone` 不应给正 shaping。

执行结果如下：

```text
script regression: 5 / 5 passed
unittest regression: 2 tests passed
```

补记一条实现性说明：

1. 当前 `sdmcts_p100` 环境未安装 `pytest`；
2. 因此测试入口额外兼容 `python -m unittest`，以保证在现有实验环境中无需额外安装依赖也可直接执行；
3. `README.md` 中已同步写明两种运行方式。

## 补充修订：`rear/prone` 改为整体几何中点判据

在 `sd_hopper_10k_v5geomrear_smoke_20260426_run1` 的 `5k eval` 中，`episode_06` 从视频观察看，落地后应几乎全程属于“后仰躺地、腿脚悬空”的失败姿态；但旧实现却把其中一部分帧判成了 `grounded_prone`。

逐帧复核后确认，误判根因在于上一版 `rear/prone` 判据采用了：

```text
nose_minus_foot_x = nose_x - foot_x
```

当足部悬空时，`foot_x` 只代表空中脚部的位置，不能再作为整体姿态的稳定参照，因此会出现：

1. 视频上是明显后仰；
2. 但 `nose_minus_foot_x` 因悬空脚位置变化而变正；
3. 从而被错误判成 `prone`，甚至在 `rear/prone` 之间抖动。

本轮比较了多种候选参照后，最终采用更贴近“整体姿态”的判据：

```text
posture_anchor_x = 0.5 * (x_min + x_max)
nose_minus_anchor_x = nose_x - posture_anchor_x
```

并据此定义：

```text
rear  : nose_minus_anchor_x < 0
prone : nose_minus_anchor_x >= 0
```

采用该判据的理由是：

1. 它直接使用当前全身几何在前后方向上的整体范围；
2. 对“腿脚悬空、身体触地”的后仰失败态更稳，不再依赖脚部位置；
3. 对先前已修正的“前伏抬身假 rear”样例仍保持正确；
4. 对真正后仰样例和直立样例也未造成回退。

修订后对真实轨迹的快速复核结果为：

1. `episode_06` 逐帧统计由“`rear + prone` 混合”变为：
   - `falling_air = 11`
   - `grounded_rear = 489`
   - `grounded_prone = 0`
2. 真假 `rear` 与 `upright` 的既有回归样例仍全部通过；
3. 新增两条 `episode_06` 回归样例后，整套回归扩展为 `7/7 passed`。

此次同时把以下诊断量写入回归报告，便于后续复核：

1. `posture_anchor_x`
2. `nose_minus_anchor_x`
3. `body_contact_x`

其中 `body_contact_x` 目前仅保留为诊断量，不作为最终 `rear/prone` 判据。

## 补充修订：移除 `upright` 的角速度门

在继续复核 `upright` 判据时，重新对照了当前设计目标与奖励语义。现有实现里，`upright` 同时承担两层作用：

1. 作为高层状态标签中的“站立接地态”；
2. 作为 `rear/prone -> upright` 恢复奖励的目标状态。

此前把 `|pitch_rate| <= 3.5` 同时加在：

1. `upright_quality_gate`
2. 最终硬状态 `upright`

其直接来源并不是某次轨迹中已经确认的具体错误，而是早期根据“避免把剧烈翻转中的瞬时接地误判为站稳”的保守收紧思路加进去的。

但在当前任务里，这条门会带来一个更直接的问题：

1. 从前倾或后仰失败姿态站起时，常会经过“身体几何已接近竖直，但仍在较快回正”的恢复瞬间；
2. 若这些瞬间被 `|pitch_rate| <= 3.5` 排除在 `upright` 之外，则 `rear/prone -> upright` 的恢复奖励会明显变稀；
3. 这与当前把 `upright` 用作恢复引导目标的设计目的并不一致。

据此，本轮只对 `upright` 做最小修订：

1. 从 `upright_quality_gate` 中移除 `stable_omega_gate`；
2. 从硬状态 `upright` 判据中移除 `|pitch_rate| <= 3.5`；
3. `prepared`、`landing_quality_gate` 与 dense landing 中的角速度约束暂时保持不变。

这样处理的理由是：

1. 直接放宽 `rear/prone -> upright` 的命中，优先缓解恢复奖励过稀；
2. 不同步放松 `prepared/landing`，避免一次性改动过大，把“恢复站起”与“稳定预备/着陆”两类语义混在一起；
3. 便于后续单独观察：放宽 `upright` 后，是否确实会增加恢复类过渡，而不是只增加错误命中。
