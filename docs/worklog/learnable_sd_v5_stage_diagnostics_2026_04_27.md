# 2026-04-27

## Hopper Hop：补充三大阶段单独诊断链

在继续审阅当前 `v5` dense pose 训练结果后，确认仅靠完整 `hopper-hop` 主任务训练，难以直接判断：

1. `landing` 奖励是否真的会把空中下落引向更好的落地姿态；
2. `ground` 奖励是否真的会把失败着地引向 `upright / prepared`；
3. `rise` 奖励是否真的会把地面稳定态引向起跳。

若这些局部引导本身并不成立，那么将三段奖励直接叠加进完整任务，只会进一步混淆失败原因。因此本轮补充“分阶段单独训练”的最小工程闭环。

## 本轮实现口径

### 1. 阶段化 reset 池

新增脚本：

- [build_hopper_stage_reset_pool.py](/home/data/round/code/sd_world_model_rl/scripts/build_hopper_stage_reset_pool.py)

作用：

1. 从既有 `official_eval/step_XXXXXXXX/episode_csv/` 中抽取阶段样本；
2. 直接保存为可复用的 `qpos/qvel` reset 池；
3. 供后续训练环境在 `reset()` 后覆写 MuJoCo 物理状态。

当前三类阶段样本筛选规则为：

1. `landing`：
   `falling_air`，且 `airborne_true=1`、`vz < -0.10`
2. `ground`：
   `grounded_rear / grounded_prone / grounded_upright`
3. `rise`：
   `grounded_prepared`，以及满足高质量门的 `grounded_upright`

这里刻意没有让 `ground` 默认包含 `grounded_prepared`，目的是避免地面阶段测试被“已经接近目标”的 reset 池稀释。

### 2. 环境 reset 包装

在 [dmcontrol.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/envs/dmcontrol.py) 中新增 `HopperStageResetWrapper`。

处理流程为：

1. 环境先按原始 `suite.load(...).reset()` 执行一次标准 reset；
2. 再从阶段池中随机抽取一条样本；
3. 将其 `qpos/qvel` 写回 MuJoCo `physics.data`；
4. `physics.forward()`；
5. 重新读取当前物理状态对应的 observation。

这样训练主循环、评估导出链与已有 TD-MPC2 流程均无需重写。

### 3. 只保留单阶段 shaping

在 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中新增：

- `stage_mode = full / landing / ground / rise`

其含义为：

1. `dense_pose_potential` 只保留对应阶段的相位分量；
2. milestone 只保留对应阶段的迁移项；
3. `recover_penalty` 只在 `ground` 模式下保留。

当前对应关系为：

1. `landing`：
   `dense_down`，
   `falling -> upright`，
   `falling -> prepared`
2. `ground`：
   `dense_ground`，
   `rear -> upright`，
   `prone -> upright`，
   `upright -> prepared`
3. `rise`：
   `dense_rise`，
   `prepared -> rising`

默认仍为 `full`，因此既有主实验口径不变。

### 4. 环境奖励缩放开关

在 [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py) 中新增：

- `sd_env_reward_scale`

作用：

```text
reward_target = sd_env_reward_scale * env_reward + alpha * shaping_reward
```

默认值仍为 `1.0`，因此主实验不变。

阶段诊断脚本中将其设为 `0.0`，目的是把单阶段测试尽量收缩为“只看结构化奖励是否有引导作用”。

### 5. 阶段化 5k smoke 脚本

新增脚本：

- [run_hopper_v5_stage_5k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_stage_5k_smoke.sh)

固定口径：

1. `steps=5000`
2. `eval_freq=5000`
3. `checkpoint_freq=5000`
4. `save_eval_traces=true`
5. `sd_reward_mode=hybrid`
6. `sd_env_reward_scale=0.0`
7. `sd_reward_hybrid_alpha=1.0`
8. `sd_stage_mode=<landing|ground|rise>`
9. `hopper_stage_reset_pool_csv=<pool_csv>`

这样即可直接用现有 `train.py` 跑单阶段 smoke，而不需要手写新的训练主程序。

## 影响评估

### 对主实验的影响

默认配置下：

1. `sd_stage_mode=full`
2. `sd_env_reward_scale=1.0`
3. `hopper_stage_reset_pool_csv=null`

因此：

1. 既有主实验奖励语义保持不变；
2. 环境仍按 Hopper 原始 reset 分布启动；
3. 这套改动只在显式启用阶段诊断时生效。

### 当前限制

当前阶段池仍来自历史 `official_eval` 轨迹，而不是人工设计的理想状态分布。因此它更适合作为“局部可引导性 smoke”，还不能直接替代更严格的受控初始化实验。

下一步更合适的工作是：

1. 先用该链各跑一轮 `landing / ground / rise` 的 `5k smoke`；
2. 分别核对 `episode_csv` 中 dense 分量和阶段命中率是否向目标方向收敛；
3. 若某一阶段单独训练都拉不动，再回到该阶段参照姿态与目标值本身做修订。

## 补充修订：新增 `prone_recover` 专用阶段

在复核 `landing` 单阶段 `5k` 结果后，确认“空中下降 -> 好落地”这条链暂时不适合作为当前结构化先验的核心展示样例。其主要原因不是训练步数不足，而是“落地前好姿态”的参照本身仍不够稳，容易把未形成有效足部支撑的下降帧赋予偏高分。

相比之下，`prone -> upright` 恢复过程具有三个优势：

1. 真实轨迹中样本数量明显更多；
2. 视频语义更清楚，人工核对更直接；
3. 结构化先验是否在显式推动恢复，更容易通过状态分布与视频直接观察。

因此本轮新增独立阶段：

- `sd_stage_mode=prone_recover`

其口径为：

1. reset 池只从已确认的 `grounded_prone` 帧中抽样；
2. 稠密奖励只表达“前伏逐步恢复到直立”的连续改善；
3. 里程碑奖励只保留 `prone -> upright`；
4. 不再混入 `rear -> upright`、`upright -> prepared` 或 `prepared -> rising`。

### reset 池

为避免把“完全趴地”和“已经抬身借脚尝试恢复”混成一类，本轮在阶段池脚本中新增两种筛选：

1. `prone_low`
   - `grounded_prone`
   - `nose_minus_anchor_x >= 0.10`
   - `torso_z <= 0.45`
   - `support_contact < 0.30`
2. `prone_lift`
   - `grounded_prone`
   - `nose_minus_anchor_x >= 0.10`
   - `0.40 < torso_z <= 0.75`
   - `support_contact >= 0.30`

同时阶段池脚本新增 `--trace-csv`，可直接只使用一到两条已人工确认语义的 `episode_csv` 构造 reset 池，而不必再把整个 `episode_csv/` 目录一并喂入。

### 稠密奖励势函数

`prone_recover` 不再复用旧 `ground_pose_score`，而改为独立势函数。继续根据支持加权评审意见修订后，采用明确的双通道结构：

```text
Phi_prone_recover = Phi_unfold + Phi_stand
```

其中：

1. `Phi_unfold = 0.25 * grounded_soft * unfold_score`
2. `Phi_stand = 0.75 * foot_soft * support_gate * body_clear_score * stand_score`
3. `unfold_score = 0.35 * pitch + 0.30 * lift + 0.20 * geom + 0.15 * knee`
4. `stand_score = 0.35 * pitch + 0.20 * lift + 0.15 * knee + 0.15 * torso_over_foot + 0.15 * spin`
5. `support_gate = support_score^2`
6. `grounded_soft = sigmoid((0.03 - z_min) / 0.015)`：只负责让前伏起步阶段保留连续信号；
7. `foot_soft = sigmoid((0.03 - foot_min_z) / 0.015)`：将高势值区域显式绑定到足部近地/接地；
8. `body_clear_score = sigmoid((body_min_z - 0.04) / 0.02)`：区分“脚在地上、身体离地”和“脚在地上、身体仍趴地”；
9. `torso_over_foot`：用躯干相对支撑点的水平偏差直接表达“是否站到脚上”；
10. `spin` 只作为站立质量弱项，不再作为整体外层乘性上限。

后续复核中进一步确认：这版恢复奖励的目标应是“静态姿态靠近参考姿态”，而不是“动作过程尽量平缓”。因此 `pitch_rate` 不再进入 `prone_recover` 主势函数；它仍保留为导出诊断量，但不再作为奖励项，以免压制那些起初角速度较大、但更有效的恢复动作。

实际 shaping 仍采用总势差，但占有奖励只挂在 `Phi_stand` 上：

```text
r_dense =
0.40 * (0.995 * Phi(next) - Phi(prev))
+ 0.04 * Phi_stand(next)
```

这样做的目的有两层：
1. `Phi_unfold` 仍负责“愿意起步尝试恢复”；
2. 只有 `Phi_stand` 会带来持续占有收益，从而把“站到脚上”塑造成高势值和持续收益的必要条件。

## 补充修订：`prone_recover v4` 评估回读后的两处修补

在 `sd_hopper_5k_pronerecover_pure_pronelow_v4_seed2_20260427_live` 的 `5k` 评估完成后，出现了两个新的实证问题：

1. `official_eval/episode_csv` 中 `dense_prone_recover_component` 明显非零，但 `dense_pose_potential` 几乎恒为 `0`；

# 2026-04-28

## Hopper SD：从奖励线切回纯 dynamics 替代线

在多轮 Hopper 奖励设计尝试后，正式停止沿 shaping / milestone / action prior 继续做主实验优化，改回“纯 SD dynamics 替代 TD-MPC2 world model”的干净验证线。新的工作目标收缩为三项：

1. 明确 `sd_hopper` 在 TD-MPC2 中到底接管了哪些接口；
2. 判断当前 17 维结构状态是否具备支撑 `encode / next` 的最小能力；
3. 找出根部动力学中最可能阻碍规划的结构性缺口。

为此新增并落地以下内容：

1. 纯环境奖励、无 residual 的 Hopper 基线脚本：
   [run_hopper_v5_10k_pureenv_noresid.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_pureenv_noresid.sh)
2. 独立监督拟合脚本：
   [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py)
3. world-model 能力总审计：
   [sd_hopper_world_model_capability_audit.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_world_model_capability_audit.md)
4. `vx / pitch_rate` 专项审查：
   [sd_hopper_vx_pitchrate_model_audit.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_vx_pitchrate_model_audit.md)

## Hopper SD：两轮根部动作作用链修补

基于动作敏感性诊断，确认旧版本存在明确的结构性退化：

1. 对每个固定样本状态，24 个动作给出的 `sd_delta_vx` 唯一值个数恒为 `1`；
2. 同样地，`sd_delta_pitch_rate` 唯一值个数也恒为 `1`；
3. 这说明旧版本中 `action` 对根部前向速度与俯仰角速度的一步影响在方程上被切断。

### 第一轮修补

在 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中完成：

1. `F_leg / F_x / F_z` 改为使用动作更新后的 `next_q`、`next_qdot`、`next_activation`；
2. `next_omega` 中加入 grounded 接触力矩 `tau_contact`。

结果：

1. grounded 样本恢复了明显的动作区分；
2. 但 airborne 样本仍大面积保持 `vx / pitch_rate` 同值塌缩。

### 第二轮修补

在不新增可学习参数、避免旧 checkpoint strict load 断裂的前提下，继续补入：

1. 小幅的空中根部 `vx` 动作耦合先验；
2. 小幅的空中根部 `omega` 动作耦合先验。

结果见：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_sensitivity/geommid_seed3_step50000_after_fix2/summary.json)

修复前后最关键的结构变化为：

1. 修复前：
   `vx_unique_hist = {1: 20}`，`pitch_rate_unique_hist = {1: 20}`
2. 第一轮后：
   `vx_unique_hist = {1: 12, 2: 3, 24: 5}`，`pitch_rate_unique_hist = {1: 14, 2: 1, 15: 1, 20: 1, 23: 1, 24: 2}`
3. 第二轮后：
   `vx_unique_hist = {22: 13, 23: 2, 24: 5}`，`pitch_rate_unique_hist = {24: 20}`

这说明：

> 当前最核心的结构性问题已经从“动作作用链缺失”修复为“动作作用链已接通，但定量精度仍待校准”。

## Hopper SD：监督拟合作为修复前后基线

修复前已完成一轮 GPU 监督拟合，结果位于：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1/summary.json)

修复后重新从头拟合，结果位于：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_after_fix2/summary.json)

修复后结果并未同步变好，关键指标为：

1. `one_step_struct_rmse`：`0.6318 -> 0.6409`
2. `rollout_5_rmse`：`1.1015 -> 1.1402`
3. `rollout_10_rmse`：`1.1860 -> 1.2461`
4. `vx_mae`：`0.0727 -> 0.0944`

这轮结果的解释应严格区分：

1. 修复前拟合结果仍有价值，但只能作为“结构修补前基线”；
2. 修复后拟合结果才代表当前模型；
3. 当前瓶颈已经转为“系数与标定质量不足”，而不是“接口结构本身不存在”。

## Hopper SD：第一轮监督校准

在结构修补完成后，继续审查监督拟合脚本，确认旧版训练目标本身存在两处直接问题：

1. `omega` 与 `activation` 被误放进 angle wrapping loss；
2. 默认 `rollout_weight=0.0`，实际上只在做一步监督，没有任何多步训练约束。

因此本轮对 [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py) 做了最小但有针对性的修订：

1. 仅保留 `theta, waist, hip, knee, ankle` 为环绕角度通道；
2. 显式提高 `vx` 与 `omega` 的损失权重；
3. 将 `activation` 改为弱监督项；
4. 默认启用 `rollout_horizon=5`、`rollout_weight=0.25`；
5. 新增 `pitch_rate_mae` 导出，避免继续只看 `pitch_mae`。

修订后重跑结果位于：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_after_fix2_calib1/summary.json)

以统一数据集回算的关键对比如下：

1. 修复前基线：
   `pitch_rate_mae = 0.90249`，`vx_mae = 0.10061`
2. 结构修补后：
   `pitch_rate_mae = 0.88123`，`vx_mae = 0.09437`
3. 第一轮校准后：
   `pitch_rate_mae = 0.86727`，`vx_mae = 0.08113`

同时，多步 rollout 指标也出现改善：

1. `rollout_10_rmse`：`1.24607 -> 1.17632`
2. `rollout_20_rmse`：`1.21126 -> 1.08886`

但这轮也暴露出新的权衡：

1. `one_step_struct_rmse` 略升；
2. `vz_mae` 恶化到 `0.39966`；
3. 加入 rollout 训练后，单次拟合耗时显著上升。

据此，下一轮应优先处理两件事：

1. 继续分离根部关键量与总体 RMSE 的优化目标；
2. 修整 rollout 训练实现的效率，否则监督迭代成本会过高。
2. 十条评估轨迹几乎全部停留在 `grounded_prone`，`grounded_upright_frac=0`、`full_foot_support_frac=0`，说明 `stand` 通道在当前 `5k` 预算下仍过于难以进入。

复查代码后确认：

1. 训练时 world model 的 `sd_stage_mode=prone_recover` 已正确生效；
2. 但 `online_trainer.py` 在导出 `official_eval` 轨迹时，单独新建了一个默认 `stage_mode=full` 的 `DMControlHopperSDModel()` probe；
3. 因此 `episode_csv` 中的 `dense_pose_potential` 实际上并不对应当前训练正在使用的 `prone_recover` 分支，而是默认 `full` 口径；
4. 同时，当前 `Phi_stand = 0.75 * foot_soft * support_score^2 * body_clear_score * stand_score` 对“单足承重、身体开始离地但尚未整足支撑”的中间恢复态压制过强。

据此做出两项最小修补：

### 1. 修正 `official_eval` 的阶段 probe 口径

在 `online_trainer._hopper_trace_model()` 中，不再使用默认构造，而改为：

```text
DMControlHopperSDModel(stage_mode=cfg.sd_stage_mode)
```

这样导出的 `episode_csv` 将与训练时实际生效的 stage 分支一致。

出于同样原因，`scripts/export_hopper_eval_traces.py` 也新增了 `--sd_stage_mode` 参数，便于离线复核时保持同一奖励语义。

### 2. 适度放宽 `Phi_stand` 激活门

保持双通道结构和“高势值必须依赖足部支撑”的原则不变，仅放宽门函数强度：

1. `foot_soft` 的尺度从 `0.015` 调整为 `0.03`
2. `body_clear_score = sigmoid((body_min_z - 0.02) / 0.03)`
   原先为 `sigmoid((body_min_z - 0.04) / 0.02)`
3. `support_gate` 从纯平方门
   `support_score^2`
   调整为
   `support_score * (0.35 + 0.65 * support_score)`

这样做的意图是：

1. 保留“整足支撑优于单足支撑”的排序；
2. 但不再把“前掌或后掌已经明确承重、身体也开始离地”的中间恢复态压得接近零；
3. 让 `5k ~ 10k` 这种小预算实验中，`Phi_stand` 至少能够对“往脚上转移支撑”的尝试给出可见信号；

# 2026-04-28

## Hopper：转向纯 world-model 验证线后的两项入口修补

在确认 Hopper 主线不再继续围绕 shaping 设计迭代后，本轮把工作重点收缩为两条更干净的验证线：

1. 纯环境奖励、无 residual latent 的 TD-MPC2 Hopper 基线；
2. 脱离 TD-MPC2 闭环的 Hopper SD 独立监督拟合。

相应新增入口分别为：

1. [run_hopper_v5_10k_pureenv_noresid.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_pureenv_noresid.sh)
2. [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py)

在对第二个入口做静态和最小运行校验时，又发现并修补了两个会直接污染拟合结论的实现问题。

### 1. 多步 rollout 不能建立在随机打散的转移上

初版 `fit_hopper_sd_supervised.py` 虽然导出了 `rollout_5_rmse / rollout_10_rmse`，但训练 / 验证切分仍按单步转移随机抽样。这样一来：

1. 验证集中的相邻样本不再保证来自同一时间连续轨迹；
2. 训练中若直接在 mini-batch 内拼多步 rollout，也会把不连续样本误当作时间序列。

这会使多步误差指标失去语义，也会让可选的 rollout loss 变成错误监督。

本轮改为：

1. 数据集显式记录 `episode_id`；
2. 训练 / 验证按 `episode` 切分；
3. `valid_rollout_starts()` 只在同一条 episode 内筛选连续起点；
4. 若启用 `rollout_weight > 0`，训练阶段的多步损失也只从这些连续起点构造。

这样导出的 `rollout_5_rmse / rollout_10_rmse` 才真正表示多步时间展开误差，而不是随机转移上的伪指标。

### 2. 独立拟合脚本的默认 MuJoCo 后端需与当前机器环境一致

初版脚本为了避免 `DISPLAY` 依赖，默认把 `MUJOCO_GL` 设为 `egl`。实际在本机环境中：

1. `egl` 无法初始化 headless display；
2. `osmesa` 依赖也不完整；
3. `glfw` 虽然会打印 `DISPLAY` 缺失警告，但 `dm_control` 的导入、reset 与 step 都能正常执行。

因此本轮将监督拟合脚本的默认后端改回 `MUJOCO_GL=glfw`，并把 `dm_control` 延迟到数据采样阶段再导入。这样：

1. `python scripts/fit_hopper_sd_supervised.py --help` 可直接使用；
2. 最小 smoke 能在当前环境中完整走通；
3. 入口行为与现有 Hopper 训练脚本保持一致。

### 3. 当前校验结果

已完成以下检查：

1. `bash -n scripts/run_hopper_v5_10k_pureenv_noresid.sh`
2. `python -m py_compile scripts/fit_hopper_sd_supervised.py`
3. `python scripts/fit_hopper_sd_supervised.py --help`
4. 极小规模 smoke：
   `python scripts/fit_hopper_sd_supervised.py --seed 0 --episodes 2 --max_steps 5 --epochs 1 --batch_size 4 --device cpu --out_dir /tmp/hopper_sd_supervised_smoke`

其中第 4 项已成功产出 `summary.json / history.json / hopper_sd_supervised.pt`，说明：

1. 随机 Hopper 数据采样可用；
2. 结构状态编码与一步前向传播可用；
3. 训练循环、指标汇总与结果落盘链路可用。

## 2026-04-28

### Hopper：补做“SD 是否具备 TD-MPC2 所需接口能力”的专门审计

在停止继续讨论 shaping 线后，本轮补做了一份只面向 world-model 能力本身的审计文档：

- [sd_hopper_world_model_capability_audit.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_world_model_capability_audit.md)

该文档只回答一个问题：

> 当前 `sd_hopper` 设计，是否具备为 TD-MPC2 提供 `encode(obs) -> z` 与 `next(z, a) -> z'` 两条核心接口所需的能力。

本轮给出的主要结论有三点。

#### 1. 当前替换范围已经是真正接管了 planner / trainer 的状态接口

当 `world_model_kind=sd_hopper` 时：

1. `TDMPC2` 不再实例化原始黑盒 `WorldModel`，而是实例化 `SDWorldModel`；
2. 规划 rollout 与训练 consistency rollout 都会直接调用 `SDWorldModel.encode()` 与 `SDWorldModel.next()`；
3. 当前仍未替换的是 learned `reward / Q / pi` 头。

也就是说，当前 Hopper 线真正被替代的是：

1. `encode(obs) -> z`
2. `next(z, a) -> z'`

而不是整套 `reward / value / policy`。

#### 2. 17 维结构状态总体上是合理候选，但不是“已证明闭合”的 state

审计后确认：

1. `rootx` 的省略在 Hopper-Hop 中是合理的，不构成主要缺陷；
2. `z_min` 是几何冗余派生量，不是主要风险；
3. 当前最大的闭合性风险来自 `A`：
   它一方面由当前观测启发式估计，另一方面又在 rollout 中被当作递推内部状态。

因此，更准确的结论不是“17 维状态已闭合”，而是：

> 该状态覆盖了主要运动学、速度与接触信息，可视为一个近似 Markov state 候选；但其闭合性仍需要通过多步 rollout 与 planner 分布下的验证来确认。

#### 3. 独立监督拟合脚本当前只是必要条件验证，不是充分条件验证

当前脚本已经能够验证：

1. 随机动作分布下的一步拟合；
2. 同一 episode 连续片段上的 `1 / 5 / 10` 步 rollout；
3. 关键结构量误差。

但它还不能回答：

1. planner 采样分布上的稳定性；
2. `20+` 步 horizon 漂移；
3. 动作敏感性是否与环境一致；
4. `reward / Q / pi` 头是否能在 `17` 维结构 latent 上稳定学习。

因此，文档中的总判断被明确写为：

> 当前 Hopper SD 设计已经具备“作为 TD-MPC2 world-model 接口候选实现”的基本形式条件，但尚没有足够证据证明它已经具备“稳定替代黑盒 world model”的能力条件。

后续实验顺序也据此收缩为：

1. 独立监督拟合补齐 `20` 步 rollout；
2. 增加动作敏感性测试；
3. 在纯环境奖励、无 residual 的闭环实验中回看 `struct_consistency_loss`；
4. 若独立拟合良好但闭环仍差，再检查 `reward / Q / pi` 头的结构 latent 适配性。

### Hopper：补齐最小验证工具

为把上述四步从“建议”落成可执行入口，本轮又补了两项工具。

#### 1. 独立监督拟合新增 `rollout_20_rmse`

在 [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py) 中，评估指标由：

1. `rollout_1_rmse`
2. `rollout_5_rmse`
3. `rollout_10_rmse`

扩展为：

4. `rollout_20_rmse`

这样可以直接观察：

1. 误差是否仅在 `1 -> 5` 步快速放大；
2. 还是到 `10 -> 20` 步仍持续恶化；
3. 从而判断当前 SD 更像“局部一步近似器”还是“可承受 planner 中短视野 rollout 的动力学核”。

#### 2. 新增动作敏感性诊断脚本

新增脚本：

- [analyze_hopper_sd_action_sensitivity.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_sd_action_sensitivity.py)

该脚本固定读取真实 `official_eval/episode_csv` 中的若干状态，并对每个状态施加同一组动作 bank，然后分别：

1. 用真实 DMControl Hopper 环境推进一步；
2. 用当前 `DMControlHopperSDModel` 推进一步；
3. 比较两者在以下关键量上的一步响应：
   `height / vx / vz / pitch / pitch_rate / support_contact / full_foot_support / body_contact_z / z_min`

导出结果分为：

1. `action_sensitivity_details.csv`：逐状态、逐动作的细粒度响应对照；
2. `summary.json`：各关键量的平均绝对误差与方向一致率。

这样可以专门回答此前审计中尚未闭合的一个问题：

> 即便 SD 一步拟合误差不高，它对动作改变的响应方向与幅度，是否与真实环境一致。

这一步对于 TD-MPC2 尤其关键，因为 planner 依赖的不是“静态状态识别”，而是“动作一改，未来轨迹如何变化”。

### 最小运行校验

新增工具已完成最小 smoke：

```bash
python scripts/analyze_hopper_sd_action_sensitivity.py \
  --trace-csv logs/hopper-hop/3/sd_hopper_5k_v5minimal_smoke_20260425_run1/official_eval/step_0005000/episode_csv/episode_09.csv \
  --samples-per-trace 1 \
  --amplitudes 0.25 \
  --out-dir /tmp/hopper_sd_action_sensitivity_smoke
```

结果已成功落盘 `action_sensitivity_details.csv` 与 `summary.json`。

这次 smoke 也给出了第一条真正有诊断价值的信号：

1. `height / vz / body_contact_z / z_min` 的一步响应方向在该样本上与环境基本一致；
2. 但 `vx` 的 `dir_match_mean = 0.0`；
3. `pitch_rate` 的 `dir_match_mean = 0.5`。

这说明新脚本不是“只验证能不能跑”，而是真的能够把“SD 对动作响应的方向是否对”这个问题显式量化出来。后续正式动作敏感性实验应优先关注：

1. `vx`
2. `pitch_rate`
3. `pitch`

这三类量，因为它们最可能直接影响 planner 对 hop / recover 行为的短视 rollout 判断。

### Hopper：`vx / pitch_rate` 建模专项审查

在动作敏感性工具跑完 `step_0050000` 的正式批量比较后，又补做了一份只针对根部动力学关键薄弱点的专项审查文档：

- [sd_hopper_vx_pitchrate_model_audit.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_vx_pitchrate_model_audit.md)

这轮审查最重要的结论不是“某些量误差较大”，而是确认了一个更硬的结构性问题：

> 当前 `action` 对 `next_vx` 与 `next_omega` 在同一步内几乎没有直接作用。

代码层面，根因是：

1. `next_vx / next_vz` 使用的 `F_leg` 仍由当前 `activation`、当前 `kin`、当前 `touch` 计算，而不是动作更新后的量；
2. `next_omega` 完全由
   `tau_bal = -balance_theta * theta - balance_omega * omega`
   决定，不依赖 `action`，也不依赖接触力矩。

因此，在同一初始状态下，即便动作符号翻转：

1. 真实环境中的 `env_delta_pitch_rate` 会随之翻转；
2. 但 SD 的 `sd_delta_pitch_rate` 仍保持相同；
3. `sd_delta_vx` 也在样本内对所有动作保持常数。

用动作敏感性明细进一步统计后可见：

1. 对每个固定样本状态，24 个动作对应的 `sd_delta_vx` 唯一值个数恒为 `1`；
2. 同样地，`sd_delta_pitch_rate` 与 `sd_delta_vz` 的唯一值个数也恒为 `1`。

这说明当前问题不是“参数尚未拟合好”，而是：

> 更新方程本身把动作对根部动力学的同一步影响延后到了下一步。

据此，本轮将问题优先级重新排序为：

1. P0：先修“动作未进入同一步 `vx / vz / omega` 更新”
2. P1：再看是否需要补切向接触 / 摩擦状态
3. P2：最后才讨论裁剪或其他次要数值问题

这一结论对后续修改方向的影响很直接：

1. 继续单纯增加训练预算无法解决该问题；
2. 必须先修正 `action -> root dynamics` 的结构作用链；
3. 修正后再重新跑动作敏感性与独立监督拟合，才有意义继续判断 planner 接入效果。
4. 仍然避免把纯贴地 prone 姿态误抬成高 `stand` 势值。

### 3. 新增统一 `10k` 阶段化脚本

为避免再次手写长命令并确保后续如需严格接续到 `20k` 时语义一致，本轮新增：

- [run_hopper_v5_stage_10k_smoke.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_stage_10k_smoke.sh)

固定口径为：

1. `steps=10000`
2. `eval_freq=5000`
3. `checkpoint_freq=5000`
4. `sd_stage_mode=<landing|ground|rise|prone_recover>`
5. `sd_env_reward_scale=0.0`
6. `sd_reward_hybrid_alpha=1.0`

后续若要判断“训练不足”还是“奖励过严”，统一使用该脚本，不再手写命令。

## 补充修订：`v5relaxed 10k` 后继续强化 `stand`，并引入 prone 稳态负向项

在 `sd_hopper_10k_pronerecover_pure_pronelow_v5relaxed_seed2_20260427_run1` 完整结束后，结果显示：

1. `5000` 与 `10000` 两个评估点的平均 `episode_reward` 仍为 `0.0`；
2. `10000` 步 `official_eval` 中，10 条轨迹全部仍为 `grounded_prone_frac=1.0`，`grounded_upright_frac=0.0`，`full_foot_support_frac=0.0`；
3. 与 `v4` 相比，`stand_component` 的确被放开，但仍不足以把策略从 prone 稳态中拉出。

离线按新代码回灌 `step_0010000` 的 `episode_csv` 后，得到更具体的诊断：

1. `mean_stand_component` 已可达约 `0.0043`，相比 `v4 5k` 的约 `0.0011` 明显提高；
2. 但 `mean_unfold_component` 仍稳定在约 `0.058` 左右；
3. 也就是说，策略确实更容易进入“有一些脚上承重/抬身”的中间态，但高势值区依旧没有足够强地偏向“继续把支撑转移到脚上并保持”；
4. 同时，长期保持 prone 仍几乎不需要付出代价，因而形成了稳定零分盆地。

据此，本轮继续做两项最小结构修补：

### 1. 进一步提高单足承重在 `stand` 分支中的可见性

此前 `support_score = 0.5 * support_contact + 0.5 * full_foot_support`。

现在改为：

```text
support_score = 0.75 * support_contact + 0.25 * full_foot_support
```

其目的不是放弃“整足支撑优于单足支撑”，而是：

1. 让“前掌或后掌已经明确承重，但另一侧尚未形成支撑”的中间恢复态，不再因为 `full_foot_support=0` 而被压得过低；
2. 保留 `full_foot_support=1` 时最高得分仍优于单足承重；
3. 让 `stand` 分支更接近“从无支撑到单足承重，再到更稳定支撑”的渐进过程。

### 2. 降低 `unfold` 上限、提高 `stand` 占有收益

本轮将：

1. `PRONE_RECOVER_UNFOLD_CAP` 从 `0.25` 下调至 `0.15`
2. `PRONE_RECOVER_STAND_ABS_WEIGHT` 从 `0.04` 上调至 `0.08`

这样做的意图是：

1. `unfold` 仍负责让策略愿意起步、抬身、展开；
2. 但策略若只停留在“抬身/展开/前移”而不继续站到脚上，可占据的高势值空间明显缩小；
3. 与之相对，真正进入 `stand` 分支后，持续占有收益翻倍，更有动力维持脚上承重恢复态。

### 3. 新增 `prone_stagnation` 负向静态惩罚

此前 `prone_recover` 阶段只有正向势差与 `stand` 占有项，缺少对“长期趴着不动”的明确负向信号。

本轮新增：

```text
prone_stagnation =
    grounded_prone
    * low_body
    * (1 - body_clear_score)
    * (1 - support_score)
```

并在 `prone_recover` 阶段中额外减去：

```text
0.03 * prone_stagnation(next)
```

该项只依赖下一时刻静态姿态，而不依赖角速度或姿态变化过程，目的是：

1. 惩罚“身体低、身体仍贴地、脚上几乎没有支撑”的稳定 prone 盆地；
2. 对已经抬身、身体开始离地、脚上已有承重的尝试，惩罚自动减弱；
3. 避免再次回到“reward 全靠正激励，但 agent 选择原地贴地零分稳态”的局面。

为便于后续核对，该诊断量也一并导出到 `episode_csv`：

- `prone_recover_prone_stagnation`

### 4. 离线回灌结论

用 `v5relaxed 10k` 的 `official_eval/step_0010000/episode_csv` 重新按新公式回灌后，得到：

1. `agg_mean_stand` 约 `0.0090`
2. `agg_mean_stagnation` 约 `0.488`
3. 有明显脚上承重的几条轨迹，`mean_stand` 已可达 `0.016 ~ 0.018`
4. 纯低支撑 prone 轨迹，`mean_stand` 仍接近 `0`

因此，这一版修订满足当前工程目标：

1. 能显式区分“部分恢复尝试”与“贴地 prone 稳态”；
2. 没有重新把纯贴地低质量姿态抬成高 `stand` 势值；
3. 奖励语义已经再次变化，后续验证必须重新从头训练，不可接续 `v5relaxed` 的 checkpoint。

## 补充修订：按第一性原理将 `prone_recover` 主奖励收缩为 `progress + core`

在 `v6pushstand` 的轨迹复盘后，确认继续沿着 `rear`、`prone_stagnation` 等失败路径逐项修补，会让奖励定义越来越复杂，也越来越难以确认“当前到底在奖励什么”。因此本轮回到最直接的物理目标，只保留：

1. 身体主要重量回到脚上；
2. 非足部身体部分离地；
3. 躯干明显抬高；
4. 躯干大致竖直。

据此，将 `prone_recover` 势函数进一步收缩为：

```text
progress =
    0.40 * support_contact
    + 0.30 * body_clear_score
    + 0.30 * lift_score

core =
    full_foot_support
    * body_clear_score
    * pitch_score
    * lift_score
```

总势函数为：

```text
Phi =
    0.20 * grounded_soft * progress
    + 0.80 * core
```

对应 shaping 为：

```text
r_dense =
    0.40 * (0.995 * Phi(next) - Phi(prev))
    + 0.05 * core(next)
```

这里不再单独惩罚 `rear`，也不再单独惩罚 `prone_stagnation`。理由是：

1. `rear` 本质上只是另一种失败态，不应靠新增一条专门惩罚去定义；
2. 只要目标态本身定义得足够清楚，`rear` 和“坐地但双脚着地”都会因为 `body_clear / lift / pitch / full_foot_support` 不满足而自然拿不到高分；
3. 奖励应优先表达“最终想要什么状态”，而不是“列举所有不想要的错误路径”。

### 保留与删除

保留的主奖励量：

1. `support_contact`
2. `full_foot_support`
3. `body_clear_score`
4. `lift_score`
5. `pitch_score`

保留但仅作为诊断导出的量：

1. `geom`
2. `knee`
3. `spin`
4. `torso_over_foot`

移出主奖励路径的量：

1. `prone_stagnation` 负向惩罚
2. 通过 `rear`、`support_gate`、`torso_over_foot` 等失败路径修补项主导奖励的做法

### 离线核对

用 `v6pushstand` 的 `step_0010000` 轨迹按新公式回灌后：

1. 10 条轨迹的 `mean_core_component` 全部为 `0`
2. 说明这些轨迹虽然存在脚部接触、抬身和前后翻转，但没有任何一条真正满足“整足支撑 + 身体离地 + 竖直 + 高度足够”的目标态

进一步检查 `step_0005000` 中那条 `max_full_foot_support = 0.986` 的轨迹帧后，发现：

1. 该帧 `core_component` 仅约 `2.6e-4`
2. `body_clear_score`、`pitch_score`、`lift_score` 都很低

这说明新的 `core` 定义没有把“坐地但双脚接触”误判成高分目标态。

因此，这一轮简化后的奖励设计更符合当前原则：

1. 高分只来自清晰目标态；
2. 过程性改善只拿较弱 `progress` 信号；
3. 不再通过不断堆加失败路径惩罚来修补行为。
## 2026-04-27

新增了阶段化 `100k` 稳定启动脚本 [run_hopper_v5_stage_100k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_stage_100k.sh)，并在 [HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md](/home/data/round/code/sd_world_model_rl/docs/HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md) 中登记该入口。

这样处理的原因是：当前 `prone_recover` 的 `10k` 验证已出现“足部充分支撑但仍以头顶支地”的过渡态，说明势函数方向尚未完全失效；若继续判断是否值得拉长预算，必须保持与阶段 `5k / 10k` 完全相同的奖励语义，而现有 [run_hopper_v5_100k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k.sh) 固定回到主线 `curriculum` 奖励，不能直接用于阶段长测。

因此，本轮将长测入口显式拆分，避免再次手写训练命令或误把主线奖励与阶段奖励混用。后续所有 `prone_recover` 长实验均应走该脚本，以保证 reset 池、纯 shaping 监督与阶段语义一致。

## 2026-04-28

在 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py) 的模型更新日志中新增了 `struct_consistency_loss` 与 `resid_consistency_loss` 两个诊断量。

新增这两列的原因是：现有 `consistency_loss` 只反映完整 latent 的一步 rollout 误差，而当前 `SDWorldModel` 的 latent 由前 `17` 维结构状态与后续 residual 子空间拼接构成。仅观察总的 `consistency_loss`，无法判断“结构状态推演没有学好”还是“residual 子空间误差较大”。本次补充后，可以分别追踪：

1. `struct_consistency_loss`：显式结构状态 `z_struct` 的一步预测误差；
2. `resid_consistency_loss`：残差子空间 `z_resid` 的一步预测误差；
3. `consistency_loss`：两者合在一起的完整 latent 误差。

本轮仅增加诊断日志，不改变任何训练损失权重与优化目标。这样可以先回答“当前训练失败主要卡在结构状态推演，还是卡在 residual 学习”这一问题，再决定是否需要单独给结构状态增加显式监督项。

在重新梳理当前 Hopper 方向后，本轮又补了一条与结构化奖励彻底解耦的基线入口 [run_hopper_v5_10k_pureenv_noresid.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_10k_pureenv_noresid.sh)。该脚本显式设置：

1. `sd_reward_mode=learned`，仅保留环境奖励监督；
2. `sd_latent_dim=17`，关闭 residual latent；
3. `sd_action_prior_enabled=false`；
4. `sd_height_loss_coef=0.0`。

这样处理的原因是：若要回答“SD 是否能在 TD-MPC2 原算法中平滑取代黑盒 world model”，首先必须剥离 shaping、curriculum、residual 与动作先验，让实验只测试纯结构 dynamics 接口本身。

同时新增独立监督拟合脚本 [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py)。该脚本不依赖 TD-MPC2 训练闭环，只做以下三件事：

1. 从 DMControl Hopper 收集随机转移数据；
2. 用一步状态监督训练 `DMControlHopperSDModel`；
3. 输出一步误差、多步 rollout 误差，以及 `height / vx / vz / pitch / support_contact / full_foot_support / body_contact_z` 等关键结构量误差。

本轮意图是先回答“SD 本体是否具备拟合与推演能力”，再决定是否值得把它重新接回 TD-MPC2 规划链条。

在完成作用链修补与两轮无残差监督校准后，又重新查阅了 `sd-mcts-pcsp/docs` 中关于 `transition residual / internal residual` 的设计边界，并据此决定：Hopper 方向不再把“纯结构化必须无残差”当作原则本身。新的边界改为：

1. 显式 SD 主干继续决定主要动力学语义；
2. 若残差存在，只能是 transition residual；
3. 残差只补结构外余量，不得形成第二主干。

基于这一口径，在 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中新增了一个严格受限的 Hopper internal residual：

1. 只作用于 `vx / vz / omega` 的一步增量；
2. 使用 `state + action -> tiny MLP -> tanh -> bounded delta`；
3. 最后一层零初始化；
4. 默认关闭；
5. 不直接修改位置量、接触量或最终读出。

同时将该入口接到：

1. [sd_world_model.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/sd_world_model.py)
2. [config.yaml](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/config.yaml)
3. [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py)

并完成了一个关键正确性检查：

1. `use_transition_residual=false` 与旧模型一致；
2. `use_transition_residual=true` 但零初始化时，前向输出与无残差版逐元素一致；
3. 因此该入口不会偷偷污染现有基线。

随后新增监督对照：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid1/summary.json)

相对当前无残差最佳口径 `calib2`，结果为：

1. `one_step_struct_rmse`：`0.64415 -> 0.62297`
2. `rollout_5_rmse`：`1.10797 -> 1.06285`
3. `rollout_10_rmse`：`1.17789 -> 1.12732`
4. `rollout_20_rmse`：`1.09166 -> 1.04595`
5. `vx_mae`：`0.09129 -> 0.06626`
6. `vz_mae`：`0.39855 -> 0.24343`
7. `pitch_rate_mae`：`0.86442 -> 0.73091`

这一轮结果已经足以支持新的方法判断：

1. 受限 internal residual 不是边际改良，而是当前最强的监督基线；
2. 当前 Hopper 的主要剩余误差更像“结构外余量”，而不是“继续硬调纯结构主干”即可解决的问题；
3. 后续主线应从“纯结构化继续调参”切换为“结构主干 + 受限内部残差”，再决定是否接回 TD-MPC2。

在此基础上，又继续做了第一轮 residual scale 扫描，先比较：

1. `0.5x residual`
2. `1.0x residual`
3. `2.0x residual`

对应结果文件分别为：

1. [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_transitionresid0p5/summary.json)
2. [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid1/summary.json)
3. [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid2p0/summary.json)

结果呈现出清晰单调改善：

1. `vx_mae`：`0.07188 -> 0.06626 -> 0.06182`
2. `vz_mae`：`0.31851 -> 0.24343 -> 0.13085`
3. `pitch_rate_mae`：`0.79232 -> 0.73091 -> 0.63742`
4. `one_step_struct_rmse`：`0.63235 -> 0.62297 -> 0.60854`
5. `rollout_20_rmse`：`1.06794 -> 1.04595 -> 1.00090`

因此，到这一轮为止，最重要的新结论不是“残差有帮助”这么宽泛，而是：

1. 当前 residual 还没有出现过冲迹象；
2. 至少在 `0.5x -> 2.0x` 区间内，增大残差强度仍在持续吃到推演误差收益；
3. 当前最佳点已从“是否用残差”推进到“如何继续放大或丰富 residual 输入”这一层面。

随后又补做了 `3.0x / 4.0x residual` 的延伸对照，并新增了残差责任分析脚本 [analyze_hopper_transition_residual_usage.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_transition_residual_usage.py)。从：

1. [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu1_transitionresid3p0/summary.json)
2. [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid4p0/summary.json)
3. [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_seed0_ep32_e200_gpu2_transitionresid4p0/residual_usage_analysis.json)

可以读到一个更强的结构结论：当前 residual 的主要责任并不是微调 readout，而是在 grounded 相位里近乎整段反向抵消主干的 `vz` 更新。最直观的数字是：

1. grounded `vz.mean_abs_base_delta ≈ 0.389`
2. grounded `vz.mean_abs_resid_delta ≈ 0.394`
3. 二者方向几乎总相反

这与 `g * dt ≈ 0.392` 非常接近，说明主干在 grounded 帧里仍近似把根部竖直更新当成“自由落体减重力”。因此，当前最值得修的不是继续放大 residual，而是 grounded 接触支撑本身。

基于这一结论，在 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 中又做了两轮 grounded 主干修补，且保持了两条边界不变：

1. `STATE_DIM=17` 不变；
2. residual 仍只允许修 `vx / vz / omega` 增量。

主干修补内容为：

1. 将 grounded 接触从“单点腿弹簧”改成“前掌/后跟分点法向支撑 + 切向滑移摩擦 + 分点力矩”；
2. 再增加一层 grounded 支撑基线，使接触门打开时，主干先具备抵消重力的最小支撑能力，再由几何与腿部驱动作修正。

对应快速 smoke 如下：

1. 第一轮无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix_noresid_cpu/summary.json)
2. 第一轮 `4.0x residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix_resid4_cpu/summary.json)
3. 第二轮无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix2_noresid_cpu/summary.json)
4. 第二轮 `4.0x residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix2_resid4_cpu/summary.json)
5. 第二轮责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix2_resid4_cpu/residual_usage_analysis.json)

这两轮 smoke 的结论是：

1. 无残差 `vz_mae`：`0.3665 -> 0.3365`，说明 grounded 支撑回填已开始把责任拉回主干；
2. 但 `4.0x residual` 责任分析里，`vz.mean_abs_base_delta` 虽从 `0.3615` 进一步降到 `0.3239`，residual 仍在 grounded 样本上以反向、同量级方式修正；
3. 这说明当前方向是正确的，但 grounded 支撑仍然偏弱，尚未到“residual 只做局部修正”的程度。

因此，本轮没有把结论写成“问题已修复”，而是把下一步工作点明确收敛到：

1. 继续增强 grounded 法向支撑主干；
2. 明确建模 grounded 切向滑移 / 摩擦；
3. 在主干责任回升后，再考虑是否为 residual 增加短历史窗口，而不是继续无条件放大 residual 强度。

在上述两轮 grounded 主干修补之后，又进一步把 grounded 支撑基线从“线性乘 `total_gate`”改成了“对 `total_gate` 施加更陡的软门”。这样做的原因是：前一轮结果已经表明接触锚定几乎不改变 `vz` 量级，而支撑不足更像来自“中等接触证据时主干仍不敢给足法向支撑”。

对应的新结果为：

1. 无残差同口径：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_noresid_cpu/summary.json)
2. `4.0x residual` 同口径：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_resid4_cpu/summary.json)
3. 新责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix4_resid4_cpu/residual_usage_analysis.json)

这轮的关键数字是：

1. 无残差 `vz_mae`：`0.3365 -> 0.3144`
2. 无残差 `rollout_20_rmse`：`2.1232 -> 2.0515`
3. residual 责任分析里，`vz.mean_abs_base_delta`：`0.3239 -> 0.2941`

因此可以更具体地更新判断：

1. grounded 法向支撑门过软，确实是当前主干 `vz` 欠支撑的一个直接原因；
2. 支撑门陡化比接触锚定更有效；
3. 但 residual 仍在 `vz / omega` 上承担较大责任，当前还不能说 grounded 主干已经足够。

在此基础上，又继续对 grounded 支撑力在前掌/后跟之间的分配加入了“身体投影偏置”。对应结果为：

1. 无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_noresid_cpu/summary.json)
2. `4.0x residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_cpu/summary.json)

结果显示：

1. 无残差 `vz_mae` 继续下降到 `0.2919`；
2. `rollout_20_rmse` 小幅下降到 `2.0444`；
3. 但 `vx_mae` 恶化，`pitch_rate_mae` 基本无改善。

这说明 grounded 主干回填对 `vz` 仍然有效，但对 `vx / omega` 的边际收益已经开始明显分化；也就是说，继续只靠 grounded 支撑分配微调，已经接近“只改竖直、不改切向/姿态”的局部极限。

随后又系统审查了 `dm_control` Hopper 仿真器实现与 XML 模型。读出的最关键先验包括：

1. `control_timestep = 0.02`，而当前 TD-MPC2 包装器每个 agent 动作执行环境两次，因此当前 `SD` 的有效步长 `0.04` 并没有设错；
2. 真实总质量约 `12.44`；
3. reset 姿态下估算的俯仰等效惯量约 `2.86`；
4. 关节 `damping = 0.05`、`armature = 0.2`，执行器 gear 为 `30 / 40 / 30 / 10`。

这些信息很有价值，因此尝试直接把 `mass / inertia / gear / damping` 回填进主干初始化。但快速 smoke 结果为：

1. 无残差：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_simprior1_noresid_cpu/summary.json)
2. `4.0x residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_simprior1_resid4_cpu/summary.json)

现象非常一致：

1. 一步 `vx / vz` 误差并未明显恶化；
2. 但 `rollout_10 / rollout_20` 明显变差；
3. `pitch_rate` 也没有变好。

因此，这一轮没有保留“直接硬回填 sim prior”的代码，而是将活跃实现回退到上一版 grounded 主干最优口径。相应的方法判断也更明确了：

1. 仿真器代码提供的先验是高价值信息源；
2. 但它们不能在不重标当前归一化接触/驱动链的前提下，直接替换进主干量纲；
3. 下一轮若要继续用这些先验，应按“系统重标”而不是“局部补丁”的方式推进。

基于这一判断，又进一步做了一次真正的“系统重标”版本：除 `mass / inertia / gear / damping` 外，还把 grounded 法向支撑、空中耦合等所有显式力/力矩样项一起按质量或惯量同步重标。对应结果为：

1. [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_simprior2_noresid_cpu/summary.json)

结果比上一版 `simprior1` 更差：

1. `rollout_10_rmse = 2.88`
2. `rollout_20_rmse = 3.22`
3. `pitch_rate_mae = 1.03`

因此，本轮得出的更强方法结论是：

1. 问题不在于“sim prior 加得还不够系统”，因为更系统的重标反而更差；
2. 当前 17 维主干和这套显式接触近似，尚承接不了物理量纲更真的整体重标；
3. 所以活跃代码再次回退到 `contactfix5` 口径，并将下一阶段重点转向 residual 侧的能力增强，而不是继续在同一主干上做 sim prior 量纲替换。

## 2026-04-28：为 Hopper transition residual 补入“结构上下文 + 主干提议量”输入

在确认 `contactfix5` 已是当前 grounded 主干最佳口径之后，本轮没有继续回到 `sim prior` 或主干量纲重标，而是直接针对 residual 做了一次低风险增强。

这样做的背景是：既有 `SD + residual` 文档中已经明确残差边界应为“读取结构化当前量，只补主干未解释的局部余量”，而当前 Hopper residual 仍只看裸 `state + action`。这会造成一个明显问题：当当前误差本身就来自接触链、支撑分配和主干提议量时，residual 其实拿不到最相关的当前信息。

因此，本轮对 [dmcontrol_hopper_sd_model.py](/home/data/round/code/sd_world_model_rl/src/world_model/dmcontrol_hopper_sd_model.py) 做了如下修订：

1. 保持 `transition_residual_head` 的输出位置不变，仍只修 `vx / vz / omega` 三个一步增量；
2. 保持零初始化、`tanh` 限幅和既有 `TRANSITION_RESIDUAL_BASE_SCALE` 不变；
3. 新增 `_transition_residual_inputs()`，把 residual 输入从“裸 `state + action`”扩展为：
   `state + action + next_activation + joint_acc + 当前接触几何 + 当前腿几何 + 当前接触分配/法向/滑移量 + 主干自己的 base_delta`；
4. 为此在 `_contact_reaction()` 中把 `toe_vx / heel_vx / toe_vz / heel_vz` 透传出来，供 residual 使用。

对应同口径 smoke：

1. 旧版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_cpu/summary.json)
2. 新版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary.json)
3. 新版责任分析：
   [residual_usage_analysis.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/residual_usage_analysis.json)

关键结果为：

1. `one_step_struct_rmse`：`0.72395 -> 0.71918`
2. `vx_mae`：`0.13342 -> 0.08705`
3. `vz_mae`：`0.17875 -> 0.15467`
4. `pitch_rate_mae`：`0.77846 -> 0.74535`
5. `rollout_20_rmse`：`1.64983 -> 1.67611`

因此，本轮最准确的工程判断是：

1. 当前 residual 先前确实存在“输入上下文过贫”的问题，尤其表现在 `vx`；
2. 把接触与主干提议量显式喂给 residual 后，一步关键根部量明显改善；
3. 但长时 rollout 没有同步改善，说明这还不能视为“规划可用性已经解决”；
4. `omega` 仍是最顽固的剩余瓶颈，后续更应围绕多步一致性与俯仰力矩链继续收束，而不是回头恢复主干量纲硬替换。

同日又进一步收紧了一条验收口径：当前 TD-MPC2 Hopper 配置里的真实规划/训练 rollout `horizon = 3`，因此后续监督评估不再把 `20` 步自由 rollout 作为主目标，而是改为主看 `1 / 3 / 5` 步误差，并把 `10 / 20` 步只保留为漂移诊断。这一修订不会改写历史结果，只影响后续结果解读与脚本默认主指标。

同日继续沿这条新口径做了一次更保守的训练侧修订：不改 `SD` 或 residual 结构，只在 [fit_hopper_sd_supervised.py](/home/data/round/code/sd_world_model_rl/scripts/fit_hopper_sd_supervised.py) 中额外加入 `3 / 5` 步 `vx / vz / omega` 的短视界附加损失，其中 `omega` 权重略高。对应同口径 smoke 为：

1. 原 `feature-context residual`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_cpu/summary.json)
2. 短视界损失版：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss_cpu/summary.json)

结果表明：

1. `rollout_1_rmse`：`0.67576 -> 0.67496`
2. `rollout_3_rmse`：`1.18643 -> 1.18081`
3. `rollout_5_rmse`：`1.45345 -> 1.44230`
4. `pitch_rate_mae`：`0.74535 -> 0.73885`

但代价同样存在：

1. `vx_mae`：`0.08705 -> 0.09226`
2. `vz_mae`：`0.15467 -> 0.16231`

因此，本轮更准确的工程结论是：短视界附加损失方向正确，且已经证明它确实主要改善 TD-MPC2 真正关心的 `1 / 3 / 5` 步区间；但当前 `0.25` 权重还偏强，下一步应优先做小范围权重扫描，而不是继续增加结构复杂度。

随后立即补做了 `short_rollout_weight = 0.10 / 0.15 / 0.25` 的三点扫描。结果文件为：

1. `0.10`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/summary.json)
2. `0.15`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss015_cpu/summary.json)
3. `0.25`：
   [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss_cpu/summary.json)

相对无短视界附加损失的当前 `feature-context residual` 基线，扫描结论很清楚：

1. `0.10`：`rollout_3/5` 继续下降，`vx_mae` 还略有改善，`vz_mae` 只轻微回退；
2. `0.15`：`pitch_rate_mae` 最低，但 `vz_mae` 回退更明显；
3. `0.25`：`vz_mae` 与 `vx_mae` 都开始明显恶化，权重偏强。

因此，这一步的正式工程判断是：若主目标是服务 TD-MPC2 的 `horizon = 3`，则 `short_rollout_weight = 0.10` 是当前最平衡的新监督基线；下一步不再继续扫更大权重，而应以这条 `0.10` 线接回小预算闭环验证。

同日又将这条 `5k pure-env + supervised SD init` 验证线固化为正式长测入口。原因很直接：现有 [run_hopper_v5_100k.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k.sh) 仍绑定旧的奖励混合主线，而当前要验证的是“改进后的 `SD + residual` 是否能在不引入任何 shaping 的前提下直接服务 TD-MPC2 闭环训练”。为避免再次手写长命令并混入口径错误，新增了 [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)，并在 [HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md](/home/data/round/code/sd_world_model_rl/docs/HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md) 中登记该入口。

该脚本固定以下不变量：

1. `steps=100000`
2. `eval_freq=25000`
3. `checkpoint_freq=25000`
4. `sd_latent_dim=17`
5. `sd_use_transition_residual=true`
6. `sd_reward_mode=learned`
7. `sd_env_reward_scale=1.0`
8. `sd_action_prior_enabled=false`
9. `sd_supervised_init_from=run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu`

这样处理后，后续三种子 `100k` 启动可完全复用脚本，不再依赖人工拼接参数串。

同日新增独立设计文档 [sd_hopper_sd_residual_design.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_sd_residual_design.md)。新增该文档的原因是：当前 Hopper 线已从“奖励/状态机调参”明确转回“`SD dynamics + internal residual` 是否能承担 TD-MPC2 world model 核心职责”，而旧文档大多仍混有奖励设计、阶段化训练或已废弃的外层 latent residual 讨论。新的独立文档只保留当前活跃事实：

1. `contactfix5` 主干；
2. 17 维结构状态；
3. 内部 `transition residual` 的输入、输出与限幅边界；
4. `shortloss010` 监督口径；
5. 监督初始化接回 TD-MPC2 的正式接线方式；
6. 当前最优 checkpoint、关键数值与已知局限。

这样处理后，后续讨论 Hopper SD 时，可以直接引用这一份独立文档，而不必再从多份奖励与实验工作日志中拼接当前实现口径。

同日又根据外部专家评审意见，新增下一阶段验证计划文档 [sd_hopper_tdmpc2_validation_plan.md](/home/data/round/code/sd_world_model_rl/docs/sd_hopper_tdmpc2_validation_plan.md)。新增该文档的原因是：当前外部意见并没有推翻 `SD transition core` 这条方向，而是进一步收紧了“接下来该如何严谨验证”的要求。最重要的新增点不是新公式，而是把后续工作从“继续手调模型”转到“先做 phase-sliced error、residual audit、policy-distribution validation、action-ranking diagnostic，再做 ablation-driven 的 50k/100k 闭环实验”。

该计划文档固定了以下执行顺序：

1. 先补四类短平快诊断：
   1. `phase-sliced error`
   2. `internal residual audit`
   3. `policy-distribution validation`
   4. `action ranking diagnostic`
2. 同步保留 `5k / 10k` smoke 作为稳定性检查；
3. 后续闭环主实验必须至少包含：
   1. `vanilla TD-MPC2`
   2. `SD + no init`
   3. `SD + supervised init`
   4. `SD + supervised init + residual off`
   5. `SD + supervised init + residual on`
4. 在这些结论明确前，不再恢复任何自定义结构化 reward shaping。

这样处理后，下一阶段 Hopper 工作就不再依赖口头共识，而有了一份明确的验证与消融执行基线。
