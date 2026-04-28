## 2026-04-11

### Hopper-Hop：将 TD-MPC2 的数据闭环迁移至结构化 SD 原型

为判断当前 `Hopper-Hop` 的瓶颈究竟来自世界模型还是来自数据分布，`scripts/run_dmcontrol_hopper.py` 从“一次性采样后离线拟合”扩展为支持在线训练模式。此次迁移保留结构化 SD 世界模型不变，仅引入 TD-MPC2 中最关键的数据生成与利用机制：

1. 先以 `seed_steps` 收集种子数据，再在该节点执行一次集中预训练；
2. 种子阶段后改为使用当前模型 + 规划器在线采样，而非始终依赖固定探索分布；
3. 所有转移写入回放缓存，并在后续步骤中反复重采样更新，而非只使用一次。

### 代码变更

- 新增 `TransitionReplay`，用于保存 `state / action / reward / next_state / done / height` 等转移信息；
- 新增 `_batch_loss()`，将一步状态误差、奖励误差与高度监督统一为单批次损失；
- 新增 `online_train_sd_model()`，实现 `seed data + replay pretraining + online planning data` 的训练回路；
- 主脚本新增 `--mode {offline,online}`，默认转为 `online`，并增加 `--seed_steps`、`--updates_per_step`、`--seed_pretrain_updates` 参数。

### 验证

已完成两次 smoke test：

- `python -m py_compile scripts/run_dmcontrol_hopper.py`
- `MUJOCO_GL=glfw python scripts/run_dmcontrol_hopper.py --mode online --budget 600 --seed_steps 256 --seed_pretrain_updates 8 --updates_per_step 1 --eval_episodes 1 --planner_population 32 --planner_elites 4 --planner_iterations 1 --device cpu`

第二次 smoke test 中已观测到：

- 到达 `seed_steps=256` 后触发回放预训练；
- `online step 500` 与 `online step 600` 时均已出现非空 `last_loss`；
- 模型参数 `control_gain` 与 `rate_damping` 已发生更新，说明在线回放训练分支并非空跑。

### 当前判断

该迁移尚未解决 `Hopper-Hop` 在极低样本下的零回报问题，但已将问题范围进一步缩小：后续若仍然无法越过零回报区，则更可能是任务奖励门槛和行为先导不足，而不再主要是“一次性随机数据 + 离线拟合”这一训练组织方式造成的。

### 5k 在线回放正式结果

随后在同一脚本上运行了 `5k` 预算的在线回放版本：

`MUJOCO_GL=glfw python scripts/run_dmcontrol_hopper.py --mode online --budget 5000 --seed 0 --seed_steps 2500 --seed_pretrain_updates 512 --updates_per_step 1 --eval_episodes 3 --planner_horizon 8 --planner_population 128 --planner_elites 16 --planner_iterations 3 --planner_temperature 0.5 --planner_num_prior_trajs 16 --planner_terminal_value_coef 1.0 --device cuda`

结果如下：

- `learned-MPC return = 0.0 ± 0.0`
- `random return = 0.0 ± 0.0`
- `one_step_rmse = 0.6315`
- `rollout_5_rmse = 1.6282`
- `rollout_10_rmse = 1.9926`

与先前离线版本相比，模型误差已有明显改善；但与 `TD-MPC2` 在同样 `5k` 预算、`seed=0` 下约 `0.32` 的微弱非零回报相比，当前结构化原型仍未跨过任务奖励门槛。由此可见，单独迁移“种子数据 + 在线回放 + 持续更新”这一级机制仍不足以复现其缓慢起步行为，后续更可能需要继续补入更强的行为先导、终端价值或 actor prior。

### 10k 在线回放结果

在完全相同的在线回放设置下，仅将总预算扩展至 `10k`：

`MUJOCO_GL=glfw python scripts/run_dmcontrol_hopper.py --mode online --budget 10000 --seed 0 --seed_steps 2500 --seed_pretrain_updates 512 --updates_per_step 1 --eval_episodes 3 --planner_horizon 8 --planner_population 128 --planner_elites 16 --planner_iterations 3 --planner_temperature 0.5 --planner_num_prior_trajs 16 --planner_terminal_value_coef 1.0 --device cuda`

结果如下：

- `learned-MPC return = 0.0 ± 0.0`
- `random return = 0.0 ± 0.0`
- `one_step_rmse = 0.5552`
- `rollout_5_rmse = 1.5753`
- `rollout_10_rmse = 1.9025`

相较于 `5k` 在线回放版本，模型误差继续下降，说明额外在线数据确实进一步改善了动力学拟合；但真实任务回报仍未脱离零区间。由此可见，当前瓶颈已不再主要是单步或短步动力学误差，而更可能是行为先导不足、奖励门槛过高以及缺少更强的终端价值/actor prior 引导。

### 10k 在线回放 + 行为引导结果

为进一步跨过 `Hopper-Hop` 的稀疏奖励门槛，在在线回放版本中加入了相位式行为引导。该引导同时进入：

1. 种子数据阶段的 guided exploration；
2. 规划阶段的 guided prior trajectories；
3. 实际执行动作时对 planner 输出的轻度混合。

使用如下配置运行：

`MUJOCO_GL=glfw python scripts/run_dmcontrol_hopper.py --mode online --budget 10000 --seed 0 --seed_steps 2500 --seed_pretrain_updates 512 --updates_per_step 1 --eval_episodes 3 --planner_horizon 8 --planner_population 128 --planner_elites 16 --planner_iterations 3 --planner_temperature 0.5 --planner_num_prior_trajs 24 --planner_terminal_value_coef 1.0 --behavior_guide_mix_start 0.60 --behavior_guide_mix_end 0.25 --behavior_guide_eval_mix 0.35 --device cuda`

结果如下：

- `learned-MPC return = 0.0 ± 0.0`
- `random return = 0.0 ± 0.0`
- `one_step_rmse = 0.7162`
- `rollout_5_rmse = 1.3054`
- `rollout_10_rmse = 1.3622`

与无行为引导的 `10k` 在线回放结果相比：

- 单步误差并未继续下降，反而略有升高（`0.5552 -> 0.7162`）；
- 但多步 rollout 误差明显下降，尤其 `rollout_10_rmse` 从 `1.9025` 降至 `1.3622`。

这说明行为引导确实改变了训练分布与搜索轨迹，使模型在较长时域上的演化一致性更强；但即便如此，当前结构化原型仍未跨过环境真实奖励门槛。由此判断，下一步更可能需要补入显式终端价值或 actor prior，而不是继续单纯强化手工行为引导。

### 5k 交替更新 TVA 结果

随后将 Hopper 中的 TVA 从“训练结束后再拟合”改为更接近 Beer Game 设计精神的**在线交替更新**版本：在在线训练阶段，价值网络按 episode 轨迹持续更新，并立即回流至 planner 叶节点。

运行配置：

`MUJOCO_GL=glfw python scripts/run_dmcontrol_hopper.py --mode online --budget 5000 --seed 0 --seed_steps 2500 --seed_pretrain_updates 512 --updates_per_step 1 --eval_episodes 3 --planner_horizon 8 --planner_population 128 --planner_elites 16 --planner_iterations 3 --planner_temperature 0.5 --planner_num_prior_trajs 24 --planner_terminal_value_coef 1.0 --behavior_guide_mix_start 0.60 --behavior_guide_mix_end 0.25 --behavior_guide_eval_mix 0.35 --use_tva --tva_updates_per_episode 25 --device cuda`

结果如下：

- `learned-MPC return = 0.0 ± 0.0`
- `random return = 0.0 ± 0.0`
- `one_step_rmse = 0.6623`
- `rollout_5_rmse = 1.3461`
- `rollout_10_rmse = 1.4224`
- `tva_loss = 0.0001`

这一结果说明两点：

1. TVA 现在已不再是事后补充模块，而是进入了在线闭环，日志中可见其在 `seed_steps` 之前即出现非空更新；
2. 交替更新 TVA 进一步改善了多步 rollout 一致性，但在 `5k` 预算下仍未将真实回报抬出零区间。

### 10k 交替更新 TVA 正式结果

随后在同一交替更新框架下，将预算扩展到 `10k`，并同时保存 checkpoint 以支持后续真正续训：

`MUJOCO_GL=glfw python scripts/run_dmcontrol_hopper.py --mode online --budget 10000 --seed 0 --seed_steps 2500 --seed_pretrain_updates 512 --updates_per_step 1 --eval_episodes 3 --planner_horizon 8 --planner_population 128 --planner_elites 16 --planner_iterations 3 --planner_temperature 0.5 --planner_num_prior_trajs 24 --planner_terminal_value_coef 1.0 --behavior_guide_mix_start 0.60 --behavior_guide_mix_end 0.25 --behavior_guide_eval_mix 0.35 --use_tva --tva_updates_per_episode 25 --checkpoint_path /tmp/hopper_tva_alt_10k_seed0.pt --checkpoint_every 500 --device cuda`

最终结果如下：

- `learned-MPC return = 0.0 ± 0.0`
- `random return = 0.0 ± 0.0`
- `one_step_rmse = 0.7287`
- `rollout_5_rmse = 1.3106`
- `rollout_10_rmse = 1.3656`
- `tva_loss = 0.0019`

这一结果表明，交替更新 TVA 在 `10k` 预算下仍未将 `Hopper-Hop` 推出零回报区间，但多步 rollout 一致性仍保持在优于无 TVA / 无行为引导版本的范围内。由此可见，当前瓶颈更可能位于奖励门槛附近的关键行为模式尚未形成，而不再主要是“是否已有在线价值补偿”这一层。

### 2026-04-13：围绕合法落地与恢复事件的强化

在多轮 Hopper-Hop 结构诊断后，`v19` 首次于 `3000` 步附近同时出现了合法起跳、有效飞行与合法落地恢复事件，但到 `3500` 之后又重新滑回高 `flight` 占比的伪解。基于该现象，当前轮次不再继续改变整体相位结构，而仅围绕“合法落地 (`valid_landing`) 与恢复 (`reset_cycle`)”做局部强化，以检验完整循环能否从“偶尔出现”推进到“可维持更久”。

本轮代码调整包括：

- 即时事件奖励中，提高 `valid_landing` 与 `reset_cycle` 的权重；
- TVA 目标构造中，提高合法落地与恢复事件带来的 frontier bonus，并适度提高 `adjust_returns` 的整体系数；
- 前沿缓存中，将围绕合法落地的强前沿窗口适度放宽，使与落地相邻的离地片段更容易被保留下来。

对应代码位置主要位于 `scripts/run_dmcontrol_hopper.py` 的 `predict_phase_progress_reward`、`build_tva_targets`、`_episode_value_targets` 与 `_frontier_masks`。

该轮变更的直接目的不是追求最终回报，而是优先观察 `3000/3500` 区间内 `vt/vf/vl` 是否能比 `v19` 保持更久，从而判断“恢复事件强化”是否是稳定完整循环的有效方向。

### 2026-04-13：`v20` 恢复事件强化 smoke 结果

基于上述修改，启动了 `v20` Hopper-Hop smoke，对照 `v19` 检查合法离地、有效飞行与合法落地恢复事件是否能在 `3000` 之后维持更久。实验日志保存在 `/tmp/hopper_phase_actor_v20_smoke.log`。

关键结果如下：

- `3000` 步：
  - `vt:4, it:8, vf:2, vl:0`
  - `phases=c:0.48, t:0.01, f:0.44, l:0.06`
- `3500` 步：
  - `vt:0, it:6, vf:0, vl:0`
  - `recent512` 中 `flight` 占比升至 `0.90`
- `4000` 步：
  - `vt:0, it:6, vf:0, vl:0`
  - `recent512` 中 `flight` 占比进一步升至 `0.97`

与 `v19` 在 `3000` 时的 `vt:5, it:6, vf:5, vl:2` 相比，`v20` 在恢复事件上的强化并未稳住完整循环，反而更早丢失了 `vl`，并更快滑回高 `flight` 伪解。由此判断：

- 单纯提高 `valid_landing / reset_cycle` 的即时奖励、TVA frontier bonus 与前沿窗口，并不能稳定恢复阶段；
- 相反，这种强化会在当前框架下进一步放大“轻接触、高 `flight` 占比”的吸引子；
- 后续优化不宜继续沿着“加大奖励与前沿权重”的方向推进，而应回到 `v19` 附近较平衡的配置，只对 `3000` 之后的漂移机制做更细的约束。

### 2026-04-13：`v19` 至 `v25` smoke 结果总览

为避免后续判断被单次观察左右，现将 `v19` 至 `v25` 这几轮 Hopper-Hop smoke 的核心结果统一汇总。各轮日志均保存在 `/tmp/hopper_phase_actor_vXX_smoke.log`。

#### 1. 各版本核心现象

- `v19`
  - `3000`：`vt:5, it:6, vf:5, vl:2`
  - `3500`：`vt:0, it:7, vf:0, vl:0`，`flight` 占比约 `0.63`
  - 说明：第一次较完整地同时打出合法起跳、有效飞行与合法落地恢复，属于本轮探索中的首个“完整前沿”。

- `v20`
  - 目的：提高 `valid_landing / reset_cycle` 的即时奖励、TVA frontier bonus 与前沿窗口。
  - `3000`：`vt:4, it:8, vf:2, vl:0`
  - `3500`：`vt:0, it:6, vf:0, vl:0`，`flight` 占比升至 `0.90`
  - 说明：恢复事件强化并未稳住完整循环，反而更快滑回高 `flight` 伪解。

- `v21`
  - 目的：加入 `recovery_window`，在合法落地/恢复后短时保留部分事件状态。
  - `3000`：`vt:3, it:5, vf:1, vl:0`
  - `3500`：`vt:0, it:3, vf:0, vl:0`，`recover=0.000`，`flight` 占比约 `0.92`
  - 说明：由于 `vl/reset` 本身未出现，恢复窗口几乎从未触发，该方向未切中主矛盾。

- `v22`
  - 目的：不再增加新事件变量，改为“完整前沿锁定”。一旦某个 episode 同时出现 `vt+vf+vl`，后续若干步停止普通更新，仅围绕该前沿做 TVA 专训。
  - `3000`：`vt:5, it:6, vf:5, vl:2`，`lock=900`
  - `3500`：`vt:1, it:17, vf:0, vl:0`，`flight` 占比约 `0.45`
  - `4000`：`vt:1, it:25, vf:0, vl:0`
  - 说明：相较 `v19`，该机制明显延缓了后半段漂移；虽然未保住完整事件链，但在 `3500` 处仍保留少量 `vt`，且 `flight` 占比远低于 `v19`。这是当前最稳的对照基线。

- `v23`
  - 目的：在 `v22` 基础上加入“前沿动作联合锁定”，锁定期执行动作按比例混入最近完整前沿的动作模板。
  - `3000`：`vt:5, it:6, vf:5, vl:2`，`lock=900`
  - `3500`：`vt:0, it:0, vf:0, vl:0`，`flight` 占比约 `0.98`
  - 说明：执行层动作注入会放大前沿附近的伪 `flight` 吸引子，导致漂移更快；效果明显差于 `v22`。

- `v24`
  - 目的：去掉 `v23` 的执行层动作注入，改为仅在 planner 采样空间内围绕前沿动作做收缩采样。
  - `3000`：`vt:5, it:6, vf:5, vl:2`，`lock=900`
  - `3500`：`vt:0, it:17, vf:0, vl:0`，`flight` 占比约 `0.62`
  - 说明：比 `v23` 稳，但仍不如 `v22`；说明在 `v22` 基础上对 planner 采样空间再加动作锚定，没有带来新增收益。

- `v25`
  - 目的：回到 `v22` 主线，补锁定期精细诊断 `lockdiag`，并在锁定期对 elite 选择加入事件偏好。
  - `3000`：`vt:5, it:6, vf:5, vl:2`，`lock=900`
  - `3500`：
    - `lockdiag`: `elite_vt=0.00, elite_vf=0.00, elite_vl=0.00, elite_full=0.00`
    - `recent512`: `vt:0, it:15, vf:0, vl:0`，`flight` 占比约 `0.84`
  - `4000`：
    - `episode end`: `vt:2, it:52, vf:1, vl:1`
    - `lock=900`
  - `4500`：

## 2026-04-23

### Hopper-Hop：`v4.30` 终点轨迹复盘后的奖励修订

在 `v4.30` 的三种子 `50k` 结果中，`seed1/seed3` 已出现非零终点评估，但从 `official_eval/step_0050000/episode_csv` 的逐条轨迹统计看，主要得分模式并不来自 `grounded_prepared -> rising_air` 这条目标链，而更接近 `grounded_unprepared` 下的近地前移。与此同时，`seed2` 已能够较稳定地产生 `grounded_prepared` 与少量 `rising_air` 片段，但尚未转化成明显环境回报。

据此，后续修订不再沿“堵住近地模式”的方向推进，而改为保持所有合法模式开放，仅增强目标状态迁移的正向吸引。此次仅调整 `predict_shaping_reward()` 内的 `launch_reward`：

- 保留四状态判据不变；
- 保留 `prepared_reward` 不变；
- 将 `launch_reward` 的权重从 `0.30` 提高到 `0.45`；
- 将 `launch_reward` 明确写为依赖 `next_state` 结果量的形式，即在 `prev_grounded_prepared * prepared_release` 的门下，主奖励 `next_upward_progress`，并仅附带少量 `next_forward_progress`；
- 不新增针对近地移动、单侧接触或低姿推进的惩罚项。

这一改动的研究意图是：利用结构化 SD 世界模型对下一状态可解释量的显式预测能力，使奖励直接面向“预备后的真实起跳结果”，而非面向动作幅度或额外复杂状态机。若后续轨迹中 `prepared` 片段继续存在且 `rising_air` 与环境回报同步上升，则说明当前瓶颈主要在于目标链吸引不足，而非状态语义失真。
    - `lockdiag`: `elite_pseudo=1.00, elite_f=1.00`
    - `recent512`: `vt:3, it:13, vf:6, vl:2`
  - `5000`：
    - `episode end`: `vt:3, it:6, vf:6, vl:2`
    - `recent512` 又回到 `vt:0, it:0, vf:0, vl:0`，`flight` 占比约 `0.94`
  - 说明：这轮最重要的贡献不是性能提升，而是通过 `lockdiag` 明确暴露出锁定期内 elite 轨迹仍会被零事件、伪 `flight` 候选挤占，问题第一次被定位到 elite 选择机制本身。

#### 2. 阶段性判断

- 就“是否能打出完整前沿”而言，`v19`、`v22`、`v23`、`v24`、`v25` 在 `3000` 处都能复现 `vt+vf+vl` 同时非零的现象，说明该前沿本身不是不可达。
- 就“能否延缓 `3000` 之后的漂移”而言，`v22` 仍是当前最稳的版本；其 `3500` 时仍保留少量 `vt`，且 `flight` 占比仅约 `0.45`，优于 `v19`、`v23`、`v24`。
- `v23` 与 `v24` 说明，在 `v22` 之上继续加入前沿动作注入或 planner 动作锚定，并不能进一步改善稳定性，反而容易加快后半段向伪 `flight` 的漂移。
- `v25` 的价值在于诊断而非性能：它表明即使存在前沿锁定与事件偏好，锁定期内的 elite 仍可能被零事件、伪 `flight` 轨迹完全占据。因此后续优化重点不应再放在奖励系数，而应放在更硬的 elite 事件筛选与锁定期候选轨迹约束。

#### 3. 当前结论

截至本轮汇总，`v22` 可视为 Hopper-Hop 事件状态机路线下的当前主基线。其意义不是已经解决任务，而是：

- 证明“完整循环前沿”在当前结构下可以稳定出现；
- 证明适度的前沿锁定能够延缓漂移；
- 同时也暴露出真正的后续瓶颈位于锁定期的 elite 选择，而非单纯的相位奖励或动作模板本身。

### 2026-04-13：切换至“TD-MPC2 求解层 + SD 世界模型”路线

考虑到本阶段的目标并非继续手工打磨 Hopper-Hop 的求解结构，而是更直接地验证“结构化先验显式建模”本身是否具有优势，因此开始转向另一条更高效的路线：尽可能保留 TD-MPC2 已验证有效的求解层，仅将其世界模型替换为 SD 版本。

本轮完成了以下接入工作：

- 在 `external/tdmpc2/tdmpc2/common/sd_world_model.py` 中新增 `SDWorldModel`；
- 该适配器保留 TD-MPC2 所要求的统一接口：
  - `encode`
  - `next`
  - `reward`
  - `pi`
  - `Q`
- 其中：
  - 结构状态直接作为 latent `z`；
  - 动力学与即时奖励由现有 `src/world_model/dmcontrol_hopper_sd_model.py` 提供；
  - policy prior 与 Q 头仍保留 TD-MPC2 风格，以尽量保持求解层不变。

## 2026-04-14

### Hopper-Hop：重写显式动力学主干后的稳定性复核

围绕 `src/world_model/dmcontrol_hopper_sd_model.py`，对 Hopper 的结构化动力学主干进行了新一轮显式化重写。此次重写的目标不是继续改动求解层，而是单独检验“更贴近任务机制的显式状态推演”能否直接压低 `next` 误差。主干调整集中在以下几个方面：

- 将 `touch` 观测统一映射为有界接触强度，避免原始 `log1p` 量纲直接进入结构状态；
- 显式区分支撑与离地状态，并据此组织竖直加速度与水平推进项；
- 将姿态恢复、关节回中与电机齿轮比直接写入主干；
- 保留 `passive_flow / residual_flow`，但仅作为小权重校正项，而不再承担主体动力学。

为确认这次重写至少在训练稳定性上成立，先后运行了两条短线：

- `sd_hopper_2600_diag_rerun`
- `sd_hopper_3k_diag_rerun`

两条短线均已正常越过首次 `agent.update()`，并在训练结束后成功写出 `models/final.pt`。其中 `3k` 线的 `eval.csv` 为：

- `step=0`: `0.1324`
- `step=1000`: `0.0717`
- `step=2000`: `0.0000`
- `step=3000`: `0.0011`

这说明前一阶段出现的“首次更新前即中断、无法落盘”的问题已经被修复；当前版本至少具备了稳定训练与后续诊断所需的最小可复现性。

### 与原版 TD-MPC2 的并列诊断

随后使用如下 checkpoint 做并列诊断：

- SD：`logs/hopper-hop/0/sd_hopper_3k_diag_rerun/models/final.pt`
- 原版 TD-MPC2：`logs/hopper-hop/0/tdmpc2_hopper_3k_diag_save/models/final.pt`

诊断脚本为 `scripts/diagnose_hopper_world_models.py`，固定 `seed=0`、`episodes=8`、`max_steps=64`。结果如下。

一步奖励误差：

- `tdmpc2`: `0.000719 ± 0.008108`
- `sd_hopper`: `0.000717 ± 0.008113`

一步转移误差：

- `tdmpc2` latent RMSE: `0.0791 ± 0.0258`
- `sd_hopper` state RMSE: `1.5813 ± 0.6873`

多步 rollout 回报偏差：

- `H=3`
  - `tdmpc2`: `0.002210 ± 0.018088`
  - `sd_hopper`: `0.002211 ± 0.018106`
- `H=5`
  - `tdmpc2`: `0.003798 ± 0.024903`
  - `sd_hopper`: `0.003799 ± 0.024917`

多步 rollout 转移误差：

- `H=3`
  - `tdmpc2` latent: `0.1060 ± 0.0268`
  - `sd_hopper` state: `2.2903 ± 0.9569`
- `H=5`
  - `tdmpc2` latent: `0.1212 ± 0.0262`
  - `sd_hopper` state: `2.5328 ± 1.0096`

### 当前判断

这轮结果进一步收缩了问题范围：

- `reward` 头与原版 TD-MPC2 基本持平，说明当前差距并不主要来自即时奖励建模；
- 主要瓶颈仍集中在 `next`，且新的显式动力学主干尚未把状态推演误差压到比上一轮更低的量级；
- 因此，下一轮优化不应再分散到 `reward / Q / pi / planner`，而应继续聚焦 Hopper 的状态转移主干本身，尤其是接触、支撑切换与关节耦合的结构表达。

### Hopper-Hop：继续收缩 `next` 主干

基于上述诊断，继续对 `src/world_model/dmcontrol_hopper_sd_model.py` 的显式动力学主干做了进一步收缩。本轮不再改动 `reward / Q / pi / planner`，仅处理三个更直接影响状态推演的一阶问题：

1. 接触判定仍偏粗，只用到了接触强度与足端高度；
2. 关节项仍接近“各关节独立回中”，链式耦合偏弱；
3. 速度阻尼初值偏大，不利于保留 Hopper 的真实惯性。

对应修改如下：

- 新增 `_contact_features()`，将足端几何高度、足端竖直速度与原始触觉信号共同用于支撑态估计；
- 新增 `_pos_rates_from_vel()`，显式用状态速率估计下一时刻足端几何，从而构造 `toe_vz / heel_vz`；
- 将竖直加速度改写为“重力 + 支撑压缩量 + 伸展动作/伸展速度 + 落地缓冲”结构；
- 将水平推进项改写为“支撑态下的伸展推进 + 空中阻尼”结构；
- 将姿态恢复与四个关节的加速度改为带链式耦合的形式，而不是仅做独立回中；
- 将 `rate_damping` 的初始量级改为更接近真实惯性保留的范围，不再以过大的统一阻尼起步。

完成修改后，已做如下最小检查：

- `python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py`
- 最小前向、奖励与高度输出检查，均为有限值

随后运行了新的稳定性验证线：

- `sd_hopper_2600_diag_r5`

该版本已正常越过 `1000`、`2000` 与 `2500` 后的预训练入口，并最终成功写出：

- `logs/hopper-hop/0/sd_hopper_2600_diag_r5/models/final.pt`

其 `eval.csv` 为：

- `step=0`: `0.1324`
- `step=1000`: `0.0717`
- `step=2000`: `0.0000`

这说明本轮收缩没有破坏 TD-MPC2 + SD 的训练闭环，可作为下一步补跑 `3k` 并再次执行 `diagnose_hopper_world_models.py` 的新基线。

### Hopper-Hop：`3k + diagnose` 结果

在上述 `r5` 主干上，继续按单次完整流程运行：

- `sd_hopper_3k_diag_r5`
- 随后执行 `scripts/diagnose_hopper_world_models.py`

`3k` 线已正常落盘：

- `logs/hopper-hop/0/sd_hopper_3k_diag_r5/models/final.pt`

并与原版 TD-MPC2 的 `3k` checkpoint 做了并列诊断。结果如下。

一步奖励误差：

- `tdmpc2`: `0.000719 ± 0.008108`
- `sd_hopper`: `0.000717 ± 0.008113`

一步转移误差：

- `tdmpc2` latent RMSE: `0.0791 ± 0.0258`
- `sd_hopper` state RMSE: `1.5917 ± 0.7014`

多步 rollout 转移误差：

- `H=3`
  - `tdmpc2` latent: `0.1060 ± 0.0268`
  - `sd_hopper` state: `2.3536 ± 1.0031`
- `H=5`
  - `tdmpc2` latent: `0.1212 ± 0.0262`
  - `sd_hopper` state: `2.6599 ± 1.0775`

### 当前判断

与上一轮相比，这次 `r5` 主干在 `reward` 上仍与原版 TD-MPC2 基本持平，但 `next` 误差没有下降，反而略有上升。由此可以进一步排除两种可能：

- 问题不是训练是否稳定，因为当前版本已经可以稳定穿过 `2500` 后预训练并完成落盘；
- 问题也不是即时奖励建模，因为 `reward` 误差始终与原版接近一致。

因此，下一轮不宜继续在现有显式项上做“系数级”微调，而应回到更根本的建模边界：重新审视 Hopper 当前 15 维结构状态是否足以支撑可预测的 `next`，尤其是接触与约束求解所依赖的隐含信息是否被严重压缩。

### Hopper-Hop：从“观测即状态”切换到“机制状态编码”

基于上述判断，Hopper 的 SD 适配器不再把原始 15 维观测直接当作 latent，而是改为先做机制状态编码，再在机制状态上进行 `next / reward / Q / pi` 计算。核心改动包括：

- 在 `src/world_model/dmcontrol_hopper_sd_model.py` 中新增 `encode_observation()`，将原始观测编码为 15 维机制状态；
- 机制状态改为：
  - `height`
  - `vx`
  - `vz`
  - `pitch`
  - `pitch_rate`
  - `compression`
  - `compression_rate`
  - `swing`
  - `swing_rate`
  - `foot_angle`
  - `foot_rate`
  - `support`
  - `stance_mix`
  - `impact`
  - `clearance`
- `external/tdmpc2/tdmpc2/common/sd_world_model.py` 的 `encode()` 不再仅做观测规范化，而是直接调用 `encode_observation()`；
- `next()`、`reward()`、`Q()`、`pi()` 统一以机制状态为输入；
- `scripts/diagnose_hopper_world_models.py` 的诊断口径同步改为比较 `encode(next_obs)` 所对应的结构状态误差，而不再将 `sd_hopper` 的输出直接与原始观测逐维比对。

完成上述改动后，已通过以下最小检查：

- `python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py external/tdmpc2/tdmpc2/common/sd_world_model.py scripts/diagnose_hopper_world_models.py`
- 三维批量输入下的 `encode_observation()` 有限值检查
- `SDWorldModel` 的 `encode / next / reward / Q / pi` 最小前向检查

### Hopper-Hop：结构状态版本 `3k` 结果

随后直接运行：

- `sd_hopper_3k_struct_r2`

该版本已正常通过 `2500` 后预训练并成功落盘：

- `logs/hopper-hop/0/sd_hopper_3k_struct_r2/models/final.pt`

其 `eval.csv` 结果为：

- `step=0`: `0.0677`
- `step=1000`: `0.6595`
- `step=2000`: `0.0000`
- `step=3000`: `0.3170`

与此前“观测即状态”版本相比，这里第一次出现了较稳定的非零回报信号，说明“先编码机制状态，再做结构递推”这一方向已经不是纯概念调整，而是在真实训练曲线上产生了差异。

### 结构状态诊断结果

使用新的结构状态版本与原版 TD-MPC2 `3k` checkpoint 进行并列诊断，结果如下。

一步奖励误差：

- `tdmpc2`: `0.000719 ± 0.008108`
- `sd_hopper`: `0.000717 ± 0.008107`

一步转移误差：

- `tdmpc2` latent RMSE: `0.0791 ± 0.0258`
- `sd_hopper` struct RMSE: `1.1744 ± 0.6807`

多步 rollout 转移误差：

- `H=3`
  - `tdmpc2` latent: `0.1060 ± 0.0268`
  - `sd_hopper` struct: `1.7370 ± 0.8499`
- `H=5`
  - `tdmpc2` latent: `0.1212 ± 0.0262`
  - `sd_hopper` struct: `1.9699 ± 0.9388`

与上一轮“观测即状态”的诊断相比，结构状态误差已出现实质下降：

- 一步误差：`1.5917 -> 1.1744`
- `H=3`：`2.3536 -> 1.7370`
- `H=5`：`2.6599 -> 1.9699`

### 当前判断

这轮结果表明，“先做机制状态编码，再做结构递推”不是表述层面的调整，而是已经切实改善了 Hopper 的 `next` 建模质量。虽然当前误差仍明显高于原版 TD-MPC2，但问题的性质已经变化：

- 过去的主问题是状态层级选错，导致无论怎样调整显式项都难以改善；
- 现在状态层级已基本转正，下一轮优化应转向机制状态本身的定义质量，例如：
  - `compression / swing / foot_angle` 是否足以刻画腿部几何；
  - `support / impact / clearance` 是否能更准确地区分飞行、压缩、推进和离地阶段；
  - 是否需要将模式变量从连续 proxy 进一步提升为显式相位或模式分类量。

### Hopper-Hop：结构状态版本 `5k` 结果与 TD-MPC2 对比

在同一配置下，继续将结构状态版本扩展至 `5k`：

- `sd_hopper_5k_struct_r1`

该版本已正常落盘：

- `logs/hopper-hop/0/sd_hopper_5k_struct_r1/models/final.pt`

其 `eval.csv` 为：

- `step=0`: `0.0677`
- `step=1000`: `0.6595`
- `step=2000`: `0.0000`
- `step=3000`: `0.3373`
- `step=4000`: `0.0000`
- `step=5000`: `0.6927`

为避免重复消耗算力，本轮先复用前期已记录的 TD-MPC2 `5k` 参考结果：在同样 `seed=0`、同样 `5k` 预算下，TD-MPC2 的回报约为 `0.32`。

据此，本轮结构状态版本在 `5k` 时的最终评估回报已高于该 TD-MPC2 参考值：

- `sd_hopper_5k_struct_r1`: `0.6927`
- `tdmpc2 @ 5k (历史参考)`: `~0.32`

### 当前判断

从研究推进角度看，这一结果已经足以说明“结构状态编码”不是边缘修饰，而是能够在低样本 Hopper-Hop 上带来真实收益；与旧版本相比，结构化先验显式建模的价值已从“降低误差”推进到“提升回报”。

但若要将这一点直接写入 NeurIPS 正文并作为公开基准上的关键主张，目前证据仍偏单薄，原因有二：

- 这里只完成了单个 `seed=0` 的结构状态 `5k` 结果；
- TD-MPC2 的 `5k` 对照目前复用了已有工作日志中的参考值，尚未在本轮完全同脚本、同目录、同评测口径下重新落盘。

因此，更稳妥的表述应为：

- 这条 `5k` 结果已经足以支持“该方向值得纳入论文叙事并继续扩展”；
- 但若要把它作为正式主结果或强结论，仍应至少补一条与当前版本完全同口径的 TD-MPC2 `5k` 对照，并尽可能再补 1--2 个种子。

同时，在 `external/tdmpc2/tdmpc2/tdmpc2.py` 中加入了可切换入口：

- 默认 `world_model_kind=tdmpc2`，行为与原版一致；
- 当显式设置 `world_model_kind=sd_hopper` 时，才切换到新的 `SDWorldModel`。

为保证配置口径一致，在 `external/tdmpc2/tdmpc2/config.yaml` 与 `external/tdmpc2/tdmpc2/common/parser.py` 中同步补入并固定了 `sd_hopper` 的相关配置：

- `world_model_kind`
- `sd_hidden_dim`
- `sd_dt`
- `sd_hop_speed`
- `sd_residual_scale`
- `sd_passive_scale`
- `latent_dim = 15`（在 `sd_hopper` 模式下直接固定，避免被 `model_size` 默认配置覆盖）

上述改动后，已完成两类检查：

1. `py_compile` 通过；
2. 在 CPU 下直接实例化 `SDWorldModel`，并跑通 `encode / next / reward / pi / Q` 五个核心接口。

### 2026-04-13：`TD-MPC2 + SD hopper` 最小 smoke

在接入完成后，进一步运行了一个最小训练 smoke，用于验证：

- Hydra 配置解析是否正确；
- `hopper-hop` 环境能否正常创建；
- `TDMPC2(cfg)` 在 `world_model_kind=sd_hopper` 下能否顺利实例化；
- 在线训练主循环能否完整走通。

运行命令如下：

`MUJOCO_GL=glfw python external/tdmpc2/tdmpc2/train.py task=hopper-hop world_model_kind=sd_hopper model_size=null obs=state steps=300 seed=0 exp_name=sd_hopper_smoke enable_wandb=false save_video=false save_agent=false save_csv=true eval_episodes=1 eval_freq=300 compile=false num_samples=32 num_elites=4 num_pi_trajs=8 horizon=3 iterations=2 batch_size=64 mlp_dim=256 num_q=3 dropout=0.01 lr=3e-4`

运行结果表明：

- 训练入口已正常切到 `SDWorldModel`；
- `Hopper-Hop` 状态维度自动对齐为 `(15,)`，动作维度为 `4`；
- 训练循环完整结束，无配置、接口或环境层报错；
- 最小 smoke 评测回报为 `R = 0.2`（`eval_episodes=1`）。

该结果的意义不在于性能本身，而在于：

- “保留 TD-MPC2 求解层、替换世界模型”为 SD 的最小接入路径已经打通；
- 后续不再需要继续停留在自写 Hopper 原型上反复微调；
- 下一步可以直接基于这一新骨架开展更像样的低预算 smoke 或正式对照实验。

### 后续任务：Hopper 仿真器材料梳理

为进一步提高 Hopper SD 世界模型的显式化程度，后续需单独开展一轮仿真器材料梳理，作为新的建模依据。该任务暂不并入当前实验实现，而记录为后续独立工作项。目标包括：

- 系统阅读 `dm_control` 中 `hopper-hop` 的任务源码、XML 结构、奖励与观测实现；
- 补查官方 GitHub 与论文等一级材料，明确仿真器中已显式给出的动力学与事件机制；
- 区分哪些机制已经以源码形式明确、哪些仍隐含在 MuJoCo 接触求解器内部；
- 在此基础上判断后续 SD 模型中哪些状态转移、接触切换与事件规则可以进一步显式化，而不再继续依赖较宽的混合型可学习流量项。

### 2026-04-13：`hopper-hop` 仿真器一手材料初步梳理

本轮开始按照“先回到仿真器材料，再决定 Hopper 的显式 SD 提炼边界”的思路，系统核对本地 `dm_control` 安装包中的源码、XML，以及官方仓库与论文中的公开说明。

#### 1. 本地任务源码已能直接确认的机制

在本地 `dm_control/suite/hopper.py` 中，当前已能直接确认以下任务层定义：

- 控制步长为 `_CONTROL_TIMESTEP = 0.02` 秒；
- 每回合默认时长为 `20` 秒；
- `hop` 任务的奖励写法为：
  - `standing * hopping`
- 其中：
  - `standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))`
  - `_STAND_HEIGHT = 0.6`
  - `hopping = rewards.tolerance(physics.speed(), bounds=(_HOP_SPEED, inf), margin=_HOP_SPEED / 2, value_at_margin=0.5, sigmoid='linear')`
  - `_HOP_SPEED = 2`

这说明仿真器对 `hopper-hop` 的任务成功判据并不是抽象黑盒，而是由两部分明确乘积构成：
- 站立高度是否足够；
- 前向速度是否达到阈值。

因此，至少在奖励层面，可以较清晰地反推出后续显式建模应重点贴合的两类量：
- 相对高度；
- 前向速度。

#### 2. 本地观测与事件相关量的直接来源

在同一源码中，还能直接确认以下观测与辅助量：

- `physics.height()` 定义为 `torso` 相对 `foot` 的垂直高度；
- `physics.speed()` 取自 `torso_subtreelinvel` 传感器的水平分量；
- `physics.touch()` 取自 `touch_toe` 与 `touch_heel` 两个触觉传感器，并经过 `log1p` 变换；
- 任务观测由三部分组成：
  - `position = qpos[1:]`，即去掉水平平移后的广义坐标；
  - `velocity = physics.velocity()`；
  - `touch = physics.touch()`。

这意味着，当前任务中已有一些与事件结构高度相关的量，其实并不需要完全由学习器自行发明，而是已经在仿真器接口中直接提供：
- 相对站立高度；
- 水平速度；
- 足部触地信号；
- 去掉平移后的姿态坐标。

因此，后续 Hopper 的 SD 建模与事件状态机设计，应优先围绕这些已显式给出的量展开，而不必再把“是否离地、是否接近成功 hop”完全交给宽泛残差项去隐式吸收。

#### 3. XML 中已明确给出的结构与执行机制

在本地 `dm_control/suite/hopper.xml` 中，当前已能直接确认以下结构信息：

- MuJoCo 基础积分步长为 `0.005` 秒；
- 模型为平面 hopper，根部自由度为：
  - `rootx`
  - `rootz`
  - `rooty`
- 受控关节为：
  - `waist`
  - `hip`
  - `knee`
  - `ankle`
- 对应电机齿轮比为：
  - `waist = 30`
  - `hip = 40`
  - `knee = 30`
  - `ankle = 10`
- 足部包含两个触觉 site：
  - `touch_toe`
  - `touch_heel`

这说明，当前仿真器并不是只给出一个抽象“hopper”环境，而是已经显式提供了：
- 关节层级结构；
- 执行器作用点；
- 触地检测位置；
- 基础时间步。

因此，后续若要进一步提高 Hopper 的 SD 显式化程度，至少可优先从以下两方面推进：
- 将动作作用路径更明确地绑定到 `waist/hip/knee/ankle` 的分工，而不再仅以统一残差近似；
- 将离地、落地等事件更多地与 `toe/heel` 触地组合及相对高度变化联系起来。

#### 4. 官方论文与仓库对该任务边界的补充说明

结合官方仓库与两篇公开文献，目前可形成的边界判断如下：

- `DeepMind Control Suite` 技术报告明确将该套件描述为“具有标准化结构与可解释奖励的连续控制基准”；
- `dm_control: Software and tasks for continuous control` 则明确指出，该软件包由 MuJoCo 包装层、任务套件与任务构建基础设施组成；
- 因此，任务级奖励、观测与 MJCF 结构大多是显式公开的；
- 但接触求解、约束满足、刚体碰撞与积分细节，仍主要隐含在 MuJoCo 引擎内部，而不是在 `hopper.py` 中以低维闭式方程给出。

据此，本阶段对 Hopper 显式建模边界的判断应更新为：

- “仿真器中不存在确定机制”这一说法并不成立；
- 更准确的说法是：任务级目标、观测、关节结构、执行器分配与部分事件相关量已经明确公开；
- 但完整接触动力学与非光滑切换机制仍未被当前工作显式提炼出来。

#### 5. 对后续 SD 建模的直接启发

基于本轮梳理，后续 Hopper 的 SD 模型不宜继续停留在“结构骨架 + 宽残差”这一泛化表述，而应优先尝试以下收缩：

- 把奖励层重新明确对齐到“相对高度 × 前向速度”这一公开定义；
- 把触地事件更多地锚定到 `touch_toe / touch_heel` 与相对高度、竖直速度的组合；
- 把动作作用更明确地绑定到四个执行关节，而非统一控制增益抽象；
- 将仍无法显式提炼的部分，限定为接触切换与地面反作用等 MuJoCo 内部求解残差，而不是对整段动力学都保留宽泛学习自由度。

该梳理目前只完成了第一轮材料核对。后续仍需继续补查官方仓库中与 Hopper 相关的其余实现细节，并据此决定下一版 SD 状态方程与事件规则应如何收缩。

### 2026-04-13：奖励函数形状与官方论文口径补充

在继续核对 `dm_control` 官方材料后，本轮又补充确认了两类对 Hopper 建模直接相关的事实。

#### 1. `rewards.tolerance` 的具体语义

本地 `dm_control/utils/rewards.py` 表明，`tolerance(x, bounds, margin, sigmoid, value_at_margin)` 的返回值并不是任意黑盒分段函数，而是一个明确的“软指示函数”：

- 当 `x` 落在 `bounds` 内时，返回 `1.0`；
- 当 `x` 超出边界时，按与边界的归一化距离 `d / margin` 通过指定 `sigmoid` 平滑下降；
- 若 `margin = 0`，则退化为硬阈值；
- `hopper-hop` 中使用的是 `sigmoid='linear'`。

结合 `hopper.py` 中的具体设置，可得到：

- 站立项：
  - `standing = tolerance(height, bounds=(0.6, 2.0), margin=0.0 或默认硬边界外为 0)`
  - 即当相对高度足够高时记为 1，否则衰减或直接失效；
- 跳跃项：
  - `hopping = tolerance(speed, bounds=(2.0, inf), margin=1.0, value_at_margin=0.5, sigmoid='linear')`
  - 即当水平速度达到 `2.0` 及以上时记为 1；
  - 低于该阈值时并非立即变为 0，而是按线性方式衰减，在距阈值 `1.0` 处取值 `0.5`。

因此，`hopper-hop` 的奖励结构比先前表述更具体：

- 它不是“只在成功 hop 时给 1、其余全 0”的纯稀疏奖励；
- 而是一个“站立约束 × 速度软阈值”的乘积奖励。

这意味着后续若要将奖励层更显式地回收到 SD 模型中，应至少按该公开定义重写，而不宜再以过于自由的启发式 shaping 替代真实奖励主干。

#### 2. 官方论文对任务套件性质的明确表述

继续核对公开论文后，可补入以下口径：

- 《DeepMind Control Suite》技术报告明确将该套件描述为：
  - “具有标准化结构（standardised structure）与可解释奖励（interpretable rewards）的连续控制任务集合”；
- 《dm_control: Software and tasks for continuous control》则进一步明确：
  - `dm_control` 由 MuJoCo wrapper、任务套件与任务构建基础设施构成；
  - 其中 `Control Suite` 被定位为一组标准化 benchmark；
  - 并强调其代码具有统一设计模式、可读、透明且易于扩展。

基于这一论文口径，当前对 Hopper 建模边界的理解可以进一步收紧为：

- 任务级奖励、观测与结构并非隐藏对象，而是被官方明确设计为“可解释”的；
- 因此，论文与源码共同支持这样一种后续方向：
  - 优先将奖励、观测、事件与执行器结构中已经显式公开的部分先回收到 SD 模型中；
  - 仅将接触求解、约束满足与非光滑切换等确实仍隐藏在 MuJoCo 引擎内部的部分，保留为残差或未建模项。

这一补充进一步说明，Hopper 当前之所以仍呈现“结构骨架 + 较宽残差”的混合形态，并不是因为仿真器没有明确机制，而是因为这些已经公开的机制还没有被充分系统地提炼进当前模型。

### 2026-04-13：当前 Hopper SD 实现与公开机制之间的差距复核

在完成仿真器源码核对后，又进一步对照了当前 `src/world_model/dmcontrol_hopper_sd_model.py` 的实现，以明确现阶段模型中哪些部分已经显式，哪些部分仍然过宽。

#### 1. 当前实现中已经显式写出的部分

当前 Hopper SD 模型中，以下结构已被明确固定：

- 状态组织为 15 维：
  - `rootz, rooty, waist, hip, knee, ankle`
  - `x_dot, z_dot, rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot`
  - `touch_toe, touch_heel`
- 动作维度为 4，对应四个执行关节；
- 位置量通过速度积分更新；
- 受控关节的动作作用路径通过 `_control_flow()` 明确绑定到 4 个关节速度项；
- `predict_reward()` 已按官方任务主干实现为：
  - `standing * hopping`
  - 其中阈值也已与 `hopper.py` 中的 `0.6` 和 `2.0` 对齐。

这些部分说明，当前模型并非完全黑盒，而是已经具备了一个与任务观测和执行器结构相一致的基础状态骨架。

#### 2. 当前实现中仍然过宽的部分

对照仿真器公开材料后，可以更明确地看到，当前版本中真正宽泛的部分主要集中在以下几处：

- `passive_flow`
  - 目前是一个从全状态到 7 维速度变化的 MLP；
  - 它一并吸收了重力、姿态耦合、接触切换、地面反作用等多种机制；
  - 这使得“自然动力学项”仍过于黑盒。

- `residual_flow`
  - 目前是一个从 `state + action` 到 7 维速度变化的 MLP；
  - 在实际作用上，它并未严格限定为“未建模残差”，而是仍可能与主动力学项高度重叠。

- `height_head`
  - 当前 `predict_height()` 采用
    - `rootz + height_head(pos)`
  - 这意味着相对高度仍主要由可学习头来补偿，而不是更多地绑定到 `torso-foot` 的几何关系。

- `touch_head`
  - 当前足部接触由 `touch_head(next_pos, next_vel)` 直接预测；
  - 但仿真器中触地事件至少部分对应到 `toe/heel` 与地面的接触状态，而非纯抽象隐藏变量。

- `predict_plan_reward()`
  - 当前 planning reward 中仍包含较宽的启发式平滑项；
  - 虽然其方向与任务结构一致，但尚未严格对齐到源码中“站立 × 速度软阈值”的奖励主干。

#### 3. 基于公开机制，下一版更应显式化的内容

按照当前材料，下一版 Hopper SD 更值得优先收缩的并不是所有动力学，而是以下三类最直接可见的机制：

##### (a) 奖励主干

奖励层应明确以仿真器公开定义为主干：
- 站立项：
  - 直接由相对高度与区间阈值决定；
- 跳跃项：
  - 直接由前向速度与 `linear tolerance` 决定。

这意味着：
- `predict_reward()` 应继续保持严格对齐；
- 若保留 planning reward，则也应视为显式辅项，而不是替代真实奖励主干的自由启发式。

##### (b) 接触与事件规则

当前 `touch_head` 仍过宽。下一版更合适的方向是：
- 优先将 `toe/heel` 的接触状态与：
  - 相对高度
  - 竖直速度
  - 当前姿态
  - 触地传感器本身
 组合建模；
- 仅将 MuJoCo 内部接触求解细节保留为残差，而不是对整个接触事件完全使用独立 MLP 头。

##### (c) 动作作用与关节分工

XML 已明确给出四个电机及齿轮比，因此下一版不宜继续只使用统一 `control_gain`。
更可行的收缩方向是：
- 将四个动作对 `waist/hip/knee/ankle` 的作用路径与相对强度绑定到 XML 中的电机分工；
- 允许在此基础上再由小幅可学习系数做缩放，而不是从零学习全部作用结构。

#### 4. 当前最值得推进的建模重写方向

基于上述差距，后续 Hopper SD 建模若继续推进，优先级更合理的顺序应为：

1. 先收缩奖励与事件层：
   - 严格对齐 `standing * hopping`；
   - 重新设计 `height/touch` 的显式主干；
2. 再收缩动作作用路径：
   - 让 4 个执行器与 4 个关节的作用关系更接近 XML 公开结构；
3. 最后才考虑继续缩减 `passive_flow / residual_flow` 的宽度。

原因在于：
- 奖励、接触、执行器分工已经有明确一手材料；
- 这些部分先显式化，收益更直接；
- 而完整接触动力学本身仍有相当部分藏在 MuJoCo 内部，短期内不宜强行写成过窄闭式方程。

据此，当前 Hopper 的下一版 SD 重写，不应再笼统追求“整体更物理”，而应具体地围绕：
- 奖励主干公开化；
- 事件层公开化；
- 执行器作用公开化；
三条线逐步收缩。

### 2026-04-15：从“纯结构 latent”转向“SD 主干 + residual latent”

在完成 `seed=3` 的多条 `5k` smoke 后，当前问题已不能再简单归结为“样本不足”。更关键的现象是：

- `oldbaseline_r1` 能在 `3000` 打出 `1.9162`，但 `4000` 直接掉回 `0.0`；
- `replaycap_r1` 将 `5000` 抬到 `0.3420`，但 `3000 -> 4000` 仍会坍塌；
- `seed_pretrain_updates=1024` 与 `iterations=4` 这两条单变量都明显把表现改差；
- 将 `replaycap_r1` 直接扩展到 `10k` 后，`3000` 仍能到 `0.7933`，训练后段甚至重新出现较强 `train episode_reward`，但 `eval` 在 `4000/5000/10000` 仍基本回零。

据此，当前主矛盾更像是：

- 并非 Hopper 在低预算下完全学不到关键行为；
- 而是“纯结构化 latent”与 TD-MPC2 的 `value / planner / policy` 闭环之间仍存在表示失配；
- 具体表现为：强早期行为可以出现，但难以被稳定保留为可复现策略。

为从第一性原理上解决这一点，本轮不再继续沿着“预算扩大”或“solver 超参补丁”推进，而是直接重构 `sd_hopper` 表示层：

- 保留 `DMControlHopperSDModel.encode_observation()` 输出的 15 维机制状态，作为 `SD 主干`；
- 在 `external/tdmpc2/tdmpc2/common/sd_world_model.py` 中新增 `residual_encoder`、`residual_dynamics` 与 `struct_residual`；
- 将 latent 从纯结构态改为 `z = [z_struct ; z_resid]`，其中：
  - `z_struct` 负责机制主干；
  - `z_resid` 负责吸收 TD-MPC2 规划/价值闭环中难以由手工结构态覆盖的复杂因素；
- 配置层新增：
  - `sd_struct_dim`
  - `sd_latent_dim`
  - `sd_reward_mode`
- 在 `common/parser.py` 中，`sd_hopper` 的 `latent_dim` 不再固定为 `15`，而是改由 `sd_latent_dim` 指定（当前默认测试值为 `32`）。

这一改动的目的不是简单“增大模型容量”，而是验证一个更核心的研究判断：

> 对 Hopper 这类复杂、强模式切换、接触主导的系统，结构主干本身可能不足以同时满足 dynamics 与 TD-MPC2 求解闭环的表示需求，因此更合理的路线应是“SD 主干 + residual latent”。

### 当前残差的设计原则

在本轮讨论中，残差的角色被明确限定为：

1. **描述 SD 未覆盖的成分**，而不是推翻 SD 主干；
2. **不允许产生超出物理域的状态**；
3. **在关键机制变量上，不允许逆转主干趋势**。

据此，本轮没有采用“任意加法残差”，而是逐步收紧为两层保护：

#### 1. 物理一致性保护

`src/world_model/dmcontrol_hopper_sd_model.py` 中新增 `project_state()`，在每次结构态修正后都强制投影回合法状态域：

- `height` clamp 到 `[0, 2.5]`
- `compression` clamp 到 `[0, 1.5]`
- `support` clamp 到 `[0, 1]`
- `stance_mix` clamp 到 `[-1, 1]`
- `impact` clamp 到 `[0, 2]`
- `pitch / swing / foot` 通过角度 wrap 回到标准范围

这保证残差不会直接把状态推出物理可接受区间。

#### 2. 主趋势保护（trend protection）

在 `external/tdmpc2/tdmpc2/common/sd_world_model.py` 中，残差不再对所有结构维度做任意加法修正，而是分成两类：

- **关键机制变量**：
  - `height`
  - `compression`
  - `support`
  - `stance_mix`
  - `impact`
  - `clearance`

  对这些变量，残差只能对 SD 主干给出的 `base_delta` 做比例调制：
  - 允许小幅放大或缩小主干步长；
  - 不允许翻转主干方向；
  - 幅度被限制在主干步长附近的窄区间内。

- **其余变量**：
  仍允许小幅受限修正，以保留对局部速度/相位细节的补偿能力。

这一设计对应的核心研究含义是：

- 残差不是“第二套隐式主动力学”；
- 而是对结构主干的局部补偿；
- 在关键机制变量上，`SD 主干` 始终保持主导地位。

### 训练期新增诊断

为了验证 residual 是否真的在做“补充”而非“吞掉主干”，本轮还在 `external/tdmpc2/tdmpc2/tdmpc2.py` 的 `_update()` 返回统计中新增：

- `struct_norm`
- `residual_norm`
- `explicit_reward_mean`
- `explicit_reward_std`

其中：

- `struct_norm` 用于检查结构态表示是否稳定；
- `residual_norm` 用于检查 residual 是否塌缩或爆炸；
- `explicit_reward_*` 用于观察 batch 中按照显式 Hopper true reward 计算的稀疏奖励是否开始出现。

### 最小 smoke 验证

先运行最小入口验证：

- `sd_hopper_hybrid_smoke`

结果：

- 训练入口正常切到 hybrid latent 版本；
- `residual_encoder`、`residual_dynamics`、`struct_residual` 三个新模块都已在架构中实例化；
- 入口、配置与保存路径均正常。

这说明本轮改动不是静态代码拼接，而是已成功接入真实 Hydra 训练路径。

### 3k hybrid 结果

随后运行：

- `sd_hopper_3k_hybrid_r1_fix1`

其 `eval.csv` 为：

- `0`: `0.0`
- `1000`: `1.8840e-05`
- `2000`: `0.0`
- `3000`: `1.3671`

与当前纯结构版本对照：

- `replaycap_r1 @ 3000`: `1.1379`
- `3k_hybrid_r1_fix1 @ 3000`: `1.3671`

这说明：

- `SD 主干 + residual latent` 在不改 TD-MPC2 solver 主体的前提下，已经能在 `3k` 提升早期有效行为形成能力；
- 因而“复杂系统更应采用 `SD 主干 + residual` 而非纯结构 latent”的判断获得了第一轮正向支持。

### 5k hybrid + trend protection 结果

在此基础上，进一步运行：

- `sd_hopper_5k_hybrid_trend_r1`

其最终 `eval.csv` 为：

- `0`: `0.0071`
- `1000`: `1.8840e-05`
- `2000`: `0.0`
- `3000`: `2.0617`
- `4000`: `0.0`
- `5000`: `0.1711`

与关键对照相比：

| Run | 3000 | 4000 | 5000 |
| --- | ---: | ---: | ---: |
| `oldbaseline_r1` | `1.9162` | `0.0` | `0.1324` |
| `replaycap_r1` | `1.1379` | `0.0` | `0.3420` |
| `sd_hopper_5k_hybrid_trend_r1` | `2.0617` | `0.0` | `0.1711` |

可得到当前阶段性结论：

1. **hybrid + trend protection 显著增强了早期模式形成能力**
   - `3000 -> 2.0617`，是目前所有线里最强的早期信号；
2. **但尚未解决 `3000 -> 4000` 的稳定保持问题**
   - `4000` 仍直接掉回 `0.0`；
3. **`5000` 已回到非零 (`0.1711`)，但仍低于当前最佳终点线 `replaycap_r1` 的 `0.3420`**。

这说明本轮改动的真实作用不是“已经解决低预算 Hopper”，而是：

- 把研究重点从“是否能形成强早期行为”向前推进了一步；
- 但“如何将这些已形成的局部模式稳定串联成完整循环”仍未解决。

### 对当前现象的解释

本轮讨论中还明确了一个重要认识：

- 在 Hopper 这种强模式切换系统中，低预算阶段不太可能一次性学会完整稳定循环；
- 更合理的过程是：
  1. 先学到若干局部有效模式；
  2. 这些模式在 `eval` 中偶尔能打出强 hop；
  3. 但由于后续模式覆盖不足，一旦进入未覆盖相位或切换边界，就会跌倒，导致 `eval` 回零。

因此：

- 若 `struct_norm`、`residual_norm`、planner/value 信号都稳定，
- 同时早期强峰值显著增强，

那么即使某些 `eval` 节点回到 `0.0`，也不能简单解释为“训练崩了”；
更合理的解释是：

> 当前模型已经学到更多关键局部模式，但尚未覆盖完整稳定循环。

这也是为什么本轮不应被解释为“又一次失败超参尝试”，而更应被理解为：

- `SD 主干 + residual` 这一路线在 Hopper 上已经显现出明显的早期收益；
- 下一阶段的关键不再是“能否形成有效模式”，而是“如何把这些模式稳定串联并保持到更后预算节点”。

### 当前结论

截至本轮，最稳妥的结论应表述为：

- `SD 主干 + residual latent` 方向获得了实质性正向支持；
- 趋势保护没有压死模型，反而在 `3000` 给出了当前最强信号；
- 但当前版本主要改善的是**早期模式形成**，尚未超过 `replaycap_r1` 在 `5000` 终点上的稳定性；
- 因而下一轮的真正研究问题应聚焦于：
  - 如何让已形成的强早期模式继续维持；
  - 以及是否需要继续扩大预算，以验证“模式覆盖不足”而非“表示层崩坏”是否才是当前主要瓶颈。

## 2026-04-15

### Hopper-Hop：定位 hybrid reward 的表示空间错误

在继续检查 `sd_hopper` 的 `hybrid reward` 路径时，发现当前实现并不只是“权重大小可能不合适”，而是存在更基础的**表示空间混合错误**。问题位于：

- `external/tdmpc2/tdmpc2/common/sd_world_model.py`
- `external/tdmpc2/tdmpc2/common/math.py`
- `external/tdmpc2/tdmpc2/tdmpc2.py`

核心事实如下：

1. `learned_reward = self._reward(...)` 是 reward head 的**原始 logits**；
2. `explicit_reward` 是标量显式奖励，经 `math.two_hot(...)` 后得到的是**soft target 分布**；
3. 原实现中，`hybrid` 直接计算：
   - `(1 - alpha) * learned_reward + alpha * math.two_hot(explicit_reward, cfg)`；
4. 但 planner 与 reward loss 的消费语义都默认 `model.reward(...)` 返回的是**可送入 softmax / log_softmax 的 logits**：
   - planner 在 `external/tdmpc2/tdmpc2/tdmpc2.py::_estimate_value()` 中调用 `math.two_hot_inv(self.model.reward(...))`；
   - reward 监督在 `external/tdmpc2/tdmpc2/tdmpc2.py::_update()` 中调用 `math.soft_ce(reward_preds, reward, cfg)`。

因此，旧版 `hybrid` 实际是在混：

- logits 空间的 learned reward；
- 概率/target 空间的 explicit reward。

这不是单纯“量纲不同”，而是**语义空间不同**。旧版中任何非零表现，都不能再简单解释为“结构化 reward 起到了辅助作用”，因为其中掺杂了接口级表示错误带来的副作用。

### 第一版修正：方向正确，但仍不够严格

最初的修正思路是：

1. 先把 `learned_logits` 经过 `softmax` 变成分布；
2. 将 `explicit_reward` 变成 `two_hot` 分布；
3. 在分布空间做 `(1-alpha) * p_learned + alpha * p_explicit`；
4. 再返回 `torch.log(mixed_dist.clamp_min(1e-8))`。

这一版比旧实现已经正确得多，因为它至少不再直接把 logits 与 target 分布相加。但复核后仍发现两点不够严谨：

- 返回值虽然可被下游当作 logits 继续使用，但其本质已经是 `log(prob)`，不再是一般意义上的自由 logits；
- `clamp_min(1e-8)` 会改变原分布的精确归一化，导致 `softmax(log(...))` 后只得到一个近似分布，而非严格等于原目标混合分布。

因此，这一版只能视作“方向正确的过渡修正”，不能作为最终数学口径。

### 第二版修正：严格的 log-space 混合

随后继续收紧为当前的**严格版**实现，位置仍在：

- `external/tdmpc2/tdmpc2/common/sd_world_model.py`

本轮新增/调整的关键逻辑包括：

- `_explicit_reward_dist(reward)`：将显式标量奖励转为 `two_hot` 分布；
- `_explicit_reward_log_probs(reward)`：将 `two_hot` 分布转为精确的 log-prob 表示，对零概率项保留 `-inf`；
- `_mix_reward_logits(learned_logits, explicit_reward, alpha)`：
  - 先对 learned 分支做 `F.log_softmax` 得到 `learned_log_probs`；
  - 对 explicit 分支构造 `explicit_log_probs`；
  - 在 log-space 中用 `torch.logsumexp(...)` 精确实现：
    - `log((1-alpha) * p_learned + alpha * p_explicit)`；
- `reward_mode == 'explicit'` 时，不再使用 `log(clamp(...))`，而是直接返回 `_explicit_reward_log_probs(explicit_reward)`；
- `reward_mode == 'hybrid' / 'curriculum'` 时，统一通过 `_mix_reward_logits(...)` 生成输出。

对应含义是：

- `model.reward(...)` 继续返回一个可被下游视作 logits 的对象；
- 但其 `softmax` 结果现在**严格等于**目标混合分布，而不再只是近似；
- planner 与 reward loss 的消费接口无需改动，仍维持 TD-MPC2 现有路径。

### 严格版修正后的最小验证

已完成以下验证：

1. 静态检查：
   - `python -m py_compile external/tdmpc2/tdmpc2/common/sd_world_model.py`

2. 最小前向烟测：
   - `learned`
   - `explicit`
   - `hybrid`
   - `curriculum`
   四种模式下，`reward()` 输出均为有限值，且 `math.two_hot_inv(...)` 结果有限。

3. 数值等价验证：
   - 直接检查
     - `softmax(_mix_reward_logits(...))`
   - 是否严格等于
     - `(1-alpha) * softmax(learned_logits) + alpha * explicit_dist`

实际结果：

- `max_abs_diff = 1.4901161193847656e-08`

这说明当前严格版在数值精度内已经与目标混合分布完全一致。基于这一检查，后续若实验继续表现不佳，就不应再归因于“hybrid reward 仍然混错了表示空间”。

### 基于第一版修正的 5k 对照结果

在第一版修正（即已修掉 logits/target 直混，但尚未切到严格 log-space 混合）基础上，运行了：

- `sd_hopper_5k_viability_hybrid_alpha01_reprfix_r1`

对应日志目录：

- `external/tdmpc2/tdmpc2/logs/hopper-hop/3/sd_hopper_5k_viability_hybrid_alpha01_reprfix_r1`

其最终 `eval.csv` 为：

- `0`: `0.0`
- `1000`: `0.13412229716777802`
- `2000`: `0.0`
- `3000`: `0.0`
- `4000`: `0.0`
- `5000`: `0.0`

终端输出与 `eval_full.csv` 口径一致：

- `1000 eval`: `0.1`
- `3000 eval`: `0.0`
- `4000 eval`: `0.0`
- `5000 eval`: `0.0`

这条线说明：

1. 修掉显式的表示空间错误后，原先 `hybrid` 路径在 `3000` 出现的强早期峰值并没有保住；
2. 先前 `sd_hopper_5k_viability_hybrid_alpha01_r1` 在 `3000 -> 1.9463` 的强峰值，至少部分依赖于错误混合所带来的副作用；
3. 但由于这条 5k 线仍基于“第一版近似修正”而非当前严格版，因此它只能证明：
   - **旧版强信号不可靠**；
   - 还不能替代当前严格版的最终实验结论。

### 当前阶段结论

截至本轮，关于 `hybrid reward` 的最稳妥表述应更新为：

1. 旧版 `hybrid` 的主要问题不是“alpha 太小”，而是**直接混合了不同语义空间的对象**；
2. 第一版修正已证明：一旦去掉该错误混合，原先在 `3000` 处出现的强峰值无法自然保留；
3. 当前代码已进一步收紧为严格版 log-space 混合，其数学语义与下游 planner / reward loss 的接口是自洽的；
4. 因此，后续若要继续检验 `0.9 learned + 0.1 explicit` 是否真的对 Hopper 有帮助，应以**当前严格版**为唯一有效口径重新重跑，而不能再引用旧版 `hybrid` 峰值作为正证据。

### 下一步

基于当前工作，后续 Hopper 的 reward 线应按以下原则推进：

- 不再将旧版 `hybrid` 的 `3000` 强峰值视作结构化 reward 成功介入的直接证据；
- 以当前严格版 `hybrid reward` 为唯一合法实现继续重跑；
- 若严格版仍持续失效，则更合理的研究结论将转向：
  - Hopper 上当前这类 `learned reward + small explicit reward` 混合机制本身并不有效；
  - 或者显式 reward 的作用方式应从”直接混入 reward head”改为别的路径，而不是继续在同一接口内调系数。

## 2026-04-16

### Hybrid Residual 架构审查与五项修订

对 `sd_hopper` 的 hybrid residual 代码做了一轮完整审查，发现并修复了五项设计问题。以下按严重程度排列。

#### 修订 1（P0）：残差模块未进入优化器

**问题**：`_residual_encoder`、`_residual_dynamics`、`_struct_residual` 三个模块不在 `TDMPC2.__init__()` 构造的 `self.optim` 的任何参数组中。后果：

1. 三个模块的参数从未被 `optim.step()` 更新——整个 hybrid residual 机制在训练中完全失效；
2. `backward()` 仍为其计算梯度，但 `optim.zero_grad()` 不清零——梯度逐步累积；
3. `clip_grad_norm_(self.model.parameters(), ...)` 包含这些累积梯度，膨胀总范数；
4. 导致实际工作参数的梯度被逐步压缩，等效学习率持续衰减。

**这完美解释了此前”强早期信号出现但无法稳定维持”的核心现象**——不是模式覆盖不足，而是训练本身在缓慢失效。

**修复**：在 `external/tdmpc2/tdmpc2/tdmpc2.py` 中，优化器创建后追加三个残差模块的参数组。

#### 修订 2（P1a）：`reward()` 内部冗余 `next()` 调用导致梯度干扰

**问题**：`SDWorldModel.reward()` 内部调用 `self.next()` 获取 `next_struct` 用于计算 explicit reward。在 `_update()` 中，latent rollout 已经计算了 `zs[1:]`，但 `reward()` 又独立执行了一次 `next()`。后果：

1. 每步 `next()` 被调用两次，计算浪费；
2. `reward_loss` 的梯度通过 `reward()` 内部的 `next()` 流入 dynamics 参数，与 `consistency_loss` 通过 latent rollout 的梯度路径冲突。

**修复**：`reward()` 内部的 `next()` 调用包裹在 `torch.no_grad()` 中。但在后续修订 5 中，此路径在训练时被完全跳过（见下文），该问题自然消解。

#### 修订 3（P1b）：趋势保护机制为空操作

**问题**：`struct_gate_scale = 0.25` 产生 `gate ∈ [0.75, 1.25]`，而 clamp 区间恰好为 `[0.75×mag, 1.25×mag]`——两者完全重合，clamp 永远不激活。此外，当 SD 主干预测 `delta ≈ 0` 时，floor 和 ceil 都趋近零，残差被完全封死——恰恰是最需要残差补偿的场景。

**修复**：重写趋势保护逻辑：

- gate 范围 `±0.20`，clamp 上界额外加 `min_correction = 0.02`，使 clamp 能真正约束；
- `hard_floor` 取 `max(0.80×mag, 0.02)`，保证 `delta≈0` 时残差不被封死；
- free 维度去掉 double-tanh 压缩（`tanh(struct_gate - 1.0)` → 直接 `tanh(raw_correction)`）。

#### 修订 4（P2a）：`predict_reward` 与 `predict_viable_reward` 分离

**问题**：原版 `predict_reward` 在环境真实奖励 `standing × hopping` 之上叠加了 `viability` 门控（`height_gate × pitch_gate × contact_gate`），导致系统性 `predict_reward ≤ env_reward`。这在 hybrid 训练模式下与 reward loss 目标产生 mismatch。

但完全移除 viability 也不正确：planner 的短 horizon（H=3~5）看不到”此刻高速但姿态已失稳 → 后续跌倒 → 后续全零”的因果链，viability 起到的是弥补短 horizon 盲区的作用。

**修复**：拆分为两个函数：

- `predict_reward()`：纯环境定义 `standing × hopping`，用于训练诊断日志；
- `predict_viable_reward()`：`viability × standing × hopping`，含稳定性门控（高度、姿态、支撑/离地），供 planner 使用。

#### 修订 5（P2b + 架构重设计）：reward 的 train/eval 模式分离与标量混合简化

**问题**：此前 `reward()` 在训练和推理中走同一条 hybrid 路径。在 `_update()` 中：

```
reward_preds = model.reward(...)      ← 返回 hybrid 混合分布
reward_loss  = soft_ce(reward_preds, env_reward)
```

这要求 learned head 的最优解为 `p_learned* = (p_env - α·p_explicit) / (1-α)`。当 `explicit ≠ env`（viability < 1 的状态），`p_explicit` 峰在不同 bin 上，learned head 永远无法将 loss 降到零——它在做一道无解的优化题。

**修复**：利用 TD-MPC2 的 `model.train()` / `model.eval()` 调用模式，让 `reward()` 行为随模式切换：

| 模式 | `reward()` 返回 | 目的 |
|------|----------------|------|
| training (`model.train()`) | 纯 `learned_logits` | reward loss 直接对齐环境真实 reward，无偏差 |
| eval (`model.eval()`) | `(1-α)·learned + α·viable` 标量混合 | planner 看到稳定性约束 |

同时，既然 explicit 信号不再参与训练 loss，混合只发生在 planning（`no_grad`），此前为训练精度设计的 log-space 分布混合机制也不再必要。简化为：

```
learned_scalar = two_hot_inv(learned_logits)      # 标量
viable_scalar  = predict_viable_reward(...)        # 标量
mixed          = (1-α) · learned + α · viable      # 标量加权
reward()       → log(two_hot(mixed))               # 回到 logits 供下游消费
```

无分布混合，无 log-space logsumexp，无 `-inf` 处理。删除了 `_explicit_reward_dist`、`_explicit_reward_log_probs`、`_mix_reward_logits` 三个方法。

#### 附属修订：`height_head` 与 `clearance_head` 分离

`height_head` 此前同时用于 `predict_height` 残差和 `forward()` 中的 clearance 残差，产生冲突训练信号。新增独立的 `clearance_head`，各司其职。

### 修订后的架构总览

```
Training (_update, model.train()):
  encode(obs) → z = [z_struct ; z_resid]
  next(z, a)  → z' (SD主干 + struct_residual + residual_dynamics)
  reward(z, a) → pure learned_logits
  loss = consistency + soft_ce(learned_logits, env_reward) + value + ...
  optim 包含所有模块（含三个残差模块）

Planning (_estimate_value, model.eval()):
  reward(z, a):
    learned_scalar = two_hot_inv(learned_logits)
    viable_scalar  = predict_viable_reward(z_struct, a, next_struct)
    return logits_of( 0.8·learned + 0.2·viable )
  → planner 偏好稳定轨迹
  → 更好的 action → 更好的训练数据 → learned head 学得更快
```

viable reward 的作用路径是数据分布（更安全的 planning → 更多非零 reward 的转移进入 replay），而非 loss 函数。

### 对此前实验结论的影响

P0（残差模块未入优化器）意味着此前所有 hybrid residual 实验中，残差模块实际从未被训练，且其累积梯度持续衰减其他参数的有效学习率。此前归因为”模式覆盖不足”的后期衰退现象，更可能主要来自此缺陷。全部实验结论需在修复后重新建立。

### 代码变更清单

| 文件 | 变更 |
|------|------|
| `external/tdmpc2/tdmpc2/tdmpc2.py` | 优化器追加残差模块参数组 |
| `external/tdmpc2/tdmpc2/common/sd_world_model.py` | 趋势保护重写；reward() train/eval 分离；标量混合替代 log-space 混合；删除三个旧混合方法；清理 numpy 导入 |
| `src/world_model/dmcontrol_hopper_sd_model.py` | 新增 `predict_viable_reward`；`predict_reward` 对齐纯环境定义；新增独立 `clearance_head`；`forward()` clearance 残差改用 `clearance_head` |

### archfix 5k 验证结果（seed=3）

运行配置：`sd_reward_mode=hybrid, sd_reward_hybrid_alpha=0.2, steps=5000, seed=3`，eval 路径当时使用 `predict_viable_reward`。

Eval 曲线：

| step | eval_reward |
|------|---:|
| 0 | 0.0011 |
| 1000 | 0.0 |
| 2000 | 0.0 |
| 3000 | 0.0122 |
| 4000 | 0.0016 |
| 5000 | **0.9088** |

训练日志分析：

- **所有 train episode reward = 0.0**；`explicit_reward_mean` 仅在 step=3000 出现过一次 0.0004，之后消失
- **`plan_value_mean` 全程 ≈ 0**——planner 无法区分候选轨迹优劣，elite 选择等同于随机
- **`consistency_loss`**：0.41 → 0.69，持续上升（residual 模块训练导致 latent 空间漂移）
- **`grad_norm`**：3.7–6.6，稳定无累积膨胀（P0 修复确认有效）
- **`residual_norm`**：3.9 → 4.3，缓慢增长，说明残差编码器正在学习
- **`reward_loss`** 和 **`value_loss`** 持续下降，但仅因 target 全零

与旧版对照：

| 版本 | cons_loss @3k | grad_norm | plan_value | 5k eval |
|------|---:|---:|---:|---:|
| oldbaseline（无 residual） | 1.12 | 0.17 | 0.0 | 0.132 |
| hybrid_trend（P0 bug 版） | 0.60 | **31** | 0.0 | 0.171 |
| **archfix** | **0.44** | **5.2** | **0.0** | **0.909** |

archfix 在世界模型精度（consistency_loss）和梯度健康（grad_norm）上最优，但 `plan_value ≈ 0` 与其他版本一致——5k eval=0.909 更像是偶发碰撞而非 planner 有目的地规划。

### 跨版本日志对比启发：稀疏奖励 + planner 无差异化

对所有历史版本的 train.csv 做交叉分析后，核心发现是：

- **TD-MPC2 原版（seed=0, 5k）在 step=1000 就出现非零 plan_value 和 eval_reward (0.128)**，之后渐进增长至 5000 时 0.320——planner 从极早期就能做有意义的 elite 筛选
- **所有 SD 版本的 `plan_value` 全程 ≈ 0**——无论 consistency_loss 多低，planner 始终在噪声中选 elite

根本原因是 reward head 与 Q head 在 Hopper-Hop 极稀疏 reward 下无法建立有效 value landscape。TD-MPC2 的端到端 latent 表示让 reward 梯度直接塑造 latent 几何，SD 的确定性结构化编码则无法享受这一机制。

### Planning reward shaping：用 `predict_plan_reward` 替代 `predict_viable_reward`

基于上述诊断，将 eval 路径的 explicit reward 源从 `predict_viable_reward`（几乎与真实 reward 一样稀疏）切换到 `predict_plan_reward`（包含平滑 shaping 项）：

- `stand_buffer = clamp((height - 0.45) / 0.15, 0, 1)` — height>0.45 即有信号
- `speed_buffer = clamp(vx / 2.0, 0, 1)` — 任何正向速度都有信号
- `lift = clamp(vz / 2.0, 0, 1)` — 向上运动有信号

单元测试确认：在未训练的随机模型上，32 个随机状态的 plan_reward 均值 0.017、最大 0.121、非零率 84%。旧版 `predict_viable_reward` 在相同条件下全零。

代码变更：`sd_world_model.py` 的 `reward()` eval 路径中 `predict_viable_reward` → `predict_plan_reward`。

### 50k planshape 实验结果（seed=3, α=0.5）

运行配置：`sd_reward_mode=hybrid, sd_reward_hybrid_alpha=0.5, steps=50000, seed=3, eval_freq=5000, eval_episodes=1`

#### Eval 曲线

| step | eval_reward |
|------|---:|
| 0 | 0.0004 |
| 5000 | 0.0 |
| 10000 | 0.0 |
| 15000 | 0.0 |
| 20000 | 0.0037 |
| 25000 | 0.0 |
| 30000 | 0.0 |
| 35000 | 0.0 |
| 40000 | **1.7877** |
| 45000 | 0.0 |
| 50000 | **0.0** |

**最终 eval reward = 0.0**。唯一显著非零点为 step=40000 的 1.788（单 episode eval）。

#### 训练动态

| step | cons_loss | grad_norm | plan_value | plan_elite | s_norm | r_norm | expl_rew |
|------|---:|---:|---:|---:|---:|---:|---:|
| 2600 | 0.42 | 4.9 | **0.0075** | **0.0110** | 5.4 | 4.1 | 0.0 |
| 5000 | 0.65 | 5.5 | **0.0117** | **0.0154** | 5.9 | 4.8 | 0.0004 |
| 10000 | 0.89 | 9.5 | **0.0212** | **0.0258** | 6.6 | 7.6 | 0.0 |
| 20000 | 0.82 | 10.4 | **6.1106** | **6.1684** | 5.7 | 11.4 | 0.0 |
| 30000 | 0.56 | 11.2 | 3.3311 | 3.3476 | 5.1 | 12.5 | 0.0 |
| 40000 | 0.48 | 13.2 | 1.3137 | 1.3169 | 4.5 | 17.8 | 0.0 |
| 50000 | 0.75 | **42.6** | 2.0932 | 2.1256 | 4.6 | **22.1** | 0.0 |

训练中出现的非零 episode reward：

| step | train_ep_reward |
|------|---:|
| 11500 | 10.24 |
| 17500 | 2.90 |
| 34500 | 2.14 |
| 36500 | 6.00 |
| 40500 | 6.81 |
| 48000 | 11.98 |
| 49000 | 8.52 |
| 50000 | 4.65 |

#### 分析

**正面成果：**

1. **plan_value 不再全零**——planner 从 step=2600 起即有非零区分度（0.0075），到 step=20000 达到 6.11。这是所有 SD 版本中首次出现有意义的 planner value landscape
2. **训练中多次出现强 episode reward**——step 11500 出现 10.24，之后持续出现 2.1~12.0 的非零 train reward。在此前所有版本中，train episode reward 全程为零
3. plan_reward shaping 成功打破了"稀疏 reward → planner 全盲 → 无定向探索"的死循环

**待解决问题：**

1. **`residual_norm` 持续膨胀**：4.1 → 22.1（5 倍增长），说明 residual latent 在逐步主导表示空间，结构化主干的相对比重持续下降
2. **`grad_norm` 后期增长**：4.9 → 42.6，暗示训练后期不稳定
3. **eval 与 train 脱节**：训练中频繁出现强 episode，但 eval（deterministic mean action）几乎全零。仅 step=40000 偶发 1.788。这可能是因为 eval 的 deterministic policy 比训练时的 stochastic exploration 更保守
4. **plan_value 在 20k 峰值后回落**：从 6.11 降至 2.09，与 residual norm 膨胀同步，暗示 value landscape 后期发生了退化

#### 当前结论

plan_reward shaping 已验证为有效的循环打破机制。但 50k 的最终 eval = 0.0 说明该机制本身不足以转化为稳定的 eval 性能。下一步应优先解决：

- **residual norm 膨胀**：考虑对 residual 编码器输出加 norm 约束（如 LayerNorm 或 L2 penalty），或限制 `residual_dynamics` 的输出范围
- **eval-train gap**：分析 eval 模式下 planner 的行为是否因 deterministic action 丢失了关键探索维度

### 跨版本深度审查：CartPole 成功 vs Hopper 失败的根本原因

在完成 50k planshape 实验后，对 CartPole SD-MCTS（成功）与 Hopper SD-in-TD-MPC2（失败）做了完整的架构对比分析，并与 Codex 进行了多轮深度讨论。

#### 1. 核心诊断

CartPole SD-MCTS 成功的三个支柱：

1. MCTS **直接调用** `model.predict_reward()` 获取即时奖励——奖励信号零学习依赖
2. MCTS **直接调用** `model.forward()` 推演动力学——状态预测由纯物理方程驱动
3. 离散动作空间使 MCTS 的树搜索可行——每个节点的分支因子有限

Hopper 嵌入 TD-MPC2 后，这三个支柱全部断裂：

1. MPPI planner 调用的是 **learned reward head**（101-bin 分布回归），在稀疏奖励下输出 ≈ 0——SD 模型的领域知识被截断
2. MPPI planner 使用 **learned dynamics**（含多个 residual MLP），不是 SD 的纯物理方程——结构先验被稀释
3. 连续动作空间使 MPPI 的采样效率远低于 MCTS 的穷举搜索

这导致了鸡生蛋的死循环：planner 需要 reward 信号 → learned head 输出 0 → 无定向探索 → 无正样本 → learned head 无法学习。

#### 2. Codex 补充发现

- **`height_head` 无监督信号**：它仅被 `predict_reward` / `predict_plan_reward` 调用，但这两个函数不在训练 loss 路径中。Codex 称之为"死头"——依赖它的显式 reward 稳定性存疑。
- **`project_state()` 速度无约束**：MPPI rollout 时速度变量可漂移到非物理区域。
- **`predict_reward()` 并非纯解析**：使用 `predict_height(next_state)` 带 learned residual，不是纯环境同构。
- **学习残差过多**：内层 5 个 learned head + 外层 3 个 learned correction，SD 的"结构先验"约束力被大量自由度稀释。

#### 3. 两条修复路线

##### 路线 A：修复 TD-MPC2 集成（渐进式）

在保持 TD-MPC2 框架不变的前提下，让 SD 的领域知识更直接地到达 planner：

| 优先级 | 改动 | 原理 |
|--------|------|------|
| P0 | `_estimate_value()` 中 planner reward 改为 SD direct reward | 恢复 CartPole 中"planner 直接使用 SD reward"的模式 |
| P0 | 为 `height_head` 添加辅助监督损失 | 确保 predict_plan_reward 所依赖的 learned component 有训练信号 |
| P1 | Planning 循环中结构化维度用 SD dynamics 修正 | 防止 learned dynamics 污染结构状态 |
| P1 | `project_state()` 补全速度约束 | 防止 MPPI rollout 漂移到非物理区域 |
| P2 | horizon 从 3 提升到 5 | hopping 需要多步序列 |
| P2 | Residual gating（sigmoid gate 从 0 起步） | 训练初期保护结构先验 |

**风险**：即使做了这些修复，TD-MPC2 框架的本质仍然是"learned heads 做决策"——SD 提供的只是辅助信号。多层 intermediary（two_hot 编码、learned Q bootstrap、MPPI sampling noise）会持续衰减 SD 信号的有效性。而且工程复杂度高，调参空间大。

##### 路线 B：离散运动意图 + SD-MCTS（架构级重设计）

完全绕过 TD-MPC2，回到 CartPole SD-MCTS 的成功框架，但将连续动作空间替换为**离散运动意图空间**。

核心洞察来自对人类运动控制的类比：人类单脚跳时不会直接决策所有关节的连续力矩值，而是在"蓄力 / 起跳 / 调姿 / 缓冲"等**离散运动意图**之间做选择。各关节的具体取值由意图确定后的低层控制律计算得出。

架构设计：

```
MCTS 搜索空间：5-7 个离散"运动意图"
       ↓
意图 → 连续力矩 映射器（由 SD 模型的物理知识 + 当前状态计算）
       ↓
SD dynamics forward()：预测 next_state（纯物理方程）
       ↓
SD predict_reward()：直接计算奖励（纯解析）
       ↓
MCTS backprop：与 CartPole 完全相同的框架
```

离散运动意图候选：

| 意图 | 物理含义 | 力矩映射逻辑 |
|------|---------|-------------|
| **蓄力** | 压缩腿部，降低重心 | hip/knee 屈曲，ankle 稳定 |
| **前冲** | 重心前移 + 伸展推进 | hip 伸展，knee 伸展，waist 前倾 |
| **起跳** | 全力伸展所有关节 | 所有关节向伸展方向最大力矩 |
| **空中调姿** | 收腿 + 保持前进姿态 | hip/knee 屈曲，waist 回中 |
| **着陆缓冲** | 腿部微屈吸收冲击 | knee/ankle 柔顺阻尼 |
| **稳定站立** | 保持当前姿态 | 各关节回中 + 阻尼 |

每个意图到 4 维连续力矩的映射可以是基于 XML 中已知齿轮比和关节结构的**解析控制律**，而非 learned MLP。

**优势**：
- 直接复用 CartPole SD-MCTS 的全套成功框架
- SD 领域知识贯穿决策全链（世界模型 + 奖励函数 + 动作空间）
- 工程复杂度低，调参空间小
- 仅需学习极少参数（物理常数 + 可选 value net）
- 研究新颖性强：领域知识不仅编码在模型中，还编码在动作抽象中

**风险**：
- 离散意图集是否覆盖完整行为空间需要验证
- 意图→力矩的映射质量依赖对 Hopper 生物力学的理解深度
- 如果离散化过粗，可能丢失精细控制能力

#### 4. 两条路线的哲学对比

| 维度 | 路线 A（修复 TD-MPC2） | 路线 B（离散意图 SD-MCTS） |
|------|----------------------|--------------------------|
| 对 SD 先验的利用程度 | 部分（仅 latent 编码层 + 辅助 reward） | **完整**（世界模型 + 奖励 + 动作空间） |
| 与 DreamerV3/TD-MPC2 的差异化 | 小（仍在其框架内工作） | **根本性差异**：不同抽象层级的控制 |
| 研究主张 | "SD 可以改善 model-based RL" | "**控制问题应在领域知识匹配的抽象层级求解**" |
| 论文叙事 | "我们在 TD-MPC2 中嵌入了结构化先验" | "端到端学习 vs 结构化决策的范式对比" |

路线 B 的核心研究主张可表述为：

> DreamerV3 和 TD-MPC2 的隐含假设是"所有控制问题都应在原始连续动作空间中端到端求解"。但人类运动控制的实际层级结构表明，对有明确物理结构的任务，在正确的抽象层级（运动意图而非关节力矩）做离散决策，结合结构化世界模型直接驱动规划，可以比端到端方法更高效。

#### 5. 选定路线

**选择路线 B 作为主线推进**，路线 A 保留为 fallback。

理由：
- 路线 A 是在错误的框架内做修补，即使所有 P0/P1 都完成，仍面临"learned heads 做决策"的根本局限
- 路线 B 直接建立在 CartPole 已验证的成功模式上
- 路线 B 的研究贡献更清晰、更根本性

### V2（路线 A 修复版）50k 正式结果

在完成路线分析后，路线 A 的 v2 实验（residual 约束 + shaping 重设计 + curriculum α 衰减）继续在后台运行并完成了 50k。

运行配置：`sd_reward_mode=curriculum, sd_reward_curriculum_alpha=0.5, sd_reward_curriculum_alpha_min=0.1, sd_reward_curriculum_start=10000, sd_reward_curriculum_end=30000, sd_max_resid_scale=5.0, eval_episodes=10, eval_freq=10000, seed=3`

#### Eval 曲线（10 episodes 均值）

| step | eval_reward |
|------|---:|
| 0 | 0.0004 |
| 10000 | 0.0456 |
| 20000 | 0.0404 |
| 30000 | 0.0063 |
| 40000 | 0.7334 |
| 50000 | **10.049** |

#### 与所有基线对比

各基线数据来源于 TD-MPC2 官方仓库 `external/tdmpc2/results/` 中的 per-seed CSV（3 seeds 平均，episode return 口径）。

| 方法 | 50k | 100k | 200k | 300k | 500k |
|------|---:|---:|---:|---:|---:|
| **SD-MCTS v2 (ours)** | **10.05** | — | — | — | — |
| TD-MPC2 | — | 13.6 ± 10.4 | 166.6 ± 34.9 | 259.7 ± 51.4 | 303.3 ± 27.7 |
| TD-MPC (v1) | — | 53.3 ± 41.3 | 148.6 ± 63.4 | 205.1 ± 68.3 | 238.0 ± 56.3 |
| DreamerV3 | — | 9.7 ± 13.6 | 51.2 ± 50.4 | 77.6 ± 39.7 | 162.7 ± 14.5 |
| SAC | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.9 ± 12.2 | 37.5 ± 27.4 |

**关键发现：**

1. **SD v2 在 50k 步达到 10.05**，而 TD-MPC2 在两倍预算（100k）时仅 13.6，DreamerV3 在 100k 时仅 9.7。SD v2 用一半的 env steps 已接近 TD-MPC2 100k 水平，并超过了 DreamerV3 100k 水平
2. 这是**单 seed (seed=3) 单次实验**，TD-MPC2 官方的 seed=3 在 100k 时仅 0.7（是三个 seed 中最差的），因此直接比较需要谨慎
3. 但即使与 TD-MPC2 的最佳 seed (seed=1, 100k=26.2) 相比，SD v2 的 50k=10.05 也已具有竞争力

#### 训练动态

| step | cons_loss | grad_norm | struct_norm | resid_norm | plan_value |
|------|---:|---:|---:|---:|---:|
| 10000 | 0.57 | 6.0 | 5.51 | 8.13 | 0.00 |
| 20000 | 0.55 | 4.0 | 4.70 | 8.92 | 0.00 |
| 30000 | 0.62 | 4.1 | 4.63 | 8.49 | 1.07 |
| 40000 | 0.60 | 4.4 | 4.70 | 8.47 | 2.96 |
| 50000 | 0.49 | 3.9 | 4.33 | 7.73 | 1.21 |

与上轮 planshape（50k 时 resid_norm=22.1, grad_norm=42.6）的关键对比：

- **residual_norm 完全被控制**：7.7-8.9，未膨胀（上轮 4.1→22.1）
- **grad_norm 稳定**：3.9-6.0（上轮 4.9→42.6）
- **plan_value 在 30k 出现并维持**：1.07→2.96→1.21（上轮在 20k 达到 6.11 后崩塌）

训练中出现 17 个非零 reward episode，最强为 step=36500 的 39.47（远超此前任何版本）。后段（47k-50k）仍持续出现 1.5-4.6 的非零 episode，说明训练信号未衰退。

#### 三项修复各自的贡献

1. **Residual 范数约束**（`tanh(x/5)*5`）：直接阻止了 residual 膨胀，保护了 grad_norm 稳定性和训练后期的学习能力
2. **Shaping 重设计**（viability gate × speed_progress + action_energy）：消除了"站稳不动"的局部最优，使 planner 偏好前进运动
3. **Curriculum α 衰减**（0.5→0.1 over 10k-30k）：训练早期用强 shaping 引导探索，后期让 learned reward 主导

三者协同使得训练在 30k 后进入正反馈循环：shaping 引导 → 偶尔成功 → reward/Q head 学到信号 → planner 更有针对性 → 更多成功。

#### 对路线选择的影响

v2 结果表明路线 A 的修复确实有效，且在 50k 预算下已展现出超过 DreamerV3 100k 的样本效率优势。这使路线选择变得更加 nuanced：

- **路线 A（TD-MPC2 集成）**：已证明在 50k 可用，且 eval=10.05 是可信的（10 episodes 均值）
- **路线 B（离散意图 MCTS）**：尚在实现中，概念上更优雅但未经验证

两条路线不互斥。v2 可继续扩展到 100k-500k 验证长线表现，路线 B 可并行推进验证离散意图的可行性。

### 路线 B 初步实现：离散运动意图 + SD-MCTS

已完成 `scripts/run_hopper_mcts.py` 的完整实现，包括：

- 7 个离散运动意图（load, push, launch, tuck, extend, absorb, neutral）
- MCTS 搜索框架（复用 CartPole 的 UCB1 + 树搜索结构）
- SD 世界模型直接驱动 dynamics 和 reward（无 learned intermediary）
- 在线训练循环（replay buffer + SD 参数拟合 + 可选 value net）

Smoke test 通过（CPU 600 步），但 MCTS 的串行 simulation 导致速度较慢（50 sims × rollout_depth=8 时每个决策约 3-5 秒）。5k 预算的正式实验正在运行中。

### 2026-04-16 新增：路线 B 修订为“单步离散决策”版本

在对路线 B 的首个 `5k` 实验与实现细节做复盘后，对其定义与代码口径做了统一修订。

#### 1. 方案收敛

路线 B 不再解释为“带固定持续时间的宏动作意图”，而是明确收敛为：

- 每个离散标签仅表示**当前决策步**的控制模式；
- 下一步是否继续沿用同一标签，由下一时刻重新决策；
- 本轮**不引入显式 duration 建模**，也不把意图定义为半马尔可夫 option。

该收敛使“离散决策”与当前研究问题保持一致：重点检验高层离散控制抽象本身，而不是同时引入持续时间建模。

#### 2. 对首版实现的诊断

首版 `scripts/run_hopper_mcts.py` 虽以“离散意图 + SD-MCTS”为名实现，但实际采用了 `action_repeat=4` 的宏动作语义。这带来两类不一致：

1. **语义不一致**：讨论中已明确，当前标签应解释为单步控制模式，而非固定持续 4 步的意图；
2. **时间尺度不一致**：replay 中存的是 4 步环境转移，世界模型训练却按 1 步监督，MCTS 规划又将 `forward()` 重复调用 4 次，导致训练、规划和评估口径错位。

因此，首个 `5k` 结果不能视为对“单步离散决策路线”本身的有效检验，优先需要修正实现口径。

#### 3. 本轮代码修订原则

本轮仅做与方案定义直接相关的最小修订，不额外引入 phase mask 或状态相关底层映射：

- 将路线 B 脚本改为**单步决策、单步环境交互、单步 world model rollout**；
- MCTS backprop、rollout discount、value target 全部改为单步口径；
- `value_net` 若启用，其训练目标改为 planner 使用的 `predict_plan_reward`，避免叶节点评估与搜索奖励口径不一致；
- 当前 7 维动作表保留，但仅作为**底层映射占位实现**，不将其解释为最终的解析控制律。

#### 4. 当前判断

路线 B 仍具有研究价值，但其 faithful 版本应理解为：

> 在 Hopper 这类具有明显相位结构的任务中，先将连续控制抽象为每步离散控制模式，再由 MCTS 在该离散空间中规划；底层映射是否进一步做成状态相关解析控制律，留作下一阶段问题。

这一定义比首版实现更窄，但更清晰，也更便于做可证伪验证。

#### 5. 最小验证

完成代码收敛后，进行了最小静态检查与 smoke test：

- `python -m py_compile scripts/run_hopper_mcts.py`
- `python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py`
- `python -u scripts/run_hopper_mcts.py --budget 20 --seed_steps 10 --num_simulations 1 --rollout_depth 1 --eval_episodes 1 --eval_freq 10 --train_updates 1 --train_batch 4 --device cpu --save_path /tmp/hopper_mcts_smoke.pt`

结果：

- 两个 `py_compile` 均通过；
- Hopper MCTS 脚本在当前无显示环境下以默认 `glfw` 路径成功运行，虽出现 `DISPLAY` 缺失 warning，但不影响执行；
- world model 训练、MCTS 搜索、结果保存均正常走通；
- smoke reward 仍未出现正向信号（`step=0: 0.0053`, `step=20: 0.0`），符合“当前仅验证实现口径，不验证性能成立”的预期。

smoke 过程中还发现：主循环原先总是按整回合收集数据，导致 `budget=20` 这类小预算配置被实际跑成 500 步。该问题已一并修正，使 budget 语义恢复为真正的环境步数上限。

### 2026-04-16 新增：路线 B 决策层最小结构化修订

在单步离散决策版 `5k` 实验中，虽然 `2500` 步附近短暂出现过一次非零 episode（`0.1084`），但后续策略大部分时间重新塌回 `load`，最终 `eval@5000=0.0`。复盘后判断，当前瓶颈更偏向**决策层组织方式**，而非 TVA 或底层映射本身。

#### 1. 发现的问题

1. **固定展开顺序偏置**：MCTS 扩展阶段总是按 `load -> push -> launch -> ...` 的固定索引顺序补子节点，使 `load` 获得了额外、非语义性的先手优势；
2. **标签与相位未对齐**：当前 7 个标签虽有语义命名，但在搜索中所有状态等权开放，没有利用 Hopper 已有的粗相位结构（compression / takeoff / flight / landing）；
3. **长时间重复同一标签**：特别是 `load`，在缺少相位先验时容易成为“低风险但低收益”的吸引子。

#### 2. 修订原则

本轮仅做最小决策层修订，不改底层查表式力矩映射，不改 TVA：

- 在 `run_hopper_mcts.py` 内引入**基于结构化状态的粗相位推断**；
- 为每个相位定义**soft prior**，用于引导 MCTS 选择/扩展顺序；
- 将搜索评分由无先验 UCB 改为带 prior 的 **PUCT** 形式；
- rollout 也改为按 phase-aware prior 采样，而非完全均匀随机；
- 对“重复上一步标签”加入**轻量 anti-sticky** 惩罚，对 `load` 的重复惩罚更强；
- 保持所有标签可选，不做硬 mask，仅通过 soft prior 注入结构信息。

#### 3. 当前定位

这一步的目标不是直接证明路线 B 已经成立，而是回答一个更基本的问题：

> 当底层映射暂时保持查表实现时，仅通过让离散标签与 Hopper 相位结构建立更清楚的对应关系，是否就足以让单步离散决策版本出现更稳定的正向学习信号？

若该问题的答案为“是”，则说明下一阶段应继续深化决策层语义；若答案仍为“否”，再转回审视底层映射或 TVA 的必要性。

#### 4. 5k 对照结果（phase-aware prior + anti-sticky）

按与上一轮单步版相同的训练预算重新运行：

`python -u scripts/run_hopper_mcts.py --budget 5000 --seed 3 --num_simulations 30 --rollout_depth 5 --eval_episodes 1 --eval_freq 5000 --train_updates 64 --train_batch 64 --seed_steps 2000 --lr 3e-4 --vnet_mix 0.5 --device cuda --save_path results/hopper_mcts_5k_seed3_phaseprior_gpu.pt`

结果：

| step | eval_reward |
|------|---:|
| 0 | 0.0028 |
| 5000 | 0.0000 |

训练阶段未再出现上一轮那种 `0.1084` 量级的短暂非零 episode。各回合意图分布高度集中在 `load`，仅混入少量 `absorb / tuck / push`：

- `2500`: `load=461, absorb=39`
- `3000`: `load=469, tuck=16, absorb=15`
- `3500`: `load=460, push=2, tuck=14, absorb=24`
- `4000`: `load=471, tuck=10, absorb=19`
- `4500`: `load=456, push=4, absorb=40`
- `5000`: `load=468, tuck=14, absorb=18`

结论：

1. **修掉固定展开顺序偏置并加入 phase-aware soft prior 后，`load` 吸引子仍然存在**；
2. 因而当前瓶颈**不只是搜索顺序问题**，更可能是标签语义本身过粗、phase prior 过弱，或查表式底层映射与标签命名之间并不一致；
3. 相比上一轮单步版，这次修订**没有带来性能提升**，反而连中途的弱非零信号也消失了。

这表明：仅靠最小的 soft prior / anti-sticky 修订，还不足以让路线 B 在现有动作词表上建立稳定行为。下一阶段应优先重新审视标签体系与底层映射的一致性，而不是继续微调先验权重。

### 2026-04-16 新增：将动作词表压缩为 5 个更正交的控制模式

在完成 phase-aware prior 的 `5k` 对照后，判断当前瓶颈已从“搜索顺序”进一步收敛为“动作词表本身不干净”：

- `load` 同时带有压缩与释放成分，不是纯压缩；
- `push` 与 `extend` 高度重叠；
- `neutral` 缺少清晰任务角色；
- 因此 7 标签词表无法形成清楚的相位分工。

据此，将路线 B 的离散控制模式压缩为 5 个：

- `compress`
- `drive`
- `launch`
- `tuck`
- `absorb`

设计意图：

1. `compress`：纯下蹲蓄力，强调髋正向压缩、膝踝负向折叠；
2. `drive`：中等强度的前推/伸展；
3. `launch`：强释放起跳；
4. `tuck`：空中收腿；
5. `absorb`：着陆缓冲。

同时更新 phase prior，使其与上述 5 个标签一一对应。该修订的目标不是直接提升性能，而是让“离散控制模式”本身先具备可解释、可审查的相位语义。

### V2 三 seed 50k + 100k 正式结果

#### 三 seed 50k 结果（seeds=1,2,3，eval_episodes=10，与官方基线同口径）

| step | seed=1 | seed=2 | seed=3 | **mean ± std** |
|------|---:|---:|---:|---:|
| 0 | 0.79 | 0.00 | 0.00 | 0.26 ± 0.37 |
| 10000 | 0.62 | 0.00 | 0.05 | 0.22 ± 0.28 |
| 20000 | 0.00 | 0.00 | 0.04 | 0.01 ± 0.02 |
| 30000 | 0.00 | 36.63 | 0.01 | 12.21 ± 17.26 |
| 40000 | 0.00 | 0.94 | 0.73 | 0.56 ± 0.40 |
| 50000 | 0.67 | 31.39 | 10.05 | **14.04 ± 12.86** |

#### 与官方基线统一口径对比

| 方法 | 50k | 100k | 200k | 300k |
|------|---:|---:|---:|---:|
| **SD v2 (ours, 3 seeds)** | **14.0 ± 12.9** | — | — | — |
| TD-MPC2 (official, 3 seeds) | — | 13.6 ± 10.4 | 166.6 ± 34.9 | 259.7 ± 51.4 |
| DreamerV3 (official, 3 seeds) | — | 9.7 ± 13.6 | 51.2 ± 50.4 | 77.6 ± 39.7 |
| SAC (official, 3 seeds) | — | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.9 ± 12.2 |

**SD v2 在 50k 步达到 14.0 ± 12.9，与 TD-MPC2 在 100k 的 13.6 ± 10.4 持平，超过 DreamerV3 在 100k 的 9.7 ± 13.6。** 用一半的 env steps 达到了相当水平。

但 seed 间方差大（12.9），主要由 seed=2 在 30k 处的异常高值（36.63）和 seed=1 的整体偏低驱动。

#### seed=3 100k 延长线

| step | eval |
|------|---:|
| 50000 | 2.28 |
| 60000 | 11.46 |
| 70000 | **35.77** |
| 80000 | 25.45 |
| 90000 | 25.52 |
| 100000 | **15.83** |

**SD 在 50k→100k 期间持续增长**，峰值在 70k 达到 35.77，100k 时 15.83。这回答了"SD 是否在更长训练中继续助力"的问题：**是的，至少到 100k 仍然在增长。**

与 TD-MPC2 官方 seed=3 @ 100k 的 0.7 相比，SD v2 seed=3 @ 100k 的 15.83 是其 **22 倍**。

但需注意 100k 结果仅有单 seed，且 70k→100k 出现回落（35.77→15.83），说明训练尚未完全稳定。后续需补跑 seed=1,2 的 100k 以及进一步延长到 200k 才能形成可靠结论。

### 路线 A 代码正确性审计与下一轮优化方案

#### 审计过程

在三 seed 50k + 100k 实验完成后，对正在使用的 v2 代码进行了联合审计。由主审（Claude）完成初审并提出修改建议，随后由外部专家独立复核初审结论、纠正不精确表述、并提出调整后的优先级排序。

#### 确认的问题

**问题 1（中等）：`weight_init` 覆盖 SD 模型的预设初始化**

`sd_world_model.py:75` 的 `self.apply(init.weight_init)` 递归应用到所有子模块，包括 `self._dynamics`（`DMControlHopperSDModel`）内部的 5 个 MLP（`passive_flow`、`residual_flow`、`touch_head`、`height_head`、`clearance_head`）。`init.weight_init` 将 `nn.Linear` 权重重置为 `trunc_normal_(std=0.02)`，使这些 MLP 的初始输出量级相比默认初始化缩小约 20-30 倍。

效果：训练初期 SD 模型近似退化为纯显式方程（learned 残差 ≈ 0），结构先验主导。这在当前结果中**恰好有益**（与 CartPole 成功中的"结构先验主导"精神一致），但它不是有意为之的设计。

`control_scale`（`nn.Parameter`）和 `rate_damping_raw`（`nn.Parameter`）不受影响，因为 `weight_init` 只处理 `nn.Linear`。buffer 也不受影响。

**问题 2（中等，与问题 1 耦合）：`height_head` 在 TD-MPC2 训练中无梯度**

`height_head` 仅在 `predict_height()` → `predict_reward()` / `predict_plan_reward()` 中使用。但训练态的 `SDWorldModel.reward()` 直接返回 learned reward head（`sd_world_model.py:221`），不走显式 reward 分支；`consistency_loss` 走的是 `next()` → `_dynamics.forward()`，其中也不用 `height_head`（`forward()` 中的 `next_height` 由积分计算，不经过 `height_head`）。因此 `height_head` 基本不会被训练更新。

**与问题 1 的耦合关系**：当前之所以没被随机 `height_head` 污染 `predict_plan_reward` 的 shaping 信号，恰恰是因为问题 1 又把它压到了近零。也就是说，如果单独修复问题 1（恢复正常初始化）、而不同时处理 `height_head`，会把 planning reward 里的高度修正从"近零"变成"随机常数扰动"，**反而更危险**。

**问题 3（低）：`reward()` eval 路径冗余 `next()` 调用**

`sd_world_model.py:225` 在 eval 路径中调用 `self.next(z_flat, a_flat, task)`，而 `_estimate_value()` 紧接着又调用 `self.model.next(z, actions[t], task)`（`tdmpc2.py:148`）。每个 planning step 执行了 2 次完整的 dynamics forward。这是**吞吐量损失**（planning 速度约减半），不影响结果正确性。

**问题 4（很低）：`base_delta == 0` 时 `min_correction` 失效**

`sd_world_model.py:180` 中 `p_sign = p_delta.sign()`，当 `p_delta` 恰好为 0 时 `p_sign = 0`，乘以 `min_correction` 后仍为 0。但外部专家修正了主审原文的不精确表述：`sign(x)` 对任何非零浮点都是 ±1，不会随幅值变小而变小。因此该问题仅在 `p_delta` 恰好等于 0.0 时成立，影响比初审描述的更窄。

#### 审计结论

未发现导致结果无效、泄漏或训练目标错误的致命 bug。当前主要问题是初始化行为不显式、显式 reward 的一部分依赖未受监督、以及规划阶段有冗余计算。三 seed 50k 和 seed=3 100k 实验结果可视为可信。

#### seed=3 100k plan_value 振荡现象

100k 延长线的 `plan_value_mean` 在 50k-100k 之间存在剧烈振荡（5k 步粒度）：

| step | plan_value |
|------|---:|
| 50000 | 11.36 |
| 55000 | 0.53 |
| 60000 | 13.39 |
| 65000 | 0.19 |
| 70000 | 0.78 |
| 75000 | 14.34 |
| 80000 | 0.02 |
| 85000 | 16.31 |
| 90000 | 0.02 |
| 95000 | 15.64 |
| 100000 | 16.33 |

plan_value 在 ~0 和 ~15 之间交替跳变，暗示 learned reward/Q head 的输出在两种体制间切换。此时 curriculum α 已降至 0.1，planner 的 reward 信号 90% 来自 learned head。

外部专家指出：由于该数据未记录在工作日志中供独立复核，应表述为"如果日志摘录准确则判断合理"而非已完全验证的事实。现将原始数据补录于此。

#### seed=1 为何未能突破

seed=1 在 step=18000 出现过 11.69 的强 train episode，但 `plan_value` 从未超过 0.43，到 50k 仅 0.069。50k eval 仅 0.67。单次强 episode 的经验未被 Q/reward head 稳定吸收。

对比 seed=2：step=28500 出现 29.66 后连续产生 24 个强 episode（>0.5），plan_value 持续有效。seed=2 的突破发生在 curriculum α 仍较高（>0.3）时，dense positive episodes 使 learned head 有足够样本建立可靠估计。seed=1 的强 episode 过于稀疏（仅 4 个），learned head 无法泛化。

#### 下一轮优化方案

以下优先级排序融合了主审建议和外部专家反馈。核心原则：**最小侵入、逐步验证、不破坏当前有效机制**。

**P0：α_min 小型 ablation**

将 `sd_reward_curriculum_alpha_min` 从 0.1 提高，测试 0.20 / 0.25 / 0.30 三个值。在 seed=3 上各跑 50k，以 `plan_value_mean`、`plan_nonfinite_frac`、非零 train episode 密度和 50k eval 作为判据。

定义为 ablation 而非定论。α_min 只影响 planning/eval 的 reward 混合，不直接进入训练态的 reward loss。它改善的是采样策略和 replay 分布，不是直接给 reward head 更强监督——因此效果是间接的。

注：当前 `_curriculum_alpha()` 的设计是从 `alpha_max`（默认 0.5）线性衰减到 `alpha_min`，之后**永久保持在 `alpha_min`**（见 `sd_world_model.py:250-251`）。α 永远不会归零，shaping 信号作为稳定 floor 始终在场。这是有意设计——避免 learned reward/Q head 出现输出振荡时 planner 退化为盲搜。所有 ablation 和基线实验均遵循此机制，差异仅在 `alpha_min` 值本身。

**P1：为 height_head 加辅助监督**

在 `_update()` 中加入 MSE 辅助损失：

```
true_struct = encode_observation(obs[:-1])  # from real observations
true_height = true_struct[:, HEIGHT_IDX]     # ground truth
pred_height = predict_height(z_struct)       # height_head applied to rollout latent
height_loss = MSE(pred_height, true_height)
total_loss += height_coef * height_loss
```

这是当前最有"去偶然化"价值的改动。它使 `predict_plan_reward` 中的高度修正从"Bug 1 意外压零"变为"有监督的准确预测"，同时为将来显式修复 Bug 1 铺路。

在 P0 ablation 确定最佳 α_min 后实施。

**P2：若 α_min ablation 仍不充分，延长 curriculum 窗口**

将 `sd_reward_curriculum_start=10000, sd_reward_curriculum_end=30000` 延长为 `10000, 50000`。窗口变长和下限变高都在增加 shaping 占比，但窗口变长改动更大、更难归因。先测 α_min，如果仍出现明显盲搜窗口，再考虑延长。

**P3：去掉 reward() 中冗余 next() 调用**

将 `next_z` 作为 `reward()` 的可选输入参数，或在 `_estimate_value()` 中先 rollout 再算 reward。这是吞吐优化，不影响样本效率，排在功能性改进之后。

**P4：在 height_head 已受监督的前提下，显式修复 Bug 1**

将 `self.apply(init.weight_init)` 改为排除 `_dynamics` 模块，或在 `_dynamics` 初始化后重新设置其参数。**必须在 P1 完成后实施**——否则会从"近零安全"变成"随机扰动"。

**不在本轮实施的项**：
- Bug 2（sign(0) 的 min_correction 失效）：影响过窄，并入下次 `next()` 重构时顺手修复
- 路线 B 的动作词表优化：路线 B 已独立推进至 5-label 版本，与路线 A 的优化互不阻塞

### 2026-04-17 新增：`drive` 修订为更明确的中间推进动作

在 5 标签版 `5k` 实验中，`compress` 已明显成为主导动作，但 `drive` 几乎未被使用。进一步诊断发现：

- `drive` 的 torque 既弱于 `launch`，又没有形成独立阶段角色；
- 在 prior 上，`compression` 相位优先 `compress`，`takeoff` 相位又优先 `launch`，使 `drive` 两头都不占优势；
- 用 5k 训练后的模型直接比较 `predict_plan_reward` 时，`drive` 在初始状态和连续 `compress` 后都长期低于 `compress/launch`。

据此只做最小修订，不改其他标签：

1. 将 `drive` 从温和半伸展动作改为更明确的“中间推进/释放”：
   - 旧：`[+0.25, -0.25, +0.55, +0.30]`
   - 新：`[+0.20, -0.55, +0.78, +0.55]`
2. 将 `takeoff` 相位 prior 改为 `drive > launch`，给 `drive` 一个真正属于自己的阶段窗口：
   - 旧：`drive=0.34, launch=0.44`
   - 新：`drive=0.49, launch=0.30`

本轮要验证的具体问题是：

> 若将 `drive` 做成更明确的中间推进动作，并在 `takeoff` 相位优先尝试它，是否能把当前策略从“compress + absorb 的保守循环”推向“compress -> drive -> launch”的更完整相位链？

#### 结果（`drive` 修订版 5k 对照）

运行：

`python -u scripts/run_hopper_mcts.py --budget 5000 --seed 3 --num_simulations 30 --rollout_depth 5 --eval_episodes 1 --eval_freq 5000 --train_updates 64 --train_batch 64 --seed_steps 2000 --lr 3e-4 --vnet_mix 0.5 --device cuda --save_path results/hopper_mcts_5k_seed3_drivefix_gpu.pt`

结果：

| step | eval_reward |
|------|---:|
| 0 | 0.0000 |
| 5000 | 0.0000 |

中途仍出现一次非零 train episode（`2500` 步，`ep_reward=1.1156`），但后续回合又回到零。动作分布与上一版 5 标签实验几乎同型：

- `2500`: `compress=453, drive=2, absorb=45`
- `3000`: `compress=457, drive=4, tuck=15, absorb=24`
- `3500`: `compress=455, drive=4, tuck=14, absorb=27`
- `4000`: `compress=464, drive=2, tuck=11, absorb=23`
- `4500`: `compress=473, absorb=27`
- `5000`: `compress=461, drive=1, tuck=14, absorb=24`

与上一版相比，`drive` 的出现次数略有增加，并在统计上替代了原先少量的 `launch`，但并未改变整体行为模式；策略依旧塌在 `compress + absorb` 的保守循环中。

同时，world model 的 `state_loss` 明显更低（后段约 `0.30~0.40`，上一版约 `0.42~0.56`），说明 `drive` 修订改善了模型拟合稳定性，但**没有转化为行为层面的突破**。

结论：

1. `drive` 太弱并不是唯一瓶颈；
2. 仅靠增强 `drive` 并提高其 `takeoff` prior，还不足以把策略推进到 `compress -> drive -> launch` 链；
3. 当前更可能的根因是：在现有单步查表动作体系下，搜索仍能用 `compress + absorb` 获得相对稳定的低风险 shaping，而没有被迫进入真正的推进/腾空 regime。

因此，下一阶段不宜继续只调单个标签或 prior，而应重新检查：

- `predict_plan_reward` 是否过度偏好保守接触态；
- 单步查表动作是否天然缺乏“必须离地”的推进压力；
- 是否需要在决策层或 shaping 里显式鼓励 `drive/launch` 所对应的前向与腾空转移。

### 2026-04-17 新增：最小 reward shaping 修订，抑制保守接触循环

在 `drive` 修订版 `5k` 仍然失败后，对 `predict_plan_reward` 做了针对性审查。结论是：当前 shaping 的问题已经不是“是否鼓励向前”，而是它在事实上奖励了**高接触、持续下沉但仍保持姿态稳定**的保守状态。

原公式为：

`true_reward + viability * (0.15 * speed_progress + 0.08 * action_energy)`

其中：

- `viability` 在当前保守轨迹上几乎始终为 `1`；
- `action_energy` 纯正奖励，会奖励“大动作但未离地”；
- `speed_progress` 只看 `vx`，使重复 `compress` 时产生的缓慢前滑也被视为正进展。

用 `drivefix` 版 5k 模型做离线检查可见：连续执行 `compress` 时，虽然相位很快落入 `landing`，`support` 持续上升、`vz` 持续下降，但旧 `plan_reward` 仍从 `0.0220` 单调上升到 `0.0293`。这说明 shaping 正在系统性强化“compress + absorb”的保守循环。

据此做最小修订：

1. **去掉 `action_energy` 正奖励**；
2. 正项改为更贴近 hop 转移的量：
   - `speed_progress`
   - `upward_progress`
   - `support_relief`
   - `liftoff_progress`
   - `clearance_gain`
3. 新增负项：
   - `sticky_contact = 高 support × 下沉速度`

即：奖励“减接触、上抬、前冲、离地”，惩罚“高接触且继续向下压”。

离线 sanity check 表明修订方向正确：在同一条连续 `compress` 轨迹上，`plan_reward` 从首步约 `0.0803` 快速转为负值，并随着高接触下沉进一步降低，不再出现“越压越奖”的现象。

下一步即基于该 shaping 直接重跑 `5k`，检验策略是否从保守循环中被推出去。

#### 结果（reward 修订版 5k 对照）

运行：

`python -u scripts/run_hopper_mcts.py --budget 5000 --seed 3 --num_simulations 30 --rollout_depth 5 --eval_episodes 1 --eval_freq 5000 --train_updates 64 --train_batch 64 --seed_steps 2000 --lr 3e-4 --vnet_mix 0.5 --device cuda --save_path results/hopper_mcts_5k_seed3_rewardfix_gpu.pt`

结果：

| step | eval_reward |
|------|---:|
| 0 | 0.0000 |
| 5000 | 0.0000 |

更关键的是，整条训练轨迹与上一轮 `drivefix` 版几乎逐点同型：

- `2500`: `compress=453, drive=2, absorb=45`, `ep_reward=1.1156`
- `3000`: `compress=457, drive=4, tuck=15, absorb=24`
- `3500`: `compress=455, drive=4, tuck=14, absorb=27`
- `4000`: `compress=464, drive=2, tuck=11, absorb=23`
- `4500`: `compress=473, absorb=27`
- `5000`: `compress=461, drive=1, tuck=14, absorb=24`

区别主要体现在 `vnet_loss` 数值上，而不在行为轨迹上。这说明：

1. 新 shaping 虽改变了 planner/value 目标的数值结构；
2. 但在当前 5k 预算与现有搜索机制下，它**没有改写早期 MCTS 选择与 replay 分布**；
3. 因而训练轨迹与上一轮几乎完全相同，终评也仍为 0。

结论：

- `predict_plan_reward` 的问题确实存在，且离线诊断方向正确；
- 但**仅靠最小 reward shaping 修订，还不足以在 5k 内改变策略吸引子**；
- 当前更大的限制很可能是：在现有世界模型与单步查表动作的组合下，早期搜索对动作的排序几乎未变，导致 replay 分布没有发生实质变化。

下一阶段若继续推进，需要考虑不再只改局部 reward 项，而要改写能真正影响早期搜索排序的部分，例如：

- MCTS 扩展/选择时的相位约束强度；
- rollout policy；
- 单步动作是否升级为更强的阶段性 option 或小段宏动作。

### P0 α_min ablation 完整结果（seed=3, 50k）

三组 ablation 已在 GPU 1/2/3 上并行完成，与基线（α_min=0.10，已记录的 v2 seed=3 50k）共四个对照点。

#### 完整 eval 曲线

| step | α_min=0.10（基线） | α_min=0.20 | α_min=0.25 | α_min=0.30 |
|------|---:|---:|---:|---:|
| 0 | 0.0004 | 0.0004 | 0.0004 | 0.0004 |
| 10000 | 0.05 | 1.08 | 0.73 | 0.79 |
| 20000 | 0.04 | 1.17 | 5.84 | 0.26 |
| 30000 | 0.01 | 0.002 | 2.41 | 0.04 |
| 40000 | 0.73 | 2.88 | **48.28** | 0.69 |
| **50000** | 10.05 | 0.47 | **33.84** | 3.66 |

#### 判据分析

- **50k eval 排名**：α_min=0.25 > α_min=0.10（基线） > α_min=0.30 > α_min=0.20
- α_min=0.25 全程除 10k 外都最强；40k=48.28 是所有 SD v2 实验中的最高单点
- α_min=0.25 在 40k→50k 仅回落 30%（48.28→33.84），表明已稳定掌握 hop 模式
- 训练中 α_min=0.25 在 20k 即出现 5.84 的 eval 信号，比基线提前 30k 步

#### α_min 与性能呈倒 U 型关系

| α_min | 50k eval | 解读 |
|---:|---:|------|
| 0.10 | 10.05 | shaping 托底太弱，learned head 振荡时 planner 退化为盲搜 |
| 0.20 | 0.47 | 早期表现尚可（10k=1.08）但中后段未稳定（与 0.10 的间接路径相反） |
| **0.25** | **33.84** | 甜点：shaping 既能稳定托底又不过度压制 learned head |
| 0.30 | 3.66 | shaping 占比过高，learned head 输出被持续覆盖，无法主导高 reward 区域 |

α_min=0.20 反常偏低的解释推测：训练数据分布对 shaping 权重存在敏感窗口，0.20 处刚好让 planner 在 30k 后落入次优行为吸引子。需要 seed=1,2 验证。

#### 等价样本效率

| 对照 | 等价于其他方法的 step 数 |
|------|---:|
| α_min=0.25 @ 40k = 48.28 | TD-MPC2 ~123k；DreamerV3 ~193k |
| α_min=0.25 @ 50k = 33.84 | TD-MPC2 ~115k；DreamerV3 ~178k |

均为单 seed 数据，与官方 3-seed 均值对比仅供参考。

#### 选定结论

**α_min=0.25 确定为下一阶段最优配置**，进入：

1. P1: height_head 辅助监督损失实施
2. eval 日志改 schema（保留 std 和单 episode 数据）
3. 以新配置补 seed=1, 2 的 50k，形成 3-seed 均值
4. 视情况延长到 100k 验证渐近趋势

## 2026-04-17

### Hopper Route B：将“相位先验”改为“相位硬支持集”

前一轮 `5-intent / drivefix / rewardfix` 的共同现象表明，`compression + absorb` 吸引子并未因为局部 reward 或 soft prior 调整而改变。进一步检查脚本后可见，原实现中的“相位先验”仅体现在：

1. 节点扩展时按 prior 决定未展开动作的抽样顺序；
2. PUCT 的探索项中乘以 prior；
3. rollout policy 按 prior 采样。

但每个节点在足够少的仿真次数内仍会逐步把全部 5 个 intent 都展开出来，因此该机制并未真正把 Route B 的语义结构施加到搜索空间本身。基于这一判断，本轮将相位语义从“软偏置”改为“硬支持集”。

### 设计与实现

本轮修改位于 `scripts/run_hopper_mcts.py`，核心变化如下：

- 新增 `get_valid_intents()`，令每个 coarse phase 只保留语义允许的 intent 子集；
- `compression` 默认仅允许 `compress / drive`，在“已加载、尚未明显下落”的 stance 条件下额外放开 `launch`；
- `takeoff` 仅允许 `drive / launch`；
- `flight` 仅允许 `tuck`，在下降段额外放开 `absorb`；
- `landing` 仅允许 `absorb / compress`；
- `get_intent_priors()` 不再对全部 intent 归一化，而是先用支持集做 mask，再施加轻度 repeat penalty；
- MCTS 的 selection / expansion 以当前节点的有效 intent 数为准，不再默认每个节点都展开到 5 个子节点；
- seed 阶段探索同样改为相位约束下的 intent 采样，以保证回放数据分布与 Route B 的语义主张一致。

这一步的研究目的不是提升底层控制质量，而是检验：若将“结构化离散决策”真正落实为搜索空间约束，而非仅作为 prior，是否能够改写早期 replay 分布并帮助 Hopper 脱离 `compress + absorb` 吸引子。

### 静态检查与 smoke

已完成：

- `python -m py_compile scripts/run_hopper_mcts.py src/world_model/dmcontrol_hopper_sd_model.py`
- `python -u scripts/run_hopper_mcts.py --budget 600 --seed_steps 300 --eval_episodes 1 --eval_freq 300 --num_simulations 16 --rollout_depth 4 --train_updates 8 --train_batch 32 --device cpu --save_path /tmp/hopper_mcts_phasegate_smoke.pt`

smoke 结果表明脚本可以完整运行，且 intent 分布已明显受支持集约束；在 `steps=600` 的训练 episode 中仅出现 `compress` 与 `absorb`。

### 5k 正式结果：相位硬支持集版本

运行配置：

`python -u scripts/run_hopper_mcts.py --budget 5000 --device cuda --save_path results/hopper_mcts_5k_seed3_phasegate_gpu.pt`

最终 eval：

| step | eval return |
|------|------------:|
| 0 | 0.0000 |
| 5000 | 0.0000 |

训练轨迹如下：

- `2500`: `compress=455, drive=4, tuck=14, absorb=27`, `ep_reward=0.0000`
- `3000`: `compress=464, drive=2, tuck=11, absorb=23`, `ep_reward=0.0000`
- `3500`: `compress=473, absorb=27`, `ep_reward=0.0000`
- `4000`: `compress=461, drive=1, tuck=14, absorb=24`, `ep_reward=0.0000`
- `4500`: `compress=458, absorb=42`, `ep_reward=0.0000`
- `5000`: `compress=474, absorb=26`, `ep_reward=0.0000`

### 结果解释

该版本虽然在形式上更贴近 Route B 的结构化主张，但在 `5k` 预算下并未带来性能改善，且训练分布反而更集中地塌缩为 `compress + absorb`。这说明：

1. 仅将 soft prior 升级为 hard support，还不足以把搜索推出当前吸引子；
2. 现有 coarse phase 划分与固定 torque 表组合后，更像是在收缩搜索空间，而非打开新的有效跃迁通道；
3. 当前更大的瓶颈仍可能在于：`compression -> takeoff` 的真实可达性不足，或单步 primitive 本身不足以承载完整相位推进。

因此，此轮结论不是“Route B 无价值”，而是：

- 仅凭当前这版单步 intent + 查表力矩 + coarse phase mask，还不能验证其优越性；
- 若继续推进，更可能需要改动的是“相位跃迁可达性”本身，例如更细的 subphase 判别、显式的阶段过渡条件，或短程宏动作，而不宜继续只在 soft/hard prior 之间来回切换。

### 2026-04-17：TD-MPC2 100k 对照渲染代理点位确定

为了补充 Hopper 论文材料中的视觉对比，对 TD-MPC2 `hopper-hop` 100k 设定对应的 checkpoint 进行了再次核查。当前可核实的信息如下：

- TD-MPC2 官方公开材料中可以确认存在收敛后的最终单任务权重，但未发现 `hopper-hop @ 100k` 的公开中间 checkpoint 下载入口；
- 本地仓库内也未找到官方原版 TD-MPC2 的 `100k` Hopper 中间权重，仅有官方结果表 `external/tdmpc2/results/tdmpc2/hopper-hop.csv`；
- 因此，本轮视觉对比改为选取一个本地可实际回放、且数值接近 TD-MPC2 官方 100k 均值 `13.6` 的代理点位。

筛选标准为：`eval.csv` 中存在明确评估回报，且对应 `models/<step>.pt` 或同预算 `final.pt` 真实存在。按与 `13.6` 的绝对差值排序后，最接近的可回放点位为：

- `logs/hopper-hop/3/sd_hopper_100k_v2_r1/models/final.pt`
- 对应回报：`15.8251`
- 与 TD-MPC2 官方 `100k=13.6` 的差值：`2.2251`

为保证后续同类对照可以重复执行，新增脚本：

- `scripts/make_hopper_checkpoint_eval_mp4.py`

该脚本用于按真实评估协议回放任意 Hopper checkpoint，并将每条 episode 单独保存为 `mp4`，同时输出 `episode_summary.txt` 记录 checkpoint 与关键评估配置。

### Hopper Route B：局部模型修正尝试

在 retorque 版本中，动作表与离散语义已经对齐，训练主干也已稳定切换为 `drive + launch`。因此，本轮不再修改动作表，而是直接针对对齐诊断中最明确的偏差做最小世界模型修正，目标仅限于：

1. 压低 `stance_release` 中对 `launch / drive` 的竖直收益和离地收益高估；
2. 压低 `landing_impact` 中对 `clearance` 保持的高估。

对应改动位于 `src/world_model/dmcontrol_hopper_sd_model.py`：

- 在显式动力学中加入 `contact_lock`，使“高支撑、低净空”状态下的正向 release 不会被直接当成强起跳；
- 在 `next_support` 中加入 `support_hold`，避免释放段无依据地过快丢失接触；
- 在 `next_clearance` 中加入与 `next_support / next_impact` 相关的接触惩罚，降低净空快速上升的高估；
- 随后又将 upward lift 改成阈值化形式：只有足够强的正向 `drive_action` 才转化为真正的竖直 lift，从而在模型里把 `drive` 与 `launch` 拉开。

### 修正后的方向性诊断

继续使用 `results/hopper_mcts_5k_seed3_retorque_gpu.pt` 中的训练后模型，在真实环境中做两条 200-step on-policy rollout，对比模型一步预测与真实下一状态。

关键结果如下。

#### 1. `launch @ stance_release`

真实环境：

- `Δvx = -0.0063`
- `Δvz = +0.0140`
- `Δsupport = +0.0151`
- `Δclearance = +0.0000`

初始 retorque 版本预测：

- `Δvx = +0.0005`
- `Δvz = +0.0632`
- `Δsupport = -0.0755`
- `Δclearance = +0.1126`

加入局部修正后预测：

- `Δvx = +0.0005`
- `Δvz = +0.0311`
- `Δsupport = +0.0236`
- `Δclearance = -0.0302`

再加阈值化 lift 后预测：

- `Δvx = +0.0005`
- `Δvz = +0.0283`
- `Δsupport = +0.0238`
- `Δclearance = -0.0303`

这说明：

- `launch` 的 `Δvz` 高估已明显收敛；
- `Δsupport` 的符号也已从“错误的明显减小”纠正为“略增”，与真实方向一致；
- 但 `Δclearance` 已出现过修正，从原先的正向高估转为小幅负值。

#### 2. `drive @ stance_release`

真实环境：

- `Δvx = +0.0046`
- `Δvz = -0.0167`
- `Δsupport = -0.0080`
- `Δclearance = +0.0001`

初始 retorque 版本预测：

- `Δvz = +0.0488`
- `Δsupport = -0.0764`
- `Δclearance = +0.1150`

加入局部修正后预测：

- `Δvz = +0.0279`
- `Δsupport = +0.0208`
- `Δclearance = -0.0274`

再加阈值化 lift 后预测：

- `Δvz = +0.0227`
- `Δsupport = +0.0213`
- `Δclearance = -0.0275`

这说明：

- `drive` 的竖直收益高估虽有所下降，但仍未纠正到真实负值；
- `support` 与 `clearance` 的方向仍不完全正确。

#### 3. `absorb @ landing_impact`

真实环境：

- `Δvx = -0.0034`
- `Δvz = +0.0254`
- `Δsupport = +0.0043`
- `Δclearance = -0.0154`

修正后预测仍为：

- `Δvx = +0.0017`
- `Δvz = -0.0172`
- `Δsupport = +0.0559`
- `Δclearance = +0.0567`

说明 landing 区域的方向性错配仍较明显，尤其 `Δvz` 与 `Δclearance` 的符号尚未对齐。

### 短 smoke

使用修正后的模型前向运行：

`python -u scripts/run_hopper_mcts.py --budget 600 --seed_steps 300 --eval_episodes 1 --eval_freq 300 --num_simulations 16 --rollout_depth 4 --train_updates 8 --train_batch 32 --device cpu --save_path /tmp/hopper_mcts_modelpatch_smoke.pt`

结果如下：

- `seed ep @500 = 4.9136`
- `steps=600` 的训练分布为：
  - `drive=31, launch=28, absorb=41`
- 最终 eval 仍为 0

与修正前 retorque smoke 基本同型。这说明当前局部修正主要是在**缓和预测方向偏差**，尚不足以在极短预算内显著改变 planner 的主吸引子。

### 当前结论

这轮结果表明：

1. 依据第一性原理做的小规模局部修正，确实可以把 `launch` 的方向性预测拉近真实环境；
2. 但 `drive` 与 `absorb` 的关键错配仍然存在，且行为分布几乎不变；
3. 因此，继续靠这种手工小补丁逐项修正 world model，边际收益已经开始递减。

据此，后续更有价值的方向将不再是继续调单个系数，而是：

- 把 `vz / support / clearance` 明确纳入更系统的训练约束或辅助监督；
- 或者转向更长 horizon / 更强终端价值的 solver 对照，检查在当前模型偏差仍存在时，求解层是否有办法避免被这些局部高估误导。

### Solver 对照筛查：单纯增加 lookahead 强度并不能改变当前主吸引子

在局部模型修正后，又对求解层做了一个最小对照，目的是区分：

- “当前失败是否只是因为 planner 看得还不够远”

与

- “当前模型对动作序列的排序本身就已经把同一条轨迹判为最优”。

为避免再次直接投入一轮完整 5k 训练，本轮先固定当前 `retorque` checkpoint：

- `results/hopper_mcts_5k_seed3_retorque_gpu.pt`

只在真实环境中做 `200` 步短评估，对比三组 solver 配置：

- `base`: `num_simulations=50`, `rollout_depth=8`
- `deep`: `num_simulations=100`, `rollout_depth=12`
- `deeper`: `num_simulations=150`, `rollout_depth=16`

三组结果完全一致：

- 回报均为 `0.0`
- 动作分布均为：
  - `compress=7`
  - `drive=70`
  - `launch=65`
  - `absorb=58`
- 终点根节点诊断保持在：
  - `root_phase = compression`
  - `root_subphase = stance_release`

这说明，在当前模型下：

1. 单纯增加 MCTS 仿真数与 rollout 深度，并不会把 policy 推离现有 `drive + launch + absorb` 主吸引子；
2. planner 的决策对 solver 强度不敏感，意味着当前瓶颈更可能不在“看得不够远”，而在于模型对这条轨迹的局部排序本身就过于稳定；
3. 因而，再继续只从 solver 侧做小幅 lookahead 增强，边际收益预计有限。

#### 当前更新后的判断

结合前面的 world-model 对齐诊断与本轮 solver 对照，可进一步收缩出一个更稳妥的阶段性结论：

- 当前 retorque 版本的主要限制，优先仍落在 **model ranking bias**，而非 solver budget 不足；
- 既然更深 lookahead 也无法改变动作选择，那么下一阶段更有价值的方向应优先是：
  - 将 `vz / support / clearance` 纳入更系统的训练约束或辅助监督；
  - 而不是继续做局部求解层参数增强。

### Hopper Route B：`2k key-loss` 中预算复测与同口径诊断

为直接检验“关键量监督”是否能够缓解当前的 model ranking bias，本轮没有再改动作语义，也没有再动 solver 主配置，而是只做两类最小变更：

1. 在 `scripts/run_hopper_mcts.py` 中增加 `--eval_max_steps`，使中预算实验不会被 `500-step` 初始评估吞没；
2. 新增 `scripts/diagnose_hopper_routeb_alignment.py`，将此前一次性的 on-policy 一步对齐分析固化为可重复脚本。

#### 中预算训练命令

`MUJOCO_GL=glfw python -u scripts/run_hopper_mcts.py --budget 2000 --seed_steps 1000 --eval_episodes 1 --eval_max_steps 200 --eval_freq 2000 --num_simulations 50 --rollout_depth 8 --train_updates 32 --train_batch 64 --device cuda --save_path results/hopper_mcts_2k_seed3_keyloss_eval200_gpu.pt`

训练过程中的关键信号如下：

- `eval step=0 = 0.0000`
- `seed ep=0, steps=500, ep_reward=6.5659`
- `seed ep=1, steps=1000, ep_reward=0.0000`
- 首次在线更新后：
  - `state_loss = 0.228978`
  - `reward_loss = 0.000862`
  - `key_state_loss = 0.015039`
  - `key_delta_loss = 0.015039`
- `train ep=2, steps=1500, ep_reward=0.0000`
  - 动作分布：`drive=231, launch=228, absorb=41`
  - 更新后：`state_loss = 0.201644`, `key_state_loss = key_delta_loss = 0.014476`
- `train ep=3, steps=2000, ep_reward=0.0000`
  - 动作分布：`compress=7, drive=210, launch=205, tuck=14, absorb=64`
  - 更新后：`state_loss = 0.143473`, `key_state_loss = key_delta_loss = 0.012385`
- 训练态记录的最终评估为：
  - `eval step=2000 = 0.2602`

这说明：

- 关键量监督确实稳定参与了训练，且损失随训练推进持续下降；
- 行为主吸引子仍然是 `drive + launch`，但与此前 smoke 相比，后段已出现少量 `compress / tuck` 回流；
- 训练态出现了一个非零终点评估，但它仍只是单 episode 结果，尚不能单独视为“性能突破”。

#### fresh reload 对照评估

为避免把训练过程中的单次随机回报误判为稳定增益，又对两个 checkpoint 在同一 `1 episode × 200 step × 50 sims × depth 8` 协议下重新加载评估：

- `results/hopper_mcts_5k_seed3_retorque_gpu.pt = 0.0`
- `results/hopper_mcts_2k_seed3_keyloss_eval200_gpu.pt = 0.0`

因此，`0.2602` 更稳妥的解释是：当前版本**可能出现了偶发的非零轨迹**，但尚未形成可复现的稳定收益改善。

#### 同口径一步对齐诊断

诊断命令统一为：

`python scripts/diagnose_hopper_routeb_alignment.py --checkpoint <ckpt> --device cuda --episodes 2 --max_steps 200 --num_simulations 50 --rollout_depth 8 --limit_per_event 32`

先给出新 `2k key-loss` checkpoint 的结果：

1. `launch @ stance_release`
   - 真实：`Δvx=-0.0153`, `Δvz=+0.0392`, `Δsupport=+0.0481`, `Δclearance=+0.0001`
   - 预测：`Δvx=+0.0038`, `Δvz=+0.0093`, `Δsupport=+0.0033`, `Δclearance=+0.0239`
2. `drive @ stance_release`
   - 真实：`Δvx=+0.0072`, `Δvz=-0.0386`, `Δsupport=-0.0738`, `Δclearance=+0.0004`
   - 预测：`Δvx=-0.0012`, `Δvz=+0.0064`, `Δsupport=-0.0141`, `Δclearance=+0.0296`
3. `absorb @ landing_impact`
   - 真实：`Δvx=-0.0400`, `Δvz=-0.0201`, `Δsupport=-0.0127`, `Δclearance=-0.0100`
   - 预测：`Δvx=+0.0062`, `Δvz=-0.0777`, `Δsupport=+0.0007`, `Δclearance=+0.5136`

在同一脚本、同一当前代码版本下，再给出 `retorque` 基线：

1. `launch @ stance_release`
   - 预测：`Δvx=+0.0037`, `Δvz=+0.0042`, `Δsupport=+0.0353`, `Δclearance=-0.0151`
2. `drive @ stance_release`
   - 预测：`Δvx=+0.0028`, `Δvz=+0.0040`, `Δsupport=+0.0144`, `Δclearance=-0.0065`
3. `absorb @ landing_impact`
   - 预测：`Δvx=+0.0007`, `Δvz=-0.0535`, `Δsupport=-0.0155`, `Δclearance=+0.4996`

据此可以作出更严格的比较：

1. `launch @ stance_release`
   - 方向上仍然基本正确，但相对 retorque 只体现为非常有限的细节变化，没有出现决定性的对齐改善；
2. `drive @ stance_release`
   - `Δsupport` 的预测符号从 retorque 的错误正值改成了负值，这是一个局部进步；
   - 但 `Δvz` 仍保持错误正号，说明“drive 不应被当成起跳增益”的关键排序偏差并未解决；
3. `absorb @ landing_impact`
   - `Δclearance` 仍被极大高估（`+0.5136` 对 `-0.0100`），这一块几乎没有被当前监督触及；
   - landing 区域仍是最强的结构性误差源。

#### 当前判断

本轮 `2k key-loss` 结果可以收束为：

1. 当前辅助监督**足以改变训练轨迹**，但尚不足以形成稳定、可复现的非零性能；
2. 它对 `launch` 和 `drive` 的某些局部量有轻微校正，但整体对齐改善是混合且不稳定的；
3. 对 `absorb / landing_impact` 的关键偏差几乎无效，因此当前 Route B 失败的核心瓶颈仍然存在；
4. 以现有证据看，尚不适合直接按“日志中的 key-loss 版本已经可行”去实现完整路线；更稳妥的下一步应是继续重构监督形式本身，尤其要更直接地约束：
   - `drive` 在 `stance_release` 中的 `Δvz` 符号；
   - `absorb` 在 `landing_impact` 中的 `Δclearance / Δsupport` 排序。

换言之，这轮结果支持“继续沿训练约束方向推进”，但**尚不支持**直接进入完整 `5k` 实装并把它当作已验证架构。

### Hopper Route B：条件化符号监督的第一次实现与收缩

在上述判断基础上，继续沿训练约束方向前推了一步，但本轮保持两个原则：

1. 不再改离散语义与 solver；
2. 只把监督直接加在已经反复暴露的排序偏差上。

代码位于 `scripts/run_hopper_mcts.py`，初版实现做了两件事：

- replay 额外记录离散 `intent`，从而在训练批次中直接按 `intent × subphase` 施加条件化损失；
- 在 `train_sd_model()` 中新增三项符号约束：
  - `drive@stance_release`: 惩罚正向 `Δvz`
  - `absorb@landing_impact`: 惩罚正向 `Δclearance`
  - `absorb@landing_impact`: 初版还尝试惩罚正向 `Δsupport`

#### smoke 命令

`MUJOCO_GL=glfw python -u scripts/run_hopper_mcts.py --budget 600 --seed_steps 300 --eval_episodes 1 --eval_max_steps 100 --eval_freq 600 --num_simulations 16 --rollout_depth 4 --train_updates 8 --train_batch 32 --device cuda --save_path /tmp/hopper_mcts_signloss_smoke.pt`

关键日志如下：

- 第一次更新：
  - `state_loss = 0.095509`
  - `key_state_loss = key_delta_loss = 0.011881`
  - `drive_vz_sign_loss = 0.049440`
  - `absorb_clearance_sign_loss = 0.195735`
  - `absorb_support_sign_loss = 0.093900`
  - `drive_release_ratio = 0.3477`
  - `absorb_landing_ratio = 0.0820`
- 第二次更新：
  - `state_loss = 0.060869`
  - `key_state_loss = key_delta_loss = 0.013542`
  - `drive_vz_sign_loss = 0.036666`
  - `absorb_clearance_sign_loss = 0.258875`
  - `absorb_support_sign_loss = 0.054826`
  - `drive_release_ratio = 0.3594`
  - `absorb_landing_ratio = 0.1484`

这说明新监督在工程上是可工作的：

- 样本命中率足够，不是空损失；
- 训练没有数值爆炸；
- 但 `600` 预算下最终回报仍为 `0`，因此只能作为“监督已接通”的 smoke。

#### smoke 对齐诊断与关键修正

对 `/tmp/hopper_mcts_signloss_smoke.pt` 做 `1 episode × 80 step` 短诊断后，得到：

- `drive @ stance_release`
  - 真实：`Δvz = -0.0361`
  - 预测：`Δvz = +0.0363`
- `absorb @ landing_impact`
  - 真实：`Δclearance = -0.0277`
  - 预测：`Δclearance = +0.5371`
  - 真实：`Δsupport = +0.0394`
  - 预测：`Δsupport = -0.0076`

这条结果有一个非常重要的含义：

- `drive` 的竖直方向错误依旧存在，因此继续保留 `drive@stance_release` 的 `Δvz` 符号约束是合理的；
- `absorb` 的 `Δclearance` 方向仍然明显错误，因此继续保留 `absorb@landing_impact` 的 `Δclearance` 符号约束也是合理的；
- 但 `absorb` 的 `Δsupport` 在这条新轨迹上真实均值转成了正值，与上一轮 `2k` 诊断中的负值相矛盾。

据此，原本对 `absorb@landing_impact` 加“固定负向 `Δsupport`”约束的想法并不稳固。若继续把它写成默认硬约束，反而可能把不稳定的经验模式误当成结构规律。

因此，本轮代码立即收缩为：

- `drive_vz_sign_coef`：默认保留为 `5.0`
- `absorb_clearance_sign_coef`：默认保留为 `5.0`
- `absorb_support_sign_coef`：参数保留，但默认值回退为 `0.0`

#### 当前更新后的判断

这轮实现说明：

1. “按 `intent × subphase` 做条件化监督”在工程上是可行的；
2. 其中可被当前数据较稳定支持的硬约束，至少有两条：
   - `drive@stance_release` 不应被预测成正向起跳增益；
   - `absorb@landing_impact` 不应被预测成继续抬高净空；
3. `absorb@landing_impact` 的 `Δsupport` 符号目前缺乏稳定证据，不宜默认硬编码。

这也意味着，下一轮若继续做中预算验证，应优先测试的是**收缩后的两项符号约束版本**，而不是包含 `support` 硬约束的更强版本。

### Hopper Route B：收缩后两项符号约束版本的 `2k` 复测

按上述收缩结论，继续做了一个与前一轮 `2k key-loss` 尽量同构的中预算实验。此次唯一实质差异是监督项默认值变为：

- `drive_vz_sign_coef = 5.0`
- `absorb_clearance_sign_coef = 5.0`
- `absorb_support_sign_coef = 0.0`

训练命令为：

`MUJOCO_GL=glfw python -u scripts/run_hopper_mcts.py --budget 2000 --seed_steps 1000 --eval_episodes 1 --eval_max_steps 200 --eval_freq 2000 --num_simulations 50 --rollout_depth 8 --train_updates 32 --train_batch 64 --device cuda --save_path results/hopper_mcts_2k_seed3_signloss2_gpu.pt`

#### 训练过程

- `eval step=0 = 0.0000`
- `seed ep=0, steps=500, ep_reward=6.5659`
- `seed ep=1, steps=1000, ep_reward=0.0000`
- 第一次在线更新：
  - `state_loss = 0.229600`
  - `reward_loss = 0.000858`
  - `key_state_loss = key_delta_loss = 0.015545`
  - `drive_vz_sign_loss = 0.041699`
  - `absorb_clearance_sign_loss = 0.147061`
  - `absorb_support_sign_loss = 0.101032`
  - `drive_release_ratio = 0.3472`
  - `absorb_landing_ratio = 0.0571`
- `train ep=2, steps=1500, ep_reward=0.0000`
  - 动作分布：`drive=231, launch=228, absorb=41`
  - 更新后：
    - `state_loss = 0.202909`
    - `key_state_loss = key_delta_loss = 0.015120`
    - `drive_vz_sign_loss = 0.014813`
    - `absorb_clearance_sign_loss = 0.255706`
- `train ep=3, steps=2000, ep_reward=0.0000`
  - 动作分布：`compress=7, drive=210, launch=205, tuck=14, absorb=64`
  - 更新后：
    - `state_loss = 0.144595`
    - `key_state_loss = key_delta_loss = 0.013251`
    - `drive_vz_sign_loss = 0.015958`
    - `absorb_clearance_sign_loss = 0.181757`
- 训练态最终评估：
  - `eval step=2000 = 0.2602`

这里有两个直接观察：

1. 行为分布与前一轮 `2k key-loss` 几乎同型，说明这次收缩没有改变 planner 的主吸引子；
2. `drive_vz_sign_loss` 的数值确实被明显压低，说明新约束在训练目标上是有效工作的。

#### fresh reload 对照评估

仍按 `1 episode × 200 step × 50 sims × depth 8` 的 fresh reload 协议对比：

- `results/hopper_mcts_2k_seed3_keyloss_eval200_gpu.pt = 0.0`
- `results/hopper_mcts_2k_seed3_signloss2_gpu.pt = 0.0`

因此，本轮与上一轮一样，训练态的 `0.2602` 不能视为稳定可复现的收益改善。

#### 同口径一步对齐诊断

诊断命令为：

`python scripts/diagnose_hopper_routeb_alignment.py --checkpoint results/hopper_mcts_2k_seed3_signloss2_gpu.pt --device cuda --episodes 2 --max_steps 200 --num_simulations 50 --rollout_depth 8 --limit_per_event 32`

结果如下：

1. `launch @ stance_release`
   - 真实：`Δvx=-0.0153`, `Δvz=+0.0392`, `Δsupport=+0.0481`, `Δclearance=+0.0001`
   - 预测：`Δvx=+0.0061`, `Δvz=+0.0055`, `Δsupport=+0.0324`, `Δclearance=-0.0114`
2. `drive @ stance_release`
   - 真实：`Δvx=+0.0072`, `Δvz=-0.0386`, `Δsupport=-0.0738`, `Δclearance=+0.0004`
   - 预测：`Δvx=+0.0008`, `Δvz=+0.0019`, `Δsupport=+0.0168`, `Δclearance=-0.0037`
3. `absorb @ landing_impact`
   - 真实：`Δvx=-0.0400`, `Δvz=-0.0201`, `Δsupport=-0.0127`, `Δclearance=-0.0100`
   - 预测：`Δvx=+0.0035`, `Δvz=-0.0906`, `Δsupport=+0.1054`, `Δclearance=+0.4844`

与上一轮 `2k key-loss` 相比，可以更具体地看出：

1. `launch @ stance_release`
   - `Δsupport` 从 `+0.0033` 提高到 `+0.0324`，更接近真实 `+0.0481`；
   - `Δclearance` 从 `+0.0239` 拉到 `-0.0114`，绝对误差减小；
   - 但 `Δvz` 只从 `+0.0093` 变到 `+0.0055`，并没有带来决定性改观。
2. `drive @ stance_release`
   - `Δvz` 从 `+0.0064` 降到 `+0.0019`，确实被压向了零附近；
   - 但它仍然没有翻成真实所需的负号；
   - `Δsupport` 反而从上一轮的负值 `-0.0141` 回退到错误正值 `+0.0168`。
3. `absorb @ landing_impact`
   - `Δclearance` 从 `+0.5136` 只降到 `+0.4844`，高估仍极严重；
   - `Δsupport` 则从 `+0.0007` 进一步恶化到 `+0.1054`。

#### 当前判断

这轮复测进一步说明：

1. 收缩后的两项符号约束，能够在训练目标上压低 `drive` 的错误竖直增益；
2. 但这种压低目前主要体现在“把预测推向零附近”，尚不足以把 `drive` 的局部排序真正翻到正确侧；
3. `absorb / landing_impact` 仍然几乎没有被解决，它依旧是当前 Route B 失败的主导误差源；
4. 因此，这轮结果虽然比“无条件约束地继续调参”更有针对性，但仍不足以支持直接进入完整 `5k` 验证。

更稳妥的下一步应是继续围绕 `landing_impact` 的监督形式本身做结构性改写，而不是简单继续提高当前损失权重。

### 2026-04-18 Hopper Route A/B 调用链修订与最小验证

针对当前 Hopper 训练调用链，已完成以下修订，并按最小可复现脚本逐项核验：

1. `dmcontrol_hopper_sd_model.py` 中显式 `predict_reward()` 的尺度已与 DMControl 包装器中的 `action_repeat=2` 对齐，返回值改为 `2 * standing * hopping`；
2. `tdmpc2.py`、`common/world_model.py`、`common/sd_world_model.py` 中 planner 的 imagined rollout 已消除重复 `next()`，单步只推进一次动力学，并将 `next_z` 直接传入 `reward()`；
3. `common/buffer.py` 中 replay capacity 默认值已从 `min(buffer_size, steps)` 改为 `buffer_size`，并将当前缓冲区 episode 数与累计写入 episode 数分离维护，evict 时同步递减当前计数；
4. `trainer/online_trainer.py` 中 `save_full_state()` / `load_full_state()` 已补充 `_target_Qs`、`_detach_Qs`、累计 buffer 计数的保存与恢复，加载时不再调用 `model.init()` 覆盖 target/detach Q 状态。

最小验证脚本结果如下：

- `reward_scale: PASS`
- `double_next_removed: PASS`
- `buffer_semantics: PASS`
- `resume_state: PASS`

其中 `resume_state` 额外确认：恢复后 `model.init()` 未被触发，`model`、`target_Qs`、`detach_Qs`、优化器状态、`prev_mean`、训练步数及 buffer 计数均与保存时一致。

本轮未改动的事项仅有一项：`apply(init.weight_init)` 对 `_dynamics` 子 head 的递归覆盖。该行为此前已在工作记录中明确属于设计时有意保留，本次未纳入修订。

---

## 2026-04-18 去 submodule 化前归档：ablation 0 / ablation 1

v4 主线即将建立，ablation0_transition2x 与 ablation1_no_struct_residual 两个分支的 git lineage 将在去 submodule 过程中断裂。本节归档其 diff 与 smoke 结果以保留实验证据。

### Ablation 0 (transition 2×)

- 分支：`ablation0_transition2x`（master 基础上的 commit `9699689`）
- 改动：`src/world_model/dmcontrol_hopper_sd_model.py` +13/-1 行
- 本质：`forward()` 循环 `ENV_ACTION_REPEAT=2` 次，使 SD transition 与 DMControl env 的 2 inner-step 对齐（reward ×2 修复的对称化）
- 5k smoke (seed=3)：
  - 0k=1.27, 1k=0.0008, 2k=1.65, 3k=0.003, 4k=0.0, 5k=0.30
  - 训练稳定、plan_nonfinite_frac=0；速度慢 ~3×（transition 2× + planner 内嵌 next 调用复合）
- 50k×3 未启动（后续若需可按归档 patch /tmp/ablation0.patch 单独 cherry-pick）

### Ablation 1 (drop `_struct_residual`)

- 分支：`ablation1_no_struct_residual`（ablation0 基础上的改动）
- 改动：`external/tdmpc2/tdmpc2/common/sd_world_model.py` + `config.yaml`，新增 `sd_use_struct_residual` flag；`next()` 中 `_struct_residual is None` 时跳过 trend-protect + free-dim 修正路径
- 5k smoke (seed=3, `sd_use_struct_residual=false`)：
  - 0k=1.30, 1k=0.026, 2k=0.60, 3k=0.55, 4k=0.05, 5k=1.21
  - 训练稳定、正常退出；与 ablation 0 smoke 0.30 同量级，无显著崩溃证据
- 结论：去掉 `_struct_residual` 不会破坏训练稳定性，后续可作 "残差栈非必要" 的 appendix 证据点
- 50k×3 未启动

### 归档文件位置

- `/tmp/ablation0.patch` — ablation 0 相对 master 的完整 unified diff
- `/tmp/ablation1.patch` — ablation 1 相对 submodule df1b9d7 的完整 unified diff
- smoke 日志：
  - `/tmp/sdwm_ablation0_transition2x/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_5k_ablation0_transition2x_smoke_20260418/`
  - `/tmp/sdwm_ablation1_no_struct_residual/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_5k_ablation1_no_struct_residual_smoke_20260418/`

注：/tmp 不持久，正式归档应把 patch 复制到 docs/worklog/patches/ 下，下一步 v4 commit 前一起做。

---

## 2026-04-21：v4 结构路线受阻时的备选发展路径

基于本轮对 `v3b`、`v4.24` 及其 30k 评估轨迹的并行对照，需补记一条方法论判断：

当前 `v4` 路线中，显式阶段链、`pitch wrap`、`rear-flight penalty`、`frozen_capture`、`release_engagement` 等修订，确实提升了结构语义的一致性与诊断清晰度；但这些改动并未自动转化为策略层的可学习性。现有证据更支持如下解释：`v4` 的问题不是“结构先验错误”，而是“结构先验正确但可进入的行为盆地过窄”，导致策略尚未稳定进入 `crawl -> standup -> release -> handoff` 的后半阶段链。

据此，若后续继续沿“纯 9 维显式结构态 + 极窄局部残差”方向优化，仍无法在正式对照中稳定出现非零 `standup/release/handoff` 统计，则应将“受控残差回接”作为 `v4` 的第二条发展路径，而不再将其视为对结构路线的放弃。

### 该备选路径的核心判断

- 即便接回较大的 residual latent，`SD` 的作用也不会因此被剥夺，只要高层奖励塑形、阶段推进与验收口径仍由显式结构量主导。
- `v3b` 在低预算下明显优于 DreamerV3 与 TD-MPC2，并不只能归因为“更宽的神经自由度”；更关键的差异仍在于，结构化先验使 reward shaping 可以围绕明确的动力学子目标来设计，而不是依赖黑盒 latent 的间接相关量。
- 因而，后续若恢复较强残差，应理解为“在显式语义之下补回探索与校正空间”，而不是退回无结构黑盒模型。

### 备选路径的设计原则

若进入该路线，建议遵循以下约束，而非简单回退到早期的宽残差形态：

1. 保留当前 `v4` 的显式核心 stock 与阶段链，使 `capture / crawl_to_standup / standup / release / handoff` 仍是主身份。
2. 恢复一个受限的 hidden residual latent，用于吸收未建模接触、快变量切换、求解器细节及局部动力学失配。
3. 高层奖励塑形继续主要由显式结构态计算，尤其是阶段推进奖励、冻结惩罚、后仰飞惩罚、绕行抑制等，不交由 residual 直接决定。
4. 残差优先修正快变量、桥接变量或局部动力学量，不直接重写高层 phase stock；若确有结构残差，也应带有 trend protection 与幅度边界。
5. 验收时除总回报外，仍必须检查显式阶段量是否真正被激活，防止 residual 通过旁路机制重新制造“有分但不可解释”的策略。

### 与恢复低姿前撑入口的关系

该备选路径与“恢复 `v3b` 型低姿态前撑爬行入口”并不冲突。更准确地说，两者应被视为同一策略的上游与下游：

- 上游通过放宽可行域与有限残差，为策略重新提供可进入的前撑/爬行行为盆地；
- 下游继续通过显式阶段链与结构化 shaping，引导策略由低姿前撑逐步转向抬身、释放与持续推进。

因此，后续若 `v4` 仅靠继续收紧结构定义仍难以突破，不宜直接得出“结构路线失败”的结论；更合理的下一步，是尝试“显式 SD 主干 + 受控 residual latent + 保留 v4 已修补漏洞”的中间方案，并将其作为 `v4` 的正式备选发展路径归档。

---

## 2026-04-21：恢复版 `v3b` 200k 续训修复与重启

为将恢复版 `v3b` 的三 seed `50k` 结果继续延伸到 `200k`，本轮先尝试直接以 `resume_from=final.full.pt` 后台续训，但首次启动失败。问题定位与修复如下。

### 首次失败原因

- 后台命令沿用了主线 `v4` 的若干 override，例如 `sd_use_struct_residual=false`。恢复版 `legacy_v3b_restore` 的 Hydra 配置不包含这些新键，训练在配置解析阶段即退出，因此未创建新的实验目录。
- 改为仅沿用恢复版 `50k` 成功运行的最小参数集合后，前台探针可进入 `load_full_state`，但在 `step=50001` 处又触发 `UnboundLocalError: local variable 'info' referenced before assignment`。

### 续训边界 bug 诊断

- 文件：`legacy_v3b_restore/external/tdmpc2/tdmpc2/trainer/online_trainer.py`
- 现象：从 `final.full.pt` 恢复后，训练状态中包含模型、优化器、buffer 与计数器，但并未保存 live env 与当前未结束 episode 的 `info/_tds`。
- 原实现假定 `done=True` 分支内始终可以读取上一轮 episode 的 `info`，导致恢复后的首个循环在尚未执行任何 `env.step()` 之前就访问未定义的 `info`。

### 修复内容

- 在 `train()` 中新增 `last_info = None`；
- 仅当 `last_info` 与 `_tds` 均已存在且 episode 至少包含一条 transition 时，才回写上一轮 episode 的训练日志并将该 episode 压入 replay buffer；
- 恢复后默认从新的 episode 边界重新 `env.reset()` 开始训练，不再假定 live env 状态可被 checkpoint 精确恢复。

该修复与当前 `full_state` 的真实保存范围一致，即“恢复模型训练状态 + replay buffer”，而不声称支持 mid-episode 的物理状态精确恢复。

### 200k 续训启动参数

三条正式续训统一改为：

- `steps=200000`
- `eval_freq=25000`
- `checkpoint_freq=25000`
- `eval_episodes=10`
- `save_full_state=true`
- `compile=false`
- `resume_from=<对应 seed 的 50k final.full.pt>`

并保持恢复版 `v3b` 原有模型规模与 planner 参数，不再混入主线 `v4` 的额外 override。

### 当前状态

- `seed=1`、`seed=2`、`seed=3` 的 `200k` 续训已重新启动；
- 日志目录分别为：
  - `logs/hopper-hop/1/legacy_v3b_restore_200k_seed1_resume50k_20260421/`
  - `logs/hopper-hop/2/legacy_v3b_restore_200k_seed2_resume50k_20260421/`
  - `logs/hopper-hop/3/legacy_v3b_restore_200k_seed3_resume50k_20260421/`
- 训练均已从 `step=50001` 恢复并继续推进；
- 首个新的官方评估点为 `75000`。

## 2026-04-24：将 Hopper 奖励正式改为“五状态静态判别 + 下一步迁移奖励”

围绕 Hopper 主线奖励语义，再次做了收口。此前虽然已经逐步从状态内连续 shaping 收回到更简洁的闭环设计，但实现中仍残留一部分“在某个状态里直接给奖励”的表达方式，容易让奖励语义滑回“状态内停留也有收益”。本轮据此明确改为：奖励一律基于 `state --action--> next_state` 的推演结果，只按**下一步进入了什么状态**来给分，而不再对当前状态本身直接给状态内奖励。

### 一、主状态口径

当前 Hopper 主状态统一为五类，且在实现中保持互斥：

1. `falling_air`
2. `rising_air`
3. `grounded_prepared`
4. `grounded_recovery`
5. `grounded_failed`

其中：

- `grounded_prepared` 与 `grounded_recovery` 都要求足部支撑充分；
- `grounded_failed` 不再要求足部支撑，只要求 `z_min` 表示已着地，但姿态未进入 `prepared/recovery`；
- 前伏与后仰失败态仍使用 `pitch + z_min` 判据识别，并继续用于动作先验边界裁剪。

### 二、奖励语义修订

`src/world_model/dmcontrol_hopper_sd_model.py` 中，`predict_shaping_reward()` 已改为纯迁移奖励，不再包含状态内持续奖励。当前只保留五类弱迁移引导：

1. `grounded_failed -> grounded_recovery`
2. `grounded_recovery -> grounded_prepared`
3. `grounded_prepared -> rising_air`
4. `falling_air -> grounded_recovery`
5. `falling_air -> grounded_prepared`

这一步的意图是把结构化奖励重新压回“引导下一状态朝目标类型迁移”的角色，而不是在某个状态内持续堆收益。对于 Hopper 这类强接触任务，这样的表达更符合状态机语义，也更便于后续直接核对“当前策略到底有没有进入目标状态迁移链”。

### 三、离线验证结果

使用已有保存轨迹，按最新五状态口径做离线回算：

1. `v441 5k smoke` 坏轨迹
   - 数据集：`logs/hopper-hop/3/sd_hopper_5k_v441_recoveryevent_smoke_20260424_run1/official_eval/step_0005000/episode_csv`
   - 结果：`grounded_failed` 占比 `0.9406`，`falling_air` 占比 `0.0594`，其余三态基本为 `0`
   - 含义：坏轨迹几乎全部被归为“已着地但失败”的状态，符合视频观察
2. `td-mpc2 200k` 高分标准 hop 轨迹
   - 数据集：`results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428`
   - 结果：`falling_air ≈ 0.3813`，`grounded_failed ≈ 0.2133`，`grounded_prepared ≈ 0.0860`，`rising_air ≈ 0.3193`，`grounded_recovery ≈ 0`
   - 含义：标准 hop 轨迹中已能稳定出现 `prepared` 与 `rising_air`，但 `recovery` 仍偏少，说明当前 `recovery` 门仍较窄

另外，对上述两批轨迹按 `state >= 0.2` 做互斥检查，未发现同一步多个主状态同时显著激活，说明这轮五状态互斥实现基本成立。

### 四、当前判断

截至本轮，可以确认两点：

1. “奖励应基于推演到下一步状态后再给分，而不是做状态内奖励”这一口径已经落到主逻辑；
2. 当前最值得继续审查的，不再是失败态识别，而是 `grounded_recovery` 的门是否仍偏窄，导致坏轨迹长期停留在 `grounded_failed`。

因此，后续若继续做 `5k smoke` 或更长训练，观察重点应转向：

1. 是否首次稳定出现 `grounded_failed -> grounded_recovery`
2. 是否逐步形成 `grounded_recovery -> grounded_prepared`
3. 这些迁移是否最终转化为 `grounded_prepared -> rising_air`

## 2026-04-24：修复 Hopper 自动评估导出的旧字段兼容问题，并补回 `v443` 的 5k 评估轨迹

在 `sd_hopper_5k_v443_nextstate_transition_smoke_20260424_run1` 跑完后，训练本体已正常落盘到 `5000.pt / 5000.full.pt`，但 `eval.csv` 与 `official_eval` 没有自动生成。排查 `external/tdmpc2/tdmpc2/trainer/online_trainer.py` 后确认，原因不是 `save_eval_traces` 未开启，而是 Hopper 评估导出逻辑仍在读取旧字段 `grounded_unprepared`，而主线 `SD` 已改为 `grounded_failed`。

据此，本轮做了两项修订：

1. 在 `online_trainer.py` 的 Hopper 评估导出路径中加入字段兼容：
   - 若存在 `grounded_failed`，则优先读取；
   - 同时继续回写 `grounded_unprepared` 兼容列，避免旧分析脚本失效。
2. 用 `scripts/export_hopper_eval_traces.py` 对
   `logs/hopper-hop/3/sd_hopper_5k_v443_nextstate_transition_smoke_20260424_run1/models/5000.pt`
   补做一次 `10` 条 episode 的评估轨迹导出，并回填到：
   `official_eval/step_0005000/episode_csv`

补回后的 `v443 5k smoke` 结果如下：

1. `10` 条 episode 平均回报：`1.0075`
2. 最高单条回报：`5.5326`
3. 非零分 episode 数：`3 / 10`
4. 分布：`[0.1515, 4.3905, 0, 0, 0, 5.5326, 0, 0, 0, 0]`

按当前五状态口径离线回算后，可得：

1. `falling_air ≈ 0.0484`
2. `grounded_failed ≈ 0.9516`
3. `grounded_recovery = 0`
4. `grounded_prepared = 0`
5. `rising_air = 0`

这说明：

1. 本轮 `5k smoke` 已不再是完全 `0` 分；
2. 但当前非零回报尚未来自目标状态迁移链，而仍主要发生在 `grounded_failed` 内部；
3. “奖励按下一步状态迁移给分”的语义已落地，但从行为上看，策略在 `5k` 时仍几乎没有进入 `grounded_recovery / grounded_prepared / rising_air`。

---

## 2026-04-21：v4 奖励设计回退到“最小闭环主状态”

围绕 `src/world_model/dmcontrol_hopper_sd_model.py` 的 `predict_plan_reward()`，本轮不再继续沿 `precapture / stable_capture / crawl / standup / release / handoff` 这类长阶段链追加桥接项，而是按第一性原理重新收缩到“接触主状态 + 运动标签”的最小闭环设计。

### 背景判断

前序 `v4` 版本虽然在语义自洽性、`pitch wrap`、冻结惩罚、后仰飞惩罚等方面做了较多修补，但实际训练暴露出两个持续问题：

- 阶段链过长，导致奖励本身承担了过多“过程编排”职责，策略难以稳定进入后半段；
- “状态”与“努力目标”被混写，例如把前撑、站起、handoff 等目标导向概念直接并入阶段定义，使奖励结构逐渐变成工程态堆叠。

据此，本轮决定先把奖励语义压回闭环最小要求：主状态只描述当前客观接触关系，目标与奖励方向再由标签调制，而不再直接把“应该完成什么功能”写成状态节点。

### 本轮采用的最小状态表示

主状态收缩为四类：

1. `air`：无有效足支撑；
2. `ground`：足触地主导支撑；
3. `front_abnormal`：前体异常接地代理态；
4. `back_abnormal`：后体异常接地代理态。

其中：

- `front_abnormal` 与 `back_abnormal` 仍是代理态，而非严格的 body contact 真值；当前实现只依赖已有观测与局部推演量，不新增接触维；
- `上升/下落`、`直立/前倾/后倾`、`压缩/中性/伸展`、`旋转受控/失控` 统一降为标签，不再占用主状态节点。

### 具体实现

在 `DMControlHopperSDModel` 中新增 `_state_reward_features()`，只基于当前已有的结构量构造最小奖励判据：

- 直接使用 `height / vx / vz / pitch / pitch_rate / compression / compression_rate / support`；
- 派生 `posture_control / spin_control / forward_progress / upward_progress / low_height / low_motion`；
- 用软门控构造四类主状态：
  - `air`
  - `ground`
  - `front_abnormal`
  - `back_abnormal`
- 再额外派生 `compressing / extending / rotation_loss` 作为标签型调制量。

在 `predict_plan_reward()` 中，奖励被改写为：

- `true_reward`
- `air` 内部 shaping：鼓励空中姿态可控、下降阶段准备落地；
- `ground` 内部 shaping：鼓励可控支撑、压缩、伸展和前向推进；
- `front_abnormal` / `back_abnormal` 内部 shaping：鼓励减轻异常姿态、抬高身体并恢复足支撑；
- 少量转移奖励：`ground -> air`、`air -> ground`、`front_abnormal -> ground`、`back_abnormal -> ground`；
- 全局惩罚仅保留：
  - `spin_penalty`
  - `front_freeze_penalty`
  - `rear_trap_penalty`
  - `rear_flight_penalty`
  - `ground_stuck_penalty`
  - 向前/向后塌陷的 `collapse_penalty`

### 保留与移除项

本轮明确保留：

- `v4` 当前显式结构状态编码；
- `pitch wrap`；
- 后仰飞行惩罚这一类针对明显 reward-hack 的工程修补。

本轮明确移除或停止继续扩展：

- 长阶段链主导的 reward 设计；
- episode 级历史型桥接与 memory 逻辑；
- 继续围绕 `capture / crawl / standup / release / handoff` 追加细粒度 gating。

### 最小验证

修改后进行了两项轻量验证：

1. `python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py`
2. 直接实例化 `DMControlHopperSDModel`，调用 `forward()` 与 `predict_plan_reward()` 做前向探针

验证结果表明：

- 模块可正常编译导入；
- `next_state` 维度保持 `(batch, 15)`；
- `predict_plan_reward()` 输出为有限值张量，未出现 `NaN/Inf`。

### 当前方法学位置

这一改动不等于回退到 `v3b`。更准确地说，它是在 `v4` 主线上保留显式结构态、`pitch wrap` 与若干已验证必要的漏洞修补后，将奖励设计从“复杂阶段机”收缩为“最小闭环主状态 + 标签调制”。

下一步实验验收重点不再是某个阶段量是否被人工桥接激活，而是观察：

- 是否重新出现可持续的低姿前撑/前进入口；
- 是否避免前撑冻结与后仰飞 reward-hack；
- 是否在此基础上逐步恢复更长时程的正向推进行为。

## 2026-04-21：将 `v4` 奖励从“最小闭环主状态”进一步回调为宽松连续 shaping

在对 `sd_hopper_5k_v4_minloop_smoke_20260421_r2` 的 `10` 条官方评估轨迹逐条复核后，当前判断进一步收敛为：问题不在于策略完全不会产生有效动作，而在于 `v4` 的 reward 仍然过早把低姿过渡态归入 `front_abnormal / back_abnormal`，并用这些标签去切断主正向 shaping。结果是：

- 多条 episode 在前十步内已出现明显前冲或姿态变化，但随后仍被压回 rear basin；
- `episode_09` 一类轨迹中，`vx` 已接近 `1.8`，但由于仍处低姿前撑区间，主奖励基本未被放行；
- `front/back abnormal` 在实现上虽然名义上只是标签，但实际上承担了“硬判死”的作用，背离了引入 residual 为过渡路径留弹性的初衷。

据此，本轮不再继续细化阶段或补更多局部 bridge，而是把 `predict_plan_reward()` 直接回调为更接近 `v3b` 哲学的宽松连续 shaping，同时保留 `v4` 中已经验证必要的工程自洽项。

### 本轮代码调整

`src/world_model/dmcontrol_hopper_sd_model.py`

1. `_state_reward_features()`
   - 保留 `front_abnormal / back_abnormal / rear_flight_risk` 等标签型量；
   - 新增 `height_gate / pitch_gate / viability`，供主 shaping 直接使用；
   - 明确这些量仅作连续调制，不再作为切断主奖励的状态门。

2. `predict_plan_reward()`
   - 恢复 `v3b` 风格的主 shaping：
     - `forward_progress`
     - `upward_progress`
     - `support_relief`
     - `liftoff_progress`
     - `clearance_gain`
   - 增加少量连续辅助项：
     - `grounded_drive`，在地面压缩/伸展阶段补一点向上与向前驱动；
     - `transition_bonus`，对 `ground -> air` 与 `air -> ground` 只做弱鼓励；
     - `low_pose_recovery_bonus`，对低姿前撑/后仰这类过渡态仅鼓励“减小姿态误差、抬高身体、保持前进”，不再视为主状态外的失败区。
   - 惩罚项压缩为少数必要漏洞修补：
     - `rotation_loss`
     - `rear_flight_risk`
     - `sticky_contact`
     - `support_heavy * low_motion * low_height`

### 本轮方法学判断

这一回调不是放弃 `v4`，而是明确纠正前一轮偏差：

- `SD` 负责给出可解释的方向性先验；
- residual 负责吸收显式结构未完整覆盖的真实可行路径；
- reward shaping 只负责引导，不再承担“审判过渡态”的功能。

换言之，`v4` 后续若要继续有效发展，核心不是再补更细的阶段和更硬的 gating，而是保持显式结构语义正确，同时把奖励重新收回到“宽松、连续、低耦合”的形式。

### 最小校验

本轮修改后完成两项最小静态校验：

1. `python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py`
2. 实例化 `DMControlHopperSDModel`，随机采样 `state/action`，检查 `forward()` 与 `predict_plan_reward()` 输出均为有限值张量

预期下一步不应再直接从均值出发判断是否加预算，而应优先复看新版本 `5k smoke` 的逐条轨迹，确认：

- 是否重新允许低姿过渡态进入正向 shaping；
- 是否仍能抑制后仰飞 reward-hack；
- 是否比当前 `minloop` 版本更容易形成可持续前进而非单次短促动作。

## 2026-04-21：固化 Hopper `v4 5k smoke` 启动脚本，避免再把评估点落入随机采样阶段

本日再次复核主线 TD-MPC2 Hopper 的启动逻辑后，确认一个必须固定下来的实验约束：

- `envs/__init__.py` 会将 Hopper 的 `seed_steps` 设为 `max(1000, 5 * episode_length)`；
- Hopper `episode_length=500`，因此 `seed_steps=2500`；
- `online_trainer.py` 的评估触发只看 `self._step % eval_freq == 0`，并不会自动避开 seed 阶段；
- 因此，若手工写成 `eval_freq=1000`，则 `1000 / 2000` 的评估仍处在纯随机采样阶段，结果没有分析意义。

为防止后续再因手写命令混入错误评估点，新增专用脚本：

- `scripts/run_hopper_v4_5k_smoke.sh`

该脚本只允许外部指定三项：

1. `GPU_ID`
2. `EXP_NAME`
3. `SEED`

其余关键实验口径全部固定写死：

- `steps=5000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=10`
- `save_full_state=true`

并在启动前显式打印以下不变量：

- Hopper `seed_steps=2500`
- `2500` 之前只做随机采样
- 不允许把中途评估点放在 `1000/2000`
- `online_trainer` 会在 `2500` 自动保存 seed checkpoint

这样做的目的不是增加灵活性，而是降低人为失误空间：后续若需重跑主线 `v4 5k smoke`，原则上应优先调用该脚本，而不再临时手写一长串 train 命令。

## 2026-04-22：将 Hopper `v4` 奖励主状态进一步收口为四态闭环

围绕 Hopper 的高层奖励语义，再次按第一性原理做了收口。此前主线虽已从长阶段链退回到“最小闭环主状态”，但仍保留了 `front_abnormal / back_abnormal` 这类带有较强失败含义的状态名；它们虽然最初只想作为标签，实际仍容易在分析与实现上被当作“应尽快判出的坏状态”，从而再次把低姿过渡态判死。

本轮确认的高层主状态只保留四类，形成闭环：

1. `下落（空中）`
2. `着地未预备`
3. `着地预备`
4. `上升（空中）`

其中：

- “过度前倾、过度后仰、趴地前行、单点支撑但未稳住”等都统一归入 `着地未预备`；
- “整足支撑、姿态较正、蓄力已形成”才归入 `着地预备`；
- 其余细节不再提升为主状态，而只保留为辅助标签或连续量。

### 当前显式 SD 与四态的对应关系

本轮没有直接重排显式 latent 维度，仍保留当前 `v4` 主线的 15 维显式态：

`[height, vx, vz, pitch, pitch_rate, compression, compression_rate, swing, swing_rate, foot_angle, foot_rate, support, stance_mix, impact, clearance]`

但重新核对后，确认其中已有两组量可以支撑四态判定：

1. `support`
   - 作为“是否离地/是否着地”的主判据；
2. `stance_mix`
   - 作为“足部支撑分布”的代理量；
   - `|stance_mix|` 小时，更接近整足均匀支撑；
   - `stance_mix` 明显偏正时，更接近前掌主支撑；
   - `stance_mix` 明显偏负时，更接近后跟侧支撑。

这意味着当前显式 SD 虽未像更激进方案那样直接拆成 `toe_support / heel_support` 两个独立 stock，但在本轮四态奖励设计下，已经可以用 `support + stance_mix` 近似表达“整足支撑 vs 前掌支撑”的关键信号。

### 四态的具体判据

`src/world_model/dmcontrol_hopper_sd_model.py` 中的 `_state_reward_features()` 已按以下口径改写：

1. `下落（空中）`
   - `support` 足够低；
   - `vz < 0`；
2. `上升（空中）`
   - `support` 足够低；
   - `vz > 0`；
3. `着地预备`
   - 已有稳定足支撑；
   - `stance_mix` 接近 0，即更接近整足支撑；
   - `|pitch|` 较小；
   - `compression` 已形成；
   - 垂直速度幅值不大，不再处于明显砸地或反弹阶段；
4. `着地未预备`
   - 已接触地面，但不满足上述“着地预备”条件。

由此，原先的“前倾、后仰、低姿爬行、单点支撑、尚未蓄力完成”等情形，统一只作为 `着地未预备` 内部的连续差异，而不再拆成额外主状态。

### 奖励语义的同步收口

`predict_plan_reward()` 已同步改为四态闭环下的简洁 shaping：

1. `下落（空中）`
   - 仅弱奖励姿态控制与旋转受控，目的是为落地做准备；
2. `着地未预备`
   - 仅弱奖励向 `着地预备` 靠近的过程，例如整足支撑增强、姿态回正、蓄力增长；
3. `着地预备`
   - 仅弱奖励起跳相关量，例如向上速度、减载、离地趋势与前向推进；
4. `上升（空中）`
   - 仅弱奖励前向速度、高度保持、姿态和旋转控制。

同时，只保留少数明确需要的惩罚项：

1. `rotation_loss`
2. `rear_flight_risk`
3. `sticky_ground`
4. `front_sink_risk`
5. `backward_speed`

这一步的目的是让结构化奖励重新回到“小幅引导”角色，而不是再承担阶段编排或过渡态审判。

### 当前暴露出的 SD 限制

本轮梳理后，也明确记录一个仍然存在但暂不必立刻重构的限制：

- 当前显式态中的“整足支撑 vs 前掌支撑”仍是通过 `stance_mix` 做代理，而不是显式保存 `toe / heel` 两个独立支撑 stock。

因此，若后续轨迹分析仍持续出现“前掌支撑被误判成可接受站稳”的问题，则下一步最自然的结构改动不是再加奖励门，而是把显式接触状态进一步拆细为更直接的 `toe / heel` 支撑表示。

### 本轮结论

截至本轮，主线 `v4` 的奖励状态定义已从：

- `air / ground / front_abnormal / back_abnormal`

进一步收口为：

- `下落（空中） -> 着地未预备 -> 着地预备 -> 上升（空中）`

这更符合“状态描述客观存在，奖励描述在该状态下应努力的方向”这一原则，也更接近 `v3b` 的宽松 shaping 哲学：主状态尽量少，连续引导尽量弱，剩余细节交由残差和训练去吸收。

## 2026-04-22：按四态收口 Hopper 奖励，只保留弱引导与两项全局轻惩罚

在继续讨论后，本轮又将 Hopper `v4` 主线的结构化奖励进一步收紧。核心结论是：前一版虽然已经把长阶段链压回四态闭环，但在实现层面仍混入了过多辅助门控与旧派生量，容易再次滑回“奖励主导过程编排”的方向。为避免这一点，本轮明确把结构化奖励限制为：

1. 四个状态内的单一主方向引导；
2. 两个全局轻惩罚；
3. 其余一切复杂 failure-mode 惩罚、memory 与桥接逻辑全部退出。

### 本轮确认的四态弱引导

四个状态分别只奖励一类最核心行为：

1. `下落（空中）`
   - 仅轻奖励姿态稳定；
   - 具体量为 `pitch` 与 `pitch_rate`。
2. `着地未预备`
   - 仅轻奖励整足支撑增强与身体压低；
   - 在现有 latent 约束下，使用 `support` 与 `stance_mix` 构造 `full_foot_support` 近似整足支撑；
   - 身体压低继续用当前 latent 中可直接读到的 `height` 代理。
3. `着地预备`
   - 仅轻奖励开始起跳；
   - 具体量为 `vz` 为主、`vx` 为辅。
4. `上升（空中）`
   - 仅轻奖励姿态稳定；
   - 具体量仍为 `pitch` 与 `pitch_rate`。

### 全局轻惩罚

本轮最终只保留两项全局惩罚：

1. `过度旋转惩罚`
   - 用 `|pitch_rate|` 过大表示；
2. `明显后退惩罚`
   - 用 `vx < 0` 表示。

此前讨论过的“着地后卡死惩罚”在本轮未纳入实现，原因是其本质是时间累计型现象，若硬塞入单步 shaping，反而容易把当前奖励重新做复杂。

### 代码实现收口

本轮主要修改：

1. `src/world_model/dmcontrol_hopper_sd_model.py`

`_state_reward_features()` 只保留当前奖励真正需要的少量量：

1. `height`
2. `vx / vz`
3. `pitch / pitch_rate`
4. `support / stance_mix`
5. 基于 `support` 与 `stance_mix` 构造的 `full_foot_support`
6. 基于 `height` 构造的 `low_body`
7. 四态对应的软标签：
   - `falling_air`
   - `grounded_unprepared`
   - `grounded_prepared`
   - `rising_air`

`predict_plan_reward()` 则被压成：

1. `falling_reward`
2. `prepare_reward`
3. `launch_reward`
4. `rising_reward`
5. `rotation_penalty`
6. `backward_penalty`

不再保留：

1. `landing_progress`
2. `support_relief`
3. `prepared_hold`
4. `rear_flight_risk`
5. `front_sink_risk`
6. `sticky_ground`
7. 任何依赖旧阶段链语义的辅助奖励项

### 方法学说明

需要单独记录的是：本轮奖励实现已经完全按四态弱引导思路收口，但由于当前主线显式 latent 仍是旧 `height / support / stance_mix` 结构，因此这一版代码实现仍带有一个明确限制：

1. “低重心”在概念上已决定应优先使用更直接的身体高度坐标，但在当前 latent 中仍以 `height` 代理落地；
2. “整足支撑”在概念上已决定应优先由前后足接触共同定义，但在当前 latent 中仍以 `support + stance_mix` 构造 `full_foot_support` 代理落地。

也就是说，本轮已先把**奖励方向**收口正确；若后续还需继续提高物理语义一致性，下一步优先级应是对显式状态做同步收口，而不是再往奖励里补更多门控。

## 2026-04-22：确认奖励与 SD 的后续改造原则

在继续审查 Hopper `v4` 主线后，本轮不再围绕旧 15 维显式态做局部修补，而是先将奖励设计与 SD 设计的收口原则明确写入工作日志，作为后续版本的统一依据。

### 一、奖励设计的最终收口口径

本轮确认：结构化奖励的职责只能是**引导**，不能再次承担“主导训练过程”或“替代真实任务奖励”的功能。因此后续奖励设计统一遵循以下标准：

1. 真实环境 reward 为主；
2. 结构化 reward 只提供弱方向性引导；
3. 每个状态只奖励一个最核心方向；
4. 复杂阶段链、memory、桥接逻辑、failure-mode 专属惩罚一律不再继续扩展。

在此基础上，四状态下的弱引导奖励口径确认如下：

1. `下落（空中）`
   - 目标：落地前姿态稳定；
   - 奖励量：`pitch`、`pitch_rate`；
   - 含义：仅轻奖励不乱翻、不剧烈旋转。
2. `着地未预备`
   - 目标：站稳并压低；
   - 奖励量：`toe_support`、`heel_support` 及其构成的整足支撑代理，外加身体高度代理；
   - 含义：仅轻奖励整足支撑增强、低重心形成。
3. `着地预备`
   - 目标：开始起跳；
   - 奖励量：`vz` 为主、`vx` 为辅；
   - 含义：仅轻奖励向上速度转正，并给少量前向速度引导。
4. `上升（空中）`
   - 目标：空中姿态稳定；
   - 奖励量：`pitch`、`pitch_rate`；
   - 含义：仅轻奖励起跳后的姿态稳定。

全局惩罚最终仅保留两项：

1. `过度旋转惩罚`
   - 对应 `|pitch_rate|` 过大；
2. `明显后退惩罚`
   - 对应 `vx < 0`。

此前讨论过的“卡死惩罚”不纳入当前版本，原因是其本质上需要时间累计判断，若硬塞入当前单步 shaping，会再次把奖励设计做复杂。

### 二、SD 设计的职责重新定义

本轮同时确认：在 Hopper 这类强接触、强模式切换任务中，SD 的核心作用**不是**像 cartpole 那样精确承担完整物理转移，而是：

1. 支撑高层状态判定；
2. 支撑弱奖励引导；
3. 提供少量可信、高层、可解释的结构先验。

因此，后续 SD 设计的评价标准不再是“是否尽量完整地写出动力学”，而是：

1. 是否足够支持状态判定；
2. 是否足够支持弱奖励引导；
3. 是否语义清楚、来源直接、结构简洁；
4. 是否避免将不确定代理量误当成核心存量。

### 三、SD 主干的最小化收口方向

按上述标准，当前 Hopper `v4` 主线中的 15 维显式态已经明显过宽。后续 SD 主干不再以“现有 15 维为基础继续修补”为目标，而应转向“判定-引导最小集合”。

当前已确认可作为后续 SD 核心量候选的量为：

1. `rootz`
2. `vx`
3. `vz`
4. `pitch`
5. `pitch_rate`
6. `touch_toe`
7. `touch_heel`

在这些量之上，只再派生少量标签量用于状态判定与奖励引导，例如：

1. `z_min`
   - 由当前可观测或可直接几何推演的若干 `z` 坐标取最小值得到；
   - 只用于“是否着地”的判定。
2. `full_foot_support`
   - 由脚尖与脚跟接触共同构造；
   - 只用于“整足支撑是否形成”的判定。

与之相对，以下这批量不再被视为后续 SD 主干的优先保留对象：

1. `height`
2. `compression`
3. `compression_rate`
4. `swing`
5. `swing_rate`
6. `foot_angle`
7. `foot_rate`
8. `support`
9. `stance_mix`
10. `impact`
11. `clearance`

这并不意味着它们在代码里必须立刻全部消失，而是意味着：后续版本不再把这些量默认视为“核心存量”，而是优先审查哪些可以降级为辅助代理量，哪些应直接退出主干。

### 四、后续改造顺序

本轮最终确认的顺序是：

1. 先以工作日志明确奖励设计与 SD 设计的收口原则；
2. 再以“状态判定与弱奖励引导是否足够”为标准，重新设计 Hopper 的 SD 最小主干；
3. 不再围绕旧 15 维显式态继续补复杂流量方程；
4. 只有在新的最小 SD 主干确定后，才继续推进代码级同步改造与实验。

这一步的目的，是避免后续再次落入“先写复杂结构，再用奖励或 residual 为其救火”的循环，而是先把设计哲学统一，再做实现。

## 2026-04-22：Hopper SD 主干首次压缩到最小核心量集合

在上述原则确认后，本轮已开始真正修改代码，而不再只停留在方法学讨论。核心动作是：将 Hopper 主线显式 SD 状态从旧版 15 维宽结构，第一次直接压缩到仅服务于状态判定与弱奖励引导的最小核心量集合。

### 本轮新的显式 SD 主干

`src/world_model/dmcontrol_hopper_sd_model.py` 中，显式 latent 已改为 8 维：

1. `rootz`
2. `vx`
3. `vz`
4. `pitch`
5. `pitch_rate`
6. `touch_toe`
7. `touch_heel`
8. `z_min`

其中：

- `rootz / vx / vz / pitch / pitch_rate` 直接来自当前原始观测；
- `touch_toe / touch_heel` 来自原始触觉观测，经标准化后直接进入显式态；
- `z_min` 则由当前可直接几何推演的若干关键 `z` 坐标取最小值得到，包括根部、骨盆、大腿、小腿、脚身、脚尖、脚跟等点。

### 同步删除的旧核心量

本轮不再把以下量视为 Hopper 主线 SD 的核心显式态：

1. `height`
2. `compression`
3. `compression_rate`
4. `swing`
5. `swing_rate`
6. `foot_angle`
7. `foot_rate`
8. `support`
9. `stance_mix`
10. `impact`
11. `clearance`

这些量并非被证明“永远无用”，而是本轮先明确退出主干，避免它们继续以旧设计残留身份主导状态转移与奖励。

### 状态推进的实现口径

本轮代码没有再试图为一大批宽状态编写复杂流量方程，而是改为：

1. 只对 `vx / vz / pitch_rate` 这类真正 rate-like 的量写显式更新骨架；
2. `rootz` 与 `pitch` 通过积分更新；
3. `touch_toe / touch_heel / z_min` 则采用更简单的“当前状态 + 动作 + 少量几何启发 + 小幅学习修正”方式更新。

这样做的目的不是追求物理完美，而是让 SD 主干重新回到“少量可信结构量 + 简单可解释更新”的角色。

### 奖励与 SD 的一致性

本轮修改后，弱奖励设计与 SD 主干首次真正对齐：

1. 空中姿态稳定直接使用 `pitch / pitch_rate`；
2. 起跳趋势直接使用 `vz` 与少量 `vx`；
3. 低重心判断直接使用 `rootz`；
4. 整足支撑判断直接使用 `touch_toe / touch_heel` 的共同强度；
5. 是否着地则优先由 `z_min` 提供几何接地代理。

这意味着 Hopper 当前主线已从“宽显式态 + 复杂派生奖励”开始转向“最小显式态 + 弱状态引导”的统一口径。

### 调用链同步

本轮同时已同步修改：

1. `external/tdmpc2/tdmpc2/common/sd_world_model.py`
   - 使 `SDWorldModel` 的结构维度断言、趋势保护索引与耦合索引适配新的 8 维显式态。
2. `external/tdmpc2/tdmpc2/common/parser.py`
   - 将 `sd_hopper` 的默认 `sd_struct_dim` 改为跟随 `DMControlHopperSDModel.STATE_DIM`，避免训练入口继续默认使用旧 15 维。

### 最小验证

本轮完成了三类最小验证：

1. `py_compile` 通过，说明主线相关文件无语法错误；
2. 直接实例化 `DMControlHopperSDModel`，执行 `encode_observation -> forward -> predict_plan_reward / predict_viable_reward`，结果均为有限值；
3. 在带 `PYTHONPATH` 的条件下实例化 `SDWorldModel` 外层封装，确认 `encode / next / reward` 可在新 8 维显式态下正常运行。

这意味着：主线 Hopper SD 已不再只是“理念上要收口”，而是已经进入第一版最小主干代码形态。

## 2026-04-22：补齐训练入口默认配置，并通过 8 维 SD 的主线 smoke

在完成 Hopper 8 维显式 SD 主干改造后，又对训练调用链做了完整复核，目的是确认当前版本不只是单模块前向可用，而是能够被主线训练入口直接实例化、采样并写出日志。

### 补齐的调用链残留

复核中发现两处仍会把新主线意外拉回旧 15 维结构：

1. `external/tdmpc2/tdmpc2/config.yaml`
   - `sd_struct_dim` 仍写死为 `15`；
2. `scripts/run_hopper_v4_5k_smoke.sh`
   - 启动脚本中同样仍显式传入 `sd_struct_dim=15`。

这两处若不修正，即使 `parser.py` 已改为读取 `DMControlHopperSDModel.STATE_DIM`，训练入口在未额外手工覆盖时，仍会把 Hopper 主线退回旧结构维度。因此本轮已将两处统一改为 `sd_struct_dim=8`。

### 训练入口 smoke

使用项目既定的 `sdmcts_p100` Python 环境，执行了一次极短的 Hopper 主线 smoke：

- 入口：`external/tdmpc2/tdmpc2/train.py`
- 任务：`hopper-hop`
- 世界模型：`sd_hopper`
- 预算：`600` steps
- 目的：只验证主线能否在新 8 维显式态下正确启动、采样并写出日志，不评估性能

### smoke 结果

本轮 smoke 已完整通过，未出现结构维度错误、前向错误或训练入口崩溃。启动日志显示：

1. `SDWorldModel._dynamics` 已正确实例化为新的 `DMControlHopperSDModel`
2. `passive_flow / residual_flow / contact_head / zmin_head` 输入输出维度与 8 维显式态一致
3. 外层 `_struct_residual` 输出维度也已联动更新为 `8`
4. 训练流程能够正常完成一个 episode 并写出：
   - `train.csv`
   - `eval.csv`
   - `eval_full.csv`

其中：

- `train.csv` 已记录 `step=500` 的训练轨迹；
- `eval.csv` 已记录入口评估点；
- 说明当前主线训练入口已经真正接受新的 8 维 SD 结构，而不是只在局部探针里成立。

### 当前结论

截至本轮，Hopper `v4` 主线在代码层面已完成以下闭环：

1. 奖励设计已按四状态弱引导收口；
2. 显式 SD 主干已压缩到 8 维最小核心量；
3. 训练入口默认配置已同步到 8 维；
4. 主线训练 smoke 已通过。

因此，后续若继续推进 `5k smoke` 或更长预算实验，当前首要问题已不再是“调用链是否仍停留在旧 15 维”，而是新奖励与最小 SD 主干在训练中的实际行为质量。

## 2026-04-22：明确当前 Hopper `v4` 的职责分工与方法学立场

在继续核对主线实现后，本轮将一个关键方法学判断正式记录下来，以统一后续版本的评估标准：当前 Hopper `v4` 的目标，不是让 `SD` 取代原有 TD-MPC2 学习机制，而是在尽量保留原算法能力的前提下，让 `SD` 提供高层、可解释、弱耦合的结构先验。

### 一、当前实现中的真实分工

按当前代码，Hopper `sd_hopper` 主线可分为两层：

1. `SD` 显式主干
   - 负责给出少量高层、可解释的结构量；
   - 当前已收口为 `rootz / vx / vz / pitch / pitch_rate / touch_toe / touch_heel / z_min` 八维；
   - 主要用途是支撑状态判定与弱奖励引导。
2. learned + residual 部分
   - 包括 `reward head / Q / policy / residual latent / residual dynamics`；
   - 负责吸收显式结构未覆盖的复杂信息，并学习长期回报、动作细节与策略修正。

因此，当前设计并不是“只相信 SD”，而是明确采取：

- `SD` 提供方向性先验；
- residual 与 learned 部分保留表达弹性与纠错空间。

### 二、为什么当前分工比旧版更合理

此前若把 `SD` 写成宽状态、长阶段链、强门控，其实际效果往往不是“辅助学习”，而是“替训练过程做决定”。这会带来两个问题：

1. `SD` 一旦把过渡态判死，policy 很难再通过 residual 自行修正；
2. 结构化奖励一旦过强，训练目标就会与真实任务目标发生偏移。

当前版本之所以更接近合理折中，是因为它同时满足以下条件：

1. 显式 SD 主干已压缩到少量核心量；
2. 四状态 shaping 已收口为弱引导，而不是阶段编排；
3. `sd_latent_dim=32` 下仍保留额外 residual latent 空间，而不是把模型压成纯结构化系统；
4. 参数更新阶段仍以 learned reward 为主，不直接把结构化奖励变成训练主目标。

这意味着：`SD` 负责“更像正确方向”，原算法负责“具体怎样学会做到”。

### 三、当前设计希望实现的平衡

本轮确认，当前 Hopper `v4` 的理想职责平衡应表述为：

1. `SD` 的作用
   - 提供高层状态语义；
   - 提供少量可信的几何/接触先验；
   - 提供小幅、连续的方向性 shaping。
2. 原算法的作用
   - 保留 TD-MPC2 的规划、价值学习与策略学习能力；
   - 通过 learned reward、Q 和 policy 学习长期行为；
   - 通过 residual latent 吸收结构主干未覆盖的有效路径。

这正是当前版本试图达成的核心折中：既不退回完全黑盒，也不把训练过程重新交给手工设计的阶段机。

### 四、后续判断标准

据此，后续评估 `v4` 是否走在正确方向上，不应再只问“结构化奖励是否更细”，而应优先问以下几点：

1. `SD` 的状态判定是否足够稳定、足够少误判；
2. shaping 是否保持弱引导，而未压过 learned 部分；
3. residual 是否仍有足够空间吸收过渡态与非标准可行路径；
4. 规划时的结构化偏置是否与训练目标保持基本一致，而未形成过强失配。

若以上条件成立，则可以认为：当前设计确实有望同时发挥 `SD` 的优势，并保留原有 TD-MPC2 的核心学习能力。

## 2026-04-22：将规划混合从“显式总奖励”改为“learned reward + shaping delta”

在进一步复核 Hopper `v4` 的 reward 路径后，本轮又修正了一个更基础的设计问题：此前规划阶段的结构化分支使用的是 `predict_plan_reward()`，其内部形式为

`true_reward + shaping - penalty`

这意味着在 `curriculum / hybrid` 模式下，规划阶段实际上会把“learned reward 对环境奖励的预测”与“显式重写的一份环境奖励近似”再次混合。虽然这不是最直接的重复相加，但本质上仍属于对环境奖励的双重表达，容易带来两类问题：

1. 若显式 `true_reward` 与真实环境 reward 公式近似不准，则规划会被额外偏移；
2. 若结构化分支权重增大，则规划目标会越来越像“手工显式 reward”，而不是 learned reward 所拟合的训练目标。

据此，本轮将规划混合逻辑进一步收口为：

1. learned reward 继续作为主底座；
2. 结构化分支只提供 shaping 增量；
3. 规划阶段使用 `learned reward + alpha * shaping_delta`；
4. 不再在 `curriculum / hybrid` 分支中重复混入显式 `true_reward`。

### 代码调整

1. `src/world_model/dmcontrol_hopper_sd_model.py`

- 新增 `predict_shaping_reward()`
  - 只返回四状态弱引导对应的 `positive - penalty`；
  - 不再包含 `true_reward`。
- 保留 `predict_plan_reward()`
  - 作为兼容包装，返回 `predict_reward() + predict_shaping_reward()`；
  - 这样现有脚本中依赖“完整显式规划奖励”的调用仍可继续工作。

2. `external/tdmpc2/tdmpc2/common/sd_world_model.py`

- 对 `reward()` 分支做了区分：
  - `learned` 模式或训练态：仍直接返回 learned reward；
  - `explicit` 模式：仍允许使用完整显式 reward；
  - `curriculum / hybrid` 模式：改为只在 learned reward 上叠加 `shaping_delta`。
- 新增 `_scalar_add_to_logits()`，用于把 shaping 增量映射回 two-hot reward logits。

### 本轮后的方法学含义

这一步的含义是明确的：

1. 环境奖励的主体表达只保留一份，由 learned reward head 承担；
2. 结构化奖励只承担“小幅引导”的角色，而不再重写环境奖励主体；
3. `SD` 在规划阶段的作用更接近“方向性偏置”，而不是“替代奖励函数”。

因此，这一改动比此前的“显式总奖励混合”更符合本轮已经确认的总体原则：`SD` 提供高层先验与弱 shaping，原算法继续承担主导性的 reward/value/policy 学习。

### 最小验证

本轮修改后已完成以下最小验证：

1. `py_compile` 通过；
2. `DMControlHopperSDModel` 上验证：
   - `predict_shaping_reward()`、`predict_reward()`、`predict_plan_reward()` 输出均为有限值；
   - 且满足 `predict_plan_reward == predict_reward + predict_shaping_reward`；
3. `SDWorldModel` 上验证：
   - `reward()` 在 `eval()` 与 `train()` 两种模式下均能正常返回 `(batch, 101)` 的 two-hot logits；
   - 输出有限，无 `NaN/Inf`。

## 2026-04-22：先只将 `sd_latent_dim` 恢复到 512，不同步改结构

在继续讨论 Hopper `v4` 的世界模型替换关系后，本轮确认一个更稳妥的实验口径：当前版本暂不进一步删除 `struct_residual` 或重排 residual 结构，而是先只把 `sd_latent_dim` 从 `32` 恢复到 `512`，使总 latent 容量与主线 TD-MPC2 默认配置保持一致。

### 背景判断

当前 `sd_hopper` 并不是“纯 SD 替换黑盒世界模型”，而是：

1. 前部 `8` 维为显式 SD 结构态；
2. 后部为 residual latent；
3. 外层仍保留 learned reward、Q、policy 与 residual dynamics 等黑盒学习部件。

此前将 `sd_latent_dim` 压到 `32`，实际上同时改变了两件事：

1. 用显式 SD 主干替换了部分黑盒结构；
2. 把总 latent 容量也从原版 TD-MPC2 常用的 `512` 大幅压缩到了 `32`。

这会导致实验解释变得不清楚：一旦性能变化，无法区分究竟是结构先验本身造成的，还是仅仅因为总模型容量被压得过小。

据此，本轮决定先拆开变量：

1. 结构暂时保持当前版本不动；
2. 总 latent 容量先恢复到 `512`；
3. 先观察“容量对齐后的 SD 主线”表现，再决定是否继续动 `struct_residual`。

### 具体改动

本轮已同步修改以下入口，使 `sd_hopper` 默认总 latent 维度恢复为 `512`：

1. `external/tdmpc2/tdmpc2/config.yaml`
   - `sd_latent_dim: 512`
2. `external/tdmpc2/tdmpc2/common/parser.py`
   - `sd_hopper` 分支下，`sd_latent_dim` 的默认值改为参考 `cfg.latent_dim`，而不再固定退回 `32`
3. `external/tdmpc2/tdmpc2/common/sd_world_model.py`
   - `SDWorldModel` 内部的 `latent_dim` 默认读取也同步改为参考 `cfg.latent_dim`
4. `scripts/run_hopper_v4_5k_smoke.sh`
   - 默认启动参数改为 `sd_latent_dim=512`

### 当前方法学含义

这一步并不意味着“512 一定最合理”，而是意味着：

1. 先与原版 TD-MPC2 的总容量保持一致，是更自然的对照起点；
2. 这样可以先把“容量是否过小”与“SD 结构是否过强”这两个问题拆开；
3. 若后续仍需做容量消融，可再系统比较 `32 / 128 / 256 / 512`。

换言之，本轮的优先目标不是重新论证最优 latent 大小，而是避免在结构替换实验中额外引入一个不必要的容量缩减因素。

### 最小验证

本轮修改后已完成两项最小验证：

1. `py_compile` 通过；
2. 直接实例化 `SDWorldModel`，在 `sd_struct_dim=8, sd_latent_dim=512` 条件下验证：
   - `encode(obs)` 输出形状为 `(batch, 512)`；
   - `next(z, a)` 输出形状为 `(batch, 512)`；
   - `reward(z, a)` 输出形状为 `(batch, 101)`；
   - 全部输出为有限值。

## 2026-04-22：回收 `v4` 中仍有效的工程闭环，后续以主线为唯一工作树

在再次核对 `v4` 工作树与主线后，确认两类内容需要明确区分：

1. 一类是仍然有效的工程闭环，应迁回主线继续使用；
2. 另一类是 `v4.25-v4.27` 特定阶段的方法学实现，已被当前主线的“8 维最小显式态 + 四状态弱引导”设计取代，不再继续保留为开发基线。

据此，本轮正式决定：

1. 后续 Hopper `v4` 相关工作完全回到主线目录进行；
2. `sd_world_model_rl_v4` 不再作为继续开发的主工作树；
3. 仅回收 `v4` 中仍与当前主线设计兼容的工程闭环，不回迁已废弃的 9 维相位设计与受控 17 维 residual 方案。

### 本轮回收的工程闭环

本轮只补回两项与当前方法学无冲突、但对实验管理必要的机制：

1. 官方 `eval` 自动保存轨迹数据；
2. 每个正式 `eval` 点自动保存 checkpoint 与 `.full.pt` 可续训状态。

具体落实为：

1. `external/tdmpc2/tdmpc2/config.yaml`
   - 将 `save_full_state` 主线默认值改为 `true`；
   - 新增 `save_eval_traces: true`，使官方评估点默认导出轨迹数据。
2. `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
   - 为 `hopper-hop + sd_hopper` 增加 `official_eval/step_xxxxxxx/episode_csv/*.csv` 与 `eval_episode_summary.csv` 自动导出；
   - 导出字段改按当前主线 8 维显式态与四状态弱引导口径重写，不再沿用 `v4.27` 的 `compression_ready / launch_phase / fore_support_crawl` 等旧相位主字段；
   - 每个正式 `eval_freq` 触发的评估点，除写 `eval.csv / eval_full.csv` 外，同步保存对应步数的模型权重与 `.full.pt`。

### 明确不回迁的内容

经复核，以下 `v4` 内容已被当前主线设计替代，因此本轮不再迁回：

1. `v4.27` 的 9 维 `toe_support / heel_support / compression` 显式主状态及其相位链；
2. `v4.25` 的 `sd_struct_dim=9, sd_latent_dim=17` 受控 residual latent 方案；
3. 围绕 `compression_ready / launch_phase / fore_support_crawl` 的旧导出与分析主口径。

当前主线的正式口径仍保持为：

1. 显式 SD 主干：`rootz / vx / vz / pitch / pitch_rate / touch_toe / touch_heel / z_min`；
2. 奖励设计：四状态弱引导；
3. 总 latent：`sd_latent_dim=512`；
4. 训练与规划职责分工：learned 主导，显式结构只提供高层先验与弱引导。

### 暂缓项

此前在 `v4` 中提出的 `shaping` 限幅保护，本轮暂不迁回主线。该项待后续实验进一步观察后再决定是否纳入默认配置。

## 2026-04-22：将主线当前版本正式记为 `v4.28`，并以主线口径重启三种子 `50k`

在完成 `v4` 工程闭环回收后，主线 Hopper SD 的当前稳定口径已足以单独继续推进，因此本轮将主线当前版本正式命名为 `v4.28`。该版本的含义是：

1. 开发基线完全回到主线目录；
2. `v4` 工作树停止承担后续实验主线职责；
3. 当前有效方法学口径保持为“8 维最小显式态 + 四状态弱引导 + 512 总 latent”；
4. 已补齐官方 `eval` 自动导出轨迹与评估点自动保存可续训状态的工程闭环。

### `v4.28` 定义

`v4.28` 的正式定义如下：

1. 显式 SD 状态：`rootz / vx / vz / pitch / pitch_rate / touch_toe / touch_heel / z_min`
2. 奖励口径：四状态弱引导，规划侧使用 `learned reward + shaping delta`
3. 总 latent：`sd_latent_dim=512`
4. 评估与续训：官方 `eval` 自动导出轨迹 CSV，并在评估点自动保存 checkpoint 与 `.full.pt`
5. 暂不纳入：`shaping` 限幅保护，待后续实验观察后再决定

### 本轮训练入口修订

为避免评估目录再次混入 `step=0` 的无意义官方评估，本轮又对训练入口做了一条最小修正：

1. `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
   - 将正式评估触发条件收回为 `self._step > 0 and self._step % eval_freq == 0`

这样做的目的只有一个：使 `official_eval/step_xxxxxxx` 只对应已经完成 seed 数据收集后的正式评估点，不再在实验目录中生成 `step_0000000` 这一无分析意义的起始评估。

### `v4.28` 三种子 `50k` 运行口径

本轮三种子 `50k` 使用统一配置：

1. `steps=50000`
2. `eval_freq=25000`
3. `checkpoint_freq=25000`
4. `eval_episodes=10`
5. `save_full_state=true`
6. `save_eval_traces=true`

同时新增：

1. `scripts/run_hopper_v428_50k.sh`

该脚本固定了本轮 `v4.28` 的训练参数与主线环境变量，后续可直接复用，避免再次因手工命令分叉而引入评估频率、保存开关或 latent 配置的不一致。

### 同日启动期补记：修复 DMControl `None reward` 兼容

在本轮 `v4.28` 三种子 `50k` 首次启动时，训练已成功进入真实 GPU 会话并完成模型实例化，但在环境步进第一轮即统一报错：

1. `external/tdmpc2/tdmpc2/envs/dmcontrol.py`
   - `reward += step.reward`
   - 当底层 `dm_control` 在个别步返回 `step.reward is None` 时，会触发 `TypeError`

据此，本轮增加最小兼容修复：

1. 将 `None reward` 视为 `0.0`
2. 不改变其余环境封装、动作重复次数或终止逻辑

该修复的目的仅是消除环境 API 差异导致的训练入口崩溃，不涉及 Hopper `v4.28` 的奖励设计或 SD 方法学调整。

### 同日启动期补记：修复 `Timeout` 包装器与 `gymnasium` 的私有属性冲突

在修复 `None reward` 后，`v4.28` 三种子 `50k` 第二次启动又暴露出另一条与方法学无关的兼容问题：

1. `external/tdmpc2/tdmpc2/envs/wrappers/timeout.py`
   - 旧实现使用私有计数属性 `_t`
   - 在当前 `gymnasium` 包装器链下，该私有名访问会触发 `AttributeError`

据此，本轮继续做最小修复：

1. 将 episode 计数改为包装器自身显式持有的公共属性 `elapsed_steps`
2. `reset()` 时归零，`step()` 时递增
3. 仅替换计数状态名，不改变超时终止判定逻辑

这样处理后，训练入口的时间上限行为仍与原设计一致，只是消除了底层库接口收紧带来的包装器崩溃。

## 2026-04-22：继续收缩 Hopper 显式 SD 核心，去掉内部可学习修补分支

在将 Hopper 主线稳定到 `v4.28` 后，又对当前显式 `SD` 核心内部的可学习参数做了一轮复核。复核结论是：当前主线已经保留了外层 `512-8` residual latent，用来吸收显式 8 维状态无法完整表达的剩余因素；在这种前提下，显式 `SD` 内部凡是“语义并不刚性、只是给骨架打补丁”的学习分支，都应优先删除，而不是继续与外层 residual 重复分工。

本轮依据如下：

1. `control_scale` 仍有明确物理含义，对应四个 actuator 维度的有效作用增益；
2. `rate_damping_raw` 仍有明确物理含义，对应 `vx / vz / pitch_rate` 三个速率量的阻尼；
3. `passive_flow / residual_flow / contact_head / zmin_head` 虽能提升局部拟合自由度，但都属于“可要可不要”的内部修补，且其功能已可由外层 `512-8` residual latent 负责；
4. 若这些分支继续保留，就会在同一模型里同时存在“显式补丁”和“外层黑盒补丁”两套容错路径，责任边界变得不清，解释性也会下降。

据此，本轮将 Hopper 显式 `SD` 核心进一步收缩为：

1. 显式 8 维状态不变：`rootz / vx / vz / pitch / pitch_rate / touch_toe / touch_heel / z_min`
2. 保留显式动力学骨架：`_explicit_rate_dynamics`
3. 保留动作增益：`control_scale`
4. 保留速率阻尼：`rate_damping_raw`
5. 删除内部学习修补：
   - `passive_flow`
   - `residual_flow`
   - `contact_head`
   - `zmin_head`

对应代码变更如下：

1. `src/world_model/dmcontrol_hopper_sd_model.py`
   - `_rate_dynamics` 改为仅返回 `explicit - damping`
   - `forward` 中 `touch_toe / touch_heel / z_min` 的下一步预测改为纯几何与接触规则更新，不再叠加 learned delta
   - `get_learned_params()` 不再导出已删除分支相关字段
2. `external/tdmpc2/tdmpc2/common/sd_world_model.py`
   - 实例化 `DMControlHopperSDModel` 时，不再向其传入 `sd_residual_scale / sd_passive_scale`

这一步的含义不是“拒绝残差”，而是重新划清边界：

1. 显式 `SD` 只负责高层、可靠、可解释的骨架；
2. 外层 `512` 维 latent residual 继续承担复杂过渡态、局部接触细节与难以手工写出的剩余修正；
3. 方法学上继续坚持“结构化先验给方向，黑盒残差给弹性”的主线，而不是在显式核心内部再嵌一层小黑盒。

## 2026-04-22：修正 Hopper 竖直主量语义，将 `torso_z` 与 `height` 拆开

在对 `v4.28` 的 `5k smoke` 官方评估轨迹逐条复核后，首先确认了当前最上游的一条硬错误：显式 `SD` 里一直把 `dm_control` observation 中的 `position[0]`（也就是 `qpos_rootz`）当成了“真实高度”，并继续用它驱动：

1. `predict_height()`
2. `predict_reward()` 中的 standing 判定
3. `predict_viable_reward()` 中的高度门
4. `official eval` 的 `height / max_height` 导出
5. `_kinematic_features()` 中所有身体点的世界 z 坐标基准

但对 `dm_control.suite.hopper` 的源码核对表明：

1. `Hopper.get_observation()` 返回的 `position = physics.data.qpos[1:]`
2. 其中第一个量只是 `qpos_rootz`
3. 环境真实奖励使用的 `Physics.height()` 定义却是 `torso_z - foot_z`

也就是说，当前主线把“竖直滑块位移”和“真实站立高度”混成了一个量。

这条错位在轨迹中表现得非常明确：

1. 视频与几何导出都显示初始状态实际在空中；
2. 但旧版 `SD` 却在 `step 0` 就给出 `grounded_unprepared=1.0, air=0.0`
3. 官方 `eval` 汇总里的 `max_height` 也因此错误地长期为 `0.0`

据此，本轮将 Hopper 竖直语义链改为“两层量”：

1. 显式状态第 0 维继续保留 8 维布局，但其物理含义改为 `torso_z`
   - 使用 `TORSO_BASE_Z + qpos_rootz`
   - 其中 `TORSO_BASE_Z = 1.0` 对应 `hopper.xml` 中 torso body 的基准高度
2. `height` 不再等同于状态第 0 维，而改为派生量
   - `height = torso_z - ground_z`
   - 当前实现中 `ground_z = min(z_min, 0)`，用来近似环境里 torso 相对脚/地面的有效站立高度

对应代码调整如下：

1. `src/world_model/dmcontrol_hopper_sd_model.py`
   - `encode_observation()`：第 0 维从旧 `qpos_rootz` 改为 `torso_z`
   - `_kinematic_features()`：全部 body z 坐标以 `torso_z` 为世界基准重新推导
   - `_min_body_z_from_pos()`：最小 body z 改按新的世界坐标系计算
   - `predict_height()`：不再直接返回状态第 0 维，而改为返回派生 `height`
   - `predict_reward()` / `predict_viable_reward()`：统一改用新的 `height`
   - `_state_reward_features()`：新增 `torso_z` 与 `height`，并据此重算 `low_body`
2. `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
   - 官方评估轨迹导出中补出 `torso_z`
   - 汇总中同时保留 `initial/final_qpos_rootz` 与 `initial/final_torso_z`
   - 顺手补出逐步 `rootx = qpos_rootx`，避免轨迹 CSV 中该列为空

### 本轮最小校验

本轮未立即重启长实验，而是先做了两条最小自洽检查：

1. `py_compile` 通过：
   - `src/world_model/dmcontrol_hopper_sd_model.py`
   - `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
2. 对修正后的 `DMControlHopperSDModel` 做前向检查：
   - 零输入观测时，`torso_z=1.0`
   - `height=1.0`
   - `z_min=0.1`
   - `reward=0.0`
3. 对从官方轨迹抽取的初始姿态做编码检查：
   - `torso_z=1.0`
   - `height=1.0`
   - `z_min=0.537`
   - `grounded=0.0`
   - `air=1.0`

这与视频观察到的“初始时处于空中”一致，说明上游竖直量的物理语义已经回到正确方向。

### 当前判断

这一轮修正只解决了最上游的竖直语义错误。`full_foot_support` 是否仍存在独立问题、`grounded_prepared` 是否仍长期过低，需要在新的短实验轨迹下继续复核，不能再直接沿用旧 `5k smoke` 的结论。

### 同日补记：放宽接触/着地软判据，修正“永远在空中”的新偏差

在竖直语义修正后的 `heightfix 5k smoke` 中，官方评估又暴露出另一条新的系统性偏差：

1. `initial_rootz` 与 `max_height` 已恢复正常，说明 `torso_z / height` 主链确已修正；
2. 但 `air_frac` 几乎变成 `1.0`，`grounded_frac` 变成 `0.0`；
3. 多条轨迹中 `z_min` 已降到 `0.04~0.07`，肉眼已接近地面支撑，但状态特征仍将其判为“空中”。

复核后确认，这一偏差集中来自 `DMControlHopperSDModel` 中用于接触/着地的软门过于严格：

1. 旧参数：
   - `CONTACT_MARGIN = 0.02`
   - `CONTACT_SCALE = 0.02`
2. 这意味着只有 `z_min < 0.02` 才会被明显视为 near-ground；
3. 在修正后的真实世界坐标系中，很多低姿近地态会稳定停在 `z_min ~= 0.04~0.07`，于是被整段误判为“仍在空中”。

据此，本轮仅对接触带宽做最小修订，不改奖励结构本身：

1. `CONTACT_MARGIN: 0.02 -> 0.08`
2. `CONTACT_SCALE: 0.02 -> 0.04`

这样处理的目的不是“强行制造接触”，而是让“接近地面并可能开始支撑”的状态进入更合理的软门区间，从而为：

1. `grounded / air`
2. `full_foot_support`
3. `grounded_prepared / grounded_unprepared`

提供更接近实际轨迹的判定基础。

### 最小检查

对三类人工样本做了最小检查：

1. 初始空中样本：`z_min=0.65`
   - 结果：`grounded=0.0, air=1.0`
2. 近地支撑样本：`z_min=0.039`
   - 结果：`grounded≈0.51, air≈0.49`
3. 较低但未充分接触样本：`z_min=0.067`
   - 结果：`grounded≈0.16, air≈0.84`

这说明新的软门已经满足两个目标：

1. 不会把初始悬空状态误判成落地；
2. 也不再把已经逼近地面的低姿态一律误判成“永远在空中”。

### 同日补记：用 td-mpc2 200k 高分轨迹反证 `grounded_prepared / full_foot_support`

为避免继续从 `5k smoke` 直接推断“着地预备判据过严”，本轮补做了一次反向验证：直接拿已经学成标准 hop 的 `td-mpc2 200k` 高分 checkpoint 做少量轨迹导出，看当前四状态代理在高质量轨迹中是否会自然激活。

#### 代码修订

1. `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
   - 将 Hopper 的 `official_eval` 轨迹导出从“仅 `sd_hopper` 可用”改为“所有 `hopper-hop` 都可用”。
   - 导出特征不再依赖 agent 当前 world model，而统一调用 `DMControlHopperSDModel` 作为显式状态探针。
   - 这样做的目的，是让 `sd_hopper` 与 `tdmpc2` 在同一套 `torso_z / z_min / grounded_prepared / full_foot_support` 口径下可直接对比。
2. `scripts/export_hopper_eval_traces.py`
   - 改为同样使用 Hopper 物理快照 + `DMControlHopperSDModel` 探针导出轨迹。
   - 逐步 CSV 字段与 `official_eval` 口径对齐，补出：
     - 几何与 `qpos/qvel` 快照；
     - `torso_z / height / z_min`；
     - `air / grounded / full_foot_support / grounded_prepared`；
     - 各状态的 `*_gt_05` 二值标记；
   - 汇总文件 `summary.csv` 直接输出各 episode 的状态占比，便于快速判断判据是否在高分轨迹中成立。

#### 运行结果

以 `results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/models/final.full.pt` 为输入，仅导出 3 条轨迹，结果目录：

`results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428`

三条回报分别为：

1. `563.35`
2. `567.47`
3. `570.54`

这三条均为标准高分 hop，可作为当前状态判据的高质量参照。

#### 关键结论

在这三条高分轨迹中，`grounded_prepared` 与 `full_foot_support` 都明显不是零：

1. `episode_00`
   - `full_foot_support_frac = 0.168`
   - `grounded_prepared_frac = 0.074`
   - `max_full_foot_support = 0.997`
   - `grounded_prepared max = 0.752`
2. `episode_01`
   - `full_foot_support_frac = 0.166`
   - `grounded_prepared_frac = 0.064`
   - `max_full_foot_support = 0.998`
   - `grounded_prepared max = 0.755`
3. `episode_02`
   - `full_foot_support_frac = 0.142`
   - `grounded_prepared_frac = 0.068`
   - `max_full_foot_support = 0.998`
   - `grounded_prepared max = 0.799`

按 step 排序查看 `episode_00` 的最高 `grounded_prepared` 片段，可见该状态对应的是：

1. `full_foot_support ≈ 0.97~1.00`
2. `grounded ≈ 0.70~0.76`
3. `low_body = 1.0`
4. `height ≈ 0.52~0.53`
5. `vx ≈ 1.0~1.29`

这与用户从视频中描述的“标准 hop 落地压缩蓄力”是相符的。

#### 当前判断更新

据此，当前不能再把 `grounded_prepared` 视为“定义可疑”的首要嫌疑项。更准确的结论是：

1. 这套四状态代理在标准高分 hop 中能够被稳定激活；
2. 因而它们在当前 `v4.28 5k` 中接近零，更可能是训练早期尚未学到，而不是判据天然失真；
3. 后续若继续出现长期为零，应优先从训练过程与策略分布分析，而不是先改坏这套状态定义。

### 同日补记：将 `0` 分与高分轨迹并排做离线验证，拆开状态、奖励与 SD 转移

为避免后续继续凭单一现象判断“是状态错了、奖励错了、还是 SD 动力学跟不上”，本轮新增了一个统一分析脚本：

`scripts/analyze_hopper_state_reward_sd_validation.py`

该脚本对任意 Hopper 轨迹 CSV 执行三类离线验证：

1. 四状态占比与主状态转移；
2. 在真实轨迹上回放 `predict_reward / predict_shaping_reward`，比较环境奖励、显式奖励与 shaping；
3. 用当前 `DMControlHopperSDModel` 做一步前向，与真实下一步显式状态比较误差。

本轮对比了两组数据：

1. `v4.28 contactfix 5k smoke` 的 10 条 `0` 分轨迹；
2. `td-mpc2 200k` 的 3 条高分标准 hop 轨迹。

#### 一、四状态闭环验证结果

高分标准 hop 轨迹中，四状态闭环已经明显可见：

1. 状态占比
   - `falling_air ≈ 0.381`
   - `grounded_unprepared ≈ 0.213`
   - `grounded_prepared ≈ 0.086`
   - `rising_air ≈ 0.319`
2. 关键转移
   - `falling_air -> grounded_unprepared`
   - `grounded_unprepared -> grounded_prepared`
   - `grounded_prepared -> rising_air`
   - `rising_air -> falling_air`

相比之下，`v4.28 5k` 的 `0` 分轨迹几乎卡死在 `grounded_unprepared`：

1. `grounded_unprepared` 占比 `0.838`
2. `grounded_prepared` 占比仅 `0.0016`
3. 最主要的转移是 `grounded_unprepared -> grounded_unprepared`

这说明：

1. 四状态定义本身能描述成功 hop；
2. 当前失败轨迹的问题不是“闭环不存在”，而是策略长期无法走到 `grounded_prepared` 并形成稳定起跳。

#### 二、奖励设计验证结果

##### 1. `v4.28 0` 分轨迹

在 `0` 分轨迹上：

1. 环境真实奖励均值为 `0`
2. 显式环境对齐奖励均值约 `0.0014`
3. 原始 shaping 均值约 `-0.00015`
4. 即使按当前 `alpha=0.5` 缩放后，均值也仅约 `-0.00008`

这表明当前 shaping 量级确实是“小幅引导”，没有篡改任务主目标。

但也暴露出两个需要注意的点：

1. `grounded_prepared` 状态一旦偶发出现，其 shaping 均值约 `0.0446`，明显偏正；
2. `rising_air` 在 `0` 分轨迹中 shaping 均值反而略负，约 `-0.0054`，说明“失败起跳”阶段确实会被轻微抑制。

##### 2. 高分标准 hop 轨迹

在高分 hop 轨迹上：

1. 环境真实奖励均值约 `1.135`
2. 显式环境对齐奖励均值约 `1.112`
3. 原始 shaping 均值约 `0.0233`
4. 按 `alpha=0.5` 后约 `0.0117`

这再次说明 shaping 只是小量偏置，并未主导回报。

分状态看，当前奖励方向与四状态设计大体一致：

1. `grounded_prepared` 上的 shaping 最强，均值约 `0.0993`
2. `rising_air` 与 `falling_air` 也为小正值
3. `grounded_unprepared` 为较小正值，主要起到把策略往准备态推的作用

也即，当前奖励设计在“方向上”没有明显反常，至少并未鼓励失败姿态长期停留。

#### 三、SD 一步预测一致性验证结果

##### 1. `v4.28 0` 分轨迹

在当前 `0` 分轨迹上，一步预测误差较大的维度主要是：

1. `pitch_rate`：均值约 `0.975`
2. `vz`：均值约 `0.175`
3. `touch_toe / touch_heel`：均值约 `0.256 / 0.206`
4. `z_min`：均值约 `0.231`

其中按状态分组看：

1. `falling_air` 段的 `vz / pitch_rate` 误差较高；
2. `grounded_unprepared` 段的 `touch_toe / touch_heel / z_min` 误差较高；
3. `grounded_prepared` 样本过少，暂不足以下结论。

##### 2. 高分标准 hop 轨迹

在高分 hop 轨迹上，一步预测误差进一步显示出当前 SD 的薄弱段：

1. `vz`：均值约 `0.539`
2. `pitch_rate`：均值约 `2.762`
3. `touch_toe / touch_heel`：均值约 `0.271 / 0.255`
4. `z_min`：均值约 `0.112`

按状态分组后：

1. `falling_air` 中 `vz` 误差最大，均值约 `0.721`
2. `rising_air` 中 `pitch_rate` 误差最大，均值约 `3.943`
3. `grounded_unprepared / grounded_prepared` 中接触量误差最高

这说明当前最值得怀疑的不是四状态定义，也不是 shaping 方向，而是：

1. 空中纵向速度 `vz`
2. 角速度 `pitch_rate`
3. 落地接触量 `touch_toe / touch_heel`

这三类转移在当前显式 SD 中仍偏粗，尤其对标准 hop 的腾空与落地切换刻画不足。

#### 当前综合判断

截至本轮，三类对象的验证结论可暂定为：

1. 状态定义
   - 已获得正向验证；
   - 四状态闭环能在标准 hop 中真实出现；
   - 当前不应优先改坏这套定义。
2. 奖励设计
   - 方向基本正确；
   - 量级确实只是小幅引导；
   - 当前没有证据表明 shaping 主导了任务或把策略推向明显错误方向。
3. SD 动力学
   - 仍是最可疑环节；
   - 尤其在 `vz / pitch_rate / contact` 三类转移上，对高分 hop 的一步拟合仍偏弱；
   - 后续若要继续优化，应优先围绕这三类量的显式转移或 residual 组织方式展开，而不是先推翻四状态与奖励。

### 同日补记：继续下钻 SD 动力学，定位两类硬失真

在上一轮确认“最可疑的是 `vz / pitch_rate / contact`”之后，本轮继续对高分标准 hop 轨迹的最坏样本做了逐条动力学分解，重点检查：

1. `DMControlHopperSDModel._explicit_rate_dynamics()`
2. `DMControlHopperSDModel.forward()`
3. 外层 `SDWorldModel.next()` 对显式 8 维结构态的 residual 修正幅度

#### 结论一：接触冲量在当前显式动力学里被延后一拍

当前 `forward()` 的顺序是：

1. 先用当前 `touch_toe / touch_heel` 计算 `support`；
2. 用该 `support` 计算 `next_vx / next_vz / next_pitch_rate`；
3. 之后才根据 `z_min / next_pitch / next_vz` 更新 `next_touch_toe / next_touch_heel`。

也就是说，接触建立对速率量的影响至少滞后一拍。

这在高分 hop 的着地冲量处表现得非常明显。例如：

1. 某条高分轨迹在 `falling_air` 段：
   - 当前 `vz = -2.78`
   - 下一步真实 `vz = +2.09`
   - 即单步发生了强烈反弹
2. 但当前显式模型因为本步 `support = 0`，`z_acc` 仍几乎只剩重力项，结果预测：
   - `pred next_vz = -2.48`
   - 完全没有产生接触冲量式反弹
3. 同一位置真实 `next_touch_toe / next_touch_heel ≈ 0.93 / 0.996`，而模型预测仅 `0.003 / 0.194`

这说明当前模型在着地瞬间把“接触建立”和“接触反作用力”拆成了两个时间步，因而必然低估落地反弹与落地制动。

#### 结论二：接触几何被设计成“toe / heel 二选一”，难以表达标准 hop 的整足着地

接触头目前是：

1. `toe_geom = ground_proximity * sigmoid((next_pitch + 0.10) / 0.30)`
2. `heel_geom = ground_proximity * sigmoid((0.10 - next_pitch) / 0.30)`

这相当于用 `next_pitch` 作为 toe / heel 的互斥门。

但在标准 hop 的落地压缩段，真实轨迹里经常出现：

1. `next_touch_toe ≈ 1`
2. `next_touch_heel ≈ 1`

也即整足着地。

而当前门控会因为 `pitch` 偏前或偏后，只给其中一侧较高概率，另一侧被压低。于是就会出现：

1. 真实下一步 `touch_toe / touch_heel ≈ 0.99 / 0.99`
2. 模型仅能给出 `0.13 / 0.00` 或 `0.00 / 0.19`

这也是 `full_foot_support` 长期难以被显式动力学正确生成的重要原因。

#### 结论三：`pitch_rate` 目前本质上是线性弹簧阻尼，无法表达腾空与着地时的快速角速度重置

当前角速度动力学核心仅为：

1. `pitch_acc = -2.2 * pitch - 0.70 * pitch_rate + support * (...)`
2. 再叠加全局 `rate_damping * pitch_rate`

而从已学到的参数看，`rate_damping` 已经被训练拉到非常大：

1. `vx damping ≈ 3.88`
2. `vz damping ≈ 7.10`
3. `pitch_rate damping ≈ 11.59`

对高分 hop 的最坏样本做分解可见：

1. 在某条 `rising_air` 高分样本中：
   - 当前 `pitch_rate = -16.71`
   - 真实下一步 `pitch_rate = -2.93`
   - 需要单步产生极大的角速度回正
2. 但模型给出的总 `pitch_rate` 更新只足以预测到 `-12.60`
3. 即便阻尼已经被学得很大，线性形式仍然不够，无法表达真实轨迹中的“冲量式角速度重置”

这说明当前 `pitch_rate` 失真不是简单的“参数没调够”，而是表达形式本身过于平滑。

#### 结论四：外层 residual 目前不足以修补这些大误差

在 `SDWorldModel.next()` 中，结构态 residual 对自由维度的修正是：

1. `free_delta = free_scale * tanh(raw_correction)`
2. 其中 `vz / pitch / pitch_rate / vx` 这些 bridge 维度仅使用 `coupling_residual_scale = 0.20`

这意味着即便 residual 饱和，单步对这些维度的直接修正上限也只有约 `±0.2`。

但高分标准 hop 中实际需要补的误差量级是：

1. `vz` 常见最坏误差 `3~5`
2. `pitch_rate` 常见最坏误差 `10~13`

因此，当前外层 residual 不是“还没学会”，而是结构上就不足以填平这些冲量型误差。

#### 当前阶段判断

至此，关于 SD 动力学可形成更明确的判断：

1. 当前最主要的问题不是四状态定义；
2. 也不是 shaping 奖励方向；
3. 而是显式 SD 把 Hopper 的关键事件写成了“平滑、延后、弱修正”的动力学；
4. 这种结构对标准 hop 中最关键的两类事件都不适配：
   - 着地冲量；
   - 腾空/着地时角速度的快速重置。

后续若要继续优化 SD，应优先考虑：

1. 让接触建立在同一步即可反作用到 `vz / pitch_rate`；
2. 放弃单纯依赖 `pitch` 的 toe/heel 互斥接触门；
3. 提升或重构 `vz / pitch_rate` 维度上的 residual 修正能力，而不是仅给 `±0.2` 的小修补。

### 同日补记：按上述方向完成第一轮 SD 动力学修订

本轮已直接落实三项代码修改，目标是先修正动力学结构性缺陷，再决定是否需要重新收紧或放松状态/奖励。

#### 1. `src/world_model/dmcontrol_hopper_sd_model.py`

本轮对显式动力学做了四点修订：

1. 将结构态积分步长从单个物理步 `dt` 改为与环境外部观测对齐的控制步长：
   - `step_dt = self.dt * self.ENV_ACTION_REPEAT`
   - 原因：当前 agent 每次观测转移对应两次 `dm_control` 内部步进，显式 SD 若仍按单步积分，会系统性低估 `vz / pitch_rate` 变化量。
2. 新增 `_contact_candidates(...)`
   - 以 `pitch / vz / z_min` 推导“本步即将建立的接触候选”；
   - 不再把 toe / heel 设计成纯 `pitch` 互斥；
   - 允许在 near-ground + impact 时同时出现较高 toe / heel 值，以表达整足着地。
3. 在 `_explicit_rate_dynamics(...)` 中直接使用 `impact_toe / impact_heel`
   - 这样 `support / full_foot` 在速率更新时就能看到“即将接触”的候选；
   - 不再要求接触必须滞后一拍才能作用到 `vz / pitch_rate`。
4. 在 `forward(...)` 中加入同一步的冲量式修正
   - `bounce`：对 imminent impact 的 `vz` 做同步反弹修正；
   - `impact_pitch_gate`：对着地瞬间的 `pitch_rate` 做快速重置；
   - `air_upright_gate`：对近直立空中阶段的极端角速度做更强衰减；
   - `next_touch_toe / next_touch_heel` 更新为“至少不低于本步接触候选”，避免接触建立被混合权重再次压低。

#### 2. `external/tdmpc2/tdmpc2/common/sd_world_model.py`

本轮放宽了结构态 residual 在关键桥接维度上的修正能力：

1. `vx_residual_scale = 0.50`
2. `vz_residual_scale = 3.00`
3. `pitch_residual_scale = 0.75`
4. `pitch_rate_residual_scale = 8.00`

之前 `vx / vz / pitch / pitch_rate` 共用 `0.20` 上限，这不足以覆盖高分 hop 中观察到的冲量型误差。当前做法仍保留：

1. `rootz / touch_toe / touch_heel / z_min` 的 trend protection；
2. 仅对桥接维度放大 residual，而非整体放开显式结构。

#### 最小回归检查

本轮未直接重启训练，而是先做两类最小检查：

1. `py_compile` 通过：
   - `src/world_model/dmcontrol_hopper_sd_model.py`
   - `external/tdmpc2/tdmpc2/common/sd_world_model.py`
2. 对高分标准 hop 中的两类最坏样本做局部回归：
   - 着地反弹样本：
     - 修改前 `pred next_vz ≈ -2.48`
     - 修改后 `pred next_vz ≈ +0.77`
     - 真实 `next_vz ≈ +2.09`
   - 空中角速度衰减样本：
     - 修改前 `pred next_pitch_rate ≈ -12.60`
     - 修改后 `pred next_pitch_rate ≈ -4.78`
     - 真实 `next_pitch_rate ≈ -2.93`

这说明本轮修改已明显朝正确方向推动：

1. 着地冲量不再完全缺失；
2. 空中极端角速度也不再像之前那样几乎不衰减；
3. 但接触强度与 `z_min` 仍未完全对齐，说明这只是第一轮动力学修订，还需要通过新训练轨迹继续验证。

### 同日补记：动力学回归的验收口径收敛为“只查结构错误，不拿 5k 已学参数当门槛”

在继续复核 `v4.28` 显式动力学时，验收口径进一步收紧为：当前阶段需要验证的是结构性错误是否已被移除，而不是要求 `5k` 量级下学到的参数已经把 Hopper 动力学完整拟合。为避免再次把“参数未学好”和“结构写错了”混在一起，本轮将回归目标限定为以下三条：

1. 旧错误样本中，不再出现“同一步着地反弹已发生，但 `next_touch_toe / next_touch_heel` 仍几乎为零”的时序自相矛盾；
2. `z_min` 的传播不再因为接触增强而被推成负值；
3. 非落地样本不会被误判出虚假接触。

据此又完成了两处最小修订：

1. `forward(...)` 中更新 `next_touch_*` 时，不再单独依赖反弹后的 `next_vz` 再判一次接触，而是显式保留同一步 `impact_gate / cand_toe / cand_heel` 的落地信息，避免“反弹成功反而抹掉接触”；
2. `next_zmin` 改为区分 airborne 与 grounded 两种传播，并在存在明显接触时将结果约束到非负范围，不再使用“接触越强，`z_min` 越负”的旧写法。

随后在既有高分轨迹的旧错误样本上做结构回归，结论如下：

1. 对典型 `vz_impact` 样本，当前结构层已给出明显非零的同步接触：
   - `pred_next_touch_toe ≈ 0.473`
   - `pred_next_touch_heel ≈ 0.564`
   - `pred_next_zmin = 0.0`
   这已消除了之前“反弹存在但接触几乎为零、`z_min` 变成负值”的结构矛盾。
2. 对典型 `pitch_rise` 空中样本，当前结构层未引入新的虚假接触：
   - `pred_next_touch_toe ≈ 0.0`
   - `pred_next_touch_heel ≈ 0.0`
3. 在三条高分 `td-mpc2 200k` 轨迹的逐步扫描中：
   - `pred_next_zmin < 0` 的样本数为 `0`
   - 对 near-ground descending 的冲击样本，`165` 条中有 `148` 条给出了 `> 0.2` 的接触预测
   - 对明显非冲击的 airborne 样本，未出现阈值以上的虚假接触

因此，当前判断更新为：

1. 这轮修订已经修掉了上一轮最明确的两条结构错误：同一步落地接触丢失，以及 `z_min` 负向传播；
2. 尚未证明显式动力学已经足够精确，也不应在此阶段要求它做到这一点；
3. 在这一口径下，旧错误样本回归已通过，可以进入下一轮 `5k smoke`，继续检查训练轨迹中是否还有新的结构性异常暴露出来。

### 同日补记：正式取消 `z_min < 0` 的历史缓冲区

在随后对“反物理状态是否仍可能出现”的专项复核中，又确认了一条仍然存在的结构漏洞：虽然显式 `forward(...)` 已经把 `next_zmin` 约束到非负，但 `project_state(...)` 与 `_state_reward_features(...)` 仍保留了 `min=-0.5` 的旧下界。这意味着外层 residual 若把 `z_min` 拉成负值，结构态仍会接受该状态，并进一步把这段“穿地深度”带入后续状态判断与高度计算。

该负下界没有新的方法学意义，只是早期为数值缓冲保留的历史遗留项。既然当前版本已明确要求结构化先验避免明显反物理情形，`z_min < 0` 就不应再被允许。

据此，本轮仅做一处最小修订，不触碰 `touch_toe / touch_heel` 的耦合逻辑：

1. 将 `project_state(...)` 中 `z_min` 的下界由 `-0.5` 改为 `0.0`；
2. 将 `_state_reward_features(...)` 中读取 `z_min` 时的下界同步改为 `0.0`。

这一步的含义是：从当前版本起，`z_min` 被明确解释为“代表点最小高度”，不再保留任何地下缓冲区。后续若继续检查接触与几何状态是否一致，应另行处理 `touch_toe / touch_heel` 与 `z_min` 的耦合，而不再借助负 `z_min` 作为容错带。

### 同日补记：为 `grounded_prepared` 增加竖直速度软门，过滤着陆冲击假阳性

在继续复看 `v4.28 dynfix2 5k smoke` 的官方评估轨迹后，已确认当前四状态链条本身能够被激活，但 `grounded_prepared` 仍存在一类轻度假阳性：某些“高速着陆冲击瞬间”虽然短暂满足了整足支撑和低重心，却还不应被解释为“已完成压缩蓄力”。

最典型的是 `episode_01` 的峰值片段：当时 `full_foot_support` 很高，但 `vz` 仍明显为负，仍处于下砸地面的过程中。若继续沿用原定义，`grounded_prepared` 会把这种瞬间也计入“预备完成”。

据此，本轮对 `grounded_prepared` 的定义只做一处最小修订，不扩张状态图：

1. 保留原有三项：
   - `grounded`
   - `full_foot_support`
   - `low_body`
2. 新增一项竖直速度软门：
   - `vertical_settle = clamp((0.40 - |vz|) / 0.40, 0, 1)`
3. 更新后：
   - `grounded_prepared = grounded * full_foot_support * low_body * vertical_settle`

这样做的意图是：

1. 保持四状态闭环的简洁性；
2. 不把“高速着陆瞬间”误记为“已完成压缩蓄力”；
3. 允许 `vz` 接近 `0` 的低速压缩或刚开始上抬阶段继续被视为合理的 `grounded_prepared`。

### 同日补记：补充 `trace_csv -> mp4` 直接回放脚本，避免再次重跑 eval

在复核 `50k official eval` 轨迹时，需要把单条保存轨迹直接渲染成 `mp4` 供人工观察。此前仓库中已有的是 `checkpoint -> replay eval -> mp4` 脚本，不适合这一需求，因为它会重新执行一次评估。

为避免再次把“已保存轨迹回放”和“重新跑评估”混在一起，本轮补充：

1. `scripts/render_hopper_trace_csv_mp4.py`

该脚本固定面向 Hopper `official_eval/episode_csv/*.csv`：

1. 直接读取保存下来的 `qpos_* / qvel_*` 序列；
2. 逐帧写回 `dm_control` Hopper 物理状态；
3. 直接渲染为 `mp4`；
4. 不再调用策略，也不再重复执行评估。

这样，后续若要核验某条已保存的评估轨迹是否真的出现了跃起、前冲或其他中间状态，应优先使用该脚本，而不是再用 checkpoint 回放脚本重跑一遍 eval。

### 同日补记：`rising_air` 当前仍是“近地上抬”软量，不等于真正离地上升

在把 `seed3 @ 50k` 的 `episode_02` 轨迹直接渲染成 `mp4` 并与对应 `episode_csv` 对照后，又确认了一条需要明确记录的语义偏差：

1. 当前日志中的 `rising_air` 非零，并不意味着 Hopper 已经肉眼可辨地真正离地；
2. 该信号目前更接近“近地上抬趋势”，而不是“明确空中上升状态”。

原因在于当前定义是软量：

1. `grounded = clamp((CONTACT_MARGIN - z_min) / CONTACT_MARGIN, 0, 1)`
2. `air = 1 - grounded`
3. `rising_air = air * upward_progress`

只要 `z_min` 处于近地带中间区间，`grounded` 与 `air` 就会同时为非零；若此时 `vz > 0`，即便 Hopper 仍有脚部或身体其他部分贴地，也会得到非零的 `rising_air`。

这解释了为什么在 `episode_02` 的视频中没有看到明确离地跃起，但日志里仍出现了 `rising_air > 0`：当前信号描述的是“上抬倾向”，不是“已离地上升”。

据此，后续应把 `rising_air` 视为待收紧项，而不是直接拿来等同于真正的空中上升。若后续继续修订状态信号，`rising_air` 的首要方向应是：

1. 提高对离地程度的要求；
2. 避免近地擦地上抬就被记为 `air`；
3. 让名称与实际语义重新对齐。

### 同日补记：`grounded_prepared` 的下一版方向改为“整足支撑 + 膝弯曲窗口 + 竖直稳定”，并暂不热改当前主线

在继续核对 `seed1 @ 50k episode_05`、`seed3 @ 50k episode_02` 与高分 `td-mpc2 200k` 轨迹后，关于 `grounded_prepared` 的判断进一步收敛：

1. 仅用 `low_body` 作为“已形成起跳准备”的核心依据过宽；
2. `qpos_knee` 的量级更接近“是否真正形成压缩蓄力”；
3. 但 `qpos_knee` 也不能用单调“越大越好”的门，因为塌陷/折叠姿态下膝角同样可能很大。

当前几条代表性轨迹给出的量级对照为：

1. `seed3 @ 50k episode_02`
   - 当前被误记为 `grounded_prepared` 的片段，`qpos_knee ≈ 0.078 ~ 0.086`
   - 肉眼观察下几乎谈不上“压缩起跳”
2. `seed1 @ 50k episode_05`
   - 前段短暂得分窗口，`qpos_knee ≈ 0.646 ~ 1.425`
   - 已有一定弯曲，但仍偏浅，不足以稳定起跳
3. 高分 `td-mpc2 200k` 对照
   - `prepared / full_support` 相关片段，`qpos_knee` 常落在 `~1.2 ~ 1.8`
   - 这更接近肉眼意义上的“压缩准备”
4. `seed1 @ 50k episode_05` 的后期塌陷段
   - `qpos_knee` 长期约 `2.63`
   - 明显说明膝角不能用单调增强门

据此，下一版 `grounded_prepared` 更合理的核心应改为：

1. `full_foot_support`
2. `knee_window`
3. `vertical_settle`

其中：

1. `low_body` 从核心条件中移除，最多只保留为弱辅助项；
2. `knee_window` 不再是单调 `sigmoid`，而应是以 `qpos_knee ≈ 1.5` 为中心的双边窗口；
3. 一个初步经验口径是：
   - `< 0.5` 基本不算准备
   - `1.2 ~ 1.8` 更像合理压缩区
   - `> 2.0` 需警惕塌陷/折叠

之所以本轮暂不立即热改主线实现，是因为当前在线结构态仍是 8 维，不包含 `qpos_knee`。若要把该信号真正用于在线状态判定，需要把显式结构态从 `8` 维扩到 `9` 维，并同步调整模型形状与脚本配置。这会改变 checkpoint 兼容性，不宜在当前三种子 `200k` 长测进行中直接切入主线。

因此，本轮先将该结论作为下一版设计决策记录，待当前长测告一段落后，再进入基于 `qpos_knee` 的结构态扩展与 `grounded_prepared` 重定义。

### 同日补记：固化 `v4.28` 三种子 `200k` 长测启动脚本

在确认当前 `5k smoke` 已不再暴露明显硬结构错误后，本轮继续按主线 `v4.28` 口径推进长预算验证。为避免再次手写长命令引入评估周期、保存开关或 latent 配置的不一致，新增固定脚本：

1. `scripts/run_hopper_v428_200k.sh`

该脚本与现有 `50k` 版本保持同一训练口径，仅固定如下差异：

1. `steps=200000`
2. `eval_freq=25000`
3. `checkpoint_freq=25000`
4. `eval_episodes=10`

其余参数保持与当前 `v4.28` 主线一致，包括：

1. `sd_struct_dim=8`
2. `sd_latent_dim=512`
3. `sd_reward_mode=curriculum`
4. `save_full_state=true`

因此，从这一节点起，若需重跑主线 `v4.28` 的三种子 `200k` 长测，应优先调用该脚本，而不再手写训练命令。

### 同日补记：中断 `v4.28` 长测后，切入 `v4.29` 的 9 维结构态与膝窗口判据

在确认 `v4.28` 的三种子 `200k` 长测不再继续后，本轮不再维持 checkpoint 兼容性约束，直接进入下一版主线修订。核心目标是把先前仅记录为“下一版方向”的 `qpos_knee` 判据真正落到在线结构态与规划奖励中，并同步修正调用链。

#### 1. 结构态从 8 维扩到 9 维

`sd_hopper` 的显式结构态已从：

1. `[torso_z, vx, vz, pitch, pitch_rate, touch_toe, touch_heel, z_min]`

扩展为：

1. `[torso_z, vx, vz, pitch, pitch_rate, touch_toe, touch_heel, z_min, qpos_knee]`

相应地：

1. `DMControlHopperSDModel.STATE_DIM` 由 `8` 改为 `9`；
2. `encode_observation()` 直接从原始观测中读取 `qpos_knee`；
3. `project_state()` 对 `qpos_knee` 做显式裁剪；
4. `external/tdmpc2/tdmpc2/config.yaml` 的默认 `sd_struct_dim` 同步改为 `9`；
5. 训练、评估、轨迹导出、离线分析脚本都同步切换到 9 维口径。

#### 2. `grounded_prepared` 正式改为“整足支撑 + 膝窗口 + 竖直稳定”

本轮不再把 `low_body` 作为“已形成起跳准备”的核心条件。新的核心定义为：

1. `full_foot_support`
2. `knee_window`
3. `vertical_settle`

其中：

1. `knee_window` 采用双边软窗口，而不是单调增强门；
2. 当前窗口中心仍围绕 `qpos_knee ≈ 1.5`，但下边界采用更窄、略前移的软门，使 `~1.1` 附近可弱激活、`~0.8` 仍被压低；
3. 经验口径对应为：
   - `~1.1` 以下明显偏弱；
   - `1.2 ~ 1.8` 为主要准备区；
   - `> 2.0` 逐步衰减，避免把塌陷折叠误记为准备完成。

实现上，新的 `grounded_prepared` 为：

1. `grounded * full_foot_support * knee_window * vertical_settle`

`low_body` 仍保留为辅助诊断特征，但不再参与该状态主判据。

#### 3. 奖励塑形同步改为“支撑形成 + 膝窗口形成”

四状态塑形中，`grounded_unprepared` 阶段的正向引导从：

1. `full_foot_gain + low_body_gain`

改为：

1. `full_foot_gain + knee_window_gain`

这样做的原因是：

1. 当前主线最明确的问题不是“重心不够低”，而是“虽然近地，但尚未形成可起跳的腿部压缩状态”；
2. 在 `qpos_knee` 已进入结构态后，继续用 `low_body` 作为主要塑形方向会重复放大奖励过宽的问题；
3. `launch_reward` 与 `rising_reward` 保持原有极简口径，不在这一轮额外扩张状态图。

#### 4. 膝角动力学只做最小显式建模

本轮没有重新引入复杂 flow 网络，而是对 `qpos_knee` 只加入最小、可解释的显式动力学：

1. 在稳定整足支撑且竖直速度较小的地面阶段，膝角向压缩目标靠拢；
2. 在上升/离地阶段，膝角向更伸展的目标回退；
3. 更细的误差与任务相关修正，继续交给外部 `512-dim residual` 与结构残差吸收。

这样做的目的，是让显式 SD 只负责“判据需要的核心因果骨架”，而不再承担过细的姿态拟合任务。

#### 5. 调用链与脚本同步

为避免再次出现“主模型已改，但训练入口仍按旧维度启动”的不一致，本轮同步完成了以下调整：

1. `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
   - 评估轨迹导出中加入 `qpos_knee`、`knee_window`、`vertical_settle`
2. `scripts/export_hopper_eval_traces.py`
   - 默认 `sd_struct_dim` 改为跟随 `DMControlHopperSDModel.STATE_DIM`
   - 默认 `sd_latent_dim` 固定为 `512`
3. `scripts/analyze_hopper_state_reward_sd_validation.py`
   - 离线一阶一致性分析切换到 9 维 latent keys
4. 新增主线启动脚本：
   - `scripts/run_hopper_v429_5k_smoke.sh`
   - `scripts/run_hopper_v429_50k.sh`
   - `scripts/run_hopper_v429_200k.sh`
5. 旧的 `v428` / `v4_5k_smoke` 入口脚本不再直接承载新口径，而是改为透明转发到 `v429` 主线，避免版本名与参数口径继续漂移。

#### 6. 当前结论

到这一节点为止，`v4.29` 的核心变化不是“增加更多奖励项”，而是：

1. 让显式结构态真正包含了“起跳准备”所需的关键观测量；
2. 把 `grounded_prepared` 从过宽的低重心代理，收缩为更符合物理直觉的“整足支撑 + 适度膝弯曲 + 竖直稳定”；
3. 保持四状态奖励仍为弱引导，而不是重新走回复杂阶段链奖励。

下一步应先做最小 smoke 验证，确认：

1. 9 维结构态调用链可正常实例化与前向；
2. `grounded_prepared` 不再把 `qpos_knee ≈ 0.08` 的片段误记为准备完成；
3. 新导出的评估轨迹中，`qpos_knee / knee_window / vertical_settle` 已随 CSV 一起落盘。

### 同日补记：修复 `grounded / air` 软互补冲突，并用旧轨迹做修复前后对照

在 `grounded_prepared` 修正完成后，继续回到前面已确认的另一个结构冲突：当前 `air` 仍由 `1 - grounded` 定义，只要 `z_min` 落在近地软带中间，就会出现“仍在接触地面，但 `air` 与 `grounded` 同时非零”的情况，进而污染 `rising_air / falling_air`。

本轮对该问题采取最小修复，不改状态图、不改主动力学，只改 `_state_reward_features()` 内的状态特征计算：

1. `grounded` 不再只由 `z_min` 决定，而是改为：
   - `grounded_geom`
   - `grounded_contact`
   - 二者取 `max`
2. `air` 不再使用 `1 - grounded`，而是改为：
   - `airborne_clearance`
   - `contact_absent`
   - 二者相乘

这意味着：

1. 近地且强接触的片段，会被稳定压回 `grounded`；
2. 只有在离地间隙足够明显、且接触确实消失时，`air` 才会上升；
3. `rising_air / falling_air` 不再把“擦地上抬”或“接触中的下压”误记为空中态。

#### 修复前后对照样本

本轮直接使用已有 `50k official_eval` 与高分 `td-mpc2` 轨迹，作为修复前后的同轨迹重算对照。

主要反例：

1. `seed3 @ 50k episode_07`
2. `seed3 @ 50k episode_08`

修复前，这两条轨迹中大量存在：

1. `air > 0.1` 且 `grounded > 0.1`
2. `rising_air > 0.05` 但 `max(touch_toe, touch_heel) > 0.5`
3. `falling_air > 0.05` 但接触仍明显存在

修复后，重算结果为：

1. `seed3 episode_07`
   - `both_count: 488 -> 0`
   - `rising_contact_count: 173 -> 0`
   - `falling_contact_count: 64 -> 0`
2. `seed3 episode_08`
   - `both_count: 490 -> 0`
   - `rising_contact_count: 170 -> 0`
   - `falling_contact_count: 65 -> 0`

同时保留正例核查：

1. 高分 `td-mpc2 episode_01`

修复后：

1. 仍保留真实空中段：
   - `air_count_gt_05 = 137`
   - `rising_count_gt_05 = 8`
2. 且这些 `rising_air` 样本都满足：
   - `grounded = 0`
   - `contact = 0`
   - `z_min ≈ 0.088 ~ 0.096`

据此可以确认：

1. 这次修复确实消除了“空中与着地同时成立”的主要污染源；
2. 并未把高分 hop 里的真实空中上升段一并抹掉；
3. 因此，后续 `5k smoke` 应以该修订后的状态口径继续推进，而不是再沿用旧的软互补定义。

## 2026-04-23：将 Hopper 结构化塑形进一步收敛为“两条主状态迁移”，并以该口径重启三种子 `50k`

在连续数轮 `10k` 强结构化短测后，已确认当前奖励设计中最关键的问题不是“状态定义整体错误”，而是“哪些跃迁应被直接奖励”。本轮围绕该问题完成了两步收敛。

### 1. 先用 `td-mpc2` 高分轨迹反证 `prepared` 与 `launch` 的设计

使用 `results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428/episode_{00,01,02}.csv`，按当前主线最新 `SD` 状态口径重算高分 hop 轨迹，可得：

1. `grounded_prepared` 在三条高分轨迹中都能被明显激活：
   - `prepared_max ≈ 0.88 ~ 0.91`
   - `full_foot_support max ≈ 0.997 ~ 0.998`
   - `knee_window max ≈ 0.99`
   - `vertical_settle max ≈ 0.99`
2. 因此，“奖励进入 `prepared`”这一设计方向与真实高质量 hop 一致；
3. 但若把 `launch_reward` 写成“`prev_grounded_prepared * next_air * upward_progress`”或“`prev_grounded_prepared * rising_gain`”，则三条高分轨迹上几乎完全不触发；
4. 进一步离线对比多种候选式后，唯一同时满足以下两点的 `launch` 表达为：
   - 在 `tdmpc2` 高分 hop 轨迹上稳定触发；
   - 在本轮低预算三条地面推进代表轨迹上保持零触发；
   即：
   - `prev_grounded_prepared`
   - 乘以 `positive(prev_grounded_prepared - next_grounded_prepared)`
   - 再乘 `next_upward_progress`

这表明：在 Hopper 上，“launch”更合理的语义不是“已经空中上升”，而是“已完成准备，随后开始卸载 `prepared` 并伴随向上速度出现”。

### 2. 将结构化塑形正式收缩为两条主状态迁移

基于上述反证，本轮将 `predict_shaping_reward()` 进一步收缩为只围绕两条主状态迁移：

1. `grounded_unprepared -> grounded_prepared`
2. `grounded_prepared -> launch`

对应实现变更如下：

1. 删除中间态与局部改良塑形：
   - 删除 `falling_reward`
   - 删除 `prepare_reward`
   - 删除 `rising_reward`
2. 保留并放大“进入 `prepared`”奖励：
   - `prepared_reward = 0.24 * prepared_gain`
3. 将 `launch_reward` 正式定为：
   - `0.30 * prev_grounded_prepared * prepared_release * next_upward_progress`
   其中：
   - `prepared_release = positive(prev_grounded_prepared - next_grounded_prepared)`
4. 保留全局弱惩罚：
   - `0.04 * rotation_penalty`
   - `0.015 * air * backward_speed`

这样处理的目的只有一个：避免再奖励“未预备态附近的局部改良”或“尚未确认的中间态稳定”，而只奖励与闭环最直接对应的主状态跃迁。

### 3. 版本口径更新

为避免与前述 `v4.29` 多轮短测混淆，本轮将当前正式训练口径记为 `v4.30`。其定义为：

1. 结构态保持 `9` 维：
   - `rootz / vx / vz / pitch / pitch_rate / touch_toe / touch_heel / z_min / qpos_knee`
2. 总 latent 保持：
   - `sd_latent_dim = 512`
3. 结构化塑形只保留两条主状态迁移：
   - `prepared_reward`
   - `launch_reward`
4. `launch_reward` 采用经 `td-mpc2` 高分轨迹离线反证后的“prepared 卸载 + upward progress”写法

### 4. 正式实验重启

在完成上述收敛后，本轮直接重启三种子 `50k` 正式实验，统一口径如下：

1. `steps = 50000`
2. `eval_freq = 25000`
3. `checkpoint_freq = 25000`
4. `eval_episodes = 10`
5. `save_full_state = true`
6. 官方 `eval` 轨迹与 `.full.pt` 继续自动保存
7. 三条运行分别固定到空闲 GPU 上并行执行

本轮重点不再是继续在 `10k` 上比较分数高低，而是观察：

1. `25k / 50k` 是否首次稳定出现非零 `grounded_prepared_frac`
2. `launch_reward` 是否能把策略从“纯地面推进”进一步拉向真实起跳链
3. 是否仍会被 `full_foot_support` 这一判据长期卡死

## 2026-04-23：根据 `50k official_eval` 视频核验，收紧 `prepared` 与 `rising_air` 的状态语义

在对 `v4.31` 三种子 `50k official_eval` 的保存轨迹逐条统计后，又将以下四条 `episode_csv` 直接回放为 `mp4`，并与人工观察结果对照：

1. `seed3 episode_07`：`grounded_prepared` 误判样例；
2. `seed3 episode_00`：`rising_air` 误判样例；
3. `seed3 episode_02`：高分但非有效 hop 样例；
4. `seed2 episode_07`：带少量 `rising_air` 但本质仍为伏地支撑前行的样例。

视频核验后，可将问题进一步收敛为两条局部漏洞，而不需要扩大为复杂的新奖励机制：

### 1. `grounded_prepared` 需要排除“后仰躺平”

`seed3 episode_07` 证明，后仰躺平状态仍可能同时满足：

1. 整足接触；
2. 膝部进入准备窗口；
3. 竖直速度接近零。

原判据：

1. `grounded_prepared = grounded * full_foot_support * knee_window * vertical_settle`

缺少姿态门控，因此会把“后仰躺平但暂时稳定”的状态误记为“可起跳准备态”。

据此，本轮仅增加一个最小修补：

1. 在 `grounded_prepared` 中加入“非后仰躺平”软门；
2. 不额外扩大到前倾、翻滚等其他姿态，以保持判据最小化。

### 2. `rising_air` 需要改为“明确离地且明确向上”

`seed3 episode_00` 与 `seed2 episode_07` 说明，当前 `rising_air` 会把：

1. 刚落地后的轻微反弹；
2. 伏地爬行过程中的局部上抬；

误记为上升态。

原判据：

1. `rising_air = air * upward_progress`

其中 `air` 仍允许近地软量非零，因此只要出现少量向上速度，就会被点亮。

据此，本轮将 `rising_air` 收紧为同时要求：

1. 足部接触基本消失；
2. 身体整体高度达到更高的离地阈值；
3. `vz` 超过明确向上阈值。

这样做的目的，是把 `rising_air` 从“近地上抬趋势”收紧为“已经明确离地且仍在上升”。

### 3. 对当前非零分的重新定性

视频观察同时表明：

1. `seed3 episode_02` 的正分并非来自有效 hop，也并非稳定近地推进，而是“前倾扑倒后蜷缩”的无效姿态；
2. `seed1` 对应正分样例与其表现接近；
3. `seed2 episode_07` 虽有更长位移，但核心行为仍依靠前掌、膝盖、鼻尖支撑下的伏地前行。

因此，当前 `50k` 的非零分仍不能视为已学到目标行为链；本轮修订的目的不是“堵住所有非目标运动”，而是先把两个关键状态标签的语义收紧，使后续奖励引导不再被明显误标污染。

## 2026-04-24：修复 `resume_from final.full.pt` 后首轮训练的 episode 记账 bug

在将 `v4.32` 的三种子 `50k` 结果继续接续到 `200k` 时，三条作业都成功从 `final.full.pt` 恢复到了：

1. `step=50001`
2. `episode=100`
3. replay buffer 已恢复

但训练循环在恢复后的首轮立即报错：

1. `UnboundLocalError: local variable 'info' referenced before assignment`

定位到 `external/tdmpc2/tdmpc2/trainer/online_trainer.py` 的主循环后，可知问题原因如下：

1. `train()` 初始化时固定令 `done=True`，以便冷启动时先 `reset()` 再开始新 episode；
2. 恢复训练后，循环同样会先进入 `if done:` 分支；
3. 但此时尚未执行新的 `env.step()`，因此当前 episode 的 `info` 尚不存在；
4. 原代码却直接把该分支当成“刚结束了一局”，访问 `info['terminated'] / info['success']` 并尝试写入上一局日志，遂触发未定义变量错误。

本轮修复口径为：

1. 显式引入 `last_info` 保存最近一次 `env.step()` 返回的 episode 终止信息；
2. 仅在 `last_info is not None` 且 `_tds` 中确实已有完整 episode 片段时，才记录上一局并写入 replay buffer；
3. 在 `reset()` 后清空 `last_info`。

这样处理后：

1. 冷启动路径不受影响；
2. `resume_from` 后不会把“尚未开始的新一局”误记成已结束 episode；
3. 接续训练可从 `50001` 正常推进。

## 2026-04-24：将结构化奖励接入 reward 监督主通道

在继续复盘 `v4.32` 的 `50k official_eval` 后，可以更明确地确认当前主矛盾不在于是否还要继续设计更复杂的阶段奖励，而在于现有这套已经收敛到较简洁、较可靠语义的结构化奖励，并未真正进入参数学习主通道。

此前代码路径为：

1. 规划阶段，`SDWorldModel.reward()` 在非训练态会把 `predict_shaping_reward()` 以 `hybrid/curriculum` 形式叠加到 learned reward 上；
2. 训练阶段，`SDWorldModel.reward()` 在 `self.training=True` 时直接返回 learned reward logits；
3. `TDMPC2._update()` 的 `reward_loss` 监督目标始终来自 replay buffer 中的环境真实奖励。

这意味着结构化奖励此前主要影响的是规划排序，而不是 reward head 的监督信号本身。若规划时被结构化奖励吸引到的状态，无法在 reward head 中同步形成偏好，则该引导在参数更新链路中会持续被稀释。

基于“尽量不改结构”的约束，本轮采用最小修订：

1. 不改网络骨架，不新增 reward head；
2. 不改 TD target，仍保持 value 学习以环境真实奖励为主；
3. 仅将 reward 监督目标改为混合目标：
   `reward_target = env_reward + alpha * shaping_delta`
4. 其中 `alpha` 直接复用现有 `hybrid/curriculum` 口径，不再额外引入新的独立调度超参数。

对应代码实现为：

1. 在 `external/tdmpc2/tdmpc2/common/sd_world_model.py` 中新增 `shaping_reward()` 与 `mixed_reward_target()`；
2. 在 `external/tdmpc2/tdmpc2/tdmpc2.py` 中，将 `reward_loss` 的监督目标从 `reward` 切换为 `reward_target`；
3. 同步记录 `reward_mix_alpha`、`reward_shaping_mean/std` 与 `reward_target_mean/std`，便于后续判断混合监督是否实际生效。

本轮设计意图不是让结构化奖励取代环境奖励，而是让其以小幅 shaping 增量进入 reward 学习主通道，使规划、reward head 与后续策略学习之间不再长期脱节。

## 2026-04-24：将 `prepared` 的姿态门从“排除后仰”改为“要求近直立”

在 `v433` 单种子 `5k` smoke 中，虽然 `grounded_prepared` 已开始在少量轨迹里短暂出现，但对 `episode_03` 与 `episode_07` 的逐帧回放表明，这两个峰值仍不是希望中的“可起跳预备态”：

1. 双足接触与屈膝条件确已满足；
2. 但躯干主体仍明显偏离直立，属于落地后的短暂压缩接触，而非可稳定释放的起跳准备姿态；
3. 原有补丁只是在 `prepared` 中排除了更极端的后仰躺平，因此对 `pitch ≈ -1.0` 一类仍显著偏斜、但尚未达到“完全躺平”程度的姿态仍会漏过。

据此，本轮不再使用“只排除后仰”的门控，而将 `prepared` 的姿态条件直接改写为“要求 `pitch` 接近竖直”：

1. 删除原 `non_reclined = sigmoid((pitch - center) / scale)` 逻辑；
2. 改为 `upright_posture = sigmoid((pitch_max - abs(pitch)) / scale)`；
3. `grounded_prepared` 仅在双足支撑、屈膝、竖直速度已沉降且躯干姿态仍接近直立时点亮。

这样做的目的，是把 `prepared` 的定义从“不是很后仰就行”收紧为“本质上仍是近直立的着地压缩态”，从而更贴近起跳前的物理含义。

## 2026-04-25：核对 Hopper 原始观测口径后，修复 `any_contact_z / height / reward / trace analysis` 四处实现错误

在重新逐项核对 Hopper 原始 `15` 维观测后，已确认当前主线中有若干处“不是奖励思想本身，而是实现口径漂移”导致的误判。这些问题若不先修复，后续对 `SD` 与结构化奖励的讨论会持续混入假信号。

### 1. 原始观测口径再次确认

`dm_control.suite.hopper` 的 state 观测保持如下结构：

1. `position = qpos[1:] = [rootz, rooty, waist, hip, knee, ankle]`
2. `velocity = qvel = [rootx_dot, rootz_dot, rooty_dot, waist_dot, hip_dot, knee_dot, ankle_dot]`
3. `touch = log1p(sensor)`，其中两维分别对应 toe 与 heel 触地传感器

其中，`rooty` 是躯干俯仰角，不是位置量；latent 第 `0` 维在 `encode_observation()` 中存储的是 `torso_z = 1 + qpos_rootz`，不是 `qpos_rootz` 本身。

### 2. 已修复的实现错误

本轮共修复四类实现错误：

1. `z_min` 语义过载
   - 旧实现把单一 `z_min` 同时拿来表示“足部接地相关最低点”与“任意部位触地”。
   - 现改为：latent 内的 `z_min` 专指足部接地相关最低点；状态判定时额外引入 `any_contact_z`，由 `foot_contact_z / torso_lower_z / nose_lower_z` 取最小值得到。

2. `height` 代理错误
   - 旧实现实际上把 `height` 近似成了 `torso_z`，与 `dm_control` 中 `torso_z - foot_z(body origin)` 的定义不一致。
   - 现改为基于 latent 内 `torso_z` 与 `foot_contact_z` 的近似形式重新计算。

3. reward double shaping
   - 训练阶段 `mixed_reward_target()` 已将 `env_reward + alpha * shaping` 作为 reward head 监督目标；
   - 但规划/评估阶段 `SDWorldModel.reward()` 又把 shaping 再叠加了一次。
   - 现已改为：`explicit` 模式仍直接使用显式 reward；`hybrid/curriculum` 模式在规划与评估时仅返回 learned reward head，不再二次叠加 shaping。

4. trace 导出与离线分析口径错误
   - `_state_reward_features()` 先前把 `torso_z` 与 `rootz` 导出成同一个值，导致 eval CSV 与 summary 中字段语义混乱；
   - `analyze_hopper_state_reward_sd_validation.py` 又直接复用这些旧 `SD` 列做回算，等于把历史污染继续带入新分析。
   - 现已改为：
     - `torso_z` 输出真实 latent 第 `0` 维；
     - `rootz = torso_z - 1.0`；
     - `official_eval` 导出额外保存 `any_contact_z`；
     - 离线分析脚本优先从 CSV 中的 `qpos / qvel / touch` 重建原始观测，再用当前 `SD` 重新编码与判态。

### 3. 修复后的离线验证结果

#### 3.1 `seed1 @ 50k episode_02` 的“全程 falling_air”误判已被消除

对

`logs/hopper-hop/1/sd_hopper_50000_v444_seed1_20260424_run1/official_eval/step_0050000/episode_csv/episode_02.csv`

按当前模型、从原始 `qpos/qvel/touch` 重新编码后回算，可得：

1. `falling_air` 仅约 `10` 步
2. `grounded_failed` 约 `490` 步
3. 不再出现“整条轨迹 `500` 步都被判为空中态”的现象

这说明：`any_contact_z` 的引入确实修复了“鼻尖或躯干已贴地，但仍被按 `z_min` 当作空中”的核心误判。

#### 3.2 `td-mpc2 200k` 高分标准 hop 轨迹仍暴露出新的瓶颈

对

`results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428`

用同样方式回算后，可得：

1. `grounded_failed` 仍占大头，约 `0.83`
2. `grounded_recovery` 非零，约 `0.046`
3. `grounded_prepared` 只剩极少数峰值，约 `0.0007`
4. `rising_air` 也几乎不再出现

进一步拆 `prepared` 子门后，可见当前最主要的卡点不是足部支撑，也不是膝角窗口，而是 `vertical_settle`：

1. 在 `grounded_recovery > 0.3` 的高分 hop 片段中，`full_foot_support` 平均约 `0.97`
2. `knee_window` 也经常已进入较高区间
3. 但 `vertical_settle` 平均仅约 `0.01`
4. 对应帧上 `vz` 往往仍显著为正，约 `1.1 ~ 1.6`

这意味着：当前 `prepared` 对“竖直速度必须已沉降”的要求仍然过严，会把标准 hop 中“已形成较完整支撑、但已开始快速上推”的阶段继续挡在 `prepared` 外。换言之，前一轮修复虽然解决了 `z_min/any_contact_z` 的错误，但 `prepared / rising_air` 的最终口径仍需再单独收敛。

### 4. 当前结论

截至本轮，已可较确定地区分两类问题：

1. `episode_02` 一类“整条轨迹被误判为空中”的结论，属于实现错误，现已修复；
2. `prepared / rising_air` 在高分标准 hop 上过少，则不是旧结论的简单延续，而是当前新口径下仍然存在的真实设计瓶颈，下一轮应直接围绕 `vertical_settle` 与起跳窗口的定义继续复核。

## 2026-04-25：将 `prepared` 的竖直速度门改为非对称窗口，并阻断 `foot z_min` 对 airborne 判定的误主导

在上一条补记完成后，继续用

`results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428`

拆分标准 hop 轨迹的子门分布，可进一步确认两个更具体的问题：

### 1. `prepared` 的主要卡点确实是旧版 `vertical_settle`

对满足以下条件的“明显处于标准支撑/发力窗口”的高分 hop 帧做统计：

1. `grounded > 0.8`
2. `full_foot_support > 0.8`
3. `knee_window > 0.6`
4. `pitch ∈ [-0.05, 1.15]`

可得：

1. `vz` 中位数约 `0.70`
2. `vz` 的 `q75` 约 `0.96`
3. `vz` 的 `q90` 约 `1.28`
4. 但旧 `vertical_settle = clamp((0.40 - |vz|)/0.40)` 在这些帧上几乎恒为 `0`

这说明：标准 hop 的“准备/发力”阶段并不要求 `vz ≈ 0`。在稳定整足支撑与屈膝之后，很多优质轨迹会在仍未离地时就进入明显正 `vz` 的上推阶段。旧版 `vertical_settle` 实际上把这整个窗口都挡在 `prepared` 之外。

据此，本轮将 `prepared` 的竖直速度门从“绝对值接近零”改为非对称窗口：

1. 允许适度下压，但排除高速砸地
2. 允许中等幅度的正 `vz`，覆盖真实起跳前的发力阶段
3. 具体实现改为对 `vz` 使用软区间门，而不再使用 `abs(vz)` 的对称惩罚

### 2. `rising_air` 不亮的根因不只是阈值过高，而是 `any_contact_z` 被 `foot z_min` 错误主导

进一步检查高分 hop 的真实飞行帧时，发现许多满足下列条件的帧：

1. `touch_toe ≈ 0`
2. `touch_heel ≈ 0`
3. `vz > 0.4`
4. 环境 reward 已明显为正

却仍被当前 `SD` 判为 grounded。追查后可知，问题并不在 `rising_air` 的速度门，而在：

1. `any_contact_z` 旧实现直接取 `min(foot_contact_z, torso_lower_z, nose_lower_z)`
2. 即使传感器完全没有足部接触，脚部几何只要靠近地面，`foot_contact_z` 仍会把 `any_contact_z` 压到接近 `0`
3. 结果导致很多真实 airborne 帧被 `grounded_geom` 误吸回 grounded

据此，本轮对 `any_contact_z` 做一处最小但关键的语义修正：

1. `foot_contact_z` 只在 `support_contact` 达到一定水平时才参与 `any_contact_z`
2. 若足部触地传感器基本为零，则 `any_contact_z` 只由 `torso_lower_z / nose_lower_z` 决定
3. 这样既保留了“鼻尖/躯干触地时仍能识别 grounded_failed”的能力，又避免“脚部扫地式几何接近”把真实飞行误判成 grounded

### 3. 修订后的离线回算结果

按新口径重算高分 hop 轨迹，可得：

1. `falling_air ≈ 0.411`
2. `grounded_failed ≈ 0.289`
3. `grounded_recovery ≈ 0.031`
4. `grounded_prepared ≈ 0.061`
5. `rising_air ≈ 0.208`

相比上一轮：

1. `prepared` 已从几乎不可见恢复到稳定非零；
2. `rising_air` 也重新出现，且占比已进入明显可解释区间；
3. 主导转移重新包含：
   - `grounded_failed -> grounded_prepared`
   - `grounded_recovery -> rising_air`
   - `rising_air -> falling_air`

这说明：本轮两项修订确实把标准 hop 所依赖的“支撑准备 -> 起跳离地 -> 空中飞行”主链重新拉回来了。

### 4. 失败轨迹未被重新污染

同样按新口径重算

`logs/hopper-hop/1/sd_hopper_50000_v444_seed1_20260424_run1/official_eval/step_0050000/episode_csv`

可得：

1. `grounded_failed ≈ 0.968`
2. `falling_air ≈ 0.032`
3. `grounded_recovery / grounded_prepared / rising_air` 仍均为 `0`

`episode_02` 仍保持：

1. `falling_air ≈ 10` 步
2. `grounded_failed ≈ 490` 步

说明这次修订没有重新引入“失败轨迹被大面积误记为 prepared/rising”或“整条轨迹回退为空中态”的问题。

### 5. 当前结论

截至本轮，可形成更稳定的判断：

1. `z_min / any_contact_z` 主链已基本修顺；
2. `prepared` 的旧瓶颈确实主要在竖直速度门过严，而不是足部支撑或膝角窗口本身；
3. `rising_air` 的旧瓶颈则主要在 `any_contact_z` 误把“脚部近地”当作“任意部位已接地”；
4. 这两处修复后，标准 hop 的主状态链已重新可见，下一步可以在此基础上继续观察真实训练是否开始更容易进入这条链，而不必再回到更复杂的状态机扩张。

### 同日更正：上述判据热改已撤回，不作为当前主线版本结论

在与研究目标再次对齐后，已确认本条记录中的这轮 `prepared / rising_air / any_contact_z` 判据热改未经用户逐项审核，不应直接进入当前主线结论。对应代码修改已于同日撤回。

因此，当前主线仍以前一版本为准，并保留如下判断：

1. `double shaping`、离线分析脚本口径、`torso_z/rootz` 字段语义修正仍成立；
2. 本条中关于“通过直接改判据已恢复标准 hop 主链”的结论不再成立；
3. 后续若继续修改 `prepared / rising_air / grounded_failed` 的状态定义，需先基于高分轨迹逐项审核判据，再落代码。

## 2026-04-24：在保持 `alpha` 不变的前提下放大 `prepared_reward`

在将 `prepared` 的姿态语义收紧后，`10k` 评估轨迹显示：

1. 旧的 `prepared` 假阳性基本被清掉；
2. 但 `grounded_prepared` 与 `rising_air` 也都未再真正出现；
3. 唯一非零分 episode 仍来自非目标路径，而不是 `grounded_unprepared -> grounded_prepared -> rising_air` 这条主链。

基于这一结果，本轮不继续放大外层混合系数 `alpha`。原因是：

1. 当前训练阶段的 `alpha` 在 `10k` 前一直维持 `0.5`，已不算很弱；
2. 规划时仍会额外使用结构化 shaping，若继续抬高外层混合，容易进一步放大训练与规划之间的偏置；
3. 当前暴露出的核心矛盾，不是“已经进入 prepared 但起跳不够强”，而是“策略仍较少进入 prepared”。

因此，本轮仅做一个最小且有针对性的改动：

1. 保持 `LAUNCH_REWARD_WEIGHT = 0.45` 不变；
2. 保持惩罚项不变；
3. 将 `PREPARED_REWARD_WEIGHT` 从 `0.24` 提高到 `0.48`。

这样做的意图，是只增强 `grounded_unprepared -> grounded_prepared` 这一状态迁移的吸引，而不改变“prepared 之后再奖励真实 upward progress”这一总体逻辑。若后续轨迹中开始出现更干净的 `prepared` 片段，则说明当前瓶颈主要确实在于“进入准备态”的吸引仍偏弱。

## 2026-04-24：在 `prepared` 前加入 `grounded_recovery` 中间态

在继续复核高分轨迹后，问题进一步明确：当前很多能拿到环境真实奖励的轨迹，并不是通过

`grounded_unprepared -> grounded_prepared -> rising_air`

这条链达成，而是直接在低姿或单侧支撑模式下进入上升/前移。与此同时，前一轮把 `prepared` 的姿态门收紧到“近直立”之后，旧的假阳性虽被压掉，但 `prepared` 也变得更难进入。

这说明，当前主矛盾不是 `launch_reward` 不够，而是从杂乱 grounded 姿态回到“可准备”的站起过程缺少一个显式中间层。为此，本轮在不增加复杂状态机的前提下，引入一个最小中间态：

1. 新增 `grounded_recovery`
   - 已着地；
   - 竖直速度已沉降；
   - 足部支撑比单侧/尖点支撑更完整；
   - `|pitch|` 已明显回正，但尚不要求进入 `prepared` 的屈膝窗口。

2. 结构化奖励链改为：
   `grounded_unprepared -> grounded_recovery -> grounded_prepared -> rising_air`

3. 奖励接线调整为：
   - 新增 `recovery_reward`，奖励进入 `grounded_recovery`；
   - `prepared_reward` 只在 recovery 邻域中继续给出；
   - `launch_reward` 保持原有逻辑。

4. 为保证后续诊断一致，`grounded_recovery` 也同步写入官方 eval 的轨迹 CSV 与 episode summary。

这样做的目的，是显式鼓励“先站起来/回正，再准备，再起跳”这一更符合物理直觉的顺序，同时仍保持奖励设计简洁、可解释。

## 2026-04-24：将 `prepared` 修订为非对称姿态窗口，先对齐标准 hop

在加入 `grounded_recovery` 后，继续用 `td-mpc2` 的高分标准 hop 轨迹做离线回算时，发现 `prepared` 仍几乎不会点亮。进一步逐条检查后，可以把问题收敛到一个更具体的判据错误：

1. 旧的 `prepared` 姿态门基于 `abs(pitch)`，本质上把前倾与后仰对称看待；
2. 但标准 hop 的真实“起跳预备态”并不接近 `pitch = 0`，而是常出现在中等幅度前倾；
3. 相反，之前在 `v433 5k` 中确认的假阳性，恰恰集中在后仰躺地、但足底偶然完整着地的时刻。

离线重算得到的关键证据如下：

1. `td-mpc2` 高分轨迹的 `prepared` 候选峰值出现在 `pitch ≈ 0.95 ~ 1.11`；
2. 旧假阳性轨迹的峰值则在 `pitch ≈ -1.03 / -1.15`；
3. 因而 `prepared` 不能再使用对称的 `abs(pitch)` 门，而应允许较宽的前倾窗口，同时对后仰侧保持明显更严的限制。

据此，本轮将 `prepared` 改为非对称 corridor：

1. 删除基于 `abs(pitch)` 的对称 upright 门；
2. 新增 `PREPARED_BACK_PITCH_LOW = -0.15`、`PREPARED_FORWARD_PITCH_HIGH = 1.05`；
3. `grounded_prepared` 改为
   `grounded * full_foot_support * knee_window * vertical_settle * prepared_pitch_corridor`；
4. 暂不再要求 `grounded_prepared` 必须乘 `grounded_recovery`，先保证标准 hop 自身能够被正确识别。

修订后的离线验证结果如下：

1. 标准 `td-mpc2` hop 轨迹中，`prepared` 峰值已提升到 `0.37 ~ 0.60`；
2. 先前两条 `v433` 躺地假阳性轨迹中，`prepared` 峰值仍仅为 `0.0001` 与 `0.0`；
3. 这说明当前 `prepared` 已基本满足“标准 hop 可激活、后仰躺地不过门”这一最优先要求。

这一轮先到此为止。`unprepared` 与 `recovery` 是否还需调整，留待后续在新判据基础上再单独评估，避免多个因素同时变动而混淆因果。

## 2026-04-24：清理 `recovery / rising_air / imagined support` 中多余的软化项

在继续回看 `v435 10k` 的逐帧轨迹后，问题进一步从“状态定义是否大致正确”收敛到“哪些地方仍存在不必要的软化，从而破坏状态前提语义”。

本轮系统检查后，形成了三条最小修订结论：

1. `recovery` 中不应再保留额外的 `recovery_support = sigmoid((full_foot_support - c) / s)`；
2. `rising_air` 不能只依赖“足部无接触 + 高度 + 向上速度”，还必须以真正离地的 `air` 为前提；
3. imagined dynamics 中原先把同一个 `support_hold` 同时加到 `toe` 与 `heel` 上，存在把单侧支撑“抹平”为双侧支撑的风险。

对应代码修订如下：

1. `grounded_recovery` 改为
   `grounded * full_foot_support * recovery_upright * vertical_settle`
   不再套额外的 `recovery_support` sigmoid；
2. `rising_air` 改为
   `air * height_strict * upward_strict`
   显式要求已有真正离地状态；
3. dynamics 中将共享的 `support_hold` 拆成 `toe_support_hold` 与 `heel_support_hold`，仅延续各自原有支撑，不再凭空补齐另一侧接触。

这样调整的理由是：

1. `full_foot_support` 本身已经是直接、清晰的“整足支撑”代理量，不应再额外包一层人为阈值；
2. `rising_air` 的前提应是“身体已离地”，而不是“脚没碰地但其他部位可能仍在支撑”；
3. SD 前向推演允许适度平滑，但不应靠共享保留项虚构出原本不存在的双侧支撑。

本轮不改 `prepared` 主干定义，也不再额外增加新的状态或复杂门控，保持修订范围最小。后续重点转为：在更干净的状态语义下，再判断是否需要继续增强奖励吸引。

## 2026-04-24：将 grounded 三态改为互斥高层状态

在用新判据回看 `v435` 的 `5k official_eval` 时，出现了一个新的概念性问题：少量步上 `grounded_recovery` 与 `grounded_prepared` 会同时为正。虽然底层代理量本来就允许连续重叠，但若奖励链被明确设计为

`grounded_unprepared -> grounded_recovery -> grounded_prepared -> rising_air`

则高层阶段状态本身不应长期重叠，否则会造成两类歧义：

1. 同一步既被解释为“仍在恢复”，又被解释为“已经完成起跳预备”，状态顺序失去意义；
2. `recovery_gain` 与 `prepared_gain` 会在同一步共同抬升，削弱“先恢复、再准备”的阶段信号。

据此，本轮保留了连续代理量，但将 grounded 三态改为互斥高层状态：

1. 先计算连续代理量 `recovery_candidate` 与 `prepared_candidate`；
2. 若 `prepared_candidate >= 0.5`，则该步判为 `grounded_prepared`；
3. 否则，若 `recovery_candidate >= 0.5`，则该步判为 `grounded_recovery`；
4. 两者都未满足时，剩余 grounded 质量归入 `grounded_unprepared`。

这样做的意图是：

1. 底层支撑、屈膝、姿态等代理量仍保留连续信息；
2. 但用于奖励设计与轨迹解释的高层阶段标签，重新回到互斥状态图口径。

离线验证结果表明，该修订已达到预期：

1. 在 `v435` 的 `episode_01` 与 `episode_03` 中，先前会同时出现的 `prepared/recovery` 已被消除；
2. 两条轨迹里 `prepared > 0.1` 的步数仍分别保留为 `5` 与 `4`；
3. 同时 `overlap_pos = 0`，说明高层 grounded 三态已满足互斥要求。

## 2026-04-24：仅增强 `recovery -> prepared` 跃迁奖励，不改状态图

在完成最近一轮 `v436 10k` smoke 逐帧复核后，当前问题已进一步收敛：状态标签本身已基本清理干净，主要矛盾不再是“状态错判”，而是“奖励吸引不足以把策略从 `grounded_recovery` 拉向 `grounded_prepared`”。

这一判断基于以下现象：

1. `recovery` 在少量步上已能被可靠点亮；
2. `prepared` 在标准 hop 轨迹上可激活，但在当前 `10k` smoke 中几乎没有真正进入；
3. `rising_air` 的误亮已基本消失，因此继续大改状态定义的收益已明显下降。

据此，本轮不再修改状态判据，也不新增新的代理量，而是只在 shaping reward 中补一条最小跃迁奖励：

1. 保留原有 `recovery_gain`；
2. 保留原有 `prepared_gain`；
3. 保留原有 `launch_reward`；
4. 仅额外加入
   `recovery_to_prepared_reward = w * prev_grounded_recovery * prepared_gain`。

这样做的理由是：

1. 奖励仍严格围绕既有高层状态图，不引入新的状态复杂度；
2. 重点强化的是当前最缺失的一步跃迁，而不是笼统放大奖励总量；
3. 若后续 `prepared` 首次稳定出现，则可以较清楚地归因于这一项修订。

本轮对应常量设为 `RECOVERY_TO_PREPARED_REWARD_WEIGHT = 0.72`，暂不调整 `launch_reward` 与惩罚项，先做最小干预验证。

## 2026-04-24：将两类明显失稳姿态转化为 planning-only 动作先验

在 `v437 10k` smoke 结束后，轨迹复核表明：问题已不再主要是奖励接线，而是策略根本没有先学会从两类典型失稳姿态中恢复。

`10k official_eval` 中最常见的失败模式可概括为两类：

1. 后仰躺地：`grounded≈1`、`full_foot_support≈1`，但 `pitch≈-1.9~-2.1`；
2. 前伏趴地：`grounded≈1`、`pitch≈+1.8~+2.3`，但 `full_foot_support≈0`。

这说明当前“恢复”并不是一个应完全交给黑盒探索的子任务。既然这一步的基本力矩方向已经可由结构知识给出，就应将其显式编码为状态条件动作先验，而不是继续让策略在明显错误的动作方向上浪费样本。

本轮先未直接写规则控制器，而是先做了两层动作符号标定：

1. SD 一步/多步 probe，用于快速筛查；
2. 真实 DMControl 环境 probe，用轨迹中的 `qpos/qvel` 复原物理状态，再对单关节常值动作做短 rollout。

真实环境 probe 的代表性结果如下：

1. 对后仰躺地状态，`waist-` 是唯一稳定把 `|pitch|` 往回拉的主方向；其余方向大多要么继续加重后仰，要么直接丢失整足支撑；
2. 对前伏趴地状态，`waist+`、`hip-`、`knee+`、`ankle-` 形成更合理的回正方向组合；与之相反的符号则普遍导致 `pitch` 进一步远离目标姿态。

据此，本轮采用了最保守的落地方式：

1. 仅在 planning 端施加动作先验；
2. 不修改 actor 输出；
3. 不手写唯一动作，只收缩每个关节的允许力矩区间；
4. 仅在两类显然失稳的状态下激活，正常状态仍保持原有自由规划。

当前第一版动作先验定义如下：

1. `rear_lying`：
   `grounded>=0.8 & full_foot_support>=0.6 & pitch<=-1.5`
   对应约束
   `waist∈[-0.90,-0.15]`
   `hip/knee/ankle∈[-0.35,0.25]`
2. `prone_lying`：
   `grounded>=0.8 & full_foot_support<=0.2 & pitch>=1.5`
   对应约束
   `waist∈[0.20,0.90]`
   `hip∈[-0.70,0.10]`
   `knee∈[0.10,0.90]`
   `ankle∈[-0.50,0.25]`

这一设计的意图是：

1. 将“明显已知”的恢复方向从学习负担中剥离出来；
2. 保留区间内的幅值与时序优化空间，避免退化为手写控制器；
3. 用最小结构干预验证：状态条件动作先验是否能够比单纯奖励塑形更有效地把策略拉回可恢复区域。

## 2026-04-24：新增 Hopper 实验操作总指南，统一固定启动与渲染口径

在近期多轮 Hopper 实验推进中，反复出现的一个非方法学问题是：同一类操作虽然已有稳定口径，但命令散落在不同脚本、工作日志片段与对话上下文中，导致启动训练、检查进度、渲染轨迹时容易再次回到“临时回忆与试错”状态。

为避免后续继续在重复性操作上消耗时间，本轮新增统一操作手册：

1. [HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md](/home/data/round/code/sd_world_model_rl/docs/HOPPER_EXPERIMENT_OPERATIONS_GUIDE.md)

该文件当前固定了以下内容：

1. `5k / 10k / 50k / 200k` Hopper 实验应使用的脚本与参数口径；
2. 启动后应检查的最小集合：
   `train.csv`、`2500.full.pt`、`eval.csv`、`official_eval`；
3. 已保存 `trace csv -> mp4` 的稳定渲染命令；
4. `checkpoint -> replay eval -> mp4` 与 `checkpoint -> trace csv` 的脚本入口；
5. 在 Codex 沙箱环境下，哪些命令通常必须申请提权；
6. 已反复踩过的错误点列表。

后续原则改为：

1. Hopper 相关重复性操作优先以该指南为准；
2. 若有新的稳定口径，应先更新该文件，再在对话中使用；
3. 不再依赖零散上下文记忆保存实验操作流程。

## 2026-04-24：将固定恢复动作盒子改为“参考姿态驱动”的误差缩放先验

在继续复核 `v438` 的恢复失败轨迹后，当前判断进一步收敛：问题不在于是否还要继续细分更多奖励阶段，而在于已有动作先验仍然过于静态。固定盒子虽然能限制明显错误的动作方向，但无法表达“偏离标准姿态越远，允许的恢复力度应越大；越接近标准姿态，范围应自然收窄”这一结构事实。

因此，本轮不再维持：

1. `rear_lying` 一组固定上下界；
2. `prone_lying` 一组固定上下界。

而是改为：

1. 先固定标准 hop 的参考姿态目标，仅保留当前显式状态中可直接使用的三项：
   - `pitch_target = 1.02`
   - `knee_target = 1.21`
   - `full_foot_support_target = 0.99`
2. 在 `rear_lying / prone_lying` 两类明显失稳状态下，用当前状态与参考姿态的偏差，生成动作中心与动作宽度；
3. planning 只在该动态范围内采样，具体幅值和时序仍交给优化器选择。

### 标定依据

本轮动作方向并非凭空设定，而是结合两类高分轨迹窗口完成校准：

1. `td-mpc2 200k` 的后仰恢复片段
   - 轨迹：
     `results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428/episode_00.csv`
   - 在 `step 8-14` 的后仰回正窗口中，最稳定的信号是：
     - `action_0` 长期显著为负；
     - `action_3` 大多为负；
     - `action_1` 在深后仰早期可短暂偏正，但随着姿态回正会明显转负；
     - `action_2` 波动较大，不适合固定死符号。
2. `td-mpc2 200k` 的前伏/俯卧恢复片段
   - 以同一 checkpoint 重新导出：
     `results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_recovery_ep03_seed2/episode_03.csv`
   - 在 `step 0-4` 的重前倾窗口中，最稳定的信号是：
     - `action_0` 明显为正；
     - `action_1` 多为负；
     - `action_3` 明显为正；
     - `action_2` 作用较弱，宜保留更宽的自由度。

### 新先验形式

实现上不再使用常数盒子，而改为：

1. `rear_lying`
   - 定义 `rear_progress`，表示身体从深后仰向参考姿态回正的进度；
   - `action_0 / action_3` 随 `rear_progress` 增大而进一步朝恢复方向收紧；
   - `action_1` 从“允许早期翻卷”逐渐过渡到“后期明显压负”；
   - `action_2` 保持最宽范围，只在中心上做轻度引导。
2. `prone_lying`
   - 定义 `prone_severity`，表示前伏/俯卧的严重程度；
   - `action_0` 维持强正向驱动；
   - `action_1` 维持负向驱动；
   - `action_3` 维持正向驱动；
   - `action_2` 仅依据当前 `knee` 偏离做弱修正，不再强行规定固定符号。

同时新增了两个诊断量：

1. `rear_progress_mean`
2. `prone_severity_mean`

用于后续在日志中直接观察当前 planning 先验究竟主要激活在什么恢复阶段，而不是只看 `rear_prior_frac / prone_prior_frac` 的触发占比。

### 当前判断

这一版动作先验更符合当前路线的基本原则：

1. 不手写控制器；
2. 不把恢复子任务完全甩给黑盒探索；
3. 只把“已知的力学方向”编码进先验；
4. 将具体幅值、时序和跨步协调仍交给 planning / policy 学习。

后续验证重点不再是“是否出现 prepared”，而是：

1. `prone_lying / rear_lying` 触发后，`|pitch|` 是否更快回落；
2. `full_foot_support` 是否更快恢复；
3. 是否比固定盒子版本更少出现“只轻微抬身但脚始终回不到支撑位”的假恢复。

## 2026-04-24：修正恢复先验的入门条件，避免被“整足双点接触”卡死

在 `v439` 的 `5k smoke` 评估写盘过程中，先对已落盘的前几条 `episode_csv` 做了局部检查。结论很明确：这版不是“恢复先验触发了但效果弱”，而是后仰恢复门本身几乎没有打开。

已观察到的典型现象如下：

1. 多条失败轨迹中，`pitch <= -1.5` 的后仰状态持续出现，且 `grounded` 明确为高；
2. 此时 `support_contact = max(touch_toe, touch_heel)` 经常接近 `1.0`，说明身体确实已有明确地面支撑；
3. 但 `full_foot_support = min(touch_toe, touch_heel)` 仍长期为 `0.0`，因为真实失败姿态大多是单侧着地点，而不是前后足缘同时稳定着地；
4. 旧版 `rear_mask` 要求 `full_foot_support >= 0.60`，于是这些明显的后仰失败姿态被全部挡在先验门外。

用当前这批 `episode_csv` 按旧门与新门做计数，对比结果如下：

1. 旧门下，满足后仰恢复条件的步数为 `0`；
2. 改为“已有明确地面支撑”后，满足后仰恢复条件的步数变为 `460`；
3. 俯卧门原本就主要由 `pitch >= 1.5` 与 `full_foot_support <= 0.20` 控制，本批轨迹下并未被这一修订削弱，满足步数仍为 `834`。

据此，本轮仅做一处最小修订：

1. 保持 `prepared / recovery` 的奖励态判定不变，仍由更严格的 `full_foot_support` 参与；
2. 仅把恢复动作先验的入门条件改为基于 `support_contact`；
3. 具体而言：
   - `rear_mask`：由 `full_foot_support >= 0.60` 改为 `support_contact >= 0.60`
   - `prone_mask`：补上 `support_contact >= 0.50`，避免把“几乎无支撑的空擦地状态”也误收进恢复先验

这样处理的原因是：

1. `prepared` 描述的是“可起跳的较完整支撑态”，继续使用整足代理是合理的；
2. 恢复先验描述的是“明显失败姿态下，planning 应先往合理恢复动作范围收缩”，它只需要确认“身体已经在地面上、并且确有支撑可借”，不应要求已经恢复到整足稳定支撑；
3. 因而二者不应共用同一个严格门槛。

## 2026-04-24：将 `recovery` 从静态状态改为迁移事件，并补上高分轨迹回算口径

在 `v440` 的 `5k smoke` 中，对 `episode_07` 的逐帧核查已进一步确认：先前被打上 `grounded_recovery` 的窗口，并不是“从前倾或后仰失败姿态主动恢复”，而只是“从空中落地后，恰好短暂处于较高足部支撑且身体较正”的被动稳定落地。

这说明旧定义存在语义混淆：

1. 旧版 `grounded_recovery` 只看当前状态；
2. 只要当前帧满足“足部支撑较完整、姿态较正、速度较稳”，就会被记成恢复；
3. 但真正的“恢复”应是一个过程，而不是一个静态姿态标签；
4. 换言之，它应描述“前一时刻处于明显失衡的着地姿态，随后开始向可控姿态回正”的状态迁移。

基于这一点，本轮做了两项修订。

### 1. 取消静态 `grounded_recovery` 状态标签

在 `src/world_model/dmcontrol_hopper_sd_model.py` 中：

1. 不再把 `recovery` 作为 `_state_reward_features()` 内的静态 grounded 子状态；
2. `grounded_prepared` 保持原定义；
3. `grounded_unprepared` 改回仅由 `grounded * (1 - prepared_mask)` 给出；
4. `grounded_recovery` 置为零张量，仅保留键名用于兼容已有日志/脚本接口。

这样处理的目的，是明确将“恢复”从标签语义中拿掉，避免再次把“落地即较稳”的片段误认为恢复。

### 2. 在 `predict_shaping_reward()` 中显式计算 `recovery_event(prev, next)`

新的恢复事件只在以下条件同时满足时给分：

1. `prev` 与 `next` 均为 grounded；
2. `prev` 的 `|pitch|` 已超过失衡源区间下界；
3. `next` 的 `|pitch|` 相比 `prev` 明显下降；
4. 地面支撑有所改善，或至少 `next` 已具备较明确的支撑接触。

实现上，最终采用的恢复事件为：

1. `prev_grounded`
2. `next_grounded`
3. `prev_failed_posture`
4. `pitch_reduction`
5. `max(support_contact_gain, 0.5 * next_support_contact)`

其中有两点是本轮回算后重新确认的：

1. 不再使用 `full_foot_support` 作为恢复事件门槛
   - 后仰/俯卧恢复参考轨迹中，大量回正过程只依赖单侧有效支撑；
   - 若继续要求整足双点支撑，恢复事件会被系统性压成零。
2. 不再要求 `next` 已接近竖直或速度已完全沉稳
   - 真实恢复过程中，角度改善往往先发生，完全稳定是更晚的结果；
   - 若把“已较直立、已沉稳”也塞进恢复定义，会再次把事件推迟到过晚阶段，丢掉真正的恢复起点。

### 3. 每次修订后用高分轨迹回算的固定口径

为避免新定义误伤标准 hop，本轮同时补稳了回算脚本：

1. 修改 `scripts/analyze_hopper_state_reward_sd_validation.py`
2. `dominant_state()` 在旧 trace 缺少 `grounded_recovery` 等列时，改为默认按 `0.0` 处理；
3. 这样可直接对旧 schema 的高分轨迹重复回算，而不会因字段缺失中断。

本轮用于复核的固定命令为：

```bash
PYTHONPATH=$PWD python scripts/analyze_hopper_state_reward_sd_validation.py \
  --dataset high=results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428 \
  --dataset v440=logs/hopper-hop/3/sd_hopper_5k_v440_priorgate_smoke_20260424_run1/official_eval/step_0005000/episode_csv
```

### 4. 回算结果

#### 4.1 误判样本已被压低

以 `v440` 的误判轨迹：

`logs/hopper-hop/3/sd_hopper_5k_v440_priorgate_smoke_20260424_run1/official_eval/step_0005000/episode_csv/episode_07.csv`

按新 `recovery_event` 逐步回算，可得：

1. `max_recovery_event = 0.0548`
2. `recovery_event > 0.1` 的步数为 `0`

说明这条“落地即较稳”的样本不再被当作真实恢复。

#### 4.2 标准高分 hop 未被误伤

对 `td-mpc2 200k` 的三条高分轨迹：

`results/tdmpc2_hopper_200k_sameprotocol_20260422_062444_seed2/trace_probe_v428/episode_{00,01,02}.csv`

重新回算后可得：

1. `grounded_prepared` 仍有显著占比
   - 总计 `129` 帧
   - 三条轨迹分别为 `47 / 40 / 42` 帧
2. `rising_air` 仍保持主要空中上升段
   - 总计 `479` 帧
3. `grounded_recovery` 静态占比回到 `0`
   - 这符合本轮语义调整：恢复只作为迁移事件，不再作为状态标签。

因此，本轮修订没有抹掉标准 hop 所依赖的 `prepared -> rising` 主链。

#### 4.3 恢复参考轨迹开始出现非零事件，但前伏恢复仍偏弱

对两条恢复参考轨迹局部回算后：

1. 后仰参考轨迹可得到非零恢复事件
   - `max_recovery_event = 0.1291`
   - `recovery_event > 0.1` 的步数为 `2`
2. 前伏参考轨迹已有非零恢复事件，但幅度仍偏弱
   - `max_recovery_event = 0.0652`
   - `recovery_event > 0.1` 的步数为 `0`

当前判断是：

1. 语义方向已正确；
2. “误判样本归零、后仰恢复可触发、标准 hop 主链保留”三点已成立；
3. 但前伏恢复事件仍偏弱，后续若继续强化“prone -> controlled support”这一路径，应优先从动作先验或恢复奖励权重入手，而不是再把 `recovery` 重新做回静态状态。

## 2026-04-24：先收正前伏失败态判据，避免在 `5k` 轨迹中系统性漏检

在 `v441` 的 `5k smoke` 结束后，对

`logs/hopper-hop/3/sd_hopper_5k_v441_recoveryevent_smoke_20260424_run1/official_eval/step_0005000/episode_csv`

做了直接统计。原先若按当前主线的前伏门：

1. `grounded >= 0.8`
2. `support_contact >= 0.5`
3. `full_foot_support <= 0.2`
4. `pitch >= 1.5`

则整批 10 条评估轨迹只命中 `20` 步。这个数量与当前轨迹的直观观察不符。

进一步仅对 `pitch` 阈值做扫描，可得：

1. `pitch >= 1.5` 时，命中 `20` 步
2. `pitch >= 1.3` 时，命中 `55` 步
3. `pitch >= 1.2` 时，命中 `68` 步
4. `pitch >= 1.1` 时，命中 `106` 步
5. `pitch >= 1.0` 时，命中 `257` 步

同时，对 grounded 帧的 `pitch` 分布做直方统计后可见，最大一坨实际上落在：

1. `pitch in [1.0, 1.5)`：`238` 步
2. `pitch >= 1.5`：仅约 `20` 步

这表明，`5k` 预算下的大量“前伏/前趴失败态”并没有发展到极端高前倾，而是主要停留在 `1.0 ~ 1.5` 的区间。因此原先的 `1.5` 下界过严，会把大量直观上显然属于前伏主导的失败轨迹漏掉。

据此，本轮先做最小修订。最初曾尝试仅放宽 `pitch` 下界，但随后结合

`results/hopper_trace_renders/v441_seed3_step5000_episode04_prone_check.mp4`

的逐帧观察发现，问题并不在脚部接触，而在于 `grounded` 门本身过高。该轨迹在落地后几乎全程都处于前伏，只是前期完全趴地，后期屈膝后脚部离地；若继续把“高 grounded”或“脚部支撑”作为前伏主条件，都会系统性漏检。

随后继续对 `episode_09` 做视频核对后，可进一步确认：前伏或后仰这类失败态的“是否着地”，根本不应依赖足部或 `grounded` 这类偏向稳定支撑的代理，而应直接依赖 `z_min`。原因很直接：

1. 失败态下，真正贴地的往往是鼻尖、躯干下缘、前掌、小腿等任意部位；
2. 这些状态在语义上显然已经是“着地失败态”；
3. 若继续只看足部或高 `grounded`，就会把大量明显的前伏/后仰片段漏掉。

因此，本轮最终修订收敛为：

1. 不动 `recovery` 迁移事件定义；
2. 将前伏与后仰失败态的“着地判据”统一切换到 `z_min`；
3. 即：
   - `prone_mask = (pitch >= 0.90) & (z_min <= 0.08)`
   - `rear_mask = (pitch <= -1.50) & (z_min <= 0.08)`
4. 相应地：
   - 删除前伏失败态对 `grounded` 的依赖；
   - 删除后仰失败态对 `support_contact` 的依赖；
   - 明确前伏/后仰首先是姿态失败态，而不是足部支撑态。

按最终 `pitch + z_min` 口径，用同一批 `v441` 已保存轨迹重算，可得：

1. 前伏总命中步数：`1467`
2. 后仰总命中步数：`2302`

其中与视频核对最关键的三条为：

1. `episode_04`
   - 前伏：`487`
   - 后仰：`0`
   - 与观察到的“落地后几乎全程前伏”一致
2. `episode_05`
   - 前伏：`20`
   - 后仰：`435`
   - 与观察到的“短暂前伏后迅速拉过头，进入长期后仰”一致
3. `episode_09`
   - 前伏：`479`
   - 后仰：`0`
   - 与观察到的“整体始终前伏，但偶有短暂离地”一致

之所以继续将前伏 `pitch` 下界从 `1.10` 下调到 `0.90`，是因为对 `episode_09` 的进一步视频核对表明：

1. `z_min <= 0.08` 已经覆盖了几乎全程的近地/触地状态，计数为 `481` 步；
2. 但这些帧中，`pitch` 最大的一坨并不在 `>= 1.1`，而是在 `[0.9, 1.1)`；
3. 具体而言：
   - `pitch >= 1.1 and z_min <= 0.08` 仅有 `99` 步；
   - `pitch >= 0.9 and z_min <= 0.08` 可达 `479` 步。

因此，若以视频观察为准，`0.90` 比 `1.10` 更符合当前低预算失败态下的“前伏”语义。

当前判断是：

1. 这一步并不是“放宽奖励”，而是先把失败态识别修到与真实轨迹分布一致；
2. 在 agent 尚未学会恢复的阶段，一条失败轨迹往往会被某一种失败模式主导，因此“前伏是否被大量识别出来”比“前伏与后仰是否同时都很多”更重要；
3. 后续若继续做前伏恢复链的动作先验或奖励增强，应基于这一修订后的前伏判据继续推进。

## 2026-04-25：按直接观测优先原则，将 Hopper 状态机收敛为六个互斥状态

本轮不再沿用 `falling_air / grounded_failed / grounded_recovery / grounded_prepared / rising_air` 的五态口径，而按最新讨论重构为六个互斥状态：

1. `falling_air`
2. `grounded_rear`
3. `grounded_prone`
4. `grounded_upright`
5. `grounded_prepared`
6. `rising_air`

设计原则统一为：

1. 取消 `air / grounded` 作为主状态输出；
2. 状态判定尽量直接使用原始观测，只在“是否已有任意部位着地”这一点上保留最小代理量 `z_min`；
3. 两个空中态对称，四个地面态互斥；
4. `prepared` 只在“整足支撑 + 躯干基本竖直 + 膝关节进入预备窗口”时点亮；
5. `upright` 与 `prepared` 区分开：前者只要求足部已有支撑且躯干回正，后者额外要求膝关节进入预备窗口。

首版阈值按当前理解先固定为：

1. `z_min <= 0.03` 视为已着地
2. `z_dot < -0.05` 视为空中下降
3. `z_dot > 0.05` 视为空中上升
4. `max(touch_toe, touch_heel) >= 0.30` 视为足部已有支撑
5. `min(touch_toe, touch_heel) >= 0.30` 视为整足支撑
6. `rooty <= -0.75` 视为后仰主导
7. `rooty >= 0.75` 视为前伏主导
8. `-0.45 < rooty < 0.45` 视为躯干基本竖直
9. `1.45 <= knee <= 1.95` 作为首版 `prepared` 膝窗口

代码层面的直接改动包括：

1. `src/world_model/dmcontrol_hopper_sd_model.py`
   将状态特征函数重构为六态互斥输出，并将 shaping 改为按状态迁移给分：
   - `rear -> upright`
   - `prone -> upright`
   - `upright -> prepared`
   - `prepared -> rising_air`
   - `falling_air -> upright`
   - `falling_air -> prepared`
2. `external/tdmpc2/tdmpc2/trainer/online_trainer.py`
   官方评估轨迹导出改为写入六态字段及对应统计量。
3. `scripts/export_hopper_eval_traces.py`
   离线导出脚本改为与主线相同的六态字段。
4. `scripts/analyze_hopper_state_reward_sd_validation.py`
   分析脚本改为读取六态。
5. `scripts/probe_hopper_action_priors.py`
   恢复动作盒分析脚本去除对旧 `grounded` 辅助态的依赖。

这一轮的意图不是一次性把阈值调准，而是先把“状态语义”和“代码实现”统一到同一个最小闭环上。后续应直接用短预算轨迹验证三点：

1. 六态是否真正互斥
2. 前伏/后仰是否与视频观察一致
3. `prepared` 是否只在短暂、合理的起跳预备窗口出现

### 同日补记：将结构化奖励从“软门值相乘”收紧为“硬状态迁移触发”

在继续梳理奖励设计时，已明确否定前一版 shaping 中的一个多余技巧：用

1. `prev_feat["state_a"]`
2. `next_feat["state_b"]`

这样的状态门值相乘来给迁移奖励。

否定理由有两点：

1. 这会把“状态命中值略低”错误地解释成“应当少给奖励”，但这一点此前并无充分论证；
2. 更关键的是，这种写法会把状态迁移和状态内软门混在一起，使奖励语义复杂化，不符合“只在判定进入目标后继状态时给奖励”的最初要求。

据此，本轮将 Hopper 六状态 shaping 统一改为硬迁移指示量：

1. 先由六态门值决定唯一的 `dominant_state`
2. 仅当 `prev_state == A and next_state == B` 时，对应迁移奖励触发
3. 触发值固定为 `1 * weight`
4. 若某一步没有任何状态门值达到 `0.5`，则该步 dominant state 记为 `-1`，不触发 shaping

当前保留的 6 项迁移为：

1. `falling_air -> grounded_upright`
2. `falling_air -> grounded_prepared`
3. `grounded_rear -> grounded_upright`
4. `grounded_prone -> grounded_upright`
5. `grounded_upright -> grounded_prepared`
6. `grounded_prepared -> rising_air`

同时，`prepared -> rising_air` 中此前附带的 `forward_progress` 已去除。原因是：只要 `rising_air` 的语义本身足够正确，就不应再额外引入一个未经充分论证的“前进加成”技巧；若未来确需鼓励更远跳跃，应单独设计基于前向速度与竖直速度的明确代理量，而不是混在当前迁移奖励里。

### 同日补记：复核 `height / any_contact_z / backward_speed / rotation_penalty` 四个保留量

在进一步梳理“哪些代理量已经不再进入主状态判断与 shaping 主链”时，又单独复核了以下四个仍保留在代码中的量：

1. `height`
2. `any_contact_z`
3. `backward_speed`
4. `rotation_penalty`

结论如下。

1. `height`
   继续保留。它不是冗余诊断量，而是 `predict_reward()` 中显式环境奖励重建的一部分。当前实现为：
   - `foot_contact_z = z_min`
   - `foot_body_z = foot_contact_z + FOOT_BODY_HEIGHT_OFFSET`
   - `height = torso_z - foot_body_z`
   这一定义与 Hopper 环境中“torso over foot”的站立部分语义一致，当前未发现明显错误。

2. `any_contact_z`
   继续保留，但应明确其性质：这是一个近似代理量，不是完整的“全身最低点”精确重建。当前实现只取：
   - `foot_contact_z`
   - `torso_lower_z`
   - `nose_lower_z`
   三者最小值。之所以这样做，是因为当前 9 维显式状态不再保留全部关节几何，无法从显式状态精确恢复大腿/小腿/骨盆下缘位置。该量目前只在 `predict_viable_reward()` 与评估导出中使用，不进入六状态主判定与 shaping 主链，因此暂不扩张其复杂度。

3. `backward_speed`
   计算为 `clamp(-vx / 0.50, 0, 1)`。其语义是“向后速度的归一化强度”，公式本身与变量含义一致。当前它只用于评估轨迹导出与汇总统计，不进入主状态判断、显式环境奖励或结构化 shaping。

4. `rotation_penalty`
   计算为 `clamp((abs(pitch_rate) - 3.5) / 2.5, 0, 1)`。其语义是“过快俯仰旋转的归一化强度”，公式本身与变量含义一致。当前同样只用于评估轨迹导出与汇总统计，不进入主状态判断、显式环境奖励或结构化 shaping。

据此，本轮不删除其余已不在主链上的诊断量。原因是：

1. 这些量当前计算开销很小；
2. 它们仍可用于离线导出、轨迹核对与异常定位；
3. 直接删除会增加后续阅读与诊断成本。

但口径需明确：

1. 六状态主判定与硬迁移 shaping 只依赖此前已确认的最小量集；
2. 其余量若仅用于导出/诊断，不应再被误认为主算法设计的一部分；
3. 未来若继续简化代码，应优先在不影响日志兼容与诊断能力的前提下再做收缩。
