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

## 2026-04-18 v4 立项：SD 结构化诊断与非对称接触建模

### 背景与动机

修复 bug + 去 submodule + 重跑 50k × 3（`sd_hopper_50k_v3b_hh005_fixrerun_20260418`）后，均值 **23.0 ± 12.6**（seed 1=5.64 / seed 2=27.99 / seed 3=35.39）。相比 v3b 32.93 下降约 10 分，但仍显著优于 TD-MPC2 100k (13.6 ± 10.4) 和 DreamerV3 100k (9.7 ± 13.6) —— **一半预算打赢两倍预算的 baseline**。

然而 50k × 10 eval 中 seed 3 仍有 6 条 0-reward 轨迹；对 v3b checkpoint 的 10 episode 渲染观察到：

1. 零回报轨迹几乎都终止于"骨盆/躯干贴地、躯干后仰 30-45°、一腿前屈"的**坐死盆地**
2. 高回报轨迹并非跳跃，而是**"前躯支撑 + 脚部配合"的低姿前向爬行**
3. planner 的 viability 门用 `cos(pitch)`，**前后对称**，不能区分"前倾可恢复"与"后仰死锁"

三位外部专家（结构独立咨询）的意见一致：这是 SD 建模的一阶问题，应纳入结构化求解过程，方向是"更少但更准"的结构量。

### Phase 0：状态日志硬数据诊断

**脚本**：`scripts/phase0_state_log.py`（v4 worktree 内）  
**Checkpoint**：`logs/hopper-hop/3/sd_hopper_50k_v3b_hh005/models/final.pt`（v3b）  
**输出**：`results/phase0_state_log/episode_{00..09}.csv` + `summary.csv`

#### Summary（来自 `summary.csv`）

| 组 | episodes | 成败 | final_torso_z 范围 | final_pitch 范围 | max_rootx 范围 | backward_pitch_frac |
|---|---|---|---|---|---|---|
| 成功（R>88）| 1, 5, 9 | 94.8 / 99.0 / 101.6 | 0.69–0.88 | +0.04 ~ +0.54 | 7.17–7.51 | **0% / 0% / 0%** |
| 失败（R≈0）| 0, 2, 3, 4, 6, 7, 8 | 0.007 / 0 / 0.001 / 0 / 0 / 0 / 0 | **0.07–0.20** | **-1.40 ~ -0.74**（除 ep6 旋转异常）| 0–0.38 | **78%–99%** |

**完全分离**：torso_z gap ≥ 0.49m；pitch gap ≥ 0.34 rad；rootx gap ≥ 6.8m。

#### 代表性轨迹快照（step 0 / 10 / 30 / 60 / 120 / 240 / 499）

```
ep0 (fail R=0.007):  
  step 0:   torso=1.07  pitch=+0.32  ← 初始前倾
  step 10:  torso=0.86  pitch=-0.55  ← 10 步内已倒向后
  step 30:  torso=0.23  pitch=-0.62  ← 骨盆接地
  step 60:  torso=0.09  pitch=-1.40  ← 坐死锁定
  step 499: torso=0.09  pitch=-1.40  ← 500 步无恢复

ep2 (fail R=0):
  step 0:   torso=0.93  pitch=-2.95  ← 初始几乎倒立
  step 30:  torso=0.29  pitch=-0.37  ← 短暂恢复
  step 60:  torso=0.37  pitch=+0.32  ← 前倾成功一次
  step 120: torso=0.11  pitch=-1.18  ← 再次坠入坐死
  step 499: torso=0.15  pitch=-0.97

ep5 (succ R=99):
  step 0:   torso=0.99  pitch=+1.76  ← 初始强前倾
  step 30:  torso=0.17  pitch=+2.48  ← 躯干低但 pitch 保持正
  step 60:  torso=0.18  pitch=+2.67  ← 爬行中
  step 240: torso=0.18  pitch=+2.68
  step 499: torso=0.88  pitch=+0.27  rootx=+7.50  ← 立起来前进 7.5m

ep9 (succ R=101):
  step 0:   torso=0.95  pitch=+2.27  ← 初始强前倾
  step 10:  torso=0.19  pitch=+1.75
  step 30:  torso=0.83  pitch=+0.22  ← 起跳恢复
  step 499: torso=0.80  pitch=+0.04  rootx=+7.17
```

#### 关键推论

1. **初始姿态符号不是单一预测器**：ep0 初始 pitch=+0.32（前倾）仍失败；ep2 初始 pitch=-2.95（几乎倒立）短暂恢复过。**决定成败的是恢复动力学**，不是初态。
2. **低 torso_z 不是失败信号**：成功 ep5 在 step 30-240 的 torso_z 长期保持 0.17-0.24（"爬行"姿态）；失败 ep0 也是 torso_z≈0.08。**低姿本身不可判定**。
3. **"低 torso_z" ∧ "负 pitch" 同时成立 → 不可逆**：所有失败 episode 的终态都是 `torso<0.25 ∧ pitch<-0.7`；所有成功 episode 的低姿段 `torso<0.25` 时 pitch 都保持 +1.7 ~ +2.7。
4. **v3b 学到的"成功"策略**：前倾爬行（pitch≈+2.5 低姿）→ 积累前向速度 → 最终立起来（torso→0.8, pitch→0.1）→ 续航 7m。不是论文里想象的"垂直跳跃"。

### v4 设计（基于 Phase 0 硬数据）

#### 新增 3 个结构量（纯代数推演，非 learned head）

| 量 | 公式 | 语义 | 阈值来源 |
|---|---|---|---|
| `torso_down` | `sigmoid((0.35 - torso_z) / 0.05)` | 躯干是否贴地 | 成功组长期低姿 0.17-0.24，失败组 0.07-0.10；0.35 给出 clean margin |
| `rear_trap` | `torso_down * sigmoid(-(pitch + 0.3) / 0.1)` | 死盆地标签（躯干贴地 ∧ pitch<-0.3）| 成功组 backward_pitch_frac=0%；失败组 78-99% |
| `forward_recoverability` | `sigmoid((pitch - 0.3) / 0.1) * sigmoid(torso_z / 0.5)` | 前倾可救（pitch>+0.3 ∧ torso 未完全倒）| 成功组 forward_pitch_frac=77-83%；失败组 0-2% |

#### Viability 门改造

- **原**：`predict_plan_reward` 用 `cos(pitch)` 做姿态 viability（前后对称）
- **新**：`forward_recoverability * (1 - rear_trap)`
  - 前倾可救 → viability ≈ 1（原有 standing×hopping reward 放行）
  - 死盆地触发 → viability ≈ 0（planner 拒绝该轨迹）

#### 代码落点

1. `src/world_model/dmcontrol_hopper_sd_model.py`：
   - `STATE_DIM` 15 → 18
   - `TORSO_DOWN_IDX=15`、`REAR_TRAP_IDX=16`、`FORWARD_RECOVER_IDX=17`
   - `encode_observation()` 追加三列代数计算
   - `forward()` / `_forward_one_substep()` 推进这三维
   - `predict_plan_reward()` 用新门控
2. `external/tdmpc2/tdmpc2/common/sd_world_model.py`：
   - `struct_dim` 假设自动跟随（从 `DMControlHopperSDModel.STATE_DIM` 读取）；检查 `_trend_protected_idx` / `_coupling_bridge_idx` 覆盖范围是否需要更新

#### 兼容性保证

- 与 ablation 1 的 `sd_use_struct_residual=false` 正交：新结构量不经过 `_struct_residual`，纯代数
- 与 ablation 0 的 `ENV_ACTION_REPEAT=2` 正交：新量通过 `forward()` 外层的循环自然继承双步推进
- 训练侧无需新辅助监督：三个量都是 obs 的代数函数，梯度从 consistency_loss 自然流动

### 执行顺序

1. ✅ Phase 0 状态日志（完成，硬数据 CSV 已落盘）
2. ⏳ v4 baseline smoke（进行中，GPU 0，5k seed=3，对比 fixrerun seed 3 10k=0.92）
3. 🔜 v4 结构改造代码落地（`sd_v4_asymmetric_contact` 分支）
4. 🔜 v4 structural smoke 5k seed=3
5. 🔜 v4 full 50k × 3（GPU 1/2/3）
6. 🔜 对 v4 checkpoint 重跑 Phase 0（渲染 + 状态日志）验证死盆地是否被打破

### 基线锚点汇总

| 配置 | 50k eval (mean ± std) | 代码版本 | 备注 |
|---|---|---|---|
| v3b_hh005（含 bug，已废弃）| 32.93 ± 13.03 | pre-fix + routeB | 不可复现，ablation/appendix |
| fixrerun（bug 修复后）| **23.0 ± 12.6** | master `a612a3f` | **论文当前主结果**，干净可复现 |
| ablation 0 + 1（transition 2× + no struct residual）| 待实验 | sd_v4 `ce5ed36` | v4 baseline smoke 验证后决定 |
| v4（+ asymmetric contact）| 待实验 | sd_v4 + structural commit | 目标 ≥ 23，同时希望 seed 1 的 5.64 盆地被打破 |
| TD-MPC2 100k (official) | 13.6 ± 10.4 | —— | 两倍预算 |
| DreamerV3 100k (official) | 9.7 ± 13.6 | —— | 两倍预算 |
