# Learnable SD v5：探索压缩假设、日志证据与修补方案复审

## 中文正式记录

本文记录 2026-05-01 对当前 Hopper planning-aware SD 实验线的专项复审。本次复审只分析代码、日志和实验现象，不修改训练代码。核心问题是：随着 SD 状态推演能力增强，是否可能反而压缩 TD-MPC2 原有探索，使策略更容易停留在局部最优行为盆地。

该担心不应被理解为“SD 越准越有害”。更准确的假设是：面向决策的 SD 世界模型不应以完整复刻环境动力学为唯一目标，而应为任务目标提供足够完备的状态推演能力；但当前 planning-aware 实现把“任务相关”过于简单地近似为当前 replay 分布中的 `td_positive` 样本。若当前策略已经偏向贴地前冲、低姿态弹跳或前倾撞地等局部高分行为，`td_positive` 权重会进一步提高这些片段在在线 SD 监督中的权重，使 SD core 在局部行为盆地内更准确，planner 在该盆地内更自信，却未必更好地覆盖稳定 hop 所需的 `grounded_upright -> grounded_prepared -> rising_air` 链条。

### 代码审查结论

当前 TD-MPC2 主训练框架没有被 planning-aware SD 版本直接破坏。原 TD-MPC2 的探索机制仍然存在，包括 seed data 阶段之后由 `agent.act()` 产生行为、`mpc=true` 时进入 latent-space planner、planner 中同时包含 policy prior 轨迹和随机 MPPI action 轨迹、elite 选择带有 Gumbel 扰动、训练模式下执行动作加入噪声，以及 actor loss 中保留 entropy 项。因此，如果出现探索不足，更合理的解释不是“TD-MPC2 原框架已经没有探索”，而是 SD 侧的拟合精度分配可能让 planner 更快、更自信地落入当前局部行为盆地，从而削弱这些探索机制在实践中的有效性。

当前实验也没有启用 SD action prior。`sd_action_prior_enabled=false`，因此不存在由 SD action box 直接裁剪动作空间、显式压制探索的问题。最可疑的机制是 `sd_online_planning_weight_enabled=true` 与 `sd_online_planning_weight_mode=td_positive` 的组合。该机制不改变 replay 抽样、Q loss、reward loss、actor update 或 MPPI 采样过程，但会把在线 SD 监督的样本权重分配给当前 critic 和当前 replay 分布已经认为较好的 transition。

### 关键 flag 复核

`sd_online_supervision_enabled` 启用独立在线 SD 监督支路。启用后，SD transition core 由 `sd_optim` 更新，而不是继续与 TD-MPC2 主 optimizer 混合更新。该设计把 SD core 的职责限定为 replay transition 的结构态监督学习，使主 TD-MPC2 损失主要训练 residual、reward/value/actor 等下游模块，避免同一组 dynamics 参数被两个目标、两个 optimizer 以不同动量状态同时更新。该设计正确，应保留。

`sd_detach_core_in_main_loss` 默认值为 `true`。在 `sd_online_supervision_enabled=true` 且该 flag 为真时，`SDWorldModel.next()` 会对 `_dynamics` 输出的 `next_struct_base` 执行 detach，再参与后续 residual 合成。其语义是主 consistency loss 不再直接反向训练 SD core，SD core 只由独立在线 SD 监督支路训练。该设计与“共享 transition core + 独立 SD 监督”的方法论一致，应保留。

`sd_online_supervision_phase_weight` 默认值为 `2.0`。代码根据真实结构态识别 `landing`、`takeoff`、`near_contact` 与 `high_omega`，并对这些机制关键相位提高在线 SD 监督权重。该机制不依赖当前 critic 或 episode return，更接近“任务相关状态推演”的本意，因为 Hopper 的稳定跳跃确实依赖接触发生、接触释放、近地状态和角速度控制。该设计方向正确，但仍未显式识别 `grounded_upright`、`grounded_prepared`、`full_foot_support` 或 `grounded_prepared -> rising_air` 等更高层姿态门控。

`sd_online_supervision_omega_weight=0.25` 与 `sd_online_supervision_contact_weight=0.25` 分别对 pitch rate 与 toe/heel contact 通道追加辅助监督损失，并使用同一套 step weights。它们属于 SD 侧辅助目标，重点约束 landing、takeoff、姿态恢复和足端支撑判断中的关键信息源，不改变 TD-MPC2 主训练框架。该设计可保留。

`sd_online_supervision_one_step_weight=1.0` 与 `sd_online_supervision_rollout_weight=1.0` 分别约束 replay transition 的一步结构态预测和当前 horizon 下的短视界 rollout 误差。一步监督提供局部动力学锚点，短视界 rollout 监督提供误差积累约束。该组合合理，仍属于 SD 侧监督。

`sd_online_planning_weight_enabled` 默认值为 `false`，但当前 planning-aware 脚本显式设为 `true`。启用后，`_sd_planning_weights()` 会从 TD target 或 reward target 生成样本权重。当前脚本使用 `td_positive`，即对 TD target 取正部，按 batch 均值归一化，再由 `sd_online_planning_weight_coef=1.0` 调整，并用 `sd_online_planning_weight_max=5.0` 截断。若正值模式下 batch 均值接近零，则回退为均匀权重。该权重会乘入 one-step SD 监督的 `step_weights`，rollout SD 监督则使用 `planning_weights.mean(dim=0, keepdim=True)`。

工程上，该实现是自洽的：shape 检查明确，权重不会在无正信号时变成全零，也不是简单放大整体 SD 学习率。研究语义上，该 flag 是当前最需要警惕的部分。`td_positive` 实际上要求当前 critic 在当前 replay 分布内判断哪些 transition 更值得 SD 学准；若当前策略尚未发现稳定站立和起跳链条，critic 就可能把局部高分行为误标为最有价值样本，使 SD core 的精度被分配到当前局部行为盆地，而不是分配到尚未充分探索但对最终 hop 更关键的姿态转移。

### 日志分析结论

当前三种子 `200k` resume 线为 seed1 `sd_hopper_200k_pureenv_sdinit_planning_sd_seed1_resume75k_20260501_run2`、seed2 `sd_hopper_200k_pureenv_sdinit_planning_sd_seed2_resume75k_20260501_run2`、seed3 `sd_hopper_200k_pureenv_sdinit_planning_sd_seed3_resume75k_20260501_run2`。

截至本次分析，官方评估结果如下：

| Step | seed1 | seed2 | seed3 | Mean | Sample std |
|---:|---:|---:|---:|---:|---:|
| 75k | 0.8098 | 0.0028 | 25.6353 | 8.8160 | 14.5715 |
| 100k | 19.2628 | 22.8490 | 42.8613 | 28.3244 | 12.7164 |
| 125k | 52.4183 | 25.6385 | 108.5301 | 62.1956 | 42.3019 |
| 150k | 94.5237 | 29.0374 | 57.2399 | 60.2670 | 32.8479 |

该结果说明当前方法并非完全失败。从 `75k` 到 `125k`，三种子均值从 `8.8160` 上升到 `62.1956`，收益层面存在有效学习。但收益并非稳定单调改善，`150k` 均值没有继续超过 `125k`，seed3 从 `108.5301` 回落到 `57.2399`。seed2 在 `150k` 的 episode 分布为 `0, 102.169, 0, 0, 89.298, 0, 0, 98.907, 0, 0.00066`，说明高分行为仍是间歇出现，而不是稳定策略。

训练日志中，`150k` 之后 recent train return 比对应 eval 更乐观：seed1 最近检查到 `168400` step，最近 10 个训练 episode reward 均值约 `104.02`；seed2 最近检查到 `158800` step，最近 10 个训练 episode reward 均值约 `58.12`；seed3 最近检查到 `165600` step，最近 10 个训练 episode reward 均值约 `112.16`。这说明训练仍在向后推进，且训练分布中存在更强高分片段，但不能直接证明正确姿态链条已经形成。

### 姿态证据

`100k` seed3 的高分轨迹中已经出现若干 `rising_air` 片段，但多数仍表现为前倾、贴地或低姿态弹跳，尚未形成稳定的严格 `grounded_upright` 或 `grounded_prepared` 阶段。

`125k` seed3 的 `episode_02` 出现了一个更明确的正确信号：存在单帧严格 `grounded_upright`。该帧约位于 `37 / 500`，时间约 `0.74s`，状态为 `pitch=-0.149`、`torso_z=0.839`、`foot_grounded=1`、`full_foot_support=0.989`、`body_collision=0`、`vx=1.599`。该帧说明策略已经能够短暂穿过正确姿态门槛，但下一帧以后由于 pitch rate 迅速变化，行为又回到前倾或贴地模式。

因此，当前更准确的描述是：正确姿态开始以瞬时 gate crossing 的形式出现，但尚未形成稳定相位链条。这与探索压缩假设相容。

### 对方法论的修正

“SD 的设计目标不是完美刻画系统动力机制，而是为任务目标提供完备的状态推演能力”这一哲学判断仍成立。需要修正的是当前实现中的代理变量。第一版 planning-aware 实现把“面向决策”简化成了“面向当前 critic 的正 TD 目标”，这一步过于简单。

从第一性原理看，任务相关状态推演至少包含两类信息：当前回报或价值函数已经能够识别的高价值片段，以及当前策略尚未稳定掌握、但对最终任务结构必不可少的机制瓶颈。Hopper 中第二类信息包括接触建立、接触释放、足端支撑、躯干直立、角速度抑制、准备姿态形成以及从准备姿态进入腾空。仅用 `td_positive` 加权，会偏向第一类信息，并可能忽略第二类信息。

### 建议的最小修补方案

下一条最干净的对照实验不应修改 TD-MPC2 主训练框架。建议只关闭 critic-dependent 的 SD 样本加权，保留机制关键相位监督：

```bash
sd_online_supervision_enabled=true
sd_online_supervision_one_step_weight=1.0
sd_online_supervision_rollout_weight=1.0
sd_online_supervision_omega_weight=0.25
sd_online_supervision_contact_weight=0.25
sd_online_supervision_phase_weight=2.0
sd_online_planning_weight_enabled=false
sd_action_prior_enabled=false
sd_detach_core_in_main_loss=true
```

该方案保留 SD core 的独立在线监督、one-step 与短视界 rollout 的结构态保真约束、接触/角速度/关键相位机制监督，以及 TD-MPC2 原有 replay、Q/reward loss、actor update 与 MPPI planner；同时移除当前 critic/replay 分布对 SD 拟合精度分配的直接牵引，并继续保持 `sd_action_prior_enabled=false`，避免引入动作空间裁剪这一混杂变量。

该修补方案最符合“回应探索压缩问题，但不破坏原训练框架”的要求。它把 SD 侧任务相关性从 critic-dependent proxy 回退到 mechanism-dependent proxy。

### 后续可选增强

若关闭 `sd_online_planning_weight_enabled` 后仍出现姿态链条不稳定，可以继续在 SD 侧增加 reward-agnostic 的稀有相位权重，而不是立即恢复 TD-positive 权重。可选方向包括对 `grounded_upright`、`full_foot_support`、`grounded_prepared`、`grounded_upright -> grounded_prepared`、`grounded_prepared -> rising_air` 增加监督权重。这些增强仍属于 SD 监督目标内部的机制建模，不改变 TD-MPC2 的 replay、critic、actor 或 planner 框架。

### 连带发现的问题

当前脚本的 planning budget 低于 TD-MPC2 默认值。当前使用 `num_samples=128`、`num_elites=16`、`num_pi_trajs=16`、`iterations=3`，而 TD-MPC2 默认配置为 `num_samples=512`、`num_elites=64`、`num_pi_trajs=24`、`iterations=6`。这不构成训练框架破坏，但会减少 planner 搜索多样性。若后续要判断局部最优主要来自 SD 侧目标，最好至少安排一次半恢复 planning budget 的对照，例如 `num_samples=256`、`num_elites=32`、`iterations=6`。

近期训练日志中出现过 `plan_std_mean=0.05`，即 planner std 达到配置的 `min_std`。这属于合法 TD-MPC2 行为，但说明在部分状态上 planner 已经收缩到最小采样尺度。在这种情况下，继续使用 TD-positive SD 加权可能进一步强化当前局部盆地。

当前日志中的 `sd_online_planning_weight_mean` 诊断能力不足。由于权重按 batch mean 归一化，该指标通常接近 `1.0`，只能说明没有显著改变平均学习率尺度，不能说明权重分布是否高度集中。若后续重新研究 planning weight，应额外记录 `sd_online_planning_weight_std`、`sd_online_planning_weight_max_observed`、`sd_online_planning_weight_clamp_frac`、正权重样本比例，以及高权重样本对应的姿态相位分布。

后续汇报必须把 return 指标与姿态指标分开。episode return 上升不等价于正确姿态链条形成。高分 episode 可能来自前冲、贴地、低姿态弹跳等局部行为，也可能只包含短暂正确姿态。进展汇报应同时包含当前 step、eval 均值与 episode 分布、高分轨迹姿态分类、是否出现连续 `grounded_upright`、`grounded_prepared` 和 `rising_air`，以及正确姿态是否只是单帧瞬态。

### 总结

本次复审确认，当前相关 flag 的工程实现基本正确。`sd_online_supervision_enabled`、`sd_detach_core_in_main_loss`、`sd_online_supervision_phase_weight`、`sd_online_supervision_omega_weight` 与 `sd_online_supervision_contact_weight` 符合“SD core 独立监督、机制关键相位加权、不破坏 TD-MPC2 主框架”的设计目标。

当前最需要回退的是 `sd_online_planning_weight_enabled=true` 与 `td_positive` 的组合。该组合不是实现错误，而是研究语义过强：它把任务相关性绑定到当前 critic 和当前 replay 分布，可能使 SD 更准确地服务于局部高分行为，而不是服务于最终稳定 hop 所需的完整姿态链条。建议下一条最小对照关闭 `sd_online_planning_weight_enabled`，保留机制关键相位 SD 在线监督，并继续保持 `sd_action_prior_enabled=false`。

## Background

This note records the May 1, 2026 review of the current Hopper planning-aware SD line. The working concern was that a more accurate SD transition core might unintentionally reduce the effective exploration supplied by TD-MPC2, causing the learned controller to remain in a local behavioral basin.

The concern should not be interpreted as "SD accuracy is harmful." The more precise hypothesis is:

> A decision-oriented SD model should not approximate the full environment for its own sake; it should support reliable task-relevant state rollout. However, the current implementation may have used an overly simple proxy for "task-relevant": positive TD targets under the current replay distribution.

If the current replay distribution is already biased toward a local high-return behavior, such as prone hopping or low-posture forward bouncing, TD-positive SD weighting can make the SD core more accurate exactly in that local basin. This can make planning more confident inside the basin without improving coverage of the key transitions needed for a stable hop cycle.

## Code Review Summary

The original TD-MPC2 exploration mechanisms remain present in the code.

1. Environment interaction uses random actions only for seed data; after `seed_steps`, behavior is generated by `agent.act()`.
   Code: `external/tdmpc2/tdmpc2/trainer/online_trainer.py`, lines 498-501.
2. With `mpc=true`, `agent.act()` calls the latent-space planner.
   Code: `external/tdmpc2/tdmpc2/tdmpc2.py`, lines 144-163.
3. The planner samples policy-prior trajectories, random MPPI action trajectories, selects elites, samples an elite via Gumbel noise, and adds action noise at execution time when not in eval mode.
   Code: `external/tdmpc2/tdmpc2/tdmpc2.py`, lines 200-269.
4. The actor update includes an entropy term.
   Code: `external/tdmpc2/tdmpc2/tdmpc2.py`, lines 296-304.

The current planning-aware SD experiment does not directly clamp actions through the SD action prior. The relevant flag is `sd_action_prior_enabled=false`.

1. Default configuration has `sd_action_prior_enabled=false`.
   Code: `external/tdmpc2/tdmpc2/config.yaml`, line 33.
2. Current planning-aware scripts also explicitly pass `sd_action_prior_enabled=false`.
   Scripts: `scripts/run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh` and `scripts/run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh`.

Therefore, the most likely exploration-bias source is not explicit action-box restriction. It is the SD online supervision weighting.

## Flag Semantics

### `sd_online_supervision_enabled`

This flag enables the independent online SD supervision update. When enabled, `model._dynamics` is removed from the TD-MPC2 main optimizer and updated by `sd_optim` only. This is consistent with the current design: the SD core is supervised as a replay transition model, while TD-MPC2 heads and residual modules are trained by the main objective.

This design is conceptually correct and should be retained.

### `sd_detach_core_in_main_loss`

Default value: `true`.

When `sd_online_supervision_enabled=true` and `sd_detach_core_in_main_loss=true`, `SDWorldModel.next()` detaches `next_struct_base` before residual composition. Thus the main TD-MPC2 consistency loss does not train the SD core directly.

Relevant implementation: `external/tdmpc2/tdmpc2/common/sd_world_model.py`, lines 220-224.

This prevents double optimization of the same SD core through incompatible objectives. The design is correct for the current separation of responsibilities.

### `sd_online_supervision_phase_weight`

Default value: `2.0`.

This flag increases replay supervision weight for mechanism-critical transitions: landing, takeoff, near-contact, and high angular velocity.

Relevant implementation: `external/tdmpc2/tdmpc2/tdmpc2.py`, lines 377-398.

This is reward-agnostic and mechanism-oriented. It does not alter TD-MPC2 replay sampling, Q loss, actor loss, or planner sampling. It is the cleanest current expression of "task-relevant SD accuracy" because it emphasizes transition bottlenecks without asking the current critic to decide which transitions matter.

The design is directionally correct. A limitation is that it does not explicitly identify `grounded_upright`, `grounded_prepared`, or `prepared -> rising_air` as special targets. It only captures them indirectly through takeoff and contact geometry. A future SD-side extension could add rare-transition phase gates while staying outside the TD-MPC2 training pipeline.

### `sd_online_supervision_omega_weight` and `sd_online_supervision_contact_weight`

Default values: `0.25` and `0.25`.

These weights add extra SD losses on pitch rate and toe/heel contact channels, using the same step weights as the main one-step SD supervision.

Relevant implementation: `external/tdmpc2/tdmpc2/tdmpc2.py`, lines 457-470 and 488-493.

The design is correct as an SD-side auxiliary objective. It targets dynamics quantities that are central to landing, takeoff, and posture transitions. It does not change TD-MPC2 actor/Q/replay logic.

### `sd_online_planning_weight_enabled`

Default value: `false`. Current planning-aware scripts explicitly set it to `true`.

When enabled, this flag computes sample weights from TD targets or reward targets, normalizes them by the batch mean, clips them by `sd_online_planning_weight_max`, and multiplies them into SD one-step and rollout supervision weights.

Relevant implementation:

1. Weight construction: `external/tdmpc2/tdmpc2/tdmpc2.py`, lines 400-430.
2. Application to one-step SD supervision: lines 446-455.
3. Application to rollout SD supervision: lines 476-480.

The mechanics are internally consistent:

1. `td_positive` and `reward_positive` use `clamp_min(0)`.
2. `td_abs` and `reward_abs` use absolute value.
3. Weights are normalized by the batch mean, so enabling the mechanism is not intended to simply increase the effective SD learning rate.
4. If the batch has no positive signal in a positive mode, it falls back to uniform weights.
5. Shape checking is explicit.

However, the research semantics are now suspect. `td_positive` asks the current critic and current replay distribution to identify which transitions should be learned more accurately by SD. This can be circular:

1. early policy finds a local behavior basin;
2. critic assigns positive TD targets to that basin;
3. SD becomes more accurate in that basin;
4. planner becomes more confident there;
5. behavior distribution continues to feed the same basin.

This does not break TD-MPC2's algorithmic plumbing, but it can bias the SD side toward the current policy's local optimum. In that sense, it is a plausible mechanism for suppressing the practical effect of the original TD-MPC2 exploration sources.

## Log Evidence

The current three-seed 200k resume line is:

1. seed1: `sd_hopper_200k_pureenv_sdinit_planning_sd_seed1_resume75k_20260501_run2`;
2. seed2: `sd_hopper_200k_pureenv_sdinit_planning_sd_seed2_resume75k_20260501_run2`;
3. seed3: `sd_hopper_200k_pureenv_sdinit_planning_sd_seed3_resume75k_20260501_run2`.

The eval trajectory through 150k is:

| Step | seed1 | seed2 | seed3 | Mean | Sample std |
|---:|---:|---:|---:|---:|---:|
| 75k | 0.8098 | 0.0028 | 25.6353 | 8.8160 | 14.5715 |
| 100k | 19.2628 | 22.8490 | 42.8613 | 28.3244 | 12.7164 |
| 125k | 52.4183 | 25.6385 | 108.5301 | 62.1956 | 42.3019 |
| 150k | 94.5237 | 29.0374 | 57.2399 | 60.2670 | 32.8479 |

This supports several observations:

1. The method is not failing in return terms. From 75k to 125k, the mean improves strongly.
2. The behavior is not monotonically improving. Seed3 drops from `108.53` at 125k to `57.24` at 150k.
3. Seed1 improves strongly by 150k, but posture analysis before this point showed that high return did not yet imply stable `grounded_prepared` formation.
4. Seed2 remains high-variance at 150k: its 150k score is `29.0374`, with episode scores `0, 102.169, 0, 0, 89.298, 0, 0, 98.907, 0, 0.00066`.

Recent train-return signals after 150k are stronger than the corresponding eval means:

1. seed1 latest checked step `168400`, last-10 train episode mean `104.02`;
2. seed2 latest checked step `158800`, last-10 train episode mean `58.12`;
3. seed3 latest checked step `165600`, last-10 train episode mean `112.16`.

This suggests that return quality is still evolving, but it also reinforces the need to separate "score is improving" from "the correct posture chain has formed."

## Posture Evidence

At 100k, high-score episodes already contained repeated `rising_air` snippets, especially in seed3, but no stable strict `grounded_upright` or `grounded_prepared` segment.

At 125k, seed3 showed a more concrete posture signal:

1. `seed3 / step_0125000 / episode_02` had one strict `grounded_upright` frame.
2. The key frame was at index `37 / 500`, around `0.74s`.
3. It had `pitch=-0.149`, `torso_z=0.839`, `foot_grounded=1`, `full_foot_support=0.989`, `body_collision=0`, and `vx=1.599`.
4. It was not stable: the next frame returned to prone-like behavior because `pitch_rate` rose sharply.

The interpretation is that correct posture is appearing as a transient gate crossing, not as a stable behavioral phase. This is consistent with a system that has discovered high-return forward bouncing but has not yet stabilized the intended `grounded_upright -> grounded_prepared -> rising_air` chain.

## Interpretation

The guiding philosophy remains valid:

> The SD component is not meant to be a complete physical simulator. It is meant to supply state rollouts that are complete enough for the task-relevant decisions that TD-MPC2 must make.

The implementation issue is that the first planning-aware proxy equated "task-relevant" with "currently TD-positive." That proxy is too simple. A current critic can score local behaviors positively before the correct posture chain is discovered. Once that happens, TD-positive SD weighting can make the SD core most accurate in the discovered local mode.

A better implementation of the philosophy should identify task-relevant dynamics mechanistically, not only through current return estimates. For Hopper, such mechanisms include contact onset, contact release, angular velocity control, full-foot support, upright support, prepared stance, and transition into rising air.

## Recommended Repair

The cleanest next experiment should not modify the TD-MPC2 training framework. It should remove the reward/critic-dependent SD weighting while preserving mechanism-critical SD supervision.

Recommended flags:

```bash
sd_online_supervision_enabled=true
sd_online_supervision_one_step_weight=1.0
sd_online_supervision_rollout_weight=1.0
sd_online_supervision_omega_weight=0.25
sd_online_supervision_contact_weight=0.25
sd_online_supervision_phase_weight=2.0
sd_online_planning_weight_enabled=false
sd_action_prior_enabled=false
sd_detach_core_in_main_loss=true
```

This preserves original TD-MPC2 replay sampling, Q/reward losses, actor update, MPPI exploration path, and SD-only replay supervision on mechanism-critical transition phases.

It removes critic-dependent sample prioritization inside SD supervision and the circular coupling between current TD-positive local behavior and SD accuracy allocation.

If a further SD-side refinement is needed, it should remain reward-agnostic and mechanism-based. A suitable future extension would add optional rare-transition weights for `grounded_upright`, `grounded_prepared`, `full_foot_support`, `grounded_upright -> grounded_prepared`, and `grounded_prepared -> rising_air`.

## Secondary Issues Found During Review

### Reduced planning budget

The current scripts use:

```bash
num_samples=128
num_elites=16
num_pi_trajs=16
iterations=3
```

TD-MPC2 default config uses:

```yaml
num_samples: 512
num_elites: 64
num_pi_trajs: 24
iterations: 6
```

This does not break the original framework, but it reduces the planner's search diversity. If compute allows, a later comparison should test at least a half-restored planning budget, such as `num_samples=256`, `num_elites=32`, `iterations=6`, before attributing all local-optimum behavior to SD supervision.

### Planner std can collapse locally

Recent train logs contain some rows with `plan_std_mean=0.05`, exactly the configured `min_std`. This is legal TD-MPC2 behavior, but it means that, at some states, the planner has already collapsed to the minimum exploration scale after elite fitting. This is another reason to avoid adding SD-side weighting that further sharpens the current local basin.

### `sd_online_planning_weight_mean` is not diagnostic enough

Because weights are mean-normalized, `sd_online_planning_weight_mean` is almost always near `1.0`. This confirms that the code avoids changing average learning-rate scale, but it does not reveal how concentrated the weights are. If this mechanism is revisited, additional logging such as weight standard deviation, max, and fraction at clamp limits should be added.

## Conclusion

The current code correctly implements the intended flags mechanically. The problem is semantic rather than syntactic: `td_positive` is a fragile proxy for task relevance. It can align SD learning with whatever behavior the current critic already values, including local optima.

The next clean ablation should disable `sd_online_planning_weight_enabled` and retain mechanism-critical online SD supervision. This keeps the TD-MPC2 training framework intact while testing whether SD can support task-relevant state rollout without narrowing exploration around the current TD-positive behavior basin.
