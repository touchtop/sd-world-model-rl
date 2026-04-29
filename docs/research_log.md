# Research Log

## 2026-04-29

Hopper `pureenv_sdinit_online_sup` 三种子 `50k` 正式对照 `run3` 已全部结束，并据此启动接续至 `100k` 的三种子训练。`50k run3` 的恢复源为：

- seed1: `logs/hopper-hop/1/sd_hopper_50000_pureenv_sdinit_online_sup_seed1_20260429_run3/models/final.full.pt`；
- seed2: `logs/hopper-hop/2/sd_hopper_50000_pureenv_sdinit_online_sup_seed2_20260429_run3/models/final.full.pt`；
- seed3: `logs/hopper-hop/3/sd_hopper_50000_pureenv_sdinit_online_sup_seed3_20260429_run3/models/final.full.pt`。

接续训练采用 `resume_from` 恢复完整 trainer 状态，不再传入 `sd_supervised_init_from`，以避免 full-state 恢复后重复执行监督初始化。三条新运行的命名与资源分配如下：

- seed1: `sd_hopper_100000_pureenv_sdinit_online_sup_seed1_resume50k_20260429_run1`，GPU 1，PID 2017505；
- seed2: `sd_hopper_100000_pureenv_sdinit_online_sup_seed2_resume50k_20260429_run1`，GPU 2，PID 2018577；
- seed3: `sd_hopper_100000_pureenv_sdinit_online_sup_seed3_resume50k_20260429_run1`，GPU 3，PID 2019644。

启动时首先在沙箱内尝试运行，但 `torch.cuda.is_available()` 为 false，未进入训练循环；随后改为非沙箱 GPU 环境启动。三条线均已确认从 `final.full.pt` 成功恢复，恢复点为 `step=50001`、`episode=100`、`total_updates=50000`、buffer `50100 steps / 100 eps`，并已进入新的训练循环。关键口径保持 `steps=100000`、`eval_freq=25000`、`checkpoint_freq=25000`、`eval_episodes=10`、`sd_latent_dim=512`、`sd_struct_loss_scale_mode=batch_std`、`sd_online_supervision_enabled=true`、`save_full_state=true`、`save_eval_traces=true`。该接续实验用于观察 `50k -> 100k` 阶段在线 `SD` 监督是否能继续改善整体 latent consistency、关键相位轨迹质量与闭环规划表现。

重新启动 Hopper `pureenv_sdinit_online_sup` 三种子 `50k` 对照实验。为保留此前仅运行至约 `3k` step 的 `run1` 半截记录，本次统一使用 `run2` 命名：

- seed1: `sd_hopper_50000_pureenv_sdinit_online_sup_seed1_20260429_run2`，GPU 0，PID 1240330；
- seed2: `sd_hopper_50000_pureenv_sdinit_online_sup_seed2_20260429_run2`，GPU 2，PID 1240331；
- seed3: `sd_hopper_50000_pureenv_sdinit_online_sup_seed3_20260429_run2`，GPU 3，PID 1240332。

三条运行均沿用 `pure-env + sd_latent_dim=17 + internal transition residual + supervised init + online SD supervision` 口径，步数为 `50k`，`eval_freq=25000`，`checkpoint_freq=25000`，`eval_episodes=10`。启动后已确认三条进程进入 TD-MPC2 训练入口，并在 `2500` step 写出 full checkpoint。该操作用于重新获得完整三种子 online supervision 对照结果，避免将此前半截 `run1` 误作为有效闭环结论。

随后对当前 SD 模型与 TD-MPC2 协同接口进行审查，发现 `SD` 结构态直接使用原始物理量，未做通道尺度归一化时，`omega / qdot / vz` 等大尺度通道会主导 `consistency_loss` 与在线 `SD` 监督损失，而 `touch_toe / touch_heel / z_min / H` 等接触相位关键小尺度通道的梯度相对不足。由于当前 online supervision 的研究目的正是压住 `landing / takeoff / near_contact / high_omega` 等关键相位误差，因此继续运行未归一化损失版本会使正式对照结论不够干净。

据此停止 `run2` 三种子进程，将其保留为在线监督接线 smoke 记录，而不作为正式 `50k` 对照结果。随后完成如下修复：

1. 保留默认 `sd_latent_dim=512` 与主线脚本中的 learned residual subspace，使当前主线继续采用“显式 17 维 SD 结构态 + learned residual latent”的设计；`sd_latent_dim=17` 仅保留为 SD-only 消融口径；
2. 新增 batch 统计驱动的结构态损失尺度归一化，对 `H, vx, vz, theta, omega, A, q, qdot, touch, z_min` 等通道按当前 batch 目标结构态标准差归一化，并设置统一下限；
3. 主 TD-MPC2 `consistency_loss` 中的 SD 结构态部分、在线 `SD` 监督的 `1-step` 与 rollout 损失、以及额外的 `omega/contact` 监督项均改为使用同一尺度归一化；
4. full checkpoint 保存与恢复中补入 `sd_optim` 状态，保证启用 online supervision 后可严格恢复。

该修复的目的不是改变 planner 接口，而是使 TD-MPC2 联合目标与独立在线监督支路在结构态误差定义上保持一致，避免未归一化物理量导致优化目标偏离关键接触相位。

针对上述修复继续补充一项口径校正：Hopper 主线不应把 `sd_latent_dim` 收缩为 `17`，因为当前问题的动力机制复杂，显式 `SD` 结构态不应被要求完整覆盖所有规划相关因素。主线恢复并保持 `sd_latent_dim=512`，即“17 维显式结构态 + 495 维 learned residual latent”。为避免 residual latent 因维度数更多而在 `consistency_loss` 中重新淹没结构态，损失合成改为结构态块与残差块默认等权，由 `sd_latent_consistency_struct_weight=1.0` 和 `sd_latent_consistency_resid_weight=1.0` 控制。此前尝试的固定通道尺度不再作为主线；当前结构态尺度依据 replay batch 目标结构态的逐通道标准差自适应估计，`sd_struct_loss_scale_floor=0.1` 仅作为数值稳定下限。

针对 Hopper `SD transition core` 当前闭环表现，补充完成一轮高层方法论收敛与文档整理。核心判断是：`SD dynamics` 的拟合目标与 TD-MPC2 的规划目标不应再被视为天然一致。当前证据已经显示，`consistency_loss` 在中期以后反弹，`75k` replay 上的 `landing / takeoff / high_omega` 误差仍显著，而 `100k` 闭环行为改善主要表现为“后仰失败态 -> 前倾推进态”的恢复子程序，而非 `hop` 行为链形成。

据此，后续研究口径正式拆分为两条线：

1. `model fidelity`：评估共享 `SD transition core` 在当前 replay 分布上的一步与短视界动力学保真度；
2. `planning usefulness`：评估该 core 是否改善动作排序、规划稳定性与闭环样本效率。

同时决定，若继续改训练方式，应优先采用“共享 transition core + 独立在线 SD 监督支路”，而不是引入两个相互竞争的 dynamics 头。详细分析见：

- [learnable_sd_v5_objective_decoupling_2026_04_29.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_objective_decoupling_2026_04_29.md)

## 2026-04-30

在比较当前 `pureenv_sdinit_online_sup` 与上一组同样去除奖励塑形的 `pureenv_sdinit` 后，确认当前在线 `SD` 监督版本在强 seed3 上低于上一组 `75k` 结果。该结果提示：平均结构态预测监督并不必然改善 TD-MPC2 的动作排序与闭环收益，单纯提高 `sd_online_supervision_lr` 也不一定能解决目标错配。

据此开始准备下一版本 planning-aware `SD` 目标。第一步已加入默认关闭的样本加权入口：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

该入口在启用时使用 TD 目标或奖励目标对在线 `SD` 监督样本加权，使结构态监督优先关注对 planner 更可能有价值的 replay 片段。权重按 batch 内 signal 均值归一化，避免启用该机制后等价于整体放大 `SD` 学习率。当前实现仍保留物理监督锚点，不直接把 `SD` 目标改成最大化回报；完整 action-ranking consistency loss 将作为后续扩展。详细记录见：

- [learnable_sd_v5_planning_aware_sd_2026_04_30.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_planning_aware_sd_2026_04_30.md)

相关方法文档与决策文档同步建立：

- [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
- [decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md)
- [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)

在此基础上，继续完成了第一版最小在线 `SD` 监督支路接线。实现保持单一共享 `SD transition core`，不引入第二个 dynamics 头，而是在 TD-MPC2 的 `_update()` 主流程中追加一次独立 `sd_optim` 监督更新。当前版本使用 replay batch 中的真实 `obs/action` 直接计算 `1-step` 与当前视界下的 `3-step` 监督误差，并对 `landing / takeoff / near_contact / high_omega` 关键相位提高权重，同时额外约束 `omega` 与 `touch_toe/touch_heel` 通道。

本轮实现的目的，是把“目标解耦”的方法论结论落实为一个最小、可 smoke 的工程闭环，为下一步检查训练日志、关键相位误差与闭环行为变化提供直接抓手。

同时，补充固定了一条更高层的方法论澄清：`RL` 规划目标与 `SD` 状态推演精度既不能被当成完全相同，也不能被当成彼此无关。二者在“未来状态演化必须可被可靠刻画”这一点上存在内在统一性，但在损失刻画上又分别强调平均保真度与任务相关误差。因此，当前在线 `SD` 监督支路的意义，不是预设二者天然一致，而是尝试在共享 `transition core` 上显式建立这种统一，并检验同一个动力学核心能否同时满足推演保真度与规划可用性两侧要求。详细文字已补入：

- [learnable_sd_v5_objective_decoupling_2026_04_29.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_objective_decoupling_2026_04_29.md)
