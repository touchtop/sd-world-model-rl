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

在 `pureenv_sdinit_online_sup` 三种子 `100k` 结束后，对补跑的 `official_eval/step_0100000` 结果进行姿态复核。三种子均值为 `18.74`，高于外部 TD-MPC2 官方 CSV 中 `hopper-hop` 100k 三种子均值 `13.63`，但优势主要来自 seed3。轨迹层面，agent 已经出现从倒地或低姿态中抬升并短暂接近站立的动作，尤其 seed2 `episode_03` 在宽松姿态判据下有最长连续 `18` 帧的接近直立片段。该轨迹已按操作指南渲染为：

- [sd_online_sup_100k_seed2_ep03_most_upright.mp4](/home/data/round/code/sd_world_model_rl/results/hopper_trace_renders/sd_online_sup_100k_seed2_ep03_most_upright.mp4)

该现象表明当前 `SD` 优化方向并非只强化贴地爬行，而是已经开始支持“起身/姿态转正”的可规划动作结构。后续版本因此继续采用 planning-aware `SD` 监督方向。

据此新增 `100k` planning-aware `SD` 三种子启动脚本：

- [run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh)

该脚本沿用 `5k` planning-aware smoke 已验证配置，保持纯环境奖励、supervised `SD` 初始化、共享 `SD transition core`、在线 `SD` 监督、TD-positive planning-aware 样本重加权和 `sd_latent_dim=512`。训练目标仍为 `100k`，但实际 `steps` 设置为 `100500`，用于给 `100k` 评估触发留出一个 Hopper episode 的边界缓冲，减少末端 `eval.csv` 与 `official_eval/step_0100000/` 漏写。

随后按该脚本启动 planning-aware `SD` 三种子 `100k` 正式实验：

- seed1: `sd_hopper_100k_pureenv_sdinit_planning_sd_seed1_20260430_run1`，GPU 1，PID 2706767，会话 74704；
- seed2: `sd_hopper_100k_pureenv_sdinit_planning_sd_seed2_20260430_run1`，GPU 2，PID 2708360，会话 35611；
- seed3: `sd_hopper_100k_pureenv_sdinit_planning_sd_seed3_20260430_run1`，GPU 3，PID 2711933，会话 54319。

三条线均已确认加载 supervised `SD` checkpoint，参数显示 `steps=100500`、`eval_freq=25000`、`checkpoint_freq=25000`、`save_full_state=true`、`save_eval_traces=true`，并在 `2500` step 写出 `2500.pt` 与 `2500.full.pt`。

在比较当前 `pureenv_sdinit_online_sup` 与上一组同样去除奖励塑形的 `pureenv_sdinit` 后，确认当前在线 `SD` 监督版本在强 seed3 上低于上一组 `75k` 结果。该结果提示：平均结构态预测监督并不必然改善 TD-MPC2 的动作排序与闭环收益，单纯提高 `sd_online_supervision_lr` 也不一定能解决目标错配。

据此开始准备下一版本 planning-aware `SD` 目标。第一步已加入默认关闭的样本加权入口：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

该入口在启用时使用 TD 目标或奖励目标对在线 `SD` 监督样本加权，使结构态监督优先关注对 planner 更可能有价值的 replay 片段。权重按 batch 内 signal 均值归一化，避免启用该机制后等价于整体放大 `SD` 学习率。当前实现仍保留物理监督锚点，不直接把 `SD` 目标改成最大化回报；完整 action-ranking consistency loss 将作为后续扩展。详细记录见：

- [learnable_sd_v5_planning_aware_sd_2026_04_30.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_planning_aware_sd_2026_04_30.md)

随后对该发现的高层含义进行了整理：`SD` 世界模型不应被定位为对环境动力学的完备复刻，而应被定位为“足以支撑正确决策的结构化抽象”。该口径与人类智能中的世界理解相近，即有限表征并不追求完美复现世界，而是服务于行动选择、关键状态辨别和后果预期。为避免“偏置”被误解为有害统计偏差，并准确反映 `SD` 来源于系统工程和系统仿真中的系统动力学（System Dynamics），本阶段术语暂定为 `Decision-Oriented System-Dynamics-Guided World Model`，中文为“面向决策的系统动力学引导世界模型”。

同时记录一项未来理论方向：若该工作进一步上升到方法论创新，核心难点可能在于从数学和统计层面论证 `SD` 刻画精度何时已经足以支撑决策，而不是单纯追求对真实环境的完备拟合。该问题需要刻画模型误差、价值误差、动作排序误差与闭环策略表现之间的关系，并区分 replay 分布、规划 imagined 分布和真实闭环访问分布。当前阶段仍以经验性训练规划支撑能力为主，该理论充分性问题暂作为后续工作方向记录。

相关方法文档与决策文档同步建立：

- [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
- [decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md)
- [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)

在此基础上，继续完成了第一版最小在线 `SD` 监督支路接线。实现保持单一共享 `SD transition core`，不引入第二个 dynamics 头，而是在 TD-MPC2 的 `_update()` 主流程中追加一次独立 `sd_optim` 监督更新。当前版本使用 replay batch 中的真实 `obs/action` 直接计算 `1-step` 与当前视界下的 `3-step` 监督误差，并对 `landing / takeoff / near_contact / high_omega` 关键相位提高权重，同时额外约束 `omega` 与 `touch_toe/touch_heel` 通道。

本轮实现的目的，是把“目标解耦”的方法论结论落实为一个最小、可 smoke 的工程闭环，为下一步检查训练日志、关键相位误差与闭环行为变化提供直接抓手。

同时，补充固定了一条更高层的方法论澄清：`RL` 规划目标与 `SD` 状态推演精度既不能被当成完全相同，也不能被当成彼此无关。二者在“未来状态演化必须可被可靠刻画”这一点上存在内在统一性，但在损失刻画上又分别强调平均保真度与任务相关误差。因此，当前在线 `SD` 监督支路的意义，不是预设二者天然一致，而是尝试在共享 `transition core` 上显式建立这种统一，并检验同一个动力学核心能否同时满足推演保真度与规划可用性两侧要求。详细文字已补入：

- [learnable_sd_v5_objective_decoupling_2026_04_29.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_objective_decoupling_2026_04_29.md)

随后根据外部专家对当前 `SDWorldModel` 与 TD-MPC2 接口的静态审查，完成一轮接口修复与文档补录。主要修复包括：

1. 启用 `sd_online_supervision_enabled=true` 时，`model._dynamics` 不再同时归属于 TD-MPC2 主优化器与 `sd_optim`，而是只由 `sd_optim` 管理，避免同一参数被两个 Adam 优化器用不同动量状态更新；
2. `sd_trend_protect_min_correction` 从原先隐含的绝对下限 `0.02` 改为显式配置，默认值为 `0.0`，避免 `ROOTZ / ZMIN` 等小信号米制维度被强制产生单位错配的虚假漂移；
3. `sd_reward_mode=learned` 的 reward target 统一返回已经乘以 `sd_env_reward_scale` 的环境奖励，避免 learned、hybrid、curriculum 路径之间出现静默尺度不一致；
4. 结构态 residual 继续保留为主线设计，但各通道 residual cap 已显式配置化，并新增默认关闭的 `sd_struct_residual_penalty_coef`，用于后续检查 residual 是否过度覆盖系统动力学 core；
5. full checkpoint 恢复逻辑补充 optimizer 参数组不兼容时的兼容路径，以支持本轮优化器归属变化后的旧 checkpoint 恢复。

本轮没有取消 residual，也没有把 `sd_latent_dim` 改回 `17`，因为当前设计仍要求“17 维显式系统动力学结构态 + learned residual latent”共同表达 Hopper 的复杂动力机制。在线 `SD` 监督仍锚定 `_dynamics` 本体，外层 residual heads 由 TD-MPC2 主损失更新；该分工用于区分系统动力学 core 的保真锚点与 residual 对规划相关未建模机制的补偿。详细记录见：

- [learnable_sd_v5_tdmpc2_interface_review_2026_04_30.md](/home/data/round/code/sd_world_model_rl/docs/worklog/learnable_sd_v5_tdmpc2_interface_review_2026_04_30.md)

同时同步更新：

- [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
- [decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md)
- [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)

根据复审意见继续补充一个小补丁。启用 `sd_online_supervision_enabled=true` 时，新增默认口径 `sd_detach_core_in_main_loss=true`：`SDWorldModel.next()` 中 `_dynamics` 输出的 `next_struct_base` 在主 consistency 路径中 detach 后参与 residual 合成。该修复使主 `consistency_loss` 不再给 `_dynamics` 计算随后被丢弃的梯度，语义上也更明确地表示：`SD core` 由独立在线 `SD` 监督训练，主 consistency 主要训练外层 residual heads 与 TD-MPC2 下游头。

同时新增 `struct_residual_activity=mean(abs(tanh(raw_struct_residual)))` 训练指标，用于下一轮 smoke 判断结构态 residual 是否长期高强度兜底。当前仍未直接收紧 `sd_vz_residual_scale=3.0` 与 `sd_pitch_rate_residual_scale=8.0`，原因是残差补偿是主线设计的一部分，直接收紧会与优化器归属修复形成混杂变量。下一次 5k 或 10k smoke 应明确观察 `struct_consistency_loss / resid_consistency_loss`、`struct_residual_activity`、在线 `SD` loss 趋势和闭环得分，再决定是否将默认 residual scale 收紧到如 `vz=1.0`、`pitch_rate=3.0` 并配合 `sd_struct_residual_penalty_coef=0.05` 做消融。

此外，full checkpoint 恢复中的 optimizer state 兼容路径已扩展为捕获 `(ValueError, KeyError, RuntimeError)`，并在告警中记录旧参数组数与新参数组数，以便定位接口修复前后 optimizer 参数组变化导致的恢复差异。
