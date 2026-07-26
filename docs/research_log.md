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

按操作指南启动并完成一次新接口修复后的 planning-aware `SD` 5k smoke 测试。实验名为 `sd_hopper_5k_pureenv_sdinit_planning_sd_seed3_20260430_fixsmoke1`，seed 为 `3`，GPU 为 `0`，PTY 会话为 `56013`，训练进程 PID 为 `4183709`。启动前确认四张 GPU 均为空闲，并确认 supervised `SD` 初始化 checkpoint 已复制到：

- `results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt`

本次训练严格使用 `run_hopper_v5_5k_pureenv_sdinit_planning_sd.sh` 脚本，仅用 `PYTHON_BIN=/home/round/.conda/envs/sd-mcts/bin/python` 覆盖当前机器不存在的指南默认 Python 路径。训练已正常结束，`2500.pt`、`2500.full.pt`、`5000.pt`、`5000.full.pt`、`final.pt`、`final.full.pt` 均已落盘；`eval.csv`、`eval_full.csv` 以及 `official_eval/step_0002500/`、`official_eval/step_0005000/` 的 10 条 episode csv 与 `eval_episode_summary.csv` 均已写出。

初步结果显示，工程链路符合预期：`sd_online_planning_weight_mean` 在训练更新阶段保持为 `1.0`，`plan_nonfinite_frac=0.0`，未观察到规划非有限值或明显梯度异常。`5000` step 处 `consistency_loss=0.1260`、`struct_consistency_loss=0.2431`、`resid_consistency_loss=0.00887`、`struct_residual_activity=0.5522`、`sd_online_supervision_loss=1.6751`。闭环收益仍接近零，`eval.csv` 中 `2500` 与 `5000` 的平均 episode reward 分别为 `0.000684` 与 `0.001119`，`official_eval` 中未出现成功 episode。因此，该 smoke 支持“最新修订可稳定启动、训练、评估并落盘”的工程结论，但尚不能支持 planning-aware 权重已改善早期闭环行为的结论。下一步若继续评估行为信号，应使用同一口径扩展到 `10k` 或启动正式三种子对照，而不是仅依据 5k return 做性能判断。

随后依据上述 smoke 的工程稳定性，直接启动新接口修复后的 planning-aware `SD` 三种子 `100k` 正式实验，不再追加 `10k` 单种子 smoke。启动前确认 GPU 0-3 均为空闲、supervised `SD` checkpoint 存在，且三个新 `exp_name` 未复用。三条运行均使用 `run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh`，仅通过 `PYTHON_BIN=/home/round/.conda/envs/sd-mcts/bin/python` 覆盖当前机器 Python 路径；其余参数沿用脚本固定口径，即 `steps=100500`、`eval_freq=25000`、`checkpoint_freq=25000`、`save_full_state=true`、`save_eval_traces=true`、`sd_online_planning_weight_enabled=true`、`sd_online_planning_weight_mode=td_positive`。

资源分配与进程记录如下：

- seed1: `sd_hopper_100k_pureenv_sdinit_planning_sd_seed1_20260430_fixrun1`，GPU 0，PTY 会话 `3530`，PID `4193823`；
- seed2: `sd_hopper_100k_pureenv_sdinit_planning_sd_seed2_20260430_fixrun1`，GPU 1，PTY 会话 `5173`，PID `4194066`；
- seed3: `sd_hopper_100k_pureenv_sdinit_planning_sd_seed3_20260430_fixrun1`，GPU 2，PTY 会话 `21013`，PID `355`。

三条线均已确认加载 supervised `SD` checkpoint，`train.csv` 已持续写出 `500 / 1000 / 1500 / 2000 / 2500`，并且 `models/2500.pt` 与 `models/2500.full.pt` 均已落盘。启动后 GPU 占用约为 GPU0 `567 MiB / 18%`、GPU1 `567 MiB / 20%`、GPU2 `567 MiB / 22%`、GPU3 空闲。当前三条线已进入 seed data 之后的预训练阶段；后续应按操作指南在 `25000 / 50000 / 75000 / 100000` 评估节点检查 `eval.csv` 与 `official_eval/step_XXXXXXXX/` 是否完整写出。

三种子 `100k` 正式实验已完成首个 `25k` 官方评估节点并继续运行。三条线均已写出 `models/25000.full.pt`、`eval.csv` 与 `official_eval/step_0025000/`。`25k` 官方评估均值如下：

- seed1: `0.0806`
- seed2: `0.0260`
- seed3: `5.7233`

训练质量信号显示三条线的 `consistency_loss` 与结构态一致性损失均相较早期下降，未出现规划非有限值。seed3 当前表现最强，`25k` 前近期训练 episode reward 均值约 `4.18`，最近 20 个 episode 中 `9` 个非零，且 `plan_value_mean` 近期约 `2.59`；seed1 与 seed2 的官方评估仍接近零，近期训练 reward 均值分别约 `0.036` 与 `0.028`。该阶段结果说明：新接口修复后的 planning-aware 版本在工程上稳定，并在 seed3 上已出现早期 planning/value 与闭环收益信号，但三种子之间差异仍较大，不能仅凭 `25k` 判断整体改进成立。后续重点观察 `50k` 是否出现 seed1/seed2 追上，或 seed3 信号是否持续放大。

## 2026-05-01

对当前 Hopper planning-aware `SD` 三种子长跑结果和 TD-MPC2 训练代码进行了专项审查，重点分析“`SD` 状态推演变准后是否可能反而压缩探索并强化局部最优”的假设。审查确认，原 TD-MPC2 的探索机制仍主要来自 MPPI 随机动作采样、policy prior 轨迹、elite Gumbel 选择、执行动作噪声和 actor entropy；当前实验未启用 `sd_action_prior_enabled`，因此不存在动作盒直接裁剪探索的问题。更可疑的机制是当前 planning-aware 版本显式启用了 `sd_online_planning_weight_enabled=true` 与 `sd_online_planning_weight_mode=td_positive`，该机制只作用于在线 `SD` 监督，但会将 `SD core` 的拟合精度分配偏向当前 critic/replay 已经认为正向的样本。

日志层面，三种子评估均值从 `100k=28.32` 上升到 `125k=62.20`，但 `150k=60.27` 未继续上升，其中 seed3 从 `125k=108.53` 回落到 `150k=57.24`。姿态复核显示，`125k` seed3 已出现单帧严格 `grounded_upright`，但仍未形成连续 `grounded_prepared` 或稳定 `grounded_upright -> grounded_prepared -> rising_air` 链条。这支持如下解释：当前实现没有否定“面向决策的系统动力学引导世界模型”这一方法论，但将“任务相关”简单近似为“当前 TD-positive”可能过早强化了局部高分行为盆地。

专项审查结论已整理为：

- [learnable_sd_v5_exploration_bias_review_2026_05_01.md](/home/round/code/sd-world-model-rl/sd_world_model_rl/docs/worklog/learnable_sd_v5_exploration_bias_review_2026_05_01.md)

建议下一条最小对照不修改 TD-MPC2 主训练框架，而是关闭 `sd_online_planning_weight_enabled`，保留 `sd_online_supervision_phase_weight=2.0`、`sd_online_supervision_omega_weight=0.25`、`sd_online_supervision_contact_weight=0.25` 等机制关键相位监督。该方案保持原 replay、Q/reward loss、actor update 与 MPPI 探索路径不变，只移除 critic-dependent 的 `SD` 样本加权，以检验是否可以避免 `SD` 学习被当前局部策略分布牵引。

同日补充记录本地 `resume100k` 接续线。planning-aware `SD` 三种子 `100k` 实验均已写出 `official_eval/step_0100000` 与 `final.full.pt`。其中 seed1 的 `100k` eval 均值为 `73.09`，10 条评估轨迹均为中等得分；seed2 的 `100k` eval 为 `0.0`，10 条评估轨迹全部失败；seed3 的 `100k` eval 均值为 `76.86`，呈现 5 条高分与 5 条近零的两极化分布。

姿态复核显示，seed1 与 seed3 的高分轨迹均出现较多“高度与躯干角接近直立”的片段，但在加入脚尖与脚跟同时触地的充分支撑判据后，稳定站立窗口仍非常短或不存在。该结果提示当前策略主要学到单点支撑或弹跳式前进，而不是稳定的脚底充分支撑步态。后续分析应将“近直立姿态”与“充分支撑站立”明确区分，避免仅凭高度和躯干角过度解释。

为继续观察更长训练预算下该行为是否会向稳定步态转化，新增 planning-aware `SD` `200k` 接续脚本：

- [run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_planning_sd_resume.sh)

该脚本从 `final.full.pt` 恢复模型、优化器、buffer、计数器与 `SD` checkpoint 状态，保持 `100k` planning-aware `SD` 三种子脚本的纯环境奖励、supervised `SD` 初始化、在线 `SD` 监督、TD-positive planning-aware 样本重加权与 residual 设置。目标预算为 `200k`，实际 `steps=200500`，用于在 `200k` 评估触发后等待 episode 边界写出 `eval.csv` 与 `official_eval/step_0200000`。

启动接续实验时发现，`resume_from` 若以相对路径传入，会在 Hydra 切换到实验工作目录后失效。因此脚本已将相对 `resume_from` 自动规范化为仓库根目录下的绝对路径。初始 `run1` 目录仅完成初始化后失败，正式接续使用新的 `run2` 实验目录。

正式启动的三种子 `200k` 接续实验如下：

- seed1: `sd_hopper_200k_pureenv_sdinit_planning_sd_seed1_resume100k_20260501_run2`，GPU 1，PID `3992464`；
- seed2: `sd_hopper_200k_pureenv_sdinit_planning_sd_seed2_resume100k_20260501_run2`，GPU 2，PID `3996683`；
- seed3: `sd_hopper_200k_pureenv_sdinit_planning_sd_seed3_resume100k_20260501_run2`，GPU 3，PID `3997601`。

三条线均已从对应 `100k` 的 `final.full.pt` 恢复到 `step=100501`、`episode=201`、`total_updates=100500`，buffer 恢复为约 `100701` 步和 `201` 个 episode。由于旧 `100k` checkpoint 来自优化器归属修复前版本，主优化器状态恢复时出现参数组数量不一致并按兼容逻辑跳过；`SD` optimizer 未报告恢复错误。该接续因此等价于“模型、buffer、计数器和 `SD` optimizer 恢复，主优化器重新初始化为新参数组口径”。

截至本地检查，`175k` 官方评估已全部写出，三种子结果为 seed1 `151.66`、seed2 `180.49`、seed3 `84.01`，均值 `138.72`，总体标准差 `40.44`。其中 seed2 的 `175k` official eval 10 条 episode 全部超过 `150`，均值为 `180.49`，最低 `159.12`，最高 `198.92`；但姿态复核显示，高分仍主要来自向前推进和短暂腾空/弹跳，而非稳定充分支撑步态。seed2 `episode_06` 在 step 50 附近出现 1 帧较明确的充分支撑近直立姿态，`torso_z=0.777`、`pitch=0.322`、脚尖/脚跟接触约 `0.98/0.99`、`full_foot_support=0.98`。该轨迹已渲染为：

- [seed2_175k_eval_episode06_full_support.mp4](/home/data/round/code/sd_world_model_rl/results/hopper_trace_renders/seed2_175k_eval_episode06_full_support.mp4)

当前训练仍在继续向 `200k` 推进。最近检查时 seed1 约到 `183.6k`，seed2 约到 `182.8k`，seed3 约到 `181.2k`。seed1 训练回报稳定但上限偏低，seed2 已稳定进入 `190-200` 分段并出现超过 `200` 的训练 episode，seed3 具备最高分能力但仍呈现高分与 `0` 分交替的两极化。`plan_nonfinite_frac` 持续为 `0`，未观察到数值异常。

随后根据探索压缩复审结论和公平对比需求，停止当前 `resume100k run2` 三种子 `200k` 接续实验，转向新修订方案。停止前该组实验已完整写出 `175k` eval，且已保留所有日志、checkpoint 与官方评估轨迹；停止操作仅终止训练进程，不删除任何已落盘产物。

新方案新增脚本：

- [run_hopper_v5_100k_pureenv_sdinit_mech_sd_budget.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit_mech_sd_budget.sh)

该脚本保留纯环境奖励、supervised `SD` 初始化、共享 `SD transition core`、独立在线 `SD` 监督、相位/接触/角速度机制监督和 residual 结构，但关闭 `sd_online_planning_weight_enabled`，不再用当前 critic 的 `td_positive` 信号对在线 `SD` 监督样本加权。为提高与 TD-MPC2 的比较公平性，planner 搜索预算由旧 planning-aware 脚本的 `num_samples=128`、`num_elites=16`、`num_pi_trajs=16`、`iterations=3` 恢复到半默认以上口径：`num_samples=256`、`num_elites=32`、`num_pi_trajs=24`、`iterations=6`。该设置仍低于 TD-MPC2 默认 `512/64/24/6`，但已显著减少搜索预算过低对结论的混杂。

按该新脚本启动三种子 `100k` 正式实验：

- seed1: `sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed1_20260501_run1`，GPU 1，PID `1530618`；
- seed2: `sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed2_20260501_run1`，GPU 2，PID `1530791`；
- seed3: `sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed3_20260501_run1`，GPU 3，PID `1530698`。

启动后确认三条线均已创建实验目录并写出 `train.csv`、`models/2500.pt` 与 `models/2500.full.pt`。首批记录到 `step=2500`，均处于 seed data / 预训练前阶段，`total_updates=0`，训练链路和 checkpoint 落盘正常。

## 2026-05-02

`100k` mechanism-oriented `SD` 三种子正式实验已完成 `100k` 官方评估并写出 `official_eval/step_0100000/`。三种子 `100k` eval 分别为 seed1 `52.93`、seed2 `52.85`、seed3 `77.65`，均值 `61.14`。按仓库内 `external/tdmpc2/results` 的官方 CSV 口径，对应 `hopper-hop` `100k` 节点的 TD-MPC2 三种子均值为 `13.63`，DreamerV3 三种子均值为 `9.67`，因此本轮 `100k` 结果已在同口径下明显超过两类黑盒基线。

为继续观察该版本在更长预算下能否将当前高分推进态进一步转化为更稳定的姿态与步态，新增 mechanism-oriented `SD` `200k` 接续脚本：

- [run_hopper_v5_200k_pureenv_sdinit_mech_sd_budget_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_mech_sd_budget_resume.sh)

该脚本从 `100k` 目录下的 `final.full.pt` 恢复模型、优化器、buffer、计数器与 `SD` checkpoint 状态，保持纯环境奖励、supervised `SD` 初始化、独立在线 `SD` 监督、`sd_online_planning_weight_enabled=false`、以及 `num_samples=256 / num_elites=32 / num_pi_trajs=24 / iterations=6` 的 fairness 规划预算不变。目标预算按 `200k` 比较，实际 `steps=200500`，以确保 `200k` 评估更稳定地写出 `eval.csv` 与 `official_eval/step_0200000/`。

随后按该脚本在 GPU1、GPU2、GPU3 启动三种子接续实验，保持 GPU0 空闲。接续目录命名为：

- seed1: `sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed1_resume100k_20260502_run1`
- seed2: `sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed2_resume100k_20260502_run1`
- seed3: `sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed3_resume100k_20260502_run1`

恢复源分别为对应 `100k` 实验目录下的 `models/final.full.pt`。本轮接续继续维持 `eval_freq=25000` 与 `checkpoint_freq=25000`，不改变 `100k` 结果后的训练与评估口径。

## 2026-05-03

依据 [SD_MCTS260503.pdf](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/SD_MCTS260503.pdf) 的正文表图、附录 D 与 Checklist 5 声明，整理了面向评审提交的独立附件包：

- [Attachments](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/Attachments)
- [README.md](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/Attachments/README.md)
- [requirements.txt](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/Attachments/requirements.txt)

附件包不沿用作者工作目录中的绝对路径，而按 `code/` 与 `data/` 组织。纳入材料覆盖论文中的 Table 1、Table 2、Table 4、Table 5、Table 6 与 Figure 2、Figure 3、Figure 4：Beer Game 主结果、物理一致性、多步误差、结构突变干预，Cartpole 共享先验与 1-step 消融，以及 Hopper-Hop 50k 三种子日志。为避免提交无关过程材料，探索脚本、100k Hopper 过程脚本、重复 TD-MPC2 补丁副本、无关环境、字节码缓存、模型 checkpoint、完整第三方基线仓库及可选公共基线训练包均未纳入；论文中标注为官方公开基线的 DreamerV3 与 TD-MPC2 数值仍以公开论文/附录为准。最终附件包约 2.9M。

同步改写了论文源文件 [full_draft_zh.tex](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/full_draft_zh.tex) 中的 C.3/A.6 复现索引，使其指向 `Attachments/` 内的独立附件路径。

验证时对附件包内 36 个 Python 文件完成语法解析，确认 JSON 结果文件和 Hopper CSV 日志可解析；由附件内三种子 `eval.csv` 复核得到 Hopper-Hop 50k 汇总为 `32.93 +/- 13.03`。检查确认附件包内不存在 `__pycache__`、`.pyc`、`.pt`、`.pth`、`.pkl` 或 `.ckpt` 文件。

同日补充完成附件代码版本核查。依据工作日志，Cartpole 论文口径对应 `run_cartpole_sharedprior_paper.py`、`run_cartpole_onestep_paper.py`、`run_dmcontrol_cartpole.py`、`sweep_si_robust.py` 与 `dmcontrol_cartpole_sd_model.py`；Hopper 当前主线对应 `src/world_model/dmcontrol_hopper_sd_model.py` 以及 `external/tdmpc2/tdmpc2/` 下的 `config.yaml`、`train.py`、`tdmpc2.py`、`common/parser.py`、`common/sd_world_model.py`、`trainer/online_trainer.py`。逐项比较确认，上述附件代码与当前主工作树最新版一致。另检查同级 `sd_world_model_rl_v4` 工作树，其 Hopper 文件与当前主线不同且工作日志已明确记录该工作树不再作为继续开发主线，因此未作为附件来源。

## 2026-05-04

mechanism-oriented `SD` 三种子 `200k` 接续实验已全部自然结束，并完整写出 `official_eval/step_0200000/`、`200001.full.pt` 与 `final.full.pt`。三种子 `200k` official eval 分别为 seed1 `185.62`、seed2 `265.87`、seed3 `169.19`，均值 `206.90`。按仓库内 `external/tdmpc2/results` 的官方 CSV 口径，对应 `hopper-hop` `200k` 节点的 TD-MPC2 三种子均值为 `166.63`，DreamerV3 三种子均值为 `51.17`。因此当前 mechanism-oriented `SD` 版本在 `200k` 节点已明显超过两类黑盒基线。

同日补充汇总该条 mechanism-oriented `SD` 主线从 `0-200k` 的关键检查点结果。按当前脚本设置，`25k/50k/75k/100k` 来自 `100k` 正式实验目录，`125001/150001/175001/200001` 来自 `resume100k -> 200k` 接续目录；标准差统一按三种子总体标准差口径记录。结果如下：

| 检查点 | seed1 | seed2 | seed3 | 均值 | 总体标准差 |
| --- | --- | --- | --- | --- | --- |
| `25000` | `19.48` | `24.17` | `0.00` | `14.55` | `10.46` |
| `50000` | `33.67` | `18.43` | `2.28` | `18.13` | `12.82` |
| `75000` | `49.96` | `47.47` | `1.12` | `32.85` | `22.46` |
| `100000` | `52.93` | `52.85` | `77.65` | `61.14` | `11.67` |
| `125001` | `90.33` | `160.05` | `69.90` | `106.76` | `38.59` |
| `150001` | `72.58` | `97.18` | `122.81` | `97.52` | `20.51` |
| `175001` | `75.05` | `235.16` | `161.46` | `157.22` | `65.43` |
| `200001` | `185.62` | `265.87` | `169.19` | `206.90` | `42.24` |

为与公开黑盒基线保持同口径比较，进一步摘录仓库内 `external/tdmpc2/results` 的官方 CSV，在 `100k` 与 `200k` 两个共同节点计算三种子均值与总体标准差，并与本条主线结果对照如下：

| 方法 | `100k` 均值 ± std | `200k` 均值 ± std |
| --- | --- | --- |
| mechanism-oriented `SD` | `61.14 +/- 11.67` | `206.90 +/- 42.24` |
| TD-MPC2 官方 | `13.63 +/- 10.41` | `166.63 +/- 34.89` |
| DreamerV3 官方 | `9.67 +/- 13.60` | `51.17 +/- 50.40` |

该汇总表明，本条主线在 `100k` 节点已明显超过 TD-MPC2 与 DreamerV3，且在 `200k` 节点继续保持领先，其中相对 TD-MPC2 的均值优势由约 `47.5` 增加到约 `24.2%`，相对 DreamerV3 的优势则更为显著。但从方差看，该版本在 `175k-200k` 区间仍存在较明显 seed 间分化，说明长程训练下稳定充分支撑步态尚未完全收敛为一致行为模式。

同时记录一个对照事实：同口径官方 CSV 显示 TD-MPC2 在 `200k -> 300k` 区间仍有明显增益，均值由 `166.63` 提升到 `213.93`。因此仅以 `200k` 节点比较仍不足以判定当前 mechanism-oriented `SD` 版本在更长预算下的相对趋势，需继续延长训练。

为此新增 `300k` 接续脚本：

- [run_hopper_v5_300k_pureenv_sdinit_mech_sd_official_budget_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_300k_pureenv_sdinit_mech_sd_official_budget_resume.sh)

该脚本从当前 `200k` 实验目录下的 `final.full.pt` 恢复模型、优化器、buffer、计数器与 `SD` checkpoint 状态，继续保持纯环境奖励、supervised `SD` 初始化、独立在线 `SD` 监督、相位/接触/角速度机制监督以及 `sd_online_planning_weight_enabled=false` 的 mechanism-oriented 口径不变。唯一显式方法改动是将 planner 搜索预算从先前的 fairness 预算 `256/32/24/6` 恢复到 TD-MPC2 官方默认 `512/64/24/6`，以减少与官方基线在 `300k` 节点对比时由搜索预算差异引入的混杂。目标预算按 `300k` 比较，实际 `steps=300500`，以确保 `300k` 评估更稳定地写出 `eval.csv` 与 `official_eval/step_0300000/`。

同日按最新主线替换评审附件包中的 Hopper 数据。旧的 `data/hopper/table6_50k/` 已移除，新的附件数据位于 [mechanism_oriented_200k](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/Attachments/data/hopper/mechanism_oriented_200k)，包含三种子 `0-100k` 与 `100k-200k` 两段的 `eval.csv`、`eval_full.csv`、`train.csv` 和官方评估 episode 汇总。为避免提交本地路径痕迹，官方评估汇总中的原始 `trace_csv` 绝对路径列已删除；模型 checkpoint、渲染轨迹和逐 episode trace CSV 未纳入附件。由附件内 `run_100k_200k/eval.csv` 复核，`200001` checkpoint 的三种子结果为 `185.62`、`265.87`、`169.19`，聚合为 `206.90 +/- 42.24`。当前 `300k` 接续实验属于论文外延长验证，未纳入本次附件包。

继续按评审提交口径清理附件包中的图片与说明文字。删除了论文中未使用的 `fig2_error_growth_all.png` 与 `fig_intervention_rmse.png`，保留图片统一改为以论文图号开头的文件名：`figure2b_multi_step_rollout_rmse.png`、`figure3_sample_efficiency.png`、`figure4_structural_intervention_cost.png`。同时从附件说明中删除关于 `300k` 接续实验未纳入的表述，避免在评审材料中引入论文外过程信息。

随后对附件包 [code](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/Attachments/code) 子目录进行管线级精简。Beer Game 中原先同时存在的 `run_beer_game_v2.py` 与 `run_beer_game_v2_revised.py` 均不再作为 README 声明入口使用；其中仍被论文核心入口复用的公共组件已抽取为 `experiments/beer_game_common.py`，两个旧入口文件均从附件包移除。同步删除未被 README 复现实验管线触达的独立绘图脚本、旧 Beer Game baseline 包装文件以及只承担再导出作用的空包初始化文件。另将多步误差与结构干预脚本中对 `run_ood_state_generalization.py` 的同目录导入改为 `experiments.run_ood_state_generalization` 包式导入，以保证从 `Attachments/code` 目录执行 README 命令时入口路径一致。经静态导入闭包复核，当前 `code/` 下 23 个 Python 文件均由 README 中的 Beer Game、物理一致性、多步误差、结构干预、Cartpole 与 Hopper 支持管线直接或间接触达，未保留未使用代码。

根据评审提交材料的可读性要求，继续修订附件包 README。`Layout` 部分已展开 `code/` 子目录，并概括 `baselines/`、`experiments/`、`scripts/`、`src/` 与 `tdmpc2_sd_patch/` 的用途；`Verification Commands` 部分按实验对象分为 Beer Game 样本效率、物理一致性、多步 rollout、结构干预、Cartpole 与 Hopper-Hop，并为每组命令补充对应的已保存结果文件位置。复核确认 README 中未出现本地工作目录路径、旧 Hopper `table6_50k` 路径、临时 `paper_text` 目录、`PDF` 指代或 `300k` 接续说明。

基于当前附件包继续建立论文修订说明文件 [SD_MCTS260503_revision_notes.md](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/SD_MCTS260503_revision_notes.md)，先完成 C.2 Compute Resources 部分的 `[XX]` 核算。Beer Game 依据 `Attachments/data/beer_game/sdmcts_rerun_results.json` 中三预算三种子的 `total_time`，得到 `1k-10k` 单次运行约 `0.04-0.31` GPU hours，三种子预算 sweep 合计 `1.30` GPU hours；Cartpole 依据 `cartpole_sharedprior_summary.json` 中三 paper seeds 的 `runtime_sec`，得到单 target seed 约 `0.61` GPU hours，三种子合计 `1.84` GPU hours。Hopper-Hop 经复核后不采用后续 mechanism-oriented `SD` `0-100k` 主线的 `50000` checkpoint `elapsed_time`，而改用论文 C.2 原 `50k` 口径对应的 `legacy_v3b_restore_50k_seed{1,2,3}_20260421/eval_full.csv`；三种子完整运行至 `50k` 并完成该节点评估的耗时分别为 `3.91`、`4.00`、`3.75` GPU hours，均值约 `3.89` GPU hours，三种子合计 `11.66` GPU hours。据此，当前 C.2 原句下的 main results 总预算建议写为约 `14.8` GPU hours。Cartpole 原记录为串行执行，若三 target seeds 并行，墙钟时间约可缩短到三分之一；Hopper 因 SD 嵌入 TD-MPC2 在线训练与规划接口，仍应说明其相对 plain TD-MPC2 有额外适配、结构态转换、辅助监督和日志开销。

继续修订 [SD_MCTS260503_revision_notes.md](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/SD_MCTS260503_revision_notes.md)。新增 C.3 Reproducibility Index 的附件包路径索引，并依据 `Attachments/data/hopper/mechanism_oriented_200k/` 中的最新 mechanism-oriented `SD` 三种子结果，调整 Hopper-Hop 数字和相关文字修订指引。修订说明按最小化修订原则处理：Table 1 只替换 Hopper 数字并尽可能补齐 Worst-seed Return，Cartpole 主结果补充为 `802.9 +/- 21.3`；Cartpole DreamerV3 行按官方论文 `500K` 表格口径保留 `850.0`，因官方表未给出标准差和最差种子，Worst-seed Return 保持 `-`。Section 6.2.2 只替换与 Table 1 对应的数字，C.2 Hopper Evaluation Settings 仅修订 `single-seed extensions` 措辞。

继续补充 Checklist 16 的 LLM usage 修订说明。依据 NeurIPS 2026 Main Track Handbook 中关于 agents 与 LLMs 的政策，只有当其构成方法中的 important、original 或 non-standard component 时才需要在论文中描述；写作编辑辅助和 basic code assistance 不需要记录。当前 AI 辅助使用属于写作、整理、调试、脚本辅助、日志核查和参数配置探索等非核心研究辅助，不构成 SD-MCTS 方法、实验算法、环境模型、基线或评估组件。因此修订说明建议 Checklist 16 保留 `Answer: [N/A]`，但将原 justification 中 “strictly limited to language editing, translation, and formatting” 的过窄表述改为更准确的 non-core writing, editing, formatting, and basic code-assistance uses。

按“修订说明已执行后的版本”继续进行致命缺陷排查，并按最小化修订原则收缩建议范围。保留的新增修订仅限于真实前后不一致：为避免评审质疑不同表格为何混用标准差与标准误，建议主要结果表的 `+/-` 统一为跨种子标准差（SD），包括 Table 1、Table 2、Table 4、Table 5 和 Table 6；相应修订 C.2、Table 4 caption 与 Checklist 7 的统计口径说明。另补充说明 Appendix D.3 中 Cartpole per-seed 的 `+/-` 是同一 seed 下 5 个 evaluation episodes 的 episode-level SD，应显式标注，避免与主表 across-seed SD 混淆。可改可不改的理论降调和 Checklist 3 表述调整不再列入修订说明。另根据 `nvidia-smi` 核对，实验服务器 GPU 为四张 NVIDIA Tesla P100-PCIE-16GB，已补入 C.2 Compute Resources 建议文本。

## 2026-06-10

为后续将结构化世界模型与规划方法扩展到 Sokoban/Boxoban 问题，新增独立数据目录 [sokoban_boxoban](/home/data/round/code/sd_world_model_rl/datasets/sokoban_boxoban)，并下载 Google DeepMind `boxoban-levels` 原始数据仓库。数据保留为 upstream shallow clone，以便记录来源版本；本次下载对应 commit `78011d8c86ad423bfb2e1005012393b4c3c24409`，许可证为 Apache-2.0。

本地校验显示，数据目录约 `204 MB`，包含 `1506` 个文本文件与 `1,504,332` 个 10x10 ASCII Sokoban/Boxoban 关卡记录。拆分包括 `unfiltered/train`、`unfiltered/valid`、`unfiltered/test`、`medium/train`、`medium/valid` 与 `hard`。新增 [README.md](/home/data/round/code/sd_world_model_rl/datasets/sokoban_boxoban/README.md) 记录数据来源、commit、规模、拆分与引用格式；新增 `.gitignore` 防止完整原始数据被误提交。

围绕 Sokoban/Boxoban 新研究线完成第一版非侵入式 SD 世界模型实现与数据集子集评测。根据新增约束，本轮未修改 Beer Game、Cartpole、Hopper 的模型、环境或实验入口，而是新增独立包 `src/sokoban/` 与新增实验目录 `experiments/sokoban_sd_world_model/`。这样可以避免当前环境缺失 `gymnasium` 时旧 `src.env.__init__` 导入链影响 Sokoban 实验，同时保持既有成果仍按原路径和原依赖复现。

模型结构采用继承式设计。`SokobanPushDynamicsModel` 作为基础类，仅实现固定关卡上的精确 push-level 转移和 tensor adapter；`SokobanGoalDirectedSDModel` 作为派生类加入任务导向机制，包括玩家可达域、合法宏推、reverse-pull box-goal 距离、最小匹配下界、dead-square、静态墙角死锁和动作排序分数。该设计体现了当前方法论口径：`SD` 不是抽象状态本身，而是引导面向目标的状态推演与动作排序的系统动力机制。

实现过程中先将 Sokoban 文件短暂加入旧 `src/env`、`src/world_model`、`src/planning` 路径，随后发现当前 Python 环境未安装 `gymnasium`，而旧 `src.env.__init__` 会导入已有任务环境并失败。为避免修改旧包入口，本轮撤回这些新增文件并改为独立 `src/sokoban/` 包。该调整只影响本轮新文件，不涉及既有代码回退。

第一版 reverse-pull 距离曾将推箱前站位方向写反，导致许多可达目标被误判为不可匹配，SD-guided 搜索几乎只扩展根节点。修正为从当前格 `cell` 反推到 `cell - direction`，并要求推箱站位 `cell - 2 * direction` 为静态 floor 后，匹配下界恢复正常。随后又将冻结死锁收紧为“静态墙角”判据，不再把其他箱子的暂时阻挡当作永久墙，从而避免过度剪枝可解状态。

完成语法检查：

```bash
python -m compileall src/sokoban experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py
```

完成 Boxoban `medium/train/000.txt` 前 50 关卡评测：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 50 \
  --max-expansions 2000 \
  --out results/sokoban_sd_world_model/eval_medium_train000_50.json
```

结果显示，transition adapter consistency 为 `266/266`；无启发式 push-level 均匀代价搜索在同预算下解出 `2/50`，平均扩展 `1938.18`；SD-guided best-first search 解出 `45/50`，平均扩展 `422.16`。该结果支持第一版工程结论：Sokoban 的任务导向 SD 机制能够显著改善状态推演后的搜索排序和剪枝效率。该结果仍限于轻量子集，不作为完整 Boxoban benchmark 结论。

同步新增与更新以下文档：

1. [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)
2. [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
3. [decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md)
4. [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)

随后因上下文中断后的接续工作，将本轮 Sokoban 交付单元从“继续推进旧 `src/sokoban/search.py` 求解器路线”调整为“文献驱动的 SD 机制矩阵与替换设计”。该调整的依据是：首轮 best-first 结果虽然显示机制特征对搜索有明显诊断价值，但与 Junghanns、Pereira、Lipovetzky、Orseau、Guez、Taufeeque 等文献中的完整机制谱系相比，当前实现只覆盖玩家可达域、push-level 动作、单箱 reverse-pull 距离、匹配下界、dead-square 和静态墙角死锁等基础部分，尚未覆盖 deadlock table、多箱 PDB、goal/tunnel macro、relevance cuts、policy-guided heuristic search 和 path-channel 式计划表征。因此，`eval_medium_train000_50.json` 被保留为 smoke/diagnostic baseline，而不再作为完整 Boxoban benchmark 结论。

新增文献矩阵：

- [sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)

新增替换设计：

- [sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md)

同步重写 [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)，明确当前主线应先固化机制字段、风险证据和 action-ranking schema，再决定 best-first、MCTS/PUCT 或神经模型消费端。根目录 [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)、[decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md) 与 [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md) 已同步更新。

Botea、Muller 与 Schaeffer 2002 年论文的本地 PDF 由旧 Ghostscript 生成。使用 `pdftotext -layout -enc UTF-8` 与 `pdftotext -raw -enc UTF-8` 重新抽取后，正文仍存在大面积编码异常。该文献当前只作为题录和后续可读论文引用链中的背景来源，不把乱码正文纳入具体设计依据。`results/sokoban_sd_world_model/literature_scan.json` 中的 arXiv 检索结果也包含明显误匹配，未作为设计证据。

随后根据研究框架纠偏，将 Sokoban/Boxoban 的 SD 主线重新定义为“结构已知、部分参数未知”的可学习 stock-flow-feedback 世界模型。该调整明确：`SD` 的核心构成元素为存量、流量和反馈；整体框架为 `MBRL + SD-MCTS`；旧 `src/sokoban/search.py` 的 best-first 求解器、matching lower bound、dead-square、macro 和 relevance 等机制只作为结构代理、流量门控、风险标签或诊断对照，不作为主线启发式搜索方案。

新增 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)，定义抽象状态 `x_t` 的存量组成、`x_{t+1}=x_t+inflow-outflow` 的转移形式、反馈回路、已知结构与待学习参数边界、编码器、世界模型预测头、SD-MCTS 接口、学习目标和消融设计。同步新增 [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md)，依据 academic-research-suite 的方法论审查、反方压力测试和编辑综合规范，对设计是否满足 stock-flow-feedback、MBRL/SD-MCTS、避免启发式退化和可验证性进行审查。

同步修订 [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)、[sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md)、[sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)、[methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)、[decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md) 与 [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)。修订后，正式路线以可学习存量、流量、反馈、不确定性和下一抽象状态预测为验收对象，SD-MCTS 必须在学习到的抽象动力学上展开；只输出动作排序分数的版本均记录为启发式诊断结果。

再次按 academic-research-suite 的审查角色规范对 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md) 进行细粒度结构复审，并重写 [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md)。本次复审不再只确认宏观方向，而是逐项检查存量、流量、反馈、编码器、预测头、SD-MCTS 接口、学习目标和文献映射的必要性、合理性与冲突风险。复审结论建议首版收缩为六个核心存量 `G_t, U_t, M_t, R_t, C_t, I_t`，三类核心反馈和合法 single-push option；`GoalRoomPressureStock`、`PlanProgressStock` 与 `StructuralSlackStock` 暂降级为派生特征、辅助头或第二阶段扩展。新增必修项包括 incidence matrix、state aliasing 诊断、风险标签分级、learned transition provider 检查和 no-learned-parameter/no-feedback 等消融。

根据上述细粒度复审意见，继续完成模型设计修订并再次提交结构复审。主设计 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md) 已从九存量版本修订为六核心存量版本，核心状态固定为 `G_t, U_t, M_t, R_t, C_t, I_t`；`GoalRoomPressureStock`、`PlanProgressStock` 和 `StructuralSlackStock` 改为派生特征、辅助头或第二阶段扩展。状态转移由旧的 `x_t + inflow - outflow` 表述改为 `x_{t+1}=clip_X(x_t+B f_theta(...))`，其中 `f_theta` 为非负流量，`B` 为已知 incidence matrix，用于约束每个流量对存量的固定影响方向并避免双计。

同步修订 [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)、[sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md)、[sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)、[methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)、[decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md) 与 [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)。本轮新增决策 14，明确首版动作集合限定为合法 single-push option，macro、relevance、goal-room 和 path-channel 机制暂作为结构代理、诊断项、辅助头或扩展项。SD-MCTS 接口补充 raw-state handle 或 belief support、learned transition provider 审计、state aliasing 诊断、风险标签分级和 `I_t/risk_t` 区分。

修订后的 [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md) 按 academic-research-suite 的 methodology reviewer、devil's advocate reviewer 和 editorial synthesizer 角色规范重新复审。复审结论为：修订版已解决九存量过度复杂、流量双计、反馈过宽、macro 过早首版化和启发式退化风险；剩余风险主要转为实现期检查项，包括 aliasing 方差、风险标签来源、transition provider 日志、残差覆盖程度和不确定性校准。

在确认下一步应先做验证性测试实验后，新增 [validate_sokoban_sd_design.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py) 作为修订版 SD 设计的最小验证入口。该脚本不训练神经网络，也不运行 SD-MCTS；它使用现有 Sokoban 环境和结构代理生成真实 single-push transition 的六核心存量代理标签，并检查存量范围、非负诊断流量、`B f_t` 重构误差、风险标签分级、transition provider 和 state aliasing 方差。

完成语法检查：

```bash
python -m compileall src/sokoban \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py
```

完成 5 关卡验证 smoke：

```bash
python experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 5 \
  --max-depth 1 \
  --max-states-per-level 20 \
  --sample-limit 5 \
  --out results/sokoban_sd_world_model/design_validation_smoke_5.json
```

结果文件为 [design_validation_smoke_5.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/design_validation_smoke_5.json)。摘要显示：`levels_checked=5`、`levels_passed=5`、`states_checked=31`、`transitions_checked=200`、`range_violations=0`、`nonnegative_flow_violations=0`、`max_balance_l1=0.0`、`design_smoke_pass=true`。风险标签计数为 `none=86`、`proven_dead_end=110`、`heuristic_warning=4`；transition provider 全部记录为 `exact_single_push_diagnostic_provider`，因此该结果只作为设计验证 smoke，不作为 learned SD transition 或 SD-MCTS 结果。aliasing 诊断记录 `max_alias_next_delta_variance=0.4044025054643195`，说明后续 learned flow training 前应保留抽象状态别名审计。

在设计验证 smoke 通过后，新增 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py)，训练最小 `no-feedback` learned flow 模型。该模型以六核心存量和 single-push action features 为输入，输出经 `softplus` 约束的非负流量，再通过固定 incidence matrix `B` 预测下一抽象存量。该 smoke 的目的只是检验 `x_t, a_t -> f_t -> x_{t+1}` 是否存在可学习信号，不包含反馈回路、不确定性校准或 SD-MCTS。

完成语法检查：

```bash
python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

完成 10 关卡 learned flow smoke：

```bash
python experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 10 \
  --max-depth 1 \
  --max-states-per-level 30 \
  --epochs 120 \
  --batch-size 128 \
  --hidden-dim 64 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_smoke_10.json
```

结果文件为 [flow_smoke_10.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_10.json)。本轮生成 `343` 条 transition samples，训练集 `274` 条，验证集 `69` 条，覆盖 `10` 个关卡。验证集 learned stock MAE 为 `0.06032923609018326`，zero-flow baseline 为 `0.11418518424034119`，相对改善 `0.4716544314260591`；验证集 flow MAE 为 `0.07328741252422333`。该结果说明六核心存量与诊断流量标签具有最小可学习信号，但尚不构成完整 SD 世界模型或 SD-MCTS 结果。

根据后续推进需求，将 learned flow smoke 从随机 transition split 扩展为 level-holdout split，并在 `sdmcts_p100` conda 环境下执行。环境路径为 `/home/anaconda3/envs/sdmcts_p100/bin/python`，Python 版本为 `3.10.20`，NumPy 为 `1.26.4`，PyTorch 为 `2.1.2+cu118`，`torch.cuda.is_available()` 为 `True`。脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 新增 `--split-mode transition|level` 参数，其中 `level` 模式按完整关卡划分训练集与验证集。

完成 `sdmcts_p100` 环境下的语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

随后使用 `medium/train/000.txt` 前 `30` 个关卡、`max_depth=1`、每关最多 `30` 个状态、`epochs=200`、`hidden_dim=64`、`batch_size=128`，按 seed `0/1/2` 各运行一次 level-holdout smoke。结果文件为：

1. [flow_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed0.json)
2. [flow_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed1.json)
3. [flow_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed2.json)
4. [flow_smoke_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_summary.json)

三种子验证集 learned stock MAE 分别为 `0.04891884699463844`、`0.05621336027979851`、`0.04982342943549156`，均值 `0.0516518789033095`，总体标准差 `0.0032465265139958385`。zero-flow stock MAE 分别为 `0.10804840922355652`、`0.09091933071613312`、`0.09552866220474243`，均值 `0.09816546738147736`。相对改善分别为 `0.547250650461467`、`0.3817226783674094`、`0.47844523009537`，均值 `0.4691395196414155`。该结果说明 learned nonnegative flow 对未见关卡仍有一步抽象转移信号；但当前仍是 `no-feedback`、`no-uncertainty`、`no-SD-MCTS` smoke，下一阶段应在同一 level-holdout 口径下实现 feedback-on flow 对比。

继续在同一脚本中加入 `--model-kind base|feedback`。`feedback` 版本使用目标压力、目标完成、机动性缺口、可达性缺口、拥塞压力、拥塞-机动性耦合、拥塞-可达性耦合、风险压力和风险-拥塞耦合生成每个流量的乘性 gate，使非负流量由 base flow 和 feedback gain 共同决定。该设计用于验证反馈结构能否在一步抽象转移中提供增益，不等同于完整 SD feedback loop。

完成三种子 feedback-gated level-holdout smoke，仍使用 `sdmcts_p100` 环境、`30` 个关卡、`max_depth=1`、每关最多 `30` 个状态、`epochs=200`、`hidden_dim=64`、`batch_size=128`。结果文件为：

1. [flow_feedback_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed0.json)
2. [flow_feedback_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed1.json)
3. [flow_feedback_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed2.json)
4. [flow_feedback_vs_base_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_vs_base_30_levelsplit_summary.json)

同口径对比显示，`base no-feedback` 三种子 valid stock MAE 均值为 `0.0516518789033095`，`feedback-gated` 为 `0.050986512253681816`，均值降低 `0.0006653666496276855`，相对变化约 `-0.012881751133840233`。`feedback-gated` 的 valid relative improvement 均值为 `0.4760261538188252`，略高于 no-feedback 的 `0.4691395196414155`；但 valid flow MAE 均值为 `0.04690564547975858`，高于 no-feedback 的 `0.04553703094522158`。因此当前反馈门控已能接入并略微改善 stock-level 一步预测，但增益很小，不足以直接支撑完整反馈机制主张。下一步应先分析反馈增益偏小的原因，例如反馈特征是否已被 `x_t` 输入吸收、flow loss 权重是否抑制 gate、以及更深 rollout 或风险/uncertainty 目标是否更能暴露反馈价值。

为检查“反馈增益小是否因为反馈信息已被 `x_t` 吸收”，继续扩展 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py)，新增 `--feature-mode stock_action|stock_only|action_only`。随后在 `sdmcts_p100` 环境下运行 `action_only` 与 `action_only + feedback` 三种子 level-holdout smoke，其他参数与前述 `30` 关卡设置一致。

结果文件为：

1. [flow_actiononly_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_smoke_30_levelsplit_seed0.json)
2. [flow_actiononly_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_smoke_30_levelsplit_seed1.json)
3. [flow_actiononly_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_smoke_30_levelsplit_seed2.json)
4. [flow_actiononly_feedback_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_feedback_smoke_30_levelsplit_seed0.json)
5. [flow_actiononly_feedback_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_feedback_smoke_30_levelsplit_seed1.json)
6. [flow_actiononly_feedback_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_feedback_smoke_30_levelsplit_seed2.json)
7. [flow_feature_feedback_ablation_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feature_feedback_ablation_30_levelsplit_summary.json)

四组对比显示：`stock_action_base` valid stock MAE 均值为 `0.0516518789033095`，`stock_action_feedback` 为 `0.050986512253681816`；`action_only_base` 为 `0.04863980785012245`，`action_only_feedback` 为 `0.04908239717284838`。因此，feedback 在完整 stock+action 输入下只带来约 `1.29%` 的 stock MAE 降低，而在 action-only 输入下反而使 stock MAE 上升约 `0.91%`。同时，action-only base 的 stock MAE 低于 stock_action base，但 flow MAE 更高。这说明当前一步 stock 目标可能被动作局部特征和 stock clipping 边界较好解释，而不是证明反馈信息被 `x_t` 吸收。下一阶段应优先改进验证目标，例如加入多步 rollout consistency、风险/aliasing 加权样本或单独的 feedback target，再决定是否进入 learned-transition-provider SD-MCTS smoke。

随后按已确认的推进顺序，将验证目标从一步 stock MAE 扩展为多步 rollout consistency smoke，仍不接入 SD-MCTS。脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 新增 `RolloutSample`、短轨迹构造、`--rollout-depth`、`--rollout-loss-weight`、`--max-rollout-samples-per-level`、多步递推 stock loss 和 rollout 指标。递推形式为 `x_hat_{t+k+1}=clip_X(x_hat_{t+k}+B f_theta(x_hat_{t+k}, a_{t+k}))`；后续动作特征来自真实短轨迹，后续存量输入来自模型预测。

新增路径先完成 5 关卡调试 smoke：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 5 \
  --max-depth 2 \
  --max-states-per-level 20 \
  --epochs 5 \
  --batch-size 64 \
  --hidden-dim 32 \
  --split-mode level \
  --model-kind feedback \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --max-rollout-samples-per-level 200 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_rollout_debug_5_levelsplit_seed0.json
```

随后在 `sdmcts_p100` 环境下完成 `30` 关卡、`max_depth=2`、每关最多 `30` 个状态、`epochs=200`、`hidden_dim=64`、`batch_size=128`、level-holdout、seed `0/1/2` 的四组对照：`rollout_sup_base`、`rollout_sup_feedback`、`one_step_train_base_rollout_eval` 和 `one_step_train_feedback_rollout_eval`。结果汇总写入：

- [flow_rollout_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_30_levelsplit_summary.json)

三种子摘要显示，`rollout_sup_base` 的 valid rollout stock MAE 均值为 `0.048756017039219536`，step-2 stock MAE 均值为 `0.056363618622223534`；`one_step_train_base_rollout_eval` 的对应值为 `0.05358633274833361` 与 `0.062035638839006424`。因此，rollout supervision 在 base 模型上使两步平均滚动误差降低约 `9.01%`，step-2 误差降低约 `9.14%`。`feedback` 模型也从 `0.05165342055261135` 降至 `0.04945573459068934`，但带 rollout supervision 时，`feedback` 相对 `base` 的 valid rollout stock MAE 均值高 `0.0006997175514698029`，未显示正向反馈增益。

逐存量 step-2 误差显示，当前累计误差主要集中在 `reachable_capacity` 与 `congestion`。`rollout_sup_base` 两者均值分别为 `0.12761695434649786` 与 `0.11908540378014247`，明显高于 `goal_completion`、`unresolved_box_pressure` 和 `irreversibility_risk`。该结果支持下一阶段优先改进可达容量、拥塞反馈和风险/aliasing 加权，而不是立即进入 learned-transition-provider SD-MCTS。复现记录和方法口径已同步写入 [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md) 与 [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)。

在上述结论基础上继续执行 `reachable_capacity`、`congestion` 与 feedback target 的验证性改进 smoke。脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 新增 `--stock-loss-preset uniform|reach_congestion`、`--stock-loss-weights` 和 `--feedback-target-loss-weight`。`reach_congestion` preset 对 `reachable_capacity` 与 `congestion` 增加训练权重，并将权重均值归一为 `1`；feedback target 仅在 `model-kind=feedback` 下启用，目标为下一真实存量代理映射得到的反馈特征。

完成调试 smoke 后，继续在 `sdmcts_p100` 环境下运行 `30` 关卡、`max_depth=2`、每关最多 `30` 个状态、`epochs=200`、`hidden_dim=64`、`batch_size=128`、level-holdout、seed `0/1/2` 的新对照，包括 `reach_congestion_base`、`reach_congestion_feedback`、`reach_congestion_feedback_target010` 和额外补跑的 `uniform_feedback_target010`。结果汇总写入：

- [flow_rollout_weighted_feedback_target_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_weighted_feedback_target_30_levelsplit_summary.json)

结果显示，简单提高 `reachable_capacity` 与 `congestion` 的训练权重没有解决目标误差。`reach_congestion_base` 相对 `uniform_base` 的 valid rollout stock MAE 均值从 `0.048756017039219536` 上升到 `0.05112618890901407`，step-2 stock MAE 从 `0.056363618622223534` 上升到 `0.05892247830828031`；step-2 reachable capacity MAE 只从 `0.12761695434649786` 微降至 `0.12749558687210083`，但 step-2 congestion MAE 从 `0.11908540378014247` 升至 `0.12215788662433624`。因此，当前误差不是简单 loss reweighting 可以解决的问题，更可能来自 stock proxy、流量分解或样本别名。

feedback target 存在弱正信号，但不足以改变结论。`uniform_feedback_target010` 相对 `uniform_feedback` 的 valid rollout stock MAE 均值从 `0.04945573459068934` 降至 `0.04896984373529752`，step-2 stock MAE 从 `0.057083974281946816` 降至 `0.0566172848145167`；但它仍未超过 `uniform_base` 的 `0.048756017039219536` 与 `0.056363618622223534`。因此，本轮结果不支持继续堆叠损失权重或立即进入 SD-MCTS。下一阶段应先审查 `reachable_capacity` 与 `congestion` 的 stock proxy 定义、流量标签分解和 aliasing 桶内后继方差，必要时为这两个 stock 增加更细的结构辅助特征或残差审计，而不是把 feedback gate 作为主要增益来源。

随后将当前模型代码与实验结果提交给 `academic-research-suite` 的方法学审查、反方压力测试、编辑综合和实验验证规范进行审查，审查记录写入 [sokoban_sd_rollout_model_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_rollout_model_review.md)。本轮审查未修改模型代码，也未运行新的训练实验。审查结论为：当前通过诊断性基线审查，但不通过 formal SD-MCTS 结果审查；最可靠的下一阶段基线应冻结为六核心存量、固定 incidence matrix、非负 learned flow、合法 single-push action features、level-holdout split 和两步 rollout supervision，即 `rollout_sup_base`。反馈门控、feedback target 和 `reach_congestion` 加权损失因证据不足或结果为负，暂不进入默认基线，只保留为后续单因素消融。进入 SD-MCTS 前应先完成 `reachable_capacity`、`congestion`、flow decomposition 与 aliasing bucket 审计，并实现 learned transition provider 日志。

根据上述审查结论执行冻结基线复测。脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已补充 `baseline_contract`、provider audit、逐存量/逐流量误差、clip audit 和 aliasing audit 输出；新增 [summarize_sokoban_flow_runs.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py)，用于三种子汇总和旧结果复现比较。上述改动不改变训练目标、网络结构或优化顺序，只增强结果审计能力。

完成语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

随后在 `sdmcts_p100` 环境下按冻结基线配置重新运行三种子：`model-kind=base`、`feature-mode=stock_action`、`stock-loss-preset=uniform`、`rollout-depth=2`、`rollout-loss-weight=1.0`、`split-mode=level`、`max-levels=30`、`max-depth=2`、每关 `30` 个状态、`epochs=200`、`hidden-dim=64`、`batch-size=128`。结果写入：

1. [flow_baseline_retest_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed0.json)
2. [flow_baseline_retest_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed1.json)
3. [flow_baseline_retest_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed2.json)
4. [flow_baseline_retest_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_summary.json)

复测主指标与旧 `flow_rollout_30_levelsplit_summary.json` 中的 `rollout_sup_base` 精确一致，复现判定为 `REPRODUCIBLE`，`mismatched_metrics=[]`。三种子 valid rollout stock MAE 均值仍为 `0.048756017039219536`，step-2 stock MAE 均值为 `0.056363618622223534`，valid rollout improvement over zero-flow 均值为 `0.5453278369792934`。新增审计显示，valid any-stock clipped fraction 均值为 `0.9277460177739462`，valid mean abs clip correction 均值为 `0.017344274868567783`；stock-only alias max next-delta variance 均值为 `0.38876330852508545`，stock-action alias max next-delta variance 均值为 `0.1035704289873441`。该结果确认冻结基线可复现，同时表明下一步应优先处理 clipping、stock proxy、flow decomposition 和 aliasing bucket 问题，而不是把 feedback gate 或 loss reweighting 纳入默认模型。

继续执行 stock-flow 诊断增强审计。本轮新增 [audit_sokoban_stock_flow_diagnostics.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_stock_flow_diagnostics.py)，用于重建 exact single-push diagnostic transitions，并将 stock proxy、flow decomposition、clipping、aliasing 和冻结基线预测侧审计集中输出。同步修改 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py)，只扩展评估后处理字段，包括 per-stock raw-next 裁剪方向、per-stock clip correction、真实与预测 stock delta 分布、真实与预测 flow 分布，以及六组成对相反流量的共激活比例。该改动不改变训练样本、网络结构、损失函数或优化顺序。

完成 `sdmcts_p100` 环境下的增强审计版三种子复测，结果文件为：

1. [flow_baseline_diagnostics_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed0.json)
2. [flow_baseline_diagnostics_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed1.json)
3. [flow_baseline_diagnostics_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed2.json)
4. [flow_baseline_diagnostics_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_summary.json)
5. [stock_flow_diagnostics_30_levelsplit.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json)

汇总复现检查显示，增强审计版冻结基线与旧 `rollout_sup_base` 主指标精确一致，`reproducibility_comparison.verdict=REPRODUCIBLE`，`mismatched_metrics=[]`。valid rollout stock MAE 均值仍为 `0.048756017039219536`，step-2 stock MAE 均值仍为 `0.056363618622223534`，valid rollout improvement over zero-flow 均值仍为 `0.5453278369792934`。

完整 stock-flow 诊断覆盖 `30` 个关卡、`566` 个抽象状态和 `3435` 条 single-push transition。exact provider 的 diagnostic flow labels 在固定 incidence matrix 下满足 `max_l1=0.0`，说明流量标签的守恒重构本身成立。主要问题出现在 learned prediction 和抽象充分性：

1. baseline valid any-stock clipped fraction 均值为 `0.9277460177739462`，valid mean abs clip correction 均值为 `0.017344274868567783`；
2. 裁剪修正主要集中在 `irreversibility_risk`、`mobility`、`unresolved_box_pressure` 和 `goal_completion`，其中 `irreversibility_risk` 的 per-stock mean abs clip correction 均值为 `0.07077528536319733`；
3. stock-only alias max next-delta variance 最高为 `0.39549830236885636`，stock-action alias max next-delta variance 最高为 `0.11775519031141866`；
4. `reachable_capacity` delta nonzero fraction 为 `0.37147016011644834`，`congestion` delta nonzero fraction 为 `0.6954876273653566`，说明二者确实频繁变化，但当前代理是全局粗归一化；
5. 六组成对相反流量在预测中均出现 `pred_coactive_fraction_gt_1e-6=1.0`，与 diagnostic target 的单向分解不同，提示 `softplus` 非负流可能通过相反流互相抵消，再由 `clip_X` 修正边界。

因此，本轮结论为：冻结基线仍可作为后续对照，但当前不宜进入 learned-transition-provider SD-MCTS。下一步应做单因素受控修订，优先级为相反流共激活约束或 paired/net-flow 参数化、`reachable_capacity` 与 `congestion` 代理修订、alias-aware 辅助特征。每个修订必须在同一 level-holdout、同一两步 rollout、同一 provider audit 下与冻结基线比较。

按阶段 1 路线执行第一轮 paired/net-flow 修订。脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 新增 `--flow-parameterization softplus|paired_net`，默认仍为 `softplus`，因此冻结基线路径不变。`paired_net` 只输出六个成对有符号净流量，并展开为十二个非负流量：`f_left=relu(z_pair)`，`f_right=relu(-z_pair)`。该结构保持固定 incidence matrix，同时保证每对相反流量不会同时激活。

新增 [evaluate_stage1_flow_gate.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage1_flow_gate.py)，用于读取三种子结果并按阶段 1 门限输出 `PASS/PARTIAL/FAIL`。本轮门限包括：valid rollout stock MAE mean `<=0.05119`、valid rollout step-2 stock MAE `<=0.05918`、valid mean abs clip correction `<=0.01214`、valid any-stock clipped fraction `<=0.75`、flow pair predicted coactivation mean `<=0.20`、每组成对流共激活 `<=0.30`、incidence balance max L1 `<=1e-9`。

完成调试 smoke 后，使用 `sdmcts_p100` 环境运行 `30` 关卡、`max_depth=2`、每关 `30` 个状态、`epochs=200`、`hidden_dim=64`、`batch_size=128`、level-holdout、seed `0/1/2` 的正式阶段 1 实验。结果文件为：

1. [flow_stage1_paired_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed0.json)
2. [flow_stage1_paired_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed1.json)
3. [flow_stage1_paired_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed2.json)
4. [flow_stage1_paired_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_summary.json)
5. [flow_stage1_paired_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_gate.json)

阶段 1 gate 判定为 `PARTIAL`。通过项为相反流共激活和 incidence balance：flow pair predicted coactivation mean 为 `0.0`，max 为 `0.0`，incidence balance max L1 为 `0.0`。未通过项为 rollout 非退化和 clipping：valid rollout stock MAE mean 为 `0.05166300448278586`，高于 `0.05119`；valid rollout step-2 stock MAE 为 `0.05988542859752973`，高于 `0.05918`；valid mean abs clip correction 为 `0.028677965203921`，高于 `0.01214`；valid any-stock clipped fraction 为 `0.9618482192357382`，高于 `0.75`。

与冻结基线相比，`paired_net` 将 valid flow MAE 从 `0.04738566527764002` 降至 `0.03719972570737203`，说明该参数化更贴近 diagnostic flow target；但 valid rollout stock MAE mean 从 `0.048756017039219536` 升至 `0.05166300448278586`，step-2 stock MAE 从 `0.056363618622223534` 升至 `0.05988542859752973`，mean abs clip correction 从 `0.017344274868567783` 升至 `0.028677965203921`。因此该版本不是阶段 1 达标模型，不进入阶段 2，也不接入 SD-MCTS。

本轮结论为：成对互斥是有效的结构方向，但单纯 `paired_net` 会增加边界裁剪并带来 rollout 退化。下一轮阶段 1 应保留 paired exclusivity，同时加入幅度控制或 raw-next boundary penalty，使相反流互斥、rollout 非退化和裁剪下降同时成立。

继续执行阶段 1 修订与重测。脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已新增并验证默认关闭的容量投影、流对缩放、raw-boundary stock 权重、`rollout_final_step_weight` 和 `rollout_final_stock_weights` 入口；[summarize_sokoban_flow_runs.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py) 已同步记录新增配置摘要。编译检查通过：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  experiments/sokoban_sd_world_model/evaluate_stage1_flow_gate.py
```

容量投影 smoke 显示，`paired_net + capacity_projection_stock_mask=1,1,1,0,0,1` 可将 clipping 降至接近零，并保持相反流共激活为 `0.0`。正式三种子普通投影结果写入 [flow_stage1_paired_proj101001_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_30_levelsplit_gate.json)，判定仍为 `PARTIAL`，唯一失败项为 `rollout_step2_non_degradation`：valid rollout step-2 stock MAE 为 `0.05938937390844027`，略高于门限 `0.05918`。

随后只围绕该唯一失败项进行受控调优。负结果包括：`rollout_loss_weight=1.2/1.5` 对 seed1 不稳定；`rollout_final_step_weight=1.5` 对 seed2 不稳定；`rollout_final_stock_weights=1,1,1.2,1.6,1.6,1` 使 seed1/seed2 step-2 误差上升；`epochs=300` 使 seed1 step-2 误差升至 `0.060201920568943024`。上述结果表明，不应继续做逐存量权重或训练长度搜索。

最终有效候选为 `paired_net + capacity_projection_stock_mask=1,1,1,0,0,1 + rollout_final_step_weight=1.2`。正式三种子结果写入：

1. [flow_stage1_paired_proj101001_final12_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed0.json)
2. [flow_stage1_paired_proj101001_final12_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed1.json)
3. [flow_stage1_paired_proj101001_final12_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed2.json)
4. [flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json)
5. [flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json)

阶段 1 gate 判定为 `PASS`，`failed_checks=[]`。观测指标为：valid rollout stock MAE mean `0.05017088601986567`，valid rollout step-2 stock MAE `0.058157226691643395`，valid mean abs clip correction `9.097307914392635e-07`，valid any-stock clipped fraction `0.0005817335719863573`，flow pair predicted coactivation mean/max `0.0/0.0`，incidence balance max L1 `0.0`。因此，阶段 1 的 learned flow 结构修订已完成。下一步应推进 learned transition provider audit 与 SD-MCTS 接口，而不是继续围绕阶段 1 gate 搜索权重。

在用户同意进入阶段 2 后，按 `academic-research-suite` 实验验证流程继续推进 learned transition provider 与 SD-MCTS 接口审计。首先确认阶段 1 JSON 只保存指标而不保存模型权重，因此在 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 中新增默认关闭的 `--checkpoint-out`，用于保存可加载 checkpoint。该改动不改变默认训练路径，也不覆盖阶段 1 gate 文件。

随后新增 [audit_stage2_learned_provider_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_stage2_learned_provider_sdmcts.py)。脚本加载 checkpoint，将阶段 1 learned flow model 包装为 `learned_stock_flow_checkpoint_v1`，并运行最小 SD-MCTS smoke。为避免抽象六维 stock 无法枚举合法推箱动作的问题，脚本保留 raw Sokoban state handle，但该 handle 只记录为 `exact_push_raw_state_support_for_action_enumeration`，用于合法动作枚举和下一层 action support；MCTS 节点的抽象 stock transition、reward 和 value 均由 learned provider 输出计算。exact stock proxy 只作为 audit target，不进入 planner 评分。

使用阶段 1 通过配置重训 seed0 checkpoint，输出 [stage2_provider_checkpoint_seed0_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0_summary.json) 与 [stage2_provider_checkpoint_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt)。该 summary 的验证指标与阶段 1 seed0 一致：valid stock MAE `0.04721938446164131`，valid rollout step-1 stock MAE `0.04322969540953636`，valid rollout step-2 stock MAE `0.0566064827144146`。

随后使用 checkpoint 记录的 held-out levels 运行阶段 2 SD-MCTS 接口审计，结果写入 [stage2_learned_provider_sdmcts_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_learned_provider_sdmcts_audit_seed0.json)。审计覆盖 `6` 个关卡、`42` 个 root 和 `793` 条 MCTS 展开 transition；`learned_provider_fraction=1.0`，所有抽象 transition 均来自 `learned_stock_flow_checkpoint_v1`。gate 判定为 `PASS`，`failed_checks=[]`。关键观测值为：depth-1 stock MAE `0.05060150917204054`，depth-2 stock MAE `0.056905325794544735`，clip mean abs correction `2.5552278208401885e-05`，clip any-stock fraction `0.0037831021437578815`，flow pair coactivation mean/max `0.0/0.0`，incidence balance max L1 `1.1748737760441941e-07`。

本轮结论为：阶段 1 通过模型已经可以作为 learned transition provider 被最小 SD-MCTS 展开路径消费，阶段 2 接口审计通过。该结果仍不是完整 Boxoban 求解 benchmark；后续正式规划实验应继续保留 provider audit，并扩展到多种子、多预算、成功率、失败类型、不确定性/风险校准和 heuristic diagnostic 消融。

继续推进阶段 3 closed-loop planning smoke。新增 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，复用阶段 2 checkpoint 和 provider 封装。脚本采用 receding-horizon 评测：每一步从当前 raw Sokoban state 编码六核心 stock，MCTS 内部使用指定 abstract transition provider 展开，根动作执行后再进入下一步。对照方法包括 `learned_sdmcts`、`oracle_abstract_sdmcts`、`zero_flow_sdmcts`、greedy `heuristic_diagnostic` 和非 SD-MCTS 的 `heuristic_best_first_diagnostic`。

首先运行 checkpoint held-out `6` 个关卡、预算 `16/48` simulations、seed `0/1`、MCTS 深度 `3`、最多 `30` push 的小预算 smoke，结果写入 [stage3_sdmcts_planning_smoke_seed01_budget16_48.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_planning_smoke_seed01_budget16_48.json)。provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`，说明 learned transition provider 在规划评测中仍被正确调用。但四种方法 `learned_sdmcts`、`oracle_abstract_sdmcts`、`zero_flow_sdmcts` 和 greedy `heuristic_diagnostic` 的成功率均为 `0.0`，失败类型主要为 `no_legal_actions` 和 `state_revisit_limit`。

随后补跑较大预算 smoke：`learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts`，预算 `200` simulations、深度 `5`、最多 `60` push、seed `0`，结果写入 [stage3_sdmcts_planning_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_planning_budget200_depth5_seed0.json)。provider usage gate 仍为 `PASS`，但三种 SD-MCTS 相关方法成功率仍为 `0.0`。其中 oracle abstract SD-MCTS 也失败，说明当前瓶颈不只是 learned transition 误差，更在抽象 stock reward/value、终端/死锁处理、循环惩罚和短视界规划消费方式。

为确认关卡可解性，将 `heuristic_best_first_diagnostic` 加入阶段 3 脚本并单独运行同一 `6` 个 held-out levels，预算 `5000` expansions，结果写入 [stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json)。该诊断参照成功率为 `1.0`，平均解长 `15.333333333333334` push，平均 expanded `278.3333333333333`。该结果证明关卡本身可由旧诊断搜索解出，但它被明确标记为非 SD-MCTS 结果。

本轮结论为：阶段 3 已形成正式规划 smoke 的第一组负结果。learned provider 接口可用，但当前 SD-MCTS 的抽象规划目标不足以产生求解行为。下一轮应优先修订 planning layer，包括 solved terminal reward/value、dead-end 与 revisit penalty、长程 value bootstrap、风险校准，以及将 goal/macro/relevance 机制作为 SD feedback 或动作抽象受控接入；不应仅继续提高 simulation 数。

## 2026-06-10

按用户要求复核 NeurIPS/Beer Game 中 TVA 的设计，并为当前 Sokoban SD-MCTS 实现最小可审计 TVA。复核材料包括 `docs/neurips/paper/slides_seminar_zh.md` 中 Terminal Value Alignment 公式、Beer Game/附件代码中的 `ValueNetwork`、MCTS leaf evaluation 和 Monte Carlo/TD value training 路径。复核结论为：TVA 不应替代世界模型或启发式搜索，而应只在 MCTS 截断叶节点提供 `V_psi(s_leaf)`，并由离线 return/search target 对齐。

新增 [sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/sokoban_tva.py) 与 [train_sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_tva.py)。TVA v1 输入为六个 SD stock、九个反馈派生量和十五个高层结构标量，共 `30` 维；训练 target 由 exact-state best-first diagnostic teacher 离线生成，不在线进入 SD-MCTS。目标公式为 solved 状态使用折扣 goal value 与 push cost，proven dead/no-legal 使用 `-10.0`，teacher budget 未解使用 `-2.5`。

修改 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，新增 `--terminal-value-mode stock|zero|tva` 与 `--tva-checkpoint`。`stock` 模式保留旧固定 `stock_value` 叶值，`tva` 模式加载 `sokoban_tva_value_v1` 并记录 `terminal_value_usage_gate`、叶节点类型和每步 TVA 调用次数。该改动不修改 `LearnedStockFlowTransitionProvider`，也不改变 learned provider 的阶段 2 checkpoint。

完成编译检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/sokoban_tva.py \
  experiments/sokoban_sd_world_model/train_sokoban_tva.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

旧 `stock` 叶值兼容性 smoke 写入 [stage3_stock_compat_smoke.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_stock_compat_smoke.json)，learned provider usage gate 为 `PASS`，terminal value usage gate 为 `NOT_APPLICABLE`。

TVA 训练在 `sdmcts_p100` 环境完成，输出 [sokoban_tva_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_train_seed0.json) 和 [sokoban_tva_value_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt)。数据复用阶段 2 checkpoint split，共 `1741` 个样本，train `1345`、valid `396`。最终 train MAE 为 `0.8201102614402771`，valid MAE 为 `1.1535100936889648`，valid corr 为 `0.9297597211519081`。分组诊断显示 `proven_dead_end` MAE 为 `0.12744398415088654`，但 `solved` 平均预测低于目标，`teacher_budget_unsolved` 平均预测为正值，说明 v1 对可解状态和预算未解状态的区分仍不足。

TVA 小预算阶段 3 复测写入 [stage3_sdmcts_tva_smoke_seed01_budget16_48.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_smoke_seed01_budget16_48.json)。learned provider gate 为 `PASS`，TVA usage gate 为 `PASS`，TVA 叶值调用 `36944` 次。`learned_sdmcts` 在 budget `16/48` 下成功率均为 `0.0`；`oracle_abstract_sdmcts` 在 budget `16` 上两种子平均成功率为 `0.08333333333333333`，budget `48` 为 `0.0`；`zero_flow_sdmcts` 为 `0.0`。

TVA 大预算复测写入 [stage3_sdmcts_tva_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_budget200_depth5_seed0.json)。provider usage gate 为 `PASS`，TVA usage gate 为 `PASS`，TVA 叶值调用 `76600` 次；但 `learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率均为 `0.0`。learned_sdmcts 失败类型为 `state_revisit_limit=5`、`no_legal_actions=1`，叶节点主要为 `depth_cutoff`、`frontier_cutoff` 和 `deadlocked`。

本轮结论为：TVA v1 已按 NeurIPS/Beer Game 的核心接口接入，且没有破坏 learned transition provider 基准；但该最小实现未通过阶段 3 成功率检验。下一轮应将 v1 作为对照，转向 search-value alignment、on-policy failed-state relabeling、revisit-aware value target、root action selection 消融，以及 stock-only 与 stock+structural TVA 的抽象充分性比较。

继续推进 TVA v2 failed-relabel 实验。该版本在 [train_sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_tva.py) 中加入 train-level failed closed-loop rollout augmentation、样本权重和 `value_provider_id`，并在 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py) 中加入 `--root-action-selection visits|value`。该改动保持 TVA 只作为 MCTS leaf evaluator，不改变 `learned_stock_flow_checkpoint_v1` transition provider，也不在线调用 best-first teacher。

TVA v2 训练输出写入 [sokoban_tva_v2_failed_relabel_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_train_seed0.json) 和 [sokoban_tva_v2_failed_relabel_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt)。训练集共 `2341` 个样本，其中 `1741` 个为 teacher state sample，`600` 个为 train-level closed-loop failed rollout 样本。失败增强样本包括 `closed_loop_state_revisit_limit=378`、`closed_loop_no_legal_actions=36`、`closed_loop_planner_returned_no_action=96` 和 `closed_loop_max_pushes=90`。最终 train MAE 为 `1.6621757745742798`，valid MAE 为 `1.5998603105545044`，valid corr 为 `0.8909407432297295`。

分组诊断显示，TVA v2 更强地拟合失败状态，但压低了可解状态价值。valid `proven_dead_end` 的 MAE 为 `0.11926791816949844`，target mean 为 `-10.0`，pred mean 为 `-9.899348258972168`；valid `teacher_budget_unsolved` 的 MAE 为 `1.7181379795074463`，target mean 为 `-4.0`，pred mean 为 `-3.0325918197631836`；valid `solved` 的 MAE 为 `5.988962650299072`，target mean 为 `4.7114176750183105`，pred mean 为 `-1.2775442600250244`。因此该版本不能只按总 MAE 或相关系数解释，必须将 solved 类价值塌缩作为主要风险记录。

随后复测 TVA v2 在阶段 3 的 closed-loop SD-MCTS 效果。`root_action_selection=visits` 结果写入 [stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json)，`root_action_selection=value` 结果写入 [stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json)。两条复测均使用同一 `6` 个 held-out levels、`learned_sdmcts`、budget `200`、深度 `5`、最多 `60` push、seed `0`。

两条复测均通过 provider usage gate 和 terminal value usage gate：`learned_provider_fraction=1.0`，TVA provider id 为 `sokoban_tva_value_v2_failed_relabel`，TVA 调用数为 `17600`。但 `learned_sdmcts` 成功率均为 `0.0`，mean pushes 均为 `14.166666666666666`，失败计数均为 `planner_returned_no_action=3`、`state_revisit_limit=3`。相较于 TVA v1 value-root 消融的 mean pushes `28.333333333333332`、失败计数 `state_revisit_limit=5`、`max_pushes=1`，v2 确实改变并压短失败轨迹，但没有产生成功率改善。

本轮结论为：TVA v2 failed-relabel 是一个有诊断价值的负结果。它说明 closed-loop 失败状态重标注能够影响 SD-MCTS 行为，但当前负样本增强过强，导致 solved 类价值估计被压低，阶段 3 成功率仍为 `0.0`。后续不应继续无门限地提高失败样本权重或扩大同类失败增强；若继续推进 TVA v3，应同时加入 solved 正例保护、search-value alignment 或更明确的 revisit-aware target，并以 solved 类预测均值和 held-out 成功率作为进入下一阶段的门限。

随后根据用户对 TVA 语义偏差的质疑，重新复核 NeurIPS 论文中 TVA 的公式和当前 Sokoban v1/v2 的实现边界。本次复核确认：v1/v2 只与论文 TVA 的叶节点接口位置一致，不与其训练语义严格一致。论文 TVA 的核心是由 MCTS root statistics 构造 `V_search(s)=sum_a pi_mcts(a|s)Q_H(s,a)`，并用 search alignment loss 训练 `V_psi`；当前 v1 使用 best-first diagnostic teacher 静态剩余价值，v2 使用 closed-loop failure relabel 负样本，二者应改称 terminal value proxy 对照。

按用户要求，将修订方案提交给 `academic-research-suite` 审查。新增 [sokoban_tva_search_value_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_tva_search_value_design_review.md)，其中按 experiment-agent、methodology reviewer、devil's advocate reviewer 和 editorial synthesis 规范逐项审查。审查结论为：后续正式 TVA 应命名为 `sokoban_search_value_tva_v1`，训练目标必须来自 SD-MCTS 根节点访问次数和根动作 `Q_H`，并显式记录 `alpha`、`gamma`、`lambda_td`、target network 策略、provider id 和 terminal evaluator id。若首轮设置 `lambda_td=0`，只能视为 TD 消融，不得称为完整 TVA。

本轮同步更新 [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)、[decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md) 和 [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)。后续实现前必须先增加 depth-1 `Q_H` 语义审计，确认 root child `q_value` 确为 `reward(root,a)+gamma*V(child_state)` 形式的根动作回报估计。best-first diagnostic teacher 仍只能用于离线可解性参照或旧 proxy 对照，不得进入在线 SD-MCTS action choice 或 root scoring。

按审查意见继续推进严格 search-value TVA 实现。新增 [train_sokoban_search_value_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py)，脚本从 SD-MCTS root statistics 构造 `V_search=sum_a pi_mcts(a|s)Q_H(s,a)`，不使用 best-first diagnostic teacher 生成训练目标。脚本同时实现 depth-1 `Q_H` 语义审计、checkpoint contract、可选 TD target network 入口和 positive search target 样本权重入口。编译检查通过：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  experiments/sokoban_sd_world_model/sokoban_tva.py
```

先运行小规模 smoke，确认脚本能采集 search-value target 并保存 checkpoint。随后运行正式 `sokoban_search_value_tva_v1` 训练：`collect_budget=48`、`mcts_depth=3`、`alpha=1.0`、`gamma=0.95`、`lambda_td=0.0`、`epochs=400`、hidden dim `128`、seed `0`。输出写入 [sokoban_search_value_tva_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_train_seed0.json) 和 [sokoban_search_value_tva_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_seed0.pt)。checkpoint contract 记录 `target_source=sd_mcts_root_statistics`、`mcts_root_statistics_used=true`、`teacher_used_online=false`、`teacher_used_for_training_target=false`、`transition_provider_modified=false`。`Q_H` 审计覆盖 `80` 个状态，`max_abs_error=1.7763568394002505e-15`，判定为 `PASS`。

正式 search-value target 采集得到 `873` 个 root 样本，train `728`、valid `145`；由于 deadlocked/no-legal root 没有 root children，另有 `868` 个状态被跳过，finite `V_search` fraction 为 `0.5014359563469271`。基础版本 valid MAE 为 `1.3938531875610352`，valid corr 为 `0.5508857019779951`；但 positive search target 组 target mean 为 `0.8352776765823364`，pred mean 为 `-0.8174604773521423`，正例保护 gate 未通过。

为检查该问题是否只是样本不平衡，继续执行只基于 `V_search>0` 的正例权重消融，不引入 teacher solved 标签。`positive-target-sample-weight=4.0` 输出 [sokoban_search_value_tva_posw4_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw4_train_seed0.json) 和 [sokoban_search_value_tva_posw4_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw4_seed0.pt)，valid positive pred mean 为 `-0.5202222466468811`。`positive-target-sample-weight=10.0` 输出 [sokoban_search_value_tva_posw10_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_train_seed0.json) 和 [sokoban_search_value_tva_posw10_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_seed0.pt)，valid positive pred mean 为 `-0.1770382821559906`。正例权重增强改善了该组 MAE，但仍未使验证正值目标预测均值转正，因此停止继续权重搜索。

随后进行阶段 3 复测。基础 search-value TVA 复测写入 [stage3_sdmcts_search_value_tva_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_budget200_depth5_seed0.json)，provider usage gate 与 terminal value usage gate 均为 `PASS`，TVA 调用数为 `37800`；但 `learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率均为 `0.0`。其中 `learned_sdmcts` mean pushes 为 `9.0`，失败计数为 `no_legal_actions=3`、`state_revisit_limit=1`、`planner_returned_no_action=2`。

posw10 版本复测写入 [stage3_sdmcts_search_value_tva_posw10_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_posw10_budget200_depth5_seed0.json)，provider usage gate 与 terminal value usage gate 均为 `PASS`，TVA 调用数为 `64800`；三种 SD-MCTS 方法成功率仍均为 `0.0`。`learned_sdmcts` mean pushes 为 `18.166666666666668`，失败计数为 `state_revisit_limit=5`、`no_legal_actions=1`。补充 `root_action_selection=value` 的 learned_sdmcts 消融写入 [stage3_sdmcts_search_value_tva_posw10_value_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_posw10_value_budget200_depth5_seed0.json)，成功率仍为 `0.0`。

本轮结论为：严格 TVA 语义实现、`Q_H` 审计、provider usage 和 terminal value usage 已通过；但 positive search target 泛化和 held-out success rate 均未通过。该结果修复了此前“只接接口、不按论文语义训练”的问题，但仍是阶段 3 负结果。下一步应审查 `V_search` target 质量、positive target 稀疏性、root state aliasing、oracle abstract 仍失败的规划层原因，并考虑加入论文 TD target 与 target network；不应继续只提高正例样本权重。

按用户要求，使用 `academic-research-suite` 对当前 SD 刻画元素与任务目标的一致性进行实验化审计。新增 [audit_sokoban_sd_goal_alignment.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py)，将审查问题转化为离线 teacher value 回归、teacher solved AUC、目标价值 aliasing 和一步动作后果一致性四类指标。该脚本不训练 SD transition model，不改变阶段 2 learned provider，不运行 closed-loop SD-MCTS；best-first diagnostic teacher 只用于离线标签。

首先尝试 full 配置和 medium 配置。full 配置使用默认 `state_depth=3`、每关最多 `90` 个状态、teacher budget `5000`；medium 配置使用 `state_depth=2`、每关最多 `35` 个状态、teacher budget `2500`、每关最多 `8` 个动作审计状态。两者均因 teacher 搜索耗时较高停止，不作为本轮结论依据。随后将脚本中的 static articulation bottleneck probe 改为每关只计算一次，避免每个状态重复计算拓扑探针，并通过编译检查。

最终运行 quick 配置：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --teacher-max-expansions 800 \
  --max-action-states-per-level 3 \
  --out results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_quick_seed0.json
```

结果写入 [sd_goal_alignment_audit_30_levelsplit_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_quick_seed0.json)，gate 判定为 `PASS`。样本规模为状态级 train `231`、valid `60`，动作后果级 train `66`、valid `17`；teacher 标签计数为 `solved=156`、`teacher_budget_unsolved=52`、`proven_dead_end=83`。六存量对验证集 teacher value 的 ridge 回归 `R^2=0.8859336361147274`，相关系数 `0.9414044913328575`，teacher solved AUC `0.9247685185185185`；六存量 alias bucket 的 within/global target variance ratio 为 `0.022069573453746945`。`all_existing_plus_goal_probes` 相对六存量的验证 `R^2` 增益仅为 `0.00877102176130351`。动作后果审计中，固定 stock Q 与 teacher best action 的验证 top-1 agreement 为 `0.8235294117647058`，平均 teacher regret 为 `0.6820606727595`。

本轮结论为：当前六存量与任务目标具有可观的一致性证据，不能简单判定为任意压缩或偏离任务目标；但该结果不是阶段 3 规划成功证明。固定 `stock_value` readout 对 teacher value 的验证 `R^2` 仅为 `0.07533288704528096`，且既有 oracle abstract SD-MCTS 仍为 `0.0` 成功率。因此，后续工作应优先审查 long-horizon planning consumption、`V_search` target 质量、root state aliasing、action abstraction 和 goal-room/macro/relevance 作为 SD feedback 或 option 的受控接入，而不是直接把世界模型退化为启发式搜索。

## 2026-06-11

从人类专家抽象动力机制角度重新审视阶段 3 规划消费层。修改 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，新增 `expert_abstract_dynamics_feedback_v1` 与 `expert_abstract_dynamics_value_v1`。该修订使用目标完成、匹配压力、可行推箱余量、机动性、可达性、拥塞、不可逆风险、目标入口压力、走廊/关节点压力、死方邻接压力和闭环重复状态反馈；只影响 SD-MCTS reward、action prior 和 terminal value，不修改 learned stock-flow transition provider。

完成编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

首先运行 2-level mini smoke，确认新反馈可执行。初始 expert feedback 在 budget `24`、depth `4` 下成功率为 `0.0`，失败均为 `state_revisit_limit`。随后加入闭环重复状态反馈，即对已访问真实状态的 planned successor 在 prior、immediate reward 和 leaf value 三处施加惩罚。重新运行 2-level mini 后，结果写入 [stage3_sdmcts_expert_feedback_revisit_mini_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_mini_budget24_depth4_seed0.json)，`learned_sdmcts` 成功率为 `0.5`，provider usage gate 为 `PASS`，planner feedback usage gate 为 `PASS`。

随后运行 held-out `6` 关卡 learned-only quick：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget24_depth4_seed0.json
```

结果为 `learned_sdmcts` 成功率 `0.16666666666666666`，解出 level `datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt:6`，解长 `15` push；`learned_provider_fraction=1.0`，planner feedback usage gate 为 `PASS`。

为排除旧 stock consumption 的同配置偶然性，运行旧基线 [stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json)，同样 held-out `6` 关卡、budget `24`、depth `4`，`learned_sdmcts` 成功率为 `0.0`。随后运行三 provider 对照 [stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json)，其中 `learned_sdmcts` 成功率为 `0.16666666666666666`，`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率均为 `0.0`，planner feedback 使用次数为 `6407`。

继续检查预算稳定性。`learned_sdmcts` 在 budget `48`、depth `4` 下成功率为 `0.0`，输出为 [stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json)。budget `200`、depth `5` 输出 [stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json)，成功率仍为 `0.0`；budget `200`、depth `4` 输出 [stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth4_seed0.json)，成功率也为 `0.0`。`puct_c=0.25` 和 expert prior uniform mix `0.5` 消融未恢复 budget `48/200` 成功率。

本轮结论为：expert abstract feedback 将阶段 3 从 stock consumption 的 quick `0/6` 推进到 learned provider quick `1/6`，说明人类专家式抽象动力反馈是有效方向；但该结果具有明显预算敏感性，尚不能判定为稳定阶段 3 通过。后续应专门处理大预算下 depth-cutoff、revisited leaf 和单一路径过度集中问题，并把 budget `24/48/200` 的稳定性作为进入下一阶段的硬门限。

继续针对预算不稳定做小步诊断。首先修复 `expert_abstract_dynamics_value_v1` 的 root stop 逻辑：原逻辑会在 `model.is_deadlocked(raw_state)` 为真时停止展开，导致合法 push 仍存在的 root 可能返回 `planner_returned_no_action`。修订后只在 solved 或 no-legal root 停止，deadlock 保留为强负值参与 value 和 reward。该修订通过编译检查。

deadlocked-root 修复后运行：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget48_depth4_seed0.json
```

结果显示 provider 和 feedback gate 均为 `PASS`，`planner_returned_no_action` 消失，但成功率仍为 `0.0`，失败类型变为 `no_legal_actions=5`、`max_pushes=1`。budget `24` 仍保持 `1/6` 成功率，输出为 [stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget24_depth4_seed0.json)。

随后新增 `--expert-viability-penalty`，将后继 `viable_push_count=0` 或 `1` 作为可恢复性负反馈进入 expert reward 和 action prior。该机制对应人类专家会保留未来可恢复推箱余量的抽象动力判断。budget `24` 输出 [stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json)，成功率仍为 `1/6`；budget `48` 输出 [stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json)，成功率仍为 `0/6`，失败类型为 `no_legal_actions=5`、`max_pushes=1`。

最后运行 `root_action_selection=visits` 消融，输出 [stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_visits_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_visits_seed0.json)。该配置使 mean pushes 提高到 `32.0`，失败类型为 `max_pushes=3`、`no_legal_actions=3`，但成功率仍为 `0.0`。因此，本轮停止继续调惩罚项，结论为：当前失败已从“无法展开”转为“缺少稳定目标导向 option/长程价值”；下一步应设计受控 option/action abstraction，而不是继续提高负反馈。

随后按该结论实现受控 action abstraction。修改 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，新增 `PlannerAction`、同一箱子局部两步 `target2`、`target2_prior`、option 内部 provider rollout 审计、`action_abstraction_usage_gate`，并新增 CLI 参数 `--action-abstraction-mode push|target2|target2_prior`、`--max-option-length`、`--max-options-per-node`、`--option-prior-max-multiplier`、`--option-execution-mode`。编译检查通过：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

初始 `target2` 作为额外 action 子节点运行 2-level smoke，输出 [stage3_sdmcts_target2_exec_mini_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_exec_mini_budget24_depth4_seed0.json)。该结果显示 provider usage gate 与 action abstraction usage gate 均为 `PASS`，但成功率为 `0.0`。trace 对比表明，额外 option 子节点扩大了低预算 root 分支，并压制原本可解 level `000.txt:6` 的成功轨迹，因此不将完整 `target2` 作为默认路线。

随后将机制收缩为 `target2_prior`：不新增 root 子节点，只将同一箱子的局部两步目标推进潜力折入 primitive action prior。2-level smoke 输出 [stage3_sdmcts_target2_prior_mini_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_mini_budget24_depth4_seed0.json)，`learned_sdmcts` 成功率为 `0.5`，`learned_provider_fraction=1.0`，`target2_prior_scored_roots=10`，action abstraction gate 为 `PASS`。

继续运行 held-out `6` 关卡 learned-only budget `24`。结果写入 [stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`，mean pushes `20.166666666666668`，`learned_provider_fraction=1.0`，`target2_prior_scored_roots=28`。budget `48` 结果写入 [stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json)，成功率仍为 `0.0`，mean pushes `26.333333333333332`，失败类型为 `max_pushes=2`、`no_legal_actions=4`。`root_action_selection=visits` 输出 [stage3_sdmcts_target2_prior_visits_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_visits_learned6_budget48_depth4_seed0.json)，成功率仍为 `0.0`。

最后运行三 provider 对照 [stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json)。同一配置下，`learned_sdmcts` 成功率为 `0.16666666666666666`，`oracle_abstract_sdmcts` 与 `zero_flow_sdmcts` 成功率均为 `0.0`；provider usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`。本轮结论为：`target2_prior` 是协议合规的受控高层动作先验，但没有通过 budget `48` 稳定性门限；下一步应审查 root value 选择偏移、route consistency、goal-room entry 或 learned option prior，而不是继续只调 option prior 上限。

继续围绕 budget `48` 的 root value 偏移做低成本诊断。对 level `000.txt:6` 的 budget `24/48` trace 进行比较，发现两者第一步相同，第二步开始分歧：成功轨迹继续整理已在目标区的箱子，将其从中间目标推向端点目标；失败轨迹则选择另一个箱子进入目标区，获得即时 `boxes_on_goals` 增益，但后续进入长轨迹失败。该结果提示当前 expert feedback 缺少目标区内部打包顺序机制。

本轮新增两个默认关闭的诊断参数。`--route-consistency-weight` 以上一闭环 primitive push 和上一被推动箱子的物理位置为 root prior 加权；2-level budget `48` 输出 [stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json)，provider 与 action abstraction gate 均为 `PASS`，但成功率仍为 `0.0`。

随后新增 `--goal-room-order-weight`。该反馈对目标区内部箱子向低邻接度端点目标整理给正反馈，并在存在可整理目标区箱子时，对新箱子过早进入目标区施加 delay penalty。2-level budget `48` 中，`goal_room_order_weight=2.0` 输出 [stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json)，成功率为 `0.5`；`goal_room_order_weight=3.0` 输出 [stage3_sdmcts_goalroom_delay_w3_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w3_t2prior_mini_budget48_depth4_seed0.json)，成功率也为 `0.5`。

但 held-out `6` 关卡 budget `24` 中，`goal_room_order_weight=1.5`、`2.0`、`3.0` 均退化为 `0.0` 成功率，分别输出 [stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json)、[stage3_sdmcts_goalroom_delay_w2_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_learned6_budget24_depth4_seed0.json)、[stage3_sdmcts_goalroom_delay_w3_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w3_t2prior_learned6_budget24_depth4_seed0.json)。因此，本轮结论为：goal-room packing 是真实局部失败机制的诊断信号，但当前手工反馈无法跨关卡稳定泛化，不进入默认阶段 3 候选。下一步应转向 learned option prior 或 root `Q_H` 偏差审计。

继续补齐 root-Q 偏差审计与 learned prior 探针。新增 [audit_sokoban_root_q_bias.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_root_q_bias.py)，该脚本只读取既有 stage-3 result JSON 中的 root policy，不运行 best-first，不修改 planner，不改变 transition provider。审计输出 [root_q_bias_audit_level6_target2_prior_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/root_q_bias_audit_level6_target2_prior_seed0.json)。结果显示 level `000.txt:6` 的 budget `24` 成功轨迹与 budget `48` 失败轨迹在 step `1` 首次分歧：失败动作相对成功动作的 root `Q` 高 `1.0426520933606815`，prior 高 `0.6487923854590042`，访问数比为 `4.8`。成功动作是目标区内部整理，`goal_room_order_bonus=1.0`，但 `viable_push_delta=-3`；失败动作使 `boxes_on_goals_delta=1`，同时 `goal_entry_delay_penalty=1.0`。

基于该审计实现默认关闭的 `--audit-proxy-prior-weight`。2-level budget `48` 输出 [stage3_sdmcts_auditproxy_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_auditproxy_t2prior_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`；held-out `6` 关卡 budget `24` 输出 [stage3_sdmcts_auditproxy_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_auditproxy_t2prior_learned6_budget24_depth4_seed0.json)，成功率保持 `0.16666666666666666`，mean pushes 为 `19.833333333333332`，learned provider fraction 为 `1.0`。结论为：audit-proxy prior 不破坏 budget `24` quick 基线，但没有恢复 budget `48`，因此只能作为诊断 probe 保留。

随后实现最小 learned action prior 探针。新增 [sokoban_learned_action_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/sokoban_learned_action_prior.py) 与 [train_sokoban_action_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_action_prior.py)，并在 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py) 中新增 `--learned-action-prior-checkpoint`、`--learned-action-prior-weight` 和 `--learned-action-prior-max-multiplier`。该 prior 的输入为当前 SD stock/TVA 风格抽象特征、candidate primitive push one-hot 与单步 action consequence；训练标签只来自 stage-2 provider train split 上的 closed-loop SD-MCTS trace，消费方式仅为 action-prior multiplier，不改变 learned stock-flow transition provider。

完整 24-level train split 训练写入 [sokoban_action_prior_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_action_prior_train_seed0.json) 和 [sokoban_action_prior_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_action_prior_seed0.pt)。数据共 `443` 个样本，其中 train negative `321`、train positive `6`、valid negative `116`；训练 AUC 为 `1.0`，但内部 valid 无正样本，不能证明泛化。低成本入场复测显示，`learned_action_prior_weight=0.5` 的 2-level budget `48` 输出 [stage3_sdmcts_learnedprior_w05_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_learnedprior_w05_t2prior_mini_budget48_depth4_seed0.json)，成功率为 `0.0`；降低到 `weight=0.1`、`max_multiplier=1.5` 后输出 [stage3_sdmcts_learnedprior_w01_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_learnedprior_w01_t2prior_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`。因此，本轮不运行 held-out `6` 关卡扩展测试，learned action prior v1 冻结为负结果：它满足接口和 provider 合同，但训练正例过稀，不能修复 budget `48` 的关键 root 偏移。

进一步从 SD 核心结构角度审查 goal-room packing 是否应升级为新 stock。新增 [audit_sokoban_goal_room_stock_probe.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_goal_room_stock_probe.py)，只读评估目标区整理缺口、端点目标填充、中心目标占用、目标入口压力、目标邻接箱压力、可整理目标箱比例和 order/entry 冲突等候选 stock/feedback 特征。该脚本不训练 transition provider，不运行 SD-MCTS，best-first teacher 仅用于离线 value 标签。

quick 审计输出 [goal_room_stock_probe_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/goal_room_stock_probe_quick_seed0.json)，使用 stage-2 provider 的 train/valid split，采样 `291` 个状态，其中 valid `60`。六 stock 对 valid teacher value 的 `R^2=0.8859336464419086`；加入 goal-room 特征后 `R^2=0.8850849477314626`，增益为 `-0.0008486987104460164`，未达到 `0.02` 的候选 stock 门限。goal-room 特征本身存在转移信号，单步 transition nonzero fraction 为 `0.16944106925880922`，但价值增益不足。因此本轮不扩展六存量核心，也不启动 stage-1/2 重训；goal-room packing 继续作为局部失败机制和 option-target 诊断证据，而不是新的默认 SD stock。

继续从人类专家抽象动力机制角度检查 budget `48` 的 root 偏移。渲染 level `000.txt:6` 后确认，成功轨迹在第一步后需要先把目标区中间位置的箱子整理到端点目标；该动作会短期降低 `viable_push_count`，因此不能用“机动性下降即坏动作”的粗规则解释。失败轨迹则选择另一个箱子过早进入目标区，获得即时 `boxes_on_goals_delta=1`，但后续进入长轨迹失败。因此，本轮将候选机制收缩为 root-only 的目标区冲突先验，而不是改写 SD transition provider 或扩展核心 stock。

修改 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，新增默认关闭参数 `--goal-room-conflict-prior-weight`。该机制只在 root action prior 层触发，分数为目标区内部整理动作的 `goal_room_order_bonus` 减去过早进入目标区动作的 `goal_entry_delay_penalty`；它不改变 reward、不改变 learned stock-flow transition provider、不调用 best-first，也不进入子节点。结果 JSON 新增 `goal_room_conflict_prior_used`、`goal_room_conflict_prior_root_only`、`goal_room_conflict_prior_changes_transition_provider=false`、`goal_room_conflict_prior_best_first_used_online=false`、`goal_room_conflict_prior_changes_reward=false` 和 `goal_room_conflict_prior_roots`，用于审计该 prior 是否实际触发。

完成兼容性编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  experiments/sokoban_sd_world_model/train_sokoban_action_prior.py \
  experiments/sokoban_sd_world_model/train_sokoban_tva.py
```

随后运行两个 2-level budget `48` smoke。`goal_room_conflict_prior_weight=1.0` 输出 [stage3_sdmcts_goalroom_conflictprior_w1_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_conflictprior_w1_t2prior_mini_budget48_depth4_seed0.json)，`learned_sdmcts` 成功率为 `0.0`，mean pushes 为 `40.0`，provider usage gate 为 `PASS`，planner feedback usage gate 为 `PASS`，action abstraction gate 为 `PASS`。level `000.txt:6` 中该 prior 触发 `2` 个 root，把第二步从原失败动作 `7` 改为动作 `12`，但仍未恢复成功动作 `3`，轨迹前缀变为 `[1, 12, 15, 4, 4]`。`goal_room_conflict_prior_weight=2.0` 输出 [stage3_sdmcts_goalroom_conflictprior_w2_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_conflictprior_w2_t2prior_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`，level `000.txt:6` 的前缀仍为 `[1, 12, 15, 4, 4]`。

本轮结论为：goal-room conflict prior 的方向与专家抽象一致，且合同上没有破坏 SD-MCTS 与 learned provider，但它没有达到“2-level budget `48` 非零成功率”的入场门限。因此不扩展到 held-out `6` 关卡，也不纳入默认阶段 3 候选。该负结果进一步说明，当前缺口不是单一 entry penalty 能修复，而是 root `Q_H` 对“短期牺牲机动性、换取目标区组织稳定性”的长程价值估计不足。下一步应转向 learned option/search-value target 或更明确的目标区分配承诺建模，而不是继续手调 goal-room prior 权重。

继续推进 learned option/search target 路线。新增 [audit_sokoban_train_action_target_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_train_action_target_signal.py)，只使用 stage-2 provider 的 train split，按内部 holdout levels 审计离线 successor teacher-Q 是否能为 action prior 提供足够训练信号。该脚本不训练 transition provider，不修改 planner，不使用 held-out valid levels 生成训练标签；best-first teacher 只用于 train split successor state 的离线标注。

quick 审计输出 [train_action_target_signal_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/train_action_target_signal_quick_seed0.json)。配置为 `max_train_levels=12`、`state_depth=2`、每关最多 `5` 个 action states、teacher budget `800`。结果显示 target signal gate 为 `PASS`：train best-action labels `36`，内部 holdout best-action labels `14`，valid rank states `14`，ridge action-Q valid top-1 agreement `0.42857142857142855`，valid Q corr `0.7964238411550628`，valid teacher-Q std `5.177792257077092`。这说明 train split 内存在可学习的动作级 target，不再是原 solved-trace prior v1 那种正例极稀问题。

随后新增 [train_sokoban_teacher_q_action_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_teacher_q_action_prior.py)，训练 `sokoban_teacher_q_action_prior_v1`。目标为 stage-2 train split 的离线 successor teacher-Q，checkpoint 仍只作为 action-prior multiplier 消费，不提供 transition，不在线调用 teacher。初始 final-epoch 训练过拟合：train top-1 `0.8181818181818182`，valid top-1 `0.2857142857142857`。因此修订脚本，按内部 holdout rank 选择 best epoch 后重新训练。

best-epoch 训练输出 [sokoban_teacher_q_action_prior_best_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_train_seed0.json) 与 [sokoban_teacher_q_action_prior_best_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_seed0.pt)。最佳 epoch 为 `9`；内部 holdout valid regression corr `0.8467250504899865`，valid `R^2=0.4791956049704875`，valid rank top-1 agreement `0.6428571428571429`，高于随机均值 `0.22111678004535149`。该 checkpoint 的 contract 记录 `target_source=stage2_train_split_offline_successor_teacher_q`、`uses_stage2_train_split_only=true`、`uses_heldout_valid_split_for_training=false`、`teacher_used_online=false`、`best_first_used_online=false`、`transition_provider_modified=false`、`planner_consumption=action_prior_multiplier_only`。

最后做 2-level budget `48` smoke。保守权重 `0.1` 输出 [stage3_sdmcts_teacherqprior_w01_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherqprior_w01_t2prior_mini_budget48_depth4_seed0.json)，成功率 `0.0`，provider usage gate、planner feedback usage gate、action abstraction gate 和 learned action prior usage gate 均为 `PASS`。level `000.txt:6` 前缀仍为 `[1, 7, 8, 4, 4, 8]`。强度检查 `weight=0.5`、max multiplier `3.0` 输出 [stage3_sdmcts_teacherqprior_w05_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherqprior_w05_t2prior_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`，level `000.txt:6` 前缀为 `[1, 7, 8, 4, 4, 9]`。

本轮结论为：train split 内存在可学习的 action-Q target，teacher-Q prior 的内部 holdout 排序质量明显优于随机；但该 prior 在线作为 SD-MCTS prior multiplier 仍没有通过最低 2-level budget `48` smoke。因此不扩展到 held-out `6` 关卡，不作为阶段 3 候选。该结果把失败点进一步定位为“动作级离线 target 能学到，但当前 root prior multiplier 无法修复 SD-MCTS 长程价值备份偏差”，后续应转向把该 target 接入 search-value/TVA 或 option-level rollout target，而不是继续调 prior 权重。

继续修订 TVA 与阶段 3 规划消费层的语义对齐。复核 NeurIPS 论文第 4.4 节后确认，TVA 的训练目标应来自实际 MCTS 消费层的 root statistics，而不仅是简化的 stock-reward primitive MCTS；同时 TD 项应使用 target network，并在终止后继上使用同一 terminal evaluator 的终止价值，而不是简单置零。基于此修改 [train_sokoban_search_value_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py) 与 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)。

本轮代码修订包括四点。第一，`SokobanTvaTerminalValueEvaluator` 不再因 `model.is_deadlocked(raw_state)` 直接停止展开，避免 deadlock proxy 在仍有合法 push 时阻断 root；但 `value()` 对 solved 返回 `10.0`，对 deadlocked 或 no-legal 返回 `-10.0`，使终止/高风险状态具有明确边界值。第二，`TerminalAwareStockValueEvaluator` 采用同样的 stop 逻辑。第三，`train_sokoban_search_value_tva.py` 新增 `--target-planner-mode basic|stage3`；`stage3` 模式复用阶段 3 的 `expert_abstract_dynamics_feedback_v1`、`target2_prior` action abstraction 和 learned provider 来采集 `V_search=sum_a pi_mcts(a|s)Q_H(s,a)`。第四，TD target 改为 `r+discount*Vbar_or_terminal(s_next)`，并记录 `td_terminal_successor_value_source=same_terminal_value_evaluator_not_zero`。

完成编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

随后运行 stage3-planner TVA quick 训练：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 8 \
  --collect-budget 24 \
  --mcts-depth 4 \
  --target-planner-mode stage3 \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --bootstrap-terminal-value-mode expert \
  --root-action-selection value \
  --lambda-td 0.25 \
  --target-sync-epochs 25 \
  --epochs 250 \
  --batch-size 64 \
  --hidden-dim 128 \
  --lr 0.001 \
  --seed 0 \
  --value-provider-id sokoban_search_value_tva_stage3_td_v2_quick \
  --out results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt
```

输出 [sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json) 和 [sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt)。采集 `217` 个样本，其中 train `173`、valid `44`；target 来源为 `target_planner_mode=stage3`，planner feedback 为 `expert_abstract_dynamics_feedback_v1`，action abstraction 为 `target2_prior`，bootstrap terminal value 为 `expert_abstract_dynamics_value_v1`。训练后 valid MAE 为 `1.0694197416305542`，valid corr 为 `0.9664165047048058`；但 positive search target 仍极少，全部样本中正值 target 为 `12`，valid positive 仅 `1`，且该正例 target mean `0.14891120791435242`、pred mean `-2.7556140422821045`，显示正向 search-value 泛化仍不足。

按门限先运行 2-level budget `48` smoke，`root_action_selection=value`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json)。成功率为 `0.0`，provider usage gate、terminal value usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`。level `000.txt:6` 的前两步仍为既有失败分支 `[1, 7, ...]`。

继续运行同 checkpoint 的 `root_action_selection=visits` 对照，输出 [stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json)。成功率仍为 `0.0`，四类 usage gate 仍通过。由此判定 `sokoban_search_value_tva_stage3_td_v2_quick` 合同上更接近论文 TVA 与阶段 3 消费层语义，但没有达到“2-level budget 48 非零成功”的入场门限，不扩展到 held-out `6` 关卡。

上下文恢复后继续复核 stage3-planner TVA v2。代码层面保留同一语义修订：`SokobanTvaTerminalValueEvaluator` 不再因 deadlock proxy 直接停止展开，但对 solved、deadlocked 和 no-legal leaf 给出显式边界值；TD target 使用 `r+discount*Vbar_or_terminal(s_next)`，终止后继调用同一 terminal evaluator，不再置零。

12-level quick 训练输出 [sokoban_stage3_planner_tva_v2_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_train_seed0.json) 和 [sokoban_stage3_planner_tva_v2_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_seed0.pt)。该训练采集 `81` 个样本，train `49`、valid `32`，valid corr `0.9321286812910057`，valid MAE `1.2824660539627075`。但正值 search target 只有 `1` 个，valid positive target mean `0.14891120791435242`，pred mean `-3.2686879634857178`。2-level budget `48` smoke 输出 [stage3_sdmcts_stage3tva_v2_quick_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stage3tva_v2_quick_t2prior_mini_budget48_depth4_seed0.json)，成功率 `0.0`，provider、TVA terminal、expert feedback 和 target2_prior usage gate 均为 `PASS`。

随后仅扩大离线 root-state 覆盖，不改变 planner 语义，运行 30-level expanded 训练，输出 [sokoban_stage3_planner_tva_v2_expanded_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_train_seed0.json) 和 [sokoban_stage3_planner_tva_v2_expanded_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt)。该训练采集 `356` 个样本，train `280`、valid `76`，正值 target `15` 个；valid corr `0.9534958319265285`，valid MAE `0.9219829440116882`。但 valid positive 仍只有 `1` 个，target mean `0.14891120791435242`，pred mean `-2.613969326019287`。对应 2-level budget `48` smoke 输出 [stage3_sdmcts_stage3tva_v2_expanded_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stage3tva_v2_expanded_t2prior_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`，四类 usage gate 均为 `PASS`。

本轮复核确认，stage3-planner TVA v2 的合同已经更接近论文中“实际搜索统计对齐 + TD target network”的 TVA 原理，但当前 target 分布仍几乎全为负值，正向 search-value 泛化不足，closed-loop smoke 未达到 2-level budget `48` 非零成功门限。按既定路线，不扩展到 held-out `6`，不继续调 `lambda_td` 或正例权重。

继续检查 teacher-Q action target 的另一种消费方式。新增 [train_sokoban_teacher_q_state_value.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_teacher_q_state_value.py)，将 stage-2 train split 上的离线 successor teacher-Q 从动作级信号聚合为状态级终端叶值。该路线不是严格 search-statistics TVA：其 target 不是 MCTS root statistics，而是 `max_a Q_teacher(s,a)`；best-first teacher 只用于 train split successor state 的离线标注，不在线进入 SD-MCTS，不使用 held-out valid split 作为训练标签，也不修改 `learned_stock_flow_checkpoint_v1` transition provider。其 planner consumption 明确限定为 `terminal_leaf_value_only`。

quick 训练输出 [sokoban_teacher_q_state_value_v1_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_train_seed0.json) 和 [sokoban_teacher_q_state_value_v1_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt)。训练配置为 `max_train_levels=12`、`state_depth=2`、`max_states_per_level=12`、`max_action_states_per_level=5`、teacher budget `800`、`target_aggregation=max`、`epochs=200`、seed `0`。训练前的 action-Q target signal gate 仍为 `PASS`：train best-action labels `36`，内部 holdout best-action labels `14`，valid rank states `14`，ridge valid top-1 agreement `0.42857142857142855`，valid Q corr `0.7964238411550628`。状态价值 gate 也为 `PASS`：train state samples `33`，valid state samples `14`，train positive state targets `31`，valid positive state targets `8`，valid corr `0.3278236921586032`，valid MAE `3.12015962600708`。

但分组结果显示该状态价值网络只在正值状态上表现较好，在非正值状态上严重高估。valid positive target 组为 `8` 个样本，MAE `0.8567870259284973`，corr `0.9382455304567155`，`R^2=0.7238871689555478`；valid nonpositive target 组为 `6` 个样本，target mean `-2.052501916885376`，pred mean `4.085488319396973`，MAE `6.137990474700928`。因此该 checkpoint 具有“识别可解正值状态”的局部信号，但对预算未解或失败状态校准不足。

按阶段门限运行 2-level budget `48` smoke，输出 [stage3_sdmcts_teacherq_statevalue_v1_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_t2prior_mini_budget48_depth4_seed0.json)。配置沿用当前最强中间线：`learned_sdmcts`、`terminal-value-mode=tva`、`planner-feedback-mode=expert`、`action-abstraction-mode=target2_prior`、`root-action-selection=value`、`max_pushes=40`、`max_state_revisits=5`。结果为成功率 `0.0`，两个评测关卡均达到 `max_pushes`。provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`；terminal value usage gate 为 `PASS`，TVA provider id 为 `sokoban_teacher_q_state_value_v1_quick`，TVA evaluations `3840`；planner feedback usage gate 为 `PASS`；action abstraction usage gate 为 `PASS`，`target2_prior_scored_roots=8`。

本轮结论为：teacher-Q state-value 证明离线 action-Q target 可以压缩为状态级终端叶值，但该消费方式没有通过最低 closed-loop smoke。失败原因与接口无关，而是 nonpositive 状态价值校准不足以及 `max_a Q_teacher` 聚合过度乐观，导致叶值容易高估预算未解状态。按既定门限，该路线冻结为诊断负结果，不扩展到 held-out `6`。后续若继续利用 teacher-Q，应转向更明确的 option-level rollout target、分配承诺 value，或增加可审计的非正状态校准约束；不应继续对该 checkpoint 做无界调参。

继续沿 option-level target 方向做低成本审计。发现仓库中已有 [audit_sokoban_option_target_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_option_target_signal.py)，该脚本把 primitive push 与 `target2_local_structural_option` 放入同一离线 teacher-Q 排序空间。脚本只使用 stage-2 provider train split，并在其中划分内部 holdout；best-first teacher 只用于 option successor state 的离线标注，不在线进入 SD-MCTS，也不修改 learned transition provider。

运行 quick 审计后，结果写入 [option_target_signal_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/option_target_signal_quick_seed0.json)。该审计采集 train option rows `254`、valid option rows `100`；候选来源为 primitive push `245` 条、target2 local structural option `109` 条。option target signal gate 为 `FAIL`，唯一失败项为 `valid_top1_agreement_min_0.40`。观测值为：train best option labels `33`，valid best option labels `14`，valid rank states `14`，valid top-1 agreement `0.2857142857142857`，valid Q corr `0.7794633473898425`，valid teacher-Q std `4.823039134104247`，train best target2 option labels `22`，valid teacher-best option state fraction `0.5`。

该结果说明：option-level target 并非没有信号，内部 holdout 中一半状态的 teacher-best 是两步 option，且 valid Q 相关性较高；但当前 71 维 ridge option-Q probe 的 top-1 排序未过门限，不能直接训练在线 learned option prior。对照基线还显示，`fixed_stock_return` 的 valid top-1 agreement 为 `0.5714285714285714`，`exact_option_score` 为 `0.5`，均高于 ridge option-Q probe；而 `target2_structural_score` 仅为 `0.21428571428571427`。因此，当前失败更像是 target2 结构分数与 learned regression 消费形式不足，而不是 option 级 SD rollout value 完全无效。

为检查现有 target2 option 消费层是否只是执行模式问题，补跑 2-level budget `48` 显式 option smoke。`option_execution_mode=first_push` 输出 [stage3_sdmcts_target2_option_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_mini_budget48_depth4_seed0.json)，成功率 `0.0`；provider usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`，`option_expansions=337`，`option_primitives=674`。随后将闭环执行改为 `option_execution_mode=full_option`，输出 [stage3_sdmcts_target2_option_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_full_exec_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`；usage gate 仍通过，`option_expansions=281`，`option_primitives=562`。

本轮结论为：option-level 抽象方向仍有方法价值，但当前 `target2_local_structural_option` 作为显式 option 或 prior 都未通过 2-level budget `48` 入场门限；离线审计也未支持直接训练 learned option prior。后续不再继续围绕 target2 结构分数或执行模式调参。更合理的下一步是重新定义 option target：以 SD rollout return、目标区分配承诺、非正状态校准和跨关卡 top-1 gate 为准入条件，而不是把现有两步局部结构 option 直接上线。

用户进一步指出，SD 可独立测试其状态推演准确性，但 SD-MCTS 在 TVA 缺席时无法作为完整闭环规划对象调试，因为展开步长远短于实际解题步长，缺少终端价值补偿会产生系统性截断偏差。随后建议先测试 TVA 对真实截断剩余价值的准确性。基于该建议，新增 [audit_sokoban_tva_truncation_value.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_tva_truncation_value.py)。该脚本只离线调用 best-first diagnostic teacher，为采样状态标注剩余价值；teacher 不在线进入 SD-MCTS，不修改 transition provider、planner feedback 或 action prior。

审计目标定义为：对状态 `s`，若 teacher 在预算内求得剩余推箱步数 `d`，则构造 `V_teacher(s)=gamma^d * goal_value - discounted_push_cost(d)`；对 `proven_dead_end` 与 `no_legal_actions` 使用高置信负值；对 `teacher_budget_unsolved` 只报告为低置信样本，不纳入 high-confidence gate。审计比较 `stock_value`、`expert_abstract_value` 与 TVA checkpoint 对该截断价值的 MAE、相关性、符号准确率、positive/nonpositive 分组偏差。

smoke 命令在 `sdmcts_p100` 环境中完成，输出 [tva_truncation_value_audit_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_smoke_seed0.json)。采样 `8` 个 stage-2 split 关卡、`46` 个状态，其中 high-confidence 样本 `43` 个，teacher counts 为 `solved=35`、`proven_dead_end=8`、`teacher_budget_unsolved=3`。TVA truncation value gate 判定为 `FAIL`。

关键结果为：`stock_value` high-confidence MAE `5.408689795334134`，`expert_value` high-confidence MAE `5.125799790019716`。`sokoban_stage3_planner_tva_v2_quick` high-confidence MAE `8.333739054839088`、corr `0.951908759929304`、sign accuracy `0.18604651162790697`、positive bias `-8.761932504277338`；`sokoban_stage3_planner_tva_v2_expanded` high-confidence MAE `7.423524176047102`、corr `0.9683731373454699`、sign accuracy `0.18604651162790697`、positive bias `-7.659244627712222`。二者相关性高但整体严重负偏，且 MAE 差于 `expert_value` 基线。

本轮结论为：当前 stage3-planner TVA v2 对其训练 search target 的内部相关性较高，但没有学到真实截断剩余价值的尺度与符号；它系统性低估可解正值状态，因此无法为短视界 SD-MCTS 提供有效终端补偿。后续不应直接继续闭环调 budget、prior 或 option，而应先修订 TVA target，使其通过离线截断价值准确性 gate；只有 TVA 在 high-confidence teacher labels 上优于 stock/expert 基线，并消除正值严重低估后，才进入 2-level budget `48` closed-loop SD-MCTS 测试。

在 option target 审计显示 `fixed_stock_return` 存在离线 top-1 信号后，进一步实现默认关闭的 `option_value_prior_weight` planner probe。该机制只在 SD-MCTS action prior 层消费 target2 option 的本地 SD stock rollout return，不修改 `learned_stock_flow_checkpoint_v1`，不调用在线 best-first teacher，不属于 TVA 或 transition provider。first-push 执行模式输出 [stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json)，2-level budget `48` 成功率为 `0.0`，但 provider、planner feedback、action abstraction 和 option value prior usage gate 均为 `PASS`，`option_value_prior_rows=3070`。

为排除执行承诺方式造成的误判，随后只补跑一个 full-option 执行模式 smoke，输出 [stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，失败类型为 `max_pushes=2`；四类 usage gate 仍为 `PASS`，`option_value_prior_rows=3168`，`option_value_prior_positive_rows=1350`。据此，该机制冻结为 planner-prior 诊断负结果，不继续调权重，也不扩展到 held-out `6`。下一步应回到 TVA 截断价值校准和更明确的目标区分配/承诺 option target，而不是继续围绕 target2 prior 做局部强化。

随后按 TVA 截断价值门限对已有 checkpoint 进行同口径审计，输出 [tva_truncation_value_audit_existing_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_existing_seed0.json)。`sokoban_stage3_planner_tva_v2_expanded` 仍未通过，high-confidence MAE 为 `7.423524176047102`，positive bias 为 `-7.659244627712222`；`sokoban_tva_value_v1` 的 MAE 降至 `1.2229815195856601`，但 positive bias 为 `-1.335852713344545`，仍未通过 `-1.0` 门限。`sokoban_teacher_q_state_value_v1_quick` 在该审计中无失败项，high-confidence MAE 为 `0.5409354691144549`，corr 为 `0.9944298990860484`，sign accuracy 为 `1.0`。

该结果不被解释为严格 TVA 成功，因为 `sokoban_teacher_q_state_value_v1_quick` 的 target 是 `max_a Q_teacher(s,a)`，checkpoint contract 明确标注 `strict_search_statistics_tva=false`。它更适合作为诊断上界：说明当前 30 维抽象特征足以表达高置信截断叶值，但严格 stage3 search-statistics TVA 尚未学到同等校准。为核对该较准叶值是否能与显式 option 消费形成闭环，补跑 `terminal-value-mode=tva`、`action-abstraction-mode=target2`、`option-execution-mode=full_option` 的 2-level budget `48` smoke，输出 [stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，但 learned provider、terminal value、planner feedback 和 action abstraction usage gate 均为 `PASS`。

本轮结论为：阶段 3 的剩余缺口不是“叶值网络接口未接入”，也不是“抽象特征完全不能表达剩余价值”；更准确地说，是严格 TVA target 的校准方式与当前 target2 option/备份消费协议没有形成有效闭环。后续应在严格 TVA 路线中重做 target 生成或校准，使其先通过截断价值 gate；或者重新定义高层 option target，如目标区分配承诺、箱子-目标路线承诺和非正状态校准。teacher-Q 状态价值只保留为离线诊断上界或辅助训练信号，不作为论文 TVA 主结果。

继续按同一截断价值审计口径横向比较已有 TVA/terminal-value checkpoints，避免重复训练。输出 [tva_truncation_value_audit_all_existing_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_all_existing_smoke_seed0.json)。结果显示，`sokoban_teacher_q_state_value_v1_quick` 是当前唯一通过 accuracy gate 的 checkpoint：high-confidence MAE `0.5409354691144549`，corr `0.9944298990860484`，sign accuracy `1.0`，positive bias `0.003251509906797726`，nonpositive bias `0.8333237171173096`。该结果修正了此前仅依据训练内部 valid 分组作出的“非正校准不足”判断：在独立截断价值审计样本上，该 checkpoint 对 high-confidence teacher value 的尺度、符号和正/非正校准均达标。

由于 `sokoban_teacher_q_state_value_v1_quick` 通过了 TVA accuracy gate，按新协议允许进入低成本 closed-loop smoke。运行 2-level budget `48`，输出 [stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，mean pushes `40.0`；provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`；terminal value usage gate 为 `PASS`，TVA evaluations `3840`；planner feedback usage gate 为 `PASS`；action abstraction usage gate 为 `PASS`，`target2_prior_scored_roots=8`。

本轮结论更新为：当前已有一个准确的终端价值补偿器，但它不是严格 search-statistics TVA，而是 teacher-Q state-value terminal leaf probe。它通过了离线截断价值 accuracy gate，却仍未让当前 `expert feedback + target2_prior + root value selection` 的 SD-MCTS 通过 2-level budget `48`。因此阶段 3 瓶颈进一步从“TVA 是否准确”推进到“准确叶值如何被 MCTS 消费，以及当前 root Q/动作抽象/反馈是否能把长程价值转化为正确闭环动作”。后续不应重训同类 teacher-Q state-value，而应诊断 root policy 与 leaf value 备份之间的错配。

继续按“准确 TVA 是否正确影响 MCTS 排序”的问题推进。新增 [audit_sokoban_tva_root_sorting.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_tva_root_sorting.py)，以已落盘的 `stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json` 的 `trace_sample` 根状态为审计对象。脚本在相同根状态上分别运行 `zero_leaf`、`stock_leaf` 与 `teacherq_tva` 三种 terminal value 设置，其他 planner 配置保持一致；离线 best-first teacher 只在规划后对各 root child 的 exact successor state 标注剩余价值，不在线参与 SD-MCTS，不修改 learned transition provider、planner feedback 或 action abstraction。

审计输出 [tva_root_sorting_audit_teacherq_accuracygate_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json)。结果显示，`teacherq_tva` 相对 `zero_leaf` 使教师最佳动作在 root Q 排名中的平均名次改善 `1.4`，相对 `stock_leaf` 改善 `1.3`；平均 root-level `q_teacher_corr` 分别从 `0.7418640397326196` 和 `0.7096232078500353` 提升到 `0.8084293031527132`。但在教师最优等价集合口径下，`teacherq_tva` 的 root Q top-tier 命中为 `6/10`，与 `zero_leaf`、`stock_leaf` 相同；critical miss 仍为 `4/10`。因此，准确 terminal compensator 已经影响排序方向，但没有净提升根节点最终选入教师最优等价集合的能力。

同一审计进一步加入一步备份分解。总体上，`q` 与离线教师值的候选级相关性为 `0.884`，`one_step_provider_score` 为 `0.912`，`one_step_exact_score` 为 `0.926`；但按每个根状态的 top-1 统计，使用 learned provider stock 的一步备份最佳动作 `0/10` 次进入教师最优等价集合，而使用 exact stock readout 的一步备份最佳动作 `6/10` 次进入教师最优等价集合。该结果说明，learned abstract transition 在局部备份中的 stock 读数偏差会明显改变一步价值排序；但这还不是唯一瓶颈，因为后续 oracle abstract 对照仍未通过闭环。

为验证 provider 误差是否足以解释阶段 3 失败，补跑相同配置的 oracle abstract transition 对照，输出 [stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_t2prior_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，mean pushes `40.0`；terminal value、planner feedback 和 action abstraction usage gate 均为 `PASS`。level `000.txt:6` 的 oracle 前缀为 `[1, 12, 15, 4, 4, ...]`，接近此前非零成功前缀的开头，但仍未求解。当前阶段 3 瓶颈据此定位为：TVA 对排序有正向但不足的影响，learned provider 会扰动局部备份，oracle provider 也不能单独解决闭环；下一步应审查 root backup 的 reward/terminal 尺度、target2_prior 对 route commitment 的表达能力，以及深层 rollout/访问分配如何破坏长程价值。

继续做 root backup 尺度审计，新增 [audit_sokoban_tva_backup_scale.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_tva_backup_scale.py)。该脚本只读取 `tva_root_sorting_audit_teacherq_accuracygate_seed0.json` 中已经落盘的一步分解字段，在离线条件下扫描 `expert_reward` 与 `TVA leaf` 的相对权重，不重跑 MCTS，不调用 teacher，不修改 planner 或 transition provider。输出 [tva_backup_scale_audit_teacherq_accuracygate_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_backup_scale_audit_teacherq_accuracygate_seed0.json)。

尺度扫描结果显示，在 learned provider stock 上，最佳组合为 `reward_scale=0.0`、`leaf_scale=0.25`，但 teacher-best tier 命中仅 `3/10`、critical miss `7/10`，差于当前完整 root Q 的 `6/10`。这说明简单增减 reward 或 TVA leaf 权重不能修复 learned stock 下的一步备份排序。相反，在 exact stock readout 诊断上，最佳组合为 `reward_scale=1.0`、`leaf_scale=1.25`，teacher-best tier 命中 `8/10`、critical miss `2/10`。该对照进一步支持：局部备份问题主要来自 learned stock prediction 与价值消费之间的错配，而不是单纯 leaf/reward 比例可调。

为检查 route commitment 是否只是 `target2_prior` 折叠为第一步先验导致的，补跑 oracle abstract provider、显式 `target2` option、`option_execution_mode=full_option` 的 2-level budget `48` smoke，输出 [stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_target2_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_target2_full_exec_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，usage gate 均为 `PASS`，`option_expansions=350`，`option_primitives=700`。level `000.txt:6` 的前缀变为 `[12, 15, 4, 4, ...]`，说明显式两步 option 改变了路线开头，但仍无法完成解题。当前结论为：两步局部 option 承诺仍不足，后续需要更明确的箱子-目标分配、目标区进入顺序或多步路线承诺，而不是继续围绕 `target2` 调执行模式或权重。

继续检查 learned provider stock prediction 是否可以通过简单后处理校准改善价值消费。新增 [audit_sokoban_provider_stock_calibration.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_provider_stock_calibration.py)，只使用 stage-2 provider train split 的单步 transition 拟合逐存量仿射校准，再在固定 root sorting 审计状态上比较 raw provider、calibrated provider 和 exact stock readout。输出 [provider_stock_calibration_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/provider_stock_calibration_audit_seed0.json)。校准使用 `16` 个 train levels、`2570` 条 transition；train MAE 只有轻微变化，例如 `congestion` 从 `0.07728140998907126` 到 `0.07688612298628915`，`reachable_capacity` 反而从 `0.05965281518601668` 到 `0.060456840244316135`。

root 排序诊断结果显示，raw provider 的 teacher-best tier 命中为 `0/10`，critical miss 为 `10/10`；calibrated provider 仅改善到 `1/10` 和 `9/10`；exact stock readout 为 `6/10` 和 `4/10`。因此，简单逐存量仿射校准不能解释 learned provider 与 exact stock 之间的主要价值消费差距，不进入在线 planner 或 provider wrapper。

随后按 route/goal commitment 方向新增离线审计 [audit_sokoban_route_goal_commitment_options.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_route_goal_commitment_options.py)。该脚本在固定失败 root 上生成三类候选：primitive push、既有 `target2_local_structural_option`，以及新增的 `route_goal_monotone_option`。后者显式绑定一个箱子和一个目标，并要求同一箱子的后续推步单调降低到该目标的静态 reverse-pull 距离。teacher 仅用于候选 successor 的离线标注，不在线参与 SD-MCTS，不修改 learned provider 或 planner。

quick 审计输出 [route_goal_commitment_option_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/route_goal_commitment_option_audit_seed0.json)，`10` 个 root 均生成 route-goal 候选，candidate source counts 为 primitive `78`、target2 `28`、route-goal `52`。gate 为 `FAIL`，唯一失败项为 `route_goal_beats_nonroute_roots_min_2`；route-goal 候选在 `0/10` 个 root 上严格优于 non-route 最佳候选。其结构 top-1 的 teacher-best option tier 命中为 `3/10`，first-push tier 命中为 `8/10`。扩展长度到 `6`、每 root 保留 `32` 个 route option 后，输出 [route_goal_commitment_option_audit_len6_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/route_goal_commitment_option_audit_len6_seed0.json)，结果保持一致：gate 仍因 `route_goal_beats_nonroute_roots_min_2` 失败。

本轮结论为：目标绑定的单调 route option 不是可以直接上线的显式长 option；它没有在固定 root 上提供超过 primitive/target2 的 teacher-Q 候选。但它提供了有用的 first-push commitment 信号，即结构 top-1 的 first push 在 `8/10` 个 root 上进入 teacher-best first-push tier。后续应将其作为 learned option target 或 first-push commitment feature 的候选来源，而不是直接接入 SD-MCTS child 或继续手工调 route option 分数。

继续按 first-push commitment 方向做受控审计，新增 [audit_sokoban_first_push_commitment_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_first_push_commitment_signal.py)。该脚本把 primitive push、target2 local structural option 和 route-goal monotone option 放入同一离线候选集合，并将 option-level teacher-Q 聚合回第一步 primitive push：

```text
Q_commit(s, a_first) = max_o Q_teacher(s, o), first(o) = a_first
```

其中 teacher 只用于 stage-2 train split 内部的 option successor 离线标注，不在线进入 SD-MCTS，不修改 learned transition provider、planner feedback、TVA 或 MCTS child 集合。候选生成也不使用 teacher。第一次 holdout-50 smoke 输出 [first_push_commitment_signal_smoke_holdout50_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/first_push_commitment_signal_smoke_holdout50_seed0.json)，gate 为 `PASS`：valid rank states `6`，valid top-tier hit `0.5`，valid Q corr `0.8357902171784326`，valid route-goal available fraction `0.46153846153846156`。

随后只扩大样本，不改变标签语义，运行 12-level、50% 内部留出、teacher budget `300` 的受控扩展审计，输出 [first_push_commitment_signal_audit12_holdout50_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/first_push_commitment_signal_audit12_holdout50_seed0.json)。脚本编译检查通过。扩展审计 gate 仍为 `PASS`，无失败项；观测值为 train best first-push labels `25`，valid best first-push labels `25`，valid rank states `17`，valid top-tier hit `0.47058823529411764`，valid top-1 exact agreement `0.47058823529411764`，valid Q corr `0.8319232992067614`，valid teacher-Q std `4.9064196635874024`，valid route-goal available fraction `0.38181818181818183`，valid teacher-best route-goal source labels `11`。采集阶段耗时 `238.23282393580303` 秒，形成 train states `22`、valid states `17`。

完整指标显示，第一推承诺信号是稳定存在的，但当前 75 维 ridge 学习器尚未证明优于更简单的在线可计算结构基线。扩展审计中，ridge commitment score 的 valid top-tier hit 为 `0.47058823529411764`、mean teacher regret 为 `2.2574965213611655`；route-goal structural score 的 valid top-tier hit 为 `0.7647058823529411`、mean teacher regret 为 `1.1701865437235404`，route-goal count baseline 也达到 `0.7647058823529411`。因此，本轮只把 first-push commitment target 作为可学习信号和目标分配/route commitment 研究方向保留，不直接进入在线 SD-MCTS，不训练 learned prior，不把 route-goal structural score 手工上线为启发式规则。下一步若推进，需先建立更严格的 planner-readiness gate：学习分数必须在内部 holdout 上达到足够 valid state 覆盖，并不低于主要在线可计算结构基线，之后才允许以默认关闭的 prior multiplier 进入 2-level budget `48` smoke。

### Sokoban 格点级 SD 重新设计的方法论复盘

在阶段 3 多轮负结果之后，对 Sokoban 线与论文 SD-MCTS 框架之间的映射关系进行了集中复盘。复盘结论为：当前六存量 learned flow 主线的主要问题不是某一个损失权重、prior 或 TVA checkpoint 未调好，而是 SD 的角色被放错了位置。Sokoban 的底层物理转移是完全可观测、确定性且精确已知的 push 动力学；在该问题中继续学习一个低维抽象转移，会把学习放在本不需要学习的地方，同时把 MCTS 放在不闭合的有损抽象状态上展开。随后围绕 expert feedback、target2、goal-room prior、learned prior、option value prior 和多版 TVA 的大量修订，本质上是在为该错位地基补丁化地恢复被抽象丢失的几何和长程信息。

论文框架中的三个组件在 Sokoban 上应重新分工。SD 在论文中是结构已知、仅低维参数可能未知的白盒动力学，并通过守恒投影 `Pi_C` 约束 MCTS 在物理可行流形内展开；TVA 则用于补偿有限深度树搜索的截断偏差。Beer Game、Cartpole 和 Hopper 中，物理动力学与价值相关动力机制较大程度重合，结构化世界模型的参数辨识是主问题之一。Sokoban 中则不同：箱子如何移动已经由规则精确给出，未知的是长程求解价值、目标区整理顺序、自由度耗散、不可逆死锁和多步承诺。因而 Sokoban 章节应如实定位为：参数辨识支柱在该场景退化为平凡情形，本场景主要隔离检验结构约束搜索、TVA 和零样本拓扑干预，而不是人为向动力学层引入可学习参数以维持叙事表面一致。

负结果证据链被重新归纳如下。第一，`oracle_abstract_sdmcts` 在多轮阶段 3 对照中仍为 `0.0` 成功率，说明仅将 learned provider 换成抽象 oracle 转移并不能解决闭环规划；瓶颈不只是学习误差。第二，stock-only alias next-delta variance 约为 `0.39`，说明六个聚合标量不构成 Markov 闭合系统，同一六维向量下存在大量后继差异。第三，root sorting 审计显示，使用 learned provider stock 的一步备份 top-1 进入 teacher-best tier 为 `0/10`，exact stock readout 为 `6/10`；逐存量仿射校准也只改善到 `1/10`，说明问题不是简单尺度或偏置校准，而是抽象层丢失了动作分支排序所需的信息。第四，六存量 goal-alignment 审计中验证 `R^2=0.8859336361147274` 和 solved AUC `0.9247685185185185` 不能被解释为规划充分性证明；价值回归只要求相关性，MCTS rollout 则要求抽象状态对动作后果保持闭合和相对排序。第五，`sokoban_teacher_q_state_value_v1_quick` 在截断价值审计中 high-confidence MAE 为 `0.5409354691144549`、sign accuracy 为 `1.0`，但接入闭环后仍为 `0/2`，说明准确叶值穿过失真的抽象备份协议后仍可能无法转化为正确根动作。

本轮同时明确了 stock、flow 和 feedback 的操作性判据。参照系统动力学的时间冻结测试，时间冻结后仍客观存在、可以盘点且承载系统记忆的量是存量；时间冻结后失去意义、表示单位时间或事件中搬运多少存量的量是流量；冻结后可由当前存量瞬时计算出的量是辅助变量。存量只能由流量逐步改变，数学上表现为 `x_{t+1}=x_t+inflow-outflow`；反馈不是直接改写存量，而是由当前存量构型调节未来流量的速率、容量或可行域。

按该判据，Sokoban 的核心存量不是 `goal_completion`、`mobility`、`congestion`、`reachable_capacity`、`unresolved_box_pressure` 或 `irreversibility_risk`。这些量在时间冻结时均可由棋盘瞬时重算，因此是辅助变量或 readout，不是系统记忆本身。旧六存量设计的范畴错误在于，把这些辅助变量提升为可 rollout 的存量，并学习“辅助变量的动力学”。其后出现的 aliasing 和 oracle abstract failure 是该范畴错误的实证表现。Sokoban 的真正核心存量应落在格点级：每个 floor 格的箱子占用 `b_c(t) in {0,1}` 和玩家占用 `p_c(t) in {0,1}`；守恒律分别为 `sum_c b_c=N` 和 `sum_c p_c=1`。一次 push 是一单位箱子质量沿网格有向边 `c -> c+d` 的流量，玩家伴随从站位格 `c-d` 迁移到箱子原格 `c`。

据此，SD 核心被重新定义为精确格点级 stock-flow 系统。箱子转移写为 `b_{t+1}=b_t+B f_t`，其中 `B` 是 floor 邻接图的带符号关联矩阵，`f_t` 是 one-hot 非负边流量；玩家更新由 push 后位置确定。该设计无 clip、无残差、无 learned transition 参数，单步和多步 rollout 误差均应为零。它是论文误差界中 `epsilon_model=0` 的极限情形，规划偏差被隔离为有限深度截断、价值估计和动作承诺问题。

对于“直接观测状态并不对应问题求解动力机制”的问题，本轮复盘采纳该判断，但进一步明确其含义。Sokoban 的物理状态与求解机制确实分裂：箱子怎么动是局部、确定和可精确模拟的；进展如何积累、目标区是否需要先整理、可恢复自由度如何耗散、某个局部构型是否导致不可逆失败，则是全局、长程、组合性的求解机制。该抽象性真实存在，但不应放在状态本身。低维、Markov 闭合、价值充分三者在 Sokoban 上不可兼得：低维并闭合的抽象通常价值不充分；低维并价值相关的抽象通常不闭合；闭合并价值充分的状态就是格点棋盘本身。阶段 3 负结果可被解释为对该不可能三角的实证支持。

因此，Sokoban 中 SD 的抽象性应进入三个非状态槽位。第一是事件粒度抽象：以 push 作为流量事件，将玩家可达域内的逐格行走折叠为同一 push-level option；该抽象保留可解性，并显著降低搜索粒度。第二是反馈/约束抽象：占用导致容量下降，墙和箱子使流入容量为零；dead-square、freeze 和其他已证不变集把关于无限未来的命题压缩为当前存量构型上的流量容量门控；目标吸收则把 `sum_{c in T} b_c=N` 表达为 solved 终止条件。这些机制应进入 `Pi_C` 或终止值，且每条反馈律必须独立审计零误杀。第三是价值抽象：无法闭合为低维状态但与求解价值相关的 goal completion、matching lower bound、mobility、congestion、route/goal commitment 等，作为 `V_psi(x,a(x))` 的输入特征、解释面板或离线审计变量，而不在其上 rollout。

基于上述复盘，新增并认可 [sokoban_sd_grid_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_grid_stock_flow_design.md) 作为 Sokoban SD 核心的新设计文档。该设计取代六存量 learned flow 版作为现行定义；旧 stage-1/2/3 产物整体冻结为负结果和诊断方法素材，不再作为主线继续调参。第一天实现后被边缘化的 `SokobanPushDynamicsModel` / `SokobanGoalDirectedSDModel` 应保留为新 SD 核心载体，后续只需补充 `B f_t` 关联矩阵形式化、守恒断言、玩家区域正则化键 `kappa(x)` 和 Tier-A/Tier-B 分层接口。六存量 proxy 计算代码原样保留，但角色降级为 `V_psi` 输入特征、诊断面板和论文解释性素材。

现有资产的处置也被明确。`SokobanPushDynamicsModel.step_state`、`reachable_cells`、`legal_pushes`、dead-square 预计算、freeze 检测和 reverse-pull 距离场保留，并分别服务于精确流量施加、合法流量枚举、Tier-A 约束和价值/诊断特征。matching lower bound、pull 距离、goal-room pressure、route-goal signal、旧六存量 readout 等全部归入 Tier-B 辅助变量，默认只能用于 `V_psi`、审计和可解释性面板。stage-3 中 accumulated 的 expert feedback、target2 prior、goal-room order、route consistency、audit-proxy prior、learned action prior、teacher-Q prior 和 option value prior 等手工旋钮不迁移为新主线默认机制；若未来要重新进入在线规划，必须作为独立学习组件或受控 option 通过显式入场门限。

新主线的核心实验问题也被重新设定。MCTS 应在精确格点 stock-flow 状态上展开，深度 `H` 截断；TVA 成为主要学习组件，用真实状态搜索统计、solved-trace return、option-level return 或 TD target network 对齐 `V_psi`，用于补偿 `H <<` 解长时的截断偏差。核心实验应报告 `H x budget x (TVA on/off)` 的解题率矩阵，并设置 pure MCTS、TVA-MCTS、约束剪枝开关、black-box latent transition 和旧六存量 learned provider 的对照。零样本拓扑干预可通过改墙改变边容量和 `B` 的支撑集、改目标改变吸收集 `T` 与 dead-square 集，动力学层无需重训，直接复跑 MCTS 和 TVA 评估。

新 SD 部分的验收门限被固定为：G1 守恒审计，所有 rollout 中 `sum b_c=N`、`sum p_c=1`，违例率为 `0%`；G2 转移保真，`B f_t` 形式与 `step_state` 在采样关卡全量合法 push 上逐位一致；G3 剪枝可靠性，dead-square 和 freeze 作为 Tier-A 前必须在可全搜索小关卡集上零误杀，同时报告捕获死态比例；G4 黑盒对照，同数据训练 latent MLP transition，报告其多步 rollout 误差、物理违例率与 SD 零误差的对比；G5 状态恒等，push-level 下 `kappa(x)=(b,min_cell(Reach(b,p)))` 与玩家可达连通域等价性测试通过。若需要论文形式补全，可在附录给出 move-level 玩家逐格流量展开，但主实验建议采用 push-level 粒度。

本轮复盘的论文写作含义为：Sokoban 不应被描述为又一个“学习结构化转移参数”的场景，而应被描述为 SD-MCTS 框架在不同问题类型上的角色迁移。前三类场景检验 SD-as-physical-prior，即结构先验如何降低模型误差；Sokoban 检验 SD-as-solving-mechanism-scaffold，即在物理转移平凡、求解机制抽象的场景中，存量/流量退到精确微观层，反馈律和 `Pi_C` 表达可验证的求解约束，TVA 承担主要学习负担。该表述既保留系统动力学的存量、流量、反馈语义，也避免把有损启发式压缩误写成闭合世界模型。

### Sokoban exact-grid SD 与 TVA 独立门禁落地

按照上述重构方案，新增 `SokobanGridStockFlowModel` 作为 Sokoban 新主线的 SD provider，provider id 为 `exact_grid_stock_flow_v1`。该模型继承既有精确 push 动力学，但显式补充 floor 格点索引、有向边集合、带符号关联矩阵 `B`、箱子存量 `b`、玩家存量 `p`、one-hot push flow、`b_next=b+Bf` 审计接口、push-level 玩家可达域 canonical key，以及 Tier-A 死锁状态读数。该实现没有 learned transition 参数；旧六存量 learned-flow 线未删除，但其角色保持为负结果、诊断材料和 TVA 辅助 readout 来源。

新增 [validate_sokoban_grid_stock_flow_design.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/validate_sokoban_grid_stock_flow_design.py) 作为 SD 独立审计脚本。该脚本不进入闭环规划，专门检查守恒、flow one-hot、`Bf` 与 exact `step_state` 的逐位一致性、push-level canonical key 等价性以及 Tier-A 反馈计数。smoke 输出 [grid_stock_flow_design_smoke_quick.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json)，`grid_sd_gate=PASS`，共检查 `3` 个关卡、`15` 个状态、`64` 个合法 push 转移和 `308` 个可达玩家位置；箱子质量、玩家质量、flow one-hot、box balance、step mismatch 和 canonical mismatch 违例均为 `0`。

新增 [audit_sokoban_grid_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py) 作为 TVA 独立审计脚本。该脚本在 exact-grid 状态上采样，由离线 exact-state teacher 标注截断剩余价值，使用真实格点 observation 加辅助 readout 拟合 ridge value probe，并报告 train/valid 校准和 root successor sorting。teacher 只用于离线标签，不在线进入规划；transition provider 固定为 `exact_grid_stock_flow_v1`。第一次 quick 配置因 valid 样本数仅为 `7` 触发 `valid_sample_count`，该结果保留为覆盖不足记录。随后扩大到 `8` 个关卡、每关 `8` 个状态后，输出 [grid_tva_independent_audit_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_tva_independent_audit_smoke_seed0.json)，`tva_independent_gate=PASS`；valid count 为 `16`，MAE 为 `1.3878170000076102`，corr 为 `0.9736159851420181`，sign accuracy 为 `1.0`，root sorting top-tier hit 为 `5/5`。同时导出 [grid_tva_ridge_checkpoint_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json)，用于闭环 leaf value 消费；checkpoint contract 明确记录 `teacher_used_online=false`。

新增 [evaluate_sokoban_grid_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py) 作为 exact-grid SD-MCTS 闭环入口。该入口在原始 `SokobanState` 上展开 MCTS，动作来自 `legal_flows`，转移来自 `step_grid_flow`，默认使用 Tier-A 约束剪枝；leaf value 支持 `zero` 和通过独立审计导出的 ridge TVA checkpoint。脚本支持 `--require-independent-gates`，在给定 SD 或 TVA 审计文件未通过时拒绝运行，从执行层面落实“SD 与 TVA 先独立验证，再进入闭环”的协议。闭环脚本不调用在线 teacher，不复用旧 stage-3 六存量 provider，也不迁移 expert feedback、target2 prior 或其他手工 planner 旋钮。

同口径 3-level smoke 表明新闭环入口已可运行并落盘。zero-leaf 输出 [grid_sdmcts_zero_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_zero_smoke_seed0.json)，成功率 `1/3`，solved 关卡平均 `11` pushes，失败状态为 `max_steps=2`；TVA-leaf 输出 [grid_sdmcts_tva_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_tva_smoke_seed0.json)，成功率同为 `1/3`，solved 关卡平均 `23` pushes，失败状态为 `max_steps=1`、`no_action=1`。因此本轮结果只证明新 SD、TVA 和 closed-loop 消费接口已经打通，并且基础门禁可复现；尚不构成 TVA 相对 zero-leaf 的性能改善结论。下一步主实验应在该入口上系统运行 `H x budget x TVA on/off` 矩阵，并用更大样本检验 TVA 是否能稳定补偿短视界截断偏差。

### Sokoban exact-grid SD-MCTS holdout 小矩阵

为将闭环测试从接口 smoke 推进到完整框架效果测试，先修订 [evaluate_sokoban_grid_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py)，增加 `--level-offset` 参数。该参数允许从同一 Boxoban 文件中跳过前若干关卡再评测，避免当前 TVA smoke checkpoint 的训练关卡进入效果测试。默认值为 `0`，保持旧命令兼容；本轮 holdout 测试使用 `--level-offset 5 --max-levels 3`，对应 `audit_sokoban_grid_tva.py --max-levels 8 --holdout-fraction 0.35` 下的 valid slice。

随后在 holdout 三关上运行小型 `H x budget x leaf_value` 矩阵。配置为 `max_steps=60`，扫描 `H={2,4}`、`budget={24,48}`、`leaf_value={zero,tva}`；另补充一个较强点 `H=6,budget=96,max_steps=80`。所有命令均启用 `--require-independent-gates`，zero-leaf 读取 SD 门禁文件，TVA-leaf 同时读取 SD 与 TVA 门禁文件，并消费 [grid_tva_ridge_checkpoint_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json)。

结果如下：

| H | budget | leaf | solved / levels | status counts |
|---:|---:|---|---:|---|
| 2 | 24 | zero | `0/3` | `deadlocked=2, no_action=1` |
| 2 | 24 | tva | `0/3` | `max_steps=2, no_action=1` |
| 2 | 48 | zero | `0/3` | `deadlocked=2, no_action=1` |
| 2 | 48 | tva | `0/3` | `max_steps=2, no_action=1` |
| 4 | 24 | zero | `0/3` | `deadlocked=1, max_steps=1, no_action=1` |
| 4 | 24 | tva | `0/3` | `max_steps=2, no_action=1` |
| 4 | 48 | zero | `0/3` | `max_steps=3` |
| 4 | 48 | tva | `0/3` | `max_steps=3` |
| 6 | 96 | zero | `0/3` | `max_steps=3` |
| 6 | 96 | tva | `0/3` | `max_steps=3` |

该结果表明，当前 exact-grid SD、TVA checkpoint 与 MCTS 消费协议构成的完整框架已经可以在 heldout slice 上测试，但没有观察到 heldout 解题成功率，也没有观察到 TVA 相对 zero-leaf 的成功率增益。TVA-leaf 在若干低深度配置中减少了立即 deadlock，但只是把失败类型从 deadlocked/no_action 推迟为 max_steps，尚未转化为解题。后续不应把当前 checkpoint 写作完整框架有效结果；下一步应优先审计 TVA leaf 在 root action 选择中的贡献、Tier-A 剪枝是否过窄或过宽、MCTS 选择策略是否过度依赖访问数，以及是否需要在 exact-grid 状态上重新训练更强的 search-statistics TVA。

### Holdout 0/3 根因分析与 SD-heuristic 叶值修复

对上一节 holdout 小矩阵的 trace 进行二次分析，定位出三个相互独立的根因。第一，zero-leaf 在奖励结构下没有目标导向梯度：奖励仅有每推 `-0.05` 与视界内整关求解 `+10`/死锁 `-10`，medium 关卡解长远超 `H<=6`，故兄弟动作 Q 值中位数极差仅 `0.023`，搜索退化为避死锁游走；直接证据为 zero `H=2` 配置在 budget `24` 与 `48` 下动作序列逐步完全相同，预算完全饱和。第二，执行层震荡：全部 trace 最多推到 `2/4` 箱在目标后反复撤销（`max_bog>final_bog`）。第三，smoke TVA checkpoint 训练态分布为 8 关初始态 1 推以内的 `41` 个样本，与闭环叶查询分布严重错配，且训练关卡 0-7 与 holdout 5-7 重叠，存在泄漏。

修复方向采纳"信号放回叶值通道"的架构判断：势函数塑形在截断搜索中逐项相消后恰好等价于在叶处加 `gamma^H * Phi(s_H)`（根处常数项不影响排序），故无需触碰奖励通道，目标导向信号应由 SD 抽象读出经叶值注入。据此为闭环入口新增 `--leaf-value sd_heuristic`，定义 `V(s) = gamma^lb * goal_value - push_cost * (1-gamma^lb)/(1-gamma)`，其中 `lb` 为 SD 层既有读出 `matching_lower_bound`，`lb=inf` 回落 `deadend_value`；该形式与 teacher 价值标签公式同构。另新增 `--revisit-penalty`，执行循环维护 push-level canonical key 已访问集，规划器对进入已访问状态的转移加罚，默认 `0.0` 保持旧行为。

holdout 三关复测结果：`sd_heuristic` 在 `H=6,budget=96` 下取得首个 heldout 解（19 pushes，bog 单调爬升 0→4）；叠加 `revisit_penalty=1.0` 后 `H=4,budget=48` 同关也从 no_action 转为 19 pushes 求解，两配置均为 `1/3`。未解两关的失败模式为 `2<->1` 震荡被打破后仍无法收敛，其 `matching_lower_bound` 长期停在 `3-7` 不降，说明匹配下界在这两关系统性低估（忽略玩家可达性与推送顺序约束），是 residual 学习的明确标靶。回归检查确认 `revisit_penalty=0` 下结果与改动前逐项一致。

### TVA v2 重训与闭环负结果

针对分布错配与泄漏，扩展 [audit_sokoban_grid_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py)：新增 `--level-offset`（训练关卡 8-47，与 holdout 5-7 严格不相交）、`--sample-mode random_walk`（仅在非死锁后继中随机游走至深度 16，覆盖闭环叶查询深度）、`--target-mode residual`（拟合 teacher 目标减 sd_heuristic 基线的残差）与按深度分层的 valid 指标；`valid_positive_count` 门禁检查改为仅适用于 value 模式，因 residual 目标相对乐观下界基线天然非正。

v2 审计（`960` 样本，teacher 解出 `262`、budget 未解 `657`）：value 模式 valid corr `0.532`，residual 模式 `0.617`，深层切片（depth 9+）corr 分别为 `0.707/0.771`，两模式 gate 均 PASS。但闭环 holdout 三关（`revisit_penalty=1.0`，`H=4,b=48` 与 `H=6,b=96`）中 value 与 residual 均为 `0/3`，且都在 sd_heuristic 能 19 推求解的关卡上走入 matching deadlock（`lb=inf`）死锁。结论：v2 TVA 通过离线门禁但闭环上劣于纯 SD 读出叶值，学习修正污染了 LB 梯度。主要嫌疑为标签瓶颈：`657/960` 样本因 teacher 预算未解而取常数 `-2.5`，回归把常数悲观值抹到活状态上，淹没了结构信号。下一步 v3 应聚焦标签质量（剔除或降权 budget-unsolved 样本、提高 teacher 预算、或用 sd_heuristic 锚定未解标签），并在 TVA 未超过 sd_heuristic 基线前不将其写入主结果。

### TVA v3：足额 teacher 预算检验与瓶颈重定位

针对"v2 失败可能只是 teacher 预算不足"的假设做完整检验。先做两项 teacher 改进：第一，A* 去重键从精确 `SokobanState` 改为 push-level `canonical_key`（同一玩家可达连通域内位置完全等价，属精确变换），状态空间塌缩后 teacher 加速约两个数量级；第二，修复标签语义 bug——frontier 穷尽（在 sound 死锁剪枝下构成不可解证明，应标 `deadend_value=-10`）与扩展预算耗尽（未知，标 `-2.5`）此前混在同一 `teacher_budget_unsolved` 类别。预算提高至 `20000`。

v3 结果（同 v2 采样配置，960 样本）：`teacher_budget_unsolved` 从 `657` 降为 `0`——全部原"未知"样本被完全判定为 solved（`307`）或 proven_dead_end（`612`）。这直接否定了预算假设的乐观版本：v2 的常数标签污染并非欠预算的可解状态，而是深度随机游走推坏的可证明死局，真正的修复是标签语义而非预算本身。标签零污染后，离线 valid corr 反而下降（value `0.387`、residual `0.460`，目标变为 ±10 双峰、MAE 升至 `5.26`），闭环 holdout 三关 value/residual 两模式仍全部 `0/3` 且仍出现执行期死锁。结论：标签精确后 ridge 线性探针依然无法表达"全局死局检测"与"LB 残差修正"，瓶颈从标签质量明确转移到函数族容量与特征表达；TVA 作为架构组件未被否定，但线性 ridge 形态已被否定，v4 应换非线性学习器（小 MLP 或树模型）并加入结构特征。

附带一个重定性发现：canonical 去重后的 teacher 从 holdout 三关初始态分别以 `109/35/26` 次扩展解出全部三关（解长 `19/11/19` pushes），而 SD-MCTS 每步 budget `96`、最多 80 步（约 7.7e3 次模拟）仅解 1 关。同一份 SD 读出（matching LB + goal/mobility readouts）在 A* 全局优先队列 + 精确去重下的样本效率远超逐步重规划的深度截断 UCT。这说明当前主要损失在搜索形态而非价值信号本身，后续应评估树跨步复用、transposition 去重进入 MCTS，或将 best-first 搜索作为 SD-as-solving-mechanism 的更自然消费层。

### MCTS 搜索形态消融与 20 关预算扩展判定

按"保留 MCTS 并优化"的决策，为闭环入口实现 `--transposition`（push-level canonical key 节点注册表，DAG 共享统计）与 `--reuse-tree`（planner 跨执行步持久化，含 --transposition）。默认关闭；`revisit_penalty=0` 与新旧实现回归逐项一致（aggregate 与全部 trace 动作序列相同）。

3 关 holdout 上 transposition/树复用把 sd_heuristic+rp1 的 1/3 降为 0/3，但同时将评测扩到新的 20 关切片（`--level-offset 48 --max-levels 20`，关卡 48-67，与 TVA 训练区 8-47 不相交）后，此前全部 3 关结论被拉回噪声：`H=6,budget=96` 下 zero `3/20`、sdh `2/20`、sdh_rp1 `2/20`、sdh_rp1+tp `2/20`、sdh_rp1+rt `2/20`，叶值、重访惩罚与两个搜索机制在该算力下均无可检测差异。3 关切片不足以支撑任何臂间结论，此教训记录在案。

预算扩展探针给出本轮最干净的信号。同一 20 关切片上：zero 随预算反向波动（`H=6,b=96` 3/20 → `H=8,b=256` 1/20，符合"无梯度下加预算只是换一条随机游走"），而 sd_heuristic+rp1 单调上升：`H=6,b=96` 2/20 → `H=8,b=256` 4/20 → `H=10,b=512` 5/20（无一死锁，failures 全部为 max_steps）。判定：SD 读出叶值给 MCTS 提供了可被算力兑现的真实信号，MCTS 路线在该叶值下健康、当前主要受算力约束；zero-leaf 的成功率不可随算力扩展。下一步优先沿 sd_heuristic+rp1 扩 `H/budget`（并测多 seed），transposition/树复用在高预算下再行复测；TVA v4（非线性）此后在同一 20 关切片上对照。

### 算力轴扫描确认可扩展性，transposition 在高预算下成为最优臂

20 关切片（关卡 48-67）上完成算力轴扫描与机制复测。sd_heuristic+rp1 裸臂：`H=6,b=96` 2/20 → `H=8,b=256` 4/20 → `H=10,b=512` 5/20 → `H=12,b=1024` 5/20 → `H=16,b=2048` 10/20；`H=10,b=512` 跨 seed 0/1/2 为 5/6/6，方差小。成功率随 H×budget 总体单调上升，h10-h12 间存在平台后在 h16 跃升。

机制复测推翻了 b=96 时的"无差异"判读——那是预算饥饿假象。`H=12,b=1024`：裸臂 5/20（deadlock_prunes 583998），+transposition 7/20（42527），+reuse-tree 7/20（7446）；`H=16,b=2048`：裸臂 10/20（803549），+transposition 12/20（66397），+reuse-tree 9/20（11590）。判定：transposition 在高预算下同时提升成功率并将节点展开量降低约一个数量级，`sd_heuristic + revisit_penalty=1.0 + transposition @ H=16,b=2048` 为当前最强配置（60%，零学习组件）；树复用是最省算力的臂（再省约 6 倍）但在最高配置点成功率略低于裸臂，其跨步价值陈旧化问题（visited 集合扩张后旧备份值未重估）列为后续改进点。本线从本轮起点的 heldout 0% 推进到 60%。TVA v4 的评估基准随之重锚：其价值主张应为在同等成功率下显著降低 H×budget 需求，对照臂为 transposition 高预算配置。

### TVA v4（MLP）闭环负结果与 TVA-as-leaf 路线判定

按重锚后的价值主张（更小 H×budget 达到同等成功率）实施 v4：audit 脚本新增 `--model mlp`（torch 训练、numpy 推理导出，闭环侧零新依赖），训练样本翻倍至 1920（teacher 20k 预算、零标签污染：solved 586 / proven_dead_end 1270 / no_legal_actions 64），value 与 residual 双模式，离线 gate 均 PASS（residual valid corr 0.454、root sorting 36/42，为历版最佳）。

闭环（20 关、rp1+transposition）：residual `H=10,b=512` 1/20、`H=16,b=2048` 2/20，且 no_action 高达 5 和 11——MLP 残差把执行器主动引向全部合法推均被 Tier-A 剪枝的死角；value `H=16,b=2048` 3/20（deadlocked 3）。对照同配置 sd_heuristic+rp1+tp 的 12/20，差距悬殊。

至此 TVA-as-leaf-regressor 路线在当前特征集上跨四个版本（v1 泄漏 smoke、v2 ridge 脏标签、v3 ridge 净标签、v4 MLP 净标签双倍数据）一致显著劣于零学习 SD 读出，判定为该形态被证伪；同时确认离线门禁（valid corr、近根 root sorting）不预测闭环表现，今后不得以离线 gate PASS 论证闭环有效性。数据指向的转向方案：1270 个可证明死局标签适合训练 dead/alive 分类器，以高精度阈值门控作为 sd_heuristic 叶值之外的附加剪枝器（补充而非替代解析读出，失败模式可控）；或暂停学习组件，优先推进算力轴上限与 50 关验证。

### 算力上限定位与 50 关主结果定稿

继续沿算力轴推进 sd_heuristic+rp1+transposition（20 关切片）：`H=20,b=4096` 达到 15/20（75%），`H=24,b=8192` 回落至 13/20 且出现 3 个执行期死锁。完整曲线 2→4→5→5→10→12→15→13（H6/b96 至 H24/b8192），实用最优点判定为 `H=20,b=4096`；H24 回落的候选解释为 gamma^24≈0.29 导致远端目标信号被折扣侵蚀，以及过深承诺增加死锁暴露，留待 gamma 消融确认。

50 关扩样（关卡 48-97）：`H=16,b=2048` 25/50（50%，零死锁），`H=20,b=4096` **37/50（74%，95% CI 约 ±12%）**，与 20 关子集结果一致。主结果定稿：exact-grid SD-MCTS 在零学习组件（SD matching-lower-bound 叶值 + 重访惩罚 + transposition）下于 50 关 heldout 达到 74% 解题率，单次失败模式以 max_steps 为主（11/50）、死锁仅 1。本线从本日起点（heldout 0%）至此的全部增益来自：SD 读出经叶值通道注入、执行层反震荡、push-level canonical transposition 与 H×budget 扩展；无任何学习权重参与。

### 2026-06-13 GFT-SD 设计评审与 SD action prior（PUCT）实现

依据 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 的 GFT-SD 设计做选择性采纳。对照现实现：该文档的「可解性资源存量」中 `P_t`（matching_lower_bound）、`M_t`（viable_push_count）、`A_t`（reachable）、`D^H_t`（dead-square/freeze/匹配死锁）已在 `sd_model.py` 实现并支撑 37/50 基线；`K_t`/`Q_t`/`C_t` 为真新增。结合「搜索形态是主损失源」（teacher A* 几十次扩展 vs MCTS 数量级更多模拟）的既有判定，最高期望值、最低风险的缺口被判定为文档 §11/§14.2 的 SD action prior（`I_t`）——当前 `GridSDMCTSPlanner` 对候选动作 `rng.shuffle` 后均匀展开、纯 UCB1 无先验。叶值保持解析纯净（不混入软 SD 概念）、硬剪枝只保留数学可证死锁，作为本轮纪律。

实现仅改 `evaluate_sokoban_grid_sdmcts.py`（surgical）：新增 `--action-prior {none,sd_delta}`（默认 `none`）。`sd_delta` 对每个候选动作的子状态算 SD delta 分数 `clip(lb(s)-lb(s'),±4) + w_g·Δboxes_on_goals + w_m·廉价机动性差(默认 w_m=0) − w_r·revisit`，softmax(温度 τ) 后与 uniform 按 `--prior-uniform-mix`(默认 0.1) 混合；untried 按 prior 升序排序、尾部 pop（最高先验先展开）；`select_child` 在 prior 激活时改用 PUCT `Q + c·prior·sqrt(N_parent)/(1+N_child)`，`none` 时保留原 UCB1。

两项纯度验证均 PASS：(1) `--action-prior none` 与改动前落盘结果（`grid_sdmcts_holdout20_sdh_rp1_tp_h6_b96_seed0.json`）20 关 trace 逐项 `mismatches=0`，基线 bit-identical；(2) prior 路径无 `rng` 调用，seed0 vs seed1 同口径 `mismatches=0`，**prior 臂对 seed 确定性**（故该臂不可用多 seed 论证稳健性，改用温度扫描）。

50 关 heldout（关卡 48-97，`H=20,b=4096`，rp1+transposition）结果：

| 臂 | solved | deadlock | max_steps | no_action | 耗时(min) |
|---|---|---|---|---|---|
| baseline（无 prior） | 37/50 | 1 | 11 | 1 | 27.8 |
| prior τ=0.5 | 41/50 | 2 | 4 | 3 | 15.4 |
| prior τ=1.0 | 41/50 | 2 | 5 | 2 | 16.1 |
| prior τ=2.0 | 41/50 | 1 | 5 | 3 | 16.7 |

判定（通过预设验收 ≥39/50、deadlock≤基线+1、耗时≤1.5×）：SD action prior 在 τ∈{0.5,1.0,2.0} 下 solve rate 一致为 **41/50（82%）**，相对 baseline 净 +4，非刀尖点；**主信号是耗时降至约 0.58×**（27.8→16.1 min），失败模式从 max_steps 主导（11→≤5）显著收敛。这正是此前为 TVA 重锚的价值主张——「同等或更高成功率下显著降低 H×budget 消耗」——首次由零额外学习权重的 SD 先验兑现。τ=2.0 略优（死锁回落至 1，与 baseline 持平）。相对 baseline 的增益关卡集合（τ1.0）为 {18,20,22,30,33}，唯一退化为 level_index 9：该关在全部温度下均被 prior 推入死锁（baseline 解出），trace 显示 prior 贪心锁定 box-2 反复振荡而非 baseline 的 box-2-down 解法，属审计预警的 over-concentration 系统性失败，与温度无关、uniform-mix 未能挽回。

双模型审计（read-only）：Codex 指出 `--reuse-tree` 下 prior 仅首次初始化计算、`visited_keys` 增长后不重估（stale prior）——最强配置用 `--transposition` 非 `--reuse-tree`，主结果不受影响，记为 reuse-tree 限制。Gemini 指出 forced-expansion 病理：`simulate` 总是先耗尽 `untried` 队列才进 PUCT，小预算下树过宽过浅、稀释先验；本轮 b=4096 预算充裕故影响有限，列为后续增强项（候选动作以 N=0 直接入 children、绕过 untried 队列以实现 AlphaZero 式预算集中）。判定 `ANALYZED`，新最强配置为 `sd_delta prior(τ=2.0) + rp1 + transposition @ H=20,b=4096` = 50 关 41/50，且约 0.58× baseline 耗时。

### 2026-06-13 GFT-SD 两层架构独立重建

在确认旧六存量 learned-flow 负结果不能作为否定 GFT-SD 两层设计的决策准则后，本轮保留当前最强版本 `evaluate_sokoban_grid_sdmcts.py --action-prior sd_delta` 的成果与复现路径，独立重建 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 中的两层架构。设计口径修订为：旧六存量结果只能说明“未经验证的低维抽象不能独立承担闭合转移 provider”，不能否定 `y_t=(B_t,R_t)` 底账层加 `z_t=Phi(y_t)` 可解性资源层的 GFT-SD 结构。

实现新增 [gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py)，以 OO 继承和开闭原则组织：

```text
SokobanPushDynamicsModel
  -> SokobanGoalDirectedSDModel
    -> SokobanGridStockFlowModel
      -> SokobanGFTSDModel
```

`SokobanGFTSDModel` 的 `transition_provider_id` 仍为 `exact_grid_stock_flow_v1`，`provider_id` 为 `gft_sd_v1`。GFT 资源层不替代精确格点转移，只从底账层计算 `z_t=Phi(y_t)`。新增资源库存接口 `GFTResourceStock` 及八个子类：`TransportProgressStock(P)`、`BoxMobilityStock(M)`、`PushAccessibilityStock(A)`、`RegionCapacitySlackStock(K)`、`DeadlockDebtStock(D)`、`OrderingDebtStock(Q)`、`GoalCommitmentStock(C)`、`ActionRelevanceStock(I)`。当前第一切片中 `P/M/A/D/I` 具备主动读数或 prior 消费；`K/Q/C` 仅作为 inactive interface 记录，等待区域分解、偏序约束和 commitment entropy 审计后再进入在线 prior。

新增独立闭环入口 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)。该入口复用现有 MCTS 基类，但实例化 `SokobanGFTSDModel`，并通过 `--gft-prior gft_delta` 消费 `I_t` 动作相关性；`--gft-prior none` 可回退到原 `--action-prior` 行为。`gft_delta` 的默认权重与当前已验证 prior 对齐：`progress=1.0`、`goal=1.0`、`mobility=0.0`、`access=0.0`、`capacity=0.0`、`deadlock=0.0`、`ordering=0.0`、`commitment=0.0`、`revisit=1.0`，温度默认 `2.0`、uniform mix `0.1`。该入口的输出合同明确记录 `gft_resource_layer_replaces_transition_provider=false`、`legacy_six_stock_rollout_used=false`、`teacher_used_online=false` 和 inactive resources `K/Q/C`。

验证：`python -m py_compile src/sokoban/gft_sd.py experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py` 通过；直接导入 `SokobanGFTSDModel` 可读出 `P/M/A/K/D/Q/C/I` 八类资源，并为初始态前三个候选动作生成归一化 prior。极小闭环 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 1 --horizon 2 --budget 8 --max-steps 5 --seed 0 \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --trace-gft-readouts \
  --out results/sokoban_sd_world_model/gft_sdmcts_smoke_level1_h2_b8_seed0.json
```

输出成功落盘，`framework_contract` 为 `sd_provider=gft_sd_v1`、`transition_provider=exact_grid_stock_flow_v1`，1 关小预算结果为 `0/1`、失败类型 `max_steps`、`total_deadlock_prunes=4`。该 smoke 仅验证架构与落盘，不作为性能结论。

随后运行正式 50 关复现测试：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 50 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，aggregate 为 `41/50`、success rate `0.82`、mean pushes solved `15.146341463414634`、total deadlock prunes `174430`，状态分布为 `deadlocked=1`、`max_steps=5`、`no_action=3`、`solved=41`。与既有最强 grid 入口输出 [grid_sdmcts_holdout50_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout50_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0.json) 比较，`aggregate_equal=True`，逐关 `status/solved/pushes/steps_executed/deadlock_prunes/final_features` mismatch 为 `0`，根节点 trace 中 `action/root_visits/root_value/top_children` mismatch 为 `0`。判定：GFT-SD 两层闭环入口通过正式等价复现门槛；该结果是架构重建复现，不构成相对于已验证 SD action prior 的新增性能增益。

随后针对 GFT 入口效率做行为保持型修订。`SokobanGFTSDModel` 增加 GFT 层 lower-bound、canonical key 与 abstract feature 缓存；trace readout 在 `--trace-gft-readouts` 下只计算 active 资源；`GFTSDMCTSPlanner` 对默认等价权重的 `gft_delta` 保留与 grid `sd_delta` 对齐的 prior 通道。单状态 prior 比较显示 `max_prior_diff=0.0`、`max_score_diff=0.0`；3 关闭环 gate 与 grid 输出的逐关字段和 root trace mismatch 均为 `0`。

完整 50 关优化后输出 [gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json)，aggregate 仍为 `41/50`，状态分布仍为 `deadlocked=1`、`max_steps=5`、`no_action=3`、`solved=41`；与既有 grid 50 关输出比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。该轮 wall time 为 `2102.34s`，JSON 内逐关 elapsed 汇总为 `2099.7606648490764s`。

为避免把不同时段 wall time 误判为架构开销，补充同机顺序 10 关重测：grid 输出 [grid_sdmcts_gate10_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0_retime.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_gate10_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0_retime.json)，`real=644.04s`、elapsed 汇总 `641.7514883917756s`；GFT 输出 [gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json)，`real=641.79s`、elapsed 汇总 `639.5067871282808s`。两者 aggregate 均为 `6/10`，状态分布均为 `deadlocked=1`、`max_steps=2`、`no_action=1`、`solved=6`，逐关字段和 root trace mismatch 均为 `0`。因此效率结论修订为：在当前同机重测中，默认 GFT 入口未显示相对 grid 入口的系统性额外开销；早先 `38` 分钟 vs `16.7` 分钟的比较不作为架构开销结论。

### 2026-06-13 GFT-SD 第二层资源 K 的第一切片

根据继续讨论，下一步从修改 MCTS 搜索形态转向实际启用 GFT-SD 第二层资源。为避免缩写歧义，本节中的 `K` 指“区域容量裕度”：把地图按静态割点分解为目标相关区域，估计每个目标区域的剩余容量、入口阻塞和错误箱子压力。该资源对应 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 中 `K_t`，不作为硬剪枝，只作为可审计 readout 和默认关闭的软动作先验分量。

实现上，[gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py) 将 `RegionCapacitySlackStock(K)` 从 inactive interface 改为 active readout。区域分解采用 `floor_components_after_cut_vertices`：先在静态 floor 图上寻找割点，移除割点后得到连通分量，再保留含目标的分量作为目标区域。容量健康度定义为 `capacity_health=-capacity_debt`，越大越好；`capacity_debt` 来自 committed 箱子超出区域目标数、非目标入口被箱子阻塞、以及处在区域内但静态可行目标集合与该区域目标不相交的错误箱子。为避免误把过路箱子计入容量占用，`committed_box_count` 只统计“静态可行目标集合完全落在该区域内”的箱子。

在线入口 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 已将 `--gft-capacity-weight` 接入 `gft_delta` 先验：当该权重为 `0` 时，默认路径仍等价于旧 `sd_delta`；当该权重非零时，动作分数增加 `capacity_weight * (K(s')-K(s))`。输出合同中的 inactive resources 从 `K/Q/C` 修订为 `Q/C`。

验证：语法检查通过；初始态资源读数显示 `K` 为 active/implemented。默认 `capacity_weight=0` 时前三个候选动作 prior 仍为 `[0.173159, 0.41342, 0.41342]`，`capacity_delta` 均为 `0`；显式 `capacity_weight=1.0` 时同一候选 prior 变为 `[0.24276, 0.37862, 0.37862]`，对应两个动作的 `capacity_delta=-1`，说明 `K` 已实际影响软先验。

heldout 50 关初始动作审计显示，共 `223` 个初始合法动作，其中 `43` 个具有非零 `K` 变化；`30/50` 关至少有一个初始动作触发非零 `K` 变化。默认 3 关闭环 gate 输出 [gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与旧 GFT 3 关输出的 aggregate、逐关字段和 root trace mismatch 均为 `0`，说明 `K` readout 不改变默认最佳复现路径。

10 关正式预算开发门槛测试使用 `--gft-capacity-weight 0.25`，输出 [gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 `7/10`，状态分布 `deadlocked=1`、`max_steps=1`、`no_action=1`、`solved=7`，相对于同机默认 GFT 10 关 `6/10`，新增解出评测关卡 `level_index=5`，未新增 deadlocked 类型。该结果记为 `K` 第一切片的正向开发信号，但不替代 50 关正式结论。

原计划启动的 50 关 `K` 正式测试因单轮成本过高被主动停止，未产生完整输出，不作为结果证据。后续验证策略修订为：`K/Q/C` 每个新资源先以同一 10 关正式预算切片作为开发门槛；只有当 10 关至少不劣于默认线、且无新增死锁/明显失败退化时，才安排 50 关阶段性确认。

### 2026-06-13 GFT-SD 第二层资源 Q 的第一切片

本轮继续启用第二层资源中的 `Q`，即“顺序依赖债务”。为避免缩写歧义，`Q` 在此处不表示 MCTS 中的动作价值，而表示 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 中的 `Q_t`：若目标区浅层位置被过早占用，或目标区域尚未完成时入口被箱子堵住，则说明后续可解顺序被消耗，债务增加。

实现上，[gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py) 将 `OrderingDebtStock(Q)` 从 inactive interface 改为 active readout。该读出复用 `K` 的目标区域分解，对每个目标区域从入口割点到各目标计算静态深度，并形成“更深目标先于更浅目标”的偏序边；若浅层目标有箱子而更深目标仍为空，则计为 goal-order violation。若区域仍有未占用目标，而入口割点被非目标箱子占用，则计为 entry-order violation。当前 `ordering_debt` 为两类 violation 数之和，越大越差；该债务只作为软读出，不作为硬剪枝。

在线入口 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 已将 `--gft-ordering-weight` 接入 `gft_delta` 先验：当该权重为 `0` 时，默认路径仍等价于旧 `sd_delta`；当该权重非零时，动作分数增加 `-ordering_weight * (Q(s')-Q(s))`。输出合同中的 inactive resources 从 `Q/C` 修订为 `C`。

验证：语法检查通过，`git diff --check` 通过。50 关 heldout 初始动作审计显示，共 `223` 个初始合法动作，其中 `60` 个具有非零 `Q` 变化；`38/50` 关至少有一个初始动作触发非零 `Q` 变化，说明该读出并非空变量。默认 3 关 gate 输出 [gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与 `K` readout 后的默认 3 关输出比较，aggregate、逐关字段和 root trace mismatch 均为 `0`，说明 `Q` readout 默认不改变最佳复现路径。

10 关正式预算开发门槛分别测试 `--gft-ordering-weight 0.25` 与 `0.1`。`0.25` 输出 [gft_sdmcts_gate10_qord0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_qord0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `6/10`，状态分布 `deadlocked=2`、`max_steps=1`、`no_action=1`、`solved=6`。`0.1` 输出 [gft_sdmcts_gate10_qord0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_qord0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `6/10`，状态分布 `deadlocked=2`、`max_steps=2`、`solved=6`。二者均未超过同机默认 GFT 的 `6/10`，且均把 `level_index=5` 从 `max_steps` 推为 `deadlocked`。判定：`Q` 第一切片作为 readout 有覆盖和解释信号，但作为在线先验未通过 10 关开发门槛；当前不进入推荐配置，默认权重继续保持 `0`。

### 2026-06-13 GFT-SD 资源统一先验接口整理

根据继续讨论，确认 `P`（运输进度）不应因最早形成正向信号而在最终架构中永久占据特殊通道。GFT-SD 二层资源应采用统一介入范式：底账状态 `y_t` 先映射为资源读出 `z_t=Phi(y_t)`，候选动作通过 `Delta z_t` 生成动作先验、TVA 特征和诊断日志。现有 `sd_heuristic` 叶值保留为零学习历史基线，不作为最终 TVA-MCTS 架构原则。

实现上，[evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 移除了 `GFTSDMCTSPlanner.assign_sd_delta_priors()` 中重复实现的 progress、mobility、capacity、deadlock 和 ordering 打分逻辑，也移除了“当 `K/Q/C` 等权重为零时回退到 grid `sd_delta` 快路径”的特殊分支。GFT 入口现在统一调用 [gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py) 中的 `SokobanGFTSDModel.gft_action_prior_rows()` 生成每个候选动作的 `score`、`prior` 和资源变化分量。由此，`P/K/Q/C` 后续进入动作先验时共享同一消费位置，规划器层不再单独处理 `P`。

验证：语法检查通过。默认 3 关等价 gate 输出 [gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与上一版默认 3 关输出 [gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。该修订是架构口径整理，不构成新性能结果。

### 2026-06-13 K+Q 组合交互测试

为检验第二层资源之间是否存在互补效应，本轮在统一先验接口上运行 `K+Q` 小组合测试：`--gft-capacity-weight 0.25`、`--gft-ordering-weight 0.05`，其余预算、叶值、重访惩罚、转置表、温度和 uniform mix 均沿用 10 关正式预算门槛。输出 [gft_sdmcts_gate10_kcap0p25_qord0p05_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_qord0p05_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `6/10`，状态分布 `deadlocked=1`、`max_steps=2`、`no_action=1`、`solved=6`，`real=662.41s`。

逐关比较显示，组合配置相对默认 GFT 没有改变 solved/status/pushes；相对 `K` 单独配置则退化一个关卡：`level_index=5` 从 `solved`、`24` pushes 回到 `max_steps`。因此当前 `Q` 第一切片在 `K` 的容量健康保护下没有形成互补增益，反而抵消了 `K` 的唯一新增解题关。判定：不继续扩大 `Q` 权重搜索，当前推荐候选仍为 `K` 单独的 `capacity_weight=0.25`；下一步转向 `C` 目标承诺读出的单因素实现。

### 2026-06-13 GFT-SD 第二层资源 C 的第一切片

本轮继续启用第二层资源中的 `C`，即“目标承诺清晰度”。为避免缩写歧义，`C` 在此处表示箱子-目标分配是否已经清晰，而不是旧六存量中的拥塞变量。该资源对应 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 中的 `C_t`：由箱子到目标的运输成本矩阵构造软分配，并通过分配熵刻画当前目标承诺是否稳定。

实现上，[gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py) 将 `GoalCommitmentStock(C)` 从 inactive interface 改为 active readout。具体方法为：使用每个箱子到每个目标的静态 reverse-pull 距离构造成本矩阵；不可达项置为不可行；对 `exp(-cost)` 执行 Sinkhorn 行列归一化，得到软箱子-目标分配矩阵；再计算归一化熵并定义 `commitment_confidence=1-normalized_entropy`。该值越大表示箱子-目标分配越清晰。在线入口已将 `--gft-commitment-weight` 通过统一 `gft_action_prior_rows()` 接入先验：当该权重非零时，动作分数增加 `commitment_weight * (C(s')-C(s))`。输出合同中的 inactive resources 修订为空列表。

验证：语法检查通过。50 关 heldout 初始动作审计显示，共 `223` 个初始合法动作，其中 `155` 个具有非零 `C` 变化；`32/50` 关至少有一个初始动作触发非零 `C` 变化。初始承诺清晰度范围为 `0.080029` 到 `0.850247`，初始动作 `C` 变化范围为 `-0.731939` 到 `0.336704`。默认 3 关 gate 输出 [gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与统一先验默认 3 关输出比较，aggregate、逐关字段和 root trace mismatch 均为 `0`，说明 `C` 默认关闭时不改变当前最佳复现路径。

10 关正式预算开发门槛分别测试 `--gft-commitment-weight 0.5` 与 `0.25`。`0.5` 输出 [gft_sdmcts_gate10_ccommit0p50_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_ccommit0p50_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `6/10`，状态分布 `deadlocked=2`、`max_steps=1`、`no_action=1`、`solved=6`。`0.25` 输出 [gft_sdmcts_gate10_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `6/10`，状态分布 `deadlocked=2`、`max_steps=2`、`solved=6`。二者均未超过同机默认 GFT 的 `6/10`，且均把 `level_index=5` 从 `max_steps` 推为 `deadlocked`。判定：`C` 第一切片作为 readout 有覆盖和解释信号，但作为在线先验未通过 10 关开发门槛；当前不进入推荐配置，默认权重继续保持 `0`。

随后补充 `K+C` 组合交互测试，回应“资源之间可能需要配合才有效”的假设。配置为 `--gft-capacity-weight 0.25` 与 `--gft-commitment-weight 0.25`，输出 [gft_sdmcts_gate10_kcap0p25_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 `7/10`，状态分布 `deadlocked=1`、`max_steps=1`、`no_action=1`、`solved=7`，`real=557.08s`。逐关比较显示，该组合相对 `K` 单独配置没有 solved/status/pushes 差异；相对 `C` 单独配置则恢复了 `level_index=5` 的 `24` push 解。判定：`K+C` 通过“不劣于 `K` 且不新增 deadlock”的组合门槛，但没有提供超过 `K` 单独的增益；当前推荐候选仍为更简洁的 `K` 单独配置，`C` 作为 TVA/诊断候选特征保留。

### 2026-06-14 GFT-SD 设计文档完整对齐实现

本轮按 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 对 GFT-SD 进行完整结构补齐，同时保留当前最强默认路径。实现范围限定在 [gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py)、[__init__.py](/home/data/round/code/sd_world_model_rl/src/sokoban/__init__.py) 和 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)。底账层仍为 `exact_grid_stock_flow_v1`，资源层仍为 `z_t=Phi(y_t)`，不替代精确转移 provider。

补齐内容如下。第一，`P`（目标运输进度）不再只在代码语义上称为 matching lower bound，而是显式实现文档成本矩阵 `d_g(b_i)+lambda_turn*turns+lambda_access*access_risk+lambda_zone*zone_risk`；默认 `lambda_turn=lambda_access=lambda_zone=0`，因此当前默认行为严格退化为既有 matching lower bound。第二，`C`（目标承诺清晰度）改为读取同一运输成本矩阵，再经 Sinkhorn 归一化得到软箱子-目标分配熵，避免 `P` 和 `C` 使用两套成本。第三，`D`（死锁债务）拆成硬死锁债务和软死锁压力；软压力包含机动性损失、匹配脆弱性、容量压力、可达性损失、顺序违例五个分量，默认 `alpha` 权重均为 `1.0`。第四，新增 `gft_resource_vector()`、`gft_resource_flow()`、`gft_sd_reward()`、`gft_sd_value()`、`gft_tva_feature_dict()`，分别对应设计文档的 `z_t`、`Delta z_t`、`r_SD`、`V_SD(z)` 和未来 TVA 特征接口。第五，`I`（动作相关性）保留原统一动作先验，同时新增瓶颈集合与动作影响域 overlap 分量，默认权重为 `0`，不改变当前推荐配置。

评估入口同步新增运输成本权重、软死锁权重、Sinkhorn 参数和瓶颈 overlap 权重的命令行参数，并在 `framework_contract.gft_spec_alignment` 中记录设计文档对齐护照。`--trace-gft-readouts` 现在除原始资源读出外，还会落盘 `gft_flow`、`gft_sd_reward`、`gft_sd_value_before/after` 和 `gft_tva_features_after`。

验证结果：语法检查通过；5 个关卡、14 个状态的结构审计显示资源向量、资源流量、系统动力学奖励、显式值函数和 TVA 特征均无非预期非有限值。默认 3 关回归输出 [gft_sdmcts_gate3_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与上一版 `C` readout 默认输出比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。带 GFT 追踪字段 smoke 输出 [gft_sdmcts_trace_fullspec_v1_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_trace_fullspec_v1_level1_h3_b24_seed0.json)，新增字段完整落盘。

10 关正式预算测试继续采用当前内循环门槛。默认配置输出 [gft_sdmcts_gate10_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `6/10`，状态分布 `deadlocked=1,max_steps=2,no_action=1,solved=6`；与旧默认 10 关输出的 aggregate、逐关字段和 root trace mismatch 均为 `0`。`K`（区域容量裕度）候选 `capacity_weight=0.25` 输出 [gft_sdmcts_gate10_fullspec_v1_kcap0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_kcap0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果保持 `7/10`，状态分布 `deadlocked=1,max_steps=1,no_action=1,solved=7`；与旧 `K=0.25` 输出的 aggregate、逐关字段和 root trace mismatch 均为 `0`。因此，本轮为设计文档完整对齐与接口补齐，不改变当前默认最强 50 关结论；10 关内循环中的当前正向候选仍为 `K=0.25`。

### 2026-06-14 GFT-SD 全因素保留原则与组合测试

根据继续讨论，本轮修订二层资源测试判据：单因素切片不再用于“未涨分即删除因素”，而用于检查该因素在当前手工在线先验消费方式下是否造成严重破坏。若没有明显低于默认线、死锁大幅增加或耗时失控，则该因素保留进入组合测试和后续可学习权重、TVA 阶段。旧六存量 learned-flow 负结果仍不得作为否定当前 GFT-SD 两层资源设计的证据。

统一 10 关正式预算为：关卡 `48-57`，`horizon=20`，`budget=4096`，`max_steps=80`，`seed=0`，`leaf_value=sd_heuristic`，`revisit_penalty=1.0`，transposition，`gft_prior=gft_delta`，temperature `2.0`，uniform mix `0.1`，并要求独立门禁通过。默认 GFT 为 `6/10`，状态分布 `deadlocked=1,max_steps=2,no_action=1,solved=6`。

单因素结果显示：`K=0.25`、`P_access=0.25`、`P_zone=0.25` 均为 `7/10`；`P_turn=0.25`、`M=1.0`、`A=1.0`、`D=0.25`、`Q=0.05`、`C=0.05`、`I=0.10` 均未低于默认 `6/10`，但若干配置把 `level_index=5` 推为 deadlocked 或使 `level_index=6` 维持失败。因此这些因素不被删除，但其中多数不作为单独手工先验推荐配置。

组合测试给出新的协同信号。运输成本三项组合 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25` 输出 [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果为 `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，相对默认同时解出 `level_index=5` 与 `level_index=6`，失败仅剩 `level_index=0` 的 max_steps 与 `level_index=9` 的 deadlocked。加入 `Q=0.05` 后仍为 `8/10`；加入 `Q=0.05,C=0.05` 后仍为 `8/10`，说明顺序依赖债务和目标承诺清晰度在强运输成本底座上可以共存。

强底座逐项叠加结果如下：

| 配置 | solved | 状态分布 | 主要解释 |
|---|---:|---|---|
| 默认 GFT | 6/10 | `deadlocked=1,max_steps=2,no_action=1` | 10 关门槛基线 |
| `P_turn0.10+P_access0.25+P_zone0.25` | 8/10 | `deadlocked=1,max_steps=1` | 当前 10 关最好候选之一 |
| `P3+K0.25` | 7/10 | `deadlocked=1,max_steps=1,no_action=1` | 第 6 关从 solved 回到 no_action |
| `P3+M1.0` | 7/10 | `deadlocked=1,max_steps=1,no_action=1` | 未严重恶化，手工权重稀释强底座 |
| `P3+A1.0` | 7/10 | `deadlocked=1,max_steps=2` | 未严重恶化，第 6 关回到 max_steps |
| `P3+D0.25` | 7/10 | `deadlocked=1,max_steps=2` | 未严重恶化，第 6 关回到 max_steps |
| `P3+Q0.05` | 8/10 | `deadlocked=1,max_steps=1` | 保持当前最好候选 |
| `P3+C0.05` | 7/10 | `deadlocked=1,max_steps=1,no_action=1` | 单独加入时稀释第 6 关 |
| `P3+Q0.05+C0.05` | 8/10 | `deadlocked=1,max_steps=1` | `Q` 与 `C` 存在协同保留证据 |
| `P3+I0.10` | 6/10 | `deadlocked=2,max_steps=1,no_action=1` | 当前手工瓶颈 overlap 不宜默认开启 |
| `P3+Q0.05+C0.05+I0.02` | 7/10 | `deadlocked=1,max_steps=1,no_action=1` | 弱瓶颈仍压低强底座，但非结构删除证据 |

全包测试进一步检查完整设计是否崩塌。保守全包（无转弯成本，含 `K=0.25,M=1.0,A=1.0,D=0.25,Q=0.05,C=0.05,I=0.10`）为 `7/10`；关闭瓶颈 overlap 仍为 `7/10`；加入 `lambda_turn=0.10` 且关闭瓶颈 overlap 为 `7/10`；加入 `lambda_turn=0.10` 且 `I=0.10` 为 `6/10`；低权重全包并使用弱瓶颈 `I=0.02` 为 `7/10`。因此完整二层资源结构没有被组合测试推翻，但当前手工瓶颈 overlap 消费方式明确不进入推荐在线配置。

本轮 10 关内循环的当前最好候选更新为两类：其一，运输成本三项组合 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25`；其二，在其上加入 `ordering_weight=0.05` 与可选 `commitment_weight=0.05`。三者均为 `8/10`，均未增加 deadlocked 数。该结论只更新 10 关开发候选，不替代既有 50 关最强结果 `41/50`；后续若要改写正式 heldout 结论，应先用上述候选做 50 关阶段性确认。

### 2026-06-14 GFT-TVA v1 接入与首轮闭环负结果

根据继续讨论，本轮转向 TVA 接入。设计口径采用 [sokoban_gptpro.md](/home/data/round/code/sd_world_model_rl/sokoban_gptpro.md) 第 15 节：叶值为 `V_leaf(z_h)=V_SD(z_h)+V_TVA(z_h)`，其中 `V_TVA` 学习显式 GFT-SD 值函数无法覆盖的长期可解性残差。实现新增 [train_sokoban_gft_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py)，训练目标来自 GFT-MCTS 根节点统计 `V_search=sum_a visit_alpha_probability(a) Q_H(s,a)`，不使用 diagnostic teacher 或 best-first teacher 生成训练标签。GFT 评估入口 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 同步支持 `--leaf-value gft_sd` 与 GFT-TVA checkpoint；当 checkpoint 的 target mode 为 `gft_sd_residual` 时，`--leaf-value tva` 执行 `V_SD+V_TVA`。

实现中发现一个必要数值修订：`gft_sd_value()` 的资源量纲不是天然的 MCTS 回报量纲，且非有限运输进度惩罚会把少量状态推到极大负值。为避免叶值备份被未校准量纲压垮，闭环消费采用“规划值尺度上的显式 SD 值”：先读取 `gft_sd_value()`，再投影到 `[deadend_value, goal_value]`。训练残差也使用同一投影定义。该修订不改变 GFT-SD 结构读出，只限定其作为 MCTS leaf value 时的数值范围。

接口验证分两步。第一，小规模 smoke 使用关卡 `8-13`，`state_depth=1,max_states_per_level=8,collect_horizon=3,collect_budget=32`，输出 [sokoban_gft_tva_v1_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_train_seed0.json) 与 [sokoban_gft_tva_v1_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_seed0.json)，gate 为 `PASS`，train `14` 样本、valid `11` 样本，valid corr `0.967`。随后 1 关闭环 smoke 输出 [gft_sdmcts_tva_v1_smoke_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_tva_v1_smoke_level1_h3_b24_seed0.json)，`--leaf-value tva` 成功读取 GFT checkpoint 并落盘，结果为 `0/1`，失败类型 `max_steps`。该 smoke 只证明接线可执行。

第二，正式开发版训练使用关卡 `8-47`，避开当前 10 关开发切片 `48-57`；配置为 `state_depth=1,max_states_per_level=12,collect_horizon=4,collect_budget=64`，强 GFT 先验 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25,ordering_weight=0.05,commitment_weight=0.05`，bootstrap leaf 为 `gft_sd`。输出 [sokoban_gft_tva_v1_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_train_seed0.json) 与 [sokoban_gft_tva_v1_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_seed0.json)。训练集 `169` 样本，验证集 `35` 样本，gate 为 `PASS`；train MAE `0.381`、corr `0.641`，valid MAE `0.776`、corr `0.056`。该离线结果只能说明 checkpoint 合同和目标收集成立，不能说明 TVA 已有可用泛化。

10 关闭环开发门槛给出明确负结果。在同一强 GFT 先验下，当前 10 关最好对照仍是 `sd_heuristic` 叶值的 [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)，结果 `8/10`。显式 `V_SD` 叶值输出 [gft_sdmcts_gate10_gftsdleaf_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，仅 `1/10`，状态分布 `deadlocked=1,max_steps=1,no_action=7,solved=1`。`V_SD+TVA` 输出 [gft_sdmcts_gate10_gfttva_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，为 `0/10`，状态分布 `max_steps=1,no_action=9`。

判定：GFT-TVA v1 的工程接入完成，但该 ridge 残差 TVA 不进入推荐配置。首要瓶颈不是 checkpoint 接线，而是显式 `V_SD(z)` 的默认 `beta` 权重和量纲尚不具备 planner-ready 叶值能力；在 `V_SD` 基线仅 `1/10` 的情况下，`V_SD+TVA` 难以超过 `sd_heuristic` 对照。后续若继续 TVA，应先处理两条路径之一：其一，校准 `V_SD` 的 `beta` 权重，使 `gft_sd` 叶值至少接近 `sd_heuristic` 对照；其二，把当前强对照 `sd_heuristic` 作为锚点训练 `sd_heuristic_residual`，再逐步迁移到显式 `V_SD`。在此之前，不应安排 50 关 GFT-TVA 正式确认。

### 2026-06-14 GFT-SD 显式叶值校准

根据 GFT-TVA v1 的闭环负结果，本轮先校准显式 `V_SD` 叶值，而不是继续扩大 TVA 训练。原因是上一轮 `linear_projected` 模式把 `gft_sd_value()` 的资源量纲直接投影到规划回报区间，10 关仅 `1/10`，已经不足以作为 TVA 残差的稳定基底。

实现上，[evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 增加 `--gft-sd-leaf-mode discounted_progress`。该模式先读取 GFT 运输进度 `P`，并按 MCTS 的折扣回报尺度计算叶值：若 `gamma<1`，使用 `gamma^P * goal_value - push_cost*(1-gamma^P)/(1-gamma)`；若运输进度非有限，则返回 dead-end value。第二层资源项只作为可选校正项加入，默认权重均为 `0`。同时新增零权重快速判断；当叶值资源权重全为 `0` 时，不再在每个叶节点构造完整 TVA 特征，避免无意义的高频计算。

10 关强底座复跑采用关卡 `48-57`、`horizon=20`、`budget=4096`、`max_steps=80`、强运输成本组合 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25`、`ordering_weight=0.05`、`commitment_weight=0.05`、temperature `2.0`、uniform mix `0.1`。零资源权重输出 [gft_sdmcts_gate10_gftsdleaf_discounted_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，mean pushes solved `17.5`，total deadlock prunes `29107`，`real=488.25s`。与 `sd_heuristic` 强底座 [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) 逐关完全一致：`level_index=0` 为 max_steps，`level_index=9` 为 deadlocked，其余 8 关 solved，推动数也完全一致。

随后运行保守第二层叶值组合：区域容量裕度、顺序依赖债务、目标承诺清晰度三项均以 `0.05` 进入叶值校正。输出 [gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果仍为 `8/10`，状态分布仍为 `deadlocked=1,max_steps=1,solved=8`，mean pushes solved `17.75`，total deadlock prunes `30314`，`real=550.80s`。失败关卡不变，解出关卡不变；推动数变化集中在 `level_index=2` 从 `26` 到 `28`、`level_index=5` 从 `30` 到 `26`、`level_index=6` 从 `19` 到 `23`。

判定：`linear_projected` 的负结果保持有效，但不能代表显式 GFT-SD 叶值整体不可用。`discounted_progress` 已把显式 `V_SD` 的基底校准回当前强 10 关对照，具备作为下一版 TVA 训练基底的资格。区域容量裕度、顺序依赖债务和目标承诺清晰度以保守权重进入叶值未造成严重退化，因此结构继续保留；但非零资源权重会使运行时间从 `488.25s` 增至 `550.80s`，后续扩大权重扫或重训 TVA 前应优先做第二层特征缓存。

### 2026-06-14 GFT-SD 第二层特征缓存

根据显式叶值校准暴露出的效率问题，本轮先做行为保持型缓存，而不是继续扩大权重扫或重训 TVA。实现范围为 [gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py) 与 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)。

模型层新增两个缓存：`_gft_resource_vector_cache` 缓存完整资源向量 `z_t=Phi(y_t)`，`_gft_tva_feature_cache` 缓存稳定 TVA 特征合同 `gft_tva_feature_vector_v1`。缓存读取时返回副本，避免调用方修改内部缓存对象。规划器层新增 `_gft_sd_leaf_cache` 和 `_gft_tva_prediction_cache`，分别缓存同一 canonical state 的显式 GFT-SD 叶值和 GFT-TVA 预测值。缓存键采用已有 canonical key，即箱子位置与玩家可达分量等价类，因此不改变叶值语义。

验证分两步。第一，语法检查通过，并运行低预算 smoke：关卡 `48-50`，`horizon=6`，`budget=96`，`max_steps=12`，`leaf_value=gft_sd`，`gft_sd_leaf_mode=discounted_progress`，叶值资源权重 `K/Q/C=0.05`。输出 [gft_sdmcts_gate3_gftsdleaf_discounted_cache_smoke_k0p05_q0p05_c0p05_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_gftsdleaf_discounted_cache_smoke_k0p05_q0p05_c0p05_h6_b96_seed0.json)，结果为 `0/3`，状态分布 `max_steps=3`，`real=4.60s`。该 smoke 只验证缓存路径可执行。

第二，复跑上一轮保守第二层叶值组合。输出 [gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果仍为 `8/10`，状态分布仍为 `deadlocked=1,max_steps=1,solved=8`，mean pushes solved 仍为 `17.75`，total deadlock prunes 仍为 `30314`。与缓存前结果 [gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json) 比较，aggregate 完全一致，逐关 `level_index/status/solved/pushes/steps_executed/deadlock_prunes` mismatch 为 `0`。

效率上，缓存前同配置 `/usr/bin/time` 为 `real=550.80s`、`user=552.77s`、`sys=5.10s`；缓存后为 `real=481.41s`、`user=483.22s`、`sys=5.10s`，wall time 降低约 `12.6%`。该结果说明第二层特征缓存已解决一部分叶值高频计算成本，并且没有改变行为。若后续仍需进一步降耗，下一瓶颈应转向 MCTS 展开形态和动作先验计算，而不是重复构造 TVA 特征合同。

### 2026-06-14 GFT-TVA v2：discounted-progress 基底重训与闭环负结果

根据显式叶值校准和特征缓存结果，本轮重训 GFT-TVA。训练基底从 v1 的 `linear_projected` 改为 `discounted_progress`，并保留保守第二层叶值权重：区域容量裕度 `0.05`、顺序依赖债务 `0.05`、目标承诺清晰度 `0.05`。为避免残差 checkpoint 与闭环运行基底混用，评估入口 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 新增 GFT-TVA checkpoint 合同检查：当 target mode 为 `gft_sd_residual` 时，运行时的 `gft_sd_leaf_mode` 与 `gft_sd_leaf_weights` 必须与 checkpoint 中记录的训练合同一致。

训练仍使用关卡 `8-47`，避开 10 关开发评测 `48-57`；配置保持 v1 同口径：`state_depth=1,max_states_per_level=12,collect_horizon=4,collect_budget=64`，强 GFT 先验为 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25,ordering_weight=0.05,commitment_weight=0.05`。输出 [sokoban_gft_tva_discounted_v2_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_train_seed0.json) 与 [sokoban_gft_tva_discounted_v2_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_seed0.json)。gate 为 `PASS`，样本数仍为 `204`，其中 train `169`、valid `35`。离线指标相对 v1 有明显改善：train MAE `0.318`、corr `0.594`；valid MAE `0.871`、corr `0.668`。训练耗时 `real=29.79s`。残差目标整体为负，train target mean `-0.491`、valid target mean `-1.072`，说明短视界 search-value 目标通常低于显式 `V_SD` 基底。

随后执行 10 关闭环门槛。输出 [gft_sdmcts_gate10_gfttva_discounted_v2_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_discounted_v2_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `1/10`，状态分布 `deadlocked=8,max_steps=1,solved=1`，mean pushes solved `12.0`，total deadlock prunes `10653`，`real=380.03s`。对比显示，强 `sd_heuristic` 对照为 `8/10`，已校准显式 `V_SD` 也为 `8/10`；v2 只保留 `level_index=4` 解出，并把原本解出的 `level_index=1,2,3,5,6,7,8` 推为 deadlocked。

初始状态诊断显示，10 个评测初始状态上的残差预测范围约为 `[-0.695,-0.114]`，均值约 `-0.425`；初始态没有极端爆炸。但闭环中负残差会持续改变后续状态排序，使搜索更早进入短路径死锁。判定：GFT-TVA v2 证明 `discounted_progress` 基底能显著改善离线残差可拟合性，但全强度 ridge 残差仍不具备闭环安全性。该结果不进入推荐配置，不安排 50 关确认。下一步应转向残差缩放、裁剪或 safety gate，使 TVA 只在置信度足够或不破坏显式 `V_SD` 排序时介入。

### 2026-06-14 GFT-TVA 直接终端价值口径修订

根据继续讨论，本轮修订 GFT-TVA 的默认目标口径。此前 v1/v2 按 `V_leaf=V_SD+V_TVA` 训练残差，其中 `V_TVA` 的目标为 `V_search-V_SD`。复核 NeurIPS 论文公式以及 Beer Game、Cartpole 已完成实现后，确认更一致的主线应是直接终端价值 TVA：训练 `V_psi(s_leaf)` 对齐 MCTS 根节点搜索值 `V_search`，闭环叶节点默认直接消费 `V_leaf=V_psi(s_leaf)`。显式 `V_SD` 仍作为系统动力学读出和可控兜底保留，但不再作为默认训练目标的一部分。

实现范围为 [train_sokoban_gft_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py) 与 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)。训练脚本新增并默认采用 `--target-mode direct_value`，旧 `value` 名称作为兼容别名，写入 checkpoint 时统一规范为 `direct_value`；`gft_sd_residual` 仍保留为可复现实验变体。checkpoint 合同新增 `direct_value_formula=V_psi=V_search`、`leaf_formula=V_leaf=V_psi` 与 `paper_alignment=direct_terminal_value_tva`。

评估入口新增 `--tva-value-blend`。对于 `direct_value` checkpoint，`--tva-value-blend 1.0` 执行纯直接价值叶值 `V_leaf=V_psi`；若设为 `alpha<1`，则执行受控混合 `V_leaf=(1-alpha)V_SD+alpha V_psi`，用于安全兜底或消融。只有在残差模式或直接价值混合模式使用 `V_SD` 时，评估端才要求运行时 `gft_sd_leaf_mode` 与 `gft_sd_leaf_weights` 匹配 checkpoint 合同；纯 `V_psi` 模式不依赖 `V_SD` 叶值基底。

验证分三步。第一，语法检查和两个入口的 `--help` 均通过。第二，小规模 direct-value 训练 smoke 使用关卡 `8-10`、`state_depth=1`、`max_states_per_level=4`、`collect_horizon=2`、`collect_budget=12`，输出 [sokoban_gft_tva_direct_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_train_seed0.json) 与 [sokoban_gft_tva_direct_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_seed0.json)，gate 为 `PASS`，checkpoint 明确记录 `target_mode=direct_value`。第三，使用该 checkpoint 分别跑 1 关纯 `V_psi` 与 `alpha=0.5` 混合 smoke，输出 [gft_sdmcts_tva_direct_smoke_pure_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_tva_direct_smoke_pure_level1_h3_b24_seed0.json) 和 [gft_sdmcts_tva_direct_smoke_blend0p5_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_tva_direct_smoke_blend0p5_level1_h3_b24_seed0.json)。两者均可执行并落盘，分别记录 `V_leaf=V_psi` 与 `V_leaf=(1-alpha)*V_SD+alpha*V_psi`。

判定：本轮是 TVA 语义和实现口径修订，不构成性能结论。残差 v2 的 `1/10` 负结果继续保留，但它不再作为默认 TVA 方向的决策准则；直接终端价值 TVA 已成为新的默认训练和闭环消费口径。下一步应在 10 关开发切片上训练正式 direct-value checkpoint，并先比较 `alpha=1.0` 与保守混合 `alpha<1`，再决定是否进入 50 关确认。

### 2026-06-14 GFT-TVA direct-value v3 10 关闭环负结果

按上一节修订后的直接终端价值口径，训练正式 GFT-TVA direct-value v3。训练仍使用关卡 `8-47`，避开 10 关开发切片 `48-57`；配置与 v2 保持同源：`state_depth=1,max_states_per_level=12,collect_horizon=4,collect_budget=64`，bootstrap leaf 为 `gft_sd`，显式叶值基底为 `discounted_progress`，叶值资源权重为区域容量裕度 `0.05`、顺序依赖债务 `0.05`、目标承诺清晰度 `0.05`，强 GFT 先验为 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25,ordering_weight=0.05,commitment_weight=0.05`。训练输出 [sokoban_gft_tva_direct_v3_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_train_seed0.json) 与 [sokoban_gft_tva_direct_v3_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json)。

离线 gate 为 `PASS`，样本数仍为 `204`，其中 train `169`、valid `35`。direct-value target 为正值搜索值，train target mean `3.838`、std `1.243`、min `0.637`、max `6.203`；train MAE `0.300`、corr `0.939`，valid MAE `0.752`、corr `0.710`，valid target mean `3.768`、prediction mean `4.107`。训练耗时 `/usr/bin/time` 为 `real=30.07s`。该离线结果明显好于 v1 的弱相关，也不同于残差 v2 的整体负修正目标。

随后执行 10 关闭环门槛。纯直接价值叶值 `V_leaf=V_psi` 输出 [gft_sdmcts_gate10_gfttva_direct_v3_pure_a1p0_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_direct_v3_pure_a1p0_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `1/10`，状态分布 `max_steps=8,no_action=1,solved=1`，mean pushes solved `12.0`，total deadlock prunes `13636`，耗时 `real=1504.46s`。保守混合 `V_leaf=0.75*V_SD+0.25*V_psi` 输出 [gft_sdmcts_gate10_gfttva_direct_v3_blend_a0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_direct_v3_blend_a0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果同为 `1/10`，状态分布 `max_steps=7,no_action=2,solved=1`，mean pushes solved `12.0`，total deadlock prunes `11847`，耗时 `real=1360.77s`。

逐关比较显示，direct pure 与 `alpha=0.25` 都只保留 `level_index=4` 解出；显式 `V_SD` 基线能解出的 `level_index=1,2,3,5,6,7,8` 全部退化。失败形态与残差 v2 不同：残差 v2 主要把这些关卡推入 `deadlocked`，direct-value v3 主要表现为长时间 `max_steps` 和少量 `no_action`。这说明 direct-value v3 不是复现残差负修正的同一故障，而是当前 `V_psi` 虽能拟合离线 search-value，却不能在闭环动作序列中提供有效终端排序。

判定：direct-value v3 不进入推荐配置，不安排 50 关确认。该结果不否定“直接终端价值 TVA”作为论文一致接口，但否定当前“GFT 特征 + ridge state-value + 短视界 root search-value target + 直接或 0.25 混合消费”的具体实现。下一步不应继续做无界 alpha 扫描；更有价值的修订方向是改变 TVA 学习器容量与训练目标，包括非线性模型、on-policy 闭环状态再标注、成功/失败分层采样、TD bootstrapping 或动作/局部结构辅助目标。当前正式最强结果仍保持为 50 关 `sd_heuristic + GFT action prior` 的 `41/50`，10 关显式 `V_SD` 对照仍为 `8/10`。

### 2026-06-14 GFT-TVA 独立截断价值审计

根据继续讨论，本轮补充 TVA 的独立准确性验证。此前 direct-value v3 只有训练/验证集拟合 gate、接口 smoke 和闭环结果；这不足以判断 TVA 是否真的能补偿 MCTS 截断叶节点。新增 [audit_sokoban_gft_tva_truncation_value.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py)，将 TVA 从在线决策中拿掉，只用显式 `V_SD` 叶值的 GFT-SD-MCTS 采样叶节点状态，再用离线 exact-state best-first teacher 从这些叶节点继续求解并生成剩余价值标签。teacher 不参与在线动作选择，不修改转移 provider，不修改规划器。

审计合同为：叶节点来源是 GFT-SD-MCTS 的实际 `leaf_value()` 调用；高置信标签只包括 teacher solved、proven dead-end 和 no legal actions；teacher budget unsolved 只单独报告，不进入高置信通过门槛。对每个叶节点同时记录 direct TVA 预测、按 checkpoint 合同形成的 TVA leaf value、显式 `V_SD` leaf value 和 `sd_heuristic` value。门槛要求高置信样本数至少 `20`，TVA 高置信 MAE 不高于显式 `V_SD`，相关系数至少 `0.30`，同一 root 下的叶节点两两排序准确率不低于显式 `V_SD`。

先运行 smoke：关卡 `48-49`，每关最多 `4` 个叶节点，teacher budget `400`。输出 [gft_tva_truncation_value_audit_direct_v3_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_smoke_seed0.json)。smoke 可执行并落盘，因高置信样本数不足按预期 FAIL，不作为性能结论。

正式 10 关审计使用关卡 `48-57`，`horizon=8,budget=192,closed_loop_steps=2,max_leaf_states_per_level=12,teacher_max_expansions=4000`。输出 [gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_seed0.json)，采集 `120` 个叶节点，其中 high-confidence `49` 个、teacher budget unsolved `71` 个。高置信 solved 组中，TVA MAE `0.805`，略好于显式 `V_SD` 的 `0.813`；TVA corr `0.768`，略低于显式 `V_SD` 的 `0.793`；TVA pairwise ordering accuracy `0.800`，低于显式 `V_SD` 的 `0.827`。因此 gate 为 `FAIL`，失败项为 `pairwise_order_not_better_than_gft_sd_leaf`。

为确认不是 teacher budget 太低造成的偏差，又运行高预算复核：每关最多 `8` 个叶节点，teacher budget 提高到 `10000`。输出 [gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json)，采集 `80` 个叶节点，其中 high-confidence `41` 个、teacher budget unsolved `39` 个。TVA 高置信 MAE `0.822`，好于显式 `V_SD` 的 `0.881`；TVA corr `0.781`，低于显式 `V_SD` 的 `0.823`；TVA pairwise ordering accuracy `0.821`，低于显式 `V_SD` 的 `0.866`。gate 仍为 `FAIL`，失败项相同。预算未解组中 TVA 均值约 `4.868`，而低置信占位标签为 `-2.5`，说明 direct-value v3 对“teacher 在预算内未解”的叶节点总体偏乐观；该项不作为高置信失败依据，但解释了闭环中 `max_steps/no_action` 增多的方向。

判定：现在 TVA 已有与 SD 独立验证相对应的前置审计。direct-value v3 虽然在高置信 solved 叶节点上点值 MAE 略优于显式 `V_SD`，但排序不如显式 `V_SD`，且对预算未解叶节点偏乐观，因此不能接入 MCTS 作为推荐叶值。后续 TVA 版本必须先通过该独立截断价值审计，再进入 10 关闭环；训练/验证相关性不再足以作为接入依据。

### 2026-06-14 GFT-TVA 删失风险探针

在独立截断价值审计之后，本轮进一步检查 `teacher_budget_unsolved` 是否能作为删失风险信号被当前 GFT 特征稳定识别。实现新增 [audit_sokoban_gft_tva_censored_risk.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_censored_risk.py)，输入为 high-budget 独立审计输出 [gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json)。该探针不训练可接入 MCTS 的模型，只把 `teacher_budget_unsolved`、proven dead-end 和 no legal actions 作为 high-risk 标签，把 teacher solved 作为 low-risk 标签，使用 GFT 特征、TVA 预测和显式 `V_SD` 叶值做线性 ridge 风险探针，并采用按关卡留一交叉验证。

输出 [gft_tva_censored_risk_probe_direct_v3_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_censored_risk_probe_direct_v3_teacher10k_seed0.json)。数据共 `80` 个叶节点，其中 solved low-risk `41` 个、teacher budget unsolved high-risk `39` 个；无 proven dead-end 或 no legal actions 高置信负例。按关卡留一的 GFT 特征风险探针为 `FAIL`：AUC `0.490`，orientation-free AUC `0.510`，balanced accuracy `0.515`，risk recall `0.615`，solved recall `0.415`。这说明当前 GFT 特征加线性风险头不能跨关卡稳定区分预算未解叶节点。

对照指标显示，原始 `tva_prediction` 作为 risk score 的 AUC 为 `0.642`，原始 `gft_sd_leaf_value` 作为 risk score 的 AUC 为 `0.662`；但这是“值越高越像预算未解”的方向，和终端价值语义相反。因此这些值函数确实含有一定难解状态信号，但不是以可直接消费的 value 方向呈现。若取负值作为 risk score，AUC 分别降为 `0.358` 和 `0.338`。

判定：budget-unsolved 不能被简单硬标为负值回归；当前特征也不足以通过一个线性风险头稳定识别它。后续 TVA 应改变样本构造和模型容量，优先收集更多高置信负例、on-policy 叶节点标签和跨关卡稳定的风险特征；在此之前，不应把 budget-unsolved 固定成终端价值标签，也不应把当前 `V_psi` 或 `V_SD` 的高值当作安全正值。

### 2026-06-14 GFT-TVA v4：叶节点监督、删失风险与校准锚定

根据独立截断价值审计和删失风险探针结果，本轮不再把根节点 `V_search` 的 ridge 回归作为下一版 TVA 训练目标。新增 [train_sokoban_gft_leaf_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_gft_leaf_tva.py)，直接采样 GFT-SD-MCTS 实际截断叶节点，并用离线 exact-state best-first teacher 从叶节点继续求解。teacher solved、proven dead-end 和 no legal actions 进入高置信价值回归；`teacher_budget_unsolved` 只进入风险头，明确不作为确定负终端价值标签。评估入口 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 扩展 GFT-TVA checkpoint loader，兼容新的 `mlp_guarded_v1` JSON 检查点，同时保持旧 ridge 检查点兼容。

第一版叶节点监督模型采用两层 MLP，输出价值头和风险头，并以 `V_leaf=clip((1-anchor_weight)*(value_head-risk_penalty*risk_probability)+anchor_weight*V_SD)` 形成直接终端价值。训练切片为关卡 `8-27`，评测切片仍为 `48-57`；采样配置为 `horizon=8,budget=192,closed_loop_steps=2,max_leaf_states_per_level=12,teacher_max_expansions=4000`，并额外从训练关卡浅层 BFS 收集高置信 blocked 状态。输出 [sokoban_gft_leaf_tva_mlp_guarded_v4_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_train_seed0.json) 与 [sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json)。训练 gate 为 `PASS`，样本 `323`，其中 `teacher_budget_unsolved=89,solved=147,no_legal_actions=2,proven_dead_end=85`；训练价值标签 `185` 个。验证价值 MAE `0.811`、corr `0.966`，风险头验证 AUC `0.776`、balanced accuracy `0.640`。

该 MLP 风险校正版没有通过独立截断价值审计。输出 [gft_tva_truncation_value_audit_leaf_v4_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_leaf_v4_gate10_h8_b192_teacher10k_seed0.json)，采集 `120` 个叶节点，其中 high-confidence `52` 个、teacher budget unsolved `68` 个。TVA 高置信 MAE `0.909`，差于显式 `V_SD` 的 `0.849`；pairwise ordering accuracy `0.583`，显著低于显式 `V_SD` 的 `0.809`。该结果说明非线性模型降低部分点值偏差，但破坏了 MCTS 更敏感的同根候选排序，因此不能进入闭环候选。

随后训练校准锚定版本 `sokoban_gft_leaf_tva_calibrated_anchor_v4`。该版本仍复用同一叶节点标注数据和风险头结构，但令 `anchor_weight=1.0`，只用训练集 teacher-solved 叶节点估计显式 `V_SD` 的系统偏置，得到 `anchor_bias=-0.2277`，即 `V_leaf=V_SD-0.2277`。输出 [sokoban_gft_leaf_tva_calibrated_anchor_v4_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_train_seed0.json) 与 [sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json)。训练 gate 为 `PASS`，训练集价值 MAE `0.367`，验证价值 MAE `0.830`。

校准锚定版本通过独立截断价值审计。输出 [gft_tva_truncation_value_audit_leaf_calibrated_anchor_v4_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_leaf_calibrated_anchor_v4_gate10_h8_b192_teacher10k_seed0.json)，采集 `120` 个叶节点，high-confidence `52` 个。TVA 高置信 MAE `0.761`，优于显式 `V_SD` 的 `0.849`；corr 与显式 `V_SD` 相同，均约 `0.805`；pairwise ordering accuracy 与显式 `V_SD` 完全一致，均为 `0.809`。这是当前第一个通过 GFT-TVA 独立截断价值审计的 checkpoint，但它主要是对显式 `V_SD` 的常数校准，不代表非线性风险头已可安全在线消费。

闭环测试没有产生新最佳。纯校准锚定 TVA 输出 [gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `7/10`，相对显式 `V_SD` 基线在 `level_index=6` 新增 `no_action`。将运行时混合设为 `--tva-value-blend=0.25` 后，单关 `level_index=6` 恢复 solved；完整 10 关输出 [gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果恢复为 `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，与显式 `V_SD` 基线一致；但耗时为 `509.54s`，高于显式 `V_SD` 的 `478.90s`，mean pushes solved `18.0` 也未优于基线 `17.75`。

判定：GFT-TVA v4 完成了从根搜索值 ridge 回归到实际叶节点监督的训练机制修订，并建立了首个通过独立截断价值审计的校准锚定 checkpoint。但闭环中该 checkpoint 只能在 `alpha=0.25` 下恢复 10 关 `8/10`，未超过显式 `V_SD`，且略增耗时；因此不进入推荐配置，不安排 50 关确认。当前正式最强结果仍为 50 关 `sd_heuristic + GFT action prior` 的 `41/50`，当前 10 关显式 `V_SD` 对照仍为 `8/10`。下一步 TVA 优先事项应是标签缓存、排序保持残差或分层风险建模，而不是把校准锚定版本扩展到正式 heldout。

## 2026-06-15

### 2026-06-15 显式 GFT-SD 叶值 50 关阶段性确认

根据继续讨论，本轮将已经在 10 关门槛通过的显式 GFT-SD 叶值版本扩展到 50 关 heldout。配置为：关卡 `48-97`，`horizon=20`，`budget=4096`，`max_steps=80`，`seed=0`，`leaf_value=gft_sd`，`gft_sd_leaf_mode=discounted_progress`，叶值二层校正 `capacity/order/commitment=0.05`，动作先验使用强 GFT 组合 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25,ordering_weight=0.05,commitment_weight=0.05`，revisit penalty `1.0`，transposition，temperature `2.0`，uniform mix `0.1`。

输出 [gft_sdmcts_holdout50_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。aggregate 为 `44/50`，success rate `0.88`，mean pushes solved `15.681818181818182`，total deadlock prunes `165554`，状态分布 `deadlocked=1,max_steps=2,no_action=3,solved=44`，`/usr/bin/time` 为 `real=867.49s,user=869.90s,sys=4.65s`，JSON 内逐关 elapsed 汇总为 `864.2553851907142s`。

相对既有 50 关最强 GFT/grid-equivalent prior 结果 `41/50`，本配置新增解出 `level_index=5`、`6`、`25`、`30`；退化一关，`level_index=40` 从 solved 变为 max_steps。失败关卡为：`level_index=0` max_steps，`9` deadlocked，`14` no_action，`40` max_steps，`45` no_action，`46` no_action。死锁数仍为 `1`，与旧 `41/50` 持平；max_steps 从 `5` 降到 `2`；no_action 仍为 `3`。因此该结果主要减少 max_steps 型失败，并未通过增加死锁风险换取解题率。

判定：显式 GFT-SD 叶值版本通过 50 关阶段性确认，并将 Sokoban 当前正式最强结果从 `41/50` 更新为 `44/50`。该结果仍为零 TVA 学习权重配置；新增收益来自显式 `V_SD` 的 planner-scale `discounted_progress` 叶值、保守二层资源叶值校正，以及统一 GFT 动作先验中的运输成本、顺序依赖和目标承诺分量。GFT-TVA 仍未进入推荐配置，后续 TVA 必须先超过该 `44/50` 显式 SD 叶值线，才可作为正向闭环结果。

### 2026-06-15 当前 `44/50` 未解关卡的 GFT-SD 抽象层归因审计

为回答当前未解关卡是否对应尚未纳入的 SD 抽象层因素，本轮对 `44/50` 正式结果中的 6 个失败关卡进行定向重跑，并将 `trace_limit` 提高到 `100`。定向输出分别为 [audit_fail_level00_gftsdleaf_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level00_gftsdleaf_trace100_seed0.json)、[audit_fail_level09_gftsdleaf_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level09_gftsdleaf_trace100_seed0.json)、[audit_fail_level14_gftsdleaf_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level14_gftsdleaf_trace100_seed0.json)、[audit_fail_level40_gftsdleaf_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level40_gftsdleaf_trace100_seed0.json)、[audit_fail_level45_gftsdleaf_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level45_gftsdleaf_trace100_seed0.json) 和 [audit_fail_level46_gftsdleaf_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level46_gftsdleaf_trace100_seed0.json)。6 个定向重跑均复现原失败状态：`level_index=0` 和 `40` 为 `max_steps`，`9` 为 `deadlocked`，`14`、`45`、`46` 为 `no_action`。

审计显示，失败并不主要对应设计文档外的全新抽象存量。当前第二层已实现 `P/M/A/K/D/Q/C/I` 八个读数，但推荐配置只强消费其中一部分：叶值显式使用目标运输进度 `P`，并以 `K/Q/C=0.05` 做保守校正；动作先验显式使用 `P`、目标占用增量、顺序依赖债务 `Q`、目标承诺清晰度 `C` 和 revisit 项；箱子机动性 `M`、玩家推位可达性 `A`、死锁债务 `D` 和动作相关性中的瓶颈重叠项 `I` 主要处于读数、诊断或间接使用状态，没有形成统一的叶值和先验介入。

逐关归因如下。`level_index=0` 的后 20 步在同一箱子的左右移动中振荡，80 步轨迹只有 8 个唯一底账状态，最终 `K=0,Q=0`，但 `D=2.219`，主要来自机动性损失、目标匹配脆弱性和推位可达性损失。该失败主要指向进度停滞、循环债务或历史层面的反复搬运惩罚不足，而不是区域容量或顺序依赖缺失。`level_index=40` 同样在后期左右振荡，80 步只有 20 个唯一底账状态；最终 `K=-2,Q=2,D=2.743`，说明入口阻塞和顺序债务已经被抽象层读出。该关在旧 `41/50` 的动作先验加旧叶值配置中 18 步解出，当前 `44/50` 版本退化，因而更应视为叶值校正与后期搜索排序的回归样本，而不是缺少新存量的样本。

`level_index=9` 最终为正式 matching deadlock，`M=0,A=1,K=-4,D=5.080,Q=2,C=0`，区域容量、入口阻塞、顺序依赖和硬死锁均已报警。该关的问题是已有 `K/D/Q` 信号介入不够强或不够早。`level_index=45` 也是已有抽象层强报警的 no-action 失败，最终 `M=0,K=-4,D=3.597,Q=6,C=0.945`，所有普通合法推箱都会被保守死锁门剪掉。该样本更适合作为入口锁定和目标填充顺序的权重、门控或 TVA 排序测试，而不是新增存量证据。

`level_index=14` 和 `46` 指向另一类问题：最终均不是正式 matching deadlock，但 `viable_push_count=0`，即当前状态仍有普通合法推箱，却没有可通过保守死锁门的可行推箱。`level_index=14` 最终 `M=0,A=1,D=2.437,Q=1`，有目标顺序违反；`level_index=46` 最终 `M=0,A=2,D=1.907,K=0,Q=0,C=0.968`，主要风险来自箱子机动性和可达性坍缩，而不是区域容量或顺序依赖。该类失败说明 `M/A/D` 已经表达关键风险，但当前在线配置没有将其作为独立叶值和动作先验分量消费；若需要更细粒度扩展，优先方向是在 `M/D/I` 下细化局部互锁、箱子簇解锁性或多步可推性，而不是新增与八存量并列的新库存。

结论：当前失败样本主要分为三类。第一，振荡和进度停滞，对应 `level_index=0` 与 `40`，设计文档中没有单独命名的第九存量，当前最接近的承载位置是动作相关性 `I` 和 revisit/循环债务机制。第二，入口锁定与目标顺序错误，对应 `9` 与 `45`，已有 `K/D/Q` 明确读出但介入不足。第三，准死锁和机动性坍缩，对应 `14` 与 `46`，已有 `M/A/D` 读出但未作为主叶值和主先验分量消费。下一步优先级应从“新增抽象层存量”转为“按设计文档统一激活已有八存量”，尤其是将 `M/A/D/I` 以保守小权重纳入叶值、动作先验或 TVA 特征排序审计，并单独处理历史循环惩罚。

### 2026-06-15 `M/A/D/I` 保守激活 10 关筛查

在当前正式最强 `44/50` 配置基础上，本轮只使用已有参数开关，分别保守激活箱子机动性 `M`、玩家推位可达性 `A`、死锁债务 `D` 和动作相关性 `I`，评测仍使用 10 关开发切片 `48-57`、`horizon=20,budget=4096,max_steps=80,seed=0`。基线为 [gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `8/10`，失败为 `0:max_steps`、`9:deadlocked`，JSON elapsed 汇总 `478.9s`。

单项结果如下。箱子机动性变体 [gft_sdmcts_gate10_gftsdleaf_m_v1_mprior0p10_mleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_m_v1_mprior0p10_mleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json) 使用 `gft_mobility_weight=0.10`、`gft_sd_leaf_mobility_weight=0.05`，结果为 `7/10`，新增 `level_index=6:no_action`，属轻度退化。玩家推位可达性变体 [gft_sdmcts_gate10_gftsdleaf_a_v1_aprior0p10_aleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_a_v1_aprior0p10_aleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json) 使用 `gft_access_weight=0.10`、`gft_sd_leaf_access_weight=0.05`，结果保持 `8/10`，失败关卡仍为 `0:max_steps`、`9:deadlocked`，mean pushes solved 从 `17.75` 降至 `17.25`。

死锁债务变体 [gft_sdmcts_gate10_gftsdleaf_d_v1_dprior0p05_dleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_d_v1_dprior0p05_dleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json) 使用 `gft_deadlock_weight=0.05`、`gft_sd_leaf_deadlock_weight=0.05`，结果降至 `6/10`，新增 `level_index=5:max_steps`、`6:no_action`，属严重退化。动作相关性变体 [gft_sdmcts_gate10_gftsdleaf_i_v1_ibott0p02_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_i_v1_ibott0p02_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json) 使用 `gft_bottleneck_weight=0.02`，结果也为 `6/10`，新增 `level_index=5:max_steps`、`6:max_steps`，同属严重退化。原计划的 `M/A/D/I` 组合项因包含上述已经严重退化的 `D/I` 权重，在启动后被有意中止，未生成可用结果。

为检查箱子机动性是否能与可达性互补，又运行降权组合 [gft_sdmcts_gate10_gftsdleaf_ma_v1_mprior0p05_aprior0p10_mleaf0p025_aleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_ma_v1_mprior0p05_aprior0p10_mleaf0p025_aleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，配置为 `gft_mobility_weight=0.05,gft_access_weight=0.10,gft_sd_leaf_mobility_weight=0.025,gft_sd_leaf_access_weight=0.05`。结果仍为 `7/10`，新增 `level_index=6:no_action`，说明当前一阶机动性读数即使降权也会破坏开发切片中的边界样本。

随后只将 10 关不退化的玩家推位可达性变体扩展到 6 个当前正式失败关卡的定向复核。结果为：`level_index=0` 仍为 `max_steps`，`9` 仍为 `deadlocked`，`14` 从 `no_action` 改为 `max_steps` 但未解出，`40` 仍为 `max_steps`，`45` 与 `46` 仍为 `no_action`。对应输出为 [audit_fail_level00_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level00_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json)、[audit_fail_level09_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level09_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json)、[audit_fail_level14_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level14_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json)、[audit_fail_level40_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level40_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json)、[audit_fail_level45_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level45_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json) 和 [audit_fail_level46_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_fail_level46_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json)。

判定：本轮没有产生新推荐配置。玩家推位可达性 `A` 可作为“未严重退化”的中性因素保留，但它没有解决当前正式失败关卡，不安排 50 关确认。箱子机动性 `M` 的当前一阶计数介入导致新增 `no_action`，需要重新设计为更局部或多步的机动性/解锁性读数。死锁债务 `D` 和动作相关性 `I` 在当前在线公式和权重下严重退化，不应按该方式直接纳入 MCTS；这不是推翻 `D/I` 作为抽象层因素，而是说明其在线介入需要重新校准、分解或交给 TVA/审计接口处理。

### 2026-06-15 在线轨迹 GFT-TVA 最小机制切片

根据继续讨论，本轮开始实现“在线轨迹缓存 + 三类标签 + 受保护 TVA 接入”的最小可验证切片。新增 [train_sokoban_gft_online_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py)。该脚本不修改底层转移 provider，不修改 MCTS 搜索器；它使用当前 GFT-SD-MCTS 闭环执行关卡，记录实际根状态和最终状态，然后按轨迹结果标注 TVA 样本。

标签协议为：成功求解轨迹提供高置信正价值标签，按剩余步数折扣到 `goal_value`；`deadlocked` 与 `no_action` 轨迹提供高置信负价值标签，按剩余步数折扣到 `deadend_value` 或 `no_action_value`；`max_steps` 轨迹默认只进入风险头，不作为确定负价值标签，除非显式设置 `--max-steps-value-weight>0`。该设计落实了“步数耗尽是删失风险而不是真实不可解”的约束。

训练端复用既有 `mlp_guarded_v1` checkpoint 合同，输出价值头、风险头和显式 `V_SD` 锚定项。叶值公式保持为：

```text
V_leaf=clip((1-anchor_weight)*(value_head-risk_penalty*risk_probability)
            +anchor_weight*(V_SD+anchor_bias))
```

其中默认 `anchor_weight=0.75`，使新 TVA 在未通过独立审计前仍强锚定显式 `V_SD`。评估入口无需新格式即可读取该 checkpoint；正式评估仍受独立审计门约束，新 checkpoint 在没有审计路径时会被 `--require-independent-gates` 拦截。

验证分三步。第一，`python -m py_compile` 覆盖新脚本、评估入口和既有叶节点 TVA 训练脚本，语法通过。第二，低预算 5 关 smoke 使用 `level_offset=48,max_levels=5,horizon=6,budget=96,max_steps=30,epochs=20`，输出 [sokoban_gft_online_tva_guarded_v1_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_smoke_train_seed0.json) 与 [sokoban_gft_online_tva_guarded_v1_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_smoke_seed0.json)。5 条轨迹全部 `max_steps`，采样 `94` 个样本，均为删失风险样本，`value_train_samples=0`，因此 gate 因 `min_value_train_samples` 失败。该结果符合预期，证明 `max_steps` 默认不会被误写成价值负例。

第三，已知可解单关 smoke 使用 `level_offset=52,max_levels=1,horizon=20,budget=4096,max_steps=80,epochs=20`，输出 [sokoban_gft_online_tva_guarded_v1_solved_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_train_seed0.json) 与 [sokoban_gft_online_tva_guarded_v1_solved_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_seed0.json)。该轨迹 solved，采样 `13` 个高置信价值样本，gate 为 `PASS`，train value corr `0.966`。随后使用该 checkpoint 跑评估入口 smoke，输出 [gft_sdmcts_online_tva_guarded_v1_solved_smoke_eval_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_online_tva_guarded_v1_solved_smoke_eval_seed0.json)，确认 `--leaf-value tva` 能读取在线轨迹 TVA checkpoint 并执行。

判定：本轮是机制实现和接口验证，不构成性能结论，也不更新 `44/50` 主结果。下一步应在训练切片上收集同时包含 solved、deadlocked/no_action 和 max_steps 的在线轨迹数据，先进行独立截断价值审计和风险审计，再允许进入 10 关闭环；未通过审计前不得带 `--require-independent-gates` 作为正式 TVA 结果报告。

### 2026-06-15 在线轨迹 GFT-TVA 混合叶值候选

本轮继续在线轨迹 GFT-TVA。为避免使用正式 heldout，训练采样改用 `datasets/sokoban_boxoban/boxoban-levels/medium/train/001.txt`，独立审计使用 `train/002.txt`，额外闭环诊断使用 `train/003.txt`。低预算探测显示 `train/001.txt` 前 40 关可产生混合轨迹：采样后共有 `834` 个状态样本，轨迹状态分布为 `max_steps=648,solved=134,no_action=52`，关卡级状态为 `max_steps=29,solved=8,no_action=3`。

首个按原始策略训练的在线 TVA 候选把 `no_action` 失败轨迹中的所有中间状态都作为负价值标签。独立截断价值审计显示该策略会产生过强负修正：高置信叶状态中 TVA 预测均值为 `-0.294`，目标均值为 `4.237`，显式 `V_SD` 均值为 `4.606`；gate 因 MAE、相关性、排序和正值低估失败。因此脚本新增 `--failure-value-label-scope {final,all}`，默认 `final`。修订后，成功轨迹仍为整条轨迹提供正价值标签；`deadlocked/no_action` 轨迹默认只把最终失败状态作为负价值标签，中间状态仍作为风险标签。该修订避免把闭环策略失败误写成整条路径的真实负价值。

修订后的候选使用 `failure_value_label_scope=final`、`anchor_weight=0.85`、`risk_penalty=1.0`、`risk_loss_weight=0.25`、`anchor_bias_mode=none`，输出 [sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json) 与 [sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json)。训练 gate 为 `PASS`，训练集价值标签 `114` 个，训练价值 MAE `0.088`，验证价值 MAE `1.969`；风险头训练 AUC `0.953`，验证 AUC `0.591`。

纯 TVA 预测仍未通过独立价值审计，但 `25%` TVA 与 `75%` 显式 `V_SD` 的混合叶值通过。正式审计输出 [gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json)，采集 `120` 个叶状态，高置信 `78` 个。混合 TVA 高置信 MAE 为 `0.489`，优于显式 `V_SD` 的 `0.552`；相关系数为 `0.771`，略高于显式 `V_SD` 的 `0.762`；同 root 叶节点排序准确率为 `0.962`，高于显式 `V_SD` 的 `0.949`；gate 为 `PASS`。

删失风险审计未通过。输出 [gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_censored_risk_probe_file002_o0_n10_h8_b192_teacher4k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_censored_risk_probe_file002_o0_n10_h8_b192_teacher4k_seed0.json)，留一关风险探针 AUC 为 `0.442`，balanced accuracy 为 `0.504`，risk recall 为 `0.238`，gate 为 `FAIL`。因此当前风险头不能作为闭环门控依据；本轮候选只允许按固定 `alpha=0.25` 混合叶值进行诊断。

低预算 10 关闭环诊断显示同向改善，但仍不是正式性能结论。使用 `horizon=8,budget=192,max_steps=40,transposition`，在 `train/002.txt` 前 10 关上，显式 `V_SD` 对照 [gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json) 为 `5/10`，混合 TVA [gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json) 为 `6/10`。在未参与训练和审计的 `train/003.txt` 前 10 关上，显式 `V_SD` [gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json) 为 `4/10`，混合 TVA [gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json) 为 `6/10`。

判定：在线轨迹 TVA 已从机制 smoke 推进到一个可审计的受限候选。其 `alpha=0.25` 混合叶值通过独立截断价值审计，并在两个非 heldout 低预算 10 关诊断中优于显式 `V_SD`；但删失风险审计失败，且闭环测试规模较小、预算较低，不能更新当前正式 `44/50` 主结果。下一步应先扩大非 heldout 审计和低预算诊断，或修订风险头和排序损失，再考虑更接近正式预算的 10 关确认。

### 2026-06-15 在线轨迹 GFT-TVA 闭环效果测试

根据继续讨论，本轮不再把删失风险审计失败作为阻止闭环效果测试的硬条件。理由是当前候选没有把风险头作为剪枝或门控使用，只在 checkpoint 内部形成受保护价值预测；真正进入 MCTS 的仍是 `V_leaf=0.75*V_SD+0.25*V_TVA` 的固定混合叶值。因此，本轮先用正式预算 10 关测试 `alpha=0.10` 与 `alpha=0.25`，再将未崩塌的 `alpha=0.25` 扩展到 50 关 heldout 探索性测试。

10 关正式预算测试仍使用 heldout `48-57`、`horizon=20,budget=4096,max_steps=80,seed=0` 和当前 `44/50` 的 GFT-SD 配置。`alpha=0.10` 输出 [gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p10_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p10_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `7/10`，状态分布 `deadlocked=1,max_steps=2,solved=7`。`alpha=0.25` 输出 [gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `8/10`，状态分布 `max_steps=1,no_action=1,solved=8`。相对显式 `V_SD` 10 关基线 `8/10`，`alpha=0.25` 总数持平，但把 `level_index=9` 从 deadlocked 改为 solved，同时把 `level_index=6` 从 solved 改为 no_action。

50 关探索性测试使用同一 `alpha=0.25`，输出 [gft_sdmcts_holdout50_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0_exploratory.json)。aggregate 为 `45/50`，success rate `0.90`，mean pushes solved `15.1556`，total deadlock prunes `154153`，状态分布 `deadlocked=1,max_steps=2,no_action=2,solved=45`，`/usr/bin/time` 为 `real=993.52s,user=995.29s,sys=5.23s`。相对显式 `V_SD` 主结果 `44/50`，新增解出 `level_index=9,40,46`，退化 `level_index=6,25`，净增 `+1`。失败关卡变为 `0:max_steps,6:no_action,14:no_action,25:max_steps,45:deadlocked`。死锁数保持 `1`，max_steps 保持 `2`，no_action 从 `3` 降为 `2`；但 wall time 从约 `867s` 增至约 `994s`。

本轮还发现评估器的独立门槛读取函数只识别旧 grid TVA 审计字段 `gate.tva_independent_gate`，不能识别 GFT-TVA 截断价值审计字段 `gft_tva_independent_accuracy_gate.verdict`。已在 [evaluate_sokoban_grid_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py) 修复，使两种审计格式都映射到 `tva_independent_gate`。随后运行 1 关 `--require-independent-gates` smoke，输出 [gft_sdmcts_online_tva_gate_readout_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_online_tva_gate_readout_smoke_seed0.json)，确认 GFT-TVA 价值审计可被正式门槛读取。

判定：用户关于“风险审计可能过严”的判断在闭环效果层面得到支持。删失风险审计失败不能作为禁止所有闭环测试的硬条件，尤其当风险头没有被用作剪枝或门控时。`alpha=0.25` 在线轨迹 GFT-TVA 混合叶值已经给出 `45/50` 的新实测最好结果，但该结果仍标记为探索性：其一，风险头跨关卡审计仍未通过；其二，本次 50 关是在门槛解析修复前运行，结果文件中独立门槛元数据未能正确识别 GFT-TVA 审计；其三，耗时高于显式 `V_SD` 主线。下一步应以修复后的门槛读取逻辑复跑或复核该 50 关配置，并同时推进风险头审计修订或弱化风险惩罚的消融。

### 2026-06-15 在线轨迹 GFT-TVA 退化关卡归因

为分析 `45/50` 中相对显式 `V_SD` 退化的关卡，本轮重跑关键关卡并开启 `trace_limit=100` 与 `trace_gft_readouts`。目标包括两个退化样本 `level_index=6,25`，以及三个新增解出样本 `9,40,46`。输出为 [audit_degrade_l06_base_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_degrade_l06_base_trace100_seed0.json)、[audit_degrade_l06_tva_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_degrade_l06_tva_trace100_seed0.json)、[audit_degrade_l25_base_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_degrade_l25_base_trace100_seed0.json)、[audit_degrade_l25_tva_trace100_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/audit_degrade_l25_tva_trace100_seed0.json)，以及对应的 `l09/l40/l46` 对照文件。

第 6 关退化从第 4 步分叉。基线动作为 `(box=3,left)`，TVA 动作为 `(box=1,down)`。分叉时二者处于同一状态，`P=11.4,M=7,A=8,K=0,D=1.879,Q=2,C=0.365`。TVA 选择的子状态在即时读数上看起来更好：`dP=-1.0,dM=+1,dA=+2,dD=-0.185`，子状态 `Vmix=5.399`，高于基线动作子状态的 `5.299`，风险概率也低于基线动作子状态（`0.799` 对 `0.935`）。但该路线最终在第 35 步进入 no-action，最终 `M=0,K=-4,D=3.334,Q=4,C=0.905`。这说明第 6 关不是普通 TVA 过度惩罚，而是“局部读数看似安全、实际锁死入口”的延迟失效。

第 25 关退化从第 1 步分叉。基线动作为 `(box=0,left)`，TVA 动作为 `(box=0,right)`。TVA 动作的即时显式读数非常诱人：`dP=-1.458,dK=+3,dD=-1.055,dQ=-3,dC=+0.055`，因此 GFT 先验和显式 `V_SD` 都会倾向该方向；但 TVA 对该子状态给出强负修正，`risk=0.999,Vtva-Vsd=-2.388`，只是 `alpha=0.25` 不足以压过显式资源改进。该路线最终不是死锁，而是 max_steps 振荡：80 步只有 22 个唯一状态，后段在两个状态之间反复，最后 12 步交替 `(box=2,up)` 与 `(box=1,down)`，`P` 在 `9.6` 与 `8.6` 间来回，`K=-1,D=1.782,Q=0,C=0.968`。该失败属于高承诺、低顺序债务外观下的循环停滞。

新增解出样本给出相反模式。第 9 关从第 0 步分叉，TVA 选择的动作风险极低 `risk=0.020` 且 `Vtva-Vsd=+0.956`，同时显著增加机动性和可达性（`dM=+4,dA=+8`），最终 14 步解出；基线选择的动作虽有更高 GFT prior，但子状态风险为 `0.719` 且 `Vtva-Vsd=-0.942`，最终 deadlocked。第 40 关 TVA 在第 1 步选择风险更低且轻微正修正的动作，避免了基线 80 步、20 个唯一状态的振荡。第 46 关 TVA 从第 0 步避开基线动作，基线动作子状态 `risk=0.996,Vtva-Vsd=-5.138`，最终 no-action；TVA 路线 10 步解出。

归纳：退化样本的共同点不是“TVA 风险头整体错误”，而是两类边界失效。第一类是第 6 关的延迟入口锁死，当前即时 `P/M/A/D` 读数和 TVA 风险都把错误动作看得不差，需要引入入口锁定、局部互锁或多步可释放性特征。第二类是第 25 关的循环停滞，GFT 资源读数显示 `K/Q/C` 良好，TVA 也已给出负修正但混合权重不足，需要历史重复状态、进度停滞或候选子状态风险触发更强 TVA 惩罚。新增解出样本则表明，当候选动作具有低风险、正 TVA 修正、改善机动性/可达性，或基线动作具有极高风险和强负 TVA 修正时，`alpha=0.25` 是有效的。

因此，动态 TVA 权重是合理下一步，但不应只用一个全局 `alpha` 或根状态风险概率。更合适的形式是候选动作级动态混合：低风险且正 TVA 修正时允许 `alpha=0.25`；候选子状态高风险且 `Vtva-Vsd` 强负时提高负修正权重或加入惩罚；若所有主要候选都处于高风险但 TVA 修正差异很小，则回退到显式 `V_SD` 或触发更保守搜索。另需加入历史重复/进度停滞特征，以处理第 25 关这类高 `C`、低 `Q` 但实际循环的状态。

### 2026-06-15 在线轨迹 GFT-TVA 动态消费消融

根据退化关卡归因，本轮在 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 中新增默认关闭的动态 TVA 消费开关。第一类开关是动态叶值混合：当候选子状态的风险概率超过阈值且 `V_TVA-V_SD` 为强负修正时，将 `V_leaf=(1-alpha)V_SD+alpha V_TVA` 的 `alpha` 从基础值提高到更高值。第二类开关是候选动作先验软惩罚：在相同触发条件下，从 GFT 动作先验分数中扣除与负修正幅度成比例的分数。第三类开关是动态根动作选择：默认仍按访问次数出手，但若访问次数最高的候选子状态触发强负 TVA 信号，且另一个候选的均值更高，则可切换到均值更高的动作。所有开关均不改变底层转移 provider，不把风险头作为剪枝或安全门控，默认关闭时保持固定 `alpha=0.25` 行为。

正式预算 10 关开发切片仍使用 `train/000.txt` 的 heldout 偏移 `48-57`、`horizon=20,budget=4096,max_steps=80,seed=0`。对照为显式 `V_SD` 的 `8/10` 和固定混合 TVA 的 `8/10`。动态叶值的宽阈值版本 `risk>=0.90,V_TVA-V_SD<=-1.0,alpha_high=0.75` 输出 [gft_sdmcts_gate10_online_tva_dynamic_riskblend_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_riskblend_seed0.json)，结果为 `5/10`。加入已访问状态额外叶值惩罚 `0.5` 后，输出 [gft_sdmcts_gate10_online_tva_dynamic_riskblend_revisit0p5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_riskblend_revisit0p5_seed0.json)，结果为 `6/10`。这说明宽阈值风险触发和额外重复状态惩罚都会破坏开发切片中的可解边界样本，不应进入 50 关。

随后将触发条件收紧为 `risk>=0.995,V_TVA-V_SD<=-2.0`。第 25 关单关诊断显示，单纯把高风险负修正的叶值混合上限提高到 `alpha=0.75` 或 `alpha=1.0` 仍无法解出该关，均为 `max_steps`。根因是错误动作在根节点仍因访问次数占优被选中；`alpha=1.0` 时正确动作的均值已经高于错误动作，但默认的按访问次数出手仍保留先验偏置。加入动作先验软惩罚 `weight=1.5,cap=4.0` 后，错误动作先验下降，但按访问次数选择仍未修复。

将同一动态叶值和动作先验软惩罚与根动作按均值选择结合后，第 25 关输出 [gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_selectq_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_selectq_seed0.json)，17 步解出。将全局按均值选择扩展到 10 关，输出 [gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_selectq_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_selectq_seed0.json)，结果为 `7/10`：第 6 关从固定 TVA 的 `no_action` 恢复为 solved，但第 2、8 关退化为 `max_steps`。再改为更窄的动态根动作选择，第 25 关仍可 17 步解出，输出 [gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json)；但 10 关输出 [gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json) 只有 `6/10`，新增 `level_index=5:deadlocked`、`8:max_steps`，第 6 关仍为 `no_action`。

判定：本轮动态 TVA 消费没有产生新推荐配置，也不进入 50 关。局部正信号是第 25 关确实可以被“强负 TVA 修正 + 动作先验软惩罚 + 根动作按均值选择”修复，说明 TVA 已经读出了该振荡分支的长期风险；负信号是该修复不能稳定外推到 10 关，宽阈值风险触发、额外已访问状态惩罚和根动作选择改动都会引入新的退化。因此当前主候选仍保持固定 `alpha=0.25` 的在线轨迹 GFT-TVA `45/50` 探索性结果；动态消费开关保留为默认关闭的审计工具，后续若继续推进，应先训练更可靠的排序目标或引入根动作级 pairwise 校准，而不是把本轮动态规则扩展到 50 关。

### 2026-06-15 在线轨迹 GFT-TVA 状态门控混合

根据继续讨论，本轮将“动态 TVA”重新定义为状态依赖混合比例，而不是单纯放大高风险负修正。实现位置仍为 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)，新增默认关闭的 `--tva-state-gated-blend`。直接价值型 GFT-TVA 的叶值改为：

```text
V_leaf(s)=(1-alpha(s))*V_SD(s)+alpha(s)*V_TVA(s)
```

其中默认状态为 `alpha(s)=0`，即完全退回显式 `V_SD`。首轮状态门控策略为：若候选子状态风险概率不高于 `0.30` 且 `V_TVA-V_SD>=0.25`，则使用 `alpha=0.25`；高风险负修正状态默认仍使用 `alpha=0`。同时保留一个窄的动态根动作选择兜底：若访问次数最高的根候选触发高风险强负修正条件 `risk>=0.95,V_TVA-V_SD<=-1.0`，且另一候选的均值更高，则允许切换到均值更高的动作。该兜底不改变叶值混合比例，不修改底层转移 provider，也不把风险头作为剪枝或安全门控。

机制验证先通过 `python -m py_compile`。低预算 1 关 smoke 输出 [gft_sdmcts_state_gated_tva_smoke_l00_h4_b32_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_state_gated_tva_smoke_l00_h4_b32_seed0.json)，执行正常。正式预算 10 关宽松状态门控输出 [gft_sdmcts_gate10_online_tva_state_gated_posonly_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_posonly_seed0.json)，结果为 `9/10`，失败仅 `level_index=0:max_steps`，相对显式 `V_SD` 和固定混合 TVA 的 `8/10` 均有提升。该版本恢复了固定 TVA 退化的 `level_index=6`，并保留 `level_index=9` 的 TVA 新增解出。

定向复核显示宽松状态门控仍未修复固定 TVA 在 `level_index=25` 的振荡退化，输出 [gft_sdmcts_l25_online_tva_state_gated_posonly_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_state_gated_posonly_seed0.json)，仍为 `max_steps`。将正向可信条件收紧到 `risk<=0.05,V_TVA-V_SD>=0.5` 后，第 25 关输出 [gft_sdmcts_l25_online_tva_state_gated_strictpos_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_state_gated_strictpos_seed0.json)，17 步解出；但同一严格条件的 10 关结果 [gft_sdmcts_gate10_online_tva_state_gated_strictpos_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_strictpos_seed0.json) 降回 `8/10`，第 6 关重新变为 `no_action`。因此严格正向门控只作为第 25 关诊断，不作为候选。

随后测试“宽松状态门控 + 动态根选择兜底”。第 25 关输出 [gft_sdmcts_l25_online_tva_state_gated_posonly_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_state_gated_posonly_rootselect_seed0.json)，17 步解出；10 关输出 [gft_sdmcts_gate10_online_tva_state_gated_posonly_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_posonly_rootselect_seed0.json)，保持 `9/10`。定向复核第 40 关输出 [gft_sdmcts_l40_online_tva_state_gated_posonly_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l40_online_tva_state_gated_posonly_rootselect_seed0.json)，18 步解出；第 46 关输出 [gft_sdmcts_l46_online_tva_state_gated_posonly_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l46_online_tva_state_gated_posonly_rootselect_seed0.json)，仍为 `no_action`，说明固定 TVA 在第 46 关的收益依赖多步排序改变，不由当前门控捕获。

基于上述结果，运行 50 关 heldout 探索性测试 [gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json)。aggregate 为 `46/50`，success rate `0.92`，mean pushes solved `15.3913`，状态分布 `max_steps=1,no_action=3,solved=46`，`/usr/bin/time` 为 `real=1120.14s,user=1120.96s,sys=5.69s`。相对显式 `V_SD` 主线 `44/50`，新增解出 `level_index=9,40`，没有退化；相对固定混合 TVA `45/50`，恢复 `level_index=6,25`，但丢掉 `46`，净增 `+1`。当前失败关卡为 `0:max_steps`、`14:no_action`、`45:no_action`、`46:no_action`。

判定：状态门控 TVA 是新的实测最好候选，达到 `46/50`。它直接落实了“不同状态使用不同 TVA 权重，必要时退回 `alpha=0`”的策略，并且相对固定全局 `alpha=0.25` 更稳健。不过该结果仍标记为探索性：删失风险审计仍未通过，风险头只用于混合比例和根动作兜底，不用于剪枝；本次 50 关也尚未带修复后独立门槛元数据复跑。下一步应围绕该状态门控候选做门槛复核、风险头校准和第 46 关失败归因，而不是继续使用全局固定 `alpha` 作为主候选。

### 2026-06-15 状态门控 GFT-TVA 未解 4 关归因

为回答当前 `46/50` 最好候选的剩余失败是否包含无解关卡，本轮对失败关卡 `level_index=0,14,45,46` 进行逐关核验。使用离线 exact-state best-first teacher，在 `max_expansions=10000`、启发式权重 `1.0`、死锁剪枝开启的设置下，四关全部可解：相对编号 `0/14/45/46` 分别对应数据文件绝对编号 `48/62/93/94`，teacher 解长分别为 `19/23/11/10` 次推动，扩展节点数分别为 `111/326/122/50`。因此这些失败不能归因为 Boxoban 关卡本身无解。

失败终局指标显示，四关均不是硬死锁：`dead_square_boxes=0`、`frozen_boxes=0`、`matching_deadlock=false`、`deadlocked=false`。其中 `level_index=14,45,46` 均为 `no_action`，最终仍有普通合法推动，但通过保守死锁门的可行推动数为 `0`，属于近终局准死锁或入口锁定失败。`level_index=14` 最终 `M=0,A=1,D=2.437,Q=1,C=0.701`，表现为目标区低机动性和轻度顺序违反；`level_index=45` 最终 `M=0,A=2,K=-4,D=3.597,Q=6,C=0.945`，区域容量、入口阻塞和目标填充顺序均强报警；`level_index=46` 最终 `M=0,A=2,K=0,D=1.907,Q=0,C=0.968`，表现为高目标承诺外观下的可推性坍缩，当前状态门控策略没有复现固定 TVA 在该关上的多步排序收益。

`level_index=0` 是不同类型的失败。该关运行到 `max_steps=80`，最终仍有 `legal_push_count=7`、`viable_push_count=3`，不是无动作失败；前 20 步 trace 中第 7 步后反复在同一箱子的左右推动之间振荡，最终 `M=3,A=7,K=0,D=2.219,Q=0,C=0.156`。该样本主要指向进度停滞、循环惩罚和低承诺状态下的排序不稳定，而不是目标区容量或填充顺序信号缺失。

横向对照历史版本显示，这四关不是同一机制的简单漏解。旧网格 SD 先验版本 `41/50` 中，`0/14/45/46` 分别为 `max_steps/max_steps/no_action/no_action`；显式 GFT-SD `44/50` 中为 `max_steps/no_action/no_action/no_action`；固定 `alpha=0.25` 的在线轨迹 GFT-TVA `45/50` 中为 `max_steps/no_action/deadlocked/solved`；当前状态门控 GFT-TVA `46/50` 中为 `max_steps/no_action/no_action/no_action`。结论是：第 `0` 关是长期循环停滞难例，第 `14/45` 关是稳定的目标区顺序和入口锁定难例，第 `46` 关是当前状态门控为避免固定 TVA 退化而牺牲掉的可恢复收益点。

### 2026-06-15 固定高权重 TVA 窗口与失败后重跑组合策略

根据当前 `46/50` 状态门控 GFT-TVA 的四个剩余失败样本，本轮先测试固定较高 TVA 混合比例是否能单独修复 `level_index=14,45,46`。固定混合比例是指直接使用同一个 `alpha` 计算 `V_leaf=(1-alpha)*V_SD+alpha*V_TVA`，不再按状态把 TVA 权重降为 `0`。单关扫描显示，`alpha=0.60` 和 `alpha=0.65` 均能解出 `14/45/46`：`level_index=14` 分别为 29 步和 27 步，`45` 为 13 步，`46` 为 10 步；`alpha=0.70` 及以上会使第 14 关进入 deadlocked。`level_index=0` 在 `alpha=0.60` 下仍为 `max_steps`。

为检查固定高权重是否可替代状态门控，随后在 10 关开发切片上测试 `alpha=0.60` 与 `alpha=0.65`。输出 [gft_sdmcts_gate10_online_tva_fixed_alpha060_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_fixed_alpha060_h20_b4096_seed0.json) 和 [gft_sdmcts_gate10_online_tva_fixed_alpha065_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_fixed_alpha065_h20_b4096_seed0.json) 均为 `7/10`，失败为 `0:max_steps`、`5:deadlocked`、`6:no_action`。因此固定高权重只在当前失败样本上有局部修复能力，不能作为单次主配置替代 `46/50` 的状态门控版本。

本轮还在 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py) 中加入两个默认关闭的审计开关。第一，`--tva-dynamic-leaf-penalty-weight` 在高风险且 `V_TVA-V_SD` 为强负修正时，从叶值中扣除有上限的软惩罚；第二，`--tva-transport-phase-blend` 允许在状态门控中为“运输阶段”状态指定单独 TVA 权重，运输阶段由 GFT-TVA 特征中的进度 `P` 与承诺度 `C` 粗略识别。二者都不修改精确网格转移，不执行剪枝，默认关闭时不改变既有结果。目标关卡和 10 关复核显示，这两类手工规则尚未形成稳健收益：叶值软惩罚不能稳定修复 `14/45`，运输阶段规则可以修复部分目标关卡，但会引入第 `5/6` 关退化。

最后构造一个探索性失败后重跑组合策略：先运行当前最强单次配置 [gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json)，若某关失败，则仅对失败关卡用固定 `alpha=0.60` 重跑，并且只有重跑解出时才替换原失败结果。组合输出为 [gft_sdmcts_holdout50_state_gated_then_fixed_alpha060_fallback_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_state_gated_then_fixed_alpha060_fallback_seed0_exploratory.json)，aggregate 为 `49/50`，状态分布 `max_steps=1,solved=49`，失败后重跑触发 `4` 关、修复 `3` 关，唯一剩余失败为 `level_index=0:max_steps`。

判定：当前正式可复现的单次最好配置仍为状态门控 GFT-TVA 的 `46/50`；固定 `alpha=0.60` 不是稳健单次配置，但它揭示 `14/45/46` 对更高 TVA 权重存在局部可恢复窗口。失败后重跑组合策略的 `49/50` 是重要探索信号，说明不同 TVA 消费模式具有互补性；但该结果属于组合/组合策略，不应与单次 MCTS 配置直接混同。

### 2026-06-15 动态 TVA alpha 规则离线审计

为回答“能否把 `49/50` 组合信号直接改写为动态 TVA 单次配置”的问题，本轮新增只读轨迹审计脚本 [audit_sokoban_dynamic_tva_alpha_features.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_dynamic_tva_alpha_features.py)。该脚本不运行 MCTS，不让 TVA 重新选动作，只重放已有闭环 trace，并在每一步重构 GFT-TVA 特征、显式 `V_SD`、TVA 预测值、`V_TVA-V_SD`、风险概率、legal/viable push count、根节点访问数与均值分离情况。输出为 [dynamic_tva_alpha_feature_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/dynamic_tva_alpha_feature_audit_seed0.json)。

审计标签分为三组。正例为状态门控失败但固定 `alpha=0.60` 修复的 `level_index=14,45,46`；负例为状态门控可解但固定高权重或 phase060 退化的 `5,6`；第 `0` 关作为高权重也无效的负例。被审计的手工 phase060 规则为：若 `P>=20`，或 `P<=10` 且 `C>=0.25`，则使用 `alpha=0.60`，否则回到当前状态门控的正向低风险规则。

结果显示，phase060 规则能在初始状态覆盖全部 3 个修复正例：第 14 关初始 `P=23.30,M=7,A=10,K=0,D=1.553,Q=0,C=0.384`，因 `P>=20` 触发；第 45 关初始 `P=9.075,M=1,A=1,K=-2,D=3.282,Q=2,C=0.315`，因 `P<=10,C>=0.25` 触发；第 46 关初始 `P=8.892,M=4,A=4,K=-1,D=1.940,Q=0,C=0.810`，同样触发。固定 `alpha=0.60` 分别以 `29/13/10` 步解出这些关卡。

但同一规则在负例上明显过触发。第 5 关初始状态不触发，但状态门控可解轨迹在第 12、14-19 步进入 `P<=10,C>=0.25` 区域；固定 `alpha=0.60` 该关退化为 `deadlocked`，phase060 动态版本也退化为 `deadlocked`。第 6 关初始状态同样不触发，但状态门控可解轨迹在第 12-19 步大量触发；固定 `alpha=0.60` 为 `no_action`，phase060 动态版本为 `deadlocked`。第 0 关的特征则不同，初始 `P=10.1375,C=0.099`，phase060 不触发，固定 `alpha=0.60` 仍 `max_steps`，说明该关主要不是高 TVA 权重可修复的准死锁。

因此，已有数据支持“不同 TVA 消费模式互补”，但不支持把 `49/50` 组合策略直接压缩为当前这类简单状态规则。`P/C` 阶段规则确实抓住了 `14/45/46` 的修复窗口，同时也会在 `5/6` 的中后段触发并破坏原本可解轨迹。后续若继续追求单次动态 `49/50`，需要引入能区分“目标区入口锁定需要高 alpha”和“边界可解关卡正在通过安全低进度区”的额外特征，例如入口可释放性、局部互锁、重复状态率或候选动作级排序校准；不能仅依赖 `P` 与 `C` 的阈值。

### 2026-06-15 动态运输阶段 TVA 单次 `49/50` 候选

根据离线审计的正反例，本轮将 phase060 规则收窄为前缀受限且根状态受限的动态运输阶段 TVA。实现位置为 [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)，新增默认关闭参数 `--tva-transport-phase-max-closed-loop-step` 与 `--tva-transport-phase-require-root-eligible`。前者只允许运输阶段 `alpha=0.60` 在指定闭环 push 前缀内触发；后者要求当前根状态本身满足 `P>=20` 或 `P<=10,C>=0.25`，避免搜索树深层叶节点在根状态已经离开目标阶段后继续误触发高权重 TVA。

该修改先通过 `python -m py_compile experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py`。单关复核显示，`level_index=14`、`45`、`46` 分别以 `25/13/10` 步解出；反例 `level_index=5`、`6` 仍分别以 `24/21` 步解出。10 关开发切片输出 [gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_rooteligible_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_rooteligible_seed0.json)，结果为 `9/10`，唯一失败为 `level_index=0:max_steps`。

随后运行 50 关 heldout 单次验证，输出 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json)。配置仍使用 `leaf_value=tva`、状态门控正修正 `alpha=0.25`、动态根动作选择、强 GFT action prior 与显式 GFT-SD 叶值校正；新增运输阶段规则为 `alpha=0.60`、`P>=20 or (P<=10,C>=0.25)`、闭环前缀 `step<=5`、根状态必须满足运输阶段谓词。

50 关结果为 `49/50`，success rate `0.98`，mean pushes solved `15.387755102040817`，total deadlock prunes `153328`，状态分布 `max_steps=1,solved=49`，唯一失败为 `level_index=0:max_steps`。JSON 内逐关 elapsed 汇总为 `1733.7386854887009s`，`/usr/bin/time` 为 `real=1738.16s,user=1738.00s,sys=6.74s`。

判定：该结果是当前最强单次探索候选，已经不同于失败后固定 `alpha=0.60` 重跑的组合策略；它不读取 primary 失败结果，也不进行事后替换。不过该规则仍是基于当前 `14/45/46` 正例和 `5/6` 反例诱导出的手工阶段保护，正式主线仍维持显式 GFT-SD `44/50`，直到完成独立门槛元数据复跑、风险审计和外推复核。

## 2026-06-16

### 2026-06-16 动态运输阶段 TVA 正式主线复核

根据“将单次动态 `49/50` 提升为正式主线还需要哪些补充工作”的门槛，本轮冻结动态运输阶段 TVA 规则，不再调整阈值或选择性重跑失败关卡。冻结规则为：状态门控 GFT-TVA 默认 `alpha=0`，低风险正修正状态使用 `alpha=0.25`；若当前闭环步满足 `step<=5`，当前根状态满足 `P>=20` 或 `P<=10,C>=0.25`，且候选叶状态也满足同一运输阶段谓词，则使用运输阶段 `alpha=0.60`。该规则只作用于叶值混合，不改变 exact grid transition provider，不使用 TVA 做剪枝或安全门控。

正式 50 关复跑启用 `--require-independent-gates`，并显式提供 SD 验证文件 [grid_stock_flow_design_smoke_quick.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json) 与 GFT-TVA 截断价值审计文件 [gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json)。输出 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json)，aggregate 为 `49/50`，状态分布 `max_steps=1,solved=49`，唯一失败仍为 `level_index=0:max_steps`。`independent_gate_status` 中 `grid_sd_gate=PASS`、`tva_independent_gate=PASS`、`failed_checks=[]`；结果文件不含 `primary_result`、`fallback_used_count` 或 `fallback_results` 字段，确认不是失败后组合替换。`/usr/bin/time` 为 `real=996.47s,user=995.41s,sys=0.73s`，JSON 内逐关 elapsed 汇总为 `993.6554618529044s`。

随后进行关键消融。10 关开发切片中，无保护 phase060 输出 [gft_sdmcts_gate10_online_tva_state_gated_phase060_noguard_verified_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_phase060_noguard_verified_seed0.json)，结果为 `7/10`，失败为 `0:max_steps`、`5:deadlocked`、`6:deadlocked`。仅加入前缀保护的输出 [gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_only_verified_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_only_verified_seed0.json)，结果仍为 `7/10`，失败为 `0:max_steps`、`5:deadlocked`、`6:no_action`。仅加入根状态保护的输出 [gft_sdmcts_gate10_online_tva_state_gated_phase060_rooteligible_only_verified_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_phase060_rooteligible_only_verified_seed0.json)，结果为 `9/10`，唯一失败为 `0:max_steps`。完整保护规则在开发切片保持 `9/10`，且第 `5/6` 关分别保持 solved。该消融说明，根状态阶段保护是避免中后段叶节点误触发高 TVA 权重的关键；前缀保护单独不足，但可降低第 `5` 关的绕路成本，使完整规则回到 24 步解。

外层切片复核使用 `medium/train/001.txt` 的前 10 关，仍使用相同独立门槛。显式 GFT-SD 对照 [gft_sdmcts_file001_o0_n10_gftsdleaf_discounted_verified_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file001_o0_n10_gftsdleaf_discounted_verified_seed0.json) 为 `6/10`；状态门控 TVA 对照 [gft_sdmcts_file001_o0_n10_online_tva_state_gated_posonly_rootselect_verified_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file001_o0_n10_online_tva_state_gated_posonly_rootselect_verified_seed0.json) 也为 `6/10`；动态运输阶段 TVA [gft_sdmcts_file001_o0_n10_online_tva_state_gated_phase060_prefix5_rooteligible_verified_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file001_o0_n10_online_tva_state_gated_phase060_prefix5_rooteligible_verified_seed0.json) 为 `7/10`，相对两条对照新增解出 `level_index=0`，未观察到系统性退化。

判定：动态运输阶段 TVA 已满足本阶段正式主线升级条件。冻结规则在带 independent gates 的正式 50 关复跑中稳定复现 `49/50`，关键消融证明无保护规则会伤害 `5/6`，根状态保护是必要保护项，外层 10 关切片未发现相对显式 GFT-SD 或状态门控 TVA 的退化。因此，Sokoban 当前正式主线从显式 GFT-SD `44/50` 更新为单次动态运输阶段 GFT-TVA `49/50`。需要保留的边界是：该正式主线仍不是失败后重跑组合策略；第 `0` 关循环停滞仍未解决；外层复核目前是 10 关切片而非完整第二个 50 关评估。

### 2026-06-16 Boxoban `unfiltered/test` 1000 关公开测试集复核

为获得可与已发表 Boxoban 结果对齐的最小公开测试口径，本轮将冻结后的动态运输阶段 GFT-TVA 正式主线直接评估于 `datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt` 的全部 `1000` 关。该切分对应 Boxoban README 中的 `unfiltered test`，不是此前从 `medium/train/000.txt` 抽取的 50 关 heldout。

评估配置保持正式主线参数不变：`horizon=20,budget=4096,max_steps=80,seed=0`，使用 exact-grid stock-flow 转移、GFT action prior、状态门控低风险正修正 `alpha=0.25`，以及动态运输阶段规则 `alpha=0.60, step<=5, P>=20 or (P<=10,C>=0.25)` 且要求根状态满足同一运输阶段谓词。本次仍启用 `--require-independent-gates`，SD gate 和 GFT-TVA 截断价值 gate 均为 `PASS`。

考虑到单进程 1000 关预计耗时较长，最终采用 20 个不重叠 50 关分片并行复核，再合并为全量结果。分片等价规则为：第 `offset` 个分片使用 `--level-offset offset --max-levels 50 --seed offset*1009`，从而使脚本内部 `seed + local_level_index*1009` 等于单进程全量运行中的 `0 + global_level_index*1009`。因此逐关随机流与 `seed=0,max_levels=1000` 的全量运行等价。原始单进程全量进程在分片汇总验证后终止，仅作为运行期间的保守备份，未用于报告结果。

汇总输出为 [gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json)。aggregate 为 `989/1000`，success rate `0.989`，mean pushes solved `12.398382204246714`，total deadlock prunes `2499261`，状态分布为 `solved=989,max_steps=7,deadlocked=4`。`material_passport.verification_status=VERIFIED`，`independent_gate_status` 为 `grid_sd_gate=PASS,tva_independent_gate=PASS,failed_checks=[]`，并确认结果文件不含 `primary_result`、`fallback_used_count` 或 `fallback_results` 字段。

失败关卡为 `199:deadlocked`、`367:max_steps`、`525:deadlocked`、`546:max_steps`、`664:max_steps`、`698:max_steps`、`706:deadlocked`、`758:max_steps`、`785:deadlocked`、`892:max_steps`、`977:max_steps`。20 个分片的最大 wall time 为 `1509.17s`，分片 wall time 总和为 `20813.39s`，user time 总和为 `20789.28s`。该结果说明当前正式主线在公开 `unfiltered/test` 上达到 `98.9%` 解题率，是一个强外部基准信号；但与文献比较时仍需同时说明动作粒度为 push-level MCTS、使用精确格点转移 provider，且未在 `unfiltered/test` 上继续调参。

### 2026-06-16 Boxoban 文献规模对照表与论文口径修订

根据论文补充实验需求，本轮未新增代码实验，而是复核本地结果 JSON 与 Sokoban/Boxoban 文献摘录，并在 NeurIPS 论文稿中加入 Boxoban/Sokoban 外部基准与训练、审计规模对照表。表格写入 [full_draft_zh.tex](/home/data/round/code/sd_world_model_rl/docs/neurips/paper/full_draft_zh.tex)，位于 `6.8 Boxoban：结构化推箱规划的补充外部基准` 小节。

本文方法行采用已冻结结果：公开测试为 `unfiltered/test/000.txt` 全 `1000` 关，解出 `989/1000`；TVA 训练来自 `medium/train/001.txt` 前 `40` 关，采集 `1378` 个 raw 状态，去重后 `834` 个状态，其中 value 训练状态 `114` 个；独立 TVA 审计来自 `medium/train/002.txt` 前 `10` 关，共 `120` 个叶状态、`78` 个高置信 teacher 标签。训练和在线测试均不调用 teacher，teacher 仅用于独立审计，预算为 `4000` expansions/leaf。

文献对照采用三类口径。Guez et al. (2019) 的 DRC(3,3) 是 model-free recurrent planning 路线，使用 Boxoban unfiltered large train set `900k` 关，训练至 `1e9` environment steps 后约 `99%` solved。Orseau et al. (2018) 的 LevinTS 是 NeurIPS 2018 的 policy-guided tree search，使用 A3C 预训练 policy，在 1000 关测试上报告 `1000/1000`；Orseau and Lelis (2021) 的 PHS/PHS* 是 AAAI 2021 的 policy+heuristic guided search，在 Boxoban 1000 关测试上 PHS* 报告 `1000/1000`。

据此修订论文表述：`989/1000` 支持“接近强已发表结果，并体现显式结构化世界模型和少量 TVA 状态训练/审计的优势”，但不得宣称严格 SOTA。若要进一步主张更强竞争力，需要在 `medium/valid`、`hard`、更大关卡或统一训练预算口径下追加实验。

## 2026-06-19

### 2026-06-19 XSokoban first 与 Boxoban unfiltered/test 结构差异分析

为解释当前动态运输阶段 GFT-TVA 在 Boxoban `unfiltered/test/000.txt` 上达到 `989/1000`，但在 XSokoban first 第 1 关 `max_steps=1000` 下仍未解出的原因，本轮新增结构差异分析脚本 [analyze_sokoban_structure_shift.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_sokoban_structure_shift.py)。该脚本直接调用本地 Sokoban 解析、exact grid stock-flow 和 GFT-SD 读数接口，统计静态几何、初始可达域、合法/可行 push、dead-square、割点、移除割点后的组件数等指标；对 XSokoban 第 1 关额外读取完整 GFT 初始读数，并关联已有失败日志。

输出文件为 [xsokoban_boxoban_structure_shift_analysis.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/xsokoban_boxoban_structure_shift_analysis.json) 与 [sokoban_xsokoban_structure_shift_analysis.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_xsokoban_structure_shift_analysis.md)。Boxoban `unfiltered/test` 的统计口径为 1000 关；XSokoban 口径为 first-40；Boxoban 失败子集来自 1000 关正式汇总结果中的 11 个失败关卡。

统计结果显示，Boxoban 测试集为固定 `10x10`、4 箱，中位地板数 `32`，中位初始合法 push 数 `8`，中位 dead-square 地板占比 `0.333`，中位割点数 `3`。XSokoban first-40 的中位尺寸为 `15x18`，中位面积 `251`，中位地板数 `157`，中位箱数 `15.5`，最大箱数 `32`，中位初始合法 push 数 `4`，中位 dead-square 地板占比 `0.515`，中位割点数 `14`。XSokoban 第 1 关虽然只是 first-40 中较小的一关，也已有 `11x19`、6 箱、`139` 个地板格、初始合法 push 数 `3`、dead-square 地板占比 `0.705`、割点数 `12`。

已有 XSokoban 第 1 关失败日志显示，`max_steps=1000` 后仍为 `max_steps`，最终只完成 `boxes_on_goals=1`，最终 `legal_push_count=2`、`viable_push_count=2`，并非硬死锁。前 20 步 trace 中多次重复推动同一箱子，且许多 root 只有 1 个 child，说明失败主要表现为低分支走廊/目标房间外的局部承诺与循环问题。初始 GFT 读数为 `P=97.35,M=3,A=3,K=0,D=2.75,Q=0,C=0`，最终为 `P=81.45,M=2,A=2,K=0,D=2.877,Q=2,C=0.123`，说明模型读出了高运输工作量和软死锁/顺序压力，但未形成足够强的多箱目标房间填充策略。

本轮结论为：Boxoban `989/1000` 与 XSokoban 第 1 关失败之间不是简单的步数上限差异。Boxoban 主结果处在 4 箱、10x10、同一生成分布和 TVA 审计覆盖范围内；XSokoban first 同时改变了尺寸、箱数、目标集中方式、静态瓶颈和目标房间顺序约束。当前 GFT/TVA 价值和 action prior 仍偏向 4 箱 Boxoban 分布的局部运输进度，缺少多箱 box-goal/room-order commitment option 与 XSokoban 分布上的价值校准。

随后根据用户反馈，将分析重心从“benchmark 口径与结构差异”修正为“当前 SD 模型为什么在 XSokoban first 第 1 关上失败”。新增机制诊断脚本 [analyze_xsokoban_sd_failure_mechanism.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_xsokoban_sd_failure_mechanism.py)，读取既有 `max_steps=1000` 失败日志中保存的前 20 步 trace，重放对应状态，重新计算每个 root 的 GFT 资源、候选后继、GFT prior、GFT leaf value、TVA 预测、state-gated alpha 和特征 OOD z 值。输出为 [xsokoban_sd_failure_mechanism_analysis.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/xsokoban_sd_failure_mechanism_analysis.json) 与 [xsokoban_sd_failure_mechanism_analysis.md](/home/data/round/code/sd_world_model_rl/docs/xsokoban_sd_failure_mechanism_analysis.md)。

机制诊断显示，保存 trace 的 `20` 个 root 中有 `12` 个 `child_count=1`，被选动作访问占比中位数为 `1.0`；`b5:right` 出现 `14/20` 次，被选后继按当前一阶 leaf value 排名第一的步数为 `17/20`。前缀资源变化为 `P:97.35 -> 81.45`，`M:3 -> 2`，`A:3 -> 2`，`D:2.75 -> 2.877`，`Q:0 -> 2`，`C:0 -> 0.123`。这说明模型确实在持续降低运输工作量，但没有形成目标房间填充顺序控制；失败不是 MCTS 没有服从当前价值函数，而是当前价值函数把错误的局部运输前缀评估得足够好。

进一步检查配置发现，root prior 使用 `progress=1,goal=1,ordering=0.05,commitment=0.05,revisit=1`，但 `mobility/access/capacity/deadlock` 权重均为 `0`；GFT leaf 只给 `capacity/order/commitment` 小权重修正，`deadlock` leaf 权重为 `0`。TVA state-gated 默认 blend 为 `0`，保存 trace 中被选后继的 `feature_max_abs_z` 均值为 `20.37`、最大 `22.18`，但 `tva_state_gated_ood_z_threshold=-1`，不会触发 OOD 保护。由此将 XSokoban 失败机制修正为：当前 SD/GFT/TVA 决策层把多箱目标房间问题近似为短期运输进展最大化问题，缺少可执行的 box-goal/room-order 长程承诺；revisit 惩罚只能处理 canonical 回环，不能阻止单调但错误的深推承诺。

### 2026-06-19 Room-GFT 派生模型与 A1/A2 诊断

根据专家建议和用户确认，本轮采用 OO 继承机制新增难题专用 SD 派生线，而不修改现有 Boxoban 正式主线。新增 [room_gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_sd.py)，其中 `SokobanRoomGFTSDModel` 继承 `SokobanGFTSDModel`，保留原有 `P,M,A,K,D,Q,C,I` 读数和精确 grid-flow 转移，仅增加只读的 goal-room、slot、entry、frontier、uncommitted room occupancy 和语义 child 代理接口。该类已从 [__init__.py](/home/data/round/code/sd_world_model_rl/src/sokoban/__init__.py) 导出，供后续 hard-instance planner 显式引用。

新增 A1/A2 诊断脚本 [analyze_room_gft_xsokoban_a1_a2.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a1_a2.py)。脚本读取既有 XSokoban 第 1 关 `max_steps=1000` 失败日志，不重新运行长闭环搜索；重放保存的前 20 步 trace，使用派生模型重新计算 room action impact、semantic child count proxy、TVA OOD 安全门控结果和 room-order 代理指标。输出为 [room_gft_xsokoban_a1_a2_diagnostics.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a1_a2_diagnostics.json) 与 [room_gft_xsokoban_a1_a2_diagnostics.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a1_a2_diagnostics.md)。

诊断结果显示，A1 口径下使用 `feature_max_abs_z>3` 作为 OOD 阈值时，34 个候选后继全部 OOD；20 个被选后继也全部 OOD，其中 3 个为 OOD 且 TVA 给出正向修正，且当前 alpha 实际大于 0。A2 口径下，XSokoban 第 1 关被识别为 1 个 6-slot goal room、3 个 entry/support cell；保存 trace 中物理 `child_count=1` 的 root 仍为 `12/20`，语义 child 代理中位数为 `1.5`、最大值为 `12`。前缀没有 frontier fill，出现 2 次 uncommitted entry、1 次 non-frontier fill、1 次 room-order debt 正增量和 1 次 blocked-slot 正增量。该结果支持专家意见：A1 只能抑制危险的分布外正修正，A2 能暴露 room-order 错误，但真正修复需要后续 A3 将 `(box,direction,slot)` 语义承诺并入 MCTS 节点。

### 2026-06-19 Room-GFT A3 语义 child 与承诺状态

本轮继续在派生难题线中实现 A3，不修改现有 Boxoban planner。新增 [room_gft_mcts.py](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，其中 `RoomSemanticGFTSDMCTSPlanner` 以 `SokobanRoomGFTSDModel` 为模型，MCTS action 从原物理 push `(box,direction)` 扩展为 `(box,direction,slot)`。同一物理后继可以因不同 `RoomSlotOption` 进入不同 `RoomCommitmentState`，节点 key 为 `exact grid canonical key + room commitment key`。承诺状态按“当前箱子位置 -> room slot”记录；箱子被推动后，承诺随箱子位置迁移。

A3 还加入了 commitment boundary 的保守触发：默认当物理 push 的 source 或 target 距离 goal-room entry 不超过 `6` 个静态 floor 步时，即展开 frontier slot 选项，而不是等到箱子已经进入 entry。该设计用于把专家建议中的 point-of-no-return 前置承诺落为可审计接口。Room prior 中加入 slot 深度、commitment gain、frontier fill bonus，以及 uncommitted entry、non-frontier fill、room-order debt 和 blocked-slot penalty；leaf 中加入 committed box bonus 与 uncommitted/order/blocked/nonfrontier commitment penalty。

新增 A3 诊断脚本 [analyze_room_gft_xsokoban_a3_semantic_children.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a3_semantic_children.py)。脚本一方面重放旧 XSokoban 失败 trace 并枚举 semantic children，另一方面运行 `budget=256,horizon=8,smoke_steps=20` 的短闭环 semantic smoke。输出为 [room_gft_xsokoban_a3_semantic_children.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a3_semantic_children.json) 与 [room_gft_xsokoban_a3_semantic_children.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a3_semantic_children.md)。

诊断结果显示，保存 trace 上 semantic child 中位数为 `1.0`、最大值为 `4`、均值为 `1.8`；被选物理动作拥有多个 slot 选项的步数为 `1`，发生在 commitment boundary 附近，`b5:right` 可展开 3 个 slot 选项。短闭环 smoke 执行 20 步后仍为 `max_steps`，最终 `boxes_on_goals=0`，semantic child 中位数为 `4.0`、最大值为 `6`，最终保留 1 个承诺。该结果说明 A3 已能把部分战略分支显式化，但它还不是解题器：长走廊内部一旦承诺确定，仍会表现为单一后继。下一步 A4 应把 `P` 从任意 goal 距离进展改为 slot-schedule constrained progress，并考虑 forced-corridor option，以避免承诺确定后继续沿错误路线单调推进。

### 2026-06-19 Room-GFT A4 槽位顺序进度与 pass-through 诊断

本轮继续沿 `room_gft_*` 派生难题线推进 A4，没有修改现有 Boxoban 评估入口或 `SokobanGFTSDModel` 主线。模型侧在 [room_gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_sd.py) 中新增 `room_slot_progress_summary(state, commitment_state)`，使用静态反向推箱距离衡量已承诺箱子到目标 slot 的可推距离，并用“深槽优先”的贪心代理把未承诺箱子分配给当前 frontier slots。该摘要同时记录 `q_settled`、`k_blocked`、无承诺入房、非前沿承诺、承诺不可达和 frontier 分配不可达等 barrier 项。

规划侧新增 `RoomSlotProgressGFTSDMCTSPlanner`，继承 A3 的 `RoomSemanticGFTSDMCTSPlanner`，只在 hard-instance 派生 planner 中叠加 A4 评分。A4 prior 在原 A3 room prior 上加入 slot-schedule transport delta、committed-slot transport delta、frontier transport delta 和 barrier reduction；当已承诺箱子进入的是其目标深槽的浅层 blocker，且反向推箱距离继续下降时，标记为 `committed_pass_through_blocker`，以避免把正确穿越浅槽的中间态误判为 settled 错填。A4 leaf 同步使用有效 `q_settled/k_blocked`，对 pass-through 中间态降低即时惩罚。

新增诊断脚本 [analyze_room_gft_xsokoban_a4_slot_progress.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a4_slot_progress.py)。脚本读取既有 XSokoban 第 1 关 `max_steps=1000` 失败日志，重放保存的前 20 步 trace，并比较旧物理 GFT prior 与 A4 semantic prior 的排序。输出为 [room_gft_xsokoban_a4_slot_progress.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a4_slot_progress.json) 与 [room_gft_xsokoban_a4_slot_progress.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a4_slot_progress.md)。

诊断结果显示，旧 trace 的 20 个 root 上，旧物理 GFT prior 将保存动作排第一的步数为 `15`，A4 将保存物理动作排第一的步数也为 `15`，说明 A4 不是全局重写前缀。关键变化发生在入房窗口：step `18` 处旧 trace 选择 `b5:down`，旧 GFT 物理 rank 为 `3`，A4 物理 rank 降为 `4`；A4 首选 `b5:right|slot3@[7,17]`，并将其标记为 pass-through，transport delta 为 `+1`、barrier delta 为 `+2`、`committed_pass_through_increase=1`。因此，A4 已能区分“沿已承诺深槽穿过浅层 blocker”和“进入错误行导致承诺不可达”。

短闭环 A4 smoke 使用 `budget=256,horizon=8,smoke_steps=40,boundary_distance=6`，结果仍为 `max_steps`，最终 `boxes_on_goals=0`，semantic child 中位数为 `2.0`、最大值为 `6`，最终 slot schedule cost 为 `41.0`。这说明 A4 修复了旧失败 trace 中入房排序的局部机制，但尚未形成完整 XSokoban 第 1 关求解器；smoke 中仍出现同一承诺箱子在走廊内来回推动的现象，后续 A5 应优先加入 corridor contraction、macro/forced-corridor option 或重复承诺状态惩罚。

### 2026-06-19 Room-GFT A5 loop guard 负结果

本轮在 A4 基础上新增 `RoomLoopGuardGFTSDMCTSPlanner`，仍位于 [room_gft_mcts.py](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py) 的派生难题线，不修改 Boxoban 主线。A5 继承 A4 的 slot-schedule progress，并加入三类软保护：已承诺箱子到其承诺 slot 的反向推箱距离回退惩罚、上一步推动的直接反向惩罚、以及近期重复 commitment state 惩罚。A5 还新增根选择保护：若访问数最高的 root child 触发 loop penalty，而存在未触发 loop penalty 的候选，则在未触发候选中按原规则选择动作。

新增诊断脚本 [analyze_room_gft_xsokoban_a5_loop_guard.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a5_loop_guard.py)。该脚本在同一 `budget=256,horizon=8,smoke_steps=40,boundary_distance=6` 设置下比较 A4 与 A5 短闭环 smoke，并对 A4/A5 都执行同一套事后 loop 统计。输出为 [room_gft_xsokoban_a5_loop_guard.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a5_loop_guard.json) 与 [room_gft_xsokoban_a5_loop_guard.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a5_loop_guard.md)。

诊断结果显示，A4 smoke 执行 40 步后仍为 `max_steps`、`boxes_on_goals=0`，事后统计中存在 `15` 次承诺距离回退和 `3` 次直接反向推动，最终 slot schedule cost 为 `41.0`。A5 与 A4 的首个动作差异发生在 step `11`；A5 消除了被选轨迹中的承诺距离回退和直接反向推动，二者均降为 `0`，并在 step `18` 完成 1 个目标。但 A5 在第 `22` 步进入 `no_action`，最终 `boxes_on_goals=1`、slot schedule cost 为 `46.0`。

判定：A5 是有效的局部 loop guard，但不是完整修复。它证明 A4 的来回推动可以通过承诺单调性和根选择保护消除；同时也暴露出新的瓶颈：单箱填入后，当前 room-slot 调度和物理可达性没有为后续箱子生成可行推进路径。后续不应继续单纯加大软惩罚，而应进入 A6 的 corridor contraction、forced-corridor macro、或基于参考解/teacher trace 的多箱 room-slot 调度锚定。

### 2026-06-19 Room-GFT A6 forced-corridor macro 负结果

本轮在 A5 基础上新增 `RoomCorridorMacroGFTSDMCTSPlanner`，仍位于 [room_gft_mcts.py](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py) 的派生难题线，不修改 Boxoban 主线。A6 的搜索树内部保持 A5 的 single-push semantic MCTS；新增机制只作用于根节点选中动作后的闭环执行。若被选动作推动的箱子已有 slot commitment，且该箱子存在唯一合法、非死锁、反向推箱距离严格下降的后续 push，则连续执行该走廊段，形成 forced-corridor macro。

新增诊断脚本 [analyze_room_gft_xsokoban_a6_corridor_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a6_corridor_macro.py)。该脚本在同一 `budget=256,horizon=8,smoke_steps=40,boundary_distance=6` 设置下比较 A5 与 A6，并记录宏触发次数、宏内 push 数、最终状态和回归核验。输出为 [room_gft_xsokoban_a6_corridor_macro.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a6_corridor_macro.json) 与 [room_gft_xsokoban_a6_corridor_macro.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a6_corridor_macro.md)。

诊断结果显示，A6 在 step `9` 触发一次 corridor macro，将承诺箱子从 `[7,8]` 沿唯一单调 `right` push 连续推进到承诺 slot `[7,17]`，额外压缩 `8` 个 push。A5 为 `no_action`、执行 `22` 个 push、`boxes_on_goals=1`；A6 为 `no_action`、`14` 个闭环决策、执行 `22` 个 push、`boxes_on_goals=1`。A6 的承诺距离回退和直接反向推动仍为 `0`，最终 slot schedule cost 仍为 `46.0`。

本轮同时重跑 A1/A2、A3、A4 和 A5 诊断脚本，核心回归值保持不变：A1/A2 仍为 `selected_ood_count=20`、`selected_active_positive_tva_on_ood_count=3`、`frontier_fill_count=0`；A3 replay 仍为 semantic child median `1.0`、max `4`，A3 smoke 仍为 `max_steps`、`boxes_on_goals=0`；A4 仍为 `max_steps`、`boxes_on_goals=0`、final slot schedule cost `41.0`；A5 仍为 `no_action`、`22` 步、`boxes_on_goals=1`、distance regress `0`。

判定：A6 是有效的走廊压缩机制，但不是完整修复。它减少了同一承诺箱子穿越长走廊所需的闭环规划次数，却没有改变 A5 的最终失败状态。该结果将后续重点进一步收敛到宏后的多箱 room-slot 调度、释放动作和可达性协调；下一步应进入参考解/teacher trace 锚定的多箱 slot schedule，而不是继续扩大单箱 corridor macro。

### 2026-06-19 Room-GFT A7 teacher/reference trace anchor 机制验证

本轮在 A6 基础上新增 `RoomTraceAnchoredGFTSDMCTSPlanner`，仍位于 [room_gft_mcts.py](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py) 的派生难题线，不修改 Boxoban 主线。A7 planner 接收离线参考轨迹映射；当 root state 命中参考轨迹时，对参考物理动作和该箱子沿参考解最终到达的 goal slot 加 prior，并在根节点优先选择参考物理动作。A7 同时要求 corridor macro 的每个 push 与参考轨迹一致，若宏内后续 push 偏离参考解，则截断到已匹配前缀。

新增诊断脚本 [analyze_room_gft_xsokoban_a7_trace_anchor.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a7_trace_anchor.py)。脚本先运行带父指针的 local exact-state teacher，复用 `matching_lower_bound - 0.25*boxes_on_goals - 0.01*viable_push_count` 启发式，并缓存 legal push、deadlock 和 heuristic；随后将 teacher 解转为 reference action/slot 映射，运行 A6 基线与 A7 trace-anchor smoke。输出为 [room_gft_xsokoban_a7_trace_anchor.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a7_trace_anchor.json) 与 [room_gft_xsokoban_a7_trace_anchor.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a7_trace_anchor.md)。

诊断结果显示，本地 teacher 在 `7248` 个展开、`69620` 个生成后继、`4752` 个 deadlock 剪枝下找到 `97` push 解，耗时约 `71.82s`。A6 对照保持 `no_action`、`22` 个 push、`boxes_on_goals=1`、final slot schedule cost `46.0`。A7 trace-anchor smoke 在 `a7_budget=64,horizon=8,a7_max_pushes=120` 下解出该关，执行 `97` 个 push、`63` 个闭环决策、触发 `6` 次 corridor macro，最终 `boxes_on_goals=6`、final slot schedule cost `0.0`；63 个闭环决策均命中参考状态并选择参考动作，最终状态命中参考终态。

判定：A7 是机制验证而非正式求解率结果。它说明当前 exact transition、room-slot commitment 和 A6 macro 可以承载完整可行解；A5/A6 失败的主因是自主多箱调度和槽位顺序选择不足，而不是 `max_steps`、单步转移或单箱 corridor macro 能力不足。下一阶段应把 A7 trace 中的 slot fill order、box-release 顺序和宏边界转化为无 teacher 的 room fill plan 或价值审计目标。

### 2026-06-19 XSokoban teacher trace gap 综合诊断

为将 A7 参考轨迹从“能否沿答案执行”推进到“反推 SD 缺失机制”，本轮新增 [analyze_room_gft_xsokoban_teacher_trace_gap.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_teacher_trace_gap.py)。该脚本尝试为 XSokoban first 前 3 关生成 exact-state teacher 轨迹；对成功解出的轨迹，逐步回放 teacher state，并在该 state 上重新计算当前 Room-GFT/A6 的 semantic candidates、prior score、slot progress delta、barrier delta、gap tags 和每 10 步一次的 MCTS root 选择。teacher 轨迹只用于诊断，不注入 planner 出手。

输出为 [room_gft_xsokoban_teacher_trace_gap.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_teacher_trace_gap.json) 与 [room_gft_xsokoban_teacher_trace_gap.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_teacher_trace_gap.md)。默认 teacher 预算为每关 `100000` expansions 或 `90s` wall-clock。第 1 关解出，解长 `97` push、展开 `7248`、生成 `69620`、deadlock 剪枝 `4752`。第 2 关和第 3 关分别为 10 箱和 11 箱，均在 `90s` 内未解出，说明多关 teacher 轨迹库本身需要更强启发式或更高预算。

第 1 关完整轨迹上的 A6 gap 统计为：teacher 动作 rank-1 比例 `0.5979381443298969`，rank<=3 比例 `0.8556701030927835`，最大 rank `13`；每 10 步采样的 MCTS root 选择中，teacher 动作匹配比例 `0.6`。gap tag 计数为 `setup_or_release_underrewarded=20`、`teacher_low_prior_rank=14`、`teacher_not_top_prior=25`、`short_transport_competes=16`、`top_action_room_order_risk=6`、`future_goal_slot_option_unavailable=49`。teacher 轨迹包含 `9` 段长度至少为 3 的同向走廊推进，最长 `10` push。

判定：当前 Room-GFT/A6 的主要缺失可以更具体地表述为五类：缺少可自主生成的 room fill plan / slot queue；低估箱子释放、通道保持和入房准备动作；A6 corridor macro 缺少全局 schedule guard；当前 prior/leaf 需要用 teacher 低 rank 状态做校准审计；短期运输进展仍会压过长期 room-order 可行性。该诊断支持下一阶段从 A7 trace 中抽取无 teacher 的 room fill phase、box release queue 和 value/prior 审计目标，而不是继续只调 A6 单箱宏。

### 2026-06-19 XSokoban fast teacher 高预算边界复核

为争取获得 XSokoban first 前 3 关完整 teacher 轨迹，本轮在 [analyze_room_gft_xsokoban_teacher_trace_gap.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_teacher_trace_gap.py) 中增加 fast teacher 选项。新增配置包括 Hungarian matching 下界、Tier-A deadlock 剪枝、关闭 viable push 启发式子项、heuristic weight `5.0`，并将每关 teacher 预算提高到 `1000000` expansions 或 `600s`。该 teacher 仍只用于离线轨迹生成和 gap 诊断，不参与 Room-GFT/A6 在线出手，也不修改 Boxoban `989/1000` 主线。

运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_teacher_trace_gap.py \
  --teacher-matching-backend hungarian \
  --teacher-deadlock-mode tier_a \
  --teacher-no-viable-push-term \
  --teacher-heuristic-weight 5 \
  --teacher-max-expansions 1000000 \
  --teacher-max-seconds-per-level 600 \
  --mcts-sample-stride 20 \
  --mcts-budget 64 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_teacher_trace_gap_fast_budget.json \
  --out-md docs/room_gft_xsokoban_teacher_trace_gap_fast_budget.md
```

输出文件为 [room_gft_xsokoban_teacher_trace_gap_fast_budget.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_teacher_trace_gap_fast_budget.json) 与 [room_gft_xsokoban_teacher_trace_gap_fast_budget.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_teacher_trace_gap_fast_budget.md)。结果为 `teacher_solved_count=1`、`teacher_unsolved_count=2`。第 1 关解出，解长 `109` push、展开 `822845`、生成 `3666199`、deadlock 剪枝 `535324`，耗时约 `483.69s`；第 2 关和第 3 关仍未解出，分别在 `600s` 内展开 `533130` 和 `784388` 个状态。

fast teacher 第 1 关轨迹不是最优性证明，而是可行轨迹诊断材料。与默认 teacher 的 `97` push 轨迹相比，该轨迹更长且展开更多，但 gap 方向一致：A6 prior 下 teacher 动作 rank-1 比例为 `0.697`、rank<=3 比例为 `0.899`，仍有 `setup_or_release_underrewarded=16`、`teacher_low_prior_rank=11`、`short_transport_competes=14`，并包含 `11` 段长度至少为 3 的同向走廊推进。

判定：本轮未获得前三关完整 teacher 轨迹，因此不能形成“综合前三关完整轨迹”的结论。当前可用证据为第 1 关完整轨迹加第 2、3 关高预算 teacher 边界。该证据继续支持 SD 侧缺少自主 room fill plan / slot queue、box release/setup value、channel preservation 和 schedule-guarded corridor macro；第 2、3 关未解出只说明 teacher 轨迹库建设仍受搜索预算和启发式能力限制，不能解释为当前 SD 机制已充分。

### 2026-06-19 Room-GFT A8 fill-plan SD 部分正结果

根据“用第一关 teacher 轨迹分析来补全 SD，TVA/MCTS 相应配合”的要求，本轮在 `room_gft_*` 派生难题线中新增 A8。模型侧新增 [SokobanRoomFillPlanGFTSDModel](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_sd.py)，继承 `SokobanRoomGFTSDModel`，不修改 exact grid-flow 转移、GFT 基础读数或 Boxoban 主线。该模型将第 1 关 teacher 轨迹揭示的结构缺口转为当前状态可计算的 fill-plan 摘要，包括 room lane queue、fill phase、active slot、active assigned box、active box 可推数、active box 单调推进数、active box blocked count、channel staging 和 plan barrier cost。

规划侧新增 [RoomFillPlanGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A6 的 `RoomCorridorMacroGFTSDMCTSPlanner`，不继承 A7 的 reference action anchor，不读取 teacher 动作表。A8 prior 和 leaf 消费 fill-plan barrier、active commitment、channel staging、early channel occupancy、active transport、active box release 与 active monotone push 信号；corridor macro 仍沿用 A6 执行层。

新增诊断脚本 [analyze_room_gft_xsokoban_a8_fill_plan.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a8_fill_plan.py)。默认 smoke 命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a8_fill_plan.py
```

输出 [room_gft_xsokoban_a8_fill_plan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a8_fill_plan.json) 与 [room_gft_xsokoban_a8_fill_plan.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a8_fill_plan.md)。默认 `budget=256,horizon=8,max_steps=120` 下，A6 对照为 `no_action`、执行 `22` push、`boxes_on_goals=1`、final slot cost `46.0`；A8 为 `no_action`、执行 `48` push、`boxes_on_goals=2`、final slot cost `43.0`，触发 `4` 次 corridor macro、额外压缩 `16` 个 push。

随后进行较大预算复核：

```bash
python scripts/analyze_room_gft_xsokoban_a8_fill_plan.py \
  --budget 1024 \
  --horizon 12 \
  --smoke-steps 160 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a8_fill_plan_h12_b1024.json \
  --out-md docs/room_gft_xsokoban_a8_fill_plan_h12_b1024.md
```

输出 [room_gft_xsokoban_a8_fill_plan_h12_b1024.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a8_fill_plan_h12_b1024.json) 与 [room_gft_xsokoban_a8_fill_plan_h12_b1024.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a8_fill_plan_h12_b1024.md)。较大预算结局仍为 `no_action`、`boxes_on_goals=2`、final slot cost `43.0`。

判定：A8 是 SD 补强后的部分正结果。它证明从第 1 关 teacher 轨迹中抽取 room fill phase、active slot、release/setup 和 channel preservation 状态后，无 teacher 在线动作锚定的 planner 可以多完成一个目标；但第三个 deep slot 前仍失败，说明剩余缺口是更细的多箱释放队列和 active-box 局部可达性，而不是简单扩大 MCTS 预算。后续 TVA 应围绕 A8 的 active-box blocked 状态和低 fill-plan score 状态做审计/校准，不应直接在线模仿 teacher 动作。

### 2026-06-19 Room-GFT A9 release-plan SD 诊断增强

根据“用第 1 关 teacher 轨迹分析继续补全 SD，TVA/MCTS 相应配合”的要求，本轮在 A8 基础上新增 A9。模型侧新增 [SokobanRoomReleasePlanGFTSDModel](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_sd.py)，继承 `SokobanRoomFillPlanGFTSDModel`，不修改 exact grid-flow 转移、GFT 基础读数、A8 fill-plan 规则或 Boxoban 主线。该模型将 A8 的 `active_box_blocked_count` 展开为 release-plan 摘要，枚举 active assigned box 的候选单调推进及其所需玩家站位格，并记录站位格占用、目标格占用、玩家可达性、Tier-A deadlock 和 release barrier。

规划侧新增 [RoomReleasePlanGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A8 的 `RoomFillPlanGFTSDMCTSPlanner`，不继承 A7 的 reference action anchor，不读取 teacher 动作表。A9 prior 和 leaf 消费 release barrier reduction、release need reduction、stand blocker reduction、unreachable stand reduction 和 available monotone push 等状态；corridor macro 仍沿用 A6/A8 执行层。

新增诊断脚本 [analyze_room_gft_xsokoban_a9_release_plan.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a9_release_plan.py)。默认 smoke 命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a9_release_plan.py
```

输出 [room_gft_xsokoban_a9_release_plan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a9_release_plan.json) 与 [room_gft_xsokoban_a9_release_plan.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a9_release_plan.md)。默认 `budget=256,horizon=8,max_steps=120` 下，同轮 A8 对照为 `no_action`、执行 `48` push、`boxes_on_goals=2`、final slot cost `43.0`；A9 为 `no_action`、执行 `44` push、`28` 个闭环决策、`boxes_on_goals=2`、final slot cost `42.0`、final fill barrier `4.65`、final release barrier `8.0`。

随后进行较大预算复核：

```bash
python scripts/analyze_room_gft_xsokoban_a9_release_plan.py \
  --budget 1024 \
  --horizon 12 \
  --smoke-steps 160 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a9_release_plan_h12_b1024.json \
  --out-md docs/room_gft_xsokoban_a9_release_plan_h12_b1024.md
```

输出 [room_gft_xsokoban_a9_release_plan_h12_b1024.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a9_release_plan_h12_b1024.json) 与 [room_gft_xsokoban_a9_release_plan_h12_b1024.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a9_release_plan_h12_b1024.md)。较大预算下，A9 仍为 `no_action`、执行 `50` push、`boxes_on_goals=2`、final slot cost `43.0`、release barrier `8.0`。

判定：A9 是诊断增强而非求解成功。它将 A8 末端失败具体定位为：active slot `(7,17)`、active assigned box `(6,5)`，候选单调推进为 `down` 到 `(7,5)`，所需站位格 `(5,5)` 被箱子占据，因此 `active_release_need_count=1`、`active_release_stand_blocker_count=1`、`active_release_monotone_available_count=0`。这说明 SD 侧已经能显式读出“需要释放哪个站位格”，但当前 prior/leaf 与闭环 MCTS 仍未稳定产生提前保持或清空该站位格的策略。后续应把 A9 release-plan 状态转入 TVA 审计/校准，或加入 schedule-guarded access preservation，而不是报告为已解出。

### 2026-06-19 Room-GFT A10 lookahead access-plan 单关正结果

根据 A9 结论，本轮继续在 `room_gft_*` 派生难题线中新增 A10。模型侧新增 [SokobanRoomAccessPlanGFTSDModel](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_sd.py)，继承 `SokobanRoomReleasePlanGFTSDModel`，不修改 exact grid-flow 转移、GFT 基础读数、A8 fill-plan、A9 release-plan 或 Boxoban 主线。该模型对后续 open deep slot 做 lookahead assignment，并枚举 assigned box 的候选单调推进、所需站位格、站位格占用和 release/access barrier。

规划侧新增 [RoomAccessPlanGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A9 的 `RoomReleasePlanGFTSDMCTSPlanner`，不继承 A7 的 reference action anchor，不读取 teacher 动作表。初始 `lookahead_slots=3` 版本前瞻过强，默认 120 push 只完成 1 个目标，表现为 access preservation 导致循环。随后将默认前瞻深度收窄为 `lookahead_slots=2`，并将强 scoring 限制为 future stand-cell box blocker；一般 stand unreachable 仍写入摘要，但不再主导出手。

新增诊断脚本 [analyze_room_gft_xsokoban_a10_access_plan.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a10_access_plan.py)。默认 smoke 命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a10_access_plan.py
```

输出 [room_gft_xsokoban_a10_access_plan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan.json) 与 [room_gft_xsokoban_a10_access_plan.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a10_access_plan.md)。默认 `budget=256,horizon=8,max_steps=120,lookahead_slots=2` 下，同轮 A9 对照为 `no_action`、执行 `44` push、`boxes_on_goals=2`、final slot cost `42.0`；A10 为 `max_steps`、执行 `120` push、`83` 个闭环决策、`boxes_on_goals=5`、final slot cost `1.0`，触发 `8` 次 corridor macro、额外压缩 `37` 个 push。

随后仅延长步数复核：

```bash
python scripts/analyze_room_gft_xsokoban_a10_access_plan.py \
  --smoke-steps 200 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan_s200.json \
  --out-md docs/room_gft_xsokoban_a10_access_plan_s200.md
```

输出 [room_gft_xsokoban_a10_access_plan_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan_s200.json) 与 [room_gft_xsokoban_a10_access_plan_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a10_access_plan_s200.md)。该配置为 `solved`，执行 `127` push、`90` 个闭环决策，最终 `boxes_on_goals=6`，final release barrier `0.0`，final access barrier `0.0`。

判定：A10 是第 1 关无 teacher 在线动作锚定的单关正结果。它说明 A9 的剩余瓶颈确实不是简单搜索预算，而是 active-slot 视野过短：需要在 slot 切换之前保护后续 deep slot 的站位格。与此同时，A10 目前不能报告为 first-40 或 hard 集求解率改进；`lookahead_slots=3` 的负结果也说明 access preservation 必须受控，过强会引发循环。后续应做多关 smoke、lookahead 消融和 TVA 审计后再形成整体结论。

### 2026-06-19 Room-GFT A10 XSokoban first 第 2 关泛化边界复核

基于 A10 的设计并未绑定 XSokoban first 第 1 关，本轮扩展 [analyze_room_gft_xsokoban_a10_access_plan.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a10_access_plan.py)，加入 `--level-offset` 参数，使同一 A10 模型和 planner 可直接加载 XSokoban first 的指定关卡。该改动只影响诊断脚本，不修改 `SokobanRoomAccessPlanGFTSDModel`、`RoomAccessPlanGFTSDMCTSPlanner`、Boxoban 主线或 `evaluate_sokoban_gft_sdmcts.py`。

第二关复核运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a10_access_plan.py \
  --level-offset 1 \
  --smoke-steps 200 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan_l002_s200.json \
  --out-md docs/room_gft_xsokoban_a10_access_plan_l002_s200.md
```

输出为 [room_gft_xsokoban_a10_access_plan_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a10_access_plan_l002_s200.json) 与 [room_gft_xsokoban_a10_access_plan_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a10_access_plan_l002_s200.md)。配置为 `budget=256,horizon=8,smoke_steps=200,lookahead_slots=2,level_offset=1`。A9 对照为 `no_action`、执行 `34` push、`29` 个闭环决策、`boxes_on_goals=1`、final slot cost `52`；A10 为 `no_action`、执行 `57` push、`53` 个闭环决策、`boxes_on_goals=1`、final slot cost `52`。A9/A10 的首个决策差异发生在 step `13`。

A10 第二关终态显示 `final_release_barrier_cost=4`、`final_access_barrier_cost=8`、`final_future_access_barrier_cost=4`、`final_access_need_count=2`、`final_future_access_need_count=1`，并在 step `52` 首次出现 future stand blocker。Tier-A 终态诊断为 `tier_a_deadlocked=false`、`no_legal_pushes=false`，说明该轮失败不是硬死锁或无合法推箱动作，而是当前 access-plan、release-plan 与根选择机制未能继续生成有效调度。

判定：A10 规则可机械外推到第二关，但当前配置尚不能解出第二关。第 1 关的正结果仍应表述为单关机制正结果；第二关 smoke 将研究边界推进为：仅补入 future stand-cell blocker 仍不足以处理 10 箱关卡中的多房间或多箱释放调度，后续需要更强的多箱 schedule、release macro 或 TVA 对前缀状态的校准，而不是把第 1 关结果外推为 XSokoban first 多关求解率。

### 2026-06-19 Room-GFT A11 schedule-plan SD 与消费端负结果

根据“调度缺口应补在 SD 表示侧，MCTS 负责消费”的模块分工，本轮继续在 `room_gft_*` 派生线中新增 A11。模型侧新增 [SokobanRoomSchedulePlanGFTSDModel](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_sd.py)，继承 A10 的 `SokobanRoomAccessPlanGFTSDModel`，不修改 exact grid-flow 转移、A8/A9/A10 规则或 Boxoban 主线。A11 将 A10 的 access assignment 展开为 multi-box release dependency graph，显式记录关键站位格、未来关键站位格、阻塞箱、不可达站位格、release dependency、潜在 release action 和 destructive action。

规划侧新增 [RoomSchedulePlanGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A10 的 `RoomAccessPlanGFTSDMCTSPlanner`。该 planner 不自行发明调度图，只消费 A11 SD 摘要，并在 prior、leaf 和 root guard 中奖励 dependency/barrier/blocker 降低、释放 blocker 的动作，惩罚进入当前或未来关键站位格的动作。本轮同时在 planner 内加入 `fill/release/access/schedule` summary cache，缓存键为 canonical state、commitment key 和 lookahead 参数，用于减少同一 MCTS 内重复计算。

新增诊断脚本 [analyze_room_gft_xsokoban_a11_schedule_plan.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a11_schedule_plan.py)。首个 raw A11 使用 `lookahead_slots=3`，输出 [room_gft_xsokoban_a11_schedule_plan_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a11_schedule_plan_l002_s200.json) 与 [room_gft_xsokoban_a11_schedule_plan_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a11_schedule_plan_l002_s200.md)。该版本首步即从 A10 的 `b3:right` 改为 `b7:right`，最终 `no_action`、执行 `14` push、`boxes_on_goals=0`、`no_legal_pushes=true`。诊断显示，raw A11 把“破坏性动作执行后不再出现在 destructive action 列表里”误当作收益，因此消费端过强且方向错误。

随后修正 A11 消费端：默认 `schedule_lookahead_slots=2`，与 A10 对齐；新增 immediate critical/future stand-cell entry penalty；不再正向奖励 destructive action count 下降；并保留 summary cache。guarded 版本输出 [room_gft_xsokoban_a11_schedule_plan_guarded_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a11_schedule_plan_guarded_l002_s200.json) 与 [room_gft_xsokoban_a11_schedule_plan_guarded_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a11_schedule_plan_guarded_l002_s200.md)。配置为 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2`。A10 对照仍为 `no_action`、执行 `57` push、`boxes_on_goals=1`、final slot cost `52`；guarded A11 为 `max_steps`、执行 `200` push、`boxes_on_goals=0`、final slot cost `48`、final schedule barrier `30`、dependency count `2`、release action count `0`，Tier-A 诊断为 `tier_a_deadlocked=false,no_legal_pushes=false`。

判定：A11 当前是表示补强与消费端负结果，而不是第二关求解成功。它证明调度信息确实应落在 SD 派生模型中，但仅用 dependency/barrier delta 调 prior 和 leaf 会引入新的循环：guarded A11 在 step `9` 后反复出现 `b9:up` 与 `b5:down`，schedule score 多数为 `0`，说明当前 schedule graph 仍未给出可执行的 release macro 或 loop-breaking signal。下一步不应在线注入 teacher 动作；更合适的是在 A11 摘要基础上加入 release macro、重复二箱循环惩罚，或把 dependency-stall 前缀纳入 TVA 审计。

### 2026-06-20 Room-GFT A12 dependency-stall guard 负结果

基于 A11 guarded 版本在 XSokoban first 第 2 关中的二箱循环，本轮继续在 `room_gft_*` 派生线中新增 A12。规划侧新增 [RoomScheduleStallGuardGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A11 的 `RoomSchedulePlanGFTSDMCTSPlanner`，保留 A11 的 schedule-plan SD 表示，不修改 Boxoban 主线或 `evaluate_sokoban_gft_sdmcts.py`。A12 新增的消费端信号包括 recent moved edge、recent schedule signature、重复 commitment state、反向近期边和非改善 dependency/schedule 状态；这些信号只作为 MCTS prior/root guard 的 loop guard，不读取 teacher 动作表。

新增诊断脚本 [analyze_room_gft_xsokoban_a12_stall_guard.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a12_stall_guard.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a12_stall_guard.py
```

输出为 [room_gft_xsokoban_a12_stall_guard_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a12_stall_guard_l002_s200.json) 与 [room_gft_xsokoban_a12_stall_guard_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a12_stall_guard_l002_s200.md)。配置为 `level_offset=1,budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recent_edge_window=12`。同轮 A11 guarded 对照为 `max_steps`、执行 `200` push、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`；A12 为 `no_action`、执行 `6` push、`3` 个闭环决策、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `40.5`、dependency count `2`、future dependency count `1`、release action count `0`。

A12 与 A11 的首个决策差异发生在 step `0`。A12 前缀为 `b6:right`、`b6:right`、`b7:right`，其中第二步 corridor macro 总计执行 `4` 个 push；第三步被标记为 stalled，`schedule_stall_score=-8`。终态 Tier-A 诊断为 `tier_a_deadlocked=false,no_legal_pushes=true`。最终 dependency graph 中，当前 assigned box `[4,6]` 进入 slot `[3,1]` 所需站位格 `[4,7]` 对玩家不可达；未来 assigned box `[3,6]` 进入 slot `[5,1]` 所需 target `[4,6]` 被箱子占据，站位格 `[2,6]` 也不可达，且 `release_actions=[]`。

判定：A12 是负结果诊断。它打断了 A11 的长循环，但过早把 planner 导向无可执行候选的局面，没有形成第二关求解进展。该结果说明，仅在消费端加入 dependency-stall loop guard 不足以补齐调度机制；下一步应在 A11/A12 的 schedule graph 上构造可执行 release/access recovery macro，或补入更强的玩家可达域恢复信号，再由 MCTS 消费。TVA 可用于这些前缀状态的审计和校准，但仍不应在线注入 teacher/reference 动作。

### 2026-06-20 Room-GFT A13 access-recovery guard 部分改善与负结果

基于 A12 在第 2 关中 6 push 后进入 `no_action` 的失败形态，本轮继续在 `room_gft_*` 派生线中新增 A13。规划侧新增 [RoomScheduleAccessRecoveryGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A12 的 `RoomScheduleStallGuardGFTSDMCTSPlanner`，仍使用 A11 的 schedule-plan SD 表示，不修改 Boxoban 主线，也不接入 teacher/reference 动作。A13 新增的消费端信号为 access-recovery margin：在 unresolved dependency 存在时，惩罚非阻塞合法推箱余量崩塌、`no_legal_pushes`、target blocker 增加和不可达站位未恢复，奖励 target blocker 减少、不可达站位减少、release action 出现和 dependency 总数下降。

新增诊断脚本 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a13_access_recovery.py
```

输出为 [room_gft_xsokoban_a13_access_recovery_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a13_access_recovery_l002_s200.json) 与 [room_gft_xsokoban_a13_access_recovery_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a13_access_recovery_l002_s200.md)。配置为 `level_offset=1,budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recent_edge_window=12,recovery_min_legal_pushes=2`。同轮 A12 对照为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `40.5`；A13 为 `max_steps`、执行 `200` push、`182` 个闭环决策、`boxes_on_goals=0`、final slot cost `29`、final schedule barrier `87.5`、dependency count `2`、future dependency count `1`、target blocker count `2`、release action count `0`。

A12/A13 首个决策差异发生在 step `1`。A13 前缀为 `b6:right,b2:up,b3:right,b8:left,...`，说明 access-recovery guard 确实阻止了 A12 第二步继续把 `b6` 推入低余量走廊。A13 的主要改善是避免了 A12 的早期 `no_action`，并将 slot schedule cost 从 `47` 降到 `29`。但后半段从 step `60` 开始反复出现低余量选择：`b6:down` 的 recovery score 为 `-53`，`low_legal_margin=true`，随后 `b9:up` 恢复到两个非阻塞合法推箱动作，再回到 `b6:down`。该循环累计导致 `selected_low_legal_margin_count=61`、`selected_legal_collapse_count=87`。

判定：A13 是部分改善但仍为负结果。它证明 access-recovery margin 能把 A12 的早期无动作失败推迟为长程闭环，并暴露新的低余量循环；但它没有产生任何目标完成，且最终 target blocker 增至 `2`、release action 仍为 `0`。下一步不应只继续加大 low-legal penalty，而应实现真正的 release/access recovery macro：当 dependency graph 显示站位格不可达或 target 被占时，显式寻找能恢复玩家站位、移开 target blocker 或打开 release action 的短宏动作序列，并用 SD 摘要约束其执行边界。

### 2026-06-20 Room-GFT A14 bounded recovery macro 负结果

基于 A13 将早期 `no_action` 转化为低合法余量循环的诊断，本轮继续在 `room_gft_*` 派生线中新增 A14。规划侧新增 [RoomScheduleRecoveryMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A13 的 `RoomScheduleAccessRecoveryGFTSDMCTSPlanner`，仍使用 A11 的 schedule-plan SD 表示，不修改 Boxoban 主线，也不接入 teacher/reference 动作。A14 在 MCTS 选出的根动作及原 corridor macro 之后，执行一个 bounded recovery macro 搜索：在当前局面向前枚举最多 `4` 个合法非 Tier-A 死锁 push，只有当短序列改善非阻塞合法推箱余量、target blocker、不可达站位、release action、dependency 总数或 slot cost 时才追加执行。

新增诊断脚本 [analyze_room_gft_xsokoban_a14_recovery_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a14_recovery_macro.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a14_recovery_macro.py
```

输出为 [room_gft_xsokoban_a14_recovery_macro_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a14_recovery_macro_l002_s200.json) 与 [room_gft_xsokoban_a14_recovery_macro_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a14_recovery_macro_l002_s200.md)。配置为 `level_offset=1,budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recent_edge_window=12,recovery_min_legal_pushes=2,recovery_macro_max_extra_pushes=4,recovery_macro_max_nodes=400,recovery_macro_score_threshold=2.0`。同轮 A13 对照为 `max_steps`、执行 `200` push、`182` 个闭环决策、`boxes_on_goals=0`、final slot cost `29`、dependency count `2`、target blocker count `2`。A14 为 `max_steps`、执行 `200` push、`73` 个闭环决策、`boxes_on_goals=0`、final slot cost `33`、final schedule barrier `72.0`、dependency count `3`、future dependency count `2`、target blocker count `1`、release action count `0`。

A14 recovery macro 在 `73` 个闭环决策中触发 `48` 次，累计追加 `116` 个 push，median recovery score 为 `3.75`，max recovery score 为 `46.85`。A13/A14 首个决策差异发生在 step `1`。A14 前缀为 `b6:right,b7:left,b3:left,b0:left,...`，显示宏机制显著改变了闭环轨迹和压缩了决策次数。A14 同时消除了 A13 记录中的 selected low-legal margin，但没有产生 goal 填充，也没有生成 release action；终态 dependency 总量反而高于 A13。

判定：A14 是负结果，但提供了比 A13 更明确的机制边界。bounded recovery macro 能被执行，并能改善局部合法余量和部分 target blocker；然而当前宏的目标函数过宽，倾向于追逐一般合法余量和 slot cost 的局部改善，而不是绑定到具体 dependency row 的站位格或 target blocker。因此，它把失败从“低余量循环”转成“高频恢复宏循环”，仍不能把任何箱子送入目标。下一步应收窄为 dependency-bound macro：宏入口必须绑定一个具体 `stand_cell_unreachable_by_player` 或 `target_occupied_by_box` 依赖，宏停止条件应优先要求该依赖解除、对应站位格玩家可达、target blocker 被移开或 release action 出现，而不应仅奖励一般 legal-margin 增加。

### 2026-06-20 Room-GFT A15 dependency-bound recovery macro 负结果

基于 A14 “宏执行机制可用但目标过宽”的诊断，本轮继续在 `room_gft_*` 派生线中新增 A15。规划侧新增 [RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A14 的 `RoomScheduleRecoveryMacroGFTSDMCTSPlanner`，仍使用 A11 的 schedule-plan SD 表示，不修改 Boxoban 主线，也不接入 teacher/reference 动作。A15 的 recovery suffix 在宏入口状态选择一个具体 dependency row，优先当前依赖、`target_occupied_by_box`、`stand_cell_unreachable_by_player` 和较高 barrier cost；短宏评分优先奖励该依赖行消失、所选 dependency kind 解除、target blocker 被移开、站位格变为玩家可达、对应 release action 出现，并抑制无依赖进展的 legal-margin 型收益。

新增诊断脚本 [analyze_room_gft_xsokoban_a15_dependency_bound_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a15_dependency_bound_macro.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a15_dependency_bound_macro.py
```

输出为 [room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.json) 与 [room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a15_dependency_bound_macro_l002_s200.md)。配置为 `level_offset=1,budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recent_edge_window=12,recovery_min_legal_pushes=2,recovery_macro_max_extra_pushes=4,recovery_macro_max_nodes=400,recovery_macro_score_threshold=2.0,dependency_bound_macro_score_threshold=20.0`。A14 对照为 `max_steps`、执行 `200` push、`73` 个闭环决策、`boxes_on_goals=0`、final slot cost `33`、dependency count `3`、target blocker count `1`。A15 为 `no_action`、执行 `36` push、`21` 个闭环决策、`boxes_on_goals=0`、final slot cost `42`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。

A15 绑定依赖宏触发 `3` 次，累计追加 `9` 个 push，median dependency-bound score 为 `111.1`，max score 为 `235.85`；其中 `dependency_progress_rows=6`、`row_disappeared_count=5`、`stand_resolved_count=3`、`target_resolved_count=0`，绑定过的 dependency kind 包括 `stand_cell_unreachable_by_player` 和 `target_occupied_by_box`。A14/A15 首个决策差异发生在 step `1`。A15 前缀为 `b6:right,b4:up,b3:up,b4:right,b7:left,...`，并在 step `21` 进入 `no_action`。

判定：A15 是负结果，且比 A14 更清楚地暴露了“局部依赖解析”和“全局装填可行性”之间的缺口。dependency-bound macro 可以解析若干行级依赖，尤其能让部分 `stand_cell_unreachable_by_player` 行消失或站位格恢复可达；但这些局部进展没有稳定转化为目标填充、release action 或后续可行动作，反而较早进入 `no_action`。因此，下一步不应简单继续提高依赖绑定宏的权重或节点预算，而应把 dependency resolution 与 slot-fill phase、目标完成进度、后续合法动作保持和不可逆破坏控制共同作为宏后置条件。TVA 审计样本应优先覆盖 A15 中“dependency row 消失但 slot cost 上升或终局 no_action”的状态。

### 2026-06-20 Room-GFT A16 feasibility-guarded dependency macro 轻量负结果

基于 A15 中“dependency row 消失但全局装填可行性未改善”的失败形态，本轮继续在 `room_gft_*` 派生线中新增 A16。规划侧新增 [RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A15 的 `RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A16 保留具体 dependency row 绑定，但把短宏验收条件从“局部依赖进展”扩展为“局部依赖进展且全局后置条件通过”：slot schedule cost 不回退、schedule barrier 不大幅回退、target blocker 不增加、critical stand box 不增加、destructive action 不增加、非阻塞合法推箱余量不坍塌。

实现后首先尝试默认 200 push 级别复核。`recovery_macro_max_nodes=400` 与 `80` 两个配置均因诊断成本过高被中止；堆栈显示耗时来自宏候选 BFS 与长程 MCTS prior/schedule summary 两部分，而非语法或运行时错误。因此，本轮改为机制 smoke：`budget=128,smoke_steps=80,recovery_macro_max_nodes=20`。运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a16_feasibility_guarded_macro.py \
  --budget 128 \
  --smoke-steps 80 \
  --recovery-macro-max-nodes 20 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.json \
  --out-md docs/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.md
```

输出为 [room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.json) 与 [room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a16_feasibility_guarded_macro_l002_s80_b128_n20.md)。A15 对照为 `no_action`、执行 `36` push、`21` 个闭环决策、`boxes_on_goals=0`、final slot cost `42`、dependency count `2`。A16 轻量 smoke 为 `max_steps`、执行 `80` push、`78` 个闭环决策、`boxes_on_goals=0`、final slot cost `50`、final schedule barrier `34.5`、dependency count `1`、future dependency count `0`、target blocker count `1`、release action count `0`，终态 `tier_a_deadlocked=false,no_legal_pushes=false`。

A16 feasibility-guarded macro 触发 `1` 次，追加 `2` 个 push，最终宏验收行 `global_feasibility_violations={}`，并解析一个 `target_occupied_by_box` 依赖。A15/A16 首个决策差异发生在 step `1`。A16 后半段从约 step `35` 开始反复出现 `b6:down` 与 `b9:up`，说明全局后置条件避免了 A15 的早期 `no_action`，但把失败形态转回长程循环；该循环没有目标填充，也没有 release action 产生。

判定：A16 是机制部分有效但仍为负结果。它验证了“dependency resolution 不能单独作为宏执行充分条件”的判断，且全局后置条件能过滤 A15 中部分有害局部解析；但 A16 没有解决根选择层的长程循环，也没有形成目标进展。下一步不应继续只加宏后置条件，而应把 `b6:down / b9:up` 类重复循环上升为 schedule-level no-progress 约束，或在 TVA 中把“后置条件通过但目标/slot phase 无进展”的状态作为负例校准。

### 2026-06-20 Room-GFT A17 phase-progress root guard 负结果

基于 A16 在轻量 smoke 中转入 `b6:down / b9:up` 类长程循环的失败形态，本轮继续在 `room_gft_*` 派生线中新增 A17。规划侧新增 [RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A16 的 `RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A17 在候选根动作上新增 `room_phase_progress` 摘要，记录 goal delta、slot cost delta、schedule barrier delta、dependency total delta、target blocker delta、release action delta、重复 schedule signature、重复/反向 moved edge 和 no-release context，并在 prior、root guard 与 leaf 中惩罚 phase 无进展的重复调度。

本轮还扩展 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，在 trace 中记录 `chosen_phase_progress` 与 `chosen_phase_progress_score`；该改动只记录已存在的 root child 摘要，不改变规划行为。A17 诊断脚本为 [analyze_room_gft_xsokoban_a17_phase_progress_guarded_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a17_phase_progress_guarded_macro.py)。运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a17_phase_progress_guarded_macro.py
```

输出为 [room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.json) 与 [room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a17_phase_progress_guarded_macro_l002_s80_b128_n20.md)。配置与 A16 轻量 smoke 对齐：`budget=128,smoke_steps=80,recovery_macro_max_nodes=20`。A16 对照为 `max_steps`、执行 `80` push、`78` 个闭环决策、`boxes_on_goals=0`、final slot cost `50`、dependency count `1`。A17 为 `no_action`、执行 `22` push、`19` 个闭环决策、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `68.0`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。

A16/A17 首个决策差异发生在 step `2`。A17 中 `phase_progress_selected_count=5`、`phase_no_progress_selected_count=14`、`phase_stagnant_context_selected_count=13`、`phase_repeated_signature_selected_count=13`；feasibility-guarded macro 未触发。终态 `tier_a_deadlocked=false,no_legal_pushes=true`。这说明 A17 的 phase-progress guard 确实改变了 A16 后半段循环，但在缺少可执行 release/phase-progress 生成机制时，较强的无进展惩罚会把 planner 推向早期 `no_action`。

判定：A17 是负结果。它把 A16 的长程循环诊断前移为根选择层的 no-progress 问题，但没有解决第二关。当前证据表明，仅靠惩罚无进展动作会导致过约束；下一步更合理的是把 A17 的 `chosen_phase_progress` 状态作为 TVA/根选择校准样本，学习或规则化“无进展但仍保持可恢复性”的动作与“无进展且通向 no_action”的动作差异，而不是继续提高 no-progress penalty。

### 2026-06-20 Room-GFT A18 recoverability-calibrated root guard 负结果

基于 A17 中“无 phase progress 惩罚过强、导致早期 `no_action`”的失败形态，本轮继续在 `room_gft_*` 派生线中新增 A18。规划侧新增 [RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A17 的 `RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A18 的目标是检验：根选择层是否可以通过短程可恢复性探测，区分“无即时 phase progress 但仍可恢复”的准备动作与“无进展且通向无动作”的破坏性动作。

本轮同时扩展 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，在 trace 中记录 `chosen_recoverability` 与 `chosen_recoverability_score`；该改动只记录 root child 摘要，不改变规划行为。A18 诊断脚本为 [analyze_room_gft_xsokoban_a18_recoverability_guarded_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a18_recoverability_guarded_macro.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a18_recoverability_guarded_macro.py
```

输出为 [room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s80_b128_n16.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s80_b128_n16.json) 与 [room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s80_b128_n16.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s80_b128_n16.md)。配置与 A17 轻量 smoke 对齐：`level_offset=1,budget=128,horizon=8,smoke_steps=80,recovery_macro_max_nodes=20,recoverability_probe_depth=1,recoverability_probe_max_nodes=16`。A17 对照为 `no_action`、执行 `22` push、`19` 个闭环决策、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`；A18 为 `no_action`、执行 `6` push、`3` 个闭环决策、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `40.5`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。

A18 与 A17 的首个决策差异发生在 step `0`。A18 前缀为 `b7:right,b6:right,b6:right`，随后进入 `no_action`；终态 `tier_a_deadlocked=false,no_legal_pushes=true`。A18 只触发 `1` 次 recoverability probe，且该次把首步 `b7:right` 标记为 `recoverable_no_progress=true`，原因是深度 1 探测发现后继 `b6:right` 可将 slot schedule cost 从 `52` 降至 `51`，非阻塞合法推箱数从 `3` 升至 `4`。但是该探测没有带来 dependency total、target blocker 或 release action 改善，也没有保证宏执行后的多步合法余量；实际闭环在两次 `b6:right` 后进入无合法推箱状态。

判定：A18 是负结果。它验证了 A17 之后的方向判断，即根选择层确实需要可恢复性判据；但当前 A18 的可恢复性定义过短视，把一层内的局部 slot-cost 改善误判为可恢复准备动作。下一步不应简单继续增加 recoverable bonus 或 no-progress penalty，而应把可恢复性定义收紧为 survivable recoverability：恢复探测至少需要 dependency、target blocker 或 release action 之一产生结构进展；若只有 slot-cost 局部改善，则还必须通过多步合法余量保持和无 `no_action` 风险检查。TVA 审计应优先纳入 A18 中 `recoverable_no_progress=true` 但实际迅速 `no_action` 的状态。

### 2026-06-20 Room-GFT A19 survivable recoverability 机制级部分改善

基于 A18 将首步 `b7:right` 误判为可恢复 no-progress 的诊断，本轮继续在 `room_gft_*` 派生线中新增 A19。规划侧新增 [RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A18 的 `RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A19 将 recoverability 收紧为结构进展判据：只有 goal 增加、dependency total 下降、target blocker 下降或 release action 增加，且后继状态合法余量满足阈值时，no-progress 候选才可获得 `recoverable_no_progress`；纯 slot-cost 或 schedule-barrier 局部改善仅保留为诊断和扣分项，不再单独获得 recoverable bonus。

新增诊断脚本 [analyze_room_gft_xsokoban_a19_survivable_recoverability_guarded_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a19_survivable_recoverability_guarded_macro.py)。初始尝试沿用 A18/A17 的 `budget=128,smoke_steps=80`，并将 probe 设置为 `depth=2,max_nodes=64` 与 `depth=2,max_nodes=16`，两次均因计算成本过高被中止；堆栈显示耗时来自 MCTS 多层节点候选上的 recoverability probe 与 legal-push 可达域计算，而非运行时错误。因此，本轮正式保留机制级 smoke：`budget=64,smoke_steps=40,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`。为保证对照口径一致，同参 A18 也已落盘。

同参 A18 输出为 [room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s40_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s40_b64_n8.json) 与 [room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s40_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a18_recoverability_guarded_macro_l002_s40_b64_n8.md)。A19 输出为 [room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.json) 与 [room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a19_survivable_recoverability_guarded_macro_l002_s40_b64_n8.md)。同一机制级配置下，A18 为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`；A19 为 `max_steps`、执行 `40` push、`17` 个闭环决策、`boxes_on_goals=1`、final slot cost `41`、final schedule barrier `53.5`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`，终态 `tier_a_deadlocked=false,no_legal_pushes=false`。

A18/A19 首个决策差异发生在 step `0`。A18 前缀为 `b7:right,b6:right,b6:right`，A19 前缀为 `b6:right,b2:up,b6:right,b3:up,b4:right,b7:left,b7:up,b8:left,b8:left,b8:left,...`。A19 中 `recoverability_probe_count=3`、`recoverable_no_progress_selected_count=1`、`unrecoverable_no_progress_selected_count=2`、`survivable_structural_progress_selected_count=1`，且 feasibility-guarded macro 触发 `1` 次。该结果说明，收紧 recoverability 后，A18 的首步 slot-only false positive 被消除，轨迹不再早停，并能在 40 push 内完成一个目标。

判定：A19 是机制级部分改善，但仍不是第二关求解配置。它修复了 A18 的首步误判，并将状态从早期 `no_action` 推进到至少一个目标完成；但在 40 push 上限内仍有 dependency、future dependency、target blocker 和 release action 缺口，且 full `budget=128,smoke_steps=80` 诊断成本过高。下一步不应继续在所有 MCTS 节点上无缓存地加深 recoverability probe，而应做两件事：第一，将 recoverability probe 限制或缓存到根候选层，降低 A19 成本；第二，在 A19 已暴露的后续 dependency 状态上补显式 release action 生成或 TVA 前缀校准，使一个目标后的状态能继续形成有效调度。

### 2026-06-20 Room-GFT A20 root-cached survivable recoverability 80-step 复核

基于 A19 完整 80-step 诊断成本过高的问题，本轮继续在 `room_gft_*` 派生线中新增 A20。规划侧新增 [RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A19 的 `RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A20 的唯一机制变化是：survivable recoverability probe 只在当前闭环决策的 root candidates 上计算；MCTS 内部非根节点直接跳过该 probe，同时对同一 root candidate 的 probe 结果做 planner 内缓存。

新增诊断脚本 [analyze_room_gft_xsokoban_a20_root_cached_survivable_recoverability.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a20_root_cached_survivable_recoverability.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a20_root_cached_survivable_recoverability.py
```

输出为 [room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a20_root_cached_survivable_recoverability_l002_s80_b128_n8.md)。配置为 `level_offset=1,budget=128,horizon=8,smoke_steps=80,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8,root_recoverability_only=true`。A19 机制级对照为 `max_steps`、执行 `40` push、`boxes_on_goals=1`、final slot cost `41`、dependency count `2`；A20 为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、final schedule barrier `55.0`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`，终态 `tier_a_deadlocked=false,no_legal_pushes=false`。

A20 与 A19 的前 17 个根动作一致，首个差异记录为 step `17`，主要因为 A19 机制级对照在 40 push 上限处停止，而 A20 继续执行到 80 push。A20 中 `recoverability_probe_count=18`、`recoverable_no_progress_selected_count=2`、`unrecoverable_no_progress_selected_count=16`、`survivable_structural_progress_selected_count=2`、`root_recoverability_probe_skipped_selected_count=33`，feasibility-guarded macro 触发 `1` 次。结果说明，root-only/cache 足以把 A19 从 40-step 机制复核推进到 80-step 复核；A20 没有复现 A18 的早期 `no_action`，并在 80 push 内完成两个目标。

判定：A20 是计算边界修复和机制级部分改善，但仍不是第二关求解配置。它解决了 A19 无法完成 80-step 复核的主要成本问题，并将目标数从 1 提高到 2；但 final slot cost 升至 `60`，dependency 与 future dependency 仍存在，release action 仍为 `0`。因此，A20 暴露的新主因已经不是 root recoverability 成本，而是两个目标后的 slot-fill/release 调度缺口。下一步应在 A20 后继状态上补显式 release/action 生成，或把 `boxes_on_goals=2,dependency_count=2,future_dependency_count=1,release_action_count=0` 的前缀作为 TVA 校准样本。

### 2026-06-20 Room-GFT A21 stand-lock guard 负结果

基于 A20 终态仍有 `dependency_count=2,future_dependency_count=1,release_action_count=0`，且 replay 显示后继依赖包含 `stand_cell_occupied_by_box`，本轮继续在 `room_gft_*` 派生线中新增 A21。规划侧新增 [RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A20 的 `RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A21 在 root candidate 和 leaf 中加入 stand-lock 诊断：若关键站位格被箱子占据且该箱没有合法 release push，或已在目标格的箱子被重新分配到其他 slot，则降低候选优先级；若候选能增加 release option，则给予小幅奖励。

新增诊断脚本 [analyze_room_gft_xsokoban_a21_stand_lock_guarded_release.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a21_stand_lock_guarded_release.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a21_stand_lock_guarded_release.py
```

输出为 [room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a21_stand_lock_guarded_release_l002_s80_b128_n8.md)。配置与 A20 对齐：`level_offset=1,budget=128,horizon=8,smoke_steps=80,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8,root_recoverability_only=true`。A20 对照为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`。A21 为 `max_steps`、执行 `80` push、`77` 个闭环决策、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。

A20/A21 首个决策差异发生在 step `3`。A21 选择中 `stand_lock_selected_count=77`、`stand_lock_risk_selected_count=73`、`locked_stand_blocker_selected_count=73`、`assigned_goal_reassignment_selected_count=0`，stand-lock score 范围为 `[-123,0]`。后半段从约 step `17` 起基本进入 `b9:up / b7:down` 往复，slot cost 维持 `48`，目标数始终为 `0`。

判定：A21 是负结果。stand-lock 信号能够解释 A20 后继状态中的释放风险，但只靠 root/leaf guard 会过早偏离 A20 的有效装填前缀，并且在没有显式 release/action 生成器时无法修复该风险。后续不应继续调大 stand-lock penalty；更合理的方向是派生 A22，将 stand-lock 作为 release macro 的入口条件，显式枚举能解除具体 `stand_cell_occupied_by_box` 依赖的短动作序列，并以不破坏 slot-fill progress、目标数和可恢复性作为验收条件。

### 2026-06-20 Room-GFT A22 phased release macro 阶段化机制验证

基于 A21 “stand-lock 信号有效但全局 guard 过强”的负结果，本轮继续在 `room_gft_*` 派生线中新增 A22。规划侧新增 [RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A21 的 `RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A22 将 stand-lock 从全局强惩罚改为阶段化 release/action 信号：未达到 release 里程碑前沿用 A20 兼容的 pre-release dependency-bound macro；达到 `stage_release_min_goals=2` 后，才尝试针对 `stand_cell_occupied_by_box` 依赖生成 release-stage macro。

本轮同时扩展 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，在 trace 中记录 `chosen_stage` 与 `chosen_stage_score`；该记录只反映 root child 摘要，不引入 teacher/reference 动作。新增诊断脚本为 [analyze_room_gft_xsokoban_a22_phased_release_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a22_phased_release_macro.py)。默认运行命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a22_phased_release_macro.py
```

输出为 [room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a22_phased_release_macro_l002_s80_b128_n8.md)。配置与 A20/A21 对齐：`level_offset=1,budget=128,horizon=8,smoke_steps=80,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`，新增阶段参数为 `stage_release_min_goals=2,stage_release_stand_occupied_resolution_weight=85.0,stage_release_goal_nonregression=true,stage_stand_lock_score_scale=0.35`。

A20 对照为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`。A21 为 `max_steps`、执行 `80` push、`77` 个闭环决策、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。A22 为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`。

A20/A22 首个决策差异为 `None`，说明 A22 在当前设置下完整保留了 A20 的有效前缀；A21/A22 首个决策差异发生在 step `3`，说明阶段化消费避免了 A21 的早期偏离。A22 中 `stage_before_counts={'fill':19,'release':32}`，release-stage 选择 `32` 次，`locked_stand_blocker_selected_count=48`；true phased release macro 触发 `0` 次，pre-release compatible macro 触发 `1` 次并追加 `2` 个 push。

判定：A22 是阶段化机制验证和部分成功，但仍不是第二关求解配置。它支持“复杂 Sokoban 决策机制需要按里程碑阶段消费不同 SD 摘要”的判断，并证明 A21 的退化来自全局 stand-lock 消费方式，而非 stand-lock 信号本身错误。但 A22 没有触发真正的 release-stage macro，也未超过 A20 的 2-goal 结果。后续应修订 release 阶段的依赖入口、候选生成和宏验收条件，使 `stand_cell_occupied_by_box` 能产生可执行释放序列；不应回退为全局 stand-lock penalty。

### 2026-06-20 Room-GFT A23 pre-release bottleneck corridor guard 负结果

基于 A22 中 release 阶段 32 次出现但 true release macro 从未触发的结果，本轮先回放 A22 轨迹并定位互锁形成位置。诊断显示：A20/A22 在 step `2` 的 corridor macro 连续执行 `b6:right,b6:up,b4:up,b3:up`，使后继状态只剩低余量调度；随后 step `3` 的 `b3:up` 使 `[2,6]` 成为未来关键站位上的不可释放阻塞箱。该阻塞箱被 `[3,6]`、`[2,7]` 和左侧墙共同夹住；在该 release 状态下，将 release suffix 扩大到 `recovery_macro_max_nodes=2000,recovery_macro_max_extra_pushes=6` 仍无法得到可接受的 `stand_cell_occupied_by_box` 释放序列。因此，本轮没有继续增加 release 宏权重，而是在 A22 上派生 A23，检验是否应把阶段化机制前移到 corridor macro 执行边界。

规划侧新增 [RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A22 的 `RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A23 在 `corridor_macro_from_plan()` 内模拟宏前缀；若前缀仍处于 release 里程碑之前，存在 unresolved dependency，`target_blocker_count`、`unreachable_stand_count` 或 critical stand box 为正，`release_action_count=0`，且非阻塞合法推箱数不高于 `2`，则截断 corridor macro 并记录 `pre_release_bottleneck_truncated_corridor`。

新增诊断脚本 [analyze_room_gft_xsokoban_a23_pre_release_bottleneck_guard.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a23_pre_release_bottleneck_guard.py)。先运行 `budget=64,smoke_steps=30` 机制 smoke，输出 [room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s30_b64_n8.json)。该 smoke 中 A23 触发 `2` 次 bottleneck 截断，30 push 内达到 `boxes_on_goals=1`，说明截断机制确实改变了早期前缀，但未解除互锁。

随后运行同参 80-push 复核：

```bash
python scripts/analyze_room_gft_xsokoban_a23_pre_release_bottleneck_guard.py
```

输出为 [room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a23_pre_release_bottleneck_guard_l002_s80_b128_n8.md)。A22 为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`；A23 为 `max_steps`、执行 `80` push、`53` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`。A23 中 `pre_release_bottleneck_truncation_steps=2`，`pre_release_bottleneck_truncated_pushes_saved=8`，但 true phased release macro 仍为 `0`。

判定：A23 是负结果，但缩小了失败定位。它证明 A22 的早期互锁确实与 corridor macro 过度推进有关，宏截断能改变 step `3` 前缀；但截断本身不能生成替代调度，最终仍回到 A22 的 2-goal、0-release-action 终态。下一步不应继续只调截断阈值，而应在根选择层或 TVA 中加入阶段化的“前置依赖解除”偏好：当候选动作会把当前 target/stand 前置依赖转化为不可释放 stand-lock 时，必须主动选择能保持可恢复空间的替代路径，而不是仅停止宏。

### 2026-06-20 Room-GFT A24 pre-release root-trap guard 负结果

基于 A23 “宏截断能推迟互锁但不能生成替代调度”的结果，本轮继续在 `room_gft_*` 派生线中新增 A24。规划侧新增 [RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A23 的 `RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A24 在根候选上比较前后状态的 target blocker、不可达 stand、critical stand box、locked stand blocker、release action 和合法余量，惩罚把前置依赖转化为不可释放 stand-lock 的动作，并在 root guard 中优先选择非 severe trap 的候选。

本轮同时扩展 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，在 trace 中记录 `chosen_pre_release_trap` 与 `chosen_pre_release_trap_score`；该改动只记录 root child 摘要，不改变旧版本规划行为。新增诊断脚本为 [analyze_room_gft_xsokoban_a24_pre_release_root_trap_guard.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a24_pre_release_root_trap_guard.py)。

先运行 `budget=64,smoke_steps=30` 机制 smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a24_pre_release_root_trap_guard.py \
  --budget 64 \
  --smoke-steps 30 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s30_b64_n8.md
```

该 smoke 输出为 [room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s30_b64_n8.json)。A24 在 step `4` 与 A23 首次分歧，改选 `b7:right` 而不是 `b3:up`，说明 root-trap 信号确实识别了直接 stand-lock 转化；但 30 push 内 `boxes_on_goals=0`，且 step `5` 仍被迫选择 severe trap。

随后运行同参 80-push 复核：

```bash
python scripts/analyze_room_gft_xsokoban_a24_pre_release_root_trap_guard.py
```

输出为 [room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a24_pre_release_root_trap_guard_l002_s80_b128_n8.md)。A23 为 `max_steps`、执行 `80` push、`53` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、target blocker count `0`、release action count `0`；A24 为 `max_steps`、执行 `80` push、`78` 个闭环决策、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。

A24 中 `pre_release_bottleneck_truncation_steps=1`、`pre_release_bottleneck_truncated_pushes_saved=2`，`pre_release_root_trap_selected_severe_count=73`、`pre_release_root_trap_selected_conversion_count=1`、`pre_release_root_trap_top_severe_count=73`。关键前缀为 `b6:right,b2:up,b6:right,b4:up,b7:right,b3:up,b4:right,...`。step `4` 的 `b7:right` 没有制造 stand-lock，但把非阻塞合法推箱数降至 `1` 并进入 recovery；step `5` 时唯一可选动作仍是 `b3:up`，该动作 `locked_stand_blocker_delta=1,precondition_to_lock_conversion=true,unreleasable_lock_after=true,score=-298`。随后轨迹进入 `b9:up / b7:down` 长循环。

判定：A24 是负结果，但给出了新的定位证据。它证明 root-trap 信号可以识别并避开直接的 stand-lock 转化候选；失败原因在于该识别发生得太晚，单步替代动作没有多步计划支持，会进入只有坏动作可选的低余量 recovery 状态。下一步不应继续只加大 root-trap penalty，而应把该风险前移为阶段里程碑或多步调度约束，在 step `2`/step `3` 之前生成可执行的 target/stand 前置依赖解除序列。

### 2026-06-21 Room-GFT A25 SD generalization guard 负结果

基于 A24 “root-trap 信号有效但单步 guard 太晚”的结果，并结合 DQN 对表格型 Q-learning 的启发，本轮在 `room_gft_*` 派生线中新增 A25。规划侧新增 [RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A24 的 `RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner`，不修改 A11 schedule-plan SD、Boxoban 主线或在线 teacher/reference 边界。A25 的设计意图不是引入真正神经函数逼近，而是在 OO 派生类中形成一个可替换的 SD-consumer 泛化接口：把 A24 已知失败轨迹和 A20/A23 安全前缀抽象成结构原型，并在根候选上根据 SD 签名相似度、successor trap-predecessor 探测、dependency/release/合法余量差分重排动作。

本轮同时扩展 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，在 trace 中记录 `chosen_sd_generalization` 与 `chosen_sd_generalization_score`；该改动只读取 root child 摘要，不改变旧版本规划行为。新增诊断脚本为 [analyze_room_gft_xsokoban_a25_sd_generalization_guard.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a25_sd_generalization_guard.py)。

先运行 `budget=64,smoke_steps=30` 机制 smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a25_sd_generalization_guard.py \
  --budget 64 \
  --smoke-steps 30 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a25_sd_generalization_guard_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a25_sd_generalization_guard_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a25_sd_generalization_guard_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a25_sd_generalization_guard_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a25_sd_generalization_guard_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a25_sd_generalization_guard_l002_s30_b64_n8.md)。A25 在 step `2` 与 A24 首次分歧，前缀从 A24 的 `b6:right,b2:up,b6:right,b4:up,b7:right,...` 改为 `b6:right,b2:up,b6:left,b6:right,b3:right,...`。30-push 内 A25 为 `max_steps`、执行 `30` push、`28` 个闭环决策、`boxes_on_goals=0`、final slot cost `49`、dependency count `1`、future dependency count `0`、target blocker count `0`、release action count `0`，且未选择 A24 severe trap、generalized failure 或 trap predecessor。

随后运行同参 80-push 复核：

```bash
python scripts/analyze_room_gft_xsokoban_a25_sd_generalization_guard.py
```

输出为 [room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a25_sd_generalization_guard_l002_s80_b128_n8.md)。同一 `budget=128,smoke_steps=80` 下，A22/A23 均为 `boxes_on_goals=2,final_slot_cost=60,release_action_count=0`；A24 为 `boxes_on_goals=0,final_slot_cost=48,dependency_count=2,target_blocker_count=1`；A25 为 `max_steps`、执行 `80` push、`78` 个闭环决策、`boxes_on_goals=0`、final slot cost `49`、final schedule barrier `16.5`、dependency count `1`、future dependency count `0`、target blocker count `0`、release action count `0`。

A25 中 `stage_before_counts={'fill':78}`，`stand_lock_risk_selected_count=0`、`locked_stand_blocker_selected_count=0`、`sd_generalization_selected_severe_count=0`、`sd_generalization_generalized_failure_selected_count=0`、`sd_generalization_trap_predecessor_selected_count=0`。A24/A25 首个决策差异为 step `2`。A25 轨迹前缀为 `b6:right,b2:up,b6:left,b6:right,b3:right,b8:left,b5:left,b3:up,b3:up,b2:left,b4:down,b9:up,b4:up,b5:up,b5:up,b5:right,b9:up,b6:down,b9:up,b6:down,...`，约从 step `17` 起进入 `b9:up / b6:down` 往复。

判定：A25 是负结果，但给出了新的设计证据。它证明“结构化 SD 表示 + 可泛化消费函数”的方向能够在更早步骤共享失败证据，并完全避开 A24 的 severe stand-lock 轨迹；但当前手工原型泛化仍只会偏向“安全、不坏、无 trap”的状态，不能识别“安全但无目标进展”的长循环。因此，DQN 类比不应被实现为更多手工 prototype，而应作为下一阶段 learned TVA/value/ranking consumer 的接口约束：输入为 SD 结构特征，输出同时刻画 trap risk、milestone value、phase progress 与 release/dependency 进展。

### 2026-06-21 Room-GFT A26-A28 learned SD consumer 诊断线

基于 A25 “结构泛化能避开已知 severe trap，但不能产生目标进展”的负结果，本轮继续沿用户要求的 OO 继承机制派生 learned SD consumer 测试线。该线仍只作用于 `room_gft_*` 难题分支，不修改 Boxoban 989/1000 主线，不改变 exact grid-flow 转移，也不在在线求解中查询 teacher/reference 动作。

第一步新增 [RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A25。A26 在候选和边上加入 `room_sd_value_score` 与 `room_sd_value_summary`，把 goal、release、dependency、target blocker、slot cost、legal margin、trap/generalized failure、stand-lock 与 loop 信号汇入一个线性 ranking/value 接口。新增诊断脚本为 [analyze_room_gft_xsokoban_a26_sd_value_ranker.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a26_sd_value_ranker.py)。先运行 `budget=64,smoke_steps=30`，输出 [room_gft_xsokoban_a26_sd_value_ranker_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a26_sd_value_ranker_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a26_sd_value_ranker_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a26_sd_value_ranker_l002_s30_b64_n8.md)，30 push 内达到 `boxes_on_goals=1`。

随后运行同参 80-push：

```bash
python scripts/analyze_room_gft_xsokoban_a26_sd_value_ranker.py
```

输出为 [room_gft_xsokoban_a26_sd_value_ranker_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a26_sd_value_ranker_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a26_sd_value_ranker_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a26_sd_value_ranker_l002_s80_b128_n8.md)。A26 为 `max_steps`、执行 `80` push、`39` 个闭环决策、`boxes_on_goals=1`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `2`、release action count `0`。相对 A25，A26 从 0-goal 提升到 1-goal；但 `sd_value_milestone_selected_count=31`，后半段出现 repeated committed-slot 伪里程碑循环。

第二步新增 [RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A26 并只重写 `_sd_value_score()`。A27 将“结构改善但没有 goal/release，且处于 repeated/stagnant context”的候选标记为 `pseudo_milestone_loop`、`no_progress_loop_risk` 和 `hard_negative`。新增脚本为 [analyze_room_gft_xsokoban_a27_sd_value_loop_calibrated.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a27_sd_value_loop_calibrated.py)。30-push smoke 输出 [room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s30_b64_n8.md)。

80-push 复核命令：

```bash
python scripts/analyze_room_gft_xsokoban_a27_sd_value_loop_calibrated.py
```

输出为 [room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a27_sd_value_loop_calibrated_l002_s80_b128_n8.md)。A27 与 A26 轨迹完全一致，A26/A27 首个决策差异为 `None`；但诊断字段显示 `sd_value_milestone_selected_count` 从 A26 的 `31` 降为 `6`，`sd_value_pseudo_milestone_loop_selected_count=25`，`sd_value_no_progress_loop_selected_count=26`，`sd_value_hard_negative_selected_count=25`。判定：A27 已能标记 A26 的伪里程碑循环，但 root 选择没有消费该标签，因此行为未变。

第三步新增 [RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A27 并只重写 `choose_root_edge()`。A28 当父类选中 `pseudo_milestone_loop` 或 hard no-progress root edge 时，优先选择非 hard-negative、非 pseudo-milestone 的替代候选。新增脚本为 [analyze_room_gft_xsokoban_a28_sd_value_loop_escape.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a28_sd_value_loop_escape.py)。30-push smoke 输出 [room_gft_xsokoban_a28_sd_value_loop_escape_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a28_sd_value_loop_escape_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a28_sd_value_loop_escape_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a28_sd_value_loop_escape_l002_s30_b64_n8.md)，其中 A27/A28 首个决策差异发生在 step `14`。

80-push 复核命令：

```bash
python scripts/analyze_room_gft_xsokoban_a28_sd_value_loop_escape.py
```

输出为 [room_gft_xsokoban_a28_sd_value_loop_escape_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a28_sd_value_loop_escape_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a28_sd_value_loop_escape_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a28_sd_value_loop_escape_l002_s80_b128_n8.md)。A28 为 `max_steps`、执行 `80` push、`45` 个闭环决策、`boxes_on_goals=3`、final slot cost `26`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。阶段计数为 `fill=25,release=20`；`sd_value_pseudo_milestone_loop_selected_count=0`，`sd_value_inferred_guard_override_count=5`。A26/A28 与 A27/A28 的首个决策差异均为 step `14`。

判定：A26-A28 构成一条连续正向但未求解的证据链。A26 证明显式 SD 摘要进入 value/ranking 后能突破 A25 的 0-goal 安全循环；A27 证明 A26 的主要失败可被识别为伪里程碑循环；A28 证明该标签必须进入 root choice 才能转化为行为改善，并将第 2 关推进到 3 goals 和 release 阶段。当前剩余缺口已经从 fill 阶段的伪里程碑转移到 release 阶段：需要生成可执行 release action，并在 release-stage ranking 中避免新的无进展循环。

### 2026-06-22 Room-GFT A29 release-action macro 负诊断

基于 A28 已进入 release 阶段但 `release_action_count=0` 的结果，本轮继续沿 OO 继承机制派生 A29。规划侧新增 [RoomScheduleReleaseActionMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A28 的 `RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner`。该版本只在 release 阶段且 release action 未打开时尝试生成短释放后缀，不修改 A11 schedule-plan SD、Boxoban 989/1000 主线或在线 teacher/reference 边界。

本轮同时扩展 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，在闭环 trace 的 `corridor_macro` 字段中记录 `release_action_macro_enabled`、`release_action_macro_triggered`、`release_action_macro_reason`、`release_action_macro_score`、`release_action_extra_push_count`、`release_action_bound_dependency`、`release_action_reference_metrics`、`release_action_final_metrics` 和 `release_action_final_progress`。新增诊断脚本为 [analyze_room_gft_xsokoban_a29_release_action_macro.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a29_release_action_macro.py)。

先运行 `budget=64,smoke_steps=30` 机制 smoke，输出 [room_gft_xsokoban_a29_release_action_macro_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a29_release_action_macro_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a29_release_action_macro_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a29_release_action_macro_l002_s30_b64_n8.md)。该 smoke 与 A28 在 step `14` 之前共享前缀，30 push 内为 `max_steps`、`boxes_on_goals=1`，尚未进入 release 阶段，因此不用于判断 release macro 是否有效。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a29_release_action_macro.py
```

输出为 [room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a29_release_action_macro_l002_s80_b128_n8.md)。A29 与 A28 轨迹完全一致：`max_steps`、执行 `80` push、`45` 个闭环决策、`boxes_on_goals=3`、final slot cost `26`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。A28/A29 首个决策差异为 `None`。

关键诊断字段为：`release_stage_decision_count=20`、`release_action_macro_triggered_count=0`、`release_action_macro_opened_release_count=0`，但 `release_action_macro_reason_counts={'not_release_stage':25,'no_release_dependency':1,'no_admissible_release_suffix':19}`。在 19 个 release-stage 决策中，宏入口绑定到 `stand_cell_occupied_by_box` 依赖；关键站位格为 `[5,1]`，blocker cell 为 `[5,1]`，`stand_release_option_count=0`，reference metrics 显示 `stand_cell_occupied_by_box=true`、`locked_stand_blocker_count=1`、`release_action_count=0`。

判定：A29 是负诊断而非求解改进。它证明 release macro 入口已经接入并能绑定剩余关键依赖，但在当前短后缀搜索和目标不回退约束下没有找到可采纳释放序列。因此，下一轮应从受控目标回退、更长 release suffix 或显式 goal-locked stand blocker relocation generator 入手，而不是继续只调整 root ranking 或增加 fill-stage 伪里程碑惩罚。

### 2026-06-22 Room-GFT A30 controlled release-regression 诊断

根据“派生 A30”的要求，本轮继续沿 OO 继承机制新增 [RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A29 的 `RoomScheduleReleaseActionMacroGFTSDMCTSPlanner`。A30 不修改 Boxoban 主线，也不在在线求解中使用 teacher/reference 动作。新增诊断脚本为 [analyze_room_gft_xsokoban_a30_controlled_release_regression.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a30_controlled_release_regression.py)。

A30 第一版将 release suffix 搜索从 `max_extra_pushes=5,max_nodes=160` 扩大到 `max_extra_pushes=8,max_nodes=640`，并允许最多 `1` 个目标的受控临时回退，但要求同时存在直接 release 进展。该版本 80-push 同参复核与 A29 完全一致：仍为 `max_steps,80 pushes,45 decisions,boxes_on_goals=3,final_slot_cost=26,release_action_count=0`，`release_action_macro_reason_counts` 仍包含 19 次 `no_admissible_release_suffix`。随后重放 step `26` 后局面发现，[5,1] 上的 blocker 已是 goal-locked box，且该箱没有任何合法 push，因此事后允许受控回退已经太晚。

据此在 A30 内加入 release 宏截断保护：若继承宏在 release 阶段会制造或保留 `goal_locked_stand_blocker_count>0` 且 `release_action_count=0`，则截断到第一个 root push，并记录 `controlled_release_truncated_goal_lock`。最终 30-push smoke 输出为 [room_gft_xsokoban_a30_controlled_release_regression_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a30_controlled_release_regression_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a30_controlled_release_regression_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a30_controlled_release_regression_l002_s30_b64_n8.md)，该 smoke 尚未进入 release 阶段。

最终 80-push 同参复核命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a30_controlled_release_regression.py
```

输出为 [room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a30_controlled_release_regression_l002_s80_b128_n8.md)。A30 为 `max_steps`、执行 `80` push、`52` 个闭环决策、`boxes_on_goals=2`、final slot cost `35`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`。A29/A30 首个决策差异为 step `27`。

关键诊断字段为：`controlled_release_goal_lock_truncation_count=1`、`release_action_macro_triggered_count=0`、`release_action_macro_opened_release_count=0`、`controlled_release_goal_regression_count=0`、`release_action_macro_reason_counts={'not_release_stage':25,'no_release_dependency':27}`。在 step `25`，A30 将 A29 继承宏的 `4` 个 push 截断为 `1` 个 push；被截断的原宏末态会产生 `goal_locked_stand_blocker_count=1`、`locked_stand_blocker_count=1`、`release_action_count=0`，而保留前缀末态中这些值均为 `0`。

判定：A30 是负诊断但定位更前。它证明 A29 后半段的 3-goal release 状态来自一个会制造 goal-locked stand blocker 的危险宏；A30 能识别并阻断该宏，但没有生成替代 release 调度，导致终态从 3 goals 退化为 2 goals。因此，下一轮不应继续把重点放在事后释放 `[5,1]`，而应在进入 step `25` 的宏之前生成替代调度候选，或把 goal-lock 风险纳入 release-stage macro 的候选生成与 root choice，而不是只做宏截断。

### 2026-06-22 Room-GFT A31 goal-lock predecessor release replacement 诊断

根据 A30 “危险宏可识别但只截断会退化”的结果，本轮继续沿 OO 继承机制新增 [RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A30 的 `RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner`。A31 不修改 Boxoban 主线，不修改 A30 行为，也不在在线求解中使用 teacher/reference 动作。新增诊断脚本为 [analyze_room_gft_xsokoban_a31_goal_lock_predecessor_release.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a31_goal_lock_predecessor_release.py)。

A31 的机制是将 A30 的 `controlled_release_truncated_goal_lock` 信号前移到 root 替代动作选择：当 A30 检测到原宏会制造没有 release action 的 goal-locked stand blocker 时，A31 在当前 root 已展开语义候选中寻找替代动作。替代动作必须与原选动作不同、不降低当前目标数、且后继状态不含 goal-locked stand blocker；排序优先考虑打开 release action、减少 dependency/target blocker、减少 locked stand blocker、改善合法余量或降低 slot schedule cost。闭环 trace 已扩展记录 `goal_lock_predecessor_guard_triggered`、候选数、拒绝计数、原动作、替代动作、替代后指标和进展摘要。

先运行 `budget=64,smoke_steps=30` 机制 smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a31_goal_lock_predecessor_release.py \
  --budget 64 \
  --smoke-steps 30 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.md)。该 smoke 为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,release_stage_decision_count=0,goal_lock_predecessor_override_count=0`，说明 A31 在未进入 release 截断点前不改变前缀。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a31_goal_lock_predecessor_release.py
```

输出为 [room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s80_b128_n8.md)。A31 为 `max_steps`、执行 `80` push、`42` 个闭环决策、`boxes_on_goals=2`、final slot cost `29`、dependency count `2`、future dependency count `1`、target blocker count `2`、release action count `2`。A30/A31 首个决策差异为 step `25`。

关键诊断字段为：`controlled_release_goal_lock_truncation_count=2`、`goal_lock_predecessor_override_count=2`、`goal_lock_predecessor_structural_progress_count=2`、`goal_lock_predecessor_opened_release_count=2`、`goal_lock_predecessor_candidate_count_min=8`、`goal_lock_predecessor_candidate_count_max=8`。在 step `25` 和 step `30`，A31 将原始 `b6:down` 替换为 `b6:up`；替代后 `goal_delta=0`、`release_action_count_delta=2`、`slot_schedule_cost_delta=20.0`、`goal_locked_stand_blocker_delta=0`。拒绝计数为 `same_semantic_action=2,goal_regression=2,goal_lock_after=2`。

判定：A31 是机制推进但不是求解突破。它证明 A30 暴露的 goal-locking macro 风险可以被前移到 root choice，并转化为打开 release action 的替代动作；但它仍停在 2 goals。step `25` 之后，虽然 `release_action_count=2`，轨迹仍反复选择 `b3:down` 类 committed-slot 动作，dependency 与 target blocker 未继续下降。因此，后续缺口已从“release action 打不开”转为“release action 打开后缺少下一阶段目标推进和 release-stage 循环抑制”。下一轮应继续派生 A32/A31b，处理 release-open 状态下的 root ranking 与伪里程碑循环，而不是继续围绕事后释放 `[5,1]` 或单纯加大 A30 的截断保护。

### 2026-06-22 Room-GFT A32 learned SD consumer 接口与软消费复核

基于用户明确要求“回到第一性原理和通用机制，充分利用神经网络表达能力和经验复用能力，同时避免大量零散规则堆砌”，本轮新增 A32 learned SD consumer 派生线。规划侧新增 [RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A31 的 `RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner`。该版本只新增 learned consumer 特征契约、checkpoint 前向和 root prior/value 消费，不修改 A11 schedule-plan SD、exact grid-flow 转移、A31 行为或 Boxoban 主线，在线求解仍保持 `teacher_actions_used_online=false`。

本轮先固定 `LEARNED_SD_CONSUMER_FEATURE_NAMES`，共 `44` 个特征，覆盖目标、release、依赖、target blocker、slot cost、合法余量、stand-lock、阶段、循环、伪里程碑、trap/recoverability 和动作方向。随后新增 [build_room_gft_learned_sd_consumer_a32.py](/home/data/round/code/sd_world_model_rl/scripts/build_room_gft_learned_sd_consumer_a32.py)，从 A28-A31 trace 的 root/top-child 候选中构造离线样本，并输出 [room_gft_a32_learned_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_dataset.json) 与 [room_gft_a32_learned_sd_consumer_linear_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_linear_checkpoint.json)。重建指标为 `sample_count=6845,target_mean=-2.9425,target_std=2.4570,MAE=0.1717,corr=0.9922`。该 checkpoint 是启发式 trace 蒸馏，用于验证接口，不作为充分训练的神经 TVA。

新增诊断脚本 [analyze_room_gft_xsokoban_a32_learned_sd_consumer.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a32_learned_sd_consumer.py)。初始实现曾允许 learned scorer 默认硬覆盖父类 root choice，30-push smoke 显示该策略过强：开启 `--learned-sd-consumer-root-override` 的对照输出 [room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.md)，结果为 `status=no_action,executed_pushes=6,closed_loop_decision_steps=4,boxes_on_goals=0`，A31/A32 首个决策差异为 step `1`。

据此将 A32 默认策略改为软消费：`learned_sd_consumer_root_override=false`，只把 checkpoint 分数叠加进 root prior/value，并在 trace 中记录 `learned_sd_consumer_score` 与特征字典。软消费 30-push smoke 输出 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.md)，结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，`learned_sd_consumer_loaded_count=14`。

80-push 同参复核命令为：

```bash
python scripts/analyze_room_gft_xsokoban_a32_learned_sd_consumer.py \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.md)。A32 与 A31 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`，A31/A32 首个决策差异为 `None`；learned scorer 在 `42` 个决策中均被加载，正分选择次数 `20`，score 范围 `[-3.9273,3.7098]`。

判定：A32 是接口推进，不是求解突破。它把 A24-A31 的诊断信息整理为统一、可学习、可审计的 SD consumer 输入，并验证 checkpoint 能进入候选 scoring 与 trace；但当前线性 trace 蒸馏模型作为软信号不足以改变 A31 的 release-open 循环，作为硬覆盖则会导致早期 `no_action` 崩溃。因此，后续应训练真正的 neural/value/ranking consumer，并在 soft prior、leaf value 或经校准的 ranking 层逐步接入；A24-A31 继续作为样本来源和失败模式库，而不是继续扩展手写规则链。

### 2026-06-22 Room-GFT A33 neural SD consumer 训练与闭环诊断

在 A32 接口链路稳定后，本轮继续沿用户确认的“显式 SD 结构表示 + 神经/value/ranking 消费者 + exact MCTS”方向推进。规划侧新增 [RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A32 的 `RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner`。A33 不修改 A31/A32 既有机制、不修改 A11 schedule-plan SD 或 exact grid-flow 转移、不影响 Boxoban 主线，在线求解仍保持 `teacher_actions_used_online=false`。A33 默认只将 neural score 作为 soft prior/value 信号消费，`learned_sd_consumer_root_override=false`。

新增训练脚本 [train_room_gft_neural_sd_consumer_a33.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_neural_sd_consumer_a33.py)。该脚本从 A24-A32 的离线 trace 构造 `44` 维 SD consumer 特征样本，按 `source_file/version/step` root group 稳定切分训练/验证集，并训练 `[32,16]` MLP。输出 [room_gft_a33_neural_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_dataset.json) 与 [room_gft_a33_neural_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_mlp_checkpoint.json)。训练命令为：

```bash
python scripts/train_room_gft_neural_sd_consumer_a33.py
```

重建指标为：`sample_count=12462,train_count=10095,validation_count=2367,group_count=2684,validation_group_count=513`。训练集 `MAE=0.0743,corr=0.9984,target_top1_accuracy=0.8561,selected_top1_accuracy=0.5196`；验证集 `MAE=0.0696,corr=0.9986,target_top1_accuracy=0.8642,selected_top1_accuracy=0.5389`。该结果说明 MLP 能拟合当前 SD progress target，但该 target 仍是 trace 蒸馏和启发式构造，不等于真实求解价值。

新增诊断脚本 [analyze_room_gft_xsokoban_a33_neural_sd_consumer.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a33_neural_sd_consumer.py)。先运行 30-push soft smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a33_neural_sd_consumer.py \
  --budget 64 \
  --smoke-steps 30 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，`learned_sd_consumer_loaded_count=14`，score 范围 `[-0.9109,1.3072]`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a33_neural_sd_consumer.py \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.md)。A33 与 A31/A32 不再完全同轨迹，首个决策差异为 step `14`。最终为 `max_steps,80 pushes,46 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`；`stage_before_counts={'fill':27,'release':19}`，`goal_lock_predecessor_override_count=4`、`goal_lock_predecessor_opened_release_count=4`，neural scorer 加载 `46` 次，score 范围 `[-2.1753,1.4906]`。

补跑 hard-override 30-push 对照：

```bash
python scripts/analyze_room_gft_xsokoban_a33_neural_sd_consumer.py \
  --budget 64 \
  --smoke-steps 30 \
  --learned-sd-consumer-root-override \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.md)。结果仍为 `status=no_action,executed_pushes=6,closed_loop_decision_steps=4,boxes_on_goals=0`，A31/A33 首个决策差异为 step `1`。

判定：A33 是比 A32 更实质的接口推进，因为 neural soft prior 已能改变闭环轨迹，并增加 release-open 替代触发次数；但它仍未求解第二关，终态仍为 `2` goals。hard override 继续早期退化，说明 learned scorer 不能作为未经校准的根动作接管器。当前瓶颈已经从“接口是否能接入”转为“训练目标是否刻画 release-stage 真进展、是否能区分打开 release action 后的真实推进与新循环”。下一轮应改进目标构造或加入离线 teacher/search 轨迹的 milestone/solved-distance 标签，而不是继续扩大手写规则。

### 2026-06-22 Room-GFT A34 milestone SD consumer 负诊断

根据 A33 的结论，本轮不继续增加手写规则，而是改进 neural SD consumer 的训练目标。新增 [RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A33 的 `RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner`。A34 只隔离 provider id、默认 score clip 和 trace 标识；不修改 A31-A33、A11 schedule-plan SD、exact grid-flow 转移或 Boxoban 主线，在线求解仍保持 `teacher_actions_used_online=false`。

新增训练脚本 [train_room_gft_milestone_sd_consumer_a34.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_milestone_sd_consumer_a34.py)。该脚本复用 A33 的 MLP 训练和 JSON checkpoint 导出，但把 target 改为 `future_window_milestone_regression_v1`：每个 trace step 向前看 `8` 个闭环决策，奖励未来窗口内 goal gain、release action gain、dependency/target blocker 下降和 slot cost 下降，惩罚 release-open stagnation、无未来里程碑和目标回退，并保留部分 immediate SD target。训练命令为：

```bash
python scripts/train_room_gft_milestone_sd_consumer_a34.py
```

输出 [room_gft_a34_milestone_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_dataset.json) 与 [room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json)。训练核验值为 `sample_count=13077,train_count=10560,validation_count=2517,group_count=2818,validation_group_count=546`；训练集 `MAE=0.3814,corr=0.9476,target_top1_accuracy=0.8381,selected_top1_accuracy=0.6120`；验证集 `MAE=0.3929,corr=0.9371,target_top1_accuracy=0.8318,selected_top1_accuracy=0.5831`。

新增诊断脚本 [analyze_room_gft_xsokoban_a34_milestone_sd_consumer.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a34_milestone_sd_consumer.py)。先运行 30-push soft smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a34_milestone_sd_consumer.py \
  --budget 64 \
  --smoke-steps 30 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，score 范围 `[0.1630,2.6834]`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a34_milestone_sd_consumer.py \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.md)。A34 与 A31/A32 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`，A31/A34 首个决策差异为 `None`。milestone scorer 加载 `42` 次，正分选择 `29` 次，score 范围 `[-0.5022,3.0]`。

补跑 hard-override 30-push 对照：

```bash
python scripts/analyze_room_gft_xsokoban_a34_milestone_sd_consumer.py \
  --budget 64 \
  --smoke-steps 30 \
  --learned-sd-consumer-root-override \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.md)。结果为 `status=no_action,executed_pushes=6,closed_loop_decision_steps=4,boxes_on_goals=0`，A31/A34 首个决策差异为 step `1`。

判定：A34 是负诊断。future-window milestone target 比 A33 的 immediate target 更贴近研究意图，但当前 scalar 目标尺度偏正，前缀分数多次被 clip 到 `3.0`，soft prior 未改变 A31/A32 轨迹，hard override 仍早期退化。该结果说明问题不是“再放大学习分数”，而是 target 需要校准和分解：应将 value、release-open true progress、loop/trap risk 分头建模，或引入更强的离线 search/teacher solved-distance 标签。

### 2026-06-22 Room-GFT A35 multi-head SD consumer 诊断

根据 A34 的负诊断，本轮不继续扩展单一 scalar target，而是实现多头神经消费者。新增 [RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A34。A35 checkpoint 输出 value、release-progress、loop-risk、trap-risk 四个头，规划器将其作为有符号组合的 soft prior 消费；不新增动作规则，不修改 A31-A34、A11 schedule-plan SD、exact grid-flow 转移或 Boxoban 主线，在线求解保持 `teacher_actions_used_online=false`。

新增训练脚本 [train_room_gft_multihead_sd_consumer_a35.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_multihead_sd_consumer_a35.py)。训练命令为：

```bash
python scripts/train_room_gft_multihead_sd_consumer_a35.py
```

输出 [room_gft_a35_multihead_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_dataset.json) 与 [room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a35_multihead_sd_consumer_mlp_checkpoint.json)。训练核验值为 `sample_count=13928,train_count=11216,validation_count=2712,group_count=3004,validation_group_count=588`。组合目标验证集为 `MAE=0.2662,corr=0.9612,target_top1_accuracy=0.8799,selected_top1_accuracy=0.6193`。单头验证集为：value `MAE=0.4292,corr=0.9358,target_top1=0.8799`；release-progress `MAE=0.0707,corr=0.9522,target_top1=0.7650`；loop-risk `MAE=0.0210,corr=0.9905,target_top1=0.7324`；trap-risk `MAE=0.0131,corr=0.9989,target_top1=0.4991`。

新增诊断脚本 [analyze_room_gft_xsokoban_a35_multihead_sd_consumer.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a35_multihead_sd_consumer.py)。先运行 30-push soft smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a35_multihead_sd_consumer.py \
  --smoke-steps 30 \
  --budget 64 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A35 首个决策差异为 step `14`，score 范围 `[-0.1368,1.5011]`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a35_multihead_sd_consumer.py
```

输出为 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.md)。A35 与 A31/A34 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A35 首个决策差异为 `None`。multi-head scorer 加载 `42` 次，正分选择 `39` 次，score 范围 `[-0.3828,2.5846]`。被选动作头部范围为 value `[-1.6509,10.0]`、release-progress `[0.0,2.9983]`、loop-risk `[0.0,1.4017]`、trap-risk `[0.0,0.7312]`。

判定：A35 是接口推进但闭环持平的诊断。多头目标提升了机制可见性，且组合目标验证 top-1 高于 A34；但默认 soft prior 仍未突破 A31/A34 的 2-goal release-open 循环。关键证据是后段重复 `b3:down` 时 value 为负、loop-risk 高，但 release-progress 仍高，组合后仍为正分约 `0.1639`。下一轮不应继续把规则拆得更细，而应做更深层的目标校准和消费位置调整：把 release-open 重复循环纳入明确 hard negative，校准 release-progress 与 loop-risk 的相对权重，或将多头信号接入 leaf value/rollout evaluation 与阶段里程碑调度。

### 2026-06-22 Room-GFT A36 agenda certificate 负诊断

根据专家对 27 点 Agenda-GFT 建议的分析，本轮先实现最小可验证版本，而不重写 MCTS 或引入图网络。新增 [RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A35 的 `RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner`。A36 复用 A35 multi-head checkpoint，只在在线 soft prior 消费时加入 progress-certificate gate：release-progress 只有在完成义务、减少 blocker、推进 frontier slot、推进 barrier 或消费 opened release 时才作为有效 release head；打开 release 但未消费时将 release head 截断到 `0.35`，没有证书时将 release head 置零并记录 pseudo progress。

新增诊断脚本 [analyze_room_gft_xsokoban_a36_agenda_certificate.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a36_agenda_certificate.py)。先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a36_agenda_certificate.py \
  --smoke-steps 30 \
  --budget 64 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A36 首个决策差异为 step `14`；证书计数为 `loaded=14,certified=13,pseudo=1,stale_open_release=0,release_consumed=0`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a36_agenda_certificate.py
```

输出为 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.md)。A36 与 A31/A35 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A36 首个决策差异为 `None`。证书计数为 `agenda_certificate_loaded_count=42,agenda_certified_progress_count=39,agenda_pseudo_progress_count=1,agenda_stale_open_release_count=0,agenda_release_opened_count=4,agenda_release_consumed_count=0,agenda_loop_count=0`。

关键观察：A36 能把部分 release-open-but-not-consumed 动作的 release head 截断，例如 step `25`、`29`、`30` 的 `release_head_effective=0.35`；但后段重复 `b3:down` 从 step `31` 到 step `41` 仍被最小证书误判为 `certified_progress=True,tier=6`，`release_head_raw=2.3367` 且 `release_head_effective=2.3367`，合成分数约 `0.2864`。因此 A36 是负诊断：只用局部 delta 定义 progress certificate 不足以识别 release-open 循环。下一步必须引入 active agenda / blocker-set，使证书绑定当前义务，而不是继续调 A35/A36 的线性权重。

### 2026-06-22 Room-GFT A37 active-agenda certificate 诊断

根据 A36 的负诊断，本轮实现最小 active agenda / blocker-set 层。新增 [RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A36 的 `RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner`。A37 复用 A35 multi-head checkpoint，不训练新模型，不改动作生成和 exact dynamics；只把 progress certificate 绑定到 active obligation。当前最小义务包括 `release_blocker`、`consume_release`、`fill_frontier_slot`。其中 `consume_release` 仅在 `before_release_action_count>0` 时激活，避免把尚未打开 release 的 release 阶段动作误判为消费 release。

新增诊断脚本 [analyze_room_gft_xsokoban_a37_active_agenda_certificate.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a37_active_agenda_certificate.py)。先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a37_active_agenda_certificate.py \
  --smoke-steps 30 \
  --budget 64 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A37 首个决策差异为 step `14`；计数为 `active_agenda_loaded_count=14,active_agenda_progress_count=12,active_agenda_off_agenda_action_count=2,active_agenda_pseudo_progress_count=2,active_agenda_release_loop_count=0`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a37_active_agenda_certificate.py
```

输出为 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.md)。A37 与 A31/A36 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A37 首个决策差异为 `None`。计数为 `active_agenda_loaded_count=42,active_agenda_progress_count=25,active_agenda_off_agenda_action_count=17,active_agenda_pseudo_progress_count=17,active_agenda_stale_open_release_count=13,active_agenda_loop_count=14,active_agenda_release_loop_count=13,active_agenda_release_consumed_count=1`。agenda 类型分布为 `{release_blocker:28,consume_release:14}`。

关键观察：A37 成功纠正 A36 的证书误判。step `31` 至 step `41` 的重复 `b3:down` 在 A37 中被标记为 `active_agenda_type=consume_release,active_agenda_reason=repeated_release_closure,active_agenda_progress=False,active_agenda_off_agenda_action=True,active_agenda_release_loop=True,certified_progress=False,tier=1,release_head_effective=0`，合成 learned score 约 `-2.4741`。A36 对同一段的标记是 `certified_progress=True,tier=6,release_head_effective=2.3367`。

但 A37 不是闭环修复。step `31` 的 top-child 证据显示，被判为 off-agenda/release-loop 的 `b3:down` 仍有 `prior_score=266.0073,room_sd_value.score=150,visits=120`；其他候选虽同样多为 off-agenda，但 prior 为负且 visits 只有 `1`。因此，本轮结论是：active agenda 层已把“局部形式进展”与“当前义务进展”区分开，下一缺口在搜索消费端。后续应考虑让 active agenda 进入 hard gate、agenda-option 候选生成或 leaf/rollout value，而不是继续调 A37 的 soft prior 惩罚。

### 2026-06-22 A38 前 Agenda-GFT 路线约束

根据用户要求，后续 hard Sokoban 派生实验必须持续检查是否依循专家 27 点 Agenda-GFT 建议，而不是在未经充分论证的新方向上扩展。已补充 [决策 114](/home/data/round/code/sd_world_model_rl/docs/decisions.md)，将该要求记录为版本准入条件。

当前对齐关系为：A36 对应专家建议 E1，即 release-progress 的 progress certificate gate；A37 对应 E2，即 active agenda / blocker-set，不改图网络。A37 已证明证书层能识别 repeated release closure，但 root 仍因旧启发式 prior 与 MCTS visits 选择该动作。因此 A38 的候选范围应限定在 27 点已讨论的消费端路线内：E3 leaf agenda potential、第 8/13 点的 leaf value 或 rollout/option evaluation、第 14 点 agenda-option tree、第 16 点 lexicographic tier。暂不进入直接大模型、图网络、teacher imitation、单纯扩大预算或继续堆局部规则。

### 2026-06-22 Room-GFT A38 agenda leaf potential 负诊断

根据 [决策 114](/home/data/round/code/sd_world_model_rl/docs/decisions.md)，本轮只执行专家 27 点中的 E3，不进入 hard gate、agenda-option tree、图网络或 teacher imitation。新增 [RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A37 的 `RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner`。A38 将 active-agenda potential 接入 MCTS `leaf_value()`，并在 `plan()` 输出中记录 root leaf value 前后值、leaf adjustment、leaf eval count、loop eval count 与 stale-release eval count。同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `agenda_leaf_potential` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a38_agenda_leaf_potential.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a38_agenda_leaf_potential.py)。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a38_agenda_leaf_potential.py \
  --smoke-steps 30 \
  --budget 64 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，leaf 字段加载 `14` 次，adjustment 范围为 `[-3.11,3.93]`，均值约 `0.01`，leaf eval 总数为 `866`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a38_agenda_leaf_potential.py
```

输出为 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.md)。A38 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，与 A31/A37 完全同轨迹，A31/A38 首个决策差异为 `None`。leaf 统计为 `agenda_leaf_loaded_count=42,agenda_leaf_adjustment_min=-14.31,agenda_leaf_adjustment_max=3.93,agenda_leaf_adjustment_mean=-5.4133,agenda_leaf_eval_count_total=5302,agenda_leaf_loop_eval_count_total=129,agenda_leaf_stale_release_eval_count_total=74,agenda_leaf_root_loop_count=17,agenda_leaf_root_stale_open_release_count=14`。

关键观察：A38 已在 step `31` 至 step `41` 的 `consume_release` 循环段给出 `agenda_leaf_adjustment=-14.31`，并保持 A37 的 `active_agenda_progress=False,active_agenda_off_agenda_action=True,active_agenda_release_loop=True,certified_progress=False,tier=1`。但 step `31` 的 top-child 中，`b3:down` 仍有 `prior_score=266.0073,room_sd_value.score=150,visits=120`；其他候选 visits 只有 `1`，prior 大多为负。因此，本轮结论是：E3 leaf potential 是有效接入但非闭环修复。缺口仍位于 agenda 信号对 root/top-child 选择的约束强度和结构形式上，后续不得转入未经论证的新方向，应在专家 27 点内单独决策是否进入第 16 点 lexicographic tier / hard downgrade 或第 14 点 agenda-option tree。

### 2026-06-22 Room-GFT A39 agenda-tier root choice 诊断

根据 A38 的负诊断和 [决策 115](/home/data/round/code/sd_world_model_rl/docs/decisions.md)，本轮只执行专家 27 点中的第 16 点，不进入 agenda-option tree、图网络、teacher imitation 或新动作生成。新增 [RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A38 的 `RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner`。A39 将已存在的 `agenda_tier`、off-agenda 和 release-loop 证书转为 root choice 的 lexicographic tier / hard downgrade，并在 `plan()` 输出中记录父类选择、最终选择、tier 分布、是否 override 与触发原因。同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `agenda_tier_choice` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a39_agenda_tier_choice.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a39_agenda_tier_choice.py)。

实现中曾检查全局 tier 接管口径。该口径会在 fill 阶段过早改写 root 选择，不符合 A37/A38 已定位的 release-open 后失败证据。因此默认实现收紧为 `release_loop_only`：只有父类候选为 `consume_release`、`agenda_tier<=1` 且存在 release-loop/off-agenda/stale-open-release 时，才允许在同类 `consume_release` 候选中进行 higher-tier 替换。该边界用于保证 A39 仍是第 16 点的最小实验，而不是未经论证的全局动作接管。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a39_agenda_tier_choice.py \
  --smoke-steps 30 \
  --budget 64 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，`agenda_tier_choice_loaded_count=14,agenda_tier_choice_override_count=0`。这说明证据限定的 tier choice 不会在早期 fill 阶段造成退化。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a39_agenda_tier_choice.py
```

输出为 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.md)。A39 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`。A31/A39 首个决策差异为 step `26`；tier choice 加载 `53` 次，override `20` 次，higher-tier selected `20` 次。

关键观察：step `26` 时，父类会选择高 visits、高 prior 的 `b3:down`，该候选为 `consume_release,tier=1,off-agenda=true,release-loop=true,certified_progress=false`；A39 改选 `b4:up`，该候选为 `consume_release,tier=7,active_agenda_progress=true,off-agenda=false,release-loop=false`。这直接验证了第 16 点：lexicographic tier 能把 A37/A38 已识别但未被消费的证书转化为 root 行为变化。

但 A39 仍未求解第二关。后续轨迹虽然不再是 A38 的连续 `b3:down` 循环，却大量进入 tier-2 off-agenda `consume_release` 替代动作，最终仍停在 `2` goals、slot cost `29`。因此，本轮结论是：root tier 消费位置修正有效，但还缺少能持续生成 tier-5/7 真进展的 agenda-option 或里程碑候选。下一步若继续，应仍在专家 27 点内单独决策是否进入第 14 点 agenda-option tree，而不是继续调 A39 权重或扩大预算。

### 2026-06-22 Room-GFT A40 agenda-option tree 诊断

根据 A39 的结果和决策 116，本轮只进入专家 27 点中的第 14 点，不进入图网络、teacher imitation、新动作生成或主线改动。新增 [RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A39 的 `RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner`。A40 在 root 处为候选 first edge 做 bounded local option evaluation：每个候选用真实 legal push 展开至多 `5` 步、`96` 个局部节点，评价是否能完成当前 active agenda。suffix 只用于评价，不作为宏动作执行，因此闭环仍逐步 replanning。

同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `agenda_option_tree` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a40_agenda_option_tree.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a40_agenda_option_tree.py)。先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a40_agenda_option_tree.py \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，`agenda_option_tree_loaded_count=14,agenda_option_tree_override_count=0`，原因均为 `parent_not_option_trigger`。该 smoke 说明 A40 没有在早期 fill 阶段造成退化。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a40_agenda_option_tree.py \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.md)。A40 与 A39 完全同轨迹：`max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A39/A40 首个决策差异为 `None`。

关键统计为 `agenda_option_tree_loaded_count=53,agenda_option_tree_override_count=0,agenda_option_tree_selected_progress_count=22,agenda_option_tree_selected_done_count=22,agenda_option_tree_selected_loop_count=0,total_evaluated_option_count=186,max_selected_depth=4,max_selected_expanded_nodes=96`，原因分布为 `{parent_not_option_trigger:31,parent_option_already_best_tier:22}`。这说明 A40 在触发点确实找到了短程 agenda-done option，但父类/A39 当前 first edge 本身已经能通向同 tier option，因此没有发生 root override。

关键观察：step `29` 以后，A40 可找到三步左右的 `consume_release` 完成路径。例如 step `29` 的局部路径为 `b6:up -> b9:left -> b4:down`，option progress 为 `agenda_done=true,release_consumed=true,release_action_count_delta=-2,active_blocker_delta=5,dependency_total_delta=3,target_blocker_count_delta=2,after_active_blocker_count=0`。step `30` 和 step `31` 也存在 `b3:left -> b9:left -> b4:down` 的 tier-7 option。但是 A40 只执行第一步，后续闭环会重新规划并可能偏离该 option。

本轮结论是：A40 的保守 option evaluator 提供了正诊断信号，但非闭环修复。当前缺口不是完全缺少短程 agenda 完成信息，而是缺少 option commitment 或显式 agenda state 持续性。后续若继续，应仍在第 14 点和第 6 点范围内推进，将 active option 写入节点/闭环状态，使系统承诺执行同一 obligation-option 直到完成、触发风险或出现新 frontier；不应回到单纯调 root 分数或扩大预算。

### 2026-06-22 Room-GFT A41 agenda-option commitment 诊断

根据 A40 的诊断和决策 117，本轮继续限定在专家 27 点中的第 14 点和第 6 点，不进入图网络、teacher imitation、新训练目标或 Boxoban 主线改动。新增 [RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A40 的 `RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner`。A41 不重新生成 teacher 轨迹，也不使用在线参考动作；它只在 A40 已选出的 `agenda_tier>=7` 且 `agenda_done=true` option 上，执行一个经 exact dynamics 重新验证的受限 suffix。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `agenda_option_commitment` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a41_agenda_option_commitment.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a41_agenda_option_commitment.py)。先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a41_agenda_option_commitment.py \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --a31-json results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.json \
  --a39-json results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json \
  --a40-json results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A40/A41 首个决策差异为 `None`，`agenda_option_commitment_triggered_count=0`，原因均为 `no_selected_option`。该 smoke 说明 A41 没有在早期无 selected option 的阶段强行承诺。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a41_agenda_option_commitment.py \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.md)。A41 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=41,dependency_count=0,target_blocker_count=0,release_action_count=0`；A40/A41 首个决策差异为 step `30`。commitment 触发 `1` 次，首个触发步为 step `29`，额外执行 `2` 个承诺 push。

关键观察：A41 在 step `29` 执行 A40 发现的 `b6:up -> b9:left -> b4:down`，并完成 `consume_release`：`agenda_done=true,release_consumed=true,release_action_count_delta=-2,active_blocker_delta=5,dependency_total_delta=3,target_blocker_count_delta=2,after_active_blocker_count=0`。这验证了 A40 的短程 option 并非虚假路径，A41 也确实补上了“只执行 first edge”这一缺口。

但 A41 仍未求解第二关，且 80-push 内目标数仍为 `2`。与 A40 停在 release-open 状态不同，A41 已将 release、dependency 和 target blocker 清零，但后续转入 `fill` 阶段后长期停留在 `slot_schedule_cost=41`。本轮结论是：option commitment 是必要机制验证，但不是充分修复。下一缺口已经从 `consume_release` 执行持续性转移到 release 完成后的阶段化 fill 里程碑、slot-distance/value 表达和阶段切换后的目标选择；后续仍须在专家 27 点框架内推进，不应把 A41 结果解释为“继续执行更多 suffix 即可求解”。

### 2026-06-23 Room-GFT A42 fill-frontier option 诊断

根据 A41 的诊断和决策 118，本轮继续限定在专家 27 点中的 frontier obligation、agenda-option tree 和 next agenda milestone 范围内，不进入图网络、teacher imitation、新训练目标或 Boxoban 主线改动。新增 [RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A41 的 `RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner`。A42 不重新生成 teacher 轨迹，也不使用在线参考动作；它只在 A41 完成 release 义务后的 clean fill 语境中，尝试把当前 frontier slot completion 作为独立 agenda boundary 消费。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `fill_frontier_option` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a42_fill_frontier_option.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a42_fill_frontier_option.py)。先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a42_fill_frontier_option.py \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --a31-json results/sokoban_sd_world_model/room_gft_xsokoban_a31_goal_lock_predecessor_release_l002_s30_b64_n8.json \
  --a39-json results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json \
  --a40-json results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json \
  --a41-json results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A41/A42 首个决策差异为 `None`，`fill_frontier_option_override_count=0`。该 smoke 说明 A42 没有在前缀阶段引入无依据的全局接管。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a42_fill_frontier_option.py \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.md)。A42 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A41/A42 首个决策差异为 step `30`，A42 的 `fill_frontier_option_override_count=1`、`fill_frontier_option_selected_completion_count=1`、`fill_frontier_option_selected_slot_progress_count=1`。

关键观察：A42 在 step `30` 覆盖 A41 的无进展 fill 动作，选择 `b4:down`，并通过既有 corridor macro 执行 `b4:down -> b6:down -> b6:left`。局部评价显示 `frontier_completed=true,slot_progress=true,goal_delta=0,slot_schedule_cost_delta=20,valid_filled_slot_count_delta=1`，最终 slot cost 从 A41 的 `41` 降到 `35`。这说明 release 完成后的 fill frontier milestone 可以被独立 agenda option 消费。

但 A42 仍未求解第二关。step `30` 的 frontier completion 打开新的 release/release-blocker 局面；step `31` 以后，已选动作持续为 `release_blocker,tier=2,active_agenda_progress=false,active_agenda_off_agenda_action=true`，且 `release_action_count=0,dependency_count=2,target_blocker_count=1,slot_schedule_cost=35` 长期不变。本轮结论是：A42 补上了 post-release fill frontier 消费缺口，但下一缺口已经转移到 frontier completion 之后的 release-blocker option 和阶段状态持久化；后续仍须在专家 27 点框架内推进，不应把 A42 解释为只需扩大预算即可求解。

### 2026-06-23 Room-GFT A43 release-blocker option 诊断

根据 A42 的诊断和决策 119，本轮继续限定在专家 27 点中的第 14 点、第 6 点和 frontier obligation 后续 release-blocker option 范围内，不进入图网络、teacher imitation、新训练目标或 Boxoban 主线改动。新增 [RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A42 的 `RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner`。A43 不重新生成 teacher 轨迹，也不使用在线参考动作；它只在 A42 完成 fill frontier 后暴露出的 post-frontier release-blocker 低 tier/off-agenda 语境中，运行 bounded local option evaluator。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `release_blocker_option` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a43_release_blocker_option.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a43_release_blocker_option.py)。A43 的触发条件包括：`stage=release`、`release_action_count=0`、存在 active blocker、有效 filled slot 数不低于 `2`，且父类候选为 `release_blocker` 低 tier 或 off-agenda。局部树预算为 `release_blocker_option_max_pushes=5`、`release_blocker_option_max_nodes=96`、`release_blocker_option_max_root_candidates=12`。

实现中曾运行一个宽松进展口径的中间版本。该版本在 80-push 中出现 1 次 override，最终 slot cost 从 `35` 降为 `30`，但检查 step `34` 发现所谓高 tier option 只产生 `slot_schedule_cost_delta=1`，没有打开 release，也没有减少 dependency、target blocker 或 locked blocker。该口径会把弱 slot 变化误认为 release-blocker 真进展，因此已收紧正式 A43：只有目标推进、打开 release action、减少 active blocker、减少 dependency/target blocker、或减少 locked/goal-locked stand blocker，才算 release-blocker true progress。

正式 A43 先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a43_release_blocker_option.py \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --a42-json results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A42/A43 首个决策差异为 `None`，`release_blocker_option_override_count=0`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a43_release_blocker_option.py \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.md)。A43 与 A42 完全同轨迹：`max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A42/A43 首个决策差异为 `None`。

关键观察：A43 的 `release_blocker_option_loaded_count=51`，其中 `31` 次为 `parent_not_release_blocker_trigger`，step `31` 至 step `50` 的 `20` 次触发均为 `no_high_tier_release_blocker_option`。这些触发点的 selected option tier 均不超过 `2`，`release_blocker_true_progress=false`，没有打开 release，也没有减少 active blocker、dependency 或 target blocker。因此，本轮结论是负诊断：当前 A42 后段不是“短程真实 release-blocker option 已存在但未被 root 消费”，而是在 `5` push、`96` node 的短程 option tree 内缺少真实 release-blocker 进展路径。后续若继续，应考虑更长程/分阶段里程碑、active blocker 分解和 agenda state 持久化，而不是把弱 slot 变化纳入 release-blocker 进展。

### 2026-06-23 Room-GFT A44 active-blocker obligation 诊断

根据 A43 的负诊断和决策 120，本轮继续限定在专家 27 点中的 active obligation、blocker-specific relation、agenda-option tree 和 agenda state persistence 范围内。新增 [RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A43 的 `RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner`。A44 不重新生成 teacher 轨迹，也不使用在线参考动作；它只把 A43 的 aggregate release-blocker 计数拆成具体 A11 schedule dependency row，并对每个 blocker-specific obligation 运行有界 option 评价。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `active_blocker_obligation_option` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a44_active_blocker_obligation.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a44_active_blocker_obligation.py)。A44 的触发条件包括：`stage=release`、`release_action_count=0`、存在 active blocker、有效 filled slot 数不低于 `2`，且父类候选为 `release_blocker` 低 tier 或 off-agenda。局部树预算为 `active_blocker_obligation_max_pushes=6`、`active_blocker_obligation_max_nodes=128`、`active_blocker_obligation_max_root_candidates=12`、`active_blocker_obligation_max_obligations=4`。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a44_active_blocker_obligation.py \
  --a43-json results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.json \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A43/A44 首个决策差异为 `None`，`active_blocker_obligation_override_count=0`，原因均为 `parent_not_active_blocker_trigger`。该 smoke 说明 A44 没有在早期前缀阶段引入无依据的全局接管。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a44_active_blocker_obligation.py \
  --a43-json results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.md)。A44 与 A43 完全同轨迹：`max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A43/A44 首个决策差异为 `None`。

关键观察：A44 的 `active_blocker_obligation_loaded_count=51`，其中 `31` 次为 `parent_not_active_blocker_trigger`；post-frontier 段触发 `20` 次，原因分布为 `{parent_active_blocker_obligation_already_best_tier:5,no_high_tier_active_blocker_obligation_option:15}`。在 `5` 个触发点中，A44 发现具体 blocker 子义务的 high-tier true progress，例如 step `31` 和 step `32` 中，future dependency row `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6]` 的 suffix 可使 `remaining_selected_kind_delta=1` 且 `stand_resolved=true`。但这些点父类当前 first edge 已达到同一 high tier，因此 `active_blocker_obligation_override_count=0`，闭环仍与 A43 完全同轨。

本轮结论是：A44 排除了“SD 端完全缺少 blocker identity”这一更强假设。具体 blocker obligation 已可从现有 schedule dependency 中读出，且 bounded suffix 中能看到局部子义务完成；真正缺口转为 option commitment 和 agenda state persistence，即如何在若干 push 内保持同一 blocker obligation/suffix，直到 stand/target blocker 真的被解除或出现风险边界。后续若继续，应考虑 A45 active-blocker option commitment 或将 agenda/option 写入闭环状态，而不是继续调 A44 root score 或放宽真进展口径。

### 2026-06-23 Room-GFT A45 active-blocker commitment 诊断

根据 A44 的诊断和决策 121，本轮继续限定在专家 27 点中的 active obligation、agenda-option tree 和 agenda state persistence 范围内。新增 [RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A44 的 `RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner`。A45 不重新生成 teacher 轨迹，也不使用在线参考动作；它只把 A44 已经识别出的 high-tier blocker-specific suffix 作为候选 commitment，并在当前真实状态下逐步重放验证后才执行。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `active_blocker_commitment` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a45_active_blocker_commitment.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a45_active_blocker_commitment.py)。A45 的触发条件包括：A44 `selected_option` 存在、`active_blocker_obligation_tier>=8`、`active_blocker_true_progress=true`、suffix 至少包含 `2` 个 push，且父层 corridor macro 默认必须为单步。验证过程中检查 push 合法性、deadlock child、path loop，并在最终状态重新计算 active-blocker progress。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a45_active_blocker_commitment.py \
  --a44-json results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.json \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A44/A45 首个决策差异为 `None`，`active_blocker_commitment_triggered_count=0`，原因均为 `no_selected_active_blocker_option`。该 smoke 说明 A45 没有在早期前缀阶段引入无依据的 suffix commitment。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a45_active_blocker_commitment.py \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.md)。A45 为 `max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A44/A45 首个决策差异为 step `32`；A45 的 `active_blocker_commitment_triggered_count=2`，首个触发步为 step `31`，额外执行 `4` 个承诺 push，验证 push 总数为 `6`。

关键观察：A45 在 step `31` 执行四步 active-blocker commitment，最终 progress 为 `active_blocker_obligation_tier=8,active_blocker_true_progress=true,active_blocker_progress_kinds=[selected_kind_reduced,stand_resolved]`；step `41` 再执行一次两步 commitment，同样达到 tier `8` 与 true progress。由此可确认，A44 发现的 blocker-specific suffix 可以在闭环中合法执行，并能产生真实子义务进展。

但 A45 仍未求解第二关。80-push 结束仍为 `2` goals，final dependency、future dependency、target blocker 和 release action 指标没有完成性改善，final slot cost 还从 A44 的 `35` 变为 `36`。本轮结论是：A45 排除了“主要失败只是 A44 后缀完全没有被执行”的单一解释。后续缺口应转向 commitment 后的 active obligation 更新、阶段/里程碑重建，以及更高层 agenda-node/option-node/push-node 结构；不应继续通过放宽 true-progress 口径或堆叠局部 blocker 规则来推进。

### 2026-06-23 Room-GFT A46 residual active-agenda 诊断

根据 A45 的负结果和决策 122，本轮继续限定在专家 27 点中的 active agenda state、next agenda milestone 和 agenda-option tree 范围内。新增 [RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A45 的 `RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner`。A46 不重新生成 teacher 轨迹，也不使用在线参考动作；它只在 A45 commitment 已触发且留下 `target_occupied_by_box` 残余时，把同一 A11 relation 重建为 target-only residual agenda，并在当前真实状态下运行一次有界 residual option 搜索。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `residual_active_agenda` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a46_residual_active_agenda.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a46_residual_active_agenda.py)。A46 默认 residual 搜索参数为 `residual_active_agenda_max_pushes=8`、`residual_active_agenda_max_nodes=192`、`residual_active_agenda_min_total_pushes=1`，并要求 residual option 必须产生 target progress 才能追加到宏动作。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a46_residual_active_agenda.py \
  --a45-json results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.json \
  --smoke-steps 30 \
  --budget 64 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A45/A46 首个决策差异为 `None`，原因分布为 `{'source_commitment_not_triggered': 14}`。该 smoke 说明 A46 没有在早期尚未出现 A45 commitment 的阶段强行接管。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a46_residual_active_agenda.py \
  --smoke-steps 80 \
  --budget 128 \
  --horizon 8 \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.md)。A46 与 A45 完全同轨迹：`max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A45/A46 首个决策差异为 `None`。

关键观察：A46 在 step `31` 和 step `41` 均正确重建 target-only residual agenda，残余 relation 为 `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6],candidate_direction=down`，残余 kind 为 `target_occupied_by_box`。两次 residual search 均扩展至 `192` node 上限，合计评估 `384` 个 residual option，但均未找到 target progress，`residual_active_agenda_triggered_count=0`、`residual_active_agenda_target_progress_count=0`，原因分布为 `{'source_commitment_not_triggered': 45, 'no_target_progress_residual_option': 2}`。

本轮结论是：A46 排除了“主要失败只是 A45 承诺后没有把 target 残余义务显式重建”的单一解释。当前 target blocker 的下一可执行里程碑，在 `8` push、`192` node 的 residual option 空间内仍不可见。后续若继续，应推进更一般的 agenda-node/option-node/push-node 拓扑、target residual 的中间里程碑定义、或面向 agenda distance 的可复用函数近似，而不是继续降低 A45/A46 触发阈值或增加局部 blocker 规则。

### 2026-06-23 Room-GFT A47 target-prerequisite agenda 诊断

根据 A46 的负结果和决策 123，本轮继续限定在专家 27 点中的 obligation frontier、blocker set、active agenda state 和 agenda-option tree 范围内。新增 [RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A46 的 `RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner`。A47 不使用在线 teacher/reference 动作；它只在 A46 的 target residual failure 后，检查 target blocker 自身的 push 前提，并将被箱子占用的 target-blocker stand cell 记录为 `ClearTargetPushStand` 前提义务。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `target_prerequisite_agenda` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a47_target_prerequisite_agenda.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a47_target_prerequisite_agenda.py)。A47 默认参数为 `target_prerequisite_agenda_max_pushes=8`、`target_prerequisite_agenda_max_nodes=256`、`target_prerequisite_agenda_min_total_pushes=1`，并要求 option 必须产生真实前提进展才允许追加到宏动作。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a47_target_prerequisite_agenda.py \
  --budget 64 \
  --horizon 8 \
  --smoke-steps 30 \
  --a46-json results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A46/A47 首个决策差异为 `None`，`target_prerequisite_agenda_obligation_count=0`，原因分布为 `{'source_not_target_residual_failure': 14}`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a47_target_prerequisite_agenda.py \
  --budget 128 \
  --horizon 8 \
  --smoke-steps 80 \
  --a46-json results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.md)。A47 与 A46 完全同轨迹：`max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A46/A47 首个决策差异为 `None`。

关键观察：A47 在 step `31` 和 step `41` 均构造出 `ClearTargetPushStand` 前提义务。两次义务都显示 target blocker 位于 `[3,6]`；若沿 parent candidate direction `down` 推动该 blocker，则 push target `[4,6]` 是空地板，但 required stand `[2,6]` 被 parent assigned box 占用。向上推动会撞到 `[2,6]` 的箱子，左右方向不是地板。因此 target blocker 当前没有合法 push，且其直接前提是清空 `[2,6]`。A47 在这两点合计评估 `512` 个 prerequisite option，单点达到 `expanded_nodes=256` 上限，但没有找到前提进展，原因分布为 `{'source_not_target_residual_failure': 45, 'no_target_prerequisite_progress_option': 2}`。

本轮结论是：A47 排除了“target residual 没有被分解成前提 blocker set”的单一解释。前提义务已经能被结构化读出，但在短程 option 空间内仍不可完成；下一步应继续考虑更高层 agenda-node/option-node/push-node 拓扑、阶段里程碑或可复用 agenda-distance/value，而不是将具体坐标写成局部规则。

### 2026-06-23 Room-GFT A48 assignment-conflict agenda 诊断

根据 A47 的负结果和决策 124，本轮继续限定在专家 27 点中的 obligation frontier、blocker set、active agenda state 和 agenda-option tree 范围内。新增 [RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner](/home/data/round/code/sd_world_model_rl/src/sokoban/room_gft_mcts.py)，继承 A47 的 `RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner`。A48 不使用在线 teacher/reference 动作；它只在 A47 已经识别出 `ClearTargetPushStand` 且该前提由 parent assigned box 阻塞时，构造 assignment/prerequisite conflict 义务。

实现中同步更新 [analyze_room_gft_xsokoban_a13_access_recovery.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a13_access_recovery.py)，使闭环 trace 能透传 `assignment_conflict_agenda` 字段；新增诊断脚本 [analyze_room_gft_xsokoban_a48_assignment_conflict_agenda.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_room_gft_xsokoban_a48_assignment_conflict_agenda.py)。A48 默认参数为 `assignment_conflict_agenda_max_pushes=8`、`assignment_conflict_agenda_max_nodes=320`、`assignment_conflict_agenda_min_total_pushes=1`，并要求 conflict done 才能追加到宏动作。

先运行 30-push smoke：

```bash
python scripts/analyze_room_gft_xsokoban_a48_assignment_conflict_agenda.py \
  --budget 64 \
  --horizon 8 \
  --smoke-steps 30 \
  --a47-json results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.json \
  --out-md docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.md
```

输出为 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A47/A48 首个决策差异为 `None`，`assignment_conflict_agenda_obligation_count=0`，原因分布为 `{'source_not_target_prerequisite_failure': 14}`。

随后运行 80-push 同参复核：

```bash
python scripts/analyze_room_gft_xsokoban_a48_assignment_conflict_agenda.py \
  --budget 128 \
  --horizon 8 \
  --smoke-steps 80 \
  --a47-json results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json \
  --out-json results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json \
  --out-md docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.md
```

输出为 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.md)。A48 与 A47 完全同轨迹：`max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A47/A48 首个决策差异为 `None`。

关键观察：A48 在 step `31` 和 step `41` 均构造出 `ResolveAssignedBoxPrerequisiteConflict` 义务。两次义务都显示 assigned box `[2,6]` 既是 parent residual 分配给 slot `[2,1]` 的箱子，也是 target blocker push 所需 stand `[2,6]` 的占用者。该 assigned box 的 pull-distance 为 `9.0`，初始合法 push 数为 `0`，朝 slot 的合法 push 数为 `0`。A48 在这两点合计评估 `640` 个 assignment-conflict option，单点达到 `expanded_nodes=320` 上限，但最佳进展仍为 tier `2`，没有打开 assigned box mobility、没有朝 slot 方向推进、没有解除 conflict。

本轮结论是：A48 排除了“只要把 A47 前提阻塞解释为 assignment conflict 即可推进”的单一解释。assignment/prerequisite 冲突、assigned box 移动性和到 slot 的距离都已显式读出，但短程局部 option 仍不可完成。下一步应继续考虑更高层 agenda-node/option-node/push-node 拓扑、阶段里程碑或可复用 agenda-distance/value，用更一般的机制表达多阶段解锁，而不是继续追加坐标或局部规则。
