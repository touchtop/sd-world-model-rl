# Methods

## Hopper：SD 拟合与规划验证的当前方法口径

### 目标

当前 Hopper 主线的研究目标不再是设计奖励塑形，而是检验：

1. 当前 `SD transition core` 是否能够在 TD-MPC2 使用的状态-动作分布上保持足够的短视界动力学保真度；
2. 该 core 是否能够改善 TD-MPC2 的动作排序、规划稳定性与闭环样本效率。

因此，方法上必须明确区分以下两个对象：

1. `model fidelity`
2. `planning usefulness`

二者相关，但不再预设天然一致。

### 当前活跃训练口径

当前闭环 Hopper 主线采用如下设置：

1. `world_model_kind=sd_hopper`
2. `sd_struct_dim=17`
3. `sd_latent_dim=512`
4. `sd_use_transition_residual=true`
5. `sd_transition_residual_scale=4.0`
6. `sd_reward_mode=learned`
7. `sd_action_prior_enabled=false`
8. 不使用旧的结构化奖励塑形
9. `consistency_loss` 中的结构态与残差态默认按块等权，而不是按维度数加权
10. SD 结构态一致性损失使用 batch 目标标准差做逐通道归一化

对应脚本为：

- [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)
- [run_hopper_v5_200k_pureenv_sdinit_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh)

### 当前在线训练的损失结构

在 RL 闭环中，`SDWorldModel` 不是独立按监督目标训练，而是作为 TD-MPC2 world model 与 reward/Q 一起联合优化。当前总损失由以下部分组成：

1. `consistency_loss`
2. `reward_loss`
3. `value_loss`
4. `termination_loss`
5. `height_loss`（当前配置中基本关闭）

这意味着在线训练优化的是“对当前 TD-MPC2 闭环有用的 world model”，而不是“平均动力学误差最小的 SD 拟合器”。

截至 `2026-04-29` 的修复版本中，若 `world_model_kind=sd_hopper`，`consistency_loss` 不再直接对完整 512 维 latent 做统一均方误差，而是先拆成 17 维显式结构态与 learned residual latent 两个块。结构态块按当前 batch 中目标结构态的逐通道标准差进行归一化，并设置标量下限防止近常量通道数值爆炸；残差块保留普通均方误差。两个块默认等权合成，由 `sd_latent_consistency_struct_weight=1.0` 与 `sd_latent_consistency_resid_weight=1.0` 控制。

这一处理保留了 learned residual latent 对未建模动力机制的补偿作用，但避免 495 维残差块仅因维度更多而在 `consistency_loss` 中压过 17 维结构态。结构态的 batch 标准差归一化用于避免 `omega / qdot / vz` 等大尺度通道主导梯度，使 `H / touch_toe / touch_heel / z_min` 等接触相位相关通道在优化目标中保持可见。

截至 `2026-04-30` 的接口修复版本中，若显式启用 `sd_online_supervision_enabled=true`，共享 `SD transition core` 即 `model._dynamics` 不再放入 TD-MPC2 主优化器 `self.optim`，而只由 `sd_optim` 管理。主优化器继续更新 encoder、reward、Q、policy 以外的 residual heads 等 TD-MPC2 侧参数；`sd_optim` 单独更新 `_dynamics`。这样可以避免同一组 `_dynamics` 参数被两个 Adam 优化器分别维护动量和二阶矩，从而消除优化器状态错位。随后进一步补充 `sd_detach_core_in_main_loss=true` 默认口径：启用 online supervision 时，`model.next()` 中 `_dynamics` 生成的 `next_struct_base` 在主 consistency 路径中只作为 residual 合成基座使用，不再参与主损失反传。由此，主 `consistency_loss` 不再训练 `SD core`，而主要训练外层 residual heads 与 TD-MPC2 下游头；`SD core` 的训练来源是独立在线 `SD` 监督。

该版本还固定了两项接口口径。第一，趋势保护维度不再使用固定的绝对最小修正下限，`sd_trend_protect_min_correction` 默认改为 `0.0`，避免在 `ROOTZ / ZMIN` 等小信号段引入单位错配的虚假漂移。第二，`sd_reward_mode=learned` 时，`mixed_reward_target()` 也统一返回已经乘以 `sd_env_reward_scale` 的环境奖励目标，避免 learned、hybrid、curriculum 模式之间出现静默尺度不一致。

结构态 residual 仍被保留，因为其设计目的正是补偿系统动力学 core 难以完整表达的 Hopper 复杂动力机制。为避免 residual 失去可审计性，当前版本将 `sd_struct_residual_scale`、`sd_vx_residual_scale`、`sd_vz_residual_scale`、`sd_pitch_residual_scale`、`sd_pitch_rate_residual_scale`、`sd_knee_residual_scale` 显式配置化，并新增默认关闭的 `sd_struct_residual_penalty_coef`。该正则项作用于 `tanh(raw_struct_residual)` 的平方均值，用于后续显式消融“残差补偿”与“残差覆盖 SD core”的边界；当前默认关闭，以避免在未完成 smoke 与消融前把新的超参数混入主线对照。训练日志新增 `struct_residual_activity=mean(abs(tanh(raw_struct_residual)))`，用于在 smoke 阶段判断 residual 是否过强。

### 诊断方法

当前优先采用以下诊断：

1. `policy-distribution error`
2. `phase-sliced error`
3. `action ranking diagnostic`
4. `touch onset timing`
5. `consistency_loss` 的阶段趋势分析

其中：

1. `policy-distribution error` 用于判断随机监督初始化与当前 replay 分布之间的偏移；
2. `phase-sliced error` 用于识别 `landing / takeoff / high_omega / near_contact` 等关键相位；
3. `action ranking diagnostic` 用于判断模型是否能支持规划真正需要的动作排序；
4. `consistency_loss` 仅作为在线联合训练中的粗粒度诊断，不再被视为对“规划可用性”的充分代理。

### 当前方法论结论

截至 `2026-04-29`，当前主线采用以下解释框架：

1. 若 `consistency_loss` 下降，不自动推出规划质量提升；
2. 若 return 提升，不自动推出 `SD` 已被良好拟合；
3. 对 Hopper 而言，`landing / takeoff / high_omega / high_vz_impact` 的误差优先级高于平均 grounded 段误差；
4. 规划真正依赖的是任务相关的短视界动力学与动作排序，而不是统一意义上的平均 RMSE。

### 下一阶段推荐方法

若继续改训练方式，当前推荐方法不是构造两个竞争的 dynamics 头，而是：

1. 保持单一共享的 `SD transition core`；
2. 保留现有 RL 联合目标；
3. 额外增加一条独立的在线 `SD` 监督支路，直接作用于同一 core；
4. 该支路优先优化：
   1. 一步结构态误差，
   2. `1 / 3 / 5` 步 rollout 误差，
   3. `landing / takeoff / near_contact / high_omega` 相位加权误差。

这样可以在不污染 planner 接口的前提下，显式维护 `SD` 的短视界保真度。

### 当前已落地的最小实现

截至 `2026-04-29`，上述方向已先落地最小版本，其实现位于 [tdmpc2.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/tdmpc2.py)。

实现原则如下：

1. 不新增第二个 dynamics 头；
2. 继续复用当前 `SDWorldModel._dynamics` 作为唯一 transition core；
3. 启用 online supervision 时，`SDWorldModel._dynamics` 只由 `sd_optim` 管理，不再同时由 TD-MPC2 主优化器管理；
4. 启用 online supervision 时，主 consistency 路径默认 detach `_dynamics` 输出，使 `consistency_loss` 不再训练 `SD core`；
5. 在 TD-MPC2 每次 `_update()` 的主模型更新与 target-Q 更新之后，再执行一次独立的 `sd_optim` 监督更新；
6. 监督数据直接来自当前 replay batch 的真实 `obs, action, obs_next`。

### 当前在线监督支路的损失构成

当前最小在线监督支路由以下部分组成：

1. `sd_online_one_step_loss`：基于真实结构态的 `1-step` 转移误差；
2. `sd_online_rollout_loss`：基于同一 batch 的短视界 rollout 误差；
3. `sd_online_omega_loss`：对 `pitch_rate / omega` 通道的附加监督；
4. `sd_online_contact_loss`：对 `touch_toe, touch_heel` 接触通道的附加监督。

总损失为：

1. `sd_online_supervision_one_step_weight * one_step_loss`
2. `+ sd_online_supervision_rollout_weight * rollout_loss`
3. `+ sd_online_supervision_omega_weight * omega_loss`
4. `+ sd_online_supervision_contact_weight * contact_loss`

上述 `one_step_loss`、`rollout_loss`、`omega_loss` 与 `contact_loss` 均使用同一批目标结构态的逐通道标准差做尺度归一化。相位加权用于提高 `landing / takeoff / near_contact / high_omega` 的样本权重，通道尺度归一化用于校正不同物理量单位和数值范围造成的梯度偏置，二者作用不同。

当前尺度口径为：`sd_struct_loss_scale_mode=batch_std`，`sd_struct_loss_scale_floor=0.1`。即每个 replay batch 内先按目标结构态估计通道标准差，再用 `0.1` 作为统一下限。该尺度来自当前训练分布的目标统计量，不是手工指定的固定物理通道系数。

### 当前 rollout 监督视界

当前最小实现先支持：

1. `1-step`
2. `3-step`

原因不是方法论限制，而是当前 [buffer.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/common/buffer.py) 的 `sample()` 只返回 `horizon+1` 长度窗口，而当前 Hopper 训练配置中 `horizon=3`。因此，在不改采样接口的前提下，当前最自然、最干净的最小实现就是 `1-step + 3-step`。

若后续需要显式加入 `5-step` 监督，应同步修改 replay 采样窗口长度或额外增加独立监督采样路径。

### 当前关键相位加权

当前在线监督不是完全均匀加权，而是对以下相位提高权重：

1. `landing`
2. `takeoff`
3. `near_contact`
4. `high_omega`

其动机是：这些相位对 TD-MPC2 的动作排序与短视界规划最敏感，而它们也是此前 replay 诊断中误差最顽固的部分。

### 下一版本：planning-aware SD 权重入口

截至 `2026-04-30`，下一版本已加入默认关闭的 planning-aware 样本权重入口。该入口不直接改变 planner，也不把 `SD` 训练目标改成最大化回报，而是在现有结构态监督锚点上，根据 TD 目标或奖励目标对 replay 样本加权。

本阶段方法论名称暂定为 `Decision-Oriented System-Dynamics-Guided World Model`，中文为“面向决策的系统动力学引导世界模型”。该名称用于强调：`SD` 指系统动力学（System Dynamics），其来源是系统工程和系统仿真中的建模思想；这里的 `SD` 不是为了构建环境动力学的完备副本，而是提供一种有系统动力学归纳偏置的状态抽象，使 world model 在候选动作排序、关键相位判断和闭环控制上更有用。相应地，评价口径不应只依赖平均状态预测误差，还应包含动作排序质量、规划稳定性和闭环样本效率。

新增配置为：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

可选模式包括 `td_positive`、`td_abs`、`reward_positive` 与 `reward_abs`。启用后，权重会同时作用于 `1-step` 结构态监督和当前短视界 rollout 监督。权重按当前 batch 的 planning signal 均值归一化，即 `sd_online_planning_weight_coef=1.0` 时，未截断前的平均权重约为 `1`，避免该机制退化为整体提高 `SD` 学习率。若正值模式下当前 batch 没有正 signal，则回退为均匀权重。

该入口用于验证：相比平均状态预测误差，按 planning signal 重加权的 `SD` 监督是否更能改善动作排序和闭环样本效率。

该版本仍不是完整的 action-ranking consistency loss。完整排序目标需要额外构造动作 bank，并用真实短 rollout 或高保真离线诊断得到动作优劣标签；当前权重入口只是为该方向提供低风险的第一步。

## Sokoban：stock-flow-feedback SD 世界模型方法口径

### 目标

Sokoban/Boxoban 研究线用于研究抽象高层状态空间下的系统动力学世界模型设计方法，不以求解 Sokoban 本身为目标。该研究线的 `SD` 必须由存量、流量和反馈构成，初始形态为结构已知、部分参数未知，并服务于 `MBRL + SD-MCTS` 框架。

当前主设计见 [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)，审查记录见 [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md)。

### 抽象状态与转移

原始状态 `s_t` 先由编码器投影为抽象 SD 状态 `x_t`。首版 `x_t` 由六个核心存量构成：目标完成 `G_t`、未解决压力 `U_t`、机动性 `M_t`、可达容量 `R_t`、拥塞 `C_t` 和不可逆风险 `I_t`。goal-room pressure、plan progress 和 structural slack 暂作为派生特征、辅助头或第二阶段扩展，不进入首版核心 stock vector。

世界模型采用带 incidence matrix 的 stock-flow 形式：

```text
f_t = softplus(H_theta(x_t, a_t, c, h_t, b_t)) * gate(a_t, c, z_t)
x_{t+1} = clip_X(x_t + B f_t)
b_t = feedback_theta(x_t, a_t, c, y_t, sigma_t)
```

其中 `c` 为静态关卡结构，`f_t` 为非负流量，`B` 为已知的流量到存量影响矩阵，`b_t` 为反馈，`theta` 为需要通过真实交互、短 rollout 或离线轨迹学习的参数。该定义要求模型输出存量、流量、反馈、不确定性、可执行性和下一抽象状态，而不是只输出 `policy_score`、`heuristic_score` 或 `risk_score`。`I_t` 表示当前状态风险压力或概率，`risk_t` 表示动作后果或分支风险，二者在日志和损失中必须分开记录。

### 文献机制的角色

Junghanns 与 Schaeffer、Pereira、Lipovetzky、Guez、Orseau、Martin、Taufeeque 等文献中的 Sokoban/Boxoban 机制用于以下用途：

1. 结构先验：玩家可达域、push-level single-push 动作、目标区入口、cut-square、tunnel 和 macro 结构；
2. 观测代理：匹配残差、目标占用、合法推箱数、可达面积、goal-room pressure、path-channel 或计划通道 proxy；
3. 流量门控：动作合法性、目标区填充条件、拥塞释放条件和风险触发条件；
4. 风险标签：dead-square、静态墙角、trap/invariant、局部 PDB 或短 rollout 失败；
5. 诊断项：matching lower bound、deadlock evidence、relevance proxy 和 macro candidate。

这些机制不再被定义为 SD 的核心启发式价值函数。

### SD-MCTS 接口

SD-MCTS 在学习到的抽象 SD 动力学上规划。首版动作集合限定为合法 single-push option；macro 只作为诊断代理或第二阶段扩展。MCTS 节点保存 `x_t` 或其分布、uncertainty、raw-state handle 或 belief support，扩展时调用 SD 世界模型预测后继存量、流量、任务进展、风险、不确定性和可执行性；选中的 single-push option 再映射为低层推箱动作。若使用文献机制产生 action prior，该 prior 只能作为学习模型或策略头的输入，不能替代 learned SD transition。正式日志必须记录 rollout transition provider，用于确认正式线没有退回到精确 push dynamics 加手工评分。

### 当前诊断草稿

为保持既有成果可按原路径复现，当前仍保留新增包路径：

1. `src/sokoban/env.py`
2. `src/sokoban/sd_model.py`
3. `src/sokoban/search.py`
4. `experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py`

该原型实现了精确 push-level 转移、玩家可达域、合法推箱、reverse-pull 距离、匹配下界、dead-square 与静态墙角诊断，并在 `results/sokoban_sd_world_model/eval_medium_train000_50.json` 中记录了 50 关卡 smoke 结果。该结果只能说明结构代理具有诊断信号；它没有学习参数化流量，没有反馈回路，也没有 SD-MCTS，因此不作为完整 SD 世界模型结果或 Boxoban benchmark 结论。

### 验证性 smoke 口径

在进入 learned flow training 和 SD-MCTS 前，先运行设计验证 smoke。对应入口为：

1. `experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py`

该脚本不训练神经网络，也不报告求解性能。它使用真实 single-push transition 构造六核心存量代理标签，检查以下内容：

1. 存量范围是否保持在 `[0, 1]`；
2. 诊断流量是否非负；
3. `B f_t` 是否能重构观测到的抽象存量变化；
4. 风险标签是否按 `proven_dead_end / short_rollout_failure / heuristic_warning / none` 分级记录；
5. 当前 transition provider 是否明确标注为诊断用精确 single-push，而不是 learned SD transition；
6. state aliasing 桶内后继变化方差是否被记录。

该 smoke 的通过只说明设计 schema、标签构造和审计字段可以落盘，不说明已经实现可学习 SD 世界模型。

### Learned flow smoke 口径

在设计验证 smoke 通过后，使用以下入口训练最小 learned flow 模型：

1. `experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py`

该脚本学习 `x_t, a_t -> f_t`，其中 `f_t` 被 `softplus` 约束为非负，并通过固定 incidence matrix `B` 得到 `x_{t+1}`。当前版本不启用反馈、不训练不确定性头、不接入 SD-MCTS，只用于验证六核心存量和诊断流量标签是否存在可学习信号。

首轮比较对象为 zero-flow baseline，即假设 `x_{t+1}=x_t`。若 learned flow 在验证集上不能优于该基线，则说明当前 stock proxy 或 action feature 尚不足以支撑后续 learned SD transition。

随机 transition 切分只作为最初 smoke。后续推进以 level-holdout split 为优先证据，因为它能避免同一关卡中的相近状态同时进入训练和验证。若 level-holdout learned flow 仍稳定优于 zero-flow baseline，才进入 feedback-on flow、uncertainty calibration 和 learned-transition-provider SD-MCTS smoke。

反馈验证采用同一 level-holdout split，比较 `base no-feedback` 与 `feedback-gated` flow model。当前 feedback-gated 版本只用目标压力、拥塞-机动性-风险等反馈特征对非负流量做乘性门控，因此它只验证“反馈结构可接入并可能改善一步转移”，不等同于完整 SD 反馈回路。若该门控相对 no-feedback 的提升很小，应先改进反馈特征、损失或标签，再进入 SD-MCTS。

为判断反馈增益小是否只是因为 `x_t` 已经吸收了反馈信息，增加 feature-mode 消融：`stock_action`、`stock_only` 和 `action_only`。若 `action_only + feedback` 明显优于 `action_only`，但 `stock_action + feedback` 只略优于 `stock_action`，可解释为反馈信息主要被 `x_t` 吸收；若 `action_only + feedback` 也不优于 `action_only`，则更可能说明当前一步标签主要由动作局部特征解释，或者当前反馈特征、gate 形式和一步目标不足以暴露反馈价值。

### 多步 rollout consistency smoke 口径

在一步 learned flow smoke 后，下一步验证不直接进入 SD-MCTS，而是先检查 learned stock-flow transition 在短轨迹上反复应用时是否保持一致。该测试仍使用诊断性 exact single-push provider 生成标签，不训练不确定性头，也不报告求解性能。

[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已加入 `--rollout-depth`、`--rollout-loss-weight` 和 `--max-rollout-samples-per-level`。当 `rollout_depth>1` 时，脚本从真实 single-push transition 构造短轨迹，并按以下递推评估模型：

```text
\hat{x}_{t+k+1}=clip_X(\hat{x}_{t+k}+B f_\theta(\hat{x}_{t+k}, a_{t+k}))
```

其中后续步动作特征来自真实短轨迹动作，存量输入来自模型上一预测步。若 `rollout_loss_weight>0`，训练目标在一步 stock/flow 损失之外增加多步 stock 预测损失；若 `rollout_loss_weight=0`，只训练一步模型，但仍在相同短轨迹上报告 rollout 指标。

本轮首版 rollout smoke 固定 `rollout_depth=2`，并在 level-holdout split 下比较四组：`base + rollout supervision`、`feedback + rollout supervision`、`base one-step train + rollout eval` 和 `feedback one-step train + rollout eval`。主要指标为 step-1/step-2 stock MAE、两步平均 stock MAE、zero-flow 两步平均基线、flow MAE by step 和 step-2 per-stock MAE。

该 smoke 的方法学作用是检查抽象存量转移在连续预测下是否稳定，并判断反馈结构是否在累计误差中比一步误差更可见。当前它仍不构成 learned-transition-provider SD-MCTS，因为 MCTS 节点尚未调用 learned transition 展开，也未加入不确定性校准或风险校准。

### 关键存量加权与 feedback target smoke 口径

多步 rollout smoke 显示 step-2 误差主要集中在 `reachable_capacity` 与 `congestion` 后，进一步加入两个可审计训练入口：

1. `--stock-loss-preset reach_congestion`：训练时提高 `reachable_capacity` 与 `congestion` 的 stock loss 权重，并将权重均值归一为 `1`，避免简单改变整体学习率；
2. `--feedback-target-loss-weight`：仅用于 `model-kind=feedback`，增加 next-feedback auxiliary target。目标为真实下一存量代理映射得到的反馈特征，模型通过共享 feedback context 同时生成 flow gate 与 next-feedback prediction。

该实验只改变 learned flow smoke 的训练目标，不改变评估指标。所有指标仍按未加权 stock MAE、flow MAE、rollout MAE 和 per-stock MAE 报告，以避免把训练权重直接写入结论指标。

首轮结果显示，简单 `reachable_capacity/congestion` 加权并未降低目标存量的验证误差：`reachable_capacity` 基本不变，`congestion` 反而上升。uniform stock loss 下的 feedback target 有小幅改善，但仍未超过 no-feedback base。因此，当前方法记录为负结果或弱信号，不作为进入 SD-MCTS 的充分条件。

### 冻结最小基线与审计复测口径

根据代码、结果和 `academic-research-suite` 审查，当前 Sokoban/Boxoban 线冻结以下诊断性最小 learned stock-flow baseline：

```text
six stocks + fixed incidence matrix + nonnegative learned flows
+ legal single-push action features
+ level-holdout split
+ uniform stock loss
+ two-step rollout supervision
+ explicit diagnostic provider audit
```

该基线对应 `baseline_id=sokoban_six_stock_nonnegative_flow_rollout_sup_base_v1`。实现中仍使用诊断性 `exact_single_push_diagnostic_provider` 生成监督标签，`sd_mcts_enabled=false`，因此它是进入 formal learned-transition-provider SD-MCTS 前的世界模型 smoke，而不是 SD-MCTS 规划结果。

[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 现已在结果 JSON 中补充 `baseline_contract` 与 `audit` 字段。审计字段包括 provider audit、逐存量 MAE、逐流量 MAE、clip correction、stock-only alias bucket 方差和 stock-action alias bucket 方差。审计字段只用于解释模型误差来源，不改变训练目标。

复测结果显示，冻结基线三种子主指标与旧 `rollout_sup_base` 精确一致，复现判定为 `REPRODUCIBLE`。新增审计同时显示，验证集上 `clip` 修正较频繁，且 stock-only alias bucket 仍存在较高 next-delta 方差；因此下一阶段应先审查 stock proxy、flow decomposition 和 aliasing，而不是把 feedback gate、feedback target 或 loss reweighting 纳入默认模型。

### Stock-flow 诊断审计口径

在冻结最小基线复测通过后，新增 stock-flow 诊断审计入口：

1. `experiments/sokoban_sd_world_model/audit_sokoban_stock_flow_diagnostics.py`

该脚本不训练模型，也不接入 SD-MCTS。它复用当前六存量代理、十二流量定义、固定 incidence matrix 和冻结基线相同的 level-holdout 切分规则，重建 exact single-push diagnostic transitions，并将其与冻结基线预测侧审计合并到同一 JSON。审计目标是区分以下误差来源：

1. stock proxy 是否过粗或归一化方式不合适；
2. diagnostic flow decomposition 是否满足 `B f_t = x_{t+1}-x_t`；
3. learned softplus flows 是否出现相反流量共激活与抵消；
4. `clip_X` 是否在预测中承担过多边界修正；
5. stock-only 与 stock-action alias bucket 是否仍存在高后继方差。

对应结果文件为：

1. `results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed0.json`
2. `results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed1.json`
3. `results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed2.json`
4. `results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_summary.json`
5. `results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json`

增强审计版冻结基线与旧 `rollout_sup_base` 主指标保持精确一致，说明新增字段没有改变训练路径。完整诊断覆盖 `30` 个关卡、`566` 个抽象状态和 `3435` 条 single-push transition。exact diagnostic flow labels 的 incidence balance `max_l1=0.0`，说明当前诊断标签在固定矩阵下可守恒重构。

本轮诊断的主要发现如下：

1. 冻结基线的 valid any-stock clipped fraction 均值为 `0.9277460177739462`，valid mean abs clip correction 均值为 `0.017344274868567783`。裁剪主要集中在 `irreversibility_risk`、`goal_completion`、`mobility` 和 `unresolved_box_pressure`，而不是 `reachable_capacity` 与 `congestion` 本身；
2. stock-only alias max next-delta variance 最高为 `0.39549830236885636`，stock-action alias max next-delta variance 最高为 `0.11775519031141866`。加入当前 action features 后别名降低，但未消失；
3. `reachable_capacity` 的 exact delta nonzero fraction 为 `0.37147016011644834`，`congestion` 为 `0.6954876273653566`。二者确实频繁变化，但分别由 `reachable_count/floor_count` 和 `1-legal_push_count/(4 n_boxes)` 这类全局粗归一化代理构成；
4. 由于 `softplus` 使每个流量严格为正，冻结基线预测中六组相反流量的 `pred_coactive_fraction_gt_1e-6` 均为 `1.0`，而 diagnostic target 的相反流共激活应接近零。这说明 learned flow 可能依赖相反流量互相抵消，再由 `clip_X` 修正边界；
5. 因此，当前误差不宜继续用 loss reweighting 或 feedback gate 堆叠处理。下一阶段应采用单因素受控修订：先测试 paired/net-flow 参数化或稀疏流量约束，再测试 `reachable_capacity` 与 `congestion` 的代理修订，最后再考虑 alias-aware 辅助特征。

该诊断结论不否定冻结基线的价值。冻结基线仍作为后续所有修订的对照；但在 clipping、相反流共激活和 stock-action aliasing 未改善前，不应进入 learned-transition-provider SD-MCTS。

### 阶段 1：paired/net-flow 修订口径

阶段 1 的目标是处理 learned flow 的结构病灶，而不是修订存量代理或接入 SD-MCTS。新增 `--flow-parameterization softplus|paired_net`，默认仍为 `softplus`，以保持冻结基线路径可复现。`paired_net` 版本只用于阶段 1 单因素实验。

`paired_net` 保留十二个非负流量和固定 incidence matrix，但神经网络只输出六个成对有符号净流量。每对相反流量按以下方式展开：

```text
f_left = relu(z_pair)
f_right = relu(-z_pair)
```

因此每对流量至多一个方向为正，可以从结构上消除 `softplus` 独立非负流导致的相反流量共激活。该修订不改变 stock proxy、不改变 action features、不启用 feedback gate、不启用 SD-MCTS。

阶段 1 通过门限固定为：

1. valid rollout stock MAE mean `<=0.05119`；
2. valid rollout step-2 stock MAE `<=0.05918`；
3. valid mean abs clip correction `<=0.01214`；
4. valid any-stock clipped fraction `<=0.75`；
5. flow pair predicted coactivation mean `<=0.20`；
6. every flow pair predicted coactivation `<=0.30`；
7. incidence balance max L1 `<=1e-9`。

第一轮 `paired_net` 结果写入 [flow_stage1_paired_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_summary.json) 和 [flow_stage1_paired_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_gate.json)。结果为 `PARTIAL`：相反流共激活与 incidence balance 通过，但 rollout 非退化和 clipping 门限失败。

主要数值如下：

1. valid flow MAE 从冻结基线 `0.04738566527764002` 降至 `0.03719972570737203`，说明 paired/net-flow 更贴近 diagnostic flow target；
2. flow pair predicted coactivation mean 和 max 均为 `0.0`，通过门限；
3. valid rollout stock MAE mean 从 `0.048756017039219536` 升至 `0.05166300448278586`，超过 `0.05119` 门限；
4. valid rollout step-2 stock MAE 从 `0.056363618622223534` 升至 `0.05988542859752973`，超过 `0.05918` 门限；
5. valid mean abs clip correction 从 `0.017344274868567783` 升至 `0.028677965203921`，未达到下降到 `0.01214` 的要求；
6. valid any-stock clipped fraction 从 `0.9277460177739462` 升至 `0.9618482192357382`，未达到 `0.75` 的要求。

因此，`paired_net` 不能作为阶段 1 达标模型，但它验证了相反流共激活可以通过结构参数化消除。下一轮阶段 1 应在 paired exclusivity 基础上处理流量幅度和边界问题，例如加入幅度约束、raw-next boundary penalty 或按 stock 上下界缩放净流量，而不是回到独立 `softplus` 流量。

第二轮阶段 1 修订在 `paired_net` 基础上加入默认关闭的容量一致性与 rollout 校准入口。`--capacity-projection-stock-mask` 对指定存量执行后网络投影，只缩放会使被保护存量越出 `[0,1]` 的入流或出流组；默认 mask 为全零，不改变冻结基线。本轮达标候选使用 mask `[1,1,1,0,0,1]`，即保护 `goal_completion`、`unresolved_box_pressure`、`mobility` 与 `irreversibility_risk`，不直接约束 `reachable_capacity` 与 `congestion`，以保留二者对 rollout 误差的表达自由度。`--rollout-final-step-weight` 为 rollout 最后一步训练损失的标量权重，默认 `1.0`；达标候选使用 `1.2`，仅用于修正两步 rollout 的轻微超限。另新增 `--rollout-final-stock-weights`，默认全 `1.0`，本轮测试表明逐存量 final-step 加权未改善 gate，不进入达标配置。

阶段 1 达标配置为 `paired_net + capacity_projection_stock_mask=1,1,1,0,0,1 + rollout_final_step_weight=1.2`，其余冻结基线相关训练口径保持不变：`model-kind=base`、`feature-mode=stock_action`、`stock-loss-preset=uniform`、`rollout-depth=2`、`rollout-loss-weight=1.0`、`split-mode=level`、`max-levels=30`、`max-depth=2`、每关 `30` 个状态、`epochs=200`、`batch-size=128`、seed `0/1/2`。结果写入 [flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json) 和 [flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json)。

该 gate 判定为 `PASS`。三种子 valid rollout stock MAE mean 为 `0.05017088601986567`，低于门限 `0.05119`；valid rollout step-2 stock MAE 为 `0.058157226691643395`，低于门限 `0.05918`；valid mean abs clip correction 为 `9.097307914392635e-07`，低于门限 `0.01214`；valid any-stock clipped fraction 为 `0.0005817335719863573`，低于门限 `0.75`；flow pair predicted coactivation mean 和 max 均为 `0.0`；incidence balance max L1 为 `0.0`。因此，阶段 1 的 learned flow 结构病灶已在当前门限下完成修订。该结果仍不等同于 learned-transition-provider SD-MCTS；下一阶段应先实现 learned transition provider audit，再进入 SD-MCTS 接口验证。

### 阶段 2：learned provider 与 SD-MCTS 接口审计口径

阶段 2 的目标不是报告 Boxoban 求解率，而是验证阶段 1 通过的 learned stock-flow 模型是否能以 transition provider 形式被 SD-MCTS 消费。为此，[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 新增默认关闭的 `--checkpoint-out`，用于保存可加载的 learned flow checkpoint。默认不写 checkpoint，因此不改变既有阶段 1 JSON 输出。

新增 [audit_stage2_learned_provider_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_stage2_learned_provider_sdmcts.py)。该脚本加载 checkpoint，将神经流量模型封装为 `learned_stock_flow_checkpoint_v1`，并在每个 MCTS 展开节点执行：

```text
\hat{f}_t = f_\theta(\hat{x}_t, a_t)
\hat{x}_{t+1} = clip_X(\hat{x}_t + B \hat{f}_t)
```

SD-MCTS 的节点同时保留 raw Sokoban state handle，但该 handle 只用于枚举合法 single-push action 并推进下一层可执行动作集合，记录为 `exact_push_raw_state_support_for_action_enumeration`。抽象 stock transition、节点 reward 和 value 只读取 learned provider 输出；exact stock proxy 仅作为 `exact_stock_proxy_target_for_audit_only` 参与误差审计，不参与 planner 评分。

阶段 2 smoke gate 的通过条件包括：所有 MCTS 展开 transition 的 provider 必须为 `learned_stock_flow_checkpoint_v1`；每个 root 至少展开一个 child；预测流量非负、预测 stock 位于 `[0,1]`、`x+Bf` incidence 重构误差小于 `1e-6`；成对相反流共激活不超过阶段 1 门限；clip correction 不超过阶段 1 门限；depth-1 与 depth-2 stock MAE 分别不超过 `0.06` 和 `0.07`。这些门限用于验证接口可靠性和短视界一致性，不替代后续正式规划性能评测。

阶段 2 seed0 checkpoint 与 SD-MCTS smoke 结果写入 [stage2_provider_checkpoint_seed0_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0_summary.json)、[stage2_provider_checkpoint_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt) 和 [stage2_learned_provider_sdmcts_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_learned_provider_sdmcts_audit_seed0.json)。审计使用 checkpoint 记录的 held-out levels，共 `6` 个关卡、`42` 个 root、`793` 条 MCTS 展开 transition。gate 判定为 `PASS`，`failed_checks=[]`；learned provider fraction 为 `1.0`，depth-1 stock MAE 为 `0.05060150917204054`，depth-2 stock MAE 为 `0.056905325794544735`，clip mean abs correction 为 `2.5552278208401885e-05`，clip any-stock fraction 为 `0.0037831021437578815`，flow pair coactivation mean/max 均为 `0.0`，incidence balance max L1 为 `1.1748737760441941e-07`。

该结果说明阶段 1 模型已能作为 learned transition provider 接入最小 SD-MCTS 展开路径。它仍不是完整 Boxoban benchmark：当前 MCTS smoke 使用较小搜索深度和抽象 stock reward/value，只验证 provider 接口、守恒、边界、短视界误差和 provider 日志。下一阶段才评估规划预算、成功率、失败类型、不确定性校准和与 heuristic diagnostic path 的消融对比。

### 阶段 3：closed-loop SD-MCTS 规划 smoke 口径

阶段 3 将阶段 2 的接口审计扩展为闭环规划评测。新增 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，采用 receding-horizon 形式：每一步从当前 raw Sokoban state 重新编码六核心 stock，MCTS 内部节点由指定 abstract transition provider 展开，选择根动作后在真实 Sokoban push dynamics 中执行，再进入下一步规划。

该脚本包含四类对照：

1. `learned_sdmcts`：abstract transition provider 为阶段 2 checkpoint 的 `learned_stock_flow_checkpoint_v1`；
2. `oracle_abstract_sdmcts`：abstract transition 直接使用 exact push 后的 stock proxy，作为抽象 reward/value 与 MCTS 控制逻辑的上界诊断；
3. `zero_flow_sdmcts`：`x_{t+1}=x_t`，作为 no-learned-transition ablation；
4. `heuristic_diagnostic` 和 `heuristic_best_first_diagnostic`：启发式诊断参照，明确标记为非 SD-MCTS。

首次 held-out planning smoke 使用 checkpoint 记录的 `6` 个验证关卡，预算 `16/48` simulations，seed `0/1`，MCTS 深度 `3`，最多 `30` 次 push。结果写入 [stage3_sdmcts_planning_smoke_seed01_budget16_48.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_planning_smoke_seed01_budget16_48.json)。provider 使用审计通过：`learned_provider_fraction=1.0`，`failed_checks=[]`。但是 `learned_sdmcts`、`oracle_abstract_sdmcts`、`zero_flow_sdmcts` 和 greedy `heuristic_diagnostic` 的成功率均为 `0.0`，失败类型集中在 `no_legal_actions` 与 `state_revisit_limit`。

随后补跑较大预算 smoke：预算 `200` simulations，深度 `5`，最多 `60` 次 push，seed `0`，结果写入 [stage3_sdmcts_planning_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_planning_budget200_depth5_seed0.json)。`learned_sdmcts` provider gate 仍为 `PASS`，`learned_provider_fraction=1.0`，但 `learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 的成功率仍为 `0.0`。其中 oracle abstract 也失败，说明当前主要瓶颈不是 learned transition 误差本身，而是六 stock 抽象 reward/value、终端/死锁信号和短视界 MCTS 控制不足以支撑 Sokoban 长程求解。

作为关卡可解性参照，`heuristic_best_first_diagnostic` 在相同 `6` 个 held-out levels、`5000` expansion 预算下成功率为 `1.0`，平均解长 `15.333333333333334` push，结果写入 [stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json)。该结果只说明关卡本身可由经典诊断搜索解出，不构成 SD-MCTS 主线结果。

阶段 3 的方法学结论是：learned transition provider 已被正确消费，但当前规划层未通过成功率检验。下一轮不应继续只增加 MCTS simulation 数，而应优先修订规划目标和抽象状态消费方式，包括显式终端价值、可恢复/不可恢复风险校准、避免循环的 state/revisit penalty、长程 value bootstrap、以及将 goal-room、macro 或 relevance 机制作为 SD feedback/action abstraction，而不是作为静态启发式替代世界模型。

### 后续验收口径

正式结果至少需要报告三类指标：

1. 世界模型指标：抽象状态预测误差、`B f_t` 流量守恒误差、风险校准、不确定性校准和 state aliasing 诊断；
2. 规划指标：SD-MCTS 成功率、预算敏感性、thinking time 效应、失败类型和样本效率；
3. 方法论指标：`no-learned-parameter`、`no-feedback`、`black-box latent MBRL`、`heuristic-search diagnostic`、`learned-transition-provider audit` 等消融。

### Stage-3 TVA v1：终端价值对齐接入

依据 NeurIPS 材料中的 Terminal Value Alignment 设计，本阶段将价值网络限定为 MCTS 截断叶节点的剩余价值估计器，而不改变 learned stock-flow transition provider。Beer Game 简化实现中的共同机制为：MCTS 仍由结构化世界模型生成内部想象状态，叶节点调用 `V_psi(s_leaf)`；价值网络用 Monte Carlo return 或 TD/search target 训练，并在推理时以反归一化值参与回传。本阶段采用其中最小、可审计的 Monte Carlo teacher 版本。

新增 [sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/sokoban_tva.py) 与 [train_sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_tva.py)。TVA 输入由六个 SD stock、九个反馈派生量和十五个高层结构标量组成，共 `30` 维。前六维来自 SD-MCTS 节点持有的 stock；反馈量包括目标压力、机动性缺口、可达性缺口、拥塞-机动性耦合、风险-拥塞耦合等；结构标量包括目标完成比例、归一化 matching lower bound、合法/可行 push 比例、dead-square/frozen box 比例、deadlocked、solved 和 no-legal 标志。该输入扩展只用于 terminal value，不进入 `x_{t+1}=clip_X(x_t+B f_theta(x_t,a_t))` transition provider。

离线 teacher 使用 exact-state best-first diagnostic 从采样状态求剩余解长。目标值定义为：

```text
V_target(s) =
  gamma^d * goal_value - push_cost * (1 - gamma^d) / (1 - gamma),  if solved in d pushes
  deadend_value,                                                   if proven dead/no legal
  unsolved_value,                                                  if teacher budget exhausted
```

本轮参数为 `gamma=0.95`、`goal_value=10.0`、`push_cost=0.05`、`deadend_value=-10.0`、`unsolved_value=-2.5`。训练复用阶段 2 checkpoint 的 level split，只在 `24` 个 train levels 上拟合，在 checkpoint 的 `6` 个 valid levels 上报告泛化误差。采样口径为 `state_depth=3`、每关最多 `90` 个状态、teacher budget `5000` expansions。训练得到 `1741` 个样本，其中 train `1345`、valid `396`。价值网络为两层 ReLU MLP，`hidden_dim=128`，训练 `400` epoch，目标按 train mean/std 归一化。

输出文件为 [sokoban_tva_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_train_seed0.json) 与 [sokoban_tva_value_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt)。训练集 MAE 为 `0.8201102614402771`，验证集 MAE 为 `1.1535100936889648`，验证集相关系数为 `0.9297597211519081`。分类看，`proven_dead_end` 拟合较好，MAE 为 `0.12744398415088654`；`solved` 类平均目标为 `4.7114176750183105`、平均预测为 `2.546757459640503`；`teacher_budget_unsolved` 类目标为 `-2.5`、平均预测为 `1.2138255834579468`。因此 v1 已能学习死锁负值，但对可解状态和预算未解状态仍存在混淆。

[evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py) 新增 `--terminal-value-mode stock|zero|tva` 与 `--tva-checkpoint`。`stock` 保留旧固定 stock readout；`zero` 为 terminal-value 消融；`tva` 在叶节点调用 checkpoint-backed `sokoban_tva_value_v1`，并记录 `terminal_value_usage_gate`。TVA 模式下，tree expansion 仍由 learned/oracle/zero provider 给出，teacher 不在线使用。

复测结果写入 [stage3_sdmcts_tva_smoke_seed01_budget16_48.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_smoke_seed01_budget16_48.json) 与 [stage3_sdmcts_tva_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_budget200_depth5_seed0.json)。小预算 `16/48`、seed `0/1` 下，learned provider gate 为 `PASS`，TVA usage gate 为 `PASS`，TVA 叶值调用 `36944` 次；但 `learned_sdmcts` 成功率仍为 `0.0`，`oracle_abstract_sdmcts` 仅在 budget `16` 的两种子汇总中达到平均成功率 `0.08333333333333333`。较大预算 `200`、深度 `5`、seed `0` 下，TVA usage gate 仍为 `PASS`，TVA 叶值调用 `76600` 次；`learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率均为 `0.0`。

本轮 TVA v1 的结论为有效接入但规划性能未通过。失败仍主要集中在 `state_revisit_limit` 与 `no_legal_actions`，且大预算下多数叶节点仍为 `depth_cutoff`，说明当前问题不只是 provider 接线或是否存在终端价值，而是 TVA target 对 solved 与 budget-unsolved 的区分、循环/重复状态处理、以及六 stock 抽象在深层规划中的可辨识性仍不足。下一轮不应把 v1 解释为通过阶段 3 的 TVA，而应在保留该最小可审计实现的基础上，推进 search-value alignment、on-policy failed-state relabeling、revisit-aware value target 和更稳定的动作选择消融。

### Stage-3 TVA v2：失败闭环状态重标注

TVA v2 以 TVA v1 为对照，不改变 learned stock-flow transition provider，也不把 best-first teacher 在线接入 SD-MCTS。修订目标限定为：将 train levels 上由当前 closed-loop SD-MCTS 产生的失败状态重新标注为负值样本，使 terminal value evaluator 更明确地区分重复访问、无合法动作、planner 无动作返回和达到最大 push 数等失败形态。

[train_sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_tva.py) 新增 `--augment-failed-rollouts` 路径。该路径加载阶段 2 learned provider checkpoint 与已有 TVA checkpoint，在 provider checkpoint 的 train levels 上运行 closed-loop SD-MCTS，收集失败轨迹状态，并将每个失败轨迹状态加入 TVA 训练集。valid levels 不参与失败轨迹增强，仍只用于泛化评估。失败状态 target 定义为：

```text
V_failed(s_t) = gamma^min(T-1-t, H) * closed_loop_failure_value
```

其中本轮使用 `closed_loop_failure_value=-8.0`、`H=10`、`gamma=0.95`。该 target 表示“沿当前 closed-loop 策略即将走向失败”的剩余价值，而不是人工启发式评分。teacher best-first 仍只用于离线状态标签；failed rollout augmentation 使用的是 train-level closed-loop 失败经验。

本轮同时加入样本权重。teacher solved 样本权重为 `1.25`，proven dead/no-legal 样本权重为 `1.0`，teacher budget unsolved 样本权重为 `2.0`，closed-loop failed rollout 样本权重为 `2.0`。训练损失为按 batch 内平均权重归一化后的加权 MSE，避免仅因平均权重变大而等价于提高整体学习率。

[evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py) 新增 `--root-action-selection visits|value`。`visits` 按根节点访问次数选择动作，平局再比较 value；`value` 按根子节点 value 选择动作，平局再比较 visits。该消融只改变根动作选择规则，不改变 MCTS 内部 provider、TVA leaf evaluation 或 teacher 使用边界。TVA checkpoint 的 `value_provider_id` 由 checkpoint contract 读取，使 v1 与 v2 在评测日志中可区分。

TVA v2 训练输出为 [sokoban_tva_v2_failed_relabel_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_train_seed0.json) 与 [sokoban_tva_v2_failed_relabel_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt)。训练集总样本数为 `2341`，其中原 teacher 样本 `1741` 个，closed-loop failed rollout 样本 `600` 个。失败增强样本来源为 `closed_loop_state_revisit_limit=378`、`closed_loop_no_legal_actions=36`、`closed_loop_planner_returned_no_action=96` 和 `closed_loop_max_pushes=90`。

TVA v2 的总体验证集 MAE 为 `1.5998603105545044`，验证集相关系数为 `0.8909407432297295`。分组看，`proven_dead_end` MAE 为 `0.11926791816949844`，目标均值 `-10.0`、预测均值 `-9.899348258972168`；`teacher_budget_unsolved` MAE 为 `1.7181379795074463`，目标均值 `-4.0`、预测均值 `-3.0325918197631836`；但 `solved` 类 MAE 升至 `5.988962650299072`，目标均值 `4.7114176750183105`、预测均值 `-1.2775442600250244`。因此 v2 更强地压低失败和未解状态，但明显牺牲了可解状态的正价值估计。

阶段 3 复测在同一 `6` 个 held-out levels、`learned_sdmcts`、budget `200`、深度 `5`、最多 `60` push、seed `0` 下执行。`root_action_selection=visits` 的结果写入 [stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json)，`root_action_selection=value` 的结果写入 [stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json)。两条复测均通过 learned provider usage gate 和 terminal value usage gate，`learned_provider_fraction=1.0`，TVA provider id 为 `sokoban_tva_value_v2_failed_relabel`，TVA 调用均为 `17600` 次。

两条复测的 `learned_sdmcts` 成功率均为 `0.0`，平均 push 数均为 `14.166666666666666`，失败计数均为 `planner_returned_no_action=3`、`state_revisit_limit=3`。相较于 TVA v1 value-root 消融的平均 push 数 `28.333333333333332`、失败计数 `state_revisit_limit=5`、`max_pushes=1`，v2 改变了失败形态并缩短了失败轨迹，但没有产生求解成功率。该结果记录为有诊断价值的负结果，不作为阶段 3 通过模型。

### Stage-3 TVA 修订：search-value alignment 口径

根据对 NeurIPS 论文 TVA 公式、Beer Game 实验代码和当前 Sokoban v1/v2 结果的复核，后续正式 TVA 不再以 best-first diagnostic teacher 的静态剩余价值作为主训练目标。`sokoban_tva_value_v1` 与 `sokoban_tva_value_v2_failed_relabel` 固定作为 terminal value proxy 对照：前者是 teacher-supervised terminal value proxy，后者是 closed-loop failure relabel value proxy。二者可用于接口和负结果参照，但不再称为严格论文语义的 TVA。

正式下一版命名为 `sokoban_search_value_tva_v1`。其目标来自 SD-MCTS 根节点搜索统计：

```text
V_search(s_t) = sum_a pi_mcts(a | s_t) Q_H(s_t, a)
pi_mcts(a | s_t) = N(s_t, a)^alpha / sum_b N(s_t, b)^alpha
```

训练损失采用论文 TVA 的 search alignment 项，并显式记录 TD 项：

```text
L_val =
  MSE(V_psi(s_t), stop_gradient(V_search(s_t)))
  + lambda_td * MSE(V_psi(s_t), r_t + gamma * Vbar_psi(s_{t+1}))
```

若首轮为降低复杂度设置 `lambda_td=0`，则只能标注为论文 TVA 的 TD 消融，不能标注为完整 TVA。若启用 TD，必须使用 target network `Vbar_psi` 并记录同步策略。

实现前必须增加 `Q_H` 语义审计。现有 MCTS root policy 中的 `q_value` 只有在 depth-1 审计满足 `child.value = reward(root,a) + gamma * V(child_state)` 时，才能作为 `Q_H(s_t,a)` 进入 `V_search`。target collection 需要记录 `root_policy`、`visits`、`q_value`、`search_budget`、`search_depth`、`provider_id`、`terminal_value_id`、`alpha`、`gamma`、finite target fraction 和 root child coverage。

该修订已经按 `academic-research-suite` 的 experiment-agent、methodology reviewer、devil's advocate reviewer 和 editorial synthesis 规范完成设计审查，审查文档为 [sokoban_tva_search_value_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_tva_search_value_design_review.md)。审查结论为：该路线可以作为下一步实现依据，但验收顺序必须先满足 TVA 语义合规，再评估 closed-loop 性能。best-first diagnostic teacher 仍只能用于离线可解性参照或旧 proxy 对照，不得进入在线 SD-MCTS action choice 或 root scoring。

后续进入阶段 3 通过判定前，必须同时报告以下 gate：`target_source=sd_mcts_root_statistics`、`mcts_root_statistics_used=true`、`teacher_used_online=false`、`transition_provider_modified=false`、learned provider fraction、terminal value usage gate、finite `V_search` fraction、root child coverage、solved/dead-end/budget-unsolved 分组价值指标、held-out success rate、失败类型和 oracle abstract 对照。

### SD 目标一致性审计口径

为检验当前 SD 抽象是否服务任务目标，新增 [audit_sokoban_sd_goal_alignment.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py)。该脚本不训练 SD transition model，不运行 closed-loop SD-MCTS，不改变阶段 2 provider，也不把 best-first teacher 在线接入规划。诊断性 teacher 只用于离线标注采样状态和一步动作后果。

审计指标包括：状态级 teacher value 回归、teacher solved AUC、验证集抽象状态 alias bucket 内 teacher value 方差、一步动作后果 top-1 一致率与 teacher regret。比较的特征组包括 `six_stocks`、`six_stocks_plus_feedback`、`six_stocks_plus_existing_structural`、`six_stocks_plus_goal_probes` 和 `all_existing_plus_goal_probes`。

本轮先尝试 full 与 medium 配置，但离线 teacher 搜索耗时较高，均停止且不作为结果解释。随后将静态 articulation bottleneck probe 改为每关只计算一次，并运行 quick 配置：

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

结果写入 [sd_goal_alignment_audit_30_levelsplit_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_quick_seed0.json)。样本规模为状态级 train `231`、valid `60`，动作后果级 train `66`、valid `17`。teacher 标签计数为 `solved=156`、`teacher_budget_unsolved=52`、`proven_dead_end=83`。

quick 审计 gate 判定为 `PASS`。六存量对验证集 teacher value 的 ridge 回归 `R^2=0.8859336361147274`，相关系数 `0.9414044913328575`，teacher solved AUC `0.9247685185185185`。加入 existing feedback/structural 后，验证 `R^2` 分别为 `0.8965768263764594` 和 `0.8987603377865637`。`all_existing_plus_goal_probes` 的验证 `R^2=0.8947046578760309`，相对六存量仅增加 `0.00877102176130351`。六存量验证 alias bucket 的 within/global target variance ratio 为 `0.022069573453746945`。动作后果审计中，固定 stock Q 与 teacher best action 的验证 top-1 agreement 为 `0.8235294117647058`，平均 teacher regret 为 `0.6820606727595`。

该审计支持“当前六存量与任务目标具有显著一致性”，但不能解释为阶段 3 规划已经充分。固定 `stock_value` readout 对 teacher value 的验证 `R^2` 仅为 `0.07533288704528096`，说明局部可排序信号不等于长程价值函数；同时既有阶段 3 中 oracle abstract SD-MCTS 仍为 `0.0` 成功率。因此当前结论是：六存量不是偏离任务目标的任意压缩，但长程规划消费、动作抽象、终端价值和 search-value 泛化仍是后续瓶颈。

### Stage-3 Search-Value TVA v1：严格语义实现与负结果

新增 [train_sokoban_search_value_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py)，用于实现严格 search-value TVA。该脚本不调用 best-first diagnostic teacher 生成训练目标，而是在阶段 2 learned provider 上运行 MCTS root search，按根节点访问次数与根动作回报估计构造：

```text
V_search(s_t) = sum_a pi_mcts(a | s_t) Q_H(s_t, a)
pi_mcts(a | s_t) = N(s_t,a)^alpha / sum_b N(s_t,b)^alpha
```

实现中新增 depth-1 `Q_H` 语义审计：当 `mcts_depth=1` 时，逐 root child 检查 `q_value = reward(root,a) + gamma * V(child_state)`。正式训练的审计覆盖 `80` 个状态，`max_abs_error=1.7763568394002505e-15`，判定为 `PASS`。checkpoint contract 明确记录 `target_source=sd_mcts_root_statistics`、`mcts_root_statistics_used=true`、`teacher_used_online=false`、`teacher_used_for_training_target=false`、`transition_provider_modified=false`、`provider_id=learned_stock_flow_checkpoint_v1`。

首轮 `sokoban_search_value_tva_v1` 使用 `collect_budget=48`、`mcts_depth=3`、`alpha=1.0`、`gamma=0.95`、`lambda_td=0.0`，因此它是论文 TVA 的 search-alignment 主项实现，同时是 TD 项消融。采集得到 `873` 个可训练 root 样本，train `728`、valid `145`；由于 deadlocked/no-legal root 没有 root children，另有 `868` 个状态被跳过，finite `V_search` fraction 为 `0.5014359563469271`。训练结果为 valid MAE `1.3938531875610352`，valid corr `0.5508857019779951`。但 positive search target 组仍未通过正例保护：valid positive target mean 为 `0.8352776765823364`，pred mean 为 `-0.8174604773521423`。

随后进行仅基于 search target 符号的正例保护消融，不引入 teacher solved 标签。`positive-target-sample-weight=4.0` 时，valid positive pred mean 为 `-0.5202222466468811`；`positive-target-sample-weight=10.0` 时，valid positive pred mean 为 `-0.1770382821559906`。权重增强能改善训练集正例和验证正例 MAE，但仍未使验证正值目标均值转正。因此，本轮停止继续调权重，将该问题记录为 search-value target 稀疏、抽象状态别名或价值特征泛化不足，而不是通过更高权重硬推。

阶段 3 复测沿用 held-out `6` 个 valid levels、budget `200`、depth `5`、seed `0`。`sokoban_search_value_tva_v1` 在 learned/oracle/zero-flow 三组上 provider gate 和 TVA usage gate 均通过，TVA 调用总数为 `37800`，但三组成功率均为 `0.0`；learned_sdmcts mean pushes 为 `9.0`，失败计数为 `no_legal_actions=3`、`state_revisit_limit=1`、`planner_returned_no_action=2`。`positive-target-sample-weight=10.0` 版本同样通过 usage gate，TVA 调用总数为 `64800`，但 learned/oracle/zero-flow 成功率仍均为 `0.0`；learned_sdmcts mean pushes 为 `18.166666666666668`，失败计数为 `state_revisit_limit=5`、`no_legal_actions=1`。root action selection by value 的 learned_sdmcts 消融也为 `0.0` 成功率。

因此，本轮结论为：严格论文语义的 search-value TVA 目标采集、`Q_H` 审计和 checkpoint contract 已实现，但阶段 3 仍未通过。失败原因不再是“只用了接口没有按 TVA 语义训练”，而是更具体地转为：当前 search-value target 中正值样本稀少且跨关卡泛化不足，六 stock + 结构特征仍不能稳定识别正向根搜索价值，且 oracle abstract 对照仍未产生成功率。下一步不应继续无门限搜索正例权重，而应优先检查 root state aliasing、`V_search` target 质量、action abstraction/goal-room/macro/relevance 作为 SD feedback 或 option 的受控接入，以及启用论文 TD 项时的 target network 稳定性。

### Stage-3 Expert Abstract Feedback：规划消费层修订

基于“人类专家从直接观测中抽象动力机制”的重新审视，本轮没有扩展六个核心存量，也没有修改阶段 2 的 `learned_stock_flow_checkpoint_v1` transition provider。修订集中在 SD-MCTS 如何消费已有 SD 抽象：新增 `expert_abstract_dynamics_feedback_v1` 作为规划反馈层，新增 `expert_abstract_dynamics_value_v1` 作为非 TVA 的专家抽象 terminal value 对照。该反馈层使用目标完成、匹配压力、可行推箱余量、机动性、可达性、拥塞、不可逆风险、目标入口压力、走廊/关节点压力、死方邻接压力和闭环重复状态计数，形成 bounded reward、action prior 和 leaf value。上述信号是 SD 反馈消费层，不进入 `x_{t+1}=clip(x_t+Bf_theta(x_t,a_t))`，也不替代 learned transition。

[evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py) 新增 `--planner-feedback-mode stock|expert`、`--terminal-value-mode expert`、`--expert-delta-weight`、`--expert-prior-temperature`、`--expert-prior-uniform-mix` 和 `--revisit-feedback-penalty`。结果 JSON 记录 `planner_feedback_usage_gate`、`planner_feedback_counts`、`revisit_feedback_hits`、`planner_feedback_changes_transition_provider=false`、`expert_feedback_teacher_used_online=false` 和 `expert_feedback_best_first_used_online=false`，以审计该修订没有把 best-first teacher 或启发式搜索接入在线规划。

同预算同深度的旧 stock consumption 对照为 `learned_sdmcts`、budget `24`、depth `4`、held-out `6` 关卡，成功率 `0.0`，输出为 [stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json)。expert feedback quick 结果为 [stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`，解出 level `000.txt:6`，解长 `15` push；learned provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`；planner feedback usage gate 为 `PASS`，`planner_feedback_evaluations=6407`。同文件中 `oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 在该配置下成功率仍为 `0.0`。

该结果说明专家抽象反馈能把阶段 3 从完全零成功率推进到 quick 配置下的非零成功率，但尚不能判定为稳定阶段 3 通过。追加复测显示：`learned_sdmcts` 在 budget `48`、depth `4` 下成功率仍为 `0.0`，输出为 [stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json)；budget `200`、depth `5` 下成功率为 `0.0`，输出为 [stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json)。降低 PUCT 到 `0.25` 或将 expert prior uniform mix 提高到 `0.5` 未恢复大预算成功率。因此，本轮结论是“规划消费层方向性改善，但预算稳定性未通过”；后续门限应要求同一 held-out split 上 budget `24/48/200` 至少保持非零成功率，且 learned provider fraction、feedback usage gate 和 teacher-online 禁用 contract 同时通过。

后续复核发现，`expert_abstract_dynamics_value_v1` 的 `should_stop` 原先会在 `model.is_deadlocked(raw_state)` 时停止树展开。这对 terminal leaf 合理，但在 closed-loop 中可能遇到“deadlock proxy 触发但仍存在合法 push”的 root，导致 `planner_returned_no_action`。因此修订为只在 solved 或 no-legal root 停止；deadlock 仍作为强负值进入 value、reward 和 leaf type，不再阻断 root 展开。修订后 [stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget48_depth4_seed0.json) 中 `planner_returned_no_action` 消失，但成功率仍为 `0.0`，失败转为 `no_legal_actions=5`、`max_pushes=1`。

随后加入单因素 `--expert-viability-penalty`，对后继 `viable_push_count=0` 或 `1` 的动作在 expert reward 和 action prior 中施加惩罚。该修订意在表达人类专家会保留未来可恢复推箱余量的抽象动力机制。budget `24`、depth `4` 仍保持 `1/6` 成功率，输出为 [stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json)；budget `48`、depth `4` 仍为 `0/6`，输出为 [stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json)。`root_action_selection=visits` 消融将 budget `48` 的 mean pushes 提高到 `32.0`，失败为 `max_pushes=3`、`no_legal_actions=3`，但成功率仍为 `0.0`，输出为 [stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_visits_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_visits_seed0.json)。

因此，当前瓶颈已从“是否能展开”和“是否能避免明显循环”转向“是否具有稳定的目标导向长程 option 或价值”。继续加大 deadlock、viability 或 revisit 惩罚会使轨迹更保守，但不会自动产生解。下一步应优先把目标区域进入、箱子-目标分配路线和有限宏动作设计为受控 option/action abstraction 或 learned option prior，并继续保持 learned stock-flow provider 的 transition contract。

### Stage-3 Target2 Option Prior：受控高层动作抽象负结果

基于上述结论，本轮新增受控 `target2` 系列 action abstraction。实现位于 [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)，新增参数包括 `--action-abstraction-mode push|target2|target2_prior`、`--max-option-length`、`--max-options-per-node`、`--option-prior-max-multiplier` 和 `--option-execution-mode first_push|full_option`。`target2` 只枚举同一箱子的局部两步推进，不调用 best-first，不进行在线路径搜索；若作为 option 边展开，option 内每个 primitive push 均逐步调用当前 transition provider，并记录 provider counts。

初始 `target2` 作为额外 action 子节点会扩大低预算 root 分支，2-level smoke 中虽通过 `learned_provider_fraction=1.0` 和 action abstraction gate，但成功率为 `0.0`。进一步诊断显示，它会压制原本可解 level `000.txt:6` 的 root 价值排序。因此修订为 `target2_prior`：不新增 root 子节点，而是把同一箱子的两步局部目标推进潜力折入对应 primitive push 的先验权重。该模式记录 `target2_prior_scored_roots`，但 `option_expansions=0`，因为它是结构先验而非额外 transition 边。

`target2_prior` 在 2-level smoke 中恢复 `1/2` 成功率，并通过 provider、feedback 和 action abstraction gate。held-out `6` 关卡 budget `24` 结果写入 [stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`，mean pushes `20.166666666666668`，`learned_provider_fraction=1.0`，`target2_prior_scored_roots=28`。budget `48` 结果写入 [stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json)：成功率仍为 `0.0`，mean pushes `26.333333333333332`，失败类型为 `max_pushes=2`、`no_legal_actions=4`。`root_action_selection=visits` 输出 [stage3_sdmcts_target2_prior_visits_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_visits_learned6_budget48_depth4_seed0.json)，成功率仍为 `0.0`。

三 provider 对照写入 [stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json)。同一配置下 `learned_sdmcts` 成功率为 `0.16666666666666666`，`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率均为 `0.0`；provider usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`。因此 `target2_prior` 是合规的受控高层动作先验，但尚未通过阶段 3 稳定性门限。下一步不应继续仅调 option prior 上限，而应审查 budget 增大后 root value 选择为何从成功轨迹偏移，并将 route consistency、goal-room entry 或 learned option prior 作为单因素机制接入。

### Stage-3 Route/Goal-Room Packing Feedback：诊断性负结果

随后对 budget `24` 与 `48` 的 root 选择偏移进行逐步诊断。level `000.txt:6` 在 budget `24` 可解、budget `48` 失败；两者第一步相同，第二步开始分歧。成功轨迹在第一步后继续整理已位于目标区的箱子，将其从中间目标推向端点目标；失败轨迹则被“立即增加一个箱子上目标”的局部收益吸引，过早填入下方目标。这说明当前 expert feedback 强化了目标完成数，但没有显式刻画目标区内部打包顺序。

本轮实现两个默认关闭的诊断反馈。其一为 `--route-consistency-weight`，用上一闭环 primitive push 和上一被推动箱子的物理位置对当前 root prior 做路线延续加权。该机制保持 provider contract，不调用搜索器，也不改变 transition provider。2-level budget `48` 结果 [stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json) 显示 provider 与 action abstraction gate 均通过，但成功率仍为 `0.0`。

其二为 `--goal-room-order-weight`，对“目标区内部箱子向低邻接度端点目标整理”的动作给予正反馈，并在存在可整理目标区箱子时，对新箱子过早进入目标区施加 delay penalty。该机制同样只进入 expert feedback 的 reward/prior，不修改 learned stock-flow provider。2-level budget `48` 中，`goal_room_order_weight=2.0` 输出 [stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json)，成功率恢复为 `0.5`；`goal_room_order_weight=3.0` 输出 [stage3_sdmcts_goalroom_delay_w3_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w3_t2prior_mini_budget48_depth4_seed0.json)，也为 `0.5`。

但扩展到 held-out `6` 关卡后，该机制不能通过进入条件。`goal_room_order_weight=1.5`、`2.0`、`3.0` 在 budget `24` 下均为 `0.0` 成功率，分别输出 [stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json)、[stage3_sdmcts_goalroom_delay_w2_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_learned6_budget24_depth4_seed0.json) 和 [stage3_sdmcts_goalroom_delay_w3_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w3_t2prior_learned6_budget24_depth4_seed0.json)。因此，goal-room packing 是有效的局部诊断信号，但当前手工权重会损害跨关卡 quick 基线，不能作为阶段 3 默认改进。

本轮后续方向应从“继续手调 goal-room 权重”转为两类可审计路线：一是采集成功/失败 root 的 search-value 或 route-consistency target，训练 learned option prior；二是审查当前 root `Q_H` 对低先验必要过渡动作的估计偏差，尤其是目标区整理动作造成短期机动性下降时，价值备份是否过度惩罚。只有当新机制同时满足 budget `24` 不低于当前 `1/6` 且 budget `48` 恢复非零成功率时，才可进入阶段 3 候选。

### Stage-3 Root-Q Bias 与 Learned Action Prior 探针

[audit_sokoban_root_q_bias.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_root_q_bias.py) 用既有 stage-3 root policy 对 budget `24` 成功轨迹和 budget `48` 失败轨迹进行逐步对照，不运行 best-first，不修改 planner。level `000.txt:6` 在 step `1` 的分歧显示：失败动作相对成功动作的 root `Q` 高 `1.0426520933606815`，prior 高 `0.6487923854590042`，访问数比为 `4.8`。该证据说明当前规划消费层会偏好短期 `boxes_on_goals_delta=1` 的动作，而低估 `goal_room_order_bonus=1.0` 但短期降低可行推箱余量的整理动作。

`--audit-proxy-prior-weight` 是该审计的局部 prior probe，只作为默认关闭的规划层诊断项。它在 2-level budget `48` 下仍为 `0/2`，在 held-out `6` 关卡 budget `24` 下保持 `1/6`。因此其方法地位是“解释性 probe”，不是阶段 3 候选。

为尝试从手工 proxy 过渡到可审计的 learned prior，本轮新增 [sokoban_learned_action_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/sokoban_learned_action_prior.py) 与 [train_sokoban_action_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_action_prior.py)。prior 输入为当前 SD stock/TVA 风格抽象特征、candidate primitive push 方向和单步 action consequence；训练数据只来自 stage-2 provider train split 上的 closed-loop SD-MCTS trace。该 prior 在线只作为 action-prior multiplier 消费，结果合同记录 `learned_action_prior_changes_transition_provider=false`、`learned_action_prior_best_first_used_online=false` 和 `learned_action_prior_teacher_used_online=false`。

当前 learned action prior v1 的训练数据正例过稀：`443` 个样本中 train positive 仅 `6` 个，内部 valid 无正样本。虽然训练 AUC 为 `1.0`，但不能证明泛化。2-level budget `48` 中，权重 `0.5` 与 `0.1` 的 learned prior 均为 `0/2`，因此不进入 held-out `6` 关卡扩展测试。后续若继续该方向，进入条件应先满足：train split 或离线 target 中具有足够正例，内部 holdout 同时含正负类且 AUC 高于随机，随后再要求 held-out budget `24` 不低于 `1/6`、budget `48` 恢复非零成功率。

### Goal-Room Candidate Stock Probe

为避免把局部失败机制直接提升为核心 SD stock，本轮新增只读候选 stock 审计 [audit_sokoban_goal_room_stock_probe.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_goal_room_stock_probe.py)。候选特征包括目标区整理缺口、端点目标填充、中心目标占用、目标入口压力、目标邻接箱压力、目标区箱子可移动性、可整理目标箱比例、premature entry、order action availability 和 order/entry conflict。审计只使用离线 teacher value 标签，不训练 transition provider，也不运行 closed-loop SD-MCTS。

quick 审计显示，这些特征具有可推演的单步转移信号，transition nonzero fraction 为 `0.16944106925880922`；但它们没有提升 valid teacher value 拟合。六 stock 的 valid `R^2=0.8859336464419086`，加入 goal-room 特征后 `R^2=0.8850849477314626`，增益 `-0.0008486987104460164`，低于 `0.02` 候选 stock 门限。因此当前方法结论是：goal-room packing 是真实局部失败机制，但不应在现有证据下扩展为第七个核心 stock。后续若继续使用，应作为 option target、局部 route/packing probe 或 search target 分层变量，而不是直接破坏已经通过阶段 1/2 的六存量 provider。

### Goal-Room Conflict Root Prior Probe

在 root-Q bias audit 和 goal-room stock probe 之后，新增默认关闭的 `--goal-room-conflict-prior-weight` 作为更窄的 root prior probe。该机制只在 root action prior 层处理目标区内部整理与新箱子进入目标区之间的冲突：

```text
conflict_score(a) =
  goal_room_order_bonus(a) - goal_entry_delay_penalty(a)
prior(a) <- prior(a) * exp(w * clip(conflict_score(a), -2.5, 2.5))
```

其中 `goal_room_order_bonus` 表示已在目标区的箱子向低邻接度端点目标移动，`goal_entry_delay_penalty` 表示在存在可整理目标区箱子时，另一个箱子过早进入目标区。该 probe 只在 root 使用，不改变 reward，不改变 terminal value，不改变 `learned_stock_flow_checkpoint_v1` transition provider，不调用 best-first，也不作为在线求解器。

2-level budget `48` smoke 中，`w=1.0` 与 `w=2.0` 均通过 provider、planner feedback 和 action abstraction gate，但成功率均为 `0.0`。level `000.txt:6` 中，该 prior 确实改变了 budget `48` 的第二步选择，使失败前缀从 `[1, 7, 8, 4, 4]` 变为 `[1, 12, 15, 4, 4]`，但没有恢复 budget `24` 成功轨迹中的 `[1, 3, 12, 15, 4]`。因此该机制被记录为诊断负结果，不进入 held-out `6` 关卡扩展测试，也不作为阶段 3 默认配置。

该结果的方法学含义是：目标区冲突不是单一“禁止过早入目标”的局部规则可解决的问题；必要动作可能短期降低机动性和可达性，但保留更好的目标区组织。后续应把该问题转化为 learned search-value、option target 或分配承诺 target，而不是继续手工调 prior 权重。

### Train-Split Teacher-Q Action Target Probe

为避免继续从少量 closed-loop solved trace 中学习 action prior，本轮新增 train-split 离线 action-Q target 审计。脚本 [audit_sokoban_train_action_target_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_train_action_target_signal.py) 只使用 stage-2 provider 的 train levels，并在其中划分内部 holdout levels。对每个采样状态的合法 push，先执行一次真实 successor step，再用离线 best-first teacher 标注 successor value，构造：

```text
Q_teacher(s,a) = stock_reward(x_s, x_{s'}) + gamma * V_teacher(s')
```

这里的 teacher 只用于离线训练信号审计，不在线参与 SD-MCTS，不改变 `learned_stock_flow_checkpoint_v1`。action feature 使用既有 `ACTION_PRIOR_FEATURE_NAMES`，包括当前 SD/TVA 抽象特征、候选 push、after-stock、stock delta、goal-room order/delay、target2 structural score 等。审计先用 ridge action-Q probe 检查 train/internal-holdout 排序信号，只有 target signal gate 通过时才允许训练 learned prior。

quick 审计 gate 通过：train best-action labels `36`，内部 holdout best-action labels `14`，valid rank states `14`，ridge valid top-1 agreement `0.42857142857142855`，valid Q corr `0.7964238411550628`。这表明动作级 target 足够用于小规模 learned-prior 试验。

随后新增 [train_sokoban_teacher_q_action_prior.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_teacher_q_action_prior.py)，训练 `sokoban_teacher_q_action_prior_v1`。训练目标为标准化后的 `Q_teacher(s,a)`，模型仍沿用 `SokobanActionPriorNet`，在线消费方式仍为：

```text
prior(a) <- prior(a) * exp(weight * predicted_teacher_q_score(a))
```

初始 final-epoch checkpoint 在内部 holdout 上排序退化，因此脚本被修订为按内部 holdout rank top-1、regret 和 Q corr 选择 best epoch。best-epoch 版本的 valid top-1 agreement 为 `0.6428571428571429`，valid Q corr 为 `0.8467250504899865`，说明训练本身有效。

但阶段 3 2-level budget `48` smoke 未通过。`learned_action_prior_weight=0.1` 与 `0.5` 均为 `0/2`，虽然 provider usage、planner feedback、action abstraction 和 learned action prior usage gate 均通过。该结果说明：离线动作级 teacher-Q 能被学习，但作为 root prior multiplier 仍不足以修复当前 SD-MCTS 的长程备份偏差。该 prior 不进入默认配置；后续应考虑将 teacher-Q/search target 用于 TVA、option rollout target 或更明确的 high-level option，而不是继续提高 prior 权重。

### Stage-3 Planner-Aligned TVA v2：语义修订与负结果

根据 NeurIPS 论文第 4.4 节的 TVA 公式，价值网络训练目标应与实际 MCTS 搜索统计对齐：

```text
V_search(s_t) = sum_a pi_mcts(a|s_t) Q_H(s_t,a)
pi_mcts(a|s_t) = N(s_t,a)^alpha / sum_b N(s_t,b)^alpha
L_val = MSE(V_psi(s_t), stop_gradient(V_search(s_t)))
        + lambda_td * MSE(V_psi(s_t), r_t + discount * Vbar_or_terminal(s_{t+1}))
```

此前 `sokoban_search_value_tva_v1` 已经实现了 `V_search` 对齐，但 target collection 使用简化的 stock-reward primitive MCTS，与阶段 3 实际使用的 `expert feedback + target2_prior` 消费层不一致；同时 TD 项未启用，且终止后继置零不符合当前 Sokoban reward/terminal value 分工。本轮因此新增 `target_planner_mode=stage3`，使 TVA target 直接来自阶段 3 同配置 SD-MCTS：learned provider 保持 `learned_stock_flow_checkpoint_v1`，planner feedback 使用 `expert_abstract_dynamics_feedback_v1`，action abstraction 使用 `target2_prior`，bootstrap terminal value 使用 `expert_abstract_dynamics_value_v1`。

实现上，`SokobanTvaTerminalValueEvaluator` 和 search-target 用的 terminal-aware evaluator 不再因 `deadlocked` proxy 直接停止展开；`deadlocked` 仍作为强负值进入 leaf value。TD target 改为使用样本记录的 `td_discount`，并在 solved、deadlocked 或 no-legal 后继上调用同一 terminal evaluator，记录为 `td_terminal_successor_value_source=same_terminal_value_evaluator_not_zero`。该修订只改变 TVA/terminal value 消费层，不改变 learned stock-flow transition provider。

quick 训练配置为 `max_states_per_level=8`、`collect_budget=24`、`mcts_depth=4`、`lambda_td=0.25`、`target_sync_epochs=25`、`epochs=250`。结果文件为 [sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json)，checkpoint 为 [sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt)。该训练采集 `217` 个样本，valid corr `0.9664165047048058`，valid MAE `1.0694197416305542`；但 positive search target 稀少，valid positive 仅 `1`，且该正例被预测为负值。

2-level budget `48` smoke 显示，该 TVA v2 未通过入场门限。`root_action_selection=value` 输出 [stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json)，成功率 `0.0`；`root_action_selection=visits` 输出 [stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json)，成功率仍为 `0.0`。两个结果均通过 provider、terminal value、planner feedback 与 action abstraction usage gate。

因此，`sokoban_search_value_tva_stage3_td_v2_quick` 作为“更贴近论文 TVA 语义与阶段 3 planner 语义”的候选保留，但冻结为负结果，不进入 held-out `6` 扩展测试。后续若继续 TVA 路线，应优先处理正向 search-value target 稀疏和跨关卡泛化问题，或将目标改为 option-level value/rollout target，而不是继续用 primitive root prior multiplier 或手工 goal-room 权重补救。

上下文恢复后追加了同一方法的复核版实现与扩展采样。复核版仍采用 `target_planner_mode=stage3`，即 `V_search` 来自阶段 3 的 learned provider、`expert_abstract_dynamics_feedback_v1` 和 `target2_prior`；TD target 仍为 `r+discount*Vbar_or_terminal(s_next)`，其中终止后继价值来自同一 terminal evaluator，而不是置零。`SokobanTvaTerminalValueEvaluator` 的 stop 逻辑同步修订为只在 solved 或 no-legal root 停止，deadlock proxy 只作为强负值 leaf readout。

复核 quick 版 `sokoban_stage3_planner_tva_v2_quick` 使用 `max_levels=12`、`max_states_per_level=8`、`collect_budget=24`、`mcts_depth=4`、`lambda_td=0.25` 和 `epochs=160`。训练样本 `81`，valid corr `0.9321286812910057`，valid MAE `1.2824660539627075`，但正值 search target 仅 `1` 个。2-level budget `48` smoke 成功率为 `0.0`，四类 usage gate 均通过。

复核 expanded 版 `sokoban_stage3_planner_tva_v2_expanded` 仅扩大离线 root-state 覆盖，使用 `max_levels=30`、`max_states_per_level=16`、`collect_budget=24`、`mcts_depth=4`、`lambda_td=0.25` 和 `epochs=160`。训练样本 `356`，train `280`、valid `76`，正值 target `15` 个；valid corr `0.9534958319265285`，valid MAE `0.9219829440116882`。但 valid positive 仍只有 `1` 个，target mean `0.14891120791435242`，pred mean `-2.613969326019287`。对应 2-level budget `48` smoke 成功率仍为 `0.0`，provider、terminal value、planner feedback 和 action abstraction usage gate 均为 `PASS`。

因此，当前 TVA v2 的方法状态是：语义合规性已提高，但搜索目标分布正例覆盖不足，且 closed-loop 规划未通过最低入场门限。后续不应继续对该 checkpoint 调 `lambda_td`、正例权重或训练轮数；若继续 TVA 路线，应改变 target 设计，例如构造 option-level search value、分配承诺 value 或更长程 rollout target。

### Teacher-Q State-Value Terminal Leaf Probe

为检验离线 successor teacher-Q 是否能作为状态级终端价值，而不是 primitive root prior multiplier，新增 `sokoban_teacher_q_state_value_v1_quick`。该方法复用 [audit_sokoban_train_action_target_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_train_action_target_signal.py) 的 train-split action-Q collection。对同一抽象状态的合法 push 后继，先构造离线

```text
Q_teacher(s,a) = stock_reward(x_s, x_{s'}) + gamma * V_teacher(s')
```

再用 `max_a Q_teacher(s,a)` 聚合为状态价值 target。这里的 `V_teacher` 只来自 stage-2 provider train split 内部的离线 best-first successor 标注；held-out valid levels 不生成训练标签；teacher 不在线参与 SD-MCTS；transition provider 不被修改。

模型结构沿用 `SokobanTvaValueNet`，输入为 `TVA_FEATURE_NAMES` 的 `30` 维抽象状态向量：六个核心 stock、九个反馈派生量和十五个高层结构量。该 checkpoint 的 contract 明确标注：

```text
target_source = stage2_train_split_offline_successor_teacher_q_state_aggregation
target_formula = max_a Q_teacher(s,a)
planner_consumption = terminal_leaf_value_only
strict_search_statistics_tva = false
```

因此它不是论文语义下严格的 search-statistics TVA，而是对“teacher-Q 是否可作为叶值补偿”的诊断性状态价值 probe。其作用层级是 terminal value consumption，不属于 transition provider、planner feedback 或 action prior。

当前 quick 训练的 action-Q target signal gate 和 state-value gate 均通过，但 valid 分组显示显著校准问题：positive target 组 MAE 为 `0.8567870259284973`、corr 为 `0.9382455304567155`，nonpositive target 组 target mean 为 `-2.052501916885376`、pred mean 为 `4.085488319396973`、MAE 为 `6.137990474700928`。对应 2-level budget `48` closed-loop smoke 为 `0/2`，尽管 learned provider、terminal value、planner feedback 和 target2 prior usage gate 均为 `PASS`。

据此，该路线不进入默认阶段 3 配置。后续若继续使用 teacher-Q 状态价值，必须至少满足两个修订条件：第一，nonpositive/budget-unsolved 状态必须有显式校准 gate，避免 `max_a Q_teacher` 将局部可行动作高估为全局可解状态；第二，低成本 closed-loop 入场仍以 2-level budget `48` 非零成功率为必要条件，不能以内部分组相关性替代规划测试。

### Option-Level Target Signal 与 Target2 消费复核

阶段 3 失败已经从 provider 接线问题转为“目标导向 option 与长程价值备份不足”。因此本轮补充 option-level target 审计，而不是继续调 primitive action prior 或 terminal leaf value。审计脚本 [audit_sokoban_option_target_signal.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_option_target_signal.py) 的候选集合为：

```text
primitive push options
+ target2_local_structural_option
```

其中 target2 option 是同一箱子的两步局部推进序列。标注过程对每个 option 逐步执行 raw state support step，并累积 stock reward，再对 option successor state 调用离线 best-first teacher 得到：

```text
Q_teacher(s,o) = sum_k gamma^k r_stock(s_k,o_k)
                 + gamma^K V_teacher(s_K)
```

teacher 仅用于 stage-2 train split 内部的离线 successor 标注，不在线参与 SD-MCTS，不修改 transition provider。option feature 由 action-prior 特征加上 `duration`、`target2_structural_score`、option 内 stock return、fixed stock return 和 first-push target2 rank 构成，共 `71` 维。

quick 审计结果为 `FAIL`，但失败具有方向性含义。ridge option-Q 的 valid Q corr 为 `0.7794633473898425`，说明回归值存在相关信号；但 valid top-1 agreement 只有 `0.2857142857142857`，低于 `0.40` 门限。teacher-best 来源显示，train 中 `22/33` 个 best label 来自 target2 option，valid 中 `7/14` 个 best label 来自 target2 option，说明两步 option 在离线 teacher-Q 中确实承担部分高层动作角色。与此同时，`fixed_stock_return` 的 valid top-1 agreement 为 `0.5714285714285714`，`exact_option_score` 为 `0.5`，高于 ridge option-Q 与 `target2_structural_score` 的 `0.21428571428571427`。这表明现有 target2 结构分数不足以作为默认 option prior，learned option-Q 也未达到上线门限，但 option 级 SD rollout return 是值得进一步研究的信号。

随后复核现有 target2 option 在 SD-MCTS 中的 closed-loop 消费。`action_abstraction_mode=target2` 会把 target2 option 作为显式 MCTS child；option 内每个 primitive 的抽象 transition 都调用当前 learned provider，contract 记录 `option_best_first_used_online=false`、`option_exact_abstract_transition_used_online=false`、`option_primitive_transition_provider=learned_stock_flow_checkpoint_v1`。在 2-level budget `48` 下，`option_execution_mode=first_push` 和 `full_option` 均为 `0/2`，但 action abstraction usage gate 均为 `PASS`。因此失败不是 option 没有展开或没有执行，而是当前 option 生成与价值备份不足。

据此，现有 `target2_local_structural_option` 冻结为诊断机制，不进入默认阶段 3 配置。下一版 option 方法必须先在离线审计中同时满足：valid top-1 agreement 不低于 `0.40`，valid Q corr 不低于 `0.20`，teacher-best option 覆盖充足，并报告 nonpositive/budget-unsolved 校准；随后仍需通过 2-level budget `48` closed-loop 非零成功率。若只是在现有 target2 结构分数上调 prior multiplier 或 full-option 执行策略，不再视为结构性推进。

### TVA 截断价值准确性审计口径

SD 世界模型与 SD-MCTS 的测试边界进一步区分如下：SD transition provider 可以独立测试，因为其职责是状态推演准确性；SD-MCTS 闭环规划不能在 TVA 或等价终端价值补偿缺席时作为有效阶段 3 调试对象，因为当前 MCTS 展开深度远短于 Sokoban 实际解题步长，无补偿叶值必然产生局部最优和截断偏差。因此，后续阶段 3 的顺序调整为：先独立验证 SD 状态推演，再独立验证 TVA 截断价值准确性，最后才运行带 TVA 的 SD-MCTS closed-loop smoke。

新增 [audit_sokoban_tva_truncation_value.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_tva_truncation_value.py)。该审计只使用离线 best-first diagnostic teacher 构造评测标签，不在线参与 SD-MCTS，不修改 `learned_stock_flow_checkpoint_v1`、planner feedback 或 action prior。对可解状态，截断价值标签为：

```text
V_teacher(s) = gamma^d * goal_value - discounted_push_cost(d)
```

其中 `d` 是从状态 `s` 到解的剩余推箱步数。对 `proven_dead_end` 和 `no_legal_actions` 使用高置信负值；`teacher_budget_unsolved` 只作为低置信样本报告，不纳入 high-confidence gate，避免把 teacher 预算不足误标为不可解。

审计比较三类 leaf value：固定 `stock_value`、`expert_abstract_value` 和 TVA checkpoint 输出。指标包括 high-confidence MAE、RMSE、相关性、符号准确率、positive value bias 和 nonpositive value bias。TVA accuracy gate 要求 high-confidence 样本数不少于 `20`，TVA MAE 不高于 `stock_value` 与 `expert_value` 中较优者，high-confidence corr 不低于 `0.30`，positive underprediction bias 不低于 `-1.0`，nonpositive overprediction bias 不高于 `1.0`。

smoke 结果显示，`sokoban_stage3_planner_tva_v2_quick` 与 `sokoban_stage3_planner_tva_v2_expanded` 均未通过该 gate。二者 high-confidence corr 分别为 `0.951908759929304` 与 `0.9683731373454699`，说明排序相关性存在；但 high-confidence MAE 分别为 `8.333739054839088` 与 `7.423524176047102`，均差于 `expert_value` 的 `5.125799790019716`。二者 positive bias 分别为 `-8.761932504277338` 与 `-7.659244627712222`，显示可解正值状态被系统性低估。

据此，当前 TVA v2 的失败不再仅表述为 closed-loop smoke 失败，而应明确定位为“真实截断剩余价值尺度和符号校准失败”。后续若继续 TVA 路线，必须先改变 target 生成或校准方式，使 TVA 在该离线截断价值审计中优于 stock/expert 基线并消除正值严重低估；否则不进入新的 SD-MCTS closed-loop 阶段测试。

### Option-Value Prior Planner Probe

为检验 option-level SD rollout return 是否能以受控方式被 SD-MCTS 消费，新增默认关闭参数 `--option-value-prior-weight`。该机制的层级是 planner action prior，不属于 SD transition provider、planner feedback 或 TVA。对每个 target2 planner action，先沿 option primitive 序列逐步执行 raw-state support step，并在每步读取六 stock 抽象状态；局部得分为 option 内 `stock_reward` 的折扣和，加上叶端 `stock_value`。若 option 中途到达 solved、deadlocked 或 no-legal 状态，则按既有终止边界调整 leaf value。

在线合约限定如下：

```text
option_value_prior_source = local_sd_stock_rollout_return_no_teacher
option_value_prior_changes_transition_provider = false
option_value_prior_best_first_used_online = false
option_value_prior_teacher_used_online = false
```

该 prior 只对 root/node action prior 做乘性调整：

```text
prior(o) <- prior(o) * exp(clip(weight * option_stock_return(o), -2, 2))
```

其中 `weight=0.1` 是本轮唯一测试权重。first-push 执行模式与 full-option 执行模式均运行 2-level budget `48` smoke；两者均为 `0/2`，但 option-value prior usage gate 均为 `PASS`。因此该机制冻结为诊断负结果，不进入默认阶段 3 配置，也不继续调权重。后续若继续 option 方向，应重新定义目标区分配、箱子-目标路线承诺或带非正状态校准的 option target，而不是继续扩大当前 target2 option-value prior。

### 截断价值审计与严格 TVA 语义边界

TVA 截断价值审计现在同时用于两个目的：第一，验证候选 leaf value 是否能补偿短视界 SD-MCTS 的剩余价值；第二，区分严格 TVA、teacher-distilled leaf value 与 teacher-Q diagnostic probe 的语义边界。严格 TVA 的 target 必须来自实际规划器的 MCTS root statistics，即：

```text
V_search(s) = sum_a pi_mcts(a | s) Q_H(s, a)
```

而 teacher-Q 状态价值 probe 的 target 为：

```text
V_teacherQ(s) = max_a Q_teacher(s, a)
```

后者可以作为离线诊断上界或辅助信号，但不能替代论文语义下的 TVA。当前同口径审计显示，`sokoban_teacher_q_state_value_v1_quick` 在 high-confidence 截断价值上通过 gate，MAE 为 `0.5409354691144549`，corr 为 `0.9944298990860484`，positive bias 为 `0.003251509906797726`；但它的 checkpoint contract 标注 `strict_search_statistics_tva=false`。因此，该结果证明当前 30 维抽象特征可以承载截断叶值信息，但不证明 stage3 planner-aligned TVA 已经成功。

闭环补测进一步显示，即使用该较准的 teacher-Q 状态价值作为 terminal leaf evaluator，并启用显式 `target2` option 与 `full_option` 执行，2-level budget `48` 仍为 `0/2`。因此，阶段 3 的当前缺口应被表述为“严格 TVA target 校准与高层 option/备份消费协同不足”，而不是单纯“叶值接口没有接入”或“抽象状态完全无法表达剩余价值”。

横向比较已有 checkpoint 后，`sokoban_teacher_q_state_value_v1_quick` 通过 TVA accuracy gate：high-confidence MAE `0.5409354691144549`，corr `0.9944298990860484`，sign accuracy `1.0`，positive bias `0.003251509906797726`，nonpositive bias `0.8333237171173096`。该模型的 target 来源为离线 successor teacher-Q 的状态级聚合，不是严格论文语义下的 `V_search=sum pi_mcts Q_H`，因此方法地位应写为“accuracy-gate passed terminal value compensator / teacher-Q state-value probe”，而不是 search-statistics TVA。

按照“先 TVA accuracy，再 SD-MCTS closed-loop”的顺序，使用该 checkpoint 重跑 2-level budget `48` smoke。结果仍为 `0/2`，但 provider usage、terminal value usage、planner feedback usage 和 action abstraction usage gate 全部通过。由此，下一步方法重点应从“训练一个更准确的状态价值网络”转为“检查准确叶值在当前 SD-MCTS 中的消费失败”：包括 root child Q 备份、value-based root selection、target2_prior 对必要过渡动作的先验压制、以及 expert feedback reward 与终端价值尺度是否冲突。

### TVA Root Sorting 与一步备份分解审计

为避免把闭环失败简单归因于“TVA 没有生效”，新增 root sorting 审计。该审计以已有 SD-MCTS closed-loop 结果中的 `trace_sample` 根状态为固定状态集合，在每个根状态上分别运行三种 terminal leaf evaluator：`zero_leaf`、`stock_leaf` 和通过 accuracy gate 的 `teacherq_tva`。除 terminal value 外，transition provider、planner feedback、action abstraction、budget、depth、root action selection 与随机种子保持一致。

审计不改变在线规划过程。离线 best-first teacher 只在每次规划结束后，对 root policy 中每个候选 first-push successor state 标注剩余价值，用于评估 MCTS root Q 排序是否朝 teacher-supported successor 移动。评价不只使用单一 top-1，因为 Sokoban 中多个动作可能具有相同剩余步数或相同 teacher value；因此同时报告：

```text
teacher_best_q_rank
q_best_in_teacher_best_tier
critical_miss
q_teacher_corr
prior_teacher_corr
visit_teacher_corr
```

其中 `teacher_best_tier` 是 teacher value 距最大值不超过 `teacher_tie_tolerance` 的动作集合；`critical_miss` 表示 MCTS 选中动作的 teacher value 比该集合最大值低超过 `critical_gap=0.25`。

为定位 root Q 错配来源，审计还对每个 root child 计算一步备份分解：

```text
one_step_provider_score = expert_reward(s, a, learned_stock_next)
                          + gamma * TVA(s_next, learned_stock_next)

one_step_exact_score = expert_reward(s, a, exact_stock_next)
                       + gamma * TVA(s_next, exact_stock_next)
```

`exact_stock_next` 仅作为诊断读数，用于判断 learned abstract transition 的 stock 预测是否改变 value ordering；它不进入在线 SD-MCTS，也不替换 learned provider。该审计发现，`teacherq_tva` 提高了 root Q 与 teacher value 的平均相关性，但 root Q top-tier 命中没有超过 baseline；同时 learned provider 版本的一步备份 top-1 从未进入 teacher-best tier，而 exact stock readout 的一步备份 top-1 达到 `6/10`。因此后续方法不应只继续训练叶值网络，还需要处理 transition-provider stock prediction、expert reward 尺度和 route-level option commitment 的协同问题。

### Root Backup 尺度扫描与 Route Commitment 复核

为判断 root backup 错配是否可以通过简单调节 reward/leaf 比例解决，增加离线尺度扫描。该扫描不重跑 MCTS，而是读取 root sorting 审计中每个候选动作的 `expert_reward_provider_stock`、`expert_reward_exact_stock`、`tva_provider_leaf_value` 和 `tva_exact_leaf_value`，构造：

```text
scaled_backup = reward_scale * expert_reward
                + gamma * leaf_scale * TVA_leaf_value
```

扫描只用于诊断，不作为在线 planner。若 learned provider stock 下存在明显优于当前 root Q 的尺度组合，才允许考虑新增受控 planner 参数进入 2-level smoke。实际结果显示，learned provider stock 下最佳组合只有 `3/10` teacher-best tier 命中，低于当前完整 root Q 的 `6/10`，因此不新增 planner 调权参数。exact stock readout 下最佳组合达到 `8/10`，说明尺度并非无关，但其改善依赖更准确的 stock transition。

route commitment 复核采用更严格的 oracle abstract transition 对照和显式 `target2` option 执行。该设置用 oracle stock transition 消除 learned provider 抽象转移误差，并允许 MCTS child 表示两步 option，closed loop 执行完整 option。该测试仍未通过 2-level budget `48`，说明当前 `target2_local_structural_option` 的时间跨度和目标语义不足。后续 option 设计不应只扩展两步局部同箱推进，而应显式表示箱子-目标分配、目标区进入顺序和路线承诺，并在上线前通过 teacher-best tier root sorting gate。

### Provider Stock Calibration 与 Route-Goal Option 审计

为检验 learned provider stock prediction 的价值消费误差是否可由简单后处理解释，新增 train-split 仿射校准审计。该审计只用 stage-2 provider train split 单步 transition 拟合逐存量 `exact_stock ~= scale * predicted_stock + bias`，再将该校准用于固定 root sorting 审计状态的一步 backup 读数。该过程不重训 provider，不修改 online transition provider，不重跑 MCTS，也不调用在线 teacher。

结果表明，逐存量仿射校准不足以作为阶段 3 修订机制。校准后 teacher-best tier 命中仅从 raw provider 的 `0/10` 提升到 `1/10`，critical miss 从 `10/10` 降到 `9/10`；而 exact stock readout 仍为 `6/10` 和 `4/10`。因此，当前 provider 问题不能简化为各 stock 的全局 scale/bias 偏移，更可能涉及状态条件化误差、stock 间组合误差或 planner 消费层对 `congestion`、`mobility`、`reachable_capacity` 等机制存量的敏感性。

为检验更强的 route/goal commitment option 是否值得进入在线 SD-MCTS，新增离线候选审计。`route_goal_monotone_option` 绑定一个物理箱子和一个目标，要求 option 内每一步继续推动同一箱子，并使该箱子到绑定目标的静态 reverse-pull 距离严格下降。该生成器不使用 best-first，不使用 teacher，不搜索完整解；teacher 只在候选生成后离线标注 successor state 的 `Q_teacher`。

审计门限设置为：固定 root 数不少于 `5`，route-goal 候选覆盖至少 `50%` root，route-goal 最佳候选至少在 `2` 个 root 上严格优于 non-route 最佳候选，且 route-only structural top-1 的 option tier 命中不低于 `0.30`、first-push tier 命中不低于 `0.40`。quick 与 length-6 两个版本均未通过，失败项均为 `route_goal_beats_nonroute_roots_min_2`。这说明该 option 族在当前固定失败 root 上主要复现了 target2 或 primitive 已覆盖的高价值 first push，而没有提供新的显式 option 价值上界。

由此，本轮方法结论为：不接入仿射 calibrated provider，不接入 route-goal monotone option 作为在线 MCTS child，也不继续手工调 route score。route-goal 审计保留为结构证据：它的 first-push tier 命中较高，可用于后续构造 learned first-push commitment feature、目标分配 target 或 option-target 数据集；但任何上线候选仍必须先通过离线 teacher-best tier gate 和 2-level budget `48` 非零成功率。

### First-Push Commitment Target 审计

在 route-goal monotone option 直接上线失败后，方法重心转为检查其是否能提供第一推承诺信号。第一推承诺 target 的定义为：

```text
Q_commit(s, a_first) = max_o Q_teacher(s, o), first(o) = a_first
```

候选 option 包括 primitive push、`target2_local_structural_option` 和 `route_goal_monotone_option`。每个 option 的 successor state 由 Sokoban raw support step 执行，option 内累计 SD stock reward，再由离线 best-first diagnostic teacher 标注剩余价值。teacher 只在 stage-2 provider train split 及其内部 holdout 中用于离线标注；它不在线参与 SD-MCTS，也不参与候选生成，不修改 `learned_stock_flow_checkpoint_v1`、planner feedback、TVA 或 MCTS action abstraction。

第一推学习 probe 使用 action-prior 特征加 route/goal commitment 附加量，共 `75` 维。新增附加特征包括 route-goal 是否可用、route-goal 数量、最佳 route-goal 结构分数、持续步数、距离增益、终点距离归一化、是否到达目标，以及 target2 是否可用和最佳 target2 结构分数。该 probe 当前使用 ridge regression，只作为离线信号审计，不作为在线 prior。

当前门限分为两层。数据充分性门限要求 train/valid best first-push label 数、valid rank state 数、valid top-tier hit、valid Q corr、teacher-Q 方差和 route-goal 可用率达到最低值。受控扩展审计通过该门限：valid rank states `17`，valid top-tier hit `0.47058823529411764`，valid Q corr `0.8319232992067614`，valid route-goal available fraction `0.38181818181818183`。这说明 first-push commitment target 有可学习信号，且 route-goal option 在 teacher-best first-push labels 中占有实质比例。

但该结果尚未满足 planner-readiness 条件。扩展审计中，ridge commitment score 的 valid top-tier hit 为 `0.47058823529411764`，低于 route-goal structural score 和 route-goal count baseline 的 `0.7647058823529411`；其 mean teacher regret `2.2574965213611655` 也高于 route-goal structural score 的 `1.1701865437235404`。因此，该机制当前不能直接训练在线 learned first-push prior，更不能把结构分数直接作为启发式规则上线。后续若继续，应先改进学习器或特征选择，并增加相对基线门限：学习分数在内部 holdout 上必须达到足够样本覆盖，且 top-tier hit、critical miss 或 regret 至少不劣于主要在线可计算结构基线，才可进入默认关闭的 SD-MCTS prior smoke。

### Sokoban Exact-Grid Stock-Flow SD 主线

Sokoban 新主线采用 `SokobanGridStockFlowModel` 作为 SD provider，provider id 为 `exact_grid_stock_flow_v1`。状态存量定义为 floor 格上的箱子占用 `b_c in {0,1}` 与玩家占用 `p_c in {0,1}`，守恒律为 `sum_c b_c=N`、`sum_c p_c=1`。一次 push 被定义为 floor 邻接图上一条有向边的 one-hot 非负流量 `f`，箱子更新为 `b_next=b+Bf`，其中 `B` 为带符号关联矩阵；玩家更新为 push 后站在被推动箱子的原格。

该 SD 层没有 learned transition 参数。Sokoban 中抽象性不再通过低维状态 rollout 表达，而是分布在三个位置：第一，事件粒度采用 push-level flow，将玩家可达域内的逐格移动折叠；第二，Tier-A/Tier-B 反馈和约束调节流量可行域，例如 wall/box 容量、dead-square、freeze、matching lower bound 和目标吸收；第三，goal completion、mobility、congestion、matching lower bound 等辅助 readout 进入 TVA 特征和诊断，不作为可 rollout 存量。

SD 独立门禁由 `validate_sokoban_grid_stock_flow_design.py` 执行。门禁要求采样状态中箱子质量、玩家质量、flow one-hot、`Bf` box balance、`step_grid_flow` 与 exact `step_state` 一致性、push-level canonical key 等价性均无违例。smoke 结果为 `grid_sd_gate=PASS`，当前记录为 `3` 关、`15` 状态、`64` 合法 push 转移、`308` 个可达玩家位置，全部违例计数为 `0`。

### Exact-Grid TVA 独立审计

TVA 作为新主线的主要学习组件，必须先脱离闭环进行独立验证。当前审计脚本为 `audit_sokoban_grid_tva.py`：在 exact-grid 状态上采样，使用离线 exact-state teacher 标注截断剩余价值，构造真实格点 observation 加辅助 readout 的特征，并拟合 ridge value probe。teacher 只用于离线标签，不进入在线规划；审计输出同时保存 value probe checkpoint，用于后续 MCTS leaf value。

TVA 独立门禁包括 valid 样本数、正值样本数、valid 相关性和 root successor sorting 覆盖。当前通过配置为 `max_levels=8`、`state_depth=1`、`max_states_per_level=8`、`teacher_max_expansions=300`。输出 `grid_tva_independent_audit_smoke_seed0.json`，`tva_independent_gate=PASS`；valid count `16`，MAE `1.3878170000076102`，corr `0.9736159851420181`，sign accuracy `1.0`，root sorting top-tier hit `5/5`。较小 quick 配置因 valid count `7` 未过样本数门限，该结果只作为覆盖不足记录。

### Exact-Grid SD-MCTS 闭环入口

闭环入口改为 `evaluate_sokoban_grid_sdmcts.py`。该脚本在真实 `SokobanState` 上展开 MCTS，动作来自 `SokobanGridStockFlowModel.legal_flows`，转移来自 `step_grid_flow`，默认约束为 Tier-A 死锁剪枝。叶值模式包括 `zero` 和 `tva`；`tva` 模式只能读取独立 TVA 审计导出的 ridge checkpoint，不在线调用 teacher。

该入口支持 `--require-independent-gates`。启用后，脚本会读取 SD 审计文件中的 `grid_sd_gate`，以及 TVA 模式下 TVA 审计文件中的 `tva_independent_gate`；任一门禁未通过则拒绝运行闭环。该约束用于防止尚未独立验证的 SD 或 TVA 直接进入阶段 3 消费层。

当前 3-level smoke 只用于验证闭环接口，不作为性能结论。zero-leaf 在 `horizon=3`、`budget=24`、`max_steps=30` 下成功 `1/3`；TVA-leaf 同口径也成功 `1/3`。随后为避免 TVA 训练关卡泄漏，闭环入口增加 `--level-offset`，并在当前 TVA smoke checkpoint 的 holdout slice（offset `5`、`3` 关）上运行小矩阵。`H={2,4}`、`budget={24,48}` 加补充点 `H=6,budget=96` 的 zero-leaf 与 TVA-leaf 均为 `0/3`。因此该实现目前只证明完整链路可执行，尚未证明 TVA 相对 zero-leaf 有 heldout 解题增益。后续正式实验仍需报告更大 `H x budget x TVA on/off` 矩阵，并将 TVA 增益、约束剪枝、zero-leaf baseline 和旧六存量 learned provider 对照分开解释。

### SD Action Prior（树策略先验）

闭环入口支持 `--action-prior {none,sd_delta}`（默认 `none`）。该机制对应 GFT-SD 设计（`sokoban_gptpro.md` §11/§14.2）中的动作相关性 `I_t`，把已有 SD 读出从单一叶值通道扩展到 MCTS 树策略，针对「搜索形态是主损失源」的瓶颈。设计纪律：叶值 `sd_heuristic` 保持解析纯净（不混入软 SD 概念），软 SD 信号只经先验进入树策略，硬剪枝只保留数学可证死锁。

`sd_delta` 对每个候选动作 `a`（子状态 `s'`）计算 delta 分数 `score(a) = clip(lb(s)−lb(s'), ±4) + w_g·(boxes_on_goals(s')−boxes_on_goals(s)) + w_m·mobility_delta_norm − w_r·1[canonical_key(s')∈visited_keys]`，其中 `lb` 为 matching lower bound（与叶值共享按 boxes 元组的缓存），机动性用合法 push 计数的廉价代理（不做逐 push 死锁检查，默认 `w_m=0`），revisit 复用执行级 `visited_keys`。先验为 `(1−mix)·softmax(score/τ) + mix·uniform`，`τ=--prior-temperature`、`mix=--prior-uniform-mix`。

树侧消费：`initialise_node` 在 prior 激活时按 `(prior, prior_score)` 升序排序 `untried`（尾部 pop 使最高先验最先展开），`none` 时保留 `rng.shuffle`；`select_child` 在 prior 激活时用 PUCT `Q + c_puct·prior·sqrt(N_parent)/(1+N_child)`，`none` 时保留原 UCB1（含未访问边 `+inf` 优先）。prior 路径不调用 `rng`，对 seed 确定性。`none` 臂经 20 关 trace 回归确认与改动前 bit-identical。

50 关 heldout（关卡 48-97，`H=20,b=4096`，rp1+transposition）：`sd_delta` 在 `τ∈{0.5,1.0,2.0}` 一致达 `41/50`（vs baseline `37/50`），耗时约 `0.58×` baseline，失败模式从 max_steps 主导收敛。已知 over-concentration 失败：level_index 9 在全部温度下被先验推入振荡死锁（baseline 解出）。审计列出的后续增强项：候选边以 `N=0` 直接入 `children`、绕过 `untried` 强制展开队列，以在小预算下实现更彻底的 AlphaZero 式预算集中。

### SD-Heuristic 叶值与执行层重访惩罚

闭环入口叶值模式扩展为 `zero`、`sd_heuristic`、`tva` 三种。`sd_heuristic` 不引入任何学习参数，直接将 SD 抽象读出 `matching_lower_bound` 换算为截断价值：`V(s) = gamma^lb * goal_value - push_cost * (1-gamma^lb)/(1-gamma)`，`lb=inf` 时回落 `deadend_value`。该形式与 teacher 价值标签公式同构。其方法论依据为：势函数塑形在截断搜索中逐项相消后等价于叶值启发（根处常数项不影响动作排序），因此目标导向信号统一经叶值通道注入，奖励通道保持每推成本与终局值的原始语义，与 SD 层"面向任务目标的状态推演"定位一致。`sd_heuristic` 同时构成 TVA 必须超越的零学习基线。

执行层新增 `--revisit-penalty`（默认 `0.0`）。执行循环维护 push-level canonical key 已访问集合，规划器在 simulate 中对进入已访问状态的转移在每推成本上加罚。该机制只惩罚回到执行历史状态的转移，不限制树内自身探索，用于打破短视界重规划导致的推上/推下极限环。

### TVA v2：分布覆盖、残差目标与深度分层审计

TVA 审计脚本扩展四项能力。第一，`--level-offset` 使训练关卡切片与闭环 heldout 切片严格不相交（v2 训练为关卡 8-47，heldout 为 5-7），消除 v1 smoke checkpoint 的训练泄漏。第二，`--sample-mode random_walk` 在仅含非死锁后继的随机游走上采样至深度 16，覆盖闭环 MCTS 叶查询的真实深度分布；v1 的 BFS `state_depth=1` 采样保留为 `bfs` 模式。第三，`--target-mode residual` 拟合 teacher 目标减 `sd_heuristic` 基线的残差，闭环消费时由 checkpoint 的 `target_mode` 字段自动叠加基线；`valid_positive_count` 门禁仅适用于 value 模式，因残差目标相对乐观下界基线天然非正。第四，valid 指标按状态深度分层（0-2、3-8、9+）报告，防止近根分布上的高相关掩盖深层失配。root successor sorting 审计保持 BFS 近根采样，因其考察对象即根部动作排序。闭环侧增加 checkpoint `value_scale` 与运行时参数的一致性校验，残差重构在量纲不一致时拒绝运行。

v2 闭环结论为诚实负结果：value 与 residual 模式离线门禁均 PASS（valid corr `0.532/0.617`，深层切片 `0.707/0.771`），但 holdout 三关闭环均 `0/3`，劣于 `sd_heuristic + revisit_penalty` 的 `1/3`，且在 sd_heuristic 可解关卡上引入 matching deadlock。当前判定：TVA 在标签瓶颈（`657/960` 样本为 teacher 预算未解的常数 `-2.5`）解决之前不得进入主结果；离线 gate PASS 不构成闭环有效性证明。

### Two-Layer GFT-SD OO 重建

为独立重建 `sokoban_gptpro.md` 的两层 GFT-SD 设计，本轮新增 `SokobanGFTSDModel`。该模型继承 `SokobanGridStockFlowModel`，因此底账层 `y_t` 仍是已审计的精确格点存量-流量状态：箱子占用 `b`、玩家占用 `p`、push one-hot flow `f` 和关联矩阵 `B`。GFT-SD 资源层 `z_t=Phi(y_t)` 作为只读 readout 层存在，不替代 transition provider，不把旧六存量 learned-flow 重新作为 rollout 状态。

OO 结构如下：

```text
SokobanPushDynamicsModel
  -> SokobanGoalDirectedSDModel
    -> SokobanGridStockFlowModel
      -> SokobanGFTSDModel
```

资源库存采用开闭接口 `GFTResourceStock`，每个库存独立实现 `compute(model, state)`，返回 `GFTResourceValue`。现有库存包括：

| 代码 | 类 | 当前状态 | 作用 |
|---|---|---|---|
| `P` | `TransportProgressStock` | active | 用 matching lower bound 表示剩余运输功，低值更优 |
| `M` | `BoxMobilityStock` | active readout | 记录 legal/viable push 数，默认不进入 prior |
| `A` | `PushAccessibilityStock` | active readout | 记录玩家可达与可执行 push 支撑，默认不进入 prior |
| `K` | `RegionCapacitySlackStock` | active readout / optional prior | 区域容量裕度，基于割点分解的目标区域容量健康度 |
| `D` | `DeadlockDebtStock` | active readout | 记录 dead-square、static freeze、matching infeasibility |
| `Q` | `OrderingDebtStock` | active readout / optional prior | 顺序依赖债务，基于目标区域入口深度的偏序约束 |
| `C` | `GoalCommitmentStock` | active readout / optional prior | 目标承诺清晰度，基于箱子-目标软分配熵 |
| `I` | `ActionRelevanceStock` | active prior | 动作相关性，作为 MCTS soft prior 消费 |

独立闭环入口为 `evaluate_sokoban_gft_sdmcts.py`。该入口复用 exact-grid MCTS 基类，但实例化 `SokobanGFTSDModel`，并通过 `--gft-prior gft_delta` 使用 GFT action relevance。`gft_delta` 当前第一版等价于已验证的 SD action prior 的 GFT 化表达：`progress=clip(lb(s)-lb(s'), ±4)`、`goal=Δboxes_on_goals`、`revisit=1[canonical_key(s') in visited]`，其他资源权重默认 `0`。默认权重为 `progress=1.0`、`goal=1.0`、`mobility=0.0`、`access=0.0`、`capacity=0.0`、`deadlock=0.0`、`ordering=0.0`、`commitment=0.0`、`revisit=1.0`，温度 `2.0`，uniform mix `0.1`。

动作先验的实现口径已统一为“二层资源变化接口”。`GFTSDMCTSPlanner` 不再在规划器层复制 `P` 的 matching lower bound 计算，也不再通过 grid 入口的旧 `sd_delta` 快路径特殊处理运输进度；所有候选动作均由 `SokobanGFTSDModel.gft_action_prior_rows()` 生成 `score`、`prior` 和各资源变化分量。因此，`P/K/Q/C` 后续进入先验时共享同一消费位置。现有 `sd_heuristic` 叶值仍保留为零学习历史基线和对照项，但它不代表最终 TVA-MCTS 架构原则；未来接入 TVA 时，应由 TVA 读取统一资源向量与底账状态特征，而不是只读取 `P`。

方法边界：旧六存量 learned-flow 负结果不再作为否定 GFT-SD 两层设计的依据，只作为工程警示，即低维抽象层不得脱离底账层独立承担闭合转移 provider。GFT-SD 入口已在同一 50 关、同一 `H=20`、`budget=4096`、`sd_heuristic`、`revisit_penalty=1.0`、transposition、`gft_delta` prior、temperature `2.0`、uniform mix `0.1` 设置下复现既有 grid 入口的 `41/50`，并且逐关字段与根节点 trace 完全一致。因此，方法叙述可将 `SokobanGFTSDModel` 作为两层架构主入口。

效率口径：默认 `gft_delta` 第一切片与 grid `sd_delta` 等价，因此在线热路径被保持为行为等价实现；GFT 诊断 readout 和默认关闭的资源接口不进入默认 prior 计算。早先 GFT 50 关 wall time 与旧 grid 50 关 wall time 不再直接比较为架构开销结论。当前同机顺序 10 关正式预算重测显示 grid `real=644.04s`、GFT `real=641.79s`，aggregate、逐关字段和 root trace 完全一致，未观察到默认 GFT 入口相对 grid 入口的系统性额外开销。后续启用 `K/Q/C` 或非零 `access/deadlock/ordering/commitment` 权重时，需重新做同机效率 gate。

#### K：区域容量裕度第一切片

`K` 表示目标相关区域的容量健康度，用于刻画“某个目标区域是否已经被承诺箱子、错误箱子或入口阻塞耗尽容量”。当前实现采用静态 floor 图的割点分解：移除割点后得到连通分量，保留含目标的分量作为目标区域；目标本身若位于割点上，则单独形成一个目标区域。该分解是保守的结构读出，不作为硬约束。

每个区域记录 `goal_count`、`box_count`、`committed_box_count`、`blocked_entry_count`、`wrong_box_count`、`raw_slack` 和 `capacity_debt`。其中 `committed_box_count` 只统计静态可行目标集合完全落在该区域目标集合内的箱子，避免把可能去其他目标的过路箱子误判为容量占用；`blocked_entry_count` 统计非目标入口割点上被箱子占用的情况；`wrong_box_count` 统计处于区域内、尚未在目标上、且静态可行目标集合与该区域目标集合不相交的箱子。区域总读数为 `capacity_health=-sum(capacity_debt)`，数值越大越好。

`K` 通过 `--gft-capacity-weight` 作为默认关闭的软先验分量进入 `I_t` 动作相关性：

```text
score(a) += capacity_weight * (K(s') - K(s))
```

因此它只调整动作优先级，不剪枝、不改变精确格点转移、不改变 TVA 或叶值。默认 `capacity_weight=0` 时，当前 `41/50` 复现路径保持不变。

开发验证采用 10 关正式预算切片，而不是每次都运行 50 关。`capacity_weight=0.25` 在同机 10 关切片上从默认 GFT 的 `6/10` 提升到 `7/10`，新增解出 `level_index=5`，状态分布从 `deadlocked=1,max_steps=2,no_action=1,solved=6` 变为 `deadlocked=1,max_steps=1,no_action=1,solved=7`。该结果说明 `K` 具备正向开发信号，但仍需在更多切片或 50 关阶段性确认前保持为可选配置。

#### Q：顺序依赖债务第一切片

`Q` 表示目标区填充顺序和入口保留顺序的债务，用于刻画“浅层目标是否过早占用、目标区域未完成时入口是否被箱子堵住”。当前实现不学习偏序图，而是从 `K` 已使用的静态区域分解中构造一个保守偏序读出：对每个含目标区域，从入口割点到各目标计算静态最短深度；更深目标应先于更浅目标被填入。若浅层目标已有箱子而更深目标仍未占用，则计为一条 goal-order violation。若目标区域仍有未占用目标，而入口割点被非目标箱子占用，则计为一条 entry-order violation。`Q` 的数值为两类 violation 数之和，越大越差；当前静态构图不产生 cycle，因此 cycle count 记录为 `0`。

`Q` 通过 `--gft-ordering-weight` 作为默认关闭的软先验分量进入 `I_t` 动作相关性：

```text
score(a) -= ordering_weight * (Q(s') - Q(s))
```

因此，降低顺序债务的动作会被升权，增加顺序债务的动作会被降权。该机制不剪枝、不改变精确格点转移、不改变 TVA 或叶值。默认 `ordering_weight=0` 时，当前 `41/50` 复现路径保持不变。

初始动作审计显示该读出不是空信号：50 关 heldout 的 `223` 个初始合法动作中有 `60` 个动作改变 `Q`，`38/50` 关至少有一个初始动作触发非零 `Q` 变化。默认 3 关等价 gate 与旧 GFT 输出的 aggregate、逐关字段和 root trace mismatch 均为 `0`。但 10 关正式预算门槛中，`ordering_weight=0.25` 和 `0.1` 均为 `6/10`，没有超过默认 GFT 的 `6/10`，且均把 `level_index=5` 从 `max_steps` 推为 `deadlocked`，deadlocked 从 `1` 增至 `2`。因此，`Q` 第一切片保留为 active readout 和审计变量，但在该单因素阶段不进入推荐配置；后续完整因素测试显示，`ordering_weight=0.05` 在强运输成本底座上可保持 `8/10`，所以该单因素结论不得外推为 `Q` 结构无效。

组合交互测试采用 `capacity_weight=0.25` 与 `ordering_weight=0.05`，目的是检验 `Q` 是否能在 `K` 的容量健康保护下提供互补信号。该配置在同一 10 关正式预算切片上为 `6/10`，状态分布 `deadlocked=1,max_steps=2,no_action=1,solved=6`，与默认 GFT 总数相同，但相对 `K` 单独的 `7/10` 退化；逐关比较显示 `K` 单独解出的 `level_index=5` 在组合后回到 `max_steps`。因此当前 `Q` 构图不进入 `K+Q` 推荐组合。

#### C：目标承诺清晰度第一切片

`C` 表示箱子-目标分配是否清晰。当前实现使用每个箱子到每个目标的静态 reverse-pull 距离构造运输成本矩阵；不可达位置记为不可行。对可行成本矩阵取 `exp(-cost)` 后执行 Sinkhorn 行列归一化，得到软箱子-目标分配矩阵。若分配矩阵接近一对一确定分配，则熵低、承诺清晰度高；若多个目标仍然同等可能，则熵高、承诺清晰度低。当前读数为：

```text
C(s) = 1 - normalized_entropy(Sinkhorn(exp(-cost(box, goal))))
```

该读数越大越好，默认不影响在线规划。`C` 通过 `--gft-commitment-weight` 作为默认关闭的软先验分量进入 `I_t` 动作相关性：

```text
score(a) += commitment_weight * (C(s') - C(s))
```

初始动作审计显示 `C` 覆盖较广：50 关 heldout 的 `223` 个初始合法动作中有 `155` 个动作改变 `C`，`32/50` 关至少有一个初始动作触发非零 `C` 变化，初始承诺清晰度范围约 `0.080029` 到 `0.850247`，初始动作变化范围约 `-0.731939` 到 `0.336704`。默认 3 关等价 gate 与上一版统一先验输出的 aggregate、逐关字段和 root trace mismatch 均为 `0`。

但 10 关正式预算门槛中，`commitment_weight=0.5` 与 `0.25` 均为 `6/10`，没有超过默认 GFT 的 `6/10`，且均把 `level_index=5` 从 `max_steps` 推为 `deadlocked`，deadlocked 从 `1` 增至 `2`。补充组合测试使用 `capacity_weight=0.25` 与 `commitment_weight=0.25`，结果为 `7/10`，状态分布 `deadlocked=1,max_steps=1,no_action=1,solved=7`，与 `K` 单独配置逐关 solved/status/pushes 完全一致。因此，`C` 第一切片保留为 active readout 和 TVA/诊断候选特征；它在 `K+C` 中未造成额外退化，但也未提供超过 `K` 单独的增益。后续完整因素测试进一步显示，`commitment_weight=0.05` 与强运输成本底座及 `ordering_weight=0.05` 配合时保持 `8/10`，所以 `C` 不再被视为“单独不推荐即结构弱”的因素。

### GFT-SD 完整结构对齐版本

当前 GFT-SD 版本按 `sokoban_gptpro.md` 明确定义为两层系统。底账层为精确 Sokoban 符号状态 `y_t=(B_t,R_t)`，其中 `B_t` 是排序后的箱子位置集合，`R_t` 是当前箱子布局下玩家可达区域；该层由 `SokobanGridStockFlowModel` 负责，存量为格点箱子占用和玩家占用，流量为一次合法 push 对应的 one-hot floor edge，转移严格保持 `b_next=b+Bf`。抽象层为 `z_t=Phi(y_t)=(P,M,A,K,D,Q,C,I)`，只从底账层读出，不独立 rollout，不替代转移 provider。

八个资源库存的当前定义如下。`P` 是目标运输剩余功，由箱子到目标的成本矩阵最优匹配得到；成本项为 `d_g(b_i)+lambda_turn*turns+lambda_access*access_risk+lambda_zone*zone_risk`，默认三个 `lambda` 均为 `0`，因此默认值等价于既有 matching lower bound。`M` 是箱子机动性，使用 viable push count；`A` 是玩家推位可达性，使用 legal push count 与 reachable count；`K` 是区域容量裕度，使用目标区域容量债务的负值；`D` 是死锁债务，等于硬死锁债务加软死锁压力，软压力由机动性损失、匹配脆弱性、容量压力、可达性损失和顺序违例五项加权；`Q` 是顺序依赖债务，来自目标区域入口深度偏序和入口阻塞违例；`C` 是目标承诺清晰度，使用同一运输成本矩阵经 Sinkhorn 得到的箱子-目标软分配熵；`I` 是动作相关性，用于动作先验。

显式流量接口为 `gft_resource_flow(state, child_state)`。符号约定与设计文档一致：`P` 的流量为 `P_t-P_{t+1}`，正值表示运输剩余功下降；`M/A/K` 的流量为后一状态减前一状态，正值表示资源增加；`D/Q` 的流量为后一状态减前一状态，正值表示债务增加；`C` 的流量为分配熵下降 `H_C(t)-H_C(t+1)`。由于 `P` 可能在死锁状态为非有限值，接口同时记录原始流量和用于在线消费的 bounded progress。

系统动力学奖励接口为 `gft_sd_reward()`，实现 `r_SD=w_P f_P+w_M f_M+w_A f_A+w_K f_K-w_D f_D-w_Q f_Q-c_push`，并可附加终局奖励和死局奖励。显式值函数接口为 `gft_sd_value()`，实现 `V_SD(z)=-beta_P P+beta_M M+beta_A A+beta_K K-beta_D D-beta_Q Q`，并保留 `C` 的可选权重以供后续实验。`gft_tva_feature_dict()` 输出稳定特征合同，供未来严格 TVA 训练读取统一 GFT 资源向量和软死锁分量。

当前方法区分三种配置层级。第一，正式 50 关已确认配置仍为默认 GFT/grid-equivalent prior：动作先验权重 `progress=1.0`、`goal=1.0`、`revisit=1.0`，`mobility/access/capacity/deadlock/ordering/commitment/bottleneck=0.0`，运输成本扩展 `lambda_turn=lambda_access=lambda_zone=0.0`，温度 `2.0`，uniform mix `0.1`。该配置在 50 关 heldout 上复现 `41/50`，仍是当前正式最强结果。

第二，10 关开发候选已从 `K` 单独候选更新为运输成本三项组合及其轻量扩展。当前最好 10 关配置为 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25`，结果 `8/10`；在此基础上加入 `ordering_weight=0.05` 仍为 `8/10`；再加入 `commitment_weight=0.05` 仍为 `8/10`。这些配置说明 `Q` 和 `C` 在强运输成本底座上存在保留价值，但该结论仍属于 10 关内循环，不替代 50 关正式结论。

第三，结构保留不等于手工在线默认开启。`K/M/A/D/C` 以当前手工权重加入强运输成本底座时多为 `7/10`，未构成删除因素的严重负证据；`I` 的瓶颈 overlap 以 `0.10` 加入强底座时为 `6/10`、弱权重 `0.02` 加入 `P+Q+C` 时为 `7/10`，因此当前手工瓶颈 overlap 不进入推荐在线配置，但仍作为 `gft_bottleneck_summary()`、动作相关性诊断和未来可学习权重候选保留。后续因素裁剪原则为：只有在合理权重和组合上下文中反复出现明显低于默认线、死锁大幅增加或耗时失控，才考虑从在线消费候选中降级；单因素未涨分只说明当前手工消费方式证据不足。

### GFT-TVA v1：严格 search-value 残差接入

GFT-TVA 的目标是把 `sokoban_gptpro.md` 中的叶值定义落到当前两层 GFT-SD 架构中：

```text
V_leaf(z_h) = V_SD(z_h) + V_TVA(z_h)
```

其中 `V_SD(z)` 来自 `SokobanGFTSDModel.gft_sd_value()`，`V_TVA(z)` 学习显式资源值函数无法覆盖的长期可解性残差。训练脚本为 `train_sokoban_gft_tva.py`。该脚本不调用 diagnostic teacher 或 best-first teacher 生成标签，而是使用同一 GFT-MCTS 家族的根节点统计构造 search-value 目标：

```text
V_search(s) = sum_a p_alpha(a | s) Q_H(s,a)
p_alpha(a | s) ∝ visits(s,a)^alpha
target_residual = V_search(s) - V_SD(s)
```

当前 v1 使用 ridge 回归作为最小可审计学习器，输入为 `gft_tva_feature_dict()` 的稳定特征合同 `gft_tva_feature_vector_v1`。checkpoint 为 JSON，包含 `feature_contract`、`value_scale`、`target_mode=gft_sd_residual`、ridge 标准化参数和线性权重。闭环入口 `evaluate_sokoban_gft_sdmcts.py` 在 `--leaf-value tva` 下读取该 checkpoint；若 target mode 为 `gft_sd_residual`，则执行 `projected_V_SD + prediction`。旧 grid TVA checkpoint 仍可由同一参数读取。

由于 `gft_sd_value()` 的资源量纲并不天然等于 MCTS 回报量纲，闭环消费采用规划值尺度投影：先按显式公式计算 `V_SD(z)`，再裁剪到 `[deadend_value, goal_value]`。训练残差使用同一投影定义，避免未校准的非有限运输进度惩罚或大尺度资源值主导 MCTS 备份。该投影只作用于 leaf value 消费层，不改变 GFT 资源向量、资源流量、`r_SD` 或诊断日志。

v1 首轮结果为负。训练集为关卡 `8-47`，与 10 关开发评测 `48-57` 不重叠；强 GFT 先验为 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25,ordering_weight=0.05,commitment_weight=0.05`。离线训练 gate 通过，`204` 样本中 train `169`、valid `35`，valid MAE `0.776`、corr `0.056`。闭环 10 关中，强 GFT 先验加 `sd_heuristic` 叶值为 `8/10`；改用显式 `gft_sd` 叶值降为 `1/10`；`V_SD+TVA` 降为 `0/10`。因此 GFT-TVA v1 证明了严格接口已接入，但不提供在线性能收益。

方法判定为：当前不应继续扩大 ridge GFT-TVA 的 50 关测试。后续 TVA 有两个更合理方向。第一，先校准 `V_SD` 的 `beta` 权重或值尺度，使显式 `gft_sd` 叶值在 10 关门槛上至少接近 `sd_heuristic`。第二，保留当前强 `sd_heuristic` 作为锚点，训练 `sd_heuristic_residual`，等残差信号闭环有效后再迁移到纯显式 `V_SD`。在任一方向通过 10 关门槛前，正式 50 关最强结果仍以 `sd_heuristic + GFT action prior` 的 `41/50` 为准。

#### GFT-SD 显式叶值的当前校准口径

显式 `V_SD` 进入 MCTS 叶节点时区分两个消费模式。`linear_projected` 是 GFT-TVA v1 的首轮负结果模式：先调用 `gft_sd_value()`，再把所得资源值裁剪到 `[deadend_value, goal_value]`。该模式保留为审计对照，但当前不作为推荐叶值，因为它在强 GFT 先验的 10 关门槛上仅为 `1/10`。

当前 planner-ready 的显式叶值候选为 `discounted_progress`。该模式不直接使用未校准的资源线性和，而是把目标运输进度 `P` 映射到与 `sd_heuristic` 相同的规划回报尺度：

```text
V_SD_base(s) = gamma^P * goal_value - push_cost * (1 - gamma^P) / (1 - gamma)
```

当 `gamma=1` 时，退化为 `goal_value - push_cost * P`；当 `P` 非有限时，返回 `deadend_value`。该定义仍然使用 GFT-SD 第二层的运输进度读出，但避免把 `gft_sd_value()` 中尚未校准的 `beta` 权重直接交给 MCTS 备份。

第二层资源可以作为叶值校正项进入同一公式之后：

```text
V_SD(s) = V_SD_base(s)
        + w_M M/max_pushes
        + w_A A/max_pushes
        + w_K K/n_boxes
        - w_D D/n_boxes
        - w_Q Q/n_boxes
        + w_C C
```

其中 `M` 为箱子机动性，`A` 为玩家推位可达性，`K` 为区域容量裕度，`D` 为死锁债务，`Q` 为顺序依赖债务，`C` 为目标承诺清晰度。默认所有叶值资源权重为 `0`，因此 `discounted_progress` 的基底与当前强 `sd_heuristic` 对照逐关一致。保守资源组合 `w_K=w_Q=w_C=0.05` 在 10 关门槛上保持 `8/10`，说明这些抽象因素可以进入叶值而不造成严重退化；但其运行时间高于零权重基底，后续扩大测试前应缓存第二层特征。

因此，下一版 GFT-TVA 若继续采用 `V_leaf=V_SD+V_TVA`，训练残差应优先锚定 `discounted_progress` 口径，而不是沿用 `linear_projected` 负结果口径。正式 50 关最强结论仍暂不改变，仍为 `sd_heuristic + GFT action prior` 的 `41/50`。

#### GFT-SD 第二层特征缓存

GFT-SD 第二层资源采用按状态缓存的内部实现，以保证在线规划中的重复叶值评估不会反复重建同一资源合同。模型层缓存包括 `gft_resource_vector(state)` 的完整资源向量和 `gft_tva_feature_dict(state)` 的稳定 TVA 特征字典；规划器层缓存包括 `gft_sd_leaf_value(state)` 和 GFT-TVA 预测值。缓存键使用 canonical state，即箱子位置与玩家可达分量等价类；这与现有 SD 叶值缓存口径一致，且满足当前 GFT 资源读出对玩家具体坐标的不变性假设。

该缓存只改变计算路径，不改变输出合同。`gft_resource_vector()` 和 `gft_tva_feature_dict()` 对外仍返回普通字典，并在命中缓存时返回副本，避免调用方修改内部缓存。`trace_gft_readouts`、结果 JSON、TVA checkpoint 合同和 `gft_tva_feature_vector_v1` 的特征顺序均保持不变。

当前效率验证使用保守第二层叶值组合 `w_K=w_Q=w_C=0.05`。缓存前后 10 关 aggregate 和逐关 `status/solved/pushes/steps_executed/deadlock_prunes` 完全一致，均为 `8/10`；wall time 从 `550.80s` 降至 `481.41s`。因此后续 TVA 训练或闭环评估可以默认依赖该缓存实现。该结果也说明，若继续追求更大效率收益，应进一步检查搜索展开策略和动作先验计算，而不是继续围绕特征合同组装做微调。

#### GFT-TVA v2 的当前方法判定

GFT-TVA v2 采用已校准的 `discounted_progress` 显式叶值作为残差基底，并保留 `w_K=w_Q=w_C=0.05` 的保守第二层叶值校正。训练合同为：

```text
V_search(s) = sum_a p_alpha(a|s) Q_H(s,a)
V_TVA target = V_search(s) - V_SD_discounted_progress(s)
V_leaf(s) = V_SD_discounted_progress(s) + V_TVA(s)
```

评估入口现在检查 GFT-TVA checkpoint 的显式叶值合同。若 checkpoint 的 target mode 为 `gft_sd_residual`，则运行时的 `gft_sd_leaf_mode` 和 `gft_sd_leaf_weights` 必须与 checkpoint 记录一致；否则停止评估。这一检查用于防止把 `linear_projected`、零权重 `discounted_progress` 或 `K/Q/C=0.05` 等不同基底下学到的残差混用。

v2 的离线拟合比 v1 更好，valid corr 从 `0.056` 提高到 `0.668`，但闭环 10 关从显式 `V_SD` 的 `8/10` 降到 `1/10`，并产生 `8` 个 deadlocked。残差目标和预测整体为负，说明当前短视界 search-value 目标倾向于给显式 `V_SD` 做负修正；该负修正在闭环中会连续改变状态排序，导致搜索偏向短路径死锁。初始状态预测没有极端爆炸，因此问题不是单点数值异常，而是全强度残差缺少闭环安全门。

因此，当前方法结论为：`discounted_progress` 基底适合继续作为显式 `V_SD`，但全强度 ridge GFT-TVA v2 不进入推荐在线配置。后续 TVA 不应直接扩大数据或跑 50 关，而应先加入残差缩放、残差裁剪、非正残差门控、或基于不确定性/排序保持的 safety gate，并仍以 10 关门槛验证是否至少不低于显式 `V_SD`。

#### GFT-TVA 当前直接终端价值口径

后续 GFT-TVA 主线从残差口径修订为直接终端价值口径。该口径与 NeurIPS 论文中 `V_psi(s)` 对齐搜索值、以及 Beer Game 和 Cartpole 已完成实现中“叶节点直接调用价值网络”的消费方式一致。训练目标为：

```text
V_search(s) = sum_a p_alpha(a|s) Q_H(s,a)
V_psi target = V_search(s)
```

闭环默认叶值为：

```text
V_leaf(s_leaf) = V_psi(s_leaf)
```

显式 `V_SD` 不再默认作为 TVA 训练残差的基底，而是保留为三类用途：第一，作为可解释系统动力学读出；第二，作为 `--leaf-value gft_sd` 的零学习对照；第三，作为直接价值 TVA 的运行时兜底混合项。混合公式为：

```text
V_leaf(s_leaf) = (1 - alpha) * V_SD(s_leaf) + alpha * V_psi(s_leaf)
```

其中 `alpha` 由 `--tva-value-blend` 控制。`alpha=1.0` 是论文一致的纯直接价值消费；`alpha<1.0` 只作为安全兜底和消融，不改变 checkpoint 的训练目标。残差口径 `gft_sd_residual` 继续保留，用于复现 v1/v2 负结果和后续对照，但不再是默认训练模式。

实现合同随之调整。`train_sokoban_gft_tva.py` 的默认 `--target-mode` 为 `direct_value`；历史名称 `value` 只作为兼容别名，写入结果时统一规范为 `direct_value`。checkpoint 记录 `direct_value_formula=V_psi=V_search`、`leaf_formula=V_leaf=V_psi` 和 `paper_alignment=direct_terminal_value_tva`。`evaluate_sokoban_gft_sdmcts.py` 在读取 `direct_value` checkpoint 时，默认直接返回 TVA 预测；只有当 `--tva-value-blend<1.0` 时才同时读取显式 `V_SD`，并要求运行时 `gft_sd_leaf_mode` 与 `gft_sd_leaf_weights` 和 checkpoint 合同一致。

当前只完成了接口修订和小规模 smoke，尚未形成正式性能结论。下一轮正式实验应沿用 10 关开发切片 `48-57`，训练关卡仍避开该切片，并比较纯直接价值 `alpha=1.0` 与保守混合 `alpha<1.0`。正式 50 关最强结果仍暂不改变，仍为 `sd_heuristic + GFT action prior` 的 `41/50`。

#### GFT-TVA direct-value v3 的当前方法判定

direct-value v3 已完成首轮正式 10 关评估。该版本训练目标符合当前直接终端价值口径：

```text
V_psi target = V_search(s)
V_leaf = V_psi                  # pure direct-value
V_leaf = 0.75 * V_SD + 0.25 * V_psi  # conservative blend
```

训练仍使用关卡 `8-47`，与 10 关开发评测 `48-57` 分离。离线指标为 gate `PASS`、valid MAE `0.752`、valid corr `0.710`，说明 ridge 模型可以在该数据集上拟合 search-value target 的主要方差。target 本身为正值搜索值，train target mean `3.838`，因此该版本不同于 residual v2 的整体负残差修正。

闭环结果为负。纯 `V_psi` 在 10 关中为 `1/10`，`alpha=0.25` 混合也为 `1/10`；二者都只解出 `level_index=4`。显式 `V_SD` 对照在相同 10 关和同一强 GFT 先验下为 `8/10`。失败形态以 `max_steps` 和 `no_action` 为主，而 residual v2 以 `deadlocked` 为主。这表明当前 direct-value v3 不是简单复现残差 TVA 的死锁偏置，而是离线 root search-value 拟合与闭环执行状态分布、动作排序之间仍存在错配。

因此，当前方法结论为：直接终端价值 TVA 继续作为论文一致的接口口径保留，但 `GFT feature vector + ridge scalar value + 当前短视界 root search-value target` 这一具体实现不进入推荐在线配置。后续不宜继续做无界 `alpha` 扫描；若继续 TVA，应优先改变学习器容量和 target 构造，例如非线性价值模型、on-policy 闭环状态再标注、成功/失败分层采样、TD bootstrapping、动作排序辅助损失或局部结构目标。正式 50 关结果仍以 `sd_heuristic + GFT action prior` 的 `41/50` 为当前最强配置。

#### GFT-TVA 独立截断价值审计

TVA 后续不再只依赖训练/验证集拟合指标或闭环试错。与 SD 结构可独立验证相对应，TVA 必须先通过独立截断价值审计，证明其在 MCTS 实际会遇到的叶节点状态上能够给出可靠终端价值。当前审计由 `audit_sokoban_gft_tva_truncation_value.py` 实现。

审计流程如下。第一，用显式 `V_SD` 叶值运行 GFT-SD-MCTS，并记录规划过程中实际调用 `leaf_value()` 的叶节点状态；TVA 不参与采样、动作选择或树搜索。第二，对这些叶节点状态使用离线 exact-state best-first teacher 继续求解，生成剩余价值标签。第三，在同一批状态上比较 TVA 预测、按 checkpoint 合同形成的 TVA leaf value、显式 `V_SD` leaf value 和 `sd_heuristic` value。第四，只用高置信 teacher 标签进入通过门槛；teacher budget unsolved 状态单独报告，用于观察价值函数是否对难解状态系统性乐观或悲观。

当前通过条件包括：高置信标签数量不少于 `20`；TVA 高置信 MAE 不高于显式 `V_SD`；TVA 高置信相关系数不低于 `0.30`；同一 root 下叶节点的两两排序准确率不低于显式 `V_SD`；正值状态不得明显低估，非正值状态不得明显高估。排序条件被列为门槛，是因为 MCTS 主要消费 value 的相对排序；点值 MAE 略优但排序更差，仍可能破坏动作选择。

direct-value v3 已在该审计下失败。`teacher_max_expansions=4000` 的 10 关审计中，TVA 高置信 MAE `0.805`，略好于显式 `V_SD` 的 `0.813`，但 pairwise ordering accuracy `0.800` 低于显式 `V_SD` 的 `0.827`。`teacher_max_expansions=10000` 的复核中，TVA 高置信 MAE `0.822`，好于显式 `V_SD` 的 `0.881`，但排序准确率 `0.821` 仍低于显式 `V_SD` 的 `0.866`。因此当前 direct-value v3 不能作为 MCTS 叶值接入。

方法约束随之更新：任何后续 TVA 版本必须先通过独立截断价值审计，再进入 10 关闭环门槛。若审计只表现为点值 MAE 改善但排序不如显式 `V_SD`，不得进入推荐配置。若审计中 budget-unsolved 状态上出现系统性正偏，也应在训练目标中处理难解状态或引入不确定性/失败状态标注，而不是通过闭环混合权重掩盖。

#### 删失标签与难解状态风险

`teacher_budget_unsolved` 不再被视为真实负终端价值标签。它表示离线 teacher 在给定预算内未能定性该状态，属于删失观测：状态可能不可解，也可能只是搜索预算不足。因此，该类状态不得直接硬回归到固定负值；更合适的处理包括风险分类、不确定性标记、弱排序约束、分层采样或更高预算复核。

本轮新增删失风险探针，用 high-budget 截断价值审计中的叶节点作为数据。标签定义为：teacher solved 为 low-risk，teacher budget unsolved、proven dead-end 和 no legal actions 为 high-risk。由于当前样本中只有 solved 与 budget-unsolved，探针实际检验的是“预算未解风险”是否可由当前 GFT 特征跨关卡识别。评估采用按关卡留一交叉验证，避免将同一关卡内的可分性误判为泛化能力。

结果显示，当前 GFT 特征加线性 ridge 风险头无法稳定识别 budget-unsolved：AUC `0.490`，balanced accuracy `0.515`。原始 `V_SD` 和 `V_psi` 对 budget-unsolved 有一定方向性 AUC，但方向是值越高越像预算未解，这与终端价值语义相反。这说明当前 scalar value 同时混入了“正向进度”和“难解状态乐观偏差”，不能直接作为安全叶值。

因此，下一版 TVA 若要处理难解状态，应优先改变数据和模型，而不是只改变闭环消费权重。可行方向包括：收集更多高置信 dead-end/no-legal 负例，扩大 teacher 预算或使用多种离线求解器交叉标注，增加 on-policy 叶节点采样，使用非线性模型或多头结构同时预测终端价值与风险/不确定性，以及引入排序辅助损失。只有当风险/不确定性头在按关卡 heldout 上通过独立审计后，才考虑让它门控 MCTS 叶值。

#### GFT-TVA v4 叶节点监督与校准锚定

GFT-TVA v4 将训练对象从根节点搜索统计改为实际截断叶节点。采样时，GFT-SD-MCTS 使用显式 `V_SD` 叶值和 GFT 动作先验运行；每次 `leaf_value()` 被调用时记录叶节点状态。TVA 不参与采样动作选择。随后离线 exact-state best-first teacher 从该叶节点继续求解，生成标签。

标签分三类处理。teacher solved、proven dead-end 和 no legal actions 为高置信价值标签，进入终端价值回归；teacher budget unsolved 为删失风险标签，只进入风险头或风险诊断，不回归到固定负值。训练集还从训练关卡浅层 BFS 中补充 high-confidence blocked 状态，以避免只有 solved 正例而缺少明确负例。

当前实现的 `mlp_guarded_v1` checkpoint 是 JSON 格式，仍使用 `gft_tva_feature_vector_v1` 特征合同。网络包含共享 MLP、价值头和风险头；评估器以 NumPy 路径读取权重，无需在线依赖 Torch。一般形式为：

```text
risk_probability = sigmoid(risk_head(Phi(s)))
guarded_value = value_head(Phi(s)) - risk_penalty * risk_probability
V_leaf = clip((1 - anchor_weight) * guarded_value
              + anchor_weight * (V_SD + anchor_bias))
```

其中 `V_SD` 使用当前 planner-ready 的 `discounted_progress` 口径及 `K/Q/C=0.05` 保守叶值权重。`anchor_bias` 可由训练集 teacher-solved 叶节点的 `target - V_SD` 中位数估计。该偏置用于修正显式 `V_SD` 的系统性点值偏差，同时保持同根候选排序。

当前结果给出两个方法约束。第一，全强度非线性风险校正虽然可拟合训练标签，但在独立截断价值审计中破坏排序，不能直接接入 MCTS。第二，校准锚定版本通过独立审计，但闭环只在 `--tva-value-blend=0.25` 时恢复到显式 `V_SD` 的 `8/10`，未产生新收益。因此，校准锚定 checkpoint 只作为“通过独立审计的 TVA 训练基线”，不作为推荐在线配置。后续若继续 v4，应优先加入标签缓存、排序保持残差、按 root 的 pairwise loss 或更稳定的风险/不确定性建模。

#### 当前 50 关正式配置：显式 GFT-SD 叶值

截至 2026-06-15，Sokoban 当前正式最强配置更新为显式 GFT-SD 叶值版本，50 关 heldout 结果为 `44/50`。该配置仍不使用 TVA 学习权重；学习组件尚未进入推荐闭环。其方法组成如下：

```text
transition provider = exact_grid_stock_flow_v1
resource layer = GFT z_t = Phi(y_t)
leaf value = explicit V_SD in discounted_progress planner scale
leaf resource correction = K/Q/C = 0.05
action prior = GFT delta prior with transport turn/access/zone + Q + C
tree search = MCTS with transposition and revisit penalty
```

具体超参为：`horizon=20`，`budget=4096`，`max_steps=80`，`revisit_penalty=1.0`，transposition，`gft_prior=gft_delta`，temperature `2.0`，uniform mix `0.1`；动作先验权重为 `progress=1.0,goal=1.0,revisit=1.0,ordering=0.05,commitment=0.05`，运输成本权重为 `lambda_turn=0.10,lambda_access=0.25,lambda_zone=0.25`；叶值为 `--leaf-value gft_sd --gft-sd-leaf-mode discounted_progress`，叶值二层校正为 `capacity=0.05,ordering=0.05,commitment=0.05`。

该配置相对上一正式最强 `sd_heuristic + GFT action prior` 的 `41/50` 净增 `3` 关，死锁数保持 `1`，max_steps 从 `5` 降至 `2`，no_action 仍为 `3`。因此当前方法叙述应把 `44/50` 作为 Sokoban 主结果，把 `41/50` 作为被显式 `V_SD` 叶值版本超越的历史强基线。后续 TVA 只有在独立审计通过且闭环超过该 `44/50` 线时，才可更新推荐配置。

#### 未解关卡的抽象层归因方法

当前未解关卡不再只按最终状态标签分类，还按 GFT-SD 八个资源读数进行归因。审计读取每步 `z_t=Phi(y_t)`，并比较目标运输进度 `P`、箱子机动性 `M`、玩家推位可达性 `A`、区域容量裕度 `K`、死锁债务 `D`、顺序依赖债务 `Q`、目标承诺清晰度 `C` 和动作相关性 `I` 的轨迹。该审计的目的不是改变 MCTS，而是判断失败是否来自设计文档外的缺失抽象因素，还是来自已有读数没有被在线策略充分消费。

审计口径区分三种情况。第一，若最终或早期读数中 `K/D/Q` 已经显著报警，则失败归为区域容量、入口锁定或目标顺序风险已被表达但介入不足。第二，若 `viable_push_count=0`、`M=0` 或 `D` 中的 mobility/access 压力升高，而 `K/Q` 不显著，则归为箱子机动性、玩家推位可达性或软死锁压力已被表达但当前在线权重不足。第三，若状态在少量底账状态之间反复出现，且 `P` 在两个值之间来回变化，则归为进度停滞和循环债务问题；该问题不对应 `sokoban_gptpro.md` 中新增的第九存量，更适合由动作相关性 `I`、revisit 惩罚、历史循环债务或 TVA 长期排序来承载。

按该口径，当前 `44/50` 的 6 个失败关卡分布为：`level_index=9` 和 `45` 属于已有 `K/D/Q` 强报警但介入不足；`level_index=14` 和 `46` 属于 `M/A/D` 表达出的准死锁或机动性坍缩，但 `M/A/D` 尚未作为主叶值和主先验分量消费；`level_index=0` 和 `40` 属于后期振荡与进度停滞，其中 `40` 还是当前显式叶值版本相对旧 `41/50` 的唯一回归关。后续方法优先级因此调整为统一激活现有八存量，尤其是保守纳入 `M/A/D/I`，并建立循环/停滞审计；暂不把失败解释为设计文档外必须新增一组并列存量。

#### `M/A/D/I` 在线激活筛查后的方法约束

在当前显式 GFT-SD 叶值线上，`M/A/D/I` 的首次保守激活筛查不更新推荐配置。玩家推位可达性 `A` 的小权重设置 `gft_access_weight=0.10,gft_sd_leaf_access_weight=0.05` 在 10 关开发切片上保持 `8/10`，失败形态不变，并使 solved 关卡 mean pushes 从 `17.75` 降至 `17.25`。因此 `A` 可以作为中性因素保留在后续组合候选中，但它在 6 个正式失败关卡定向复核中没有新增解出，只把 `level_index=14` 从 `no_action` 延后为 `max_steps`。它不构成新主线，也不安排 50 关确认。

箱子机动性 `M` 的当前一阶实现，即使用 legal/viable push count 的即时变化作为动作先验和叶值校正，在本轮表现不稳定。`gft_mobility_weight=0.10,gft_sd_leaf_mobility_weight=0.05` 将 10 关从 `8/10` 降至 `7/10`，新增 `level_index=6:no_action`；降权后与 `A` 组合仍为 `7/10`。因此后续不应继续简单放大或组合当前一阶 `M`，而应优先改成更局部、更目标条件化或多步解锁性的机动性读数，例如箱子簇互锁、关键通道旁箱子的可释放性、两步可推性或局部不可逆推位损失。

死锁债务 `D` 和动作相关性 `I` 当前在线公式不合格。`D` 的小权重设置 `gft_deadlock_weight=0.05,gft_sd_leaf_deadlock_weight=0.05` 将 10 关降至 `6/10`，新增 `level_index=5:max_steps`、`6:no_action`；`I` 的瓶颈重叠设置 `gft_bottleneck_weight=0.02` 也降至 `6/10`，新增 `level_index=5:max_steps`、`6:max_steps`。这说明当前 `D/I` 的在线介入会破坏可解边界样本的排序。该结果不否定 `D/I` 作为 GFT-SD 抽象层因素，但方法上要求：在重新接入前，必须先进行分量级审计、权重重标定或转为 TVA/离线排序特征，不得直接沿用本轮在线权重进入组合或 50 关。

因此，当前主线仍保持为显式 GFT-SD 叶值 `P+K/Q/C` 加 GFT action prior 的 `44/50`。后续若继续处理当前 6 个失败关卡，应优先做两个方向：第一，针对 `0/40` 的历史循环和进度停滞机制；第二，针对 `14/46` 的局部互锁和多步解锁性机制。单纯加入当前形式的一阶 `M`、整体 `D` 或瓶颈重叠 `I` 不再是高优先级路径。

#### 在线轨迹 GFT-TVA 机制

在线轨迹 GFT-TVA 与此前离线 TVA 的主要区别是采样分布。离线 TVA 使用预先构造的状态或 MCTS 叶节点，再由 teacher 标注；在线轨迹 TVA 则直接记录当前 GFT-SD-MCTS 闭环执行中实际访问的根状态和最终状态。因此，它优先覆盖当前系统真实会遇到的误判、振荡、准死锁和失败状态，而不是扩大一批可能分布不匹配的离线样本。

当前实现由 `train_sokoban_gft_online_tva.py` 提供。采样时默认使用当前最强显式 `V_SD` 叶值和 GFT action prior；TVA 不参与采样，除非显式设置 `--sampling-leaf-value=tva`。每条轨迹按最终状态分为三类标签。成功求解轨迹为高置信正价值标签，状态价值按剩余 push 数折扣到 `goal_value`。`deadlocked` 与 `no_action` 轨迹为高置信负价值标签，状态价值按剩余 push 数折扣到 `deadend_value` 或 `no_action_value`。`max_steps` 轨迹默认为删失风险标签，只训练风险头，不作为确定负价值标签；只有显式设置 `--max-steps-value-weight>0` 时才会作为弱价值标签进入回归。

训练 checkpoint 复用 `mlp_guarded_v1` 格式，包含共享 MLP、价值头、风险头和显式 `V_SD` 锚定项。当前默认在线叶值公式为：

```text
V_leaf=clip((1-anchor_weight)*(value_head-risk_penalty*risk_probability)
            +anchor_weight*(V_SD+anchor_bias))
```

默认 `anchor_weight=0.75`，风险概率通过 `risk_penalty` 下调未解或坏状态风险，显式 `V_SD` 仍提供主要锚定。该公式的目的不是保证新 TVA 一定优于基线，而是避免未校准价值头直接替换显式 SD 叶值。闭环正式评估仍必须通过独立截断价值审计和风险审计；若只完成在线训练 smoke，不得进入推荐配置。

本机制切片已通过两类 smoke。全 `max_steps` 低预算 smoke 产生风险样本但无价值样本，gate 因 `min_value_train_samples` 失败，证明步数耗尽不会被默认误标为确定负价值。单个 solved 轨迹 smoke 产生 `13` 个高置信正价值样本，gate 为 `PASS`，并能被评估入口以 `--leaf-value tva` 读取执行。当前结果只证明机制可运行，不代表在线 TVA 已有闭环性能收益。

#### 在线轨迹 GFT-TVA 的当前受限候选

在线轨迹 TVA 的负价值标签策略已修订为默认只标注失败终点。参数 `--failure-value-label-scope` 控制 `deadlocked/no_action` 轨迹的价值标签范围，默认值为 `final`；此时，成功轨迹的根状态仍按剩余 push 数折扣到 `goal_value`，失败轨迹中只有最终失败状态回归到 `deadend_value` 或 `no_action_value`，失败轨迹的中间状态只进入风险头。可选值 `all` 用于复现实验或消融，但首轮审计显示它会把闭环策略失败误写成整条路径的真实负价值，导致在线 TVA 对高置信正价值叶状态严重低估。

当前受限候选的训练集来自 `train/001.txt` 前 40 关，低预算采样参数为 `horizon=8,budget=192,max_steps=40,seed=0`。训练配置为 `failure_value_label_scope=final`、`anchor_weight=0.85`、`risk_penalty=1.0`、`risk_loss_weight=0.25`、`anchor_bias_mode=none`、`epochs=300`、`hidden_dim=64`、`hidden_layers=2`。训练数据包含 `834` 个状态样本，轨迹状态分布为 `max_steps=648,solved=134,no_action=52`，训练价值标签为 `114` 个。checkpoint 仍使用 `mlp_guarded_v1` 和 `gft_tva_feature_vector_v1` 合同。

该候选不以纯 TVA 叶值进入 MCTS，而只以保守混合形式进入：

```text
V_leaf = 0.75 * V_SD + 0.25 * V_psi_guarded
```

其中 `V_SD` 是当前 planner-ready 的 `discounted_progress` 显式 GFT-SD 叶值，并带 `K/Q/C=0.05` 保守二层校正；`V_psi_guarded` 是 checkpoint 内部锚定和风险惩罚后的直接价值预测。`alpha=0.25` 不是正式推荐超参，而是当前候选在独立截断价值审计中通过的最小安全混合点。

独立截断价值审计使用未参与训练的 `train/002.txt` 前 10 关，`horizon=8,budget=192,teacher_max_expansions=4000`。TVA 不参与采样，叶状态由显式 `V_SD` 的 GFT-SD-MCTS 产生，再由离线 exact-state best-first teacher 标注。`alpha=0.25` 混合叶值在 `78` 个高置信样本上通过 gate：MAE `0.489`，优于显式 `V_SD` 的 `0.552`；相关系数 `0.771`，略高于显式 `V_SD` 的 `0.762`；同 root 排序准确率 `0.962`，高于显式 `V_SD` 的 `0.949`。

删失风险审计仍为负结果。以同一审计样本做按关卡留一风险探针，AUC 为 `0.442`，balanced accuracy 为 `0.504`，risk recall 为 `0.238`，未达到风险门控要求。因此当前风险头只作为训练中的辅助正则和诊断读数，不得用于闭环剪枝、叶值屏蔽或安全门控。

当前低预算闭环诊断只说明候选值得继续审计，不更新主方法。`train/002.txt` 前 10 关中，显式 `V_SD` 为 `5/10`，混合 TVA 为 `6/10`；未参与训练和审计的 `train/003.txt` 前 10 关中，显式 `V_SD` 为 `4/10`，混合 TVA 为 `6/10`。这些结果采用 `horizon=8,budget=192,max_steps=40`，不等价于正式 50 关预算。正式主配置仍为显式 GFT-SD 叶值 `44/50`；在线 TVA 只有在更大非 heldout 审计、风险/排序审计和接近正式预算的 10 关门槛中继续成立，才可挑战该主结果。

#### 在线轨迹 GFT-TVA 的闭环效果状态

当前 `alpha=0.25` 在线轨迹 GFT-TVA 混合叶值已完成一次 50 关 heldout 探索性测试，结果为 `45/50`。该配置沿用当前显式 GFT-SD 主线的动作先验和显式叶值基底，只将叶值从纯 `V_SD` 改为：

```text
V_leaf = 0.75 * V_SD + 0.25 * V_TVA_guarded
```

其中 `V_TVA_guarded` 来自在线轨迹训练的 `mlp_guarded_v1` checkpoint。风险头仍只影响 checkpoint 内部的 guarded value，不作为 MCTS 的动作剪枝、状态剪枝或独立门控。因此，删失风险审计失败应解释为“当前风险头不能安全门控”，而不应解释为“混合叶值不能闭环测试”。

正式预算 10 关结果显示，`alpha=0.10` 为 `7/10`，低于显式 `V_SD` 的 `8/10`；`alpha=0.25` 为 `8/10`，与显式 `V_SD` 持平，但改变失败构成：它解出原来的 `level_index=9` deadlock，同时丢掉 `level_index=6`。50 关探索性测试进一步显示，`alpha=0.25` 相对显式 `V_SD` 的 `44/50` 净增 1 关，达到 `45/50`：新增解出 `9,40,46`，退化 `6,25`；死锁数保持 `1`，max_steps 保持 `2`，no_action 从 `3` 降为 `2`，mean pushes solved 从 `15.68` 降至 `15.16`，但 wall time 从约 `867s` 增至约 `994s`。

因此，当前方法状态更新为：`44/50` 仍是已完成门槛记录下的正式显式 GFT-SD 主线；`45/50` 是在线轨迹 GFT-TVA 混合叶值的新实测最好候选。将 `45/50` 提升为正式推荐配置前，需要至少完成一次带修复后独立门槛元数据的复跑或等价复核，并处理风险审计与耗时问题。风险头本身不得用于安全门控，除非后续删失风险审计通过。

#### 动态 TVA 消费的当前状态

为分析固定混合 TVA 在 `level_index=6` 和 `25` 上的退化，评估入口增加了默认关闭的动态 TVA 消费开关。其设计目的不是建立新推荐配置，而是审计 TVA 信号能否在更细粒度的候选动作层面纠正错误排序。

动态叶值混合使用候选子状态的风险概率和价值修正 `V_TVA-V_SD`。当风险概率超过阈值且价值修正为强负值时，叶值公式中的 `alpha` 可从基础混合比例提高到较高值：

```text
V_leaf = (1 - alpha(s)) * V_SD + alpha(s) * V_TVA
```

动态动作先验软惩罚使用同一个触发条件，在 GFT 动作先验分数中扣除与负修正幅度成比例的分数。动态根动作选择默认仍按访问次数出手；只有当访问次数最高的候选子状态触发强负 TVA 信号，且另一个候选的均值更高时，才允许切换到均值更高的动作。这些机制均不修改底层精确网格转移，不把风险头作为剪枝或安全门控。

本轮 10 关正式预算消融结果为负。宽阈值动态叶值 `risk>=0.90,V_TVA-V_SD<=-1.0,alpha_high=0.75` 为 `5/10`；加已访问状态额外叶值惩罚 `0.5` 为 `6/10`；收紧阈值并加入动作先验软惩罚、全局按均值选择为 `7/10`；收紧阈值并加入动态根动作选择为 `6/10`。这些结果均低于固定混合 TVA 和显式 `V_SD` 的 `8/10`。

局部诊断显示，第 25 关可以由收紧阈值的动态叶值、动作先验软惩罚和根动作按均值选择修复，17 步解出。这说明 TVA 已经读出该振荡分支的长期风险；但该信号的消费方式不能稳定外推到开发切片。因此，动态 TVA 消费目前只作为默认关闭的审计入口保留，不进入推荐配置，也不用于 50 关扩展。后续若继续推进，应优先训练根动作级排序目标或 pairwise 校准，而不是继续手工扩大风险触发规则。

#### 状态门控 GFT-TVA 混合

当前新的实测最好候选是状态门控 GFT-TVA 混合，50 关 heldout 探索性结果为 `46/50`。该版本把动态 TVA 明确定义为状态依赖混合比例：

```text
V_leaf(s)=(1-alpha(s))*V_SD(s)+alpha(s)*V_TVA(s)
```

其中 `alpha(s)` 可以取 `0`。当前候选的默认策略是：绝大多数状态使用 `alpha=0`，完全退回显式 GFT-SD 叶值；只有候选子状态满足 `risk_probability<=0.30` 且 `V_TVA-V_SD>=0.25` 时，才使用 `alpha=0.25`。高风险负修正状态默认不直接提高叶值 TVA 权重，以避免第 25 关中等强度负修正导致的振荡退化。

该候选还包含一个根动作选择兜底。树内选择仍使用既有 GFT action prior 和 PUCT；最终根动作默认仍按访问次数选择。只有当访问次数最高的根候选满足 `risk_probability>=0.95` 且 `V_TVA-V_SD<=-1.0`，并且另一个候选的均值更高时，才切换到均值更高的候选。该机制的作用是修复第 25 关中“错误动作访问次数略高但均值更低”的选择问题；它不修改叶值公式，不改变转移 provider，也不把风险头用作剪枝。

当前 50 关结果为 `46/50`，相对显式 GFT-SD 主线 `44/50` 新增解出 `level_index=9,40`，无退化；相对固定 `alpha=0.25` TVA 的 `45/50`，恢复 `6,25`，丢掉 `46`，净增 `+1`。失败关卡为 `0:max_steps`、`14:no_action`、`45:no_action`、`46:no_action`。因此方法状态更新为：显式 GFT-SD 叶值 `44/50` 仍是已完整审计的正式主线；状态门控 GFT-TVA `46/50` 是新的实测最好候选；固定全局 `alpha=0.25` 的 `45/50` 退为对照候选。

该候选仍有两个限制。第一，删失风险审计此前未通过，因此风险头不能作为安全剪枝或硬门控，只能作为混合比例和根动作选择兜底的软信号。第二，本次 `46/50` 仍是探索性 50 关结果，尚未带修复后独立门槛元数据复跑。正式推荐配置应在复跑或等价复核后再更新。

#### 固定高权重 TVA 与组合式失败后重跑

状态门控 GFT-TVA 的剩余失败样本显示，部分关卡并非需要新的转移模型，而是需要在特定搜索区域更强地消费 TVA 价值。固定高权重测试使用同一个 `alpha` 计算：

```text
V_leaf=(1-alpha)*V_SD+alpha*V_TVA
```

在 `level_index=14,45,46` 的单关诊断中，`alpha=0.60` 可分别以 `29/13/10` 步解出，`alpha=0.65` 也可解出这三关。这说明当前在线 TVA 已包含能够修复近终局准死锁和入口锁定失败的长期价值信号。与此同时，固定高权重在 10 关开发切片上只有 `7/10`，新增 `5:deadlocked` 与 `6:no_action`，因此不能作为状态门控的替代主配置。

为审计该互补性，评估入口保留两个默认关闭机制。动态叶值软惩罚 `--tva-dynamic-leaf-penalty-weight` 只在高风险且 TVA 给出强负修正时扣减叶值；运输阶段混合 `--tva-transport-phase-blend` 只在状态门控启用时，根据 GFT-TVA 特征中的进度 `P` 与承诺度 `C` 给某些运输阶段状态指定单独的 TVA 权重。二者均不改变精确网格转移、不做剪枝、不作为安全门控，默认关闭时不影响 `46/50` 配置。

当前还记录了一个探索性组合策略：先运行状态门控 GFT-TVA；若单关失败，再对该失败关卡用固定 `alpha=0.60` 重跑；只有重跑成功时才替换原结果。该策略得到 `49/50`，只剩 `level_index=0:max_steps`。该结果的含义是“状态门控和固定高权重 TVA 存在互补消费模式”，不是一个单次 MCTS 配置，也不应替代当前 `46/50` 单次配置作为主方法。若后续要把该信号转为正式方法，需要把失败后重跑机制改写为预先定义的组合搜索策略，并单独评估时间成本、触发规则和 heldout 外推。

截至本条记录，Sokoban 线的结果层级为：显式 GFT-SD 叶值 `44/50` 是已完整审计的正式主线；状态门控 GFT-TVA `46/50` 是当前最强单次候选；状态门控后固定 `alpha=0.60` 失败重跑的 `49/50` 是探索性组合策略正信号。

#### 动态 TVA alpha 规则审计方法

为避免把组合策略收益误解释为单次动态规律，当前新增一个离线轨迹重放审计。该审计由 [audit_sokoban_dynamic_tva_alpha_features.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_dynamic_tva_alpha_features.py) 实现，只读取已有结果文件，不运行 MCTS，不改变动作序列，也不让 TVA 重新参与搜索。审计对象包括状态门控 `46/50` 结果、固定 `alpha=0.60` 的单关 fallback、固定高权重 10 关对照，以及 phase060 单关动态规则结果。

每个 trace 前缀按保存的 push action 重放，逐步重构 exact grid state，并计算 GFT-TVA 特征 `P/M/A/K/D/Q/C`、`legal_push_count`、`viable_push_count`、容量债务、顺序债务、软死锁压力、显式 `V_SD`、TVA 预测值、`V_TVA-V_SD`、风险概率，以及根节点访问数最高动作与均值最高动作是否分离。当前审计默认分析前 `20` 步，原因是部分历史结果只保存了 `trace_limit=20`；该限制仅影响后段细节，不影响初始触发和早期误触发判断。

样本标签采用对照式定义。正例为状态门控失败、固定 `alpha=0.60` 解出的 `level_index=14,45,46`；负例为状态门控可解、固定高权重或 phase060 动态规则退化的 `5,6`；第 `0` 关作为高权重无效的循环停滞负例。被检验的 phase060 规则为：

```text
if P >= 20 or (P <= 10 and C >= 0.25):
    alpha = 0.60
else:
    alpha = posonly_state_gate
```

其中 `posonly_state_gate` 是当前状态门控候选使用的低风险正修正规则，即 `risk_probability<=0.30` 且 `V_TVA-V_SD>=0.25` 时使用 `alpha=0.25`，否则 `alpha=0`。

审计结论是该 phase060 规则不满足单次配置要求。它在初始状态覆盖全部正例 `14/45/46`，但也在负例 `5/6` 的可解轨迹中后段触发，并且对应的 phase060 单关运行均退化为 `deadlocked`。因此，当前 `49/50` 只能作为“状态门控与固定高权重互补”的组合策略信号；若要转化为单次动态 TVA，需要新增能够区分入口锁定修复窗口与安全低进度区的特征或排序目标。

后续修订将该负结果压缩为一个更窄的在线规则：运输阶段高权重 `alpha=0.60` 只允许在闭环前缀步触发，并可要求当前根状态本身也满足运输阶段谓词。实现上新增两个默认关闭保护项：`--tva-transport-phase-max-closed-loop-step` 限制可触发的闭环 push 前缀，`--tva-transport-phase-require-root-eligible` 要求根状态满足同一 `P/C` 条件后，才允许树内叶节点使用运输阶段权重。该设计的意图是保留 `14/45/46` 初始或早期入口锁定状态中的高权重 TVA 修复窗口，同时阻止 `5/6` 在可解轨迹后段的叶节点误触发高权重 TVA。

使用 `--tva-transport-phase-blend 0.60`、`P>=20 or (P<=10,C>=0.25)`、`--tva-transport-phase-max-closed-loop-step 5` 与 `--tva-transport-phase-require-root-eligible` 后，单次 50 关 heldout 运行得到 `49/50`。输出为 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json)，aggregate 为 `49/50`，success rate `0.98`，mean pushes solved `15.387755102040817`，状态分布 `max_steps=1,solved=49`，唯一失败为 `level_index=0:max_steps`。其中 `14/45/46` 分别以 `25/13/10` 步解出，反例 `5/6` 仍分别以 `24/21` 步解出。

该结果与失败后固定 `alpha=0.60` 重跑的组合策略不同：它是在一次预先指定配置下完成的单次 MCTS 评估，不读取 primary 失败结果，也不对失败关卡进行事后替换。因此，Sokoban 当前结果层级更新为：显式 GFT-SD 叶值 `44/50` 仍是已完整审计的正式主线；状态门控 GFT-TVA `46/50` 是较保守的单次候选；带前缀与根状态保护的动态运输阶段 TVA `49/50` 是当前最强单次探索候选；失败后固定高权重重跑的 `49/50` 仍是组合策略对照。由于该动态规则仍由当前失败与反例集诱导，正式主结果更新前仍需要独立门槛元数据复跑、风险审计和更外层 heldout 或种子复核。

#### 动态运输阶段 GFT-TVA 正式主线

上述候选随后完成独立门槛复核并提升为 Sokoban 当前正式主线。冻结规则保持不变：默认 `alpha=0`，低风险正修正状态 `alpha=0.25`；在闭环前缀 `step<=5` 内，若当前根状态满足 `P>=20` 或 `P<=10,C>=0.25`，且候选叶状态也满足同一运输阶段谓词，则使用 `alpha=0.60`。该规则只改变直接价值型 GFT-TVA 的叶值混合比例，不改变 exact grid stock-flow 转移，不用 TVA 风险头剪枝，也不进行失败后重跑。

正式复跑输出为 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json)。该运行启用 `--require-independent-gates`，SD 验证 gate 与 GFT-TVA 截断价值 gate 均为 `PASS`，`failed_checks=[]`，material passport 标记为 `VERIFIED`。aggregate 为 `49/50`，状态分布 `max_steps=1,solved=49`，唯一失败为 `level_index=0:max_steps`。该文件不包含 `primary_result`、`fallback_used_count` 或 `fallback_results` 字段，因而不是组合策略结果。

关键消融支持两个保护项的设计。无保护 phase060 在开发切片上为 `7/10`，新增 `5:deadlocked`、`6:deadlocked`；仅前缀保护仍为 `7/10`，新增 `5:deadlocked`、`6:no_action`；仅根状态保护恢复为 `9/10`；完整前缀加根状态保护也为 `9/10`，并使第 `5` 关解长回到 24 步。由此可见，根状态阶段保护是避免中后段叶节点误触发高 TVA 权重的主要机制，前缀保护提供额外保守边界。

外层切片复核使用 `medium/train/001.txt` 前 10 关。显式 GFT-SD 与状态门控 TVA 对照均为 `6/10`，动态运输阶段 TVA 为 `7/10`，未观察到系统性退化。基于正式 50 关复跑、独立门槛、关键消融和外层切片复核，Sokoban 当前结果层级更新为：动态运输阶段 GFT-TVA `49/50` 是正式主线；显式 GFT-SD `44/50` 保留为零 TVA 学习权重对照；状态门控 GFT-TVA `46/50` 保留为较保守的单次对照；失败后固定高权重重跑的 `49/50` 保留为组合策略对照。仍需注明，第 `0` 关循环停滞未解决，外层复核目前是 10 关切片而非第二个完整 50 关 heldout。

#### Boxoban `unfiltered/test` 外部基准口径

为与已发表 Boxoban 结果建立最小公开测试口径，动态运输阶段 GFT-TVA 正式主线进一步评估于 Boxoban `unfiltered/test/000.txt` 的完整 `1000` 关。方法配置不做调参，保持正式主线：exact-grid stock-flow 转移、push-level GFT-SD-MCTS、`horizon=20,budget=4096,max_steps=80,seed=0`、状态门控 GFT-TVA 默认 `alpha=0`、低风险正修正 `alpha=0.25`，以及闭环前缀 `step<=5` 内的运输阶段高权重 `alpha=0.60`。TVA 仍只进入叶值混合和窄根动作选择，不用于剪枝或转移替代。

本次外部基准采用 20 个不重叠 50 关分片并行运行，并按逐关随机流等价规则合并。脚本内部使用 `rng=random.Random(args.seed + level_idx*1009)`；因此对 `level_offset=offset` 的分片使用 `seed=offset*1009`，可使分片中每个局部关卡的随机流等于单进程全量 `seed=0` 时对应全局关卡的随机流。该分片只改变 wall-clock 调度，不改变逐关搜索配置。

合并结果为 `989/1000`，success rate `0.989`，失败分布为 `max_steps=7,deadlocked=4`。失败关卡为 `199`、`367`、`525`、`546`、`664`、`698`、`706`、`758`、`785`、`892`、`977`。输出文件为 [gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json)，`material_passport` 为 `VERIFIED`，20 个分片均通过 independent gates，且汇总文件没有 fallback 字段。

该结果可作为与 Boxoban `unfiltered test` 文献口径对比的正式外部基准结果。比较时需要同时报告方法差异：本方法是 push-level MCTS 加精确格点转移和 GFT/TVA 叶值评估，不是端到端 model-free agent；结果反映搜索器在标准测试关卡上的求解率，而非同一训练预算或同一动作频率下的 RL agent 表现。

#### Boxoban 训练/审计规模与文献规模对照

论文补充表采用如下比较口径。本文方法的可学习 TVA 部分使用 `medium/train/001.txt` 前 40 关在线轨迹训练：采集 `1378` 个 raw 状态，去重后为 `834` 个状态，其中 `train_samples=653`、`valid_samples=181`，实际进入 value 头训练的样本为 `114` 个；状态来源为 `online_root_state=794` 与 `online_final_state=40`。该 TVA 训练不使用 teacher：checkpoint contract 记录 `teacher_used_online=false`、`teacher_used_for_training_target=false`。

独立 TVA 截断价值审计使用 `medium/train/002.txt` 前 10 关，每关最多 12 个叶状态，共 `120` 个审计叶状态。离线 exact-state teacher 仅在审计中标注这些叶状态，预算为 `teacher_max_expansions=4000`；其中 `78` 个为 high-confidence solved 标签，`42` 个为 `teacher_budget_unsolved` 低置信标签。审计 gate 为 `PASS`：TVA leaf high-confidence MAE 为 `0.48948865033750666`，显式 GFT-SD leaf MAE 为 `0.5517001792097376`；TVA pairwise accuracy 为 `0.9617021276595744`，GFT-SD leaf pairwise accuracy 为 `0.948936170212766`。

文献对照分为两类。第一类是 Guez et al. (2019) 的 model-free planning DRC：其 Boxoban unfiltered large train set 为 `900k` 关，DRC(3,3) 报告训练至 `1e9` environment steps 后 `unfiltered` test 约 `99%` solved。第二类是 Orseau et al. (2018) 和 Orseau and Lelis (2021) 的 policy-guided 或 policy+heuristic search：前者在 NeurIPS 2018 论文中报告 LevinTS 在 1000 关测试上 `1000/1000`，总 expansions 约 `6.6e6`，加入 1% policy noise 后约 `5.0e6`；后者在 AAAI 2021 PHS 论文中使用 Boxoban 前 `50,000` 个训练问题、7 天 CPU bootstrap 训练和 2 天测试预算，PHS* 在 Sokoban 10x10 test 上 `1000/1000`，平均 expansions 约 `1522.1`。

因此，当前 `989/1000` 的论文表述应为：在标准 `unfiltered/test` 上接近已发表强搜索器和 model-free planning 方法的解题率，同时使用显式规则转移和远小于大规模 model-free RL 的 TVA 训练/审计状态数。不得写成严格 SOTA；也不得把 `989/1000` 与从零学习规则的 model-free RL 训练步数做直接等价的样本效率倍数相除。

#### XSokoban first 结构外推与失败解释

XSokoban first-40 不作为 Boxoban `unfiltered/test` 的同分布替代，而是作为更强的分布外压力测试。当前结构统计输出 [xsokoban_boxoban_structure_shift_analysis.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/xsokoban_boxoban_structure_shift_analysis.json) 表明，Boxoban `unfiltered/test/000.txt` 固定为 `10x10`、4 箱；XSokoban first-40 的中位尺寸为 `15x18`，中位箱数 `15.5`，最大箱数 `32`，中位地板数 `157`，中位 dead-square 地板占比 `0.515`，中位静态割点数 `14`。因此，XSokoban 不仅尺寸更大，而且目标更集中、走廊/房间瓶颈更多、每箱初始可推动自由度更低。

对当前失败的 XSokoban 第 1 关，应使用如下解释口径。该关为 `11x19`、6 箱、`139` 个地板格，初始合法 push 数只有 `3`，dead-square 地板占比为 `0.705`，静态割点数为 `12`。已有 `max_steps=1000` 结果仍未解出，最终只完成 1 个目标；前 20 步 trace 中出现同一箱子的重复推动和多个单 child root。因此，失败不能只归因为 `max_steps=80` 太短；即使放大到 `1000`，当前局部 push-level MCTS 仍没有进入稳定的多箱目标房间填充策略。

当前方法在 Boxoban 上表现较强的原因是：exact grid transition provider 与 GFT-SD 运输匹配在 4 箱、10x10 分布上提供了高质量局部进展信号；TVA 训练和独立审计也来自 Boxoban/medium 切片，覆盖了类似尺度的叶状态。XSokoban first 则越过了这些条件：精确箱-目标匹配和 Sinkhorn commitment 在多箱目标房间中变成高熵分配问题，ordering/capacity 读数虽能报警但不足以产生长程承诺，TVA 也没有在 11x19、6 箱及更多箱状态上校准。后续方法推进应优先考虑显式 box-goal/room-order commitment option、参考解锚定和 XSokoban 分布上的价值审计，而不是单纯继续增加闭环 `max_steps`。

#### XSokoban first 第 1 关的 SD 决策层失败机制

为避免将 XSokoban 失败仅解释为数据集结构差异，新增机制诊断脚本 [analyze_xsokoban_sd_failure_mechanism.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_xsokoban_sd_failure_mechanism.py)。该脚本只读取既有 `max_steps=1000` 失败日志中保存的前 20 步 trace，不重新运行长闭环实验；对每个已保存 root 重放状态，重新计算候选动作、GFT prior、GFT leaf value、TVA 预测、state-gated alpha 和 GFT 资源变化。

诊断结果表明，失败的直接机制是当前 SD/GFT/TVA 决策层的目标错配，而不只是搜索预算或步数限制。保存 trace 的 `20` 个 root 中有 `12` 个 `child_count=1`，被选动作访问占比中位数为 `1.0`，`b5:right` 出现 `14/20` 次。被选后继按当前一阶 leaf value 排名第一的步数为 `17/20`，说明 MCTS 大体服从当前价值函数；问题在于该价值函数将局部运输前缀评估得过好。前缀资源变化为 `P:97.35 -> 81.45`、`Q:0 -> 2`、`D:2.75 -> 2.877`、`C:0 -> 0.123`，即运输工作量下降，但目标房间顺序债务上升，承诺仍弱。

当前最佳配置的 root prior 权重为 `progress=1,goal=1,ordering=0.05,commitment=0.05,revisit=1`，而 `mobility/access/capacity/deadlock` 权重均为 `0`；GFT leaf 中 `deadlock` 权重也为 `0`，仅对 `capacity/order/commitment` 施加小修正。因此，模型虽然计算了 D/K/A/M 等资源，却没有把软死锁、容量压力和可达性下降转化为足够强的动作约束。TVA 方面，state-gated 默认 blend 为 `0`，trace 中被选后继的 TVA `feature_max_abs_z` 均值为 `20.37`、最大 `22.18`，但 OOD gate 关闭；TVA 因此没有承担分布外保护或房间顺序保护。该诊断将后续方法重点固定为显式 box-goal/room-slot commitment、目标房间填充顺序约束和 XSokoban 分布上的价值校准。

#### XSokoban 难题派生 Room-GFT 诊断线

为推进 XSokoban/Hard 难题修复，新增 `SokobanRoomGFTSDModel` 作为 `SokobanGFTSDModel` 的 OO 派生类。该派生类不覆盖原有精确转移、GFT 资源读数、GFT action prior 或 TVA 特征接口；因此，现有 Boxoban `989/1000` 正式主线的默认行为不受影响。新增接口只服务于 hard-instance 诊断和后续 planner 扩展。

静态 room 分解采用保守近似：首先将相邻 goal cell 聚合为 goal-room component；随后把与该 component 相邻的 floor cell 作为 entry/support cell；最后在 `goal cells + entries` 的局部图上计算 entry distance。若浅层 slot 位于 entry 到深层 slot 的最短路径上，则浅层 slot 被标记为该深层 slot 的 blocker，形成“深处 slot 应先填、入口侧 slot 后填”的偏序代理。该方法不能替代完整 Sokoban pattern database 或 domain macro，但足以对 XSokoban 第 1 关的 2x3 目标房间生成 right-column-before-left-column 的顺序代理。

动态诊断新增三组只读指标。第一，`room_state_summary(state)` 记录 `frontier_slot_count`、`filled_slot_count`、`q_settled`、`k_blocked`、`c_good`、`c_bad`、`uncommitted_entry`、`uncommitted_in_room` 和 `uncommitted_room_or_entry`。第二，`room_action_summary(state, action)` 为物理 push 打上 `CROSS_ROOM_BOUNDARY`、`ENTER_ROOM_ENTRY`、`UNCOMMITTED_ENTRY`、`FILL_FRONTIER_SLOT`、`FILL_NONFRONTIER_SLOT`、`INCREASE_ROOM_ORDER_DEBT`、`BLOCK_UNFILLED_SLOT` 等 impact code。第三，`room_semantic_options(state, action)` 计算当物理动作触及 goal room 时可展开的 frontier slot 选项数，作为 A3 语义 child 的代理指标。

A1 诊断规则为：当 TVA 特征 `feature_max_abs_z` 超过阈值且 `V_TVA>V_SD` 时，禁止 TVA 对叶值做正向修正；若 TVA 只给出更差估计，则暂不屏蔽。当前脚本默认阈值为 `3.0`，该阈值用于发现 XSokoban trace 明显越过 Boxoban TVA 训练/审计分布，而不是作为已调优的正式 planner 超参。A2 诊断仅报告 room-order 信号，不改变动作选择；真正的修复将放在 A3，即把 `(box,direction,slot)` 承诺作为派生 hard-instance MCTS 节点的一部分。

#### A3 语义承诺 MCTS

A3 新增 `RoomSemanticGFTSDMCTSPlanner`，仅用于 `SokobanRoomGFTSDModel` 派生难题线。该 planner 不复用原 `children[action.flat_index]` 的节点结构，因为同一个物理 push 必须允许多个语义后继。A3 的 action 定义为 `RoomSemanticAction(physical_action, RoomSlotOption)`，其中 `RoomSlotOption` 可为 `none`、`slot` 或 `uncommitted`。默认正式诊断不展开 `uncommitted` 选项；若没有可用 frontier slot，则保留 `uncommitted` 作为兜底。

A3 的节点状态为 `(SokobanState, RoomCommitmentState)`。`RoomCommitmentState` 由若干 `RoomSlotCommitment(box_cell, room_id, slot_id, slot_cell)` 构成，表示某个箱子当前位置已承诺给某个 room slot。由于现有 Sokoban state 已经把箱子身份排序消去了，A3 暂以箱子当前位置作为承诺载体；每次物理 push 后，如果被推动箱子已有承诺，则承诺迁移到新箱子位置；如果该 semantic action 选择了新 slot，则建立或替换该箱子的承诺。MCTS transposition key 为 `gft_canonical_key(state) + commitment_state.key`，因此物理局面相同但 slot commitment 不同的节点不会合并。

语义 child 的触发分两类。第一，物理 push 直接触及 goal room、entry 或 slot。第二，物理 push 的 source 或 target 到任一 room entry 的静态 floor 距离不超过 `semantic_commitment_boundary_distance`，默认值为 `6`。第二类对应专家建议中的 commitment boundary / point-of-no-return，用于在长走廊进入目标房间前提前展开 slot 承诺。

A3 prior 在原 GFT action prior 上叠加 room 语义项：slot entry depth、commitment gain 和 frontier fill 为正项；uncommitted entry、non-frontier fill、room-order debt 增量和 blocked-slot 增量为负项。A3 leaf 在显式 GFT-SD discounted progress 基础上叠加 room commitment bonus，并惩罚 uncommitted room/entry、`q_settled`、`k_blocked` 和 non-frontier commitment。该版本仍是机制最小实现，不使用 TVA，也不声明为正式 XSokoban 求解配置。

#### A4 槽位顺序约束进度

A4 继续只作用于 `room_gft_*` 派生难题线。模型接口 `room_slot_progress_summary(state, commitment_state)` 将运输进展从“任意箱子到任意 goal 的匹配距离”改为“已承诺箱子到其承诺 slot、未承诺箱子到当前 frontier slot”的槽位顺序代理。已承诺部分使用静态反向推箱距离；未承诺部分按 room 内深槽优先的顺序，将可用箱子贪心分配给未填且未被承诺的 frontier slots。不可达距离不记为无穷大，而是使用 `height+width` 作为有限惩罚，以便 MCTS prior 能比较候选动作。

A4 barrier 由以下原始项组成：`q_settled`、`k_blocked`、`uncommitted_room_or_entry`、`nonfrontier_commitment`、`committed_infeasible` 和 `frontier_assignment_infeasible`。默认 barrier cost 为：

```text
6*q_settled_effective
+ 6*k_blocked_effective
+ 2*uncommitted_room_or_entry
+ 3*nonfrontier_commitment
+ (height+width)*committed_infeasible
+ 0.5*(height+width)*frontier_assignment_infeasible
```

其中 `q_settled_effective` 和 `k_blocked_effective` 会扣除 `committed_pass_through_blocker`。该例外用于处理深槽必须经过浅槽的情况：若已承诺箱子进入的是目标深槽的浅层 blocker，且到承诺 slot 的反向推箱距离继续下降，则该中间态视为 pass-through，而不是 settled 错填。

规划器 `RoomSlotProgressGFTSDMCTSPlanner` 继承 A3 的 `RoomSemanticGFTSDMCTSPlanner`，保留 `(physical push, slot option)` 语义 child 和 commitment state 机制。A4 prior 在 A3 room prior 上叠加：

```text
0.55 * total_transport_delta
+ 1.25 * committed_transport_delta
+ 0.35 * frontier_transport_delta
+ 0.80 * barrier_delta
+ 8.00 * committed_pass_through_increase
- 8.00 * committed_infeasible_increase
```

其中 delta 均按“父状态 cost - 子状态 cost”计算，正值表示进展或 barrier 降低。A4 leaf 在 A3 leaf 基础上加入 `0.03*total_transport_cost`、`0.08*barrier_cost` 和 `3.0*committed_infeasible` 的扣分，并使用有效 `q_settled/k_blocked`，避免正确 pass-through 中间态被过早压低。

当前 A4 只通过机制诊断，不作为正式求解配置。旧 XSokoban 第 1 关失败 trace 中，A4 在入房窗口 step `18` 将旧动作 `b5:down` 从旧 GFT rank `3` 降为 A4 rank `4`，并首选 `b5:right|slot3@[7,17]`。短闭环 smoke 仍为 `max_steps`，40 步后 `boxes_on_goals=0`，说明后续仍需 corridor contraction、macro/forced-corridor option 或重复承诺状态惩罚。

#### A5 承诺单调 loop guard

A5 继续只作用于 `room_gft_*` 派生难题线。规划器 `RoomLoopGuardGFTSDMCTSPlanner` 继承 A4 的 `RoomSlotProgressGFTSDMCTSPlanner`，在不剪枝合法动作的前提下，为已承诺箱子的非单调移动加入软惩罚。A5 当前不修改 `SokobanRoomGFTSDModel` 的静态 room 分解，也不改变 exact grid-flow 转移。

A5 prior 在 A4 prior 之上加入：

```text
- 8.0 * committed_distance_regress
- 6.0 * direct_reverse_previous_push
- 5.0 * repeated_commitment_state
```

其中 `committed_distance_regress` 指被推动箱子已有 slot commitment 时，子状态中该箱子到承诺 slot 的反向推箱距离相对父状态增加的幅度；`direct_reverse_previous_push` 指当前 push 直接撤销上一步推动；`repeated_commitment_state` 指子节点的 `exact canonical state + room commitment key` 已出现在近期闭环历史中。A5 leaf 还对近期重复 commitment state 加 `1.0` 的额外扣分。

A5 额外加入根选择保护。树内模拟仍使用 PUCT 和 soft prior；根节点出手时，若访问数最高的 child 触发 loop penalty，而存在未触发 loop penalty 的 child，则在未触发 child 中按原 `select_by` 规则选择。这一保护用于避免“被 prior 惩罚但仍因访问数最高而出手”的情形。

当前 A5 是负结果而非求解配置。在同一短闭环 smoke 中，A4 的事后 loop 统计为 `15` 次承诺距离回退和 `3` 次直接反向推动，40 步后 `boxes_on_goals=0`；A5 将这两类被选行为降为 `0`，并完成 1 个目标，但在第 22 步进入 `no_action`，最终 `boxes_on_goals=1`。因此，A5 证明 loop guard 可消除 A4 的走廊回退，却不能独立解决后续多箱 room-slot 调度。下一阶段应转向 corridor contraction、forced-corridor macro 或参考解锚定，而不是继续只调 A5 软惩罚权重。

#### A6 forced-corridor macro

A6 继续只作用于 `room_gft_*` 派生难题线。规划器 `RoomCorridorMacroGFTSDMCTSPlanner` 继承 A5 的 `RoomLoopGuardGFTSDMCTSPlanner`。其搜索树内部仍使用 A5 的 single-push semantic MCTS；新增机制只发生在根节点出手之后：若被选动作推动的是一个已承诺箱子，且该箱子到承诺 slot 存在唯一合法、非死锁、反向推箱距离严格下降的后续 push，则把这些连续 push 作为 forced-corridor macro 一次性执行。该机制用于压缩长走廊中的确定性单调推进，不改变 exact grid-flow 转移，也不回写 Boxoban 主线。

当前宏触发条件是保守的。默认要求后续 committed-box monotone push 唯一，最多追加 `8` 个 extra pushes，且宏总长度至少为 `2` 才记为触发。宏执行过程中，commitment state 随箱子位置迁移；每个宏内 push 仍逐步写入 visited key 和 recent commitment key，因此宏不是跳过状态审计，而是减少闭环重新规划次数。若宏中途到达承诺 slot、出现无单调 push、出现多个可选单调 push、丢失承诺或触及 push budget，则停止宏。

当前 A6 也是负/部分结果。在同一短闭环 smoke 中，A6 于 step `9` 触发一次 corridor macro，把 `b5:right` 从 `[7,8]` 连续推进到承诺 slot `[7,17]`，额外压缩 `8` 个 push。相对 A5，A6 将闭环决策数从 `22` 降为 `14`，但实际执行 push 数仍为 `22`，最终仍为 `no_action`、`boxes_on_goals=1`、`final_slot_schedule_cost=46.0`。因此，A6 证明 forced-corridor macro 能压缩已承诺单箱走廊推进，但不能解决宏后的多箱 room-slot 调度和可达性瓶颈。下一步应转向参考解或 teacher trace 锚定的多箱 slot schedule，而不是继续只扩大单箱 corridor macro。

#### A7 teacher/reference trace anchor

A7 继续只作用于 `room_gft_*` 派生难题线。规划器 `RoomTraceAnchoredGFTSDMCTSPlanner` 继承 A6 的 `RoomCorridorMacroGFTSDMCTSPlanner`，不修改 exact grid-flow 转移、room 静态分解或 Boxoban 正式主线。A7 的新增输入是一个离线生成的参考轨迹映射：对命中参考轨迹的 root state，记录下一步参考物理动作、该动作推动箱子的 source/target，以及沿参考解追踪得到的未来 goal cell。

A7 的 prior 在 A6 之上加入两类 trace-anchor 项。若候选物理动作的 `flat_index` 等于参考动作，则加 `30.0`；若该候选同时选择了 slot option，且该 slot cell 等于该箱子沿参考解最终到达的 goal cell，则再加 `10.0`；若参考动作匹配但 slot 不匹配，则扣 `2.0`。根节点出手时，若当前 state 命中参考轨迹且存在参考物理动作对应的 child，则在这些 child 中优先选择未来 goal slot 匹配者，再按访问数、均值和 prior score 打破并列。A7 还要求 corridor macro 与参考轨迹逐 push 匹配；若宏内后续动作偏离参考轨迹，则将宏截断到已匹配前缀。

参考轨迹由 A7 诊断脚本内的本地 exact-state teacher 生成。该 teacher 复用既有 TVA teacher 的 best-first 口径，但增加父指针以重建 action path；启发式为 `matching_lower_bound - 0.25*boxes_on_goals - 0.01*viable_push_count`，并对 legal push、deadlock 和 heuristic 做局部缓存。XSokoban 第 1 关在 `teacher_max_expansions=100000` 下被解出，参考解长度为 `97` 个 push，实际展开 `7248` 个状态，生成 `69620` 个后继，剪去 `4752` 个 deadlock 后继。

当前 A7 是机制定位结果。在同一 A6 基线对照下，A6 仍为 `no_action`、执行 `22` 个 push、`boxes_on_goals=1`；A7 trace-anchor smoke 在 `a7_budget=64,horizon=8,a7_max_pushes=120` 下为 `solved`，执行 `97` 个 push、`63` 个闭环决策、触发 `6` 次 corridor macro，最终 `boxes_on_goals=6`、slot schedule cost 为 `0.0`。63 个闭环决策均命中参考状态并选择参考动作，最终状态命中参考终态。该结果说明，当前 exact transition、room-slot commitment 和 corridor macro 能承载可行解；A5/A6 的主要失败点是自主多箱调度和槽位顺序选择，而不是单步转移或 `max_steps` 本身。A7 不得报告为无 teacher 的 XSokoban 求解率。

#### Teacher 轨迹驱动的 SD 缺失机制分析

在 A7 之后，新增 teacher trace gap 诊断。该诊断不把 teacher 轨迹注入 planner 出手，而是在 teacher 轨迹的每一个 root state 上重新计算当前 Room-GFT/A6 的候选动作、semantic slot option、prior score、slot progress delta、barrier delta 和少量 MCTS 采样选择，用于回答“正确 teacher 动作在当前 SD 评分中为何没有被稳定选中”。

默认诊断范围为 XSokoban first 前 3 关，teacher 预算为每关 `100000` expansions 或 `90s` wall-clock。结果显示，第 1 关 teacher 可解，解长 `97` push、展开 `7248` 个状态；第 2 关为 10 箱，第 3 关为 11 箱，均在 `90s` teacher 时间预算内未解出。因此，当前多关结论仍以第 1 关完整轨迹为主，第 2/3 关提供的是 teacher 轨迹库建设的预算边界，而不是 Room-GFT 缺失不存在的证据。

对第 1 关完整 teacher 轨迹，A6 prior 层将 teacher 物理动作排第 1 的比例为 `0.598`，排入前 3 的比例为 `0.856`，最大 rank 为 `13`；按每 10 步采样的 MCTS root 选择，teacher 动作匹配比例为 `0.600`。gap tags 中，`setup_or_release_underrewarded=20`、`teacher_low_prior_rank=14`、`short_transport_competes=16`、`top_action_room_order_risk=6`。teacher 轨迹还包含 `9` 段长度至少为 3 的同向 corridor run，最长为 `10` push。

据此，将当前 SD 缺失归纳为五类。第一，缺少可自主生成的 `room_fill_plan_state`：A7 的成功依赖 future-goal/slot trace，而 A6 只有单箱 commitment，没有全局 room fill phase 或 slot queue。第二，缺少 `box_release_and_setup_value`：20 个 teacher 步被标记为准备或释放动作即时收益不足，说明当前 value/prior 仍偏向即时运输下降。第三，A6 corridor macro 需要 schedule guard：宏执行层可用，但应由自主 fill plan 决定何时触发、哪个箱子进入哪个 slot。第四，需要 planner choice calibration：teacher 动作并非总在当前 prior/MCTS 排名前列，应将低 rank teacher 状态转为价值/先验审计样本。第五，需要显式处理长程运输与槽位调度 tradeoff：`short_transport_competes=16` 表明局部运输进展仍会与长期 room-order 可行性冲突。

随后为争取获得前三关完整轨迹，增加 fast teacher 预算与实现。fast 配置使用 Hungarian matching 计算箱子-目标匹配下界、Tier-A deadlock 剪枝、关闭 viable push 子项、heuristic weight `5.0`，每关预算提高到 `1000000` expansions 或 `600s`。该配置不保证最优解，只用于尝试生成可行 teacher 轨迹。结果为：第 1 关解出，解长 `109` push，展开 `822845` 个状态；第 2 关和第 3 关仍未解出，分别在 `600s` 内展开 `533130` 和 `784388` 个状态。fast-budget 结果说明，当前可用综合证据仍主要来自第 1 关完整轨迹；第 2、3 关未解出不能作为 SD 机制充分性的证据，只说明 10/11 箱 XSokoban teacher 轨迹库本身需要更强搜索、更多预算或额外领域启发式。

fast-budget 第 1 关轨迹与原 teacher 轨迹给出的缺失项方向一致：A6 prior 下 teacher 动作 rank-1 比例为 `0.697`，rank<=3 比例为 `0.899`，仍有 `setup_or_release_underrewarded=16`、`teacher_low_prior_rank=11`、`short_transport_competes=14`，并出现 `11` 段长度至少为 3 的同向 corridor run。因此，“SD 所缺信息”的表述应补充为：不仅需要 room fill phase/slot queue，还需要把箱子释放、准备动作、通道保持和长走廊宏触发边界作为可更新的 SD 状态或反馈项，而不是只在 MCTS/TVA 末端调权。

#### A8 teacher-trace-distilled fill-plan SD

A8 继续只作用于 `room_gft_*` 派生难题线。模型侧新增 `SokobanRoomFillPlanGFTSDModel`，继承 `SokobanRoomGFTSDModel`，不改变 exact grid-flow 转移、GFT 基础读数或 Boxoban 主线。A8 将 XSokoban 第 1 关 teacher 轨迹中反复出现的结构压缩为当前状态可计算的 fill-plan 摘要，而不是保存或在线查询 teacher 动作表。

A8 fill-plan 摘要包括：目标房间 lane queue、当前 fill phase、active slot、channel lane、active assigned box、active box 可推数、active box 单调推进数、active box blocked count、channel staging box count、early channel occupancy 和 plan barrier cost。第 1 关 teacher 轨迹显示，浅槽多次作为通道和中转，而不是一开始就作为最终填充目标。因此 A8 不采用简单“所有深槽完成前禁止浅槽占用”的硬约束，而是将 channel/staging 作为状态项和软反馈项。

规划侧新增 `RoomFillPlanGFTSDMCTSPlanner`，继承 A6 的 `RoomCorridorMacroGFTSDMCTSPlanner`，不继承 A7 的 reference action anchor。A8 prior 在 A6 prior 之上加入 fill-plan barrier reduction、active commitment bonus、channel staging bonus、early channel penalty、active transport delta、active box release delta 和 active monotone push delta；leaf 同步扣除 fill-plan barrier、active transport cost、active blocked count，并保留 A6 的 corridor macro 执行层。

默认 A8 smoke 使用 `budget=256,horizon=8,max_steps=120,boundary_distance=6`。相对 A6，A8 从 `no_action, executed_pushes=22, boxes_on_goals=1, final_slot_schedule_cost=46.0` 改为 `no_action, executed_pushes=48, boxes_on_goals=2, final_slot_schedule_cost=43.0`，触发 `4` 次 corridor macro，额外压缩 `16` 个 push。较大预算复核 `budget=1024,horizon=12,max_steps=160` 未改变 A8 结局，仍为 `boxes_on_goals=2`、`no_action`。因此，A8 是 SD 补强后的部分正结果：它证明 teacher 轨迹归纳出的 fill-plan/release 信号能让无参考动作 planner 多完成一个深槽，但仍缺少第三个 deep slot 的更细释放队列、局部可达性保护和多箱协同调度。

A8 不能报告为 XSokoban 第 1 关求解成功，也不能与 A7 reference-conditioned solved 混写。其主要方法含义是：SD 端确实缺少可更新的 room fill phase、active slot、release/setup 和 channel preservation 状态；补上第一层后性能有方向性改善，但仍需要下一版继续建模“active box 被其他箱子阻塞时，哪些非 active push 是释放动作”的因果状态。TVA 的后续配合应以 A8 暴露的 active-box blocked 状态、低 fill-plan score 状态和 teacher 低 rank 状态为审计样本，而不是直接模仿 teacher 动作。

#### A9 active-box release-plan SD

A9 继续只作用于 `room_gft_*` 派生难题线。模型侧新增 `SokobanRoomReleasePlanGFTSDModel`，继承 `SokobanRoomFillPlanGFTSDModel`，不改变 exact grid-flow 转移、GFT 基础读数、A8 fill-plan 规则或 Boxoban 主线。A9 将 A8 末端的粗粒度 `active_box_blocked_count` 展开为当前状态可计算的 release-plan 摘要。

A9 release-plan 摘要围绕 active assigned box 枚举所有能降低 active slot 反向推箱距离的候选单调推进。对每个候选推进，记录推动方向、箱子目标格、所需玩家站位格、站位格是否被箱子占据、站位格是否可由玩家到达、目标格是否被占据、子状态是否触发 Tier-A deadlock，以及该候选是否已成为可执行的 monotone push。聚合项包括 `active_release_candidate_push_count`、`active_release_need_count`、`active_release_stand_blocker_count`、`active_release_unreachable_stand_count`、`active_release_monotone_available_count`、`release_blocker_box_count` 和 `release_barrier_cost`。

规划侧新增 `RoomReleasePlanGFTSDMCTSPlanner`，继承 A8 的 `RoomFillPlanGFTSDMCTSPlanner`。A9 prior 在 A8 prior 之上加入 release barrier reduction、release need reduction、stand blocker reduction、target blocker reduction、unreachable stand reduction、available monotone push increase 和 blocker-box reduction；leaf 同步扣除 release barrier、release need 和 stand blocker，并奖励已可用的 active monotone push。A9 不保存 teacher 状态表，也不读取 teacher 动作表；teacher 轨迹仅用于指出应补入哪些状态变量。

默认 A9 smoke 使用 `budget=256,horizon=8,max_steps=120,boundary_distance=6`。相对同轮 A8 对照，A9 首个决策差异发生在 step `3`；A9 结局仍为 `no_action`、`boxes_on_goals=2`，执行 `44` 个 push、`28` 个闭环决策，final slot cost 为 `42.0`、final fill barrier 为 `4.65`、final release barrier 为 `8.0`。较大预算复核 `budget=1024,horizon=12,max_steps=160` 仍未解出，A9 为 `no_action`、执行 `50` 个 push、`boxes_on_goals=2`、final slot cost `43.0`、release barrier `8.0`。

A9 的主要增益是诊断可解释性而非求解率改进。两轮 A9 末端均定位到同一局部缺口：active slot 为 `(7,17)`，active assigned box 为 `(6,5)`，候选单调推进为 `down` 到 `(7,5)`，所需站位格 `(5,5)` 被另一箱占据，因此 `active_release_need_count=1`、`active_release_stand_blocker_count=1`、`active_release_monotone_available_count=0`。这说明 SD 端已经能显式读出“需要释放哪个站位格”，但当前 MCTS/TVA 消费端仍没有足够稳定地提前保持或清空该站位格。后续不应把 A9 报告为求解成功；更合适的下一步是围绕这些 release-plan 状态构造 TVA 审计样本或更强的前瞻式 schedule guard。

#### A10 lookahead access-plan SD

A10 继续只作用于 `room_gft_*` 派生难题线。模型侧新增 `SokobanRoomAccessPlanGFTSDModel`，继承 `SokobanRoomReleasePlanGFTSDModel`，不改变 exact grid-flow 转移、GFT 基础读数、A8 fill-plan 规则、A9 release-plan 规则或 Boxoban 主线。A10 的新增状态是 lookahead access-plan：对后续若干个 open deep slot 也枚举候选单调推进和所需玩家站位格，而不只看当前 active slot。

初始实现使用 `lookahead_slots=3` 时，前瞻信号过强，planner 为避免 future access blocker 进入长循环，默认 120 push 仅完成 1 个目标。因此正式 A10 诊断将默认前瞻深度收窄为 `lookahead_slots=2`，并将 scoring 重点限制在“未来站位格被箱子占据”的强信号上；一般的 stand unreachable 仍写入摘要，但不作为强 prior/leaf 项主导出手。

A10 planner `RoomAccessPlanGFTSDMCTSPlanner` 继承 A9 的 `RoomReleasePlanGFTSDMCTSPlanner`。其 prior 在 A9 基础上加入 future stand-cell blocker reduction 项，并保留根选择保护：若候选动作会新增 future stand-cell box blocker，而存在不新增此类 blocker 的候选，则优先在后者中选择。leaf 只给 future stand blocker 轻量扣分，避免像 `lookahead_slots=3` 那样因一般可达性项过强而偏离 room fill plan。

默认 `budget=256,horizon=8,smoke_steps=120,lookahead_slots=2` 下，A10 相对 A9 从 `no_action, boxes_on_goals=2, executed_pushes=44, final_slot_schedule_cost=42.0` 改为 `max_steps, boxes_on_goals=5, executed_pushes=120, final_slot_schedule_cost=1.0`。这说明前瞻 access-plan 已明显改变调度，并避免了 A9 在第三个 deep slot 前的站位格阻塞，但 120 push 上限下仍未完成最后一个目标。

延长步数复核使用相同模型和 planner，仅将 `smoke_steps` 改为 `200`。该配置解出 XSokoban first 第 1 关：`status=solved`，执行 `127` 个 push、`90` 个闭环决策，触发 `8` 次 corridor macro，最终 `boxes_on_goals=6`，final access barrier 为 `0.0`，final release barrier 为 `0.0`。该结果仍不是 A7 式 reference-conditioned solver；在线求解过程中不读取 teacher 动作表。A10 的方法含义是：A9 的主要缺口确实在 active slot 视野过短，必须把未来 deep slot 的站位格保护前移到 SD 状态和 MCTS 消费端；但该修复目前只在 XSokoban 第 1 关形成单关正结果，尚不能外推为 XSokoban first-40 或 hard 集的求解率提升。

为检验 A10 规则是否绑定第 1 关，诊断脚本加入 `--level-offset` 参数，并在 XSokoban first 第 2 关运行相同 A10 配置。第二关复核使用 `budget=256,horizon=8,smoke_steps=200,lookahead_slots=2,level_offset=1`。A9 为 `no_action`、执行 `34` push、`boxes_on_goals=1`；A10 为 `no_action`、执行 `57` push、`53` 个闭环决策、`boxes_on_goals=1`，A9/A10 首个决策差异为 step `13`，最终 slot schedule cost 均为 `52`。该结果说明 A10 的状态规则和 planner 可机械外推到第二关，但不能在当前配置下解出第二关。

第二关终态的 Tier-A 诊断为 `tier_a_deadlocked=false`、`no_legal_pushes=false`，同时仍有 `final_access_barrier_cost=8`、`final_future_access_barrier_cost=4`、`final_access_stand_blocker_count=1` 和 `final_future_access_stand_blocker_count=1`。因此，第二关失败不应解释为 `smoke_steps=200` 不足，而应解释为当前多箱 schedule、release/access preservation 和根选择机制仍不足。后续若推进 A11，应优先针对多箱释放队列、跨 slot 的释放宏动作和 TVA 前缀校准，而不是只报告更大的步数预算。

#### A11 multi-box schedule-plan SD

A11 继续只作用于 `room_gft_*` 派生难题线。模型侧新增 `SokobanRoomSchedulePlanGFTSDModel`，继承 A10 的 `SokobanRoomAccessPlanGFTSDModel`，不改变 exact grid-flow 转移、GFT 基础读数、A8 fill-plan、A9 release-plan、A10 access-plan 或 Boxoban 主线。A11 的新增状态是 multi-box release dependency graph：从 A10 的 access assignments 中抽取当前和未来 slot 所需的玩家站位格、候选单调推进、阻塞箱、不可达站位格、target blocker、潜在 release action 和 destructive action。

该设计遵循模块分工：调度知识表示在 SD 侧生成，MCTS 只消费这些显式摘要。规划侧新增 `RoomSchedulePlanGFTSDMCTSPlanner`，继承 A10 planner。A11 prior/leaf 读取 `schedule_barrier_cost`、`dependency_count`、`future_dependency_count`、`blocker_box_count`、`critical_stand_box_count`、`release_action_count` 和 `destructive_action_count`；root guard 进一步禁止会立即进入当前或未来关键站位格的候选动作。为降低计算开销，A11 planner 内部加入 per-planner summary cache，缓存 `fill/release/access/schedule` 摘要。

首个 raw A11 使用 `lookahead_slots=3`，结果为负：XSokoban first 第 2 关上，A11 首步从 A10 的 `b3:right` 改为 `b7:right`，最终 `no_action`、执行 `14` push、`boxes_on_goals=0`、`no_legal_pushes=true`。该负结果暴露了一个消费端错误：如果只奖励 destructive action count 下降，planner 会把“已经执行破坏性动作后该动作不再可用”误判为收益。因此，raw A11 不作为后续默认配置。

修正后的 guarded A11 将 `schedule_lookahead_slots` 收窄为 `2`，与 A10 对齐；新增 immediate critical/future stand-cell entry penalty；不再正向奖励 destructive action count 下降。该版本在同一第二关 `budget=256,horizon=8,smoke_steps=200` 下为 `max_steps`，执行 `200` push、`boxes_on_goals=0`、final slot cost `48`、final schedule barrier `30`、dependency count `2`、release action count `0`。相对 A10，它避免了早期 `no_action`，但没有完成目标，并在后半段出现二箱循环。因此，A11 目前是表示补强和负结果诊断，不是求解能力改进。

当前 A11 的方法含义是：多箱调度缺口确实应在 SD 侧显式表示，但 MCTS 消费端不能只依赖静态 dependency/barrier delta。后续更合理的方向是在 A11 graph 上构造 release macro、dependency-stall loop guard，或将 schedule-stall 前缀转为 TVA 审计/校准样本；不应把 teacher 动作表在线注入，也不应把 A11 报告为 XSokoban 第 2 关求解成功。

#### A12 dependency-stall guard

A12 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleStallGuardGFTSDMCTSPlanner`，继承 A11 的 `RoomSchedulePlanGFTSDMCTSPlanner`。该版本不改变 A11 的 `SokobanRoomSchedulePlanGFTSDModel`，也不修改 exact grid-flow 转移、A8-A11 规则或 Boxoban 主线。A12 的目标是检验：在 SD 侧已有 multi-box dependency graph 的情况下，消费端加入依赖停滞和重复调度签名保护，是否足以打破 A11 第二关中的二箱循环。

A12 在候选动作上记录 `room_schedule_stall_score` 与 `room_schedule_stall_summary`。该摘要比较父子状态的 `slot_schedule_cost`、`schedule_barrier_cost`、`dependency_count`、`future_dependency_count` 和 `release_action_count`，并结合最近移动边、反向边、重复 commitment state 与重复 schedule signature 判定是否为 dependency-stall 行为。MCTS prior 在 A11 prior 之上加入软惩罚；根选择阶段若最高候选同时表现为 stalled 和重复/反向调度，则优先选择非停滞候选。该机制仍只消费 SD 摘要，不保存 teacher 状态表，也不读取 teacher 动作表。

XSokoban first 第 2 关默认 A12 诊断使用 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recent_edge_window=12`。A11 guarded 对照为 `max_steps`、执行 `200` push、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`；A12 为 `no_action`、执行 `6` push、`3` 个闭环决策、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `40.5`、dependency count `2`、future dependency count `1`、release action count `0`。A12 前缀为 `b6:right,b6:right,b7:right`，与 A11 首个决策即不同。

当前 A12 的方法含义是：dependency-stall guard 能阻断 A11 的长循环，但不能独立生成可行释放路径。A12 终态仍存在当前站位格不可达和未来 target 被当前箱占据的问题，且没有 release action 可供 planner 消费。因此，下一阶段不应继续只增加 loop penalty 或普通 MCTS budget，而应在 schedule graph 上补充 release/access recovery macro、玩家可达域恢复项，或将这些 no-action 前缀纳入 TVA 审计/校准。

#### A13 access-recovery guard

A13 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleAccessRecoveryGFTSDMCTSPlanner`，继承 A12 的 `RoomScheduleStallGuardGFTSDMCTSPlanner`。该版本仍使用 A11 的 `SokobanRoomSchedulePlanGFTSDModel`，不改变 exact grid-flow 转移、A8-A12 规则或 Boxoban 主线。A13 的目标是检验：在 A12 已能识别 dependency-stall 后，额外保护玩家可达域和非阻塞合法推箱余量，是否能避免第 2 关过早进入 `no_action`。

A13 在候选动作上记录 `room_access_recovery_score` 与 `room_access_recovery_summary`。该摘要读取 A11/A12 的 schedule graph，并计算父子状态的非阻塞合法推箱数、legal collapse、low-legal margin、`no_legal_pushes`、target blocker delta、unreachable stand delta、release action delta 和 dependency total delta。MCTS prior 在 A12 prior 之上加入 access-recovery 分数；根选择阶段若最高候选在 unresolved dependency 下会进入 `no_legal_pushes` 或低合法余量状态，则优先选择不触发该风险的候选。leaf value 也对 unresolved dependency 下的低合法余量状态加入轻量扣分。该机制仍只消费 SD 摘要和 exact legal-push 读数，不保存或查询 teacher 动作表。

XSokoban first 第 2 关默认 A13 诊断使用 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recent_edge_window=12,recovery_min_legal_pushes=2`。A12 对照为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、final schedule barrier `40.5`；A13 为 `max_steps`、执行 `200` push、`182` 个闭环决策、`boxes_on_goals=0`、final slot cost `29`、final schedule barrier `87.5`、dependency count `2`、future dependency count `1`、target blocker count `2`、release action count `0`。A12/A13 首个决策差异发生在 step `1`，说明 A13 确实拦截了 A12 中导致早期低余量状态的第二步。

当前 A13 的方法含义是：access-recovery margin 是有效诊断信号，但还不是可执行的 recovery macro。它避免了 A12 的早期 `no_action`，并暴露后半段低合法余量循环；但由于没有显式生成“移动哪个箱子以恢复哪个站位格或 target”的宏动作，最终仍无法完成目标。后续应在 A11/A13 的 dependency graph 上实现短程 release/access recovery macro，并用 target blocker、stand reachability 和 legal-margin 作为宏执行的进入与停止条件。

#### A14 bounded recovery macro

A14 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleRecoveryMacroGFTSDMCTSPlanner`，继承 A13 的 `RoomScheduleAccessRecoveryGFTSDMCTSPlanner`。该版本仍使用 A11 的 `SokobanRoomSchedulePlanGFTSDModel`，不改变 exact grid-flow 转移、A8-A13 规则或 Boxoban 主线。A14 的目标是检验：从 A13 已识别的低合法余量和 target blocker 诊断，是否可以进入一个保守的短程恢复宏动作。

A14 的 recovery macro 在 MCTS 选出根动作并应用原 corridor macro 后触发。宏搜索只枚举合法且非 Tier-A 死锁的 push，默认最多追加 `4` 个 push、最多展开 `400` 个候选节点。宏评分比较宏入口状态和候选终态的非阻塞合法推箱数、target blocker count、unreachable stand count、release action count、dependency total 和 slot schedule cost；若最优短序列得分低于 `2.0`，则不执行恢复宏。宏内后续动作使用 `RoomSlotOption(kind="none")` 迁移既有 commitment，不新增 teacher/reference 约束。

XSokoban first 第 2 关默认 A14 诊断使用 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2,recovery_macro_max_extra_pushes=4,recovery_macro_max_nodes=400,recovery_macro_score_threshold=2.0`。A13 对照为 `max_steps`、执行 `200` push、`182` 个闭环决策、`boxes_on_goals=0`、final slot cost `29`、dependency count `2`、target blocker count `2`；A14 为 `max_steps`、执行 `200` push、`73` 个闭环决策、`boxes_on_goals=0`、final slot cost `33`、final schedule barrier `72.0`、dependency count `3`、future dependency count `2`、target blocker count `1`、release action count `0`。A14 recovery macro 触发 `48` 次，累计追加 `116` 个 push。

当前 A14 的方法含义是：可执行恢复宏机制已经打通，但宏目标过宽。它显著压缩闭环决策次数并降低 target blocker，却没有完成任何目标，且 dependency 总数增加。这说明后续不应只调宏搜索节点数或得分阈值，而应将宏绑定到具体 schedule dependency：针对某个 `stand_cell_unreachable_by_player` 或 `target_occupied_by_box` row 生成恢复序列，并以该 row 的解除、站位格可达、target blocker 移开或 release action 出现作为首要停止条件。

#### A15 dependency-bound recovery macro

A15 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner`，继承 A14 的 `RoomScheduleRecoveryMacroGFTSDMCTSPlanner`。该版本不改变 A11 的 `SokobanRoomSchedulePlanGFTSDModel`、exact grid-flow 转移、A8-A14 规则或 Boxoban 主线。A15 的目标是检验：在 A14 已经具备可执行短宏后，若宏搜索绑定到一个具体 schedule dependency row，是否能比一般恢复指标更稳定地解除玩家站位或 target blocker。

A15 在宏入口状态从 `schedule["dependencies"]` 中选择一个依赖行。候选依赖限定为包含 `stand_cell_unreachable_by_player` 或 `target_occupied_by_box` 的行；选择优先级依次为当前依赖优先于未来依赖、`target_occupied_by_box` 优先、`stand_cell_unreachable_by_player` 优先、`candidate_barrier_cost` 较高、`horizon_index` 较小。依赖行以 `room_id`、`slot_cell`、`stand_cell`、`target_cell` 和 `candidate_direction` 形成关系键；`assigned_box` 不进入键，以避免被移动箱子的坐标变化导致同一站位关系无法识别。

宏评分比较入口状态与候选短序列终态的绑定依赖指标，包括依赖行是否仍存在、所选 dependency kind 是否仍存在、绑定站位格是否可由玩家到达、绑定 target 是否仍被箱子占据、绑定 blocker 是否仍在原位置、对应 release action 是否出现。只有当候选序列产生依赖相关进展时，才允许超过阈值执行；单纯增加一般合法推箱数或降低普通 slot cost 不再足以触发 dependency-bound macro。默认参数为 `recovery_macro_max_extra_pushes=4`、`recovery_macro_max_nodes=400`、`recovery_macro_score_threshold=2.0`、`dependency_bound_macro_score_threshold=20.0`。

XSokoban first 第 2 关默认 A15 诊断使用 `budget=256,horizon=8,smoke_steps=200,schedule_lookahead_slots=2`。A14 对照为 `max_steps`、执行 `200` push、`73` 个闭环决策、`boxes_on_goals=0`、final slot cost `33`、dependency count `3`；A15 为 `no_action`、执行 `36` push、`21` 个闭环决策、`boxes_on_goals=0`、final slot cost `42`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。A15 dependency-bound macro 触发 `3` 次，累计追加 `9` 个 push，`row_disappeared_count=5`，`stand_resolved_count=3`，`target_resolved_count=0`。

当前 A15 的方法含义是：绑定具体依赖行能够让宏目标更可解释，但仍不能单独构成求解机制。行级依赖消失或站位格恢复可达并不自动意味着 slot-fill phase 正确、目标填充推进或后续合法动作保持。后续设计应把依赖解除作为必要条件之一，而不是充分条件；宏后置条件还应检查目标完成进度、slot schedule cost 不恶化、关键站位不被新破坏、以及非阻塞合法推箱余量不会坍塌。A15 继续作为负结果诊断，不作为 XSokoban 第 2 关求解配置。

#### A16 feasibility-guarded dependency macro

A16 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner`，继承 A15 的 `RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A15 规则或 Boxoban 主线。A16 的目标是检验：A15 的局部 dependency row 解析必须经过全局可行性后置条件过滤，才能作为短宏执行结果。

A16 保留 A15 的依赖行选择与行级评分，但在候选短宏最终状态上新增验收条件：`dependency_progress=true`，slot schedule cost 不回退，schedule barrier 回退不超过阈值，非阻塞合法推箱数不低于 `recovery_min_legal_pushes`，不进入 `no_legal_pushes` 或低合法余量状态，target blocker、dependency total、critical stand box 和 destructive action 不增加。若候选不满足这些条件，则 `macro_admissible=false`，即使其局部 dependency score 很高也不会作为可执行宏。默认阈值为 `feasibility_guard_max_slot_cost_regression=0.0`、`feasibility_guard_max_schedule_barrier_regression=8.0`。

完整 200 push 诊断在 `recovery_macro_max_nodes=400` 和 `80` 下均因计算成本过高被中止；这说明 A16 的后置条件虽然收窄执行，但仍需为大量被拒绝候选付出搜索和 schedule summary 计算成本。因此，本轮只保留轻量机制 smoke：`budget=128,smoke_steps=80,recovery_macro_max_nodes=20`。该 smoke 相对 A15 从 `no_action,36 pushes,boxes_on_goals=0,final_slot_cost=42,dependency_count=2` 改为 `max_steps,80 pushes,boxes_on_goals=0,final_slot_cost=50,dependency_count=1,target_blocker_count=1`。A16 触发一次 feasibility-guarded macro，追加 `2` 个 push，最终宏验收行无全局后置条件 violation，并解析一个 target dependency。

当前 A16 的方法含义是：全局后置条件是必要的，它能避免 A15 的局部依赖解析直接导向早期 `no_action`；但它仍未提供目标填充进展，后半段出现 `b6:down / b9:up` 长循环。因此，A16 不是求解配置，而是把缺口进一步定位到根选择层和 TVA 校准层：后续需要把“后置条件通过但目标/slot phase 无进展”的状态标为负例，并加入 schedule-level no-progress 或 phase-progress 约束。

#### A17 phase-progress root guard

A17 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner`，继承 A16 的 `RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A16 规则或 Boxoban 主线。A17 的目标是检验：A16 后半段的 `b6:down / b9:up` 类循环是否可以通过根选择层的 schedule-phase no-progress guard 打断。

A17 在候选根动作上新增 `room_phase_progress_score` 与 `room_phase_progress_summary`。phase progress 定义为以下任一条件成立：goal 增加、slot schedule cost 下降超过 `0.5`、dependency total 下降、target blocker 下降、release action 增加，或 schedule barrier 明显下降。若当前存在 dependency 且候选不产生 phase progress，则标记为 `no_progress`；若 child schedule signature 已在近期出现且没有 release action，则标记为 `stagnant_context`。A17 在 prior 中惩罚 no-progress、stagnant context、重复 signature、重复 moved edge、反向 moved edge 和 no-release context；root guard 在最高候选落入重复无进展状态时，优先选择有 phase progress 的候选或较少重复的候选。

为便于诊断，闭环 runner 额外记录 `chosen_phase_progress` 与 `chosen_phase_progress_score`。该记录不影响规划决策。A17 轻量 smoke 使用与 A16 相同的 `budget=128,smoke_steps=80,recovery_macro_max_nodes=20`。结果相对 A16 从 `max_steps,80 pushes,boxes_on_goals=0,final_slot_cost=50,dependency_count=1` 变为 `no_action,22 pushes,boxes_on_goals=0,final_slot_cost=47,dependency_count=2`。A17 选择中 `phase_no_progress_selected_count=14`、`phase_stagnant_context_selected_count=13`，最终 `no_legal_pushes=true`。

当前 A17 的方法含义是：schedule-phase no-progress guard 能改变 A16 的长程循环，但单纯惩罚无进展动作会过约束根选择，使 planner 过早进入 `no_action`。这说明根选择层需要更细的可恢复性判据：区分“无直接 phase progress 但保持可恢复空间”的准备动作，与“无进展且导致后续无动作”的破坏性动作。A17 因此是负结果诊断，不作为求解配置；其 trace 更适合作为 TVA 或根选择校准样本。

#### A18 recoverability-calibrated root guard

A18 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner`，继承 A17 的 `RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A17 规则或 Boxoban 主线。A18 的目标是对 A17 的 no-progress 根保护进行可恢复性校准：无即时 phase progress 的候选不再一律视为坏动作，而是通过短程探测判断后继是否仍能产生 slot、dependency、target blocker 或 release action 进展。

A18 在候选根动作上新增 `room_recoverability_score` 与 `room_recoverability_summary`。当候选已产生 phase progress，或当前不存在 unresolved dependency 时，探测跳过。否则，从候选 child 状态开始做 bounded probe，默认 `recoverability_probe_depth=1`、`recoverability_probe_max_nodes=16`；若探测到后继可产生 recoverability progress 且候选 child 的非阻塞合法推箱数达到 `recovery_min_legal_pushes=2`，则标记为 `recoverable_no_progress` 并给予 root prior bonus；反之标记为 `unrecoverable_no_progress` 并施加 penalty。leaf value 对 unresolved dependency 且低合法余量的状态保留轻量扣分。该机制仍只消费 SD 摘要和 exact legal-push 读数，不保存或查询 teacher 动作表。

XSokoban first 第 2 关 A18 诊断使用与 A17 相同的轻量配置：`budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=1,recoverability_probe_max_nodes=16`。A17 对照为 `no_action`、执行 `22` push、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`；A18 为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`、target blocker count `1`、release action count `0`。A18 仅选择 `1` 次 recoverable no-progress，且未选择 unrecoverable no-progress。

当前 A18 的方法含义是：可恢复性校准方向成立，但当前定义不足。首步 `b7:right` 被标记为 recoverable，是因为深度 1 探测看到 `b6:right` 可带来 `slot_schedule_cost_delta=1.0`，同时合法推箱数未坍塌；然而这个局部 slot-cost 恢复没有减少 dependency、target blocker 或增加 release action，也没有保证宏和后续多步选择后的 legal margin。实际轨迹在 `b7:right,b6:right,b6:right` 后进入 `no_action`。因此 A18 不是求解配置，而是一个 false-positive recoverability 诊断：下一版可恢复性判据应要求结构进展或多步 survivability，不能只依赖一层 slot-cost 局部改善。

#### A19 survivable recoverability guard

A19 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`，继承 A18 的 `RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A18 规则或 Boxoban 主线。A19 的目标是修复 A18 的 false-positive recoverability：纯 slot-cost 局部改善不再作为 no-progress 准备动作的充分放行条件。

A19 仍在候选根动作上记录 `room_recoverability_score` 与 `room_recoverability_summary`，但重写 recoverability probe 判据。结构进展定义为 goal 增加、dependency total 下降、target blocker 下降或 release action 增加；只有结构进展与合法余量同时成立时，probe 才返回 `recoverable=true`。slot schedule cost 或 schedule barrier 改善仍被记录为 `slot_only_progress` 和 `slot_only_survivable`，但不再产生 recoverable bonus，并可作为扣分项。A19 还扩大默认探测深度到 `2`，但正式诊断中将 `recoverability_probe_max_nodes` 限制为 `8`，因为更高节点数会使多层 MCTS prior 计算成本过高。

XSokoban first 第 2 关的 A19 正式保留机制级 smoke：`budget=64,horizon=8,smoke_steps=40,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8,survivable_recoverability_min_legal_pushes=3`。同参 A18 为 `no_action`、执行 `6` push、`boxes_on_goals=0`、final slot cost `47`、dependency count `2`；A19 为 `max_steps`、执行 `40` push、`17` 个闭环决策、`boxes_on_goals=1`、final slot cost `41`、dependency count `2`、target blocker count `1`、release action count `0`。A19 前缀从 A18 的 `b7:right,b6:right,b6:right` 改为 `b6:right,b2:up,b6:right,...`，说明首步 slot-only false positive 已被拦截。

当前 A19 的方法含义是：recoverability 的结构进展约束确实能修复 A18 的早期误判，并产生机制级正向信号；但它仍没有解出第二关。A19 的剩余缺口不是“SD 端完全缺信息”，而是目标完成后仍缺少可执行 release action 生成、后续 dependency 释放和成本可控的根层校准。由于完整 `budget=128,smoke_steps=80` 配置计算成本过高，后续若沿 A19 继续推进，应先把 recoverability probe 缓存或限制到根层，再考虑扩大步数和预算。

#### A20 root-cached survivable recoverability

A20 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`，继承 A19 的 `RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A19 规则或 Boxoban 主线。A20 的目标是降低 A19 的计算成本，而不是改变 recoverability 的结构进展判据。

A20 在每次 `plan()` 调用开始时记录当前闭环决策的 root state/commitment key，并清空 root recoverability cache。`_recoverability_score()` 只在当前 state key 等于 root key 时执行 A19 的 survivable probe；MCTS 内部非根节点直接写入 `probe_skipped=true,skip_reason=non_root_node`，不再枚举后继 recoverability probe。对 root candidates，则以 root key、semantic action key、child canonical key、child commitment key 和 probe 参数构造缓存键，避免同一根候选重复计算。root 选择仍可读取 `recoverable_no_progress` 与 `unrecoverable_no_progress` 摘要；leaf value、schedule summary、宏执行边界均保持 A19/A16 既有逻辑。

XSokoban first 第 2 关 A20 使用 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8,root_recoverability_only=true`。A19 机制级对照为 `max_steps`、执行 `40` push、`boxes_on_goals=1`、final slot cost `41`；A20 为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`，终态不死锁且仍有合法推箱动作。

当前 A20 的方法含义是：将 recoverability probe 限定在 root 层是必要且有效的计算边界修复。它使 A19 能够回到 80-step 诊断，并保留结构进展型 recoverability 对 A18 早期 false positive 的修正。但 A20 仍未解出第二关；两个目标完成后，slot schedule cost 回升且 release action 仍为 0，说明剩余缺口主要是后续 release/action 生成和跨目标调度，而不是继续扩大 root recoverability probe。

#### A21 stand-lock guarded release 诊断

A21 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner`，继承 A20 的 `RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A20 规则或 Boxoban 主线，也不接入 teacher/reference 在线动作。A21 的目标不是生成 release action，而是检验 A20 两个目标后的失败是否可由不可释放的 critical stand blocker 解释。

A21 在候选根动作和 leaf 状态上新增 `room_stand_lock_score` 与 `room_stand_lock_summary`。该摘要读取 A11 schedule-plan dependencies，统计关键站位格被箱子占据的依赖、其中没有合法释放 push 的 locked stand blocker、站位阻塞箱是否位于目标格、以及已在目标格上的 assigned box 是否被重新分派到其他 slot。候选评分惩罚 locked stand blocker 增加、绝对 locked blocker、assigned-goal reassignment，并奖励 release option 增加。root guard 若最高候选存在 locked stand blocker 或 assigned-goal reassignment，则优先选择风险更低的候选；leaf value 对存在 locked blocker 或 assigned-goal reassignment 的状态追加轻量扣分。

XSokoban first 第 2 关 A21 使用与 A20 对齐的 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8,root_recoverability_only=true`。A20 对照为 `max_steps`、执行 `80` push、`51` 个闭环决策、`boxes_on_goals=2`、final slot cost `60`、dependency count `2`、future dependency count `1`、target blocker count `0`、release action count `0`。A21 为 `max_steps`、执行 `80` push、`77` 个闭环决策、`boxes_on_goals=0`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。

当前 A21 的方法含义是：stand-lock 风险是真实诊断信号，但单纯在 MCTS 消费端加入 guard 会过约束调度。A21 与 A20 的首个决策差异发生在 step `3`；A21 后续选择中 `locked_stand_blocker_selected_count=73`，并在后半段形成 `b9:up / b7:down` 往复，目标完成数从 A20 的 `2` 降为 `0`。因此，A21 是负结果诊断。下一步不应继续调大 stand-lock penalty，而应在 A20/A21 暴露的依赖状态上显式生成 release/action 候选，或把 stand-lock 信号作为 release macro 的入口条件和 TVA 校准特征。

#### A22 phased release macro

A22 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner`，继承 A21 的 `RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A21 规则或 Boxoban 主线，也不接入 teacher/reference 在线动作。A22 的目标是检验：stand-lock 不应作为全局强惩罚，而应作为阶段化 release/action 的入口信号。

A22 新增阶段识别摘要 `room_stage_score` 与 `room_stage_summary`，并在闭环 trace 中记录 `chosen_stage` 与 `chosen_stage_score`。默认 `stage_release_min_goals=2`，即在完成至少 2 个目标前保持 A20/A16 兼容的 pre-release dependency-bound macro；进入 release 或 recovery 阶段后，才尝试绑定 `stand_cell_occupied_by_box` 依赖并生成 release-stage macro。A22 同时覆盖 A21 的 leaf value，使 stand-lock leaf penalty 只在 release/recovery 阶段生效，避免准备和装填阶段被全局 stand-lock 惩罚提前改写。

XSokoban first 第 2 关 A22 使用与 A20/A21 对齐的 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`，并设置 `stage_release_min_goals=2,stage_release_stand_occupied_resolution_weight=85.0,stage_release_goal_nonregression=true,stage_stand_lock_score_scale=0.35`。A20 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=60,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`；A21 为 `max_steps,80 pushes,77 decisions,boxes_on_goals=0,final_slot_cost=48,dependency_count=2,target_blocker_count=1,release_action_count=0`；A22 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=60,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`。A20/A22 首个决策差异为 `None`，A21/A22 首个决策差异为 step `3`。

A22 的阶段计数为 `fill=19,release=32`，release-stage 选择 `32` 次；true phased release macro 触发 `0` 次，pre-release compatible macro 触发 `1` 次并追加 `2` 个 push，`locked_stand_blocker_selected_count=48`。当前 A22 的方法含义是：阶段化消费能够避免 A21 的全局 stand-lock guard 退化，并严格保留 A20 的有效前缀；但它没有解出第 2 关，也没有真正触发 release-stage macro。后续不应回到全局 stand-lock penalty，而应修订 release 阶段的 dependency row 选择、候选生成和宏验收条件，使 `stand_cell_occupied_by_box` 从阶段信号转化为可执行释放动作。

#### A23 pre-release bottleneck corridor guard

A23 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner`，继承 A22 的 `RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A22 规则或 Boxoban 主线，也不接入 teacher/reference 在线动作。A23 的目标是检验：A22 的 release 失败是否主要来自 A6 corridor macro 在 fill 阶段越过必要的 dependency/release 前置检查。

A23 不修改 root MCTS 的候选生成，而是在 `corridor_macro_from_plan()` 返回前模拟宏前缀。若宏前缀仍处于 release 里程碑之前，存在 unresolved dependency，`target_blocker_count`、`unreachable_stand_count` 或 critical stand box 仍为正，`release_action_count=0`，且非阻塞合法推箱数不高于 `pre_release_corridor_guard_legal_margin=2`，则将 corridor macro 截断到该前缀，记录 `pre_release_bottleneck_truncated_corridor`。默认 `pre_release_corridor_guard_min_pushes=2`，避免在第一个普通宏动作后立即截断。

XSokoban first 第 2 关 A23 使用与 A22 对齐的 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`，新增 `pre_release_corridor_guard_min_pushes=2,pre_release_corridor_guard_legal_margin=2`。A22 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=60,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`；A23 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=60,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`。A22/A23 首个决策差异为 step `3`。

A23 的 `pre_release_bottleneck_truncation_steps=2`，共节省 `8` 个宏内 push；阶段计数为 `fill=21,release=32`，release-stage 选择仍为 `32` 次，true phased release macro 仍为 `0`。当前 A23 的方法含义是：预释放 bottleneck guard 能改变早期宏执行边界，证明 A22 的互锁确实与 corridor macro 过度推进有关；但它只推迟了互锁，没有生成替代调度，也没有改善终态。后续重点应从“截断宏”转向“根选择层生成可替代的前置依赖解除动作”，尤其是在只有 `b3:up` 这类坏动作和低余量替代动作并存时，需要阶段化 TVA/根选择机制主动偏向可恢复的替代路径。

#### A24 pre-release root-trap guard

A24 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner`，继承 A23 的 `RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A23 规则或 Boxoban 主线，也不接入 teacher/reference 在线动作。A24 的目标是检验：在 A23 已能截断过度 corridor macro 后，根选择层是否可以识别并避开“把 target/stand 前置依赖转化为不可释放 stand-lock”的候选。

A24 在 root candidates 上新增 `room_pre_release_trap_score` 与 `room_pre_release_trap_summary`。该摘要比较父子状态的 `target_blocker_count`、`unreachable_stand_count`、`critical_stand_box_count`、`future_critical_stand_box_count`、`release_action_count`、非阻塞合法推箱数以及 A21/A22 已有的 `locked_stand_blocker_count`。默认只在当前闭环 root 上计算该风险，非 root 节点写入 `probe_skipped=non_root_node`。若候选在 release 里程碑前将前置依赖转化为 `locked_stand_blocker`、产生不可释放 critical stand blocker，或在无 release action 的低合法余量 bottleneck 中增加 critical stand box，则给予惩罚；若减少 dependency、打开 release action 或改善合法余量，则给予小幅奖励。闭环 trace 同步记录 `chosen_pre_release_trap` 与 `chosen_pre_release_trap_score`，该记录不改变旧版本 planner 的行为。

XSokoban first 第 2 关 A24 使用与 A23 对齐的 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`，并新增 `pre_release_root_trap_legal_margin=2,pre_release_root_trap_locked_increase_penalty=110.0,pre_release_root_trap_locked_absolute_penalty=30.0,pre_release_root_trap_conversion_penalty=70.0`。A23 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=60,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`；A24 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=48,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A23/A24 首个决策差异发生在 step `4`。

A24 在 step `4` 成功避开 A23 的 `b3:up`：该候选会使 `locked_stand_blocker_delta=1`、`precondition_to_lock_conversion=true`、`unreleasable_lock_after=true`，score 为 `-298`；A24 改选 `b7:right`。但是该替代动作把状态推进到 `recovery`，非阻塞合法推箱数降至 `1`，下一步只有 `b3:up` 可选，仍被迫生成同一个 severe trap。80-push 结果中 `pre_release_root_trap_selected_severe_count=73`、`pre_release_root_trap_selected_conversion_count=1`、`pre_release_root_trap_top_severe_count=73`，并进入 `b9:up / b7:down` 长循环。

当前 A24 的方法含义是：根层 trap 信号本身是有效的，它能识别并在 step `4` 避开直接 stand-lock 转化；但单步 root guard 已经太晚，替代动作没有配套的多步前置依赖解除计划，反而把 planner 推入只有坏动作可选的 recovery 状态。因此，后续不应继续只提高 root-trap penalty，而应把该信号前移为阶段里程碑或多步调度约束：在进入低余量 bottleneck 之前生成能解除 target/stand 前置依赖、保持合法余量并打开 release action 的替代序列。

#### A25 SD generalization guard

A25 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner`，继承 A24 的 `RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A24 规则或 Boxoban 主线，也不接入 teacher/reference 在线动作。A25 的目标是检验 DQN 类比给出的结构化启发：保留显式 SD 表示，但不要只依赖逐条手工 guard，而是在 MCTS 消费端加入可泛化的结构判断接口。

A25 在 `RoomSemanticCandidate` 与 `RoomSemanticEdge` 上新增 `room_sd_generalization_score` 和 `room_sd_generalization_summary`。该摘要只在当前闭环 root candidates 上计算；非根节点写入跳过标记。A25 将候选后继状态编码为 SD 签名，包括 pre-release、dependency active、target blocker、unreachable stand、critical stand box、locked stand、release closed、low margin 和 single-action margin。随后将该签名与四类手工原型比较：A24 step 5 single-action severe trap、A24 post-trap locked loop、A20/A23 safe fill prefix、open release candidate。评分同时读取 A24 的前后状态差分，并加入一层 successor probe，用于识别 child 是否已经成为“下一步只有 severe trap 可选”的 trap predecessor。

默认参数为 `sd_generalization_legal_margin=2,sd_generalization_predecessor_penalty=95.0,sd_generalization_all_bad_successor_penalty=125.0,sd_generalization_failure_similarity_penalty=70.0,sd_generalization_success_similarity_bonus=20.0,sd_generalization_dependency_reduction_bonus=18.0,sd_generalization_release_open_bonus=24.0,sd_generalization_legal_margin_bonus=4.0`。A25 的 root prior 在 A24 prior 上叠加该 generalization score；root guard 若最高候选被标记为 generalized failure 或 trap predecessor，则优先选择非 generalized failure 且非 A24 severe trap 的替代候选。闭环 trace 同步记录 `chosen_sd_generalization` 与 `chosen_sd_generalization_score`，该记录不影响旧版本 planner。

XSokoban first 第 2 关 A25 使用与 A24 对齐的 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`。A24 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=48,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`；A25 为 `max_steps,80 pushes,78 decisions,boxes_on_goals=0,final_slot_cost=49,dependency_count=1,future_dependency_count=0,target_blocker_count=0,release_action_count=0`。A24/A25 首个决策差异发生在 step `2`。A25 中 `stand_lock_risk_selected_count=0`、`locked_stand_blocker_selected_count=0`、`sd_generalization_selected_severe_count=0`、`sd_generalization_generalized_failure_selected_count=0`、`sd_generalization_trap_predecessor_selected_count=0`，score 范围为 `[5.714,57.714]`。

A25 的方法含义是：原型式 SD-consumer 泛化可以把 A24 的失败证据前移，并在 step `2` 改写根选择，使轨迹完全避开 A24/A23 的 severe stand-lock 路径；但该手工原型并未生成目标进展。80-push 轨迹仍停留在 fill 阶段，并在约 step `17` 后进入 `b9:up / b6:down` 循环，目标完成数仍为 `0`。因此，A25 是负结果但具有机制意义：DQN 类比指向的方向成立，即 SD 需要一个能在未见结构组合间共享经验的消费函数；但手工原型近似过粗，容易把“没有显式陷阱”误当成“有求解进展”。后续应将该接口替换为可学习 TVA/value/ranking consumer，并把 phase-progress、milestone value、dependency release 和 trap avoidance 共同纳入训练或审计目标。

#### A26 SD value-ranked consumer

A26 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner`，继承 A25 的 `RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A8-A25 规则或 Boxoban 主线，也不接入 teacher/reference 在线动作。A26 的目标是把 A25 的原型式泛化接口推进为 learned SD consumer 的线性接口占位：输入仍是显式 SD 摘要，输出改为可被 root prior 和 root choice 共同消费的 value/ranking 分数。

A26 在 `RoomSemanticCandidate` 与 `RoomSemanticEdge` 上新增 `room_sd_value_score` 和 `room_sd_value_summary`，并在闭环 trace 中记录 `chosen_sd_value` 与 `chosen_sd_value_score`。评分读取 goal、release action、dependency、target blocker、slot schedule cost、schedule barrier、合法余量、recoverability、A24 root trap、A25 generalized failure、stand-lock、重复 schedule signature、重复 edge 和反向 edge 等特征。该线性组合不是最终神经 TVA，而是为后续可学习 ranking/value consumer 固定接口与审计字段。

XSokoban first 第 2 关 A26 使用与 A25 对齐的 `budget=128,horizon=8,smoke_steps=80,schedule_lookahead_slots=2,recovery_macro_max_nodes=20,recoverability_probe_depth=2,recoverability_probe_max_nodes=8`。30-push smoke 已达到 `boxes_on_goals=1`；80-push 复核为 `max_steps`、执行 `80` push、`39` 个闭环决策、`boxes_on_goals=1`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `2`、release action count `0`。A25/A26 首个决策差异为 step `2`；A26 中 `sd_value_milestone_selected_count=31`、`sd_value_no_progress_loop_selected_count=1`、`sd_value_inferred_guard_override_count=25`。

A26 的方法含义是：把 SD 摘要转为 value/ranking 消费后，A25 的 0-goal 安全循环被打破，并产生 1-goal 短程进展；但该线性 value 仍把反复移动同一 committed-slot 结构误判为里程碑，最终进入 `b3:down` 重复循环。该结果说明缺口不是 SD 端完全缺信息，而是消费端缺少对“结构进展但无 goal/release 的伪里程碑”的负例校准。

#### A27 SD value loop calibration

A27 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner`，继承 A26 的 `RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner`，只重写 `_sd_value_score()`。该版本不改变候选生成、基础转移、A25 泛化 guard 或 Boxoban 主线，也不接入 teacher/reference 在线动作。

A27 将“有 dependency/target blocker 结构改善，但没有 goal 增加或 release action 打开，并处在 repeated signature、repeated edge、reverse edge 或 stagnant context 中”的候选标记为 `pseudo_milestone_loop`。默认 `sd_value_pseudo_milestone_penalty=190.0`、`sd_value_repeated_milestone_penalty=90.0`。一旦触发伪里程碑循环，摘要会将 `milestone_progress` 改为 false，并同时标记 `no_progress_loop_risk=true`、`hard_negative=true`。

XSokoban first 第 2 关 A27 80-push 复核与 A26 轨迹完全一致，A26/A27 首个决策差异为 `None`。A27 为 `max_steps`、执行 `80` push、`39` 个闭环决策、`boxes_on_goals=1`、final slot cost `48`、dependency count `2`、future dependency count `1`、target blocker count `2`、release action count `0`。但诊断标签发生关键变化：`sd_value_milestone_selected_count` 从 A26 的 `31` 降为 `6`，`sd_value_pseudo_milestone_loop_selected_count=25`，`sd_value_no_progress_loop_selected_count=26`，`sd_value_hard_negative_selected_count=25`。

A27 的方法含义是：伪里程碑识别本身成立，A26 的主要循环已经能被 SD value 摘要标记为 hard negative；但只生成标签不足以改变 root 行为。该结果把缺口进一步定位到 root choice 消费层：如果高层调度不直接消费 `pseudo_milestone_loop`，线性 value 的诊断信息不会自动转化为替代动作选择。

#### A28 SD value loop escape

A28 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner`，继承 A27 的 `RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner`，只重写 `choose_root_edge()`。当父类选择的 root edge 被标记为 `pseudo_milestone_loop`，或同时存在 `hard_negative` 与 `no_progress_loop_risk` 时，A28 优先选择非 hard-negative、非 pseudo-milestone 的替代候选；若不存在这样的候选，则逐级放宽为非 pseudo-milestone 或更高 sd-value 候选。

XSokoban first 第 2 关 A28 30-push smoke 与 A27 在目标数上相同，均为 `boxes_on_goals=1`，但 A27/A28 首个决策差异发生在 step `14`，且 A28 已将被选伪里程碑循环数降为 `0`。80-push 同参复核显示，A28 为 `max_steps`、执行 `80` push、`45` 个闭环决策、`boxes_on_goals=3`、final slot cost `26`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。阶段计数为 `fill=25,release=20`；`sd_value_milestone_selected_count=12`、`sd_value_no_progress_loop_selected_count=19`、`sd_value_pseudo_milestone_loop_selected_count=0`、`sd_value_inferred_guard_override_count=5`。A26/A28 与 A27/A28 的首个决策差异均为 step `14`。

A28 的方法含义是：A27 的伪里程碑标签必须被 root 选择直接消费，才能从诊断信号转化为行为改善。A28 已从 A26/A27 的 1-goal 循环推进到 3-goal 和 release 阶段，证明 learned SD consumer 方向具有实质增益；但它仍未解出第二关，release action 仍为 `0`，后段存在 release-stage 无进展循环。因此，当前难题线的下一缺口不再是 fill 阶段伪里程碑，而是 release 阶段可执行释放动作生成、release-state ranking 和目标不回退约束。

#### A29 release-action macro

A29 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleReleaseActionMacroGFTSDMCTSPlanner`，继承 A28 的 `RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner`，只在 release 阶段且 `release_action_count=0` 时尝试生成释放动作后缀。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A26-A28 learned SD consumer、Boxoban 正式主线或在线 teacher/reference 边界。

A29 的 release-action macro 绑定当前 schedule-plan 中的 `stand_cell_occupied_by_box` 依赖，围绕被占据的关键站位格搜索短后缀。候选评分奖励打开 release action、解除站位格占用、减少 locked stand blocker、减少 dependency/target blocker 和保持合法余量；同时惩罚目标回退、dependency/target blocker 回退、低合法余量和过长后缀。默认后缀搜索参数为 `release_action_macro_max_extra_pushes=5`、`release_action_macro_max_nodes=160`、`release_action_macro_score_threshold=24.0`，并保持 `stage_release_goal_nonregression=true`。

XSokoban first 第 2 关 A29 30-push smoke 未进入 release 阶段，因此只作为机制前缀核验。80-push 同参复核显示，A29 与 A28 轨迹完全一致：`max_steps`、执行 `80` push、`45` 个闭环决策、`boxes_on_goals=3`、final slot cost `26`、dependency count `2`、future dependency count `1`、target blocker count `1`、release action count `0`。A28/A29 首个决策差异为 `None`，`release_stage_decision_count=20`，`release_action_macro_triggered_count=0`，`release_action_macro_opened_release_count=0`。

A29 的关键负诊断不应表述为“release macro 没有运行”。trace 摘要显示，release 阶段入口已经进入，并反复绑定到 `stand_cell_occupied_by_box` 依赖；失败原因计数为 `not_release_stage=25`、`no_release_dependency=1`、`no_admissible_release_suffix=19`。对应依赖的站位格为 `[5,1]`，blocker cell 为 `[5,1]`，`stand_release_option_count=0`，reference metrics 中 `locked_stand_blocker_count=1`、`release_action_count=0`。因此，A29 将剩余缺口进一步定位为：当前短后缀搜索和目标不回退验收条件下，没有可采纳的释放序列。下一步应处理受控目标回退、更长释放序列或显式 goal-locked stand blocker relocation 生成，而不是只继续增加 root value 惩罚。

#### A30 controlled release-regression macro

A30 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner`，继承 A29 的 `RoomScheduleReleaseActionMacroGFTSDMCTSPlanner`。该版本保留 A29 的 release-action macro 入口，并新增两个局部机制：第一，release suffix 的搜索深度从 `5` 提高到 `8`、节点上限从 `160` 提高到 `640`，在存在直接释放进展时允许最多 `1` 个目标的受控临时回退；第二，若继承的 release 宏会制造或保留没有 release action 的 goal-locked stand blocker，则截断该宏并记录 `controlled_release_truncated_goal_lock`。

XSokoban first 第 2 关 A30 30-push smoke 与 A29 一样尚未进入 release 阶段，仅用于机制前缀核验。80-push 同参复核显示，A30 不再与 A29 完全同轨迹：A29/A30 首个决策差异为 step `27`。但 A30 不是正向改进。A29 为 `max_steps,80 pushes,45 decisions,boxes_on_goals=3,final_slot_cost=26,release_action_count=0`；A30 为 `max_steps,80 pushes,52 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=0,release_action_count=0`。

A30 的关键诊断是 `controlled_release_goal_lock_truncation_count=1`。在 step `25`，A29 继承宏原本会执行 `4` 个 push，并在后继状态制造 `goal_locked_stand_blocker_count=1`、`locked_stand_blocker_count=1`、`release_action_count=0` 的状态；A30 将该宏截断为 `1` 个 push，保留状态中 `goal_locked_stand_blocker_count=0`、`locked_stand_blocker_count=0`。这说明危险宏可被识别并阻断。

但 A30 同时显示，单纯阻断危险宏不足以求解。截断后，release 阶段出现 `release_action_macro_reason_counts={'not_release_stage':25,'no_release_dependency':27}`，没有触发 release-action macro，也没有打开 release action；后段转入 2-goal release 循环。该结果把下一缺口从“事后释放 goal-locked blocker”前移为“进入危险宏之前生成替代调度”：需要在 step `25` 前后显式产生能保持目标进展、避免 goal-lock、并打开 release/dependency 进展的候选，而不仅是截断坏宏或允许事后回退。

#### A31 goal-lock predecessor release replacement

A31 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner`，继承 A30 的 `RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner`。该版本不改变 A11 schedule-plan SD、exact grid-flow 转移、A28-A30 既有行为或 Boxoban 主线，也不在在线求解中使用 teacher/reference 动作。A31 的目标是把 A30 的 `controlled_release_truncated_goal_lock` 信号从“宏执行后的截断”前移到 release root 候选替换。

A31 在 `plan_with_corridor_macro()` 后处理层运行。若 A30 返回的宏被标记为 `controlled_release_truncated_goal_lock=true`，A31 会读取当前 root 的已展开 `RoomSemanticEdge` 候选，筛选与原选动作不同、不会降低已完成目标数、且后继状态中 `goal_locked_stand_blocker_count=0` 的替代动作。候选排序优先考虑结构进展：打开 release action、减少 dependency/target blocker、减少 locked stand blocker、改善合法余量或降低 slot schedule cost。若找到替代项，A31 将本次返回 plan 改写为替代 root 的单步动作，并记录 `goal_lock_predecessor_guard_triggered`、原动作、替代动作、候选数、拒绝计数、替代后指标与进展摘要。该机制只改变本次 root action 消费，不修改 MCTS 展开、SD 模型或旧版本 planner。

XSokoban first 第 2 关 A31 30-push smoke 使用 `budget=64,horizon=8,smoke_steps=30`，尚未进入 release 阶段：`max_steps,30 pushes,14 decisions,boxes_on_goals=1,release_stage_decision_count=0,goal_lock_predecessor_override_count=0`。80-push 同参复核使用 `budget=128,horizon=8,smoke_steps=80`。A30 为 `max_steps,80 pushes,52 decisions,boxes_on_goals=2,final_slot_cost=35,release_action_count=0`；A31 为 `max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`。A30/A31 首个决策差异为 step `25`。

A31 的关键诊断是 `goal_lock_predecessor_override_count=2`、`goal_lock_predecessor_structural_progress_count=2`、`goal_lock_predecessor_opened_release_count=2`、`controlled_release_goal_lock_truncation_count=2`。在 step `25` 与 step `30`，原始动作均为 `b6:down`，会进入 A30 标记的 goal-locking macro；A31 改选 `b6:up`，替代后 `goal_delta=0`、`release_action_count_delta=2`、`slot_schedule_cost_delta=20.0`、`goal_locked_stand_blocker_delta=0`。这说明 A31 成功将 A30 的危险宏识别信号转化为 release root 替代动作，并打开了 release action。

但 A31 仍未求解第二关，目标数停在 `2`。step `25` 后虽然 `release_action_count=2`，但轨迹随后反复选择 `b3:down` 类 committed-slot 动作，slot cost 保持在 `29`，dependency 和 target blocker 仍未被继续消解。因此，A31 是机制推进而非求解突破：它把缺口从“release action 打不开”推进到“release action 打开后缺少下一阶段目标推进与循环抑制”。后续应在 A31 基础上继续派生，重点处理 release-action-open 状态下的 root ranking、重复 committed-slot 循环识别和跨阶段里程碑推进，而不是再回到事后释放 goal-locked blocker。

#### A32 learned SD consumer interface

A32 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner`，继承 A31 的 `RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner`。该版本不修改 A11 schedule-plan SD、exact grid-flow 转移、A31 既有机制或 Boxoban 主线，也不在在线求解中查询 teacher/reference 动作。A32 的目标是落实“显式 SD 负责机制表达，神经/可学习消费者负责跨状态组合价值泛化，MCTS 负责 exact dynamics 搜索”的接口，而不是继续堆叠手写规则。

A32 固定 `LEARNED_SD_CONSUMER_FEATURE_NAMES`，共 `44` 个特征，覆盖 goal/release/dependency/target/slot/legal delta，后继 dependency、release、stand-lock 计数，阶段标志，循环和重复边标志，伪里程碑、hard negative、trap、recoverability 标志，以及动作方向和归一化动作编号。checkpoint 支持 `linear` 与 `mlp` 两种前向格式，规划时使用纯 Python 载入 JSON，不依赖在线 torch；若未提供 checkpoint，则行为退化为 A31，只保留可追踪字段。

离线构造脚本 `scripts/build_room_gft_learned_sd_consumer_a32.py` 从 A28-A31 的 trace JSON 中抽取 root/top-child 候选样本，输出 [room_gft_a32_learned_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_dataset.json) 与 [room_gft_a32_learned_sd_consumer_linear_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a32_learned_sd_consumer_linear_checkpoint.json)。当前样本数为 `6845`，目标均值 `-2.9425`、标准差 `2.4570`，线性 ridge 蒸馏的训练内 `MAE=0.1717`、`corr=0.9922`。这些指标只说明 trace-distillation 接口自洽，不能视为充分训练的神经 TVA 或求解器。

A32 默认将 learned score 作为 root prior/value 的软信号消费，即 `_room_prior_score()` 会把 checkpoint 分数写入候选 prior，并在 trace 的 `chosen_sd_value` 中记录 `learned_sd_consumer_score`、特征字典和 `learned_sd_consumer_root_override_enabled=false`。构造参数 `learned_sd_consumer_root_override` 与脚本参数 `--learned-sd-consumer-root-override` 只作为诊断开关，默认关闭；若开启，则 learned scorer 可以直接覆盖父类根选择。

XSokoban first 第 2 关 A32 软消费 30-push smoke 使用 `budget=64,horizon=8,smoke_steps=30`，输出 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A32 首个决策差异为 step `14`，`learned_sd_consumer_loaded_count=14`。

同参 80-push 复核使用 `budget=128,horizon=8,smoke_steps=80`，输出 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a32_learned_sd_consumer_soft_l002_s80_b128_n8.md)。A32 与 A31 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`，A31/A32 首个决策差异为 `None`；learned scorer 在 `42` 个决策中均被加载，正分选择次数为 `20`，score 范围为 `[-3.9273,3.7098]`。

硬覆盖对照使用同一 30-push smoke 但开启 `--learned-sd-consumer-root-override`，输出 [room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a32_learned_sd_consumer_hardoverride_l002_s30_b64_n8.md)。该配置在 step `1` 即偏离 A31，最终 `status=no_action,executed_pushes=6,closed_loop_decision_steps=4,boxes_on_goals=0`。因此 A32 的方法结论是：learned consumer 接口已经连通，但当前 trace 蒸馏 checkpoint 只能作为可审计软信号；把弱 learned scorer 直接作为 root hard override 会破坏搜索稳定性。后续应训练真正的 neural/value/ranking consumer，并以软 prior、leaf value 或受校准 ranking 方式逐步接入，而不是让未充分训练的模型接管根动作。

#### A33 neural SD consumer

A33 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner`，继承 A32 的 `RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner`。该版本不修改 A31/A32 既有机制、A11 schedule-plan SD、exact grid-flow 转移或 Boxoban 主线，也不在在线求解中查询 teacher/reference 动作。A33 的作用是把 A32 的线性 trace 蒸馏推进为实际 MLP checkpoint，并检验“显式 SD 特征 + 神经函数逼近 + MCTS exact search”的最小闭环。

A33 planner 保留 A32 的 checkpoint JSON 前向格式，但默认设置更保守：`learned_sd_consumer_weight=0.35`、`learned_sd_consumer_root_override=false`、`neural_sd_consumer_score_clip=4.0`。若 checkpoint 提供 `feature_scaling`，A33 会用训练时的动作编号归一化分母重写 `action_box_index_norm` 与 `action_flat_index_norm`，避免训练和在线前向的尺度不一致。trace 中继续记录 `learned_sd_consumer_raw_score`、`learned_sd_consumer_score`、特征字典，并新增 `neural_sd_consumer_score_clip`。

训练脚本 [train_room_gft_neural_sd_consumer_a33.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_neural_sd_consumer_a33.py) 使用 A24-A32 的离线 trace，按 `source_file/version/step` root group 做稳定哈希切分，默认 `validation_fraction=0.2,seed=3301,hidden_sizes=[32,16],epochs=800,lr=0.001,weight_decay=0.0001`。输出为 [room_gft_a33_neural_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_dataset.json) 与 [room_gft_a33_neural_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a33_neural_sd_consumer_mlp_checkpoint.json)。训练样本数 `12462`，训练集 `10095`、验证集 `2367`，root group 数 `2684`、验证 group 数 `513`。验证指标为 `MAE=0.0696,corr=0.9986,target_top1_accuracy=0.8642,selected_top1_accuracy=0.5389`。其中 target-top1 衡量对启发式 SD progress target 的排序拟合，selected-top1 只作为与历史 planner 选择的一致性参考，不作为正确性标签。

A33 soft 30-push smoke 输出 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，`learned_sd_consumer_loaded_count=14`，score 范围 `[-0.9109,1.3072]`。该 smoke 未出现 A32 hard override 的早期 `no_action`。

A33 soft 80-push 同参复核输出 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a33_neural_sd_consumer_soft_l002_s80_b128_n8.md)。A33 相对 A31/A32 产生闭环行为变化：A31/A33 首个决策差异为 step `14`。结果为 `max_steps,80 pushes,46 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`，stage 计数为 `fill=27,release=19`，`goal_lock_predecessor_override_count=4`、`goal_lock_predecessor_opened_release_count=4`，neural scorer 加载 `46` 次，正分选择 `28` 次，score 范围 `[-2.1753,1.4906]`。

A33 hard override 30-push 对照输出 [room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a33_neural_sd_consumer_hardoverride_l002_s30_b64_n8.md)。该配置仍在 step `1` 偏离并退化为 `status=no_action,executed_pushes=6,boxes_on_goals=0`。因此 A33 的方法结论是：神经消费者已经能以软 prior 方式改变搜索轨迹，并保持 teacher/mainline 边界；但当前目标仍是启发式 trace-distillation target，未解决 release-stage 真实进展与循环校准。下一步应改进训练目标，而不是开启 hard override 或继续添加手写规则。

#### A34 milestone SD consumer

A34 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner`，继承 A33 的 `RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner`。该版本不新增 Sokoban 手写动作规则，只隔离 milestone checkpoint 的 provider id、默认 score clip 和 trace 标记；不修改 A31-A33、A11 schedule-plan SD、exact grid-flow 转移或 Boxoban 主线，在线求解仍不查询 teacher/reference 动作。

训练脚本 [train_room_gft_milestone_sd_consumer_a34.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_milestone_sd_consumer_a34.py) 复用 A33 的 MLP 导出格式，但将目标从即时 SD progress 改为 future-window milestone target。该目标在每个 trace step 后向前看 `8` 个闭环决策，奖励未来窗口内的 goal gain、release action gain、dependency/target blocker 下降和 slot cost 下降，惩罚 release-open stagnation、无未来里程碑和目标回退；同时保留一部分 immediate SD target 作为局部结构信号。训练输入为 A24-A34 可用前序 trace 中的 top-child 候选，输出 [room_gft_a34_milestone_sd_consumer_dataset.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_dataset.json) 与 [room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_a34_milestone_sd_consumer_mlp_checkpoint.json)。

A34 checkpoint 使用 `hidden_sizes=[32,16],epochs=900,lr=0.001,weight_decay=0.0001,seed=3401,future_horizon=8`。样本数 `13077`，训练集 `10560`、验证集 `2517`，root group 数 `2818`、验证 group 数 `546`。验证指标为 `MAE=0.3929,corr=0.9371,target_top1_accuracy=0.8318,selected_top1_accuracy=0.5831`。相比 A33，该目标更难拟合但更接近“未来进展”问题；selected-top1 更高说明它更贴近部分历史成功/候选选择，但仍不能视为真实求解价值。

A34 soft 30-push smoke 输出 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，`learned_sd_consumer_loaded_count=14`，score 范围 `[0.1630,2.6834]`，未出现早期 `no_action`。

A34 soft 80-push 同参复核输出 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a34_milestone_sd_consumer_soft_l002_s80_b128_n8.md)。A34 与 A31/A32 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,future_dependency_count=1,target_blocker_count=2,release_action_count=2`，A31/A34 首个决策差异为 `None`；milestone scorer 加载 `42` 次，正分选择 `29` 次，score 范围 `[-0.5022,3.0]`，其中多个前缀分数被 clip 到 `3.0`。

A34 hard override 30-push 对照输出 [room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a34_milestone_sd_consumer_hardoverride_l002_s30_b64_n8.md)。该配置仍在 step `1` 偏离并退化为 `status=no_action,executed_pushes=6,boxes_on_goals=0`。因此 A34 的方法结论是：future-window milestone target 比 A33 更贴近理论目标，但当前尺度偏正且判别力不足，soft prior 未改变 A31/A32 的 release-open 循环，hard override 仍不稳定。下一步应做目标校准、多头分解或引入更强离线 solved-distance/teacher-search 标签，而不是继续放大单一 scalar score。

#### A35 multi-head SD consumer

A35 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner`，继承 A34 的 `RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner`。该版本不新增 Sokoban 手写动作规则，只把 A34 的单一 milestone scalar 拆成 value、release-progress、loop-risk、trap-risk 四个神经输出头；不修改 A31-A34、A11 schedule-plan SD、exact grid-flow 转移或 Boxoban 主线，在线求解仍不查询 teacher/reference 动作。

训练脚本 [train_room_gft_multihead_sd_consumer_a35.py](/home/data/round/code/sd_world_model_rl/scripts/train_room_gft_multihead_sd_consumer_a35.py) 复用 A34 的 future-window 数据构造，但为每个候选补充 `targets` 字段。`value` 头沿用 A34 future-window milestone target；`release_progress` 头刻画 release action 打开、dependency/target blocker 消解、slot 推进及未来窗口中的 release/dependency 进展；`loop_risk` 头刻画伪里程碑、重复边、重复 schedule signature、release-open stagnation 与无未来里程碑；`trap_risk` 头刻画 trap、severe trap、generalized failure、stand-lock、unrecoverable no-progress、low-margin 和 hard negative。四个头共享同一 `44` 维 SD 特征契约，在线消费时按 `0.45*value + 1.15*release_progress - 1.05*loop_risk - 1.25*trap_risk` 合成，再乘以 `learned_sd_consumer_weight=0.35` 并用 `neural_sd_consumer_score_clip=3.0` 截断。

A35 checkpoint 使用 `hidden_sizes=[32,16],epochs=900,lr=0.001,weight_decay=0.0001,seed=3501,future_horizon=8`。样本数 `13928`，训练集 `11216`、验证集 `2712`，root group 数 `3004`、验证 group 数 `588`。组合目标验证指标为 `MAE=0.2662,corr=0.9612,target_top1_accuracy=0.8799,selected_top1_accuracy=0.6193`。各头验证指标为：value `MAE=0.4292,corr=0.9358,target_top1=0.8799`；release-progress `MAE=0.0707,corr=0.9522,target_top1=0.7650`；loop-risk `MAE=0.0210,corr=0.9905,target_top1=0.7324`；trap-risk `MAE=0.0131,corr=0.9989,target_top1=0.4991`。这些指标说明多头目标可拟合，但不等价于闭环求解能力。

A35 soft 30-push smoke 输出 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A35 首个决策差异为 step `14`，score 范围 `[-0.1368,1.5011]`。该 smoke 说明多头 soft prior 可以改变早期轨迹，但小预算下未带来进展。

A35 soft 80-push 同参复核输出 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a35_multihead_sd_consumer_soft_l002_s80_b128_n8.md)。A35 与 A31/A34 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A35 首个决策差异为 `None`。multi-head scorer 加载 `42` 次，正分选择 `39` 次，score 范围 `[-0.3828,2.5846]`。被选动作的头部范围为 value `[-1.6509,10.0]`、release-progress `[0.0,2.9983]`、loop-risk `[0.0,1.4017]`、trap-risk `[0.0,0.7312]`。

A35 的方法结论是：多头表达解决了 A34 单一 scalar 在可审计性上的混叠问题，但默认组合权重仍未改变 80-push 闭环失败结构。尤其在 release-open 后，`b3:down` 重复段仍被合成分数赋予正值约 `0.1639`，说明 release-progress 头对该循环的正贡献仍抵消了 loop-risk 惩罚。因此 A35 不是求解突破，而是把下一缺口定位到“多头校准和消费方式”：需要对 release-open 循环构造更强 hard negative、校准 release-progress 与 loop-risk 的相对权重，或将多头信号接入 leaf value/rollout evaluation，而不是让 root soft prior 独立承担阶段转换。

#### A36 agenda progress-certificate gate

A36 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner`，继承 A35 的 `RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner`。该版本不训练新 checkpoint，不新增动作生成规则，也不修改 exact grid-flow 转移或 Boxoban 主线；它复用 A35 multi-head checkpoint，只在在线消费时加入最小 agenda progress-certificate gate，用于检验“release opened 不等于 release consumed，没有证书的 release-progress 不应继续作为真实进展”这一假设。

A36 的证书目前仍是最小局部版本。它读取候选动作已有的 phase progress、pre-release trap、stand-lock 和 SD-value 摘要，定义 `goal_progress`、`blocker_reduction`、`slot_frontier_progress`、`barrier_progress`、`release_opened`、`release_assigned`、`release_consumed`、`stale_open_release`、`agenda_loop` 与 `pseudo_progress`。若 release-progress 头为正但没有上述证书，则将 `release_head_effective=0` 并记录 `release_gate_reason=no_certificate`；若只是打开 release 而未消费，则将 release head 上限截断为 `0.35` 并记录 `opened_not_consumed_cap`；若证书成立，则保留有效 release head 并给小幅 agenda bonus。A36 同时记录 `agenda_tier`，但该版本仍只把证书调整作为 soft prior 的一部分消费，不改变 MCTS 树结构。

A36 soft 30-push smoke 输出 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A36 首个决策差异为 step `14`。证书字段正常落盘：`agenda_certificate_loaded_count=14,agenda_certified_progress_count=13,agenda_pseudo_progress_count=1,agenda_stale_open_release_count=0,agenda_release_consumed_count=0`。

A36 soft 80-push 同参复核输出 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a36_agenda_certificate_soft_l002_s80_b128_n8.md)。A36 与 A31/A35 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A36 首个决策差异为 `None`。证书计数为 `agenda_certificate_loaded_count=42,agenda_certified_progress_count=39,agenda_pseudo_progress_count=1,agenda_stale_open_release_count=0,agenda_release_opened_count=4,agenda_release_consumed_count=0,agenda_loop_count=0`，tier 分布为 `{5:15,6:22,3:1,7:2,4:2}`。

A36 的关键负诊断是：最小局部 certificate gate 仍未识别 A31/A35 后段 `b3:down` 重复循环。step `31` 至 step `41` 的重复 `b3:down` 仍被标记为 `certified_progress=True,tier=6`，`release_head_raw=2.3367` 且 `release_head_effective=2.3367`，合成分数约 `0.2864`。这说明仅用局部 slot/dependency/target delta 构造证书仍会把循环中的形式进展误判为真实进展。A36 因此支持专家意见中的下一步：必须引入 active agenda 与 blocker-set 层，使证书从“局部 delta 是否改善”升级为“是否减少当前 active obligation 的 blocker，或是否消费已打开 release”。继续调 release 权重或扩大预算不足以解决该缺口。

#### A37 active-agenda progress certificate

A37 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner`，继承 A36 的 `RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner`。该版本不训练新 checkpoint，不新增动作生成规则，不修改 exact grid-flow 转移、MCTS 展开结构或 Boxoban 主线；它复用 A35 multi-head checkpoint，并把 A36 的局部 certificate 升级为 active-obligation certificate。

A37 的最小 active agenda 包含三类义务：`release_blocker`、`consume_release` 与 `fill_frontier_slot`。当当前状态已有 opened release，即 `before_release_action_count>0` 时，active agenda 进入 `consume_release`；否则若存在 dependency、target blocker 或 release 阶段上下文，则为 `release_blocker`；若无上述 blocker 但仍有待填 slot，则为 `fill_frontier_slot`。证书不再直接接受局部 delta，而是记录 `active_agenda_blocker_before`、`active_agenda_blocker_after`、`active_agenda_blocker_delta`、`active_agenda_progress`、`active_agenda_off_agenda_action`、`active_agenda_release_loop`、`active_agenda_type` 与 `active_agenda_reason`。在 `consume_release` 中，重复上下文下关闭 release、slot cost 回退且没有 goal/frontier 进展的动作被标记为 `repeated_release_closure`，其 release head 被置零，并施加 off-agenda、stale-release 与 release-loop 惩罚。

A37 soft 30-push smoke 输出 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A37 首个决策差异为 step `14`。active-agenda 字段正常落盘：`active_agenda_loaded_count=14,active_agenda_progress_count=12,active_agenda_off_agenda_action_count=2,active_agenda_pseudo_progress_count=2,active_agenda_release_loop_count=0`。

A37 soft 80-push 同参复核输出 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a37_active_agenda_certificate_soft_l002_s80_b128_n8.md)。A37 与 A31/A35/A36 仍完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A37 首个决策差异为 `None`。计数为 `active_agenda_loaded_count=42,active_agenda_progress_count=25,active_agenda_off_agenda_action_count=17,active_agenda_pseudo_progress_count=17,active_agenda_stale_open_release_count=13,active_agenda_release_loop_count=13,active_agenda_release_consumed_count=1`，agenda 类型分布为 `{release_blocker:28,consume_release:14}`。

A37 的核心诊断是：它纠正了 A36 的证书误判，但没有改变闭环选择。A36 中 step `31` 至 step `41` 的重复 `b3:down` 被标为 `certified_progress=True,tier=6,release_head_effective=2.3367`；A37 将同一段标为 `active_agenda_type=consume_release,active_agenda_reason=repeated_release_closure,active_agenda_progress=False,active_agenda_off_agenda_action=True,active_agenda_release_loop=True,certified_progress=False,tier=1,release_head_effective=0`，合成 learned score 约为 `-2.4741`。但 root top-child 仍选择该动作：step `31` 时 `b3:down` 的 `prior_score=266.0073,room_sd_value.score=150,visits=120`，而其他候选均为 `visits=1` 且 prior 为负。该证据说明 A37 的 active agenda 层已能识别伪进展，缺口转移到 MCTS 消费端：soft prior 惩罚不足以覆盖旧启发式 SD 分数和树搜索访问优势，后续需要 hard gate、agenda-option 展开或 leaf/rollout value 接入，而不是继续细化局部规则。

#### A38 agenda leaf potential

A38 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner`，继承 A37 的 `RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的 E3：将 active-agenda potential 接入 MCTS leaf value，检验 active agenda 是否只有作为 root soft prior 不足，而进入树内叶价值后能改变闭环选择。A38 不训练新 checkpoint，不新增动作生成规则，不引入图网络、teacher imitation、hard gate 或 agenda-option tree，不修改 exact grid-flow 转移或 Boxoban 主线；在线求解仍保持 `teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

A38 的 leaf potential 读取当前 leaf 状态的 schedule-plan 摘要、decision-stage 摘要和近期 schedule signature。最小 agenda 类型沿用 A37 语义，分为 `consume_release`、`release_blocker`、`fill_frontier_slot` 与 `goal_completion`。其 leaf adjustment 采用显式可审计形式：`neutral_offset + bonus - penalty`，并在叠加到父类 `leaf_value()` 后按 `[deadend_value, goal_value]` 截断。默认参数为 `agenda_leaf_neutral_offset=10.00,agenda_leaf_slot_cost_weight=0.04,agenda_leaf_blocker_penalty=2.25,agenda_leaf_open_release_bonus=0.55,agenda_leaf_stale_release_penalty=5.50,agenda_leaf_loop_penalty=7.50,agenda_leaf_goal_locked_penalty=4.00,agenda_leaf_low_legal_penalty=2.50,agenda_leaf_release_stage_bonus=0.35`。其中 `neutral_offset` 用于避免所有根 leaf value 在负修正下直接被 `deadend_value=-10` 截断，保证正负 potential 均可观测。

A38 soft 30-push smoke 输出 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`；leaf potential 正常落盘，`agenda_leaf_loaded_count=14,agenda_leaf_adjustment_min=-3.11,agenda_leaf_adjustment_max=3.93,agenda_leaf_adjustment_mean=0.01,agenda_leaf_eval_count_total=866,agenda_leaf_loop_eval_count_total=0`。该 smoke 说明 E3 信号已进入叶价值并可被 trace 审计。

A38 soft 80-push 同参复核输出 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a38_agenda_leaf_potential_soft_l002_s80_b128_n8.md)。A38 与 A31/A37 完全同轨迹：`max_steps,80 pushes,42 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A31/A38 首个决策差异为 `None`。leaf 统计为 `agenda_leaf_loaded_count=42,agenda_leaf_adjustment_min=-14.31,agenda_leaf_adjustment_max=3.93,agenda_leaf_adjustment_mean=-5.4133,agenda_leaf_eval_count_total=5302,agenda_leaf_loop_eval_count_total=129,agenda_leaf_stale_release_eval_count_total=74,agenda_leaf_root_loop_count=17,agenda_leaf_root_stale_open_release_count=14`，类型分布为 `{release_blocker:28,consume_release:14}`。

A38 的方法结论是：E3 leaf agenda potential 已经有效接入，并在 release-loop 叶节点上产生明显负修正；但它仍不能改变当前根动作选择。step `31` 时被 A37/A38 标为 `consume_release`、off-agenda、release-loop 的 `b3:down` 仍保有 `prior_score=266.0073,room_sd_value.score=150,visits=120`，而其他候选只有 `visits=1` 且 prior 多为负。该负结果说明，仅把 active agenda 作为可加 leaf value 仍不足以压过旧启发式 prior 和访问数优势。后续若继续推进，应仍限定在专家 27 点内，优先评估第 16 点的 lexicographic tier / hard downgrade 或第 14 点的 agenda-option tree，而不是继续堆局部规则或单纯扩大预算。

#### A39 agenda-tier root choice

A39 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner`，继承 A38 的 `RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的第 16 点：将 A37/A38 已有的 `agenda_tier`、`active_agenda_off_agenda_action` 与 `active_agenda_release_loop` 从 root soft prior 或 additive leaf value 转为 root choice 的 lexicographic tier / hard downgrade。A39 不训练新 checkpoint，不新增动作生成规则，不引入图网络、teacher imitation 或 agenda-option tree，不修改 exact grid-flow 转移或 Boxoban 主线；在线求解仍保持 `teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

A39 默认不是全局 tier 接管，而是证据限定的 `release_loop_only` 模式。只有当父类选择的 root edge 满足 `active_agenda_type=consume_release`、`agenda_tier<=1`，且被标记为 release-loop、off-agenda 或 stale-open-release 时，才触发 tier 降级；候选替代项也限定在 `consume_release` 且非 release-loop 的 root edges 内。默认参数为 `agenda_tier_choice_mode=release_loop_only,agenda_tier_choice_parent_max_tier=1,agenda_tier_choice_min_improvement=1`。该限制来自 A37/A38 的证据链：真正被确认的失败点是 release-open 后的 repeated release closure，而不是 fill 阶段所有低 tier 动作。因此 A39 不把第 16 点扩展为未经论证的全局动作接管器。

A39 soft 30-push smoke 输出 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，A31/A39 首个决策差异为 step `14`，`agenda_tier_choice_loaded_count=14,agenda_tier_choice_override_count=0`。该 smoke 说明 `release_loop_only` 约束没有在早期 fill 阶段造成全局 tier 接管退化。

A39 soft 80-push 同参复核输出 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a39_agenda_tier_choice_soft_l002_s80_b128_n8.md)。A39 相对 A31/A38 产生闭环行为变化：A31/A39 首个决策差异为 step `26`，A38/A39 首个差异同样位于 step `26`。A39 为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`。tier 统计为 `agenda_tier_choice_loaded_count=53,agenda_tier_choice_override_count=20,agenda_tier_choice_higher_tier_selected_count=20`；parent tier 分布为 `{5:14,6:8,1:21,2:6,7:4}`，selected tier 分布为 `{5:14,6:8,1:1,2:25,7:5}`。

A39 的关键正证据是 step `26`。父类选择会落在 `b3:down`，该候选有 `visits=110,prior_score=292.9614,room_sd_value.score=150`，但被 active agenda 标记为 `consume_release,tier=1,off-agenda=true,release-loop=true,certified_progress=false`。A39 将其降级，并选择 `b4:up`，该候选为 `consume_release,tier=7,active_agenda_progress=true,off-agenda=false,release-loop=false`，尽管它只有 `visits=1`、`mean_value=-4.91`。这说明第 16 点的 lexicographic tier 确实能压过旧启发式 prior 与 visits，改变 A37/A38 无法改变的 root 选择。

A39 仍不是求解突破。虽然它打散了 A38 中的连续 `b3:down` release-loop，最终仍停在 `2` goals、slot cost `29`。后段大量 override 只是从 tier-1 release-loop 切换到 tier-2 off-agenda `consume_release` 候选，未能持续生成 tier-5/7 的真实进展动作。因此 A39 的方法结论是：第 16 点的消费位置修正是有效的，但当前候选集和 agenda 层仍缺少可持续推进当前义务的 option 结构。后续若继续推进，应在专家 27 点内评估第 14 点 agenda-option tree，或先改进 active agenda 的候选生成与义务里程碑定义，而不是继续扩大预算或调单一分数。

#### A40 bounded agenda-option tree

A40 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner`，继承 A39 的 `RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的第 14 点，即从单步 push/root ranking 推进到 agenda-option tree。A40 采用保守实现：不新增可执行宏动作，不改变 exact grid-flow 转移，不训练新 checkpoint，不引入图网络或 teacher imitation；它只在 root 处为候选第一步运行 bounded local option evaluator，判断该第一步之后是否能在短程真实 push tree 内完成当前 active agenda。在线求解仍保持 `teacher_actions_used_online=false`、`mainline_mutation=false`、`learned_sd_consumer_root_override=false`。

A40 的 option evaluator 以当前 root 的结构摘要作为 reference agenda。摘要包括 `boxes_on_goals`、`release_action_count`、`dependency_total`、`target_blocker_count`、`locked_stand_blocker_count`、`goal_locked_stand_blocker_count`、`slot_schedule_cost`、`schedule_barrier_cost`、`nonblocked_legal_push_count` 与 schedule signature。agenda 类型沿用 A37/A39 的最小定义：已有 open release 时为 `consume_release`，否则若仍有 active blocker 或 release 阶段上下文则为 `release_blocker`，再否则为 `fill_frontier_slot` 或 `goal_completion`。每个候选 first edge 后，A40 用真实 legal push 展开最多 `agenda_option_tree_max_pushes=5` 步、最多 `agenda_option_tree_max_nodes=96` 个节点、最多评价 `agenda_option_tree_max_root_candidates=10` 个 root 候选；深层 commitment 使用 `RoomSlotOption(kind="none")` 的保守推进，以避免在 A40 中引入新的 slot-option 分支生成规则。

局部 option 的完成谓词不是“解完整关卡”，而是“到达当前 agenda 的下一个边界”。在 `consume_release` 中，若 open release 被关闭并伴随 active blocker 下降、slot/frontier 进展或合法余量改善，则标为 `agenda_done=true` 和 tier `7`；若只产生形式上的 release closure、重复 signature、path loop 或 slot 回退，则标为 off-agenda、agenda-loop 或低 tier。A40 只用该局部评价选择 root first edge，不执行 suffix。因此它是第 14 点的最小 option-evaluation 版本，不是 option-commitment 版本。

A40 soft 30-push smoke 输出 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，与 A39 前缀一致；`agenda_option_tree_loaded_count=14,agenda_option_tree_override_count=0`，原因均为 `parent_not_option_trigger`。这说明 A40 在早期 fill/release-blocker 阶段没有造成未经论证的全局接管。

A40 soft 80-push 同参复核输出 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a40_agenda_option_tree_soft_l002_s80_b128_n8.md)。A40 与 A39 完全同轨迹：`max_steps,80 pushes,53 decisions,boxes_on_goals=2,final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`，A39/A40 首个决策差异为 `None`。option tree 统计为 `agenda_option_tree_loaded_count=53,agenda_option_tree_override_count=0,agenda_option_tree_selected_progress_count=22,agenda_option_tree_selected_done_count=22,agenda_option_tree_selected_loop_count=0,total_evaluated_option_count=186,max_selected_depth=4,max_selected_expanded_nodes=96`。原因分布为 `{parent_not_option_trigger:31,parent_option_already_best_tier:22}`。

A40 的核心诊断是：它能找到短程可完成当前 `consume_release` 义务的 option，但因为只执行第一步，闭环随后可能偏离该 option。step `29` 之后，A40 多次发现三步左右的 tier-7 option，例如 step `29` 的已选 root `b6:up` 对应局部路径 `b6:up -> b9:left -> b4:down`，其 option progress 为 `agenda_done=true,release_consumed=true,release_action_count_delta=-2,active_blocker_delta=5,dependency_total_delta=3,target_blocker_count_delta=2,after_active_blocker_count=0`。类似地 step `30`、`31` 的 `b3:left -> b9:left -> b4:down` 也被评为 tier-7 agenda-done。可是 A40 并不承诺执行后续 `b9:left -> b4:down`，下一闭环决策会重新规划并可能改走其他 tier-2 off-agenda 路线。

因此，A40 不是求解突破，而是把缺口进一步定位清楚：当前系统并非完全“不知道”短程 agenda 完成路径，而是缺少把 option 作为持续议程承诺来执行和维护的机制。后续若继续推进，仍应限定在专家第 14 点或第 6 点附近：将 active agenda/option 写入节点状态或闭环 commitment，使 MCTS 在若干步内保持同一 obligation-option，直到 agenda 完成、遇到风险或到达新 frontier；而不是继续调 root 分数或扩大局部 option evaluator 的节点预算。

#### A41 bounded agenda-option commitment

A41 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner`，继承 A40 的 `RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的第 14 点和第 6 点：A40 负责在 root 处发现短程 agenda-option，A41 检验该 option 是否需要作为持续议程状态被执行和维护。A41 不训练新 checkpoint，不在线调用 teacher/reference，不改变 exact grid-flow 转移，不修改 Boxoban 主线，也不启用 learned root override；它只消费 A40 已经选出的短程 option。

A41 的承诺条件是保守的：仅当 A40 的 `selected_option` 达到 `agenda_tier>=7` 且 `agenda_done=true`，并且该 option 至少包含 `2` 个 push 时，才考虑执行后缀。为避免与既有 corridor/release 宏混叠，若父层 `corridor_macro` 已经包含多步宏，A41 默认放弃承诺；若为单步动作，则重新用真实 legal push 验证 A40 给出的每个 suffix push，遇到非法动作、deadlock child 或 path loop 即中止。默认参数为 `agenda_option_commitment_min_tier=7`、`agenda_option_commitment_require_done=true`、`agenda_option_commitment_max_extra_pushes=4`、`agenda_option_commitment_min_total_pushes=2`。闭环 trace 新增 `agenda_option_commitment` 字段，记录触发原因、已验证动作、最终 progress 和额外承诺 push 数。

A41 soft 30-push smoke 输出 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48,release_action_count=0`，与 A40 前缀一致；`agenda_option_commitment_triggered_count=0`，原因均为 `no_selected_option`。这说明 A41 不会在早期未出现 A40 option 的阶段强行接管。

A41 soft 80-push 同参复核输出 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a41_agenda_option_commitment_soft_l002_s80_b128_n8.md)。A40/A41 首个决策差异发生在 step `30`，因为 A41 在 step `29` 将 A40 已发现的 `b6:up -> b9:left -> b4:down` 作为三步承诺执行。A41 仍为 `max_steps,80 pushes,53 decisions,boxes_on_goals=2`，但最终结构状态不同：`final_slot_cost=41,dependency_count=0,target_blocker_count=0,release_action_count=0`；A40 对应为 `final_slot_cost=29,dependency_count=2,target_blocker_count=2,release_action_count=2`。A41 的 commitment 统计为 `loaded_count=53,triggered_count=1,first_trigger_step=29,extra_push_count=2,done_count=1`，原因分布为 `{no_selected_option:52,validated_option_committed:1}`。

关键观察是：A41 的承诺机制确实完成了 A40 识别出的 `consume_release` 义务。step `29` 的三步路径 `b6:up -> b9:left -> b4:down` 经重新验证后执行，最终 progress 为 `agenda_done=true,release_consumed=true,release_action_count_delta=-2,active_blocker_delta=5,dependency_total_delta=3,target_blocker_count_delta=2,after_active_blocker_count=0`。但 release 义务完成后，闭环转入 `fill` 阶段并长期停留在 `slot_schedule_cost=41`，没有在 80-push 内推进到第三个目标。因此，A41 是机制验证而非求解突破：它证明 A40 的短程 option 可以被承诺执行，但下一缺口已转移到 release 完成后的 fill 阶段里程碑、slot-distance/value 表达和阶段切换后的目标选择。

#### A42 post-release fill-frontier option

A42 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner`，继承 A41 的 `RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中关于 frontier obligation、agenda-option tree 和 next agenda milestone 的建议，目标是检验 A41 完成 `consume_release` 之后，是否需要把 fill 阶段的当前 frontier slot completion 作为可消费的独立 agenda boundary。A42 不训练新 checkpoint，不在线调用 teacher/reference，不改变 exact grid-flow 转移，不修改 Boxoban 主线，也不启用 learned root override。

A42 的触发条件被限制在 clean post-release fill context：当前阶段为 `fill`，`release_action_count=0`，`dependency_total=0`，`target_blocker_count=0`，已有有效填充 slot，且父类候选对 `fill_frontier_slot` 处于低 tier 或 off-agenda。候选必须满足无 deadlock、存在合法后继、默认不回退已完成目标，并且完成当前 frontier slot 或至少使 slot schedule cost 下降 `fill_frontier_option_min_slot_delta=1.0`。排序优先 frontier completion、slot 进展、有效 slot 数增加和结构风险下降；若父类候选已经同等或更优，则不覆盖。

A42 soft 30-push smoke 输出 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A41/A42 首个决策差异为 `None`，`fill_frontier_option_override_count=0`，原因均为 `parent_not_fill_frontier_trigger`。这说明 A42 不会在早期尚未进入 clean fill 语境时全局接管。

A42 soft 80-push 同参复核输出 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a42_fill_frontier_option_soft_l002_s80_b128_n8.md)。A42 为 `max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。A41/A42 首个决策差异为 step `30`；A41 的最终 slot cost 为 `41`，A42 降为 `35`，目标数仍为 `2`。

关键正证据是 step `30`。A42 覆盖 A41 的无进展 fill 动作，选择 `b4:down`，随后通过既有 corridor macro 执行 `b4:down -> b6:down -> b6:left` 三个 push。该 option 的局部评价为 `frontier_completed=true,slot_progress=true,goal_delta=0,slot_schedule_cost_delta=20,valid_filled_slot_count_delta=1`，因此确实补上了 A41 暴露出的 fill frontier milestone 消费缺口。

但 A42 仍未求解第二关。step `30` 的 fill frontier completion 打开了新的 release/release-blocker 局面；step `31` 以后，已选动作持续被标为 `release_blocker,tier=2,active_agenda_progress=false,active_agenda_off_agenda_action=true`，并且 `release_action_count=0,dependency_count=2,target_blocker_count=1,slot_schedule_cost=35` 长期不变。因此，A42 是局部机制推进而非求解突破：它证明 post-release fill frontier 可以作为独立 agenda boundary 被消费，但下一缺口已经转移到 frontier completion 之后的 release-blocker option、阶段状态持久化和可执行里程碑生成。

#### A43 post-frontier release-blocker option

A43 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner`，继承 A42 的 `RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的 agenda-option tree、frontier obligation 后续 release-blocker option 和 agenda state persistence。A43 不训练新 checkpoint，不在线调用 teacher/reference，不改变 exact grid-flow 转移，不修改 Boxoban 主线，也不启用 learned root override。

A43 的触发条件被限制在 A42 暴露出的 post-frontier release-blocker 语境：当前阶段为 `release`，`release_action_count=0`，仍存在 active blocker，已有至少 `2` 个有效 filled slot，且父类候选为 `release_blocker` 低 tier、off-agenda 或无 active-agenda 进展。A43 采用与 A40 同构的 bounded local option evaluator，每个候选 first edge 后使用真实 legal push 展开最多 `release_blocker_option_max_pushes=5` 步、最多 `release_blocker_option_max_nodes=96` 个局部节点；它只用局部评价选择 root first edge，不执行 suffix。

实现中曾检查更宽松的 release-blocker 进展谓词。该口径会把 `slot_schedule_cost_delta=1` 的弱 slot 变化误判为 `release_blocker` 进展，即使没有打开 release、没有减少 dependency、target blocker 或 locked blocker。为保持与 A42 暴露的真实缺口一致，正式 A43 将 release-blocker 真进展收紧为：目标推进、打开 release action、减少 active blocker、减少 dependency/target blocker、或减少 locked/goal-locked stand blocker。单纯 slot cost 微降不再构成 release-blocker true progress。

A43 soft 30-push smoke 输出 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A42/A43 首个决策差异为 `None`，`release_blocker_option_override_count=0`，原因均为 `parent_not_release_blocker_trigger`。这说明 A43 不会在早期前缀阶段全局接管。

A43 soft 80-push 同参复核输出 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a43_release_blocker_option_soft_l002_s80_b128_n8.md)。A43 与 A42 完全同轨迹：`max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A42/A43 首个决策差异为 `None`。release-blocker option 字段加载 `51` 次，其中 `31` 次为 `parent_not_release_blocker_trigger`，step `31` 至 step `50` 的 `20` 次触发均为 `no_high_tier_release_blocker_option`，`release_blocker_option_override_count=0`。

因此，A43 是负诊断而非闭环修复。它说明 A42 后段的停滞不是因为已有短程真实 release-blocker option 没有被 root 选择消费；在当前 `5` push、`96` node 的 bounded option tree 内，未发现能打开 release 或减少 blocker 的 high-tier option。下一缺口更可能位于更长程/分阶段里程碑、agenda state 持久化、状态表达中的 blocker 分解，或需要把 release-blocker option 扩展到更高层的 agenda-node/option-node 结构，而不是继续调 A43 的 root score 或把弱 slot 变化当作 release-blocker 进展。

#### A44 active-blocker obligation option

A44 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner`，继承 A43 的 `RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的 active obligation、blocker-specific relation、agenda-option tree 和 agenda state persistence。A44 不训练新 checkpoint，不在线调用 teacher/reference，不改变 exact grid-flow 转移，不修改 Boxoban 主线，也不启用 learned root override。

A44 的核心改动是把 A43 的 aggregate release-blocker 计数拆成具体 A11 schedule dependency row。每个可选义务保留 `room_id`、`slot_cell`、`stand_cell`、`target_cell`、`blocker_cells`、`dependency_kinds` 与 `future` 标记，并以 `stand_cell_occupied_by_box`、`stand_cell_unreachable_by_player`、`target_occupied_by_box` 作为可审计的 selected dependency kinds。触发条件仍限制在 post-frontier release-blocker 语境：当前阶段为 `release`，`release_action_count=0`，仍存在 active blocker，已有至少 `2` 个有效 filled slot，且父类候选为 `release_blocker` 低 tier 或 off-agenda。每个候选 root edge 后，A44 对最多 `4` 个具体 blocker obligation 分别运行 bounded option evaluator，默认 `active_blocker_obligation_max_pushes=6`、`active_blocker_obligation_max_nodes=128`、`active_blocker_obligation_max_root_candidates=12`。

A44 的真进展谓词不再使用 slot cost 微降，而是直接继承 A15 dependency-bound 语义：dependency row 消失、selected kind 减少、target blocker 解除、stand 可达性恢复、selected blocker cell 离开被阻塞位置。若只改善合法动作余量但没有推进当前 dependency row，则仍记为低 tier。该版本仍只用局部评价选择 root first edge，不执行 suffix，因此它检验的是“具体 blocker 子义务是否可见、是否被当前 root 选择消费”，不是完整 option commitment。

A44 soft 30-push smoke 输出 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A43/A44 首个决策差异为 `None`，`active_blocker_obligation_override_count=0`，原因均为 `parent_not_active_blocker_trigger`。这说明 A44 不会在早期前缀阶段全局接管。

A44 soft 80-push 同参复核输出 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a44_active_blocker_obligation_soft_l002_s80_b128_n8.md)。A44 与 A43 完全同轨迹：`max_steps,80 pushes,51 decisions,boxes_on_goals=2,final_slot_cost=35,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A43/A44 首个决策差异为 `None`。A44 字段加载 `51` 次，其中 `31` 次为 `parent_not_active_blocker_trigger`，post-frontier 段触发 `20` 次；原因分布为 `{parent_active_blocker_obligation_already_best_tier:5,no_high_tier_active_blocker_obligation_option:15}`，`active_blocker_obligation_override_count=0`。

关键新证据是：A44 在 `5` 个触发点发现了 blocker-specific true progress，但父类当前 first edge 已达到同一 high tier，因而没有发生 root override。例如 step `31` 和 step `32` 中，future dependency row `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6]` 的候选 suffix 能使 `remaining_selected_kind_delta=1` 且 `stand_resolved=true`，被评为 tier `8`。这说明 A43 的 aggregate option 负结果并不等于 SD 端完全缺少 blocker identity；A44 已经能把具体 blocker 子义务读出并在短程 suffix 中看到局部完成路径。

因此，A44 的结论是“可见但未持久执行”。它没有提高目标数，也没有降低最终 dependency/target blocker；但它把缺口从“是否缺少 blocker-specific relation”进一步收窄为：具体 blocker obligation 已可审计可见，部分 suffix 也可产生真实子义务进展，但闭环仍只执行 first edge，随后 replanning 不能保持同一 blocker obligation 和 suffix 承诺。后续应优先进入 A45 级别的 active-blocker option commitment 或将 agenda/option 写入闭环状态，而不是继续调 A44 的 root 分数、放宽真进展口径或把更多局部条件堆成规则。

#### A45 active-blocker option commitment

A45 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner`，继承 A44 的 `RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的 active obligation、agenda-option tree 和 agenda state persistence。A45 不训练新 checkpoint，不在线调用 teacher/reference，不改变 exact grid-flow 转移，不修改 Boxoban 主线，也不启用 learned root override；它只消费 A44 已经识别出的 blocker-specific high-tier suffix。

A45 的承诺条件保持保守：A44 的 `selected_option` 必须存在，`active_blocker_obligation_tier>=8`，默认要求 `active_blocker_true_progress=true`，且 suffix 至少包含 `2` 个 push。若父层 `corridor_macro` 已经包含多步宏，A45 默认放弃承诺，以避免与既有宏动作混叠。触发前会在当前真实状态下逐步重放 A44 给出的 `best_actions`，检查每个 push 的合法性、deadlock child 和 path loop，并在最终状态重新计算 `_dependency_bound_metrics`、`_dependency_bound_score` 与 active-blocker progress；只有最终仍满足高 tier 和 true progress 时，才把该 suffix 写入 `corridor_macro.physical_actions` 执行。默认参数为 `active_blocker_commitment_min_tier=8`、`active_blocker_commitment_require_true_progress=true`、`active_blocker_commitment_max_extra_pushes=5`、`active_blocker_commitment_min_total_pushes=2`。

A45 soft 30-push smoke 输出 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A44/A45 首个决策差异为 `None`，`active_blocker_commitment_triggered_count=0`，原因均为 `no_selected_active_blocker_option`。这说明 A45 不会在早期尚未出现 active-blocker obligation 的阶段强行执行后缀。

A45 soft 80-push 同参复核输出 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a45_active_blocker_commitment_soft_l002_s80_b128_n8.md)。A45 为 `max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`。相对 A44，闭环决策数从 `51` 降到 `47`，A44/A45 首个决策差异为 step `32`；A45 commitment 加载 `47` 次，触发 `2` 次，首个触发为 step `31`，额外执行 `4` 个承诺 push，验证 push 总数为 `6`。

关键正证据是 step `31` 与 step `41`。step `31` 中，A45 将 A44 已识别的 active-blocker suffix 执行成四步承诺，最终 progress 为 `active_blocker_obligation_tier=8,active_blocker_true_progress=true,active_blocker_progress_kinds=[selected_kind_reduced,stand_resolved]`；step `41` 再次执行两步承诺，最终同样达到 tier `8` 和 true progress。这说明 A44 暴露出的 blocker-specific suffix 并非仅存在于离线评价中，A45 可以在闭环中合法重放并产生真实子义务进展。

但 A45 仍未求解第二关。80-push 结束时仍为 `2` goals，`dependency_count=2`、`future_dependency_count=1`、`target_blocker_count=1`、`release_action_count=0`，且 final slot cost 从 A44 的 `35` 变为 `36`。因此 A45 的结论是：active-blocker suffix commitment 是可执行的，但不是充分修复。当前失败不能再简单归因于“后缀完全没有被持久执行”；下一缺口转向 commitment 后的新阶段/新义务选择、active obligation 状态更新、以及更高层 agenda-node/option-node/push-node 结构，而不是继续增加局部 blocker 规则或放宽 true-progress 口径。

#### A46 residual active-agenda continuation

A46 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner`，继承 A45 的 `RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的第 6 点、第 11.2 点和第 14 点：A45 承诺若只完成 `stand_resolved` 而留下同一 blocker relation 的 `target_occupied_by_box`，A46 将该残余项重建为 target-only residual agenda，并在同一宏边界后运行一次有界 option 搜索。

A46 不训练新 checkpoint，复用 A35 multi-head checkpoint，不在线调用 teacher/reference，不改变 exact transition，不修改 Boxoban 主线，也不启用 learned root override。默认 residual 搜索参数如下：

```text
a46_planner = room_schedule_residual_active_agenda_macro_gft_sdmcts_v1
residual_active_agenda_max_pushes = 8
residual_active_agenda_max_nodes = 192
residual_active_agenda_min_total_pushes = 1
residual_active_agenda_require_target_progress = true
sd_value_consumer = residual_active_agenda_multihead_soft_prior_v1
agenda_gft_alignment = E6_E11_E14_residual_active_agenda
teacher_actions_used_online = false
mainline_mutation = false
learned_sd_consumer_root_override = false
```

残余义务只在 A45 commitment 已触发，且最终 `final_progress.unresolved_target=true`、最后一个 commitment row 的 `remaining_selected_kinds` 仍包含 `target_occupied_by_box` 时生成。A46 将原 A11 dependency relation 保持不变，但把 `selected_dependency_kinds` 收缩为 `["target_occupied_by_box"]`。只有 residual option 使 dependency row 消失、target blocker 解除或 selected blocker cell 离开时，才允许追加到原宏动作；若没有 target progress，则动作保持 A45 原样，只记录负证据。

A46 soft 30-push smoke 输出 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，`residual_active_agenda_reason_counts={'source_commitment_not_triggered': 14}`，说明早期尚无 A45 commitment 时不会接管。

A46 soft 80-push 同参复核输出 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a46_residual_active_agenda_soft_l002_s80_b128_n8.md)。A46 与 A45 完全同轨迹：`max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A45/A46 首个决策差异为 `None`。

关键负证据是：A46 在 step `31` 和 step `41` 都正确重建了 target-only residual agenda，合计评估 `384` 个 residual option 节点，单点均达到 `expanded_nodes=192` 上限，但 `residual_active_agenda_triggered_count=0`、`residual_active_agenda_target_progress_count=0`，原因分布为 `{'source_commitment_not_triggered': 45, 'no_target_progress_residual_option': 2}`。这说明当前失败不能再简化为“承诺后没有重建 target 残余义务”；在该有界 residual option 空间中，target blocker 的下一里程碑仍不可见。后续若继续，应转向更一般的 agenda-node/option-node/push-node 搜索拓扑、target residual 的中间里程碑定义或可复用函数近似，而不是继续重复 A45 式局部 suffix 执行。

#### A47 target-prerequisite agenda

A47 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner`，继承 A46 的 `RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的 obligation frontier、每个义务的 blocker set、active agenda state 和 agenda-option tree。A47 不训练新 checkpoint，复用 A35 multi-head checkpoint，不在线调用 teacher/reference，不改变 exact grid-flow 转移，不修改 Boxoban 主线，也不启用 learned root override。

A47 的方法目标是把 A46 暴露出的 target residual failure 再分解一层。若 A46 在 target-only residual agenda 中找不到 target progress，A47 会检查 `target_cell` 上的 blocker box 是否有合法 push。若该 blocker 没有合法 push，且存在一个可清空 target residual 的方向，其 push target 是空地板、但所需 player stand cell 被另一个箱子占用，则构造 `ClearTargetPushStand` 前提义务。该义务保留 parent residual obligation、target blocker cell、blocker push direction、required stand cell、stand blocker box index、stand blocker 是否为 parent assigned box，以及四个方向的 target blocker push candidates。完成谓词为 `target_blocker_moved`、`target_blocker_push_legal` 或 `target_push_stand_empty_reachable`。

A47 的默认参数如下：

```text
a47_planner = room_schedule_residual_prerequisite_agenda_macro_gft_sdmcts_v1
target_prerequisite_agenda_max_pushes = 8
target_prerequisite_agenda_max_nodes = 256
target_prerequisite_agenda_min_total_pushes = 1
target_prerequisite_agenda_require_progress = true
sd_value_consumer = target_prerequisite_agenda_multihead_soft_prior_v1
agenda_gft_alignment = E3_E6_E11_E14_target_prerequisite_agenda
teacher_actions_used_online = false
mainline_mutation = false
learned_sd_consumer_root_override = false
```

若 bounded prerequisite search 找到真实前提进展，A47 会将该 suffix 追加到宏动作；否则动作保持 A46 原样，只记录负证据。真实前提进展要求 target blocker 已移动、target blocker push 已合法，或 required stand cell 已清空且可达。单纯合法动作数改善不构成前提完成。

A47 soft 30-push smoke 输出 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A46/A47 首个决策差异为 `None`，`target_prerequisite_agenda_obligation_count=0`，原因分布为 `{'source_not_target_residual_failure': 14}`。这说明 A47 没有在早期尚未出现 target residual failure 的阶段接管。

A47 soft 80-push 同参复核输出 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a47_target_prerequisite_agenda_soft_l002_s80_b128_n8.md)。A47 与 A46 完全同轨迹：`max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A46/A47 首个决策差异为 `None`。

关键证据是：A47 在 step `31` 和 step `41` 都构造出 `ClearTargetPushStand` 前提义务。两次义务均指向同一 relation：parent residual 为 `slot_cell=[2,1],stand_cell=[1,6],target_cell=[3,6],candidate_direction=down`；target blocker 位于 `[3,6]`，向下推动该 blocker 需要 stand cell `[2,6]`，而 `[2,6]` 被 parent assigned box 占用。四个方向的 candidate 诊断显示：向上会撞到 `[2,6]` 的箱子，向下缺少 stand，左右为墙。两次 prerequisite search 均达到 `expanded_nodes=256` 上限，合计评估 `512` 个 option，但 `target_prerequisite_agenda_triggered_count=0`、`target_prerequisite_agenda_committed_push_count=0`，原因分布为 `{'source_not_target_residual_failure': 45, 'no_target_prerequisite_progress_option': 2}`。

因此，A47 的结论是“前提义务可见但不可短程完成”。它进一步排除了“target residual 只是没有被分解到前提 blocker set”的解释：前提 blocker set 已经能从当前状态和 residual relation 中读出，但在 `8` push、`256` node 的短程 option 空间中仍不能清空 required stand。下一步应继续沿专家 27 点推进更高层的 agenda-node/option-node/push-node 拓扑、阶段性 milestone 或可复用 agenda-distance/value 表达，而不是把 `[2,6]` 等局部坐标写成规则。

#### A48 assignment-conflict agenda

A48 继续只作用于 `room_gft_*` 派生难题线。规划侧新增 `RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner`，继承 A47 的 `RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner`。该版本对应专家 27 点中的 obligation frontier、blocker set、active agenda state 和 agenda-option tree。A48 不训练新 checkpoint，复用 A35 multi-head checkpoint，不在线调用 teacher/reference，不改变 exact-grid transition，不修改 Boxoban 主线，也不启用 learned root override。

A48 的方法目标是把 A47 中 `required_stand_blocker_is_parent_assigned_box=true` 的情形提升为 assignment/prerequisite conflict，而不是写成局部坐标规则。若 A47 的 `ClearTargetPushStand` 义务失败，且 required stand blocker 正是 parent residual 中分配给 slot 的 assigned box，则 A48 构造 `ResolveAssignedBoxPrerequisiteConflict` 义务。该义务保留 assigned box cell、assigned slot cell、required stand cell、target blocker cell、assigned box 到 slot 的 pull-distance、assigned box 四个方向的 push candidates、当前合法 push 数和朝 slot 合法 push 数。完成谓词为 required stand 清空且可达、target blocker push 合法，或 assigned box 离开 required stand 并解除占用。

A48 的默认参数如下：

```text
a48_planner = room_schedule_prerequisite_assignment_conflict_macro_gft_sdmcts_v1
assignment_conflict_agenda_max_pushes = 8
assignment_conflict_agenda_max_nodes = 320
assignment_conflict_agenda_min_total_pushes = 1
assignment_conflict_agenda_require_done = true
sd_value_consumer = assignment_conflict_agenda_multihead_soft_prior_v1
agenda_gft_alignment = E3_E6_E11_E14_assignment_conflict_agenda
teacher_actions_used_online = false
mainline_mutation = false
learned_sd_consumer_root_override = false
```

A48 soft 30-push smoke 输出 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.json) 与 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s30_b64_n8.md)。结果为 `max_steps,30 pushes,14 decisions,boxes_on_goals=1,final_slot_cost=48`，A47/A48 首个决策差异为 `None`，`assignment_conflict_agenda_obligation_count=0`，原因分布为 `{'source_not_target_prerequisite_failure': 14}`。

A48 soft 80-push 同参复核输出 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.json) 与 [room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.md](/home/data/round/code/sd_world_model_rl/docs/room_gft_xsokoban_a48_assignment_conflict_agenda_soft_l002_s80_b128_n8.md)。A48 与 A47 完全同轨迹：`max_steps,80 pushes,47 decisions,boxes_on_goals=2,final_slot_cost=36,dependency_count=2,future_dependency_count=1,target_blocker_count=1,release_action_count=0`，A47/A48 首个决策差异为 `None`。

关键证据是：A48 在 step `31` 和 step `41` 都构造出 assignment-conflict obligation。两次义务均显示 assigned box `[2,6]`、assigned slot `[2,1]`、required stand `[2,6]`、target blocker `[3,6]`，assigned box 到 slot 的 pull-distance 为 `9.0`，assigned box 初始合法 push 数为 `0`，朝 slot 的合法 push 数也为 `0`。两次 assignment-conflict search 均达到 `expanded_nodes=320` 上限，合计评估 `640` 个 option；最佳 option 仍为 tier `2`，没有 assigned box mobility opened、没有 assigned box toward-slot opened、没有 pull-distance 改善、没有 conflict done。因此 `assignment_conflict_agenda_triggered_count=0`、`assignment_conflict_agenda_committed_push_count=0`，原因分布为 `{'source_not_target_prerequisite_failure': 45, 'no_assignment_conflict_done_option': 2}`。

因此，A48 的结论是“assignment/prerequisite 冲突可见但短程不可解除”。它进一步排除了“只要识别 assigned box 与 prerequisite 的冲突即可推进”的解释：冲突关系、assigned box 移动性和朝 slot 方向的距离都已读出，但当前局部 option 空间仍不能打开该 assigned box。后续应转向更高层的 agenda-node/option-node/push-node 拓扑、阶段性 milestone、或可复用 agenda-distance/value 表达，特别是需要表达和复用“先释放 assigned box 的移动性，再清空 prerequisite stand，再释放 target blocker”的多阶段经验。
