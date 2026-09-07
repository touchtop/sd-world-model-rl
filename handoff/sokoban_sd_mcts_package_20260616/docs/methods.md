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
