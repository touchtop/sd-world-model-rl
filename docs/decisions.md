# Decisions

## 决策 1

**决策**  
将 Hopper 当前主线的研究目标正式拆分为 `model fidelity` 与 `planning usefulness` 两条线。

**背景**  
当前 `SD transition core` 的监督指标、在线 `consistency_loss`、policy replay 误差与闭环行为形态已经显示：动力学拟合更准确，并不自动意味着规划质量同步提升；反之，闭环 return 改善也不自动意味着 `SD` 已被良好拟合。

**选项**  
1. 继续把 `SD` 拟合质量与规划质量视为同一问题。  
2. 在分析与验证口径上显式拆分二者。

**结论**  
采用选项 2。

**影响**  
后续 Hopper 文档、实验汇报与诊断脚本应分别回答：

1. 模型在 replay 分布上的动力学保真度如何；
2. 模型对动作排序、规划稳定性与样本效率的帮助如何。

## 决策 2

**决策**  
若后续继续修改训练方式，优先采用“共享 transition core + 独立在线 SD 监督支路”，而不是设计两个竞争的 dynamics 头。

**背景**  
当前在线 RL 联合损失会把 world model 拉向“对当前策略有用”的近似，而不会自动维护 `SD` 在所有关键相位上的短视界精度。单靠继续拉长 RL 预算，未能保证 `consistency_loss` 与关键相位误差持续改善。

**选项**  
1. 保持现有纯联合损失，不额外约束 `SD`。  
2. 构造两个独立 dynamics 头，一个服务监督拟合，一个服务规划。  
3. 保持单一共享 dynamics core，并增加独立的在线 `SD` 监督更新路径。

**结论**  
采用选项 3。

**影响**  
后续若落地实现：

1. planner 与 world model 接口仍保持唯一；
2. `SD` 的短视界监督目标可被显式维护；
3. 训练解释上可清楚区分“联合 RL 目标”与“独立 dynamics fidelity 目标”。

## 决策 3

**决策**  
后续 Hopper 线不再把“继续加长 RL 步数”作为改善 `SD` 精度的首选手段。

**背景**  
当前 `seed3` 接续实验中，`consistency_loss` 在中期后反弹；`50k -> 75k` 的 replay 误差没有表现出稳定持续改善；而闭环行为改善主要表现为“后仰 -> 前倾推进”的恢复，而非 `hop` 形成。

**选项**  
1. 继续主要依赖更长 RL 训练预算。  
2. 优先修改训练目标与诊断策略，再决定是否加长预算。

**结论**  
采用选项 2。

**影响**  
下一阶段重点应放在：

1. 在线 `SD` 监督支路设计，
2. 关键相位加权误差，
3. 同分布 replay 诊断与动作排序诊断，

而不是先投入更大规模闭环训练。

## 决策 4

**决策**  
第一版在线 `SD` 监督支路先采用“单一共享 transition core + 独立优化器 + 1-step 与当前视界 rollout 监督”的最小实现。

**背景**  
方法论上已经确定，不应引入两个相互竞争的 dynamics 头；但若一次性同时改 replay 接口、长视界监督和 planner 接线，变量会过多，难以判断训练异常究竟来自目标设计还是工程实现。

**选项**  
1. 直接实现长视界、多采样路径的完整在线监督版本。  
2. 先保持当前 replay 采样接口不变，仅在共享 `SD transition core` 上追加独立在线监督更新。  
3. 改为双 dynamics 头，分别服务 RL 与监督拟合。

**结论**  
采用选项 2。

**影响**  
当前实现具有以下边界：

1. 监督视界先限制为 `1-step + 3-step`；
2. 关键相位优先覆盖 `landing / takeoff / near_contact / high_omega`；
3. 后续如需 `5-step` 或更长视界，应单独修改 replay 采样接口，而不是在本轮最小实现中混入。

## 决策 5

**决策**  
Hopper 主线保留 `sd_latent_dim=512`，并将 TD-MPC2 主 `consistency_loss` 拆成结构态块与 learned residual latent 块，默认按块等权合成。

**背景**  
显式 17 维 `SD` 结构态不应被要求完整覆盖 Hopper 的所有复杂动力机制，因此需要保留 learned residual latent 作为补偿子空间。但如果继续对完整 512 维 latent 直接做统一均方误差，495 维 residual 块会因维度数量远大于 17 维结构态而主导一致性损失。

**选项**  
1. 将 `sd_latent_dim` 固定为 `17`，完全关闭外层 residual latent。  
2. 保留 `sd_latent_dim=512`，但仍按完整 latent 维度直接做统一均方误差。  
3. 保留 `sd_latent_dim=512`，将结构态块和 residual 块分开计损失，并用显式块级权重合成。

**结论**  
采用选项 3。默认设置为 `sd_latent_consistency_struct_weight=1.0` 与 `sd_latent_consistency_resid_weight=1.0`。

**影响**  
该决策保留 residual latent 对未建模动力机制的补偿能力，同时避免高维 residual 因维度数量自然压过显式结构态。结构态块内部不使用人工固定通道系数，而是通过 `sd_struct_loss_scale_mode=batch_std` 依据 replay batch 目标结构态的逐通道标准差归一化；`sd_struct_loss_scale_floor=0.1` 仅作为数值稳定下限。

## 决策 6

**决策**  
下一版本先实现默认关闭的 planning-aware `SD` 样本权重入口，而不是直接把 `SD` 训练目标改为完整动作排序一致性损失。

**背景**  
当前 `pureenv_sdinit_online_sup` 与上一组 `pureenv_sdinit` 对比显示，平均结构态预测监督未必能提升强 seed 的闭环收益，说明 `SD` 保真目标与 planner 动作排序目标之间仍存在错配。但完整动作排序一致性损失需要额外构造动作 bank、真实短 rollout 标签或离线动作优劣诊断，工程变量较多。

**选项**  
1. 继续只优化平均结构态预测误差。  
2. 先根据 TD 目标或奖励目标对在线 `SD` 监督样本进行均值归一化重加权。  
3. 立即实现完整 action-ranking consistency loss。

**结论**  
采用选项 2。该入口默认关闭，仅在显式实验脚本中启用；启用后，权重按 batch 内 planning signal 均值归一化，并用 `sd_online_planning_weight_max` 截断。

**影响**  
该决策提供了低风险的第一步验证：若 planning-weighted `SD` 监督改善样本效率，则说明目标错配至少部分来自样本重要性；若无明显改善，则后续应转向完整动作排序一致性目标，而不是继续提高 `sd_online_supervision_lr`。
