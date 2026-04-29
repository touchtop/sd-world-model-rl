# Hopper Planning-Aware SD 目标准备记录（2026-04-30）

## 背景

`pureenv_sdinit_online_sup` 三种子接续实验在 `75k` 处显示：在线 `SD` 监督可以改善部分损失并使弱 seed 出现更多非零评估，但相较上一组同样去除奖励塑形的 `pureenv_sdinit`，强 seed3 的评估得分明显偏低。该现象说明，单纯压低结构态预测误差并不必然提升 TD-MPC2 的规划效果。

从第一性原理看，`SD` 状态预测与 `RL` 规划存在共同基础，即未来状态演化必须被可靠刻画；但二者的优化重点不同。`SD` 监督通常最小化平均状态误差，而 planner 需要在候选动作之间得到正确排序。因此，下一版本应从“平均预测更准”推进到“对规划更有用”。

## 本轮准备

本轮先加入最小 planning-aware 权重脚手架，不改变默认行为。新增配置如下：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

启用后，在线 `SD` 监督支路可根据 TD 目标或奖励目标对样本加权，使对 planner 更可能有价值的 replay 片段在 `1-step` 与短 rollout 结构态监督中获得更高权重。权重按当前 batch 的 signal 均值归一化，未截断前平均权重约为 `1`，以避免把该机制与整体提高 `SD` 学习率混在一起。

## 当前边界

该版本仍不是完整动作排序损失。它只完成第一步：将 `SD` 监督从完全平均误差，推进到按 planning signal 进行样本重加权。这样做的原因是：

1. 改动面较小，不需要重写 planner 或 replay 接口；
2. 默认关闭，不影响既有实验可复现性；
3. 可通过 5k smoke 快速检查是否导致梯度爆炸或训练统计异常；
4. 为后续真正的 action-ranking consistency loss 保留接口位置。

## 后续版本目标

下一步应在该脚手架上继续推进两类验证：

1. `planning-weighted SD supervision`：对比 `td_positive`、`td_abs`、`reward_positive` 三种权重源，观察是否改善 `75k` 评估与训练 episode 分布；
2. `action-ranking consistency`：在真实状态上采样动作 bank，用真实短 rollout 或离线诊断得到动作排序，再要求 SD imagined rollout 的排序与之保持一致。

若第一类权重版本已经明显改善样本效率，则说明当前瓶颈至少部分来自平均监督目标与 planning signal 的错配；若仍无改善，则应优先投入第二类排序一致性目标，而不是继续提高 `sd_online_supervision_lr`。
