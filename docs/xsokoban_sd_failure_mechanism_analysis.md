# XSokoban first 第 1 关的 SD 模型失败机制分析

## 核心结论

当前失败主要不是由 `max_steps` 或单次 MCTS 展开预算不足造成，而是由当前 SD/GFT/TVA 决策层的结构性偏差造成：模型把多箱目标房间问题压缩为局部运输进展最大化问题，缺少可执行的 box-goal/room-order 长程承诺。
在保存的前 20 步 trace 中，规划很快进入低分支走廊状态，`child_count=1` 的 root 有 `12/20` 个；被选动作访问占比中位数为 `1`。这说明 4096 次 MCTS 展开多数时候没有在多个战略分支之间比较，而是在窄通道中反复确认同一个后继。
前缀中动作 `b5:right` 出现 `14` 次，表现为把同一运输方向持续推深。该行为能降低短期运输成本，但没有保证目标房间的填充顺序。
同时，被选后继按一阶 leaf value 排名第一的步数为 `17/20`。这说明失败不是因为 MCTS 没有服从当前价值函数，而是当前价值函数本身把错误的局部运输前缀评估得足够好。

## 当前配置暴露的关键缺口

1. **动作先验几乎只奖励运输进展**。本次配置中 root prior 权重为 `progress=1`、`goal=1`、`ordering=0.05`、`commitment=0.05`，但 `mobility/access/capacity/deadlock` 权重均为 `0`。因此，模型能计算 D/K/A/M 等资源，却不会在动作先验中强烈惩罚软死锁、容量压力或可达性恶化。
2. **叶值同样弱化房间顺序风险**。GFT leaf 只给 `capacity/order/commitment` 各很小的修正，且 `deadlock` leaf 权重为 0；目标房间顺序错误在成为硬死锁之前，不足以覆盖 progress 和 goal 的短期收益。
3. **TVA 校正没有承担 OOD 保护**。TVA 的 state-gated 默认 blend 为 0，只在早期 transport phase 或低风险正修正时介入；同时 OOD z-threshold 被关闭。保存 trace 中被选后继的 TVA 特征 `max_abs_z` 均值为 `20.368`，最大值为 `22.177`；这些状态明显超出 TVA 训练/审计尺度，但当前配置不会据此触发明确的 OOD 降级或保守策略。
4. **revisit 惩罚无法处理单调错误承诺**。重复向右推进同一箱子会产生新 box layout，不是 canonical revisit；因此 revisit 只能惩罚回环，不能阻止“持续向错误目标房间深处推进”的策略。

## 前缀资源变化

| 指标 | trace 初始 root | 第 20 步后 | 前缀变化 |
|---|---:|---:|---:|
| `P` | 97.35 | 81.45 | -15.9 |
| `M` | 3 | 2 | -1 |
| `A` | 3 | 2 | -1 |
| `K` | 0 | 0 | 0 |
| `D` | 2.75 | 2.877 | 0.127 |
| `Q` | 0 | 2 | 2 |
| `C` | 0 | 0.123 | 0.123 |

这里的关键现象是：`P` 明显下降，说明模型确实在制造运输进展；但 `Q` 从 0 升高，`D` 没有显著下降，`C` 仍很低。这是典型的“局部距离进展成立，但目标房间承诺和顺序控制失败”。

TVA 介入方式也支持这一判断：前缀中 alpha 触发计数为 `{'transport_phase': 6, 'default': 12, 'positive_delta': 2}`。前 6 步主要由 transport phase 规则介入，之后大多回到默认 alpha=0；少数 positive_delta 介入发生在已经出现顺序债务的状态上，不能提供房间顺序保护。

## 代表性 root 诊断

| step | selected | child_count | visit_share | prior_rank | leaf_rank | progress_delta | ordering_delta | deadlock_delta | leaf_alpha | alpha_reason |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | b5:up | 3 | 0.941 | 2 | 1 | -0.788 | 0 | 0 | 0.6 | transport_phase |
| 1 | b5:right | 4 | 0.943 | 2 | 4 | 0.625 | 0 | 0 | 0.6 | transport_phase |
| 2 | b2:right | 2 | 0.874 | 2 | 1 | -0.787 | 0 | 0 | 0.6 | transport_phase |
| 3 | b0:down | 3 | 0.961 | 3 | 3 | 0.625 | 0 | 0 | 0.6 | transport_phase |
| 4 | b0:down | 1 | 1 | 1 | 1 | 0.958 | 0 | 0 | 0.6 | transport_phase |
| 5 | b1:down | 1 | 1 | 1 | 1 | 1.042 | 0 | 0 | 0.6 | transport_phase |
| 6 | b5:right | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 7 | b5:right | 1 | 1 | 1 | 1 | 0.958 | 0 | 0 | 0 | default |
| 8 | b5:right | 1 | 1 | 1 | 1 | 1.167 | 0 | 0 | 0 | default |
| 9 | b5:right | 2 | 0.996 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 10 | b5:right | 2 | 0.995 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 11 | b5:right | 2 | 0.997 | 1 | 1 | 0.875 | 0 | 0 | 0 | default |
| 12 | b5:right | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 13 | b5:right | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 14 | b5:right | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 15 | b5:right | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | default |
| 16 | b5:right | 1 | 1 | 1 | 1 | -0.562 | 3 | 0 | 0.25 | positive_delta |
| 17 | b5:right | 1 | 1 | 1 | 1 | 2.688 | -3 | 0 | 0 | default |
| 18 | b5:down | 4 | 0.813 | 3 | 4 | 1.1 | 0 | 0 | 0 | default |
| 19 | b5:right | 1 | 0.909 | 1 | 1 | 1 | 2 | 0 | 0.25 | positive_delta |

## 为什么最终 1000 步仍失败

最终状态仍为 `max_steps`，只完成 `1` 个目标，最终 `P=81.45`、`M=2`、`A=2`、`D=2.877`、`Q=2`、`C=0.123`。
这说明模型没有走入硬死锁，而是停留在“仍有合法 push、但目标房间填充顺序已经不受控制”的状态。增大 `max_steps` 只能给这种局部策略更多执行时间，不能自动生成多箱房间的承诺结构。

## 直接方法含义

若继续推进 XSokoban，优先修复点不应是单纯增大闭环步数，而是让 SD 层产生可消费的长程承诺：显式 box-goal/room-slot 绑定、目标房间入口到深处的填充顺序、对错误深推的早期惩罚，以及 XSokoban 分布上的 TVA/OOD 校准。
