# XSokoban first 第 1 关 Room-GFT A1/A2 诊断

## 诊断结论

本轮建立了 `SokobanRoomGFTSDModel` 的只读派生诊断线。它继承当前 `SokobanGFTSDModel`，不修改 Boxoban 主线 planner、默认评估入口或既有结果文件。
A1 诊断使用 `feature_max_abs_z>3` 作为 OOD 阈值。保存 trace 的 `20` 个被选后继中，OOD 状态为 `20` 个，OOD 且 TVA 给出正向修正的状态为 `3` 个；其中当前 alpha 实际大于 0 的正向 OOD 修正为 `3` 个。因此，A1 的直接作用是切断少量但危险的分布外正向补偿；它本身不足以生成新的房间填充策略。
A2 诊断识别出 `1` 个 goal room。第一个 room 含 `6` 个 slot、`3` 个 entry/support cell。保存 trace 中物理 `child_count=1` 的 root 为 `12/20`；语义 child 代理的中位数为 `1.5`、最大值为 `12`。这说明 A3 需要把 room-slot commitment 真正并入 MCTS 节点；仅做只读诊断不会自动增加搜索分支。

## 静态 room-slot 结构

| room | entries | slots | frontier rule proxy |
|---:|---:|---:|---|
| 0 | 3 | 6 | [6, 16] waits [[6, 17]]; [7, 16] waits [[7, 17]]; [8, 16] waits [[8, 17]] |

## A1：TVA OOD 正修正审计

| 指标 | 数值 |
|---|---:|
| candidate states | 34 |
| candidate OOD states | 34 |
| candidate positive TVA on OOD | 6 |
| selected OOD states | 20 |
| selected positive TVA on OOD | 3 |
| selected active positive TVA on OOD | 3 |
| selected z mean / max | 20.368 / 22.177 |

## A2：room-order 与承诺代理

| 指标 | 数值 |
|---|---:|
| frontier fill count | 0 |
| non-frontier fill count | 1 |
| uncommitted entry transitions | 2 |
| uncommitted deep pushes | 0 |
| room-order debt positive delta sum | 1 |
| blocked-slot positive delta sum | 1 |
| selected P-down/Q-up steps | 1 |

## 前缀逐步摘要

| step | selected | physical children | semantic proxy | impact codes | z | alpha | safe reason | room delta |
|---:|---|---:|---:|---|---:|---:|---|---|
| 0 | b5:up | 3 | 3 | none | 22.137 | 0.6 | ood_positive_suppressed | 0 |
| 1 | b5:right | 4 | 4 | none | 21.983 | 0.6 | current_rule_retained | 0 |
| 2 | b2:right | 2 | 2 | none | 22.177 | 0.6 | current_rule_retained | 0 |
| 3 | b0:down | 3 | 3 | none | 22.023 | 0.6 | current_rule_retained | 0 |
| 4 | b0:down | 1 | 1 | none | 21.787 | 0.6 | current_rule_retained | 0 |
| 5 | b1:down | 1 | 1 | none | 21.53 | 0.6 | current_rule_retained | 0 |
| 6 | b5:right | 1 | 1 | none | 21.284 | 0 | current_rule_retained | 0 |
| 7 | b5:right | 1 | 1 | none | 21.048 | 0 | current_rule_retained | 0 |
| 8 | b5:right | 1 | 1 | none | 20.76 | 0 | current_rule_retained | 0 |
| 9 | b5:right | 2 | 2 | none | 20.514 | 0 | current_rule_retained | 0 |
| 10 | b5:right | 2 | 2 | none | 20.268 | 0 | current_rule_retained | 0 |
| 11 | b5:right | 2 | 2 | none | 20.052 | 0 | current_rule_retained | 0 |
| 12 | b5:right | 1 | 1 | none | 19.806 | 0 | current_rule_retained | 0 |
| 13 | b5:right | 1 | 1 | none | 19.559 | 0 | current_rule_retained | 0 |
| 14 | b5:right | 1 | 1 | none | 19.313 | 0 | current_rule_retained | 0 |
| 15 | b5:right | 1 | 1 | none | 19.067 | 0 | current_rule_retained | 0 |
| 16 | b5:right | 1 | 1 | none | 19.205 | 0.25 | ood_positive_suppressed | 0 |
| 17 | b5:right | 1 | 3 | CROSS_ROOM_BOUNDARY,ENTER_ROOM_ENTRY,INCREASE_UNCOMMITTED_ROOM_OR_ENTRY,UNCOMMITTED_ENTRY | 18.543 | 0 | current_rule_retained | uncommitted_entry:1 |
| 18 | b5:down | 4 | 12 | ENTER_ROOM_ENTRY,UNCOMMITTED_ENTRY | 18.272 | 0 | current_rule_retained | 0 |
| 19 | b5:right | 1 | 3 | BLOCK_UNFILLED_SLOT,FILL_NONFRONTIER_SLOT,INCREASE_ROOM_ORDER_DEBT | 18.026 | 0.25 | ood_positive_suppressed | k_blocked:1,q_settled:1,uncommitted_entry:-1 |

## 资源变化

| resource | step 0 root | step 20 after | delta |
|---|---:|---:|---:|
| `P` | 97.35 | 81.45 | -15.9 |
| `M` | 3 | 2 | -1 |
| `A` | 3 | 2 | -1 |
| `K` | 0 | 0 | 0 |
| `D` | 2.75 | 2.877 | 0.127 |
| `Q` | 0 | 2 | 2 |
| `C` | 0 | 0.123 | 0.123 |

## 方法含义

A1 是必要的安全门槛：在 XSokoban 这种明显越界状态上，TVA 不应允许分布外正向修正覆盖显式房间风险。A2 则表明，当前失败 trace 已经可以被解释为 room-slot frontier、无承诺 entry 和顺序债务问题。下一步 A3 应在新的 hard-instance 派生 planner 中把物理 child 扩展为 `(box,direction,slot)` 语义 child，并把这些承诺状态纳入节点 identity，而不是继续只在物理状态上合并。
