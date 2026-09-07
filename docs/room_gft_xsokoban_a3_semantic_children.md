# XSokoban first 第 1 关 Room-GFT A3 语义 child 诊断

## 核心结论

A3 已将物理 push 扩展为 `(box,direction,slot)` 语义 child，并把 room-slot commitment 纳入 MCTS node identity。该实现位于派生 hard-instance planner 中，不修改现有 Boxoban 主线 planner。
在旧失败 trace 的 `20` 个 root 上重放枚举时，semantic child 中位数为 `1`，最大为 `4`。旧 trace 中物理 child 为 1 但 A3 产生语义分裂的步数为 `0`；被选物理动作拥有多个 slot 选项的步数为 `1`。这表明 A3 已能在 commitment boundary 处提前展开 slot 承诺，但长走廊内部一旦承诺确定，仍会表现为单一后继；这部分需要 A4 的 slot-schedule progress 或 forced-corridor option 继续处理。
短闭环 semantic smoke 的状态为 `max_steps`，执行 `20` 步，最终 `boxes_on_goals=0`。该 smoke 只验证 A3 分支机制和承诺状态推进，不作为 XSokoban 解题率结果。

## 保存 trace 上的语义分裂

| step | saved action | physical log | semantic children | selected options | adopted commitment | impact codes |
|---:|---|---:|---:|---:|---|---|
| 0 | b5:up | 3 | 3 | 1 | none | none |
| 1 | b5:right | 4 | 4 | 1 | none | none |
| 2 | b2:right | 2 | 2 | 1 | none | none |
| 3 | b0:down | 3 | 3 | 1 | none | none |
| 4 | b0:down | 1 | 1 | 1 | none | none |
| 5 | b1:down | 1 | 1 | 1 | none | none |
| 6 | b5:right | 1 | 1 | 1 | none | none |
| 7 | b5:right | 1 | 1 | 1 | none | none |
| 8 | b5:right | 1 | 1 | 1 | none | none |
| 9 | b5:right | 2 | 2 | 1 | none | none |
| 10 | b5:right | 2 | 2 | 1 | none | none |
| 11 | b5:right | 2 | 4 | 3 | room 0 slot 1 [6, 17] | none |
| 12 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | none |
| 13 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | none |
| 14 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | none |
| 15 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | none |
| 16 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | none |
| 17 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | CROSS_ROOM_BOUNDARY,ENTER_ROOM_ENTRY,INCREASE_UNCOMMITTED_ROOM_OR_ENTRY,UNCOMMITTED_ENTRY |
| 18 | b5:down | 4 | 4 | 1 | room 0 slot 1 [6, 17] | ENTER_ROOM_ENTRY,UNCOMMITTED_ENTRY |
| 19 | b5:right | 1 | 1 | 1 | room 0 slot 1 [6, 17] | BLOCK_UNFILLED_SLOT,FILL_NONFRONTIER_SLOT,INCREASE_ROOM_ORDER_DEBT |

## 短闭环 smoke

| step | physical action | semantic children | top commitment | boxes_on_goals | committed boxes |
|---:|---|---:|---|---:|---:|
| 0 | b5:up | 3 | none | 0 | 0 |
| 1 | b3:left | 4 | none | 0 | 0 |
| 2 | b5:right | 3 | none | 0 | 0 |
| 3 | b5:right | 1 | none | 0 | 0 |
| 4 | b5:right | 1 | none | 0 | 0 |
| 5 | b5:right | 1 | none | 0 | 0 |
| 6 | b5:right | 4 | none | 0 | 0 |
| 7 | b5:right | 4 | none | 0 | 0 |
| 8 | b5:right | 5 | slot 3 [7, 17] | 0 | 1 |
| 9 | b5:right | 3 | slot 3 [7, 17] | 0 | 1 |
| 10 | b5:right | 3 | slot 3 [7, 17] | 0 | 1 |
| 11 | b5:right | 3 | slot 3 [7, 17] | 0 | 1 |
| 12 | b1:up | 3 | none | 0 | 1 |
| 13 | b2:down | 5 | none | 0 | 1 |
| 14 | b2:left | 5 | none | 0 | 1 |
| 15 | b0:down | 5 | none | 0 | 1 |
| 16 | b0:down | 4 | none | 0 | 1 |
| 17 | b2:left | 5 | none | 0 | 1 |
| 18 | b5:right | 6 | slot 3 [7, 17] | 0 | 1 |
| 19 | b2:right | 6 | none | 0 | 1 |

## 方法含义

A3 开始处理 `child_count=1` 掩盖战略分支的问题：同一物理后继可以因为不同 room-slot 承诺而进入不同节点。当前版本仍是最小机制实现，commitment 以箱子当前位置和 slot 绑定表示，并只在 room boundary 或 entry 邻域触发语义分裂；后续 A4 应把运输进展从任意 goal 距离改为 slot-schedule constrained progress，并进一步强化非 frontier 填充和无承诺 entry 的 barrier。
