# XSokoban first 第 1 关 Room-GFT A5 loop guard 诊断

## 核心结论

A5 在 A4 槽位顺序进度基础上加入 committed-box loop guard：对承诺槽位距离回退、直接反向推动和重复承诺状态施加软惩罚。该 planner 为 `RoomLoopGuardGFTSDMCTSPlanner`，仍只属于 hard-instance 派生线。
A4 smoke 状态为 `max_steps`，执行 `40` 步后 `boxes_on_goals=0`；A5 smoke 状态为 `no_action`，执行 `22` 步后 `boxes_on_goals=1`。二者首个动作差异发生在 step `11`。
A4 选择轨迹中 distance-regress 步数为 `15`，direct-reverse 步数为 `3`；A5 对应为 `0` 和 `0`。该结果用于检查 loop guard 是否实际进入动作选择，而非只改变未选 child 的 prior。

## A4/A5 摘要

| planner | status | steps | boxes_on_goals | final slot cost | loop penalty steps | distance regress | direct reverse | repeated commitment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A4 | max_steps | 40 | 0 | 41 | 16 | 15 | 3 | 0 |
| A5 | no_action | 22 | 1 | 46 | 0 | 0 | 0 | 0 |

## A5 轨迹前缀

| step | action | commitment | goals | slot cost | loop score | regress | reverse | repeated |
|---:|---|---|---:|---:|---:|---:|---|---|
| 0 | b3:left | none | 0 | 44 | 0 | 0 | False | False |
| 1 | b5:up | none | 0 | 45 | 0 | 0 | False | False |
| 2 | b5:right | none | 0 | 44 | 0 | 0 | False | False |
| 3 | b5:right | none | 0 | 43 | 0 | 0 | False | False |
| 4 | b5:right | none | 0 | 42 | 0 | 0 | False | False |
| 5 | b5:right | none | 0 | 41 | 0 | 0 | False | False |
| 6 | b5:right | none | 0 | 40 | 0 | 0 | False | False |
| 7 | b5:right | none | 0 | 39 | 0 | 0 | False | False |
| 8 | b4:up | none | 0 | 40 | 0 | 0 | False | False |
| 9 | b5:right | slot3@[7, 17] | 0 | 39 | 0 | 0 | False | False |
| 10 | b5:right | slot3@[7, 17] | 0 | 38 | 0 | 0 | False | False |
| 11 | b5:right | slot3@[7, 17] | 0 | 37 | 0 | 0 | False | False |
| 12 | b5:right | slot3@[7, 17] | 0 | 36 | 0 | 0 | False | False |
| 13 | b5:right | slot3@[7, 17] | 0 | 35 | 0 | 0 | False | False |
| 14 | b1:up | none | 0 | 35 | 0 | 0 | False | False |
| 15 | b0:down | none | 0 | 35 | 0 | 0 | False | False |
| 16 | b5:right | slot3@[7, 17] | 0 | 34 | 0 | 0 | False | False |
| 17 | b5:right | slot3@[7, 17] | 0 | 35 | 0 | 0 | False | False |
| 18 | b5:right | slot3@[7, 17] | 1 | 32 | 0 | 0 | False | False |
| 19 | b5:right | slot3@[7, 17] | 1 | 46 | 0 | 0 | False | False |
| 20 | b0:down | none | 1 | 46 | 0 | 0 | False | False |
| 21 | b1:down | none | 1 | 46 | 0 | 0 | False | False |
| 22 | no_action | NA | NA | NA | NA | NA | NA | NA |

## 方法含义

A5 的目标是处理 A4 smoke 暴露出的走廊内承诺箱子来回推动，而不是改变 room-slot 静态结构。若 A5 仍不能推进到 goal，则说明仅靠软 loop guard 不足，需要进一步引入 corridor contraction、forced-corridor macro 或参考解锚定。
