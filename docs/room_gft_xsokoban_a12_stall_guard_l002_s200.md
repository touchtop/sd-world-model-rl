# XSokoban first 第 2 关 Room-GFT A12 dependency-stall guard 诊断

## 核心结论

A12 保留 A11 的 schedule-plan SD 表示，只在 MCTS 消费端加入 dependency-stall loop guard。该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定。
A11 guarded smoke 为 `max_steps`，执行 `200` 个 push 后 `boxes_on_goals=0`；A12 smoke 为 `no_action`，执行 `6` 个 push、`3` 个闭环决策后 `boxes_on_goals=0`。
A11/A12 首个决策差异为 `0`；A12 最终 slot cost 为 `47`，dependency count 为 `2`，selected stalled count 为 `1`。

## A11/A12 摘要

| planner | status | decision steps | pushes | goals | final slot cost | dependencies | stalled selections | reverse recent | repeated signatures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A11 | max_steps | 200 | 200 | 0 | 48 | 2 | NA | NA | NA |
| A12 | no_action | 3 | 6 | 0 | 47 | 2 | 1 | 0 | 0 |

## A12 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | stall score | stalled | reverse | repeated signature |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 0 | b6:right | 1 | 0 | 50 | 2 | 0 | False | False | False |
| 1 | b6:right | 4 | 0 | 46 | 2 | 0 | False | False | False |
| 2 | b7:right | 1 | 0 | 47 | 2 | -8 | True | False | False |
| 3 | no_action | 0 | NA | NA | 2 | NA | NA | NA | NA |

## 方法含义

A12 检验的是：在 A11 已有调度图的情况下，单纯加入 dependency-stall loop guard 是否足以打破第二关的二箱循环。若仍未解出，应继续转向 release macro 或 TVA 前缀校准。
