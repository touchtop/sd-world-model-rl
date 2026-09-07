# XSokoban first 第 2 关 Room-GFT A19 survivable recoverability 诊断

## 核心结论

A19 继承 A18，但将可恢复性收紧为结构进展，避免把纯 slot-cost 局部改善直接当作 recoverable no-progress。
A18 对照为 `no_action`，执行 `6` 个 push 后 `boxes_on_goals=0`；A19 smoke 为 `max_steps`，执行 `40` 个 push、`17` 个闭环决策后 `boxes_on_goals=1`。
A18/A19 首个决策差异为 `0`；A19 recoverable no-progress selections 为 `1`，unrecoverable selections 为 `2`，slot-only rejected selections 为 `0`。

## A18/A19 摘要

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | recoverable | unrecoverable | slot-only rejected |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A18 | no_action | 3 | 6 | 0 | 47 | 2 | 1 | 1 | 0 | NA |
| A19 | max_steps | 17 | 40 | 1 | 41 | 2 | 1 | 1 | 2 | 0 |

## A19 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | no progress | recoverable | unrecoverable | slot-only rejected | score |
|---:|---|---:|---:|---:|---:|---|---|---|---|---:|
| 0 | b6:right | 1 | 0 | 50 | 2 | False | False | False | False | 0 |
| 1 | b2:up | 1 | 0 | 51 | 1 | False | False | False | False | 0 |
| 2 | b6:right | 4 | 0 | 47 | 2 | False | False | False | False | 0 |
| 3 | b3:up | 1 | 0 | 48 | 2 | True | False | True | False | -56 |
| 4 | b4:right | 1 | 0 | 48 | 2 | True | False | True | False | -56 |
| 5 | b7:left | 3 | 0 | 47 | 1 | False | False | False | False | 0 |
| 6 | b7:up | 6 | 0 | 43 | 1 | False | False | False | False | 0 |
| 7 | b8:left | 1 | 0 | 42 | 1 | False | False | False | False | 0 |
| 8 | b8:left | 1 | 0 | 41 | 1 | False | False | False | False | 0 |
| 9 | b8:left | 1 | 0 | 40 | 1 | False | False | False | False | 0 |
| 10 | b4:up | 3 | 1 | 46 | 2 | False | False | False | False | 0 |
| 11 | b3:down | 2 | 1 | 46 | 2 | False | False | False | False | 0 |
| 12 | b3:up | 2 | 1 | 46 | 2 | False | False | False | False | 0 |
| 13 | b8:left | 8 | 1 | 42 | 2 | True | True | False | False | 48 |
| 14 | b3:down | 2 | 1 | 42 | 2 | False | False | False | False | 0 |
| 15 | b3:up | 2 | 1 | 42 | 2 | False | False | False | False | 0 |
| 16 | b5:up | 1 | 1 | 41 | 2 | False | False | False | False | 0 |

## 方法含义

A19 检验的是：A18 的 false-positive recoverability 是否来自纯 slot-cost 局部恢复。若 A19 仍失败，则应进一步分析根选择替代动作是否缺少可执行 release 生成机制。
