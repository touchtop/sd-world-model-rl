# XSokoban first 第 2 关 Room-GFT A18 recoverability guard 诊断

## 核心结论

A18 继承 A17，但对 no-progress 根候选增加短程可恢复性探测，允许无即时 phase progress 但仍能保留后续恢复机会的准备动作。
A17 对照为 `no_action`，执行 `22` 个 push 后 `boxes_on_goals=0`；A18 smoke 为 `no_action`，执行 `6` 个 push、`3` 个闭环决策后 `boxes_on_goals=0`。
A17/A18 首个决策差异为 `0`；A18 recoverable no-progress selections 为 `1`，unrecoverable selections 为 `0`。

## A17/A18 摘要

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | recoverable no-progress | unrecoverable no-progress |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A17 | no_action | 19 | 22 | 0 | 47 | 2 | 1 | NA | NA |
| A18 | no_action | 3 | 6 | 0 | 47 | 2 | 1 | 1 | 0 |

## A18 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | no progress | recoverable | unrecoverable | score |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 0 | b7:right | 1 | 0 | 52 | 2 | True | True | False | 48 |
| 1 | b6:right | 1 | 0 | 51 | 2 | False | False | False | 0 |
| 2 | b6:right | 4 | 0 | 47 | 2 | False | False | False | 0 |
| 3 | no_action | 0 | NA | NA | 2 | NA | NA | NA | NA |

## 方法含义

A18 检验的是：根选择层需要区分无即时进展的准备动作与不可恢复动作，不能只看 immediate phase delta。
