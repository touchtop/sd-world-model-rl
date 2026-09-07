# XSokoban first 第 2 关 Room-GFT A17 phase-progress guard 诊断

## 核心结论

A17 继承 A16 的 feasibility-guarded macro，并在根选择层加入 schedule-phase no-progress guard，用于惩罚 dependency/slot phase 不变时的重复动作和反向动作。
A16 对照为 `max_steps`，执行 `80` 个 push 后 `boxes_on_goals=0`；A17 smoke 为 `no_action`，执行 `22` 个 push、`19` 个闭环决策后 `boxes_on_goals=0`。
A16/A17 首个决策差异为 `2`；A17 phase no-progress selections 为 `14`，stagnant selections 为 `13`。

## A16/A17 摘要

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | no-progress selections |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A16 | max_steps | 78 | 80 | 0 | 50 | 1 | 1 | NA |
| A17 | no_action | 19 | 22 | 0 | 47 | 2 | 1 | 14 |

## A17 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | phase progress | no progress | stagnant | phase score |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 0 | b6:right | 1 | 0 | 50 | 2 | True | False | False | 0 |
| 1 | b2:up | 1 | 0 | 51 | 1 | True | False | False | 0 |
| 2 | b7:right | 1 | 0 | 52 | 1 | False | True | False | -24 |
| 3 | b6:right | 4 | 0 | 48 | 2 | True | False | True | 0 |
| 4 | b3:up | 1 | 0 | 49 | 2 | False | True | False | -24 |
| 5 | b4:right | 1 | 0 | 49 | 2 | False | True | True | -76 |
| 6 | b6:left | 1 | 0 | 49 | 2 | False | True | True | -76 |
| 7 | b4:up | 1 | 0 | 49 | 2 | False | True | True | -76 |
| 8 | b4:up | 1 | 0 | 49 | 2 | False | True | True | -76 |
| 9 | b2:left | 1 | 0 | 48 | 2 | True | False | False | 0 |
| 10 | b2:left | 1 | 0 | 47 | 2 | True | False | False | 0 |
| 11 | b4:up | 1 | 0 | 47 | 2 | False | True | True | -76 |
| 12 | b3:left | 1 | 0 | 47 | 2 | False | True | True | -76 |
| 13 | b6:up | 1 | 0 | 47 | 2 | False | True | True | -76 |
| 14 | b9:left | 1 | 0 | 47 | 2 | False | True | True | -76 |
| 15 | b5:left | 1 | 0 | 47 | 2 | False | True | True | -76 |
| 16 | b9:up | 1 | 0 | 47 | 2 | False | True | True | -76 |
| 17 | b3:left | 1 | 0 | 47 | 2 | False | True | True | -94 |
| 18 | b7:up | 1 | 0 | 47 | 2 | False | True | True | -94 |
| 19 | no_action | 0 | NA | NA | 2 | NA | NA | NA | NA |

## 方法含义

A17 检验的是：仅过滤宏执行结果不足以避免根选择层长循环，还需要对 schedule phase 无进展的重复动作施加直接约束。
