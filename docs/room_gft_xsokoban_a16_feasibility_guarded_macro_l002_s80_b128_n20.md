# XSokoban first 第 2 关 Room-GFT A16 feasibility-guarded macro 诊断

## 核心结论

A16 继承 A15 的 dependency-bound macro，但只有当 slot cost、target blocker、critical stand、destructive action 和合法动作余量等全局后置条件通过时才执行短宏。
A15 对照为 `no_action`，执行 `36` 个 push 后 `boxes_on_goals=0`；A16 smoke 为 `max_steps`，执行 `80` 个 push、`78` 个闭环决策后 `boxes_on_goals=0`。
A15/A16 首个决策差异为 `1`；A16 可行性保护宏触发 `1` 次，追加 `2` 个 push，最终 dependency count 为 `1`。

## A15/A16 摘要

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | guarded macro triggers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A15 | no_action | 21 | 36 | 0 | 42 | 2 | 1 | NA |
| A16 | max_steps | 78 | 80 | 0 | 50 | 1 | 1 | 1 |

## A16 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | guarded macro | macro extra | best score | feasibility | progress |
|---:|---|---:|---:|---:|---:|---|---:|---:|---|---|
| 0 | b6:right | 1 | 0 | 50 | 2 | False | 0 | NA | NA | NA |
| 1 | b2:up | 1 | 0 | 51 | 1 | False | 0 | NA | NA | NA |
| 2 | b3:right | 1 | 0 | 51 | 1 | False | 0 | NA | NA | NA |
| 3 | b8:left | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 4 | b4:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 5 | b4:left | 1 | 0 | 50 | 2 | False | 0 | NA | NA | NA |
| 6 | b3:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 7 | b3:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 8 | b2:left | 3 | 0 | 49 | 1 | True | 2 | 155.15 | pass | target |
| 9 | b2:left | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 10 | b4:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 11 | b9:left | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 12 | b4:right | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 13 | b4:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 14 | b9:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 15 | b4:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 16 | b5:right | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 17 | b9:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 18 | b6:down | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 19 | b9:left | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 20 | b9:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 21 | b4:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 22 | b5:down | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 23 | b5:right | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 24 | b5:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 25 | b5:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 26 | b5:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 27 | b9:up | 1 | 0 | 49 | 1 | False | 0 | NA | NA | NA |
| 28 | b4:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 29 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 30 | b6:left | 1 | 0 | 51 | 2 | False | 0 | NA | NA | NA |
| 31 | b6:right | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 32 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 33 | b7:left | 1 | 0 | 51 | 2 | False | 0 | NA | NA | NA |
| 34 | b7:right | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 35 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 36 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 37 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 38 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 39 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 40 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 41 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 42 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 43 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 44 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 45 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 46 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 47 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 48 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 49 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 50 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 51 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 52 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 53 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 54 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 55 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 56 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 57 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 58 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 59 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 60 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 61 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 62 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 63 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 64 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 65 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 66 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 67 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 68 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 69 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 70 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 71 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 72 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 73 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 74 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 75 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 76 | b9:up | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |
| 77 | b6:down | 1 | 0 | 50 | 1 | False | 0 | NA | NA | NA |

## 方法含义

A16 检验的是：A15 的局部 dependency row 解析必须与全局 slot-fill 可行性同时成立，否则短宏不应执行。
