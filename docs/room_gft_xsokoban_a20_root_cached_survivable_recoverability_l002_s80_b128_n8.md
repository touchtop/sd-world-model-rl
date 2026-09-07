# XSokoban first 第 2 关 Room-GFT A20 root-cached recoverability 诊断

## 核心结论

A20 继承 A19，但只在根候选层计算 survivable recoverability，并在单次闭环决策内缓存 root-candidate probe。
A19 对照为 `max_steps`，执行 `40` 个 push 后 `boxes_on_goals=1`；A20 smoke 为 `max_steps`，执行 `80` 个 push、`51` 个闭环决策后 `boxes_on_goals=2`。
A19/A20 首个决策差异为 `17`；A20 recoverable no-progress selections 为 `2`，unrecoverable selections 为 `16`。

## A19/A20 摘要

| planner | configured steps | status | decisions | pushes | goals | slot cost | dependencies | target blockers | recoverability probes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A19 | 40 | max_steps | 17 | 40 | 1 | 41 | 2 | 1 | 3 |
| A20 | 80 | max_steps | 51 | 80 | 2 | 60 | 2 | 0 | 18 |

## A20 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | no progress | recoverable | unrecoverable | score |
|---:|---|---:|---:|---:|---:|---|---|---|---:|
| 0 | b6:right | 1 | 0 | 50 | 2 | False | False | False | 0 |
| 1 | b2:up | 1 | 0 | 51 | 1 | False | False | False | 0 |
| 2 | b6:right | 4 | 0 | 47 | 2 | False | False | False | 0 |
| 3 | b3:up | 1 | 0 | 48 | 2 | True | False | True | -56 |
| 4 | b4:right | 1 | 0 | 48 | 2 | True | False | True | -56 |
| 5 | b7:left | 3 | 0 | 47 | 1 | False | False | False | 0 |
| 6 | b7:up | 6 | 0 | 43 | 1 | False | False | False | 0 |
| 7 | b8:left | 1 | 0 | 42 | 1 | False | False | False | 0 |
| 8 | b8:left | 1 | 0 | 41 | 1 | False | False | False | 0 |
| 9 | b8:left | 1 | 0 | 40 | 1 | False | False | False | 0 |
| 10 | b4:up | 3 | 1 | 46 | 2 | False | False | False | 0 |
| 11 | b3:down | 2 | 1 | 46 | 2 | False | False | False | 0 |
| 12 | b3:up | 2 | 1 | 46 | 2 | False | False | False | 0 |
| 13 | b8:left | 8 | 1 | 42 | 2 | True | True | False | 48 |
| 14 | b3:down | 2 | 1 | 42 | 2 | False | False | False | 0 |
| 15 | b3:up | 2 | 1 | 42 | 2 | False | False | False | 0 |
| 16 | b5:up | 1 | 1 | 41 | 2 | False | False | False | 0 |
| 17 | b4:up | 1 | 1 | 40 | 2 | False | False | False | 0 |
| 18 | b0:left | 1 | 2 | 49 | 2 | False | False | False | 0 |
| 19 | b4:down | 2 | 2 | 49 | 2 | False | False | False | 0 |
| 20 | b4:up | 2 | 2 | 49 | 2 | False | False | False | 0 |
| 21 | b0:down | 2 | 2 | 49 | 2 | False | False | False | 0 |
| 22 | b0:up | 2 | 2 | 49 | 2 | False | False | False | 0 |
| 23 | b7:down | 1 | 2 | 48 | 2 | False | False | False | 0 |
| 24 | b8:left | 1 | 2 | 47 | 2 | False | False | False | 0 |
| 25 | b8:left | 1 | 2 | 46 | 2 | False | False | False | 0 |
| 26 | b8:left | 1 | 2 | 45 | 2 | False | False | False | 0 |
| 27 | b4:down | 2 | 2 | 45 | 2 | False | False | False | 0 |
| 28 | b7:left | 1 | 2 | 45 | 2 | True | True | False | 48 |
| 29 | b4:down | 2 | 2 | 45 | 2 | False | False | False | 0 |
| 30 | b4:up | 1 | 2 | 66 | 2 | False | False | False | 0 |
| 31 | b7:left | 1 | 2 | 65 | 2 | False | False | False | 0 |
| 32 | b7:down | 1 | 2 | 64 | 2 | False | False | False | 0 |
| 33 | b8:left | 1 | 2 | 63 | 2 | False | False | False | 0 |
| 34 | b8:left | 1 | 2 | 62 | 2 | False | False | False | 0 |
| 35 | b9:left | 1 | 2 | 62 | 2 | True | False | True | -56 |
| 36 | b6:up | 1 | 2 | 62 | 2 | True | False | True | -56 |
| 37 | b6:up | 1 | 2 | 62 | 2 | True | False | True | -56 |
| 38 | b4:left | 1 | 2 | 61 | 2 | False | False | False | 0 |
| 39 | b4:left | 1 | 2 | 60 | 2 | False | False | False | 0 |
| 40 | b9:left | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 41 | b6:up | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 42 | b5:left | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 43 | b5:left | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 44 | b9:right | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 45 | b9:up | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 46 | b7:right | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 47 | b7:down | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 48 | b9:left | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 49 | b9:up | 1 | 2 | 60 | 2 | True | False | True | -56 |
| 50 | b7:up | 1 | 2 | 60 | 2 | True | False | True | -56 |

## 方法含义

A20 检验的是：将 recoverability probe 限制到根候选层后，是否可以把 A19 从 40-step 机制 smoke 推回 80-step 诊断。
