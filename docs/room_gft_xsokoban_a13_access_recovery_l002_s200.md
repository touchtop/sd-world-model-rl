# XSokoban first 第 2 关 Room-GFT A13 access-recovery guard 诊断

## 核心结论

A13 保留 A11 的 schedule-plan SD 表示和 A12 的 dependency-stall guard，在 MCTS 消费端额外加入 access-recovery margin。该版本不读取 teacher 动作表。
A12 smoke 为 `no_action`，执行 `6` 个 push 后 `boxes_on_goals=0`；A13 smoke 为 `max_steps`，执行 `200` 个 push、`182` 个闭环决策后 `boxes_on_goals=0`。
A12/A13 首个决策差异为 `1`；A13 最终 slot cost 为 `29`，dependency count 为 `2`，low-legal selections 为 `61`。

## A12/A13 摘要

| planner | status | decision steps | pushes | goals | final slot cost | dependencies | target blockers | low-legal selections |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A12 | no_action | 3 | 6 | 0 | 47 | 2 | NA | NA |
| A13 | max_steps | 182 | 200 | 0 | 29 | 2 | 2 | 61 |

## A13 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | recovery score | legal delta | low legal | no legal child |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | b6:right | 1 | 0 | 50 | 2 | 2 | 1 | False | False |
| 1 | b2:up | 1 | 0 | 51 | 1 | -12 | -1 | False | False |
| 2 | b3:right | 1 | 0 | 51 | 1 | 8 | 4 | False | False |
| 3 | b8:left | 1 | 0 | 50 | 1 | -16 | -2 | False | False |
| 4 | b4:down | 1 | 0 | 50 | 1 | -8 | -1 | False | False |
| 5 | b4:left | 1 | 0 | 50 | 2 | -8 | 1 | False | False |
| 6 | b3:up | 1 | 0 | 50 | 1 | 18 | 4 | False | False |
| 7 | b3:up | 1 | 0 | 50 | 1 | 4 | 2 | False | False |
| 8 | b2:left | 1 | 0 | 50 | 1 | 20 | 2 | False | False |
| 9 | b0:down | 5 | 0 | 47 | 0 | 12 | 3 | False | False |
| 10 | b0:left | 5 | 0 | 42 | 1 | -8 | -1 | False | False |
| 11 | b0:left | 1 | 0 | 41 | 1 | 0 | 0 | False | False |
| 12 | b0:left | 5 | 0 | 36 | 2 | 0 | 0 | False | False |
| 13 | b0:down | 1 | 0 | 36 | 2 | 0 | 0 | False | False |
| 14 | b5:left | 1 | 0 | 37 | 2 | -8 | -1 | False | False |
| 15 | b8:up | 1 | 0 | 37 | 2 | -8 | -1 | False | False |
| 16 | b6:right | 1 | 0 | 36 | 2 | 2 | 1 | False | False |
| 17 | b5:down | 1 | 0 | 36 | 2 | -16 | -2 | False | False |
| 18 | b3:down | 1 | 0 | 36 | 2 | 0 | 0 | False | False |
| 19 | b4:down | 1 | 0 | 36 | 2 | -8 | -1 | False | False |
| 20 | b3:up | 1 | 0 | 36 | 2 | 0 | 0 | False | False |
| 21 | b3:up | 1 | 0 | 36 | 2 | 0 | 0 | False | False |
| 22 | b0:up | 1 | 0 | 36 | 2 | -8 | -1 | False | False |
| 23 | b0:left | 1 | 0 | 36 | 2 | 2 | 1 | False | False |
| 24 | b0:left | 1 | 0 | 36 | 2 | 0 | 0 | False | False |
| 25 | b0:left | 1 | 0 | 36 | 2 | 0 | 0 | False | False |
| 26 | b0:left | 4 | 0 | 32 | 2 | 0 | 0 | False | False |
| 27 | b3:down | 1 | 0 | 33 | 2 | -10 | 0 | False | False |
| 28 | b8:up | 1 | 0 | 33 | 2 | 8 | 4 | False | False |
| 29 | b8:up | 1 | 0 | 33 | 2 | -8 | -1 | False | False |
| 30 | b4:up | 1 | 0 | 33 | 2 | 2 | 1 | False | False |
| 31 | b4:right | 1 | 0 | 33 | 2 | -8 | -1 | False | False |
| 32 | b4:right | 1 | 0 | 33 | 2 | 4 | 2 | False | False |
| 33 | b4:up | 1 | 0 | 33 | 2 | 0 | 0 | False | False |
| 34 | b3:up | 1 | 0 | 33 | 2 | 0 | 0 | False | False |
| 35 | b0:up | 1 | 0 | 33 | 2 | -8 | -1 | False | False |
| 36 | b0:left | 1 | 0 | 33 | 2 | 2 | 1 | False | False |
| 37 | b0:left | 1 | 0 | 33 | 2 | 0 | 0 | False | False |
| 38 | b0:left | 1 | 0 | 33 | 2 | 0 | 0 | False | False |
| 39 | b0:left | 4 | 0 | 29 | 2 | 0 | 0 | False | False |
| 40 | b5:left | 1 | 0 | 29 | 2 | -16 | -2 | False | False |
| 41 | b9:left | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 42 | b9:up | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 43 | b5:down | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 44 | b5:right | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 45 | b5:down | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 46 | b9:left | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 47 | b9:up | 1 | 0 | 29 | 2 | 6 | 3 | False | False |
| 48 | b5:up | 1 | 0 | 29 | 2 | -16 | -2 | False | False |
| 49 | b5:right | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 50 | b5:up | 1 | 0 | 29 | 2 | 0 | 0 | False | False |
| 51 | b4:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 52 | b0:up | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 53 | b0:left | 1 | 0 | 29 | 2 | 0 | 0 | False | False |
| 54 | b0:left | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 55 | b0:left | 1 | 0 | 29 | 2 | 0 | 0 | False | False |
| 56 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 57 | b0:left | 1 | 0 | 29 | 2 | 0 | 0 | False | False |
| 58 | b0:left | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 59 | b0:down | 1 | 0 | 29 | 2 | -8 | -1 | False | False |
| 60 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 61 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 62 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 63 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 64 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 65 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 66 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 67 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 68 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 69 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 70 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 71 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 72 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 73 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 74 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 75 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 76 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 77 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |
| 78 | b6:down | 1 | 0 | 29 | 2 | -53 | -1 | True | False |
| 79 | b9:up | 1 | 0 | 29 | 2 | 2 | 1 | False | False |

## 方法含义

A13 检验的是：在 A12 已能识别 dependency-stall 的情况下，额外保护非阻塞合法推箱余量、target blocker 和不可达站位恢复，是否能避免过早进入无动作状态。
