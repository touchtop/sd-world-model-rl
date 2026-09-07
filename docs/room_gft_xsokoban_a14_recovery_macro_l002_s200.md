# XSokoban first 第 2 关 Room-GFT A14 recovery macro 诊断

## 核心结论

A14 保留 A11-A13 的 SD 与消费端信号，并在根动作后追加 bounded release/access recovery macro。该版本不读取 teacher 动作表。
A13 smoke 为 `max_steps`，执行 `200` 个 push 后 `boxes_on_goals=0`；A14 smoke 为 `max_steps`，执行 `200` 个 push、`73` 个闭环决策后 `boxes_on_goals=0`。
A13/A14 首个决策差异为 `1`；A14 最终 slot cost 为 `33`，recovery macro 触发 `48` 次，追加 `116` 个 push。

## A13/A14 摘要

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | recovery macro triggers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A13 | max_steps | 182 | 200 | 0 | 29 | 2 | 2 | NA |
| A14 | max_steps | 73 | 200 | 0 | 33 | 3 | 1 | 48 |

## A14 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | recovery macro | macro extra | macro score |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 0 | b6:right | 5 | 0 | 50 | 2 | True | 4 | 35 |
| 1 | b7:left | 4 | 0 | 47 | 1 | True | 3 | 46.85 |
| 2 | b3:left | 5 | 0 | 45 | 1 | True | 2 | 8.3 |
| 3 | b0:left | 5 | 0 | 40 | 3 | False | 0 | NA |
| 4 | b0:left | 1 | 0 | 40 | 3 | False | 0 | NA |
| 5 | b0:left | 5 | 0 | 35 | 3 | False | 0 | NA |
| 6 | b0:down | 1 | 0 | 35 | 3 | False | 0 | NA |
| 7 | b4:up | 3 | 0 | 35 | 3 | True | 2 | 3.5 |
| 8 | b9:right | 1 | 0 | 35 | 3 | False | 0 | NA |
| 9 | b4:left | 2 | 0 | 35 | 3 | True | 1 | 3.75 |
| 10 | b5:left | 2 | 0 | 35 | 3 | True | 1 | 7.75 |
| 11 | b5:up | 2 | 0 | 35 | 3 | True | 1 | 11.75 |
| 12 | b4:right | 3 | 0 | 35 | 3 | True | 2 | 15.5 |
| 13 | b5:left | 4 | 0 | 35 | 3 | True | 3 | 11.25 |
| 14 | b0:up | 5 | 0 | 34 | 3 | True | 4 | 3.8 |
| 15 | b0:left | 3 | 0 | 31 | 3 | False | 0 | NA |
| 16 | b4:up | 5 | 0 | 32 | 3 | True | 4 | 6.2 |
| 17 | b8:left | 1 | 0 | 32 | 3 | False | 0 | NA |
| 18 | b4:left | 2 | 0 | 32 | 3 | True | 1 | 3.75 |
| 19 | b4:down | 2 | 0 | 32 | 3 | True | 1 | 3.75 |
| 20 | b4:down | 2 | 0 | 32 | 3 | True | 1 | 11.75 |
| 21 | b4:up | 4 | 0 | 32 | 3 | True | 3 | 7.25 |
| 22 | b9:right | 2 | 0 | 32 | 3 | True | 1 | 15.75 |
| 23 | b4:up | 4 | 0 | 32 | 3 | True | 3 | 7.25 |
| 24 | b9:up | 2 | 0 | 32 | 3 | True | 1 | 7.75 |
| 25 | b3:down | 1 | 0 | 33 | 3 | False | 0 | NA |
| 26 | b9:left | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 27 | b9:up | 4 | 0 | 33 | 3 | True | 3 | 7.25 |
| 28 | b3:up | 1 | 0 | 33 | 3 | False | 0 | NA |
| 29 | b3:up | 1 | 0 | 33 | 3 | False | 0 | NA |
| 30 | b0:up | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 31 | b0:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 32 | b0:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 33 | b0:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 34 | b0:left | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 35 | b0:down | 1 | 0 | 33 | 3 | False | 0 | NA |
| 36 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 37 | b3:down | 1 | 0 | 33 | 3 | False | 0 | NA |
| 38 | b5:right | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 39 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 40 | b5:down | 3 | 0 | 33 | 3 | True | 2 | 7.5 |
| 41 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 42 | b5:up | 4 | 0 | 33 | 3 | True | 3 | 7.25 |
| 43 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 44 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 45 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 46 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 47 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 48 | b5:up | 4 | 0 | 33 | 3 | True | 3 | 7.25 |
| 49 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 50 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 51 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 52 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 53 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 54 | b5:up | 4 | 0 | 33 | 3 | True | 3 | 7.25 |
| 55 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 56 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 57 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 58 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 59 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 60 | b5:up | 4 | 0 | 33 | 3 | True | 3 | 7.25 |
| 61 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 62 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 63 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 64 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 65 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 66 | b5:up | 4 | 0 | 33 | 3 | True | 3 | 7.25 |
| 67 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 68 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 69 | b5:left | 5 | 0 | 33 | 3 | True | 4 | 3 |
| 70 | b5:left | 1 | 0 | 33 | 3 | False | 0 | NA |
| 71 | b5:down | 2 | 0 | 33 | 3 | True | 1 | 3.75 |
| 72 | b5:up | 3 | 0 | 33 | 3 | True | 3 | 7.25 |

## 方法含义

A14 检验的是：从 A13 的低合法余量诊断进入可执行短宏动作后，是否能恢复站位、移开 target blocker 或打开 release action。
