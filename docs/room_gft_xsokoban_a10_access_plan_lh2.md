# XSokoban first 第 1 关 Room-GFT A10 access-plan 诊断

## 核心结论

A10 在 A9 release-plan SD 基础上加入 lookahead access-plan 状态。该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定；它只消费当前状态可计算的后续 deep slot 站位格保护信号。
A9 smoke 为 `no_action`，执行 `44` 个 push 后 `boxes_on_goals=2`；A10 smoke 为 `max_steps`，执行 `120` 个 push、`83` 个闭环决策后 `boxes_on_goals=5`。
A9/A10 首个决策差异为 `7`；A10 最终 slot cost 为 `1`，access barrier 为 `8`，future access barrier 为 `0`。

## A9/A10 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | release barrier | access barrier | first future blocker step | max future blockers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A9 | no_action | 28 | 44 | 2 | 42 | 8 | NA | NA | NA |
| A10 | max_steps | 83 | 120 | 5 | 1 | -3 | 8 | NA | 0 |

## A10 轨迹前缀

| step | action | executed pushes | macro | goals | slot cost | access barrier | future access barrier | future blockers | access score |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | b3:left | 1 | False | 0 | 44 | 6 | 2 | 0 | 0 |
| 1 | b1:up | 1 | False | 0 | 44 | 2.5 | -1.5 | 0 | 0 |
| 2 | b2:down | 1 | False | 0 | 43 | -4.5 | -1.5 | 0 | 0 |
| 3 | b5:right | 1 | False | 0 | 42 | -4.5 | -1.5 | 0 | 0 |
| 4 | b5:right | 1 | False | 0 | 41 | -4.5 | -1.5 | 0 | 0 |
| 5 | b3:down | 1 | False | 0 | 40 | -4.5 | -1.5 | 0 | 0 |
| 6 | b2:left | 1 | False | 0 | 40 | -1 | 2 | 0 | -8 |
| 7 | b5:right | 1 | False | 0 | 39 | -1 | 2 | 0 | 0 |
| 8 | b5:right | 7 | True | 0 | 34 | -4 | 2 | 0 | 0 |
| 9 | b2:right | 1 | False | 0 | 34 | -7.5 | -1.5 | 0 | 0 |
| 10 | b3:down | 1 | False | 0 | 33 | -7.5 | -1.5 | 0 | 0 |
| 11 | b4:right | 1 | False | 0 | 32 | -7.5 | -1.5 | 0 | 0 |
| 12 | b4:right | 1 | False | 0 | 31 | -7.5 | -1.5 | 0 | 0 |
| 13 | b3:right | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 14 | b2:left | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 15 | b2:down | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 16 | b0:down | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 17 | b2:down | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 18 | b3:right | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 19 | b1:up | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 20 | b2:up | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 21 | b1:down | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 22 | b0:down | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 23 | b2:down | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 24 | b3:left | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 25 | b3:left | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 26 | b2:up | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 27 | b4:left | 1 | False | 0 | 31 | -7.5 | -1.5 | 0 | 0 |
| 28 | b2:down | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 29 | b0:up | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 30 | b1:up | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 31 | b4:right | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 32 | b0:down | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 33 | b1:down | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 34 | b1:right | 1 | False | 0 | 29 | -7.5 | -1.5 | 0 | 0 |
| 35 | b2:up | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 36 | b4:left | 1 | False | 0 | 31 | -7.5 | -1.5 | 0 | 0 |
| 37 | b2:down | 1 | False | 0 | 30 | -7.5 | -1.5 | 0 | 0 |
| 38 | b5:up | 3 | True | 1 | 40 | -4.5 | -1.5 | 0 | 0 |
| 39 | b5:right | 1 | False | 1 | 39 | -4.5 | -1.5 | 0 | 0 |
| 40 | b2:up | 1 | False | 1 | 40 | -4.5 | -1.5 | 0 | 0 |
| 41 | b5:right | 1 | False | 1 | 39 | -4.5 | -1.5 | 0 | 0 |
| 42 | b5:right | 7 | True | 1 | 34 | -7.5 | -1.5 | 0 | 0 |
| 43 | b5:down | 3 | True | 2 | 45 | -3 | 0 | 0 | 0 |
| 44 | b2:down | 1 | False | 2 | 44 | -3 | 0 | 0 | 0 |
| 45 | b2:down | 1 | False | 2 | 43 | -3 | 0 | 0 | 0 |
| 46 | b4:right | 1 | False | 2 | 42 | -3 | 0 | 0 | 0 |
| 47 | b4:right | 1 | False | 2 | 41 | -3 | 0 | 0 | 0 |
| 48 | b0:down | 1 | False | 2 | 41 | -3 | 0 | 0 | 0 |
| 49 | b4:right | 1 | False | 2 | 40 | -3 | 0 | 0 | 0 |
| 50 | b4:right | 9 | True | 3 | 48 | -4.5 | -1.5 | 0 | 0 |
| 51 | b1:left | 1 | False | 3 | 47 | -4.5 | -1.5 | 0 | 0 |
| 52 | b1:down | 1 | False | 3 | 46 | -4.5 | -1.5 | 0 | 0 |
| 53 | b1:down | 1 | False | 3 | 45 | -4.5 | -1.5 | 0 | 0 |
| 54 | b1:down | 1 | False | 3 | 44 | -4.5 | -1.5 | 0 | 0 |
| 55 | b3:right | 1 | False | 3 | 43 | -4.5 | -1.5 | 0 | 0 |
| 56 | b3:right | 1 | False | 3 | 42 | -4.5 | -1.5 | 0 | 0 |
| 57 | b0:down | 1 | False | 3 | 41 | -4.5 | -1.5 | 0 | 0 |
| 58 | b3:right | 1 | False | 3 | 40 | -4.5 | -1.5 | 0 | 0 |
| 59 | b3:right | 7 | True | 3 | 35 | -7.5 | -1.5 | 0 | 0 |
| 60 | b0:left | 1 | False | 3 | 34 | -7.5 | -1.5 | 0 | 0 |
| 61 | b2:right | 1 | False | 3 | 33 | -7.5 | -1.5 | 0 | 0 |
| 62 | b2:right | 1 | False | 3 | 32 | -7.5 | -1.5 | 0 | 0 |
| 63 | b2:right | 1 | False | 3 | 31 | -7.5 | -1.5 | 0 | 0 |
| 64 | b2:right | 1 | False | 3 | 30 | -7.5 | -1.5 | 0 | 0 |
| 65 | b2:right | 1 | False | 3 | 29 | -7.5 | -1.5 | 0 | 0 |
| 66 | b0:left | 1 | False | 3 | 28 | -7.5 | -1.5 | 0 | 0 |
| 67 | b2:left | 1 | False | 3 | 29 | -7.5 | -1.5 | 0 | 0 |
| 68 | b0:down | 1 | False | 3 | 28 | -7.5 | -1.5 | 0 | 0 |
| 69 | b0:down | 1 | False | 3 | 27 | -7.5 | -1.5 | 0 | 0 |
| 70 | b2:right | 1 | False | 3 | 26 | -7.5 | -1.5 | 0 | 0 |
| 71 | b3:right | 2 | True | 4 | 22 | -4.5 | -1.5 | 0 | 0 |
| 72 | b0:up | 1 | False | 4 | 23 | -4.5 | -1.5 | 0 | 0 |
| 73 | b3:right | 1 | False | 4 | 22 | -4.5 | -1.5 | 0 | 0 |
| 74 | b3:right | 7 | True | 4 | 17 | -7.5 | -1.5 | 0 | 0 |
| 75 | b3:right | 1 | False | 5 | 1 | 8 | 0 | 0 | -16 |
| 76 | b0:down | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |
| 77 | b0:down | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |
| 78 | b2:right | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |
| 79 | b2:right | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |
| 80 | b2:right | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |
| 81 | b2:right | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |
| 82 | b2:right | 1 | False | 5 | 1 | 8 | 0 | 0 | 0 |

## 方法含义

A10 检验的是：把未来 deep slot 的站位格提前纳入 SD 状态后，是否足以改变 A9 在第三个 deep slot 前的失败。若仍未解出，则说明当前缺口不只是“看见 future blocker”，还包括释放动作是否仍可达、是否需要更长程宏动作或 TVA 对前缀状态进行校准。
