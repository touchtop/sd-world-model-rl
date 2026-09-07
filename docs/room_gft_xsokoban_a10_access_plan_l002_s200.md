# XSokoban first 第 2 关 Room-GFT A10 access-plan 诊断

## 核心结论

A10 在 A9 release-plan SD 基础上加入 lookahead access-plan 状态。该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定；它只消费当前状态可计算的后续 deep slot 站位格保护信号。
A9 smoke 为 `no_action`，执行 `34` 个 push 后 `boxes_on_goals=1`；A10 smoke 为 `no_action`，执行 `57` 个 push、`53` 个闭环决策后 `boxes_on_goals=1`。
A9/A10 首个决策差异为 `13`；A10 最终 slot cost 为 `52`，access barrier 为 `8`，future access barrier 为 `4`。

## A9/A10 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | release barrier | access barrier | first future blocker step | max future blockers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A9 | no_action | 29 | 34 | 1 | 52 | 4 | NA | NA | NA |
| A10 | no_action | 53 | 57 | 1 | 52 | 4 | 8 | 52 | 1 |

## A10 轨迹前缀

| step | action | executed pushes | macro | goals | slot cost | access barrier | future access barrier | future blockers | access score |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | b3:right | 1 | False | 0 | 51 | 6 | 2 | 0 | 0 |
| 1 | b7:left | 1 | False | 0 | 50 | 6 | 2 | 0 | 0 |
| 2 | b8:left | 1 | False | 0 | 49 | 6 | 2 | 0 | 0 |
| 3 | b8:left | 1 | False | 0 | 48 | 6 | 2 | 0 | 0 |
| 4 | b4:down | 1 | False | 0 | 48 | 6 | 2 | 0 | 0 |
| 5 | b8:left | 1 | False | 0 | 48 | 6 | 2 | 0 | 0 |
| 6 | b4:left | 1 | False | 0 | 48 | 6 | 2 | 0 | 0 |
| 7 | b3:up | 1 | False | 0 | 48 | 6 | 2 | 0 | 0 |
| 8 | b3:up | 1 | False | 0 | 48 | 6 | 2 | 0 | 0 |
| 9 | b1:left | 1 | False | 0 | 48 | 2.5 | -1.5 | 0 | 0 |
| 10 | b0:right | 1 | False | 0 | 49 | 2.5 | -1.5 | 0 | 0 |
| 11 | b2:down | 4 | True | 0 | 47 | 1 | -3 | 0 | 0 |
| 12 | b2:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 13 | b9:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 14 | b3:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 15 | b4:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 16 | b4:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 17 | b9:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 18 | b2:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 19 | b4:right | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 20 | b3:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 21 | b4:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 22 | b2:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 23 | b3:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 24 | b3:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 25 | b3:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 26 | b8:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 27 | b3:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 28 | b3:right | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 29 | b9:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 30 | b9:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 31 | b4:right | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 32 | b4:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 33 | b9:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 34 | b8:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 35 | b3:right | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 36 | b9:right | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 37 | b3:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 38 | b3:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 39 | b9:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 40 | b2:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 41 | b4:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 42 | b2:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 43 | b4:left | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 44 | b4:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 45 | b8:up | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 46 | b9:right | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 47 | b4:down | 1 | False | 0 | 47 | 1 | -3 | 0 | 0 |
| 48 | b3:left | 1 | False | 1 | 56 | 1 | -3 | 0 | 0 |
| 49 | b9:left | 1 | False | 1 | 56 | 1 | -3 | 0 | 0 |
| 50 | b8:up | 1 | False | 1 | 56 | 1 | -3 | 0 | 0 |
| 51 | b9:left | 1 | False | 1 | 56 | 1 | -3 | 0 | 0 |
| 52 | b3:left | 2 | True | 1 | 52 | 8 | 4 | 1 | 0 |
| 53 | no_action | 0 | False | NA | NA | 8 | 4 | 1 | NA |

## 方法含义

A10 检验的是：把未来 deep slot 的站位格提前纳入 SD 状态后，是否足以改变 A9 在第三个 deep slot 前的失败。若仍未解出，则说明当前缺口不只是“看见 future blocker”，还包括释放动作是否仍可达、是否需要更长程宏动作或 TVA 对前缀状态进行校准。
