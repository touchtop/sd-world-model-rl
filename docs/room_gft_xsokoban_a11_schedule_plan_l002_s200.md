# XSokoban first 第 2 关 Room-GFT A11 schedule-plan 诊断

## 核心结论

A11 在 A10 access-plan 基础上加入 state-derived multi-box release dependency graph。该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定。
A10 smoke 为 `no_action`，执行 `57` 个 push 后 `boxes_on_goals=1`；A11 smoke 为 `no_action`，执行 `14` 个 push、`11` 个闭环决策后 `boxes_on_goals=0`。
A10/A11 首个决策差异为 `0`；A11 最终 slot cost 为 `49`，schedule barrier 为 `79.667`，dependency count 为 `3`。

## A10/A11 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | access barrier | schedule barrier | dependencies | release actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A10 | no_action | 53 | 57 | 1 | 52 | 8 | NA | NA | NA |
| A11 | no_action | 11 | 14 | 0 | 49 | 18.667 | 79.667 | 3 | 0 |

## A11 轨迹前缀

| step | action | pushes | macro | goals | slot cost | schedule barrier | dependencies | release actions | schedule score |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | b7:right | 1 | False | 0 | 52 | 29 | 2 | 0 | 28.067 |
| 1 | b3:right | 1 | False | 0 | 52 | 29 | 2 | 0 | 0 |
| 2 | b2:up | 1 | False | 0 | 53 | 39.5 | 2 | 0 | -13.4 |
| 3 | b6:right | 1 | False | 0 | 52 | 43.5 | 2 | 0 | -14.2 |
| 4 | b6:right | 4 | True | 0 | 48 | 50.667 | 3 | 0 | 11.067 |
| 5 | b3:up | 1 | False | 0 | 49 | 79.667 | 3 | 0 | -62.2 |
| 6 | b9:left | 1 | False | 0 | 49 | 79.667 | 3 | 0 | 0 |
| 7 | b6:left | 1 | False | 0 | 49 | 79.667 | 3 | 0 | 0 |
| 8 | b4:up | 1 | False | 0 | 49 | 79.667 | 3 | 0 | 0 |
| 9 | b4:up | 1 | False | 0 | 49 | 79.667 | 3 | 0 | 0 |
| 10 | b4:up | 1 | False | 0 | 49 | 79.667 | 3 | 0 | 0 |
| 11 | no_action | 0 | False | NA | NA | 79.667 | 3 | 0 | NA |

## 方法含义

A11 检验的是：将多箱释放依赖显式放入 SD 状态后，MCTS 是否能够消费这些调度信号。若仍未解出，则剩余缺口应继续定位在 release macro、跨房间调度或 TVA 前缀校准，而不是 teacher 在线动作锚定。
