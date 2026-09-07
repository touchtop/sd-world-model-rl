# XSokoban first 第 1 关 Room-GFT A9 release-plan 诊断

## 核心结论

A9 在 A8 fill-plan SD 基础上加入 active-box release-plan 状态。该版本不读取 teacher 动作表，也不继承 A7 的逐状态参考锚定；它只消费当前状态可计算的候选单调推进、所需站位格、站位格阻塞和玩家可达性信号。
A8 smoke 为 `no_action`，执行 `48` 个 push 后 `boxes_on_goals=2`；A9 smoke 为 `no_action`，执行 `44` 个 push、`28` 个闭环决策后 `boxes_on_goals=2`。
A8/A9 首个决策差异为 `3`；A9 最终 slot cost 为 `42`，fill barrier 为 `4.65`，release barrier 为 `8`。

## A8/A9 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | fill barrier | release barrier | release need | stand blockers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A8 | no_action | 32 | 48 | 2 | 43 | 4.65 | NA | NA | NA |
| A9 | no_action | 28 | 44 | 2 | 42 | 4.65 | 8 | 1 | 1 |

## A9 轨迹前缀

| step | action | executed pushes | macro | goals | slot cost | fill barrier | release barrier | release need | stand blockers | release score |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | b3:left | 1 | False | 0 | 44 | -0.35 | 4 | 1 | 0 | 0 |
| 1 | b1:up | 1 | False | 0 | 44 | -0.35 | 4 | 1 | 0 | 0 |
| 2 | b2:down | 1 | False | 0 | 43 | -2.35 | -3 | 0 | 0 | 26.6 |
| 3 | b5:right | 1 | False | 0 | 42 | -1.9 | -3 | 0 | 0 | 0 |
| 4 | b5:right | 1 | False | 0 | 41 | -1.95 | -3 | 0 | 0 | 0 |
| 5 | b3:down | 1 | False | 0 | 40 | -1.95 | -3 | 0 | 0 | 0 |
| 6 | b2:left | 1 | False | 0 | 40 | -1.95 | -3 | 0 | 0 | 0 |
| 7 | b2:down | 1 | False | 0 | 39 | -1.95 | -3 | 0 | 0 | 0 |
| 8 | b4:right | 1 | False | 0 | 39 | -1.95 | -3 | 0 | 0 | 0 |
| 9 | b5:right | 1 | False | 0 | 38 | -2 | -3 | 0 | 0 | 0 |
| 10 | b5:right | 7 | True | 0 | 33 | -9.85 | -6 | 0 | 0 | 0 |
| 11 | b4:right | 1 | False | 0 | 32 | -9.85 | -6 | 0 | 0 | 0 |
| 12 | b0:down | 1 | False | 0 | 32 | -9.85 | -6 | 0 | 0 | 0 |
| 13 | b4:right | 1 | False | 0 | 31 | -9.85 | -6 | 0 | 0 | 0 |
| 14 | b4:right | 1 | False | 0 | 30 | -9.85 | -6 | 0 | 0 | 0 |
| 15 | b4:right | 1 | False | 0 | 29 | -9.85 | -6 | 0 | 0 | 0 |
| 16 | b1:up | 1 | False | 0 | 29 | -9.85 | -6 | 0 | 0 | 0 |
| 17 | b1:down | 1 | False | 0 | 29 | -9.85 | -6 | 0 | 0 | 0 |
| 18 | b1:down | 1 | False | 0 | 29 | -9.85 | -6 | 0 | 0 | 0 |
| 19 | b1:left | 1 | False | 0 | 29 | -9.85 | -6 | 0 | 0 | 0 |
| 20 | b4:left | 1 | False | 0 | 30 | -9.85 | -6 | 0 | 0 | 0 |
| 21 | b0:down | 1 | False | 0 | 30 | -9.85 | -6 | 0 | 0 | 0 |
| 22 | b5:up | 3 | True | 1 | 39 | -1.9 | -3 | 0 | 0 | -12.4 |
| 23 | b5:right | 1 | False | 1 | 38 | -1.95 | -3 | 0 | 0 | 0 |
| 24 | b0:down | 1 | False | 1 | 38 | -1.95 | -3 | 0 | 0 | 0 |
| 25 | b5:right | 1 | False | 1 | 37 | -2 | -3 | 0 | 0 | 0 |
| 26 | b5:right | 7 | True | 1 | 32 | -9.85 | -6 | 0 | 0 | 0 |
| 27 | b5:down | 3 | True | 2 | 42 | 4.65 | 8 | 1 | 1 | -12.4 |
| 28 | no_action | 0 | False | NA | NA | 4.65 | 8 | 1 | 1 | NA |

## 方法含义

A9 检验的是：A8 末端暴露的 active box blocked 状态，是否可以通过显式建模站位格释放来转化为无 teacher 在线动作锚定的改进。若 A9 仍未解出，则应记录其对 release barrier 和 stand blocker 的实际影响，并据此判断后续缺口是否已经转向更长程的多箱调度或 TVA 价值校准。
