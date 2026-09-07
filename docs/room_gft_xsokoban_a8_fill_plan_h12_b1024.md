# XSokoban first 第 1 关 Room-GFT A8 fill-plan 诊断

## 核心结论

A8 在 A6 forced-corridor macro 基础上加入 teacher-trace-distilled fill-plan SD 状态。该版本不读取 teacher 动作表，也不使用 A7 的逐状态参考锚定；它只消费当前状态可计算的 active slot、channel/staging 与 fill-plan barrier。
A6 smoke 为 `no_action`，执行 `27` 个 push 后 `boxes_on_goals=1`；A8 smoke 为 `no_action`，执行 `48` 个 push、`32` 个闭环决策后 `boxes_on_goals=2`。
A6/A8 首个决策差异为 `1`；A8 最终 slot cost 为 `43`，fill-plan barrier 为 `4.65`。

## A6/A8 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | final fill barrier | macro triggers |
|---|---|---:|---:|---:|---:|---:|---:|
| A6 | no_action | 19 | 27 | 1 | 43 | NA | 1 |
| A8 | no_action | 32 | 48 | 2 | 43 | 4.65 | 4 |

## A8 轨迹前缀

| step | action | executed pushes | macro | goals | slot cost | fill barrier | fill score |
|---:|---|---:|---|---:|---:|---:|---:|
| 0 | b3:left | 1 | False | 0 | 44 | -0.35 | 0 |
| 1 | b1:up | 1 | False | 0 | 44 | -0.35 | 0 |
| 2 | b2:down | 1 | False | 0 | 43 | -2.35 | 8.4 |
| 3 | b3:down | 1 | False | 0 | 42 | -1.85 | -2.6 |
| 4 | b5:right | 1 | False | 0 | 41 | -1.9 | 0.26 |
| 5 | b5:right | 1 | False | 0 | 40 | -1.95 | 0.26 |
| 6 | b2:left | 1 | False | 0 | 40 | -1.95 | 0 |
| 7 | b2:down | 1 | False | 0 | 39 | -1.95 | 0 |
| 8 | b4:right | 1 | False | 0 | 39 | -1.95 | 0 |
| 9 | b5:right | 1 | False | 0 | 38 | -2 | 0.26 |
| 10 | b5:right | 7 | True | 0 | 33 | -9.85 | 5.06 |
| 11 | b5:up | 3 | True | 1 | 42 | 4.7 | -16.94 |
| 12 | b5:right | 1 | False | 1 | 41 | 4.7 | 0 |
| 13 | b5:right | 1 | False | 1 | 40 | -1.85 | 16.06 |
| 14 | b5:right | 1 | False | 1 | 39 | -1.9 | 0.26 |
| 15 | b5:right | 1 | False | 1 | 38 | -1.95 | 0.26 |
| 16 | b5:right | 1 | False | 1 | 37 | -2 | 0.26 |
| 17 | b1:down | 1 | False | 1 | 37 | -2 | 0 |
| 18 | b5:right | 7 | True | 1 | 32 | -9.85 | 5.06 |
| 19 | b5:down | 3 | True | 2 | 44 | 4.65 | -16.94 |
| 20 | b1:down | 1 | False | 2 | 44 | 4.65 | 0 |
| 21 | b1:left | 1 | False | 2 | 43 | 4.65 | 0 |
| 22 | b1:left | 1 | False | 2 | 42 | 4.65 | 0 |
| 23 | b0:down | 1 | False | 2 | 42 | 4.65 | 0 |
| 24 | b1:left | 1 | False | 2 | 43 | 4.65 | 0 |
| 25 | b0:up | 1 | False | 2 | 43 | 4.65 | 0 |
| 26 | b1:right | 1 | False | 2 | 42 | 4.65 | 0 |
| 27 | b0:down | 1 | False | 2 | 42 | 4.65 | 0 |
| 28 | b1:right | 1 | False | 2 | 43 | 4.65 | 0 |
| 29 | b1:right | 1 | False | 2 | 43 | 4.65 | 0 |
| 30 | b0:down | 1 | False | 2 | 42 | 4.65 | 0 |
| 31 | b0:left | 1 | False | 2 | 43 | 4.65 | 0 |
| 32 | no_action | 0 | False | NA | NA | 4.65 | NA |

## 方法含义

A8 检验的是 teacher 第 1 关轨迹揭示的结构缺口能否被补成无参考动作的 SD 状态。若 A8 仍未解出，则应从轨迹中继续抽取更强的释放队列、通道保持或局部可达性状态；若 A8 改善，则后续再考虑把同类状态接入 TVA 审计样本，而不是把 teacher 动作在线注入。
