# XSokoban first 第 1 关 Room-GFT A6 forced-corridor macro 诊断

## 核心结论

A6 在 A5 loop guard 基础上加入 root-level forced-corridor macro。宏只在已承诺箱子存在唯一单调推进方向时触发，树内仍保持 single-push MCTS。
A5 smoke 为 `no_action`，执行 `22` 个 push 后 `boxes_on_goals=1`；A6 smoke 为 `no_action`，执行 `22` 个 push、`14` 个闭环决策后 `boxes_on_goals=1`。
A6 触发 corridor macro `1` 次，额外压缩 push 数 `8`。该结果用于判断走廊压缩是否真正改变闭环执行，而不是只改变评分。

## A5/A6 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | macro triggers | macro extra pushes |
|---|---|---:|---:|---:|---:|---:|---:|
| A5 | no_action | 22 | 22 | 1 | 46 | 0 | 0 |
| A6 | no_action | 14 | 22 | 1 | 46 | 1 | 8 |

## A6 轨迹前缀

| step | action | executed pushes | macro | goals | slot cost | reason |
|---:|---|---:|---|---:|---:|---|
| 0 | b3:left | 1 | False | 0 | 44 | selected_box_uncommitted |
| 1 | b5:up | 1 | False | 0 | 45 | selected_box_uncommitted |
| 2 | b5:right | 1 | False | 0 | 44 | selected_box_uncommitted |
| 3 | b5:right | 1 | False | 0 | 43 | selected_box_uncommitted |
| 4 | b5:right | 1 | False | 0 | 42 | selected_box_uncommitted |
| 5 | b5:right | 1 | False | 0 | 41 | selected_box_uncommitted |
| 6 | b5:right | 1 | False | 0 | 40 | selected_box_uncommitted |
| 7 | b5:right | 1 | False | 0 | 39 | selected_box_uncommitted |
| 8 | b4:up | 1 | False | 0 | 40 | selected_box_uncommitted |
| 9 | b5:right | 9 | True | 1 | 46 | committed_slot_reached |
| 10 | b1:up | 1 | False | 1 | 46 | selected_box_uncommitted |
| 11 | b0:down | 1 | False | 1 | 46 | selected_box_uncommitted |
| 12 | b0:down | 1 | False | 1 | 46 | selected_box_uncommitted |
| 13 | b1:down | 1 | False | 1 | 46 | selected_box_uncommitted |
| 14 | no_action | 0 | False | NA | NA | NA |

## 方法含义

A6 检验的是 corridor contraction 是否能减少 A5 的逐步走廊推进开销。若仍进入 no_action 或不能增加目标完成数，则说明当前主要瓶颈已转向宏后多箱 room-slot 调度，需要参考解/teacher trace 锚定或更强的 room option，而不是只压缩单箱走廊。
