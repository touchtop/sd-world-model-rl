# XSokoban first 第 2 关 Room-GFT A25 SD generalization guard 诊断

## 核心结论

A25 继承 A24，保留 pre-release trap 风险评分，并新增原型式 SD-consumer 泛化评分；该版本不接入 teacher/reference 在线动作。
A22 为 `max_steps`、pushes=80、decisions=51、goals=2；A23 为 `max_steps`、pushes=80、decisions=53、goals=2；A24 为 `max_steps`、pushes=80、decisions=78、goals=0；A25 为 `max_steps`、pushes=30、decisions=28、goals=0。
A24/A25 首个决策差异为 `2`；A25 generalized failure 选择步数为 `0`，trap-predecessor 选择步数为 `0`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | bottleneck truncations | severe trap | generalized failure | trap predecessor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A22 | max_steps | 51 | 80 | 2 | 60 | 2 | 0 | 0 | NA | NA | NA | NA |
| A23 | max_steps | 53 | 80 | 2 | 60 | 2 | 0 | 0 | 2 | NA | NA | NA |
| A24 | max_steps | 78 | 80 | 0 | 48 | 2 | 1 | 0 | 1 | 73 | NA | NA |
| A25 | max_steps | 28 | 30 | 0 | 49 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## A25 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | trap score | gen score | gen fail | predecessor | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 26 | 41.714 | False | False | below_min_total_pushes |
| 2 | b6:left | 1 | 0 | fill | 52 | 2 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 3 | b6:right | 1 | 0 | fill | 51 | 1 | 30 | 45.714 | False | False | selected_box_uncommitted |
| 4 | b3:right | 1 | 0 | fill | 51 | 1 | 6 | 21.714 | False | False | selected_box_uncommitted |
| 5 | b8:left | 1 | 0 | fill | 50 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 6 | b5:left | 1 | 0 | fill | 50 | 2 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 7 | b3:up | 1 | 0 | fill | 50 | 1 | 42 | 57.714 | False | False | selected_box_uncommitted |
| 8 | b3:up | 1 | 0 | fill | 50 | 1 | -2 | 13.714 | False | False | selected_box_uncommitted |
| 9 | b2:left | 3 | 0 | fill | 49 | 1 | 32 | 39.429 | False | False | pre_release_compatible_dependency_macro |
| 10 | b4:down | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 11 | b9:up | 1 | 0 | fill | 49 | 2 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 12 | b4:up | 1 | 0 | fill | 49 | 1 | 42 | 57.714 | False | False | selected_box_uncommitted |
| 13 | b5:up | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 14 | b5:up | 1 | 0 | fill | 49 | 1 | -2 | 13.714 | False | False | selected_box_uncommitted |
| 15 | b5:right | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 16 | b9:up | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 17 | b6:down | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 18 | b9:up | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 19 | b6:down | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 20 | b9:up | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 21 | b6:down | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 22 | b9:up | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 23 | b6:down | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 24 | b9:up | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 25 | b6:down | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |
| 26 | b9:up | 1 | 0 | fill | 49 | 1 | -6 | 9.714 | False | False | selected_box_uncommitted |
| 27 | b6:down | 1 | 0 | fill | 49 | 1 | -10 | 5.714 | False | False | selected_box_uncommitted |

## 方法含义

A25 检验的是：能否把 A24 已知失败形态抽象为 SD 结构原型，并将其泛化到未见根候选上。该版本是后续可学习 TVA/value consumer 的接口雏形，而不是最终神经网络实现。
