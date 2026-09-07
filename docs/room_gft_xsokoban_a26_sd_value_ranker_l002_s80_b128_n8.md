# XSokoban first 第 2 关 Room-GFT A26 SD value ranker 诊断

## 核心结论

A26 继承 A25，并新增 root-only 的线性 SD value/ranking consumer。该版本固定 learned-consumer 接口，但仍不接入 teacher/reference 在线动作。
A24 为 `max_steps`、pushes=80、decisions=78、goals=0；A25 为 `max_steps`、pushes=80、decisions=78、goals=0；A26 为 `max_steps`、pushes=80、decisions=39、goals=1。
A25/A26 首个决策差异为 `2`；A26 milestone 选择步数为 `31`，no-progress loop 选择步数为 `1`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | severe trap | generalized failure | value milestone | value loop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A24 | max_steps | 78 | 80 | 0 | 48 | 2 | 1 | 0 | 73 | NA | NA | NA |
| A25 | max_steps | 78 | 80 | 0 | 49 | 1 | 0 | 0 | 0 | 0 | NA | NA |
| A26 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | 0 | 1 | 31 | 1 |

## A26 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | sd value | milestone | loop risk | hard neg | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 18 | False | False | False | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 54 | True | False | False | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 34 | True | False | False | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 10 | False | False | False | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | -144 | False | True | False | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 162 | True | False | False | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 32 | False | False | False | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | -58 | False | False | False | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 26 | False | False | False | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 10 | False | False | False | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 210 | True | False | False | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 18 | False | False | False | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 106 | True | False | False | committed_slot_reached |
| 14 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 15 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 16 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 17 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 18 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 19 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 20 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 21 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 22 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 23 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 24 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 25 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 26 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 27 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 28 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 29 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 30 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 31 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 32 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 33 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 34 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 35 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 36 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 37 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |
| 38 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | committed_slot_reached |

## 方法含义

A26 检验的是：在 A25 已能避开已知 trap 之后，能否把 phase progress、release/dependency 进展、重复无进展循环和 trap 风险放入同一个可替换的 SD value/ranking consumer。当前线性权重是 learned consumer 的接口占位，不是最终训练模型。
