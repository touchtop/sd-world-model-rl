# XSokoban first 第 2 关 Room-GFT A27 SD value loop-calibrated 诊断

## 核心结论

A27 继承 A26，并校准重复伪里程碑。该版本保留 root-only SD value/ranking consumer 接口，但仍不接入 teacher/reference 在线动作。
A24 为 `max_steps`、pushes=80、decisions=78、goals=0；A25 为 `max_steps`、pushes=80、decisions=78、goals=0；A26 为 `max_steps`、pushes=80、decisions=39、goals=1；A27 为 `max_steps`、pushes=80、decisions=39、goals=1。
A26/A27 首个决策差异为 `None`；A27 milestone 选择步数为 `6`，pseudo milestone loop 选择步数为 `25`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | severe trap | generalized failure | value milestone | value loop | pseudo milestone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A24 | max_steps | 78 | 80 | 0 | 48 | 2 | 1 | 0 | 73 | NA | NA | NA | NA |
| A25 | max_steps | 78 | 80 | 0 | 49 | 1 | 0 | 0 | 0 | 0 | NA | NA | NA |
| A26 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | 0 | 1 | 31 | 1 | NA |
| A27 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | 0 | 1 | 6 | 26 | 25 |

## A27 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | sd value | milestone | loop risk | pseudo | hard neg | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 18 | False | False | False | False | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 54 | True | False | False | False | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 34 | True | False | False | False | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 10 | False | False | False | False | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | -144 | False | True | False | False | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 162 | True | False | False | False | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 32 | False | False | False | False | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | -58 | False | False | False | False | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 26 | False | False | False | False | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 10 | False | False | False | False | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 210 | True | False | False | False | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 18 | False | False | False | False | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 166 | True | False | False | False | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 106 | True | False | False | False | committed_slot_reached |
| 14 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 15 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 16 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 17 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 18 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 19 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 20 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 21 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 22 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 23 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 24 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 25 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 26 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 27 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 28 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 29 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 30 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 31 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 32 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 33 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 34 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 35 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 36 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 37 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |
| 38 | b3:down | 2 | 1 | fill | 48 | 2 | -24 | False | True | True | True | committed_slot_reached |

## 方法含义

A27 检验的是：A26 的 value/ranking consumer 是否需要显式区分真实里程碑与重复伪里程碑。当前校准仍是线性规则，但其接口对应后续 learned consumer 的负例标签。
