# XSokoban first 第 2 关 Room-GFT A34 milestone SD-consumer 诊断

## 核心结论

A34 继承 A33/A32/A31，并加载 future-window milestone target 训练得到的 MLP checkpoint。默认配置只把 milestone score 作为 root prior/value 的软信号消费，不启用 root hard override；该版本不接入 teacher/reference 在线动作，也不替代 exact Sokoban dynamics。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2、release actions=2；A34 为 `max_steps`、pushes=30、decisions=14、goals=1、release actions=0。
A31/A34 首个决策差异为 `14`；A34 milestone scorer 加载步数为 `14`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | milestone loaded | positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | NA | NA |
| A34 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 14 | 14 |

## A34 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | release actions | milestone score | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 1.477 | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 0 | 0.696 | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 0 | 1.894 | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 0 | 2.683 | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | 0 | 2.58 | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 0 | 2.484 | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 0 | 1.759 | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | 0 | 0.163 | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 0 | 1.819 | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 0 | 1.561 | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 0 | 1.61 | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 0 | 0.362 | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 0 | 1.07 | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 0 | 0.597 | committed_slot_reached |

## 方法含义

A34 的主要作用是检验 future-window milestone 目标是否优于 A33 的即时 SD progress 目标。该 checkpoint 仍是离线 trace 蒸馏结果，不能视为充分训练的通用求解器。
