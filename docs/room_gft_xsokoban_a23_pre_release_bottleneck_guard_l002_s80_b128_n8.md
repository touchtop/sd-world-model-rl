# XSokoban first 第 2 关 Room-GFT A23 pre-release bottleneck guard 诊断

## 核心结论

A23 继承 A22，仅在预释放阶段对 corridor macro 加入 bottleneck 截断；该版本不接入 teacher/reference 在线动作。
A22 为 `max_steps`、pushes=80、decisions=51、goals=2；A23 为 `max_steps`、pushes=80、decisions=53、goals=2。
A22/A23 首个决策差异为 `3`；A23 corridor 截断步骤数为 `2`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | release macro steps | bottleneck truncations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A22 | max_steps | 51 | 80 | 2 | 60 | 2 | 0 | 0 | 0 | NA |
| A23 | max_steps | 53 | 80 | 2 | 60 | 2 | 0 | 0 | 0 | 2 |

## A23 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | macro reason |
|---:|---|---:|---:|---|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 1 | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 2 | selected_box_uncommitted |
| 6 | b7:left | 3 | 0 | fill | 47 | 1 | pre_release_compatible_dependency_macro |
| 7 | b7:up | 6 | 0 | fill | 43 | 1 | ambiguous_monotone_committed_push |
| 8 | b8:left | 1 | 0 | fill | 42 | 1 | selected_box_uncommitted |
| 9 | b8:left | 1 | 0 | fill | 41 | 1 | selected_box_uncommitted |
| 10 | b8:left | 1 | 0 | fill | 40 | 1 | below_min_total_pushes |
| 11 | b4:up | 3 | 1 | fill | 46 | 2 | committed_slot_reached |
| 12 | b3:down | 2 | 1 | fill | 46 | 2 | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 46 | 2 | committed_slot_reached |
| 14 | b8:left | 2 | 1 | fill | 46 | 2 | pre_release_bottleneck_truncated_corridor |
| 15 | b8:up | 6 | 1 | fill | 42 | 2 | ambiguous_monotone_committed_push |
| 16 | b3:down | 2 | 1 | fill | 42 | 2 | committed_slot_reached |
| 17 | b3:up | 2 | 1 | fill | 42 | 2 | committed_slot_reached |
| 18 | b5:up | 1 | 1 | fill | 41 | 2 | below_min_total_pushes |
| 19 | b4:up | 1 | 1 | fill | 40 | 2 | below_min_total_pushes |
| 20 | b0:left | 1 | 2 | fill | 49 | 2 | below_min_total_pushes |
| 21 | b4:down | 2 | 2 | release | 49 | 2 | committed_slot_reached |
| 22 | b4:up | 2 | 2 | release | 49 | 2 | committed_slot_reached |
| 23 | b0:down | 2 | 2 | release | 49 | 2 | ambiguous_monotone_committed_push |
| 24 | b0:up | 2 | 2 | release | 49 | 2 | committed_slot_reached |
| 25 | b7:down | 1 | 2 | release | 48 | 2 | selected_box_uncommitted |
| 26 | b8:left | 1 | 2 | release | 47 | 2 | selected_box_uncommitted |
| 27 | b8:left | 1 | 2 | release | 46 | 2 | selected_box_uncommitted |
| 28 | b8:left | 1 | 2 | release | 45 | 2 | below_min_total_pushes |
| 29 | b4:down | 2 | 2 | release | 45 | 2 | committed_slot_reached |
| 30 | b7:left | 1 | 2 | release | 45 | 2 | selected_box_uncommitted |
| 31 | b4:down | 2 | 2 | release | 45 | 2 | committed_slot_reached |
| 32 | b4:up | 1 | 2 | release | 66 | 2 | below_min_total_pushes |
| 33 | b7:left | 1 | 2 | release | 65 | 2 | selected_box_uncommitted |
| 34 | b7:down | 1 | 2 | release | 64 | 2 | selected_box_uncommitted |
| 35 | b8:left | 1 | 2 | release | 63 | 2 | selected_box_uncommitted |
| 36 | b8:left | 1 | 2 | release | 62 | 2 | selected_box_uncommitted |
| 37 | b9:left | 1 | 2 | release | 62 | 2 | selected_box_uncommitted |
| 38 | b6:up | 1 | 2 | release | 62 | 2 | selected_box_uncommitted |
| 39 | b6:up | 1 | 2 | release | 62 | 2 | selected_box_uncommitted |
| 40 | b4:left | 1 | 2 | release | 61 | 2 | selected_box_uncommitted |
| 41 | b4:left | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 42 | b9:left | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 43 | b6:up | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 44 | b5:left | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 45 | b5:left | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 46 | b9:right | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 47 | b9:up | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 48 | b7:right | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 49 | b7:down | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 50 | b9:left | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 51 | b9:up | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |
| 52 | b7:up | 1 | 2 | release | 60 | 2 | selected_box_uncommitted |

## 方法含义

A23 检验的是：A22 的失败是否主要来自 corridor macro 在 fill 阶段越过了必要的 dependency/release 前置检查。若截断能改变早期互锁前缀，则说明后续应把阶段化机制进一步前移到宏执行边界；若仍无改善，则说明还需要根选择层生成替代调度，而非只截断宏。
