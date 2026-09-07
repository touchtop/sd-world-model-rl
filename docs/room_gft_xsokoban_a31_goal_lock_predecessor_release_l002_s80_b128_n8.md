# XSokoban first 第 2 关 Room-GFT A31 goal-lock predecessor 诊断

## 核心结论

A31 继承 A30，并将 A30 的 goal-lock 宏截断信号前移到 release 阶段 root 候选替换。该版本仍不接入 teacher/reference 在线动作。
A28 为 `max_steps`、pushes=80、decisions=45、goals=3；A29 为 `max_steps`、pushes=80、decisions=45、goals=3；A30 为 `max_steps`、pushes=80、decisions=52、goals=2；A31 为 `max_steps`、pushes=80、decisions=42、goals=2。
A30/A31 首个决策差异为 `25`；A31 root 替代次数为 `2`，替代后打开 release action 次数为 `2`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | truncations | root replacements | replacement progress | opened release |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A28 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | NA | NA | NA | NA |
| A29 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | NA | NA | NA | 0 |
| A30 | max_steps | 52 | 80 | 2 | 35 | 2 | 0 | 0 | 1 | NA | NA | 0 |
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

## A31 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | macro reason | root replacement | candidates | opened release |
|---:|---|---:|---:|---|---:|---:|---|---|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | selected_box_uncommitted | False | None | False |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | below_min_total_pushes | False | None | False |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | pre_release_bottleneck_truncated_corridor | False | None | False |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | no_monotone_committed_push | False | None | False |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | below_min_total_pushes | False | None | False |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | selected_box_uncommitted | False | None | False |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | selected_box_uncommitted | False | None | False |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | pre_release_compatible_dependency_macro | False | None | False |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | pre_release_compatible_dependency_macro | False | None | False |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | pre_release_compatible_dependency_macro | False | None | False |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | committed_slot_reached | False | None | False |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | selected_box_uncommitted | False | None | False |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | committed_slot_reached | False | None | False |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | committed_slot_reached | False | None | False |
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | selected_box_uncommitted | False | None | False |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | selected_box_uncommitted | False | None | False |
| 16 | b8:left | 1 | 1 | fill | 47 | 2 | selected_box_uncommitted | False | None | False |
| 17 | b8:left | 1 | 1 | fill | 46 | 2 | selected_box_uncommitted | False | None | False |
| 18 | b8:left | 1 | 1 | fill | 45 | 2 | below_min_total_pushes | False | None | False |
| 19 | b2:left | 1 | 1 | fill | 44 | 2 | selected_box_uncommitted | False | None | False |
| 20 | b8:left | 2 | 1 | fill | 44 | 1 | pre_release_bottleneck_truncated_corridor | False | None | False |
| 21 | b8:up | 6 | 1 | fill | 40 | 1 | ambiguous_monotone_committed_push | False | None | False |
| 22 | b3:down | 2 | 1 | fill | 40 | 1 | committed_slot_reached | False | None | False |
| 23 | b3:up | 2 | 1 | fill | 40 | 1 | committed_slot_reached | False | None | False |
| 24 | b6:left | 1 | 2 | fill | 49 | 1 | below_min_total_pushes | False | None | False |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | goal_lock_predecessor_root_replacement | True | 8 | True |
| 26 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 27 | b3:up | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 28 | b4:up | 1 | 2 | release | 51 | 1 | below_min_total_pushes | False | None | False |
| 29 | b0:down | 2 | 2 | release | 49 | 1 | ambiguous_monotone_committed_push | False | None | False |
| 30 | b6:up | 1 | 2 | release | 29 | 2 | goal_lock_predecessor_root_replacement | True | 8 | True |
| 31 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 32 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 33 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 34 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 35 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 36 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 37 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 38 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 39 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 40 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |
| 41 | b3:down | 2 | 2 | release | 29 | 2 | committed_slot_reached | False | None | False |

## 方法含义

A31 已把 A30 的 goal-lock 截断信号前移到 root 替代动作选择。本轮触发替代 2 次；若目标数仍未恢复，说明当前同根候选只能避开坏宏，尚不能生成足够的 release 调度进展。
