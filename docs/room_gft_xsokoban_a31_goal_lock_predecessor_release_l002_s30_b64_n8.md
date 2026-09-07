# XSokoban first 第 2 关 Room-GFT A31 goal-lock predecessor 诊断

## 核心结论

A31 继承 A30，并将 A30 的 goal-lock 宏截断信号前移到 release 阶段 root 候选替换。该版本仍不接入 teacher/reference 在线动作。
A28 为 `max_steps`、pushes=80、decisions=45、goals=3；A29 为 `max_steps`、pushes=80、decisions=45、goals=3；A30 为 `max_steps`、pushes=80、decisions=52、goals=2；A31 为 `max_steps`、pushes=30、decisions=14、goals=1。
A30/A31 首个决策差异为 `14`；A31 root 替代次数为 `0`，替代后打开 release action 次数为 `0`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | truncations | root replacements | replacement progress | opened release |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A28 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | NA | NA | NA | NA |
| A29 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | NA | NA | NA | 0 |
| A30 | max_steps | 52 | 80 | 2 | 35 | 2 | 0 | 0 | 1 | NA | NA | 0 |
| A31 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |

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

## 方法含义

A31 未触发 root 替代动作。若 A30 仍存在 goal-lock 截断，则说明当前根候选集合中没有满足非目标回退与非 goal-lock 条件的替代动作；若 A30 截断也未再出现，则说明前缀轨迹已经偏离原 failure 入口。
