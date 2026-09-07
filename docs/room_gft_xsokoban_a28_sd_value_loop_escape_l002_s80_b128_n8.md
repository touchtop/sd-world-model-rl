# XSokoban first 第 2 关 Room-GFT A28 SD value loop-escape 诊断

## 核心结论

A28 继承 A27，并把伪里程碑循环标签接入 root escape 选择。该版本仍不接入 teacher/reference 在线动作。
A24 为 `max_steps`、pushes=80、decisions=78、goals=0；A25 为 `max_steps`、pushes=80、decisions=78、goals=0；A26 为 `max_steps`、pushes=80、decisions=39、goals=1；A27 为 `max_steps`、pushes=80、decisions=39、goals=1；A28 为 `max_steps`、pushes=80、decisions=45、goals=3。
A27/A28 首个决策差异为 `14`；A28 milestone 选择步数为 `12`，pseudo milestone loop 选择步数为 `0`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | severe trap | generalized failure | value milestone | value loop | pseudo milestone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A24 | max_steps | 78 | 80 | 0 | 48 | 2 | 1 | 0 | 73 | NA | NA | NA | NA |
| A25 | max_steps | 78 | 80 | 0 | 49 | 1 | 0 | 0 | 0 | 0 | NA | NA | NA |
| A26 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | 0 | 1 | 31 | 1 | NA |
| A27 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | 0 | 1 | 6 | 26 | 25 |
| A28 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | 0 | 20 | 12 | 19 | 0 |

## A28 轨迹前缀

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
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | -108 | False | True | False | False | selected_box_uncommitted |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | -84 | False | True | False | False | selected_box_uncommitted |
| 16 | b8:left | 1 | 1 | fill | 47 | 2 | 10 | False | False | False | False | selected_box_uncommitted |
| 17 | b8:left | 1 | 1 | fill | 46 | 2 | 10 | False | False | False | False | selected_box_uncommitted |
| 18 | b8:left | 1 | 1 | fill | 45 | 2 | 100 | True | False | False | False | below_min_total_pushes |
| 19 | b2:left | 1 | 1 | fill | 44 | 2 | 10 | False | False | False | False | selected_box_uncommitted |
| 20 | b8:left | 2 | 1 | fill | 44 | 1 | 92 | True | False | False | False | pre_release_bottleneck_truncated_corridor |
| 21 | b8:up | 6 | 1 | fill | 40 | 1 | 74 | False | False | False | False | ambiguous_monotone_committed_push |
| 22 | b3:down | 2 | 1 | fill | 40 | 1 | 212 | True | False | False | False | committed_slot_reached |
| 23 | b3:up | 2 | 1 | fill | 40 | 1 | 152 | True | False | False | False | committed_slot_reached |
| 24 | b6:left | 1 | 2 | fill | 49 | 1 | 150 | True | False | False | False | below_min_total_pushes |
| 25 | b6:down | 4 | 2 | release | 35 | 2 | 34 | False | False | False | False | phased_release_dependency_macro |
| 26 | b9:left | 1 | 2 | release | 35 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 27 | b8:down | 1 | 2 | release | 35 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 28 | b8:left | 1 | 2 | release | 35 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 29 | b8:left | 1 | 2 | release | 35 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 30 | b8:left | 1 | 2 | release | 34 | 2 | -105 | False | False | False | False | below_min_total_pushes |
| 31 | b7:left | 1 | 2 | release | 34 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 32 | b7:down | 1 | 2 | release | 34 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 33 | b4:up | 1 | 2 | release | 34 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 34 | b3:left | 1 | 2 | release | 34 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 35 | b3:left | 1 | 2 | release | 34 | 2 | -91 | False | False | False | False | selected_box_uncommitted |
| 36 | b7:left | 8 | 2 | release | 30 | 2 | -277 | False | True | False | False | ambiguous_monotone_committed_push |
| 37 | b6:left | 2 | 3 | release | 26 | 2 | 95 | True | False | False | False | committed_slot_reached |
| 38 | b9:up | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 39 | b8:left | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 40 | b9:left | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 41 | b9:left | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 42 | b8:down | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 43 | b9:left | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |
| 44 | b9:up | 1 | 3 | release | 26 | 2 | -267 | False | True | False | False | selected_box_uncommitted |

## 方法含义

A28 检验的是：A27 只给伪里程碑循环打标签仍不足，root 选择是否必须直接消费这些标签。当前 80-push 结果表明，root escape 能够在第 14 步后摆脱 A26/A27 的伪里程碑循环，并推进到 3 个目标和 release 阶段；但 release action 仍为 0，后段仍存在 release-stage 无进展循环。因此，A28 是阶段性改善而非求解配置，后续缺口集中在 release 阶段的可执行释放动作生成和价值排序。
