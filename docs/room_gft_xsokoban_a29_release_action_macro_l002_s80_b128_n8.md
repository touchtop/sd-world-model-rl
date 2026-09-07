# XSokoban first 第 2 关 Room-GFT A29 release-action macro 诊断

## 核心结论

A29 继承 A28，并只在 release 阶段新增释放动作宏。该版本仍不接入 teacher/reference 在线动作。
A24 为 `max_steps`、pushes=80、decisions=78、goals=0；A25 为 `max_steps`、pushes=80、decisions=78、goals=0；A26 为 `max_steps`、pushes=80、decisions=39、goals=1；A27 为 `max_steps`、pushes=80、decisions=39、goals=1；A28 为 `max_steps`、pushes=80、decisions=45、goals=3；A29 为 `max_steps`、pushes=80、decisions=45、goals=3。
A28/A29 首个决策差异为 `None`；A29 release macro 触发步数为 `0`，打开 release action 步数为 `0`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | release macro | opened release | value milestone | value loop | pseudo milestone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A24 | max_steps | 78 | 80 | 0 | 48 | 2 | 1 | 0 | NA | NA | NA | NA | NA |
| A25 | max_steps | 78 | 80 | 0 | 49 | 1 | 0 | 0 | NA | NA | NA | NA | NA |
| A26 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | NA | NA | 31 | 1 | NA |
| A27 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | NA | NA | 6 | 26 | 25 |
| A28 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | NA | NA | 12 | 19 | 0 |
| A29 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | 0 | 0 | 12 | 19 | 0 |

## A29 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | release macro | opened release | sd value | milestone | loop risk | pseudo | macro reason |
|---:|---|---:|---:|---|---:|---:|---|---|---:|---|---|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | False | False | 18 | False | False | False | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | False | False | 54 | True | False | False | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | False | False | 34 | True | False | False | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | False | False | 10 | False | False | False | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | False | False | -144 | False | True | False | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | False | False | 162 | True | False | False | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | False | False | 32 | False | False | False | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | False | False | -58 | False | False | False | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | False | False | 26 | False | False | False | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | False | False | 10 | False | False | False | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | False | False | 210 | True | False | False | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | False | False | 18 | False | False | False | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | False | False | 166 | True | False | False | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | False | False | 106 | True | False | False | committed_slot_reached |
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | False | False | -108 | False | True | False | selected_box_uncommitted |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | False | False | -84 | False | True | False | selected_box_uncommitted |
| 16 | b8:left | 1 | 1 | fill | 47 | 2 | False | False | 10 | False | False | False | selected_box_uncommitted |
| 17 | b8:left | 1 | 1 | fill | 46 | 2 | False | False | 10 | False | False | False | selected_box_uncommitted |
| 18 | b8:left | 1 | 1 | fill | 45 | 2 | False | False | 100 | True | False | False | below_min_total_pushes |
| 19 | b2:left | 1 | 1 | fill | 44 | 2 | False | False | 10 | False | False | False | selected_box_uncommitted |
| 20 | b8:left | 2 | 1 | fill | 44 | 1 | False | False | 92 | True | False | False | pre_release_bottleneck_truncated_corridor |
| 21 | b8:up | 6 | 1 | fill | 40 | 1 | False | False | 74 | False | False | False | ambiguous_monotone_committed_push |
| 22 | b3:down | 2 | 1 | fill | 40 | 1 | False | False | 212 | True | False | False | committed_slot_reached |
| 23 | b3:up | 2 | 1 | fill | 40 | 1 | False | False | 152 | True | False | False | committed_slot_reached |
| 24 | b6:left | 1 | 2 | fill | 49 | 1 | False | False | 150 | True | False | False | below_min_total_pushes |
| 25 | b6:down | 4 | 2 | release | 35 | 2 | False | False | 34 | False | False | False | phased_release_dependency_macro |
| 26 | b9:left | 1 | 2 | release | 35 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 27 | b8:down | 1 | 2 | release | 35 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 28 | b8:left | 1 | 2 | release | 35 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 29 | b8:left | 1 | 2 | release | 35 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 30 | b8:left | 1 | 2 | release | 34 | 2 | False | False | -105 | False | False | False | below_min_total_pushes |
| 31 | b7:left | 1 | 2 | release | 34 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 32 | b7:down | 1 | 2 | release | 34 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 33 | b4:up | 1 | 2 | release | 34 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 34 | b3:left | 1 | 2 | release | 34 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 35 | b3:left | 1 | 2 | release | 34 | 2 | False | False | -91 | False | False | False | selected_box_uncommitted |
| 36 | b7:left | 8 | 2 | release | 30 | 2 | False | False | -277 | False | True | False | ambiguous_monotone_committed_push |
| 37 | b6:left | 2 | 3 | release | 26 | 2 | False | False | 95 | True | False | False | committed_slot_reached |
| 38 | b9:up | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 39 | b8:left | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 40 | b9:left | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 41 | b9:left | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 42 | b8:down | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 43 | b9:left | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |
| 44 | b9:up | 1 | 3 | release | 26 | 2 | False | False | -267 | False | True | False | selected_box_uncommitted |

## 方法含义

A29 检验的是：在 A28 已能进入 release 阶段后，是否需要显式生成释放动作宏。当前结果中 release-action macro 尚未触发；`no_admissible_release_suffix` 出现 19 次，说明入口已经运行，但当前短搜索与目标不回退验收条件找不到可接受释放后缀。剩余缺口因此集中在受控目标回退、更长释放序列或 release 阶段候选生成，而不只是 root ranking。
