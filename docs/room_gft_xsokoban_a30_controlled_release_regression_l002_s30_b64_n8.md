# XSokoban first 第 2 关 Room-GFT A30 controlled release-regression 诊断

## 核心结论

A30 继承 A29，并在 release 阶段允许受控临时目标回退。该版本仍不接入 teacher/reference 在线动作。
A24 为 `max_steps`、pushes=80、decisions=78、goals=0；A25 为 `max_steps`、pushes=80、decisions=78、goals=0；A26 为 `max_steps`、pushes=80、decisions=39、goals=1；A27 为 `max_steps`、pushes=80、decisions=39、goals=1；A28 为 `max_steps`、pushes=80、decisions=45、goals=3；A29 为 `max_steps`、pushes=80、decisions=45、goals=3；A30 为 `max_steps`、pushes=30、decisions=14、goals=1。
A29/A30 首个决策差异为 `14`；A30 release macro 触发步数为 `0`，打开 release action 步数为 `0`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | release macro | opened release | value milestone | value loop | pseudo milestone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A24 | max_steps | 78 | 80 | 0 | 48 | 2 | 1 | 0 | NA | NA | NA | NA | NA |
| A25 | max_steps | 78 | 80 | 0 | 49 | 1 | 0 | 0 | NA | NA | NA | NA | NA |
| A26 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | NA | NA | 31 | 1 | NA |
| A27 | max_steps | 39 | 80 | 1 | 48 | 2 | 2 | 0 | NA | NA | 6 | 26 | 25 |
| A28 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | NA | NA | 12 | 19 | 0 |
| A29 | max_steps | 45 | 80 | 3 | 26 | 2 | 1 | 0 | 0 | 0 | 12 | 19 | 0 |
| A30 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 | 6 | 1 | 0 |

## A30 轨迹前缀

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

## 方法含义

A30 检验的是：在 A29 已确认 release 后缀不可采纳后，受控目标回退是否能生成释放动作宏。当前结果中 controlled release macro 尚未触发；`no_admissible_release_suffix` 出现 0 次，说明入口已经运行，但当前短搜索与目标不回退验收条件找不到可接受释放后缀。剩余缺口因此集中在受控目标回退、更长释放序列或 release 阶段候选生成，而不只是 root ranking。
