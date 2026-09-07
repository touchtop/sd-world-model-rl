# XSokoban first 第 2 关 Room-GFT A24 pre-release root trap guard 诊断

## 核心结论

A24 继承 A23，保留预释放 corridor macro 截断，并新增根层 pre-release trap 风险评分与 guard；该版本不接入 teacher/reference 在线动作。
A22 为 `max_steps`、pushes=80、decisions=51、goals=2；A23 为 `max_steps`、pushes=80、decisions=53、goals=2；A24 为 `max_steps`、pushes=30、decisions=28、goals=0。
A22/A24 首个决策差异为 `3`；A23/A24 首个决策差异为 `4`；A24 bottleneck 截断步骤数为 `1`，被选 severe root-trap 步数为 `23`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | release macro steps | bottleneck truncations | selected severe root-trap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A22 | max_steps | 51 | 80 | 2 | 60 | 2 | 0 | 0 | 0 | NA | NA |
| A23 | max_steps | 53 | 80 | 2 | 60 | 2 | 0 | 0 | 0 | 2 | NA |
| A24 | max_steps | 28 | 30 | 0 | 48 | 2 | 1 | 0 | 0 | 1 | 23 |

## A24 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | trap score | severe trap | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | -6 | False | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 26 | False | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 1 | -34 | False | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | -34 | False | no_monotone_committed_push |
| 4 | b7:right | 1 | 0 | fill | 48 | 2 | -34 | False | selected_box_uncommitted |
| 5 | b3:up | 1 | 0 | recovery | 49 | 2 | -298 | True | below_min_total_pushes |
| 6 | b4:right | 1 | 0 | recovery | 49 | 2 | -60 | True | selected_box_uncommitted |
| 7 | b6:left | 1 | 0 | fill | 49 | 2 | -32 | True | selected_box_uncommitted |
| 8 | b4:up | 1 | 0 | fill | 49 | 2 | -36 | True | selected_box_uncommitted |
| 9 | b4:up | 1 | 0 | fill | 49 | 2 | -32 | True | selected_box_uncommitted |
| 10 | b2:left | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 11 | b6:up | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 12 | b9:left | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 13 | b5:left | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 14 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 15 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 16 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 17 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 18 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 19 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 20 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 21 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 22 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 23 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 24 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 25 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |
| 26 | b9:up | 1 | 0 | fill | 48 | 2 | -36 | True | selected_box_uncommitted |
| 27 | b7:down | 1 | 0 | fill | 48 | 2 | -40 | True | selected_box_uncommitted |

## 方法含义

A24 检验的是：在 A23 已经截断过度 corridor macro 之后，根选择层能否识别并避开把 target/stand 前置依赖转化为不可释放 stand-lock 的候选。若仍无改善，则说明缺口不只是风险排序，而是需要更强的阶段里程碑或多步替代调度生成。
