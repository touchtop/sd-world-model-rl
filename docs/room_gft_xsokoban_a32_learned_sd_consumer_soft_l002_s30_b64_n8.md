# XSokoban first 第 2 关 Room-GFT A32 learned SD-consumer 诊断

## 核心结论

A32 继承 A31，并加载离线 trace 蒸馏得到的 compact SD candidate scorer。默认配置仅将 learned score 作为 root prior/value 的软信号消费，不启用 learned scorer 对父类根选择的硬覆盖；该版本不接入 teacher/reference 在线动作，也不替代 exact Sokoban dynamics。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2、release actions=2；A32 为 `max_steps`、pushes=30、decisions=14、goals=1、release actions=0。
A31/A32 首个决策差异为 `14`；A32 learned scorer 加载步数为 `14`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | learned loaded | learned positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | NA | NA |
| A32 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 14 | 8 |

## A32 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | release actions | learned score | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0.705 | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 0 | 1.096 | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 0 | -1.098 | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 0 | -0.429 | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | 0 | -2.342 | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 0 | 1.99 | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 0 | -0.57 | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | 0 | -2.023 | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 0 | -0.203 | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 0 | 0.439 | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 0 | 3.25 | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 0 | 0.248 | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 0 | 3.109 | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 0 | 0.928 | committed_slot_reached |

## 方法含义

A32 的主要作用是验证 learned-consumer 接口链路：统一 SD 特征、离线 checkpoint、root ranking 消费和 trace 记录已经连通。该版本的 checkpoint 来自 A28-A31 trace 的启发式蒸馏，不能视为充分训练的神经 TVA。
