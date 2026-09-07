# XSokoban first 第 2 关 Room-GFT A35 multi-head SD-consumer 诊断

## 核心结论

A35 继承 A34/A33/A32/A31，并加载 value、release-progress、loop-risk、trap-risk 四头 MLP checkpoint。默认配置只将组合分数作为 root prior/value 的软信号消费，不启用 root hard override；该版本不接入 teacher/reference 在线动作，也不替代 exact Sokoban dynamics。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2、release actions=2；A35 为 `max_steps`、pushes=80、decisions=42、goals=2、release actions=2。
A31/A35 首个决策差异为 `None`；A35 multi-head scorer 加载步数为 `42`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | multi-head loaded | positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | NA | NA |
| A35 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | 42 | 39 |

## A35 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | release actions | score | value | release | loop | trap | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0.899 | 3.958 | 0.701 | 0.018 | 0 | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 0 | 0.714 | 1.955 | 1.015 | 0.001 | 0.004 | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 0 | 1.158 | 5.502 | 1.378 | 0.002 | 0.6 | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 0 | 1.259 | 7.438 | 0.553 | 0.056 | 0.263 | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | 0 | 1.009 | 7.377 | 0.457 | 0.642 | 0.23 | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 0 | 1.501 | 7.116 | 0.977 | 0.002 | 0.028 | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 0 | 0.692 | 4.89 | 0.264 | 0.465 | 0.031 | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | 0 | -0.137 | -0.013 | 0.124 | 0.482 | 0.017 | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 0 | 0.796 | 5.041 | 0.302 | 0.326 | 0 | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 0 | 1.052 | 5.569 | 0.503 | 0 | 0.062 | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 0 | 1.076 | 4.582 | 0.881 | 0 | 0 | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 0 | 0.28 | 0.962 | 0.417 | 0.063 | 0.036 | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 0 | 1.306 | 3.776 | 1.801 | 0.038 | 0 | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 0 | 0.786 | 1.977 | 1.226 | 0.04 | 0.01 | committed_slot_reached |
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | 0 | -0.383 | -0.996 | 0.043 | 0.662 | 0 | selected_box_uncommitted |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | 0 | -0.187 | 0.062 | 0.129 | 0.678 | 0 | selected_box_uncommitted |
| 16 | b8:left | 1 | 1 | fill | 47 | 2 | 0 | 1.039 | 5.046 | 0.617 | 0.012 | 0 | selected_box_uncommitted |
| 17 | b8:left | 1 | 1 | fill | 46 | 2 | 0 | 1.138 | 5.639 | 0.632 | 0.013 | 0 | selected_box_uncommitted |
| 18 | b8:left | 1 | 1 | fill | 45 | 2 | 0 | 1.995 | 10 | 1.078 | 0.015 | 0.019 | below_min_total_pushes |
| 19 | b2:left | 1 | 1 | fill | 44 | 2 | 0 | 1.611 | 8.83 | 0.573 | 0 | 0.023 | selected_box_uncommitted |
| 20 | b8:left | 2 | 1 | fill | 44 | 1 | 0 | 1.253 | 5.657 | 0.898 | 0 | 0 | pre_release_bottleneck_truncated_corridor |
| 21 | b8:up | 6 | 1 | fill | 40 | 1 | 0 | 1.697 | 8.723 | 0.808 | 0 | 0.005 | ambiguous_monotone_committed_push |
| 22 | b3:down | 2 | 1 | fill | 40 | 1 | 0 | 2.585 | 10 | 2.543 | 0 | 0.032 | committed_slot_reached |
| 23 | b3:up | 2 | 1 | fill | 40 | 1 | 0 | 1.625 | 5.854 | 1.764 | 0.014 | 0.005 | committed_slot_reached |
| 24 | b6:left | 1 | 2 | fill | 49 | 1 | 0 | 0.349 | 2.338 | 0 | 0.052 | 0 | below_min_total_pushes |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | 2 | 1.675 | 3.74 | 2.998 | 0 | 0.276 | goal_lock_predecessor_root_replacement |
| 26 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 1.244 | 3.148 | 2.649 | 0.845 | 0.017 | committed_slot_reached |
| 27 | b3:up | 2 | 2 | release | 29 | 2 | 2 | 1.158 | 1.908 | 2.627 | 0.541 | 0.003 | committed_slot_reached |
| 28 | b4:up | 1 | 2 | release | 51 | 1 | 0 | 0.766 | 1.253 | 1.418 | 0.006 | 0 | below_min_total_pushes |
| 29 | b0:down | 2 | 2 | release | 49 | 1 | 0 | 1.332 | 4.33 | 2.706 | 0.325 | 0.731 | ambiguous_monotone_committed_push |
| 30 | b6:up | 1 | 2 | release | 29 | 2 | 2 | 1.032 | 3.348 | 2.491 | 0.962 | 0.33 | goal_lock_predecessor_root_replacement |
| 31 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 32 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 33 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 34 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 35 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 36 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 37 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 38 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 39 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 40 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |
| 41 | b3:down | 2 | 2 | release | 29 | 2 | 2 | 0.164 | -1.651 | 2.337 | 1.402 | 0.003 | committed_slot_reached |

## 方法含义

A35 的主要作用是检验 A34 单一 milestone scalar 是否造成机制混叠。四个头共享同一 SD 特征空间和同一离线 trace 来源，仍属于结构化 SD 表示的神经消费者，而不是新增在线 teacher 或手写动作规则。
