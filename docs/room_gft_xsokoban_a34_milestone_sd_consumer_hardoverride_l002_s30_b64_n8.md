# XSokoban first 第 2 关 Room-GFT A34 milestone SD-consumer 诊断

## 核心结论

A34 继承 A33/A32/A31，并加载 future-window milestone target 训练得到的 MLP checkpoint。默认配置只把 milestone score 作为 root prior/value 的软信号消费，不启用 root hard override；该版本不接入 teacher/reference 在线动作，也不替代 exact Sokoban dynamics。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2、release actions=2；A34 为 `no_action`、pushes=6、decisions=4、goals=0、release actions=0。
A31/A34 首个决策差异为 `1`；A34 milestone scorer 加载步数为 `4`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | milestone loaded | positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | NA | NA |
| A34 | no_action | 4 | 6 | 0 | 47 | 2 | 1 | 0 | 4 | 3 |

## A34 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | release actions | milestone score | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 1.477 | selected_box_uncommitted |
| 1 | b6:right | 2 | 0 | fill | 48 | 2 | 0 | 1.004 | pre_release_bottleneck_truncated_corridor |
| 2 | b4:up | 2 | 0 | fill | 46 | 2 | 0 | 2.008 | no_monotone_committed_push |
| 3 | b7:right | 1 | 0 | recovery | 47 | 2 | 0 | -0.446 | selected_box_uncommitted |

## 方法含义

A34 的主要作用是检验 future-window milestone 目标是否优于 A33 的即时 SD progress 目标。该 checkpoint 仍是离线 trace 蒸馏结果，不能视为充分训练的通用求解器。
