# XSokoban first 第 2 关 Room-GFT A32 learned SD-consumer 诊断

## 核心结论

A32 继承 A31，并加载离线 trace 蒸馏得到的 compact SD candidate scorer。默认配置仅将 learned score 作为 root prior/value 的软信号消费，不启用 learned scorer 对父类根选择的硬覆盖；该版本不接入 teacher/reference 在线动作，也不替代 exact Sokoban dynamics。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2、release actions=2；A32 为 `no_action`、pushes=6、decisions=4、goals=0、release actions=0。
A31/A32 首个决策差异为 `1`；A32 learned scorer 加载步数为 `4`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | learned loaded | learned positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | 2 | 2 | NA | NA |
| A32 | no_action | 4 | 6 | 0 | 47 | 2 | 1 | 0 | 4 | 2 |

## A32 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | release actions | learned score | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0.705 | selected_box_uncommitted |
| 1 | b6:right | 2 | 0 | fill | 48 | 1 | 0 | 1.473 | pre_release_bottleneck_truncated_corridor |
| 2 | b4:up | 2 | 0 | fill | 46 | 2 | 0 | -0.278 | no_monotone_committed_push |
| 3 | b7:right | 1 | 0 | recovery | 47 | 2 | 0 | -2.097 | selected_box_uncommitted |

## 方法含义

A32 的主要作用是验证 learned-consumer 接口链路：统一 SD 特征、离线 checkpoint、root ranking 消费和 trace 记录已经连通。该版本的 checkpoint 来自 A28-A31 trace 的启发式蒸馏，不能视为充分训练的神经 TVA。
