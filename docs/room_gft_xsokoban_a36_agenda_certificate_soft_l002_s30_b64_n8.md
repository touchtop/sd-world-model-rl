# XSokoban first 第 2 关 Room-GFT A36 agenda certificate 诊断

## 核心结论

A36 继承 A35，并复用 A35 multi-head checkpoint。新增部分不是新 teacher 或新动作规则，而是最小 agenda progress-certificate gate：release-progress 只有在完成义务、减少 blocker、推进 frontier slot 或消费 opened release 时才保留为有效进展。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2；A36 为 `max_steps`、pushes=30、decisions=14、goals=1。
A31/A36 首个决策差异为 `14`；A36 certificate 加载步数为 `14`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | certified | pseudo | stale release | consumed release |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | NA | NA | NA | NA |
| A36 | max_steps | 14 | 30 | 1 | 48 | 0 | 13 | 1 | 0 | 0 |

## A36 轨迹前缀

| step | action | pushes | goals | stage | slot cost | release actions | score | rel raw | rel eff | cert | pseudo | stale | tier | macro reason |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 0 | 1.021 | 0.701 | 0.701 | True | False | False | 5 | selected_box_uncommitted |
| 1 | b2:up | 1 | 0 | fill | 51 | 0 | 0.837 | 1.015 | 1.015 | True | False | False | 6 | below_min_total_pushes |
| 2 | b6:right | 2 | 0 | fill | 49 | 0 | 0.919 | 1.378 | 0.35 | True | False | False | 5 | pre_release_bottleneck_truncated_corridor |
| 3 | b4:up | 2 | 0 | fill | 47 | 0 | 1.381 | 0.553 | 0.553 | True | False | False | 5 | no_monotone_committed_push |
| 4 | b3:up | 1 | 0 | fill | 48 | 0 | 0.58 | 0.457 | 0 | False | True | False | 3 | below_min_total_pushes |
| 5 | b4:right | 1 | 0 | fill | 48 | 0 | 1.624 | 0.977 | 0.977 | True | False | False | 6 | selected_box_uncommitted |
| 6 | b8:right | 1 | 0 | fill | 48 | 0 | 0.815 | 0.264 | 0.264 | True | False | False | 5 | selected_box_uncommitted |
| 7 | b7:left | 3 | 0 | fill | 47 | 0 | -0.014 | 0.124 | 0.124 | True | False | False | 5 | pre_release_compatible_dependency_macro |
| 8 | b7:up | 6 | 0 | fill | 44 | 0 | 0.918 | 0.302 | 0.302 | True | False | False | 5 | pre_release_compatible_dependency_macro |
| 9 | b5:left | 5 | 0 | fill | 41 | 0 | 1.175 | 0.503 | 0.503 | True | False | False | 5 | pre_release_compatible_dependency_macro |
| 10 | b3:left | 2 | 1 | fill | 49 | 0 | 1.199 | 0.881 | 0.881 | True | False | False | 7 | committed_slot_reached |
| 11 | b6:down | 1 | 1 | fill | 48 | 0 | 0.403 | 0.417 | 0.417 | True | False | False | 5 | selected_box_uncommitted |
| 12 | b3:down | 2 | 1 | fill | 48 | 0 | 1.428 | 1.801 | 1.801 | True | False | False | 6 | committed_slot_reached |
| 13 | b3:up | 2 | 1 | fill | 48 | 0 | 0.908 | 1.226 | 1.226 | True | False | False | 6 | committed_slot_reached |

## 方法含义

A36 检验专家建议中的最小命题：release opened 不等于 release consumed，没有 progress certificate 的 release-progress 不应继续作为正向进展消费。该版本仍只使用离线 checkpoint，在线求解不调用 teacher/reference。
