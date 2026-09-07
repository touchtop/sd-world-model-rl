# XSokoban first 第 2 关 Room-GFT A37 active agenda 诊断

## 核心结论

A37 继承 A36，复用 A35 multi-head checkpoint。新增部分是最小 active-agenda 层：先判定当前义务为 fill_frontier_slot、release_blocker 或 consume_release，再判断候选动作是否推进该义务。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2；A37 为 `max_steps`、pushes=80、decisions=42、goals=2。
A31/A37 首个决策差异为 `None`；A37 active-agenda 加载步数为 `42`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | active progress | off agenda | pseudo | release loop | consumed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | NA | NA | NA | NA | NA |
| A37 | max_steps | 42 | 80 | 2 | 29 | 2 | 25 | 17 | 17 | 13 | 1 |

## A37 轨迹前缀

| step | action | pushes | goals | stage | slot cost | release actions | score | agenda | blk delta | progress | off | loop | rel raw | rel eff | tier |
|---:|---|---:|---:|---|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|
| 0 | b6:right | 1 | 0 | fill | 50 | 0 | 1.021 | release_blocker | 0 | True | False | False | 0.701 | 0.701 | 5 |
| 1 | b2:up | 1 | 0 | fill | 51 | 0 | 0.837 | release_blocker | 1 | True | False | False | 1.015 | 1.015 | 6 |
| 2 | b6:right | 2 | 0 | fill | 49 | 0 | 0.859 | release_blocker | -3 | True | False | False | 1.378 | 0.2 | 5 |
| 3 | b4:up | 2 | 0 | fill | 47 | 0 | 1.381 | release_blocker | 0 | True | False | False | 0.553 | 0.553 | 5 |
| 4 | b3:up | 1 | 0 | fill | 48 | 0 | 0.108 | release_blocker | 0 | False | True | False | 0.457 | 0 | 1 |
| 5 | b4:right | 1 | 0 | fill | 48 | 0 | 1.624 | release_blocker | 2 | True | False | False | 0.977 | 0.977 | 6 |
| 6 | b8:right | 1 | 0 | fill | 48 | 0 | 0.043 | release_blocker | 0 | False | True | False | 0.264 | 0 | 2 |
| 7 | b7:left | 3 | 0 | fill | 47 | 0 | -0.014 | release_blocker | -2 | True | False | False | 0.124 | 0.124 | 5 |
| 8 | b7:up | 6 | 0 | fill | 44 | 0 | 0.918 | release_blocker | 0 | True | False | False | 0.302 | 0.302 | 5 |
| 9 | b5:left | 5 | 0 | fill | 41 | 0 | 1.175 | release_blocker | 0 | True | False | False | 0.503 | 0.503 | 5 |
| 10 | b3:left | 2 | 1 | fill | 49 | 0 | 1.199 | release_blocker | 0 | True | False | False | 0.881 | 0.881 | 7 |
| 11 | b6:down | 1 | 1 | fill | 48 | 0 | 0.403 | release_blocker | 0 | True | False | False | 0.417 | 0.417 | 5 |
| 12 | b3:down | 2 | 1 | fill | 48 | 0 | 1.428 | release_blocker | 2 | True | False | False | 1.801 | 1.801 | 6 |
| 13 | b3:up | 2 | 1 | fill | 48 | 0 | 0.908 | release_blocker | 2 | True | False | False | 1.226 | 1.226 | 6 |
| 14 | b6:right | 1 | 1 | fill | 48 | 0 | -0.942 | release_blocker | 0 | False | True | False | 0.043 | 0 | 2 |
| 15 | b8:up | 1 | 1 | fill | 48 | 0 | -0.782 | release_blocker | 0 | False | True | False | 0.129 | 0 | 2 |
| 16 | b8:left | 1 | 1 | fill | 47 | 0 | 1.161 | release_blocker | 0 | True | False | False | 0.617 | 0.617 | 5 |
| 17 | b8:left | 1 | 1 | fill | 46 | 0 | 1.26 | release_blocker | 0 | True | False | False | 0.632 | 0.632 | 5 |
| 18 | b8:left | 1 | 1 | fill | 45 | 0 | 2.118 | release_blocker | 1 | True | False | False | 1.078 | 1.078 | 6 |
| 19 | b2:left | 1 | 1 | fill | 44 | 0 | 1.733 | release_blocker | 0 | True | False | False | 0.573 | 0.573 | 5 |
| 20 | b8:left | 2 | 1 | fill | 44 | 0 | 1.375 | release_blocker | 1 | True | False | False | 0.898 | 0.898 | 6 |
| 21 | b8:up | 6 | 1 | fill | 40 | 0 | 1.819 | release_blocker | 0 | True | False | False | 0.808 | 0.808 | 5 |
| 22 | b3:down | 2 | 1 | fill | 40 | 0 | 2.707 | release_blocker | 3 | True | False | False | 2.543 | 2.543 | 6 |
| 23 | b3:up | 2 | 1 | fill | 40 | 0 | 1.747 | release_blocker | 3 | True | False | False | 1.764 | 1.764 | 6 |
| 24 | b6:left | 1 | 2 | fill | 49 | 0 | 0.472 | release_blocker | 0 | True | False | False | 0 | 0 | 7 |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | 0.724 | release_blocker | -2 | True | False | False | 2.998 | 0.2 | 5 |
| 26 | b3:down | 2 | 2 | release | 29 | 2 | -1.52 | consume_release | 5 | False | True | True | 2.649 | 0 | 1 |
| 27 | b3:up | 2 | 2 | release | 29 | 2 | -1.597 | consume_release | 5 | False | True | True | 2.627 | 0 | 1 |
| 28 | b4:up | 1 | 2 | release | 51 | 0 | 1.098 | consume_release | 2 | True | False | False | 1.418 | 1.418 | 7 |
| 29 | b0:down | 2 | 2 | release | 49 | 0 | 0.498 | release_blocker | -2 | True | False | False | 2.706 | 0.2 | 5 |
| 30 | b6:up | 1 | 2 | release | 29 | 2 | 0.285 | release_blocker | -2 | True | False | False | 2.491 | 0.2 | 5 |
| 31 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 32 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 33 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 34 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 35 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 36 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 37 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 38 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 39 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 40 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |
| 41 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | 5 | False | True | True | 2.337 | 0 | 1 |

## 方法含义

A37 检验专家建议中的 active obligation 命题：progress certificate 必须绑定当前义务。局部 release closure 或 blocker delta 只有在服务当前义务时才保留为 certified progress；重复上下文中的 release closure 被记录为 off-agenda/release-loop。
