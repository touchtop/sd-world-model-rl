# XSokoban first 第 2 关 Room-GFT A38 agenda leaf potential 诊断

## 核心结论

A38 继承 A37，复用 A35 multi-head checkpoint。新增部分对应专家 27 点中的 E3：在 MCTS leaf value 中加入显式 agenda potential，检验 active-agenda 信号从 root soft prior 移到 leaf 消费端后是否改变闭环。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2；A38 为 `max_steps`、pushes=80、decisions=42、goals=2。
A31/A38 首个决策差异为 `None`；A38 leaf 加载步数为 `42`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | active progress | off agenda | leaf adj mean | leaf loop eval | leaf stale eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | NA | NA | NA | NA | NA |
| A38 | max_steps | 42 | 80 | 2 | 29 | 2 | 25 | 17 | -5.413 | 129 | 74 |

## A38 轨迹前缀

| step | action | pushes | goals | stage | slot cost | release actions | score | agenda | progress | off | loop | leaf type | leaf adj | leaf eval | leaf loop eval |
|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---:|---:|---:|
| 0 | b6:right | 1 | 0 | fill | 50 | 0 | 1.021 | release_blocker | True | False | False | release_blocker | 1.56 | 113 | 0 |
| 1 | b2:up | 1 | 0 | fill | 51 | 0 | 0.837 | release_blocker | True | False | False | release_blocker | 1.6 | 69 | 0 |
| 2 | b6:right | 2 | 0 | fill | 49 | 0 | 0.859 | release_blocker | True | False | False | release_blocker | 3.81 | 128 | 0 |
| 3 | b4:up | 2 | 0 | fill | 47 | 0 | 1.381 | release_blocker | True | False | False | release_blocker | -3.11 | 128 | 0 |
| 4 | b3:up | 1 | 0 | fill | 48 | 0 | 0.108 | release_blocker | False | True | False | release_blocker | -3.03 | 128 | 0 |
| 5 | b4:right | 1 | 0 | fill | 48 | 0 | 1.624 | release_blocker | True | False | False | release_blocker | -3.07 | 128 | 0 |
| 6 | b8:right | 1 | 0 | fill | 48 | 0 | 0.043 | release_blocker | False | True | False | release_blocker | 3.93 | 128 | 0 |
| 7 | b7:left | 3 | 0 | fill | 47 | 0 | -0.014 | release_blocker | True | False | False | release_blocker | 3.93 | 128 | 0 |
| 8 | b7:up | 6 | 0 | fill | 44 | 0 | 0.918 | release_blocker | True | False | False | release_blocker | -0.78 | 128 | 0 |
| 9 | b5:left | 5 | 0 | fill | 41 | 0 | 1.175 | release_blocker | True | False | False | release_blocker | 1.84 | 128 | 0 |
| 10 | b3:left | 2 | 1 | fill | 49 | 0 | 1.199 | release_blocker | True | False | False | release_blocker | 1.96 | 128 | 0 |
| 11 | b6:down | 1 | 1 | fill | 48 | 0 | 0.403 | release_blocker | True | False | False | release_blocker | -2.86 | 128 | 0 |
| 12 | b3:down | 2 | 1 | fill | 48 | 0 | 1.428 | release_blocker | True | False | False | release_blocker | -2.82 | 128 | 0 |
| 13 | b3:up | 2 | 1 | fill | 48 | 0 | 0.908 | release_blocker | True | False | False | release_blocker | -2.82 | 128 | 0 |
| 14 | b6:right | 1 | 1 | fill | 48 | 0 | -0.942 | release_blocker | False | True | False | release_blocker | -2.82 | 128 | 0 |
| 15 | b8:up | 1 | 1 | fill | 48 | 0 | -0.782 | release_blocker | False | True | False | release_blocker | -2.82 | 128 | 0 |
| 16 | b8:left | 1 | 1 | fill | 47 | 0 | 1.161 | release_blocker | True | False | False | release_blocker | -2.82 | 128 | 0 |
| 17 | b8:left | 1 | 1 | fill | 46 | 0 | 1.26 | release_blocker | True | False | False | release_blocker | -2.78 | 128 | 0 |
| 18 | b8:left | 1 | 1 | fill | 45 | 0 | 2.118 | release_blocker | True | False | False | release_blocker | -2.74 | 128 | 0 |
| 19 | b2:left | 1 | 1 | fill | 44 | 0 | 1.733 | release_blocker | True | False | False | release_blocker | -0.45 | 128 | 0 |
| 20 | b8:left | 2 | 1 | fill | 44 | 0 | 1.375 | release_blocker | True | False | False | release_blocker | -0.41 | 128 | 0 |
| 21 | b8:up | 6 | 1 | fill | 40 | 0 | 1.819 | release_blocker | True | False | False | release_blocker | -0.66 | 128 | 0 |
| 22 | b3:down | 2 | 1 | fill | 40 | 0 | 2.707 | release_blocker | True | False | False | release_blocker | 2 | 128 | 0 |
| 23 | b3:up | 2 | 1 | fill | 40 | 0 | 1.747 | release_blocker | True | False | False | release_blocker | 2 | 128 | 0 |
| 24 | b6:left | 1 | 2 | fill | 49 | 0 | 0.472 | release_blocker | True | False | False | release_blocker | 2 | 128 | 0 |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | 0.724 | release_blocker | True | False | False | release_blocker | -5.86 | 128 | 10 |
| 26 | b3:down | 2 | 2 | release | 29 | 2 | -1.52 | consume_release | False | True | True | consume_release | -14.31 | 128 | 8 |
| 27 | b3:up | 2 | 2 | release | 29 | 2 | -1.597 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 28 | b4:up | 1 | 2 | release | 51 | 0 | 1.098 | consume_release | True | False | False | consume_release | -14.31 | 128 | 6 |
| 29 | b0:down | 2 | 2 | release | 49 | 0 | 0.498 | release_blocker | True | False | False | release_blocker | -5.94 | 128 | 10 |
| 30 | b6:up | 1 | 2 | release | 29 | 2 | 0.285 | release_blocker | True | False | False | release_blocker | -5.86 | 128 | 11 |
| 31 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 32 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 33 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 34 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 35 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 36 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 37 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 38 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 39 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 40 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |
| 41 | b3:down | 2 | 2 | release | 29 | 2 | -2.474 | consume_release | False | True | True | consume_release | -14.31 | 128 | 7 |

## 方法含义

A38 只检验 E3 leaf agenda potential，不新增图网络、teacher imitation、hard gate 或 agenda-option tree。若该版本仍与 A37 同轨迹，则说明仅将显式 agenda potential 接入 leaf 仍不足以改变当前 root/top-child 优势。
