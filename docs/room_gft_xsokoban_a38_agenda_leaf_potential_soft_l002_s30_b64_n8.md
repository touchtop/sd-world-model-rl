# XSokoban first 第 2 关 Room-GFT A38 agenda leaf potential 诊断

## 核心结论

A38 继承 A37，复用 A35 multi-head checkpoint。新增部分对应专家 27 点中的 E3：在 MCTS leaf value 中加入显式 agenda potential，检验 active-agenda 信号从 root soft prior 移到 leaf 消费端后是否改变闭环。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2；A38 为 `max_steps`、pushes=30、decisions=14、goals=1。
A31/A38 首个决策差异为 `14`；A38 leaf 加载步数为 `14`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | active progress | off agenda | leaf adj mean | leaf loop eval | leaf stale eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | NA | NA | NA | NA | NA |
| A38 | max_steps | 14 | 30 | 1 | 48 | 0 | 12 | 2 | 0.01 | 0 | 0 |

## A38 轨迹前缀

| step | action | pushes | goals | stage | slot cost | release actions | score | agenda | progress | off | loop | leaf type | leaf adj | leaf eval | leaf loop eval |
|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---:|---:|---:|
| 0 | b6:right | 1 | 0 | fill | 50 | 0 | 1.021 | release_blocker | True | False | False | release_blocker | 1.56 | 60 | 0 |
| 1 | b2:up | 1 | 0 | fill | 51 | 0 | 0.837 | release_blocker | True | False | False | release_blocker | 1.6 | 38 | 0 |
| 2 | b6:right | 2 | 0 | fill | 49 | 0 | 0.859 | release_blocker | True | False | False | release_blocker | 3.81 | 64 | 0 |
| 3 | b4:up | 2 | 0 | fill | 47 | 0 | 1.381 | release_blocker | True | False | False | release_blocker | -3.11 | 64 | 0 |
| 4 | b3:up | 1 | 0 | fill | 48 | 0 | 0.108 | release_blocker | False | True | False | release_blocker | -3.03 | 64 | 0 |
| 5 | b4:right | 1 | 0 | fill | 48 | 0 | 1.624 | release_blocker | True | False | False | release_blocker | -3.07 | 64 | 0 |
| 6 | b8:right | 1 | 0 | fill | 48 | 0 | 0.043 | release_blocker | False | True | False | release_blocker | 3.93 | 64 | 0 |
| 7 | b7:left | 3 | 0 | fill | 47 | 0 | -0.014 | release_blocker | True | False | False | release_blocker | 3.93 | 64 | 0 |
| 8 | b7:up | 6 | 0 | fill | 44 | 0 | 0.918 | release_blocker | True | False | False | release_blocker | -0.78 | 64 | 0 |
| 9 | b5:left | 5 | 0 | fill | 41 | 0 | 1.175 | release_blocker | True | False | False | release_blocker | 1.84 | 64 | 0 |
| 10 | b3:left | 2 | 1 | fill | 49 | 0 | 1.199 | release_blocker | True | False | False | release_blocker | 1.96 | 64 | 0 |
| 11 | b6:down | 1 | 1 | fill | 48 | 0 | 0.403 | release_blocker | True | False | False | release_blocker | -2.86 | 64 | 0 |
| 12 | b3:down | 2 | 1 | fill | 48 | 0 | 1.428 | release_blocker | True | False | False | release_blocker | -2.82 | 64 | 0 |
| 13 | b3:up | 2 | 1 | fill | 48 | 0 | 0.908 | release_blocker | True | False | False | release_blocker | -2.82 | 64 | 0 |

## 方法含义

A38 只检验 E3 leaf agenda potential，不新增图网络、teacher imitation、hard gate 或 agenda-option tree。若该版本仍与 A37 同轨迹，则说明仅将显式 agenda potential 接入 leaf 仍不足以改变当前 root/top-child 优势。
