# XSokoban first 第 2 关 Room-GFT A22 phased release macro 诊断

## 核心结论

A22 继承 A21，但将 stand-lock 从全局强 guard 改为阶段化 release/action 生成入口；默认在完成至少 2 个目标后才进入 release 阶段。
A20 为 `max_steps`、goals=2；A21 为 `max_steps`、goals=0；A22 为 `max_steps`、pushes=80、decisions=51、goals=2。
A20/A22 首个决策差异为 `None`；A21/A22 首个决策差异为 `3`。
A22 与 A20 目标数持平，主要检验阶段化消费是否避免 A21 的退化。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | release actions | release-stage decisions | release macro steps | pre-release macro steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A20 | max_steps | 51 | 80 | 2 | 60 | 2 | 0 | 0 | NA | NA | NA |
| A21 | max_steps | 77 | 80 | 0 | 48 | 2 | 1 | 0 | NA | NA | NA |
| A22 | max_steps | 51 | 80 | 2 | 60 | 2 | 0 | 0 | 32 | 0 | 1 |

## A22 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dependencies | locked after | phased macro |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0 |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 0 | 0 |
| 2 | b6:right | 4 | 0 | fill | 47 | 2 | 0 | 0 |
| 3 | b3:up | 1 | 0 | fill | 48 | 2 | 1 | 0 |
| 4 | b4:right | 1 | 0 | fill | 48 | 2 | 1 | 0 |
| 5 | b7:left | 3 | 0 | fill | 47 | 1 | 1 | 0 |
| 6 | b7:up | 6 | 0 | fill | 43 | 1 | 1 | 0 |
| 7 | b8:left | 1 | 0 | fill | 42 | 1 | 1 | 0 |
| 8 | b8:left | 1 | 0 | fill | 41 | 1 | 1 | 0 |
| 9 | b8:left | 1 | 0 | fill | 40 | 1 | 1 | 0 |
| 10 | b4:up | 3 | 1 | fill | 46 | 2 | 1 | 0 |
| 11 | b3:down | 2 | 1 | fill | 46 | 2 | 1 | 0 |
| 12 | b3:up | 2 | 1 | fill | 46 | 2 | 1 | 0 |
| 13 | b8:left | 8 | 1 | fill | 42 | 2 | 1 | 0 |
| 14 | b3:down | 2 | 1 | fill | 42 | 2 | 1 | 0 |
| 15 | b3:up | 2 | 1 | fill | 42 | 2 | 1 | 0 |
| 16 | b5:up | 1 | 1 | fill | 41 | 2 | 1 | 0 |
| 17 | b4:up | 1 | 1 | fill | 40 | 2 | 1 | 0 |
| 18 | b0:left | 1 | 2 | fill | 49 | 2 | 1 | 0 |
| 19 | b4:down | 2 | 2 | release | 49 | 2 | 1 | 0 |
| 20 | b4:up | 2 | 2 | release | 49 | 2 | 1 | 0 |
| 21 | b0:down | 2 | 2 | release | 49 | 2 | 1 | 0 |
| 22 | b0:up | 2 | 2 | release | 49 | 2 | 1 | 0 |
| 23 | b7:down | 1 | 2 | release | 48 | 2 | 1 | 0 |
| 24 | b8:left | 1 | 2 | release | 47 | 2 | 1 | 0 |
| 25 | b8:left | 1 | 2 | release | 46 | 2 | 1 | 0 |
| 26 | b8:left | 1 | 2 | release | 45 | 2 | 1 | 0 |
| 27 | b4:down | 2 | 2 | release | 45 | 2 | 1 | 0 |
| 28 | b7:left | 1 | 2 | release | 45 | 2 | 1 | 0 |
| 29 | b4:down | 2 | 2 | release | 45 | 2 | 1 | 0 |
| 30 | b4:up | 1 | 2 | release | 66 | 2 | 2 | 0 |
| 31 | b7:left | 1 | 2 | release | 65 | 2 | 2 | 0 |
| 32 | b7:down | 1 | 2 | release | 64 | 2 | 2 | 0 |
| 33 | b8:left | 1 | 2 | release | 63 | 2 | 2 | 0 |
| 34 | b8:left | 1 | 2 | release | 62 | 2 | 2 | 0 |
| 35 | b9:left | 1 | 2 | release | 62 | 2 | 2 | 0 |
| 36 | b6:up | 1 | 2 | release | 62 | 2 | 2 | 0 |
| 37 | b6:up | 1 | 2 | release | 62 | 2 | 2 | 0 |
| 38 | b4:left | 1 | 2 | release | 61 | 2 | 2 | 0 |
| 39 | b4:left | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 40 | b9:left | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 41 | b6:up | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 42 | b5:left | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 43 | b5:left | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 44 | b9:right | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 45 | b9:up | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 46 | b7:right | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 47 | b7:down | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 48 | b9:left | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 49 | b9:up | 1 | 2 | release | 60 | 2 | 2 | 0 |
| 50 | b7:up | 1 | 2 | release | 60 | 2 | 2 | 0 |

## 方法含义

A22 检验的是：复杂 Sokoban 决策机制是否应按里程碑阶段消费不同的 SD 摘要。若 A22 保留 A20 前缀而避免 A21 的目标数退化，则说明阶段化消费方向成立；若仍无法触发 release macro，则说明下一步需要修订阶段入口或依赖绑定条件。
