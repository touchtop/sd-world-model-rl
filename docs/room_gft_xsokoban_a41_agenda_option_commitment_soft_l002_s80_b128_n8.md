# XSokoban first 第 2 关 Room-GFT A41 agenda-option commitment 诊断

## 核心结论

A41 继承 A40，对应专家 27 点中的第 14 点和第 6 点：A40 负责在 root 处发现短程 agenda-option，A41 只在该 option 已达到 tier-7 且 `agenda_done=true` 时，执行一个受限承诺后缀。
A40 为 `max_steps`、pushes=80、decisions=53、goals=2；A41 为 `max_steps`、pushes=80、decisions=53、goals=2。
A40/A41 首个决策差异为 `30`；A41 commitment 触发 `1` 次，额外执行 `2` 个承诺 push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | option done | commitment triggers | commitment extra pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A40 | max_steps | 53 | 80 | 2 | 29 | 2 | 22 | NA | NA |
| A41 | max_steps | 53 | 80 | 2 | 41 | 0 | 1 | 1 | 2 |

## A41 轨迹前缀

| step | action | executed pushes | goals | stage | slot cost | option tier | commitment | reason | committed extra |
|---:|---|---:|---:|---|---:|---:|---|---|---:|
| 0 | b6:right | 1 | 0 | fill | 50 | None | False | no_selected_option | 0 |
| 1 | b2:up | 1 | 0 | fill | 51 | None | False | no_selected_option | 0 |
| 2 | b6:right | 2 | 0 | fill | 49 | None | False | no_selected_option | 0 |
| 3 | b4:up | 2 | 0 | fill | 47 | None | False | no_selected_option | 0 |
| 4 | b3:up | 1 | 0 | fill | 48 | None | False | no_selected_option | 0 |
| 5 | b4:right | 1 | 0 | fill | 48 | None | False | no_selected_option | 0 |
| 6 | b8:right | 1 | 0 | fill | 48 | None | False | no_selected_option | 0 |
| 7 | b7:left | 3 | 0 | fill | 47 | None | False | no_selected_option | 0 |
| 8 | b7:up | 6 | 0 | fill | 44 | None | False | no_selected_option | 0 |
| 9 | b5:left | 5 | 0 | fill | 41 | None | False | no_selected_option | 0 |
| 10 | b3:left | 2 | 1 | fill | 49 | None | False | no_selected_option | 0 |
| 11 | b6:down | 1 | 1 | fill | 48 | None | False | no_selected_option | 0 |
| 12 | b3:down | 2 | 1 | fill | 48 | None | False | no_selected_option | 0 |
| 13 | b3:up | 2 | 1 | fill | 48 | None | False | no_selected_option | 0 |
| 14 | b6:right | 1 | 1 | fill | 48 | None | False | no_selected_option | 0 |
| 15 | b8:up | 1 | 1 | fill | 48 | None | False | no_selected_option | 0 |
| 16 | b8:left | 1 | 1 | fill | 47 | None | False | no_selected_option | 0 |
| 17 | b8:left | 1 | 1 | fill | 46 | None | False | no_selected_option | 0 |
| 18 | b8:left | 1 | 1 | fill | 45 | None | False | no_selected_option | 0 |
| 19 | b2:left | 1 | 1 | fill | 44 | None | False | no_selected_option | 0 |
| 20 | b8:left | 2 | 1 | fill | 44 | None | False | no_selected_option | 0 |
| 21 | b8:up | 6 | 1 | fill | 40 | None | False | no_selected_option | 0 |
| 22 | b3:down | 2 | 1 | fill | 40 | None | False | no_selected_option | 0 |
| 23 | b3:up | 2 | 1 | fill | 40 | None | False | no_selected_option | 0 |
| 24 | b6:left | 1 | 2 | fill | 49 | None | False | no_selected_option | 0 |
| 25 | b6:up | 1 | 2 | release | 29 | None | False | no_selected_option | 0 |
| 26 | b4:up | 1 | 2 | release | 51 | None | False | no_selected_option | 0 |
| 27 | b0:down | 2 | 2 | release | 49 | None | False | no_selected_option | 0 |
| 28 | b6:up | 1 | 2 | release | 29 | None | False | no_selected_option | 0 |
| 29 | b6:up | 3 | 2 | release | 41 | 7 | True | validated_option_committed | 2 |
| 30 | b9:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 31 | b8:down | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 32 | b3:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 33 | b8:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 34 | b8:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 35 | b7:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 36 | b7:down | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 37 | b9:up | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 38 | b7:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 39 | b3:left | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 40 | b7:right | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 41 | b9:right | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 42 | b8:right | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 43 | b7:down | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 44 | b8:up | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 45 | b8:right | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 46 | b7:down | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 47 | b9:up | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 48 | b8:up | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 49 | b8:down | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 50 | b8:up | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 51 | b7:up | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |
| 52 | b7:right | 1 | 2 | fill | 41 | None | False | no_selected_option | 0 |

## 方法含义

A41 不新增 teacher，也不把参考轨迹作为在线锚定；它检验的是模型自身在 A40 中已经发现的 option 是否必须跨闭环决策保持。本轮结果显示，承诺机制可以完成当前 consume-release 义务，但仍未求解第二关；下一缺口应定位到 release 完成后的阶段化目标、slot-distance/value 表达和可复用函数近似，而不是继续单点规则叠加。
