# XSokoban first 第 2 关 Room-GFT A41 agenda-option commitment 诊断

## 核心结论

A41 继承 A40，对应专家 27 点中的第 14 点和第 6 点：A40 负责在 root 处发现短程 agenda-option，A41 只在该 option 已达到 tier-7 且 `agenda_done=true` 时，执行一个受限承诺后缀。
A40 为 `max_steps`、pushes=30、decisions=14、goals=1；A41 为 `max_steps`、pushes=30、decisions=14、goals=1。
A40/A41 首个决策差异为 `None`；A41 commitment 触发 `0` 次，额外执行 `0` 个承诺 push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | option done | commitment triggers | commitment extra pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A40 | max_steps | 14 | 30 | 1 | 48 | 0 | 0 | NA | NA |
| A41 | max_steps | 14 | 30 | 1 | 48 | 0 | 0 | 0 | 0 |

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

## 方法含义

A41 不新增 teacher，也不把参考轨迹作为在线锚定；它检验的是模型自身在 A40 中已经发现的 option 是否必须跨闭环决策保持。本轮 30-push 结果没有进入可触发 option commitment 的阶段，主要用于确认早期前缀未退化；机制效果以 80-push 复核为准。
