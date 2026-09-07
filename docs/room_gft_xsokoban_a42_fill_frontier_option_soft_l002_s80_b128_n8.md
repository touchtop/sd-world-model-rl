# XSokoban first 第 2 关 Room-GFT A42 fill-frontier option 诊断

## 核心结论

A42 继承 A41，对应专家 27 点中关于 frontier obligation、agenda option 和 next agenda milestone 的建议。它只在 clean fill 阶段把“完成当前 frontier slot”的候选作为 option 消费。
A41 为 `max_steps`、pushes=80、decisions=53、goals=2；A42 为 `max_steps`、pushes=80、decisions=51、goals=2。
A41/A42 首个决策差异为 `30`；A42 fill-frontier override `1` 次，completion `1` 次。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | fill overrides | fill completion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A41 | max_steps | 53 | 80 | 2 | 41 | 0 | 0 | 0 | NA | NA |
| A42 | max_steps | 51 | 80 | 2 | 35 | 2 | 1 | 0 | 1 | 1 |

## A42 轨迹前缀

| step | action | pushes | goals | stage | slot cost | release | fill override | reason | selected tier | slot delta | valid delta |
|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|
| 0 | b6:right | 1 | 0 | fill | 50 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 1 | b2:up | 1 | 0 | fill | 51 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 2 | b6:right | 2 | 0 | fill | 49 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 3 | b4:up | 2 | 0 | fill | 47 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 4 | b3:up | 1 | 0 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 5 | b4:right | 1 | 0 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 6 | b8:right | 1 | 0 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 7 | b7:left | 3 | 0 | fill | 47 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 8 | b7:up | 6 | 0 | fill | 44 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 9 | b5:left | 5 | 0 | fill | 41 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 10 | b3:left | 2 | 1 | fill | 49 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 11 | b6:down | 1 | 1 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 12 | b3:down | 2 | 1 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 13 | b3:up | 2 | 1 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 14 | b6:right | 1 | 1 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 15 | b8:up | 1 | 1 | fill | 48 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 16 | b8:left | 1 | 1 | fill | 47 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 17 | b8:left | 1 | 1 | fill | 46 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 18 | b8:left | 1 | 1 | fill | 45 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 19 | b2:left | 1 | 1 | fill | 44 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 20 | b8:left | 2 | 1 | fill | 44 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 21 | b8:up | 6 | 1 | fill | 40 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 22 | b3:down | 2 | 1 | fill | 40 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 23 | b3:up | 2 | 1 | fill | 40 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 24 | b6:left | 1 | 2 | fill | 49 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 26 | b4:up | 1 | 2 | release | 51 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 27 | b0:down | 2 | 2 | release | 49 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 28 | b6:up | 1 | 2 | release | 29 | 2 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 29 | b6:up | 3 | 2 | release | 41 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 30 | b4:down | 3 | 2 | fill | 35 | 0 | True | fill_frontier_option_selected | 9 | 20 | 1 |
| 31 | b3:left | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 32 | b3:left | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 33 | b9:left | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 34 | b7:down | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 35 | b7:left | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 36 | b8:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 37 | b7:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 38 | b7:right | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 39 | b7:right | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 40 | b8:left | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 41 | b7:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 42 | b6:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 43 | b5:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 44 | b4:left | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 45 | b9:right | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 46 | b9:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 47 | b9:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 48 | b8:up | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 49 | b9:down | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |
| 50 | b9:right | 1 | 2 | release | 35 | 0 | False | parent_not_fill_frontier_trigger | None | NA | None |

## 方法含义

A42 检验的是 release 完成后是否需要把 fill frontier completion 作为独立 agenda boundary。它允许该 boundary 之后出现新的 release frontier，但不允许把无 slot 进展的动作伪装成 fill 进展。
