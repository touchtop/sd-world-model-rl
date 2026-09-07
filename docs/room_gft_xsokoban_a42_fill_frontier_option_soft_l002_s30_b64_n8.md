# XSokoban first 第 2 关 Room-GFT A42 fill-frontier option 诊断

## 核心结论

A42 继承 A41，对应专家 27 点中关于 frontier obligation、agenda option 和 next agenda milestone 的建议。它只在 clean fill 阶段把“完成当前 frontier slot”的候选作为 option 消费。
A41 为 `max_steps`、pushes=30、decisions=14、goals=1；A42 为 `max_steps`、pushes=30、decisions=14、goals=1。
A41/A42 首个决策差异为 `None`；A42 fill-frontier override `0` 次，completion `0` 次。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | fill overrides | fill completion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A41 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | NA | NA |
| A42 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 |

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

## 方法含义

A42 检验的是 release 完成后是否需要把 fill frontier completion 作为独立 agenda boundary。它允许该 boundary 之后出现新的 release frontier，但不允许把无 slot 进展的动作伪装成 fill 进展。
