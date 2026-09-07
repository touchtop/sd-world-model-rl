# XSokoban first 第 2 关 Room-GFT A40 agenda-option tree 诊断

## 核心结论

A40 继承 A39，对应专家 27 点中的第 14 点：在 root 处将单步候选提升为 bounded agenda-option 评价。后续 suffix 只用于评价，不作为宏动作执行。
A39 为 `max_steps`、pushes=80、decisions=53、goals=2；A40 为 `max_steps`、pushes=80、decisions=53、goals=2。
A39/A40 首个决策差异为 `None`；A40 option override 次数为 `0`，selected progress 次数为 `22`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | tier override | option loaded | option override | option progress | option done | option loop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A39 | max_steps | 53 | 80 | 2 | 29 | 2 | 20 | NA | NA | NA | NA | NA |
| A40 | max_steps | 53 | 80 | 2 | 29 | 2 | 20 | 53 | 0 | 22 | 22 | 0 |

## A40 轨迹前缀

| step | action | pushes | goals | stage | slot cost | root tier | option tier | override | option reason | option depth | option progress | option done | loop |
|---:|---|---:|---:|---|---:|---:|---:|---|---|---:|---|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 1 | b2:up | 1 | 0 | fill | 51 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 2 | b6:right | 2 | 0 | fill | 49 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 3 | b4:up | 2 | 0 | fill | 47 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 4 | b3:up | 1 | 0 | fill | 48 | 1 | None | False | parent_not_option_trigger | None | None | None | None |
| 5 | b4:right | 1 | 0 | fill | 48 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 6 | b8:right | 1 | 0 | fill | 48 | 2 | None | False | parent_not_option_trigger | None | None | None | None |
| 7 | b7:left | 3 | 0 | fill | 47 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 8 | b7:up | 6 | 0 | fill | 44 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 9 | b5:left | 5 | 0 | fill | 41 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 10 | b3:left | 2 | 1 | fill | 49 | 7 | None | False | parent_not_option_trigger | None | None | None | None |
| 11 | b6:down | 1 | 1 | fill | 48 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 12 | b3:down | 2 | 1 | fill | 48 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 13 | b3:up | 2 | 1 | fill | 48 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | None | False | parent_not_option_trigger | None | None | None | None |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | None | False | parent_not_option_trigger | None | None | None | None |
| 16 | b8:left | 1 | 1 | fill | 47 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 17 | b8:left | 1 | 1 | fill | 46 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 18 | b8:left | 1 | 1 | fill | 45 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 19 | b2:left | 1 | 1 | fill | 44 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 20 | b8:left | 2 | 1 | fill | 44 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 21 | b8:up | 6 | 1 | fill | 40 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 22 | b3:down | 2 | 1 | fill | 40 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 23 | b3:up | 2 | 1 | fill | 40 | 6 | None | False | parent_not_option_trigger | None | None | None | None |
| 24 | b6:left | 1 | 2 | fill | 49 | 7 | None | False | parent_not_option_trigger | None | None | None | None |
| 25 | b6:up | 1 | 2 | release | 29 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 26 | b4:up | 1 | 2 | release | 51 | 7 | None | False | parent_not_option_trigger | None | None | None | None |
| 27 | b0:down | 2 | 2 | release | 49 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 28 | b6:up | 1 | 2 | release | 29 | 5 | None | False | parent_not_option_trigger | None | None | None | None |
| 29 | b6:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 30 | b3:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 31 | b3:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 32 | b7:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 33 | b8:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 34 | b4:down | 2 | 2 | release | 29 | 7 | None | False | parent_not_option_trigger | None | None | None | None |
| 35 | b4:up | 2 | 2 | release | 29 | 7 | None | False | parent_not_option_trigger | None | None | None | None |
| 36 | b7:right | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 37 | b8:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 38 | b9:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 39 | b7:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 40 | b7:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 41 | b7:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 42 | b4:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 43 | b8:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 4 | True | True | False |
| 44 | b8:right | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 45 | b9:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 46 | b8:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 4 | True | True | False |
| 47 | b9:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 4 | True | True | False |
| 48 | b8:right | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 49 | b8:right | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 4 | True | True | False |
| 50 | b9:down | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 3 | True | True | False |
| 51 | b9:left | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 4 | True | True | False |
| 52 | b8:up | 1 | 2 | release | 29 | 2 | 7 | False | parent_option_already_best_tier | 4 | True | True | False |

## 方法含义

A40 检验的是：A39 已能按 tier 改写 root 选择后，短程 option 评价是否能进一步稳定识别可完成当前 consume-release 议程的候选。若仍不能求解，缺口更可能在显式 agenda state 或 learned agenda distance，而非单步分数。
