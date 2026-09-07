# XSokoban first 第 2 关 Room-GFT A15 dependency-bound macro 诊断

## 核心结论

A15 继承 A14 的 bounded recovery macro，但短宏搜索必须绑定到一个具体的 `stand_cell_unreachable_by_player` 或 `target_occupied_by_box` 依赖行。
A14 对照为 `max_steps`，执行 `200` 个 push 后 `boxes_on_goals=0`；A15 smoke 为 `no_action`，执行 `36` 个 push、`21` 个闭环决策后 `boxes_on_goals=0`。
A14/A15 首个决策差异为 `1`；A15 绑定依赖宏触发 `3` 次，追加 `9` 个 push，最终 dependency count 为 `2`。

## A14/A15 摘要

| planner | status | decisions | pushes | goals | slot cost | dependencies | target blockers | bound macro triggers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A14 | max_steps | 73 | 200 | 0 | 33 | 3 | 1 | NA |
| A15 | no_action | 21 | 36 | 0 | 42 | 2 | 1 | 3 |

## A15 轨迹前缀

| step | action | pushes | goals | slot cost | dependencies | bound kind | bound macro | macro extra | best score | progress |
|---:|---|---:|---:|---:|---:|---|---|---:|---:|---|
| 0 | b6:right | 4 | 0 | 49 | 1 | stand_cell_unreachable_by_player | True | 3 | 150.9 | row |
| 1 | b4:up | 2 | 0 | 47 | 2 | NA | False | 0 | NA | NA |
| 2 | b3:up | 1 | 0 | 48 | 2 | NA | False | 0 | NA | NA |
| 3 | b4:right | 5 | 0 | 47 | 2 | stand_cell_unreachable_by_player,target_occupied_by_box | True | 4 | 120.15 | stand |
| 4 | b7:left | 3 | 0 | 47 | 1 | stand_cell_unreachable_by_player | True | 2 | 235.85 | row,stand |
| 5 | b8:left | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 6 | b5:down | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 7 | b2:left | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 8 | b4:up | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 9 | b3:left | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 10 | b3:left | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 11 | b5:up | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 12 | b5:right | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 13 | b5:up | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 14 | b5:up | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 15 | b9:up | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 16 | b6:left | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 17 | b5:up | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 18 | b4:left | 1 | 0 | 46 | 1 | NA | False | 0 | NA | NA |
| 19 | b7:left | 6 | 0 | 42 | 2 | NA | False | 0 | NA | NA |
| 20 | b8:right | 1 | 0 | 42 | 2 | NA | False | 0 | NA | NA |
| 21 | no_action | 0 | NA | NA | 2 | NA | NA | 0 | NA | NA |

## 方法含义

A15 检验的是：A14 已可执行短宏后，若宏目标从一般恢复指标改为具体依赖行，是否能更稳定地修复不可达站位或移开目标格阻塞箱。
