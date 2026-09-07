# XSokoban first 第 2 关 Room-GFT A46 residual active-agenda 诊断

## 核心结论

A46 继承 A45，对应专家 27 点中的 active agenda state、distance to next agenda milestone 与 agenda-option tree。A45 承诺若只完成 `stand_resolved` 而留下同一 relation 的 `target_occupied_by_box`，A46 才在该宏边界后追加一次有界 residual option 搜索；若未找到 target 进展，则不改变 A45 动作，只记录负证据。
A45 为 `max_steps`、pushes=30、decisions=14、goals=1；A46 为 `max_steps`、pushes=30、decisions=14、goals=1。
A45/A46 首个决策差异为 `None`；A46 residual agenda 触发 `0` 次，追加 `0` 个 residual push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | abc triggers | residual triggers | residual pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A45 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | NA | NA |
| A46 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 |

## A46 轨迹前缀

| step | action | pushes | goals | slot cost | dep | target | abc | residual reason | residual pushes | residual target |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | b6:right | 1 | 0 | 50 | 2 | 0 | False | source_commitment_not_triggered | 0 | None |
| 1 | b2:up | 1 | 0 | 51 | 1 | 1 | False | source_commitment_not_triggered | 0 | None |
| 2 | b6:right | 2 | 0 | 49 | 2 | 1 | False | source_commitment_not_triggered | 0 | None |
| 3 | b4:up | 2 | 0 | 47 | 2 | 1 | False | source_commitment_not_triggered | 0 | None |
| 4 | b3:up | 1 | 0 | 48 | 2 | 1 | False | source_commitment_not_triggered | 0 | None |
| 5 | b4:right | 1 | 0 | 48 | 1 | 1 | False | source_commitment_not_triggered | 0 | None |
| 6 | b8:right | 1 | 0 | 48 | 1 | 1 | False | source_commitment_not_triggered | 0 | None |
| 7 | b7:left | 3 | 0 | 47 | 1 | 1 | False | source_commitment_not_triggered | 0 | None |
| 8 | b7:up | 6 | 0 | 44 | 1 | 1 | False | source_commitment_not_triggered | 0 | None |
| 9 | b5:left | 5 | 0 | 41 | 1 | 1 | False | source_commitment_not_triggered | 0 | None |
| 10 | b3:left | 2 | 1 | 49 | 2 | 2 | False | source_commitment_not_triggered | 0 | None |
| 11 | b6:down | 1 | 1 | 48 | 2 | 2 | False | source_commitment_not_triggered | 0 | None |
| 12 | b3:down | 2 | 1 | 48 | 2 | 2 | False | source_commitment_not_triggered | 0 | None |
| 13 | b3:up | 2 | 1 | 48 | 2 | 2 | False | source_commitment_not_triggered | 0 | None |

## 方法含义

A46 检验的是 A45 后的残余义务是否能在同一宏边界内继续推进。该版本不在线调用 teacher，不引入参考轨迹锚定，不改变 exact transition，也不改 Boxoban 主线。
