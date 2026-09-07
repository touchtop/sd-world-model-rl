# XSokoban first 第 2 关 Room-GFT A48 assignment-conflict 诊断

## 核心结论

A48 继承 A47，对应专家 27 点中的 obligation frontier、blocker set、active agenda state 与 agenda-option tree。A47 若发现 target blocker 所需站位被 parent assigned box 占用，A48 将其记录为 assignment/prerequisite 冲突义务，而不是坐标级规则。
A47 为 `max_steps`、pushes=30、decisions=14、goals=1；A48 为 `max_steps`、pushes=30、decisions=14、goals=1。
A47/A48 首个决策差异为 `None`；A48 构造 assignment-conflict obligation `0` 次，触发 `0` 次，追加 `0` 个 conflict push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | prereq obligations | conflict obligations | conflict triggers | conflict pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A47 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | NA | NA | NA |
| A48 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 | 0 |

## A48 轨迹前缀

| step | action | pushes | goals | slot cost | dep | target | prereq reason | conflict reason | conflict pushes | conflict best tier |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| 0 | b6:right | 1 | 0 | 50 | 2 | 0 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 1 | b2:up | 1 | 0 | 51 | 1 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 2 | b6:right | 2 | 0 | 49 | 2 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 3 | b4:up | 2 | 0 | 47 | 2 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 4 | b3:up | 1 | 0 | 48 | 2 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 5 | b4:right | 1 | 0 | 48 | 1 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 6 | b8:right | 1 | 0 | 48 | 1 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 7 | b7:left | 3 | 0 | 47 | 1 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 8 | b7:up | 6 | 0 | 44 | 1 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 9 | b5:left | 5 | 0 | 41 | 1 | 1 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 10 | b3:left | 2 | 1 | 49 | 2 | 2 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 11 | b6:down | 1 | 1 | 48 | 2 | 2 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 12 | b3:down | 2 | 1 | 48 | 2 | 2 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |
| 13 | b3:up | 2 | 1 | 48 | 2 | 2 | source_not_target_residual_failure | source_not_target_prerequisite_failure | 0 | None |

## 方法含义

A48 检验的是 target prerequisite 与当前 slot assignment 之间的冲突。该版本不在线调用 teacher，不引入参考轨迹锚定，不改变 exact transition，也不改 Boxoban 主线。
