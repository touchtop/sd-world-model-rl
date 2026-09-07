# XSokoban first 第 2 关 Room-GFT A47 target-prerequisite 诊断

## 核心结论

A47 继承 A46，对应专家 27 点中的 obligation frontier、blocker set、active agenda state 与 agenda-option tree。A46 若在 target-only residual agenda 中找不到 target progress，A47 会进一步检查 target blocker 自身是否因 push stand cell 被箱子占用而不可执行，并形成 `ClearTargetPushStand` 前提义务。
A46 为 `max_steps`、pushes=30、decisions=14、goals=1；A47 为 `max_steps`、pushes=30、decisions=14、goals=1。
A46/A47 首个决策差异为 `None`；A47 构造 prerequisite obligation `0` 次，触发 `0` 次，追加 `0` 个 prerequisite push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | residual triggers | prereq obligations | prereq triggers | prereq pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A46 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | NA | NA | NA |
| A47 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 | 0 |

## A47 轨迹前缀

| step | action | pushes | goals | slot cost | dep | target | residual reason | prereq reason | prereq pushes | prereq best tier |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| 0 | b6:right | 1 | 0 | 50 | 2 | 0 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 1 | b2:up | 1 | 0 | 51 | 1 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 2 | b6:right | 2 | 0 | 49 | 2 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 3 | b4:up | 2 | 0 | 47 | 2 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 4 | b3:up | 1 | 0 | 48 | 2 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 5 | b4:right | 1 | 0 | 48 | 1 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 6 | b8:right | 1 | 0 | 48 | 1 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 7 | b7:left | 3 | 0 | 47 | 1 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 8 | b7:up | 6 | 0 | 44 | 1 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 9 | b5:left | 5 | 0 | 41 | 1 | 1 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 10 | b3:left | 2 | 1 | 49 | 2 | 2 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 11 | b6:down | 1 | 1 | 48 | 2 | 2 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 12 | b3:down | 2 | 1 | 48 | 2 | 2 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |
| 13 | b3:up | 2 | 1 | 48 | 2 | 2 | source_commitment_not_triggered | source_not_target_residual_failure | 0 | None |

## 方法含义

A47 检验的是 residual target blocker 的下一层可执行前提。该版本不在线调用 teacher，不引入参考轨迹锚定，不改变 exact transition，也不改 Boxoban 主线。
