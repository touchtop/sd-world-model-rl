# XSokoban first 第 2 关 Room-GFT A45 active-blocker commitment 诊断

## 核心结论

A45 继承 A44，对应专家 27 点中的 active obligation、agenda option tree 与 agenda state persistence。A44 负责识别具体 blocker obligation 的高 tier 后缀，A45 只在后缀经 exact transition 重放验证后，把该后缀作为 bounded commitment 执行。
A44 为 `max_steps`、pushes=30、decisions=14、goals=1；A45 为 `max_steps`、pushes=30、decisions=14、goals=1。
A44/A45 首个决策差异为 `None`；A45 commitment 触发 `0` 次，额外执行 `0` 个承诺 push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | abo true progress | abc triggers | abc extra pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A44 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | NA | NA |
| A45 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 | 0 |

## A45 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dep | target | release | abo reason | abo tier | abc | abc reason | abc extra | final tier | progress kinds |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |

## 方法含义

A45 检验的是 A44 暴露出的消费端缺口：具体 blocker-specific 后缀已经可见，但闭环若每步只执行第一个 root edge，可能无法保持同一义务的连续执行。该版本仍不在线调用 teacher，不引入参考轨迹锚定，不改 exact transition，也不改 Boxoban 主线。
