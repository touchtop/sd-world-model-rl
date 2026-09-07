# XSokoban first 第 2 关 Room-GFT A45 active-blocker commitment 诊断

## 核心结论

A45 继承 A44，对应专家 27 点中的 active obligation、agenda option tree 与 agenda state persistence。A44 负责识别具体 blocker obligation 的高 tier 后缀，A45 只在后缀经 exact transition 重放验证后，把该后缀作为 bounded commitment 执行。
A44 为 `max_steps`、pushes=80、decisions=51、goals=2；A45 为 `max_steps`、pushes=80、decisions=47、goals=2。
A44/A45 首个决策差异为 `32`；A45 commitment 触发 `2` 次，额外执行 `4` 个承诺 push。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | abo true progress | abc triggers | abc extra pushes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A44 | max_steps | 51 | 80 | 2 | 35 | 2 | 1 | 0 | 5 | NA | NA |
| A45 | max_steps | 47 | 80 | 2 | 36 | 2 | 1 | 0 | 2 | 2 | 4 |

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
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 16 | b8:left | 1 | 1 | fill | 47 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 17 | b8:left | 1 | 1 | fill | 46 | 2 | 2 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 18 | b8:left | 1 | 1 | fill | 45 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 19 | b2:left | 1 | 1 | fill | 44 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 20 | b8:left | 2 | 1 | fill | 44 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 21 | b8:up | 6 | 1 | fill | 40 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 22 | b3:down | 2 | 1 | fill | 40 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 23 | b3:up | 2 | 1 | fill | 40 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 24 | b6:left | 1 | 2 | fill | 49 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | 2 | 2 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 26 | b4:up | 1 | 2 | release | 51 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 27 | b0:down | 2 | 2 | release | 49 | 1 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 28 | b6:up | 1 | 2 | release | 29 | 2 | 2 | 2 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 29 | b6:up | 3 | 2 | release | 41 | 0 | 0 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 30 | b4:down | 3 | 2 | fill | 35 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 31 | b3:left | 4 | 2 | release | 35 | 2 | 1 | 0 | parent_active_blocker_obligation_already_best_tier | 8 | True | validated_active_blocker_option_committed | 3 | 8 | selected_kind_reduced,stand_resolved |
| 32 | b7:left | 1 | 2 | release | 35 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 33 | b8:left | 1 | 2 | release | 35 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 34 | b8:left | 1 | 2 | release | 35 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 35 | b8:left | 1 | 2 | release | 34 | 2 | 1 | 0 | parent_not_active_blocker_trigger | None | False | no_selected_active_blocker_option | 0 | None |  |
| 36 | b7:down | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 37 | b9:left | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 38 | b9:up | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 39 | b7:right | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 40 | b7:up | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 41 | b7:up | 2 | 2 | release | 34 | 2 | 1 | 0 | parent_active_blocker_obligation_already_best_tier | 8 | True | validated_active_blocker_option_committed | 1 | 8 | selected_kind_reduced,stand_resolved |
| 42 | b5:up | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 43 | b4:left | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 44 | b9:right | 1 | 2 | release | 34 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 45 | b8:left | 1 | 2 | release | 35 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |
| 46 | b8:left | 1 | 2 | release | 36 | 2 | 1 | 0 | no_high_tier_active_blocker_obligation_option | 2 | False | selected_active_blocker_option_below_min_tier | 0 | None |  |

## 方法含义

A45 检验的是 A44 暴露出的消费端缺口：具体 blocker-specific 后缀已经可见，但闭环若每步只执行第一个 root edge，可能无法保持同一义务的连续执行。该版本仍不在线调用 teacher，不引入参考轨迹锚定，不改 exact transition，也不改 Boxoban 主线。
