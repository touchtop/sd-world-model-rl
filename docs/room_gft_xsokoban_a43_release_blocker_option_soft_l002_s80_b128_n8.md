# XSokoban first 第 2 关 Room-GFT A43 release-blocker option 诊断

## 核心结论

A43 继承 A42，对应专家 27 点中的 agenda-option tree、frontier obligation 后续 release-blocker option 和 agenda state persistence。它只在 post-frontier release-blocker 低 tier/off-agenda 状态中评价短程 option，并只选择第一步。
A42 为 `max_steps`、pushes=80、decisions=51、goals=2；A43 为 `max_steps`、pushes=80、decisions=51、goals=2。
A42/A43 首个决策差异为 `None`；A43 release-blocker override `0` 次。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | rb overrides | rb progress |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A42 | max_steps | 51 | 80 | 2 | 35 | 2 | 1 | 0 | NA | NA |
| A43 | max_steps | 51 | 80 | 2 | 35 | 2 | 1 | 0 | 0 | 0 |

## A43 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dep | target | release | rb override | reason | option tier | option progress |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 14 | b6:right | 1 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 15 | b8:up | 1 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 16 | b8:left | 1 | 1 | fill | 47 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 17 | b8:left | 1 | 1 | fill | 46 | 2 | 2 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 18 | b8:left | 1 | 1 | fill | 45 | 2 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 19 | b2:left | 1 | 1 | fill | 44 | 2 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 20 | b8:left | 2 | 1 | fill | 44 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 21 | b8:up | 6 | 1 | fill | 40 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 22 | b3:down | 2 | 1 | fill | 40 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 23 | b3:up | 2 | 1 | fill | 40 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 24 | b6:left | 1 | 2 | fill | 49 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | 2 | 2 | False | parent_not_release_blocker_trigger | None | None |
| 26 | b4:up | 1 | 2 | release | 51 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 27 | b0:down | 2 | 2 | release | 49 | 1 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 28 | b6:up | 1 | 2 | release | 29 | 2 | 2 | 2 | False | parent_not_release_blocker_trigger | None | None |
| 29 | b6:up | 3 | 2 | release | 41 | 0 | 0 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 30 | b4:down | 3 | 2 | fill | 35 | 2 | 1 | 0 | False | parent_not_release_blocker_trigger | None | None |
| 31 | b3:left | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 32 | b3:left | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 33 | b9:left | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 34 | b7:down | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 35 | b7:left | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 36 | b8:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 37 | b7:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 38 | b7:right | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 39 | b7:right | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 40 | b8:left | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 41 | b7:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 42 | b6:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 43 | b5:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 44 | b4:left | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 45 | b9:right | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 2 | False |
| 46 | b9:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 47 | b9:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 2 | False |
| 48 | b8:up | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |
| 49 | b9:down | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 2 | False |
| 50 | b9:right | 1 | 2 | release | 35 | 2 | 1 | 0 | False | no_high_tier_release_blocker_option | 1 | False |

## 方法含义

A43 检验的是 A42 完成 fill frontier 后，release-blocker 义务是否需要短程 option 评价。该版本不执行 suffix，因此若仍失败，缺口应继续定位为 option commitment、agenda state persistence 或更高层里程碑生成，而不是 teacher 在线锚定。
