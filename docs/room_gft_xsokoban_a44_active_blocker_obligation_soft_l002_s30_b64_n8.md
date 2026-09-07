# XSokoban first 第 2 关 Room-GFT A44 active-blocker obligation 诊断

## 核心结论

A44 继承 A43，对应专家 27 点中的 active obligation、blocker-specific relation、agenda option tree 和 agenda state persistence。它不再只用 aggregate blocker count，而是把 A11 schedule dependency row 暴露为具体 active blocker obligation。
A43 为 `max_steps`、pushes=30、decisions=14、goals=1；A44 为 `max_steps`、pushes=30、decisions=14、goals=1。
A43/A44 首个决策差异为 `None`；A44 active-blocker override `0` 次，top true-progress `0` 次。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | dependency | target blocker | release actions | abo overrides | abo true progress |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A43 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | NA | NA |
| A44 | max_steps | 14 | 30 | 1 | 48 | 2 | 2 | 0 | 0 | 0 |

## A44 轨迹前缀

| step | action | pushes | goals | stage | slot cost | dep | target | release | abo override | reason | option tier | true progress | progress kinds |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 2 | 0 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 1 | b2:up | 1 | 0 | fill | 51 | 1 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 2 | b6:right | 2 | 0 | fill | 49 | 2 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 3 | b4:up | 2 | 0 | fill | 47 | 2 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 4 | b3:up | 1 | 0 | fill | 48 | 2 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 5 | b4:right | 1 | 0 | fill | 48 | 1 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 6 | b8:right | 1 | 0 | fill | 48 | 1 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 7 | b7:left | 3 | 0 | fill | 47 | 1 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 8 | b7:up | 6 | 0 | fill | 44 | 1 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 9 | b5:left | 5 | 0 | fill | 41 | 1 | 1 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 10 | b3:left | 2 | 1 | fill | 49 | 2 | 2 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 11 | b6:down | 1 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 12 | b3:down | 2 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_active_blocker_trigger | None | None |  |
| 13 | b3:up | 2 | 1 | fill | 48 | 2 | 2 | 0 | False | parent_not_active_blocker_trigger | None | None |  |

## 方法含义

A44 检验的是 A43 负诊断后的下一层原因：当前失败是否来自 aggregate blocker count 隐去了具体 active blocker identity。该版本仍不在线调用 teacher，不改 exact transition，也不改 Boxoban 主线。
