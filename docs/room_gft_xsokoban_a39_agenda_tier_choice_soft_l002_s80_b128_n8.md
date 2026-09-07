# XSokoban first 第 2 关 Room-GFT A39 agenda tier choice 诊断

## 核心结论

A39 继承 A38，复用 A35 multi-head checkpoint。新增部分对应专家 27 点中的第 16 点：把已有 agenda_tier/off-agenda/release-loop 证书从软分数转为root choice 的 lexicographic tier 比较。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2；A39 为 `max_steps`、pushes=80、decisions=53、goals=2。
A31/A39 首个决策差异为 `26`；A39 tier override 次数为 `20`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | tier loaded | tier override | higher-tier selected | off agenda | release loop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | NA | NA | NA | NA | NA |
| A39 | max_steps | 53 | 80 | 2 | 29 | 2 | 53 | 20 | 20 | 26 | 0 |

## A39 轨迹前缀

| step | action | pushes | goals | stage | slot cost | release actions | parent tier | selected tier | max tier | override | agenda | progress | off | loop |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 0 | b6:right | 1 | 0 | fill | 50 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 1 | b2:up | 1 | 0 | fill | 51 | 0 | 6 | 6 | 6 | False | release_blocker | True | False | False |
| 2 | b6:right | 2 | 0 | fill | 49 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 3 | b4:up | 2 | 0 | fill | 47 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 4 | b3:up | 1 | 0 | fill | 48 | 0 | 1 | 1 | 1 | False | release_blocker | False | True | False |
| 5 | b4:right | 1 | 0 | fill | 48 | 0 | 6 | 6 | 6 | False | release_blocker | True | False | False |
| 6 | b8:right | 1 | 0 | fill | 48 | 0 | 2 | 2 | 5 | False | release_blocker | False | True | False |
| 7 | b7:left | 3 | 0 | fill | 47 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 8 | b7:up | 6 | 0 | fill | 44 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 9 | b5:left | 5 | 0 | fill | 41 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 10 | b3:left | 2 | 1 | fill | 49 | 0 | 7 | 7 | 7 | False | release_blocker | True | False | False |
| 11 | b6:down | 1 | 1 | fill | 48 | 0 | 5 | 5 | 6 | False | release_blocker | True | False | False |
| 12 | b3:down | 2 | 1 | fill | 48 | 0 | 6 | 6 | 6 | False | release_blocker | True | False | False |
| 13 | b3:up | 2 | 1 | fill | 48 | 0 | 6 | 6 | 6 | False | release_blocker | True | False | False |
| 14 | b6:right | 1 | 1 | fill | 48 | 0 | 2 | 2 | 2 | False | release_blocker | False | True | False |
| 15 | b8:up | 1 | 1 | fill | 48 | 0 | 2 | 2 | 2 | False | release_blocker | False | True | False |
| 16 | b8:left | 1 | 1 | fill | 47 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 17 | b8:left | 1 | 1 | fill | 46 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 18 | b8:left | 1 | 1 | fill | 45 | 0 | 6 | 6 | 6 | False | release_blocker | True | False | False |
| 19 | b2:left | 1 | 1 | fill | 44 | 0 | 5 | 5 | 6 | False | release_blocker | True | False | False |
| 20 | b8:left | 2 | 1 | fill | 44 | 0 | 6 | 6 | 6 | False | release_blocker | True | False | False |
| 21 | b8:up | 6 | 1 | fill | 40 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 22 | b3:down | 2 | 1 | fill | 40 | 0 | 6 | 6 | 7 | False | release_blocker | True | False | False |
| 23 | b3:up | 2 | 1 | fill | 40 | 0 | 6 | 6 | 7 | False | release_blocker | True | False | False |
| 24 | b6:left | 1 | 2 | fill | 49 | 0 | 7 | 7 | 7 | False | release_blocker | True | False | False |
| 25 | b6:up | 1 | 2 | release | 29 | 2 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 26 | b4:up | 1 | 2 | release | 51 | 0 | 1 | 7 | 7 | True | consume_release | True | False | False |
| 27 | b0:down | 2 | 2 | release | 49 | 0 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 28 | b6:up | 1 | 2 | release | 29 | 2 | 5 | 5 | 5 | False | release_blocker | True | False | False |
| 29 | b6:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 30 | b3:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 31 | b3:left | 1 | 2 | release | 29 | 2 | 2 | 2 | 2 | False | consume_release | False | True | False |
| 32 | b7:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 33 | b8:left | 1 | 2 | release | 29 | 2 | 2 | 2 | 7 | False | consume_release | False | True | False |
| 34 | b4:down | 2 | 2 | release | 29 | 2 | 7 | 7 | 7 | False | consume_release | True | False | False |
| 35 | b4:up | 2 | 2 | release | 29 | 2 | 7 | 7 | 7 | False | consume_release | True | False | False |
| 36 | b7:right | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 37 | b8:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 38 | b9:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 39 | b7:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 40 | b7:up | 1 | 2 | release | 29 | 2 | 2 | 2 | 2 | False | consume_release | False | True | False |
| 41 | b7:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 42 | b4:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 43 | b8:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 44 | b8:right | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 45 | b9:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 46 | b8:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 47 | b9:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 48 | b8:right | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 49 | b8:right | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 50 | b9:down | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 51 | b9:left | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |
| 52 | b8:up | 1 | 2 | release | 29 | 2 | 1 | 2 | 2 | True | consume_release | False | True | False |

## 方法含义

A39 只检验 lexicographic tier / hard downgrade，不新增动作生成或 teacher 信号。如果它改变轨迹但仍未求解，说明 active agenda 的分层消费方向有效，后续仍应在专家 27 点内围绕 option tree 或更精细的 tier 定义做单独决策。
