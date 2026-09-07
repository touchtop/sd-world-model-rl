# XSokoban first 第 2 关 Room-GFT A39 agenda tier choice 诊断

## 核心结论

A39 继承 A38，复用 A35 multi-head checkpoint。新增部分对应专家 27 点中的第 16 点：把已有 agenda_tier/off-agenda/release-loop 证书从软分数转为root choice 的 lexicographic tier 比较。
A31 为 `max_steps`、pushes=80、decisions=42、goals=2；A39 为 `max_steps`、pushes=30、decisions=14、goals=1。
A31/A39 首个决策差异为 `14`；A39 tier override 次数为 `0`。

## 摘要对照

| planner | status | decisions | pushes | goals | slot cost | release actions | tier loaded | tier override | higher-tier selected | off agenda | release loop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A31 | max_steps | 42 | 80 | 2 | 29 | 2 | NA | NA | NA | NA | NA |
| A39 | max_steps | 14 | 30 | 1 | 48 | 0 | 14 | 0 | 0 | 2 | 0 |

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

## 方法含义

A39 只检验 lexicographic tier / hard downgrade，不新增动作生成或 teacher 信号。如果它改变轨迹但仍未求解，说明 active agenda 的分层消费方向有效，后续仍应在专家 27 点内围绕 option tree 或更精细的 tier 定义做单独决策。
