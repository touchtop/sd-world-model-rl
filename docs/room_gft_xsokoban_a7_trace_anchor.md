# XSokoban first 第 1 关 Room-GFT A7 trace-anchor 诊断

## 核心结论

A7 在 A6 forced-corridor macro 基础上加入 teacher/reference trace anchor。该 anchor 只在当前状态命中参考轨迹时生效，对参考物理动作和该箱子未来目标槽位加 prior，并在根节点优先选择参考物理动作。
本地 exact-state teacher 在 `7248` 个展开、`71.821` 秒内解出该关，参考解长度为 `97` 个 push。
A6 smoke 为 `no_action`，执行 `22` 个 push 后 `boxes_on_goals=1`；A7 trace-anchor smoke 为 `solved`，执行 `97` 个 push、`63` 个闭环决策后 `boxes_on_goals=6`。
该结果用于机制定位：若参考调度被显式锚定，当前 exact transition、room-slot commitment 和 corridor macro 可以沿可行轨迹推进；A5/A6 的失败主要来自多箱调度/槽位顺序选择，而不是单步转移或步数上限本身。A7 不是无 teacher 的通用求解率结果。

## A6/A7 摘要

| planner | status | decision steps | executed pushes | boxes_on_goals | final slot cost | macro triggers | reference hits | reference matches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | no_action | 14 | 22 | 1 | 46 | 1 | 0 | 0 |
| A7 | solved | 63 | 97 | 6 | 0 | 6 | 63 | 63 |

## A7 轨迹前缀

| step | action | executed pushes | macro | reference step | reference match | goals | slot cost |
|---:|---|---:|---|---:|---|---:|---:|
| 0 | b3:left | 1 | False | 0 | True | 0 | 44 |
| 1 | b1:up | 1 | False | 1 | True | 0 | 44 |
| 2 | b2:down | 1 | False | 2 | True | 0 | 43 |
| 3 | b5:right | 1 | False | 3 | True | 0 | 42 |
| 4 | b1:down | 1 | False | 4 | True | 0 | 42 |
| 5 | b5:right | 1 | False | 5 | True | 0 | 41 |
| 6 | b5:right | 1 | False | 6 | True | 0 | 40 |
| 7 | b5:right | 7 | True | 7 | True | 0 | 35 |
| 8 | b5:right | 1 | False | 14 | True | 1 | 44 |
| 9 | b3:down | 1 | False | 15 | True | 1 | 43 |
| 10 | b3:down | 1 | False | 16 | True | 1 | 42 |
| 11 | b4:right | 1 | False | 17 | True | 1 | 41 |
| 12 | b2:left | 1 | False | 18 | True | 1 | 41 |
| 13 | b4:right | 1 | False | 19 | True | 1 | 40 |
| 14 | b4:right | 1 | False | 20 | True | 1 | 39 |
| 15 | b4:right | 7 | True | 21 | True | 1 | 34 |
| 16 | b5:up | 1 | False | 28 | True | 1 | 21 |
| 17 | b5:right | 1 | False | 29 | True | 2 | 30 |
| 18 | b2:down | 1 | False | 30 | True | 2 | 29 |
| 19 | b0:down | 1 | False | 31 | True | 2 | 29 |
| 20 | b2:down | 1 | False | 32 | True | 2 | 28 |
| 21 | b2:down | 1 | False | 33 | True | 2 | 27 |
| 22 | b4:right | 1 | False | 34 | True | 2 | 26 |
| 23 | b0:down | 1 | False | 35 | True | 2 | 26 |
| 24 | b4:right | 1 | False | 36 | True | 2 | 25 |
| 25 | b4:right | 1 | False | 37 | True | 2 | 24 |
| 26 | b4:right | 7 | True | 38 | True | 2 | 19 |
| 27 | b2:right | 1 | False | 45 | True | 2 | 33 |
| 28 | b5:down | 1 | False | 46 | True | 2 | 20 |
| 29 | b4:right | 1 | False | 47 | True | 3 | 17 |
| 30 | b1:down | 1 | False | 48 | True | 3 | 16 |
| 31 | b0:down | 1 | False | 49 | True | 3 | 16 |
| 32 | b0:left | 1 | False | 50 | True | 3 | 16 |
| 33 | b1:down | 1 | False | 51 | True | 3 | 15 |
| 34 | b1:down | 1 | False | 52 | True | 3 | 14 |
| 35 | b3:right | 1 | False | 53 | True | 3 | 13 |
| 36 | b0:left | 1 | False | 54 | True | 3 | 13 |
| 37 | b3:right | 1 | False | 55 | True | 3 | 12 |
| 38 | b3:right | 1 | False | 56 | True | 3 | 11 |
| 39 | b3:right | 9 | True | 57 | True | 4 | 2 |

## 方法含义

A7 把 teacher 调用限定为一次离线参考轨迹生成，teacher 不参与 Boxoban 主线、A6 基线或在线泛化评测。因此，A7 的意义是定位失败机制和验证修复方向：后续若要形成无参考解版本，需要把 trace anchor 中暴露出的多箱 slot schedule 转化为可自主生成的 room fill plan、释放动作约束或价值审计目标。
