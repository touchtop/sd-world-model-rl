# XSokoban first 第 1 关 Room-GFT A4 槽位进度诊断

## 核心结论

A4 在 A3 的 `(box,direction,slot)` 语义 child 基础上加入 slot-schedule constrained progress。该 planner 为 `RoomSlotProgressGFTSDMCTSPlanner`，继承 A3 planner，不替换 Boxoban 主线。
旧失败 trace 的 `20` 个 root 上，旧物理 GFT prior 将保存动作排第一的步数为 `15`；A4 将保存物理动作排第一的步数为 `15`，首次改变发生在 step `0`。
更关键的是入房窗口：step `18` 处旧 trace 选择的入房分支被 A4 降级，A4 首选沿已承诺深槽继续穿过浅层阻塞槽。因此，A4 不是简单压制所有走廊内 `b5:right`，而是把 slot commitment 改为行一致的深槽，并区分正确的浅槽穿越与错误行的无承诺/不可达推进。
A4 短闭环 smoke 状态为 `max_steps`，执行 `40` 步，最终 `boxes_on_goals=0`。该结果仍是机制诊断，不作为 XSokoban 解题率。

## 保存 trace 重排摘要

| step | saved action | old GFT rank | A4 physical rank | A4 semantic rank | A4 best | saved commitment | slot delta | barrier delta | impacts |
|---:|---|---:|---:|---:|---|---|---:|---:|---|
| 0 | b5:up | 2 | 2 | 2 | b3:left|none | none | -1 | 0 | none |
| 1 | b5:right | 2 | 1 | 1 | b5:right|none | none | 1 | 0 | none |
| 2 | b2:right | 2 | 2 | 2 | b5:right|none | none | -1 | 0 | none |
| 3 | b0:down | 3 | 3 | 3 | b4:down|none | none | 0 | 0 | none |
| 4 | b0:down | 1 | 1 | 1 | b0:down|none | none | 1 | 0 | none |
| 5 | b1:down | 1 | 1 | 1 | b1:down|none | none | 1 | 0 | none |
| 6 | b5:right | 1 | 1 | 1 | b5:right|none | none | 1 | 0 | none |
| 7 | b5:right | 1 | 1 | 1 | b5:right|none | none | 1 | 0 | none |
| 8 | b5:right | 1 | 1 | 1 | b5:right|none | none | 1 | 0 | none |
| 9 | b5:right | 1 | 1 | 1 | b5:right|none | none | 1 | 0 | none |
| 10 | b5:right | 1 | 1 | 1 | b5:right|none | none | 1 | 0 | none |
| 11 | b5:right | 1 | 2 | 2 | b5:left|none | slot3@[7, 17] | 1 | 0 | none |
| 12 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | 1 | 0 | none |
| 13 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | 1 | 0 | none |
| 14 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | 1 | 0 | none |
| 15 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | 1 | 0 | none |
| 16 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | 1 | 0 | none |
| 17 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | -1 | -2 | CROSS_ROOM_BOUNDARY,ENTER_ROOM_ENTRY,INCREASE_UNCOMMITTED_ROOM_OR_ENTRY,UNCOMMITTED_ENTRY |
| 18 | b5:down | 3 | 4 | 4 | b5:right|slot3@[7, 17] | slot3@[7, 17] | -58 | -30 | ENTER_ROOM_ENTRY,UNCOMMITTED_ENTRY |
| 19 | b5:right | 1 | 1 | 1 | b5:right|slot3@[7, 17] | slot3@[7, 17] | -10 | -10 | BLOCK_UNFILLED_SLOT,FILL_NONFRONTIER_SLOT,INCREASE_ROOM_ORDER_DEBT |

## 入房窗口

| step | saved action | A4 best | selected slot score | top child details |
|---:|---|---|---:|---|
| 15 | b5:right | b5:right|slot3@[7, 17] | 1.8 | b5:right|slot3@[7, 17]; transport_delta=1; barrier_delta=0; pass=0 |
| 16 | b5:right | b5:right|slot3@[7, 17] | 1.8 | b5:right|slot3@[7, 17]; transport_delta=1; barrier_delta=0; pass=0 |
| 17 | b5:right | b5:right|slot3@[7, 17] | 0.2 | b5:right|slot3@[7, 17]; transport_delta=1; barrier_delta=-2; pass=0 |
| 18 | b5:down | b5:right|slot3@[7, 17] | -82.4 | b5:right|slot3@[7, 17]; transport_delta=1; barrier_delta=2; pass=1 |
| 19 | b5:right | b5:right|slot3@[7, 17] | -8 | b5:right|slot3@[7, 17]; transport_delta=0; barrier_delta=-10; pass=0 |

## A4 短闭环 smoke

| step | physical action | top commitment | boxes_on_goals | slot schedule cost |
|---:|---|---|---:|---:|
| 0 | b3:left | none | 0 | 44 |
| 1 | b5:up | none | 0 | 45 |
| 2 | b5:right | none | 0 | 44 |
| 3 | b5:right | none | 0 | 43 |
| 4 | b5:right | none | 0 | 42 |
| 5 | b5:right | none | 0 | 41 |
| 6 | b5:right | none | 0 | 40 |
| 7 | b5:right | none | 0 | 39 |
| 8 | b4:up | none | 0 | 40 |
| 9 | b5:right | slot5@[8, 17] | 0 | 39 |
| 10 | b5:right | slot5@[8, 17] | 0 | 38 |
| 11 | b1:up | none | 0 | 38 |
| 12 | b5:right | slot5@[8, 17] | 0 | 37 |
| 13 | b5:right | slot5@[8, 17] | 0 | 36 |
| 14 | b5:right | slot5@[8, 17] | 0 | 35 |
| 15 | b5:right | slot5@[8, 17] | 0 | 34 |
| 16 | b5:right | slot5@[8, 17] | 0 | 35 |
| 17 | b5:left | slot5@[8, 17] | 0 | 34 |
| 18 | b5:left | slot5@[8, 17] | 0 | 35 |
| 19 | b5:left | slot5@[8, 17] | 0 | 36 |
| 20 | b5:left | slot5@[8, 17] | 0 | 37 |
| 21 | b5:left | slot5@[8, 17] | 0 | 38 |
| 22 | b5:left | slot5@[8, 17] | 0 | 39 |
| 23 | b5:left | slot5@[8, 17] | 0 | 40 |
| 24 | b5:right | slot5@[8, 17] | 0 | 39 |
| 25 | b0:down | none | 0 | 39 |
| 26 | b5:right | slot5@[8, 17] | 0 | 38 |
| 27 | b5:right | slot5@[8, 17] | 0 | 37 |
| 28 | b5:right | slot5@[8, 17] | 0 | 36 |
| 29 | b5:right | slot5@[8, 17] | 0 | 35 |
| 30 | b5:right | slot5@[8, 17] | 0 | 34 |
| 31 | b5:right | slot5@[8, 17] | 0 | 35 |
| 32 | b5:left | slot5@[8, 17] | 0 | 34 |
| 33 | b5:left | slot5@[8, 17] | 0 | 35 |
| 34 | b5:left | slot5@[8, 17] | 0 | 36 |
| 35 | b5:left | slot5@[8, 17] | 0 | 37 |
| 36 | b5:left | slot5@[8, 17] | 0 | 38 |
| 37 | b5:left | slot5@[8, 17] | 0 | 39 |
| 38 | b5:left | slot5@[8, 17] | 0 | 40 |
| 39 | b5:left | none | 0 | 41 |

## 方法含义

A4 将旧 GFT 的任意目标运输进展替换为与 room frontier 和已承诺 slot 对齐的进度代理。因此，它可以识别“向某个深槽承诺后是否继续朝该槽可推地接近”，并把错误行推进、承诺不可达和真正阻塞槽作为强负信号。同时，A4 对深槽路径中必须短暂停留的浅层阻塞槽加入 pass-through 例外，避免把正确的穿越动作误判为 settled 错填。
