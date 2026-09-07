# XSokoban first 与 Boxoban unfiltered/test 结构差异分析

## 结论摘要

Boxoban `unfiltered/test/000.txt` 的 1000 关是固定 10x10、4 箱、目标较分散的同分布测试；XSokoban first-40 则来自经典人工关卡，尺寸、箱数、目标密度和瓶颈结构均明显外推。
当前最佳 SD-MCTS 在 Boxoban 上达到 989/1000，主要依赖 4 箱下可精确计算且与训练/审计分布一致的 GFT 运输匹配、局部 push-level 搜索和小规模 TVA 叶值校正。XSokoban first 第 1 关已经是 11x19、6 箱、目标集中成房间，`max_steps=1000` 仍未解出；这不是单纯步数上限问题，而是结构分布、目标分配复杂度、局部循环和 TVA/GFT 标定范围同时越界。

## 数据与结果口径

- Boxoban 测试集：`datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt`，`1000` 关。
- XSokoban first-40：`datasets/sokoban_xsokoban/xsokoban_first40_boxoban_format.txt`，`40` 关。
- Boxoban 最佳结果：989/1000，失败类型为 7 个 `max_steps`、4 个 `deadlocked`。
- XSokoban 第 1 关结果：`max_steps=1000`，未解出，最终 `boxes_on_goals=1`，耗时 1927.8 秒。

## 结构统计对照

| 指标 | Boxoban unfiltered/test | XSokoban first-40 | XSokoban 第 1 关 |
|---|---:|---:|---:|
| 高度（中位数） | 10 | 15 | 11 |
| 宽度（中位数） | 10 | 18 | 19 |
| 网格面积（中位数） | 100 | 251 | 209 |
| 可行走地板数（中位数） | 32 | 157 | 139 |
| 箱子数（中位数） | 4 | 15.5 | 6 |
| 箱子/地板密度（中位数） | 0.125 | 0.108 | 0.043 |
| 初始玩家可达格数（中位数） | 22.5 | 9 | 31 |
| 初始玩家可达比例（中位数） | 0.839 | 0.062 | 0.223 |
| 初始合法 push 数（中位数） | 8 | 4 | 3 |
| 初始合法 push/箱（中位数） | 2 | 0.275 | 0.5 |
| dead-square 地板占比（中位数） | 0.333 | 0.515 | 0.705 |
| 静态割点数（中位数） | 3 | 14 | 12 |
| 移除割点后的组件数（中位数） | 3 | 9 | 8 |
| 最近目标曼哈顿距（中位数） | 2 | 1 | 1 |

XSokoban first-40 的中位箱数为 15.5，最大箱数为 32；Boxoban 测试集箱数恒为 4。这直接突破了当前模型中若干按 4 箱分布调校的归纳偏置。XSokoban 第 1 关虽然只是 first-40 中较小的一关，也已有 `6` 个箱子、`139` 个地板格和 `12` 个静态割点。

## 与 Boxoban 失败关卡的关系

Boxoban 1000 关中仅 11 关失败。失败关卡的结构统计仍大体处在 10x10、4 箱分布内，与 XSokoban 第 1 关相比，主要差异不是单个指标略难，而是 XSokoban 同时改变了尺寸、箱数、目标组织方式和走廊/房间结构。

| 指标 | Boxoban 全体中位数 | Boxoban 失败关卡中位数 | XSokoban 第 1 关 |
|---|---:|---:|---:|
| 高度 | 10 | 10 | 11 |
| 宽度 | 10 | 10 | 19 |
| 网格面积 | 100 | 100 | 209 |
| 可行走地板数 | 32 | 34 | 139 |
| 箱子数 | 4 | 4 | 6 |
| 箱子/地板密度 | 0.125 | 0.118 | 0.043 |
| 初始玩家可达格数 | 22.5 | 1 | 31 |
| 初始玩家可达比例 | 0.839 | 0.05 | 0.223 |
| 初始合法 push 数 | 8 | 2 | 3 |
| 初始合法 push/箱 | 2 | 0.5 | 0.5 |
| dead-square 地板占比 | 0.333 | 0.316 | 0.705 |
| 静态割点数 | 3 | 4 | 12 |
| 移除割点后的组件数 | 3 | 3 | 8 |
| 最近目标曼哈顿距 | 2 | 2 | 1 |

## 失败机制解释

1. **组合规模外推**：Boxoban 测试集固定 4 箱，XSokoban first-40 至少 6 箱，部分关卡达到数十箱。当前 GFT 的精确箱-目标匹配、Sinkhorn commitment 和 ordering/capacity 读数在 4 箱上稳定，但在多箱目标房间中会变成高熵、多等价分配问题。XSokoban 第 1 关初始 commitment 仅为 `0`，说明箱-目标绑定本身很弱。

2. **目标集中与房间顺序约束增强**：Boxoban unfiltered 常见目标较分散，匹配下界下降即可提供有效局部进展信号。XSokoban 第 1 关有 6 个目标集中在右下目标房间，进入顺序和填充顺序比单纯距离更关键；失败日志最终仍有 ordering debt 和 soft-deadlock pressure，说明模型能读出风险，但该读数没有转化为足够强的长程承诺。

3. **push-level 局部搜索陷入低分支循环**：XSokoban `max_steps=1000` 日志中，前 20 步 trace 已出现多次同一箱子重复向右推动，许多 root 只有 1 个有效 child。提高 `max_steps` 允许更长执行，但不能改变每一步局部 MCTS 的动作排序和承诺机制，因此仍会在长走廊/目标房间外围消耗搜索。

4. **TVA 训练与审计分布不覆盖 XSokoban**：当前 TVA value 训练来自 40 个 Boxoban/medium 切片状态，独立审计也是 Boxoban 叶状态；它没有见过 11x19、6 箱及更多箱、目标房间式 XSokoban 状态。因此 Boxoban 上有用的 `V_SD + TVA` 校正，在 XSokoban 上缺少可靠校准证据。

5. **评价尺度不只是不够长**：第 1 关在 `max_steps=1000` 下仍只完成 1 个目标，最终 matching/transport 工作量仍高。这说明失败不只是 `80` 步上限偏短，而是规划前缀没有进入正确的目标填充策略。

## 后续实验建议

- 对 XSokoban 第 1 关先做离线可解性与最短/可行 push 参考解验证，作为诊断轨迹锚点。
- 在不改变论文主结果的前提下，增加 `--trace-gft-readouts` 复跑 XSokoban 第 1 关前 50-100 步，检查 P/M/A/K/D/Q/C 的时间序列是否饱和或误导。
- 为多箱目标房间加入显式 box-goal/room-order commitment option，而不是继续只放大 `max_steps`。
- 若要系统评估泛化，应分开报告 Boxoban hard、XSokoban first-40 与大尺寸 XSokoban 子集；三者不能互相替代。
