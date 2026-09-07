# XSokoban teacher 轨迹驱动的 Room-GFT 缺失机制分析

## 核心结论

本次尝试分析 XSokoban first 前 `3` 关。在本次 teacher 配置下解出 `1` 关，未解出 `2` 关。
teacher 配置为 matching backend `dp`、deadlock mode `exact`、heuristic weight `1`、viable term `True`、每关最多 `90` 秒或 `100000` 个展开。
teacher 轨迹只用于诊断当前 Room-GFT/A6 评分层的缺口，不注入 planner 出手，也不作为无 teacher 求解率结果。
当前最稳定的结论是：A6 的主要缺失不是 exact transition 或 corridor macro 的执行能力，而是缺少可自主生成的 room fill plan、箱子释放/准备动作价值和多箱槽位调度状态。

## Teacher 可用性

| level | boxes | teacher status | pushes | expanded | generated | seconds |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 6 | solved | 97 | 7248 | 69620 | 73.817 |
| 1 | 10 | teacher_time_budget_unsolved | NA | 139 | 2872 | 90.001 |
| 2 | 11 | teacher_time_budget_unsolved | NA | 892 | 4925 | 90.05 |

## 已解 teacher 轨迹上的 A6 评分差距

| level | rank scope | rank1 frac | rank<=3 frac | max rank | setup/release underrewarded | low rank | long corridor runs | MCTS match frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97 | 0.598 | 0.856 | 13 | 20 | 14 | 9 | 0.6 |

## 缺失机制清单

### room_fill_plan_state

诊断依据：A7 解依赖 future-goal/slot trace，而 A6 在相同结构下不能自主生成后续槽位顺序。

修订方向：在 RoomCommitmentState 之外增加 room fill phase、frontier slot queue 或可更新的 fill-plan stock。

### box_release_and_setup_value

诊断依据：gap tag 中 setup_or_release_underrewarded=20，teacher_low_prior_rank=14。

修订方向：增加释放后续箱子、保持通道和准备入房动作的正反馈，而不是只奖励即时 transport/slot cost 下降。

### autonomous_corridor_option_with_schedule_guard

诊断依据：teacher 轨迹中长度大于等于 3 的同向走廊段共有 9 段。

修订方向：保留 A6 corridor macro，但由自主 fill plan 决定何时触发、触发哪个箱子和哪个 slot。

### planner_choice_calibration

诊断依据：rank-1 fraction=0.598；MCTS sampled teacher-match fraction=0.600。

修订方向：对 teacher 轨迹上 teacher action rank 低的位置生成 value/prior 审计样本，用于校准 A6 leaf 和 room prior。

### long_horizon_transport_vs_schedule_tradeoff

诊断依据：short_transport_competes=16。

修订方向：将短期运输下降与后续 room-order 可行性分离计入价值，避免局部 P 下降覆盖全局调度风险。

## 解释边界

若第 2、3 关在本次 teacher 时间预算内未解出，则多关综合结论目前主要来自已解关卡。这不是 A6 缺失不存在的证据，而是说明 teacher 侧也需要更强启发式、缓存或更高预算才能形成完整多关轨迹库。
