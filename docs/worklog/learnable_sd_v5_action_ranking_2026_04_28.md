# Hopper SD Action Ranking 诊断记录（2026-04-28）

## 目标

在已完成 `phase-sliced error`、`residual usage`、`policy-distribution validation` 后，继续执行验证计划中的下一项：

> 对真实、可回放的 Hopper 状态，比较当前模型对候选动作的排序，与环境短视界真实后果的排序是否一致。

本轮诊断的重点不是最终回报高低，而是：

1. 在 planner 真正会访问的状态上，`SDWorldModel + learned reward + Q + pi` 是否能区分更好与更差的动作；
2. 如果不能，失败是因为 dynamics 排名错了，还是因为当前状态上真实任务回报本身几乎无区分度。

## 新增脚本

新增脚本：

- [analyze_hopper_action_ranking_diagnostic.py](/home/data/round/code/sd_world_model_rl/scripts/analyze_hopper_action_ranking_diagnostic.py)

脚本实现如下：

1. 读取 `official_eval/episode_csv/episode_XX.csv`，复用其中的 `qpos/qvel` 恢复 MuJoCo 物理状态；
2. 从同一 agent checkpoint 加载 `SDWorldModel`，保留其 `reward / Q / pi` 头；
3. 对每个采样状态构造候选动作集：
   1. `policy_mean`
   2. `zero`
   3. 4 个关节的正负 basis 动作
   4. 若干围绕 `policy_mean` 的随机扰动
4. 对每个候选动作同时计算：
   1. 模型预测的 `3-step reward + tail Q` bootstrap 分数
   2. 环境真实 `3-step env reward + tail Q(true_z_3)` bootstrap 分数
   3. 纯环境 `3-step` 累积 reward
5. 统计：
   1. `Spearman` 排序相关
   2. `true-best action` 在模型预测中的 best-case rank
   3. `top1 / top3` 命中率
   4. 按基态标签分组的结果

这里需要明确：

1. 由于 Hopper 在 `25k` 时环境 reward 仍极度稀疏，主要排序信号来自 `env reward + Q(true_z_h)` 的 bootstrap 目标；
2. 这不是“纯环境 oracle value”，而是对当前 planner 实际使用接口的一种近似对照。

## 首轮结果一：只看 early falling 状态

使用：

- checkpoint：
  [25000.pt](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/2/sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1/models/25000.pt)
- trace：
  `seed2` 的 `25k official_eval`
- 过滤：
  `samples_per_trace=1, step<=5`

结果目录：

- [seed2_25k_falling/summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_ranking/seed2_25k_falling/summary.json)

关键结果：

1. `10` 个采样状态全部为 `falling_air`
2. `informative_state_count = 7`
3. `informative_spearman_pred_vs_true_boot_mean = -0.0883`
4. `informative_spearman_pred_vs_true_env_mean = -0.4154`
5. `informative_top1_hit_mean = 0.0`
6. `informative_top3_contains_true_top1_mean = 0.0`
7. `informative_pred_best_true_boot_mean = 0.0106`
8. `informative_true_best_true_boot_mean = 0.0707`

结论很直接：

> 在最早的下落相位上，当前模型对候选动作的排序不是“弱相关”，而是接近零甚至负相关；模型挑出的动作，真实短视界后果明显差于真实最优动作。

这与此前 `policy-distribution validation` 中观察到的 `airborne / landing / high_vz_impact` 薄弱点是一致的。

## 定向结果三：只看“即将落地”的 pre-landing falling 状态

为了避免“过早的 falling 状态距离接地还太远，短视界信号不足”这一解释，又对脚本做了两点增强：

1. 支持按 trace 的 `dominant_state_id` 过滤基态；
2. 支持只保留“未来若干步内会接地”的 falling 状态；
3. 额外记录更长的真实环境回报 `true_long_env_return`，用于比较模型短视界分数与更长任务后果的一致性。

使用：

- 同一 `seed2@25k` checkpoint 与 official_eval trace
- 过滤：
  `state_name_filter=falling_air`
  `require_trace_landing_within=6`
- 对照：
  `horizon=3`
  `true_env_horizon=10`

结果目录：

- [seed2_25k_prelanding/summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_ranking/seed2_25k_prelanding/summary.json)

关键结果：

1. `num_samples = 20`
2. 全部基态均为 `falling_air`
3. `informative_state_count = 4`
4. `informative_spearman_pred_vs_true_boot_mean = -0.1587`
5. `informative_spearman_pred_vs_true_env_mean = -0.3721`
6. `informative_spearman_pred_vs_true_long_env_mean = -0.2544`
7. `informative_top1_hit_mean = 0.0`
8. `informative_top3_contains_true_top1_mean = 0.0`
9. `informative_pred_best_true_long_env_mean = 0.0370`
10. `informative_true_best_true_long_env_mean = 0.5725`

这组结果比“只看最初 0~5 步 falling 状态”更强，因为它明确说明：

> 即使只看那些已经非常接近接地、理论上最该考验落地动作选择的空中状态，当前模型的动作排序仍然是负相关的，且模型挑出的动作在 10 步真实环境回报上远差于真实最优动作。

因此，当前问题不能再解释为“只是取样太早”或“3 步窗口太短”。更准确的说法应是：

> 当前 `SDWorldModel` 在 `falling_air -> landing` 这一关键转换带上的动作价值排序，确实还不可靠。

## 首轮结果二：落地后到中段失败演化

使用：

- 同一 checkpoint 与 trace
- 过滤：
  `samples_per_trace=3, 10<=step<=300, exclude_done`

结果目录：

- [seed2_25k_filtered/summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_action_ranking/seed2_25k_filtered/summary.json)

关键结果：

1. `num_samples = 30`
2. 基态分布：
   1. `falling_air = 3`
   2. `grounded_rear = 8`
   3. `grounded_prone = 18`
   4. `grounded_upright = 1`
3. `informative_state_count = 1`
4. 全体平均 `top1_hit_mean = 0.8667`、`top3_contains_true_top1_mean = 0.8667`
5. 但这并不意味着模型排序优秀，因为绝大多数状态的真实 bootstrap 回报几乎完全退化为常数

这里最重要的现象不是 `top1` 数值本身，而是：

> 在 30 个落地后状态里，只有 1 个状态对候选动作表现出可分辨的真实短视界差异；其余 29 个状态上，真实 bootstrap 目标几乎是平的。

因此，这部分状态上的高 `top1` 命中率主要是 ties 造成的表面现象，不应被解释为动作排序已经可靠。

对于唯一的非退化状态：

1. `informative_spearman_pred_vs_true_boot_mean = -0.1516`
2. `informative_top1_hit_mean = 0.0`
3. `informative_top3_contains_true_top1_mean = 0.0`
4. `informative_pred_best_true_boot_mean = 1.3572`
5. `informative_true_best_true_boot_mean = 2.0119`

即使在这个少数“有区分度”的状态上，排序也是错的。

## 结果解释

本轮 action ranking 诊断给出两条非常关键的结论。

### 1. 当前最需要修的，确实是 airborne / landing 相关动力学

在 early falling 状态上，真实短视界候选动作是有明显优劣差异的，但模型排序为负相关，且选不出真实 top action。

这说明：

> 当前世界模型在“动作会如何改变落地姿态和后续短视界价值”这个问题上，尚未提供可用排序。

这与此前的：

1. `airborne_true` 下 `vz / omega` 误差放大，
2. `landing` 下 `dominant_state_match` 偏低，
3. `high_vz_impact` 下误差显著更大，

完全一致。

### 2. 失败地面簇上的排序问题，目前很多时候甚至无从谈起

落地后大部分 `grounded_rear / grounded_prone` 状态上，候选动作的真实短视界 bootstrap 目标几乎是平的。

这意味着：

1. 当前 policy / Q 头尚未在这些失败状态上学出有区分度的价值地形；
2. 即使 dynamics 再略微改善，也未必立刻能在这些状态上看到漂亮的 action ranking 指标；
3. 因此，当前最有信息量的 ranking 状态反而是下落与接触转换初期，而不是已经塌入失败簇后的后半段。

## 当前阶段结论

可以将当前结论收敛为：

1. `policy-distribution validation` 说明：当前 dynamics 在 replay 分布上已比纯监督初始化更适配；
2. `action ranking diagnostic` 说明：但在真正关键的 early falling / landing 相位，模型仍不能正确排序动作；
3. 定向 `pre-landing` 诊断进一步确认：即使只看即将接地的 falling 状态，排序错误仍然存在；
4. 对已经塌入 `rear/prone` 失败簇的状态，当前短视界真实目标大多已退化为近常数，因此这部分状态不适合作为主要 ranking 验证集。

因此，下一步更合理的推进方向是：

1. 针对 `falling_air -> landing` 片段做更聚焦的 phase-targeted 排序诊断；
2. 继续从接触转换、离地判别、角速度响应这三条线上检查 SD 主干与 internal residual；
3. 暂不重新引入奖励 shaping，也不应把当前高 `top1` ties 误判为世界模型已经可用于有效规划。
