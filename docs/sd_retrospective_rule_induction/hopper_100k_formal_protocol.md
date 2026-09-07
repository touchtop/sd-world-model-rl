# Hopper 100k Formal Protocol

## 目标

该协议用于评估 SD retrospective rule induction 在 Hopper-Hop 上的正式有效性。正式测试不再使用中途 checkpoint 接续训练作为主要证据，而是从零开始训练到 `100k`，并与官方 TD-MPC2 `100k` 口径对照。

## 前置条件

正式 100k 运行前，规则生成机制必须先通过以下检查：

1. 逻辑审查：规则状态谓词、动作方向、冲突削弱、信心更新和上线门控与 Hopper 任务机制一致。
2. smoke test：规则 prior 能被训练入口加载，训练日志能观察到规则命中和惩罚项，短程运行不出现工程中断或明显崩塌。
3. 轨迹误伤审查：规则不得明显压制已有好姿态片段中的关键动作方向。

## 主测试口径

主测试采用空规则库从头闭环生成规则，不使用从高预算轨迹中离线挖出的初始规则库。这样可以避免将 `200k` 或其他高预算策略信息泄漏到 `0-100k` 正式测试中。

固定设置如下：

```text
task                         = hopper-hop
world_model_kind             = tdmpc2
seeds                        = 1,2,3
total_budget                 = 100k
cycle_steps                  = 5000
cycles                       = 20
formal_eval_freq             = 25000
formal_checkpoint_freq       = 25000
formal_eval_episodes         = 10
sd_online_supervision        = false
rule_prior_mode              = penalty_only
rule_prior_allowed_statuses  = v8_replay_safe_candidate
rule_required_status         = v8_replay_safe_candidate
```

第一周期使用空规则库，默认不启用规则 prior；从第二周期开始，训练只读取上一周期闭环生成并满足上线状态的规则。

## 启动命令

正式主测试命令如下。GPU 编号可按实际空闲资源调整。

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --gpu-ids 1,2,3 \
  --seeds 1,2,3 \
  --cycles 20 \
  --cycle-steps 5000 \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --exp-prefix tdmpc2_v8formal_100k_empty_seed123_YYYYMMDD \
  --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_empty_seed123_YYYYMMDD \
  --rule-prior-allowed-statuses v8_replay_safe_candidate \
  --rule-required-status v8_replay_safe_candidate \
  --rule-prior-mode penalty_only \
  --rule-prior-opposite-penalty 0.02 \
  --rule-prior-cap-penalty 0.02 \
  --rule-prior-preferred-cap 0.70 \
  --max-cycle-candidates 3 \
  --sd-epochs 3 \
  --sd-max-transitions 50000
```

不传入 `--initial-resume`、`--initial-sd-checkpoint` 和 `--initial-rule-library`。脚本会创建空初始规则库，并从零开始训练。

## seeded-rule 变体

若需要测试一个已经通过逻辑审查和 smoke test 的离线规则库，可另设 seeded-rule 变体。该变体必须显式记录规则来源，并不得与空库主测试混为同一证据。

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --gpu-ids 1,2,3 \
  --seeds 1,2,3 \
  --cycles 20 \
  --cycle-steps 5000 \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --exp-prefix tdmpc2_v8formal_100k_seeded_seed123_YYYYMMDD \
  --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_seeded_seed123_YYYYMMDD \
  --initial-rule-library PATH_TO_REVIEWED_RULE_LIBRARY.json \
  --enable-rule-prior-cycle1 \
  --rule-prior-allowed-statuses v8_replay_safe_candidate \
  --rule-required-status v8_replay_safe_candidate \
  --rule-prior-mode penalty_only \
  --rule-prior-opposite-penalty 0.02 \
  --rule-prior-cap-penalty 0.02 \
  --rule-prior-preferred-cap 0.70
```

## 官方对照

当前本地附件中记录的官方参考值为：

```text
TD-MPC2 official, 100k: mean = 13.63, population_std = 10.41
TD-MPC2 official, 200k: mean = 166.63, population_std = 34.89
```

来源：

```text
docs/neurips/paper/Attachments/data/hopper/mechanism_oriented_200k/hopper_public_baseline_comparison.csv
external/tdmpc2/results/tdmpc2/hopper-hop.csv
```

正式报告应优先使用 `100k` 同预算对照；`200k` 只作为训练阶段性背景，不作为 100k 主结论的比较基准。

## 结果审查

正式报告至少包含以下内容：

1. 三种子 `25k / 50k / 75k / 100k` 标准 `10ep` eval 均值、标准差和逐 seed 分数。
2. `100k` 与官方 TD-MPC2 `100k` 的同预算比较。
3. 最终规则库状态分布，包括上线规则、候选规则、削弱规则和被拒规则数量。
4. 规则命中统计，包括 `rule_prior_hit`、非零惩罚比例和平均先验项。
5. 轨迹审查，包括规则命中子状态下动作分布变化、好姿态片段、失败模式和规则解释是否一致。

## 中止条件

若出现以下情况，应中止正式运行或将其降级为工程调试：

1. 训练入口误启用 `sd_hopper` 或在线 SD supervision。
2. 规则 prior 未被加载，或 `rule_prior_*` 指标全程缺失。
3. 规则库来自高预算轨迹但未标记为 seeded-rule 变体。
4. formal eval 不是标准 `10` 条 episode。
5. 周期总结未能生成 SD explainer、反事实归因或闭环规则库。
