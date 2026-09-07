# Reproducibility

## 初始记录

本研究线第一阶段依赖已有 Hopper 轨迹与日志，不重新定义环境或基线。

当前主要数据来源为：

1. `logs/hopper-hop/*/sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed*_20260501_run1/`
2. `logs/hopper-hop/*/sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed*_resume100k_20260502_run1/`
3. `logs/hopper-hop/*/sd_hopper_300k_pureenv_sdinit_mech_sd_official_budget_seed*_resume200k_20260504_run1/`

第一阶段默认固定随机种子，用于轨迹抽样、规则训练/验证划分和反事实动作采样。具体脚本、环境版本和随机种子将在首个实现脚本落地后补充。

## Hopper 100k formal protocol

正式 Hopper 规则方法测试统一采用从头 `100k` 训练口径。协议文档为：

```text
docs/sd_retrospective_rule_induction/hopper_100k_formal_protocol.md
```

主测试不传入 `--initial-resume`、`--initial-sd-checkpoint` 或 `--initial-rule-library`，由闭环脚本创建空初始规则库，并在第一周期关闭规则 prior。规则生成、SD 接续拟合和闭环规则库更新仍保持 `5k` 周期，标准 eval 与 checkpoint 为每 `25k` 一次，eval episode 数为 `10`。

官方 TD-MPC2 对照值来自：

```text
docs/neurips/paper/Attachments/data/hopper/mechanism_oriented_200k/hopper_public_baseline_comparison.csv
external/tdmpc2/results/tdmpc2/hopper-hop.csv
```

当前记录的同预算参考为：

```text
TD-MPC2 official, 100k: mean = 13.63, population_std = 10.41
```

## 代码入口

当前已建立的离线代码入口为：

- [run_offline_v1.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/run_offline_v1.py)
- [build_hopper_manifest.py](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/build_hopper_manifest.py)

对应初始配置为：

- [hopper_offline_v1.yaml](/home/data/round/code/sd_world_model_rl/experiments/sd_retrospective_rule_induction/configs/hopper_offline_v1.yaml)

默认输出位置为：

- `results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_manifest.json`
- `results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_selected_trajectories.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_attributions.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.md`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v1_rules.json`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v1_rules.md`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v1_rule_validation.json`

当前配置固定 `random_seed=20260505`。该随机种子将在后续轨迹抽样、训练/验证划分和反事实动作采样脚本中统一沿用。

第一版复现实验命令为：

```bash
python experiments/sd_retrospective_rule_induction/run_offline_v1.py
```

本地验证结果：索引 `9` 条运行，选择 `258 / 360` 条 episode，成功完成 `258` 条 episode 的归因，生成 `3` 条候选规则，并以 seed `3` 作为默认 hold-out seed 完成离线验证。

加入反事实动作比较后，本地验证结果扩展为：生成 `774` 个关键状态反事实记录，默认每条轨迹最多 `3` 个状态，每个状态比较实际动作、零动作和 `24` 个单关节探测动作，rollout horizon 为 `5`。

## 第二版复现入口

第二版命令为：

```bash
python experiments/sd_retrospective_rule_induction/run_offline_v2.py --epochs 3 --max-transitions 50000
```

主要输出：

- `results/sd_retrospective_rule_induction/sd_explainers/hopper_offline_v2_sd_explainer.pt`
- `results/sd_retrospective_rule_induction/sd_explainers/hopper_offline_v2_sd_explainer_summary.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v2_counterfactuals_explainer.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v2_counterfactuals_explainer.md`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v2_state_action_rules.json`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v2_state_action_rules.md`

本地验证结果：解释器使用 `50,000` 条转移训练，训练集 `45,010`，验证集 `4,990`，生成 `774` 个解释器反事实状态记录，并归纳出 `8` 条状态-动作级候选规则。

第二版管线已继续纳入第三版规则一致性审查，新增输出：

- `results/sd_retrospective_rule_induction/reports/hopper_offline_v3_rule_consistency.json`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v3_rule_consistency.md`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v3_reliable_rules.json`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v3_reliable_rules.md`

本地审查结果：`8` 条 v2 候选规则中，`1` 条通过当前可信候选门槛，即 `unsupported_high_torque -> prefer hip_neg`。该规则覆盖 `3` 个 seed 与 `9` 个周期，支持数为 `28`，平均反事实优势为 `0.1823`，好轨迹误伤率为 `0.071`。

## 第四版复现入口

第四版脚本已纳入 `run_offline_v2.py` 的尾部，也可以单独运行：

```bash
python experiments/sd_retrospective_rule_induction/analyze_substate_rules.py
python experiments/sd_retrospective_rule_induction/replay_rule_scoring.py
python experiments/sd_retrospective_rule_induction/build_rule_library.py
```

主要输出：

- `results/sd_retrospective_rule_induction/rules/hopper_offline_v4_substate_rules.json`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v4_substate_rules.md`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v4_replay_rule_scoring.json`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v4_replay_rule_scoring.md`
- `results/sd_retrospective_rule_induction/rules/library.json`

本地验证结果：子状态切分识别 `33` 个子状态并生成 `4` 条 v4 子状态候选规则；离线回放显示 v3 粗规则 `unsupported_high_torque -> prefer hip_neg` 命中 `80,620` 个逐帧状态，其中好轨迹片段为 `23,646` 个，提示该粗规则仍需进一步门控或降级为 `needs_split`。

## 第五版复现入口

第五版脚本已纳入 `run_offline_v2.py` 的尾部，也可以单独运行：

```bash
python experiments/sd_retrospective_rule_induction/validate_substate_rules.py
python experiments/sd_retrospective_rule_induction/update_rule_library_status.py
```

主要输出：

- `results/sd_retrospective_rule_induction/reports/hopper_offline_v5_substate_rule_validation.json`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v5_substate_rule_validation.md`
- `results/sd_retrospective_rule_induction/rules/library_v5.json`
- `results/sd_retrospective_rule_induction/rules/library_v5.md`

本地验证结果：`4` 条 v4 子状态候选规则中，`unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos` 通过回放审查，命中 `2,029` 个逐帧状态，好轨迹命中率为 `0.053`，差轨迹命中率为 `0.887`，覆盖 `3` 个 seed 与 `12` 个周期，状态升级为 `replay_validated`。其余 `3` 条子状态规则保持 `candidate`。v3 粗规则 `unsupported_high_torque -> prefer hip_neg` 被标记为 `needs_split`。

当前 v5 规则库仍设置 `online_enabled=false`。规则库状态分布为：`needs_split=1`，`replay_validated=1`，`candidate=3`。

## 第六版复现入口

第六版新增在线 smoke 入口：

```bash
bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh <gpu_id> [exp_name] [seed]
```

该脚本默认关闭规则先验：

```bash
RULE_PRIOR_ENABLED=false bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 0 sd_hopper_v6_5k_rule_prior_off_seed1 1
```

显式开启规则先验的命令为：

```bash
RULE_PRIOR_ENABLED=true bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 0 sd_hopper_v6_5k_rule_prior_on_seed1 1
```

主要代码入口：

- `experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py`
- `experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py`
- `scripts/run_hopper_v6_5k_rule_prior_smoke.sh`

默认规则库：

- `results/sd_retrospective_rule_induction/rules/library_v5.json`

默认输出位置仍沿用 TD-MPC2 日志规范：

- `logs/hopper-hop/<seed>/<exp_name>/`

v6 smoke 的默认参数为：`steps=5000`，`eval_freq=2500`，`checkpoint_freq=2500`，`eval_episodes=5`，`num_samples=128`，`num_elites=16`，`num_pi_trajs=12`，`horizon=3`，`iterations=4`。规则先验参数为：`rule_prior_weight=0.05`，`rule_prior_opposite_penalty=0.02`，`rule_prior_max_bonus=0.10`。

已完成静态验证：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/*.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
PYTHONPATH=external/tdmpc2/tdmpc2:. python experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py --help
```

规则库读取检查确认当前只加载 `1` 条 `replay_validated` 规则，即 `unsupported_high_torque|omega_pos|vx_mid|support_none|rear -> prefer hip_pos`。

本地最小规则先验检查使用 CPU 构造 `HopperRulePriorV6`，对形状为 `(horizon=3, samples=5, action_dim=4)` 的候选动作张量返回长度为 `5` 的候选偏置向量，确认规则先验接口可被 MPPI 候选序列调用。

第六版 quick smoke 使用较轻设置完成了一轮成对验证：

```bash
STEPS=3000 EVAL_FREQ=3000 CHECKPOINT_FREQ=3000 EVAL_EPISODES=1 SEED_PRETRAIN_UPDATES=20 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=false bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 2 sd_hopper_v6_3k_rule_prior_off_seed1_finalonly_20260505 1
STEPS=3000 EVAL_FREQ=3000 CHECKPOINT_FREQ=3000 EVAL_EPISODES=1 SEED_PRETRAIN_UPDATES=20 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=true bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 3 sd_hopper_v6_3k_rule_prior_on_seed1_finalonly_20260505 1
```

结果报告：

- `results/sd_retrospective_rule_induction/reports/hopper_v6_rule_prior_quick_smoke_20260505.md`

本地结果：规则开启与关闭的最终 `episode_reward` 均为 `0.0`，但规则开启线在目标子状态命中时将 hip 动作完全推向 `hip_pos`，并伴随更差的 forward displacement、full-foot support、backward-speed fraction 和 rotation-penalty fraction。该 quick smoke 不支持增强当前在线规则干预。

第六点一版离线规则干预审查命令为：

```bash
python experiments/sd_retrospective_rule_induction/simulate_rule_prior_variants_v61.py
```

主要输出：

- `results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit.json`
- `results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit.md`

本地审查结果：在规则关闭 trace 中，目标子状态命中 `54` 次，penalty-only 会惩罚 `35` 个实际动作、保持 `19` 个动作不变；在规则开启失败 trace 中，目标子状态命中 `41` 次，全部为正向饱和 `hip_pos`，penalty-only 会惩罚全部 `41` 个动作。该结果说明，penalty-only 能避免继续正向强化失败 trace 中的饱和 `hip_pos`。

下一轮在线 penalty-only quick smoke 可使用：

```bash
STEPS=3000 EVAL_FREQ=3000 CHECKPOINT_FREQ=3000 EVAL_EPISODES=1 SEED_PRETRAIN_UPDATES=20 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=true RULE_PRIOR_MODE=penalty_only RULE_PRIOR_OPPOSITE_PENALTY=0.02 RULE_PRIOR_PREFERRED_CAP=0.70 RULE_PRIOR_CAP_PENALTY=0.02 bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 3 sd_hopper_v6p1_3k_rule_prior_penalty_only_seed1_20260505 1
```

本地 penalty-only quick smoke 已完成。主要输出：

- `logs/hopper-hop/1/sd_hopper_v6p1_3k_rule_prior_penalty_only_seed1_20260505/eval.csv`
- `logs/hopper-hop/1/sd_hopper_v6p1_3k_rule_prior_penalty_only_seed1_20260505/train.csv`
- `logs/hopper-hop/1/sd_hopper_v6p1_3k_rule_prior_penalty_only_seed1_20260505/official_eval/step_0003000/eval_episode_summary.csv`
- `results/sd_retrospective_rule_induction/reports/hopper_v6p1_penalty_only_quick_smoke_20260505.md`
- `results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit_with_penalty_smoke.json`
- `results/sd_retrospective_rule_induction/reports/hopper_v6p1_rule_prior_variant_audit_with_penalty_smoke.md`

本地结果：penalty-only 最终 `episode_reward=0.0`，`episode_success=0.0`。相较 v6 `prefer` 线，目标子状态命中从 `41` 降至 `6`，饱和 `hip_pos` 从 `41` 降至 `0`，`backward_speed_frac` 从 `0.202` 降至 `0.010`，`rotation_penalty_frac` 从 `0.150` 降至 `0.016`。相较 rule-off 基线，penalty-only 仍未恢复 full-foot support，`final_rootx` 也较低。

## 第六点一版 20k pilot 入口

Hopper 任务奖励稀疏且阶段依赖明显，3k quick smoke 仅用于局部行为信号检查。下一轮采用 20k 成对 pilot：

```bash
STEPS=20000 EVAL_FREQ=5000 CHECKPOINT_FREQ=5000 EVAL_EPISODES=3 SEED_PRETRAIN_UPDATES=500 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=false bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 2 sd_hopper_v6p1_20k_rule_prior_off_seed1_20260505 1
STEPS=20000 EVAL_FREQ=5000 CHECKPOINT_FREQ=5000 EVAL_EPISODES=3 SEED_PRETRAIN_UPDATES=500 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=true RULE_PRIOR_MODE=penalty_only RULE_PRIOR_OPPOSITE_PENALTY=0.02 RULE_PRIOR_PREFERRED_CAP=0.70 RULE_PRIOR_CAP_PENALTY=0.02 bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 3 sd_hopper_v6p1_20k_rule_prior_penalty_only_seed1_20260505 1
```

对比时应同时读取 `eval.csv`、`train.csv` 和 `official_eval/step_*/eval_episode_summary.csv`，不只比较 `episode_reward`。

## TVA 上下文动作竞争复现入口

`200k` 接续线 `tdmpc2_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1` 的最终规则库可用以下命令执行 TVA 上下文动作竞争审计：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_tva_context_action_competition.py \
  --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json \
  --tva-counterfactual-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_counterfactual_audit.json \
  --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_context_action_competition_audit.json
```

本地结果：

```text
contexts = 17
contexts_with_conflicts = 7
TVA-supported rules = 21
TVA-suspect rules = 18
matched_context_records = 1547
```

可用以下命令生成只读影子规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_tva_context_competition_shadow_library.py \
  --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json \
  --audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_context_action_competition_audit.json \
  --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_competition_shadow.json
```

本地结果：

```text
changed = 34
status_counts = {'confidence_retired': 58, 'confidence_active': 51, 'confidence_observing': 44}
reason_counts = {
  'tva_context_competition_observed_no_change': 12,
  'tva_context_competition_suspect_weakened': 18,
  'tva_context_competition_supported_strengthened': 21,
  'not_in_tva_context_competition_audit': 44
}
```

可用以下命令执行三种子 eval-only 对照：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py \
  --seeds 1 2 3 \
  --gpus 1 1 1 \
  --modes rule_off original tva_context_shadow \
  --eval-episodes 10 \
  --no-resume
```

输出为：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_competition_shadow_eval_20260520.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_competition_shadow_eval_20260520.md
```

本地 eval-only 结果：

```text
rule_off mean = 125.407
original mean = 127.866
tva_context_shadow mean = 120.461
```

行为审计命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py \
  --group rule_off logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_rule_off_20260520/official_eval/step_0200040/episode_csv \
  --group rule_off logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_rule_off_20260520/official_eval/step_0200040/episode_csv \
  --group rule_off logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_rule_off_20260520/official_eval/step_0200040/episode_csv \
  --group original logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_original_20260520/official_eval/step_0200040/episode_csv \
  --group original logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_original_20260520/official_eval/step_0200040/episode_csv \
  --group original logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_original_20260520/official_eval/step_0200040/episode_csv \
  --group tva_context_shadow logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_tva_context_shadow_20260520/official_eval/step_0200040/episode_csv \
  --group tva_context_shadow logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_tva_context_shadow_20260520/official_eval/step_0200040/episode_csv \
  --group tva_context_shadow logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_tva_context_shadow_20260520/official_eval/step_0200040/episode_csv \
  --output results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_competition_shadow_eval_posture_20260520.json
```

对应 rule-effect 审计输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_eval_original_rule_effect_20260520.json
results/sd_retrospective_rule_induction/reports/hopper_200k_eval_tva_context_shadow_rule_effect_20260520.json
```

本地行为审计结果：原规则库与 TVA 影子库的 selected-gate 命中率分别为 `0.6159` 和 `0.6092`；加权有害方向比例分别约为 `0.3011` 和 `0.3016`。该结果不支持将 TVA 上下文竞争影子库直接替换为 live 规则库。

role/effect-gated 保守版本命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_tva_context_role_gated_shadow_library.py \
  --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json \
  --tva-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_context_action_competition_audit.json \
  --effect-audit results/sd_retrospective_rule_induction/reports/hopper_200k_eval_original_rule_effect_20260520.json \
  --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_shadow.json
```

本地结果：

```text
changed = 7
status_counts = {'confidence_retired': 58, 'confidence_active': 61, 'confidence_observing': 34}
reason_counts includes:
  tva_suspect_but_recovery_consistent_protected = 3
  tva_supported_and_effect_consistent = 11
```

role-gated eval-only 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py \
  --seeds 1 2 3 \
  --gpus 1 1 1 \
  --modes tva_context_role_gated \
  --eval-episodes 10 \
  --no-resume \
  --summary results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_shadow_eval_20260520.json
```

本地结果：

```text
tva_context_role_gated seed means = 102.667 / 123.509 / 154.039
tva_context_role_gated mean = 126.738
```

对应审计输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_shadow_eval_posture_20260520.json
results/sd_retrospective_rule_induction/reports/hopper_200k_eval_tva_context_role_gated_rule_effect_20260520.json
```

本地审计结果：role-gated 版本 selected-gate 命中率为 `0.6200`，加权方向一致率约 `0.7010`，加权有害方向比例约 `0.2990`。该结果支持将 role/effect-gated TVA 上下文竞争保留为下一轮 shadow-only 候选，但仍不足以直接进入正式在线重训。

四模式统一汇总可复用已有报告生成，不重复执行 eval：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py \
  --seeds 1 2 3 \
  --gpus 1 1 1 \
  --modes rule_off original tva_context_shadow tva_context_role_gated \
  --eval-episodes 10 \
  --resume \
  --summary results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_all_shadow_eval_20260521.json
```

统一汇总输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_all_shadow_eval_20260521.json
results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_all_shadow_eval_20260521.md
```

### 20k pilot 入口更正

上述最初命令保留为历史记录。实际执行中发现该入口默认仍会走 `sd_hopper` 配置，导致在线 SD 监督训练被误启用。误启动的实验名为：

```text
sd_hopper_v6p1_20k_rule_prior_off_seed1_20260505
sd_hopper_v6p1_20k_rule_prior_penalty_only_seed1_20260505
```

该组运行已停止，不作为 20k pilot 结果。修正后的纯 TD-MPC2 命令为：

```bash
WORLD_MODEL_KIND=tdmpc2 SD_ONLINE_SUPERVISION_ENABLED=false STEPS=20000 EVAL_FREQ=5000 CHECKPOINT_FREQ=5000 EVAL_EPISODES=3 SEED_PRETRAIN_UPDATES=500 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=false bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 2 tdmpc2_v6p1_20k_rule_prior_off_seed1_20260505 1
WORLD_MODEL_KIND=tdmpc2 SD_ONLINE_SUPERVISION_ENABLED=false STEPS=20000 EVAL_FREQ=5000 CHECKPOINT_FREQ=5000 EVAL_EPISODES=3 SEED_PRETRAIN_UPDATES=500 NUM_SAMPLES=64 NUM_ELITES=8 NUM_PI_TRAJS=8 ITERATIONS=3 RULE_PRIOR_ENABLED=true RULE_PRIOR_MODE=penalty_only RULE_PRIOR_OPPOSITE_PENALTY=0.02 RULE_PRIOR_PREFERRED_CAP=0.70 RULE_PRIOR_CAP_PENALTY=0.02 bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 3 tdmpc2_v6p1_20k_rule_prior_penalty_only_seed1_20260505 1
```

主要输出：

- `logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_off_seed1_20260505/eval.csv`
- `logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_off_seed1_20260505/train.csv`
- `logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_off_seed1_20260505/official_eval/step_*/eval_episode_summary.csv`
- `logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_penalty_only_seed1_20260505/eval.csv`
- `logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_penalty_only_seed1_20260505/train.csv`
- `logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_penalty_only_seed1_20260505/official_eval/step_*/eval_episode_summary.csv`
- `results/sd_retrospective_rule_induction/reports/hopper_v6p1_tdmpc2_20k_penalty_only_vs_off_seed1_20260505.md`

复核命令：

```bash
pgrep -af "train_hopper_v6_rule_prior.py"
tail -n 20 logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_off_seed1_20260505/eval.csv
tail -n 20 logs/hopper-hop/1/tdmpc2_v6p1_20k_rule_prior_penalty_only_seed1_20260505/eval.csv
```

## 第七版闭环入口

第一版闭环 runner 为：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --seed 1 --cycles 3 --cycle-steps 20000
```

`--cycle-steps` 是可控参数，可替换为 `10000`、`20000` 或其他周期长度。该参数只决定每次 TD-MPC2 训练后何时触发 SD 总结，不改变方法论定义。

关键默认参数：

- `world_model_kind=tdmpc2`
- `sd_online_supervision_enabled=false`
- `rule_prior_mode=penalty_only`
- `cycles=3`
- `cycle_steps=20000`
- `eval_freq=5000`
- `checkpoint_freq=5000`
- `eval_episodes=3`
- `sd_epochs=3`
- `sd_max_transitions=50000`
- `rule_required_status=replay_validated`

主要代码入口：

- `experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py`
- `experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py`
- `experiments/sd_retrospective_rule_induction/fit_sd_explainer.py`
- `scripts/run_hopper_v6_5k_rule_prior_smoke.sh`

主要输出位置：

- TD-MPC2 日志：`logs/hopper-hop/<seed>/<exp_prefix>_cycle*_target*/`
- 周期总结：`results/sd_retrospective_rule_induction/closed_loop/hopper_v7/cycle_*/`
- 周期 SD 解释器：`results/sd_retrospective_rule_induction/closed_loop/hopper_v7/cycle_*/sd_explainers/sd_explainer_cycle_*.pt`
- 周期规则库：`results/sd_retrospective_rule_induction/closed_loop/hopper_v7/cycle_*/rules/library_closed_loop_cycle_*.json`

最小参数展开检查：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --cycles 1 --cycle-steps 1000 --exp-prefix drycheck_v7 --dry-run
```

规则库单步更新器检查：

```bash
python experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py --base-library results/sd_retrospective_rule_induction/rules/library_v5.json --cycle-library results/sd_retrospective_rule_induction/rules/library_v5.json --output /tmp/closed_loop_rule_update_test.json --cycle 1
```

本地静态验证：

```bash
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
python -m py_compile experiments/sd_retrospective_rule_induction/fit_sd_explainer.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
```

## 第七版 10k x 5 闭环实验

实验整理日期：2026-05-06。实验命名使用 2026-05-05 后缀，以保留启动时的运行标识。

主命令：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 2 --seed 1 --cycles 5 --cycle-steps 10000 --exp-prefix tdmpc2_v7_closed_loop_10k5_seed1_20260505 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v7_10k5_seed1_20260505 --eval-freq 5000 --checkpoint-freq 5000 --eval-episodes 3 --seed-pretrain-updates 500 --num-samples 64 --num-elites 8 --num-pi-trajs 8 --iterations 3 --rule-prior-mode penalty_only
```

运行环境说明：沙箱内 PyTorch 无法访问 CUDA 设备，首次启动因 `torch.cuda.is_available()` 断言失败。随后在具备 GPU 访问权限的环境中运行，CUDA 可见设备数为 `4`，本次指定 `--gpu-id 2`。

固定参数：

- seed：`1`
- cycles：`5`
- cycle_steps：`10000`
- total target steps：`50000`
- eval_freq：`5000`
- checkpoint_freq：`5000`
- eval_episodes：`3`
- seed_pretrain_updates：`500`
- num_samples：`64`
- num_elites：`8`
- num_pi_trajs：`8`
- horizon：`3`
- iterations：`3`
- rule_prior_mode：`penalty_only`

主训练约束：

- `world_model_kind=tdmpc2`
- `sd_online_supervision_enabled=false`
- `sd_supervised_init_from=null`

主要输出：

```text
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle001_target10000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle002_target20000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle003_target30000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle004_target40000
logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle005_target50000
results/sd_retrospective_rule_induction/closed_loop/hopper_v7_10k5_seed1_20260505
results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_seed1_20260505.md
```

复核命令：

```bash
python - <<'PY'
import csv, glob, os
print('cycle,step,episode_reward')
for path in sorted(glob.glob('logs/hopper-hop/1/tdmpc2_v7_closed_loop_10k5_seed1_20260505_cycle*_target*/eval.csv')):
    cycle = os.path.basename(os.path.dirname(path)).split('_cycle')[1].split('_')[0]
    with open(path) as f:
        for row in csv.DictReader(f):
            print(cycle, row['step'], row['episode_reward'], sep=',')
PY

python - <<'PY'
import glob, json, os
print('cycle,action,selected,rules')
for path in sorted(glob.glob('results/sd_retrospective_rule_induction/closed_loop/hopper_v7_10k5_seed1_20260505/cycle_*/rules/library_closed_loop_cycle_*.json')):
    cycle = os.path.basename(os.path.dirname(os.path.dirname(path))).split('_')[1]
    with open(path) as f:
        payload = json.load(f)
    update = payload.get('closed_loop_update', {})
    print(cycle, update.get('action'), update.get('selected_rule_id'), len(payload.get('rules', [])), sep=',')
PY
```

规则漏斗审计命令：

```bash
python experiments/sd_retrospective_rule_induction/audit_closed_loop_rule_funnel.py
python -m py_compile experiments/sd_retrospective_rule_induction/audit_closed_loop_rule_funnel.py
```

规则漏斗审计输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_rule_funnel_audit_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v7_closed_loop_10k5_rule_funnel_audit_20260506.md
```

周期候选层补写既有 `10k x 5` 结果时，只重放规则生成和规则库更新段，不重新训练 TD-MPC2、不重新拟合 SD。复核命令为：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/analyze_substate_rules.py experiments/sd_retrospective_rule_induction/build_rule_library.py experiments/sd_retrospective_rule_induction/update_rule_library_status.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/audit_closed_loop_rule_funnel.py
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --cycles 1 --cycle-steps 1000 --exp-prefix drycheck_v7_candidates --dry-run
```

补写后最终闭环规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v7_10k5_seed1_20260505/cycle_005/rules/library_closed_loop_cycle_005.json
```

其中包含初始 `offline_v5` 规则 `5` 条、周期候选证据 `5` 条。可上线 `replay_validated` 规则仍为 `1` 条。

第七点一版推荐 pilot 命令：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 2 --seed 1 --cycles 10 --cycle-steps 5000 --summary-window-cycles 3 --trajectory-score-mode stage --rule-prior-allowed-statuses replay_validated,rolling_validated --exp-prefix tdmpc2_v71_closed_loop_5k10_seed1_20260506 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v71_5k10_seed1_20260506 --eval-freq 5000 --checkpoint-freq 5000 --eval-episodes 3 --seed-pretrain-updates 500 --num-samples 64 --num-elites 8 --num-pi-trajs 8 --iterations 3 --rule-prior-mode penalty_only
```

第七点一版静态验证：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/stratify_trajectories.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --seed 1 --cycles 2 --cycle-steps 5000 --summary-window-cycles 3 --trajectory-score-mode stage --rule-prior-allowed-statuses replay_validated,rolling_validated --exp-prefix drycheck_v71 --dry-run
```

第七点一版 `5k x 10` pilot 运行命令：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 2 --seed 1 --cycles 10 --cycle-steps 5000 --align-resumed-eval --summary-window-cycles 3 --trajectory-score-mode stage --rule-prior-allowed-statuses replay_validated,rolling_validated --exp-prefix tdmpc2_v71a_closed_loop_5k10_seed1_20260506 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506 --eval-freq 5000 --checkpoint-freq 5000 --eval-episodes 3 --seed-pretrain-updates 500 --num-samples 64 --num-elites 8 --num-pi-trajs 8 --iterations 3 --rule-prior-mode penalty_only
```

主要输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506
results/sd_retrospective_rule_induction/reports/hopper_v71a_closed_loop_5k10_seed1_20260506.md
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle001_target5000
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle002_target10001
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle003_target15002
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle004_target20003
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle005_target25004
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle006_target30005
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle007_target35006
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle008_target40007
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle009_target45008
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle010_target50009
```

最终输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_010/rules/library_closed_loop_cycle_010.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_010/sd_explainers/sd_explainer_cycle_010.pt
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle010_target50009/models/final.full.pt
```

第七点一版 `100k` 接续训练命令：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 2 --seed 1 --start-cycle 11 --cycles 20 --cycle-steps 5000 --align-resumed-eval --summary-window-cycles 3 --trajectory-score-mode stage --rule-prior-allowed-statuses replay_validated,rolling_validated --exp-prefix tdmpc2_v71a_closed_loop_5k10_seed1_20260506 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506 --initial-rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_010/rules/library_closed_loop_cycle_010.json --initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_010/sd_explainers/sd_explainer_cycle_010.pt --resume-from logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle010_target50009/models/final.full.pt --eval-freq 5000 --checkpoint-freq 5000 --eval-episodes 3 --seed-pretrain-updates 500 --num-samples 64 --num-elites 8 --num-pi-trajs 8 --iterations 3 --rule-prior-mode penalty_only
```

接续训练新增的 runner 参数：

```text
--start-cycle 11
--resume-from logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle010_target50009/models/final.full.pt
```

静态验证命令：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
```

最终输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/rules/library_closed_loop_cycle_020.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_closed_loop_100k_continuation_seed1_20260506.md
```

## 第七点一版 100k 显式规则 eval 对照

不重训练的 checkpoint eval 入口：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py
```

`rule_off` 命令：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw /home/anaconda3/envs/sdmcts_p100/bin/python -u experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state checkpoint=logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt seed=1 exp_name=tdmpc2_v71a_100k_eval_rule_off_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_eval_traces=true eval_episodes=3 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 +rule_prior_enabled=false +rule_prior_library=results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/rules/library_closed_loop_cycle_020.json +rule_prior_mode=penalty_only "+rule_prior_allowed_statuses='replay_validated,rolling_validated'" +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_rule_off_seed1_20260506.json
```

`rule_on` 命令：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw /home/anaconda3/envs/sdmcts_p100/bin/python -u experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state checkpoint=logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt seed=1 exp_name=tdmpc2_v71a_100k_eval_rule_on_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_eval_traces=true eval_episodes=3 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 +rule_prior_enabled=true +rule_prior_library=results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/rules/library_closed_loop_cycle_020.json +rule_prior_mode=penalty_only +rule_prior_opposite_penalty=0.02 +rule_prior_preferred_cap=0.70 +rule_prior_cap_penalty=0.02 "+rule_prior_allowed_statuses='replay_validated,rolling_validated'" +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_rule_on_seed1_20260506.json
```

主要输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_rule_off_seed1_20260506/official_eval/step_0100020/eval_episode_summary.csv
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_rule_on_seed1_20260506/official_eval/step_0100020/eval_episode_summary.csv
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_rule_off_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_rule_on_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_explicit_rule_eval_seed1_20260506.md
```

本地结果：两组 eval 的 `episode_reward_list` 均为 `0.000000;0.000000;0.000000`。`rule_on` 成功安装 `8` 条规则，但当前实现中 `rotation_risk` 事件不会实际命中。

## 规则信心重放入口

静态验证：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/replay_rule_confidence.py experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
```

基于 `cycle001-020` 的信心重放命令：

```bash
python experiments/sd_retrospective_rule_induction/replay_rule_confidence.py --root results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506 --start-cycle 1 --end-cycle 20 --output results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json
```

主要输出：

```text
results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json
results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.md
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_confidence_replay_seed1_20260506.md
```

本地结果：

```text
rules = 19
confidence_active = 5
confidence_observing = 6
confidence_retired = 8
```

在线 prior 兼容性改动：

```text
online_rule_prior_v6.py 读取 effective_prior_scale / confidence；旧规则库缺失该字段时默认 1.0。
```

Confidence active 不重训练 eval 命令：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw /home/anaconda3/envs/sdmcts_p100/bin/python -u experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state checkpoint=logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt seed=1 exp_name=tdmpc2_v71a_100k_eval_conf_active_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_eval_traces=true eval_episodes=3 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 +rule_prior_enabled=true +rule_prior_library=results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json +rule_prior_mode=penalty_only +rule_prior_opposite_penalty=0.02 +rule_prior_preferred_cap=0.70 +rule_prior_cap_penalty=0.02 "+rule_prior_allowed_statuses='confidence_active'" +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_conf_active_seed1_20260506.json
```

输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_conf_active_seed1_20260506/official_eval/step_0100020/eval_episode_summary.csv
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_conf_active_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_confidence_eval_seed1_20260506.md
```

本地结果：`episode_reward_list=0.000000;0.000000;0.000000`，`rule_prior_num_rules=5`。

## Confidence active 多事件 eval

多事件匹配静态验证：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py
```

不重训练 eval 命令：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw /home/anaconda3/envs/sdmcts_p100/bin/python -u experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state checkpoint=logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt seed=1 exp_name=tdmpc2_v71a_100k_eval_conf_active_multievent_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_eval_traces=true eval_episodes=3 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 +rule_prior_enabled=true +rule_prior_library=results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json +rule_prior_mode=penalty_only +rule_prior_opposite_penalty=0.02 +rule_prior_preferred_cap=0.70 +rule_prior_cap_penalty=0.02 "+rule_prior_allowed_statuses='confidence_active'" +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_conf_active_multievent_seed1_20260506.json
```

输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_100k_eval_conf_active_multievent_seed1_20260506/official_eval/step_0100020/eval_episode_summary.csv
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_eval_conf_active_multievent_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_100k_confidence_multievent_eval_seed1_20260506.md
```

本地结果：`episode_reward_list=0.000000;0.000000;0.000000`。动作审计显示 `rotation_risk` 规则已实际命中 `12` 次动作检查。

## 150k confidence prior 接续与 eval 诊断

接续训练关键输入：

```text
resume_from = logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
initial_rule_library = results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json
```

接续训练关键参数：

```text
seed = 1
gpu_id = 2
start_cycle = 21
cycles = 30
cycle_steps = 5000
align_resumed_eval = true
summary_window_cycles = 3
trajectory_score_mode = stage
world_model_kind = tdmpc2
sd_online_supervision_enabled = false
rule_prior_allowed_statuses = confidence_active
rule_prior_mode = penalty_only
eval_episodes = 3
```

最终输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_conf_closed_loop_5k_seed1_20260506_cycle030_target150029/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_030/sd_explainers/sd_explainer_cycle_030.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_030/rules/library_closed_loop_cycle_030.json
```

不训练 eval 诊断命令需使用训练一致的环境设置：

```text
MUJOCO_GL=glfw
CUDA_VISIBLE_DEVICES=2
PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl
```

150k rule-on 9-episode eval 输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_150k_diag_eval_mode_seed1_20260506/
results/sd_retrospective_rule_induction/reports/hopper_v71a_150k_diag_eval_mode_seed1_20260506.json
```

本地结果：

```text
episode_reward = 64.03283167978871
episode_reward_list = 0.000000;0.000000;0.000000;141.175848;145.752106;0.000000;41.827946;94.016391;153.523193
```

150k train-like 9-episode eval 输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_150k_diag_trainlike_mode_seed1_20260506/
results/sd_retrospective_rule_induction/reports/hopper_v71a_150k_diag_trainlike_mode_seed1_20260506.json
```

本地结果：

```text
episode_reward = 63.692446033204355
episode_reward_list = 0.000000;0.000000;46.197573;151.813953;153.348914;0.000000;41.987734;38.186667;141.697173
```

150k rule-off 9-episode eval 输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_150k_diag_rule_off_eval9_seed1_20260506/
results/sd_retrospective_rule_induction/reports/hopper_v71a_150k_diag_rule_off_eval9_seed1_20260506.json
```

本地结果：

```text
episode_reward = 88.29123075887684
episode_reward_list = 104.431589;0.000000;128.110380;158.912107;146.062089;0.000000;0.000000;118.723386;138.381525
```

## 100k 至 150k 10-episode rule on/off sweep

入口脚本：

```text
experiments/sd_retrospective_rule_induction/eval_hopper_v71a_checkpoint_sweep.py
```

静态检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/eval_hopper_v71a_checkpoint_sweep.py experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py
```

运行命令：

```bash
MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl /home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/eval_hopper_v71a_checkpoint_sweep.py --gpu-id 2 --eval-episodes 10
```

评估范围：

```text
cycle020: logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt
cycle021-030: logs/hopper-hop/1/tdmpc2_v71a_conf_closed_loop_5k_seed1_20260506_cycle{cycle}_target{target}/models/final.full.pt
```

规则库：

```text
cycle020: results/sd_retrospective_rule_induction/rules/hopper_v71a_100k_confidence_replay.json
cycle021-030: results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_{cycle}/rules/library_closed_loop_cycle_{cycle}.json
```

输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506/
results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506.md
```

本地汇总结果：

```text
cycle020: off=1.077,  on=1.525,  on-off=+0.448
cycle021: off=1.057,  on=1.057,  on-off=+0.000
cycle022: off=19.431, on=19.046, on-off=-0.385
cycle023: off=21.556, on=22.502, on-off=+0.946
cycle024: off=3.183,  on=3.866,  on-off=+0.683
cycle025: off=30.370, on=31.311, on-off=+0.941
cycle026: off=34.785, on=34.157, on-off=-0.629
cycle027: off=34.806, on=35.172, on-off=+0.367
cycle028: off=38.305, on=37.268, on-off=-1.037
cycle029: off=50.888, on=70.566, on-off=+19.678
cycle030: off=82.414, on=57.692, on-off=-24.723
```

## Rule keyness audit for cycle029 and cycle030

入口脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py
```

静态检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py
```

运行命令：

```bash
python experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py
```

输入：

```text
logs/hopper-hop/1/tdmpc2_v71a_10ep_sweep_cycle029_step145028_rule_off_seed1_20260506/official_eval/step_0145029/
logs/hopper-hop/1/tdmpc2_v71a_10ep_sweep_cycle029_step145028_rule_on_seed1_20260506/official_eval/step_0145029/
logs/hopper-hop/1/tdmpc2_v71a_10ep_sweep_cycle030_step150029_rule_off_seed1_20260506/official_eval/step_0150030/
logs/hopper-hop/1/tdmpc2_v71a_10ep_sweep_cycle030_step150029_rule_on_seed1_20260506/official_eval/step_0150030/
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_030/rules/library_closed_loop_cycle_030.json
```

输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_rule_keyness_audit_cycle029_030_seed1_20260506.md
results/sd_retrospective_rule_induction/reports/hopper_v71a_rule_keyness_audit_cycle029_030_seed1_20260506.json
```

本地结果摘要：

```text
active_rules = 5
unique_active_substates = 3
matched_steps = 3333 / 20000
matched_frac = 0.167
high_reward_ge_100_rule_covered_frac = 0.279
low_reward_le_1_rule_covered_frac = 0.063
high_reward_ge_100_rear_frac = 0.055
low_reward_le_1_rear_frac = 0.427
```

## Explainability v0 audit

入口脚本：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v71a_explainability_v0.py
```

静态检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/audit_hopper_v71a_explainability_v0.py
```

运行命令：

```bash
python experiments/sd_retrospective_rule_induction/audit_hopper_v71a_explainability_v0.py
```

输入：

```text
cycle029/cycle030 10-episode rule-on/off eval traces
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_029/
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_conf_100k_to150k_seed1_20260506/cycle_030/
```

输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.md
results/sd_retrospective_rule_induction/reports/hopper_v71a_explainability_v0_cycle029_030_seed1_20260506.json
```

本地结果摘要：

```text
num_rules_scored = 42
top_rule = unsupported_high_torque|omega_pos|vx_mid|support_none|rear->prefer:hip_pos
top_rule_expl_v0 = 0.382
best_active_rule = unsupported_high_torque|omega_neg|vx_low|support_none|prone->prefer:waist_neg
best_active_rule_expl_v0 = 0.136
active_knee_pos_rules_expl_v0 < 0
```

## Explainability-v0 eval rule smoke

规则库构造入口：

```text
experiments/sd_retrospective_rule_induction/build_hopper_v71a_explainability_rules.py
```

静态检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/build_hopper_v71a_explainability_rules.py experiments/sd_retrospective_rule_induction/eval_hopper_v71a_explainability_rules.py
```

构造命令：

```bash
python experiments/sd_retrospective_rule_induction/build_hopper_v71a_explainability_rules.py
```

规则库输出：

```text
results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json
results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.md
```

eval-only 入口：

```text
experiments/sd_retrospective_rule_induction/eval_hopper_v71a_explainability_rules.py
```

运行命令：

```bash
python experiments/sd_retrospective_rule_induction/eval_hopper_v71a_explainability_rules.py --gpu-id 2 --eval-episodes 10
```

输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506/
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506.md
```

本地结果：

```text
cycle029:
  rule_off       = 50.888
  confidence_on  = 70.566
  explain_on     = 48.565

cycle030:
  rule_off       = 82.414
  confidence_on  = 57.692
  explain_on     = 50.615
```

## Explainability-v0 short resumed training entry

入口脚本：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v71a_explainability_short_resume.py
```

静态检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v71a_explainability_short_resume.py
```

dry-run 命令：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v71a_explainability_short_resume.py --dry-run
```

默认实验配置：

```text
resume_from = logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
initial_rule_library = results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506
start_cycle = 21
cycles = 22
cycle_steps = 5000
rule_prior_allowed_statuses = explain_active
rule_required_status = explain_active
rule_prior_opposite_penalty = 0.01
rule_prior_cap_penalty = 0.01
trajectory_score_mode = stage
summary_window_cycles = 3
```

该入口用于短接续验证，不是长预算正式训练。

## Explainability-v0 short resumed training run

运行命令：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v71a_explainability_short_resume.py --gpu-id 2
```

实际闭环配置：

```text
resume_from = logs/hopper-hop/1/tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019/models/final.full.pt
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_5k10_seed1_20260506/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
initial_rule_library = results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506
cycles = 21, 22
cycle_steps = 5000
eval_episodes = 3
sd_epochs = 3
sd_device = cpu
rule_prior_allowed_statuses = explain_active
rule_required_status = explain_active
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_opposite_penalty = 0.01
rule_prior_preferred_cap = 0.70
rule_prior_cap_penalty = 0.01
```

主要输出：

```text
logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle021_target105020/
logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_021/
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/
```

周期内 quick eval：

```text
cycle021 target105020: 0.0
cycle022 target110021: 0.0
```

训练 episode 回报摘要：

```text
cycle021:
  nonzero = 3 / 10
  max = 6.935

cycle022:
  nonzero = 3 / 10
  max = 78.874
```

最终规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/rules/library_closed_loop_cycle_022.json
```

状态统计：

```text
explain_active = 12
cycle_candidate = 1
```

最终 SD checkpoint：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/sd_explainers/sd_explainer_cycle_022.pt
```

最终 TD-MPC2 full-state checkpoint：

```text
logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt
```

最终 checkpoint 的 `10` 条 episode 对照命令：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl /home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state seed=1 exp_name=tdmpc2_v71a_explain_short_final10_rule_off_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_full_state=false save_eval_traces=true eval_episodes=10 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 sd_stage_mode=full +checkpoint=logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt +eval_action_mode=eval +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_short_final_10ep_seed1_20260506/cycle022_rule_off.json +rule_prior_enabled=false
```

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl /home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state seed=1 exp_name=tdmpc2_v71a_explain_short_final10_explain_on_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_full_state=false save_eval_traces=true eval_episodes=10 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 sd_stage_mode=full +checkpoint=logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt +eval_action_mode=eval +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_short_final_10ep_seed1_20260506/cycle022_explain_on.json +rule_prior_enabled=true +rule_prior_library=results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/rules/library_closed_loop_cycle_022.json +rule_prior_allowed_statuses=explain_active +rule_prior_mode=penalty_only +rule_prior_weight=0.05 +rule_prior_opposite_penalty=0.01 +rule_prior_preferred_cap=0.70 +rule_prior_cap_penalty=0.01
```

对照输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_short_final_10ep_seed1_20260506/cycle022_rule_off.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_short_final_10ep_seed1_20260506/cycle022_explain_on.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_short_final_10ep_seed1_20260506/summary.md
```

对照结果：

```text
rule_off:
  mean = 8.822
  std = 16.000
  list = 0.000;0.000;0.000;21.066;15.689;0.000;0.000;0.000;51.463;0.000

explain_on:
  mean = 15.955
  std = 26.914
  list = 0.000;0.000;0.000;80.774;48.390;0.000;0.000;0.000;30.383;0.000
```

## Explainability-v0 中期接续观察

接续训练命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 2 --seed 1 --resume-from logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt --initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/sd_explainers/sd_explainer_cycle_022.pt --initial-rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/rules/library_closed_loop_cycle_022.json --exp-prefix tdmpc2_v71a_explain_long_5k_seed1_20260506 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506 --start-cycle 23 --cycles 40 --cycle-steps 5000 --align-resumed-eval --summary-window-cycles 3 --trajectory-score-mode stage --rule-prior-allowed-statuses explain_active --rule-required-status explain_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-preferred-cap 0.70 --rule-prior-cap-penalty 0.01 --eval-episodes 3 --eval-freq 5000 --checkpoint-freq 5000 --sd-epochs 3 --sd-max-transitions 50000 --sd-batch-size 512 --sd-device cpu --max-states-per-episode 3 --counterfactual-horizon 5 --random-per-cycle 10
```

主要产物：

```text
logs/hopper-hop/1/tdmpc2_v71a_explain_long_5k_seed1_20260506_cycle040_target200039/models/final.full.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/sd_explainers/sd_explainer_cycle_040.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/rules/library_closed_loop_cycle_040.json
```

周期摘要：

```text
cycle023 target115022: quick_eval=0.000, train_nonzero=3/10, train_mean=21.8, train_max=86.8
cycle024 target120023: quick_eval=0.000, train_nonzero=3/10, train_mean=27.6, train_max=94.2
cycle025 target125024: quick_eval=0.000, train_nonzero=3/10, train_mean=28.4, train_max=111.4
cycle026 target130025: quick_eval=0.000, train_nonzero=3/10, train_mean=39.1, train_max=137.6
cycle027 target135026: quick_eval=0.000, train_nonzero=3/10, train_mean=40.5, train_max=154.0
cycle028 target140027: quick_eval=0.000, train_nonzero=3/10, train_mean=38.5, train_max=145.4
cycle029 target145028: quick_eval=0.000, train_nonzero=3/10, train_mean=45.6, train_max=155.7
cycle030 target150029: quick_eval=0.000, train_nonzero=4/10, train_mean=58.0, train_max=155.9
cycle031 target155030: quick_eval=0.000, train_nonzero=3/10, train_mean=53.8, train_max=189.9
cycle032 target160031: quick_eval=58.571, train_nonzero=3/10, train_mean=53.0, train_max=194.9
cycle033 target165032: quick_eval=0.000, train_nonzero=4/10, train_mean=77.0, train_max=204.9
cycle034 target170033: quick_eval=0.000, train_nonzero=4/10, train_mean=70.4, train_max=189.8
cycle035 target175034: quick_eval=0.000, train_nonzero=5/10, train_mean=99.7, train_max=203.8
cycle036 target180035: quick_eval=0.000, train_nonzero=5/10, train_mean=81.9, train_max=206.7
cycle037 target185036: quick_eval=0.000, train_nonzero=4/10, train_mean=79.0, train_max=200.9
cycle038 target190037: quick_eval=74.194, train_nonzero=4/10, train_mean=88.5, train_max=229.2
cycle039 target195038: quick_eval=0.000, train_nonzero=6/10, train_mean=115.3, train_max=232.6
cycle040 target200039: quick_eval=0.000, train_nonzero=4/10, train_mean=89.7, train_max=237.4
```

最终规则库状态：

```text
explain_active = 11
rolling_validated = 3
cycle_candidate = 14
```

最终 checkpoint 的 `10` 条 episode 对照命令：

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl /home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state seed=1 exp_name=tdmpc2_v71a_explain_long_final10_rule_off_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_full_state=false save_eval_traces=true eval_episodes=10 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 sd_stage_mode=full +checkpoint=logs/hopper-hop/1/tdmpc2_v71a_explain_long_5k_seed1_20260506_cycle040_target200039/models/final.full.pt +eval_action_mode=eval +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_long_final_10ep_seed1_20260506/cycle040_rule_off.json +rule_prior_enabled=false
```

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=glfw PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl /home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py task=hopper-hop world_model_kind=tdmpc2 model_size=null obs=state seed=1 exp_name=tdmpc2_v71a_explain_long_final10_explain_on_seed1_20260506 enable_wandb=false save_video=false save_agent=false save_csv=true save_full_state=false save_eval_traces=true eval_episodes=10 compile=false batch_size=128 lr=3e-4 dropout=0.01 mlp_dim=256 num_q=3 num_samples=64 num_elites=8 num_pi_trajs=8 horizon=3 iterations=3 sd_stage_mode=full +checkpoint=logs/hopper-hop/1/tdmpc2_v71a_explain_long_5k_seed1_20260506_cycle040_target200039/models/final.full.pt +eval_action_mode=eval +eval_report=results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_long_final_10ep_seed1_20260506/cycle040_explain_on.json +rule_prior_enabled=true +rule_prior_library=results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/rules/library_closed_loop_cycle_040.json +rule_prior_allowed_statuses=explain_active +rule_prior_mode=penalty_only +rule_prior_weight=0.05 +rule_prior_opposite_penalty=0.01 +rule_prior_preferred_cap=0.70 +rule_prior_cap_penalty=0.01
```

对照输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_long_final_10ep_seed1_20260506/cycle040_rule_off.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_long_final_10ep_seed1_20260506/cycle040_explain_on.json
results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_long_final_10ep_seed1_20260506/summary.md
```

对照结果：

```text
rule_off:
  mean = 88.514
  std = 108.492
  list = 219.547;0.000;0.000;231.711;212.722;0.000;0.000;0.000;221.155;0.000

explain_on:
  mean = 89.203
  std = 109.324
  list = 223.137;0.000;0.000;232.392;214.622;0.000;0.000;0.000;221.880;0.000
```

## 三种子共享规则 100k 实验

新增与修改的脚本：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py
experiments/sd_retrospective_rule_induction/build_hopper_manifest.py
experiments/sd_retrospective_rule_induction/stratify_trajectories.py
scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

语法检查：

```bash
python -m py_compile experiments/sd_retrospective_rule_induction/export_hopper_replay_traces.py experiments/sd_retrospective_rule_induction/build_hopper_manifest.py experiments/sd_retrospective_rule_induction/stratify_trajectories.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 /home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --seeds 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3
```

配置要点：

```text
seeds = 1,2,3
gpus = 1,2,3
cycle_steps = 5000
cycles = 20
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
checkpoint_at_seed = false
world_model_kind = tdmpc2
sd_online_supervision_enabled = false
rule_prior_mode = penalty_only
rule_prior_allowed_statuses = replay_validated,rolling_validated
trajectory_score_mode = stage
summary_window_cycles = 3
```

主要输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/
logs/hopper-hop/1/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle*/
logs/hopper-hop/2/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle*/
logs/hopper-hop/3/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle*/
```

第 `1` 周期已完成的关键产物：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_001/manifests/manifest.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_001/manifests/selected_trajectories.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_001/sd_explainers/sd_explainer_cycle_001.pt
results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_001/rules/library_closed_loop_cycle_001.json
```

第 `1` 周期检查结果：

```text
indexed runs = 3
train replay episodes exported = 10 per seed
selected episodes = 19 / 30
SD transitions = 9481
counterfactual state records = 57
substate candidate rules = 6
closed-loop action = no_rule_added
```

完成状态：

```text
completed_cycles = 20
final_output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3
final_rule_library = results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_020/rules/library_closed_loop_cycle_020.json
final_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v72_multiseed_shared_100k_seed123_20260507_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
final_tdmpc2_seed1 = logs/hopper-hop/1/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle020_target100019/models/final.full.pt
final_tdmpc2_seed2 = logs/hopper-hop/2/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle020_target100019/models/final.full.pt
final_tdmpc2_seed3 = logs/hopper-hop/3/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle020_target100019/models/final.full.pt
```

正式评估文件：

```text
25k:
logs/hopper-hop/1/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle005_target25004/official_eval/step_0025004/eval_episode_summary.csv
logs/hopper-hop/2/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle005_target25004/official_eval/step_0025004/eval_episode_summary.csv
logs/hopper-hop/3/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle005_target25004/official_eval/step_0025004/eval_episode_summary.csv

50k:
logs/hopper-hop/1/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle010_target50009/official_eval/step_0050009/eval_episode_summary.csv
logs/hopper-hop/2/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle010_target50009/official_eval/step_0050009/eval_episode_summary.csv
logs/hopper-hop/3/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle010_target50009/official_eval/step_0050009/eval_episode_summary.csv

75k:
logs/hopper-hop/1/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle015_target75014/official_eval/step_0075014/eval_episode_summary.csv
logs/hopper-hop/2/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle015_target75014/official_eval/step_0075014/eval_episode_summary.csv
logs/hopper-hop/3/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle015_target75014/official_eval/step_0075014/eval_episode_summary.csv

100k:
logs/hopper-hop/1/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle020_target100019/official_eval/step_0100019/eval_episode_summary.csv
logs/hopper-hop/2/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle020_target100019/official_eval/step_0100019/eval_episode_summary.csv
logs/hopper-hop/3/tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle020_target100019/official_eval/step_0100019/eval_episode_summary.csv
```

## v8 formal 100k empty-rule run

正式空规则库闭环实验配置：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_empty_seed123_20260508
exp_prefix = tdmpc2_v8formal_100k_empty_seed123_20260508
seeds = 1,2,3
cycles = 20
cycle_steps = 5000
total_steps = 100000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = empty
sd_online_supervision_enabled = false
```

最终规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_empty_seed123_20260508/cycle_020/rules/library_closed_loop_cycle_020.json
```

最终 TD-MPC2 checkpoints：

```text
logs/hopper-hop/1/tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019/models/final.full.pt
logs/hopper-hop/2/tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019/models/final.full.pt
logs/hopper-hop/3/tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019/models/final.full.pt
```

最终标准 eval 文件：

```text
logs/hopper-hop/1/tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019/eval_full.csv
logs/hopper-hop/2/tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019/eval_full.csv
logs/hopper-hop/3/tdmpc2_v8formal_100k_empty_seed123_20260508_cycle020_target100019/eval_full.csv
```

逐步轨迹审查脚本与输出：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/audit_hopper_v8formal_100k_final.py
python experiments/sd_retrospective_rule_induction/audit_hopper_v8formal_100k_final.py

results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_empty_seed123_final_audit_20260508.json
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_empty_seed123_final_audit_20260508.md
```

## v8 formal 100k empty-rule run with transition branch

修订后新一轮正式实验命令为：

```text
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --gpu-ids 1,2,3 \
  --cycles 20 \
  --cycle-steps 5000 \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --exp-prefix tdmpc2_v8formal_100k_v8trans_empty_seed123_20260508_r2 \
  --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2 \
  --rule-required-status v8_replay_safe_candidate \
  --rule-prior-allowed-statuses v8_replay_safe_candidate \
  --rule-prior-mode penalty_only \
  --rule-prior-opposite-penalty 0.01 \
  --rule-prior-cap-penalty 0.01 \
  --max-online-rules-per-cycle 2 \
  --max-cycle-candidates 6 \
  --rolling-weaken-max-good-share 0.40 \
  --enable-v8-transition-rules
```

该实验仍从空规则库启动，第一周期不启用 rule prior。新增 `--enable-v8-transition-rules` 后，每周期会额外生成以下产物：

```text
cycle_xxx/reports/v8_success_fragments.json
cycle_xxx/reports/v8_transition_actions.json
cycle_xxx/reports/v8_transition_counterfactuals.json
cycle_xxx/rules/v8_transition_rules.json
cycle_xxx/reports/v8_transition_rule_validation.json
cycle_xxx/reports/v8_transition_replay_safety.json
cycle_xxx/rules/v8_transition_replay_safe_rules.json
```

启动状态：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2
cycle001 = completed
cycle002 = running
```

该实验后续在用户确认后中止。由于中止时主控已推进至 `cycle016` 训练阶段，完整可复查节点为：

```text
cycle005 / 25k:
  logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_100k_v8trans_empty_seed123_20260508_r2_cycle005_target25004/eval_full.csv

cycle010 / 50k:
  logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_100k_v8trans_empty_seed123_20260508_r2_cycle010_target50009/eval_full.csv

cycle015 / 75k:
  logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_100k_v8trans_empty_seed123_20260508_r2_cycle015_target75014/eval_full.csv
```

最后完整规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_015/rules/library_closed_loop_cycle_015.json
```

中止时已创建但未完成正式评估的部分周期为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_016
```

## v8 transition phase-any smoke

规则键聚合修订后，使用 `cycle015` 真实产物执行离线 smoke。临时输出目录为：

```text
/tmp/hopper_v8_transition_patch_smoke
```

执行步骤：

```text
python experiments/sd_retrospective_rule_induction/mine_hopper_v8_transition_actions.py \
  --fragments results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_015/reports/v8_success_fragments.json \
  --output /tmp/hopper_v8_transition_patch_smoke/v8_transition_actions.json \
  --min-support 5 \
  --min-seeds 2

python experiments/sd_retrospective_rule_induction/counterfactual_hopper_v8_transition_actions.py \
  --evidence /tmp/hopper_v8_transition_patch_smoke/v8_transition_actions.json \
  --output /tmp/hopper_v8_transition_patch_smoke/v8_transition_counterfactuals.json \
  --model-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_015/sd_explainers/sd_explainer_cycle_015.pt \
  --max-records-per-key 40 \
  --min-support 5 \
  --min-advantage 0.02

python experiments/sd_retrospective_rule_induction/induce_hopper_v8_transition_rules.py \
  --observational /tmp/hopper_v8_transition_patch_smoke/v8_transition_actions.json \
  --counterfactuals /tmp/hopper_v8_transition_patch_smoke/v8_transition_counterfactuals.json \
  --output /tmp/hopper_v8_transition_patch_smoke/v8_transition_rules.json \
  --min-seeds 2

python experiments/sd_retrospective_rule_induction/validate_hopper_v8_transition_rules.py \
  --rules /tmp/hopper_v8_transition_patch_smoke/v8_transition_rules.json \
  --observational /tmp/hopper_v8_transition_patch_smoke/v8_transition_actions.json \
  --counterfactuals /tmp/hopper_v8_transition_patch_smoke/v8_transition_counterfactuals.json \
  --output /tmp/hopper_v8_transition_patch_smoke/v8_transition_rule_validation.json \
  --min-support 2 \
  --min-seeds 1 \
  --min-advantage 0.02

python experiments/sd_retrospective_rule_induction/audit_hopper_v8_transition_replay_safety.py \
  --validation /tmp/hopper_v8_transition_patch_smoke/v8_transition_rule_validation.json \
  --train-replay-export results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_015/train_replay_exports/seed1.json \
  --train-replay-export results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_015/train_replay_exports/seed2.json \
  --train-replay-export results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8trans_empty_seed123_20260508_r2/cycle_015/train_replay_exports/seed3.json \
  --output /tmp/hopper_v8_transition_patch_smoke/v8_transition_replay_safety.json \
  --max-episodes-per-seed 0

python experiments/sd_retrospective_rule_induction/export_hopper_v8_replay_safe_rules.py \
  --safety /tmp/hopper_v8_transition_patch_smoke/v8_transition_replay_safety.json \
  --output /tmp/hopper_v8_transition_patch_smoke/v8_transition_replay_safe_rules.json \
  --max-false-intervention-rate 0.10
```

结果：

```text
transition action records = 133
rule-key groups = 17
observational candidates = 2
SD counterfactual records = 78
SD counterfactual candidates = 4
merged transition rules = 6
heldout_supported rules = 3
replay_safety_supported rules = 1
exported v8 replay-safe transition rules = 1
```

导出规则：

```text
transition|transition_gap|omega_pos|vx_mid|support_none|phase_any|support_risk->soft_prefer:waist_pos
```

## v8 transition phase-any 100k 正式重跑

本轮正式实验使用空规则库从头训练，训练预算和评估口径如下：

```text
date = 2026-05-08
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2
exp_prefix = tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = v8_replay_safe_candidate
rule_prior_mode = penalty_only
rule_prior_opposite_penalty = 0.01
rule_prior_cap_penalty = 0.01
max_online_rules_per_cycle = 2
max_cycle_candidates = 6
rolling_weaken_max_good_share = 0.40
enable_v8_transition_rules = true
```

启动命令采用 `setsid -f bash -c ...` 后台脱离方式。普通 `nohup ... &` 在当前执行环境中会立即退出且不产生有效日志，因此不作为本轮可复现启动方式。主控日志为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2/launcher.log
```

周期训练日志位于：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2/cycle_*/train_logs/seed*_gpu*.log
```

本轮已在 `cycle008` 启动后中止。中止时可复查的关键产物包括：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2/cycle_005/rules/library_closed_loop_cycle_005.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2/cycle_006/rules/library_closed_loop_cycle_006.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_empty_seed123_20260508_r2/cycle_007/rules/library_closed_loop_cycle_007.json
```

`cycle006` 训练日志中的规则 prior 指标可由以下只读文件复核：

```text
logs/hopper-hop/1/tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2_cycle006_target30005/train.csv
logs/hopper-hop/2/tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2_cycle006_target30005/train.csv
logs/hopper-hop/3/tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2_cycle006_target30005/train.csv
```

`cycle005` 与 `cycle006` 的动作方向对比使用 retrospective train traces：

```text
logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2_cycle005_target25004/retrospective_train_traces/step_0025005/episode_csv
logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2_cycle006_target30005/retrospective_train_traces/step_0030006/episode_csv
```

终止命令使用进程匹配和 `TERM` 信号。普通 `kill` 在当前执行环境中对该后台进程组未能生效，最终使用提升权限的 `/bin/kill -TERM` 中断残留训练进程。终止后确认：

```text
pgrep -af 'tdmpc2_v8formal_100k_v8phaseany_empty_seed123_20260508_r2|run_hopper_v7_multiseed_shared_rules.py'
# 无输出

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
# GPU1-3 显存回到空闲占用
```

## v8 transition phase-any widefunnel 100k 重跑

本轮从头实验用于检验放宽规则发现漏斗后的在线闭环表现。启动时间为 `2026-05-08`，入口为：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

实验目录和前缀为：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1
exp_prefix = tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1
```

启动命令为：

```text
setsid -f bash -c 'python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1 --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses v8_replay_safe_candidate --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --enable-v8-transition-rules > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/launcher.log 2>&1'
```

关键参数：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = v8_replay_safe_candidate
rule_prior_mode = penalty_only
rule_prior_opposite_penalty = 0.01
rule_prior_cap_penalty = 0.01
max_online_rules_per_cycle = 2
max_cycle_candidates = 12
max_states_per_episode = 10
rolling_weaken_max_good_share = 0.40
enable_v8_transition_rules = true
```

启动后确认 `cycle001` 三种子训练进程均存在，且第一周期配置为 `rule_prior_enabled=false`、`resume_from=null`、`rule_library=shared_rules/library_cycle_000_empty.json`。

该实验已完成至 `cycle020 / 100019` steps。终点主要产物如下：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/cycle_020/rules/library_closed_loop_cycle_020.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
logs/hopper-hop/1/tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019/models/final.full.pt
logs/hopper-hop/2/tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019/models/final.full.pt
logs/hopper-hop/3/tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019/models/final.full.pt
```

标准 `10ep` official eval 文件为：

```text
logs/hopper-hop/1/tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019/official_eval/step_0100019/eval_episode_summary.csv
logs/hopper-hop/2/tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019/official_eval/step_0100019/eval_episode_summary.csv
logs/hopper-hop/3/tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019/official_eval/step_0100019/eval_episode_summary.csv
```

终点轨迹审查命令为：

```text
python experiments/sd_retrospective_rule_induction/audit_hopper_v8formal_100k_final.py --run-name tdmpc2_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1_cycle020_target100019 --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/cycle_020/rules/library_closed_loop_cycle_020.json --output results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_final_audit_20260509.json
```

输出文件为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_final_audit_20260509.json
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_final_audit_20260509.md
```

规则作用审查脚本为：

```text
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

由于 `cycle020` 训练与 final eval 实际加载的是 `cycle019` 规则库，规则作用审查命令为：

```text
python experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_widefunnel_empty_seed123_20260508_r1/cycle_019/rules/library_closed_loop_cycle_019.json --output results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_cycle019_active_20260509.json
```

输出文件为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_cycle019_active_20260509.json
results/sd_retrospective_rule_induction/reports/hopper_v8formal_100k_v8phaseany_widefunnel_seed123_rule_effects_cycle019_active_20260509.md
```

rule-effect confidence 更新脚本为：

```text
experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
```

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
python experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
```

输出文件为：

```text
results/sd_retrospective_rule_induction/rules/hopper_v8phaseany_widefunnel_cycle019_effect_confidence_20260509.json
results/sd_retrospective_rule_induction/rules/hopper_v8phaseany_widefunnel_cycle019_effect_confidence_20260509.md
```

confidence 规则库 eval-only 对比脚本为：

```text
experiments/sd_retrospective_rule_induction/eval_hopper_v8_widefunnel_confidence.py
```

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/eval_hopper_v8_widefunnel_confidence.py
python experiments/sd_retrospective_rule_induction/eval_hopper_v8_widefunnel_confidence.py --dry-run
```

正式执行命令为：

```text
python experiments/sd_retrospective_rule_induction/eval_hopper_v8_widefunnel_confidence.py
```

输出文件为：

```text
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509.md
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed1_rule_off.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed1_original_cycle019.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed1_confidence_active.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed2_rule_off.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed2_original_cycle019.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed2_confidence_active.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed3_rule_off.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed3_original_cycle019.json
results/sd_retrospective_rule_induction/reports/hopper_v8wide_confidence_eval_only_100k_seed123_20260509/seed3_confidence_active.json
```

## rule-effect confidence 在线闭环 100k 重跑

首次 `r1` 启动后发现 v8 transition 规则候选携带量仍使用默认 `6`，与本轮协议中的 `12` 不一致。该运行已在早期中止，不作为正式结果。代码修正后，正式在线闭环实验从空规则库以 `r2` 启动，运行目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2
```

启动命令为：

```text
setsid -f bash -c 'python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2 --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses confidence_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --v8-max-cycle-transition-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --enable-v8-transition-rules --enable-rule-effect-confidence > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/launcher.log 2>&1'
```

启动后检查命令为：

```text
pgrep -af 'tdmpc2_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2|run_hopper_v7_multiseed_shared_rules.py'
tail -n 80 results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/launcher.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

启动确认结果为：主控进程和 seed1、seed2、seed3 三个训练进程均存在；`cycle001` 三个训练进程分别使用 GPU1、GPU2、GPU3；第一周期 `rule_prior_enabled=false`，并加载空规则库：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/shared_rules/library_cycle_000_empty.json
```

本轮每周期 confidence 规则库预期输出位置为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_*/rules/library_closed_loop_cycle_*_confidence.json
```

## rule-effect confidence 100k 检查点规则开关诊断

为区分 100k 低分来自显式 eval-time 规则先验，还是训练后检查点本身已经偏弱，按 Hopper 操作指南对最终 `100k` 检查点执行 10ep 标准评估。评估使用 `MUJOCO_GL=glfw`、`eval_episodes=10`、`eval_action_mode=eval`，并分别比较关闭规则、启用 cycle019 confidence-active 规则、启用 cycle020 confidence-active 规则三种条件。

检查点位置为：

```text
logs/hopper-hop/1/tdmpc2_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2_cycle020_target100019/models/final.full.pt
logs/hopper-hop/2/tdmpc2_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2_cycle020_target100019/models/final.full.pt
logs/hopper-hop/3/tdmpc2_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2_cycle020_target100019/models/final.full.pt
```

规则库位置为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_019/rules/library_closed_loop_cycle_019_confidence.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_020/rules/library_closed_loop_cycle_020_confidence.json
```

诊断报告与逐种子 JSON 输出为：

```text
results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/summary.md
results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/seed*_rule_off_cycle019lib.json
results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/seed*_rule_on_cycle019lib.json
results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/seed*_rule_on_cycle020lib.json
```

结果为：

```text
rule_off_cycle019lib: seed1=9.3539, seed2=0.0452, seed3=1.5603, mean=3.6531
rule_on_cycle019lib: seed1=6.0831, seed2=0.0407, seed3=1.5503, mean=2.5580
rule_on_cycle020lib: seed1=3.2839, seed2=0.0407, seed3=1.5502, mean=1.6249
```

该诊断表明，100k 低分不能仅归因于 eval-time 显式规则先验，因为关闭规则后 seed2 和 seed3 仍处于低分状态。但规则先验也并非中性：在 seed1 上，关闭规则得到 `9.3539`，启用 cycle019 规则降至 `6.0831`，启用 cycle020 最终规则库进一步降至 `3.2839`。因此，本轮结果更符合混合故障解释：部分种子的训练后检查点本身已经偏弱，而当前规则先验在仍有一定可恢复性能的 seed1 上产生了额外负向干预。

## rule-effect confidence 规则机制缺陷审查

基于上述开关诊断，进一步审查规则生成、信心更新和在线动作先验实现。审查报告为：

```text
results/sd_retrospective_rule_induction/reports/v8formal_100k_rule_mechanism_defect_audit_20260509.md
```

主要结论为：当前机制允许最近被判定为 `effect_harmful_weakened` 的规则继续保持 `confidence_active`；规则效果审计主要衡量动作方向一致性，并不直接衡量回报或阶段转移收益；同一状态下可同时上线同一关节的相反有效动作方向，在 `penalty_only` 口径下会共同惩罚正负两侧动作，使规划器偏向小动作；新规则默认信心为 `1.0`，而缺失或低命中证据衰减较慢。这些缺陷共同支持“训练期规则偏置可能已经写入 checkpoint”的解释。

已据此修订 rule-effect confidence 安全门控。涉及代码为：

```text
experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

修订后的关键默认值为：

```text
confidence_default = 0.50
confidence_active_threshold = 0.65
confidence_positive_context_threshold = 0.05
confidence_hard_harmful_threshold = 0.45
confidence_hard_harmful_min_hits = 100
confidence_hard_harmful_margin = 0.05
confidence_conflict_keep_confidence_margin = 0.15
confidence_conflict_keep_harmful_margin = 0.10
confidence_conflict_demote_margin = 0.05
```

验证命令为：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 1 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --exp-prefix dry --output-root /tmp/hopper_dry --enable-rule-effect-confidence --dry-run
```

旧 `cycle020` 离线 smoke 使用以下命令：

```text
python experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_020/rules/library_closed_loop_cycle_020.json --audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2/cycle_020/reports/rule_effect_audit.json --output results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/library_cycle020_confidence_patched_smoke.json --target-status v8_replay_safe_candidate --target-status confidence_active --target-status confidence_observing --target-status confidence_retired
```

输出文件为：

```text
results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/library_cycle020_confidence_patched_smoke.json
results/sd_retrospective_rule_induction/reports/v8formal_100k_onoff_eval_20260509/library_cycle020_confidence_patched_smoke.md
```

smoke 结果为：active 规则数由原 cycle020 confidence 库的 `31` 降为 `24`，`effect_harmful_deactivated=5`，`active_conflict_demoted=2`，active 规则中的相反有效方向冲突为 `0`。其中冲突判断采用有效方向而非原始 action family；例如 `avoid:hip_neg` 与 `soft_prefer:hip_pos` 均视为抑制负向 hip 动作。

## confidence 安全门控 100k 从头重跑

修订 confidence 安全门控后，从空规则库启动新的 `100k` 三种子闭环实验。运行目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1
```

启动命令为：

```text
setsid -f bash -c 'python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_100k_confidence_safe_seed123_20260510_r1 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1 --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses confidence_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --v8-max-cycle-transition-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --confidence-default 0.50 --confidence-active-threshold 0.65 --confidence-positive-context-threshold 0.05 --confidence-hard-harmful-threshold 0.45 --confidence-hard-harmful-min-hits 100 --confidence-hard-harmful-margin 0.05 --confidence-conflict-keep-confidence-margin 0.15 --confidence-conflict-keep-harmful-margin 0.10 --confidence-conflict-demote-margin 0.05 --enable-v8-transition-rules --enable-rule-effect-confidence > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1/launcher.log 2>&1'
```

本轮主要配置为：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = shared_rules/library_cycle_000_empty.json
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = confidence_active
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
confidence_default = 0.50
confidence_active_threshold = 0.65
confidence_positive_context_threshold = 0.05
confidence_hard_harmful_threshold = 0.45
confidence_hard_harmful_min_hits = 100
```

启动确认命令为：

```text
pgrep -af 'tdmpc2_v8formal_100k_confidence_safe_seed123_20260510_r1|run_hopper_v7_multiseed_shared_rules.py'
tail -n 80 results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_safe_seed123_20260510_r1/launcher.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

启动确认结果为：主控进程和 seed1、seed2、seed3 三个训练进程均存在；`cycle001` 使用空规则库，且三个训练进程均为 `rule_prior_enabled=false`。

### `grounded_hop` 诊断化 100k 复核线

实验名：

```text
tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1.sh
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1
```

关键配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
trajectory_score_mode = stage
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_allowed_statuses = confidence_active
rule_required_status = v8_replay_safe_candidate
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
```

## 2026-06-03 no-gate rich-init resume-to-500k run

启动脚本：

```text
scripts/launch_hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.sh
```

启动命令：

```text
setsid env PYTHONUNBUFFERED=1 bash scripts/launch_hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.sh > logs/launch/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.out 2>&1 < /dev/null &
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1
logs/launch/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1.out
```

主要配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
start_cycle = 61
cycles = 100
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_stochastic_gate = false
initial_tva_terminal_prior_policy = strong
```

接续输入：

```text
initial_rule_library = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/rules/library_closed_loop_cycle_060_confidence.json
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/sd_explainers/sd_explainer_cycle_060.pt
initial_tva_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/tva/hopper_tva_cycle_060.pt
initial_resume seed1 = logs/hopper-hop/1/tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059/models/final.full.pt
initial_resume seed2 = logs/hopper-hop/2/tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059/models/final.full.pt
initial_resume seed3 = logs/hopper-hop/3/tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059/models/final.full.pt
```

启动后检查：

```text
actual master pid = 1968796
cycle = 61
target_steps = 305060
tva_terminal_prior_policy_this_cycle = strong
resume step reported by all three seed logs = 300060
```

## 2026-05-29 rich teacher initialization extraction

输出目录：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1
```

新增脚本：

```text
experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_rich_init_library.py
```

候选生成命令使用 `teacher_core_init` seed1 的 `100k、125k、150k、175k、200k` official eval 轨迹，并使用 `cycle035` 的 SD/TVA checkpoint。核心参数为：

```text
phase_schema = simple_v1
context_mode = broad_context_v1
outcome_mode = cycle_v1
past_window = 8
future_window = 20
horizon = 5
max_records_per_context = 32
min_context_support = 20
min_good_outcome_rate = 0.65
min_joint_top3_rate = 0.50
min_joint_best_rate = 0.08
seed = 20260529
```

主要命令：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/generate_hopper_broad_context_counterfactual_rules.py \
  --trace-glob 'logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle020_target100019/official_eval/step_0100019/episode_csv/episode_*.csv' \
  --trace-glob 'logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle025_target125024/official_eval/step_0125024/episode_csv/episode_*.csv' \
  --trace-glob 'logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle030_target150029/official_eval/step_0150029/episode_csv/episode_*.csv' \
  --trace-glob 'logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle035_target175034/official_eval/step_0175034/episode_csv/episode_*.csv' \
  --trace-glob 'logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv/episode_*.csv' \
  --sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_035/sd_explainers/sd_explainer_cycle_035.pt \
  --tva-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_035/tva/hopper_tva_cycle_035.pt \
  --output results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/counterfactual_generated_teacher_rich_simple_v1_broad_context_cycle_v1.json \
  --phase-schema simple_v1 --context-mode broad_context_v1 --outcome-mode cycle_v1 \
  --past-window 8 --future-window 20 --horizon 5 --gamma 0.99 \
  --min-context-support 20 --max-records-per-context 32 \
  --min-good-outcome-rate 0.65 --max-unstable-rate 0.30 \
  --min-joint-top3-rate 0.50 --weak-joint-top3-rate 0.35 --min-joint-best-rate 0.08 \
  --max-best-family-entropy 0.95 --max-rules 120 --seed 20260529 --device cpu
```

审计与库构建产物：

```text
counterfactual_generated_teacher_rich_simple_v1_broad_context_cycle_v1.json:
  contexts = 49
  rules = 30
  decisions = {'counterfactual_rule_candidate': 30, 'rejected_no_stable_counterfactual_family': 17, 'weak_counterfactual_pattern': 2}

reliability_high_teacher_growth_chain_sdtva.json:
  statuses = {'supported': 28, 'weak': 2}

reliability_low_strict_minrepro_100k_sdtva.json:
  statuses = {'no_overlap': 12, 'not_supported': 8, 'supported': 6, 'weak': 4}

library_counterfactual_teacher_rich_init_confidence.json:
  rules = 30
  status_counts = {'confidence_active': 22, 'confidence_observing': 8}
```

触发审计命令：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json \
  --output results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/audit_rich_active_on_current300_early_25_75.json \
  --audit-status confidence_active \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle005_target25004/official_eval/step_0025004/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle005_target25004/official_eval/step_0025004/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle005_target25004/official_eval/step_0025004/episode_csv \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle010_target50009/official_eval/step_0050009/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle010_target50009/official_eval/step_0050009/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle010_target50009/official_eval/step_0050009/episode_csv \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle015_target75014/official_eval/step_0075014/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle015_target75014/official_eval/step_0075014/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1_cycle015_target75014/official_eval/step_0075014/episode_csv \
  --max-rules-md 80
```

早期失败轨迹触发审计结果：

```text
episodes = 90
episode_reward_mean = 3.006
episode_reward_max = 92.112
any_rule_selected_gate_hit_frac = 0.0494
gate_hit_reward_corr = 0.5585
```

高质量成长链路触发审计结果：

```text
results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/audit_rich_active_on_teacher_growth_chain.json
episodes = 50
episode_reward_mean = 479.566
episode_reward_max = 614.571
any_rule_selected_gate_hit_frac = 0.4320
gate_hit_reward_corr = 0.7429
```

库构建命令：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_rich_init_library.py \
  --candidates results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/counterfactual_generated_teacher_rich_simple_v1_broad_context_cycle_v1.json \
  --high-audit results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/reliability_high_teacher_growth_chain_sdtva.json \
  --low-audit results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/reliability_low_strict_minrepro_100k_sdtva.json \
  --output results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json \
  --allowed-high-status supported \
  --blocking-low-status supported \
  --min-context-support 20 --min-counterfactual-records 20 \
  --min-source-top3 0.50 --min-source-best 0.08 --min-high-top3 0.55 \
  --max-active-rules 24 --max-observing-rules 24
```

验证命令：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/build_counterfactual_teacher_rich_init_library.py \
  experiments/sd_retrospective_rule_induction/generate_hopper_broad_context_counterfactual_rules.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_counterfactual_generated_rule_reliability.py
```

## 2026-05-29 rich teacher initialization 300k run

被终止的旧实验：

```text
hopper_v8formal_300k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1
```

新实验：

```text
hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1.sh
```

启动命令：

```text
mkdir -p logs/launch results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1
setsid -f bash scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1.sh \
  > logs/launch/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1.out 2>&1
```

主要配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 60
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_seed = 20260528
rule_prior_stochastic_eval = false
```

启动检查：

```text
run_config.json:
  exp_prefix = tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_stochgate0950995_rngiso_seed123_20260529_r1
  cycles = 60
  initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
  protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1
```

主控进程和三种子 `cycle001_target5000` 子进程已启动。`cycle001` 为从零训练阶段，尚无 SD/TVA checkpoint，因此 `tva_terminal_prior_policy = off`。

## 2026-05-28 stochastic gate RNG isolation

当前随机门控实现修订涉及以下文件：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

新增配置：

```text
rule_prior_stochastic_seed = 20260528
RULE_PRIOR_STOCHASTIC_SEED = 20260528
```

在多种子闭环训练中，实际传入单个训练进程的门控种子为：

```text
rule_prior_stochastic_seed + cycle * 1000 + seed
```

该种子只用于规则随机门控的私有 CPU `torch.Generator`，不用于 TD-MPC2/MPPI 的候选动作采样、Gumbel 选择或最终动作探索噪声。

验证命令：

```text
PYTHONPATH=/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl /home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
```

另外执行了最小随机数审计：固定 `torch.manual_seed(777)` 后调用 `HopperRulePriorV6._rule_apply_multiplier` 一百次，调用前后的全局 PyTorch RNG 状态保持一致，输出为：

```text
global_rng_unchanged= True
```

## 2026-05-28 rng-isolated near-deterministic stochastic gate run

启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1.sh
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1
logs/launch/hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_rngiso_seed123_20260528_r1.out
```

主要配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_counterfactual_teacher_retain8_confidence.json
enable_rule_prior_stochastic_gate = true
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_seed = 20260528
rule_prior_stochastic_eval = false
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
```

## 2026-05-31 rich 初始化无随机门控 300k 消融

启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh
scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1
logs/launch/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.out
```

主要配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 60
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_rich_init_v1/library_counterfactual_teacher_rich_init_confidence.json
protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_allowed_statuses = confidence_active
enable_rule_prior_stochastic_gate = false
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
enable_tva_audit = true
enable_tva_shadow_update = true
enable_tva_consequence_gate = true
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
selection_seed = 20260507
```

本轮不传入以下参数：

```text
--enable-rule-prior-stochastic-gate
--rule-prior-min-apply-prob
--rule-prior-max-apply-prob
--rule-prior-confidence-prob-power
--rule-prior-stochastic-seed
```

正式后台启动命令：

```text
setsid env PYTHONUNBUFFERED=1 bash scripts/launch_hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh > logs/launch/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.out 2>&1 < /dev/null &
```

启动 PID 为 `735645`。启动后检查到 `cycle001` 三个训练进程分别运行在 GPU1、GPU2、GPU3，训练命令中 `rule_prior_stochastic_gate=false`。

`cycle001` 三种子实际训练进程中的门控种子为：

```text
seed1 = 20261529
seed2 = 20261530
seed3 = 20261531
```

状态标签口径：`grounded_hop` 作为诊断性派生特征保留，但不进入 `state_confident`，不从 `grounded_prone` 中排他扣除，也不进入 `_dominant_state_ids`。该线用于复核上一条 hoplabel 线早期退化是否来自核心状态划分扰动。

### `grounded_hop` 诊断化 100k 至 200k 接续线

实验名：

```text
tdmpc2_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1
```

启动脚本：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1.sh
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1
```

关键配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
trajectory_score_mode = stage
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_allowed_statuses = confidence_active
rule_required_status = v8_replay_safe_candidate
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
initial_tva_terminal_prior_policy = strong
```

接续输入：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1/cycle_020/rules/library_closed_loop_cycle_020_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

initial_tva_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1/cycle_020/tva/hopper_tva_cycle_020.pt

initial_resumes =
1:logs/hopper-hop/1/tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1_cycle020_target100019/models/final.full.pt
2:logs/hopper-hop/2/tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1_cycle020_target100019/models/final.full.pt
3:logs/hopper-hop/3/tdmpc2_v8formal_100k_tva_terminal_prior_gated_hopdiag_fromscratch_seed123_20260519_r1_cycle020_target100019/models/final.full.pt
```

### TVA terminal prior 接口门控从头训练脚本

为复现“TVA 进入时机由接口审计阈值决定”的闭环口径，新增启动脚本：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
```

核心差异为在原 `rule+TVA terminal prior` 从头训练协议上增加：

```text
--enable-tva-terminal-prior-gate
--tva-terminal-gate-horizons 1,3
--tva-terminal-gate-min-samples 30
--tva-terminal-gate-min-oracle-pearson 0.55
--tva-terminal-gate-min-sd-pearson 0.55
--tva-terminal-gate-min-sd-r2 0.20
--tva-terminal-gate-max-sd-rmse-ratio 0.90
```

每个周期结束后生成：

```text
cycle_xxx/reports/tva_sd_interface_audit_gate.json
cycle_xxx/reports/tva_terminal_prior_gate.json
```

下一周期 `cycle_config.json` 中的 `tva_terminal_prior_allowed_this_cycle` 记录该周期是否实际启用 TVA terminal prior。脚本检查命令：

```text
bash -n scripts/launch_hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1.sh
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/audit_hopper_tva_sd_interface.py
```

强/弱分层门控修订后，新增参数如下：

```text
--enable-tva-terminal-prior-weak-gate
--tva-terminal-weak-prior-weight 0.005
--tva-terminal-weak-prior-max-bonus 0.01
--tva-terminal-weak-gate-horizon-mode any
--tva-terminal-weak-gate-min-samples 30
--tva-terminal-weak-gate-min-oracle-pearson 0.50
--tva-terminal-weak-gate-min-sd-pearson 0.50
--tva-terminal-weak-gate-min-sd-r2 0.00
--tva-terminal-weak-gate-max-sd-rmse-ratio 1.00
```

从 `cycle017` checkpoint 接续并在 `cycle018` 初始启用弱 TVA policy 的脚本为：

```text
scripts/launch_hopper_v8formal_200k_tva_terminal_prior_weakgated_resume_cycle018_seed123_20260518_r1.sh
```

该脚本使用：

```text
--start-cycle 18
--initial-tva-terminal-prior-policy weak
--initial-rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1/cycle_017/rules/library_closed_loop_cycle_017_confidence.json
--initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1/cycle_017/sd_explainers/sd_explainer_cycle_017.pt
--initial-tva-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_fromscratch_seed123_20260518_r1/cycle_017/tva/hopper_tva_cycle_017.pt
```

### TVA consequence gate shadow-only 315k 短程采集

正式运行脚本：

```text
scripts/launch_hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.sh
```

启动命令：

```text
setsid bash -x scripts/launch_hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.sh \
  > logs/sd_retrospective_rule_induction/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3.debug.log 2>&1 < /dev/null &
```

运行目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_315k_tva_consequence_shadowonly_seed123_20260515_r3
```

核心配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
start_cycle = 61
cycles = 63
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_opposite_penalty = 0.01
rule_prior_preferred_cap = 0.70
rule_prior_cap_penalty = 0.01
rule_prior_allowed_statuses = confidence_active
enable_tva_audit = true
tva_audit_sample_mode = label_event
tva_audit_max_records = 1200
enable_tva_shadow_update = true
tva_shadow_min_effect_hit_ratio = 0.003
tva_shadow_max_best_family_entropy = 0.65
enable_tva_consequence_gate = true
tva_consequence_horizon = 5
tva_consequence_min_relation_count = 10
tva_consequence_min_outcome_margin = 0.0
```

接续输入：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/cycle_060/rules/library_closed_loop_cycle_060_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/cycle_060/sd_explainers/sd_explainer_cycle_060.pt

initial_tva_checkpoint =
results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515/hopper_tva_cycle_060.pt

initial_resumes =
1:logs/hopper-hop/1/tdmpc2_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1_cycle060_target300059/models/final.full.pt
2:logs/hopper-hop/2/tdmpc2_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1_cycle060_target300059/models/final.full.pt
3:logs/hopper-hop/3/tdmpc2_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1_cycle060_target300059/models/final.full.pt
```

说明：`r2` 运行名仅用于排查后台启动问题。正式采集结果以 `r3` 目录为准。

### 200k 至 300k 旧规则线接续实验

本轮实验使用固定启动脚本：

```text
scripts/launch_hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1.sh
```

持久运行会话：

```text
tmux session: hopper300k_oldscheme
launcher log:
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1/launcher_logs/tmux_hopper300k_oldscheme.log
```

运行环境：

```text
python = /home/anaconda3/envs/sdmcts_p100/bin/python
MUJOCO_GL = glfw
PYTHONPATH = external/tdmpc2/tdmpc2:<repo-root>
gpu_ids = 1,2,3
seeds = 1,2,3
```

运行参数：

```text
exp_prefix = tdmpc2_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1
start_cycle = 41
cycles = 60
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
```

接续输入：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/cycle_040/rules/library_closed_loop_cycle_040_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/cycle_040/sd_explainers/sd_explainer_cycle_040.pt

initial_resumes =
1:logs/hopper-hop/1/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/models/final.full.pt
2:logs/hopper-hop/2/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/models/final.full.pt
3:logs/hopper-hop/3/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/models/final.full.pt
```

规则 prior 参数：

```text
rule_prior_allowed_statuses = confidence_active
rule_required_status = v8_replay_safe_candidate
rule_prior_mode = penalty_only
rule_prior_weight = 0.05
rule_prior_opposite_penalty = 0.01
rule_prior_preferred_cap = 0.70
rule_prior_cap_penalty = 0.01
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
```

规则自学习参数：

```text
max_online_rules_per_cycle = 2
max_cycle_candidates = 12
rolling_min_cycles = 2
rolling_min_support = 6
rolling_min_advantage = 0.05
rolling_max_good_share = 0.25
rolling_weaken_max_good_share = 0.40
confidence_default = 0.56
confidence_active_threshold = 0.55
confidence_harmful_threshold = 0.45
confidence_hard_harmful_threshold = 0.50
confidence_hard_harmful_min_hits = 100
```

启动后检查：

```text
cycle041 target_steps = 205040
seed1 resumed from step = 200040
seed2 resumed from step = 200040
seed3 resumed from step = 200040
Installed v6 rule prior: rules = 59
sd_online_supervision_enabled = false
```

## 2026-05-14 阶段约束规则 eval-only 结构烟测

环境：

```text
python = /home/anaconda3/envs/sdmcts_p100/bin/python
PYTHONPATH = /home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2:/home/data/round/code/sd_world_model_rl
MUJOCO_GL = glfw
```

被测 checkpoint：

```text
logs/hopper-hop/1/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/models/final.full.pt
logs/hopper-hop/2/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/models/final.full.pt
logs/hopper-hop/3/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/models/final.full.pt
```

规则库：

```text
full rule-on:
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json

stage-only:
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_toe_recovery_stage_constraint_only.json
```

eval-only 共同参数：

```text
eval_episodes = 10
save_csv = true
save_eval_traces = true
save_video = false
save_agent = false
compile = false
batch_size = 128
lr = 3e-4
dropout = 0.01
mlp_dim = 256
num_q = 3
num_samples = 64
num_elites = 8
num_pi_trajs = 8
horizon = 3
iterations = 3
sd_stage_mode = full
rule_prior_weight = 0.05
rule_prior_opposite_penalty = 0.01
rule_prior_max_bonus = 0.10
rule_prior_mode = penalty_only
rule_prior_preferred_cap = 0.70
rule_prior_cap_penalty = 0.01
rule_prior_allowed_statuses = confidence_active
```

输出目录：

```text
full rule-on/off reports:
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_evalonly_20260514/

stage-only reports:
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_stageonly_evalonly_20260514/

stage-only run dirs:
logs/hopper-hop/1/tdmpc2_stageconstraint_evalonly_185k_seed1_rule_stage_only_20260514_r1
logs/hopper-hop/2/tdmpc2_stageconstraint_evalonly_185k_seed2_rule_stage_only_20260514_r1
logs/hopper-hop/3/tdmpc2_stageconstraint_evalonly_185k_seed3_rule_stage_only_20260514_r1
```

结构对比命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/compare_hopper_stage_constraint_evalonly.py \
  --modes off stage_only \
  --baseline-mode off \
  --compare-mode stage_only \
  --mode-run-template off=tdmpc2_stageconstraint_evalonly_185k_seed{seed}_rule_off_20260514_r4 \
  --mode-run-template stage_only=tdmpc2_stageconstraint_evalonly_185k_seed{seed}_rule_stage_only_20260514_r1 \
  --output results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_stageonly_evalonly_20260514/post_trigger_structure_compare_stage_only_vs_off.json
```

规则效果审计命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --trace-dir logs/hopper-hop/1/tdmpc2_stageconstraint_evalonly_185k_seed1_rule_stage_only_20260514_r1/official_eval/step_0185037/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_stageconstraint_evalonly_185k_seed2_rule_stage_only_20260514_r1/official_eval/step_0185037/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_stageconstraint_evalonly_185k_seed3_rule_stage_only_20260514_r1/official_eval/step_0185037/episode_csv \
  --rule-library results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_toe_recovery_stage_constraint_only.json \
  --output results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_stageonly_evalonly_20260514/rule_effect_audit_stage_only.json \
  --high-episode-reward 200
```

语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/compare_hopper_stage_constraint_evalonly.py
```

## 2026-05-14 约束优先评分与 SD checkpoint eval-only 复测

语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_stage_constraint_candidate_ranking.py \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

生成规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --base-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/cycle_037/rules/library_closed_loop_cycle_037_confidence.json \
  --output results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint_priority.json \
  --confidence 0.75 \
  --horizon 3 \
  --stage-prior-weight 0.04 \
  --stage-max-bonus 0.12
```

手动从上述规则库提取单规则 stage-only 版本：

```text
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_toe_recovery_stage_constraint_priority_only.json
```

候选排序审计：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_hopper_stage_constraint_candidate_ranking.py \
  --rule-library results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint_priority.json \
  --baseline-rule-library results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json \
  --rule-id toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support \
  --baseline-rule-id toe_recovery_recoverable_low_support->stage_effect:constrain_recoverable_toe_support \
  --sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/cycle_037/sd_explainers/sd_explainer_cycle_037.pt \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/retrospective_train_traces/step_0185037/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/retrospective_train_traces/step_0185037/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/retrospective_train_traces/step_0185037/episode_csv \
  --output results/sd_retrospective_rule_induction/reports/hopper_185k_cycle037_stage_constraint_priority_candidate_ranking.json
```

带 SD checkpoint 的 stage-only eval-only 关键新增参数：

```text
rule_prior_library =
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_toe_recovery_stage_constraint_priority_only.json

rule_prior_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/cycle_037/sd_explainers/sd_explainer_cycle_037.pt
```

输出目录：

```text
results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_priority_sdckpt_evalonly_20260514/

logs/hopper-hop/1/tdmpc2_stageconstraint_evalonly_185k_seed1_rule_stage_priority_sdckpt_20260514_r1
logs/hopper-hop/2/tdmpc2_stageconstraint_evalonly_185k_seed2_rule_stage_priority_sdckpt_20260514_r1
logs/hopper-hop/3/tdmpc2_stageconstraint_evalonly_185k_seed3_rule_stage_priority_sdckpt_20260514_r1
```

结构对比命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/compare_hopper_stage_constraint_evalonly.py \
  --modes off stage_priority_sdckpt \
  --baseline-mode off \
  --compare-mode stage_priority_sdckpt \
  --mode-run-template off=tdmpc2_stageconstraint_evalonly_185k_seed{seed}_rule_off_20260514_r4 \
  --mode-run-template stage_priority_sdckpt=tdmpc2_stageconstraint_evalonly_185k_seed{seed}_rule_stage_priority_sdckpt_20260514_r1 \
  --output results/sd_retrospective_rule_induction/reports/hopper_185k_stageconstraint_priority_sdckpt_evalonly_20260514/post_trigger_structure_compare_stage_priority_sdckpt_vs_off.json
```

## 2026-05-14 阶段约束式脚尖恢复规则

本次修订在以下代码文件中完成：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

旧拆分阶段实验已停止，停止时 `cycle037 / 185k` 已完成，`cycle038 / 190k` 刚启动：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1
```

新规则库生成命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --base-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/cycle_037/rules/library_closed_loop_cycle_037_confidence.json \
  --output results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json \
  --horizon 3
```

生成结果：

```text
results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json
rules = 134
```

只读审计 smoke 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library results/sd_retrospective_rule_induction/rules/hopper_185k_cycle037_confidence_plus_toe_recovery_stage_constraint.json \
  --output results/sd_retrospective_rule_induction/reports/hopper_185k_cycle037_stage_constraint_rule_effect_audit.json \
  --high-episode-reward 100.0 \
  --continue-score-min 0.6 \
  --action-threshold 0.2 \
  --audit-status cycle_candidate \
  --audit-status cycle_supported \
  --audit-status rolling_validated \
  --audit-status v8_replay_safe_candidate \
  --audit-status confidence_active \
  --audit-status confidence_observing \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/retrospective_train_traces/step_0185037/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/retrospective_train_traces/step_0185037/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1_cycle037_target185036/retrospective_train_traces/step_0185037/episode_csv
```

输出：

```text
results/sd_retrospective_rule_induction/reports/hopper_185k_cycle037_stage_constraint_rule_effect_audit.json
results/sd_retrospective_rule_induction/reports/hopper_185k_cycle037_stage_constraint_rule_effect_audit.md
```

可恢复阶段约束规则在旧 `cycle037` 行为上的只读审计结果：

```text
state_hits = 45039
stage_effective = 22366
stage_harmful = 22203
direction_consistent_rate = 0.49659184262528033
direction_harmful_rate = 0.4929727569439819
```

语法检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

## 2026-05-14：脚尖恢复拆分阶段规则

代码验证：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

旧统一规则实验已停止：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1
last_completed_cycle = cycle033
last_target_steps = 165032
last_formal_eval = cycle030 / 150029
```

基于 `cycle033` 规则库生成拆分阶段规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --base-library \
  results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1/cycle_033/rules/library_closed_loop_cycle_033_confidence.json \
  --output \
  results/sd_retrospective_rule_induction/rules/hopper_165k_cycle033_confidence_plus_toe_recovery_split_stage_effect.json
```

输出：

```text
rules = 120
active_rule = toe_recovery_recoverable_low_support->stage_effect:stabilize_recoverable_toe_support
observing_rule = toe_recovery_unstable_low_support->stage_effect:observe_unstable_toe_support
```

在 `cycle033` 三种子训练回放上执行拆分规则审计：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library \
  results/sd_retrospective_rule_induction/rules/hopper_165k_cycle033_confidence_plus_toe_recovery_split_stage_effect.json \
  --output \
  results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1/cycle_033/reports/rule_effect_audit_split_toe_recovery.json \
  --high-episode-reward 100.0 \
  --continue-score-min 0.6 \
  --action-threshold 0.2 \
  --audit-status cycle_candidate \
  --audit-status cycle_supported \
  --audit-status rolling_validated \
  --audit-status v8_replay_safe_candidate \
  --audit-status confidence_active \
  --audit-status confidence_observing \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1_cycle033_target165032/retrospective_train_traces/step_0165033/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1_cycle033_target165032/retrospective_train_traces/step_0165033/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1_cycle033_target165032/retrospective_train_traces/step_0165033/episode_csv
```

审计结果：

```text
toe_recovery_recoverable_low_support:
  state_hits = 41809
  stage_effective = 27899
  stage_harmful = 13067
  direction_consistent_rate = 0.6672965151044034
  direction_harmful_rate = 0.31254036212298786

toe_recovery_unstable_low_support:
  state_hits = 36650
  stage_effective = 12897
  stage_harmful = 23348
  direction_consistent_rate = 0.3518963165075034
  direction_harmful_rate = 0.6370532060027285
```

拆分阶段规则接续实验启动命令：

```bash
setsid -f bash -c '/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --gpu-ids 1,2,3 \
  --cycles 40 \
  --start-cycle 34 \
  --cycle-steps 5000 \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --exp-prefix tdmpc2_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1 \
  --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1 \
  --initial-rule-library results/sd_retrospective_rule_induction/rules/hopper_165k_cycle033_confidence_plus_toe_recovery_split_stage_effect.json \
  --initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1/cycle_033/sd_explainers/sd_explainer_cycle_033.pt \
  --initial-resume 1:logs/hopper-hop/1/tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1_cycle033_target165032/models/final.full.pt \
  --initial-resume 2:logs/hopper-hop/2/tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1_cycle033_target165032/models/final.full.pt \
  --initial-resume 3:logs/hopper-hop/3/tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1_cycle033_target165032/models/final.full.pt \
  --rule-required-status v8_replay_safe_candidate \
  --rule-prior-allowed-statuses confidence_active \
  --rule-prior-mode penalty_only \
  --rule-prior-opposite-penalty 0.01 \
  --rule-prior-cap-penalty 0.01 \
  --max-online-rules-per-cycle 2 \
  --max-cycle-candidates 12 \
  --v8-max-cycle-transition-candidates 12 \
  --max-states-per-episode 10 \
  --rolling-weaken-max-good-share 0.40 \
  --confidence-default 0.56 \
  --confidence-active-threshold 0.55 \
  --confidence-positive-context-threshold 0.0 \
  --confidence-harmful-threshold 0.45 \
  --confidence-hard-harmful-threshold 0.50 \
  --confidence-hard-harmful-min-hits 100 \
  --confidence-hard-harmful-margin 0.05 \
  --confidence-conflict-keep-confidence-margin 0.15 \
  --confidence-conflict-keep-harmful-margin 0.10 \
  --confidence-conflict-demote-margin 0.05 \
  --enable-v8-transition-rules \
  --enable-rule-effect-confidence \
  > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1/launcher.log 2>&1'
```

启动确认：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_splitstage_seed123_20260514_r1
cycle034_target_steps = 170033
resume_step = 165033
active_rules_loaded = 56
seed1_process = started on gpu1
seed2_process = started on gpu2
seed3_process = started on gpu3
```

## 2026-05-13 toe-recovery stagefix 接续实验

停止触发口径错误的旧实验：

```text
kill 3729092 3729094
kill 3820180
```

旧实验目录保留为诊断产物：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toelaunch_stage_seed123_20260513_r1
```

修订阶段规则生成脚本后执行编译检查：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

生成修正版 100k 起点规则库：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --base-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/rules/library_closed_loop_cycle_020_confidence.json \
  --output results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json
```

生成结果：

```text
rules = 98
injected_rule = toe_recovery_low_support->stage_effect:stabilize_toe_only_launch
confidence = 0.75
stage_prior_weight = 0.08
stage_max_bonus = 0.12
```

使用旧 `toelaunch_stage` 的 `cycle023` replay trace 执行触发 smoke：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json \
  --output results/sd_retrospective_rule_induction/reports/toe_recovery_stagefix_smoke_20260513/cycle023_replay_rule_effect_audit.json \
  --high-episode-reward 100.0 \
  --continue-score-min 0.6 \
  --action-threshold 0.2 \
  --audit-status confidence_active \
  --audit-status confidence_observing \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_toelaunch_stage_seed123_20260513_r1_cycle023_target115022/retrospective_train_traces/step_0115023/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_toelaunch_stage_seed123_20260513_r1_cycle023_target115022/retrospective_train_traces/step_0115023/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_toelaunch_stage_seed123_20260513_r1_cycle023_target115022/retrospective_train_traces/step_0115023/episode_csv
```

触发 smoke 结果：

```text
state_hits = 56949
stage_effective = 35405
stage_harmful = 20603
direction_consistent_rate = 0.6217
direction_harmful_rate = 0.3618
```

启动修正版三种子 `100k -> 200k` 接续实验。当前执行环境中普通 `nohup ... &` 启动会使训练随父会话退出，因此正式启动采用 `setsid -f bash -c ...`：

```text
setsid -f bash -c '/home/anaconda3/envs/sdmcts_p100/bin/python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 40 --start-cycle 21 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1 --initial-rule-library results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_stabilize_stage_effect.json --initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt --initial-resume 1:logs/hopper-hop/1/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt --initial-resume 2:logs/hopper-hop/2/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt --initial-resume 3:logs/hopper-hop/3/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses confidence_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --v8-max-cycle-transition-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --confidence-default 0.56 --confidence-active-threshold 0.55 --confidence-positive-context-threshold 0.0 --confidence-harmful-threshold 0.45 --confidence-hard-harmful-threshold 0.50 --confidence-hard-harmful-min-hits 100 --confidence-hard-harmful-margin 0.05 --confidence-conflict-keep-confidence-margin 0.15 --confidence-conflict-keep-harmful-margin 0.10 --confidence-conflict-demote-margin 0.05 --enable-v8-transition-rules --enable-rule-effect-confidence > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toerecovery_stagefix_seed123_20260513_r1/launcher.log 2>&1'
```

启动确认：主控进程与 `cycle021` 的 seed1、seed2、seed3 训练进程均存在，三种子均从 `tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt` 恢复。

## 2026-05-13

### 阶段效果规则 eval-only 烟测

代码检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

规则库：

```text
results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_recovery_stage_effect.json
```

尝试过的完整评估命令使用 `MUJOCO_GL=glfw`、`eval_episodes=10`、`horizon=5`、`rule_prior_allowed_statuses=confidence_active`。由于阶段效果评分额外执行候选序列短视界结构推演，完整评估耗时较高，本轮中止于部分轨迹落盘后。已落盘并用于烟测审计的轨迹目录为：

```text
logs/hopper-hop/2/tdmpc2_stage_effect_rule_eval_h5_seed2_20260513_glfw_r2/official_eval/step_0000000/episode_csv
```

审计命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_recovery_stage_effect.json \
  --trace-dir logs/hopper-hop/2/tdmpc2_stage_effect_rule_eval_h5_seed2_20260513_glfw_r2/official_eval/step_0000000/episode_csv \
  --output results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/stage_rule_on_h5_seed2_partial_audit.json \
  --audit-status confidence_active \
  --max-rules-md 20
```

输出：

```text
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/stage_rule_on_h5_seed2_partial_audit.json
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/stage_rule_on_h5_seed2_partial_audit.md
```

### toe-only launch stabilize 规则烟测

生成普通强度规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

输出：

```text
results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_launch_stabilize_stage_effect.json
```

生成强介入规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --output results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_launch_stabilize_stage_effect_strong.json \
  --stage-prior-weight 0.16 \
  --stage-max-bonus 0.25 \
  --confidence 0.90
```

普通强度 `3ep` eval 报告：

```text
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/seed2_toelaunch_stabilize_rule_on_h5_smoke3.json
logs/hopper-hop/2/tdmpc2_stage_effect_toelaunch_stabilize_eval_h5_seed2_20260513_smoke3/official_eval/step_0000000/episode_csv
```

强介入 `3ep` eval 报告：

```text
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/seed2_toelaunch_stabilize_strong_rule_on_h5_smoke3.json
logs/hopper-hop/2/tdmpc2_stage_effect_toelaunch_stabilize_strong_eval_h5_seed2_20260513_smoke3/official_eval/step_0000000/episode_csv
```

审计输出：

```text
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/toelaunch_stabilize_rule_on_h5_seed2_smoke3_audit.json
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/toelaunch_stabilize_rule_on_h5_seed2_smoke3_audit.md
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/toelaunch_stabilize_strong_rule_on_h5_seed2_smoke3_audit.json
results/sd_retrospective_rule_induction/reports/stage_effect_eval_20260513/toelaunch_stabilize_strong_rule_on_h5_seed2_smoke3_audit.md
```

### toe-only launch stabilize 100k->200k 接续实验

基于 `100k cycle020` 规则库生成无后验泄漏的初始规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --base-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/rules/library_closed_loop_cycle_020_confidence.json \
  --output results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_launch_stabilize_stage_effect.json
```

接续实验目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toelaunch_stage_seed123_20260513_r1
```

启动命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --gpu-ids 1,2,3 \
  --cycles 40 \
  --start-cycle 21 \
  --cycle-steps 5000 \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --exp-prefix tdmpc2_v8formal_200k_toelaunch_stage_seed123_20260513_r1 \
  --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_toelaunch_stage_seed123_20260513_r1 \
  --initial-rule-library results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_launch_stabilize_stage_effect.json \
  --initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt \
  --initial-resume 1:logs/hopper-hop/1/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt \
  --initial-resume 2:logs/hopper-hop/2/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt \
  --initial-resume 3:logs/hopper-hop/3/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt \
  --rule-required-status v8_replay_safe_candidate \
  --rule-prior-allowed-statuses confidence_active \
  --rule-prior-mode penalty_only \
  --rule-prior-opposite-penalty 0.01 \
  --rule-prior-cap-penalty 0.01 \
  --max-online-rules-per-cycle 2 \
  --max-cycle-candidates 12 \
  --v8-max-cycle-transition-candidates 12 \
  --max-states-per-episode 10 \
  --rolling-weaken-max-good-share 0.40 \
  --confidence-default 0.56 \
  --confidence-active-threshold 0.55 \
  --confidence-positive-context-threshold 0.0 \
  --confidence-harmful-threshold 0.45 \
  --confidence-hard-harmful-threshold 0.50 \
  --confidence-hard-harmful-min-hits 100 \
  --confidence-hard-harmful-margin 0.05 \
  --confidence-conflict-keep-confidence-margin 0.15 \
  --confidence-conflict-keep-harmful-margin 0.10 \
  --confidence-conflict-demote-margin 0.05 \
  --enable-v8-transition-rules \
  --enable-rule-effect-confidence
```

启动确认：

```text
cycle021 target_steps = 105020
resume step = 100020
rule_prior_enabled = true
installed active rules = 48
initial rule library = results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_launch_stabilize_stage_effect.json
```

## 2026-05-13

### stage_effect_prior 初版实现验证

修改文件：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

语法检查：

```text
python -m py_compile \
  experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py
```

构建阶段式规则库 smoke：

```text
python experiments/sd_retrospective_rule_induction/build_hopper_toe_recovery_stage_rule.py \
  --output /tmp/hopper_stage_rule_library.json
```

输出：

```text
wrote stage-effect rule library: /tmp/hopper_stage_rule_library.json
rules: 143
```

已同时生成默认输出：

```text
results/sd_retrospective_rule_induction/rules/hopper_cycle040_confidence_plus_toe_recovery_stage_effect.json
```

在线 prior smoke 使用单条 `stage_effect_prior` 临时规则库、CPU、固定合成观测和确定性候选动作序列。结果为：

```text
rules = 1
hit = 1.0
bonus_mean = 0.0
bonus_abs_max = 0.0008381903171539307
```

阶段式审计 smoke 使用 seed2 最终正式 eval 轨迹：

```text
python experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library /tmp/stage_only_rule_library.json \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2_cycle040_target200039/official_eval/step_0200039/episode_csv \
  --output /tmp/stage_only_rule_audit.json \
  --audit-status confidence_active \
  --max-rules-md 5
```

关键输出：

```text
num_rules = 1
selected_gate_hits = 1629
stage_effective = 650
stage_harmful = 820
direction_consistent_rate = 0.39901780233271944
direction_harmful_rate = 0.5033763044812769
```

confidence 兼容 smoke：

```text
python experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py \
  --library /tmp/stage_only_rule_library.json \
  --audit /tmp/stage_only_rule_audit.json \
  --output /tmp/stage_only_rule_confidence.json \
  --target-status confidence_active \
  --active-threshold 0.55 \
  --retired-threshold 0.2
```

输出显示该临时规则可被现有 confidence 更新读取，并因回放阶段目标有害率较高从 `confidence_active` 降为 `confidence_observing`：

```text
status = confidence_observing
confidence = 0.5
last_effect_confidence_reason = effect_harmful_deactivated
```

### 脚尖恢复规则 auditfix 接续实验

审计修补验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

旧 `cycle021` 和 `cycle022` replay trace 审计 smoke 使用：

```text
python experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --rule-library <cycle rules/library_closed_loop_cycle_XXX.json> \
  --output /tmp/toe_audit_cycleXXX.json \
  --high-episode-reward 100.0 \
  --continue-score-min 0.6 \
  --action-threshold 0.2 \
  --audit-status cycle_candidate \
  --audit-status cycle_supported \
  --audit-status rolling_validated \
  --audit-status v8_replay_safe_candidate \
  --audit-status confidence_active \
  --audit-status confidence_observing \
  --trace-dir logs/hopper-hop/1/<run>/retrospective_train_traces/<step>/episode_csv \
  --trace-dir logs/hopper-hop/2/<run>/retrospective_train_traces/<step>/episode_csv \
  --trace-dir logs/hopper-hop/3/<run>/retrospective_train_traces/<step>/episode_csv
```

重跑实验目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2
```

启动配置：

```text
exp_prefix = tdmpc2_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2
seeds = 1,2,3
gpu_ids = 1,2,3
start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = confidence_active
rule_prior_mode = penalty_only
rule_prior_opposite_penalty = 0.01
rule_prior_cap_penalty = 0.01
max_online_rules_per_cycle = 2
max_cycle_candidates = 12
v8_max_cycle_transition_candidates = 12
max_states_per_episode = 10
rolling_weaken_max_good_share = 0.40
confidence_default = 0.56
confidence_active_threshold = 0.55
confidence_positive_context_threshold = 0.0
confidence_harmful_threshold = 0.45
confidence_hard_harmful_threshold = 0.50
confidence_hard_harmful_min_hits = 100
confidence_hard_harmful_margin = 0.05
confidence_conflict_keep_confidence_margin = 0.15
confidence_conflict_keep_harmful_margin = 0.10
confidence_conflict_demote_margin = 0.05
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
```

接续输入：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

initial_resumes =
1:logs/hopper-hop/1/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
2:logs/hopper-hop/2/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
3:logs/hopper-hop/3/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
```

启动命令记录在：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_auditfix_seed123_20260513_r2/launch_command.txt
```

启动确认：主控进程和 `cycle021` 的 seed1、seed2、seed3 训练进程均存在，三个训练进程均加载 `hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json`。

## 2026-05-13 脚尖恢复规则诊断

诊断脚本：

```text
experiments/sd_retrospective_rule_induction/diagnose_hopper_toe_recovery_rule.py
```

默认 SD checkpoint：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt
```

默认轨迹来源：

```text
logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/
```

复现实验命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/diagnose_hopper_toe_recovery_rule.py
python experiments/sd_retrospective_rule_induction/diagnose_hopper_toe_recovery_rule.py
```

关键参数：

```text
top_k_traces = 8
horizon = 15
amplitudes = 0.35 0.70 1.00
min_support = 8
min_support_rate = 0.35
```

输出文件：

```text
results/sd_retrospective_rule_induction/reports/hopper_100k_toe_recovery_sd_diagnostic.json
results/sd_retrospective_rule_induction/reports/hopper_100k_toe_recovery_sd_diagnostic.md
```

本次输出摘要：

```text
diagnostic_states = 192
sd_supported = 98
sd_supported_rate = 0.510
candidate_rule = toe_recovery_low_support->delay_release_and_restore_full_foot_support
```

## 2026-05-13 脚尖恢复规则接续训练

已中止旧接续实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_candidate_seed123_20260512_r1
```

新增规则库：

```text
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json
results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_sd.md
```

规则库摘要：

```text
total_rules = 98
confidence_active = 48
source=sd_diagnostic_toe_recovery = 1
injected_rule = toe_recovery_low_support->soft_prefer:waist_pos
```

启动的新接续实验：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_seed123_20260513_r1
```

启动命令：

```text
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --gpu-ids 1,2,3 \
  --cycles 40 \
  --start-cycle 21 \
  --cycle-steps 5000 \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --exp-prefix tdmpc2_v8formal_200k_confidence_toerecovery_seed123_20260513_r1 \
  --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_toerecovery_seed123_20260513_r1 \
  --initial-rule-library results/sd_retrospective_rule_induction/rules/hopper_100k_cycle020_confidence_plus_toe_recovery_sd.json \
  --initial-sd-checkpoint results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt \
  --initial-resume 1:logs/hopper-hop/1/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt \
  --initial-resume 2:logs/hopper-hop/2/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt \
  --initial-resume 3:logs/hopper-hop/3/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt \
  --rule-required-status v8_replay_safe_candidate \
  --rule-prior-allowed-statuses confidence_active \
  --rule-prior-mode penalty_only \
  --rule-prior-opposite-penalty 0.01 \
  --rule-prior-cap-penalty 0.01 \
  --max-online-rules-per-cycle 2 \
  --max-cycle-candidates 12 \
  --v8-max-cycle-transition-candidates 12 \
  --max-states-per-episode 10 \
  --rolling-weaken-max-good-share 0.40 \
  --confidence-default 0.56 \
  --confidence-active-threshold 0.55 \
  --confidence-positive-context-threshold 0.0 \
  --confidence-harmful-threshold 0.45 \
  --confidence-hard-harmful-threshold 0.50 \
  --confidence-hard-harmful-min-hits 100 \
  --confidence-hard-harmful-margin 0.05 \
  --confidence-conflict-keep-confidence-margin 0.15 \
  --confidence-conflict-keep-harmful-margin 0.10 \
  --confidence-conflict-demote-margin 0.05 \
  --enable-v8-transition-rules \
  --enable-rule-effect-confidence
```

启动确认：

```text
cycle = 021
target_steps = 105020
seed1/seed2/seed3 resumed from step = 100020
installed rule prior rules = 48
```

### confidence candidate r3 修订前检查

`hopper_v8formal_100k_confidence_mid_seed123_20260510_r2` 已在 `cycle002` 初期中止，原因是 `cycle001` 产生的 `cycle_candidate` 未进入 rule-effect confidence 更新，导致下一周期实际安装规则数为 `0`。

修订文件：

```text
experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py
```

修订内容为：rule-effect audit 与 confidence update 增加覆盖 `cycle_candidate`、`cycle_supported`、`rolling_validated` 和 `confidence_observing`。在线训练仍通过 `--rule-prior-allowed-statuses confidence_active` 控制，只使用已转换为 active 的规则。

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/apply_rule_effect_confidence.py
```

使用 r2 `cycle001` 真实产物执行离线 smoke：

```text
audit output = /tmp/r2_cycle001_rule_effect_audit_candidate.json
confidence output = /tmp/r2_cycle001_confidence_candidate.json
rule-effect audit rules = 7
gate_hit_frac = 0.3417
confidence_active = 4
confidence_observing = 3
```

修订后已从头启动 `r3`：

```text
output_root = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3
exp_prefix = tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 20
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = confidence_active
confidence_default = 0.56
confidence_active_threshold = 0.55
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
```

启动确认结果为：主控进程和 seed1、seed2、seed3 三个训练进程均存在；`cycle001` 使用空规则库，且三个训练进程均为 `rule_prior_enabled=false`。下一检查点为 `cycle001` 总结后的 `library_closed_loop_cycle_001_confidence.json`，需确认其中是否出现非零 `confidence_active` 规则。

`cycle001` 检查结果：

```text
library_closed_loop_cycle_001_confidence.json rules = 5
confidence_active = 3
confidence_observing = 2
rule_effect_gate_hit_frac = 0.2004
```

`cycle002` 三个种子日志均显示：

```text
Installed v6 rule prior: rules=3 mode=penalty_only weight=0.05
```

因此，`r3` 从第二周期开始已经实际启用规则 prior，不再是空规则库训练。

## 200k TVA 上下文竞争后果门控评估复现入口

以下命令复现 `cycle040 / 200k` 上 TVA context role-gated shadow 的短窗口后果门控、eval-only 和行为审计。所有命令均为只读或生成新报告，不修改在线训练产物。

短窗口后果审计：

```bash
python experiments/sd_retrospective_rule_induction/audit_rule_shadow_short_window_consequences.py \
  --source-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json \
  --shadow-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_shadow.json \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_tva_context_role_gated_20260520/official_eval/step_0200040/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_tva_context_role_gated_20260520/official_eval/step_0200040/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_tva_context_role_gated_20260520/official_eval/step_0200040/episode_csv \
  --output results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_short_window_consequence_20260521.json
```

根据后果审计生成 consequence-gated shadow：

```bash
python experiments/sd_retrospective_rule_induction/apply_rule_shadow_consequence_gate.py \
  --source-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json \
  --shadow-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_shadow.json \
  --consequence-audit results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_short_window_consequence_20260521.json \
  --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_consequence_shadow.json
```

Eval-only：

```bash
python experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py \
  --seeds 1 2 3 \
  --gpus 1 1 1 \
  --modes tva_context_role_consequence \
  --eval-episodes 10 \
  --no-resume \
  --summary results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_shadow_eval_20260521.json
```

规则命中审计：

```bash
python experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  --step step_0200040 \
  --seeds 1 2 3 \
  --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_role_gated_consequence_shadow.json \
  --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_tva_context_role_consequence_20260520/official_eval/step_0200040/episode_csv \
  --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_tva_context_role_consequence_20260520/official_eval/step_0200040/episode_csv \
  --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_tva_context_role_consequence_20260520/official_eval/step_0200040/episode_csv \
  --output results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_rule_effects_20260521.json
```

姿态比较审计：

```bash
python experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py \
  --group original logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_original_20260520/official_eval/step_0200040/episode_csv \
  --group original logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_original_20260520/official_eval/step_0200040/episode_csv \
  --group original logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_original_20260520/official_eval/step_0200040/episode_csv \
  --group rule_off logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_rule_off_20260520/official_eval/step_0200040/episode_csv \
  --group rule_off logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_rule_off_20260520/official_eval/step_0200040/episode_csv \
  --group rule_off logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_rule_off_20260520/official_eval/step_0200040/episode_csv \
  --group role_gated logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_tva_context_role_gated_20260520/official_eval/step_0200040/episode_csv \
  --group role_gated logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_tva_context_role_gated_20260520/official_eval/step_0200040/episode_csv \
  --group role_gated logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_tva_context_role_gated_20260520/official_eval/step_0200040/episode_csv \
  --group role_consequence logs/hopper-hop/1/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed1_tva_context_role_consequence_20260520/official_eval/step_0200040/episode_csv \
  --group role_consequence logs/hopper-hop/2/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed2_tva_context_role_consequence_20260520/official_eval/step_0200040/episode_csv \
  --group role_consequence logs/hopper-hop/3/tdmpc2_v8formal_200k_tva_context_shadow_eval_seed3_tva_context_role_consequence_20260520/official_eval/step_0200040/episode_csv \
  --output results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_consequence_posture_comparison_20260521.json
```

关键结果为：短窗口后果门控保留 `2/7` 条 role-gated 改动；consequence-gated shadow eval-only 三种子均值为 `124.106`；规则 selected-gate 命中比例为 `0.6203`；姿态比较中 `near_good_frac=0.2203`，低于原规则库 `0.2288`。

## 200k TVA 上下文归并 shadow 复现入口

以下命令复现跨周期后果汇总 smoke 和同上下文动作族归并 shadow。二者均不修改 live confidence library。

跨周期后果汇总 smoke：

```bash
python experiments/sd_retrospective_rule_induction/summarize_rule_shadow_cross_cycle_consequences.py \
  --audit results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_short_window_consequence_20260521.json \
  --output results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_cross_cycle_consequence_smoke_20260521.json
```

当前只有一个输入 audit，因此输出中 `stable_supported_rules=0`，符合至少两个周期支持才标记稳定的默认设定。

同上下文动作族归并 shadow：

```bash
python experiments/sd_retrospective_rule_induction/build_tva_context_consolidated_shadow_library.py \
  --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_confidence.json \
  --tva-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/reports/tva_context_action_competition_audit.json \
  --effect-audit results/sd_retrospective_rule_induction/reports/hopper_200k_eval_original_rule_effect_20260520.json \
  --consequence-audit results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_role_gated_short_window_consequence_20260521.json \
  --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_tva_terminal_prior_gated_hopdiag_resume100k_seed123_20260520_r1/cycle_040/rules/library_closed_loop_cycle_040_tva_context_consolidated_shadow.json
```

关键输出：

```text
candidate groups = 13
eligible groups = 1
confidence changed = 1
active rules = 61
```

唯一实际削弱规则：

```text
unsupported_high_torque|omega_pos|vx_mid|support_none|prone->prefer:ankle_pos
confidence: 0.481741 -> 0.240870
```

consolidated shadow 的 eval-only 入口已加入 `eval_hopper_tva_context_competition_shadow_200k.py`，可用以下 dry-run 检查加载路径：

```bash
python experiments/sd_retrospective_rule_induction/eval_hopper_tva_context_competition_shadow_200k.py \
  --seeds 1 \
  --gpus 1 \
  --modes tva_context_consolidated \
  --eval-episodes 1 \
  --dry-run \
  --summary results/sd_retrospective_rule_induction/reports/hopper_200k_tva_context_consolidated_shadow_eval_dryrun_20260521.json
```

`r3` 已完整运行至 `100k`，最终输出为：

```text
final shared rule library:
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/rules/library_closed_loop_cycle_020_confidence.json

final shared SD checkpoint:
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

final TD-MPC2 checkpoints:
logs/hopper-hop/1/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
logs/hopper-hop/2/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
logs/hopper-hop/3/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
```

Formal eval 结果：

```text
25k:  seed1=0.0015632650, seed2=0.0093906116, seed3=0.4287151357
50k:  seed1=0.6520965532, seed2=0.2164936973, seed3=51.9109005879
75k:  seed1=0.9073506086, seed2=0.0498013223, seed3=53.8061851926
100k: seed1=0.7226828889, seed2=42.7913077991, seed3=70.1487782332
```

### 100k 至 200k 接续实验

接续实验目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_confidence_candidate_seed123_20260512_r1
```

启动配置：

```text
exp_prefix = tdmpc2_v8formal_200k_confidence_candidate_seed123_20260512_r1
seeds = 1,2,3
gpu_ids = 1,2,3
start_cycle = 21
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_required_status = v8_replay_safe_candidate
rule_prior_allowed_statuses = confidence_active
enable_v8_transition_rules = true
enable_rule_effect_confidence = true
```

接续输入：

```text
initial_rule_library =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/rules/library_closed_loop_cycle_020_confidence.json

initial_sd_checkpoint =
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020/sd_explainers/sd_explainer_cycle_020.pt

initial_resumes =
1:logs/hopper-hop/1/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
2:logs/hopper-hop/2/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
3:logs/hopper-hop/3/tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019/models/final.full.pt
```

启动确认：`cycle021` 三个训练日志均显示从 `step=100020` 恢复，并安装 `47` 条 active 规则。

该 `r1` 运行至 `cycle006` 后中止。中止原因是 `cycle001` 至 `cycle005` 均未形成任何 `confidence_active` 规则，`25k` 之前实际退化为纯 TD-MPC2，不再符合验证规则闭环价值的实验目的。`25k` formal eval 为：

```text
seed1 = 0.0685
seed2 = 0.0301
seed3 = 0.2159
mean  = 0.1049
```

基于 r1 `cycle005` 产物进行参数 smoke 后，启动中间门控版本 `r2`。运行目录为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_mid_seed123_20260510_r2
```

中间门控参数为：

```text
confidence_default = 0.56
confidence_active_threshold = 0.55
confidence_positive_context_threshold = 0.0
confidence_harmful_threshold = 0.45
confidence_hard_harmful_threshold = 0.50
confidence_hard_harmful_min_hits = 100
```

该口径在 r1 `cycle005` smoke 中产生 `2` 条 `confidence_active`，保留相反有效动作方向冲突仲裁和硬下线机制。

启动命令为：

```text
setsid -f bash -c 'python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --gpu-ids 1,2,3 --cycles 20 --cycle-steps 5000 --formal-eval-freq 25000 --formal-checkpoint-freq 25000 --formal-eval-episodes 10 --exp-prefix tdmpc2_v8formal_100k_confidence_mid_seed123_20260510_r2 --output-root results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_mid_seed123_20260510_r2 --rule-required-status v8_replay_safe_candidate --rule-prior-allowed-statuses confidence_active --rule-prior-mode penalty_only --rule-prior-opposite-penalty 0.01 --rule-prior-cap-penalty 0.01 --max-online-rules-per-cycle 2 --max-cycle-candidates 12 --v8-max-cycle-transition-candidates 12 --max-states-per-episode 10 --rolling-weaken-max-good-share 0.40 --confidence-default 0.56 --confidence-active-threshold 0.55 --confidence-positive-context-threshold 0.0 --confidence-harmful-threshold 0.45 --confidence-hard-harmful-threshold 0.50 --confidence-hard-harmful-min-hits 100 --confidence-hard-harmful-margin 0.05 --confidence-conflict-keep-confidence-margin 0.15 --confidence-conflict-keep-harmful-margin 0.10 --confidence-conflict-demote-margin 0.05 --enable-v8-transition-rules --enable-rule-effect-confidence > results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_100k_confidence_mid_seed123_20260510_r2/launcher.log 2>&1'
```

启动确认结果为：主控进程和 seed1、seed2、seed3 三个训练进程均存在；`cycle001` 使用空规则库，且三个训练进程均为 `rule_prior_enabled=false`。

## 2026-05-23 标签校验与代码修正记录

校验对象为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1
logs/hopper-hop/{1,2,3}/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv
```

标签校验报告为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/label_calibration_seed1_high_quality_eval.md
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/label_calibration_seed1_high_quality_eval.json
```

本次修正涉及以下脚本和模块：

```text
src/world_model/dmcontrol_hopper_sd_model.py
external/tdmpc2/tdmpc2/trainer/online_trainer.py
experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py
```

语法检查命令：

```text
python -m py_compile src/world_model/dmcontrol_hopper_sd_model.py external/tdmpc2/tdmpc2/trainer/online_trainer.py experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/audit_hopper_v71a_rule_keyness.py
```

该命令已通过。旧轨迹 sanity check 结果为：

```text
rows = 5000
raw_grounded_hop_frac = 0.199
old_summary_hop_gt_frac = 0.000
new_summary_hop_gt_frac = 0.199
raw_grounded_prone_frac = 0.562
hop_priority_phase_hop_frac = 0.199
```

因此，复现实验或后续报告若使用 2026-05-23 以前的 summary CSV，应避免直接引用其中旧逻辑生成的 `grounded_hop_frac`；应从 episode CSV 重算，或重新运行正式评估生成新 summary。

修正后只读审计命令：

```text
python experiments/sd_retrospective_rule_induction/audit_hopper_posture_comparison.py --group seed1 logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --group seed2 logs/hopper-hop/2/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --group seed3 logs/hopper-hop/3/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/posture_comparison_corrected_labels.json

python experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py --run-name tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039 --step step_0200039 --seeds 1 2 3 --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_039/rules/library_closed_loop_cycle_039_confidence.json --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/rule_effect_audit_official_eval_using_cycle039_rules_corrected_labels.json --max-rules-md 80
```

输出文件：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/posture_comparison_corrected_labels.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/posture_comparison_corrected_labels.md
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/rule_effect_audit_official_eval_using_cycle039_rules_corrected_labels.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/rule_effect_audit_official_eval_using_cycle039_rules_corrected_labels.md
```

规则质量分层审计命令：

```text
python experiments/sd_retrospective_rule_induction/audit_hopper_corrected_label_rule_quality.py --rule-effect-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/rule_effect_audit_official_eval_using_cycle039_rules_corrected_labels.json --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_039/rules/library_closed_loop_cycle_039_confidence.json --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.json
```

输出文件：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.md
```

脚本语法检查：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/audit_hopper_corrected_label_rule_quality.py
```

## 2026-05-23 信心校准随机规则门控

涉及文件：

```text
experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py
scripts/run_hopper_v6_5k_rule_prior_smoke.sh
experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

新增多种子参数：

```text
--enable-rule-prior-stochastic-gate
--rule-prior-min-apply-prob
--rule-prior-max-apply-prob
--rule-prior-confidence-prob-power
--rule-prior-stochastic-eval
```

对应环境变量：

```text
RULE_PRIOR_STOCHASTIC_GATE
RULE_PRIOR_MIN_APPLY_PROB
RULE_PRIOR_MAX_APPLY_PROB
RULE_PRIOR_CONFIDENCE_PROB_POWER
RULE_PRIOR_STOCHASTIC_EVAL
```

默认值保持关闭：

```text
RULE_PRIOR_STOCHASTIC_GATE=false
RULE_PRIOR_MIN_APPLY_PROB=0.40
RULE_PRIOR_MAX_APPLY_PROB=0.98
RULE_PRIOR_CONFIDENCE_PROB_POWER=1.0
RULE_PRIOR_STOCHASTIC_EVAL=false
```

验证命令：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py
bash -n scripts/run_hopper_v6_5k_rule_prior_smoke.sh
python experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py --help | rg -n "stochastic|apply-prob|confidence-prob"
```

## 2026-05-23 修正标签初始规则库与随机门控 smoke

构建初始规则库：

```text
python experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_initial_library.py --source-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_039/rules/library_closed_loop_cycle_039_confidence.json --quality-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.json --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.json
```

输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.md
```

随机门控 smoke：

```text
python experiments/sd_retrospective_rule_induction/smoke_hopper_stochastic_rule_gate.py --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.json --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_initial_v1.json --stride 5 --max-rows 1500 --num-samples 64 --seed 7 --min-apply-prob 0.10 --max-apply-prob 0.95 --confidence-prob-power 1.0
```

输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_initial_v1.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_initial_v1.md
```

脚本语法检查：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_initial_library.py experiments/sd_retrospective_rule_induction/smoke_hopper_stochastic_rule_gate.py
```

## 2026-05-24 修正标签冷启动初始规则库

构建 cold-start 初始规则库：

```text
python experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_coldstart_library.py --teacher-library results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json --closed-loop-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_039/rules/library_closed_loop_cycle_039_confidence.json --quality-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/corrected_label_rule_quality_audit.json --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.json
```

输出：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.json
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.md
```

语法检查：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_coldstart_library.py
```

成熟轨迹随机门控 smoke：

```text
python experiments/sd_retrospective_rule_induction/smoke_hopper_stochastic_rule_gate.py --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.json --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle040_target200039/official_eval/step_0200039/episode_csv --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_label_coldstart_initial_v2_mature.json --stride 5 --max-rows 1500 --num-samples 64 --seed 7 --min-apply-prob 0.10 --max-apply-prob 0.95 --confidence-prob-power 1.0
```

25k 早期轨迹 smoke：

```text
python experiments/sd_retrospective_rule_induction/smoke_hopper_stochastic_rule_gate.py --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.json --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle005_target25004/official_eval/step_0025004/episode_csv --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle005_target25004/official_eval/step_0025004/episode_csv --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle005_target25004/official_eval/step_0025004/episode_csv --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_label_coldstart_initial_v2_early25k.json --stride 5 --max-rows 1500 --num-samples 64 --seed 7 --min-apply-prob 0.10 --max-apply-prob 0.95 --confidence-prob-power 1.0
```

旧紧凑库 25k 对照 smoke：

```text
python experiments/sd_retrospective_rule_induction/smoke_hopper_stochastic_rule_gate.py --library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_initial_v1.json --trace-dir logs/hopper-hop/1/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle005_target25004/official_eval/step_0025004/episode_csv --trace-dir logs/hopper-hop/2/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle005_target25004/official_eval/step_0025004/episode_csv --trace-dir logs/hopper-hop/3/tdmpc2_v8formal_200k_teacher_core_init_seed123_20260522_r1_cycle005_target25004/official_eval/step_0025004/episode_csv --output results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/reports/stochastic_rule_gate_smoke_corrected_initial_v1_early25k.json --stride 5 --max-rows 1500 --num-samples 64 --seed 7 --min-apply-prob 0.10 --max-apply-prob 0.95 --confidence-prob-power 1.0
```

## 2026-05-24 cold-start 三种子重训启动

语法检查：

```text
python -m py_compile experiments/sd_retrospective_rule_induction/online_rule_prior_v6.py experiments/sd_retrospective_rule_induction/closed_loop_rule_update.py experiments/sd_retrospective_rule_induction/build_hopper_corrected_label_coldstart_library.py
bash -n scripts/launch_hopper_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1.sh
```

停止上一条使用紧凑规则库的会话：

```text
tmux kill-session -t hopper_stochgate_tva_200k_r3
```

启动新实验：

```text
tmux new-session -d -s hopper_corrected_coldstart_tva_200k_r1 'bash scripts/launch_hopper_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1.sh 2>&1 | tee logs/launch/hopper_corrected_coldstart_tva_200k_r1.log'
```

实验目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1
```

## 2026-05-24 teacher-core 最小改动重训线

停止 corrected cold-start 旧线：

```text
tmux kill-session -t hopper_corrected_coldstart_tva_200k_r1
```

语法检查：

```text
bash -n scripts/launch_hopper_v8formal_200k_teacher_core_stochgate_tva_minimal_seed123_20260524_r1.sh
```

启动命令：

```text
mkdir -p logs/launch
tmux new-session -d -s hopper_teacher_core_stochgate_tva_minimal_200k_r1 'bash scripts/launch_hopper_v8formal_200k_teacher_core_stochgate_tva_minimal_seed123_20260524_r1.sh 2>&1 | tee logs/launch/hopper_teacher_core_stochgate_tva_minimal_200k_r1.log'
```

实验目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_stochgate_tva_minimal_seed123_20260524_r1
```

关键配置：

```text
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_standard_hop_seed1_200k_teacher_core_v1/library_teacher_core_confidence.json
enable_rule_prior_stochastic_gate = true
rule_prior_min_apply_prob = 0.10
rule_prior_max_apply_prob = 0.95
rule_prior_confidence_prob_power = 1.0
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
enable_tva_audit = true
enable_tva_shadow_update = true
enable_tva_consequence_gate = true
```

## 2026-05-26 counterfactual teacher retain-8 run

新一轮三种子训练命令：

```text
bash scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1.sh
```

主要输入：

```text
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_counterfactual_teacher_retain8_confidence.json
candidate_rules = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/counterfactual_generated_teacher_simple_v1_broad_context_cycle_v1.json
high_reliability_audit = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/reliability_high_teacher_cycle035_sdtva.json
low_reliability_audit = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/reliability_low_strict_minrepro_100k_cycle040_sdtva.json
```

主要配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
rule_prior_allowed_statuses = confidence_active
protected_teacher_origins = teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
enable_tva_audit = true
enable_tva_shadow_update = true
enable_tva_consequence_gate = true
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1
logs/launch/hopper_v8formal_200k_counterfactual_teacher_retain8_seed123_20260526_r1.out
```

## 2026-05-27 near-deterministic stochastic gate run

启动脚本：

```text
scripts/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1.sh
```

输出目录：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1
logs/launch/hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_stochgate0950995_seed123_20260527_r1.out
```

主要配置：

```text
seeds = 1,2,3
gpu_ids = 1,2,3
cycles = 40
cycle_steps = 5000
formal_eval_freq = 25000
formal_checkpoint_freq = 25000
formal_eval_episodes = 10
initial_rule_library = results/sd_retrospective_rule_induction/teacher_core/hopper_teacher_transition_context_v1/library_counterfactual_teacher_retain8_confidence.json
enable_rule_prior_stochastic_gate = true
rule_prior_min_apply_prob = 0.95
rule_prior_max_apply_prob = 0.995
rule_prior_confidence_prob_power = 1.0
rule_prior_stochastic_eval = false
enable_tva_terminal_prior = true
enable_tva_terminal_prior_gate = true
enable_tva_terminal_prior_weak_gate = true
```

## 2026-06-06 resume400k strict continuation

有效运行标识：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume400k_20260606_r2
tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume400k_20260606_r2
```

主控 PID：

```text
1920918
```

启动方式：

```text
setsid bash -c 'exec /home/anaconda3/envs/sdmcts_p100/bin/python -u experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py ... > "$OUT/launcher.log" 2>&1' < /dev/null &
```

关键输入：

```text
initial_rule_library = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/rules/library_closed_loop_cycle_080_confidence.json
initial_sd_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/sd_explainers/sd_explainer_cycle_080.pt
initial_tva_checkpoint = results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/tva/hopper_tva_cycle_080.pt
initial_resume(seed1) = logs/hopper-hop/1/tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079/models/final.full.pt
initial_resume(seed2) = logs/hopper-hop/2/tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079/models/final.full.pt
initial_resume(seed3) = logs/hopper-hop/3/tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079/models/final.full.pt
```

启动后验证：

```text
seed1: resumed step=400080, episode=800, total_updates=398079, target=405080, GPU1
seed2: resumed step=400080, episode=800, total_updates=398079, target=405080, GPU2
seed3: resumed step=400080, episode=800, total_updates=398079, target=405080, GPU3
```
