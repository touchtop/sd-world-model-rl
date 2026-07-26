# SD Retrospective Rule Induction

该目录用于“基于系统动力学的反向归因与规则归纳”研究线的离线代码。第一阶段代码只读取既有 Hopper 训练产物，不修改 TD-MPC2 主训练逻辑。

## 边界

当前研究线遵循以下隔离原则：

1. 只读输入：读取 `logs/hopper-hop/`、已有 `results/hopper_*` 诊断结果和 `src/world_model/dmcontrol_hopper_sd_model.py`。
2. 独立输出：所有 manifest、归因结果、规则树和报告写入 `results/sd_retrospective_rule_induction/`。
3. 不改主线：不修改 `external/tdmpc2/`、现有 `run_hopper_v5_*` 脚本或当前 Hopper 主线配置。
4. 离线优先：第一阶段只做轨迹分层、失败归因、反事实诊断和规则验证，不把规则接入在线训练。

v6 开始允许进行在线 smoke，但仍保持独立入口和默认关闭规则开关；在线接入只使用通过回放审查的子状态规则，并且只作为 MPPI 候选动作重加权。

## 初始文件

- `configs/hopper_offline_v1.yaml`：第一阶段离线实验配置。
- `build_hopper_manifest.py`：从已有 Hopper 日志中建立只读轨迹索引。
- `run_offline_v1.py`：顺序执行第一版离线闭环。

## 第一版脚本

第一版脚本按单一职责拆分：

1. `stratify_trajectories.py`：按 return、相位事件和随机抽样划分好轨迹、差轨迹和中间轨迹。
2. `attribute_failures.py`：执行结构态事件归因和短视界候选反事实点定位。
3. `counterfactual_actions.py`：使用 `DMControlHopperSDModel` 对关键状态做短视界动作反事实比较。
4. `induce_rules.py`：将稳定归因结果归纳为候选规则树。
5. `validate_rules.py`：在 held-out seed、周期和轨迹上验证规则覆盖率与误杀率。

当前第一版尚不重新训练独立 SD 解释器，而是使用已落盘 trace CSV 中的 SD 派生结构态字段完成归因，并使用当前 `DMControlHopperSDModel` 结构模型进行短视界动作反事实比较。该处理用于先验证离线经验总结闭环，后续再加入 `fit_sd_explainer.py`。

## 第二版脚本

第二版在第一版数据索引和归因结果基础上继续增加：

1. `fit_sd_explainer.py`：从已选轨迹中抽样转移，拟合独立 SD 解释器。
2. `run_offline_v2.py`：顺序执行解释器拟合、解释器反事实、状态-动作级规则归纳和规则一致性审查。
3. `induce_state_action_rules_v2.py`：将反事实记录归纳为 `state_event -> action_family` 形式的规则。
4. `analyze_rule_consistency.py`：审查 v2 规则的跨 seed 稳定性、冲突和好轨迹误伤，生成 v3 可信候选。

第二版仍不接入 TD-MPC2 在线训练。规则只作为离线候选知识保存。

## 第六版在线 smoke

第六版新增：

1. `online_rule_prior_v6.py`：读取 v5 规则库中的 `replay_validated` 子状态规则，并实现低强度 MPPI 候选价值重加权。
2. `train_hopper_v6_rule_prior.py`：独立 TD-MPC2 训练入口，默认不安装规则先验；仅当 `rule_prior_enabled=true` 时安装运行时补丁。
3. `scripts/run_hopper_v6_5k_rule_prior_smoke.sh`：5k Hopper smoke 脚本，`RULE_PRIOR_ENABLED` 默认值为 `false`。
4. `simulate_rule_prior_variants_v61.py`：在已落盘 eval trace 上审查 `prefer` 与 `penalty_only` 干预会作用于哪些动作片段。

该版本不修改 `external/tdmpc2/`，不改写既有 `run_hopper_v5_*` 脚本，也不使用硬动作替换。

第六点一版新增 `RULE_PRIOR_MODE=penalty_only`。该模式不再正向奖励规则动作族，只惩罚相反方向或同向过饱和动作，用于把规则从动作建议降级为风险筛选。

## 运行

从仓库根目录执行：

```bash
python experiments/sd_retrospective_rule_induction/run_offline_v1.py
```

主要输出：

- `results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_manifest.json`
- `results/sd_retrospective_rule_induction/manifests/hopper_offline_v1_selected_trajectories.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_attributions.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.json`
- `results/sd_retrospective_rule_induction/attributions/hopper_offline_v1_counterfactuals.md`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v1_rules.json`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v1_rules.md`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v1_rule_validation.json`

第二版执行：

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
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v3_rule_consistency.json`
- `results/sd_retrospective_rule_induction/reports/hopper_offline_v3_rule_consistency.md`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v3_reliable_rules.json`
- `results/sd_retrospective_rule_induction/rules/hopper_offline_v3_reliable_rules.md`

第六版 smoke 对照执行：

```bash
RULE_PRIOR_ENABLED=false bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 0 sd_hopper_v6_5k_rule_prior_off_seed1 1
RULE_PRIOR_ENABLED=true bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh 0 sd_hopper_v6_5k_rule_prior_on_seed1 1
```

第七版闭环执行：

```bash
python experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py --gpu-id 0 --seed 1 --cycles 3 --cycle-steps 20000
```

`--cycle-steps` 是周期总结频率参数，可按实验预算调整。闭环中 TD-MPC2 主训练保持纯 `tdmpc2`，SD 解释器只在周期总结阶段从上一周期 checkpoint 接续训练；每个周期最多将一条通过回放审查的最显著规则并入下一周期规则库。

第七版后续加入周期候选规则层。`cycle_candidate` 和 `cycle_supported` 只作为证据累计状态写入规则库，不参与在线动作先验；在线先验仍只读取 `replay_validated`。这一层用于保存单周期弱信号，避免在跨周期累计前丢失近阈值候选。

第七点一版支持更高频的可靠闭环：`cycle_steps=5000` 负责触发总结，`summary_window_cycles` 控制总结读取最近几个周期，`trajectory_score_mode=stage` 可在 reward 全零时按阶段进展选择 near-good 对照。若需要让滚动累计规则参与在线先验，可显式设置 `rule_prior_allowed_statuses=replay_validated,rolling_validated`。

第八版设计新增成功延续与阶段衔接规则。该版本不替换现有失败恢复规则，而是补充 `continue` 和 `transition` 两类规则角色，用于覆盖局部好姿态片段和好片段后的退化窗口。设计文档位于：

- [docs/sd_retrospective_rule_induction/hopper_v8_success_extension_rule_design.md](/home/data/round/code/sd_world_model_rl/docs/sd_retrospective_rule_induction/hopper_v8_success_extension_rule_design.md)

后续实现应按“通用引擎 + Hopper 适配器”拆分。Hopper 的 `pitch`、`vx`、`support`、`falling`、`waist/hip/knee/ankle` 等仍可用于当前实验，但不应直接固化为规则闭环主流程的不可替换假设。

第一版片段挖掘脚本为：

- `mine_hopper_v8_success_segments.py`：从 Hopper trace CSV 中提取成功片段、成功中断片段和阶段衔接缺口，并检查这些片段是否已被现有 v7 规则覆盖。
- `mine_hopper_v8_transition_actions.py`：从 v8 成功片段报告中抽取 `last_success / pre_degradation / degradation_onset` 等关键状态-动作证据，输出精炼阶段键、下一步 `S_continue / S_transition` 变化和离线观测候选规则。
- `counterfactual_hopper_v8_transition_actions.py`：使用 SD 解释器对 v8 转接状态执行短视界动作反事实，按 `continue / transition / joint` 目标生成 `soft_prefer` 候选。
- `induce_hopper_v8_transition_rules.py`：合并观测层 `avoid` 候选与 SD 反事实 `soft_prefer` 候选，标记一致支持与冲突规则。
- `validate_hopper_v8_transition_rules.py`：按 episode-level held-out 子集验证 v8 转接规则，并在同一规则键、同一关节内执行冲突削弱。
- `audit_hopper_v8_transition_replay_safety.py`：将 held-out 支持规则投放到训练 replay 轨迹中，审查好姿态误伤率和实际动作方向冲突。
- `export_hopper_v8_replay_safe_rules.py`：将低误伤的 v8 转接规则导出为独立候选库，供后续跨 checkpoint eval 或短接续 smoke 使用。
- `eval_hopper_v8_safe_prior.py`：在 `200k` checkpoint 上比较 `rule_off / v7_current / v8_safe` 三种 eval-only 模式，确认 v8 safe 窄候选无明显推理期破坏性。

默认复核当前 v72 三种子 `200k` 最终评估轨迹：

```bash
python experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py --use-default-v72-200k
python experiments/sd_retrospective_rule_induction/mine_hopper_v8_transition_actions.py
python experiments/sd_retrospective_rule_induction/counterfactual_hopper_v8_transition_actions.py
python experiments/sd_retrospective_rule_induction/induce_hopper_v8_transition_rules.py
python experiments/sd_retrospective_rule_induction/validate_hopper_v8_transition_rules.py
python experiments/sd_retrospective_rule_induction/audit_hopper_v8_transition_replay_safety.py
python experiments/sd_retrospective_rule_induction/export_hopper_v8_replay_safe_rules.py
python experiments/sd_retrospective_rule_induction/eval_hopper_v8_safe_prior.py --seeds 1 2 3 --gpus 1 2 3 --eval-episodes 10
```

## Hopper 100k 正式测试

规则生成机制通过逻辑审查和 smoke test 后，正式有效性测试统一采用从头 `100k` 训练，不再以接续训练均值增长作为主要证据。协议文档位于：

- [docs/sd_retrospective_rule_induction/hopper_100k_formal_protocol.md](/home/data/round/code/sd_world_model_rl/docs/sd_retrospective_rule_induction/hopper_100k_formal_protocol.md)

主测试采用空初始规则库，不传入 `--initial-resume`、`--initial-sd-checkpoint` 或 `--initial-rule-library`。若测试预先离线审查过的规则库，应作为 seeded-rule 变体单独报告，并明确规则来源。
