# Reproducibility

## 当前 Hopper 主线环境

### 运行环境

1. Python 解释器路径：`/home/anaconda3/envs/sdmcts_p100/bin/python`
2. 主要训练入口：
   - [external/tdmpc2/tdmpc2/train.py](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/train.py)
3. 当前主要实验任务：`task=hopper-hop`

### 当前活跃 world model 口径

1. `world_model_kind=sd_hopper`
2. `sd_struct_dim=17`
3. `sd_latent_dim=512`
4. `sd_use_transition_residual=true`
5. `sd_transition_residual_scale=4.0`
6. `sd_reward_mode=learned`
7. `sd_action_prior_enabled=false`
8. `sd_height_loss_coef=0.0`
9. `sd_struct_loss_scale_mode=batch_std`
10. `sd_struct_loss_scale_floor=0.1`
11. `sd_latent_consistency_struct_weight=1.0`
12. `sd_latent_consistency_resid_weight=1.0`
13. `sd_trend_protect_min_correction=0.0`
14. `sd_struct_residual_penalty_coef=0.0`
15. `sd_detach_core_in_main_loss=true`

### 当前在线 SD 监督支路默认配置

截至 `2026-04-29`，TD-MPC2 默认配置文件 [config.yaml](/home/data/round/code/sd_world_model_rl/external/tdmpc2/tdmpc2/config.yaml) 已补入在线 `SD` 监督支路所需参数。默认值如下：

1. `sd_online_supervision_enabled=false`
2. `sd_online_supervision_lr=3e-4`
3. `sd_online_supervision_one_step_weight=1.0`
4. `sd_online_supervision_rollout_weight=1.0`
5. `sd_online_supervision_omega_weight=0.25`
6. `sd_online_supervision_contact_weight=0.25`
7. `sd_online_supervision_phase_weight=2.0`
8. `sd_struct_loss_scale_mode=batch_std`
9. `sd_struct_loss_scale_floor=0.1`
10. `sd_latent_consistency_struct_weight=1.0`
11. `sd_latent_consistency_resid_weight=1.0`

说明如下：

1. 默认仍关闭该支路，以保证旧实验脚本不会因默认行为改变而失真；
2. 一旦显式启用，共享 `SD transition core` 即 `model._dynamics` 将从 TD-MPC2 主优化器中移除，只由独立 `sd_optim` 管理；
3. `sd_detach_core_in_main_loss=true` 时，启用 online supervision 后主 consistency 路径不再训练 `SD core`，而由独立在线 `SD` 监督训练 `_dynamics`；
4. 当前最小实现依赖 replay batch 的 `horizon+1` 采样窗口，因此实际监督视界为 `1-step + 3-step`。
5. 自 `2026-04-29` 修复版本起，主 `consistency_loss` 的 SD 结构态部分与在线 SD 监督损失均使用 batch 目标结构态标准差归一化。
6. 主 `consistency_loss` 中的结构态块与 residual latent 块默认按块等权，避免 495 维残差块仅因维度数量压过 17 维显式结构态。

### 当前 residual 与趋势保护配置

截至 `2026-04-30`，`SDWorldModel.next()` 中外层结构态 residual 与趋势保护参数均已显式配置化。默认值如下：

1. `sd_struct_residual_scale=0.10`
2. `sd_vx_residual_scale=0.50`
3. `sd_vz_residual_scale=3.00`
4. `sd_pitch_residual_scale=0.75`
5. `sd_pitch_rate_residual_scale=8.00`
6. `sd_knee_residual_scale=0.35`
7. `sd_struct_residual_penalty_coef=0.0`
8. `sd_detach_core_in_main_loss=true`
9. `sd_trend_protect_max_scale=0.20`
10. `sd_trend_protect_min_correction=0.0`

说明如下：

1. residual 保留为主线设计，因为显式系统动力学结构态不被要求完整覆盖 Hopper 的所有动力机制；
2. `sd_struct_residual_penalty_coef` 默认关闭，仅作为后续 residual 覆盖风险消融入口；
3. `struct_residual_activity=mean(abs(tanh(raw_struct_residual)))` 已写入训练日志，用于 smoke 阶段判断 residual 是否过强；
4. `sd_trend_protect_min_correction=0.0` 取消此前绝对最小修正下限，避免在小位移、小接触变化阶段引入单位错配；
5. `sd_vz_residual_scale=3.00` 与 `sd_pitch_rate_residual_scale=8.00` 当前仍保持历史默认值，原因是 residual 是主线设计的一部分，默认收紧会改变实验变量。下一轮 smoke 应同时观察 `struct_consistency_loss / resid_consistency_loss`、`struct_residual_activity` 与闭环得分，再决定是否将默认值收紧到如 `vz=1.0`、`pitch_rate=3.0`。

### 下一版本 planning-aware SD 权重配置

截至 `2026-04-30`，配置文件新增以下默认关闭的 planning-aware 权重参数：

1. `sd_online_planning_weight_enabled=false`
2. `sd_online_planning_weight_mode=td_positive`
3. `sd_online_planning_weight_coef=1.0`
4. `sd_online_planning_weight_max=5.0`

默认关闭意味着既有 `pureenv_sdinit_online_sup` 实验不受影响。启用该开关后，在线 `SD` 监督损失会根据 TD 目标或奖励目标对样本加权，用于验证 planning signal 是否能缓解平均结构态监督与闭环规划目标之间的错配。权重在 batch 内按均值归一化，并用 `sd_online_planning_weight_max` 截断，以降低启用该机制后因有效学习率变化造成的混淆。

### 当前监督初始化 checkpoint

用于当前 `pure-env + sdinit` 主线的监督初始化 checkpoint 为：

- [hopper_sd_supervised.pt](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt)

该 checkpoint 的关键信息为：

1. `seed=0`
2. `episodes=12`
3. `max_steps=250`
4. `epochs=30`
5. `batch_size=256`
6. `short_rollout_weight=0.10`
7. `use_transition_residual=true`
8. `transition_residual_scale=4.0`

对应摘要文件：

- [summary.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/summary.json)

### 当前闭环实验脚本

当前主要使用以下脚本：

1. [run_hopper_v5_5k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_5k_pureenv_sdinit.sh)
2. [run_hopper_v5_100k_pureenv_sdinit.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_100k_pureenv_sdinit.sh)
3. [run_hopper_v5_200k_pureenv_sdinit_resume.sh](/home/data/round/code/sd_world_model_rl/scripts/run_hopper_v5_200k_pureenv_sdinit_resume.sh)

### 当前讨论所依据的关键实验

1. `100k` 三种子 pure-env + sdinit 主实验：
   - seed1: [sd_hopper_100k_pureenv_sdinit_seed1_20260428_run1](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/1/sd_hopper_100k_pureenv_sdinit_seed1_20260428_run1)
   - seed2: [sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/2/sd_hopper_100k_pureenv_sdinit_seed2_20260428_run1)
   - seed3: [sd_hopper_100k_pureenv_sdinit_seed3_20260428_run1](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_100k_pureenv_sdinit_seed3_20260428_run1)
2. `200k` seed3 接续实验：
   - [sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_probe](/home/data/round/code/sd_world_model_rl/logs/hopper-hop/3/sd_hopper_200k_pureenv_sdinit_resume_from10k_seed3_20260428_probe)

### 当前关键诊断文件

1. `75k replay` 误差诊断：
   - [diagnostics_20260429_seed3_resume75k_policy_error.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/diagnostics_20260429_seed3_resume75k_policy_error.json)
2. `50k replay` 误差诊断：
   - [diagnostics_20260429_seed3_resume50k_policy_error.json](/home/data/round/code/sd_world_model_rl/results/hopper_sd_supervised/diagnostics_20260429_seed3_resume50k_policy_error.json)

### 说明

当前研究结论明确依赖以下假设：

1. 评估重点是 TD-MPC2 当前实际使用的短视界（尤其 `1 / 3 / 5` 步），而不是长视界自由仿真；
2. 当前 `SD` 的用途是作为 TD-MPC2 的 transition core，而不是独立完成完整控制任务；
3. 当前闭环行为结果不应被解读为“跳跃任务已解决”，而应被视为 transition core 与 planner 相互作用的诊断样本。

## 2026-06-10 Sokoban/Boxoban 数据来源记录

为后续 Sokoban/Boxoban 扩展实验建立原始数据来源记录。数据集下载至：

- [datasets/sokoban_boxoban/boxoban-levels](/home/data/round/code/sd_world_model_rl/datasets/sokoban_boxoban/boxoban-levels)

来源仓库与版本：

1. 上游仓库：`https://github.com/google-deepmind/boxoban-levels.git`
2. 本地下载方式：`git clone --depth 1`
3. commit：`78011d8c86ad423bfb2e1005012393b4c3c24409`
4. 许可证：Apache License 2.0

本地校验摘要：

1. 数据目录大小约 `204 MB`；
2. 文本文件数为 `1506`；
3. 关卡记录总数为 `1,504,332`；
4. 每个关卡为 10x10 ASCII Sokoban/Boxoban 表示；
5. 原始数据目录通过 [datasets/sokoban_boxoban/.gitignore](/home/data/round/code/sd_world_model_rl/datasets/sokoban_boxoban/.gitignore) 排除在版本控制之外，派生子集、张量化缓存和实验输出应另存于生成数据目录。

## 2026-06-10 Sokoban SD 世界模型首轮评测

新增代码路径：

1. [src/sokoban/env.py](/home/data/round/code/sd_world_model_rl/src/sokoban/env.py)
2. [src/sokoban/sd_model.py](/home/data/round/code/sd_world_model_rl/src/sokoban/sd_model.py)
3. [src/sokoban/search.py](/home/data/round/code/sd_world_model_rl/src/sokoban/search.py)
4. [experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py)

当前运行环境中的 `gymnasium` 未安装，因此 Sokoban 新研究线避免依赖旧 `src.env` 包入口，改用独立 `src.sokoban` 包。该选择不改变既有实验的依赖声明；原 Beer Game、Cartpole、Hopper 复现仍按原要求安装依赖。

语法检查命令：

```bash
python -m compileall src/sokoban experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py
```

首轮评测命令：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_sd_model.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 50 \
  --max-expansions 2000 \
  --out results/sokoban_sd_world_model/eval_medium_train000_50.json
```

结果文件：

1. [eval_smoke_5.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/eval_smoke_5.json)
2. [eval_medium_train000_20.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/eval_medium_train000_20.json)
3. [eval_medium_train000_50.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/eval_medium_train000_50.json)
4. [literature_scan.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/literature_scan.json)

`eval_medium_train000_50.json` 摘要：

1. transition adapter consistency：`266 / 266`，通过率 `1.0`；
2. unguided uniform-cost search：`2 / 50` solved，平均扩展 `1938.18`；
3. SD-guided best-first search：`45 / 50` solved，平均扩展 `422.16`；
4. 两组搜索使用相同关卡、相同 `2000` push-level 扩展预算。

该评测自 `2026-06-10` 文献复核后降级为诊断性草稿结果。其用途是验证机制字段和搜索消费端的连通性，不作为完整 Boxoban benchmark 或最终求解器性能结论。

## 2026-06-10 Sokoban 文献矩阵与设计文档复现记录

新增文档：

1. [sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)
2. [sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md)
3. [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)
4. [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)
5. [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md)

文献 PDF 与抽取文本位于：

1. [pdfs](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature/pdfs)
2. [text](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature/text)

Botea、Muller 与 Schaeffer 2002 年 PDF 的重新抽取命令：

```bash
pdftotext -layout -enc UTF-8 \
  docs/sokoban_sd_literature/pdfs/botea_muller_schaeffer_2002_abstraction.pdf \
  /tmp/botea_layout.txt

pdftotext -raw -enc UTF-8 \
  docs/sokoban_sd_literature/pdfs/botea_muller_schaeffer_2002_abstraction.pdf \
  /tmp/botea_raw.txt
```

两种抽取结果均存在严重编码异常，因此该文献只在矩阵中作为 `B` 级证据保留。`results/sokoban_sd_world_model/literature_scan.json` 中存在明显误匹配的 arXiv 结果，不作为本轮设计证据。

## 2026-06-10 Sokoban stock-flow SD 设计与审查记录

本轮未运行新的训练或评测实验，只完成设计文档与方法论审查。设计依据为本地已下载 PDF、抽取文本、文献矩阵和当前诊断性原型代码。

新增或修订产物：

1. [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)：定义 Sokoban/Boxoban 抽象 SD 世界模型的存量、流量、反馈、学习目标和 SD-MCTS 接口；
2. [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md)：依据 academic-research-suite 的 methodology reviewer、devil's advocate reviewer 和 editorial synthesizer 角色规范完成方法论审查；
3. [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)：将旧启发式搜索路线改写为诊断草稿，并指向 stock-flow 主设计；
4. [sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md)：将替换路线改为可学习流量、反馈和 SD-MCTS；
5. [sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)：将文献机制映射为存量代理、流量输入、反馈输入、风险标签和诊断项。

复现边界：

1. 旧 `eval_medium_train000_50.json` 仍只作为结构代理与搜索消费端连通性的 smoke/diagnostic baseline；
2. 当前尚未实现可学习 stock-flow-feedback 模型；
3. 当前尚未实现 SD-MCTS over learned abstract dynamics；
4. 后续正式实验需要同时记录世界模型指标、流量守恒误差、风险校准、规划指标和启发式诊断消融。

## 2026-06-10 Sokoban SD 模型设计修订与复审记录

本轮未运行新的代码实验、训练任务或 Boxoban 评测。操作范围仅限设计文档、方法记录、决策记录、复现记录和复审文档。

修订后首版设计口径如下：

1. 核心存量从九个收缩为六个：`G_t, U_t, M_t, R_t, C_t, I_t`；
2. `GoalRoomPressureStock`、`PlanProgressStock` 和 `StructuralSlackStock` 降级为派生特征、辅助头或第二阶段扩展；
3. 状态转移统一写为 `x_{t+1}=clip_X(x_t+B f_theta(...))`，其中 `f_theta` 为非负流量，`B` 为已知 incidence matrix；
4. 核心反馈保留目标压力反馈、拥塞-机动性-风险反馈和不确定性反馈；
5. 首版动作集合限定为合法 single-push option，macro 只作为诊断代理或后续扩展；
6. SD-MCTS 节点需保存抽象状态分布、uncertainty、raw-state handle 或 belief support，并记录 rollout transition provider；
7. 风险标签分为可证明 dead-end、短 rollout failure 和 heuristic warning；
8. 正式验收增加 state aliasing、learned-transition-provider audit、no-learned-parameter、no-feedback、black-box latent MBRL 和 heuristic-search diagnostic 消融。

本轮更新的文档包括：

1. [sokoban_sd_stock_flow_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design.md)
2. [sokoban_sd_stock_flow_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_stock_flow_design_review.md)
3. [sokoban_sd_world_model.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_world_model.md)
4. [sokoban_sd_replacement_design.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_replacement_design.md)
5. [sokoban_sd_literature_matrix.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature_matrix.md)
6. [methods.md](/home/data/round/code/sd_world_model_rl/docs/methods.md)
7. [decisions.md](/home/data/round/code/sd_world_model_rl/docs/decisions.md)
8. [reproducibility.md](/home/data/round/code/sd_world_model_rl/docs/reproducibility.md)
9. [research_log.md](/home/data/round/code/sd_world_model_rl/docs/research_log.md)

## 2026-06-10 Sokoban SD 设计验证 smoke

新增验证入口：

1. [validate_sokoban_sd_design.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py)

该脚本用于验证修订后六核心存量设计的 schema、标签构造和审计字段，不训练可学习世界模型，也不运行 SD-MCTS。当前 provider 明确记录为 `exact_single_push_diagnostic_provider`，因此该结果仍属于设计验证 smoke，而不是 learned SD transition 结果。

语法检查命令：

```bash
python -m compileall src/sokoban \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py
```

验证 smoke 命令：

```bash
python experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 5 \
  --max-depth 1 \
  --max-states-per-level 20 \
  --sample-limit 5 \
  --out results/sokoban_sd_world_model/design_validation_smoke_5.json
```

结果文件：

1. [design_validation_smoke_5.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/design_validation_smoke_5.json)

结果摘要：

1. `levels_checked=5`，`levels_passed=5`；
2. `states_checked=31`，`transitions_checked=200`；
3. `range_violations=0`；
4. `nonnegative_flow_violations=0`；
5. `max_balance_l1=0.0`；
6. 风险标签计数：`none=86`，`proven_dead_end=110`，`heuristic_warning=4`；
7. transition provider：`exact_single_push_diagnostic_provider=200`；
8. `max_alias_next_delta_variance=0.4044025054643195`；
9. `design_smoke_pass=true`。

复现边界：

1. 该 smoke 只验证六核心存量、非负诊断流量、incidence matrix 和 provider/aliasing 审计字段可以生成；
2. 该 smoke 不包含学习参数、反馈训练、不确定性校准或 SD-MCTS；
3. 下一阶段若进入 learned flow training，应把本脚本输出作为标签构造和审计 schema 的最小基线。

## 2026-06-10 Sokoban learned flow smoke

新增训练入口：

1. [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py)

该脚本训练最小 learned flow 模型，学习 `x_t, a_t -> f_t`，并通过固定 incidence matrix `B` 预测下一抽象存量。当前版本为 `no-feedback`、`no-SD-MCTS` smoke，transition 标签仍来自 `exact_single_push_diagnostic_provider`。

语法检查命令：

```bash
python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

训练 smoke 命令：

```bash
python experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 10 \
  --max-depth 1 \
  --max-states-per-level 30 \
  --epochs 120 \
  --batch-size 128 \
  --hidden-dim 64 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_smoke_10.json
```

结果文件：

1. [flow_smoke_10.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_10.json)

数据与配置摘要：

1. `samples=343`；
2. `train_samples=274`，`valid_samples=69`；
3. `levels_with_samples=10`；
4. `input_dim=15`；
5. `epochs=120`，`hidden_dim=64`，`lr=0.001`，`seed=0`。

结果摘要：

1. 训练集 learned stock MAE：`0.046007703989744186`；
2. 训练集 zero-flow stock MAE：`0.1053290069103241`；
3. 验证集 learned stock MAE：`0.06032923609018326`；
4. 验证集 zero-flow stock MAE：`0.11418518424034119`；
5. 验证集相对 zero-flow 改善：`0.4716544314260591`；
6. 验证集 flow MAE：`0.07328741252422333`。

复现边界：

1. 该结果说明六核心存量和诊断流量标签存在可学习的一步转移信号；
2. 该结果尚未验证反馈回路、不确定性校准、风险校准或 SD-MCTS；
3. 下一阶段应加入 `no-feedback` 与 feedback-on 对比，随后再进入 learned transition provider 驱动的 SD-MCTS smoke。

## 2026-06-10 Sokoban learned flow level-holdout smoke

本轮使用 `sdmcts_p100` conda 环境执行：

1. Python：`/home/anaconda3/envs/sdmcts_p100/bin/python`
2. Python 版本：`3.10.20`
3. NumPy：`1.26.4`
4. PyTorch：`2.1.2+cu118`
5. `torch.cuda.is_available()`：`True`

脚本 [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已补充 `--split-mode level`，用于按关卡留出验证集，避免随机 transition split 的同关卡泄漏。

语法检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

三种子 level-holdout 命令模板：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 1 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed<seed>.json
```

结果文件：

1. [flow_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed0.json)
2. [flow_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed1.json)
3. [flow_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_seed2.json)
4. [flow_smoke_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_smoke_30_levelsplit_summary.json)

三种子验证集摘要：

| 指标 | seed0 | seed1 | seed2 | 均值 | 总体标准差 |
| --- | --- | --- | --- | --- | --- |
| learned stock MAE | `0.04891884699463844` | `0.05621336027979851` | `0.04982342943549156` | `0.0516518789033095` | `0.0032465265139958385` |
| zero-flow stock MAE | `0.10804840922355652` | `0.09091933071613312` | `0.09552866220474243` | `0.09816546738147736` | `0.0072372134961558535` |
| relative improvement | `0.547250650461467` | `0.3817226783674094` | `0.47844523009537` | `0.4691395196414155` | `0.06789611953409615` |
| flow MAE | `0.04675167053937912` | `0.04350985586643219` | `0.04634956642985344` | `0.04553703094522158` | `0.0014427984448686162` |

复现边界：

1. 该结果较随机 transition split 更能说明 learned flow 对未见关卡有初步泛化信号；
2. 该结果仍是 `no-feedback`、`no-uncertainty`、`no-SD-MCTS`；
3. 下一阶段应实现 feedback-on flow smoke，并与当前 `no-feedback` level-holdout 结果做同口径对比。

## 2026-06-10 Sokoban feedback-gated flow smoke

[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已新增 `--model-kind base|feedback`。其中 `feedback` 版本使用目标压力、目标完成、机动性缺口、可达性缺口、拥塞压力、拥塞-机动性耦合、拥塞-可达性耦合、风险压力和风险-拥塞耦合生成流量乘性门控。

三种子 feedback-gated level-holdout 命令模板：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 1 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind feedback \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed<seed>.json
```

结果文件：

1. [flow_feedback_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed0.json)
2. [flow_feedback_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed1.json)
3. [flow_feedback_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_smoke_30_levelsplit_seed2.json)
4. [flow_feedback_vs_base_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feedback_vs_base_30_levelsplit_summary.json)

同口径三种子对比：

| 模型 | valid stock MAE 均值 | 总体标准差 | valid relative improvement 均值 | valid flow MAE 均值 |
| --- | --- | --- | --- | --- |
| `base no-feedback` | `0.0516518789033095` | `0.0032465265139958385` | `0.4691395196414155` | `0.04553703094522158` |
| `feedback-gated` | `0.050986512253681816` | `0.003173560523246053` | `0.4760261538188252` | `0.04690564547975858` |

`feedback-gated` 相对 `base no-feedback` 的 valid stock MAE 均值降低 `0.0006653666496276855`，相对变化约 `-0.012881751133840233`。该结果说明当前显式反馈门控可运行，并略微改善一步 stock 预测，但增益较小，暂不足以作为强反馈机制主张。

复现边界：

1. 该实验仍不包含不确定性校准、风险校准或 SD-MCTS；
2. 当前 feedback gate 是最小结构接入，不代表完整系统动力学反馈回路已经验证；
3. 下一步应优先分析 feedback 增益较小的原因，再决定是否进入 learned-transition-provider SD-MCTS smoke。

## 2026-06-10 Sokoban feedback feature-mode ablation

[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已新增 `--feature-mode stock_action|stock_only|action_only`，用于检查 feedback 增益较小是否因为反馈信息已被 `x_t` 吸收。

本轮运行 `action_only` 与 `action_only + feedback` 三种子 level-holdout smoke，其他参数沿用 `30` 关卡、`max_depth=1`、每关最多 `30` 个状态、`epochs=200`、`hidden_dim=64`、`batch_size=128`、`sdmcts_p100` 环境。

结果文件：

1. [flow_actiononly_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_smoke_30_levelsplit_seed0.json)
2. [flow_actiononly_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_smoke_30_levelsplit_seed1.json)
3. [flow_actiononly_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_smoke_30_levelsplit_seed2.json)
4. [flow_actiononly_feedback_smoke_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_feedback_smoke_30_levelsplit_seed0.json)
5. [flow_actiononly_feedback_smoke_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_feedback_smoke_30_levelsplit_seed1.json)
6. [flow_actiononly_feedback_smoke_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_actiononly_feedback_smoke_30_levelsplit_seed2.json)
7. [flow_feature_feedback_ablation_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_feature_feedback_ablation_30_levelsplit_summary.json)

同口径四组对比：

| 模型 | valid stock MAE 均值 | valid flow MAE 均值 | valid relative improvement 均值 |
| --- | --- | --- | --- |
| `stock_action_base` | `0.0516518789033095` | `0.04553703094522158` | `0.4691395196414155` |
| `stock_action_feedback` | `0.050986512253681816` | `0.04690564547975858` | `0.4760261538188252` |
| `action_only_base` | `0.04863980785012245` | `0.0570392037431399` | `0.5005144172166145` |
| `action_only_feedback` | `0.04908239717284838` | `0.06114010885357857` | `0.4960138807923717` |

消融结论：

1. `stock_action_feedback` 相对 `stock_action_base` 的 valid stock MAE 变化为 `-0.0006653666496276855`，约 `-1.2881751133840233%`；
2. `action_only_feedback` 相对 `action_only_base` 的 valid stock MAE 变化为 `0.0004425893227259295`，约 `0.9099323009040532%`；
3. `action_only_base` 的 valid stock MAE 低于 `stock_action_base`，但 flow MAE 更高，说明当前一步 stock 目标可能更多由动作局部特征和裁剪边界解释，而不是由完整 stock context 解释；
4. 当前证据不支持“feedback 增益小主要因为 `x_t` 已经吸收反馈信息”的解释。更合理的解释是当前 feedback 特征、乘性 gate 或一步 stock 目标不足以暴露反馈回路价值。

复现边界：

1. 该消融仍只验证一步 abstract stock prediction；
2. flow MAE 与 stock MAE 方向不完全一致，说明裁剪后的 stock 目标可能掩盖部分流量分解误差；
3. 下一阶段应优先增加多步 rollout consistency、风险/aliasing 加权样本或反馈目标，而不是直接进入完整 SD-MCTS。

## 2026-06-10 Sokoban multi-step rollout consistency smoke

[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 已新增多步滚动一致性能力。新增参数包括：

1. `--rollout-depth`：短轨迹 rollout 深度，本轮为 `2`；
2. `--rollout-loss-weight`：多步 stock prediction loss 权重，本轮比较 `1.0` 与 `0.0`；
3. `--max-rollout-samples-per-level`：每个关卡收集的完整短轨迹上限，本轮为 `2000`。

语法检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py
```

调试 smoke 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 5 \
  --max-depth 2 \
  --max-states-per-level 20 \
  --epochs 5 \
  --batch-size 64 \
  --hidden-dim 32 \
  --split-mode level \
  --model-kind feedback \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --max-rollout-samples-per-level 200 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_rollout_debug_5_levelsplit_seed0.json
```

三种子正式 smoke 命令模板：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind <base|feedback> \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight <1.0|0.0> \
  --max-rollout-samples-per-level 2000 \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/<run_name>_seed<seed>.json
```

结果文件：

1. [flow_rollout_base_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_base_30_levelsplit_seed0.json)
2. [flow_rollout_base_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_base_30_levelsplit_seed1.json)
3. [flow_rollout_base_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_base_30_levelsplit_seed2.json)
4. [flow_rollout_feedback_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_feedback_30_levelsplit_seed0.json)
5. [flow_rollout_feedback_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_feedback_30_levelsplit_seed1.json)
6. [flow_rollout_feedback_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_feedback_30_levelsplit_seed2.json)
7. [flow_rollouteval_nosup_base_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollouteval_nosup_base_30_levelsplit_seed0.json)
8. [flow_rollouteval_nosup_base_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollouteval_nosup_base_30_levelsplit_seed1.json)
9. [flow_rollouteval_nosup_base_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollouteval_nosup_base_30_levelsplit_seed2.json)
10. [flow_rollouteval_nosup_feedback_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollouteval_nosup_feedback_30_levelsplit_seed0.json)
11. [flow_rollouteval_nosup_feedback_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollouteval_nosup_feedback_30_levelsplit_seed1.json)
12. [flow_rollouteval_nosup_feedback_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollouteval_nosup_feedback_30_levelsplit_seed2.json)
13. [flow_rollout_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_30_levelsplit_summary.json)

三种子摘要：

| 组别 | valid stock MAE 均值 | valid flow MAE 均值 | valid rollout stock MAE 均值 | valid rollout step-2 stock MAE 均值 | rollout 相对 zero-flow 改善均值 |
| --- | --- | --- | --- | --- | --- |
| `rollout_sup_base` | `0.04440526167551676` | `0.04738566527764002` | `0.048756017039219536` | `0.056363618622223534` | `0.5453278369792934` |
| `rollout_sup_feedback` | `0.044753131767114006` | `0.04975706711411476` | `0.04945573459068934` | `0.057083974281946816` | `0.5387324912152597` |
| `one_step_train_base_rollout_eval` | `0.04724068194627762` | `0.03734490896264712` | `0.05358633274833361` | `0.062035638839006424` | `0.5000239138831838` |
| `one_step_train_feedback_rollout_eval` | `0.045533218731482826` | `0.03830864901343981` | `0.05165342055261135` | `0.0597091851135095` | `0.5179881048878595` |

对比结论：

1. `base` 模型加入 rollout supervision 后，valid rollout stock MAE 均值从 `0.05358633274833361` 降至 `0.048756017039219536`，step-2 stock MAE 从 `0.062035638839006424` 降至 `0.056363618622223534`；
2. `feedback` 模型加入 rollout supervision 后，valid rollout stock MAE 均值从 `0.05165342055261135` 降至 `0.04945573459068934`，step-2 stock MAE 从 `0.0597091851135095` 降至 `0.057083974281946816`；
3. 带 rollout supervision 时，`feedback` 相对 `base` 的 valid rollout stock MAE 均值高 `0.0006997175514698029`，未显示正向反馈增益；
4. step-2 逐存量误差显示主要误差集中在 `reachable_capacity` 与 `congestion`，`rollout_sup_base` 两者均值分别为 `0.12761695434649786` 与 `0.11908540378014247`。

复现边界：

1. 本轮结果仍使用诊断性 exact single-push provider 生成标签，不是 learned transition provider 驱动的 SD-MCTS；
2. 结果支持“多步 stock rollout loss 能改善短视界累计一致性”的 smoke 结论；
3. 结果不支持“当前 feedback gate 在两步滚动中已有稳定增益”的结论；
4. 下一阶段应先针对 `reachable_capacity`、`congestion` 和反馈目标改进模型，再进入 learned-transition-provider SD-MCTS smoke。

## 2026-06-10 Sokoban weighted rollout 与 feedback-target smoke

为检查 `reachable_capacity`、`congestion` 的 step-2 误差能否通过训练目标修正，[train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py) 新增：

1. `--stock-loss-preset uniform|reach_congestion`；
2. `--stock-loss-weights`；
3. `--feedback-target-loss-weight`。

`reach_congestion` preset 的原始权重为 `[1.0, 1.0, 1.0, 2.5, 2.5, 1.0]`，按均值归一后写入结果文件为 `[0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 1.6666666269302368, 1.6666666269302368, 0.6666666865348816]`。`feedback-target-loss-weight=0.1` 仅用于 `model-kind=feedback`。

语法检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py
```

调试 smoke 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 3 \
  --max-depth 2 \
  --max-states-per-level 10 \
  --epochs 2 \
  --batch-size 32 \
  --hidden-dim 16 \
  --split-mode level \
  --model-kind feedback \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --stock-loss-preset reach_congestion \
  --feedback-target-loss-weight 0.1 \
  --max-rollout-samples-per-level 50 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_weighted_debug_feedback_target_3_levelsplit_seed0.json
```

正式 smoke 命令模板：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind <base|feedback> \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --stock-loss-preset <uniform|reach_congestion> \
  --feedback-target-loss-weight <0.0|0.1> \
  --max-rollout-samples-per-level 2000 \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/<run_name>_seed<seed>.json
```

结果文件：

1. [flow_rollout_uniform_feedback_target010_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_uniform_feedback_target010_30_levelsplit_seed0.json)
2. [flow_rollout_uniform_feedback_target010_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_uniform_feedback_target010_30_levelsplit_seed1.json)
3. [flow_rollout_uniform_feedback_target010_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_uniform_feedback_target010_30_levelsplit_seed2.json)
4. [flow_rollout_rc_base_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_base_30_levelsplit_seed0.json)
5. [flow_rollout_rc_base_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_base_30_levelsplit_seed1.json)
6. [flow_rollout_rc_base_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_base_30_levelsplit_seed2.json)
7. [flow_rollout_rc_feedback_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_feedback_30_levelsplit_seed0.json)
8. [flow_rollout_rc_feedback_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_feedback_30_levelsplit_seed1.json)
9. [flow_rollout_rc_feedback_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_feedback_30_levelsplit_seed2.json)
10. [flow_rollout_rc_feedback_target010_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_feedback_target010_30_levelsplit_seed0.json)
11. [flow_rollout_rc_feedback_target010_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_feedback_target010_30_levelsplit_seed1.json)
12. [flow_rollout_rc_feedback_target010_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_rc_feedback_target010_30_levelsplit_seed2.json)
13. [flow_rollout_weighted_feedback_target_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_rollout_weighted_feedback_target_30_levelsplit_summary.json)

三种子摘要：

| 组别 | valid rollout stock MAE 均值 | step-2 stock MAE 均值 | step-2 reachable capacity MAE 均值 | step-2 congestion MAE 均值 | valid flow MAE 均值 |
| --- | --- | --- | --- | --- | --- |
| `uniform_base` | `0.048756017039219536` | `0.056363618622223534` | `0.12761695434649786` | `0.11908540378014247` | `0.04738566527764002` |
| `uniform_feedback` | `0.04945573459068934` | `0.057083974281946816` | `0.12893449266751608` | `0.12103007237116496` | `0.04975706711411476` |
| `uniform_feedback_target010` | `0.04896984373529752` | `0.0566172848145167` | `0.129164457321167` | `0.12084175149599712` | `0.04955196753144264` |
| `reach_congestion_base` | `0.05112618890901407` | `0.05892247830828031` | `0.12749558687210083` | `0.12215788662433624` | `0.04716350386540095` |
| `reach_congestion_feedback` | `0.050664703051249184` | `0.05824825664361318` | `0.13176398475964865` | `0.12284435580174129` | `0.050431594252586365` |
| `reach_congestion_feedback_target010` | `0.050306430707375206` | `0.05791137491663297` | `0.1306424761811892` | `0.12274520844221115` | `0.049810374776522316` |

对比结论：

1. `reach_congestion_base` 相对 `uniform_base` 的 valid rollout stock MAE 均值上升 `0.0023701718697945323`，step-2 stock MAE 上升 `0.0025588596860567775`；
2. `reach_congestion_base` 的 step-2 reachable capacity MAE 仅降低 `0.00012136747439703277`，但 step-2 congestion MAE 上升 `0.003072482844193772`；
3. `uniform_feedback_target010` 相对 `uniform_feedback` 的 valid rollout stock MAE 均值降低 `0.00048589085539181565`，step-2 stock MAE 降低 `0.00046668946743011475`；
4. 但 `uniform_feedback_target010` 仍未超过 `uniform_base`，其 valid rollout stock MAE 均值高 `0.00021382669607798307`，step-2 stock MAE 高 `0.0002536661922931671`。

复现边界：

1. 简单提高 `reachable_capacity` 与 `congestion` 的训练权重没有解决这两个 stock 的 step-2 误差问题；
2. next-feedback auxiliary target 有弱正信号，但不足以抵消当前 feedback gate 相对 base 的劣势；
3. 该结果仍不支持进入 learned-transition-provider SD-MCTS，下一步应优先检查 stock proxy 与 flow decomposition，而不是继续堆叠损失权重。

## 2026-06-10 Sokoban frozen baseline 审计复测

根据 [sokoban_sd_rollout_model_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_rollout_model_review.md) 的审查结论，本轮冻结诊断性最小基线：

1. `baseline_id=sokoban_six_stock_nonnegative_flow_rollout_sup_base_v1`；
2. `model-kind=base`；
3. `feature-mode=stock_action`；
4. `stock-loss-preset=uniform`；
5. `rollout-depth=2`；
6. `rollout-loss-weight=1.0`；
7. `split-mode=level`；
8. `transition_labels_from=exact_single_push_diagnostic_provider`；
9. `sd_mcts_enabled=false`。

新增或修改代码：

1. [train_sokoban_flow_smoke.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py)：新增 `baseline_contract`、provider audit、逐存量/逐流量误差、clip audit 和 aliasing audit 输出；
2. [summarize_sokoban_flow_runs.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py)：新增三种子结果汇总与旧结果复现比较脚本。

语法检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

调试 smoke 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 3 \
  --max-depth 2 \
  --max-states-per-level 10 \
  --epochs 2 \
  --batch-size 32 \
  --hidden-dim 16 \
  --split-mode level \
  --model-kind base \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --max-rollout-samples-per-level 50 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_baseline_retest_debug_3_levelsplit_seed0.json
```

正式三种子复测命令模板：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind base \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --stock-loss-preset uniform \
  --max-rollout-samples-per-level 2000 \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed<seed>.json
```

汇总命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  --group-name baseline_retest_rollout_sup_base \
  --files \
    results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed2.json \
  --compare-summary results/sokoban_sd_world_model/flow_rollout_30_levelsplit_summary.json \
  --compare-group rollout_sup_base \
  --repro-threshold 0.05 \
  --out results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_summary.json
```

结果文件：

1. [flow_baseline_retest_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed0.json)
2. [flow_baseline_retest_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed1.json)
3. [flow_baseline_retest_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_seed2.json)
4. [flow_baseline_retest_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_retest_30_levelsplit_summary.json)

三种子复测摘要：

| 指标 | seed0 | seed1 | seed2 | 均值 | 总体标准差 |
| --- | --- | --- | --- | --- | --- |
| valid stock MAE | `0.04631908982992172` | `0.04741789028048515` | `0.03947880491614342` | `0.04440526167551676` | `0.0035122948584480683` |
| valid flow MAE | `0.053571492433547974` | `0.050502531230449677` | `0.038082972168922424` | `0.04738566527764002` | `0.0066962529334331005` |
| valid rollout stock MAE mean | `0.05028420127928257` | `0.05048411525785923` | `0.045499734580516815` | `0.048756017039219536` | `0.0023039853859271584` |
| valid rollout step-2 stock MAE | `0.0569138340651989` | `0.059068288654088974` | `0.053108733147382736` | `0.056363618622223534` | `0.002463889639661186` |
| valid rollout improvement over zero-flow | `0.5426761971719848` | `0.502383079420081` | `0.5909242343458144` | `0.5453278369792934` | `0.0361953719338671` |

与旧 `rollout_sup_base` 结果比较，以上主指标均为 exact match，`reproducibility_comparison.verdict=REPRODUCIBLE`，`mismatched_metrics=[]`。

新增审计摘要：

| 审计指标 | seed0 | seed1 | seed2 | 均值 | 总体标准差 |
| --- | --- | --- | --- | --- | --- |
| valid any-stock clipped fraction | `0.9121779799461365` | `0.9458987712860107` | `0.9251613020896912` | `0.9277460177739462` | `0.013887248552134917` |
| valid mean abs clip correction | `0.024118032306432724` | `0.013977224007248878` | `0.013937568292021751` | `0.017344274868567783` | `0.00478979717832328` |
| stock-only alias max next-delta variance | `0.3904161751270294` | `0.3803754448890686` | `0.3954983055591583` | `0.38876330852508545` | `0.006283534287570066` |
| stock-action alias max next-delta variance | `0.09125160425901413` | `0.10170450061559677` | `0.11775518208742142` | `0.1035704289873441` | `0.010900188865121537` |

复现边界：

1. 复测证明冻结基线主指标在当前环境下可精确复现；
2. 新增审计不改变训练结果，只补充结果解释所需的可审计字段；
3. clipping 与 aliasing 审计显示当前抽象存量仍存在明显诊断问题，进入 learned-transition-provider SD-MCTS 前应先检查 stock proxy、flow decomposition 和 alias bucket；
4. 本轮仍不是 formal SD-MCTS 结果，因为 transition 标签仍来自 `exact_single_push_diagnostic_provider`。

## 2026-06-10 Sokoban stock-flow 诊断增强复测

本轮继续使用 `sdmcts_p100` conda 环境：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python
```

先执行语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/audit_sokoban_stock_flow_diagnostics.py \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  experiments/sokoban_sd_world_model/validate_sokoban_sd_design.py \
  src/sokoban
```

随后在不改变冻结基线训练配置的前提下，重新运行三种子增强审计版：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind base \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --stock-loss-preset uniform \
  --max-rollout-samples-per-level 2000 \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed<seed>.json
```

三种子汇总命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  --group-name baseline_diagnostics_rollout_sup_base \
  --files \
    results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed2.json \
  --compare-summary results/sokoban_sd_world_model/flow_rollout_30_levelsplit_summary.json \
  --compare-group rollout_sup_base \
  --repro-threshold 0.05 \
  --out results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_summary.json
```

stock-flow 诊断命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/audit_sokoban_stock_flow_diagnostics.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --seeds 0 1 2 \
  --alias-round-decimals 2 \
  --top-k 8 \
  --sample-limit-per-bucket 4 \
  --baseline-files \
    results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed2.json \
  --out results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json
```

输出文件：

1. [flow_baseline_diagnostics_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed0.json)
2. [flow_baseline_diagnostics_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed1.json)
3. [flow_baseline_diagnostics_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed2.json)
4. [flow_baseline_diagnostics_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_summary.json)
5. [stock_flow_diagnostics_30_levelsplit.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json)

复现结果：

1. `flow_baseline_diagnostics_30_levelsplit_summary.json` 与旧 `flow_rollout_30_levelsplit_summary.json` 中 `rollout_sup_base` 的主指标 exact match，`reproducibility_comparison.verdict=REPRODUCIBLE`，`mismatched_metrics=[]`；
2. valid rollout stock MAE 均值为 `0.048756017039219536`，step-2 stock MAE 均值为 `0.056363618622223534`，valid rollout improvement over zero-flow 均值为 `0.5453278369792934`；
3. `stock_flow_diagnostics_30_levelsplit.json` 覆盖 `30` 个关卡、`566` 个状态和 `3435` 条 transition；
4. exact provider 的 incidence balance `max_l1=0.0`；
5. baseline valid any-stock clipped fraction 均值为 `0.9277460177739462`，valid mean abs clip correction 均值为 `0.017344274868567783`；
6. stock-action alias max next-delta variance 最高为 `0.11775519031141866`；
7. `reachable_capacity` delta nonzero fraction 为 `0.37147016011644834`，`congestion` delta nonzero fraction 为 `0.6954876273653566`。

复现边界：

1. 本轮仍是 diagnostic learned stock-flow baseline 与审计，不是 learned-transition-provider SD-MCTS；
2. 新增审计字段用于解释误差来源，不改变训练目标；
3. 由于相反流量共激活、裁剪和 aliasing 均较明显，下一步应进行单因素模型修订与复测。

## 2026-06-10 Sokoban 阶段 1 paired/net-flow 修订

本轮继续使用 `sdmcts_p100` conda 环境。先完成语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  experiments/sokoban_sd_world_model/evaluate_stage1_flow_gate.py
```

调试 smoke 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 3 \
  --max-depth 2 \
  --max-states-per-level 10 \
  --epochs 2 \
  --batch-size 32 \
  --hidden-dim 16 \
  --split-mode level \
  --model-kind base \
  --flow-parameterization paired_net \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --stock-loss-preset uniform \
  --max-rollout-samples-per-level 50 \
  --seed 0 \
  --out results/sokoban_sd_world_model/flow_stage1_paired_debug_3_levelsplit_seed0.json
```

正式三种子命令模板：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind base \
  --flow-parameterization paired_net \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --stock-loss-preset uniform \
  --max-rollout-samples-per-level 2000 \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed<seed>.json
```

汇总命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  --group-name stage1_paired_net \
  --files \
    results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed2.json \
  --compare-summary results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_summary.json \
  --compare-group baseline_diagnostics_rollout_sup_base \
  --repro-threshold 0.05 \
  --out results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_summary.json
```

阶段 1 gate 判定命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage1_flow_gate.py \
  --summary results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_summary.json \
  --files \
    results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed2.json \
  --diagnostics results/sokoban_sd_world_model/stock_flow_diagnostics_30_levelsplit.json \
  --out results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_gate.json
```

输出文件：

1. [flow_stage1_paired_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed0.json)
2. [flow_stage1_paired_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed1.json)
3. [flow_stage1_paired_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_seed2.json)
4. [flow_stage1_paired_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_summary.json)
5. [flow_stage1_paired_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_30_levelsplit_gate.json)

阶段 1 判定结果为 `PARTIAL`。通过项：

1. flow pair predicted coactivation mean 为 `0.0`，低于门限 `0.20`；
2. flow pair predicted coactivation max 为 `0.0`，低于门限 `0.30`；
3. incidence balance max L1 为 `0.0`。

未通过项：

1. valid rollout stock MAE mean 为 `0.05166300448278586`，高于门限 `0.05119`；
2. valid rollout step-2 stock MAE 为 `0.05988542859752973`，高于门限 `0.05918`；
3. valid mean abs clip correction 为 `0.028677965203921`，高于门限 `0.01214`；
4. valid any-stock clipped fraction 为 `0.9618482192357382`，高于门限 `0.75`。

与冻结基线比较：

1. valid flow MAE 从 `0.04738566527764002` 降至 `0.03719972570737203`；
2. valid rollout stock MAE mean 从 `0.048756017039219536` 升至 `0.05166300448278586`；
3. valid rollout step-2 stock MAE 从 `0.056363618622223534` 升至 `0.05988542859752973`；
4. valid mean abs clip correction 从 `0.017344274868567783` 升至 `0.028677965203921`。

复现边界：

1. `paired_net` 解决了相反流共激活问题，但不是阶段 1 达标模型；
2. 该结果不支持进入阶段 2 或 SD-MCTS；
3. 下一轮应保留成对互斥思想，但增加流量幅度或 raw-next 边界约束。

### Stage-1 Capacity Projection + Final-step Rollout Calibration

阶段 1 达标候选使用同一 `sdmcts_p100` 环境与同一 `medium/train/000.txt` 数据源。正式三种子命令模板如下：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind base \
  --flow-parameterization paired_net \
  --capacity-projection-stock-mask 1,1,1,0,0,1 \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --rollout-final-step-weight 1.2 \
  --stock-loss-preset uniform \
  --max-rollout-samples-per-level 2000 \
  --seed <0|1|2> \
  --out results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed<seed>.json
```

汇总命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/summarize_sokoban_flow_runs.py \
  --group-name stage1_paired_capacity_projection_final12 \
  --files \
    results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed2.json \
  --out results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json
```

阶段 1 gate 判定命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage1_flow_gate.py \
  --summary results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json \
  --files \
    results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed0.json \
    results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed1.json \
    results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed2.json \
  --diagnostics results/sokoban_sd_world_model/flow_baseline_diagnostics_30_levelsplit_seed0.json \
  --out results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json
```

输出文件：

1. [flow_stage1_paired_proj101001_final12_30_levelsplit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed0.json)
2. [flow_stage1_paired_proj101001_final12_30_levelsplit_seed1.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed1.json)
3. [flow_stage1_paired_proj101001_final12_30_levelsplit_seed2.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_seed2.json)
4. [flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_summary.json)
5. [flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/flow_stage1_paired_proj101001_final12_30_levelsplit_gate.json)

阶段 1 判定结果为 `PASS`。观测值为：valid rollout stock MAE mean `0.05017088601986567`，valid rollout step-2 stock MAE `0.058157226691643395`，valid mean abs clip correction `9.097307914392635e-07`，valid any-stock clipped fraction `0.0005817335719863573`，flow pair predicted coactivation mean/max `0.0/0.0`，incidence balance max L1 `0.0`。本结果仍为 learned stock-flow world-model 诊断阶段结果，未启用 learned-transition-provider SD-MCTS。

### Stage-2 Learned Provider SD-MCTS Interface Audit

阶段 2 使用同一 `sdmcts_p100` 环境。先用阶段 1 通过配置补齐可加载 checkpoint，命令如下：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_flow_smoke.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-depth 2 \
  --max-states-per-level 30 \
  --epochs 200 \
  --batch-size 128 \
  --hidden-dim 64 \
  --split-mode level \
  --model-kind base \
  --flow-parameterization paired_net \
  --capacity-projection-stock-mask 1,1,1,0,0,1 \
  --feature-mode stock_action \
  --rollout-depth 2 \
  --rollout-loss-weight 1.0 \
  --rollout-final-step-weight 1.2 \
  --stock-loss-preset uniform \
  --max-rollout-samples-per-level 2000 \
  --seed 0 \
  --out results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0_summary.json \
  --checkpoint-out results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt
```

该 checkpoint summary 的 seed0 验证指标与阶段 1 seed0 一致：valid stock MAE `0.04721938446164131`，valid rollout step-1 stock MAE `0.04322969540953636`，valid rollout step-2 stock MAE `0.0566064827144146`。

随后运行 learned provider + SD-MCTS 接口审计：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/audit_stage2_learned_provider_sdmcts.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-root-depth 1 \
  --max-root-states-per-level 12 \
  --num-simulations 48 \
  --mcts-depth 2 \
  --seed 0 \
  --out results/sokoban_sd_world_model/stage2_learned_provider_sdmcts_audit_seed0.json
```

输出文件：

1. [stage2_provider_checkpoint_seed0_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0_summary.json)
2. [stage2_provider_checkpoint_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt)
3. [stage2_learned_provider_sdmcts_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage2_learned_provider_sdmcts_audit_seed0.json)

审计结果为 `PASS`，`failed_checks=[]`。审计覆盖 checkpoint 记录的 `6` 个 held-out levels、`42` 个 root 和 `793` 条 MCTS 展开 transition。所有抽象 transition provider 均为 `learned_stock_flow_checkpoint_v1`，learned provider fraction 为 `1.0`；raw exact push 只记录为 `exact_push_raw_state_support_for_action_enumeration`，用于合法动作枚举。

关键观测值如下：

1. roots with expanded children fraction `1.0`；
2. flow nonnegative violations `0`；
3. stock range violations `0`；
4. incidence balance max L1 `1.1748737760441941e-07`；
5. clip any-stock fraction `0.0037831021437578815`；
6. clip mean abs correction `2.5552278208401885e-05`；
7. flow pair predicted coactivation mean/max `0.0/0.0`；
8. depth-1 stock MAE `0.05060150917204054`；
9. depth-2 stock MAE `0.056905325794544735`。

复现边界：该结果只验证 learned transition provider 能被最小 SD-MCTS 展开消费，并不报告 Sokoban/Boxoban 求解率。正式规划实验仍需另行运行多种子、多预算和失败类型分析。

### Stage-3 Closed-loop Planning Smoke

阶段 3 使用同一 `sdmcts_p100` 环境和阶段 2 checkpoint。脚本为：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py --help
```

小预算多方法 smoke 命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,zero_flow_sdmcts,oracle_abstract_sdmcts,heuristic_diagnostic \
  --budgets 16,48 \
  --seeds 0,1 \
  --mcts-depth 3 \
  --max-pushes 30 \
  --max-state-revisits 3 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_planning_smoke_seed01_budget16_48.json
```

较大预算 SD-MCTS smoke 命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,oracle_abstract_sdmcts,zero_flow_sdmcts \
  --budgets 200 \
  --seeds 0 \
  --mcts-depth 5 \
  --max-pushes 60 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_planning_budget200_depth5_seed0.json
```

best-first diagnostic 参照命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods heuristic_best_first_diagnostic \
  --budgets 5000 \
  --seeds 0 \
  --out results/sokoban_sd_world_model/stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json
```

输出文件：

1. [stage3_sdmcts_planning_smoke_seed01_budget16_48.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_planning_smoke_seed01_budget16_48.json)
2. [stage3_sdmcts_planning_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_planning_budget200_depth5_seed0.json)
3. [stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_heuristic_best_first_diagnostic_heldout6_budget5000.json)

小预算 smoke 结果：provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`，但 `learned_sdmcts`、`oracle_abstract_sdmcts`、`zero_flow_sdmcts` 和 greedy `heuristic_diagnostic` 在预算 `16/48` 下成功率均为 `0.0`。

较大预算 smoke 结果：provider usage gate 仍为 `PASS`，`learned_provider_fraction=1.0`；但 `learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 在预算 `200`、深度 `5`、最多 `60` push 下成功率仍为 `0.0`。失败类型主要为 `no_legal_actions` 和 `state_revisit_limit`。

best-first diagnostic 参照：在相同 `6` 个 checkpoint held-out levels 上，`heuristic_best_first_diagnostic` 以 `5000` expansion 预算达到成功率 `1.0`，平均解长 `15.333333333333334` push，平均 expanded `278.3333333333333`。该结果只作为关卡可解性与启发式诊断参照，不属于 SD-MCTS 主结果。

### Stage-3 TVA v1 Reproduction

TVA v1 使用同一 `sdmcts_p100` conda 环境。新增脚本编译检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/sokoban_tva.py \
  experiments/sokoban_sd_world_model/train_sokoban_tva.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

阶段 3 旧 stock 叶值兼容性 smoke 命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --use-checkpoint-valid-levels \
  --max-eval-levels 1 \
  --methods learned_sdmcts \
  --budgets 2 \
  --seeds 0 \
  --mcts-depth 1 \
  --max-pushes 2 \
  --terminal-value-mode stock \
  --out results/sokoban_sd_world_model/stage3_stock_compat_smoke.json
```

TVA 训练命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 3 \
  --max-states-per-level 90 \
  --teacher-max-expansions 5000 \
  --epochs 400 \
  --batch-size 128 \
  --hidden-dim 128 \
  --lr 0.001 \
  --seed 0 \
  --out results/sokoban_sd_world_model/sokoban_tva_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt
```

输出文件：

1. [sokoban_tva_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_train_seed0.json)
2. [sokoban_tva_value_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt)

训练数据共 `1741` 个状态样本，train `1345`、valid `396`。target 归一化使用 train mean `-3.6920604705810547` 和 std `6.25695276260376`。最终 train MAE 为 `0.8201102614402771`，valid MAE 为 `1.1535100936889648`，valid corr 为 `0.9297597211519081`。

TVA 小预算复测命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,oracle_abstract_sdmcts,zero_flow_sdmcts \
  --budgets 16,48 \
  --seeds 0,1 \
  --mcts-depth 3 \
  --gamma 0.95 \
  --max-pushes 30 \
  --max-state-revisits 3 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_smoke_seed01_budget16_48.json
```

TVA 大预算复测命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,oracle_abstract_sdmcts,zero_flow_sdmcts \
  --budgets 200 \
  --seeds 0 \
  --mcts-depth 5 \
  --gamma 0.95 \
  --max-pushes 60 \
  --max-state-revisits 5 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_budget200_depth5_seed0.json
```

输出文件：

1. [stage3_stock_compat_smoke.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_stock_compat_smoke.json)
2. [stage3_sdmcts_tva_smoke_seed01_budget16_48.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_smoke_seed01_budget16_48.json)
3. [stage3_sdmcts_tva_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_budget200_depth5_seed0.json)

小预算复测中，provider usage gate 为 `PASS`，terminal value usage gate 为 `PASS`，TVA 叶值调用 `36944` 次。`learned_sdmcts` 在 budget `16` 和 `48` 下成功率均为 `0.0`；`oracle_abstract_sdmcts` 在 budget `16` 的两种子平均成功率为 `0.08333333333333333`，budget `48` 为 `0.0`；`zero_flow_sdmcts` 为 `0.0`。

大预算复测中，provider usage gate 为 `PASS`，terminal value usage gate 为 `PASS`，TVA 叶值调用 `76600` 次。`learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 的成功率均为 `0.0`。learned_sdmcts 的失败计数为 `state_revisit_limit=5`、`no_legal_actions=1`。

## 2026-06-10 Sokoban TVA v2 failed-relabel 复现记录

本轮使用同一 `sdmcts_p100` 环境：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python
```

代码改动范围：

1. [train_sokoban_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_tva.py)：新增 failed closed-loop rollout augmentation、样本权重、失败 target、`value_provider_id` 和 root action selection 训练轨迹采集参数；
2. [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)：新增 `--root-action-selection visits|value`，并从 TVA checkpoint contract 读取 provider id。

TVA v2 训练命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 3 \
  --max-states-per-level 90 \
  --teacher-max-expansions 5000 \
  --gamma 0.95 \
  --goal-value 10.0 \
  --push-cost 0.05 \
  --deadend-value -10.0 \
  --unsolved-value -4.0 \
  --solved-sample-weight 1.25 \
  --deadend-sample-weight 1.0 \
  --unsolved-sample-weight 2.0 \
  --augment-failed-rollouts \
  --failed-rollout-tva-checkpoint results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt \
  --failed-rollout-budget 48 \
  --failed-rollout-depth 3 \
  --failed-rollout-max-pushes 30 \
  --failed-rollout-max-state-revisits 3 \
  --failed-rollout-seeds 0,1 \
  --failed-rollout-max-levels 24 \
  --failed-rollout-max-samples 600 \
  --failed-rollout-sample-weight 2.0 \
  --closed-loop-failure-value -8.0 \
  --closed-loop-discount-horizon 10 \
  --root-action-selection visits \
  --epochs 400 \
  --batch-size 128 \
  --hidden-dim 128 \
  --lr 0.001 \
  --weight-decay 1e-5 \
  --seed 0 \
  --value-provider-id sokoban_tva_value_v2_failed_relabel \
  --out results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt
```

TVA v2 输出文件：

1. [sokoban_tva_v2_failed_relabel_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_train_seed0.json)
2. [sokoban_tva_v2_failed_relabel_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt)

训练摘要：

1. 总样本数 `2341`，train `1945`，valid `396`；
2. source counts：`teacher_state_sample=1741`，`closed_loop_state_revisit_limit=378`，`closed_loop_no_legal_actions=36`，`closed_loop_planner_returned_no_action=96`，`closed_loop_max_pushes=90`；
3. failed rollout collection failure counts：`state_revisit_limit=19`，`no_legal_actions=3`，`planner_returned_no_action=6`，`max_pushes=3`；
4. train MAE `1.6621757745742798`，valid MAE `1.5998603105545044`，valid corr `0.8909407432297295`；
5. valid `proven_dead_end` MAE `0.11926791816949844`，target mean `-10.0`，pred mean `-9.899348258972168`；
6. valid `solved` MAE `5.988962650299072`，target mean `4.7114176750183105`，pred mean `-1.2775442600250244`；
7. valid `teacher_budget_unsolved` MAE `1.7181379795074463`，target mean `-4.0`，pred mean `-3.0325918197631836`。

TVA v2 `visits` root selection 复测命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 200 \
  --seeds 0 \
  --mcts-depth 5 \
  --gamma 0.95 \
  --max-pushes 60 \
  --max-state-revisits 5 \
  --root-action-selection visits \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json
```

TVA v2 `value` root selection 复测命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 200 \
  --seeds 0 \
  --mcts-depth 5 \
  --gamma 0.95 \
  --max-pushes 60 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json
```

复测输出文件：

1. [stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_visits_budget200_depth5_seed0.json)
2. [stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_v2_failed_relabel_value_budget200_depth5_seed0.json)

两条复测结果一致：

1. provider usage gate：`PASS`，`learned_provider_fraction=1.0`；
2. terminal value usage gate：`PASS`，`tva_provider_id=sokoban_tva_value_v2_failed_relabel`，`tva_evaluations=17600`；
3. `learned_sdmcts` success rate `0.0`；
4. mean pushes `14.166666666666666`；
5. failure counts：`planner_returned_no_action=3`，`state_revisit_limit=3`；
6. terminal leaf counts：`deadlocked=4037`，`frontier_cutoff=1706`，`depth_cutoff=11727`，`no_legal_actions=130`。

## 2026-06-10 Sokoban search-value TVA 修订审查记录

本轮未运行新的训练或评测实验，只完成 TVA 修订方案的 `academic-research-suite` 审查。使用的 skill 路由为 experiment-agent plan/validation、academic-paper-reviewer methodology reviewer、devil's advocate reviewer 和 editorial synthesis。审查文档为：

1. [sokoban_tva_search_value_design_review.md](/home/data/round/code/sd_world_model_rl/docs/sokoban_tva_search_value_design_review.md)

审查结论要求后续正式 TVA checkpoint 满足以下可复现 contract：

1. `value_provider_id=sokoban_search_value_tva_v1`；
2. `target_source=sd_mcts_root_statistics`；
3. `mcts_root_statistics_used=true`；
4. `teacher_used_online=false`；
5. `transition_provider_modified=false`；
6. `provider_id=learned_stock_flow_checkpoint_v1`；
7. 记录 `alpha`、`gamma`、`lambda_td`、`search_budget`、`search_depth`、`terminal_value_id`、target network 同步策略和随机种子；
8. 记录 finite `V_search` fraction、root child coverage、depth-1 `Q_H` 语义审计结果、solved/dead-end/budget-unsolved 分组指标。

后续实现时应在同一 conda 环境下运行：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python
```

本轮未产生新的模型 checkpoint 或结果 JSON；因此 reproducibility verdict 为 `CANNOT_VERIFY`，原因是该轮为设计审查而非可执行实验。

## 2026-06-10 Sokoban search-value TVA v1 复现记录

本轮使用同一 `sdmcts_p100` 环境：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python
```

新增脚本：

1. [train_sokoban_search_value_tva.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py)

编译检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m compileall \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  experiments/sokoban_sd_world_model/sokoban_tva.py
```

基础 search-value TVA 训练命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 3 \
  --max-states-per-level 90 \
  --collect-budget 48 \
  --mcts-depth 3 \
  --puct-c 1.25 \
  --gamma 0.95 \
  --alpha 1.0 \
  --bootstrap-terminal-value-mode stock \
  --root-action-selection visits \
  --lambda-td 0.0 \
  --qh-audit-max-states 80 \
  --epochs 400 \
  --batch-size 128 \
  --hidden-dim 128 \
  --lr 0.001 \
  --weight-decay 1e-5 \
  --seed 0 \
  --value-provider-id sokoban_search_value_tva_v1 \
  --out results/sokoban_sd_world_model/sokoban_search_value_tva_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_search_value_tva_seed0.pt
```

基础 search-value TVA 输出：

1. [sokoban_search_value_tva_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_train_seed0.json)
2. [sokoban_search_value_tva_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_seed0.pt)

基础训练摘要：

1. checkpoint contract：`target_source=sd_mcts_root_statistics`，`mcts_root_statistics_used=true`，`teacher_used_online=false`，`teacher_used_for_training_target=false`，`transition_provider_modified=false`；
2. `Q_H` depth-1 语义审计：`checked_states=80`，`max_abs_error=1.7763568394002505e-15`，`verdict=PASS`；
3. 样本数 `873`，train `728`，valid `145`，finite `V_search` fraction `0.5014359563469271`；
4. valid MAE `1.3938531875610352`，valid corr `0.5508857019779951`；
5. valid positive search target：target mean `0.8352776765823364`，pred mean `-0.8174604773521423`，未通过正例保护。

基础 search-value TVA 阶段 3 复测命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,oracle_abstract_sdmcts,zero_flow_sdmcts \
  --budgets 200 \
  --seeds 0 \
  --mcts-depth 5 \
  --gamma 0.95 \
  --max-pushes 60 \
  --max-state-revisits 5 \
  --root-action-selection visits \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_search_value_tva_seed0.pt \
  --out results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_budget200_depth5_seed0.json
```

复测输出：

1. [stage3_sdmcts_search_value_tva_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_budget200_depth5_seed0.json)

基础复测结果：provider usage gate 为 `PASS`，terminal value usage gate 为 `PASS`，TVA provider id 为 `sokoban_search_value_tva_v1`，TVA 调用数 `37800`。`learned_sdmcts`、`oracle_abstract_sdmcts` 和 `zero_flow_sdmcts` 成功率均为 `0.0`。`learned_sdmcts` mean pushes 为 `9.0`，失败计数为 `no_legal_actions=3`、`state_revisit_limit=1`、`planner_returned_no_action=2`。

随后运行 positive search target 权重消融。`positive-target-sample-weight=4.0` 输出为：

1. [sokoban_search_value_tva_posw4_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw4_train_seed0.json)
2. [sokoban_search_value_tva_posw4_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw4_seed0.pt)

`positive-target-sample-weight=10.0` 训练命令与基础命令相同，仅增加：

```bash
  --positive-target-sample-weight 10.0 \
  --nonpositive-target-sample-weight 1.0 \
  --value-provider-id sokoban_search_value_tva_v1_posw10 \
  --out results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_seed0.pt
```

posw10 输出：

1. [sokoban_search_value_tva_posw10_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_train_seed0.json)
2. [sokoban_search_value_tva_posw10_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_seed0.pt)

正例权重消融摘要：

1. posw4 valid positive pred mean `-0.5202222466468811`；
2. posw10 valid positive pred mean `-0.1770382821559906`；
3. 两者仍未使 valid positive search target 预测均值转正，因此正例保护 gate 未通过。

posw10 阶段 3 visits-root 复测输出：

1. [stage3_sdmcts_search_value_tva_posw10_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_posw10_budget200_depth5_seed0.json)

结果为：provider usage gate `PASS`，terminal value usage gate `PASS`，TVA 调用数 `64800`；`learned_sdmcts`、`oracle_abstract_sdmcts`、`zero_flow_sdmcts` 成功率均为 `0.0`。`learned_sdmcts` mean pushes 为 `18.166666666666668`，失败计数为 `state_revisit_limit=5`、`no_legal_actions=1`。

posw10 value-root learned_sdmcts 消融输出：

1. [stage3_sdmcts_search_value_tva_posw10_value_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_search_value_tva_posw10_value_budget200_depth5_seed0.json)

该消融 provider usage gate 与 terminal value usage gate 均为 `PASS`，`learned_sdmcts` 成功率仍为 `0.0`，mean pushes 为 `18.166666666666668`。

本轮判定：TVA 语义实现、`Q_H` 审计、provider usage 和 terminal value usage 均通过；但正例保护和 held-out success rate 未通过。因此该轮为 `ANALYZED` 负结果，不是阶段 3 通过模型。

### SD 目标一致性审计复现记录

脚本：

1. [audit_sokoban_sd_goal_alignment.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py)

编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py
```

quick 审计命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --teacher-max-expansions 800 \
  --max-action-states-per-level 3 \
  --out results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_quick_seed0.json
```

输出：

1. [sd_goal_alignment_audit_30_levelsplit_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_quick_seed0.json)

复现摘要：

1. Material Passport：`origin_skill=academic-research-suite/experiment-agent`，`origin_mode=validate`，`verification_status=ANALYZED`；
2. 状态级样本：train `231`，valid `60`；
3. 动作后果样本：train `66`，valid `17`；
4. teacher 标签计数：`solved=156`，`teacher_budget_unsolved=52`，`proven_dead_end=83`；
5. gate 判定：`PASS`，`failed_checks=[]`；
6. 六存量 teacher value valid `R^2=0.8859336361147274`，valid corr `0.9414044913328575`，teacher solved AUC `0.9247685185185185`；
7. 六存量 alias within/global target variance ratio `0.022069573453746945`；
8. `all_existing_plus_goal_probes` 相对六存量 valid `R^2` 增益 `0.00877102176130351`；
9. 动作后果 valid top-1 agreement `0.8235294117647058`，mean teacher regret `0.6820606727595`。

已停止但不作为结果解释的重配置：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py \
  --out results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_seed0.json
```

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_sd_goal_alignment.py \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 35 \
  --teacher-max-expansions 2500 \
  --max-action-states-per-level 8 \
  --out results/sokoban_sd_world_model/sd_goal_alignment_audit_30_levelsplit_medium_seed0.json
```

上述 full 与 medium 配置因离线 teacher 搜索耗时较高停止，未生成有效 JSON，不进入结论。

## 2026-06-11 Sokoban expert abstract feedback 复现记录

本轮使用 `sdmcts_p100` conda 环境，只修改阶段 3 评测脚本：

1. [evaluate_stage3_sdmcts_planning.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py)

编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

旧 stock consumption 同配置对照：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode stock \
  --terminal-value-mode stock \
  --out results/sokoban_sd_world_model/stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json
```

旧对照输出为 [stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stock_consumption_learned6_budget24_depth4_seed0.json)，`learned_sdmcts` 成功率 `0.0`，provider usage gate 为 `PASS`。

expert feedback 三 provider quick 对照：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,oracle_abstract_sdmcts,zero_flow_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json
```

输出为 [stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_provider_compare_budget24_depth4_seed0.json)。关键结果：

1. `learned_sdmcts` 成功率 `0.16666666666666666`，解出 `datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt:6`，解长 `15` push；
2. `oracle_abstract_sdmcts` 成功率 `0.0`；
3. `zero_flow_sdmcts` 成功率 `0.0`；
4. provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`；
5. planner feedback usage gate 为 `PASS`，`planner_feedback_id=expert_abstract_dynamics_feedback_v1`，`planner_feedback_evaluations=6407`；
6. planning contract 记录 `planner_feedback_changes_transition_provider=false`、`expert_feedback_teacher_used_online=false`、`expert_feedback_best_first_used_online=false`。

budget 稳定性复测：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json
```

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 200 \
  --seeds 0 \
  --mcts-depth 5 \
  --max-pushes 60 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json
```

稳定性输出：

1. [stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.0`，provider gate 与 feedback gate 均为 `PASS`；
2. [stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth5_seed0.json)：`learned_sdmcts` 成功率 `0.0`，provider gate 与 feedback gate 均为 `PASS`；
3. [stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.0`；
4. [stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_puct025_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget48_depth4_puct025_seed0.json)：`puct_c=0.25` 时成功率仍为 `0.0`；
5. [stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth4_mix050_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_revisit_learned6_budget200_depth4_mix050_seed0.json)：expert prior uniform mix `0.5` 时成功率仍为 `0.0`。

本轮复现判定为 `ANALYZED`：quick 配置证明 expert feedback 可产生非零 closed-loop 成功率，并且 learned provider 合约未被破坏；budget `48/200` 复测失败，说明该修订尚未达到稳定阶段 3 通过。

### Expert feedback deadlocked-root 与 viability 复测

继续在 `sdmcts_p100` 环境下复测 deadlocked-root 展开修复和 viability feedback。编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

deadlocked-root 修复后，budget `24` 命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget24_depth4_seed0.json
```

deadlocked-root 修复后，budget `48` 命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget48_depth4_seed0.json
```

deadlocked-root 输出：

1. [stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`；
2. [stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_deadrootfix_learned6_budget48_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.0`，失败类型为 `no_legal_actions=5`、`max_pushes=1`，`planner_returned_no_action` 消失。

viability feedback budget `24` 与 `48` 命令分别为：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json
```

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --out results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json
```

viability 输出：

1. [stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`，mean pushes `19.833333333333332`；
2. [stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.0`，mean pushes `21.333333333333332`，失败类型 `no_legal_actions=5`、`max_pushes=1`；
3. [stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_visits_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_expert_feedback_viability_learned6_budget48_depth4_visits_seed0.json)：`root_action_selection=visits` 时成功率 `0.0`，mean pushes `32.0`，失败类型 `max_pushes=3`、`no_legal_actions=3`。

本节复现判定为 `ANALYZED`。deadlocked-root 修复解决 `planner_returned_no_action`；viability feedback 和 visits-root 使策略更保守，但没有通过 budget `48` 成功率门限。

### 2026-06-11 target2_prior action abstraction

编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

2-level `target2_prior` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_mini_budget24_depth4_seed0.json
```

输出 [stage3_sdmcts_target2_prior_mini_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_mini_budget24_depth4_seed0.json)：成功率 `0.5`，provider usage gate `PASS`，action abstraction usage gate `PASS`。

held-out `6` 关卡 budget `24` 与 `48`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json
```

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json
```

输出：

1. [stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`，`learned_provider_fraction=1.0`，`target2_prior_scored_roots=28`；
2. [stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_learned6_budget48_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.0`，失败类型 `max_pushes=2`、`no_legal_actions=4`。

provider compare：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts,oracle_abstract_sdmcts,zero_flow_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json
```

输出 [stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_prior_provider_compare_budget24_depth4_seed0.json)：`learned_sdmcts` 成功率 `0.16666666666666666`，`oracle_abstract_sdmcts=0.0`，`zero_flow_sdmcts=0.0`；provider、feedback 与 action abstraction gate 均为 `PASS`。

本节复现判定为 `ANALYZED`。`target2_prior` 合规接入，但未通过 budget `48` 稳定性门限。

### 2026-06-11 route consistency and goal-room packing diagnostics

route consistency 2-level budget `48`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --route-consistency-weight 0.5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_routecell_strong_t2prior_mini_budget48_depth4_seed0.json)：成功率 `0.0`，provider 与 action abstraction gate 均为 `PASS`。

goal-room packing 2-level budget `48`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --goal-room-order-weight 2.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_mini_budget48_depth4_seed0.json)：2-level 成功率 `0.5`，provider 与 action abstraction gate 均为 `PASS`。权重 `3.0` 的同口径输出 [stage3_sdmcts_goalroom_delay_w3_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w3_t2prior_mini_budget48_depth4_seed0.json)，成功率同为 `0.5`。

held-out `6` 关卡 budget `24` 检查：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 6 \
  --methods learned_sdmcts \
  --budgets 24 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --goal-room-order-weight 1.5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json
```

权重 `1.5`、`2.0`、`3.0` 的 held-out `6` 关卡 budget `24` 成功率均为 `0.0`。对应输出为：

1. [stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w15_t2prior_learned6_budget24_depth4_seed0.json)；
2. [stage3_sdmcts_goalroom_delay_w2_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w2_t2prior_learned6_budget24_depth4_seed0.json)；
3. [stage3_sdmcts_goalroom_delay_w3_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_delay_w3_t2prior_learned6_budget24_depth4_seed0.json)。

本节复现判定为 `ANALYZED`。route consistency 与 goal-room packing 均未满足进入阶段 3 候选的门限。

### Root-Q Bias、Audit Proxy 与 Learned Action Prior

root-Q bias 审计：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_root_q_bias.py

conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_root_q_bias.py \
  --out results/sokoban_sd_world_model/root_q_bias_audit_level6_target2_prior_seed0.json
```

输出 [root_q_bias_audit_level6_target2_prior_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/root_q_bias_audit_level6_target2_prior_seed0.json)。关键观测为 step `1` 首次分歧，`q_gap_failure_minus_success=1.0426520933606815`，`prior_gap_failure_minus_success=0.6487923854590042`，`visit_ratio_failure_to_success=4.8`。

audit-proxy prior：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --audit-proxy-prior-weight 1.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_auditproxy_t2prior_mini_budget48_depth4_seed0.json
```

2-level budget `48` 成功率为 `0.0`。held-out `6` 关卡 budget `24` 输出 [stage3_sdmcts_auditproxy_t2prior_learned6_budget24_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_auditproxy_t2prior_learned6_budget24_depth4_seed0.json)，成功率为 `0.16666666666666666`。

learned action prior 编译与训练：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/sokoban_learned_action_prior.py \
  experiments/sokoban_sd_world_model/train_sokoban_action_prior.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py

conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_action_prior.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-train-levels 24 \
  --collection-budget 24 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --epochs 120 \
  --batch-size 128 \
  --hidden-dim 96 \
  --seed 0 \
  --out results/sokoban_sd_world_model/sokoban_action_prior_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_action_prior_seed0.pt
```

输出 [sokoban_action_prior_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_action_prior_train_seed0.json) 与 [sokoban_action_prior_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_action_prior_seed0.pt)。样本总数 `443`，train positive `6`，train negative `321`，valid negative `116`，内部 valid 无正样本。

learned prior 入场复测：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-option-length 2 \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --learned-action-prior-checkpoint results/sokoban_sd_world_model/sokoban_action_prior_seed0.pt \
  --learned-action-prior-weight 0.1 \
  --learned-action-prior-max-multiplier 1.5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_learnedprior_w01_t2prior_mini_budget48_depth4_seed0.json
```

`weight=0.1` 与 `weight=0.5` 的 2-level budget `48` 成功率均为 `0.0`，输出分别为 [stage3_sdmcts_learnedprior_w01_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_learnedprior_w01_t2prior_mini_budget48_depth4_seed0.json) 和 [stage3_sdmcts_learnedprior_w05_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_learnedprior_w05_t2prior_mini_budget48_depth4_seed0.json)。本节复现判定为 `ANALYZED`，learned action prior v1 未进入阶段 3 候选。

### Goal-Room Candidate Stock Probe

候选 stock 审计编译与运行：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_goal_room_stock_probe.py

conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_goal_room_stock_probe.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --teacher-max-expansions 800 \
  --out results/sokoban_sd_world_model/goal_room_stock_probe_quick_seed0.json
```

输出 [goal_room_stock_probe_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/goal_room_stock_probe_quick_seed0.json)。关键结果为：candidate stock gate `FAIL`，失败项为 `valid_r2_gain_min_0.02`；六 stock valid `R^2=0.8859336464419086`，stock plus goal-room valid `R^2=0.8850849477314626`，增益 `-0.0008486987104460164`；goal-room transition nonzero fraction 为 `0.16944106925880922`。本节复现判定为 `ANALYZED`，不进入阶段 1/2 重训。

### Goal-Room Conflict Root Prior Probe

兼容性编译检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  experiments/sokoban_sd_world_model/train_sokoban_action_prior.py \
  experiments/sokoban_sd_world_model/train_sokoban_tva.py
```

2-level budget `48`、`goal_room_conflict_prior_weight=1.0`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --goal-room-conflict-prior-weight 1.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_goalroom_conflictprior_w1_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_goalroom_conflictprior_w1_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_conflictprior_w1_t2prior_mini_budget48_depth4_seed0.json)。`learned_sdmcts` 成功率 `0.0`，mean pushes `40.0`，`learned_provider_fraction=1.0`，planner feedback usage gate 为 `PASS`，action abstraction gate 为 `PASS`。planning contract 记录该 prior 为 root-only，且不改变 transition provider、reward 或在线搜索器。

2-level budget `48`、`goal_room_conflict_prior_weight=2.0`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --goal-room-conflict-prior-weight 2.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_goalroom_conflictprior_w2_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_goalroom_conflictprior_w2_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_goalroom_conflictprior_w2_t2prior_mini_budget48_depth4_seed0.json)。`learned_sdmcts` 成功率仍为 `0.0`，mean pushes `40.0`，三个 usage gate 均通过。level `000.txt:6` 中该 prior 触发 `2` 个 root，但失败前缀保持 `[1, 12, 15, 4, 4]`。本节复现判定为 `ANALYZED`，该 prior 未进入阶段 3 候选。

### Train-Split Teacher-Q Action Prior

新增脚本编译：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_train_action_target_signal.py \
  experiments/sokoban_sd_world_model/train_sokoban_teacher_q_action_prior.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  experiments/sokoban_sd_world_model/sokoban_learned_action_prior.py
```

train-split action target signal quick audit：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_train_action_target_signal.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-train-levels 12 \
  --internal-holdout-fraction 0.25 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --max-action-states-per-level 5 \
  --teacher-max-expansions 800 \
  --out results/sokoban_sd_world_model/train_action_target_signal_quick_seed0.json
```

输出 [train_action_target_signal_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/train_action_target_signal_quick_seed0.json)。target signal gate 为 `PASS`：train best-action labels `36`，valid best-action labels `14`，valid rank states `14`，ridge valid top-1 agreement `0.42857142857142855`，valid Q corr `0.7964238411550628`，valid teacher-Q std `5.177792257077092`。

teacher-Q action prior best-epoch 训练：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_teacher_q_action_prior.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-train-levels 12 \
  --internal-holdout-fraction 0.25 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --max-action-states-per-level 5 \
  --teacher-max-expansions 800 \
  --epochs 160 \
  --batch-size 128 \
  --hidden-dim 96 \
  --seed 0 \
  --out results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_seed0.pt
```

输出 [sokoban_teacher_q_action_prior_best_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_train_seed0.json) 和 [sokoban_teacher_q_action_prior_best_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_seed0.pt)。best epoch 为 `9`；内部 holdout valid top-1 agreement `0.6428571428571429`，valid Q corr `0.8467250504899865`，valid `R^2=0.4791956049704875`。checkpoint contract 记录 `target_source=stage2_train_split_offline_successor_teacher_q`、`uses_stage2_train_split_only=true`、`uses_heldout_valid_split_for_training=false`、`teacher_used_online=false`、`best_first_used_online=false`、`transition_provider_modified=false`。

2-level budget `48` smoke，权重 `0.1`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --learned-action-prior-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_seed0.pt \
  --learned-action-prior-weight 0.1 \
  --learned-action-prior-max-multiplier 1.5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherqprior_w01_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherqprior_w01_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherqprior_w01_t2prior_mini_budget48_depth4_seed0.json)。成功率 `0.0`，provider、planner feedback、action abstraction、learned action prior usage gate 均为 `PASS`。

2-level budget `48` smoke，权重 `0.5`：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --planner-feedback-mode expert \
  --terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --max-options-per-node 4 \
  --option-prior-max-multiplier 2.0 \
  --learned-action-prior-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_action_prior_best_seed0.pt \
  --learned-action-prior-weight 0.5 \
  --learned-action-prior-max-multiplier 3.0 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherqprior_w05_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherqprior_w05_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherqprior_w05_t2prior_mini_budget48_depth4_seed0.json)。成功率仍为 `0.0`，四个 usage gate 均通过。本节复现判定为 `ANALYZED`，teacher-Q action prior 未进入阶段 3 候选。

Stage3-planner TVA v2 代码检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

stage3-planner TVA v2 quick 训练：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 8 \
  --collect-budget 24 \
  --mcts-depth 4 \
  --target-planner-mode stage3 \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --bootstrap-terminal-value-mode expert \
  --root-action-selection value \
  --lambda-td 0.25 \
  --target-sync-epochs 25 \
  --epochs 250 \
  --batch-size 64 \
  --hidden-dim 128 \
  --lr 0.001 \
  --seed 0 \
  --value-provider-id sokoban_search_value_tva_stage3_td_v2_quick \
  --out results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt
```

输出 [sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_train_seed0.json) 和 [sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt)。样本数 `217`，valid MAE `1.0694197416305542`，valid corr `0.9664165047048058`，正值 target 样本数 `12`。

2-level budget `48` smoke，root action selection by value：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_mini_budget48_depth4_seed0.json)。成功率 `0.0`，provider、terminal value、planner feedback、action abstraction usage gate 均为 `PASS`。

2-level budget `48` smoke，root action selection by visits：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection visits \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_tva_stage3_td_v2_quick_t2prior_visits_mini_budget48_depth4_seed0.json)。成功率仍为 `0.0`，四个 usage gate 均为 `PASS`。本节复现判定为 `ANALYZED`，该 TVA v2 未进入阶段 3 候选。

Stage3-planner TVA v2 context-recovery quick 训练：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 12 \
  --state-depth 2 \
  --max-states-per-level 8 \
  --collect-budget 24 \
  --mcts-depth 4 \
  --puct-c 1.25 \
  --gamma 0.95 \
  --alpha 1.0 \
  --target-planner-mode stage3 \
  --planner-feedback-mode expert \
  --bootstrap-terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --lambda-td 0.25 \
  --target-sync-epochs 20 \
  --epochs 160 \
  --batch-size 64 \
  --hidden-dim 128 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --qh-audit-max-states 20 \
  --value-provider-id sokoban_stage3_planner_tva_v2_quick \
  --out results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_seed0.pt
```

输出 [sokoban_stage3_planner_tva_v2_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_train_seed0.json) 和 [sokoban_stage3_planner_tva_v2_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_seed0.pt)。样本数 `81`，valid corr `0.9321286812910057`，valid MAE `1.2824660539627075`，正值 target `1` 个。

对应 2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --puct-c 1.25 \
  --gamma 0.95 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --out results/sokoban_sd_world_model/stage3_sdmcts_stage3tva_v2_quick_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_stage3tva_v2_quick_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stage3tva_v2_quick_t2prior_mini_budget48_depth4_seed0.json)。成功率 `0.0`，四个 usage gate 均为 `PASS`。

Stage3-planner TVA v2 context-recovery expanded 训练：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_search_value_tva.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --state-depth 2 \
  --max-states-per-level 16 \
  --collect-budget 24 \
  --mcts-depth 4 \
  --puct-c 1.25 \
  --gamma 0.95 \
  --alpha 1.0 \
  --target-planner-mode stage3 \
  --planner-feedback-mode expert \
  --bootstrap-terminal-value-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --lambda-td 0.25 \
  --target-sync-epochs 20 \
  --epochs 160 \
  --batch-size 128 \
  --hidden-dim 128 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --qh-audit-max-states 20 \
  --value-provider-id sokoban_stage3_planner_tva_v2_expanded \
  --out results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt
```

输出 [sokoban_stage3_planner_tva_v2_expanded_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_train_seed0.json) 和 [sokoban_stage3_planner_tva_v2_expanded_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt)。样本数 `356`，valid corr `0.9534958319265285`，valid MAE `0.9219829440116882`，正值 target `15` 个。

对应 2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --puct-c 1.25 \
  --gamma 0.95 \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --root-action-selection value \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --out results/sokoban_sd_world_model/stage3_sdmcts_stage3tva_v2_expanded_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_stage3tva_v2_expanded_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_stage3tva_v2_expanded_t2prior_mini_budget48_depth4_seed0.json)。成功率 `0.0`，provider、terminal value、planner feedback 和 action abstraction usage gate 均为 `PASS`。本节复现判定为 `ANALYZED`，expanded 版 TVA v2 不进入阶段 3 候选。

Teacher-Q state-value terminal leaf probe 训练：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/train_sokoban_teacher_q_state_value.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-train-levels 12 \
  --internal-holdout-fraction 0.25 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --max-action-states-per-level 5 \
  --teacher-max-expansions 800 \
  --target-aggregation max \
  --epochs 200 \
  --batch-size 64 \
  --hidden-dim 128 \
  --seed 0 \
  --value-provider-id sokoban_teacher_q_state_value_v1_quick \
  --out results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt
```

输出 [sokoban_teacher_q_state_value_v1_quick_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_train_seed0.json) 和 [sokoban_teacher_q_state_value_v1_quick_seed0.pt](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt)。训练使用 seed `0`，action-Q target signal gate 为 `PASS`，state-value gate 为 `PASS`。状态样本为 train `33`、valid `14`；valid MAE `3.12015962600708`，valid corr `0.3278236921586032`。分组指标显示 positive target 组 MAE `0.8567870259284973`，nonpositive target 组 MAE `6.137990474700928`，且 nonpositive target mean `-2.052501916885376`、prediction mean `4.085488319396973`。

对应 2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherq_statevalue_v1_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_t2prior_mini_budget48_depth4_seed0.json)。结果为 `learned_sdmcts` 成功率 `0.0`，episodes `2`，failure counts 为 `max_pushes=2`。provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`；terminal value usage gate 为 `PASS`，TVA evaluations `3840`；planner feedback usage gate 为 `PASS`；action abstraction usage gate 为 `PASS`，`target2_prior_scored_roots=8`。本节复现判定为 `ANALYZED`，该 probe 不进入阶段 3 候选。

Option-level target signal quick 审计：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_option_target_signal.py

conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_option_target_signal.py \
  --provider-checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 30 \
  --max-train-levels 12 \
  --internal-holdout-fraction 0.25 \
  --state-depth 2 \
  --max-states-per-level 12 \
  --max-action-states-per-level 5 \
  --max-options-per-node 6 \
  --teacher-max-expansions 800 \
  --seed 0 \
  --out results/sokoban_sd_world_model/option_target_signal_quick_seed0.json
```

输出 [option_target_signal_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/option_target_signal_quick_seed0.json)。option target signal gate 为 `FAIL`，失败项为 `valid_top1_agreement_min_0.40`。观测值为 train best option labels `33`、valid best option labels `14`、valid rank states `14`、valid top-1 agreement `0.2857142857142857`、valid Q corr `0.7794633473898425`、valid teacher-Q std `4.823039134104247`、train best target2 option labels `22`、valid teacher-best option state fraction `0.5`。对照基线中，`fixed_stock_return` valid top-1 agreement 为 `0.5714285714285714`，`exact_option_score` 为 `0.5`，`target2_structural_score` 为 `0.21428571428571427`。

显式 target2 option，first-push 执行模式，2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode expert \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2 \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_option_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_target2_option_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_mini_budget48_depth4_seed0.json)。成功率为 `0.0`，provider usage gate 为 `PASS`，planner feedback usage gate 为 `PASS`，action abstraction usage gate 为 `PASS`，`option_expansions=337`，`option_primitives=674`。

显式 target2 option，full-option 执行模式，2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode expert \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2 \
  --option-execution-mode full_option \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_option_full_exec_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_target2_option_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_full_exec_mini_budget48_depth4_seed0.json)。成功率仍为 `0.0`，provider usage gate 为 `PASS`，planner feedback usage gate 为 `PASS`，action abstraction usage gate 为 `PASS`，`option_expansions=281`，`option_primitives=562`。本节复现判定为 `ANALYZED`，当前 target2 option 消费不进入阶段 3 候选。

### TVA Truncation-Value Accuracy Audit

TVA 截断价值准确性审计代码检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_truncation_value.py
```

TVA 截断价值 smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_truncation_value.py \
  --max-eval-levels 8 \
  --max-states-per-level 6 \
  --teacher-max-expansions 800 \
  --out results/sokoban_sd_world_model/tva_truncation_value_audit_smoke_seed0.json
```

输出 [tva_truncation_value_audit_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_smoke_seed0.json)。审计合同记录 `teacher_usage=offline_state_label_only`、`teacher_used_online=false`、`best_first_used_online=false`、`planner_modified=false`、`transition_provider_modified=false`，并明确 `sdmcts_closed_loop_without_tva_is_truncation_bias_diagnostic_only=true`。

采样 `8` 个 stage-2 split 关卡，共 `46` 个状态，其中 train `34`、valid `12`；root state counts 为 `active=38`、`deadlocked=8`；teacher counts 为 `solved=35`、`proven_dead_end=8`、`teacher_budget_unsolved=3`。high-confidence gate 使用 `solved`、`proven_dead_end` 与 `no_legal_actions`，不把 `teacher_budget_unsolved` 当作高置信负例。

TVA truncation value gate 为 `FAIL`。基线方面，`stock_value` high-confidence MAE 为 `5.408689795334134`，`expert_value` high-confidence MAE 为 `5.125799790019716`。`sokoban_stage3_planner_tva_v2_quick` high-confidence MAE 为 `8.333739054839088`、corr 为 `0.951908759929304`、sign accuracy 为 `0.18604651162790697`、positive bias 为 `-8.761932504277338`。`sokoban_stage3_planner_tva_v2_expanded` high-confidence MAE 为 `7.423524176047102`、corr 为 `0.9683731373454699`、sign accuracy 为 `0.18604651162790697`、positive bias 为 `-7.659244627712222`。

本节复现判定为 `ANALYZED`。当前 stage3-planner TVA v2 在真实截断剩余价值上相关性高但负偏严重，且 MAE 差于 `expert_value` 基线，因此不进入新的 closed-loop SD-MCTS 调参。

### Option-Value Prior Planner Probe

option-value prior 代码检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py
```

first-push 执行模式，2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode expert \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2 \
  --option-value-prior-weight 0.1 \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_mini_budget48_depth4_seed0.json)。成功率为 `0.0`，provider usage gate、planner feedback usage gate、action abstraction usage gate 和 option value prior usage gate 均为 `PASS`；`option_value_prior_rows=3070`，`option_value_prior_positive_rows=1264`。

full-option 执行模式，2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode expert \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2 \
  --option-value-prior-weight 0.1 \
  --option-execution-mode full_option \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_target2_option_valueprior_w01_full_exec_mini_budget48_depth4_seed0.json)。成功率仍为 `0.0`，失败类型为 `max_pushes=2`；provider usage gate、planner feedback usage gate、action abstraction usage gate 和 option value prior usage gate 均为 `PASS`；`option_value_prior_rows=3168`，`option_value_prior_positive_rows=1350`。

本节复现判定为 `ANALYZED`。该 planner prior 已确认被在线消费，但未通过 2-level budget `48` 非零成功门限，因此不继续调权重，不进入 held-out `6` 扩展测试。

### Existing TVA / Terminal-Value Checkpoint Truncation Audit

已有 TVA 与 terminal-value checkpoint 的同口径截断价值审计：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_truncation_value.py \
  --max-eval-levels 8 \
  --max-states-per-level 6 \
  --teacher-max-expansions 800 \
  --tva-checkpoints \
  results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt,results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt,results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt,results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt \
  --out results/sokoban_sd_world_model/tva_truncation_value_audit_existing_seed0.json
```

输出 [tva_truncation_value_audit_existing_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_existing_seed0.json)。审计采样 `8` 个 stage-2 split 关卡，共 `46` 个状态，其中 high-confidence 样本 `43` 个。`stock_value` high-confidence MAE 为 `5.408689795334134`，`expert_value` 为 `5.125799790019716`。

主要 checkpoint 结果如下：`sokoban_tva_value_v1` high-confidence MAE 为 `1.2229815195856601`，corr 为 `0.9637950040821051`，但 positive bias 为 `-1.335852713344545`，未通过 bias 门限；`sokoban_stage3_planner_tva_v2_expanded` high-confidence MAE 为 `7.423524176047102`，positive bias 为 `-7.659244627712222`，未通过 MAE 与 bias 门限；`sokoban_teacher_q_state_value_v1_quick` 无失败项，high-confidence MAE 为 `0.5409354691144549`，corr 为 `0.9944298990860484`，sign accuracy 为 `1.0`，positive bias 为 `0.003251509906797726`，nonpositive bias 为 `0.8333237171173096`。

需注意，`sokoban_teacher_q_state_value_v1_quick` 的 checkpoint contract 为 `strict_search_statistics_tva=false`，target 为 `max_a Q_teacher(s,a)`，因此本节只能判定其为 terminal leaf probe 的截断价值校准通过，不能判定严格 TVA 通过。

为核对该较准 leaf value 与显式 option 消费是否能形成闭环成功，补跑 2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2 \
  --option-execution-mode full_option \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_v1_target2_full_exec_mini_budget48_depth4_seed0.json)。成功率仍为 `0.0`；provider usage gate、terminal value usage gate、planner feedback usage gate 和 action abstraction usage gate 均为 `PASS`，`tva_evaluations=3360`，`option_expansions=287`，`option_primitives=574`。

本节复现判定为 `ANALYZED`。结论为：非严格 teacher-Q 状态价值 probe 可以通过离线截断价值 gate，但当前 SD-MCTS 闭环仍未通过，严格 stage3 search-statistics TVA 仍需重新设计 target 或校准方式。

全部已有 TVA/terminal-value checkpoint 横向截断价值审计：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_truncation_value.py \
  --max-eval-levels 8 \
  --max-states-per-level 6 \
  --teacher-max-expansions 800 \
  --tva-checkpoints results/sokoban_sd_world_model/sokoban_tva_value_seed0.pt,results/sokoban_sd_world_model/sokoban_tva_v2_failed_relabel_seed0.pt,results/sokoban_sd_world_model/sokoban_search_value_tva_seed0.pt,results/sokoban_sd_world_model/sokoban_search_value_tva_posw4_seed0.pt,results/sokoban_sd_world_model/sokoban_search_value_tva_posw10_seed0.pt,results/sokoban_sd_world_model/sokoban_search_value_tva_stage3_td_v2_quick_seed0.pt,results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_quick_seed0.pt,results/sokoban_sd_world_model/sokoban_stage3_planner_tva_v2_expanded_seed0.pt,results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt \
  --out results/sokoban_sd_world_model/tva_truncation_value_audit_all_existing_smoke_seed0.json
```

输出 [tva_truncation_value_audit_all_existing_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_truncation_value_audit_all_existing_smoke_seed0.json)。整体 gate 因多个 checkpoint 失败而为 `FAIL`，但 `sokoban_teacher_q_state_value_v1_quick` 单独通过所有 provider-level checks：high-confidence MAE `0.5409354691144549`，corr `0.9944298990860484`，sign accuracy `1.0`，positive bias `0.003251509906797726`，nonpositive bias `0.8333237171173096`。

使用该 accuracy-gate passed terminal compensator 运行 2-level budget `48` smoke：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods learned_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json)。结果为 `0/2`，mean pushes `40.0`；provider usage gate 为 `PASS`，`learned_provider_fraction=1.0`；terminal value usage gate 为 `PASS`，TVA provider id 为 `sokoban_teacher_q_state_value_v1_quick`，TVA evaluations `3840`；planner feedback usage gate 为 `PASS`；action abstraction usage gate 为 `PASS`，`target2_prior_scored_roots=8`。本节复现判定为 `ANALYZED`，不扩展 held-out `6`。

### TVA Root Sorting and Oracle Provider Diagnostic

TVA root sorting 审计脚本检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_root_sorting.py
```

TVA root sorting 审计命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_root_sorting.py \
  --out results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json
```

输出 [tva_root_sorting_audit_teacherq_accuracygate_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_root_sorting_audit_teacherq_accuracygate_seed0.json)。审计使用 `stage3_sdmcts_teacherq_statevalue_accuracygate_t2prior_mini_budget48_depth4_seed0.json` 中两个 episode 的前 `5` 个 `trace_sample` 根状态，共 `10` 个 root。审计合约为 `teacher_usage=offline_successor_label_only`、`teacher_used_online=false`、`best_first_used_online=false`、`transition_provider_modified=false`、`planner_modified=false`。

主要结果：相对 `zero_leaf`，`teacherq_tva` 的 `mean_teacher_best_q_rank_improvement` 为 `1.4`，`q_teacher_corr` 均值从 `0.7418640397326196` 提升到 `0.8084293031527132`；相对 `stock_leaf`，名次改善为 `1.3`，`q_teacher_corr` 从 `0.7096232078500353` 提升到 `0.8084293031527132`。但 `q_best_in_teacher_best_tier` 均为 `6/10`，`critical_miss` 均为 `4/10`，说明 TVA 对排序方向有影响，但未净提升 root top-tier 选择。

一步备份分解显示，`one_step_provider_score` 的 root-level 平均 teacher corr 为 `0.7815805819105814`，候选级总体相关性为 `0.912`；但 `one_step_provider_best_in_teacher_best_tier_tva=0/10`。人工复核同一输出中的 exact stock readout 诊断显示，`one_step_exact_score` 的 top-1 有 `6/10` 次进入 teacher-best tier。该对比说明 learned provider 的 stock prediction 在局部备份中会显著改变动作排序。

为检查 provider 误差是否为唯一瓶颈，运行 oracle abstract transition 对照：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods oracle_abstract_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2_prior \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_t2prior_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_t2prior_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_t2prior_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，mean pushes `40.0`；terminal value usage gate 为 `PASS`，TVA evaluations `3840`；planner feedback usage gate 为 `PASS`；action abstraction usage gate 为 `PASS`，`target2_prior_scored_roots=10`。本节复现判定为 `ANALYZED`。oracle 对照说明 learned provider 误差不是唯一瓶颈，后续还需检查 root backup、先验和 option commitment。

Root backup scale 审计脚本检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_backup_scale.py
```

离线尺度扫描命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_tva_backup_scale.py \
  --out results/sokoban_sd_world_model/tva_backup_scale_audit_teacherq_accuracygate_seed0.json
```

输出 [tva_backup_scale_audit_teacherq_accuracygate_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/tva_backup_scale_audit_teacherq_accuracygate_seed0.json)。审计合约为 `offline_scale_scan_only=true`、`teacher_called=false`、`mcts_rerun=false`、`transition_provider_modified=false`、`planner_modified=false`。learned provider stock 下最佳组合为 `reward_scale=0.0`、`leaf_scale=0.25`，teacher-best tier 命中 `3/10`、critical miss `7/10`；exact stock readout 诊断下最佳组合为 `reward_scale=1.0`、`leaf_scale=1.25`，teacher-best tier 命中 `8/10`、critical miss `2/10`。因此本轮未新增在线 planner scale 参数。

显式 route commitment 对照：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/evaluate_stage3_sdmcts_planning.py \
  --checkpoint results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --use-checkpoint-valid-levels \
  --max-levels 30 \
  --max-eval-levels 2 \
  --methods oracle_abstract_sdmcts \
  --budgets 48 \
  --seeds 0 \
  --mcts-depth 4 \
  --terminal-value-mode tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_teacher_q_state_value_v1_quick_seed0.pt \
  --planner-feedback-mode expert \
  --action-abstraction-mode target2 \
  --option-execution-mode full_option \
  --root-action-selection value \
  --max-pushes 40 \
  --max-state-revisits 5 \
  --out results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_target2_full_exec_mini_budget48_depth4_seed0.json
```

输出 [stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_target2_full_exec_mini_budget48_depth4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/stage3_sdmcts_teacherq_statevalue_accuracygate_oracle_target2_full_exec_mini_budget48_depth4_seed0.json)。结果仍为 `0/2`，mean pushes `40.0`；terminal value usage gate 为 `PASS`，TVA evaluations `3504`；planner feedback usage gate 为 `PASS`；action abstraction usage gate 为 `PASS`，`option_expansions=350`、`option_primitives=700`。本节复现判定为 `ANALYZED`。

### Provider Stock Calibration and Route-Goal Commitment Option Audit

Provider stock calibration 审计脚本检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_provider_stock_calibration.py
```

离线校准审计命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_provider_stock_calibration.py \
  --out results/sokoban_sd_world_model/provider_stock_calibration_audit_seed0.json
```

输出 [provider_stock_calibration_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/provider_stock_calibration_audit_seed0.json)。审计合约为 `train_split_only_calibrator_fit=true`、`teacher_called=false`、`mcts_rerun=false`、`transition_provider_modified=false`、`planner_modified=false`、`calibrator_online_enabled=false`。校准使用 `16` 个 train levels、`2570` 条 transition。root 审计结果为：raw provider teacher-best tier 命中 `0/10`、critical miss `10/10`；calibrated provider 命中 `1/10`、critical miss `9/10`；exact stock readout 命中 `6/10`、critical miss `4/10`。本节复现判定为 `ANALYZED`，不进入闭环测试。

Route-goal commitment option 审计脚本检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_route_goal_commitment_options.py
```

quick 审计命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_route_goal_commitment_options.py \
  --out results/sokoban_sd_world_model/route_goal_commitment_option_audit_seed0.json
```

输出 [route_goal_commitment_option_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/route_goal_commitment_option_audit_seed0.json)。审计合约为 `teacher_usage=offline_option_successor_label_only`、`teacher_used_online=false`、`best_first_used_online=false`、`mcts_rerun=false`、`planner_modified=false`、`transition_provider_modified=false`、`candidate_generation_uses_teacher=false`。结果为 gate `FAIL`，失败项为 `route_goal_beats_nonroute_roots_min_2`。观测值为：root count `10`，route-goal candidate root fraction `1.0`，route-goal beats nonroute roots `0`，route-only structural top-1 tier hit rate `0.3`，route-only structural first-tier hit rate `0.8`。

扩展长度审计命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_route_goal_commitment_options.py \
  --max-route-length 6 \
  --max-route-options-per-root 32 \
  --out results/sokoban_sd_world_model/route_goal_commitment_option_audit_len6_seed0.json
```

输出 [route_goal_commitment_option_audit_len6_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/route_goal_commitment_option_audit_len6_seed0.json)。结果与 quick 版一致：gate `FAIL`，失败项为 `route_goal_beats_nonroute_roots_min_2`，route-goal candidate root fraction `1.0`，route-goal beats nonroute roots `0`，route-only structural top-1 tier hit rate `0.3`，route-only structural first-tier hit rate `0.8`。本节复现判定为 `ANALYZED`，不进入在线 SD-MCTS smoke。

### First-Push Commitment Signal Audit

First-push commitment signal 审计脚本检查：

```bash
conda run -n sdmcts_p100 python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_first_push_commitment_signal.py
```

holdout-50 smoke 命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_first_push_commitment_signal.py \
  --max-train-levels 8 \
  --internal-holdout-fraction 0.5 \
  --max-states-per-level 8 \
  --max-action-states-per-level 3 \
  --teacher-max-expansions 300 \
  --max-route-options-per-root 8 \
  --max-target2-options-per-root 4 \
  --out results/sokoban_sd_world_model/first_push_commitment_signal_smoke_holdout50_seed0.json
```

输出 [first_push_commitment_signal_smoke_holdout50_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/first_push_commitment_signal_smoke_holdout50_seed0.json)。gate 为 `PASS`；valid rank states `6`，valid top-tier hit `0.5`，valid top-1 exact agreement `0.5`，valid Q corr `0.8357902171784326`，valid route-goal available fraction `0.46153846153846156`。

受控扩展审计命令：

```bash
conda run -n sdmcts_p100 python \
  experiments/sokoban_sd_world_model/audit_sokoban_first_push_commitment_signal.py \
  --max-train-levels 12 \
  --internal-holdout-fraction 0.5 \
  --max-states-per-level 8 \
  --max-action-states-per-level 4 \
  --teacher-max-expansions 300 \
  --max-route-options-per-root 8 \
  --max-target2-options-per-root 4 \
  --out results/sokoban_sd_world_model/first_push_commitment_signal_audit12_holdout50_seed0.json
```

输出 [first_push_commitment_signal_audit12_holdout50_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/first_push_commitment_signal_audit12_holdout50_seed0.json)。审计合约为 `teacher_usage=offline_option_successor_label_only`、`teacher_used_online=false`、`best_first_used_online=false`、`planner_modified=false`、`transition_provider_modified=false`、`candidate_options=primitive_pushes_plus_target2_plus_route_goal_monotone_options`。结果为 gate `PASS`，无失败项；train best first-push labels `25`，valid best first-push labels `25`，valid rank states `17`，valid top-tier hit `0.47058823529411764`，valid Q corr `0.8319232992067614`。完整比较显示，route-goal structural score 的 valid top-tier hit 为 `0.7647058823529411`，高于当前 ridge commitment score。本节复现判定为 `ANALYZED`，不进入在线 SD-MCTS smoke。

## 2026-06-11 Sokoban exact-grid SD / TVA / SD-MCTS 重构复现记录

语法检查：

```bash
python -m py_compile \
  src/sokoban/sd_model.py \
  src/sokoban/__init__.py \
  experiments/sokoban_sd_world_model/validate_sokoban_grid_stock_flow_design.py \
  experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

结果：退出码 `0`。

SD 独立门禁 smoke：

```bash
python experiments/sokoban_sd_world_model/validate_sokoban_grid_stock_flow_design.py \
  --max-levels 3 \
  --max-depth 1 \
  --max-states-per-level 20 \
  --sample-limit 10 \
  --out results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json
```

输出 [grid_stock_flow_design_smoke_quick.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json)。结果为 `grid_sd_gate=PASS`，levels checked `3`，states checked `15`，transitions checked `64`，reachable players checked `308`。`box_mass_violations`、`player_mass_violations`、`flow_onehot_violations`、`box_balance_violations`、`step_mismatch_violations` 和 `canonical_mismatches` 均为 `0`。

TVA 独立门禁 quick 覆盖检查：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  --max-levels 6 \
  --state-depth 1 \
  --max-states-per-level 5 \
  --max-action-states-per-level 2 \
  --teacher-max-expansions 300 \
  --out results/sokoban_sd_world_model/grid_tva_independent_audit_quick_seed0.json
```

输出 [grid_tva_independent_audit_quick_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_tva_independent_audit_quick_seed0.json)。结果为 `tva_independent_gate=FAIL`，失败项为 `valid_sample_count`；valid count 为 `7`。该结果用于记录 quick 配置覆盖不足，不作为 TVA 性能失败结论。

TVA 独立门禁 smoke 与 checkpoint 导出：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  --max-levels 8 \
  --state-depth 1 \
  --max-states-per-level 8 \
  --max-action-states-per-level 2 \
  --teacher-max-expansions 300 \
  --out results/sokoban_sd_world_model/grid_tva_independent_audit_smoke_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json
```

输出 [grid_tva_independent_audit_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_tva_independent_audit_smoke_seed0.json) 和 [grid_tva_ridge_checkpoint_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json)。结果为 `tva_independent_gate=PASS`，valid count `16`，valid MAE `1.3878170000076102`，valid corr `0.9736159851420181`，sign accuracy `1.0`，root sorting top-tier hit `5/5`。

Exact-grid SD-MCTS zero-leaf smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --max-levels 3 \
  --horizon 3 \
  --budget 24 \
  --max-steps 30 \
  --leaf-value zero \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_zero_smoke_seed0.json
```

输出 [grid_sdmcts_zero_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_zero_smoke_seed0.json)。结果为 levels evaluated `3`，solved count `1`，success rate `0.3333333333333333`，mean pushes solved `11.0`，status counts 为 `solved=1`、`max_steps=2`。

Exact-grid SD-MCTS TVA-leaf smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --max-levels 3 \
  --horizon 3 \
  --budget 24 \
  --max-steps 30 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/grid_tva_independent_audit_smoke_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_tva_smoke_seed0.json
```

输出 [grid_sdmcts_tva_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_tva_smoke_seed0.json)。结果为 levels evaluated `3`，solved count `1`，success rate `0.3333333333333333`，mean pushes solved `23.0`，status counts 为 `solved=1`、`max_steps=1`、`no_action=1`。本节复现判定为 `ANALYZED`：closed-loop 接口、独立门禁读取和 TVA checkpoint 消费均已验证，但当前 smoke 不支持 TVA 优于 zero-leaf 的性能结论。

## 2026-06-11 Sokoban exact-grid SD-MCTS holdout 小矩阵

为避免评测集与 TVA smoke checkpoint 的训练关卡重叠，`evaluate_sokoban_grid_sdmcts.py` 增加 `--level-offset` 参数。语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

结果：退出码 `0`。

holdout 配置使用 `--level-offset 5 --max-levels 3`，对应当前 TVA 审计 `max_levels=8`、`holdout_fraction=0.35` 的 valid slice。所有闭环命令均启用 `--require-independent-gates`。zero-leaf 命令模板：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 5 \
  --max-levels 3 \
  --horizon H \
  --budget B \
  --max-steps 60 \
  --leaf-value zero \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout_zero_hH_bB_seed0.json
```

TVA-leaf 命令模板：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 5 \
  --max-levels 3 \
  --horizon H \
  --budget B \
  --max-steps 60 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_smoke_seed0.json \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/grid_tva_independent_audit_smoke_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout_tva_hH_bB_seed0.json
```

实际运行矩阵与输出：

| H | budget | leaf | output | solved / levels | status counts |
|---:|---:|---|---|---:|---|
| 2 | 24 | zero | [grid_sdmcts_holdout_zero_h2_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_zero_h2_b24_seed0.json) | `0/3` | `deadlocked=2, no_action=1` |
| 2 | 24 | tva | [grid_sdmcts_holdout_tva_h2_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_tva_h2_b24_seed0.json) | `0/3` | `max_steps=2, no_action=1` |
| 2 | 48 | zero | [grid_sdmcts_holdout_zero_h2_b48_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_zero_h2_b48_seed0.json) | `0/3` | `deadlocked=2, no_action=1` |
| 2 | 48 | tva | [grid_sdmcts_holdout_tva_h2_b48_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_tva_h2_b48_seed0.json) | `0/3` | `max_steps=2, no_action=1` |
| 4 | 24 | zero | [grid_sdmcts_holdout_zero_h4_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_zero_h4_b24_seed0.json) | `0/3` | `deadlocked=1, max_steps=1, no_action=1` |
| 4 | 24 | tva | [grid_sdmcts_holdout_tva_h4_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_tva_h4_b24_seed0.json) | `0/3` | `max_steps=2, no_action=1` |
| 4 | 48 | zero | [grid_sdmcts_holdout_zero_h4_b48_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_zero_h4_b48_seed0.json) | `0/3` | `max_steps=3` |
| 4 | 48 | tva | [grid_sdmcts_holdout_tva_h4_b48_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_tva_h4_b48_seed0.json) | `0/3` | `max_steps=3` |

补充强点使用 `--max-steps 80`：

| H | budget | leaf | output | solved / levels | status counts |
|---:|---:|---|---|---:|---|
| 6 | 96 | zero | [grid_sdmcts_holdout_zero_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_zero_h6_b96_seed0.json) | `0/3` | `max_steps=3` |
| 6 | 96 | tva | [grid_sdmcts_holdout_tva_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout_tva_h6_b96_seed0.json) | `0/3` | `max_steps=3` |

本节复现判定为 `ANALYZED`。完整框架已可在 heldout slice 上评测，但当前小矩阵没有产生 heldout 成功率，也没有显示 TVA 相对 zero-leaf 的成功率增益。

## 2026-06-12 SD-heuristic 叶值、重访惩罚与 TVA v2 复现记录

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py
```

结果：退出码 `0`。

holdout 三关（`--level-offset 5 --max-levels 3`，seed `0`，均启用 `--require-independent-gates`，SD 门禁文件同前）。`sd_heuristic` 叶值：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 5 --max-levels 3 --horizon 6 --budget 96 --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout_sdh_h6_b96_seed0.json
```

叠加重访惩罚时追加 `--revisit-penalty 1.0`，`H=4` 配置使用 `--budget 48 --max-steps 60`。结果矩阵：

| leaf | revisit | H=4,b=48 | H=6,b=96 | output 前缀 |
|---|---:|---:|---:|---|
| zero | 0 | `0/3` | `0/3` | `grid_sdmcts_holdout_zero_*`（前一轮） |
| sd_heuristic | 0 | `0/3`（`no_action=2`） | `1/3`（19 pushes） | `grid_sdmcts_holdout_sdh_*` |
| sd_heuristic | 1.0 | `1/3`（19 pushes） | `1/3`（19 pushes） | `grid_sdmcts_holdout_sdh_rp1_*` |
| tva v2 value | 1.0 | `0/3` | `0/3`（含 deadlocked=1） | `grid_sdmcts_holdout_tvaV2val_rp1_*` |
| tva v2 residual | 1.0 | `0/3` | `0/3`（含 deadlocked=1） | `grid_sdmcts_holdout_tvaV2res_rp1_*` |

回归检查：`--revisit-penalty` 默认 `0` 下重跑 `sdh_h6_b96`，aggregate 与改动前输出逐项一致（`grid_sdmcts_holdout_sdh_h6_b96_seed0_regression.json`）。

TVA v2 训练与审计（训练关卡 8-47，与 heldout 5-7 不相交）：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  --level-offset 8 --max-levels 40 --sample-mode random_walk --state-depth 16 \
  --max-states-per-level 24 --teacher-max-expansions 1000 --seed 0 --target-mode value \
  --out results/sokoban_sd_world_model/grid_tva_independent_audit_v2_value_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_v2_value_seed0.json
```

residual 模式将 `--target-mode` 改为 `residual`，输出文件名中 `value` 换为 `residual`。两次审计 gate 均 `PASS`；样本 `960`（teacher 解出 `262`、budget 未解 `657`、no_legal_actions `41`），valid corr value `0.532` / residual `0.617`，depth 9+ 切片 corr `0.707` / `0.771`。

量纲守卫验证：闭环以 `--gamma 0.9` 消费 `gamma=0.95` 的 v2 checkpoint 时按预期拒绝（`TVA checkpoint value_scale mismatches runtime args`）；匹配量纲时正常运行。

本节复现判定为 `ANALYZED`。`sd_heuristic + revisit_penalty=1.0` 取得 heldout 首批解（两配置各 `1/3`，同关 19 pushes）；TVA v2 离线门禁通过但闭环 `0/3` 劣于零学习基线，记录为负结果，归因为 teacher 预算未解样本（`657/960`）的常数标签淹没结构信号。

### 2026-06-12 补充：TVA v3 足额预算检验

teacher A* 改 canonical key 去重并修复 frontier 穷尽标签语义（`proven_dead_end` 与 `teacher_budget_unsolved` 分离）后，以 `--teacher-max-expansions 20000` 重训：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  --level-offset 8 --max-levels 40 --sample-mode random_walk --state-depth 16 \
  --max-states-per-level 24 --teacher-max-expansions 20000 --seed 0 --target-mode value \
  --out results/sokoban_sd_world_model/grid_tva_independent_audit_v3_value_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_v3_value_seed0.json
```

residual 模式同上改 `--target-mode residual` 与文件名。teacher 计数：solved `307`、proven_dead_end `612`、no_legal_actions `41`、budget_unsolved `0`；耗时约 `583 s`/模式。闭环（`--revisit-penalty 1.0`，`H=4,b=48` 与 `H=6,b=96`）value/residual 均 `0/3`，输出 `grid_sdmcts_holdout_tvaV3{value,residual}_rp1_*_seed0.json`。teacher 直接求解 holdout 三关初始态：`109/35/26` 次扩展、解长 `19/11/19` pushes（50000 预算上限，远未用满）。本节判定 `ANALYZED`：足额预算下 ridge TVA 闭环仍劣于 sd_heuristic 基线，瓶颈重定位为学习器容量与 MCTS 搜索形态。

### 2026-06-12 补充：搜索形态消融与 20 关预算扩展

闭环入口新增 `--transposition` 与 `--reuse-tree`（默认关闭）。回归检查：`sd_heuristic` 无惩罚 `H=6,b=96` 三关复跑与既有结果 aggregate 及 trace 动作序列逐项一致。20 关切片命令样例：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 48 --max-levels 20 --horizon 10 --budget 512 --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout20_sdh_rp1_h10_b512_seed0.json
```

结果（输出前缀 `grid_sdmcts_holdout20_*_seed0.json`）：

| arm | H=6,b=96 | H=8,b=256 | H=10,b=512 |
|---|---:|---:|---:|
| zero | `3/20` | `1/20` | – |
| sdh | `2/20` | – | – |
| sdh_rp1 | `2/20` | `4/20` | `5/20` |
| sdh_rp1 + transposition | `2/20` | – | – |
| sdh_rp1 + reuse-tree | `2/20` | – | – |

本节判定 `ANALYZED`：sd_heuristic+rp1 成功率随 H×budget 单调上升且零死锁，zero-leaf 不随预算扩展；transposition/树复用在 b=96 下无可检测差异。

### 2026-06-12 补充：算力轴扫描与机制高预算复测

命令样例（其余配置仅改 `--horizon/--budget/--seed` 与机制旗标）：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 48 --max-levels 20 --horizon 16 --budget 2048 --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout20_sdh_rp1_tp_h16_b2048_seed0.json
```

20 关结果（sd_heuristic+rp1，seed 0，括号为 total_deadlock_prunes）：

| config | 裸臂 | +transposition | +reuse-tree |
|---|---:|---:|---:|
| H=6,b=96 | `2/20` | `2/20` | `2/20` |
| H=8,b=256 | `4/20` | – | – |
| H=10,b=512 | `5/20`（seed1/2 为 `6/20`） | – | – |
| H=12,b=1024 | `5/20`（583998） | `7/20`（42527） | `7/20`（7446） |
| H=16,b=2048 | `10/20`（803549） | `12/20`（66397） | `9/20`（11590） |

zero-leaf 对照：`H=6,b=96` `3/20`、`H=8,b=256` `1/20`，不随预算扩展。本节判定 `ANALYZED`：最强配置为 sdh_rp1+transposition @ H=16,b=2048（`12/20`），输出文件前缀 `grid_sdmcts_holdout20_*`。

### 2026-06-12 补充：TVA v4 MLP 训练与闭环负结果

训练命令（residual 模式；value 模式改 `--target-mode value` 与文件名）：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  --level-offset 8 --max-levels 40 --sample-mode random_walk --state-depth 16 \
  --max-states-per-level 48 --teacher-max-expansions 20000 --seed 0 \
  --target-mode residual --model mlp \
  --out results/sokoban_sd_world_model/grid_tva_independent_audit_v4_residual_mlp_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/grid_tva_ridge_checkpoint_v4_residual_mlp_seed0.json
```

闭环（rp1+transposition）结果：residual `H=10,b=512` `1/20`、`H=16,b=2048` `2/20`（no_action 5/11）；value `H=16,b=2048` `3/20`。输出前缀 `grid_sdmcts_holdout20_tvaV4{res,val}_rp1_tp_*`。对照 sd_heuristic+rp1+tp `H=16,b=2048` 为 `12/20`。本节判定 `ANALYZED`：v4 负结果，TVA-as-leaf 形态在当前特征集被证伪，离线 gate 与闭环表现脱钩再次确认。

### 2026-06-12 补充：算力上限与 50 关定稿

命令样例（50 关最优点）：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 48 --max-levels 50 --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout50_sdh_rp1_tp_h20_b4096_seed0.json
```

结果：20 关 `H=20,b=4096` `15/20`、`H=24,b=8192` `13/20`（deadlocked 3，曲线见顶回落）；50 关 `H=16,b=2048` `25/50`、`H=20,b=4096` `37/50`（deadlocked 1、max_steps 11）。输出前缀 `grid_sdmcts_holdout{20,50}_sdh_rp1_tp_*`。本节判定 `ANALYZED`，主结果为 50 关 74%。

### 2026-06-13 补充：SD action prior（PUCT）

在 50 关最优点基础上加 `--action-prior sd_delta`（默认 `none` 保基线 bit-identical）。温度扫描命令样例：

```bash
for T in 0.5 1.0 2.0; do
python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --level-offset 48 --max-levels 50 --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --action-prior sd_delta --prior-temperature ${T} --prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_holdout50_sdh_rp1_tp_prior_t${T/./p}_h20_b4096_seed0.json
done
```

结果：`τ∈{0.5,1.0,2.0}` 一致 `41/50`（vs baseline `37/50`），耗时约 `0.58×` baseline（27.8→16.1 min），失败模式收敛（max_steps 11→≤5）；`τ=2.0` deadlock 回落至 1。`τ=1.0` 默认输出文件名为 `..._prior_h20_b4096_seed0.json`（无温度后缀）。prior 臂对 seed 确定性（路径无 `rng`），故以温度扫描而非多 seed 论证稳健性。基线 bit-identical 经 `--action-prior none` 对 `grid_sdmcts_holdout20_sdh_rp1_tp_h6_b96_seed0.json` 的 20 关 trace 回归确认（`mismatches=0`）。本节判定 `ANALYZED`，新最强配置 50 关 `82%`。

### 2026-06-13 补充：GFT-SD 两层 OO 架构重建

新增文件：

1. [gft_sd.py](/home/data/round/code/sd_world_model_rl/src/sokoban/gft_sd.py)
2. [evaluate_sokoban_gft_sdmcts.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py)

导出更新：[__init__.py](/home/data/round/code/sd_world_model_rl/src/sokoban/__init__.py) 新增 `SokobanGFTSDModel`、`GFTResourceStock`、`GFTResourceValue`、`GFTStateReadout`、八个资源库存类和 `DEFAULT_GFT_PRIOR_WEIGHTS`。

语法检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

资源层导入与 prior 归一化检查：

```bash
python - <<'PY'
from src.sokoban import SokobanGFTSDModel, load_boxoban_levels
levels = load_boxoban_levels('datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt', max_levels=1)
model = SokobanGFTSDModel(levels[0])
state = model.initial_state
print(model.provider_id, model.transition_provider_id)
print(sorted(model.gft_state(state).to_dict()['resources']))
legal = model.legal_flows(state)
rows = model.gft_action_prior_rows(state, [(a, model.step_grid_flow(state, a)) for a in legal[:3]], temperature=2.0)
print(len(rows), [round(r['prior'], 6) for r in rows])
PY
```

输出：

```text
gft_sd_v1 exact_grid_stock_flow_v1
['A', 'C', 'D', 'I', 'K', 'M', 'P', 'Q']
3 [0.173159, 0.41342, 0.41342]
```

极小闭环 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --max-levels 1 --horizon 2 --budget 8 --max-steps 5 --seed 0 \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --trace-gft-readouts \
  --out results/sokoban_sd_world_model/gft_sdmcts_smoke_level1_h2_b8_seed0.json
```

输出 [gft_sdmcts_smoke_level1_h2_b8_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_smoke_level1_h2_b8_seed0.json)。aggregate 为 levels evaluated `1`、solved count `0`、success rate `0.0`、total deadlock prunes `4`、status counts `max_steps=1`。该 smoke 仅验证独立 GFT-SD 入口可运行、合同字段可落盘、资源层可记录；不作为性能结论。

输出合同记录 `sd_provider=gft_sd_v1`、`transition_provider=exact_grid_stock_flow_v1`、`resource_state=gft_phi_y_readout`、`gft_resource_layer_replaces_transition_provider=false`、`legacy_six_stock_rollout_used=false`、`teacher_used_online=false`。该入口初建时 inactive resources 为 `K/Q/C`；后续 `K`、`Q` 与 `C` 已分别作为默认关闭的 active readout 接入，当前 inactive resources 为空列表。

正式 50 关等价复现测试：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 50 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。aggregate 为 levels evaluated `50`、solved count `41`、success rate `0.82`、mean pushes solved `15.146341463414634`、total deadlock prunes `174430`，status counts 为 `deadlocked=1`、`max_steps=5`、`no_action=3`、`solved=41`。`framework_contract` 仍记录 `sd_provider=gft_sd_v1`、`transition_provider=exact_grid_stock_flow_v1`、`gft_resource_layer_replaces_transition_provider=false`、`legacy_six_stock_rollout_used=false`、`teacher_used_online=false`；`independent_gate_status.failed_checks=[]`。

与 [grid_sdmcts_holdout50_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_holdout50_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0.json) 的比较结果为：aggregate 完全一致；逐关 `level_index/status/solved/pushes/steps_executed/deadlock_prunes/final_features` mismatch 数为 `0`；trace 中 `action/root_visits/root_value/top_children` mismatch 数为 `0`。因此该测试作为 GFT-SD 两层入口的正式等价复现 gate 记为 `PASS`。

行为保持型效率修订后，正式 50 关重测命令为：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 50 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json
```

输出 [gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json)。aggregate 仍为 `41/50`、success rate `0.82`、total deadlock prunes `174430`；与既有 grid 50 关输出比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。`/usr/bin/time` 输出为 `real 2102.34`、`user 2103.02`、`sys 5.45`；JSON 内逐关 elapsed 汇总为 `2099.7606648490764` 秒。

同机顺序 10 关正式预算效率重测：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --action-prior sd_delta --prior-temperature 2.0 --prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/grid_sdmcts_gate10_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0_retime.json

/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json
```

grid 输出 [grid_sdmcts_gate10_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0_retime.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/grid_sdmcts_gate10_sdh_rp1_tp_prior_t2p0_h20_b4096_seed0_retime.json)，`real 644.04`、JSON elapsed 汇总 `641.7514883917756` 秒；GFT 输出 [gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json)，`real 641.79`、JSON elapsed 汇总 `639.5067871282808` 秒。两者 aggregate 均为 `6/10`，逐关字段和 root trace mismatch 均为 `0`。该同机重测用于修正效率解释：默认 GFT 入口未显示相对 grid 入口的系统性额外开销；跨时段 full-run wall time 不单独作为架构开销证据。

### 2026-06-13 补充：K 区域容量裕度第一切片

语法检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  src/sokoban/__init__.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

`K` readout 与 prior 检查：

```bash
python - <<'PY'
from src.sokoban import SokobanGFTSDModel, load_boxoban_levels
level = load_boxoban_levels('datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt', max_levels=1)[0]
model = SokobanGFTSDModel(level)
state = model.initial_state
resources = model.gft_state(state).to_dict()['resources']
print(resources['K']['active'], resources['K']['implemented'], resources['K']['value'])
legal = model.legal_flows(state)
candidates = [(a, model.step_grid_flow(state, a)) for a in legal[:3]]
rows = model.gft_action_prior_rows(state, candidates, temperature=2.0)
rows_k = model.gft_action_prior_rows(state, candidates, weights={'capacity': 1.0}, temperature=2.0)
print([round(r['prior'], 6) for r in rows], [round(r['components']['capacity_delta'], 6) for r in rows])
print([round(r['prior'], 6) for r in rows_k], [round(r['components']['capacity_delta'], 6) for r in rows_k])
PY
```

关键输出为 `K active=True`、`implemented=True`；默认 prior 为 `[0.173159, 0.41342, 0.41342]` 且 `capacity_delta=[0.0,0.0,0.0]`；`capacity_weight=1.0` 时同一候选 prior 为 `[0.24276,0.37862,0.37862]`，对应 `capacity_delta=[0.0,-1.0,-1.0]`。

默认路径 3 关等价 gate：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 3 --horizon 6 --budget 96 \
  --max-steps 30 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json
```

输出 [gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与旧 GFT 3 关输出比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。

10 关正式预算开发门槛：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-capacity-weight 0.25 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 levels evaluated `10`、solved count `7`、success rate `0.7`、mean pushes solved `16.714285714285715`、total deadlock prunes `41994`，status counts 为 `deadlocked=1`、`max_steps=1`、`no_action=1`、`solved=7`。`/usr/bin/time` 输出为 `real 540.73`、`user 542.90`、`sys 4.85`。相对于同机默认 GFT 10 关 `6/10`，新增解出 `level_index=5`，未新增 deadlocked 类型。

曾启动同配置 50 关正式测试，但因单轮耗时过高主动停止；目标输出 [gft_sdmcts_holdout50_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) 未落盘。该中止运行不作为结果证据。后续 `K/Q/C` 资源开发以内循环 10 关正式预算为门槛，50 关只用于阶段性确认。

### 2026-06-13 补充：Q 顺序依赖债务第一切片

语法与格式检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  src/sokoban/__init__.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py

git diff --check
```

初始动作覆盖审计：

```bash
python -c "from src.sokoban import SokobanGFTSDModel, load_boxoban_levels; levels=load_boxoban_levels('datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt', max_levels=98)[48:98]; total=0; changed=0; levels_changed=[]; constraints=[]; max_delta=0; init_q=[];\
for i,lvl in enumerate(levels):\
    m=SokobanGFTSDModel(lvl); s=m.initial_state; parent=m.gft_ordering_debt_cached(s); c=[(a,m.step_grid_flow(s,a)) for a in m.legal_flows(s)]; deltas=[m.gft_ordering_debt_cached(ch)-parent for _,ch in c]; total+=len(deltas); nz=sum(1 for d in deltas if d!=0); changed+=nz; init_q.append(parent); constraints.append((len(m.gft_ordering_constraints['goal_order_edges']), len(m.gft_ordering_constraints['entry_order_edges']))); max_delta=max(max_delta, max([abs(d) for d in deltas], default=0));\
    if nz: levels_changed.append((i,nz,len(deltas),deltas));\
print({'levels':len(levels),'initial_actions':total,'nonzero_actions':changed,'levels_with_nonzero':len(levels_changed),'max_abs_delta':max_delta,'init_q_nonzero':sum(1 for q in init_q if q!=0),'constraint_levels':sum(1 for g,e in constraints if g+e>0)})"
```

关键输出为 `levels=50`、`initial_actions=223`、`nonzero_actions=60`、`levels_with_nonzero=38`、`max_abs_delta=3.0`、`init_q_nonzero=21`、`constraint_levels=46`。

默认路径 3 关等价 gate：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 3 --horizon 6 --budget 96 \
  --max-steps 30 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json
```

输出 [gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与 [gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_kreadout_v2_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。

10 关正式预算开发门槛，`ordering_weight=0.25`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.25 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_qord0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_qord0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_qord0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `6/10`，status counts 为 `deadlocked=2`、`max_steps=1`、`no_action=1`、`solved=6`，`/usr/bin/time` 为 `real 607.10`、`user 609.16`、`sys 5.01`。相对默认 GFT 10 关，`level_index=5` 从 `max_steps` 退化为 `deadlocked`。

10 关正式预算开发门槛，`ordering_weight=0.1`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_qord0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_qord0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_qord0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `6/10`，status counts 为 `deadlocked=2`、`max_steps=2`、`solved=6`，`/usr/bin/time` 为 `real 697.54`、`user 699.50`、`sys 5.03`。相对默认 GFT 10 关，`level_index=5` 从 `max_steps` 退化为 `deadlocked`，`level_index=6` 从 `no_action` 变为 `max_steps`。该结果不满足“至少不劣于默认线且无新增死锁”的 10 关开发门槛。

### 2026-06-13 补充：GFT 资源统一先验接口

语法检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  src/sokoban/__init__.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

默认路径 3 关等价 gate：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 3 --horizon 6 --budget 96 \
  --max-steps 30 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json
```

输出 [gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，aggregate 为 `0/3`，status counts 为 `max_steps=2`、`no_action=1`。与 [gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_qreadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。该检查用于确认 `GFTSDMCTSPlanner` 统一调用 `SokobanGFTSDModel.gft_action_prior_rows()` 后，默认行为保持等价。

### 2026-06-13 补充：K+Q 组合交互测试

10 关正式预算组合测试：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-capacity-weight 0.25 --gft-ordering-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_qord0p05_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_kcap0p25_qord0p05_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_qord0p05_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `6/10`，status counts 为 `deadlocked=1`、`max_steps=2`、`no_action=1`、`solved=6`，`/usr/bin/time` 为 `real 662.41`、`user 664.27`、`sys 5.21`。相对默认 GFT 10 关，solved/status/pushes 无变化；相对 `K` 单独配置，`level_index=5` 从 `solved`、`24` pushes 退化为 `max_steps`。

### 2026-06-13 补充：C 目标承诺清晰度第一切片

语法检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  src/sokoban/__init__.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

初始动作覆盖审计：

```bash
python -c "from src.sokoban import SokobanGFTSDModel, load_boxoban_levels; levels=load_boxoban_levels('datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt', max_levels=98)[48:98]; total=0; changed=0; levels_changed=[]; vals=[]; deltas_all=[];\
for i,lvl in enumerate(levels):\
    m=SokobanGFTSDModel(lvl); s=m.initial_state; parent=m.gft_commitment_confidence_cached(s); vals.append(parent); c=[(a,m.step_grid_flow(s,a)) for a in m.legal_flows(s)]; deltas=[m.gft_commitment_confidence_cached(ch)-parent for _,ch in c]; total+=len(deltas); nz=sum(1 for d in deltas if abs(d)>1e-9); changed+=nz; deltas_all.extend(deltas);\
    if nz: levels_changed.append((i,nz,len(deltas),[round(d,4) for d in deltas]));\
print({'levels':len(levels),'initial_actions':total,'nonzero_actions':changed,'levels_with_nonzero':len(levels_changed),'init_conf_min':round(min(vals),6),'init_conf_max':round(max(vals),6),'delta_min':round(min(deltas_all),6),'delta_max':round(max(deltas_all),6)})"
```

关键输出为 `levels=50`、`initial_actions=223`、`nonzero_actions=155`、`levels_with_nonzero=32`、`init_conf_min=0.080029`、`init_conf_max=0.850247`、`delta_min=-0.731939`、`delta_max=0.336704`。

默认路径 3 关等价 gate：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 3 --horizon 6 --budget 96 \
  --max-steps 30 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json
```

输出 [gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)，与 [gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_unifiedprior_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。

10 关正式预算开发门槛，`commitment_weight=0.5`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-commitment-weight 0.5 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_ccommit0p50_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_ccommit0p50_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_ccommit0p50_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `6/10`，status counts 为 `deadlocked=2`、`max_steps=1`、`no_action=1`、`solved=6`，`/usr/bin/time` 为 `real 616.46`、`user 618.09`、`sys 5.41`。相对默认 GFT 10 关，`level_index=5` 从 `max_steps` 退化为 `deadlocked`。

10 关正式预算开发门槛，`commitment_weight=0.25`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-commitment-weight 0.25 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `6/10`，status counts 为 `deadlocked=2`、`max_steps=2`、`solved=6`，`/usr/bin/time` 为 `real 716.56`、`user 718.66`、`sys 4.84`。相对默认 GFT 10 关，`level_index=5` 从 `max_steps` 退化为 `deadlocked`，`level_index=6` 从 `no_action` 变为 `max_steps`。该结果不满足“至少不劣于默认线且无新增死锁”的 10 关开发门槛。

10 关正式预算组合测试，`capacity_weight=0.25`、`commitment_weight=0.25`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-capacity-weight 0.25 --gft-commitment-weight 0.25 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_kcap0p25_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_ccommit0p25_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `7/10`，status counts 为 `deadlocked=1`、`max_steps=1`、`no_action=1`、`solved=7`，`/usr/bin/time` 为 `real 557.08`、`user 558.40`、`sys 4.55`。相对 `K` 单独配置，solved/status/pushes 无差异；相对 `C=0.25` 单独配置，`level_index=5` 从 `deadlocked` 恢复为 `solved`、`24` pushes，`level_index=6` 从 `max_steps` 恢复为 `no_action`。

### 2026-06-14 补充：GFT-SD 完整结构对齐实现与 10 关测试

语法检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  src/sokoban/__init__.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

结构审计：

```bash
python - <<'PY'
import math
from src.sokoban import SokobanGFTSDModel, load_boxoban_levels
levels = load_boxoban_levels('datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt', max_levels=53)[48:53]
checked = 0
nonfinite = []
for li, level in enumerate(levels):
    model = SokobanGFTSDModel(level)
    states = [model.initial_state]
    for action in model.legal_flows(model.initial_state)[:4]:
        states.append(model.step_grid_flow(model.initial_state, action))
    for state in states:
        checked += 1
        z = model.gft_resource_vector(state)
        value = model.gft_sd_value(state)
        feats = model.gft_tva_feature_dict(state)['features']
        for name, val in {**z, 'V_SD': value['value'], **feats}.items():
            if isinstance(val, (float, int)) and not math.isfinite(float(val)):
                nonfinite.append((li, name, val))
    for action in model.legal_flows(model.initial_state)[:4]:
        child = model.step_grid_flow(model.initial_state, action)
        flow = model.gft_resource_flow(model.initial_state, child)
        reward = model.gft_sd_reward(model.initial_state, child)
        for name, val in {**flow['bounded_flows'], 'r_sd': reward['r_sd'], 'r_total': reward['r_total']}.items():
            if val is not None and isinstance(val, (float, int)) and not math.isfinite(float(val)):
                nonfinite.append((li, name, val))
print({'levels': len(levels), 'states_checked': checked, 'nonfinite_count': len(nonfinite), 'sample_nonfinite': nonfinite[:5]})
PY
```

关键输出为 `levels=5`、`states_checked=14`、`nonfinite_count=0`。该审计覆盖 `gft_resource_vector()`、`gft_resource_flow()`、`gft_sd_reward()`、`gft_sd_value()` 和 `gft_tva_feature_dict()`。

默认路径 3 关等价 gate：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 3 --horizon 6 --budget 96 \
  --max-steps 30 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate3_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json
```

输出 [gft_sdmcts_gate3_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json)。与 [gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_creadout_v1_default_sdh_rp1_tp_gftprior_t2p0_h6_b96_seed0.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。

带 GFT 追踪字段 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 1 --horizon 3 --budget 24 \
  --max-steps 5 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --trace-gft-readouts \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_trace_fullspec_v1_level1_h3_b24_seed0.json
```

输出 [gft_sdmcts_trace_fullspec_v1_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_trace_fullspec_v1_level1_h3_b24_seed0.json)。trace 中新增 `gft_flow`、`gft_sd_reward`、`gft_sd_value_before`、`gft_sd_value_after` 和 `gft_tva_features_after`，并正常落盘。

默认 10 关正式预算：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_default_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `6/10`，status counts 为 `deadlocked=1`、`max_steps=2`、`no_action=1`、`solved=6`，`/usr/bin/time` 为 `real 312.61`、`user 315.46`、`sys 4.41`。与旧默认 10 关输出 [gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast_retime.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。

10 关正式预算，`capacity_weight=0.25`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-capacity-weight 0.25 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_kcap0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_fullspec_v1_kcap0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_kcap0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json)。结果为 solved `7/10`，status counts 为 `deadlocked=1`、`max_steps=1`、`no_action=1`、`solved=7`，`/usr/bin/time` 为 `real 264.37`、`user 267.13`、`sys 4.54`。与旧 `K=0.25` 输出 [gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_kcap0p25_v2_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) 比较，aggregate、逐关字段和 root trace mismatch 均为 `0`。

### 2026-06-14 补充：GFT-SD 全因素 10 关组合测试

除特别说明外，本节所有运行共享以下命令模板：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value sd_heuristic --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  [EXTRA_ARGS] \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/[OUTPUT_JSON]
```

关键新增运行如下：

| 目的 | EXTRA_ARGS | 输出 | solved | status counts |
|---|---|---|---:|---|
| 运输成本三项组合 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 8/10 | `deadlocked=1,max_steps=1,solved=8` |
| 运输成本三项组合加区域容量裕度 | `--gft-capacity-weight 0.25 --gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25` | [gft_sdmcts_gate10_fullspec_v1_kcap0p25_pturn0p10_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_kcap0p25_pturn0p10_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=1,no_action=1,solved=7` |
| 弱转弯成本加容量、可达性、区域风险 | `--gft-capacity-weight 0.25 --gft-transport-turn-weight 0.05 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25` | [gft_sdmcts_gate10_fullspec_v1_kcap0p25_pturn0p05_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_kcap0p25_pturn0p05_pacc0p25_pzone0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=2,solved=7` |
| 强底座加箱子机动性 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-mobility-weight 1.0` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_mob1p00_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_mob1p00_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=1,no_action=1,solved=7` |
| 强底座加玩家推位可达性 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-access-weight 1.0` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_acc1p00_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_acc1p00_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=2,solved=7` |
| 强底座加软死锁债务 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-deadlock-weight 0.25` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_ddead0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_ddead0p25_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=2,solved=7` |
| 强底座加顺序依赖债务 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-ordering-weight 0.05` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 8/10 | `deadlocked=1,max_steps=1,solved=8` |
| 强底座加目标承诺清晰度 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-commitment-weight 0.05` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=1,no_action=1,solved=7` |
| 强底座加瓶颈 overlap | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-bottleneck-weight 0.10` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_ibott0p10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_ibott0p10_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 6/10 | `deadlocked=2,max_steps=1,no_action=1,solved=6` |
| 强底座加顺序依赖与目标承诺 | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-ordering-weight 0.05 --gft-commitment-weight 0.05` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 8/10 | `deadlocked=1,max_steps=1,solved=8` |
| 强底座加顺序依赖、目标承诺、弱瓶颈 overlap | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-ordering-weight 0.05 --gft-commitment-weight 0.05 --gft-bottleneck-weight 0.02` | [gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_ibott0p02_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_ibott0p02_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=1,no_action=1,solved=7` |
| 转弯成本下保守全包，关闭瓶颈 overlap | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-capacity-weight 0.25 --gft-mobility-weight 1.0 --gft-access-weight 1.0 --gft-deadlock-weight 0.25 --gft-ordering-weight 0.05 --gft-commitment-weight 0.05 --gft-bottleneck-weight 0.0` | [gft_sdmcts_gate10_fullspec_v1_fullbundle_pturn0p10_conservative_i0_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_fullbundle_pturn0p10_conservative_i0_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=1,no_action=1,solved=7` |
| 转弯成本下保守全包，开启瓶颈 overlap | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-capacity-weight 0.25 --gft-mobility-weight 1.0 --gft-access-weight 1.0 --gft-deadlock-weight 0.25 --gft-ordering-weight 0.05 --gft-commitment-weight 0.05 --gft-bottleneck-weight 0.10` | [gft_sdmcts_gate10_fullspec_v1_fullbundle_pturn0p10_conservative_ibott0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_fullbundle_pturn0p10_conservative_ibott0p10_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 6/10 | `deadlocked=1,max_steps=3,solved=6` |
| 转弯成本下低权重全包，弱瓶颈 overlap | `--gft-transport-turn-weight 0.10 --gft-transport-access-weight 0.25 --gft-transport-zone-weight 0.25 --gft-capacity-weight 0.25 --gft-mobility-weight 0.25 --gft-access-weight 0.25 --gft-deadlock-weight 0.05 --gft-ordering-weight 0.01 --gft-commitment-weight 0.01 --gft-bottleneck-weight 0.02` | [gft_sdmcts_gate10_fullspec_v1_fullbundle_pturn0p10_lowweights_ibott0p02_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_fullspec_v1_fullbundle_pturn0p10_lowweights_ibott0p02_v1_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0.json) | 7/10 | `deadlocked=1,max_steps=2,solved=7` |

上述结果的共同判读为：当前 10 关最好候选为运输成本三项组合，或在其上加入 `ordering_weight=0.05` 与 `commitment_weight=0.05`；三者均为 `8/10`。所有结构因素均保留，瓶颈 overlap 在当前手工在线先验下不推荐默认开启。

### 2026-06-14 补充：GFT-TVA v1 接入、训练与闭环测试

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py
```

GFT-TVA 训练 smoke：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  --level-offset 8 --max-levels 6 --valid-fraction 0.33 \
  --state-depth 1 --max-states-per-level 8 \
  --collect-horizon 3 --collect-budget 32 \
  --bootstrap-leaf-value gft_sd \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_seed0.json
```

输出 [sokoban_gft_tva_v1_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_train_seed0.json) 与 [sokoban_gft_tva_v1_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_seed0.json)。gate 为 `PASS`；train `14` 样本、valid `11` 样本；valid MAE `2.124`，valid corr `0.967`；`/usr/bin/time` 为 `real 3.94`、`user 6.53`、`sys 5.57`。

GFT-TVA 1 关闭环 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 1 --horizon 3 --budget 24 \
  --max-steps 5 --seed 0 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_seed0.json \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --trace-gft-readouts \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/sokoban_gft_tva_v1_smoke_train_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_tva_v1_smoke_level1_h3_b24_seed0.json
```

输出 [gft_sdmcts_tva_v1_smoke_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_tva_v1_smoke_level1_h3_b24_seed0.json)。结果为 solved `0/1`，状态分布 `max_steps=1`，证明 GFT checkpoint 可被闭环读取并执行。

GFT-TVA v1 开发版训练：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  --level-offset 8 --max-levels 40 --valid-fraction 0.2 \
  --state-depth 1 --max-states-per-level 12 \
  --collect-horizon 4 --collect-budget 64 \
  --bootstrap-leaf-value gft_sd \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/sokoban_gft_tva_v1_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_tva_v1_seed0.json
```

输出 [sokoban_gft_tva_v1_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_train_seed0.json) 与 [sokoban_gft_tva_v1_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_v1_seed0.json)。训练集为关卡 `8-47`，不重叠 10 关开发评测 `48-57`；样本数为 `204`，其中 train `169`、valid `35`。gate 为 `PASS`；train MAE `0.381`、corr `0.641`；valid MAE `0.776`、corr `0.056`。`/usr/bin/time` 为 `real 24.28`、`user 27.08`、`sys 5.21`。

10 关显式 `V_SD` 叶值对照：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value gft_sd \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gftsdleaf_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `1/10`，状态分布 `deadlocked=1,max_steps=1,no_action=7,solved=1`，`/usr/bin/time` 为 `real 453.81`、`user 455.92`、`sys 4.75`。

10 关 `V_SD+TVA` 测试：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_v1_seed0.json \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/sokoban_gft_tva_v1_train_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gfttva_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_v1_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `0/10`，状态分布 `max_steps=1,no_action=9`，`/usr/bin/time` 为 `real 526.74`、`user 528.71`、`sys 4.91`。该结果是闭环负结果，不进入推荐配置。

### 2026-06-14 补充：GFT-SD 显式叶值折扣进展校准

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py
```

10 关 `discounted_progress` 零资源权重叶值：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gftsdleaf_discounted_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，mean pushes solved `17.5`，total deadlock prunes `29107`，`/usr/bin/time` 为 `real 488.25`、`user 489.77`、`sys 4.60`。与 `sd_heuristic` 强底座逐关 status 和 pushes 完全一致。

10 关 `discounted_progress` 加保守第二层叶值校正：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，mean pushes solved `17.75`，total deadlock prunes `30314`，`/usr/bin/time` 为 `real 550.80`、`user 552.77`、`sys 5.10`。失败关卡与零权重基底一致。

### 2026-06-14 补充：GFT-SD 第二层特征缓存验证

语法检查：

```bash
python -m py_compile \
  src/sokoban/gft_sd.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py
```

低预算缓存路径 smoke：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 3 --horizon 6 --budget 96 \
  --max-steps 12 --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate3_gftsdleaf_discounted_cache_smoke_k0p05_q0p05_c0p05_h6_b96_seed0.json
```

输出 [gft_sdmcts_gate3_gftsdleaf_discounted_cache_smoke_k0p05_q0p05_c0p05_h6_b96_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate3_gftsdleaf_discounted_cache_smoke_k0p05_q0p05_c0p05_h6_b96_seed0.json)。结果为 solved `0/3`，状态分布 `max_steps=3`，`/usr/bin/time` 为 `real 4.60`、`user 6.99`、`sys 4.68`。该 smoke 只验证非零第二层叶值权重经过缓存路径后可执行。

10 关保守第二层叶值组合缓存复测：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，mean pushes solved `17.75`，total deadlock prunes `30314`，`/usr/bin/time` 为 `real 481.41`、`user 483.22`、`sys 5.10`。与缓存前同配置 aggregate 完全一致，逐关 `level_index/status/solved/pushes/steps_executed/deadlock_prunes` mismatch 为 `0`；wall time 从 `550.80s` 降至 `481.41s`。

### 2026-06-14 补充：GFT-TVA v2 discounted-progress 基底

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  src/sokoban/gft_sd.py
```

GFT-TVA v2 训练：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  --level-offset 8 --max-levels 40 --valid-fraction 0.2 \
  --state-depth 1 --max-states-per-level 12 \
  --collect-horizon 4 --collect-budget 64 \
  --bootstrap-leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --value-provider-id sokoban_gft_search_value_tva_discounted_v2 \
  --out results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_seed0.json
```

输出 [sokoban_gft_tva_discounted_v2_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_train_seed0.json) 与 [sokoban_gft_tva_discounted_v2_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_seed0.json)。训练使用关卡 `8-47`，不重叠 10 关开发评测 `48-57`；样本数 `204`，train `169`、valid `35`。gate 为 `PASS`。train MAE `0.318`、corr `0.594`；valid MAE `0.871`、corr `0.668`。`/usr/bin/time` 为 `real 29.79`、`user 32.48`、`sys 5.47`。

GFT-TVA v2 10 关闭环：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_seed0.json \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/sokoban_gft_tva_discounted_v2_train_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_discounted_v2_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gfttva_discounted_v2_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_discounted_v2_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `1/10`，状态分布 `deadlocked=8,max_steps=1,solved=1`，mean pushes solved `12.0`，total deadlock prunes `10653`，`/usr/bin/time` 为 `real 380.03`、`user 381.88`、`sys 5.29`。该结果显著低于同基底显式 `V_SD` 的 `8/10`，不进入推荐配置。

### 2026-06-14 补充：GFT-TVA 直接终端价值口径修订

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  src/sokoban/gft_sd.py
```

入口参数检查：

```bash
python experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py --help
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py --help
```

direct-value TVA 训练 smoke：

```bash
python experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  --level-offset 8 --max-levels 3 --valid-fraction 0.34 \
  --state-depth 1 --max-states-per-level 4 \
  --collect-horizon 2 --collect-budget 12 \
  --bootstrap-leaf-value gft_sd \
  --target-mode direct_value \
  --min-train-samples 1 --min-valid-samples 1 \
  --value-provider-id sokoban_gft_tva_direct_smoke \
  --out results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_seed0.json
```

输出 [sokoban_gft_tva_direct_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_train_seed0.json) 与 [sokoban_gft_tva_direct_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_seed0.json)。gate 为 `PASS`，train `5` 样本、valid `2` 样本；checkpoint 记录 `target_mode=direct_value`、`direct_value_formula=V_psi=V_search`、`leaf_formula=V_leaf=V_psi`。

direct-value TVA 混合叶值 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --level-offset 48 --max-levels 1 \
  --horizon 3 --budget 24 --max-steps 5 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_seed0.json \
  --tva-value-blend 0.5 \
  --out results/sokoban_sd_world_model/gft_sdmcts_tva_direct_smoke_blend0p5_level1_h3_b24_seed0.json
```

输出 [gft_sdmcts_tva_direct_smoke_blend0p5_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_tva_direct_smoke_blend0p5_level1_h3_b24_seed0.json)。该 smoke solved `0/1`，状态为 `max_steps`，只用于验证 `V_leaf=(1-alpha)*V_SD+alpha*V_psi` 调用路径。

direct-value TVA 纯叶值 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --level-offset 48 --max-levels 1 \
  --horizon 3 --budget 24 --max-steps 5 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_smoke_seed0.json \
  --tva-value-blend 1.0 \
  --out results/sokoban_sd_world_model/gft_sdmcts_tva_direct_smoke_pure_level1_h3_b24_seed0.json
```

输出 [gft_sdmcts_tva_direct_smoke_pure_level1_h3_b24_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_tva_direct_smoke_pure_level1_h3_b24_seed0.json)。该 smoke solved `0/1`，状态为 `max_steps`，只用于验证 `V_leaf=V_psi` 调用路径。

### 2026-06-14 补充：GFT-TVA direct-value v3 10 关评估

GFT-TVA direct-value v3 训练：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  --level-offset 8 --max-levels 40 --valid-fraction 0.2 \
  --state-depth 1 --max-states-per-level 12 \
  --collect-horizon 4 --collect-budget 64 \
  --bootstrap-leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --target-mode direct_value \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --value-provider-id sokoban_gft_search_value_tva_direct_v3 \
  --out results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json
```

输出 [sokoban_gft_tva_direct_v3_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_train_seed0.json) 与 [sokoban_gft_tva_direct_v3_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json)。训练使用关卡 `8-47`，不重叠 10 关开发评测 `48-57`；样本数 `204`，train `169`、valid `35`。gate 为 `PASS`。train MAE `0.300`、corr `0.939`；valid MAE `0.752`、corr `0.710`。target stats train 为 mean `3.838`、std `1.243`、min `0.637`、max `6.203`。`/usr/bin/time` 为 `real 30.07`、`user 32.72`、`sys 5.53`。

GFT-TVA direct-value v3 纯 `V_psi` 10 关闭环：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json \
  --tva-value-blend 1.0 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_train_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_direct_v3_pure_a1p0_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gfttva_direct_v3_pure_a1p0_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_direct_v3_pure_a1p0_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `1/10`，状态分布 `max_steps=8,no_action=1,solved=1`，mean pushes solved `12.0`，total deadlock prunes `13636`，`/usr/bin/time` 为 `real 1504.46`、`user 1506.55`、`sys 4.63`。

GFT-TVA direct-value v3 `alpha=0.25` 混合 10 关闭环：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 --horizon 20 --budget 4096 \
  --max-steps 80 --seed 0 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json \
  --tva-value-blend 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_train_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_direct_v3_blend_a0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gfttva_direct_v3_blend_a0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_direct_v3_blend_a0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `1/10`，状态分布 `max_steps=7,no_action=2,solved=1`，mean pushes solved `12.0`，total deadlock prunes `11847`，`/usr/bin/time` 为 `real 1360.77`、`user 1362.70`、`sys 4.84`。

### 2026-06-14 补充：GFT-TVA 独立截断价值审计

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_tva.py \
  src/sokoban/gft_sd.py
```

审计 smoke：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  --level-offset 48 --max-levels 2 \
  --horizon 4 --budget 24 \
  --closed-loop-steps 1 \
  --max-leaf-states-per-level 4 \
  --teacher-max-expansions 400 \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json \
  --out results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_smoke_seed0.json
```

输出 [gft_tva_truncation_value_audit_direct_v3_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_smoke_seed0.json)。该 smoke 采集 `8` 个叶节点，脚本可执行并落盘；因高置信样本不足，gate 按预期为 `FAIL`，不作为性能结论。

10 关正式独立审计：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  --level-offset 48 --max-levels 10 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 12 \
  --teacher-max-expansions 4000 \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json \
  --out results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_seed0.json
```

输出 [gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_seed0.json)。采集 `120` 个叶节点，high-confidence `49` 个，teacher budget unsolved `71` 个。gate 为 `FAIL`，失败项为 `pairwise_order_not_better_than_gft_sd_leaf`。高置信组中，TVA MAE `0.805`、corr `0.768`、pairwise ordering accuracy `0.800`；显式 `V_SD` MAE `0.813`、corr `0.793`、pairwise ordering accuracy `0.827`。`/usr/bin/time` 为 `real 160.55`、`user 162.59`、`sys 5.04`。

10 关高预算标签复核：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  --level-offset 48 --max-levels 10 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 8 \
  --teacher-max-expansions 10000 \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_tva_direct_v3_seed0.json \
  --out results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json
```

输出 [gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json)。采集 `80` 个叶节点，high-confidence `41` 个，teacher budget unsolved `39` 个。gate 仍为 `FAIL`，失败项仍为 `pairwise_order_not_better_than_gft_sd_leaf`。高置信组中，TVA MAE `0.822`、corr `0.781`、pairwise ordering accuracy `0.821`；显式 `V_SD` MAE `0.881`、corr `0.823`、pairwise ordering accuracy `0.866`。budget-unsolved 组中 TVA pred mean `4.868`，显示该 checkpoint 对未在教师预算内解出的叶节点偏乐观。`/usr/bin/time` 为 `real 125.63`、`user 127.93`、`sys 4.81`。

### 2026-06-14 补充：GFT-TVA 删失风险探针

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_censored_risk.py \
  experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py
```

删失风险探针：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_censored_risk.py \
  --audit-path results/sokoban_sd_world_model/gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json \
  --out results/sokoban_sd_world_model/gft_tva_censored_risk_probe_direct_v3_teacher10k_seed0.json
```

输出 [gft_tva_censored_risk_probe_direct_v3_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_censored_risk_probe_direct_v3_teacher10k_seed0.json)。样本为 `80` 个叶节点，其中 solved low-risk `41` 个，teacher budget unsolved high-risk `39` 个。按关卡留一的 GFT 特征 ridge 风险探针 gate 为 `FAIL`：AUC `0.490`，orientation-free AUC `0.510`，balanced accuracy `0.515`，risk recall `0.615`，solved recall `0.415`。对照中，原始 `tva_prediction` 作为 risk score 的 AUC 为 `0.642`，原始 `gft_sd_leaf_value` 作为 risk score 的 AUC 为 `0.662`，但这是值越高越像 budget-unsolved 的反终端价值方向。

### 2026-06-14 补充：GFT-TVA v4 叶节点监督训练

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/train_sokoban_gft_leaf_tva.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py
```

叶节点监督 MLP 风险校正版训练：

```bash
python experiments/sokoban_sd_world_model/train_sokoban_gft_leaf_tva.py \
  --level-offset 8 --max-levels 20 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 12 \
  --teacher-max-expansions 4000 \
  --targeted-state-depth 3 \
  --targeted-max-states-per-level 48 \
  --targeted-max-blocked-per-level 6 \
  --epochs 300 --batch-size 64 \
  --hidden-dim 64 --hidden-layers 2 \
  --out results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json
```

输出 [sokoban_gft_leaf_tva_mlp_guarded_v4_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_train_seed0.json) 与 [sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json)。训练样本 `323`，训练价值标签 `185`，teacher counts 为 `teacher_budget_unsolved=89,solved=147,no_legal_actions=2,proven_dead_end=85`。训练 gate 为 `PASS`，验证价值 MAE `0.811`、corr `0.966`，风险头验证 AUC `0.776`。

对应独立审计：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  --level-offset 48 --max-levels 10 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 12 \
  --teacher-max-expansions 10000 \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_leaf_tva_mlp_guarded_v4_seed0.json \
  --out results/sokoban_sd_world_model/gft_tva_truncation_value_audit_leaf_v4_gate10_h8_b192_teacher10k_seed0.json
```

输出 [gft_tva_truncation_value_audit_leaf_v4_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_leaf_v4_gate10_h8_b192_teacher10k_seed0.json)。审计样本 `120`，high-confidence `52`，teacher budget unsolved `68`。gate 为 `FAIL`，失败项为 `mae_not_better_than_gft_sd_leaf` 和 `pairwise_order_not_better_than_gft_sd_leaf`；TVA MAE `0.909`，显式 `V_SD` MAE `0.849`；TVA pairwise `0.583`，显式 `V_SD` pairwise `0.809`。

校准锚定版本训练：

```bash
python experiments/sokoban_sd_world_model/train_sokoban_gft_leaf_tva.py \
  --level-offset 8 --max-levels 20 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 12 \
  --teacher-max-expansions 4000 \
  --targeted-state-depth 3 \
  --targeted-max-states-per-level 48 \
  --targeted-max-blocked-per-level 6 \
  --epochs 50 --batch-size 64 \
  --hidden-dim 64 --hidden-layers 2 \
  --anchor-weight 1.0 \
  --anchor-bias-mode train_median \
  --anchor-bias-labels solved \
  --value-provider-id sokoban_gft_leaf_tva_calibrated_anchor_v4 \
  --out results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json
```

输出 [sokoban_gft_leaf_tva_calibrated_anchor_v4_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_train_seed0.json) 与 [sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json)。训练 gate 为 `PASS`，`anchor_bias=-0.22770166397094727`。

校准锚定版本独立审计：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  --level-offset 48 --max-levels 10 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 12 \
  --teacher-max-expansions 10000 \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json \
  --out results/sokoban_sd_world_model/gft_tva_truncation_value_audit_leaf_calibrated_anchor_v4_gate10_h8_b192_teacher10k_seed0.json
```

输出 [gft_tva_truncation_value_audit_leaf_calibrated_anchor_v4_gate10_h8_b192_teacher10k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_tva_truncation_value_audit_leaf_calibrated_anchor_v4_gate10_h8_b192_teacher10k_seed0.json)。gate 为 `PASS`；TVA high-confidence MAE `0.761`，显式 `V_SD` MAE `0.849`；二者 pairwise ordering accuracy 均为 `0.809`。

10 关闭环纯校准锚定 TVA：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json \
  --constraint-mode tier_a --select-by visits \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 `7/10`，状态分布 `deadlocked=1,max_steps=1,no_action=1,solved=7`。

10 关 `alpha=0.25` 混合：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_leaf_tva_calibrated_anchor_v4_seed0.json \
  --tva-value-blend 0.25 \
  --constraint-mode tier_a --select-by visits \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta --gft-prior-temperature 2.0 --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_gfttva_calibrated_anchor_v4_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 `8/10`，状态分布 `deadlocked=1,max_steps=1,solved=8`，总耗时 `509.54s`。同口径显式 `V_SD` 对照为 `8/10`，总耗时 `478.90s`，因此该混合版本不构成新最佳。

### 2026-06-15 补充：显式 GFT-SD 叶值 50 关阶段性确认

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 50 \
  --horizon 20 --budget 4096 --max-steps 80 \
  --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_holdout50_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_gftsdleaf_discounted_cache_v1_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)。结果为 solved `44/50`，success rate `0.88`，mean pushes solved `15.681818181818182`，total deadlock prunes `165554`，status counts 为 `deadlocked=1`、`max_steps=2`、`no_action=3`、`solved=44`。`/usr/bin/time` 为 `real 867.49`、`user 869.90`、`sys 4.65`。

与旧 50 关最强 [gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_sdh_rp1_tp_gftprior_t2p0_h20_b4096_seed0_delegatedfast.json) 比较，新增解出 `level_index=5,6,25,30`，退化 `level_index=40`，净增 `+3` 关。失败关卡为 `0:max_steps`、`9:deadlocked`、`14:no_action`、`40:max_steps`、`45:no_action`、`46:no_action`。

### 2026-06-15 补充：`44/50` 未解关卡定向审计

本次审计复用显式 GFT-SD 叶值 50 关正式配置，只将 `--max-levels` 改为 `1`、`--trace-limit` 改为 `100`、开启 `--trace-gft-readouts`，并分别设置 `--level-offset` 为失败关卡在数据文件中的全局偏移。50 关 heldout 从 `--level-offset 48` 开始，因此本次 6 个本地失败关卡 `0,9,14,40,45,46` 分别对应全局偏移 `48,57,62,88,93,94`。

命令模板如下，其中 `<offset>` 替换为上述全局偏移，`<name>` 替换为输出文件名：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset <offset> --max-levels 1 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --trace-limit 100 \
  --trace-gft-readouts \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/<name>.json
```

输出文件和复现状态如下：`audit_fail_level00_gftsdleaf_trace100_seed0.json` 为 `max_steps`，`audit_fail_level09_gftsdleaf_trace100_seed0.json` 为 `deadlocked`，`audit_fail_level14_gftsdleaf_trace100_seed0.json` 为 `no_action`，`audit_fail_level40_gftsdleaf_trace100_seed0.json` 为 `max_steps`，`audit_fail_level45_gftsdleaf_trace100_seed0.json` 为 `no_action`，`audit_fail_level46_gftsdleaf_trace100_seed0.json` 为 `no_action`。6 个定向结果均复现正式 50 关结果中的对应失败状态。

审计后还运行了文档和语法检查：

```bash
git diff --check docs/research_log.md docs/methods.md docs/reproducibility.md docs/decisions.md
python -m py_compile experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py src/sokoban/gft_sd.py src/sokoban/__init__.py
```

### 2026-06-15 补充：`M/A/D/I` 保守激活 10 关筛查

所有筛查均复用当前显式 GFT-SD 叶值正式配置：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --leaf-value gft_sd \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --require-independent-gates \
  <variant-args> \
  --out results/sokoban_sd_world_model/<output>.json
```

变体参数与输出如下。

```text
M: --gft-mobility-weight 0.10 --gft-sd-leaf-mobility-weight 0.05
output: gft_sdmcts_gate10_gftsdleaf_m_v1_mprior0p10_mleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
result: 7/10, status deadlocked=1,max_steps=1,no_action=1,solved=7

A: --gft-access-weight 0.10 --gft-sd-leaf-access-weight 0.05
output: gft_sdmcts_gate10_gftsdleaf_a_v1_aprior0p10_aleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
result: 8/10, status deadlocked=1,max_steps=1,solved=8

D: --gft-deadlock-weight 0.05 --gft-sd-leaf-deadlock-weight 0.05
output: gft_sdmcts_gate10_gftsdleaf_d_v1_dprior0p05_dleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
result: 6/10, status deadlocked=1,max_steps=2,no_action=1,solved=6

I: --gft-bottleneck-weight 0.02
output: gft_sdmcts_gate10_gftsdleaf_i_v1_ibott0p02_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
result: 6/10, status deadlocked=1,max_steps=3,solved=6

MA: --gft-mobility-weight 0.05 --gft-access-weight 0.10 --gft-sd-leaf-mobility-weight 0.025 --gft-sd-leaf-access-weight 0.05
output: gft_sdmcts_gate10_gftsdleaf_ma_v1_mprior0p05_aprior0p10_mleaf0p025_aleaf0p05_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
result: 7/10, status deadlocked=1,max_steps=1,no_action=1,solved=7
```

原计划的 `M/A/D/I` 全组合项在启动后被中止，因为其中包含已经在单项中严重退化的 `D` 和 `I` 权重；该中止不产生可用结果文件，不作为性能结论。

玩家推位可达性 `A` 单项随后在 6 个正式失败关卡上定向复核。命令同上，但 `--max-levels 1`，并将 `--level-offset` 分别设为 `48,57,62,88,93,94`。输出与状态如下。

```text
audit_fail_level00_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json: max_steps
audit_fail_level09_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json: deadlocked
audit_fail_level14_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json: max_steps
audit_fail_level40_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json: max_steps
audit_fail_level45_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json: no_action
audit_fail_level46_gftsdleaf_a_v1_aprior0p10_aleaf0p05_seed0.json: no_action
```

### 2026-06-15 补充：在线轨迹 GFT-TVA 机制 smoke

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/train_sokoban_gft_leaf_tva.py
```

低预算风险样本 smoke：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 5 \
  --horizon 6 --budget 96 --max-steps 30 --seed 0 \
  --epochs 20 --batch-size 32 --hidden-dim 32 --hidden-layers 1 \
  --min-value-train-samples 1 --min-risk-classes 1 \
  --out results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_smoke_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_smoke_seed0.json
```

输出 [sokoban_gft_online_tva_guarded_v1_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_smoke_train_seed0.json) 与 [sokoban_gft_online_tva_guarded_v1_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_smoke_seed0.json)。5 条轨迹均为 `max_steps`，采样 `94` 个删失风险样本，`value_train_samples=0`，gate 因 `min_value_train_samples` 为 `FAIL`；该结果验证 `max_steps` 默认不作为确定价值负例。

成功轨迹价值标签 smoke：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 52 --max-levels 1 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --epochs 20 --batch-size 16 --hidden-dim 32 --hidden-layers 1 \
  --min-value-train-samples 1 --min-risk-classes 1 \
  --out results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_seed0.json
```

输出 [sokoban_gft_online_tva_guarded_v1_solved_smoke_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_train_seed0.json) 与 [sokoban_gft_online_tva_guarded_v1_solved_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_seed0.json)。该轨迹为 solved，采样 `13` 个高置信价值样本，gate 为 `PASS`，train value corr `0.9659`。

评估入口读取 checkpoint smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 52 --max-levels 1 \
  --horizon 3 --budget 24 --max-steps 5 --seed 0 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_solved_smoke_seed0.json \
  --tva-value-blend 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --revisit-penalty 1.0 --transposition \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_online_tva_guarded_v1_solved_smoke_eval_seed0.json
```

输出 [gft_sdmcts_online_tva_guarded_v1_solved_smoke_eval_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_online_tva_guarded_v1_solved_smoke_eval_seed0.json)，执行完成并落盘。若带 `--require-independent-gates`，该 checkpoint 会因缺少独立 TVA 审计路径被正式门槛拦截；这符合当前安全接入约束。

### 2026-06-15 补充：在线轨迹 GFT-TVA 混合叶值候选

语法检查：

```bash
python -m py_compile experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py
```

非 heldout 混合轨迹探测：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/001.txt \
  --level-offset 0 --max-levels 40 \
  --horizon 8 --budget 192 --max-steps 40 --seed 0 \
  --epochs 1 --batch-size 64 --hidden-dim 64 --hidden-layers 2 \
  --min-value-train-samples 20 --min-risk-classes 2 \
  --out results/sokoban_sd_world_model/sokoban_gft_online_tva_mix_probe_file001_o0_n40_h8_b192_s40_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_online_tva_mix_probe_file001_o0_n40_h8_b192_s40_seed0.json
```

输出显示样本 `834` 个，轨迹状态 `max_steps=648,solved=134,no_action=52`，关卡状态 `max_steps=29,solved=8,no_action=3`，gate 为 `PASS`。

修订标签策略后的候选训练：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/train_sokoban_gft_online_tva.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/001.txt \
  --level-offset 0 --max-levels 40 \
  --horizon 8 --budget 192 --max-steps 40 --seed 0 \
  --epochs 300 --batch-size 64 --hidden-dim 64 --hidden-layers 2 \
  --failure-value-label-scope final \
  --anchor-weight 0.85 \
  --anchor-bias-mode none \
  --risk-penalty 1.0 \
  --risk-loss-weight 0.25 \
  --min-value-train-samples 20 --min-risk-classes 2 \
  --out results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json \
  --checkpoint-out results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json
```

输出 [sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json) 和 [sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json)。训练 gate 为 `PASS`，value train samples 为 `114`。

独立截断价值审计：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_truncation_value.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/002.txt \
  --level-offset 0 --max-levels 10 \
  --horizon 8 --budget 192 \
  --closed-loop-steps 2 \
  --max-leaf-states-per-level 12 \
  --teacher-max-expansions 4000 \
  --seed 0 \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json
```

输出 [gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json)，gate 为 `PASS`，高置信 MAE `0.489`，显式 `V_SD` 高置信 MAE `0.552`。

删失风险审计：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/audit_sokoban_gft_tva_censored_risk.py \
  --audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --out results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_censored_risk_probe_file002_o0_n10_h8_b192_teacher4k_seed0.json
```

输出 [gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_censored_risk_probe_file002_o0_n10_h8_b192_teacher4k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_censored_risk_probe_file002_o0_n10_h8_b192_teacher4k_seed0.json)，gate 为 `FAIL`，AUC `0.442`，balanced accuracy `0.504`，risk recall `0.238`。

低预算闭环对照命令模板如下，`<leaf-args>` 替换为显式 `V_SD` 或在线 TVA 参数，`<file>` 替换为 `002.txt` 或 `003.txt`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/<file> \
  --level-offset 0 --max-levels 10 \
  --horizon 8 --budget 192 --max-steps 40 --seed 0 \
  --transposition \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  <leaf-args> \
  --out results/sokoban_sd_world_model/<output>.json
```

显式 `V_SD` 的 `<leaf-args>` 为 `--leaf-value gft_sd`。在线 TVA 的 `<leaf-args>` 为：

```bash
--leaf-value tva \
--tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
--tva-value-blend 0.25 \
--tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json
```

`train/002.txt` 前 10 关结果：显式 `V_SD` 输出 [gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json)，为 `5/10`；在线 TVA 输出 [gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file002_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json)，为 `6/10`。`train/003.txt` 前 10 关结果：显式 `V_SD` 输出 [gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_gftsd_baseline_seed0.json)，为 `4/10`；在线 TVA 输出 [gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_file003_o0_n10_h8_b192_s40_trans_online_tva_failfinal_anchor0p85_blend0p25_seed0.json)，为 `6/10`。

### 2026-06-15 补充：在线轨迹 GFT-TVA 正式预算闭环效果测试

10 关 `alpha=0.10`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.10 \
  --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p10_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p10_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p10_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `7/10`，`real=301.08s`。

10 关 `alpha=0.25`：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0.json)，结果为 `8/10`，`real=260.18s`。

50 关 `alpha=0.25` 探索性测试：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 50 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0_exploratory.json
```

输出 [gft_sdmcts_holdout50_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_failfinal_anchor0p85_blend0p25_k0p05_q0p05_c0p05_pturn0p10_pacc0p25_pzone0p25_qord0p05_ccommit0p05_h20_b4096_seed0_exploratory.json)。结果为 `45/50`，状态分布 `deadlocked=1,max_steps=2,no_action=2,solved=45`，`real=993.52s,user=995.29s,sys=5.23s`。

独立门槛读取修复后运行 smoke：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 49 --max-levels 1 \
  --horizon 3 --budget 24 --max-steps 5 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --require-independent-gates \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_online_tva_gate_readout_smoke_seed0.json
```

输出 [gft_sdmcts_online_tva_gate_readout_smoke_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_online_tva_gate_readout_smoke_seed0.json)，执行完成，说明 GFT-TVA 截断价值审计字段已能被 `--require-independent-gates` 识别。

### 在线轨迹 GFT-TVA 动态消费消融

语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

宽阈值动态叶值 10 关：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-dynamic-blend \
  --tva-dynamic-high-risk-threshold 0.90 \
  --tva-dynamic-negative-delta-threshold 1.0 \
  --tva-dynamic-high-risk-blend 0.75 \
  --tva-dynamic-revisit-penalty 0.0 \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_riskblend_seed0.json
```

输出 [gft_sdmcts_gate10_online_tva_dynamic_riskblend_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_riskblend_seed0.json)，结果为 `5/10`，`real=475.37s`。加入 `--tva-dynamic-revisit-penalty 0.5` 的版本输出 [gft_sdmcts_gate10_online_tva_dynamic_riskblend_revisit0p5_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_riskblend_revisit0p5_seed0.json)，结果为 `6/10`，`real=384.95s`。

第 25 关收紧阈值、动作先验软惩罚和动态根选择诊断：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 73 --max-levels 1 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-dynamic-blend \
  --tva-dynamic-high-risk-threshold 0.995 \
  --tva-dynamic-negative-delta-threshold 2.0 \
  --tva-dynamic-high-risk-blend 1.0 \
  --tva-dynamic-prior-penalty-weight 1.5 \
  --tva-dynamic-prior-penalty-cap 4.0 \
  --tva-dynamic-root-select \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json
```

输出 [gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json)，结果为 `1/1`，17 步解出，`real=12.90s`。

同一动态根选择配置扩展到 10 关：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-dynamic-blend \
  --tva-dynamic-high-risk-threshold 0.995 \
  --tva-dynamic-negative-delta-threshold 2.0 \
  --tva-dynamic-high-risk-blend 1.0 \
  --tva-dynamic-prior-penalty-weight 1.5 \
  --tva-dynamic-prior-penalty-cap 4.0 \
  --tva-dynamic-root-select \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json
```

输出 [gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_rootselect_seed0.json)，结果为 `6/10`，`real=374.35s`。全局根动作按均值选择的 10 关对照输出 [gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_selectq_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_dynamic_strict_alpha1_priorpen1p5_selectq_seed0.json)，结果为 `7/10`，`real=402.49s`。

### 在线轨迹 GFT-TVA 状态门控混合

状态门控混合新增参数后再次执行语法检查：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py
```

10 关宽松状态门控：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-state-gated-blend \
  --tva-state-gated-default-blend 0.0 \
  --tva-state-gated-positive-blend 0.25 \
  --tva-state-gated-positive-risk-max 0.30 \
  --tva-state-gated-positive-delta-min 0.25 \
  --tva-state-gated-negative-blend 0.0 \
  --tva-state-gated-negative-risk-min 0.95 \
  --tva-state-gated-negative-delta-min 1.0 \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_posonly_seed0.json
```

输出 [gft_sdmcts_gate10_online_tva_state_gated_posonly_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_posonly_seed0.json)，结果为 `9/10`，`real=241.09s`。严格正向可信条件 `risk<=0.05,V_TVA-V_SD>=0.5` 的第 25 关诊断输出 [gft_sdmcts_l25_online_tva_state_gated_strictpos_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l25_online_tva_state_gated_strictpos_seed0.json)，17 步解出；但严格条件 10 关输出 [gft_sdmcts_gate10_online_tva_state_gated_strictpos_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_strictpos_seed0.json)，结果为 `8/10`。

50 关状态门控候选使用宽松状态门控并启用动态根动作选择兜底：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 50 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-state-gated-blend \
  --tva-state-gated-default-blend 0.0 \
  --tva-state-gated-positive-blend 0.25 \
  --tva-state-gated-positive-risk-max 0.30 \
  --tva-state-gated-positive-delta-min 0.25 \
  --tva-state-gated-negative-blend 0.0 \
  --tva-state-gated-negative-risk-min 0.95 \
  --tva-state-gated-negative-delta-min 1.0 \
  --tva-dynamic-root-select \
  --tva-dynamic-high-risk-threshold 0.95 \
  --tva-dynamic-negative-delta-threshold 1.0 \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json
```

输出 [gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json)。结果为 `46/50`，状态分布 `max_steps=1,no_action=3,solved=46`，`real=1120.14s,user=1120.96s,sys=5.69s`。

### 在线轨迹 GFT-TVA 固定高权重窗口与组合策略

本节记录 `46/50` 状态门控候选之后的补充实验。语法检查命令为：

```bash
python -m py_compile \
  experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  experiments/sokoban_sd_world_model/evaluate_sokoban_grid_sdmcts.py \
  experiments/sokoban_sd_world_model/audit_sokoban_grid_tva.py \
  src/sokoban/gft_sd.py
```

固定高权重单关复核使用同一在线 GFT-TVA checkpoint，并关闭状态门控。以第 14 关为例：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 62 --max-levels 1 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.60 \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_l14_fixed_alpha060_seed0.json
```

同一命令将 `--level-offset` 改为 `93` 和 `94`，并改写输出文件名，可复现 [gft_sdmcts_l45_fixed_alpha060_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l45_fixed_alpha060_seed0.json) 与 [gft_sdmcts_l46_fixed_alpha060_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l46_fixed_alpha060_seed0.json)。三者分别为 `29/13/10` 步 solved。将 `--level-offset` 改为 `48` 得到 [gft_sdmcts_l00_fixed_alpha060_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_l00_fixed_alpha060_seed0.json)，结果仍为 `max_steps`。

固定高权重 10 关检查如下：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 --max-levels 10 \
  --horizon 20 --budget 4096 --max-steps 80 --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.60 \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_fixed_alpha060_h20_b4096_seed0.json
```

输出 [gft_sdmcts_gate10_online_tva_fixed_alpha060_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_fixed_alpha060_h20_b4096_seed0.json)，结果为 `7/10`，失败为 `0:max_steps`、`5:deadlocked`、`6:no_action`。将 `--tva-value-blend` 改为 `0.65` 得到 [gft_sdmcts_gate10_online_tva_fixed_alpha065_h20_b4096_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_fixed_alpha065_h20_b4096_seed0.json)，结果同为 `7/10`。

失败后重跑组合结果文件为 [gft_sdmcts_holdout50_state_gated_then_fixed_alpha060_fallback_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_state_gated_then_fixed_alpha060_fallback_seed0_exploratory.json)。其 `primary_result` 是 [gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json)，失败后重跑来源为 `level_index=0,14,45,46` 的固定 `alpha=0.60` 单关文件。组合规则为：若 primary 失败且 fallback solved，则用 fallback 结果替换；否则保留 primary 结果。aggregate 为 `49/50`，`fallback_used_count=4`，`fallback_solved_count=3`，唯一失败为 `0:max_steps`。

结果核验命令：

```bash
python - <<'PY'
import json
from pathlib import Path
for name in [
    'gft_sdmcts_holdout50_online_tva_state_gated_posonly_rootselect_seed0_exploratory.json',
    'gft_sdmcts_holdout50_state_gated_then_fixed_alpha060_fallback_seed0_exploratory.json',
]:
    path = Path('results/sokoban_sd_world_model') / name
    data = json.loads(path.read_text())
    print(name)
    print(data['aggregate'])
    failures = [
        (row.get('level_index'), row.get('status'))
        for row in data['levels']
        if row.get('status') != 'solved'
    ]
    print(failures)
PY
```

### 动态 TVA alpha 规则离线审计

本审计只重放已保存的 Sokoban 结果 trace，不运行 MCTS，不改变动作序列。代码文件为 [audit_sokoban_dynamic_tva_alpha_features.py](/home/data/round/code/sd_world_model_rl/experiments/sokoban_sd_world_model/audit_sokoban_dynamic_tva_alpha_features.py)。语法检查命令为：

```bash
python -m py_compile experiments/sokoban_sd_world_model/audit_sokoban_dynamic_tva_alpha_features.py
```

审计运行命令为：

```bash
python experiments/sokoban_sd_world_model/audit_sokoban_dynamic_tva_alpha_features.py
```

默认输入包括当前状态门控 `46/50` 文件、固定 `alpha=0.60` 的 10 关和单关文件、以及 phase060 单关文件。默认输出为 [dynamic_tva_alpha_feature_audit_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/dynamic_tva_alpha_feature_audit_seed0.json)。本次输出的规则触发统计为：

```json
{
  "negative_high_alpha_harm:prefix_phase060": 2,
  "positive_fixed_alpha_repair:initial_phase060": 3,
  "positive_fixed_alpha_repair:prefix_phase060": 3
}
```

解释口径为：phase060 规则覆盖全部三个固定高权重修复正例 `14/45/46`，但也在两个高权重负例 `5/6` 的轨迹前缀中触发。该结果支持将 `49/50` 记录为组合策略信号，而不是单次动态 alpha 规则。

### 动态运输阶段 TVA 单次 50 关候选

本候选在状态门控 GFT-TVA 基础上启用前缀受限和根状态受限的运输阶段高权重 TVA。语法检查命令为：

```bash
python -m py_compile experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py
```

10 关开发切片命令为：

```bash
python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 \
  --max-levels 10 \
  --horizon 20 \
  --budget 4096 \
  --max-steps 80 \
  --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-dynamic-high-risk-threshold 0.95 \
  --tva-dynamic-negative-delta-threshold 1.0 \
  --tva-dynamic-root-select \
  --tva-state-gated-blend \
  --tva-state-gated-default-blend 0 \
  --tva-state-gated-positive-blend 0.25 \
  --tva-state-gated-positive-risk-max 0.30 \
  --tva-state-gated-positive-delta-min 0.25 \
  --tva-state-gated-negative-blend 0 \
  --tva-state-gated-negative-risk-min 0.95 \
  --tva-state-gated-negative-delta-min 1.0 \
  --tva-transport-phase-blend 0.60 \
  --tva-transport-phase-high-progress-min 20 \
  --tva-transport-phase-low-progress-max 10 \
  --tva-transport-phase-commitment-min 0.25 \
  --tva-transport-phase-max-closed-loop-step 5 \
  --tva-transport-phase-require-root-eligible \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_rooteligible_seed0.json
```

10 关输出 [gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_rooteligible_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_rooteligible_seed0.json) 为 `9/10`，唯一失败为 `level_index=0:max_steps`。

50 关 heldout 命令为：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 \
  --max-levels 50 \
  --horizon 20 \
  --budget 4096 \
  --max-steps 80 \
  --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-dynamic-high-risk-threshold 0.95 \
  --tva-dynamic-negative-delta-threshold 1.0 \
  --tva-dynamic-root-select \
  --tva-state-gated-blend \
  --tva-state-gated-default-blend 0 \
  --tva-state-gated-positive-blend 0.25 \
  --tva-state-gated-positive-risk-max 0.30 \
  --tva-state-gated-positive-delta-min 0.25 \
  --tva-state-gated-negative-blend 0 \
  --tva-state-gated-negative-risk-min 0.95 \
  --tva-state-gated-negative-delta-min 1.0 \
  --tva-transport-phase-blend 0.60 \
  --tva-transport-phase-high-progress-min 20 \
  --tva-transport-phase-low-progress-max 10 \
  --tva-transport-phase-commitment-min 0.25 \
  --tva-transport-phase-max-closed-loop-step 5 \
  --tva-transport-phase-require-root-eligible \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json
```

50 关输出 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_seed0_exploratory.json)，aggregate 为 `49/50`，状态分布 `max_steps=1,solved=49`，唯一失败为 `level_index=0:max_steps`。`/usr/bin/time` 为 `real=1738.16s,user=1738.00s,sys=6.74s`，JSON 内逐关 elapsed 汇总为 `1733.7386854887009s`。该结果为单次配置输出，不是失败后重跑组合结果。

### 动态运输阶段 GFT-TVA 正式主线复核

正式复跑在上述冻结规则基础上加入 independent gates：

```bash
/usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt \
  --level-offset 48 \
  --max-levels 50 \
  --horizon 20 \
  --budget 4096 \
  --max-steps 80 \
  --seed 0 \
  --transposition \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --tva-value-blend 0.25 \
  --tva-dynamic-high-risk-threshold 0.95 \
  --tva-dynamic-negative-delta-threshold 1.0 \
  --tva-dynamic-root-select \
  --tva-state-gated-blend \
  --tva-state-gated-default-blend 0 \
  --tva-state-gated-positive-blend 0.25 \
  --tva-state-gated-positive-risk-max 0.30 \
  --tva-state-gated-positive-delta-min 0.25 \
  --tva-state-gated-negative-blend 0 \
  --tva-state-gated-negative-risk-min 0.95 \
  --tva-state-gated-negative-delta-min 1.0 \
  --tva-transport-phase-blend 0.60 \
  --tva-transport-phase-high-progress-min 20 \
  --tva-transport-phase-low-progress-max 10 \
  --tva-transport-phase-commitment-min 0.25 \
  --tva-transport-phase-max-closed-loop-step 5 \
  --tva-transport-phase-require-root-eligible \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-transport-turn-weight 0.10 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --require-independent-gates \
  --out results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json
```

输出 [gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_holdout50_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0.json)，aggregate 为 `49/50`，状态分布 `max_steps=1,solved=49`，唯一失败为 `level_index=0:max_steps`。`independent_gate_status` 为 `grid_sd_gate=PASS,tva_independent_gate=PASS,failed_checks=[]`，material passport 为 `VERIFIED`。`/usr/bin/time` 为 `real=996.47s,user=995.41s,sys=0.73s`，JSON 内逐关 elapsed 汇总为 `993.6554618529044s`。

关键消融结果：

```text
gft_sdmcts_gate10_online_tva_state_gated_phase060_noguard_verified_seed0.json:
  7/10, failures = 0:max_steps, 5:deadlocked, 6:deadlocked

gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_only_verified_seed0.json:
  7/10, failures = 0:max_steps, 5:deadlocked, 6:no_action

gft_sdmcts_gate10_online_tva_state_gated_phase060_rooteligible_only_verified_seed0.json:
  9/10, failures = 0:max_steps

gft_sdmcts_gate10_online_tva_state_gated_phase060_prefix5_rooteligible_seed0.json:
  9/10, failures = 0:max_steps
```

外层切片复核使用 `datasets/sokoban_boxoban/boxoban-levels/medium/train/001.txt` 的 `level_offset=0,max_levels=10`：

```text
gft_sdmcts_file001_o0_n10_gftsdleaf_discounted_verified_seed0.json:
  6/10, failures = 0:max_steps, 1:max_steps, 6:max_steps, 8:max_steps

gft_sdmcts_file001_o0_n10_online_tva_state_gated_posonly_rootselect_verified_seed0.json:
  6/10, failures = 0:max_steps, 1:max_steps, 6:max_steps, 8:max_steps

gft_sdmcts_file001_o0_n10_online_tva_state_gated_phase060_prefix5_rooteligible_verified_seed0.json:
  7/10, failures = 1:max_steps, 6:max_steps, 8:max_steps
```

解释口径：正式主线为单次动态运输阶段 GFT-TVA `49/50`；显式 GFT-SD `44/50` 为零 TVA 对照；状态门控 GFT-TVA `46/50` 为较保守单次对照；失败后固定 `alpha=0.60` 重跑的 `49/50` 仍为组合策略对照。

### Boxoban `unfiltered/test` 1000 关公开测试集复核

数据切分为 `datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt`，共 `1000` 关。为缩短 wall-clock time，正式结果由 20 个不重叠 50 关分片合并得到。分片等价规则为：评估脚本内部每关随机流为 `args.seed + level_idx*1009`；对 `level_offset=offset` 的分片设置 `seed=offset*1009`，即可等价于单进程全量 `seed=0` 的对应全局关卡随机流。

分片运行命令模板如下：

```bash
for offset in $(seq 0 50 950); do
  shard=$(printf '%03d_%03d' "$offset" "$((offset + 49))")
  seed=$((offset * 1009))
  out="results/sokoban_sd_world_model/test1000_shards/gft_sdmcts_boxoban_unfiltered_test1000_shard${shard}_phase060_prefix5_rooteligible_verified_gates_seedmatch.json"
  log="results/sokoban_sd_world_model/test1000_shards/gft_sdmcts_boxoban_unfiltered_test1000_shard${shard}_phase060_prefix5_rooteligible_verified_gates_seedmatch.log"
  (
    /usr/bin/time -p python experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
      --levels datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt \
      --level-offset "$offset" \
      --max-levels 50 \
      --horizon 20 \
      --budget 4096 \
      --max-steps 80 \
      --seed "$seed" \
      --transposition \
      --leaf-value tva \
      --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
      --tva-value-blend 0.25 \
      --tva-dynamic-high-risk-threshold 0.95 \
      --tva-dynamic-negative-delta-threshold 1.0 \
      --tva-dynamic-root-select \
      --tva-state-gated-blend \
      --tva-state-gated-default-blend 0 \
      --tva-state-gated-positive-blend 0.25 \
      --tva-state-gated-positive-risk-max 0.30 \
      --tva-state-gated-positive-delta-min 0.25 \
      --tva-state-gated-negative-blend 0 \
      --tva-state-gated-negative-risk-min 0.95 \
      --tva-state-gated-negative-delta-min 1.0 \
      --tva-transport-phase-blend 0.60 \
      --tva-transport-phase-high-progress-min 20 \
      --tva-transport-phase-low-progress-max 10 \
      --tva-transport-phase-commitment-min 0.25 \
      --tva-transport-phase-max-closed-loop-step 5 \
      --tva-transport-phase-require-root-eligible \
      --gft-prior gft_delta \
      --gft-prior-temperature 2.0 \
      --gft-prior-uniform-mix 0.1 \
      --gft-ordering-weight 0.05 \
      --gft-commitment-weight 0.05 \
      --gft-transport-turn-weight 0.10 \
      --gft-transport-access-weight 0.25 \
      --gft-transport-zone-weight 0.25 \
      --gft-sd-leaf-mode discounted_progress \
      --gft-sd-leaf-capacity-weight 0.05 \
      --gft-sd-leaf-ordering-weight 0.05 \
      --gft-sd-leaf-commitment-weight 0.05 \
      --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
      --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
      --require-independent-gates \
      --out "$out"
  ) >"$log" 2>&1 &
done
wait
```

合并检查使用如下逻辑：读取 20 个分片 JSON，确认每片 `max_levels=50`、`seed=level_offset*1009`、`grid_sd_gate=PASS`、`tva_independent_gate=PASS`、`failed_checks=[]`，然后将局部 `level_index` 加上 `level_offset` 恢复为全局编号，并检查覆盖 `0-999` 且无重复。合并输出为 [gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json)。

汇总核验结果：

```text
levels_evaluated = 1000
solved_count = 989
success_rate = 0.989
mean_pushes_solved = 12.398382204246714
total_deadlock_prunes = 2499261
status_counts = {deadlocked: 4, max_steps: 7, solved: 989}
independent_gate_status = {grid_sd_gate: PASS, tva_independent_gate: PASS, failed_checks: []}
material_passport.verification_status = VERIFIED
fallback fields present = false
```

失败关卡：

```text
199:deadlocked
367:max_steps
525:deadlocked
546:max_steps
664:max_steps
698:max_steps
706:deadlocked
758:max_steps
785:deadlocked
892:max_steps
977:max_steps
```

20 个分片的最大 `/usr/bin/time` wall time 为 `real=1509.17s`，分片 wall time 总和为 `20813.39s`，user time 总和为 `20789.28s`。运行期间曾启动一个单进程 `max_levels=1000` 备份进程；在分片汇总验证通过后已终止该备份进程，其未用于报告结果。

### Boxoban 文献规模对照表复现来源

本轮未运行新的训练或评测，只复核已有结果文件和本地文献摘录，并将对照表写入论文稿。表格中的本文结果来源如下：

1. Boxoban `unfiltered/test` 结果：输出 [gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json)，记录 `levels_evaluated=1000`、`solved_count=989`、`success_rate=0.989`、`status_counts={deadlocked:4,max_steps:7,solved:989}`、`material_passport.verification_status=VERIFIED`。
2. TVA 训练规模：输出 [sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_train_seed0.json)，记录训练源为 `datasets/sokoban_boxoban/boxoban-levels/medium/train/001.txt`、`max_levels=40`、`raw_samples=1378`、`samples_after_dedup=834`、`train_samples=653`、`valid_samples=181`、`value_train_samples=114`、`source_counts.online_root_state=794`、`source_counts.online_final_state=40`。checkpoint contract 记录 `teacher_used_online=false`、`teacher_used_for_training_target=false`。
3. TVA 独立审计规模：输出 [gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json](/home/data/round/code/sd_world_model_rl/results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json)，记录审计源为 `medium/train/002.txt` 前 `10` 关，每关 `12` 个叶状态，共 `120` 个叶状态；`teacher_max_expansions=4000`；teacher counts 为 `solved=78`、`teacher_budget_unsolved=42`；high-confidence count 为 `78`。审计 gate 为 `PASS`，TVA leaf high-confidence MAE 为 `0.48948865033750666`，GFT-SD leaf high-confidence MAE 为 `0.5517001792097376`。

文献对照数字来源于本地全文摘录：

1. Orseau et al. `Single-Agent Policy Tree Search With Guarantees`，本地文本 [orseau_etal_2018_single_agent_policy_tree_search.txt](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature/text/orseau_etal_2018_single_agent_policy_tree_search.txt)，题录为 arXiv `1811.10928v2`、`28 Nov 2018`，会议为 NeurIPS 2018；表 1 报告 LevinTS 与 LevinTS(*) 在 1000 关上均 `1000` solved。
2. Orseau and Lelis `Policy-Guided Heuristic Search with Guarantees`，本地文本 [orseau_lelis_2021_policy_guided_heuristic_search.txt](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature/text/orseau_lelis_2021_policy_guided_heuristic_search.txt)，会议为 AAAI-21；Sokoban 10x10 test 表报告 LevinTS、PHSh、PHS* 均 `1000` solved，其中 PHS* 平均 expansions 为 `1522.1`。
3. Guez et al. `An Investigation of Model-Free Planning`，本地文本 [guez_etal_2019_model_free_planning.txt](/home/data/round/code/sd_world_model_rl/docs/sokoban_sd_literature/text/guez_etal_2019_model_free_planning.txt)，表 2 报告 DRC(3,3) 在 unfiltered Sokoban 上 `1e9` steps 后 `99%` solved；数据规模实验记录 Large train set 为 `900k` levels。

论文表述口径：该对照表只支持“near-SOTA / competitive with SOTA-class search and model-free planning baselines, with much smaller learned TVA data and explicit structure”。它不支持严格 SOTA 断言，因为已发表的 LevinTS 与 PHS/PHS* 搜索方法在同类 1000 关测试上报告过 `1000/1000`。
