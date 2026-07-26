# Research Log

## 2026-06-04

### 建立面向规则生成的灰因果图研究线

新研究线被建立为独立文件系统，代码目录为 `experiments/gray_causal_rule_induction/`，文档目录为 `docs/gray_causal_rule_induction/`，结果目录为 `results/gray_causal_rule_induction/`。该结构与当前正在运行的 `sd_retrospective_rule_induction` 主线平行，避免覆盖既有 SD 轻盈介入 TD-MPC2 实验的代码、脚本、日志和结果。

本轮决策为：暂不改动现有 SD 模型主体，也不派生替代在线训练流程的灰 SD 子类。当前 SD 已在既有 Hopper 实验中表现出可用的短视界结构预测与反事实排序能力，因此优先将灰因果图用于规则生成、规则筛选和规则强度控制。灰因果图层只读复用既有规则库、SD 反事实审计、replay 规则效果审计和 TVA 审计产物。

研究线的第一版实现目标是形成 `GrayRuleCertificate` 审计：对每条规则建立上下文与动作族到短视界结构后果及 TVA 价值后果的灰边，检查机制准入、动作稳定、价值可用和结构安全四类证据，并据此给出 `strict_action_certificate`、`value_certificate`、`shadow_only` 或 `reject_or_retire` 的建议。该建议只写入新结果目录，不回写原规则库。

### 完成第一版只读证书审计

新增代码文件：

```text
experiments/gray_causal_rule_induction/gray_rule_graph.py
experiments/gray_causal_rule_induction/audit_gray_rule_certificates.py
```

第一版代码实现 `GrayRuleCertificateBuilder` 和命令行审计脚本。脚本读取既有规则库、规则效果审计、TVA shadow update、TVA consequence gate 与 TVA terminal gate，并将证书结果写入新结果目录。脚本包含输出路径保护，拒绝将灰规则证书写回 `sd_retrospective_rule_induction` 目录。

已使用 `cycle_075` 产物完成一次 sanity check。输入规则库为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_075/rules/library_closed_loop_cycle_075_confidence.json
```

输出为：

```text
results/gray_causal_rule_induction/cycle075_gray_rule_certificates.json
results/gray_causal_rule_induction/cycle075_gray_rule_certificates.md
```

审计覆盖 `267` 条规则。第一版证书分布为：`strict_action_certificate=157`、`value_certificate=2`、`shadow_only=92`、`reject_or_retire=16`。其中 `confidence_active` 规则中，`strict_action_certificate=126`、`value_certificate=1`、`reject_or_retire=3`。该结果用于验证证书管线可运行和可审计，尚不直接作为在线规则过滤标准。

实现过程中对 TVA 证据准入作出一次收紧：TVA shadow 文件中的命中计数本身不再被视为独立 TVA 机制证据，只有 TVA effect ratio、mean advantage 或动作族方向边际提供实际信号时，才计入 TVA 证据来源。该修订用于保持证书解释与灰世界模型中的机制证据准入一致。

### 完成 cycle079 灰过滤规则库与 400k eval-only smoke

基于 `cycle_079` 的完整规则库与审计产物，生成灰证书与规则库变体。灰证书输出为：

```text
results/gray_causal_rule_induction/cycle079_gray_rule_certificates.json
results/gray_causal_rule_induction/cycle079_gray_rule_certificates.md
```

规则库变体输出目录为：

```text
results/gray_causal_rule_induction/rule_library_variants_cycle079/
```

本轮修订了 `export_gray_filtered_rule_libraries.py` 的输出格式。第一版变体曾导出为裸列表，但现有在线 prior 加载器要求顶层对象包含 `rules` 字段。修订后派生规则库保留源规则库的顶层元数据结构，仅替换 `rules` 字段并增加 `gray_filter_export` 标记。该修订保证灰规则变体可直接用于现有 eval-only 入口，不需要修改 `online_rule_prior_v6.py`。

随后新增 `eval_gray_rule_variants.py`，复用 `eval_hopper_v7_rule_prior.py`，在 `cycle_080` 的 400k checkpoint 上比较 `cycle_079` 原始 active 规则库与灰过滤变体。所有模式均启用同一 `cycle_079` TVA terminal prior，以隔离规则库差异；`rule_off` 仅关闭规则 prior。由于当前机器上 `MUJOCO_GL=egl` 无法初始化 headless EGL，eval-only 使用与主线可运行配置一致的 `MUJOCO_GL=glfw`。

输出目录为：

```text
results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/
results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3_summary.json
results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3_summary.md
```

3 episode smoke 的三种子均值如下：`rule_off=359.080`，`cycle079_original_active=355.452`，`cycle079_drop_reject_active=363.974`，`cycle079_certified_active=363.974`，`cycle079_strict_active=364.232`，`cycle079_teacher_v8_active=360.611`，`cycle079_teacher_v8_certified_active=353.022`。相对原始 active 规则，保守灰过滤提升约 `+8.523`，strict 变体提升约 `+8.780`；相对 rule-off，保守灰过滤提升约 `+4.894`，strict 变体提升约 `+5.152`。

该结果支持继续推进灰因果规则生成方向，但其证据等级仍为 smoke。主要结论是：只退役灰证书判为 `reject_or_retire` 的 4 条 active 规则，即可恢复并略超同 TVA 下的 rule-off 表现；强行只保留教师或 v8 来源规则会弱化收益；在来源过滤基础上进一步证书过滤则表现低于原始 active 和 rule-off，显示过度收缩风险。

### 核对 400k 与生成 cycle080 灰证书

400k / `cycle_080` 的训练、正式评估与后处理均已完成。`cycle_080` 已落盘 `library_closed_loop_cycle_080_confidence.json`、`rule_effect_audit.json`、`tva_shadow_update.json`、`tva_shadow_consequence_gated_update.json` 和 `tva_terminal_prior_gate.json`。主线随后已经启动 `cycle_081`，目标步数为 `405080`。

基于 `cycle_080` 完整产物生成灰证书：

```text
results/gray_causal_rule_induction/cycle080_gray_rule_certificates.json
results/gray_causal_rule_induction/cycle080_gray_rule_certificates.md
```

`cycle_080` 共覆盖 `271` 条规则，证书分布为：`strict_action_certificate=156`、`value_certificate=4`、`shadow_only=92`、`reject_or_retire=19`。在 `confidence_active` 规则中，`strict_action_certificate=125`、`value_certificate=1`、`reject_or_retire=4`。

同步导出 `cycle_080` 规则库变体：

```text
results/gray_causal_rule_induction/rule_library_variants_cycle080/
```

保守变体 `drop_reject_active` 与 `certified_active` 均保留 `126` 条 active 规则、退役 `4` 条 active 规则；`strict_active` 保留 `125` 条 active 规则；`teacher_v8_active` 保留 `31` 条 active 规则；`teacher_v8_certified_active` 保留 `28` 条 active 规则。该结构与 `cycle_079` 的灰过滤结果基本一致，说明灰证书在相邻周期上具有一定稳定性。

### 泛化 eval-only 入口并准备 cycle080 后续验证

在保守灰过滤 smoke 获得正向结果后，`eval_gray_rule_variants.py` 被改造为通用 eval-only 入口。新增参数包括 `--rule-cycle`、`--tva-cycle`、`--checkpoint-cycle` 和 `--checkpoint-run-name`，可在不改动脚本常量的情况下评估任意相邻周期的灰规则变体。默认行为仍保持为 `cycle_079` 规则变体在 `cycle_080` / 400k checkpoint 上的 3 episode smoke，因此既有结果可继续复用。

已完成两个 dry-run 检查。第一，默认模式仍正确指向 `cycle_079` 规则变体和 `cycle_080` checkpoint。第二，`--rule-cycle 80 --tva-cycle 80 --checkpoint-cycle 81` 能生成 `cycle_080` 规则变体在 `cycle_081` checkpoint 上的 eval-only 命令，输出目录默认为：

```text
results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/
```

当前主线 `cycle_081` 仍在 GPU 1、2、3 上训练，且尚未产生 `final.full.pt`，因此暂不启动新 eval-only 任务，以免干扰正在运行的闭环实验。后续在 `cycle_081` checkpoint 完整落盘后，可优先运行 `rule_off`、`cycle080_original_active`、`cycle080_drop_reject_active` 和 `cycle080_strict_active` 四个模式。

随后复查发现 `cycle_081` 三个种子的 `final.full.pt` 已落盘，训练阶段结束；主线自动流程进入 replay trace 导出和后处理阶段。由于该闭环 launcher 会在后处理完成后自动启动下一周期训练，仍不在灰线中插入 GPU eval-only 任务，以避免与 `cycle_082` 争用 GPU。当前灰线状态为：已准备好 `cycle_080` 规则变体评估 `cycle_081` checkpoint 的通用命令，等待明确 GPU 空窗或主线暂停后执行。

### 完成 cycle080 规则变体在 cycle081 checkpoint 上的 eval-only 验证

在用户确认 GPU0 仅存在显存占用而无计算负载后，使用 GPU0 执行 `cycle_080` 规则变体到 `cycle_081` checkpoint 的 3 episode、三种子 eval-only smoke。执行期间 GPU0 维持实际计算负载，评估进程均正常完成；其他主线 GPU 未被灰线评估调用。

输出目录为：

```text
results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/
results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep_summary.json
results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep_summary.md
```

四个模式的三种子均值如下：`rule_off=354.215`，`cycle080_original_active=351.972`，`cycle080_drop_reject_active=350.000`，`cycle080_strict_active=355.277`。对应加载规则数分别为 `0`、`130`、`126` 和 `125`，与 `cycle_080` 灰证书导出的规则库规模一致。

本轮结果与 `cycle_079 -> cycle_080` smoke 并不完全一致。`strict_active` 相对原始 active 提升约 `+3.305`，相对同 TVA 下 rule-off 提升约 `+1.062`，仍显示严格灰证书过滤有继续研究价值；但 `drop_reject_active` 相对原始 active 下降约 `-1.972`，相对 rule-off 下降约 `-4.215`，不支持将“仅退役 reject_or_retire active 规则”直接作为默认策略。由于每种子仅 3 episode，当前结论仍应视为方向性证据。

阶段性判断为：灰因果规则层值得继续推进，但下一步重点应收窄到 `strict_active` 或以严格证书为核心的规则生成机制，而不是泛化为所有保守过滤策略均有效。后续若进入短接续训练，应优先选择 `strict_active` 与 rule-off、原始 active 对照，并增加 episode 数或训练窗口以降低采样波动。

### 终止原三种子主线并启动 300k 灰规则闭环训练

根据用户指令，原 `20260603_r1` 三种子主线实验在 `cycle_081` 后处理阶段被终止。终止前，`cycle_081` 三个种子的 `final.full.pt` 均已落盘，训练已经完成到 `405080` 步；当时仍在运行的子任务属于规则后处理链，而非 GPU 训练。终止操作仅作用于旧主线 launcher 及其子进程。

新增闭环训练入口：

```text
experiments/gray_causal_rule_induction/run_hopper_v7_multiseed_gray_rules.py
experiments/gray_causal_rule_induction/launch_hopper_300k_gray_strict.sh
```

该入口采用包装方式复用原 `run_hopper_v7_multiseed_shared_rules.py`，不改写原主线脚本。新机制在每个 cycle 的 confidence 规则库生成后，于下一 cycle 训练前执行灰证书审计与规则库导出，并选择 `strict_active` 变体作为在线规则 prior。由于 `cycle_001` 尚无本运行自身的 replay、TVA 与 consequence 证据，首轮仍沿用原 300k 口径中的 rich teacher confidence 初始规则库；灰因果规则机制从 `cycle_002` 起接入。

新实验以原 300k 口径重新训练：三种子 `1,2,3`，默认 GPU `1,2,3`，`60` 个 cycle，每 cycle `5000` 步，正式评估频率与 checkpoint 频率均为 `25000` 步，保留 SD、TVA、TVA terminal gate、TVA shadow update、v8 transition rules 与 rule-effect confidence 配置。输出目录为：

```text
results/gray_causal_rule_induction/closed_loop/hopper_v8formal_300k_graycausal_strict_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260604_r1/
```

启动前完成 `py_compile` 与 2-cycle dry-run。dry-run 验证 `cycle_002` 的 `cycle_config.json` 已包含 `gray_causal_rule_prior.online_rule_library`，并正确指向上一 cycle 的 `cycle001_strict_active.json`。实际训练随后以脱离会话方式启动，launcher PID 为 `787626`。启动检查显示三个 seed 均进入 `cycle_001` 训练，在线规则库为初始 rich teacher confidence library，加载 active 规则数为 `22`。

### 核对 25k 正式评估结果

新 300k 灰规则闭环训练的 `cycle_005_target25004` 已产生三种子正式评估文件。核对 `eval.csv`、`eval_full.csv` 与 `official_eval/step_0025004/eval_episode_summary.csv` 后确认，25k 评估不是严格全零，但整体处于近零水平。

三种子的 10 episode 正式评估均值分别为：seed 1 为 `0.8159493334707804`，seed 2 为 `0.002356162085197866`，seed 3 为 `0.0`。其中 seed 1 有 2 个非零 episode，奖励为 `1.546466596540995` 与 `6.613026738166809`；seed 2 有 1 个极小非零 episode，奖励为 `0.02356162085197866`；seed 3 的 10 个 episode 均为 `0.0`。三种子的成功率均为 `0.0`，平均 episode 长度均为 `500.0`。

因此，25k 评估应表述为“近零但非严格全零”。该结果说明早期 25k 阶段尚未形成有效跳跃策略，后续判断需继续依赖 50k、75k 等正式评估点及相邻 cycle 的规则证书变化。

## 2026-06-11

### 为 1M 接续实验加入有限记忆样本控制

对 500k 接续至 1M 的三种子闭环实验进行端到端耗时复查。以 `cycle n` 开始到 `cycle n+5` 开始作为 25k 墙钟时间，早期 `cycle101-cycle105` 为 `7.77 h`，近期 `cycle141-cycle145` 为 `11.00 h`，整体后处理成本已显著上升。

阶段拆分显示，主要增长来自闭环后处理样本规模扩大，而不是单纯规则条数膨胀。选中 episode 从 `cycle101` 的 `1217` 增至 `cycle146` 的 `5255`，反事实记录从 `12170` 增至 `52550`；对应 `counterfactual_actions` 从约 `11.2 min` 增至约 `49-51 min`，`replay_rule_scoring` 从约 `2.1 min` 增至约 `11 min`。

为避免后处理无限保留历史样本，`stratify_trajectories.py` 增加有限记忆选择策略。该策略在原 good/bad/random 分层后执行，不改变原始 manifest 和 replay trace，只限制进入 SD、TVA、反事实与规则挖掘链条的 episode 数。当前策略保留正式评估样本，每个 cycle step 最多保留 `600` 条，总数最多保留 `1800` 条，并优先保留 good/bad 高对比样本。

当前 1M 实验根目录新增策略文件：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_1m_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume500k_20260607_r1/selection_memory_policy.json
```

该文件仅影响后续周期的样本选择，不回写 `cycle146` 及更早周期。使用 `cycle146` manifest 进行临时验证时，样本选择从 `5255 / 13080` 收缩为 `1800 / 13080`；保留后标签分布为 good `897`、bad `885`、middle/random `18`，未形成明显单侧偏置。

同时为 `run_hopper_v7_closed_loop.py` 和 `run_hopper_v7_multiseed_shared_rules.py` 增加显式参数接入，便于未来新实验通过命令行指定 `--selection-memory-policy`、`--max-selected-episodes` 和 `--max-selected-per-cycle-step`。当前正在运行的 launcher 已在内存中加载旧调度函数，因此本轮运行主要依赖策略文件自动发现机制，使后续 `stratify_trajectories.py` 子进程读取该策略。

### 为规则后处理加入多 CPU 并行

在有限记忆策略之外，进一步为主要 CPU 后处理脚本加入多进程并行。服务器逻辑核心数为 `72`，因此默认并行度采用 `min(16, os.cpu_count())`，也可通过环境变量 `SD_RETRO_WORKERS` 或命令行 `--workers` 覆盖。该设置避免后处理继续单核运行，同时不默认占满所有逻辑核心。

新增公共 worker 解析函数：

```text
experiments/sd_retrospective_rule_induction/common.py::resolve_workers
```

已改造脚本包括：

```text
experiments/sd_retrospective_rule_induction/counterfactual_actions.py
experiments/sd_retrospective_rule_induction/replay_rule_scoring.py
experiments/sd_retrospective_rule_induction/validate_substate_rules.py
experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py
experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py
```

其中 `counterfactual_actions.py` 按 episode 并行，每个 worker 初始化一次 SD explainer；`replay_rule_scoring.py` 从原先“每条规则重复读取所有轨迹”改为“每个 episode 读取一次并同时评分所有规则”；`validate_substate_rules.py`、`mine_hopper_v8_success_segments.py` 与 `audit_hopper_v8_widefunnel_rule_effects.py` 均按 episode 或 trace 并行汇总。

`run_hopper_v7_closed_loop.py` 与 `run_hopper_v7_multiseed_shared_rules.py` 增加 `--postprocess-workers` 参数，并在 dry-run 中确认 `counterfactual_actions.py`、`replay_rule_scoring.py`、`validate_substate_rules.py`、`mine_hopper_v8_success_segments.py` 和 `audit_hopper_v8_widefunnel_rule_effects.py` 均会收到 `--workers` 参数。当前已运行的 1M launcher 已在内存中加载旧调度函数，但这些脚本本身默认自动并行，因此从后续实际调用开始仍会生效。

完成小样本一致性验证。基于 `cycle146` 的 12 个 selected episode 和 8 条 trace 构造 `/tmp/sd_retro_parallel_test/` 临时输入，比较 `--workers 1` 与 `--workers 2/4` 输出；除新增记录字段 `workers` 外，五个脚本的核心 JSON 输出均完全一致。

当前主线状态同时更新：`cycle146` 后处理已完成并进入 `cycle147 / target=735146` 三种子训练。有限记忆策略将在 `cycle147` 训练结束后的样本选择阶段首次正式作用于主线后处理。
