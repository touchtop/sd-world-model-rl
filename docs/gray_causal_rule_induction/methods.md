# Methods

## 研究对象

本研究线将 Hopper 规则生成过程表示为一个局部灰因果图，而不是完整 Hopper 环境因果图。图中的核心对象为上下文 `C`、动作族 `A`、短视界结构后果 `Y_h`、TVA 价值后果 `V_h`、规则状态 `R` 与白化强度 `rho`。

核心灰边为：

```text
C, do(A_family) -> Y_h / V_h
```

其中 `C` 来自规则谓词或子状态键，`A_family` 来自规则动作族，`Y_h` 包括支撑、旋转、碰撞、速度和 hop 相关结构变量，`V_h` 来自 TVA 终端价值或动作族排序。SD 反事实、真实 replay 后果和 TVA 排序一致性均被视为这条灰边的证据来源。

## 证书条件

第一版证书包含四个检查项。

机制准入要求规则指向明确上下文和动作族，并至少具有一种可审计机制证据，包括 SD 反事实支持、replay 后果支持或 TVA 动作族排序支持。该条件对应灰世界模型中的证据准入边界，避免仅凭最终回报相关性生成规则。

动作稳定要求规则触发上下文中的动作族排序具有足够低的不确定性。当前实现使用 `best_family_entropy`、方向一致率、TVA same/opposite margin 等代理量，后续可替换为 MPPI candidate ranking 的动作跨度。

价值可用要求规则触发或动作族偏好在短窗口后果中具有正边际。当前实现使用 replay 中的 `positive_reward_hit_rate`、`high_episode_hit_rate`、TVA mean advantage 和 good outcome rate 作为代理量。

结构安全要求规则不显著推高相反动作、错误干预或结构失败风险。当前实现使用 `direction_harmful_rate`、TVA opposite rate、good state hit rate 和 consequence gate 作为代理量。

## 输出语义

审计输出不直接改变在线规则库，只给出建议：

```text
strict_action_certificate
value_certificate
shadow_only
reject_or_retire
```

`strict_action_certificate` 表示机制准入、动作稳定和结构安全均通过，且规则可作为低强度 active prior 的候选。`value_certificate` 表示动作尚未完全稳定，但价值后果和安全条件足以支持 shadow 或低强度 active。`shadow_only` 表示已有机制迹象但证据不足。`reject_or_retire` 表示证据显示潜在有害或严重不稳定。

## 规则库变体

灰证书审计后可派生若干只读规则库变体。派生过程保留源规则库的顶层元数据结构，仅替换 `rules` 字段，以兼容现有在线 prior 加载器。当前使用的变体包括：

`drop_reject_active` 仅将 active 规则中证书为 `reject_or_retire` 的规则退役；`certified_active` 仅保留 active 规则中的 strict 或 value 证书规则；`strict_active` 仅保留 strict action 证书规则；`teacher_v8_active` 仅保留教师来源或 v8 transition replay-safety 来源的 active 规则；`teacher_v8_certified_active` 在来源过滤基础上进一步要求 strict 或 value 证书。

前两类为保守过滤，目标是清理局部有害规则而不改变规则库主体。后两类为强筛选对照，用于检验来源偏置或过度收缩是否损害规则 prior 的可用性。

## Eval-only 验证

第一轮 eval-only 验证使用 `cycle_080` 的 400k `final.full.pt` checkpoint，比较 `cycle_079` 原始 active 规则库与灰过滤变体。为隔离规则库差异，所有模式均使用同一 `cycle_079` SD checkpoint 和 TVA terminal prior checkpoint；`rule_off` 模式关闭规则 prior，但保留 TVA terminal prior。

第二轮 eval-only 验证使用 `cycle_081` 的 `final.full.pt` checkpoint，比较 `cycle_080` 原始 active 规则库、`drop_reject_active` 和 `strict_active`。所有模式均使用同一 `cycle_080` SD checkpoint 与 TVA terminal prior checkpoint。该轮用于检验灰证书过滤在相邻闭环周期上的稳定性，而不是检验完整训练收益。

该 eval-only 验证不训练、不改写主线实验目录，所有报告写入：

```text
results/gray_causal_rule_induction/eval_only/
```

当前 smoke 使用每种子 3 episode。该设置只能作为方向性证据，不能替代后续 10 episode 或短接续训练验证。

截至 `cycle080 -> cycle081` smoke，`strict_active` 在相邻周期上保持轻微正向信号，但 `drop_reject_active` 未复现上一轮正向结果。因此，后续方法学重点应转向以严格动作证书为核心的规则生成或规则激活，而不将所有保守过滤策略合并解释为同一结论。

## 与当前 TD-MPC2 实验的边界

本研究线不修改 SD 主模型、不修改 TD-MPC2 reward、不替换动作、不修改正在运行的启动脚本。所有结果均写入 `results/gray_causal_rule_induction/` 或用户显式指定的新路径。

## 闭环训练接入

闭环训练接入采用包装入口，而不是改写原 `run_hopper_v7_multiseed_shared_rules.py`。包装入口复用原训练、SD 拟合、TVA 拟合、规则挖掘、confidence 更新、TVA shadow update 和 TVA gate 流程；唯一新增环节发生在下一 cycle 训练前：上一 cycle 生成的 confidence 规则库被灰证书审计，随后导出 `strict_active` 规则库，并将该派生规则库作为在线 rule prior 输入。

首个 cycle 不进行灰过滤，因为尚无该运行自身的 replay、TVA 与 consequence 证据。`cycle_001` 因而沿用初始 rich teacher confidence 规则库；从 `cycle_002` 起，在线规则 prior 指向上一 cycle 的 `gray_causal_rule_induction/rule_library_variants/cycleXXX_strict_active.json`。该设计避免使用后验 cycle080/cycle081 规则信息，也避免改动既有 SD 模型本体。

## 有限记忆样本选择

闭环规则生成链条采用有限记忆样本选择，避免随着训练步数增长而将所有历史轨迹持续纳入 SD、TVA、反事实生成和规则审计。该机制不删除原始 replay trace，不修改正式评估记录，也不回写旧周期结果；它只限制 `selected_trajectories.json` 中进入后处理链条的 episode 数。

默认选择流程仍先按 cycle step 计算 good/bad 分位阈值，并保留 good、bad 与少量 random middle 样本。随后若存在 `selection_memory_policy.json` 或显式命令行参数，则执行第二阶段记忆控制：保留 `preserve_sources` 指定来源，按标签平衡保留 good 与 bad 中最极端、最有对比度的 episode，并在总量超限时优先保留较新的 cycle step。

当前 1M 接续实验使用的策略为：`max_selected_per_cycle_step=600`，`max_selected_episodes=1800`，`preserve_sources=["official_eval"]`。该设置将最近三周期窗口中的后处理 episode 数从约 `5200` 控制到约 `1800`，目标是在保留规则生成所需高对比证据的同时，将反事实生成和 replay scoring 的端到端成本压回可控范围。

## 多 CPU 后处理

有限记忆控制样本规模，多 CPU 后处理控制剩余样本的处理吞吐。规则生成链条中的若干脚本具有天然 episode 或 trace 独立性，因此采用 `ProcessPoolExecutor` 进行数据并行。默认 worker 数通过 `resolve_workers` 解析：若命令行未显式指定，则读取 `SD_RETRO_WORKERS`；若环境变量也未设置，则使用 `min(16, os.cpu_count())`。该默认值利用多核服务器资源，但避免占满全部逻辑核心。

当前并行化范围包括 SD 反事实动作比较、可靠规则 replay scoring、子状态规则验证、v8 success fragment mining 和 widefunnel rule-effect audit。反事实动作比较在每个 worker 中初始化一次 SD explainer；规则 replay scoring 改为每个 episode 只读取一次 trace，再同时评分所有规则，以减少重复 I/O。

所有并行化脚本保留 `--workers 1` 单进程回退路径。小样本一致性验证显示，除输出中新增 `workers` 记录字段外，单进程和多进程核心统计一致。
