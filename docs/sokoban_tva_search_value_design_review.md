# Sokoban Search-Value TVA Revision Review

## Material Passport

- Origin Skill: academic-research-suite
- Routed Workflows: experiment-agent plan/validation, academic-paper-reviewer methodology review, devil's advocate stress test, editorial synthesis
- Origin Date: 2026-06-10
- Verification Status: REVIEWED_DESIGN_ONLY
- Version Label: sokoban_search_value_tva_revision_review_v1

## 修订方案提交稿

本轮修订将当前 `sokoban_tva_value_v1` 与 `sokoban_tva_value_v2_failed_relabel` 重新标注为终端价值代理对照，而不是严格意义上的 Terminal Value Alignment。正式 TVA 的下一版命名为 `sokoban_search_value_tva_v1`，其训练目标必须由 SD-MCTS 搜索树统计构造，不再以 best-first diagnostic teacher 的静态剩余解长作为 TVA 主目标。

修订后的 TVA 保持以下边界：

1. `LearnedStockFlowTransitionProvider` 不变，MCTS 内部抽象转移仍来自 learned stock-flow checkpoint；
2. raw Sokoban state handle 只用于合法 push 枚举和下一层动作支持，不进入抽象 transition/value 的评分；
3. best-first diagnostic teacher 不在线进入 SD-MCTS，也不作为 search-value TVA 主训练目标；
4. TVA 网络只作为 MCTS 截断叶节点的 `V_psi(s_leaf)`，不替代世界模型，不替代动作启发函数。

严格 TVA 目标按论文公式实现：

```text
V_search(s_t) = sum_a pi_mcts(a | s_t) Q_H(s_t, a)
pi_mcts(a | s_t) = N(s_t, a)^alpha / sum_b N(s_t, b)^alpha

L_val =
  MSE(V_psi(s_t), stop_gradient(V_search(s_t)))
  + lambda_td * MSE(V_psi(s_t), r_t + gamma * Vbar_psi(s_{t+1}))
```

其中 `Q_H(s_t,a)` 必须取自根节点 child 的回传均值，语义为执行根动作后的折扣回报估计。现有 `choose_mcts_action()` 中 child backup 的值已经包含该根动作 reward 与后续 bootstrap，但实现前必须增加 depth-1 单元审计，验证 `child.value == reward(root,a) + gamma * V(child_state)` 在深度 1 情况下成立。

## 实现修订清单

1. 新增 `run_mcts_root_search_stats()` 或等价 helper，从现有 MCTS 中只收集根节点统计，不执行真实环境动作。
2. 新增 `compute_search_value_target(root_policy, alpha)`，只使用 `visits` 与 `q_value` 计算 `V_search`；`probability` 字段可记录原始 visit fraction，但训练目标必须按 `alpha` 重新计算。
3. 新增 `train_sokoban_search_value_tva.py`，避免继续扩展 teacher proxy 脚本造成语义混淆。
4. 每个训练样本保存 `features`、`v_search`、`root_policy`、`root_children`、`search_budget`、`search_depth`、`provider_id`、`terminal_value_id`，若启用 TD，还保存 `selected_action`、`reward`、`next_features`、`terminal_flag`。
5. 价值网络初始化可使用 v1 checkpoint 作为共享先验，但 checkpoint contract 必须记录 `initialization_source`，并明确 `target_source=sd_mcts_root_statistics`。v1/v2 不得被记录为 search-value target。
6. 默认训练损失启用 search alignment；TD 项以 `lambda_td` 显式记录，并使用 frozen target network `Vbar_psi`。若为了首轮诊断设 `lambda_td=0`，必须在文档中写明这是论文目标的消融，而不是完整 TVA。
7. 阶段 3 复测继续使用 `evaluate_stage3_sdmcts_planning.py --terminal-value-mode tva`，但 checkpoint id 必须为 `sokoban_search_value_tva_v1`，并报告 `terminal_value_usage_gate`、`learned_provider_fraction`、成功率、失败类型、叶节点类型分布和 root policy 统计。

## Methodology Review

### 研究问题对齐

修订方案与当前研究目标一致。研究目标不是用启发式搜索解 Sokoban，而是检验抽象高层状态下，系统动力学世界模型如何与 SD-MCTS 形成可学习的任务导向转移与价值闭环。search-value TVA 将价值网络训练目标绑定到 MCTS 在 learned SD world model 上形成的根节点搜索估计，避免把静态 teacher value 误写成 TVA。

### 结构合理性

修订方案保留了 stock-flow-feedback 世界模型的边界。六核心 stock、固定 incidence matrix、非负 flow、capacity projection 与 learned provider 仍是转移模型；TVA 输入中的 structural scalars 只作为 value features，不改变 transition provider。因此该方案没有把世界模型退化为启发式函数。

### 训练目标合理性

`V_search` 目标是必要项，因为论文中 TVA 的关键语义是让 `V_psi` 对齐 MCTS 搜索统计，而不是对齐外部 teacher 的单点标签。TD 项也是必要的结构项，因为它为相邻状态之间提供一致性约束；若首轮为降低复杂度只运行 `lambda_td=0`，应作为消融记录，不能称为完整论文 TVA。

### 数据与评估合理性

训练根状态应来自 provider checkpoint 的 train levels，held-out valid levels 只用于评估。MCTS target collection 必须记录 provider id 与 terminal evaluator id，以审计 target 是否来自 learned SD-MCTS 而非 exact teacher。评估仍需同时包含 learned、oracle abstract、zero-flow 和 heuristic diagnostic 对照，但 heuristic 只能作为诊断参照。

### 可复现性要求

所有脚本必须固定随机种子，记录 conda 环境 `/home/anaconda3/envs/sdmcts_p100/bin/python`、level split、search budget、MCTS depth、`alpha`、`gamma`、`lambda_td`、bootstrap checkpoint、target network 同步策略和输出路径。训练输出 JSON 必须包含 target summary、root child coverage、finite target fraction、provider usage summary 和 checkpoint contract。

## Devil's Advocate Stress Test

### Strongest Counter-Argument

修订方案虽然恢复了论文 TVA 的形式语义，但仍可能只是把当前失败的 SD-MCTS 搜索行为蒸馏进价值网络。如果六核心 stock 抽象缺少箱子具体拓扑、隧道、goal-room 和多箱死锁等可区分信息，那么 `V_search` 本身可能已经是低质量目标。此时 search-value TVA 会稳定地复制短视界搜索偏差，而不一定产生更好的长期规划能力。另一方面，若第一轮 target collection 使用 v1 checkpoint 初始化或 bootstrap，teacher proxy 的偏差仍可能通过叶节点 bootstrap 间接进入 `Q_H`。因此，修订方案必须把“严格按论文构造目标”与“目标质量足以改善规划”区分开：前者是实现合规门限，后者必须由 held-out closed-loop 结果和 oracle abstract 对照验证。

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Location |
|---|---|---|---|
| 1 | TVA 语义 | 若不实现 `V_search=sum pi_mcts Q_H`，则仍不是论文 TVA，只是 value proxy。 | training target |
| 2 | 在线边界 | 若 best-first teacher 进入 MCTS action choice 或 root scoring，主线会退化为启发式搜索。 | planning loop |

#### MAJOR

| # | Dimension | Issue Description | Location |
|---|---|---|---|
| 1 | 自举偏差 | 使用 v1/v2 初始化或 bootstrap 时，teacher proxy 偏差可能间接污染 `Q_H`。需记录并做 stock/zero/bootstrap 消融。 | target collection |
| 2 | Q 值语义 | 现有 root child `q_value` 必须经单元审计确认是根动作回报，而不是 child state value 的未折扣读数。 | MCTS backup |
| 3 | 抽象充分性 | 六 stock 可能无法区分关键 Sokoban 拓扑状态，`V_search` 可能受 aliasing 限制。需报告 root-state alias bucket 与 target 方差。 | value features |
| 4 | TD 一致性 | 若省略 TD target，应标注为 `lambda_td=0` 消融；若启用 TD，需要 target network 防止自举振荡。 | loss |

#### MINOR

| # | Dimension | Issue Description | Location |
|---|---|---|---|
| 1 | 命名 | v1/v2 文件名含 `tva` 容易误导，应在文档中固定称为 terminal value proxy。 | docs/results |
| 2 | 指标 | 只看 MAE/corr 不足，应补充 solved 类均值、target finite fraction、root child coverage。 | metrics |

### Ignored Alternative Explanations/Paths

1. 阶段 3 失败可能来自 action abstraction 缺失，而不是 TVA target。若 search-value TVA 仍失败，应优先检查 macro/goal-room/relevance 是否应作为 SD feedback 或 option action，而不是继续调 value loss。
2. oracle abstract SD-MCTS 也失败，说明 learned transition 误差不是唯一瓶颈。search-value TVA 的评估必须包含 oracle abstract 对照，否则无法判断 value 学习是否真正改善了规划消费层。

## Editorial Synthesis

### 综合判定

该修订方案可以作为下一步实现依据，但必须按“先合规、再性能”的顺序推进。第一验收目标是修复 TVA 的方法语义：训练目标来自 MCTS root statistics，并能审计 `Q_H`、`pi_mcts`、`V_search` 和 TD target。第二验收目标才是 closed-loop 成功率或 oracle abstract 改善。不得用 v1/v2 的 teacher target、failed relabel target 或启发式 best-first 结果替代 search-value alignment。

### 必须执行的门限

1. Strict TVA contract gate：checkpoint 中 `target_source=sd_mcts_root_statistics`，`mcts_root_statistics_used=true`，`teacher_used_online=false`，`transition_provider_modified=false`。
2. Q-value semantics gate：depth-1 审计通过，root child `q_value` 等于根动作 reward 加折扣叶值。
3. Provider gate：target collection 与 closed-loop eval 中 learned provider fraction 为 `1.0`。
4. Target coverage gate：有效 root children fraction 为 `1.0` 或明确列出无 child 状态；finite `V_search` fraction 为 `1.0`。
5. Solved protection gate：valid solved 类预测均值不得为负，且必须报告 solved、dead-end、budget-unsolved 分组指标。
6. Planning gate：若 held-out `learned_sdmcts` 成功率仍为 `0.0`，不能判定阶段 3 通过；若 oracle abstract + search-value TVA 有稳定改善但 learned 无改善，应记录为 transition/value mismatch，而不是直接宣布方法成功。

### 执行路线

1. 文档修订：将 v1/v2 固定改名为 terminal value proxy 对照，新增 search-value TVA 合同。
2. 代码修订：新增 search stats collection、`V_search` target builder、TD target network 和 checkpoint contract。
3. Smoke 审计：运行 depth-1 Q 语义测试、target collection smoke 和 compileall。
4. 训练复测：在 `sdmcts_p100` 环境训练 `sokoban_search_value_tva_v1`，随后在同一 held-out 6 levels 上执行 learned/oracle/zero-flow 对照。
5. 结果审查：无论正负，都同步更新 `methods.md`、`decisions.md`、`reproducibility.md` 和 `research_log.md`，并保留 v1/v2 作为对照，不覆盖旧结果。
