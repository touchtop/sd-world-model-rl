# SD_MCTS260503 修订说明

本文档记录针对 `SD_MCTS260503` 论文版本的独立修订建议。阐述性说明使用中文；可直接替换或补入论文的具体内容使用英文，以保持与论文正文语言一致。

## 1. C.2 Compute Resources

### 1.1 修订指引

将 C.2 中 Compute Resources 段落的四处 `[XX]` 替换为已核算的计算资源数值。建议口径如下：Beer Game 单次运行写为 `0.04--0.31` GPU hours，Cartpole 单 target seed 写为 `0.61` GPU hours，Hopper-Hop `50k` 单种子写为 `3.9` GPU hours，主结果总预算写为 `14.8` GPU hours。

同时建议补充两点解释：Cartpole 记录中的三 target seeds 为串行执行，若三张 GPU 并行运行，墙钟时间约可缩短到三分之一；Hopper-Hop 将 SD 机制模型嵌入 TD-MPC2 在线训练和规划循环中，接口适配、结构态转换、辅助监督和详细日志记录尚未充分优化，因此相对 plain TD-MPC2 有额外运行开销。

### 1.2 建议替换文本

建议将 C.2 中 Compute Resources 段落替换为：

```text
Compute Resources All experiments were conducted locally on an internal server equipped with four NVIDIA Tesla P100-PCIE-16GB GPUs; no cloud computing APIs were utilized. Based on the recorded logs, a single Beer Game run over the 1k--10k interaction-budget range required approximately 0.04--0.31 GPU hours, the Cartpole shared-prior target-seed evaluation required approximately 0.61 GPU hours per seed, and a Hopper-Hop run to the 50k checkpoint required approximately 3.9 GPU hours. The total computational budget for the main SD-MCTS results at these checkpoints is estimated to be 14.8 GPU hours, including 1.3 GPU hours for the Beer Game three-seed budget sweep, 1.8 GPU hours for the Cartpole three-seed shared-prior evaluation, and 11.7 GPU hours for the Hopper-Hop three-seed 50k runs.
```

建议在同一段后追加：

```text
The Cartpole runs were executed serially in the recorded experiment; running the three target seeds in parallel would reduce the wall-clock time by roughly a factor of three. The Hopper-Hop runtime is higher than a plain TD-MPC2 run because the SD mechanism model is executed through an overlay interface inside the TD-MPC2 online training and planning loop, where the adaptation layer, structured-state conversions, auxiliary supervision, and detailed logging were not yet fully optimized for throughput.
```

## 2. C.3 Reproducibility Index

### 2.1 修订指引

C.3 的 Reproducibility Index 应改为面向 `Attachments/` 的独立复现索引，不再使用作者工作目录中的旧路径。建议将当前 “To facilitate independent reproducibility...” 之后的项目列表整体替换为附件包中的相对路径索引，并补入当前附件包已覆盖但原 C.3 未列出的 Beer Game 样本效率与 Hopper-Hop mechanism-oriented SD 主线。

### 2.2 建议替换文本

建议将 C.3 中 Reproducibility Index 的项目列表替换为：

```text
To facilitate independent reproducibility, the core results of this paper are indexed by the supplemental package submitted with the paper. All paths below are relative to the root of `Attachments/`.

• Beer Game sample efficiency, Table 1, Table 4, and Figure 3: Entry script `code/experiments/run_comparison_revised.py`; supporting results and figure files `data/beer_game/sdmcts_rerun_results.json`, `data/beer_game/dreamerv3_ppo_historical_results.json`, `data/beer_game/table1_table4_merged_results.json`, `data/beer_game/table1_sample_efficiency.tex`, and `data/beer_game/figure3_sample_efficiency.png`.
• Beer Game physical consistency and Figure 2(a): Entry script `code/experiments/run_internal_physics_consistency.py`; supporting result file `data/physics_consistency/results.json`.
• Beer Game multi-step rollout error and Figure 2(b): Entry script `code/experiments/run_multi_step_rollout.py`; supporting result files `data/multi_step_rollout/multi_step_results.json`, `data/multi_step_rollout/error_curves.csv`, and `data/multi_step_rollout/figure2b_multi_step_rollout_rmse.png`.
• Structural intervention results, Table 2, and Figure 4: Entry script `code/experiments/run_causal_intervention.py`; supporting result files `data/interventions/results.json` and `data/interventions/figure4_structural_intervention_cost.png`.
• Cartpole main results and Table 5: Entry script `code/scripts/run_cartpole_sharedprior_paper.py`; supporting result file `data/cartpole/cartpole_sharedprior_summary.json`.
• Cartpole 1-step supplementary ablation: Entry script `code/scripts/run_cartpole_onestep_paper.py`; supporting result file `data/cartpole/cartpole_onestep_summary.json`.
• Hopper-Hop mechanism-oriented SD results: Core SD model `code/src/world_model/dmcontrol_hopper_sd_model.py`; TD-MPC2 overlay files under `code/tdmpc2_sd_patch/`; supporting logs and aggregate files under `data/hopper/mechanism_oriented_200k/`, including `hopper_mechanism_oriented_summary.csv`, `hopper_mechanism_oriented_final_200k.json`, and `hopper_public_baseline_comparison.csv`.

The supplemental package also provides `requirements.txt` and `README.md`, which specify the environment, verification commands, result locations, and the TD-MPC2 overlay procedure used for Hopper-Hop.
```

## 3. Hopper-Hop Results and Text Revisions

### 3.1 修订指引

Hopper-Hop 相关数字应统一改为 `Attachments/data/hopper/mechanism_oriented_200k/` 中的 mechanism-oriented `SD` 三种子结果。论文主叙述应保留高效规划的核心结论：mechanism-oriented `SD` 在 50k interactions 下的三种子均值已超过 TD-MPC2 和 DreamerV3 的 100k 官方均值。正文应采取最小化修订原则，优先替换旧数字，避免重写已经成立的论证结构。

建议全文将旧的 50k 数值替换为最新附件包结果，并保留“50k 超过 100k 公开基线均值”的高效规划叙述。Hopper-Hop 相关表格、正文结果句和附录说明应统一使用以下结果口径：

```text
Mechanism-oriented SD, 50k: 18.13 +/- 12.82
Mechanism-oriented SD, 100k: 61.14 +/- 11.67
Mechanism-oriented SD, 200k: 206.90 +/- 42.24
Mechanism-oriented SD, 200k per-seed scores: 185.62, 265.87, 169.19
TD-MPC2 official, 100k: 13.63 +/- 10.41
TD-MPC2 official, 200k: 166.63 +/- 34.89
DreamerV3 official, 100k: 9.67 +/- 13.60
DreamerV3 official, 200k: 51.17 +/- 50.40
```

### 3.2 Table 1

建议 Table 1 保持原结构，只替换 Hopper-Hop 行中的最新数字，并尽可能补齐当前可核对结果能够支撑的 Worst-seed Return。为避免 `standard error` 与 `standard deviation` 混用，Table 1 建议统一报告 mean +/- population standard deviation across seeds；没有 seed-level 官方表格口径的外部结果保留单点值和 `-`。若同步补齐全表最后一列，建议如下：

```text
Environment                      Method                         Interaction Budget   Mean Return          Worst-seed Return
Beer Game                        SD-MCTS (Ours)                 1k steps             -710 +/- 45          -756.8
(Long-delay Discrete Planning)   DreamerV3                      10k steps            -713 +/- 157         -876.6
                                 PPO                            10k steps            -6200 +/- 3230      -10412.0

Cartpole-Swingup                 SD-MCTS (Ours)                 35k + 5k steps       802.9 +/- 21.3      772.8
(Sensitive Cont. Control)        DreamerV3                      500k steps           850.0                -

Hopper-Hop                       Mechanism-oriented SD (Ours)   50k steps            18.13 +/- 12.82      2.28
(Complex Contact & Sparse Reward) TD-MPC2                       100k steps           13.63 +/- 10.41      0.70
                                 DreamerV3                      100k steps            9.67 +/- 13.60      0.00
```

### 3.3 Section 6.2.2

建议 6.2.2 严格作为 Table 1 的解释，只对原有结果句做数字替换，保留原小标题和论证结构。将原句替换为：

```text
Performance Inversion under Halved Budgets: As shown in Table 1, the structurally augmented framework achieves a mean return of 18.13 +/- 12.82 with only a 50k step budget, outperforming the official TD-MPC2 (13.63 +/- 10.41) and DreamerV3 (9.67 +/- 13.60) evaluated at 100k steps.
```

### 3.4 C.2 Hopper Evaluation Settings

建议 C.2 中 Hopper Evaluation Settings 仅修订 “single-seed extensions” 相关措辞。将原句：

```text
Specifically, the structured Hopper results for 50k steps, 100k steps, 200k steps, and single-seed extensions follow the same training and evaluation protocols, only varying the real-world environment interaction budget.
```

替换为：

```text
Specifically, the structured Hopper results for 50k steps, 100k steps, and 200k steps follow the same three-seed training and evaluation protocols, only varying the real-world environment interaction budget.
```

### 3.5 Appendix D.4 and Table 6

建议将 D.4 中 “Implementation Details and Table 6” 段落替换为：

```text
Implementation Details and Table 6 This experiment is not a direct replication of the native SD-MCTS design. Instead, it evaluates whether a mechanism-oriented SD world model can be integrated into the TD-MPC2 online training and planning loop. The SD extension keeps the TD-MPC2 solver interface, uses the `sd_hopper` world-model adapter, and applies online SD supervision to the structured transition core. Table 6 reports the resulting three-seed performance from 50k to 200k interactions, together with the public TD-MPC2 and DreamerV3 reference results at 100k and 200k.
```

建议将 Table 6 替换为：

```text
Table 6: Comparison of results on public benchmarks for Hopper-Hop

Method                                  50k steps          100k steps          200k steps
Mechanism-oriented SD (Ours, 3 seeds)   18.13 +/- 12.82    61.14 +/- 11.67     206.90 +/- 42.24
TD-MPC2 (Official, 3 seeds) [6]         -                  13.63 +/- 10.41     166.63 +/- 34.89
DreamerV3 (Official, 3 seeds) [4]       -                   9.67 +/- 13.60      51.17 +/- 50.40
```

建议将 D.4 中 “Experimental Results and Per-seed Breakdown” 开头替换为：

```text
Experimental Results and Per-seed Breakdown The mechanism-oriented SD extension improves steadily across the 0--200k interaction range. At 50k interactions, the three-seed mean return is 18.13 +/- 12.82, already exceeding the official TD-MPC2 and DreamerV3 100k averages. At 100k interactions, the mean return rises to 61.14 +/- 11.67, further enlarging the early-budget gap. At 200k interactions, the method reaches 206.90 +/- 42.24, with per-seed scores of 185.62, 265.87, and 169.19. This 200k result also exceeds the budget-matched TD-MPC2 official average of 166.63 +/- 34.89 and the DreamerV3 official average of 51.17 +/- 50.40.
```

## 4. Checklist 16: Declaration of LLM Usage

### 4.1 修订指引

根据 NeurIPS 2026 Main Track Handbook 中 “Author Use of Agents and Large Language Models (LLMs)” 的说明，只有当 agents 或 LLMs 是方法中的 important、original 或 non-standard component 时，才需要在 experimental setup 或等价位置描述；spell checkers、grammar suggestions、aid for editing purposes 和 basic code assistance 不需要在论文中记录。当前 AI 辅助使用属于写作、整理、调试、脚本辅助和参数配置探索等非核心研究辅助，不构成 SD-MCTS 方法、实验算法、环境模型、基线或评估组件。因此 Checklist 16 建议保留 `Answer: [N/A]`。

不过，原 justification 中 “strictly limited to language editing, translation, and formatting” 过窄，建议改为覆盖 non-core writing, editing, formatting, and basic code-assistance uses，同时明确这些辅助未影响核心方法、科学严谨性或原创性。

### 4.2 建议替换文本

建议将 Checklist 16 的 answer 和 justification 替换为：

```text
Answer: [N/A]
Justification: Large Language Models (LLMs) and agentic AI tools were not used as important, original, or non-standard components of the core methodology (SD-MCTS), environment modeling, baseline algorithms, experiment execution, or quantitative evaluation. Any AI-assisted use was limited to non-core manuscript preparation and research-engineering assistance, such as language editing, translation, formatting, script organization, debugging assistance, log inspection, supplemental-material organization, consistency checks over author-provided code and result files, and limited parameter-configuration exploration. These uses did not determine the core algorithmic design, experimental protocol, reported numerical results, scientific claims, or originality of the research.
```

## 5. Remaining Consistency Fix

### 5.1 统计口径

为避免评审质疑为什么不同表格使用不同统计口径，建议将论文中主要结果表的 `+/-` 统一为跨种子的标准差（standard deviation, SD），包括 Table 1、Table 2、Table 4、Table 5 和 Table 6。不要在正文或 checklist 中混写 “Standard Deviation/Error”，也不要把 Table 1 或 Table 4 误说明为 standard error。建议将 C.2 中 Beer Game 段落的相关句子替换为：

```text
Table 1 and Table 4 report mean returns with across-seed standard deviations where seed-level results are available.
```

同时建议将 Table 4 caption 中的 “standard error” 改为 “standard deviation”，并将 Table 4 数值替换为：

```text
Table 4: Sample efficiency comparison (average cumulative return and standard deviation across 3 seeds)

Sample Budget   SD-MCTS (Ours)   DreamerV3       PPO
1,000 steps     -710 +/- 45      -6785 +/- 3852  -4335 +/- 1166
3,000 steps     -731 +/- 57      -1373 +/- 335   -4472 +/- 1632
10,000 steps    -713 +/- 32      -713 +/- 157    -6200 +/- 3230
```

Checklist 7 中 “Standard Deviation/Error” 的写法也过于含混，建议统一写为 standard deviation。建议将 justification 替换为：

```text
Justification: Yes. Quantitative tables specify the dispersion statistic used for each result. The main quantitative tables report mean values with across-seed standard deviations where seed-level results are available, and extended tables in Appendix D provide per-seed values where applicable.
```

另需注意 Appendix D.3 中 Cartpole 的 per-seed 表述（如 “Seed 0 achieved 772.8 +/- 33.5”）不是跨种子标准差，而是该 seed 下 5 个 evaluation episodes 的 episode-level standard deviation。建议在该句中明确写出 `episode-level standard deviation over 5 evaluation episodes`，避免与主表中的 across-seed standard deviation 混淆。
