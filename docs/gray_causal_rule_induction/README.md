# Gray Causal Rule Induction

本目录记录面向 Hopper 规则生成机制的灰因果图研究线。该研究线与 `sd_retrospective_rule_induction` 并行，不覆盖既有 SD 轻盈介入 TD-MPC2 实验。

当前阶段的研究对象不是替换 SD 主模型，而是在 SD、replay 与 TVA 证据之后建立规则证书层。该层将规则理解为上下文、动作族与短视界结构后果之间的灰边，并输出规则状态与轻量 prior 强度建议。

代码目录为 `experiments/gray_causal_rule_induction/`，默认输出目录为 `results/gray_causal_rule_induction/`。

截至 2026-06-04，`cycle_079` 灰过滤规则库已在 400k checkpoint 上完成 3 episode、三种子 eval-only smoke。保守过滤变体相对原始 active 规则库显示正向结果：`drop_reject_active/certified_active` 均值为 `363.974`，高于原始 active 的 `355.452` 和同 TVA 下 rule-off 的 `359.080`。`cycle080 -> cycle081` smoke 中，`strict_active` 仍保持轻微正向信号，而 `drop_reject_active` 未复现上一轮收益。因此，当前闭环验证优先采用 `strict_active` 灰证书准入。

新闭环训练入口为：

```text
experiments/gray_causal_rule_induction/run_hopper_v7_multiseed_gray_rules.py
experiments/gray_causal_rule_induction/launch_hopper_300k_gray_strict.sh
```

该入口复用原三种子 Hopper 闭环训练口径，但从 `cycle_002` 起将上一轮 confidence 规则库转换为灰证书 `strict_active` 变体后再接入在线规则 prior。当前 300k 运行输出目录为：

```text
results/gray_causal_rule_induction/closed_loop/hopper_v8formal_300k_graycausal_strict_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260604_r1/
```
