# Strict Context Variant Summary

## Scope

本审计在严格入口标签基础上比较多种上下文定义，目标是判断此前数学方法无法提取规则是否主要由入口标签噪声造成，或仍受上下文混合限制。审计不训练新模型、不写入规则库、不启动在线训练。

输入数据：

```text
entry_candidate_segments_strict_no_pre_entry.json
entry_candidate_segments_strict_entry_cycle_only.json
```

新增审计：

```text
entry_strict_context_variant_audit_no_pre_entry.json
entry_strict_context_variant_audit_entry_cycle_only.json
entry_counterfactual_audit_strict_no_pre_entry.json
```

## Main Result

`no_pre_entry` 口径保留 `269` 条窗口，其中 `entry_success=117`、`matched_failure=152`。在该口径下，原始 anchor 上下文、refined phase 上下文、加入 anchor type 的上下文以及窗口主导 refined phase 上下文均产生同一条候选：

```text
low_entry::omega_pos|vx_high|support_none|other|rotation_risk
  -> prefer waist_neg
```

最严格的 `anchor_refined_type` 版本为：

```text
low_entry::omega_pos|vx_high|support_none|other|rotation_risk|first_sustained_forward
  -> prefer waist_neg
```

该候选包含 `11` 条窗口，其中成功 `6` 条、失败 `5` 条。成功侧 `waist_neg` 命中率为 `1.000`，失败侧为 `0.000`，动作族 AUC 为 `1.000`，后续奖励差为 `+6.818`，后续 root-x 差为 `+0.364`，成功窗口横跨 `cycle007/017/018`。

`entry_cycle_only` 口径只保留 `135` 条窗口，没有产生 stable-like 候选。该结果说明：只保留入口当周期会使样本过少；保留入口后一周期更适合当前规则审计，但仍应排除前周期和角色冲突窗口。

## Counterfactual Check

对 `no_pre_entry` 严格数据重跑 SD/TVA 入口反事实审计后，上述候选的模型支持并不充分：

```text
entry_success rows: 6
  waist_neg SD top-3:  2/6
  waist_neg TVA top-3: 4/6
  waist_neg SD best:   2/6
  waist_neg TVA best:  2/6

matched_failure rows: 5
  waist_neg SD top-3:  3/5
  waist_neg TVA top-3: 0/5
  waist_neg SD best:   0/5
  waist_neg TVA best:  0/5
```

因此，该候选不是可直接上线规则。它更适合作为下一步局部诊断对象：统计后果支持较强，TVA 对失败侧的排斥较清楚，但 SD 短视界结构态分数仍会在部分失败状态中把 `waist_neg` 排到较前位置。

## Model Maturity Caveat

上述反事实检查不能被解释为对候选规则的直接否定，因为早期 SD 和 TVA 均未经过充分训练。按周期分层后，候选上下文呈现明显成熟度差异：

```text
entry_success early <= cycle008:
  waist_neg SD top-3  = 0/2
  waist_neg TVA top-3 = 0/2

entry_success later >= cycle017:
  waist_neg SD top-3  = 2/4
  waist_neg TVA top-3 = 4/4

cycle018 entry_success:
  waist_neg SD best  = 2/2
  waist_neg TVA best = 2/2
```

这说明早期 SD/TVA 的“不支持”很可能混入模型未成熟因素。后续规则审计应将模型成熟度作为证据权重：早期 SD/TVA 反事实更适合作为弱证据或诊断信号，不能与后期模型成熟后的反事实同等对待。

## Conclusion

严格入口标签修订解决了部分噪声问题；上下文变体审计也说明，经过更合适的上下文划分后，确实可以找回少量规则形态候选。但这并没有完全解决“数学方法找不出规则”的问题，因为候选数量少，且最强候选在 SD/TVA 反事实层面的支持仍不一致。

下一步不应继续扩大分类器复杂度。更合适的推进方式是保留 `no_pre_entry` 严格标签作为默认入口审计口径，并围绕少量候选上下文检查三件事：同一上下文内失败侧为何被 SD 排序混入、TVA 的拒绝信号是否更可靠、以及该候选是否能在模型成熟后和真实后果窗口中通过跨周期验证。
