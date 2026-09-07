# 严格入口标签与数学审计对比

## 数据口径

在不覆盖原始入口数据的前提下，新增两版严格入口数据：

```text
entry_candidate_segments_strict_no_pre_entry.json
entry_candidate_segments_strict_entry_cycle_only.json
```

构造规则为：

```text
no_pre_entry:
  保留入口当周期和入口后一周期

entry_cycle_only:
  仅保留入口当周期

共同处理:
  移除 success/failure 角色冲突窗口
  同一窗口跨阈值重复出现时只保留一个最高阈值实例
```

样本数为：

```text
raw:              568 windows
no_pre_entry:     269 windows
entry_cycle_only: 135 windows
```

## 标签信号变化

严格标签明显增强了真实后果可分性：

| 口径 | Success | Failure | Future reward AUC | Future rootx AUC |
| --- | ---: | ---: | ---: | ---: |
| raw | 262 | 306 | 0.663 | 0.753 |
| no_pre_entry | 117 | 152 | 0.714 | 0.799 |
| entry_cycle_only | 53 | 82 | 0.712 | 0.800 |

若在原始数据上做更严格的入口当周期与角色一致窗口消融，future reward AUC 可到 `0.755`，future rootx AUC 可到 `0.812`。这说明入口标签本身有真实信号，原始口径主要被前周期扩展、跨阈值重复和角色冲突稀释。

## TVA 效果空间 SVM

严格标签没有恢复 SVM 可分性：

| 口径 | 模型 | Global AUC | Low-entry AUC | High-entry AUC | Stable contexts |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | logistic_l2 | 0.322 | 0.284 | 0.332 | 0 |
| raw | linear_svm | 0.320 | 0.283 | 0.305 | 0 |
| raw | rbf_svm | 0.347 | 0.376 | 0.304 | 0 |
| no_pre_entry | logistic_l2 | 0.331 | 0.333 | 0.380 | 0 |
| no_pre_entry | linear_svm | 0.323 | 0.315 | 0.386 | 0 |
| no_pre_entry | rbf_svm | 0.359 | 0.376 | 0.303 | 0 |
| entry_cycle_only | logistic_l2 | 0.235 | 0.210 | 0.320 | 0 |
| entry_cycle_only | linear_svm | 0.241 | 0.232 | 0.289 | 0 |
| entry_cycle_only | rbf_svm | 0.397 | 0.424 | 0.286 | 0 |

因此，当前 `z_phi(s,a)` 特征和 success/failure 动作标签之间仍不存在稳定的跨周期分类边界。RBF SVM 也没有提供可用的非线性上限。

## 局部动作效果优势空间

严格标签对局部优势空间有一定改善：

| 口径 | 方法 | Tier | Transfer AUC | AUC>=0.60 | Positive margin | Mean cosine | Stable |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| raw | ridge_lda | low_entry | 0.482 | 0.083 | 0.500 | -0.129 | no |
| raw | ridge_lda | high_entry | 0.576 | 0.500 | 1.000 | 0.364 | no |
| no_pre_entry | ridge_lda | low_entry | 0.536 | 0.333 | 0.750 | -0.008 | no |
| no_pre_entry | ridge_lda | high_entry | 0.601 | 0.500 | 1.000 | 0.093 | no |
| entry_cycle_only | ridge_lda | low_entry | 0.499 | 0.250 | 0.500 | -0.022 | no |
| entry_cycle_only | ridge_lda | high_entry | 0.613 | 0.500 | 1.000 | -0.029 | no |

这说明严格标签确实降低了一部分噪声，尤其 `no_pre_entry` 对 low/high entry 都有改善；但方向仍未达到 stable tier 标准。

## 结论

修订入口标签体系可以改善基础后果信号，也能小幅改善局部效果空间方向；但它不能单独解决“数学方法找不出规则”的问题。当前剩余瓶颈主要在上下文定义：同一 `context_key` 下仍混合多个局部控制机制，导致动作族方向跨周期不稳定。

下一步不应继续增加模型复杂度。应在严格标签基础上，进一步修订上下文定义，使其区分当前混在一起的局部机制；随后只重跑轻量的局部优势或线性筛选审计。
