# VNet 训练退化诊断与修复记录 (2026-03-26)

## 问题现象
- T4 配置 MCTS return 仅 283（baseline T0 为 798）
- VNet val_loss 0.78（baseline 0.60）

## 根因
`generate_imagination_data` 在某次未版本控制的修改中退化，涉及三处变更：
1. **heuristic 符号翻转**：balance 控制器从 `-(3*sin + thd + ...)` 改为 `+4*sin + 1.5*thd - ...`
2. **移除了 30% uniform random 锚点**：改为纯 real_states 的 4 类分层采样
3. **添加了 branch augmentation**（30% 轨迹加入随机扰动）

这导致 imagination 数据从 [0, 49.6] 压缩到 [0, 28.0]，VNet 训练质量下降。

## 恢复方法
从 `.claude/projects/` 会话记录 `b5499ce0` 中找回原始代码。恢复配置：
- OLD heuristic：`a = -(3.0*sin_th + 1.0*theta_dot + 0.4*x + 0.3*x_dot)`
- 70/30 anchor split：70% real states（50% 加 0.1 扰动）+ 30% uniform random
- 无 branching，subsample 每 4 步

## 验证结果

### T0 配置复现
| 指标 | Baseline 原始 | 恢复后 |
|------|:---:|:---:|
| R1 max return | 49.6 | 49.6 |
| VNet val_loss | 0.60 | 0.67 |
| 样本数 | 30000 | 30000 |

### T4 seed 0（SI 改进 + baseline imagination）
| 指标 | T4 退化版 | T4 修复后 | Baseline T0 |
|------|:---:|:---:|:---:|
| mp 误差 | 2.7% | 2.7% | 47% |
| R1 max | 44.8 | 49.8 | 49.6 |
| VNet val | 0.78 | 0.73 | 0.60 |
| MCTS return | 283 | **524** | 798 |

### T4 seed 42（泛化验证）
| 指标 | Baseline T0 | T4 修复后 |
|------|:---:|:---:|
| mp 误差 | — | 0.9% |
| VNet val | 0.79 | 0.55 |
| MCTS return | 626 | **460** |

两个 seed 均稳定恢复，修复具备泛化能力。T4 相对 baseline 仍有 ~30% 差距，
主要来源：mixed policy 高能量锚点使 imagination 数据分布更复杂。

## 失败的尝试
- θ wrap in model_forward_k → VNet loss 暴涨（丢失轨迹时间编码）
- 合成 near-upright 锚点 → 回报范围过宽（[0,70]），VNet 无法拟合
- h=120 → 同上
- R2 bootstrap → 数据稀释，反而退化

## 关键教训
1. heuristic 的"改进"反而有害：平衡控制器太完美（return→70）会制造 VNet 难以拟合的双峰分布
2. 30% uniform random 锚点是 VNet 训练的关键——确保 near-upright 区域有覆盖
3. 代码版本控制极其重要：scripts/ 目录未纳入 git 导致此次回溯耗时数小时
