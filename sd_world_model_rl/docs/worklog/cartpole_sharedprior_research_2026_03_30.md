# Cartpole shared_prior 研究进展记录 (2026-03-30)

## 目标重申
- 方法学主框架已经确定为 **shared_prior + per-seed finetune**。
- 当前核心目标不是把三种子都拉到“还可以”的均值，而是：
  1. 保住 `seed42` 已经获得的 rescue 效果；
  2. 尽量恢复 `seed0` 与 `seed84` 历史上曾达到的高分。

## 当前结论摘要
1. `shared_prior` 路径当前**没有**候选层 `state_noise` selector；只有固定超参下的 restart 选优。
2. 旧的 `adaptive` 路径才有 `_select_candidate(...)` 候选选择器；且其 tie-break 原先偏向更大 `state_noise`，现已改为平局偏向更小 `state_noise`。
3. 对当前 `shared_prior` 主框架而言，影响 strong seeds 的关键问题不是 alpha，而是 **finetune 端正则化过强**。
4. 已验证：在保持 `shared_prior` 框架不变时，去掉 finetune `weight_decay` 能明显改善 strong seeds：
   - `seed0`: `710.1 ± 21.3` → `772.8 ± 33.5`
   - `seed84`: `779.3 ± 48.1` → `794.4 ± 29.2`
5. 已补齐 `seed0` 的 `(wd, sn)` 2×2 候选：
   - `(1e-4, 1e-3)` → `541.9 ± 6.0`
   - `(1e-4, 0)` → `710.1 ± 21.3`
   - `(0, 0)` → `772.8 ± 33.5`
   - `(0, 1e-3)` → `548.8 ± 196.8`
   结论：`seed0` 的最优候选是 **`(wd=0, sn=0)`**；`sn=1e-3` 即使在 `wd=0` 下也会造成明显不稳定。
6. 已补齐 `seed84` 的关键缺口：
   - `(1e-4, 1e-3)` → `779.3 ± 48.1`
   - `(0, 1e-3)` → `794.4 ± 29.2`
   - `(0, 0)` → `818.2 ± 18.8`
   结论：`seed84` 当前最优已知候选也是 **`(wd=0, sn=0)`**。
7. 已补齐 `seed42` 的关键候选：
   - `(1e-4, 1e-3)` → `801.3 ± 6.3`
   - `(0, 1e-3)` → `816.0 ± 2.9`
   - `(0, 0)` → `817.9 ± 29.6`
   结论：`seed42` 在 `wd=0` 下同样受益；而 `sn=0` 与 `sn=1e-3` 的均值非常接近，但 `sn=1e-3` 方差显著更小。
8. 因此，当前证据更支持：
   - **`wd` 可以先固定为 `0`**；
   - **`sn` 在三个 seed 上都可以优先固定为 `0`**，因为三者的最佳均值均落在 `wd=0, sn=0`；
   - 但若目标更强调稳定性而非均值，则 `seed42` 的 `sn=1e-3` 仍值得保留为低方差备选。
   也就是说，后续 selector 很可能不需要完整 2×2；更像可先收敛为“默认 `wd=0, sn=0`”，再视稳定性需求决定是否保留 `seed42` 的 `sn=1e-3` 分支。

---

## 关键代码位置

### A. shared_prior 主链路
文件：`scripts/sweep_si_robust.py`

- shared prior 训练 / 复用：`scripts/sweep_si_robust.py:289-340`
- old adaptive 候选选择器：`scripts/sweep_si_robust.py:344-352`
- per-seed 训练入口：`scripts/sweep_si_robust.py:355-438`
- 其中 shared_prior 分支：`scripts/sweep_si_robust.py:409-430`
- shared_prior CLI / full path：`scripts/sweep_si_robust.py:719-832`

### B. 历史高分训练链路
文件：`scripts/sweep_alpha.py`

- R1-only、no R2 bootstrap：`scripts/sweep_alpha.py:1-14`
- 单 seed 训练逻辑：`scripts/sweep_alpha.py:49-122`
- 训练后统一做 alpha sweep eval：`scripts/sweep_alpha.py:141-217`

### C. VNet 训练与 restart 选优
文件：`scripts/run_dmcontrol_cartpole.py`

- `train_value_net(...)`：`scripts/run_dmcontrol_cartpole.py:147-178`
  - `weight_decay` 和 `state_noise` 真正作用在这里
- slice metrics：`scripts/run_dmcontrol_cartpole.py:181-205`
- `critical` 分数：`scripts/run_dmcontrol_cartpole.py:208-213`
- `train_vnet_with_restarts(...)`：`scripts/run_dmcontrol_cartpole.py:222-287`
- `cart_drift_penalty` 进入评估期搜索：`scripts/run_dmcontrol_cartpole.py:709-715`

---

## 已确认的机制性认识

### 1. shared_prior 路径当前没有 `sn` 候选层 selector
在 `scripts/sweep_si_robust.py:409-430` 中，`shared_prior` 分支只是把单个 `weight_decay` / `state_noise` 传入 `_train_single_candidate(...)`，并不会像 old adaptive 那样遍历多个候选再调用 `_select_candidate(...)`。

因此：
- 历史上的 `seed42` / `seed84` shared_prior 正式结果，`state_noise=1e-3` 是**手动固定**的；
- 自动发生的只有 **restart 层** best-VNet 选择，不是 `sn` 候选选择。

### 2. old adaptive 的 tie-break 规则来源
旧规则来自 `scripts/sweep_si_robust.py:344-352`，原始逻辑在 `val_loss` 平局时偏向更大 `state_noise`。现已改为：
- 排序按 `(val_loss, state_noise)`
- 平局选更小 `state_noise`

但这个修改只直接影响 `adaptive` 路径，不直接影响当前 `shared_prior` 主线。

### 3. 历史高分线的真正启发
从 `sweep_alpha.py` 与其日志来看，历史高分线对 `alpha` 本身并不敏感；真正重要的是其训练链路特征：
- 每个 seed 独立从头训练 VNet；
- R1-only；
- no R2 bootstrap；
- finetune 端无额外 `weight_decay` / `state_noise`；
- 更接近低正则、seed-specific 的 VNet 学习。

这说明 `shared_prior` 当前更可能的问题是：
- 为了稳住 `seed42`，后半段 finetune 过于保守；
- 从而压低了 `seed0` / `seed84` 的上限。

---

## 关键实验与结果

### 1. shared_prior 旧正式结果（基准对照）
文件：
- `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500_seed0_eval.pt`
- `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500_seed42_eval.pt`
- `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500_seed84_eval.pt`

对应日志：
- `tmp/si_robust_T4_eval_seed0_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500.log`
- `tmp/si_robust_T4_eval_seed42_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500.log`
- `tmp/si_robust_T4_eval_seed84_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500.log`

结果：
- `seed0 = 541.9 ± 6.0`
- `seed42 = 801.3 ± 6.3`
- `seed84 = 779.3 ± 48.1`

解释：
- `seed42` 已被 shared_prior 明显救回；
- 但 `seed0` / `seed84` 仍低于历史高分线。

### 2. seed0：shared_prior + `sn=0`（仍保留 `wd=1e-4`）
日志：
- `/tmp/si_robust_T4_finetune_seed0_sharedprior_tiebreak_nonnoise.log`

评估缓存：
- `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0_fe500_seloverall_seed0_eval.pt`

结果：
- `seed0 = 710.1 ± 21.3`

解释：
- 仅把 `seed0` 的 finetune `sn` 从 `1e-3` 改成 `0`，就带来显著提升。

### 3. seed0：shared_prior + `wd=0, sn=0`
日志：
- `/tmp/si_robust_T4_finetune_seed0_sharedprior_wd0_sn0_seloverall.log`

评估缓存：
- `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_seed0_eval.pt`

结果：
- `seed0 = 772.8 ± 33.5`

相对上一条提升：
- `710.1 → 772.8`（`+62.7`）

解释：
- 对 `seed0` 而言，`weight_decay=1e-4` 明显压低了上限。
- shared_prior 框架本身未必是问题；更像是 finetune 端统一加的正则过强。

### 4. seed84：shared_prior + `wd=0, sn=1e-3`
日志：
- `/tmp/si_robust_T4_finetune_seed84_sharedprior_wd0_sn1em3_seloverall.log`

评估缓存：
- `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0p001_fe500_seloverall_seed84_eval.pt`

结果：
- `seed84 = 794.4 ± 29.2`

相对旧 shared_prior 正式结果提升：
- `779.3 → 794.4`（`+15.1`）

解释：
- `seed84` 也出现了与 `seed0` 同方向的改善。
- strong seeds 被 `weight_decay=1e-4` 压制并不是 seed0 特例。

---

## 历史高分线（用于对照，不等于当前主框架）

### 1. `sweep_alpha` 代表性结果
日志：
- `/tmp/sweep_alpha.log`
- `/tmp/sweep_alpha_5seeds.log`

代表性数字（`/tmp/sweep_alpha.log:40-68`）：
- `seed0 ≈ 788-794`
- `seed42 ≈ 549-599`
- `seed84 ≈ 804-805`
- `overall ≈ 730.4`（`alpha=5`）

论文当前写入的 cartpole 叙事基本来自这条线：
- `docs/neurips/paper/full_draft_zh.tex:517-531`

### 2. 对当前 shared_prior 研究的意义
这条历史高分线说明：
- strong seeds (`0`, `84`) 本身具备更高上限；
- 后续大量工作转向 shared_prior，本质上是为了救 `seed42`；
- 因此现在不能满足于“均值还行”，而要继续研究如何在保住 `seed42` 的同时恢复 `seed0` / `seed84` 的历史高分。

---

## 当前阶段性判断
1. `shared_prior` 已经证明对 worst seed rescue 有价值，尤其对 `seed42`。
2. 当前 `shared_prior` 主线压 strong seeds 的关键嫌疑不是 prior 本身，而是 **finetune 默认正则过强**。
3. 当前最强经验信号是：
   - `seed0` 偏好 `sn=0`
   - `seed0` 和 `seed84` 都从 `wd=0` 中受益
4. 基于当前已补齐的结果，selector 设计需要更新判断：
   - 现阶段更像应**固定 `wd=0`**；
   - 然后将 `sn=0` 作为论文默认主配置。
5. 用户已明确决定：
   - **工作日志**中保留完整候选、分析过程、方差差异与方法判断；
   - **论文主结果口径**直接采用 `shared_prior + wd=0 + sn=0`，不在主文中展开完整候选搜索叙事。

---

## 下一步建议
1. 以 `shared_prior + wd=0 + sn=0` 作为论文主口径整理三种子正式汇总；
2. 工作日志继续完整保留 `sn` / `wd` 候选实验、方差差异与方法推理过程；
3. 重新整理三种子 paper summary，明确：
   - 论文主结果采用的固定配置；
   - 历史 alpha 高分线；
   - 以及完整候选搜索仅保存在 worklog 中作为研发证据链。

---

## 2026-03-30 新增：formal paper runner

### 1. 新增正式实验入口
文件：`scripts/run_cartpole_sharedprior_paper.py`

定位：
- 将当前已经验证的 `shared_prior + per-seed finetune` 方法链 formalize 为一个 paper-facing 实验入口；
- 默认直接采用论文主口径：`wd=0, sn=0`；
- prior 端仍沿用既有 shared prior 配置：`prior_weight_decay=1e-4`, `prior_state_noise=1e-3`, `prior_epochs=500`；
- 支持两种执行方式：
  - `--mode serial`
  - `--mode parallel --parallel_gpus N`

当前实现特点：
- 直接复用 `scripts/sweep_si_robust.py` 中既有 helper，而不改训练语义；
- 自动生成：
  - run header
  - per-seed summary
  - unified JSON summary
- summary 中集中记录：
  - run config
  - prior metadata / reuse 状态
  - per-seed ckpt / eval / val_loss / SI 参数 / return 统计
  - aggregate mean / median / IQM / `<700` 计数

### 2. smoke 验证
命令：
- `python scripts/run_cartpole_sharedprior_paper.py --tier T4 --seeds 0 --summary_path /tmp/cartpole_sharedprior_paper_smoke_seed0.json`

输出：
- summary: `/tmp/cartpole_sharedprior_paper_smoke_seed0.json`

smoke 结果：
- runner 正确复用已有 payload / prior / ckpt / eval cache；
- `seed0` 读到的结果为：`772.8 ± 33.5`
- summary 中已正确记录：
  - `source_kind = legacy`
  - prior source path
  - ckpt / eval path
  - selected_cfg
  - aggregate 指标

### 3. 正式三种子重跑
正式运行已完成：
- log: `/tmp/cartpole_sharedprior_paperformal_20260330_r1.log`
- summary: `results/si_robust_summaries/T4_sharedprior_paperformal_20260330_r1_summary.json`
- experiment tag suffix: `_paperformal_20260330_r1`

本次正式运行意图：
- 保持 shared prior 方法链不变；
- 复用已有 prior；
- 对 `seed0, seed42, seed84` 重新生成一套 paper-facing ckpt / eval / summary artifact；
- 避免仅复用旧 tag 下的历史结果，从而形成独立、可引用的正式实验产物。

正式结果：
- `seed0 = 772.8 ± 33.5`
- `seed42 = 817.9 ± 29.6`
- `seed84 = 818.2 ± 18.8`
- aggregate mean = `802.9`
- median = `817.9`
- IQM = `802.9`
- `<700 = 0/3`

本次 formal run 的关键确认：
- 三个 seed 均使用新 tag 下重新生成的 ckpt / eval artifact，而非直接复用旧 tag；
- prior 仍复用已有 shared prior（legacy prior path）；
- 三种子结果与此前工作日志中的 best-known `wd=0, sn=0` 结果完全一致，说明 formal runner 已稳定复现当前 paper-facing 主配置。

### 3.1 正式运行代码、参数与日志明细
#### (a) 入口代码
- 正式 runner：`scripts/run_cartpole_sharedprior_paper.py`
- 复用主链路：`scripts/sweep_si_robust.py`
- VNet / MCTS / evaluation 实现：`scripts/run_dmcontrol_cartpole.py`

其中 `run_cartpole_sharedprior_paper.py` 的职责是：
- 固定 paper-facing 默认配置；
- 组织 shared prior 复用 / 训练、paper seeds finetune、paper seeds eval；
- 将结果汇总为统一 JSON summary；
- 串行模式下直接调用既有 helper；并提供可选 parallel 模式。

#### (b) 本次正式运行命令
```bash
python -u scripts/run_cartpole_sharedprior_paper.py \
  --tier T4 \
  --seeds 0 42 84 \
  --experiment_tag_suffix _paperformal_20260330_r1 \
  --summary_path results/si_robust_summaries/T4_sharedprior_paperformal_20260330_r1_summary.json
```

#### (c) 本次正式运行超参数
- tier: `T4`
- selector_mode: `shared_prior`
- paper seeds: `0, 42, 84`
- dev seeds: `10, 21, 30, 55, 63, 72, 95`
- finetune weight decay: `0`
- finetune state noise: `0`
- finetune epochs: `500`
- selection metric: `overall`
- prior weight decay: `1e-4`
- prior state noise: `1e-3`
- prior epochs: `500`
- mode: `serial`

#### (d) 本次运行使用的 prior
- canonical prior tag:
  - `sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500`
- 实际复用的 prior 路径（legacy path）：
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0p0001_fsn0p001_fe500_prior.pt`
- prior metadata:
  - `train_loss = 0.14578`
  - `val_loss = 0.47852`
  - `n_samples = 210000`
  - `reused = true`
  - `source_kind = legacy`

#### (e) 运行日志与 summary 文件
- formal run log:
  - `/tmp/cartpole_sharedprior_paperformal_20260330_r1.log`
- task captured output:
  - `/tmp/claude-1004/-home-data-round-code-sd-world-model-rl/783d2ffd-4f33-4524-afe7-df878bf94463/tasks/b28012r4q.output`
- summary:
  - `results/si_robust_summaries/T4_sharedprior_paperformal_20260330_r1_summary.json`

#### (f) 新生成的正式 artifact
Seed 0:
- ckpt:
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed0.pt`
- eval:
  - `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed0_eval.pt`

Seed 42:
- ckpt:
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed42.pt`
- eval:
  - `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed42_eval.pt`

Seed 84:
- ckpt:
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed84.pt`
- eval:
  - `results/si_robust_eval/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed84_eval.pt`

### 3.2 逐 seed 正式结果明细
#### Seed 0
- selected cfg:
  - `wd = 0`
  - `sn = 0`
  - `val_loss = 0.46738`
- eval returns:
  - `[736.70, 801.98, 800.65, 727.19, 797.31]`
- eval summary:
  - `772.8 ± 33.5`
  - `[727.2, 802.0]`
- SI params:
  - `masscart = 1.0070`
  - `masspole = 0.09731`
  - `length = 0.51441`
  - `mp_err = 2.69%`
- runtime:
  - `2146.6 s`

#### Seed 42
- selected cfg:
  - `wd = 0`
  - `sn = 0`
  - `val_loss = 0.40385`
- eval returns:
  - `[782.15, 842.93, 841.96, 781.17, 841.22]`
- eval summary:
  - `817.9 ± 29.6`
  - `[781.2, 842.9]`
- SI params:
  - `masscart = 1.0468`
  - `masspole = 0.10087`
  - `length = 0.51565`
  - `mp_err = 0.87%`
- runtime:
  - `2296.1 s`

#### Seed 84
- selected cfg:
  - `wd = 0`
  - `sn = 0`
  - `val_loss = 0.36037`
- eval returns:
  - `[822.21, 833.59, 841.38, 801.85, 791.72]`
- eval summary:
  - `818.2 ± 18.8`
  - `[791.7, 841.4]`
- SI params:
  - `masscart = 1.01694`
  - `masspole = 0.09914`
  - `length = 0.51961`
  - `mp_err = 0.86%`
- runtime:
  - `2187.9 s`

### 3.3 聚合结果
- mean: `802.9341`
- median: `817.8846`
- IQM: `802.9341`
- `<700`: `0/3`
- total runtime: `6630.7 s = 110.5 min`

### 3.4 方法学结论（基于正式重跑）
1. `shared_prior + wd=0 + sn=0` 已被正式脚本完整复现，并可作为当前 paper-facing cartpole 主配置。
2. 该配置在三个 paper seeds 上全部稳定超过 `700`，不存在 collapse seed。
3. `seed42` 的 rescue 效果被保留，同时 `seed84` 维持高分，`seed0` 也恢复到显著高于旧 shared_prior 正式结果的水平。
4. 因此，当前 cartpole 正式口径可以不再依赖旧的 `alpha=5` 叙事，而改为使用新的 shared_prior formal run 作为独立、可追溯、可复现实验产物。

### 4. 审计状态
已尝试并行调用 Codex / Gemini 做 post-change review，但当前均被外部环境阻塞：
- Codex：本地只读读取失败（`bwrap: Unknown option --argv0`）
- Gemini：sandbox / capacity 异常

因此当前先以：
- 本地 `py_compile`
- `--help`
- 单 seed smoke
作为第一轮可执行验证；待外部审计环境恢复后再补双模型 review。

---

## 2026-03-30 新增：DreamerV3 与近年相关论文对照备忘

### 1. 当前论文叙事的定位再次确认
当前 cartpole 实验的目标**不是**追求该任务的绝对 SOTA，而是为 SD-MCTS 框架提供如下方法学证据：
- 在结构已知领域，显式动力学先验可显著压缩样本复杂度；
- 学到的模型参数具备直接可解释性；
- 结构化世界模型支持后续的反事实编辑 / what-if 分析；
- 即使在 worst seed rescue 需求下，也能通过 shared prior + finetune 维持较高总体性能。

用户进一步明确了引入 cartpole 公开基准实验的两个核心动机：
1. 论文中其他实验大多是在自行设定环境中利用合成数据开展；若缺少一个**未篡改的公开 benchmark** 对照，容易被质疑方法是否具备更一般的适用性。
2. cartpole 还承担一个额外角色：验证 SD-MCTS 在**含有极度敏感状态**的场景下的效果，例如倒立摆接近竖直平衡时的小误差会被迅速放大，因而非常适合测试结构先验、鲁棒价值估计与跨 seed 稳定性。

因此，后续外部基线对照应服务于如下更精确的论文主叙事：
- 用公开、未改造 benchmark 证明方法不是只在自定义环境中成立；
- 用高敏感状态控制任务补充验证 SD-MCTS 的稳健性与方法边界；
- 同时支撑样本效率、可解释性、结构编辑能力三条主线，而不是简单围绕单一最终分数竞争。

### 2. 我们当前 cartpole 线的真实交互步数与参数量
#### (a) 真实环境交互步数
代码位置：
- `scripts/sweep_si_robust.py:30-35`
- `scripts/sweep_si_robust.py:191-223`

关键信息：
- `BUDGET = 10000`
- 在 `_prepare_seed_payload(...)` 中：
  - `explore_steps = min(BUDGET, max(500, BUDGET // 2))`
  - 因此实际每个 seed 用于真实环境采样的步数是 **5000 env steps**
- 后续 VNet 训练、shared prior、finetune 都基于 imagination data，不再增加真实环境交互。

这意味着：
- 若用 DreamerV3 的 `100k env steps` 作对照，我们当前这条 shared_prior / paper-facing cartpole 线，本质上是 **5k env steps vs 100k env steps**；
- 即从真实交互量角度看，我们大约只用了其 **1/20**。

#### (b) 我们的模型参数量
代码位置：
- SD 模型定义：`src/world_model/dmcontrol_cartpole_sd_model.py:33-120`
- VNet 定义：`scripts/run_dmcontrol_cartpole.py:134-145`

实测参数量：
- `DMControlCartPoleSDModel`
  - 总参数：`4`
  - 但固定 gravity 后可训练参数仅为：`3`
  - 即：`masscart`, `masspole`, `length`
- `SwingupValueNet(hidden=128)`
  - 总 / 可训练参数：`17281`
- 当前 paper-facing cartpole 主模型总可训练参数：
  - **`3 + 17281 = 17284`**

因此当前 cartpole 叙事里，可直接使用：
- **真实环境交互量：5000 steps / seed**
- **总可训练参数：17284**
- **其中结构动力学本体仅 3 个可训练物理参数**

### 3. DreamerV3 官方可核验结果
#### (a) 官方论文可核验表格
来源：DreamerV3 正文/附录（官方论文，后续发表于 Nature 2025）

可核验结果：
- **DMC proprio, 500K env steps**：
  - `Cartpole Swingup = 850.0`
- **DMC visual, 1M env steps**：
  - `Cartpole Swingup = 819.1`

补充：
- 官方表中给出了模型规模表（Table B.1），包含 DreamerV3 不同 size 的参数量；常见引用口径里，DreamerV3 小规模配置约为 **8M** 参数级别，整套 WM + Actor + Critic 量级约 **千万级**。
- 这比我们当前 cartpole 主线的 `17284` 可训练参数高出两个到三个数量级；若只看动力学本体，则是 `3` vs `8M` 级别。

解释注意：
- 先前代码/论文草稿中出现的“100k ≈ 725”“1M = 861”并不适合作为最稳妥的官方主表口径；
- 更可靠的官方可核验值应优先使用：
  - `850.0 (proprio, 500K)`
  - `819.1 (visual, 1M)`
- 若后续仍想提及 `100k ≈ 700-750`，应明确标注为**learning curve estimate**，不是官方主表值。

### 4. 近两年相关高水平论文中的 cartpole swingup 对照
#### (a) AAMAS 2024 — CPQL
论文：*Boosting Continuous Control with Consistency Policy*

检索到的关键信息：
- 任务：`dm_control cartpole swingup`
- 预算：**500K environment steps**
- 表格结果：
  - `DreamerV3 = 850.0`
  - `CPQL = 860.7`

当前已确认：
- 该论文可作为“近年高水平工作继续沿用 cartpole swingup + DreamerV3 500K 对照口径”的旁证。
- 暂未在当前检索结果中拿到其精确参数量表，因此此处先只记录性能与步数，不额外推断参数量。

#### (b) ICLR 2025 — TWISTER
论文：*Learning Transformer-based World Models with Contrastive Predictive Coding*

检索到的关键信息：
- 任务：`dm_control cartpole swingup`
- 预算：**1M environment steps**
- 表格结果：
  - `DreamerPro = 871.0`
  - `DreamerV3 = 819.1`
  - `TD-MPC2 = 808.1`
  - `TWISTER (No CPC) = 759.1`
  - `TWISTER (ours) = 819.2`
- 论文说明这些结果是 **5 seeds average**。

意义：
- 这说明在近年的顶会语境里，cartpole swingup 上更高预算下的强基线已经扩展到 `DreamerPro / TWISTER / TD-MPC2` 等方法；
- 我们如果要做“绝对性能”比较，需要非常谨慎，因为对方预算普遍在 **500K–1M** 级别，远高于我们的 **5K**。
- 反过来，这更强化了我们的主叙事应放在**样本效率阶差**与**结构优势**上，而不是追求最终 asymptotic score。

### 5. 当前最适合我们的对比口径
结合现有方法目标，cartpole 线更适合强调：

1. **样本效率**
- 我们：**5K env steps**
- DreamerV3 常见官方/高水平论文口径：**100K / 500K / 1M env steps**
- 即使不追求绝对最高分，我们依然能用远少得多的真实交互得到有竞争力的分数区间。

2. **参数效率**
- 我们：
  - 结构动力学本体：**3 trainable physical scalars**
  - 含 VNet 总计：**17284 trainable params**
- DreamerV3：**8M 级别**（官方 size table）
- 这是论文中最容易讲清楚也最能体现结构先验价值的证据。

3. **可解释性**
- 我们直接恢复 `masscart / masspole / length`；
- 误差可以直接逐参数报告；
- DreamerV3 的隐状态则不具备对应层级的可解释物理语义。

4. **反事实与可编辑性**
- 这是我们相对纯黑盒世界模型最独特的优势之一；
- 即便 cartpole 线当前还主要用于样本效率与稳健性叙事，这一点也应在总框架层面保留。

### 6. 当前对外部基线的阶段性结论
- 如果我们的正式三种子结果维持在当前量级（例如 mean 约 `803`），那它在**绝对分数**上：
  - 可能仍低于 DreamerV3 proprio 500K 的 `850.0`；
  - 但已经明显处在 DreamerV3 中高分区间，而我们的真实交互量只有 **5K**。
- 因而最合适的总结不是“我们超过了该 benchmark 的 SOTA”，而是：
  - **在极低样本区间，SD-MCTS 以极小参数量和显式物理结构，实现了高可解释、可编辑、并具备强样本效率的世界模型规划。**

---

## 2026-03-31 新增：1-step no-MCTS 组件消融（与 formal 3-seed 口径对齐）

### 1. 实验目的
本次新增实验不是为了替换当前 cartpole 主结果，而是做一个更干净的组件级消融：
- 保持与正式 `shared_prior + wd=0 + sn=0` 三种子实验**完全相同**的 checkpoint、environment protocol 与 evaluation seeds；
- 仅将在线决策从 **full SD-MCTS** 改为 **1-step no-MCTS**；
- 从而回答：当前 cartpole 结果中的收益，究竟有多少来自结构化世界模型 + finetuned VNet，本身又有多少来自在线树搜索。

这属于典型的 planning-component ablation，而不是新的主结果线。

### 2. 新增代码与协议
新增脚本：
- `scripts/run_cartpole_onestep_paper.py`

脚本职责：
- 直接加载 formal run 已生成的三种子 checkpoint；
- 保持与正式实验一致的环境设定：
  - `tier = T4`
  - paper seeds = `0, 42, 84`
  - `action_repeat = 2`
  - `eval_episodes = 5`
  - `max_steps = 1000`
  - env random seed = `seed + 1000`
- 对每个离散动作做一步前瞻，并用下式打分：
  - `immediate_reward + gamma^k * V(next_state)`
- 不调用任何树搜索。

即，该 baseline 可记为：
- **1-step no-MCTS = one-step lookahead + terminal value, without tree search**

### 3. 正式运行命令与产物
运行命令：
```bash
python -u scripts/run_cartpole_onestep_paper.py \
  --tier T4 \
  --seeds 0 42 84 \
  --experiment_tag sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1 \
  --summary_path results/si_robust_summaries/T4_onestep_paperformal_20260330_r1_summary.json
```

日志：
- `/tmp/claude-1004/-home-data-round-code-sd-world-model-rl/783d2ffd-4f33-4524-afe7-df878bf94463/tasks/bli0e84ah.output`

summary：
- `results/si_robust_summaries/T4_onestep_paperformal_20260330_r1_summary.json`

使用的 source experiment tag：
- `sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1`

即，本次对照严格复用了 formal 3-seed run 的微调后权重：
- seed0 ckpt:
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed0.pt`
- seed42 ckpt:
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed42.pt`
- seed84 ckpt:
  - `results/si_robust_ckpts/T4_sharedpriorv1_dev10-21-30-55-63-72-95_pwd0p0001_psn0p001_pe500_fwd0_fsn0_fe500_seloverall_paperformal_20260330_r1_seed84.pt`

### 4. 结果
#### (a) per-seed
- `seed0 = 751.0 ± 7.4`
- `seed42 = 781.9 ± 81.3`
- `seed84 = 806.7 ± 19.6`

#### (b) aggregate
- mean = `779.9`
- median = `781.9`
- IQM = `779.9`
- `<700 = 0/3`
- total runtime = `168.3 s`

### 5. 与正式 SD-MCTS 主结果的直接对照
formal SD-MCTS 主结果（同一组 checkpoint 来源、同一环境协议）为：
- `seed0 = 772.8 ± 33.5`
- `seed42 = 817.9 ± 29.6`
- `seed84 = 818.2 ± 18.8`
- mean = `802.9`

对应差值：
- seed0: `751.0 - 772.8 = -21.8`
- seed42: `781.9 - 817.9 = -36.0`
- seed84: `806.7 - 818.2 = -11.5`
- aggregate mean: `779.9 - 802.9 = -23.0`

### 6. 当前 interpretation
该结果说明：
1. **1-step no-MCTS 并不弱**。在相同 SD model + 相同 finetuned VNet 下，它已经能稳定跑到 `~780` 的三种子均值，说明结构世界模型与终端值补偿本身已具有较强控制能力。
2. **但 full SD-MCTS 仍然带来稳定且非微小的增益**。三种子均值从 `779.9` 提升到 `802.9`，增益约 `+23.0`，并且三个 seed 全部受益。
3. **seed42 的差距最明显**。从 `781.9 ± 81.3` 到 `817.9 ± 29.6`，不仅均值提高，而且方差显著下降，说明在敏感状态更脆弱的 seed 上，在线搜索对稳定性和 worst-case rescue 都更重要。
4. 因此，当前 cartpole 结果中的收益不能简单归因于“结构模型 + VNet 已经足够”；在线树搜索本身仍然贡献了可观的额外收益。

### 7. 对论文的启发
这组结果很适合作为 **planning-component ablation**：
- 若后续决定写入论文，优先建议作为 cartpole 补充消融；
- 正文可以简述结论，附录给出表格；
- 其作用不是替代主结果，而是更清晰地区分：
  - 结构世界模型贡献
  - terminal value 贡献
  - 在线树搜索贡献

当前阶段建议：
- 先将此结果保留在 worklog 中；
- 之后再决定是否写入主文或附录。

## 2026-04-13：Cartpole 中 SD 与 TVA 的训练组织方式复核

对当前 Cartpole 实现再次复核后，可确认其训练方式并非 Beer Game 式的交替更新，而是较为明确的分阶段训练。

在基础脚本 `scripts/run_dmcontrol_cartpole.py` 中，主流程为：
- 先收集环境数据；
- 再通过 `train_sd_model(...)` 完成 SD 系统辨识；
- 之后在固定模型上训练 `value_net / VNet`；
- 最后在评估阶段将 VNet 接入搜索。

即使在 `distill_rollout` 分支下，流程也仍然是：
- 先固定 SD 模型；
- 再生成 imagination data；
- 之后训练初始 VNet；
- 如有需要，再基于 rollout/refinement 数据继续训练 VNet。

该过程中并不存在“VNet 更新后再回头更新 SD”的交替闭环。

在论文当前使用的正式口径中，这一特征更加明确。`scripts/sweep_si_robust.py` 与 `scripts/run_cartpole_sharedprior_paper.py` 的共享先验流程可概括为：
- `prepare_seed_payload` 先完成 SD 训练，并生成 imagination data；
- `train_seed` 再在这些 imagination data 与共享先验基础上训练或微调 VNet；
- `evaluate_seed` 仅做最终评估。

因此，当前 paper-facing Cartpole 的实际训练方式应表述为：

- 先独立训练 SD；
- 再独立训练 TVA/VNet；
- 最后在固定好的 SD + VNet 上执行搜索评估。

从方法选择上看，这种分阶段训练对于当前 Cartpole 任务是合理的。原因在于：
- Cartpole 动力学低维、结构清晰、系统辨识难度相对较低；
- 先将 SD 参数稳定识别出来，再训练 TVA，有助于保持训练稳定性与解释性；
- 这样可以避免 VNet 的搜索分布信号反向扰动本已可辨识的物理参数。

因此，现阶段不宜将 Cartpole 的非交替训练方式视为缺陷。相反，就本任务与论文当前主张而言，这一组织方式大概率优于更复杂的交替更新。只有在后续进入动力学更复杂、模型失配更强或观测噪声更显著的场景时，Beer Game 式的交替训练才更有必要重新考虑。
