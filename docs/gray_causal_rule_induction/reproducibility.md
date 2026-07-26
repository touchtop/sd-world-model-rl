# Reproducibility

## 环境

当前研究线使用仓库现有 Python 环境与标准库实现第一版只读审计。第一版代码不依赖额外第三方库。

eval-only 使用仓库既有 `sdmcts_p100` 环境：

```text
/home/anaconda3/envs/sdmcts_p100/bin/python
```

Hopper eval 当前使用 `MUJOCO_GL=glfw`。在本机上 `MUJOCO_GL=egl` 会触发 headless EGL 初始化失败，导致 `dm_control` 后端无法注册 `hopper-hop`。

## 输入

输入为既有 `sd_retrospective_rule_induction` 实验中已经落盘的 JSON 产物，包括规则库、规则效果审计、TVA shadow update、TVA consequence gate 和 TVA terminal prior gate。输入文件只读使用。

## 输出

默认输出目录为：

```text
results/gray_causal_rule_induction/
```

脚本默认不覆盖已存在输出文件，除非显式传入 `--overwrite`。

已生成的关键输出包括：

```text
results/gray_causal_rule_induction/cycle079_gray_rule_certificates.json
results/gray_causal_rule_induction/rule_library_variants_cycle079/
results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3_summary.json
results/gray_causal_rule_induction/cycle080_gray_rule_certificates.json
results/gray_causal_rule_induction/rule_library_variants_cycle080/
results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep_summary.json
results/gray_causal_rule_induction/closed_loop/hopper_v8formal_300k_graycausal_strict_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260604_r1/
```

## 随机性

第一版证书审计为确定性过程，不使用随机采样。若后续加入抽样式规则反事实重审，应固定随机种子并在本文件中记录。

eval-only smoke 使用 checkpoint 对应种子 `1,2,3`，每种子 `3` 个 episode。该结果受 episode 采样影响，仅作为方向性验证。

## 复现命令

语法检查：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/gray_causal_rule_induction/*.py
```

运行 300k 灰因果规则闭环训练：

```bash
experiments/gray_causal_rule_induction/launch_hopper_300k_gray_strict.sh
```

默认配置为三种子 `1,2,3`、GPU `1,2,3`、`60` 个 cycle、每 cycle `5000` 步，输出目录为：

```text
results/gray_causal_rule_induction/closed_loop/hopper_v8formal_300k_graycausal_strict_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260604_r1/
```

如需改变 GPU，可在启动时设置 `GPU_IDS`，例如：

```bash
GPU_IDS=0,2,3 experiments/gray_causal_rule_induction/launch_hopper_300k_gray_strict.sh
```

重新导出 `cycle_079` 灰过滤规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/gray_causal_rule_induction/export_gray_filtered_rule_libraries.py \
  --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_079/rules/library_closed_loop_cycle_079_confidence.json \
  --certificates results/gray_causal_rule_induction/cycle079_gray_rule_certificates.json \
  --output-dir results/gray_causal_rule_induction/rule_library_variants_cycle079 \
  --prefix cycle079 \
  --overwrite
```

运行 `cycle_079` 规则变体在 400k checkpoint 上的 3 episode smoke：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -u \
  experiments/gray_causal_rule_induction/eval_gray_rule_variants.py \
  --eval-episodes 3 \
  --seeds 1 2 3 \
  --gpus 1 2 3 \
  --mujoco-gl glfw
```

运行 `cycle_080` 规则变体在 `cycle_081` checkpoint 上的 3 episode smoke：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -u \
  experiments/gray_causal_rule_induction/eval_gray_rule_variants.py \
  --rule-cycle 80 \
  --tva-cycle 80 \
  --checkpoint-cycle 81 \
  --seeds 1 2 3 \
  --gpus 0 0 0 \
  --modes rule_off original_active drop_reject_active strict_active \
  --eval-episodes 3 \
  --mujoco-gl glfw
```

该命令在 GPU0 上执行。运行时 GPU0 存在既有显存占用但无计算负载，评估期间未使用 GPU 1、2、3。输出目录为：

```text
results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/
```

生成 `cycle_080` 灰证书：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/gray_causal_rule_induction/audit_gray_rule_certificates.py \
  --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/rules/library_closed_loop_cycle_080_confidence.json \
  --rule-effect-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/reports/rule_effect_audit.json \
  --tva-shadow-update results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/reports/tva_shadow_update.json \
  --tva-consequence-gate results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/reports/tva_shadow_consequence_gated_update.json \
  --tva-terminal-gate results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/reports/tva_terminal_prior_gate.json \
  --output-json results/gray_causal_rule_induction/cycle080_gray_rule_certificates.json \
  --output-md results/gray_causal_rule_induction/cycle080_gray_rule_certificates.md \
  --overwrite
```

导出 `cycle_080` 灰过滤规则库：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/gray_causal_rule_induction/export_gray_filtered_rule_libraries.py \
  --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/rules/library_closed_loop_cycle_080_confidence.json \
  --certificates results/gray_causal_rule_induction/cycle080_gray_rule_certificates.json \
  --output-dir results/gray_causal_rule_induction/rule_library_variants_cycle080 \
  --prefix cycle080 \
  --overwrite
```

## 2026-06-11 有限记忆样本选择

当前 1M 接续实验的样本记忆策略文件为：

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_1m_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume500k_20260607_r1/selection_memory_policy.json
```

策略内容为：

```json
{
  "policy_version": "bounded_retrospective_selection_v1",
  "max_selected_episodes": 1800,
  "max_selected_per_cycle_step": 600,
  "preserve_sources": ["official_eval"]
}
```

验证命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/stratify_trajectories.py \
  --manifest results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_1m_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume500k_20260607_r1/cycle_146/manifests/manifest.json \
  --output /tmp/hopper_cycle146_selected_bounded.json \
  --good-quantile 0.80 \
  --bad-quantile 0.20 \
  --random-per-cycle 10 \
  --score-mode stage \
  --seed 20260653 \
  --memory-policy results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_1m_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume500k_20260607_r1/selection_memory_policy.json
```

验证结果为 `selected episodes: 1800 / 13080`。`memory_stats` 中保留后标签分布为 good `897`、bad `885`、middle `18`，主要 cycle step 的保留数为 `600`、`600` 和 `581`，正式评估来源被保留。

语法检查命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -m py_compile \
  experiments/sd_retrospective_rule_induction/common.py \
  experiments/sd_retrospective_rule_induction/counterfactual_actions.py \
  experiments/sd_retrospective_rule_induction/replay_rule_scoring.py \
  experiments/sd_retrospective_rule_induction/validate_substate_rules.py \
  experiments/sd_retrospective_rule_induction/mine_hopper_v8_success_segments.py \
  experiments/sd_retrospective_rule_induction/audit_hopper_v8_widefunnel_rule_effects.py \
  experiments/sd_retrospective_rule_induction/stratify_trajectories.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py
```

多 CPU 后处理默认 worker 数为 `min(16, os.cpu_count())`，当前服务器 `os.cpu_count()` 为 `72`。可通过以下方式覆盖：

```bash
export SD_RETRO_WORKERS=8
```

或在支持的脚本中传入：

```bash
--workers 8
```

launcher 级复现实验可使用：

```bash
--postprocess-workers 8
```

小样本一致性验证基于 `/tmp/sd_retro_parallel_test/`。验证内容包括：

```text
replay_rule_scoring.py: workers 1 vs 4 -> MATCH
validate_substate_rules.py: workers 1 vs 4 -> MATCH
mine_hopper_v8_success_segments.py: workers 1 vs 4 -> MATCH
audit_hopper_v8_widefunnel_rule_effects.py: workers 1 vs 4 -> MATCH
counterfactual_actions.py: workers 1 vs 2 -> MATCH
```

dry-run 验证命令：

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python \
  experiments/sd_retrospective_rule_induction/run_hopper_v7_closed_loop.py \
  --gpu-id 0 \
  --cycles 1 \
  --cycle-steps 5 \
  --eval-freq 5 \
  --checkpoint-freq 5 \
  --dry-run \
  --output-root /tmp/sd_retro_parallel_dryrun \
  --postprocess-workers 4 \
  --enable-v8-transition-rules \
  --enable-rule-effect-confidence | grep -- '--workers'
```

该 dry-run 确认 `counterfactual_actions.py`、`replay_rule_scoring.py`、`validate_substate_rules.py`、`mine_hopper_v8_success_segments.py` 和 `audit_hopper_v8_widefunnel_rule_effects.py` 均会接收 `--workers 4`。
