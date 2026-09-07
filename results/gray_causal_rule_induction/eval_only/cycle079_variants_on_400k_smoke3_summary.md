# Gray Rule Variant Eval-only Sweep

Run: `tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle080_target400079`
Cycle reference: `79`
SD checkpoint: `/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_079/sd_explainers/sd_explainer_cycle_079.pt`
TVA checkpoint: `/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_079/tva/hopper_tva_cycle_079.pt`

## Mode Summary

| mode | seed means | mean | std | loaded rules |
|---|---|---:|---:|---:|
| `rule_off` | `446.736; 331.332; 299.172` | 359.080 | 63.357 | 0.0 |
| `cycle079_original_active` | `436.284; 336.469; 293.601` | 355.452 | 59.777 | 130.0 |
| `cycle079_drop_reject_active` | `456.731; 338.535; 296.656` | 363.974 | 67.781 | 126.0 |
| `cycle079_certified_active` | `456.731; 338.535; 296.656` | 363.974 | 67.781 | 126.0 |
| `cycle079_strict_active` | `460.818; 333.408; 298.470` | 364.232 | 69.770 | 125.0 |
| `cycle079_teacher_v8_active` | `456.842; 331.052; 293.939` | 360.611 | 69.712 | 31.0 |
| `cycle079_teacher_v8_certified_active` | `434.075; 331.052; 293.939` | 353.022 | 59.282 | 28.0 |

## Deltas

| mode | vs rule_off | vs original active |
|---|---:|---:|
| `rule_off` | 0.000 | 3.629 |
| `cycle079_original_active` | -3.629 | 0.000 |
| `cycle079_drop_reject_active` | 4.894 | 8.523 |
| `cycle079_certified_active` | 4.894 | 8.523 |
| `cycle079_strict_active` | 5.152 | 8.780 |
| `cycle079_teacher_v8_active` | 1.531 | 5.159 |
| `cycle079_teacher_v8_certified_active` | -6.058 | -2.429 |

## Episode Rewards

### seed1 `rule_off`

- mean/std/min/max: `446.736` / `9.022` / `434.628` / `456.275`
- rewards: `456.275;449.305;434.628`
- rules loaded: `0`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_rule_off.json`

### seed2 `rule_off`

- mean/std/min/max: `331.332` / `18.440` / `305.258` / `344.823`
- rewards: `305.258;343.914;344.823`
- rules loaded: `0`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_rule_off.json`

### seed3 `rule_off`

- mean/std/min/max: `299.172` / `1.947` / `297.759` / `301.925`
- rewards: `301.925;297.833;297.759`
- rules loaded: `0`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_rule_off.json`

### seed1 `cycle079_original_active`

- mean/std/min/max: `436.284` / `7.979` / `426.558` / `446.103`
- rewards: `426.558;446.103;436.192`
- rules loaded: `130`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_cycle079_original_active.json`

### seed2 `cycle079_original_active`

- mean/std/min/max: `336.469` / `2.286` / `334.719` / `339.699`
- rewards: `334.719;339.699;334.989`
- rules loaded: `130`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_cycle079_original_active.json`

### seed3 `cycle079_original_active`

- mean/std/min/max: `293.601` / `5.661` / `285.638` / `298.298`
- rewards: `296.867;298.298;285.638`
- rules loaded: `130`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_cycle079_original_active.json`

### seed1 `cycle079_drop_reject_active`

- mean/std/min/max: `456.731` / `21.835` / `425.930` / `474.048`
- rewards: `425.930;470.214;474.048`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_cycle079_drop_reject_active.json`

### seed2 `cycle079_drop_reject_active`

- mean/std/min/max: `338.535` / `2.766` / `334.719` / `341.188`
- rewards: `334.719;339.699;341.188`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_cycle079_drop_reject_active.json`

### seed3 `cycle079_drop_reject_active`

- mean/std/min/max: `296.656` / `7.543` / `286.488` / `304.533`
- rewards: `286.488;304.533;298.947`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_cycle079_drop_reject_active.json`

### seed1 `cycle079_certified_active`

- mean/std/min/max: `456.731` / `21.835` / `425.930` / `474.048`
- rewards: `425.930;470.214;474.048`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_cycle079_certified_active.json`

### seed2 `cycle079_certified_active`

- mean/std/min/max: `338.535` / `2.766` / `334.719` / `341.188`
- rewards: `334.719;339.699;341.188`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_cycle079_certified_active.json`

### seed3 `cycle079_certified_active`

- mean/std/min/max: `296.656` / `7.543` / `286.488` / `304.533`
- rewards: `286.488;304.533;298.947`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_cycle079_certified_active.json`

### seed1 `cycle079_strict_active`

- mean/std/min/max: `460.818` / `15.173` / `440.623` / `477.196`
- rewards: `477.196;440.623;464.634`
- rules loaded: `125`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_cycle079_strict_active.json`

### seed2 `cycle079_strict_active`

- mean/std/min/max: `333.408` / `7.136` / `323.951` / `341.188`
- rewards: `335.086;323.951;341.188`
- rules loaded: `125`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_cycle079_strict_active.json`

### seed3 `cycle079_strict_active`

- mean/std/min/max: `298.470` / `4.875` / `292.596` / `304.533`
- rewards: `298.279;304.533;292.596`
- rules loaded: `125`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_cycle079_strict_active.json`

### seed1 `cycle079_teacher_v8_active`

- mean/std/min/max: `456.842` / `10.850` / `443.135` / `469.668`
- rewards: `469.668;443.135;457.722`
- rules loaded: `31`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_cycle079_teacher_v8_active.json`

### seed2 `cycle079_teacher_v8_active`

- mean/std/min/max: `331.052` / `2.251` / `328.993` / `334.184`
- rewards: `329.980;328.993;334.184`
- rules loaded: `31`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_cycle079_teacher_v8_active.json`

### seed3 `cycle079_teacher_v8_active`

- mean/std/min/max: `293.939` / `7.444` / `283.550` / `300.610`
- rewards: `297.657;300.610;283.550`
- rules loaded: `31`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_cycle079_teacher_v8_active.json`

### seed1 `cycle079_teacher_v8_certified_active`

- mean/std/min/max: `434.075` / `23.399` / `410.472` / `465.963`
- rewards: `465.963;410.472;425.791`
- rules loaded: `28`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed1_cycle079_teacher_v8_certified_active.json`

### seed2 `cycle079_teacher_v8_certified_active`

- mean/std/min/max: `331.052` / `2.251` / `328.993` / `334.184`
- rewards: `329.980;328.993;334.184`
- rules loaded: `28`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed2_cycle079_teacher_v8_certified_active.json`

### seed3 `cycle079_teacher_v8_certified_active`

- mean/std/min/max: `293.939` / `7.444` / `283.550` / `300.610`
- rewards: `297.657;300.610;283.550`
- rules loaded: `28`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle079_variants_on_400k_smoke3/seed3_cycle079_teacher_v8_certified_active.json`
