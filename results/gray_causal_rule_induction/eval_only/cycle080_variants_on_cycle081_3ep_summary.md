# Gray Rule Variant Eval-only Sweep

Run: `tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1_cycle081_target405080`
Checkpoint cycle: `81`
Rule cycle: `80`
TVA cycle: `80`
SD checkpoint: `/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/sd_explainers/sd_explainer_cycle_080.pt`
TVA checkpoint: `/home/data/round/code/sd_world_model_rl/results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_080/tva/hopper_tva_cycle_080.pt`

## Mode Summary

| mode | seed means | mean | std | loaded rules |
|---|---|---:|---:|---:|
| `rule_off` | `439.874; 327.041; 295.729` | 354.215 | 61.905 | 0.0 |
| `cycle080_original_active` | `426.016; 334.047; 295.853` | 351.972 | 54.629 | 130.0 |
| `cycle080_drop_reject_active` | `424.756; 330.950; 294.294` | 350.000 | 54.938 | 126.0 |
| `cycle080_strict_active` | `444.463; 328.137; 293.231` | 355.277 | 64.654 | 125.0 |

## Deltas

| mode | vs rule_off | vs original active |
|---|---:|---:|
| `rule_off` | 0.000 | 2.243 |
| `cycle080_original_active` | -2.243 | 0.000 |
| `cycle080_drop_reject_active` | -4.215 | -1.972 |
| `cycle080_strict_active` | 1.062 | 3.305 |

## Episode Rewards

### seed1 `rule_off`

- mean/std/min/max: `439.874` / `16.477` / `419.935` / `460.288`
- rewards: `460.288;419.935;439.399`
- rules loaded: `0`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed1_rule_off.json`

### seed2 `rule_off`

- mean/std/min/max: `327.041` / `10.994` / `313.329` / `340.245`
- rewards: `313.329;340.245;327.548`
- rules loaded: `0`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed2_rule_off.json`

### seed3 `rule_off`

- mean/std/min/max: `295.729` / `6.827` / `288.062` / `304.646`
- rewards: `294.479;288.062;304.646`
- rules loaded: `0`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed3_rule_off.json`

### seed1 `cycle080_original_active`

- mean/std/min/max: `426.016` / `21.980` / `396.048` / `448.150`
- rewards: `433.849;396.048;448.150`
- rules loaded: `130`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed1_cycle080_original_active.json`

### seed2 `cycle080_original_active`

- mean/std/min/max: `334.047` / `7.045` / `324.412` / `341.059`
- rewards: `324.412;336.671;341.059`
- rules loaded: `130`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed2_cycle080_original_active.json`

### seed3 `cycle080_original_active`

- mean/std/min/max: `295.853` / `2.840` / `291.944` / `298.604`
- rewards: `298.604;297.012;291.944`
- rules loaded: `130`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed3_cycle080_original_active.json`

### seed1 `cycle080_drop_reject_active`

- mean/std/min/max: `424.756` / `14.466` / `404.519` / `437.474`
- rewards: `437.474;404.519;432.275`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed1_cycle080_drop_reject_active.json`

### seed2 `cycle080_drop_reject_active`

- mean/std/min/max: `330.950` / `4.725` / `324.412` / `335.416`
- rewards: `324.412;333.021;335.416`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed2_cycle080_drop_reject_active.json`

### seed3 `cycle080_drop_reject_active`

- mean/std/min/max: `294.294` / `2.582` / `290.841` / `297.048`
- rewards: `290.841;294.992;297.048`
- rules loaded: `126`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed3_cycle080_drop_reject_active.json`

### seed1 `cycle080_strict_active`

- mean/std/min/max: `444.463` / `11.314` / `428.870` / `455.367`
- rewards: `455.367;428.870;449.152`
- rules loaded: `125`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed1_cycle080_strict_active.json`

### seed2 `cycle080_strict_active`

- mean/std/min/max: `328.137` / `8.704` / `315.901` / `335.416`
- rewards: `315.901;333.095;335.416`
- rules loaded: `125`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed2_cycle080_strict_active.json`

### seed3 `cycle080_strict_active`

- mean/std/min/max: `293.231` / `1.752` / `290.841` / `294.992`
- rewards: `290.841;294.992;293.859`
- rules loaded: `125`
- report: `/home/data/round/code/sd_world_model_rl/results/gray_causal_rule_induction/eval_only/cycle080_variants_on_cycle081_3ep/seed3_cycle080_strict_active.json`
