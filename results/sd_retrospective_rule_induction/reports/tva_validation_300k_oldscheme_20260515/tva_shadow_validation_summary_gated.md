# TVA Shadow Validation Summary

Run root: `results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1`
Shadow directory: `results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515`
Cycles: `[41, 50, 60]`

## Cycle Overview

| Cycle | Rules | Matched pairs | Strengthened | Weakened | Unchanged | Mean delta | Strengthened effect consistent/harmful | Weakened effect consistent/harmful |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 143 | 2068 | 0 | 0 | 143 | 0.0000 | 0.000/0.000 | 0.000/0.000 |
| 50 | 157 | 2285 | 0 | 0 | 157 | 0.0000 | 0.000/0.000 | 0.000/0.000 |
| 60 | 180 | 2356 | 2 | 3 | 175 | 0.0008 | 0.309/0.191 | 0.000/0.000 |

## Persistent TVA Shadow Changes

### Strengthened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| none |  |  |

### Weakened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| none |  |  |

## Key Rules

### Cycle 41

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 19 | 0.000/0.105 | 0.704/0.296 | `low_tva_hits_soft_decay` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 48 | 0.021/0.229 | 0.809/0.191 | `tva_shadow_opposite_but_low_margin` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.541 -> 0.541 | +0.000 | 52 | 0.462/0.173 | 0.536/0.464 | `tva_shadow_positive_but_diffuse_best_family` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.560 | +0.000 | 48 | 0.104/0.583 | 0.684/0.316 | `tva_shadow_positive_but_low_directional_rate_margin` |

### Cycle 50

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 14 | 0.071/0.071 | 0.694/0.306 | `low_tva_hits_soft_decay` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 34 | 0.000/0.471 | 0.813/0.187 | `insufficient_tva_effect_coverage` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.210 -> 0.210 | +0.000 | 48 | 0.708/0.062 | 0.482/0.518 | `tva_shadow_positive_but_low_target_margin` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.560 | +0.000 | 34 | 0.147/0.765 | 0.683/0.317 | `insufficient_tva_effect_coverage` |

### Cycle 60

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 17 | 0.059/0.294 | 0.688/0.312 | `low_tva_hits_soft_decay` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 42 | 0.000/0.381 | 0.821/0.179 | `insufficient_tva_effect_coverage` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 43 | 0.000/0.372 | 0.810/0.190 | `insufficient_tva_effect_coverage` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.000 -> 0.093 | +0.093 | 51 | 0.725/0.039 | 0.000/0.000 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.560 | +0.000 | 42 | 0.119/0.714 | 0.676/0.324 | `insufficient_tva_effect_coverage` |
