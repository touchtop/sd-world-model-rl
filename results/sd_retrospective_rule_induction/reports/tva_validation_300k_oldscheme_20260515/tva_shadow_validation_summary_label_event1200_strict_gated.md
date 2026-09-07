# TVA Shadow Validation Summary

Run root: `results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1`
Shadow directory: `results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515`
Cycles: `[41, 50, 60]`

## Cycle Overview

| Cycle | Rules | Matched pairs | Strengthened | Weakened | Unchanged | Mean delta | Strengthened effect consistent/harmful | Weakened effect consistent/harmful |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 143 | 7956 | 1 | 3 | 139 | -0.0049 | 0.000/0.000 | 0.743/0.257 |
| 50 | 157 | 7957 | 4 | 5 | 148 | 0.0010 | 0.451/0.299 | 0.000/0.000 |
| 60 | 180 | 8525 | 7 | 4 | 169 | 0.0024 | 0.491/0.224 | 0.000/0.000 |

## Persistent TVA Shadow Changes

### Strengthened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| `transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` | `[41, 50, 60]` | `+0.1259` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | `[50, 60]` | `+0.0935` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` | `[50, 60]` | `+0.0326` |

### Weakened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_neg` | `[50, 60]` | `-0.0381` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_neg` | `[50, 60]` | `-0.0281` |
| `unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:hip_neg` | `[50, 60]` | `-0.0266` |

## Key Rules

### Cycle 41

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 109 | 0.018/0.110 | 0.704/0.296 | `insufficient_tva_effect_coverage` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.872 | -0.128 | 147 | 0.034/0.218 | 0.809/0.191 | `tva_shadow_harmful_weakened` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.541 -> 0.541 | +0.000 | 171 | 0.450/0.140 | 0.536/0.464 | `tva_shadow_positive_but_diffuse_best_family` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.212 | -0.348 | 147 | 0.102/0.605 | 0.684/0.316 | `tva_shadow_harmful_deactivated` |

### Cycle 50

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 112 | 0.062/0.027 | 0.694/0.306 | `insufficient_tva_effect_coverage` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 118 | 0.008/0.551 | 0.813/0.187 | `insufficient_tva_effect_coverage` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.210 -> 0.297 | +0.087 | 127 | 0.701/0.079 | 0.482/0.518 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.560 | +0.000 | 118 | 0.144/0.653 | 0.683/0.317 | `insufficient_tva_effect_coverage` |

### Cycle 60

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 129 | 0.062/0.178 | 0.688/0.312 | `insufficient_tva_effect_coverage` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 120 | 0.008/0.350 | 0.821/0.179 | `insufficient_tva_effect_coverage` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 141 | 0.007/0.362 | 0.810/0.190 | `insufficient_tva_effect_coverage` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.000 -> 0.100 | +0.100 | 146 | 0.740/0.041 | 0.000/0.000 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.560 | +0.000 | 120 | 0.125/0.683 | 0.676/0.324 | `insufficient_tva_effect_coverage` |
