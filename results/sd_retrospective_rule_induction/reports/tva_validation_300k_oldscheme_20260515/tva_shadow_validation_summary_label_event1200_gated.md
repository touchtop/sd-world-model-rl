# TVA Shadow Validation Summary

Run root: `results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1`
Shadow directory: `results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515`
Cycles: `[41, 50, 60]`

## Cycle Overview

| Cycle | Rules | Matched pairs | Strengthened | Weakened | Unchanged | Mean delta | Strengthened effect consistent/harmful | Weakened effect consistent/harmful |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 143 | 7956 | 8 | 7 | 128 | -0.0057 | 0.370/0.255 | 0.549/0.308 |
| 50 | 157 | 7957 | 6 | 9 | 142 | -0.0065 | 0.395/0.272 | 0.342/0.102 |
| 60 | 180 | 8525 | 8 | 9 | 163 | -0.0072 | 0.501/0.249 | 0.415/0.141 |

## Persistent TVA Shadow Changes

### Strengthened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| `transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` | `[41, 50, 60]` | `+0.1259` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | `[41, 50, 60]` | `+0.0788` |
| `unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` | `[41, 50]` | `+0.0414` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_pos` | `[41, 50, 60]` | `+0.0360` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_pos` | `[50, 60]` | `+0.0298` |

### Weakened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | `[41, 50, 60]` | `-0.3759` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | `[41, 50, 60]` | `-0.3427` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | `[41, 50, 60]` | `-0.3386` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_neg` | `[41, 50, 60]` | `-0.0586` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:waist_neg` | `[50, 60]` | `-0.0381` |
| `unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:hip_neg` | `[41, 50, 60]` | `-0.0294` |

## Key Rules

### Cycle 41

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 109 | 0.018/0.110 | 0.704/0.296 | `insufficient_tva_effect_coverage` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.872 | -0.128 | 147 | 0.034/0.218 | 0.809/0.191 | `tva_shadow_harmful_weakened` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.541 -> 0.591 | +0.050 | 171 | 0.450/0.140 | 0.536/0.464 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.212 | -0.348 | 147 | 0.102/0.605 | 0.684/0.316 | `tva_shadow_harmful_deactivated` |

### Cycle 50

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 112 | 0.062/0.027 | 0.694/0.306 | `insufficient_tva_effect_coverage` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.500 | -0.500 | 118 | 0.008/0.551 | 0.813/0.187 | `tva_shadow_harmful_deactivated` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.210 -> 0.297 | +0.087 | 127 | 0.701/0.079 | 0.482/0.518 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.255 | -0.305 | 118 | 0.144/0.653 | 0.683/0.317 | `tva_shadow_harmful_deactivated` |

### Cycle 60

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 129 | 0.062/0.178 | 0.688/0.312 | `insufficient_tva_effect_coverage` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.500 | -0.500 | 120 | 0.008/0.350 | 0.821/0.179 | `tva_shadow_harmful_deactivated` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.500 | -0.500 | 141 | 0.007/0.362 | 0.810/0.190 | `tva_shadow_harmful_deactivated` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.000 -> 0.100 | +0.100 | 146 | 0.740/0.041 | 0.000/0.000 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.197 | -0.363 | 120 | 0.125/0.683 | 0.676/0.324 | `tva_shadow_harmful_deactivated` |
