# TVA Shadow Validation Summary

Run root: `results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_confidence_toerecovery_auditfix_seed123_20260514_r1`
Shadow directory: `results/sd_retrospective_rule_induction/reports/tva_validation_300k_oldscheme_20260515`
Cycles: `[41, 50, 60]`

## Cycle Overview

| Cycle | Rules | Matched pairs | Strengthened | Weakened | Unchanged | Mean delta | Strengthened effect consistent/harmful | Weakened effect consistent/harmful |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 143 | 2068 | 11 | 14 | 118 | -0.0014 | 0.434/0.293 | 0.501/0.284 |
| 50 | 157 | 2285 | 10 | 15 | 132 | -0.0016 | 0.323/0.277 | 0.507/0.227 |
| 60 | 180 | 2356 | 16 | 19 | 145 | -0.0013 | 0.482/0.268 | 0.446/0.186 |

## Persistent TVA Shadow Changes

### Strengthened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| `transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->soft_prefer:hip_pos` | `[41, 60]` | `+0.0141` |
| `unsupported_high_torque|omega_neg|vx_low|support_none|falling->prefer:ankle_neg` | `[41, 50, 60]` | `+0.0043` |
| `rotation_risk|omega_neg|vx_low|support_none|falling->prefer:ankle_neg` | `[41, 50, 60]` | `+0.0038` |

### Weakened

| Rule | Cycles | Mean delta |
| --- | --- | ---: |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | `[41, 50, 60]` | `-0.0715` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | `[41, 50, 60]` | `-0.0700` |
| `rotation_risk|omega_pos|vx_low|support_none|falling->prefer:hip_neg` | `[50, 60]` | `-0.0302` |
| `unsupported_high_torque|omega_pos|vx_mid|support_none|falling->prefer:hip_neg` | `[41, 50, 60]` | `-0.0294` |
| `rotation_risk|omega_pos|vx_low|support_none|falling->prefer:knee_neg` | `[41, 50, 60]` | `-0.0211` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:knee_neg` | `[41, 50, 60]` | `-0.0142` |
| `unsupported_high_torque|omega_neg|vx_low|support_none|falling->prefer:ankle_pos` | `[41, 50, 60]` | `-0.0121` |
| `unsupported_high_torque|omega_pos|vx_low|support_none|falling->prefer:waist_pos` | `[41, 50, 60]` | `-0.0071` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:knee_neg` | `[41, 60]` | `-0.0070` |
| `transition|transition_gap|omega_pos|vx_high|support_none|phase_any|rotation_risk->soft_prefer:knee_neg` | `[41, 50, 60]` | `-0.0062` |
| `rotation_risk|omega_neg|vx_low|support_none|falling->prefer:knee_pos` | `[41, 50]` | `-0.0048` |

## Key Rules

### Cycle 41

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 19 | 0.000/0.000 | 0.704/0.296 | `low_tva_hits_soft_decay` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 48 | 0.000/0.000 | 0.809/0.191 | `tva_shadow_observed_no_large_change` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.541 -> 0.530 | -0.011 | 52 | 0.115/0.058 | 0.536/0.464 | `tva_shadow_soft_weakened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.495 | -0.065 | 48 | 0.021/0.146 | 0.684/0.316 | `tva_shadow_soft_weakened` |

### Cycle 50

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 14 | 0.000/0.000 | 0.694/0.306 | `low_tva_hits_soft_decay` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 1.000 | +0.000 | 34 | 0.000/0.000 | 0.813/0.187 | `tva_shadow_observed_no_large_change` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | missing |  |  |  |  |  |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.210 -> 0.227 | +0.018 | 48 | 0.229/0.021 | 0.482/0.518 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.483 | -0.077 | 34 | 0.029/0.235 | 0.683/0.317 | `tva_shadow_harmful_weakened` |

### Cycle 60

| Rule | Conf -> shadow | Delta | TVA hits | TVA same/opposite | Effect consistent/harmful | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toe_recovery_low_support->soft_prefer:waist_pos` | 0.750 -> 0.750 | +0.000 | 17 | 0.000/0.059 | 0.688/0.312 | `low_tva_hits_soft_decay` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.972 | -0.028 | 42 | 0.000/0.071 | 0.821/0.179 | `tva_shadow_soft_weakened` |
| `rotation_risk|omega_pos|vx_high|support_none|falling->prefer:ankle_pos` | 1.000 -> 0.973 | -0.027 | 43 | 0.000/0.070 | 0.810/0.190 | `tva_shadow_soft_weakened` |
| `rotation_risk|omega_pos|vx_mid|support_none|falling->prefer:hip_pos` | 0.000 -> 0.065 | +0.065 | 51 | 0.529/0.000 | 0.000/0.000 | `tva_shadow_consistent_strengthened` |
| `unsupported_high_torque|omega_pos|vx_high|support_none|falling->prefer:hip_neg` | 0.560 -> 0.488 | -0.072 | 42 | 0.071/0.214 | 0.676/0.324 | `tva_shadow_harmful_weakened` |
