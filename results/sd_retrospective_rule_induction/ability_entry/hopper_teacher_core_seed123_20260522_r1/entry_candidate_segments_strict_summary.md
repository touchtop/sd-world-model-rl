# Hopper Strict Entry Segments

Source: `results/sd_retrospective_rule_induction/ability_entry/hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json`

| Mode | Raw | Selected | Unique | Roles | Offsets | Thresholds |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `no_pre_entry` | 568 | 269 | 269 | `{'matched_failure': 152, 'entry_success': 117}` | `{'0': 120, '1': 149}` | `{'1.0': 66, '10.0': 94, '100.0': 63, '50.0': 46}` |
| `entry_cycle_only` | 568 | 135 | 135 | `{'matched_failure': 82, 'entry_success': 53}` | `{'0': 135}` | `{'1.0': 38, '10.0': 39, '100.0': 31, '50.0': 27}` |

## Label Policy

严格数据集移除同一窗口在不同事件中产生 success/failure 角色冲突的样本，并对同一窗口跨阈值重复出现的情况只保留一个最高阈值实例。`entry_cycle_only` 仅保留入口当周期；`no_pre_entry` 保留入口当周期和入口后一周期。