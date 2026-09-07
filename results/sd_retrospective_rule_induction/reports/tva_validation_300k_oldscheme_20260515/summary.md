# TVA Validation Summary: 300k Old-Scheme Hopper

Scope: audit-only TVA validation on cycles 041, 050, and 060. TVA training used 5000 transitions and 10 epochs per cycle, with checkpoint continuation across cycles. Counterfactual audit used stratified 300-record samples.

| cycle | reward-TVA Pearson | reward-TVA Spearman | bad TVA | good TVA | bad cf adv | good cf adv | bad top3 | good top3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 041 | 0.860 | 0.719 | -16.65 | 58.42 | 9.38 | 13.63 | 0.248 | 0.412 |
| 050 | 0.972 | 0.717 | -21.56 | 66.40 | 14.89 | 10.66 | 0.170 | 0.360 |
| 060 | 0.949 | 0.654 | -25.03 | 78.20 | 10.06 | 14.16 | 0.210 | 0.400 |

## Interpretation

- TVA carries a strong episode-level value signal: all three audited cycles show high positive reward-TVA Pearson correlation.
- Label separation is clear: good trajectories have much higher TVA means than bad trajectories in every audited cycle.
- TVA counterfactual action ranking is informative but not yet sufficient as a rule updater by itself: good-label actual actions have better top-3 rates than bad-label actual actions, but TVA still reports sizeable alternative-action advantages even in good records.
- The next safe step is shadow update, not online intervention: use TVA scores to propose confidence changes beside the current rule library and compare them with existing confidence updates before using them in planner priors.
