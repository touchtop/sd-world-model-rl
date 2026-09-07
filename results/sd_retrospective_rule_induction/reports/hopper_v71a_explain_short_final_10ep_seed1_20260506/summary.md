# Hopper v7.1a Explainability Short Resume Final Eval

Checkpoint:

```text
logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt
```

Rule library:

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_short_100k_to110k_seed1_20260506/cycle_022/rules/library_closed_loop_cycle_022.json
```

| mode | episodes | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| rule_off | 10 | 8.822 | 16.000 | 0.000 | 51.463 |
| explain_on | 10 | 15.955 | 26.914 | 0.000 | 80.774 |

Episode rewards:

```text
rule_off:
0.000;0.000;0.000;21.066;15.689;0.000;0.000;0.000;51.463;0.000

explain_on:
0.000;0.000;0.000;80.774;48.390;0.000;0.000;0.000;30.383;0.000

explain_on - rule_off:
0.000;0.000;0.000;+59.707;+32.701;0.000;0.000;0.000;-21.080;0.000
```

Interpretation:

```text
The explainability-rule prior showed a small positive mean effect on this final
10-episode checkpoint comparison, but the absolute performance remained low and
the effect was concentrated in a few initial states. This should be treated as a
weak compatibility signal after short resumed training, not as task-level success.
```
