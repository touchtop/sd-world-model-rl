# Hopper v7.1a Explainability Long Resume Summary

## Scope

This run extends the explainability-rule closed loop from the short `100k -> 110k` validation to a medium-budget `110k -> 200k` observation. The purpose is to inspect whether the same rule mechanism can remain stable and produce adaptation signals over a longer interval, not to claim Hopper task success at this budget.

## Training Run

```text
start checkpoint: logs/hopper-hop/1/tdmpc2_v71a_explain_short_5k_seed1_20260506_cycle022_target110021/models/final.full.pt
cycles: 23..40
cycle steps: 5000
final checkpoint: logs/hopper-hop/1/tdmpc2_v71a_explain_long_5k_seed1_20260506_cycle040_target200039/models/final.full.pt
final SD checkpoint: results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/sd_explainers/sd_explainer_cycle_040.pt
final rule library: results/sd_retrospective_rule_induction/closed_loop/hopper_v71a_explain_long_110k_to200k_seed1_20260506/cycle_040/rules/library_closed_loop_cycle_040.json
```

## Cycle Summary

```text
cycle target quick_eval train_nonzero train_mean train_max active rolling candidates cycle_rules
023   115022 0.000      3/10          21.8       86.8      12     0       2          1
024   120023 0.000      3/10          27.6       94.2      12     0       3          2
025   125024 0.000      3/10          28.4       111.4     12     0       4          3
026   130025 0.000      3/10          39.1       137.6     12     0       4          3
027   135026 0.000      3/10          40.5       154.0     12     1       4          4
028   140027 0.000      3/10          38.5       145.4     12     1       4          3
029   145028 0.000      3/10          45.6       155.7     12     1       6          3
030   150029 0.000      4/10          58.0       155.9     12     2       7          5
031   155030 0.000      3/10          53.8       189.9     12     2       7          5
032   160031 58.571     3/10          53.0       194.9     12     2       7          3
033   165032 0.000      4/10          77.0       204.9     12     2       7          2
034   170033 0.000      4/10          70.4       189.8     12     2       8          4
035   175034 0.000      5/10          99.7       203.8     12     3       7          5
036   180035 0.000      5/10          81.9       206.7     12     3       9          5
037   185036 0.000      4/10          79.0       200.9     12     3       10         3
038   190037 74.194     4/10          88.5       229.2     12     3       12         5
039   195038 0.000      6/10          115.3      232.6     11     3       14         4
040   200039 0.000      4/10          89.7       237.4     11     3       14         3
```

No cycle added a new online rule. The updater repeatedly retained candidate evidence but did not promote it under the online thresholds.

## Final 10-Episode Comparison

```text
rule_off:
  mean = 88.514
  std  = 108.492
  list = 219.547;0.000;0.000;231.711;212.722;0.000;0.000;0.000;221.155;0.000

explain_on:
  mean = 89.203
  std  = 109.324
  list = 223.137;0.000;0.000;232.392;214.622;0.000;0.000;0.000;221.880;0.000
```

The mean difference is `+0.690`. The rule prior slightly increases the same successful episodes but does not change which initial states recover. This supports the interpretation that the current rule set is active and not obviously harmful at this checkpoint, but it has not yet changed the episode-level recovery basin.

## Interpretation

The longer run strengthens the training-side signal: high-reward training segments become more frequent and reach above `230`, and quick eval occasionally becomes nonzero. However, quick eval remains unstable, and final rule-on/off differences are small. The main bottleneck remains rule identification and rule coverage rather than the mechanical ability to inject rules into planning.
