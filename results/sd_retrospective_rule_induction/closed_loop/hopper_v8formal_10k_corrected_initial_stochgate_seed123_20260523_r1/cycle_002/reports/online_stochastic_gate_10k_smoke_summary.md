# Online stochastic gate 10k smoke summary

Date: 2026-05-23

Run:

```text
results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_10k_corrected_initial_stochgate_seed123_20260523_r1
```

Protocol:

```text
seeds: 1,2,3
cycles: 2
cycle_steps: 5000
total budget: 10000
formal eval episodes: 10 per checkpoint
initial rule library: library_corrected_label_initial_v1.json
initial active rules: 12
rule prior from cycle 1: enabled
stochastic rule gate: enabled
p_apply = 0.10 + 0.85 * confidence
stochastic eval: disabled; eval uses expected multiplier
TVA terminal prior: disabled
```

Main observations:

1. Online integration was successful. All three seeds loaded the corrected-label initial rule library, reported `rules=12`, and wrote stochastic gate metrics.
2. Cycle 001 completed the full closed-loop path: replay export, trajectory selection, SD fitting, counterfactual attribution, rule induction, rule validation, and shared library update.
3. Cycle 002 initialized SD from the cycle 001 checkpoint (`init_loaded_tensors=30`) and completed the same closed-loop path.
4. No new rule was promoted to `confidence_active` in either cycle. The final shared library contains the 12 initial active rules plus non-active candidates or rolling-validated rules.
5. Online rule coverage was sparse in the first 10k from scratch. Cycle 001 had one matched rule across the three seeds, and cycle 002 had one matched rule per seed.
6. The stochastic gate behaved as designed where rules were matched. Cycle 002 produced 3 matched rules, 1 applied rule, and 2 dropped rules, with matched-rule apply probability `0.576`.

Rule prior usage:

| Cycle | Seed | Matched | Applied | Dropped | Mean apply probability on hits |
|---|---:|---:|---:|---:|---:|
| 001 | 1 | 0 | 0 | 0 | 0.000 |
| 001 | 2 | 0 | 0 | 0 | 0.000 |
| 001 | 3 | 1 | 1 | 0 | 0.576 |
| 002 | 1 | 1 | 0 | 1 | 0.576 |
| 002 | 2 | 1 | 1 | 0 | 0.576 |
| 002 | 3 | 1 | 0 | 1 | 0.576 |

Formal eval returns from exported replay summaries:

| Cycle | Seed | Episodes | Mean return | Max return |
|---|---:|---:|---:|---:|
| 001 | 1 | 10 | 0.150 | 1.501 |
| 001 | 2 | 10 | 0.005 | 0.047 |
| 001 | 3 | 10 | 0.001 | 0.005 |
| 002 | 1 | 20 | 0.148 | 1.501 |
| 002 | 2 | 20 | 0.009 | 0.141 |
| 002 | 3 | 20 | 0.001 | 0.006 |

SD fitting:

| Cycle | Transitions | Train | Validation | Validation loss | Init |
|---|---:|---:|---:|---:|---|
| 001 | 17465 | 15469 | 1996 | 0.695 | none |
| 002 | 41916 | 37924 | 3992 | 0.530 | cycle 001 SD |

Interpretation:

The smoke test validates the engineering path for corrected-label initial rules plus confidence-calibrated stochastic gating. It does not validate performance, because the initial rule library barely covers early from-scratch behavior within 10k steps. The immediate implication is that the mechanism can be used safely in a longer run, but the current high-quality initial rules are mainly mature-policy rules rather than early-entry exploration rules.
