# v8 formal 100k rule-prior on/off diagnostic

Date: 2026-05-09

## Purpose

This diagnostic evaluates the final 100k checkpoints from `hopper_v8formal_100k_v8phaseany_confidence_empty_seed123_20260509_r2` under three inference conditions:

- `rule_off_cycle019lib`: explicit rule prior disabled.
- `rule_on_cycle019lib`: cycle019 confidence-active rules enabled; this is the rule library closest to the final 95k-100k training interval.
- `rule_on_cycle020lib`: final cycle020 confidence-active rules enabled; this is post-hoc relative to the 100k training result.

All evaluations use the Hopper operation-guide convention: `MUJOCO_GL=glfw`, `eval_episodes=10`, `eval_action_mode=eval`, TD-MPC2 checkpoint loading from `final.full.pt`, and `penalty_only` rule-prior mode with `opposite_penalty=0.01`, `preferred_cap=0.70`, `cap_penalty=0.01`.

## Results

| condition | seed1 | seed2 | seed3 | mean |
| --- | ---: | ---: | ---: | ---: |
| rule off | 9.3539 | 0.0452 | 1.5603 | 3.6531 |
| rule on, cycle019 | 6.0831 | 0.0407 | 1.5503 | 2.5580 |
| rule on, cycle020 | 3.2839 | 0.0407 | 1.5502 | 1.6249 |

Episode maxima:

| condition | seed1 max | seed2 max | seed3 max |
| --- | ---: | ---: | ---: |
| rule off | 28.8695 | 0.2967 | 9.1915 |
| rule on, cycle019 | 23.2419 | 0.4037 | 9.1366 |
| rule on, cycle020 | 23.2419 | 0.4037 | 9.1367 |

Rule counts:

| condition | rules installed |
| --- | ---: |
| rule off | 0 |
| rule on, cycle019 | 30 |
| rule on, cycle020 | 31 |

## Interpretation

The 100k low-score result cannot be attributed only to the explicit eval-time rule prior. When the rule prior is disabled, seed2 and seed3 remain low, indicating that at least part of the poor result is already present in the learned checkpoint.

The rule prior is nevertheless not neutral. On seed1, disabling rules yields `9.3539`, while cycle019 rules reduce it to `6.0831`, and cycle020 rules reduce it further to `3.2839`. Thus the current rule prior has a measurable negative inference-time effect in at least one seed.

Cycle020 adds one active rule relative to cycle019 under this eval setting. Its post-hoc final-library effect does not improve the final checkpoint behavior; it further lowers seed1 and leaves seed2/seed3 essentially unchanged.

The diagnostic therefore supports a mixed failure account: the training trajectory/checkpoint quality is already weak in two seeds, and the explicit rule prior can additionally suppress useful behavior in the seed where the checkpoint has some recoverable performance.

