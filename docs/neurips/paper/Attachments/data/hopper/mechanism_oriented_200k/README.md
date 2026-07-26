# Hopper-Hop Mechanism-Oriented SD Logs

This directory contains compact CSV artifacts for the latest completed mechanism-oriented SD Hopper-Hop mainline at the 200k checkpoint.

The run is split into two phases:

- `run_0_100k`: initial three-seed run from 0 to 100k environment steps.
- `run_100k_200k`: continuation from the 100k checkpoint to the 200k checkpoint.

Summary files:

- `hopper_mechanism_oriented_summary.csv`: per-checkpoint three-seed rewards, mean, and population standard deviation.
- `hopper_mechanism_oriented_final_200k.json`: final 200k aggregate and source files.
- `hopper_public_baseline_comparison.csv`: compact comparison against public TD-MPC2 and DreamerV3 aggregates.

Per-seed directories include `eval.csv`, `eval_full.csv`, `train.csv`, and official evaluation episode summaries. Model checkpoints, rendered traces, and per-episode trace CSV files are omitted from this review package.
