# SD-MCTS Supplemental Package

This supplemental package supports the core claims in the paper. It is not a mirror of the authors' working directory. It includes the code entries, result files, figures, and logs needed to verify the paper's central tables and figures.

## Layout

```text
Attachments/
├── code/
│   ├── baselines/          # Black-box world-model baseline used in Beer Game comparisons
│   ├── experiments/        # Beer Game sample efficiency, physical consistency, rollout, and intervention runners
│   ├── scripts/            # Cartpole shared-prior and 1-step evaluation runners
│   ├── src/
│   │   ├── agent/          # Policy and value networks used by the MCTS experiments
│   │   ├── env/            # Beer Game environments
│   │   ├── planning/       # MCTS planner
│   │   └── world_model/    # SD world models for Beer Game, Cartpole, and Hopper-Hop
│   └── tdmpc2_sd_patch/    # TD-MPC2 overlay files for the Hopper-Hop SD mainline
├── data/                   # Artifacts supporting the paper tables and figures
├── requirements.txt
└── README.md
```

## Included Evidence

| Paper item | Supporting material |
| --- | --- |
| Table 1, Table 4, Figure 3: Beer Game sample efficiency | `data/beer_game/`, `code/experiments/run_comparison_revised.py` |
| Figure 2(a): physical consistency | `data/physics_consistency/results.json`, `code/experiments/run_internal_physics_consistency.py` |
| Figure 2(b): multi-step rollout RMSE | `data/multi_step_rollout/`, `code/experiments/run_multi_step_rollout.py` |
| Table 2, Figure 4: structural interventions | `data/interventions/`, `code/experiments/run_causal_intervention.py` |
| Table 1 and Table 5: Cartpole | `data/cartpole/`, `code/scripts/run_cartpole_sharedprior_paper.py`, `code/scripts/run_cartpole_onestep_paper.py` |
| Hopper-Hop mechanism-oriented SD mainline | `data/hopper/mechanism_oriented_200k/`, `code/src/world_model/dmcontrol_hopper_sd_model.py`, `code/tdmpc2_sd_patch/` |

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cartpole and Hopper require `dm-control` and MuJoCo. Hopper additionally requires installing the public TD-MPC2 codebase and overlaying `code/tdmpc2_sd_patch/` onto the corresponding TD-MPC2 package files.

## Verification Commands

Run commands from `Attachments/code`. Result paths below are relative to `Attachments/`.

Beer Game sample-efficiency results support Table 1, Table 4, and Figure 3.

```bash
python experiments/run_comparison_revised.py --budgets 1000 3000 10000 --seeds 3 --eval-episodes 10
```

Stored results:

```text
data/beer_game/sdmcts_rerun_results.json
data/beer_game/dreamerv3_ppo_historical_results.json
data/beer_game/table1_table4_merged_results.json
data/beer_game/table1_sample_efficiency.tex
data/beer_game/figure3_sample_efficiency.png
```

Physical-consistency results support Figure 2(a).

```bash
python experiments/run_internal_physics_consistency.py --n-seeds 3 --eval-episodes 10
```

Stored results:

```text
data/physics_consistency/results.json
```

Multi-step rollout results support Figure 2(b).

```bash
python experiments/run_multi_step_rollout.py
```

Stored results:

```text
data/multi_step_rollout/multi_step_results.json
data/multi_step_rollout/error_curves.csv
data/multi_step_rollout/figure2b_multi_step_rollout_rmse.png
```

Structural-intervention results support Table 2 and Figure 4.

```bash
python experiments/run_causal_intervention.py --seeds 3 --eval-episodes 10
```

Stored results:

```text
data/interventions/results.json
data/interventions/figure4_structural_intervention_cost.png
```

Cartpole results support Table 1 and Table 5. The full shared-prior rerun is compute heavier:

```bash
python scripts/run_cartpole_sharedprior_paper.py --mode serial
```

Stored results:

```text
data/cartpole/cartpole_sharedprior_summary.json
data/cartpole/cartpole_onestep_summary.json
```

The 1-step Cartpole ablation in the original run used shared-prior checkpoints. Checkpoints are omitted from this review package, and the reported ablation summary is included as JSON.

Hopper-Hop is supported by the latest completed mechanism-oriented SD mainline logs at the 200k checkpoint.

To rerun Hopper-Hop, install the public TD-MPC2 codebase, overlay the matching files from `code/tdmpc2_sd_patch/` onto that TD-MPC2 package, and keep `Attachments/code` on `PYTHONPATH` so that `src/world_model/dmcontrol_hopper_sd_model.py` is importable. The SD mainline uses `world_model_kind=sd_hopper`, online SD supervision enabled, planning-weighted SD supervision disabled, and the mechanism-oriented Hopper model in `code/src/world_model/dmcontrol_hopper_sd_model.py`.

Example launch form after applying the overlay:

```bash
PYTHONPATH=<Attachments/code>:$PYTHONPATH python tdmpc2/train.py \
  task=hopper-hop world_model_kind=sd_hopper obs=state steps=200500 seed=1 \
  sd_online_supervision_enabled=true sd_online_planning_weight_enabled=false \
  num_samples=256 num_elites=32 num_pi_trajs=24 iterations=6 \
  eval_freq=25000 checkpoint_freq=25000 save_csv=true save_agent=true
```

Stored results:

```text
data/hopper/mechanism_oriented_200k/hopper_mechanism_oriented_summary.csv
data/hopper/mechanism_oriented_200k/hopper_mechanism_oriented_final_200k.json
data/hopper/mechanism_oriented_200k/hopper_public_baseline_comparison.csv
data/hopper/mechanism_oriented_200k/seed1/run_0_100k/eval.csv
data/hopper/mechanism_oriented_200k/seed1/run_100k_200k/eval.csv
```

## Omitted From This Supplemental Package

Exploratory scripts, bytecode caches, unrelated environments, duplicate plots, large model checkpoints, full third-party baseline repositories, and optional public-baseline training packages are intentionally omitted. Public official baseline scores for DreamerV3 and TD-MPC2 remain cited in the paper; compact aggregate JSON files are included where they support the paper tables.
