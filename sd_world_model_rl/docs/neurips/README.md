# SD-MCTS NeurIPS Experiments

This directory contains the experimental code for the NeurIPS submission.

## Directory Structure

```
docs/neurips/
├── src/                          # Source code
│   ├── env/                      # Environment implementations
│   │   ├── beer_game_env.py      # Beer Game supply chain
│   │   ├── delayed_collapse_env.py
│   │   └── task_allocation_env.py
│   ├── world_model/              # World model implementations
│   │   ├── beer_game_sd_model.py # SD-based world model
│   │   ├── delayed_collapse_model.py
│   │   └── task_sd_model.py
│   ├── planning/                 # Planning algorithms
│   │   └── mcts.py               # MCTS implementation
│   ├── agent/                    # Agent components
│   │   └── networks.py           # Neural network architectures
│   └── baselines/                # Baseline implementations
│       ├── blackbox_world_model.py  # MLP world model (ablation)
│       ├── sota_adapters.py      # DreamerV3/TD-MPC2/MuZero adapters
│       └── beer_game_baselines.py
│
├── experiments/                  # Experiment scripts
│   ├── run_sample_efficiency.py  # P0: Sample efficiency curves
│   ├── run_physics_violation.py  # P0: Physics violation analysis
│   ├── run_zero_shot_transfer.py # P1: Zero-shot transfer
│   └── baselines/                # Baseline-specific experiments
│
├── results/                      # Experiment outputs
│   ├── sample_efficiency/
│   ├── physics_violation/
│   ├── ood_robustness/
│   ├── zero_shot_transfer/
│   └── ablations/
│
├── figures/                      # Generated figures for paper
│
├── neurips_work_plan.md          # Detailed work plan
├── section_outline_zh.md         # Chinese section outline
└── requirements.txt              # Python dependencies
```

## Experiment Priority

| Priority | Experiment | Script | Purpose |
|----------|------------|--------|---------|
| P0 | Sample Efficiency | `run_sample_efficiency.py` | Core claim: 3-5x fewer samples |
| P0 | Physics Violation | `run_physics_violation.py` | Core claim: 0% violations |
| P1 | Zero-Shot Transfer | `run_zero_shot_transfer.py` | Transfer across topologies |
| P1 | OOD Robustness | `run_ood_robustness.py` | Demand shock resilience |
| P2 | Ablations | `run_ablations.py` | Component contributions |

## Running Experiments

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For SOTA baselines (optional)
pip install stable-baselines3  # For PPO
# DreamerV3: https://github.com/danijar/dreamerv3
# TD-MPC2: https://github.com/nicklashansen/tdmpc2
```

### Quick Start

```bash
# Run sample efficiency experiment
cd docs/neurips
python experiments/run_sample_efficiency.py

# Run physics violation analysis
python experiments/run_physics_violation.py
```

## Baseline Status

| Baseline | Status | Notes |
|----------|--------|-------|
| SD-MCTS | ✅ Ready | From IJOC |
| BlackBox-MCTS | ✅ Ready | Implemented |
| PPO | ✅ Ready | Via stable-baselines3 |
| DreamerV3 | 🔧 Adapter ready | Needs installation |
| TD-MPC2 | 🔧 Adapter ready | Needs installation |
| MuZero | 🔧 Adapter ready | Needs installation |

## Expected Results

Based on IJOC experiments and theoretical analysis:

| Metric | SD-MCTS | BlackBox | DreamerV3 |
|--------|---------|----------|-----------|
| Samples to 90% | ~10K | ~50K | ~100K |
| Physics Violation | 0% | ~15% | ~8% |
| Transfer (3→5 node) | >85% | <50% | <40% |
