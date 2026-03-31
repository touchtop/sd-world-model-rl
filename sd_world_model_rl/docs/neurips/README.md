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

---

## Experiment Reproduction Commands

### Table 1: Sample Efficiency (Main Result)

**Current paper data source**: `results/main_revised/20260331_full_merged/results.json`

```bash
cd /home/data/round/code/sd_world_model_rl/docs/neurips

# Revised runner used for the current paper Table 1 SD-MCTS line
python experiments/run_comparison_revised.py \
    --budgets 1000 3000 10000 \
    --seeds 3 \
    --eval-episodes 10

# Quick test (single budget, single seed)
python experiments/run_comparison_revised.py \
    --experiment main \
    --budgets 1000 \
    --seeds 1
```

**Configuration**:
- Environment: Beer Game 4-echelon, T=36 steps, 81 discrete actions
- Methods in the current paper table: `sd_mcts`, `dreamerv3`, `ppo`
- Evaluation: 10 episodes with seeds 1000-1009
- Note: `results/main/20260208_201101/results.json` is retained as the historical DreamerV3/PPO source used when merging the revised Table 1 input.

**Estimated runtime** (3 seeds):
| Budget | Time |
|--------|------|
| 1K | ~5 min |
| 3K | ~40 min |
| 10K | ~3 hrs |

### Table 2: Ablation Study

**Data source**: `results/ablation/20260212_102554/results.json`

```bash
cd /home/data/round/code/sd_world_model_rl/docs/neurips

# Core scenarios (5 variants × 3 scenarios × 1 seed = 15 runs)
python experiments/run_ablation_principled.py \
    --suite core \
    --seeds 1 \
    --eval-episodes 10

# Multi-seed for statistical significance (recommended)
python experiments/run_ablation_principled.py \
    --suite core \
    --seeds 5 \
    --eval-episodes 10

# View experiment design
python experiments/run_ablation_principled.py --describe
```

**Variants**:
| Variant | Model | VNet | MCTS | Tests |
|---------|-------|------|------|-------|
| `sd_mcts` | SD | ✓ | ✓ | Baseline |
| `blackbox_mlp` | MLP | ✓ | ✓ | Structural prior |
| `no_vnet` | SD | ✗ | ✓ | Terminal value |
| `greedy_1step` | SD | ✗ | ✗ | Multi-step planning |
| `wrong_prior` | SD* | ✓ | ✓ | Prior correctness |

**Scenarios**:
| Scenario | Budget | T | Description |
|----------|--------|---|-------------|
| `standard` | 3000 | 36 | Standard setup |
| `low_sample_shift` | 1000 | 36 | Low data + OOD |
| `long_horizon_delay` | 10000 | 100 | Long horizon |

### Results Directory

```
docs/neurips/results/
├── main_revised/
│   ├── 20260331_full_sdmcts/ # Revised SD-MCTS rerun
│   └── 20260331_full_merged/ # Current Table 1 merged data
├── main/
│   └── 20260208_201101/      # Historical DreamerV3/PPO source
│       ├── config.json
│       └── results.json
├── ablation/
│   └── 20260212_102554/      # Table 2 final data
│       └── results.json
└── figures/
    ├── table1_sample_efficiency.tex
    └── table2_ablation_study.tex
```
