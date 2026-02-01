# SD-MCTS: Neuro-Symbolic Planning for Operations Research

This repository contains the code and data for reproducing the experiments in:

**"SD-MCTS: Structure-Preserving World Models with Terminal Value Learning for Operations Research"**

Submitted to INFORMS Journal on Computing (IJOC)

## Quick Start

```bash
# Create environment
conda create -n sdmcts python=3.10
conda activate sdmcts
pip install -r requirements.txt

# Reproduce main results
python experiments/run_m1.py --seeds 50                    # Scenario 1-2
python experiments/run_beer_game_v2.py --horizon-wall      # Experiment 3A
python experiments/run_beer_game_v2.py --resilience        # Experiment 3B
```

## Repository Structure

```
sd-mcts-ijoc/
├── src/                   # Source code
│   ├── env/               # Environment implementations
│   ├── world_model/       # Structured SD world models
│   ├── planning/          # MCTS implementation
│   ├── agent/             # Neural networks (policy, value)
│   └── baselines/         # Domain-specific baselines
├── experiments/           # Experiment scripts
│   └── baselines/         # MPC implementations
├── results/               # Experimental outputs
└── *.md                   # Paper drafts
```

## Reproducing Specific Tables

| Table | Command | Output |
|-------|---------|--------|
| Table 1 (Scenario 1) | `python experiments/run_m1.py --seeds 50` | `results/scenario1/` |
| Table 2c (Horizon Wall) | `python experiments/run_beer_game_v2.py --horizon-wall` | `results/beer_game_v2/horizon_wall/` |
| Table 2d (Resilience) | `python experiments/run_beer_game_v2.py --resilience` | `results/beer_game_v2/resilience/` |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy

See `requirements.txt` for complete dependencies.

## License

MIT License - see LICENSE file.
