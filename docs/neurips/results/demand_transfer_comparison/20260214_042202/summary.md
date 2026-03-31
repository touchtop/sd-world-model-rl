============================================================
SUMMARY: SD-MCTS vs BlackBox-MCTS Demand Transfer
============================================================

SD-MCTS budget: 15000
BlackBox budget: 30000
Seeds: 5
Eval episodes: 20


## Table 1: Absolute Costs (Mean ± SE)

| Method | Budget | Step | Sinusoidal | Random Walk | Bursty |
|--------|--------|------|------------|-------------|--------|
| sd_mcts | 15000 | 650±13 | 872±40 | 1405±81 | 1076±32 |
| blackbox_mcts | 30000 | 638±12 | 916±21 | 1449±62 | 1130±36 |


## Table 2: Transfer Ratio (Test Cost / Train Cost)

| Method | Step→Sinus | Step→RandWalk | Step→Bursty | Avg OOD |
|--------|------------|---------------|-------------|---------|
| sd_mcts | 1.34x | 2.16x | 1.66x | **1.72x** |
| blackbox_mcts | 1.44x | 2.27x | 1.77x | **1.83x** |


## Table 3: Training Performance (Final Avg Reward)

| Method | Budget | Final Reward (Avg±SE) |
|--------|--------|----------------------|
| sd_mcts | 15000 | -946.9±17.7 |
| blackbox_mcts | 30000 | -942.0±23.6 |


## Key Findings

- SD-MCTS average OOD transfer ratio: 1.72x
- BlackBox average OOD transfer ratio: 1.83x
- SD-MCTS generalization advantage: 6.2% less degradation