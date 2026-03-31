# Demand Transfer Code Readout (Requested Files)

Notes:
- The requested paths `/home/data/round/code/sd_world_model_rl/run_beer_game_v2.py`,
  `/home/data/round/code/sd_world_model_rl/run_ablation_principled.py`,
  and `/home/data/round/code/sd_world_model_rl/run_demand_transfer.py`
  do not exist in this repo.
- The closest NeurIPS experiment files are:
  - `docs/neurips/experiments/run_beer_game_v2.py`
  - `docs/neurips/experiments/run_ablation_principled.py`
  - `docs/neurips/experiments/run_demand_transfer.py`
- The requested `envs/beer_game_env_flat.py` does not exist. The env is in
  `src/env/beer_game_env.py` (BeerGameEnv + BeerGameEnvFlat).

---

## 1) docs/neurips/experiments/run_beer_game_v2.py

### Import statements (lines 11-32)
```python
11	import argparse
12	import numpy as np
13	import torch
14	import torch.nn.functional as F
15	import torch.optim as optim
16	from pathlib import Path
17	from typing import List, Tuple, Callable, Optional
18	from dataclasses import dataclass
19	import json
20	from scipy import stats
22	import sys
23	sys.path.insert(0, str(Path(__file__).parent.parent))
25	from src.env.beer_game_env import BeerGameEnv, BeerGameEnvFlat
26	from src.world_model.beer_game_sd_model import BeerGameSDModel
27	from src.agent import PolicyNetwork, ValueNetwork
28	from src.planning import mcts_search
29	from src.baselines.beer_game_baselines import (
30	    BaseStockPolicy, AnchoringAdjustmentPolicy,
31	    OracleMPCPolicy, ConstantOrderPolicy, ReactivePolicy, StermanPolicy,
32	)
```

### Function definitions
- `train_sd_mcts` signature (lines 104-111)
```python
104	def train_sd_mcts(
105	    env: BeerGameEnvFlat,
106	    env_steps: int = 5000,
107	    gamma: float = 0.99,
108	    device: str = 'cpu',
109	    verbose: bool = True,
110	    world_model_kwargs: Optional[dict] = None,
111	) -> Tuple[BeerGameSDModel, PolicyNetwork, ValueNetwork, List[float]]:
```

- `compute_statistics` signature (lines 258-277)
```python
258	def compute_statistics(data: List[float], name: str = "") -> dict:
259	    """Compute comprehensive statistics for a data array."""
260	    arr = np.array(data)
261	    n = len(arr)
262	    mean = np.mean(arr)
263	    std = np.std(arr, ddof=1) if n > 1 else 0.0
264	    se = std / np.sqrt(n) if n > 0 else 0.0
265	    ci_low = mean - 1.96 * se
266	    ci_high = mean + 1.96 * se
267	    median = np.median(arr)
268	    return {
269	        'mean': mean,
270	        'std': std,
271	        'se': se,
272	        'ci_95_low': ci_low,
273	        'ci_95_high': ci_high,
274	        'median': median,
275	        'n': n,
276	        'raw': arr.tolist(),
277	    }
```

- `evaluate_sd_mcts` signature (lines 280-289)
```python
280	def evaluate_sd_mcts(
281	    env: BeerGameEnv,
282	    world_model: BeerGameSDModel,
283	    policy_net: PolicyNetwork,
284	    value_net: ValueNetwork,
285	    n_episodes: int = 10,
286	    gamma: float = 0.99,
287	    device: str = 'cpu',
288	    num_simulations: int = 200,
289	) -> dict:
```

### Helper class definitions
- `ReplayBuffer` (lines 44-59)
```python
44	class ReplayBuffer:
45	    def __init__(self, capacity: int = 10000):
46	        self.buffer = []
47	        self.capacity = capacity
48	
49	    def add(self, transition: Transition):
50	        if len(self.buffer) >= self.capacity:
51	            self.buffer.pop(0)
52	        self.buffer.append(transition)
```

- `NormalizedValueNet` (lines 88-101)
```python
88	class NormalizedValueNet(torch.nn.Module):
89	    """Wrapper that denormalizes value network output for MCTS compatibility."""
90	    def __init__(self, value_net: ValueNetwork, get_stats_fn):
91	        super().__init__()
92	        self.value_net = value_net
93	        self.get_stats_fn = get_stats_fn  # Returns (mean, std)
94	
95	    def forward(self, states: torch.Tensor) -> torch.Tensor:
96	        normalized_value = self.value_net(states)
97	        mean, std = self.get_stats_fn()
98	        return normalized_value * std + mean  # Denormalize for MCTS
```

### Hyperparameter values (epsilon, buffer_capacity, batch_size, num_updates)
- Epsilon, random steps, sims (lines 149-152)
```python
149	    # Single-phase training (adapted for 81-action space)
150	    random_steps = 2000  # Longer exploration for larger action space
151	    epsilon = 0.3  # Higher exploration for 81 actions
152	    num_sims = 200  # More simulations for better action selection
```

- Buffer capacity (line 133)
```python
133	    buffer = ReplayBuffer(capacity=5000)
```

- World model updates per episode + batch size 32 (lines 210-214)
```python
210	        # World model training (5 updates per episode)
211	        for _ in range(5):
212	            if len(buffer) >= 32:
213	                batch = buffer.sample(32)
214	                states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
```

---

## 2) docs/neurips/experiments/run_ablation_principled.py

### Import statements (lines 15-52)
```python
15	import argparse
16	import copy
17	import json
18	import os
19	import time
22	from dataclasses import dataclass, field
23	from datetime import datetime
24	from pathlib import Path
25	from typing import Any, Dict, List, Optional, Tuple
26	import sys
28	import numpy as np
29	import torch
30	import torch.nn.functional as F
31	import torch.optim as optim
33	SCRIPT_DIR = Path(__file__).resolve().parent
34	NEURIPS_DIR = SCRIPT_DIR.parent
35	IJOC_DIR = NEURIPS_DIR.parent / "ijoc"
36	sys.path.insert(0, str(IJOC_DIR))
37	sys.path.insert(0, str(NEURIPS_DIR))
39	from src.env.beer_game_env import BeerGameEnvFlat
40	from src.world_model.beer_game_sd_model import BeerGameSDModel
41	from src.agent.networks import ValueNetwork
42	from src.planning.mcts import mcts_search
44	from experiments.run_beer_game_v2 import (
45	    BeerGameSDModelWrapper,
46	    make_beer_game_reward_fn,
47	    NormalizedValueNet,
48	    ReplayBuffer,
49	    Transition,
50	)
51	from baselines.blackbox_world_model import BlackBoxWorldModel, train_blackbox_model
```

### Hyperparameter values (epsilon, buffer_capacity, batch_size, num_updates)
- Buffer capacity + epsilon + random_steps (lines 214-226)
```python
214	    buffer = ReplayBuffer(capacity=5000)
215	    reward_fn = make_beer_game_reward_fn(h=env.h, p=env.p, n_echelons=env.n_echelons)
217	    return_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
222	    normalized_vnet = NormalizedValueNet(value_net, get_return_stats) if value_net else None
223	    use_mcts = variant_config["use_mcts"]
224	    gamma = 0.99
225	    epsilon = 0.3
226	    random_steps = min(2000, max(500, sample_budget // 3))
```

- World model updates per episode + batch size 32 (lines 285-289)
```python
285	        # Train world model
286	        if len(buffer) >= 32:
287	            for _ in range(5):
288	                batch = buffer.sample(32)
289	                states_t = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
```

---

## 3) src/env/beer_game_env.py (BeerGameEnv + BeerGameEnvFlat)

### BeerGameEnv.__init__ demand-related parameters (lines 54-71)
```python
54	    def __init__(
55	        self,
56	        n_echelons: int = 4,
57	        tau_info: int = 2,              # Information delay (orders)
58	        tau_ship: int = 2,              # Shipping delay (goods)
59	        initial_inventory: int = 12,    # Initial inventory per echelon
60	        initial_pipeline: int = 4,      # Initial orders in pipeline
61	        h: float = 0.5,                 # Holding cost per unit per period
62	        p: float = 1.0,                 # Backlog penalty per unit per period
63	        O_max: int = 16,                # Maximum order quantity
64	        order_levels: int = 9,          # Discretization: 0, 2, 4, ..., O_max
65	        T: int = 36,                    # Episode length (classic Beer Game)
66	        demand_low: int = 4,            # Initial demand level
67	        demand_high: int = 8,           # Demand after step change
68	        demand_step_time: int = 5,      # When demand steps up
69	        demand_std: float = 0.0,        # Demand noise std (0 = deterministic)
70	        production_capacity: int = 20,  # Factory max production per period
71	        seed: Optional[int] = None,
```

### Demand pattern mechanism (_get_customer_demand) (lines 156-170)
```python
156	    def _get_customer_demand(self) -> int:
157	        """Get customer demand for current period.
158	
159	        Supports deterministic step function (classic Beer Game) or
160	        stochastic demand with Gaussian noise around base level.
161	        """
162	        if self.t < self.demand_step_time:
163	            base_demand = self.demand_low
164	        else:
165	            base_demand = self.demand_high
166	
167	        if self.demand_std > 0:
168	            noise = self.rng.normal(0, self.demand_std)
169	            return max(0, int(round(base_demand + noise)))
170	        return base_demand
```

### BeerGameEnvFlat.__init__ signature (lines 336-349)
```python
336	class BeerGameEnvFlat(BeerGameEnv):
346	    def __init__(self, order_levels: int = 5, **kwargs):
347	        # First init parent with MultiDiscrete
348	        kwargs['order_levels'] = order_levels
349	        super().__init__(**kwargs)
```

---

## 4) docs/neurips/experiments/run_demand_transfer.py

### Import statements (lines 10-24)
```python
10	import os
11	import sys
12	import json
13	import numpy as np
14	import torch
15	from typing import Dict, List
16	from datetime import datetime
17	from pathlib import Path
19	# Add project root to path
20	sys.path.insert(0, str(Path(__file__).parent.parent.parent))
22	from src.env.beer_game_env import BeerGameEnv
23	from demand_patterns import create_demand_pattern
```

### Demand-pattern env wrapper (lines 38-52)
```python
38	class DemandPatternEnv(BeerGameEnv):
39	    """Beer Game environment with pluggable demand patterns."""
41	    def __init__(self, demand_pattern, **kwargs):
42	        self.demand_pattern = demand_pattern
43	        super().__init__(**kwargs)
45	    def _get_customer_demand(self) -> int:
46	        """Override to use pluggable demand pattern."""
47	        return self.demand_pattern.get_demand()
49	    def reset(self, *, seed=None, options=None):
50	        self.demand_pattern.reset()
51	        return super().reset(seed=seed, options=options)
```

### CONFIG hyperparameters (lines 26-35)
```python
26	CONFIG = {
27	    "experiment_name": "demand_transfer",
28	    "train_pattern": "step",
29	    "test_patterns": ["step", "sinusoidal", "random_walk", "bursty"],
30	    "methods": ["sd_mcts", "blackbox_mcts"],  # DreamerV3 requires separate handling
31	    "train_budget": 30000,
32	    "n_seeds": 5,
33	    "eval_episodes": 30,
34	    "n_echelons": 4,
35	}
```

### Training update frequency + batch size (lines 170-174)
```python
170	            # Train networks (every 10 steps)
171	            if steps % 10 == 0 and len(replay_buffer) >= 64:
172	                batch = [replay_buffer[i] for i in
173	                        np.random.choice(len(replay_buffer), 64, replace=False)]
174	
```
