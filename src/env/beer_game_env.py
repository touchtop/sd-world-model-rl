"""Beer Game Environment - Four-Echelon Supply Chain Simulation.

The classic MIT Beer Game (Sterman 1989) demonstrates the bullwhip effect:
demand variability amplifies as orders propagate upstream through the supply chain.

Structure: Customer -> Retailer -> Wholesaler -> Distributor -> Factory

Stock-flow dynamics per echelon i:
    I_{i,t+1} = I_{i,t} + Arrivals_{i,t} - Shipments_{i,t}
    B_{i,t+1} = B_{i,t} + Orders_received_{i,t} - Shipments_{i,t}
    Pipeline_{i,t+1} = shift(Pipeline_{i,t}, new_order)

Delays:
    - Information delay: tau_info = 2 periods (order placed -> order received by upstream)
    - Shipping delay: tau_ship = 2 periods (shipment sent -> shipment arrives)

Why SD-MCTS should excel:
1. Stock-flow structure naturally captures inventory conservation laws
2. Value network learns to anticipate bullwhip propagation
3. Multi-echelon coordination emerges from learned terminal cost
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List, Dict


class BeerGameEnv(gym.Env):
    """Four-echelon supply chain (Beer Game) environment.

    Echelons (downstream to upstream):
        0: Retailer (faces customer demand)
        1: Wholesaler
        2: Distributor
        3: Factory (produces goods)

    State representation per echelon (6 variables each):
        - Inventory (I): on-hand stock
        - Backlog (B): unfulfilled orders
        - Pipeline[0]: orders/shipments arriving next period
        - Pipeline[1]: orders/shipments arriving in 2 periods
        - OrderPipeline[0]: orders reaching upstream next period
        - OrderPipeline[1]: orders reaching upstream in 2 periods

    Plus global state:
        - Current timestep (t)
        - Current customer demand (D_t)

    Total state dimension: 6 * 4 + 2 = 26
    """

    ECHELON_NAMES = ['Retailer', 'Wholesaler', 'Distributor', 'Factory']

    def __init__(
        self,
        n_echelons: int = 4,
        tau_info: int = 2,              # Information delay (orders)
        tau_ship: int = 2,              # Shipping delay (goods)
        initial_inventory: int = 12,    # Initial inventory per echelon
        initial_pipeline: int = 4,      # Initial orders in pipeline
        h: float = 0.5,                 # Holding cost per unit per period
        p: float = 1.0,                 # Backlog penalty per unit per period
        O_max: int = 16,                # Maximum order quantity
        order_levels: int = 9,          # Discretization: 0, 2, 4, ..., O_max
        T: int = 36,                    # Episode length (classic Beer Game)
        demand_low: int = 4,            # Initial demand level
        demand_high: int = 8,           # Demand after step change
        demand_step_time: int = 5,      # When demand steps up
        demand_std: float = 0.0,        # Demand noise std (0 = deterministic)
        production_capacity: int = 20,  # Factory max production per period
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n_echelons = n_echelons
        self.tau_info = tau_info
        self.tau_ship = tau_ship
        self.initial_inventory = initial_inventory
        self.initial_pipeline = initial_pipeline
        self.h = h
        self.p = p
        self.O_max = O_max
        self.order_levels = order_levels
        self.T = T
        self.demand_low = demand_low
        self.demand_high = demand_high
        self.demand_step_time = demand_step_time
        self.demand_std = demand_std
        self.production_capacity = production_capacity
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # State space: 6 variables per echelon + 2 global
        # Per echelon: [inventory, backlog, ship_pipeline_0, ship_pipeline_1,
        #               order_pipeline_0, order_pipeline_1]
        state_dim = n_echelons * 6 + 2

        # Bounds for observation space
        max_inventory = 100.0
        max_backlog = 100.0
        max_pipeline = 50.0

        low = np.zeros(state_dim, dtype=np.float32)
        high = np.array(
            [max_inventory, max_backlog, max_pipeline, max_pipeline,
             max_pipeline, max_pipeline] * n_echelons +
            [float(T), float(demand_high * 2)],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: sequential ordering - one action per echelon per substep
        # Or combined: MultiDiscrete for all echelons at once
        # Using combined approach with discrete order levels
        self.action_space = spaces.MultiDiscrete([order_levels] * n_echelons)

        # Pre-compute order quantities for each discrete level
        self.order_quantities = np.linspace(0, O_max, order_levels).astype(int)

        # Initialize state variables
        self._init_state()

    def _init_state(self):
        """Initialize all state variables."""
        # Inventory at each echelon
        self.inventory = np.full(self.n_echelons, self.initial_inventory, dtype=np.float32)

        # Backlog (unfulfilled orders) at each echelon
        self.backlog = np.zeros(self.n_echelons, dtype=np.float32)

        # Incoming shipment pipeline: pipeline[i][j] arrives at echelon i in j+1 periods
        # Shape: (n_echelons, tau_ship)
        self.incoming_pipeline = np.full(
            (self.n_echelons, self.tau_ship),
            self.initial_pipeline,
            dtype=np.float32
        )

        # Outgoing order pipeline (for information delay)
        # Orders placed by echelon i arrive at echelon i+1's backlog after tau_info periods
        self.order_pipeline = np.full(
            (self.n_echelons, self.tau_info),
            self.initial_pipeline,
            dtype=np.float32
        )

        self.t = 0
        self.current_demand = self.demand_low

        # Tracking for metrics
        self.total_cost = 0.0
        self.order_history = [[] for _ in range(self.n_echelons)]
        self.inventory_history = [[] for _ in range(self.n_echelons)]

    def _get_customer_demand(self) -> int:
        """Get customer demand for current period.

        Supports deterministic step function (classic Beer Game) or
        stochastic demand with Gaussian noise around base level.
        """
        if self.t < self.demand_step_time:
            base_demand = self.demand_low
        else:
            base_demand = self.demand_high

        if self.demand_std > 0:
            noise = self.rng.normal(0, self.demand_std)
            return max(0, int(round(base_demand + noise)))
        return base_demand

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        obs = []
        for i in range(self.n_echelons):
            obs.extend([
                self.inventory[i],
                self.backlog[i],
                self.incoming_pipeline[i, 0],  # Arrives next period
                self.incoming_pipeline[i, 1] if self.tau_ship > 1 else 0.0,  # Arrives in 2 periods
                self.order_pipeline[i, 0],  # Arrives upstream next period
                self.order_pipeline[i, 1] if self.tau_info > 1 else 0.0,  # Arrives upstream in 2 periods
            ])
        obs.extend([float(self.t), float(self.current_demand)])
        return np.array(obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step.

        Order of operations (classic Beer Game):
        1. Receive incoming shipments (advance pipeline)
        2. Receive incoming orders (from downstream or customer)
        3. Ship goods to fulfill orders (limited by inventory)
        4. Place orders to upstream echelon
        5. Compute costs
        """
        # Convert discrete actions to order quantities
        orders = np.array([self.order_quantities[a] for a in action], dtype=np.float32)

        # Update demand for this period
        demand = self._get_customer_demand()
        self.current_demand = demand

        # Step 1: Receive incoming shipments (advance pipeline)
        arrivals = self.incoming_pipeline[:, 0].copy()
        self.incoming_pipeline = np.roll(self.incoming_pipeline, -1, axis=1)
        self.incoming_pipeline[:, -1] = 0  # Clear the last slot

        # Add arrivals to inventory
        self.inventory += arrivals

        # Step 2: Receive incoming orders
        # Retailer receives customer demand
        # Other echelons receive orders from downstream (after info delay)
        incoming_orders = np.zeros(self.n_echelons, dtype=np.float32)
        incoming_orders[0] = demand  # Customer demand to retailer

        # Orders from downstream echelons (from order pipeline)
        for i in range(1, self.n_echelons):
            incoming_orders[i] = self.order_pipeline[i - 1, 0]

        # Advance order pipeline
        self.order_pipeline = np.roll(self.order_pipeline, -1, axis=1)
        self.order_pipeline[:, -1] = 0

        # Add incoming orders to backlog
        self.backlog += incoming_orders

        # Step 3: Ship goods (fulfill orders)
        shipments = np.zeros(self.n_echelons, dtype=np.float32)
        for i in range(self.n_echelons):
            # Ship min(inventory, backlog)
            ship_amount = min(self.inventory[i], self.backlog[i])
            shipments[i] = ship_amount
            self.inventory[i] -= ship_amount
            self.backlog[i] -= ship_amount

        # Actually, shipments from echelon i+1 go to echelon i's incoming pipeline
        for i in range(self.n_echelons - 1):
            self.incoming_pipeline[i, -1] = shipments[i + 1]

        # Factory (last echelon) receives from "production"
        # Production = orders placed by factory, limited by capacity
        factory_idx = self.n_echelons - 1
        factory_order = min(orders[factory_idx], self.production_capacity)
        self.incoming_pipeline[factory_idx, -1] = factory_order

        # Step 4: Place orders (go into order pipeline)
        for i in range(self.n_echelons):
            self.order_pipeline[i, -1] = orders[i]

        # Track order history
        for i in range(self.n_echelons):
            self.order_history[i].append(orders[i])
            self.inventory_history[i].append(self.inventory[i])

        # Step 5: Compute cost
        holding_cost = self.h * np.sum(np.maximum(0, self.inventory))
        backlog_cost = self.p * np.sum(self.backlog)
        period_cost = holding_cost + backlog_cost
        self.total_cost += period_cost

        # Reward is negative cost
        reward = -period_cost

        self.t += 1
        self.current_demand = self._get_customer_demand()
        terminated = self.t >= self.T

        info = {
            'inventory': self.inventory.copy(),
            'backlog': self.backlog.copy(),
            'orders': orders.copy(),
            'shipments': shipments.copy(),
            'arrivals': arrivals.copy(),
            'period_cost': period_cost,
            'total_cost': self.total_cost,
            'customer_demand': demand,
        }

        return self._get_obs(), reward, terminated, False, info

    def compute_bullwhip_ratio(self) -> np.ndarray:
        """Compute bullwhip ratio for each echelon.

        Following Lee et al. (1997) "The Bullwhip Effect in Supply Chains":
            BWE_i = Var(O_i) / Var(D_i)
        where O_i is orders placed by echelon i, D_i is demand received by echelon i.

        For retailer: D_0 = customer demand
        For upstream echelons: D_i = orders from echelon i-1

        Ratio > 1 indicates demand amplification (bullwhip effect).
        """
        if len(self.order_history[0]) < 2:
            return np.ones(self.n_echelons)

        # Customer demand sequence (retailer's incoming demand)
        demand_sequence = np.array([
            self.demand_low if t < self.demand_step_time else self.demand_high
            for t in range(len(self.order_history[0]))
        ], dtype=np.float32)

        bullwhip = np.zeros(self.n_echelons)

        # Retailer: compare orders to customer demand
        retailer_order_var = np.var(self.order_history[0])
        demand_var = np.var(demand_sequence) + 1e-6
        bullwhip[0] = retailer_order_var / demand_var

        # Upstream echelons: compare orders to downstream's orders (their demand)
        for i in range(1, self.n_echelons):
            upstream_order_var = np.var(self.order_history[i])
            downstream_order_var = np.var(self.order_history[i - 1]) + 1e-6
            bullwhip[i] = upstream_order_var / downstream_order_var

        return bullwhip

    def get_metrics(self) -> Dict:
        """Get summary metrics for the episode."""
        return {
            'total_cost': self.total_cost,
            'bullwhip_ratios': self.compute_bullwhip_ratio(),
            'avg_inventory': [np.mean(h) for h in self.inventory_history],
            'avg_backlog': np.mean([np.sum(self.backlog)]),  # Simplified
            'order_variance': [np.var(h) for h in self.order_history],
        }


class BeerGameEnvFlat(BeerGameEnv):
    """Beer Game with flattened discrete action space for standard MCTS.

    Instead of MultiDiscrete, uses single Discrete action that encodes
    all echelon orders. For small order_levels, this is tractable.

    With order_levels=5, total actions = 5^4 = 625
    With order_levels=3, total actions = 3^4 = 81
    """

    def __init__(self, order_levels: int = 5, **kwargs):
        # First init parent with MultiDiscrete
        kwargs['order_levels'] = order_levels
        super().__init__(**kwargs)

        # Override action space to flat Discrete
        self.flat_action_size = order_levels ** self.n_echelons
        self.action_space = spaces.Discrete(self.flat_action_size)

        # Pre-compute action decoding
        self._action_bases = [order_levels ** i for i in range(self.n_echelons)]

    def _decode_action(self, flat_action: int) -> np.ndarray:
        """Decode flat action index to per-echelon orders."""
        actions = np.zeros(self.n_echelons, dtype=int)
        remaining = flat_action
        for i in range(self.n_echelons - 1, -1, -1):
            actions[i] = remaining // self._action_bases[i]
            remaining = remaining % self._action_bases[i]
        return actions

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step with flat action."""
        multi_action = self._decode_action(action)
        return super().step(multi_action)


class BeerGameEnvSequential(BeerGameEnv):
    """Beer Game with sequential action selection.

    Each environment step consists of 4 substeps, one per echelon.
    MCTS selects one echelon's order at a time.

    This reduces action space from O_levels^4 to O_levels * 4.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Override action space to single echelon order
        self.action_space = spaces.Discrete(self.order_levels)

        # Observation gains one dimension for the current echelon index
        base_space = self.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([base_space.low, [0.0]]),
            high=np.concatenate([base_space.high, [float(self.n_echelons - 1)]]),
            dtype=np.float32,
        )

        # Track substep within environment step
        self.current_echelon = 0
        self.pending_orders = np.zeros(self.n_echelons, dtype=np.int64)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_echelon = 0
        self.pending_orders = np.zeros(self.n_echelons, dtype=np.int64)
        # Add current echelon to observation
        return self._augment_obs(obs), info

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Add current echelon index to observation."""
        return np.concatenate([obs, [float(self.current_echelon)]])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute substep (single echelon order decision)."""
        # Record order for current echelon
        self.pending_orders[self.current_echelon] = action

        self.current_echelon += 1

        if self.current_echelon < self.n_echelons:
            # Not yet complete - return intermediate observation
            obs = self._augment_obs(self._get_obs())
            return obs, 0.0, False, False, {'substep': True}

        # All echelons have ordered - execute full environment step
        self.current_echelon = 0
        obs, reward, terminated, truncated, info = super().step(self.pending_orders)
        self.pending_orders = np.zeros(self.n_echelons, dtype=np.int64)

        info['substep'] = False
        return self._augment_obs(obs), reward, terminated, truncated, info
