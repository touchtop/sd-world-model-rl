"""SD World Model for Beer Game.

Differentiable world model encoding supply chain stock-flow dynamics.

Structure (hard-coded):
    - Inventory conservation: I_{t+1} = I_t + arrivals - shipments
    - Backlog conservation: B_{t+1} = B_t + orders_in - shipments
    - Pipeline delay: shift register for information and shipping delays
    - Non-negativity constraints via clamp operations

Parameters (learnable):
    - Demand rate (for mean-field approximation)
    - Lead times (if treated as uncertain)
    - Cost coefficients (for sensitivity analysis)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class BeerGameSDModel(nn.Module):
    """Differentiable SD world model for Beer Game dynamics.

    Encodes the stock-flow structure of a 4-echelon supply chain
    with information and shipping delays.

    State layout (26 dimensions):
        [0:6]    - Echelon 0 (Retailer): I, B, ship[0], ship[1], order[0], order[1]
        [6:12]   - Echelon 1 (Wholesaler): I, B, ship[0], ship[1], order[0], order[1]
        [12:18]  - Echelon 2 (Distributor): I, B, ship[0], ship[1], order[0], order[1]
        [18:24]  - Echelon 3 (Factory): I, B, ship[0], ship[1], order[0], order[1]
        [24]     - Current timestep
        [25]     - Current customer demand
    """

    def __init__(
        self,
        n_echelons: int = 4,
        tau_info: int = 2,
        tau_ship: int = 2,
        order_levels: int = 9,
        O_max: int = 16,
        h: float = 0.5,
        p: float = 1.0,
        demand_low: float = 4.0,
        demand_high: float = 8.0,
        demand_step_time: int = 5,
        production_capacity: int = 20,
        learn_demand: bool = True,
        learn_costs: bool = False,
        prior_belief: bool = False,
    ):
        super().__init__()

        self.n_echelons = n_echelons
        self.tau_info = tau_info
        self.tau_ship = tau_ship
        self.order_levels = order_levels
        self.O_max = O_max
        self.production_capacity = production_capacity
        self.prior_belief = prior_belief

        # Store true values for reference
        self.register_buffer('true_demand_step_time', torch.tensor(float(demand_step_time)))

        # Initialize demand parameters (possibly with wrong prior)
        init_step_time = float(demand_step_time)
        init_high = float(demand_high)
        if prior_belief:
            init_step_time += np.random.uniform(-3.0, 3.0)
            init_high *= np.random.uniform(0.7, 1.3)

        # Pre-compute order quantities (floor to match env's int truncation)
        order_quantities = torch.linspace(0, O_max, order_levels, dtype=torch.float32)
        order_quantities = torch.floor(order_quantities)
        self.register_buffer('order_quantities', order_quantities)

        # Known cost parameters (or learnable)
        if learn_costs:
            self._h_raw = nn.Parameter(torch.tensor(np.log(np.exp(h) - 1), dtype=torch.float32))
            self._p_raw = nn.Parameter(torch.tensor(np.log(np.exp(p) - 1), dtype=torch.float32))
        else:
            self.register_buffer('h', torch.tensor(h, dtype=torch.float32))
            self.register_buffer('p', torch.tensor(p, dtype=torch.float32))
        self.learn_costs = learn_costs

        # Learnable demand parameters
        if learn_demand:
            # Use softplus to ensure positivity for demand values
            self._demand_low_raw = nn.Parameter(
                torch.tensor(np.log(np.exp(demand_low) - 1), dtype=torch.float32)
            )
            self._demand_high_raw = nn.Parameter(
                torch.tensor(np.log(np.exp(init_high) - 1), dtype=torch.float32)
            )
            # Step time is learnable (no positivity constraint needed)
            self._demand_step_time = nn.Parameter(
                torch.tensor(init_step_time, dtype=torch.float32)
            )
        else:
            self.register_buffer('demand_low', torch.tensor(demand_low, dtype=torch.float32))
            self.register_buffer('demand_high', torch.tensor(init_high, dtype=torch.float32))
            self.register_buffer('_demand_step_time', torch.tensor(init_step_time, dtype=torch.float32))
        self.learn_demand = learn_demand

        # State dimension
        self.state_dim = n_echelons * 6 + 2

    @property
    def holding_cost(self) -> torch.Tensor:
        if self.learn_costs:
            return F.softplus(self._h_raw)
        return self.h

    @property
    def backlog_cost(self) -> torch.Tensor:
        if self.learn_costs:
            return F.softplus(self._p_raw)
        return self.p

    def get_demand(self, t: torch.Tensor) -> torch.Tensor:
        """Get demand for timestep t (mean-field approximation)."""
        if self.learn_demand:
            demand_low = F.softplus(self._demand_low_raw)
            demand_high = F.softplus(self._demand_high_raw)
            # Straight-through estimator: hard gate forward, soft gate backward
            # This aligns model behavior with environment's discrete step
            sharpness = 2.0
            step_offset = 0.5  # Align sigmoid midpoint with discrete boundary
            logits = (t + step_offset - self._demand_step_time) * sharpness
            gate_soft = torch.sigmoid(logits)
            gate_hard = (t >= self._demand_step_time).to(gate_soft.dtype)
            gate = gate_hard.detach() - gate_soft.detach() + gate_soft
            return demand_low + (demand_high - demand_low) * gate
        else:
            demand_low = self.demand_low
            demand_high = self.demand_high
            return torch.where(
                t < self._demand_step_time,
                demand_low.expand_as(t),
                demand_high.expand_as(t)
            )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state given current state and action.

        Args:
            state: (batch, 26) current state
            action: (batch, n_echelons) order indices per echelon
                    OR (batch,) flat action index for BeerGameEnvFlat

        Returns:
            next_state: (batch, 26) predicted next state
        """
        batch_size = state.shape[0]
        device = state.device

        # Parse state
        # Each echelon has 6 variables: I, B, ship[0], ship[1], order[0], order[1]
        echelon_states = state[:, :self.n_echelons * 6].view(batch_size, self.n_echelons, 6)
        inventory = echelon_states[:, :, 0]  # (batch, n_echelons)
        backlog = echelon_states[:, :, 1]
        ship_0 = echelon_states[:, :, 2]  # Arrives next period
        ship_1 = echelon_states[:, :, 3]  # Arrives in 2 periods
        order_0 = echelon_states[:, :, 4]  # Arrives upstream next period
        order_1 = echelon_states[:, :, 5]  # Arrives upstream in 2 periods

        t = state[:, -2]  # Current timestep
        current_demand = state[:, -1]  # Current demand (from observation)

        # Decode actions to order quantities
        if action.dim() == 1:
            # Flat action - need to decode
            orders = self._decode_flat_action(action)
        else:
            # Multi-discrete action
            orders = self.order_quantities[action.long()]  # (batch, n_echelons)

        # Step 1: Receive incoming shipments
        arrivals = ship_0.clone()

        # Step 2: Advance pipelines (shift register)
        new_ship_0 = ship_1.clone()
        new_ship_1 = torch.zeros_like(ship_1)

        # Step 3: Receive incoming orders
        incoming_orders = torch.zeros(batch_size, self.n_echelons, device=device)
        incoming_orders[:, 0] = current_demand  # Retailer receives customer demand
        for i in range(1, self.n_echelons):
            incoming_orders[:, i] = order_0[:, i - 1]

        # Step 4: Update inventory and backlog
        new_inventory = inventory + arrivals
        new_backlog = backlog + incoming_orders

        # Step 5: Ship goods (fulfill orders)
        shipments = torch.minimum(new_inventory, new_backlog)
        new_inventory = new_inventory - shipments
        new_backlog = new_backlog - shipments

        # Step 6: Shipments go into downstream pipeline
        # Echelon i+1's shipment goes to echelon i's pipeline
        for i in range(self.n_echelons - 1):
            new_ship_1[:, i] = shipments[:, i + 1]

        # Factory receives production (clamped to capacity)
        factory_production = torch.clamp(orders[:, -1], max=self.production_capacity)
        new_ship_1[:, -1] = factory_production

        # Step 7: Advance order pipeline (information delay)
        new_order_0 = order_1.clone()
        new_order_1 = torch.zeros_like(order_1)
        new_order_1 = new_order_1 + orders

        # Clamp to non-negative (conservation law enforcement)
        new_inventory = torch.clamp(new_inventory, min=0)
        new_backlog = torch.clamp(new_backlog, min=0)
        new_ship_0 = torch.clamp(new_ship_0, min=0)
        new_ship_1 = torch.clamp(new_ship_1, min=0)
        new_order_0 = torch.clamp(new_order_0, min=0)
        new_order_1 = torch.clamp(new_order_1, min=0)

        # Reconstruct state
        new_echelon_states = torch.stack([
            new_inventory, new_backlog, new_ship_0, new_ship_1, new_order_0, new_order_1
        ], dim=2)  # (batch, n_echelons, 6)

        next_demand = self.get_demand(t + 1)
        new_state = torch.cat([
            new_echelon_states.view(batch_size, -1),
            (t + 1).unsqueeze(1),
            next_demand.unsqueeze(1),
        ], dim=1)

        return new_state

    def _decode_flat_action(self, flat_action: torch.Tensor) -> torch.Tensor:
        """Decode flat action index to per-echelon orders."""
        batch_size = flat_action.shape[0]
        device = flat_action.device

        actions = torch.zeros(batch_size, self.n_echelons, device=device, dtype=torch.long)
        remaining = flat_action.long()

        for i in range(self.n_echelons - 1, -1, -1):
            base = self.order_levels ** i
            actions[:, i] = remaining // base
            remaining = remaining % base

        return self.order_quantities[actions]

    def predict_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict reward (negative cost) for state-action pair.

        Cost = h * sum(I^+) + p * sum(B)
        """
        if next_state is None:
            next_state = self.forward(state, action)

        batch_size = next_state.shape[0]

        # Parse next state for inventory and backlog
        echelon_states = next_state[:, :self.n_echelons * 6].view(
            batch_size, self.n_echelons, 6
        )
        inventory = echelon_states[:, :, 0]
        backlog = echelon_states[:, :, 1]

        # Compute costs
        holding_cost = self.holding_cost * torch.sum(torch.clamp(inventory, min=0), dim=1)
        backlog_cost = self.backlog_cost * torch.sum(backlog, dim=1)

        total_cost = holding_cost + backlog_cost
        reward = -total_cost

        return reward

    def get_learned_params(self) -> dict:
        """Return current learned parameter values."""
        params = {}

        if self.learn_demand:
            params['demand_low'] = F.softplus(self._demand_low_raw).item()
            params['demand_high'] = F.softplus(self._demand_high_raw).item()
            params['demand_step_time'] = self._demand_step_time.item()
            params['true_step_time'] = self.true_demand_step_time.item()

        if self.learn_costs:
            params['h'] = F.softplus(self._h_raw).item()
            params['p'] = F.softplus(self._p_raw).item()

        return params


class BeerGameWorldModelWrapper:
    """Numpy wrapper for BeerGameSDModel for use with MCTS.

    Provides interface expected by mcts_search function.
    """

    def __init__(self, model: BeerGameSDModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Predict next state (numpy interface)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)

            action_t = torch.tensor(action, dtype=torch.long, device=self.device)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)
            if action_t.dim() == 1 and action_t.shape[0] == self.model.n_echelons:
                action_t = action_t.unsqueeze(0)

            next_state = self.model.forward(state_t, action_t)
            return next_state.squeeze(0).cpu().numpy()

    def predict_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Predict reward (numpy interface)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)

            action_t = torch.tensor(action, dtype=torch.long, device=self.device)
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)
            if action_t.dim() == 1 and action_t.shape[0] == self.model.n_echelons:
                action_t = action_t.unsqueeze(0)

            reward = self.model.predict_reward(state_t, action_t)
            return reward.squeeze().cpu().item()
