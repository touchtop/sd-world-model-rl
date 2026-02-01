"""Baseline policies for Beer Game experiments.

Implements classical supply chain policies:
1. Base-Stock Policy: Order to reach target inventory position
2. Anchoring-and-Adjustment: Sterman (1989) behavioral model
3. Oracle MPC: Model Predictive Control with perfect information
"""
import numpy as np
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


class BeerGamePolicy(ABC):
    """Abstract base class for Beer Game policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select order quantities for all echelons.

        Args:
            state: (26,) observation vector

        Returns:
            orders: (4,) order quantities per echelon
        """
        pass

    def reset(self):
        """Reset policy state if needed."""
        pass


class BaseStockPolicy(BeerGamePolicy):
    """Base-stock (order-up-to) inventory policy.

    Each echelon orders to reach a target inventory position:
        Order = max(0, S - IP)

    where IP (inventory position) = Inventory - Backlog + SupplyLine

    This is optimal for single-echelon systems with known lead times.
    For multi-echelon, it's a strong heuristic.
    """

    def __init__(
        self,
        target_levels: List[int] = None,
        n_echelons: int = 4,
        order_levels: int = 9,
        O_max: int = 16,
    ):
        """
        Args:
            target_levels: Target inventory position per echelon.
                          Default uses lead_time * avg_demand formula.
            n_echelons: Number of echelons
            order_levels: Number of discrete order levels
            O_max: Maximum order quantity
        """
        self.n_echelons = n_echelons

        # Default target levels: (lead_time + review_period) * demand + safety_stock
        # For Beer Game: lead_time = 4 (2 info + 2 ship), demand = 8, safety = 4
        if target_levels is None:
            target_levels = [36, 36, 36, 36]  # Conservative targets
        self.target_levels = np.array(target_levels, dtype=np.float32)

        # Order discretization
        self.order_levels = order_levels
        self.O_max = O_max
        self.order_quantities = np.linspace(0, O_max, order_levels)

    def _parse_state(self, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Parse state vector into components."""
        inventory = state[0::6][:self.n_echelons]
        backlog = state[1::6][:self.n_echelons]
        ship_0 = state[2::6][:self.n_echelons]
        ship_1 = state[3::6][:self.n_echelons]
        order_0 = state[4::6][:self.n_echelons]
        order_1 = state[5::6][:self.n_echelons]
        return inventory, backlog, ship_0, ship_1, order_0, order_1

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Compute order-up-to quantities."""
        inventory, backlog, ship_0, ship_1, order_0, order_1 = self._parse_state(state)

        # Inventory position = I - B + supply line (ship + order pipelines)
        supply_line = ship_0 + ship_1 + order_0 + order_1
        inv_position = inventory - backlog + supply_line

        # Order = max(0, S - IP)
        raw_orders = np.maximum(0, self.target_levels - inv_position)

        # Discretize to available order levels
        orders = np.zeros(self.n_echelons, dtype=int)
        for i in range(self.n_echelons):
            # Find closest discrete order level
            idx = np.argmin(np.abs(self.order_quantities - raw_orders[i]))
            orders[i] = idx

        return orders


class AnchoringAdjustmentPolicy(BeerGamePolicy):
    """Sterman (1989) behavioral model of human ordering.

    Reference: Sterman, J.D. (1989). "Modeling Managerial Behavior: Misperceptions
    of Feedback in a Dynamic Decision Making Experiment." Management Science, 35(3).

    Human subjects in Beer Game experiments exhibit systematic biases:
    1. Anchoring on recent demand (underweight order pipeline)
    2. Adjustment based on stock discrepancy
    3. Adjustment based on supply line discrepancy

    Order = max(0, D̂ + α(S* - S) + β(SL* - SL))

    where:
        D̂ = expected demand, D̂_t = D̂_{t-1} + θ(D_t - D̂_{t-1})
        S* = target stock
        S = actual stock (inventory - backlog)
        SL* = target supply line
        SL = actual supply line (orders in pipeline)

    Default parameters from Sterman (1989) Table 2:
        α = 0.36 (stock adjustment, empirically estimated)
        β = 0.26 (supply line adjustment, systematically underweighted)
        θ = 0.2  (demand smoothing factor)

    This policy recreates the bullwhip effect observed in experiments.
    """

    def __init__(
        self,
        # Sterman (1989) Table 2 empirical estimates
        alpha: float = 0.36,    # Stock adjustment weight (empirical)
        beta: float = 0.26,     # Supply line adjustment (underweighted by humans)
        theta: float = 0.2,     # Demand smoothing factor
        target_stock: float = 12.0,
        target_supply_line: float = 8.0,
        n_echelons: int = 4,
        order_levels: int = 9,
        O_max: int = 16,
    ):
        """
        Args:
            alpha: Weight on stock discrepancy
            beta: Weight on supply line discrepancy (humans typically underweighted)
            theta: Exponential smoothing for demand estimation
            target_stock: Desired inventory level
            target_supply_line: Desired orders in pipeline
        """
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.target_stock = target_stock
        self.target_supply_line = target_supply_line
        self.n_echelons = n_echelons
        self.order_levels = order_levels
        self.O_max = O_max
        self.order_quantities = np.linspace(0, O_max, order_levels)

        # State for demand estimation
        self.demand_estimate = np.full(n_echelons, 4.0)  # Initial demand estimate

    def reset(self):
        """Reset demand estimation state."""
        self.demand_estimate = np.full(self.n_echelons, 4.0)

    def _parse_state(self, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Parse state vector."""
        inventory = state[0::6][:self.n_echelons]
        backlog = state[1::6][:self.n_echelons]
        ship_0 = state[2::6][:self.n_echelons]
        ship_1 = state[3::6][:self.n_echelons]
        order_0 = state[4::6][:self.n_echelons]
        order_1 = state[5::6][:self.n_echelons]
        current_demand = state[-1]
        return inventory, backlog, ship_0, ship_1, order_0, order_1, current_demand

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Compute orders using anchoring-and-adjustment heuristic."""
        inventory, backlog, ship_0, ship_1, order_0, order_1, current_demand = self._parse_state(state)

        # Update demand estimate (exponential smoothing)
        # For simplicity, use observed demand at retailer as proxy
        observed_demand = np.zeros(self.n_echelons)
        observed_demand[0] = current_demand
        observed_demand[1:] = order_0[:-1]

        self.demand_estimate = self.theta * observed_demand + (1 - self.theta) * self.demand_estimate

        # Compute actual stock and supply line
        stock = inventory - backlog
        supply_line = ship_0 + ship_1 + order_0 + order_1

        # Compute order using anchoring-and-adjustment
        raw_orders = np.zeros(self.n_echelons)
        for i in range(self.n_echelons):
            # Expected demand
            expected_demand = self.demand_estimate[i]

            # Adjustment 2: Stock discrepancy
            stock_adj = self.alpha * (self.target_stock - stock[i])

            # Adjustment 3: Supply line discrepancy (often underweighted by humans)
            sl_adj = self.beta * (self.target_supply_line - supply_line[i])

            # Final order
            order = expected_demand + stock_adj + sl_adj
            raw_orders[i] = max(0, order)

        # Discretize orders
        orders = np.zeros(self.n_echelons, dtype=int)
        for i in range(self.n_echelons):
            idx = np.argmin(np.abs(self.order_quantities - raw_orders[i]))
            orders[i] = idx

        return orders


class ConstantOrderPolicy(BeerGamePolicy):
    """Simple constant order policy (for baseline comparison)."""

    def __init__(
        self,
        order_rate: int = 4,
        n_echelons: int = 4,
        order_levels: int = 9,
        O_max: int = 16,
    ):
        self.n_echelons = n_echelons
        self.order_quantities = np.linspace(0, O_max, order_levels)

        # Find closest discrete level to order_rate
        self.order_idx = np.argmin(np.abs(self.order_quantities - order_rate))

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return np.full(self.n_echelons, self.order_idx, dtype=int)


class OracleMPCPolicy(BeerGamePolicy):
    """Model Predictive Control with perfect model knowledge.

    Rolls out trajectories using known dynamics and selects
    action sequence minimizing total cost over horizon H.

    Uses random shooting for tractability (evaluates K random sequences).
    """

    def __init__(
        self,
        horizon: int = 8,
        n_samples: int = 100,
        n_echelons: int = 4,
        order_levels: int = 9,
        O_max: int = 16,
        h: float = 0.5,
        p: float = 1.0,
        demand_low: int = 4,
        demand_high: int = 8,
        demand_step_time: int = 5,
        production_capacity: int = 20,
        seed: int = None,
    ):
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_echelons = n_echelons
        self.order_levels = order_levels
        self.O_max = O_max
        self.h = h
        self.p = p
        self.demand_low = demand_low
        self.demand_high = demand_high
        self.demand_step_time = demand_step_time
        self.production_capacity = production_capacity
        self.rng = np.random.default_rng(seed)

        self.order_quantities = np.linspace(0, O_max, order_levels)

    def _simulate_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        t: int,
    ) -> Tuple[np.ndarray, float]:
        """Simulate one step of Beer Game dynamics."""
        # Parse state
        inventory = state[0::6][:self.n_echelons].copy()
        backlog = state[1::6][:self.n_echelons].copy()
        ship_0 = state[2::6][:self.n_echelons].copy()
        ship_1 = state[3::6][:self.n_echelons].copy()
        order_0 = state[4::6][:self.n_echelons].copy()
        order_1 = state[5::6][:self.n_echelons].copy()

        orders = self.order_quantities[action]

        # Get demand
        demand = self.demand_high if t >= self.demand_step_time else self.demand_low

        # Receive shipments
        arrivals = ship_0.copy()

        # Advance pipeline
        new_ship_0 = ship_1.copy()
        new_ship_1 = np.zeros(self.n_echelons)

        new_order_0 = order_1.copy()
        new_order_1 = np.zeros(self.n_echelons)

        # Receive orders
        incoming_orders = np.zeros(self.n_echelons)
        incoming_orders[0] = demand
        for i in range(1, self.n_echelons):
            incoming_orders[i] = order_0[i - 1]

        # Update inventory and backlog
        inventory = inventory + arrivals
        backlog = backlog + incoming_orders

        # Ship goods
        shipments = np.minimum(inventory, backlog)
        inventory = inventory - shipments
        backlog = backlog - shipments

        # Update downstream pipelines
        for i in range(self.n_echelons - 1):
            new_ship_1[i] = shipments[i + 1]
        new_ship_1[-1] = min(orders[-1], self.production_capacity)  # Factory production

        for i in range(self.n_echelons):
            new_order_1[i] = orders[i]

        # Compute cost
        cost = self.h * np.sum(np.maximum(0, inventory)) + self.p * np.sum(backlog)

        # Reconstruct state
        new_state = np.zeros(self.n_echelons * 6 + 2)
        for i in range(self.n_echelons):
            base = i * 6
            new_state[base] = inventory[i]
            new_state[base + 1] = backlog[i]
            new_state[base + 2] = new_ship_0[i]
            new_state[base + 3] = new_ship_1[i]
            new_state[base + 4] = new_order_0[i]
            new_state[base + 5] = new_order_1[i]
        new_state[-2] = t + 1
        new_state[-1] = demand

        return new_state, cost

    def _evaluate_sequence(
        self,
        state: np.ndarray,
        action_sequence: np.ndarray,
    ) -> float:
        """Evaluate total cost of action sequence."""
        current_state = state.copy()
        t = int(state[-2])
        total_cost = 0.0

        for h in range(self.horizon):
            current_state, cost = self._simulate_step(
                current_state,
                action_sequence[h],
                t + h,
            )
            total_cost += cost

        return total_cost

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using random shooting MPC."""
        best_cost = float('inf')
        best_action = None

        for _ in range(self.n_samples):
            # Generate random action sequence
            action_sequence = self.rng.integers(
                0, self.order_levels,
                size=(self.horizon, self.n_echelons)
            )

            # Evaluate
            cost = self._evaluate_sequence(state, action_sequence)

            if cost < best_cost:
                best_cost = cost
                best_action = action_sequence[0].copy()

        return best_action


class ReactivePolicy(BeerGamePolicy):
    """Reactive policy that matches observed demand.

    Orders = observed incoming orders (with possible smoothing).
    This is what many naive players do - react to what they see.
    """

    def __init__(
        self,
        smoothing: float = 0.0,
        n_echelons: int = 4,
        order_levels: int = 9,
        O_max: int = 16,
    ):
        self.smoothing = smoothing
        self.n_echelons = n_echelons
        self.order_quantities = np.linspace(0, O_max, order_levels)
        self.last_orders = None

    def reset(self):
        self.last_orders = None

    def select_action(self, state: np.ndarray) -> np.ndarray:
        # Get current demand (as proxy for incoming orders)
        current_demand = state[-1]

        # For all echelons, order based on perceived demand
        raw_orders = np.full(self.n_echelons, current_demand)

        if self.last_orders is not None and self.smoothing > 0:
            raw_orders = self.smoothing * self.order_quantities[self.last_orders] + \
                        (1 - self.smoothing) * raw_orders

        # Discretize
        orders = np.zeros(self.n_echelons, dtype=int)
        for i in range(self.n_echelons):
            orders[i] = np.argmin(np.abs(self.order_quantities - raw_orders[i]))

        self.last_orders = orders
        return orders


class StermanPolicy(BeerGamePolicy):
    """Sterman (1989) anchoring-and-adjustment heuristic.

    Models human decision-making behavior in the Beer Game.
    Humans typically underweight supply line information, causing bullwhip effect.

    Order = Expected_Demand + α*(Target_Stock - Stock) + β*(Target_SL - Supply_Line)

    Key insight: β << α (humans underweight supply line), causing oscillations.

    Reference: Sterman, J.D. (1989). Modeling managerial behavior: Misperceptions
    of feedback in a dynamic decision making experiment. Management Science, 35(3).
    """

    def __init__(
        self,
        alpha: float = 0.26,  # Stock adjustment weight (from Sterman's estimates)
        beta: float = 0.09,   # Supply line adjustment weight (underweighted!)
        target_stock: float = 12.0,
        target_supply_line: float = 12.0,
        demand_smoothing: float = 0.36,  # Exponential smoothing for demand
        n_echelons: int = 4,
        order_levels: int = 9,
        O_max: int = 16,
    ):
        """
        Args:
            alpha: Weight for stock discrepancy adjustment (typically 0.2-0.4)
            beta: Weight for supply line discrepancy (typically 0.05-0.15, underweighted)
            target_stock: Desired inventory level
            target_supply_line: Desired pipeline inventory
            demand_smoothing: Exponential smoothing factor for demand estimation
        """
        self.alpha = alpha
        self.beta = beta
        self.target_stock = target_stock
        self.target_supply_line = target_supply_line
        self.demand_smoothing = demand_smoothing
        self.n_echelons = n_echelons
        self.order_levels = order_levels
        self.O_max = O_max
        self.order_quantities = np.linspace(0, O_max, order_levels)

        # State for demand estimation
        self.demand_estimate = np.full(n_echelons, 4.0)

    def reset(self):
        """Reset demand estimation state."""
        self.demand_estimate = np.full(self.n_echelons, 4.0)

    def _parse_state(self, state: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Parse state vector."""
        inventory = state[0::6][:self.n_echelons]
        backlog = state[1::6][:self.n_echelons]
        ship_0 = state[2::6][:self.n_echelons]
        ship_1 = state[3::6][:self.n_echelons]
        order_0 = state[4::6][:self.n_echelons]
        order_1 = state[5::6][:self.n_echelons]
        current_demand = state[-1]
        return inventory, backlog, ship_0, ship_1, order_0, order_1, current_demand

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Compute orders using Sterman's anchoring-and-adjustment heuristic."""
        inventory, backlog, ship_0, ship_1, order_0, order_1, current_demand = self._parse_state(state)

        # Observed incoming orders for each echelon
        observed_orders = np.zeros(self.n_echelons)
        observed_orders[0] = current_demand  # Retailer sees customer demand
        for i in range(1, self.n_echelons):
            observed_orders[i] = order_0[i - 1]  # Upstream sees downstream orders

        # Update demand estimate (exponential smoothing)
        self.demand_estimate = (self.demand_smoothing * observed_orders +
                                (1 - self.demand_smoothing) * self.demand_estimate)

        # Compute stock (net inventory position)
        stock = inventory - backlog

        # Compute supply line (orders in transit + orders placed)
        supply_line = ship_0 + ship_1 + order_0 + order_1

        # Sterman's ordering rule
        raw_orders = np.zeros(self.n_echelons)
        for i in range(self.n_echelons):
            # Expected demand (anchor)
            order = self.demand_estimate[i]

            # Stock adjustment (humans do this reasonably well)
            order += self.alpha * (self.target_stock - stock[i])

            # Supply line adjustment (humans underweight this - key behavioral flaw!)
            order += self.beta * (self.target_supply_line - supply_line[i])

            raw_orders[i] = max(0, order)

        # Discretize orders
        orders = np.zeros(self.n_echelons, dtype=int)
        for i in range(self.n_echelons):
            orders[i] = np.argmin(np.abs(self.order_quantities - raw_orders[i]))

        return orders
