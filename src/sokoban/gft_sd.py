"""Two-layer GFT-SD readout layer for Sokoban.

The lower layer remains the audited exact grid stock-flow model.  This module
adds a separate Goal-conditioned Feasible Transport System Dynamics (GFT-SD)
resource layer, matching ``sokoban_gptpro.md``: the resource layer is computed
from the Markov ledger state rather than used as a replacement transition
state.
"""

from __future__ import annotations

import math
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .env import ACTION_NAMES, DIRECTIONS, Cell, add_cells
from .sd_model import SokobanGridStockFlowModel, SokobanPushAction, SokobanState


def _safe_float(value: float | int | bool | None) -> float | int | bool | None:
    if isinstance(value, bool) or value is None:
        return value
    value = float(value)
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(float(value))
    if isinstance(value, (float, int, bool)) or value is None:
        return _safe_float(value)
    return value


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _matching_progress(parent_lb: float, child_lb: float) -> float:
    """Bounded transport-progress delta used by the current best prior."""

    if math.isfinite(parent_lb) and math.isfinite(child_lb):
        return _clamp(parent_lb - child_lb, -4.0, 4.0)
    if math.isfinite(parent_lb):
        return -4.0
    if math.isfinite(child_lb):
        return 4.0
    return 0.0


def _merge_float_weights(defaults: Mapping[str, float], overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    weights = {key: float(value) for key, value in defaults.items()}
    for key, value in dict(overrides or {}).items():
        if key not in weights:
            raise KeyError(f"Unknown GFT weight {key!r}; expected one of {sorted(weights)}")
        weights[key] = float(value)
    return weights


def _finite_delta(before: float, after: float) -> float | None:
    if math.isfinite(before) and math.isfinite(after):
        return float(after - before)
    return None


def _ratio(numer: float, denom: float) -> float:
    return _clamp(float(numer) / max(float(denom), 1.0), 0.0, 1.0)


@dataclass(frozen=True)
class GFTResourceValue:
    """One GFT resource stock readout for a concrete ledger state."""

    code: str
    name: str
    value: float
    active: bool
    implemented: bool
    components: Mapping[str, Any]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "value": _jsonable(self.value),
            "active": bool(self.active),
            "implemented": bool(self.implemented),
            "components": _jsonable(dict(self.components)),
            "description": self.description,
        }


@dataclass(frozen=True)
class GFTStateReadout:
    """Full GFT resource layer ``z_t = Phi(y_t)`` for one state."""

    ledger_key: tuple
    resources: tuple[GFTResourceValue, ...]

    def to_dict(self, *, active_only: bool = False) -> dict[str, Any]:
        resources = [r for r in self.resources if (r.active or not active_only)]
        return {
            "ledger_key": {
                "boxes": [list(cell) for cell in self.ledger_key[0]],
                "player_component_min": list(self.ledger_key[1]),
            },
            "resources": {r.code: r.to_dict() for r in resources},
        }


class GFTResourceStock:
    """Base class for open-ended GFT resource stocks.

    Subclasses compute a resource from the exact ledger state.  The resource
    layer is intentionally read-only with respect to transition dynamics.
    """

    code = "?"
    name = "unnamed"
    description = ""
    active = True
    implemented = True

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        raise NotImplementedError

    def value(
        self,
        model: "SokobanGFTSDModel",
        state: SokobanState,
        *,
        value: float,
        components: Mapping[str, Any] | None = None,
    ) -> GFTResourceValue:
        return GFTResourceValue(
            code=self.code,
            name=self.name,
            value=float(value),
            active=bool(self.active),
            implemented=bool(self.implemented),
            components=components or {},
            description=self.description,
        )


class TransportProgressStock(GFTResourceStock):
    code = "P"
    name = "transport_progress"
    description = "Remaining goal-conditioned transport work from the document cost matrix."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        summary = model.gft_transport_summary_cached(state)
        lb = float(summary["transport_cost"])
        return self.value(
            model,
            state,
            value=lb,
            components={
                "matching_lower_bound": lb,
                "cost_formula": summary["cost_formula"],
                "transport_cost_weights": summary["transport_cost_weights"],
                "matching_assignment": summary["matching_assignment"],
                "feasible": summary["feasible"],
                "boxes_on_goals": model.boxes_on_goals(state),
                "lower_is_better": True,
            },
        )


class BoxMobilityStock(GFTResourceStock):
    code = "M"
    name = "box_mobility"
    description = "Future push freedom, measured by legal and viable push counts."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        features = model.gft_abstract_features_cached(state)
        return self.value(
            model,
            state,
            value=float(features.viable_push_count),
            components={
                "legal_push_count": features.legal_push_count,
                "viable_push_count": features.viable_push_count,
                "higher_is_better": True,
            },
        )


class PushAccessibilityStock(GFTResourceStock):
    code = "A"
    name = "push_accessibility"
    description = "Player reachability support for feasible push positions."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        features = model.gft_abstract_features_cached(state)
        return self.value(
            model,
            state,
            value=float(features.legal_push_count),
            components={
                "reachable_count": features.reachable_count,
                "legal_push_count": features.legal_push_count,
                "floor_count": len(model.floor_cells),
                "higher_is_better": True,
            },
        )


class RegionCapacitySlackStock(GFTResourceStock):
    code = "K"
    name = "region_capacity_slack"
    description = "Goal-region capacity health from cut-vertex region decomposition; higher is better."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        summary = model.gft_capacity_summary_cached(state)
        return self.value(
            model,
            state,
            value=float(summary["capacity_health"]),
            components=summary,
        )


class DeadlockDebtStock(GFTResourceStock):
    code = "D"
    name = "deadlock_debt"
    description = "Hard deadlock debt plus normalized soft deadlock pressure."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        summary = model.gft_deadlock_summary_cached(state)
        return self.value(
            model,
            state,
            value=float(summary["deadlock_debt"]),
            components=summary,
        )


class OrderingDebtStock(GFTResourceStock):
    code = "Q"
    name = "ordering_debt"
    description = "Goal-region ordering debt from entry-depth partial-order constraints; higher is worse."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        summary = model.gft_ordering_summary_cached(state)
        return self.value(
            model,
            state,
            value=float(summary["ordering_debt"]),
            components=summary,
        )


class GoalCommitmentStock(GFTResourceStock):
    code = "C"
    name = "goal_commitment"
    description = "Box-goal assignment commitment from Sinkhorn-normalized transport costs; higher is better."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        summary = model.gft_commitment_summary_cached(state)
        return self.value(
            model,
            state,
            value=float(summary["commitment_confidence"]),
            components=summary,
        )


class ActionRelevanceStock(GFTResourceStock):
    code = "I"
    name = "action_relevance"
    description = "Action-dependent relevance score consumed as a soft MCTS prior."

    def compute(self, model: "SokobanGFTSDModel", state: SokobanState) -> GFTResourceValue:
        return self.value(
            model,
            state,
            value=0.0,
            components={
                "status": "action_dependent",
                "state_scalar": "not_applicable",
            },
        )


def default_gft_resource_stocks() -> tuple[GFTResourceStock, ...]:
    return (
        TransportProgressStock(),
        BoxMobilityStock(),
        PushAccessibilityStock(),
        RegionCapacitySlackStock(),
        DeadlockDebtStock(),
        OrderingDebtStock(),
        GoalCommitmentStock(),
        ActionRelevanceStock(),
    )


DEFAULT_GFT_PRIOR_WEIGHTS: dict[str, float] = {
    "progress": 1.0,
    "goal": 1.0,
    "mobility": 0.0,
    "access": 0.0,
    "capacity": 0.0,
    "deadlock": 0.0,
    "ordering": 0.0,
    "commitment": 0.0,
    "bottleneck": 0.0,
    "revisit": 1.0,
}


DEFAULT_GFT_TRANSPORT_COST_WEIGHTS: dict[str, float] = {
    "turn": 0.0,
    "access": 0.0,
    "zone": 0.0,
}


DEFAULT_GFT_SOFT_DEADLOCK_WEIGHTS: dict[str, float] = {
    "mobility_loss": 1.0,
    "matching_fragility": 1.0,
    "capacity_stress": 1.0,
    "access_loss": 1.0,
    "order_violation": 1.0,
}


DEFAULT_GFT_SD_REWARD_WEIGHTS: dict[str, float] = {
    "progress": 1.0,
    "mobility": 0.0,
    "access": 0.0,
    "capacity": 0.0,
    "deadlock": 1.0,
    "ordering": 1.0,
}


DEFAULT_GFT_SD_VALUE_WEIGHTS: dict[str, float] = {
    "progress": 1.0,
    "mobility": 0.0,
    "access": 0.0,
    "capacity": 0.0,
    "deadlock": 1.0,
    "ordering": 1.0,
    "commitment": 0.0,
}


class SokobanGFTSDModel(SokobanGridStockFlowModel):
    """Exact ledger state plus a GFT-SD resource readout layer.

    The transition provider remains the exact grid stock-flow system inherited
    from :class:`SokobanGridStockFlowModel`.  GFT resources are computed as
    ``z_t = Phi(y_t)`` and can be consumed by priors, leaf values, diagnostics,
    or future TVA feature builders without changing the transition provider.
    """

    provider_id = "gft_sd_v1"
    transition_provider_id = SokobanGridStockFlowModel.provider_id

    def __init__(
        self,
        level,
        resource_stocks: Sequence[GFTResourceStock] | None = None,
        *,
        transport_cost_weights: Mapping[str, float] | None = None,
        soft_deadlock_weights: Mapping[str, float] | None = None,
        commitment_beta: float = 1.0,
        sinkhorn_iterations: int = 25,
    ):
        super().__init__(level)
        self.gft_resource_stocks = tuple(resource_stocks or default_gft_resource_stocks())
        self.gft_transport_cost_weights = _merge_float_weights(
            DEFAULT_GFT_TRANSPORT_COST_WEIGHTS,
            transport_cost_weights,
        )
        self.gft_soft_deadlock_weights = _merge_float_weights(
            DEFAULT_GFT_SOFT_DEADLOCK_WEIGHTS,
            soft_deadlock_weights,
        )
        self.gft_commitment_beta = float(commitment_beta)
        self.gft_sinkhorn_iterations = max(int(sinkhorn_iterations), 0)
        self.gft_capacity_regions = self._build_capacity_regions()
        self.gft_ordering_constraints = self._build_ordering_constraints()
        self._gft_turn_distances: tuple[dict[Cell, int], ...] | None = None
        self._gft_lb_cache: dict[tuple, float] = {}
        self._gft_key_cache: dict[SokobanState, tuple] = {}
        self._gft_features_cache: dict[SokobanState, Any] = {}
        self._gft_transport_cache: dict[SokobanState, dict[str, Any]] = {}
        self._gft_capacity_cache: dict[SokobanState, dict[str, Any]] = {}
        self._gft_ordering_cache: dict[SokobanState, dict[str, Any]] = {}
        self._gft_commitment_cache: dict[SokobanState, dict[str, Any]] = {}
        self._gft_deadlock_cache: dict[SokobanState, dict[str, Any]] = {}
        self._gft_resource_vector_cache: dict[SokobanState, dict[str, float]] = {}
        self._gft_tva_feature_cache: dict[SokobanState, dict[str, Any]] = {}

    def boxes_on_goals(self, state: SokobanState) -> int:
        return sum(1 for box in state.boxes if box in self.level.goals)

    def gft_matching_lower_bound_cached(self, boxes: Iterable) -> float:
        key = tuple(boxes)
        cached = self._gft_lb_cache.get(key)
        if cached is None:
            cached = self.matching_lower_bound(key)
            self._gft_lb_cache[key] = cached
        return cached

    def gft_canonical_key_cached(self, state: SokobanState) -> tuple:
        cached = self._gft_key_cache.get(state)
        if cached is None:
            cached = self.canonical_key(state)
            self._gft_key_cache[state] = cached
        return cached

    def gft_abstract_features_cached(self, state: SokobanState):
        cached = self._gft_features_cache.get(state)
        if cached is None:
            cached = self.abstract_features(state)
            self._gft_features_cache[state] = cached
        return cached

    @classmethod
    def gft_spec_alignment(cls) -> dict[str, Any]:
        return {
            "design_document": "sokoban_gptpro.md",
            "two_layer_state": {
                "ledger_state_y": "y_t=(B_t,R_t), exact box positions plus player reachable component",
                "resource_state_z": "z_t=Phi(y_t)=(P,M,A,K,D,Q,C,I)",
                "transition_provider": cls.transition_provider_id,
                "resource_layer_replaces_transition_provider": False,
            },
            "resource_stocks": {
                "P": "goal-conditioned transport progress / remaining transport work",
                "M": "box mobility",
                "A": "player push-position accessibility",
                "K": "goal-region capacity slack",
                "D": "hard deadlock debt plus soft deadlock pressure",
                "Q": "ordering-dependency debt",
                "C": "goal commitment clarity from Sinkhorn assignment entropy",
                "I": "action relevance for the soft MCTS prior",
            },
            "transport_cost": {
                "formula": "d_g(b_i)+lambda_turn*turns+lambda_access*access_risk+lambda_zone*zone_risk",
                "default_weights": dict(DEFAULT_GFT_TRANSPORT_COST_WEIGHTS),
                "default_reduces_to_matching_lower_bound": True,
            },
            "soft_deadlock": {
                "formula": "alpha1*mobilityLoss+alpha2*matchingFragility+alpha3*capacityStress+alpha4*accessLoss+alpha5*orderViolation",
                "default_weights": dict(DEFAULT_GFT_SOFT_DEADLOCK_WEIGHTS),
            },
            "interfaces": {
                "resource_vector": "gft_resource_vector(state)",
                "resource_flow": "gft_resource_flow(state, child_state)",
                "sd_reward": "gft_sd_reward(state, child_state)",
                "sd_value": "gft_sd_value(state)",
                "tva_features": "gft_tva_feature_dict(state)",
                "action_prior": "gft_action_prior_rows(state, candidates)",
            },
            "learnable_parameters": {
                "sd_layer": 0,
                "tva_layer": "external; consumes ledger and GFT feature vectors when trained",
            },
        }

    def _transport_weights_are_base_lb(self) -> bool:
        return all(abs(value) <= 1e-12 for value in self.gft_transport_cost_weights.values())

    def _compute_goal_turn_distances(self) -> tuple[dict[Cell, int], ...]:
        """Static minimum turn counts along reverse-push paths to each goal."""

        all_turns: list[dict[Cell, int]] = []
        for goal in self.goals_sorted:
            best_by_state: dict[tuple[Cell, int], tuple[int, int]] = {}
            best_by_cell: dict[Cell, tuple[int, int]] = {goal: (0, 0)}
            heap: list[tuple[int, int, Cell, int]] = [(0, 0, goal, -1)]
            while heap:
                turns, pushes, cell, last_direction = heapq.heappop(heap)
                state_key = (cell, last_direction)
                if best_by_state.get(state_key, (math.inf, math.inf)) < (turns, pushes):
                    continue
                for direction_idx, direction in enumerate(DIRECTIONS):
                    prev_box = (cell[0] - direction[0], cell[1] - direction[1])
                    player_stand = (cell[0] - 2 * direction[0], cell[1] - 2 * direction[1])
                    if not self.level.is_floor(prev_box) or not self.level.is_floor(player_stand):
                        continue
                    turn_cost = 0 if last_direction in (-1, direction_idx) else 1
                    nxt = (turns + turn_cost, pushes + 1)
                    nxt_state = (prev_box, direction_idx)
                    if nxt >= best_by_state.get(nxt_state, (math.inf, math.inf)):
                        continue
                    best_by_state[nxt_state] = nxt
                    if nxt < best_by_cell.get(prev_box, (math.inf, math.inf)):
                        best_by_cell[prev_box] = nxt
                    heapq.heappush(heap, (nxt[0], nxt[1], prev_box, direction_idx))
            all_turns.append({cell: turns for cell, (turns, _pushes) in best_by_cell.items()})
        return tuple(all_turns)

    def gft_goal_turn_distances(self) -> tuple[dict[Cell, int], ...]:
        if self._gft_turn_distances is None:
            self._gft_turn_distances = self._compute_goal_turn_distances()
        return self._gft_turn_distances

    def _min_cost_assignment(self, cost: np.ndarray) -> tuple[float, list[int]]:
        n = int(cost.shape[0])
        if n == 0:
            return 0.0, []
        dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
        for box_idx in range(n):
            next_dp: dict[int, tuple[float, list[int]]] = {}
            for mask, (prefix_cost, prefix_assignment) in dp.items():
                for goal_idx in range(n):
                    if mask & (1 << goal_idx):
                        continue
                    edge_cost = float(cost[box_idx, goal_idx])
                    if not math.isfinite(edge_cost):
                        continue
                    next_mask = mask | (1 << goal_idx)
                    candidate = (prefix_cost + edge_cost, prefix_assignment + [goal_idx])
                    if candidate[0] < next_dp.get(next_mask, (math.inf, []))[0]:
                        next_dp[next_mask] = candidate
            dp = next_dp
            if not dp:
                return math.inf, []
        return dp.get((1 << n) - 1, (math.inf, []))

    def _access_risks_by_box(self, state: SokobanState) -> dict[Cell, float]:
        reachable = self.reachable_cells(state)
        boxes = set(state.boxes)
        risks: dict[Cell, float] = {}
        for box in state.boxes:
            supported = 0
            possible = 0
            for direction in DIRECTIONS:
                player_stand = (box[0] - direction[0], box[1] - direction[1])
                box_target = add_cells(box, direction)
                if not self.level.is_floor(player_stand) or not self.level.is_floor(box_target):
                    continue
                if box_target in boxes:
                    continue
                possible += 1
                if player_stand in reachable:
                    supported += 1
            risks[box] = 1.0 if possible == 0 else 1.0 - _ratio(supported, possible)
        return risks

    def _zone_risks_by_goal(self, state: SokobanState) -> dict[Cell, float]:
        summary = self.gft_capacity_summary_cached(state)
        risks = {goal: 0.0 for goal in self.goals_sorted}
        for row in summary["regions"]:
            raw_slack = row["raw_slack"]
            stress = float(row["capacity_debt"]) + max(0.0, float(-raw_slack))
            risk = _ratio(stress, max(int(row["goal_count"]), 1))
            for goal_raw in row["goals"]:
                risks[tuple(goal_raw)] = risk
        return risks

    def gft_transport_summary_cached(self, state: SokobanState) -> dict[str, Any]:
        cached = self._gft_transport_cache.get(state)
        if cached is not None:
            return cached

        boxes = tuple(state.boxes)
        n = len(boxes)
        cost = np.full((n, n), np.inf, dtype=np.float64)
        turn_matrix = np.zeros((n, n), dtype=np.float64)
        access_matrix = np.zeros((n, n), dtype=np.float64)
        zone_matrix = np.zeros((n, n), dtype=np.float64)
        access_risks = self._access_risks_by_box(state)
        zone_risks = self._zone_risks_by_goal(state)
        turn_distances = self.gft_goal_turn_distances() if self.gft_transport_cost_weights["turn"] != 0.0 else None

        for box_idx, box in enumerate(boxes):
            for goal_idx, goal in enumerate(self.goals_sorted):
                base = float(self.pull_distances[goal_idx].get(box, math.inf))
                if not math.isfinite(base):
                    continue
                turns = 0.0
                if turn_distances is not None:
                    turns = float(turn_distances[goal_idx].get(box, 0))
                access_risk = float(access_risks.get(box, 1.0))
                zone_risk = float(zone_risks.get(goal, 0.0))
                turn_matrix[box_idx, goal_idx] = turns
                access_matrix[box_idx, goal_idx] = access_risk
                zone_matrix[box_idx, goal_idx] = zone_risk
                cost[box_idx, goal_idx] = (
                    base
                    + self.gft_transport_cost_weights["turn"] * turns
                    + self.gft_transport_cost_weights["access"] * access_risk
                    + self.gft_transport_cost_weights["zone"] * zone_risk
                )

        transport_cost, assignment = self._min_cost_assignment(cost)
        feasible = bool(math.isfinite(transport_cost))
        summary = {
            "transport_cost": float(transport_cost),
            "matching_lower_bound": float(transport_cost),
            "matching_assignment": [int(idx) for idx in assignment],
            "feasible": feasible,
            "cost_formula": "d_g(b_i)+lambda_turn*turns+lambda_access*access_risk+lambda_zone*zone_risk",
            "transport_cost_weights": dict(self.gft_transport_cost_weights),
            "default_reduces_to_matching_lower_bound": self._transport_weights_are_base_lb(),
            "cost_matrix": cost.tolist(),
            "turn_matrix": turn_matrix.tolist(),
            "access_risk_matrix": access_matrix.tolist(),
            "zone_risk_matrix": zone_matrix.tolist(),
            "access_risk_by_box": {str(list(box)): float(risk) for box, risk in access_risks.items()},
            "zone_risk_by_goal": {str(list(goal)): float(risk) for goal, risk in zone_risks.items()},
            "lower_is_better": True,
        }
        self._gft_transport_cache[state] = summary
        return summary

    def gft_transport_progress_cached(self, state: SokobanState) -> float:
        if self._transport_weights_are_base_lb():
            return float(self.gft_matching_lower_bound_cached(state.boxes))
        return float(self.gft_transport_summary_cached(state)["transport_cost"])

    def _floor_neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        neighbors = []
        for direction in DIRECTIONS:
            nxt = add_cells(cell, direction)
            if self.level.is_floor(nxt):
                neighbors.append(nxt)
        return tuple(neighbors)

    def _articulation_cells(self) -> set[Cell]:
        floor = set(self.floor_cells)
        adjacency = {cell: [nxt for nxt in self._floor_neighbors(cell) if nxt in floor] for cell in floor}
        visited: set[Cell] = set()
        discovery: dict[Cell, int] = {}
        low: dict[Cell, int] = {}
        parent: dict[Cell, Cell | None] = {}
        articulation: set[Cell] = set()
        time = 0

        def dfs(cell: Cell) -> None:
            nonlocal time
            visited.add(cell)
            discovery[cell] = time
            low[cell] = time
            time += 1
            child_count = 0
            for nxt in adjacency[cell]:
                if nxt not in visited:
                    parent[nxt] = cell
                    child_count += 1
                    dfs(nxt)
                    low[cell] = min(low[cell], low[nxt])
                    if parent.get(cell) is None and child_count > 1:
                        articulation.add(cell)
                    if parent.get(cell) is not None and low[nxt] >= discovery[cell]:
                        articulation.add(cell)
                elif nxt != parent.get(cell):
                    low[cell] = min(low[cell], discovery[nxt])

        for cell in self.floor_cells:
            if cell not in visited:
                parent[cell] = None
                dfs(cell)
        return articulation

    def _build_capacity_regions(self) -> tuple[dict[str, Any], ...]:
        articulation = self._articulation_cells()
        regular_floor = set(self.floor_cells) - articulation
        seen: set[Cell] = set()
        raw_regions: list[tuple[Cell, ...]] = []
        for start in sorted(regular_floor):
            if start in seen:
                continue
            queue: deque[Cell] = deque([start])
            seen.add(start)
            cells: list[Cell] = []
            while queue:
                cell = queue.popleft()
                cells.append(cell)
                for nxt in self._floor_neighbors(cell):
                    if nxt in regular_floor and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            raw_regions.append(tuple(sorted(cells)))

        goal_regions: list[dict[str, Any]] = []
        for cells in raw_regions:
            cell_set = set(cells)
            goals = tuple(sorted(cell for cell in cells if cell in self.level.goals))
            if not goals:
                continue
            entries = tuple(
                sorted(
                    entry
                    for cell in cells
                    for entry in self._floor_neighbors(cell)
                    if entry in articulation
                )
            )
            goal_regions.append(
                {
                    "region_id": len(goal_regions),
                    "cells": cells,
                    "goals": goals,
                    "entries": entries,
                    "cell_count": len(cell_set),
                }
            )

        for goal in sorted(cell for cell in self.level.goals if cell in articulation):
            entries = tuple(sorted(nxt for nxt in self._floor_neighbors(goal) if nxt != goal))
            goal_regions.append(
                {
                    "region_id": len(goal_regions),
                    "cells": (goal,),
                    "goals": (goal,),
                    "entries": entries,
                    "cell_count": 1,
                }
            )

        if not goal_regions:
            goals = tuple(sorted(self.level.goals))
            goal_regions.append(
                {
                    "region_id": 0,
                    "cells": tuple(sorted(self.floor_cells)),
                    "goals": goals,
                    "entries": (),
                    "cell_count": len(self.floor_cells),
                }
            )
        return tuple(goal_regions)

    def _can_reach_region_goal_static(self, box: Cell, goals: Sequence[Cell]) -> bool:
        feasible = self._box_feasible_goals_static(box)
        return any(goal in feasible for goal in goals)

    def _box_feasible_goals_static(self, box: Cell) -> frozenset[Cell]:
        return frozenset(
            goal
            for goal, dist in zip(self.goals_sorted, self.pull_distances)
            if dist.get(box, math.inf) < math.inf
        )

    def gft_capacity_summary_cached(self, state: SokobanState) -> dict[str, Any]:
        cached = self._gft_capacity_cache.get(state)
        if cached is not None:
            return cached

        box_set = set(state.boxes)
        region_rows: list[dict[str, Any]] = []
        total_debt = 0
        min_raw_slack = math.inf
        for region in self.gft_capacity_regions:
            cells = set(region["cells"])
            goals = tuple(region["goals"])
            goal_set = set(goals)
            boxes_in_region = tuple(sorted(box for box in state.boxes if box in cells))
            boxes_on_region_goals = tuple(sorted(box for box in boxes_in_region if box in goals))
            committed = []
            for box in state.boxes:
                feasible = self._box_feasible_goals_static(box)
                if feasible and feasible <= goal_set:
                    committed.append(box)
            committed_boxes = tuple(sorted(committed))
            wrong = []
            for box in boxes_in_region:
                feasible = self._box_feasible_goals_static(box)
                if box not in goals and feasible and feasible.isdisjoint(goal_set):
                    wrong.append(box)
            wrong_boxes = tuple(sorted(wrong))
            blocked_entries = tuple(
                sorted(entry for entry in region["entries"] if entry in box_set and entry not in self.level.goals)
            )
            goal_count = len(goals)
            box_count = len(boxes_in_region)
            committed_count = len(committed_boxes)
            overcapacity = max(0, committed_count - goal_count)
            raw_slack = goal_count - committed_count - len(blocked_entries) - len(wrong_boxes)
            debt = int(overcapacity + len(blocked_entries) + len(wrong_boxes))
            total_debt += debt
            min_raw_slack = min(min_raw_slack, raw_slack)
            region_rows.append(
                {
                    "region_id": int(region["region_id"]),
                    "cell_count": int(region["cell_count"]),
                    "goal_count": int(goal_count),
                    "box_count": int(box_count),
                    "committed_box_count": int(committed_count),
                    "boxes_on_region_goals": int(len(boxes_on_region_goals)),
                    "entry_count": int(len(region["entries"])),
                    "blocked_entry_count": int(len(blocked_entries)),
                    "wrong_box_count": int(len(wrong_boxes)),
                    "overcapacity": int(overcapacity),
                    "raw_slack": int(raw_slack),
                    "capacity_debt": int(debt),
                    "goals": [list(cell) for cell in goals],
                    "blocked_entries": [list(cell) for cell in blocked_entries],
                    "wrong_boxes": [list(cell) for cell in wrong_boxes],
                }
            )

        summary = {
            "capacity_health": float(-total_debt),
            "capacity_debt": int(total_debt),
            "min_raw_slack": None if math.isinf(min_raw_slack) else int(min_raw_slack),
            "stressed_region_count": int(sum(1 for row in region_rows if row["capacity_debt"] > 0)),
            "region_count": int(len(region_rows)),
            "decomposition": "floor_components_after_cut_vertices",
            "higher_is_better": True,
            "hard_constraint": False,
            "regions": region_rows,
        }
        self._gft_capacity_cache[state] = summary
        return summary

    def gft_capacity_health_cached(self, state: SokobanState) -> float:
        return float(self.gft_capacity_summary_cached(state)["capacity_health"])

    def _build_ordering_constraints(self) -> dict[str, Any]:
        goal_edges: list[dict[str, Any]] = []
        entry_edges: list[dict[str, Any]] = []

        for region in self.gft_capacity_regions:
            region_cells = set(region["cells"])
            entries = tuple(region["entries"])
            goals = tuple(region["goals"])
            if entries:
                nodes = region_cells | set(entries)
                dist: dict[Cell, int] = {entry: 0 for entry in entries}
                queue: deque[Cell] = deque(entries)
                while queue:
                    cell = queue.popleft()
                    for nxt in self._floor_neighbors(cell):
                        if nxt not in nodes or nxt in dist:
                            continue
                        dist[nxt] = dist[cell] + 1
                        queue.append(nxt)

                goals_by_depth: dict[int, list[Cell]] = {}
                for goal in goals:
                    if goal in dist:
                        goals_by_depth.setdefault(dist[goal], []).append(goal)
                depths = sorted(goals_by_depth)
                for front_depth, deep_depth in zip(depths, depths[1:]):
                    for front_goal in sorted(goals_by_depth[front_depth]):
                        for deep_goal in sorted(goals_by_depth[deep_depth]):
                            goal_edges.append(
                                {
                                    "region_id": int(region["region_id"]),
                                    "pre_goal": deep_goal,
                                    "post_goal": front_goal,
                                    "pre_depth": int(deep_depth),
                                    "post_depth": int(front_depth),
                                    "reason": "deeper_goal_before_shallow_goal",
                                }
                            )

            for entry in entries:
                entry_edges.append(
                    {
                        "region_id": int(region["region_id"]),
                        "entry": entry,
                        "goals": goals,
                        "reason": "region_goals_before_entry_block",
                    }
                )

        return {
            "builder": "entry_depth_goal_partial_order_v1",
            "goal_order_edges": tuple(goal_edges),
            "entry_order_edges": tuple(entry_edges),
            "cycle_count": 0,
        }

    def gft_ordering_summary_cached(self, state: SokobanState) -> dict[str, Any]:
        cached = self._gft_ordering_cache.get(state)
        if cached is not None:
            return cached

        box_set = set(state.boxes)
        goal_violations: list[dict[str, Any]] = []
        for edge in self.gft_ordering_constraints["goal_order_edges"]:
            pre_goal = edge["pre_goal"]
            post_goal = edge["post_goal"]
            if post_goal in box_set and pre_goal not in box_set:
                goal_violations.append(
                    {
                        "region_id": int(edge["region_id"]),
                        "pre_goal": list(pre_goal),
                        "post_goal": list(post_goal),
                        "pre_depth": int(edge["pre_depth"]),
                        "post_depth": int(edge["post_depth"]),
                        "reason": edge["reason"],
                    }
                )

        entry_violations: list[dict[str, Any]] = []
        for edge in self.gft_ordering_constraints["entry_order_edges"]:
            entry = edge["entry"]
            if entry not in box_set or entry in self.level.goals:
                continue
            missing_goals = tuple(goal for goal in edge["goals"] if goal not in box_set)
            if missing_goals:
                entry_violations.append(
                    {
                        "region_id": int(edge["region_id"]),
                        "entry": list(entry),
                        "missing_goal_count": int(len(missing_goals)),
                        "missing_goals": [list(goal) for goal in missing_goals],
                        "reason": edge["reason"],
                    }
                )

        violated_edges = len(goal_violations) + len(entry_violations)
        cycle_count = int(self.gft_ordering_constraints["cycle_count"])
        ordering_debt = violated_edges + cycle_count
        summary = {
            "ordering_debt": int(ordering_debt),
            "violated_edge_count": int(violated_edges),
            "cycle_count": cycle_count,
            "goal_order_edge_count": int(len(self.gft_ordering_constraints["goal_order_edges"])),
            "entry_order_edge_count": int(len(self.gft_ordering_constraints["entry_order_edges"])),
            "constraint_builder": self.gft_ordering_constraints["builder"],
            "higher_is_worse": True,
            "hard_constraint": False,
            "goal_order_violations": goal_violations,
            "entry_violations": entry_violations,
        }
        self._gft_ordering_cache[state] = summary
        return summary

    def gft_ordering_debt_cached(self, state: SokobanState) -> float:
        return float(self.gft_ordering_summary_cached(state)["ordering_debt"])

    def gft_commitment_summary_cached(self, state: SokobanState) -> dict[str, Any]:
        cached = self._gft_commitment_cache.get(state)
        if cached is not None:
            return cached

        boxes = tuple(state.boxes)
        goals = self.goals_sorted
        n = len(boxes)
        if n == 0:
            summary = {
                "commitment_confidence": 1.0,
                "assignment_entropy": 0.0,
                "normalized_entropy": 0.0,
                "feasible": True,
                "sinkhorn_beta": float(self.gft_commitment_beta),
                "sinkhorn_iterations": 0,
                "transport_cost_weights": dict(self.gft_transport_cost_weights),
                "mean_best_goal_probability": 1.0,
                "min_best_goal_probability": 1.0,
                "cost_matrix": [],
                "assignment_matrix": [],
                "best_goal_indices": [],
                "higher_is_better": True,
                "hard_constraint": False,
            }
            self._gft_commitment_cache[state] = summary
            return summary

        transport = self.gft_transport_summary_cached(state)
        cost = np.asarray(transport["cost_matrix"], dtype=np.float64)

        feasible = bool(np.all(np.isfinite(cost).any(axis=1)) and np.all(np.isfinite(cost).any(axis=0)))
        beta = self.gft_commitment_beta
        iterations = self.gft_sinkhorn_iterations
        if feasible:
            finite_cost = np.where(np.isfinite(cost), cost, np.inf)
            logits = np.where(np.isfinite(finite_cost), -beta * finite_cost, -np.inf)
            max_logit = np.max(logits[np.isfinite(logits)])
            matrix = np.where(np.isfinite(logits), np.exp(logits - max_logit), 0.0)
            for _ in range(iterations):
                row_sums = matrix.sum(axis=1, keepdims=True)
                matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0.0)
                col_sums = matrix.sum(axis=0, keepdims=True)
                matrix = np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums > 0.0)
            row_sums = matrix.sum(axis=1, keepdims=True)
            assignment = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0.0)
            positive = assignment[assignment > 0.0]
            entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
            max_entropy = float(n * math.log(n)) if n > 1 else 1.0
            normalized_entropy = 0.0 if n <= 1 else _clamp(entropy / max_entropy, 0.0, 1.0)
            confidence = 1.0 - normalized_entropy
            best_probs = np.max(assignment, axis=1)
            best_goal_indices = np.argmax(assignment, axis=1).astype(int).tolist()
        else:
            assignment = np.zeros((n, n), dtype=np.float64)
            entropy = 0.0
            normalized_entropy = 1.0
            confidence = 0.0
            best_probs = np.zeros(n, dtype=np.float64)
            best_goal_indices = [-1 for _ in boxes]

        cost_rows = [
            [None if not math.isfinite(float(value)) else int(value) for value in row]
            for row in cost.tolist()
        ]
        summary = {
            "commitment_confidence": float(confidence),
            "assignment_entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "feasible": bool(feasible),
            "sinkhorn_beta": float(beta),
            "sinkhorn_iterations": int(iterations if feasible else 0),
            "transport_cost_weights": dict(self.gft_transport_cost_weights),
            "mean_best_goal_probability": float(np.mean(best_probs)) if best_probs.size else 0.0,
            "min_best_goal_probability": float(np.min(best_probs)) if best_probs.size else 0.0,
            "cost_matrix": cost_rows,
            "assignment_matrix": assignment.round(6).tolist(),
            "best_goal_indices": best_goal_indices,
            "higher_is_better": True,
            "hard_constraint": False,
        }
        self._gft_commitment_cache[state] = summary
        return summary

    def gft_commitment_confidence_cached(self, state: SokobanState) -> float:
        return float(self.gft_commitment_summary_cached(state)["commitment_confidence"])

    def gft_deadlock_summary_cached(self, state: SokobanState) -> dict[str, Any]:
        cached = self._gft_deadlock_cache.get(state)
        if cached is not None:
            return cached

        features = self.gft_abstract_features_cached(state)
        hard = int(features.dead_square_boxes + features.frozen_boxes + int(features.matching_deadlock))
        max_pushes = max(self.level.n_boxes * len(ACTION_NAMES), 1)
        mobility_loss = _ratio(max_pushes - features.viable_push_count, max_pushes)
        access_loss = _ratio(max_pushes - features.legal_push_count, max_pushes)

        transport = self.gft_transport_summary_cached(state)
        commitment = self.gft_commitment_summary_cached(state)
        if not transport["feasible"]:
            matching_fragility = 1.0
        else:
            matching_fragility = _clamp(float(commitment["normalized_entropy"]), 0.0, 1.0)

        capacity = self.gft_capacity_summary_cached(state)
        capacity_stress = _ratio(float(capacity["capacity_debt"]), max(self.level.n_boxes, 1))

        ordering = self.gft_ordering_summary_cached(state)
        ordering_edges = max(
            int(ordering["goal_order_edge_count"]) + int(ordering["entry_order_edge_count"]),
            1,
        )
        order_violation = _ratio(float(ordering["ordering_debt"]), ordering_edges)

        soft_components = {
            "mobility_loss": float(mobility_loss),
            "matching_fragility": float(matching_fragility),
            "capacity_stress": float(capacity_stress),
            "access_loss": float(access_loss),
            "order_violation": float(order_violation),
        }
        soft = sum(self.gft_soft_deadlock_weights[key] * value for key, value in soft_components.items())
        summary = {
            "deadlock_debt": float(hard + soft),
            "hard_deadlock_debt": int(hard),
            "soft_deadlock_pressure": float(soft),
            "soft_deadlock_components": soft_components,
            "soft_deadlock_weights": dict(self.gft_soft_deadlock_weights),
            "dead_square_boxes": int(features.dead_square_boxes),
            "static_frozen_boxes": int(features.frozen_boxes),
            "matching_deadlock": bool(features.matching_deadlock),
            "higher_is_worse": True,
        }
        self._gft_deadlock_cache[state] = summary
        return summary

    def gft_deadlock_debt_cached(self, state: SokobanState) -> float:
        return float(self.gft_deadlock_summary_cached(state)["deadlock_debt"])

    def gft_state(self, state: SokobanState, *, active_only: bool = False) -> GFTStateReadout:
        stocks = tuple(stock for stock in self.gft_resource_stocks if stock.active) if active_only else self.gft_resource_stocks
        return GFTStateReadout(
            ledger_key=self.gft_canonical_key_cached(state),
            resources=tuple(stock.compute(self, state) for stock in stocks),
        )

    def gft_delta(self, state: SokobanState, child_state: SokobanState) -> dict[str, Any]:
        before = {row.code: row for row in self.gft_state(state).resources}
        after = {row.code: row for row in self.gft_state(child_state).resources}
        delta: dict[str, Any] = {}
        for code, prev in before.items():
            nxt = after[code]
            raw_delta = None
            if math.isfinite(prev.value) and math.isfinite(nxt.value):
                raw_delta = float(nxt.value - prev.value)
            delta[code] = {
                "before": prev.to_dict(),
                "after": nxt.to_dict(),
                "delta": _jsonable(raw_delta),
            }
        return delta

    def gft_resource_vector(self, state: SokobanState) -> dict[str, float]:
        """Return the scalar GFT stock vector ``z_t=Phi(y_t)``."""

        cached = self._gft_resource_vector_cache.get(state)
        if cached is None:
            cached = {row.code: float(row.value) for row in self.gft_state(state).resources}
            self._gft_resource_vector_cache[state] = cached
        return dict(cached)

    def gft_bottleneck_summary(self, state: SokobanState) -> dict[str, Any]:
        """Identify currently stressed resource stocks for the action relevance term."""

        features = self.gft_abstract_features_cached(state)
        deadlock = self.gft_deadlock_summary_cached(state)
        capacity = self.gft_capacity_summary_cached(state)
        ordering = self.gft_ordering_summary_cached(state)
        commitment = self.gft_commitment_summary_cached(state)
        transport = self.gft_transport_summary_cached(state)
        max_pushes = max(self.level.n_boxes * len(ACTION_NAMES), 1)
        codes: set[str] = set()
        reasons: dict[str, str] = {}

        if math.isfinite(float(transport["transport_cost"])) and float(transport["transport_cost"]) > 0.0:
            codes.add("P")
            reasons["P"] = "remaining_transport_work_positive"
        if features.viable_push_count <= max(self.level.n_boxes, 1):
            codes.add("M")
            reasons["M"] = "low_viable_push_count"
        if _ratio(max_pushes - features.legal_push_count, max_pushes) >= 0.5:
            codes.add("A")
            reasons["A"] = "low_reachable_push_support"
        if int(capacity["capacity_debt"]) > 0:
            codes.add("K")
            reasons["K"] = "positive_capacity_debt"
        if int(deadlock["hard_deadlock_debt"]) > 0 or float(deadlock["soft_deadlock_pressure"]) >= 1.0:
            codes.add("D")
            reasons["D"] = "hard_or_soft_deadlock_pressure"
        if int(ordering["ordering_debt"]) > 0:
            codes.add("Q")
            reasons["Q"] = "positive_ordering_debt"
        if float(commitment["normalized_entropy"]) >= 0.5:
            codes.add("C")
            reasons["C"] = "high_goal_assignment_entropy"

        return {
            "bottleneck_codes": sorted(codes),
            "reasons": reasons,
            "thresholds": {
                "M_low_viable_push_count": max(self.level.n_boxes, 1),
                "A_access_loss": 0.5,
                "D_soft_deadlock_pressure": 1.0,
                "C_normalized_entropy": 0.5,
            },
        }

    def gft_resource_flow(self, state: SokobanState, child_state: SokobanState) -> dict[str, Any]:
        """Return document-signed resource flows for one exact push transition."""

        before = self.gft_resource_vector(state)
        after = self.gft_resource_vector(child_state)
        stock_delta = {code: _finite_delta(before[code], after[code]) for code in before}
        p_flow = None
        if math.isfinite(before["P"]) and math.isfinite(after["P"]):
            p_flow = float(before["P"] - after["P"])
        bounded_p_flow = _matching_progress(before["P"], after["P"])

        parent_commitment = self.gft_commitment_summary_cached(state)
        child_commitment = self.gft_commitment_summary_cached(child_state)
        entropy_reduction = float(parent_commitment["assignment_entropy"] - child_commitment["assignment_entropy"])
        commitment_delta = float(after["C"] - before["C"]) if math.isfinite(before["C"]) and math.isfinite(after["C"]) else None

        flows = {
            "P": p_flow,
            "M": stock_delta["M"],
            "A": stock_delta["A"],
            "K": stock_delta["K"],
            "D": stock_delta["D"],
            "Q": stock_delta["Q"],
            "C": entropy_reduction,
            "I": None,
        }
        bounded_flows = dict(flows)
        bounded_flows["P"] = bounded_p_flow

        return {
            "stock_before": _jsonable(before),
            "stock_after": _jsonable(after),
            "stock_delta": _jsonable(stock_delta),
            "flows": _jsonable(flows),
            "bounded_flows": _jsonable(bounded_flows),
            "commitment_delta": _jsonable(commitment_delta),
            "flow_semantics": {
                "P": "P_t-P_t1, positive means remaining transport work decreased",
                "M": "M_t1-M_t, positive means future push freedom increased",
                "A": "A_t1-A_t, positive means reachable push support increased",
                "K": "K_t1-K_t, positive means region capacity health increased",
                "D": "D_t1-D_t, positive means deadlock debt increased",
                "Q": "Q_t1-Q_t, positive means ordering debt increased",
                "C": "H_C_t-H_C_t1, positive means assignment entropy decreased",
                "I": "action-dependent relevance, supplied by gft_action_relevance",
            },
        }

    def gft_action_impact_summary(self, state: SokobanState, child_state: SokobanState) -> dict[str, Any]:
        flow = self.gft_resource_flow(state, child_state)
        bounded = flow["bounded_flows"]
        stock_delta = flow["stock_delta"]
        impacted: set[str] = set()
        for code in ("P", "M", "A", "K", "D", "Q"):
            value = bounded[code] if code == "P" else stock_delta[code]
            if value is not None and abs(float(value)) > 1e-9:
                impacted.add(code)
        c_delta = flow["commitment_delta"]
        if c_delta is not None and abs(float(c_delta)) > 1e-9:
            impacted.add("C")
        return {
            "impact_codes": sorted(impacted),
            "flow": flow,
        }

    def gft_sd_reward(
        self,
        state: SokobanState,
        child_state: SokobanState,
        *,
        weights: Mapping[str, float] | None = None,
        push_cost: float = 0.05,
        goal_value: float = 0.0,
        dead_value: float = 0.0,
    ) -> dict[str, Any]:
        """Compute the document ``r_SD`` and optional terminal reward terms."""

        weights = _merge_float_weights(DEFAULT_GFT_SD_REWARD_WEIGHTS, weights)
        flow = self.gft_resource_flow(state, child_state)
        bounded = flow["bounded_flows"]
        f_p = float(bounded["P"])
        f_m = float(bounded["M"] or 0.0)
        f_a = float(bounded["A"] or 0.0)
        f_k = float(bounded["K"] or 0.0)
        f_d = float(bounded["D"] or 0.0)
        f_q = float(bounded["Q"] or 0.0)
        components = {
            "progress": weights["progress"] * f_p,
            "mobility": weights["mobility"] * f_m,
            "access": weights["access"] * f_a,
            "capacity": weights["capacity"] * f_k,
            "deadlock": -weights["deadlock"] * f_d,
            "ordering": -weights["ordering"] * f_q,
            "push_cost": -float(push_cost),
        }
        r_sd = float(sum(components.values()))
        r_goal = float(goal_value) if self.is_solved(child_state) and not self.is_solved(state) else 0.0
        r_dead = float(dead_value) if self.is_deadlocked(child_state) and not self.is_solved(child_state) else 0.0
        return {
            "r_sd": r_sd,
            "r_goal": r_goal,
            "r_dead": r_dead,
            "r_total": float(r_sd + r_goal + r_dead),
            "components": _jsonable(components),
            "weights": dict(weights),
            "push_cost": float(push_cost),
        }

    def gft_sd_value(
        self,
        state: SokobanState,
        *,
        weights: Mapping[str, float] | None = None,
        nonfinite_progress_penalty: float = 1_000_000.0,
    ) -> dict[str, Any]:
        """Compute the explicit ``V_SD(z)`` term used by the future TVA interface."""

        weights = _merge_float_weights(DEFAULT_GFT_SD_VALUE_WEIGHTS, weights)
        vector = self.gft_resource_vector(state)
        p = float(vector["P"]) if math.isfinite(float(vector["P"])) else float(nonfinite_progress_penalty)
        components = {
            "progress": -weights["progress"] * p,
            "mobility": weights["mobility"] * float(vector["M"]),
            "access": weights["access"] * float(vector["A"]),
            "capacity": weights["capacity"] * float(vector["K"]),
            "deadlock": -weights["deadlock"] * float(vector["D"]),
            "ordering": -weights["ordering"] * float(vector["Q"]),
            "commitment": weights["commitment"] * float(vector["C"]),
        }
        return {
            "value": float(sum(components.values())),
            "components": _jsonable(components),
            "resource_vector": _jsonable(vector),
            "weights": dict(weights),
            "nonfinite_progress_penalty": float(nonfinite_progress_penalty),
        }

    def gft_tva_feature_dict(self, state: SokobanState) -> dict[str, Any]:
        """Stable GFT feature contract for future TVA training."""

        cached = self._gft_tva_feature_cache.get(state)
        if cached is not None:
            return {
                "feature_names": list(cached["feature_names"]),
                "features": dict(cached["features"]),
                "contract": cached["contract"],
            }

        vector = self.gft_resource_vector(state)
        transport = self.gft_transport_summary_cached(state)
        commitment = self.gft_commitment_summary_cached(state)
        deadlock = self.gft_deadlock_summary_cached(state)
        capacity = self.gft_capacity_summary_cached(state)
        ordering = self.gft_ordering_summary_cached(state)
        features = {
            "P": float(vector["P"]) if math.isfinite(float(vector["P"])) else 1_000_000.0,
            "M": float(vector["M"]),
            "A": float(vector["A"]),
            "K": float(vector["K"]),
            "D": float(vector["D"]),
            "Q": float(vector["Q"]),
            "C": float(vector["C"]),
            "boxes_on_goals": float(self.boxes_on_goals(state)),
            "transport_feasible": float(bool(transport["feasible"])),
            "assignment_entropy": float(commitment["assignment_entropy"]),
            "assignment_normalized_entropy": float(commitment["normalized_entropy"]),
            "capacity_debt": float(capacity["capacity_debt"]),
            "ordering_debt": float(ordering["ordering_debt"]),
            "hard_deadlock_debt": float(deadlock["hard_deadlock_debt"]),
            "soft_deadlock_pressure": float(deadlock["soft_deadlock_pressure"]),
        }
        for key, value in deadlock["soft_deadlock_components"].items():
            features[f"soft_deadlock_{key}"] = float(value)
        payload = {
            "feature_names": list(features.keys()),
            "features": _jsonable(features),
            "contract": "gft_tva_feature_vector_v1",
        }
        self._gft_tva_feature_cache[state] = payload
        return {
            "feature_names": list(payload["feature_names"]),
            "features": dict(payload["features"]),
            "contract": payload["contract"],
        }

    def gft_tva_features(self, state: SokobanState) -> np.ndarray:
        feature_dict = self.gft_tva_feature_dict(state)["features"]
        return np.asarray([feature_dict[key] for key in feature_dict], dtype=np.float64)

    def gft_action_relevance(
        self,
        state: SokobanState,
        action: SokobanPushAction,
        child_state: SokobanState | None = None,
        *,
        visited_keys: Iterable[tuple] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute the current ``I_t`` prior score for one candidate push."""

        child_state = child_state if child_state is not None else self.step_grid_flow(state, action)
        weights = {**DEFAULT_GFT_PRIOR_WEIGHTS, **dict(weights or {})}
        visited = set(visited_keys or ())

        parent_lb = self.gft_transport_progress_cached(state)
        child_lb = self.gft_transport_progress_cached(child_state)
        progress = _matching_progress(parent_lb, child_lb)
        goal_delta = self.boxes_on_goals(child_state) - self.boxes_on_goals(state)

        mobility_delta = 0.0
        if weights["mobility"] != 0.0:
            max_pushes = max(self.level.n_boxes * len(ACTION_NAMES), 1)
            parent_legal = len(self.legal_flows(state))
            child_legal = len(self.legal_flows(child_state))
            mobility_delta = (child_legal - parent_legal) / max_pushes

        access_delta = 0.0
        if weights["access"] != 0.0:
            floor_count = max(len(self.floor_cells), 1)
            access_delta = (len(self.reachable_cells(child_state)) - len(self.reachable_cells(state))) / floor_count

        deadlock_delta = 0.0
        if weights["deadlock"] != 0.0:
            deadlock_delta = self.gft_deadlock_debt_cached(child_state) - self.gft_deadlock_debt_cached(state)

        revisit = 1.0 if self.gft_canonical_key_cached(child_state) in visited else 0.0
        capacity_delta = 0.0
        if weights["capacity"] != 0.0:
            capacity_delta = self.gft_capacity_health_cached(child_state) - self.gft_capacity_health_cached(state)
        ordering_delta = 0.0
        if weights["ordering"] != 0.0:
            ordering_delta = self.gft_ordering_debt_cached(child_state) - self.gft_ordering_debt_cached(state)
        commitment_delta = 0.0
        if weights["commitment"] != 0.0:
            commitment_delta = (
                self.gft_commitment_confidence_cached(child_state) - self.gft_commitment_confidence_cached(state)
            )
        bottleneck_overlap = 0.0
        bottleneck_codes: list[str] = []
        impact_codes: list[str] = []
        if weights["bottleneck"] != 0.0:
            bottleneck = self.gft_bottleneck_summary(state)
            impact = self.gft_action_impact_summary(state, child_state)
            bottleneck_codes = bottleneck["bottleneck_codes"]
            impact_codes = impact["impact_codes"]
            bottleneck_overlap = float(len(set(bottleneck_codes) & set(impact_codes)))

        score = (
            weights["progress"] * progress
            + weights["goal"] * goal_delta
            + weights["mobility"] * mobility_delta
            + weights["access"] * access_delta
            + weights["capacity"] * capacity_delta
            - weights["deadlock"] * deadlock_delta
            - weights["ordering"] * ordering_delta
            + weights["commitment"] * commitment_delta
            + weights["bottleneck"] * bottleneck_overlap
            - weights["revisit"] * revisit
        )

        return {
            "action": {
                "box_index": int(action.box_index),
                "direction": int(action.direction),
                "flat_index": int(action.flat_index),
            },
            "score": float(score),
            "components": {
                "progress_delta": float(progress),
                "goal_delta": int(goal_delta),
                "mobility_delta": float(mobility_delta),
                "access_delta": float(access_delta),
                "capacity_delta": float(capacity_delta),
                "deadlock_delta": float(deadlock_delta),
                "ordering_delta": float(ordering_delta),
                "commitment_delta": float(commitment_delta),
                "bottleneck_overlap": float(bottleneck_overlap),
                "bottleneck_codes": bottleneck_codes,
                "impact_codes": impact_codes,
                "revisit": float(revisit),
            },
            "weights": dict(weights),
        }

    def gft_action_prior_rows(
        self,
        state: SokobanState,
        candidates: Sequence[tuple[SokobanPushAction, SokobanState]],
        *,
        visited_keys: Iterable[tuple] | None = None,
        weights: Mapping[str, float] | None = None,
        temperature: float = 1.0,
        uniform_mix: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Return prior probabilities for candidates in input order."""

        if not candidates:
            return []
        temperature = max(float(temperature), 1e-9)
        uniform_mix = _clamp(float(uniform_mix), 0.0, 1.0)
        weights = {**DEFAULT_GFT_PRIOR_WEIGHTS, **dict(weights or {})}
        visited = set(visited_keys or ())
        parent_lb = self.gft_transport_progress_cached(state)
        parent_goals = self.boxes_on_goals(state)
        use_mobility = weights["mobility"] != 0.0
        use_access = weights["access"] != 0.0
        use_capacity = weights["capacity"] != 0.0
        use_deadlock = weights["deadlock"] != 0.0
        use_ordering = weights["ordering"] != 0.0
        use_commitment = weights["commitment"] != 0.0
        use_bottleneck = weights["bottleneck"] != 0.0
        max_pushes = max(self.level.n_boxes * len(ACTION_NAMES), 1)
        floor_count = max(len(self.floor_cells), 1)
        parent_legal = len(self.legal_flows(state)) if use_mobility else 0
        parent_reachable = len(self.reachable_cells(state)) if use_access else 0
        parent_d = 0.0
        if use_deadlock:
            parent_d = self.gft_deadlock_debt_cached(state)
        parent_capacity = self.gft_capacity_health_cached(state) if use_capacity else 0.0
        parent_ordering = self.gft_ordering_debt_cached(state) if use_ordering else 0.0
        parent_commitment = self.gft_commitment_confidence_cached(state) if use_commitment else 0.0
        parent_bottleneck_codes = self.gft_bottleneck_summary(state)["bottleneck_codes"] if use_bottleneck else []

        rows: list[dict[str, Any]] = []
        for action, child in candidates:
            child_lb = self.gft_transport_progress_cached(child)
            progress = _matching_progress(parent_lb, child_lb)
            goal_delta = self.boxes_on_goals(child) - parent_goals
            mobility_delta = 0.0
            if use_mobility:
                mobility_delta = (len(self.legal_flows(child)) - parent_legal) / max_pushes
            access_delta = 0.0
            if use_access:
                access_delta = (len(self.reachable_cells(child)) - parent_reachable) / floor_count
            deadlock_delta = 0.0
            if use_deadlock:
                child_d = self.gft_deadlock_debt_cached(child)
                deadlock_delta = float(child_d - parent_d)
            revisit = 1.0 if self.gft_canonical_key_cached(child) in visited else 0.0
            capacity_delta = 0.0
            if use_capacity:
                capacity_delta = self.gft_capacity_health_cached(child) - parent_capacity
            ordering_delta = 0.0
            if use_ordering:
                ordering_delta = self.gft_ordering_debt_cached(child) - parent_ordering
            commitment_delta = 0.0
            if use_commitment:
                commitment_delta = self.gft_commitment_confidence_cached(child) - parent_commitment
            bottleneck_overlap = 0.0
            impact_codes: list[str] = []
            if use_bottleneck:
                impact_codes = self.gft_action_impact_summary(state, child)["impact_codes"]
                bottleneck_overlap = float(len(set(parent_bottleneck_codes) & set(impact_codes)))
            score = (
                weights["progress"] * progress
                + weights["goal"] * goal_delta
                + weights["mobility"] * mobility_delta
                + weights["access"] * access_delta
                + weights["capacity"] * capacity_delta
                - weights["deadlock"] * deadlock_delta
                - weights["ordering"] * ordering_delta
                + weights["commitment"] * commitment_delta
                + weights["bottleneck"] * bottleneck_overlap
                - weights["revisit"] * revisit
            )
            rows.append(
                {
                    "action": {
                        "box_index": int(action.box_index),
                        "direction": int(action.direction),
                        "flat_index": int(action.flat_index),
                    },
                    "score": float(score),
                    "components": {
                        "progress_delta": float(progress),
                        "goal_delta": int(goal_delta),
                        "mobility_delta": float(mobility_delta),
                        "access_delta": float(access_delta),
                        "capacity_delta": float(capacity_delta),
                        "deadlock_delta": float(deadlock_delta),
                        "ordering_delta": float(ordering_delta),
                        "commitment_delta": float(commitment_delta),
                        "bottleneck_overlap": float(bottleneck_overlap),
                        "bottleneck_codes": parent_bottleneck_codes,
                        "impact_codes": impact_codes,
                        "revisit": float(revisit),
                    },
                    "weights": dict(weights),
                }
            )
        logits = np.asarray([row["score"] for row in rows], dtype=np.float64) / temperature
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        denom = float(np.sum(probs))
        uniform = 1.0 / len(rows)
        probs = probs / denom if denom > 0.0 else np.full(len(rows), uniform)
        for row, prob in zip(rows, probs):
            row["prior"] = float((1.0 - uniform_mix) * prob + uniform_mix * uniform)
        return rows
