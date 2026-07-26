"""Goal-directed SD world model for Sokoban."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .env import Cell, DIRECTIONS, SokobanLevel, SokobanState, add_cells


INF_COST = 10**9


@dataclass(frozen=True)
class SokobanPushAction:
    """Push-level macro action over the state's sorted box tuple."""

    box_index: int
    direction: int

    @property
    def flat_index(self) -> int:
        return self.box_index * len(DIRECTIONS) + self.direction


@dataclass(frozen=True)
class SokobanAbstractFeatures:
    """Task-oriented abstract state summary used by the SD mechanism."""

    reachable_count: int
    legal_push_count: int
    viable_push_count: int
    boxes_on_goals: int
    matching_lower_bound: float
    dead_square_boxes: int
    frozen_boxes: int
    matching_deadlock: bool
    solved: bool

    @property
    def deadlocked(self) -> bool:
        return self.dead_square_boxes > 0 or self.frozen_boxes > 0 or self.matching_deadlock

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["deadlocked"] = self.deadlocked
        return data


@dataclass(frozen=True)
class SokobanGridFlow:
    """One push as a unit flow over the static floor adjacency graph."""

    source: Cell
    target: Cell
    direction: int
    edge_index: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": list(self.source),
            "target": list(self.target),
            "direction": int(self.direction),
            "edge_index": int(self.edge_index),
        }


class SokobanPushDynamicsModel(nn.Module):
    """Exact push-level transition model for one Sokoban level."""

    def __init__(self, level: SokobanLevel):
        super().__init__()
        self.level = level
        self.n_boxes = level.n_boxes
        self.state_dim = 1 + self.n_boxes
        self.action_dim = self.n_boxes * len(DIRECTIONS)

    @property
    def initial_state(self) -> SokobanState:
        return self.level.initial_state

    def cell_to_index(self, cell: Cell) -> int:
        return cell[0] * self.level.width + cell[1]

    def index_to_cell(self, idx: int) -> Cell:
        return int(idx) // self.level.width, int(idx) % self.level.width

    def encode_tensor_state(self, state: SokobanState) -> torch.Tensor:
        values = [self.cell_to_index(state.player)]
        values.extend(self.cell_to_index(box) for box in state.boxes)
        return torch.tensor(values, dtype=torch.float32)

    def decode_tensor_state(self, tensor_state: torch.Tensor) -> SokobanState:
        values = tensor_state.detach().cpu().to(dtype=torch.long).view(-1).tolist()
        if len(values) != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {len(values)}")
        player = self.index_to_cell(values[0])
        boxes = tuple(self.index_to_cell(v) for v in values[1:])
        return SokobanState(player=player, boxes=boxes)

    def decode_action(self, action: int | torch.Tensor) -> SokobanPushAction:
        if isinstance(action, torch.Tensor):
            action = int(action.detach().cpu().view(-1)[0].item())
        return SokobanPushAction(
            box_index=int(action) // len(DIRECTIONS),
            direction=int(action) % len(DIRECTIONS),
        )

    def is_solved(self, state: SokobanState) -> bool:
        return set(state.boxes) == set(self.level.goals)

    def step_state(self, state: SokobanState, action: SokobanPushAction | int) -> SokobanState:
        if isinstance(action, int):
            action = self.decode_action(action)
        if action.box_index < 0 or action.box_index >= len(state.boxes):
            return state
        box = state.boxes[action.box_index]
        direction = DIRECTIONS[action.direction]
        player_stand = (box[0] - direction[0], box[1] - direction[1])
        box_target = add_cells(box, direction)
        boxes = set(state.boxes)
        if state.player != player_stand:
            return state
        if self.level.is_wall(box_target) or box_target in boxes:
            return state
        boxes.remove(box)
        boxes.add(box_target)
        return SokobanState(player=box, boxes=tuple(boxes))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Tensor adapter using ``[player_cell, box0_cell, ...]`` state encoding."""

        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        action = action.view(-1)
        if action.numel() == 1 and state.shape[0] > 1:
            action = action.expand(state.shape[0])
        out: list[torch.Tensor] = []
        for row, act in zip(state, action):
            next_state = self.step_state(self.decode_tensor_state(row), self.decode_action(act))
            out.append(self.encode_tensor_state(next_state).to(device=state.device))
        result = torch.stack(out, dim=0)
        return result[0] if single else result


class SokobanGoalDirectedSDModel(SokobanPushDynamicsModel):
    """Sokoban SD model with task-directed abstract mechanisms."""

    def __init__(
        self,
        level: SokobanLevel,
        deadlock_penalty: float = 100.0,
        matching_weight: float = 1.5,
        goal_weight: float = 4.0,
        mobility_weight: float = 0.05,
    ):
        super().__init__(level)
        self.deadlock_penalty = float(deadlock_penalty)
        self.matching_weight = float(matching_weight)
        self.goal_weight = float(goal_weight)
        self.mobility_weight = float(mobility_weight)
        self.goals_sorted = tuple(sorted(level.goals))
        self.floor_cells = tuple(
            (r, c)
            for r in range(level.height)
            for c in range(level.width)
            if level.is_floor((r, c))
        )
        self.pull_distances = self._compute_pull_distances()
        self.dead_squares = frozenset(
            cell
            for cell in self.floor_cells
            if cell not in self.level.goals
            and all(dist.get(cell, INF_COST) >= INF_COST for dist in self.pull_distances)
        )

    def _compute_pull_distances(self) -> Tuple[Dict[Cell, int], ...]:
        """Reverse-pull box distances to each goal on the static wall map."""

        all_distances: list[dict[Cell, int]] = []
        for goal in self.goals_sorted:
            dist = {goal: 0}
            queue: deque[Cell] = deque([goal])
            while queue:
                cell = queue.popleft()
                for direction in DIRECTIONS:
                    prev_box = (cell[0] - direction[0], cell[1] - direction[1])
                    player_stand = (cell[0] - 2 * direction[0], cell[1] - 2 * direction[1])
                    if not self.level.is_floor(prev_box) or not self.level.is_floor(player_stand):
                        continue
                    if prev_box in dist:
                        continue
                    dist[prev_box] = dist[cell] + 1
                    queue.append(prev_box)
            all_distances.append(dist)
        return tuple(all_distances)

    def reachable_cells(self, state: SokobanState) -> frozenset[Cell]:
        boxes = set(state.boxes)
        seen = {state.player}
        queue: deque[Cell] = deque([state.player])
        while queue:
            cell = queue.popleft()
            for direction in DIRECTIONS:
                nxt = add_cells(cell, direction)
                if nxt in seen or nxt in boxes or not self.level.is_floor(nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return frozenset(seen)

    def legal_pushes(self, state: SokobanState) -> Tuple[SokobanPushAction, ...]:
        reachable = self.reachable_cells(state)
        boxes = set(state.boxes)
        actions: list[SokobanPushAction] = []
        for box_index, box in enumerate(state.boxes):
            for direction_index, direction in enumerate(DIRECTIONS):
                player_stand = (box[0] - direction[0], box[1] - direction[1])
                box_target = add_cells(box, direction)
                if player_stand not in reachable:
                    continue
                if self.level.is_wall(box_target) or box_target in boxes:
                    continue
                actions.append(SokobanPushAction(box_index, direction_index))
        return tuple(actions)

    def step_state(self, state: SokobanState, action: SokobanPushAction | int) -> SokobanState:
        if isinstance(action, int):
            action = self.decode_action(action)
        if action not in set(self.legal_pushes(state)):
            return state
        box = state.boxes[action.box_index]
        direction = DIRECTIONS[action.direction]
        boxes = set(state.boxes)
        boxes.remove(box)
        boxes.add(add_cells(box, direction))
        return SokobanState(player=box, boxes=tuple(boxes))

    def matching_lower_bound(self, boxes: Iterable[Cell]) -> float:
        boxes = tuple(boxes)
        if not boxes:
            return 0.0
        dp = {0: 0.0}
        for box in boxes:
            next_dp: dict[int, float] = {}
            for mask, cost in dp.items():
                for goal_idx, dist in enumerate(self.pull_distances):
                    if mask & (1 << goal_idx):
                        continue
                    d = float(dist.get(box, INF_COST))
                    if d >= INF_COST:
                        continue
                    new_mask = mask | (1 << goal_idx)
                    next_dp[new_mask] = min(next_dp.get(new_mask, float("inf")), cost + d)
            dp = next_dp
            if not dp:
                return float("inf")
        return min(dp.values()) if dp else float("inf")

    def frozen_boxes(self, state: SokobanState) -> int:
        frozen = 0
        for box in state.boxes:
            if box in self.level.goals:
                continue
            # First implementation uses only static wall corners as hard
            # irreversible deadlocks. Other boxes can move, so treating them as
            # permanent walls would over-prune solvable Boxoban states.
            vertical = self.level.is_wall((box[0] - 1, box[1])) or self.level.is_wall((box[0] + 1, box[1]))
            horizontal = self.level.is_wall((box[0], box[1] - 1)) or self.level.is_wall((box[0], box[1] + 1))
            if vertical and horizontal:
                frozen += 1
        return frozen

    def is_deadlocked(self, state: SokobanState) -> bool:
        if self.is_solved(state):
            return False
        if not np.isfinite(self.matching_lower_bound(state.boxes)):
            return True
        for box in state.boxes:
            if box not in self.level.goals and box in self.dead_squares:
                return True
        return self.frozen_boxes(state) > 0

    def abstract_features(self, state: SokobanState) -> SokobanAbstractFeatures:
        legal = self.legal_pushes(state)
        reachable = self.reachable_cells(state)
        lb = self.matching_lower_bound(state.boxes)
        dead_square_boxes = sum(
            1 for box in state.boxes if box not in self.level.goals and box in self.dead_squares
        )
        viable = 0
        for action in legal:
            if not self.is_deadlocked(self.step_state(state, action)):
                viable += 1
        return SokobanAbstractFeatures(
            reachable_count=len(reachable),
            legal_push_count=len(legal),
            viable_push_count=viable,
            boxes_on_goals=sum(1 for box in state.boxes if box in self.level.goals),
            matching_lower_bound=lb,
            dead_square_boxes=dead_square_boxes,
            frozen_boxes=self.frozen_boxes(state),
            matching_deadlock=not np.isfinite(lb),
            solved=self.is_solved(state),
        )

    def transition_score(self, state: SokobanState, action: SokobanPushAction) -> float:
        next_state = self.step_state(state, action)
        if next_state == state:
            return -self.deadlock_penalty
        prev = self.abstract_features(state)
        nxt = self.abstract_features(next_state)
        if nxt.solved:
            return 1000.0
        if nxt.deadlocked:
            return -self.deadlock_penalty
        prev_lb = prev.matching_lower_bound if np.isfinite(prev.matching_lower_bound) else 0.0
        next_lb = nxt.matching_lower_bound if np.isfinite(nxt.matching_lower_bound) else prev_lb + 10.0
        return (
            self.goal_weight * (nxt.boxes_on_goals - prev.boxes_on_goals)
            + self.matching_weight * (prev_lb - next_lb)
            + self.mobility_weight * (nxt.viable_push_count - prev.viable_push_count)
            - 0.01
        )

    def rank_actions(self, state: SokobanState) -> List[Tuple[float, SokobanPushAction, SokobanState]]:
        ranked = []
        for action in self.legal_pushes(state):
            next_state = self.step_state(state, action)
            ranked.append((self.transition_score(state, action), action, next_state))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def select_action(self, state: SokobanState) -> Optional[SokobanPushAction]:
        ranked = self.rank_actions(state)
        return ranked[0][1] if ranked else None


class SokobanGridStockFlowModel(SokobanGoalDirectedSDModel):
    """Exact grid-level stock-flow SD model for Sokoban.

    This is the formal SD core for the revised Sokoban line. The state stocks
    are grid-cell box occupancy ``b`` and player occupancy ``p``. A push is a
    one-hot non-negative flow over the floor adjacency graph, and the box stock
    update is exactly ``b_next = b + B f``.
    """

    provider_id = "exact_grid_stock_flow_v1"

    def __init__(self, level: SokobanLevel):
        super().__init__(level)
        self.floor_index = {cell: idx for idx, cell in enumerate(self.floor_cells)}
        directed_edges: list[tuple[Cell, Cell, int]] = []
        for cell in self.floor_cells:
            for direction_idx, direction in enumerate(DIRECTIONS):
                target = add_cells(cell, direction)
                if self.level.is_floor(target):
                    directed_edges.append((cell, target, direction_idx))
        self.directed_edges = tuple(directed_edges)
        self.edge_index = {
            (source, target): idx
            for idx, (source, target, _direction_idx) in enumerate(self.directed_edges)
        }
        incidence = np.zeros((len(self.floor_cells), len(self.directed_edges)), dtype=np.int8)
        for edge_idx, (source, target, _direction_idx) in enumerate(self.directed_edges):
            incidence[self.floor_index[source], edge_idx] = -1
            incidence[self.floor_index[target], edge_idx] = 1
        self.grid_incidence_matrix = incidence

    @property
    def grid_state_dim(self) -> int:
        return 2 * len(self.floor_cells)

    @property
    def flow_dim(self) -> int:
        return len(self.directed_edges)

    def box_stock(self, state: SokobanState) -> np.ndarray:
        stock = np.zeros(len(self.floor_cells), dtype=np.int8)
        for box in state.boxes:
            stock[self.floor_index[box]] = 1
        return stock

    def player_stock(self, state: SokobanState) -> np.ndarray:
        stock = np.zeros(len(self.floor_cells), dtype=np.int8)
        stock[self.floor_index[state.player]] = 1
        return stock

    def grid_stock(self, state: SokobanState) -> np.ndarray:
        return np.concatenate([self.box_stock(state), self.player_stock(state)], axis=0)

    def canonical_player_cell(self, state: SokobanState) -> Cell:
        reachable = self.reachable_cells(state)
        return min(reachable) if reachable else state.player

    def canonical_key(self, state: SokobanState) -> tuple[tuple[Cell, ...], Cell]:
        """Push-level identity key: box layout plus reachable-component representative."""

        return state.boxes, self.canonical_player_cell(state)

    def legal_flows(self, state: SokobanState) -> Tuple[SokobanPushAction, ...]:
        return self.legal_pushes(state)

    def grid_flow_for_action(self, state: SokobanState, action: SokobanPushAction | int) -> SokobanGridFlow:
        if isinstance(action, int):
            action = self.decode_action(action)
        legal = set(self.legal_pushes(state))
        if action not in legal:
            raise ValueError(f"Action {action} is not a legal push in the supplied state.")
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        edge_idx = self.edge_index[(source, target)]
        return SokobanGridFlow(source=source, target=target, direction=action.direction, edge_index=edge_idx)

    def flow_vector(self, flow: SokobanGridFlow | int) -> np.ndarray:
        edge_index = flow.edge_index if isinstance(flow, SokobanGridFlow) else int(flow)
        vector = np.zeros(self.flow_dim, dtype=np.int8)
        vector[edge_index] = 1
        return vector

    def box_delta_from_flow(self, flow: SokobanGridFlow | int) -> np.ndarray:
        return self.grid_incidence_matrix @ self.flow_vector(flow)

    def step_grid_flow(self, state: SokobanState, action: SokobanPushAction | int) -> SokobanState:
        """Apply an exact legal push flow and return the next raw state."""

        # Reuse the audited exact push implementation so grid-flow and raw
        # dynamics cannot silently diverge.
        self.grid_flow_for_action(state, action)
        return self.step_state(state, action)

    def is_tier_a_deadlocked(self, state: SokobanState) -> bool:
        """Conservative Tier-A dead state used by the revised grid design."""

        if self.is_solved(state):
            return False
        dead_square = any(box not in self.level.goals and box in self.dead_squares for box in state.boxes)
        return bool(dead_square or self.frozen_boxes(state) > 0)

    def tier_a_status(self, state: SokobanState) -> Dict[str, object]:
        dead_square_boxes = sum(
            1 for box in state.boxes if box not in self.level.goals and box in self.dead_squares
        )
        frozen_boxes = self.frozen_boxes(state)
        return {
            "solved": self.is_solved(state),
            "dead_square_boxes": int(dead_square_boxes),
            "static_frozen_boxes": int(frozen_boxes),
            "tier_a_deadlocked": bool(
                (not self.is_solved(state)) and (dead_square_boxes > 0 or frozen_boxes > 0)
            ),
            "no_legal_pushes": len(self.legal_pushes(state)) == 0,
        }

    def audit_grid_transition(self, state: SokobanState, action: SokobanPushAction | int) -> Dict[str, object]:
        if isinstance(action, int):
            action = self.decode_action(action)
        flow = self.grid_flow_for_action(state, action)
        next_state = self.step_state(state, action)
        before_box = self.box_stock(state)
        after_box = self.box_stock(next_state)
        before_player = self.player_stock(state)
        after_player = self.player_stock(next_state)
        flow_delta = self.box_delta_from_flow(flow)
        exact_delta = after_box - before_box
        balance_l1 = int(np.abs(flow_delta - exact_delta).sum())
        equivalent_step = self.step_grid_flow(state, action) == next_state
        return {
            "provider_id": self.provider_id,
            "action": {
                "box_index": int(action.box_index),
                "direction": int(action.direction),
                "flat_index": int(action.flat_index),
            },
            "flow": flow.to_dict(),
            "box_mass_before": int(before_box.sum()),
            "box_mass_after": int(after_box.sum()),
            "player_mass_before": int(before_player.sum()),
            "player_mass_after": int(after_player.sum()),
            "flow_nonnegative": True,
            "flow_onehot": int(self.flow_vector(flow).sum()) == 1,
            "box_balance_l1": balance_l1,
            "step_matches_grid_flow": bool(equivalent_step),
            "canonical_key_before": {
                "boxes": [list(cell) for cell in self.canonical_key(state)[0]],
                "player_component_min": list(self.canonical_key(state)[1]),
            },
            "tier_a_after": self.tier_a_status(next_state),
        }
