"""Push-level Sokoban search utilities."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

from .env import SokobanState
from .sd_model import SokobanGoalDirectedSDModel


@dataclass(frozen=True)
class SokobanSearchResult:
    solved: bool
    solution_pushes: Optional[int]
    expanded: int
    generated: int
    pruned_deadlocks: int
    frontier_size: int
    elapsed_sec: float

    def to_dict(self) -> dict:
        return {
            "solved": self.solved,
            "solution_pushes": self.solution_pushes,
            "expanded": self.expanded,
            "generated": self.generated,
            "pruned_deadlocks": self.pruned_deadlocks,
            "frontier_size": self.frontier_size,
            "elapsed_sec": self.elapsed_sec,
        }


def best_first_search(
    model: SokobanGoalDirectedSDModel,
    max_expansions: int = 5000,
    heuristic_weight: float = 1.0,
    prune_deadlocks: bool = True,
) -> SokobanSearchResult:
    """Run push-level best-first search on a fixed Sokoban level."""

    t0 = perf_counter()
    start = model.initial_state
    if model.is_solved(start):
        return SokobanSearchResult(True, 0, 0, 0, 0, 0, perf_counter() - t0)

    def heuristic(state: SokobanState) -> float:
        if heuristic_weight <= 0:
            return 0.0
        lb = model.matching_lower_bound(state.boxes)
        if lb == float("inf"):
            return 1e6
        features = model.abstract_features(state)
        return (
            heuristic_weight * lb
            - 0.25 * features.boxes_on_goals
            - 0.01 * features.viable_push_count
        )

    frontier: list[tuple[float, int, int, SokobanState]] = []
    counter = 0
    heapq.heappush(frontier, (heuristic(start), 0, counter, start))
    best_g: Dict[SokobanState, int] = {start: 0}
    expanded = 0
    generated = 0
    pruned = 0

    while frontier and expanded < max_expansions:
        _, g, _, state = heapq.heappop(frontier)
        if g != best_g.get(state):
            continue
        if model.is_solved(state):
            return SokobanSearchResult(
                True, g, expanded, generated, pruned, len(frontier), perf_counter() - t0
            )
        expanded += 1
        for action in model.legal_pushes(state):
            next_state = model.step_state(state, action)
            if next_state == state:
                continue
            generated += 1
            if prune_deadlocks and model.is_deadlocked(next_state):
                pruned += 1
                continue
            next_g = g + 1
            if next_g >= best_g.get(next_state, 10**9):
                continue
            best_g[next_state] = next_g
            counter += 1
            heapq.heappush(frontier, (next_g + heuristic(next_state), next_g, counter, next_state))

    return SokobanSearchResult(
        False, None, expanded, generated, pruned, len(frontier), perf_counter() - t0
    )
