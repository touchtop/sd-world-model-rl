"""Room-aware GFT-SD extensions for hard Sokoban instances.

This module deliberately derives from :class:`SokobanGFTSDModel` instead of
modifying it.  The goal is to support XSokoban/Boxoban-hard diagnostics and
future room-commitment planners without changing the current Boxoban mainline.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from .env import ACTION_NAMES, DIRECTIONS, Cell, add_cells
from .gft_sd import SokobanGFTSDModel
from .sd_model import SokobanPushAction, SokobanState


@dataclass(frozen=True)
class RoomSlot:
    """One goal-room slot with static order metadata."""

    room_id: int
    slot_id: int
    cell: Cell
    entry_distance: int
    blockers_for_deeper_slots: tuple[Cell, ...]
    must_follow: tuple[Cell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": int(self.room_id),
            "slot_id": int(self.slot_id),
            "cell": list(self.cell),
            "entry_distance": int(self.entry_distance),
            "blockers_for_deeper_slots": [list(cell) for cell in self.blockers_for_deeper_slots],
            "must_follow": [list(cell) for cell in self.must_follow],
        }


@dataclass(frozen=True)
class GoalRoom:
    """Static goal-room approximation used by the hard-instance SD line."""

    room_id: int
    slots: tuple[RoomSlot, ...]
    entries: tuple[Cell, ...]
    room_cells: tuple[Cell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": int(self.room_id),
            "entries": [list(cell) for cell in self.entries],
            "room_cells": [list(cell) for cell in self.room_cells],
            "slots": [slot.to_dict() for slot in self.slots],
        }


@dataclass(frozen=True)
class RoomSlotOption:
    """One semantic slot choice attached to a physical push."""

    kind: str
    room_id: int | None = None
    slot_id: int | None = None
    slot_cell: Cell | None = None
    entry_distance: int | None = None

    @property
    def key(self) -> tuple:
        return (
            self.kind,
            None if self.room_id is None else int(self.room_id),
            None if self.slot_id is None else int(self.slot_id),
            None if self.slot_cell is None else tuple(self.slot_cell),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "room_id": None if self.room_id is None else int(self.room_id),
            "slot_id": None if self.slot_id is None else int(self.slot_id),
            "slot_cell": None if self.slot_cell is None else list(self.slot_cell),
            "entry_distance": None if self.entry_distance is None else int(self.entry_distance),
        }


@dataclass(frozen=True)
class RoomSlotCommitment:
    """Current commitment of one box position to one room slot."""

    box_cell: Cell
    room_id: int
    slot_id: int
    slot_cell: Cell

    @property
    def key(self) -> tuple:
        return (tuple(self.box_cell), int(self.room_id), int(self.slot_id), tuple(self.slot_cell))

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_cell": list(self.box_cell),
            "room_id": int(self.room_id),
            "slot_id": int(self.slot_id),
            "slot_cell": list(self.slot_cell),
        }


@dataclass(frozen=True)
class RoomCommitmentState:
    """Semantic commitment state carried by the hard-instance planner."""

    commitments: tuple[RoomSlotCommitment, ...] = ()

    @property
    def key(self) -> tuple:
        return tuple(commitment.key for commitment in self.commitments)

    def commitment_for_box(self, box_cell: Cell) -> RoomSlotCommitment | None:
        for commitment in self.commitments:
            if commitment.box_cell == box_cell:
                return commitment
        return None

    def assigned_slot_cells(self, *, excluding_box: Cell | None = None) -> set[Cell]:
        return {
            commitment.slot_cell
            for commitment in self.commitments
            if excluding_box is None or commitment.box_cell != excluding_box
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitments": [commitment.to_dict() for commitment in self.commitments],
        }


def _sorted_cells(cells: Iterable[Cell]) -> tuple[Cell, ...]:
    return tuple(sorted(set(cells)))


class SokobanRoomGFTSDModel(SokobanGFTSDModel):
    """Hard-instance room-aware GFT-SD model.

    The base GFT resources ``P,M,A,K,D,Q,C,I`` remain unchanged.  This subclass
    adds read-only room/slot diagnostics that can be consumed by separate
    hard-instance planners and audits.
    """

    provider_id = "room_gft_sd_v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_index_by_cell = {goal: idx for idx, goal in enumerate(self.goals_sorted)}
        self.goal_rooms = self._build_goal_rooms()
        self.room_slots_by_cell = {
            slot.cell: slot
            for room in self.goal_rooms
            for slot in room.slots
        }
        self.room_unreachable_transport_penalty = float(max(self.level.height + self.level.width, 1))
        floor = set(self.floor_cells)
        self.room_entry_distances = tuple(
            self._room_graph_distances(room.entries, floor)
            for room in self.goal_rooms
        )

    def _goal_components(self) -> list[set[Cell]]:
        remaining = set(self.level.goals)
        components: list[set[Cell]] = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            comp = {start}
            queue: deque[Cell] = deque([start])
            while queue:
                cell = queue.popleft()
                for direction in DIRECTIONS:
                    nxt = add_cells(cell, direction)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        comp.add(nxt)
                        queue.append(nxt)
            components.append(comp)
        components.sort(key=lambda comp: min(comp))
        return components

    def _room_graph_distances(self, starts: Iterable[Cell], traversable: set[Cell]) -> dict[Cell, int]:
        dist: dict[Cell, int] = {}
        queue: deque[Cell] = deque()
        for start in sorted(set(starts)):
            if start in dist:
                continue
            dist[start] = 0
            queue.append(start)
        while queue:
            cell = queue.popleft()
            for direction in DIRECTIONS:
                nxt = add_cells(cell, direction)
                if nxt not in traversable or nxt in dist:
                    continue
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
        return dist

    def _shortest_distance_within(self, source: Cell, target: Cell, traversable: set[Cell]) -> int | None:
        if source == target:
            return 0
        dist = self._room_graph_distances([source], traversable)
        return dist.get(target)

    def _build_goal_rooms(self) -> tuple[GoalRoom, ...]:
        rooms: list[GoalRoom] = []
        for room_id, goal_comp in enumerate(self._goal_components()):
            entries = {
                add_cells(goal, direction)
                for goal in goal_comp
                for direction in DIRECTIONS
                if self.level.is_floor(add_cells(goal, direction)) and add_cells(goal, direction) not in goal_comp
            }
            traversable = set(goal_comp) | entries
            entry_dist = self._room_graph_distances(entries, traversable)
            slot_cells = _sorted_cells(goal_comp)

            # A shallower slot that lies on a shortest entry-to-deeper-slot path
            # must remain free until the deeper slot is filled.  Therefore the
            # deeper slot must be filled before the shallower blocker.
            must_follow_by_slot: dict[Cell, set[Cell]] = {slot: set() for slot in slot_cells}
            blockers_by_deeper: dict[Cell, set[Cell]] = {slot: set() for slot in slot_cells}
            for deeper in slot_cells:
                deeper_dist = entry_dist.get(deeper)
                if deeper_dist is None:
                    continue
                for blocker in slot_cells:
                    if blocker == deeper:
                        continue
                    blocker_dist = entry_dist.get(blocker)
                    if blocker_dist is None or blocker_dist >= deeper_dist:
                        continue
                    tail = self._shortest_distance_within(blocker, deeper, traversable)
                    if tail is not None and blocker_dist + tail == deeper_dist:
                        blockers_by_deeper[deeper].add(blocker)
                        must_follow_by_slot[blocker].add(deeper)

            room_slots = []
            for slot_id, cell in enumerate(slot_cells):
                room_slots.append(
                    RoomSlot(
                        room_id=room_id,
                        slot_id=slot_id,
                        cell=cell,
                        entry_distance=int(entry_dist.get(cell, 10**9)),
                        blockers_for_deeper_slots=_sorted_cells(blockers_by_deeper[cell]),
                        must_follow=_sorted_cells(must_follow_by_slot[cell]),
                    )
                )
            rooms.append(
                GoalRoom(
                    room_id=room_id,
                    slots=tuple(room_slots),
                    entries=_sorted_cells(entries),
                    room_cells=_sorted_cells(traversable),
                )
            )
        return tuple(rooms)

    def room_static_summary(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "room_count": len(self.goal_rooms),
            "rooms": [room.to_dict() for room in self.goal_rooms],
        }

    def empty_room_commitment_state(self) -> RoomCommitmentState:
        return RoomCommitmentState(())

    def normalize_room_commitment_state(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
    ) -> RoomCommitmentState:
        if commitment_state is None:
            return self.empty_room_commitment_state()
        boxes = set(state.boxes)
        seen_boxes: set[Cell] = set()
        seen_slots: set[Cell] = set()
        kept: list[RoomSlotCommitment] = []
        for commitment in sorted(commitment_state.commitments, key=lambda item: item.key):
            if commitment.box_cell not in boxes:
                continue
            if commitment.slot_cell not in self.room_slots_by_cell:
                continue
            if commitment.box_cell in seen_boxes or commitment.slot_cell in seen_slots:
                continue
            seen_boxes.add(commitment.box_cell)
            seen_slots.add(commitment.slot_cell)
            kept.append(commitment)
        return RoomCommitmentState(tuple(kept))

    def room_commitment_key(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
    ) -> tuple:
        normalized = self.normalize_room_commitment_state(state, commitment_state)
        return self.gft_canonical_key_cached(state), normalized.key

    def _filled_slots_for_room(self, state: SokobanState, room: GoalRoom) -> set[Cell]:
        boxes = set(state.boxes)
        return {slot.cell for slot in room.slots if slot.cell in boxes}

    def _frontier_slots_for_room(self, state: SokobanState, room: GoalRoom) -> set[Cell]:
        filled = self._filled_slots_for_room(state, room)
        frontier = set()
        for slot in room.slots:
            if slot.cell in filled:
                continue
            if all(required in filled for required in slot.must_follow):
                frontier.add(slot.cell)
        return frontier

    def _valid_filled_slots_for_room(self, state: SokobanState, room: GoalRoom) -> set[Cell]:
        filled = self._filled_slots_for_room(state, room)
        valid = set()
        for slot in room.slots:
            if slot.cell in filled and all(required in filled for required in slot.must_follow):
                valid.add(slot.cell)
        return valid

    def _slot_pull_distance(self, box_cell: Cell, slot_cell: Cell) -> float:
        goal_idx = self.goal_index_by_cell.get(slot_cell)
        if goal_idx is None:
            return math.inf
        raw = self.pull_distances[goal_idx].get(box_cell)
        if raw is None or raw >= 10**8:
            return math.inf
        return float(raw)

    def _finite_room_transport_cost(self, distance: float) -> tuple[float, bool]:
        if math.isfinite(distance):
            return float(distance), True
        return float(self.room_unreachable_transport_penalty), False

    def _greedy_slot_assignment(
        self,
        source_boxes: Iterable[Cell],
        target_slots: Iterable[RoomSlot],
    ) -> tuple[float, list[dict[str, Any]], int]:
        """Assign open frontier slots to uncommitted boxes as a deterministic proxy.

        The ordering is intentionally schedule-aware: deeper slots are claimed
        before their shallower blockers.  This is a diagnostic transport proxy,
        not an optimal Sokoban solver.
        """

        unused = set(source_boxes)
        rows: list[dict[str, Any]] = []
        total_cost = 0.0
        infeasible = 0
        slots = sorted(
            target_slots,
            key=lambda slot: (slot.room_id, -slot.entry_distance, slot.slot_id, slot.cell),
        )
        for slot in slots:
            best_box = None
            best_distance = math.inf
            for box in sorted(unused):
                distance = self._slot_pull_distance(box, slot.cell)
                if distance < best_distance:
                    best_box = box
                    best_distance = distance
            if best_box is None:
                cost, feasible = self._finite_room_transport_cost(math.inf)
            else:
                cost, feasible = self._finite_room_transport_cost(best_distance)
                unused.remove(best_box)
            total_cost += cost
            if not feasible:
                infeasible += 1
            rows.append(
                {
                    "box_cell": None if best_box is None else list(best_box),
                    "slot": slot.to_dict(),
                    "pull_distance": None if not feasible else float(best_distance),
                    "cost": float(cost),
                    "feasible": bool(feasible),
                }
            )
        return float(total_cost), rows, int(infeasible)

    def room_slot_progress_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
    ) -> dict[str, Any]:
        """Slot-schedule constrained progress proxy for hard Sokoban rooms.

        Lower costs are better.  Committed boxes are measured against their
        committed slots with static reverse-push distances.  Uncommitted boxes
        are greedily assigned to currently legal frontier slots, deeper slots
        first, so the proxy follows the inferred room fill schedule.
        """

        commitments = self.normalize_room_commitment_state(state, commitment_state)
        boxes = set(state.boxes)
        committed_boxes = {commitment.box_cell for commitment in commitments.commitments}
        assigned_slots = {commitment.slot_cell for commitment in commitments.commitments}
        valid_filled_slots: set[Cell] = set()
        frontier_slots: set[Cell] = set()
        slot_by_cell: dict[Cell, RoomSlot] = {}
        for room in self.goal_rooms:
            valid_filled_slots.update(self._valid_filled_slots_for_room(state, room))
            frontier_slots.update(self._frontier_slots_for_room(state, room))
            slot_by_cell.update({slot.cell: slot for slot in room.slots})

        committed_rows: list[dict[str, Any]] = []
        committed_transport_cost = 0.0
        committed_infeasible_count = 0
        committed_satisfied_count = 0
        committed_pass_through_count = 0
        nonfrontier_commitment_count = 0
        for commitment in commitments.commitments:
            slot = slot_by_cell.get(commitment.slot_cell)
            box_slot = slot_by_cell.get(commitment.box_cell)
            distance = self._slot_pull_distance(commitment.box_cell, commitment.slot_cell)
            cost, feasible = self._finite_room_transport_cost(distance)
            committed_transport_cost += cost
            if not feasible:
                committed_infeasible_count += 1
            if commitment.box_cell == commitment.slot_cell and commitment.slot_cell in valid_filled_slots:
                committed_satisfied_count += 1
            pass_through = bool(
                box_slot is not None
                and commitment.slot_cell in box_slot.must_follow
                and feasible
                and commitment.box_cell != commitment.slot_cell
            )
            if pass_through:
                committed_pass_through_count += 1
            if commitment.slot_cell not in frontier_slots and commitment.box_cell != commitment.slot_cell:
                nonfrontier_commitment_count += 1
            committed_rows.append(
                {
                    "box_cell": list(commitment.box_cell),
                    "slot": None if slot is None else slot.to_dict(),
                    "pull_distance": None if not feasible else float(distance),
                    "cost": float(cost),
                    "feasible": bool(feasible),
                    "satisfied": bool(
                        commitment.box_cell == commitment.slot_cell
                        and commitment.slot_cell in valid_filled_slots
                    ),
                    "committed_pass_through_blocker": bool(pass_through),
                    "nonfrontier_open_commitment": bool(
                        commitment.slot_cell not in frontier_slots and commitment.box_cell != commitment.slot_cell
                    ),
                }
            )

        open_frontier_cells = frontier_slots - valid_filled_slots - assigned_slots
        open_frontier_slots = [slot_by_cell[cell] for cell in open_frontier_cells if cell in slot_by_cell]
        source_boxes = boxes - committed_boxes - valid_filled_slots
        uncommitted_frontier_cost, assignment_rows, infeasible_frontier_count = self._greedy_slot_assignment(
            source_boxes,
            open_frontier_slots,
        )

        room_totals = self.room_state_summary(state)["totals"]
        q_settled_effective = max(0, int(room_totals["q_settled"]) - committed_pass_through_count)
        k_blocked_effective = max(0, int(room_totals["k_blocked"]) - committed_pass_through_count)
        barrier_counts = {
            "q_settled": int(room_totals["q_settled"]),
            "k_blocked": int(room_totals["k_blocked"]),
            "q_settled_effective": int(q_settled_effective),
            "k_blocked_effective": int(k_blocked_effective),
            "committed_pass_through_blocker": int(committed_pass_through_count),
            "uncommitted_room_or_entry": int(room_totals["uncommitted_room_or_entry"]),
            "nonfrontier_commitment": int(nonfrontier_commitment_count),
            "committed_infeasible": int(committed_infeasible_count),
            "frontier_assignment_infeasible": int(infeasible_frontier_count),
        }
        barrier_cost = (
            6.0 * barrier_counts["q_settled_effective"]
            + 6.0 * barrier_counts["k_blocked_effective"]
            + 2.0 * barrier_counts["uncommitted_room_or_entry"]
            + 3.0 * barrier_counts["nonfrontier_commitment"]
            + self.room_unreachable_transport_penalty * barrier_counts["committed_infeasible"]
            + 0.5 * self.room_unreachable_transport_penalty * barrier_counts["frontier_assignment_infeasible"]
        )
        total_transport_cost = float(committed_transport_cost + uncommitted_frontier_cost)
        slot_schedule_cost = float(total_transport_cost + barrier_cost)
        return {
            "lower_is_better": True,
            "transport_proxy": "reverse_push_distance_to_committed_slots_plus_greedy_frontier_slots",
            "unreachable_transport_penalty": float(self.room_unreachable_transport_penalty),
            "committed_box_count": int(len(committed_boxes)),
            "committed_satisfied_count": int(committed_satisfied_count),
            "committed_pass_through_blocker_count": int(committed_pass_through_count),
            "assigned_slot_count": int(len(assigned_slots)),
            "open_frontier_slot_count": int(len(open_frontier_slots)),
            "available_uncommitted_box_count": int(len(source_boxes)),
            "valid_filled_slot_count": int(len(valid_filled_slots)),
            "frontier_slot_count": int(len(frontier_slots)),
            "committed_transport_cost": float(committed_transport_cost),
            "uncommitted_frontier_transport_cost": float(uncommitted_frontier_cost),
            "total_transport_cost": float(total_transport_cost),
            "barrier_counts": barrier_counts,
            "barrier_cost": float(barrier_cost),
            "slot_schedule_cost": float(slot_schedule_cost),
            "committed_rows": committed_rows,
            "frontier_assignment_rows": assignment_rows,
        }

    def room_state_summary(self, state: SokobanState) -> dict[str, Any]:
        boxes = set(state.boxes)
        room_rows = []
        totals = {
            "frontier_slot_count": 0,
            "filled_slot_count": 0,
            "valid_filled_slot_count": 0,
            "q_settled": 0,
            "k_blocked": 0,
            "c_good": 0,
            "c_bad": 0,
            "uncommitted_entry": 0,
            "uncommitted_in_room": 0,
            "uncommitted_room_or_entry": 0,
        }
        for room in self.goal_rooms:
            filled = self._filled_slots_for_room(state, room)
            frontier = self._frontier_slots_for_room(state, room)
            q_settled_slots = []
            blocked_slots = []
            valid_filled = []
            for slot in room.slots:
                missing_required = [required for required in slot.must_follow if required not in filled]
                if slot.cell in filled:
                    if missing_required:
                        q_settled_slots.append(slot.cell)
                    else:
                        valid_filled.append(slot.cell)
                else:
                    occupied_blockers = [blocker for blocker in slot.blockers_for_deeper_slots if blocker in boxes]
                    if occupied_blockers:
                        blocked_slots.append(slot.cell)

            uncommitted = [
                box
                for box in boxes
                if box in room.room_cells and box not in filled and box not in room.entries
            ]
            uncommitted_entry = [box for box in boxes if box in room.entries]
            uncommitted_room_or_entry = len(uncommitted) + len(uncommitted_entry)
            c_bad = len(q_settled_slots) + len(blocked_slots) + uncommitted_room_or_entry
            row = {
                "room_id": int(room.room_id),
                "filled_slots": [list(cell) for cell in sorted(filled)],
                "frontier_slots": [list(cell) for cell in sorted(frontier)],
                "valid_filled_slots": [list(cell) for cell in sorted(valid_filled)],
                "q_settled_slots": [list(cell) for cell in sorted(q_settled_slots)],
                "k_blocked_slots": [list(cell) for cell in sorted(blocked_slots)],
                "uncommitted_entry_boxes": [list(cell) for cell in sorted(uncommitted_entry)],
                "uncommitted_in_room_boxes": [list(cell) for cell in sorted(uncommitted)],
                "uncommitted_room_or_entry": uncommitted_room_or_entry,
                "c_good": len(valid_filled),
                "c_bad": c_bad,
            }
            room_rows.append(row)
            totals["frontier_slot_count"] += len(frontier)
            totals["filled_slot_count"] += len(filled)
            totals["valid_filled_slot_count"] += len(valid_filled)
            totals["q_settled"] += len(q_settled_slots)
            totals["k_blocked"] += len(blocked_slots)
            totals["c_good"] += len(valid_filled)
            totals["c_bad"] += c_bad
            totals["uncommitted_entry"] += len(uncommitted_entry)
            totals["uncommitted_in_room"] += len(uncommitted)
            totals["uncommitted_room_or_entry"] += uncommitted_room_or_entry
        return {
            "totals": totals,
            "rooms": room_rows,
        }

    def room_semantic_slot_options(
        self,
        state: SokobanState,
        action: SokobanPushAction,
        child_state: SokobanState | None = None,
        *,
        commitment_state: RoomCommitmentState | None = None,
        include_uncommitted_option: bool = False,
        commitment_boundary_distance: int = 0,
    ) -> tuple[RoomSlotOption, ...]:
        child_state = child_state if child_state is not None else self.step_grid_flow(state, action)
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        commitments = self.normalize_room_commitment_state(state, commitment_state)
        existing = commitments.commitment_for_box(source)
        assigned = commitments.assigned_slot_cells(excluding_box=source)
        options: list[RoomSlotOption] = []
        touched_room = False

        for room in self.goal_rooms:
            room_slots = {slot.cell for slot in room.slots}
            entry_dist = self.room_entry_distances[room.room_id]
            near_boundary = (
                int(commitment_boundary_distance) > 0
                and min(
                    entry_dist.get(source, 10**9),
                    entry_dist.get(target, 10**9),
                )
                <= int(commitment_boundary_distance)
            )
            touches_room = (
                source in room.room_cells
                or target in room.room_cells
                or source in room.entries
                or target in room.entries
                or target in room_slots
                or near_boundary
            )
            if not touches_room:
                continue
            touched_room = True
            if existing is not None and existing.room_id == room.room_id:
                slot = self.room_slots_by_cell.get(existing.slot_cell)
                if slot is not None:
                    options.append(
                        RoomSlotOption(
                            kind="slot",
                            room_id=slot.room_id,
                            slot_id=slot.slot_id,
                            slot_cell=slot.cell,
                            entry_distance=slot.entry_distance,
                        )
                    )
                    continue
            frontier = self._frontier_slots_for_room(state, room)
            for slot in room.slots:
                if slot.cell not in frontier or slot.cell in assigned:
                    continue
                options.append(
                    RoomSlotOption(
                        kind="slot",
                        room_id=room.room_id,
                        slot_id=slot.slot_id,
                        slot_cell=slot.cell,
                        entry_distance=slot.entry_distance,
                    )
                )

        if not touched_room:
            return (RoomSlotOption(kind="none"),)
        if include_uncommitted_option or not options:
            options.append(RoomSlotOption(kind="uncommitted"))
        return tuple(sorted(options, key=lambda item: item.key))

    def advance_room_commitment_state(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
        action: SokobanPushAction,
        option: RoomSlotOption,
        child_state: SokobanState | None = None,
    ) -> RoomCommitmentState:
        child_state = child_state if child_state is not None else self.step_grid_flow(state, action)
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        child_boxes = set(child_state.boxes)
        commitments = self.normalize_room_commitment_state(state, commitment_state)
        moved_existing = commitments.commitment_for_box(source)
        kept: list[RoomSlotCommitment] = []
        for commitment in commitments.commitments:
            if commitment.box_cell == source:
                continue
            if commitment.box_cell in child_boxes:
                kept.append(commitment)

        if option.kind == "slot" and option.slot_cell is not None:
            kept = [
                commitment
                for commitment in kept
                if commitment.slot_cell != option.slot_cell and commitment.box_cell != target
            ]
            kept.append(
                RoomSlotCommitment(
                    box_cell=target,
                    room_id=int(option.room_id),
                    slot_id=int(option.slot_id),
                    slot_cell=option.slot_cell,
                )
            )
        elif moved_existing is not None and target in child_boxes:
            kept.append(
                RoomSlotCommitment(
                    box_cell=target,
                    room_id=moved_existing.room_id,
                    slot_id=moved_existing.slot_id,
                    slot_cell=moved_existing.slot_cell,
                )
            )
        return self.normalize_room_commitment_state(child_state, RoomCommitmentState(tuple(kept)))

    def room_commitment_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
    ) -> dict[str, Any]:
        commitments = self.normalize_room_commitment_state(state, commitment_state)
        committed_boxes = {commitment.box_cell for commitment in commitments.commitments}
        assigned_slots = {commitment.slot_cell for commitment in commitments.commitments}
        room_summary = self.room_state_summary(state)
        uncommitted_room_or_entry = 0
        nonfrontier_commitments = 0
        for room in self.goal_rooms:
            frontier = self._frontier_slots_for_room(state, room)
            for commitment in commitments.commitments:
                if commitment.room_id == room.room_id and commitment.slot_cell not in frontier:
                    if commitment.slot_cell not in set(state.boxes):
                        nonfrontier_commitments += 1
            room_cells = set(room.room_cells) | set(room.entries)
            uncommitted_room_or_entry += sum(
                1 for box in state.boxes if box in room_cells and box not in committed_boxes
            )
        return {
            "commitment_state": commitments.to_dict(),
            "committed_box_count": len(committed_boxes),
            "assigned_slot_count": len(assigned_slots),
            "nonfrontier_commitment_count": int(nonfrontier_commitments),
            "uncommitted_room_or_entry_count": int(uncommitted_room_or_entry),
            "room_totals": room_summary["totals"],
        }

    def room_semantic_options(
        self,
        state: SokobanState,
        action: SokobanPushAction,
        child_state: SokobanState | None = None,
    ) -> dict[str, Any]:
        """Approximate semantic commitment choices for a physical push."""

        child_state = child_state if child_state is not None else self.step_grid_flow(state, action)
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        options = self.room_semantic_slot_options(state, action, child_state)
        touched_rooms: list[int] = []
        reasons: set[str] = set()
        for room in self.goal_rooms:
            room_slots = {slot.cell for slot in room.slots}
            touches_room = (
                source in room.room_cells
                or target in room.room_cells
                or source in room.entries
                or target in room.entries
                or target in room_slots
            )
            if not touches_room:
                continue
            touched_rooms.append(int(room.room_id))
            if source not in room.room_cells and target in room.room_cells:
                reasons.add("cross_room_boundary")
            if target in room.entries:
                reasons.add("enter_room_entry")
            if target in room_slots:
                reasons.add("target_room_slot")
        if touched_rooms and not any(option.kind == "slot" for option in options):
            reasons.add("no_frontier_slot_available")
        return {
            "action": {
                "box_index": int(action.box_index),
                "direction": int(action.direction),
                "source": list(source),
                "target": list(target),
            },
            "physical_child_state_changed": child_state != state,
            "touched_room_ids": touched_rooms,
            "option_slots": [option.to_dict() for option in options if option.kind == "slot"],
            "semantic_child_count_proxy": int(len(options)),
            "reasons": sorted(reasons),
        }

    def room_action_summary(
        self,
        state: SokobanState,
        action: SokobanPushAction,
        child_state: SokobanState | None = None,
    ) -> dict[str, Any]:
        child_state = child_state if child_state is not None else self.step_grid_flow(state, action)
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        before = self.room_state_summary(state)
        after = self.room_state_summary(child_state)
        impact_codes: set[str] = set()
        target_slot = None
        frontier_target = False
        nonfrontier_target = False
        for room in self.goal_rooms:
            room_slots = {slot.cell for slot in room.slots}
            frontier = self._frontier_slots_for_room(state, room)
            if source not in room.room_cells and target in room.room_cells:
                impact_codes.add("CROSS_ROOM_BOUNDARY")
            if target in room.entries:
                impact_codes.add("ENTER_ROOM_ENTRY")
                impact_codes.add("UNCOMMITTED_ENTRY")
            if target in room_slots:
                target_slot = target
                if target in frontier:
                    frontier_target = True
                    impact_codes.add("FILL_FRONTIER_SLOT")
                else:
                    nonfrontier_target = True
                    impact_codes.add("FILL_NONFRONTIER_SLOT")
        before_totals = before["totals"]
        after_totals = after["totals"]
        if after_totals["q_settled"] > before_totals["q_settled"]:
            impact_codes.add("INCREASE_ROOM_ORDER_DEBT")
        if after_totals["k_blocked"] > before_totals["k_blocked"]:
            impact_codes.add("BLOCK_UNFILLED_SLOT")
        if after_totals["uncommitted_in_room"] > before_totals["uncommitted_in_room"]:
            impact_codes.add("UNCOMMITTED_DEEP_PUSH")
        if after_totals["uncommitted_room_or_entry"] > before_totals["uncommitted_room_or_entry"]:
            impact_codes.add("INCREASE_UNCOMMITTED_ROOM_OR_ENTRY")
        return {
            "action": {
                "box_index": int(action.box_index),
                "direction": int(action.direction),
                "source": list(source),
                "target": list(target),
            },
            "target_slot": list(target_slot) if target_slot is not None else None,
            "frontier_target": bool(frontier_target),
            "nonfrontier_target": bool(nonfrontier_target),
            "impact_codes": sorted(impact_codes),
            "room_before": before,
            "room_after": after,
        }


class SokobanRoomFillPlanGFTSDModel(SokobanRoomGFTSDModel):
    """A8 room-aware SD model with explicit fill-plan diagnostics.

    The class keeps the exact grid-flow transition and all GFT readouts from
    :class:`SokobanRoomGFTSDModel`.  It only adds state-derived plan variables
    distilled from the first-level teacher analysis: lane fill order,
    channel/staging pressure, and active-slot release/setup signals.
    """

    provider_id = "room_fill_plan_gft_sd_v1"

    def _entry_straight_approach_length(self, room: GoalRoom, entry: Cell) -> int:
        room_cells = set(room.room_cells)
        best = 0
        for direction in DIRECTIONS:
            first = add_cells(entry, direction)
            if first in room_cells or not self.level.is_floor(first):
                continue
            length = 0
            cell = first
            while self.level.is_floor(cell) and cell not in room_cells:
                length += 1
                cell = add_cells(cell, direction)
            best = max(best, length)
        return int(best)

    def room_channel_entry(self, room: GoalRoom) -> Cell | None:
        """Pick the entry most aligned with the external corridor."""

        if not room.entries:
            return None
        return max(
            room.entries,
            key=lambda entry: (
                self._entry_straight_approach_length(room, entry),
                -abs(entry[0] - sum(cell[0] for cell in room.entries) / len(room.entries)),
                -entry[0],
                -entry[1],
            ),
        )

    def room_slot_lane_entry(self, room: GoalRoom, slot: RoomSlot) -> Cell | None:
        if not room.entries:
            return None
        return min(
            room.entries,
            key=lambda entry: (
                abs(entry[0] - slot.cell[0]) + abs(entry[1] - slot.cell[1]),
                abs(entry[0] - slot.cell[0]),
                entry,
            ),
        )

    def room_fill_plan_queue(self, room: GoalRoom) -> tuple[dict[str, Any], ...]:
        """Return a deterministic lane queue: off-channel lanes first.

        In the XSokoban first-room pattern, the teacher uses the central lane as
        a staging/channel lane and finalizes it after the off-channel lanes.
        This method encodes that structural rule without storing teacher states.
        """

        channel_entry = self.room_channel_entry(room)
        lane_rows: dict[Cell | None, list[RoomSlot]] = {}
        for slot in room.slots:
            lane_entry = self.room_slot_lane_entry(room, slot)
            lane_rows.setdefault(lane_entry, []).append(slot)

        def lane_key(item: tuple[Cell | None, list[RoomSlot]]) -> tuple:
            lane_entry, _slots = item
            is_channel = lane_entry == channel_entry
            if channel_entry is None or lane_entry is None:
                offset = 0
            else:
                offset = abs(lane_entry[0] - channel_entry[0]) + abs(lane_entry[1] - channel_entry[1])
            return (int(is_channel), offset, (10**6, 10**6) if lane_entry is None else lane_entry)

        rows: list[dict[str, Any]] = []
        for lane_entry, slots in sorted(lane_rows.items(), key=lane_key):
            channel_lane = lane_entry == channel_entry
            for slot in sorted(slots, key=lambda item: (-item.entry_distance, item.slot_id, item.cell)):
                rows.append(
                    {
                        "queue_index": len(rows),
                        "room_id": int(room.room_id),
                        "slot_obj": slot,
                        "slot": slot.to_dict(),
                        "slot_cell": slot.cell,
                        "lane_entry": lane_entry,
                        "channel_lane": bool(channel_lane),
                        "deep_slot": bool(not slot.must_follow),
                    }
                )
        return tuple(rows)

    def room_fill_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
    ) -> dict[str, Any]:
        """State-derived room fill plan consumed by the A8 planner."""

        commitments = self.normalize_room_commitment_state(state, commitment_state)
        boxes = set(state.boxes)
        committed_by_slot = {
            commitment.slot_cell: commitment
            for commitment in commitments.commitments
        }
        committed_by_box = {
            commitment.box_cell: commitment
            for commitment in commitments.commitments
        }
        rooms_out: list[dict[str, Any]] = []
        totals = {
            "active_slot_count": 0,
            "filled_prefix_count": 0,
            "early_channel_occupancy": 0,
            "early_channel_final_commitment": 0,
            "channel_staging_box_count": 0,
            "active_commitment_count": 0,
            "active_box_push_count": 0,
            "active_box_monotone_push_count": 0,
            "active_box_blocked_count": 0,
            "active_transport_cost": 0.0,
            "active_transport_feasible_count": 0,
            "active_transport_infeasible_count": 0,
            "plan_barrier_cost": 0.0,
        }
        for room in self.goal_rooms:
            queue = self.room_fill_plan_queue(room)
            queue_cells = [row["slot_cell"] for row in queue]
            channel_cells = {row["slot_cell"] for row in queue if row["channel_lane"]}
            channel_entries = set()
            channel_entry = self.room_channel_entry(room)
            if channel_entry is not None:
                channel_entries.add(channel_entry)

            prefix = 0
            for cell in queue_cells:
                if cell in boxes:
                    prefix += 1
                else:
                    break

            deep_open_rows = [
                row for row in queue if row["deep_slot"] and row["slot_cell"] not in boxes
            ]
            if deep_open_rows:
                active_rows = [deep_open_rows[0]]
                fill_phase = "deep_slots"
            else:
                active_rows = [
                    row
                    for row in queue
                    if not row["deep_slot"] and row["slot_cell"] not in boxes
                ]
                if active_rows:
                    active_rows = [active_rows[0]]
                fill_phase = "shallow_slots"
            active_cells = {row["slot_cell"] for row in active_rows}
            active_row = min(
                active_rows,
                key=lambda row: (int(row["channel_lane"]), int(row["queue_index"])),
                default=None,
            )
            active_slot = None if active_row is None else active_row["slot_cell"]

            open_before_channel = 0
            first_channel_index = None
            for row in queue:
                if row["channel_lane"]:
                    first_channel_index = int(row["queue_index"])
                    break
            if fill_phase == "deep_slots":
                open_before_channel = sum(
                    1
                    for row in queue
                    if row["deep_slot"] and not row["channel_lane"] and row["slot_cell"] not in boxes
                )
            elif first_channel_index is not None:
                open_before_channel = sum(
                    1
                    for row in queue
                    if not row["deep_slot"]
                    and not row["channel_lane"]
                    and row["slot_cell"] not in boxes
                )

            queue_rows = []
            for row in queue:
                cell = row["slot_cell"]
                commitment = committed_by_slot.get(cell)
                occupied = cell in boxes
                prefix_required_open = sum(
                    1 for prev in queue_cells[: int(row["queue_index"])] if prev not in boxes
                )
                queue_rows.append(
                    {
                        "queue_index": int(row["queue_index"]),
                        "slot": row["slot"],
                        "lane_entry": None if row["lane_entry"] is None else list(row["lane_entry"]),
                        "channel_lane": bool(row["channel_lane"]),
                        "deep_slot": bool(row["deep_slot"]),
                        "occupied": bool(occupied),
                        "prefix_required_open": int(prefix_required_open),
                        "committed_box_cell": None if commitment is None else list(commitment.box_cell),
                    }
                )

            early_channel_occupancy = 0
            early_channel_final_commitment = 0
            channel_staging_box_count = 0
            active_commitment_count = 0
            for box in boxes:
                commitment = committed_by_box.get(box)
                in_channel = box in channel_cells or box in channel_entries
                committed_to_channel = commitment is not None and commitment.slot_cell in channel_cells
                committed_to_active = commitment is not None and commitment.slot_cell in active_cells
                if committed_to_active:
                    active_commitment_count += 1
                if in_channel and open_before_channel > 0 and not (
                    commitment is not None and commitment.slot_cell not in channel_cells
                ):
                    early_channel_occupancy += 1
                if committed_to_channel and open_before_channel > 0:
                    early_channel_final_commitment += 1
                if in_channel and commitment is not None and commitment.slot_cell not in channel_cells:
                    channel_staging_box_count += 1

            active_transport_cost = 0.0
            active_transport_feasible = 0
            active_transport_infeasible = 0
            active_assignment = None
            active_box_push_count = 0
            active_box_monotone_push_count = 0
            active_box_blocked_count = 0
            if active_rows:
                deep_filled_cells = {
                    row["slot_cell"]
                    for row in queue
                    if row["deep_slot"] and row["slot_cell"] in boxes
                }
                shallow_settled_cells = set()
                if fill_phase == "shallow_slots":
                    shallow_settled_cells = {
                        row["slot_cell"]
                        for row in queue
                        if not row["deep_slot"] and row["slot_cell"] in boxes
                    }
                source_boxes = boxes - deep_filled_cells - shallow_settled_cells
                active_transport_cost, assignment_rows, active_transport_infeasible = self._greedy_slot_assignment(
                    source_boxes,
                    [row["slot_obj"] for row in active_rows],
                )
                active_transport_feasible = max(0, len(active_rows) - active_transport_infeasible)
                active_slot_by_box = {
                    tuple(row["box_cell"]): tuple(row["slot"]["cell"])
                    for row in assignment_rows
                    if row.get("box_cell") is not None and row.get("feasible")
                }
                for box, slot_cell in active_slot_by_box.items():
                    before_distance = self._slot_pull_distance(box, slot_cell)
                    box_push_count = 0
                    box_monotone_count = 0
                    for action in self.legal_flows(state):
                        source = state.boxes[action.box_index]
                        if source != box:
                            continue
                        target = add_cells(source, DIRECTIONS[action.direction])
                        child = self.step_grid_flow(state, action)
                        if self.is_tier_a_deadlocked(child):
                            continue
                        box_push_count += 1
                        after_distance = self._slot_pull_distance(target, slot_cell)
                        if math.isfinite(before_distance) and math.isfinite(after_distance):
                            if after_distance < before_distance:
                                box_monotone_count += 1
                    active_box_push_count += box_push_count
                    active_box_monotone_push_count += box_monotone_count
                    if box_push_count == 0:
                        active_box_blocked_count += 1
                active_assignment = {
                    "active_slot_count": int(len(active_rows)),
                    "cost": float(active_transport_cost),
                    "feasible_count": int(active_transport_feasible),
                    "infeasible_count": int(active_transport_infeasible),
                    "rows": assignment_rows,
                    "active_box_push_count": int(active_box_push_count),
                    "active_box_monotone_push_count": int(active_box_monotone_push_count),
                    "active_box_blocked_count": int(active_box_blocked_count),
                }

            plan_barrier_cost = (
                8.0 * early_channel_occupancy
                + 12.0 * early_channel_final_commitment
                - 3.0 * channel_staging_box_count
                - 2.0 * active_commitment_count
                + 4.0 * active_box_blocked_count
                - 0.5 * active_box_push_count
                - 1.5 * active_box_monotone_push_count
                + self.room_unreachable_transport_penalty * active_transport_infeasible
                + 0.05 * active_transport_cost
            )
            totals["active_slot_count"] += int(active_slot is not None)
            totals["filled_prefix_count"] += int(prefix)
            totals["early_channel_occupancy"] += int(early_channel_occupancy)
            totals["early_channel_final_commitment"] += int(early_channel_final_commitment)
            totals["channel_staging_box_count"] += int(channel_staging_box_count)
            totals["active_commitment_count"] += int(active_commitment_count)
            totals["active_box_push_count"] += int(active_box_push_count)
            totals["active_box_monotone_push_count"] += int(active_box_monotone_push_count)
            totals["active_box_blocked_count"] += int(active_box_blocked_count)
            totals["active_transport_cost"] += float(active_transport_cost)
            totals["active_transport_feasible_count"] += int(active_transport_feasible)
            totals["active_transport_infeasible_count"] += int(active_transport_infeasible)
            totals["plan_barrier_cost"] += float(plan_barrier_cost)

            rooms_out.append(
                {
                    "room_id": int(room.room_id),
                    "channel_entry": None if channel_entry is None else list(channel_entry),
                    "channel_slot_cells": [list(cell) for cell in sorted(channel_cells)],
                    "open_before_channel_count": int(open_before_channel),
                    "fill_phase": fill_phase,
                    "filled_prefix_count": int(prefix),
                    "active_slots": [row["slot"] for row in active_rows],
                    "active_slot": None if active_slot is None else next(
                        row["slot"] for row in queue if row["slot_cell"] == active_slot
                    ),
                    "active_assignment": active_assignment,
                    "early_channel_occupancy": int(early_channel_occupancy),
                    "early_channel_final_commitment": int(early_channel_final_commitment),
                    "channel_staging_box_count": int(channel_staging_box_count),
                    "active_commitment_count": int(active_commitment_count),
                    "active_box_push_count": int(active_box_push_count),
                    "active_box_monotone_push_count": int(active_box_monotone_push_count),
                    "active_box_blocked_count": int(active_box_blocked_count),
                    "active_transport_cost": float(active_transport_cost),
                    "plan_barrier_cost": float(plan_barrier_cost),
                    "queue": queue_rows,
                }
            )
        return {
            "provider_id": self.provider_id,
            "lower_is_better": True,
            "plan_source": "teacher_trace_distilled_channel_staging_rule",
            "totals": totals,
            "rooms": rooms_out,
        }

    def room_fill_plan_slot_rank(self, slot_cell: Cell) -> tuple[int, bool] | None:
        for room in self.goal_rooms:
            for row in self.room_fill_plan_queue(room):
                if row["slot_cell"] == slot_cell:
                    return int(row["queue_index"]), bool(row["channel_lane"])
        return None


class SokobanRoomReleasePlanGFTSDModel(SokobanRoomFillPlanGFTSDModel):
    """A9 room-aware SD model with active-box release diagnostics.

    A8 exposes that the active assigned box is blocked, but it does not explain
    which local access cell must be released.  A9 keeps A8 unchanged and adds a
    state-derived release plan around the active assigned box: monotone push
    candidates, their required player stand cells, and the boxes or reachability
    constraints that prevent those pushes.
    """

    provider_id = "room_release_plan_gft_sd_v1"

    def _active_assignment_rows(
        self,
        fill_summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for room in fill_summary["rooms"]:
            active_assignment = room.get("active_assignment")
            if not active_assignment:
                continue
            for assignment in active_assignment.get("rows", []):
                if not assignment.get("feasible"):
                    continue
                box_cell = assignment.get("box_cell")
                slot = assignment.get("slot") or {}
                slot_cell = slot.get("cell")
                if box_cell is None or slot_cell is None:
                    continue
                rows.append(
                    {
                        "room_id": int(room["room_id"]),
                        "fill_phase": room.get("fill_phase"),
                        "active_slot": room.get("active_slot"),
                        "box_cell": tuple(box_cell),
                        "slot_cell": tuple(slot_cell),
                        "pull_distance": assignment.get("pull_distance"),
                        "cost": float(assignment.get("cost", 0.0)),
                    }
                )
        return rows

    def _release_candidate_rows(
        self,
        state: SokobanState,
        *,
        box_cell: Cell,
        slot_cell: Cell,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        boxes = set(state.boxes)
        reachable = self.reachable_cells(state)
        before_distance = self._slot_pull_distance(box_cell, slot_cell)
        candidate_rows: list[dict[str, Any]] = []
        blocker_cells: set[Cell] = set()
        blocker_boxes: set[Cell] = set()
        required_stands: set[Cell] = set()
        totals = {
            "candidate_push_count": 0,
            "release_need_count": 0,
            "stand_blocker_count": 0,
            "target_blocker_count": 0,
            "unreachable_stand_count": 0,
            "reachable_stand_count": 0,
            "monotone_available_count": 0,
            "deadlocked_candidate_count": 0,
            "release_barrier_cost": 0.0,
        }
        box_index = state.boxes.index(box_cell) if box_cell in state.boxes else None
        for direction_index, direction in enumerate(DIRECTIONS):
            stand_cell = (box_cell[0] - direction[0], box_cell[1] - direction[1])
            target_cell = add_cells(box_cell, direction)
            stand_is_floor = self.level.is_floor(stand_cell)
            target_is_floor = self.level.is_floor(target_cell)
            after_distance = self._slot_pull_distance(target_cell, slot_cell) if target_is_floor else math.inf
            monotone = (
                stand_is_floor
                and target_is_floor
                and math.isfinite(before_distance)
                and math.isfinite(after_distance)
                and after_distance < before_distance
            )
            if not monotone:
                continue

            stand_occupied = stand_cell in boxes
            target_occupied = target_cell in boxes
            stand_reachable = stand_cell in reachable
            child_tier_a_deadlocked = False
            if (
                box_index is not None
                and not stand_occupied
                and not target_occupied
                and stand_reachable
            ):
                action = SokobanPushAction(box_index, direction_index)
                child_tier_a_deadlocked = self.is_tier_a_deadlocked(self.step_grid_flow(state, action))
            available = (
                box_index is not None
                and not stand_occupied
                and not target_occupied
                and stand_reachable
                and not child_tier_a_deadlocked
            )
            blockers: list[str] = []
            if stand_occupied:
                blockers.append("stand_cell_occupied_by_box")
                blocker_cells.add(stand_cell)
                blocker_boxes.add(stand_cell)
            elif stand_is_floor and not stand_reachable:
                blockers.append("stand_cell_unreachable_by_player")
                blocker_cells.add(stand_cell)
            if target_occupied:
                blockers.append("target_occupied")
                blocker_cells.add(target_cell)
                blocker_boxes.add(target_cell)
            if child_tier_a_deadlocked:
                blockers.append("child_tier_a_deadlocked")
            if available:
                blockers.append("available")
            release_needed = not available
            barrier = (
                8.0 * float(stand_occupied)
                + 7.0 * float(target_occupied)
                + 4.0 * float(stand_is_floor and not stand_occupied and not stand_reachable)
                + 10.0 * float(child_tier_a_deadlocked)
                - 3.0 * float(available)
            )

            totals["candidate_push_count"] += 1
            totals["release_need_count"] += int(release_needed)
            totals["stand_blocker_count"] += int(stand_occupied)
            totals["target_blocker_count"] += int(target_occupied)
            totals["unreachable_stand_count"] += int(
                stand_is_floor and not stand_occupied and not stand_reachable
            )
            totals["reachable_stand_count"] += int(stand_reachable)
            totals["monotone_available_count"] += int(available)
            totals["deadlocked_candidate_count"] += int(child_tier_a_deadlocked)
            totals["release_barrier_cost"] += float(barrier)
            required_stands.add(stand_cell)
            candidate_rows.append(
                {
                    "direction": int(direction_index),
                    "direction_name": ACTION_NAMES[direction_index],
                    "stand_cell": list(stand_cell),
                    "target_cell": list(target_cell),
                    "before_distance": float(before_distance),
                    "after_distance": float(after_distance),
                    "distance_delta": float(before_distance - after_distance),
                    "stand_is_floor": bool(stand_is_floor),
                    "target_is_floor": bool(target_is_floor),
                    "stand_occupied_by_box": bool(stand_occupied),
                    "target_occupied_by_box": bool(target_occupied),
                    "stand_reachable_by_player": bool(stand_reachable),
                    "child_tier_a_deadlocked": bool(child_tier_a_deadlocked),
                    "monotone_available": bool(available),
                    "release_needed": bool(release_needed),
                    "blocker_codes": blockers,
                    "barrier_cost": float(barrier),
                }
            )

        return candidate_rows, {
            **totals,
            "required_stand_cells": [list(cell) for cell in sorted(required_stands)],
            "release_blocker_cells": [list(cell) for cell in sorted(blocker_cells)],
            "release_blocker_boxes": [list(cell) for cell in sorted(blocker_boxes)],
            "release_blocker_box_count": int(len(blocker_boxes)),
        }

    def room_release_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
    ) -> dict[str, Any]:
        """State-derived active-box release plan consumed by the A9 planner."""

        fill = self.room_fill_plan_summary(state, commitment_state)
        active_rows = self._active_assignment_rows(fill)
        rooms: list[dict[str, Any]] = []
        totals = {
            "active_release_assignment_count": 0,
            "active_release_candidate_push_count": 0,
            "active_release_need_count": 0,
            "active_release_stand_blocker_count": 0,
            "active_release_target_blocker_count": 0,
            "active_release_unreachable_stand_count": 0,
            "active_release_reachable_stand_count": 0,
            "active_release_monotone_available_count": 0,
            "active_release_deadlocked_candidate_count": 0,
            "release_blocker_box_count": 0,
            "release_barrier_cost": 0.0,
        }
        blocker_boxes: set[Cell] = set()
        by_room: dict[int, list[dict[str, Any]]] = {}
        for active in active_rows:
            candidate_rows, row_totals = self._release_candidate_rows(
                state,
                box_cell=active["box_cell"],
                slot_cell=active["slot_cell"],
            )
            blocker_boxes.update(tuple(cell) for cell in row_totals["release_blocker_boxes"])
            totals["active_release_assignment_count"] += 1
            totals["active_release_candidate_push_count"] += int(row_totals["candidate_push_count"])
            totals["active_release_need_count"] += int(row_totals["release_need_count"])
            totals["active_release_stand_blocker_count"] += int(row_totals["stand_blocker_count"])
            totals["active_release_target_blocker_count"] += int(row_totals["target_blocker_count"])
            totals["active_release_unreachable_stand_count"] += int(row_totals["unreachable_stand_count"])
            totals["active_release_reachable_stand_count"] += int(row_totals["reachable_stand_count"])
            totals["active_release_monotone_available_count"] += int(row_totals["monotone_available_count"])
            totals["active_release_deadlocked_candidate_count"] += int(row_totals["deadlocked_candidate_count"])
            totals["release_barrier_cost"] += float(row_totals["release_barrier_cost"])
            by_room.setdefault(int(active["room_id"]), []).append(
                {
                    "fill_phase": active.get("fill_phase"),
                    "active_slot": active.get("active_slot"),
                    "box_cell": list(active["box_cell"]),
                    "slot_cell": list(active["slot_cell"]),
                    "pull_distance": active.get("pull_distance"),
                    "assignment_cost": float(active["cost"]),
                    "candidate_pushes": candidate_rows,
                    "totals": row_totals,
                }
            )
        totals["release_blocker_box_count"] = int(len(blocker_boxes))
        for room in fill["rooms"]:
            room_id = int(room["room_id"])
            assignments = by_room.get(room_id, [])
            room_totals = {
                "active_release_assignment_count": len(assignments),
                "active_release_candidate_push_count": sum(
                    int(row["totals"]["candidate_push_count"]) for row in assignments
                ),
                "active_release_need_count": sum(
                    int(row["totals"]["release_need_count"]) for row in assignments
                ),
                "active_release_stand_blocker_count": sum(
                    int(row["totals"]["stand_blocker_count"]) for row in assignments
                ),
                "active_release_target_blocker_count": sum(
                    int(row["totals"]["target_blocker_count"]) for row in assignments
                ),
                "active_release_unreachable_stand_count": sum(
                    int(row["totals"]["unreachable_stand_count"]) for row in assignments
                ),
                "active_release_reachable_stand_count": sum(
                    int(row["totals"]["reachable_stand_count"]) for row in assignments
                ),
                "active_release_monotone_available_count": sum(
                    int(row["totals"]["monotone_available_count"]) for row in assignments
                ),
                "release_barrier_cost": sum(float(row["totals"]["release_barrier_cost"]) for row in assignments),
            }
            rooms.append(
                {
                    "room_id": room_id,
                    "fill_phase": room.get("fill_phase"),
                    "active_slot": room.get("active_slot"),
                    "totals": room_totals,
                    "active_release_assignments": assignments,
                }
            )
        return {
            "provider_id": self.provider_id,
            "lower_is_better": True,
            "plan_source": "state_derived_active_box_stand_cell_release_plan",
            "fill_plan_totals": fill["totals"],
            "totals": totals,
            "rooms": rooms,
        }


class SokobanRoomAccessPlanGFTSDModel(SokobanRoomReleasePlanGFTSDModel):
    """A10 room-aware SD model with lookahead access preservation.

    A9 diagnoses the active slot only.  A10 extends the same release machinery
    to the next open deep slots so that a future active box's required player
    stand cell can be protected before the slot becomes current.
    """

    provider_id = "room_access_plan_gft_sd_v1"

    def _lookahead_access_assignments(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
        *,
        lookahead_slots: int,
    ) -> list[dict[str, Any]]:
        commitments = self.normalize_room_commitment_state(state, commitment_state)
        boxes = set(state.boxes)
        committed_by_slot = {
            commitment.slot_cell: commitment
            for commitment in commitments.commitments
        }
        committed_boxes = {
            commitment.box_cell
            for commitment in commitments.commitments
        }
        rows: list[dict[str, Any]] = []
        for room in self.goal_rooms:
            queue = self.room_fill_plan_queue(room)
            deep_open_rows = [
                row for row in queue if row["deep_slot"] and row["slot_cell"] not in boxes
            ]
            if not deep_open_rows:
                deep_open_rows = [
                    row for row in queue if row["slot_cell"] not in boxes
                ]
            selected = deep_open_rows[: max(int(lookahead_slots), 1)]
            deep_filled_cells = {
                row["slot_cell"]
                for row in queue
                if row["deep_slot"] and row["slot_cell"] in boxes
            }
            used_boxes: set[Cell] = set()
            source_boxes = set(boxes) - deep_filled_cells
            for horizon_index, row in enumerate(selected):
                slot = row["slot_obj"]
                slot_cell = row["slot_cell"]
                commitment = committed_by_slot.get(slot_cell)
                assigned_box = None
                assigned_distance = math.inf
                committed_assignment = False
                if commitment is not None and commitment.box_cell in boxes:
                    assigned_box = commitment.box_cell
                    assigned_distance = self._slot_pull_distance(assigned_box, slot_cell)
                    committed_assignment = True
                else:
                    for box in sorted(source_boxes - used_boxes - committed_boxes):
                        distance = self._slot_pull_distance(box, slot_cell)
                        if distance < assigned_distance:
                            assigned_box = box
                            assigned_distance = distance
                feasible = assigned_box is not None and math.isfinite(assigned_distance)
                if assigned_box is not None:
                    used_boxes.add(assigned_box)
                rows.append(
                    {
                        "room_id": int(room.room_id),
                        "horizon_index": int(horizon_index),
                        "current_active": bool(horizon_index == 0),
                        "fill_phase": "deep_slots" if row["deep_slot"] else "shallow_slots",
                        "slot": row["slot"],
                        "slot_cell": slot_cell,
                        "box_cell": assigned_box,
                        "pull_distance": None if not feasible else float(assigned_distance),
                        "assignment_cost": (
                            float(assigned_distance)
                            if feasible
                            else float(self.room_unreachable_transport_penalty)
                        ),
                        "feasible": bool(feasible),
                        "committed_assignment": bool(committed_assignment),
                    }
                )
        return rows

    def room_access_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
        *,
        lookahead_slots: int = 3,
    ) -> dict[str, Any]:
        """Lookahead release/access summary for active and future slots."""

        assignments = self._lookahead_access_assignments(
            state,
            commitment_state,
            lookahead_slots=lookahead_slots,
        )
        rooms: list[dict[str, Any]] = []
        by_room: dict[int, list[dict[str, Any]]] = {}
        totals = {
            "access_assignment_count": 0,
            "access_future_assignment_count": 0,
            "access_candidate_push_count": 0,
            "access_need_count": 0,
            "access_future_need_count": 0,
            "access_stand_blocker_count": 0,
            "access_future_stand_blocker_count": 0,
            "access_target_blocker_count": 0,
            "access_unreachable_stand_count": 0,
            "access_reachable_stand_count": 0,
            "access_monotone_available_count": 0,
            "access_blocker_box_count": 0,
            "access_barrier_cost": 0.0,
            "access_future_barrier_cost": 0.0,
        }
        blocker_boxes: set[Cell] = set()
        for assignment in assignments:
            box_cell = assignment["box_cell"]
            if box_cell is None or not assignment["feasible"]:
                candidate_rows: list[dict[str, Any]] = []
                row_totals = {
                    "candidate_push_count": 0,
                    "release_need_count": 1,
                    "stand_blocker_count": 0,
                    "target_blocker_count": 0,
                    "unreachable_stand_count": 0,
                    "reachable_stand_count": 0,
                    "monotone_available_count": 0,
                    "deadlocked_candidate_count": 0,
                    "release_barrier_cost": float(self.room_unreachable_transport_penalty),
                    "required_stand_cells": [],
                    "release_blocker_cells": [],
                    "release_blocker_boxes": [],
                    "release_blocker_box_count": 0,
                }
            else:
                candidate_rows, row_totals = self._release_candidate_rows(
                    state,
                    box_cell=box_cell,
                    slot_cell=assignment["slot_cell"],
                )
            is_future = not bool(assignment["current_active"])
            horizon_weight = 1.0 / float(1 + int(assignment["horizon_index"]))
            weighted_barrier = float(row_totals["release_barrier_cost"]) * horizon_weight
            blocker_boxes.update(tuple(cell) for cell in row_totals["release_blocker_boxes"])
            totals["access_assignment_count"] += 1
            totals["access_future_assignment_count"] += int(is_future)
            totals["access_candidate_push_count"] += int(row_totals["candidate_push_count"])
            totals["access_need_count"] += int(row_totals["release_need_count"])
            totals["access_future_need_count"] += int(is_future) * int(row_totals["release_need_count"])
            totals["access_stand_blocker_count"] += int(row_totals["stand_blocker_count"])
            totals["access_future_stand_blocker_count"] += int(is_future) * int(
                row_totals["stand_blocker_count"]
            )
            totals["access_target_blocker_count"] += int(row_totals["target_blocker_count"])
            totals["access_unreachable_stand_count"] += int(row_totals["unreachable_stand_count"])
            totals["access_reachable_stand_count"] += int(row_totals["reachable_stand_count"])
            totals["access_monotone_available_count"] += int(row_totals["monotone_available_count"])
            totals["access_barrier_cost"] += float(weighted_barrier)
            totals["access_future_barrier_cost"] += float(weighted_barrier) if is_future else 0.0
            by_room.setdefault(int(assignment["room_id"]), []).append(
                {
                    **assignment,
                    "box_cell": None if box_cell is None else list(box_cell),
                    "slot_cell": list(assignment["slot_cell"]),
                    "candidate_pushes": candidate_rows,
                    "totals": row_totals,
                    "horizon_weight": float(horizon_weight),
                    "weighted_barrier_cost": float(weighted_barrier),
                }
            )
        totals["access_blocker_box_count"] = int(len(blocker_boxes))
        for room in self.goal_rooms:
            rows = by_room.get(int(room.room_id), [])
            room_totals = {
                "access_assignment_count": len(rows),
                "access_future_assignment_count": sum(int(not row["current_active"]) for row in rows),
                "access_need_count": sum(int(row["totals"]["release_need_count"]) for row in rows),
                "access_future_need_count": sum(
                    int(not row["current_active"]) * int(row["totals"]["release_need_count"])
                    for row in rows
                ),
                "access_stand_blocker_count": sum(int(row["totals"]["stand_blocker_count"]) for row in rows),
                "access_future_stand_blocker_count": sum(
                    int(not row["current_active"]) * int(row["totals"]["stand_blocker_count"])
                    for row in rows
                ),
                "access_monotone_available_count": sum(
                    int(row["totals"]["monotone_available_count"]) for row in rows
                ),
                "access_barrier_cost": sum(float(row["weighted_barrier_cost"]) for row in rows),
            }
            rooms.append(
                {
                    "room_id": int(room.room_id),
                    "lookahead_slots": int(lookahead_slots),
                    "totals": room_totals,
                    "access_assignments": rows,
                }
            )
        return {
            "provider_id": self.provider_id,
            "lower_is_better": True,
            "plan_source": "state_derived_lookahead_slot_access_preservation",
            "lookahead_slots": int(lookahead_slots),
            "totals": totals,
            "rooms": rooms,
        }


class SokobanRoomSchedulePlanGFTSDModel(SokobanRoomAccessPlanGFTSDModel):
    """A11 room-aware SD model with explicit multi-box scheduling state.

    A10 exposes access blockers for current and future slots, but the planner
    still has to infer which other boxes should be moved to release those
    blockers.  A11 turns that missing relation into a state-derived dependency
    graph: required stand cells, blocker boxes, unreachable stand cells, and
    legal pushes that can release or damage the current schedule.
    """

    provider_id = "room_schedule_plan_gft_sd_v1"

    def room_schedule_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None,
        *,
        lookahead_slots: int = 3,
    ) -> dict[str, Any]:
        access = self.room_access_plan_summary(
            state,
            commitment_state,
            lookahead_slots=lookahead_slots,
        )
        boxes = set(state.boxes)
        dependencies: list[dict[str, Any]] = []
        critical_stand_cells: set[Cell] = set()
        future_critical_stand_cells: set[Cell] = set()
        critical_slot_cells: set[Cell] = set()
        blocker_boxes: set[Cell] = set()
        unreachable_stands: set[Cell] = set()
        deadlocked_candidates = 0
        target_blockers = 0

        for room in access["rooms"]:
            for assignment in room.get("access_assignments", []):
                slot_cell = tuple(assignment["slot_cell"])
                critical_slot_cells.add(slot_cell)
                is_future = not bool(assignment.get("current_active", False))
                box_cell = assignment.get("box_cell")
                assigned_box = None if box_cell is None else tuple(box_cell)
                for candidate in assignment.get("candidate_pushes", []):
                    stand_cell = tuple(candidate["stand_cell"])
                    target_cell = tuple(candidate["target_cell"])
                    critical_stand_cells.add(stand_cell)
                    if is_future:
                        future_critical_stand_cells.add(stand_cell)
                    blockers = list(candidate.get("blocker_codes", []))
                    release_needed = bool(candidate.get("release_needed", False))
                    if not release_needed:
                        continue
                    dependency_kinds: list[str] = []
                    blocker_cells: list[Cell] = []
                    if candidate.get("stand_occupied_by_box"):
                        dependency_kinds.append("stand_cell_occupied_by_box")
                        blocker_boxes.add(stand_cell)
                        blocker_cells.append(stand_cell)
                    if candidate.get("target_occupied_by_box"):
                        dependency_kinds.append("target_occupied_by_box")
                        blocker_boxes.add(target_cell)
                        blocker_cells.append(target_cell)
                        target_blockers += 1
                    if (
                        candidate.get("stand_is_floor")
                        and not candidate.get("stand_occupied_by_box")
                        and not candidate.get("stand_reachable_by_player")
                    ):
                        dependency_kinds.append("stand_cell_unreachable_by_player")
                        unreachable_stands.add(stand_cell)
                    if candidate.get("child_tier_a_deadlocked"):
                        dependency_kinds.append("child_tier_a_deadlocked")
                        deadlocked_candidates += 1
                    dependencies.append(
                        {
                            "room_id": int(assignment["room_id"]),
                            "horizon_index": int(assignment["horizon_index"]),
                            "future": bool(is_future),
                            "assigned_box": None if assigned_box is None else list(assigned_box),
                            "slot_cell": list(slot_cell),
                            "stand_cell": list(stand_cell),
                            "target_cell": list(target_cell),
                            "blocker_cells": [list(cell) for cell in blocker_cells],
                            "dependency_kinds": dependency_kinds or blockers,
                            "candidate_direction": candidate.get("direction_name"),
                            "candidate_barrier_cost": float(candidate.get("barrier_cost", 0.0)),
                        }
                    )

        protected_cells = set(critical_stand_cells)
        release_actions: list[dict[str, Any]] = []
        destructive_actions: list[dict[str, Any]] = []
        for action in self.legal_flows(state):
            source = state.boxes[action.box_index]
            target = add_cells(source, DIRECTIONS[action.direction])
            child = self.step_grid_flow(state, action)
            if self.is_tier_a_deadlocked(child):
                continue
            releases_blocker = source in blocker_boxes and target not in protected_cells
            protects_future = source in future_critical_stand_cells and target not in future_critical_stand_cells
            damages_schedule = target in protected_cells and source not in protected_cells
            if releases_blocker or protects_future:
                release_actions.append(
                    {
                        "box_index": int(action.box_index),
                        "direction": int(action.direction),
                        "direction_name": ACTION_NAMES[action.direction],
                        "source": list(source),
                        "target": list(target),
                        "releases_blocker": bool(releases_blocker),
                        "protects_future_stand": bool(protects_future),
                    }
                )
            if damages_schedule:
                destructive_actions.append(
                    {
                        "box_index": int(action.box_index),
                        "direction": int(action.direction),
                        "direction_name": ACTION_NAMES[action.direction],
                        "source": list(source),
                        "target": list(target),
                    }
                )

        critical_box_count = len(boxes & critical_stand_cells)
        future_critical_box_count = len(boxes & future_critical_stand_cells)
        dependency_count = len(dependencies)
        barrier_cost = (
            float(access["totals"]["access_barrier_cost"])
            + 6.0 * dependency_count
            + 5.0 * len(blocker_boxes)
            + 4.0 * len(unreachable_stands)
            + 8.0 * critical_box_count
            + 5.0 * future_critical_box_count
            + 6.0 * target_blockers
            + 8.0 * deadlocked_candidates
            + 4.0 * len(destructive_actions)
            - 2.0 * len(release_actions)
        )
        totals = {
            "schedule_assignment_count": int(access["totals"]["access_assignment_count"]),
            "schedule_future_assignment_count": int(access["totals"]["access_future_assignment_count"]),
            "dependency_count": int(dependency_count),
            "future_dependency_count": int(sum(int(row["future"]) for row in dependencies)),
            "blocker_box_count": int(len(blocker_boxes)),
            "critical_stand_cell_count": int(len(critical_stand_cells)),
            "future_critical_stand_cell_count": int(len(future_critical_stand_cells)),
            "critical_stand_box_count": int(critical_box_count),
            "future_critical_stand_box_count": int(future_critical_box_count),
            "unreachable_stand_count": int(len(unreachable_stands)),
            "target_blocker_count": int(target_blockers),
            "deadlocked_candidate_count": int(deadlocked_candidates),
            "release_action_count": int(len(release_actions)),
            "destructive_action_count": int(len(destructive_actions)),
            "schedule_barrier_cost": float(barrier_cost),
        }
        return {
            "provider_id": self.provider_id,
            "lower_is_better": True,
            "plan_source": "state_derived_multi_box_release_dependency_graph",
            "lookahead_slots": int(lookahead_slots),
            "access_plan_totals": access["totals"],
            "totals": totals,
            "critical_stand_cells": [list(cell) for cell in sorted(critical_stand_cells)],
            "future_critical_stand_cells": [list(cell) for cell in sorted(future_critical_stand_cells)],
            "critical_slot_cells": [list(cell) for cell in sorted(critical_slot_cells)],
            "blocker_boxes": [list(cell) for cell in sorted(blocker_boxes)],
            "unreachable_stand_cells": [list(cell) for cell in sorted(unreachable_stands)],
            "dependencies": dependencies,
            "release_actions": release_actions,
            "destructive_actions": destructive_actions,
            "rooms": access["rooms"],
        }
