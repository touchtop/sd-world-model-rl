"""Semantic room-commitment MCTS for hard Sokoban instances.

This planner is intentionally separate from the existing Boxoban mainline
planner.  It consumes :class:`SokobanRoomGFTSDModel` and expands semantic
children of the form ``(physical push, room-slot commitment)``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from .env import ACTION_NAMES, DIRECTIONS, Cell, add_cells
from .gft_sd import DEFAULT_GFT_PRIOR_WEIGHTS
from .room_gft_sd import (
    RoomCommitmentState,
    RoomSlotOption,
    SokobanRoomAccessPlanGFTSDModel,
    SokobanRoomFillPlanGFTSDModel,
    SokobanRoomGFTSDModel,
    SokobanRoomReleasePlanGFTSDModel,
    SokobanRoomSchedulePlanGFTSDModel,
)
from .sd_model import SokobanPushAction, SokobanState


LEARNED_SD_CONSUMER_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "goal_delta",
    "release_action_count_delta",
    "dependency_total_delta",
    "target_blocker_count_delta",
    "slot_schedule_cost_delta",
    "schedule_barrier_cost_delta",
    "legal_push_count_delta",
    "after_legal_push_count",
    "after_dependency_total",
    "after_target_blocker_count",
    "after_release_action_count",
    "after_goal_locked_stand_blocker_count",
    "after_locked_stand_blocker_count",
    "after_stand_release_option_count",
    "stage_before_fill",
    "stage_before_release",
    "stage_before_recovery",
    "stage_after_release",
    "stage_after_recovery",
    "phase_progress",
    "no_progress",
    "stagnant_context",
    "repeated_schedule_signature",
    "repeated_edge",
    "reverse_recent_edge",
    "no_release_context",
    "pseudo_milestone_loop",
    "hard_negative",
    "soft_negative",
    "trap_risk",
    "severe_trap_risk",
    "generalized_failure_risk",
    "trap_predecessor",
    "stand_lock_risk",
    "recoverable_no_progress",
    "unrecoverable_no_progress",
    "low_margin",
    "action_direction_up",
    "action_direction_down",
    "action_direction_left",
    "action_direction_right",
    "action_box_index_norm",
    "action_flat_index_norm",
)


def _softmax(scores: list[float], temperature: float, uniform_mix: float) -> list[float]:
    if not scores:
        return []
    temperature = max(float(temperature), 1e-9)
    uniform_mix = min(max(float(uniform_mix), 0.0), 1.0)
    peak = max(scores)
    exps = [math.exp((score - peak) / temperature) for score in scores]
    denom = sum(exps)
    uniform = 1.0 / len(scores)
    if denom <= 0.0 or not math.isfinite(denom):
        probs = [uniform for _ in scores]
    else:
        probs = [value / denom for value in exps]
    return [(1.0 - uniform_mix) * prob + uniform_mix * uniform for prob in probs]


@dataclass(frozen=True)
class RoomSemanticAction:
    """One MCTS action: a physical push plus an optional slot commitment."""

    physical_action: SokobanPushAction
    option: RoomSlotOption

    @property
    def key(self) -> tuple:
        return int(self.physical_action.flat_index), self.option.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_action": {
                "box_index": int(self.physical_action.box_index),
                "direction": int(self.physical_action.direction),
                "direction_name": ACTION_NAMES[self.physical_action.direction],
                "flat_index": int(self.physical_action.flat_index),
            },
            "commitment_option": self.option.to_dict(),
            "semantic_key": [self.physical_action.flat_index, list(self.option.key)],
        }


@dataclass
class RoomSemanticCandidate:
    action: RoomSemanticAction
    child_state: SokobanState
    child_commitment_state: RoomCommitmentState
    prior: float = 0.0
    prior_score: float = 0.0
    physical_prior_score: float = 0.0
    room_prior_score: float = 0.0
    room_action_summary: dict[str, Any] = field(default_factory=dict)
    room_commitment_summary: dict[str, Any] = field(default_factory=dict)
    room_slot_progress_delta: float = 0.0
    room_slot_transport_delta: float = 0.0
    room_slot_committed_transport_delta: float = 0.0
    room_slot_frontier_transport_delta: float = 0.0
    room_slot_barrier_delta: float = 0.0
    room_slot_progress_summary: dict[str, Any] = field(default_factory=dict)
    room_loop_penalty_score: float = 0.0
    room_loop_summary: dict[str, Any] = field(default_factory=dict)
    room_fill_plan_score: float = 0.0
    room_fill_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_release_plan_score: float = 0.0
    room_release_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_access_plan_score: float = 0.0
    room_access_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_schedule_plan_score: float = 0.0
    room_schedule_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_schedule_stall_score: float = 0.0
    room_schedule_stall_summary: dict[str, Any] = field(default_factory=dict)
    room_access_recovery_score: float = 0.0
    room_access_recovery_summary: dict[str, Any] = field(default_factory=dict)
    room_phase_progress_score: float = 0.0
    room_phase_progress_summary: dict[str, Any] = field(default_factory=dict)
    room_recoverability_score: float = 0.0
    room_recoverability_summary: dict[str, Any] = field(default_factory=dict)
    room_stand_lock_score: float = 0.0
    room_stand_lock_summary: dict[str, Any] = field(default_factory=dict)
    room_stage_score: float = 0.0
    room_stage_summary: dict[str, Any] = field(default_factory=dict)
    room_pre_release_trap_score: float = 0.0
    room_pre_release_trap_summary: dict[str, Any] = field(default_factory=dict)
    room_sd_generalization_score: float = 0.0
    room_sd_generalization_summary: dict[str, Any] = field(default_factory=dict)
    room_sd_value_score: float = 0.0
    room_sd_value_summary: dict[str, Any] = field(default_factory=dict)
    room_trace_anchor_score: float = 0.0
    room_trace_anchor_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomSemanticEdge:
    action: RoomSemanticAction
    child_state: SokobanState
    child_commitment_state: RoomCommitmentState
    child_node: "RoomSemanticNode"
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    prior_score: float = 0.0
    room_prior_score: float = 0.0
    room_slot_progress_delta: float = 0.0
    room_slot_transport_delta: float = 0.0
    room_slot_committed_transport_delta: float = 0.0
    room_slot_frontier_transport_delta: float = 0.0
    room_slot_barrier_delta: float = 0.0
    room_slot_progress_summary: dict[str, Any] = field(default_factory=dict)
    room_loop_penalty_score: float = 0.0
    room_loop_summary: dict[str, Any] = field(default_factory=dict)
    room_fill_plan_score: float = 0.0
    room_fill_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_release_plan_score: float = 0.0
    room_release_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_access_plan_score: float = 0.0
    room_access_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_schedule_plan_score: float = 0.0
    room_schedule_plan_summary: dict[str, Any] = field(default_factory=dict)
    room_schedule_stall_score: float = 0.0
    room_schedule_stall_summary: dict[str, Any] = field(default_factory=dict)
    room_access_recovery_score: float = 0.0
    room_access_recovery_summary: dict[str, Any] = field(default_factory=dict)
    room_phase_progress_score: float = 0.0
    room_phase_progress_summary: dict[str, Any] = field(default_factory=dict)
    room_recoverability_score: float = 0.0
    room_recoverability_summary: dict[str, Any] = field(default_factory=dict)
    room_stand_lock_score: float = 0.0
    room_stand_lock_summary: dict[str, Any] = field(default_factory=dict)
    room_stage_score: float = 0.0
    room_stage_summary: dict[str, Any] = field(default_factory=dict)
    room_pre_release_trap_score: float = 0.0
    room_pre_release_trap_summary: dict[str, Any] = field(default_factory=dict)
    room_sd_generalization_score: float = 0.0
    room_sd_generalization_summary: dict[str, Any] = field(default_factory=dict)
    room_sd_value_score: float = 0.0
    room_sd_value_summary: dict[str, Any] = field(default_factory=dict)
    room_trace_anchor_score: float = 0.0
    room_trace_anchor_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class RoomSemanticNode:
    state: SokobanState
    commitment_state: RoomCommitmentState
    depth_from_root: int
    untried: list[RoomSemanticCandidate] | None = None
    children: dict[tuple, RoomSemanticEdge] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    pruned_on_init: int = 0


class RoomSemanticGFTSDMCTSPlanner:
    """MCTS over physical Sokoban states plus room-slot commitments."""

    provider_id = "room_semantic_gft_sdmcts_v1"

    def __init__(
        self,
        model: SokobanRoomGFTSDModel,
        *,
        horizon: int,
        budget: int,
        rng: random.Random,
        uct_c: float,
        gamma: float,
        goal_value: float,
        push_cost: float,
        deadend_value: float,
        constraint_mode: str = "tier_a",
        select_by: str = "visits",
        revisit_penalty: float = 0.0,
        visited_keys: set | None = None,
        transposition: bool = True,
        gft_prior_weights: dict[str, float] | None = None,
        gft_prior_temperature: float = 2.0,
        gft_prior_uniform_mix: float = 0.1,
        gft_sd_leaf_weights: dict[str, float] | None = None,
        include_uncommitted_semantic_option: bool = False,
        semantic_commitment_boundary_distance: int = 6,
        room_slot_depth_weight: float = 0.25,
        room_commitment_gain_weight: float = 1.5,
        room_frontier_fill_bonus: float = 2.0,
        room_uncommitted_entry_penalty: float = 3.0,
        room_nonfrontier_fill_penalty: float = 4.0,
        room_order_debt_penalty: float = 4.0,
        room_blocked_slot_penalty: float = 3.0,
        room_leaf_commitment_bonus: float = 0.5,
        room_leaf_uncommitted_penalty: float = 2.0,
        room_leaf_order_penalty: float = 4.0,
        room_leaf_blocked_penalty: float = 3.0,
        room_leaf_nonfrontier_commitment_penalty: float = 2.0,
    ):
        self.model = model
        self.horizon = int(horizon)
        self.budget = int(budget)
        self.rng = rng
        self.uct_c = float(uct_c)
        self.gamma = float(gamma)
        self.goal_value = float(goal_value)
        self.push_cost = float(push_cost)
        self.deadend_value = float(deadend_value)
        self.constraint_mode = str(constraint_mode)
        self.select_by = str(select_by)
        self.revisit_penalty = float(revisit_penalty)
        self.visited_keys = visited_keys if visited_keys is not None else set()
        self.transposition = bool(transposition)
        self.gft_prior_weights = {**DEFAULT_GFT_PRIOR_WEIGHTS, **dict(gft_prior_weights or {})}
        self.gft_prior_temperature = max(float(gft_prior_temperature), 1e-9)
        self.gft_prior_uniform_mix = min(max(float(gft_prior_uniform_mix), 0.0), 1.0)
        self.gft_sd_leaf_weights = dict(gft_sd_leaf_weights or {})
        self.include_uncommitted_semantic_option = bool(include_uncommitted_semantic_option)
        self.semantic_commitment_boundary_distance = max(int(semantic_commitment_boundary_distance), 0)
        self.room_slot_depth_weight = float(room_slot_depth_weight)
        self.room_commitment_gain_weight = float(room_commitment_gain_weight)
        self.room_frontier_fill_bonus = float(room_frontier_fill_bonus)
        self.room_uncommitted_entry_penalty = float(room_uncommitted_entry_penalty)
        self.room_nonfrontier_fill_penalty = float(room_nonfrontier_fill_penalty)
        self.room_order_debt_penalty = float(room_order_debt_penalty)
        self.room_blocked_slot_penalty = float(room_blocked_slot_penalty)
        self.room_leaf_commitment_bonus = float(room_leaf_commitment_bonus)
        self.room_leaf_uncommitted_penalty = float(room_leaf_uncommitted_penalty)
        self.room_leaf_order_penalty = float(room_leaf_order_penalty)
        self.room_leaf_blocked_penalty = float(room_leaf_blocked_penalty)
        self.room_leaf_nonfrontier_commitment_penalty = float(room_leaf_nonfrontier_commitment_penalty)
        self._node_registry: dict[tuple, RoomSemanticNode] = {}
        self._leaf_cache: dict[tuple, float] = {}
        self._summary_cache: dict[tuple, dict[str, Any]] = {}
        self.deadlock_prunes = 0
        slot_distances = [
            slot.entry_distance
            for room in self.model.goal_rooms
            for slot in room.slots
            if slot.entry_distance < 10**8
        ]
        self._max_slot_entry_distance = max(slot_distances) if slot_distances else 1

    def _summary_cache_key(
        self,
        name: str,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        *extra: Any,
    ) -> tuple:
        return (
            str(name),
            self.model.gft_canonical_key_cached(state),
            commitment_state.key,
            tuple(extra),
        )

    def state_key(self, state: SokobanState, commitment_state: RoomCommitmentState | None = None) -> tuple:
        return self.model.room_commitment_key(state, commitment_state)

    def node_for(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        depth: int,
    ) -> RoomSemanticNode:
        normalized = self.model.normalize_room_commitment_state(state, commitment_state)
        if not self.transposition:
            return RoomSemanticNode(state, normalized, depth)
        key = self.state_key(state, normalized)
        node = self._node_registry.get(key)
        if node is None:
            node = RoomSemanticNode(state, normalized, depth)
            self._node_registry[key] = node
        return node

    def is_blocked(self, state: SokobanState) -> bool:
        if self.constraint_mode == "none" or self.model.is_solved(state):
            return False
        if self.constraint_mode == "tier_a":
            return self.model.is_tier_a_deadlocked(state)
        return self.model.is_deadlocked(state)

    def transition_cost(self, child_state: SokobanState) -> float:
        cost = self.push_cost
        if self.revisit_penalty > 0.0 and self.model.gft_canonical_key_cached(child_state) in self.visited_keys:
            cost += self.revisit_penalty
        return cost

    def candidate_actions(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> tuple[list[RoomSemanticCandidate], int]:
        candidates: list[RoomSemanticCandidate] = []
        pruned = 0
        commitment_state = self.model.normalize_room_commitment_state(state, commitment_state)
        for physical_action in self.model.legal_flows(state):
            child = self.model.step_grid_flow(state, physical_action)
            if self.is_blocked(child):
                pruned += 1
                continue
            options = self.model.room_semantic_slot_options(
                state,
                physical_action,
                child,
                commitment_state=commitment_state,
                include_uncommitted_option=self.include_uncommitted_semantic_option,
                commitment_boundary_distance=self.semantic_commitment_boundary_distance,
            )
            for option in options:
                child_commitment = self.model.advance_room_commitment_state(
                    state,
                    commitment_state,
                    physical_action,
                    option,
                    child,
                )
                semantic_action = RoomSemanticAction(physical_action=physical_action, option=option)
                candidates.append(
                    RoomSemanticCandidate(
                        action=semantic_action,
                        child_state=child,
                        child_commitment_state=child_commitment,
                    )
                )
        self.assign_priors(state, commitment_state, candidates)
        return candidates, pruned

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        option = candidate.action.option
        physical = candidate.action.physical_action
        room_action = self.model.room_action_summary(state, physical, candidate.child_state)
        before_commit = self.model.room_commitment_summary(state, commitment_state)
        after_commit = self.model.room_commitment_summary(candidate.child_state, candidate.child_commitment_state)
        impact_codes = set(room_action["impact_codes"])
        score = 0.0
        if option.kind == "slot" and option.entry_distance is not None:
            score += self.room_slot_depth_weight * (
                float(option.entry_distance) / max(float(self._max_slot_entry_distance), 1.0)
            )
        score += self.room_commitment_gain_weight * max(
            0,
            int(after_commit["committed_box_count"]) - int(before_commit["committed_box_count"]),
        )
        if "FILL_FRONTIER_SLOT" in impact_codes:
            score += self.room_frontier_fill_bonus
        if option.kind == "uncommitted" and (
            "ENTER_ROOM_ENTRY" in impact_codes or "CROSS_ROOM_BOUNDARY" in impact_codes
        ):
            score -= self.room_uncommitted_entry_penalty
        if "FILL_NONFRONTIER_SLOT" in impact_codes:
            score -= self.room_nonfrontier_fill_penalty
        if "INCREASE_ROOM_ORDER_DEBT" in impact_codes:
            score -= self.room_order_debt_penalty
        if "BLOCK_UNFILLED_SLOT" in impact_codes:
            score -= self.room_blocked_slot_penalty
        candidate.room_action_summary = room_action
        candidate.room_commitment_summary = after_commit
        return float(score)

    def assign_priors(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidates: list[RoomSemanticCandidate],
    ) -> None:
        if not candidates:
            return
        physical_rows = self.model.gft_action_prior_rows(
            state,
            [(candidate.action.physical_action, candidate.child_state) for candidate in candidates],
            visited_keys=self.visited_keys,
            weights=self.gft_prior_weights,
            temperature=self.gft_prior_temperature,
            uniform_mix=0.0,
        )
        scores: list[float] = []
        for candidate, row in zip(candidates, physical_rows):
            room_score = self._room_prior_score(state, commitment_state, candidate)
            candidate.physical_prior_score = float(row["score"])
            candidate.room_prior_score = room_score
            candidate.prior_score = float(row["score"] + room_score)
            scores.append(candidate.prior_score)
        for candidate, prior in zip(
            candidates,
            _softmax(scores, self.gft_prior_temperature, self.gft_prior_uniform_mix),
        ):
            candidate.prior = float(prior)

    def gft_sd_leaf_value(self, state: SokobanState) -> float:
        progress = self.model.gft_transport_progress_cached(state)
        if not math.isfinite(progress):
            base = self.deadend_value
        elif self.gamma == 1.0:
            base = self.goal_value - self.push_cost * progress
        else:
            discount = self.gamma**progress
            base = discount * self.goal_value - self.push_cost * (1.0 - discount) / (1.0 - self.gamma)
        weights = self.gft_sd_leaf_weights
        if weights:
            features = self.model.gft_tva_feature_dict(state)["features"]
            max_pushes = max(self.model.level.n_boxes * len(ACTION_NAMES), 1)
            n_boxes = max(self.model.level.n_boxes, 1)
            base += (
                float(weights.get("mobility", 0.0)) * float(features["M"]) / max_pushes
                + float(weights.get("access", 0.0)) * float(features["A"]) / max_pushes
                + float(weights.get("capacity", 0.0)) * float(features["K"]) / n_boxes
                - float(weights.get("deadlock", 0.0)) * float(features["D"]) / n_boxes
                - float(weights.get("ordering", 0.0)) * float(features["Q"]) / n_boxes
                + float(weights.get("commitment", 0.0)) * float(features["C"])
            )
        return float(base)

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        if self.model.is_solved(state):
            return self.goal_value
        if self.is_blocked(state):
            return self.deadend_value
        key = self.state_key(state, commitment_state)
        cached = self._leaf_cache.get(key)
        if cached is not None:
            return cached
        value = self.gft_sd_leaf_value(state)
        room = self.model.room_state_summary(state)["totals"]
        commitment = self.model.room_commitment_summary(state, commitment_state)
        value += self.room_leaf_commitment_bonus * float(commitment["committed_box_count"])
        value -= self.room_leaf_uncommitted_penalty * float(commitment["uncommitted_room_or_entry_count"])
        value -= self.room_leaf_order_penalty * float(room["q_settled"])
        value -= self.room_leaf_blocked_penalty * float(room["k_blocked"])
        value -= self.room_leaf_nonfrontier_commitment_penalty * float(
            commitment["nonfrontier_commitment_count"]
        )
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        value = float(max(low, min(high, value)))
        self._leaf_cache[key] = value
        return value

    def initialise_node(self, node: RoomSemanticNode) -> None:
        if node.untried is not None:
            return
        candidates, pruned = self.candidate_actions(node.state, node.commitment_state)
        candidates.sort(key=lambda candidate: (candidate.prior, candidate.prior_score))
        node.untried = candidates
        node.pruned_on_init = pruned
        self.deadlock_prunes += pruned

    def select_child(self, node: RoomSemanticNode) -> RoomSemanticEdge:
        parent_visits = max(node.visits, 1)

        def score(edge: RoomSemanticEdge) -> float:
            bonus = self.uct_c * edge.prior * math.sqrt(parent_visits) / (1.0 + edge.visits)
            return edge.mean_value + bonus

        return max(node.children.values(), key=score)

    def simulate(self, node: RoomSemanticNode, depth_remaining: int) -> float:
        if self.model.is_solved(node.state):
            value = self.goal_value
        elif self.is_blocked(node.state):
            value = self.deadend_value
        elif depth_remaining <= 0:
            value = self.leaf_value(node.state, node.commitment_state)
        else:
            self.initialise_node(node)
            if not node.untried and not node.children:
                value = self.deadend_value
            elif node.untried:
                candidate = node.untried.pop()
                child_node = self.node_for(
                    candidate.child_state,
                    candidate.child_commitment_state,
                    node.depth_from_root + 1,
                )
                child_value = self.leaf_value(candidate.child_state, candidate.child_commitment_state)
                value = -self.transition_cost(candidate.child_state) + self.gamma * child_value
                edge = RoomSemanticEdge(
                    action=candidate.action,
                    child_state=candidate.child_state,
                    child_commitment_state=candidate.child_commitment_state,
                    child_node=child_node,
                    prior=candidate.prior,
                    prior_score=candidate.prior_score,
                    room_prior_score=candidate.room_prior_score,
                    room_slot_progress_delta=candidate.room_slot_progress_delta,
                    room_slot_transport_delta=candidate.room_slot_transport_delta,
                    room_slot_committed_transport_delta=candidate.room_slot_committed_transport_delta,
                    room_slot_frontier_transport_delta=candidate.room_slot_frontier_transport_delta,
                    room_slot_barrier_delta=candidate.room_slot_barrier_delta,
                    room_slot_progress_summary=candidate.room_slot_progress_summary,
                    room_loop_penalty_score=candidate.room_loop_penalty_score,
                    room_loop_summary=candidate.room_loop_summary,
                    room_fill_plan_score=candidate.room_fill_plan_score,
                    room_fill_plan_summary=candidate.room_fill_plan_summary,
                    room_release_plan_score=candidate.room_release_plan_score,
                    room_release_plan_summary=candidate.room_release_plan_summary,
                    room_access_plan_score=candidate.room_access_plan_score,
                    room_access_plan_summary=candidate.room_access_plan_summary,
                    room_schedule_plan_score=candidate.room_schedule_plan_score,
                    room_schedule_plan_summary=candidate.room_schedule_plan_summary,
                    room_schedule_stall_score=candidate.room_schedule_stall_score,
                    room_schedule_stall_summary=candidate.room_schedule_stall_summary,
                    room_access_recovery_score=candidate.room_access_recovery_score,
                    room_access_recovery_summary=candidate.room_access_recovery_summary,
                    room_phase_progress_score=candidate.room_phase_progress_score,
                    room_phase_progress_summary=candidate.room_phase_progress_summary,
                    room_recoverability_score=candidate.room_recoverability_score,
                    room_recoverability_summary=candidate.room_recoverability_summary,
                    room_stand_lock_score=candidate.room_stand_lock_score,
                    room_stand_lock_summary=candidate.room_stand_lock_summary,
                    room_stage_score=candidate.room_stage_score,
                    room_stage_summary=candidate.room_stage_summary,
                    room_pre_release_trap_score=candidate.room_pre_release_trap_score,
                    room_pre_release_trap_summary=candidate.room_pre_release_trap_summary,
                    room_sd_generalization_score=candidate.room_sd_generalization_score,
                    room_sd_generalization_summary=candidate.room_sd_generalization_summary,
                    room_sd_value_score=candidate.room_sd_value_score,
                    room_sd_value_summary=candidate.room_sd_value_summary,
                    room_trace_anchor_score=candidate.room_trace_anchor_score,
                    room_trace_anchor_summary=candidate.room_trace_anchor_summary,
                )
                edge.visits = 1
                edge.value_sum = value
                node.children[candidate.action.key] = edge
            else:
                edge = self.select_child(node)
                child_value = self.simulate(edge.child_node, depth_remaining - 1)
                value = -self.transition_cost(edge.child_state) + self.gamma * child_value
                edge.visits += 1
                edge.value_sum += value

        node.visits += 1
        node.value_sum += value
        return value

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        if self.select_by == "q":
            return max(root.children.values(), key=lambda edge: (edge.mean_value, edge.visits))
        return max(root.children.values(), key=lambda edge: (edge.visits, edge.mean_value))

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(state, commitment_state)
        root = self.node_for(state, commitment_state, 0)
        prunes_before = self.deadlock_prunes
        for _ in range(max(self.budget, 1)):
            self.simulate(root, self.horizon)
        plan_prunes = self.deadlock_prunes - prunes_before
        if not root.children:
            return {
                "semantic_action": None,
                "physical_action": None,
                "next_commitment_state": commitment_state,
                "root_visits": root.visits,
                "root_value": root.value_sum / root.visits if root.visits else 0.0,
                "semantic_child_count": 0,
                "physical_child_count": 0,
                "deadlock_prunes": plan_prunes,
                "children": [],
            }
        best = self.choose_root_edge(root)
        children = self.root_children_summary(root)
        return {
            "semantic_action": best.action,
            "physical_action": best.action.physical_action,
            "next_commitment_state": best.child_commitment_state,
            "root_visits": root.visits,
            "root_value": root.value_sum / root.visits if root.visits else 0.0,
            "semantic_child_count": len(root.children),
            "physical_child_count": len({edge.action.physical_action.flat_index for edge in root.children.values()}),
            "deadlock_prunes": plan_prunes,
            "children": children,
        }

    def root_children_summary(self, root: RoomSemanticNode) -> list[dict[str, Any]]:
        rows = []
        for edge in root.children.values():
            rows.append(
                {
                    "semantic_action": edge.action.to_dict(),
                    "physical_action": edge.action.to_dict()["physical_action"],
                    "commitment_state": edge.child_commitment_state.to_dict(),
                    "visits": int(edge.visits),
                    "mean_value": float(edge.mean_value),
                    "prior": float(edge.prior),
                    "prior_score": float(edge.prior_score),
                    "room_prior_score": float(edge.room_prior_score),
                    "room_slot_progress_delta": float(edge.room_slot_progress_delta),
                    "room_slot_transport_delta": float(edge.room_slot_transport_delta),
                    "room_slot_committed_transport_delta": float(edge.room_slot_committed_transport_delta),
                    "room_slot_frontier_transport_delta": float(edge.room_slot_frontier_transport_delta),
                    "room_slot_barrier_delta": float(edge.room_slot_barrier_delta),
                    "room_slot_progress": edge.room_slot_progress_summary,
                    "room_loop_penalty_score": float(edge.room_loop_penalty_score),
                    "room_loop": edge.room_loop_summary,
                    "room_fill_plan_score": float(edge.room_fill_plan_score),
                    "room_fill_plan": edge.room_fill_plan_summary,
                    "room_release_plan_score": float(edge.room_release_plan_score),
                    "room_release_plan": edge.room_release_plan_summary,
                    "room_access_plan_score": float(edge.room_access_plan_score),
                    "room_access_plan": edge.room_access_plan_summary,
                    "room_schedule_plan_score": float(edge.room_schedule_plan_score),
                    "room_schedule_plan": edge.room_schedule_plan_summary,
                    "room_schedule_stall_score": float(edge.room_schedule_stall_score),
                    "room_schedule_stall": edge.room_schedule_stall_summary,
                    "room_access_recovery_score": float(edge.room_access_recovery_score),
                    "room_access_recovery": edge.room_access_recovery_summary,
                    "room_phase_progress_score": float(edge.room_phase_progress_score),
                    "room_phase_progress": edge.room_phase_progress_summary,
                    "room_recoverability_score": float(edge.room_recoverability_score),
                    "room_recoverability": edge.room_recoverability_summary,
                    "room_stand_lock_score": float(edge.room_stand_lock_score),
                    "room_stand_lock": edge.room_stand_lock_summary,
                    "room_stage_score": float(edge.room_stage_score),
                    "room_stage": edge.room_stage_summary,
                    "room_pre_release_trap_score": float(edge.room_pre_release_trap_score),
                    "room_pre_release_trap": edge.room_pre_release_trap_summary,
                    "room_sd_generalization_score": float(edge.room_sd_generalization_score),
                    "room_sd_generalization": edge.room_sd_generalization_summary,
                    "room_sd_value_score": float(edge.room_sd_value_score),
                    "room_sd_value": edge.room_sd_value_summary,
                    "room_trace_anchor_score": float(edge.room_trace_anchor_score),
                    "room_trace_anchor": edge.room_trace_anchor_summary,
                    "solved_child": bool(self.model.is_solved(edge.child_state)),
                    "blocked_child": bool(self.is_blocked(edge.child_state)),
                }
            )
        rows.sort(key=lambda row: (row["visits"], row["mean_value"]), reverse=True)
        return rows


class RoomSlotProgressGFTSDMCTSPlanner(RoomSemanticGFTSDMCTSPlanner):
    """A4 planner with slot-schedule constrained progress terms.

    The class inherits the A3 semantic child machinery and only changes the
    hard-instance scoring layer.  It is not used by the Boxoban mainline.
    """

    provider_id = "room_slot_progress_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        room_slot_transport_progress_weight: float = 0.55,
        room_slot_committed_progress_weight: float = 1.25,
        room_slot_frontier_progress_weight: float = 0.35,
        room_slot_barrier_reduction_weight: float = 0.80,
        room_slot_committed_pass_through_bonus: float = 8.0,
        room_slot_infeasible_commitment_penalty: float = 8.0,
        room_leaf_slot_transport_cost_weight: float = 0.03,
        room_leaf_slot_barrier_cost_weight: float = 0.08,
        room_leaf_slot_infeasible_penalty: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.room_slot_transport_progress_weight = float(room_slot_transport_progress_weight)
        self.room_slot_committed_progress_weight = float(room_slot_committed_progress_weight)
        self.room_slot_frontier_progress_weight = float(room_slot_frontier_progress_weight)
        self.room_slot_barrier_reduction_weight = float(room_slot_barrier_reduction_weight)
        self.room_slot_committed_pass_through_bonus = float(room_slot_committed_pass_through_bonus)
        self.room_slot_infeasible_commitment_penalty = float(room_slot_infeasible_commitment_penalty)
        self.room_leaf_slot_transport_cost_weight = float(room_leaf_slot_transport_cost_weight)
        self.room_leaf_slot_barrier_cost_weight = float(room_leaf_slot_barrier_cost_weight)
        self.room_leaf_slot_infeasible_penalty = float(room_leaf_slot_infeasible_penalty)

    def _slot_progress_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.model.room_slot_progress_summary(state, commitment_state)
        after = self.model.room_slot_progress_summary(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        total_delta = float(before["slot_schedule_cost"] - after["slot_schedule_cost"])
        transport_delta = float(before["total_transport_cost"] - after["total_transport_cost"])
        committed_delta = float(before["committed_transport_cost"] - after["committed_transport_cost"])
        frontier_delta = float(
            before["uncommitted_frontier_transport_cost"]
            - after["uncommitted_frontier_transport_cost"]
        )
        barrier_delta = float(before["barrier_cost"] - after["barrier_cost"])
        before_infeasible = int(before["barrier_counts"]["committed_infeasible"])
        after_infeasible = int(after["barrier_counts"]["committed_infeasible"])
        infeasible_delta = max(0, after_infeasible - before_infeasible)
        pass_through_delta = max(
            0,
            int(after["barrier_counts"]["committed_pass_through_blocker"])
            - int(before["barrier_counts"]["committed_pass_through_blocker"]),
        )
        score = (
            self.room_slot_transport_progress_weight * transport_delta
            + self.room_slot_committed_progress_weight * committed_delta
            + self.room_slot_frontier_progress_weight * frontier_delta
            + self.room_slot_barrier_reduction_weight * barrier_delta
            + self.room_slot_committed_pass_through_bonus * float(pass_through_delta)
            - self.room_slot_infeasible_commitment_penalty * float(infeasible_delta)
        )
        candidate.room_slot_progress_delta = total_delta
        candidate.room_slot_transport_delta = transport_delta
        candidate.room_slot_committed_transport_delta = committed_delta
        candidate.room_slot_frontier_transport_delta = frontier_delta
        candidate.room_slot_barrier_delta = barrier_delta
        candidate.room_slot_progress_summary = {
            "before": {
                "slot_schedule_cost": float(before["slot_schedule_cost"]),
                "total_transport_cost": float(before["total_transport_cost"]),
                "committed_transport_cost": float(before["committed_transport_cost"]),
                "uncommitted_frontier_transport_cost": float(before["uncommitted_frontier_transport_cost"]),
                "barrier_cost": float(before["barrier_cost"]),
                "barrier_counts": before["barrier_counts"],
            },
            "after": {
                "slot_schedule_cost": float(after["slot_schedule_cost"]),
                "total_transport_cost": float(after["total_transport_cost"]),
                "committed_transport_cost": float(after["committed_transport_cost"]),
                "uncommitted_frontier_transport_cost": float(after["uncommitted_frontier_transport_cost"]),
                "barrier_cost": float(after["barrier_cost"]),
                "barrier_counts": after["barrier_counts"],
            },
            "delta": {
                "slot_schedule_cost": total_delta,
                "total_transport_cost": transport_delta,
                "committed_transport_cost": committed_delta,
                "uncommitted_frontier_transport_cost": frontier_delta,
                "barrier_cost": barrier_delta,
                "committed_infeasible_increase": int(infeasible_delta),
                "committed_pass_through_increase": int(pass_through_delta),
            },
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._slot_progress_score(state, commitment_state, candidate))

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        if self.model.is_solved(state):
            return self.goal_value
        if self.is_blocked(state):
            return self.deadend_value
        key = self.state_key(state, commitment_state)
        cached = self._leaf_cache.get(key)
        if cached is not None:
            return cached
        value = self.gft_sd_leaf_value(state)
        room = self.model.room_state_summary(state)["totals"]
        commitment = self.model.room_commitment_summary(state, commitment_state)
        slot_progress = self.model.room_slot_progress_summary(state, commitment_state)
        barrier_counts = slot_progress["barrier_counts"]
        q_settled = int(barrier_counts.get("q_settled_effective", room["q_settled"]))
        k_blocked = int(barrier_counts.get("k_blocked_effective", room["k_blocked"]))
        value += self.room_leaf_commitment_bonus * float(commitment["committed_box_count"])
        value -= self.room_leaf_uncommitted_penalty * float(commitment["uncommitted_room_or_entry_count"])
        value -= self.room_leaf_order_penalty * float(q_settled)
        value -= self.room_leaf_blocked_penalty * float(k_blocked)
        value -= self.room_leaf_nonfrontier_commitment_penalty * float(
            commitment["nonfrontier_commitment_count"]
        )
        value -= self.room_leaf_slot_transport_cost_weight * float(slot_progress["total_transport_cost"])
        value -= self.room_leaf_slot_barrier_cost_weight * float(slot_progress["barrier_cost"])
        value -= self.room_leaf_slot_infeasible_penalty * float(
            slot_progress["barrier_counts"]["committed_infeasible"]
        )
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        value = float(max(low, min(high, value)))
        self._leaf_cache[key] = value
        return value


class RoomLoopGuardGFTSDMCTSPlanner(RoomSlotProgressGFTSDMCTSPlanner):
    """A5 planner with commitment-monotonic loop guards.

    The planner inherits A4 slot-schedule progress and adds diagnostic penalties
    for committed-box backtracking.  The guards are soft prior/leaf terms; they
    do not prune legal Sokoban transitions.
    """

    provider_id = "room_loop_guard_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        previous_physical_action: SokobanPushAction | None = None,
        previous_moved_source: Cell | None = None,
        previous_moved_target: Cell | None = None,
        recent_commitment_keys: Iterable[tuple] | None = None,
        room_loop_distance_regress_penalty: float = 8.0,
        room_loop_direct_reverse_penalty: float = 6.0,
        room_loop_repeated_commitment_penalty: float = 5.0,
        room_leaf_loop_repeated_commitment_penalty: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.previous_physical_action = previous_physical_action
        self.previous_moved_source = previous_moved_source
        self.previous_moved_target = previous_moved_target
        self.recent_commitment_keys = set(recent_commitment_keys or ())
        self.room_loop_distance_regress_penalty = float(room_loop_distance_regress_penalty)
        self.room_loop_direct_reverse_penalty = float(room_loop_direct_reverse_penalty)
        self.room_loop_repeated_commitment_penalty = float(room_loop_repeated_commitment_penalty)
        self.room_leaf_loop_repeated_commitment_penalty = float(room_leaf_loop_repeated_commitment_penalty)

    def _moved_cells(self, state: SokobanState, action: SokobanPushAction) -> tuple[Cell, Cell]:
        source = state.boxes[action.box_index]
        target = add_cells(source, DIRECTIONS[action.direction])
        return source, target

    def _committed_distance_for_box(
        self,
        commitment_state: RoomCommitmentState,
        box_cell: Cell,
    ) -> float | None:
        commitment = commitment_state.commitment_for_box(box_cell)
        if commitment is None:
            return None
        distance = self.model._slot_pull_distance(box_cell, commitment.slot_cell)
        return float(distance) if math.isfinite(distance) else math.inf

    def _room_loop_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        action = candidate.action.physical_action
        source, target = self._moved_cells(state, action)
        before_distance = self._committed_distance_for_box(commitment_state, source)
        after_distance = self._committed_distance_for_box(candidate.child_commitment_state, target)
        distance_regress = 0.0
        if before_distance is not None and after_distance is not None:
            if math.isfinite(before_distance) and math.isfinite(after_distance):
                distance_regress = max(0.0, after_distance - before_distance)
            elif math.isfinite(before_distance) and not math.isfinite(after_distance):
                distance_regress = 1.0

        direct_reverse = bool(
            self.previous_moved_source is not None
            and self.previous_moved_target is not None
            and source == self.previous_moved_target
            and target == self.previous_moved_source
        )
        child_key = self.state_key(candidate.child_state, candidate.child_commitment_state)
        repeated_commitment_state = child_key in self.recent_commitment_keys
        score = -(
            self.room_loop_distance_regress_penalty * distance_regress
            + self.room_loop_direct_reverse_penalty * float(direct_reverse)
            + self.room_loop_repeated_commitment_penalty * float(repeated_commitment_state)
        )
        candidate.room_loop_penalty_score = float(score)
        candidate.room_loop_summary = {
            "moved_source": list(source),
            "moved_target": list(target),
            "before_committed_distance": None if before_distance is None else float(before_distance),
            "after_committed_distance": None if after_distance is None else float(after_distance),
            "distance_regress": float(distance_regress),
            "direct_reverse_previous_push": bool(direct_reverse),
            "repeated_commitment_state": bool(repeated_commitment_state),
            "recent_commitment_key_count": int(len(self.recent_commitment_keys)),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._room_loop_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if best.room_loop_penalty_score >= 0.0:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if edge.room_loop_penalty_score >= 0.0
        ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(alternatives, key=lambda edge: (edge.mean_value, edge.visits, edge.prior_score))
        return max(alternatives, key=lambda edge: (edge.visits, edge.mean_value, edge.prior_score))

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        if self.room_leaf_loop_repeated_commitment_penalty > 0.0:
            key = self.state_key(state, commitment_state)
            if key in self.recent_commitment_keys:
                value -= self.room_leaf_loop_repeated_commitment_penalty
                low = min(self.deadend_value, self.goal_value)
                high = max(self.deadend_value, self.goal_value)
                value = float(max(low, min(high, value)))
        return value


class RoomCorridorMacroGFTSDMCTSPlanner(RoomLoopGuardGFTSDMCTSPlanner):
    """A6 planner with a conservative forced-corridor macro.

    The search tree remains the A5 single-push tree.  After a root action is
    selected, the planner may contract a deterministic corridor segment for the
    same committed box when there is exactly one legal monotone push toward the
    committed slot.  This keeps the mechanism isolated from the Boxoban
    mainline and from A3-A5 diagnostics.
    """

    provider_id = "room_corridor_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        corridor_macro_enabled: bool = True,
        corridor_macro_max_extra_pushes: int = 8,
        corridor_macro_min_total_pushes: int = 2,
        corridor_macro_require_unique_monotone: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.corridor_macro_enabled = bool(corridor_macro_enabled)
        self.corridor_macro_max_extra_pushes = max(int(corridor_macro_max_extra_pushes), 0)
        self.corridor_macro_min_total_pushes = max(int(corridor_macro_min_total_pushes), 1)
        self.corridor_macro_require_unique_monotone = bool(corridor_macro_require_unique_monotone)

    def _physical_action_dict(self, state: SokobanState, action: SokobanPushAction) -> dict[str, Any]:
        source, target = self._moved_cells(state, action)
        return {
            "box_index": int(action.box_index),
            "direction": int(action.direction),
            "direction_name": ACTION_NAMES[action.direction],
            "flat_index": int(action.flat_index),
            "source": list(source),
            "target": list(target),
        }

    def _slot_option_for_commitment(self, commitment) -> RoomSlotOption:
        slot = self.model.room_slots_by_cell.get(commitment.slot_cell)
        return RoomSlotOption(
            kind="slot",
            room_id=int(commitment.room_id),
            slot_id=int(commitment.slot_id),
            slot_cell=commitment.slot_cell,
            entry_distance=None if slot is None else int(slot.entry_distance),
        )

    def _monotone_committed_pushes(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        commitment,
    ) -> list[dict[str, Any]]:
        before_distance = self.model._slot_pull_distance(commitment.box_cell, commitment.slot_cell)
        if not math.isfinite(before_distance):
            return []
        option = self._slot_option_for_commitment(commitment)
        rows: list[dict[str, Any]] = []
        for action in self.model.legal_flows(state):
            source, target = self._moved_cells(state, action)
            if source != commitment.box_cell:
                continue
            child = self.model.step_grid_flow(state, action)
            if self.is_blocked(child):
                continue
            child_commitment = self.model.advance_room_commitment_state(
                state,
                commitment_state,
                action,
                option,
                child,
            )
            moved_commitment = child_commitment.commitment_for_box(target)
            if moved_commitment is None:
                continue
            after_distance = self.model._slot_pull_distance(target, moved_commitment.slot_cell)
            if not math.isfinite(after_distance):
                continue
            distance_delta = float(before_distance - after_distance)
            if distance_delta <= 0.0:
                continue
            rows.append(
                {
                    "action": action,
                    "child_state": child,
                    "child_commitment_state": child_commitment,
                    "before_distance": float(before_distance),
                    "after_distance": float(after_distance),
                    "distance_delta": float(distance_delta),
                    "option": option,
                }
            )
        rows.sort(
            key=lambda row: (
                float(row["distance_delta"]),
                -int(row["action"].flat_index),
            ),
            reverse=True,
        )
        return rows

    def corridor_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.corridor_macro_enabled or plan.get("physical_action") is None:
            return {
                "enabled": bool(self.corridor_macro_enabled),
                "triggered": False,
                "reason": "disabled_or_no_action",
                "physical_actions": [],
                "after_commitment_states": [],
                "rows": [],
            }

        first_action = plan["physical_action"]
        first_child = self.model.step_grid_flow(state, first_action)
        first_commitment_state = self.model.normalize_room_commitment_state(
            first_child,
            plan["next_commitment_state"],
        )
        first_source, first_target = self._moved_cells(state, first_action)
        first_commitment = first_commitment_state.commitment_for_box(first_target)
        if first_commitment is None:
            return {
                "enabled": True,
                "triggered": False,
                "reason": "selected_box_uncommitted",
                "physical_actions": [first_action],
                "after_commitment_states": [first_commitment_state],
                "rows": [
                    {
                        "macro_index": 0,
                        "physical_action": self._physical_action_dict(state, first_action),
                        "reason": "initial_selected_push",
                    }
                ],
            }

        sequence = [first_action]
        commitment_states = [first_commitment_state]
        rows = [
            {
                "macro_index": 0,
                "physical_action": self._physical_action_dict(state, first_action),
                "before_distance": self._committed_distance_for_box(commitment_state, first_source),
                "after_distance": self._committed_distance_for_box(first_commitment_state, first_target),
                "distance_delta": None,
                "reason": "initial_selected_push",
            }
        ]
        current_state = first_child
        current_commitment_state = first_commitment_state
        current_commitment = first_commitment
        stop_reason = "max_extra_pushes"

        for macro_index in range(1, self.corridor_macro_max_extra_pushes + 1):
            if current_commitment.box_cell == current_commitment.slot_cell:
                stop_reason = "committed_slot_reached"
                break
            monotone = self._monotone_committed_pushes(
                current_state,
                current_commitment_state,
                current_commitment,
            )
            if not monotone:
                stop_reason = "no_monotone_committed_push"
                break
            if self.corridor_macro_require_unique_monotone and len(monotone) != 1:
                stop_reason = "ambiguous_monotone_committed_push"
                break
            chosen = monotone[0]
            action = chosen["action"]
            before_push_state = current_state
            source, target = self._moved_cells(current_state, action)
            sequence.append(action)
            current_state = chosen["child_state"]
            current_commitment_state = chosen["child_commitment_state"]
            commitment_states.append(current_commitment_state)
            rows.append(
                {
                    "macro_index": int(macro_index),
                    "physical_action": self._physical_action_dict(before_push_state, action),
                    "before_distance": float(chosen["before_distance"]),
                    "after_distance": float(chosen["after_distance"]),
                    "distance_delta": float(chosen["distance_delta"]),
                    "reason": "unique_monotone_committed_push",
                    "candidate_count": int(len(monotone)),
                }
            )
            moved = current_commitment_state.commitment_for_box(target)
            if moved is None:
                stop_reason = "commitment_lost"
                break
            current_commitment = moved
            if current_commitment.box_cell == current_commitment.slot_cell:
                stop_reason = "committed_slot_reached"
                break
        else:
            stop_reason = "max_extra_pushes"

        triggered = len(sequence) >= self.corridor_macro_min_total_pushes
        return {
            "enabled": True,
            "triggered": bool(triggered),
            "reason": stop_reason if triggered else "below_min_total_pushes",
            "physical_actions": sequence,
            "after_commitment_states": commitment_states,
            "rows": rows,
            "macro_push_count": int(len(sequence)),
            "extra_push_count": int(max(0, len(sequence) - 1)),
        }

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(state, commitment_state)
        plan = self.plan(state, commitment_state)
        plan["corridor_macro"] = self.corridor_macro_from_plan(state, commitment_state, plan)
        return plan


class RoomFillPlanGFTSDMCTSPlanner(RoomCorridorMacroGFTSDMCTSPlanner):
    """A8 planner with teacher-trace-distilled fill-plan state.

    Unlike A7, this planner does not receive a reference action table.  It
    consumes state-derived fill-plan summaries from
    :class:`SokobanRoomFillPlanGFTSDModel`: active slot, channel/staging counts,
    and plan barrier costs.
    """

    provider_id = "room_fill_plan_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        room_fill_plan_barrier_reduction_weight: float = 1.20,
        room_fill_plan_active_commitment_bonus: float = 5.0,
        room_fill_plan_channel_staging_bonus: float = 3.0,
        room_fill_plan_early_channel_penalty: float = 10.0,
        room_fill_plan_channel_final_penalty: float = 16.0,
        room_fill_plan_active_transport_weight: float = 0.20,
        room_fill_plan_active_release_weight: float = 2.0,
        room_fill_plan_active_monotone_weight: float = 4.0,
        room_fill_plan_active_blocked_penalty: float = 6.0,
        room_leaf_fill_plan_barrier_weight: float = 0.15,
        room_leaf_active_transport_weight: float = 0.03,
        room_leaf_active_blocked_penalty: float = 1.5,
        room_leaf_early_channel_penalty: float = 2.0,
        room_leaf_channel_staging_bonus: float = 0.8,
        room_fill_plan_root_guard: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.room_fill_plan_barrier_reduction_weight = float(room_fill_plan_barrier_reduction_weight)
        self.room_fill_plan_active_commitment_bonus = float(room_fill_plan_active_commitment_bonus)
        self.room_fill_plan_channel_staging_bonus = float(room_fill_plan_channel_staging_bonus)
        self.room_fill_plan_early_channel_penalty = float(room_fill_plan_early_channel_penalty)
        self.room_fill_plan_channel_final_penalty = float(room_fill_plan_channel_final_penalty)
        self.room_fill_plan_active_transport_weight = float(room_fill_plan_active_transport_weight)
        self.room_fill_plan_active_release_weight = float(room_fill_plan_active_release_weight)
        self.room_fill_plan_active_monotone_weight = float(room_fill_plan_active_monotone_weight)
        self.room_fill_plan_active_blocked_penalty = float(room_fill_plan_active_blocked_penalty)
        self.room_leaf_fill_plan_barrier_weight = float(room_leaf_fill_plan_barrier_weight)
        self.room_leaf_active_transport_weight = float(room_leaf_active_transport_weight)
        self.room_leaf_active_blocked_penalty = float(room_leaf_active_blocked_penalty)
        self.room_leaf_early_channel_penalty = float(room_leaf_early_channel_penalty)
        self.room_leaf_channel_staging_bonus = float(room_leaf_channel_staging_bonus)
        self.room_fill_plan_root_guard = bool(room_fill_plan_root_guard)

    @property
    def fill_plan_model(self) -> SokobanRoomFillPlanGFTSDModel:
        if not isinstance(self.model, SokobanRoomFillPlanGFTSDModel):
            raise TypeError("RoomFillPlanGFTSDMCTSPlanner requires SokobanRoomFillPlanGFTSDModel.")
        return self.model

    def cached_fill_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        key = self._summary_cache_key("fill", state, commitment_state)
        cached = self._summary_cache.get(key)
        if cached is None:
            cached = self.fill_plan_model.room_fill_plan_summary(state, commitment_state)
            self._summary_cache[key] = cached
        return cached

    def _fill_plan_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_fill_plan_summary(state, commitment_state)
        after = self.cached_fill_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        barrier_delta = float(before_totals["plan_barrier_cost"] - after_totals["plan_barrier_cost"])
        active_commitment_delta = int(after_totals["active_commitment_count"]) - int(
            before_totals["active_commitment_count"]
        )
        channel_staging_delta = int(after_totals["channel_staging_box_count"]) - int(
            before_totals["channel_staging_box_count"]
        )
        early_channel_delta = int(after_totals["early_channel_occupancy"]) - int(
            before_totals["early_channel_occupancy"]
        )
        channel_final_delta = int(after_totals["early_channel_final_commitment"]) - int(
            before_totals["early_channel_final_commitment"]
        )
        active_transport_delta = float(
            before_totals["active_transport_cost"] - after_totals["active_transport_cost"]
        )
        active_push_delta = int(after_totals["active_box_push_count"]) - int(
            before_totals["active_box_push_count"]
        )
        active_monotone_delta = int(after_totals["active_box_monotone_push_count"]) - int(
            before_totals["active_box_monotone_push_count"]
        )
        active_blocked_delta = int(after_totals["active_box_blocked_count"]) - int(
            before_totals["active_box_blocked_count"]
        )
        score = (
            self.room_fill_plan_barrier_reduction_weight * barrier_delta
            + self.room_fill_plan_active_commitment_bonus * float(max(0, active_commitment_delta))
            + self.room_fill_plan_channel_staging_bonus * float(max(0, channel_staging_delta))
            - self.room_fill_plan_early_channel_penalty * float(max(0, early_channel_delta))
            - self.room_fill_plan_channel_final_penalty * float(max(0, channel_final_delta))
            + self.room_fill_plan_active_transport_weight * active_transport_delta
            + self.room_fill_plan_active_release_weight * float(active_push_delta)
            + self.room_fill_plan_active_monotone_weight * float(active_monotone_delta)
            - self.room_fill_plan_active_blocked_penalty * float(max(0, active_blocked_delta))
        )
        option = candidate.action.option
        option_rank = None
        option_channel_lane = None
        if option.kind == "slot" and option.slot_cell is not None:
            rank = self.fill_plan_model.room_fill_plan_slot_rank(option.slot_cell)
            if rank is not None:
                option_rank, option_channel_lane = rank
        candidate.room_fill_plan_score = float(score)
        candidate.room_fill_plan_summary = {
            "before": {
                "totals": before_totals,
                "rooms": before["rooms"],
            },
            "after": {
                "totals": after_totals,
                "rooms": after["rooms"],
            },
            "delta": {
                "plan_barrier_cost": barrier_delta,
                "active_commitment_count": int(active_commitment_delta),
                "channel_staging_box_count": int(channel_staging_delta),
                "early_channel_occupancy": int(early_channel_delta),
                "early_channel_final_commitment": int(channel_final_delta),
                "active_transport_cost": active_transport_delta,
                "active_box_push_count": int(active_push_delta),
                "active_box_monotone_push_count": int(active_monotone_delta),
                "active_box_blocked_count": int(active_blocked_delta),
            },
            "option_slot_queue_index": option_rank,
            "option_slot_channel_lane": option_channel_lane,
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._fill_plan_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.room_fill_plan_root_guard:
            return best
        best_delta = best.room_fill_plan_summary.get("delta", {})
        if (
            int(best_delta.get("early_channel_occupancy", 0)) <= 0
            and int(best_delta.get("early_channel_final_commitment", 0)) <= 0
        ):
            return best
        alternatives = []
        for edge in root.children.values():
            delta = edge.room_fill_plan_summary.get("delta", {})
            if int(delta.get("early_channel_occupancy", 0)) <= 0 and int(
                delta.get("early_channel_final_commitment", 0)
            ) <= 0:
                alternatives.append(edge)
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (edge.mean_value, edge.visits, edge.room_fill_plan_score, edge.prior_score),
            )
        return max(
            alternatives,
            key=lambda edge: (edge.visits, edge.mean_value, edge.room_fill_plan_score, edge.prior_score),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        fill = self.cached_fill_plan_summary(state, commitment_state)
        totals = fill["totals"]
        value -= self.room_leaf_fill_plan_barrier_weight * float(totals["plan_barrier_cost"])
        value -= self.room_leaf_active_transport_weight * float(totals["active_transport_cost"])
        value -= self.room_leaf_active_blocked_penalty * float(totals["active_box_blocked_count"])
        value -= self.room_leaf_early_channel_penalty * float(totals["early_channel_occupancy"])
        value += self.room_leaf_channel_staging_bonus * float(totals["channel_staging_box_count"])
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        return float(max(low, min(high, value)))


class RoomReleasePlanGFTSDMCTSPlanner(RoomFillPlanGFTSDMCTSPlanner):
    """A9 planner with active-box stand-cell release state.

    This planner keeps A8 fill-plan scoring and consumes the A9 release-plan
    summary.  It rewards moves that release the player stand cells required for
    monotone active-box pushes and penalizes moves that make those local access
    constraints worse.  No teacher actions are used online.
    """

    provider_id = "room_release_plan_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        room_release_barrier_reduction_weight: float = 1.8,
        room_release_need_reduction_weight: float = 4.0,
        room_release_stand_blocker_reduction_weight: float = 8.0,
        room_release_target_blocker_reduction_weight: float = 6.0,
        room_release_unreachable_stand_reduction_weight: float = 3.0,
        room_release_available_monotone_weight: float = 7.0,
        room_release_blocker_box_reduction_weight: float = 5.0,
        room_release_worsening_penalty: float = 3.0,
        room_leaf_release_barrier_weight: float = 0.18,
        room_leaf_release_need_penalty: float = 1.2,
        room_leaf_release_stand_blocker_penalty: float = 2.5,
        room_leaf_release_available_bonus: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.room_release_barrier_reduction_weight = float(room_release_barrier_reduction_weight)
        self.room_release_need_reduction_weight = float(room_release_need_reduction_weight)
        self.room_release_stand_blocker_reduction_weight = float(room_release_stand_blocker_reduction_weight)
        self.room_release_target_blocker_reduction_weight = float(room_release_target_blocker_reduction_weight)
        self.room_release_unreachable_stand_reduction_weight = float(room_release_unreachable_stand_reduction_weight)
        self.room_release_available_monotone_weight = float(room_release_available_monotone_weight)
        self.room_release_blocker_box_reduction_weight = float(room_release_blocker_box_reduction_weight)
        self.room_release_worsening_penalty = float(room_release_worsening_penalty)
        self.room_leaf_release_barrier_weight = float(room_leaf_release_barrier_weight)
        self.room_leaf_release_need_penalty = float(room_leaf_release_need_penalty)
        self.room_leaf_release_stand_blocker_penalty = float(room_leaf_release_stand_blocker_penalty)
        self.room_leaf_release_available_bonus = float(room_leaf_release_available_bonus)

    @property
    def release_plan_model(self) -> SokobanRoomReleasePlanGFTSDModel:
        if not isinstance(self.model, SokobanRoomReleasePlanGFTSDModel):
            raise TypeError("RoomReleasePlanGFTSDMCTSPlanner requires SokobanRoomReleasePlanGFTSDModel.")
        return self.model

    def cached_release_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        key = self._summary_cache_key("release", state, commitment_state)
        cached = self._summary_cache.get(key)
        if cached is None:
            cached = self.release_plan_model.room_release_plan_summary(state, commitment_state)
            self._summary_cache[key] = cached
        return cached

    def _release_plan_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_release_plan_summary(state, commitment_state)
        after = self.cached_release_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        barrier_delta = float(before_totals["release_barrier_cost"] - after_totals["release_barrier_cost"])
        need_delta = int(before_totals["active_release_need_count"]) - int(
            after_totals["active_release_need_count"]
        )
        stand_blocker_delta = int(before_totals["active_release_stand_blocker_count"]) - int(
            after_totals["active_release_stand_blocker_count"]
        )
        target_blocker_delta = int(before_totals["active_release_target_blocker_count"]) - int(
            after_totals["active_release_target_blocker_count"]
        )
        unreachable_delta = int(before_totals["active_release_unreachable_stand_count"]) - int(
            after_totals["active_release_unreachable_stand_count"]
        )
        available_delta = int(after_totals["active_release_monotone_available_count"]) - int(
            before_totals["active_release_monotone_available_count"]
        )
        blocker_box_delta = int(before_totals["release_blocker_box_count"]) - int(
            after_totals["release_blocker_box_count"]
        )
        worsening = (
            max(0, -need_delta)
            + max(0, -stand_blocker_delta)
            + max(0, -target_blocker_delta)
            + max(0, -unreachable_delta)
            + max(0, -blocker_box_delta)
        )
        score = (
            self.room_release_barrier_reduction_weight * barrier_delta
            + self.room_release_need_reduction_weight * float(need_delta)
            + self.room_release_stand_blocker_reduction_weight * float(stand_blocker_delta)
            + self.room_release_target_blocker_reduction_weight * float(target_blocker_delta)
            + self.room_release_unreachable_stand_reduction_weight * float(unreachable_delta)
            + self.room_release_available_monotone_weight * float(available_delta)
            + self.room_release_blocker_box_reduction_weight * float(blocker_box_delta)
            - self.room_release_worsening_penalty * float(worsening)
        )
        candidate.room_release_plan_score = float(score)
        candidate.room_release_plan_summary = {
            "before": {
                "totals": before_totals,
                "rooms": before["rooms"],
            },
            "after": {
                "totals": after_totals,
                "rooms": after["rooms"],
            },
            "delta": {
                "release_barrier_cost": float(barrier_delta),
                "active_release_need_count": int(need_delta),
                "active_release_stand_blocker_count": int(stand_blocker_delta),
                "active_release_target_blocker_count": int(target_blocker_delta),
                "active_release_unreachable_stand_count": int(unreachable_delta),
                "active_release_monotone_available_count": int(available_delta),
                "release_blocker_box_count": int(blocker_box_delta),
                "worsening_count": int(worsening),
            },
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._release_plan_score(state, commitment_state, candidate))

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        release = self.cached_release_plan_summary(state, commitment_state)
        totals = release["totals"]
        value -= self.room_leaf_release_barrier_weight * float(totals["release_barrier_cost"])
        value -= self.room_leaf_release_need_penalty * float(totals["active_release_need_count"])
        value -= self.room_leaf_release_stand_blocker_penalty * float(
            totals["active_release_stand_blocker_count"]
        )
        value += self.room_leaf_release_available_bonus * float(
            totals["active_release_monotone_available_count"]
        )
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        return float(max(low, min(high, value)))


class RoomAccessPlanGFTSDMCTSPlanner(RoomReleasePlanGFTSDMCTSPlanner):
    """A10 planner with lookahead access preservation.

    A10 keeps A9 release-plan scoring and adds a lookahead access-plan term for
    upcoming room slots.  The intent is to protect future player stand cells
    before they become the current active box's immediate release constraint.
    """

    provider_id = "room_access_plan_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        room_access_lookahead_slots: int = 2,
        room_access_barrier_reduction_weight: float = 0.0,
        room_access_future_barrier_reduction_weight: float = 0.0,
        room_access_need_reduction_weight: float = 0.0,
        room_access_future_need_reduction_weight: float = 0.0,
        room_access_stand_blocker_reduction_weight: float = 8.0,
        room_access_future_stand_blocker_reduction_weight: float = 14.0,
        room_access_available_monotone_weight: float = 0.0,
        room_access_worsening_penalty: float = 4.0,
        room_leaf_access_barrier_weight: float = 0.0,
        room_leaf_future_access_barrier_weight: float = 0.0,
        room_leaf_future_access_need_penalty: float = 0.0,
        room_leaf_future_stand_blocker_penalty: float = 0.5,
        room_access_root_guard: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.room_access_lookahead_slots = max(int(room_access_lookahead_slots), 1)
        self.room_access_barrier_reduction_weight = float(room_access_barrier_reduction_weight)
        self.room_access_future_barrier_reduction_weight = float(room_access_future_barrier_reduction_weight)
        self.room_access_need_reduction_weight = float(room_access_need_reduction_weight)
        self.room_access_future_need_reduction_weight = float(room_access_future_need_reduction_weight)
        self.room_access_stand_blocker_reduction_weight = float(room_access_stand_blocker_reduction_weight)
        self.room_access_future_stand_blocker_reduction_weight = float(
            room_access_future_stand_blocker_reduction_weight
        )
        self.room_access_available_monotone_weight = float(room_access_available_monotone_weight)
        self.room_access_worsening_penalty = float(room_access_worsening_penalty)
        self.room_leaf_access_barrier_weight = float(room_leaf_access_barrier_weight)
        self.room_leaf_future_access_barrier_weight = float(room_leaf_future_access_barrier_weight)
        self.room_leaf_future_access_need_penalty = float(room_leaf_future_access_need_penalty)
        self.room_leaf_future_stand_blocker_penalty = float(room_leaf_future_stand_blocker_penalty)
        self.room_access_root_guard = bool(room_access_root_guard)

    @property
    def access_plan_model(self) -> SokobanRoomAccessPlanGFTSDModel:
        if not isinstance(self.model, SokobanRoomAccessPlanGFTSDModel):
            raise TypeError("RoomAccessPlanGFTSDMCTSPlanner requires SokobanRoomAccessPlanGFTSDModel.")
        return self.model

    def cached_access_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        *,
        lookahead_slots: int,
    ) -> dict[str, Any]:
        key = self._summary_cache_key("access", state, commitment_state, int(lookahead_slots))
        cached = self._summary_cache.get(key)
        if cached is None:
            cached = self.access_plan_model.room_access_plan_summary(
                state,
                commitment_state,
                lookahead_slots=int(lookahead_slots),
            )
            self._summary_cache[key] = cached
        return cached

    def _access_plan_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_access_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_access_lookahead_slots,
        )
        after = self.cached_access_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
            lookahead_slots=self.room_access_lookahead_slots,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        barrier_delta = float(before_totals["access_barrier_cost"] - after_totals["access_barrier_cost"])
        future_barrier_delta = float(
            before_totals["access_future_barrier_cost"] - after_totals["access_future_barrier_cost"]
        )
        need_delta = int(before_totals["access_need_count"]) - int(after_totals["access_need_count"])
        future_need_delta = int(before_totals["access_future_need_count"]) - int(
            after_totals["access_future_need_count"]
        )
        stand_blocker_delta = int(before_totals["access_stand_blocker_count"]) - int(
            after_totals["access_stand_blocker_count"]
        )
        future_stand_blocker_delta = int(before_totals["access_future_stand_blocker_count"]) - int(
            after_totals["access_future_stand_blocker_count"]
        )
        available_delta = int(after_totals["access_monotone_available_count"]) - int(
            before_totals["access_monotone_available_count"]
        )
        worsening = (
            max(0, -need_delta)
            + max(0, -future_need_delta)
            + max(0, -stand_blocker_delta)
            + max(0, -future_stand_blocker_delta)
        )
        score = (
            self.room_access_barrier_reduction_weight * barrier_delta
            + self.room_access_future_barrier_reduction_weight * future_barrier_delta
            + self.room_access_need_reduction_weight * float(need_delta)
            + self.room_access_future_need_reduction_weight * float(future_need_delta)
            + self.room_access_stand_blocker_reduction_weight * float(stand_blocker_delta)
            + self.room_access_future_stand_blocker_reduction_weight * float(future_stand_blocker_delta)
            + self.room_access_available_monotone_weight * float(available_delta)
            - self.room_access_worsening_penalty * float(worsening)
        )
        candidate.room_access_plan_score = float(score)
        candidate.room_access_plan_summary = {
            "before": {
                "totals": before_totals,
                "rooms": before["rooms"],
            },
            "after": {
                "totals": after_totals,
                "rooms": after["rooms"],
            },
            "delta": {
                "access_barrier_cost": float(barrier_delta),
                "access_future_barrier_cost": float(future_barrier_delta),
                "access_need_count": int(need_delta),
                "access_future_need_count": int(future_need_delta),
                "access_stand_blocker_count": int(stand_blocker_delta),
                "access_future_stand_blocker_count": int(future_stand_blocker_delta),
                "access_monotone_available_count": int(available_delta),
                "worsening_count": int(worsening),
            },
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._access_plan_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.room_access_root_guard:
            return best
        best_delta = best.room_access_plan_summary.get("delta", {})
        best_worsens = (
            int(best_delta.get("access_future_stand_blocker_count", 0)) < 0
        )
        if not best_worsens:
            return best
        alternatives = []
        for edge in root.children.values():
            delta = edge.room_access_plan_summary.get("delta", {})
            if (
                int(delta.get("access_future_stand_blocker_count", 0)) >= 0
            ):
                alternatives.append(edge)
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (edge.mean_value, edge.visits, edge.room_access_plan_score, edge.prior_score),
            )
        return max(
            alternatives,
            key=lambda edge: (edge.visits, edge.mean_value, edge.room_access_plan_score, edge.prior_score),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        access = self.cached_access_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_access_lookahead_slots,
        )
        totals = access["totals"]
        value -= self.room_leaf_access_barrier_weight * float(totals["access_barrier_cost"])
        value -= self.room_leaf_future_access_barrier_weight * float(
            totals["access_future_barrier_cost"]
        )
        value -= self.room_leaf_future_access_need_penalty * float(
            totals["access_future_need_count"]
        )
        value -= self.room_leaf_future_stand_blocker_penalty * float(
            totals["access_future_stand_blocker_count"]
        )
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        return float(max(low, min(high, value)))


class RoomSchedulePlanGFTSDMCTSPlanner(RoomAccessPlanGFTSDMCTSPlanner):
    """A11 planner that consumes explicit multi-box schedule dependencies.

    The planner does not infer a schedule inside MCTS.  It reads the A11 SD
    summary and rewards actions that release blocker boxes, reduce dependency
    debt, or avoid occupying future stand cells.
    """

    provider_id = "room_schedule_plan_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        room_schedule_lookahead_slots: int = 2,
        room_schedule_barrier_reduction_weight: float = 0.8,
        room_schedule_dependency_reduction_weight: float = 4.0,
        room_schedule_future_dependency_reduction_weight: float = 5.0,
        room_schedule_blocker_box_reduction_weight: float = 6.0,
        room_schedule_critical_box_reduction_weight: float = 8.0,
        room_schedule_future_critical_box_reduction_weight: float = 10.0,
        room_schedule_release_action_weight: float = 5.0,
        room_schedule_destructive_action_reduction_weight: float = 0.0,
        room_schedule_immediate_critical_entry_penalty: float = 20.0,
        room_schedule_immediate_future_entry_penalty: float = 24.0,
        room_schedule_immediate_blocker_release_weight: float = 8.0,
        room_schedule_worsening_penalty: float = 5.0,
        room_leaf_schedule_barrier_weight: float = 0.08,
        room_leaf_schedule_dependency_penalty: float = 0.8,
        room_leaf_schedule_critical_box_penalty: float = 1.5,
        room_leaf_schedule_release_bonus: float = 0.5,
        room_schedule_root_guard: bool = True,
        **kwargs,
    ):
        super().__init__(*args, room_access_lookahead_slots=room_schedule_lookahead_slots, **kwargs)
        self.room_schedule_lookahead_slots = max(int(room_schedule_lookahead_slots), 1)
        self.room_schedule_barrier_reduction_weight = float(room_schedule_barrier_reduction_weight)
        self.room_schedule_dependency_reduction_weight = float(room_schedule_dependency_reduction_weight)
        self.room_schedule_future_dependency_reduction_weight = float(
            room_schedule_future_dependency_reduction_weight
        )
        self.room_schedule_blocker_box_reduction_weight = float(room_schedule_blocker_box_reduction_weight)
        self.room_schedule_critical_box_reduction_weight = float(room_schedule_critical_box_reduction_weight)
        self.room_schedule_future_critical_box_reduction_weight = float(
            room_schedule_future_critical_box_reduction_weight
        )
        self.room_schedule_release_action_weight = float(room_schedule_release_action_weight)
        self.room_schedule_destructive_action_reduction_weight = float(
            room_schedule_destructive_action_reduction_weight
        )
        self.room_schedule_immediate_critical_entry_penalty = float(
            room_schedule_immediate_critical_entry_penalty
        )
        self.room_schedule_immediate_future_entry_penalty = float(
            room_schedule_immediate_future_entry_penalty
        )
        self.room_schedule_immediate_blocker_release_weight = float(
            room_schedule_immediate_blocker_release_weight
        )
        self.room_schedule_worsening_penalty = float(room_schedule_worsening_penalty)
        self.room_leaf_schedule_barrier_weight = float(room_leaf_schedule_barrier_weight)
        self.room_leaf_schedule_dependency_penalty = float(room_leaf_schedule_dependency_penalty)
        self.room_leaf_schedule_critical_box_penalty = float(room_leaf_schedule_critical_box_penalty)
        self.room_leaf_schedule_release_bonus = float(room_leaf_schedule_release_bonus)
        self.room_schedule_root_guard = bool(room_schedule_root_guard)

    @property
    def schedule_plan_model(self) -> SokobanRoomSchedulePlanGFTSDModel:
        if not isinstance(self.model, SokobanRoomSchedulePlanGFTSDModel):
            raise TypeError("RoomSchedulePlanGFTSDMCTSPlanner requires SokobanRoomSchedulePlanGFTSDModel.")
        return self.model

    def cached_schedule_plan_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        *,
        lookahead_slots: int,
    ) -> dict[str, Any]:
        key = self._summary_cache_key("schedule", state, commitment_state, int(lookahead_slots))
        cached = self._summary_cache.get(key)
        if cached is None:
            cached = self.schedule_plan_model.room_schedule_plan_summary(
                state,
                commitment_state,
                lookahead_slots=int(lookahead_slots),
            )
            self._summary_cache[key] = cached
        return cached

    def _schedule_plan_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        after = self.cached_schedule_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        barrier_delta = float(before_totals["schedule_barrier_cost"] - after_totals["schedule_barrier_cost"])
        dependency_delta = int(before_totals["dependency_count"]) - int(after_totals["dependency_count"])
        future_dependency_delta = int(before_totals["future_dependency_count"]) - int(
            after_totals["future_dependency_count"]
        )
        blocker_delta = int(before_totals["blocker_box_count"]) - int(after_totals["blocker_box_count"])
        critical_box_delta = int(before_totals["critical_stand_box_count"]) - int(
            after_totals["critical_stand_box_count"]
        )
        future_critical_box_delta = int(before_totals["future_critical_stand_box_count"]) - int(
            after_totals["future_critical_stand_box_count"]
        )
        release_action_delta = int(after_totals["release_action_count"]) - int(
            before_totals["release_action_count"]
        )
        destructive_delta = int(before_totals["destructive_action_count"]) - int(
            after_totals["destructive_action_count"]
        )
        source = state.boxes[candidate.action.physical_action.box_index]
        target = add_cells(source, DIRECTIONS[candidate.action.physical_action.direction])
        critical_stands = {tuple(cell) for cell in before["critical_stand_cells"]}
        future_critical_stands = {tuple(cell) for cell in before["future_critical_stand_cells"]}
        blocker_boxes = {tuple(cell) for cell in before["blocker_boxes"]}
        immediate_critical_entry = int(target in critical_stands and source not in critical_stands)
        immediate_future_entry = int(target in future_critical_stands and source not in future_critical_stands)
        immediate_blocker_release = int(source in blocker_boxes and target not in critical_stands)
        worsening = (
            max(0, -dependency_delta)
            + max(0, -future_dependency_delta)
            + max(0, -blocker_delta)
            + max(0, -critical_box_delta)
            + max(0, -future_critical_box_delta)
            + max(0, -destructive_delta)
            + immediate_critical_entry
            + immediate_future_entry
        )
        score = (
            self.room_schedule_barrier_reduction_weight * barrier_delta
            + self.room_schedule_dependency_reduction_weight * float(dependency_delta)
            + self.room_schedule_future_dependency_reduction_weight * float(future_dependency_delta)
            + self.room_schedule_blocker_box_reduction_weight * float(blocker_delta)
            + self.room_schedule_critical_box_reduction_weight * float(critical_box_delta)
            + self.room_schedule_future_critical_box_reduction_weight * float(future_critical_box_delta)
            + self.room_schedule_release_action_weight * float(release_action_delta)
            + self.room_schedule_destructive_action_reduction_weight * float(max(0, destructive_delta))
            + self.room_schedule_immediate_blocker_release_weight * float(immediate_blocker_release)
            - self.room_schedule_immediate_critical_entry_penalty * float(immediate_critical_entry)
            - self.room_schedule_immediate_future_entry_penalty * float(immediate_future_entry)
            - self.room_schedule_worsening_penalty * float(worsening)
        )
        candidate.room_schedule_plan_score = float(score)
        candidate.room_schedule_plan_summary = {
            "before": {
                "totals": before_totals,
                "dependencies": before["dependencies"],
                "release_actions": before["release_actions"],
                "destructive_actions": before["destructive_actions"],
            },
            "after": {
                "totals": after_totals,
                "dependencies": after["dependencies"],
                "release_actions": after["release_actions"],
                "destructive_actions": after["destructive_actions"],
            },
            "delta": {
                "schedule_barrier_cost": float(barrier_delta),
                "dependency_count": int(dependency_delta),
                "future_dependency_count": int(future_dependency_delta),
                "blocker_box_count": int(blocker_delta),
                "critical_stand_box_count": int(critical_box_delta),
                "future_critical_stand_box_count": int(future_critical_box_delta),
                "release_action_count": int(release_action_delta),
                "destructive_action_count": int(destructive_delta),
                "immediate_critical_entry": int(immediate_critical_entry),
                "immediate_future_entry": int(immediate_future_entry),
                "immediate_blocker_release": int(immediate_blocker_release),
                "worsening_count": int(worsening),
            },
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._schedule_plan_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.room_schedule_root_guard:
            return best
        best_delta = best.room_schedule_plan_summary.get("delta", {})
        best_worsens = (
            int(best_delta.get("future_dependency_count", 0)) < 0
            or int(best_delta.get("future_critical_stand_box_count", 0)) < 0
            or int(best_delta.get("destructive_action_count", 0)) < 0
            or int(best_delta.get("immediate_critical_entry", 0)) > 0
            or int(best_delta.get("immediate_future_entry", 0)) > 0
        )
        if not best_worsens:
            return best
        alternatives = []
        for edge in root.children.values():
            delta = edge.room_schedule_plan_summary.get("delta", {})
            if (
                int(delta.get("future_dependency_count", 0)) >= 0
                and int(delta.get("future_critical_stand_box_count", 0)) >= 0
                and int(delta.get("destructive_action_count", 0)) >= 0
                and int(delta.get("immediate_critical_entry", 0)) == 0
                and int(delta.get("immediate_future_entry", 0)) == 0
            ):
                alternatives.append(edge)
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_schedule_plan_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_schedule_plan_score,
                edge.prior_score,
            ),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        value -= self.room_leaf_schedule_barrier_weight * float(totals["schedule_barrier_cost"])
        value -= self.room_leaf_schedule_dependency_penalty * float(totals["dependency_count"])
        value -= self.room_leaf_schedule_critical_box_penalty * float(totals["critical_stand_box_count"])
        value += self.room_leaf_schedule_release_bonus * float(totals["release_action_count"])
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        return float(max(low, min(high, value)))


class RoomScheduleStallGuardGFTSDMCTSPlanner(RoomSchedulePlanGFTSDMCTSPlanner):
    """A12 planner with dependency-stall loop guards.

    A11 exposes schedule dependencies but can still loop when schedule metrics
    remain unchanged.  A12 adds root-level and prior penalties for repeated
    physical moves, reverse shuttles, repeated exact states, and repeated
    schedule signatures under non-improving dependency states.
    """

    provider_id = "room_schedule_stall_guard_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        recent_moved_edges: Iterable[tuple] | None = None,
        recent_schedule_keys: Iterable[tuple] | None = None,
        room_schedule_stall_penalty: float = 8.0,
        room_schedule_reverse_edge_penalty: float = 18.0,
        room_schedule_repeated_edge_penalty: float = 10.0,
        room_schedule_repeated_state_penalty: float = 20.0,
        room_schedule_repeated_signature_penalty: float = 6.0,
        room_schedule_stall_root_guard: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.recent_moved_edges = {
            (tuple(source), tuple(target))
            for source, target in (recent_moved_edges or ())
        }
        self.recent_schedule_keys = set(recent_schedule_keys or ())
        self.room_schedule_stall_penalty = float(room_schedule_stall_penalty)
        self.room_schedule_reverse_edge_penalty = float(room_schedule_reverse_edge_penalty)
        self.room_schedule_repeated_edge_penalty = float(room_schedule_repeated_edge_penalty)
        self.room_schedule_repeated_state_penalty = float(room_schedule_repeated_state_penalty)
        self.room_schedule_repeated_signature_penalty = float(room_schedule_repeated_signature_penalty)
        self.room_schedule_stall_root_guard = bool(room_schedule_stall_root_guard)

    def schedule_stall_signature(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> tuple:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        totals = schedule["totals"]
        return (
            int(self.model.boxes_on_goals(state)),
            int(round(float(slot["slot_schedule_cost"]) * 1000.0)),
            int(totals["dependency_count"]),
            int(totals["future_dependency_count"]),
            int(totals["critical_stand_box_count"]),
            int(totals["release_action_count"]),
        )

    def _schedule_stall_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        after = self.cached_schedule_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        before_slot = self.model.room_slot_progress_summary(state, commitment_state)
        after_slot = self.model.room_slot_progress_summary(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        dependency_delta = int(before_totals["dependency_count"]) - int(after_totals["dependency_count"])
        future_dependency_delta = int(before_totals["future_dependency_count"]) - int(
            after_totals["future_dependency_count"]
        )
        barrier_delta = float(before_totals["schedule_barrier_cost"] - after_totals["schedule_barrier_cost"])
        slot_delta = float(before_slot["slot_schedule_cost"] - after_slot["slot_schedule_cost"])
        goal_delta = int(self.model.boxes_on_goals(candidate.child_state)) - int(self.model.boxes_on_goals(state))
        release_delta = int(after_totals["release_action_count"]) - int(before_totals["release_action_count"])
        improved = (
            goal_delta > 0
            or slot_delta > 0.0
            or dependency_delta > 0
            or future_dependency_delta > 0
            or barrier_delta > 0.5
            or release_delta > 0
        )
        source = state.boxes[candidate.action.physical_action.box_index]
        target = add_cells(source, DIRECTIONS[candidate.action.physical_action.direction])
        edge_key = (tuple(source), tuple(target))
        reverse_edge_key = (tuple(target), tuple(source))
        repeated_edge = edge_key in self.recent_moved_edges
        reverse_edge = reverse_edge_key in self.recent_moved_edges
        child_key = self.state_key(candidate.child_state, candidate.child_commitment_state)
        repeated_state = child_key in self.recent_commitment_keys
        child_signature = self.schedule_stall_signature(candidate.child_state, candidate.child_commitment_state)
        repeated_signature = child_signature in self.recent_schedule_keys
        dependency_active = int(before_totals["dependency_count"]) > 0
        stalled = bool(dependency_active and not improved)
        score = -(
            self.room_schedule_stall_penalty * float(stalled)
            + self.room_schedule_reverse_edge_penalty * float(stalled and reverse_edge)
            + self.room_schedule_repeated_edge_penalty * float(stalled and repeated_edge)
            + self.room_schedule_repeated_state_penalty * float(repeated_state)
            + self.room_schedule_repeated_signature_penalty * float(stalled and repeated_signature)
        )
        candidate.room_schedule_stall_score = float(score)
        candidate.room_schedule_stall_summary = {
            "moved_source": list(source),
            "moved_target": list(target),
            "dependency_active": bool(dependency_active),
            "improved": bool(improved),
            "stalled": bool(stalled),
            "goal_delta": int(goal_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "schedule_barrier_cost_delta": float(barrier_delta),
            "dependency_count_delta": int(dependency_delta),
            "future_dependency_count_delta": int(future_dependency_delta),
            "release_action_count_delta": int(release_delta),
            "repeated_edge": bool(repeated_edge),
            "reverse_recent_edge": bool(reverse_edge),
            "repeated_commitment_state": bool(repeated_state),
            "repeated_schedule_signature": bool(repeated_signature),
            "recent_moved_edge_count": int(len(self.recent_moved_edges)),
            "recent_schedule_key_count": int(len(self.recent_schedule_keys)),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._schedule_stall_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.room_schedule_stall_root_guard:
            return best
        best_stall = best.room_schedule_stall_summary
        best_bad = bool(best_stall.get("stalled")) and (
            bool(best_stall.get("reverse_recent_edge"))
            or bool(best_stall.get("repeated_edge"))
            or bool(best_stall.get("repeated_commitment_state"))
            or bool(best_stall.get("repeated_schedule_signature"))
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if edge.room_schedule_stall_score >= 0.0
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_schedule_stall_score > best.room_schedule_stall_score
            ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_schedule_plan_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_schedule_plan_score,
                edge.prior_score,
            ),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        signature = self.schedule_stall_signature(state, commitment_state)
        if signature in self.recent_schedule_keys:
            value -= 0.5 * self.room_schedule_repeated_signature_penalty
            low = min(self.deadend_value, self.goal_value)
            high = max(self.deadend_value, self.goal_value)
            value = float(max(low, min(high, value)))
        return value


class RoomScheduleAccessRecoveryGFTSDMCTSPlanner(RoomScheduleStallGuardGFTSDMCTSPlanner):
    """A13 planner with access-recovery guards for hard instances.

    A12 can avoid repeated schedule signatures while still steering into states
    with too few legal non-deadlocked pushes to recover the current dependency
    graph.  A13 keeps A11/A12 schedule summaries and adds a consumer-side
    recovery signal for player-access margin, target blockers, and release
    opportunities.  It does not inject teacher or reference actions.
    """

    provider_id = "room_schedule_access_recovery_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        room_access_recovery_min_legal_pushes: int = 2,
        room_access_recovery_legal_push_weight: float = 2.0,
        room_access_recovery_legal_collapse_penalty: float = 6.0,
        room_access_recovery_low_legal_penalty: float = 45.0,
        room_access_recovery_no_legal_penalty: float = 120.0,
        room_access_recovery_target_blocker_weight: float = 14.0,
        room_access_recovery_unreachable_stand_weight: float = 6.0,
        room_access_recovery_release_action_weight: float = 10.0,
        room_access_recovery_dependency_weight: float = 2.0,
        room_access_recovery_root_guard: bool = True,
        room_leaf_access_recovery_low_legal_penalty: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.room_access_recovery_min_legal_pushes = max(int(room_access_recovery_min_legal_pushes), 0)
        self.room_access_recovery_legal_push_weight = float(room_access_recovery_legal_push_weight)
        self.room_access_recovery_legal_collapse_penalty = float(
            room_access_recovery_legal_collapse_penalty
        )
        self.room_access_recovery_low_legal_penalty = float(room_access_recovery_low_legal_penalty)
        self.room_access_recovery_no_legal_penalty = float(room_access_recovery_no_legal_penalty)
        self.room_access_recovery_target_blocker_weight = float(room_access_recovery_target_blocker_weight)
        self.room_access_recovery_unreachable_stand_weight = float(
            room_access_recovery_unreachable_stand_weight
        )
        self.room_access_recovery_release_action_weight = float(room_access_recovery_release_action_weight)
        self.room_access_recovery_dependency_weight = float(room_access_recovery_dependency_weight)
        self.room_access_recovery_root_guard = bool(room_access_recovery_root_guard)
        self.room_leaf_access_recovery_low_legal_penalty = float(
            room_leaf_access_recovery_low_legal_penalty
        )

    def _nonblocked_legal_push_count(self, state: SokobanState) -> int:
        count = 0
        for action in self.model.legal_flows(state):
            child = self.model.step_grid_flow(state, action)
            if not self.is_blocked(child):
                count += 1
        return int(count)

    def _access_recovery_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        after = self.cached_schedule_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        before_dependency = int(before_totals["dependency_count"]) + int(
            before_totals["future_dependency_count"]
        )
        after_dependency = int(after_totals["dependency_count"]) + int(
            after_totals["future_dependency_count"]
        )
        dependency_active = before_dependency > 0
        before_legal = self._nonblocked_legal_push_count(state)
        after_legal = self._nonblocked_legal_push_count(candidate.child_state)
        legal_delta = int(after_legal - before_legal)
        legal_collapse = max(0, int(before_legal - after_legal))
        after_tier = self.model.tier_a_status(candidate.child_state)
        no_legal = bool(after_tier.get("no_legal_pushes")) or after_legal <= 0
        low_legal = (
            dependency_active
            and not self.model.is_solved(candidate.child_state)
            and after_dependency > 0
            and after_legal < self.room_access_recovery_min_legal_pushes
        )
        target_blocker_delta = int(before_totals["target_blocker_count"]) - int(
            after_totals["target_blocker_count"]
        )
        unreachable_delta = int(before_totals["unreachable_stand_count"]) - int(
            after_totals["unreachable_stand_count"]
        )
        release_delta = int(after_totals["release_action_count"]) - int(
            before_totals["release_action_count"]
        )
        dependency_delta = int(before_dependency - after_dependency)
        score = (
            self.room_access_recovery_legal_push_weight * float(legal_delta)
            - self.room_access_recovery_legal_collapse_penalty * float(legal_collapse)
            - self.room_access_recovery_low_legal_penalty * float(low_legal)
            - self.room_access_recovery_no_legal_penalty * float(no_legal and dependency_active)
            + self.room_access_recovery_target_blocker_weight * float(target_blocker_delta)
            + self.room_access_recovery_unreachable_stand_weight * float(unreachable_delta)
            + self.room_access_recovery_release_action_weight * float(release_delta)
            + self.room_access_recovery_dependency_weight * float(dependency_delta)
        )
        candidate.room_access_recovery_score = float(score)
        candidate.room_access_recovery_summary = {
            "dependency_active": bool(dependency_active),
            "before_dependency_total": int(before_dependency),
            "after_dependency_total": int(after_dependency),
            "before_nonblocked_legal_push_count": int(before_legal),
            "after_nonblocked_legal_push_count": int(after_legal),
            "legal_push_count_delta": int(legal_delta),
            "legal_collapse_count": int(legal_collapse),
            "low_legal_margin": bool(low_legal),
            "no_legal_pushes": bool(no_legal),
            "target_blocker_count_delta": int(target_blocker_delta),
            "unreachable_stand_count_delta": int(unreachable_delta),
            "release_action_count_delta": int(release_delta),
            "dependency_total_delta": int(dependency_delta),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._access_recovery_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.room_access_recovery_root_guard:
            return best
        best_recovery = best.room_access_recovery_summary
        best_bad = bool(best_recovery.get("dependency_active")) and (
            bool(best_recovery.get("no_legal_pushes"))
            or bool(best_recovery.get("low_legal_margin"))
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if not bool(edge.room_access_recovery_summary.get("no_legal_pushes"))
            and not bool(edge.room_access_recovery_summary.get("low_legal_margin"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_access_recovery_score > best.room_access_recovery_score
            ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_access_recovery_score,
                    edge.room_schedule_plan_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_access_recovery_score,
                edge.room_schedule_plan_score,
                edge.prior_score,
            ),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        dependency_active = (
            int(totals["dependency_count"]) + int(totals["future_dependency_count"])
        ) > 0
        if dependency_active and self._nonblocked_legal_push_count(state) < self.room_access_recovery_min_legal_pushes:
            value -= self.room_leaf_access_recovery_low_legal_penalty
            low = min(self.deadend_value, self.goal_value)
            high = max(self.deadend_value, self.goal_value)
            value = float(max(low, min(high, value)))
        return value


class RoomScheduleRecoveryMacroGFTSDMCTSPlanner(RoomScheduleAccessRecoveryGFTSDMCTSPlanner):
    """A14 planner with a bounded release/access recovery macro.

    A13 can diagnose low legal-margin loops but only scores single root pushes.
    A14 adds a conservative short macro after the selected root action.  The
    macro searches legal, non-deadlocked pushes and executes a suffix only when
    it improves schedule recovery metrics such as legal-push margin, target
    blockers, unreachable stand cells, release actions, or dependency count.
    """

    provider_id = "room_schedule_recovery_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        recovery_macro_enabled: bool = True,
        recovery_macro_max_extra_pushes: int = 4,
        recovery_macro_max_nodes: int = 400,
        recovery_macro_score_threshold: float = 2.0,
        recovery_macro_legal_push_weight: float = 4.0,
        recovery_macro_target_blocker_weight: float = 18.0,
        recovery_macro_unreachable_stand_weight: float = 10.0,
        recovery_macro_release_action_weight: float = 16.0,
        recovery_macro_dependency_weight: float = 8.0,
        recovery_macro_slot_weight: float = 0.8,
        recovery_macro_low_legal_penalty: float = 18.0,
        recovery_macro_no_legal_penalty: float = 100.0,
        recovery_macro_step_penalty: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.recovery_macro_enabled = bool(recovery_macro_enabled)
        self.recovery_macro_max_extra_pushes = max(int(recovery_macro_max_extra_pushes), 0)
        self.recovery_macro_max_nodes = max(int(recovery_macro_max_nodes), 1)
        self.recovery_macro_score_threshold = float(recovery_macro_score_threshold)
        self.recovery_macro_legal_push_weight = float(recovery_macro_legal_push_weight)
        self.recovery_macro_target_blocker_weight = float(recovery_macro_target_blocker_weight)
        self.recovery_macro_unreachable_stand_weight = float(recovery_macro_unreachable_stand_weight)
        self.recovery_macro_release_action_weight = float(recovery_macro_release_action_weight)
        self.recovery_macro_dependency_weight = float(recovery_macro_dependency_weight)
        self.recovery_macro_slot_weight = float(recovery_macro_slot_weight)
        self.recovery_macro_low_legal_penalty = float(recovery_macro_low_legal_penalty)
        self.recovery_macro_no_legal_penalty = float(recovery_macro_no_legal_penalty)
        self.recovery_macro_step_penalty = float(recovery_macro_step_penalty)

    def _recovery_macro_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        totals = schedule["totals"]
        legal = self._nonblocked_legal_push_count(state)
        dependency_total = int(totals["dependency_count"]) + int(totals["future_dependency_count"])
        return {
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "dependency_total": int(dependency_total),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "unreachable_stand_count": int(totals["unreachable_stand_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "nonblocked_legal_push_count": int(legal),
            "low_legal_margin": bool(
                dependency_total > 0 and legal < self.room_access_recovery_min_legal_pushes
            ),
            "no_legal_pushes": bool(legal <= 0),
        }

    def _recovery_macro_score(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        depth: int,
    ) -> float:
        score = (
            self.recovery_macro_legal_push_weight
            * float(int(after["nonblocked_legal_push_count"]) - int(before["nonblocked_legal_push_count"]))
            + self.recovery_macro_target_blocker_weight
            * float(int(before["target_blocker_count"]) - int(after["target_blocker_count"]))
            + self.recovery_macro_unreachable_stand_weight
            * float(int(before["unreachable_stand_count"]) - int(after["unreachable_stand_count"]))
            + self.recovery_macro_release_action_weight
            * float(int(after["release_action_count"]) - int(before["release_action_count"]))
            + self.recovery_macro_dependency_weight
            * float(int(before["dependency_total"]) - int(after["dependency_total"]))
            + self.recovery_macro_slot_weight
            * float(float(before["slot_schedule_cost"]) - float(after["slot_schedule_cost"]))
            - self.recovery_macro_low_legal_penalty * float(after["low_legal_margin"])
            - self.recovery_macro_no_legal_penalty * float(after["no_legal_pushes"])
            - self.recovery_macro_step_penalty * float(depth)
        )
        return float(score)

    def _advance_recovery_commitment(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        action: SokobanPushAction,
        child_state: SokobanState,
    ) -> RoomCommitmentState:
        return self.model.advance_room_commitment_state(
            state,
            commitment_state,
            action,
            RoomSlotOption(kind="none"),
            child_state,
        )

    def _best_recovery_suffix(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.recovery_macro_max_extra_pushes <= 0:
            return None
        best: dict[str, Any] | None = None
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "actions": [],
                "commitment_states": [],
                "rows": [],
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        expanded = 0
        while queue and expanded < self.recovery_macro_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.recovery_macro_max_extra_pushes:
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                key = self.model.gft_canonical_key_cached(child)
                if key in node["seen"]:
                    continue
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._recovery_macro_metrics(child, child_commitment)
                child_depth = depth + 1
                score = self._recovery_macro_score(reference_metrics, metrics, depth=child_depth)
                source, target = self._moved_cells(node["state"], action)
                row = {
                    "macro_index": int(child_depth),
                    "physical_action": self._physical_action_dict(node["state"], action),
                    "reason": "access_recovery_macro_push",
                    "source": list(source),
                    "target": list(target),
                    "recovery_score": float(score),
                    "recovery_metrics": metrics,
                }
                candidate = {
                    "score": float(score),
                    "actions": [*node["actions"], action],
                    "commitment_states": [*node["commitment_states"], child_commitment],
                    "rows": [*node["rows"], row],
                    "final_metrics": metrics,
                }
                if best is None or (
                    candidate["score"],
                    -len(candidate["actions"]),
                ) > (
                    best["score"],
                    -len(best["actions"]),
                ):
                    best = candidate
                if child_depth < self.recovery_macro_max_extra_pushes and not metrics["no_legal_pushes"]:
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate["actions"],
                            "commitment_states": candidate["commitment_states"],
                            "rows": candidate["rows"],
                            "seen": set(node["seen"]) | {key},
                        }
                    )
                expanded += 1
                if expanded >= self.recovery_macro_max_nodes:
                    break
        if best is None or float(best["score"]) < self.recovery_macro_score_threshold:
            return None
        return best

    def _recovery_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        base = plan.get("corridor_macro", {})
        if not self.recovery_macro_enabled or plan.get("physical_action") is None:
            return base
        before_metrics = self._recovery_macro_metrics(state, commitment_state)
        if int(before_metrics["dependency_total"]) <= 0:
            return base
        base_actions = list(base.get("physical_actions", []))
        if not base_actions:
            base_actions = [plan["physical_action"]]
        base_commitments = list(base.get("after_commitment_states", []))
        current_state = state
        current_commitment = commitment_state
        for idx, action in enumerate(base_actions):
            child = self.model.step_grid_flow(current_state, action)
            if idx < len(base_commitments):
                current_commitment = self.model.normalize_room_commitment_state(
                    child,
                    base_commitments[idx],
                )
            else:
                current_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
                base_commitments.append(current_commitment)
            current_state = child
        reference_metrics = self._recovery_macro_metrics(current_state, current_commitment)
        suffix = self._best_recovery_suffix(current_state, current_commitment, reference_metrics)
        if suffix is None:
            return {
                **base,
                "recovery_macro_enabled": True,
                "recovery_macro_triggered": False,
                "recovery_macro_reason": "no_positive_recovery_suffix",
                "recovery_reference_metrics": reference_metrics,
            }
        base_rows = list(base.get("rows", []))
        offset_rows = []
        for offset, row in enumerate(suffix["rows"], start=len(base_actions)):
            offset_rows.append({**row, "macro_index": int(offset)})
        sequence = [*base_actions, *suffix["actions"]]
        commitment_states = [*base_commitments, *suffix["commitment_states"]]
        return {
            **base,
            "enabled": True,
            "triggered": True,
            "reason": "access_recovery_macro",
            "physical_actions": sequence,
            "after_commitment_states": commitment_states,
            "rows": [*base_rows, *offset_rows],
            "macro_push_count": int(len(sequence)),
            "extra_push_count": int(max(0, len(sequence) - 1)),
            "recovery_macro_enabled": True,
            "recovery_macro_triggered": True,
            "recovery_macro_score": float(suffix["score"]),
            "recovery_reference_metrics": reference_metrics,
            "recovery_final_metrics": suffix["final_metrics"],
            "recovery_extra_push_count": int(len(suffix["actions"])),
        }

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(state, commitment_state)
        plan = super().plan_with_corridor_macro(state, commitment_state)
        plan["corridor_macro"] = self._recovery_macro_from_plan(state, commitment_state, plan)
        return plan


class RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner(RoomScheduleRecoveryMacroGFTSDMCTSPlanner):
    """A15 planner with recovery macro bound to one schedule dependency row.

    A14 proves that a bounded macro can be executed, but its objective is too
    broad.  A15 keeps the same execution boundary and binds the suffix search to
    one concrete A11 schedule dependency, currently either an unreachable stand
    cell or a target cell occupied by another box.
    """

    provider_id = "room_schedule_dependency_bound_macro_gft_sdmcts_v1"
    dependency_bound_macro_reason = "dependency_bound_recovery_macro_push"

    def __init__(
        self,
        *args,
        dependency_bound_macro_score_threshold: float = 20.0,
        dependency_bound_macro_row_disappear_weight: float = 90.0,
        dependency_bound_macro_kind_resolution_weight: float = 55.0,
        dependency_bound_macro_target_resolution_weight: float = 70.0,
        dependency_bound_macro_stand_resolution_weight: float = 70.0,
        dependency_bound_macro_blocker_weight: float = 30.0,
        dependency_bound_macro_selected_release_weight: float = 24.0,
        dependency_bound_macro_release_weight: float = 10.0,
        dependency_bound_macro_dependency_weight: float = 12.0,
        dependency_bound_macro_generic_recovery_weight: float = 5.0,
        dependency_bound_macro_legal_push_weight: float = 1.5,
        dependency_bound_macro_slot_weight: float = 0.15,
        dependency_bound_macro_unresolved_penalty: float = 18.0,
        dependency_bound_macro_new_blocker_penalty: float = 14.0,
        dependency_bound_macro_no_progress_penalty: float = 22.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dependency_bound_macro_score_threshold = float(
            dependency_bound_macro_score_threshold
        )
        self.dependency_bound_macro_row_disappear_weight = float(
            dependency_bound_macro_row_disappear_weight
        )
        self.dependency_bound_macro_kind_resolution_weight = float(
            dependency_bound_macro_kind_resolution_weight
        )
        self.dependency_bound_macro_target_resolution_weight = float(
            dependency_bound_macro_target_resolution_weight
        )
        self.dependency_bound_macro_stand_resolution_weight = float(
            dependency_bound_macro_stand_resolution_weight
        )
        self.dependency_bound_macro_blocker_weight = float(dependency_bound_macro_blocker_weight)
        self.dependency_bound_macro_selected_release_weight = float(
            dependency_bound_macro_selected_release_weight
        )
        self.dependency_bound_macro_release_weight = float(dependency_bound_macro_release_weight)
        self.dependency_bound_macro_dependency_weight = float(dependency_bound_macro_dependency_weight)
        self.dependency_bound_macro_generic_recovery_weight = float(
            dependency_bound_macro_generic_recovery_weight
        )
        self.dependency_bound_macro_legal_push_weight = float(
            dependency_bound_macro_legal_push_weight
        )
        self.dependency_bound_macro_slot_weight = float(dependency_bound_macro_slot_weight)
        self.dependency_bound_macro_unresolved_penalty = float(
            dependency_bound_macro_unresolved_penalty
        )
        self.dependency_bound_macro_new_blocker_penalty = float(
            dependency_bound_macro_new_blocker_penalty
        )
        self.dependency_bound_macro_no_progress_penalty = float(
            dependency_bound_macro_no_progress_penalty
        )

    @staticmethod
    def _dependency_cell(row: dict[str, Any], field: str) -> Cell | None:
        value = row.get(field)
        if value is None:
            return None
        return int(value[0]), int(value[1])

    @staticmethod
    def _dependency_relation_key(row: dict[str, Any]) -> tuple:
        return (
            int(row.get("room_id", -1)),
            tuple(row.get("slot_cell") or ()),
            tuple(row.get("stand_cell") or ()),
            tuple(row.get("target_cell") or ()),
            row.get("candidate_direction"),
        )

    @staticmethod
    def _dependency_relevant_kinds(row: dict[str, Any]) -> set[str]:
        kinds = set(row.get("dependency_kinds", []))
        return {
            kind
            for kind in kinds
            if kind in {"stand_cell_unreachable_by_player", "target_occupied_by_box"}
        }

    def _dependency_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "room_id": int(row.get("room_id", -1)),
            "horizon_index": int(row.get("horizon_index", -1)),
            "future": bool(row.get("future", False)),
            "assigned_box": row.get("assigned_box"),
            "slot_cell": row.get("slot_cell"),
            "stand_cell": row.get("stand_cell"),
            "target_cell": row.get("target_cell"),
            "blocker_cells": list(row.get("blocker_cells", [])),
            "dependency_kinds": list(row.get("dependency_kinds", [])),
            "selected_dependency_kinds": sorted(self._dependency_relevant_kinds(row)),
            "candidate_direction": row.get("candidate_direction"),
            "candidate_barrier_cost": float(row.get("candidate_barrier_cost", 0.0)),
            "dependency_relation_key": [
                list(item) if isinstance(item, tuple) else item
                for item in self._dependency_relation_key(row)
            ],
        }

    def _select_dependency_bound_row(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any] | None:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        candidates = [
            row
            for row in schedule.get("dependencies", [])
            if self._dependency_relevant_kinds(row)
        ]
        if not candidates:
            return None

        def priority(row: dict[str, Any]) -> tuple:
            kinds = self._dependency_relevant_kinds(row)
            return (
                int(not bool(row.get("future", False))),
                int("target_occupied_by_box" in kinds),
                int("stand_cell_unreachable_by_player" in kinds),
                float(row.get("candidate_barrier_cost", 0.0)),
                -int(row.get("horizon_index", 0)),
            )

        selected = max(candidates, key=priority)
        return self._dependency_to_dict(selected)

    def _matching_dependency_row(
        self,
        schedule: dict[str, Any],
        dependency: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = self._dependency_relation_key(dependency)
        for row in schedule.get("dependencies", []):
            if self._dependency_relation_key(row) == key:
                return row
        return None

    def _dependency_bound_release_actions(
        self,
        schedule: dict[str, Any],
        dependency: dict[str, Any],
    ) -> list[dict[str, Any]]:
        stand_cell = self._dependency_cell(dependency, "stand_cell")
        target_cell = self._dependency_cell(dependency, "target_cell")
        blocker_cells = {
            tuple(cell)
            for cell in dependency.get("blocker_cells", [])
        }
        selected_kinds = set(dependency.get("selected_dependency_kinds", []))
        if target_cell is not None and "target_occupied_by_box" in selected_kinds:
            blocker_cells.add(target_cell)
        watched = set(blocker_cells)
        if stand_cell is not None:
            watched.add(stand_cell)
        if target_cell is not None:
            watched.add(target_cell)
        rows = []
        for action in schedule.get("release_actions", []):
            source = tuple(action.get("source", ()))
            if source in watched:
                rows.append(action)
        return rows

    def _dependency_bound_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        dependency: dict[str, Any],
    ) -> dict[str, Any]:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        totals = schedule["totals"]
        boxes = set(state.boxes)
        reachable = set(self.model.reachable_cells(state))
        selected_kinds = set(dependency.get("selected_dependency_kinds", []))
        match = self._matching_dependency_row(schedule, dependency)
        present_kinds = set(match.get("dependency_kinds", [])) if match is not None else set()
        remaining_kinds = sorted(selected_kinds & present_kinds)
        stand_cell = self._dependency_cell(dependency, "stand_cell")
        target_cell = self._dependency_cell(dependency, "target_cell")
        blocker_cells = {
            tuple(cell)
            for cell in dependency.get("blocker_cells", [])
        }
        if target_cell is not None and "target_occupied_by_box" in selected_kinds:
            blocker_cells.add(target_cell)
        blockers_remaining = sorted(cell for cell in blocker_cells if cell in boxes)
        release_actions = self._dependency_bound_release_actions(schedule, dependency)
        dependency_total = int(totals["dependency_count"]) + int(totals["future_dependency_count"])
        legal = self._nonblocked_legal_push_count(state)
        return {
            "dependency_present": bool(match is not None),
            "dependency_future": None if match is None else bool(match.get("future", False)),
            "present_dependency_kinds": sorted(present_kinds),
            "remaining_selected_kinds": remaining_kinds,
            "remaining_selected_kind_count": int(len(remaining_kinds)),
            "stand_cell_reachable": bool(stand_cell is not None and stand_cell in reachable),
            "stand_cell_occupied_by_box": bool(stand_cell is not None and stand_cell in boxes),
            "target_occupied_by_box": bool(target_cell is not None and target_cell in boxes),
            "selected_blocker_remaining_count": int(len(blockers_remaining)),
            "selected_blockers_remaining": [list(cell) for cell in blockers_remaining],
            "selected_release_action_count": int(len(release_actions)),
            "selected_release_actions": release_actions,
            "boxes_on_goals": int(self.model.boxes_on_goals(state)),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "dependency_total": int(dependency_total),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "unreachable_stand_count": int(totals["unreachable_stand_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "critical_stand_box_count": int(totals["critical_stand_box_count"]),
            "future_critical_stand_box_count": int(totals["future_critical_stand_box_count"]),
            "destructive_action_count": int(totals["destructive_action_count"]),
            "nonblocked_legal_push_count": int(legal),
            "low_legal_margin": bool(
                dependency_total > 0 and legal < self.room_access_recovery_min_legal_pushes
            ),
            "no_legal_pushes": bool(legal <= 0),
        }

    def _dependency_bound_score(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        dependency: dict[str, Any],
        *,
        depth: int,
    ) -> tuple[float, dict[str, Any]]:
        selected_kinds = set(dependency.get("selected_dependency_kinds", []))
        row_disappeared = bool(before["dependency_present"]) and not bool(after["dependency_present"])
        kind_delta = int(before["remaining_selected_kind_count"]) - int(
            after["remaining_selected_kind_count"]
        )
        target_resolved = (
            "target_occupied_by_box" in selected_kinds
            and bool(before["target_occupied_by_box"])
            and not bool(after["target_occupied_by_box"])
        )
        stand_resolved = (
            "stand_cell_unreachable_by_player" in selected_kinds
            and not bool(before["stand_cell_reachable"])
            and bool(after["stand_cell_reachable"])
        )
        blocker_delta = int(before["selected_blocker_remaining_count"]) - int(
            after["selected_blocker_remaining_count"]
        )
        selected_release_delta = int(after["selected_release_action_count"]) - int(
            before["selected_release_action_count"]
        )
        release_delta = int(after["release_action_count"]) - int(before["release_action_count"])
        dependency_delta = int(before["dependency_total"]) - int(after["dependency_total"])
        target_blocker_delta = int(before["target_blocker_count"]) - int(
            after["target_blocker_count"]
        )
        unreachable_delta = int(before["unreachable_stand_count"]) - int(
            after["unreachable_stand_count"]
        )
        legal_delta = int(after["nonblocked_legal_push_count"]) - int(
            before["nonblocked_legal_push_count"]
        )
        slot_delta = float(before["slot_schedule_cost"]) - float(after["slot_schedule_cost"])
        new_target_blockers = max(0, -target_blocker_delta)
        unresolved_target = (
            "target_occupied_by_box" in selected_kinds
            and bool(after["target_occupied_by_box"])
        )
        unresolved_stand = (
            "stand_cell_unreachable_by_player" in selected_kinds
            and not bool(after["stand_cell_reachable"])
        )
        dependency_progress = bool(
            row_disappeared
            or kind_delta > 0
            or target_resolved
            or stand_resolved
            or blocker_delta > 0
            or selected_release_delta > 0
        )
        score = (
            self.dependency_bound_macro_row_disappear_weight * float(row_disappeared)
            + self.dependency_bound_macro_kind_resolution_weight * float(max(0, kind_delta))
            + self.dependency_bound_macro_target_resolution_weight * float(target_resolved)
            + self.dependency_bound_macro_stand_resolution_weight * float(stand_resolved)
            + self.dependency_bound_macro_blocker_weight * float(max(0, blocker_delta))
            + self.dependency_bound_macro_selected_release_weight * float(
                max(0, selected_release_delta)
            )
            + self.dependency_bound_macro_release_weight * float(max(0, release_delta))
            + self.dependency_bound_macro_dependency_weight * float(max(0, dependency_delta))
            + self.dependency_bound_macro_generic_recovery_weight
            * float(max(0, target_blocker_delta) + max(0, unreachable_delta))
            + self.dependency_bound_macro_legal_push_weight * float(legal_delta)
            + self.dependency_bound_macro_slot_weight * slot_delta
            - self.dependency_bound_macro_unresolved_penalty
            * float(unresolved_target + unresolved_stand)
            - self.dependency_bound_macro_new_blocker_penalty * float(new_target_blockers)
            - self.recovery_macro_low_legal_penalty * float(after["low_legal_margin"])
            - self.recovery_macro_no_legal_penalty * float(after["no_legal_pushes"])
            - self.recovery_macro_step_penalty * float(depth)
        )
        if not dependency_progress:
            score -= self.dependency_bound_macro_no_progress_penalty
        summary = {
            "row_disappeared": bool(row_disappeared),
            "remaining_selected_kind_delta": int(kind_delta),
            "target_resolved": bool(target_resolved),
            "stand_resolved": bool(stand_resolved),
            "selected_blocker_delta": int(blocker_delta),
            "selected_release_action_delta": int(selected_release_delta),
            "release_action_delta": int(release_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_blocker_delta),
            "unreachable_stand_count_delta": int(unreachable_delta),
            "legal_push_count_delta": int(legal_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "unresolved_target": bool(unresolved_target),
            "unresolved_stand": bool(unresolved_stand),
            "dependency_progress": bool(dependency_progress),
            "score": float(score),
        }
        return float(score), summary

    def _dependency_bound_candidate_acceptance(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        dependency: dict[str, Any],
        score_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "global_feasibility_pass": True,
            "global_feasibility_violations": [],
            "macro_admissible": bool(score_summary.get("dependency_progress", False)),
        }

    def _best_recovery_suffix(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.recovery_macro_max_extra_pushes <= 0:
            return None
        dependency = self._select_dependency_bound_row(state, commitment_state)
        if dependency is None:
            return None
        reference_dependency_metrics = self._dependency_bound_metrics(
            state,
            commitment_state,
            dependency,
        )
        best: dict[str, Any] | None = None
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "actions": [],
                "commitment_states": [],
                "rows": [],
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        expanded = 0
        while queue and expanded < self.recovery_macro_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.recovery_macro_max_extra_pushes:
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                key = self.model.gft_canonical_key_cached(child)
                if key in node["seen"]:
                    continue
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._dependency_bound_metrics(child, child_commitment, dependency)
                child_depth = depth + 1
                score, score_summary = self._dependency_bound_score(
                    reference_dependency_metrics,
                    metrics,
                    dependency,
                    depth=child_depth,
                )
                score_summary = {
                    **score_summary,
                    **self._dependency_bound_candidate_acceptance(
                        reference_dependency_metrics,
                        metrics,
                        dependency,
                        score_summary,
                    ),
                }
                source, target = self._moved_cells(node["state"], action)
                row = {
                    "macro_index": int(child_depth),
                    "physical_action": self._physical_action_dict(node["state"], action),
                    "reason": self.dependency_bound_macro_reason,
                    "source": list(source),
                    "target": list(target),
                    "bound_dependency": dependency,
                    "dependency_reference_metrics": reference_dependency_metrics,
                    "dependency_metrics": metrics,
                    "dependency_progress": score_summary,
                    "dependency_bound_score": float(score),
                    "recovery_score": float(score),
                }
                candidate = {
                    "score": float(score),
                    "dependency_progress": bool(score_summary["dependency_progress"]),
                    "macro_admissible": bool(score_summary.get("macro_admissible", False)),
                    "actions": [*node["actions"], action],
                    "commitment_states": [*node["commitment_states"], child_commitment],
                    "rows": [*node["rows"], row],
                    "final_metrics": {
                        "generic": self._recovery_macro_metrics(child, child_commitment),
                        "dependency_bound": metrics,
                    },
                    "bound_dependency": dependency,
                    "reference_dependency_metrics": reference_dependency_metrics,
                }
                if best is None or (
                    int(candidate["macro_admissible"]),
                    int(candidate["dependency_progress"]),
                    candidate["score"],
                    -len(candidate["actions"]),
                ) > (
                    int(best["macro_admissible"]),
                    int(best["dependency_progress"]),
                    best["score"],
                    -len(best["actions"]),
                ):
                    best = candidate
                if child_depth < self.recovery_macro_max_extra_pushes and not metrics["no_legal_pushes"]:
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate["actions"],
                            "commitment_states": candidate["commitment_states"],
                            "rows": candidate["rows"],
                            "seen": set(node["seen"]) | {key},
                        }
                    )
                expanded += 1
                if expanded >= self.recovery_macro_max_nodes:
                    break
        if best is None:
            return None
        if not bool(best.get("macro_admissible", False)):
            return None
        threshold = max(
            float(self.recovery_macro_score_threshold),
            float(self.dependency_bound_macro_score_threshold),
        )
        if float(best["score"]) < threshold:
            return None
        return best

    def _recovery_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        result = super()._recovery_macro_from_plan(state, commitment_state, plan)
        dependency_rows = [
            row
            for row in result.get("rows", [])
            if row.get("reason") == self.dependency_bound_macro_reason
        ]
        result["dependency_bound_macro_enabled"] = bool(self.recovery_macro_enabled)
        result["dependency_bound_macro_triggered"] = bool(dependency_rows)
        if dependency_rows:
            result["reason"] = "dependency_bound_recovery_macro"
            result["bound_dependency"] = dependency_rows[0].get("bound_dependency")
            result["dependency_reference_metrics"] = dependency_rows[0].get(
                "dependency_reference_metrics"
            )
            result["dependency_final_metrics"] = dependency_rows[-1].get("dependency_metrics")
            result["dependency_bound_macro_score"] = float(
                max(float(row.get("dependency_bound_score", 0.0)) for row in dependency_rows)
            )
            result["dependency_bound_extra_push_count"] = int(len(dependency_rows))
        elif result.get("recovery_macro_enabled"):
            result["dependency_bound_macro_reason"] = result.get(
                "recovery_macro_reason",
                result.get("reason", "no_dependency_bound_suffix"),
            )
        return result


class RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner(
    RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner
):
    """A16 planner that admits dependency-bound macros only if global guards pass.

    A15 can resolve a local dependency row while worsening slot-fill feasibility
    or legal-action margin.  A16 keeps the dependency binding and adds explicit
    postconditions on slot cost, target blockers, critical stand occupancy, and
    nonblocked legal-push margin.
    """

    provider_id = "room_schedule_feasibility_guarded_macro_gft_sdmcts_v1"
    dependency_bound_macro_reason = "feasibility_guarded_dependency_macro_push"

    def __init__(
        self,
        *args,
        feasibility_guard_max_slot_cost_regression: float = 0.0,
        feasibility_guard_max_schedule_barrier_regression: float = 8.0,
        feasibility_guard_allow_goal_progress: bool = True,
        feasibility_guard_violation_penalty: float = 80.0,
        feasibility_guard_slot_regression_penalty: float = 18.0,
        feasibility_guard_barrier_regression_penalty: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feasibility_guard_max_slot_cost_regression = float(
            feasibility_guard_max_slot_cost_regression
        )
        self.feasibility_guard_max_schedule_barrier_regression = float(
            feasibility_guard_max_schedule_barrier_regression
        )
        self.feasibility_guard_allow_goal_progress = bool(feasibility_guard_allow_goal_progress)
        self.feasibility_guard_violation_penalty = float(feasibility_guard_violation_penalty)
        self.feasibility_guard_slot_regression_penalty = float(
            feasibility_guard_slot_regression_penalty
        )
        self.feasibility_guard_barrier_regression_penalty = float(
            feasibility_guard_barrier_regression_penalty
        )

    def _dependency_bound_feasibility_summary(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        score_summary: dict[str, Any],
    ) -> dict[str, Any]:
        goal_delta = int(after["boxes_on_goals"]) - int(before["boxes_on_goals"])
        slot_regression = float(after["slot_schedule_cost"]) - float(before["slot_schedule_cost"])
        barrier_regression = float(after["schedule_barrier_cost"]) - float(
            before["schedule_barrier_cost"]
        )
        target_blocker_delta = int(score_summary.get("target_blocker_count_delta", 0))
        dependency_total_delta = int(score_summary.get("dependency_total_delta", 0))
        critical_delta = int(after["critical_stand_box_count"]) - int(
            before["critical_stand_box_count"]
        )
        future_critical_delta = int(after["future_critical_stand_box_count"]) - int(
            before["future_critical_stand_box_count"]
        )
        destructive_delta = int(after["destructive_action_count"]) - int(
            before["destructive_action_count"]
        )
        goal_progress = goal_delta > 0
        goal_override = bool(self.feasibility_guard_allow_goal_progress and goal_progress)
        slot_ok = (
            slot_regression <= self.feasibility_guard_max_slot_cost_regression
            or goal_override
        )
        barrier_ok = (
            barrier_regression <= self.feasibility_guard_max_schedule_barrier_regression
            or goal_override
        )
        legal_ok = (
            not bool(after["no_legal_pushes"])
            and not bool(after["low_legal_margin"])
            and int(after["nonblocked_legal_push_count"]) >= self.room_access_recovery_min_legal_pushes
        )
        target_blocker_ok = target_blocker_delta >= 0 or goal_override
        dependency_total_ok = dependency_total_delta >= 0 or goal_override
        critical_ok = critical_delta <= 0 and future_critical_delta <= 0
        destructive_ok = destructive_delta <= 0
        dependency_progress = bool(score_summary.get("dependency_progress", False))
        violations: list[str] = []
        if not dependency_progress:
            violations.append("no_dependency_progress")
        if not slot_ok:
            violations.append("slot_cost_regression")
        if not barrier_ok:
            violations.append("schedule_barrier_regression")
        if not legal_ok:
            violations.append("legal_margin")
        if not target_blocker_ok:
            violations.append("target_blocker_increase")
        if not dependency_total_ok:
            violations.append("dependency_total_increase")
        if not critical_ok:
            violations.append("critical_stand_box_increase")
        if not destructive_ok:
            violations.append("destructive_action_increase")
        macro_admissible = not violations
        return {
            "goal_delta": int(goal_delta),
            "goal_progress": bool(goal_progress),
            "slot_cost_regression": float(slot_regression),
            "schedule_barrier_regression": float(barrier_regression),
            "critical_stand_box_delta": int(critical_delta),
            "future_critical_stand_box_delta": int(future_critical_delta),
            "destructive_action_delta": int(destructive_delta),
            "slot_cost_guard_pass": bool(slot_ok),
            "schedule_barrier_guard_pass": bool(barrier_ok),
            "legal_margin_guard_pass": bool(legal_ok),
            "target_blocker_guard_pass": bool(target_blocker_ok),
            "dependency_total_guard_pass": bool(dependency_total_ok),
            "critical_stand_guard_pass": bool(critical_ok),
            "destructive_action_guard_pass": bool(destructive_ok),
            "global_feasibility_pass": bool(macro_admissible),
            "global_feasibility_violations": violations,
            "macro_admissible": bool(macro_admissible),
        }

    def _dependency_bound_score(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        dependency: dict[str, Any],
        *,
        depth: int,
    ) -> tuple[float, dict[str, Any]]:
        score, summary = super()._dependency_bound_score(
            before,
            after,
            dependency,
            depth=depth,
        )
        feasibility = self._dependency_bound_feasibility_summary(before, after, summary)
        slot_excess = max(
            0.0,
            float(feasibility["slot_cost_regression"])
            - self.feasibility_guard_max_slot_cost_regression,
        )
        barrier_excess = max(
            0.0,
            float(feasibility["schedule_barrier_regression"])
            - self.feasibility_guard_max_schedule_barrier_regression,
        )
        penalty = (
            self.feasibility_guard_violation_penalty
            * float(len(feasibility["global_feasibility_violations"]))
            + self.feasibility_guard_slot_regression_penalty * slot_excess
            + self.feasibility_guard_barrier_regression_penalty * barrier_excess
        )
        score = float(score - penalty)
        summary.update(feasibility)
        summary["feasibility_penalty"] = float(penalty)
        summary["score"] = float(score)
        return float(score), summary

    def _dependency_bound_candidate_acceptance(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        dependency: dict[str, Any],
        score_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return self._dependency_bound_feasibility_summary(before, after, score_summary)

    def _recovery_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        result = super()._recovery_macro_from_plan(state, commitment_state, plan)
        guarded_rows = [
            row
            for row in result.get("rows", [])
            if row.get("reason") == self.dependency_bound_macro_reason
        ]
        result["feasibility_guarded_macro_enabled"] = bool(self.recovery_macro_enabled)
        result["feasibility_guarded_macro_triggered"] = bool(guarded_rows)
        if guarded_rows:
            result["reason"] = "feasibility_guarded_dependency_macro"
            result["feasibility_guarded_macro_score"] = float(
                max(float(row.get("dependency_bound_score", 0.0)) for row in guarded_rows)
            )
            result["feasibility_guarded_extra_push_count"] = int(len(guarded_rows))
            result["feasibility_guarded_final_progress"] = guarded_rows[-1].get(
                "dependency_progress",
                {},
            )
        elif result.get("dependency_bound_macro_enabled"):
            result["feasibility_guarded_macro_reason"] = result.get(
                "dependency_bound_macro_reason",
                result.get("reason", "no_feasibility_guarded_suffix"),
            )
        return result


class RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner(
    RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner
):
    """A17 planner with schedule-phase no-progress root guards.

    A16 prevents locally valid dependency macros from damaging global
    feasibility, but the root policy can still shuttle boxes while the schedule
    phase does not change.  A17 adds a consumer-side guard for repeated
    no-progress phase signatures.
    """

    provider_id = "room_schedule_phase_progress_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        phase_progress_no_progress_penalty: float = 16.0,
        phase_progress_stagnant_context_penalty: float = 28.0,
        phase_progress_repeated_signature_penalty: float = 24.0,
        phase_progress_repeated_edge_penalty: float = 18.0,
        phase_progress_reverse_edge_penalty: float = 30.0,
        phase_progress_no_release_penalty: float = 8.0,
        phase_progress_root_guard: bool = True,
        phase_progress_leaf_repeated_penalty: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.phase_progress_no_progress_penalty = float(phase_progress_no_progress_penalty)
        self.phase_progress_stagnant_context_penalty = float(
            phase_progress_stagnant_context_penalty
        )
        self.phase_progress_repeated_signature_penalty = float(
            phase_progress_repeated_signature_penalty
        )
        self.phase_progress_repeated_edge_penalty = float(phase_progress_repeated_edge_penalty)
        self.phase_progress_reverse_edge_penalty = float(phase_progress_reverse_edge_penalty)
        self.phase_progress_no_release_penalty = float(phase_progress_no_release_penalty)
        self.phase_progress_root_guard = bool(phase_progress_root_guard)
        self.phase_progress_leaf_repeated_penalty = float(phase_progress_leaf_repeated_penalty)

    def _phase_progress_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        after = self.cached_schedule_plan_summary(
            candidate.child_state,
            candidate.child_commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        before_slot = self.model.room_slot_progress_summary(state, commitment_state)
        after_slot = self.model.room_slot_progress_summary(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        before_totals = before["totals"]
        after_totals = after["totals"]
        before_dependency_total = int(before_totals["dependency_count"]) + int(
            before_totals["future_dependency_count"]
        )
        after_dependency_total = int(after_totals["dependency_count"]) + int(
            after_totals["future_dependency_count"]
        )
        goal_delta = int(self.model.boxes_on_goals(candidate.child_state)) - int(
            self.model.boxes_on_goals(state)
        )
        slot_delta = float(before_slot["slot_schedule_cost"]) - float(
            after_slot["slot_schedule_cost"]
        )
        barrier_delta = float(
            before_totals["schedule_barrier_cost"] - after_totals["schedule_barrier_cost"]
        )
        dependency_delta = int(before_dependency_total - after_dependency_total)
        target_blocker_delta = int(before_totals["target_blocker_count"]) - int(
            after_totals["target_blocker_count"]
        )
        release_delta = int(after_totals["release_action_count"]) - int(
            before_totals["release_action_count"]
        )
        phase_progress = bool(
            goal_delta > 0
            or slot_delta > 0.5
            or dependency_delta > 0
            or target_blocker_delta > 0
            or release_delta > 0
            or barrier_delta > 4.0
        )
        current_signature = self.schedule_stall_signature(state, commitment_state)
        child_signature = self.schedule_stall_signature(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        current_repeated_signature = current_signature in self.recent_schedule_keys
        child_repeated_signature = child_signature in self.recent_schedule_keys
        source = state.boxes[candidate.action.physical_action.box_index]
        target = add_cells(source, DIRECTIONS[candidate.action.physical_action.direction])
        edge_key = (tuple(source), tuple(target))
        reverse_edge_key = (tuple(target), tuple(source))
        repeated_edge = edge_key in self.recent_moved_edges
        reverse_edge = reverse_edge_key in self.recent_moved_edges
        no_release_context = int(before_totals["release_action_count"]) <= 0
        dependency_active = before_dependency_total > 0
        no_progress = bool(dependency_active and not phase_progress)
        stagnant_context = bool(dependency_active and child_repeated_signature and no_release_context)
        score = -(
            self.phase_progress_no_progress_penalty * float(no_progress)
            + self.phase_progress_stagnant_context_penalty * float(no_progress and stagnant_context)
            + self.phase_progress_repeated_signature_penalty
            * float(no_progress and child_repeated_signature)
            + self.phase_progress_repeated_edge_penalty * float(no_progress and repeated_edge)
            + self.phase_progress_reverse_edge_penalty * float(no_progress and reverse_edge)
            + self.phase_progress_no_release_penalty
            * float(no_progress and no_release_context)
        )
        candidate.room_phase_progress_score = float(score)
        candidate.room_phase_progress_summary = {
            "moved_source": list(source),
            "moved_target": list(target),
            "dependency_active": bool(dependency_active),
            "phase_progress": bool(phase_progress),
            "no_progress": bool(no_progress),
            "stagnant_context": bool(stagnant_context),
            "goal_delta": int(goal_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "schedule_barrier_cost_delta": float(barrier_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_blocker_delta),
            "release_action_count_delta": int(release_delta),
            "before_dependency_total": int(before_dependency_total),
            "after_dependency_total": int(after_dependency_total),
            "before_release_action_count": int(before_totals["release_action_count"]),
            "after_release_action_count": int(after_totals["release_action_count"]),
            "current_repeated_schedule_signature": bool(current_repeated_signature),
            "child_repeated_schedule_signature": bool(child_repeated_signature),
            "repeated_edge": bool(repeated_edge),
            "reverse_recent_edge": bool(reverse_edge),
            "no_release_context": bool(no_release_context),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._phase_progress_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.phase_progress_root_guard:
            return best
        best_phase = best.room_phase_progress_summary
        best_bad = bool(best_phase.get("no_progress")) and (
            bool(best_phase.get("stagnant_context"))
            or bool(best_phase.get("child_repeated_schedule_signature"))
            or bool(best_phase.get("repeated_edge"))
            or bool(best_phase.get("reverse_recent_edge"))
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if bool(edge.room_phase_progress_summary.get("phase_progress"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if (
                    not bool(edge.room_phase_progress_summary.get("reverse_recent_edge"))
                    and not bool(edge.room_phase_progress_summary.get("repeated_edge"))
                    and edge.room_phase_progress_score > best.room_phase_progress_score
                )
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_phase_progress_score > best.room_phase_progress_score
            ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_phase_progress_score,
                    edge.room_schedule_plan_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_phase_progress_score,
                edge.room_schedule_plan_score,
                edge.prior_score,
            ),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        dependency_total = int(totals["dependency_count"]) + int(totals["future_dependency_count"])
        if (
            dependency_total > 0
            and int(totals["release_action_count"]) <= 0
            and self.schedule_stall_signature(state, commitment_state) in self.recent_schedule_keys
        ):
            value -= self.phase_progress_leaf_repeated_penalty
            low = min(self.deadend_value, self.goal_value)
            high = max(self.deadend_value, self.goal_value)
            value = float(max(low, min(high, value)))
        return value


class RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner(
    RoomSchedulePhaseProgressGuardedMacroGFTSDMCTSPlanner
):
    """A18 planner that calibrates no-progress guards with short recovery probes.

    A17 treats immediate schedule-phase no-progress as a strong negative signal.
    A18 keeps that diagnostic signal but allows no-progress setup pushes when a
    bounded probe shows that release/dependency/slot progress remains reachable.
    """

    provider_id = "room_schedule_recoverability_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        recoverability_probe_depth: int = 1,
        recoverability_probe_max_nodes: int = 16,
        recoverability_recoverable_bonus: float = 48.0,
        recoverability_unrecoverable_penalty: float = 56.0,
        recoverability_low_legal_penalty: float = 24.0,
        recoverability_root_guard: bool = True,
        recoverability_leaf_unrecoverable_penalty: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.recoverability_probe_depth = max(int(recoverability_probe_depth), 0)
        self.recoverability_probe_max_nodes = max(int(recoverability_probe_max_nodes), 1)
        self.recoverability_recoverable_bonus = float(recoverability_recoverable_bonus)
        self.recoverability_unrecoverable_penalty = float(recoverability_unrecoverable_penalty)
        self.recoverability_low_legal_penalty = float(recoverability_low_legal_penalty)
        self.recoverability_root_guard = bool(recoverability_root_guard)
        self.recoverability_leaf_unrecoverable_penalty = float(
            recoverability_leaf_unrecoverable_penalty
        )

    def _phase_recoverability_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        totals = schedule["totals"]
        dependency_total = int(totals["dependency_count"]) + int(totals["future_dependency_count"])
        legal = self._nonblocked_legal_push_count(state)
        return {
            "boxes_on_goals": int(self.model.boxes_on_goals(state)),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "dependency_total": int(dependency_total),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "nonblocked_legal_push_count": int(legal),
            "no_legal_pushes": bool(legal <= 0),
            "low_legal_margin": bool(
                dependency_total > 0 and legal < self.room_access_recovery_min_legal_pushes
            ),
        }

    def _phase_recoverability_progress(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        goal_delta = int(after["boxes_on_goals"]) - int(before["boxes_on_goals"])
        slot_delta = float(before["slot_schedule_cost"]) - float(after["slot_schedule_cost"])
        barrier_delta = float(before["schedule_barrier_cost"]) - float(after["schedule_barrier_cost"])
        dependency_delta = int(before["dependency_total"]) - int(after["dependency_total"])
        target_blocker_delta = int(before["target_blocker_count"]) - int(
            after["target_blocker_count"]
        )
        release_delta = int(after["release_action_count"]) - int(before["release_action_count"])
        progress = bool(
            goal_delta > 0
            or slot_delta > 0.5
            or barrier_delta > 4.0
            or dependency_delta > 0
            or target_blocker_delta > 0
            or release_delta > 0
        )
        return {
            "recoverability_progress": bool(progress),
            "goal_delta": int(goal_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "schedule_barrier_cost_delta": float(barrier_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_blocker_delta),
            "release_action_count_delta": int(release_delta),
        }

    def _best_phase_recovery_probe(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        if self.recoverability_probe_depth <= 0:
            return {
                "recoverable": False,
                "best_depth": None,
                "best_score": None,
                "expanded_nodes": 0,
                "best_progress": {},
            }
        best: dict[str, Any] | None = None
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "depth": 0,
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        expanded = 0
        while queue and expanded < self.recoverability_probe_max_nodes:
            node = queue.pop(0)
            if int(node["depth"]) >= self.recoverability_probe_depth:
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                key = self.model.gft_canonical_key_cached(child)
                if key in node["seen"]:
                    continue
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._phase_recoverability_metrics(child, child_commitment)
                progress = self._phase_recoverability_progress(reference_metrics, metrics)
                depth = int(node["depth"]) + 1
                score = (
                    80.0 * float(progress["recoverability_progress"])
                    + 12.0 * float(progress["goal_delta"])
                    + 8.0 * float(progress["dependency_total_delta"])
                    + 8.0 * float(progress["target_blocker_count_delta"])
                    + 6.0 * float(progress["release_action_count_delta"])
                    + 2.0 * float(progress["slot_schedule_cost_delta"])
                    + 1.0 * float(progress["schedule_barrier_cost_delta"])
                    + 1.5 * float(metrics["nonblocked_legal_push_count"])
                    - 8.0 * float(metrics["low_legal_margin"])
                    - 50.0 * float(metrics["no_legal_pushes"])
                    - 0.25 * float(depth)
                )
                row = {
                    "recoverable": bool(progress["recoverability_progress"]),
                    "best_depth": int(depth),
                    "best_score": float(score),
                    "best_progress": progress,
                    "best_metrics": metrics,
                    "best_action": self._physical_action_dict(node["state"], action),
                }
                if best is None or (
                    int(row["recoverable"]),
                    row["best_score"],
                    -row["best_depth"],
                ) > (
                    int(best["recoverable"]),
                    best["best_score"],
                    -best["best_depth"],
                ):
                    best = row
                if depth < self.recoverability_probe_depth and not metrics["no_legal_pushes"]:
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "depth": depth,
                            "seen": set(node["seen"]) | {key},
                        }
                    )
                expanded += 1
                if expanded >= self.recoverability_probe_max_nodes:
                    break
        if best is None:
            best = {
                "recoverable": False,
                "best_depth": None,
                "best_score": None,
                "best_progress": {},
                "best_metrics": {},
                "best_action": None,
            }
        best["expanded_nodes"] = int(expanded)
        return best

    def _recoverability_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        phase = candidate.room_phase_progress_summary
        if not bool(phase.get("dependency_active")) or bool(phase.get("phase_progress")):
            candidate.room_recoverability_score = 0.0
            candidate.room_recoverability_summary = {
                "probe_skipped": True,
                "skip_reason": "phase_progress_or_no_dependency",
                "recoverable_no_progress": False,
                "unrecoverable_no_progress": False,
                "score": 0.0,
            }
            return 0.0
        reference = self._phase_recoverability_metrics(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        probe = self._best_phase_recovery_probe(
            candidate.child_state,
            candidate.child_commitment_state,
            reference,
        )
        legal_ok = (
            not bool(reference["no_legal_pushes"])
            and int(reference["nonblocked_legal_push_count"]) >= self.room_access_recovery_min_legal_pushes
        )
        recoverable = bool(probe.get("recoverable")) and bool(legal_ok)
        unrecoverable = bool(phase.get("no_progress")) and not recoverable
        score = (
            self.recoverability_recoverable_bonus * float(recoverable)
            - self.recoverability_unrecoverable_penalty * float(unrecoverable)
            - self.recoverability_low_legal_penalty * float(not legal_ok)
        )
        candidate.room_recoverability_score = float(score)
        candidate.room_recoverability_summary = {
            "probe_skipped": False,
            "recoverable_no_progress": bool(recoverable),
            "unrecoverable_no_progress": bool(unrecoverable),
            "legal_margin_ok": bool(legal_ok),
            "reference_metrics": reference,
            "probe": probe,
            "phase_no_progress": bool(phase.get("no_progress")),
            "phase_stagnant_context": bool(phase.get("stagnant_context")),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._recoverability_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner.choose_root_edge(self, root)
        if not self.recoverability_root_guard:
            return best
        best_phase = best.room_phase_progress_summary
        best_recovery = best.room_recoverability_summary
        best_bad = bool(best_recovery.get("unrecoverable_no_progress")) and (
            bool(best_phase.get("stagnant_context"))
            or bool(best_phase.get("child_repeated_schedule_signature"))
            or bool(best_phase.get("repeated_edge"))
            or bool(best_phase.get("reverse_recent_edge"))
            or bool(best_recovery.get("legal_margin_ok")) is False
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if bool(edge.room_phase_progress_summary.get("phase_progress"))
            or bool(edge.room_recoverability_summary.get("recoverable_no_progress"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if not bool(edge.room_recoverability_summary.get("unrecoverable_no_progress"))
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_recoverability_score > best.room_recoverability_score
            ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_recoverability_score,
                    edge.room_phase_progress_score,
                    edge.room_schedule_plan_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_recoverability_score,
                edge.room_phase_progress_score,
                edge.room_schedule_plan_score,
                edge.prior_score,
            ),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        dependency_total = int(totals["dependency_count"]) + int(totals["future_dependency_count"])
        if dependency_total > 0 and int(totals["release_action_count"]) <= 0:
            legal = self._nonblocked_legal_push_count(state)
            if legal < self.room_access_recovery_min_legal_pushes:
                value -= self.recoverability_leaf_unrecoverable_penalty
                low = min(self.deadend_value, self.goal_value)
                high = max(self.deadend_value, self.goal_value)
                value = float(max(low, min(high, value)))
        return value


class RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner(
    RoomScheduleRecoverabilityGuardedMacroGFTSDMCTSPlanner
):
    """A19 planner with stricter survivable recoverability criteria.

    A18 can treat short slot-cost recovery as sufficient even when no
    dependency, target-blocker, or release-action progress appears.  A19 keeps
    the same root interface but only grants recoverable status to structural
    progress; slot-only recovery is kept as a diagnostic and penalty term.
    """

    provider_id = "room_schedule_survivable_recoverability_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        recoverability_probe_depth: int = 2,
        recoverability_probe_max_nodes: int = 16,
        survivable_recoverability_min_depth: int = 2,
        survivable_recoverability_min_legal_pushes: int = 3,
        survivable_slot_only_penalty: float = 24.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            recoverability_probe_depth=recoverability_probe_depth,
            recoverability_probe_max_nodes=recoverability_probe_max_nodes,
            **kwargs,
        )
        self.survivable_recoverability_min_depth = max(
            int(survivable_recoverability_min_depth), 1
        )
        self.survivable_recoverability_min_legal_pushes = max(
            int(survivable_recoverability_min_legal_pushes),
            int(self.room_access_recovery_min_legal_pushes),
        )
        self.survivable_slot_only_penalty = float(survivable_slot_only_penalty)

    def _survivable_recoverability_progress(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        depth: int,
    ) -> dict[str, Any]:
        progress = self._phase_recoverability_progress(before, after)
        structural_progress = bool(
            int(progress["goal_delta"]) > 0
            or int(progress["dependency_total_delta"]) > 0
            or int(progress["target_blocker_count_delta"]) > 0
            or int(progress["release_action_count_delta"]) > 0
        )
        slot_only_progress = bool(
            not structural_progress
            and (
                float(progress["slot_schedule_cost_delta"]) > 0.5
                or float(progress["schedule_barrier_cost_delta"]) > 4.0
            )
        )
        legal_survivable = bool(
            not after["no_legal_pushes"]
            and int(after["nonblocked_legal_push_count"])
            >= self.survivable_recoverability_min_legal_pushes
            and not after["low_legal_margin"]
        )
        no_new_dependency_pressure = bool(
            int(after["dependency_total"]) <= int(before["dependency_total"])
            and int(after["target_blocker_count"]) <= int(before["target_blocker_count"])
        )
        slot_only_survivable = bool(
            slot_only_progress
            and int(depth) >= self.survivable_recoverability_min_depth
            and legal_survivable
            and no_new_dependency_pressure
        )
        recoverable = bool(structural_progress and legal_survivable)
        return {
            **progress,
            "recoverability_progress": bool(recoverable),
            "structural_progress": bool(structural_progress),
            "slot_only_progress": bool(slot_only_progress),
            "slot_only_survivable": bool(slot_only_survivable),
            "legal_survivable": bool(legal_survivable),
            "no_new_dependency_pressure": bool(no_new_dependency_pressure),
            "survivable_depth_ok": bool(int(depth) >= self.survivable_recoverability_min_depth),
        }

    def _best_phase_recovery_probe(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        if self.recoverability_probe_depth <= 0:
            return {
                "recoverable": False,
                "best_depth": None,
                "best_score": None,
                "expanded_nodes": 0,
                "best_progress": {},
            }
        best: dict[str, Any] | None = None
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "depth": 0,
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        expanded = 0
        while queue and expanded < self.recoverability_probe_max_nodes:
            node = queue.pop(0)
            if int(node["depth"]) >= self.recoverability_probe_depth:
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                key = self.model.gft_canonical_key_cached(child)
                if key in node["seen"]:
                    continue
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._phase_recoverability_metrics(child, child_commitment)
                depth = int(node["depth"]) + 1
                progress = self._survivable_recoverability_progress(
                    reference_metrics,
                    metrics,
                    depth=depth,
                )
                score = (
                    100.0 * float(progress["recoverability_progress"])
                    + 34.0 * float(progress["structural_progress"])
                    + 12.0 * float(progress["goal_delta"])
                    + 10.0 * float(progress["dependency_total_delta"])
                    + 10.0 * float(progress["target_blocker_count_delta"])
                    + 8.0 * float(progress["release_action_count_delta"])
                    + 1.5 * float(progress["slot_schedule_cost_delta"])
                    + 0.75 * float(progress["schedule_barrier_cost_delta"])
                    + 1.0 * float(metrics["nonblocked_legal_push_count"])
                    - 12.0 * float(not progress["legal_survivable"])
                    - 24.0 * float(progress["slot_only_progress"])
                    - 50.0 * float(metrics["no_legal_pushes"])
                    - 0.5 * float(depth)
                )
                row = {
                    "recoverable": bool(progress["recoverability_progress"]),
                    "best_depth": int(depth),
                    "best_score": float(score),
                    "best_progress": progress,
                    "best_metrics": metrics,
                    "best_action": self._physical_action_dict(node["state"], action),
                }
                if best is None or (
                    int(row["recoverable"]),
                    bool(progress["structural_progress"]),
                    row["best_score"],
                    -row["best_depth"],
                ) > (
                    int(best["recoverable"]),
                    bool(best.get("best_progress", {}).get("structural_progress")),
                    best["best_score"],
                    -best["best_depth"],
                ):
                    best = row
                if depth < self.recoverability_probe_depth and not metrics["no_legal_pushes"]:
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "depth": depth,
                            "seen": set(node["seen"]) | {key},
                        }
                    )
                expanded += 1
                if expanded >= self.recoverability_probe_max_nodes:
                    break
        if best is None:
            best = {
                "recoverable": False,
                "best_depth": None,
                "best_score": None,
                "best_progress": {},
                "best_metrics": {},
                "best_action": None,
            }
        best["expanded_nodes"] = int(expanded)
        return best

    def _recoverability_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        phase = candidate.room_phase_progress_summary
        if not bool(phase.get("dependency_active")) or bool(phase.get("phase_progress")):
            candidate.room_recoverability_score = 0.0
            candidate.room_recoverability_summary = {
                "probe_skipped": True,
                "skip_reason": "phase_progress_or_no_dependency",
                "recoverable_no_progress": False,
                "unrecoverable_no_progress": False,
                "survivable_slot_only_rejected": False,
                "score": 0.0,
            }
            return 0.0
        reference = self._phase_recoverability_metrics(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        probe = self._best_phase_recovery_probe(
            candidate.child_state,
            candidate.child_commitment_state,
            reference,
        )
        progress = probe.get("best_progress", {})
        legal_ok = (
            not bool(reference["no_legal_pushes"])
            and int(reference["nonblocked_legal_push_count"])
            >= self.room_access_recovery_min_legal_pushes
        )
        recoverable = bool(probe.get("recoverable")) and bool(legal_ok)
        unrecoverable = bool(phase.get("no_progress")) and not recoverable
        slot_only_rejected = bool(
            progress.get("slot_only_progress") and not progress.get("slot_only_survivable")
        )
        score = (
            self.recoverability_recoverable_bonus * float(recoverable)
            - self.recoverability_unrecoverable_penalty * float(unrecoverable)
            - self.recoverability_low_legal_penalty * float(not legal_ok)
            - self.survivable_slot_only_penalty * float(slot_only_rejected)
        )
        candidate.room_recoverability_score = float(score)
        candidate.room_recoverability_summary = {
            "probe_skipped": False,
            "recoverable_no_progress": bool(recoverable),
            "unrecoverable_no_progress": bool(unrecoverable),
            "survivable_slot_only_rejected": bool(slot_only_rejected),
            "legal_margin_ok": bool(legal_ok),
            "reference_metrics": reference,
            "probe": probe,
            "phase_no_progress": bool(phase.get("no_progress")),
            "phase_stagnant_context": bool(phase.get("stagnant_context")),
            "score": float(score),
        }
        return float(score)

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner.choose_root_edge(self, root)
        if not self.recoverability_root_guard:
            return best
        best_phase = best.room_phase_progress_summary
        best_recovery = best.room_recoverability_summary
        best_bad = bool(best_recovery.get("unrecoverable_no_progress")) and (
            bool(best_phase.get("no_progress"))
            or bool(best_phase.get("no_release_context"))
            or bool(best_phase.get("stagnant_context"))
            or bool(best_phase.get("child_repeated_schedule_signature"))
            or bool(best_phase.get("repeated_edge"))
            or bool(best_phase.get("reverse_recent_edge"))
            or bool(best_recovery.get("survivable_slot_only_rejected"))
            or bool(best_recovery.get("legal_margin_ok")) is False
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if bool(edge.room_phase_progress_summary.get("phase_progress"))
            or bool(edge.room_recoverability_summary.get("recoverable_no_progress"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if not bool(edge.room_recoverability_summary.get("unrecoverable_no_progress"))
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_recoverability_score > best.room_recoverability_score
            ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_recoverability_score,
                    edge.room_phase_progress_score,
                    edge.room_schedule_plan_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_recoverability_score,
                edge.room_phase_progress_score,
                edge.room_schedule_plan_score,
                edge.prior_score,
            ),
        )


class RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner(
    RoomScheduleSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner
):
    """A20 planner that confines survivable recoverability probes to roots.

    A19 applies the stricter probe everywhere MCTS initialises a node, which is
    expensive on hard instances.  A20 preserves the same root-choice semantics
    but skips recoverability probes below the current planning root and caches
    probe results for root candidates within one closed-loop decision.
    """

    provider_id = "room_schedule_root_cached_survivable_recoverability_guarded_macro_gft_sdmcts_v1"

    def __init__(self, *args, root_recoverability_only: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_recoverability_only = bool(root_recoverability_only)
        self._recoverability_root_key: tuple | None = None
        self._root_recoverability_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}

    def _recoverability_skip_summary(self, reason: str) -> dict[str, Any]:
        return {
            "probe_skipped": True,
            "skip_reason": str(reason),
            "recoverable_no_progress": False,
            "unrecoverable_no_progress": False,
            "survivable_slot_only_rejected": False,
            "legal_margin_ok": True,
            "root_recoverability_only": bool(self.root_recoverability_only),
            "score": 0.0,
        }

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(state, commitment_state)
        previous_root = self._recoverability_root_key
        previous_cache = self._root_recoverability_cache
        self._recoverability_root_key = self.state_key(state, commitment_state)
        self._root_recoverability_cache = {}
        try:
            return super().plan(state, commitment_state)
        finally:
            self._recoverability_root_key = previous_root
            self._root_recoverability_cache = previous_cache

    def _recoverability_cache_key(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> tuple:
        return (
            self.state_key(state, commitment_state),
            candidate.action.key,
            self.model.gft_canonical_key_cached(candidate.child_state),
            candidate.child_commitment_state.key,
            int(self.recoverability_probe_depth),
            int(self.recoverability_probe_max_nodes),
            int(self.survivable_recoverability_min_legal_pushes),
        )

    def _recoverability_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        if self.root_recoverability_only:
            current_key = self.state_key(state, commitment_state)
            if self._recoverability_root_key is None:
                candidate.room_recoverability_score = 0.0
                candidate.room_recoverability_summary = self._recoverability_skip_summary(
                    "no_active_root"
                )
                return 0.0
            if current_key != self._recoverability_root_key:
                candidate.room_recoverability_score = 0.0
                candidate.room_recoverability_summary = self._recoverability_skip_summary(
                    "non_root_node"
                )
                return 0.0

        cache_key = self._recoverability_cache_key(state, commitment_state, candidate)
        cached = self._root_recoverability_cache.get(cache_key)
        if cached is not None:
            score, summary = cached
            candidate.room_recoverability_score = float(score)
            candidate.room_recoverability_summary = dict(summary)
            candidate.room_recoverability_summary["cache_hit"] = True
            return float(score)

        score = super()._recoverability_score(state, commitment_state, candidate)
        summary = dict(candidate.room_recoverability_summary)
        summary["cache_hit"] = False
        summary["root_recoverability_only"] = bool(self.root_recoverability_only)
        candidate.room_recoverability_summary = summary
        self._root_recoverability_cache[cache_key] = (float(score), dict(summary))
        return float(score)


class RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner(
    RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner
):
    """A21 planner that guards unreleasable critical stand blockers.

    A20 reaches states where the next slot's stand cell is occupied by a box
    that has no legal release push, often a box already sitting on a goal.  A21
    keeps A20's root-only recoverability boundary and adds a consumer-side
    stand-lock diagnostic, prior, root guard, and leaf penalty.
    """

    provider_id = "room_schedule_stand_lock_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        stand_lock_increase_penalty: float = 90.0,
        stand_lock_absolute_penalty: float = 22.0,
        stand_lock_goal_reassignment_penalty: float = 26.0,
        stand_lock_release_option_bonus: float = 18.0,
        stand_lock_root_guard: bool = True,
        stand_lock_leaf_penalty: float = 6.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stand_lock_increase_penalty = float(stand_lock_increase_penalty)
        self.stand_lock_absolute_penalty = float(stand_lock_absolute_penalty)
        self.stand_lock_goal_reassignment_penalty = float(
            stand_lock_goal_reassignment_penalty
        )
        self.stand_lock_release_option_bonus = float(stand_lock_release_option_bonus)
        self.stand_lock_root_guard = bool(stand_lock_root_guard)
        self.stand_lock_leaf_penalty = float(stand_lock_leaf_penalty)

    def _stand_lock_release_options(
        self,
        state: SokobanState,
        stand_cell: Cell,
        protected_cells: set[Cell],
    ) -> list[dict[str, Any]]:
        if stand_cell not in state.boxes:
            return []
        options: list[dict[str, Any]] = []
        box_index = state.boxes.index(stand_cell)
        for direction_index, direction in enumerate(DIRECTIONS):
            target = add_cells(stand_cell, direction)
            player_stand = (stand_cell[0] - direction[0], stand_cell[1] - direction[1])
            if target in protected_cells:
                continue
            action = SokobanPushAction(box_index, direction_index)
            if action not in self.model.legal_flows(state):
                continue
            child = self.model.step_grid_flow(state, action)
            if self.is_blocked(child):
                continue
            options.append(
                {
                    "box_index": int(box_index),
                    "direction": int(direction_index),
                    "direction_name": ACTION_NAMES[direction_index],
                    "source": list(stand_cell),
                    "target": list(target),
                    "player_stand": list(player_stand),
                }
            )
        return options

    def _stand_lock_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        key = self._summary_cache_key(
            "stand_lock_metrics",
            state,
            commitment_state,
            self.room_schedule_lookahead_slots,
        )
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached

        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        boxes = set(state.boxes)
        protected_cells = {tuple(cell) for cell in schedule.get("critical_stand_cells", [])}
        protected_cells.update(tuple(cell) for cell in schedule.get("future_critical_stand_cells", []))
        locked_rows: list[dict[str, Any]] = []
        occupied_rows: list[dict[str, Any]] = []
        assigned_goal_rows: list[dict[str, Any]] = []
        release_option_count = 0

        for row in schedule.get("dependencies", []):
            kinds = set(row.get("dependency_kinds", []))
            stand_cell_raw = row.get("stand_cell")
            if stand_cell_raw is None:
                continue
            stand_cell = tuple(stand_cell_raw)
            assigned_raw = row.get("assigned_box")
            assigned_box = None if assigned_raw is None else tuple(assigned_raw)
            slot_raw = row.get("slot_cell")
            slot_cell = None if slot_raw is None else tuple(slot_raw)
            if (
                assigned_box is not None
                and slot_cell is not None
                and assigned_box in self.model.level.goals
                and assigned_box != slot_cell
            ):
                assigned_goal_rows.append(
                    {
                        "dependency": row,
                        "assigned_box": list(assigned_box),
                        "slot_cell": list(slot_cell),
                    }
                )
            if "stand_cell_occupied_by_box" not in kinds or stand_cell not in boxes:
                continue
            release_options = self._stand_lock_release_options(
                state,
                stand_cell,
                protected_cells,
            )
            release_option_count += len(release_options)
            occupied = {
                "dependency": row,
                "stand_cell": list(stand_cell),
                "blocker_on_goal": bool(stand_cell in self.model.level.goals),
                "release_options": release_options,
                "release_option_count": int(len(release_options)),
            }
            occupied_rows.append(occupied)
            if not release_options:
                locked_rows.append(occupied)

        result = {
            "dependency_total": int(totals["dependency_count"]) + int(totals["future_dependency_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "occupied_critical_stand_count": int(len(occupied_rows)),
            "locked_stand_blocker_count": int(len(locked_rows)),
            "goal_locked_stand_blocker_count": int(
                sum(bool(row["blocker_on_goal"]) for row in locked_rows)
            ),
            "assigned_goal_reassignment_count": int(len(assigned_goal_rows)),
            "release_option_count": int(release_option_count),
            "occupied_stand_rows": occupied_rows,
            "locked_stand_rows": locked_rows,
            "assigned_goal_reassignment_rows": assigned_goal_rows,
            "schedule_totals": totals,
        }
        self._summary_cache[key] = result
        return result

    def _stand_lock_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before = self._stand_lock_metrics(state, commitment_state)
        after = self._stand_lock_metrics(candidate.child_state, candidate.child_commitment_state)
        locked_delta = int(after["locked_stand_blocker_count"]) - int(
            before["locked_stand_blocker_count"]
        )
        occupied_delta = int(after["occupied_critical_stand_count"]) - int(
            before["occupied_critical_stand_count"]
        )
        assigned_goal_delta = int(after["assigned_goal_reassignment_count"]) - int(
            before["assigned_goal_reassignment_count"]
        )
        release_option_delta = int(after["release_option_count"]) - int(
            before["release_option_count"]
        )
        score = (
            -self.stand_lock_increase_penalty * float(max(0, locked_delta))
            + self.stand_lock_increase_penalty * 0.5 * float(max(0, -locked_delta))
            - self.stand_lock_absolute_penalty * float(after["locked_stand_blocker_count"])
            - 0.5 * self.stand_lock_absolute_penalty * float(max(0, occupied_delta))
            - self.stand_lock_goal_reassignment_penalty
            * float(max(0, assigned_goal_delta) + after["assigned_goal_reassignment_count"])
            + self.stand_lock_release_option_bonus * float(max(0, release_option_delta))
        )
        candidate.room_stand_lock_score = float(score)
        candidate.room_stand_lock_summary = {
            "before": before,
            "after": after,
            "locked_stand_blocker_delta": int(locked_delta),
            "occupied_critical_stand_delta": int(occupied_delta),
            "assigned_goal_reassignment_delta": int(assigned_goal_delta),
            "release_option_delta": int(release_option_delta),
            "stand_lock_risk": bool(
                int(after["locked_stand_blocker_count"]) > 0
                or int(after["assigned_goal_reassignment_count"]) > 0
            ),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._stand_lock_score(state, commitment_state, candidate))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.stand_lock_root_guard:
            return best
        best_after = best.room_stand_lock_summary.get("after", {})
        best_locked = int(best_after.get("locked_stand_blocker_count", 0))
        best_assigned_goal = int(best_after.get("assigned_goal_reassignment_count", 0))
        if best_locked <= 0 and best_assigned_goal <= 0:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if int(edge.room_stand_lock_summary.get("after", {}).get("locked_stand_blocker_count", 0))
            < best_locked
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if int(
                    edge.room_stand_lock_summary.get("after", {}).get(
                        "assigned_goal_reassignment_count",
                        0,
                    )
                )
                < best_assigned_goal
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_stand_lock_score > best.room_stand_lock_score
            ]
        if not alternatives:
            return best
        if self.select_by == "q":
            return max(
                alternatives,
                key=lambda edge: (
                    edge.mean_value,
                    edge.visits,
                    edge.room_stand_lock_score,
                    edge.room_recoverability_score,
                    edge.room_phase_progress_score,
                    edge.prior_score,
                ),
            )
        return max(
            alternatives,
            key=lambda edge: (
                edge.visits,
                edge.mean_value,
                edge.room_stand_lock_score,
                edge.room_recoverability_score,
                edge.room_phase_progress_score,
                edge.prior_score,
            ),
        )

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = super().leaf_value(state, commitment_state)
        metrics = self._stand_lock_metrics(state, commitment_state)
        penalty_units = int(metrics["locked_stand_blocker_count"]) + int(
            metrics["assigned_goal_reassignment_count"]
        )
        if penalty_units > 0:
            value -= self.stand_lock_leaf_penalty * float(penalty_units)
            low = min(self.deadend_value, self.goal_value)
            high = max(self.deadend_value, self.goal_value)
            value = float(max(low, min(high, value)))
        return value


class RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner(
    RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner
):
    """A22 planner with milestone-gated release/action macro generation.

    A21 shows that stand-lock is a real signal but too strong as a global
    guard.  A22 treats it as a phase-local release objective: before the
    release milestone it preserves A20's root behavior; after the milestone it
    binds a short macro to a concrete ``stand_cell_occupied_by_box`` dependency.
    """

    provider_id = "room_schedule_phased_release_macro_gft_sdmcts_v1"
    dependency_bound_macro_reason = "phased_release_macro_push"

    def __init__(
        self,
        *args,
        stage_release_min_goals: int = 2,
        stage_stand_lock_active_stages: Iterable[str] = ("release", "recovery"),
        stage_stand_lock_score_scale: float = 0.35,
        stage_release_stand_occupied_resolution_weight: float = 85.0,
        stage_release_goal_nonregression: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("stand_lock_root_guard", False)
        super().__init__(*args, **kwargs)
        self.stage_release_min_goals = max(int(stage_release_min_goals), 0)
        self.stage_stand_lock_active_stages = {
            str(item) for item in stage_stand_lock_active_stages
        }
        self.stage_stand_lock_score_scale = float(stage_stand_lock_score_scale)
        self.stage_release_stand_occupied_resolution_weight = float(
            stage_release_stand_occupied_resolution_weight
        )
        self.stage_release_goal_nonregression = bool(stage_release_goal_nonregression)

    @staticmethod
    def _dependency_relevant_kinds(row: dict[str, Any]) -> set[str]:
        kinds = set(row.get("dependency_kinds", []))
        return {
            kind
            for kind in kinds
            if kind
            in {
                "stand_cell_occupied_by_box",
                "stand_cell_unreachable_by_player",
                "target_occupied_by_box",
            }
        }

    def _decision_stage_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        key = self._summary_cache_key(
            "decision_stage",
            state,
            commitment_state,
            self.room_schedule_lookahead_slots,
            self.stage_release_min_goals,
        )
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached

        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        stand_lock = self._stand_lock_metrics(state, commitment_state)
        totals = schedule["totals"]
        boxes_on_goals = int(self.model.boxes_on_goals(state))
        dependency_total = int(totals["dependency_count"]) + int(
            totals["future_dependency_count"]
        )
        legal = self._nonblocked_legal_push_count(state)
        release_ready = bool(
            boxes_on_goals >= self.stage_release_min_goals
            and dependency_total > 0
            and (
                int(stand_lock["occupied_critical_stand_count"]) > 0
                or int(totals["target_blocker_count"]) > 0
                or int(totals["release_action_count"]) <= 0
            )
        )
        if self.model.is_solved(state):
            stage = "solved"
        elif dependency_total > 0 and legal < self.room_access_recovery_min_legal_pushes:
            stage = "recovery"
        elif release_ready:
            stage = "release"
        elif float(slot["slot_schedule_cost"]) > 0.0:
            stage = "fill"
        else:
            stage = "prepare"
        result = {
            "stage": stage,
            "release_ready": bool(release_ready),
            "stage_release_min_goals": int(self.stage_release_min_goals),
            "boxes_on_goals": int(boxes_on_goals),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "dependency_total": int(dependency_total),
            "dependency_count": int(totals["dependency_count"]),
            "future_dependency_count": int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "occupied_critical_stand_count": int(
                stand_lock["occupied_critical_stand_count"]
            ),
            "locked_stand_blocker_count": int(
                stand_lock["locked_stand_blocker_count"]
            ),
            "goal_locked_stand_blocker_count": int(
                stand_lock["goal_locked_stand_blocker_count"]
            ),
            "nonblocked_legal_push_count": int(legal),
        }
        self._summary_cache[key] = result
        return result

    def _stand_lock_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        before_stage = self._decision_stage_summary(state, commitment_state)
        raw_score = RoomScheduleStandLockGuardedMacroGFTSDMCTSPlanner._stand_lock_score(
            self,
            state,
            commitment_state,
            candidate,
        )
        after_stage = self._decision_stage_summary(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        stage_active = str(before_stage["stage"]) in self.stage_stand_lock_active_stages
        score = float(raw_score) * self.stage_stand_lock_score_scale if stage_active else 0.0
        candidate.room_stand_lock_score = float(score)
        candidate.room_stand_lock_summary = {
            **candidate.room_stand_lock_summary,
            "raw_score": float(raw_score),
            "stage_active": bool(stage_active),
            "stage_scaled_score": float(score),
        }
        candidate.room_stage_score = 0.0
        candidate.room_stage_summary = {
            "before": before_stage,
            "after": after_stage,
            "stage_transition": [
                str(before_stage["stage"]),
                str(after_stage["stage"]),
            ],
            "score": 0.0,
        }
        return float(score)

    def _select_dependency_bound_row(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any] | None:
        stage = self._decision_stage_summary(state, commitment_state)
        if str(stage["stage"]) not in {"release", "recovery"}:
            return self._select_pre_release_dependency_bound_row(state, commitment_state)
        if str(stage["stage"]) == "recovery" and int(stage["boxes_on_goals"]) < self.stage_release_min_goals:
            return self._select_pre_release_dependency_bound_row(state, commitment_state)

        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        protected_cells = {tuple(cell) for cell in schedule.get("critical_stand_cells", [])}
        protected_cells.update(
            tuple(cell) for cell in schedule.get("future_critical_stand_cells", [])
        )
        boxes = set(state.boxes)
        candidates = []
        for row in schedule.get("dependencies", []):
            kinds = set(row.get("dependency_kinds", []))
            stand_raw = row.get("stand_cell")
            if "stand_cell_occupied_by_box" not in kinds or stand_raw is None:
                continue
            stand_cell = tuple(stand_raw)
            if stand_cell not in boxes:
                continue
            release_options = self._stand_lock_release_options(
                state,
                stand_cell,
                protected_cells,
            )
            row_dict = self._dependency_to_dict(row)
            row_dict["selected_dependency_kinds"] = sorted(
                self._dependency_relevant_kinds(row)
            )
            row_dict["stage_summary"] = stage
            row_dict["stand_blocker_on_goal"] = bool(stand_cell in self.model.level.goals)
            row_dict["stand_release_options"] = release_options
            row_dict["stand_release_option_count"] = int(len(release_options))
            candidates.append(row_dict)
        if not candidates:
            return None

        def priority(row: dict[str, Any]) -> tuple:
            return (
                int(not bool(row.get("future", False))),
                int(row.get("stand_release_option_count", 0) > 0),
                int(not bool(row.get("stand_blocker_on_goal"))),
                float(row.get("candidate_barrier_cost", 0.0)),
                -int(row.get("horizon_index", 0)),
            )

        return max(candidates, key=priority)

    @staticmethod
    def _pre_release_dependency_kinds(row: dict[str, Any]) -> set[str]:
        kinds = set(row.get("dependency_kinds", []))
        return {
            kind
            for kind in kinds
            if kind in {"stand_cell_unreachable_by_player", "target_occupied_by_box"}
        }

    def _select_pre_release_dependency_bound_row(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any] | None:
        stage = self._decision_stage_summary(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        candidates = [
            row
            for row in schedule.get("dependencies", [])
            if self._pre_release_dependency_kinds(row)
        ]
        if not candidates:
            return None

        def priority(row: dict[str, Any]) -> tuple:
            kinds = self._pre_release_dependency_kinds(row)
            return (
                int(not bool(row.get("future", False))),
                int("target_occupied_by_box" in kinds),
                int("stand_cell_unreachable_by_player" in kinds),
                float(row.get("candidate_barrier_cost", 0.0)),
                -int(row.get("horizon_index", 0)),
            )

        selected = max(candidates, key=priority)
        row_dict = self._dependency_to_dict(selected)
        row_dict["selected_dependency_kinds"] = sorted(
            self._pre_release_dependency_kinds(selected)
        )
        row_dict["stage_summary"] = stage
        row_dict["phase_local_dependency_mode"] = "pre_release_a20_compatible"
        return row_dict

    def _dependency_bound_score(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        dependency: dict[str, Any],
        *,
        depth: int,
    ) -> tuple[float, dict[str, Any]]:
        score, summary = RoomScheduleDependencyBoundMacroGFTSDMCTSPlanner._dependency_bound_score(
            self,
            before,
            after,
            dependency,
            depth=depth,
        )
        selected_kinds = set(dependency.get("selected_dependency_kinds", []))
        stand_occupied_resolved = bool(
            "stand_cell_occupied_by_box" in selected_kinds
            and bool(before.get("stand_cell_occupied_by_box"))
            and not bool(after.get("stand_cell_occupied_by_box"))
        )
        unresolved_stand_occupied = bool(
            "stand_cell_occupied_by_box" in selected_kinds
            and bool(after.get("stand_cell_occupied_by_box"))
        )
        if stand_occupied_resolved:
            score += self.stage_release_stand_occupied_resolution_weight
            summary["dependency_progress"] = True
        if unresolved_stand_occupied:
            score -= self.dependency_bound_macro_unresolved_penalty
        summary.update(
            {
                "stand_occupied_resolved": bool(stand_occupied_resolved),
                "unresolved_stand_occupied": bool(unresolved_stand_occupied),
            }
        )
        feasibility = self._dependency_bound_feasibility_summary(before, after, summary)
        slot_excess = max(
            0.0,
            float(feasibility["slot_cost_regression"])
            - self.feasibility_guard_max_slot_cost_regression,
        )
        barrier_excess = max(
            0.0,
            float(feasibility["schedule_barrier_regression"])
            - self.feasibility_guard_max_schedule_barrier_regression,
        )
        penalty = (
            self.feasibility_guard_violation_penalty
            * float(len(feasibility["global_feasibility_violations"]))
            + self.feasibility_guard_slot_regression_penalty * slot_excess
            + self.feasibility_guard_barrier_regression_penalty * barrier_excess
        )
        score = float(score - penalty)
        summary.update(feasibility)
        summary["feasibility_penalty"] = float(penalty)
        summary["score"] = float(score)
        return float(score), summary

    def _dependency_bound_feasibility_summary(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        score_summary: dict[str, Any],
    ) -> dict[str, Any]:
        result = RoomScheduleFeasibilityGuardedMacroGFTSDMCTSPlanner._dependency_bound_feasibility_summary(
            self,
            before,
            after,
            score_summary,
        )
        if self.stage_release_goal_nonregression and int(result["goal_delta"]) < 0:
            violations = list(result["global_feasibility_violations"])
            if "goal_regression" not in violations:
                violations.append("goal_regression")
            result["goal_nonregression_guard_pass"] = False
            result["global_feasibility_violations"] = violations
            result["global_feasibility_pass"] = False
            result["macro_admissible"] = False
        else:
            result["goal_nonregression_guard_pass"] = True
        return result

    def leaf_value(self, state: SokobanState, commitment_state: RoomCommitmentState) -> float:
        value = RoomScheduleRootCachedSurvivableRecoverabilityGuardedMacroGFTSDMCTSPlanner.leaf_value(
            self,
            state,
            commitment_state,
        )
        stage = self._decision_stage_summary(state, commitment_state)
        if str(stage["stage"]) in self.stage_stand_lock_active_stages:
            metrics = self._stand_lock_metrics(state, commitment_state)
            penalty_units = int(metrics["locked_stand_blocker_count"]) + int(
                metrics["assigned_goal_reassignment_count"]
            )
            if penalty_units > 0:
                value -= (
                    self.stage_stand_lock_score_scale
                    * self.stand_lock_leaf_penalty
                    * float(penalty_units)
                )
                low = min(self.deadend_value, self.goal_value)
                high = max(self.deadend_value, self.goal_value)
                value = float(max(low, min(high, value)))
        return value

    def _recovery_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        stage_before = self._decision_stage_summary(state, commitment_state)
        result = super()._recovery_macro_from_plan(state, commitment_state, plan)
        release_rows = [
            row
            for row in result.get("rows", [])
            if row.get("reason") == self.dependency_bound_macro_reason
            and row.get("bound_dependency", {}).get("phase_local_dependency_mode")
            != "pre_release_a20_compatible"
        ]
        pre_release_rows = [
            row
            for row in result.get("rows", [])
            if row.get("reason") == self.dependency_bound_macro_reason
            and row.get("bound_dependency", {}).get("phase_local_dependency_mode")
            == "pre_release_a20_compatible"
        ]
        result["phased_release_macro_enabled"] = bool(self.recovery_macro_enabled)
        result["phased_release_stage_before"] = stage_before
        result["phased_release_macro_triggered"] = bool(release_rows)
        result["pre_release_compatible_macro_triggered"] = bool(pre_release_rows)
        if release_rows:
            result["reason"] = "phased_release_dependency_macro"
            result["phased_release_bound_dependency"] = release_rows[0].get(
                "bound_dependency"
            )
            result["phased_release_macro_score"] = float(
                max(float(row.get("dependency_bound_score", 0.0)) for row in release_rows)
            )
            result["phased_release_extra_push_count"] = int(len(release_rows))
            result["phased_release_final_progress"] = release_rows[-1].get(
                "dependency_progress",
                {},
            )
        elif pre_release_rows:
            result["reason"] = "pre_release_compatible_dependency_macro"
            result["pre_release_compatible_extra_push_count"] = int(len(pre_release_rows))
            result["pre_release_compatible_final_progress"] = pre_release_rows[-1].get(
                "dependency_progress",
                {},
            )
        elif result.get("dependency_bound_macro_enabled"):
            result["phased_release_macro_reason"] = result.get(
                "dependency_bound_macro_reason",
                result.get("reason", "no_phased_release_suffix"),
            )
        return result


class RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner(
    RoomSchedulePhasedReleaseMacroGFTSDMCTSPlanner
):
    """A23 planner with a pre-release corridor macro safety boundary.

    A22 reaches the release stage with a stand blocker that was already made
    unreleasable during the fill phase.  A23 keeps A22's staged release logic but
    truncates deterministic corridor macros when a prefix enters a low-margin
    pre-release bottleneck with unresolved schedule dependencies.
    """

    provider_id = "room_schedule_pre_release_bottleneck_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        pre_release_corridor_guard_enabled: bool = True,
        pre_release_corridor_guard_min_pushes: int = 2,
        pre_release_corridor_guard_legal_margin: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pre_release_corridor_guard_enabled = bool(pre_release_corridor_guard_enabled)
        self.pre_release_corridor_guard_min_pushes = max(
            int(pre_release_corridor_guard_min_pushes),
            1,
        )
        self.pre_release_corridor_guard_legal_margin = max(
            int(pre_release_corridor_guard_legal_margin),
            int(self.room_access_recovery_min_legal_pushes),
        )

    def _pre_release_bottleneck_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        stage = self._decision_stage_summary(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        dependency_total = int(totals["dependency_count"]) + int(
            totals["future_dependency_count"]
        )
        legal = int(stage["nonblocked_legal_push_count"])
        boxes_on_goals = int(stage["boxes_on_goals"])
        unresolved_precondition = bool(
            int(totals["target_blocker_count"]) > 0
            or int(totals["unreachable_stand_count"]) > 0
            or int(totals["critical_stand_box_count"]) > 0
            or int(totals["future_critical_stand_box_count"]) > 0
        )
        bottleneck = bool(
            boxes_on_goals < self.stage_release_min_goals
            and dependency_total > 0
            and unresolved_precondition
            and int(totals["release_action_count"]) <= 0
            and legal <= self.pre_release_corridor_guard_legal_margin
        )
        return {
            "bottleneck": bool(bottleneck),
            "stage": stage,
            "dependency_total": int(dependency_total),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "unreachable_stand_count": int(totals["unreachable_stand_count"]),
            "critical_stand_box_count": int(totals["critical_stand_box_count"]),
            "future_critical_stand_box_count": int(
                totals["future_critical_stand_box_count"]
            ),
            "release_action_count": int(totals["release_action_count"]),
            "nonblocked_legal_push_count": int(legal),
            "boxes_on_goals": int(boxes_on_goals),
            "guard_legal_margin": int(self.pre_release_corridor_guard_legal_margin),
        }

    def corridor_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        macro = super().corridor_macro_from_plan(state, commitment_state, plan)
        if not self.pre_release_corridor_guard_enabled:
            return macro
        actions = list(macro.get("physical_actions", []))
        commitments = list(macro.get("after_commitment_states", []))
        rows = list(macro.get("rows", []))
        if len(actions) <= self.pre_release_corridor_guard_min_pushes:
            return macro

        current_state = state
        current_commitment = commitment_state
        truncate_at: int | None = None
        truncate_summary: dict[str, Any] | None = None
        for idx, action in enumerate(actions, start=1):
            child = self.model.step_grid_flow(current_state, action)
            if idx - 1 < len(commitments):
                current_commitment = self.model.normalize_room_commitment_state(
                    child,
                    commitments[idx - 1],
                )
            else:
                current_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
            current_state = child
            if idx < self.pre_release_corridor_guard_min_pushes:
                continue
            if idx >= len(actions):
                break
            summary = self._pre_release_bottleneck_summary(
                current_state,
                current_commitment,
            )
            if bool(summary["bottleneck"]):
                truncate_at = int(idx)
                truncate_summary = summary
                break

        if truncate_at is None:
            return macro

        kept_actions = actions[:truncate_at]
        kept_commitments = commitments[:truncate_at]
        kept_rows = rows[:truncate_at]
        if kept_rows:
            kept_rows[-1] = {
                **kept_rows[-1],
                "pre_release_bottleneck_guard_triggered": True,
                "pre_release_bottleneck_summary": truncate_summary,
                "pre_release_bottleneck_original_macro_push_count": int(len(actions)),
            }
        triggered = len(kept_actions) >= self.corridor_macro_min_total_pushes
        return {
            **macro,
            "triggered": bool(triggered),
            "reason": "pre_release_bottleneck_truncated_corridor",
            "physical_actions": kept_actions,
            "after_commitment_states": kept_commitments,
            "rows": kept_rows,
            "macro_push_count": int(len(kept_actions)),
            "extra_push_count": int(max(0, len(kept_actions) - 1)),
            "pre_release_bottleneck_guard_triggered": True,
            "pre_release_bottleneck_original_macro_push_count": int(len(actions)),
            "pre_release_bottleneck_truncated_push_count": int(len(kept_actions)),
            "pre_release_bottleneck_summary": truncate_summary,
        }


class RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner(
    RoomSchedulePreReleaseBottleneckGuardedMacroGFTSDMCTSPlanner
):
    """A24 planner with a root-level pre-release trap guard.

    A23 shows that shortening corridor macros can postpone, but not remove, the
    early stand-lock trap.  A24 keeps the same SD representation and macro
    machinery, then adds a root-only consumer score for transitions that convert
    target/stand preconditions into unreleasable critical stand blockers.
    """

    provider_id = "room_schedule_pre_release_root_trap_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        pre_release_root_trap_guard: bool = True,
        pre_release_root_trap_legal_margin: int = 2,
        pre_release_root_trap_locked_increase_penalty: float = 110.0,
        pre_release_root_trap_locked_absolute_penalty: float = 30.0,
        pre_release_root_trap_conversion_penalty: float = 70.0,
        pre_release_root_trap_occupied_increase_penalty: float = 18.0,
        pre_release_root_trap_low_margin_penalty: float = 24.0,
        pre_release_root_trap_no_release_penalty: float = 10.0,
        pre_release_root_trap_dependency_reduction_bonus: float = 18.0,
        pre_release_root_trap_target_reduction_bonus: float = 14.0,
        pre_release_root_trap_release_open_bonus: float = 24.0,
        pre_release_root_trap_legal_margin_bonus: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pre_release_root_trap_guard = bool(pre_release_root_trap_guard)
        self.pre_release_root_trap_legal_margin = max(
            int(pre_release_root_trap_legal_margin),
            int(self.room_access_recovery_min_legal_pushes),
        )
        self.pre_release_root_trap_locked_increase_penalty = float(
            pre_release_root_trap_locked_increase_penalty
        )
        self.pre_release_root_trap_locked_absolute_penalty = float(
            pre_release_root_trap_locked_absolute_penalty
        )
        self.pre_release_root_trap_conversion_penalty = float(
            pre_release_root_trap_conversion_penalty
        )
        self.pre_release_root_trap_occupied_increase_penalty = float(
            pre_release_root_trap_occupied_increase_penalty
        )
        self.pre_release_root_trap_low_margin_penalty = float(
            pre_release_root_trap_low_margin_penalty
        )
        self.pre_release_root_trap_no_release_penalty = float(
            pre_release_root_trap_no_release_penalty
        )
        self.pre_release_root_trap_dependency_reduction_bonus = float(
            pre_release_root_trap_dependency_reduction_bonus
        )
        self.pre_release_root_trap_target_reduction_bonus = float(
            pre_release_root_trap_target_reduction_bonus
        )
        self.pre_release_root_trap_release_open_bonus = float(
            pre_release_root_trap_release_open_bonus
        )
        self.pre_release_root_trap_legal_margin_bonus = float(
            pre_release_root_trap_legal_margin_bonus
        )
        self._pre_release_root_trap_root_key: tuple | None = None

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(
            state,
            commitment_state,
        )
        previous_root = self._pre_release_root_trap_root_key
        self._pre_release_root_trap_root_key = self.state_key(state, commitment_state)
        try:
            return super().plan(state, commitment_state)
        finally:
            self._pre_release_root_trap_root_key = previous_root

    def _pre_release_root_trap_skip_summary(self, reason: str) -> dict[str, Any]:
        return {
            "probe_skipped": True,
            "skip_reason": str(reason),
            "pre_release_context": False,
            "trap_risk": False,
            "severe_trap_risk": False,
            "score": 0.0,
        }

    def _pre_release_root_trap_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        key = self._summary_cache_key(
            "pre_release_root_trap_metrics",
            state,
            commitment_state,
            self.room_schedule_lookahead_slots,
            self.stage_release_min_goals,
            self.pre_release_root_trap_legal_margin,
        )
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached

        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        stand_lock = self._stand_lock_metrics(state, commitment_state)
        stage = self._decision_stage_summary(state, commitment_state)
        totals = schedule["totals"]
        dependency_total = int(totals["dependency_count"]) + int(
            totals["future_dependency_count"]
        )
        legal = int(stage["nonblocked_legal_push_count"])
        result = {
            "stage": str(stage["stage"]),
            "boxes_on_goals": int(self.model.boxes_on_goals(state)),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "dependency_total": int(dependency_total),
            "dependency_count": int(totals["dependency_count"]),
            "future_dependency_count": int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "unreachable_stand_count": int(totals["unreachable_stand_count"]),
            "critical_stand_box_count": int(totals["critical_stand_box_count"]),
            "future_critical_stand_box_count": int(
                totals["future_critical_stand_box_count"]
            ),
            "release_action_count": int(totals["release_action_count"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "nonblocked_legal_push_count": int(legal),
            "low_legal_margin": bool(
                dependency_total > 0 and legal <= self.pre_release_root_trap_legal_margin
            ),
            "occupied_critical_stand_count": int(
                stand_lock["occupied_critical_stand_count"]
            ),
            "locked_stand_blocker_count": int(
                stand_lock["locked_stand_blocker_count"]
            ),
            "goal_locked_stand_blocker_count": int(
                stand_lock["goal_locked_stand_blocker_count"]
            ),
            "assigned_goal_reassignment_count": int(
                stand_lock["assigned_goal_reassignment_count"]
            ),
            "stand_release_option_count": int(stand_lock["release_option_count"]),
        }
        result["precondition_count"] = int(
            result["target_blocker_count"]
            + result["unreachable_stand_count"]
            + result["critical_stand_box_count"]
            + result["future_critical_stand_box_count"]
        )
        self._summary_cache[key] = result
        return result

    def _pre_release_root_trap_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        if self._pre_release_root_trap_root_key is None:
            candidate.room_pre_release_trap_score = 0.0
            candidate.room_pre_release_trap_summary = self._pre_release_root_trap_skip_summary(
                "no_active_root"
            )
            return 0.0
        if self.state_key(state, commitment_state) != self._pre_release_root_trap_root_key:
            candidate.room_pre_release_trap_score = 0.0
            candidate.room_pre_release_trap_summary = self._pre_release_root_trap_skip_summary(
                "non_root_node"
            )
            return 0.0

        before = self._pre_release_root_trap_metrics(state, commitment_state)
        after = self._pre_release_root_trap_metrics(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        pre_release_context = bool(
            int(before["boxes_on_goals"]) < self.stage_release_min_goals
            and int(before["dependency_total"]) > 0
            and (
                int(before["precondition_count"]) > 0
                or int(before["locked_stand_blocker_count"]) > 0
                or int(before["occupied_critical_stand_count"]) > 0
            )
        )
        if not pre_release_context:
            candidate.room_pre_release_trap_score = 0.0
            candidate.room_pre_release_trap_summary = {
                **self._pre_release_root_trap_skip_summary("not_pre_release_context"),
                "before": before,
                "after": after,
            }
            return 0.0

        locked_delta = int(after["locked_stand_blocker_count"]) - int(
            before["locked_stand_blocker_count"]
        )
        occupied_delta = int(after["occupied_critical_stand_count"]) - int(
            before["occupied_critical_stand_count"]
        )
        dependency_delta = int(after["dependency_total"]) - int(
            before["dependency_total"]
        )
        target_delta = int(after["target_blocker_count"]) - int(
            before["target_blocker_count"]
        )
        unreachable_delta = int(after["unreachable_stand_count"]) - int(
            before["unreachable_stand_count"]
        )
        release_delta = int(after["release_action_count"]) - int(
            before["release_action_count"]
        )
        legal_delta = int(after["nonblocked_legal_push_count"]) - int(
            before["nonblocked_legal_push_count"]
        )
        blocker_box_delta = (
            int(after["critical_stand_box_count"])
            + int(after["future_critical_stand_box_count"])
            - int(before["critical_stand_box_count"])
            - int(before["future_critical_stand_box_count"])
        )
        precondition_to_lock_conversion = bool(
            locked_delta > 0
            and (
                int(before["target_blocker_count"]) > 0
                or int(before["unreachable_stand_count"]) > 0
                or target_delta < 0
                or unreachable_delta < 0
            )
        )
        bottleneck_after = bool(
            int(after["dependency_total"]) > 0
            and int(after["release_action_count"]) <= 0
            and bool(after["low_legal_margin"])
            and (
                int(after["precondition_count"]) > 0
                or int(after["locked_stand_blocker_count"]) > 0
            )
        )
        unreleasable_lock_after = bool(
            int(after["locked_stand_blocker_count"]) > 0
            and int(after["release_action_count"]) <= 0
        )
        trap_risk = bool(
            locked_delta > 0
            or precondition_to_lock_conversion
            or unreleasable_lock_after
            or (bottleneck_after and (occupied_delta > 0 or blocker_box_delta > 0))
        )
        severe_trap_risk = bool(
            precondition_to_lock_conversion
            or locked_delta > 0
            or unreleasable_lock_after
        )
        target_reduction_bonus = 0.0
        if not precondition_to_lock_conversion:
            target_reduction_bonus = self.pre_release_root_trap_target_reduction_bonus * float(
                max(0, -target_delta)
            )
        score = (
            -self.pre_release_root_trap_locked_increase_penalty
            * float(max(0, locked_delta))
            - self.pre_release_root_trap_locked_absolute_penalty
            * float(after["locked_stand_blocker_count"])
            - self.pre_release_root_trap_conversion_penalty
            * float(precondition_to_lock_conversion)
            - self.pre_release_root_trap_occupied_increase_penalty
            * float(max(0, occupied_delta + blocker_box_delta))
            - self.pre_release_root_trap_low_margin_penalty
            * float(bottleneck_after)
            - self.pre_release_root_trap_no_release_penalty
            * float(int(after["dependency_total"]) > 0 and int(after["release_action_count"]) <= 0)
            + self.pre_release_root_trap_dependency_reduction_bonus
            * float(max(0, -dependency_delta))
            + target_reduction_bonus
            + self.pre_release_root_trap_release_open_bonus
            * float(max(0, release_delta))
            + self.pre_release_root_trap_legal_margin_bonus
            * float(max(0, legal_delta))
        )
        candidate.room_pre_release_trap_score = float(score)
        candidate.room_pre_release_trap_summary = {
            "probe_skipped": False,
            "pre_release_context": bool(pre_release_context),
            "before": before,
            "after": after,
            "locked_stand_blocker_delta": int(locked_delta),
            "occupied_critical_stand_delta": int(occupied_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_delta),
            "unreachable_stand_count_delta": int(unreachable_delta),
            "critical_stand_box_delta": int(blocker_box_delta),
            "release_action_count_delta": int(release_delta),
            "legal_push_count_delta": int(legal_delta),
            "precondition_to_lock_conversion": bool(precondition_to_lock_conversion),
            "bottleneck_after": bool(bottleneck_after),
            "unreleasable_lock_after": bool(unreleasable_lock_after),
            "trap_risk": bool(trap_risk),
            "severe_trap_risk": bool(severe_trap_risk),
            "guard_legal_margin": int(self.pre_release_root_trap_legal_margin),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._pre_release_root_trap_score(state, commitment_state, candidate))

    def _pre_release_root_trap_choice_key(self, edge: RoomSemanticEdge) -> tuple:
        phase = edge.room_phase_progress_summary
        recovery = edge.room_recoverability_summary
        trap = edge.room_pre_release_trap_summary
        if self.select_by == "q":
            return (
                float(edge.room_pre_release_trap_score),
                bool(phase.get("phase_progress")),
                bool(recovery.get("recoverable_no_progress")),
                int(trap.get("after", {}).get("nonblocked_legal_push_count", 0)),
                edge.mean_value,
                edge.visits,
                edge.room_recoverability_score,
                edge.room_phase_progress_score,
                edge.prior_score,
            )
        return (
            float(edge.room_pre_release_trap_score),
            bool(phase.get("phase_progress")),
            bool(recovery.get("recoverable_no_progress")),
            int(trap.get("after", {}).get("nonblocked_legal_push_count", 0)),
            edge.visits,
            edge.mean_value,
            edge.room_recoverability_score,
            edge.room_phase_progress_score,
            edge.prior_score,
        )

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.pre_release_root_trap_guard:
            return best
        best_trap = best.room_pre_release_trap_summary
        if not bool(best_trap.get("severe_trap_risk")):
            return best

        alternatives = [
            edge
            for edge in root.children.values()
            if not bool(edge.room_pre_release_trap_summary.get("trap_risk"))
            and (
                bool(edge.room_phase_progress_summary.get("phase_progress"))
                or bool(edge.room_recoverability_summary.get("recoverable_no_progress"))
                or int(
                    edge.room_pre_release_trap_summary.get("after", {}).get(
                        "nonblocked_legal_push_count",
                        0,
                    )
                )
                > int(best_trap.get("after", {}).get("nonblocked_legal_push_count", 0))
            )
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if not bool(edge.room_pre_release_trap_summary.get("trap_risk"))
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_pre_release_trap_score > best.room_pre_release_trap_score
            ]
        if not alternatives:
            return best
        return max(alternatives, key=self._pre_release_root_trap_choice_key)


class RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner(
    RoomSchedulePreReleaseRootTrapGuardedMacroGFTSDMCTSPlanner
):
    """A25 planner with prototype-style SD-consumer generalization.

    This is a bridge toward a learned TVA/value consumer.  It keeps A24's
    explicit SD summaries, then adds a root-only prototype score that shares
    known failure evidence across structurally similar candidate states.
    """

    provider_id = "room_schedule_sd_generalization_guarded_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        sd_generalization_guard: bool = True,
        sd_generalization_legal_margin: int = 2,
        sd_generalization_predecessor_probe_enabled: bool = True,
        sd_generalization_predecessor_penalty: float = 95.0,
        sd_generalization_all_bad_successor_penalty: float = 125.0,
        sd_generalization_failure_similarity_penalty: float = 70.0,
        sd_generalization_success_similarity_bonus: float = 20.0,
        sd_generalization_release_open_bonus: float = 24.0,
        sd_generalization_dependency_reduction_bonus: float = 18.0,
        sd_generalization_legal_margin_bonus: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sd_generalization_guard = bool(sd_generalization_guard)
        self.sd_generalization_legal_margin = max(
            int(sd_generalization_legal_margin),
            int(self.room_access_recovery_min_legal_pushes),
        )
        self.sd_generalization_predecessor_probe_enabled = bool(
            sd_generalization_predecessor_probe_enabled
        )
        self.sd_generalization_predecessor_penalty = float(
            sd_generalization_predecessor_penalty
        )
        self.sd_generalization_all_bad_successor_penalty = float(
            sd_generalization_all_bad_successor_penalty
        )
        self.sd_generalization_failure_similarity_penalty = float(
            sd_generalization_failure_similarity_penalty
        )
        self.sd_generalization_success_similarity_bonus = float(
            sd_generalization_success_similarity_bonus
        )
        self.sd_generalization_release_open_bonus = float(
            sd_generalization_release_open_bonus
        )
        self.sd_generalization_dependency_reduction_bonus = float(
            sd_generalization_dependency_reduction_bonus
        )
        self.sd_generalization_legal_margin_bonus = float(
            sd_generalization_legal_margin_bonus
        )

    def _sd_generalization_signature(self, metrics: dict[str, Any]) -> dict[str, int]:
        legal = int(metrics["nonblocked_legal_push_count"])
        return {
            "pre_release": int(int(metrics["boxes_on_goals"]) < self.stage_release_min_goals),
            "dependency_active": int(int(metrics["dependency_total"]) > 0),
            "target_blocker": int(int(metrics["target_blocker_count"]) > 0),
            "unreachable_stand": int(int(metrics["unreachable_stand_count"]) > 0),
            "critical_stand_box": int(
                int(metrics["critical_stand_box_count"])
                + int(metrics["future_critical_stand_box_count"])
                > 0
            ),
            "locked_stand": int(int(metrics["locked_stand_blocker_count"]) > 0),
            "release_closed": int(int(metrics["release_action_count"]) <= 0),
            "low_margin": int(legal <= self.sd_generalization_legal_margin),
            "single_action_margin": int(legal <= 1),
        }

    @staticmethod
    def _sd_signature_similarity(
        left: dict[str, int],
        right: dict[str, int],
        weights: dict[str, float],
    ) -> float:
        total = sum(float(weights.get(key, 1.0)) for key in left)
        if total <= 0.0:
            return 0.0
        match = sum(
            float(weights.get(key, 1.0)) * float(int(left[key]) == int(right.get(key, 0)))
            for key in left
        )
        return float(match / total)

    def _sd_generalization_prototypes(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "a24_step5_single_action_severe_trap",
                "label": "failure",
                "signature": {
                    "pre_release": 1,
                    "dependency_active": 1,
                    "target_blocker": 1,
                    "unreachable_stand": 1,
                    "critical_stand_box": 0,
                    "locked_stand": 0,
                    "release_closed": 1,
                    "low_margin": 1,
                    "single_action_margin": 1,
                },
            },
            {
                "name": "a24_post_trap_locked_loop",
                "label": "failure",
                "signature": {
                    "pre_release": 1,
                    "dependency_active": 1,
                    "target_blocker": 1,
                    "unreachable_stand": 1,
                    "critical_stand_box": 1,
                    "locked_stand": 1,
                    "release_closed": 1,
                    "low_margin": 1,
                    "single_action_margin": 0,
                },
            },
            {
                "name": "a20_a23_safe_fill_prefix",
                "label": "success",
                "signature": {
                    "pre_release": 1,
                    "dependency_active": 1,
                    "target_blocker": 1,
                    "unreachable_stand": 1,
                    "critical_stand_box": 0,
                    "locked_stand": 0,
                    "release_closed": 1,
                    "low_margin": 0,
                    "single_action_margin": 0,
                },
            },
            {
                "name": "open_release_candidate",
                "label": "success",
                "signature": {
                    "pre_release": 1,
                    "dependency_active": 1,
                    "target_blocker": 0,
                    "unreachable_stand": 0,
                    "critical_stand_box": 0,
                    "locked_stand": 0,
                    "release_closed": 0,
                    "low_margin": 0,
                    "single_action_margin": 0,
                },
            },
        ]

    def _sd_generalization_prototype_summary(
        self,
        signature: dict[str, int],
    ) -> dict[str, Any]:
        weights = {
            "pre_release": 1.0,
            "dependency_active": 1.0,
            "target_blocker": 1.4,
            "unreachable_stand": 1.2,
            "critical_stand_box": 1.8,
            "locked_stand": 2.2,
            "release_closed": 1.4,
            "low_margin": 1.8,
            "single_action_margin": 2.2,
        }
        rows = []
        best_failure = 0.0
        best_success = 0.0
        for prototype in self._sd_generalization_prototypes():
            similarity = self._sd_signature_similarity(
                signature,
                prototype["signature"],
                weights,
            )
            row = {
                "name": prototype["name"],
                "label": prototype["label"],
                "similarity": float(similarity),
            }
            rows.append(row)
            if prototype["label"] == "failure":
                best_failure = max(best_failure, similarity)
            else:
                best_success = max(best_success, similarity)
        rows.sort(key=lambda row: row["similarity"], reverse=True)
        return {
            "signature": signature,
            "best_failure_similarity": float(best_failure),
            "best_success_similarity": float(best_success),
            "top_matches": rows[:4],
        }

    def _sd_generalization_successor_probe(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        if not self.sd_generalization_predecessor_probe_enabled:
            return {
                "probe_enabled": False,
                "trap_predecessor": False,
                "legal_successor_count": None,
                "severe_successor_count": None,
            }
        actions = [
            action
            for action in self.model.legal_flows(state)
            if not self.is_blocked(self.model.step_grid_flow(state, action))
        ]
        severe_rows = []
        non_trap_rows = []
        for action in actions:
            child = self.model.step_grid_flow(state, action)
            child_commitment = self._advance_recovery_commitment(
                state,
                commitment_state,
                action,
                child,
            )
            before = self._pre_release_root_trap_metrics(state, commitment_state)
            after = self._pre_release_root_trap_metrics(child, child_commitment)
            locked_delta = int(after["locked_stand_blocker_count"]) - int(
                before["locked_stand_blocker_count"]
            )
            occupied_delta = int(after["occupied_critical_stand_count"]) - int(
                before["occupied_critical_stand_count"]
            )
            target_delta = int(after["target_blocker_count"]) - int(
                before["target_blocker_count"]
            )
            unreachable_delta = int(after["unreachable_stand_count"]) - int(
                before["unreachable_stand_count"]
            )
            severe = bool(
                locked_delta > 0
                or (
                    int(after["locked_stand_blocker_count"]) > 0
                    and int(after["release_action_count"]) <= 0
                )
                or (
                    occupied_delta > 0
                    and int(after["release_action_count"]) <= 0
                    and int(after["nonblocked_legal_push_count"])
                    <= self.sd_generalization_legal_margin
                )
            )
            row = {
                "action": self._physical_action_dict(state, action),
                "locked_stand_blocker_delta": int(locked_delta),
                "occupied_critical_stand_delta": int(occupied_delta),
                "target_blocker_count_delta": int(target_delta),
                "unreachable_stand_count_delta": int(unreachable_delta),
                "after_legal_push_count": int(after["nonblocked_legal_push_count"]),
                "severe_successor_trap": bool(severe),
            }
            if severe:
                severe_rows.append(row)
            else:
                non_trap_rows.append(row)
        legal_count = len(actions)
        severe_count = len(severe_rows)
        all_bad = bool(legal_count > 0 and severe_count == legal_count)
        trap_predecessor = bool(
            legal_count <= 1 and severe_count >= 1
        ) or bool(all_bad and legal_count <= self.sd_generalization_legal_margin)
        return {
            "probe_enabled": True,
            "trap_predecessor": bool(trap_predecessor),
            "all_successors_severe": bool(all_bad),
            "legal_successor_count": int(legal_count),
            "severe_successor_count": int(severe_count),
            "non_trap_successor_count": int(len(non_trap_rows)),
            "severe_successors": severe_rows[:4],
            "non_trap_successors": non_trap_rows[:4],
        }

    def _sd_generalization_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        current_key = self.state_key(state, commitment_state)
        if self._pre_release_root_trap_root_key is None or (
            current_key != self._pre_release_root_trap_root_key
        ):
            candidate.room_sd_generalization_score = 0.0
            candidate.room_sd_generalization_summary = {
                "probe_skipped": True,
                "skip_reason": "non_root_or_no_active_root",
                "score": 0.0,
            }
            return 0.0

        before = self._pre_release_root_trap_metrics(state, commitment_state)
        after = self._pre_release_root_trap_metrics(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        signature = self._sd_generalization_signature(after)
        prototypes = self._sd_generalization_prototype_summary(signature)
        successor_probe = self._sd_generalization_successor_probe(
            candidate.child_state,
            candidate.child_commitment_state,
        )
        dependency_delta = int(after["dependency_total"]) - int(
            before["dependency_total"]
        )
        release_delta = int(after["release_action_count"]) - int(
            before["release_action_count"]
        )
        legal_delta = int(after["nonblocked_legal_push_count"]) - int(
            before["nonblocked_legal_push_count"]
        )
        failure_excess = max(
            0.0,
            float(prototypes["best_failure_similarity"])
            - float(prototypes["best_success_similarity"]),
        )
        success_excess = max(
            0.0,
            float(prototypes["best_success_similarity"])
            - float(prototypes["best_failure_similarity"]),
        )
        score = (
            -self.sd_generalization_failure_similarity_penalty * failure_excess
            + self.sd_generalization_success_similarity_bonus * success_excess
            - self.sd_generalization_predecessor_penalty
            * float(successor_probe.get("trap_predecessor", False))
            - self.sd_generalization_all_bad_successor_penalty
            * float(successor_probe.get("all_successors_severe", False))
            + self.sd_generalization_dependency_reduction_bonus
            * float(max(0, -dependency_delta))
            + self.sd_generalization_release_open_bonus
            * float(max(0, release_delta))
            + self.sd_generalization_legal_margin_bonus * float(max(0, legal_delta))
        )
        candidate.room_sd_generalization_score = float(score)
        candidate.room_sd_generalization_summary = {
            "probe_skipped": False,
            "before": before,
            "after": after,
            "signature": signature,
            "prototype_summary": prototypes,
            "successor_probe": successor_probe,
            "dependency_total_delta": int(dependency_delta),
            "release_action_count_delta": int(release_delta),
            "legal_push_count_delta": int(legal_delta),
            "failure_similarity_excess": float(failure_excess),
            "success_similarity_excess": float(success_excess),
            "generalized_failure_risk": bool(
                failure_excess > 0.05 or successor_probe.get("trap_predecessor", False)
            ),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._sd_generalization_score(state, commitment_state, candidate))

    def _sd_generalization_choice_key(self, edge: RoomSemanticEdge) -> tuple:
        sdg = edge.room_sd_generalization_summary
        trap = edge.room_pre_release_trap_summary
        phase = edge.room_phase_progress_summary
        recovery = edge.room_recoverability_summary
        if self.select_by == "q":
            return (
                float(edge.room_sd_generalization_score),
                float(edge.room_pre_release_trap_score),
                not bool(sdg.get("generalized_failure_risk")),
                not bool(trap.get("severe_trap_risk")),
                bool(phase.get("phase_progress")),
                bool(recovery.get("recoverable_no_progress")),
                edge.mean_value,
                edge.visits,
                edge.prior_score,
            )
        return (
            float(edge.room_sd_generalization_score),
            float(edge.room_pre_release_trap_score),
            not bool(sdg.get("generalized_failure_risk")),
            not bool(trap.get("severe_trap_risk")),
            bool(phase.get("phase_progress")),
            bool(recovery.get("recoverable_no_progress")),
            edge.visits,
            edge.mean_value,
            edge.prior_score,
        )

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.sd_generalization_guard:
            return best
        best_summary = best.room_sd_generalization_summary
        best_bad = bool(best_summary.get("generalized_failure_risk")) or bool(
            best_summary.get("successor_probe", {}).get("trap_predecessor", False)
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if not bool(edge.room_sd_generalization_summary.get("generalized_failure_risk"))
            and not bool(edge.room_pre_release_trap_summary.get("severe_trap_risk"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_sd_generalization_score > best.room_sd_generalization_score
            ]
        if not alternatives:
            return best
        return max(alternatives, key=self._sd_generalization_choice_key)


class RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner(
    RoomScheduleSDGeneralizationGuardedMacroGFTSDMCTSPlanner
):
    """A26 planner with a learned-consumer style SD value interface.

    The class keeps A25's prototype guard as an input signal, but separates
    trap avoidance from milestone progress in a root-only linear value score.
    The linear score is a placeholder for a learned TVA/value/ranking consumer.
    """

    provider_id = "room_schedule_sd_value_ranked_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        sd_value_guard: bool = True,
        sd_value_legal_margin: int = 2,
        sd_value_goal_bonus: float = 180.0,
        sd_value_release_bonus: float = 120.0,
        sd_value_dependency_bonus: float = 46.0,
        sd_value_target_bonus: float = 42.0,
        sd_value_slot_bonus: float = 10.0,
        sd_value_barrier_bonus: float = 4.0,
        sd_value_legal_margin_bonus: float = 8.0,
        sd_value_recoverable_bonus: float = 18.0,
        sd_value_dependency_regression_penalty: float = 34.0,
        sd_value_target_regression_penalty: float = 28.0,
        sd_value_slot_regression_penalty: float = 10.0,
        sd_value_no_progress_penalty: float = 54.0,
        sd_value_stagnant_penalty: float = 42.0,
        sd_value_repeated_signature_penalty: float = 30.0,
        sd_value_repeated_edge_penalty: float = 28.0,
        sd_value_reverse_edge_penalty: float = 38.0,
        sd_value_low_margin_penalty: float = 26.0,
        sd_value_trap_penalty: float = 80.0,
        sd_value_severe_trap_penalty: float = 160.0,
        sd_value_generalized_failure_penalty: float = 45.0,
        sd_value_trap_predecessor_penalty: float = 140.0,
        sd_value_stand_lock_penalty: float = 70.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sd_value_guard = bool(sd_value_guard)
        self.sd_value_legal_margin = max(
            int(sd_value_legal_margin),
            int(self.room_access_recovery_min_legal_pushes),
        )
        self.sd_value_goal_bonus = float(sd_value_goal_bonus)
        self.sd_value_release_bonus = float(sd_value_release_bonus)
        self.sd_value_dependency_bonus = float(sd_value_dependency_bonus)
        self.sd_value_target_bonus = float(sd_value_target_bonus)
        self.sd_value_slot_bonus = float(sd_value_slot_bonus)
        self.sd_value_barrier_bonus = float(sd_value_barrier_bonus)
        self.sd_value_legal_margin_bonus = float(sd_value_legal_margin_bonus)
        self.sd_value_recoverable_bonus = float(sd_value_recoverable_bonus)
        self.sd_value_dependency_regression_penalty = float(
            sd_value_dependency_regression_penalty
        )
        self.sd_value_target_regression_penalty = float(
            sd_value_target_regression_penalty
        )
        self.sd_value_slot_regression_penalty = float(sd_value_slot_regression_penalty)
        self.sd_value_no_progress_penalty = float(sd_value_no_progress_penalty)
        self.sd_value_stagnant_penalty = float(sd_value_stagnant_penalty)
        self.sd_value_repeated_signature_penalty = float(
            sd_value_repeated_signature_penalty
        )
        self.sd_value_repeated_edge_penalty = float(sd_value_repeated_edge_penalty)
        self.sd_value_reverse_edge_penalty = float(sd_value_reverse_edge_penalty)
        self.sd_value_low_margin_penalty = float(sd_value_low_margin_penalty)
        self.sd_value_trap_penalty = float(sd_value_trap_penalty)
        self.sd_value_severe_trap_penalty = float(sd_value_severe_trap_penalty)
        self.sd_value_generalized_failure_penalty = float(
            sd_value_generalized_failure_penalty
        )
        self.sd_value_trap_predecessor_penalty = float(
            sd_value_trap_predecessor_penalty
        )
        self.sd_value_stand_lock_penalty = float(sd_value_stand_lock_penalty)

    def _sd_value_skip_summary(self, reason: str) -> dict[str, Any]:
        return {
            "probe_skipped": True,
            "skip_reason": str(reason),
            "score": 0.0,
            "milestone_progress": False,
            "no_progress_loop_risk": False,
            "hard_negative": False,
        }

    def _sd_value_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        current_key = self.state_key(state, commitment_state)
        if self._pre_release_root_trap_root_key is None:
            candidate.room_sd_value_score = 0.0
            candidate.room_sd_value_summary = self._sd_value_skip_summary("no_active_root")
            return 0.0
        if current_key != self._pre_release_root_trap_root_key:
            candidate.room_sd_value_score = 0.0
            candidate.room_sd_value_summary = self._sd_value_skip_summary("non_root_node")
            return 0.0

        phase = candidate.room_phase_progress_summary
        trap = candidate.room_pre_release_trap_summary
        sdg = candidate.room_sd_generalization_summary
        recovery = candidate.room_recoverability_summary
        stand_lock = candidate.room_stand_lock_summary
        after = trap.get("after", {})

        goal_delta = int(phase.get("goal_delta", 0))
        release_delta = int(phase.get("release_action_count_delta", 0))
        dependency_delta = int(phase.get("dependency_total_delta", 0))
        target_delta = int(phase.get("target_blocker_count_delta", 0))
        slot_delta = float(phase.get("slot_schedule_cost_delta", 0.0))
        barrier_delta = float(phase.get("schedule_barrier_cost_delta", 0.0))
        legal_delta = int(trap.get("legal_push_count_delta", 0))
        legal_after = int(after.get("nonblocked_legal_push_count", 0))
        dependency_after = int(after.get("dependency_total", 0))

        no_progress = bool(phase.get("no_progress"))
        stagnant = bool(phase.get("stagnant_context"))
        repeated_signature = bool(phase.get("child_repeated_schedule_signature"))
        repeated_edge = bool(phase.get("repeated_edge"))
        reverse_edge = bool(phase.get("reverse_recent_edge"))
        no_release_context = bool(phase.get("no_release_context"))
        low_margin = bool(
            dependency_after > 0 and legal_after <= self.sd_value_legal_margin
        )

        trap_risk = bool(trap.get("trap_risk"))
        severe_trap = bool(trap.get("severe_trap_risk"))
        generalized_failure = bool(sdg.get("generalized_failure_risk"))
        trap_predecessor = bool(
            sdg.get("successor_probe", {}).get("trap_predecessor")
        )
        stand_lock_risk = bool(stand_lock.get("stand_lock_risk")) or bool(
            stand_lock.get("after", {}).get("locked_stand_blocker_count", 0)
        )
        recoverable = bool(recovery.get("recoverable_no_progress"))
        unrecoverable = bool(recovery.get("unrecoverable_no_progress"))

        milestone_score = (
            self.sd_value_goal_bonus * float(max(0, goal_delta))
            + self.sd_value_release_bonus * float(max(0, release_delta))
            + self.sd_value_dependency_bonus * float(max(0, dependency_delta))
            + self.sd_value_target_bonus * float(max(0, target_delta))
        )
        soft_progress_score = (
            self.sd_value_slot_bonus * float(max(0.0, min(slot_delta, 3.0)))
            + self.sd_value_barrier_bonus
            * float(max(0.0, min(barrier_delta, 12.0)))
            + self.sd_value_legal_margin_bonus * float(max(0, legal_delta))
            + self.sd_value_recoverable_bonus * float(recoverable)
        )
        regression_penalty = (
            self.sd_value_dependency_regression_penalty
            * float(max(0, -dependency_delta))
            + self.sd_value_target_regression_penalty * float(max(0, -target_delta))
            + self.sd_value_slot_regression_penalty
            * float(max(0.0, min(-slot_delta, 3.0)))
        )
        loop_penalty = (
            self.sd_value_no_progress_penalty * float(no_progress)
            + self.sd_value_stagnant_penalty * float(no_progress and stagnant)
            + self.sd_value_repeated_signature_penalty
            * float(no_progress and repeated_signature)
            + self.sd_value_repeated_edge_penalty * float(no_progress and repeated_edge)
            + self.sd_value_reverse_edge_penalty * float(no_progress and reverse_edge)
            + self.sd_value_low_margin_penalty * float(no_progress and low_margin)
        )
        trap_penalty = (
            self.sd_value_trap_penalty * float(trap_risk)
            + self.sd_value_severe_trap_penalty * float(severe_trap)
            + self.sd_value_generalized_failure_penalty
            * float(generalized_failure and not max(0, release_delta))
            + self.sd_value_trap_predecessor_penalty * float(trap_predecessor)
            + self.sd_value_stand_lock_penalty * float(stand_lock_risk)
            + self.sd_value_low_margin_penalty * float(unrecoverable)
        )

        milestone_progress = bool(
            goal_delta > 0 or release_delta > 0 or dependency_delta > 0 or target_delta > 0
        )
        no_progress_loop_risk = bool(
            no_progress
            and no_release_context
            and (stagnant or repeated_signature or repeated_edge or reverse_edge or low_margin)
        )
        hard_negative = bool(severe_trap or trap_predecessor)
        score = milestone_score + soft_progress_score - regression_penalty - loop_penalty - trap_penalty
        candidate.room_sd_value_score = float(score)
        candidate.room_sd_value_summary = {
            "probe_skipped": False,
            "score": float(score),
            "milestone_score": float(milestone_score),
            "soft_progress_score": float(soft_progress_score),
            "regression_penalty": float(regression_penalty),
            "loop_penalty": float(loop_penalty),
            "trap_penalty": float(trap_penalty),
            "milestone_progress": bool(milestone_progress),
            "no_progress_loop_risk": bool(no_progress_loop_risk),
            "hard_negative": bool(hard_negative),
            "soft_negative": bool(trap_risk or generalized_failure or stand_lock_risk),
            "goal_delta": int(goal_delta),
            "release_action_count_delta": int(release_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "schedule_barrier_cost_delta": float(barrier_delta),
            "legal_push_count_delta": int(legal_delta),
            "after_legal_push_count": int(legal_after),
            "after_dependency_total": int(dependency_after),
            "no_progress": bool(no_progress),
            "stagnant_context": bool(stagnant),
            "child_repeated_schedule_signature": bool(repeated_signature),
            "repeated_edge": bool(repeated_edge),
            "reverse_recent_edge": bool(reverse_edge),
            "low_margin": bool(low_margin),
            "no_release_context": bool(no_release_context),
            "trap_risk": bool(trap_risk),
            "severe_trap_risk": bool(severe_trap),
            "generalized_failure_risk": bool(generalized_failure),
            "trap_predecessor": bool(trap_predecessor),
            "stand_lock_risk": bool(stand_lock_risk),
            "recoverable_no_progress": bool(recoverable),
            "unrecoverable_no_progress": bool(unrecoverable),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._sd_value_score(state, commitment_state, candidate))

    def _sd_value_choice_key(self, edge: RoomSemanticEdge) -> tuple:
        sdv = edge.room_sd_value_summary
        phase = edge.room_phase_progress_summary
        if self.select_by == "q":
            return (
                float(edge.room_sd_value_score),
                bool(sdv.get("milestone_progress")),
                not bool(sdv.get("no_progress_loop_risk")),
                not bool(sdv.get("hard_negative")),
                bool(phase.get("phase_progress")),
                edge.mean_value,
                edge.visits,
                edge.prior_score,
            )
        return (
            float(edge.room_sd_value_score),
            bool(sdv.get("milestone_progress")),
            not bool(sdv.get("no_progress_loop_risk")),
            not bool(sdv.get("hard_negative")),
            bool(phase.get("phase_progress")),
            edge.visits,
            edge.mean_value,
            edge.prior_score,
        )

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.sd_value_guard:
            return best
        best_value = best.room_sd_value_summary
        best_bad = bool(best_value.get("hard_negative")) or (
            bool(best_value.get("no_progress_loop_risk"))
            and not bool(best_value.get("milestone_progress"))
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if bool(edge.room_sd_value_summary.get("milestone_progress"))
            and not bool(edge.room_sd_value_summary.get("hard_negative"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if not bool(edge.room_sd_value_summary.get("hard_negative"))
                and edge.room_sd_value_score > best.room_sd_value_score
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_sd_value_score > best.room_sd_value_score
            ]
        if not alternatives:
            return best
        return max(alternatives, key=self._sd_value_choice_key)


class RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner(
    RoomScheduleSDValueRankedMacroGFTSDMCTSPlanner
):
    """A27 planner that calibrates repeated pseudo-milestones in A26."""

    provider_id = "room_schedule_sd_value_loop_calibrated_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        sd_value_pseudo_milestone_penalty: float = 190.0,
        sd_value_repeated_milestone_penalty: float = 90.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sd_value_pseudo_milestone_penalty = float(sd_value_pseudo_milestone_penalty)
        self.sd_value_repeated_milestone_penalty = float(sd_value_repeated_milestone_penalty)

    def _sd_value_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        score = super()._sd_value_score(state, commitment_state, candidate)
        summary = dict(candidate.room_sd_value_summary)
        if bool(summary.get("probe_skipped")):
            return float(score)

        real_milestone = bool(
            int(summary.get("goal_delta", 0)) > 0
            or int(summary.get("release_action_count_delta", 0)) > 0
        )
        structural_milestone = bool(
            int(summary.get("dependency_total_delta", 0)) > 0
            or int(summary.get("target_blocker_count_delta", 0)) > 0
        )
        repeated_context = bool(
            summary.get("stagnant_context")
            or summary.get("child_repeated_schedule_signature")
            or summary.get("repeated_edge")
            or summary.get("reverse_recent_edge")
        )
        pseudo_milestone_loop = bool(
            structural_milestone
            and not real_milestone
            and repeated_context
            and bool(summary.get("no_release_context"))
        )
        penalty = 0.0
        if pseudo_milestone_loop:
            penalty += self.sd_value_pseudo_milestone_penalty
        elif structural_milestone and repeated_context and not real_milestone:
            penalty += self.sd_value_repeated_milestone_penalty
        if penalty > 0.0:
            score = float(score - penalty)
            summary["score"] = float(score)
            summary["loop_calibration_penalty"] = float(penalty)
            summary["pseudo_milestone_loop"] = bool(pseudo_milestone_loop)
            summary["repeated_milestone_context"] = bool(repeated_context)
            summary["milestone_progress"] = bool(real_milestone)
            summary["no_progress_loop_risk"] = bool(
                summary.get("no_progress_loop_risk") or pseudo_milestone_loop
            )
            summary["hard_negative"] = bool(
                summary.get("hard_negative") or pseudo_milestone_loop
            )
            candidate.room_sd_value_score = float(score)
            candidate.room_sd_value_summary = summary
        return float(score)


class RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner(
    RoomScheduleSDValueLoopCalibratedMacroGFTSDMCTSPlanner
):
    """A28 planner that consumes A27 pseudo-milestone loop labels at root."""

    provider_id = "room_schedule_sd_value_loop_escape_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        sd_value_loop_escape_guard: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sd_value_loop_escape_guard = bool(sd_value_loop_escape_guard)

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if not self.sd_value_loop_escape_guard:
            return best
        best_value = best.room_sd_value_summary
        best_bad = bool(best_value.get("pseudo_milestone_loop")) or (
            bool(best_value.get("hard_negative"))
            and bool(best_value.get("no_progress_loop_risk"))
        )
        if not best_bad:
            return best
        alternatives = [
            edge
            for edge in root.children.values()
            if not bool(edge.room_sd_value_summary.get("hard_negative"))
            and not bool(edge.room_sd_value_summary.get("pseudo_milestone_loop"))
        ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if not bool(edge.room_sd_value_summary.get("pseudo_milestone_loop"))
            ]
        if not alternatives:
            alternatives = [
                edge
                for edge in root.children.values()
                if edge.room_sd_value_score > best.room_sd_value_score
            ]
        if not alternatives:
            return best
        return max(alternatives, key=self._sd_value_choice_key)


class RoomScheduleReleaseActionMacroGFTSDMCTSPlanner(
    RoomScheduleSDValueLoopEscapeMacroGFTSDMCTSPlanner
):
    """A29 planner that adds a release-stage macro after A28.

    A28 reaches the release stage but keeps ``release_action_count=0``.  This
    subclass keeps A28's root selection unchanged and only inserts a bounded
    post-root macro when the current decision stage is release and the schedule
    still has no executable release action.
    """

    provider_id = "room_schedule_release_action_macro_gft_sdmcts_v1"
    release_action_macro_reason = "release_action_macro_push"

    def __init__(
        self,
        *args,
        release_action_macro_enabled: bool = True,
        release_action_macro_max_extra_pushes: int = 5,
        release_action_macro_max_nodes: int = 160,
        release_action_macro_score_threshold: float = 24.0,
        release_action_macro_release_weight: float = 140.0,
        release_action_macro_stand_weight: float = 120.0,
        release_action_macro_locked_weight: float = 90.0,
        release_action_macro_dependency_weight: float = 34.0,
        release_action_macro_target_weight: float = 30.0,
        release_action_macro_legal_weight: float = 4.0,
        release_action_macro_slot_weight: float = 2.0,
        release_action_macro_goal_regression_penalty: float = 220.0,
        release_action_macro_dependency_regression_penalty: float = 40.0,
        release_action_macro_target_regression_penalty: float = 36.0,
        release_action_macro_low_legal_penalty: float = 28.0,
        release_action_macro_no_legal_penalty: float = 120.0,
        release_action_macro_step_penalty: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.release_action_macro_enabled = bool(release_action_macro_enabled)
        self.release_action_macro_max_extra_pushes = max(
            int(release_action_macro_max_extra_pushes),
            0,
        )
        self.release_action_macro_max_nodes = max(
            int(release_action_macro_max_nodes),
            1,
        )
        self.release_action_macro_score_threshold = float(
            release_action_macro_score_threshold
        )
        self.release_action_macro_release_weight = float(
            release_action_macro_release_weight
        )
        self.release_action_macro_stand_weight = float(release_action_macro_stand_weight)
        self.release_action_macro_locked_weight = float(
            release_action_macro_locked_weight
        )
        self.release_action_macro_dependency_weight = float(
            release_action_macro_dependency_weight
        )
        self.release_action_macro_target_weight = float(release_action_macro_target_weight)
        self.release_action_macro_legal_weight = float(release_action_macro_legal_weight)
        self.release_action_macro_slot_weight = float(release_action_macro_slot_weight)
        self.release_action_macro_goal_regression_penalty = float(
            release_action_macro_goal_regression_penalty
        )
        self.release_action_macro_dependency_regression_penalty = float(
            release_action_macro_dependency_regression_penalty
        )
        self.release_action_macro_target_regression_penalty = float(
            release_action_macro_target_regression_penalty
        )
        self.release_action_macro_low_legal_penalty = float(
            release_action_macro_low_legal_penalty
        )
        self.release_action_macro_no_legal_penalty = float(
            release_action_macro_no_legal_penalty
        )
        self.release_action_macro_step_penalty = float(release_action_macro_step_penalty)

    def _release_action_macro_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        dependency: dict[str, Any] | None,
    ) -> dict[str, Any]:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        totals = schedule["totals"]
        legal = self._nonblocked_legal_push_count(state)
        stand_lock = self._stand_lock_metrics(state, commitment_state)
        dependency_total = int(totals["dependency_count"]) + int(
            totals["future_dependency_count"]
        )
        stand_cell = None if dependency is None else self._dependency_cell(
            dependency,
            "stand_cell",
        )
        target_cell = None if dependency is None else self._dependency_cell(
            dependency,
            "target_cell",
        )
        selected_release_actions: list[dict[str, Any]] = []
        if dependency is not None:
            selected_release_actions = self._dependency_bound_release_actions(
                schedule,
                dependency,
            )
        return {
            "boxes_on_goals": int(self.model.boxes_on_goals(state)),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "dependency_total": int(dependency_total),
            "dependency_count": int(totals["dependency_count"]),
            "future_dependency_count": int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "unreachable_stand_count": int(totals["unreachable_stand_count"]),
            "critical_stand_box_count": int(totals["critical_stand_box_count"]),
            "future_critical_stand_box_count": int(
                totals["future_critical_stand_box_count"]
            ),
            "release_action_count": int(totals["release_action_count"]),
            "selected_release_action_count": int(len(selected_release_actions)),
            "selected_release_actions": selected_release_actions,
            "stand_cell_occupied_by_box": bool(
                stand_cell is not None and stand_cell in state.boxes
            ),
            "target_occupied_by_box": bool(
                target_cell is not None and target_cell in state.boxes
            ),
            "locked_stand_blocker_count": int(
                stand_lock["locked_stand_blocker_count"]
            ),
            "occupied_critical_stand_count": int(
                stand_lock["occupied_critical_stand_count"]
            ),
            "nonblocked_legal_push_count": int(legal),
            "low_legal_margin": bool(
                dependency_total > 0 and legal < self.room_access_recovery_min_legal_pushes
            ),
            "no_legal_pushes": bool(legal <= 0),
        }

    def _release_action_macro_score(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        depth: int,
    ) -> tuple[float, dict[str, Any]]:
        release_delta = int(after["release_action_count"]) - int(
            before["release_action_count"]
        )
        selected_release_delta = int(after["selected_release_action_count"]) - int(
            before["selected_release_action_count"]
        )
        stand_resolved = bool(
            bool(before["stand_cell_occupied_by_box"])
            and not bool(after["stand_cell_occupied_by_box"])
        )
        locked_delta = int(before["locked_stand_blocker_count"]) - int(
            after["locked_stand_blocker_count"]
        )
        dependency_delta = int(before["dependency_total"]) - int(after["dependency_total"])
        target_delta = int(before["target_blocker_count"]) - int(
            after["target_blocker_count"]
        )
        legal_delta = int(after["nonblocked_legal_push_count"]) - int(
            before["nonblocked_legal_push_count"]
        )
        slot_delta = float(before["slot_schedule_cost"]) - float(
            after["slot_schedule_cost"]
        )
        goal_delta = int(after["boxes_on_goals"]) - int(before["boxes_on_goals"])
        opened_release = bool(release_delta > 0 or selected_release_delta > 0)
        dependency_progress = bool(
            opened_release
            or stand_resolved
            or locked_delta > 0
            or dependency_delta > 0
            or target_delta > 0
        )
        score = (
            self.release_action_macro_release_weight
            * float(max(0, release_delta + selected_release_delta))
            + self.release_action_macro_stand_weight * float(stand_resolved)
            + self.release_action_macro_locked_weight * float(max(0, locked_delta))
            + self.release_action_macro_dependency_weight
            * float(max(0, dependency_delta))
            + self.release_action_macro_target_weight * float(max(0, target_delta))
            + self.release_action_macro_legal_weight * float(legal_delta)
            + self.release_action_macro_slot_weight * float(slot_delta)
            - self.release_action_macro_goal_regression_penalty
            * float(max(0, -goal_delta))
            - self.release_action_macro_dependency_regression_penalty
            * float(max(0, -dependency_delta))
            - self.release_action_macro_target_regression_penalty
            * float(max(0, -target_delta))
            - self.release_action_macro_low_legal_penalty
            * float(after["low_legal_margin"])
            - self.release_action_macro_no_legal_penalty
            * float(after["no_legal_pushes"])
            - self.release_action_macro_step_penalty * float(depth)
        )
        macro_admissible = bool(dependency_progress and goal_delta >= 0)
        summary = {
            "score": float(score),
            "macro_admissible": bool(macro_admissible),
            "dependency_progress": bool(dependency_progress),
            "opened_release": bool(opened_release),
            "release_action_delta": int(release_delta),
            "selected_release_action_delta": int(selected_release_delta),
            "stand_occupied_resolved": bool(stand_resolved),
            "locked_stand_blocker_delta": int(locked_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_delta),
            "legal_push_count_delta": int(legal_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "goal_delta": int(goal_delta),
            "after_low_legal_margin": bool(after["low_legal_margin"]),
            "after_no_legal_pushes": bool(after["no_legal_pushes"]),
        }
        return float(score), summary

    def _select_release_action_dependency_row(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any] | None:
        stage = self._decision_stage_summary(state, commitment_state)
        if str(stage.get("stage")) != "release":
            return None
        dependency = self._select_dependency_bound_row(state, commitment_state)
        if dependency is not None and (
            "stand_cell_occupied_by_box"
            in set(dependency.get("selected_dependency_kinds", []))
        ):
            return dependency
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        boxes = set(state.boxes)
        candidates = []
        for row in schedule.get("dependencies", []):
            kinds = set(row.get("dependency_kinds", []))
            stand_raw = row.get("stand_cell")
            if "stand_cell_occupied_by_box" not in kinds or stand_raw is None:
                continue
            if tuple(stand_raw) not in boxes:
                continue
            row_dict = self._dependency_to_dict(row)
            row_dict["selected_dependency_kinds"] = sorted(
                self._dependency_relevant_kinds(row)
            )
            candidates.append(row_dict)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                int(not bool(row.get("future", False))),
                float(row.get("candidate_barrier_cost", 0.0)),
                -int(row.get("horizon_index", 0)),
            ),
        )

    def _best_release_action_suffix(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        reference_metrics: dict[str, Any],
        dependency: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.release_action_macro_max_extra_pushes <= 0:
            return None
        best: dict[str, Any] | None = None
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "actions": [],
                "commitment_states": [],
                "rows": [],
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        expanded = 0
        while queue and expanded < self.release_action_macro_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.release_action_macro_max_extra_pushes:
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                key = self.model.gft_canonical_key_cached(child)
                if key in node["seen"]:
                    continue
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._release_action_macro_metrics(
                    child,
                    child_commitment,
                    dependency,
                )
                child_depth = depth + 1
                score, progress = self._release_action_macro_score(
                    reference_metrics,
                    metrics,
                    depth=child_depth,
                )
                source, target = self._moved_cells(node["state"], action)
                row = {
                    "macro_index": int(child_depth),
                    "physical_action": self._physical_action_dict(
                        node["state"],
                        action,
                    ),
                    "reason": self.release_action_macro_reason,
                    "source": list(source),
                    "target": list(target),
                    "bound_dependency": dependency,
                    "release_action_reference_metrics": reference_metrics,
                    "release_action_metrics": metrics,
                    "release_action_progress": progress,
                    "release_action_macro_score": float(score),
                    "recovery_score": float(score),
                }
                candidate = {
                    "score": float(score),
                    "macro_admissible": bool(progress["macro_admissible"]),
                    "opened_release": bool(progress["opened_release"]),
                    "dependency_progress": bool(progress["dependency_progress"]),
                    "actions": [*node["actions"], action],
                    "commitment_states": [*node["commitment_states"], child_commitment],
                    "rows": [*node["rows"], row],
                    "final_metrics": metrics,
                    "final_progress": progress,
                }
                if best is None or (
                    int(candidate["macro_admissible"]),
                    int(candidate["opened_release"]),
                    int(candidate["dependency_progress"]),
                    candidate["score"],
                    -len(candidate["actions"]),
                ) > (
                    int(best["macro_admissible"]),
                    int(best["opened_release"]),
                    int(best["dependency_progress"]),
                    best["score"],
                    -len(best["actions"]),
                ):
                    best = candidate
                if child_depth < self.release_action_macro_max_extra_pushes and not metrics["no_legal_pushes"]:
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate["actions"],
                            "commitment_states": candidate["commitment_states"],
                            "rows": candidate["rows"],
                            "seen": set(node["seen"]) | {key},
                        }
                    )
                expanded += 1
                if expanded >= self.release_action_macro_max_nodes:
                    break
        if best is None:
            return None
        if not bool(best.get("macro_admissible", False)):
            return None
        if float(best["score"]) < self.release_action_macro_score_threshold:
            return None
        return best

    def _release_action_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        base = plan.get("corridor_macro", {})
        if not self.release_action_macro_enabled:
            return base
        if plan.get("physical_action") is None:
            return base
        stage = self._decision_stage_summary(state, commitment_state)
        if str(stage.get("stage")) != "release":
            return {
                **base,
                "release_action_macro_enabled": True,
                "release_action_macro_triggered": False,
                "release_action_macro_reason": "not_release_stage",
            }
        if int(stage.get("release_action_count", 0)) > 0:
            return {
                **base,
                "release_action_macro_enabled": True,
                "release_action_macro_triggered": False,
                "release_action_macro_reason": "release_action_already_open",
            }
        dependency = self._select_release_action_dependency_row(state, commitment_state)
        if dependency is None:
            return {
                **base,
                "release_action_macro_enabled": True,
                "release_action_macro_triggered": False,
                "release_action_macro_reason": "no_release_dependency",
            }

        base_actions = list(base.get("physical_actions", []))
        if not base_actions:
            base_actions = [plan["physical_action"]]
        base_commitments = list(base.get("after_commitment_states", []))
        current_state = state
        current_commitment = commitment_state
        for idx, action in enumerate(base_actions):
            child = self.model.step_grid_flow(current_state, action)
            if idx < len(base_commitments):
                current_commitment = self.model.normalize_room_commitment_state(
                    child,
                    base_commitments[idx],
                )
            else:
                current_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
                base_commitments.append(current_commitment)
            current_state = child

        reference_metrics = self._release_action_macro_metrics(
            current_state,
            current_commitment,
            dependency,
        )
        suffix = self._best_release_action_suffix(
            current_state,
            current_commitment,
            reference_metrics,
            dependency,
        )
        if suffix is None:
            return {
                **base,
                "release_action_macro_enabled": True,
                "release_action_macro_triggered": False,
                "release_action_macro_reason": "no_admissible_release_suffix",
                "release_action_bound_dependency": dependency,
                "release_action_reference_metrics": reference_metrics,
            }

        base_rows = list(base.get("rows", []))
        offset_rows = []
        for offset, row in enumerate(suffix["rows"], start=len(base_actions)):
            offset_rows.append({**row, "macro_index": int(offset)})
        sequence = [*base_actions, *suffix["actions"]]
        commitment_states = [*base_commitments, *suffix["commitment_states"]]
        return {
            **base,
            "enabled": True,
            "triggered": True,
            "reason": "release_action_macro",
            "physical_actions": sequence,
            "after_commitment_states": commitment_states,
            "rows": [*base_rows, *offset_rows],
            "macro_push_count": int(len(sequence)),
            "extra_push_count": int(max(0, len(sequence) - 1)),
            "release_action_macro_enabled": True,
            "release_action_macro_triggered": True,
            "release_action_macro_score": float(suffix["score"]),
            "release_action_extra_push_count": int(len(suffix["actions"])),
            "release_action_bound_dependency": dependency,
            "release_action_reference_metrics": reference_metrics,
            "release_action_final_metrics": suffix["final_metrics"],
            "release_action_final_progress": suffix["final_progress"],
        }

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(
            state,
            commitment_state,
        )
        plan = super().plan_with_corridor_macro(state, commitment_state)
        plan["corridor_macro"] = self._release_action_macro_from_plan(
            state,
            commitment_state,
            plan,
        )
        return plan


class RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner(
    RoomScheduleReleaseActionMacroGFTSDMCTSPlanner
):
    """A30 planner with controlled release-stage goal regression.

    A29 shows that the release hook can bind the remaining stand-cell
    dependency, but no suffix is admissible when completed goals may not
    regress.  A30 keeps the same entry point and allows a bounded temporary
    goal regression only when the suffix also creates direct release progress.
    """

    provider_id = "room_schedule_controlled_release_regression_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        release_action_macro_max_extra_pushes: int = 8,
        release_action_macro_max_nodes: int = 640,
        release_action_macro_score_threshold: float = 12.0,
        release_action_macro_goal_regression_penalty: float = 80.0,
        controlled_release_goal_regression_allowed: bool = True,
        controlled_release_max_goal_regression: int = 1,
        controlled_release_goal_regression_penalty: float = 55.0,
        controlled_release_regression_progress_bonus: float = 35.0,
        controlled_release_require_direct_progress: bool = True,
        controlled_release_truncate_goal_lock: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            release_action_macro_max_extra_pushes=release_action_macro_max_extra_pushes,
            release_action_macro_max_nodes=release_action_macro_max_nodes,
            release_action_macro_score_threshold=release_action_macro_score_threshold,
            release_action_macro_goal_regression_penalty=(
                release_action_macro_goal_regression_penalty
            ),
            **kwargs,
        )
        self.controlled_release_goal_regression_allowed = bool(
            controlled_release_goal_regression_allowed
        )
        self.controlled_release_max_goal_regression = max(
            int(controlled_release_max_goal_regression),
            0,
        )
        self.controlled_release_goal_regression_penalty = float(
            controlled_release_goal_regression_penalty
        )
        self.controlled_release_regression_progress_bonus = float(
            controlled_release_regression_progress_bonus
        )
        self.controlled_release_require_direct_progress = bool(
            controlled_release_require_direct_progress
        )
        self.controlled_release_truncate_goal_lock = bool(
            controlled_release_truncate_goal_lock
        )

    def _release_action_macro_score(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        depth: int,
    ) -> tuple[float, dict[str, Any]]:
        score, summary = super()._release_action_macro_score(
            before,
            after,
            depth=depth,
        )
        goal_regression = max(0, -int(summary["goal_delta"]))
        direct_release_progress = bool(
            summary["opened_release"]
            or summary["stand_occupied_resolved"]
            or int(summary["locked_stand_blocker_delta"]) > 0
            or int(summary["selected_release_action_delta"]) > 0
        )
        controlled_admissible = bool(
            self.controlled_release_goal_regression_allowed
            and goal_regression > 0
            and goal_regression <= self.controlled_release_max_goal_regression
            and bool(summary["dependency_progress"])
            and not bool(summary["after_no_legal_pushes"])
            and (
                direct_release_progress
                or not self.controlled_release_require_direct_progress
            )
        )
        if controlled_admissible:
            # Replace A29's hard no-regression penalty with a smaller bounded
            # regression cost, and only after direct release progress is present.
            score += self.release_action_macro_goal_regression_penalty * float(
                goal_regression
            )
            score -= self.controlled_release_goal_regression_penalty * float(
                goal_regression
            )
            score += self.controlled_release_regression_progress_bonus * float(
                direct_release_progress
            )
            summary["macro_admissible"] = True
        summary.update(
            {
                "score": float(score),
                "goal_regression": int(goal_regression),
                "direct_release_progress": bool(direct_release_progress),
                "controlled_goal_regression": bool(controlled_admissible),
                "controlled_release_max_goal_regression": int(
                    self.controlled_release_max_goal_regression
                ),
            }
        )
        return float(score), summary

    def _trim_goal_locking_release_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.controlled_release_truncate_goal_lock:
            return macro
        if not bool(macro.get("triggered", False)):
            return macro
        stage = self._decision_stage_summary(state, commitment_state)
        if str(stage.get("stage")) != "release":
            return macro
        actions = list(macro.get("physical_actions", []))
        commitments = list(macro.get("after_commitment_states", []))
        if len(actions) <= 1:
            return macro

        current_state = state
        current_commitment = commitment_state
        per_step: list[tuple[SokobanState, RoomCommitmentState, dict[str, Any]]] = []
        for idx, action in enumerate(actions):
            child = self.model.step_grid_flow(current_state, action)
            if idx < len(commitments):
                child_commitment = self.model.normalize_room_commitment_state(
                    child,
                    commitments[idx],
                )
            else:
                child_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
            metrics = self._stand_lock_metrics(child, child_commitment)
            per_step.append((child, child_commitment, metrics))
            current_state = child
            current_commitment = child_commitment

        final_metrics = per_step[-1][2]
        if int(final_metrics.get("goal_locked_stand_blocker_count", 0)) <= 0:
            return macro
        if int(final_metrics.get("release_action_count", 0)) > 0:
            return macro

        keep = 1
        kept_state, kept_commitment, kept_metrics = per_step[0]
        rows = list(macro.get("rows", []))[:keep]
        return {
            **macro,
            "triggered": keep >= self.corridor_macro_min_total_pushes,
            "reason": "controlled_release_truncated_goal_lock",
            "physical_actions": actions[:keep],
            "after_commitment_states": [kept_commitment],
            "rows": rows,
            "macro_push_count": int(keep),
            "extra_push_count": int(max(0, keep - 1)),
            "controlled_release_truncated_goal_lock": True,
            "controlled_release_original_macro_push_count": int(len(actions)),
            "controlled_release_goal_lock_final_metrics": final_metrics,
            "controlled_release_goal_lock_kept_metrics": kept_metrics,
            "controlled_release_goal_lock_kept_boxes_on_goals": int(
                self.model.boxes_on_goals(kept_state)
            ),
        }

    def _release_action_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        macro = super()._release_action_macro_from_plan(
            state,
            commitment_state,
            plan,
        )
        return self._trim_goal_locking_release_macro(state, commitment_state, macro)


class RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner(
    RoomScheduleControlledReleaseRegressionMacroGFTSDMCTSPlanner
):
    """A31 planner that moves A30's goal-lock signal to root replacement.

    A30 can detect that a selected release macro would create an unreleasable
    goal-locked stand blocker, but it only truncates the macro after the root
    action has already been selected.  A31 keeps that diagnostic and, when the
    truncation signal appears, replaces the root action with a conservative
    same-root candidate that avoids immediate goal-locking and preserves the
    completed-goal count.
    """

    provider_id = "room_schedule_goal_lock_predecessor_release_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        goal_lock_predecessor_guard_enabled: bool = True,
        goal_lock_predecessor_require_goal_nonregression: bool = True,
        goal_lock_predecessor_require_no_goal_lock: bool = True,
        goal_lock_predecessor_min_structural_progress: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_lock_predecessor_guard_enabled = bool(
            goal_lock_predecessor_guard_enabled
        )
        self.goal_lock_predecessor_require_goal_nonregression = bool(
            goal_lock_predecessor_require_goal_nonregression
        )
        self.goal_lock_predecessor_require_no_goal_lock = bool(
            goal_lock_predecessor_require_no_goal_lock
        )
        self.goal_lock_predecessor_min_structural_progress = bool(
            goal_lock_predecessor_min_structural_progress
        )

    def _goal_lock_predecessor_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        dependency: dict[str, Any] | None,
    ) -> dict[str, Any]:
        release = self._release_action_macro_metrics(
            state,
            commitment_state,
            dependency,
        )
        stand_lock = self._stand_lock_metrics(state, commitment_state)
        stage = self._decision_stage_summary(state, commitment_state)
        return {
            **release,
            "stage": str(stage.get("stage")),
            "goal_locked_stand_blocker_count": int(
                stand_lock["goal_locked_stand_blocker_count"]
            ),
            "occupied_critical_stand_count": int(
                stand_lock["occupied_critical_stand_count"]
            ),
            "stand_release_option_count": int(stand_lock["release_option_count"]),
        }

    def _goal_lock_predecessor_progress(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "goal_delta": int(after["boxes_on_goals"]) - int(before["boxes_on_goals"]),
            "dependency_total_delta": int(after["dependency_total"])
            - int(before["dependency_total"]),
            "target_blocker_count_delta": int(after["target_blocker_count"])
            - int(before["target_blocker_count"]),
            "release_action_count_delta": int(after["release_action_count"])
            - int(before["release_action_count"]),
            "selected_release_action_delta": int(after["selected_release_action_count"])
            - int(before["selected_release_action_count"]),
            "locked_stand_blocker_delta": int(after["locked_stand_blocker_count"])
            - int(before["locked_stand_blocker_count"]),
            "goal_locked_stand_blocker_delta": int(
                after["goal_locked_stand_blocker_count"]
            )
            - int(before["goal_locked_stand_blocker_count"]),
            "legal_push_count_delta": int(after["nonblocked_legal_push_count"])
            - int(before["nonblocked_legal_push_count"]),
            "slot_schedule_cost_delta": float(before["slot_schedule_cost"])
            - float(after["slot_schedule_cost"]),
        }

    def _goal_lock_predecessor_structural_progress(
        self,
        progress: dict[str, Any],
    ) -> bool:
        return bool(
            int(progress["release_action_count_delta"]) > 0
            or int(progress["selected_release_action_delta"]) > 0
            or int(progress["dependency_total_delta"]) < 0
            or int(progress["target_blocker_count_delta"]) < 0
            or int(progress["locked_stand_blocker_delta"]) < 0
            or int(progress["legal_push_count_delta"]) > 0
            or float(progress["slot_schedule_cost_delta"]) > 0.0
        )

    def _goal_lock_predecessor_replacement_key(
        self,
        row: dict[str, Any],
    ) -> tuple:
        edge: RoomSemanticEdge = row["edge"]
        progress = row["progress"]
        metrics = row["metrics"]
        structural_progress = bool(row["structural_progress"])
        if self.select_by == "q":
            search_key = (edge.mean_value, edge.visits)
        else:
            search_key = (edge.visits, edge.mean_value)
        return (
            int(structural_progress),
            int(progress["release_action_count_delta"])
            + int(progress["selected_release_action_delta"]),
            -int(metrics["goal_locked_stand_blocker_count"]),
            -int(metrics["locked_stand_blocker_count"]),
            -int(metrics["dependency_total"]),
            -int(metrics["target_blocker_count"]),
            int(metrics["nonblocked_legal_push_count"]),
            int(progress["goal_delta"]),
            float(progress["slot_schedule_cost_delta"]),
            *search_key,
            float(edge.room_sd_value_score),
            float(edge.room_recoverability_score),
            float(edge.room_phase_progress_score),
            float(edge.prior_score),
        )

    def _goal_lock_predecessor_select_replacement(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        root: RoomSemanticNode,
        plan: dict[str, Any],
        macro: dict[str, Any],
    ) -> dict[str, Any] | None:
        dependency = macro.get("release_action_bound_dependency")
        before = self._goal_lock_predecessor_metrics(
            state,
            commitment_state,
            dependency,
        )
        selected_key = None
        if plan.get("semantic_action") is not None:
            selected_key = plan["semantic_action"].key
        candidates: list[dict[str, Any]] = []
        rejected = {
            "same_semantic_action": 0,
            "goal_regression": 0,
            "goal_lock_after": 0,
            "no_structural_progress": 0,
        }
        for edge in root.children.values():
            if selected_key is not None and edge.action.key == selected_key:
                rejected["same_semantic_action"] += 1
                continue
            metrics = self._goal_lock_predecessor_metrics(
                edge.child_state,
                edge.child_commitment_state,
                dependency,
            )
            progress = self._goal_lock_predecessor_progress(before, metrics)
            structural_progress = self._goal_lock_predecessor_structural_progress(
                progress
            )
            if (
                self.goal_lock_predecessor_require_goal_nonregression
                and int(progress["goal_delta"]) < 0
            ):
                rejected["goal_regression"] += 1
                continue
            if (
                self.goal_lock_predecessor_require_no_goal_lock
                and int(metrics["goal_locked_stand_blocker_count"]) > 0
            ):
                rejected["goal_lock_after"] += 1
                continue
            if (
                self.goal_lock_predecessor_min_structural_progress
                and not structural_progress
            ):
                rejected["no_structural_progress"] += 1
                continue
            candidates.append(
                {
                    "edge": edge,
                    "metrics": metrics,
                    "progress": progress,
                    "structural_progress": bool(structural_progress),
                }
            )
        if not candidates:
            return {
                "replacement": None,
                "before_metrics": before,
                "candidate_count": 0,
                "rejected_counts": rejected,
            }
        best = max(candidates, key=self._goal_lock_predecessor_replacement_key)
        return {
            "replacement": best,
            "before_metrics": before,
            "candidate_count": len(candidates),
            "rejected_counts": rejected,
        }

    def _goal_lock_predecessor_replace_plan(
        self,
        state: SokobanState,
        plan: dict[str, Any],
        macro: dict[str, Any],
        selection: dict[str, Any],
    ) -> dict[str, Any]:
        replacement = selection["replacement"]
        edge: RoomSemanticEdge = replacement["edge"]
        action = edge.action.physical_action
        row = {
            "macro_index": 0,
            "physical_action": self._physical_action_dict(state, action),
            "reason": "goal_lock_predecessor_root_replacement",
        }
        replacement_macro = {
            **macro,
            "enabled": True,
            "triggered": False,
            "reason": "goal_lock_predecessor_root_replacement",
            "physical_actions": [action],
            "after_commitment_states": [edge.child_commitment_state],
            "rows": [row],
            "macro_push_count": 1,
            "extra_push_count": 0,
            "goal_lock_predecessor_guard_enabled": True,
            "goal_lock_predecessor_guard_triggered": True,
            "goal_lock_predecessor_guard_reason": "replacement_selected",
            "goal_lock_predecessor_candidate_count": int(
                selection["candidate_count"]
            ),
            "goal_lock_predecessor_rejected_counts": selection["rejected_counts"],
            "goal_lock_predecessor_original_action": (
                None
                if plan.get("physical_action") is None
                else self._physical_action_dict(state, plan["physical_action"])
            ),
            "goal_lock_predecessor_replacement_action": self._physical_action_dict(
                state,
                action,
            ),
            "goal_lock_predecessor_original_semantic_key": (
                None
                if plan.get("semantic_action") is None
                else list(plan["semantic_action"].key)
            ),
            "goal_lock_predecessor_replacement_semantic_key": list(edge.action.key),
            "goal_lock_predecessor_before_metrics": selection["before_metrics"],
            "goal_lock_predecessor_original_final_metrics": macro.get(
                "controlled_release_goal_lock_final_metrics"
            ),
            "goal_lock_predecessor_replacement_metrics": replacement["metrics"],
            "goal_lock_predecessor_replacement_progress": replacement["progress"],
            "goal_lock_predecessor_replacement_structural_progress": bool(
                replacement["structural_progress"]
            ),
            "goal_lock_predecessor_replacement_score_key": list(
                self._goal_lock_predecessor_replacement_key(replacement)
            ),
        }
        return {
            **plan,
            "semantic_action": edge.action,
            "physical_action": action,
            "next_commitment_state": edge.child_commitment_state,
            "corridor_macro": replacement_macro,
        }

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(
            state,
            commitment_state,
        )
        plan = super().plan_with_corridor_macro(state, commitment_state)
        macro = plan.get("corridor_macro", {})
        if not self.goal_lock_predecessor_guard_enabled:
            macro["goal_lock_predecessor_guard_enabled"] = False
            plan["corridor_macro"] = macro
            return plan
        if not bool(macro.get("controlled_release_truncated_goal_lock", False)):
            macro.setdefault("goal_lock_predecessor_guard_enabled", True)
            macro.setdefault("goal_lock_predecessor_guard_triggered", False)
            macro.setdefault("goal_lock_predecessor_guard_reason", "no_goal_lock_truncation")
            plan["corridor_macro"] = macro
            return plan

        root = self.node_for(state, commitment_state, 0)
        selection = self._goal_lock_predecessor_select_replacement(
            state,
            commitment_state,
            root,
            plan,
            macro,
        )
        if selection is None or selection.get("replacement") is None:
            macro["goal_lock_predecessor_guard_enabled"] = True
            macro["goal_lock_predecessor_guard_triggered"] = False
            macro["goal_lock_predecessor_guard_reason"] = "no_admissible_replacement"
            if selection is not None:
                macro["goal_lock_predecessor_candidate_count"] = int(
                    selection["candidate_count"]
                )
                macro["goal_lock_predecessor_rejected_counts"] = selection[
                    "rejected_counts"
                ]
                macro["goal_lock_predecessor_before_metrics"] = selection[
                    "before_metrics"
                ]
            plan["corridor_macro"] = macro
            return plan
        return self._goal_lock_predecessor_replace_plan(
            state,
            plan,
            macro,
            selection,
        )


class RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner(
    RoomScheduleGoalLockPredecessorReleaseMacroGFTSDMCTSPlanner
):
    """A32 planner with a compact learned SD-consumer interface.

    A32 does not add another hand-written Sokoban rule.  It exposes the small
    set of SD mechanism variables that survived A24-A31 diagnostics as a stable
    feature vector, then consumes an optional offline-trained checkpoint to
    adjust root ranking.  Without a checkpoint, A32 is intentionally identical
    to A31 except for traceable feature summaries.
    """

    provider_id = "room_schedule_learned_sd_consumer_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        learned_sd_consumer_checkpoint: str | None = None,
        learned_sd_consumer_enabled: bool = True,
        learned_sd_consumer_weight: float = 1.0,
        learned_sd_consumer_override_margin: float = 1e-6,
        learned_sd_consumer_root_override: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learned_sd_consumer_enabled = bool(learned_sd_consumer_enabled)
        self.learned_sd_consumer_weight = float(learned_sd_consumer_weight)
        self.learned_sd_consumer_override_margin = float(
            learned_sd_consumer_override_margin
        )
        self.learned_sd_consumer_root_override = bool(
            learned_sd_consumer_root_override
        )
        self.learned_sd_consumer_checkpoint_path = learned_sd_consumer_checkpoint
        self.learned_sd_consumer_payload = (
            self._load_learned_sd_consumer_checkpoint(
                learned_sd_consumer_checkpoint,
            )
            if learned_sd_consumer_checkpoint
            else None
        )

    def _load_learned_sd_consumer_checkpoint(
        self,
        path: str,
    ) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        feature_names = tuple(payload.get("feature_names", ()))
        if feature_names != LEARNED_SD_CONSUMER_FEATURE_NAMES:
            raise ValueError(
                "Learned SD consumer feature contract mismatch: "
                f"checkpoint has {len(feature_names)} features, "
                f"planner expects {len(LEARNED_SD_CONSUMER_FEATURE_NAMES)}."
            )
        return payload

    @staticmethod
    def _learned_sd_consumer_scalar(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _learned_sd_consumer_feature_dict(
        self,
        edge: RoomSemanticEdge,
    ) -> dict[str, float]:
        phase = edge.room_phase_progress_summary
        sdv = edge.room_sd_value_summary
        trap = edge.room_pre_release_trap_summary
        sdg = edge.room_sd_generalization_summary
        recovery = edge.room_recoverability_summary
        stand_lock = edge.room_stand_lock_summary
        stage = edge.room_stage_summary
        before_stage = str(stage.get("before", {}).get("stage", ""))
        after_stage = str(stage.get("after", {}).get("stage", ""))
        after = trap.get("after", {})
        after_stand = stand_lock.get("after", {})
        action = edge.action.physical_action
        max_box = max(self.model.level.n_boxes - 1, 1)
        max_action = max(self.model.level.n_boxes * len(ACTION_NAMES) - 1, 1)
        low_margin = bool(
            int(after.get("dependency_total", 0)) > 0
            and int(after.get("nonblocked_legal_push_count", 0))
            <= self.sd_value_legal_margin
        )
        direction_features = {
            f"action_direction_{name}": float(action.direction == idx)
            for idx, name in enumerate(ACTION_NAMES)
        }
        features = {
            "bias": 1.0,
            "goal_delta": self._learned_sd_consumer_scalar(phase.get("goal_delta")),
            "release_action_count_delta": self._learned_sd_consumer_scalar(
                phase.get("release_action_count_delta")
            ),
            "dependency_total_delta": self._learned_sd_consumer_scalar(
                phase.get("dependency_total_delta")
            ),
            "target_blocker_count_delta": self._learned_sd_consumer_scalar(
                phase.get("target_blocker_count_delta")
            ),
            "slot_schedule_cost_delta": self._learned_sd_consumer_scalar(
                phase.get("slot_schedule_cost_delta")
            ),
            "schedule_barrier_cost_delta": self._learned_sd_consumer_scalar(
                phase.get("schedule_barrier_cost_delta")
            ),
            "legal_push_count_delta": self._learned_sd_consumer_scalar(
                trap.get("legal_push_count_delta")
            ),
            "after_legal_push_count": self._learned_sd_consumer_scalar(
                after.get("nonblocked_legal_push_count")
            ),
            "after_dependency_total": self._learned_sd_consumer_scalar(
                after.get("dependency_total")
            ),
            "after_target_blocker_count": self._learned_sd_consumer_scalar(
                after.get("target_blocker_count")
            ),
            "after_release_action_count": self._learned_sd_consumer_scalar(
                after.get("release_action_count")
            ),
            "after_goal_locked_stand_blocker_count": self._learned_sd_consumer_scalar(
                after_stand.get("goal_locked_stand_blocker_count")
            ),
            "after_locked_stand_blocker_count": self._learned_sd_consumer_scalar(
                after_stand.get("locked_stand_blocker_count")
            ),
            "after_stand_release_option_count": self._learned_sd_consumer_scalar(
                after_stand.get("release_option_count")
            ),
            "stage_before_fill": float(before_stage == "fill"),
            "stage_before_release": float(before_stage == "release"),
            "stage_before_recovery": float(before_stage == "recovery"),
            "stage_after_release": float(after_stage == "release"),
            "stage_after_recovery": float(after_stage == "recovery"),
            "phase_progress": float(bool(phase.get("phase_progress"))),
            "no_progress": float(bool(phase.get("no_progress"))),
            "stagnant_context": float(bool(phase.get("stagnant_context"))),
            "repeated_schedule_signature": float(
                bool(phase.get("child_repeated_schedule_signature"))
            ),
            "repeated_edge": float(bool(phase.get("repeated_edge"))),
            "reverse_recent_edge": float(bool(phase.get("reverse_recent_edge"))),
            "no_release_context": float(bool(phase.get("no_release_context"))),
            "pseudo_milestone_loop": float(bool(sdv.get("pseudo_milestone_loop"))),
            "hard_negative": float(bool(sdv.get("hard_negative"))),
            "soft_negative": float(bool(sdv.get("soft_negative"))),
            "trap_risk": float(bool(trap.get("trap_risk"))),
            "severe_trap_risk": float(bool(trap.get("severe_trap_risk"))),
            "generalized_failure_risk": float(
                bool(sdg.get("generalized_failure_risk"))
            ),
            "trap_predecessor": float(
                bool(sdg.get("successor_probe", {}).get("trap_predecessor"))
            ),
            "stand_lock_risk": float(bool(stand_lock.get("stand_lock_risk"))),
            "recoverable_no_progress": float(
                bool(recovery.get("recoverable_no_progress"))
            ),
            "unrecoverable_no_progress": float(
                bool(recovery.get("unrecoverable_no_progress"))
            ),
            "low_margin": float(low_margin),
            "action_box_index_norm": float(action.box_index) / float(max_box),
            "action_flat_index_norm": float(action.flat_index) / float(max_action),
        }
        features.update(direction_features)
        return {
            name: float(features.get(name, 0.0))
            for name in LEARNED_SD_CONSUMER_FEATURE_NAMES
        }

    def _learned_sd_consumer_forward(
        self,
        feature_dict: dict[str, float],
    ) -> float:
        payload = self.learned_sd_consumer_payload
        if payload is None:
            return 0.0
        vector = [float(feature_dict[name]) for name in LEARNED_SD_CONSUMER_FEATURE_NAMES]
        model_type = str(payload.get("model_type", "linear"))
        if model_type == "linear":
            weights = [float(value) for value in payload.get("weights", [])]
            if len(weights) != len(vector):
                raise ValueError("Learned SD linear checkpoint weight dimension mismatch.")
            bias = float(payload.get("bias", 0.0))
            return float(sum(w * x for w, x in zip(weights, vector)) + bias)
        if model_type == "mlp":
            mean = [float(value) for value in payload.get("mean", [0.0] * len(vector))]
            std = [
                max(float(value), 1e-6)
                for value in payload.get("std", [1.0] * len(vector))
            ]
            activations = [(x - m) / s for x, m, s in zip(vector, mean, std)]
            for layer_index, layer in enumerate(payload.get("layers", [])):
                weights = layer["w"]
                biases = layer["b"]
                next_values = []
                for row, bias in zip(weights, biases):
                    next_values.append(
                        float(sum(float(w) * x for w, x in zip(row, activations)) + float(bias))
                    )
                if layer_index < len(payload.get("layers", [])) - 1:
                    next_values = [max(0.0, value) for value in next_values]
                activations = next_values
            return float(activations[0]) if activations else 0.0
        raise ValueError(f"Unsupported learned SD consumer model_type: {model_type}")

    def _learned_sd_consumer_score(
        self,
        candidate: RoomSemanticCandidate,
    ) -> float:
        feature_dict = self._learned_sd_consumer_feature_dict_from_candidate(candidate)
        raw_score = self._learned_sd_consumer_forward(feature_dict)
        score = self.learned_sd_consumer_weight * raw_score
        candidate.room_sd_value_score += float(score)
        candidate.room_sd_value_summary = {
            **candidate.room_sd_value_summary,
            "learned_sd_consumer_enabled": bool(self.learned_sd_consumer_enabled),
            "learned_sd_consumer_checkpoint_loaded": bool(
                self.learned_sd_consumer_payload is not None
            ),
            "learned_sd_consumer_raw_score": float(raw_score),
            "learned_sd_consumer_score": float(score),
            "learned_sd_consumer_feature_contract": "room_sd_candidate_features_v1",
            "learned_sd_consumer_root_override_enabled": bool(
                self.learned_sd_consumer_root_override
            ),
            "learned_sd_consumer_features": feature_dict,
        }
        return float(score)

    def _learned_sd_consumer_feature_dict_from_candidate(
        self,
        candidate: RoomSemanticCandidate,
    ) -> dict[str, float]:
        edge = RoomSemanticEdge(
            action=candidate.action,
            child_state=candidate.child_state,
            child_commitment_state=candidate.child_commitment_state,
            child_node=RoomSemanticNode(
                candidate.child_state,
                candidate.child_commitment_state,
                0,
            ),
            room_phase_progress_summary=candidate.room_phase_progress_summary,
            room_pre_release_trap_summary=candidate.room_pre_release_trap_summary,
            room_sd_generalization_summary=candidate.room_sd_generalization_summary,
            room_recoverability_summary=candidate.room_recoverability_summary,
            room_stand_lock_summary=candidate.room_stand_lock_summary,
            room_stage_summary=candidate.room_stage_summary,
            room_sd_value_summary=candidate.room_sd_value_summary,
        )
        return self._learned_sd_consumer_feature_dict(edge)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        if not self.learned_sd_consumer_enabled:
            return float(base)
        return float(base + self._learned_sd_consumer_score(candidate))

    def _learned_sd_choice_key(self, edge: RoomSemanticEdge) -> tuple:
        feature_dict = self._learned_sd_consumer_feature_dict(edge)
        raw_score = self._learned_sd_consumer_forward(feature_dict)
        score = self.learned_sd_consumer_weight * raw_score
        return (
            float(score),
            float(edge.room_sd_value_score),
            bool(edge.room_sd_value_summary.get("milestone_progress")),
            not bool(edge.room_sd_value_summary.get("pseudo_milestone_loop")),
            not bool(edge.room_sd_value_summary.get("hard_negative")),
            edge.visits,
            edge.mean_value,
            edge.prior_score,
        )

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        best = super().choose_root_edge(root)
        if (
            not self.learned_sd_consumer_enabled
            or self.learned_sd_consumer_payload is None
            or not root.children
            or not self.learned_sd_consumer_root_override
        ):
            return best
        learned_best = max(root.children.values(), key=self._learned_sd_choice_key)
        best_score = self._learned_sd_choice_key(best)[0]
        learned_score = self._learned_sd_choice_key(learned_best)[0]
        if learned_score > best_score + self.learned_sd_consumer_override_margin:
            return learned_best
        return best


class RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner(
    RoomScheduleLearnedSDConsumerMacroGFTSDMCTSPlanner
):
    """A33 planner with a calibrated neural SD-consumer checkpoint.

    A33 keeps A32's learned-consumer boundary but treats the checkpoint as a
    neural function approximator rather than a hard rule.  Runtime planning
    still uses exact Sokoban dynamics and online teacher/reference actions are
    not queried.
    """

    provider_id = "room_schedule_neural_sd_consumer_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        neural_sd_consumer_score_clip: float = 4.0,
        **kwargs,
    ):
        kwargs.setdefault("learned_sd_consumer_weight", 0.35)
        kwargs.setdefault("learned_sd_consumer_root_override", False)
        super().__init__(*args, **kwargs)
        self.neural_sd_consumer_score_clip = float(neural_sd_consumer_score_clip)

    def _learned_sd_consumer_feature_dict(
        self,
        edge: RoomSemanticEdge,
    ) -> dict[str, float]:
        features = super()._learned_sd_consumer_feature_dict(edge)
        scaling = (
            self.learned_sd_consumer_payload.get("feature_scaling", {})
            if self.learned_sd_consumer_payload is not None
            else {}
        )
        box_denominator = float(
            scaling.get(
                "action_box_index_norm_denominator",
                max(self.model.level.n_boxes - 1, 1),
            )
        )
        flat_denominator = float(
            scaling.get(
                "action_flat_index_norm_denominator",
                max(self.model.level.n_boxes * len(ACTION_NAMES) - 1, 1),
            )
        )
        if box_denominator > 0.0:
            features["action_box_index_norm"] = float(
                edge.action.physical_action.box_index
            ) / box_denominator
        if flat_denominator > 0.0:
            features["action_flat_index_norm"] = float(
                edge.action.physical_action.flat_index
            ) / flat_denominator
        return features

    def _learned_sd_consumer_score(
        self,
        candidate: RoomSemanticCandidate,
    ) -> float:
        feature_dict = self._learned_sd_consumer_feature_dict_from_candidate(candidate)
        raw_score = self._learned_sd_consumer_forward(feature_dict)
        unclipped = self.learned_sd_consumer_weight * raw_score
        clip = max(0.0, self.neural_sd_consumer_score_clip)
        score = float(max(-clip, min(clip, unclipped))) if clip > 0.0 else float(unclipped)
        candidate.room_sd_value_score += float(score)
        candidate.room_sd_value_summary = {
            **candidate.room_sd_value_summary,
            "learned_sd_consumer_enabled": bool(self.learned_sd_consumer_enabled),
            "learned_sd_consumer_checkpoint_loaded": bool(
                self.learned_sd_consumer_payload is not None
            ),
            "learned_sd_consumer_raw_score": float(raw_score),
            "learned_sd_consumer_unclipped_score": float(unclipped),
            "learned_sd_consumer_score": float(score),
            "learned_sd_consumer_feature_contract": "room_sd_candidate_features_v1",
            "learned_sd_consumer_root_override_enabled": bool(
                self.learned_sd_consumer_root_override
            ),
            "neural_sd_consumer_enabled": True,
            "neural_sd_consumer_score_clip": float(clip),
            "learned_sd_consumer_features": feature_dict,
        }
        return float(score)


class RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner(
    RoomScheduleNeuralSDConsumerMacroGFTSDMCTSPlanner
):
    """A34 planner using a milestone-target neural SD consumer.

    The class exists to keep the A34 checkpoint and default calibration
    separate from A33.  It does not add Sokoban-specific action rules; only the
    learned consumer's checkpoint, scale, and trace identity change.
    """

    provider_id = "room_schedule_milestone_sd_consumer_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("learned_sd_consumer_weight", 0.35)
        kwargs.setdefault("learned_sd_consumer_root_override", False)
        kwargs.setdefault("neural_sd_consumer_score_clip", 3.0)
        super().__init__(*args, **kwargs)

    def _learned_sd_consumer_score(
        self,
        candidate: RoomSemanticCandidate,
    ) -> float:
        score = super()._learned_sd_consumer_score(candidate)
        candidate.room_sd_value_summary = {
            **candidate.room_sd_value_summary,
            "milestone_sd_consumer_enabled": True,
            "milestone_sd_consumer_target": "future_window_milestone_v1",
        }
        return float(score)


class RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner(
    RoomScheduleMilestoneSDConsumerMacroGFTSDMCTSPlanner
):
    """A35 planner with a decomposed multi-head SD consumer.

    A35 keeps the learned-consumer boundary introduced in A32-A34, but avoids
    collapsing distinct SD mechanisms into one training target.  The checkpoint
    predicts value, release-progress, loop-risk, and trap-risk heads; the
    planner combines them as a bounded soft prior.  It does not add new
    Sokoban action rules or query teacher/reference actions online.
    """

    provider_id = "room_schedule_multihead_sd_consumer_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        multihead_sd_consumer_value_weight: float = 0.45,
        multihead_sd_consumer_release_weight: float = 1.15,
        multihead_sd_consumer_loop_weight: float = 1.05,
        multihead_sd_consumer_trap_weight: float = 1.25,
        **kwargs,
    ):
        kwargs.setdefault("learned_sd_consumer_weight", 0.35)
        kwargs.setdefault("learned_sd_consumer_root_override", False)
        kwargs.setdefault("neural_sd_consumer_score_clip", 3.0)
        super().__init__(*args, **kwargs)
        self.multihead_sd_consumer_weights = {
            "value": float(multihead_sd_consumer_value_weight),
            "release_progress": float(multihead_sd_consumer_release_weight),
            "loop_risk": float(multihead_sd_consumer_loop_weight),
            "trap_risk": float(multihead_sd_consumer_trap_weight),
        }

    @staticmethod
    def _clip_multihead_value(value: float, bounds: Any) -> float:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            return float(value)
        low = float(bounds[0])
        high = float(bounds[1])
        return float(max(low, min(high, value)))

    def _multihead_sd_consumer_forward(
        self,
        feature_dict: dict[str, float],
    ) -> dict[str, float]:
        payload = self.learned_sd_consumer_payload
        if payload is None:
            return {
                "value": 0.0,
                "release_progress": 0.0,
                "loop_risk": 0.0,
                "trap_risk": 0.0,
            }
        model_type = str(payload.get("model_type", ""))
        if model_type != "multihead_mlp":
            return {"value": self._learned_sd_consumer_forward(feature_dict)}
        vector = [float(feature_dict[name]) for name in LEARNED_SD_CONSUMER_FEATURE_NAMES]
        mean = [float(value) for value in payload.get("mean", [0.0] * len(vector))]
        std = [
            max(float(value), 1e-6)
            for value in payload.get("std", [1.0] * len(vector))
        ]
        head_clips = payload.get("head_output_clips", {})
        result: dict[str, float] = {}
        for head_name, head_payload in payload.get("heads", {}).items():
            activations = [(x - m) / s for x, m, s in zip(vector, mean, std)]
            layers = head_payload.get("layers", [])
            for layer_index, layer in enumerate(layers):
                weights = layer["w"]
                biases = layer["b"]
                next_values = []
                for row, bias in zip(weights, biases):
                    next_values.append(
                        float(
                            sum(float(w) * x for w, x in zip(row, activations))
                            + float(bias)
                        )
                    )
                if layer_index < len(layers) - 1:
                    next_values = [max(0.0, value) for value in next_values]
                activations = next_values
            raw_value = float(activations[0]) if activations else 0.0
            result[str(head_name)] = self._clip_multihead_value(
                raw_value,
                head_clips.get(head_name),
            )
        for required in ("value", "release_progress", "loop_risk", "trap_risk"):
            result.setdefault(required, 0.0)
        return result

    def _multihead_sd_consumer_combined_raw(
        self,
        heads: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        components = {
            "value": self.multihead_sd_consumer_weights["value"]
            * float(heads.get("value", 0.0)),
            "release_progress": self.multihead_sd_consumer_weights[
                "release_progress"
            ]
            * float(heads.get("release_progress", 0.0)),
            "loop_risk": -self.multihead_sd_consumer_weights["loop_risk"]
            * float(heads.get("loop_risk", 0.0)),
            "trap_risk": -self.multihead_sd_consumer_weights["trap_risk"]
            * float(heads.get("trap_risk", 0.0)),
        }
        return float(sum(components.values())), components

    def _learned_sd_consumer_score(
        self,
        candidate: RoomSemanticCandidate,
    ) -> float:
        feature_dict = self._learned_sd_consumer_feature_dict_from_candidate(candidate)
        heads = self._multihead_sd_consumer_forward(feature_dict)
        raw_score, components = self._multihead_sd_consumer_combined_raw(heads)
        unclipped = self.learned_sd_consumer_weight * raw_score
        clip = max(0.0, self.neural_sd_consumer_score_clip)
        score = float(max(-clip, min(clip, unclipped))) if clip > 0.0 else float(unclipped)
        candidate.room_sd_value_score += float(score)
        candidate.room_sd_value_summary = {
            **candidate.room_sd_value_summary,
            "learned_sd_consumer_enabled": bool(self.learned_sd_consumer_enabled),
            "learned_sd_consumer_checkpoint_loaded": bool(
                self.learned_sd_consumer_payload is not None
            ),
            "learned_sd_consumer_raw_score": float(raw_score),
            "learned_sd_consumer_unclipped_score": float(unclipped),
            "learned_sd_consumer_score": float(score),
            "learned_sd_consumer_feature_contract": "room_sd_candidate_features_v1",
            "learned_sd_consumer_root_override_enabled": bool(
                self.learned_sd_consumer_root_override
            ),
            "neural_sd_consumer_enabled": True,
            "neural_sd_consumer_score_clip": float(clip),
            "milestone_sd_consumer_enabled": False,
            "multihead_sd_consumer_enabled": True,
            "multihead_sd_consumer_target": "value_release_loop_trap_v1",
            "multihead_sd_consumer_heads": {
                name: float(value) for name, value in heads.items()
            },
            "multihead_sd_consumer_components": {
                name: float(value) for name, value in components.items()
            },
            "multihead_sd_consumer_weights": dict(self.multihead_sd_consumer_weights),
            "learned_sd_consumer_features": feature_dict,
        }
        return float(score)

    def _learned_sd_choice_key(self, edge: RoomSemanticEdge) -> tuple:
        feature_dict = self._learned_sd_consumer_feature_dict(edge)
        heads = self._multihead_sd_consumer_forward(feature_dict)
        raw_score, _ = self._multihead_sd_consumer_combined_raw(heads)
        score = self.learned_sd_consumer_weight * raw_score
        return (
            float(score),
            float(heads.get("release_progress", 0.0)),
            -float(heads.get("loop_risk", 0.0)),
            -float(heads.get("trap_risk", 0.0)),
            float(edge.room_sd_value_score),
            bool(edge.room_sd_value_summary.get("milestone_progress")),
            not bool(edge.room_sd_value_summary.get("pseudo_milestone_loop")),
            not bool(edge.room_sd_value_summary.get("hard_negative")),
            edge.visits,
            edge.mean_value,
            edge.prior_score,
        )


class RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner(
    RoomScheduleMultiHeadSDConsumerMacroGFTSDMCTSPlanner
):
    """A36 planner with a minimal agenda progress-certificate gate.

    A36 tests the smallest Agenda-GFT claim after A35: release-progress is not
    true progress unless it is certified by the current structural obligation.
    The class reuses the A35 multi-head checkpoint and adds a bounded, auditable
    gate around the release-progress head.  It does not change action
    generation, exact Sokoban dynamics, or online teacher usage.
    """

    provider_id = "room_schedule_agenda_certificate_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        agenda_certificate_slot_delta_threshold: float = 0.5,
        agenda_certificate_release_open_cap: float = 0.35,
        agenda_certificate_bonus: float = 0.35,
        agenda_certificate_release_assigned_bonus: float = 0.15,
        agenda_certificate_release_consumed_bonus: float = 0.60,
        agenda_certificate_pseudo_penalty: float = 0.70,
        agenda_certificate_stale_release_penalty: float = 0.90,
        agenda_certificate_loop_penalty: float = 0.50,
        agenda_certificate_release_head_floor: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agenda_certificate_slot_delta_threshold = float(
            agenda_certificate_slot_delta_threshold
        )
        self.agenda_certificate_release_open_cap = float(
            agenda_certificate_release_open_cap
        )
        self.agenda_certificate_bonus = float(agenda_certificate_bonus)
        self.agenda_certificate_release_assigned_bonus = float(
            agenda_certificate_release_assigned_bonus
        )
        self.agenda_certificate_release_consumed_bonus = float(
            agenda_certificate_release_consumed_bonus
        )
        self.agenda_certificate_pseudo_penalty = float(
            agenda_certificate_pseudo_penalty
        )
        self.agenda_certificate_stale_release_penalty = float(
            agenda_certificate_stale_release_penalty
        )
        self.agenda_certificate_loop_penalty = float(
            agenda_certificate_loop_penalty
        )
        self.agenda_certificate_release_head_floor = float(
            agenda_certificate_release_head_floor
        )

    def _agenda_certificate_summary(
        self,
        *,
        phase: dict[str, Any],
        sdv: dict[str, Any],
        trap: dict[str, Any],
        stand_lock: dict[str, Any],
        heads: dict[str, float],
    ) -> dict[str, Any]:
        goal_delta = int(self._learned_sd_consumer_scalar(phase.get("goal_delta")))
        release_delta = int(
            self._learned_sd_consumer_scalar(
                phase.get("release_action_count_delta")
            )
        )
        dependency_delta = int(
            self._learned_sd_consumer_scalar(
                phase.get("dependency_total_delta")
            )
        )
        target_delta = int(
            self._learned_sd_consumer_scalar(
                phase.get("target_blocker_count_delta")
            )
        )
        slot_delta = float(
            self._learned_sd_consumer_scalar(
                phase.get("slot_schedule_cost_delta")
            )
        )
        barrier_delta = float(
            self._learned_sd_consumer_scalar(
                phase.get("schedule_barrier_cost_delta")
            )
        )
        legal_delta = int(
            self._learned_sd_consumer_scalar(trap.get("legal_push_count_delta"))
        )
        before_release = int(
            self._learned_sd_consumer_scalar(
                phase.get("before_release_action_count")
            )
        )
        after_release = int(
            self._learned_sd_consumer_scalar(
                phase.get("after_release_action_count")
            )
        )
        locked_delta = int(
            self._learned_sd_consumer_scalar(
                stand_lock.get("locked_stand_blocker_delta")
            )
        )
        release_head = float(heads.get("release_progress", 0.0))
        repeated_context = bool(
            phase.get("stagnant_context")
            or phase.get("child_repeated_schedule_signature")
            or phase.get("repeated_edge")
            or phase.get("reverse_recent_edge")
        )
        goal_progress = bool(goal_delta > 0)
        blocker_reduction = bool(
            dependency_delta > 0 or target_delta > 0 or locked_delta < 0
        )
        slot_frontier_progress = bool(
            slot_delta > self.agenda_certificate_slot_delta_threshold
        )
        barrier_progress = bool(barrier_delta > 4.0)
        release_opened = bool(release_delta > 0)
        release_assigned = bool(
            release_opened
            and (
                blocker_reduction
                or slot_frontier_progress
                or barrier_progress
                or legal_delta > 0
            )
        )
        release_consumed = bool(
            before_release > 0
            and after_release > 0
            and (goal_progress or blocker_reduction or slot_frontier_progress)
        )
        obligation_completed = bool(goal_progress or release_consumed)
        certified_progress = bool(
            obligation_completed
            or blocker_reduction
            or slot_frontier_progress
            or barrier_progress
            or release_assigned
        )
        stale_open_release = bool(
            before_release > 0
            and after_release > 0
            and not certified_progress
            and (
                bool(phase.get("no_progress"))
                or repeated_context
                or release_head > self.agenda_certificate_release_head_floor
            )
        )
        agenda_loop = bool(stale_open_release and repeated_context)
        pseudo_progress = bool(
            release_head > self.agenda_certificate_release_head_floor
            and not certified_progress
        )
        effective_release_head = max(0.0, release_head)
        release_gate_reason = "full_release_head"
        if pseudo_progress:
            effective_release_head = 0.0
            release_gate_reason = "no_certificate"
        elif release_opened and not release_consumed:
            effective_release_head = min(
                effective_release_head,
                max(0.0, self.agenda_certificate_release_open_cap),
            )
            release_gate_reason = "opened_not_consumed_cap"
        tier = 4
        if bool(sdv.get("hard_negative")) or bool(trap.get("severe_trap_risk")):
            tier = 0
        elif agenda_loop:
            tier = 2
        elif pseudo_progress:
            tier = 3
        elif release_consumed or obligation_completed:
            tier = 7
        elif blocker_reduction:
            tier = 6
        elif certified_progress:
            tier = 5
        raw_bonus = 0.0
        raw_penalty = 0.0
        if certified_progress:
            raw_bonus += self.agenda_certificate_bonus
        if release_assigned and not release_consumed:
            raw_bonus += self.agenda_certificate_release_assigned_bonus
        if release_consumed:
            raw_bonus += self.agenda_certificate_release_consumed_bonus
        if pseudo_progress:
            raw_penalty += self.agenda_certificate_pseudo_penalty
        if stale_open_release:
            raw_penalty += self.agenda_certificate_stale_release_penalty
        if agenda_loop:
            raw_penalty += self.agenda_certificate_loop_penalty
        return {
            "agenda_certificate_enabled": True,
            "agenda_scope": "minimal_release_progress_certificate_v1",
            "goal_progress": bool(goal_progress),
            "blocker_reduction": bool(blocker_reduction),
            "slot_frontier_progress": bool(slot_frontier_progress),
            "barrier_progress": bool(barrier_progress),
            "release_opened": bool(release_opened),
            "release_assigned": bool(release_assigned),
            "release_consumed": bool(release_consumed),
            "stale_open_release": bool(stale_open_release),
            "agenda_loop": bool(agenda_loop),
            "pseudo_progress": bool(pseudo_progress),
            "certified_progress": bool(certified_progress),
            "obligation_completed": bool(obligation_completed),
            "agenda_tier": int(tier),
            "release_head_raw": float(release_head),
            "release_head_effective": float(effective_release_head),
            "release_gate_reason": release_gate_reason,
            "raw_bonus": float(raw_bonus),
            "raw_penalty": float(raw_penalty),
            "goal_delta": int(goal_delta),
            "release_action_count_delta": int(release_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "schedule_barrier_cost_delta": float(barrier_delta),
            "legal_push_count_delta": int(legal_delta),
            "before_release_action_count": int(before_release),
            "after_release_action_count": int(after_release),
            "locked_stand_blocker_delta": int(locked_delta),
            "repeated_context": bool(repeated_context),
        }

    def _agenda_certificate_components(
        self,
        heads: dict[str, float],
        certificate: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        components = {
            "value": self.multihead_sd_consumer_weights["value"]
            * float(heads.get("value", 0.0)),
            "release_progress": self.multihead_sd_consumer_weights[
                "release_progress"
            ]
            * float(certificate.get("release_head_effective", 0.0)),
            "loop_risk": -self.multihead_sd_consumer_weights["loop_risk"]
            * float(heads.get("loop_risk", 0.0)),
            "trap_risk": -self.multihead_sd_consumer_weights["trap_risk"]
            * float(heads.get("trap_risk", 0.0)),
            "agenda_bonus": float(certificate.get("raw_bonus", 0.0)),
            "agenda_penalty": -float(certificate.get("raw_penalty", 0.0)),
        }
        return float(sum(components.values())), components

    def _learned_sd_consumer_score(
        self,
        candidate: RoomSemanticCandidate,
    ) -> float:
        feature_dict = self._learned_sd_consumer_feature_dict_from_candidate(candidate)
        heads = self._multihead_sd_consumer_forward(feature_dict)
        raw_a35_score, raw_a35_components = self._multihead_sd_consumer_combined_raw(
            heads
        )
        certificate = self._agenda_certificate_summary(
            phase=candidate.room_phase_progress_summary,
            sdv=candidate.room_sd_value_summary,
            trap=candidate.room_pre_release_trap_summary,
            stand_lock=candidate.room_stand_lock_summary,
            heads=heads,
        )
        raw_score, components = self._agenda_certificate_components(
            heads,
            certificate,
        )
        unclipped = self.learned_sd_consumer_weight * raw_score
        clip = max(0.0, self.neural_sd_consumer_score_clip)
        score = float(max(-clip, min(clip, unclipped))) if clip > 0.0 else float(unclipped)
        candidate.room_sd_value_score += float(score)
        candidate.room_sd_value_summary = {
            **candidate.room_sd_value_summary,
            "learned_sd_consumer_enabled": bool(self.learned_sd_consumer_enabled),
            "learned_sd_consumer_checkpoint_loaded": bool(
                self.learned_sd_consumer_payload is not None
            ),
            "learned_sd_consumer_raw_score": float(raw_score),
            "learned_sd_consumer_raw_score_before_agenda_gate": float(
                raw_a35_score
            ),
            "learned_sd_consumer_unclipped_score": float(unclipped),
            "learned_sd_consumer_score": float(score),
            "learned_sd_consumer_feature_contract": "room_sd_candidate_features_v1",
            "learned_sd_consumer_root_override_enabled": bool(
                self.learned_sd_consumer_root_override
            ),
            "neural_sd_consumer_enabled": True,
            "neural_sd_consumer_score_clip": float(clip),
            "multihead_sd_consumer_enabled": True,
            "agenda_certificate_sd_consumer_enabled": True,
            "multihead_sd_consumer_target": "value_release_loop_trap_v1",
            "multihead_sd_consumer_heads": {
                name: float(value) for name, value in heads.items()
            },
            "multihead_sd_consumer_components": {
                name: float(value) for name, value in components.items()
            },
            "multihead_sd_consumer_components_before_agenda_gate": {
                name: float(value) for name, value in raw_a35_components.items()
            },
            "multihead_sd_consumer_weights": dict(self.multihead_sd_consumer_weights),
            "agenda_certificate": certificate,
            "learned_sd_consumer_features": feature_dict,
        }
        return float(score)

    def _learned_sd_choice_key(self, edge: RoomSemanticEdge) -> tuple:
        feature_dict = self._learned_sd_consumer_feature_dict(edge)
        heads = self._multihead_sd_consumer_forward(feature_dict)
        certificate = self._agenda_certificate_summary(
            phase=edge.room_phase_progress_summary,
            sdv=edge.room_sd_value_summary,
            trap=edge.room_pre_release_trap_summary,
            stand_lock=edge.room_stand_lock_summary,
            heads=heads,
        )
        raw_score, _ = self._agenda_certificate_components(heads, certificate)
        score = self.learned_sd_consumer_weight * raw_score
        return (
            int(certificate.get("agenda_tier", 0)),
            float(score),
            float(certificate.get("release_head_effective", 0.0)),
            -float(heads.get("loop_risk", 0.0)),
            -float(heads.get("trap_risk", 0.0)),
            float(edge.room_sd_value_score),
            bool(edge.room_sd_value_summary.get("milestone_progress")),
            not bool(edge.room_sd_value_summary.get("pseudo_milestone_loop")),
            not bool(edge.room_sd_value_summary.get("hard_negative")),
            edge.visits,
            edge.mean_value,
            edge.prior_score,
        )


class RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner(
    RoomScheduleAgendaCertificateMacroGFTSDMCTSPlanner
):
    """A37 planner with an active-agenda progress certificate.

    A37 keeps A36's multi-head consumer and release certificate, but binds the
    certificate to the current obligation.  The aim is diagnostic: a local
    release-closing delta is not treated as progress when it returns to a
    repeated blocker context without goal or frontier advancement.
    """

    provider_id = "room_schedule_active_agenda_certificate_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        active_agenda_slot_delta_threshold: float = 0.5,
        active_agenda_slot_regression_tolerance: float = 0.5,
        active_agenda_repeated_release_penalty: float = 1.20,
        active_agenda_off_agenda_penalty: float = 0.85,
        active_agenda_release_loop_penalty: float = 0.70,
        active_agenda_release_head_cap: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.active_agenda_slot_delta_threshold = float(
            active_agenda_slot_delta_threshold
        )
        self.active_agenda_slot_regression_tolerance = float(
            active_agenda_slot_regression_tolerance
        )
        self.active_agenda_repeated_release_penalty = float(
            active_agenda_repeated_release_penalty
        )
        self.active_agenda_off_agenda_penalty = float(
            active_agenda_off_agenda_penalty
        )
        self.active_agenda_release_loop_penalty = float(
            active_agenda_release_loop_penalty
        )
        self.active_agenda_release_head_cap = float(active_agenda_release_head_cap)

    def _active_agenda_metric(
        self,
        summary: dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        return self._learned_sd_consumer_scalar(summary.get(key), default)

    def _active_agenda_summary(
        self,
        *,
        phase: dict[str, Any],
        trap: dict[str, Any],
        stand_lock: dict[str, Any],
        base_certificate: dict[str, Any],
    ) -> dict[str, Any]:
        before = trap.get("before", {}) if isinstance(trap.get("before"), dict) else {}
        after = trap.get("after", {}) if isinstance(trap.get("after"), dict) else {}
        before_stand = (
            stand_lock.get("before", {})
            if isinstance(stand_lock.get("before"), dict)
            else {}
        )
        after_stand = (
            stand_lock.get("after", {})
            if isinstance(stand_lock.get("after"), dict)
            else {}
        )

        before_release = int(
            self._active_agenda_metric(
                phase,
                "before_release_action_count",
                before.get("release_action_count", 0),
            )
        )
        after_release = int(
            self._active_agenda_metric(
                phase,
                "after_release_action_count",
                after.get("release_action_count", 0),
            )
        )
        before_dependency = int(
            self._active_agenda_metric(
                phase,
                "before_dependency_total",
                before.get("dependency_total", 0),
            )
        )
        after_dependency = int(
            self._active_agenda_metric(
                phase,
                "after_dependency_total",
                after.get("dependency_total", 0),
            )
        )
        before_target = int(
            self._active_agenda_metric(
                before,
                "target_blocker_count",
                max(0.0, before_dependency),
            )
        )
        after_target = int(
            self._active_agenda_metric(
                after,
                "target_blocker_count",
                max(0.0, after_dependency),
            )
        )
        before_locked = int(
            self._active_agenda_metric(before_stand, "locked_stand_blocker_count")
        )
        after_locked = int(
            self._active_agenda_metric(after_stand, "locked_stand_blocker_count")
        )
        before_goal_locked = int(
            self._active_agenda_metric(
                before_stand,
                "goal_locked_stand_blocker_count",
            )
        )
        after_goal_locked = int(
            self._active_agenda_metric(
                after_stand,
                "goal_locked_stand_blocker_count",
            )
        )
        before_slot = float(
            self._active_agenda_metric(
                before,
                "slot_schedule_cost",
                0.0,
            )
        )
        after_slot = float(
            self._active_agenda_metric(
                after,
                "slot_schedule_cost",
                before_slot,
            )
        )
        before_stage = str(before.get("stage", ""))
        after_stage = str(after.get("stage", ""))

        goal_delta = int(base_certificate.get("goal_delta", 0))
        release_delta = int(base_certificate.get("release_action_count_delta", 0))
        slot_delta = float(base_certificate.get("slot_schedule_cost_delta", 0.0))
        legal_delta = int(base_certificate.get("legal_push_count_delta", 0))
        repeated_context = bool(base_certificate.get("repeated_context"))
        release_head = float(base_certificate.get("release_head_raw", 0.0))

        active_blocker_before = int(
            before_dependency
            + before_target
            + before_locked
            + before_goal_locked
        )
        active_blocker_after = int(
            after_dependency + after_target + after_locked + after_goal_locked
        )
        active_blocker_delta = int(active_blocker_before - active_blocker_after)

        if before_release > 0:
            agenda_type = "consume_release"
        elif active_blocker_before > 0 or before_stage == "release":
            agenda_type = "release_blocker"
        elif before_slot > 0.0 or before_stage == "fill":
            agenda_type = "fill_frontier_slot"
        else:
            agenda_type = "goal_completion"

        goal_progress = bool(goal_delta > 0)
        slot_frontier_progress = bool(
            slot_delta > self.active_agenda_slot_delta_threshold
        )
        legal_margin_progress = bool(legal_delta > 0)
        release_opened = bool(after_release > before_release or release_delta > 0)
        release_closed = bool(before_release > 0 and after_release < before_release)
        blocker_progress = bool(active_blocker_delta > 0)
        blocker_regression = bool(active_blocker_delta < 0)
        slot_regression = bool(
            slot_delta < -self.active_agenda_slot_regression_tolerance
        )
        release_reopened_or_closed_loop = bool(
            agenda_type == "consume_release"
            and release_closed
            and repeated_context
            and slot_regression
            and not goal_progress
        )

        if agenda_type == "consume_release":
            agenda_progress = bool(
                goal_progress
                or (
                    release_closed
                    and blocker_progress
                    and not release_reopened_or_closed_loop
                )
                or (release_closed and slot_frontier_progress)
                or (release_closed and legal_margin_progress and not repeated_context)
            )
            agenda_reason = "release_consumed"
            if release_reopened_or_closed_loop:
                agenda_reason = "repeated_release_closure"
        elif agenda_type == "release_blocker":
            agenda_progress = bool(
                goal_progress
                or (release_opened and not repeated_context)
                or (blocker_progress and not repeated_context)
                or slot_frontier_progress
            )
            agenda_reason = "release_opened_or_blocker_reduced"
        elif agenda_type == "fill_frontier_slot":
            agenda_progress = bool(goal_progress or slot_frontier_progress)
            agenda_reason = "frontier_slot_or_goal"
        else:
            agenda_progress = bool(goal_progress)
            agenda_reason = "goal_delta"

        off_agenda = bool(
            not agenda_progress
            and (
                release_head > self.agenda_certificate_release_head_floor
                or release_opened
                or release_closed
                or blocker_regression
                or repeated_context
            )
        )
        stale_open_release = bool(
            agenda_type == "consume_release"
            and before_release > 0
            and not agenda_progress
        )
        agenda_loop = bool(
            off_agenda
            and repeated_context
            and (release_opened or release_closed or slot_regression)
        )
        pseudo_progress = bool(
            off_agenda
            or (
                release_head > self.agenda_certificate_release_head_floor
                and not agenda_progress
            )
        )
        release_consumed = bool(
            agenda_type == "consume_release" and release_closed and agenda_progress
        )
        release_assigned = bool(
            agenda_type == "release_blocker" and release_opened and agenda_progress
        )

        return {
            "active_agenda_enabled": True,
            "active_agenda_type": agenda_type,
            "active_agenda_reason": agenda_reason,
            "active_agenda_before_stage": before_stage,
            "active_agenda_after_stage": after_stage,
            "active_agenda_blocker_before": int(active_blocker_before),
            "active_agenda_blocker_after": int(active_blocker_after),
            "active_agenda_blocker_delta": int(active_blocker_delta),
            "active_agenda_release_before": int(before_release),
            "active_agenda_release_after": int(after_release),
            "active_agenda_slot_before": float(before_slot),
            "active_agenda_slot_after": float(after_slot),
            "active_agenda_progress": bool(agenda_progress),
            "active_agenda_off_agenda_action": bool(off_agenda),
            "active_agenda_release_loop": bool(release_reopened_or_closed_loop),
            "active_agenda_repeated_context": bool(repeated_context),
            "active_agenda_slot_regression": bool(slot_regression),
            "active_agenda_release_opened": bool(release_opened),
            "active_agenda_release_closed": bool(release_closed),
            "active_agenda_blocker_progress": bool(blocker_progress),
            "active_agenda_blocker_regression": bool(blocker_regression),
            "active_agenda_goal_progress": bool(goal_progress),
            "active_agenda_slot_frontier_progress": bool(slot_frontier_progress),
            "active_agenda_legal_margin_progress": bool(legal_margin_progress),
            "active_agenda_release_consumed": bool(release_consumed),
            "active_agenda_release_assigned": bool(release_assigned),
            "active_agenda_pseudo_progress": bool(pseudo_progress),
            "active_agenda_stale_open_release": bool(stale_open_release),
            "active_agenda_loop": bool(agenda_loop),
        }

    def _agenda_certificate_summary(
        self,
        *,
        phase: dict[str, Any],
        sdv: dict[str, Any],
        trap: dict[str, Any],
        stand_lock: dict[str, Any],
        heads: dict[str, float],
    ) -> dict[str, Any]:
        certificate = super()._agenda_certificate_summary(
            phase=phase,
            sdv=sdv,
            trap=trap,
            stand_lock=stand_lock,
            heads=heads,
        )
        active = self._active_agenda_summary(
            phase=phase,
            trap=trap,
            stand_lock=stand_lock,
            base_certificate=certificate,
        )

        agenda_progress = bool(active["active_agenda_progress"])
        off_agenda = bool(active["active_agenda_off_agenda_action"])
        release_loop = bool(active["active_agenda_release_loop"])
        pseudo_progress = bool(active["active_agenda_pseudo_progress"])
        stale_open_release = bool(active["active_agenda_stale_open_release"])
        agenda_loop = bool(active["active_agenda_loop"])
        release_opened = bool(active["active_agenda_release_opened"])
        release_consumed = bool(active["active_agenda_release_consumed"])
        release_assigned = bool(active["active_agenda_release_assigned"])
        release_head = float(certificate.get("release_head_raw", 0.0))

        effective_release_head = max(0.0, release_head)
        release_gate_reason = "active_agenda_full_release_head"
        if pseudo_progress or off_agenda:
            effective_release_head = 0.0
            release_gate_reason = "active_agenda_off_agenda"
        elif release_opened and not release_consumed:
            effective_release_head = min(
                effective_release_head,
                max(0.0, self.active_agenda_release_head_cap),
            )
            release_gate_reason = "active_agenda_opened_not_consumed_cap"

        tier = 4
        if bool(sdv.get("hard_negative")) or bool(trap.get("severe_trap_risk")):
            tier = 0
        elif agenda_loop or release_loop:
            tier = 1
        elif off_agenda:
            tier = 2
        elif pseudo_progress:
            tier = 3
        elif release_consumed or bool(active["active_agenda_goal_progress"]):
            tier = 7
        elif agenda_progress and bool(active["active_agenda_blocker_progress"]):
            tier = 6
        elif agenda_progress:
            tier = 5

        raw_bonus = 0.0
        raw_penalty = 0.0
        if agenda_progress:
            raw_bonus += self.agenda_certificate_bonus
        if release_assigned:
            raw_bonus += self.agenda_certificate_release_assigned_bonus
        if release_consumed:
            raw_bonus += self.agenda_certificate_release_consumed_bonus
        if pseudo_progress:
            raw_penalty += self.agenda_certificate_pseudo_penalty
        if stale_open_release:
            raw_penalty += self.agenda_certificate_stale_release_penalty
        if agenda_loop:
            raw_penalty += self.agenda_certificate_loop_penalty
        if off_agenda:
            raw_penalty += self.active_agenda_off_agenda_penalty
        if release_loop:
            raw_penalty += self.active_agenda_repeated_release_penalty
            raw_penalty += self.active_agenda_release_loop_penalty

        certificate.update(
            {
                "agenda_scope": "active_obligation_blocker_set_v1",
                "goal_progress": bool(active["active_agenda_goal_progress"]),
                "blocker_reduction": bool(active["active_agenda_blocker_progress"]),
                "slot_frontier_progress": bool(
                    active["active_agenda_slot_frontier_progress"]
                ),
                "release_opened": bool(release_opened),
                "release_assigned": bool(release_assigned),
                "release_consumed": bool(release_consumed),
                "stale_open_release": bool(stale_open_release),
                "agenda_loop": bool(agenda_loop),
                "pseudo_progress": bool(pseudo_progress),
                "certified_progress": bool(agenda_progress),
                "obligation_completed": bool(
                    release_consumed or active["active_agenda_goal_progress"]
                ),
                "agenda_tier": int(tier),
                "release_head_effective": float(effective_release_head),
                "release_gate_reason": release_gate_reason,
                "raw_bonus": float(raw_bonus),
                "raw_penalty": float(raw_penalty),
                **active,
            }
        )
        return certificate


class RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner(
    RoomScheduleActiveAgendaCertificateMacroGFTSDMCTSPlanner
):
    """A38 planner with explicit agenda potential in the MCTS leaf value.

    A38 follows the Agenda-GFT E3 route: active agenda remains the A37
    definition, while the consumer location moves from root soft prior only to
    leaf value.  No new action generator, teacher signal, or graph model is
    introduced.
    """

    provider_id = "room_schedule_agenda_leaf_potential_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        agenda_leaf_neutral_offset: float = 10.00,
        agenda_leaf_slot_cost_weight: float = 0.04,
        agenda_leaf_blocker_penalty: float = 2.25,
        agenda_leaf_open_release_bonus: float = 0.55,
        agenda_leaf_stale_release_penalty: float = 5.50,
        agenda_leaf_loop_penalty: float = 7.50,
        agenda_leaf_goal_locked_penalty: float = 4.00,
        agenda_leaf_low_legal_penalty: float = 2.50,
        agenda_leaf_release_stage_bonus: float = 0.35,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agenda_leaf_neutral_offset = float(agenda_leaf_neutral_offset)
        self.agenda_leaf_slot_cost_weight = float(agenda_leaf_slot_cost_weight)
        self.agenda_leaf_blocker_penalty = float(agenda_leaf_blocker_penalty)
        self.agenda_leaf_open_release_bonus = float(agenda_leaf_open_release_bonus)
        self.agenda_leaf_stale_release_penalty = float(agenda_leaf_stale_release_penalty)
        self.agenda_leaf_loop_penalty = float(agenda_leaf_loop_penalty)
        self.agenda_leaf_goal_locked_penalty = float(agenda_leaf_goal_locked_penalty)
        self.agenda_leaf_low_legal_penalty = float(agenda_leaf_low_legal_penalty)
        self.agenda_leaf_release_stage_bonus = float(agenda_leaf_release_stage_bonus)
        self._agenda_leaf_potential_cache: dict[tuple, dict[str, Any]] = {}
        self._agenda_leaf_eval_count = 0
        self._agenda_leaf_penalty_total = 0.0
        self._agenda_leaf_bonus_total = 0.0
        self._agenda_leaf_loop_count = 0
        self._agenda_leaf_stale_release_count = 0

    def _agenda_leaf_stage_type(
        self,
        stage: dict[str, Any],
    ) -> str:
        stage_name = str(stage.get("stage", ""))
        release_count = int(stage.get("release_action_count", 0))
        dependency_total = int(stage.get("dependency_total", 0))
        if release_count > 0:
            return "consume_release"
        if dependency_total > 0 or stage_name == "release":
            return "release_blocker"
        if float(stage.get("slot_schedule_cost", 0.0)) > 0.0 or stage_name == "fill":
            return "fill_frontier_slot"
        return "goal_completion"

    def _agenda_leaf_potential_summary(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        key = self._summary_cache_key(
            "agenda_leaf_potential",
            state,
            commitment_state,
            self.room_schedule_lookahead_slots,
        )
        cached = self._agenda_leaf_potential_cache.get(key)
        if cached is not None:
            return cached

        stage = self._decision_stage_summary(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        signature = self.schedule_stall_signature(state, commitment_state)
        dependency_total = int(totals["dependency_count"]) + int(
            totals["future_dependency_count"]
        )
        target_blockers = int(totals["target_blocker_count"])
        release_count = int(totals["release_action_count"])
        goal_locked = int(stage.get("goal_locked_stand_blocker_count", 0))
        locked_stand = int(stage.get("locked_stand_blocker_count", 0))
        slot_cost = float(stage.get("slot_schedule_cost", 0.0))
        legal = int(stage.get("nonblocked_legal_push_count", 0))
        active_blockers = int(
            dependency_total + target_blockers + goal_locked + locked_stand
        )
        agenda_type = self._agenda_leaf_stage_type(stage)
        repeated_signature = bool(signature in self.recent_schedule_keys)
        stale_open_release = bool(
            agenda_type == "consume_release"
            and release_count > 0
            and repeated_signature
        )
        agenda_loop = bool(
            repeated_signature
            and (
                stale_open_release
                or (active_blockers > 0 and release_count > 0)
                or str(stage.get("stage")) == "release"
            )
        )
        low_legal = bool(
            active_blockers > 0
            and legal <= self.room_access_recovery_min_legal_pushes
        )
        penalty = (
            self.agenda_leaf_slot_cost_weight * slot_cost
            + self.agenda_leaf_blocker_penalty * float(active_blockers)
            + self.agenda_leaf_stale_release_penalty * float(stale_open_release)
            + self.agenda_leaf_loop_penalty * float(agenda_loop)
            + self.agenda_leaf_goal_locked_penalty * float(goal_locked)
            + self.agenda_leaf_low_legal_penalty * float(low_legal)
        )
        bonus = (
            self.agenda_leaf_open_release_bonus * float(release_count)
            + self.agenda_leaf_release_stage_bonus
            * float(agenda_type == "release_blocker" and dependency_total > 0)
        )
        adjustment = float(self.agenda_leaf_neutral_offset + bonus - penalty)
        summary = {
            "agenda_leaf_potential_enabled": True,
            "agenda_leaf_scope": "explicit_agenda_potential_e3_v1",
            "agenda_leaf_agenda_type": agenda_type,
            "agenda_leaf_stage": str(stage.get("stage")),
            "agenda_leaf_active_blocker_count": int(active_blockers),
            "agenda_leaf_dependency_total": int(dependency_total),
            "agenda_leaf_target_blocker_count": int(target_blockers),
            "agenda_leaf_release_action_count": int(release_count),
            "agenda_leaf_slot_schedule_cost": float(slot_cost),
            "agenda_leaf_repeated_signature": bool(repeated_signature),
            "agenda_leaf_stale_open_release": bool(stale_open_release),
            "agenda_leaf_loop": bool(agenda_loop),
            "agenda_leaf_low_legal": bool(low_legal),
            "agenda_leaf_goal_locked_stand_blocker_count": int(goal_locked),
            "agenda_leaf_locked_stand_blocker_count": int(locked_stand),
            "agenda_leaf_bonus": float(bonus),
            "agenda_leaf_penalty": float(penalty),
            "agenda_leaf_neutral_offset": float(self.agenda_leaf_neutral_offset),
            "agenda_leaf_adjustment": float(adjustment),
        }
        self._agenda_leaf_potential_cache[key] = summary
        return summary

    def _agenda_leaf_apply_potential(
        self,
        value: float,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        *,
        record: bool,
    ) -> float:
        summary = self._agenda_leaf_potential_summary(state, commitment_state)
        adjusted = float(value + float(summary["agenda_leaf_adjustment"]))
        low = min(self.deadend_value, self.goal_value)
        high = max(self.deadend_value, self.goal_value)
        adjusted = float(max(low, min(high, adjusted)))
        if record:
            self._agenda_leaf_eval_count += 1
            self._agenda_leaf_penalty_total += float(summary["agenda_leaf_penalty"])
            self._agenda_leaf_bonus_total += float(summary["agenda_leaf_bonus"])
            self._agenda_leaf_loop_count += int(bool(summary["agenda_leaf_loop"]))
            self._agenda_leaf_stale_release_count += int(
                bool(summary["agenda_leaf_stale_open_release"])
            )
        return adjusted

    def leaf_value(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> float:
        value = super().leaf_value(state, commitment_state)
        return self._agenda_leaf_apply_potential(
            value,
            state,
            commitment_state,
            record=True,
        )

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._agenda_leaf_eval_count = 0
        self._agenda_leaf_penalty_total = 0.0
        self._agenda_leaf_bonus_total = 0.0
        self._agenda_leaf_loop_count = 0
        self._agenda_leaf_stale_release_count = 0
        result = super().plan(state, commitment_state)
        normalized = self.model.normalize_room_commitment_state(
            state,
            commitment_state,
        )
        root_value = super().leaf_value(state, normalized)
        root_adjusted = self._agenda_leaf_apply_potential(
            root_value,
            state,
            normalized,
            record=False,
        )
        root_summary = self._agenda_leaf_potential_summary(state, normalized)
        result["agenda_leaf_potential"] = {
            **root_summary,
            "agenda_leaf_root_value_before": float(root_value),
            "agenda_leaf_root_value_after": float(root_adjusted),
            "agenda_leaf_eval_count": int(self._agenda_leaf_eval_count),
            "agenda_leaf_penalty_total": float(self._agenda_leaf_penalty_total),
            "agenda_leaf_bonus_total": float(self._agenda_leaf_bonus_total),
            "agenda_leaf_loop_eval_count": int(self._agenda_leaf_loop_count),
            "agenda_leaf_stale_release_eval_count": int(
                self._agenda_leaf_stale_release_count
            ),
        }
        return result


class RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner(
    RoomScheduleAgendaLeafPotentialMacroGFTSDMCTSPlanner
):
    """A39 planner with lexicographic agenda-tier root choice.

    A39 follows Agenda-GFT point 16.  It does not add new actions, teacher
    signals, or graph structure; it only changes how an already-computed active
    agenda certificate is consumed at the root.  If the parent planner selects a
    lower-tier action while a higher-tier root candidate exists, the chosen edge
    is hard-downgraded in favor of the best candidate from the highest available
    tier.
    """

    provider_id = "room_schedule_agenda_tier_choice_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        agenda_tier_choice_enabled: bool = True,
        agenda_tier_choice_mode: str = "release_loop_only",
        agenda_tier_choice_min_improvement: int = 1,
        agenda_tier_choice_parent_max_tier: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agenda_tier_choice_enabled = bool(agenda_tier_choice_enabled)
        self.agenda_tier_choice_mode = str(agenda_tier_choice_mode)
        self.agenda_tier_choice_min_improvement = max(
            int(agenda_tier_choice_min_improvement),
            0,
        )
        self.agenda_tier_choice_parent_max_tier = int(
            agenda_tier_choice_parent_max_tier
        )
        self._agenda_tier_choice_summary: dict[str, Any] = {}

    @staticmethod
    def _agenda_tier_certificate(edge: RoomSemanticEdge) -> dict[str, Any]:
        sd_value = edge.room_sd_value_summary or {}
        cert = sd_value.get("agenda_certificate", {})
        return cert if isinstance(cert, dict) else {}

    def _agenda_tier_value(self, edge: RoomSemanticEdge) -> int:
        cert = self._agenda_tier_certificate(edge)
        try:
            return int(cert.get("agenda_tier", 0))
        except (TypeError, ValueError):
            return 0

    def _agenda_tier_intra_key(self, edge: RoomSemanticEdge) -> tuple:
        cert = self._agenda_tier_certificate(edge)
        learned = self._learned_sd_consumer_scalar(
            edge.room_sd_value_summary.get("learned_sd_consumer_score"),
        )
        progress = bool(cert.get("active_agenda_progress"))
        off_agenda = bool(cert.get("active_agenda_off_agenda_action"))
        release_loop = bool(cert.get("active_agenda_release_loop"))
        blocker_delta = int(
            self._learned_sd_consumer_scalar(
                cert.get("active_agenda_blocker_delta"),
            )
        )
        if self.select_by == "q":
            search_key = (edge.mean_value, edge.visits)
        else:
            search_key = (edge.visits, edge.mean_value)
        return (
            self._agenda_tier_value(edge),
            progress,
            not release_loop,
            not off_agenda,
            blocker_delta,
            *search_key,
            float(learned),
            float(edge.room_sd_value_score),
            float(edge.prior_score),
        )

    def _agenda_tier_parent_is_trigger(self, edge: RoomSemanticEdge) -> bool:
        if self.agenda_tier_choice_mode == "global":
            return True
        cert = self._agenda_tier_certificate(edge)
        tier = self._agenda_tier_value(edge)
        agenda_type = str(cert.get("active_agenda_type", ""))
        return bool(
            agenda_type == "consume_release"
            and tier <= self.agenda_tier_choice_parent_max_tier
            and (
                bool(cert.get("active_agenda_release_loop"))
                or bool(cert.get("active_agenda_off_agenda_action"))
                or bool(cert.get("active_agenda_stale_open_release"))
            )
        )

    def _agenda_tier_candidate_allowed(self, edge: RoomSemanticEdge) -> bool:
        if self.agenda_tier_choice_mode == "global":
            return True
        cert = self._agenda_tier_certificate(edge)
        return bool(
            str(cert.get("active_agenda_type", "")) == "consume_release"
            and not bool(cert.get("active_agenda_release_loop"))
        )

    def _agenda_tier_action_summary(
        self,
        edge: RoomSemanticEdge | None,
    ) -> dict[str, Any] | None:
        if edge is None:
            return None
        cert = self._agenda_tier_certificate(edge)
        action = edge.action.physical_action
        return {
            "box_index": int(action.box_index),
            "direction": int(action.direction),
            "direction_name": ACTION_NAMES[int(action.direction)],
            "flat_index": int(action.flat_index),
            "agenda_tier": int(self._agenda_tier_value(edge)),
            "active_agenda_type": cert.get("active_agenda_type"),
            "active_agenda_progress": bool(cert.get("active_agenda_progress")),
            "active_agenda_off_agenda_action": bool(
                cert.get("active_agenda_off_agenda_action")
            ),
            "active_agenda_release_loop": bool(
                cert.get("active_agenda_release_loop")
            ),
            "active_agenda_blocker_delta": cert.get("active_agenda_blocker_delta"),
            "learned_sd_consumer_score": self._learned_sd_consumer_scalar(
                edge.room_sd_value_summary.get("learned_sd_consumer_score"),
            ),
            "room_sd_value_score": float(edge.room_sd_value_score),
            "visits": int(edge.visits),
            "mean_value": float(edge.mean_value),
            "prior_score": float(edge.prior_score),
        }

    def _agenda_tier_choice_record(
        self,
        root: RoomSemanticNode,
        parent_best: RoomSemanticEdge,
        selected: RoomSemanticEdge,
        *,
        override: bool,
        reason: str,
    ) -> dict[str, Any]:
        tier_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        candidates: list[dict[str, Any]] = []
        for edge in root.children.values():
            cert = self._agenda_tier_certificate(edge)
            tier = self._agenda_tier_value(edge)
            tier_counts[str(tier)] = tier_counts.get(str(tier), 0) + 1
            agenda_type = str(cert.get("active_agenda_type", "NA"))
            type_counts[agenda_type] = type_counts.get(agenda_type, 0) + 1
            candidates.append(self._agenda_tier_action_summary(edge) or {})
        candidates.sort(
            key=lambda row: (
                int(row.get("agenda_tier", 0)),
                int(row.get("visits", 0)),
                float(row.get("mean_value", 0.0)),
                float(row.get("prior_score", 0.0)),
            ),
            reverse=True,
        )
        parent_tier = self._agenda_tier_value(parent_best)
        selected_tier = self._agenda_tier_value(selected)
        max_tier = max((self._agenda_tier_value(edge) for edge in root.children.values()), default=parent_tier)
        return {
            "agenda_tier_choice_enabled": bool(self.agenda_tier_choice_enabled),
            "agenda_tier_choice_scope": "lexicographic_root_choice_e16_v1",
            "agenda_tier_choice_mode": self.agenda_tier_choice_mode,
            "agenda_tier_choice_min_improvement": int(
                self.agenda_tier_choice_min_improvement
            ),
            "agenda_tier_choice_parent_max_tier": int(
                self.agenda_tier_choice_parent_max_tier
            ),
            "agenda_tier_choice_override": bool(override),
            "agenda_tier_choice_reason": reason,
            "parent_tier": int(parent_tier),
            "selected_tier": int(selected_tier),
            "max_available_tier": int(max_tier),
            "candidate_count": int(len(root.children)),
            "tier_counts": tier_counts,
            "active_agenda_type_counts": type_counts,
            "parent_action": self._agenda_tier_action_summary(parent_best),
            "selected_action": self._agenda_tier_action_summary(selected),
            "top_tier_candidates": candidates[:8],
        }

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        parent_best = super().choose_root_edge(root)
        if not self.agenda_tier_choice_enabled or not root.children:
            self._agenda_tier_choice_summary = self._agenda_tier_choice_record(
                root,
                parent_best,
                parent_best,
                override=False,
                reason="disabled_or_empty",
            )
            return parent_best

        parent_tier = self._agenda_tier_value(parent_best)
        if not self._agenda_tier_parent_is_trigger(parent_best):
            self._agenda_tier_choice_summary = self._agenda_tier_choice_record(
                root,
                parent_best,
                parent_best,
                override=False,
                reason="parent_not_tier_trigger",
            )
            return parent_best

        allowed_edges = [
            edge
            for edge in root.children.values()
            if self._agenda_tier_candidate_allowed(edge)
        ]
        if not allowed_edges:
            self._agenda_tier_choice_summary = self._agenda_tier_choice_record(
                root,
                parent_best,
                parent_best,
                override=False,
                reason="no_allowed_tier_candidate",
            )
            return parent_best

        max_tier = max(self._agenda_tier_value(edge) for edge in allowed_edges)
        if max_tier < parent_tier + self.agenda_tier_choice_min_improvement:
            self._agenda_tier_choice_summary = self._agenda_tier_choice_record(
                root,
                parent_best,
                parent_best,
                override=False,
                reason="parent_already_highest_tier",
            )
            return parent_best

        top_tier_edges = [
            edge
            for edge in allowed_edges
            if self._agenda_tier_value(edge) == max_tier
        ]
        selected = max(top_tier_edges, key=self._agenda_tier_intra_key)
        self._agenda_tier_choice_summary = self._agenda_tier_choice_record(
            root,
            parent_best,
            selected,
            override=selected is not parent_best,
            reason="higher_agenda_tier_available",
        )
        return selected

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._agenda_tier_choice_summary = {}
        result = super().plan(state, commitment_state)
        result["agenda_tier_choice"] = dict(self._agenda_tier_choice_summary)
        return result


class RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner(
    RoomScheduleAgendaTierChoiceMacroGFTSDMCTSPlanner
):
    """A40 planner with a bounded agenda-option evaluator at the root.

    A40 follows Agenda-GFT point 14 in a conservative form.  It keeps the real
    Sokoban transition model and the A39 first-step tree, then evaluates whether
    a candidate first push can reach the next active-agenda boundary within a
    small local push tree.  The suffix is not executed as a macro; only the first
    root edge is selected, so the closed loop remains independently replanned.
    """

    provider_id = "room_schedule_agenda_option_tree_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        agenda_option_tree_enabled: bool = True,
        agenda_option_tree_mode: str = "consume_release_triggered",
        agenda_option_tree_max_pushes: int = 5,
        agenda_option_tree_max_nodes: int = 96,
        agenda_option_tree_max_root_candidates: int = 10,
        agenda_option_tree_min_tier: int = 5,
        agenda_option_tree_min_tier_improvement: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agenda_option_tree_enabled = bool(agenda_option_tree_enabled)
        self.agenda_option_tree_mode = str(agenda_option_tree_mode)
        self.agenda_option_tree_max_pushes = max(int(agenda_option_tree_max_pushes), 1)
        self.agenda_option_tree_max_nodes = max(int(agenda_option_tree_max_nodes), 1)
        self.agenda_option_tree_max_root_candidates = max(
            int(agenda_option_tree_max_root_candidates),
            1,
        )
        self.agenda_option_tree_min_tier = int(agenda_option_tree_min_tier)
        self.agenda_option_tree_min_tier_improvement = max(
            int(agenda_option_tree_min_tier_improvement),
            0,
        )
        self._agenda_option_tree_summary: dict[str, Any] = {}

    def _agenda_option_tree_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        stand_lock = self._stand_lock_metrics(state, commitment_state)
        totals = schedule["totals"]
        dependency_total = int(totals["dependency_count"]) + int(
            totals["future_dependency_count"]
        )
        active_blocker_count = int(
            dependency_total
            + int(totals["target_blocker_count"])
            + int(stand_lock["locked_stand_blocker_count"])
            + int(stand_lock["goal_locked_stand_blocker_count"])
        )
        legal = int(self._nonblocked_legal_push_count(state))
        stage = self._decision_stage_summary(state, commitment_state)
        signature = self.schedule_stall_signature(state, commitment_state)
        return {
            "boxes_on_goals": int(self.model.boxes_on_goals(state)),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "dependency_total": int(dependency_total),
            "dependency_count": int(totals["dependency_count"]),
            "future_dependency_count": int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "locked_stand_blocker_count": int(
                stand_lock["locked_stand_blocker_count"]
            ),
            "goal_locked_stand_blocker_count": int(
                stand_lock["goal_locked_stand_blocker_count"]
            ),
            "active_blocker_count": int(active_blocker_count),
            "nonblocked_legal_push_count": int(legal),
            "no_legal_pushes": bool(legal <= 0),
            "stage": str(stage.get("stage")),
            "signature": signature,
        }

    def _agenda_option_tree_agenda_type(
        self,
        metrics: dict[str, Any],
    ) -> str:
        if int(metrics["release_action_count"]) > 0:
            return "consume_release"
        if int(metrics["active_blocker_count"]) > 0 or metrics.get("stage") == "release":
            return "release_blocker"
        if float(metrics["slot_schedule_cost"]) > 0.0 or metrics.get("stage") == "fill":
            return "fill_frontier_slot"
        return "goal_completion"

    def _agenda_option_tree_progress(
        self,
        reference: dict[str, Any],
        metrics: dict[str, Any],
        *,
        depth: int,
        repeated_context: bool,
        path_loop: bool,
    ) -> dict[str, Any]:
        agenda_type = self._agenda_option_tree_agenda_type(reference)
        goal_delta = int(metrics["boxes_on_goals"]) - int(reference["boxes_on_goals"])
        release_delta = int(metrics["release_action_count"]) - int(
            reference["release_action_count"]
        )
        release_opened = bool(release_delta > 0)
        release_closed = bool(
            int(reference["release_action_count"]) > 0
            and int(metrics["release_action_count"])
            < int(reference["release_action_count"])
        )
        blocker_delta = int(reference["active_blocker_count"]) - int(
            metrics["active_blocker_count"]
        )
        dependency_delta = int(reference["dependency_total"]) - int(
            metrics["dependency_total"]
        )
        target_delta = int(reference["target_blocker_count"]) - int(
            metrics["target_blocker_count"]
        )
        locked_delta = int(reference["locked_stand_blocker_count"]) - int(
            metrics["locked_stand_blocker_count"]
        )
        goal_locked_delta = int(
            reference["goal_locked_stand_blocker_count"]
        ) - int(metrics["goal_locked_stand_blocker_count"])
        slot_delta = float(reference["slot_schedule_cost"]) - float(
            metrics["slot_schedule_cost"]
        )
        barrier_delta = float(reference["schedule_barrier_cost"]) - float(
            metrics["schedule_barrier_cost"]
        )
        legal_delta = int(metrics["nonblocked_legal_push_count"]) - int(
            reference["nonblocked_legal_push_count"]
        )
        goal_progress = bool(goal_delta > 0)
        blocker_progress = bool(blocker_delta > 0)
        slot_frontier_progress = bool(
            slot_delta > self.active_agenda_slot_delta_threshold
        )
        legal_margin_progress = bool(legal_delta > 0)
        slot_regression = bool(
            slot_delta < -self.active_agenda_slot_regression_tolerance
        )
        blocker_regression = bool(blocker_delta < 0)
        release_loop = bool(
            agenda_type == "consume_release"
            and release_closed
            and repeated_context
            and slot_regression
            and not goal_progress
        )
        if agenda_type == "consume_release":
            agenda_progress = bool(
                goal_progress
                or (release_closed and blocker_progress and not release_loop)
                or (release_closed and slot_frontier_progress)
                or (release_closed and legal_margin_progress and not repeated_context)
            )
        elif agenda_type == "release_blocker":
            agenda_progress = bool(
                goal_progress
                or (release_opened and not repeated_context)
                or (blocker_progress and not repeated_context)
                or slot_frontier_progress
            )
        elif agenda_type == "fill_frontier_slot":
            agenda_progress = bool(goal_progress or slot_frontier_progress)
        else:
            agenda_progress = bool(goal_progress)
        release_consumed = bool(
            agenda_type == "consume_release" and release_closed and agenda_progress
        )
        agenda_done = bool(goal_progress or release_consumed)
        off_agenda = bool(
            not agenda_progress
            and (
                release_opened
                or release_closed
                or blocker_regression
                or repeated_context
                or path_loop
            )
        )
        stale_open_release = bool(
            agenda_type == "consume_release"
            and int(reference["release_action_count"]) > 0
            and not agenda_progress
        )
        agenda_loop = bool(
            (off_agenda and repeated_context)
            or path_loop
            or release_loop
        )
        tier = 4
        if bool(metrics["no_legal_pushes"]) and not goal_progress:
            tier = 0
        elif agenda_loop or release_loop:
            tier = 1
        elif off_agenda or stale_open_release:
            tier = 2
        elif agenda_done:
            tier = 7
        elif agenda_progress and blocker_progress:
            tier = 6
        elif agenda_progress:
            tier = 5
        score = (
            100.0 * float(tier)
            + 30.0 * float(agenda_done)
            + 12.0 * float(max(0, blocker_delta))
            + 5.0 * float(max(0, dependency_delta))
            + 5.0 * float(max(0, target_delta))
            + 4.0 * float(max(0, locked_delta + goal_locked_delta))
            + 2.0 * float(max(0.0, min(slot_delta, 8.0)))
            + 1.0 * float(legal_delta)
            + 0.5 * float(max(0.0, barrier_delta))
            - 20.0 * float(max(0, -goal_delta))
            - 25.0 * float(max(0, -blocker_delta))
            - 30.0 * float(agenda_loop)
            - 40.0 * float(metrics["no_legal_pushes"] and not goal_progress)
            - 3.0 * float(depth)
        )
        return {
            "agenda_type": agenda_type,
            "agenda_tier": int(tier),
            "agenda_progress": bool(agenda_progress),
            "agenda_done": bool(agenda_done),
            "goal_progress": bool(goal_progress),
            "release_consumed": bool(release_consumed),
            "release_opened": bool(release_opened),
            "release_closed": bool(release_closed),
            "off_agenda": bool(off_agenda),
            "stale_open_release": bool(stale_open_release),
            "agenda_loop": bool(agenda_loop),
            "release_loop": bool(release_loop),
            "repeated_context": bool(repeated_context),
            "path_loop": bool(path_loop),
            "goal_delta": int(goal_delta),
            "release_action_count_delta": int(release_delta),
            "active_blocker_delta": int(blocker_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_delta),
            "locked_stand_blocker_delta": int(locked_delta),
            "goal_locked_stand_blocker_delta": int(goal_locked_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "schedule_barrier_cost_delta": float(barrier_delta),
            "legal_push_count_delta": int(legal_delta),
            "after_release_action_count": int(metrics["release_action_count"]),
            "after_active_blocker_count": int(metrics["active_blocker_count"]),
            "after_nonblocked_legal_push_count": int(
                metrics["nonblocked_legal_push_count"]
            ),
            "after_no_legal_pushes": bool(metrics["no_legal_pushes"]),
            "depth": int(depth),
            "score": float(score),
        }

    def _agenda_option_tree_key(self, row: dict[str, Any]) -> tuple:
        progress = row["best_progress"]
        edge = row["edge"]
        return (
            int(progress["agenda_tier"]),
            int(progress["agenda_done"]),
            int(progress["agenda_progress"]),
            -int(progress["agenda_loop"]),
            int(progress["active_blocker_delta"]),
            int(progress["dependency_total_delta"]),
            int(progress["target_blocker_count_delta"]),
            int(progress["legal_push_count_delta"]),
            float(progress["slot_schedule_cost_delta"]),
            float(progress["score"]),
            -int(progress["depth"]),
            *self._agenda_tier_intra_key(edge),
        )

    def _agenda_option_tree_action_row(
        self,
        state: SokobanState,
        action: SokobanPushAction,
        *,
        depth: int,
    ) -> dict[str, Any]:
        source, target = self._moved_cells(state, action)
        return {
            "depth": int(depth),
            "physical_action": self._physical_action_dict(state, action),
            "source": list(source),
            "target": list(target),
        }

    def _agenda_option_tree_evaluate_edge(
        self,
        root: RoomSemanticNode,
        edge: RoomSemanticEdge,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        first_action = edge.action.physical_action
        first_key = self.model.gft_canonical_key_cached(edge.child_state)
        root_key = self.model.gft_canonical_key_cached(root.state)
        first_metrics = self._agenda_option_tree_metrics(
            edge.child_state,
            edge.child_commitment_state,
        )
        first_repeated = bool(
            first_key == root_key
            or first_metrics["signature"] in self.recent_schedule_keys
            or edge.child_commitment_state.key in self.recent_commitment_keys
        )
        first_path_loop = bool(first_key == root_key)
        first_progress = self._agenda_option_tree_progress(
            reference,
            first_metrics,
            depth=1,
            repeated_context=first_repeated,
            path_loop=first_path_loop,
        )
        first_row = self._agenda_option_tree_action_row(
            root.state,
            first_action,
            depth=1,
        )
        best = {
            "best_progress": first_progress,
            "best_metrics": first_metrics,
            "best_actions": [first_row],
            "best_depth": 1,
            "expanded_nodes": 0,
        }
        queue: list[dict[str, Any]] = []
        if (
            not bool(first_progress["agenda_done"])
            and not bool(first_progress["path_loop"])
            and not bool(first_metrics["no_legal_pushes"])
        ):
            queue.append(
                {
                    "state": edge.child_state,
                    "commitment_state": edge.child_commitment_state,
                    "actions": [first_row],
                    "seen_state_keys": {root_key, first_key},
                    "seen_signatures": {
                        reference["signature"],
                        first_metrics["signature"],
                    },
                    "path_loop": first_path_loop,
                }
            )
        expanded = 0
        while queue and expanded < self.agenda_option_tree_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.agenda_option_tree_max_pushes:
                continue
            if bool(node["path_loop"]):
                continue
            if self.model.is_solved(node["state"]):
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                child_key = self.model.gft_canonical_key_cached(child)
                path_loop = bool(child_key in node["seen_state_keys"])
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                child_metrics = self._agenda_option_tree_metrics(
                    child,
                    child_commitment,
                )
                repeated = bool(
                    path_loop
                    or child_metrics["signature"] in node["seen_signatures"]
                    or child_metrics["signature"] in self.recent_schedule_keys
                    or child_commitment.key in self.recent_commitment_keys
                )
                child_depth = depth + 1
                progress = self._agenda_option_tree_progress(
                    reference,
                    child_metrics,
                    depth=child_depth,
                    repeated_context=repeated,
                    path_loop=path_loop,
                )
                action_row = self._agenda_option_tree_action_row(
                    node["state"],
                    action,
                    depth=child_depth,
                )
                candidate_actions = [*node["actions"], action_row]
                candidate = {
                    "best_progress": progress,
                    "best_metrics": child_metrics,
                    "best_actions": candidate_actions,
                    "best_depth": int(child_depth),
                    "expanded_nodes": int(expanded + 1),
                    "edge": edge,
                }
                current_best = {**best, "edge": edge}
                if self._agenda_option_tree_key(candidate) > self._agenda_option_tree_key(
                    current_best
                ):
                    best = {
                        "best_progress": progress,
                        "best_metrics": child_metrics,
                        "best_actions": candidate_actions,
                        "best_depth": int(child_depth),
                        "expanded_nodes": int(expanded + 1),
                    }
                if (
                    child_depth < self.agenda_option_tree_max_pushes
                    and not path_loop
                    and not bool(progress["agenda_done"])
                    and not bool(child_metrics["no_legal_pushes"])
                ):
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate_actions,
                            "seen_state_keys": set(node["seen_state_keys"])
                            | {child_key},
                            "seen_signatures": set(node["seen_signatures"])
                            | {child_metrics["signature"]},
                            "path_loop": bool(path_loop),
                        }
                    )
                expanded += 1
                if expanded >= self.agenda_option_tree_max_nodes:
                    break
        return {
            "edge": edge,
            "root_action": self._agenda_tier_action_summary(edge),
            "root_agenda_tier": int(self._agenda_tier_value(edge)),
            "best_progress": best["best_progress"],
            "best_metrics": {
                key: value
                for key, value in best["best_metrics"].items()
                if key != "signature"
            },
            "best_actions": best["best_actions"],
            "best_depth": int(best["best_depth"]),
            "expanded_nodes": int(max(expanded, int(best["expanded_nodes"]))),
            "option_score": float(best["best_progress"]["score"]),
        }

    def _agenda_option_tree_parent_is_trigger(
        self,
        edge: RoomSemanticEdge,
    ) -> bool:
        cert = self._agenda_tier_certificate(edge)
        if self.agenda_option_tree_mode == "global":
            return True
        return bool(
            str(cert.get("active_agenda_type", "")) == "consume_release"
            and (
                self._agenda_tier_value(edge) <= 2
                or bool(cert.get("active_agenda_off_agenda_action"))
                or bool(cert.get("active_agenda_stale_open_release"))
                or bool(cert.get("active_agenda_release_loop"))
            )
        )

    def _agenda_option_tree_candidate_allowed(
        self,
        edge: RoomSemanticEdge,
    ) -> bool:
        if self.agenda_option_tree_mode == "global":
            return self._agenda_tier_value(edge) > 0
        cert = self._agenda_tier_certificate(edge)
        return bool(
            str(cert.get("active_agenda_type", "")) == "consume_release"
            and self._agenda_tier_value(edge) > 0
        )

    def _agenda_option_tree_record(
        self,
        root: RoomSemanticNode,
        parent_best: RoomSemanticEdge,
        selected: RoomSemanticEdge,
        *,
        reason: str,
        evaluated: list[dict[str, Any]],
        parent_eval: dict[str, Any] | None,
        selected_eval: dict[str, Any] | None,
    ) -> dict[str, Any]:
        def compact_option(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "root_action": row["root_action"],
                "root_agenda_tier": int(row["root_agenda_tier"]),
                "best_progress": row["best_progress"],
                "best_actions": row["best_actions"],
                "best_depth": int(row["best_depth"]),
                "expanded_nodes": int(row["expanded_nodes"]),
                "option_score": float(row["option_score"]),
            }

        compact = []
        for row in sorted(evaluated, key=self._agenda_option_tree_key, reverse=True):
            compact.append(
                {
                    "root_action": row["root_action"],
                    "root_agenda_tier": int(row["root_agenda_tier"]),
                    "best_progress": row["best_progress"],
                    "best_actions": row["best_actions"][: self.agenda_option_tree_max_pushes],
                    "best_depth": int(row["best_depth"]),
                    "expanded_nodes": int(row["expanded_nodes"]),
                    "option_score": float(row["option_score"]),
                }
            )
        parent_tier = self._agenda_tier_value(parent_best)
        selected_tier = self._agenda_tier_value(selected)
        parent_option_tier = (
            None
            if parent_eval is None
            else int(parent_eval["best_progress"]["agenda_tier"])
        )
        selected_option_tier = (
            None
            if selected_eval is None
            else int(selected_eval["best_progress"]["agenda_tier"])
        )
        return {
            "agenda_option_tree_enabled": bool(self.agenda_option_tree_enabled),
            "agenda_option_tree_scope": "bounded_agenda_option_tree_e14_v1",
            "agenda_option_tree_mode": self.agenda_option_tree_mode,
            "agenda_option_tree_max_pushes": int(self.agenda_option_tree_max_pushes),
            "agenda_option_tree_max_nodes": int(self.agenda_option_tree_max_nodes),
            "agenda_option_tree_max_root_candidates": int(
                self.agenda_option_tree_max_root_candidates
            ),
            "agenda_option_tree_min_tier": int(self.agenda_option_tree_min_tier),
            "agenda_option_tree_min_tier_improvement": int(
                self.agenda_option_tree_min_tier_improvement
            ),
            "agenda_option_tree_override": bool(selected is not parent_best),
            "agenda_option_tree_reason": reason,
            "candidate_count": int(len(root.children)),
            "evaluated_option_count": int(len(evaluated)),
            "parent_tier": int(parent_tier),
            "selected_tier": int(selected_tier),
            "parent_option_tier": parent_option_tier,
            "selected_option_tier": selected_option_tier,
            "parent_action": self._agenda_tier_action_summary(parent_best),
            "selected_action": self._agenda_tier_action_summary(selected),
            "parent_option": None if parent_eval is None else compact_option(parent_eval),
            "selected_option": None
            if selected_eval is None
            else compact_option(selected_eval),
            "top_options": compact[:8],
        }

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        parent_best = super().choose_root_edge(root)
        if not self.agenda_option_tree_enabled or not root.children:
            self._agenda_option_tree_summary = self._agenda_option_tree_record(
                root,
                parent_best,
                parent_best,
                reason="disabled_or_empty",
                evaluated=[],
                parent_eval=None,
                selected_eval=None,
            )
            return parent_best
        if not self._agenda_option_tree_parent_is_trigger(parent_best):
            self._agenda_option_tree_summary = self._agenda_option_tree_record(
                root,
                parent_best,
                parent_best,
                reason="parent_not_option_trigger",
                evaluated=[],
                parent_eval=None,
                selected_eval=None,
            )
            return parent_best

        reference = self._agenda_option_tree_metrics(
            root.state,
            root.commitment_state,
        )
        allowed = [
            edge
            for edge in root.children.values()
            if self._agenda_option_tree_candidate_allowed(edge)
        ]
        allowed.sort(key=self._agenda_tier_intra_key, reverse=True)
        allowed = allowed[: self.agenda_option_tree_max_root_candidates]
        if not allowed:
            self._agenda_option_tree_summary = self._agenda_option_tree_record(
                root,
                parent_best,
                parent_best,
                reason="no_allowed_option_candidate",
                evaluated=[],
                parent_eval=None,
                selected_eval=None,
            )
            return parent_best

        evaluated = [
            self._agenda_option_tree_evaluate_edge(root, edge, reference)
            for edge in allowed
        ]
        parent_eval = next(
            (row for row in evaluated if row["edge"] is parent_best),
            None,
        )
        if parent_eval is None:
            parent_eval = self._agenda_option_tree_evaluate_edge(
                root,
                parent_best,
                reference,
            )
            evaluated.append(parent_eval)
        best_eval = max(evaluated, key=self._agenda_option_tree_key)
        parent_option_tier = int(parent_eval["best_progress"]["agenda_tier"])
        best_option_tier = int(best_eval["best_progress"]["agenda_tier"])
        if best_option_tier < self.agenda_option_tree_min_tier:
            self._agenda_option_tree_summary = self._agenda_option_tree_record(
                root,
                parent_best,
                parent_best,
                reason="no_high_tier_option",
                evaluated=evaluated,
                parent_eval=parent_eval,
                selected_eval=parent_eval,
            )
            return parent_best
        if (
            best_option_tier
            < parent_option_tier + self.agenda_option_tree_min_tier_improvement
        ):
            self._agenda_option_tree_summary = self._agenda_option_tree_record(
                root,
                parent_best,
                parent_best,
                reason="parent_option_already_best_tier",
                evaluated=evaluated,
                parent_eval=parent_eval,
                selected_eval=parent_eval,
            )
            return parent_best

        selected = best_eval["edge"]
        self._agenda_option_tree_summary = self._agenda_option_tree_record(
            root,
            parent_best,
            selected,
            reason="higher_agenda_option_available",
            evaluated=evaluated,
            parent_eval=parent_eval,
            selected_eval=best_eval,
        )
        return selected

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._agenda_option_tree_summary = {}
        result = super().plan(state, commitment_state)
        result["agenda_option_tree"] = dict(self._agenda_option_tree_summary)
        return result


class RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner(
    RoomScheduleAgendaOptionTreeMacroGFTSDMCTSPlanner
):
    """A41 planner with bounded commitment to a selected agenda option.

    A41 keeps A40's option tree as the source of evidence, but tests the missing
    execution mechanism diagnosed by A40: once a short option has been selected
    and independently validated under the real transition model, the closed
    loop commits to the suffix for a bounded number of pushes.  The commitment
    is only enabled when the existing macro layer would otherwise execute a
    single push, so this diagnostic does not combine option commitment with the
    older corridor or release macro suffixes.
    """

    provider_id = "room_schedule_agenda_option_commitment_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        agenda_option_commitment_enabled: bool = True,
        agenda_option_commitment_min_tier: int = 7,
        agenda_option_commitment_require_done: bool = True,
        agenda_option_commitment_max_extra_pushes: int = 4,
        agenda_option_commitment_min_total_pushes: int = 2,
        agenda_option_commitment_abort_on_base_macro: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agenda_option_commitment_enabled = bool(
            agenda_option_commitment_enabled
        )
        self.agenda_option_commitment_min_tier = int(
            agenda_option_commitment_min_tier
        )
        self.agenda_option_commitment_require_done = bool(
            agenda_option_commitment_require_done
        )
        self.agenda_option_commitment_max_extra_pushes = max(
            int(agenda_option_commitment_max_extra_pushes),
            0,
        )
        self.agenda_option_commitment_min_total_pushes = max(
            int(agenda_option_commitment_min_total_pushes),
            1,
        )
        self.agenda_option_commitment_abort_on_base_macro = bool(
            agenda_option_commitment_abort_on_base_macro
        )

    def _agenda_option_commitment_record(
        self,
        *,
        triggered: bool,
        reason: str,
        option_tree: dict[str, Any] | None,
        selected_option: dict[str, Any] | None,
        validated_rows: list[dict[str, Any]] | None = None,
        final_progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress = {}
        if isinstance(selected_option, dict):
            maybe_progress = selected_option.get("best_progress", {})
            if isinstance(maybe_progress, dict):
                progress = maybe_progress
        actions = []
        if isinstance(selected_option, dict):
            maybe_actions = selected_option.get("best_actions", [])
            if isinstance(maybe_actions, list):
                actions = maybe_actions
        return {
            "agenda_option_commitment_enabled": bool(
                self.agenda_option_commitment_enabled
            ),
            "agenda_option_commitment_scope": (
                "bounded_option_commitment_e14_e6_v1"
            ),
            "agenda_option_commitment_min_tier": int(
                self.agenda_option_commitment_min_tier
            ),
            "agenda_option_commitment_require_done": bool(
                self.agenda_option_commitment_require_done
            ),
            "agenda_option_commitment_max_extra_pushes": int(
                self.agenda_option_commitment_max_extra_pushes
            ),
            "agenda_option_commitment_min_total_pushes": int(
                self.agenda_option_commitment_min_total_pushes
            ),
            "agenda_option_commitment_abort_on_base_macro": bool(
                self.agenda_option_commitment_abort_on_base_macro
            ),
            "agenda_option_commitment_triggered": bool(triggered),
            "agenda_option_commitment_reason": reason,
            "source_option_tree_reason": None
            if not isinstance(option_tree, dict)
            else option_tree.get("agenda_option_tree_reason"),
            "source_selected_option_tier": None
            if not isinstance(option_tree, dict)
            else option_tree.get("selected_option_tier"),
            "source_option_progress": progress,
            "proposed_push_count": int(len(actions)),
            "validated_push_count": int(len(validated_rows or [])),
            "committed_extra_push_count": int(max(0, len(validated_rows or []) - 1)),
            "proposed_actions": actions,
            "validated_rows": list(validated_rows or []),
            "final_progress": dict(final_progress or {}),
        }

    def _agenda_option_commitment_match_action(
        self,
        state: SokobanState,
        row: dict[str, Any],
    ) -> SokobanPushAction | None:
        physical = row.get("physical_action", {}) if isinstance(row, dict) else {}
        if not isinstance(physical, dict):
            return None
        try:
            flat_index = int(physical["flat_index"])
            box_index = int(physical["box_index"])
            direction = int(physical["direction"])
        except (KeyError, TypeError, ValueError):
            return None
        for action in self.model.legal_flows(state):
            if (
                int(action.flat_index) == flat_index
                and int(action.box_index) == box_index
                and int(action.direction) == direction
            ):
                return action
        return None

    def _agenda_option_commitment_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
        macro: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        option_tree = plan.get("agenda_option_tree", {})
        if not self.agenda_option_commitment_enabled:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="disabled",
                option_tree=option_tree,
                selected_option=None,
            )
        if plan.get("physical_action") is None:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="no_physical_action",
                option_tree=option_tree,
                selected_option=None,
            )
        if not isinstance(option_tree, dict) or not option_tree:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="no_option_tree",
                option_tree=option_tree,
                selected_option=None,
            )
        selected_option = option_tree.get("selected_option", {})
        if not isinstance(selected_option, dict) or not selected_option:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="no_selected_option",
                option_tree=option_tree,
                selected_option=None,
            )
        progress = selected_option.get("best_progress", {})
        if not isinstance(progress, dict):
            progress = {}
        if int(progress.get("agenda_tier", 0) or 0) < self.agenda_option_commitment_min_tier:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="selected_option_below_min_tier",
                option_tree=option_tree,
                selected_option=selected_option,
            )
        if (
            self.agenda_option_commitment_require_done
            and not bool(progress.get("agenda_done", False))
        ):
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="selected_option_not_done",
                option_tree=option_tree,
                selected_option=selected_option,
            )
        proposed_actions = selected_option.get("best_actions", [])
        if not isinstance(proposed_actions, list):
            proposed_actions = []
        keep = 1 + self.agenda_option_commitment_max_extra_pushes
        proposed_actions = proposed_actions[:keep]
        if len(proposed_actions) < self.agenda_option_commitment_min_total_pushes:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="selected_option_too_short",
                option_tree=option_tree,
                selected_option=selected_option,
            )

        base_actions = list(macro.get("physical_actions", []))
        if not base_actions:
            base_actions = [plan["physical_action"]]
        if (
            self.agenda_option_commitment_abort_on_base_macro
            and len(base_actions) != 1
        ):
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="base_macro_not_single_push",
                option_tree=option_tree,
                selected_option=selected_option,
            )
        first_action = plan["physical_action"]
        first_row = proposed_actions[0]
        first_physical = (
            first_row.get("physical_action", {})
            if isinstance(first_row, dict)
            else {}
        )
        if not isinstance(first_physical, dict) or int(
            first_physical.get("flat_index", -1)
        ) != int(first_action.flat_index):
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="first_action_mismatch",
                option_tree=option_tree,
                selected_option=selected_option,
            )

        reference = self._agenda_option_tree_metrics(state, commitment_state)
        current_state = state
        current_commitment = commitment_state
        seen_state_keys = {self.model.gft_canonical_key_cached(state)}
        seen_signatures = {reference["signature"]}
        actions: list[SokobanPushAction] = []
        commitments: list[RoomCommitmentState] = []
        rows: list[dict[str, Any]] = []
        path_loop = False

        for index, row in enumerate(proposed_actions):
            action = self._agenda_option_commitment_match_action(current_state, row)
            if action is None:
                return macro, self._agenda_option_commitment_record(
                    triggered=False,
                    reason="committed_action_illegal",
                    option_tree=option_tree,
                    selected_option=selected_option,
                    validated_rows=rows,
                )
            child = self.model.step_grid_flow(current_state, action)
            if self.is_blocked(child):
                return macro, self._agenda_option_commitment_record(
                    triggered=False,
                    reason="committed_action_blocked",
                    option_tree=option_tree,
                    selected_option=selected_option,
                    validated_rows=rows,
                )
            if index == 0 and macro.get("after_commitment_states"):
                next_commitment = list(macro["after_commitment_states"])[0]
                next_commitment = self.model.normalize_room_commitment_state(
                    child,
                    next_commitment,
                )
            elif index == 0 and plan.get("next_commitment_state") is not None:
                next_commitment = self.model.normalize_room_commitment_state(
                    child,
                    plan["next_commitment_state"],
                )
            else:
                next_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
            child_key = self.model.gft_canonical_key_cached(child)
            child_metrics = self._agenda_option_tree_metrics(child, next_commitment)
            path_loop = bool(path_loop or child_key in seen_state_keys)
            rows.append(
                {
                    "macro_index": int(index),
                    "physical_action": self._physical_action_dict(
                        current_state,
                        action,
                    ),
                    "reason": "agenda_option_commitment_initial_push"
                    if index == 0
                    else "agenda_option_commitment_suffix_push",
                    "path_loop_after": bool(path_loop),
                    "schedule_signature_repeated_after": bool(
                        child_metrics["signature"] in seen_signatures
                    ),
                    "boxes_on_goals_after": int(
                        child_metrics["boxes_on_goals"]
                    ),
                    "release_action_count_after": int(
                        child_metrics["release_action_count"]
                    ),
                    "active_blocker_count_after": int(
                        child_metrics["active_blocker_count"]
                    ),
                    "slot_schedule_cost_after": float(
                        child_metrics["slot_schedule_cost"]
                    ),
                    "nonblocked_legal_push_count_after": int(
                        child_metrics["nonblocked_legal_push_count"]
                    ),
                }
            )
            actions.append(action)
            commitments.append(next_commitment)
            current_state = child
            current_commitment = next_commitment
            seen_state_keys.add(child_key)
            seen_signatures.add(child_metrics["signature"])
            if path_loop:
                return macro, self._agenda_option_commitment_record(
                    triggered=False,
                    reason="committed_option_path_loop",
                    option_tree=option_tree,
                    selected_option=selected_option,
                    validated_rows=rows,
                )

        final_metrics = self._agenda_option_tree_metrics(
            current_state,
            current_commitment,
        )
        final_progress = self._agenda_option_tree_progress(
            reference,
            final_metrics,
            depth=len(actions),
            repeated_context=bool(final_metrics["signature"] in self.recent_schedule_keys),
            path_loop=False,
        )
        if int(final_progress.get("agenda_tier", 0)) < self.agenda_option_commitment_min_tier:
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="validated_option_below_min_tier",
                option_tree=option_tree,
                selected_option=selected_option,
                validated_rows=rows,
                final_progress=final_progress,
            )
        if (
            self.agenda_option_commitment_require_done
            and not bool(final_progress.get("agenda_done", False))
        ):
            return macro, self._agenda_option_commitment_record(
                triggered=False,
                reason="validated_option_not_done",
                option_tree=option_tree,
                selected_option=selected_option,
                validated_rows=rows,
                final_progress=final_progress,
            )

        committed_macro = {
            **macro,
            "enabled": True,
            "triggered": len(actions)
            >= self.agenda_option_commitment_min_total_pushes,
            "reason": "agenda_option_commitment",
            "physical_actions": actions,
            "after_commitment_states": commitments,
            "rows": rows,
            "macro_push_count": int(len(actions)),
            "extra_push_count": int(max(0, len(actions) - 1)),
            "agenda_option_commitment_enabled": True,
            "agenda_option_commitment_triggered": True,
            "agenda_option_commitment_reason": "validated_option_committed",
            "agenda_option_commitment_source_reason": option_tree.get(
                "agenda_option_tree_reason"
            ),
            "agenda_option_commitment_final_progress": final_progress,
        }
        record = self._agenda_option_commitment_record(
            triggered=True,
            reason="validated_option_committed",
            option_tree=option_tree,
            selected_option=selected_option,
            validated_rows=rows,
            final_progress=final_progress,
        )
        return committed_macro, record

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(
            state,
            commitment_state,
        )
        plan = super().plan_with_corridor_macro(state, commitment_state)
        macro = plan.get("corridor_macro", {})
        committed_macro, record = self._agenda_option_commitment_from_plan(
            state,
            commitment_state,
            plan,
            macro,
        )
        plan["corridor_macro"] = committed_macro
        plan["agenda_option_commitment"] = record
        return plan


class RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner(
    RoomScheduleAgendaOptionCommitmentMacroGFTSDMCTSPlanner
):
    """A42 planner with a post-release fill-frontier agenda option.

    A41 proves that a release-consumption option can be committed, but the
    following clean fill stage still lacks a milestone consumer.  A42 keeps the
    same hard-instance inheritance line and promotes one specific agenda
    boundary: completing the current frontier slot.  If a legal root edge fills
    a frontier slot without regressing solved goals, it may override a parent
    no-progress fill action, even when that completion opens the next release
    frontier.
    """

    provider_id = "room_schedule_fill_frontier_option_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        fill_frontier_option_enabled: bool = True,
        fill_frontier_option_mode: str = "post_release_clean_fill",
        fill_frontier_option_min_slot_delta: float = 1.0,
        fill_frontier_option_require_goal_nonregression: bool = True,
        fill_frontier_option_parent_max_tier: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fill_frontier_option_enabled = bool(fill_frontier_option_enabled)
        self.fill_frontier_option_mode = str(fill_frontier_option_mode)
        self.fill_frontier_option_min_slot_delta = float(
            fill_frontier_option_min_slot_delta
        )
        self.fill_frontier_option_require_goal_nonregression = bool(
            fill_frontier_option_require_goal_nonregression
        )
        self.fill_frontier_option_parent_max_tier = int(
            fill_frontier_option_parent_max_tier
        )
        self._fill_frontier_option_summary: dict[str, Any] = {}

    def _fill_frontier_option_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> dict[str, Any]:
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        stage = self._decision_stage_summary(state, commitment_state)
        legal = int(self._nonblocked_legal_push_count(state))
        return {
            "boxes_on_goals": int(self.model.boxes_on_goals(state)),
            "stage": str(stage.get("stage")),
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "valid_filled_slot_count": int(slot["valid_filled_slot_count"]),
            "committed_satisfied_count": int(slot["committed_satisfied_count"]),
            "committed_box_count": int(slot["committed_box_count"]),
            "assigned_slot_count": int(slot["assigned_slot_count"]),
            "open_frontier_slot_count": int(slot["open_frontier_slot_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "dependency_total": int(totals["dependency_count"])
            + int(totals["future_dependency_count"]),
            "dependency_count": int(totals["dependency_count"]),
            "future_dependency_count": int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "destructive_action_count": int(totals["destructive_action_count"]),
            "schedule_barrier_cost": float(totals["schedule_barrier_cost"]),
            "nonblocked_legal_push_count": int(legal),
            "no_legal_pushes": bool(legal <= 0),
            "signature": self.schedule_stall_signature(state, commitment_state),
        }

    def _fill_frontier_parent_is_trigger(
        self,
        edge: RoomSemanticEdge,
        before: dict[str, Any],
    ) -> bool:
        if not self.fill_frontier_option_enabled:
            return False
        if self.fill_frontier_option_mode == "global":
            return True
        if str(before.get("stage")) != "fill":
            return False
        if float(before.get("slot_schedule_cost", 0.0)) <= 0.0:
            return False
        if int(before.get("release_action_count", 0)) != 0:
            return False
        if int(before.get("dependency_total", 0)) != 0:
            return False
        if int(before.get("target_blocker_count", 0)) != 0:
            return False
        if int(before.get("valid_filled_slot_count", 0)) <= 0:
            return False
        cert = self._agenda_tier_certificate(edge)
        agenda_type = str(cert.get("active_agenda_type", ""))
        tier = self._agenda_tier_value(edge)
        parent_progress = bool(cert.get("active_agenda_progress", False))
        return bool(
            agenda_type == "fill_frontier_slot"
            and (
                tier <= self.fill_frontier_option_parent_max_tier
                or not parent_progress
                or bool(cert.get("active_agenda_off_agenda_action", False))
            )
        )

    def _fill_frontier_progress(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        goal_delta = int(after["boxes_on_goals"]) - int(before["boxes_on_goals"])
        slot_delta = float(before["slot_schedule_cost"]) - float(
            after["slot_schedule_cost"]
        )
        valid_delta = int(after["valid_filled_slot_count"]) - int(
            before["valid_filled_slot_count"]
        )
        satisfied_delta = int(after["committed_satisfied_count"]) - int(
            before["committed_satisfied_count"]
        )
        release_delta = int(after["release_action_count"]) - int(
            before["release_action_count"]
        )
        dependency_delta = int(after["dependency_total"]) - int(
            before["dependency_total"]
        )
        target_delta = int(after["target_blocker_count"]) - int(
            before["target_blocker_count"]
        )
        destructive_delta = int(after["destructive_action_count"]) - int(
            before["destructive_action_count"]
        )
        legal_delta = int(after["nonblocked_legal_push_count"]) - int(
            before["nonblocked_legal_push_count"]
        )
        frontier_completed = bool(valid_delta > 0 or satisfied_delta > 0)
        slot_progress = bool(slot_delta >= self.fill_frontier_option_min_slot_delta)
        goal_nonregression = bool(goal_delta >= 0)
        admissible = bool(
            not bool(after["no_legal_pushes"])
            and (frontier_completed or slot_progress)
            and (
                goal_nonregression
                or not self.fill_frontier_option_require_goal_nonregression
            )
        )
        tier = 0
        if bool(after["no_legal_pushes"]) and goal_delta <= 0:
            tier = 0
        elif not goal_nonregression and self.fill_frontier_option_require_goal_nonregression:
            tier = 1
        elif frontier_completed:
            tier = 9
        elif slot_progress:
            tier = 7
        else:
            tier = 2
        score = (
            100.0 * float(tier)
            + 40.0 * float(frontier_completed)
            + 8.0 * float(max(0.0, min(slot_delta, 24.0)))
            + 20.0 * float(max(0, valid_delta + satisfied_delta))
            + 4.0 * float(max(0, legal_delta))
            - 35.0 * float(max(0, -goal_delta))
            - 2.0 * float(max(0, release_delta))
            - 2.0 * float(max(0, dependency_delta))
            - 2.0 * float(max(0, target_delta))
            - 1.0 * float(max(0, destructive_delta))
            - 40.0 * float(after["no_legal_pushes"])
        )
        return {
            "agenda_type": "fill_frontier_slot",
            "agenda_tier": int(tier),
            "admissible": bool(admissible),
            "frontier_completed": bool(frontier_completed),
            "slot_progress": bool(slot_progress),
            "goal_nonregression": bool(goal_nonregression),
            "goal_delta": int(goal_delta),
            "slot_schedule_cost_delta": float(slot_delta),
            "valid_filled_slot_count_delta": int(valid_delta),
            "committed_satisfied_count_delta": int(satisfied_delta),
            "release_action_count_delta": int(release_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_delta),
            "destructive_action_count_delta": int(destructive_delta),
            "legal_push_count_delta": int(legal_delta),
            "after_slot_schedule_cost": float(after["slot_schedule_cost"]),
            "after_valid_filled_slot_count": int(after["valid_filled_slot_count"]),
            "after_release_action_count": int(after["release_action_count"]),
            "after_dependency_total": int(after["dependency_total"]),
            "after_target_blocker_count": int(after["target_blocker_count"]),
            "after_no_legal_pushes": bool(after["no_legal_pushes"]),
            "score": float(score),
        }

    def _fill_frontier_option_key(self, row: dict[str, Any]) -> tuple:
        progress = row["progress"]
        edge = row["edge"]
        return (
            int(progress["admissible"]),
            int(progress["agenda_tier"]),
            int(progress["frontier_completed"]),
            int(progress["slot_progress"]),
            int(progress["goal_nonregression"]),
            float(progress["slot_schedule_cost_delta"]),
            int(progress["valid_filled_slot_count_delta"])
            + int(progress["committed_satisfied_count_delta"]),
            -int(max(0, progress["release_action_count_delta"])),
            -int(max(0, progress["dependency_total_delta"])),
            -int(max(0, progress["target_blocker_count_delta"])),
            float(progress["score"]),
            *self._agenda_tier_intra_key(edge),
        )

    def _fill_frontier_action_summary(
        self,
        edge: RoomSemanticEdge,
        progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._agenda_tier_action_summary(edge) or {}
        if progress is not None:
            row["fill_frontier_progress"] = progress
        return row

    def _fill_frontier_option_record(
        self,
        root: RoomSemanticNode,
        parent_best: RoomSemanticEdge,
        selected: RoomSemanticEdge,
        *,
        reason: str,
        before: dict[str, Any] | None,
        evaluated: list[dict[str, Any]],
        selected_row: dict[str, Any] | None,
    ) -> dict[str, Any]:
        compact = []
        for row in sorted(evaluated, key=self._fill_frontier_option_key, reverse=True):
            compact.append(
                {
                    "action": self._fill_frontier_action_summary(
                        row["edge"],
                        row["progress"],
                    ),
                    "progress": row["progress"],
                    "after_metrics": {
                        key: value
                        for key, value in row["after"].items()
                        if key != "signature"
                    },
                }
            )
        return {
            "fill_frontier_option_enabled": bool(self.fill_frontier_option_enabled),
            "fill_frontier_option_scope": (
                "post_release_fill_frontier_option_e4_e14_v1"
            ),
            "fill_frontier_option_mode": self.fill_frontier_option_mode,
            "fill_frontier_option_min_slot_delta": float(
                self.fill_frontier_option_min_slot_delta
            ),
            "fill_frontier_option_require_goal_nonregression": bool(
                self.fill_frontier_option_require_goal_nonregression
            ),
            "fill_frontier_option_parent_max_tier": int(
                self.fill_frontier_option_parent_max_tier
            ),
            "fill_frontier_option_override": bool(selected is not parent_best),
            "fill_frontier_option_reason": reason,
            "candidate_count": int(len(root.children)),
            "evaluated_option_count": int(len(evaluated)),
            "before_metrics": None
            if before is None
            else {
                key: value
                for key, value in before.items()
                if key != "signature"
            },
            "parent_action": self._agenda_tier_action_summary(parent_best),
            "selected_action": self._agenda_tier_action_summary(selected),
            "selected_progress": None
            if selected_row is None
            else selected_row["progress"],
            "top_fill_options": compact[:8],
        }

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        parent_best = super().choose_root_edge(root)
        if not self.fill_frontier_option_enabled or not root.children:
            self._fill_frontier_option_summary = self._fill_frontier_option_record(
                root,
                parent_best,
                parent_best,
                reason="disabled_or_empty",
                before=None,
                evaluated=[],
                selected_row=None,
            )
            return parent_best
        before = self._fill_frontier_option_metrics(
            root.state,
            root.commitment_state,
        )
        if not self._fill_frontier_parent_is_trigger(parent_best, before):
            self._fill_frontier_option_summary = self._fill_frontier_option_record(
                root,
                parent_best,
                parent_best,
                reason="parent_not_fill_frontier_trigger",
                before=before,
                evaluated=[],
                selected_row=None,
            )
            return parent_best
        evaluated = []
        for edge in root.children.values():
            after = self._fill_frontier_option_metrics(
                edge.child_state,
                edge.child_commitment_state,
            )
            progress = self._fill_frontier_progress(before, after)
            if bool(progress["admissible"]):
                evaluated.append(
                    {
                        "edge": edge,
                        "after": after,
                        "progress": progress,
                    }
                )
        if not evaluated:
            self._fill_frontier_option_summary = self._fill_frontier_option_record(
                root,
                parent_best,
                parent_best,
                reason="no_admissible_fill_frontier_option",
                before=before,
                evaluated=[],
                selected_row=None,
            )
            return parent_best
        selected_row = max(evaluated, key=self._fill_frontier_option_key)
        selected = selected_row["edge"]
        parent_row = next(
            (row for row in evaluated if row["edge"] is parent_best),
            None,
        )
        if parent_row is not None and self._fill_frontier_option_key(parent_row) >= self._fill_frontier_option_key(selected_row):
            self._fill_frontier_option_summary = self._fill_frontier_option_record(
                root,
                parent_best,
                parent_best,
                reason="parent_already_best_fill_frontier_option",
                before=before,
                evaluated=evaluated,
                selected_row=parent_row,
            )
            return parent_best
        self._fill_frontier_option_summary = self._fill_frontier_option_record(
            root,
            parent_best,
            selected,
            reason="fill_frontier_option_selected",
            before=before,
            evaluated=evaluated,
            selected_row=selected_row,
        )
        return selected

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._fill_frontier_option_summary = {}
        result = super().plan(state, commitment_state)
        result["fill_frontier_option"] = dict(self._fill_frontier_option_summary)
        return result


class RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner(
    RoomScheduleFillFrontierOptionMacroGFTSDMCTSPlanner
):
    """A43 planner with a bounded release-blocker option evaluator.

    A42 completes a post-release fill frontier and exposes the next agenda:
    after the frontier is filled, the state returns to a release-blocker
    obligation with no open release action.  A43 keeps the same hard-instance
    line and evaluates whether a candidate first push can reach release-blocker
    progress within a small local push tree.  Like A40, it only selects the
    first edge; it does not execute the suffix as a macro.
    """

    provider_id = "room_schedule_release_blocker_option_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        release_blocker_option_enabled: bool = True,
        release_blocker_option_mode: str = "post_frontier_release_blocker",
        release_blocker_option_max_pushes: int = 5,
        release_blocker_option_max_nodes: int = 96,
        release_blocker_option_max_root_candidates: int = 12,
        release_blocker_option_min_tier: int = 5,
        release_blocker_option_min_tier_improvement: int = 1,
        release_blocker_option_parent_max_tier: int = 2,
        release_blocker_option_min_valid_filled_slots: int = 2,
        release_blocker_option_min_active_blockers: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.release_blocker_option_enabled = bool(release_blocker_option_enabled)
        self.release_blocker_option_mode = str(release_blocker_option_mode)
        self.release_blocker_option_max_pushes = max(
            int(release_blocker_option_max_pushes),
            1,
        )
        self.release_blocker_option_max_nodes = max(
            int(release_blocker_option_max_nodes),
            1,
        )
        self.release_blocker_option_max_root_candidates = max(
            int(release_blocker_option_max_root_candidates),
            1,
        )
        self.release_blocker_option_min_tier = int(release_blocker_option_min_tier)
        self.release_blocker_option_min_tier_improvement = max(
            int(release_blocker_option_min_tier_improvement),
            0,
        )
        self.release_blocker_option_parent_max_tier = int(
            release_blocker_option_parent_max_tier
        )
        self.release_blocker_option_min_valid_filled_slots = max(
            int(release_blocker_option_min_valid_filled_slots),
            0,
        )
        self.release_blocker_option_min_active_blockers = max(
            int(release_blocker_option_min_active_blockers),
            0,
        )
        self._release_blocker_option_summary: dict[str, Any] = {}

    def _release_blocker_option_reference(
        self,
        root: RoomSemanticNode,
    ) -> dict[str, Any]:
        reference = self._agenda_option_tree_metrics(
            root.state,
            root.commitment_state,
        )
        slot = self.model.room_slot_progress_summary(
            root.state,
            root.commitment_state,
        )
        reference["valid_filled_slot_count"] = int(
            slot["valid_filled_slot_count"]
        )
        reference["committed_satisfied_count"] = int(
            slot["committed_satisfied_count"]
        )
        reference["open_frontier_slot_count"] = int(
            slot["open_frontier_slot_count"]
        )
        return reference

    def _release_blocker_parent_is_trigger(
        self,
        edge: RoomSemanticEdge,
        reference: dict[str, Any],
    ) -> bool:
        if not self.release_blocker_option_enabled:
            return False
        cert = self._agenda_tier_certificate(edge)
        if self.release_blocker_option_mode == "global":
            return True
        if str(reference.get("stage")) != "release":
            return False
        if int(reference.get("release_action_count", 0)) != 0:
            return False
        if (
            int(reference.get("active_blocker_count", 0))
            < self.release_blocker_option_min_active_blockers
        ):
            return False
        if (
            int(reference.get("valid_filled_slot_count", 0))
            < self.release_blocker_option_min_valid_filled_slots
        ):
            return False
        if str(cert.get("active_agenda_type", "")) != "release_blocker":
            return False
        tier = self._agenda_tier_value(edge)
        return bool(
            tier <= self.release_blocker_option_parent_max_tier
            or not bool(cert.get("active_agenda_progress", False))
            or bool(cert.get("active_agenda_off_agenda_action", False))
        )

    def _release_blocker_candidate_allowed(
        self,
        edge: RoomSemanticEdge,
    ) -> bool:
        if self.release_blocker_option_mode == "global":
            return self._agenda_tier_value(edge) > 0
        cert = self._agenda_tier_certificate(edge)
        return bool(
            str(cert.get("active_agenda_type", "")) == "release_blocker"
            and self._agenda_tier_value(edge) > 0
        )

    def _agenda_option_tree_progress(
        self,
        reference: dict[str, Any],
        metrics: dict[str, Any],
        *,
        depth: int,
        repeated_context: bool,
        path_loop: bool,
    ) -> dict[str, Any]:
        progress = super()._agenda_option_tree_progress(
            reference,
            metrics,
            depth=depth,
            repeated_context=repeated_context,
            path_loop=path_loop,
        )
        if str(progress.get("agenda_type")) != "release_blocker":
            return progress

        kinds = []
        if bool(progress.get("goal_progress", False)):
            kinds.append("goal_progress")
        if bool(progress.get("release_opened", False)):
            kinds.append("release_opened")
        for key, name in (
            ("active_blocker_delta", "active_blocker_reduced"),
            ("dependency_total_delta", "dependency_reduced"),
            ("target_blocker_count_delta", "target_blocker_reduced"),
            ("locked_stand_blocker_delta", "locked_stand_reduced"),
            ("goal_locked_stand_blocker_delta", "goal_locked_stand_reduced"),
        ):
            if int(progress.get(key, 0)) > 0:
                kinds.append(name)
        true_progress = bool(kinds)
        progress["release_blocker_true_progress"] = bool(true_progress)
        progress["release_blocker_progress_kinds"] = kinds
        if true_progress:
            return progress

        strict_loop = bool(
            progress.get("agenda_loop", False)
            or progress.get("path_loop", False)
            or progress.get("repeated_context", False)
        )
        strict_tier = 1 if strict_loop else 2
        progress["agenda_progress"] = False
        progress["agenda_done"] = False
        progress["off_agenda"] = True
        progress["agenda_loop"] = bool(strict_loop)
        progress["agenda_tier"] = int(min(int(progress.get("agenda_tier", 0)), strict_tier))
        progress["score"] = float(
            100.0 * float(progress["agenda_tier"])
            + 1.0 * float(progress.get("legal_push_count_delta", 0))
            - 30.0 * float(strict_loop)
            - 40.0
            * float(
                bool(progress.get("after_no_legal_pushes", False))
                and int(progress.get("goal_delta", 0)) <= 0
            )
            - 3.0 * float(depth)
        )
        return progress

    def _release_blocker_evaluate_edge(
        self,
        root: RoomSemanticNode,
        edge: RoomSemanticEdge,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        old_pushes = self.agenda_option_tree_max_pushes
        old_nodes = self.agenda_option_tree_max_nodes
        try:
            self.agenda_option_tree_max_pushes = self.release_blocker_option_max_pushes
            self.agenda_option_tree_max_nodes = self.release_blocker_option_max_nodes
            return self._agenda_option_tree_evaluate_edge(root, edge, reference)
        finally:
            self.agenda_option_tree_max_pushes = old_pushes
            self.agenda_option_tree_max_nodes = old_nodes

    def _release_blocker_compact_option(
        self,
        row: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "root_action": row["root_action"],
            "root_agenda_tier": int(row["root_agenda_tier"]),
            "best_progress": row["best_progress"],
            "best_actions": row["best_actions"][: self.release_blocker_option_max_pushes],
            "best_depth": int(row["best_depth"]),
            "expanded_nodes": int(row["expanded_nodes"]),
            "option_score": float(row["option_score"]),
        }

    def _release_blocker_option_record(
        self,
        root: RoomSemanticNode,
        parent_best: RoomSemanticEdge,
        selected: RoomSemanticEdge,
        *,
        reason: str,
        reference: dict[str, Any] | None,
        evaluated: list[dict[str, Any]],
        parent_eval: dict[str, Any] | None,
        selected_eval: dict[str, Any] | None,
    ) -> dict[str, Any]:
        compact = [
            self._release_blocker_compact_option(row)
            for row in sorted(evaluated, key=self._agenda_option_tree_key, reverse=True)
        ]
        compact = [row for row in compact if row is not None]
        parent_option_tier = (
            None
            if parent_eval is None
            else int(parent_eval["best_progress"]["agenda_tier"])
        )
        selected_option_tier = (
            None
            if selected_eval is None
            else int(selected_eval["best_progress"]["agenda_tier"])
        )
        return {
            "release_blocker_option_enabled": bool(
                self.release_blocker_option_enabled
            ),
            "release_blocker_option_scope": (
                "post_frontier_release_blocker_option_e14_v1"
            ),
            "release_blocker_option_mode": self.release_blocker_option_mode,
            "release_blocker_option_max_pushes": int(
                self.release_blocker_option_max_pushes
            ),
            "release_blocker_option_max_nodes": int(
                self.release_blocker_option_max_nodes
            ),
            "release_blocker_option_max_root_candidates": int(
                self.release_blocker_option_max_root_candidates
            ),
            "release_blocker_option_min_tier": int(
                self.release_blocker_option_min_tier
            ),
            "release_blocker_option_min_tier_improvement": int(
                self.release_blocker_option_min_tier_improvement
            ),
            "release_blocker_option_parent_max_tier": int(
                self.release_blocker_option_parent_max_tier
            ),
            "release_blocker_option_min_valid_filled_slots": int(
                self.release_blocker_option_min_valid_filled_slots
            ),
            "release_blocker_option_override": bool(selected is not parent_best),
            "release_blocker_option_reason": reason,
            "candidate_count": int(len(root.children)),
            "evaluated_option_count": int(len(evaluated)),
            "parent_option_tier": parent_option_tier,
            "selected_option_tier": selected_option_tier,
            "before_metrics": None
            if reference is None
            else {
                key: value
                for key, value in reference.items()
                if key != "signature"
            },
            "parent_action": self._agenda_tier_action_summary(parent_best),
            "selected_action": self._agenda_tier_action_summary(selected),
            "parent_option": self._release_blocker_compact_option(parent_eval),
            "selected_option": self._release_blocker_compact_option(selected_eval),
            "top_release_blocker_options": compact[:8],
        }

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        parent_best = super().choose_root_edge(root)
        if not self.release_blocker_option_enabled or not root.children:
            self._release_blocker_option_summary = (
                self._release_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="disabled_or_empty",
                    reference=None,
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        reference = self._release_blocker_option_reference(root)
        if not self._release_blocker_parent_is_trigger(parent_best, reference):
            self._release_blocker_option_summary = (
                self._release_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="parent_not_release_blocker_trigger",
                    reference=reference,
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        allowed = [
            edge
            for edge in root.children.values()
            if self._release_blocker_candidate_allowed(edge)
        ]
        allowed.sort(key=self._agenda_tier_intra_key, reverse=True)
        allowed = allowed[: self.release_blocker_option_max_root_candidates]
        if not allowed:
            self._release_blocker_option_summary = (
                self._release_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="no_allowed_release_blocker_candidate",
                    reference=reference,
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        evaluated = [
            self._release_blocker_evaluate_edge(root, edge, reference)
            for edge in allowed
        ]
        parent_eval = next(
            (row for row in evaluated if row["edge"] is parent_best),
            None,
        )
        if parent_eval is None:
            parent_eval = self._release_blocker_evaluate_edge(
                root,
                parent_best,
                reference,
            )
            evaluated.append(parent_eval)
        best_eval = max(evaluated, key=self._agenda_option_tree_key)
        parent_option_tier = int(parent_eval["best_progress"]["agenda_tier"])
        best_option_tier = int(best_eval["best_progress"]["agenda_tier"])
        if best_option_tier < self.release_blocker_option_min_tier:
            self._release_blocker_option_summary = (
                self._release_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="no_high_tier_release_blocker_option",
                    reference=reference,
                    evaluated=evaluated,
                    parent_eval=parent_eval,
                    selected_eval=parent_eval,
                )
            )
            return parent_best
        if (
            best_option_tier
            < parent_option_tier + self.release_blocker_option_min_tier_improvement
        ):
            self._release_blocker_option_summary = (
                self._release_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="parent_release_blocker_option_already_best_tier",
                    reference=reference,
                    evaluated=evaluated,
                    parent_eval=parent_eval,
                    selected_eval=parent_eval,
                )
            )
            return parent_best

        selected = best_eval["edge"]
        self._release_blocker_option_summary = (
            self._release_blocker_option_record(
                root,
                parent_best,
                selected,
                reason="higher_release_blocker_option_available",
                reference=reference,
                evaluated=evaluated,
                parent_eval=parent_eval,
                selected_eval=best_eval,
            )
        )
        return selected

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._release_blocker_option_summary = {}
        result = super().plan(state, commitment_state)
        result["release_blocker_option"] = dict(
            self._release_blocker_option_summary
        )
        return result


class RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner(
    RoomScheduleReleaseBlockerOptionMacroGFTSDMCTSPlanner
):
    """A44 planner with blocker-specific release obligations.

    A43 shows that the post-frontier release-blocker state has no short-range
    aggregate blocker reduction under the current option tree.  A44 keeps the
    same inheritance line but changes the unit of diagnosis: instead of treating
    all blockers as one count, it binds an option search to concrete A11
    dependency rows.  The online planner still uses only state-derived
    structure and never calls a teacher/reference policy.
    """

    provider_id = "room_schedule_active_blocker_obligation_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        active_blocker_obligation_enabled: bool = True,
        active_blocker_obligation_mode: str = "post_frontier_release_blocker",
        active_blocker_obligation_max_pushes: int = 6,
        active_blocker_obligation_max_nodes: int = 128,
        active_blocker_obligation_max_root_candidates: int = 12,
        active_blocker_obligation_max_obligations: int = 4,
        active_blocker_obligation_min_tier: int = 5,
        active_blocker_obligation_min_tier_improvement: int = 1,
        active_blocker_obligation_parent_max_tier: int = 2,
        active_blocker_obligation_min_valid_filled_slots: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.active_blocker_obligation_enabled = bool(
            active_blocker_obligation_enabled
        )
        self.active_blocker_obligation_mode = str(active_blocker_obligation_mode)
        self.active_blocker_obligation_max_pushes = max(
            int(active_blocker_obligation_max_pushes),
            1,
        )
        self.active_blocker_obligation_max_nodes = max(
            int(active_blocker_obligation_max_nodes),
            1,
        )
        self.active_blocker_obligation_max_root_candidates = max(
            int(active_blocker_obligation_max_root_candidates),
            1,
        )
        self.active_blocker_obligation_max_obligations = max(
            int(active_blocker_obligation_max_obligations),
            1,
        )
        self.active_blocker_obligation_min_tier = int(
            active_blocker_obligation_min_tier
        )
        self.active_blocker_obligation_min_tier_improvement = max(
            int(active_blocker_obligation_min_tier_improvement),
            0,
        )
        self.active_blocker_obligation_parent_max_tier = int(
            active_blocker_obligation_parent_max_tier
        )
        self.active_blocker_obligation_min_valid_filled_slots = max(
            int(active_blocker_obligation_min_valid_filled_slots),
            0,
        )
        self._active_blocker_obligation_summary: dict[str, Any] = {}

    @staticmethod
    def _active_blocker_relevant_kinds(row: dict[str, Any]) -> set[str]:
        kinds = set(row.get("dependency_kinds", []))
        return {
            kind
            for kind in kinds
            if kind
            in {
                "stand_cell_occupied_by_box",
                "stand_cell_unreachable_by_player",
                "target_occupied_by_box",
            }
        }

    def _active_blocker_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        data = self._dependency_to_dict(row)
        data["selected_dependency_kinds"] = sorted(
            self._active_blocker_relevant_kinds(row)
        )
        data["active_blocker_obligation_key"] = [
            list(item) if isinstance(item, tuple) else item
            for item in self._dependency_relation_key(row)
        ]
        return data

    def _active_blocker_obligations(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
    ) -> list[dict[str, Any]]:
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        rows = [
            row
            for row in schedule.get("dependencies", [])
            if self._active_blocker_relevant_kinds(row)
        ]

        def priority(row: dict[str, Any]) -> tuple:
            kinds = self._active_blocker_relevant_kinds(row)
            return (
                int(not bool(row.get("future", False))),
                int("stand_cell_occupied_by_box" in kinds),
                int("target_occupied_by_box" in kinds),
                int("stand_cell_unreachable_by_player" in kinds),
                -int(row.get("horizon_index", 0)),
                float(row.get("candidate_barrier_cost", 0.0)),
            )

        rows.sort(key=priority, reverse=True)
        return [
            self._active_blocker_to_dict(row)
            for row in rows[: self.active_blocker_obligation_max_obligations]
        ]

    def _active_blocker_parent_is_trigger(
        self,
        edge: RoomSemanticEdge,
        reference: dict[str, Any],
    ) -> bool:
        if not self.active_blocker_obligation_enabled:
            return False
        cert = self._agenda_tier_certificate(edge)
        if self.active_blocker_obligation_mode == "global":
            return True
        if str(reference.get("stage")) != "release":
            return False
        if int(reference.get("release_action_count", 0)) != 0:
            return False
        if int(reference.get("active_blocker_count", 0)) <= 0:
            return False
        if (
            int(reference.get("valid_filled_slot_count", 0))
            < self.active_blocker_obligation_min_valid_filled_slots
        ):
            return False
        if str(cert.get("active_agenda_type", "")) != "release_blocker":
            return False
        tier = self._agenda_tier_value(edge)
        return bool(
            tier <= self.active_blocker_obligation_parent_max_tier
            or not bool(cert.get("active_agenda_progress", False))
            or bool(cert.get("active_agenda_off_agenda_action", False))
        )

    def _active_blocker_candidate_allowed(self, edge: RoomSemanticEdge) -> bool:
        if self.active_blocker_obligation_mode == "global":
            return self._agenda_tier_value(edge) > 0
        cert = self._agenda_tier_certificate(edge)
        return bool(
            str(cert.get("active_agenda_type", "")) == "release_blocker"
            and self._agenda_tier_value(edge) > 0
        )

    def _active_blocker_progress_summary(
        self,
        progress: dict[str, Any],
        metrics: dict[str, Any],
        *,
        depth: int,
        path_loop: bool,
    ) -> dict[str, Any]:
        kinds: list[str] = []
        if bool(progress.get("row_disappeared", False)):
            kinds.append("row_disappeared")
        if int(progress.get("remaining_selected_kind_delta", 0)) > 0:
            kinds.append("selected_kind_reduced")
        if bool(progress.get("target_resolved", False)):
            kinds.append("target_resolved")
        if bool(progress.get("stand_resolved", False)):
            kinds.append("stand_resolved")
        if int(progress.get("selected_blocker_delta", 0)) > 0:
            kinds.append("selected_blocker_reduced")
        if int(progress.get("selected_release_action_delta", 0)) > 0:
            kinds.append("selected_release_opened")
        if int(progress.get("release_action_delta", 0)) > 0:
            kinds.append("release_opened")

        true_progress = bool(
            "row_disappeared" in kinds
            or "selected_kind_reduced" in kinds
            or "target_resolved" in kinds
            or "stand_resolved" in kinds
            or "selected_blocker_reduced" in kinds
        )
        selected_release_opened = bool("selected_release_opened" in kinds)
        release_opened = bool("release_opened" in kinds)
        aggregate_progress = bool(
            int(progress.get("dependency_total_delta", 0)) > 0
            or int(progress.get("target_blocker_count_delta", 0)) > 0
            or int(progress.get("unreachable_stand_count_delta", 0)) > 0
        )
        if true_progress:
            tier = 8
        elif selected_release_opened:
            tier = 6
        elif release_opened or aggregate_progress:
            tier = 4
        elif path_loop or bool(metrics.get("no_legal_pushes", False)):
            tier = 1
        else:
            tier = 2

        score = (
            100.0 * float(tier)
            + float(progress.get("score", 0.0))
            + 28.0 * float(true_progress)
            + 18.0 * float(selected_release_opened)
            + 8.0 * float(max(0, int(progress.get("dependency_total_delta", 0))))
            + 8.0
            * float(max(0, int(progress.get("target_blocker_count_delta", 0))))
            + 6.0
            * float(max(0, int(progress.get("unreachable_stand_count_delta", 0))))
            + 1.5
            * float(max(0, int(progress.get("legal_push_count_delta", 0))))
            - 25.0 * float(path_loop)
            - 35.0 * float(bool(metrics.get("no_legal_pushes", False)))
            - 2.0 * float(depth)
        )
        result = dict(progress)
        result.update(
            {
                "active_blocker_obligation_tier": int(tier),
                "active_blocker_true_progress": bool(true_progress),
                "active_blocker_progress_kinds": kinds,
                "selected_release_opened": bool(selected_release_opened),
                "release_opened": bool(release_opened),
                "aggregate_blocker_progress": bool(aggregate_progress),
                "path_loop": bool(path_loop),
                "depth": int(depth),
                "after_no_legal_pushes": bool(metrics.get("no_legal_pushes", False)),
                "after_nonblocked_legal_push_count": int(
                    metrics.get("nonblocked_legal_push_count", 0)
                ),
                "score": float(score),
            }
        )
        return result

    def _active_blocker_option_key(self, row: dict[str, Any]) -> tuple:
        progress = row["best_progress"]
        obligation = row["obligation"]
        edge = row["edge"]
        return (
            int(progress.get("active_blocker_obligation_tier", 0)),
            int(progress.get("active_blocker_true_progress", False)),
            int(progress.get("row_disappeared", False)),
            int(progress.get("target_resolved", False)),
            int(progress.get("stand_resolved", False)),
            int(progress.get("selected_blocker_delta", 0)),
            int(progress.get("selected_release_action_delta", 0)),
            int(progress.get("dependency_total_delta", 0)),
            int(progress.get("target_blocker_count_delta", 0)),
            int(progress.get("unreachable_stand_count_delta", 0)),
            int(not bool(obligation.get("future", False))),
            float(progress.get("score", 0.0)),
            -int(progress.get("depth", 0)),
            *self._agenda_tier_intra_key(edge),
        )

    def _active_blocker_action_row(
        self,
        state: SokobanState,
        action: SokobanPushAction,
        *,
        depth: int,
    ) -> dict[str, Any]:
        source, target = self._moved_cells(state, action)
        return {
            "depth": int(depth),
            "physical_action": self._physical_action_dict(state, action),
            "source": list(source),
            "target": list(target),
        }

    def _active_blocker_evaluate_edge(
        self,
        root: RoomSemanticNode,
        edge: RoomSemanticEdge,
        obligation: dict[str, Any],
    ) -> dict[str, Any]:
        reference_metrics = self._dependency_bound_metrics(
            root.state,
            root.commitment_state,
            obligation,
        )
        first_action = edge.action.physical_action
        first_key = self.model.gft_canonical_key_cached(edge.child_state)
        root_key = self.model.gft_canonical_key_cached(root.state)
        first_metrics = self._dependency_bound_metrics(
            edge.child_state,
            edge.child_commitment_state,
            obligation,
        )
        first_score, first_progress = self._dependency_bound_score(
            reference_metrics,
            first_metrics,
            obligation,
            depth=1,
        )
        first_progress["dependency_bound_score"] = float(first_score)
        first_path_loop = bool(first_key == root_key)
        first_progress = self._active_blocker_progress_summary(
            first_progress,
            first_metrics,
            depth=1,
            path_loop=first_path_loop,
        )
        first_row = self._active_blocker_action_row(
            root.state,
            first_action,
            depth=1,
        )
        best = {
            "best_progress": first_progress,
            "best_metrics": first_metrics,
            "best_actions": [first_row],
            "best_depth": 1,
            "expanded_nodes": 0,
        }
        queue: list[dict[str, Any]] = []
        if (
            int(first_progress["active_blocker_obligation_tier"])
            < self.active_blocker_obligation_min_tier
            and not first_path_loop
            and not bool(first_metrics["no_legal_pushes"])
        ):
            queue.append(
                {
                    "state": edge.child_state,
                    "commitment_state": edge.child_commitment_state,
                    "actions": [first_row],
                    "seen_state_keys": {root_key, first_key},
                    "path_loop": first_path_loop,
                }
            )
        expanded = 0
        while queue and expanded < self.active_blocker_obligation_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.active_blocker_obligation_max_pushes:
                continue
            if bool(node["path_loop"]):
                continue
            if self.model.is_solved(node["state"]):
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                child_key = self.model.gft_canonical_key_cached(child)
                path_loop = bool(child_key in node["seen_state_keys"])
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                child_metrics = self._dependency_bound_metrics(
                    child,
                    child_commitment,
                    obligation,
                )
                child_depth = depth + 1
                score, progress = self._dependency_bound_score(
                    reference_metrics,
                    child_metrics,
                    obligation,
                    depth=child_depth,
                )
                progress["dependency_bound_score"] = float(score)
                progress = self._active_blocker_progress_summary(
                    progress,
                    child_metrics,
                    depth=child_depth,
                    path_loop=path_loop,
                )
                action_row = self._active_blocker_action_row(
                    node["state"],
                    action,
                    depth=child_depth,
                )
                candidate_actions = [*node["actions"], action_row]
                candidate = {
                    "edge": edge,
                    "obligation": obligation,
                    "best_progress": progress,
                    "best_metrics": child_metrics,
                    "best_actions": candidate_actions,
                    "best_depth": int(child_depth),
                    "expanded_nodes": int(expanded + 1),
                }
                current_best = {**best, "edge": edge, "obligation": obligation}
                if self._active_blocker_option_key(
                    candidate
                ) > self._active_blocker_option_key(current_best):
                    best = {
                        "best_progress": progress,
                        "best_metrics": child_metrics,
                        "best_actions": candidate_actions,
                        "best_depth": int(child_depth),
                        "expanded_nodes": int(expanded + 1),
                    }
                if (
                    child_depth < self.active_blocker_obligation_max_pushes
                    and not path_loop
                    and int(progress["active_blocker_obligation_tier"])
                    < self.active_blocker_obligation_min_tier
                    and not bool(child_metrics["no_legal_pushes"])
                ):
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate_actions,
                            "seen_state_keys": set(node["seen_state_keys"])
                            | {child_key},
                            "path_loop": bool(path_loop),
                        }
                    )
                expanded += 1
                if expanded >= self.active_blocker_obligation_max_nodes:
                    break
        return {
            "edge": edge,
            "obligation": obligation,
            "root_action": self._agenda_tier_action_summary(edge),
            "root_agenda_tier": int(self._agenda_tier_value(edge)),
            "reference_metrics": reference_metrics,
            "best_progress": best["best_progress"],
            "best_metrics": best["best_metrics"],
            "best_actions": best["best_actions"],
            "best_depth": int(best["best_depth"]),
            "expanded_nodes": int(max(expanded, int(best["expanded_nodes"]))),
            "option_score": float(best["best_progress"]["score"]),
        }

    def _active_blocker_compact_option(
        self,
        row: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        metrics = row.get("best_metrics", {})
        return {
            "root_action": row["root_action"],
            "root_agenda_tier": int(row["root_agenda_tier"]),
            "obligation": row["obligation"],
            "reference_metrics": row["reference_metrics"],
            "best_progress": row["best_progress"],
            "best_metrics": {
                key: value
                for key, value in metrics.items()
                if key not in {"selected_release_actions"}
            },
            "best_actions": row["best_actions"][
                : self.active_blocker_obligation_max_pushes
            ],
            "best_depth": int(row["best_depth"]),
            "expanded_nodes": int(row["expanded_nodes"]),
            "option_score": float(row["option_score"]),
        }

    def _active_blocker_option_record(
        self,
        root: RoomSemanticNode,
        parent_best: RoomSemanticEdge,
        selected: RoomSemanticEdge,
        *,
        reason: str,
        reference: dict[str, Any] | None,
        obligations: list[dict[str, Any]],
        evaluated: list[dict[str, Any]],
        parent_eval: dict[str, Any] | None,
        selected_eval: dict[str, Any] | None,
    ) -> dict[str, Any]:
        compact = [
            self._active_blocker_compact_option(row)
            for row in sorted(evaluated, key=self._active_blocker_option_key, reverse=True)
        ]
        compact = [row for row in compact if row is not None]
        parent_option_tier = (
            None
            if parent_eval is None
            else int(
                parent_eval["best_progress"].get(
                    "active_blocker_obligation_tier",
                    0,
                )
            )
        )
        selected_option_tier = (
            None
            if selected_eval is None
            else int(
                selected_eval["best_progress"].get(
                    "active_blocker_obligation_tier",
                    0,
                )
            )
        )
        return {
            "active_blocker_obligation_enabled": bool(
                self.active_blocker_obligation_enabled
            ),
            "active_blocker_obligation_scope": (
                "post_frontier_blocker_specific_obligation_e6_e14_v1"
            ),
            "active_blocker_obligation_mode": self.active_blocker_obligation_mode,
            "active_blocker_obligation_max_pushes": int(
                self.active_blocker_obligation_max_pushes
            ),
            "active_blocker_obligation_max_nodes": int(
                self.active_blocker_obligation_max_nodes
            ),
            "active_blocker_obligation_max_root_candidates": int(
                self.active_blocker_obligation_max_root_candidates
            ),
            "active_blocker_obligation_max_obligations": int(
                self.active_blocker_obligation_max_obligations
            ),
            "active_blocker_obligation_min_tier": int(
                self.active_blocker_obligation_min_tier
            ),
            "active_blocker_obligation_min_tier_improvement": int(
                self.active_blocker_obligation_min_tier_improvement
            ),
            "active_blocker_obligation_parent_max_tier": int(
                self.active_blocker_obligation_parent_max_tier
            ),
            "active_blocker_obligation_override": bool(selected is not parent_best),
            "active_blocker_obligation_reason": reason,
            "candidate_count": int(len(root.children)),
            "obligation_count": int(len(obligations)),
            "evaluated_option_count": int(len(evaluated)),
            "parent_option_tier": parent_option_tier,
            "selected_option_tier": selected_option_tier,
            "before_metrics": None
            if reference is None
            else {
                key: value
                for key, value in reference.items()
                if key != "signature"
            },
            "active_obligations": obligations,
            "parent_action": self._agenda_tier_action_summary(parent_best),
            "selected_action": self._agenda_tier_action_summary(selected),
            "parent_option": self._active_blocker_compact_option(parent_eval),
            "selected_option": self._active_blocker_compact_option(selected_eval),
            "top_active_blocker_options": compact[:8],
        }

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        parent_best = super().choose_root_edge(root)
        if not self.active_blocker_obligation_enabled or not root.children:
            self._active_blocker_obligation_summary = (
                self._active_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="disabled_or_empty",
                    reference=None,
                    obligations=[],
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        reference = self._release_blocker_option_reference(root)
        if not self._active_blocker_parent_is_trigger(parent_best, reference):
            self._active_blocker_obligation_summary = (
                self._active_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="parent_not_active_blocker_trigger",
                    reference=reference,
                    obligations=[],
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        obligations = self._active_blocker_obligations(
            root.state,
            root.commitment_state,
        )
        if not obligations:
            self._active_blocker_obligation_summary = (
                self._active_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="no_active_blocker_obligation",
                    reference=reference,
                    obligations=[],
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        allowed = [
            edge
            for edge in root.children.values()
            if self._active_blocker_candidate_allowed(edge)
        ]
        allowed.sort(key=self._agenda_tier_intra_key, reverse=True)
        allowed = allowed[: self.active_blocker_obligation_max_root_candidates]
        if parent_best not in allowed:
            allowed.append(parent_best)
        if not allowed:
            self._active_blocker_obligation_summary = (
                self._active_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="no_allowed_active_blocker_candidate",
                    reference=reference,
                    obligations=obligations,
                    evaluated=[],
                    parent_eval=None,
                    selected_eval=None,
                )
            )
            return parent_best

        evaluated: list[dict[str, Any]] = []
        for edge in allowed:
            for obligation in obligations:
                evaluated.append(
                    self._active_blocker_evaluate_edge(root, edge, obligation)
                )
        parent_rows = [row for row in evaluated if row["edge"] is parent_best]
        parent_eval = (
            max(parent_rows, key=self._active_blocker_option_key)
            if parent_rows
            else None
        )
        best_eval = max(evaluated, key=self._active_blocker_option_key)
        parent_option_tier = (
            0
            if parent_eval is None
            else int(
                parent_eval["best_progress"].get(
                    "active_blocker_obligation_tier",
                    0,
                )
            )
        )
        best_option_tier = int(
            best_eval["best_progress"].get(
                "active_blocker_obligation_tier",
                0,
            )
        )
        if best_option_tier < self.active_blocker_obligation_min_tier:
            self._active_blocker_obligation_summary = (
                self._active_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="no_high_tier_active_blocker_obligation_option",
                    reference=reference,
                    obligations=obligations,
                    evaluated=evaluated,
                    parent_eval=parent_eval,
                    selected_eval=parent_eval,
                )
            )
            return parent_best
        if (
            best_option_tier
            < parent_option_tier
            + self.active_blocker_obligation_min_tier_improvement
        ):
            self._active_blocker_obligation_summary = (
                self._active_blocker_option_record(
                    root,
                    parent_best,
                    parent_best,
                    reason="parent_active_blocker_obligation_already_best_tier",
                    reference=reference,
                    obligations=obligations,
                    evaluated=evaluated,
                    parent_eval=parent_eval,
                    selected_eval=parent_eval,
                )
            )
            return parent_best

        selected = best_eval["edge"]
        reason = (
            "parent_active_blocker_obligation_already_best_tier"
            if selected is parent_best
            else "higher_active_blocker_obligation_option_available"
        )
        self._active_blocker_obligation_summary = (
            self._active_blocker_option_record(
                root,
                parent_best,
                selected,
                reason=reason,
                reference=reference,
                obligations=obligations,
                evaluated=evaluated,
                parent_eval=parent_eval,
                selected_eval=best_eval,
            )
        )
        return selected

    def plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._active_blocker_obligation_summary = {}
        result = super().plan(state, commitment_state)
        result["active_blocker_obligation_option"] = dict(
            self._active_blocker_obligation_summary
        )
        return result


class RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner(
    RoomScheduleActiveBlockerObligationMacroGFTSDMCTSPlanner
):
    """A45 planner with bounded commitment to an active-blocker option.

    A44 can identify blocker-specific suffixes that make true progress, but it
    still executes only the first root edge.  A45 keeps the same SD evidence and
    adds only a conservative consumer: if A44 selected a high-tier active
    blocker option, the suffix is replayed under the exact transition model and
    committed for a bounded number of pushes.
    """

    provider_id = "room_schedule_active_blocker_commitment_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        active_blocker_commitment_enabled: bool = True,
        active_blocker_commitment_min_tier: int = 8,
        active_blocker_commitment_require_true_progress: bool = True,
        active_blocker_commitment_max_extra_pushes: int = 5,
        active_blocker_commitment_min_total_pushes: int = 2,
        active_blocker_commitment_abort_on_base_macro: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.active_blocker_commitment_enabled = bool(
            active_blocker_commitment_enabled
        )
        self.active_blocker_commitment_min_tier = int(
            active_blocker_commitment_min_tier
        )
        self.active_blocker_commitment_require_true_progress = bool(
            active_blocker_commitment_require_true_progress
        )
        self.active_blocker_commitment_max_extra_pushes = max(
            int(active_blocker_commitment_max_extra_pushes),
            0,
        )
        self.active_blocker_commitment_min_total_pushes = max(
            int(active_blocker_commitment_min_total_pushes),
            1,
        )
        self.active_blocker_commitment_abort_on_base_macro = bool(
            active_blocker_commitment_abort_on_base_macro
        )

    def _active_blocker_commitment_record(
        self,
        *,
        triggered: bool,
        reason: str,
        option_record: dict[str, Any] | None,
        selected_option: dict[str, Any] | None,
        validated_rows: list[dict[str, Any]] | None = None,
        final_progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress = {}
        actions = []
        obligation = None
        if isinstance(selected_option, dict):
            maybe_progress = selected_option.get("best_progress", {})
            if isinstance(maybe_progress, dict):
                progress = maybe_progress
            maybe_actions = selected_option.get("best_actions", [])
            if isinstance(maybe_actions, list):
                actions = maybe_actions
            maybe_obligation = selected_option.get("obligation")
            if isinstance(maybe_obligation, dict):
                obligation = maybe_obligation
        return {
            "active_blocker_commitment_enabled": bool(
                self.active_blocker_commitment_enabled
            ),
            "active_blocker_commitment_scope": (
                "bounded_active_blocker_option_commitment_e6_e14_v1"
            ),
            "active_blocker_commitment_min_tier": int(
                self.active_blocker_commitment_min_tier
            ),
            "active_blocker_commitment_require_true_progress": bool(
                self.active_blocker_commitment_require_true_progress
            ),
            "active_blocker_commitment_max_extra_pushes": int(
                self.active_blocker_commitment_max_extra_pushes
            ),
            "active_blocker_commitment_min_total_pushes": int(
                self.active_blocker_commitment_min_total_pushes
            ),
            "active_blocker_commitment_abort_on_base_macro": bool(
                self.active_blocker_commitment_abort_on_base_macro
            ),
            "active_blocker_commitment_triggered": bool(triggered),
            "active_blocker_commitment_reason": reason,
            "source_active_blocker_obligation_reason": None
            if not isinstance(option_record, dict)
            else option_record.get("active_blocker_obligation_reason"),
            "source_selected_option_tier": int(
                progress.get("active_blocker_obligation_tier", 0) or 0
            ),
            "source_selected_true_progress": bool(
                progress.get("active_blocker_true_progress", False)
            ),
            "source_obligation": obligation,
            "proposed_push_count": int(len(actions)),
            "validated_push_count": int(len(validated_rows or [])),
            "committed_extra_push_count": int(max(0, len(validated_rows or []) - 1)),
            "proposed_actions": actions,
            "validated_rows": list(validated_rows or []),
            "final_progress": dict(final_progress or {}),
        }

    @staticmethod
    def _active_blocker_commitment_first_matches(
        action: SokobanPushAction,
        row: dict[str, Any],
    ) -> bool:
        physical = row.get("physical_action", {}) if isinstance(row, dict) else {}
        if not isinstance(physical, dict):
            return False
        try:
            return bool(
                int(physical.get("flat_index", -1)) == int(action.flat_index)
                and int(physical.get("box_index", -1)) == int(action.box_index)
                and int(physical.get("direction", -1)) == int(action.direction)
            )
        except (TypeError, ValueError):
            return False

    def _active_blocker_commitment_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
        macro: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        option_record = plan.get("active_blocker_obligation_option", {})
        if not self.active_blocker_commitment_enabled:
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="disabled",
                option_record=option_record,
                selected_option=None,
            )
        if plan.get("physical_action") is None:
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="no_physical_action",
                option_record=option_record,
                selected_option=None,
            )
        if not isinstance(option_record, dict) or not option_record:
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="no_active_blocker_obligation_option",
                option_record=option_record,
                selected_option=None,
            )
        selected_option = option_record.get("selected_option", {})
        if not isinstance(selected_option, dict) or not selected_option:
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="no_selected_active_blocker_option",
                option_record=option_record,
                selected_option=None,
            )
        progress = selected_option.get("best_progress", {})
        if not isinstance(progress, dict):
            progress = {}
        if (
            int(progress.get("active_blocker_obligation_tier", 0) or 0)
            < self.active_blocker_commitment_min_tier
        ):
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="selected_active_blocker_option_below_min_tier",
                option_record=option_record,
                selected_option=selected_option,
            )
        if (
            self.active_blocker_commitment_require_true_progress
            and not bool(progress.get("active_blocker_true_progress", False))
        ):
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="selected_active_blocker_option_not_true_progress",
                option_record=option_record,
                selected_option=selected_option,
            )
        obligation = selected_option.get("obligation", {})
        if not isinstance(obligation, dict) or not obligation:
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="selected_active_blocker_option_missing_obligation",
                option_record=option_record,
                selected_option=selected_option,
            )
        proposed_actions = selected_option.get("best_actions", [])
        if not isinstance(proposed_actions, list):
            proposed_actions = []
        keep = 1 + self.active_blocker_commitment_max_extra_pushes
        proposed_actions = proposed_actions[:keep]
        if len(proposed_actions) < self.active_blocker_commitment_min_total_pushes:
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="selected_active_blocker_option_too_short",
                option_record=option_record,
                selected_option=selected_option,
            )

        base_actions = list(macro.get("physical_actions", []))
        if not base_actions:
            base_actions = [plan["physical_action"]]
        if (
            self.active_blocker_commitment_abort_on_base_macro
            and len(base_actions) != 1
        ):
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="base_macro_not_single_push",
                option_record=option_record,
                selected_option=selected_option,
            )
        first_action = plan["physical_action"]
        if not self._active_blocker_commitment_first_matches(
            first_action,
            proposed_actions[0],
        ):
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="first_action_mismatch",
                option_record=option_record,
                selected_option=selected_option,
            )

        reference_metrics = self._dependency_bound_metrics(
            state,
            commitment_state,
            obligation,
        )
        current_state = state
        current_commitment = commitment_state
        seen_state_keys = {self.model.gft_canonical_key_cached(state)}
        actions: list[SokobanPushAction] = []
        commitments: list[RoomCommitmentState] = []
        rows: list[dict[str, Any]] = []

        for index, row in enumerate(proposed_actions):
            action = self._agenda_option_commitment_match_action(current_state, row)
            if action is None:
                return macro, self._active_blocker_commitment_record(
                    triggered=False,
                    reason="committed_active_blocker_action_illegal",
                    option_record=option_record,
                    selected_option=selected_option,
                    validated_rows=rows,
                )
            child = self.model.step_grid_flow(current_state, action)
            if self.is_blocked(child):
                return macro, self._active_blocker_commitment_record(
                    triggered=False,
                    reason="committed_active_blocker_action_blocked",
                    option_record=option_record,
                    selected_option=selected_option,
                    validated_rows=rows,
                )
            if index == 0 and macro.get("after_commitment_states"):
                next_commitment = list(macro["after_commitment_states"])[0]
                next_commitment = self.model.normalize_room_commitment_state(
                    child,
                    next_commitment,
                )
            elif index == 0 and plan.get("next_commitment_state") is not None:
                next_commitment = self.model.normalize_room_commitment_state(
                    child,
                    plan["next_commitment_state"],
                )
            else:
                next_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
            child_key = self.model.gft_canonical_key_cached(child)
            path_loop = bool(child_key in seen_state_keys)
            metrics = self._dependency_bound_metrics(
                child,
                next_commitment,
                obligation,
            )
            depth = index + 1
            score, dependency_progress = self._dependency_bound_score(
                reference_metrics,
                metrics,
                obligation,
                depth=depth,
            )
            dependency_progress["dependency_bound_score"] = float(score)
            active_progress = self._active_blocker_progress_summary(
                dependency_progress,
                metrics,
                depth=depth,
                path_loop=path_loop,
            )
            source, target = self._moved_cells(current_state, action)
            rows.append(
                {
                    "macro_index": int(index),
                    "depth": int(depth),
                    "physical_action": self._physical_action_dict(
                        current_state,
                        action,
                    ),
                    "reason": "active_blocker_commitment_initial_push"
                    if index == 0
                    else "active_blocker_commitment_suffix_push",
                    "source": list(source),
                    "target": list(target),
                    "path_loop_after": bool(path_loop),
                    "active_blocker_obligation": obligation,
                    "active_blocker_reference_metrics": reference_metrics,
                    "active_blocker_metrics": {
                        key: value
                        for key, value in metrics.items()
                        if key not in {"selected_release_actions"}
                    },
                    "active_blocker_progress": active_progress,
                    "active_blocker_commitment_score": float(
                        active_progress.get("score", 0.0)
                    ),
                }
            )
            actions.append(action)
            commitments.append(next_commitment)
            current_state = child
            current_commitment = next_commitment
            seen_state_keys.add(child_key)
            if path_loop:
                return macro, self._active_blocker_commitment_record(
                    triggered=False,
                    reason="committed_active_blocker_path_loop",
                    option_record=option_record,
                    selected_option=selected_option,
                    validated_rows=rows,
                )

        final_metrics = self._dependency_bound_metrics(
            current_state,
            current_commitment,
            obligation,
        )
        final_score, final_dependency_progress = self._dependency_bound_score(
            reference_metrics,
            final_metrics,
            obligation,
            depth=len(actions),
        )
        final_dependency_progress["dependency_bound_score"] = float(final_score)
        final_progress = self._active_blocker_progress_summary(
            final_dependency_progress,
            final_metrics,
            depth=len(actions),
            path_loop=False,
        )
        if (
            int(final_progress.get("active_blocker_obligation_tier", 0))
            < self.active_blocker_commitment_min_tier
        ):
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="validated_active_blocker_option_below_min_tier",
                option_record=option_record,
                selected_option=selected_option,
                validated_rows=rows,
                final_progress=final_progress,
            )
        if (
            self.active_blocker_commitment_require_true_progress
            and not bool(final_progress.get("active_blocker_true_progress", False))
        ):
            return macro, self._active_blocker_commitment_record(
                triggered=False,
                reason="validated_active_blocker_option_not_true_progress",
                option_record=option_record,
                selected_option=selected_option,
                validated_rows=rows,
                final_progress=final_progress,
            )

        committed_macro = {
            **macro,
            "enabled": True,
            "triggered": len(actions)
            >= self.active_blocker_commitment_min_total_pushes,
            "reason": "active_blocker_commitment",
            "physical_actions": actions,
            "after_commitment_states": commitments,
            "rows": rows,
            "macro_push_count": int(len(actions)),
            "extra_push_count": int(max(0, len(actions) - 1)),
            "active_blocker_commitment_enabled": True,
            "active_blocker_commitment_triggered": True,
            "active_blocker_commitment_reason": (
                "validated_active_blocker_option_committed"
            ),
            "active_blocker_commitment_source_reason": option_record.get(
                "active_blocker_obligation_reason"
            ),
            "active_blocker_commitment_final_progress": final_progress,
            "active_blocker_commitment_obligation": obligation,
        }
        record = self._active_blocker_commitment_record(
            triggered=True,
            reason="validated_active_blocker_option_committed",
            option_record=option_record,
            selected_option=selected_option,
            validated_rows=rows,
            final_progress=final_progress,
        )
        return committed_macro, record

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        commitment_state = self.model.normalize_room_commitment_state(
            state,
            commitment_state,
        )
        plan = super().plan_with_corridor_macro(state, commitment_state)
        macro = plan.get("corridor_macro", {})
        committed_macro, record = self._active_blocker_commitment_from_plan(
            state,
            commitment_state,
            plan,
            macro,
        )
        plan["corridor_macro"] = committed_macro
        plan["active_blocker_commitment"] = record
        return plan


class RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner(
    RoomScheduleActiveBlockerCommitmentMacroGFTSDMCTSPlanner
):
    """A46 planner that tests residual agenda persistence after A45.

    A45 proves that a blocker-specific suffix can be replayed, but its positive
    commitments only resolve the stance component while the target blocker is
    still present.  A46 keeps the same inheritance line and adds one bounded
    continuation: after an A45 commitment leaves the same relation with a
    target residual, it searches for a short suffix that completes that
    residual agenda.  If no such suffix is found, the online action is left
    unchanged and the negative evidence is recorded.
    """

    provider_id = "room_schedule_residual_active_agenda_macro_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        residual_active_agenda_enabled: bool = True,
        residual_active_agenda_max_pushes: int = 8,
        residual_active_agenda_max_nodes: int = 192,
        residual_active_agenda_min_total_pushes: int = 1,
        residual_active_agenda_require_target_progress: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.residual_active_agenda_enabled = bool(residual_active_agenda_enabled)
        self.residual_active_agenda_max_pushes = max(
            int(residual_active_agenda_max_pushes),
            1,
        )
        self.residual_active_agenda_max_nodes = max(
            int(residual_active_agenda_max_nodes),
            1,
        )
        self.residual_active_agenda_min_total_pushes = max(
            int(residual_active_agenda_min_total_pushes),
            1,
        )
        self.residual_active_agenda_require_target_progress = bool(
            residual_active_agenda_require_target_progress
        )
        self._residual_active_agenda_summary: dict[str, Any] = {}

    def _residual_active_agenda_record(
        self,
        *,
        triggered: bool,
        reason: str,
        source_record: dict[str, Any] | None,
        residual_obligation: dict[str, Any] | None = None,
        selected_option: dict[str, Any] | None = None,
        evaluated_count: int = 0,
        expanded_nodes: int = 0,
    ) -> dict[str, Any]:
        rows = []
        progress = {}
        if isinstance(selected_option, dict):
            maybe_rows = selected_option.get("rows", [])
            if isinstance(maybe_rows, list):
                rows = maybe_rows
            maybe_progress = selected_option.get("final_progress", {})
            if isinstance(maybe_progress, dict):
                progress = maybe_progress
        source_progress = {}
        if isinstance(source_record, dict):
            maybe_source = source_record.get("final_progress", {})
            if isinstance(maybe_source, dict):
                source_progress = maybe_source
        return {
            "residual_active_agenda_enabled": bool(
                self.residual_active_agenda_enabled
            ),
            "residual_active_agenda_scope": (
                "post_commitment_target_residual_agenda_e6_e11_e14_v1"
            ),
            "residual_active_agenda_max_pushes": int(
                self.residual_active_agenda_max_pushes
            ),
            "residual_active_agenda_max_nodes": int(
                self.residual_active_agenda_max_nodes
            ),
            "residual_active_agenda_min_total_pushes": int(
                self.residual_active_agenda_min_total_pushes
            ),
            "residual_active_agenda_require_target_progress": bool(
                self.residual_active_agenda_require_target_progress
            ),
            "residual_active_agenda_triggered": bool(triggered),
            "residual_active_agenda_reason": reason,
            "source_active_blocker_commitment_reason": None
            if not isinstance(source_record, dict)
            else source_record.get("active_blocker_commitment_reason"),
            "source_active_blocker_commitment_triggered": bool(
                isinstance(source_record, dict)
                and source_record.get("active_blocker_commitment_triggered", False)
            ),
            "source_active_blocker_progress_kinds": list(
                source_progress.get("active_blocker_progress_kinds", [])
                if isinstance(source_progress, dict)
                else []
            ),
            "source_unresolved_target": bool(
                source_progress.get("unresolved_target", False)
                if isinstance(source_progress, dict)
                else False
            ),
            "residual_obligation": dict(residual_obligation or {}),
            "evaluated_option_count": int(evaluated_count),
            "expanded_nodes": int(expanded_nodes),
            "committed_residual_push_count": int(len(rows) if triggered else 0),
            "validated_rows": list(rows if triggered else []),
            "final_progress": dict(progress if triggered else {}),
        }

    @staticmethod
    def _residual_active_agenda_remaining_kinds(
        source_record: dict[str, Any],
    ) -> list[str]:
        rows = source_record.get("validated_rows", [])
        if isinstance(rows, list) and rows:
            metrics = rows[-1].get("active_blocker_metrics", {})
            if isinstance(metrics, dict):
                remaining = metrics.get("remaining_selected_kinds", [])
                if isinstance(remaining, list):
                    return sorted(str(kind) for kind in remaining)
        progress = source_record.get("final_progress", {})
        if (
            isinstance(progress, dict)
            and bool(progress.get("unresolved_target", False))
        ):
            return ["target_occupied_by_box"]
        return []

    def _residual_active_agenda_obligation(
        self,
        source_record: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not bool(source_record.get("active_blocker_commitment_triggered", False)):
            return None
        source_obligation = source_record.get("source_obligation", {})
        if not isinstance(source_obligation, dict) or not source_obligation:
            return None
        remaining = self._residual_active_agenda_remaining_kinds(source_record)
        if "target_occupied_by_box" not in remaining:
            return None
        original_kinds = source_obligation.get("selected_dependency_kinds", [])
        residual = dict(source_obligation)
        residual["selected_dependency_kinds"] = ["target_occupied_by_box"]
        residual["dependency_kinds"] = ["target_occupied_by_box"]
        residual["residual_parent_selected_dependency_kinds"] = list(
            original_kinds if isinstance(original_kinds, list) else []
        )
        residual["residual_active_agenda_kind"] = "target_occupied_by_box"
        residual["residual_active_agenda_source"] = "a45_partial_commitment"
        return residual

    def _state_after_macro_actions(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
    ) -> tuple[SokobanState, RoomCommitmentState]:
        current_state = state
        current_commitment = commitment_state
        actions = list(macro.get("physical_actions", []))
        commitments = list(macro.get("after_commitment_states", []))
        for index, action in enumerate(actions):
            child = self.model.step_grid_flow(current_state, action)
            if index < len(commitments):
                next_commitment = self.model.normalize_room_commitment_state(
                    child,
                    commitments[index],
                )
            else:
                next_commitment = self._advance_recovery_commitment(
                    current_state,
                    current_commitment,
                    action,
                    child,
                )
            current_state = child
            current_commitment = next_commitment
        return current_state, current_commitment

    def _residual_active_agenda_progress_summary(
        self,
        progress: dict[str, Any],
        metrics: dict[str, Any],
        *,
        depth: int,
        path_loop: bool,
    ) -> dict[str, Any]:
        result = self._active_blocker_progress_summary(
            progress,
            metrics,
            depth=depth,
            path_loop=path_loop,
        )
        target_progress = bool(
            result.get("row_disappeared", False)
            or result.get("target_resolved", False)
            or int(result.get("selected_blocker_delta", 0)) > 0
        )
        selected_release_opened = bool(
            result.get("selected_release_opened", False)
        )
        if target_progress:
            residual_tier = 9
        elif selected_release_opened:
            residual_tier = 6
        elif bool(result.get("aggregate_blocker_progress", False)):
            residual_tier = 4
        elif path_loop or bool(metrics.get("no_legal_pushes", False)):
            residual_tier = 1
        else:
            residual_tier = 2
        residual_score = (
            120.0 * float(residual_tier)
            + float(result.get("score", 0.0))
            + 45.0 * float(target_progress)
            + 16.0 * float(selected_release_opened)
            - 2.5 * float(depth)
        )
        result.update(
            {
                "residual_active_agenda_tier": int(residual_tier),
                "residual_active_agenda_target_progress": bool(target_progress),
                "residual_active_agenda_done": bool(target_progress),
                "residual_active_agenda_score": float(residual_score),
            }
        )
        return result

    @staticmethod
    def _residual_active_agenda_option_key(row: dict[str, Any]) -> tuple:
        progress = row["final_progress"]
        return (
            int(progress.get("residual_active_agenda_done", False)),
            int(progress.get("target_resolved", False)),
            int(progress.get("row_disappeared", False)),
            int(progress.get("selected_blocker_delta", 0)),
            int(progress.get("selected_release_opened", False)),
            int(progress.get("aggregate_blocker_progress", False)),
            float(progress.get("residual_active_agenda_score", 0.0)),
            -int(progress.get("depth", 0)),
        )

    def _residual_active_agenda_search(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        residual_obligation: dict[str, Any],
    ) -> dict[str, Any]:
        reference_metrics = self._dependency_bound_metrics(
            state,
            commitment_state,
            residual_obligation,
        )
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "actions": [],
                "commitments": [],
                "rows": [],
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        best: dict[str, Any] | None = None
        evaluated_count = 0
        expanded = 0
        while queue and expanded < self.residual_active_agenda_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.residual_active_agenda_max_pushes:
                continue
            if self.model.is_solved(node["state"]):
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                child_key = self.model.gft_canonical_key_cached(child)
                path_loop = bool(child_key in node["seen"])
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._dependency_bound_metrics(
                    child,
                    child_commitment,
                    residual_obligation,
                )
                child_depth = depth + 1
                score, dependency_progress = self._dependency_bound_score(
                    reference_metrics,
                    metrics,
                    residual_obligation,
                    depth=child_depth,
                )
                dependency_progress["dependency_bound_score"] = float(score)
                residual_progress = self._residual_active_agenda_progress_summary(
                    dependency_progress,
                    metrics,
                    depth=child_depth,
                    path_loop=path_loop,
                )
                source, target = self._moved_cells(node["state"], action)
                row = {
                    "macro_index": int(child_depth - 1),
                    "depth": int(child_depth),
                    "physical_action": self._physical_action_dict(
                        node["state"],
                        action,
                    ),
                    "reason": "residual_active_agenda_suffix_push",
                    "source": list(source),
                    "target": list(target),
                    "path_loop_after": bool(path_loop),
                    "residual_active_agenda_obligation": residual_obligation,
                    "residual_active_agenda_reference_metrics": reference_metrics,
                    "residual_active_agenda_metrics": {
                        key: value
                        for key, value in metrics.items()
                        if key not in {"selected_release_actions"}
                    },
                    "residual_active_agenda_progress": residual_progress,
                    "residual_active_agenda_score": float(
                        residual_progress.get("residual_active_agenda_score", 0.0)
                    ),
                }
                candidate_actions = [*node["actions"], action]
                candidate_commitments = [*node["commitments"], child_commitment]
                candidate_rows = [*node["rows"], row]
                candidate = {
                    "actions": candidate_actions,
                    "commitments": candidate_commitments,
                    "rows": candidate_rows,
                    "final_progress": residual_progress,
                    "final_metrics": metrics,
                }
                evaluated_count += 1
                if best is None or self._residual_active_agenda_option_key(
                    candidate
                ) > self._residual_active_agenda_option_key(best):
                    best = candidate
                if (
                    child_depth < self.residual_active_agenda_max_pushes
                    and not path_loop
                    and not bool(metrics.get("no_legal_pushes", False))
                    and not bool(
                        residual_progress.get("residual_active_agenda_done", False)
                    )
                ):
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate_actions,
                            "commitments": candidate_commitments,
                            "rows": candidate_rows,
                            "seen": set(node["seen"]) | {child_key},
                        }
                    )
                expanded += 1
                if expanded >= self.residual_active_agenda_max_nodes:
                    break
        return {
            "selected_option": best,
            "evaluated_count": int(evaluated_count),
            "expanded_nodes": int(expanded),
            "reference_metrics": reference_metrics,
        }

    def _residual_active_agenda_extend_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
        source_record: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.residual_active_agenda_enabled:
            return macro, self._residual_active_agenda_record(
                triggered=False,
                reason="disabled",
                source_record=source_record,
            )
        if not bool(source_record.get("active_blocker_commitment_triggered", False)):
            return macro, self._residual_active_agenda_record(
                triggered=False,
                reason="source_commitment_not_triggered",
                source_record=source_record,
            )
        residual_obligation = self._residual_active_agenda_obligation(source_record)
        if residual_obligation is None:
            return macro, self._residual_active_agenda_record(
                triggered=False,
                reason="no_target_residual_agenda",
                source_record=source_record,
            )
        residual_state, residual_commitment = self._state_after_macro_actions(
            state,
            commitment_state,
            macro,
        )
        search = self._residual_active_agenda_search(
            residual_state,
            residual_commitment,
            residual_obligation,
        )
        selected_option = search.get("selected_option")
        if not isinstance(selected_option, dict) or selected_option is None:
            return macro, self._residual_active_agenda_record(
                triggered=False,
                reason="no_residual_active_agenda_option",
                source_record=source_record,
                residual_obligation=residual_obligation,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        final_progress = selected_option.get("final_progress", {})
        if (
            self.residual_active_agenda_require_target_progress
            and not bool(
                final_progress.get("residual_active_agenda_target_progress", False)
            )
        ):
            return macro, self._residual_active_agenda_record(
                triggered=False,
                reason="no_target_progress_residual_option",
                source_record=source_record,
                residual_obligation=residual_obligation,
                selected_option=selected_option,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        if (
            len(selected_option.get("actions", []))
            < self.residual_active_agenda_min_total_pushes
        ):
            return macro, self._residual_active_agenda_record(
                triggered=False,
                reason="residual_active_agenda_option_too_short",
                source_record=source_record,
                residual_obligation=residual_obligation,
                selected_option=selected_option,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )

        actions = list(macro.get("physical_actions", [])) + list(
            selected_option["actions"]
        )
        commitments = list(macro.get("after_commitment_states", [])) + list(
            selected_option["commitments"]
        )
        rows = list(macro.get("rows", [])) + list(selected_option["rows"])
        extended_macro = {
            **macro,
            "enabled": True,
            "triggered": True,
            "reason": "active_blocker_commitment_with_residual_active_agenda",
            "physical_actions": actions,
            "after_commitment_states": commitments,
            "rows": rows,
            "macro_push_count": int(len(actions)),
            "extra_push_count": int(max(0, len(actions) - 1)),
            "residual_active_agenda_enabled": True,
            "residual_active_agenda_triggered": True,
            "residual_active_agenda_reason": (
                "validated_target_residual_agenda_committed"
            ),
            "residual_active_agenda_obligation": residual_obligation,
            "residual_active_agenda_final_progress": final_progress,
            "residual_active_agenda_push_count": int(
                len(selected_option["actions"])
            ),
        }
        record = self._residual_active_agenda_record(
            triggered=True,
            reason="validated_target_residual_agenda_committed",
            source_record=source_record,
            residual_obligation=residual_obligation,
            selected_option=selected_option,
            evaluated_count=int(search.get("evaluated_count", 0)),
            expanded_nodes=int(search.get("expanded_nodes", 0)),
        )
        return extended_macro, record

    def _active_blocker_commitment_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
        macro: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        committed_macro, record = super()._active_blocker_commitment_from_plan(
            state,
            commitment_state,
            plan,
            macro,
        )
        extended_macro, residual_record = self._residual_active_agenda_extend_macro(
            state,
            commitment_state,
            committed_macro,
            record,
        )
        self._residual_active_agenda_summary = residual_record
        return extended_macro, record

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._residual_active_agenda_summary = {}
        plan = super().plan_with_corridor_macro(state, commitment_state)
        plan["residual_active_agenda"] = dict(
            self._residual_active_agenda_summary
        )
        return plan


class RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner(
    RoomScheduleResidualActiveAgendaMacroGFTSDMCTSPlanner
):
    """A47 planner with an explicit prerequisite for target residuals.

    A46 can reconstruct the remaining target-only agenda after A45, but the
    observed target blocker may itself have no legal push.  A47 keeps the same
    inheritance line and derives the next prerequisite from the current state:
    if pushing the target blocker requires a stand cell currently occupied by
    another box, the planner records a ClearTargetPushStand obligation and
    searches for a short suffix that makes that prerequisite true.  No
    reference action or teacher policy is used online.
    """

    provider_id = (
        "room_schedule_residual_prerequisite_agenda_macro_gft_sdmcts_v1"
    )

    def __init__(
        self,
        *args,
        target_prerequisite_agenda_enabled: bool = True,
        target_prerequisite_agenda_max_pushes: int = 8,
        target_prerequisite_agenda_max_nodes: int = 256,
        target_prerequisite_agenda_min_total_pushes: int = 1,
        target_prerequisite_agenda_require_progress: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_prerequisite_agenda_enabled = bool(
            target_prerequisite_agenda_enabled
        )
        self.target_prerequisite_agenda_max_pushes = max(
            int(target_prerequisite_agenda_max_pushes),
            1,
        )
        self.target_prerequisite_agenda_max_nodes = max(
            int(target_prerequisite_agenda_max_nodes),
            1,
        )
        self.target_prerequisite_agenda_min_total_pushes = max(
            int(target_prerequisite_agenda_min_total_pushes),
            1,
        )
        self.target_prerequisite_agenda_require_progress = bool(
            target_prerequisite_agenda_require_progress
        )
        self._target_prerequisite_agenda_summary: dict[str, Any] = {}

    @staticmethod
    def _target_prerequisite_cell(value: Any) -> Cell | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return None

    @staticmethod
    def _target_prerequisite_minus(cell: Cell, direction: Cell) -> Cell:
        return cell[0] - direction[0], cell[1] - direction[1]

    def _target_prerequisite_push_candidates(
        self,
        state: SokobanState,
        residual_obligation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        boxes = set(state.boxes)
        reachable = set(self.model.reachable_cells(state))
        blocker_cell = self._target_prerequisite_cell(
            residual_obligation.get("target_cell")
        )
        parent_direction = residual_obligation.get("candidate_direction")
        rows: list[dict[str, Any]] = []
        blocker_box_index = (
            state.boxes.index(blocker_cell)
            if blocker_cell is not None and blocker_cell in boxes
            else None
        )
        if blocker_cell is None:
            return rows
        for direction_index, direction in enumerate(DIRECTIONS):
            stand_cell = self._target_prerequisite_minus(blocker_cell, direction)
            target_cell = add_cells(blocker_cell, direction)
            stand_is_floor = self.model.level.is_floor(stand_cell)
            target_is_floor = self.model.level.is_floor(target_cell)
            stand_occupied = stand_cell in boxes
            target_occupied = target_cell in boxes
            stand_reachable = stand_cell in reachable
            child_blocked = False
            legal_push = bool(
                blocker_box_index is not None
                and stand_reachable
                and target_is_floor
                and not target_occupied
            )
            if legal_push:
                action = SokobanPushAction(
                    int(blocker_box_index),
                    int(direction_index),
                )
                child_blocked = bool(
                    self.is_blocked(self.model.step_grid_flow(state, action))
                )
            blockers: list[str] = []
            if not stand_is_floor:
                blockers.append("stand_not_floor")
            elif stand_occupied:
                blockers.append("stand_occupied_by_box")
            elif not stand_reachable:
                blockers.append("stand_unreachable_by_player")
            if not target_is_floor:
                blockers.append("push_target_not_floor")
            elif target_occupied:
                blockers.append("push_target_occupied_by_box")
            if child_blocked:
                blockers.append("child_blocked")
            rows.append(
                {
                    "blocker_cell": list(blocker_cell),
                    "blocker_box_index": None
                    if blocker_box_index is None
                    else int(blocker_box_index),
                    "direction": int(direction_index),
                    "direction_name": ACTION_NAMES[direction_index],
                    "same_direction_as_parent": bool(
                        ACTION_NAMES[direction_index] == parent_direction
                    ),
                    "stand_cell": list(stand_cell),
                    "target_cell": list(target_cell),
                    "stand_is_floor": bool(stand_is_floor),
                    "target_is_floor": bool(target_is_floor),
                    "stand_occupied_by_box": bool(stand_occupied),
                    "target_occupied_by_box": bool(target_occupied),
                    "stand_reachable_by_player": bool(stand_reachable),
                    "push_legal_now": bool(legal_push and not child_blocked),
                    "child_blocked": bool(child_blocked),
                    "blocker_codes": blockers,
                }
            )
        return rows

    def _target_prerequisite_obligation(
        self,
        state: SokobanState,
        residual_obligation: dict[str, Any],
    ) -> dict[str, Any] | None:
        boxes = set(state.boxes)
        blocker_cell = self._target_prerequisite_cell(
            residual_obligation.get("target_cell")
        )
        if blocker_cell is None or blocker_cell not in boxes:
            return None
        candidates = self._target_prerequisite_push_candidates(
            state,
            residual_obligation,
        )
        legal_count = int(sum(bool(row["push_legal_now"]) for row in candidates))
        stand_blocked = [
            row
            for row in candidates
            if bool(row.get("stand_occupied_by_box"))
            and bool(row.get("target_is_floor"))
            and not bool(row.get("target_occupied_by_box"))
            and not bool(row.get("child_blocked"))
        ]
        if legal_count > 0 or not stand_blocked:
            return None

        def priority(row: dict[str, Any]) -> tuple:
            return (
                int(row.get("same_direction_as_parent", False)),
                int(row.get("target_is_floor", False)),
                int(not bool(row.get("target_occupied_by_box", False))),
                -int(row.get("direction", 0)),
            )

        selected = max(stand_blocked, key=priority)
        required_stand = self._target_prerequisite_cell(selected.get("stand_cell"))
        assigned_box = self._target_prerequisite_cell(
            residual_obligation.get("assigned_box")
        )
        stand_blocker_box_index = (
            state.boxes.index(required_stand)
            if required_stand is not None and required_stand in boxes
            else None
        )
        return {
            "target_prerequisite_agenda_kind": "clear_target_push_stand",
            "target_prerequisite_agenda_source": "a46_target_residual",
            "parent_residual_obligation": dict(residual_obligation),
            "blocker_cell": list(blocker_cell),
            "blocker_box_index": int(state.boxes.index(blocker_cell)),
            "blocker_push_direction": selected["direction_name"],
            "blocker_push_direction_index": int(selected["direction"]),
            "required_stand_cell": selected["stand_cell"],
            "required_stand_blocker_cell": selected["stand_cell"],
            "required_stand_blocker_box_index": None
            if stand_blocker_box_index is None
            else int(stand_blocker_box_index),
            "required_stand_blocker_is_parent_assigned_box": bool(
                required_stand is not None and assigned_box == required_stand
            ),
            "blocker_push_target_cell": selected["target_cell"],
            "parent_candidate_direction": residual_obligation.get(
                "candidate_direction"
            ),
            "target_blocker_legal_push_count": int(legal_count),
            "target_blocker_push_candidates": candidates,
            "completion_predicates": [
                "target_blocker_moved",
                "target_blocker_push_legal",
                "target_push_stand_empty_reachable",
            ],
        }

    def _target_prerequisite_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        obligation: dict[str, Any],
    ) -> dict[str, Any]:
        boxes = set(state.boxes)
        reachable = set(self.model.reachable_cells(state))
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        totals = schedule["totals"]
        blocker_cell = self._target_prerequisite_cell(obligation.get("blocker_cell"))
        stand_cell = self._target_prerequisite_cell(
            obligation.get("required_stand_cell")
        )
        push_target = self._target_prerequisite_cell(
            obligation.get("blocker_push_target_cell")
        )
        direction_index = int(obligation.get("blocker_push_direction_index", -1))
        target_blocker_present = bool(blocker_cell is not None and blocker_cell in boxes)
        stand_occupied = bool(stand_cell is not None and stand_cell in boxes)
        stand_reachable = bool(stand_cell is not None and stand_cell in reachable)
        push_target_floor = bool(
            push_target is not None and self.model.level.is_floor(push_target)
        )
        push_target_occupied = bool(push_target is not None and push_target in boxes)
        target_push_legal = False
        target_push_child_blocked = False
        if (
            target_blocker_present
            and stand_reachable
            and push_target_floor
            and not push_target_occupied
            and direction_index >= 0
        ):
            action = SokobanPushAction(
                state.boxes.index(blocker_cell),  # type: ignore[arg-type]
                direction_index,
            )
            target_push_child_blocked = bool(
                self.is_blocked(self.model.step_grid_flow(state, action))
            )
            target_push_legal = not target_push_child_blocked
        return {
            "target_blocker_present": bool(target_blocker_present),
            "required_stand_occupied_by_box": bool(stand_occupied),
            "required_stand_reachable_by_player": bool(stand_reachable),
            "blocker_push_target_is_floor": bool(push_target_floor),
            "blocker_push_target_occupied_by_box": bool(push_target_occupied),
            "target_blocker_push_legal": bool(target_push_legal),
            "target_blocker_push_child_blocked": bool(target_push_child_blocked),
            "dependency_total": int(totals["dependency_count"])
            + int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "nonblocked_legal_push_count": int(
                self._nonblocked_legal_push_count(state)
            ),
            "no_legal_pushes": bool(self._nonblocked_legal_push_count(state) <= 0),
        }

    def _target_prerequisite_progress_summary(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        depth: int,
        path_loop: bool,
    ) -> dict[str, Any]:
        target_blocker_moved = bool(
            before.get("target_blocker_present", False)
            and not after.get("target_blocker_present", False)
        )
        stand_emptied = bool(
            before.get("required_stand_occupied_by_box", False)
            and not after.get("required_stand_occupied_by_box", False)
        )
        stand_empty_reachable = bool(
            not after.get("required_stand_occupied_by_box", True)
            and after.get("required_stand_reachable_by_player", False)
        )
        target_push_legal = bool(after.get("target_blocker_push_legal", False))
        dependency_delta = int(before.get("dependency_total", 0)) - int(
            after.get("dependency_total", 0)
        )
        target_blocker_delta = int(before.get("target_blocker_count", 0)) - int(
            after.get("target_blocker_count", 0)
        )
        legal_delta = int(after.get("nonblocked_legal_push_count", 0)) - int(
            before.get("nonblocked_legal_push_count", 0)
        )
        prerequisite_done = bool(target_blocker_moved or target_push_legal)
        true_progress = bool(
            prerequisite_done
            or (stand_emptied and stand_empty_reachable)
            or target_blocker_delta > 0
        )
        if prerequisite_done:
            tier = 9
        elif stand_emptied and stand_empty_reachable:
            tier = 7
        elif stand_emptied or target_blocker_delta > 0:
            tier = 5
        elif path_loop or bool(after.get("no_legal_pushes", False)):
            tier = 1
        else:
            tier = 2
        score = (
            120.0 * float(tier)
            + 45.0 * float(prerequisite_done)
            + 24.0 * float(stand_emptied and stand_empty_reachable)
            + 8.0 * float(max(0, dependency_delta))
            + 8.0 * float(max(0, target_blocker_delta))
            + 1.5 * float(max(0, legal_delta))
            - 24.0 * float(path_loop)
            - 2.0 * float(depth)
        )
        return {
            "target_prerequisite_agenda_tier": int(tier),
            "target_prerequisite_true_progress": bool(true_progress),
            "target_prerequisite_done": bool(prerequisite_done),
            "target_blocker_moved": bool(target_blocker_moved),
            "target_blocker_push_legal": bool(target_push_legal),
            "required_stand_emptied": bool(stand_emptied),
            "required_stand_empty_reachable": bool(stand_empty_reachable),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_blocker_delta),
            "legal_push_count_delta": int(legal_delta),
            "path_loop": bool(path_loop),
            "depth": int(depth),
            "score": float(score),
        }

    @staticmethod
    def _target_prerequisite_option_key(row: dict[str, Any]) -> tuple:
        progress = row["final_progress"]
        return (
            int(progress.get("target_prerequisite_done", False)),
            int(progress.get("target_blocker_push_legal", False)),
            int(progress.get("target_blocker_moved", False)),
            int(progress.get("required_stand_empty_reachable", False)),
            int(progress.get("required_stand_emptied", False)),
            int(progress.get("target_blocker_count_delta", 0)),
            int(progress.get("dependency_total_delta", 0)),
            float(progress.get("score", 0.0)),
            -int(progress.get("depth", 0)),
        )

    def _target_prerequisite_search(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        obligation: dict[str, Any],
    ) -> dict[str, Any]:
        reference_metrics = self._target_prerequisite_metrics(
            state,
            commitment_state,
            obligation,
        )
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "actions": [],
                "commitments": [],
                "rows": [],
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        best: dict[str, Any] | None = None
        evaluated_count = 0
        expanded = 0
        while queue and expanded < self.target_prerequisite_agenda_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.target_prerequisite_agenda_max_pushes:
                continue
            if self.model.is_solved(node["state"]):
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                child_key = self.model.gft_canonical_key_cached(child)
                path_loop = bool(child_key in node["seen"])
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._target_prerequisite_metrics(
                    child,
                    child_commitment,
                    obligation,
                )
                child_depth = depth + 1
                progress = self._target_prerequisite_progress_summary(
                    reference_metrics,
                    metrics,
                    depth=child_depth,
                    path_loop=path_loop,
                )
                source, target = self._moved_cells(node["state"], action)
                row = {
                    "macro_index": int(child_depth - 1),
                    "depth": int(child_depth),
                    "physical_action": self._physical_action_dict(
                        node["state"],
                        action,
                    ),
                    "reason": "target_prerequisite_agenda_suffix_push",
                    "source": list(source),
                    "target": list(target),
                    "path_loop_after": bool(path_loop),
                    "target_prerequisite_agenda_obligation": obligation,
                    "target_prerequisite_reference_metrics": reference_metrics,
                    "target_prerequisite_metrics": metrics,
                    "target_prerequisite_progress": progress,
                    "target_prerequisite_score": float(progress["score"]),
                }
                candidate_actions = [*node["actions"], action]
                candidate_commitments = [*node["commitments"], child_commitment]
                candidate_rows = [*node["rows"], row]
                candidate = {
                    "actions": candidate_actions,
                    "commitments": candidate_commitments,
                    "rows": candidate_rows,
                    "final_progress": progress,
                    "final_metrics": metrics,
                }
                evaluated_count += 1
                if best is None or self._target_prerequisite_option_key(
                    candidate
                ) > self._target_prerequisite_option_key(best):
                    best = candidate
                if (
                    child_depth < self.target_prerequisite_agenda_max_pushes
                    and not path_loop
                    and not bool(metrics.get("no_legal_pushes", False))
                    and not bool(progress.get("target_prerequisite_done", False))
                ):
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate_actions,
                            "commitments": candidate_commitments,
                            "rows": candidate_rows,
                            "seen": set(node["seen"]) | {child_key},
                        }
                    )
                expanded += 1
                if expanded >= self.target_prerequisite_agenda_max_nodes:
                    break
        return {
            "selected_option": best,
            "evaluated_count": int(evaluated_count),
            "expanded_nodes": int(expanded),
            "reference_metrics": reference_metrics,
        }

    def _target_prerequisite_record(
        self,
        *,
        triggered: bool,
        reason: str,
        residual_record: dict[str, Any] | None,
        obligation: dict[str, Any] | None = None,
        selected_option: dict[str, Any] | None = None,
        evaluated_count: int = 0,
        expanded_nodes: int = 0,
    ) -> dict[str, Any]:
        rows = []
        progress = {}
        if isinstance(selected_option, dict):
            maybe_rows = selected_option.get("rows", [])
            if isinstance(maybe_rows, list):
                rows = maybe_rows
            maybe_progress = selected_option.get("final_progress", {})
            if isinstance(maybe_progress, dict):
                progress = maybe_progress
        return {
            "target_prerequisite_agenda_enabled": bool(
                self.target_prerequisite_agenda_enabled
            ),
            "target_prerequisite_agenda_scope": (
                "target_residual_clear_push_stand_e3_e6_e14_v1"
            ),
            "target_prerequisite_agenda_max_pushes": int(
                self.target_prerequisite_agenda_max_pushes
            ),
            "target_prerequisite_agenda_max_nodes": int(
                self.target_prerequisite_agenda_max_nodes
            ),
            "target_prerequisite_agenda_min_total_pushes": int(
                self.target_prerequisite_agenda_min_total_pushes
            ),
            "target_prerequisite_agenda_require_progress": bool(
                self.target_prerequisite_agenda_require_progress
            ),
            "target_prerequisite_agenda_triggered": bool(triggered),
            "target_prerequisite_agenda_reason": reason,
            "source_residual_active_agenda_reason": None
            if not isinstance(residual_record, dict)
            else residual_record.get("residual_active_agenda_reason"),
            "source_residual_active_agenda_triggered": bool(
                isinstance(residual_record, dict)
                and residual_record.get("residual_active_agenda_triggered", False)
            ),
            "target_prerequisite_obligation": dict(obligation or {}),
            "evaluated_option_count": int(evaluated_count),
            "expanded_nodes": int(expanded_nodes),
            "committed_prerequisite_push_count": int(len(rows) if triggered else 0),
            "validated_rows": list(rows if triggered else []),
            "final_progress": dict(progress if triggered else {}),
            "best_option_progress": dict(progress),
        }

    def _target_prerequisite_extend_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
        residual_record: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.target_prerequisite_agenda_enabled:
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="disabled",
                residual_record=residual_record,
            )
        if (
            residual_record.get("residual_active_agenda_reason")
            != "no_target_progress_residual_option"
        ):
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="source_not_target_residual_failure",
                residual_record=residual_record,
            )
        residual_obligation = residual_record.get("residual_obligation", {})
        if not isinstance(residual_obligation, dict) or not residual_obligation:
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="missing_residual_obligation",
                residual_record=residual_record,
            )
        residual_state, residual_commitment = self._state_after_macro_actions(
            state,
            commitment_state,
            macro,
        )
        obligation = self._target_prerequisite_obligation(
            residual_state,
            residual_obligation,
        )
        if obligation is None:
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="no_target_push_stand_prerequisite",
                residual_record=residual_record,
            )
        search = self._target_prerequisite_search(
            residual_state,
            residual_commitment,
            obligation,
        )
        selected_option = search.get("selected_option")
        if not isinstance(selected_option, dict) or selected_option is None:
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="no_target_prerequisite_option",
                residual_record=residual_record,
                obligation=obligation,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        final_progress = selected_option.get("final_progress", {})
        if (
            self.target_prerequisite_agenda_require_progress
            and not bool(
                final_progress.get("target_prerequisite_true_progress", False)
            )
        ):
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="no_target_prerequisite_progress_option",
                residual_record=residual_record,
                obligation=obligation,
                selected_option=selected_option,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        if (
            len(selected_option.get("actions", []))
            < self.target_prerequisite_agenda_min_total_pushes
        ):
            return macro, self._target_prerequisite_record(
                triggered=False,
                reason="target_prerequisite_option_too_short",
                residual_record=residual_record,
                obligation=obligation,
                selected_option=selected_option,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        actions = list(macro.get("physical_actions", [])) + list(
            selected_option["actions"]
        )
        commitments = list(macro.get("after_commitment_states", [])) + list(
            selected_option["commitments"]
        )
        rows = list(macro.get("rows", [])) + list(selected_option["rows"])
        extended_macro = {
            **macro,
            "enabled": True,
            "triggered": True,
            "reason": "active_blocker_commitment_with_target_prerequisite_agenda",
            "physical_actions": actions,
            "after_commitment_states": commitments,
            "rows": rows,
            "macro_push_count": int(len(actions)),
            "extra_push_count": int(max(0, len(actions) - 1)),
            "target_prerequisite_agenda_enabled": True,
            "target_prerequisite_agenda_triggered": True,
            "target_prerequisite_agenda_reason": (
                "validated_target_prerequisite_agenda_committed"
            ),
            "target_prerequisite_agenda_obligation": obligation,
            "target_prerequisite_agenda_final_progress": final_progress,
            "target_prerequisite_agenda_push_count": int(
                len(selected_option["actions"])
            ),
        }
        record = self._target_prerequisite_record(
            triggered=True,
            reason="validated_target_prerequisite_agenda_committed",
            residual_record=residual_record,
            obligation=obligation,
            selected_option=selected_option,
            evaluated_count=int(search.get("evaluated_count", 0)),
            expanded_nodes=int(search.get("expanded_nodes", 0)),
        )
        return extended_macro, record

    def _residual_active_agenda_extend_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
        source_record: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        residual_macro, residual_record = super()._residual_active_agenda_extend_macro(
            state,
            commitment_state,
            macro,
            source_record,
        )
        prerequisite_macro, prerequisite_record = (
            self._target_prerequisite_extend_macro(
                state,
                commitment_state,
                residual_macro,
                residual_record,
            )
        )
        self._target_prerequisite_agenda_summary = prerequisite_record
        return prerequisite_macro, residual_record

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._target_prerequisite_agenda_summary = {}
        plan = super().plan_with_corridor_macro(state, commitment_state)
        plan["target_prerequisite_agenda"] = dict(
            self._target_prerequisite_agenda_summary
        )
        return plan


class RoomSchedulePrerequisiteAssignmentConflictMacroGFTSDMCTSPlanner(
    RoomScheduleResidualPrerequisiteAgendaMacroGFTSDMCTSPlanner
):
    """A48 planner that exposes assignment/prerequisite conflicts.

    A47 shows that the target blocker's required stand can be occupied by the
    same box assigned to the parent slot.  A48 keeps that as a relation-level
    agenda item instead of a coordinate rule: the conflict is between an
    assigned box, its slot, and the prerequisite needed to release a target
    blocker.
    """

    provider_id = (
        "room_schedule_prerequisite_assignment_conflict_macro_gft_sdmcts_v1"
    )

    def __init__(
        self,
        *args,
        assignment_conflict_agenda_enabled: bool = True,
        assignment_conflict_agenda_max_pushes: int = 8,
        assignment_conflict_agenda_max_nodes: int = 320,
        assignment_conflict_agenda_min_total_pushes: int = 1,
        assignment_conflict_agenda_require_done: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.assignment_conflict_agenda_enabled = bool(
            assignment_conflict_agenda_enabled
        )
        self.assignment_conflict_agenda_max_pushes = max(
            int(assignment_conflict_agenda_max_pushes),
            1,
        )
        self.assignment_conflict_agenda_max_nodes = max(
            int(assignment_conflict_agenda_max_nodes),
            1,
        )
        self.assignment_conflict_agenda_min_total_pushes = max(
            int(assignment_conflict_agenda_min_total_pushes),
            1,
        )
        self.assignment_conflict_agenda_require_done = bool(
            assignment_conflict_agenda_require_done
        )
        self._assignment_conflict_agenda_summary: dict[str, Any] = {}

    def _assignment_conflict_pull_distance(
        self,
        cell: Cell | None,
        slot_cell: Cell | None,
    ) -> float:
        if cell is None or slot_cell is None:
            return math.inf
        distance = self.model._slot_pull_distance(cell, slot_cell)
        return float(distance)

    def _assignment_conflict_push_candidates(
        self,
        state: SokobanState,
        *,
        assigned_box_cell: Cell | None,
        assigned_slot_cell: Cell | None,
    ) -> list[dict[str, Any]]:
        if assigned_box_cell is None:
            return []
        boxes = set(state.boxes)
        reachable = set(self.model.reachable_cells(state))
        box_index = (
            state.boxes.index(assigned_box_cell)
            if assigned_box_cell in boxes
            else None
        )
        before_distance = self._assignment_conflict_pull_distance(
            assigned_box_cell,
            assigned_slot_cell,
        )
        rows: list[dict[str, Any]] = []
        for direction_index, direction in enumerate(DIRECTIONS):
            stand_cell = self._target_prerequisite_minus(
                assigned_box_cell,
                direction,
            )
            target_cell = add_cells(assigned_box_cell, direction)
            stand_is_floor = self.model.level.is_floor(stand_cell)
            target_is_floor = self.model.level.is_floor(target_cell)
            stand_occupied = stand_cell in boxes
            target_occupied = target_cell in boxes
            stand_reachable = stand_cell in reachable
            child_blocked = False
            after_distance = (
                self._assignment_conflict_pull_distance(
                    target_cell,
                    assigned_slot_cell,
                )
                if target_is_floor
                else math.inf
            )
            pull_delta = (
                float(before_distance - after_distance)
                if math.isfinite(before_distance) and math.isfinite(after_distance)
                else 0.0
            )
            legal_push = bool(
                box_index is not None
                and stand_reachable
                and target_is_floor
                and not target_occupied
            )
            if legal_push:
                action = SokobanPushAction(int(box_index), int(direction_index))
                child_blocked = bool(
                    self.is_blocked(self.model.step_grid_flow(state, action))
                )
            blockers: list[str] = []
            if not stand_is_floor:
                blockers.append("stand_not_floor")
            elif stand_occupied:
                blockers.append("stand_occupied_by_box")
            elif not stand_reachable:
                blockers.append("stand_unreachable_by_player")
            if not target_is_floor:
                blockers.append("push_target_not_floor")
            elif target_occupied:
                blockers.append("push_target_occupied_by_box")
            if child_blocked:
                blockers.append("child_blocked")
            rows.append(
                {
                    "box_index": None if box_index is None else int(box_index),
                    "direction": int(direction_index),
                    "direction_name": ACTION_NAMES[direction_index],
                    "stand_cell": list(stand_cell),
                    "target_cell": list(target_cell),
                    "stand_is_floor": bool(stand_is_floor),
                    "target_is_floor": bool(target_is_floor),
                    "stand_occupied_by_box": bool(stand_occupied),
                    "target_occupied_by_box": bool(target_occupied),
                    "stand_reachable_by_player": bool(stand_reachable),
                    "push_legal_now": bool(legal_push and not child_blocked),
                    "child_blocked": bool(child_blocked),
                    "before_pull_distance": float(before_distance),
                    "after_pull_distance": float(after_distance),
                    "pull_distance_delta": float(pull_delta),
                    "moves_toward_assigned_slot": bool(pull_delta > 0.0),
                    "blocker_codes": blockers,
                }
            )
        return rows

    def _assignment_conflict_obligation(
        self,
        state: SokobanState,
        prerequisite_obligation: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not bool(
            prerequisite_obligation.get(
                "required_stand_blocker_is_parent_assigned_box",
                False,
            )
        ):
            return None
        parent = prerequisite_obligation.get("parent_residual_obligation", {})
        if not isinstance(parent, dict):
            return None
        assigned_box_cell = self._target_prerequisite_cell(
            parent.get("assigned_box")
        )
        assigned_slot_cell = self._target_prerequisite_cell(parent.get("slot_cell"))
        required_stand_cell = self._target_prerequisite_cell(
            prerequisite_obligation.get("required_stand_cell")
        )
        target_blocker_cell = self._target_prerequisite_cell(
            prerequisite_obligation.get("blocker_cell")
        )
        if (
            assigned_box_cell is None
            or assigned_slot_cell is None
            or required_stand_cell is None
            or assigned_box_cell != required_stand_cell
            or assigned_box_cell not in set(state.boxes)
        ):
            return None
        push_candidates = self._assignment_conflict_push_candidates(
            state,
            assigned_box_cell=assigned_box_cell,
            assigned_slot_cell=assigned_slot_cell,
        )
        legal_push_count = int(
            sum(bool(row.get("push_legal_now", False)) for row in push_candidates)
        )
        toward_slot_count = int(
            sum(
                bool(row.get("push_legal_now", False))
                and bool(row.get("moves_toward_assigned_slot", False))
                for row in push_candidates
            )
        )
        return {
            "assignment_conflict_agenda_kind": (
                "resolve_assigned_box_prerequisite_conflict"
            ),
            "assignment_conflict_agenda_source": "a47_clear_target_push_stand",
            "parent_target_prerequisite_obligation": dict(prerequisite_obligation),
            "assigned_box_cell": list(assigned_box_cell),
            "assigned_slot_cell": list(assigned_slot_cell),
            "required_stand_cell": list(required_stand_cell),
            "target_blocker_cell": None
            if target_blocker_cell is None
            else list(target_blocker_cell),
            "assigned_box_is_required_stand_blocker": True,
            "assigned_box_pull_distance": float(
                self._assignment_conflict_pull_distance(
                    assigned_box_cell,
                    assigned_slot_cell,
                )
            ),
            "assigned_box_legal_push_count": int(legal_push_count),
            "assigned_box_toward_slot_push_count": int(toward_slot_count),
            "assigned_box_push_candidates": push_candidates,
            "completion_predicates": [
                "required_stand_empty_reachable",
                "target_blocker_push_legal",
                "assigned_box_moved_from_required_stand",
            ],
        }

    def _assignment_conflict_metrics(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        obligation: dict[str, Any],
    ) -> dict[str, Any]:
        boxes = set(state.boxes)
        reachable = set(self.model.reachable_cells(state))
        assigned_box_cell = self._target_prerequisite_cell(
            obligation.get("assigned_box_cell")
        )
        assigned_slot_cell = self._target_prerequisite_cell(
            obligation.get("assigned_slot_cell")
        )
        required_stand_cell = self._target_prerequisite_cell(
            obligation.get("required_stand_cell")
        )
        target_blocker_cell = self._target_prerequisite_cell(
            obligation.get("target_blocker_cell")
        )
        parent_prereq = obligation.get("parent_target_prerequisite_obligation", {})
        target_metrics = (
            self._target_prerequisite_metrics(state, commitment_state, parent_prereq)
            if isinstance(parent_prereq, dict) and parent_prereq
            else {}
        )
        push_candidates = self._assignment_conflict_push_candidates(
            state,
            assigned_box_cell=assigned_box_cell,
            assigned_slot_cell=assigned_slot_cell,
        )
        legal_push_count = int(
            sum(bool(row.get("push_legal_now", False)) for row in push_candidates)
        )
        toward_slot_push_count = int(
            sum(
                bool(row.get("push_legal_now", False))
                and bool(row.get("moves_toward_assigned_slot", False))
                for row in push_candidates
            )
        )
        min_after_distance = min(
            [
                float(row["after_pull_distance"])
                for row in push_candidates
                if bool(row.get("push_legal_now", False))
                and math.isfinite(float(row["after_pull_distance"]))
            ],
            default=math.inf,
        )
        schedule = self.cached_schedule_plan_summary(
            state,
            commitment_state,
            lookahead_slots=self.room_schedule_lookahead_slots,
        )
        slot = self.model.room_slot_progress_summary(state, commitment_state)
        totals = schedule["totals"]
        assigned_present = bool(
            assigned_box_cell is not None and assigned_box_cell in boxes
        )
        assigned_at_required_stand = bool(
            assigned_box_cell is not None
            and required_stand_cell is not None
            and assigned_box_cell == required_stand_cell
            and assigned_box_cell in boxes
        )
        return {
            "assigned_box_present": bool(assigned_present),
            "assigned_box_at_required_stand": bool(assigned_at_required_stand),
            "required_stand_occupied_by_box": bool(
                required_stand_cell is not None and required_stand_cell in boxes
            ),
            "required_stand_reachable_by_player": bool(
                required_stand_cell is not None and required_stand_cell in reachable
            ),
            "target_blocker_present": bool(
                target_blocker_cell is not None and target_blocker_cell in boxes
            ),
            "target_blocker_push_legal": bool(
                target_metrics.get("target_blocker_push_legal", False)
            ),
            "assigned_box_pull_distance": float(
                self._assignment_conflict_pull_distance(
                    assigned_box_cell,
                    assigned_slot_cell,
                )
            ),
            "assigned_box_min_legal_after_pull_distance": float(
                min_after_distance
            ),
            "assigned_box_legal_push_count": int(legal_push_count),
            "assigned_box_toward_slot_push_count": int(toward_slot_push_count),
            "assigned_box_push_candidates": push_candidates,
            "slot_schedule_cost": float(slot["slot_schedule_cost"]),
            "dependency_total": int(totals["dependency_count"])
            + int(totals["future_dependency_count"]),
            "target_blocker_count": int(totals["target_blocker_count"]),
            "release_action_count": int(totals["release_action_count"]),
            "nonblocked_legal_push_count": int(
                self._nonblocked_legal_push_count(state)
            ),
            "no_legal_pushes": bool(self._nonblocked_legal_push_count(state) <= 0),
        }

    def _assignment_conflict_progress_summary(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        depth: int,
        path_loop: bool,
    ) -> dict[str, Any]:
        assigned_left_required_stand = bool(
            before.get("assigned_box_at_required_stand", False)
            and not after.get("assigned_box_at_required_stand", False)
        )
        required_stand_empty_reachable = bool(
            not after.get("required_stand_occupied_by_box", True)
            and after.get("required_stand_reachable_by_player", False)
        )
        target_push_legal = bool(after.get("target_blocker_push_legal", False))
        assigned_mobility_opened = bool(
            int(before.get("assigned_box_legal_push_count", 0)) <= 0
            and int(after.get("assigned_box_legal_push_count", 0)) > 0
        )
        assigned_toward_slot_opened = bool(
            int(before.get("assigned_box_toward_slot_push_count", 0)) <= 0
            and int(after.get("assigned_box_toward_slot_push_count", 0)) > 0
        )
        before_distance = float(before.get("assigned_box_pull_distance", math.inf))
        after_distance = float(after.get("assigned_box_pull_distance", math.inf))
        distance_delta = (
            before_distance - after_distance
            if math.isfinite(before_distance) and math.isfinite(after_distance)
            else 0.0
        )
        dependency_delta = int(before.get("dependency_total", 0)) - int(
            after.get("dependency_total", 0)
        )
        target_blocker_delta = int(before.get("target_blocker_count", 0)) - int(
            after.get("target_blocker_count", 0)
        )
        legal_delta = int(after.get("nonblocked_legal_push_count", 0)) - int(
            before.get("nonblocked_legal_push_count", 0)
        )
        conflict_done = bool(
            target_push_legal
            or required_stand_empty_reachable
            or (
                assigned_left_required_stand
                and not after.get("required_stand_occupied_by_box", True)
            )
        )
        intermediate_progress = bool(
            assigned_mobility_opened
            or assigned_toward_slot_opened
            or distance_delta > 0.0
            or target_blocker_delta > 0
            or dependency_delta > 0
        )
        true_progress = bool(conflict_done or intermediate_progress)
        if conflict_done:
            tier = 9
        elif assigned_toward_slot_opened or distance_delta > 0.0:
            tier = 6
        elif assigned_mobility_opened or target_blocker_delta > 0:
            tier = 5
        elif path_loop or bool(after.get("no_legal_pushes", False)):
            tier = 1
        else:
            tier = 2
        score = (
            120.0 * float(tier)
            + 48.0 * float(conflict_done)
            + 18.0 * float(assigned_toward_slot_opened)
            + 14.0 * float(assigned_mobility_opened)
            + 5.0 * float(max(0.0, distance_delta))
            + 8.0 * float(max(0, dependency_delta))
            + 8.0 * float(max(0, target_blocker_delta))
            + 1.5 * float(max(0, legal_delta))
            - 24.0 * float(path_loop)
            - 2.0 * float(depth)
        )
        return {
            "assignment_conflict_agenda_tier": int(tier),
            "assignment_conflict_true_progress": bool(true_progress),
            "assignment_conflict_done": bool(conflict_done),
            "target_blocker_push_legal": bool(target_push_legal),
            "required_stand_empty_reachable": bool(required_stand_empty_reachable),
            "assigned_box_left_required_stand": bool(assigned_left_required_stand),
            "assigned_box_mobility_opened": bool(assigned_mobility_opened),
            "assigned_box_toward_slot_opened": bool(assigned_toward_slot_opened),
            "assigned_box_pull_distance_delta": float(distance_delta),
            "dependency_total_delta": int(dependency_delta),
            "target_blocker_count_delta": int(target_blocker_delta),
            "legal_push_count_delta": int(legal_delta),
            "path_loop": bool(path_loop),
            "depth": int(depth),
            "score": float(score),
        }

    @staticmethod
    def _assignment_conflict_option_key(row: dict[str, Any]) -> tuple:
        progress = row["final_progress"]
        return (
            int(progress.get("assignment_conflict_done", False)),
            int(progress.get("target_blocker_push_legal", False)),
            int(progress.get("required_stand_empty_reachable", False)),
            int(progress.get("assigned_box_left_required_stand", False)),
            int(progress.get("assigned_box_toward_slot_opened", False)),
            int(progress.get("assigned_box_mobility_opened", False)),
            float(progress.get("assigned_box_pull_distance_delta", 0.0)),
            int(progress.get("target_blocker_count_delta", 0)),
            int(progress.get("dependency_total_delta", 0)),
            float(progress.get("score", 0.0)),
            -int(progress.get("depth", 0)),
        )

    def _assignment_conflict_search(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        obligation: dict[str, Any],
    ) -> dict[str, Any]:
        reference_metrics = self._assignment_conflict_metrics(
            state,
            commitment_state,
            obligation,
        )
        queue: list[dict[str, Any]] = [
            {
                "state": state,
                "commitment_state": commitment_state,
                "actions": [],
                "commitments": [],
                "rows": [],
                "seen": {self.model.gft_canonical_key_cached(state)},
            }
        ]
        best: dict[str, Any] | None = None
        evaluated_count = 0
        expanded = 0
        while queue and expanded < self.assignment_conflict_agenda_max_nodes:
            node = queue.pop(0)
            depth = len(node["actions"])
            if depth >= self.assignment_conflict_agenda_max_pushes:
                continue
            if self.model.is_solved(node["state"]):
                continue
            actions = sorted(
                self.model.legal_flows(node["state"]),
                key=lambda item: int(item.flat_index),
            )
            for action in actions:
                child = self.model.step_grid_flow(node["state"], action)
                if self.is_blocked(child):
                    continue
                child_key = self.model.gft_canonical_key_cached(child)
                path_loop = bool(child_key in node["seen"])
                child_commitment = self._advance_recovery_commitment(
                    node["state"],
                    node["commitment_state"],
                    action,
                    child,
                )
                metrics = self._assignment_conflict_metrics(
                    child,
                    child_commitment,
                    obligation,
                )
                child_depth = depth + 1
                progress = self._assignment_conflict_progress_summary(
                    reference_metrics,
                    metrics,
                    depth=child_depth,
                    path_loop=path_loop,
                )
                source, target = self._moved_cells(node["state"], action)
                row = {
                    "macro_index": int(child_depth - 1),
                    "depth": int(child_depth),
                    "physical_action": self._physical_action_dict(
                        node["state"],
                        action,
                    ),
                    "reason": "assignment_conflict_agenda_suffix_push",
                    "source": list(source),
                    "target": list(target),
                    "path_loop_after": bool(path_loop),
                    "assignment_conflict_agenda_obligation": obligation,
                    "assignment_conflict_reference_metrics": reference_metrics,
                    "assignment_conflict_metrics": {
                        key: value
                        for key, value in metrics.items()
                        if key != "assigned_box_push_candidates"
                    },
                    "assignment_conflict_progress": progress,
                    "assignment_conflict_score": float(progress["score"]),
                }
                candidate_actions = [*node["actions"], action]
                candidate_commitments = [*node["commitments"], child_commitment]
                candidate_rows = [*node["rows"], row]
                candidate = {
                    "actions": candidate_actions,
                    "commitments": candidate_commitments,
                    "rows": candidate_rows,
                    "final_progress": progress,
                    "final_metrics": metrics,
                }
                evaluated_count += 1
                if best is None or self._assignment_conflict_option_key(
                    candidate
                ) > self._assignment_conflict_option_key(best):
                    best = candidate
                if (
                    child_depth < self.assignment_conflict_agenda_max_pushes
                    and not path_loop
                    and not bool(metrics.get("no_legal_pushes", False))
                    and not bool(progress.get("assignment_conflict_done", False))
                ):
                    queue.append(
                        {
                            "state": child,
                            "commitment_state": child_commitment,
                            "actions": candidate_actions,
                            "commitments": candidate_commitments,
                            "rows": candidate_rows,
                            "seen": set(node["seen"]) | {child_key},
                        }
                    )
                expanded += 1
                if expanded >= self.assignment_conflict_agenda_max_nodes:
                    break
        return {
            "selected_option": best,
            "evaluated_count": int(evaluated_count),
            "expanded_nodes": int(expanded),
            "reference_metrics": reference_metrics,
        }

    def _assignment_conflict_record(
        self,
        *,
        triggered: bool,
        reason: str,
        prerequisite_record: dict[str, Any] | None,
        obligation: dict[str, Any] | None = None,
        selected_option: dict[str, Any] | None = None,
        evaluated_count: int = 0,
        expanded_nodes: int = 0,
    ) -> dict[str, Any]:
        rows = []
        progress = {}
        if isinstance(selected_option, dict):
            maybe_rows = selected_option.get("rows", [])
            if isinstance(maybe_rows, list):
                rows = maybe_rows
            maybe_progress = selected_option.get("final_progress", {})
            if isinstance(maybe_progress, dict):
                progress = maybe_progress
        return {
            "assignment_conflict_agenda_enabled": bool(
                self.assignment_conflict_agenda_enabled
            ),
            "assignment_conflict_agenda_scope": (
                "assigned_box_blocks_target_prerequisite_e3_e6_e14_v1"
            ),
            "assignment_conflict_agenda_max_pushes": int(
                self.assignment_conflict_agenda_max_pushes
            ),
            "assignment_conflict_agenda_max_nodes": int(
                self.assignment_conflict_agenda_max_nodes
            ),
            "assignment_conflict_agenda_min_total_pushes": int(
                self.assignment_conflict_agenda_min_total_pushes
            ),
            "assignment_conflict_agenda_require_done": bool(
                self.assignment_conflict_agenda_require_done
            ),
            "assignment_conflict_agenda_triggered": bool(triggered),
            "assignment_conflict_agenda_reason": reason,
            "source_target_prerequisite_agenda_reason": None
            if not isinstance(prerequisite_record, dict)
            else prerequisite_record.get("target_prerequisite_agenda_reason"),
            "source_target_prerequisite_agenda_triggered": bool(
                isinstance(prerequisite_record, dict)
                and prerequisite_record.get(
                    "target_prerequisite_agenda_triggered",
                    False,
                )
            ),
            "assignment_conflict_obligation": dict(obligation or {}),
            "evaluated_option_count": int(evaluated_count),
            "expanded_nodes": int(expanded_nodes),
            "committed_conflict_push_count": int(len(rows) if triggered else 0),
            "validated_rows": list(rows if triggered else []),
            "final_progress": dict(progress if triggered else {}),
            "best_option_progress": dict(progress),
        }

    def _assignment_conflict_extend_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
        prerequisite_record: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.assignment_conflict_agenda_enabled:
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="disabled",
                prerequisite_record=prerequisite_record,
            )
        if (
            prerequisite_record.get("target_prerequisite_agenda_reason")
            != "no_target_prerequisite_progress_option"
        ):
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="source_not_target_prerequisite_failure",
                prerequisite_record=prerequisite_record,
            )
        prerequisite_obligation = prerequisite_record.get(
            "target_prerequisite_obligation",
            {},
        )
        if not isinstance(prerequisite_obligation, dict) or not prerequisite_obligation:
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="missing_target_prerequisite_obligation",
                prerequisite_record=prerequisite_record,
            )
        conflict_state, conflict_commitment = self._state_after_macro_actions(
            state,
            commitment_state,
            macro,
        )
        obligation = self._assignment_conflict_obligation(
            conflict_state,
            prerequisite_obligation,
        )
        if obligation is None:
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="no_parent_assigned_prerequisite_conflict",
                prerequisite_record=prerequisite_record,
            )
        search = self._assignment_conflict_search(
            conflict_state,
            conflict_commitment,
            obligation,
        )
        selected_option = search.get("selected_option")
        if not isinstance(selected_option, dict) or selected_option is None:
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="no_assignment_conflict_option",
                prerequisite_record=prerequisite_record,
                obligation=obligation,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        final_progress = selected_option.get("final_progress", {})
        if (
            self.assignment_conflict_agenda_require_done
            and not bool(final_progress.get("assignment_conflict_done", False))
        ):
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="no_assignment_conflict_done_option",
                prerequisite_record=prerequisite_record,
                obligation=obligation,
                selected_option=selected_option,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        if (
            len(selected_option.get("actions", []))
            < self.assignment_conflict_agenda_min_total_pushes
        ):
            return macro, self._assignment_conflict_record(
                triggered=False,
                reason="assignment_conflict_option_too_short",
                prerequisite_record=prerequisite_record,
                obligation=obligation,
                selected_option=selected_option,
                evaluated_count=int(search.get("evaluated_count", 0)),
                expanded_nodes=int(search.get("expanded_nodes", 0)),
            )
        actions = list(macro.get("physical_actions", [])) + list(
            selected_option["actions"]
        )
        commitments = list(macro.get("after_commitment_states", [])) + list(
            selected_option["commitments"]
        )
        rows = list(macro.get("rows", [])) + list(selected_option["rows"])
        extended_macro = {
            **macro,
            "enabled": True,
            "triggered": True,
            "reason": "active_blocker_commitment_with_assignment_conflict_agenda",
            "physical_actions": actions,
            "after_commitment_states": commitments,
            "rows": rows,
            "macro_push_count": int(len(actions)),
            "extra_push_count": int(max(0, len(actions) - 1)),
            "assignment_conflict_agenda_enabled": True,
            "assignment_conflict_agenda_triggered": True,
            "assignment_conflict_agenda_reason": (
                "validated_assignment_conflict_agenda_committed"
            ),
            "assignment_conflict_agenda_obligation": obligation,
            "assignment_conflict_agenda_final_progress": final_progress,
            "assignment_conflict_agenda_push_count": int(
                len(selected_option["actions"])
            ),
        }
        record = self._assignment_conflict_record(
            triggered=True,
            reason="validated_assignment_conflict_agenda_committed",
            prerequisite_record=prerequisite_record,
            obligation=obligation,
            selected_option=selected_option,
            evaluated_count=int(search.get("evaluated_count", 0)),
            expanded_nodes=int(search.get("expanded_nodes", 0)),
        )
        return extended_macro, record

    def _target_prerequisite_extend_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        macro: dict[str, Any],
        residual_record: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prerequisite_macro, prerequisite_record = super()._target_prerequisite_extend_macro(
            state,
            commitment_state,
            macro,
            residual_record,
        )
        conflict_macro, conflict_record = self._assignment_conflict_extend_macro(
            state,
            commitment_state,
            prerequisite_macro,
            prerequisite_record,
        )
        self._assignment_conflict_agenda_summary = conflict_record
        return conflict_macro, prerequisite_record

    def plan_with_corridor_macro(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState | None = None,
    ) -> dict[str, Any]:
        self._assignment_conflict_agenda_summary = {}
        plan = super().plan_with_corridor_macro(state, commitment_state)
        plan["assignment_conflict_agenda"] = dict(
            self._assignment_conflict_agenda_summary
        )
        return plan


class RoomTraceAnchoredGFTSDMCTSPlanner(RoomCorridorMacroGFTSDMCTSPlanner):
    """A7 planner with a bounded teacher/reference trace anchor.

    This class is a diagnostic hard-instance subclass.  It does not change the
    exact transition model; it only adds root-level reference action/slot priors
    for states that appear in an externally supplied trace.
    """

    provider_id = "room_trace_anchored_gft_sdmcts_v1"

    def __init__(
        self,
        *args,
        reference_action_by_state_key: dict[tuple, dict[str, Any]] | None = None,
        trace_anchor_action_bonus: float = 30.0,
        trace_anchor_action_mismatch_penalty: float = 0.0,
        trace_anchor_slot_bonus: float = 10.0,
        trace_anchor_slot_mismatch_penalty: float = 2.0,
        trace_anchor_root_override: bool = True,
        trace_anchor_macro_require_reference: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_action_by_state_key = dict(reference_action_by_state_key or {})
        self.trace_anchor_action_bonus = float(trace_anchor_action_bonus)
        self.trace_anchor_action_mismatch_penalty = float(trace_anchor_action_mismatch_penalty)
        self.trace_anchor_slot_bonus = float(trace_anchor_slot_bonus)
        self.trace_anchor_slot_mismatch_penalty = float(trace_anchor_slot_mismatch_penalty)
        self.trace_anchor_root_override = bool(trace_anchor_root_override)
        self.trace_anchor_macro_require_reference = bool(trace_anchor_macro_require_reference)

    def _reference_state_key(self, state: SokobanState) -> tuple:
        return self.model.gft_canonical_key_cached(state)

    def reference_row(self, state: SokobanState) -> dict[str, Any] | None:
        return self.reference_action_by_state_key.get(self._reference_state_key(state))

    def _trace_anchor_score(
        self,
        state: SokobanState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        reference = self.reference_row(state)
        action = candidate.action.physical_action
        if reference is None:
            candidate.room_trace_anchor_summary = {
                "reference_state_hit": False,
                "score": 0.0,
            }
            candidate.room_trace_anchor_score = 0.0
            return 0.0

        action_match = int(reference["flat_index"]) == int(action.flat_index)
        future_goal = reference.get("future_goal_cell")
        future_goal_cell = None if future_goal is None else tuple(future_goal)
        option = candidate.action.option
        slot_match = None
        score = 0.0
        if action_match:
            score += self.trace_anchor_action_bonus
        else:
            score -= self.trace_anchor_action_mismatch_penalty
        if action_match and future_goal_cell is not None and option.kind == "slot":
            slot_match = option.slot_cell == future_goal_cell
            if slot_match:
                score += self.trace_anchor_slot_bonus
            else:
                score -= self.trace_anchor_slot_mismatch_penalty

        candidate.room_trace_anchor_score = float(score)
        candidate.room_trace_anchor_summary = {
            "reference_state_hit": True,
            "reference_step": int(reference["step"]),
            "reference_action": {
                "box_index": int(reference["box_index"]),
                "direction": int(reference["direction"]),
                "direction_name": ACTION_NAMES[int(reference["direction"])],
                "flat_index": int(reference["flat_index"]),
                "source": list(reference["source"]),
                "target": list(reference["target"]),
            },
            "future_goal_cell": None if future_goal_cell is None else list(future_goal_cell),
            "candidate_flat_index": int(action.flat_index),
            "action_match": bool(action_match),
            "slot_match": None if slot_match is None else bool(slot_match),
            "score": float(score),
        }
        return float(score)

    def _room_prior_score(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        candidate: RoomSemanticCandidate,
    ) -> float:
        base = super()._room_prior_score(state, commitment_state, candidate)
        return float(base + self._trace_anchor_score(state, candidate))

    def _reference_slot_grade(self, edge: RoomSemanticEdge, reference: dict[str, Any]) -> int:
        future_goal = reference.get("future_goal_cell")
        if future_goal is None or edge.action.option.kind != "slot":
            return 0
        return int(edge.action.option.slot_cell == tuple(future_goal))

    def choose_root_edge(self, root: RoomSemanticNode) -> RoomSemanticEdge:
        if not self.trace_anchor_root_override:
            return super().choose_root_edge(root)
        reference = self.reference_row(root.state)
        if reference is None:
            return super().choose_root_edge(root)
        matches = [
            edge
            for edge in root.children.values()
            if int(edge.action.physical_action.flat_index) == int(reference["flat_index"])
        ]
        if not matches:
            return super().choose_root_edge(root)
        return max(
            matches,
            key=lambda edge: (
                self._reference_slot_grade(edge, reference),
                edge.visits,
                edge.mean_value,
                edge.prior_score,
            ),
        )

    def _reference_macro_match_count(
        self,
        state: SokobanState,
        actions: list[SokobanPushAction],
    ) -> int:
        current = state
        matched = 0
        for action in actions:
            reference = self.reference_row(current)
            if reference is None or int(reference["flat_index"]) != int(action.flat_index):
                break
            matched += 1
            current = self.model.step_grid_flow(current, action)
        return matched

    def _trim_macro_to_reference(
        self,
        macro: dict[str, Any],
        match_count: int,
    ) -> dict[str, Any]:
        actions = list(macro.get("physical_actions", []))
        commitments = list(macro.get("after_commitment_states", []))
        rows = list(macro.get("rows", []))
        keep = min(max(int(match_count), 1), len(actions))
        trimmed_actions = actions[:keep]
        trimmed_commitments = commitments[:keep]
        trimmed = {
            **macro,
            "triggered": keep >= self.corridor_macro_min_total_pushes,
            "reason": "reference_macro_mismatch",
            "physical_actions": trimmed_actions,
            "after_commitment_states": trimmed_commitments,
            "rows": rows[:keep],
            "macro_push_count": int(keep),
            "extra_push_count": int(max(0, keep - 1)),
            "reference_macro_checked": True,
            "reference_macro_match_count": int(match_count),
            "reference_macro_trimmed": True,
        }
        return trimmed

    def corridor_macro_from_plan(
        self,
        state: SokobanState,
        commitment_state: RoomCommitmentState,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        macro = super().corridor_macro_from_plan(state, commitment_state, plan)
        actions = list(macro.get("physical_actions", []))
        if not self.trace_anchor_macro_require_reference or not actions:
            return macro
        if self.reference_row(state) is None:
            macro["reference_macro_checked"] = False
            macro["reference_macro_match_count"] = 0
            macro["reference_macro_trimmed"] = False
            return macro
        match_count = self._reference_macro_match_count(state, actions)
        if match_count == len(actions):
            macro["reference_macro_checked"] = True
            macro["reference_macro_match_count"] = int(match_count)
            macro["reference_macro_trimmed"] = False
            return macro
        return self._trim_macro_to_reference(macro, match_count)


def semantic_child_count_by_physical(
    candidates: Iterable[RoomSemanticCandidate],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        physical = candidate.action.physical_action
        label = f"b{physical.box_index}:{ACTION_NAMES[physical.direction]}"
        counts[label] = counts.get(label, 0) + 1
    return counts
