#!/usr/bin/env python3
"""Train a search-value TVA network for Sokoban SD-MCTS.

This script implements the strict TVA target used in the SD-MCTS paper:
the value target is constructed from MCTS root visit counts and root child
Q estimates. Diagnostic teachers are not used to create the training target.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage2_learned_provider_sdmcts import (  # noqa: E402
    LearnedStockFlowTransitionProvider,
    ProviderPrediction,
    action_to_dict,
    stock_reward,
    stock_value,
)
from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    ExpertAbstractTerminalValueEvaluator,
    PlannerAction,
    PlannerFeedback,
    SokobanTvaTerminalValueEvaluator,
    StockTerminalValueEvaluator,
    TerminalValueEvaluator,
    ZeroTerminalValueEvaluator,
    choose_mcts_action,
    make_planner_feedback,
    rollout_planner_action,
)
from sokoban_tva import (  # noqa: E402
    TVA_FEATURE_NAMES,
    SokobanTvaValueNet,
    feature_summary,
    relative_path,
    tva_feature_vector,
)
from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels  # noqa: E402
from src.sokoban.sd_model import SokobanPushAction  # noqa: E402
from train_sokoban_tva import (  # noqa: E402
    batch_iter,
    fallback_level_split,
    indices_for_split,
    provider_level_split,
    regression_metrics,
)
from validate_sokoban_sd_design import STOCK_NAMES, iter_states, stock_proxy  # noqa: E402


@dataclass
class SearchPlanningNode:
    raw_state: SokobanState
    stock: np.ndarray
    depth: int
    parent: "SearchPlanningNode | None" = None
    action: SokobanPushAction | None = None
    prior: float = 0.0
    reward: float = 0.0
    children: Dict[SokobanPushAction, "SearchPlanningNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchValueSample:
    features: np.ndarray
    target_value: float
    split: str
    level_id: str
    state_depth: int
    root_policy: list[dict[str, Any]]
    root_children: int
    root_visits: int
    tree_expansions: int
    provider_counts: dict[str, int]
    terminal_value_counts: dict[str, int]
    terminal_leaf_counts: dict[str, int]
    root_state_type: str
    target_source: str = "sd_mcts_root_statistics"
    td_available: bool = False
    td_reward: float = 0.0
    td_discount: float = 1.0
    td_next_features: np.ndarray | None = None
    td_terminal: bool = False
    td_next_terminal_value: float = 0.0


class TerminalAwareStockValueEvaluator(StockTerminalValueEvaluator):
    evaluator_id = "terminal_aware_fixed_stock_value_for_search_target"

    def should_stop(self, model: SokobanGoalDirectedSDModel, raw_state: SokobanState) -> bool:
        return model.is_solved(raw_state) or not model.legal_pushes(raw_state)

    def value(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
    ) -> float:
        if model.is_solved(raw_state):
            return 10.0
        if model.is_deadlocked(raw_state) or not model.legal_pushes(raw_state):
            return -10.0
        return stock_value(stock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
        help="Stage-2 learned stock-flow provider checkpoint.",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used to collect root states.",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--state-depth", type=int, default=3)
    parser.add_argument("--max-states-per-level", type=int, default=90)
    parser.add_argument("--collect-budget", type=int, default=48)
    parser.add_argument("--mcts-depth", type=int, default=3)
    parser.add_argument("--puct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--target-planner-mode",
        choices=("basic", "stage3"),
        default="basic",
        help=(
            "Source planner for V_search targets. 'basic' preserves the original "
            "stock-reward primitive MCTS collector; 'stage3' uses the same planner "
            "feedback/action abstraction configuration as stage-3 SD-MCTS."
        ),
    )
    parser.add_argument(
        "--planner-feedback-mode",
        choices=("stock", "expert"),
        default="stock",
        help="Planner feedback used only when --target-planner-mode=stage3.",
    )
    parser.add_argument("--expert-delta-weight", type=float, default=0.25)
    parser.add_argument("--expert-prior-temperature", type=float, default=1.0)
    parser.add_argument("--expert-prior-uniform-mix", type=float, default=0.10)
    parser.add_argument("--revisit-feedback-penalty", type=float, default=2.0)
    parser.add_argument("--expert-viability-penalty", type=float, default=2.0)
    parser.add_argument("--goal-room-order-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-abstraction-mode",
        choices=("push", "target2", "target2_prior"),
        default="push",
        help="Stage-3 planner action abstraction used only in stage3 target mode.",
    )
    parser.add_argument("--max-option-length", type=int, default=2)
    parser.add_argument("--max-options-per-node", type=int, default=6)
    parser.add_argument("--option-prior-max-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--option-execution-mode",
        choices=("first_push", "full_option"),
        default="first_push",
        help="Execution semantics for TD successors when stage3 target mode selects a multi-push option.",
    )
    parser.add_argument("--route-consistency-weight", type=float, default=0.0)
    parser.add_argument("--audit-proxy-prior-weight", type=float, default=0.0)
    parser.add_argument("--goal-room-conflict-prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--bootstrap-terminal-value-mode",
        choices=("stock", "zero", "tva", "expert"),
        default="stock",
        help="Leaf evaluator used only while collecting search-value targets.",
    )
    parser.add_argument(
        "--bootstrap-tva-checkpoint",
        default="",
        help="Checkpoint used when --bootstrap-terminal-value-mode=tva.",
    )
    parser.add_argument(
        "--root-action-selection",
        choices=("visits", "value"),
        default="visits",
        help="Action selection recorded for optional TD samples.",
    )
    parser.add_argument("--lambda-td", type=float, default=0.0)
    parser.add_argument("--target-sync-epochs", type=int, default=25)
    parser.add_argument(
        "--positive-target-sample-weight",
        type=float,
        default=1.0,
        help="Weight for samples whose search-value target is positive.",
    )
    parser.add_argument(
        "--nonpositive-target-sample-weight",
        type=float,
        default=1.0,
        help="Weight for samples whose search-value target is zero or negative.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qh-audit-max-states", type=int, default=80)
    parser.add_argument("--value-provider-id", default="sokoban_search_value_tva_v1")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_search_value_tva_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_search_value_tva_seed0.pt",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def puct_score(parent: SearchPlanningNode, child: SearchPlanningNode, c: float) -> float:
    exploration = c * child.prior * math.sqrt(parent.visits + 1.0) / (1.0 + child.visits)
    return child.value + exploration


def make_bootstrap_terminal_value(args: argparse.Namespace) -> TerminalValueEvaluator:
    if args.bootstrap_terminal_value_mode == "stock":
        return TerminalAwareStockValueEvaluator()
    if args.bootstrap_terminal_value_mode == "zero":
        return ZeroTerminalValueEvaluator()
    if args.bootstrap_terminal_value_mode == "expert":
        return ExpertAbstractTerminalValueEvaluator(make_planner_feedback(args, force_expert=True))
    if args.bootstrap_terminal_value_mode == "tva":
        if not args.bootstrap_tva_checkpoint:
            raise SystemExit("--bootstrap-tva-checkpoint is required when bootstrap mode is tva.")
        return SokobanTvaTerminalValueEvaluator(Path(args.bootstrap_tva_checkpoint))
    raise SystemExit(f"Unknown bootstrap terminal value mode: {args.bootstrap_terminal_value_mode}")


def root_state_type(model: SokobanGoalDirectedSDModel, state: SokobanState) -> str:
    if model.is_solved(state):
        return "solved"
    if model.is_deadlocked(state):
        return "deadlocked"
    if not model.legal_pushes(state):
        return "no_legal_actions"
    return "active"


def compute_search_value(root_policy: Sequence[dict[str, Any]], alpha: float) -> tuple[float, list[float]]:
    if not root_policy:
        raise ValueError("Cannot compute V_search without root children.")
    visits = np.array([max(float(row["visits"]), 0.0) for row in root_policy], dtype=np.float64)
    q_values = np.array([float(row["q_value"]) for row in root_policy], dtype=np.float64)
    if not np.isfinite(q_values).all():
        raise ValueError("Root policy contains non-finite q_value.")
    if alpha == 0.0:
        weights = np.ones_like(visits)
    else:
        weights = np.power(visits, float(alpha))
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(visits)
    probs = weights / weights.sum()
    return float(np.dot(probs, q_values)), [float(v) for v in probs.tolist()]


def run_mcts_root_search_stats(
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    provider: LearnedStockFlowTransitionProvider,
    terminal_value: TerminalValueEvaluator,
    *,
    num_simulations: int,
    max_depth: int,
    puct_c: float,
    gamma: float,
    root_action_selection: str,
    rng: np.random.RandomState,
) -> tuple[SokobanPushAction | None, dict[str, Any]]:
    root_stock = stock_proxy(model, root_state).as_array().astype(np.float64)
    root = SearchPlanningNode(raw_state=root_state, stock=root_stock, depth=0)
    provider_counts: Counter[str] = Counter()
    terminal_value_counts: Counter[str] = Counter()
    terminal_leaf_counts: Counter[str] = Counter()
    expansions = 0

    for _sim in range(num_simulations):
        node = root
        path = [node]
        while node.depth < max_depth and not terminal_value.should_stop(model, node.raw_state):
            legal_actions = tuple(model.legal_pushes(node.raw_state))
            if not legal_actions:
                break
            unexpanded = [action for action in legal_actions if action not in node.children]
            if unexpanded:
                action = unexpanded[int(rng.randint(len(unexpanded)))]
                prediction: ProviderPrediction = provider.predict(model, node.raw_state, node.stock, action)
                provider_counts[prediction.provider_id] += 1
                next_raw_state = model.step_state(node.raw_state, action)
                child = SearchPlanningNode(
                    raw_state=next_raw_state,
                    stock=prediction.next_stock,
                    depth=node.depth + 1,
                    parent=node,
                    action=action,
                    prior=1.0 / max(len(legal_actions), 1),
                    reward=stock_reward(node.stock, prediction.next_stock),
                )
                node.children[action] = child
                expansions += 1
                node = child
                path.append(node)
                break
            node = max(node.children.values(), key=lambda child: puct_score(node, child, puct_c))
            path.append(node)

        total = terminal_value.value(model, node.raw_state, node.stock)
        terminal_value_counts[terminal_value.evaluator_id] += 1
        if model.is_solved(node.raw_state):
            terminal_leaf_counts["solved"] += 1
        elif model.is_deadlocked(node.raw_state):
            terminal_leaf_counts["deadlocked"] += 1
        elif not model.legal_pushes(node.raw_state):
            terminal_leaf_counts["no_legal_actions"] += 1
        elif node.depth >= max_depth:
            terminal_leaf_counts["depth_cutoff"] += 1
        else:
            terminal_leaf_counts["frontier_cutoff"] += 1

        for path_node in reversed(path):
            if path_node.parent is not None:
                total = path_node.reward + gamma * total
            path_node.visits += 1
            path_node.value_sum += total

    if not root.children:
        return None, {
            "root_children": 0,
            "root_visits": int(root.visits),
            "root_value": float(root.value),
            "root_policy": [],
            "provider_counts": dict(provider_counts),
            "terminal_value_counts": dict(terminal_value_counts),
            "terminal_leaf_counts": dict(terminal_leaf_counts),
            "tree_expansions": int(expansions),
            "best_child_value": None,
            "qh_depth1_audit": {},
        }

    children = list(root.children.items())
    visit_sum = sum(child.visits for _action, child in children)
    if root_action_selection == "visits":
        best_action, best_child = max(children, key=lambda item: (item[1].visits, item[1].value, -item[0].flat_index))
    elif root_action_selection == "value":
        best_action, best_child = max(children, key=lambda item: (item[1].value, item[1].visits, -item[0].flat_index))
    else:
        raise ValueError(f"Unknown root_action_selection: {root_action_selection}")

    policy = []
    qh_errors = []
    for action, child in sorted(children, key=lambda item: item[0].flat_index):
        row = {
            "action": action_to_dict(action),
            "visits": int(child.visits),
            "q_value": float(child.value),
            "probability": float(child.visits / visit_sum) if visit_sum else 0.0,
        }
        if max_depth == 1:
            leaf = terminal_value.value(model, child.raw_state, child.stock)
            expected = child.reward + gamma * leaf
            err = abs(float(child.value) - float(expected))
            row["depth1_expected_q_value"] = float(expected)
            row["depth1_q_value_abs_error"] = float(err)
            qh_errors.append(err)
        policy.append(row)

    qh_audit = {}
    if max_depth == 1:
        qh_audit = {
            "checked_children": int(len(qh_errors)),
            "max_abs_error": float(max(qh_errors) if qh_errors else 0.0),
            "mean_abs_error": float(np.mean(qh_errors) if qh_errors else 0.0),
            "threshold": 1e-9,
            "verdict": "PASS" if qh_errors and max(qh_errors) <= 1e-9 else "FAIL",
        }
    return best_action, {
        "root_children": int(len(root.children)),
        "root_visits": int(root.visits),
        "root_value": float(root.value),
        "best_child_value": float(best_child.value),
        "root_policy": policy,
        "provider_counts": dict(provider_counts),
        "terminal_value_counts": dict(terminal_value_counts),
        "terminal_leaf_counts": dict(terminal_leaf_counts),
        "tree_expansions": int(expansions),
        "root_action_selection": root_action_selection,
        "qh_depth1_audit": qh_audit,
    }


def run_stage3_planner_root_search_stats(
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    provider: LearnedStockFlowTransitionProvider,
    terminal_value: TerminalValueEvaluator,
    planner_feedback: PlannerFeedback,
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> tuple[PlannerAction | None, dict[str, Any]]:
    """Collect root statistics from the same planner family used in stage 3.

    TVA should align the leaf value function to the search statistics actually
    consumed by SD-MCTS. This path preserves the learned transition provider and
    reuses the stage-3 feedback/action-abstraction layer without online teachers.
    """

    return choose_mcts_action(
        model,
        root_state,
        provider,
        terminal_value,
        planner_feedback,
        num_simulations=args.collect_budget,
        max_depth=args.mcts_depth,
        puct_c=args.puct_c,
        gamma=args.gamma,
        root_action_selection=args.root_action_selection,
        visited_states=Counter(),
        revisit_feedback_penalty=args.revisit_feedback_penalty,
        action_abstraction_mode=args.action_abstraction_mode,
        max_option_length=args.max_option_length,
        max_options_per_node=args.max_options_per_node,
        option_prior_max_multiplier=args.option_prior_max_multiplier,
        previous_push=None,
        previous_pushed_box_cell=None,
        route_consistency_weight=args.route_consistency_weight,
        audit_proxy_prior_weight=args.audit_proxy_prior_weight,
        goal_room_conflict_prior_weight=args.goal_room_conflict_prior_weight,
        learned_action_prior_payload=None,
        learned_action_prior_weight=0.0,
        learned_action_prior_max_multiplier=1.0,
        rng=rng,
    )


def td_successor_from_basic_action(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    stock: np.ndarray,
    action: SokobanPushAction | None,
    terminal_value: TerminalValueEvaluator,
    args: argparse.Namespace,
) -> tuple[bool, float, float, np.ndarray | None, bool, float]:
    if action is None:
        return False, 0.0, 1.0, None, False, 0.0
    next_state = model.step_state(state, action)
    next_stock = stock_proxy(model, next_state).as_array().astype(np.float32)
    td_reward = stock_reward(stock.astype(np.float64), next_stock.astype(np.float64))
    next_features = tva_feature_vector(model, next_state, next_stock)
    td_terminal = (
        model.is_solved(next_state)
        or model.is_deadlocked(next_state)
        or not model.legal_pushes(next_state)
    )
    td_next_terminal_value = (
        terminal_value.value(model, next_state, next_stock.astype(np.float64))
        if td_terminal
        else 0.0
    )
    return True, float(td_reward), float(args.gamma), next_features, bool(td_terminal), float(td_next_terminal_value)


def td_successor_from_stage3_action(
    model: SokobanGoalDirectedSDModel,
    state: SokobanState,
    stock: np.ndarray,
    action: PlannerAction | None,
    provider: LearnedStockFlowTransitionProvider,
    terminal_value: TerminalValueEvaluator,
    planner_feedback: PlannerFeedback,
    args: argparse.Namespace,
) -> tuple[bool, float, float, np.ndarray | None, bool, float]:
    if action is None:
        return False, 0.0, 1.0, None, False, 0.0
    if args.option_execution_mode == "first_push" and action.primitives:
        action = PlannerAction(
            primitives=action.primitives[:1],
            source=f"{action.source}_td_first_push",
            structural_score=action.structural_score,
        )
    rollout = rollout_planner_action(
        model,
        state,
        stock,
        action,
        provider,
        planner_feedback,
        gamma=args.gamma,
        visited_states=Counter(),
        revisit_feedback_penalty=args.revisit_feedback_penalty,
    )
    if rollout.duration <= 0:
        return False, 0.0, 1.0, None, False, 0.0
    next_features = tva_feature_vector(model, rollout.next_raw_state, rollout.next_stock.astype(np.float32))
    td_terminal = (
        model.is_solved(rollout.next_raw_state)
        or model.is_deadlocked(rollout.next_raw_state)
        or not model.legal_pushes(rollout.next_raw_state)
    )
    td_next_terminal_value = (
        terminal_value.value(model, rollout.next_raw_state, rollout.next_stock)
        if td_terminal
        else 0.0
    )
    return (
        True,
        float(rollout.reward),
        float(rollout.discount),
        next_features,
        bool(td_terminal),
        float(td_next_terminal_value),
    )


def collect_samples(
    args: argparse.Namespace,
    provider: LearnedStockFlowTransitionProvider,
    terminal_value: TerminalValueEvaluator,
) -> tuple[list[SearchValueSample], dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit("No levels loaded for search-value TVA training.")

    split = provider_level_split(args.provider_checkpoint)
    all_loaded_ids = {level.level_id for level in levels}
    if not all_loaded_ids.intersection(split["train_levels"]):
        split = fallback_level_split(sorted(all_loaded_ids), valid_fraction=0.2, seed=args.seed)

    samples: list[SearchValueSample] = []
    skipped_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    terminal_value_counts: Counter[str] = Counter()
    terminal_leaf_counts: Counter[str] = Counter()
    planner_feedback_counts: Counter[str] = Counter()
    option_candidates = 0
    option_expansions = 0
    root_child_counts: list[int] = []
    finite_targets = 0
    planner_feedback = make_planner_feedback(args) if args.target_planner_mode == "stage3" else None
    t0 = perf_counter()

    for level_index, level in enumerate(levels):
        if level.level_id in split["train_levels"]:
            split_name = "train"
        elif level.level_id in split["valid_levels"]:
            split_name = "valid"
        else:
            continue
        model = SokobanGoalDirectedSDModel(level)
        rng = np.random.RandomState(args.seed + level_index * 1009)
        level_samples = 0
        seen_states: set[SokobanState] = set()
        for state, state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            if state in seen_states:
                continue
            seen_states.add(state)
            stock = stock_proxy(model, state).as_array().astype(np.float32)
            if args.target_planner_mode == "stage3":
                assert planner_feedback is not None
                best_action, info = run_stage3_planner_root_search_stats(
                    model,
                    state,
                    provider,
                    terminal_value,
                    planner_feedback,
                    args,
                    rng,
                )
            else:
                best_action, info = run_mcts_root_search_stats(
                    model,
                    state,
                    provider,
                    terminal_value,
                    num_simulations=args.collect_budget,
                    max_depth=args.mcts_depth,
                    puct_c=args.puct_c,
                    gamma=args.gamma,
                    root_action_selection=args.root_action_selection,
                    rng=rng,
                )
            if not info["root_policy"]:
                skipped_counts[root_state_type(model, state)] += 1
                continue
            target, search_probs = compute_search_value(info["root_policy"], args.alpha)
            if not np.isfinite(target):
                skipped_counts["nonfinite_v_search"] += 1
                continue
            finite_targets += 1
            root_policy = []
            for row, prob in zip(info["root_policy"], search_probs):
                copied = dict(row)
                copied["search_probability"] = float(prob)
                root_policy.append(copied)

            td_available = False
            td_reward = 0.0
            td_discount = 1.0
            next_features = None
            td_terminal = False
            td_next_terminal_value = 0.0
            if args.target_planner_mode == "stage3":
                assert planner_feedback is not None
                (
                    td_available,
                    td_reward,
                    td_discount,
                    next_features,
                    td_terminal,
                    td_next_terminal_value,
                ) = td_successor_from_stage3_action(
                    model,
                    state,
                    stock,
                    best_action,
                    provider,
                    terminal_value,
                    planner_feedback,
                    args,
                )
            else:
                (
                    td_available,
                    td_reward,
                    td_discount,
                    next_features,
                    td_terminal,
                    td_next_terminal_value,
                ) = td_successor_from_basic_action(
                    model,
                    state,
                    stock,
                    best_action,
                    terminal_value,
                    args,
                )

            samples.append(
                SearchValueSample(
                    features=tva_feature_vector(model, state, stock),
                    target_value=target,
                    split=split_name,
                    level_id=level.level_id,
                    state_depth=int(state_depth),
                    root_policy=root_policy,
                    root_children=int(info.get("root_children", len(info["root_policy"]))),
                    root_visits=int(
                        info.get(
                            "root_visits",
                            sum(int(row.get("visits", 0)) for row in info["root_policy"]),
                        )
                    ),
                    tree_expansions=int(info["tree_expansions"]),
                    provider_counts={str(k): int(v) for k, v in info["provider_counts"].items()},
                    terminal_value_counts={str(k): int(v) for k, v in info["terminal_value_counts"].items()},
                    terminal_leaf_counts={str(k): int(v) for k, v in info["terminal_leaf_counts"].items()},
                    root_state_type=root_state_type(model, state),
                    td_available=td_available,
                    td_reward=float(td_reward),
                    td_discount=float(td_discount),
                    td_next_features=next_features,
                    td_terminal=bool(td_terminal),
                    td_next_terminal_value=float(td_next_terminal_value),
                )
            )
            provider_counts.update(info["provider_counts"])
            terminal_value_counts.update(info["terminal_value_counts"])
            terminal_leaf_counts.update(info["terminal_leaf_counts"])
            planner_feedback_counts.update(info.get("planner_feedback_counts", {}))
            option_candidates += int(info.get("option_candidates", 0))
            option_expansions += int(info.get("option_expansions", 0))
            root_child_counts.append(int(info["root_children"]))
            split_counts[split_name] += 1
            level_samples += 1

        print(
            json.dumps(
                {
                    "level_index": level_index,
                    "level_id": level.level_id,
                    "split": split_name,
                    "search_value_samples": level_samples,
                },
                ensure_ascii=False,
            )
        )

    if not any(sample.split == "train" for sample in samples):
        raise SystemExit("Search-value TVA collection produced no train samples.")

    return samples, {
        "level_split": split,
        "split_counts": dict(split_counts),
        "skipped_counts": dict(skipped_counts),
        "provider_counts": dict(provider_counts),
        "terminal_value_counts": dict(terminal_value_counts),
        "terminal_leaf_counts": dict(terminal_leaf_counts),
        "planner_feedback_counts": dict(planner_feedback_counts),
        "option_candidates": int(option_candidates),
        "option_expansions": int(option_expansions),
        "finite_v_search_fraction": float(finite_targets / max(finite_targets + sum(skipped_counts.values()), 1)),
        "root_child_summary": {
            "mean": float(np.mean(root_child_counts)) if root_child_counts else 0.0,
            "min": int(min(root_child_counts)) if root_child_counts else 0,
            "max": int(max(root_child_counts)) if root_child_counts else 0,
        },
        "collection_elapsed_sec": float(perf_counter() - t0),
    }


def audit_qh_semantics(
    args: argparse.Namespace,
    provider: LearnedStockFlowTransitionProvider,
    terminal_value: TerminalValueEvaluator,
) -> dict[str, Any]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    rows = []
    checked_states = 0
    for level_index, level in enumerate(levels):
        if checked_states >= args.qh_audit_max_states:
            break
        model = SokobanGoalDirectedSDModel(level)
        rng = np.random.RandomState(args.seed + 4242 + level_index)
        for state, _state_depth in iter_states(
            model,
            max_depth=args.state_depth,
            max_states=args.max_states_per_level,
        ):
            if checked_states >= args.qh_audit_max_states:
                break
            if not model.legal_pushes(state):
                continue
            _action, info = run_mcts_root_search_stats(
                model,
                state,
                provider,
                terminal_value,
                num_simulations=args.collect_budget,
                max_depth=1,
                puct_c=args.puct_c,
                gamma=args.gamma,
                root_action_selection=args.root_action_selection,
                rng=rng,
            )
            audit = info.get("qh_depth1_audit", {})
            if audit:
                rows.append(audit)
                checked_states += 1

    max_errors = [float(row.get("max_abs_error", 0.0)) for row in rows]
    failed = [err for err in max_errors if err > 1e-9]
    return {
        "checked_states": int(len(rows)),
        "max_abs_error": float(max(max_errors) if max_errors else 0.0),
        "mean_max_abs_error": float(np.mean(max_errors) if max_errors else 0.0),
        "failed_states": int(len(failed)),
        "threshold": 1e-9,
        "verdict": "PASS" if rows and not failed else "FAIL",
    }


def evaluate_model(
    model: SokobanTvaValueNet,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    if len(indices) == 0:
        return {}
    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        pred_norm = model(features[idx])
        pred = pred_norm.detach().cpu().numpy() * target_std + target_mean
    target = targets[idx].detach().cpu().numpy()
    return regression_metrics(pred, target)


def grouped_metrics(
    samples: Sequence[SearchValueSample],
    pred_values: np.ndarray,
    target_values: np.ndarray,
    indices: np.ndarray,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices.tolist():
        if target_values[idx] > 0.0:
            groups["positive_search_target"].append(idx)
        elif target_values[idx] < 0.0:
            groups["negative_search_target"].append(idx)
        else:
            groups["zero_search_target"].append(idx)
        groups[f"root_state:{samples[idx].root_state_type}"].append(idx)
    out: dict[str, dict[str, float]] = {}
    for name, group in sorted(groups.items()):
        arr = np.array(group, dtype=np.int64)
        out[name] = regression_metrics(pred_values[arr], target_values[arr])
        out[name]["samples"] = float(len(arr))
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    provider = LearnedStockFlowTransitionProvider(Path(args.provider_checkpoint))
    terminal_value = make_bootstrap_terminal_value(args)
    if args.target_planner_mode == "basic":
        qh_audit = audit_qh_semantics(args, provider, terminal_value)
    else:
        qh_audit = {
            "checked_states": 0,
            "verdict": "SKIPPED",
            "reason": "stage3 target mode reuses evaluate_stage3_sdmcts_planning.run_sdmcts_root_search",
            "target_planner_mode": args.target_planner_mode,
        }
    samples, collection = collect_samples(args, provider, terminal_value)

    features_np = np.stack([sample.features for sample in samples]).astype(np.float32)
    targets_np = np.array([sample.target_value for sample in samples], dtype=np.float32)
    next_features_np = np.stack(
        [
            sample.td_next_features if sample.td_next_features is not None else sample.features
            for sample in samples
        ]
    ).astype(np.float32)
    td_rewards_np = np.array([sample.td_reward for sample in samples], dtype=np.float32)
    td_discounts_np = np.array([sample.td_discount for sample in samples], dtype=np.float32)
    td_available_np = np.array([sample.td_available for sample in samples], dtype=bool)
    td_terminal_np = np.array([sample.td_terminal for sample in samples], dtype=bool)
    td_next_terminal_values_np = np.array(
        [sample.td_next_terminal_value for sample in samples],
        dtype=np.float32,
    )
    train_idx = indices_for_split(samples, "train")
    valid_idx = indices_for_split(samples, "valid")

    target_mean = float(np.mean(targets_np[train_idx]))
    target_std = float(max(np.std(targets_np[train_idx]), 1.0))
    targets_norm_np = (targets_np - target_mean) / target_std

    features = torch.tensor(features_np, dtype=torch.float32)
    next_features = torch.tensor(next_features_np, dtype=torch.float32)
    targets_norm = torch.tensor(targets_norm_np, dtype=torch.float32)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    td_rewards = torch.tensor(td_rewards_np, dtype=torch.float32)
    td_discounts = torch.tensor(td_discounts_np, dtype=torch.float32)
    td_available = torch.tensor(td_available_np, dtype=torch.bool)
    td_terminal = torch.tensor(td_terminal_np, dtype=torch.bool)
    td_next_terminal_values = torch.tensor(td_next_terminal_values_np, dtype=torch.float32)
    sample_weights_np = np.where(
        targets_np > 0.0,
        float(args.positive_target_sample_weight),
        float(args.nonpositive_target_sample_weight),
    ).astype(np.float32)
    sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32)

    model = SokobanTvaValueNet(input_dim=features.shape[1], hidden_dim=args.hidden_dim)
    target_model = deepcopy(model)
    target_model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    t0 = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        align_losses = []
        td_losses = []
        for batch in batch_iter(train_idx, args.batch_size, args.seed, epoch):
            idx = torch.tensor(batch, dtype=torch.long)
            pred = model(features[idx])
            raw_align_loss = (pred - targets_norm[idx]) ** 2
            weights = sample_weights[idx] / torch.clamp(sample_weights[idx].mean(), min=1e-12)
            align_loss = torch.mean(raw_align_loss * weights)
            loss = align_loss
            td_loss = torch.tensor(0.0)
            if args.lambda_td > 0.0:
                td_mask = td_available[idx]
                if bool(td_mask.any()):
                    with torch.no_grad():
                        next_norm = target_model(next_features[idx][td_mask])
                        next_value = next_norm * target_std + target_mean
                        next_value = torch.where(
                            td_terminal[idx][td_mask],
                            td_next_terminal_values[idx][td_mask],
                            next_value,
                        )
                        td_target = td_rewards[idx][td_mask] + td_discounts[idx][td_mask] * next_value
                        td_target_norm = (td_target - target_mean) / target_std
                    td_loss = torch.mean((pred[td_mask] - td_target_norm) ** 2)
                    loss = loss + float(args.lambda_td) * td_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
            align_losses.append(float(align_loss.detach().item()))
            td_losses.append(float(td_loss.detach().item()))

        if args.lambda_td > 0.0 and (epoch + 1) % max(args.target_sync_epochs, 1) == 0:
            target_model.load_state_dict(model.state_dict())
            target_model.eval()

        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            model.eval()
            train_metrics = evaluate_model(
                model,
                features,
                targets,
                train_idx,
                target_mean=target_mean,
                target_std=target_std,
            )
            valid_metrics = evaluate_model(
                model,
                features,
                targets,
                valid_idx,
                target_mean=target_mean,
                target_std=target_std,
            )
            row = {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "alignment_loss": float(np.mean(align_losses)) if align_losses else 0.0,
                "td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
                "train_mae": train_metrics.get("mae", 0.0),
                "valid_mae": valid_metrics.get("mae", 0.0),
                "valid_corr": valid_metrics.get("corr", 0.0),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))

    model.eval()
    with torch.no_grad():
        all_pred_norm = model(features)
        all_pred = all_pred_norm.detach().cpu().numpy() * target_std + target_mean

    metrics = {
        "train": regression_metrics(all_pred[train_idx], targets_np[train_idx]),
        "valid": regression_metrics(all_pred[valid_idx], targets_np[valid_idx]) if len(valid_idx) else {},
        "valid_by_group": grouped_metrics(samples, all_pred, targets_np, valid_idx) if len(valid_idx) else {},
    }

    root_state_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "valid"):
        counts = Counter(sample.root_state_type for sample in samples if sample.split == split_name)
        root_state_counts[split_name] = dict(counts)

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": args.value_provider_id,
        },
        "config": {
            "provider_checkpoint": relative_path(args.provider_checkpoint, ROOT),
            "levels": relative_path(args.levels, ROOT),
            "max_levels": args.max_levels,
            "state_depth": args.state_depth,
            "max_states_per_level": args.max_states_per_level,
            "collect_budget": args.collect_budget,
            "mcts_depth": args.mcts_depth,
            "puct_c": args.puct_c,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "target_planner_mode": args.target_planner_mode,
            "planner_feedback_mode": args.planner_feedback_mode,
            "expert_delta_weight": args.expert_delta_weight,
            "expert_prior_temperature": args.expert_prior_temperature,
            "expert_prior_uniform_mix": args.expert_prior_uniform_mix,
            "revisit_feedback_penalty": args.revisit_feedback_penalty,
            "expert_viability_penalty": args.expert_viability_penalty,
            "goal_room_order_weight": args.goal_room_order_weight,
            "action_abstraction_mode": args.action_abstraction_mode,
            "max_option_length": args.max_option_length,
            "max_options_per_node": args.max_options_per_node,
            "option_prior_max_multiplier": args.option_prior_max_multiplier,
            "option_execution_mode": args.option_execution_mode,
            "route_consistency_weight": args.route_consistency_weight,
            "audit_proxy_prior_weight": args.audit_proxy_prior_weight,
            "goal_room_conflict_prior_weight": args.goal_room_conflict_prior_weight,
            "bootstrap_terminal_value_mode": args.bootstrap_terminal_value_mode,
            "bootstrap_tva_checkpoint": relative_path(args.bootstrap_tva_checkpoint, ROOT)
            if args.bootstrap_tva_checkpoint
            else "",
            "root_action_selection": args.root_action_selection,
            "lambda_td": args.lambda_td,
            "target_sync_epochs": args.target_sync_epochs,
            "positive_target_sample_weight": args.positive_target_sample_weight,
            "nonpositive_target_sample_weight": args.nonpositive_target_sample_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "qh_audit_max_states": args.qh_audit_max_states,
            "value_provider_id": args.value_provider_id,
        },
        "tva_contract": {
            "value_provider_id": args.value_provider_id,
            "leaf_evaluation_only": True,
            "target_source": "sd_mcts_root_statistics",
            "target_planner_mode": args.target_planner_mode,
            "planner_feedback_id": (
                make_planner_feedback(args).feedback_id if args.target_planner_mode == "stage3" else "not_used"
            ),
            "planner_action_abstraction_mode": (
                args.action_abstraction_mode if args.target_planner_mode == "stage3" else "primitive_push_only"
            ),
            "planner_option_execution_mode_for_td": (
                args.option_execution_mode if args.target_planner_mode == "stage3" else "not_used"
            ),
            "mcts_root_statistics_used": True,
            "teacher_used_online": False,
            "teacher_used_for_training_target": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "provider_id": "learned_stock_flow_checkpoint_v1",
            "bootstrap_terminal_value_id": terminal_value.evaluator_id,
            "target_formula": "V_search=sum_a softmax_visits_alpha(a)*Q_H(s,a)",
            "td_formula": "r+discount*Vbar_or_terminal(s_next)" if args.lambda_td > 0.0 else "disabled_lambda_td_0",
            "td_terminal_successor_value_source": "same_terminal_value_evaluator_not_zero",
            "feature_names": list(TVA_FEATURE_NAMES),
            "stock_names": list(STOCK_NAMES),
        },
        "qh_semantics_gate": qh_audit,
        "dataset": {
            "samples": len(samples),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "feature_dim": int(features_np.shape[1]),
            "target_stats_train": {
                "mean": target_mean,
                "std": target_std,
                "min": float(np.min(targets_np[train_idx])),
                "max": float(np.max(targets_np[train_idx])),
            },
            "root_state_counts_by_split": root_state_counts,
            "td_available_fraction": float(np.mean(td_available_np)),
            "sample_weight_summary": {
                "mean": float(np.mean(sample_weights_np)),
                "min": float(np.min(sample_weights_np)),
                "max": float(np.max(sample_weights_np)),
                "positive_target_samples": int((targets_np > 0.0).sum()),
                "nonpositive_target_samples": int((targets_np <= 0.0).sum()),
                "positive_target_weight": float(args.positive_target_sample_weight),
                "nonpositive_target_weight": float(args.nonpositive_target_sample_weight),
            },
            "td_terminal_successor_samples": int(td_terminal_np.sum()),
            "td_terminal_successor_value_summary": {
                "mean": float(np.mean(td_next_terminal_values_np[td_terminal_np]))
                if bool(td_terminal_np.any())
                else 0.0,
                "min": float(np.min(td_next_terminal_values_np[td_terminal_np]))
                if bool(td_terminal_np.any())
                else 0.0,
                "max": float(np.max(td_next_terminal_values_np[td_terminal_np]))
                if bool(td_terminal_np.any())
                else 0.0,
            },
            "collection": collection,
            "feature_summary": feature_summary(features_np, TVA_FEATURE_NAMES),
        },
        "metrics": metrics,
        "train_by_group": grouped_metrics(samples, all_pred, targets_np, train_idx),
        "history": history,
        "elapsed_sec": float(perf_counter() - t0 + collection["collection_elapsed_sec"]),
    }

    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "checkpoint_contract": {
            "value_provider_id": args.value_provider_id,
            "leaf_evaluation_only": True,
            "target_source": "sd_mcts_root_statistics",
            "target_planner_mode": args.target_planner_mode,
            "planner_feedback_id": (
                make_planner_feedback(args).feedback_id if args.target_planner_mode == "stage3" else "not_used"
            ),
            "planner_action_abstraction_mode": (
                args.action_abstraction_mode if args.target_planner_mode == "stage3" else "primitive_push_only"
            ),
            "planner_option_execution_mode_for_td": (
                args.option_execution_mode if args.target_planner_mode == "stage3" else "not_used"
            ),
            "mcts_root_statistics_used": True,
            "teacher_used_online": False,
            "teacher_used_for_training_target": False,
            "best_first_used_online": False,
            "transition_provider_modified": False,
            "provider_id": "learned_stock_flow_checkpoint_v1",
            "bootstrap_terminal_value_id": terminal_value.evaluator_id,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "lambda_td": args.lambda_td,
            "target_network": {
                "enabled": bool(args.lambda_td > 0.0),
                "sync_epochs": int(args.target_sync_epochs),
            },
            "td_terminal_successor_value_source": "same_terminal_value_evaluator_not_zero",
        },
        "architecture": {
            "model_class": "SokobanTvaValueNet",
            "input_dim": int(features_np.shape[1]),
            "hidden_dim": int(args.hidden_dim),
        },
        "feature_names": list(TVA_FEATURE_NAMES),
        "target_stats": {
            "mean": target_mean,
            "std": target_std,
        },
        "model_state_dict": model.state_dict(),
        "summary": summary,
    }
    torch.save(checkpoint_payload, checkpoint_out)
    summary["checkpoint"] = {
        "saved": True,
        "path": relative_path(checkpoint_out, ROOT),
        "value_provider_id": args.value_provider_id,
    }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "qh_semantics_gate": summary["qh_semantics_gate"],
        "metrics": summary["metrics"],
        "checkpoint": summary["checkpoint"],
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    print(f"Wrote {args.checkpoint_out}")


if __name__ == "__main__":
    main()
