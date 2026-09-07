#!/usr/bin/env python3
"""Stage-2 audit for learned Sokoban SD transition provider and SD-MCTS wiring.

The audit consumes a learned stock-flow checkpoint from the stage-1 flow model.
It then wraps the network as a transition provider for abstract stocks and runs
a small deterministic SD-MCTS smoke. Raw Sokoban states are retained only as
handles for legal single-push action support; all abstract transition values
used by the planner come from the learned provider.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.sokoban import SokobanGoalDirectedSDModel, SokobanState, load_boxoban_levels
from src.sokoban.sd_model import SokobanPushAction
from train_sokoban_flow_smoke import (
    FLOW_PAIRS,
    FeedbackFlowNet,
    FlowNet,
    action_features,
    feedback_features_tensor,
    forward_flows_from_tensors,
    raw_next_stock,
    tensor_features_from_stock_action,
)
from validate_sokoban_sd_design import FLOW_NAMES, INCIDENCE_MATRIX, STOCK_NAMES, stock_proxy


PROVIDER_ID = "learned_stock_flow_checkpoint_v1"
RAW_STATE_SUPPORT_PROVIDER = "exact_push_raw_state_support_for_action_enumeration"
TARGET_AUDIT_PROVIDER = "exact_stock_proxy_target_for_audit_only"


@dataclass
class ProviderPrediction:
    stock_before: np.ndarray
    flows: np.ndarray
    raw_next_stock: np.ndarray
    next_stock: np.ndarray
    clip_correction: np.ndarray
    provider_id: str


@dataclass
class MCTSNode:
    raw_state: SokobanState
    stock: np.ndarray
    depth: int
    parent: "MCTSNode | None" = None
    action: SokobanPushAction | None = None
    prior: float = 0.0
    reward: float = 0.0
    children: Dict[SokobanPushAction, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Torch checkpoint written by train_sokoban_flow_smoke.py --checkpoint-out.",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory used for provider audit.",
    )
    parser.add_argument(
        "--use-checkpoint-valid-levels",
        action="store_true",
        default=False,
        help="Audit the held-out level ids recorded in the checkpoint summary when available.",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--level-offset", type=int, default=0)
    parser.add_argument("--max-root-depth", type=int, default=1)
    parser.add_argument("--max-root-states-per-level", type=int, default=12)
    parser.add_argument("--num-simulations", type=int, default=48)
    parser.add_argument("--mcts-depth", type=int, default=2)
    parser.add_argument("--puct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/stage2_learned_provider_sdmcts_audit.json",
        help="JSON output path.",
    )
    return parser


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _relative(path: str | Path) -> str:
    path = Path(path)
    return os.path.relpath(path, ROOT) if path.is_absolute() else str(path)


def action_to_dict(action: SokobanPushAction | None) -> Dict[str, Any] | None:
    if action is None:
        return None
    return {
        "box_index": int(action.box_index),
        "direction": int(action.direction),
        "flat_index": int(action.flat_index),
    }


class LearnedStockFlowTransitionProvider:
    """Checkpoint-backed provider for abstract stock-flow transitions."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.payload = _torch_load(checkpoint_path)
        self.architecture = self.payload["architecture"]
        self.summary = self.payload.get("summary", {})
        self.model_kind = str(self.architecture["model_kind"])
        self.feature_mode = str(self.architecture["feature_mode"])
        self.incidence = torch.tensor(INCIDENCE_MATRIX, dtype=torch.float32)
        self.model = self._build_model()
        self.model.load_state_dict(self.payload["model_state_dict"])
        self.model.eval()

    def _build_model(self) -> nn.Module:
        if self.model_kind == "feedback":
            return FeedbackFlowNet(
                int(self.architecture["input_dim"]),
                int(self.architecture["feedback_dim"]),
                int(self.architecture["hidden_dim"]),
                len(FLOW_NAMES),
            )
        return FlowNet(
            int(self.architecture["input_dim"]),
            int(self.architecture["hidden_dim"]),
            len(FLOW_NAMES),
            str(self.architecture["flow_parameterization"]),
            torch.tensor(self.architecture["flow_pair_scales"], dtype=torch.float32),
            torch.tensor(self.architecture["capacity_projection_stock_mask"], dtype=torch.float32),
            int(self.architecture["capacity_projection_passes"]),
        )

    def predict(
        self,
        model: SokobanGoalDirectedSDModel,
        raw_state: SokobanState,
        stock: np.ndarray,
        action: SokobanPushAction,
    ) -> ProviderPrediction:
        stock_tensor = torch.tensor(stock, dtype=torch.float32).view(1, -1)
        action_tensor = torch.tensor(
            action_features(model, raw_state, action),
            dtype=torch.float32,
        ).view(1, -1)
        features = tensor_features_from_stock_action(
            stock_tensor,
            action_tensor,
            self.feature_mode,
        )
        feedback = feedback_features_tensor(stock_tensor)
        with torch.no_grad():
            flows = forward_flows_from_tensors(
                self.model,
                features,
                feedback,
                self.model_kind,
            )
            raw_next = raw_next_stock(stock_tensor, flows, self.incidence)
            next_stock = torch.clamp(raw_next, 0.0, 1.0)

        raw_np = raw_next.detach().cpu().numpy()[0]
        next_np = next_stock.detach().cpu().numpy()[0]
        return ProviderPrediction(
            stock_before=np.asarray(stock, dtype=np.float64),
            flows=flows.detach().cpu().numpy()[0].astype(np.float64),
            raw_next_stock=raw_np.astype(np.float64),
            next_stock=next_np.astype(np.float64),
            clip_correction=np.abs(next_np - raw_np).astype(np.float64),
            provider_id=PROVIDER_ID,
        )


def stock_reward(before: np.ndarray, after: np.ndarray) -> float:
    """Task reward over learned SD stocks, not over exact Sokoban search heuristics."""

    delta = after - before
    weights = np.array([2.0, -0.8, 0.35, 0.25, -0.25, -1.0], dtype=np.float64)
    return float(np.dot(weights, delta) - 0.01)


def stock_value(stock: np.ndarray) -> float:
    weights = np.array([2.0, -0.8, 0.35, 0.25, -0.25, -1.0], dtype=np.float64)
    return float(np.dot(weights, stock))


def puct_score(parent: MCTSNode, child: MCTSNode, c: float) -> float:
    exploration = c * child.prior * math.sqrt(parent.visits + 1.0) / (1.0 + child.visits)
    return child.value + exploration


def run_sdmcts_smoke(
    model: SokobanGoalDirectedSDModel,
    root_state: SokobanState,
    provider: LearnedStockFlowTransitionProvider,
    *,
    num_simulations: int,
    max_depth: int,
    puct_c: float,
    gamma: float,
) -> Dict[str, Any]:
    root_stock = stock_proxy(model, root_state).as_array()
    root = MCTSNode(raw_state=root_state, stock=root_stock.astype(np.float64), depth=0)
    expansion_records: list[dict[str, Any]] = []
    provider_counts: Counter[str] = Counter()
    raw_support_counts: Counter[str] = Counter()

    for _sim in range(num_simulations):
        node = root
        path = [node]
        while node.depth < max_depth:
            legal_actions = tuple(model.legal_pushes(node.raw_state))
            if not legal_actions:
                break
            unexpanded = [action for action in legal_actions if action not in node.children]
            if unexpanded:
                action = unexpanded[0]
                prediction = provider.predict(model, node.raw_state, node.stock, action)
                provider_counts[prediction.provider_id] += 1
                raw_support_counts[RAW_STATE_SUPPORT_PROVIDER] += 1
                raw_next_state = model.step_state(node.raw_state, action)
                reward = stock_reward(node.stock, prediction.next_stock)
                child = MCTSNode(
                    raw_state=raw_next_state,
                    stock=prediction.next_stock,
                    depth=node.depth + 1,
                    parent=node,
                    action=action,
                    prior=1.0 / max(len(legal_actions), 1),
                    reward=reward,
                )
                node.children[action] = child
                exact_next_stock = stock_proxy(model, raw_next_state).as_array()
                expansion_records.append(
                    {
                        "depth": int(child.depth),
                        "action": action_to_dict(action),
                        "provider_id": prediction.provider_id,
                        "raw_state_support_provider": RAW_STATE_SUPPORT_PROVIDER,
                        "target_provider_for_audit_only": TARGET_AUDIT_PROVIDER,
                        "stock_before": prediction.stock_before.tolist(),
                        "predicted_flows": prediction.flows.tolist(),
                        "raw_next_stock": prediction.raw_next_stock.tolist(),
                        "predicted_next_stock": prediction.next_stock.tolist(),
                        "exact_next_stock_for_audit": exact_next_stock.tolist(),
                        "clip_mean_abs_correction": float(np.mean(prediction.clip_correction)),
                        "stock_abs_error": np.abs(prediction.next_stock - exact_next_stock).tolist(),
                    }
                )
                node = child
                path.append(node)
                break
            node = max(node.children.values(), key=lambda child: puct_score(node, child, puct_c))
            path.append(node)

        leaf_value = stock_value(node.stock)
        total = leaf_value
        for path_node in reversed(path):
            if path_node.parent is not None:
                total = path_node.reward + gamma * total
            path_node.visits += 1
            path_node.value_sum += total

    root_children = list(root.children.items())
    visit_sum = sum(child.visits for _action, child in root_children)
    policy = [
        {
            "action": action_to_dict(action),
            "visits": int(child.visits),
            "q_value": float(child.value),
            "probability": float(child.visits / visit_sum) if visit_sum else 0.0,
        }
        for action, child in sorted(root_children, key=lambda item: item[0].flat_index)
    ]
    best = max(policy, key=lambda row: row["visits"]) if policy else None
    return {
        "root_legal_actions": int(len(model.legal_pushes(root_state))),
        "expanded_children": int(len(root.children)),
        "tree_expansions": int(len(expansion_records)),
        "root_visits": int(root.visits),
        "root_value": float(root.value),
        "best_action": best["action"] if best else None,
        "root_policy": policy,
        "provider_counts": dict(provider_counts),
        "raw_state_support_counts": dict(raw_support_counts),
        "expansion_records": expansion_records,
    }


def select_levels(
    provider: LearnedStockFlowTransitionProvider,
    *,
    levels_path: str,
    max_levels: int,
    level_offset: int,
    use_checkpoint_valid_levels: bool,
) -> tuple[list[Any], dict[str, Any]]:
    checkpoint_config = provider.summary.get("config", {})
    checkpoint_dataset = provider.summary.get("dataset", {})
    load_count = max(max_levels + level_offset, int(checkpoint_config.get("max_levels", max_levels)))
    levels = load_boxoban_levels(levels_path, max_levels=load_count)

    if use_checkpoint_valid_levels:
        valid_level_ids = set(checkpoint_dataset.get("split", {}).get("valid_levels", []))
        selected = [level for level in levels if level.level_id in valid_level_ids]
        if selected:
            return selected, {
                "selection_mode": "checkpoint_valid_levels",
                "valid_level_ids": sorted(valid_level_ids),
            }

    selected = levels[level_offset : level_offset + max_levels]
    return selected, {
        "selection_mode": "offset_slice",
        "level_offset": level_offset,
        "max_levels": max_levels,
    }


def summarize_expansions(rows: Sequence[dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    flows = np.array([row["predicted_flows"] for row in rows], dtype=np.float64)
    before = np.array([row["stock_before"] for row in rows], dtype=np.float64)
    raw_next = np.array([row["raw_next_stock"] for row in rows], dtype=np.float64)
    pred_next = np.array([row["predicted_next_stock"] for row in rows], dtype=np.float64)
    exact_next = np.array([row["exact_next_stock_for_audit"] for row in rows], dtype=np.float64)
    depths = np.array([row["depth"] for row in rows], dtype=np.int64)

    reconstructed_raw = before + flows @ INCIDENCE_MATRIX.T
    incidence_balance = np.abs(reconstructed_raw - raw_next).sum(axis=1)
    clipped = np.logical_or(raw_next < 0.0, raw_next > 1.0)
    clip_correction = np.abs(pred_next - raw_next)
    stock_abs_error = np.abs(pred_next - exact_next)

    pair_coactive = []
    for left, right in FLOW_PAIRS:
        left_idx = FLOW_NAMES.index(left)
        right_idx = FLOW_NAMES.index(right)
        pair_coactive.append(np.logical_and(flows[:, left_idx] > 1e-6, flows[:, right_idx] > 1e-6))
    pair_coactive_arr = np.stack(pair_coactive, axis=1)

    per_depth = {}
    for depth in sorted(set(depths.tolist())):
        mask = depths == depth
        per_depth[f"depth_{depth}"] = {
            "stock_mae": float(stock_abs_error[mask].mean()),
            "samples": int(mask.sum()),
            "per_stock_mae": {
                name: float(stock_abs_error[mask, idx].mean())
                for idx, name in enumerate(STOCK_NAMES)
            },
        }

    return {
        "transitions_audited": int(len(rows)),
        "flow_nonnegative_violations": int((flows < -1e-9).any(axis=1).sum()),
        "stock_range_violations": int(np.logical_or(pred_next < -1e-9, pred_next > 1.0 + 1e-9).any(axis=1).sum()),
        "incidence_balance_max_l1": float(incidence_balance.max()),
        "clip_any_stock_fraction": float(clipped.any(axis=1).mean()),
        "clip_mean_abs_correction": float(clip_correction.mean()),
        "flow_pair_pred_coactivation_mean": float(pair_coactive_arr.mean()),
        "flow_pair_pred_coactivation_max": float(pair_coactive_arr.mean(axis=0).max()),
        "stock_mae": float(stock_abs_error.mean()),
        "per_stock_mae": {
            name: float(stock_abs_error[:, idx].mean())
            for idx, name in enumerate(STOCK_NAMES)
        },
        "per_depth": per_depth,
    }


def evaluate_gate(metrics: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = {
        "roots_checked_min": 1,
        "roots_with_expanded_children_fraction_min": 1.0,
        "learned_provider_fraction_min": 1.0,
        "flow_nonnegative_violations_max": 0,
        "stock_range_violations_max": 0,
        "incidence_balance_max_l1_max": 1e-6,
        "clip_any_stock_fraction_max": 0.75,
        "clip_mean_abs_correction_max": 0.01214,
        "flow_pair_pred_coactivation_mean_max": 0.20,
        "flow_pair_pred_coactivation_each_max": 0.30,
        "depth_1_stock_mae_max": 0.06,
        "depth_2_stock_mae_max": 0.07,
    }
    observed = {
        "roots_checked": aggregate["roots_checked"],
        "roots_with_expanded_children_fraction": aggregate["roots_with_expanded_children_fraction"],
        "learned_provider_fraction": aggregate["learned_provider_fraction"],
        "flow_nonnegative_violations": metrics.get("flow_nonnegative_violations"),
        "stock_range_violations": metrics.get("stock_range_violations"),
        "incidence_balance_max_l1": metrics.get("incidence_balance_max_l1"),
        "clip_any_stock_fraction": metrics.get("clip_any_stock_fraction"),
        "clip_mean_abs_correction": metrics.get("clip_mean_abs_correction"),
        "flow_pair_pred_coactivation_mean": metrics.get("flow_pair_pred_coactivation_mean"),
        "flow_pair_pred_coactivation_max": metrics.get("flow_pair_pred_coactivation_max"),
        "depth_1_stock_mae": metrics.get("per_depth", {}).get("depth_1", {}).get("stock_mae"),
        "depth_2_stock_mae": metrics.get("per_depth", {}).get("depth_2", {}).get("stock_mae"),
    }

    failed: list[str] = []
    if observed["roots_checked"] < thresholds["roots_checked_min"]:
        failed.append("roots_checked")
    if observed["roots_with_expanded_children_fraction"] < thresholds["roots_with_expanded_children_fraction_min"]:
        failed.append("roots_with_expanded_children_fraction")
    if observed["learned_provider_fraction"] < thresholds["learned_provider_fraction_min"]:
        failed.append("learned_provider_fraction")
    for key in (
        "flow_nonnegative_violations",
        "stock_range_violations",
        "incidence_balance_max_l1",
        "clip_any_stock_fraction",
        "clip_mean_abs_correction",
        "flow_pair_pred_coactivation_mean",
    ):
        value = observed[key]
        if value is None or value > thresholds[f"{key}_max"]:
            failed.append(key)
    if (
        observed["flow_pair_pred_coactivation_max"] is None
        or observed["flow_pair_pred_coactivation_max"] > thresholds["flow_pair_pred_coactivation_each_max"]
    ):
        failed.append("flow_pair_pred_coactivation_max")
    if observed["depth_1_stock_mae"] is None or observed["depth_1_stock_mae"] > thresholds["depth_1_stock_mae_max"]:
        failed.append("depth_1_stock_mae")
    if observed["depth_2_stock_mae"] is None or observed["depth_2_stock_mae"] > thresholds["depth_2_stock_mae_max"]:
        failed.append("depth_2_stock_mae")

    return {
        "verdict": "PASS" if not failed else "PARTIAL",
        "failed_checks": failed,
        "thresholds": thresholds,
        "observed": observed,
    }


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = perf_counter()

    provider = LearnedStockFlowTransitionProvider(Path(args.checkpoint))
    levels, selection = select_levels(
        provider,
        levels_path=args.levels,
        max_levels=args.max_levels,
        level_offset=args.level_offset,
        use_checkpoint_valid_levels=args.use_checkpoint_valid_levels,
    )
    if not levels:
        raise SystemExit("No levels selected for stage-2 audit.")

    root_rows: list[dict[str, Any]] = []
    expansion_rows: list[dict[str, Any]] = []
    provider_counts: Counter[str] = Counter()
    raw_support_counts: Counter[str] = Counter()

    for level in levels:
        sd_model = SokobanGoalDirectedSDModel(level)
        checked = 0
        for root_state, root_depth in sd_model_iter_states(
            sd_model,
            max_depth=args.max_root_depth,
            max_states=args.max_root_states_per_level,
        ):
            if checked >= args.max_root_states_per_level:
                break
            result = run_sdmcts_smoke(
                sd_model,
                root_state,
                provider,
                num_simulations=args.num_simulations,
                max_depth=args.mcts_depth,
                puct_c=args.puct_c,
                gamma=args.gamma,
            )
            checked += 1
            provider_counts.update(result["provider_counts"])
            raw_support_counts.update(result["raw_state_support_counts"])
            for record in result["expansion_records"]:
                record["level_id"] = level.level_id
                record["root_depth"] = int(root_depth)
                expansion_rows.append(record)
            root_rows.append(
                {
                    "level_id": level.level_id,
                    "root_depth": int(root_depth),
                    "root_legal_actions": result["root_legal_actions"],
                    "expanded_children": result["expanded_children"],
                    "tree_expansions": result["tree_expansions"],
                    "root_visits": result["root_visits"],
                    "root_value": result["root_value"],
                    "best_action": result["best_action"],
                    "root_policy": result["root_policy"][:8],
                }
            )

    metrics = summarize_expansions(expansion_rows)
    roots_checked = len(root_rows)
    roots_with_expanded = sum(1 for row in root_rows if row["expanded_children"] > 0)
    provider_total = sum(provider_counts.values())
    aggregate = {
        "levels_checked": len(levels),
        "roots_checked": roots_checked,
        "roots_with_expanded_children": roots_with_expanded,
        "roots_with_expanded_children_fraction": float(roots_with_expanded / roots_checked)
        if roots_checked
        else 0.0,
        "provider_counts": dict(provider_counts),
        "raw_state_support_counts": dict(raw_support_counts),
        "learned_provider_fraction": float(provider_counts.get(PROVIDER_ID, 0) / provider_total)
        if provider_total
        else 0.0,
        "elapsed_sec": float(perf_counter() - t0),
    }
    gate = evaluate_gate(metrics, aggregate)

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "VERIFIED" if gate["verdict"] == "PASS" else "ANALYZED",
            "version_label": "sokoban_stage2_learned_provider_sdmcts_audit_v1",
        },
        "config": {
            "checkpoint": _relative(args.checkpoint),
            "levels": _relative(args.levels),
            "level_selection": selection,
            "max_root_depth": args.max_root_depth,
            "max_root_states_per_level": args.max_root_states_per_level,
            "num_simulations": args.num_simulations,
            "mcts_depth": args.mcts_depth,
            "puct_c": args.puct_c,
            "gamma": args.gamma,
            "seed": args.seed,
        },
        "provider_contract": {
            "abstract_transition_provider": PROVIDER_ID,
            "state_encoder": "diagnostic_stock_proxy_encoder_for_root_and_audit_targets",
            "raw_state_support_provider": RAW_STATE_SUPPORT_PROVIDER,
            "target_provider_for_audit_only": TARGET_AUDIT_PROVIDER,
            "learned_transition_provider_used_for_sdmcts": True,
            "exact_push_dynamics_used_for_abstract_transition_scores": False,
            "action_scope": "legal_single_push_option",
        },
        "checkpoint": {
            "path": _relative(args.checkpoint),
            "provider_id": provider.payload.get("checkpoint_contract", {}).get("provider_id"),
            "architecture": provider.architecture,
            "stage1_summary_config": provider.summary.get("config", {}),
        },
        "aggregate": aggregate,
        "metrics": metrics,
        "gate": gate,
        "root_samples": root_rows[:20],
        "expansion_samples": expansion_rows[:20],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "metrics": metrics, "gate": gate}, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


def sd_model_iter_states(
    model: SokobanGoalDirectedSDModel,
    *,
    max_depth: int,
    max_states: int,
):
    from validate_sokoban_sd_design import iter_states

    yield from iter_states(model, max_depth=max_depth, max_states=max_states)


if __name__ == "__main__":
    main()
