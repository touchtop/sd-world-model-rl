#!/usr/bin/env python3
"""Train a small learned action prior for Sokoban SD-MCTS.

The training target is deliberately local and auditable: closed-loop SD-MCTS
traces are collected only on the stage-2 provider train split, and labels are
derived from whether selected primitive pushes occur in solved or failed
episodes. The learned prior is later consumed only as an action-prior
multiplier; it never supplies transitions or online search labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_stage3_sdmcts_planning import (  # noqa: E402
    ExpertAbstractPlannerFeedback,
    ExpertAbstractTerminalValueEvaluator,
    LearnedStockFlowTransitionProvider,
    make_planner_action_priors,
    run_episode,
)
from sokoban_learned_action_prior import (  # noqa: E402
    ACTION_PRIOR_FEATURE_NAMES,
    SokobanActionPriorNet,
    action_prior_feature_vector,
    feature_summary,
)
from src.sokoban import SokobanGoalDirectedSDModel, load_boxoban_levels  # noqa: E402
from train_sokoban_tva import batch_iter, provider_level_split  # noqa: E402
from validate_sokoban_sd_design import stock_proxy  # noqa: E402


@dataclass
class PriorSample:
    features: np.ndarray
    label: float
    weight: float
    split: str
    level_id: str
    episode_solved: bool
    failure_type: str
    step: int
    flat_index: int
    source: str


def _write_failed_summary(args: argparse.Namespace, reason: str, details: Dict[str, Any]) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "FAILED_GATE",
            "version_label": "sokoban_learned_action_prior_train_v1",
        },
        "failure_reason": reason,
        "details": details,
        "prior_contract": {
            "prior_id": args.value_provider_id,
            "checkpoint_saved": False,
            "transition_provider_modified": False,
            "planner_consumption": "action_prior_multiplier_only",
        },
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-checkpoint",
        default="results/sokoban_sd_world_model/stage2_provider_checkpoint_seed0.pt",
    )
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--max-train-levels", type=int, default=16)
    parser.add_argument("--holdout-train-fraction", type=float, default=0.25)
    parser.add_argument("--collection-budget", type=int, default=24)
    parser.add_argument("--mcts-depth", type=int, default=4)
    parser.add_argument("--puct-c", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-pushes", type=int, default=40)
    parser.add_argument("--max-state-revisits", type=int, default=5)
    parser.add_argument("--max-policy-negatives-per-root", type=int, default=2)
    parser.add_argument("--positive-sample-weight", type=float, default=2.0)
    parser.add_argument("--negative-sample-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-provider-id", default="sokoban_learned_action_prior_v1")
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/sokoban_action_prior_train_seed0.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/sokoban_sd_world_model/sokoban_action_prior_seed0.pt",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = 0.0
    total = 0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        total += len(neg)
    return float(wins / max(total, 1))


def _metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {}
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = probs >= 0.5
    return {
        "accuracy": float(np.mean(pred == (labels > 0.5))),
        "auc": _safe_auc(labels, probs),
        "label_mean": float(np.mean(labels)),
        "prob_mean": float(np.mean(probs)),
        "pos_prob_mean": float(np.mean(probs[labels > 0.5])) if np.any(labels > 0.5) else 0.0,
        "neg_prob_mean": float(np.mean(probs[labels <= 0.5])) if np.any(labels <= 0.5) else 0.0,
    }


def _selected_flat(trace_row: dict[str, Any]) -> int | None:
    action = trace_row.get("action") or {}
    flat = action.get("flat_index")
    return int(flat) if flat is not None else None


def _policy_flat(policy_row: dict[str, Any]) -> int | None:
    action = policy_row.get("action") or {}
    first = action.get("first_action") if isinstance(action, dict) else None
    if not first:
        return None
    flat = first.get("flat_index")
    return int(flat) if flat is not None else None


def _target2_score_by_flat(trace_row: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for policy_row in trace_row.get("root_policy", []):
        flat = _policy_flat(policy_row)
        if flat is None:
            continue
        action = policy_row.get("action") or {}
        out[flat] = float(action.get("structural_score", 0.0))
    return out


def _state_after_trace_prefix(model: SokobanGoalDirectedSDModel, trace: Sequence[dict[str, Any]], step: int):
    state = model.initial_state
    for row in trace[:step]:
        flat = _selected_flat(row)
        if flat is None:
            break
        legal = {action.flat_index: action for action in model.legal_pushes(state)}
        action = legal.get(flat)
        if action is None:
            break
        state = model.step_state(state, action)
    return state


def _sample_for_flat(
    model: SokobanGoalDirectedSDModel,
    state,
    *,
    flat_index: int,
    label: float,
    weight: float,
    split: str,
    level_id: str,
    episode_solved: bool,
    failure_type: str,
    step: int,
    source: str,
    target2_scores: dict[int, float],
) -> PriorSample | None:
    legal = {action.flat_index: action for action in model.legal_pushes(state)}
    action = legal.get(int(flat_index))
    if action is None:
        return None
    stock = stock_proxy(model, state).as_array().astype(np.float32)
    return PriorSample(
        features=action_prior_feature_vector(
            model,
            state,
            stock,
            action,
            target2_structural_score=target2_scores.get(int(flat_index), 0.0),
        ),
        label=float(label),
        weight=float(weight),
        split=split,
        level_id=level_id,
        episode_solved=episode_solved,
        failure_type=failure_type,
        step=int(step),
        flat_index=int(flat_index),
        source=source,
    )


def collect_trace_samples(args: argparse.Namespace) -> tuple[list[PriorSample], Dict[str, Any]]:
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    split = provider_level_split(args.provider_checkpoint)
    train_level_ids = [level_id for level_id in split["train_levels"]]
    train_levels = [level for level in levels if level.level_id in train_level_ids]
    if args.max_train_levels > 0:
        train_levels = train_levels[: args.max_train_levels]
    if not train_levels:
        raise SystemExit("No train-split levels available for action-prior collection.")

    holdout_count = max(1, int(round(len(train_levels) * args.holdout_train_fraction))) if len(train_levels) > 3 else 1
    holdout_ids = {level.level_id for level in train_levels[-holdout_count:]}

    provider = LearnedStockFlowTransitionProvider(Path(args.provider_checkpoint))
    feedback = ExpertAbstractPlannerFeedback()
    terminal_value = ExpertAbstractTerminalValueEvaluator(feedback)
    samples: list[PriorSample] = []
    episode_rows: list[dict[str, Any]] = []
    t0 = perf_counter()

    for level in train_levels:
        model = SokobanGoalDirectedSDModel(level)
        rng = np.random.RandomState(args.seed)
        episode = run_episode(
            model,
            method="learned_sdmcts",
            provider=provider,
            terminal_value=terminal_value,
            planner_feedback=feedback,
            num_simulations=args.collection_budget,
            max_depth=args.mcts_depth,
            puct_c=args.puct_c,
            gamma=args.gamma,
            root_action_selection="value",
            max_pushes=args.max_pushes,
            max_state_revisits=args.max_state_revisits,
            revisit_feedback_penalty=2.0,
            action_abstraction_mode="target2_prior",
            max_option_length=2,
            max_options_per_node=4,
            option_prior_max_multiplier=2.0,
            option_execution_mode="first_push",
            route_consistency_weight=0.0,
            audit_proxy_prior_weight=0.0,
            rng=rng,
        )
        trace = episode.get("trace_sample", [])
        split_name = "valid" if level.level_id in holdout_ids else "train"
        episode_rows.append(
            {
                "level_id": level.level_id,
                "split": split_name,
                "solved": bool(episode["solved"]),
                "failure_type": episode["failure_type"],
                "pushes": int(episode["pushes"]),
                "trace_rows": len(trace),
            }
        )
        for step, trace_row in enumerate(trace):
            state = _state_after_trace_prefix(model, trace, step)
            selected = _selected_flat(trace_row)
            if selected is None:
                continue
            target2_scores = _target2_score_by_flat(trace_row)
            selected_label = 1.0 if episode["solved"] else 0.0
            selected_weight = args.positive_sample_weight if episode["solved"] else args.negative_sample_weight
            selected_sample = _sample_for_flat(
                model,
                state,
                flat_index=selected,
                label=selected_label,
                weight=selected_weight,
                split=split_name,
                level_id=level.level_id,
                episode_solved=bool(episode["solved"]),
                failure_type=episode["failure_type"],
                step=step,
                source="selected_action_solved_positive" if episode["solved"] else "selected_action_failed_negative",
                target2_scores=target2_scores,
            )
            if selected_sample is not None:
                samples.append(selected_sample)

            negatives = []
            for policy_row in sorted(
                trace_row.get("root_policy", []),
                key=lambda row: (float(row.get("visits", 0)), float(row.get("q_value", 0.0))),
                reverse=True,
            ):
                flat = _policy_flat(policy_row)
                if flat is None or flat == selected:
                    continue
                negatives.append(flat)
                if len(negatives) >= args.max_policy_negatives_per_root:
                    break
            for flat in negatives:
                sample = _sample_for_flat(
                    model,
                    state,
                    flat_index=flat,
                    label=0.0,
                    weight=args.negative_sample_weight,
                    split=split_name,
                    level_id=level.level_id,
                    episode_solved=bool(episode["solved"]),
                    failure_type=episode["failure_type"],
                    step=step,
                    source="root_policy_contrast_negative",
                    target2_scores=target2_scores,
                )
                if sample is not None:
                    samples.append(sample)

        print(
            json.dumps(
                {
                    "level_id": level.level_id,
                    "split": split_name,
                    "solved": bool(episode["solved"]),
                    "failure_type": episode["failure_type"],
                    "trace_rows": len(trace),
                },
                ensure_ascii=False,
            )
        )

    if not samples:
        _write_failed_summary(args, "no_samples", {"episodes": episode_rows})
        raise SystemExit("Action-prior collection produced no samples.")
    if not any(sample.label > 0.5 for sample in samples):
        _write_failed_summary(args, "no_positive_samples", {"episodes": episode_rows})
        raise SystemExit("Action-prior collection produced no positive samples.")

    train_positive = [sample for sample in samples if sample.split == "train" and sample.label > 0.5]
    if not train_positive:
        positive_levels = sorted({sample.level_id for sample in samples if sample.label > 0.5})
        promoted_level = positive_levels[0]
        demoted_level = next(
            (
                row["level_id"]
                for row in reversed(episode_rows)
                if row["split"] == "train" and not row["solved"]
            ),
            None,
        )
        for sample in samples:
            if sample.level_id == promoted_level:
                sample.split = "train"
            elif demoted_level is not None and sample.level_id == demoted_level:
                sample.split = "valid"
        for row in episode_rows:
            if row["level_id"] == promoted_level:
                row["split"] = "train_promoted_positive"
            elif demoted_level is not None and row["level_id"] == demoted_level:
                row["split"] = "valid_demoted_negative"

    train_labels = [sample.label for sample in samples if sample.split == "train"]
    if not any(label > 0.5 for label in train_labels) or not any(label <= 0.5 for label in train_labels):
        _write_failed_summary(
            args,
            "train_split_missing_label_class",
            {
                "episodes": episode_rows,
                "train_positive": int(sum(label > 0.5 for label in train_labels)),
                "train_negative": int(sum(label <= 0.5 for label in train_labels)),
            },
        )
        raise SystemExit("Action-prior train split does not contain both label classes.")
    return samples, {
        "level_split": split,
        "selected_train_levels": [level.level_id for level in train_levels],
        "internal_holdout_levels": sorted(holdout_ids),
        "episodes": episode_rows,
        "collection_elapsed_sec": float(perf_counter() - t0),
    }


def indices_for_split(samples: Sequence[PriorSample], split: str) -> np.ndarray:
    return np.array([idx for idx, sample in enumerate(samples) if sample.split == split], dtype=np.int64)


def evaluate_model(
    model: SokobanActionPriorNet,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
) -> Dict[str, float]:
    if len(indices) == 0:
        return {}
    idx = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        logits = model(features[idx]).detach().cpu().numpy()
    return _metrics(labels[idx].detach().cpu().numpy(), logits)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    samples, collection = collect_trace_samples(args)

    features_np = np.stack([sample.features for sample in samples]).astype(np.float32)
    labels_np = np.array([sample.label for sample in samples], dtype=np.float32)
    weights_np = np.array([sample.weight for sample in samples], dtype=np.float32)
    mean = features_np.mean(axis=0).astype(np.float32)
    std = np.maximum(features_np.std(axis=0).astype(np.float32), 1e-6)
    norm_features_np = (features_np - mean) / std

    train_idx = indices_for_split(samples, "train")
    valid_idx = indices_for_split(samples, "valid")
    if len(train_idx) == 0:
        raise SystemExit("No train samples after internal split.")
    if len(valid_idx) == 0:
        valid_idx = train_idx

    features = torch.tensor(norm_features_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32)
    weights = torch.tensor(weights_np, dtype=torch.float32)
    model = SokobanActionPriorNet(
        input_dim=len(ACTION_PRIOR_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batch_iter(train_idx, args.batch_size, args.seed, epoch):
            idx = torch.tensor(batch, dtype=torch.long)
            logits = model(features[idx])
            loss_vec = F.binary_cross_entropy_with_logits(logits, labels[idx], reduction="none")
            loss = (loss_vec * weights[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 50 == 0:
            model.eval()
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(np.mean(losses)) if losses else 0.0,
                    "train": evaluate_model(model, features, labels, train_idx),
                    "valid": evaluate_model(model, features, labels, valid_idx),
                }
            )

    model.eval()
    final_train = evaluate_model(model, features, labels, train_idx)
    final_valid = evaluate_model(model, features, labels, valid_idx)
    label_counts = Counter(
        f"{sample.split}:{'positive' if sample.label > 0.5 else 'negative'}" for sample in samples
    )
    source_counts = Counter(sample.source for sample in samples)
    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": "sokoban_learned_action_prior_train_v1",
        },
        "config": {
            "provider_checkpoint": os.path.relpath(args.provider_checkpoint, ROOT)
            if Path(args.provider_checkpoint).is_absolute()
            else args.provider_checkpoint,
            "levels": os.path.relpath(args.levels, ROOT) if Path(args.levels).is_absolute() else args.levels,
            "max_levels": args.max_levels,
            "max_train_levels": args.max_train_levels,
            "holdout_train_fraction": args.holdout_train_fraction,
            "collection_budget": args.collection_budget,
            "mcts_depth": args.mcts_depth,
            "puct_c": args.puct_c,
            "gamma": args.gamma,
            "max_pushes": args.max_pushes,
            "max_state_revisits": args.max_state_revisits,
            "max_policy_negatives_per_root": args.max_policy_negatives_per_root,
            "positive_sample_weight": args.positive_sample_weight,
            "negative_sample_weight": args.negative_sample_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "prior_contract": {
            "prior_id": args.value_provider_id,
            "target_source": "stage2_train_split_closed_loop_sdmcts_trace_labels",
            "uses_stage2_train_split_only": True,
            "uses_valid_split_for_training": False,
            "best_first_used_online": False,
            "teacher_used_online": False,
            "transition_provider_modified": False,
            "planner_consumption": "action_prior_multiplier_only",
        },
        "collection": collection,
        "dataset": {
            "samples": len(samples),
            "train_samples": int(len(train_idx)),
            "valid_samples": int(len(valid_idx)),
            "label_counts": dict(label_counts),
            "source_counts": dict(source_counts),
            "feature_dim": int(features_np.shape[1]),
            "feature_names": list(ACTION_PRIOR_FEATURE_NAMES),
            "feature_summary": feature_summary(features_np, ACTION_PRIOR_FEATURE_NAMES),
        },
        "metrics": {
            "train": final_train,
            "valid": final_valid,
            "history": history,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    checkpoint = {
        "checkpoint_contract": {
            "value_provider_id": args.value_provider_id,
            "prior_id": args.value_provider_id,
            "target_source": "stage2_train_split_closed_loop_sdmcts_trace_labels",
            "uses_stage2_train_split_only": True,
            "transition_provider_modified": False,
            "planner_consumption": "action_prior_multiplier_only",
        },
        "architecture": {
            "model_kind": "sokoban_action_prior_binary_mlp",
            "input_dim": len(ACTION_PRIOR_FEATURE_NAMES),
            "hidden_dim": args.hidden_dim,
        },
        "feature_names": list(ACTION_PRIOR_FEATURE_NAMES),
        "feature_stats": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "model_state_dict": model.state_dict(),
        "summary": summary,
    }
    checkpoint_out = Path(args.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_out)
    print(
        json.dumps(
            {
                "samples": summary["dataset"]["samples"],
                "label_counts": summary["dataset"]["label_counts"],
                "train": final_train,
                "valid": final_valid,
                "out": str(out),
                "checkpoint_out": str(checkpoint_out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
