#!/usr/bin/env python3
"""Probe whether GFT features identify censored hard leaf states.

The target is not a terminal value.  It is a censored-risk label: states solved
by the offline teacher are labelled low risk, while teacher-budget-unsolved,
proven-dead-end, and no-legal-action states are labelled high risk.  The probe
uses level-wise leave-one-out evaluation to avoid reporting an in-sample fit as
evidence for TVA readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sokoban_gft_sdmcts import gft_model_from_level  # noqa: E402
from evaluate_sokoban_grid_sdmcts import _rel  # noqa: E402
from src.sokoban import SokobanState, load_boxoban_levels  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-path",
        default=(
            "results/sokoban_sd_world_model/"
            "gft_tva_truncation_value_audit_direct_v3_gate10_h8_b192_teacher10k_seed0.json"
        ),
    )
    parser.add_argument("--ridge-l2", type=float, default=1e-2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/gft_tva_censored_risk_probe_direct_v3_teacher10k_seed0.json",
    )
    return parser


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_from_audit(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("rows"), list):
        return list(payload["rows"])
    rows = list(payload.get("sample_rows", []))
    expected = int(payload.get("metrics", {}).get("tva_prediction", {}).get("all", {}).get("samples", len(rows)))
    if len(rows) < expected:
        raise SystemExit(
            "Audit output does not contain all rows. Re-run the truncation audit after enabling full row export."
        )
    return rows


def state_from_row(row: dict[str, Any]) -> SokobanState:
    return SokobanState(
        player=tuple(int(v) for v in row["player"]),
        boxes=tuple(tuple(int(v) for v in box) for box in row["boxes"]),
    )


def selected_levels_from_audit(payload: dict[str, Any]):
    config = payload["config"]
    levels_path = config["levels"]
    level_offset = int(config.get("level_offset", 0))
    max_levels = int(config.get("max_levels", 0))
    levels = load_boxoban_levels(levels_path, max_levels=level_offset + max_levels)
    if level_offset:
        levels = levels[level_offset:]
    return levels


def feature_matrix(payload: dict[str, Any], rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    config = payload["config"]
    args = argparse.Namespace(**config)
    levels = selected_levels_from_audit(payload)
    model_by_level = {level.level_id: gft_model_from_level(level, args) for level in levels}
    if not rows:
        raise SystemExit("No rows available for censored-risk probe.")

    first_model = model_by_level[rows[0]["level_id"]]
    first_state = state_from_row(rows[0])
    base_feature_names = list(first_model.gft_tva_feature_dict(first_state)["feature_names"])
    value_feature_names = ["tva_prediction", "gft_sd_leaf_value", "sd_heuristic_value"]
    feature_names = [f"gft:{name}" for name in base_feature_names] + value_feature_names
    features = []
    for row in rows:
        model = model_by_level[row["level_id"]]
        state = state_from_row(row)
        feature_dict = model.gft_tva_feature_dict(state)["features"]
        values = [float(feature_dict[name]) for name in base_feature_names]
        values.extend(float(row[name]) for name in value_feature_names)
        features.append(values)
    return np.asarray(features, dtype=np.float64), feature_names


def labels_from_rows(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    labels = []
    classes = []
    for row in rows:
        failure_type = str(row["teacher_failure_type"])
        if failure_type == "solved":
            labels.append(0.0)
            classes.append("solved_low_risk")
        elif failure_type in {"teacher_budget_unsolved", "proven_dead_end", "no_legal_actions"}:
            labels.append(1.0)
            classes.append(f"{failure_type}_high_risk")
        else:
            raise ValueError(f"Unsupported teacher_failure_type for censored-risk probe: {failure_type}")
    return np.asarray(labels, dtype=np.float64), classes


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.maximum(np.std(x_train, axis=0), 1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def fit_ridge_binary_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float) -> np.ndarray:
    z_train, z_test = standardize_train_test(x_train, x_test)
    train_design = np.concatenate([z_train, np.ones((z_train.shape[0], 1), dtype=np.float64)], axis=1)
    test_design = np.concatenate([z_test, np.ones((z_test.shape[0], 1), dtype=np.float64)], axis=1)
    lhs = train_design.T @ train_design
    penalty = float(l2) * np.eye(lhs.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = train_design.T @ y_train
    weights = np.linalg.solve(lhs + penalty, rhs)
    return test_design @ weights


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels > 0.5
    n_pos = int(np.sum(pos))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[pos]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores >= float(threshold)
    truth = labels > 0.5
    tp = int(np.sum(pred & truth))
    tn = int(np.sum((~pred) & (~truth)))
    fp = int(np.sum(pred & (~truth)))
    fn = int(np.sum((~pred) & truth))
    pos = max(tp + fn, 1)
    neg = max(tn + fp, 1)
    clipped = np.clip(scores, 0.0, 1.0)
    return {
        "samples": float(labels.size),
        "auc": auc_score(labels, scores),
        "auc_orientation_free": max(auc_score(labels, scores), 1.0 - auc_score(labels, scores)),
        "accuracy": float(np.mean(pred == truth)) if labels.size else 0.0,
        "balanced_accuracy": float(0.5 * (tp / pos + tn / neg)),
        "risk_recall": float(tp / pos),
        "solved_recall": float(tn / neg),
        "brier_clipped": float(np.mean((clipped - labels) ** 2)) if labels.size else 0.0,
        "score_mean": float(np.mean(scores)) if labels.size else 0.0,
        "score_mean_risk": float(np.mean(scores[truth])) if np.any(truth) else 0.0,
        "score_mean_solved": float(np.mean(scores[~truth])) if np.any(~truth) else 0.0,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def leave_one_level_out_scores(features: np.ndarray, labels: np.ndarray, level_ids: Sequence[str], *, l2: float) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    fold_summaries = {}
    unique_levels = sorted(set(level_ids))
    for level_id in unique_levels:
        test_idx = np.asarray([idx for idx, item in enumerate(level_ids) if item == level_id], dtype=np.int64)
        train_idx = np.asarray([idx for idx, item in enumerate(level_ids) if item != level_id], dtype=np.int64)
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        if len(set(labels[train_idx].tolist())) < 2:
            scores[test_idx] = float(np.mean(labels[train_idx]))
        else:
            scores[test_idx] = fit_ridge_binary_scores(features[train_idx], labels[train_idx], features[test_idx], l2)
        fold_summaries[level_id] = {
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_positive": int(np.sum(labels[train_idx] > 0.5)),
            "test_positive": int(np.sum(labels[test_idx] > 0.5)),
        }
    return scores, fold_summaries


def single_feature_scores(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def main() -> None:
    args = build_parser().parse_args()
    payload = load_json(args.audit_path)
    rows = rows_from_audit(payload)
    features, feature_names = feature_matrix(payload, rows)
    labels, label_classes = labels_from_rows(rows)
    level_ids = [str(row["level_id"]) for row in rows]
    class_counts = Counter(label_classes)

    risk_scores, fold_summaries = leave_one_level_out_scores(
        features,
        labels,
        level_ids,
        l2=args.ridge_l2,
    )
    metrics = {
        "gft_feature_ridge_risk_lolo": binary_metrics(labels, risk_scores, threshold=args.threshold),
        "tva_prediction_as_risk_score": binary_metrics(labels, single_feature_scores(rows, "tva_prediction"), threshold=args.threshold),
        "negative_tva_prediction_as_risk_score": binary_metrics(labels, -single_feature_scores(rows, "tva_prediction"), threshold=args.threshold),
        "gft_sd_leaf_value_as_risk_score": binary_metrics(labels, single_feature_scores(rows, "gft_sd_leaf_value"), threshold=args.threshold),
        "negative_gft_sd_leaf_value_as_risk_score": binary_metrics(labels, -single_feature_scores(rows, "gft_sd_leaf_value"), threshold=args.threshold),
    }
    gate_failed = []
    main_metrics = metrics["gft_feature_ridge_risk_lolo"]
    if main_metrics["auc"] < 0.70:
        gate_failed.append("risk_auc_min_0p70")
    if main_metrics["balanced_accuracy"] < 0.65:
        gate_failed.append("balanced_accuracy_min_0p65")
    if main_metrics["risk_recall"] < 0.65:
        gate_failed.append("risk_recall_min_0p65")
    summary = {
        "material_passport": {
            "origin": "sokoban_gft_tva_censored_risk_probe",
            "verification_status": "ANALYZED",
            "version_label": "gft_tva_censored_risk_probe_v1",
        },
        "config": {
            "audit_path": _rel(args.audit_path),
            "ridge_l2": args.ridge_l2,
            "threshold": args.threshold,
            "out": _rel(args.out),
        },
        "probe_contract": {
            "target": "censored_hard_leaf_risk",
            "positive_label": ["teacher_budget_unsolved", "proven_dead_end", "no_legal_actions"],
            "negative_label": ["teacher_solved"],
            "not_a_terminal_value": True,
            "not_used_online": True,
            "split": "leave_one_level_out",
        },
        "dataset": {
            "samples": int(len(rows)),
            "class_counts": dict(class_counts),
            "feature_names": feature_names,
            "audit_source_gate": payload.get("gft_tva_independent_accuracy_gate", {}),
        },
        "fold_summaries": fold_summaries,
        "metrics": metrics,
        "censored_risk_probe_gate": {
            "verdict": "PASS" if not gate_failed else "FAIL",
            "failed_checks": gate_failed,
            "thresholds": {
                "risk_auc_min": 0.70,
                "balanced_accuracy_min": 0.65,
                "risk_recall_min": 0.65,
            },
            "observed": main_metrics,
        },
        "row_predictions": [
            {
                "sample_index": int(idx),
                "level_id": row["level_id"],
                "teacher_failure_type": row["teacher_failure_type"],
                "risk_label": float(labels[idx]),
                "risk_score_lolo": float(risk_scores[idx]),
                "tva_prediction": float(row["tva_prediction"]),
                "gft_sd_leaf_value": float(row["gft_sd_leaf_value"]),
            }
            for idx, row in enumerate(rows)
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "gate": summary["censored_risk_probe_gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
