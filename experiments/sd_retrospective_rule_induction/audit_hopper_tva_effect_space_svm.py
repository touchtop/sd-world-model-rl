#!/usr/bin/env python3
"""Audit learned selectors in TVA effect space for Hopper entry rules.

This is a read-only diagnostic.  It asks whether simple parameterized models
can use TVA-shaped effect features to separate entry-success actions from
matched failures, and whether the learned score induces stable
context-action-family candidates on held-out cycles.

The online rule form remains simple:

    context_key -> prefer_action_family

Linear models are treated as directly interpretable selectors.  RBF SVM is
treated only as a nonlinear separability probe, not as an online policy object.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import mean_or_zero, write_json  # noqa: E402
from audit_hopper_ability_entry_effect_space import DEFAULT_RESULTS_ROOT, DEFAULT_SEGMENTS  # noqa: E402
from audit_hopper_ability_entry_effect_space_stability import auc_positive_greater  # noqa: E402
from audit_hopper_phi_regret_potential import collect_records, entropy_norm, parse_horizons  # noqa: E402


DEFAULT_OUTPUT = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_tva_effect_space_svm_audit.json"
)


def family_feature_names(horizons: list[int]) -> list[str]:
    names = ["current_phi"]
    for h in horizons:
        names.extend([
            f"u_h{h}",
            f"adv_h{h}",
            f"neg_regret_h{h}",
            f"delta_phi_h{h}",
        ])
    return names


def actual_features(record: dict[str, Any], horizons: list[int]) -> list[float]:
    current_phi = float(record["current_phi"])
    feat = [current_phi]
    for h in horizons:
        item = record["phi_regret"][str(h)]
        actual_u = float(item["actual_u"])
        feat.extend([
            actual_u,
            float(item["actual_adv"]),
            -float(item["actual_regret"]),
            actual_u - current_phi,
        ])
    return feat


def family_features(record: dict[str, Any], family: str, horizons: list[int]) -> list[float] | None:
    current_phi = float(record["current_phi"])
    feat = [current_phi]
    for h in horizons:
        item = record["phi_regret"][str(h)]
        by_family = {row["family"]: row for row in item["family_values"]}
        row = by_family.get(family)
        if row is None:
            return None
        u = float(row["u"])
        feat.extend([
            u,
            float(row["adv"]),
            -float(row["regret"]),
            u - current_phi,
        ])
    return feat


def candidate_families(record: dict[str, Any], horizons: list[int]) -> list[str]:
    families = set()
    for h in horizons:
        for row in record["phi_regret"][str(h)]["family_values"]:
            family = str(row["family"])
            if family != "zero":
                families.add(family)
    return sorted(families)


def make_estimator(method: str, c_value: float, gamma: str) -> Pipeline:
    if method == "logistic_l2":
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", class_weight="balanced", random_state=20260523)
    elif method == "linear_svm":
        model = LinearSVC(C=c_value, class_weight="balanced", random_state=20260523, max_iter=10000)
    elif method == "rbf_svm":
        model = SVC(C=c_value, kernel="rbf", gamma=gamma, class_weight="balanced", random_state=20260523)
    else:
        raise ValueError("unknown method: " + method)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def decision_scores(estimator: Pipeline, x: np.ndarray) -> np.ndarray:
    model = estimator.named_steps["model"]
    if hasattr(model, "decision_function"):
        return np.asarray(estimator.decision_function(x), dtype=np.float64).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=np.float64).reshape(-1)
    return np.asarray(estimator.predict(x), dtype=np.float64).reshape(-1)


def label_of(record: dict[str, Any]) -> int:
    return 1 if record["role"] == "entry_success" else 0


def can_train(records: list[dict[str, Any]]) -> bool:
    labels = {label_of(record) for record in records}
    return len(records) >= 8 and labels == {0, 1}


def summarize_scores(scores: list[float], labels: list[int]) -> dict[str, Any]:
    pos = [score for score, label in zip(scores, labels) if label == 1]
    neg = [score for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return {
            "n": len(scores),
            "positive": len(pos),
            "negative": len(neg),
            "auc_success_gt_failure": 0.0,
            "score_margin": 0.0,
            "accuracy_at_zero": 0.0,
        }
    return {
        "n": len(scores),
        "positive": len(pos),
        "negative": len(neg),
        "auc_success_gt_failure": auc_positive_greater(pos, neg),
        "score_margin": mean_or_zero(pos) - mean_or_zero(neg),
        "accuracy_at_zero": mean_or_zero(
            1.0 if (score >= 0.0 and label == 1) or (score < 0.0 and label == 0) else 0.0
            for score, label in zip(scores, labels)
        ),
    }


def fit_and_score(
    method: str,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    horizons: list[int],
    c_value: float,
    gamma: str,
) -> tuple[dict[str, Any] | None, Pipeline | None]:
    if not can_train(train_records) or not test_records:
        return None, None
    x_train = np.asarray([actual_features(record, horizons) for record in train_records], dtype=np.float64)
    y_train = np.asarray([label_of(record) for record in train_records], dtype=np.int64)
    x_test = np.asarray([actual_features(record, horizons) for record in test_records], dtype=np.float64)
    y_test = [label_of(record) for record in test_records]
    estimator = make_estimator(method, c_value, gamma)
    estimator.fit(x_train, y_train)
    scores = [float(x) for x in decision_scores(estimator, x_test).tolist()]
    return summarize_scores(scores, y_test), estimator


def top_family_for_record(estimator: Pipeline, record: dict[str, Any], horizons: list[int]) -> dict[str, Any] | None:
    rows = []
    for family in candidate_families(record, horizons):
        feat = family_features(record, family, horizons)
        if feat is None:
            continue
        score = float(decision_scores(estimator, np.asarray([feat], dtype=np.float64))[0])
        rows.append({"family": family, "score": score})
    if not rows:
        return None
    rows.sort(key=lambda item: item["score"], reverse=True)
    return {
        "best_family": rows[0]["family"],
        "best_score": rows[0]["score"],
        "second_score": rows[1]["score"] if len(rows) > 1 else rows[0]["score"],
        "margin_to_second": rows[0]["score"] - (rows[1]["score"] if len(rows) > 1 else rows[0]["score"]),
        "top_families": rows[:5],
    }


def summarize_candidate_contexts(
    candidate_rows: list[dict[str, Any]],
    min_context_records: int,
    min_context_cycles: int,
    max_entropy: float,
    min_best_rate: float,
    min_cycle_rate: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[f"{row['tier']}|{row['context_key']}"].append(row)
    contexts = []
    for key, vals in sorted(groups.items()):
        if len(vals) < min_context_records:
            continue
        cycles = sorted({int(row["cycle"]) for row in vals})
        if len(cycles) < min_context_cycles:
            continue
        counts = Counter(str(row["best_family"]) for row in vals)
        family, count = counts.most_common(1)[0]
        cycle_rates = []
        for cycle in cycles:
            cvals = [row for row in vals if int(row["cycle"]) == cycle]
            cycle_rates.append(mean_or_zero(1.0 if row["best_family"] == family else 0.0 for row in cvals))
        same_rows = [row for row in vals if row["target_family"] == family]
        other_rows = [row for row in vals if row["target_family"] != family]
        item = {
            "key": key,
            "tier": key.split("|", 1)[0],
            "context_key": key.split("|", 1)[1],
            "count": len(vals),
            "cycles": cycles,
            "top_family": family,
            "top_family_rate": float(count) / float(len(vals)),
            "cycle_best_rate_min": min(cycle_rates) if cycle_rates else 0.0,
            "cycle_best_rate_mean": mean_or_zero(cycle_rates),
            "best_family_entropy": entropy_norm(counts),
            "mean_margin_to_second": mean_or_zero(row["margin_to_second"] for row in vals if row["best_family"] == family),
            "observed_top_family_count": len(same_rows),
            "observed_top_minus_other_future_reward": mean_or_zero(row["future_reward_sum"] for row in same_rows) - mean_or_zero(row["future_reward_sum"] for row in other_rows),
            "best_family_counts": dict(counts.most_common()),
        }
        contexts.append(item)
    stable = [
        item for item in contexts
        if item["top_family_rate"] >= min_best_rate
        and item["cycle_best_rate_min"] >= min_cycle_rate
        and item["best_family_entropy"] <= max_entropy
    ]
    return {
        "num_contexts": len(contexts),
        "num_stable_like_contexts": len(stable),
        "stable_like_contexts": sorted(stable, key=lambda item: (item["top_family_rate"], -item["best_family_entropy"], item["count"]), reverse=True)[:50],
        "top_contexts": sorted(contexts, key=lambda item: (item["top_family_rate"], -item["best_family_entropy"], item["count"]), reverse=True)[:50],
    }


def leave_one_cycle_audit(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    cycles = sorted({int(record["anchor_cycle"]) for record in records})
    methods = args.methods
    reports: dict[str, Any] = {}
    for method in methods:
        rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for tier in ["low_entry", "high_entry"]:
            tier_records = [record for record in records if record["tier"] == tier]
            tier_cycles = sorted({int(record["anchor_cycle"]) for record in tier_records})
            for cycle in tier_cycles:
                train = [record for record in tier_records if int(record["anchor_cycle"]) != cycle]
                test = [record for record in tier_records if int(record["anchor_cycle"]) == cycle]
                scored, estimator = fit_and_score(method, train, test, args.horizons, args.c_value, args.gamma)
                if scored is None or estimator is None:
                    continue
                row = {"method": method, "scope": "tier_leave_one", "tier": tier, "target_cycle": cycle, **scored}
                rows.append(row)
                for record in test:
                    best = top_family_for_record(estimator, record, args.horizons)
                    if best is None:
                        continue
                    candidate_rows.append(
                        {
                            "method": method,
                            "tier": tier,
                            "cycle": int(record["anchor_cycle"]),
                            "context_key": record["context_key"],
                            "role": record["role"],
                            "target_family": record["phi_regret"][str(args.horizons[0])]["actual_primary_family"],
                            "future_reward_sum": float(record["future_reward_sum"]),
                            **best,
                        }
                    )
        # Global leave-one-cycle is useful as a nonlinear separability upper bound.
        for cycle in cycles:
            train = [record for record in records if int(record["anchor_cycle"]) != cycle]
            test = [record for record in records if int(record["anchor_cycle"]) == cycle]
            scored, _ = fit_and_score(method, train, test, args.horizons, args.c_value, args.gamma)
            if scored is not None:
                rows.append({"method": method, "scope": "global_leave_one", "tier": "all", "target_cycle": cycle, **scored})
        reports[method] = {
            "rows": rows,
            "summary": summarize_rows(rows),
            "tier_summary": {
                tier: summarize_rows([row for row in rows if row["scope"] == "tier_leave_one" and row["tier"] == tier])
                for tier in ["low_entry", "high_entry"]
            },
            "global_summary": summarize_rows([row for row in rows if row["scope"] == "global_leave_one"]),
            "candidate_contexts": summarize_candidate_contexts(
                candidate_rows,
                args.min_context_records,
                args.min_context_cycles,
                args.max_best_family_entropy,
                args.min_best_family_rate,
                args.min_cycle_best_rate,
            ),
        }
    return reports


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_auc_success_gt_failure": mean_or_zero(row["auc_success_gt_failure"] for row in rows),
        "auc_ge_060_rate": mean_or_zero(1.0 if row["auc_success_gt_failure"] >= 0.60 else 0.0 for row in rows),
        "positive_margin_rate": mean_or_zero(1.0 if row["score_margin"] > 0.0 else 0.0 for row in rows),
        "mean_score_margin": mean_or_zero(row["score_margin"] for row in rows),
        "mean_accuracy_at_zero": mean_or_zero(row["accuracy_at_zero"] for row in rows),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper TVA Effect-Space SVM Audit",
        "",
        f"Records: `{payload['num_records']}`",
        f"Skipped: `{payload['num_skipped']}`",
        f"Horizons: `{','.join(str(x) for x in payload['config']['horizons'])}`",
        "",
        "## Leave-One-Cycle Summary",
        "",
        "| Method | Scope | Count | Mean AUC | AUC>=0.60 | Positive margin | Mean margin | Stable-like contexts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, report in payload["models"].items():
        for label, summary in [
            ("global", report["global_summary"]),
            ("low_entry", report["tier_summary"]["low_entry"]),
            ("high_entry", report["tier_summary"]["high_entry"]),
        ]:
            contexts = report["candidate_contexts"]["num_stable_like_contexts"]
            lines.append(
                f"| `{method}` | `{label}` | {summary['count']} | "
                f"{summary['mean_auc_success_gt_failure']:.3f} | {summary['auc_ge_060_rate']:.3f} | "
                f"{summary['positive_margin_rate']:.3f} | {summary['mean_score_margin']:.3f} | {contexts} |"
            )
    lines.extend([
        "",
        "## Stable-Like Candidate Contexts",
        "",
        "| Method | Context | Family | Count | Cycles | Best rate | Min cycle rate | Entropy | Observed reward margin |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for method, report in payload["models"].items():
        for item in report["candidate_contexts"]["stable_like_contexts"][:20]:
            lines.append(
                f"| `{method}` | `{item['context_key']}` | `{item['top_family']}` | {item['count']} | "
                f"`{','.join(str(c) for c in item['cycles'])}` | "
                f"{item['top_family_rate']:.3f} | {item['cycle_best_rate_min']:.3f} | "
                f"{item['best_family_entropy']:.3f} | {item['observed_top_minus_other_future_reward']:.3f} |"
            )
    lines.extend([
        "",
        "## Top Candidate Contexts",
        "",
        "| Method | Context | Family | Count | Cycles | Best rate | Min cycle rate | Entropy | Observed reward margin |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for method, report in payload["models"].items():
        for item in report["candidate_contexts"]["top_contexts"][:12]:
            lines.append(
                f"| `{method}` | `{item['context_key']}` | `{item['top_family']}` | {item['count']} | "
                f"`{','.join(str(c) for c in item['cycles'])}` | "
                f"{item['top_family_rate']:.3f} | {item['cycle_best_rate_min']:.3f} | "
                f"{item['best_family_entropy']:.3f} | {item['observed_top_minus_other_future_reward']:.3f} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "线性模型若在 leave-one-cycle 中稳定超过随机水平，说明 TVA 效果空间可通过参数训练形成规则筛选器。RBF SVM 只作为非线性可分性的上限诊断；即使 RBF 有效，也不直接进入在线 planner，而应先蒸馏回简单的 context-action 规则候选。",
    ])
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=parse_horizons, default=[1, 3, 5])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--methods", nargs="+", default=["logistic_l2", "linear_svm", "rbf_svm"])
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--min-context-records", type=int, default=12)
    parser.add_argument("--min-context-cycles", type=int, default=2)
    parser.add_argument("--max-best-family-entropy", type=float, default=0.65)
    parser.add_argument("--min-best-family-rate", type=float, default=0.40)
    parser.add_argument("--min-cycle-best-rate", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, skipped = collect_records(args)
    models = leave_one_cycle_audit(records, args)
    payload = {
        "analysis_family": "hopper_tva_effect_space_svm_v1",
        "config": {
            "segments": str(args.segments),
            "results_root": str(args.results_root),
            "horizons": args.horizons,
            "feature_names": family_feature_names(args.horizons),
            "amplitudes": args.amplitudes,
            "methods": args.methods,
            "c_value": float(args.c_value),
            "gamma": args.gamma,
            "max_windows": int(args.max_windows),
            "min_context_records": int(args.min_context_records),
            "min_context_cycles": int(args.min_context_cycles),
            "max_best_family_entropy": float(args.max_best_family_entropy),
            "min_best_family_rate": float(args.min_best_family_rate),
            "min_cycle_best_rate": float(args.min_cycle_best_rate),
        },
        "num_records": len(records),
        "num_skipped": len(skipped),
        "skipped": skipped[:200],
        "models": models,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload)
    print(f"wrote TVA effect-space SVM audit: {args.output}")
    print(f"records: {len(records)} skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
